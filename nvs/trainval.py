import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from einops import rearrange
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nvs.dataset import TrainDataset, EvalDataset, _normalize_poses_identity_unit_distance
from nvs.objaverse_dataset_gpt import ObjaverseTrainDataset, ObjaverseEvalDataset
from nvs.co3d_dataset import Co3dTrainDataset, Co3dEvalDataset
from nvs.lvsm import (
    Camera,
    LVSMDecoderOnlyModel,
    LVSMDecoderOnlyModelConfig,
)
from nvs.perceptual import Perceptual
from pos_enc.utils.functional import random_SO3
from pos_enc.utils.runner import Launcher, LauncherConfig, nested_to_device
from pos_enc.timing_utils import time_block, get_timing_stats

# Get machine-dependent paths from environment variables
TRAIN_DATA_DIR = os.environ["PROPE_TRAIN_DATA_DIR"]
TEST_DATA_DIR = os.environ["PROPE_TEST_DATA_DIR"]

def write_tensor_to_image(
    x: Tensor,
    path: str,
    downscale: int = 1,
    sqrt: bool = False,
    point: Tuple[int, int] = None,
):
    # x: [H, W, 1 or 3] in (0, 1)
    assert x.ndim == 3, x.shape
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)
    if sqrt:
        # reshape image to square
        h, w = x.shape[:2]
        if h > w:
            n_images = h // w
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "(n1 n2 h) w c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
        elif h < w:
            n_images = w // h
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "h (n1 n2 w) c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = (x * 255).to(torch.uint8).detach().cpu().numpy()
    if downscale > 1:
        image = cv2.resize(image, (0, 0), fx=1.0 / downscale, fy=1.0 / downscale)
    if point is not None:
        cv2.circle(image, point, 5, (255, 0, 0), -1)
    imageio.imsave(path, image)


@dataclass
class LVSMLauncherConfig(LauncherConfig):

    # TEMP: test with another checkpoint
    overwrite_ckpt_dir: str = None

    # Dataset config
    dataset: str = "re10k"  # "re10k" or "objaverse" or "co3d"
    dataset_patch_size: int = 256
    dataset_supervise_views: int = 6
    dataset_batch_scenes: int = 4
    train_zoom_factor: float = 1.0
    train_random_zoom: bool = False
    
    # Objaverse-specific config
    objaverse_train_index_file: str = "assets/objaverse_index_train_context2.json"
    objaverse_test_index_file: str = "assets/objaverse_index_test_context2_all.json"
    objaverse_test_radial_index_file: str = "assets/objaverse_index_test_context2_radial.json"
    objaverse_test_spherical_index_file: str = "assets/objaverse_index_test_context2_spherical.json"

    # co3d specific config
    co3d_test_seen_index_file: str = "assets/co3d_test_context2_seen.json"
    co3d_test_unseen_index_file: str = "assets/co3d_test_context2_unseen.json"
    co3d_test_full_index_file: str = "assets/co3d_test_context2_full.json"

    co3d_train_categories: tuple[str, ...] = ("seen",)
    co3d_test_unseen: bool = False

    # Optimization config
    use_torch_compile: bool = False

    # Model config
    model_config: Any = field(
        default_factory=lambda: LVSMDecoderOnlyModelConfig(ref_views=2)
    )

    # Training config
    max_steps: int = 100_000  # override
    ckpt_every: int = 1000  # override
    print_every: int = 100
    visual_every: int = 100
    lr: float = 4e-4
    warmup_steps: int = 2500

    # perceptual loss weight.
    perceptual_loss_w: float = 0.5
    bg_loss_w: float = 1.0
    get_mask: bool = False

    # How many test scenes to run.
    test_every: int = 10000  # override
    test_n: Optional[int] = None
    test_input_views: int = 2
    test_supervise_views: int = 3
    test_zoom_factor: tuple[float, ...] = (1.0,)
    test_random_zoom: bool = False
    test_rad_sph: bool = False
    aug_with_world_origin_shift: bool = False
    aug_with_world_rotation: bool = False

    # Render a video
    render_video: bool = False
    render_view: bool = False

    # test index file
    test_index_fp: Optional[str] = None


class LVSMLauncher(Launcher):
    config: LVSMLauncherConfig

    # Data preprocessing.
    def preprocess(
        self, data: Dict, input_views: int
    ) -> Tuple[Tensor, Camera, Camera, Tensor]:
        data = nested_to_device(data, self.device)

        images = data["image"] / 255.0
        Ks = data["K"]
        camtoworlds = data["camtoworld"]
        image_paths = data["image_path"]
        assert images.ndim == 5, images.shape
        n_batch, n_views, height, width, _ = images.shape

        # random shift and rotate.
        aug = torch.eye(4, device=self.device).repeat(n_batch, 1, 1)
        if self.config.aug_with_world_origin_shift:
            shifts = torch.randn((n_batch, 3), device=self.device)
            aug[:, :3, 3] = shifts
        if self.config.aug_with_world_rotation:
            rotations = random_SO3((n_batch,), device=self.device)
            aug[:, :3, :3] = rotations
        camtoworlds = torch.einsum("bij,bvjk->bvik", aug, camtoworlds)

        ref_imgs = images[:, :input_views]
        tar_imgs = images[:, input_views:]
        ref_cams = Camera(
            K=Ks[:, :input_views],
            camtoworld=camtoworlds[:, :input_views],
            width=width,
            height=height,
        )
        tar_cams = Camera(
            K=Ks[:, input_views:],
            camtoworld=camtoworlds[:, input_views:],
            width=width,
            height=height,
        )
        ref_paths = np.array(image_paths)[:input_views]
        tar_paths = np.array(image_paths)[input_views:]

        tar_masks = None
        if "mask" in data:
            mask = data["mask"]
            tar_masks = mask[:, input_views:]

        processed = {
            "ref_imgs": ref_imgs,
            "tar_imgs": tar_imgs,
            "ref_cams": ref_cams,
            "tar_cams": tar_cams,
            "ref_paths": ref_paths,
            "tar_paths": tar_paths,
            "context_depths": data.get('context_depths'),
            "tar_masks": tar_masks,
        }
        return processed

    def train_initialize(self) -> Dict[str, Any]:
        # ------------- Setup Data. ------------- #
        scenes = sorted(glob.glob(f"{TRAIN_DATA_DIR}/*"))
        
        if self.config.dataset == "re10k":
            dataset = TrainDataset(
                scenes,
                patch_size=self.config.dataset_patch_size,
                zoom_factor=self.config.train_zoom_factor,
                random_zoom=self.config.train_random_zoom,
                supervise_views=self.config.dataset_supervise_views,
            )
        elif self.config.dataset == "objaverse":
            index_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"../{self.config.objaverse_train_index_file}"
            )
            get_depth = ("known" in self.config.model_config.depth_type) \
                or self.config.model_config.depth_input
            dataset = ObjaverseTrainDataset(
                scenes=scenes,
                index_file=index_file,
                patch_size=self.config.dataset_patch_size,
                input_views=2,  # Fixed for LVSM model
                supervise_views=self.config.dataset_supervise_views,
                get_depth=get_depth,
                get_mask=self.config.get_mask,
            )
        elif self.config.dataset == "co3d":
            get_depth = ("known" in self.config.model_config.depth_type) \
                or self.config.model_config.depth_input
            dataset = Co3dTrainDataset(
                categories=self.config.co3d_train_categories,
                patch_size=self.config.dataset_patch_size,
                input_views=2,
                supervise_views=self.config.dataset_supervise_views,
                load_depth=get_depth,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
            
        # print(f"train zoom_factor: {self.config.train_zoom_factor}, random_zoom: {self.config.train_random_zoom}")
        num_workers = 2 if self.config.dataset in ["co3d", "re10k", "objaverse"] else 0
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataset_batch_scenes,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        # self.logging_on_master(f"Total scenes: {len(dataset)}")

        # ------------- Setup Model. ------------- #
        model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)
        # Apply torch.compile for performance optimization if enabled
        if self.config.use_torch_compile:
            model = torch.compile(model)
        perceptual = Perceptual().to(self.device)
        # print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Optimizer. ------------- #
        # Paper A.1 "We use a weight decay of 0.05 on all parameters except
        # the weights of LayerNorm layers."
        params_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" not in n],
            "weight_decay": 0.5,
        }
        params_no_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" in n],
            "weight_decay": 0.0,
        }
        optimizer = torch.optim.AdamW(
            [params_decay, params_no_decay], lr=self.config.lr, betas=(0.9, 0.95)
        )

        # ------------- Setup Scheduler. ------------- #
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=self.config.warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_steps - self.config.warmup_steps,
                ),
            ]
        )

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "perceptual": perceptual,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            "dataiter": iter(dataloader),
            "ssim_fn": ssim_fn,
            "psnr_fn": psnr_fn,
            "lpips_fn": lpips_fn,
        }
        # print(f"Launcher(train) is intialized in rank {self.world_rank}")
        return state

    def train_iteration(
        self, step: int, state: Dict[str, Any], acc_step: int, *args, **kwargs
    ) -> None:
        dataloader = state["dataloader"]
        dataiter = state["dataiter"]
        perceptual = state["perceptual"]
        model = state["model"]
        model.train()

        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)
            state["dataiter"] = dataiter

        input_views = data["K"].shape[1] - self.config.dataset_supervise_views
        processed = self.preprocess(data, input_views=input_views)
        ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
        ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]
        context_depths = processed.get("context_depths", None)
        tar_masks = processed.get("tar_masks", None)
        # print("camtoworld shape: ", ref_cams.camtoworld.shape, tar_cams.camtoworld.shape)
        # print("K shape: ", ref_cams.K.shape, tar_cams.K.shape)

        # Enable timing only for rank 0 to avoid overhead
        timing_enabled = (self.world_rank == 0)
        timing_enabled = False
        
        # Forward.
        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            with time_block("forward", timing_enabled):
                outputs = model(ref_imgs, ref_cams, tar_cams, 
                        context_depths=context_depths,
                        timing_enabled=timing_enabled)
                outputs = torch.sigmoid(outputs)

                if self.config.get_mask:
                    outputs_fg = outputs * tar_masks
                    tar_imgs_fg = tar_imgs * tar_masks
                    mse_fg = F.mse_loss(outputs_fg, tar_imgs_fg)
                    outputs_bg = outputs * (1 - tar_masks)
                    tar_imgs_bg = tar_imgs * (1 - tar_masks)
                    mse_bg = F.mse_loss(outputs_bg, tar_imgs_bg)
                    mse = mse_fg + self.config.bg_loss_w * mse_bg
                else:
                    mse = F.mse_loss(outputs, tar_imgs)

            if torch.isnan(outputs).any():
                # Compute max focal length across all cameras in batch
                ref_focal_x = ref_cams.K[:, :, 0, 0]  # [batch, views]
                ref_focal_y = ref_cams.K[:, :, 1, 1]  # [batch, views]
                tar_focal_x = tar_cams.K[:, :, 0, 0]  # [batch, views]
                tar_focal_y = tar_cams.K[:, :, 1, 1]  # [batch, views]
                
                max_focal = torch.stack([
                    ref_focal_x.max(), ref_focal_y.max(),
                    tar_focal_x.max(), tar_focal_y.max()
                ]).max().item()
                
                # Compute max translation norm across all cameras in batch
                ref_translations = ref_cams.camtoworld[:, :, :3, 3]  # [batch, views, 3]
                tar_translations = tar_cams.camtoworld[:, :, :3, 3]  # [batch, views, 3]
                
                ref_T_norms = torch.norm(ref_translations, dim=-1)  # [batch, views]
                tar_T_norms = torch.norm(tar_translations, dim=-1)  # [batch, views]
                
                max_T = torch.stack([ref_T_norms.max(), tar_T_norms.max()]).max().item()
                
                self.logging_on_master(f"NaN detected in model output at step {step}")
                self.logging_on_master(f"Max focal length: {max_focal:.3f}")
                self.logging_on_master(f"Max translation norm: {max_T:.3f}")

            if self.config.perceptual_loss_w > 0:
                perceptual_loss = perceptual(
                    rearrange(outputs, "b v h w c -> (b v) c h w"),
                    rearrange(tar_imgs, "b v h w c -> (b v) c h w"),
                )
                loss = mse + perceptual_loss * self.config.perceptual_loss_w
            else:
                loss = mse

        # Loggings.
        if (
            self.config.visual_every > 0
            and step % self.config.visual_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            write_tensor_to_image(
                rearrange(outputs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/outputs.png",
            )
            write_tensor_to_image(
                rearrange(tar_imgs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/gts.png",
            )
            write_tensor_to_image(
                rearrange(ref_imgs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/inputs.png",
            )

        # Get timing stats at the end of iteration (only for rank 0)
        if timing_enabled:
            timing_stats = get_timing_stats().end_iteration()
        
        if (
            step % self.config.print_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            mse = F.mse_loss(outputs, tar_imgs)
            outputs = rearrange(outputs, "b v h w c-> (b v) c h w")
            tar_imgs = rearrange(tar_imgs, "b v h w c-> (b v) c h w")
            psnr = state["psnr_fn"](outputs, tar_imgs)
            # ssim = state["ssim_fn"](outputs, tar_imgs)
            # lpips = state["lpips_fn"](outputs, tar_imgs)
            
            # Build timing info string
            timing_info = ""
            if timing_enabled and timing_stats:
                timing_parts = []
                for key in ['forward', 'attention_total', 'attention', 'apply_enc', 
                            'precompute_enc', 'preprocess', 'transformer']:
                    if key in timing_stats:
                        timing_parts.append(f"{key}: {timing_stats[key]*1000:.1f}ms")
                if timing_parts:
                    timing_info = f", Timing: {', '.join(timing_parts)}"
            
            self.logging_on_master(
                f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, "
                # f"SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                f"LR: {state['scheduler'].get_last_lr()[0]:.3e}{timing_info}"
            )
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/psnr", psnr, step)
            # self.writer.add_scalar("train/ssim", ssim, step)
            # self.writer.add_scalar("train/lpips", lpips, step)
            
            # Log timing stats to tensorboard
            if timing_enabled and timing_stats:
                for key, value in timing_stats.items():
                    self.writer.add_scalar(f"train_timing/{key}_ms", value * 1000, step)
        return loss

    def test_initialize(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        
        # ----- overwrite ckpt to load ----- #
        if self.config.overwrite_ckpt_dir is not None:
            self.ckpt_dir = self.config.overwrite_ckpt_dir
            self.logging_on_master(
                f"Overwriting checkpoint directory to {self.ckpt_dir} for testing."
            )
        
        # ------------- Setup Data. ------------- #
        dataset = None
        dataloaders = dict()
        folder = TEST_DATA_DIR
        print(f"test zoom_factor: {self.config.test_zoom_factor}, random_zoom: {self.config.test_random_zoom}")
        
        if self.config.dataset == "re10k":
            if not self.config.render_video and self.config.test_index_fp is None:
                assert (
                    self.config.test_input_views == 2
                    and self.config.test_supervise_views == 3
                ), "Invalid input views and supervise views for RE10K, should be 2 and 3 respectively."
            
            for zoom_factor in self.config.test_zoom_factor:
                dataset = EvalDataset(
                    folder=folder,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=zoom_factor,
                    random_zoom=self.config.test_random_zoom,
                    first_n=self.config.test_n,
                    rank=self.world_rank,
                    world_size=self.world_size,
                    input_views=self.config.test_input_views,
                    supervise_views=self.config.test_supervise_views,
                    render_video=self.config.render_video,
                    test_index_fp=self.config.test_index_fp,
                )
                dataloader_key = f"zoom{zoom_factor}_rand" if self.config.test_random_zoom else f"zoom{zoom_factor}"
                dataloaders[dataloader_key] = (
                    self.config.test_input_views,
                    torch.utils.data.DataLoader(
                        dataset, batch_size=1, num_workers=0, pin_memory=True
                    ),
                )
        elif self.config.dataset == "objaverse":
            # temporary overwrite
            # self.config.objaverse_test_index_file = f"assets/objaverse_index_test_context{self.config.test_input_views}_all.json"
            # self.config.objaverse_test_radial_index_file = f"assets/objaverse_index_test_context{self.config.test_input_views}_radial.json"
            # self.config.objaverse_test_spherical_index_file = f"assets/objaverse_index_test_context{self.config.test_input_views}_spherical.json"
            
            get_depth = ("known" in self.config.model_config.depth_type) \
                or self.config.model_config.depth_input
            # get_depth = True
            if self.config.test_rad_sph:
                # radial
                scenes = sorted(glob.glob(f"{folder}/*"))
                index_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    f"../{self.config.objaverse_test_radial_index_file}"
                )
                dataset_radial = ObjaverseEvalDataset(
                    scenes=scenes,
                    index_file=index_file,
                    patch_size=self.config.dataset_patch_size,
                    input_views=self.config.test_input_views,
                    supervise_views=self.config.test_supervise_views,
                    first_n=self.config.test_n,
                    rank=self.world_rank,
                    world_size=self.world_size,
                    get_depth=get_depth,
                )
                dataloaders["radial"] = (
                    self.config.test_input_views,
                    torch.utils.data.DataLoader(
                        dataset_radial, batch_size=1, num_workers=0, pin_memory=True
                    ),
                )
                # spherical
                index_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    f"../{self.config.objaverse_test_spherical_index_file}"
                )
                dataset_spherical = ObjaverseEvalDataset(
                    scenes=scenes,
                    index_file=index_file,
                    patch_size=self.config.dataset_patch_size,
                    input_views=self.config.test_input_views,
                    supervise_views=self.config.test_supervise_views,
                    first_n=self.config.test_n,
                    rank=self.world_rank,
                    world_size=self.world_size,
                    get_depth=get_depth,
                )
                dataloaders["spherical"] = (
                    self.config.test_input_views,
                    torch.utils.data.DataLoader(
                        dataset_spherical, batch_size=1, num_workers=0, pin_memory=True
                    ),
                )
            else:
                scenes = sorted(glob.glob(f"{folder}/*"))
                index_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    f"../{self.config.objaverse_test_index_file}"
                )
                dataset = ObjaverseEvalDataset(
                    scenes=scenes,
                    index_file=index_file,
                    patch_size=self.config.dataset_patch_size,
                    input_views=self.config.test_input_views,
                    supervise_views=self.config.test_supervise_views,
                    first_n=self.config.test_n,
                    rank=self.world_rank,
                    world_size=self.world_size,
                    get_depth=get_depth,
                    render_video=self.config.render_video,
                )
                dataloaders["objv"] = (
                    self.config.test_input_views,
                    torch.utils.data.DataLoader(
                        dataset, batch_size=1, num_workers=0, pin_memory=True
                    ),
                )
        elif self.config.dataset == "co3d":
            get_depth = ("known" in self.config.model_config.depth_type) \
                or self.config.model_config.depth_input
            # get_depth = True
            dataset_seen = Co3dEvalDataset(
                categories=self.config.co3d_train_categories,
                index_file=self.config.co3d_test_seen_index_file,
                patch_size=self.config.dataset_patch_size,
                input_views=self.config.test_input_views,
                first_n=self.config.test_n,
                load_depth=get_depth,
                render_video=self.config.render_video,
            )
            dataloaders["seen"] = (
                self.config.test_input_views,
                torch.utils.data.DataLoader(
                    dataset_seen, batch_size=1, num_workers=0, pin_memory=True
                ),
            )
            if self.config.co3d_test_unseen:
                dataset_unseen = Co3dEvalDataset(
                    categories="unseen",
                    index_file=self.config.co3d_test_unseen_index_file,
                    patch_size=self.config.dataset_patch_size,
                    input_views=self.config.test_input_views,
                    first_n=self.config.test_n,
                    load_depth=get_depth,
                )
                dataloaders["unseen"] = (
                    self.config.test_input_views,
                    torch.utils.data.DataLoader(
                        dataset_unseen, batch_size=1, num_workers=0, pin_memory=True
                    ),
                )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
                
        # ------------- Setup Model. ------------- #
        if model is None:
            model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)
            # Apply torch.compile for performance optimization if enabled
            if self.config.use_torch_compile:
                model = torch.compile(model)
            # print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "dataloaders": dataloaders,
            "psnr_fn": psnr_fn,
            "ssim_fn": ssim_fn,
            "lpips_fn": lpips_fn,
        }
        # print(f"Launcher(Test) is intialized in rank {self.world_rank}")
        return state

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> None:
        dataloaders = state["dataloaders"]
        model = state["model"]
        model.eval()

        # Enable timing only for rank 0 to avoid overhead
        timing_enabled = (self.world_rank == 0)
        timing_enabled = False

        for label, (input_views, dataloader) in dataloaders.items():
            psnrs, lpips, ssims = [], [], []
            canvas = []  # for visualization
            for idx, data in enumerate(dataloader):
                processed = self.preprocess(data, input_views=input_views)
                ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
                ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]
                ref_paths, tar_paths = processed["ref_paths"], processed["tar_paths"]

                # save tar cams
                # if self.world_rank == 0:
                #     torch.save(tar_cams, f"logs/debug_tensor/tar_cams.pt")
                #     exit()

                context_depths = processed.get("context_depths", None)
                # print(f"ref_imgs shape: {ref_imgs.shape}, tar_imgs shape: {tar_imgs.shape}")
                # print(f"ref_cams K shape: {ref_cams.K.shape}, tar_cams K shape: {tar_cams.K.shape}")
                # print(f"ref_cams camtoworld shape: {ref_cams.camtoworld.shape}, tar_cams camtoworld shape: {tar_cams.camtoworld.shape}")
                # print(f"context_depths shape: {context_depths.shape if context_depths is not None else None}")

                # Forward.
                with torch.amp.autocast(
                    "cuda", enabled=self.config.amp, dtype=self.amp_dtype
                ):
                    with time_block("test_forward", timing_enabled):
                        outputs = model(ref_imgs, ref_cams, tar_cams, 
                                        context_depths=context_depths,
                                        timing_enabled=timing_enabled)
                        outputs = torch.sigmoid(outputs)

                if self.config.render_video:
                    # assert outputs.shape[0] == 1
                    for b in range(outputs.shape[0]):
                        path_splits = tar_paths[b, 0].split("/")
                        scene_name = path_splits[-3]
                        # dump video using imageio
                        os.makedirs(f"{self.test_dir}/{scene_name}", exist_ok=True)
                        imageio.mimwrite(
                            f"{self.test_dir}/{scene_name}/pred.mp4",
                            (outputs[0].cpu().numpy() * 255).astype(np.uint8),
                            format="ffmpeg",
                            fps=15,
                        )

                        # save gt video
                        imageio.mimwrite(
                            f"{self.test_dir}/{scene_name}/gt.mp4",
                            (tar_imgs[0].cpu().numpy() * 255).astype(np.uint8),
                            format="ffmpeg",
                            fps=15,
                        )
                elif self.config.render_view:
                    for b in range(outputs.shape[0]):
                        # scene_idx = idx * outputs.shape[0] + b
                        path_splits = tar_paths[b, 0].split("/")
                        scene_name = path_splits[-3]
                        
                        os.makedirs(f"{self.test_dir}/{scene_name}", exist_ok=True)
                        for v in range(ref_imgs.shape[1]):
                            ref_img_v = ref_imgs[b, v]
                            write_tensor_to_image(ref_img_v, f"{self.test_dir}/{scene_name}/ref{v}.png")
                    
                        for v in range(tar_imgs.shape[1]):
                            tar_img_v = tar_imgs[b, v]
                            output_v = outputs[b, v]
                            write_tensor_to_image(tar_img_v, f"{self.test_dir}/{scene_name}/target{v}.png")
                            write_tensor_to_image(output_v, f"{self.test_dir}/{scene_name}/gen{v}.png")

                        # save the predicted depth visuals (per scene)
                        visualize_predict_d = True
                        if visualize_predict_d:
                            # Get stored depth maps from model.attention
                            predicted_depth_maps = model.attention.predicted_depth_maps
                            gt_depth_maps = model.attention.gt_depth_maps
                            
                            if predicted_depth_maps:
                                scene_dir = f"{self.test_dir}/{scene_name}"
                                
                                # Save all layers' depth/uncertainty maps
                                for layer_idx, layer_data in enumerate(predicted_depth_maps):
                                    for cam_idx, cam_data in layer_data.items():
                                        cam_dir = os.path.join(scene_dir, f"c{cam_idx}")
                                        os.makedirs(cam_dir, exist_ok=True)
                                        torch.save(cam_data['mean_d'], os.path.join(cam_dir, f"mean_d_l{layer_idx + 1}.pt"))
                                        torch.save(cam_data['sigma'], os.path.join(cam_dir, f"sigma_l{layer_idx + 1}.pt"))
                                
                                # Save GT depth maps if available
                                if gt_depth_maps is not None:
                                    for c in range(gt_depth_maps.shape[1]):
                                        gt_depth = gt_depth_maps[0, c].squeeze(-1).unsqueeze(0)
                                        cam_dir = os.path.join(scene_dir, f"c{c}")
                                        os.makedirs(cam_dir, exist_ok=True)
                                        torch.save(gt_depth, os.path.join(cam_dir, "gt_d.pt"))
                            
                            # Clear stored maps for next iteration
                            model.attention.predicted_depth_maps = []
                            model.attention.gt_depth_maps = None
                            model.attention.layer_counter = 1

                else:
                    # metrics.
                    outputs = rearrange(outputs, "b v h w c -> (b v) c h w")
                    tar_imgs = rearrange(tar_imgs, "b v h w c -> (b v) c h w")
                    psnrs.append(state["psnr_fn"](outputs, tar_imgs))
                    ssims.append(state["ssim_fn"](outputs, tar_imgs))
                    lpips.append(state["lpips_fn"](outputs, tar_imgs))

            if self.config.render_video or self.config.render_view:
                return

            # dump canvas.
            # canvas = torch.cat(canvas, dim=0)
            # write_tensor_to_image(
            #     canvas, f"{self.test_dir}/rank{self.world_rank}_{label}views.png"
            # )

            def distributed_avg(data: List[float], name: str) -> float:
                # collect metric from all ranks
                collected_sizes = [None] * self.world_size
                torch.distributed.all_gather_object(collected_sizes, len(data))
                collected = [
                    torch.empty(size, device=self.device) for size in collected_sizes
                ]
                torch.distributed.all_gather(
                    collected, torch.tensor(data, device=self.device)
                )
                collected = torch.cat(collected)

                if torch.isinf(collected).any():
                    self.logging_on_master(
                        f"Inf found in {label} views, {sum(torch.isinf(collected))} inf values for {name}."
                    )
                    collected = collected[~torch.isinf(collected)]
                if torch.isnan(collected).any():
                    self.logging_on_master(
                        f"NaN found in {label} views, {sum(torch.isnan(collected))} nan values for {name}."
                    )
                    collected = collected[~torch.isnan(collected)]

                avg = collected.mean().item()
                return avg, len(collected)

            avg_psnr, n_total = distributed_avg(psnrs, "psnr")
            avg_lpips, n_total = distributed_avg(lpips, "lpips")
            avg_ssim, n_total = distributed_avg(ssims, "ssim")

            # Get timing stats for testing (only for rank 0)
            timing_info = ""
            if timing_enabled:
                timing_stats = get_timing_stats().get_recent_stats(n=min(10, len(psnrs)))
                if timing_stats:
                    timing_parts = []
                    for key in ['forward', 'attention_total', 'attention', 'apply_enc']:
                        if key in timing_stats:
                            timing_parts.append(f"{key}: {timing_stats[key]*1000:.1f}ms")
                    if timing_parts:
                        timing_info = f", Avg Timing (last {min(10, len(psnrs))} samples): {', '.join(timing_parts)}"

            self.logging_on_master(
                f"PSNR{label}: {avg_psnr:.3f}, SSIM{label}: {avg_ssim:.3f}, LPIPS{label}: {avg_lpips:.3f} "
                f"evaluated on {n_total} scenes at step {step}.{timing_info}"
            )

            if self.world_rank == 0:
                self.writer.add_scalar(f"test/psnr{label}", avg_psnr, step)
                self.writer.add_scalar(f"test/ssim{label}", avg_ssim, step)
                self.writer.add_scalar(f"test/lpips{label}", avg_lpips, step)
                
                # Log test timing stats to tensorboard
                if timing_enabled and timing_stats:
                    for key, value in timing_stats.items():
                        self.writer.add_scalar(f"test_timing/{key}_ms", value * 1000, step)
                
                with open(f"{self.test_dir}/metrics.json", "w") as f:
                    json.dump(
                        {
                            "label": label,
                            "step": step,
                            "n_total": n_total,
                            "psnr": avg_psnr,
                            "ssim": avg_ssim,
                            "lpips": avg_lpips,
                        },
                        f,
                    )

            # save the predicted d analysis
            stat_predict_d = False
            stat_d_save_path = f"{self.test_dir}/predict_d_stats.npz"
            if stat_predict_d and self.world_rank == 0:
                mean_d_loss = model.attention.mean_d_loss
                abs_rel = model.attention.abs_rel
                sq_rel = model.attention.sq_rel
                rmse = model.attention.rmse
                rmse_log = model.attention.rmse_log
                mean_sigma = model.attention.mean_sigma

                np.savez(
                    stat_d_save_path,
                    mean_d_loss=np.array(mean_d_loss),
                    abs_rel=np.array(abs_rel),
                    sq_rel=np.array(sq_rel),
                    rmse=np.array(rmse),
                    rmse_log=np.array(rmse_log),
                    mean_sigma=np.array(mean_sigma),
                )
                self.logging_on_master(f"Saved predicted depth statistics to {stat_d_save_path}")
                


        
        # Clear timing stats after test iteration
        if timing_enabled:
            get_timing_stats().clear()


if __name__ == "__main__":
    """Example usage:

    # 2GPUs dry run
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        nvs/trainval.py lvsm-dry-run --model_config.encoder.num_layers 2
    """

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")

    configs = {
        "lvsm": (
            "feedforward large view synthesis model",
            LVSMLauncherConfig(),
        ),
        "lvsm-dry-run": (
            "dry run",
            LVSMLauncherConfig(
                amp=True,
                amp_dtype="fp16",
                dataset_batch_scenes=1,
                max_steps=10,
                test_every=5,
                test_n=10,
            ),
        ),
        "lvsm-objaverse-dry-run": (
            "dry run on Objaverse",
            LVSMLauncherConfig(
                dataset="objaverse",
                amp=True,
                amp_dtype="fp16",
                dataset_batch_scenes=1,
                max_steps=10,
                test_every=5,
                test_n=10,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    launcher = LVSMLauncher(cfg)
    launcher.run()

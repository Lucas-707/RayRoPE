"""
Implementation of https://arxiv.org/abs/2410.17242
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from pos_enc.xy_rope import xyRopeDotProductAttention
from pos_enc.ray_rope_release import RayRoPE_Release_DotProductAttention
from pos_enc.ray_rope_release_nosig import RayRoPE_NoSig_DotProductAttention
from pos_enc.ray_rope_global import Global_RayRoPE_DotProductAttention
from pos_enc.timing_utils import time_block

from pos_enc.torch_official import PropeDotProductAttention
from pos_enc.utils.functional import (
    Camera,
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
)
from pos_enc.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


@dataclass
class LVSMDecoderOnlyModelConfig:

    ref_views: int
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
                predict_d='none',
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: Tuple[int, ...] = (256, 256, 3)
    cam_shape: Tuple[int, ...] = (256, 256, 6)
    patch_size: int = 8

    # How the input rays are encoded.
    ray_encoding: Literal["plucker", "camray", "none", "raymap"] = "plucker"

    pos_enc: str = "prope"
    num_rays_per_patch: int = 3
    freq_base: float = 3.0
    p0_type: Literal["in3d", "ex3d", "in2d"] = "ex3d"
    disable_vo: bool = False
    depth_type: str = "none"
    init_d: float = 0.0
    init_sig: float = 3.0
    
    inf_qd: bool = False
    use_invd: bool = True
    denc_type: str = "d"  # "d" or "inv_d" or "asinh_d"
    depth_input: bool = False # concat context depth map to ref input
    cam_transform_type: Literal["LDU", "none"] = "none"
    rope_transform_type: Literal["none", "Cayley", "Cayley_randn"] = "none"
    
    # Timing configuration
    timing_enabled: bool = False


class LVSMDecoderOnlyModel(nn.Module):
    def __init__(self, config: LVSMDecoderOnlyModelConfig):
        super().__init__()
        self.config = config
        
        head_dim = config.encoder.layer.d_model // config.encoder.layer.nhead

        if self.config.pos_enc in ["global-0+inf", "global-0+d"]:
            self.attention = Global_RayRoPE_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                rope_transform_type=self.config.rope_transform_type,
                freq_base=self.config.freq_base,
            )
        elif ('d_pj' in self.config.pos_enc or \
            'd_3d' in self.config.pos_enc) and \
            'predict_dsig' in self.config.depth_type:
            self.attention = RayRoPE_Release_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                depth_type=self.config.depth_type,
                denc_type=self.config.denc_type,
                freq_base=self.config.freq_base,
                apply_vo=(not self.config.disable_vo),
            )
        elif ('d_pj' in self.config.pos_enc or \
            'd_3d' in self.config.pos_enc) and \
            'predict_d' in self.config.depth_type:
            self.attention = RayRoPE_NoSig_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                depth_type=self.config.depth_type,
                denc_type=self.config.denc_type,
                freq_base=self.config.freq_base,
                apply_vo=(not self.config.disable_vo),
            )
        elif self.config.pos_enc == "xy_rope":
            self.attention = xyRopeDotProductAttention(
                head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
            )
        elif self.config.pos_enc == "none":
            self.attention = None
        else:
            self.attention = PropeDotProductAttention(
                head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
                # cameras=config.ref_views + config.tar_views,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                # freq_type=self.config.cam_transform_type,
            )

        assert (
            config.cam_shape[:2] == config.img_shape[:2]
        ), f"{config.cam_shape[:2]} != {config.img_shape[:2]}"

        if config.ray_encoding == "none":
            shared_rays = torch.randn(config.cam_shape)
            self.shared_rays = nn.Parameter(shared_rays, requires_grad=False)

        # query tokenizer encodes tar_cam
        self.query_tokenizer = nn.Linear(
            config.cam_shape[-1] * config.patch_size**2,
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )
        # input tokenizer encodes ref_img and ref_cam
        self.input_tokenizer = nn.Linear(
            (
                config.img_shape[-1] * config.patch_size**2
                + config.cam_shape[-1] * config.patch_size**2
                + int(config.depth_input) * 1 * config.patch_size**2
            ),
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )

        if self.config.depth_type in ['predict_d', 'known+predict_d']:
            self.config.encoder.layer.predict_d = 'predict_d'
        elif self.config.depth_type in ['predict_dsig', 'known+predict_dsig']:
            self.config.encoder.layer.predict_d = 'predict_dsig'
        elif self.config.depth_type in ['dsig_perhead', 'known+dsig_perhead']:
            self.config.encoder.layer.predict_d = 'dsig_perhead'
        self.config.encoder.layer.init_depth = self.config.init_d
        self.config.encoder.layer.init_sigma = self.config.init_sig
        
        self.encoder = self.config.encoder.setup()

        self.output_layer = nn.Linear(
            config.encoder.layer.d_model,
            config.img_shape[-1] * config.patch_size**2,
            bias=config.encoder.layer.bias,
        )
        self.init_weights()

    def init_weights(self):
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self.init_layer_weights(idx))

    def init_layer_weights(self, idx):
        # LVMS Paper A.1:
        # "We initialize the model weights with a normal distribution of zero-mean
        # and standard deviation of 0.02/(2 * (idx+ 1)) ** 0.5, where idx means
        # transform layer index."
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)

        return _init_weights

    def create_rays(self, cams: Camera) -> Tensor:
        """Convert cameras to raymaps.

        Returns:
            rays: [B, V, H, W, C]
        """
        config = self.config
        batch_size, v = cams.camtoworld.shape[:2]
        cam_dtype = cams.camtoworld.dtype
        device = cams.camtoworld.device

        if config.ray_encoding == "none":
            rays = repeat(self.shared_rays, "h w c -> b v h w c", b=batch_size, v=v)
        else:
            # Preprocess cameras into rays.
            downscale = config.img_shape[0] // config.cam_shape[0]
            rays = camera_to_raymap(
                Ks=cams.K,
                camtoworlds=(
                    torch.eye(4, dtype=cam_dtype, device=device).broadcast_to(
                        cams.camtoworld.shape
                    )
                    if config.ray_encoding == "camray"
                    else cams.camtoworld
                ),
                height=cams.height,
                width=cams.width,
                downscale=downscale,
            )
            if config.ray_encoding in ["plucker", "camray"]:
                rays = raymap_to_plucker(rays)
            else:
                assert config.ray_encoding == "raymap"
        return rays

    def forward(
        self,
        ref_imgs: Tensor,
        ref_cams: Camera,
        tar_cams: Camera,
        context_depths: Optional[Tensor] = None,
        timing_enabled: bool = False,
    ) -> Tensor:
        
        with time_block("preprocess", enabled=timing_enabled):
            # ref_imgs: [B, V1, H, W, C]
            # tar_imgs: [B, V2, H, W, C]
            batch_size, v2 = tar_cams.camtoworld.shape[:2]
            config = self.config

            # Create rays.
            # ref_rays: [B, V1, H, W, C]
            # tar_rays: [B, V2, H, W, C]
            ref_rays = self.create_rays(ref_cams)
            tar_rays = self.create_rays(tar_cams)

            # print(f"before patchify:")
            # print(f"ref_imgs shape: {ref_imgs.shape}, ref_rays shape: {ref_rays.shape}, context_depths_patch shape: {context_depths.shape}")
            # ref_imgs: [B, V1, N1, DIM1]
            ref_imgs = patchify(ref_imgs, config.patch_size)
            # ref_rays: [B, V1, N2, DIM2]
            ref_rays = patchify(ref_rays, config.patch_size)
            # tar_rays: [B, V2, N2, DIM2]
            tar_rays = patchify(tar_rays, config.patch_size)

            # context_depths: [B, V1, N1, 1]
            if self.config.depth_input:
                inverse_depths = 1.0 / context_depths
                context_depths_patch = patchify(inverse_depths, config.patch_size)
                # print(f"after patchify:")
                # print(f"ref_imgs shape: {ref_imgs.shape}, ref_rays shape: {ref_rays.shape}, context_depths_patch shape: {context_depths_patch.shape}")

            # Tokenize into
            # x: [B*V2, V1*N1, DIM1]
            # q: [B*V2, N2, DIM2]
            if self.config.depth_input:
                x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays, context_depths_patch], dim=-1))
                # print(f"after input_tokenizer: x shape: {x.shape}")
                # exit()
            else:
                x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays], dim=-1))
            x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=v2)
            q = self.query_tokenizer(tar_rays)
            q = rearrange(q, "b v2 n d -> (b v2) n d")
            q_tokens = q.shape[1]

            # --- Prepare data for geomtry-aware self-attention ---
            ref_c2ws = repeat(ref_cams.camtoworld, "b v1 x y -> (b v2) v1 x y", v2=v2)
            ref_Ks = repeat(ref_cams.K, "b v1 x y -> (b v2) v1 x y", v2=v2)
            tar_c2ws = rearrange(tar_cams.camtoworld, "b v2 x y -> (b v2) 1 x y", v2=v2)
            tar_Ks = rearrange(tar_cams.K, "b v2 x y -> (b v2) 1 x y")
            c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)  # [B, N, 4, 4] per camera
            Ks = torch.cat([ref_Ks, tar_Ks], dim=1)  # [B, N, 3, 3] per camera
            viewmats = torch.inverse(c2ws)

            # cams = {
            #     'tar_cams': tar_cams,
            #     'ref_cams': ref_cams,
            #     'ref_c2ws': ref_c2ws,
            #     'ref_Ks': ref_Ks,
            #     'tar_c2ws': tar_c2ws,
            #     'tar_Ks': tar_Ks,
            #     'cws': c2ws,
            #     'Ks': Ks,
            # }
            # torch.save(cams, f"logs/debug_tensor/cams.pt")
            # exit()

        with time_block("precompute_enc", enabled=timing_enabled):
            if "frustum" in config.pos_enc or "fpixel" in config.pos_enc or "fpoint" in config.pos_enc \
                or "d_pj" in config.pos_enc or "d_3d" in config.pos_enc \
                or config.pos_enc in ["0+d", "0+inf", "d+inf", "global-0+inf", "global-0+d"]:

                # if "known" in config.depth_type:
                #     depths_for_rope = repeat(context_depths, "b v1 h w 1 -> (b v2) v1 h w 1", v2=v2)
                # else:
                #     depths_for_rope = None

                if context_depths is not None:
                    depths_for_rope = repeat(context_depths, "b v1 h w 1 -> (b v2) v1 h w 1", v2=v2)
                else:
                    depths_for_rope = None
                self.attention._precompute_and_cache_apply_fns(
                    viewmats=viewmats, Ks=Ks, context_depths=depths_for_rope
                )

        def sdpa_fn(q, k, v, **sdpa_kwargs):
            if config.pos_enc == "gta":
                # GTA is effectively PRoPE without intrinsics.
                return self.attention(q, k, v, viewmats=viewmats, Ks=None, timing_enabled=timing_enabled, **sdpa_kwargs)
            elif config.pos_enc == "none":
                # Use the default attention.
                out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
                return out
            else:
                return self.attention(q, k, v, viewmats=viewmats, Ks=Ks, timing_enabled=timing_enabled, **sdpa_kwargs)
            
        if config.pos_enc == "none":
            sdpa_fn = F.scaled_dot_product_attention
            
        with time_block("transformer", enabled=timing_enabled):
            # run attentions
            xq = torch.cat([x, q], dim=1)
            xq = self.encoder(xq, sdpa_fn=sdpa_fn)
            q = xq[:, -q_tokens:, :]
            q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=v2)

            # output layer
            o = self.output_layer(q)
            o = unpatchify(
                o,
                height=config.img_shape[0],
                width=config.img_shape[1],
                patch_size=config.patch_size,
            )

        return o


if __name__ == "__main__":
    # Test the model.
    import tqdm

    device = "cuda:0"
    ref_views = 2
    tar_views = 4
    batch_size = 1
    height = 256
    width = 256

    ref_imgs = torch.randn(batch_size, ref_views, height, width, 3).to(device)
    ref_cams = Camera(
        K=torch.randn(1, ref_views, 3, 3).to(device),
        camtoworld=torch.randn(1, ref_views, 4, 4).to(device),
        height=height,
        width=width,
    )
    tar_cams = Camera(
        K=torch.randn(1, tar_views, 3, 3).to(device),
        camtoworld=torch.randn(1, tar_views, 4, 4).to(device),
        height=height,
        width=width,
    )

    config = LVSMDecoderOnlyModelConfig(ref_views=2)
    model = LVSMDecoderOnlyModel(config).to(device)
    with torch.autocast("cuda"):
        for _ in tqdm.trange(100):
            y = model(ref_imgs, ref_cams, tar_cams)
        assert y.shape == (batch_size, tar_views, height, width, 3)

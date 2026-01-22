import glob
import itertools
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed):
    print(f"Setting random seed to {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nested_to_device(data: Dict, device) -> Dict:
    # Recursively move tensors to device.
    if isinstance(data, dict):
        return {k: nested_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [nested_to_device(v, device) for v in data]
    elif isinstance(data, Tensor):
        return data.to(device)
    else:
        return data


@dataclass
class LauncherConfig:
    # Dump everything to this directory.
    output_dir: str = "results/dbg"

    # Maximum number of steps to train.
    max_steps: int = 100

    # Resume training from this checkpoint.
    auto_resume: bool = False
    resume: str | None = None
    only_model: bool = False

    # Save checkpoint every this many steps. (On only rank 0.)
    ckpt_every: int = 10
    # Keep this many checkpoints.
    ckpt_keeps: int = 3

    # Gradient scaling for mixed precision training.
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "fp16"

    # Check for NaN in gradients.
    check_nan_in_params: bool = False

    # Gradient Accumulation.
    acc: int = 1

    # Random seed.
    fixed_seed: bool = True
    seed: int = 42

    # Gradient clipping.
    grad_clip: float = 0.0

    # test related
    test_every: int = -1
    test_only: bool = False

    # subdirs
    ckpt_subdir: str = "ckpts"
    stats_subdir: str = "stats"
    visual_subdir: str = "visuals"
    test_subdir: str = "tests"


class Launcher:
    def __init__(self, config: LauncherConfig) -> None:
        self.config = config

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"cuda:{self.local_rank}")

        # Setup output directories.
        self.output_dir = self.config.output_dir
        self.ckpt_dir = f"{self.output_dir}/{self.config.ckpt_subdir}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.output_dir}/{self.config.stats_subdir}"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.visual_dir = f"{self.output_dir}/{self.config.visual_subdir}"
        os.makedirs(self.visual_dir, exist_ok=True)
        self.test_dir = f"{self.output_dir}/{self.config.test_subdir}"
        os.makedirs(self.test_dir, exist_ok=True)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[
            self.config.amp_dtype
        ]
        self.use_grad_scaler = self.config.amp and self.config.amp_dtype == "fp16"

        if self.world_rank == 0:
            self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tb")
            if not self.config.test_only:
                (Path(self.output_dir) / "config.yaml").write_text(yaml.dump(config))
                print(f"Wrote config to {self.output_dir}/config.yaml")

    def run(self):
        if self.config.test_only:
            self.test()
        else:
            self.train()

    # This function could be overriden to customize the behavior.
    def train_initialize(self) -> Dict[str, Any]:
        state = {}
        state["model"] = torch.nn.Linear(1, 1)
        state["optimizer"] = torch.optim.Adam(state["model"].parameters(), lr=1e-3)
        state["scheduler"] = None
        state["train_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)}
        )
        return state

    # This function could be overriden to customize the behavior.
    def test_initialize(
        self, model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        state = {}
        if model is not None:
            state["model"] = model
        else:
            state["model"] = torch.nn.Linear(1, 1)
        state["test_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)},
            times=3 if self.world_rank == 0 else 2,  # mimic multi-gpu split.
        )
        return state

    # This function could be overriden to customize the behavior.
    def train_iteration(self, step: int, state: Any, acc_step: int = 0) -> Tensor:
        model = state["model"]
        optimizer = state["optimizer"]
        train_dataiter = state["train_dataiter"]

        model.train()
        data = next(train_dataiter)
        data = nested_to_device(data, self.device)

        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            output = model(data["x"])
            loss = F.mse_loss(output, data["y"])
        return loss

    # This function could be overriden to customize the behavior.
    @torch.inference_mode()
    def test_iteration(self, step: int, state: Any, acc_step: int = 0) -> Any:
        model = state["model"]
        test_dataiter = state["test_dataiter"]

        model.eval()
        losses = []
        for data in test_dataiter:
            data = nested_to_device(data, self.device)
            with torch.amp.autocast(
                "cuda", enabled=self.config.amp, dtype=self.amp_dtype
            ):
                output = model(data["x"])
                loss = F.mse_loss(output, data["y"])
                losses.append(loss.item())

        # collect losses from all ranks
        collected_sizes = [None] * self.world_size
        torch.distributed.all_gather_object(collected_sizes, len(losses))

        collected_metrics = [
            torch.empty(size, device=self.device) for size in collected_sizes
        ]
        torch.distributed.all_gather(
            collected_metrics, torch.tensor(losses, device=self.device)
        )
        collected_metrics = torch.cat(collected_metrics)

        avg_loss = collected_metrics.mean().item()
        self.print_on_master(f"Average loss: {avg_loss}")
        if self.world_rank == 0:
            self.writer.add_scalar("test/loss", avg_loss, step)
        return avg_loss

    # This function could be overriden to customize the behavior.
    def load_state_dict_to_model(
        self, state_dict: Dict, model: torch.nn.Module
    ) -> None:
        self._loosely_load_state_dict_to_model(state_dict, model)

    def load_state_dict_to_optimizer(
        self, state_dict: Dict, optimizer: torch.optim.Optimizer
    ) -> None:
        optimizer.load_state_dict(state_dict)

    def load_state_dict_to_scheduler(
        self, state_dict: Dict, scheduler: torch.optim.lr_scheduler._LRScheduler | None
    ) -> None:
        if scheduler is not None:
            scheduler.load_state_dict(state_dict)

    def save_checkpoint(self, step: int, state: Any) -> None:
        if self.world_rank != 0:
            return
        if step == 0:
            return
        if step % self.config.ckpt_every == 0 or step == self.config.max_steps:
            state_dict = {"step": step}
            if self.world_size > 1:
                model = state["model"].module
            else:
                model = state["model"]
            # torch.compile adds this prefix to the model
            # https://github.com/pytorch/pytorch/issues/101107#issuecomment-1869839379
            state_dict["model"] = getattr(model, "_orig_mod", model).state_dict()
            state_dict["optimizer"] = state["optimizer"].state_dict()
            if state["scheduler"] is not None:
                state_dict["scheduler"] = state["scheduler"].state_dict()
            torch.save(state_dict, f"{self.ckpt_dir}/step-{step:09d}.pt")
            fps = sorted(glob.glob(f"{self.ckpt_dir}/*.pt"))
            for fp in fps[: -self.config.ckpt_keeps]:
                os.remove(fp)

    def maybe_resume(self, state: Any) -> int:
        # Load checkpoint if needed.
        step = 0
        step_for_traindata = 0
        
        ckpt_candidates = []
        if self.config.auto_resume:
            # sort to put latest checkpoint first
            ckpt_candidates = sorted(glob.glob(f"{self.ckpt_dir}/*"), reverse=True)
        elif self.config.resume:
            assert os.path.exists(
                self.config.resume
            ), f"Checkpoint {self.config.resume} not found."
            ckpt_candidates = [self.config.resume]

        for ckpt_path in ckpt_candidates:
            try:
                # It is possible that the checkpoint is corrupted due to brutal shutdown.
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception as e:
                self.print_on_master(
                    f"Error loading checkpoint {ckpt_path}: {e}. Try next candidate."
                )
                continue

            self.load_state_dict_to_model(ckpt["model"], state["model"])
            if not self.config.only_model and not self.config.test_only:
                self.load_state_dict_to_optimizer(ckpt["optimizer"], state["optimizer"])
                self.load_state_dict_to_scheduler(
                    ckpt["scheduler"], state.get("scheduler", None)
                )
                step_for_traindata = ckpt.get("step", 0)
                step = ckpt.get("step", 0) + 1
            elif self.config.test_only:
                step_for_traindata = ckpt.get("step", 0)
                step = ckpt.get("step", 0)
            self.print_on_master(
                f"Resuming from ckpt: {self.ckpt_dir}/{ckpt_path}. set step to: {step}"
            )
            break

        # restore dataloader/dataset state
        
        
        return step

    def _loosely_load_state_dict_to_model(
        self, state_dict: Dict, model: torch.nn.Module
    ) -> None:
        # torch.compile might introduces this prefix
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model = getattr(model, "_orig_mod", model)
        # Filter out parameters that do not match with the model (via key and shape check).
        state_dict_filtered = {}
        for k, v in state_dict.items():
            if k not in model.state_dict():
                print(f"Warning: {k} in ckpt but not in model state_dict.")
                continue
            if model.state_dict()[k].shape != v.shape:
                print(
                    f"Warning: {k} shape mismatch: {model.state_dict()[k].shape} vs {v.shape}"
                )
                continue
            state_dict_filtered[k] = v
        model.load_state_dict(state_dict_filtered, strict=False)
        self.print_on_master(
            f"Loosly loaded ckpt to model: "
            f"{len(state_dict)} keys in ckpt, "
            f"{len(state_dict_filtered)} keys loaded, "
            f"{len(model.state_dict())} keys in model."
        )

    def train(self):
        print("Distributed worker: %d / %d" % (self.world_rank + 1, self.world_size))

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.cuda.set_device(self.local_rank)

        # Initialize model, dataset, optimizer, scheduler ... and load checkpoint if needed
        state = self.train_initialize()
        init_step = self.maybe_resume(state)
        if self.config.test_every > 0:
            test_state = self.test_initialize(model=state["model"])

        for key in ["model", "optimizer"]:
            assert key in state, f"{key} is not in state."
        self.logging_on_master(
            f"Total trainable parameters: "
            f"{sum(p.numel() for p in state['model'].parameters() if p.requires_grad)}"
        )

        # torch.distributed.run ensures that this will work
        # by exporting all the env vars needed to initialize the process group
        torch.distributed.init_process_group(backend="nccl")

        # To device. Use DDP if needed.
        for k, v in state.items():
            if isinstance(v, torch.nn.Module):
                v = v.to(self.device)
                if self.world_size > 1 and k == "model":
                    v = DDP(v, device_ids=[self.local_rank])
                state[k] = v

        if self.use_grad_scaler:
            grad_scaler = torch.amp.GradScaler(device="cuda")

        # Training loop.
        for step in range(init_step, self.config.max_steps + 1):
            for acc_step in range(self.config.acc):
                # Train iteration.
                loss = self.train_iteration(step, state, acc_step)
                loss = loss / self.config.acc

                if loss.isnan():
                    if self.use_grad_scaler:
                        # with grad_scaler we could safely skip this step
                        print(
                            f"Warning: [step={step}] rank={self.world_rank} | loss is NaN."
                        )
                    else:
                        # without grad_scaler, we should exit.
                        print(
                            f"Fatal: [step={step}] rank={self.world_rank} | loss is NaN. exiting."
                        )
                        exit()

                # Backward.
                if self.use_grad_scaler:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # For debugging.
                if self.config.check_nan_in_params:
                    for name, param in state["model"].named_parameters():
                        if torch.isnan(param).any():
                            print(
                                f"[step={step}] rank={self.world_rank} | {name} has NaN."
                            )
                            # breakpoint()
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(
                                    f"[step={step}] rank={self.world_rank} | {name} grad has NaN."
                                )
                                # breakpoint()

            # Update model.
            model = state["model"]
            optimizer = state["optimizer"]
            scheduler = state.get("scheduler", None)

            if self.use_grad_scaler:
                grad_scaler.unscale_(optimizer)
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            # Save checkpoint
            self.save_checkpoint(step, state)

            # Test
            if (
                self.config.test_every > 0
                and step % self.config.test_every == 0
                and step > 0
            ):
                _ = self.test_iteration(step, test_state)

        # Exit.
        torch.distributed.destroy_process_group()

    def test(self):
        assert (
            self.config.resume is not None or self.config.auto_resume
        ), "Resume checkpoint or auto_resume must be provided for testing."
        print("Distributed worker: %d / %d" % (self.world_rank + 1, self.world_size))

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.cuda.set_device(self.local_rank)

        # Initialize model, dataset, ... and load checkpoint if needed
        state = self.test_initialize()
        init_step = self.maybe_resume(state)

        for key in ["model"]:
            assert key in state, f"{key} is not in state."

        # torch.distributed.run ensures that this will work
        # by exporting all the env vars needed to initialize the process group
        torch.distributed.init_process_group(backend="nccl")

        # To device. Test does not need gradient accumulation so no need for DDP.
        for k, v in state.items():
            if isinstance(v, torch.nn.Module):
                v = v.to(self.device)
                state[k] = v

        # Run test.
        _ = self.test_iteration(init_step, state)

        # Exit.
        torch.distributed.destroy_process_group()

    def print_on_master(self, msg: str) -> None:
        if self.world_rank == 0:
            print(msg)

    def logging_on_master(self, msg: str) -> None:
        if self.world_rank == 0:
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            logger = open(f"{self.output_dir}/log.txt", "a")
            logger.write(msg + "\n")
            logger.close()
            print(msg)


if __name__ == "__main__":
    """Example usage:

    Note: somali needs `NCCL_P2P_DISABLE=1`

    # 2 GPUs training
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        <THIS_SCRIPT.py> (args) 

    # 2 GPUs testing
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        <THIS_SCRIPT.py> --resume results/dbg/ckpts/step-000000100.pt --test_only

    # 2 GPUs x 2 nodes training
    NCCL_DEBUG=INFO OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc-per-node=2 --node_rank=0 \
        --rdzv-backend=c10d --rdzv-endpoint=10.55.10.177:29603 lvsm/runner.py
    NCCL_DEBUG=INFO OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc-per-node=2 --node_rank=1 \
        --rdzv-backend=c10d --rdzv-endpoint=10.55.10.177:29603 lvsm/runner.py
    """
    # Flash attention only supports gpu architectures in the range [sm80, sm90]
    # So we use Memory Efficient Attention.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    cfg = tyro.cli(LauncherConfig)
    launcher = Launcher(cfg)
    launcher.run()

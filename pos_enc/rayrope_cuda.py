# pos_enc/rayrop_cuda.py
from typing import Callable, Optional, List

import math
import torch
import torch.nn.functional as F
from typing import Callable, Optional, List

# 사용자가 작성한 CUDA 익스텐션들
from cuda import FusedRayRoPEFunction, FusedGeometry_KV
# import time

MAX_DEPTH = 100.0
MAX_D_F = 10.0
MAX_LOG_DEPTH = 3.0
MAX_ASINH_D_F = math.asinh(MAX_D_F)

class RayRoPE_DotProductAttention_CUDA(torch.nn.Module):
    """
    Self-attention with RayRoPE positional encoding for multi-view patches.
    """

    def __init__(self, head_dim: int, 
                 patches_x: int,
                 patches_y: int,
                 image_width: int,
                 image_height: int, 
                 pos_enc_type: str = 'd_pj+0_3d',
                 num_rays_per_patch: int = 3,
                 depth_type: str = 'predict_dsig',
                 denc_type: str = 'd',
                 freq_base: float = 3.0,
                 apply_vo: bool = True):
        super().__init__()

        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.num_patches = patches_x * patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.pos_enc_type = pos_enc_type
        self.num_rays_per_patch = num_rays_per_patch
        self.depth_type = depth_type
        self.denc_type = denc_type
        self.freq_base = freq_base
        self.apply_vo = apply_vo

        self.parse_pos_enc_type(pos_enc_type)
        
        self.rope_coord_dim = 3 * self.use_p0 + self.num_rays_per_patch * 3 * (int(self.use_pd) + int(self.use_pinf))
        self.rope_mat_dim = 2 * self.rope_coord_dim
        assert self.head_dim % self.rope_mat_dim == 0, f"head_dim={self.head_dim} must be multiple of {self.rope_mat_dim}"
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim
        
        print(f"[RayRoPE CUDA] head_dim: {head_dim}, rope_coord_dim: {self.rope_coord_dim}, num_rope_freqs: {self.num_rope_freqs}")

        log_min, log_max = self._calculate_freq_bands()
        self.register_buffer("log_min_freqs", log_min)
        self.register_buffer("log_max_freqs", log_max)
        
    def parse_pos_enc_type(self, pos_enc_type: str):
        self.use_p0 = False
        self.p0_type = 'none'
        self.use_pd = False
        self.pd_type = 'none'
        self.use_pinf = False
        self.pinf_type = 'none'

        for part in pos_enc_type.split('+'):
            assert '_' in part, f"pos_enc_type part {part} is invalid."
            point, ptype = part.split('_', 1)
            if point == '0':
                self.use_p0 = True
                self.p0_type = ptype
            elif point == 'd':
                self.use_pd = True
                self.pd_type = ptype
            elif point == 'inf':
                self.use_pinf = True
                self.pinf_type = ptype

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        predicted_d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        # ========================================
        """torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()"""
        # ========================================
        
        apply_fn_q, all_apply_fns_kv, apply_fn_o = self._prepare_apply_fns(predicted_d)
        
        # ========================================
        """torch.cuda.synchronize()
        end_time = time.perf_counter()

        peak_mem = torch.cuda.max_memory_allocated() 
        end_mem = torch.cuda.memory_allocated()      

        elapsed_ms = (end_time - start_time) * 1000
        peak_mb = (peak_mem - start_mem) / (1024 ** 2)
        delta_mb = (end_mem - start_mem) / (1024 ** 2)

        print(f"\n['CUDA' _prepare_apply_fns]")
        print(f"Time: {elapsed_ms:.3f} ms")
        print(f"Peak Memory: {peak_mb:.3f} MB")
        print(f"End Memory : {delta_mb:.3f} MB\n")"""
        # ========================================
        

        output = self.rayrope_dot_product_attention(
            q, k, v, 
            apply_fn_q=apply_fn_q, 
            all_apply_fns_kv=all_apply_fns_kv, 
            apply_fn_o=apply_fn_o, 
            **kwargs
        )

        return output
    
    def _calculate_freq_bands(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_periods = []
        
        # p0
        if self.use_p0:
            if self.p0_type == '3d':
                max_periods.extend([1.0 * 4.0] * 3) 
            elif self.p0_type == 'pj':
                max_periods.extend([2.0 * 4.0] * 2)
                if self.denc_type == 'inv_d':
                    max_periods.extend([20.0 * 4.0] * 1)
                elif self.denc_type == 'd':
                    max_periods.extend([MAX_D_F * 2.0 * 4.0] * 1)
                elif self.denc_type == 'asinh_d':
                    max_periods.extend([MAX_ASINH_D_F * 2.0 * 4.0] * 1)

        # pinf
        if self.use_pinf:
            dim = 3 * self.num_rays_per_patch
            max_periods.extend([2.0 * 4.0] * dim)

        # pd 
        if self.use_pd:
            if self.pd_type == '3d':
                dim = 3 * self.num_rays_per_patch
                max_periods.extend([1.0 * 4.0] * dim)
            elif self.pd_type == 'pj':
                max_periods.extend([2.0 * 4.0] * (2 * self.num_rays_per_patch))
                
                if self.denc_type == 'inv_d':
                    max_periods.extend([20.0 * 4.0] * (1 * self.num_rays_per_patch))
                elif self.denc_type == 'd':
                    max_periods.extend([MAX_D_F * 2.0 * 4.0] * (1 * self.num_rays_per_patch))
                elif self.denc_type == 'asinh_d':
                    max_periods.extend([MAX_ASINH_D_F * 2.0 * 4.0] * (1 * self.num_rays_per_patch))

        assert len(max_periods) == self.rope_coord_dim, \
            f"freqs_dim({len(max_periods)}) and rope_coord_dim({self.rope_coord_dim})should be same."

        log_min_list, log_max_list = [], []
        for max_period in max_periods:
            min_period = max_period / (self.freq_base ** (self.num_rope_freqs - 1))
            log_min_list.append(math.log(2 * math.pi / max_period))
            log_max_list.append(math.log(2 * math.pi / min_period))
            
        return torch.tensor(log_min_list, dtype=torch.float32), torch.tensor(log_max_list, dtype=torch.float32)

    def _precompute_and_cache_apply_fns(self, w2cs: torch.Tensor, Ks: Optional[torch.Tensor]):
        self.batch, self.num_cameras, _, _ = w2cs.shape

        self.w2cs = w2cs.contiguous()
        self.c2ws = _invert_SE3(w2cs).contiguous()
        Ks_norm = normalize_K(Ks, self.image_width, self.image_height)

        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), self.w2cs).contiguous()
        self.P_inv = torch.einsum("...ij,...jk->...ik", self.c2ws, _lift_K(_invert_K(Ks_norm))).contiguous()

    def _prepare_apply_fns(self, predicted_d: torch.Tensor) -> tuple:
        pos_KV_all = FusedGeometry_KV.apply(
            self.pos_enc_type, self.batch, self.num_cameras, self.num_patches, self.rope_coord_dim,
            self.patches_x, self.patches_y,
            predicted_d.contiguous(),
            self.P_inv, self.P, self.w2cs, self.c2ws
        ) # (2 * B, C, P, 12)

        # Extract pos_Q
        b_idx = torch.arange(2 * self.batch, device=pos_KV_all.device)
        c_idx = torch.arange(self.num_cameras, device=pos_KV_all.device)
        pos_Q = pos_KV_all[b_idx[:, None], c_idx, c_idx, :, :].reshape(2 * self.batch, self.num_cameras, self.num_patches, self.rope_coord_dim).contiguous()
        
        # Q and Output Apply Function
        apply_fn_q = lambda x, p=pos_Q: FusedRayRoPEFunction.apply(x, p, self.log_min_freqs, self.log_max_freqs, True)
        apply_fn_o = lambda x, p=pos_Q: FusedRayRoPEFunction.apply(x, p, self.log_min_freqs, self.log_max_freqs, False)

        # KV Apply Functions List
        all_apply_fns_kv = []
        for q_cam_idx in range(self.num_cameras):
            pos_KV_q = pos_KV_all[:, q_cam_idx, :, :, :].reshape(2 * self.batch, self.num_cameras, self.num_patches, self.rope_coord_dim)
            apply_fn_kv = lambda x, p=pos_KV_q: FusedRayRoPEFunction.apply(x, p, self.log_min_freqs, self.log_max_freqs, True)
            all_apply_fns_kv.append(apply_fn_kv)

        return apply_fn_q, all_apply_fns_kv, apply_fn_o

    def rayrope_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        apply_fn_q: Callable, all_apply_fns_kv: List[Callable], apply_fn_o: Callable,
        **kwargs
    ) -> torch.Tensor:
        
        out = torch.zeros_like(q)
        q_enc = apply_fn_q(q)

        for cam_idx, apply_fn_kv in enumerate(all_apply_fns_kv):
            k_enc = apply_fn_kv(k)
            v_enc = apply_fn_kv(v) if self.apply_vo else v
            
            q_idx = q_enc[:, :, cam_idx * self.num_patches : (cam_idx + 1) * self.num_patches, :]
            
            out_idx = F.scaled_dot_product_attention(
                query=q_idx.contiguous(),
                key=k_enc.contiguous(),
                value=v_enc.contiguous(),
                **kwargs,
            )
            out[:, :, cam_idx * self.num_patches : (cam_idx + 1) * self.num_patches, :] = out_idx

        if self.apply_vo:
            out = apply_fn_o(out)

        return out.contiguous()


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out

def normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Normalize camera intrinsics."""
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm
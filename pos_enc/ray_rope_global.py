
from functools import partial
from typing import Callable, Optional, Tuple, List

import os
import torch
import torch.nn.functional as F
from pos_enc.timing_utils import time_block
from torch.profiler import profile, record_function, ProfilerActivity


class Global_RayRoPE_DotProductAttention(torch.nn.Module):
    """Patch-RoPE attention with precomputed RoPE coefficients."""

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        rope_transform_type: str = 'none', # none or cayley or cayley_randn.
        pos_enc_type: str = 'global-0+inf', # global-0+inf, global-0+d
        num_rays_per_patch: int = 3,
        freq_base: float = 3.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.pos_enc_type = pos_enc_type
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base

        self.use_p0 = '0' in pos_enc_type
        self.use_pd = 'd' in pos_enc_type
        self.use_pinf = 'inf' in pos_enc_type


        self.rope_coord_dim = 3 * self.use_p0 + self.num_rays_per_patch * 3 * (int(self.use_pd) + int(self.use_pinf))
        self.rope_mat_dim = 2 * self.rope_coord_dim
        assert self.head_dim % (self.rope_coord_dim * 2) == 0, f"rope_enc_dim={self.head_dim} must be multiple of rope_coord_dim * 2={self.rope_coord_dim * 2}"
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim
        print(f"head_dim: {head_dim}, rope_enc_dim: {self.head_dim}, rope_coord_dim: {self.rope_coord_dim}, num_rope_freqs: {self.num_rope_freqs}")
        


    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        timing_enabled: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return frope_dot_product_attention(
            q,
            k,
            v,
            num_cameras=viewmats.shape[1],
            apply_fn_q=self.apply_fn_q,
            apply_fn_kv=self.apply_fn_kv,
            apply_fn_o=self.apply_fn_o,
            timing_enabled=timing_enabled,
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, 
        viewmats: torch.Tensor, 
        Ks: Optional[torch.Tensor],
        context_depths: Optional[torch.Tensor] = None,
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            w2cs=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            use_p0=self.use_p0,
            use_pd=self.use_pd,
            use_pinf=self.use_pinf,
            rope_transform=self.rope_enc_transform,
            num_rays_per_patch=self.num_rays_per_patch,
            coord_dim=self.rope_coord_dim,
            freq_base=self.freq_base,
            context_depths=context_depths,
        )


def frope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    num_cameras: int, 
    apply_fn_q: Callable[[torch.Tensor], torch.Tensor],
    apply_fn_kv: Callable[[torch.Tensor], torch.Tensor],
    apply_fn_o: Callable[[torch.Tensor], torch.Tensor],
    timing_enabled: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    assert q.shape == k.shape == v.shape
    num_patches = seqlen // num_cameras
    assert seqlen == num_cameras * num_patches
    

    with time_block("attention_total", timing_enabled):
        with time_block("apply_enc", timing_enabled):
            q = apply_fn_q(q)
            k = apply_fn_kv(k)
            v = apply_fn_kv(v)

    
        with time_block("attention", timing_enabled):
            out = F.scaled_dot_product_attention(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
                **kwargs,
            )
        
        with time_block("apply_enc", timing_enabled):
            out = apply_fn_o(out)

    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out.contiguous()

@torch.compile
def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    w2cs: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    use_p0: bool = True,
    use_pd: bool = False,
    use_pinf: bool = True,
    num_rays_per_patch: int = 3,
    coord_dim: int = 6,
    freq_base: float = 3.0,
    context_depths: Optional[torch.Tensor] = None,  # (batch, cameras, image_height, image_width, 1)
) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = w2cs.device
    dtype = w2cs.dtype
    (batch, num_cameras, _, _) = w2cs.shape
    num_patches = patches_x * patches_y
    seq_len = num_cameras * num_patches

    c2ws = _invert_SE3(w2cs)  # (batch, cameras, 4, 4)

    # Normalize camera intrinsics.
    assert (Ks is not None), "Camera intrinsics K must be provided"
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    del Ks
    
    # Compute the camera projection matrices we use in PRoPE.
    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
    P_T = P.transpose(-1, -2)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        c2ws,
        _lift_K(_invert_K(Ks_norm)),
    )
    assert P.shape == P_inv.shape == (batch, num_cameras, 4, 4)
    assert head_dim % (2*coord_dim) == 0, f"rope_dim={head_dim} must be multiple of 2*coord_dim={2*coord_dim}"
    num_rope_freqs = head_dim // (2 * coord_dim)

    # Prepare matrices for Q
    positions = {}
    p0_local = torch.zeros((batch, num_cameras, num_patches, 4), device=device, dtype=dtype)
    p0_local[..., -1] = 1.0
    pinf_local = _get_point_coords(Ks_norm, patches_x, patches_y, num_rays_per_patch) # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
    
    p0_global = torch.einsum("bcij,bcpj->bcpi", c2ws, p0_local)  # (batch, num_cameras, num_patches, 4)
    pinf_global = torch.einsum("bcij,bcprj->bcpri", P_inv, pinf_local)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
    pinf_global = pinf_global / torch.norm(pinf_global[..., :3], dim=-1, keepdim=True) # normalize by length

    if use_p0:
        positions['p0'] = p0_global[..., :3]

    if use_pinf:   
        positions['pinf'] = pinf_global[..., :3].reshape(batch, num_cameras, num_patches, -1)

    if use_pd:
        # B, num_contexts, H, W, 1
        num_contexts = context_depths.shape[1]
        num_targets = num_cameras - num_contexts

        target_depths = torch.full((batch, num_targets, image_height, image_width, 1), 10000.0, device=device, dtype=dtype)
        depths = torch.cat([context_depths, target_depths], dim=1)
        
        pd_local = _get_point_coords(Ks_norm, patches_x, patches_y, num_rays_per_patch, depths) # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        pd_global = torch.einsum("bcij,bcprj->bcpri", P_inv, pd_local)
        positions['pd'] = pd_global[..., :3].reshape(batch, num_cameras, num_patches, -1)

    cos_coeffs, sin_coeffs = _prepare_rope_coeffs(positions, num_rope_freqs, freq_base, device)

    if torch.isnan(cos_coeffs).any() or torch.isinf(cos_coeffs).any():
        raise ValueError("NaN/inf values found in cos_coeffs.")
    if torch.isnan(sin_coeffs).any() or torch.isinf(sin_coeffs).any():
        raise ValueError("NaN/inf values found in sin_coeffs.")

    apply_fn_q = partial(_apply_rope_coeffs, cos=cos_coeffs, sin=sin_coeffs, inverse=True)
    apply_fn_o = partial(_apply_rope_coeffs, cos=cos_coeffs, sin=sin_coeffs, inverse=False)
    apply_fn_kv = partial(_apply_rope_coeffs, cos=cos_coeffs, sin=sin_coeffs, inverse=True)

    return apply_fn_q, apply_fn_kv, apply_fn_o

def _get_point_coords(
        Ks_norm: torch.Tensor,
        patches_x: int, 
        patches_y: int, 
        num_rays_per_patch: int,
        depths: torch.Tensor = None, # [B, nC, H, W, 1] or [B, nC, nP, 1]
    ) -> torch.Tensor:
    # return the pixel space 3d homogenous coordinates

    if num_rays_per_patch == 3:
        offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    elif num_rays_per_patch == 2:
        offsets = [[0.0, 0.0], [1.0, 1.0]]
    else:
        offsets = [[0.5, 0.5]]

    device = Ks_norm.device
    batches = Ks_norm.shape[0]
    num_cameras = Ks_norm.shape[1]
    num_patches = patches_x * patches_y
    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device),
        torch.arange(patches_y, device=device),
        indexing="xy",
    )  # [H, W]


    coords = []
    for offset in offsets:
        u = ((u_base + offset[0]) / patches_x) - 0.5 #Since we assume normalized K where px=py=0
        v = ((v_base + offset[1]) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))  # [num_patches, 2]
    coords = torch.stack(coords, dim=1) # (num_patches, num_rays_per_patch, 2)
    assert coords.shape == (num_patches, num_rays_per_patch, 2)
    coords = coords.view(1, 1, num_patches, num_rays_per_patch, 2).expand(batches, num_cameras, -1, -1, -1) # (batch, num_cameras, num_patches, num_rays_per_patch, 2)

    if depths is not None:
        depths = _sample_depth_map(depths, patches_x, patches_y, offsets)  # (batch, num_cameras, num_patches, num_rays_per_patch, 1)
        max_depth = 5.0
        depths_valid = torch.clamp(depths, max=max_depth, min=1e-3)
        coords_4d = torch.cat([coords*depths_valid, depths_valid, torch.ones_like(depths)], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        # coords_4d_valid = torch.cat([coords*depths, depths, torch.ones_like(depths)], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        # coords_4d_invalid = torch.cat([coords, torch.ones_like(depths), torch.zeros_like(depths)], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        # coords_4d = torch.where(depths_valid, coords_4d_valid, coords_4d_invalid)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
    else:
        # if no depth provided, gives points at infinity (directions)
        depths = torch.ones_like(coords[..., :1])
        scales = torch.zeros_like(coords[..., :1])

        coords_4d = torch.cat([coords, depths, scales], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)

    return coords_4d

def _sample_depth_map(
    pixel_depths: torch.Tensor,  # (batch, cameras, image_height, image_width, 1)
    patches_x: int,
    patches_y: int,
    offsets: List[Tuple[float, float]],
) -> torch.Tensor:
    assert pixel_depths.ndim == 5 and pixel_depths.shape[-1] == 1
    batch, cameras, image_height, image_width, _ = pixel_depths.shape
    if image_width % patches_x != 0 or image_height % patches_y != 0:
        raise ValueError("Image resolution must be divisible by the patch grid dimensions.")

    patch_w = image_width // patches_x
    patch_h = image_height // patches_y
    device = pixel_depths.device
    dtype = pixel_depths.dtype

    num_offsets = len(offsets)

    # Grid over patch indices; match meshgrid ordering used in _get_point_coords.
    x_idx = torch.arange(patches_x, device=device, dtype=dtype)
    y_idx = torch.arange(patches_y, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x_idx, y_idx, indexing="xy")
    base_x = grid_x * float(patch_w)
    base_y = grid_y * float(patch_h)

    depths_flat = pixel_depths.reshape(batch * cameras, 1, image_height, image_width)
    samples = []
    for ox, oy in offsets:
        ox_t = torch.as_tensor(ox, device=device, dtype=dtype)
        oy_t = torch.as_tensor(oy, device=device, dtype=dtype)
        sample_x = base_x + ox_t * patch_w
        sample_y = base_y + oy_t * patch_h

        # Convert to normalized grid coordinates in [-1, 1].
        x_norm = sample_x / image_width * 2.0 - 1.0
        y_norm = sample_y / image_height * 2.0 - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).permute(1, 0, 2)
        grid = grid.unsqueeze(0).expand(batch * cameras, -1, -1, -1)
        sampled = F.grid_sample(
            depths_flat,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        samples.append(sampled.permute(0, 1, 3, 2))

    sampled_depths = torch.stack(samples, dim=-1)
    sampled_depths = sampled_depths.reshape(batch, cameras, patches_x * patches_y, num_offsets, 1)
    sampled_depths[sampled_depths > 1000] = torch.inf
    return sampled_depths

@torch.compile
def _prepare_rope_coeffs(
    positions: dict[str, torch.Tensor], # (batch, num_cameras, num_patches, coord_dim)
    num_freqs: int,
    freq_base: float,
    device: torch.device,
):
    """Prepare RoPE cosine and sine coefficients from positions."""
    (batch, num_cameras, num_patches, _) = next(iter(positions.values())).shape
    coord_dim = 0
    for key, value in positions.items():
        assert value.shape == (batch, num_cameras, num_patches, value.shape[-1])
        coord_dim += value.shape[-1]

    rope_angles = []
    for pos_name, pos in positions.items():
        if pos_name == 'p0':
            max_period = 1.0 * 4
        elif pos_name in ['pinf']:
            max_period = 2.0 * 4
        elif pos_name in ['pd']:
            max_period = 10.0 * 4
            pos = torch.clamp(pos, max=20.0, min=-20.0)
        else: 
            raise ValueError(f"Unknown position name: {pos_name}")
        
        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs,
                                max_period=max_period,
                                min_period=min_period).to(device)
        rope_angle = torch.einsum("f,bcpd->bcpfd", freqs, pos)
        rope_angles.append(rope_angle)
        if torch.isnan(rope_angle).any() or torch.isinf(rope_angle).any():
            print(f"nan: {torch.isnan(rope_angle).any()}, inf: {torch.isinf(rope_angle).any()}")
            print(f"pos_name: {pos_name}, pos min/max: {pos.min()}/{pos.max()}")
            
            jobid = os.environ.get('SLURM_JOB_ID', 'unknown')
            rank = os.environ.get('SLURM_PROCID', '0')

            debug_path = f"/home/yuwu3/prope/logs/debug_tensor/posnan_{jobid}_{rank}.pt"
            debug_info = {
                "positions": positions,
                "pos_name": pos_name,
                "pos": pos,
                "rope_angle": rope_angle,
            }
            torch.save(debug_info, debug_path)
            print(f"Debug information saved to {debug_path}")

    rope_angles = torch.cat(rope_angles, dim=-1) # (batch, num_cameras, num_patches, num_freqs, coord_dim)
    assert rope_angles.shape == (batch, num_cameras, num_patches, num_freqs, coord_dim)

    # Compute cosine and sine coefficients
    cos_coeffs = torch.cos(rope_angles)  # (batch, num_cameras, num_patches, num_freqs, coord_dim)
    sin_coeffs = torch.sin(rope_angles)  # (batch, num_cameras, num_patches, num_freqs, coord_dim)

    # Reshape to (batch, seq_len, feat_dim // 2)
    cos_out = cos_coeffs.reshape(batch, num_cameras * num_patches, num_freqs * coord_dim)
    sin_out = sin_coeffs.reshape(batch, num_cameras * num_patches, num_freqs * coord_dim)

    return cos_out, sin_out

def _get_frequency(
    num_freqs: int,
    max_period: float,
    min_period: float,
):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    log_freqs = torch.linspace(
        log_min_frequency,
        log_max_frequency,
        num_freqs,
    )
    freqs = torch.exp(log_freqs)
    ratio = freqs[1] / freqs[0]
    # print(f"max/min: {max_period}, {min_period}, Frequency ratio: {ratio:.4f}, freqs: {freqs}")
    return freqs


@torch.compile
def _apply_rope_coeffs(
    feats: torch.Tensor, # (batch, num_heads, seqlen, feat_dim)
    cos:  torch.Tensor, # (batch, seqlen, feat_dim // 2)
    sin:    torch.Tensor, # (batch, seqlen, feat_dim // 2)
    inverse: bool = False,
    interleaved: bool = False,
):
    """Apply ROPE coefficients to an input array."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    assert cos.shape == (batch, seqlen, feat_dim // 2)
    assert sin.shape == (batch, seqlen, feat_dim // 2)
    cos = cos.unsqueeze(1)  # (batch, 1, seqlen, feat_dim // 2)
    sin = sin.unsqueeze(1)      # (batch, 1, seqlen, feat_dim // 2)

    if interleaved:
        x1 = feats[..., 0::2]
        x2 = feats[..., 1::2]
    else:
        x1, x2 = torch.chunk(feats, 2, dim=-1)  # each is (batch, num_heads, seqlen, feat_dim // 2)
    

    if inverse:
        x_rotated_1 = x1 * cos + x2 * sin
        x_rotated_2 = -x1 * sin + x2 * cos
    else:
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

    if interleaved:
        out = torch.empty_like(feats)
        out[..., 0::2] = x_rotated_1
        out[..., 1::2] = x_rotated_2
    else:
        out = torch.cat([x_rotated_1, x_rotated_2], dim=-1)

    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, seqlen, num_freqs, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    num_freqs = matrix.shape[2]
    mat_dim = matrix.shape[3]
    assert matrix.shape == (batch, seqlen, num_freqs, mat_dim, mat_dim)
    assert mat_dim * num_freqs == feat_dim
    return torch.einsum(
        "bsfij,bnsfj->bnsfi",
        matrix,
        feats.reshape((batch, num_heads, seqlen, num_freqs, mat_dim)),
    ).reshape((batch, num_heads, seqlen, feat_dim))

def _apply_tiled_projmat_fast(feats, matrix):
    # feats: (B, N, S, feat_dim)
    # matrix: (B, S, F, D, D)
    B, N, S, feat_dim = feats.shape
    F, D = matrix.shape[2:4]
    feats = feats.reshape(B, N, S, F, D)  # (B, N, S, F, D)

    # Flatten (B, S, F)
    feats_flat = feats.permute(0, 2, 3, 4, 1).reshape(B * S * F, D, N)
    matrix_flat = matrix.reshape(B * S * F, D, D)

    # Batched matmul: (B*S*F, D, D) @ (B*S*F, D, N) -> (B*S*F, D, N)
    out = torch.matmul(matrix_flat, feats_flat)

    # Reshape back
    out = out.reshape(B, S, F, D, N).permute(0, 4, 1, 2, 3)  # (B, N, S, F, D)
    return out.reshape(B, N, S, F * D)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out

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


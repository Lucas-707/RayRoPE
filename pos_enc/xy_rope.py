from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F
from pos_enc.timing_utils import time_block



class xyRopeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 2,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 2,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
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
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            timing_enabled=timing_enabled,

            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor]
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    cameras = viewmats.shape[1]
    # assert q.shape == k.shape == v.shape
    # assert viewmats.shape == (batch, cameras, 4, 4)
    # assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    # assert seqlen == cameras * patches_x * patches_y
    with time_block("attention_total", timing_enabled):
        with time_block("prepare_enc", timing_enabled):
            apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
                head_dim=head_dim,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=patches_x,
                patches_y=patches_y,
                image_width=image_width,
                image_height=image_height,
                coeffs_x=coeffs_x,
                coeffs_y=coeffs_y,
            )
        with time_block("apply_enc", timing_enabled):
            q = apply_fn_q(q)
            k = apply_fn_kv(k)
            v = apply_fn_kv(v)

        with time_block("attention", timing_enabled):
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                **kwargs,
            )
        
        with time_block("apply_enc", timing_enabled):
            out = apply_fn_o(out)
    # assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out

@torch.compile
def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 2,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 2,
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]
    transforms_kv = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]
    transforms_o = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 2),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


@torch.compile
def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)

@torch.compile
def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


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
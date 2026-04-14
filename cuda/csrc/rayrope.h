// cuda/csrc/rayrope.h
#pragma once

namespace at {
    class Tensor;
}

namespace rayrope {
// -----------------------------------------------------------
/* fused_rayrope_coeffs_fwd
// inputs
 feats: (batch, num_heads, seqlen, feat_dim)
 positions: (batch, num_cameras, num_patches, coord_dim)
 log_min_freqs: (coord_dim,)
 log_max_freqs: (coord_dim,)
 freqs: (num_freqs,)
 interleaved: bool
 inverse: bool
// output
 out: (batch, num_heads, seqlen, feat_dim)
*/
void fused_rayrope_coeffs_fwd(
    // inputs
    const at::Tensor& feats,     // (B, H, N, feat_dim)
    const at::Tensor& positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    //const at::Tensor& freqs,     // (num_freqs)
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    // bool interleaved,
    bool inverse,
    // output
    at::Tensor& out             // (B, H, N, feat_dim)
);

// -----------------------------------------------------------
/* fused_rayrope_coeffs_bwd
// inputs
 feats: (batch, num_heads, seqlen, feat_dim)
 positions: (batch, num_cameras, num_patches, coord_dim)
log_min_freqs: (coord_dim,)
 log_max_freqs: (coord_dim,)
 freqs: (num_freqs,)
 interleaved: bool
 inverse: bool
// output
// grad_output
v_out: (batch, num_heads, seqlen, feat_dim)
// grad_inputs
 v_feats: (batch, num_heads, seqlen, feat_dim)
 v_positions: (batch, num_cameras, num_patches, coord_dim)
*/
void fused_rayrope_coeffs_bwd(
    // fwd_inputs
    const at::Tensor& feats,     // (B, H, N, feat_dim)
    const at::Tensor& positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    //const at::Tensor& freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // fwd_output
    // grad_output
    const at::Tensor& v_out,
    // grad_inputs
    at::Tensor& v_feats,     // (B, H, N, feat_dim)
    at::Tensor& v_positions //  (B, C, P, coord_dim) = (B, N, coord_dim)
);

} // namespace rayrope
// cuda/csrc/rayrope.h
// Torch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <math_constants.h>

// RayRoPE
#include "rayrope.h"

namespace rayrope {

namespace cg = cooperative_groups;

template <typename scalar_t, bool IS_INTERVAL>
__global__ void thread_rayrope_coeffs_fwd (
    // inputs
    const uint32_t B,              // batch
    const uint32_t H,              // num_heads
    const uint32_t C,              // num_cameras
    const uint32_t N,              // seqlen (= C * P)
    const uint32_t P,              // num_patches 
    const uint32_t half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
    const uint32_t num_freqs,      // len of freqs
    const uint32_t coord_dim,      // 
    const scalar_t *__restrict__ feats,     // (B, H, N, feat_dim)
    const scalar_t *__restrict__ positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const scalar_t *__restrict__ log_min_freqs, // (coord_dim,)
    const scalar_t *__restrict__ log_max_freqs, // (coord_dim,)
    // const scalar_t *__restrict__ freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // output
    scalar_t *__restrict__ out             // (B, H, N, feat_dim)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (idx >= total_elements) {
        return;
    }

    // Index Decoding
    uint32_t d_out = idx % half_feat_dim;
    uint32_t tmp   = idx / half_feat_dim;
    uint32_t n_idx = tmp % N;
    tmp            = tmp / N;
    uint32_t h_idx = tmp % H;
    uint32_t b_idx = tmp / H;

    // ================================
    // forward: _prepare_rope_coeff_uniformd
    // ================================
    uint32_t f_idx = d_out / coord_dim;
    uint32_t c_idx = d_out % coord_dim;
    
    scalar_t l_min = log_min_freqs[c_idx];
    scalar_t l_max = log_max_freqs[c_idx];
    scalar_t step = (num_freqs > 1) ? (l_max - l_min) / (num_freqs - 1) : 0.0f;

    scalar_t freq = exp(l_min + step * f_idx);
    // scalar_t freq = freqs[f_idx * coord_dim + c_idx];

    scalar_t cos_val, sin_val;

    if constexpr (IS_INTERVAL) { // E[rho_D(x)]
        uint64_t pos1_idx = b_idx       * (N * coord_dim) + n_idx * coord_dim + c_idx;
        uint64_t pos2_idx = (b_idx + B) * (N * coord_dim) + n_idx * coord_dim + c_idx; 
        
        scalar_t pos1 = positions[pos1_idx];
        scalar_t pos2 = positions[pos2_idx];

        scalar_t angle1 = pos1 * freq;
        scalar_t angle2 = pos2 * freq;
        scalar_t delta = angle2 - angle1;
        
        if (abs(delta) < 1e-2) { // torch.isclose(..., atol=1e-2, rtol=0)
            cos_val = cos(angle1);
            sin_val = sin(angle1);
        } else {
            cos_val = (sin(angle2) - sin(angle1)) / delta;
            sin_val = (cos(angle1) - cos(angle2)) / delta;
        }
    } else { //rho_D(x)
        uint64_t pos_idx = b_idx * (N * coord_dim) + n_idx * coord_dim + c_idx;
        scalar_t pos = positions[pos_idx];
        scalar_t angle = pos * freq;
        
        cos_val = cos(angle);
        sin_val = sin(angle);
    }

    // ================================
    // forward: _apply_rope_coeffs
    // ================================

    uint64_t feat_dim = 2 * half_feat_dim;
    uint64_t feat_idx = b_idx * (H * N * feat_dim) + h_idx * (N * feat_dim) + n_idx * feat_dim;
    uint64_t x1_idx, x2_idx;

    x1_idx = feat_idx + d_out;
    x2_idx = feat_idx + half_feat_dim + d_out;

    /*
    if (interleaved) {
        x1_idx = feat_idx + (d_out * 2);
        x2_idx = feat_idx + (d_out * 2) + 1;
    } else {
        x1_idx = feat_idx + d_out;
        x2_idx = feat_idx + half_feat_dim + d_out;
    }
    */
    scalar_t x1 = feats[x1_idx];
    scalar_t x2 = feats[x2_idx];

    scalar_t out_x1, out_x2;
    if (inverse) {
        out_x1 = x1 * cos_val + x2 * sin_val;
        out_x2 = -x1 * sin_val + x2 * cos_val;
    } else {
        out_x1 = x1 * cos_val - x2 * sin_val;
        out_x2 = x1 * sin_val + x2 * cos_val;
    }

    out[x1_idx] = out_x1;
    out[x2_idx] = out_x2;
}

template <typename scalar_t, bool IS_INTERVAL>
__global__ void thread_rayrope_feats_bwd(
    // fwd_inputs
    const uint32_t B,              // batch
    const uint32_t H,              // num_heads
    const uint32_t C,              // num_cameras
    const uint32_t N,              // seqlen (= C * P)
    const uint32_t P,              // num_patches 
    const uint32_t half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
    const uint32_t num_freqs,      // len of freqs
    const uint32_t coord_dim,      // 
    const scalar_t *__restrict__ feats,     // (B, H, N, feat_dim)
    const scalar_t *__restrict__ positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const scalar_t *__restrict__ log_min_freqs, // (coord_dim,)
    const scalar_t *__restrict__ log_max_freqs, // (coord_dim,)
    // const scalar_t *__restrict__ freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // fwd_output
    // grad_output
    const scalar_t *__restrict__ v_out,
    // grad_inputs
    scalar_t *__restrict__ v_feats     // (B, H, N, feat_dim)
    // scalar_t *__restrict__ v_positions //  (B, C, P, coord_dim) = (B, N, coord_dim)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (idx >= total_elements) {
        return;
    }

    // Index Decoding
    uint32_t d_out = idx % half_feat_dim;
    uint32_t tmp   = idx / half_feat_dim;
    uint32_t n_idx = tmp % N;
    tmp            = tmp / N;
    uint32_t h_idx = tmp % H;
    uint32_t b_idx = tmp / H;

    // ================================
    // forward: _prepare_rope_coeff_uniformd & _apply_rope_coeffs
    // ================================
    uint32_t f_idx = d_out / coord_dim;
    uint32_t c_idx = d_out % coord_dim;
    
    scalar_t l_min = log_min_freqs[c_idx];
    scalar_t l_max = log_max_freqs[c_idx];
    scalar_t step = (num_freqs > 1) ? (l_max - l_min) / (num_freqs - 1) : 0.0f;

    scalar_t freq = exp(l_min + step * f_idx);
    // scalar_t freq = freqs[f_idx * coord_dim + c_idx];
    
    scalar_t cos_val, sin_val;

    if constexpr (IS_INTERVAL) { // E[rho_D(x)]
        uint64_t pos1_idx = b_idx       * (N * coord_dim) + n_idx * coord_dim + c_idx;
        uint64_t pos2_idx = (b_idx + B) * (N * coord_dim) + n_idx * coord_dim + c_idx; 
        
        scalar_t pos1 = positions[pos1_idx];
        scalar_t pos2 = positions[pos2_idx];

        scalar_t angle1 = pos1 * freq;
        scalar_t angle2 = pos2 * freq;
        scalar_t delta = angle2 - angle1;
        
        if (abs(delta) < 1e-2) { // torch.isclose(..., atol=1e-2, rtol=0)
            cos_val = cos(angle1);
            sin_val = sin(angle1);
        } else {
            // E_cosine = (sine2 - sine1) / delta
            // E_sine = (cosine1 - cosine2) / delta
            cos_val = (sin(angle2) - sin(angle1)) / delta;
            sin_val = (cos(angle1) - cos(angle2)) / delta;
        }
    } else { //rho_D(x)
        uint64_t pos_idx = b_idx * (N * coord_dim) + n_idx * coord_dim + c_idx;
        scalar_t pos = positions[pos_idx];
        scalar_t angle = pos * freq;
        
        cos_val = cos(angle);
        sin_val = sin(angle);
    }

    uint64_t feat_dim = 2 * half_feat_dim;
    uint64_t feat_idx = b_idx * (H * N * feat_dim) + h_idx * (N * feat_dim) + n_idx * feat_dim;
    uint64_t x1_idx, x2_idx;

    x1_idx = feat_idx + d_out;
    x2_idx = feat_idx + half_feat_dim + d_out;
    /*
    if (interleaved) {
        x1_idx = feat_idx + (d_out * 2);
        x2_idx = feat_idx + (d_out * 2) + 1;
    } else {
        x1_idx = feat_idx + d_out;
        x2_idx = feat_idx + half_feat_dim + d_out;
    }
    */
    scalar_t x1 = feats[x1_idx];
    scalar_t x2 = feats[x2_idx];

    // ================================
    // backward: _apply_rope_coeffs
    // ================================
    scalar_t v_out_x1 = v_out[x1_idx];
    scalar_t v_out_x2 = v_out[x2_idx];

    scalar_t v_x1 = 0, v_x2 = 0;
    scalar_t v_cos_val = 0, v_sin_val = 0;
    if (inverse) {
        v_x1 += v_out_x1 * cos_val + v_out_x2 * -sin_val;
        v_x2 += v_out_x1 * sin_val + v_out_x2 * cos_val;

        v_cos_val += v_out_x1 * x1 + v_out_x2 * x2;
        v_sin_val += v_out_x1 * x2 + v_out_x2 * -x1;
    } else {
        v_x1 += v_out_x1 * cos_val  + v_out_x2 * sin_val;
        v_x2 += v_out_x1 * -sin_val + v_out_x2 * cos_val;

        v_cos_val += v_out_x1 * x1  + v_out_x2 * x2;
        v_sin_val += v_out_x1 * -x2 + v_out_x2 * x1;
    }

    v_feats[x1_idx] = v_x1;
    v_feats[x2_idx] = v_x2;

    // ================================
    // backward: _prepare_rope_coeff_uniformd
    // ================================
    /*
    if constexpr (IS_INTERVAL) {
        uint64_t pos1_idx = b_idx       * (N * coord_dim) + n_idx * coord_dim + c_idx;
        uint64_t pos2_idx = (b_idx + B) * (N * coord_dim) + n_idx * coord_dim + c_idx; 
        
        scalar_t pos1 = positions[pos1_idx];
        scalar_t pos2 = positions[pos2_idx];

        scalar_t angle1 = pos1 * freq;
        scalar_t angle2 = pos2 * freq;
        scalar_t delta = angle2 - angle1;

        scalar_t v_pos1 = 0, v_pos2 = 0;
        if (abs(delta) < 1e-2) { // torch.isclose(..., atol=1e-2, rtol=0)
            scalar_t v_angle = v_cos_val * -sin(angle1) + v_sin_val * cos(angle1);

            v_pos1 += 0.5 * v_angle * freq;
            v_pos2 += 0.5 * v_angle * freq;

            atomicAdd(&v_positions[pos1_idx], v_pos1);
            atomicAdd(&v_positions[pos2_idx], v_pos2);
        } else {
            scalar_t dc_da1 = (sin(angle2) - sin(angle1) - delta * cos(angle1)) / (delta * delta);
            scalar_t ds_da1 = (cos(angle1) - cos(angle2) - delta * sin(angle1)) / (delta * delta);
            
            scalar_t dc_da2 = (delta * cos(angle2) - sin(angle2) + sin(angle1)) / (delta * delta);
            scalar_t ds_da2 = (delta * sin(angle2) - cos(angle1) + cos(angle2)) / (delta * delta);

            scalar_t v_angle1 = v_cos_val * dc_da1 + v_sin_val * ds_da1;
            scalar_t v_angle2 = v_cos_val * dc_da2 + v_sin_val * ds_da2;

            v_pos1 = v_angle1 * freq;
            v_pos2 = v_angle2 * freq;

            atomicAdd(&v_positions[pos1_idx], v_pos1);
            atomicAdd(&v_positions[pos2_idx], v_pos2);
        }

    } else {
        uint64_t pos_idx = b_idx * (N * coord_dim) + n_idx * coord_dim + c_idx;
        scalar_t pos = positions[pos_idx];
        scalar_t angle = pos * freq;

        scalar_t v_pos = 0;
        v_pos += v_cos_val * -sin(angle) * freq + v_sin_val * cos(angle) * freq;
        
        atomicAdd(&v_positions[pos_idx], v_pos);
    }
    */
}

template <typename scalar_t, bool IS_INTERVAL>
__global__ void thread_rayrope_pos_bwd(
    // fwd_inputs
    const uint32_t B,              // batch
    const uint32_t H,              // num_heads
    const uint32_t C,              // num_cameras
    const uint32_t N,              // seqlen (= C * P)
    const uint32_t P,              // num_patches 
    const uint32_t half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
    const uint32_t num_freqs,      // len of freqs
    const uint32_t coord_dim,      // 
    const scalar_t *__restrict__ feats,     // (B, H, N, feat_dim)
    const scalar_t *__restrict__ positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const scalar_t *__restrict__ log_min_freqs, // (coord_dim,)
    const scalar_t *__restrict__ log_max_freqs, // (coord_dim,)
    // const scalar_t *__restrict__ freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // fwd_output
    // grad_output
    const scalar_t *__restrict__ v_out,
    // grad_inputs
    scalar_t *__restrict__ v_positions //  (B, C, P, coord_dim) = (B, N, coord_dim)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_elements = B * N * coord_dim;
    if (idx >= total_elements) {
        return;
    }

    // Index Decoding
    uint32_t c_idx = idx % coord_dim;
    uint32_t tmp   = idx / coord_dim;
    uint32_t n_idx = tmp % N;
    uint32_t b_idx = tmp / N;

    scalar_t v_pos1_total = 0.0f;
    scalar_t v_pos2_total = 0.0f;
    scalar_t v_pos_total = 0.0f;

    scalar_t l_min = log_min_freqs[c_idx];
    scalar_t l_max = log_max_freqs[c_idx];
    scalar_t step = (num_freqs > 1) ? (l_max - l_min) / (num_freqs - 1) : 0.0f;

    for (uint32_t f_idx = 0; f_idx < num_freqs; ++f_idx) {
        scalar_t freq = exp(l_min + step * f_idx);
        
        scalar_t v_cos_val_sum = 0.0f;
        scalar_t v_sin_val_sum = 0.0f;

        for (uint32_t h_idx = 0; h_idx < H; ++h_idx) {
            uint32_t d_out = f_idx * coord_dim + c_idx;
            uint64_t feat_dim = 2 * half_feat_dim;
            uint64_t feat_idx = b_idx * (H * N * feat_dim) + h_idx * (N * feat_dim) + n_idx * feat_dim;
            
            uint64_t x1_idx = feat_idx + d_out;
            uint64_t x2_idx = feat_idx + half_feat_dim + d_out;

            scalar_t x1 = feats[x1_idx];
            scalar_t x2 = feats[x2_idx];
            scalar_t v_out_x1 = v_out[x1_idx];
            scalar_t v_out_x2 = v_out[x2_idx];

            if (inverse) {
                v_cos_val_sum += v_out_x1 * x1 + v_out_x2 * x2;
                v_sin_val_sum += v_out_x1 * x2 + v_out_x2 * -x1;
            } else {
                v_cos_val_sum += v_out_x1 * x1  + v_out_x2 * x2;
                v_sin_val_sum += v_out_x1 * -x2 + v_out_x2 * x1;
            }
        }

        if constexpr (IS_INTERVAL) { // E[rho_D(x)]
            uint64_t pos1_idx = b_idx       * (N * coord_dim) + n_idx * coord_dim + c_idx;
            uint64_t pos2_idx = (b_idx + B) * (N * coord_dim) + n_idx * coord_dim + c_idx; 
            
            scalar_t pos1 = positions[pos1_idx];
            scalar_t pos2 = positions[pos2_idx];

            scalar_t angle1 = pos1 * freq;
            scalar_t angle2 = pos2 * freq;
            scalar_t delta = angle2 - angle1;
            
            if (abs(delta) < 1e-2) { // torch.isclose(..., atol=1e-2, rtol=0)
                scalar_t v_angle = v_cos_val_sum * -sin(angle1) + v_sin_val_sum * cos(angle1);
                v_pos1_total += 0.5 * v_angle * freq;
                v_pos2_total += 0.5 * v_angle * freq;
            } else {
                scalar_t dc_da1 = (sin(angle2) - sin(angle1) - delta * cos(angle1)) / (delta * delta);
                scalar_t ds_da1 = (cos(angle1) - cos(angle2) - delta * sin(angle1)) / (delta * delta);
                
                scalar_t dc_da2 = (delta * cos(angle2) - sin(angle2) + sin(angle1)) / (delta * delta);
                scalar_t ds_da2 = (delta * sin(angle2) - cos(angle1) + cos(angle2)) / (delta * delta);

                scalar_t v_angle1 = v_cos_val_sum * dc_da1 + v_sin_val_sum * ds_da1;
                scalar_t v_angle2 = v_cos_val_sum * dc_da2 + v_sin_val_sum * ds_da2;

                v_pos1_total += v_angle1 * freq;
                v_pos2_total += v_angle2 * freq;
            }
        } else { //rho_D(x)
            uint64_t pos_idx = b_idx * (N * coord_dim) + n_idx * coord_dim + c_idx;
            scalar_t pos = positions[pos_idx];
            scalar_t angle = pos * freq;
            
            v_pos_total += (v_cos_val_sum * -sin(angle) * freq + v_sin_val_sum * cos(angle) * freq);
        }

        if constexpr (IS_INTERVAL) {
            uint64_t pos1_idx = b_idx       * (N * coord_dim) + n_idx * coord_dim + c_idx;
            uint64_t pos2_idx = (b_idx + B) * (N * coord_dim) + n_idx * coord_dim + c_idx;
            v_positions[pos1_idx] = v_pos1_total;
            v_positions[pos2_idx] = v_pos2_total;
        } else {
            uint64_t pos_idx = b_idx * (N * coord_dim) + n_idx * coord_dim + c_idx;
            v_positions[pos_idx] = v_pos_total;
        }
    }
}

void fused_rayrope_coeffs_fwd(
    // inputs
    const at::Tensor& feats,     // (B, H, N, feat_dim)
    const at::Tensor& positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    // const at::Tensor& freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // output
    at::Tensor& out             // (B, H, N, feat_dim)
) {
    TORCH_CHECK(feats.is_contiguous(), "feats must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(log_min_freqs.is_contiguous(), "log_min_freqs must be contiguous");
    TORCH_CHECK(log_max_freqs.is_contiguous(), "log_max_freqs must be contiguous");
    // TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const uint32_t B = feats.size(0);
    const uint32_t H = feats.size(1);
    const uint32_t N = feats.size(2);
    const uint32_t feat_dim = feats.size(3);
    const uint32_t half_feat_dim =  feat_dim / 2;

    const uint32_t B_pos = positions.size(0);

    const uint32_t C = positions.size(1);
    const uint32_t P = positions.size(2);
    const uint32_t coord_dim = positions.size(3);
    bool is_interval = false;
    
    if (B_pos == 2 * B) {
        is_interval = true;
    } else if (B_pos == B) {
        is_interval = false;
    } else {
        TORCH_CHECK(false, "positions batch size must be B or 2*B");
    }

    TORCH_CHECK(C * P == N, "C * P must equal N");
    
    // const uint32_t num_freqs = freqs.size(0);
    const uint32_t num_freqs = half_feat_dim  / coord_dim;

    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (total_elements == 0) {
        // skip the thread if there are no elements
        return;
    }

    dim3 threads(256);
    dim3 grid((total_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES(
        feats.scalar_type(),
        "thread_rayrope_feats_fwd",
        [&]() {
            if (is_interval) {
                thread_rayrope_coeffs_fwd<scalar_t, true>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    // freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // output
                    out.data_ptr<scalar_t>()             // (B, H, N, feat_dim)
                );
            } else {
                thread_rayrope_coeffs_fwd<scalar_t, false>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    // freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // output
                    out.data_ptr<scalar_t>()             // (B, H, N, feat_dim)
                );
            }
        }
    );
}

void fused_rayrope_coeffs_bwd(
    // fwd_inputs
    const at::Tensor& feats,     // (B, H, N, feat_dim)
    const at::Tensor& positions, //  (B, C, P, coord_dim) = (B, N, coord_dim)
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    // const at::Tensor& freqs,     // (num_freqs)
    // bool interleaved,
    bool inverse,
    // fwd_output
    // grad_output
    const at::Tensor& v_out,
    // grad_inputs
    at::Tensor& v_feats,     // (B, H, N, feat_dim)
    at::Tensor& v_positions //  (B, C, P, coord_dim) = (B, N, coord_dim)
) {
    TORCH_CHECK(feats.is_contiguous(), "feats must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(log_min_freqs.is_contiguous(), "log_min_freqs must be contiguous");
    TORCH_CHECK(log_max_freqs.is_contiguous(), "log_max_freqs must be contiguous");
    // TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(v_out.is_contiguous(), "v_out must be contiguous");
    TORCH_CHECK(v_feats.is_contiguous(), "v_feats must be contiguous");
    TORCH_CHECK(v_positions.is_contiguous(), "v_positions must be contiguous");

    const uint32_t B = feats.size(0);
    const uint32_t H = feats.size(1);
    const uint32_t N = feats.size(2);
    const uint32_t feat_dim = feats.size(3);
    const uint32_t half_feat_dim =  feat_dim / 2;

    const uint32_t B_pos = positions.size(0);
    const uint32_t C = positions.size(1);
    const uint32_t P = positions.size(2);
    const uint32_t coord_dim = positions.size(3);
    bool is_interval = false;

    if (B_pos == 2 * B) {
        is_interval = true;
    } else if (B_pos == B) {
        is_interval = false;
    } else {
        TORCH_CHECK(false, "positions batch size must be B or 2*B");
    }

    TORCH_CHECK(C * P == N, "C * P must equal N");

    // const uint32_t num_freqs = freqs.size(0);
    const uint32_t num_freqs = half_feat_dim  / coord_dim;

    const uint64_t total_feats_elements = B * H * N * half_feat_dim;
    if (total_feats_elements == 0) {
        // skip the thread if there are no elements
        return;
    }

    dim3 threads(256);
    int64_t shmem_size = 0;
    dim3 grid_feats((total_feats_elements + threads.x - 1) / threads.x);
        
    AT_DISPATCH_FLOATING_TYPES(
        feats.scalar_type(),
        "thread_rayrope_feats_bwd",
        [&]() {
            if (is_interval) {
                thread_rayrope_feats_bwd<scalar_t, true>
                <<<grid_feats,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // fwd_inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    //freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // fwd_output
                    // grad_output
                    v_out.data_ptr<scalar_t>(),
                    // grad_inputs
                    v_feats.data_ptr<scalar_t>()    // (B, H, N, feat_dim)
                    // v_positions.data_ptr<scalar_t>() //  (B, C, P, coord_dim) = (B, N, coord_dim)
                );
            } else {
                thread_rayrope_feats_bwd<scalar_t, false>
                <<<grid_feats,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // fwd_inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    // freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // fwd_output
                    // grad_output
                    v_out.data_ptr<scalar_t>(),
                    // grad_inputs
                    v_feats.data_ptr<scalar_t>()    // (B, H, N, feat_dim)
                    // v_positions.data_ptr<scalar_t>() //  (B, C, P, coord_dim) = (B, N, coord_dim)
                   );
            }
        }
    );

    const uint64_t total_pos_elements = B * N * coord_dim; 
    if (total_pos_elements == 0) {
        // skip the thread if there are no elements
        return;
    }

    dim3 grid_pos((total_pos_elements + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        feats.scalar_type(),
        "thread_rayrope_pos_bwd",
        [&]() {
            if (is_interval) {
                thread_rayrope_pos_bwd<scalar_t, true>
                <<<grid_pos,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // fwd_inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    //freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // fwd_output
                    // grad_output
                    v_out.data_ptr<scalar_t>(),
                    // grad_inputs
                    v_positions.data_ptr<scalar_t>() //  (B, C, P, coord_dim) = (B, N, coord_dim)
                );
            } else {
                thread_rayrope_pos_bwd<scalar_t, false>
                <<<grid_pos,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // fwd_inputs
                    B,              // batch
                    H,              // num_heads
                    C,              // num_cameras
                    N,              // seqlen (= C * P)
                    P,              // num_patches 
                    half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                    num_freqs,      // len of freqs
                    coord_dim,      // 
                    feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                    positions.data_ptr<scalar_t>(), //  (B, C, P, coord_dim) = (B, N, coord_dim)
                    log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                    // freqs.data_ptr<scalar_t>(),     // (num_freqs)
                    // interleaved,
                    inverse,
                    // fwd_output
                    // grad_output
                    v_out.data_ptr<scalar_t>(),
                    // grad_inputs
                    v_positions.data_ptr<scalar_t>() //  (B, C, P, coord_dim) = (B, N, coord_dim)
                );
            }
        }
    );
}

}  // namespace rayrope
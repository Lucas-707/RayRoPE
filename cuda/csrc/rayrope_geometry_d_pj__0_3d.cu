// cuda/csrc/rayrope_geometry_d_pj__0_3d.cu
// Torch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <math_constants.h>

#define MAX_DEPTH 100.0f
#define MAX_LOG_DEPTH 3.0f
#define MAX_D_F 10.0f

namespace rayrope {

namespace cg = cooperative_groups;

// 4x4 Matrix and 4x1 Vector Mutiply
template <typename scalar_t>
__device__ __forceinline__ void matmul_4x4_4x1(const scalar_t* mat, const scalar_t* vec, scalar_t* out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = mat[i * 4 + 0] * vec[0] + 
                 mat[i * 4 + 1] * vec[1] + 
                 mat[i * 4 + 2] * vec[2] + 
                 mat[i * 4 + 3] * vec[3];
    }
}

// Inverse L2 Norm
template <typename scalar_t>
__device__ __forceinline__ void norm_inv_3(const scalar_t* vec, scalar_t* out) {
    scalar_t norm_sq = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2] + 1e-9f;
    *out = rsqrtf(static_cast<float>(norm_sq));
}

template <typename scalar_t>
__global__ void thread_geometry_KV_d_pj__0_3d_fwd(
    // inputs
    const uint32_t B,
    const uint32_t C, 
    const uint32_t P,
    const uint32_t patches_x, 
    const uint32_t patches_y,
    const scalar_t *__restrict__ D,       // (B, C, P, 2)
    const scalar_t *__restrict__ P_inv,   // (B, C, 4, 4)
    const scalar_t *__restrict__ P_mat,   // (B, C, 4, 4)
    const scalar_t *__restrict__ w2c,     // (B, C, 4, 4)
    const scalar_t *__restrict__ c2w,     // (B, C, 4, 4)
    // output
    scalar_t *__restrict__ pos_KV        // (2*B, C_q, C_kv, P, 12)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_threads = (uint64_t)B * C * C * P;

    if (idx >= total_threads) return;

    // Index Decoding
    int p_idx = idx % P;
    uint64_t temp = idx / P;
    int c_kv_idx = temp % C;
    temp /= C;
    int c_q_idx = temp % C;
    int b_idx = temp / C;

    int px = p_idx % patches_x;
    int py = p_idx / patches_x;

    // Depth Load
    const scalar_t* D_kv = D + ((b_idx * C + c_kv_idx) * P + p_idx) * 2;
    scalar_t d1 = expf(min(D_kv[0] - D_kv[1], (scalar_t) MAX_LOG_DEPTH));
    scalar_t d2 = expf(min(D_kv[0] + D_kv[1], (scalar_t) MAX_LOG_DEPTH));
    scalar_t depths[2] = {d1, d2};

    // Camera Load
    const scalar_t* c2w_kv_ptr = c2w + (b_idx * C + c_kv_idx) * 16;
    const scalar_t* P_inv_kv_ptr = P_inv + (b_idx * C + c_kv_idx) * 16;
    const scalar_t* w2c_q_ptr = w2c + (b_idx * C + c_q_idx) * 16;
    const scalar_t* P_q_ptr = P_mat + (b_idx * C + c_q_idx) * 16;

    // 0_3d
    scalar_t p0_world[4] = {c2w_kv_ptr[3], c2w_kv_ptr[7], c2w_kv_ptr[11], 1.0f};
    scalar_t p0_cam[4];
    matmul_4x4_4x1(w2c_q_ptr, p0_world, p0_cam);
    scalar_t p0_norm_inv = 1 / max(p0_cam[3], 1e-4f);
    scalar_t p0_3d[3] = {p0_cam[0] * p0_norm_inv, p0_cam[1] * p0_norm_inv, p0_cam[2] * p0_norm_inv};

    const float u_off[3] = {0.0f, 1.0f, 0.0f};
    const float v_off[3] = {0.0f, 0.0f, 1.0f};

    // positions of d1, d2
    #pragma unroll
    for (int d_i = 0; d_i < 2; ++d_i) {
        // (2*B, C_q, C_kv, P, 12)
        uint64_t batch_offset = (uint64_t)(b_idx + d_i * B) * C * C * P;
        uint64_t out_base = (batch_offset + (uint64_t)c_q_idx * C * P + (uint64_t)c_kv_idx * P + p_idx) * 12;
        scalar_t* out = pos_KV + out_base;

        out[0] = p0_3d[0];
        out[1] = p0_3d[1];
        out[2] = p0_3d[2];

        scalar_t disp = 1.0f / min(max(depths[d_i], (scalar_t)1e-2f), (scalar_t)MAX_DEPTH);

        for (int r = 0; r < 3; ++r) {
            scalar_t u = (px + u_off[r]) / (scalar_t)patches_x - 0.5f;
            scalar_t v = (py + v_off[r]) / (scalar_t)patches_y - 0.5f;
            
            scalar_t coords_4d[4] = {u, v, 1.0f, disp};
            scalar_t pd_world[4], pd_cam[4];
            matmul_4x4_4x1(P_inv_kv_ptr, coords_4d, pd_world);
            matmul_4x4_4x1(P_q_ptr, pd_world, pd_cam);

            // pd_dir
            scalar_t pd_norm_inv;
            norm_inv_3(pd_cam, &pd_norm_inv);
            out[3 + r * 2 + 0] = pd_cam[0] * pd_norm_inv;
            out[3 + r * 2 + 1] = pd_cam[1] * pd_norm_inv;

            // pd_depth
            scalar_t z = max(sqrtf(pd_cam[2] * pd_cam[2] + 1e-9f), 1e-4f);
            scalar_t w = max(pd_cam[3], 1e-4f);
            scalar_t pd_depth = z / w;
            out[9 + r] = min(max(pd_depth, -(scalar_t)MAX_D_F), (scalar_t)MAX_D_F);
            
        }
    }
}

template <typename scalar_t>
__global__ void thread_geometry_KV_d_pj__0_3d_bwd(
    // inputs
    const uint32_t B, 
    const uint32_t C, 
    const uint32_t P,
    const uint32_t patches_x, 
    const uint32_t patches_y,
    const scalar_t *__restrict__ D,       // (B, C, P, 2)
    const scalar_t *__restrict__ P_inv,   // (B, C, 4, 4)
    const scalar_t *__restrict__ P_mat,   // (B, C, 4, 4)
    const scalar_t *__restrict__ w2c,     // (B, C, 4, 4)
    const scalar_t *__restrict__ c2w,     // (B, C, 4, 4)
    // output
    // grad_output
    const scalar_t *__restrict__ v_pos_KV, // (2*B, C_q, C_kv, P, 12)
    // grad_input
    scalar_t *__restrict__ v_D             // (B, C, P, 2)
) {
    const uint64_t total_threads = (uint64_t)B * C * P;
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads) return;

    // Index Decoding
    int p_idx = idx % P;
    uint64_t temp = idx / P;
    int c_kv_idx = temp % C;
    int b_idx = temp / C;

    int px = p_idx % patches_x;
    int py = p_idx / patches_x;

    // Load Depths
    const scalar_t* D_kv = D + ((b_idx * C + c_kv_idx) * P + p_idx) * 2;

    scalar_t d1 = expf(min(D_kv[0] - D_kv[1], (scalar_t) MAX_LOG_DEPTH));
    scalar_t d2 = expf(min(D_kv[0] + D_kv[1], (scalar_t) MAX_LOG_DEPTH));

    // d(disp) / d(depth) 
    scalar_t d_disp_dd1 = (d1 < 1e-2f || d1 > (scalar_t)MAX_DEPTH) ? 0.0f : (-1.0f / (d1 * d1));
    scalar_t d_disp_dd2 = (d2 < 1e-2f || d2 > (scalar_t)MAX_DEPTH) ? 0.0f : (-1.0f / (d2 * d2));

    scalar_t disp1 = 1.0f / min(max(d1, 1e-2f), (scalar_t)MAX_DEPTH);
    scalar_t disp2 = 1.0f / min(max(d2, 1e-2f), (scalar_t)MAX_DEPTH);

    const scalar_t* P_inv_kv_ptr = P_inv + (b_idx * C + c_kv_idx) * 16;
    
    // 4st row of P_inv (multiplied by disp)
    scalar_t P_inv_col3[4] = {
        P_inv_kv_ptr[3], P_inv_kv_ptr[7], P_inv_kv_ptr[11], P_inv_kv_ptr[15]
    };

    const float u_off[3] = {0.0f, 1.0f, 0.0f};
    const float v_off[3] = {0.0f, 0.0f, 1.0f};

    scalar_t v_d1 = 0.0f;
    scalar_t v_d2 = 0.0f;

    // Loop all qury camera
    for (int c_q_idx = 0; c_q_idx < C; ++c_q_idx) {
        const scalar_t* P_q_ptr = P_mat + (b_idx * C + c_q_idx) * 16;

        // Jacobian: d(pd_cam) / d(disp) = P_q * P_inv_col3
        scalar_t J_cam_disp[4];
        matmul_4x4_4x1(P_q_ptr, P_inv_col3, J_cam_disp);

        #pragma unroll
        for (int d_i = 0; d_i < 2; ++d_i) {
            scalar_t disp = (d_i == 0) ? disp1 : disp2;
            
            uint64_t batch_offset = (uint64_t)(b_idx + d_i * B) * C * C * P;
            uint64_t out_base = (batch_offset + (uint64_t)c_q_idx * C * P + (uint64_t)c_kv_idx * P + p_idx) * 12;
            const scalar_t* v_out = v_pos_KV + out_base;

            scalar_t v_disp = 0.0f;

            for (int r = 0; r < 3; ++r) {
                // Forward: compute pd_cam
                scalar_t u = (px + u_off[r]) / (scalar_t)patches_x - 0.5f;
                scalar_t v = (py + v_off[r]) / (scalar_t)patches_y - 0.5f;
                scalar_t coords_4d[4] = {u, v, 1.0f, disp};
                
                scalar_t pd_world[4], pd_cam[4];
                matmul_4x4_4x1(P_inv_kv_ptr, coords_4d, pd_world);
                matmul_4x4_4x1(P_q_ptr, pd_world, pd_cam);

                // output gradient
                scalar_t v_out_x     = v_out[3 + r * 2 + 0];
                scalar_t v_out_y     = v_out[3 + r * 2 + 1];
                scalar_t v_out_depth = v_out[9 + r];

                // Chain Rul: z / w
                scalar_t z_sq = pd_cam[2] * pd_cam[2] + 1e-9f;
                scalar_t z = max(sqrtf(z_sq), 1e-4f);
                scalar_t w = max(pd_cam[3], 1e-4f);
                scalar_t pd_depth = z / w;

                v_out_depth = (pd_depth < -(scalar_t)MAX_D_F || pd_depth > (scalar_t)MAX_D_F) ? 0.0f : v_out_depth;
                scalar_t v_z = (sqrtf(z_sq) > 1e-4f) ? (v_out_depth / w) : 0.0f;
                scalar_t v_w = (pd_cam[3] > 1e-4f)   ? (-z * v_out_depth / (w * w)) : 0.0f;

                scalar_t v_c3 = v_w;
                scalar_t v_c2_depth = (sqrtf(z_sq) > 1e-4f) ? (v_z * pd_cam[2] / sqrtf(z_sq)) : 0.0f;

                // Chain Rule: c / norm
                scalar_t pd_norm_inv;
                norm_inv_3(pd_cam, &pd_norm_inv);

                scalar_t v_norm_inv = v_out_x * pd_cam[0] + v_out_y * pd_cam[1];

                // d(norm_inv)/dc = -c * (norm_inv)^3
                scalar_t d_norm_inv_factor = -v_norm_inv * pd_norm_inv * pd_norm_inv * pd_norm_inv; 

                scalar_t v_c0 = v_out_x * pd_norm_inv + d_norm_inv_factor * pd_cam[0];
                scalar_t v_c1 = v_out_y * pd_norm_inv + d_norm_inv_factor * pd_cam[1];
                scalar_t v_c2 = d_norm_inv_factor * pd_cam[2] + v_c2_depth;

                // d(pd_cam)/ddisp
                v_disp += v_c0 * J_cam_disp[0] + 
                          v_c1 * J_cam_disp[1] + 
                          v_c2 * J_cam_disp[2] + 
                          v_c3 * J_cam_disp[3];
            }

            if (d_i == 0) v_d1 += v_disp * d_disp_dd1;
            else          v_d2 += v_disp * d_disp_dd2;
        }
    }

    // Depths gradient
    scalar_t v_logd = 0.0f;
    scalar_t v_sig = 0.0f;

    // d(logd)/dd1
    // d(sig)/dd1
    if (D_kv[0] - D_kv[1] <= (scalar_t)MAX_LOG_DEPTH) { 
        v_logd += v_d1 * d1;   
        v_sig  += v_d1 * d1 * (-1.0f);
    }
    
    // d(logd)/dd2
    // d(sig)/dd2
    if (D_kv[0] + D_kv[1]  <= (scalar_t)MAX_LOG_DEPTH) {
        v_logd += v_d2 * d2;      
        v_sig  += v_d2 * d2 * (1.0f);
    }

    scalar_t* v_D_out = v_D + ((b_idx * C + c_kv_idx) * P + p_idx) * 2;
    v_D_out[0] = v_logd;
    v_D_out[1] = v_sig;
}

void geometry_KV_d_pj__0_3d_fwd(
    // inputs
    const uint32_t patches_x,
    const uint32_t patches_y,
    const at::Tensor& D,       // (B, C, P, 2)
    const at::Tensor& P_inv,   // (B, C, 4, 4)
    const at::Tensor& P_mat,   // (B, C, 4, 4)
    const at::Tensor& w2c,     // (B, C, 4, 4)
    const at::Tensor& c2w,     // (B, C, 4, 4)
    // output
    const at::Tensor& pos_KV        // (2*B, C_q, C_kv, P, 12)
) {
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    TORCH_CHECK(P_inv.is_contiguous(), "P_inv must be contiguous");
    TORCH_CHECK(P_mat.is_contiguous(), "P_mat must be contiguous");
    TORCH_CHECK(w2c.is_contiguous(), "w2c must be contiguous");
    TORCH_CHECK(c2w.is_contiguous(), "c2w must be contiguous");
    TORCH_CHECK(pos_KV.is_contiguous(), "pos_KV must be contiguous");

    const uint32_t B = D.size(0);
    const uint32_t C = D.size(1);
    const uint32_t P = D.size(2);

    const uint64_t total_threads = (uint64_t)B * C * C * P;
    if (total_threads == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    dim3 threads(256);
    dim3 grid((total_threads + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES(
        D.scalar_type(),
        "thread_geometry_KV_d_pj__0_3d_fwd",
        [&]() {
            thread_geometry_KV_d_pj__0_3d_fwd<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // inputs
                    B,
                    C,
                    P,
                    patches_x,
                    patches_y,
                    D.data_ptr<scalar_t>(),
                    P_inv.data_ptr<scalar_t>(),
                    P_mat.data_ptr<scalar_t>(),
                    w2c.data_ptr<scalar_t>(),
                    c2w.data_ptr<scalar_t>(),
                    // output
                    pos_KV.data_ptr<scalar_t>()
                );
        }
    );
}

void geometry_KV_d_pj__0_3d_bwd(
    // inputs
    const uint32_t patches_x, 
    const uint32_t patches_y,
    const at::Tensor& D,       // (B, C, P, 2)
    const at::Tensor& P_inv,   // (B, C, 4, 4)
    const at::Tensor& P_mat,   // (B, C, 4, 4)
    const at::Tensor& w2c,     // (B, C, 4, 4)
    const at::Tensor& c2w,     // (B, C, 4, 4)
    // output
    // grad_output
    const at::Tensor& v_pos_KV, // (2*B, C_q, C_kv, P, 12)
    // grad_input
    const at::Tensor& v_D             // (B, C, P, 2)
) {
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    TORCH_CHECK(P_inv.is_contiguous(), "P_inv must be contiguous");
    TORCH_CHECK(P_mat.is_contiguous(), "P_mat must be contiguous");
    TORCH_CHECK(w2c.is_contiguous(), "w2c must be contiguous");
    TORCH_CHECK(c2w.is_contiguous(), "c2w must be contiguous");
    TORCH_CHECK(v_pos_KV.is_contiguous(), "v_pos_KV must be contiguous");
    TORCH_CHECK(v_D.is_contiguous(), "v_D must be contiguous");

    const uint32_t B = D.size(0);
    const uint32_t C = D.size(1);
    const uint32_t P = D.size(2);

    const uint64_t total_threads = (uint64_t)B * C * P;
    if (total_threads == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    dim3 threads(256);
    dim3 grid((total_threads + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES(
        D.scalar_type(),
        "thread_geometry_KV_d_pj__0_3d_bwd",
        [&]() {
            thread_geometry_KV_d_pj__0_3d_bwd<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    // inputs
                    B,
                    C,
                    P,
                    patches_x,
                    patches_y,
                    D.data_ptr<scalar_t>(),
                    P_inv.data_ptr<scalar_t>(),
                    P_mat.data_ptr<scalar_t>(),
                    w2c.data_ptr<scalar_t>(),
                    c2w.data_ptr<scalar_t>(),
                    // output
                    // grad_output
                    v_pos_KV.data_ptr<scalar_t>(),
                    // grad_input
                    v_D.data_ptr<scalar_t>()
                );
        }
    );
}

PYBIND11_MODULE(fused_geometry_d_pj__0_3d, m) {
    m.def("forward", &rayrope::geometry_KV_d_pj__0_3d_fwd, "Geometry KV Forward");
    m.def("backward", &rayrope::geometry_KV_d_pj__0_3d_bwd, "Geometry KV Backward");
}

}  // namespace rayrope
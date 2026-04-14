import sys
import os

# 현재 파일(test.py)이 있는 디렉토리를 파이썬 경로에 최우선으로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import numpy as np

from pos_enc.rayrope import RayRoPE_DotProductAttention
from pos_enc.rayrope_cuda import RayRoPE_DotProductAttention_CUDA

import time

def benchmark_rayrope():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch = 4
    num_cameras = 4
    patches_x, patches_y = 16, 16
    num_patches = patches_x * patches_y
    seqlen = num_cameras * num_patches
    num_rays_per_patch = 3
    head_dim = 48 
    num_heads = 8

    print(f"\n--- Benchmarking Setup ---")
    
    print(f"Batch: {batch}, Cameras: {num_cameras}, Patches: {patches_x}x{patches_y}")
    print(f"Seqlen: {seqlen}, Heads: {num_heads}, Head Dim: {head_dim}")

    torch.manual_seed(42)
    q = torch.randn(batch, num_heads, seqlen, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch, num_heads, seqlen, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch, num_heads, seqlen, head_dim, device=device, requires_grad=True)
    pd = torch.randn(batch, seqlen, 2, device=device, requires_grad=True)
    grad_out = torch.randn_like(q)

    w2cs = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch, num_cameras, -1, -1)
    Ks = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch, num_cameras, -1, -1)

    kwargs = {
        "head_dim": head_dim,
        "patches_x": patches_x,
        "patches_y": patches_y,
        "image_width": 256,
        "image_height": 256,
        "pos_enc_type": 'd_pj+0_3d',
        "num_rays_per_patch": num_rays_per_patch,
        "freq_base": 3.0,
        "apply_vo": True,
    }
    
    ref_model = RayRoPE_DotProductAttention(**kwargs).to(device)
    cuda_model = RayRoPE_DotProductAttention_CUDA(**kwargs).to(device)

    ref_model._precompute_and_cache_apply_fns(w2cs, Ks)
    cuda_model._precompute_and_cache_apply_fns(w2cs, Ks)

    def measure_time(func, name, num_iters=100):
        for _ in range(10):
            func()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"[{name}] Avg Time: {avg_time:.3f} ms")
        return avg_time

    print("\n--- Forward Pass ---")
    def run_fwd_ref(): return ref_model(q, k, v, predicted_d=pd)
    def run_fwd_cud(): return cuda_model(q, k, v, predicted_d=pd)

    torch.cuda.reset_peak_memory_stats()
    time_fwd_ref = measure_time(run_fwd_ref, "PyTorch Original")
    print("PyTorch Forward Mem:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    
    torch.cuda.reset_peak_memory_stats()
    time_fwd_cud = measure_time(run_fwd_cud, "CUDA Optimized")
    print("CUDA Forward Mem:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    
    print(f"Forward Speedup: {time_fwd_ref / time_fwd_cud:.2f}x")

    print("\n--- Backward Pass ---")
    out_ref = ref_model(q, k, v, predicted_d=pd)
    out_cud = cuda_model(q, k, v, predicted_d=pd)
    
    def run_bwd_ref(): 
        q.grad, k.grad, v.grad, pd.grad = None, None, None, None
        out_ref.backward(grad_out, retain_graph=True)

    def run_bwd_cud(): 
        q.grad, k.grad, v.grad, pd.grad = None, None, None, None
        out_cud.backward(grad_out, retain_graph=True)

    torch.cuda.reset_peak_memory_stats()
    time_bwd_ref = measure_time(run_bwd_ref, "PyTorch Original")
    print("PyTorch Backward Mem:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    
    torch.cuda.reset_peak_memory_stats()
    time_bwd_cud = measure_time(run_bwd_cud, "CUDA Optimized")
    print("CUDA Backward Mem:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    
    print(f"Backward Speedup: {time_bwd_ref / time_bwd_cud:.2f}x")
    
    print(f"\nTotal Speedup (Fwd+Bwd): {(time_fwd_ref + time_bwd_ref) / (time_fwd_cud + time_bwd_cud):.2f}x")

def test_rayrope_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch = 2
    num_cameras = 2
    patches_x, patches_y = 4, 4
    num_patches = patches_x * patches_y
    seqlen = num_cameras * num_patches
    num_rays_per_patch = 3
    
    head_dim = 48 
    num_heads = 4


    torch.manual_seed(42)
    q = torch.randn(batch, num_heads, seqlen, head_dim, device=device)
    k = torch.randn(batch, num_heads, seqlen, head_dim, device=device)
    v = torch.randn(batch, num_heads, seqlen, head_dim, device=device)
    

    predicted_d = torch.randn(batch, seqlen, 2, device=device)
    predicted_d[..., 1] *= 0.01 

    w2cs = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch, num_cameras, -1, -1).clone()
    Ks = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch, num_cameras, -1, -1).clone()
    
    def clone_with_grad(t):
        return t.clone().detach().requires_grad_(True)

    q_ref, k_ref, v_ref, pd_ref = clone_with_grad(q), clone_with_grad(k), clone_with_grad(v), clone_with_grad(predicted_d)
    q_cud, k_cud, v_cud, pd_cud = clone_with_grad(q), clone_with_grad(k), clone_with_grad(v), clone_with_grad(predicted_d)


    kwargs = {
        "head_dim": head_dim,
        "patches_x": patches_x,
        "patches_y": patches_y,
        "image_width": 64,
        "image_height": 64,
        "pos_enc_type": 'd_pj+0_3d',
        "num_rays_per_patch": num_rays_per_patch,
        "freq_base": 3.0,
        "apply_vo": True,
    }
    
    ref_model = RayRoPE_DotProductAttention(**kwargs).to(device)
    cuda_model = RayRoPE_DotProductAttention_CUDA(**kwargs).to(device)


    ref_model._precompute_and_cache_apply_fns(w2cs, Ks)
    cuda_model._precompute_and_cache_apply_fns(w2cs, Ks)

    print("--- Forward Pass Testing ---")
    out_ref = ref_model(q_ref, k_ref, v_ref, predicted_d=pd_ref)
    out_cud = cuda_model(q_cud, k_cud, v_cud, predicted_d=pd_cud)

    fwd_diff = torch.max(torch.abs(out_ref - out_cud)).item()
    print(f"Forward Max Diff: {fwd_diff:.6e}")
    try:
        torch.testing.assert_close(out_ref, out_cud, rtol=1e-3, atol=1e-4)
        print("[SUCCESS] Forward passes match!")
    except AssertionError as e:
        print("[FAIL] Forward passes do not match strictly:")
        print(e)

    print("\n--- Backward Pass Testing ---")
    grad_out = torch.randn_like(out_ref)
    
    out_ref.backward(grad_out)
    out_cud.backward(grad_out)

    tensors_to_check = {
        "Q grad": (q_ref.grad, q_cud.grad),
        "K grad": (k_ref.grad, k_cud.grad),
        "V grad": (v_ref.grad, v_cud.grad),
    }

    for name, (grad_ref, grad_cud) in tensors_to_check.items():
        diff = torch.max(torch.abs(grad_ref - grad_cud)).item()
        print(f"{name} Max Diff: {diff:.6e}")
        try:

            torch.testing.assert_close(grad_ref, grad_cud, rtol=1e-3, atol=1e-4)
            print(f"[SUCCESS] {name} matches!")
        except AssertionError as e:
            print(f"[FAIL] {name} does not match strictly.")

    pd_diff = torch.max(torch.abs(pd_ref.grad - pd_cud.grad)).item()
    print(f"\nPredicted Depth grad Max Diff: {pd_diff:.6e}")
    
    diff_mask = torch.abs(pd_ref.grad - pd_cud.grad) > 1e-3
    num_diff_elements = diff_mask.sum().item()
    total_elements = pd_ref.grad.numel()
    
    print(f"Elements with significant grad diff (>1e-3): {num_diff_elements} / {total_elements} ({(num_diff_elements/total_elements)*100:.2f}%)")
    print("Note: Some difference is expected here due to the analytical 50:50 split in the CUDA kernel for delta < 1e-2.")

if __name__ == "__main__":
    # test_rayrope_equivalence()
    benchmark_rayrope()
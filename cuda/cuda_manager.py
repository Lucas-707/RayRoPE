import os
from torch.utils.cpp_extension import load

class GeometryKernelManager:
    _kernels = {}
    _base_dir = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_kernel(cls, pos_enc_type: str):
        if pos_enc_type not in cls._kernels:
            
            safe_name = pos_enc_type.replace('+', '__')
            cu_filename = f"rayrope_geometry_{safe_name}.cu"
            cu_filepath = os.path.join(cls._base_dir, 'csrc', cu_filename)
            
            if not os.path.exists(cu_filepath):
                raise FileNotFoundError(f"CUDA kernel file not found: {cu_filepath}")

            module = load(
                name=f"fused_geometry_{safe_name}",
                sources=[cu_filepath],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                verbose=False
            )
            cls._kernels[pos_enc_type] = module
            print(f"[Lazy Load] Compilation finished for '{pos_enc_type}'.")
        return cls._kernels[pos_enc_type]
import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_NAME = 'rayrope_cuda_ext'

csrc_dir = os.path.join(os.path.dirname(__file__), 'cuda', 'csrc')


source_files = [
    os.path.join(csrc_dir, 'bindings.cpp'),
    os.path.join(csrc_dir, 'rayrope_coeffs.cu'),
]

setup(
    name='rayrope_cuda', 
    version='0.1.0+v_lab',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name=EXT_NAME, 
            sources=source_files,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
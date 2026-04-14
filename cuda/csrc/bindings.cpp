// cuda/csrc/bindings.cpp
#include <torch/extension.h>
#include "rayrope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rayrope_coeffs_fwd", &rayrope::fused_rayrope_coeffs_fwd, "Fused RayRoPE Forward");
    m.def("fused_rayrope_coeffs_bwd", &rayrope::fused_rayrope_coeffs_bwd, "Fused RayRoPE Backward");
}
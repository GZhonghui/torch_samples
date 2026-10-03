#include <torch/extension.h>

void scale_cuda_in_place(torch::Tensor tensor, double factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "scale_in_place",
        &scale_cuda_in_place,
        "Scale a contiguous CUDA float32 tensor in place");
}

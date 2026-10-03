#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

__global__ void scale_kernel(float* values, int64_t length, float factor) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < length) {
        values[index] *= factor;
    }
}

}  // namespace

void scale_cuda_in_place(torch::Tensor tensor, double factor) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be on a CUDA device");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "tensor must be float32");
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");

    if (tensor.numel() == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(tensor.device());
    const int threads = 256;
    const int blocks =
        static_cast<int>((tensor.numel() + threads - 1) / threads);

    // data_ptr<float>() is a GPU address. Only device code may dereference it.
    // Use PyTorch's current stream so this operation follows surrounding work.
    scale_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        tensor.data_ptr<float>(), tensor.numel(), static_cast<float>(factor));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

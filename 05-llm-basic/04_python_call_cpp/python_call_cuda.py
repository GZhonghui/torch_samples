from pathlib import Path

import torch
from torch.utils.cpp_extension import load


DIRECTORY = Path(__file__).parent


def build_extension():
    """编译并加载包含 C++ 绑定和 CUDA kernel 的 PyTorch extension。"""
    return load(
        name="native_cuda_tensor",
        sources=[
            str(DIRECTORY / "cuda_tensor.cpp"),
            str(DIRECTORY / "cuda_kernel.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("本示例需要 CUDA GPU、CUDA 版 PyTorch 和 CUDA toolkit。")

    extension = build_extension()
    values = torch.tensor(
        [[1.0, -2.0], [3.5, 4.0]], device="cuda", dtype=torch.float32
    )

    print(f"GPU: {torch.cuda.get_device_name(values.device)}")
    print(f"缩放前的 CUDA tensor:\n{values}")
    print(f"tensor.data_ptr() 显存地址: 0x{values.data_ptr():x}")

    # extension 接收同一个 CUDA tensor；CUDA kernel 直接修改它的显存存储。
    extension.scale_in_place(values, 2.5)
    torch.cuda.synchronize()

    print(f"CUDA kernel 原地写入后的同一 tensor:\n{values}")
    print(
        "\n关键步骤: Python CUDA tensor -> C++ torch::Tensor -> "
        "data_ptr<float>() -> CUDA kernel 直接读写显存。"
    )


if __name__ == "__main__":
    main()

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


SOURCE_PATH = Path(__file__).with_name("native_math.cpp")


def build_library(build_dir: Path) -> Path:
    """将 C++ 源文件编译成当前平台可由 Python 加载的动态库。"""
    compiler = os.environ.get("CXX", "c++")
    if sys.platform == "darwin":
        library_path = build_dir / "libnative_math.dylib"
        link_flags = ["-dynamiclib"]
    elif sys.platform.startswith("linux"):
        library_path = build_dir / "libnative_math.so"
        link_flags = ["-shared", "-fPIC"]
    else:
        raise RuntimeError("本示例目前支持 macOS 或 Linux。")

    command = [
        compiler,
        "-std=c++11",
        "-O2",
        *link_flags,
        str(SOURCE_PATH),
        "-o",
        str(library_path),
    ]
    print("编译 C++ 动态库:")
    print("  " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"找不到 C++ 编译器 {compiler!r}，请安装编译器或设置 CXX 环境变量。"
        ) from error
    return library_path


def load_library(library_path: Path) -> ctypes.CDLL:
    """加载动态库，并声明 Python 与 C++ 之间的函数签名。"""
    library = ctypes.CDLL(str(library_path))

    # ctypes 必须知道 C++ 函数接收和返回什么类型，才能正确传递内存。
    library.add_ints.argtypes = [ctypes.c_int, ctypes.c_int]
    library.add_ints.restype = ctypes.c_int
    # data_ptr() 提供的是一段存储区的起始地址，因此此处接收裸指针地址。
    library.scale_in_place.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    library.scale_in_place.restype = None
    return library


def scale_cpu_tensor_in_cpp(
    library: ctypes.CDLL, tensor: torch.Tensor, factor: float
) -> None:
    """将 CPU float32 tensor 的存储地址传给 C++，由 C++ 原地修改数据。"""
    if tensor.device.type != "cpu":
        raise ValueError(
            "普通 C++ 函数只能直接访问 CPU tensor；CUDA tensor 需要 CUDA kernel。"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(
            "本示例中的 C++ 指针类型是 float*，tensor 必须是 torch.float32。"
        )
    if not tensor.is_contiguous():
        raise ValueError("本示例要求 contiguous tensor，保证内存按元素连续排列。")

    # tensor 在 C++ 调用期间仍被 Python 变量持有，因此 data_ptr 指向的存储有效。
    library.scale_in_place(
        ctypes.c_void_p(tensor.data_ptr()), tensor.numel(), ctypes.c_float(factor)
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="python_call_cpp_") as temp_dir:
        library = load_library(build_library(Path(temp_dir)))

        total = library.add_ints(17, 25)
        print(f"\nPython 调用 C++ add_ints(17, 25) = {total}")

        values = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
        pointer = values.data_ptr()
        print(f"缩放前的 torch tensor:\n{values}")
        print(f"tensor.data_ptr() 地址: 0x{pointer:x}")

        scale_cpu_tensor_in_cpp(library, values, 2.5)

        print(f"C++ 通过 float* 原地写入后的同一 tensor:\n{values}")
        print(
            "\n关键步骤: tensor.data_ptr() -> C++ float* -> "
            "直接读写 tensor 底层 CPU 存储。"
        )


if __name__ == "__main__":
    main()

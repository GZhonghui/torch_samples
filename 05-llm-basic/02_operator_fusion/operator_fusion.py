import time

import torch


def unfused_op(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	# 未融合版本：逐个执行算子（矩阵乘 + 加偏置 + ReLU）
	# 对 GPU 来说通常会启动多个 kernel，调度开销更高
	y = torch.matmul(x, w)
	y = y + b
	y = torch.relu(y)
	return y


@torch.compile
def fused_op(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	# 融合版本：torch.compile 会尝试把多算子合并成更少的 kernel
	# 好处：减少 kernel 启动次数、减少中间张量读写
	y = torch.matmul(x, w)
	y = y + b
	y = torch.relu(y)
	return y


def benchmark(fn, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, iters: int) -> float:
	# 预热：让编译/缓存/显卡频率稳定，避免第一次运行干扰计时
	for _ in range(10):
		fn(x, w, b)
    # is_cuda 判断张量是否在 GPU 上
	if x.is_cuda:
		torch.cuda.synchronize()

	# 计时主循环
	start = time.perf_counter()
	for _ in range(iters):
		fn(x, w, b)
	if x.is_cuda:
		torch.cuda.synchronize()
	end = time.perf_counter()
	# 返回平均耗时（毫秒）
	return (end - start) * 1000.0 / iters


def main() -> None:
	# 如果有 CUDA 就用 GPU，否则退化到 CPU
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# GPU 上用半精度更接近真实加速场景；CPU 用 float32 更稳定
	dtype = torch.float16 if device == "cuda" else torch.float32

	# 输入/权重形状：x:[batch, in_dim], w:[in_dim, out_dim], b:[out_dim]
	# 输出形状：y:[batch, out_dim]
	batch = 2048
	in_dim = 1024
	out_dim = 4096

	x = torch.randn(batch, in_dim, device=device, dtype=dtype)
	w = torch.randn(in_dim, out_dim, device=device, dtype=dtype)
	b = torch.randn(out_dim, device=device, dtype=dtype)

	# 迭代次数：越大越稳定，但耗时也更长
	iters = 200
	t_unfused = benchmark(unfused_op, x, w, b, iters)
	t_fused = benchmark(fused_op, x, w, b, iters)

	speedup = t_unfused / t_fused if t_fused > 0 else float("inf")

	# 打印结果
	print(f"device={device} dtype={dtype}")
	print(f"unfused: {t_unfused:.3f} ms/iter")
	print(f"fused:   {t_fused:.3f} ms/iter")
	print(f"speedup: {speedup:.2f}x")


if __name__ == "__main__":
	main()

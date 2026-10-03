import argparse
from collections.abc import Callable

import torch


def compute(
    x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, repeats: int
) -> None:
    """重复矩阵乘法，让一批输入的计算时间足以和下一批传输重叠。"""
    # torch.mm(..., out=output) 直接复用预分配的输出显存。
    # 这样循环中关注的是 GPU kernel 的排队与执行，不会夹杂反复申请显存的开销。
    for _ in range(repeats):
        torch.mm(x, weight, out=output)


def elapsed_ms(
    fn: Callable[[torch.cuda.Event, torch.cuda.Event], None]
) -> float:
    """用 CUDA Event 度量 GPU 工作时间，而不是 CPU 端提交耗时。"""
    # CUDA kernel/拷贝通常是异步提交的：Python 函数返回时，GPU 工作
    # 可能仍未完成。因此用 time.perf_counter() 只会测到一部分提交时间。
    # enable_timing=True 的 Event 能由 GPU 时间线计算两个标记之间的耗时。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 清空上一轮留下的 GPU 工作，确保 start 是本次实验的计时起点。
    torch.cuda.synchronize()
    # 未指定 stream 时，Event 被记录到 PyTorch 当前 stream（这里是默认 stream）。
    # record() 本身不会等待 GPU 立刻执行，只是在该 stream 中插入一个时间标记。
    start.record()
    fn(start, end)
    # CPU 只等待 end Event 到达；到达后 end 之前的被测工作必然已经执行完成。
    end.synchronize()
    return start.elapsed_time(end)


def run_serial(
    host_batches: list[torch.Tensor],
    device_inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    weight: torch.Tensor,
    repeats: int,
    finished: torch.cuda.Event | None = None,
) -> None:
    # 基准版本只使用当前默认 stream。同一 stream 上的 CUDA 操作严格有序：
    #
    # copy(batch 0) -> compute(batch 0) -> copy(batch 1) -> compute(batch 1) -> ...
    #
    # 所以 batch 1 的 H2D 传输无法在 batch 0 的计算期间执行，这里没有 overlap。
    for host, device_input, output in zip(host_batches, device_inputs, outputs):
        # 即使指定了 non_blocking=True，同一 stream 中后续的 compute 仍会
        # 等待本次 copy 完成，以保证 device_input 中的数据可用。
        device_input.copy_(host, non_blocking=True)
        compute(device_input, weight, output, repeats)
    if finished is not None:
        # finished 与上述工作记录在同一默认 stream，因此它到达时，
        # 所有串行 copy/compute 都已经完成。
        finished.record()


def run_overlapped(
    host_batches: list[torch.Tensor],
    device_inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    weight: torch.Tensor,
    repeats: int,
    start: torch.cuda.Event | None = None,
    finished: torch.cuda.Event | None = None,
) -> None:
    # 每个 CUDA stream 都是一条独立的 GPU 命令队列：
    # - copy_stream 只提交 CPU -> GPU 的 H2D 拷贝；
    # - compute_stream 只提交矩阵乘计算。
    # 若硬件支持并发拷贝/计算，且任务大小合适，两个 stream 的任务可重叠执行。
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    # 每一批数据都有自己的 ready Event，表示“该批输入已经拷贝到 GPU”。
    # 计算不能仅因为处于另一个 stream 就直接开始，否则可能读取尚未拷完的数据。
    input_ready = [torch.cuda.Event() for _ in host_batches]

    if start is not None:
        # start 记录在默认 stream 中，而实际工作发生在下面创建的两个 stream。
        # 让两个工作 stream 等待 start，可确保所有被测 copy/compute
        # 都严格发生在计时起点之后，否则跨 stream 调度顺序没有保证。
        copy_stream.wait_event(start)
        compute_stream.wait_event(start)

    # CPU 端依次“提交”各批任务，并不是等待上一批执行完成再进入下一轮。
    # 以 batch 0 和 batch 1 为例，排队后的依赖关系大致如下：
    #
    # copy_stream:    copy(0) ---------> copy(1) ---------> ...
    #                    | ready(0)        | ready(1)
    # compute_stream:     wait -> compute(0) wait -> compute(1) -> ...
    #
    # Event 保证 compute(i) 不会早于 copy(i)；不同批之间则允许
    # compute(0) 与 copy(1) 同时执行，从而形成传输/计算 overlap。
    for host, device_input, output, ready in zip(
        host_batches, device_inputs, outputs, input_ready
    ):
        with torch.cuda.stream(copy_stream):
            # 该调用将异步 H2D copy 排入 copy_stream。
            # 真正能做到 CPU -> GPU 异步传输，需要 host 位于 pinned memory；
            # main() 中创建 host_batches 时已经设置了 pin_memory=True。
            device_input.copy_(host, non_blocking=True)

            # Event 记录在当前的 copy_stream 中。因此 ready 到达意味着：
            # 该 Event 之前排入此 stream 的本批 copy 已经完成。
            ready.record()

        with torch.cuda.stream(compute_stream):
            # wait_event() 是 GPU 端队列依赖：它不会让 Python/CPU 在此阻塞，
            # 只会使 compute_stream 的后续任务等到 ready 到达再执行。
            compute_stream.wait_event(ready)
            compute(device_input, weight, output, repeats)

    if finished is not None:
        # finished 排在 compute_stream 的末尾，它到达表示最后一批计算已完成。
        # elapsed_ms() 中由 CPU 调用 finished.synchronize()，这才是本例中
        # 明确的 CPU 等待点；之前的 stream/Event 操作都只是异步提交依赖。
        with torch.cuda.stream(compute_stream):
            finished.record()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="演示 CUDA Stream 的传输/计算 overlap 与 CUDA Event 同步。"
    )
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("本示例需要 CUDA GPU 和支持 CUDA 的 PyTorch。")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    # 异步 CPU -> GPU 传输要求 CPU 数据位于 pinned (page-locked) memory。
    # 普通 pageable memory 可能被换页，CUDA 通常需要先额外复制到临时
    # pinned 缓冲区，导致 host.copy_(..., non_blocking=True) 无法充分异步。
    host_batches = [
        torch.randn(
            args.rows, args.width, dtype=dtype, pin_memory=True
        )
        for _ in range(args.batches)
    ]
    # 为每批输入预分配独立的 GPU buffer：在 copy(i+1) 进行时，
    # compute(i) 仍然读取 batch i 的 buffer，不会发生读写冲突。
    device_inputs = [
        torch.empty_like(host, device=device) for host in host_batches
    ]
    # 权重始终驻留在 GPU，传输部分只模拟每一批新输入送入 GPU 的场景。
    weight = torch.randn(args.width, args.width, device=device, dtype=dtype)
    serial_outputs = [
        torch.empty(args.rows, args.width, device=device, dtype=dtype)
        for _ in host_batches
    ]
    overlap_outputs = [torch.empty_like(output) for output in serial_outputs]

    # 第一次 CUDA 调用可能包含 CUDA context、cuBLAS 等初始化开销；
    # 先执行一次小范围相同工作并同步完成，再开始对比计时。
    run_serial(
        host_batches[:1],
        device_inputs[:1],
        serial_outputs[:1],
        weight,
        args.repeats,
    )
    torch.cuda.synchronize()

    # 串行版本：所有传输与计算位于一个 stream。
    serial_ms = elapsed_ms(
        lambda _, finished: run_serial(
            host_batches,
            device_inputs,
            serial_outputs,
            weight,
            args.repeats,
            finished,
        )
    )
    # overlap 版本：拷贝/计算位于两个 stream，用 Event 连接数据依赖。
    overlap_ms = elapsed_ms(
        lambda start, finished: run_overlapped(
            host_batches,
            device_inputs,
            overlap_outputs,
            weight,
            args.repeats,
            start,
            finished,
        )
    )

    # 两条执行路径完成同样的计算。这里检查 Event 建立的数据依赖
    # 是否正确，避免把“读到了未拷贝完成的数据”误当成性能提升。
    results_match = all(
        torch.allclose(serial, overlap)
        for serial, overlap in zip(serial_outputs, overlap_outputs)
    )
    speedup = serial_ms / overlap_ms

    print(f"GPU                    : {torch.cuda.get_device_name()}")
    print(f"批次数/输入 shape       : {args.batches} x ({args.rows}, {args.width})")
    print(f"每批矩阵乘重复次数       : {args.repeats}")
    print(f"串行 copy + compute     : {serial_ms:.3f} ms")
    print(f"双 stream overlap       : {overlap_ms:.3f} ms")
    print(f"加速比                  : {speedup:.2f}x")
    print(f"结果一致                : {results_match}")


if __name__ == "__main__":
    main()

"""Sanity check: 自定义 KV 传输扩展 vs torch .copy_ 在真实 KV 布局下结果一致。

覆盖 P1 的四种 slice 形态:H2D 前缀、D2H 单 token、D2H 前缀、D2H 整段尾部。
运行:  python learn/05_kv_transfer_check.py
"""

import torch

import mrl.kv_transfer as kv_transfer


def _flat(cache):
    units = cache.shape[0] * cache.shape[1]
    return cache.reshape(units, cache.shape[2], cache.shape[3])


def _run_case(name, dst_view, src_view):
    # 参考结果:torch 原生 strided copy。
    ref = dst_view.clone()
    ref.copy_(src_view, non_blocking=True)

    # 待测:自定义扩展。从同一初始 dst 重新拷一遍。
    out = torch.empty_like(dst_view).copy_(torch.zeros_like(dst_view))
    used_ext = kv_transfer.strided_copy_2d(out, src_view)
    torch.cuda.synchronize()

    ref_cpu = ref.to("cpu", dtype=torch.float32)
    out_cpu = out.to("cpu", dtype=torch.float32)
    ok = torch.equal(ref_cpu, out_cpu)
    print(f"[{name:18s}] ext_used={used_ext}  match={ok}  shape={tuple(src_view.shape)}")
    if not ok:
        diff = (ref_cpu - out_cpu).abs().max().item()
        raise AssertionError(f"{name}: mismatch, max abs diff={diff}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能验证 KV 传输扩展。")
    assert kv_transfer.is_available(), "扩展未能编译/加载"

    B, H, max_len, Hd = 2, 4, 16, 8
    direct_start = 3  # 跨过部分 unit,模拟 direct_unit_start。
    valid_len = 10
    write_pos = valid_len  # 单 token 追加位置。
    dev = torch.device("cuda")

    home = torch.randn(B, H, max_len, Hd, dtype=torch.float16, pin_memory=True)
    work = torch.randn(B, H, max_len, Hd, dtype=torch.float16, device=dev)
    home_f, work_f = _flat(home), _flat(work)

    # H2D 前缀:CPU home -> GPU work[direct:, :valid_len]
    _run_case(
        "H2D prefix",
        work_f[direct_start:, :valid_len],
        home_f[direct_start:, :valid_len],
    )
    # D2H 单 token:GPU work -> CPU home[direct:, write_pos:write_pos+1]
    _run_case(
        "D2H decode 1tok",
        home_f[direct_start:, write_pos : write_pos + 1],
        work_f[direct_start:, write_pos : write_pos + 1],
    )
    # D2H 前缀:GPU work -> CPU home[direct:, :valid_len]
    _run_case(
        "D2H prefill prefix",
        home_f[direct_start:, :valid_len],
        work_f[direct_start:, :valid_len],
    )
    # D2H 整段尾部:GPU work -> CPU home[direct:]
    _run_case(
        "D2H full tail",
        home_f[direct_start:],
        work_f[direct_start:],
    )

    print("All KV transfer cases matched torch .copy_.")


if __name__ == "__main__":
    main()

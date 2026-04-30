# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Microbenchmark: BF16 vs INT8 W8A8 SGL MoE kernel on CPU.

Usage (inside container):
  source /opt/venv_moe/bin/activate
  numactl -m 2 -N 2 python3 tests/kernels/moe/bench_cpu_moe.py
"""

import time

import torch

import vllm._custom_ops  # noqa: F401 — registers ops


def bench_fused_experts_cpu(
    M: int,
    N: int,
    K: int,
    E: int,
    topk: int,
    use_int8: bool,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """Return median wall-clock ms for one fused_experts_cpu call."""

    # Activations
    x = torch.randn(M, K, dtype=torch.bfloat16)
    topk_weights = torch.ones(M, topk, dtype=torch.float32) / topk
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32)

    if use_int8:
        # INT8 W8A8 path
        w1_raw = torch.randint(-10, 10, (E, 2 * N, K), dtype=torch.int8)
        w2_raw = torch.randint(-10, 10, (E, K, N), dtype=torch.int8)
        packed_w1 = torch.ops._C.convert_weight_packed(w1_raw)
        packed_w2 = torch.ops._C.convert_weight_packed(w2_raw)
        w1_scale = torch.rand(E * 2 * N, dtype=torch.float32) * 0.1 + 0.01
        w2_scale = torch.rand(E * K, dtype=torch.float32) * 0.1 + 0.01

        def run():
            out = x.clone()
            torch.ops._C.fused_experts_cpu(
                out,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                True,  # inplace
                True,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                w1_scale,
                w2_scale,
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
            return out
    else:
        # BF16 path
        w1_raw = torch.randn(E, 2 * N, K, dtype=torch.bfloat16)
        w2_raw = torch.randn(E, K, N, dtype=torch.bfloat16)
        packed_w1 = torch.ops._C.convert_weight_packed(w1_raw)
        packed_w2 = torch.ops._C.convert_weight_packed(w2_raw)

        def run():
            out = x.clone()
            torch.ops._C.fused_experts_cpu(
                out,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                True,  # inplace
                False,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                None,  # w1_scale
                None,  # w2_scale
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
            return out

    # Warmup
    for _ in range(warmup):
        run()

    # Benchmark
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    print(f"PyTorch {torch.__version__}, AMX={torch.cpu._is_amx_tile_supported()}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    configs = [
        # (M,    N,     K,    E, topk, label)
        (1, 1024, 4096, 8, 2, "decode-small"),
        (4, 1024, 4096, 8, 2, "decode-batch4"),
        (32, 1024, 4096, 8, 2, "small-prefill"),
        (128, 1024, 4096, 8, 2, "mid-prefill"),
        (512, 1024, 4096, 8, 2, "large-prefill"),
        (1024, 1024, 4096, 8, 2, "xl-prefill"),
        # DeepSeek-like dimensions
        (1, 1536, 7168, 64, 6, "ds-decode"),
        (128, 1536, 7168, 64, 6, "ds-mid-prefill"),
        (512, 1536, 7168, 64, 6, "ds-large-prefill"),
    ]

    cols = ("Config", "M", "N", "K", "E", "top", "BF16 ms", "INT8 ms", "Ratio")
    header = (
        f"{cols[0]:<20} {cols[1]:>5} {cols[2]:>5} {cols[3]:>5} "
        f"{cols[4]:>3} {cols[5]:>3} {cols[6]:>9} {cols[7]:>9} {cols[8]:>7}"
    )
    print(header)
    print("-" * len(header))

    for M, N, K, E, topk, label in configs:
        bf16_ms = bench_fused_experts_cpu(M, N, K, E, topk, use_int8=False)
        int8_ms = bench_fused_experts_cpu(M, N, K, E, topk, use_int8=True)
        ratio = int8_ms / bf16_ms if bf16_ms > 0 else float("inf")
        row = (
            f"{label:<20} {M:>5} {N:>5} {K:>5} {E:>3} {topk:>3}"
            f" {bf16_ms:>9.2f} {int8_ms:>9.2f} {ratio:>6.2f}x"
        )
        print(row)


if __name__ == "__main__":
    main()

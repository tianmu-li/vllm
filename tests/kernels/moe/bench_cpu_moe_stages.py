# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stage-level profiling for INT8 MoE: measure quantization overhead separately.

Usage (inside container):
  source /opt/venv_moe/bin/activate
  numactl -m 2 -N 2 python3 tests/kernels/moe/bench_cpu_moe_stages.py
"""

import time

import torch

import vllm._custom_ops  # noqa: F401


def bench_quantization_overhead(
    M: int, K: int, warmup: int = 5, repeat: int = 20
) -> float:
    """Measure per-token INT8 quantization time (ms) for [M, K] BF16 tensor."""
    x = torch.randn(M, K, dtype=torch.bfloat16)

    def run():
        return torch.ops._C.per_token_quant_int8_cpu(x)

    for _ in range(warmup):
        run()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def bench_full_moe(M, N, K, E, topk, use_int8, warmup=5, repeat=20):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    topk_weights = torch.ones(M, topk, dtype=torch.float32) / topk
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32)

    if use_int8:
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
                True,
                True,
                False,
                w1_scale,
                w2_scale,
                None,
                None,
                None,
                True,
            )
            return out
    else:
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
                True,
                False,
                False,
                None,
                None,
                None,
                None,
                None,
                True,
            )
            return out

    for _ in range(warmup):
        run()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    print(f"PyTorch {torch.__version__}, Threads: {torch.get_num_threads()}")
    print()

    # Stage 0 + 1.5 quantization overhead
    print("=== Per-token INT8 Quantization Overhead ===")
    print(f"{'M':>6} {'K':>6} {'quant ms':>10} {'tokens/ms':>12}")
    print("-" * 40)
    for M, K in [
        (1, 4096),
        (32, 4096),
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (128, 7168),
        (512, 7168),
    ]:
        ms = bench_quantization_overhead(M, K)
        print(f"{M:>6} {K:>6} {ms:>10.3f} {M / ms:>12.1f}")

    print()
    print("=== Quantization as % of Total INT8 MoE Time ===")
    print(f"{'Config':<18} {'M':>5} {'INT8 ms':>9} {'Quant ms':>9} {'Quant%':>7}")
    print("-" * 55)

    configs = [
        (128, 1024, 4096, 8, 2, "mid-prefill-E8"),
        (512, 1024, 4096, 8, 2, "large-prefill-E8"),
        (1024, 1024, 4096, 8, 2, "xl-prefill-E8"),
        (128, 1536, 7168, 64, 6, "ds-mid-E64"),
        (512, 1536, 7168, 64, 6, "ds-large-E64"),
    ]

    for M, N, K, E, topk, label in configs:
        int8_ms = bench_full_moe(M, N, K, E, topk, use_int8=True)
        # Stage 0: quantize [M, K], Stage 1.5: quantize [M*topk, N]
        q_input = bench_quantization_overhead(M, K)
        q_inter = bench_quantization_overhead(M * topk, N)
        total_quant = q_input + q_inter
        pct = total_quant / int8_ms * 100 if int8_ms > 0 else 0
        print(f"{label:<18} {M:>5} {int8_ms:>9.2f} {total_quant:>9.2f} {pct:>6.1f}%")


if __name__ == "__main__":
    main()

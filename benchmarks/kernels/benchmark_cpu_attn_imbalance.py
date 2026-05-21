# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for CPU attention thread-imbalance across decode/prefill/mixed.

Measures wall time of cpu_attention_with_kv_cache.  Three modes:
  decode  -- all requests are single-token decodes (original behavior)
  prefill -- all requests are full-prompt prefills
  mixed   -- a fraction of requests are prefill, the rest are decode

External per-thread timing (recommended when available):
    perf stat -e cpu-clock --per-thread -p <pid> &
    OMP_NUM_THREADS=31 numactl -m 5 -N 5 \\
    .venv/bin/python benchmarks/kernels/benchmark_cpu_attn_imbalance.py \\
        --mode prefill --num-reqs 128 --q-len 2048

Usage:
    LD_PRELOAD=".../libiomp5.so" \\
    OMP_NUM_THREADS=31 numactl -m 5 -N 5 \\
    .venv/bin/python benchmarks/kernels/benchmark_cpu_attn_imbalance.py \\
        --mode decode --num-reqs 128 --kv-len 2048
"""

import argparse
import glob
import math
import os
import time

import torch

# Initialize AMX tile registers before any attention kernel call.
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["decode", "prefill", "mixed"], default="decode")
    p.add_argument("--num-reqs", type=int, default=128)
    p.add_argument(
        "--kv-len", type=int, default=2048, help="KV length for decode requests"
    )
    p.add_argument(
        "--q-len", type=int, default=2048, help="Query length per prefill request"
    )
    p.add_argument(
        "--q-len-jitter",
        type=float,
        default=0.0,
        help="When >0, each prefill q_len is drawn uniformly from "
        "{q_len-512, q_len-256, q_len, q_len+256} (seed=0)",
    )
    p.add_argument(
        "--prefill-frac",
        type=float,
        default=0.0625,
        help="Fraction of num_reqs that are prefill in mixed mode "
        "(default 0.0625 -> 8 prefill + 120 decode for 128 reqs)",
    )
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument(
        "--cpu-affinity-probe",
        action="store_true",
        help="After warmup, read /proc/self/task/*/stat utime ticks "
        "before and after probe-iters iterations and print "
        "per-thread delta",
    )
    p.add_argument(
        "--probe-iters",
        type=int,
        default=1000,
        help="Number of kernel invocations between the utime reads in "
        "--cpu-affinity-probe (large enough that ticks >> 1)",
    )
    return p.parse_args()


def _read_thread_utimes() -> dict[str, int]:
    """Read utime (field 14) from /proc/self/task/*/stat for all threads."""
    utimes: dict[str, int] = {}
    for path in glob.glob("/proc/self/task/*/stat"):
        try:
            with open(path) as f:
                data = f.read()
            rp = data.rfind(")")
            fields = data[rp + 2 :].split()
            # After comm: state(0) ppid(1) pgrp(2) session(3) tty_nr(4)
            # tpgid(5) flags(6) minflt(7) cminflt(8) majflt(9) cmajflt(10)
            # utime(11) stime(12) ...
            utime = int(fields[11])
            tid = path.split("/")[4]
            utimes[tid] = utime
        except Exception:
            pass
    return utimes


def _make_jittered_qlens(num_reqs: int, q_len: int) -> list[int]:
    torch.manual_seed(0)
    choices = torch.tensor(
        [max(1, q_len - 512), max(1, q_len - 256), q_len, q_len + 256]
    )
    indices = torch.randint(0, 4, (num_reqs,))
    return choices[indices].tolist()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    block_size = args.block_size
    num_reqs = args.num_reqs
    kv_len = args.kv_len
    q_len = args.q_len
    num_kv_heads = args.num_kv_heads
    num_q_heads = args.num_q_heads
    head_dim = args.head_dim
    mode = args.mode

    isa = _get_attn_isa(dtype, block_size, head_dim)

    if mode == "decode":
        # ---- decode: all requests q_len=1 --------------------------------
        blocks_per_req = math.ceil(kv_len / block_size)
        num_blocks = num_reqs * blocks_per_req + 32

        kv_cache = torch.randn(
            2, num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        key_cache_packed = torch.empty(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache_packed = torch.empty_like(key_cache_packed)
        slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
        cpu_attn_reshape_and_cache(
            key=kv_cache[0].view(-1, num_kv_heads, head_dim),
            value=kv_cache[1].view(-1, num_kv_heads, head_dim),
            key_cache=key_cache_packed,
            value_cache=value_cache_packed,
            slot_mapping=slot_mapping,
            isa=isa,
        )

        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)
        seq_lens = torch.full((num_reqs,), kv_len, dtype=torch.int32)
        query = torch.randn(num_reqs, num_q_heads, head_dim, dtype=dtype)

        block_table = torch.zeros(num_reqs, blocks_per_req, dtype=torch.int32)
        for i in range(num_reqs):
            for j in range(blocks_per_req):
                block_table[i, j] = i * blocks_per_req + j

    elif mode == "prefill":
        # ---- prefill: all requests are full-prompt prefills ---------------
        if args.q_len_jitter > 0.0:
            q_lens = _make_jittered_qlens(num_reqs, q_len)
        else:
            q_lens = [q_len] * num_reqs

        total_q = sum(q_lens)
        max_q_len = max(q_lens)
        blocks_per_req = math.ceil(max_q_len / block_size)
        num_blocks = num_reqs * blocks_per_req + 32

        kv_cache = torch.randn(
            2, num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        key_cache_packed = torch.empty(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache_packed = torch.empty_like(key_cache_packed)
        slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
        cpu_attn_reshape_and_cache(
            key=kv_cache[0].view(-1, num_kv_heads, head_dim),
            value=kv_cache[1].view(-1, num_kv_heads, head_dim),
            key_cache=key_cache_packed,
            value_cache=value_cache_packed,
            slot_mapping=slot_mapping,
            isa=isa,
        )

        cumsum = [0]
        for ql in q_lens:
            cumsum.append(cumsum[-1] + ql)
        query_start_loc = torch.tensor(cumsum, dtype=torch.int32)
        seq_lens = torch.tensor(q_lens, dtype=torch.int32)
        query = torch.randn(total_q, num_q_heads, head_dim, dtype=dtype)

        block_table = torch.zeros(num_reqs, blocks_per_req, dtype=torch.int32)
        for i in range(num_reqs):
            for j in range(blocks_per_req):
                block_table[i, j] = i * blocks_per_req + j

    else:
        # ---- mixed: M prefill reqs + (num_reqs-M) decode reqs ------------
        m = round(num_reqs * args.prefill_frac)
        n_dec = num_reqs - m

        if args.q_len_jitter > 0.0:
            prefill_qlens = _make_jittered_qlens(m, q_len)
        else:
            prefill_qlens = [q_len] * m

        max_q_len = max(prefill_qlens) if prefill_qlens else 1
        prefill_blocks_per_req = math.ceil(max_q_len / block_size)
        decode_blocks_per_req = math.ceil(kv_len / block_size)
        max_blocks_per_req = max(prefill_blocks_per_req, decode_blocks_per_req)
        num_blocks = num_reqs * max_blocks_per_req + 32

        kv_cache = torch.randn(
            2, num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        key_cache_packed = torch.empty(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache_packed = torch.empty_like(key_cache_packed)
        slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
        cpu_attn_reshape_and_cache(
            key=kv_cache[0].view(-1, num_kv_heads, head_dim),
            value=kv_cache[1].view(-1, num_kv_heads, head_dim),
            key_cache=key_cache_packed,
            value_cache=value_cache_packed,
            slot_mapping=slot_mapping,
            isa=isa,
        )

        # build per-req q_lens and seq_lens: prefill first, then decode
        all_qlens = prefill_qlens + [1] * n_dec
        all_seqlens = prefill_qlens + [kv_len] * n_dec

        total_q = sum(all_qlens)
        cumsum = [0]
        for ql in all_qlens:
            cumsum.append(cumsum[-1] + ql)
        query_start_loc = torch.tensor(cumsum, dtype=torch.int32)
        seq_lens = torch.tensor(all_seqlens, dtype=torch.int32)
        query = torch.randn(total_q, num_q_heads, head_dim, dtype=dtype)

        block_table = torch.zeros(num_reqs, max_blocks_per_req, dtype=torch.int32)
        for i in range(num_reqs):
            for j in range(max_blocks_per_req):
                block_table[i, j] = i * max_blocks_per_req + j

    scale = head_dim**-0.5

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_reqs,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=dtype,
        query_start_loc=query_start_loc,
        causal=True,
        sliding_window_size=-1,
        isa=isa,
        enable_kv_split=False,
    )

    output = torch.empty_like(query)

    def _run() -> None:
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=key_cache_packed,
            value_cache=value_cache_packed,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0,
            scheduler_metadata=metadata,
            s_aux=None,
        )

    print(f"Warming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        _run()

    if args.cpu_affinity_probe:
        before = _read_thread_utimes()
        for _ in range(args.probe_iters):
            _run()
        after = _read_thread_utimes()
        per_tid = {tid: after[tid] - before.get(tid, after[tid]) for tid in after}
        # Drop the main thread (tid == pid); we only care about OMP worker deltas.
        main_tid = str(os.getpid())
        worker_items = [(int(t), d) for t, d in per_tid.items() if t != main_tid]
        worker_items.sort()  # by tid ascending (OMP threads spawn in order)
        if worker_items:
            deltas_sorted = sorted(d for _, d in worker_items)
            n = len(deltas_sorted)
            median_d = deltas_sorted[n // 2]
            print(
                f"[utime-probe] probe_iters={args.probe_iters} "
                f"workers={n} "
                f"min={deltas_sorted[0]} median={median_d} "
                f"max={deltas_sorted[-1]} ticks"
            )
            print(
                "[utime-probe] per-worker (tid-ascending): "
                + ", ".join(str(d) for _, d in worker_items)
            )

    print(f"Timing ({args.iters} iters)...")
    times_ns: list[int] = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        _run()
        times_ns.append(time.perf_counter_ns() - t0)

    times_ns.sort()
    median_us = times_ns[len(times_ns) // 2] / 1_000
    p95_us = times_ns[int(len(times_ns) * 0.95)] / 1_000

    extra = ""
    if mode in ("prefill", "mixed"):
        q_len_str = f"{q_len}±jitter" if args.q_len_jitter > 0 else str(q_len)
        extra = f" q_len={q_len_str}"
        if mode == "mixed":
            m_val = round(num_reqs * args.prefill_frac)
            extra += f" prefill_reqs={m_val} decode_reqs={num_reqs - m_val}"

    print(
        f"mode={mode}{extra} num_reqs={num_reqs} kv_len={kv_len} "
        f"num_kv_heads={num_kv_heads} num_q_heads={num_q_heads} "
        f"head_dim={head_dim} block_size={block_size} "
        f"dtype={args.dtype} isa={isa}"
    )
    print(f"Wall time  median={median_us:.1f}µs  p95={p95_us:.1f}µs")


if __name__ == "__main__":
    main()

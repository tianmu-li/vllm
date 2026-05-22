# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for CPU SHM all-reduce latency across message sizes.

Usage (TP=2, nodes 1-2, 32 threads/rank):
  VLLM_CPU_OMP_THREADS_BIND="32-63|64-95" \\
    numactl --cpunodebind=1,2 --membind=1,2 \\
    .venv/bin/python benchmarks/distributed/bench_cpu_shm_allreduce.py \\
      --ws 2 --threads 32 --label baseline

Usage (TP=4, nodes 1-4, 32 threads/rank):
  VLLM_CPU_OMP_THREADS_BIND="32-63|64-95|96-127|128-159" \\
    numactl --cpunodebind=1,2,3,4 --membind=1,2,3,4 \\
    .venv/bin/python benchmarks/distributed/bench_cpu_shm_allreduce.py \\
      --ws 4 --threads 32 --label baseline

--threads must equal the number of cores in each VLLM_CPU_OMP_THREADS_BIND
slot (e.g. "32-63" → 32 threads).  The SHM manager is initialised with this
count so the adaptive-OMP logic sees the same thread budget as a real worker.

--label is an arbitrary string embedded in the SHM group name; use different
labels for concurrent baseline/optimized runs to avoid SHM name collisions.

Output: CSV on stdout, one row per (size_bytes, ws, dtype) cell.
"""

import argparse
import os
import socket
import time
import uuid

import torch
import torch.distributed as dist

# Sizes to sweep (bytes per rank).
_SIZES_BYTES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    262144,
    524288,
    1048576,
    4194304,
]

_WARMUP_ITERS = 200
_TIMED_ITERS = 10000


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(
    rank: int,
    world_size: int,
    port: int,
    dtype_name: str,
    iters: int,
    threads: int,
    label: str,
) -> None:
    # Label + port makes the SHM group name unique across concurrent runs.
    os.environ["VLLM_DIST_IDENT"] = f"bench-{label}-tp{world_size}-{port}"

    # Set OMP thread count to match what a real vLLM worker would use.
    # Without this, torch defaults to the total machine core count and the
    # SHM manager is initialised with that inflated thread budget, making
    # adaptive-OMP comparisons meaningless and warmup extremely slow.
    torch.set_num_threads(threads)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    from vllm.platforms.cpu import CpuPlatform

    CpuPlatform.import_kernels()

    group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
    )

    comm = CpuCommunicator(
        cpu_group=group,
        device=torch.device("cpu"),
        device_group=group,
        unique_name="tp:0",
    )
    assert comm.supports_tensor_dict, (
        "SHM communicator not active — build vLLM with CPU SHM support"
    )

    dtype = getattr(torch, dtype_name)
    elem_size = dtype.itemsize
    line_elems = 64 // elem_size
    align = world_size * line_elems

    results: list[tuple[int, float, float, float]] = []

    for size_bytes in _SIZES_BYTES:
        numel = size_bytes // elem_size
        numel = (numel // align) * align
        if numel == 0:
            numel = align

        x = torch.ones(numel, dtype=dtype)

        # Warmup
        for _ in range(_WARMUP_ITERS):
            comm.all_reduce(x)
        dist.barrier(group)

        # Timed
        samples: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            comm.all_reduce(x)
            t1 = time.perf_counter_ns()
            samples.append(t1 - t0)
        dist.barrier(group)

        samples.sort()
        median_ns = samples[len(samples) // 2]
        p50_ns = median_ns
        p99_ns = samples[int(len(samples) * 0.99) - 1]
        results.append((size_bytes, median_ns, p50_ns, p99_ns))

    if rank == 0:
        print("size_bytes,ws,dtype,median_ns,p50_ns,p99_ns")
        for size_bytes, median_ns, p50_ns, p99_ns in results:
            print(
                f"{size_bytes},{world_size},{dtype_name},"
                f"{median_ns:.0f},{p50_ns:.0f},{p99_ns:.0f}"
            )

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=int, default=4, help="world size")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="bfloat16 or float32"
    )
    parser.add_argument("--iters", type=int, default=_TIMED_ITERS)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="OMP threads per rank (must match cores in each "
        "VLLM_CPU_OMP_THREADS_BIND slot)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Unique label embedded in SHM group name to prevent "
        "collisions between concurrent runs",
    )
    args = parser.parse_args()

    # Default threads: parse from VLLM_CPU_OMP_THREADS_BIND if available,
    # else fall back to total_cores / ws.
    if args.threads is None:
        bind = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", "")
        slots = [s for s in bind.split("|") if s]
        if slots:
            # Count cores in the first slot, e.g. "32-63" → 32.
            first = slots[0].strip()
            if "-" in first:
                lo, hi = first.split("-", 1)
                args.threads = int(hi) - int(lo) + 1
        if args.threads is None:
            import os as _os

            args.threads = max(1, (_os.cpu_count() or 1) // args.ws)

    # Unique label so two simultaneous invocations (baseline vs optimized)
    # never share a SHM group name.
    if args.label is None:
        args.label = uuid.uuid4().hex[:8]

    port = _find_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.ws, port, args.dtype, args.iters, args.threads, args.label),
        nprocs=args.ws,
        join=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark vLLM CPU collectives (SHM or Gloo) with one rank per NUMA/SNC node.

Launch with the companion shell script run_cpu_collectives_bench.sh, which
numactl-pins each rank to its NUMA node. Can also be launched via torchrun
if NUMA pinning is done externally.

Usage (via launcher):
    bash benchmarks/kernels/run_cpu_collectives_bench.sh --backend shm
    bash benchmarks/kernels/run_cpu_collectives_bench.sh --backend shm \\
        --collective reduce_scatter
    bash benchmarks/kernels/run_cpu_collectives_bench.sh --backend gloo \\
        --output-json bench_gloo.json

Usage (via torchrun, no NUMA pinning):
    torchrun --nproc_per_node=2 benchmarks/kernels/cpu/benchmark_cpu_collectives.py \\
        --backend shm
    torchrun --nproc_per_node=2 benchmarks/kernels/cpu/benchmark_cpu_collectives.py \\
        --backend shm --collective reduce_scatter
"""

import json
import os
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist

import vllm.utils.cpu_resource_utils as cr_utils
from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

# Sweep: input byte counts from 1 KiB to 128 MiB, with extra density in the
# 1–256 KiB region to make the flat-AR / RS+AG dispatch boundary visible.
_BYTES_TABLE = [
    1 * 1024,  # 1 KiB
    2 * 1024,  # 2 KiB
    4 * 1024,  # 4 KiB
    8 * 1024,  # 8 KiB
    16 * 1024,  # 16 KiB
    32 * 1024,  # 32 KiB
    64 * 1024,  # 64 KiB
    128 * 1024,  # 128 KiB
    256 * 1024,  # 256 KiB
    1 * 1024 * 1024,  # 1 MiB
    4 * 1024 * 1024,  # 4 MiB
    16 * 1024 * 1024,  # 16 MiB
    64 * 1024 * 1024,  # 64 MiB
    128 * 1024 * 1024,  # 128 MiB
]

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


def _numel_for_bytes(byte_count: int, dtype: torch.dtype) -> int:
    return byte_count // dtype.itemsize


def _init_dist() -> tuple[int, int]:
    """Initialize Gloo process group. torchrun sets the required env vars."""
    os.environ.setdefault("VLLM_DIST_IDENT", f"bench-{os.environ['MASTER_PORT']}")
    dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def _time_collective(
    comm: CpuCommunicator,
    numel: int,
    dtype: torch.dtype,
    warmup: int,
    trials: int,
    collective: str,
) -> dict[str, Any]:
    x = torch.empty(numel, dtype=dtype)
    byte_count = numel * dtype.itemsize

    if collective == "all_reduce":

        def timed_op() -> None:
            comm.all_reduce(x)

    elif collective == "reduce_scatter":

        def timed_op() -> None:
            comm.reduce_scatter(x, dim=0)

    else:  # all_gather

        def timed_op() -> None:
            comm.all_gather(x, dim=0)

    for _ in range(warmup):
        x.fill_(1.0)
        timed_op()
    dist.barrier()

    samples_ns: list[int] = []
    for _ in range(trials):
        x.fill_(1.0)
        t0 = time.perf_counter_ns()
        timed_op()
        samples_ns.append(time.perf_counter_ns() - t0)
    dist.barrier()

    samples_us = [s / 1000.0 for s in samples_ns]
    mean_us = statistics.mean(samples_us)
    p50_us = statistics.median(samples_us)
    sorted_samples = sorted(samples_us)
    p99_us = sorted_samples[int(len(sorted_samples) * 0.99) - 1]
    stdev_us = statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0

    mean_s = mean_us * 1e-6
    alg_bw_gbps = byte_count / mean_s / 1e9
    n = comm.world_size
    # all_reduce: 2*(N-1)/N  (NCCL bus-BW convention)
    # reduce_scatter / all_gather: (N-1)/N
    bus_factor = 2 * (n - 1) / n if collective == "all_reduce" else (n - 1) / n
    bus_bw_gbps = alg_bw_gbps * bus_factor

    return {
        "collective": collective,
        "dtype": str(dtype).replace("torch.", ""),
        "numel": numel,
        "bytes": byte_count,
        "mean_us": round(mean_us, 3),
        "p50_us": round(p50_us, 3),
        "p99_us": round(p99_us, 3),
        "stdev_us": round(stdev_us, 3),
        "alg_bw_gbps": round(alg_bw_gbps, 3),
        "bus_bw_gbps": round(bus_bw_gbps, 3),
    }


def _print_results(
    all_results: list[dict[str, Any]],
    collective: str,
    backend: str,
    world_size: int,
) -> None:
    print(f"\n{'=' * 120}")
    print(
        f"CPU Collective Benchmark  |  collective={collective}  |  "
        f"backend={backend}  |  world_size={world_size}"
    )
    print(f"{'=' * 120}")

    col_w = 14
    header = (
        f"{'dtype':<10}"
        f"{'bytes':<12}"
        f"{'mean_us':{col_w}}"
        f"{'p50_us':{col_w}}"
        f"{'p99_us':{col_w}}"
        f"{'stdev_us':{col_w}}"
        f"{'alg_bw(GB/s)':{col_w}}"
        f"{'bus_bw(GB/s)':{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        size_str = _human_bytes(r["bytes"])
        print(
            f"{r['dtype']:<10}"
            f"{size_str:<12}"
            f"{r['mean_us']:{col_w}.3f}"
            f"{r['p50_us']:{col_w}.3f}"
            f"{r['p99_us']:{col_w}.3f}"
            f"{r['stdev_us']:{col_w}.3f}"
            f"{r['alg_bw_gbps']:{col_w}.3f}"
            f"{r['bus_bw_gbps']:{col_w}.3f}"
        )

    print(f"{'=' * 120}")
    print("Times in microseconds (μs). Bandwidth in GB/s.")
    print(
        "bytes = input tensor bytes for all collectives.\n"
        "alg_bw = bytes / mean_time  (NCCL algorithmic bandwidth convention).\n"
        "bus_bw = alg_bw * 2*(N-1)/N  for all_reduce; "
        "alg_bw * (N-1)/N  for reduce_scatter / all_gather."
    )


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}TiB"


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM CPU collectives (SHM or Gloo)."
    )
    parser.add_argument(
        "--collective",
        choices=["all_reduce", "reduce_scatter", "all_gather"],
        default="all_reduce",
        help="Collective to benchmark (default: all_reduce).",
    )
    parser.add_argument(
        "--backend",
        choices=["shm", "gloo"],
        default="shm",
        help="Collective backend: 'shm' uses vLLM's shared-memory path; "
        "'gloo' falls back to torch.distributed Gloo.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per (dtype, size) pair.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of timed trials per (dtype, size) pair.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(_DTYPES.keys()),
        default=["bfloat16"],
        help="Dtypes to sweep. Default: bfloat16.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results to this JSON file (rank 0 only).",
    )
    args = parser.parse_args()

    # Load the CPU kernel .so (shm.cpp lives in _C / _C_AVX512 depending on ISA).
    # This mirrors what CpuPlatform.import_kernels() does; without it,
    # torch.ops._C.init_shm_manager is not registered when running outside the engine.
    from vllm.platforms.cpu import CpuPlatform

    CpuPlatform.import_kernels()

    rank, world_size = _init_dist()

    if rank == 0:
        numa_nodes = cr_utils.get_visible_memory_node()
        logger.info("Rank 0 NUMA nodes visible: %s", numa_nodes)
    else:
        numa_nodes = []

    cpu_group = dist.new_group(backend="gloo")

    # SHM path: unique_name must start with "tp" or "pp" (cpu_communicator.py:37-46).
    # Gloo path: any other prefix disables SHM and falls back to torch.distributed.
    unique_name = "tp:0" if args.backend == "shm" else "bench-gloo:0"
    comm = CpuCommunicator(
        cpu_group=cpu_group,
        device=torch.device("cpu"),
        device_group=cpu_group,
        unique_name=unique_name,
    )

    if args.backend == "shm" and not comm.supports_tensor_dict and rank == 0:
        # supports_tensor_dict is only True when _CPUSHMDistributed is active;
        # it encapsulates the combined arch + build precondition check.
        logger.warning(
            "SHM communicator not available (unsupported arch or world size). "
            "Falling back to Gloo for timing."
        )

    all_results: list[dict[str, Any]] = []
    for dtype_name in args.dtypes:
        dtype = _DTYPES[dtype_name]
        for byte_count in _BYTES_TABLE:
            numel = _numel_for_bytes(byte_count, dtype)
            if numel == 0:
                continue
            if (
                args.collective in ("reduce_scatter", "all_gather")
                and numel % world_size != 0
            ):
                continue

            if rank == 0:
                logger.info(
                    "Benchmarking collective=%s dtype=%s bytes=%s numel=%s",
                    args.collective,
                    dtype_name,
                    _human_bytes(byte_count),
                    numel,
                )

            result = _time_collective(
                comm, numel, dtype, args.num_warmup, args.num_trials, args.collective
            )
            all_results.append(result)

    if rank == 0:
        _print_results(all_results, args.collective, args.backend, world_size)

        if args.output_json:
            output = {
                "collective": args.collective,
                "backend": args.backend,
                "world_size": world_size,
                "numa_nodes": numa_nodes,
                "num_warmup": args.num_warmup,
                "num_trials": args.num_trials,
                "results": all_results,
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Results written to %s", args.output_json)

    if cpu_group != dist.group.WORLD:
        dist.destroy_process_group(cpu_group)


if __name__ == "__main__":
    main()

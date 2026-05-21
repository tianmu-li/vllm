#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark vLLM CPU all-reduce (SHM or Gloo) with one rank per NUMA/SNC node.

Launch with the companion shell script run_cpu_all_reduce_bench.sh, which
numactl-pins each rank to its NUMA node. Can also be launched via torchrun
if NUMA pinning is done externally.

Usage (via launcher):
    bash benchmarks/kernels/run_cpu_all_reduce_bench.sh --backend shm
    bash benchmarks/kernels/run_cpu_all_reduce_bench.sh --backend gloo \\
        --output bench_gloo.json

Usage (via torchrun, no NUMA pinning):
    torchrun --nproc_per_node=2 benchmark_cpu_all_reduce.py --backend shm
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

# Sweep: element counts (number of bfloat16 elements — bytes = numel * itemsize)
# 1 KiB to 128 MiB, with extra points in the 1–256 KiB crossover region so
# the flat-AR / RS+AG dispatch boundary is clearly visible.
_BF16_BYTES_TABLE = [
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


def _diagnose_shm(rank: int) -> None:
    """Log which SHM preconditions are met so failures are easy to triage."""
    from vllm.platforms import current_platform
    from vllm.platforms.interface import CpuArchEnum

    arch = current_platform.get_cpu_architecture()
    has_shm_op = hasattr(torch.ops._C, "init_shm_manager")
    dist_ident = os.environ.get("VLLM_DIST_IDENT", "<not set>")
    if rank == 0:
        logger.info(
            "SHM precondition check: arch=%s (need X86/ARM) | "
            "init_shm_manager present=%s | VLLM_DIST_IDENT=%r",
            arch,
            has_shm_op,
            dist_ident,
        )
    if not has_shm_op and rank == 0:
        logger.warning(
            "torch.ops._C.init_shm_manager not found — "
            "vLLM was not built with CPU SHM support "
            "(requires VLLM_USE_PRECOMPILED=0 build with CPU backend). "
            "SHM benchmark will fall back to Gloo."
        )
    if arch not in (CpuArchEnum.X86, CpuArchEnum.ARM) and rank == 0:
        logger.warning(
            "Architecture %s is not X86 or ARM — SHM is only supported on those.",
            arch,
        )


def _build_comm(
    cpu_group: dist.ProcessGroup, backend: str, rank: int
) -> CpuCommunicator:
    # SHM path: unique_name must start with "tp" or "pp" (cpu_communicator.py:37-46).
    # Gloo path: any other prefix disables SHM and falls back to torch.distributed.
    if backend == "shm":
        _diagnose_shm(rank)
    unique_name = "tp:0" if backend == "shm" else "bench-gloo:0"
    return CpuCommunicator(
        cpu_group=cpu_group,
        device=torch.device("cpu"),
        device_group=cpu_group,
        unique_name=unique_name,
    )


def _check_correctness(comm: CpuCommunicator, numel: int, dtype: torch.dtype) -> None:
    fill = 1.0
    x = torch.full((numel,), fill, dtype=dtype)
    comm.all_reduce(x)
    expected = torch.full((numel,), fill * comm.world_size, dtype=dtype)
    if not torch.allclose(x, expected, atol=1e-2, rtol=1e-2):
        raise RuntimeError(
            f"Correctness check failed for numel={numel}, dtype={dtype}. "
            f"max_err={torch.max(torch.abs(x - expected)).item()}"
        )


def _time_all_reduce(
    comm: CpuCommunicator,
    numel: int,
    dtype: torch.dtype,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    x = torch.empty(numel, dtype=dtype)

    for _ in range(warmup):
        x.fill_(1.0)
        comm.all_reduce(x)
    dist.barrier()

    samples_ns: list[int] = []
    for _ in range(trials):
        x.fill_(1.0)
        t0 = time.perf_counter_ns()
        comm.all_reduce(x)
        samples_ns.append(time.perf_counter_ns() - t0)
    dist.barrier()

    samples_us = [s / 1000.0 for s in samples_ns]
    mean_us = statistics.mean(samples_us)
    p50_us = statistics.median(samples_us)
    sorted_samples = sorted(samples_us)
    p99_us = sorted_samples[int(len(sorted_samples) * 0.99)]
    stdev_us = statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0

    byte_count = numel * dtype.itemsize
    mean_s = mean_us * 1e-6
    alg_bw_gbps = byte_count / mean_s / 1e9
    n = comm.world_size
    bus_bw_gbps = alg_bw_gbps * 2 * (n - 1) / n

    return {
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
    backend: str,
    world_size: int,
) -> None:
    print(f"\n{'=' * 120}")
    print(f"CPU All-Reduce Benchmark  |  backend={backend}  |  world_size={world_size}")
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
        "alg_bw = bytes / mean_time  (NCCL algorithmic bandwidth convention).\n"
        "bus_bw = alg_bw * 2*(N-1)/N  (per-link traffic, comparable across "
        "world sizes)."
    )


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}TiB"


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM CPU all-reduce (SHM or Gloo)."
    )
    parser.add_argument(
        "--backend",
        choices=["shm", "gloo"],
        default="shm",
        help="Collective backend: 'shm' uses vLLM's shared-memory path; "
        "'gloo' falls back to torch.distributed Gloo.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per (dtype, size) pair.",
    )
    parser.add_argument(
        "--trials",
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

    if rank == 0:
        logger.info(
            "Building CpuCommunicator with backend=%s, world_size=%s",
            args.backend,
            world_size,
        )
    comm = _build_comm(cpu_group, args.backend, rank)

    if args.backend == "shm" and not comm.supports_tensor_dict and rank == 0:
        # supports_tensor_dict is only True when _CPUSHMDistributed is active
        logger.warning(
            "SHM communicator not available (unsupported arch or world size). "
            "Falling back to Gloo for timing."
        )

    all_results: list[dict[str, Any]] = []
    for dtype_name in args.dtypes:
        dtype = _DTYPES[dtype_name]
        for byte_count in _BF16_BYTES_TABLE:
            numel = _numel_for_bytes(byte_count, dtype)
            if numel == 0:
                continue

            if rank == 0:
                logger.info(
                    "Benchmarking dtype=%s  bytes=%s  numel=%s",
                    dtype_name,
                    _human_bytes(byte_count),
                    numel,
                )

            try:
                _check_correctness(comm, numel, dtype)
            except RuntimeError as e:
                if rank == 0:
                    logger.error("Correctness check failed: %s — skipping size", e)
                dist.barrier()
                continue

            result = _time_all_reduce(comm, numel, dtype, args.warmup, args.trials)
            all_results.append(result)

    if rank == 0:
        _print_results(all_results, args.backend, world_size)

        if args.output_json:
            output = {
                "backend": args.backend,
                "world_size": world_size,
                "numa_nodes": numa_nodes,
                "warmup": args.warmup,
                "trials": args.trials,
                "results": all_results,
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Results written to %s", args.output_json)

    if cpu_group != dist.group.WORLD:
        dist.destroy_process_group(cpu_group)


if __name__ == "__main__":
    main()

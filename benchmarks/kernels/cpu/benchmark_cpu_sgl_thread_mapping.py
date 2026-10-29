# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and run the SGL dense thread mapping diagnostic.

The diagnostic records worker placement and rectangles, then checks exact-once
matrix-block coverage for legacy, logical96, and flat dense schedules.
Auto geometry and estimated A/B traffic come from the production callback and
its actual microtile loop. Quantization row ranges use the production helper
and report their own observed team, worker loads, and CPU placement.
Optional scheduler timings use an untimed warmup and no-log callbacks; they measure
callback and OpenMP scheduling overhead, not kernel performance.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import torch


def build_diagnostic(build_dir):
    """Compile the shared production scheduler for an untimed mapping probe."""
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("sgl_thread_mapping.cpp")
    common_header = root / "csrc/cpu/sgl-kernels/common.h"
    dense_header = root / "csrc/cpu/sgl-kernels/dense_scheduler.h"
    planner_header = dense_header.with_name("dense_scheduler_plan.h")
    source_paths = [
        source,
        common_header,
        dense_header,
        planner_header,
        root / "csrc/cpu/utils.hpp",
        root / "csrc/cpu/cpu_types.hpp",
    ]
    if platform.machine().lower() in ("x86_64", "amd64"):
        source_paths.append(root / "csrc/cpu/cpu_types_x86.hpp")
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    source_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    torch_dir = Path(torch.__file__).parent
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "sgl_thread_mapping"
    command = [
        os.environ.get("CXX", "g++"),
        "-O2",
        "-std=c++20",
        "-fopenmp",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"-I{root / 'csrc'}",
        f"-I{common_header.parent}",
        f"-I{torch_dir / 'include'}",
        f"-I{torch_dir / 'include/torch/csrc/api/include'}",
        str(source),
        f"-L{torch_dir / 'lib'}",
        f"-Wl,-rpath,{torch_dir / 'lib'}",
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(executable),
    ]
    if platform.machine().lower() in ("x86_64", "amd64"):
        command[1:1] = ["-mavx2", "-mfma", "-mf16c"]
    subprocess.run(command, check=True)
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest()
        != source_hashes[str(path.relative_to(root))]
        for path in source_paths
    ):
        raise RuntimeError("scheduler diagnostic sources changed during compilation")
    return executable, {
        "diagnostic_source_sha256": source_digest,
        "diagnostic_source_file_sha256": source_hashes,
        "diagnostic_compile_command": command,
        "torch_version": torch.__version__,
    }


def run_diagnostic(
    executable,
    *,
    scheduler,
    threads,
    m,
    n,
    k,
    panels=1,
    scratch_bytes=4096,
    output_bytes=2,
    fused_input_bytes=0,
    has_bias=False,
    cache_bytes=0,
    thread_limit=None,
    timing=False,
):
    """Observe a production callback in a separate process with matching env."""
    environment = os.environ.copy()
    environment["VLLM_CPU_SGL_SCHEDULER"] = scheduler
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    if thread_limit is not None:
        environment["OMP_THREAD_LIMIT"] = str(thread_limit)
    native_args = [
        str(executable),
        str(threads),
        str((m + 31) // 32),
        str((n + 31) // 32),
        scheduler,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--panels",
        str(panels),
        "--scratch-bytes",
        str(scratch_bytes),
        "--output-bytes",
        str(output_bytes),
        "--fused-input-bytes",
        str(fused_input_bytes),
        "--cache-bytes",
        str(cache_bytes),
    ]
    if has_bias:
        native_args.append("--has-bias")
    if timing:
        native_args.append("--timing")
    result = subprocess.run(
        native_args,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    quantization = report["quantization"]
    row_counts = quantization["rows_per_worker"]
    nonempty_counts = [count for count in row_counts if count]
    maximum = max(row_counts, default=0)
    minimum_nonempty = min(nonempty_counts, default=0)
    mean = sum(row_counts) / len(row_counts) if row_counts else 0
    quantization.update(
        minimum_rows=min(row_counts, default=0),
        maximum_rows=maximum,
        minimum_nonempty_rows=minimum_nonempty,
        nonempty_workers=len(nonempty_counts),
        max_to_mean_ratio=maximum / mean if mean else None,
        max_to_min_nonempty_ratio=maximum / minimum_nonempty
        if minimum_nonempty
        else None,
    )
    report.update(
        configured_threads=threads,
        actual_m=m,
        actual_n=n,
        k=k,
        panels=panels,
        configured_thread_limit=thread_limit or environment.get("OMP_THREAD_LIMIT"),
        diagnostic_command=native_args,
        diagnostic_environment={
            key: value
            for key, value in environment.items()
            if key.startswith(("OMP_", "KMP_", "GOMP_", "VLLM_CPU_"))
            or key == "LD_PRELOAD"
        },
        diagnostic_cpu_affinity=sorted(os.sched_getaffinity(0)),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler",
        choices=(
            "legacy",
            "logical96_dense",
            "flat_32x32",
            "flat_128x64",
            "flat_128x128",
            "flat_auto",
        ),
        default="logical96_dense",
    )
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--panels", type=int, default=1)
    parser.add_argument("--scratch-bytes", type=int, default=4096)
    parser.add_argument("--output-bytes", type=int, default=2)
    parser.add_argument("--fused-input-bytes", type=int, default=0)
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=0,
        help=(
            "flat_auto active-window budget override; zero uses half the detected L2, "
            "and ownership capacity is twice this budget"
        ),
    )
    parser.add_argument(
        "--m-blocks",
        type=int,
        default=1,
        help="number of M microtiles (32 rows each)",
    )
    parser.add_argument(
        "--m",
        type=int,
        help="actual M rows; defaults to --m-blocks times 32",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=192,
        help="number of N microtiles (32 columns each)",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="actual N columns; defaults to --n-blocks times 32",
    )
    parser.add_argument(
        "--thread-limit",
        type=int,
        help="set OMP_THREAD_LIMIT to exercise a smaller actual team",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="measure mean scheduler overhead across 100 no-log invocations",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=Path("/tmp/vllm-sgl-thread-mapping")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.threads <= 0
        or args.k <= 0
        or args.panels <= 0
        or args.m_blocks < 0
        or args.n_blocks < 0
    ):
        parser.error(
            "threads, K, and panels must be positive; block counts must be nonnegative"
        )
    actual_m = args.m if args.m is not None else args.m_blocks * 32
    if actual_m < 0:
        parser.error("--m must be nonnegative")
    if (actual_m + 31) // 32 != args.m_blocks:
        parser.error("--m-blocks must equal ceil(--m / 32)")
    actual_n = args.n if args.n is not None else args.n_blocks * 32
    if actual_n < 0:
        parser.error("--n must be nonnegative")
    if (actual_n + 31) // 32 != args.n_blocks:
        parser.error("--n-blocks must equal ceil(--n / 32)")
    if args.thread_limit is not None and args.thread_limit <= 0:
        parser.error("--thread-limit must be positive")
    if (
        min(
            args.scratch_bytes,
            args.output_bytes,
            args.fused_input_bytes,
            args.cache_bytes,
        )
        < 0
    ):
        parser.error("footprint byte counts must be nonnegative")
    executable, source_metadata = build_diagnostic(args.build_dir)
    report = run_diagnostic(
        executable,
        scheduler=args.scheduler,
        threads=args.threads,
        m=actual_m,
        n=actual_n,
        k=args.k,
        panels=args.panels,
        scratch_bytes=args.scratch_bytes,
        output_bytes=args.output_bytes,
        fused_input_bytes=args.fused_input_bytes,
        has_bias=args.has_bias,
        cache_bytes=args.cache_bytes,
        thread_limit=args.thread_limit,
        timing=args.timing,
    )
    report.update(source_metadata)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

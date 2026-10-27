# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and run the SGL dense thread mapping diagnostic.

The diagnostic records worker placement and rectangles, then checks exact-once
matrix-block coverage for legacy, logical96, and flat dense schedules. Optional
scheduler timings use an untimed warmup and no-log callbacks; they measure
callback and OpenMP scheduling overhead, not kernel performance.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


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
        ),
        default="logical96_dense",
    )
    parser.add_argument("--threads", type=int, default=96)
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
    if args.threads <= 0 or args.m_blocks < 0 or args.n_blocks < 0:
        parser.error("threads must be positive and block counts nonnegative")
    actual_m = args.m if args.m is not None else args.m_blocks * 32
    if actual_m < 0:
        parser.error("--m must be nonnegative")
    if (actual_m + 31) // 32 != args.m_blocks:
        parser.error("--m-blocks must equal ceil(--m / 32)")
    if args.thread_limit is not None and args.thread_limit <= 0:
        parser.error("--thread-limit must be positive")

    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("sgl_thread_mapping.cpp")
    common_header = root / "csrc/cpu/sgl-kernels/common.h"
    dense_header = root / "csrc/cpu/sgl-kernels/dense_scheduler.h"
    torch_dir = Path(torch.__file__).parent
    args.build_dir.mkdir(parents=True, exist_ok=True)
    executable = args.build_dir / "sgl_thread_mapping"
    command = [
        os.environ.get("CXX", "g++"),
        "-O2",
        "-std=c++20",
        "-fopenmp",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
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
    subprocess.run(command, check=True)

    environment = os.environ.copy()
    environment["VLLM_CPU_SGL_SCHEDULER"] = args.scheduler
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    if args.thread_limit is not None:
        environment["OMP_THREAD_LIMIT"] = str(args.thread_limit)
    native_args = [
        str(executable),
        str(args.threads),
        str(args.m_blocks),
        str(args.n_blocks),
        args.scheduler,
    ]
    if args.m is not None:
        native_args.extend(("--m", str(actual_m)))
    if args.timing:
        native_args.append("--timing")
    result = subprocess.run(
        native_args,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    report.update(
        configured_threads=args.threads,
        actual_m=actual_m,
        configured_thread_limit=args.thread_limit
        or environment.get("OMP_THREAD_LIMIT"),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        common_sha256=hashlib.sha256(common_header.read_bytes()).hexdigest(),
        dense_scheduler_sha256=hashlib.sha256(dense_header.read_bytes()).hexdigest(),
        compile_command=command,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

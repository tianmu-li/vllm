# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and run the native SGL work-assignment diagnostic.

Run with the serving LD_PRELOAD and affinity settings to inspect real CPU
placement. Dense and synthetic routed-MoE modes report scheduler plans,
physical CPU placement, tile ownership, estimated work balance, and exact-once
matrix-block coverage. Optional timings cover planning and worker completion;
they are diagnostic only, not kernel performance measurements. Estimated
panel-byte counts do not measure DRAM traffic or identify memory homes.
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
    parser.add_argument("--threads", type=int, default=90)
    parser.add_argument("--groups", default="")
    parser.add_argument(
        "--scheduler",
        choices=(
            "",
            "legacy",
            "round_robin",
            "spread_moe",
            "spread_dense",
            "spread_all",
            "tile_moe",
            "tile_dense",
            "tile_all",
        ),
        default="",
        help="scheduler override; empty preserves the native automatic behavior",
    )
    parser.add_argument(
        "--kind",
        choices=("dense", "moe"),
        default="dense",
        help="diagnostic plan to inspect",
    )
    parser.add_argument(
        "--phase",
        choices=("gate", "down"),
        help="MoE phase; down uses K as its output width",
    )
    parser.add_argument(
        "--m-blocks",
        type=int,
        default=256,
        help="dense M blocks; MoE expert count when rows are omitted",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=16,
        help="default column blocks; also supplies MoE intermediate N",
    )
    parser.add_argument("--rows", type=int, help="dense row count")
    parser.add_argument(
        "--columns",
        type=int,
        help="column count or MoE intermediate N",
    )
    parser.add_argument("--k", type=int, default=2048, help="reduction width")
    parser.add_argument(
        "--expert-rows",
        help="comma-separated routed rows per expert; zero-row experts are allowed",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="include plan-build and worker-finish wall times in the report",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=Path("/tmp/vllm-sgl-thread-mapping")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.kind == "moe" and args.rows is not None:
        parser.error("--rows applies to dense; use --expert-rows for MoE")
    if args.kind == "dense" and (
        args.phase is not None or args.expert_rows is not None
    ):
        parser.error("--phase and --expert-rows apply only to MoE")
    if args.kind == "dense" and args.rows is not None and args.rows <= 0:
        parser.error("--rows must be positive")
    if args.columns is not None and args.columns <= 0:
        parser.error("--columns must be positive")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.m_blocks <= 0 or args.n_blocks <= 0 or args.threads <= 0:
        parser.error("thread and block counts must be positive")
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("sgl_thread_mapping.cpp")
    common_header = root / "csrc/cpu/sgl-kernels/common.h"
    scheduler_header = root / "csrc/cpu/sgl-kernels/tile_scheduler.h"
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
    environment["VLLM_CPU_SGL_THREAD_GROUPS"] = args.groups
    environment["VLLM_CPU_SGL_SCHEDULER"] = args.scheduler
    native_args = [
        str(executable),
        str(args.threads),
        str(args.m_blocks),
        str(args.n_blocks),
        "--kind",
        args.kind,
        "--k",
        str(args.k),
    ]
    if args.phase is not None:
        native_args.extend(("--phase", args.phase))
    if args.rows is not None:
        native_args.extend(("--rows", str(args.rows)))
    if args.columns is not None:
        native_args.extend(("--columns", str(args.columns)))
    if args.expert_rows is not None:
        native_args.extend(("--expert-rows", args.expert_rows))
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
        groups=args.groups,
        configured_scheduler=args.scheduler,
        configured_kind=args.kind,
        configured_phase=args.phase or ("gate" if args.kind == "moe" else None),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        common_sha256=hashlib.sha256(common_header.read_bytes()).hexdigest(),
        tile_scheduler_sha256=hashlib.sha256(scheduler_header.read_bytes()).hexdigest(),
        compile_command=command,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

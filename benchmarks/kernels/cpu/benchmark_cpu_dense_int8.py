# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark warm and cold CPU INT8 dense projections."""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=92)
parser.add_argument(
    "--scheduler",
    choices=(
        "legacy",
        "logical96_dense",
        "flat_32x32",
        "flat_128x64",
        "flat_128x128",
    ),
    default="legacy",
)
selection = parser.add_mutually_exclusive_group()
selection.add_argument(
    "--case",
    action="append",
    metavar="PROJECTION:M",
    help="run one projection at M rows; may be repeated",
)
selection.add_argument(
    "--projection",
    choices=(
        "qkv",
        "gdn_qkvz",
        "output",
        "gdn_output",
        "mtp_fc",
        "shared_ordinary_gateup",
        "shared_down",
    ),
    action="append",
)
parser.add_argument("--m-values", type=int, nargs="+")
DEFAULT_M_VALUES = (1, 32, 128, 1268)
DEFAULT_PROJECTIONS = ("qkv", "output")
parser.add_argument(
    "--output-dir", type=Path, default=Path(__file__).with_name("results")
)
parser.add_argument("--output", type=Path)
parser.add_argument("--samples", type=int, default=3)
parser.add_argument("--sample-ms", type=float, default=20)
args = parser.parse_args()

if args.threads < 1:
    parser.error("--threads must be positive")
if args.samples < 1:
    parser.error("--samples must be positive")
if args.sample_ms <= 0:
    parser.error("--sample-ms must be positive")
if args.case and args.m_values is not None:
    parser.error("--case cannot be combined with --m-values")
if args.m_values is not None and any(m < 1 for m in args.m_values):
    parser.error("--m-values must be positive")

os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OMP_THREAD_LIMIT"] = str(args.threads)
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["VLLM_CPU_SGL_SCHEDULER"] = args.scheduler

import torch  # noqa: E402

import vllm._custom_ops  # noqa: E402, F401

PROJECTIONS = {
    "qkv": (2048, 9216),
    "gdn_qkvz": (2048, 12288),
    "output": (4096, 2048),
    "gdn_output": (4096, 2048),
    "mtp_fc": (4096, 2048),
    "shared_ordinary_gateup": (2048, 1024),
    "shared_down": (512, 2048),
}
FLAT_SCHEDULE_GEOMETRY = {
    "flat_32x32": (1, 1),
    "flat_128x64": (4, 2),
    "flat_128x128": (4, 4),
}
TARGET_WEIGHT_BYTES = int(1.5 * 1024**3)
WEIGHT_CHUNK_SIZE = 8


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def format_cpu_set(cpus):
    ranges = []
    start = previous = None
    for cpu in sorted(cpus):
        if start is None:
            start = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = cpu
    if start is not None:
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def runtime_metadata():
    extension = importlib.import_module("vllm._C")
    extension_path = Path(extension.__file__).resolve()
    digest = hashlib.sha256()
    with extension_path.open("rb") as library:
        for block in iter(lambda: library.read(1024 * 1024), b""):
            digest.update(block)
    cpus = os.sched_getaffinity(0)
    return {
        "cpu": cpu_model(),
        "cpu_affinity": format_cpu_set(cpus),
        "allowed_cpus": sorted(cpus),
        "requested_threads": args.threads,
        "torch_threads": torch.get_num_threads(),
        "scheduler": args.scheduler,
        "selected_schedule": args.scheduler,
        "seed": 42,
        "samples": args.samples,
        "sample_ms": args.sample_ms,
        "command": sys.argv,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("OMP_", "KMP_", "GOMP_", "VLLM_CPU_"))
        },
        "extension_path": str(extension_path),
        "extension_sha256": digest.hexdigest(),
        "torch_version": torch.__version__,
    }


def schedule_metadata(m, n):
    m_blocks = math.ceil(m / 32)
    n_blocks = math.ceil(n / 32)
    metadata = {
        "selected_schedule": args.scheduler,
        "effective_schedule": args.scheduler,
        "legacy_fallback": False,
        "legacy_schedule_active": args.scheduler == "legacy",
        "legacy_expected_team_size": None,
        "legacy_fallback_team_requirement": None,
        "legacy_team_requirement": None,
        "m_blocks": m_blocks,
        "n_blocks": n_blocks,
        "flat_superblock_task_count": None,
        "scheduler_eligible": None,
        "eligibility_reason": "not_applicable",
    }
    if args.scheduler not in FLAT_SCHEDULE_GEOMETRY:
        if args.scheduler == "legacy":
            metadata.update(
                legacy_expected_team_size=(
                    args.threads if m_blocks == 1 else max(1, (args.threads // 2) * 2)
                ),
                legacy_team_requirement=(
                    "actual OpenMP team must match the legacy requested team"
                ),
            )
        return metadata

    bm, bn = FLAT_SCHEDULE_GEOMETRY[args.scheduler]
    task_count = math.ceil(m_blocks / bm) * math.ceil(n_blocks / bn)
    eligible = m >= 128 and (bm == 1 or task_count >= 192)
    if m < 128:
        reason = "M_below_128"
    elif not eligible:
        reason = "superblock_tasks_below_192"
    else:
        reason = "eligible"
    effective_schedule = args.scheduler if eligible else "legacy"
    legacy_schedule_active = effective_schedule == "legacy"
    metadata.update(
        effective_schedule=effective_schedule,
        legacy_fallback=not eligible,
        legacy_schedule_active=legacy_schedule_active,
        legacy_expected_team_size=(
            args.threads if m_blocks == 1 else max(1, (args.threads // 2) * 2)
        )
        if legacy_schedule_active
        else None,
        legacy_fallback_team_requirement=(
            "actual OpenMP team must match the legacy requested team"
            if not eligible
            else None
        ),
        legacy_team_requirement=(
            "actual OpenMP team must match the legacy requested team"
            if legacy_schedule_active
            else None
        ),
        flat_superblock_task_count=task_count,
        scheduler_eligible=eligible,
        eligibility_reason=reason,
        flat_superblock_m_blocks=bm,
        flat_superblock_n_blocks=bn,
    )
    return metadata


def selected_cases():
    cases = {}
    if args.case:
        for case in args.case:
            try:
                projection, value = case.rsplit(":", 1)
                m = int(value)
            except ValueError:
                parser.error(f"invalid --case {case!r}; expected PROJECTION:M")
            if projection not in PROJECTIONS or m < 1:
                parser.error(
                    f"invalid --case {case!r}; expected a known projection and M > 0"
                )
            cases.setdefault(projection, []).append(m)
        return cases

    projections = args.projection or DEFAULT_PROJECTIONS
    m_values = args.m_values or DEFAULT_M_VALUES
    for projection in projections:
        cases[projection] = list(m_values)
    return cases


torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
torch.cpu._init_amx()
metadata = runtime_metadata()
results = []
cases_by_projection = selected_cases()
args.output_dir.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    for projection, m_values in cases_by_projection.items():
        k, n = PROJECTIONS[projection]
        packed_matrix_bytes = n * (k + 4)
        depth = 1 + math.ceil(TARGET_WEIGHT_BYTES / packed_matrix_bytes)
        packed = torch.empty((depth, n, k + 4), dtype=torch.int8)
        raw_first = None

        for start in range(0, depth, WEIGHT_CHUNK_SIZE):
            count = min(WEIGHT_CHUNK_SIZE, depth - start)
            raw_chunk = torch.randint(-32, 33, (count, n, k), dtype=torch.int8)
            if raw_first is None:
                raw_first = raw_chunk[0].clone()
            packed_chunk = torch.ops._C.convert_weight_packed(raw_chunk)
            packed[start : start + count].copy_(packed_chunk)
            del raw_chunk, packed_chunk

        packed_matrix_bytes = packed[0].numel() * packed.element_size()
        packed_bank_bytes = packed.numel() * packed.element_size()
        intervening_bytes = (depth - 1) * packed_matrix_bytes
        weight_pointers = [packed[i].data_ptr() for i in range(depth)]
        assert packed_bank_bytes == depth * packed_matrix_bytes
        assert len(set(weight_pointers)) == depth
        assert all(
            weight_pointers[i + 1] - weight_pointers[i] == packed_matrix_bytes
            for i in range(depth - 1)
        )
        assert intervening_bytes >= TARGET_WEIGHT_BYTES

        scales = torch.full((n,), 0.001, dtype=torch.float32)

        for m in m_values:
            x = torch.randn(m, k).bfloat16()

            def run(weight, x=x, scales=scales):
                return torch.ops._C.int8_scaled_mm_with_quant(
                    x, weight, scales, None, x.dtype, True
                )

            actual = run(packed[0])
            reference_rows = min(m, 4)
            reference = (
                x[:reference_rows].float() @ (raw_first.float() * scales[:, None]).t()
            )
            relative_l2 = float(
                (actual[:reference_rows].float() - reference).norm() / reference.norm()
            )
            assert relative_l2 < 0.03, relative_l2
            del actual, reference

            for _ in range(3):
                run(packed[0])
            start_time = time.perf_counter_ns()
            run(packed[0])
            warm_probe_ms = (time.perf_counter_ns() - start_time) / 1e6
            warm_repetitions = max(1, math.ceil(args.sample_ms / warm_probe_ms))
            warm_samples_ms = []
            for _ in range(args.samples):
                start_time = time.perf_counter_ns()
                for _ in range(warm_repetitions):
                    run(packed[0])
                warm_samples_ms.append(
                    (time.perf_counter_ns() - start_time) / 1e6 / warm_repetitions
                )

            cold_index = [0]

            def run_cold(packed=packed, cold_index=cold_index, depth=depth, run=run):
                weight = packed[cold_index[0]]
                cold_index[0] = (cold_index[0] + 1) % depth
                return run(weight)

            for _ in range(depth):
                run_cold()
            start_time = time.perf_counter_ns()
            run_cold()
            cold_probe_ms = (time.perf_counter_ns() - start_time) / 1e6
            cold_repetitions = max(
                depth,
                math.ceil(args.sample_ms / cold_probe_ms / depth) * depth,
            )
            assert cold_repetitions % depth == 0
            cold_samples_ms = []
            for _ in range(args.samples):
                start_time = time.perf_counter_ns()
                for _ in range(cold_repetitions):
                    run_cold()
                cold_samples_ms.append(
                    (time.perf_counter_ns() - start_time) / 1e6 / cold_repetitions
                )

            row = {
                "projection": projection,
                "m": m,
                "k": k,
                "n": n,
                **schedule_metadata(m, n),
                "weight_bank_bytes": packed_bank_bytes,
                "packed_weight_matrix_bytes": packed_matrix_bytes,
                "rotating_weight_matrices": depth,
                "cold_revisit_distinct_intervening_packed_bytes": intervening_bytes,
                "cold_repetitions_per_sample": cold_repetitions,
                "cold_rotation_continues_across_sample_boundaries": True,
                "cold_bank_ranges_are_disjoint": True,
                "warm_median_ms": statistics.median(warm_samples_ms),
                "warm_samples_ms": warm_samples_ms,
                "warm_repetitions_per_sample": warm_repetitions,
                "cold_median_ms": statistics.median(cold_samples_ms),
                "cold_samples_ms": cold_samples_ms,
                "correctness_relative_l2": relative_l2,
            }
            print(row, flush=True)
            results.append(row)
            del x, run, run_cold

        del raw_first, packed, scales

output = args.output or (
    args.output_dir
    / (
        f"dense_int8_{args.scheduler}_{args.threads}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            **metadata,
            "working_set_target_bytes": TARGET_WEIGHT_BYTES,
            "results": results,
            "note": (
                "Packed weight bank construction and correctness checks are "
                "outside timed intervals. Each cold sample uses full operator "
                "calls and rotates through disjoint packed-weight ranges; the "
                "rotation index continues across sample boundaries. Cold "
                "revisit distance excludes the revisited matrix. This audit "
                "tracks packed bytes and is not a DRAM counter."
            ),
        },
        indent=2,
    )
    + "\n"
)
print(f"wrote {output}")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark warm and cold CPU INT8 dense projections.

Use cold latency to qualify scheduler changes.
"""

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import mmap
import os
import platform
import resource
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
        "flat_auto",
        "single_shape",
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
    "--shape",
    metavar="M:K:N",
    help="run one custom dense shape with positive M and K/N multiples of 32",
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
parser.add_argument(
    "--scheduler-diagnostic-build-dir",
    type=Path,
    default=Path("/tmp/vllm-sgl-thread-mapping"),
    help="build directory for the untimed flat_auto shared-planner preflight",
)
args = parser.parse_args()

if args.threads < 1:
    parser.error("--threads must be positive")
if args.samples < 1:
    parser.error("--samples must be positive")
if args.sample_ms <= 0:
    parser.error("--sample-ms must be positive")
if (args.case or args.shape) and args.m_values is not None:
    parser.error("--case and --shape cannot be combined with --m-values")
if args.m_values is not None and any(m < 1 for m in args.m_values):
    parser.error("--m-values must be positive")
custom_shape = None
if args.shape:
    try:
        m, k, n = (int(dimension) for dimension in args.shape.split(":"))
    except ValueError:
        parser.error(f"invalid --shape {args.shape!r}; expected M:K:N")
    if m < 1 or k < 1 or n < 1:
        parser.error(f"invalid --shape {args.shape!r}; dimensions must be positive")
    if k % 32 != 0 or n % 32 != 0:
        parser.error(f"invalid --shape {args.shape!r}; K and N must be multiples of 32")
    custom_shape = (m, k, n)

os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OMP_THREAD_LIMIT"] = str(args.threads)
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["VLLM_CPU_SGL_SCHEDULER"] = args.scheduler
TORCH_ALREADY_IMPORTED = "torch" in sys.modules
if TORCH_ALREADY_IMPORTED and "THP_MEM_ALLOC_ENABLE" not in os.environ:
    parser.error("set THP_MEM_ALLOC_ENABLE before Python starts")
os.environ.setdefault("THP_MEM_ALLOC_ENABLE", "1")
STARTUP_CPU_AFFINITY = os.sched_getaffinity(0)

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
RESOURCE_USAGE_FIELDS = (
    "ru_utime",
    "ru_stime",
    "ru_minflt",
    "ru_majflt",
    "ru_nvcsw",
    "ru_nivcsw",
)


def numa_node_for_cpu(cpu):
    node_paths = Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*")
    nodes = {int(path.name.removeprefix("node")) for path in node_paths}
    if len(nodes) != 1:
        raise RuntimeError(f"could not identify one NUMA node for CPU {cpu}: {nodes}")
    return nodes.pop()


def read_smaps_mappings(address, tensor_bytes):
    mappings = []
    mapping = None
    rows = Path("/proc/self/smaps").read_text().splitlines()
    for line in rows:
        fields = line.split(maxsplit=5)
        if "-" in fields[0]:
            start, end = (int(value, 16) for value in fields[0].split("-"))
            overlap = max(0, min(end, address + tensor_bytes) - max(start, address))
            mapping = None
            if overlap:
                mapping = {
                    "range": fields[0],
                    "name": fields[5] if len(fields) > 5 else "",
                    "tensor_overlap_bytes": overlap,
                }
                mappings.append(mapping)
            continue
        key, _, value = line.partition(":")
        if mapping is not None and key in (
            "Size",
            "Rss",
            "KernelPageSize",
            "MMUPageSize",
            "AnonHugePages",
            "ShmemPmdMapped",
            "FilePmdMapped",
            "THPeligible",
            "VmFlags",
        ):
            mapping[key] = value.strip()

    if sum(m["tensor_overlap_bytes"] for m in mappings) != tensor_bytes:
        raise RuntimeError("the tensor range is not fully covered by /proc/self/smaps")
    for mapping in mappings:
        size_bytes = int(mapping["Size"].split()[0]) * 1024
        huge_bytes = 1024 * sum(
            int(mapping[key].split()[0])
            for key in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped")
        )
        mapping["guaranteed_tensor_huge_bytes"] = max(
            0, huge_bytes - (size_bytes - mapping["tensor_overlap_bytes"])
        )
        mapping["mapping_aggregates_allocations"] = True
    return mappings


def read_numa_mapping(mapping_start):
    rows = Path("/proc/self/numa_maps").read_text().splitlines()
    for row in rows:
        tokens = row.split()
        if int(tokens[0], 16) == mapping_start:
            return {
                "policy": tokens[1],
                "node_pages": {
                    token.split("=", 1)[0]: int(token.split("=", 1)[1])
                    for token in tokens[2:]
                    if token.startswith("N") and "=" in token
                },
            }
    raise RuntimeError(f"mapping {mapping_start:x} is absent from /proc/self/numa_maps")


def memory_setup_metadata(tensor, *, expected_node, require_huge_pages):
    tensor_bytes = tensor.numel() * tensor.element_size()
    mappings = read_smaps_mappings(tensor.data_ptr(), tensor_bytes)
    expected_policy = f"bind:{expected_node}"
    for mapping in mappings:
        mapping.update(read_numa_mapping(int(mapping["range"].split("-")[0], 16)))
        if mapping["policy"] != expected_policy or any(
            count
            for node, count in mapping["node_pages"].items()
            if node != f"N{expected_node}"
        ):
            raise RuntimeError("tensor mappings must be bound and resident on one node")
    coverage = sum(m["guaranteed_tensor_huge_bytes"] for m in mappings) / tensor_bytes
    if require_huge_pages and coverage < 0.95:
        raise RuntimeError(
            "the packed-weight tensor has less than 95% guaranteed huge-page "
            "backing; start the benchmark with THP_MEM_ALLOC_ENABLE=1 and "
            "bind its CPUs and memory to one NUMA node"
        )
    return {
        "tensor_bytes": tensor_bytes,
        "huge_page_coverage_lower_bound": coverage,
        "mappings": mappings,
    }


def preload_metadata():
    preload = os.environ.get("LD_PRELOAD", "")
    loaded_paths = {
        Path(row.split()[-1]).resolve()
        for row in Path("/proc/self/maps").read_text().splitlines()
        if len(row.split()) >= 6 and row.split()[-1].startswith("/")
    }
    required = ["libtcmalloc"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        required.append("libiomp")
    configured_paths = [Path(path) for path in preload.split(":") if path]
    missing_configuration = [
        library
        for library in required
        if not any(
            library in path.name and path.is_absolute() and path.is_file()
            for path in configured_paths
        )
    ]
    missing = [
        library
        for library in required
        if not any(
            library in path.name and path.resolve() in loaded_paths
            for path in configured_paths
        )
    ]
    if missing or missing_configuration:
        raise RuntimeError(
            "CPU kernel benchmarks require absolute tcmalloc and Intel "
            "OpenMP libraries in LD_PRELOAD before Python starts; "
            "unloaded="
            + ", ".join(missing)
            + "; missing absolute paths="
            + ", ".join(missing_configuration)
        )
    preload_files = []
    for path in preload.split(":"):
        if path and Path(path).is_file():
            library = Path(path).resolve()
            digest = hashlib.sha256(library.read_bytes()).hexdigest()
            preload_files.append({"path": str(library), "sha256": digest})
    return {
        "environment": preload,
        "preloaded_file_sha256": preload_files,
        "loaded_libraries": sorted(
            str(path)
            for path in loaded_paths
            if any(name in path.name for name in required)
        ),
    }


def numa_setup_metadata(threads):
    proclist = (
        os.environ.get("KMP_AFFINITY", "").partition("proclist=[")[2].partition("]")[0]
    )
    cpus = set()
    if proclist:
        for part in proclist.split(","):
            bounds = [int(value) for value in part.split("-")]
            cpus.update(range(bounds[0], bounds[-1] + 1))
    elif TORCH_ALREADY_IMPORTED:
        raise RuntimeError(
            "use an explicit KMP_AFFINITY proclist for staged benchmarks"
        )
    else:
        cpus = STARTUP_CPU_AFFINITY
    node_by_cpu = {cpu: numa_node_for_cpu(cpu) for cpu in cpus}
    nodes = sorted(set(node_by_cpu.values()))
    if len(nodes) != 1:
        raise RuntimeError(
            "bind the benchmark process CPUs and memory to one NUMA node; "
            f"the current CPU affinity spans nodes {nodes}"
        )
    if len(cpus) < threads:
        raise RuntimeError(
            f"thread count {threads} exceeds CPUs in process affinity ({len(cpus)})"
        )
    node = nodes[0]
    return {
        "cpu_affinity": format_cpu_set(cpus),
        "allowed_cpus": sorted(cpus),
        "source": "KMP_AFFINITY proclist" if proclist else "script-entry affinity",
        "numa_node": node,
        "numa_policy": f"expected bind:{node} from numactl --membind",
    }


def resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {name: getattr(usage, name) for name in RESOURCE_USAGE_FIELDS}


def allocate_packed_bank(shape):
    if os.environ["THP_MEM_ALLOC_ENABLE"] != "1":
        return torch.empty(shape, dtype=torch.int8)
    alignment = 2 * 1024**2
    tensor_bytes = math.prod(shape)
    rounded_bytes = math.ceil(tensor_bytes / alignment) * alignment
    allocation = mmap.mmap(
        -1,
        rounded_bytes + 2 * alignment,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    base = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
    offset = (base + mmap.PAGESIZE + alignment - 1) // alignment * alignment - base
    protect = ctypes.CDLL(None, use_errno=True).mprotect
    protect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    protect.restype = ctypes.c_int
    for address, size in (
        (base, offset),
        (
            base + offset + rounded_bytes,
            len(allocation) - offset - rounded_bytes,
        ),
    ):
        if protect(address, size, 0):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    allocation.madvise(mmap.MADV_HUGEPAGE, offset, rounded_bytes)
    # The tensor retains the mmap owner; guards isolate its VMA from the heap.
    return torch.frombuffer(
        allocation, dtype=torch.int8, count=tensor_bytes, offset=offset
    ).reshape(shape)


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
    cpus = STARTUP_CPU_AFFINITY
    master_cpus = os.sched_getaffinity(0)
    memory_placement = numa_setup_metadata(args.threads)
    preload = preload_metadata()
    return {
        "cpu": cpu_model(),
        "cpu_affinity": memory_placement["cpu_affinity"],
        "allowed_cpus": memory_placement["allowed_cpus"],
        "script_entry_thread_affinity": format_cpu_set(cpus),
        "master_cpu_affinity_after_thread_init": format_cpu_set(master_cpus),
        "memory_placement": memory_placement,
        "ld_preload": preload,
        "thp_mem_alloc_enable": os.environ["THP_MEM_ALLOC_ENABLE"],
        "thp_enabled": os.environ["THP_MEM_ALLOC_ENABLE"] == "1",
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


def schedule_metadata(m, n, k, diagnostic_executable=None):
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
    if args.scheduler == "single_shape":
        eligible = (m, n, k, args.threads) == (4096, 12288, 4096, 96)
        metadata.update(
            effective_schedule=None if eligible else "flat_auto",
            scheduler_eligible=eligible,
            expected_schedule="single_shape" if eligible else "flat_auto",
            eligibility_reason="requires BF16 4096x12288x4096 and actual team 96",
            scheduler_metadata_source=(
                "shape and requested team; actual execution team is unobserved"
            ),
        )
        return metadata
    if args.scheduler == "flat_auto":
        from benchmark_cpu_sgl_thread_mapping import run_diagnostic

        diagnostic = run_diagnostic(
            diagnostic_executable,
            scheduler=args.scheduler,
            threads=args.threads,
            m=m,
            n=n,
            k=k,
            # This public operator uses BF16 output and a single INT8 K panel.
            panels=1,
            scratch_bytes=4096,
            output_bytes=2,
        )
        metadata.update(
            scheduler_diagnostic={
                key: diagnostic[key]
                for key in (
                    "selected_schedule",
                    "effective_schedule",
                    "actual_team_size",
                    "auto_plan",
                    "worker_load",
                    "coverage",
                    "assignments",
                    "diagnostic_command",
                    "diagnostic_environment",
                    "diagnostic_cpu_affinity",
                )
            },
            scheduler_metadata_source="untimed standalone shared-planner preflight",
        )
        return metadata

    selected_geometry = args.scheduler
    if selected_geometry not in FLAT_SCHEDULE_GEOMETRY:
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

    bm, bn = FLAT_SCHEDULE_GEOMETRY[selected_geometry]
    task_count = math.ceil(m_blocks / bm) * math.ceil(n_blocks / bn)
    eligible = m >= 128 and (bm == 1 or task_count >= 192)
    if m < 128:
        reason = "M_below_128"
    elif not eligible:
        reason = "superblock_tasks_below_192"
    else:
        reason = "eligible"
    effective_schedule = selected_geometry if eligible else "legacy"
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
    if eligible:
        traversal = flat_traversal_metadata(
            m_blocks,
            n_blocks,
            m,
            args.threads,
            bm,
            bn,
            # Ordinary dense INT8 uses one K panel.
            k=k,
            panels=1,
        )
        metadata.update(flat_traversal=traversal)
    return metadata


def flat_traversal_metadata(mb, nb, m, nth, bm, bn, *, k, panels):
    """Mirror the dense scheduler's traversal and ownership selection."""
    threshold = 0.25
    max_group_n = max(bn, (1024 * 1024) // (32 * k * panels))
    max_group_m = max(bm, (1024 * 1024) // (32 * k))
    groupable = m >= threshold * (nb * 32) * panels
    partial_m = groupable and m % 32 != 0 and mb > 1
    base_task_count = math.ceil(mb / bm) * math.ceil(nb / bn)
    max_product = 1
    if groupable:
        max_product = max(1, base_task_count // nth)
        max_product = 1 << (max_product.bit_length() - 1)

    selected = None
    target_aspect = 2.0
    product = max_product
    while product >= 1 and selected is None:
        target_aspect = (
            min(4.0, max(2.0, k / 1024.0))
            if groupable and not partial_m and bm * bn * product <= 16
            else 2.0
        )
        candidates = []
        for gm in range(1, product + 1):
            if product % gm:
                continue
            gn = product // gm
            group_m = bm * gm
            group_n = bn * gn
            if (
                group_m > mb
                or group_m > max_group_m
                or group_n > nb
                or group_n > max_group_n
            ):
                continue
            physical_aspect = group_m / group_n
            score = abs(math.log2(physical_aspect / target_aspect))
            candidates.append((score, -group_n, group_m, group_n))
        if candidates:
            best_score = min(candidate[0] for candidate in candidates)
            tied_candidates = [
                candidate
                for candidate in candidates
                if candidate[0] <= best_score + 1e-12
            ]
            _, _, group_m, group_n = min(
                tied_candidates,
                key=lambda candidate: (candidate[1], candidate[2]),
            )
            selected = (group_m, group_n)
        product //= 2

    if selected is None:
        group_m, group_n = min(bm, mb), min(bn, nb)
    else:
        group_m, group_n = selected
    column_rows = 0
    column_joint_budget = None
    if groupable and not partial_m and group_m * group_n <= 16 and mb % group_m == 0:
        column_rows = group_m
        if k > 2048:
            column_joint_budget = 2 * max(1, (1024 * 1024) // (32 * k))
            max_work = column_rows * ((mb // column_rows * nb + nth - 1) // nth)
            for candidate in range(column_rows + bm, min(mb, max_work) + 1, bm):
                if mb % candidate:
                    continue
                columns = (mb // candidate * nb + nth - 1) // nth
                if (
                    candidate * columns <= max_work
                    and candidate + columns * panels <= column_joint_budget
                ):
                    column_rows = candidate
    column_width = 1
    if column_rows and groupable and k > 2048:
        candidate_width = (mb // column_rows * nb + nth - 1) // nth
        if (mb // column_rows) * ((nb + candidate_width - 1) // candidate_width) <= nth:
            column_width = candidate_width
    column_peak_work = (
        column_rows * ((mb // column_rows * nb + nth - 1) // nth)
        if column_rows
        else None
    )
    column_chunks_per_m_band = (
        (nb + column_width - 1) // column_width if column_rows else None
    )
    interleaved_m_rows = 0
    if groupable and not column_rows:
        interleaved_m_rows = mb % group_m
        if interleaved_m_rows == 0 and partial_m:
            interleaved_m_rows = 1
    if interleaved_m_rows:
        task_order = "N_fast_groups_with_interleaved_M_edge"
    elif groupable and column_rows:
        task_order = (
            "M_group_columns_N_chunks" if column_width > 1 else "M_group_columns_N_fast"
        )
    elif groupable:
        task_order = "N_fast_groups_then_microtiles"
    else:
        task_order = "M_fast_base_superblocks"
    ownership_unit = (
        "M_group_columns"
        if groupable and column_rows
        else "compute_tile"
        if groupable
        else "base_superblocks"
    )
    ownership_rank_count = (
        (mb // column_rows) * column_chunks_per_m_band
        if column_rows
        else mb * nb
        if groupable
        else base_task_count
    )
    return {
        "base_group_m": bm,
        "base_group_n": bn,
        "group_m": group_m,
        "group_n": group_n,
        "grouped": groupable,
        "partial_m": partial_m,
        "partial_m_rows": m % 32 if partial_m else 0,
        "column_rows": column_rows,
        "column_width": column_width,
        "column_chunks_per_m_band": column_chunks_per_m_band,
        "column_joint_budget": column_joint_budget,
        "column_peak_work": column_peak_work,
        "interleaved_m_rows": interleaved_m_rows,
        "ownership_unit": ownership_unit,
        "ownership_rank_count": ownership_rank_count,
        "rank_partition": (
            "proportional_floor" if column_rows else "prefix_quotient_remainder"
        ),
        "task_order": task_order,
        "tail_order": "interleaved" if interleaved_m_rows else "regular",
        "assumed_team_size": nth,
        "team_size_source": "requested threads; actual OpenMP team unmeasured",
        "max_group_m": min(mb, max_group_m),
        "max_group_n": min(nb, max_group_n),
        "group_n_cache_bytes": group_n * 32 * k * panels,
        "group_n_packed_weight_bytes": group_n * 32 * (k + 4) * panels,
        "target_aspect": target_aspect,
        "threshold": threshold,
    }


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

    if custom_shape is not None:
        return {"custom": [custom_shape]}

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
diagnostic_executable = None
if args.scheduler == "flat_auto":
    from benchmark_cpu_sgl_thread_mapping import build_diagnostic

    diagnostic_executable, diagnostic_sources = build_diagnostic(
        args.scheduler_diagnostic_build_dir
    )
    metadata["scheduler_diagnostic_provenance"] = {
        **diagnostic_sources,
        "extension_scheduler_source_binding": "unverified",
        "extension_actual_team_size": None,
        "note": (
            "The diagnostic compiles the current source separately from the loaded "
            "extension. Its observed team and plan describe that probe; the "
            "extension SHA256 identifies the timed binary and does not establish "
            "a matching source revision or its internal OpenMP team."
        ),
    }
results = []
cases_by_projection = selected_cases()
args.output_dir.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    for projection, m_values in cases_by_projection.items():
        if projection == "custom":
            ((m, k, n),) = m_values
            m_values = (m,)
        else:
            k, n = PROJECTIONS[projection]

        packed_matrix_bytes = n * (k + 4)
        depth = 1 + math.ceil(TARGET_WEIGHT_BYTES / packed_matrix_bytes)
        packed = allocate_packed_bank((depth, n, k + 4))
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
        thp_enabled = os.environ["THP_MEM_ALLOC_ENABLE"] == "1"
        packed_memory = memory_setup_metadata(
            packed,
            expected_node=metadata["memory_placement"]["numa_node"],
            require_huge_pages=thp_enabled,
        )
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
            shape_schedule_metadata = schedule_metadata(m, n, k, diagnostic_executable)
            x = torch.randn(m, k).bfloat16()
            input_memory = memory_setup_metadata(
                x,
                expected_node=metadata["memory_placement"]["numa_node"],
                require_huge_pages=False,
            )

            def run(weight, x=x, scales=scales):
                return torch.ops._C.int8_scaled_mm_with_quant(
                    x, weight, scales, None, x.dtype, True
                )

            actual = run(packed[0])
            output_memory = memory_setup_metadata(
                actual,
                expected_node=metadata["memory_placement"]["numa_node"],
                require_huge_pages=False,
            )
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
            cold_resource_deltas = []
            for _ in range(args.samples):
                resource_before = resource_usage()
                start_time = time.perf_counter_ns()
                for _ in range(cold_repetitions):
                    run_cold()
                elapsed_ms = (time.perf_counter_ns() - start_time) / 1e6
                cold_samples_ms.append(elapsed_ms / cold_repetitions)
                resource_after = resource_usage()
                cold_resource_deltas.append(
                    {
                        field: resource_after[field] - resource_before[field]
                        for field in RESOURCE_USAGE_FIELDS
                    }
                )

            row = {
                "projection": projection,
                "m": m,
                "k": k,
                "n": n,
                **shape_schedule_metadata,
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
                "cold_sample_process_resource_deltas": cold_resource_deltas,
                "correctness_relative_l2": relative_l2,
                "memory": {
                    "packed_weights": packed_memory,
                    "activation": input_memory,
                    "output": output_memory,
                    "huge_page_target_enabled": thp_enabled,
                },
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU INT8 full-operator timing with disjoint cold packed-weight banks."""

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import mmap
import os
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--shape", default="4096:4096:12288", metavar="M:K:N")
parser.add_argument("--threads", type=int, default=96)
parser.add_argument("--scheduler", choices=("legacy", "generic"), default="legacy")
parser.add_argument("--samples", type=int, default=5)
parser.add_argument("--sample-ms", type=float, default=100)
parser.add_argument("--consume-output", action="store_true")
parser.add_argument("--observe-dispatch", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
m, k, n = map(int, args.shape.split(":"))
if min(m, k, n, args.threads, args.samples, args.sample_ms) <= 0 or k % 32 or n % 32:
    parser.error("positive dimensions, threads and samples; K/N divisible by 32")
os.environ.update(
    OMP_NUM_THREADS=str(args.threads),
    OMP_THREAD_LIMIT=str(args.threads),
    OMP_DYNAMIC="FALSE",
    VLLM_CPU_SGL_SCHEDULER=args.scheduler,
)
os.environ.pop("VLLM_CPU_SGL_DISPATCH", None)
if os.environ.get("THP_MEM_ALLOC_ENABLE") != "1":
    parser.error("set THP_MEM_ALLOC_ENABLE=1 before starting Python")
startup_cpus = os.sched_getaffinity(0)

import torch  # noqa: E402

import vllm._custom_ops  # noqa: E402, F401

TARGET_WEIGHT_BYTES = int(1.5 * 1024**3)


def allocate_packed_bank(shape):
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


def measure(call, depth):
    for _ in range(depth):
        call()
    start = time.perf_counter_ns()
    call()
    probe_ms = (time.perf_counter_ns() - start) / 1e6
    repeats = max(depth, math.ceil(args.sample_ms / probe_ms / depth) * depth)
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            call()
        samples.append((time.perf_counter_ns() - start) / 1e6 / repeats)
    return samples, repeats


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
torch.cpu._init_amx()
extension = Path(importlib.import_module("vllm._C").__file__).resolve()
# Intel OpenMP can narrow the master thread after startup; its explicit worker
# list describes the measurement team in staged runs.
proclist = (
    os.environ.get("KMP_AFFINITY", "").partition("proclist=[")[2].partition("]")[0]
)
cpus = set()
for part in proclist.split(",") if proclist else ():
    limits = list(map(int, part.split("-")))
    cpus.update(range(limits[0], limits[-1] + 1))
cpus = cpus or startup_cpus
physical_cores = {
    tuple(
        (Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") / field).read_text().strip()
        for field in ("physical_package_id", "core_id")
    )
    for cpu in cpus
}
if len(physical_cores) != len(cpus):
    raise RuntimeError("bind one hardware thread per physical core")
nodes = {
    int(next(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*")).name[4:])
    for cpu in cpus
}
if len(nodes) != 1 or len(cpus) != args.threads:
    raise RuntimeError("bind exactly one socket's physical CPUs and memory")
node = nodes.pop()
preloads = [Path(p).resolve() for p in os.environ.get("LD_PRELOAD", "").split(":") if p]
loaded = Path("/proc/self/maps").read_text()
for name in ("libtcmalloc", "libiomp"):
    if not any(name in p.name and str(p) in loaded for p in preloads):
        raise RuntimeError(f"preload {name} before starting Python")

with torch.inference_mode():
    weight_shapes = ((n, k),)
    matrix_bytes = sum(rows * (cols + 4) for rows, cols in weight_shapes)
    depth = 1 + math.ceil(TARGET_WEIGHT_BYTES / matrix_bytes)
    banks, raw_first, memory = [], [], []
    for rows, cols in weight_shapes:
        bank = allocate_packed_bank((depth, rows, cols + 4))
        for start in range(0, depth, 8):
            raw = torch.randint(
                -32, 33, (min(8, depth - start), rows, cols), dtype=torch.int8
            )
            if start == 0:
                raw_first.append(raw[0].clone())
            bank[start : start + len(raw)].copy_(
                torch.ops._C.convert_weight_packed(raw)
            )
        memory.append(
            memory_setup_metadata(bank, expected_node=node, require_huge_pages=True)
        )
        banks.append(bank)
    x = torch.randn(m, k).bfloat16()
    scales = [torch.full((rows,), 0.001) for rows, _ in weight_shapes]

    def operator(index):
        return torch.ops._C.int8_scaled_mm_with_quant(
            x, banks[0][index], scales[0], None, x.dtype, True
        )

    if args.observe_dispatch:
        os.environ["VLLM_CPU_SGL_DISPATCH"] = "1"
    actual = operator(0)
    os.environ.pop("VLLM_CPU_SGL_DISPATCH", None)
    reference = x[:4].float() @ (raw_first[0].float() * scales[0][:, None]).t()
    relative_l2 = float((actual[:4].float() - reference).norm() / reference.norm())
    if relative_l2 >= 0.03:
        raise RuntimeError(f"dense reference relative L2: {relative_l2}")
    del actual

    def run(index):
        output = operator(index)
        if args.consume_output:
            return output.sum(dtype=torch.float32)
        return output

    warm, warm_repeats = measure(lambda: run(0), 1)
    index = 0

    def cold_call():
        global index
        result = run(index)
        index = (index + 1) % depth
        return result

    cold, cold_repeats = measure(cold_call, depth)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "command": sys.argv,
            "extension_path": str(extension),
            "extension_sha256": sha256(extension),
            "requested_threads": args.threads,
            "torch_threads": torch.get_num_threads(),
            "allowed_cpus": sorted(cpus),
            "scheduler": args.scheduler,
            "consume_output": args.consume_output,
            "memory_placement": {"numa_node": node},
            "preloads": [{"path": str(p), "sha256": sha256(p)} for p in preloads],
            "results": [
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "cold_median_ms": statistics.median(cold),
                    "cold_samples_ms": cold,
                    "warm_median_ms": statistics.median(warm),
                    "warm_samples_ms": warm,
                    "cold_repetitions_per_sample": cold_repeats,
                    "warm_repetitions_per_sample": warm_repeats,
                    "rotating_weight_matrices": depth,
                    "cold_revisit_distinct_intervening_packed_bytes": (depth - 1)
                    * matrix_bytes,
                    "memory": {"packed_weights": memory},
                    "correctness_relative_l2": relative_l2,
                }
            ],
            "note": "Packing and reference checks excluded; allocation, quantization, "
            "GEMM, epilogue and publication included. Cold rotation continues "
            "across samples.",
        },
        indent=2,
    )
    + "\n"
)
print(args.output)

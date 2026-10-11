# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline correctness and replay harness for the experimental NUMA MoE op."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import regex as re
import torch
import torch.nn.functional as F
import vllm._C  # noqa: F401

import vllm._custom_ops as custom_ops

BASE_POLICIES = ("static", "balanced", "split", "rows")
KERNEL_CONTROLS = ("unequal", "tp2_01", "tp2_02", "tp2_12")
POLICIES = (*BASE_POLICIES, *KERNEL_CONTROLS)
SINGLE_COPY_CONTROLS = (
    "single_node_node1",
    "single_node_node2",
    "single_node_node3",
    "unsharded_96_node1_weights",
)
NATIVE_TP2_CONTROLS = ("native_tp2_01", "native_tp2_02", "native_tp2_12")
EXISTING_CONTROLS = (*SINGLE_COPY_CONTROLS, *NATIVE_TP2_CONTROLS)
CALIBRATION_ROWS = (
    1,
    4,
    5,
    8,
    16,
    31,
    32,
    33,
    63,
    64,
    65,
    96,
    128,
    160,
    161,
    192,
    224,
    225,
    255,
    256,
    257,
    288,
    289,
    512,
    1024,
    2048,
)
SYNTHETIC_TOKEN_COUNTS = (1, 8, 32, 128, 512, 2048)
DEFAULT_CPUS = tuple(range(32, 64)) + tuple(range(64, 96)) + tuple(range(96, 128))
DEFAULT_NODES = (1, 2, 3)
PHASE_NAMES = ("schedule", "dispatch", "expert", "gather_combine", "total")
EXPECTED_SOURCE_HEAD = "c5a3a1389df9aae4be73bcac7b8eea6164af02c7"
VARIANT_QUIESCENCE_SECONDS = 0.01


def _progress(path: str | None, event: str, **fields: Any) -> None:
    if path:
        with Path(path).open("a") as stream:
            stream.write(
                json.dumps({"event": event, "time_unix": time.time(), **fields}) + "\n"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest_update(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _tensor_digest_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _tensor_digest_update(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(json.dumps(value, sort_keys=True).encode())
    else:
        raise TypeError(f"Unsupported value in tensor digest: {type(value)!r}")
    digest.update(b"\0")


def _content_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _tensor_digest_update(digest, value)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _extension_identity() -> dict[str, Any]:
    extension_path = Path(vllm._C.__file__).resolve()
    op_names = (
        "cpu_numa_moe_create",
        "cpu_numa_moe_execute",
        "cpu_numa_moe_stats",
        "cpu_numa_moe_destroy",
        "cpu_numa_moe_calibrate",
        "cpu_numa_moe_local_copy",
        "cpu_numa_moe_verify_local",
        "cpu_numa_moe_evict",
        "cpu_numa_moe_evict_local",
        "fused_experts_cpu",
    )
    return {
        "path": str(extension_path),
        "sha256": _sha256_file(extension_path) if extension_path.is_file() else None,
        "operator_schemas": {
            name: str(getattr(torch.ops._C, name).default._schema) for name in op_names
        },
    }


def _preload_identity() -> dict[str, Any]:
    value = os.environ.get("LD_PRELOAD", "")
    paths = [Path(entry).expanduser() for entry in value.replace(":", " ").split()]
    components = {}
    for name, marker in (("tcmalloc", "libtcmalloc"), ("libiomp5", "libiomp5")):
        matches = [path for path in paths if marker in path.name and path.is_file()]
        path = matches[0].resolve() if matches else None
        components[name] = {
            "path": str(path) if path is not None else None,
            "sha256": _sha256_file(path) if path is not None else None,
            "verified": path is not None,
        }
    return {
        "ld_preload": value,
        "components": components,
        "verified": all(component["verified"] for component in components.values()),
    }


def _thread_affinity(allowed_cpus: tuple[int, ...]) -> dict[str, list[int]]:
    affinity = {
        task.name: sorted(os.sched_getaffinity(int(task.name)))
        for task in Path("/proc/self/task").iterdir()
    }
    if any(not set(mask).issubset(allowed_cpus) for mask in affinity.values()):
        raise RuntimeError(f"Control threads escaped their CPU team: {affinity}")
    helper_cpus = {mask[0] for mask in affinity.values() if len(mask) == 1}
    if not set(allowed_cpus[1:]).issubset(helper_cpus):
        raise RuntimeError(f"OpenMP helpers do not cover their CPU team: {affinity}")
    return affinity


def _source_identity(script_path: Path) -> dict[str, Any]:
    repo_root = script_path.parents[3]
    diff = _git("diff", "--binary", "HEAD", cwd=repo_root)
    diff_hash = hashlib.sha256((diff or "").encode()).hexdigest()
    native_sources = (
        "cmake/cpu_extension.cmake",
        "csrc/cpu/sgl-kernels/gemm.h",
        "csrc/cpu/sgl-kernels/moe.cpp",
        "csrc/cpu/sgl-kernels/moe_executor.h",
        "csrc/cpu/sgl-kernels/moe_fp8.cpp",
        "csrc/cpu/sgl-kernels/moe_numa.cpp",
        "csrc/cpu/torch_bindings.cpp",
    )
    return {
        "revision": _git("rev-parse", "HEAD", cwd=repo_root),
        "tracked_diff_sha256": diff_hash,
        "script_sha256": _sha256_file(script_path),
        "support_script_sha256": {
            name: _sha256_file(script_path.with_name(name))
            for name in (
                "numa_moe_native_tp2.py",
                "numa_moe_diagnostics.py",
                "numa_moe_perf.py",
                "numa_moe_reference_cache.py",
            )
        },
        "native_source_sha256": {
            source: (
                _sha256_file(repo_root / source)
                if (repo_root / source).is_file()
                else None
            )
            for source in native_sources
        },
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_source_context() -> Path:
    repo_root = _repo_root()
    revision = _git("rev-parse", "HEAD", cwd=repo_root)
    if (
        revision is None
        or _git(
            "merge-base",
            "--is-ancestor",
            EXPECTED_SOURCE_HEAD,
            revision,
            cwd=repo_root,
        )
        is None
    ):
        raise RuntimeError(
            "Expected source HEAD to descend from "
            f"{EXPECTED_SOURCE_HEAD}, found {revision!r}"
        )
    extension_path = Path(vllm._C.__file__).resolve()
    try:
        extension_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(
            "Imported vllm._C extension is outside the requested checkout: "
            f"{extension_path}"
        ) from exc
    return repo_root


def _load_torch_file(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _normalize_layer_key(key: Any) -> int:
    try:
        return int(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Layer keys must be integer-convertible, got {key!r}"
        ) from exc


def _load_weights(path: Path) -> tuple[list[int], dict[int, dict[str, torch.Tensor]]]:
    payload = _load_torch_file(path)
    if isinstance(payload, dict) and "layers" in payload:
        payload = payload["layers"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Weight file must contain a non-empty layer-to-tensors mapping"
        )

    layers = {_normalize_layer_key(key): value for key, value in payload.items()}
    layer_ids = sorted(layers)
    for layer_id in layer_ids:
        layer = layers[layer_id]
        if not isinstance(layer, dict):
            raise ValueError(f"Layer {layer_id} must map names to tensors")
        missing = {"w1", "w2", "s1", "s2"} - layer.keys()
        if missing:
            raise ValueError(f"Layer {layer_id} is missing weights: {sorted(missing)}")
        for name in ("w1", "w2", "s1", "s2"):
            if not isinstance(layer[name], torch.Tensor):
                raise TypeError(f"Layer {layer_id} field {name!r} must be a tensor")
            layer[name] = layer[name].detach().cpu().contiguous()
        if layer["w1"].dtype != torch.float8_e4m3fn:
            raise ValueError(f"Layer {layer_id} w1 must be raw float8_e4m3fn")
        if layer["w2"].dtype != torch.float8_e4m3fn:
            raise ValueError(f"Layer {layer_id} w2 must be raw float8_e4m3fn")
        if layer["w1"].ndim != 3 or layer["w2"].ndim != 3:
            raise ValueError(f"Layer {layer_id} weights must have shape [E,N,K]")
        if layer["s1"].ndim != 3 or layer["s2"].ndim != 3:
            raise ValueError(f"Layer {layer_id} scales must have shape [E,Nb,Kb]")
        if layer["w1"].shape[0] != layer["w2"].shape[0]:
            raise ValueError(f"Layer {layer_id} w1/w2 expert counts differ")
        if layer["s1"].shape[0] != layer["w1"].shape[0]:
            raise ValueError(f"Layer {layer_id} w1/scales expert counts differ")
        if layer["s2"].shape[0] != layer["w2"].shape[0]:
            raise ValueError(f"Layer {layer_id} w2/scales expert counts differ")
        if layer["s1"].dtype != torch.float32 or layer["s2"].dtype != torch.float32:
            raise ValueError(f"Layer {layer_id} block scales must be float32")
        if (
            not torch.isfinite(layer["s1"]).all()
            or not torch.isfinite(layer["s2"]).all()
        ):
            raise ValueError(f"Layer {layer_id} has non-finite block scales")
        if (
            layer["w1"].shape[1] % 2
            or layer["w1"].shape[2] != layer["w2"].shape[1]
            or layer["w1"].shape[1] // 2 != layer["w2"].shape[2]
        ):
            raise ValueError(f"Layer {layer_id} gate/up and down shapes do not match")

    reference_shapes = {
        name: tuple(layers[layer_ids[0]][name].shape)
        for name in ("w1", "w2", "s1", "s2")
    }
    for layer_id in layer_ids[1:]:
        for name, expected in reference_shapes.items():
            if tuple(layers[layer_id][name].shape) != expected:
                raise ValueError(
                    f"Layer {layer_id} {name} shape differs from first layer"
                )
    return layer_ids, layers


def _load_corpus(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_torch_file(path)
    metadata = {}
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        payload = payload.get("records")
    if not isinstance(metadata, dict):
        raise ValueError("Corpus metadata must be a mapping")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Corpus must contain a non-empty ordered list of records")
    records = []
    for index, source in enumerate(payload):
        if not isinstance(source, dict):
            raise ValueError(f"Corpus record {index} must be a mapping")
        missing = {"hidden", "ids", "weights", "layer", "phase"} - source.keys()
        if missing:
            raise ValueError(f"Corpus record {index} is missing {sorted(missing)}")
        hidden = source["hidden"]
        ids = source["ids"]
        weights = source["weights"]
        if not all(isinstance(x, torch.Tensor) for x in (hidden, ids, weights)):
            raise TypeError(f"Record {index} hidden, ids, and weights must be tensors")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Record {index} ids must have int32 or int64 dtype")
        hidden = hidden.detach().cpu().contiguous()
        ids = ids.detach().cpu().to(torch.int32).contiguous()
        weights = weights.detach().cpu().float().contiguous()
        if hidden.ndim != 2 or hidden.dtype != torch.bfloat16:
            raise ValueError(f"Record {index} hidden must be BF16 [tokens, hidden]")
        if ids.ndim != 2 or weights.shape != ids.shape:
            raise ValueError(
                f"Record {index} ids and weights must share [tokens, topk]"
            )
        if hidden.shape[0] == 0 or ids.shape[0] != hidden.shape[0] or ids.shape[1] == 0:
            raise ValueError(f"Record {index} routing shape does not match hidden rows")
        if not torch.isfinite(weights).all():
            raise ValueError(f"Record {index} contains non-finite routing weights")
        if not isinstance(source["phase"], str):
            raise ValueError(f"Record {index} phase must be a string")
        record = dict(source)
        record.update(
            hidden=hidden,
            ids=ids,
            weights=weights,
            layer=_normalize_layer_key(source["layer"]),
            phase=source["phase"],
        )
        if "reference_output" in record:
            original_output = record["reference_output"]
            if (
                not isinstance(original_output, torch.Tensor)
                or original_output.shape != hidden.shape
                or original_output.dtype != hidden.dtype
            ):
                raise ValueError(f"Record {index} has an invalid captured output")
            record["reference_output"] = original_output.detach().cpu().contiguous()
        records.append(record)
    return records, metadata


def _validate_records(
    records: list[dict[str, Any]],
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
) -> dict[int, int]:
    layer_to_index = {layer_id: index for index, layer_id in enumerate(layer_ids)}
    expert_count = next(iter(layers.values()))["w1"].shape[0]
    hidden_size = next(iter(layers.values()))["w1"].shape[2]
    for record_index, record in enumerate(records):
        layer_id = record["layer"]
        if layer_id not in layer_to_index:
            raise ValueError(
                f"Record {record_index} references missing layer {layer_id}"
            )
        if record["hidden"].shape[1] != hidden_size:
            raise ValueError(
                f"Record {record_index} hidden size does not match its weights"
            )
        ids = record["ids"]
        if ids.shape[1] != 8:
            raise ValueError(f"Record {record_index} must contain top-8 routing")
        if ids.min().item() < 0 or ids.max().item() >= expert_count:
            raise ValueError(f"Record {record_index} has an invalid expert id")
        sorted_ids = ids.sort(dim=1).values
        if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError(
                f"Record {record_index} repeats an expert within a top-k row"
            )
        if record.get("concurrency") is not None and int(record["concurrency"]) <= 0:
            raise ValueError(f"Record {record_index} concurrency must be positive")
    return layer_to_index


def _manifest_payload(
    corpus_path: Path,
    weights_path: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_file_sha256": _sha256_file(corpus_path),
        "corpus_tensor_sha256": _content_hash(records),
        "weights_path": str(weights_path.resolve()),
        "weights_file_sha256": _sha256_file(weights_path),
        "weights_tensor_sha256": _content_hash(layers),
        "record_count": len(records),
        "layer_ids": layer_ids,
        "metadata": json.loads(json.dumps(metadata)),
    }


def _primary_corpus_status(
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
    block_n: int,
    block_k: int,
) -> dict[str, Any]:
    expected = {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "fixed_sonnet_prompts": True,
        "input_tokens": 1024,
        "generated_tokens": 128,
        "prompts_per_concurrency": 256,
        "concurrency": [8, 32, 128],
        "mtp_enabled": False,
        "deterministic": True,
        "captured_prefill_steps": 8,
        "captured_decode_steps": 8,
        "capture_after_warmup": True,
        "consecutive_steps": True,
        "preserves_mixed_steps": True,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    checkpoint_revision = metadata.get("checkpoint_revision")
    checkpoint_sha256 = metadata.get("checkpoint_sha256")
    prompt_set_sha256 = metadata.get("prompt_set_sha256")
    if not isinstance(checkpoint_revision, str) or not checkpoint_revision.strip():
        mismatches["checkpoint_revision"] = {
            "expected": "non-empty revision",
            "actual": checkpoint_revision,
        }

    def is_sha256(value: Any) -> bool:
        return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value))

    def parse_int(value: Any) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            return int(value)
        return None

    for key, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("prompt_set_sha256", prompt_set_sha256),
    ):
        if not is_sha256(value):
            mismatches[key] = {
                "expected": "64-character hexadecimal SHA256",
                "actual": value,
            }

    declared_order_value = metadata.get("routed_layer_order")
    try:
        declared_order = [int(layer) for layer in declared_order_value]
    except (TypeError, ValueError):
        declared_order = []
    if declared_order != layer_ids:
        mismatches["routed_layer_order"] = {
            "expected": layer_ids,
            "actual": declared_order_value,
        }

    cutoff_value = metadata.get("warmup_step_cutoff")
    cutoff = parse_int(cutoff_value)
    if cutoff is None or cutoff < 0:
        mismatches["warmup_step_cutoff"] = {
            "expected": "non-negative integer step identifier",
            "actual": cutoff_value,
        }

    per_concurrency: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    structural_mismatches = []
    actual_concurrencies = set()
    missing_concurrency_records = 0
    missing_step_records = 0
    invalid_phase_records = 0
    has_capture_window_labels = any(
        record.get("capture_window") is not None for record in records
    )
    last_step_by_concurrency: dict[int, int] = {}
    last_record_group = None
    for index, record in enumerate(records):
        raw_concurrency = record.get("concurrency")
        if raw_concurrency is None:
            missing_concurrency_records += 1
            continue
        concurrency = parse_int(raw_concurrency)
        if concurrency is None:
            structural_mismatches.append(
                f"record {index} has invalid concurrency {raw_concurrency!r}"
            )
            continue
        actual_concurrencies.add(concurrency)
        raw_step = record.get("step")
        step = parse_int(raw_step)
        if step is None:
            missing_step_records += 1
            continue
        phase = record.get("phase")
        if phase not in ("prefill", "decode", "mixed"):
            invalid_phase_records += 1
            continue
        previous_step = last_step_by_concurrency.get(concurrency)
        if previous_step is not None and step < previous_step:
            structural_mismatches.append(
                f"record {index} reorders concurrency {concurrency} steps"
            )
        group_key = (concurrency, step)
        if step in per_concurrency[concurrency] and group_key != last_record_group:
            structural_mismatches.append(
                f"record {index} interleaves layers from different captured steps"
            )
        last_step_by_concurrency[concurrency] = step
        last_record_group = group_key
        step_info = per_concurrency[concurrency].setdefault(
            step, {"layers": [], "phases": set(), "capture_windows": []}
        )
        step_info["layers"].append(record["layer"])
        step_info["phases"].add(phase)
        step_info["capture_windows"].append(record.get("capture_window"))

    actual_concurrencies = sorted(actual_concurrencies)
    step_validation = {}
    for concurrency in (8, 32, 128):
        steps = per_concurrency.get(concurrency, {})
        sorted_steps = sorted(steps)
        phases_by_step = {}
        capture_window_by_step = {}
        for step in sorted_steps:
            info = steps[step]
            phases = info["phases"]
            if info["layers"] != declared_order:
                structural_mismatches.append(
                    f"concurrency {concurrency}, step {step} layer order "
                    f"{info['layers']} does not match {declared_order}"
                )
            if len(phases) != 1:
                structural_mismatches.append(
                    f"concurrency {concurrency}, step {step} has mixed record "
                    f"phase labels {sorted(phases)}"
                )
            phases_by_step[step] = next(iter(phases)) if len(phases) == 1 else None
            capture_windows = info["capture_windows"]
            valid_capture_windows = all(
                isinstance(window, str) and window in ("prefill", "decode")
                for window in capture_windows
            )
            consistent_capture_window = (
                bool(capture_windows)
                and valid_capture_windows
                and len(set(capture_windows)) == 1
            )
            if has_capture_window_labels and not consistent_capture_window:
                structural_mismatches.append(
                    f"concurrency {concurrency}, step {step} has incomplete or "
                    "inconsistent capture_window labels "
                    f"{[repr(window) for window in capture_windows]}"
                )
            if consistent_capture_window:
                capture_window_by_step[step] = capture_windows[0]

        if has_capture_window_labels:
            for step, window in capture_window_by_step.items():
                phase = phases_by_step[step]
                compatible_phases = (
                    ("prefill", "mixed") if window == "prefill" else ("decode", "mixed")
                )
                if (
                    window not in ("prefill", "decode")
                    or phase not in compatible_phases
                ):
                    structural_mismatches.append(
                        f"concurrency {concurrency}, step {step} capture window "
                        f"{window!r} is incompatible with phase {phase!r}"
                    )
            prefill_steps = sorted(
                step
                for step, window in capture_window_by_step.items()
                if window == "prefill"
            )
            decode_steps = sorted(
                step
                for step, window in capture_window_by_step.items()
                if window == "decode"
            )
            window_label_mode = "explicit"
        else:
            prefill_steps = [
                step
                for step, phase in phases_by_step.items()
                if phase in ("prefill", "mixed")
            ]
            decode_steps = [
                step
                for step, phase in phases_by_step.items()
                if phase in ("decode", "mixed")
            ]
            window_label_mode = "derived_from_phase"
        all_steps_consecutive = bool(sorted_steps) and all(
            right == left + 1 for left, right in zip(sorted_steps, sorted_steps[1:])
        )
        prefill_consecutive = len(prefill_steps) == 8 and all(
            right == left + 1 for left, right in zip(prefill_steps, prefill_steps[1:])
        )
        decode_consecutive = len(decode_steps) == 8 and all(
            right == left + 1 for left, right in zip(decode_steps, decode_steps[1:])
        )
        after_warmup = cutoff is not None and all(
            step > cutoff for step in sorted_steps
        )
        step_validation[str(concurrency)] = {
            "step_ids": sorted_steps,
            "window_label_mode": window_label_mode,
            "all_steps_consecutive": all_steps_consecutive,
            "after_declared_warmup_cutoff": after_warmup,
            "prefill_window_step_ids": prefill_steps,
            "prefill_window_is_eight_consecutive_steps": prefill_consecutive,
            "decode_window_step_ids": decode_steps,
            "decode_window_is_eight_consecutive_steps": decode_consecutive,
            "mixed_step_count": sum(
                phase == "mixed" for phase in phases_by_step.values()
            ),
            "complete_routed_layer_order": all(
                info["layers"] == declared_order for info in steps.values()
            )
            and bool(steps)
            and declared_order == layer_ids,
        }
        if not sorted_steps:
            structural_mismatches.append(
                f"concurrency {concurrency} has no captured steps"
            )
        if not after_warmup:
            structural_mismatches.append(
                f"concurrency {concurrency} has steps at or before the "
                "declared warmup cutoff"
            )
        if not prefill_consecutive:
            structural_mismatches.append(
                f"concurrency {concurrency} lacks exactly eight consecutive "
                "prefill-containing steps"
            )
        if not decode_consecutive:
            structural_mismatches.append(
                f"concurrency {concurrency} lacks exactly eight consecutive "
                "decode-containing steps"
            )

    if missing_concurrency_records:
        structural_mismatches.append(
            f"{missing_concurrency_records} records lack concurrency"
        )
    if missing_step_records:
        structural_mismatches.append(f"{missing_step_records} records lack step")
    if invalid_phase_records:
        structural_mismatches.append(
            f"{invalid_phase_records} records have an invalid phase"
        )
    if actual_concurrencies != [8, 32, 128]:
        structural_mismatches.append(
            f"record concurrency values are {actual_concurrencies}, expected "
            "[8, 32, 128]"
        )

    first_layer = layers[layer_ids[0]] if layer_ids else None
    actual_geometry = None
    if first_layer is not None:
        actual_geometry = {
            "experts": int(first_layer["w1"].shape[0]),
            "hidden_size": int(first_layer["w1"].shape[2]),
            "intermediate_size": int(first_layer["w2"].shape[2]),
            "gate_up_size": int(first_layer["w1"].shape[1]),
            "w1_scale_blocks": list(first_layer["s1"].shape[1:]),
            "w2_scale_blocks": list(first_layer["s2"].shape[1:]),
            "block_n": block_n,
            "block_k": block_k,
            "topk": sorted({int(record["ids"].shape[1]) for record in records}),
            "record_hidden_sizes": sorted(
                {int(record["hidden"].shape[1]) for record in records}
            ),
        }
    expected_geometry = {
        "experts": 256,
        "hidden_size": 2048,
        "intermediate_size": 512,
        "gate_up_size": 1024,
        "w1_scale_blocks": [8, 16],
        "w2_scale_blocks": [16, 4],
        "block_n": 128,
        "block_k": 128,
        "topk": [8],
        "record_hidden_sizes": [2048],
    }
    geometry_matches = actual_geometry == expected_geometry
    if not geometry_matches:
        structural_mismatches.append(
            f"model geometry is {actual_geometry}, expected {expected_geometry}"
        )

    return {
        "eligible": (
            not mismatches
            and not structural_mismatches
            and actual_concurrencies == [8, 32, 128]
        ),
        "metadata_mismatches": mismatches,
        "observable_structure": {
            "geometry": actual_geometry,
            "geometry_matches_target": geometry_matches,
            "per_concurrency": step_validation,
            "structural_mismatches": structural_mismatches,
        },
        "externally_asserted_capture_metadata": {
            "declared_capture_fields": [
                *expected,
                "checkpoint_revision",
                "checkpoint_sha256",
                "prompt_set_sha256",
            ],
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_sha256_format_valid": is_sha256(checkpoint_sha256),
            "prompt_set_sha256": prompt_set_sha256,
            "prompt_set_sha256_format_valid": is_sha256(prompt_set_sha256),
            "prompt_and_capture_claims_verified_from_source": False,
        },
        "record_concurrencies": actual_concurrencies,
    }


def _verify_manifest(path: Path, actual: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads(path.read_text())
    keys = (
        "format_version",
        "corpus_file_sha256",
        "corpus_tensor_sha256",
        "weights_file_sha256",
        "weights_tensor_sha256",
        "record_count",
        "layer_ids",
        "metadata",
    )
    changed = [key for key in keys if expected.get(key) != actual.get(key)]
    if changed:
        raise ValueError(
            "Frozen manifest does not match the loaded corpus/checkpoint tensors: "
            + ", ".join(changed)
        )
    return expected


def _block_dequantize(
    weight: torch.Tensor, scales: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    experts, out_features, in_features = weight.shape
    padded_n = math.ceil(out_features / block_n) * block_n
    padded_k = math.ceil(in_features / block_k) * block_k
    pad_n, pad_k = padded_n - out_features, padded_k - in_features
    padded = F.pad(weight, (0, pad_k, 0, pad_n))
    n_blocks, k_blocks = padded_n // block_n, padded_k // block_k
    blocked = (
        padded.view(experts, n_blocks, block_n, k_blocks, block_k)
        .permute(0, 1, 3, 2, 4)
        .float()
    )
    expected_scales = (experts, n_blocks, k_blocks)
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"Scale shape {tuple(scales.shape)} does not match {expected_scales}"
        )
    return (
        (blocked * scales.float().view(experts, n_blocks, k_blocks, 1, 1))
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, padded_n, padded_k)[:, :out_features, :in_features]
    )


def _reference_chunk(
    hidden: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    brgemm_slots: torch.Tensor,
) -> torch.Tensor:
    token_count, hidden_size = hidden.shape
    topk = ids.shape[1]
    flat_hidden = hidden.float().repeat_interleave(topk, dim=0)
    flat_ids = ids.reshape(-1)
    flat_brgemm = brgemm_slots.reshape(-1)
    slot_output = torch.zeros((token_count * topk, hidden_size), dtype=hidden.dtype)
    for expert_id in torch.unique(flat_ids, sorted=True).tolist():
        for brgemm in (False, True):
            selected = (flat_ids == expert_id) & (flat_brgemm == brgemm)
            if not selected.any():
                continue
            first_weight = w1[expert_id]
            second_weight = w2[expert_id]
            if brgemm:
                first_weight = first_weight.to(hidden.dtype).float()
                second_weight = second_weight.to(hidden.dtype).float()
            gate_up = (flat_hidden[selected] @ first_weight.t()).to(hidden.dtype)
            gate_up = gate_up.float()
            intermediate = gate_up.shape[1] // 2
            activated = (
                F.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
            ).to(hidden.dtype)
            down = (activated.float() @ second_weight.t()).to(hidden.dtype)
            slot_output[selected] = (
                down.float() * routing_weights.reshape(-1)[selected, None]
            ).to(hidden.dtype)
    weighted = slot_output.view(token_count, topk, hidden_size)
    total = torch.zeros_like(hidden, dtype=torch.float32)
    for slot in range(topk):
        total += weighted[:, slot].float()
    return total.to(hidden.dtype)


def _reference_output(
    record: dict[str, Any],
    w1: torch.Tensor,
    w2: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Model BF16 stages and 32-row branch choices with independent Torch math."""
    hidden, ids, weights = record["hidden"], record["ids"], record["weights"]
    flat_ids = ids.reshape(-1)
    brgemm = torch.zeros_like(flat_ids, dtype=torch.bool)
    # Choose the kernel branch on complete expert groups before token chunking.
    for expert_id in torch.unique(flat_ids, sorted=True).tolist():
        slots = torch.nonzero(flat_ids == expert_id, as_tuple=True)[0]
        full_rows = slots.numel() // 32 * 32
        brgemm[slots[:full_rows]] = True
        if slots.numel() - full_rows > 4:
            brgemm[slots[full_rows:]] = True
    brgemm = brgemm.view_as(ids)
    output = torch.empty_like(hidden)
    for start in range(0, hidden.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, hidden.shape[0])
        output[start:end] = _reference_chunk(
            hidden[start:end],
            ids[start:end],
            weights[start:end],
            w1,
            w2,
            brgemm[start:end],
        )
    return output


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = []
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("CPU and node lists cannot contain duplicates")
    return tuple(values)


def _parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be unique")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Values must be positive")
    return values


def _validate_topology(
    cpus: tuple[int, ...], nodes: tuple[int, ...], threads_per_node: int
) -> None:
    if len(nodes) != 3 or len(set(nodes)) != 3 or len(cpus) != 3 * threads_per_node:
        raise ValueError(
            "The prototype needs exactly three node IDs and "
            "threads_per_node * 3 CPU IDs"
        )
    if not 0 < threads_per_node <= 32:
        raise ValueError("threads_per_node must be between 1 and 32")


def _prepare_packed_weights(
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    packed = {}
    for layer_id in layer_ids:
        layer = layers[layer_id]
        packed[layer_id] = (
            torch.ops._C.convert_weight_packed(layer["w1"].clone()),
            torch.ops._C.convert_weight_packed(layer["w2"].clone()),
            layer["s1"],
            layer["s2"],
        )
    return packed


def _make_synthetic_records(
    layers: dict[int, dict[str, torch.Tensor]],
    layer_id: int,
    seed: int,
    token_counts: tuple[int, ...] = SYNTHETIC_TOKEN_COUNTS,
) -> list[dict[str, Any]]:
    layer = layers[layer_id]
    expert_count, _, hidden_size = layer["w1"].shape
    topk = 8
    if expert_count < topk:
        raise ValueError("Synthetic routing requires at least eight experts")
    generator = torch.Generator().manual_seed(seed)
    records = []
    for token_count in token_counts:
        for routing in ("uniform", "zipf", "hot8", "changing"):
            step_count = 2 if routing == "changing" else 1
            for step in range(step_count):
                hidden = torch.randn(token_count, hidden_size, generator=generator).to(
                    torch.bfloat16
                ) / math.sqrt(hidden_size)
                if routing == "uniform":
                    ids = torch.stack(
                        [
                            torch.randperm(expert_count, generator=generator)[:topk]
                            for _ in range(token_count)
                        ]
                    )
                elif routing == "zipf":
                    weights = torch.arange(
                        1, expert_count + 1, dtype=torch.float32
                    ).reciprocal()
                    weights /= weights.sum()
                    ids = torch.stack(
                        [
                            torch.multinomial(
                                weights, topk, replacement=False, generator=generator
                            )
                            for _ in range(token_count)
                        ]
                    )
                else:
                    start_expert = (
                        (step * min(128, expert_count - topk))
                        if routing == "changing"
                        else 0
                    )
                    hot_experts = torch.arange(
                        start_expert, min(start_expert + topk, expert_count)
                    )
                    ids = hot_experts.repeat(token_count, 1)
                route_weights = torch.softmax(
                    torch.randn(token_count, topk, generator=generator), dim=-1
                )
                records.append(
                    {
                        "hidden": hidden.contiguous(),
                        "ids": ids.to(torch.int32).contiguous(),
                        "weights": route_weights.contiguous(),
                        "layer": layer_id,
                        "phase": "synthetic",
                        "concurrency": None,
                        "routing": routing,
                        "step": step,
                    }
                )
    return records


def _stats_to_python(stats: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return repr(value)

    return {str(key): convert(value) for key, value in stats.items()}


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = min(math.ceil(len(ordered) * 0.95) - 1, len(ordered) - 1)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.fmean(ordered),
    }


def _synthetic_complete_call_summary(
    records: list[dict[str, Any]],
    policies: tuple[str, ...],
    controls: tuple[str, ...],
    variant_validity: dict[str, bool],
    call_samples: dict[str, list[float]],
    timing_repeats: int,
    rounds: int,
    latest_round_stats: dict[str, list[dict[str, Any]]],
    stats_repeats: int,
    stats_from_timing: bool,
) -> dict[str, Any]:
    cell_indices: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        cell_indices[(int(record["hidden"].shape[0]), record["routing"])].append(index)

    single_node_controls = tuple(
        control
        for control in ("single_node_node1", "single_node_node2", "single_node_node3")
        if control in controls and variant_validity.get(control, False)
    )
    cells = []
    for (token_count, routing), indices in cell_indices.items():
        calls_by_variant: dict[str, list[float]] = {}
        round_means_by_variant: dict[str, list[float]] = {}
        for variant, samples in call_samples.items():
            calls = []
            round_means = []
            for round_index in range(rounds):
                round_start = round_index * timing_repeats * len(records)
                round_calls = [
                    samples[round_start + repeat * len(records) + index]
                    for repeat in range(timing_repeats)
                    for index in indices
                ]
                calls.extend(round_calls)
                round_means.append(statistics.fmean(round_calls))
            calls_by_variant[variant] = calls
            round_means_by_variant[variant] = round_means

        reference_candidates = {
            control: _distribution(calls_by_variant[control])
            for control in single_node_controls
            if calls_by_variant.get(control)
        }
        reference = (
            min(
                reference_candidates,
                key=lambda control: reference_candidates[control]["mean"],
            )
            if reference_candidates
            else None
        )
        reference_mean_ms = (
            reference_candidates[reference]["mean"] if reference is not None else None
        )
        policy_results = {}
        policy_statuses = {}
        single_node_results = {}
        for control in (
            "single_node_node1",
            "single_node_node2",
            "single_node_node3",
        ):
            single_node_results[control] = {
                "numerically_valid": variant_validity.get(control, False),
                "complete_call_latency_ms": _distribution(
                    calls_by_variant.get(control, [])
                ),
                "per_round_mean_ms": round_means_by_variant.get(control, []),
            }
        for policy in policies:
            distribution = _distribution(calls_by_variant.get(policy, []))
            speedup = (
                reference_mean_ms / distribution["mean"]
                if (
                    variant_validity.get(policy, False)
                    and reference_mean_ms is not None
                    and distribution is not None
                    and distribution["mean"] > 0
                )
                else None
            )
            status = (
                "invalid_policy"
                if not variant_validity.get(policy, False)
                else (
                    "unavailable"
                    if speedup is None
                    else ("pass" if speedup >= 2.0 else "fail")
                )
            )
            policy_statuses[policy] = status
            policy_results[policy] = {
                "numerically_valid": variant_validity.get(policy, False),
                "complete_call_latency_ms": distribution,
                "per_round_mean_ms": round_means_by_variant.get(policy, []),
                "speedup_vs_fastest_valid_single_node": (
                    {
                        "reference": reference,
                        "reference_mean_ms": reference_mean_ms,
                        "speedup": speedup,
                    }
                    if speedup is not None
                    else None
                ),
                "two_x_status": status,
            }
            snapshots = latest_round_stats.get(policy, [])
            if snapshots:
                cell_stats = _native_stats_summary(
                    [
                        snapshots[repeat * len(records) + index]
                        for repeat in range(stats_repeats)
                        for index in indices
                    ]
                )
                cell_stats["timing_round_index"] = rounds if stats_from_timing else None
                if not stats_from_timing:
                    cell_stats["sampling_contract"] = (
                        "separate post-timing diagnostic corpus replay"
                    )
                policy_results[policy]["native_stats_summary"] = cell_stats

        measured_statuses = [
            status for status in policy_statuses.values() if status in ("pass", "fail")
        ]
        cell_status = (
            "pass"
            if "pass" in measured_statuses
            else ("fail" if measured_statuses else "unavailable")
        )
        cells.append(
            {
                "token_count": token_count,
                "routing": routing,
                "target_band": (
                    "small"
                    if token_count < 8
                    else "large"
                    if token_count > 128
                    else "middle"
                ),
                "fastest_valid_single_node_reference": (
                    {
                        "control": reference,
                        "mean_ms": reference_mean_ms,
                    }
                    if reference is not None
                    else None
                ),
                "single_node_controls": single_node_results,
                "policies": policy_results,
                "two_x_cell_status": cell_status,
                "two_x_status_by_policy": policy_statuses,
            }
        )
    return {
        "scope": "supplemental synthetic diagnostic; excluded from promotion gates",
        "timing_repeats": timing_repeats,
        "cache_condition": "sequence_replay",
        "record_sequence": "repeat the full ordered record list within each round",
        "cells": cells,
    }


def _sum_snapshot_field(stats_samples: list[dict[str, Any]], name: str) -> Any | None:
    values = [sample[name] for sample in stats_samples if name in sample]
    if not values:
        return None
    if all(isinstance(value, (int, float)) for value in values):
        return sum(values)
    if all(isinstance(value, list) for value in values):
        result = values[0]
        for value in values[1:]:
            if len(result) != len(value):
                return None
            result = _add_nested_lists(result, value)
        return result
    return values[-1]


def _add_nested_lists(left: list[Any], right: list[Any]) -> list[Any]:
    if all(isinstance(value, (int, float)) for value in left + right):
        return [a + b for a, b in zip(left, right)]
    if all(isinstance(value, list) for value in left + right):
        return [_add_nested_lists(a, b) for a, b in zip(left, right)]
    raise ValueError("Native counter snapshots have incompatible shapes")


def _native_stats_summary(stats_samples: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, list[float]] = defaultdict(list)
    node_times: dict[str, list[float]] = defaultdict(list)
    imbalances = []
    prediction_errors = []
    prediction_ratios = []
    for sample in stats_samples:
        phase_ms = sample.get("phase_ms")
        if isinstance(phase_ms, list) and len(phase_ms) == len(PHASE_NAMES):
            for name, value in zip(PHASE_NAMES, phase_ms):
                if isinstance(value, (float, int)):
                    phases[name].append(float(value))
        node_ms = sample.get("node_ms")
        if isinstance(node_ms, list):
            numeric = [
                float(value) for value in node_ms if isinstance(value, (int, float))
            ]
            for index, value in enumerate(node_ms):
                if isinstance(value, (int, float)):
                    node_times[f"node_{index}"].append(float(value))
            if numeric and statistics.fmean(numeric) > 0:
                imbalances.append(max(numeric) / statistics.fmean(numeric) - 1)
            predicted = sample.get("predicted_node_ms")
            if isinstance(predicted, list):
                for estimate, actual in zip(predicted, node_ms):
                    if estimate > 0 and actual > 0:
                        prediction_errors.append(abs(estimate - actual) / actual)
                        prediction_ratios.append(estimate / actual)

    phase_summary = {}
    for name, values in phases.items():
        phase_summary[name] = {
            **(_distribution(values) or {}),
            "sum_ms": sum(values),
        }
    counter_names = (
        "logical_counts",
        "physical_counts",
        "split_groups",
        "padded_rows",
        "dispatch_bytes",
        "result_bytes",
    )
    counters = {
        name: _sum_snapshot_field(stats_samples, name) for name in counter_names
    }
    weight_pages_by_layer = {
        str(sample["layer_id"]): sample["weight_pages"]
        for sample in stats_samples
        if "layer_id" in sample and "weight_pages" in sample
    }
    weight_addresses_by_layer = {
        str(sample["layer_id"]): sample["weight_addresses"]
        for sample in stats_samples
        if "layer_id" in sample and "weight_addresses" in sample
    }
    runtime_layout_by_layer = {}
    for sample in stats_samples:
        if "layer_id" not in sample:
            continue
        layout = runtime_layout_by_layer.setdefault(str(sample["layer_id"]), {})
        for name in (
            "scratch_pages",
            "shard_pages",
            "numa_nodes",
            "worker_cpus",
        ):
            if name in sample:
                layout[name] = sample[name]
    all_weight_addresses = [
        int(address)
        for addresses in weight_addresses_by_layer.values()
        for node_addresses in addresses
        for address in node_addresses
    ]
    weight_storage_distinct = (
        bool(all_weight_addresses)
        and all(address > 0 for address in all_weight_addresses)
        and len(set(all_weight_addresses)) == len(all_weight_addresses)
    )
    calibration_by_layer = {}
    for sample in stats_samples:
        if "layer_id" in sample and "calibrated_rows" in sample:
            entry = calibration_by_layer.setdefault(str(sample["layer_id"]), {})
            entry["rows"] = sample["calibrated_rows"]
            for name in (
                "cost_ms",
                "fixed_node_ms",
                "calibration_group_counts",
                "calibration_single_ms",
                "calibration_batch_ms",
                "calibration_shared_ms",
            ):
                if name in sample:
                    entry[name] = sample[name]
            if "calibration_single_ms" in sample:
                entry["single_fit_residual_ms"] = [
                    [
                        fixed + variable - measured
                        for variable, measured in zip(curve, singles)
                    ]
                    for fixed, curve, singles in zip(
                        sample["fixed_node_ms"],
                        sample["cost_ms"],
                        sample["calibration_single_ms"],
                    )
                ]
    placement_pages_present = bool(weight_pages_by_layer) and all(
        isinstance(pages, list)
        and len(pages) == 3
        and all(isinstance(count, (int, float)) and count > 0 for count in pages)
        for pages in weight_pages_by_layer.values()
    )
    return {
        "sampling_contract": "summed per-call latest-execute snapshots",
        "phase_ms": phase_summary,
        "node_ms": {name: _distribution(values) for name, values in node_times.items()},
        "node_imbalance_max_over_mean": _distribution(imbalances),
        "counters": counters,
        "weight_pages_by_layer": weight_pages_by_layer,
        "placement_pages_present_on_all_nodes": placement_pages_present,
        "weight_addresses_by_layer": weight_addresses_by_layer,
        "weight_storage_addresses_distinct": weight_storage_distinct,
        "runtime_layout_by_layer": runtime_layout_by_layer,
        "intermediate_slices_by_layer": {
            str(sample["layer_id"]): sample["intermediate_slices"]
            for sample in stats_samples
            if "layer_id" in sample and "intermediate_slices" in sample
        },
        "calibration_by_layer": calibration_by_layer,
        "node_prediction": {
            "scope": (
                "inner node dispatch, expert execution, and gather; "
                "excludes caller launch/wait, allocation, and final combine"
            ),
            "relative_absolute_error": _distribution(prediction_errors),
            "predicted_over_observed": _distribution(prediction_ratios),
        },
    }


def _create_runtime(
    packed_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cpus: tuple[int, ...],
    nodes: tuple[int, ...],
    threads_per_node: int,
    block_n: int,
    block_k: int,
) -> int:
    return int(
        torch.ops._C.cpu_numa_moe_create(
            *packed_weights,
            list(cpus),
            list(nodes),
            threads_per_node,
            block_n,
            block_k,
        )
    )


def _create_runtimes(
    layers: dict[int, dict[str, torch.Tensor]],
    cpus: tuple[int, ...],
    nodes: tuple[int, ...],
    threads_per_node: int,
    block_n: int,
    block_k: int,
    progress_path: str | None = None,
) -> dict[int, int]:
    handles = {}
    try:
        for layer_id in sorted(layers):
            packed = _prepare_packed_weights([layer_id], layers)[layer_id]
            handles[layer_id] = _create_runtime(
                packed, cpus, nodes, threads_per_node, block_n, block_k
            )
            del packed
            _progress(progress_path, "runtime_layer_created", layer=layer_id)
    except Exception:
        for handle in handles.values():
            torch.ops._C.cpu_numa_moe_destroy(handle)
        raise
    return handles


def _destroy_runtimes(handles: dict[int, int]) -> None:
    for handle in handles.values():
        torch.ops._C.cpu_numa_moe_destroy(handle)


def _execute(
    handles: dict[int, int],
    record: dict[str, Any],
    policy: str,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = record["hidden"]
    layer_id = record["layer"]
    if output is None:
        output = torch.empty_like(hidden)
    torch.ops._C.cpu_numa_moe_execute(
        handles[layer_id],
        hidden,
        record["ids"],
        record["weights"],
        output,
        policy,
        layer_id,
    )
    return output


def _check_output_numerics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float,
    rtol: float,
    context: dict[str, Any],
    failures: list[dict[str, Any]],
    allow_invalid: bool = False,
) -> None:
    try:
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    except AssertionError as error:
        if not allow_invalid:
            error.add_note(str(context))
            raise
        difference = (actual.float() - reference.float()).abs()
        failures.append(
            {
                **context,
                "error": str(error),
                "max_absolute_error": float(difference.max().item()),
                "mismatched_elements": int(
                    (difference > atol + rtol * reference.float().abs()).sum().item()
                ),
                "total_elements": actual.numel(),
            }
        )


def _make_reference_cache(
    directory: str | None,
    manifest: dict[str, Any],
    corpus_hash: str,
    block_n: int,
    block_k: int,
    chunk_tokens: int,
    read_only: bool = False,
) -> Any:
    if directory is None:
        return None
    from benchmarks.kernels.cpu.numa_moe_reference_cache import OracleCache

    identity = {
        "weights_file_sha256": manifest["weights_file_sha256"],
        "weights_tensor_sha256": manifest["weights_tensor_sha256"],
        "corpus_tensor_sha256": corpus_hash,
        "benchmark_source_sha256": _sha256_file(Path(__file__)),
        "cache_source_sha256": _sha256_file(
            Path(__file__).with_name("numa_moe_reference_cache.py")
        ),
        "block_n": block_n,
        "block_k": block_k,
        "reference_chunk_tokens": chunk_tokens,
        "reference_torch_threads": 1,
        "torch_version": str(torch.__version__),
    }
    return OracleCache(Path(directory), identity, read_only=read_only)


def _check_correctness(
    handles: dict[int, int],
    records: list[dict[str, Any]],
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
    block_n: int,
    block_k: int,
    atol: float,
    rtol: float,
    reference_chunk_tokens: int,
    policies: tuple[str, ...],
    progress_path: str | None = None,
    measure_invalid_controls: bool = False,
    reference_cache: Any = None,
) -> dict[str, Any]:
    by_layer: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_layer[record["layer"]].append((index, record))
    max_abs_error = 0.0
    max_rel_error = 0.0
    compared = 0
    alias_checks = 0
    route_stats_checks = 0
    captured_output_checks = 0
    numerical_failures: list[dict[str, Any]] = []
    expected_logical_counts = None

    for layer_id in layer_ids:
        layer_records = [record for _, record in by_layer[layer_id]]
        cached_outputs = (
            reference_cache.load(layer_id, layer_records)
            if reference_cache is not None
            else None
        )
        if cached_outputs is None:
            w = layers[layer_id]
            w1_dequant = _block_dequantize(
                w["w1"], w["s1"], block_n=block_n, block_k=block_k
            )
            w2_dequant = _block_dequantize(
                w["w2"], w["s2"], block_n=block_n, block_k=block_k
            )
            cached_outputs = [
                _reference_output(
                    record, w1_dequant, w2_dequant, reference_chunk_tokens
                )
                for record in layer_records
            ]
            if reference_cache is not None:
                reference_cache.store(layer_id, layer_records, cached_outputs)
            del w1_dequant, w2_dequant
            cache_event = "reference_layer_computed"
        else:
            cache_event = "reference_layer_reused"
        _progress(progress_path, cache_event, layer=layer_id)
        references = []
        for (record_index, record), reference in zip(
            by_layer[layer_id], cached_outputs
        ):
            references.append((record_index, record, reference))
            counts = torch.bincount(
                record["ids"].reshape(-1).long(),
                minlength=layers[layer_id]["w1"].shape[0],
            )
            if expected_logical_counts is None:
                expected_logical_counts = counts
            else:
                expected_logical_counts += counts
        for policy in policies:
            for record_index, record, reference in references:
                actual = _execute(handles, record, policy)
                allow_invalid = measure_invalid_controls and (
                    policy in KERNEL_CONTROLS or policy == "rows"
                )
                context = {
                    "record_index": record_index,
                    "layer": layer_id,
                    "step": record.get("step"),
                    "policy": policy,
                }
                if "reference_output" in record:
                    original_tolerance = (
                        atol if policy in KERNEL_CONTROLS or policy == "rows" else 0
                    )
                    _check_output_numerics(
                        actual,
                        record["reference_output"],
                        original_tolerance,
                        rtol if original_tolerance else 0,
                        {**context, "reference": "captured_operator"},
                        numerical_failures,
                        allow_invalid,
                    )
                    captured_output_checks += 1
                diff = (actual.float() - reference.float()).abs()
                max_abs_error = max(max_abs_error, float(diff.max().item()))
                denominator = reference.float().abs().clamp_min(1e-12)
                max_rel_error = max(
                    max_rel_error, float((diff / denominator).max().item())
                )
                _check_output_numerics(
                    actual,
                    reference,
                    atol,
                    rtol,
                    {**context, "reference": "independent_oracle"},
                    numerical_failures,
                    allow_invalid,
                )
                compared += 1
                runtime_stats = _call_stats(handles[layer_id])
                observed_logical = runtime_stats.get("logical_counts")
                record_counts = torch.bincount(
                    record["ids"].reshape(-1).long(),
                    minlength=layers[layer_id]["w1"].shape[0],
                )
                if observed_logical is not None:
                    actual_counts = torch.as_tensor(observed_logical, dtype=torch.int64)
                    torch.testing.assert_close(actual_counts, record_counts)
                    observed_physical = runtime_stats.get("physical_counts")
                    if observed_physical is not None:
                        physical_counts = torch.as_tensor(
                            observed_physical, dtype=torch.int64
                        )
                        multiplier = 1
                        if policy in KERNEL_CONTROLS:
                            slices = runtime_stats.get("intermediate_slices")
                            if slices is None:
                                raise AssertionError(
                                    f"{policy} did not report intermediate slices"
                                )
                            multiplier = sum(int(width > 0) for width in slices)
                        torch.testing.assert_close(
                            physical_counts.sum(dim=0),
                            record_counts * multiplier,
                        )
                    route_stats_checks += 1
                if record_index == 0:
                    aliased = record["hidden"].clone()
                    alias_result = _execute(
                        handles,
                        {**record, "hidden": aliased},
                        policy,
                        output=aliased,
                    )
                    torch.testing.assert_close(alias_result, actual, atol=0, rtol=0)
                    alias_checks += 1

        del references, cached_outputs
        _progress(progress_path, "native_correctness_layer_complete", layer=layer_id)
    return {
        "passed": not numerical_failures,
        "numerical_failures": numerical_failures,
        "records": len(records),
        "policy_outputs_compared": compared,
        "alias_outputs_compared": alias_checks,
        "native_route_stats_checked": route_stats_checks,
        "captured_operator_outputs_compared": captured_output_checks,
        "reference_semantics": (
            "independent FP32 Torch projections with original 32-row branch "
            "choices and BF16 stage rounding; approximate SiLU/accumulation; "
            "captured original operator parity checked separately"
        ),
        "max_absolute_error": max_abs_error,
        "max_relative_error": max_rel_error,
        "logical_route_counts": (
            expected_logical_counts.tolist()
            if expected_logical_counts is not None
            else []
        ),
        "atol": atol,
        "rtol": rtol,
    }


def _call_stats(handle: int) -> dict[str, Any]:
    return _stats_to_python(torch.ops._C.cpu_numa_moe_stats(handle))


def _run_corpus(
    handles: dict[int, int],
    records: list[dict[str, Any]],
    policy: str,
    repeats: int = 1,
    collect_stats: bool = True,
) -> tuple[list[float], list[dict[str, Any]], dict[str, list[float]]]:
    call_ms = []
    stats_samples = []
    concurrency_ms: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        for record in records:
            start = time.perf_counter_ns()
            _execute(handles, record, policy)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            call_ms.append(elapsed_ms)
            if record.get("concurrency") is not None:
                concurrency_ms[str(int(record["concurrency"]))].append(elapsed_ms)
            if collect_stats:
                stats = _call_stats(handles[record["layer"]])
                stats["layer_id"] = record["layer"]
                stats_samples.append(stats)
    return call_ms, stats_samples, concurrency_ms


def _run_existing_op(
    record: dict[str, Any],
    packed_by_layer: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    w1, w2, s1, s2 = packed_by_layer[record["layer"]]
    output = torch.empty_like(record["hidden"])
    custom_ops.fused_experts_cpu(
        output,
        record["hidden"],
        w1,
        w2,
        record["weights"],
        record["ids"],
        custom_ops.CPUQuantMethod.FP8_W8A16,
        s1,
        s2,
        None,
        None,
        [block_n, block_k],
        is_vnni=True,
    )
    return output


def _verify_existing_op(
    records: list[dict[str, Any]],
    layers: dict[int, dict[str, torch.Tensor]],
    packed_by_layer: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    block_n: int,
    block_k: int,
    atol: float,
    rtol: float,
    chunk_tokens: int,
    reference_cache: Any = None,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["layer"]].append(record)
    max_abs_error = 0.0
    compared = 0
    captured_compared = 0
    for layer_id, layer_records in grouped.items():
        references = (
            reference_cache.load(layer_id, layer_records)
            if reference_cache is not None
            else None
        )
        if references is None:
            layer = layers[layer_id]
            w1 = _block_dequantize(layer["w1"], layer["s1"], block_n, block_k)
            w2 = _block_dequantize(layer["w2"], layer["s2"], block_n, block_k)
            references = [
                _reference_output(record, w1, w2, chunk_tokens)
                for record in layer_records
            ]
            del w1, w2
        for record, reference in zip(layer_records, references):
            actual = _run_existing_op(record, packed_by_layer, block_n, block_k)
            if "reference_output" in record:
                torch.testing.assert_close(
                    actual,
                    record["reference_output"],
                    atol=0,
                    rtol=0,
                    msg=(
                        f"Existing operator differs from capture: layer {layer_id}, "
                        f"step {record.get('step')}"
                    ),
                )
                captured_compared += 1
            error = float((actual.float() - reference.float()).abs().max().item())
            max_abs_error = max(max_abs_error, error)
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            compared += 1
        del references
    return {
        "passed": True,
        "records_compared": compared,
        "captured_operator_outputs_compared": captured_compared,
        "max_absolute_error": max_abs_error,
        "reference_cache": (
            {
                "directory": str(reference_cache.directory),
                "identity": reference_cache.identity,
            }
            if reference_cache is not None
            else None
        ),
    }


def _copy_on_node(
    tensor: torch.Tensor, node: int, allocation_cpus: tuple[int, ...]
) -> tuple[torch.Tensor, int]:
    previous_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, set(allocation_cpus))
        copy = torch.ops._C.cpu_numa_moe_local_copy(tensor, node)
        if copy.data_ptr() == tensor.data_ptr():
            raise RuntimeError("NUMA local copy unexpectedly aliases its source")
        pages = int(torch.ops._C.cpu_numa_moe_verify_local(copy, node))
    finally:
        os.sched_setaffinity(0, previous_affinity)
    if pages <= 0:
        raise RuntimeError(f"NUMA local copy for node {node} has no local pages")
    return copy, pages


def _prepare_existing_weights(
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
    copy_node: int,
    allocation_cpus: tuple[int, ...],
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, dict[str, int]],
]:
    packed_by_layer = {}
    pages_by_layer = {}
    for layer_id in layer_ids:
        layer = layers[layer_id]
        packed_w1 = torch.ops._C.convert_weight_packed(layer["w1"].clone())
        packed_w2 = torch.ops._C.convert_weight_packed(layer["w2"].clone())
        sources = {
            "w1": packed_w1,
            "w2": packed_w2,
            "s1": layer["s1"],
            "s2": layer["s2"],
        }
        local = {}
        pages = {}
        for name, tensor in sources.items():
            local[name], pages[name] = _copy_on_node(tensor, copy_node, allocation_cpus)
        packed_by_layer[layer_id] = (
            local["w1"],
            local["w2"],
            local["s1"],
            local["s2"],
        )
        pages_by_layer[str(layer_id)] = pages
    return packed_by_layer, pages_by_layer


def _control_spec(
    name: str,
    cpus: tuple[int, ...],
    nodes: tuple[int, ...],
    threads_per_node: int,
) -> tuple[int, tuple[int, ...], int, tuple[int, ...]]:
    if name in NATIVE_TP2_CONTROLS:
        pair = tuple(int(index) for index in name[-2:])
        teams = [
            cpus[index * threads_per_node : (index + 1) * threads_per_node]
            for index in pair
        ]
        return nodes[pair[0]], teams[0] + teams[1], threads_per_node, teams[0]
    if name == "unsharded_96_node1_weights":
        node1_cpus = cpus[:threads_per_node]
        return nodes[0], cpus, len(cpus), node1_cpus
    target_node = int(name[-1])
    if target_node not in nodes:
        raise ValueError(f"Cannot map control {name} to requested NUMA nodes")
    node_index = nodes.index(target_node)
    team_cpus = cpus[
        node_index * threads_per_node : (node_index + 1) * threads_per_node
    ]
    return nodes[node_index], team_cpus, threads_per_node, team_cpus


def _control_worker_main(args: argparse.Namespace) -> None:
    control = args.control_worker
    if control in NATIVE_TP2_CONTROLS:
        import numa_moe_native_tp2

        numa_moe_native_tp2.worker_main(args)
        return
    copy_node, allowed_cpus, thread_count, allocation_cpus = _control_spec(
        control, args.cpus, args.nodes, args.threads_per_node
    )
    if not set(os.sched_getaffinity(0)).issubset(allowed_cpus):
        raise RuntimeError(
            f"Worker affinity {sorted(os.sched_getaffinity(0))} does not match "
            f"{control} CPUs {list(allowed_cpus)}"
        )
    # libiomp can bind the importing thread before this entry point.
    os.sched_setaffinity(0, set(allowed_cpus))
    torch.set_num_threads(thread_count)
    torch.ones(thread_count * 4096, dtype=torch.float32).sum().item()

    weights_path = Path(args.weights).resolve()
    layer_ids, layers = _load_weights(weights_path)
    if args.mode == "replay":
        records, metadata = _load_corpus(Path(args.corpus).resolve())
        _validate_records(records, layer_ids, layers)
        manifest = _manifest_payload(
            Path(args.corpus).resolve(),
            weights_path,
            records,
            metadata,
            layer_ids,
            layers,
        )
        _verify_manifest(Path(args.manifest), manifest)
    else:
        layer_id = _normalize_layer_key(args.layer_id)
        layer_ids = [layer_id]
        layers = {layer_id: layers[layer_id]}
        records = _make_synthetic_records(
            layers, layer_id, args.seed, args.synthetic_token_counts
        )
        manifest = {
            "weights_file_sha256": _sha256_file(weights_path),
            "weights_tensor_sha256": _content_hash(layers),
        }
    corpus_hash = _content_hash(records)
    if corpus_hash != args.expected_corpus_hash:
        raise RuntimeError("Worker loaded a different corpus than the coordinator")

    packed_by_layer, placement_pages = _prepare_existing_weights(
        layer_ids, layers, copy_node, allocation_cpus
    )
    reference_cache = _make_reference_cache(
        args.reference_cache_dir,
        manifest,
        corpus_hash,
        args.block_n,
        args.block_k,
        args.reference_chunk_tokens,
        read_only=True,
    )
    correctness = _verify_existing_op(
        records,
        layers,
        packed_by_layer,
        args.block_n,
        args.block_k,
        args.atol,
        args.rtol,
        args.reference_chunk_tokens,
        reference_cache,
    )
    _worker_send(
        {
            "event": "ready",
            "control": control,
            "copy_node": copy_node,
            "threads": thread_count,
            "torch_num_threads": torch.get_num_threads(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "thread_affinity_by_tid": _thread_affinity(allowed_cpus),
            "ld_preload": os.environ.get("LD_PRELOAD"),
            "preload_identity": _preload_identity(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
            "kmp_tpause": os.environ.get("KMP_TPAUSE"),
            "placement_pages": placement_pages,
            "correctness": correctness,
            "extension": _extension_identity(),
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        if command == "stop":
            _worker_send({"event": "stopped", "control": control})
            return
        if command == "diagnostic":
            from numa_moe_diagnostics import run_diagnostics

            def evict(record):
                for tensor in packed_by_layer[record["layer"]]:
                    torch.ops._C.cpu_numa_moe_evict_local(tensor)

            diagnostics = run_diagnostics(
                records,
                lambda record: _run_existing_op(
                    record, packed_by_layer, args.block_n, args.block_k
                ),
                evict,
                args.diagnostic_repeats,
            )
            _worker_send({"event": command, "diagnostics": diagnostics})
            continue
        if command not in ("warm", "run"):
            raise ValueError(f"Unknown worker command: {command!r}")
        call_ms = []
        by_concurrency: dict[str, list[float]] = defaultdict(list)
        for record in records:
            start = time.perf_counter_ns()
            _run_existing_op(record, packed_by_layer, args.block_n, args.block_k)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            call_ms.append(elapsed_ms)
            if record.get("concurrency") is not None:
                by_concurrency[str(int(record["concurrency"]))].append(elapsed_ms)
        _worker_send(
            {
                "event": command,
                "control": control,
                "total_ms": sum(call_ms),
                "call_ms": call_ms if command == "run" else [],
                "by_concurrency_ms": (dict(by_concurrency) if command == "run" else {}),
            }
        )


def _worker_send(message: dict[str, Any]) -> None:
    print("NUMA_MOE_WORKER " + json.dumps(message), flush=True)


def _worker_receive(process: subprocess.Popen[str]) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("Worker stdout was not captured")
    for line in process.stdout:
        if line.startswith("NUMA_MOE_WORKER "):
            return json.loads(line.removeprefix("NUMA_MOE_WORKER "))
    raise RuntimeError(f"NUMA control worker exited with status {process.poll()}")


def _worker_command(
    args: argparse.Namespace, control: str, corpus_hash: str
) -> tuple[list[str], dict[str, str]]:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required to isolate existing-op baselines")
    _, allowed_cpus, threads, _ = _control_spec(
        control, args.cpus, args.nodes, args.threads_per_node
    )
    command = [
        taskset,
        "--cpu-list",
        ",".join(map(str, allowed_cpus)),
        sys.executable,
        str(Path(__file__).resolve()),
        "--control-worker",
        control,
        "--mode",
        args.mode,
        "--weights",
        args.weights,
        "--cpus",
        ",".join(map(str, args.cpus)),
        "--nodes",
        ",".join(map(str, args.nodes)),
        "--threads-per-node",
        str(args.threads_per_node),
        "--block-n",
        str(args.block_n),
        "--block-k",
        str(args.block_k),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--reference-chunk-tokens",
        str(args.reference_chunk_tokens),
        "--seed",
        str(args.seed),
        "--diagnostic-repeats",
        str(args.diagnostic_repeats),
        "--expected-corpus-hash",
        corpus_hash,
    ]
    if args.mode == "replay":
        command.extend(
            [
                "--corpus",
                args.corpus,
                "--manifest",
                args.manifest,
            ]
        )
    else:
        command.extend(
            [
                "--layer-id",
                str(args.layer_id),
                "--synthetic-token-counts",
                ",".join(map(str, args.synthetic_token_counts)),
            ]
        )
    if args.measure_invalid_controls:
        command.append("--measure-invalid-controls")
    if args.reference_cache_dir:
        command.extend(["--reference-cache-dir", args.reference_cache_dir])
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    env["KMP_AFFINITY"] = (
        "granularity=fine,explicit,proclist=["
        + ",".join(map(str, allowed_cpus))
        + "],norespect"
    )
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_WAIT_POLICY"] = "ACTIVE"
    env["KMP_BLOCKTIME"] = "5"
    env["KMP_TPAUSE"] = "0"
    repo_root = str(_repo_root())
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            repo_root,
            *(entry for entry in inherited_pythonpath.split(os.pathsep) if entry),
        ]
    )
    return command, env


def _start_control_workers(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[dict[str, subprocess.Popen[str]], dict[str, Any]]:
    corpus_hash = _content_hash(records)
    workers: dict[str, subprocess.Popen[str]] = {}
    preflight = {}
    try:
        for control in args.controls:
            command, env = _worker_command(args, control, corpus_hash)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=_repo_root(),
            )
            workers[control] = process
            ready = _worker_receive(process)
            if ready.get("event") != "ready":
                raise RuntimeError(f"Control worker failed preflight: {ready}")
            if (
                ready.get("extension", {}).get("sha256")
                != _extension_identity()["sha256"]
            ):
                raise RuntimeError(f"{control} imported a different extension")
            preflight[control] = ready
            _progress(args.progress_jsonl, "control_ready", control=control)
    except Exception:
        _stop_control_workers(workers)
        raise
    return workers, preflight


def _control_request(
    process: subprocess.Popen[str], command: str, repeats: int = 1
) -> dict[str, Any]:
    if process.stdin is None:
        raise RuntimeError("Worker stdin was not captured")
    responses = []
    for _ in range(repeats):
        process.stdin.write(json.dumps({"command": command}) + "\n")
        process.stdin.flush()
        responses.append(_worker_receive(process))
    if len(responses) == 1:
        return responses[0]
    if command != "run":
        raise ValueError("Only measured run requests can be repeated")
    if any(response.get("event") != "run" for response in responses):
        raise RuntimeError(f"Control worker returned an invalid result: {responses}")

    call_ms = [value for response in responses for value in response["call_ms"]]
    by_concurrency: dict[str, list[float]] = defaultdict(list)
    for response in responses:
        for concurrency, values in response["by_concurrency_ms"].items():
            by_concurrency[concurrency].extend(values)
    combined = {
        **responses[-1],
        "total_ms": sum(call_ms),
        "call_ms": call_ms,
        "by_concurrency_ms": dict(by_concurrency),
    }
    if all("replay_wall_ms" in response for response in responses):
        combined["replay_wall_ms"] = sum(
            response["replay_wall_ms"] for response in responses
        )
        combined["rank_replay_wall_ms"] = [
            sum(response["rank_replay_wall_ms"][rank] for response in responses)
            for rank in range(len(responses[0]["rank_replay_wall_ms"]))
        ]
    return combined


def _stop_control_workers(workers: dict[str, subprocess.Popen[str]]) -> None:
    for process in workers.values():
        if process.poll() is None and process.stdin is not None:
            with suppress(BrokenPipeError, RuntimeError):
                _control_request(process, "stop")
        if process.poll() is None:
            process.wait()


def _bootstrap_ci(
    paired_ratios: list[float], seed: int = 20261008, samples: int = 20_000
) -> list[float] | None:
    if len(paired_ratios) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        chosen = [rng.choice(paired_ratios) for _ in paired_ratios]
        means.append(math.exp(statistics.fmean(math.log(value) for value in chosen)))
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def _paired_speedups(
    round_totals: dict[str, list[float]], reference: str = "static"
) -> dict[str, Any]:
    result = {}
    if reference not in round_totals:
        return result
    for policy, totals in round_totals.items():
        if policy == reference:
            continue
        paired = [
            ref_time / policy_time
            for ref_time, policy_time in zip(round_totals[reference], totals)
        ]
        result[policy] = {
            "reference_policy": reference,
            "per_round_speedup": paired,
            "geometric_mean_speedup": math.exp(
                statistics.fmean(math.log(value) for value in paired)
            ),
            "bootstrap_95_percent_ci": _bootstrap_ci(paired),
        }
    return result


def _needs_more_rounds(round_totals: dict[str, list[float]], threshold: float) -> bool:
    if len(next(iter(round_totals.values()), [])) < 2 or "static" not in round_totals:
        return False
    for policy, totals in round_totals.items():
        if policy == "static":
            continue
        ratios = [a / b for a, b in zip(round_totals["static"], totals)]
        if min(ratios) > 0 and max(ratios) / min(ratios) - 1 > threshold:
            return True
    return False


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _progress(args.progress_jsonl, "loading_inputs", mode=args.mode)
    corpus_path = Path(args.corpus).resolve() if args.corpus else None
    weights_path = Path(args.weights).resolve()
    layer_ids, layers = _load_weights(weights_path)
    if args.mode == "replay":
        if corpus_path is None:
            raise ValueError("--corpus is required for replay mode")
        records, corpus_metadata = _load_corpus(corpus_path)
        _validate_records(records, layer_ids, layers)
        manifest = _manifest_payload(
            corpus_path, weights_path, records, corpus_metadata, layer_ids, layers
        )
        if not args.manifest:
            raise ValueError(
                "Replay comparisons require --manifest from a prior freeze"
            )
        frozen_manifest = _verify_manifest(Path(args.manifest), manifest)
        primary_corpus = _primary_corpus_status(
            corpus_metadata,
            records,
            layer_ids,
            layers,
            args.block_n,
            args.block_k,
        )
    else:
        layer_id = _normalize_layer_key(args.layer_id)
        if layer_id not in layers:
            raise ValueError(f"--layer-id {layer_id} is missing from weights")
        layer_ids = [layer_id]
        layers = {layer_id: layers[layer_id]}
        records = _make_synthetic_records(
            layers, layer_id, args.seed, args.synthetic_token_counts
        )
        _validate_records(records, layer_ids, layers)
        manifest = {
            "format_version": 1,
            "weights_path": str(weights_path),
            "weights_file_sha256": _sha256_file(weights_path),
            "weights_tensor_sha256": _content_hash(layers),
            "synthetic_seed": args.seed,
            "synthetic_token_counts": list(args.synthetic_token_counts),
            "record_count": len(records),
            "corpus_tensor_sha256": _content_hash(records),
        }
        frozen_manifest = None
        corpus_metadata = {"kind": "synthetic routing diagnostic"}
        primary_corpus = _primary_corpus_status(
            corpus_metadata,
            records,
            layer_ids,
            layers,
            args.block_n,
            args.block_k,
        )

    _validate_topology(args.cpus, args.nodes, args.threads_per_node)
    _progress(
        args.progress_jsonl,
        "inputs_verified",
        records=len(records),
        layers=len(layer_ids),
        primary_eligible=primary_corpus.get("eligible"),
    )
    reference_cache = _make_reference_cache(
        args.reference_cache_dir,
        manifest,
        manifest["corpus_tensor_sha256"],
        args.block_n,
        args.block_k,
        args.reference_chunk_tokens,
    )
    handles = _create_runtimes(
        layers,
        args.cpus,
        args.nodes,
        args.threads_per_node,
        args.block_n,
        args.block_k,
        args.progress_jsonl,
    )
    _progress(args.progress_jsonl, "runtimes_created", layers=len(handles))
    if args.controller_cpu is not None:
        os.sched_setaffinity(0, {args.controller_cpu})
    control_workers: dict[str, subprocess.Popen[str]] = {}
    try:
        if args.calibration_rows:
            for layer, handle in handles.items():
                torch.ops._C.cpu_numa_moe_calibrate(handle, list(CALIBRATION_ROWS))
                _progress(
                    args.progress_jsonl, "calibration_layer_complete", layer=layer
                )

        host_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            correctness = _check_correctness(
                handles,
                records,
                layer_ids,
                layers,
                args.block_n,
                args.block_k,
                args.atol,
                args.rtol,
                args.reference_chunk_tokens,
                tuple(args.policies),
                args.progress_jsonl,
                args.measure_invalid_controls,
                reference_cache,
            )
            correctness["reference_torch_threads"] = torch.get_num_threads()
            if reference_cache is not None:
                correctness["reference_cache"] = {
                    "directory": str(reference_cache.directory),
                    "identity": reference_cache.identity,
                    "scope": "untimed independent reference tensors; checks unchanged",
                }
        finally:
            torch.set_num_threads(host_threads)
        _progress(
            args.progress_jsonl,
            "native_correctness_complete",
            passed=correctness["passed"],
            numerical_failures=correctness["numerical_failures"],
        )

        control_workers, control_preflight = _start_control_workers(args, records)
        invalid_variants = {
            failure["policy"] for failure in correctness["numerical_failures"]
        }
        invalid_variants.update(
            control
            for control, preflight in control_preflight.items()
            if not preflight["correctness"]["passed"]
        )
        variant_validity = {
            variant: variant not in invalid_variants
            for variant in (*args.policies, *args.controls)
        }
        if args.check_only:
            native_stats = []
            for layer in layer_ids:
                stats = _call_stats(handles[layer])
                stats["layer_id"] = layer
                native_stats.append(stats)
            native_layout = _native_stats_summary(native_stats)
            return {
                "format_version": 1,
                "mode": args.mode,
                "check_only": True,
                "source": _source_identity(Path(__file__).resolve()),
                "extension": _extension_identity(),
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "controller_cpu": args.controller_cpu,
                    "ld_preload": os.environ.get("LD_PRELOAD"),
                    "preload_identity": _preload_identity(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                    "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                    "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                    "kmp_tpause": os.environ.get("KMP_TPAUSE"),
                },
                "topology": {
                    "cpus": list(args.cpus),
                    "nodes": list(args.nodes),
                    "threads_per_node": args.threads_per_node,
                },
                "checkpoint_and_corpus": manifest,
                "frozen_manifest": frozen_manifest,
                "workload": {
                    "records": len(records),
                    "ordered_layers": [record["layer"] for record in records],
                    "phases": [record["phase"] for record in records],
                    "primary_corpus_validation": primary_corpus,
                },
                "calibration": {
                    "performed": bool(args.calibration_rows),
                    "rows": list(CALIBRATION_ROWS) if args.calibration_rows else [],
                },
                "correctness": correctness,
                "variant_numerical_validity": variant_validity,
                "native_stats_after_correctness": native_stats,
                "placement_verification": {
                    "weight_pages_reported_on_all_three_nodes": (
                        native_layout["placement_pages_present_on_all_nodes"]
                    ),
                    "weight_storage_addresses_distinct": (
                        native_layout["weight_storage_addresses_distinct"]
                    ),
                    "weight_pages_by_layer": native_layout["weight_pages_by_layer"],
                    "weight_addresses_by_layer": (
                        native_layout["weight_addresses_by_layer"]
                    ),
                    "existing_operator_controls": {
                        control: control_preflight[control]["placement_pages"]
                        for control in control_preflight
                    },
                },
                "existing_control_preflight": control_preflight,
                "timing": {"status": "not_run_check_only"},
                "promotion_gate": {
                    "status": "not_evaluated_check_only",
                    "controls_available": sorted(control_preflight),
                    "controls_unavailable": sorted(
                        set(EXISTING_CONTROLS) - set(control_preflight)
                    ),
                },
            }
        variants = [*args.policies, *args.controls]
        warmup_stats = {}
        for policy in args.policies:
            time.sleep(VARIANT_QUIESCENCE_SECONDS)
            _, samples, _ = _run_corpus(handles, records, policy)
            warmup_stats[policy] = samples[-1] if samples else {}
            _progress(args.progress_jsonl, "warmup_complete", variant=policy)
        for control in args.controls:
            time.sleep(VARIANT_QUIESCENCE_SECONDS)
            response = _control_request(control_workers[control], "warm")
            warmup_stats[control] = {
                "event": response["event"],
                "total_ms": response["total_ms"],
            }
            _progress(args.progress_jsonl, "warmup_complete", variant=control)

        round_totals = {variant: [] for variant in variants}
        control_wall_samples = {control: [] for control in args.controls}
        call_samples = {variant: [] for variant in variants}
        round_call_p95 = {variant: [] for variant in variants}
        final_stats = {policy: {} for policy in args.policies}
        latest_round_stats = {policy: [] for policy in args.policies}
        stats_from_timing = args.native_stats_mode == "per-call"
        stats_repeats = (
            args.timing_repeats if stats_from_timing else args.diagnostic_repeats
        )
        concurrency_round_totals: dict[str, dict[str, list[float]]] = {
            variant: defaultdict(list) for variant in variants
        }
        concurrency_call_samples: dict[str, dict[str, list[float]]] = {
            variant: defaultdict(list) for variant in variants
        }
        rounds = args.rounds
        round_index = 0
        variant_order_by_round = []
        counter_results = []
        layout_inputs = {}
        layout_setup = []
        for record in records:
            layer = record["layer"]
            if layer not in layout_inputs:
                layout_inputs[layer] = {
                    "layer": layer,
                    **{
                        name: record[name][:1].contiguous()
                        for name in ("hidden", "ids", "weights")
                    },
                }
        if args.perf_output_dir:
            from numa_moe_perf import PerfCounters

            Path(args.perf_output_dir).mkdir(parents=True, exist_ok=True)
        while round_index < rounds:
            order = list(variants)
            offset = round_index % len(order)
            order = order[offset:] + order[:offset]
            variant_order_by_round.append(order)
            for variant in order:
                # Both native and OpenMP teams spin for up to 5 ms after work.
                time.sleep(VARIANT_QUIESCENCE_SECONDS)
                if variant in KERNEL_CONTROLS:
                    setup_start = time.perf_counter()
                    for record in layout_inputs.values():
                        _execute(handles, record, variant)
                    setup = {
                        "round": round_index + 1,
                        "variant": variant,
                        "layers": len(layout_inputs),
                        "elapsed_ms": (time.perf_counter() - setup_start) * 1000,
                    }
                    layout_setup.append(setup)
                    _progress(args.progress_jsonl, "kernel_layout_ready", **setup)
                    time.sleep(VARIANT_QUIESCENCE_SECONDS)
                collector = (
                    PerfCounters(
                        Path(args.perf_output_dir)
                        / f"round{round_index + 1}_{variant}",
                        args.cpus,
                        args.perf_events,
                    )
                    if args.perf_output_dir
                    else None
                )
                with collector if collector is not None else nullcontext():
                    if variant in args.policies:
                        timings, samples, by_concurrency = _run_corpus(
                            handles,
                            records,
                            variant,
                            args.timing_repeats,
                            collect_stats=stats_from_timing,
                        )
                        final_stats[variant] = samples[-1] if samples else {}
                        latest_round_stats[variant] = samples
                    else:
                        response = _control_request(
                            control_workers[variant],
                            "run",
                            repeats=args.timing_repeats,
                        )
                        if response.get("event") != "run":
                            raise RuntimeError(
                                f"Control worker returned an invalid result: {response}"
                            )
                        timings = response["call_ms"]
                        by_concurrency = response["by_concurrency_ms"]
                        if "replay_wall_ms" in response:
                            control_wall_samples[variant].append(
                                {
                                    "replay_wall_ms": response["replay_wall_ms"],
                                    "rank_replay_wall_ms": (
                                        response["rank_replay_wall_ms"]
                                    ),
                                }
                            )
                if collector is not None:
                    counter_sample = {
                        "round": round_index + 1,
                        "variant": variant,
                        **collector.result,
                    }
                    measured = {
                        row["event"]: float(row["value"])
                        for row in collector.result["parsed_counters"]
                    }
                    references = measured.get("longest_lat_cache.reference")
                    misses = measured.get("longest_lat_cache.miss")
                    counter_sample["llc_miss_ratio"] = (
                        misses / references
                        if references and misses is not None
                        else None
                    )
                    for direction in ("read", "write"):
                        events = [
                            f"uncore_imc/cas_count_{direction}_sch{index}/"
                            for index in (0, 1)
                        ]
                        if all(event in measured for event in events):
                            traffic_mib = sum(measured[event] for event in events)
                            counter_sample[f"imc_{direction}_mib"] = traffic_mib
                            counter_sample[f"imc_{direction}_mib_per_second"] = (
                                traffic_mib / collector.result["enabled_wall_seconds"]
                            )
                    counter_results.append(counter_sample)
                round_totals[variant].append(sum(timings))
                call_samples[variant].extend(timings)
                round_call_p95[variant].append(_distribution(timings)["p95"])
                for concurrency, values in by_concurrency.items():
                    concurrency_round_totals[variant][concurrency].append(sum(values))
                    concurrency_call_samples[variant][concurrency].extend(values)
                _progress(
                    args.progress_jsonl,
                    "round_variant_complete",
                    round=round_index + 1,
                    variant=variant,
                    total_ms=sum(timings),
                )
            round_index += 1
            if (
                round_index == args.rounds
                and round_index < args.max_rounds
                and _needs_more_rounds(round_totals, threshold=0.05)
            ):
                rounds = args.max_rounds

        if not stats_from_timing:
            for policy in args.policies:
                time.sleep(VARIANT_QUIESCENCE_SECONDS)
                if policy in KERNEL_CONTROLS:
                    for record in layout_inputs.values():
                        _execute(handles, record, policy)
                    time.sleep(VARIANT_QUIESCENCE_SECONDS)
                _, samples, _ = _run_corpus(handles, records, policy, stats_repeats)
                latest_round_stats[policy] = samples
                final_stats[policy] = samples[-1] if samples else {}
                _progress(
                    args.progress_jsonl,
                    "post_timing_stats_complete",
                    variant=policy,
                    repeats=stats_repeats,
                    records=len(records),
                )

        policy_results = {}
        routed_tokens = sum(
            int(record["hidden"].shape[0] * record["ids"].shape[1])
            for record in records
        )
        logical_groups = (
            sum(int(torch.unique(record["ids"]).numel()) for record in records)
            * stats_repeats
        )
        for variant in variants:
            total_ms = statistics.fmean(round_totals[variant])
            by_concurrency = {}
            for concurrency, totals in concurrency_round_totals[variant].items():
                routed_tokens_for_group = (
                    sum(
                        int(record["hidden"].shape[0] * record["ids"].shape[1])
                        for record in records
                        if record.get("concurrency") is not None
                        and str(int(record["concurrency"])) == concurrency
                    )
                    * args.timing_repeats
                )
                mean_ms = statistics.fmean(totals)
                group_calls = concurrency_call_samples[variant][concurrency]
                calls_per_round = len(group_calls) // rounds
                by_concurrency[concurrency] = {
                    "round_total_ms": totals,
                    "mean_replay_ms": mean_ms,
                    "useful_routed_tokens_per_second": (
                        routed_tokens_for_group / (mean_ms / 1000)
                    ),
                    "call_latency_ms": _distribution(
                        concurrency_call_samples[variant][concurrency]
                    ),
                    "call_p95_ms_by_round": [
                        _distribution(
                            group_calls[
                                index * calls_per_round : (index + 1) * calls_per_round
                            ]
                        )["p95"]
                        for index in range(rounds)
                    ],
                }
            result_for_variant = {
                "numerically_valid": variant_validity[variant],
                "timing_evidence": (
                    "complete_call_measurement"
                    if variant_validity[variant]
                    else "diagnostic_only_invalid_control_outputs"
                ),
                "round_total_ms": round_totals[variant],
                "mean_replay_ms": total_ms,
                "useful_routed_tokens_per_second": (
                    routed_tokens * args.timing_repeats / (total_ms / 1000)
                ),
                "call_latency_ms": _distribution(call_samples[variant]),
                "by_concurrency": by_concurrency,
            }
            by_phase = {}
            for phase in sorted({record["phase"] for record in records}):
                indices = [
                    index
                    for index, record in enumerate(records)
                    if record["phase"] == phase
                ]
                phase_calls = [
                    call_samples[variant][
                        round_id * args.timing_repeats * len(records)
                        + repeat_id * len(records)
                        + index
                    ]
                    for round_id in range(rounds)
                    for repeat_id in range(args.timing_repeats)
                    for index in indices
                ]
                phase_round_ms = [
                    sum(
                        call_samples[variant][
                            round_id * args.timing_repeats * len(records)
                            + repeat_id * len(records)
                            + index
                        ]
                        for index in indices
                        for repeat_id in range(args.timing_repeats)
                    )
                    for round_id in range(rounds)
                ]
                by_phase[phase] = {
                    "calls_per_round": len(indices) * args.timing_repeats,
                    "round_total_ms": phase_round_ms,
                    "mean_replay_ms": statistics.fmean(phase_round_ms),
                    "call_latency_ms": _distribution(phase_calls),
                }
            result_for_variant["by_phase"] = by_phase
            if variant in args.policies:
                native_summary = _native_stats_summary(latest_round_stats[variant])
                native_summary["timing_round_index"] = (
                    rounds if stats_from_timing else None
                )
                if not stats_from_timing:
                    native_summary["sampling_contract"] = (
                        "separate post-timing diagnostic corpus replay"
                    )
                split_groups = native_summary["counters"].get("split_groups")
                native_summary["split_frequency"] = (
                    split_groups / logical_groups
                    if isinstance(split_groups, (int, float)) and logical_groups
                    else None
                )
                result_for_variant.update(
                    native_stats_summary=native_summary,
                    native_stats_last_call=final_stats[variant],
                    execution_kind=(
                        "experimental_intermediate_shard_kernel_control"
                        if variant in KERNEL_CONTROLS
                        else "three_node_full_width_replica_policy"
                    ),
                )
            else:
                result_for_variant.update(
                    execution_kind=(
                        "existing_cpu_operator_tp2_with_cpushm"
                        if variant in NATIVE_TP2_CONTROLS
                        else (
                            "existing_unsharded_operator"
                            if variant == "unsharded_96_node1_weights"
                            else "existing_single_node_operator"
                        )
                    ),
                    control_preflight=control_preflight[variant],
                    replay_wall_by_round=control_wall_samples[variant],
                )
            policy_results[variant] = result_for_variant
        synthetic_complete_call = (
            _synthetic_complete_call_summary(
                records,
                tuple(args.policies),
                tuple(args.controls),
                variant_validity,
                call_samples,
                args.timing_repeats,
                rounds,
                latest_round_stats,
                stats_repeats,
                stats_from_timing,
            )
            if args.mode == "synthetic"
            else None
        )
        paired = _paired_speedups(round_totals)
        eligible_dynamic = [
            policy for policy in ("balanced", "split") if policy in paired
        ]
        selected_dynamic = (
            min(
                eligible_dynamic,
                key=lambda policy: policy_results[policy]["mean_replay_ms"],
            )
            if eligible_dynamic
            else None
        )
        selected_comparison = paired.get(selected_dynamic, {})
        selected_ci = selected_comparison.get("bootstrap_95_percent_ci")
        selected_speedup = selected_comparison.get("geometric_mean_speedup")
        fixed_time_reduction = (
            1
            - policy_results[selected_dynamic]["mean_replay_ms"]
            / policy_results["static"]["mean_replay_ms"]
            if selected_dynamic is not None and "static" in policy_results
            else None
        )
        fixed_advantage = bool(
            fixed_time_reduction is not None
            and fixed_time_reduction >= 0.05
            and selected_ci
            and selected_ci[0] > 1.0
        )
        complete_native_policy_set = set(POLICIES).issubset(args.policies)
        complete_existing_controls = set(SINGLE_COPY_CONTROLS).issubset(args.controls)
        native_pages_verified = bool(
            "static" in policy_results
            and policy_results["static"]["native_stats_summary"][
                "placement_pages_present_on_all_nodes"
            ]
        )
        native_addresses_distinct = bool(
            "static" in policy_results
            and policy_results["static"]["native_stats_summary"][
                "weight_storage_addresses_distinct"
            ]
        )
        placement_verified = native_pages_verified and native_addresses_distinct
        single_node_controls = [
            "single_node_node1",
            "single_node_node2",
            "single_node_node3",
        ]
        speedups_vs_fastest_node = {}
        for concurrency in ("8", "32", "128"):
            available = [
                control
                for control in single_node_controls
                if control in concurrency_round_totals
                and concurrency in concurrency_round_totals[control]
            ]
            if (
                selected_dynamic is None
                or concurrency not in concurrency_round_totals.get(selected_dynamic, {})
                or not available
            ):
                speedups_vs_fastest_node[concurrency] = None
                continue
            fastest_node = min(
                available,
                key=lambda control: statistics.fmean(
                    concurrency_round_totals[control][concurrency]
                ),
            )
            baseline_ms = statistics.fmean(
                concurrency_round_totals[fastest_node][concurrency]
            )
            dynamic_ms = statistics.fmean(
                concurrency_round_totals[selected_dynamic][concurrency]
            )
            speedups_vs_fastest_node[concurrency] = {
                "reference": fastest_node,
                "reference_mean_ms": baseline_ms,
                "dynamic_mean_ms": dynamic_ms,
                "speedup": baseline_ms / dynamic_ms,
            }
        two_x_pass = bool(
            primary_corpus["eligible"]
            and selected_dynamic is not None
            and all(
                result is not None and result["speedup"] >= 2.0
                for result in speedups_vs_fastest_node.values()
            )
        )
        measured_alternatives = [
            variant
            for variant in variants
            if variant != selected_dynamic
            and round_totals[variant]
            and variant_validity[variant]
        ]
        strongest_alternative = (
            min(
                measured_alternatives,
                key=lambda variant: statistics.fmean(round_totals[variant]),
            )
            if measured_alternatives
            else None
        )
        strongest_ratio = (
            statistics.fmean(round_totals[selected_dynamic])
            / statistics.fmean(round_totals[strongest_alternative])
            if selected_dynamic is not None and strongest_alternative is not None
            else None
        )
        no_large_regression = bool(
            strongest_ratio is not None and strongest_ratio <= 1.05
        )
        repeated_p95_regressions = 0
        p95_comparison_available = bool(
            selected_dynamic is not None
            and strongest_alternative is not None
            and len(round_call_p95[selected_dynamic]) == rounds
            and len(round_call_p95[strongest_alternative]) == rounds
        )
        if p95_comparison_available:
            paired_p95 = zip(
                round_call_p95[selected_dynamic],
                round_call_p95[strongest_alternative],
            )
            repeated_p95_regressions = sum(
                candidate > baseline * 1.10 for candidate, baseline in paired_p95
            )
        p95_by_concurrency = {}
        if p95_comparison_available:
            for concurrency in ("8", "32", "128"):
                candidate = policy_results[selected_dynamic]["by_concurrency"].get(
                    concurrency
                )
                baseline = policy_results[strongest_alternative]["by_concurrency"].get(
                    concurrency
                )
                if candidate is None or baseline is None:
                    continue
                candidate_p95 = candidate["call_p95_ms_by_round"]
                baseline_p95 = baseline["call_p95_ms_by_round"]
                regressions = sum(
                    value > reference * 1.10
                    for value, reference in zip(candidate_p95, baseline_p95)
                )
                p95_by_concurrency[concurrency] = {
                    "round_count": regressions,
                    "selected_p95_ms_by_round": candidate_p95,
                    "control_p95_ms_by_round": baseline_p95,
                }
        p95_gate_pass = (
            p95_comparison_available
            and repeated_p95_regressions < 2
            and len(p95_by_concurrency) == 3
            and all(group["round_count"] < 2 for group in p95_by_concurrency.values())
        )
        required_controls = {
            "fixed_expert_ownership_static": (
                "measured in the shared native runtime"
                if "static" in args.policies
                else "not selected"
            ),
            "balanced_whole_expert_groups": (
                "measured" if "balanced" in args.policies else "not selected"
            ),
            "split_hot_groups": "measured"
            if "split" in args.policies
            else "not selected",
            "equal_row_partition": "measured"
            if "rows" in args.policies
            else "not selected",
            "single_node_existing_operator": (
                "measured" if complete_existing_controls else "incomplete"
            ),
            "single_node_node1": (
                "measured" if "single_node_node1" in args.controls else "unavailable"
            ),
            "single_node_node2": (
                "measured" if "single_node_node2" in args.controls else "unavailable"
            ),
            "single_node_node3": (
                "measured" if "single_node_node3" in args.controls else "unavailable"
            ),
            "96_thread_unsharded_existing_operator": (
                "measured"
                if "unsharded_96_node1_weights" in args.controls
                else "unavailable"
            ),
            "unequal_256_128_128_intermediate_slices": (
                "measured" if "unequal" in args.policies else "unavailable"
            ),
            "tp2_01_kernel_control": (
                "measured" if "tp2_01" in args.policies else "unavailable"
            ),
            "tp2_02_kernel_control": (
                "measured" if "tp2_02" in args.policies else "unavailable"
            ),
            "tp2_12_kernel_control": (
                "measured" if "tp2_12" in args.policies else "unavailable"
            ),
            "native_moe_tp2": (
                "measured: existing operator and native CPU SHM reduction"
                if set(NATIVE_TP2_CONTROLS).issubset(args.controls)
                else "unavailable"
            ),
        }
        controls_complete = all(
            required_controls[name].startswith("measured")
            for name in (
                "single_node_existing_operator",
                "single_node_node1",
                "single_node_node2",
                "single_node_node3",
                "96_thread_unsharded_existing_operator",
                "native_moe_tp2",
                "tp2_01_kernel_control",
                "tp2_02_kernel_control",
                "tp2_12_kernel_control",
                "unequal_256_128_128_intermediate_slices",
            )
        )
        primary_environment = (
            tuple(args.cpus) == DEFAULT_CPUS
            and tuple(args.nodes) == DEFAULT_NODES
            and args.threads_per_node == 32
            and args.block_n == args.block_k == 128
            and args.atol <= 1e-2
            and args.rtol <= 1e-2
            and args.rounds >= 5
            and args.max_rounds >= 10
            and not args.perf_output_dir
            and stats_from_timing
        )
        primary_eligible = primary_corpus["eligible"] and primary_environment
        preload = _preload_identity()
        preload_verified = preload["verified"]
        if args.perf_output_dir:
            overall_status = "not_evaluated_instrumented_diagnostic"
        elif not stats_from_timing:
            overall_status = "not_evaluated_post_timing_stats_diagnostic"
        elif not controls_complete or not primary_eligible:
            overall_status = "blocked_missing_controls_or_primary_corpus"
        elif not placement_verified or not preload_verified:
            overall_status = "blocked_unverified_placement_or_preload"
        elif invalid_variants:
            overall_status = "fail_invalid_control_outputs"
        elif (
            complete_native_policy_set
            and fixed_advantage
            and two_x_pass
            and no_large_regression
            and p95_gate_pass
        ):
            overall_status = "pass"
        else:
            overall_status = "fail_native_dynamic_policy_gate"
        from numa_moe_diagnostics import run_diagnostics

        diagnostics = {}
        for policy in args.policies:
            diagnostics[policy] = run_diagnostics(
                records,
                lambda record, policy=policy: _execute(handles, record, policy),
                lambda record: torch.ops._C.cpu_numa_moe_evict(
                    handles[record["layer"]]
                ),
                args.diagnostic_repeats,
            )
        for control in args.controls:
            response = _control_request(control_workers[control], "diagnostic")
            if response.get("event") != "diagnostic":
                raise RuntimeError(f"Invalid diagnostic response: {response}")
            diagnostics[control] = response["diagnostics"]
        selected_summary = (
            policy_results[selected_dynamic]["native_stats_summary"]
            if selected_dynamic is not None
            else {}
        )
        reported_phases = {
            name: values["sum_ms"]
            for name, values in selected_summary.get("phase_ms", {}).items()
            if name != "total"
        }
        unmet_gates = [
            name
            for name, passed in (
                ("primary_corpus_and_environment", primary_eligible),
                ("required_controls", controls_complete),
                ("all_measured_outputs_numerically_valid", not invalid_variants),
                ("placement_and_preload", placement_verified and preload_verified),
                ("five_percent_over_static_with_confidence", fixed_advantage),
                ("two_x_at_every_primary_concurrency", two_x_pass),
                ("within_five_percent_of_strongest_control", no_large_regression),
                ("no_repeated_p95_regressions", p95_gate_pass),
            )
            if not passed
        ]
        result = {
            "format_version": 1,
            "mode": args.mode,
            "source": _source_identity(Path(__file__).resolve()),
            "extension": _extension_identity(),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "controller_cpu": args.controller_cpu,
                "ld_preload": os.environ.get("LD_PRELOAD"),
                "preload_identity": preload,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                "kmp_tpause": os.environ.get("KMP_TPAUSE"),
                "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
                "omp_places": os.environ.get("OMP_PLACES"),
            },
            "topology": {
                "cpus": list(args.cpus),
                "nodes": list(args.nodes),
                "threads_per_node": args.threads_per_node,
            },
            "preflight": {
                "runtime_handles_created": len(handles),
                "layer_ids": layer_ids,
                "initialization_in_timing": False,
                "shared_worker_teams": 3,
                "threads_per_team": args.threads_per_node,
                "controls": required_controls,
            },
            "placement_verification": {
                "weight_pages_by_layer": policy_results.get("static", {})
                .get("native_stats_summary", {})
                .get("weight_pages_by_layer", {}),
                "pages_reported_on_all_three_nodes": native_pages_verified,
                "storage_addresses_distinct": native_addresses_distinct,
                "weight_addresses_by_layer": (
                    policy_results.get("static", {})
                    .get("native_stats_summary", {})
                    .get("weight_addresses_by_layer", {})
                ),
                "runtime_layout_by_layer": (
                    policy_results.get("static", {})
                    .get("native_stats_summary", {})
                    .get("runtime_layout_by_layer", {})
                ),
                "existing_operator_controls": {
                    control: control_preflight[control]["placement_pages"]
                    for control in control_preflight
                },
            },
            "checkpoint_and_corpus": manifest,
            "frozen_manifest": frozen_manifest,
            "synthetic_complete_call": synthetic_complete_call,
            "workload": {
                "records": len(records),
                "ordered_layers": [record["layer"] for record in records],
                "phases": [record["phase"] for record in records],
                "concurrencies": sorted(
                    {
                        int(record["concurrency"])
                        for record in records
                        if record.get("concurrency") is not None
                    }
                ),
                "routed_tokens": routed_tokens,
                "synthetic_token_counts": (
                    list(args.synthetic_token_counts)
                    if args.mode == "synthetic"
                    else None
                ),
                "metadata": corpus_metadata,
                "primary_corpus_validation": primary_corpus,
            },
            "calibration": {
                "performed": bool(args.calibration_rows),
                "rows": list(CALIBRATION_ROWS) if args.calibration_rows else [],
                "included_in_timing": False,
            },
            "correctness": correctness,
            "timing": {
                "instrumented": bool(args.perf_output_dir),
                "native_stats_mode": args.native_stats_mode,
                "native_stats_repeats": stats_repeats,
                "native_stats_from_timing": stats_from_timing,
                "repeats_per_round": args.timing_repeats,
                "records_per_round": len(records) * args.timing_repeats,
                "untimed_pause_between_variants_ms": (
                    VARIANT_QUIESCENCE_SECONDS * 1000
                ),
                "hardware_counters": {
                    "scope": (
                        "system-wide on requested physical CPUs; includes replay "
                        "bookkeeping and background activity on these CPUs; "
                        "uncore events have socket scope"
                    ),
                    "excluded_stages": [
                        "initialization",
                        "calibration",
                        "correctness",
                        "warmup",
                        "kernel_control_weight_layout_installation",
                        "isolated_diagnostics",
                    ],
                    "samples": counter_results,
                },
                "rounds": rounds,
                "minimum_rounds": args.rounds,
                "maximum_rounds": args.max_rounds,
                "variant_order_by_round": variant_order_by_round,
                "kernel_layout_setup_excluded_from_timing": layout_setup,
                "paired_speedups_vs_static": paired,
                "paired_speedups_by_concurrency": {
                    concurrency: _paired_speedups(
                        {
                            policy: totals[concurrency]
                            for policy, totals in concurrency_round_totals.items()
                            if concurrency in totals
                        }
                    )
                    for concurrency in sorted(
                        {
                            concurrency
                            for totals in concurrency_round_totals.values()
                            for concurrency in totals
                        },
                        key=int,
                    )
                },
                "policies": policy_results,
                "warmup_last_stats": warmup_stats,
            },
            "promotion_gate": {
                "overall_status": overall_status,
                "primary_corpus_eligible": primary_eligible,
                "all_required_native_policies_measured": complete_native_policy_set,
                "all_existing_operator_controls_measured": (
                    set(EXISTING_CONTROLS).issubset(args.controls)
                ),
                "weight_pages_reported_for_all_nodes": native_pages_verified,
                "weight_storage_addresses_distinct": native_addresses_distinct,
                "selected_dynamic_policy": selected_dynamic,
                "variant_numerical_validity": variant_validity,
                "invalid_control_outputs": sorted(invalid_variants),
                "five_percent_over_static": {
                    "status": "pass" if fixed_advantage else "fail_or_inconclusive",
                    "speedup": selected_speedup,
                    "aggregate_time_reduction_fraction": fixed_time_reduction,
                    "paired_ci_excludes_parity": bool(
                        selected_ci and selected_ci[0] > 1.0
                    ),
                    "speedup_ci": selected_ci,
                },
                "two_x_vs_fastest_single_node": {
                    "status": (
                        "pass"
                        if two_x_pass
                        else (
                            "fail"
                            if primary_eligible
                            and all(
                                value is not None
                                for value in speedups_vs_fastest_node.values()
                            )
                            else "unavailable_or_nonprimary"
                        )
                    ),
                    "by_concurrency": speedups_vs_fastest_node,
                    "measured": (
                        selected_dynamic is not None
                        and all(
                            value is not None
                            for value in speedups_vs_fastest_node.values()
                        )
                    ),
                    "criterion": "at least 2.0x at concurrency 8, 32, and 128",
                },
                "no_more_than_five_percent_regression_vs_strongest_control": {
                    "status": (
                        "pass"
                        if no_large_regression
                        else ("fail" if strongest_ratio is not None else "unavailable")
                    ),
                    "strongest_alternative": strongest_alternative,
                    "selected_to_control_time_ratio": strongest_ratio,
                },
                "repeated_p95_regressions_above_ten_percent": {
                    "status": (
                        "pass"
                        if p95_gate_pass
                        else ("fail" if p95_comparison_available else "unavailable")
                    ),
                    "round_count": repeated_p95_regressions,
                    "comparison_policy": strongest_alternative,
                    "by_concurrency": p95_by_concurrency,
                },
                "required_preload": {
                    "status": "pass" if preload_verified else "missing_or_unverified",
                    "ld_preload": preload["ld_preload"],
                    "verified_paths_and_hashes": preload["components"],
                    "all_components_verified": preload["verified"],
                    "required_components": ["tcmalloc", "libiomp5"],
                },
                "warm_layer_and_cache_cold_diagnostics": {
                    "status": "measured",
                    "scope": "supplemental; excluded from promotion timing",
                    "cache_cold_scope": "packed weights and scales flushed before call",
                    "variants": diagnostics,
                },
                "primary_environment_verified": primary_environment,
                "failure_diagnosis": {
                    "unmet_gates": unmet_gates,
                    "selected_policy": selected_dynamic,
                    "sampling_contract": selected_summary.get("sampling_contract"),
                    "native_stats_repeats": stats_repeats,
                    "timing_round_index": selected_summary.get("timing_round_index"),
                    "largest_reported_phase": (
                        max(reported_phases, key=reported_phases.get)
                        if reported_phases
                        else None
                    ),
                    "phase_sum_ms": reported_phases,
                    "node_imbalance": selected_summary.get(
                        "node_imbalance_max_over_mean"
                    ),
                    "split_frequency": selected_summary.get("split_frequency"),
                    "strongest_control": strongest_alternative,
                    "traffic_counters": (
                        "measured in this instrumented diagnostic"
                        if counter_results
                        else "not collected; byte estimates are a model"
                    ),
                    "primary_performance_evidence": primary_eligible,
                    "decision": (
                        "retain_for_serving_design_review"
                        if overall_status == "pass"
                        else (
                            "reject_serving_promotion"
                            if overall_status.startswith("fail")
                            else "retain_prototype_pending_valid_primary_replay"
                        )
                    ),
                },
                "required_control_availability": required_controls,
                "required_controls_complete": controls_complete,
                "recommendation": (
                    "do not advance to serving integration from this harness result"
                    if overall_status != "pass"
                    else "eligible for a separate serving-integration review"
                ),
            },
        }
        return result
    finally:
        try:
            _stop_control_workers(control_workers)
        finally:
            _destroy_runtimes(handles)


def _freeze(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus).resolve()
    weights_path = Path(args.weights).resolve()
    records, metadata = _load_corpus(corpus_path)
    layer_ids, layers = _load_weights(weights_path)
    _validate_records(records, layer_ids, layers)
    manifest = _manifest_payload(
        corpus_path, weights_path, records, metadata, layer_ids, layers
    )
    output = Path(args.freeze_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline replay/synthetic harness for cpu_numa_moe."
    )
    parser.add_argument("--mode", choices=("replay", "synthetic"), default="replay")
    parser.add_argument("--corpus", help="torch file containing an ordered record list")
    parser.add_argument(
        "--weights", required=True, help="torch layer-to-weights mapping"
    )
    parser.add_argument("--manifest", help="frozen replay manifest required for replay")
    parser.add_argument(
        "--freeze-manifest",
        help="validate inputs, write a corpus/checkpoint manifest, and exit",
    )
    parser.add_argument("--layer-id", type=int, help="synthetic mode layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synthetic-token-counts",
        type=_parse_positive_int_list,
        default=SYNTHETIC_TOKEN_COUNTS,
        metavar="M[,M...]",
        help="positive synthetic token counts (default: existing matrix)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run reference, alias, routing, and placement checks without timing",
    )
    parser.add_argument(
        "--measure-invalid-controls",
        action="store_true",
        help=(
            "retain numerical failures and time invalid row/sharding controls "
            "as diagnostics; any invalid output blocks promotion"
        ),
    )
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=list(POLICIES)
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=EXISTING_CONTROLS,
        default=list(EXISTING_CONTROLS),
    )
    parser.add_argument("--cpus", type=_parse_int_list, default=DEFAULT_CPUS)
    parser.add_argument("--nodes", type=_parse_int_list, default=DEFAULT_NODES)
    parser.add_argument("--threads-per-node", type=int, default=32)
    parser.add_argument(
        "--controller-cpu",
        type=int,
        help="pin the administrative caller after creating the expert teams",
    )
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="repeat the full ordered record sequence in each measured round",
    )
    parser.add_argument("--diagnostic-repeats", type=int, default=5)
    parser.add_argument(
        "--native-stats-mode",
        choices=("per-call", "post-timing"),
        default="per-call",
        help=(
            "post-timing skips native stats extraction between measured calls, "
            "then gathers diagnostics with --diagnostic-repeats; excludes promotion"
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--reference-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--reference-cache-dir",
        help="cache independent reference tensors for untimed control validation",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_false",
        dest="calibration_rows",
        help="do not calibrate (balanced/split are invalid without it)",
    )
    parser.add_argument("--output", help="write full JSON result to this path")
    parser.add_argument("--progress-jsonl", help="append untimed stage progress events")
    parser.add_argument(
        "--perf-output-dir",
        help="collect separate per-variant hardware counters; disables promotion",
    )
    parser.add_argument(
        "--perf-events",
        nargs="+",
        default=[
            "cycles",
            "instructions",
            "longest_lat_cache.reference",
            "longest_lat_cache.miss",
            "uncore_imc/cas_count_read_sch0/",
            "uncore_imc/cas_count_read_sch1/",
            "uncore_imc/cas_count_write_sch0/",
            "uncore_imc/cas_count_write_sch1/",
        ],
    )
    parser.add_argument(
        "--control-worker",
        choices=EXISTING_CONTROLS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--expected-corpus-hash", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _validate_source_context()
    if args.control_worker:
        if not args.expected_corpus_hash:
            raise SystemExit("A control worker requires --expected-corpus-hash")
        _control_worker_main(args)
        return
    if args.freeze_manifest:
        if not args.corpus:
            raise SystemExit("--corpus is required with --freeze-manifest")
        _freeze(args)
        return
    if args.rounds <= 0 or args.max_rounds < args.rounds:
        raise SystemExit("Require 0 < --rounds <= --max-rounds")
    if args.timing_repeats <= 0:
        raise SystemExit("--timing-repeats must be positive")
    if args.reference_chunk_tokens <= 0:
        raise SystemExit("--reference-chunk-tokens must be positive")
    if args.diagnostic_repeats <= 0:
        raise SystemExit("--diagnostic-repeats must be positive")
    if len(set(args.policies)) != len(args.policies):
        raise SystemExit("--policies cannot contain duplicates")
    if len(set(args.controls)) != len(args.controls):
        raise SystemExit("--controls cannot contain duplicates")
    if set(args.controls) != set(EXISTING_CONTROLS):
        print(
            "Promotion remains blocked unless all seven existing-op controls run.",
            file=sys.stderr,
        )
    if not args.policies:
        raise SystemExit("Select at least one native policy")
    if not args.calibration_rows and set(args.policies) & {"balanced", "split"}:
        raise SystemExit("balanced and split require frozen calibration")
    if args.mode == "synthetic" and args.layer_id is None:
        raise SystemExit("--layer-id is required in synthetic mode")
    if args.mode == "replay" and not args.corpus:
        raise SystemExit("--corpus is required in replay mode")
    _validate_topology(args.cpus, args.nodes, args.threads_per_node)
    if args.controls and tuple(args.nodes) != DEFAULT_NODES:
        raise SystemExit("Existing-op controls are fixed to NUMA nodes 1, 2, and 3")

    original_affinity = os.sched_getaffinity(0)
    if args.controller_cpu is not None and args.controller_cpu not in original_affinity:
        raise SystemExit("--controller-cpu must belong to the caller's allowed CPUs")
    try:
        os.sched_setaffinity(0, set(args.cpus))
    except OSError as exc:
        raise SystemExit(f"Cannot apply requested CPU affinity: {exc}") from exc
    try:
        result = _run_benchmark(args)
    finally:
        os.sched_setaffinity(0, original_affinity)
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

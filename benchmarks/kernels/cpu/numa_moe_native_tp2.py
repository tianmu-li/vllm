# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline two-node CPU TP2 control for the NUMA MoE replay harness."""

from __future__ import annotations

import datetime
import json
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.kernels.cpu import benchmark_cpu_numa_moe as replay
from benchmarks.kernels.cpu.numa_moe_diagnostics import run_diagnostics

CONTROL_PAIRS = {
    "native_tp2_01": (0, 1),
    "native_tp2_02": (0, 2),
    "native_tp2_12": (1, 2),
}
PROCESS_GROUP_TIMEOUT = datetime.timedelta(minutes=5)
RANK_START_TIMEOUT_SECONDS = 3600
COMMAND_TIMEOUT_SECONDS = 3600
RANK_STOP_TIMEOUT_SECONDS = 30
RANK_JOIN_TIMEOUT_SECONDS = 10


def _pair_topology(
    control: str,
    cpus: tuple[int, ...],
    nodes: tuple[int, ...],
    threads_per_node: int,
) -> tuple[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]:
    if control not in CONTROL_PAIRS:
        raise ValueError(f"Unsupported native TP2 control {control!r}")
    indices = CONTROL_PAIRS[control]
    teams = [
        tuple(cpus[index * threads_per_node : (index + 1) * threads_per_node])
        for index in indices
    ]
    if any(len(team) != threads_per_node for team in teams):
        raise ValueError("Each TP2 rank must receive one complete CPU team")
    return (
        (nodes[indices[0]], nodes[indices[1]]),
        (teams[0], teams[1]),
    )


def _slice_layer(
    layer: dict[str, torch.Tensor],
    rank: int,
    block_n: int,
    block_k: int,
) -> dict[str, torch.Tensor]:
    w1, w2, s1, s2 = (
        layer["w1"],
        layer["w2"],
        layer["s1"],
        layer["s2"],
    )
    experts, gate_up_size, hidden_size = w1.shape
    intermediate_size = gate_up_size // 2
    if (
        gate_up_size != 2 * intermediate_size
        or w2.shape != (experts, hidden_size, intermediate_size)
        or intermediate_size % 2
        or intermediate_size % block_n
        or intermediate_size % block_k
        or (intermediate_size // 2) % block_n
        or (intermediate_size // 2) % block_k
        or hidden_size % block_n
        or hidden_size % block_k
    ):
        raise ValueError(
            "Native TP2 requires matching gate/up and down shapes with "
            "block-aligned equal intermediate slices"
        )

    expected_s1 = (
        experts,
        gate_up_size // block_n,
        hidden_size // block_k,
    )
    expected_s2 = (
        experts,
        hidden_size // block_n,
        intermediate_size // block_k,
    )
    if tuple(s1.shape) != expected_s1 or tuple(s2.shape) != expected_s2:
        raise ValueError(
            f"Block scale shapes must be {expected_s1} and {expected_s2}, got "
            f"{tuple(s1.shape)} and {tuple(s2.shape)}"
        )

    slice_size = intermediate_size // 2
    start = rank * slice_size
    end = start + slice_size
    gate_up = torch.cat(
        (
            w1[:, start:end, :],
            w1[:, intermediate_size + start : intermediate_size + end, :],
        ),
        dim=1,
    ).contiguous()
    down = w2[:, :, start:end].contiguous()

    gate_scale_start = start // block_n
    gate_scale_end = end // block_n
    up_scale_start = intermediate_size // block_n + gate_scale_start
    up_scale_end = intermediate_size // block_n + gate_scale_end
    scale_indices = torch.cat(
        (
            torch.arange(gate_scale_start, gate_scale_end),
            torch.arange(up_scale_start, up_scale_end),
        )
    )
    sliced_s1 = s1.index_select(1, scale_indices).contiguous()
    sliced_s2 = s2[:, :, start // block_k : end // block_k].contiguous()
    return {"w1": gate_up, "w2": down, "s1": sliced_s1, "s2": sliced_s2}


def _load_inputs(
    config: dict[str, Any],
) -> tuple[
    list[int],
    dict[int, dict[str, torch.Tensor]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    weights_path = Path(config["weights"]).resolve()
    layer_ids, layers = replay._load_weights(weights_path)
    if config["mode"] == "replay":
        corpus_path = Path(config["corpus"]).resolve()
        records, metadata = replay._load_corpus(corpus_path)
        replay._validate_records(records, layer_ids, layers)
        manifest = replay._manifest_payload(
            corpus_path,
            weights_path,
            records,
            metadata,
            layer_ids,
            layers,
        )
        replay._verify_manifest(Path(config["manifest"]), manifest)
    else:
        layer_id = replay._normalize_layer_key(config["layer_id"])
        if layer_id not in layers:
            raise ValueError(f"Synthetic layer {layer_id} is absent from weights")
        layer_ids = [layer_id]
        layers = {layer_id: layers[layer_id]}
        records = replay._make_synthetic_records(
            layers, layer_id, config["seed"], tuple(config["synthetic_token_counts"])
        )
        metadata = {"kind": "synthetic routing diagnostic"}
        manifest = {
            "weights_file_sha256": replay._sha256_file(weights_path),
            "weights_tensor_sha256": replay._content_hash(layers),
        }

    corpus_hash = replay._content_hash(records)
    if corpus_hash != config["expected_corpus_hash"]:
        raise ValueError("TP2 rank loaded a corpus different from its coordinator")
    return (
        layer_ids,
        layers,
        records,
        {
            "corpus_hash": corpus_hash,
            "metadata": metadata,
            "manifest": manifest,
            "weights_file_sha256": manifest["weights_file_sha256"],
            "weights_tensor_sha256": manifest["weights_tensor_sha256"],
        },
    )


def _prepare_rank_weights(
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
    rank: int,
    node: int,
    cpus: tuple[int, ...],
    block_n: int,
    block_k: int,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, dict[str, int]],
]:
    packed_by_layer = {}
    pages_by_layer = {}
    for layer_id in layer_ids:
        sharded_layer = {
            layer_id: _slice_layer(layers[layer_id], rank, block_n, block_k)
        }
        packed, pages = replay._prepare_existing_weights(
            [layer_id], sharded_layer, node, cpus
        )
        packed_by_layer.update(packed)
        pages_by_layer.update(pages)
        del sharded_layer, packed, pages
    return packed_by_layer, pages_by_layer


def _execute_partial(
    record: dict[str, Any],
    packed_by_layer: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    communicator: Any,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    partial = replay._run_existing_op(record, packed_by_layer, block_n, block_k)
    communicator.all_reduce(partial)
    return partial


def _extension_report() -> dict[str, Any]:
    identity = replay._extension_identity()
    identity["cpu_shm_schemas"] = {
        name: str(getattr(torch.ops._C, name).default._schema)
        for name in ("init_shm_manager", "join_shm_manager", "shm_allreduce")
    }
    return identity


def _check_outputs(
    records: list[dict[str, Any]],
    layer_ids: list[int],
    layers: dict[int, dict[str, torch.Tensor]],
    packed_by_layer: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    communicator: Any,
    config: dict[str, Any],
    reference_cache: Any = None,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["layer"]].append(record)

    compared = 0
    maximum_error = 0.0
    captured_compared = 0
    captured_maximum_error = 0.0
    numerical_failures: list[dict[str, Any]] = []
    allow_invalid = bool(config["measure_invalid_controls"])
    for layer_id in layer_ids:
        layer_records = grouped[layer_id]
        references = (
            reference_cache.load(layer_id, layer_records)
            if reference_cache is not None
            else None
        )
        if references is None:
            weights = layers[layer_id]
            w1 = replay._block_dequantize(
                weights["w1"], weights["s1"], config["block_n"], config["block_k"]
            )
            w2 = replay._block_dequantize(
                weights["w2"], weights["s2"], config["block_n"], config["block_k"]
            )
            references = [
                replay._reference_output(
                    record, w1, w2, config["reference_chunk_tokens"]
                )
                for record in layer_records
            ]
            del w1, w2
        for record, reference in zip(layer_records, references):
            actual = _execute_partial(
                record,
                packed_by_layer,
                communicator,
                config["block_n"],
                config["block_k"],
            )
            if "reference_output" in record:
                captured = record["reference_output"]
                difference = (actual.float() - captured.float()).abs()
                captured_maximum_error = max(
                    captured_maximum_error, float(difference.max().item())
                )
                replay._check_output_numerics(
                    actual,
                    captured,
                    config["atol"],
                    config["rtol"],
                    {
                        "control": config["control"],
                        "layer": layer_id,
                        "step": record.get("step"),
                        "reference": "captured_operator",
                    },
                    numerical_failures,
                    allow_invalid,
                )
                captured_compared += 1
            difference = (actual.float() - reference.float()).abs()
            maximum_error = max(maximum_error, float(difference.max().item()))
            replay._check_output_numerics(
                actual,
                reference,
                config["atol"],
                config["rtol"],
                {
                    "control": config["control"],
                    "layer": layer_id,
                    "step": record.get("step"),
                    "reference": "independent_oracle",
                },
                numerical_failures,
                allow_invalid,
            )
            compared += 1
        del references
    return {
        "passed": not numerical_failures,
        "allow_invalid_controls": allow_invalid,
        "numerical_failures": numerical_failures,
        "reference_cache": (
            {
                "directory": str(reference_cache.directory),
                "identity": reference_cache.identity,
            }
            if reference_cache is not None
            else None
        ),
        "records_compared": compared,
        "captured_operator_outputs_compared": captured_compared,
        "max_absolute_error": maximum_error,
        "captured_max_absolute_error": captured_maximum_error,
        "atol": config["atol"],
        "rtol": config["rtol"],
        "collective_order": "same per-layer grouped order on both ranks",
        "timed_record_order": "original corpus order",
    }


def _rank_process_main(
    rank: int,
    config: dict[str, Any],
    connection: Connection,
) -> None:
    initialized = False
    try:
        cpu_team = tuple(config["cpus_by_rank"][rank])
        os.sched_setaffinity(0, set(cpu_team))
        if set(os.sched_getaffinity(0)) != set(cpu_team):
            raise RuntimeError("TP2 rank did not receive its requested CPU team")
        torch.set_num_threads(config["threads"])
        torch.ones(config["threads"] * 4096, dtype=torch.float32).sum().item()
        replay._validate_source_context()

        os.environ["VLLM_DIST_IDENT"] = config["dist_ident"]
        layer_ids, layers, records, hashes = _load_inputs(config)
        node = config["nodes"][rank]
        packed_by_layer, pages = _prepare_rank_weights(
            layer_ids,
            layers,
            rank,
            node,
            cpu_team,
            config["block_n"],
            config["block_k"],
        )

        dist.init_process_group(
            backend="gloo",
            init_method=config["init_method"],
            rank=rank,
            world_size=2,
            timeout=PROCESS_GROUP_TIMEOUT,
        )
        initialized = True
        from vllm.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
            _CPUSHMDistributed,
        )

        process_group = dist.group.WORLD
        communicator = CpuCommunicator(
            cpu_group=process_group,
            device=torch.device("cpu"),
            device_group=process_group,
            unique_name="tp_numa_replay",
        )
        if not isinstance(communicator.dist_module, _CPUSHMDistributed):
            raise RuntimeError(
                "CPU TP2 requires _CPUSHMDistributed; torch.distributed "
                "fallback is not accepted"
            )

        reference_cache = replay._make_reference_cache(
            config.get("reference_cache_dir"),
            hashes["manifest"],
            hashes["corpus_hash"],
            config["block_n"],
            config["block_k"],
            config["reference_chunk_tokens"],
            read_only=True,
        )
        correctness = _check_outputs(
            records,
            layer_ids,
            layers,
            packed_by_layer,
            communicator,
            config,
            reference_cache,
        )
        extension = _extension_report()
        if extension["sha256"] is None:
            raise RuntimeError("Imported vllm._C extension has no verifiable file")
        connection.send(
            {
                "event": "ready",
                "rank": rank,
                "control": config["control"],
                "node": node,
                "cpus": list(cpu_team),
                "threads": torch.get_num_threads(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "thread_affinity_by_tid": replay._thread_affinity(cpu_team),
                "preload_identity": replay._preload_identity(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                "kmp_tpause": os.environ.get("KMP_TPAUSE"),
                "placement_pages": pages,
                "corpus_hash": hashes["corpus_hash"],
                "weights_file_sha256": hashes["weights_file_sha256"],
                "weights_tensor_sha256": hashes["weights_tensor_sha256"],
                "extension": extension,
                "communicator": {
                    "implementation": type(communicator.dist_module).__name__,
                    "unique_name": "tp_numa_replay",
                    "dist_ident": config["dist_ident"],
                    "world_size": 2,
                    "process_group_backend": "gloo",
                    "reduction_backend": "cpu_shm",
                },
                "correctness": correctness,
            }
        )

        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "stop":
                dist.barrier()
                connection.send({"event": "stopped", "rank": rank})
                return
            if command == "diagnostic":
                repeats = config["diagnostic_repeats"]

                def execute_diagnostic(
                    record: dict[str, Any],
                ) -> torch.Tensor:
                    return _execute_partial(
                        record,
                        packed_by_layer,
                        communicator,
                        config["block_n"],
                        config["block_k"],
                    )

                def evict_diagnostic(record: dict[str, Any]) -> None:
                    for tensor in packed_by_layer[record["layer"]]:
                        torch.ops._C.cpu_numa_moe_evict_local(tensor)
                    dist.barrier()

                dist.barrier()
                result = run_diagnostics(
                    records,
                    execute_diagnostic,
                    evict_diagnostic,
                    repeats,
                )
                connection.send(
                    {
                        "event": "diagnostic",
                        "rank": rank,
                        "diagnostics": result,
                    }
                )
                continue
            if command not in ("warm", "run"):
                raise ValueError(f"Unknown TP2 worker command: {command!r}")

            # All ranks leave this synchronization before starting call timing.
            dist.barrier()
            replay_start = time.perf_counter_ns()
            call_ms = []
            by_concurrency: dict[str, list[float]] = defaultdict(list)
            for record in records:
                call_start = time.perf_counter_ns()
                _execute_partial(
                    record,
                    packed_by_layer,
                    communicator,
                    config["block_n"],
                    config["block_k"],
                )
                elapsed = (time.perf_counter_ns() - call_start) / 1e6
                call_ms.append(elapsed)
                concurrency = record.get("concurrency")
                if concurrency is not None:
                    by_concurrency[str(int(concurrency))].append(elapsed)
            replay_ms = (time.perf_counter_ns() - replay_start) / 1e6
            connection.send(
                {
                    "event": command,
                    "rank": rank,
                    "replay_wall_ms": replay_ms,
                    "call_ms": call_ms,
                    "by_concurrency_ms": (dict(by_concurrency)),
                }
            )
    except BaseException:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(
                {
                    "event": "error",
                    "rank": rank,
                    "error": traceback.format_exc(),
                }
            )
        raise
    finally:
        if initialized and dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()
        connection.close()


def _receive_rank_messages(
    connections: dict[int, Connection],
    processes: dict[int, multiprocessing.Process],
    event: str,
    timeout_seconds: float,
) -> dict[int, dict[str, Any]]:
    pending = set(connections)
    responses = {}
    deadline = time.monotonic() + timeout_seconds
    while pending:
        for rank in tuple(pending):
            connection = connections[rank]
            if connection.poll(0.05):
                response = connection.recv()
                if response.get("event") == "error":
                    raise RuntimeError(
                        f"TP2 rank {rank} failed:\n{response.get('error', '')}"
                    )
                if response.get("event") != event:
                    raise RuntimeError(
                        f"TP2 rank {rank} returned {response.get('event')!r}, "
                        f"expected {event!r}"
                    )
                responses[rank] = response
                pending.remove(rank)
            elif processes[rank].exitcode is not None:
                raise RuntimeError(
                    f"TP2 rank {rank} exited with status "
                    f"{processes[rank].exitcode} before {event!r}"
                )
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for TP2 ranks: {sorted(pending)}")
    return responses


def _exchange_rank_command(
    command: str,
    connections: dict[int, Connection],
    processes: dict[int, multiprocessing.Process],
) -> dict[int, dict[str, Any]]:
    for connection in connections.values():
        connection.send({"command": command})
    timeout = (
        RANK_STOP_TIMEOUT_SECONDS if command == "stop" else COMMAND_TIMEOUT_SECONDS
    )
    return _receive_rank_messages(
        connections,
        processes,
        "stopped" if command == "stop" else command,
        timeout,
    )


def _combine_rank_timings(
    responses: dict[int, dict[str, Any]], command: str, control: str
) -> dict[str, Any]:
    rank_ids = sorted(responses)
    rank_replay_walls = [float(responses[rank]["replay_wall_ms"]) for rank in rank_ids]
    result = {
        "event": command,
        "control": control,
        "replay_wall_ms": max(rank_replay_walls),
        "rank_replay_wall_ms": rank_replay_walls,
    }
    call_lists = [responses[rank]["call_ms"] for rank in rank_ids]
    if len({len(calls) for calls in call_lists}) != 1:
        raise RuntimeError("TP2 ranks timed different numbers of replay records")
    combined_calls = [
        max(float(calls[index]) for calls in call_lists)
        for index in range(len(call_lists[0]))
    ]
    result["total_ms"] = sum(combined_calls)
    result["call_ms"] = combined_calls if command == "run" else []
    concurrency_names = set(responses[rank_ids[0]]["by_concurrency_ms"])
    if any(
        set(responses[rank]["by_concurrency_ms"]) != concurrency_names
        for rank in rank_ids[1:]
    ):
        raise RuntimeError("TP2 ranks reported different concurrency groups")
    combined_concurrency = {}
    for name in concurrency_names:
        values = [responses[rank]["by_concurrency_ms"][name] for rank in rank_ids]
        if len({len(items) for items in values}) != 1:
            raise RuntimeError(
                f"TP2 ranks timed different record counts at concurrency {name}"
            )
        combined_concurrency[name] = [
            max(float(items[index]) for items in values)
            for index in range(len(values[0]))
        ]
    result["by_concurrency_ms"] = combined_concurrency if command == "run" else {}
    return result


def _combine_rank_diagnostics(
    responses: dict[int, dict[str, Any]], control: str
) -> dict[str, Any]:
    rank_ids = sorted(responses)
    rank_diagnostics = {str(rank): responses[rank]["diagnostics"] for rank in rank_ids}
    repeat_counts = {rank_diagnostics[str(rank)]["repeats"] for rank in rank_ids}
    if len(repeat_counts) != 1:
        raise RuntimeError("TP2 ranks used different diagnostic repeat counts")
    sample_lists = [rank_diagnostics[str(rank)]["samples"] for rank in rank_ids]
    if len({len(samples) for samples in sample_lists}) != 1:
        raise RuntimeError("TP2 ranks selected different diagnostic representatives")

    repeats = repeat_counts.pop()
    samples = []
    for sample_index in range(len(sample_lists[0])):
        rank_samples = [rank_samples[sample_index] for rank_samples in sample_lists]
        representatives = {
            json.dumps(sample["representative"], sort_keys=True)
            for sample in rank_samples
        }
        if len(representatives) != 1:
            raise RuntimeError("TP2 ranks selected different representative records")
        if any(
            len(sample["warm_layer_ms"]) != repeats
            or len(sample["cold_weights_ms"]) != repeats
            for sample in rank_samples
        ):
            raise RuntimeError("TP2 rank diagnostic latency count is incomplete")
        samples.append(
            {
                "representative": rank_samples[0]["representative"],
                "warm_layer_ms": [
                    max(
                        float(sample["warm_layer_ms"][index]) for sample in rank_samples
                    )
                    for index in range(repeats)
                ],
                "cold_weights_ms": [
                    max(
                        float(sample["cold_weights_ms"][index])
                        for sample in rank_samples
                    )
                    for index in range(repeats)
                ],
            }
        )

    return {
        "event": "diagnostic",
        "control": control,
        "diagnostics": {
            "repeats": repeats,
            "samples": samples,
            "rank_raw_results": rank_diagnostics,
        },
    }


def _stop_rank_processes(
    connections: dict[int, Connection],
    processes: dict[int, multiprocessing.Process],
    graceful: bool,
) -> None:
    clean_exit_expected = graceful
    if graceful and processes:
        try:
            _exchange_rank_command("stop", connections, processes)
        except Exception:
            graceful = False
            clean_exit_expected = False
    if not graceful:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
    errors = []
    for process in processes.values():
        process.join(timeout=RANK_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=RANK_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            errors.append(f"could not stop rank process {process.name}")
        if clean_exit_expected and process.exitcode != 0:
            errors.append(
                f"rank process {process.name} exited with status {process.exitcode}"
            )
    for connection in connections.values():
        connection.close()
    if errors:
        raise RuntimeError("TP2 worker cleanup failed: " + "; ".join(errors))


def _start_rank_processes(
    config: dict[str, Any],
) -> tuple[
    dict[int, multiprocessing.Process],
    dict[int, Connection],
    dict[int, dict[str, Any]],
]:
    context = multiprocessing.get_context("spawn")
    processes = {}
    connections = {}
    try:
        for rank in (0, 1):
            parent_connection, child_connection = context.Pipe(duplex=True)
            process = context.Process(
                target=_rank_process_main,
                args=(rank, config, child_connection),
                name=f"numa-moe-tp2-rank-{rank}",
                daemon=True,
            )
            try:
                original_affinity = os.sched_getaffinity(0)
                original_kmp_affinity = os.environ.get("KMP_AFFINITY")
                try:
                    # Constrain the fresh interpreter before libiomp import.
                    os.sched_setaffinity(0, config["cpus_by_rank"][rank])
                    os.environ["KMP_AFFINITY"] = (
                        "granularity=fine,explicit,proclist=["
                        + ",".join(map(str, config["cpus_by_rank"][rank]))
                        + "],norespect"
                    )
                    process.start()
                finally:
                    os.sched_setaffinity(0, original_affinity)
                    if original_kmp_affinity is None:
                        os.environ.pop("KMP_AFFINITY", None)
                    else:
                        os.environ["KMP_AFFINITY"] = original_kmp_affinity
            except Exception:
                parent_connection.close()
                child_connection.close()
                raise
            child_connection.close()
            processes[rank] = process
            connections[rank] = parent_connection
        ready = _receive_rank_messages(
            connections,
            processes,
            "ready",
            RANK_START_TIMEOUT_SECONDS,
        )
        return processes, connections, ready
    except Exception:
        _stop_rank_processes(connections, processes, graceful=False)
        raise


def _ready_payload(
    control: str,
    nodes: tuple[int, int],
    cpus: tuple[tuple[int, ...], tuple[int, ...]],
    ready: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    ranks = [ready[rank] for rank in sorted(ready)]
    for key in ("corpus_hash", "weights_file_sha256", "weights_tensor_sha256"):
        if len({rank[key] for rank in ranks}) != 1:
            raise RuntimeError(f"TP2 ranks disagree on {key}")
    extension_hashes = {rank["extension"]["sha256"] for rank in ranks}
    extension_paths = {rank["extension"]["path"] for rank in ranks}
    if len(extension_hashes) != 1 or len(extension_paths) != 1:
        raise RuntimeError("TP2 ranks imported different native extensions")
    communicators = {json.dumps(rank["communicator"], sort_keys=True) for rank in ranks}
    if (
        len(communicators) != 1
        or ranks[0]["communicator"]["implementation"] != "_CPUSHMDistributed"
    ):
        raise RuntimeError("TP2 ranks did not initialize the same CPU SHM group")
    all_ranks_passed = all(bool(rank["correctness"]["passed"]) for rank in ranks)
    return {
        "event": "ready",
        "control": control,
        "execution_kind": "existing_cpu_operator_tp2_with_cpushm",
        "scope": "MoE replay control; not full-model TP2 throughput",
        "topology": {
            "nodes": list(nodes),
            "cpus_by_rank": [list(team) for team in cpus],
            "threads_per_rank": len(cpus[0]),
        },
        "effective_affinity_by_rank": [rank["affinity"] for rank in ranks],
        "threads_by_rank": [rank["threads"] for rank in ranks],
        "preload_by_rank": [rank["preload_identity"] for rank in ranks],
        "omp_num_threads_by_rank": [rank["omp_num_threads"] for rank in ranks],
        "mkl_num_threads_by_rank": [rank["mkl_num_threads"] for rank in ranks],
        "corpus_hash": ranks[0]["corpus_hash"],
        "weights_file_sha256": ranks[0]["weights_file_sha256"],
        "weights_tensor_sha256": ranks[0]["weights_tensor_sha256"],
        "placement_pages": {
            str(rank["node"]): rank["placement_pages"] for rank in ranks
        },
        "extension": ranks[0]["extension"],
        "communicator": ranks[0]["communicator"],
        "ranks": ranks,
        "correctness": {
            "rank_outputs": [rank["correctness"] for rank in ranks],
            "passed": all_ranks_passed,
            "all_ranks_passed": all_ranks_passed,
        },
    }


def worker_main(args: Any) -> None:
    """Run one native CPU TP2 replay control using the worker JSON-lines protocol."""
    replay._validate_source_context()
    control = args.control_worker
    selected_nodes, selected_cpus = _pair_topology(
        control, args.cpus, args.nodes, args.threads_per_node
    )
    if args.mode == "replay" and (not args.corpus or not args.manifest):
        raise ValueError("Native TP2 replay requires a corpus and frozen manifest")
    if not getattr(args, "expected_corpus_hash", None):
        raise ValueError("Native TP2 worker requires --expected-corpus-hash")
    if args.block_n != 128 or args.block_k != 128:
        raise ValueError("Native TP2 control requires 128 by 128 FP8 scale blocks")

    config = {
        "control": control,
        "nodes": selected_nodes,
        "cpus_by_rank": [list(team) for team in selected_cpus],
        "threads": args.threads_per_node,
        "weights": args.weights,
        "corpus": args.corpus,
        "manifest": args.manifest,
        "mode": args.mode,
        "layer_id": args.layer_id,
        "seed": args.seed,
        "synthetic_token_counts": args.synthetic_token_counts,
        "expected_corpus_hash": args.expected_corpus_hash,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "atol": args.atol,
        "rtol": args.rtol,
        "reference_chunk_tokens": args.reference_chunk_tokens,
        "reference_cache_dir": args.reference_cache_dir,
        "diagnostic_repeats": args.diagnostic_repeats,
        "measure_invalid_controls": bool(args.measure_invalid_controls),
        "init_method": "",
        "dist_ident": str(uuid.uuid4()),
    }
    previous_env = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "OMP_DYNAMIC",
        )
    }
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_node)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_node)
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)
    os.environ["OMP_DYNAMIC"] = "FALSE"

    processes: dict[int, multiprocessing.Process] = {}
    connections: dict[int, Connection] = {}
    with tempfile.TemporaryDirectory(prefix="vllm-numa-moe-tp2-") as temp_dir:
        config["init_method"] = (Path(temp_dir) / "gloo-store").as_uri()
        try:
            processes, connections, rank_ready = _start_rank_processes(config)
            replay._worker_send(
                _ready_payload(control, selected_nodes, selected_cpus, rank_ready)
            )
            for line in sys.stdin:
                request = json.loads(line)
                command = request.get("command")
                if command == "stop":
                    _stop_rank_processes(connections, processes, graceful=True)
                    processes, connections = {}, {}
                    replay._worker_send({"event": "stopped", "control": control})
                    return
                if command == "diagnostic":
                    rank_responses = _exchange_rank_command(
                        command,
                        connections,
                        processes,
                    )
                    replay._worker_send(
                        _combine_rank_diagnostics(rank_responses, control)
                    )
                    continue
                if command not in ("warm", "run"):
                    raise ValueError(f"Unknown native TP2 worker command: {command!r}")
                rank_responses = _exchange_rank_command(command, connections, processes)
                replay._worker_send(
                    _combine_rank_timings(rank_responses, command, control)
                )
            _stop_rank_processes(connections, processes, graceful=False)
            processes, connections = {}, {}
        except Exception as exc:
            replay._worker_send(
                {
                    "event": "error",
                    "control": control,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        finally:
            if processes:
                _stop_rank_processes(connections, processes, graceful=False)
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit("Import worker_main through benchmark_cpu_numa_moe.py")

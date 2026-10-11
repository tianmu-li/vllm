# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared isolated cache-state diagnostics for CPU MoE replay workers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def run_diagnostics(
    records: list[dict[str, Any]],
    execute: Callable[[dict[str, Any]], Any],
    evict: Callable[[dict[str, Any]], None],
    repeats: int,
) -> dict[str, Any]:
    """Measure representative MoE calls with warm and evicted weight caches.

    The first corpus record for each phase and concurrency is used. Each
    representative runs once before timing to warm its layer, followed by the
    requested warm-layer measurements and weight-cache-cold measurements.
    Weight eviction is outside each cold measurement.
    """
    if repeats <= 0:
        raise ValueError("Diagnostic repeats must be positive")

    representatives: dict[tuple[str, Any], tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        key = (record["phase"], record.get("concurrency"))
        representatives.setdefault(key, (index, record))

    samples = []
    for index, record in representatives.values():
        execute(record)

        warm_layer_ms = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            execute(record)
            warm_layer_ms.append((time.perf_counter_ns() - start) / 1e6)

        cold_weights_ms = []
        for _ in range(repeats):
            evict(record)
            start = time.perf_counter_ns()
            execute(record)
            cold_weights_ms.append((time.perf_counter_ns() - start) / 1e6)

        samples.append(
            {
                "representative": {
                    "record_index": index,
                    "phase": record["phase"],
                    "concurrency": record.get("concurrency"),
                    "layer": record["layer"],
                    "step": record.get("step"),
                    "routing": record.get("routing"),
                    "token_rows": int(record["hidden"].shape[0]),
                },
                "warm_layer_ms": warm_layer_ms,
                "cold_weights_ms": cold_weights_ms,
            }
        )

    return {"repeats": repeats, "samples": samples}

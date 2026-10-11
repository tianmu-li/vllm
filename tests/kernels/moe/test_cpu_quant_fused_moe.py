# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU quantized fused MoE kernels."""

import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from benchmarks.kernels.cpu.numa_moe_reference_cache import OracleCache
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._custom_ops as ops  # noqa: E402

if not hasattr(torch.ops._C, "fused_experts_cpu"):
    pytest.skip("fused_experts_cpu op not available", allow_module_level=True)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    """VNNI-prepack expert weights via ``convert_weight_packed``."""
    return torch.ops._C.convert_weight_packed(w)


def _deterministic_expert_routes(
    block_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create top-1 routes with exact per-expert block occupancies."""
    topk_ids = torch.cat(
        [
            torch.full((size,), expert, dtype=torch.int32)
            for expert, size in enumerate(block_sizes)
        ]
    )
    topk_ids = topk_ids.view(-1, 1)
    topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32)
    return topk_weights, topk_ids


# ===========================================================================
# FP8 W8A16 MoE
# ===========================================================================


BLOCK_SIZE = [128, 128]  # [block_n, block_k]

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_SCALE = _FP8_INFO.max  # 448.0
FACTOR_FOR_SCALE = 1e-3


def _block_dequant_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-dequantize FP8 weight [E, N, K] -> float [E, N, K]."""
    E, N, K = weight.shape
    block_n, block_k = block_size
    pad_N = (block_n - N % block_n) % block_n
    pad_K = (block_k - K % block_k) % block_k

    if pad_N > 0 or pad_K > 0:
        weight = F.pad(weight, (0, pad_K, 0, pad_N))

    n_tiles = math.ceil(N / block_n)
    k_tiles = math.ceil(K / block_k)

    weight_block = (
        weight.view(E, n_tiles, block_n, k_tiles, block_k)
        .permute(0, 1, 3, 2, 4)
        .float()
        .contiguous()
    )
    weight_scaled = (
        (weight_block * scales.view(E, n_tiles, k_tiles, 1, 1))
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if pad_N > 0 or pad_K > 0:
        weight_scaled = weight_scaled.view(E, N + pad_N, K + pad_K)
        weight_scaled = weight_scaled[..., :N, :K].contiguous()
    else:
        weight_scaled = weight_scaled.view(E, N, K)
    return weight_scaled


def ref_w8a16_block_fp8_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Reference FP8 W8A16 block-scaled fused MoE in pure torch."""
    B, D = a.shape
    topk = topk_ids.size(1)

    w1_dq = _block_dequant_weight(w1, w1_s, block_size)
    w2_dq = _block_dequant_weight(w2, w2_s, block_size)

    a_exp = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)

    topk_weight_flat = topk_weight.view(-1)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(w1_dq.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum():
            ic0 = torch.matmul(a_exp[mask], w1_dq[i].transpose(0, 1))
            ic1 = _silu_and_mul(ic0)
            out[mask] = torch.matmul(ic1, w2_dq[i].transpose(0, 1))

    return (
        (out.view(B, -1, w2_dq.shape[1]) * topk_weight_flat.view(B, -1, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


def _make_fp8_moe_weights(
    E: int,
    N: int,
    K: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random FP8 MoE weights with random block scales."""
    block_n, block_k = block_size

    w1 = (
        (torch.randn(E, 2 * N, K) * FP8_SCALE)
        .clamp(min=-FP8_SCALE, max=FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        (torch.randn(E, K, N) * FP8_SCALE)
        .clamp(min=-FP8_SCALE, max=FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )

    w1_s = (
        torch.randn(E, math.ceil(2 * N / block_n), math.ceil(K / block_k))
        * FACTOR_FOR_SCALE
    )
    w2_s = (
        torch.randn(E, math.ceil(K / block_n), math.ceil(N / block_k))
        * FACTOR_FOR_SCALE
    )
    return w1, w2, w1_s, w2_s


def _parse_cpu_list(cpu_list: str) -> set[int]:
    cpus: set[int] = set()
    for item in cpu_list.strip().split(","):
        if not item:
            continue
        bounds = item.split("-", maxsplit=1)
        start = int(bounds[0])
        end = int(bounds[1]) if len(bounds) == 2 else start
        cpus.update(range(start, end + 1))
    return cpus


def _allowed_memory_nodes() -> set[int]:
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("Mems_allowed_list:"):
            return _parse_cpu_list(line.split(":", maxsplit=1)[1])
    return set()


def _discover_cpu_numa_teams() -> tuple[list[int], list[int], int] | None:
    """Return three allowed NUMA nodes and their non-SMT CPU teams."""
    if hasattr(os, "sched_getaffinity"):
        allowed_cpus = set(os.sched_getaffinity(0))
    else:
        allowed_cpus = set(range(os.cpu_count() or 0))
    allowed_nodes = _allowed_memory_nodes()
    teams: list[tuple[int, list[int]]] = []

    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        try:
            node = int(node_path.name.removeprefix("node"))
            if allowed_nodes and node not in allowed_nodes:
                continue
            node_cpus = _parse_cpu_list((node_path / "cpulist").read_text())
        except (OSError, ValueError):
            continue

        physical_cpus: dict[tuple[int, ...], int] = {}
        for cpu in sorted(node_cpus & allowed_cpus):
            sibling_path = (
                Path("/sys/devices/system/cpu")
                / f"cpu{cpu}"
                / "topology/thread_siblings_list"
            )
            try:
                siblings = tuple(
                    sorted(
                        _parse_cpu_list(sibling_path.read_text())
                        & node_cpus
                        & allowed_cpus
                    )
                )
            except OSError:
                continue
            if siblings:
                physical_cpus[siblings] = min(siblings)

        cpus = sorted(physical_cpus.values())
        if cpus and (node_path / "meminfo").exists():
            teams.append((node, cpus))
        if len(teams) == 3:
            break

    if len(teams) != 3:
        return None

    threads_per_node = min(4, *(len(cpus) for _, cpus in teams))
    nodes = [node for node, _ in teams]
    cpus = [cpu for _, node_cpus in teams for cpu in node_cpus[:threads_per_node]]
    return cpus, nodes, threads_per_node


def _fused_fp8_moe_output(
    a: torch.Tensor,
    packed_w1: torch.Tensor,
    packed_w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(a)
    ops.fused_experts_cpu(
        output,
        a,
        packed_w1,
        packed_w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    return output


def _assert_numa_moe_stats(
    stats: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    num_experts: int,
    num_nodes: int,
) -> torch.Tensor:
    logical_counts = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64), minlength=num_experts
    )
    assert torch.equal(stats["logical_counts"], logical_counts)
    physical_counts = stats["physical_counts"]
    assert physical_counts.shape == (num_nodes, num_experts)
    assert torch.equal(physical_counts.sum(dim=0), logical_counts)
    assert stats["split_groups"].numel() == 1
    return physical_counts


FP8_NUM_TOKENS = [1, 2, 64, 121]
FP8_MOE_CONFIGS = [
    (256, 512, 8, 2),
    (256, 512, 8, 4),
    (512, 256, 8, 2),
    (512, 256, 8, 4),
    (512, 512, 8, 2),
    (512, 512, 8, 4),
    (768, 2048, 8, 2),
    (768, 2048, 8, 4),
    (768, 2048, 128, 8),
]


@pytest.mark.parametrize("M", FP8_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", FP8_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
def test_w8a16_block_fp8_cpu_fused_moe(M, N, K, E, topk, seed):
    """Test fused_experts_cpu FP8 W8A16 against dequantised torch reference."""
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_w8a16_block_fp8_moe(
        a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, BLOCK_SIZE
    )

    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)

    # Test writing into a fresh output buffer against reference
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)

    # Test writing into the input tensor itself (aliased in-place) produces
    # an identical result
    a_inplace = a.clone()
    ops.fused_experts_cpu(
        a_inplace,
        a_inplace,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(a_inplace, out, atol=0, rtol=0)


def test_w8a16_block_fp8_cpu_fused_moe_small_expert_blocks():
    """Test FP8 BRGEMM at exact and multi-block expert boundaries."""
    set_random_seed(0)
    block_sizes = (4, 5, 33)
    N, K, E = 128, 128, len(block_sizes)
    M = sum(block_sizes)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
    topk_weight, topk_ids = _deterministic_expert_routes(block_sizes)

    ref_out = ref_w8a16_block_fp8_moe(
        a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, BLOCK_SIZE
    )
    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("intermediate_size,hidden_size", [(128, 128), (512, 2048)])
def test_cpu_numa_fp8_moe_runtime_replays_routes_and_recreates(
    intermediate_size, hidden_size
):
    """Check NUMA replay against the reference, including split and alias paths."""
    op_names = (
        "cpu_numa_moe_create",
        "cpu_numa_moe_execute",
        "cpu_numa_moe_stats",
        "cpu_numa_moe_destroy",
        "cpu_numa_moe_calibrate",
    )
    if any(not hasattr(torch.ops._C, name) for name in op_names):
        pytest.skip("experimental CPU NUMA MoE operators are not available")
    topology = _discover_cpu_numa_teams()
    if topology is None:
        pytest.skip("three allowed NUMA nodes with physical CPUs are required")
        return
    cpus, nodes, threads_per_node = topology

    set_random_seed(0)
    num_experts = 8
    w1, w2, w1_s, w2_s = _make_fp8_moe_weights(
        num_experts, intermediate_size, hidden_size, BLOCK_SIZE
    )
    # Keep the random down projection variance constant across expert widths.
    w2_s *= math.sqrt(128 / intermediate_size)
    packed_w1, packed_w2 = _prepack_experts(w1), _prepack_experts(w2)
    runtime = torch.ops._C
    handle = runtime.cpu_numa_moe_create(
        packed_w1,
        packed_w2,
        w1_s,
        w2_s,
        cpus,
        nodes,
        threads_per_node,
        BLOCK_SIZE[0],
        BLOCK_SIZE[1],
    )
    try:
        multi_slot_hidden = torch.randn(
            65, hidden_size, dtype=torch.bfloat16
        ) / math.sqrt(hidden_size)
        rows = torch.arange(65)
        multi_slot_ids = torch.cat(
            (
                torch.zeros(65, 1, dtype=torch.int64),
                1 + (rows[:, None] + torch.arange(7)) % (num_experts - 1),
            ),
            dim=1,
        ).to(torch.int32)
        multi_slot_weight = torch.softmax(
            torch.randn(65, 8, dtype=torch.float32), dim=-1
        )

        hot_hidden = torch.randn(512, hidden_size, dtype=torch.bfloat16) / math.sqrt(
            hidden_size
        )
        hot_ids = torch.zeros(512, 1, dtype=torch.int32)
        hot_weight = torch.ones(512, 1, dtype=torch.float32)

        changed_hidden = torch.randn(3, hidden_size, dtype=torch.bfloat16) / math.sqrt(
            hidden_size
        )
        changed_ids = torch.tensor([[6, 7], [2, 5], [3, 4]], dtype=torch.int32)
        changed_weight = torch.softmax(torch.randn(3, 2, dtype=torch.float32), dim=-1)

        output = torch.empty_like(changed_hidden)
        runtime.cpu_numa_moe_execute(
            handle,
            changed_hidden,
            changed_ids,
            changed_weight,
            output,
            "static",
            0,
        )
        stats = runtime.cpu_numa_moe_stats(handle)
        assert stats["calibrated_rows"].numel() == 0

        calibration_rows = [1, 4, 8, 32, 64, 128, 512]
        runtime.cpu_numa_moe_calibrate(handle, calibration_rows)
        assert runtime.cpu_numa_moe_stats(handle)["calibrated_rows"].numel() == 0

        runtime.cpu_numa_moe_execute(
            handle,
            changed_hidden,
            changed_ids,
            changed_weight,
            output,
            "static",
            0,
        )
        stats = runtime.cpu_numa_moe_stats(handle)
        expected_rows = torch.tensor(calibration_rows, dtype=torch.int64)
        assert torch.equal(stats["calibrated_rows"], expected_rows)
        assert stats["cost_ms"].shape == (len(nodes), len(calibration_rows))

        expected_costs = stats["cost_ms"].clone()
        expected_worker_cpus = stats["worker_cpus"].clone()
        stats["cost_ms"].fill_(-1)
        stats["worker_cpus"].fill_(-1)
        refreshed_stats = runtime.cpu_numa_moe_stats(handle)
        assert torch.equal(refreshed_stats["cost_ms"], expected_costs)
        assert torch.equal(refreshed_stats["worker_cpus"], expected_worker_cpus)
        invalid_ids = changed_ids.clone()
        invalid_ids.view(-1)[-1] = num_experts
        with pytest.raises(RuntimeError, match="Invalid logical expert"):
            runtime.cpu_numa_moe_execute(
                handle,
                changed_hidden,
                invalid_ids,
                changed_weight,
                output,
                "static",
                0,
            )
        failed_call_stats = runtime.cpu_numa_moe_stats(handle)
        assert failed_call_stats.keys() == refreshed_stats.keys()
        for key in refreshed_stats:
            assert torch.equal(failed_call_stats[key], refreshed_stats[key])

        cases = (
            (multi_slot_hidden, multi_slot_ids, multi_slot_weight),
            (hot_hidden, hot_ids, hot_weight),
            (changed_hidden, changed_ids, changed_weight),
            (
                hot_hidden[:5],
                torch.arange(num_experts, dtype=torch.int32).repeat(5, 1),
                torch.full((5, num_experts), 1 / num_experts),
            ),
            # Multi-block expert groups with uneven worker counts and row tails.
            (
                hot_hidden[:129],
                torch.arange(num_experts, dtype=torch.int32).repeat(129, 1),
                torch.full((129, num_experts), 1 / num_experts),
            ),
            # Equal counts alone do not permit reusing another group's inputs.
            (
                hot_hidden[:12],
                torch.tensor([[0, 1]] * 6 + [[3, 4]] * 6, dtype=torch.int32),
                torch.full((12, 2), 0.5),
            ),
        )
        references = []
        for hidden, ids, weights in cases:
            reference = ref_w8a16_block_fp8_moe(
                hidden, w1, w2, w1_s, w2_s, weights, ids, BLOCK_SIZE
            )
            baseline = _fused_fp8_moe_output(
                hidden, packed_w1, packed_w2, weights, ids, w1_s, w2_s
            )
            references.append((reference, baseline))

        for layer_id, policy in enumerate(("static", "balanced", "split", "rows")):
            for index, (hidden, ids, weights) in enumerate(cases):
                output = torch.empty_like(hidden)
                runtime.cpu_numa_moe_execute(
                    handle, hidden, ids, weights, output, policy, layer_id
                )
                reference, baseline = references[index]
                # Row partitioning can move a small tail onto BRGEMM.
                tolerance = 1e-2 if policy == "rows" else 0
                torch.testing.assert_close(
                    output,
                    baseline,
                    atol=tolerance,
                    rtol=tolerance,
                    msg=lambda message, policy=policy, index=index: (
                        f"{policy}, case {index}: {message}"
                    ),
                )
                torch.testing.assert_close(reference, output, atol=1e-2, rtol=1e-2)

                stats = runtime.cpu_numa_moe_stats(handle)
                physical_counts = _assert_numa_moe_stats(
                    stats, ids, num_experts, len(nodes)
                )
                if policy == "split" and index == 1:
                    assert stats["split_groups"].item() > 0
                    assert (physical_counts[:, 0] > 0).sum() >= 2
                if policy in ("static", "balanced") and index == 1:
                    assert (physical_counts[:, 0] > 0).sum() == 1
                if policy == "rows" and index == 1:
                    assert torch.all(physical_counts[:, 0] > 0)
                if policy == "static" and index in (4, 5):
                    input_rows = 129 * len(nodes) if index == 4 else ids.numel()
                    assert stats["dispatch_bytes"].item() == (
                        input_rows * hidden_size * 2 + ids.numel() * 4
                    )

            for index in (2, 3, 4, 5, 1):
                hidden, ids, weights = cases[index]
                aliased = hidden.clone()
                runtime.cpu_numa_moe_execute(
                    handle, aliased, ids, weights, aliased, policy, layer_id
                )
                tolerance = 1e-2 if policy == "rows" and index != 2 else 0
                torch.testing.assert_close(
                    aliased, references[index][1], atol=tolerance, rtol=tolerance
                )

        # Replay changed routes after idle intervals and caller-affinity changes.
        original_affinity = os.sched_getaffinity(0)
        try:
            for affinity, quiesce, index in (
                ({cpus[0]}, True, 3),
                (original_affinity, False, 2),
                ({cpus[0]}, True, 2),
                ({cpus[0]}, False, 3),
            ):
                os.sched_setaffinity(0, affinity)
                if quiesce:
                    time.sleep(0.02)
                hidden, ids, weights = cases[index]
                aliased = hidden.clone()
                runtime.cpu_numa_moe_execute(
                    handle, aliased, ids, weights, aliased, "static", 0
                )
                torch.testing.assert_close(
                    aliased, references[index][1], atol=0, rtol=0
                )
                stats = runtime.cpu_numa_moe_stats(handle)
                _assert_numa_moe_stats(stats, ids, num_experts, len(nodes))
        finally:
            os.sched_setaffinity(0, original_affinity)

        workers_before_destroy = len(list(Path("/proc/self/task").iterdir()))
        second_w2_s = w2_s * 0.5
        second_layer = runtime.cpu_numa_moe_create(
            packed_w1,
            packed_w2,
            w1_s,
            second_w2_s,
            cpus,
            nodes,
            threads_per_node,
            *BLOCK_SIZE,
        )
        try:
            assert (
                len(list(Path("/proc/self/task").iterdir())) == workers_before_destroy
            )
            # Shared scratch must not reuse another layer's unpacked weights.
            larger_hidden = hot_hidden.repeat(2, 1)
            larger_output = torch.empty_like(larger_hidden)
            second_reference = _fused_fp8_moe_output(
                larger_hidden,
                packed_w1,
                packed_w2,
                hot_weight.repeat(2, 1),
                hot_ids.repeat(2, 1),
                w1_s,
                second_w2_s,
            )
            runtime.cpu_numa_moe_execute(
                second_layer,
                larger_hidden,
                hot_ids.repeat(2, 1),
                hot_weight.repeat(2, 1),
                larger_output,
                "static",
                1,
            )
            torch.testing.assert_close(larger_output, second_reference, atol=0, rtol=0)
            restored_hot_output = torch.empty_like(hot_hidden)
            runtime.cpu_numa_moe_execute(
                handle,
                hot_hidden,
                hot_ids,
                hot_weight,
                restored_hot_output,
                "static",
                0,
            )
            torch.testing.assert_close(
                restored_hot_output, references[1][1], atol=0, rtol=0
            )
            output = torch.empty_like(changed_hidden)
            runtime.cpu_numa_moe_execute(
                handle,
                changed_hidden,
                changed_ids,
                changed_weight,
                output,
                "static",
                0,
            )
            torch.testing.assert_close(output, references[2][1], atol=0, rtol=0)
            second_small_reference = _fused_fp8_moe_output(
                changed_hidden,
                packed_w1,
                packed_w2,
                changed_weight,
                changed_ids,
                w1_s,
                second_w2_s,
            )
            runtime.cpu_numa_moe_execute(
                second_layer,
                changed_hidden,
                changed_ids,
                changed_weight,
                output,
                "static",
                1,
            )
            torch.testing.assert_close(output, second_small_reference, atol=0, rtol=0)
            assert torch.equal(
                runtime.cpu_numa_moe_stats(handle)["scratch_addresses"],
                runtime.cpu_numa_moe_stats(second_layer)["scratch_addresses"],
            )
        finally:
            runtime.cpu_numa_moe_destroy(second_layer)
        assert len(list(Path("/proc/self/task").iterdir())) == workers_before_destroy
        runtime.cpu_numa_moe_destroy(handle)
        assert len(list(Path("/proc/self/task").iterdir())) == (
            workers_before_destroy - len(cpus)
        )
        with pytest.raises(RuntimeError, match="Invalid NUMA MoE runtime handle"):
            runtime.cpu_numa_moe_stats(handle)
        handle = None
        handle = runtime.cpu_numa_moe_create(
            packed_w1,
            packed_w2,
            w1_s,
            w2_s,
            cpus,
            nodes,
            threads_per_node,
            BLOCK_SIZE[0],
            BLOCK_SIZE[1],
        )
        output = torch.empty_like(changed_hidden)
        original_affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {cpus[0]})
            time.sleep(0.02)
            runtime.cpu_numa_moe_execute(
                handle,
                changed_hidden,
                changed_ids,
                changed_weight,
                output,
                "static",
                0,
            )
            runtime.cpu_numa_moe_destroy(handle)
            handle = None
        finally:
            os.sched_setaffinity(0, original_affinity)
        torch.testing.assert_close(output, references[2][1], atol=0, rtol=0)
    finally:
        if handle is not None:
            runtime.cpu_numa_moe_destroy(handle)


@pytest.mark.parametrize("K", [128, 2048])
def test_cpu_numa_fp8_intermediate_slices_preserve_checkpoint_scales(K):
    """Match native slice arithmetic and rotate the wider unequal slice."""
    if not hasattr(torch.ops._C, "cpu_numa_moe_create"):
        pytest.skip("experimental CPU NUMA MoE operators are not available")
    topology = _discover_cpu_numa_teams()
    if topology is None:
        pytest.skip("three allowed NUMA nodes with physical CPUs are required")
        return
    cpus, nodes, threads = topology
    set_random_seed(1)
    E, N = 2, 512
    w1, w2, s1, s2 = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
    hidden = torch.randn(17, K, dtype=torch.bfloat16) / math.sqrt(K)
    ids = torch.tensor([[0, 1]], dtype=torch.int32).expand(17, -1).contiguous()
    weights = torch.softmax(torch.randn(17, 2), dim=-1)
    handle = torch.ops._C.cpu_numa_moe_create(
        _prepack_experts(w1),
        _prepack_experts(w2),
        s1,
        s2,
        cpus,
        nodes,
        threads,
        *BLOCK_SIZE,
    )
    try:
        for policy, layer, widths in (
            ("unequal", 0, (256, 128, 128)),
            ("unequal", 1, (128, 256, 128)),
            ("unequal", 2, (128, 128, 256)),
            ("tp2_01", 0, (256, 256, 0)),
            ("tp2_02", 0, (256, 0, 256)),
            ("tp2_12", 0, (0, 256, 256)),
        ):
            begin = 0
            partials = []
            for width in widths:
                if not width:
                    continue
                end = begin + width
                sliced_w1 = torch.cat(
                    (w1[:, begin:end], w1[:, N + begin : N + end]), dim=1
                )
                sliced_s1 = torch.cat(
                    (
                        s1[:, begin // 128 : end // 128],
                        s1[:, (N + begin) // 128 : (N + end) // 128],
                    ),
                    dim=1,
                )
                partials.append(
                    _fused_fp8_moe_output(
                        hidden,
                        _prepack_experts(sliced_w1),
                        _prepack_experts(w2[:, :, begin:end].contiguous()),
                        weights,
                        ids,
                        sliced_s1,
                        s2[:, :, begin // 128 : end // 128].contiguous(),
                    )
                )
                begin = end
            expected = torch.stack(partials).float().sum(dim=0).to(torch.bfloat16)
            output = hidden.clone()
            torch.ops._C.cpu_numa_moe_execute(
                handle, output, ids, weights, output, policy, layer
            )
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            stats = torch.ops._C.cpu_numa_moe_stats(handle)
            assert stats["intermediate_slices"].tolist() == list(widths)
    finally:
        torch.ops._C.cpu_numa_moe_destroy(handle)

    w1, w2, s1, s2 = _make_fp8_moe_weights(E, N, 160, BLOCK_SIZE)
    handle = torch.ops._C.cpu_numa_moe_create(
        _prepack_experts(w1),
        _prepack_experts(w2),
        s1,
        s2,
        cpus,
        nodes,
        threads,
        *BLOCK_SIZE,
    )
    try:
        hidden = torch.zeros(1, 160, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="aligned"):
            torch.ops._C.cpu_numa_moe_execute(
                handle, hidden, ids[:1], weights[:1], hidden, "unequal", 0
            )
    finally:
        torch.ops._C.cpu_numa_moe_destroy(handle)


# ===========================================================================
# MXFP4 W4A16 MoE
# ===========================================================================


class MXFP4QuantizeUtil:
    """MXFP4 quantization utility."""

    E2M1_max = 6.0
    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    block_size = 32

    @classmethod
    def quantize(cls, input: torch.Tensor) -> tuple:
        """Quantize BF16 tensor to MXFP4 packed uint8 format."""

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            left_side = x[..., 0::2]
            right_side = x[..., 1::2]
            new_data = right_side.clone() << 4
            new_data[..., : left_side.shape[-1]] += left_side
            return new_data

        original_shape = input.shape
        input = input.view(-1, cls.block_size)
        input_amax = input.abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

        input = (input / torch.exp2(e8m0_scale)).view(original_shape)
        input_q = cast_fp4(input)
        input_q = fuse_uint4_to_uint8(input_q)
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
        return input_q, e8m0_scale

    @classmethod
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale):
        """Dequantize MXFP4 packed tensor back to float."""

        def unfuse_uint8_to_uint4(x):
            left_side = x & 0x0F
            right_side = (x >> 4) & 0x0F
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)
            result[..., 0::2] = left_side
            result[..., 1::2] = right_side
            return result

        e8m0_scale = scale
        x_unfused = unfuse_uint8_to_uint4(quantized_data)
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(torch.float32)
        magnitude = (x_unfused & 0b0111).to(torch.long)
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device)
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)
        x_float = sign.float() * x_float
        x_float = x_float.reshape(-1, cls.block_size)
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)
        x_float = x_float * scale_factor
        return x_float.reshape(original_shape).to(dtype)


def _swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """SwigLU activation used in GPT-OSS.
    Input is interleaved: [gate_0, up_0, gate_1, up_1, ...] in last dim.
    """
    gate = x[..., 0::2]
    up = x[..., 1::2]
    gate_clamped = torch.clamp(gate, max=limit)
    up_clamped = torch.clamp(up, min=-limit, max=limit)
    return gate_clamped * torch.sigmoid(alpha * gate_clamped) * (up_clamped + 1)


def ref_mxfp4_fused_moe(
    a: torch.Tensor,
    w1_dq: torch.Tensor,
    w2_dq: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Reference MXFP4 fused MoE with SiLU activation."""
    B, D = a.shape
    a_f = a.float()
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(w1_dq.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum() == 0:
            continue
        token_indices = torch.where(mask)[0]
        source_indices = token_indices // topk
        ic0 = torch.matmul(a_f[source_indices], w1_dq[i].float().T)
        ic1 = _silu_and_mul(ic0)
        out[mask] = torch.matmul(ic1, w2_dq[i].float().T)

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def ref_mxfp4_fused_moe_gptoss(
    a: torch.Tensor,
    w1_dq: torch.Tensor,
    w2_dq: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """Reference MXFP4 fused MoE with SwigLU+bias (GPT-OSS style)."""
    B, D = a.shape
    topk = topk_ids.shape[1]
    a_f = a.float()
    E = w1_dq.shape[0]
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(E):
        mask = topk_ids_flat == i
        if mask.sum() == 0:
            continue
        token_indices = torch.where(mask)[0]
        source_indices = token_indices // topk
        ic0 = torch.matmul(a_f[source_indices], w1_dq[i].float().T)
        ic0 = ic0 + w1_bias[i].float()
        ic1 = _swiglu(ic0, alpha, limit)
        ic2 = torch.matmul(ic1, w2_dq[i].float().T)
        ic2 = ic2 + w2_bias[i].float()
        out[mask] = ic2

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def _prepack_mxfp4_experts(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack MXFP4 weights and repack scales."""
    packed_w = torch.ops._C.convert_weight_packed(w)
    packed_s = torch.ops._C.convert_scale_packed(w_scale)
    return packed_w, packed_s


MXFP4_NUM_TOKENS = [1, 2, 32, 121]
MXFP4_MOE_CONFIGS = [
    (128, 128, 4, 2),
    (256, 256, 8, 4),
    (352, 256, 8, 4),
    (512, 320, 8, 4),
]


@pytest.mark.parametrize("M", MXFP4_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", MXFP4_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
def test_mxfp4_cpu_fused_moe(M, N, K, E, topk, seed):
    """Test fused_experts_mxfp4_cpu against dequantized torch reference."""
    set_random_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(M, K, dtype=dtype) / 10

    # Generate and quantize weights
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    # Routing
    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    # Reference
    ref_out = ref_mxfp4_fused_moe(a, w1dq, w2dq, topk_weight, topk_ids, topk)

    # Pack weights for kernel
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    # Kernel
    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


def test_mxfp4_cpu_fused_moe_small_expert_blocks():
    """Test MXFP4 BRGEMM at exact and multi-block expert boundaries."""
    set_random_seed(0)
    block_sizes = (4, 5, 33)
    N, K, E = 128, 128, len(block_sizes)
    M = sum(block_sizes)
    dtype = torch.bfloat16

    a = torch.randn(M, K, dtype=dtype) / 10
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    topk_weight, topk_ids = _deterministic_expert_routes(block_sizes)
    ref_out = ref_mxfp4_fused_moe(a, w1dq, w2dq, topk_weight, topk_ids, 1)

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,
        pw2s,
        None,
        None,
        None,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# Both E2M1 zero codes: 0b0000 (+0.0) and 0b1000 (-0.0), in both nibbles.
MXFP4_ZERO_BYTES = [0x00, 0x88, 0x08, 0x80]
# 0 and 255 are the ends of the E8M0 range; 127 is the identity scale.
MXFP4_E8M0_VALUES = [0, 1, 127, 200, 254, 255]


@pytest.mark.parametrize("zero_byte", MXFP4_ZERO_BYTES)
@pytest.mark.parametrize("e8m0", MXFP4_E8M0_VALUES)
def test_mxfp4_cpu_zero_codes_stay_zero(zero_byte, e8m0):
    """A zero E2M1 code stays zero for every E8M0 exponent.

    The unpack applies the block scale as an integer add on the bf16 exponent
    field, which is exact for every value in the E2M1 codebook except the two
    zeros: 0x0000 and 0x8000 have no exponent to shift, so adding to them
    produces a small finite number instead of zero. Both are special-cased, and
    this is the invariant that special case exists for.

    Worth pinning separately from ``test_mxfp4_cpu_fused_moe``: there the zero
    codes are a small fraction of random weights and a broken special case
    would stay inside the 1e-2 tolerance for the low exponents. Here every
    weight is a zero, so the output is exactly zero or it is not.
    """
    N, K, E, M = 64, 64, 2, 4
    dtype = torch.bfloat16
    set_random_seed(0)

    a = torch.randn(M, K, dtype=dtype)
    w1q = torch.full((E, 2 * N, K // 2), zero_byte, dtype=torch.uint8)
    w1s = torch.full((E, 2 * N, K // 32), e8m0, dtype=torch.uint8)
    # w2 is ordinary: the zeros have to survive the first GEMM and the
    # activation, and a nonzero w2 is what would expose it if they did not.
    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)

    topk_weight = torch.ones((M, 1), dtype=torch.float32)
    topk_ids = torch.zeros((M, 1), dtype=torch.int32)

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,
        pw2s,
        None,
        None,
        None,
    )

    # silu(0) * 0 = 0, so the whole layer collapses to exactly zero. Not
    # assert_close: any nonzero output here is a wrong unpack, not rounding.
    assert torch.equal(out, torch.zeros_like(out)), (
        f"zero code 0x{zero_byte:02x} with e8m0={e8m0} produced "
        f"max |out| = {out.abs().max().item()}"
    )


# Narrower than MXFP4_E8M0_VALUES on purpose: this test keeps nonzero weights,
# and 6.0 * 2**(255-127) is not representable in bf16, so the ends of the E8M0
# range would compare inf against inf and prove nothing. The all-zero test above
# is the one that can reach them.
MXFP4_E8M0_FINITE = [107, 127, 137]


@pytest.mark.parametrize("e8m0", MXFP4_E8M0_FINITE)
def test_mxfp4_cpu_zero_codes_mixed_with_nonzero(e8m0):
    """Zeros and nonzeros in the same 32-element scale block.

    The zero check is per lane, not per block: this fails if it is ever
    rewritten as a whole-block branch. Uses a single shared exponent so the
    reference is a plain power of two.
    """
    N, K, E, M = 64, 64, 2, 4
    dtype = torch.bfloat16
    set_random_seed(0)

    a = torch.randn(M, K, dtype=dtype)
    # Alternate a zero byte and a nonzero one along K, so every scale block
    # holds both kinds.
    pattern = torch.tensor([0x88, 0x21], dtype=torch.uint8).repeat(K // 4)
    w1q = pattern.view(1, 1, -1).expand(E, 2 * N, K // 2).contiguous()
    w1s = torch.full((E, 2 * N, K // 32), e8m0, dtype=torch.uint8)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    topk_weight = torch.ones((M, 1), dtype=torch.float32)
    topk_ids = torch.zeros((M, 1), dtype=torch.int32)
    ref_out = ref_mxfp4_fused_moe(a, w1dq, w2dq, topk_weight, topk_ids, 1)

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,
        pw2s,
        None,
        None,
        None,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [1, 32])
@pytest.mark.parametrize("N,K,E,topk", [(128, 128, 4, 2), (64, 64, 4, 2)])
@pytest.mark.parametrize("seed", [0])
def test_mxfp4_cpu_fused_moe_bias_swiglu(M, N, K, E, topk, seed):
    """Test fused_experts_mxfp4_cpu with bias and SwigLU activation (GPT-OSS)."""
    set_random_seed(seed)
    dtype = torch.bfloat16
    alpha = 1.702
    limit = 7.0

    a = torch.randn(M, K, dtype=dtype) / 10

    # Generate and quantize weights
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)
    w1_b = torch.randn(E, 2 * N, dtype=torch.float32) / 10

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)
    w2_b = torch.randn(E, K, dtype=torch.float32) / 10

    # Routing
    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    # Reference
    ref_out = ref_mxfp4_fused_moe_gptoss(
        a, w1dq, w2dq, w1_b, w2_b, topk_weight, topk_ids, alpha, limit
    )

    # Pack weights for kernel
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    # Kernel
    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        w1_bias=w1_b,
        w2_bias=w2_b,
        alpha=alpha,
        limit=limit,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# INT4 W4A16 MoE
# ===========================================================================


def _pack_int4_gptq(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values [N, K] → [N, K//8] int32 along K dim (GPTQ format)."""
    N, K = w_int4.shape
    assert K % 8 == 0
    w = w_int4.to(torch.int32)
    w_packed = torch.zeros(N, K // 8, dtype=torch.int32)
    for j in range(8):
        w_packed |= (w[:, j::8] & 0xF) << (j * 4)
    return w_packed


def _pack_int4_awq(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values [..., N] → [..., N//8] int32 along last dim (AWQ format)."""
    # AWQ packing bitshifts: indices {0,4,1,5,2,6,3,7} * 4 bits each
    _AWQ_BITSHIFTS = [0, 16, 4, 20, 8, 24, 12, 28]

    N = w_int4.shape[-1]
    assert N % 8 == 0
    w = w_int4.to(torch.int32)
    w_packed = torch.zeros(*w.shape[:-1], N // 8, dtype=torch.int32)
    for j, shift in enumerate(_AWQ_BITSHIFTS):
        w_packed |= (w[..., j::8] & 0xF) << shift
    return w_packed


def _ref_int4_moe(
    a: torch.Tensor,
    w1_int4: torch.Tensor,
    w2_int4: torch.Tensor,
    w1_zeros: torch.Tensor | None,
    w2_zeros: torch.Tensor | None,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference INT4 W4A16 group-quantized fused MoE in pure torch."""
    B = a.shape[0]
    topk = topk_ids.size(1)
    K_out = a.shape[1]

    out = torch.zeros(B, topk, K_out, dtype=torch.float32)
    for b in range(B):
        for t in range(topk):
            eid = topk_ids[b, t].item()
            x = a[b : b + 1].float()

            # Dequantize w1: [K, 2*N], groups along K (input dim)
            K_dim = w1_int4.shape[1]
            w1_dq = torch.zeros(K_dim, w1_int4.shape[2], dtype=torch.float32)
            for g in range(w1_s.shape[1]):
                k_start = g * group_size
                k_end = min((g + 1) * group_size, K_dim)
                zp = w1_zeros[eid, g, :].float() if w1_zeros is not None else 8.0
                w1_dq[k_start:k_end, :] = (
                    w1_int4[eid, k_start:k_end, :].float() - zp
                ) * w1_s[eid, g, :].float()

            ic = torch.matmul(x, w1_dq)  # [1, K] @ [K, 2*N] → [1, 2*N]
            ic = _silu_and_mul(ic)  # [1, N]

            # Dequantize w2: [N, K], groups along N (input dim)
            N_dim = w2_int4.shape[1]
            w2_dq = torch.zeros(N_dim, w2_int4.shape[2], dtype=torch.float32)
            for g in range(w2_s.shape[1]):
                n_start = g * group_size
                n_end = min((g + 1) * group_size, N_dim)
                zp = w2_zeros[eid, g, :].float() if w2_zeros is not None else 8.0
                w2_dq[n_start:n_end, :] = (
                    w2_int4[eid, n_start:n_end, :].float() - zp
                ) * w2_s[eid, g, :].float()

            oc = torch.matmul(ic, w2_dq)  # [1, N] @ [N, K] → [1, K]
            out[b, t] = oc.squeeze(0)

    return (out * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def _make_int4_moe_weights(E, N, K, group_size, quant_algo):
    """Create INT4 MoE weights in GPTQ or AWQ packed format.

    Canonical layout (input × output):
      w1_int4: [E, K, 2*N]  w2_int4: [E, N, K]

    GPTQ packed (pack transposed weight along input/K dim):
      w1_packed: [E, K//8, 2*N]  w2_packed: [E, N//8, K]
      zeros: actual int4 zero points, same packing as weights

    AWQ packed (pack along output/N dim):
      w1_packed: [E, K, 2*N//8]  w2_packed: [E, N, K//8]
      zeros: actual int4 zero points, same packing as weights

    Returns:
        w1_int4, w2_int4,
        w1_packed, w2_packed,
        w1_zeros, w2_zeros,
        w1_zeros_packed, w2_zeros_packed,
        w1_s, w2_s

    """
    w1_int4 = torch.randint(0, 16, (E, K, 2 * N), dtype=torch.int32)
    w2_int4 = torch.randint(0, 16, (E, N, K), dtype=torch.int32)

    num_groups_w1 = K // group_size
    num_groups_w2 = N // group_size
    w1_s = (
        torch.randn(E, num_groups_w1, 2 * N, dtype=torch.bfloat16) * 0.01
    ).abs() + 0.001
    w2_s = (torch.randn(E, num_groups_w2, K, dtype=torch.bfloat16) * 0.01).abs() + 0.001

    if quant_algo == ops.CPUQuantAlgo.GPTQ:
        # Pack: canonical [E, K, 2*N] → transpose [E, 2*N, K] → GPTQ pack
        # [E, 2*N, K//8] → transpose [E, K//8, 2*N]
        w1_t = w1_int4.transpose(1, 2).contiguous()  # [E, 2*N, K]
        w1_packed = (
            torch.stack([_pack_int4_gptq(w1_t[e]) for e in range(E)])
            .transpose(1, 2)
            .contiguous()
        )  # [E, K//8, 2*N]
        w2_t = w2_int4.transpose(1, 2).contiguous()  # [E, K, N]
        w2_packed = (
            torch.stack([_pack_int4_gptq(w2_t[e]) for e in range(E)])
            .transpose(1, 2)
            .contiguous()
        )  # [E, N//8, K]
        w1_zeros = w2_zeros = None
        w1_zeros_packed = torch.full(
            (E, num_groups_w1, 2 * N // 8), 0x77777777, dtype=torch.int32
        )
        w2_zeros_packed = torch.full(
            (E, num_groups_w2, K // 8), 0x77777777, dtype=torch.int32
        )
    else:  # AWQ
        # Asymmetric: actual zero points, packed along output dim.
        w1_zeros = torch.randint(1, 15, (E, num_groups_w1, 2 * N), dtype=torch.int32)
        w2_zeros = torch.randint(1, 15, (E, num_groups_w2, K), dtype=torch.int32)
        w1_packed = torch.stack(
            [_pack_int4_awq(w1_int4[e]) for e in range(E)]
        )  # [E, K, 2*N//8]
        w2_packed = torch.stack(
            [_pack_int4_awq(w2_int4[e]) for e in range(E)]
        )  # [E, N, K//8]
        w1_zeros_packed = torch.stack(
            [_pack_int4_awq(w1_zeros[e]) for e in range(E)]
        )  # [E, K//gs, 2*N//8]
        w2_zeros_packed = torch.stack(
            [_pack_int4_awq(w2_zeros[e]) for e in range(E)]
        )  # [E, N//gs, K//8]

    return (
        w1_int4,
        w2_int4,
        w1_packed,
        w2_packed,
        w1_zeros,
        w2_zeros,
        w1_zeros_packed,
        w2_zeros_packed,
        w1_s,
        w2_s,
    )


INT4_MOE_CONFIGS = [
    # (N, K, E, topk, group_size)
    (256, 512, 8, 2, 128),
    (512, 256, 8, 2, 128),
    (512, 512, 8, 4, 128),
    (768, 2048, 8, 2, 128),
]


@pytest.mark.parametrize("M", [1, 2, 64, 121])
@pytest.mark.parametrize("N,K,E,topk,group_size", INT4_MOE_CONFIGS)
@pytest.mark.parametrize("quant_algo", [ops.CPUQuantAlgo.GPTQ, ops.CPUQuantAlgo.AWQ])
@pytest.mark.parametrize("seed", [0])
def test_int4_w4a16_cpu_fused_moe(M, N, K, E, topk, group_size, quant_algo, seed):
    """Test fused_experts_cpu INT4 W4A16 for both GPTQ and AWQ quant formats."""
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    (
        w1_int4,
        w2_int4,
        w1_packed,
        w2_packed,
        w1_zeros,
        w2_zeros,
        w1_zeros_packed,
        w2_zeros_packed,
        w1_s,
        w2_s,
    ) = _make_int4_moe_weights(E, N, K, group_size, quant_algo)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int4_moe(
        a,
        w1_int4,
        w2_int4,
        w1_zeros,
        w2_zeros,
        w1_s,
        w2_s,
        topk_weight,
        topk_ids,
        group_size,
    )

    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        prepare_int4_moe_layer_for_cpu,
    )

    (blocked_w1, blocked_w2, blocked_s1, blocked_s2, blocked_z1, blocked_z2) = (
        prepare_int4_moe_layer_for_cpu(
            w1_packed,
            w2_packed,
            w1_s,
            w2_s,
            quant_algo=quant_algo,
            w13_zeros=w1_zeros_packed,
            w2_zeros=w2_zeros_packed,
        )
    )

    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        blocked_w1,
        blocked_w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT4_W4A8,
        blocked_s1,
        blocked_s2,
        blocked_z1,
        blocked_z2,
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        True,  # is_vnni
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# INT8 W8A8 MoE
# ===========================================================================


def _quantize_per_channel(w):
    """Symmetric per-channel INT8 quantisation. w: [N, K] -> (int8, scale)."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale.float()


def _quantize_per_token(x):
    """Symmetric per-token INT8 quantisation. x: [M, K] -> (int8, scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale.float()


def _ref_int8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids):
    """Reference INT8 W8A8 per-channel fused MoE in pure torch."""
    B, D = a.shape
    topk = topk_ids.size(1)

    out = torch.zeros(B, topk, w2.shape[1], dtype=torch.float32)
    for b in range(B):
        for t in range(topk):
            eid = topk_ids[b, t].item()

            x = a[b : b + 1].float()
            x_q, x_s = _quantize_per_token(x)
            ic = torch.matmul(x_q.float(), w1[eid].float().t())
            ic = ic * x_s * w1_s[eid].view(1, -1)
            ic = _silu_and_mul(ic)

            ic_q, ic_s = _quantize_per_token(ic)
            oc = torch.matmul(ic_q.float(), w2[eid].float().t())
            oc = oc * ic_s * w2_s[eid].view(1, -1)
            out[b, t] = oc.squeeze(0)

    result = (out * topk_weight.unsqueeze(-1)).sum(dim=1)
    return result.to(a.dtype)


def _make_int8_moe_weights(E, N, K):
    factor = 1e-2
    w1_f = (torch.randn(E, 2 * N, K) - 0.5) * 2
    w2_f = (torch.randn(E, K, N) - 0.5) * 2

    w1_q_list, w1_s_list = [], []
    w2_q_list, w2_s_list = [], []
    for e in range(E):
        q, s = _quantize_per_channel(w1_f[e])
        w1_q_list.append(q)
        w1_s_list.append(s)
        q, s = _quantize_per_channel(w2_f[e])
        w2_q_list.append(q)
        w2_s_list.append(s)

    return (
        torch.stack(w1_q_list),
        torch.stack(w2_q_list),
        torch.stack(w1_s_list) * factor,
        torch.stack(w2_s_list) * factor,
    )


INT8_NUM_TOKENS = [1, 2, 64, 121]
INT8_MOE_CONFIGS = [
    # (N, K, E, topk)
    (256, 512, 8, 2),
    (512, 256, 8, 2),
    (512, 512, 8, 4),
    (768, 2048, 8, 2),
]


@pytest.mark.parametrize("M", INT8_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", INT8_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("is_vnni", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_int8_w8a8_cpu_fused_moe(M, N, K, E, topk, seed, is_vnni, inplace):
    """Test fused_experts_cpu INT8 W8A8 against torch reference.

    ``inplace`` exercises writing into a fresh output buffer vs. aliasing
    the input tensor as the output buffer.
    """
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    w1_q, w2_q, w1_s, w2_s = _make_int8_moe_weights(E, N, K)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int8_moe(a, w1_q, w2_q, w1_s, w2_s, topk_weight, topk_ids)

    w1 = _prepack_experts(w1_q) if is_vnni else w1_q
    w2 = _prepack_experts(w2_q) if is_vnni else w2_q

    a_in = a.clone()
    out = a_in if inplace else torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT8_W8A8,
        w1_s,
        w2_s,
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        is_vnni,
    )
    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=2e-1,
        rtol=2e-1,
    )


# ===========================================================================
# FP8 W8A8 MoE
# ===========================================================================


requires_cpu_fp8_w8a8 = pytest.mark.skipif(
    not ops.cpu_has_amx_fp8(),
    reason="requires native AMX-FP8 hardware",
)

FP8_W8A8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_W8A8_QUANT_GROUP = 128  # quantization group size for both K and N dimensions


def _make_fp8_w8a8_weight_w13(
    E: int, two_n: int, K: int, group_K: int = FP8_W8A8_QUANT_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create w13 [E, 2N, K] FP8 with group scales [E, 2N, G] where G = K//group_K."""
    G = K // group_K
    w_list, s_list = [], []
    for _ in range(E):
        w_f32 = torch.randn(two_n, K)
        # Per-row, per-K-group quantization: scale [2N, G]
        w_re = w_f32.view(two_n, G, group_K)
        abs_max = w_re.abs().amax(dim=2, keepdim=True).clamp(min=1e-7)  # [2N, G, 1]
        scale = (abs_max / FP8_W8A8_MAX).squeeze(2)  # [2N, G]
        w_q = (
            (w_re / abs_max).clamp(-FP8_W8A8_MAX, FP8_W8A8_MAX).to(torch.float8_e4m3fn)
        )
        w_list.append(w_q.view(two_n, K).contiguous())
        s_list.append(scale.float())
    return torch.stack(w_list), torch.stack(s_list)  # [E, 2N, K], [E, 2N, G]


def _make_fp8_w8a8_weight_w2(
    E: int,
    K: int,
    N: int,
    group_K: int = FP8_W8A8_QUANT_GROUP,
    group_N: int = FP8_W8A8_QUANT_GROUP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create w2 [E, K, N] FP8 with block scales [E, K//gK, N//gN]."""
    nK = K // group_K
    nN = N // group_N
    w_list, s_list = [], []
    for _ in range(E):
        w_f32 = torch.randn(K, N)
        w_re = w_f32.view(nK, group_K, nN, group_N)
        abs_max = w_re.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-7)
        scale = (abs_max / FP8_W8A8_MAX).squeeze(1).squeeze(2)  # [nK, nN]
        w_q = (
            (w_re / abs_max).clamp(-FP8_W8A8_MAX, FP8_W8A8_MAX).to(torch.float8_e4m3fn)
        )
        w_list.append(w_q.view(K, N).contiguous())
        s_list.append(scale.float())
    return torch.stack(w_list), torch.stack(s_list)  # [E, K, N], [E, K//gK, N//gN]


def _pack_fp8_w8a8_w13(
    w13: torch.Tensor,  # [E, 2N, K] FP8
    w13_scale: torch.Tensor,  # [E, 2N, G] float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack w13 for CPU FP8 W8A8 MoE kernel using float8_linear_prepack_cpu."""
    E = w13.size(0)
    packed_list, scale_list = [], []
    for i in range(E):
        pw, ps = torch.ops._C.float8_linear_prepack_cpu(
            w13[i].contiguous(), w13_scale[i].contiguous()
        )
        packed_list.append(pw)
        scale_list.append(ps)
    return torch.stack(packed_list), torch.stack(scale_list)


def _ref_fp8_w8a8_moe(
    hidden_states: torch.Tensor,  # BF16 [M, K_hidden] (after FP8 round-trip dequant)
    w13: torch.Tensor,  # FP8 [E, 2N, K_hidden]
    w13_scale: torch.Tensor,  # float32 [E, 2N, G] (G = K_hidden//group_K)
    w2: torch.Tensor,  # FP8 [E, K_hidden, N]
    w2_scale: torch.Tensor,  # float32 [E, K_hidden//gK, N//gN]
    topk_weights: torch.Tensor,  # float32 [M, top_k]
    topk_ids: torch.Tensor,  # int64 [M, top_k]
    N: int,
    group_K: int = FP8_W8A8_QUANT_GROUP,
    group_N: int = FP8_W8A8_QUANT_GROUP,
) -> torch.Tensor:
    """Reference implementation: dequantize weights then compute MoE."""
    M, K_hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    output = torch.zeros(M, K_hidden, dtype=torch.bfloat16)

    for tok in range(M):
        x = hidden_states[tok].float()  # [K_hidden]
        for k_idx in range(top_k):
            expert_id = topk_ids[tok, k_idx].item()
            w = topk_weights[tok, k_idx].item()

            # Stage 1: x @ w13.T — gate and up projections
            w13_e = w13[expert_id].float()  # [2N, K_hidden]
            ws_e = w13_scale[expert_id].float()  # [2N, G]
            G = ws_e.shape[1]
            w13_dq = torch.zeros_like(w13_e)
            for g in range(G):
                c0, c1 = g * group_K, (g + 1) * group_K
                w13_dq[:, c0:c1] = w13_e[:, c0:c1] * ws_e[:, g : g + 1]
            gate_up = (x.unsqueeze(0) @ w13_dq.T).squeeze(0)  # [2N]

            gate = gate_up[:N]
            up = gate_up[N:]
            act = F.silu(gate) * up  # [N]

            # Stage 2: act @ w2.T — down projection
            w2_e = w2[expert_id].float()  # [K_hidden, N]
            ws2_e = w2_scale[expert_id].float()  # [K_hidden//gK, N//gN]
            nK2 = K_hidden // group_K
            nN2 = N // group_N
            w2_dq = torch.zeros_like(w2_e)
            for gi in range(nK2):
                for gj in range(nN2):
                    r0, r1 = gi * group_K, (gi + 1) * group_K
                    c0, c1 = gj * group_N, (gj + 1) * group_N
                    w2_dq[r0:r1, c0:c1] = w2_e[r0:r1, c0:c1] * ws2_e[gi, gj]
            out_e = (act.unsqueeze(0) @ w2_dq.T).squeeze(0)  # [K_hidden]

            output[tok] += (w * out_e).to(torch.bfloat16)

    return output


@requires_cpu_fp8_w8a8
@pytest.mark.parametrize(
    "E,N,K,M,top_k",
    [
        # N and K must be multiples of FP8_W8A8_QUANT_GROUP=128
        (4, 128, 256, 1, 2),  # single token (decode phase)
        (4, 128, 256, 8, 2),
        (8, 256, 512, 16, 2),
        (4, 256, 512, 4, 1),
    ],
)
def test_fp8_w8a8_cpu_fused_moe(E: int, N: int, K: int, M: int, top_k: int):
    """Test fused_experts_cpu FP8_W8A8 shape/dtype and accuracy vs a
    dequantized BF16 reference."""
    set_random_seed(42)

    w13, w13_scale = _make_fp8_w8a8_weight_w13(E, 2 * N, K)
    w2, w2_scale = _make_fp8_w8a8_weight_w2(E, K, N)

    packed_w13, packed_w13_scale = _pack_fp8_w8a8_w13(w13, w13_scale)
    packed_w2 = _prepack_experts(w2)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    x_fp8, x_scales = torch.ops._C.quantize_fp8e4m3_vec(hidden_states, True, None)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_scales.shape == (M,)

    gen = torch.Generator().manual_seed(0)
    topk_weights = torch.softmax(torch.randn(M, top_k, generator=gen), dim=-1).float()
    topk_ids = torch.zeros(M, top_k, dtype=torch.int32)
    for i in range(M):
        topk_ids[i] = torch.randperm(E, generator=torch.Generator().manual_seed(i))[
            :top_k
        ].int()

    output = torch.empty(M, K, dtype=torch.bfloat16)
    ops.fused_experts_cpu(
        output,
        x_fp8,
        packed_w13,
        packed_w2,
        topk_weights,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A8,
        packed_w13_scale,  # w1_scale: [E, Nc, G, BLOCK_N]
        w2_scale,  # w2_scale: [E, K//128, N//128]
        None,  # w1_zero
        None,  # w2_zero
        [FP8_W8A8_QUANT_GROUP, FP8_W8A8_QUANT_GROUP],  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        True,  # is_vnni (weights already packed)
        x_scales,  # a1_scale: per-token FP8 activation scales
    )

    assert output.shape == (M, K)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    x_dequant = (x_fp8.float() * x_scales.unsqueeze(1)).bfloat16()
    ref = _ref_fp8_w8a8_moe(
        x_dequant, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids.long(), N
    )
    torch.testing.assert_close(output.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("returncode", "acknowledged", "complete", "expected"),
    [
        (0, False, True, "measured"),
        (-2, True, True, "measured"),
        (-2, False, True, "unavailable"),
        (-15, True, True, "unavailable"),
        (-2, True, False, "unavailable"),
    ],
)
def test_cpu_numa_moe_perf_accepts_only_complete_acknowledged_shutdown(
    tmp_path, returncode, acknowledged, complete, expected
):
    """An owned SIGINT stop is valid only after disable and complete counters."""
    from types import SimpleNamespace

    from benchmarks.kernels.cpu.numa_moe_perf import PerfCounters

    collector = PerfCounters(tmp_path / "sample", (32,), ["cycles", "instructions"])
    collector.csv_path.write_text(
        "100;;cycles;1000000;100.00;;\n"
        + ("50;;instructions;1000000;100.00;;\n" if complete else "")
    )
    collector._process = SimpleNamespace(wait=lambda: returncode)
    collector._disable_acknowledged = acknowledged
    collector._finish_result()
    assert collector.result["status"] == expected


def test_cpu_numa_moe_reference_cache_round_trip(tmp_path):
    identity = {
        "checkpoint_sha256": "checkpoint",
        "corpus_sha256": "corpus",
        "source_sha256": "source",
        "block_sizes": [128, 128],
        "chunk_tokens": 32,
        "torch_version": torch.__version__,
    }
    records = [
        {
            "layer": 4,
            "hidden": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        },
        {
            "layer": 4,
            "hidden": torch.arange(9, dtype=torch.bfloat16).reshape(3, 3),
        },
    ]
    outputs = [record["hidden"].clone() + 1 for record in records]
    writer = OracleCache(tmp_path, identity)
    reader = OracleCache(tmp_path, identity, read_only=True)

    assert writer.load(4, records) is None
    with pytest.raises(RuntimeError, match="missing layer"):
        reader.load(4, records)
    writer.store(4, records, outputs)
    cached = reader.load(4, records)
    assert cached is not None
    assert len(cached) == len(outputs)
    for actual, expected in zip(cached, outputs):
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, expected)
    with pytest.raises(RuntimeError, match="immutable"):
        writer.store(4, records, outputs)


def test_cpu_numa_moe_reference_cache_rejects_wrong_identity_and_corruption(
    tmp_path,
):
    identity = {"corpus_sha256": "original", "chunk_tokens": 32}
    records = [
        {
            "layer": 2,
            "hidden": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        }
    ]
    outputs = [records[0]["hidden"].clone()]
    OracleCache(tmp_path, identity).store(2, records, outputs)

    with pytest.raises(RuntimeError, match="identity"):
        OracleCache(
            tmp_path,
            {"corpus_sha256": "different", "chunk_tokens": 32},
            read_only=True,
        )

    (tmp_path / "layer-2.pt").write_bytes(b"corrupted archive")
    reader = OracleCache(tmp_path, identity, read_only=True)
    with pytest.raises(RuntimeError, match="checksum"):
        reader.load(2, records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

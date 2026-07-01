# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU EP communicator tests.

The EP communicator always uses the torch.distributed path on CPU, even on a
single node, because the SHM fastpath is only enabled for tp/pp/dp groups.
That makes these single-node tests representative of the live EP code path.
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

HIDDEN_SIZE = 8
NUM_EXPERTS = 6
TOPK = 2


def _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port):
    """Init vLLM distributed env for TP=tp_size, DP=dp_size."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=tp_size,
            rank=tp_rank,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(tp_size, 1, backend="gloo")


def _make_forward_context(dp_rank, dp_size, num_tokens, num_tokens_across_dp):
    """Create a forward context with explicit DP token counts."""
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        dp_metadata = None

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=dp_size,
        is_moe_model=True,
        data_parallel_rank=dp_rank,
    )
    return set_forward_context(
        _AttnMeta(),
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=torch.tensor(num_tokens_across_dp, dtype=torch.int),
    )


def _spawn_workers(worker_fn, world_size, tp_size, dp_size, params):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    port = str(get_open_port())
    dp_port = str(get_open_port())
    err_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=worker_fn,
            args=(rank, world_size, tp_size, dp_size, port, dp_port, params, err_q),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    errors = []
    while not err_q.empty():
        errors.append(err_q.get_nowait())
    err_q.close()
    err_q.join_thread()
    if errors:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(errors))


def _filled(rows: int, cols: int, value: float, dtype=torch.float32) -> torch.Tensor:
    return torch.full((rows, cols), value, dtype=dtype)


def _concat_rank_values(sizes, cols, value_scale, dtype=torch.float32):
    chunks = [
        _filled(size, cols, value_scale * (rank + 1), dtype)
        for rank, size in enumerate(sizes)
    ]
    return torch.cat(chunks, dim=0)


def _concat_rank_ids(sizes):
    chunks = [
        torch.full((size, TOPK), rank, dtype=torch.long)
        for rank, size in enumerate(sizes)
    ]
    return torch.cat(chunks, dim=0)


def _expected_combined(rank, sizes, cols):
    total_rows = sum(sizes)
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    rows = (
        torch.arange(total_rows, dtype=torch.float32)
        .unsqueeze(1)
        .expand(
            total_rows,
            cols,
        )
    )
    tag_sum = sum(float((r + 1) * 1000) for r in range(len(sizes)))
    return rows[start:end] * len(sizes) + tag_sum


def _sp_local_sizes(dp_token_counts, tp_size):
    sizes = []
    for dp_tokens in dp_token_counts:
        local_rows = (dp_tokens + tp_size - 1) // tp_size
        sizes.extend([local_rows] * tp_size)
    return sizes


def _ragged_dispatch_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    sizes,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_comm_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.forward_context import get_forward_context

        local_rows = sizes[rank]
        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)

        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(sizes)

        with _make_forward_context(rank, dp_size, local_rows, sizes):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                gathered_hidden, gathered_router = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)

                gathered_hidden2, gathered_weights, gathered_ids = (
                    get_ep_group().dispatch(
                        hidden.clone(),
                        weights.clone(),
                        ids.clone(),
                    )
                )
                torch.testing.assert_close(gathered_hidden2, expected_hidden)
                torch.testing.assert_close(gathered_weights, expected_weights)
                torch.testing.assert_close(gathered_ids, expected_ids)

                total_rows = sum(sizes)
                expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                    1
                ).expand(total_rows, HIDDEN_SIZE).contiguous() + float(
                    (rank + 1) * 1000
                )
                combined = get_ep_group().combine(expert_out)
                torch.testing.assert_close(
                    combined,
                    _expected_combined(rank, sizes, HIDDEN_SIZE),
                )

        dist.barrier()
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


def _sequence_parallel_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    dp_token_counts,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_sp_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.forward_context import get_forward_context

        local_sizes = _sp_local_sizes(dp_token_counts, tp_size)
        dp_rank = rank // tp_size
        local_rows = local_sizes[rank]

        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)

        expected_hidden = _concat_rank_values(local_sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(local_sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(local_sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(local_sizes)

        with _make_forward_context(
            dp_rank,
            dp_size,
            dp_token_counts[dp_rank],
            dp_token_counts,
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=tp_size):
                gathered_hidden, gathered_router = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        is_sequence_parallel=True,
                    )
                )
                torch.testing.assert_close(gathered_hidden, expected_hidden)
                torch.testing.assert_close(gathered_router, expected_router)

                gathered_hidden2, gathered_weights, gathered_ids = (
                    get_ep_group().dispatch(
                        hidden.clone(),
                        weights.clone(),
                        ids.clone(),
                        is_sequence_parallel=True,
                    )
                )
                torch.testing.assert_close(gathered_hidden2, expected_hidden)
                torch.testing.assert_close(gathered_weights, expected_weights)
                torch.testing.assert_close(gathered_ids, expected_ids)

                total_rows = sum(local_sizes)
                expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                    1
                ).expand(total_rows, HIDDEN_SIZE).contiguous() + float(
                    (rank + 1) * 1000
                )
                combined = get_ep_group().combine(
                    expert_out,
                    is_sequence_parallel=True,
                )
                torch.testing.assert_close(
                    combined,
                    _expected_combined(rank, local_sizes, HIDDEN_SIZE),
                )

        dist.barrier()
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def _run_compiled_fastpath(rank, sizes):
    from vllm.distributed.parallel_state import get_ep_group

    device_communicator = get_ep_group().device_communicator
    assert device_communicator is not None
    dispatch_op = device_communicator._ep_dispatch_rl_op
    combine_op = device_communicator._ep_combine_op

    def fastpath_step(hidden_states, router_logits):
        gathered_hidden, _ = dispatch_op(hidden_states, router_logits)
        return combine_op(gathered_hidden) + hidden_states

    compiled = torch.compile(fastpath_step, fullgraph=True, backend="eager")
    local_rows = sizes[rank]
    hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
    router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
    output = compiled(hidden, router)
    torch.testing.assert_close(output, hidden * (len(sizes) + 1))


def _compile_fastpath_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    size_patterns,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_ep_compile_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
        torch._dynamo.reset()

        for sizes in size_patterns:
            with _make_forward_context(rank, dp_size, sizes[rank], sizes):
                _run_compiled_fastpath(rank, sizes)

        dist.barrier()
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


@pytest.mark.distributed
@pytest.mark.parametrize("sizes", [[2, 1], [0, 3]], ids=["ragged", "zero-rank"])
def test_cpu_ep_dispatch_combine_ragged(sizes):
    _spawn_workers(
        _ragged_dispatch_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=sizes,
    )


@pytest.mark.distributed
def test_cpu_ep_sequence_parallel_uses_ep_group():
    _spawn_workers(
        _sequence_parallel_worker,
        world_size=4,
        tp_size=2,
        dp_size=2,
        params=[3, 1],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_compile_ragged_fastpath():
    _spawn_workers(
        _compile_fastpath_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[2, 1], [0, 3]],
    )

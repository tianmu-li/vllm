# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU communicator tests for DP SHM and EP dispatch paths."""

import gc
import json
import os
import signal
import tempfile
import time
import traceback
import weakref
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils.network_utils import get_open_port

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

import vllm._custom_ops  # noqa: E402,F401  # populates torch.ops._C for HAS_CPU_SHM
from vllm.compilation.decorators import support_torch_compile  # noqa: E402

HIDDEN_SIZE = 8
NUM_EXPERTS = 6
TOPK = 2
HAS_CPU_SHM = hasattr(torch.ops._C, "init_shm_manager") and (
    current_platform.get_cpu_architecture()
    in (CpuArchEnum.X86, CpuArchEnum.ARM, CpuArchEnum.POWERPC)
)


def _cpu_ep_aot_step(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    from vllm.distributed.parallel_state import get_ep_group

    gathered_hidden, _ = get_ep_group().dispatch_router_logits(
        hidden_states,
        router_logits,
    )
    return get_ep_group().combine(gathered_hidden) + hidden_states


def _cpu_ep_inductor_postprocess(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states * 2 + 1


@support_torch_compile(
    dynamic_arg_dims={
        "hidden_states": {0: "local_rows"},
        "router_logits": {0: "local_rows"},
    }
)
class _CpuEpAotModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return _cpu_ep_aot_step(hidden_states, router_logits)


def test_cpu_shm_group_name_eligibility():
    from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator

    assert CpuCommunicator._is_cpushm_group_name("tp:0")
    assert CpuCommunicator._is_cpushm_group_name("pp:0")
    assert CpuCommunicator._is_cpushm_group_name("dp:0")
    assert CpuCommunicator._is_cpushm_group_name("ep:0")
    assert not CpuCommunicator._is_cpushm_group_name("eplb:0")
    assert not CpuCommunicator._is_cpushm_group_name("anonymous:0")


@pytest.mark.parametrize(
    ("token_counts", "sp_size", "world_size", "expected"),
    [
        ([2, 1], 1, 2, [2, 1]),
        ([5, 4, 3], 2, 6, [3, 3, 2, 2, 2, 2]),
    ],
    ids=["non-sp", "sp"],
)
def test_cpu_ep_sizes(token_counts, sp_size, world_size, expected):
    from vllm.distributed.device_communicators.cpu_communicator import (
        _get_cpu_ep_sizes,
    )

    counts = torch.tensor(token_counts, dtype=torch.int32)
    assert _get_cpu_ep_sizes(counts, sp_size, world_size) == expected


@pytest.mark.parametrize(
    ("token_counts", "sp_size", "world_size", "error"),
    [
        (torch.tensor([2.0, 1.0]), 1, 2, "integer dtype"),
        (torch.tensor([[2, 1]]), 1, 2, "one-dimensional"),
        (torch.tensor([2, -1]), 1, 2, "nonnegative"),
        (torch.tensor([2, 1]), 0, 2, "must be positive"),
        (torch.tensor([2, 1]), 2, 2, "communicator size mismatch"),
    ],
    ids=["dtype", "dimensions", "negative", "sp-zero", "world-size"],
)
def test_cpu_ep_sizes_reject_invalid_input(
    token_counts,
    sp_size,
    world_size,
    error,
):
    from vllm.distributed.device_communicators.cpu_communicator import (
        _get_cpu_ep_sizes,
    )

    with pytest.raises(RuntimeError, match=error):
        _get_cpu_ep_sizes(token_counts, sp_size, world_size)


def _bare_cpu_communicator(group_name):
    from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator

    communicator = CpuCommunicator.__new__(CpuCommunicator)
    communicator.unique_name = group_name
    return communicator


def test_cpu_communicator_registry_rejects_duplicate_live_group(monkeypatch):
    from vllm.distributed.device_communicators import cpu_communicator

    monkeypatch.setattr(
        cpu_communicator,
        "_CPU_COMMUNICATORS",
        weakref.WeakValueDictionary(),
    )
    first = _bare_cpu_communicator("ep:duplicate")
    second = _bare_cpu_communicator("ep:duplicate")

    cpu_communicator._register_cpu_communicator(first)
    with pytest.raises(RuntimeError, match="already registered"):
        cpu_communicator._register_cpu_communicator(second)


def test_cpu_communicator_registry_unregisters_only_matching_object(monkeypatch):
    from vllm.distributed.device_communicators import cpu_communicator

    monkeypatch.setattr(
        cpu_communicator,
        "_CPU_COMMUNICATORS",
        weakref.WeakValueDictionary(),
    )
    first = _bare_cpu_communicator("ep:registered")
    other = _bare_cpu_communicator("ep:registered")

    cpu_communicator._register_cpu_communicator(first)
    cpu_communicator._unregister_cpu_communicator(other)
    assert cpu_communicator._resolve_cpu_communicator("ep:registered") is first

    cpu_communicator._unregister_cpu_communicator(first)
    with pytest.raises(RuntimeError, match="is not registered"):
        cpu_communicator._resolve_cpu_communicator("ep:registered")


def test_cpu_communicator_registry_rejects_unknown_and_expired_groups(monkeypatch):
    from vllm.distributed.device_communicators import cpu_communicator

    monkeypatch.setattr(
        cpu_communicator,
        "_CPU_COMMUNICATORS",
        weakref.WeakValueDictionary(),
    )
    hidden = torch.empty(0, HIDDEN_SIZE)
    counts = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="is not registered"):
        cpu_communicator._cpu_ep_combine_v1(hidden, counts, 1, "ep:unknown")

    communicator = _bare_cpu_communicator("ep:expired")
    reference = weakref.ref(communicator)
    cpu_communicator._register_cpu_communicator(communicator)
    del communicator
    gc.collect()

    assert reference() is None
    with pytest.raises(RuntimeError, match="is not registered"):
        cpu_communicator._cpu_ep_combine_v1(hidden, counts, 1, "ep:expired")


def test_cpu_ep_dispatch_fake_outputs_share_dynamic_rows():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm.distributed.device_communicators.cpu_communicator import (
        _cpu_ep_dispatch_router_logits_v1,
        _cpu_ep_dispatch_v1,
    )

    with FakeTensorMode(shape_env=ShapeEnv()):
        hidden = torch.empty(2, HIDDEN_SIZE, dtype=torch.bfloat16)
        router = torch.empty(2, NUM_EXPERTS, dtype=torch.float32)
        weights = torch.empty(2, TOPK, dtype=torch.float32)
        ids = torch.empty(2, TOPK, dtype=torch.int64)
        counts = torch.tensor([2, 1], dtype=torch.int32)

        gathered_hidden, gathered_router = _cpu_ep_dispatch_router_logits_v1(
            hidden,
            router,
            counts,
            1,
            "dp:fake",
        )
        dispatched_hidden, dispatched_weights, dispatched_ids = _cpu_ep_dispatch_v1(
            hidden,
            weights,
            ids,
            counts,
            1,
            "dp:fake",
        )

    assert gathered_hidden.shape[0] == gathered_router.shape[0]
    assert dispatched_hidden.shape[0] == dispatched_weights.shape[0]
    assert dispatched_hidden.shape[0] == dispatched_ids.shape[0]
    assert gathered_hidden.shape[1:] == hidden.shape[1:]
    assert gathered_router.shape[1:] == router.shape[1:]
    assert dispatched_weights.shape[1:] == weights.shape[1:]
    assert dispatched_ids.shape[1:] == ids.shape[1:]
    assert gathered_hidden.dtype == hidden.dtype
    assert gathered_router.dtype == router.dtype
    assert dispatched_weights.dtype == weights.dtype
    assert dispatched_ids.dtype == ids.dtype


def test_cpu_ep_combine_fake_preserves_trailing_shape_and_dtype():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm.distributed.device_communicators.cpu_communicator import (
        _cpu_ep_combine_v1,
    )

    with FakeTensorMode(shape_env=ShapeEnv()):
        hidden = torch.empty(3, HIDDEN_SIZE, dtype=torch.bfloat16)
        counts = torch.tensor([2, 1], dtype=torch.int32)
        output = _cpu_ep_combine_v1(hidden, counts, 1, "dp:fake")

    assert output.shape[1:] == hidden.shape[1:]
    assert output.dtype == hidden.dtype


def _ensure_spawn_start_method():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


def _report_worker_failure(rank: int, err_q: mp.Queue, err: Exception) -> None:
    err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
    raise SystemExit(1) from err


def _terminate_worker_process_groups(procs) -> None:
    for proc in procs:
        if not proc.is_alive():
            continue
        if os.name == "posix" and proc.pid is not None:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()

    deadline = time.monotonic() + 5
    for proc in procs:
        if proc.is_alive():
            proc.join(max(deadline - time.monotonic(), 0))

    for proc in procs:
        if not proc.is_alive():
            continue
        if os.name == "posix" and proc.pid is not None:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.join()


def _collect_worker_failures(
    procs,
    err_q: mp.Queue,
    timeout_s: float | None = None,
) -> list[str]:
    exit_errors = []
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    for rank, proc in enumerate(procs):
        if deadline is None:
            proc.join()
        else:
            proc.join(max(deadline - time.monotonic(), 0))

    timed_out = deadline is not None and any(proc.is_alive() for proc in procs)
    if timed_out:
        _terminate_worker_process_groups(procs)

    for rank, proc in enumerate(procs):
        if proc.exitcode != 0:
            exit_errors.append(f"[Rank {rank}] worker exited with code {proc.exitcode}")
    if timed_out:
        exit_errors.append(f"Worker(s) exceeded the {timeout_s:.0f}s timeout")

    errors = []
    while not err_q.empty():
        errors.append(err_q.get_nowait())
    err_q.close()
    err_q.join_thread()
    return errors + exit_errors


def _run_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    worker_fn,
    params,
    err_q,
):
    try:
        if os.name == "posix":
            os.setsid()
        worker_fn(rank, world_size, tp_size, dp_size, port, dp_port, params, err_q)
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


def _spawn_workers(
    worker_fn,
    world_size,
    tp_size,
    dp_size,
    params,
    *,
    distributed_init_ports=None,
    dp_port=None,
    timeout_s: float | None = None,
):
    _ensure_spawn_start_method()

    if distributed_init_ports is None:
        shared_init_port = get_open_port()
        distributed_init_ports = [shared_init_port] * world_size
    elif len(distributed_init_ports) != world_size:
        raise ValueError("distributed_init_ports must provide one port per worker rank")

    if dp_port is None:
        dp_port = get_open_port()

    err_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=_run_worker,
            args=(
                rank,
                world_size,
                tp_size,
                dp_size,
                distributed_init_ports[rank],
                dp_port,
                worker_fn,
                params,
                err_q,
            ),
        )
        proc.start()
        procs.append(proc)

    failures = _collect_worker_failures(procs, err_q, timeout_s)
    if failures:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(failures))


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


def _make_forward_context(
    dp_rank,
    dp_size,
    num_tokens,
    num_tokens_across_dp,
    vllm_config=None,
):
    """Create a forward context with explicit DP token counts."""
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        dp_metadata = None

    if vllm_config is None:
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


def _assert_no_alias(output, input_):
    assert not torch._C._is_alias_of(output, input_)


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
        extra = _filled(local_rows, 1, float((rank + 1) * 1000))

        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(sizes)
        expected_extra = _concat_rank_values(sizes, 1, 1000.0)

        with _make_forward_context(rank, dp_size, local_rows, sizes):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            sentinel_sizes = [101 + rank]
            dp_metadata.local_sizes = sentinel_sizes

            hidden_input = hidden.clone()
            router_input = router.clone()
            gathered_hidden, gathered_router = get_ep_group().dispatch_router_logits(
                hidden_input,
                router_input,
            )
            torch.testing.assert_close(gathered_hidden, expected_hidden)
            torch.testing.assert_close(gathered_router, expected_router)
            _assert_no_alias(gathered_hidden, hidden_input)
            _assert_no_alias(gathered_router, router_input)
            assert dp_metadata.local_sizes is sentinel_sizes

            hidden_input = hidden.clone()
            weights_input = weights.clone()
            ids_input = ids.clone()
            gathered_hidden2, gathered_weights, gathered_ids = get_ep_group().dispatch(
                hidden_input,
                weights_input,
                ids_input,
            )
            torch.testing.assert_close(gathered_hidden2, expected_hidden)
            torch.testing.assert_close(gathered_weights, expected_weights)
            torch.testing.assert_close(gathered_ids, expected_ids)
            _assert_no_alias(gathered_hidden2, hidden_input)
            _assert_no_alias(gathered_weights, weights_input)
            _assert_no_alias(gathered_ids, ids_input)
            assert dp_metadata.local_sizes is sentinel_sizes

            total_rows = sum(sizes)
            expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                1
            ).expand(total_rows, HIDDEN_SIZE).contiguous() + float((rank + 1) * 1000)
            combined = get_ep_group().combine(expert_out)
            torch.testing.assert_close(
                combined,
                _expected_combined(rank, sizes, HIDDEN_SIZE),
            )
            _assert_no_alias(combined, expert_out)
            assert dp_metadata.local_sizes is sentinel_sizes

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                gathered_hidden3, gathered_router3, gathered_extras = (
                    get_ep_group().dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        extra_tensors=[extra.clone()],
                    )
                )
                torch.testing.assert_close(gathered_hidden3, expected_hidden)
                torch.testing.assert_close(gathered_router3, expected_router)
                assert len(gathered_extras) == 1
                torch.testing.assert_close(gathered_extras[0], expected_extra)
            assert dp_metadata.local_sizes is sentinel_sizes

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


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

        from vllm.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
            _CPUSHMDistributed,
        )
        from vllm.distributed.parallel_state import get_ep_group, get_tp_group
        from vllm.forward_context import get_forward_context

        expected_local_sizes = _sp_local_sizes(dp_token_counts, tp_size)
        dp_rank = rank // tp_size
        expected_tp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

        assert get_tp_group().ranks == expected_tp_ranks
        ep_group = get_ep_group()
        assert ep_group.ranks == list(range(world_size))

        ep_communicator = ep_group.device_communicator
        assert isinstance(ep_communicator, CpuCommunicator)
        shm_counts = {"all_gatherv": 0, "reduce_scatterv": 0}
        if HAS_CPU_SHM:
            assert isinstance(ep_communicator.dist_module, _CPUSHMDistributed)
            orig_all_gatherv = ep_communicator.dist_module.all_gatherv
            orig_reduce_scatterv = ep_communicator.dist_module.reduce_scatterv

            def counted_all_gatherv(inputs, outputs, sizes):
                shm_counts["all_gatherv"] += 1
                return orig_all_gatherv(inputs, outputs, sizes)

            def counted_reduce_scatterv(input_, output, sizes):
                shm_counts["reduce_scatterv"] += 1
                return orig_reduce_scatterv(input_, output, sizes)

            ep_communicator.dist_module.all_gatherv = counted_all_gatherv
            ep_communicator.dist_module.reduce_scatterv = counted_reduce_scatterv

        local_rows = expected_local_sizes[rank]

        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
        weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
        ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)
        extra = _filled(local_rows, 1, float((rank + 1) * 1000))

        expected_hidden = _concat_rank_values(expected_local_sizes, HIDDEN_SIZE, 1.0)
        expected_router = _concat_rank_values(expected_local_sizes, NUM_EXPERTS, 10.0)
        expected_weights = _concat_rank_values(expected_local_sizes, TOPK, 100.0)
        expected_ids = _concat_rank_ids(expected_local_sizes)
        expected_extra = _concat_rank_values(expected_local_sizes, 1, 1000.0)

        with _make_forward_context(
            dp_rank,
            dp_size,
            dp_token_counts[dp_rank],
            dp_token_counts,
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            sentinel_sizes = [201 + rank]
            dp_metadata.local_sizes = sentinel_sizes

            hidden_input = hidden.clone()
            router_input = router.clone()
            gathered_hidden, gathered_router = ep_group.dispatch_router_logits(
                hidden_input,
                router_input,
                is_sequence_parallel=True,
            )
            torch.testing.assert_close(gathered_hidden, expected_hidden)
            torch.testing.assert_close(gathered_router, expected_router)
            _assert_no_alias(gathered_hidden, hidden_input)
            _assert_no_alias(gathered_router, router_input)
            assert dp_metadata.local_sizes is sentinel_sizes

            hidden_input = hidden.clone()
            weights_input = weights.clone()
            ids_input = ids.clone()
            gathered_hidden2, gathered_weights, gathered_ids = ep_group.dispatch(
                hidden_input,
                weights_input,
                ids_input,
                is_sequence_parallel=True,
            )
            torch.testing.assert_close(gathered_hidden2, expected_hidden)
            torch.testing.assert_close(gathered_weights, expected_weights)
            torch.testing.assert_close(gathered_ids, expected_ids)
            _assert_no_alias(gathered_hidden2, hidden_input)
            _assert_no_alias(gathered_weights, weights_input)
            _assert_no_alias(gathered_ids, ids_input)
            assert dp_metadata.local_sizes is sentinel_sizes

            total_rows = sum(expected_local_sizes)
            expert_out = torch.arange(total_rows, dtype=torch.float32).unsqueeze(
                1
            ).expand(total_rows, HIDDEN_SIZE).contiguous() + float((rank + 1) * 1000)
            combined = ep_group.combine(
                expert_out,
                is_sequence_parallel=True,
            )
            torch.testing.assert_close(
                combined,
                _expected_combined(rank, expected_local_sizes, HIDDEN_SIZE),
            )
            _assert_no_alias(combined, expert_out)
            assert dp_metadata.local_sizes is sentinel_sizes

            with dp_metadata.sp_local_sizes(sequence_parallel_size=tp_size) as sizes:
                assert sizes == expected_local_sizes
                gathered_hidden3, gathered_router3, gathered_extras = (
                    ep_group.dispatch_router_logits(
                        hidden.clone(),
                        router.clone(),
                        is_sequence_parallel=True,
                        extra_tensors=[extra.clone()],
                    )
                )
                torch.testing.assert_close(gathered_hidden3, expected_hidden)
                torch.testing.assert_close(gathered_router3, expected_router)
                assert len(gathered_extras) == 1
                torch.testing.assert_close(gathered_extras[0], expected_extra)
            assert dp_metadata.local_sizes is sentinel_sizes

        if HAS_CPU_SHM:
            assert shm_counts == {"all_gatherv": 3, "reduce_scatterv": 1}

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def _run_compiled_fastpath(
    rank,
    tp_size,
    dp_size,
    size_patterns,
    is_sequence_parallel,
):
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.forward_context import get_forward_context

    ep_group = get_ep_group()

    def fastpath_step(hidden_states, router_logits, topk_weights, topk_ids):
        gathered_hidden, _ = ep_group.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel=is_sequence_parallel,
        )
        dispatched_hidden, _, _ = ep_group.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel=is_sequence_parallel,
        )
        combined = ep_group.combine(
            gathered_hidden + dispatched_hidden,
            is_sequence_parallel=is_sequence_parallel,
        )
        return combined + hidden_states

    compiled = torch.compile(
        fastpath_step,
        fullgraph=True,
        dynamic=True,
        backend="inductor",
    )
    collective_size = ep_group.world_size if is_sequence_parallel else dp_size
    dp_rank = rank // tp_size

    with (
        torch._dynamo.config.patch(error_on_recompile=True),
        torch.fx.experimental._config.patch(use_duck_shape=False),
    ):
        for token_counts in size_patterns:
            local_sizes = (
                _sp_local_sizes(token_counts, tp_size)
                if is_sequence_parallel
                else token_counts
            )
            local_rows = local_sizes[rank if is_sequence_parallel else dp_rank]
            hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
            router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))
            weights = _filled(local_rows, TOPK, float((rank + 1) * 100))
            ids = torch.full((local_rows, TOPK), rank, dtype=torch.long)
            for tensor in (hidden, router, weights, ids):
                torch._dynamo.decorators.mark_unbacked(
                    tensor,
                    0,
                    shape_id="local_rows",
                )

            with _make_forward_context(
                dp_rank,
                dp_size,
                token_counts[dp_rank],
                token_counts,
            ):
                dp_metadata = get_forward_context().dp_metadata
                assert dp_metadata is not None
                sentinel_sizes = [301 + rank]
                dp_metadata.local_sizes = sentinel_sizes
                output = compiled(hidden, router, weights, ids)
                assert dp_metadata.local_sizes is sentinel_sizes

            torch.testing.assert_close(
                output,
                hidden * (2 * collective_size + 1),
            )


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
        is_sequence_parallel, patterns = size_patterns
        _run_compiled_fastpath(
            rank,
            tp_size,
            dp_size,
            patterns,
            is_sequence_parallel,
        )

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _aot_cache_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        phase, cache_root, token_counts = params
        os.environ["VLLM_DIST_IDENT"] = f"test_cpu_ep_aot_{phase}_{port}"
        os.environ["VLLM_CACHE_ROOT"] = cache_root
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache_root) / "torchinductor")
        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"] = "1"
        os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
        os.environ["VLLM_USE_BYTECODE_HOOK"] = "0"

        from torch._dynamo.utils import counters

        from vllm.compilation.counter import compilation_counter
        from vllm.config import (
            CompilationConfig,
            CompilationMode,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.config.compilation import DynamicShapesConfig, DynamicShapesType
        from vllm.envs import disable_envs_cache
        from vllm.forward_context import get_forward_context

        disable_envs_cache()
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)
        torch._dynamo.reset()
        counters.clear()
        before = compilation_counter.clone()

        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend="inductor",
                splitting_ops=[
                    "vllm::cpu_ep_dispatch_router_logits_v1",
                    "vllm::cpu_ep_combine_v1",
                ],
                dynamic_shapes_config=DynamicShapesConfig(
                    type=DynamicShapesType.UNBACKED
                ),
            )
        )
        local_rows = token_counts[rank]
        hidden = _filled(local_rows, HIDDEN_SIZE, float(rank + 1))
        router = _filled(local_rows, NUM_EXPERTS, float((rank + 1) * 10))

        with (
            set_current_vllm_config(vllm_config),
            _make_forward_context(
                rank,
                dp_size,
                local_rows,
                token_counts,
                vllm_config=vllm_config,
            ),
            torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
            torch.fx.experimental._config.patch(use_duck_shape=False),
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            sentinel_sizes = [401 + rank]
            dp_metadata.local_sizes = sentinel_sizes
            model = _CpuEpAotModel(vllm_config=vllm_config)  # type: ignore[call-arg]
            dist.barrier()
            output = model(hidden, router)
            assert dp_metadata.local_sizes is sentinel_sizes
            dist.barrier()
            torch._dynamo.reset()
            torch._dynamo.decorators.mark_unbacked(
                output,
                0,
                shape_id="local_rows",
            )
            postprocess = torch.compile(
                _cpu_ep_inductor_postprocess,
                fullgraph=True,
                dynamic=True,
                backend="inductor",
            )
            postprocessed = postprocess(output)

        torch.testing.assert_close(output, hidden * (dp_size + 1))
        torch.testing.assert_close(postprocessed, output * 2 + 1)
        dist.barrier()

        snapshot = {
            "aot_compiles": (
                compilation_counter.num_aot_compiles - before.num_aot_compiles
            ),
            "aot_saved": (
                compilation_counter.num_aot_artifacts_saved
                - before.num_aot_artifacts_saved
            ),
            "aot_loaded": (
                compilation_counter.num_aot_artifacts_loaded
                - before.num_aot_artifacts_loaded
            ),
            "standalone_saved": (
                compilation_counter.num_compiled_artifacts_saved
                - before.num_compiled_artifacts_saved
            ),
            "standalone_loaded": (
                compilation_counter.num_compiled_artifacts_loaded
                - before.num_compiled_artifacts_loaded
            ),
            "fxgraph_cache_hit": counters["inductor"]["fxgraph_cache_hit"],
            "fxgraph_cache_miss": counters["inductor"]["fxgraph_cache_miss"],
        }
        result_path = Path(cache_root) / f"{phase}-rank-{rank}.json"
        result_path.write_text(json.dumps(snapshot), encoding="utf-8")
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _get_dp_shm_communicator():
    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
        _CPUSHMDistributed,
    )
    from vllm.distributed.parallel_state import get_dp_group

    dp_group = get_dp_group()
    communicator = dp_group.device_communicator
    assert isinstance(communicator, CpuCommunicator)
    assert isinstance(communicator.dist_module, _CPUSHMDistributed)
    return dp_group, communicator


def _get_tp_shm_communicator():
    from vllm.distributed.device_communicators.cpu_communicator import (
        CpuCommunicator,
        _CPUSHMDistributed,
    )
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    communicator = tp_group.device_communicator
    assert isinstance(communicator, CpuCommunicator)
    assert communicator.unique_name.startswith("tp:")
    assert isinstance(communicator.dist_module, _CPUSHMDistributed)
    return tp_group, communicator


def _get_ragged_buffer_capacity(communicator, tensor, buffers, *, per_rank=False):
    buffer = buffers.get(communicator._ragged_buffer_key(tensor))
    if buffer is None:
        return 0
    if per_rank:
        return buffer.shape[0] // communicator.world_size
    return buffer.shape[0]


def _get_ragged_capacities(communicator, tensor):
    return (
        _get_ragged_buffer_capacity(
            communicator,
            tensor,
            communicator._ragged_pad_buffers,
        ),
        _get_ragged_buffer_capacity(
            communicator,
            tensor,
            communicator._ragged_shm_gather_buffers,
            per_rank=True,
        ),
    )


def _ragged_shm_buffers_worker(
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
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_ragged_buffers_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        small_sizes, large_sizes = size_patterns
        dp_group, communicator = _get_dp_shm_communicator()

        small_hidden = _filled(small_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        small_result = dp_group.all_gatherv(small_hidden.clone(), sizes=small_sizes)
        torch.testing.assert_close(
            small_result,
            _concat_rank_values(small_sizes, HIDDEN_SIZE, 1.0),
        )
        small_caps = _get_ragged_capacities(communicator, small_hidden)
        small_result_repeat = dp_group.all_gatherv(
            small_hidden.clone(),
            sizes=small_sizes,
        )
        torch.testing.assert_close(
            small_result_repeat,
            _concat_rank_values(small_sizes, HIDDEN_SIZE, 1.0),
        )
        small_repeat_caps = _get_ragged_capacities(communicator, small_hidden)

        large_hidden = _filled(large_sizes[rank], HIDDEN_SIZE, float(rank + 1))
        large_result = dp_group.all_gatherv(large_hidden.clone(), sizes=large_sizes)
        torch.testing.assert_close(
            large_result,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        large_caps = _get_ragged_capacities(communicator, large_hidden)

        large_result_repeat = dp_group.all_gatherv(
            large_hidden.clone(),
            sizes=large_sizes,
        )
        torch.testing.assert_close(
            large_result_repeat,
            _concat_rank_values(large_sizes, HIDDEN_SIZE, 1.0),
        )
        repeat_caps = _get_ragged_capacities(communicator, large_hidden)

        assert small_caps == (max(small_sizes), max(small_sizes))
        assert small_repeat_caps == small_caps
        assert large_caps == (max(large_sizes), max(large_sizes))
        assert large_caps[0] > small_caps[0]
        assert large_caps[1] > small_caps[1]
        assert repeat_caps == large_caps

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _uniform_sizes_shm_worker(
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
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_uniform_sizes_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, communicator = _get_dp_shm_communicator()
        hidden = _filled(sizes[rank], HIDDEN_SIZE, float(rank + 1))
        expected_hidden = _concat_rank_values(sizes, HIDDEN_SIZE, 1.0)

        explicit_sizes = dp_group.all_gatherv(hidden.clone(), sizes=sizes)
        implicit_sizes = dp_group.all_gatherv(hidden.clone(), sizes=None)

        torch.testing.assert_close(explicit_sizes, expected_hidden)
        torch.testing.assert_close(implicit_sizes, expected_hidden)
        torch.testing.assert_close(explicit_sizes, implicit_sizes)
        assert _get_ragged_capacities(communicator, hidden) == (0, 0)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _batched_all_gatherv_worker(
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
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_batched_gatherv_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, communicator = _get_dp_shm_communicator()

        def _unexpected_all_gather_into_tensor(*args, **kwargs):
            raise AssertionError("batched list all_gatherv used per-tensor gather")

        communicator.dist_module.all_gather_into_tensor = (  # type: ignore[method-assign]
            _unexpected_all_gather_into_tensor
        )

        for sizes in size_patterns:
            rows = sizes[rank]
            hidden = (
                torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32)
                .reshape(rows, HIDDEN_SIZE)
                .to(torch.bfloat16)
                + rank * 100
            )
            router = _filled(rows, NUM_EXPERTS, float((rank + 1) * 10))
            ids = torch.full((rows, TOPK), rank, dtype=torch.long)

            result = dp_group.all_gatherv([hidden, router, ids], sizes=sizes)
            repeated = dp_group.all_gatherv([hidden, router, ids], sizes=sizes)

            hidden_chunks = [
                (
                    torch.arange(size * HIDDEN_SIZE, dtype=torch.float32)
                    .reshape(size, HIDDEN_SIZE)
                    .to(torch.bfloat16)
                    + src_rank * 100
                )
                for src_rank, size in enumerate(sizes)
            ]
            expected_hidden = torch.cat(hidden_chunks, dim=0)
            expected_router = torch.cat(
                [
                    _filled(size, NUM_EXPERTS, float((src_rank + 1) * 10))
                    for src_rank, size in enumerate(sizes)
                ],
                dim=0,
            )
            expected_ids = _concat_rank_ids(sizes)

            for gathered in (result, repeated):
                torch.testing.assert_close(gathered[0], expected_hidden)
                torch.testing.assert_close(gathered[1], expected_router)
                torch.testing.assert_close(gathered[2], expected_ids)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _all_zero_gatherv_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_zero_gather_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_dp_group

        sizes = [0] * dp_size
        hidden = torch.empty((0, HIDDEN_SIZE), dtype=torch.float32)
        gathered = get_dp_group().all_gatherv(hidden.clone(), sizes=sizes)

        assert gathered.shape == (0, HIDDEN_SIZE)
        assert gathered.numel() == 0

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _implicit_uneven_reduce_scatterv_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        os.environ.setdefault(
            "VLLM_DIST_IDENT", f"test_cpu_dp_uneven_reduce_scatterv_{port}"
        )
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.parallel_state import get_dp_group

        hidden = _filled(3, HIDDEN_SIZE, float(rank + 1))
        with pytest.raises(
            AssertionError,
            match="Implicit reduce_scatterv requires the scatter dimension",
        ):
            get_dp_group().reduce_scatterv(hidden, dim=0, sizes=None)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _dp_shm_group_name_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            ensure_model_parallel_initialized,
            get_dp_group,
            init_distributed_environment,
        )
        from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        distributed_init_method = f"tcp://127.0.0.1:{port}"

        vllm_config = VllmConfig()
        vllm_config.parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            data_parallel_master_ip="127.0.0.1",
            _data_parallel_master_port_list=[int(dp_port)],
        )
        os.environ["VLLM_DIST_IDENT"] = _get_cpushm_dist_ident(
            vllm_config.parallel_config,
            distributed_init_method,
        )

        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=tp_size,
                rank=tp_rank,
                distributed_init_method=distributed_init_method,
                local_rank=rank,
                backend="gloo",
            )
            ensure_model_parallel_initialized(tp_size, 1, backend="gloo")

            dp_group = get_dp_group()
            communicator = dp_group.device_communicator
            assert communicator is not None
            assert communicator._all_group_ranks_share_shm_group_name()

            dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _moe_parallel_config_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_config_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            get_tp_group,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        expected_tp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        expected_dp_ranks = list(range(tp_rank, world_size, tp_size))

        assert get_tp_group().ranks == expected_tp_ranks
        assert get_dp_group().ranks == expected_dp_ranks
        assert get_dp_group().rank_in_group == dp_rank
        assert get_ep_group().ranks == list(range(world_size))

        no_ep_parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=False,
        )
        no_ep_moe_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=1,
            dp_size_=dp_size,
            sp_size_=1,
            vllm_parallel_config=no_ep_parallel_config,
        )
        assert no_ep_moe_config.tp_size == world_size
        assert no_ep_moe_config.tp_rank == rank
        assert no_ep_moe_config.dp_size == dp_size
        assert no_ep_moe_config.dp_rank == dp_rank
        assert no_ep_moe_config.ep_size == 1
        assert no_ep_moe_config.ep_rank == 0
        assert not no_ep_moe_config.use_ep

        ep_parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=True,
        )
        ep_moe_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=1,
            dp_size_=dp_size,
            sp_size_=1,
            vllm_parallel_config=ep_parallel_config,
        )
        assert ep_moe_config.tp_size == 1
        assert ep_moe_config.tp_rank == 0
        assert ep_moe_config.dp_size == dp_size
        assert ep_moe_config.dp_rank == dp_rank
        assert ep_moe_config.ep_size == world_size
        assert ep_moe_config.ep_rank == rank
        assert ep_moe_config.use_ep

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _tp_shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Confirm the TP all-reduce path uses SHM and matches gloo."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_tp_shm_all_reduce_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        tp_group, _ = _get_tp_shm_communicator()
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(rank)

        ref = tensor.clone()
        dist.all_reduce(ref, group=tp_group.cpu_group)

        orig_all_reduce = dist.all_reduce

        def forbidden_all_reduce(*args, **kwargs):
            raise AssertionError("TP SHM all_reduce fell back to torch.distributed")

        dist.all_reduce = forbidden_all_reduce
        try:
            shm_result = tp_group.all_reduce(tensor.clone())
        finally:
            dist.all_reduce = orig_all_reduce

        torch.testing.assert_close(shm_result, ref)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _dp_shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Confirm the DP SHM all-reduce matches the gloo reference exactly."""
    try:
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_dp_shm_all_reduce_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, _ = _get_dp_shm_communicator()
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(rank)

        # Gloo reference (SUM over the DP group's cpu_group).
        ref = tensor.clone()
        dist.all_reduce(ref, group=dp_group.cpu_group)

        shm_result = dp_group.all_reduce(tensor.clone())

        torch.testing.assert_close(shm_result, ref)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _dp_metadata_shm_all_reduce_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    params,
    err_q,
):
    """Confirm CPU DP metadata sync uses SHM instead of torch.distributed."""
    try:
        os.environ.setdefault(
            "VLLM_DIST_IDENT", f"test_cpu_dp_metadata_shm_all_reduce_{port}"
        )
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        _get_dp_shm_communicator()

        from vllm.config.parallel import ParallelConfig
        from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

        dp_rank = rank // tp_size
        num_tokens_unpadded = dp_rank + 1
        num_tokens_padded = num_tokens_unpadded + 10
        expected_tokens = torch.tensor([13, 13, 13], dtype=torch.int32)

        orig_all_reduce = dist.all_reduce

        def forbidden_all_reduce(*args, **kwargs):
            raise AssertionError(
                "CPU DP metadata all_reduce fell back to torch.distributed"
            )

        parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            disable_nccl_for_dp_synchronization=True,
        )

        dist.all_reduce = forbidden_all_reduce
        try:
            should_ubatch, num_tokens_after_padding, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens_unpadded,
                    allow_microbatching=False,
                    parallel_config=parallel_config,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=1,
                )
            )
        finally:
            dist.all_reduce = orig_all_reduce

        assert not should_ubatch
        assert synced_cudagraph_mode == 1
        assert num_tokens_after_padding is not None
        assert num_tokens_after_padding.dtype == torch.int32
        torch.testing.assert_close(num_tokens_after_padding, expected_tokens)

        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


def _dp_shm_reduce_scatterv_worker(
    rank,
    world_size,
    tp_size,
    dp_size,
    port,
    dp_port,
    cases,
    err_q,
):
    """Confirm DP SHM reduce_scatterv matches all-reduce plus local slice."""
    try:
        os.environ.setdefault(
            "VLLM_DIST_IDENT", f"test_cpu_dp_shm_reduce_scatterv_{port}"
        )
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        dp_group, _ = _get_dp_shm_communicator()
        from vllm.distributed.device_communicators.cpu_communicator import (
            _CPUSHMDistributed,
        )

        dp_communicator = dp_group.device_communicator
        assert isinstance(dp_communicator.dist_module, _CPUSHMDistributed)
        shm_reduce_scatterv_calls = 0
        orig_reduce_scatterv = dp_communicator.dist_module.reduce_scatterv

        def counted_reduce_scatterv(input_, output, sizes):
            nonlocal shm_reduce_scatterv_calls
            shm_reduce_scatterv_calls += 1
            return orig_reduce_scatterv(input_, output, sizes)

        dp_communicator.dist_module.reduce_scatterv = counted_reduce_scatterv

        for sizes, explicit_sizes, dim, dtype in cases:
            total_rows = sum(sizes)
            if dim == 0:
                tensor = torch.arange(
                    total_rows * HIDDEN_SIZE,
                    dtype=dtype,
                ).reshape(total_rows, HIDDEN_SIZE)
            else:
                tensor = torch.arange(
                    HIDDEN_SIZE * total_rows,
                    dtype=dtype,
                ).reshape(HIDDEN_SIZE, total_rows)
            tensor = tensor + float((rank + 1) * 1000)

            ref = tensor.clone()
            if ref.numel():
                dist.all_reduce(ref, group=dp_group.cpu_group)

            start = sum(sizes[:rank])
            expected = ref.narrow(dim, start, sizes[rank]).contiguous()
            scatter_sizes = sizes if explicit_sizes else None
            for _ in range(2):
                result = dp_group.reduce_scatterv(
                    tensor.clone(),
                    dim=dim,
                    sizes=scatter_sizes,
                )
                torch.testing.assert_close(result, expected)

        assert shm_reduce_scatterv_calls == 2 * len(cases)
        dist.barrier()
    except Exception as err:
        _report_worker_failure(rank, err_q, err)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "sizes",
    [[2, 1], [0, 3], [4, 0], [0, 0]],
    ids=["ragged", "zero-rank", "reversed-zero", "all-zero"],
)
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
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=[3, 1, 5],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_compile_non_sp_ragged_fastpath():
    _spawn_workers(
        _compile_fastpath_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=(False, [[2, 1], [0, 3], [4, 0], [0, 0]]),
        timeout_s=60,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_compile_sp_ragged_fastpath():
    _spawn_workers(
        _compile_fastpath_worker,
        world_size=4,
        tp_size=2,
        dp_size=2,
        params=(True, [[2, 1], [0, 3], [4, 0], [0, 0]]),
        timeout_s=60,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
def test_cpu_ep_aot_cache_fallback_reuses_inductor_cache():
    with tempfile.TemporaryDirectory(prefix="vllm-cpu-ep-aot-") as cache_root:
        _spawn_workers(
            _aot_cache_worker,
            world_size=2,
            tp_size=1,
            dp_size=2,
            params=("cold", cache_root, [2, 1]),
        )
        _spawn_workers(
            _aot_cache_worker,
            world_size=2,
            tp_size=1,
            dp_size=2,
            params=("warm", cache_root, [0, 3]),
        )

        warm_results = []
        for rank in range(2):
            cold = json.loads(
                (Path(cache_root) / f"cold-rank-{rank}.json").read_text(
                    encoding="utf-8"
                )
            )
            warm = json.loads(
                (Path(cache_root) / f"warm-rank-{rank}.json").read_text(
                    encoding="utf-8"
                )
            )
            warm_results.append(warm)

            assert cold["aot_compiles"] == 1
            assert cold["aot_saved"] == 1
            assert cold["aot_loaded"] == 0
            if torch.__version__.split("+", maxsplit=1)[0].startswith("2.11."):
                assert warm["aot_loaded"] == 0
                assert warm["aot_compiles"] == 1

            if cold["standalone_saved"] > 0:
                assert warm["standalone_loaded"] == cold["standalone_saved"]
                assert warm["standalone_saved"] == 0

        assert sum(result["fxgraph_cache_hit"] for result in warm_results) > 0
        assert sum(result["fxgraph_cache_miss"] for result in warm_results) == 0


@pytest.mark.distributed
def test_cpu_dp_group_ranks_share_shm_group_name():
    _spawn_workers(
        _dp_shm_group_name_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
        distributed_init_ports=[get_open_port(), get_open_port()],
        dp_port=get_open_port(),
    )


@pytest.mark.distributed
def test_cpu_moe_tp2_dp3_parallel_config():
    _spawn_workers(
        _moe_parallel_config_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_ragged_shm_buffers_reuse_and_grow():
    _spawn_workers(
        _ragged_shm_buffers_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[[1, 0], [3, 1]],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_all_gatherv_uniform_sizes_matches_direct_shm_gather():
    _spawn_workers(
        _uniform_sizes_shm_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=[2, 2],
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
@pytest.mark.parametrize(
    ("dp_size", "size_patterns"),
    [
        (2, [[2, 1], [2, 0], [0, 0]]),
        (3, [[1, 0, 2], [0, 1, 0], [0, 0, 0]]),
    ],
    ids=["dp2", "dp3"],
)
def test_cpu_dp_all_gatherv_batched_list_fastpath(dp_size, size_patterns):
    _spawn_workers(
        _batched_all_gatherv_worker,
        world_size=dp_size,
        tp_size=1,
        dp_size=dp_size,
        params=size_patterns,
    )


@pytest.mark.distributed
def test_cpu_dp_all_gatherv_all_zero_rows():
    _spawn_workers(
        _all_zero_gatherv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
def test_cpu_dp_reduce_scatterv_implicit_sizes_require_even_split():
    _spawn_workers(
        _implicit_uneven_reduce_scatterv_worker,
        world_size=2,
        tp_size=1,
        dp_size=2,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_shm_all_reduce_matches_gloo():
    _spawn_workers(
        _dp_shm_all_reduce_worker,
        world_size=6,
        tp_size=1,
        dp_size=6,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_tp_shm_all_reduce_matches_gloo_without_fallback():
    _spawn_workers(
        _tp_shm_all_reduce_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_metadata_shm_all_reduce_without_gloo_fallback():
    _spawn_workers(
        _dp_metadata_shm_all_reduce_worker,
        world_size=6,
        tp_size=2,
        dp_size=3,
        params=None,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_CPU_SHM, reason="CPU SHM communicator required")
def test_cpu_dp_shm_reduce_scatterv_matches_all_reduce_slice():
    _spawn_workers(
        _dp_shm_reduce_scatterv_worker,
        world_size=6,
        tp_size=1,
        dp_size=6,
        params=[
            ([2, 0, 3, 1, 4, 0], True, 0, torch.float32),
            ([1, 0, 2, 1, 0, 3], True, 1, torch.float32),
            ([2, 2, 2, 2, 2, 2], False, 0, torch.float32),
            ([2, 2, 2, 2, 2, 2], False, 1, torch.float32),
            ([0, 0, 0, 0, 0, 0], True, 0, torch.float32),
            ([0, 2, 1, 3, 0, 2], True, 0, torch.bfloat16),
        ],
    )

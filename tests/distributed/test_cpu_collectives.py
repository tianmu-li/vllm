# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU SHM collective correctness tests for all supported TP sizes."""

import os
import time

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator
from vllm.platforms.cpu import CpuPlatform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

# load ops once at import time so hasattr checks below are accurate
CpuPlatform.import_kernels()

requires_cpu_shm = pytest.mark.skipif(
    not hasattr(torch.ops._C, "init_shm_manager"),
    reason="vLLM not built with CPU SHM support",
)

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_WORLD_SIZES = [2, 3, 4, 6, 8]
_BUCKETS = ["tiny", "below_thresh", "above_thresh", "head_plus_tail", "multi_tile"]

# atol + rtol * expected bounds the bf16/fp16 half-ULP rounding error at any
# magnitude produced by this test (multi_tile sums reach ~20 across 8 ranks).
_ATOL = {torch.bfloat16: 5e-3, torch.float16: 2e-3, torch.float32: 1e-5}
_RTOL = {torch.bfloat16: 5e-3, torch.float16: 2e-3, torch.float32: 1e-5}


def _numel_for_bucket(bucket: str, world_size: int, dtype: torch.dtype) -> int:
    line_elems = 64 // dtype.itemsize
    align = world_size * line_elems
    if bucket == "tiny":
        return 8
    if bucket == "below_thresh":
        # largest aligned count whose byte size is just below 128 KiB
        below = (128 * 1024 - 256) // dtype.itemsize
        return (below // align) * align
    if bucket == "above_thresh":
        above = (128 * 1024 + 256) // dtype.itemsize
        n = (above // align) * align
        if n * dtype.itemsize < 128 * 1024:
            n += align
        return n
    if bucket == "head_plus_tail":
        # ~256 KiB aligned head + 7 tail elements (exercises tail flat-AR)
        head_elems = (256 * 1024 // dtype.itemsize // align) * align
        return head_elems + 7
    if bucket == "multi_tile":
        # 8 MiB total so per-thread tiles span >1 half-buffer slot
        n = (8 * 1024 * 1024) // dtype.itemsize
        return (n // align) * align
    raise ValueError(f"unknown bucket {bucket}")


def _make_input(rank: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    # values in [0.125, ~3]; (i & 0xFF) keeps multi_tile sums bounded to ~25
    i = torch.arange(numel, dtype=torch.float32)
    data = (rank + 1) * 0.125 + (i.long() & 0xFF).to(torch.float32) * 0.0078125
    return data.to(dtype)


def _expected_allreduce(
    world_size: int, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    acc = torch.zeros(numel, dtype=torch.float32)
    for r in range(world_size):
        acc += _make_input(r, numel, dtype).float()
    return acc


def _init_comm(rank: int, world_size: int, port: str) -> CpuCommunicator:
    os.environ["VLLM_DIST_IDENT"] = f"test-tp{world_size}-{port}"
    # 4 threads: enough for the multi-threaded SHM path; keeps contention low
    torch.set_num_threads(4)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    CpuPlatform.import_kernels()
    group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    return CpuCommunicator(
        cpu_group=group,
        device=torch.device("cpu"),
        device_group=group,
        unique_name="tp:0",
    )


def _spawn(target, world_size: int, env_extra: dict, timeout: int = 120) -> None:
    port = str(get_open_port())
    processes = []
    for i in range(world_size):
        env = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            **env_extra,
        }
        p = mp.Process(target=target, args=(env,))
        processes.append(p)
        p.start()

    start_time = time.time()
    failed_processes: list[tuple[int, int]] = []
    timed_out = False

    while time.time() - start_time < timeout:
        all_done = True
        for i, p in enumerate(processes):
            if p.is_alive():
                all_done = False
            elif p.exitcode != 0:
                failed_processes.append((i, p.exitcode))
        if failed_processes or all_done:
            break
        time.sleep(0.1)
    else:
        timed_out = True

    for p in processes:
        if p.is_alive():
            p.kill()
            p.join()

    if timed_out:
        raise AssertionError(f"Distributed test timed out after {timeout}s")
    if failed_processes:
        error_msg = "Distributed test failed:\n"
        for rank, status in failed_processes:
            error_msg += f"  Rank {rank}: Exit code {status}\n"
        raise AssertionError(error_msg)


def _worker(env: dict) -> None:
    update_environment_variables(env)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    port = os.environ["MASTER_PORT"]
    bucket = os.environ["_BUCKET"]
    dtype = getattr(torch, os.environ["_DTYPE"])

    comm = _init_comm(rank, world_size, port)
    atol, rtol = _ATOL[dtype], _RTOL[dtype]
    numel = _numel_for_bucket(bucket, world_size, dtype)

    # all_reduce
    x = _make_input(rank, numel, dtype)
    comm.all_reduce(x)
    torch.testing.assert_close(
        x.float(), _expected_allreduce(world_size, numel, dtype), atol=atol, rtol=rtol
    )

    # reduce_scatter and all_gather require numel divisible by world_size
    # with each per-rank chunk a multiple of 64 bytes
    line_elems = 64 // dtype.itemsize
    align = world_size * line_elems
    rs_numel = (numel // align) * align
    if rs_numel >= align:
        chunk = rs_numel // world_size
        expected_full = _expected_allreduce(world_size, rs_numel, dtype)

        # reduce_scatter (1-D, dim=0)
        x_rs = _make_input(rank, rs_numel, dtype)
        out_rs = comm.reduce_scatter(x_rs, dim=0)
        torch.testing.assert_close(
            out_rs.float(),
            expected_full[rank * chunk : (rank + 1) * chunk],
            atol=atol,
            rtol=rtol,
        )

        # all_gather (1-D, dim=0)
        x_ag = _make_input(rank, chunk, dtype)
        out_ag = comm.all_gather(x_ag, dim=0)
        expected_ag = torch.cat(
            [_make_input(r, chunk, dtype).float() for r in range(world_size)]
        )
        torch.testing.assert_close(out_ag.float(), expected_ag, atol=atol, rtol=rtol)

        # reduce_scatter and all_gather (2-D, dim=1): movedim round-trip check.
        # uses a fixed small hidden size so this stays fast across all buckets.
        hidden = world_size * line_elems * 4
        chunk_2d = hidden // world_size
        batch = 3

        x_rs_2d = _make_input(rank, batch * hidden, dtype).reshape(batch, hidden)
        out_rs_2d = comm.reduce_scatter(x_rs_2d, dim=1)
        exp_rs_2d = _expected_allreduce(world_size, batch * hidden, dtype).reshape(
            batch, hidden
        )
        torch.testing.assert_close(
            out_rs_2d.float(),
            exp_rs_2d[:, rank * chunk_2d : (rank + 1) * chunk_2d],
            atol=atol,
            rtol=rtol,
        )

        x_ag_2d = _make_input(rank, batch * chunk_2d, dtype).reshape(batch, chunk_2d)
        out_ag_2d = comm.all_gather(x_ag_2d, dim=1)
        exp_ag_2d = torch.cat(
            [
                _make_input(r, batch * chunk_2d, dtype).reshape(batch, chunk_2d).float()
                for r in range(world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(out_ag_2d.float(), exp_ag_2d, atol=atol, rtol=rtol)

    dist.destroy_process_group()


@requires_cpu_shm
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_cpu_shm_collectives(world_size: int, bucket: str, dtype: torch.dtype) -> None:
    dtype_name = str(dtype).replace("torch.", "")
    _spawn(_worker, world_size, {"_BUCKET": bucket, "_DTYPE": dtype_name})


def _worker_unsupported_ws(env: dict) -> None:
    update_environment_variables(env)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    port = os.environ["MASTER_PORT"]

    comm = _init_comm(rank, world_size, port)

    # SHM must be disabled for unsupported world_size; Gloo fallback active
    assert not comm.supports_tensor_dict, (
        f"SHM should be disabled for world_size={world_size}"
    )

    # all_reduce must still work correctly through the Gloo fallback
    x = torch.full((256,), float(rank + 1), dtype=torch.float32)
    comm.all_reduce(x)
    expected = torch.full(
        (256,), float(sum(r + 1 for r in range(world_size))), dtype=torch.float32
    )
    torch.testing.assert_close(x, expected)

    dist.destroy_process_group()


@requires_cpu_shm
def test_cpu_shm_unsupported_world_size() -> None:
    """SHM must be silently disabled for world_size not in {2,3,4,6,8}."""
    _spawn(_worker_unsupported_ws, 5, {})

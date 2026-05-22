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
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_WORLD_SIZES = [2, 3, 4, 6, 8]
_BUCKETS = ["tiny", "below_thresh", "above_thresh", "head_plus_tail", "multi_tile"]

# Tolerances per dtype for assert_close(actual.float(), expected).
# Inputs are bounded to magnitude <=2 so these are generous but not vacuous.
_ATOL = {torch.bfloat16: 5e-3, torch.float16: 2e-3, torch.float32: 1e-5}
_RTOL = {torch.bfloat16: 5e-3, torch.float16: 2e-3, torch.float32: 1e-5}


def _numel_for_bucket(bucket: str, world_size: int, dtype: torch.dtype) -> int:
    line_elems = 64 // dtype.itemsize
    align = world_size * line_elems
    if bucket == "tiny":
        return 8
    if bucket == "below_thresh":
        # Largest aligned count whose byte size is just below 128 KiB.
        below = (128 * 1024 - 256) // dtype.itemsize
        return (below // align) * align
    if bucket == "above_thresh":
        above = (128 * 1024 + 256) // dtype.itemsize
        n = (above // align) * align
        if n * dtype.itemsize < 128 * 1024:
            n += align
        return n
    if bucket == "head_plus_tail":
        # ~256 KiB of aligned head + 7 tail elements (exercises tail flat-AR).
        head_elems = (256 * 1024 // dtype.itemsize // align) * align
        return head_elems + 7
    if bucket == "multi_tile":
        # 8 MiB total so per-thread tiles span >1 half-buffer slot.
        n = (8 * 1024 * 1024) // dtype.itemsize
        return (n // align) * align
    raise ValueError(f"unknown bucket {bucket}")


def _make_input(rank: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Values in [0.125, ~2.1]; using (i & 0xFF) bounds multi_tile magnitudes.
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


def _worker(env: dict) -> None:
    update_environment_variables(env)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    port = os.environ["MASTER_PORT"]
    bucket = os.environ["_BUCKET"]
    dtype = getattr(torch, os.environ["_DTYPE"])

    os.environ["VLLM_DIST_IDENT"] = f"test-tp{world_size}-{port}"
    # 4 threads: enough for multi-threaded SHM path; keeps contention low.
    torch.set_num_threads(4)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    from vllm.platforms.cpu import CpuPlatform

    CpuPlatform.import_kernels()

    group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    comm = CpuCommunicator(
        cpu_group=group,
        device=torch.device("cpu"),
        device_group=group,
        unique_name="tp:0",
    )

    assert comm.supports_tensor_dict, (
        "SHM communicator not active — vLLM must be built with CPU SHM support"
    )

    numel = _numel_for_bucket(bucket, world_size, dtype)
    atol = _ATOL[dtype]
    rtol = _RTOL[dtype]

    # all_reduce
    x = _make_input(rank, numel, dtype)
    comm.all_reduce(x)
    expected = _expected_allreduce(world_size, numel, dtype)
    torch.testing.assert_close(x.float(), expected, atol=atol, rtol=rtol)

    # reduce_scatter + all_gather only make sense when numel is divisible by ws.
    line_elems = 64 // dtype.itemsize
    align = world_size * line_elems
    rs_numel = (numel // align) * align
    if rs_numel >= align:
        # reduce_scatter
        x_rs = _make_input(rank, rs_numel, dtype)
        out_rs = comm.reduce_scatter(x_rs, dim=0)
        chunk = rs_numel // world_size
        expected_full = _expected_allreduce(world_size, rs_numel, dtype)
        expected_chunk = expected_full[rank * chunk : (rank + 1) * chunk]
        torch.testing.assert_close(out_rs.float(), expected_chunk, atol=atol, rtol=rtol)

        # all_gather
        x_ag = _make_input(rank, chunk, dtype)
        out_ag = comm.all_gather(x_ag, dim=0)
        expected_ag = torch.cat(
            [_make_input(r, chunk, dtype).float() for r in range(world_size)]
        )
        torch.testing.assert_close(out_ag.float(), expected_ag, atol=atol, rtol=rtol)

    dist.destroy_process_group()


def _spawn(
    world_size: int, bucket: str, dtype: torch.dtype, timeout: int = 120
) -> None:
    port = str(get_open_port())
    dtype_name = str(dtype).replace("torch.", "")

    processes = []
    for i in range(world_size):
        env = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "_BUCKET": bucket,
            "_DTYPE": dtype_name,
        }
        p = mp.Process(target=_worker, args=(env,))
        processes.append(p)
        p.start()

    start_time = time.time()
    failed_processes = []
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
        raise AssertionError(
            f"Distributed test timed out after {timeout}s (processes were still alive)"
        )
    if failed_processes:
        error_msg = "Distributed test failed:\n"
        for rank, status in failed_processes:
            error_msg += f"  Rank {rank}: Exit code {status}\n"
        raise AssertionError(error_msg)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_cpu_shm_collectives(world_size: int, bucket: str, dtype: torch.dtype) -> None:
    _spawn(world_size, bucket, dtype)

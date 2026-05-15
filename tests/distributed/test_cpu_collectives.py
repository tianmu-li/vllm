# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU SHM collective correctness tests for all supported TP sizes."""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["VLLM_DIST_IDENT"] = f"test-tp{world_size}-{port}"
    # Limit OMP threads so all workers fit on one NUMA node without contention.
    torch.set_num_threads(1)
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
        "SHM communicator not active — vLLM must be built with CPU SHM support "
        "(VLLM_USE_PRECOMPILED=0, VLLM_TARGET_DEVICE=cpu)"
    )

    ws = world_size

    # all_reduce — two sizes that bracket RSAG_BYTE_THRESHOLD (32 KiB):
    # ws*256 stays below the threshold (flat AR path);
    # ws*16384 (>=32 KiB for ws>=2) exercises the RS+AG path.
    # Both are chosen to be divisible by ws so each rank chunk is
    # a multiple of 64 bytes (required by memcpy_to_shm).
    for numel in (ws * 256, ws * 16384):
        x = torch.ones(numel, dtype=torch.bfloat16)
        comm.all_reduce(x)
        assert torch.all(x == ws), f"all_reduce fail numel={numel}"

    # reduce_scatter — numel divisible by world_size
    numel = ws * 512
    x = torch.ones(numel, dtype=torch.bfloat16)
    out = comm.reduce_scatter(x, dim=0)
    expected = torch.full((512,), float(ws), dtype=torch.bfloat16)
    assert torch.allclose(out, expected), "reduce_scatter fail"

    # all_gather
    x = torch.full((512,), float(rank), dtype=torch.bfloat16)
    out = comm.all_gather(x, dim=0)
    expected = torch.cat(
        [torch.full((512,), float(r), dtype=torch.bfloat16) for r in range(ws)]
    )
    assert torch.allclose(out, expected), "all_gather fail"

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 8])
def test_cpu_shm_collectives(world_size: int) -> None:
    port = _find_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )

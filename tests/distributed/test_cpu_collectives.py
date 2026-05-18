# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU SHM collective correctness tests for all supported TP sizes."""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_WORLD_SIZES = [2, 3, 4, 6, 8]
_BUCKETS = ["tiny", "below_thresh", "above_thresh", "head_plus_tail", "multi_tile"]

# Tolerances per dtype for assert_close(actual.float(), expected).
# Inputs are bounded to magnitude <=2 so these are generous but not vacuous.
_ATOL = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 1e-5}
_RTOL = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 1e-5}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _numel_for_bucket(bucket: str, world_size: int, dtype: torch.dtype) -> int:
    line_elems = 64 // dtype.itemsize
    align = world_size * line_elems
    if bucket == "tiny":
        return 8
    if bucket == "below_thresh":
        # Largest aligned count whose byte size is just below 32 KiB.
        below = (32 * 1024 - 256) // dtype.itemsize
        return (below // align) * align
    if bucket == "above_thresh":
        above = (32 * 1024 + 256) // dtype.itemsize
        n = (above // align) * align
        if n * dtype.itemsize < 32 * 1024:
            n += align
        return n
    if bucket == "head_plus_tail":
        # ~64 KiB of aligned head + 7 tail elements (exercises tail flat-AR).
        head_elems = (64 * 1024 // dtype.itemsize // align) * align
        return head_elems + 7
    if bucket == "multi_tile":
        # 8 MiB total so per-thread tiles span >1 half-buffer slot.
        n = (8 * 1024 * 1024) // dtype.itemsize
        return (n // align) * align
    raise ValueError(f"unknown bucket {bucket}")


def _make_input(rank: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Values in [0.125, ~2.1]; using (i & 0xFF) bounds multi_tile magnitudes.
    data = torch.tensor(
        [(rank + 1) * 0.125 + (i & 0xFF) * 0.0078125 for i in range(numel)],
        dtype=torch.float32,
    )
    return data.to(dtype)


def _expected_allreduce(
    world_size: int, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    acc = torch.zeros(numel, dtype=torch.float32)
    for r in range(world_size):
        acc += _make_input(r, numel, dtype).float()
    return acc


def _worker(
    rank: int,
    world_size: int,
    port: int,
    bucket: str,
    dtype: torch.dtype,
    threshold_override: str | None,
) -> None:
    if threshold_override is not None:
        os.environ["VLLM_CPU_RSAG_THRESHOLD_BYTES"] = threshold_override

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


def _spawn(world_size, port, bucket, dtype, threshold_override=None):
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, port, bucket, dtype, threshold_override),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_cpu_shm_collectives(world_size: int, bucket: str, dtype: torch.dtype) -> None:
    port = _find_free_port()
    _spawn(world_size, port, bucket, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bucket", ["tiny", "above_thresh"])
def test_force_rsag_path(bucket: str, dtype: torch.dtype) -> None:
    """VLLM_CPU_RSAG_THRESHOLD_BYTES=0 forces RS+AG even on tiny inputs."""
    port = _find_free_port()
    _spawn(4, port, bucket, dtype, threshold_override="0")


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bucket", ["tiny", "above_thresh"])
def test_force_flat_ar_path(bucket: str, dtype: torch.dtype) -> None:
    """VLLM_CPU_RSAG_THRESHOLD_BYTES=999999999 forces flat AR on all inputs."""
    port = _find_free_port()
    _spawn(4, port, bucket, dtype, threshold_override="999999999")

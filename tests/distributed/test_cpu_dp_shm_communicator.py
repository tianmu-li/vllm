# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import traceback

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)


def _spawn_workers(worker_fn, world_size, params):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    err_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(target=worker_fn, args=(rank, world_size, params, err_q))
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


def _dp_shm_group_name_worker(rank, world_size, params, err_q):
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            ensure_model_parallel_initialized,
            get_dp_group,
            init_distributed_environment,
        )
        from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident

        init_ports = params["init_ports"]
        dp_port = params["dp_port"]
        distributed_init_method = f"tcp://127.0.0.1:{init_ports[rank]}"

        vllm_config = VllmConfig()
        vllm_config.parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=world_size,
            data_parallel_rank=rank,
            data_parallel_master_ip="127.0.0.1",
            _data_parallel_master_port_list=[dp_port],
        )
        os.environ["VLLM_DIST_IDENT"] = _get_cpushm_dist_ident(
            vllm_config.parallel_config,
            distributed_init_method,
        )

        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=distributed_init_method,
                local_rank=rank,
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1, backend="gloo")

            dp_group = get_dp_group()
            device_communicator = dp_group.device_communicator
            assert device_communicator is not None
            assert device_communicator._all_group_ranks_share_shm_group_name()

            dist.barrier()
    except Exception as err:
        err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
        raise SystemExit(1) from err


@pytest.mark.distributed
def test_cpu_dp_group_ranks_share_shm_group_name():
    _spawn_workers(
        _dp_shm_group_name_worker,
        world_size=2,
        params={
            "init_ports": [get_open_port(), get_open_port()],
            "dp_port": get_open_port(),
        },
    )

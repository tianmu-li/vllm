# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.platforms import current_platform
from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)


def _make_parallel_config(dp_port: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_port_list=[dp_port],
    )


def test_get_cpushm_dist_ident_uses_dp_rendezvous_for_single_node_dp():
    ident_rank0 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_rank1 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11002",
    )

    assert ident_rank0 == ident_rank1 == "127.0.0.1:26001"


def test_get_cpushm_dist_ident_differs_for_different_dp_rendezvous():
    ident_a = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_b = _get_cpushm_dist_ident(
        _make_parallel_config(26002),
        "tcp://127.0.0.1:11002",
    )

    assert ident_a != ident_b

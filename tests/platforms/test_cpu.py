# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.cpu import CpuPlatform


@pytest.mark.parametrize(
    ("capabilities", "expected_extension"),
    [
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
                "amx_tile": True,
                "amx_bf16": True,
            },
            "_C",
        ),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            "_C_AVX512_BF16_VNNI",
        ),
        ({"avx512_f": True}, "_C_AVX512"),
        ({}, "_C_AVX2"),
    ],
)
def test_x86_cpu_extension_selection(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: dict[str, bool],
    expected_extension: str,
) -> None:
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: capabilities)

    assert CpuPlatform._get_x86_cpu_extension_name() == expected_extension

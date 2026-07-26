# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.platforms.cpu import (
    CpuPlatform,
    is_avx512_bf16_vnni_supported,
)

SGL_OPS = {
    "_C::causal_conv1d_fwd_cpu",
    "_C::causal_conv1d_update_cpu",
    "_C::causal_conv1d_weight_pack",
    "_C::convert_scale_packed",
    "_C::convert_weight_packed",
    "_C::convert_weight_packed_scale_zp",
    "_C::fp8_scaled_mm_cpu",
    "_C::fused_experts_cpu",
    "_C::int4_scaled_mm_cpu",
    "_C::int8_scaled_mm_with_quant",
    "_C::weight_packed_linear",
}


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
            "_C_AVX512",
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


def _get_registered_ops(extension_name: str) -> set[str]:
    extension_dir = Path(__file__).resolve().parents[2] / "vllm"
    script = """
import json
from pathlib import Path
import sys

import torch

extension_dir = Path(sys.argv[1])
extension_name = sys.argv[2]
paths = list(extension_dir.glob(f"{extension_name}.*.so"))
if len(paths) != 1:
    raise RuntimeError(f"Expected one {extension_name} extension, found {paths}")
torch.ops.load_library(str(paths[0]))
print(json.dumps(torch._C._dispatch_get_all_op_names()))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(extension_dir), extension_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(result.stdout))


@pytest.mark.skipif(
    not torch.cpu.get_capabilities().get("avx512_f", False),
    reason="_C_AVX512 requires AVX512F",
)
def test_avx512_sgl_op_registration_matches_runtime_capabilities() -> None:
    registered_ops = _get_registered_ops("_C_AVX512")

    if is_avx512_bf16_vnni_supported():
        assert SGL_OPS.issubset(registered_ops)
    else:
        assert SGL_OPS.isdisjoint(registered_ops)


def test_avx2_does_not_register_sgl_ops() -> None:
    registered_ops = _get_registered_ops("_C_AVX2")

    assert SGL_OPS.isdisjoint(registered_ops)

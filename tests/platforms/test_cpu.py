# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm._custom_ops import CPUQuantMethod
from vllm.platforms import CpuArchEnum
from vllm.platforms.cpu import (
    CpuPlatform,
    _is_cpu_quant_kernel_supported,
    _is_fused_experts_cpu_method_supported,
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


@pytest.mark.parametrize(
    ("capabilities", "expected_bf16_vnni", "expected_amx"),
    [
        ({"avx512_f": True}, False, False),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            True,
            False,
        ),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
                "amx_tile": True,
                "amx_bf16": True,
            },
            True,
            True,
        ),
    ],
)
def test_cpu_quant_capability_requirements(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: dict[str, bool],
    expected_bf16_vnni: bool,
    expected_amx: bool,
) -> None:
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: capabilities)

    assert is_avx512_bf16_vnni_supported() is expected_bf16_vnni
    assert _is_cpu_quant_kernel_supported(require_amx=False) is expected_bf16_vnni
    assert _is_cpu_quant_kernel_supported() is expected_amx


@pytest.mark.parametrize(
    "method",
    [
        CPUQuantMethod.FP8_W8A16,
        CPUQuantMethod.MXFP4,
        CPUQuantMethod.INT8_W8A8,
        CPUQuantMethod.INT4_W4A8,
    ],
)
def test_fused_experts_verified_methods_allow_bf16_vnni(
    monkeypatch: pytest.MonkeyPatch,
    method: CPUQuantMethod,
) -> None:
    monkeypatch.setattr(
        torch.cpu,
        "get_capabilities",
        lambda: {
            "avx512_f": True,
            "avx512_bf16": True,
            "avx512_vnni": True,
        },
    )

    assert _is_fused_experts_cpu_method_supported(method)
    assert not _is_fused_experts_cpu_method_supported(99)


def test_fused_experts_unknown_method_requires_amx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cpu,
        "get_capabilities",
        lambda: {
            "avx512_f": True,
            "avx512_bf16": True,
            "avx512_vnni": True,
            "amx_tile": True,
            "amx_bf16": True,
        },
    )

    assert _is_fused_experts_cpu_method_supported(99)


@pytest.mark.parametrize(
    "expert_name",
    [
        "CPUExpertsFp8",
        "CPUExpertsMxfp4",
        "CPUExpertsInt8",
        "CPUExpertsInt4",
    ],
)
@pytest.mark.parametrize(
    ("capabilities", "has_native_op", "expected"),
    [
        ({"avx512_f": True}, True, False),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            False,
            False,
        ),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            True,
            True,
        ),
    ],
)
def test_x86_quantized_experts_require_isa_and_native_op(
    monkeypatch: pytest.MonkeyPatch,
    expert_name: str,
    capabilities: dict[str, bool],
    has_native_op: bool,
    expected: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.experts import cpu_moe

    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: capabilities)
    monkeypatch.setattr(
        cpu_moe,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True,
            get_cpu_architecture=lambda: CpuArchEnum.X86,
        ),
    )
    monkeypatch.setattr(cpu_moe, "_has_fused_experts_cpu_op", lambda: has_native_op)

    assert getattr(cpu_moe, expert_name)._supports_current_device() is expected


@pytest.mark.parametrize(
    ("capabilities", "has_native_op", "expected"),
    [
        ({"avx512_f": True}, True, False),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            False,
            False,
        ),
        (
            {
                "avx512_f": True,
                "avx512_bf16": True,
                "avx512_vnni": True,
            },
            True,
            True,
        ),
    ],
)
def test_cpu_fp8_scaled_mm_requires_isa_and_native_op(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: dict[str, bool],
    has_native_op: bool,
    expected: bool,
) -> None:
    from vllm.model_executor.kernels.linear.scaled_mm import cpu as cpu_scaled_mm

    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: capabilities)
    monkeypatch.setattr(
        cpu_scaled_mm,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: True),
    )
    monkeypatch.setattr(
        cpu_scaled_mm.ops,
        "_supports_cpu_fp8_w8a16",
        has_native_op,
    )

    supported, _ = cpu_scaled_mm.CPUFp8BlockScaledMMKernel.is_supported()

    assert supported is expected


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

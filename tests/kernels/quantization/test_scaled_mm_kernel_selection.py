# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only).

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
import os
import subprocess
import sys
import textwrap
from abc import ABC
from unittest.mock import patch

import pytest
import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    AiterInt8ScaledMMLinearKernel,
    CPUInt8ScaledMMLinearKernel,
    HummingFP8ScaledMMLinearKernel,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    TritonInt8ScaledMMLinearKernel,
    _get_linear_backend,
    _resolve_backend_kernels,
    init_fp8_linear_kernel,
    init_int8_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_auto_linear_backend_is_unchanged_without_overrides():
    kernels: list[type] = [object]
    quantizations = (
        "fp8_block_w8a8",
        "fp8_w8a8",
        "int8_w8a8",
        "mixed_precision",
        "mxfp8",
        "mxfp4",
        "mxfp6",
        "w8a16_fp8",
        "nvfp4_w4a4",
        "nvfp4_w4a16",
    )
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))

    with set_current_vllm_config(config):
        for quantization in quantizations:
            assert _get_linear_backend(quantization=quantization) == "auto"
            assert (
                _resolve_backend_kernels(kernels, "test", quantization=quantization)
                is kernels
            )


def test_linear_backend_override_is_quantization_specific():
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"nvfp4_w4a16": "humming"},
        )
    )

    with set_current_vllm_config(config):
        assert _get_linear_backend(quantization="nvfp4_w4a16") == "humming"
        assert _get_linear_backend(quantization="nvfp4_w4a4") == "cutlass"
        assert _get_linear_backend(quantization="fp8_w8a8") == "cutlass"


@patch.object(HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_fp8_linear_backend_override(platform_mock, _):
    platform_mock._enum = PlatformEnum.CUDA
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"fp8_w8a8": "humming"},
        )
    )

    with set_current_vllm_config(config):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(128, 128),
        )

    assert type(kernel) is HummingFP8ScaledMMLinearKernel


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_cpu_kernel_implements_is_supported():
    """Test that CPUInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(CPUInt8ScaledMMLinearKernel, "is_supported"), (
        "CPUInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        CPUInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(CPUInt8ScaledMMLinearKernel.is_supported), (
        "CPUInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    result, reason = CPUInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_aiter_kernel_implements_is_supported():
    """Test that AiterInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(AiterInt8ScaledMMLinearKernel, "is_supported"), (
        "AiterInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        AiterInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(AiterInt8ScaledMMLinearKernel.is_supported), (
        "AiterInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    # (will return False on CPU, which is expected)
    result, reason = AiterInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"
    # On CPU, it should return False with a reason about requiring ROCm
    # This validates the method works correctly even on non-ROCm platforms


@pytest.mark.parametrize(
    ("is_static_input_scheme", "is_channelwise", "is_fused_module", "expected"),
    [
        # per-tensor activations x per-tensor weights
        (True, False, False, False),
        # per-tensor activations x per-channel weights
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, False),
        # per-token activations x per-tensor weights
        (False, False, False, False),
        # per-token activations x per-channel weights
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ],
)
def test_aiter_int8_kernel_can_implement_scale_layouts(
    is_static_input_scheme, is_channelwise, is_fused_module, expected
):
    config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=is_static_input_scheme,
        is_channelwise=is_channelwise,
        input_symmetric=True,
        is_fused_module=is_fused_module,
    )
    can_impl, _ = AiterInt8ScaledMMLinearKernel.can_implement(config)
    assert can_impl is expected


def _make_compressed_tensors_per_tensor_int8_scheme():
    from compressed_tensors.quantization import QuantizationStrategy

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Int8,
    )

    return CompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.TENSOR,
        is_static_input_scheme=False,
        input_symmetric=True,
    )


def _make_quark_per_tensor_int8_scheme():
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticTensorSym,
    )

    return QuarkW8A8Int8(kInt8StaticTensorSym, kInt8DynamicTokenSym)


@pytest.mark.parametrize(
    "make_scheme",
    [
        _make_compressed_tensors_per_tensor_int8_scheme,
        _make_quark_per_tensor_int8_scheme,
    ],
    ids=["compressed_tensors", "quark"],
)
@pytest.mark.parametrize("output_partition_sizes", [[16, 32], [48]], ids=str)
@patch.object(TritonInt8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch.object(AiterInt8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch(
    "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1
)
@patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_aiter_int8_per_tensor_weight_selection_and_loading(
    platform_mock,
    _tp_rank,
    _tp_size,
    _aiter_supported,
    _triton_supported,
    output_partition_sizes,
    make_scheme,
):
    """Per-tensor weights x per-token activations through the scheme's own
    create_weights/process_weights_after_loading: AITER is selected only for
    fused modules, whose shard scales must become per-channel after loading."""
    platform_mock._enum = PlatformEnum.ROCM
    platform_mock.get_device_capability.return_value = None
    scheme = make_scheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=output_partition_sizes,
        input_size_per_partition=32,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    is_fused = len(output_partition_sizes) > 1
    expected_kernel = (
        AiterInt8ScaledMMLinearKernel if is_fused else TritonInt8ScaledMMLinearKernel
    )
    assert type(scheme.kernel) is expected_kernel
    if not is_fused:
        return

    n = sum(output_partition_sizes)
    shard_scales = torch.tensor([0.5, 2.0])
    layer.weight.data.copy_(torch.ones(n, 32, dtype=torch.int8))
    layer.weight_scale.data.copy_(shard_scales)
    scheme.process_weights_after_loading(layer)

    # AiterInt8ScaledMMLinearKernel.apply_weights expects [K, N] weights and
    # one scale per output channel.
    assert layer.weight.shape == (32, n)
    assert layer.weight_scale.numel() == layer.weight.shape[1]
    expected = torch.repeat_interleave(
        shard_scales, torch.tensor(output_partition_sizes)
    )
    assert torch.equal(layer.weight_scale.flatten(), expected)


def test_cpu_kernel_accepts_all_configs():
    """Test that CPUInt8ScaledMMLinearKernel accepts all config combinations."""
    configs = [
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=False,
            is_static_input_scheme=True,
            input_symmetric=True,
        ),
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=False,
        ),
    ]

    for config in configs:
        can_impl, reason = CPUInt8ScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUInt8ScaledMMLinearKernel should accept config {config}: {reason}"
        )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
def test_cpu_sgl_int8_linear_accepts_3d_input():
    """The SGL int8 path must accept B x T x D activations (Whisper encoder)."""
    N, K = 64, 128
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(-127, 128, (N, K), dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(N, 1) * 0.01 + 1e-3, requires_grad=False
    )
    layer.bias = torch.nn.Parameter(
        torch.randn(N, dtype=torch.bfloat16), requires_grad=False
    )
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    kernel = CPUInt8ScaledMMLinearKernel(
        config,
        ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"),
    )
    kernel.process_weights_for_sgl(layer)
    kernel.linear_method = kernel._apply_weights_sgl

    x = torch.randn(2, 3, K, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, layer.bias)
    ref = kernel.apply_weights(layer, x.reshape(-1, K), layer.bias)

    assert out.shape == (2, 3, N)
    torch.testing.assert_close(out, ref.reshape(2, 3, N), rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
def test_cpu_sgl_int8_schedulers_preserve_scaled_mm_result(tmp_path):
    """SGL scheduler modes preserve bias-fused dense INT8 GEMM results."""
    script = textwrap.dedent(
        f"""
        import os
        import torch
        import vllm._custom_ops

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        outputs = []
        for m, n, has_bias in ((128, 128, True), (33, 96, False)):
            x = torch.randn(m, 64).bfloat16()
            weight = torch.randint(-30, 31, (n, 64), dtype=torch.int8)
            weight_scale = torch.rand(n, dtype=torch.float32) * 0.01
            bias = torch.rand(n, dtype=torch.float32) if has_bias else None
            result = torch.ops._C.int8_scaled_mm_with_quant(
                x, weight, weight_scale, bias, x.dtype, False
            )
            input_scale = x.float().abs().amax(dim=-1).clamp_min(1e-7) / 127
            qx = (x.float() / input_scale[:, None]).round().to(torch.int32)
            reference = qx.float() @ weight.float().T
            reference *= input_scale[:, None] * weight_scale[None, :]
            if has_bias:
                reference += bias[None, :]
            torch.testing.assert_close(
                result.float(), reference, atol=0.2, rtol=0.02
            )
            outputs.append(result)
        torch.save(tuple(outputs), {str(tmp_path / "result.pt")!r})
        """
    )
    outputs = {}
    configurations = (
        (6, None, None),
        (6, "", "2,1,3"),
        (6, "legacy", "2,1,3"),
        (6, "round_robin", "2,1,3"),
        (6, "spread_dense", "2,1,3"),
        (6, "tile_dense", "2,1,3"),
        (6, "spread_moe", "2,1,3"),
        (6, "spread_all", "2,1,3"),
        (6, "tile_all", "2,1,3"),
        (7, None, None),
        (7, "tile_dense", "2,2,3"),
    )
    for threads, scheduler, groups in configurations:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_THREAD_LIMIT"] = str(threads)
        env.pop("VLLM_CPU_SGL_SCHEDULER", None)
        env.pop("VLLM_CPU_SGL_THREAD_GROUPS", None)
        if scheduler is not None:
            env["VLLM_CPU_SGL_SCHEDULER"] = scheduler
        if groups is not None:
            env["VLLM_CPU_SGL_THREAD_GROUPS"] = groups
        subprocess.run([sys.executable, "-c", script], check=True, env=env)
        outputs[threads, scheduler or "unset", groups is not None] = torch.load(
            tmp_path / "result.pt", weights_only=True
        )

    for threads in (6, 7):
        baseline = outputs[threads, "unset", False]
        for case in configurations:
            case_threads, scheduler, groups = case
            if case_threads == threads:
                torch.testing.assert_close(
                    baseline,
                    outputs[threads, scheduler or "unset", groups is not None],
                    rtol=0,
                    atol=0,
                )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
@pytest.mark.parametrize(
    ("scheduler", "groups", "thread_limit", "error"),
    [
        (None, "2,2", None, b"sums to"),
        (
            "round_robin",
            "2,1,3",
            "4",
            b"requires fixed, non-nested OpenMP teams",
        ),
        ("not_a_scheduler", "2,1,3", None, b"VLLM_CPU_SGL_SCHEDULER"),
        (
            "spread_dense",
            "",
            None,
            b"requires VLLM_CPU_SGL_THREAD_GROUPS",
        ),
        (
            "tile_dense",
            "2,1,3",
            "4",
            b"requires fixed, non-nested OpenMP teams",
        ),
    ],
)
def test_cpu_sgl_int8_scheduler_rejects_invalid_configuration(
    scheduler, groups, thread_limit, error
):
    """Reject invalid modes, missing groups, mismatched teams, and limits."""
    script = textwrap.dedent(
        """
import torch
import vllm._custom_ops

torch.set_num_threads(6)
torch.set_num_interop_threads(1)
x = torch.randn(128, 64).bfloat16()
weight = torch.randint(-30, 31, (128, 64), dtype=torch.int8)
scale = torch.ones(128)
torch.ops._C.int8_scaled_mm_with_quant(x, weight, scale, None, x.dtype, False)
        """
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "6"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_THREAD_LIMIT"] = "6"
    env.pop("VLLM_CPU_SGL_SCHEDULER", None)
    env["VLLM_CPU_SGL_THREAD_GROUPS"] = groups
    if scheduler is not None:
        env["VLLM_CPU_SGL_SCHEDULER"] = scheduler
    if thread_limit is None:
        env.pop("OMP_THREAD_LIMIT", None)
    else:
        env["OMP_THREAD_LIMIT"] = thread_limit
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert error in result.stderr


class OOTInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_register_oot_linear_kernel(platform_mock):
    """Test that the linear kernel registration works correctly."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTInt8ScaledMMLinearKernel, PlatformEnum.OOT, "int8")

    kernel = init_int8_linear_kernel(True, True, True, "module")

    assert isinstance(kernel, OOTInt8ScaledMMLinearKernel), (
        "init_int8_linear_kernel should return an instance of the registered kernel"
    )


@pytest.mark.parametrize(
    "possible_kernels", [_POSSIBLE_KERNELS, _POSSIBLE_NVFP4_KERNELS]
)
def test_humming_priority_only_reorders_humming_and_marlin(possible_kernels):
    """Preferring Humming must not demote Marlin below unrelated kernels."""
    kernels = possible_kernels[PlatformEnum.CUDA]
    original = list(kernels)

    assert prioritize_humming(kernels, compute_capability=80) == original

    names = [k.__name__ for k in prioritize_humming(kernels, compute_capability=90)]
    humming = next(i for i, name in enumerate(names) if "Humming" in name)
    assert "Marlin" in names[humming + 1]
    del names[humming]
    assert names == [k.__name__ for k in original if "Humming" not in k.__name__]
    assert kernels == original

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
def test_cpu_sgl_int8_dense_schedulers_preserve_scaled_mm_result(tmp_path):
    """Dense SGL INT8 scheduler modes preserve exact GEMM results."""
    script = textwrap.dedent(
        """
        import os
        import sys
        import torch
        import vllm._custom_ops
        from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
            CPUInt8ScaledMMLinearKernel,
            Int8ScaledMMLinearLayerConfig,
        )

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        case_group = sys.argv[2]
        cases = {
            "base": (
                (127, 320, False, False),
                (128, 320, True, True),
                (512, 1024, False, False),
                (1, 32, False, False),
                (33, 96, False, True),
                (1, 32, True, False),
                (128, 320, False, False),
                (1, 6144, False, True),
                (128, 32, False, True),
            ),
            "flat32": (
                (127, 320, False, False),
                (128, 320, True, True),
                (512, 1024, False, False),
            ),
            "flat32_uniform": ((512, 1024, False, False),),
            "flat128x64": (
                (385, 3072, False, True),
                (385, 3008, False, False),
            ),
            "flat128x128": (
                (385, 6144, False, True),
                (385, 5952, False, False),
            ),
            "flat128x128_eligible": ((385, 6144, False, True),),
            "flat128x128_tall": (
                (3105, 1056, False, False),
                (3071, 1056, False, False),
            ),
            "flat_auto": (
                (127, 320, False, False),  # legacy below the flat-mode threshold
                (768, 6144, False, False),  # aligned, ungrouped wide workload
                (768, 6176, False, False),  # N tail keeps the 32x32 base
                (769, 6144, False, False),  # M tail keeps the 32x32 base
                (768, 2048, False, False),  # grouped shape keeps existing groups
            ),
            "flat_auto_wide": ((768, 6144, False, False),),
        }[case_group]
        outputs = []
        for m, n, has_bias, is_vnni in cases:
            torch.manual_seed(m + n)
            x = torch.randn(m, 64).bfloat16()
            weight = torch.randint(-30, 31, (n, 64), dtype=torch.int8)
            reference_weight = weight.float()
            if is_vnni:
                weight = torch.ops._C.convert_weight_packed(weight)
            weight_scale = torch.rand(n, dtype=torch.float32) * 0.01
            bias = torch.rand(n, dtype=torch.float32) if has_bias else None
            result = torch.ops._C.int8_scaled_mm_with_quant(
                x, weight, weight_scale, bias, x.dtype, is_vnni
            )
            input_scale = x.float().abs().amax(dim=-1).clamp_min(1e-7) / 127
            qx = (x.float() / input_scale[:, None]).round().to(torch.int32)
            reference = qx.float() @ reference_weight.T
            reference *= input_scale[:, None] * weight_scale[None, :]
            if has_bias:
                reference += bias[None, :]
            torch.testing.assert_close(
                result.float(), reference, atol=0.2, rtol=0.02
            )
            outputs.append(result)
        if case_group == "base":
            # Unsupported column tails use the existing oneDNN layer fallback.
            torch.set_default_dtype(torch.bfloat16)
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(
                torch.randint(-30, 31, (33, 64), dtype=torch.int8),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.rand(33, 1, dtype=torch.float32) * 0.01,
                requires_grad=False,
            )
            layer.bias = torch.nn.Parameter(torch.randn(33), requires_grad=False)
            layer.logical_widths = [33]
            layer.input_scale = None
            layer.input_zero_point = None
            kernel = CPUInt8ScaledMMLinearKernel(
                Int8ScaledMMLinearLayerConfig(
                    is_channelwise=True,
                    is_static_input_scheme=False,
                    input_symmetric=True,
                ),
                (
                    "weight",
                    "weight_scale",
                    "input_scale",
                    "input_zero_point",
                    "azp_adj",
                ),
            )
            kernel.process_weights_after_loading(layer)
            outputs.append(
                kernel.apply_weights(layer, torch.randn(2, 3, 64), layer.bias)
            )
        torch.save(tuple(outputs), sys.argv[1])
        """
    )

    def run(
        scheduler,
        requested_threads,
        thread_limit,
        output_path,
        case_group="base",
        max_active_levels=None,
    ):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(requested_threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_THREAD_LIMIT"] = str(thread_limit)
        if max_active_levels is not None:
            env["OMP_MAX_ACTIVE_LEVELS"] = str(max_active_levels)
        env["VLLM_CPU_SGL_SCHEDULER"] = scheduler
        subprocess.run(
            [sys.executable, "-c", script, str(output_path), case_group],
            check=True,
            env=env,
        )
        return torch.load(output_path, weights_only=True)

    legacy_outputs = {}
    for actual_threads in (2, 95, 96, 97):
        # Set the requested team to the actual team size for the legacy
        # reference. A constrained logical96_dense run below uses a larger
        # request, but is compared against this correctly sized reference.
        legacy_outputs[actual_threads] = run(
            "legacy",
            actual_threads,
            actual_threads,
            tmp_path / f"legacy-{actual_threads}.pt",
        )

    for actual_threads in (95, 96, 97):
        logical_outputs = run(
            "logical96_dense",
            actual_threads,
            actual_threads,
            tmp_path / f"logical96-dense-{actual_threads}.pt",
        )
        for legacy, logical in zip(
            legacy_outputs[actual_threads], logical_outputs, strict=True
        ):
            torch.testing.assert_close(legacy, logical, rtol=0, atol=0)

    constrained_outputs = run(
        "logical96_dense",
        requested_threads=128,
        thread_limit=2,
        output_path=tmp_path / "logical96-dense-128-2.pt",
    )
    for legacy, logical in zip(legacy_outputs[2], constrained_outputs, strict=True):
        torch.testing.assert_close(legacy, logical, rtol=0, atol=0)

    flat32_outputs = run(
        "flat_32x32",
        95,
        95,
        tmp_path / "flat32.pt",
        "flat32",
    )
    for legacy, flat in zip(legacy_outputs[95][:3], flat32_outputs, strict=True):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    flat32_constrained_outputs = run(
        "flat_32x32",
        requested_threads=128,
        thread_limit=2,
        output_path=tmp_path / "flat32-128-2.pt",
        case_group="flat32_uniform",
    )
    for legacy, flat in zip(
        legacy_outputs[2][2:3], flat32_constrained_outputs, strict=True
    ):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    flat32_mismatched_outputs = run(
        "flat_32x32",
        requested_threads=2,
        thread_limit=1,
        output_path=tmp_path / "flat32-2-1.pt",
        case_group="flat32_uniform",
    )
    for legacy, flat in zip(
        legacy_outputs[2][2:3], flat32_mismatched_outputs, strict=True
    ):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    flat32_serialized_outputs = run(
        "flat_32x32",
        requested_threads=2,
        thread_limit=2,
        output_path=tmp_path / "flat32-2-serialized.pt",
        case_group="flat32_uniform",
        max_active_levels=0,
    )
    for legacy, flat in zip(
        legacy_outputs[2][2:3], flat32_serialized_outputs, strict=True
    ):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    for scheduler, case_group, actual_threads in (
        ("flat_128x64", "flat128x64", 97),
        ("flat_128x128", "flat128x128", 95),
    ):
        legacy_outputs_for_shape = run(
            "legacy",
            actual_threads,
            actual_threads,
            tmp_path / f"legacy-{case_group}.pt",
            case_group,
        )
        flat_outputs = run(
            scheduler,
            actual_threads,
            actual_threads,
            tmp_path / f"{scheduler}.pt",
            case_group,
        )
        for legacy, flat in zip(legacy_outputs_for_shape, flat_outputs, strict=True):
            torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    legacy_eligible_128 = run(
        "legacy",
        2,
        2,
        tmp_path / "legacy-flat128x128-2.pt",
        "flat128x128_eligible",
    )
    flat_eligible_128 = run(
        "flat_128x128",
        requested_threads=128,
        thread_limit=2,
        output_path=tmp_path / "flat128x128-128-2.pt",
        case_group="flat128x128_eligible",
    )
    for legacy, flat in zip(legacy_eligible_128, flat_eligible_128, strict=True):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    tall_legacy_outputs = {}
    for actual_threads in (2, 7, 95):
        tall_legacy_outputs[actual_threads] = run(
            "legacy",
            actual_threads,
            actual_threads,
            tmp_path / f"legacy-flat128x128-tall-{actual_threads}.pt",
            "flat128x128_tall",
        )
        tall_flat_outputs = run(
            "flat_128x128",
            actual_threads,
            actual_threads,
            tmp_path / f"flat128x128-tall-{actual_threads}.pt",
            "flat128x128_tall",
        )
        for legacy, flat in zip(
            tall_legacy_outputs[actual_threads], tall_flat_outputs, strict=True
        ):
            torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    tall_constrained_outputs = run(
        "flat_128x128",
        requested_threads=128,
        thread_limit=2,
        output_path=tmp_path / "flat128x128-tall-128-2.pt",
        case_group="flat128x128_tall",
    )
    for legacy, flat in zip(
        tall_legacy_outputs[2], tall_constrained_outputs, strict=True
    ):
        torch.testing.assert_close(legacy, flat, rtol=0, atol=0)

    auto_legacy_outputs = run(
        "legacy",
        32,
        32,
        tmp_path / "legacy-flat-auto-32.pt",
        "flat_auto",
    )
    auto_outputs = run(
        "flat_auto",
        32,
        32,
        tmp_path / "flat-auto-32.pt",
        "flat_auto",
    )
    for legacy, auto in zip(auto_legacy_outputs, auto_outputs, strict=True):
        torch.testing.assert_close(legacy, auto, rtol=0, atol=0)

    auto_constrained_legacy_outputs = run(
        "legacy",
        requested_threads=2,
        thread_limit=2,
        output_path=tmp_path / "legacy-flat-auto-2.pt",
        case_group="flat_auto_wide",
    )
    auto_constrained_outputs = run(
        "flat_auto",
        requested_threads=128,
        thread_limit=2,
        output_path=tmp_path / "flat-auto-128-2.pt",
        case_group="flat_auto_wide",
    )
    for legacy, auto in zip(
        auto_constrained_legacy_outputs, auto_constrained_outputs, strict=True
    ):
        torch.testing.assert_close(legacy, auto, rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
@pytest.mark.parametrize("scheduler", ["flat_dense", "not_a_scheduler"])
def test_cpu_sgl_int8_rejects_removed_or_unknown_scheduler(scheduler):
    """Reject removed scheduler names with a clear configuration error."""
    script = textwrap.dedent(
        """
        import torch
        import vllm._custom_ops

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        x = torch.randn(1, 64).bfloat16()
        weight = torch.randint(-30, 31, (32, 64), dtype=torch.int8)
        scale = torch.ones(32)
        torch.ops._C.int8_scaled_mm_with_quant(
            x, weight, scale, None, x.dtype, False
        )
        """
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["OMP_THREAD_LIMIT"] = "2"
    env["OMP_DYNAMIC"] = "FALSE"
    env["VLLM_CPU_SGL_SCHEDULER"] = scheduler
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert b"VLLM_CPU_SGL_SCHEDULER" in result.stderr
    assert scheduler.encode() in result.stderr


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

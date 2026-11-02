# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only).

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
import json
import os
import subprocess
import sys
import textwrap
from abc import ABC
from pathlib import Path
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
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel
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
@pytest.mark.parametrize(
    "scheduler,requested_threads,thread_limit,serialized,case_group",
    [
        ("generic", 1, 1, False, "tails"),
        ("generic", 3, 3, False, "tails"),
        ("generic", 7, 7, False, "tails"),
        ("generic", 95, 95, False, "base"),
        ("generic", 96, 96, False, "base"),
        ("generic", 97, 97, False, "base"),
        ("generic", 128, 2, False, "base"),
        ("generic", 2, 2, True, "base"),
        ("generic", 24, 24, False, "cohorts"),
        ("generic", 25, 25, False, "cohorts"),
        ("generic", 48, 48, False, "cohorts"),
        ("generic", 95, 95, False, "cohorts"),
        ("generic", 96, 96, False, "cohorts"),
        ("generic", 96, 25, False, "cohorts"),
        ("generic", 24, 24, False, "legacy_plan"),
        ("generic", 25, 25, False, "legacy_plan"),
        ("generic", 1, 1, False, "local"),
        ("generic", 31, 31, False, "local"),
        ("generic", 96, 96, False, "local"),
    ],
)
def test_cpu_sgl_int8_dense_schedulers_preserve_scaled_mm_result(
    tmp_path, scheduler, requested_threads, thread_limit, serialized, case_group
):
    """Actual-team scheduling preserves every output and leaves inputs intact."""
    script = textwrap.dedent(
        """
        import os
        import sys
        import torch
        import vllm._custom_ops

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.set_num_interop_threads(1)
        cases = {
            "base": (
                (0, 32, 64, False, False, "bfloat16"),
                (1, 6144, 64, False, True, "bfloat16"),
                (33, 96, 64, True, False, "bfloat16"),
                (129, 352, 512, True, True, "bfloat16"),
                (512, 1024, 64, False, True, "bfloat16"),
            ),
            "tails": (
                (257, 288, 576, True, True, "bfloat16"),
                (513, 320, 1088, False, False, "bfloat16"),
                (320, 256, 2560, True, True, "bfloat16"),
            ),
            "cohorts": (
                (1025, 3072, 1024, False, False, "bfloat16"),
                (1025, 3072, 1088, False, False, "bfloat16"),
                (1023, 4128, 2048, True, True, "bfloat16"),
                (1025, 3072, 2560, False, True, "bfloat16"),
                (1024, 4096, 1056, True, False, "bfloat16"),
                (1024, 4096, 960, False, True, "bfloat16"),
            ),
            "legacy_plan": (
                (2048, 6144, 4096, True, True, "bfloat16"),
            ),
            # Region tails, short strips, and K512/K64 remainder blocks.
            "local": (
                (1000, 4160, 1088, True, True, "bfloat16"),
                (777, 4096, 2560, False, True, "bfloat16"),
                (300, 10240, 512, False, False, "bfloat16"),
            ),
        }[sys.argv[2]]
        outputs = []
        calls = []
        for m, n, k, has_bias, is_vnni, dtype in cases:
            torch.manual_seed(m + n)
            x = torch.randn(m, k).to(getattr(torch, dtype))
            storage = None
            if m >= 256 and k >= 512:
                if not has_bias:
                    storage = torch.full((m, k + 32), 123, dtype=x.dtype)
                    storage[:, :k].copy_(x)
                    x = storage[:, :k]
                x[0].zero_()
                x[1].fill_(1e-8)
                x[1, 1::2] = -1e-8
                # BF16 values bracket ties when the row maximum is 127.
                boundary = torch.tensor(
                    [-127, 127, -1.5078125, -1.5, -1.4921875,
                     -0.50390625, -0.5, -0.498046875, 0,
                     0.498046875, 0.5, 0.50390625,
                     1.4921875, 1.5, 1.5078125, 126], dtype=x.dtype,
                )
                x[[2, 31, 32, m - 1]] = boundary.repeat(k // 16)
                # Pending outputs cross panels with different row scales.
                x[96:128] = boundary.repeat(k // 16) * 0.125
                x[224:256] = boundary.repeat(k // 16) * 8
                if not has_bias:
                    patterns = (
                        (0, 32, 64), (64, 32, 0), (32, 0, 64), (64, 0, 32),
                        (0, k - 32, 32), (0, k - 16, 32),
                        (16, k - 32, 48), (0, 17, 64),
                    )
                    row = 64
                    for payload in (
                        0x7FC1, 0xFFC1, 0x7FFF, 0xFFFF,
                        0x7F81, 0xFF81, 0x7FBF, 0xFFBF,
                    ):
                        for large, nan, small in patterns:
                            x[row].fill_(1)
                            x[row, large] = 64
                            x[row, small] = -0.5
                            x.view(torch.int16)[row, nan] = (
                                payload if payload < 0x8000 else payload - 0x10000
                            )
                            row += 1
                    for payload in (0x7F80, 0xFF80):
                        x[row].fill_(1)
                        x[row, 0] = 64
                        x[row, 64] = -0.5
                        x.view(torch.int16)[row, 32] = (
                            payload if payload < 0x8000 else payload - 0x10000
                        )
                        row += 1
            weight = torch.randint(-30, 31, (n, k), dtype=torch.int8)
            if is_vnni:
                weight = torch.ops._C.convert_weight_packed(weight)
            weight_scale = torch.rand(n, dtype=torch.float32) * 0.01
            bias = torch.rand(n, dtype=torch.float32) if has_bias else None
            inputs = (
                storage if storage is not None else x, weight, weight_scale,
            ) + ((bias,) if bias is not None else ())
            saved_inputs = tuple(tensor.clone() for tensor in inputs)
            result = torch.ops._C.int8_scaled_mm_with_quant(
                x, weight, weight_scale, bias, x.dtype, is_vnni
            )
            calls.append((m, n, k, dtype))
            if (
                sys.argv[2] in ("cohorts", "legacy_plan", "local")
                and dtype == "bfloat16" and k >= 512
            ):
                # Alternate AMX and brgemm to cover tile-state restoration.
                small_x = torch.ones(33, 64, dtype=x.dtype)
                small_weight = torch.ones(32, 64, dtype=torch.int8)
                torch.ops._C.int8_scaled_mm_with_quant(
                    small_x, small_weight, torch.ones(32), None, x.dtype, False
                )
                calls.append((33, 32, 64, dtype))
                repeated = torch.ops._C.int8_scaled_mm_with_quant(
                    x, weight, weight_scale, bias, x.dtype, is_vnni
                )
                calls.append((m, n, k, dtype))
                torch.testing.assert_close(
                    repeated, result, rtol=0, atol=0, equal_nan=True
                )
            for tensor, saved in zip(inputs, saved_inputs, strict=True):
                assert torch.equal(tensor.view(torch.uint8), saved.view(torch.uint8))
            outputs.append(result)
        torch.save({"outputs": tuple(outputs), "calls": calls}, sys.argv[1])
        """
    )
    actual_threads = 1 if serialized else min(requested_threads, thread_limit)
    outputs = []
    dispatches = []
    for mode, requested, limit in (
        ("legacy", actual_threads, actual_threads),
        (scheduler, requested_threads, thread_limit),
    ):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(requested)
        env["OMP_THREAD_LIMIT"] = str(limit)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_MAX_ACTIVE_LEVELS"] = "0" if serialized else "1"
        env["VLLM_CPU_SGL_SCHEDULER"] = mode
        env["VLLM_CPU_SGL_DISPATCH"] = "1" if mode != "legacy" else "0"
        path = tmp_path / f"{mode}.pt"
        result = subprocess.run(
            [sys.executable, "-c", script, str(path), case_group],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        saved = torch.load(path, weights_only=True)
        outputs.append(saved["outputs"])
        if mode != "legacy":
            dispatches = [
                json.loads(line.removeprefix("SGL_DISPATCH "))
                for line in result.stderr.splitlines()
                if line.startswith("SGL_DISPATCH ")
            ]
    for legacy, scheduled in zip(*outputs, strict=True):
        torch.testing.assert_close(legacy, scheduled, rtol=0, atol=0, equal_nan=True)
    assert dispatches, "the operator did not report its executed dispatch"
    assert all(record["actual_team_size"] == actual_threads for record in dispatches)
    amx = sys.platform == "linux" and check_cpu_sgl_kernel(32, 32, torch.int8)
    supported_cache = False
    if amx:
        l2 = (
            Path(f"/sys/devices/system/cpu/cpu{min(os.sched_getaffinity(0))}")
            / "cache/index2/size"
        )
        supported_cache = l2.read_text().strip() == "2048K"
    assert len(dispatches) == len(saved["calls"])
    for record, (m, n, k, dtype) in zip(dispatches, saved["calls"], strict=True):
        amx_shape = (
            supported_cache
            and dtype == "bfloat16"
            and m >= 256
            and n >= 256
            and n % 32 == 0
            and k % 64 == 0
        )
        cohort_eligible = (
            amx_shape
            and 24 <= actual_threads <= 96
            and k >= 1024
            and (m // 32) * (n // 32) >= 32 * actual_threads
        )
        local_eligible = (
            amx_shape
            and actual_threads <= 96
            and 512 <= k <= 4096
            and ((m + 31) // 32) * (n // 32) >= actual_threads
        )
        if m >= 256 and (
            case_group == "legacy_plan"
            or (case_group == "local" and (actual_threads < 96 or k != 2560))
        ):
            assert record["kernel"] == "amx_local"
        if record["kernel"] == "amx_local":
            assert local_eligible
            assert record["macro_m"] == 256 and record["macro_n"] == 32
            assert record["k_block"] == min(1024, k)
            assert record["activation_bytes"] > m * (k + 68)
            assert record["scratch_bytes"] == 4 * actual_threads * 256 * 48
        elif cohort_eligible:
            assert record["kernel"] in ("amx_cohort", "legacy_team")
            replicas = record["replicas"]
            assert (record["kernel"] == "legacy_team") == (replicas == 1)
            # Local execution replaces single-replica fallback when eligible.
            assert replicas > 1 or not local_eligible
            assert 1 <= replicas <= actual_threads
            assert record["macro_m"] in (128, 256)
            assert record["macro_n"] in (32, 64)
            assert record["k_block"] in (1024, 2048, k)
            # Replica 0 shares the K + 64 stride quantization buffer.
            activation_bytes = m * (k + 68) + (replicas - 1) * (
                (m * (k + 68) + 63) // 64 * 64
            )
            assert record["activation_bytes"] == activation_bytes
            scratch_bytes = (
                0
                if replicas == 1
                else 4 * actual_threads * record["macro_m"] * (record["macro_n"] + 16)
            )
            assert record["scratch_bytes"] == scratch_bytes
            if case_group == "cohorts" and k == 1024:
                assert record["kernel"] == "amx_cohort"
            if case_group == "cohorts" and actual_threads in (25, 95) and replicas > 1:
                assert actual_threads % replicas != 0
            if n == 4128 and actual_threads in (95, 96):
                assert (n // 32) % replicas != 0
        else:
            # Sparse outputs without a cohort plan keep generic execution.
            assert (
                not local_eligible or ((m + 31) // 32) * (n // 32) < 32 * actual_threads
            )
            assert record["kernel"] == "generic"


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
def test_cpu_sgl_int8_rejects_unknown_scheduler():
    """Reject an unknown scheduler name with a clear configuration error."""
    scheduler = "not_a_scheduler"
    script = textwrap.dedent(
        """
        import torch
        import vllm._custom_ops

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        x = torch.randn(1, 64).bfloat16()
        weight = torch.randint(-30, 31, (32, 64), dtype=torch.int8)
        torch.ops._C.int8_scaled_mm_with_quant(
            x, weight, torch.ones(32), None, x.dtype, False
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

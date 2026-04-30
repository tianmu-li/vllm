# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Functional tests for CPUSGLIW8A8Int8MoEMethod and CPUSGLFp8W8A16MoEMethod.

Tests exercise weight packing, forward pass correctness (via reference
comparison), routing dispatch, and validation edge cases.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa E501
    CPUSGLIW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a16_fp8_cpu import (  # noqa E501
    CPUSGLFp8W8A16MoEMethod,
)
from vllm.platforms import CpuArchEnum, current_platform

if not current_platform.is_cpu():
    pytest.skip("CPU-only tests", allow_module_level=True)

if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
    pytest.skip("x86-only tests (AMX required)", allow_module_level=True)

if not torch.cpu._is_amx_tile_supported():
    pytest.skip("AMX not supported on this machine", allow_module_level=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
TOP_K = 2
BATCH = 16

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_moe_int8(
    x: torch.Tensor,  # [M, H]  bfloat16
    w13: torch.Tensor,  # [E, 2N, H]  int8
    w2: torch.Tensor,  # [E, H, N]   int8
    w13_scale: torch.Tensor,  # [E, 2N]  float32
    w2_scale: torch.Tensor,  # [E, H]   float32
    topk_weights: torch.Tensor,  # [M, K]  float32
    topk_ids: torch.Tensor,  # [M, K]  int32
) -> torch.Tensor:
    """Token-by-token reference for INT8 W8A8 MoE (SiLU gated)."""
    M, H = x.shape
    E, N2, _ = w13.shape
    N = N2 // 2

    out = torch.zeros(M, H, dtype=torch.float32)
    for m in range(M):
        acc = torch.zeros(H, dtype=torch.float32)
        for ki in range(TOP_K):
            eid = topk_ids[m, ki].item()
            wt = topk_weights[m, ki].item()

            w13_fp = w13[eid].float() * w13_scale[eid].unsqueeze(1)  # [2N, H]
            gate_up = x[m].float() @ w13_fp.T  # [2N]
            gate, up = gate_up[:N], gate_up[N:]
            hidden = torch.sigmoid(gate) * gate * up

            w2_fp = w2[eid].float() * w2_scale[eid].unsqueeze(1)  # [H, N]
            acc += wt * (hidden @ w2_fp.T)

        out[m] = acc

    return out.to(x.dtype)


def _ref_moe_fp8_w8a16(
    x: torch.Tensor,  # [M, H]  bfloat16
    w13: torch.Tensor,  # [E, 2N, H]  float8_e4m3fn
    w2: torch.Tensor,  # [E, H, N]   float8_e4m3fn
    w13_scale: torch.Tensor,  # [E, nb_n13, nb_k13]  float32
    w2_scale: torch.Tensor,  # [E, nb_n2, nb_k2]   float32
    block_n: int,
    block_k: int,
    topk_weights: torch.Tensor,  # [M, K]  float32
    topk_ids: torch.Tensor,  # [M, K]  int32
) -> torch.Tensor:
    """Token-by-token reference for FP8 W8A16 MoE (SiLU gated).

    Dequantizes each weight tile using its block scale before computing the
    matrix product in float32.
    """
    M, H = x.shape
    E, N2, _ = w13.shape
    N = N2 // 2

    def dequant_block(
        w_fp8: torch.Tensor, scale: torch.Tensor, bn: int, bk: int
    ) -> torch.Tensor:
        """Dequantize [out_f, in_f] fp8 weights with block scales [nb_n, nb_k]."""
        out_f, in_f = w_fp8.shape
        result = torch.zeros(out_f, in_f, dtype=torch.float32)
        for ni in range(math.ceil(out_f / bn)):
            for ki in range(math.ceil(in_f / bk)):
                nr = slice(ni * bn, min((ni + 1) * bn, out_f))
                kr = slice(ki * bk, min((ki + 1) * bk, in_f))
                result[nr, kr] = w_fp8[nr, kr].float() * scale[ni, ki].item()
        return result

    out = torch.zeros(M, H, dtype=torch.float32)
    for m in range(M):
        acc = torch.zeros(H, dtype=torch.float32)
        for ki in range(TOP_K):
            eid = topk_ids[m, ki].item()
            wt = topk_weights[m, ki].item()

            w13_fp = dequant_block(
                w13[eid], w13_scale[eid], block_n, block_k
            )  # [2N, H]
            gate_up = x[m].float() @ w13_fp.T
            gate, up = gate_up[:N], gate_up[N:]
            hidden = torch.sigmoid(gate) * gate * up

            w2_fp = dequant_block(w2[eid], w2_scale[eid], block_n, block_k)  # [H, N]
            acc += wt * (hidden @ w2_fp.T)

        out[m] = acc

    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Shared mock factories
# ---------------------------------------------------------------------------


def _make_layer_mock(
    activation: MoEActivation = MoEActivation.SILU,
    enable_eplb: bool = False,
) -> MagicMock:
    layer = MagicMock()
    layer.top_k = TOP_K
    layer.use_grouped_topk = False
    layer.renormalize = True
    layer.activation = activation
    layer.enable_eplb = enable_eplb
    return layer


def _make_moe_config() -> MagicMock:
    cfg = MagicMock(spec=FusedMoEConfig)
    cfg.is_act_and_mul = True
    cfg.has_bias = False
    return cfg


def _make_int8_method_and_weights(
    hidden_size: int,
    intermediate_size: int,
) -> tuple:
    """Return (method, layer, orig_w13, orig_w2, orig_w13_scale, orig_w2_scale).

    create_weights and process_weights_after_loading are already called.
    Original (unpacked) weights are returned for reference comparison.
    """
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )
    method = CPUSGLIW8A8Int8MoEMethod(weight_quant, input_quant, _make_moe_config())
    layer = _make_layer_mock()

    method.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.bfloat16,
    )

    orig_w13 = torch.randint(
        -10, 10, (NUM_EXPERTS, 2 * intermediate_size, hidden_size), dtype=torch.int8
    )
    orig_w2 = torch.randint(
        -10, 10, (NUM_EXPERTS, hidden_size, intermediate_size), dtype=torch.int8
    )
    orig_w13_scale = torch.rand(NUM_EXPERTS, 2 * intermediate_size, 1) * 0.1 + 0.01
    orig_w2_scale = torch.rand(NUM_EXPERTS, hidden_size, 1) * 0.1 + 0.01

    layer.w13_weight = torch.nn.Parameter(orig_w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(orig_w2.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        orig_w13_scale.clone(), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        orig_w2_scale.clone(), requires_grad=False
    )

    method.process_weights_after_loading(layer)
    return method, layer, orig_w13, orig_w2, orig_w13_scale, orig_w2_scale


def _make_fp8_method_and_weights(
    hidden_size: int,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 32,
) -> tuple:
    """Return (method, layer, w13, w2, w13_scale, w2_scale, block_n, block_k).

    create_weights and process_weights_after_loading are already called.
    """
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[block_n, block_k],
    )
    method = CPUSGLFp8W8A16MoEMethod(weight_quant, _make_moe_config())
    layer = _make_layer_mock()

    method.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.bfloat16,
    )

    orig_w13 = torch.randn(NUM_EXPERTS, 2 * intermediate_size, hidden_size).to(
        torch.float8_e4m3fn
    )
    orig_w2 = torch.randn(NUM_EXPERTS, hidden_size, intermediate_size).to(
        torch.float8_e4m3fn
    )

    nb_n13 = math.ceil(2 * intermediate_size / block_n)
    nb_k13 = math.ceil(hidden_size / block_k)
    nb_n2 = math.ceil(hidden_size / block_n)
    nb_k2 = math.ceil(intermediate_size / block_k)

    orig_w13_scale = torch.rand(NUM_EXPERTS, nb_n13, nb_k13) * 0.1 + 0.01
    orig_w2_scale = torch.rand(NUM_EXPERTS, nb_n2, nb_k2) * 0.1 + 0.01

    layer.w13_weight = torch.nn.Parameter(orig_w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(orig_w2.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        orig_w13_scale.clone(), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        orig_w2_scale.clone(), requires_grad=False
    )

    method.process_weights_after_loading(layer)
    return (
        method,
        layer,
        orig_w13,
        orig_w2,
        orig_w13_scale,
        orig_w2_scale,
        block_n,
        block_k,
    )


def _make_dispatch_mocks(true_flags: "list[str]") -> tuple:
    """Return (quant_config, layer) mocks for routing tests.

    All _is_* methods default to False; those in true_flags are set to True.
    """
    all_flags = [
        "_is_mxfp4",
        "_is_mxfp8",
        "_is_wNa16_group_channel",
        "_is_nvfp4_format",
        "_is_fp8_w8a8_sm90",
        "_is_fp8_w8a8_sm100",
        "_is_fp8_w8a8",
        "_is_dynamic_token_w8a8",
        "_is_fp8_w8a16",
        "_is_fp8_w4a8_sm90",
        "_is_dynamic_token_w4a8_int",
    ]
    qc = MagicMock()
    qc._add_fused_moe_to_target_scheme_map.return_value = None
    for flag in all_flags:
        getattr(qc, flag).return_value = flag in true_flags

    layer = MagicMock()
    layer.moe_config = _make_moe_config()
    return qc, layer


# ---------------------------------------------------------------------------
# INT8 W8A8 forward test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_size,intermediate_size", [(128, 128), (256, 128)])
def test_cpu_sgl_int8_w8a8_moe(hidden_size: int, intermediate_size: int):
    """CPUSGLIW8A8Int8MoEMethod: pack weights and compare forward pass to reference."""
    method, layer, orig_w13, orig_w2, orig_w13_scale, orig_w2_scale = (
        _make_int8_method_and_weights(hidden_size, intermediate_size)
    )

    # Verify packing shape: K dim grows by 4 (VNNI compensation row)
    assert layer.w13_weight.shape == (
        NUM_EXPERTS,
        2 * intermediate_size,
        hidden_size + 4,
    ), f"w13 packed shape wrong: {layer.w13_weight.shape}"
    assert layer.w13_weight.dtype == torch.int8
    assert layer._sgl_w13_scale.shape == (NUM_EXPERTS, 2 * intermediate_size)
    assert layer._sgl_w2_scale.shape == (NUM_EXPERTS, hidden_size)

    torch.manual_seed(42)
    x = torch.randn(BATCH, hidden_size, dtype=torch.bfloat16)
    router_logits = torch.randn(BATCH, NUM_EXPERTS, dtype=torch.float32)

    # Get reference routing and output
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=TOP_K,
        use_grouped_topk=False,
        renormalize=True,
    )
    ref = _ref_moe_int8(
        x,
        orig_w13,
        orig_w2,
        orig_w13_scale.squeeze(-1),
        orig_w2_scale.squeeze(-1),
        topk_weights,
        topk_ids,
    )

    out = method.apply_monolithic(layer, x.clone(), router_logits)

    assert out.shape == (BATCH, hidden_size), f"Output shape mismatch: {out.shape}"
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    max_diff = (out.float() - ref.float()).abs().max().item()
    # Loose tolerance: SGL kernel quantizes activations to INT8 per-token
    # while the reference uses floating-point, introducing ~1 INT8-step error
    # that scales with the output magnitude (~100 for these weight/act scales).
    assert torch.allclose(out.float(), ref.float(), atol=10.0, rtol=0.0), (
        f"INT8 output diverges from reference: max_diff={max_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# FP8 W8A16 forward test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,block_n,block_k",
    [(128, 128, 32, 32), (256, 128, 64, 32)],
)
def test_cpu_sgl_fp8_w8a16_moe(
    hidden_size: int, intermediate_size: int, block_n: int, block_k: int
):
    """CPUSGLFp8W8A16MoEMethod: pack weights and compare forward pass to reference."""
    method, layer, orig_w13, orig_w2, orig_w13_scale, orig_w2_scale, bn, bk = (
        _make_fp8_method_and_weights(hidden_size, intermediate_size, block_n, block_k)
    )

    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer._sgl_block_size == [block_n, block_k]

    torch.manual_seed(42)
    x = torch.randn(BATCH, hidden_size, dtype=torch.bfloat16)
    router_logits = torch.randn(BATCH, NUM_EXPERTS, dtype=torch.float32)

    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=TOP_K,
        use_grouped_topk=False,
        renormalize=True,
    )
    ref = _ref_moe_fp8_w8a16(
        x,
        orig_w13,
        orig_w2,
        orig_w13_scale,
        orig_w2_scale,
        bn,
        bk,
        topk_weights,
        topk_ids,
    )

    out = method.apply_monolithic(layer, x.clone(), router_logits)

    assert out.shape == (BATCH, hidden_size), f"Output shape mismatch: {out.shape}"
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    max_diff = (out.float() - ref.float()).abs().max().item()
    # FP8 quantization noise scales with output magnitude and block size;
    # atol=5.0 accommodates the larger 256-dim case with block_n=64.
    assert torch.allclose(out.float(), ref.float(), atol=5.0, rtol=0.0), (
        f"FP8 output diverges from reference: max_diff={max_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


def test_get_moe_method_routes_cpu_int8():
    """get_moe_method routes to CPUSGLIW8A8Int8MoEMethod on CPU."""
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )
    qc, layer = _make_dispatch_mocks(["_is_dynamic_token_w8a8"])
    qc.get_scheme_dict.return_value = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "format": None,
    }

    result = CompressedTensorsMoEMethod.get_moe_method(
        qc, layer, "model.layers.0.mlp.experts"
    )
    assert isinstance(result, CPUSGLIW8A8Int8MoEMethod), (
        f"Expected CPUSGLIW8A8Int8MoEMethod, got {type(result)}"
    )


def test_get_moe_method_routes_cpu_fp8():
    """get_moe_method routes to CPUSGLFp8W8A16MoEMethod on CPU."""
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[32, 32],
    )
    qc, layer = _make_dispatch_mocks(["_is_fp8_w8a16"])
    qc.get_scheme_dict.return_value = {
        "weights": weight_quant,
        "input_activations": None,
        "format": None,
    }

    result = CompressedTensorsMoEMethod.get_moe_method(
        qc, layer, "model.layers.0.mlp.experts"
    )
    assert isinstance(result, CPUSGLFp8W8A16MoEMethod), (
        f"Expected CPUSGLFp8W8A16MoEMethod, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# Negative / edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_id", ["int8", "fp8"])
def test_cpu_sgl_wrong_activation(method_id: str):
    """apply_monolithic rejects non-SiLU activations."""
    if method_id == "int8":
        method, layer, *_ = _make_int8_method_and_weights(128, 128)
    else:
        method, layer, *_ = _make_fp8_method_and_weights(128, 128)

    layer.activation = MoEActivation.GELU
    x = torch.randn(BATCH, 128, dtype=torch.bfloat16)
    router_logits = torch.randn(BATCH, NUM_EXPERTS, dtype=torch.float32)

    with pytest.raises(AssertionError):
        method.apply_monolithic(layer, x, router_logits)


@pytest.mark.parametrize("method_id", ["int8", "fp8"])
def test_cpu_sgl_eplb_rejected(method_id: str):
    """apply_monolithic rejects EPLB-enabled layers."""
    if method_id == "int8":
        method, layer, *_ = _make_int8_method_and_weights(128, 128)
    else:
        method, layer, *_ = _make_fp8_method_and_weights(128, 128)

    layer.enable_eplb = True
    x = torch.randn(BATCH, 128, dtype=torch.bfloat16)
    router_logits = torch.randn(BATCH, NUM_EXPERTS, dtype=torch.float32)

    with pytest.raises(AssertionError):
        method.apply_monolithic(layer, x, router_logits)


def test_cpu_sgl_constructor_validation():
    """Constructors raise ValueError on invalid configs."""
    # FP8 with non-BLOCK strategy must be rejected
    bad_fp8_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    with pytest.raises(ValueError, match="BLOCK"):
        CPUSGLFp8W8A16MoEMethod(bad_fp8_quant, _make_moe_config())

    # INT8 with non-TOKEN input strategy must be rejected (pydantic rejects
    # static TOKEN, so we use TENSOR+dynamic=False as a non-dynamic alt)
    w_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    non_dynamic_input_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    with pytest.raises(ValueError, match="dynamic"):
        CPUSGLIW8A8Int8MoEMethod(w_quant, non_dynamic_input_quant, _make_moe_config())

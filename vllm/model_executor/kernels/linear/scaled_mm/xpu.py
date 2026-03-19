# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """XPU kernel for W8A8 integer quantization using oneDNN int8_gemm_w8a8.

    Weights are symmetric or asymmetric (per-channel or per-group) quantized int8.
    Activations are dynamically quantized per-token to symmetric or asymmetric int8.
    Currently configured for symmetric quantization (zero points set to zero).
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUInt8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Transpose weight scale from [N, K/gs] or [N, 1] to [K/gs, N] or [1, N]
        # so that the C++ kernel receives scales in [num_groups, n] format.
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight_scale = getattr(layer, w_s_name)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.t().data.contiguous(), requires_grad=False),
        )
        weight_zero_points = torch.zeros_like(layer.weight_scale, dtype=torch.int32)
        layer.weight_zero_point = torch.nn.Parameter(
            weight_zero_points, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])  # [M, K]
        from vllm._xpu_ops import xpu_ops as ops

        # Symmetric per-token quantization; zero point is set to zero.
        quant_x, x_scale, x_zero = ops.dynamic_per_token_int4_int8_quant_ref(
            reshaped_x, True, 8
        )

        out = torch.ops._xpu_C.int8_gemm_w8a8(
            quant_x,
            x_scale,
            x_zero,  # activation zero points (zero for symmetric)
            layer.weight.t(),  # [K, N] NT-format view
            layer.weight_scale,  # [K/gs, N] or [1, N]
            layer.weight_zero_point,  # weight zero points (zero for symmetric)
            -1,
            bias,
        )
        return out.to(x.dtype)

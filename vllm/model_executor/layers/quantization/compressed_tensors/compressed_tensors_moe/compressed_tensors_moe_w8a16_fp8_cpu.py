# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import CpuArchEnum, current_platform


class CPUSGLFp8W8A16MoEMethod(CompressedTensorsMoEMethod):
    """CPU-only FP8 W8A16 MoE via SGL fused_experts_cpu kernel (x86 AMX).

    Weights are stored as float8_e4m3fn with block-wise float32 scales.
    Activations are BF16/FP16 (no quantization applied to activations).
    Only QuantizationStrategy.BLOCK is supported — block scales are mandatory
    for the SGL FP8 kernel.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        if not current_platform.is_cpu():
            raise ValueError("CPUSGLFp8W8A16MoEMethod is CPU-only.")
        if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
            raise ValueError("CPUSGLFp8W8A16MoEMethod requires x86 (AMX) architecture.")
        if not torch.cpu._is_amx_tile_supported():
            raise ValueError("CPUSGLFp8W8A16MoEMethod requires AMX tile support.")
        if weight_quant.strategy != QuantizationStrategy.BLOCK:
            raise ValueError(
                "CPUSGLFp8W8A16MoEMethod only supports BLOCK quantization strategy "
                f"(got {weight_quant.strategy}). Use block-wise FP8 quantization."
            )
        self.weight_quant = weight_quant
        # block_structure is a list [block_N, block_K]
        self.weight_block_size: list[int] = list(weight_quant.block_structure)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        block_n, block_k = self.weight_block_size[0], self.weight_block_size[1]

        layer.weight_block_size = self.weight_block_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Block-wise scales: [E, ceil(N/block_N), ceil(K/block_K)]
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_num_shards
                * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        N_w13, K = layer.w13_weight.shape[1], layer.w13_weight.shape[2]
        N_w2, K_w2 = layer.w2_weight.shape[1], layer.w2_weight.shape[2]
        if N_w13 % 16 != 0 or K % 32 != 0:
            raise ValueError(
                f"w13_weight dims ({N_w13}, {K}) do not satisfy SGL AMX alignment "
                f"requirements (N%16==0, K%32==0)."
            )
        if N_w2 % 16 != 0 or K_w2 % 32 != 0:
            raise ValueError(
                f"w2_weight dims ({N_w2}, {K_w2}) do not satisfy SGL AMX alignment "
                f"requirements (N%16==0, K%32==0)."
            )

        # VNNI-pack fp8 weights; convert_weight_packed handles float8_e4m3fn
        packed_w13 = torch.ops._C.convert_weight_packed(layer.w13_weight)
        replace_parameter(
            layer,
            "w13_weight",
            torch.nn.Parameter(packed_w13, requires_grad=False),
        )
        packed_w2 = torch.ops._C.convert_weight_packed(layer.w2_weight)
        replace_parameter(
            layer,
            "w2_weight",
            torch.nn.Parameter(packed_w2, requires_grad=False),
        )

        layer._sgl_block_size = self.weight_block_size

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert layer.activation == MoEActivation.SILU, (
            f"{layer.activation} is not supported by SGL FP8 MoE kernel; "
            "use SiLU activation."
        )
        assert not layer.enable_eplb, "EPLB not supported for SGL FP8 W8A16 MoE."

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=layer.top_k,
            use_grouped_topk=layer.use_grouped_topk,
            renormalize=layer.renormalize,
        )

        torch.ops._C.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids.to(torch.int32),
            True,  # inplace
            False,  # use_int8_w8a8
            True,  # use_fp8_w8a16
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer._sgl_block_size,  # [block_N, block_K]
            None,  # a1_scale (A16 — no activation quantization)
            None,  # a2_scale
            True,  # is_vnni (weights already VNNI-packed)
        )
        return x

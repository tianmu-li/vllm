# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata geometry must come from the builder's own group."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackendImpl,
    CPUAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

requires_cpu = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU attention backend"
)

# Laguna's shape: 48 query heads model-wide, 64 on its sliding layers, both
# against 8 KV heads.
MODEL_WIDE_NUM_HEADS = 48
NUM_KV_HEADS = 8


def _layers(layer_num_heads: list[int]):
    """Stand-in attention layers, one per head count, as one attention group."""
    return {
        f"layer_{i}": SimpleNamespace(
            impl=MagicMock(
                spec=CPUAttentionBackendImpl,
                num_heads=num_heads,
                sliding_window=None,
            )
        )
        for i, num_heads in enumerate(layer_num_heads)
    }


def _build(layer_num_heads: list[int]) -> CPUAttentionMetadataBuilder:
    layers = _layers(layer_num_heads)
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.get_num_attention_heads.return_value = MODEL_WIDE_NUM_HEADS
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.uniform_decode_query_len = 4
    vllm_config.speculative_config = None
    kv_cache_spec = SimpleNamespace(
        num_kv_heads=NUM_KV_HEADS, head_size=64, block_size=16
    )

    with (
        patch(
            "vllm.v1.attention.backends.utils.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch(
            "vllm.v1.attention.backends.cpu_attn.get_layers_from_vllm_config",
            return_value=layers,
        ),
    ):
        return CPUAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=list(layers),
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )


@requires_cpu
@pytest.mark.parametrize("group_num_heads", [MODEL_WIDE_NUM_HEADS, 64, 16])
def test_num_heads_comes_from_the_group(group_num_heads):
    """The group's own count wins, even when it is not the model-wide one."""
    builder = _build([group_num_heads, group_num_heads])
    assert builder.num_heads == group_num_heads


@requires_cpu
def test_mixed_head_counts_in_one_group_are_rejected():
    """Grouping guarantees uniformity; a mixed group means that broke."""
    with pytest.raises(AssertionError, match="share num_heads"):
        _build([MODEL_WIDE_NUM_HEADS, 64])


@pytest.mark.parametrize(
    "query_lens,is_prefilling,expected",
    [
        ([4, 17, 0, 1], [False, True, False, False], [True, False, False, True]),
        ([5, 1], [False, True], [False, False]),
    ],
)
def test_cpu_decode_mask_uses_request_state_and_verification_limit(
    query_lens, is_prefilling, expected
):
    """Short prefill rows and padding must not become grouped decode work."""
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            uniform_decode_query_len=4,
            speculative_config=None,
        )
    )
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0),
        is_prefilling=torch.tensor(is_prefilling),
    )

    actual = CPUAttentionMetadataBuilder._build_decode_mask(
        builder, common, causal=True
    )

    assert actual.tolist() == expected


def test_cpu_decode_mask_keeps_medusa_layout_unchanged():
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            uniform_decode_query_len=4,
            speculative_config=SimpleNamespace(method="medusa"),
        )
    )
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        is_prefilling=torch.tensor([False]),
    )

    actual = CPUAttentionMetadataBuilder._build_decode_mask(
        builder, common, causal=True
    )

    assert not actual.any()


@pytest.mark.parametrize(
    "causal,expected",
    [
        (torch.tensor([True, False]), [True, False]),
        (False, [False, False]),
    ],
)
def test_cpu_decode_mask_uses_request_causality(causal, expected):
    builder = SimpleNamespace(
        is_cross_attention=False,
        vllm_config=SimpleNamespace(
            uniform_decode_query_len=4,
            speculative_config=None,
        ),
    )
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False]),
    )

    actual = CPUAttentionMetadataBuilder._build_decode_mask(
        builder, common, causal=causal
    )

    assert actual.tolist() == expected


@pytest.mark.parametrize(
    "is_prefilling,is_cross_attention,causal",
    [
        (None, False, True),
        (torch.tensor([False]), True, True),
        (torch.tensor([False]), False, None),
    ],
)
def test_cpu_decode_mask_fails_closed_without_request_state(
    is_prefilling, is_cross_attention, causal
):
    builder = SimpleNamespace(
        is_cross_attention=is_cross_attention,
        vllm_config=SimpleNamespace(
            uniform_decode_query_len=4,
            speculative_config=None,
        ),
    )
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        is_prefilling=is_prefilling,
    )

    actual = CPUAttentionMetadataBuilder._build_decode_mask(
        builder, common, causal=causal
    )

    assert not actual.any()


@requires_cpu
@requires_cpu
def test_cpu_builder_forwards_decode_mask_to_scheduler():
    builder = _build([8, 8])
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=100,
        query_start_loc=torch.tensor([0, 4, 5], dtype=torch.int32),
        seq_lens=torch.tensor([100, 101], dtype=torch.int32),
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.arange(5, dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([False, True]),
    )

    with patch(
        "vllm.v1.attention.backends.cpu_attn.ops.cpu_attn_get_scheduler_metadata",
        wraps=ops.cpu_attn_get_scheduler_metadata,
    ) as scheduler:
        metadata = builder.build(0, common)

    decode_mask = scheduler.call_args.kwargs["decode_mask"]
    assert decode_mask.dtype == torch.bool
    assert decode_mask.device.type == "cpu"
    assert decode_mask.is_contiguous()
    assert decode_mask.tolist() == [True, False]
    assert scheduler.call_args.kwargs["_scheduler_policy"] == "per-request"
    assert metadata.scheduler_metadata is not None
    assert metadata.scheduler_metadata.numel() > 0


def test_flash_attention_geometry_comes_from_the_group():
    """FA3 must use group geometry for layers without an ``impl`` wrapper."""
    layers = {f"layer_{i}": SimpleNamespace(num_heads=16) for i in range(2)}
    vllm_config = MagicMock()
    vllm_config.model_config.get_num_attention_heads.return_value = MODEL_WIDE_NUM_HEADS
    vllm_config.model_config.get_num_kv_heads.return_value = NUM_KV_HEADS
    vllm_config.model_config.get_head_size.return_value = 128
    vllm_config.model_config.rswa_window = None
    vllm_config.model_config.is_mm_prefix_lm = False
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = (
        False
    )
    vllm_config.compilation_config.max_cudagraph_capture_size = None
    kv_cache_spec = SimpleNamespace(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )

    with (
        patch(
            "vllm.v1.attention.backends.utils.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch(
            "vllm.distributed.parallel_state.get_dcp_group",
            side_effect=AssertionError,
        ),
        patch(
            "vllm.v1.attention.backends.flash_attn.get_flash_attn_version",
            return_value=3,
        ),
    ):
        builder = FlashAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=list(layers),
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.aot_schedule
    assert builder.num_heads_q == 16
    assert builder.num_heads_kv == 2
    assert builder.headdim == 64

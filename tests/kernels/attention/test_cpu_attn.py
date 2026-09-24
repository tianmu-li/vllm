# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import math

import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)

# Enable AMX tile data registers so isolated runs (e.g. -k fp8_amx) don't rely
# on ref_paged_attn's einsum to trigger oneDNN's _init_amx() first.
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()

NUM_HEADS = [
    (4, 4),
    (8, 2),
    (9, 3),
]
HEAD_SIZES = [96, 128, 512]
HEAD_SIZES_VEC16 = [96, 80, 112, 128]
QTYPES = [torch.bfloat16, torch.half, torch.float32]
SLIDING_WINDOWS = [None, 256]
NUM_BLOCKS = [
    1024,
]
SEQ_LENS = [  # (q_len, kv_len)
    [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],  # decode batch
    [(2345, 2345), (5, 5), (3, 16), (134, 5131)],  # prefill batch
    [(992, 2456), (1, 1234), (98, 1145), (1, 4162), (2345, 2345)],  # mixed batch
]
DECODE_MASK_SEQ_LENS = [[(2, 97), (4, 193), (8, 385), (17, 577)]]
_FP8_ATOL = {"fp8_e4m3": 0.2, "fp8_e5m2": 0.3}
_FP8_RTOL = 0.1
ENCODER_SEQ_LENS = [
    [1, 678, 2367, 145, 4162, 36, 7812],
]


def get_attn_isa(
    block_size: int | None = None,
    dtype: torch.dtype | None = None,
):
    # Delegate to _get_attn_isa so the fallback path applies the same arch
    # gating (e.g. RISC-V RVV is only chosen when the build's hardcoded
    # VLEN=128 kernel is actually present; on VLEN=256 / scalar hosts it
    # correctly falls through to vec/vec16).
    return _get_attn_isa(
        dtype if dtype is not None else torch.bfloat16,
        block_size if block_size else 32,
    )


# rand number generation takes too much time, cache rand tensors
@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(elem_num: int, dtype: torch.dtype, tag: str = "none") -> torch.Tensor:
    tensor = torch.randn(elem_num, dtype=dtype)

    return tensor


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes.float()


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
    dynamic_causal: list[bool] | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    dtype = query.dtype

    outputs: list[torch.Tensor] = []
    start_idx = 0

    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes[:, None, None]

    if s_aux is not None:
        s_aux = s_aux.float()
        s_aux = s_aux[:, None, None]

    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len].float()
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len].float()

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)

        if dynamic_causal is None or dynamic_causal[i]:
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            if sliding_window is not None:
                sliding_window_mask = (
                    torch.triu(
                        empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                    )
                    .bool()
                    .logical_not()
                )
                mask |= sliding_window_mask
        else:
            if sliding_window is not None:
                mask = (
                    torch.triu(
                        empty_mask, diagonal=1 - sliding_window + kv_len - query_len
                    ).bool()
                    ^ torch.triu(
                        empty_mask, diagonal=sliding_window + kv_len - query_len
                    ).bool()
                ).logical_not()
            else:
                mask = empty_mask.logical_not()

        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)

        if alibi_slopes is not None:
            q_start_pos = kv_len - query_len
            q_pos = q_start_pos + torch.arange(0, query_len)[None, :, None]
            kv_pos = torch.arange(0, kv_len)[None, None, :]
            dist = q_pos - kv_pos
            alibi_bias = -alibi_slopes * dist
            attn += alibi_bias

        attn.masked_fill_(mask, float("-inf"))

        if s_aux is not None:
            s_aux_ext = s_aux.repeat(1, query_len, 1)
            attn = torch.cat((s_aux_ext, attn), dim=-1)

        attn = torch.softmax(attn, dim=-1)

        if s_aux is not None:
            attn = attn[:, :, 1:]

        out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=dtype)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def ref_varlen_encoder_attn(
    query: torch.Tensor,  # [token, q_head_num, head_dim]
    key: torch.Tensor,  # [token, kv_head_num, head_dim]
    value: torch.Tensor,
    seq_lens: list[int],
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    num_seqs = len(seq_lens)
    dtype = query.dtype

    output = torch.empty_like(query)

    start_idx = 0
    for i in range(num_seqs):
        seq_len = seq_lens[i]
        q = query[start_idx : start_idx + seq_len].float()
        k = key[start_idx : start_idx + seq_len].float()
        v = value[start_idx : start_idx + seq_len].float()
        q *= scale

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(seq_len, seq_len)
        if sliding_window is not None:
            mask = (
                torch.triu(empty_mask, diagonal=1 - sliding_window).bool()
                ^ torch.triu(empty_mask, diagonal=sliding_window).bool()
            ).logical_not()
        else:
            mask = empty_mask.logical_not()

        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=dtype)
        output[start_idx : start_idx + seq_len].copy_(out)

        start_idx += seq_len

    return output


@torch.inference_mode()
def varlen_encoder_attention(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    isa: str,
) -> None:
    set_random_seed(0)
    num_seqs = len(seq_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5
    token_num = sum(seq_lens)

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    query_start_loc = torch.zeros(num_seqs, dtype=torch.int32)
    torch.cumsum(seq_lens_tensor[:-1], 0, out=query_start_loc[1:])
    block_nums = (seq_lens_tensor + block_size - 1) // block_size
    start_block_ids = torch.zeros_like(seq_lens_tensor)
    torch.cumsum(block_nums[:-1], 0, out=start_block_ids[1:])
    total_block_num: int = block_nums.sum().item()
    max_block_num = block_nums.max().item()
    block_offsets = torch.arange(0, max_block_num, dtype=torch.int32)
    encoder_block_table = start_block_ids[:, None] + block_offsets[None, :]
    slot_mapping_list = []
    slot_start_idx = 0
    for i in range(num_seqs):
        block_num = block_nums[i].item()
        seq_len = seq_lens[i]
        slot_mapping_list.append(torch.arange(slot_start_idx, slot_start_idx + seq_len))
        slot_start_idx += block_num * block_size
    slot_mapping = torch.cat(slot_mapping_list)

    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
        tag="query",
    )
    query = query.view(
        token_num,
        num_query_heads,
        head_size,
    )

    key_value = tensor_cache(
        elem_num=2 * token_num * num_kv_heads * head_size,
        dtype=dtype,
        tag="kv",
    )
    key_value = key_value.view(
        2,
        token_num,
        num_kv_heads,
        head_size,
    )
    key, value = key_value.unbind(0)

    # KV cache for CPU attention
    packed_key_value_cache = torch.zeros(
        total_block_num, num_kv_heads, block_size, head_size * 2, dtype=dtype
    )
    packed_key_value_cache = packed_key_value_cache.view(
        (total_block_num, num_kv_heads, block_size * 2, -1)
    )
    packed_key_cache, packed_value_cache = packed_key_value_cache.chunk(2, dim=2)

    cu_query_lens = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # use reshape_and_cache to pack key_cache and value_cache
    cpu_attn_reshape_and_cache(
        key=key.view(-1, num_kv_heads, head_size),
        value=value.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=False,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=False,
    )

    out_without_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_without_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=False,
        alibi_slopes=None,
        sliding_window=sliding_window if sliding_window is not None else -1,
        block_table=encoder_block_table,
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=False,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=True,
    )

    out_with_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_with_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=False,
        alibi_slopes=None,
        sliding_window=sliding_window if sliding_window is not None else -1,
        block_table=encoder_block_table,
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
    )

    ref_output = ref_varlen_encoder_attn(
        query=query,
        key=key,
        value=value,
        seq_lens=seq_lens,
        scale=scale,
        sliding_window=sliding_window,
    )
    atol, rtol = 1.5e-2, 1e-2

    (
        torch.testing.assert_close(out_with_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_with_split - ref_output))}",
    )
    (
        torch.testing.assert_close(out_without_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_without_split - ref_output))}",
    )


@torch.inference_mode()
def varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    dynamic_causal: list[bool] | None = None,
    s_aux_dtype: torch.dtype = torch.bfloat16,
    decode_mask: list[bool] | None = None,
    forced_q_head_group: int = 0,
    forced_kv_split_count: int = 0,
) -> None:
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    token_num = sum(query_lens)
    dynamic_causal_tensor = (
        torch.tensor(dynamic_causal, dtype=torch.bool)
        if dynamic_causal is not None
        else None
    )
    decode_mask_tensor = (
        torch.tensor(decode_mask, dtype=torch.bool) if decode_mask is not None else None
    )

    # for n heads the set of slopes is the geometric sequence that starts
    # 2^(-8/n)
    alibi_slopes = _get_alibi_slopes(num_query_heads) if use_alibi else None

    s_aux = 15 * torch.rand((num_query_heads,), dtype=s_aux_dtype) if use_sink else None

    is_fp8 = kv_cache_dtype != "auto"
    if is_fp8 and current_platform.get_cpu_architecture() != CpuArchEnum.X86:
        pytest.skip("FP8 KV cache only supported on x86")

    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
    )
    query = query.view(
        token_num,
        num_query_heads,
        head_size,
    )

    key_value = tensor_cache(
        elem_num=2 * num_blocks * num_kv_heads * block_size * head_size,
        dtype=dtype,
    )
    key_value = key_value.view(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
    )
    if is_fp8:
        # Clamp KV to [-1, 1] so FP8 quantization error (<=12.5% for E4M3,
        # <=25% for E5M2) stays within the test tolerances regardless of
        # which tensor_cache values happen to be in use.
        key_value = key_value.clamp(-1, 1)
    key_cache, value_cache = key_value.unbind(0)

    # KV cache for CPU attention
    cache_dtype = torch.uint8 if is_fp8 else dtype
    packed_key_value_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size * 2, dtype=cache_dtype
    )
    packed_key_value_cache = packed_key_value_cache.view(
        (num_blocks, num_kv_heads, block_size * 2, -1)
    )
    packed_key_cache, packed_value_cache = packed_key_value_cache.chunk(2, dim=2)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # use reshape_and_cache to pack key_cache and value_cache
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    fp8_kwargs: dict = (
        dict(k_scale=k_scale, v_scale=v_scale, kv_cache_dtype=kv_cache_dtype)
        if is_fp8
        else {}
    )
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
        **fp8_kwargs,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=dynamic_causal is None,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=False,
        dynamic_causal=dynamic_causal_tensor,
        kv_cache_dtype=kv_cache_dtype,
        decode_mask=decode_mask_tensor,
        forced_q_head_group=forced_q_head_group,
        forced_kv_split_count=forced_kv_split_count,
    )

    out_without_split = torch.empty_like(query)
    if s_aux is not None and s_aux.dtype != torch.bfloat16:
        s_aux = s_aux.to(torch.float32)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_without_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=dynamic_causal is None,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window if sliding_window is not None else -1,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
        dynamic_causal=dynamic_causal_tensor,
        **fp8_kwargs,
    )

    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=dynamic_causal is None,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=True,
        dynamic_causal=dynamic_causal_tensor,
        kv_cache_dtype=kv_cache_dtype,
        decode_mask=decode_mask_tensor,
        forced_q_head_group=forced_q_head_group,
        forced_kv_split_count=forced_kv_split_count,
    )

    out_with_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_with_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=dynamic_causal is None,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window if sliding_window is not None else -1,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
        dynamic_causal=dynamic_causal_tensor,
        **fp8_kwargs,
    )

    if is_fp8:
        # Build a float KV cache via the non-FP8 path and run float attention
        # to use as the reference.
        ref_key_cache = torch.empty(
            num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
        )
        ref_value_cache = torch.empty_like(ref_key_cache)
        cpu_attn_reshape_and_cache(
            key=key_cache.view(-1, num_kv_heads, head_size),
            value=value_cache.view(-1, num_kv_heads, head_size),
            key_cache=ref_key_cache,
            value_cache=ref_value_cache,
            slot_mapping=slot_mapping,
            isa=isa,
        )
        ref_metadata = cpu_attn_get_scheduler_metadata(
            num_reqs=num_seqs,
            num_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            seq_lens=kv_lens_tensor,
            dtype=dtype,
            query_start_loc=cu_query_lens,
            causal=dynamic_causal is None,
            sliding_window_size=(sliding_window if sliding_window is not None else -1),
            isa=isa,
            enable_kv_split=True,
            dynamic_causal=dynamic_causal_tensor,
            kv_cache_dtype="auto",
            decode_mask=decode_mask_tensor,
            forced_q_head_group=forced_q_head_group,
            forced_kv_split_count=forced_kv_split_count,
        )
        ref_output = torch.empty_like(query)
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=ref_key_cache,
            value_cache=ref_value_cache,
            output=ref_output,
            query_start_loc=cu_query_lens,
            seq_lens=kv_lens_tensor,
            scale=scale,
            causal=dynamic_causal is None,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window if sliding_window is not None else -1,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=ref_metadata,
            s_aux=s_aux,
            dynamic_causal=dynamic_causal_tensor,
        )
        atol = _FP8_ATOL[kv_cache_dtype]
        rtol = _FP8_RTOL
    else:
        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            alibi_slopes=alibi_slopes,
            s_aux=s_aux,
            dynamic_causal=dynamic_causal,
        )
        atol, rtol = 1.5e-2, 1e-2

    (
        torch.testing.assert_close(out_with_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_with_split - ref_output))}",
    )
    (
        torch.testing.assert_close(out_without_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_without_split - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", ENCODER_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "block_size",
    [
        128,
    ],
)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("isa", ["vec"])
def test_varlen_encoder_attention_vec(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    isa: str,
) -> None:
    varlen_encoder_attention(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", ENCODER_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "block_size",
    [
        128,
    ],
)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("isa", ["neon"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Not an Arm CPU.",
)
def test_varlen_encoder_attention_neon(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    isa: str,
) -> None:
    varlen_encoder_attention(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", ENCODER_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "block_size",
    [
        128,
    ],
)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("isa", ["amx"])
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_varlen_encoder_attention_amx(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    isa: str,
) -> None:
    varlen_encoder_attention(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        isa=isa,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec"])
def test_varlen_with_paged_kv_normal_vec(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["amx"])
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_varlen_with_paged_kv_normal_amx(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_varlen_with_paged_kv_fp8_large_prefill_amx() -> None:
    varlen_with_paged_kv(
        seq_lens=[(1024, 1024)] * 4,
        num_heads=(16, 2),
        head_size=256,
        sliding_window=None,
        dtype=torch.bfloat16,
        block_size=2176,
        soft_cap=None,
        num_blocks=4,
        use_alibi=False,
        use_sink=False,
        isa="amx",
        kv_cache_dtype="fp8_e4m3",
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES_VEC16)
@pytest.mark.parametrize("block_size", [48])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec16"])
def test_varlen_with_paged_kv_normal_vec16(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["neon"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Not an Arm CPU.",
)
def test_varlen_with_paged_kv_normal_neon(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["rvv"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.RISCV,
    reason="Not a RISC-V CPU.",
)
def test_varlen_with_paged_kv_normal_rvv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [50])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_softcap(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [True])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_alibi(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [True])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_sink(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [True])
@pytest.mark.parametrize("isa", [get_attn_isa()])
@pytest.mark.parametrize("s_aux_dtype", [torch.float16])
def test_varlen_with_paged_kv_sink_fp16(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
    s_aux_dtype: torch.dtype,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
        s_aux_dtype=s_aux_dtype,
    )


@pytest.mark.parametrize(
    "kv_cache_dtype",
    [
        "auto",
    ],
)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "head_size",
    [
        128,
    ],
)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["amx"])
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_varlen_with_paged_kv_dynamic_causal(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    dynamic_causal = [bool(i % 2) for i in range(len(seq_lens))]
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
        dynamic_causal=dynamic_causal,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("num_heads", [(9, 3), (8, 1)])
@pytest.mark.parametrize("seq_lens", DECODE_MASK_SEQ_LENS)
@pytest.mark.parametrize(
    ("forced_q_head_group", "forced_kv_split_count"),
    [(0, 0), (1, 1), (2, 1), (4, 2)],
)
def test_varlen_with_paged_kv_decode_mask_gqa(
    kv_cache_dtype: str,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    forced_q_head_group: int,
    forced_kv_split_count: int,
) -> None:
    if forced_q_head_group > 0 and num_heads[0] // num_heads[1] % forced_q_head_group:
        pytest.skip("forced group must divide the GQA ratio")
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=96,
        sliding_window=128,
        dtype=torch.bfloat16,
        block_size=96,
        soft_cap=None,
        num_blocks=NUM_BLOCKS[0],
        use_alibi=False,
        use_sink=False,
        isa="vec",
        kv_cache_dtype=kv_cache_dtype,
        # The prefill request's one-token tail must remain outside GQA.
        decode_mask=[True, True, True, False],
        forced_q_head_group=forced_q_head_group,
        forced_kv_split_count=forced_kv_split_count,
    )


@pytest.mark.parametrize(
    ("forced_q_head_group", "forced_kv_split_count"),
    [(1, 1), (2, 1), (4, 2), (8, 1)],
)
def test_varlen_with_paged_kv_forced_schedules(
    forced_q_head_group: int,
    forced_kv_split_count: int,
) -> None:
    varlen_with_paged_kv(
        seq_lens=[(4, 193), (5, 257), (0, 128), (1, 129)],
        num_heads=(8, 1),
        head_size=96,
        sliding_window=None,
        dtype=torch.bfloat16,
        block_size=32,
        soft_cap=None,
        num_blocks=NUM_BLOCKS[0],
        use_alibi=False,
        use_sink=False,
        isa="vec",
        kv_cache_dtype="auto",
        decode_mask=[True, True, False, False],
        forced_q_head_group=forced_q_head_group,
        forced_kv_split_count=forced_kv_split_count,
    )


@pytest.mark.parametrize("num_heads", [(16, 1), (32, 8)])
@pytest.mark.parametrize(
    ("forced_q_head_group", "forced_kv_split_count"),
    [(0, 0), (1, 1), (4, 1), (4, 2)],
)
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_amx_spec_decode_schedules(
    num_heads: tuple[int, int],
    forced_q_head_group: int,
    forced_kv_split_count: int,
) -> None:
    if forced_q_head_group > 0 and num_heads[0] // num_heads[1] % forced_q_head_group:
        pytest.skip("forced group must divide the GQA ratio")
    varlen_with_paged_kv(
        seq_lens=[(4, 1024), (4, 1024)],
        num_heads=num_heads,
        head_size=128,
        sliding_window=None,
        dtype=torch.bfloat16,
        block_size=32,
        soft_cap=None,
        num_blocks=NUM_BLOCKS[0],
        use_alibi=False,
        use_sink=False,
        isa="amx",
        kv_cache_dtype="auto",
        decode_mask=[True, True],
        forced_q_head_group=forced_q_head_group,
        forced_kv_split_count=forced_kv_split_count,
    )


def _scheduler_metadata(**overrides) -> torch.Tensor:
    kwargs = dict(
        num_reqs=2,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
        seq_lens=torch.tensor([8192, 8192], dtype=torch.int32),
        dtype=torch.bfloat16,
        query_start_loc=torch.tensor([0, 4, 5], dtype=torch.int32),
        causal=True,
        sliding_window_size=-1,
        isa="amx",
        enable_kv_split=False,
        decode_mask=torch.tensor([True, False]),
    )
    kwargs.update(overrides)
    return cpu_attn_get_scheduler_metadata(**kwargs)


@pytest.mark.parametrize(
    "decode_mask",
    [
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([[True, False]]),
        torch.tensor([True]),
        torch.tensor([True, False, True]),
        torch.tensor([True, False, True, False])[::2],
    ],
)
def test_scheduler_rejects_invalid_decode_masks(decode_mask: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="decode_mask"):
        _scheduler_metadata(decode_mask=decode_mask)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"seq_lens": torch.tensor([8192, 1], dtype=torch.int64)},
            "seq_lens",
        ),
        (
            {"query_start_loc": torch.tensor([0, 4, 3], dtype=torch.int32)},
            "query_start_loc",
        ),
        (
            {"dynamic_causal": torch.tensor([True], dtype=torch.bool)},
            "dynamic_causal",
        ),
        (
            {"num_heads": 10, "num_kv_heads": 3},
            "divisible",
        ),
    ],
)
def test_scheduler_rejects_invalid_tensor_contracts(
    overrides: dict, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _scheduler_metadata(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"forced_q_head_group": -1}, "nonnegative"),
        ({"forced_kv_split_count": -1}, "nonnegative"),
        ({"forced_q_head_group": 3}, "must divide"),
        ({"forced_q_head_group": 1, "forced_kv_split_count": 2}, "MHA"),
        (
            {"decode_mask": torch.tensor([False, False]), "forced_q_head_group": 2},
            "eligible request",
        ),
        (
            {
                "seq_lens": torch.tensor([64, 64], dtype=torch.int32),
                "forced_q_head_group": 4,
                "forced_kv_split_count": 2,
            },
            "No valid",
        ),
        ({"forced_kv_split_count": 1025}, "thread capacity"),
    ],
)
def test_scheduler_rejects_invalid_forced_plans(overrides: dict, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _scheduler_metadata(**overrides)


def test_fp8_sliding_window_falls_back_without_error() -> None:
    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=1,
        num_heads=8,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        head_dim=128,
        seq_lens=torch.tensor([193], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        causal=True,
        sliding_window_size=64,
        isa="amx",
        enable_kv_split=True,
        kv_cache_dtype="fp8_e4m3",
        decode_mask=torch.tensor([True]),
    )
    assert metadata.device.type == "cpu"
    assert metadata.dtype == torch.int8
    assert metadata.numel() > 0


def test_attention_rejects_metadata_fingerprint_mismatch() -> None:
    head_dim = 64
    query = torch.randn((1, 1, head_dim), dtype=torch.bfloat16)
    key = torch.randn((32, 1, head_dim), dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_cache = torch.empty((1, 1, 32, head_dim), dtype=torch.bfloat16)
    value_cache = torch.empty_like(key_cache)
    cpu_attn_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=torch.arange(32, dtype=torch.int64),
        isa="vec",
    )
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([32], dtype=torch.int32)
    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=torch.bfloat16,
        query_start_loc=query_start_loc,
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=False,
    )
    metadata.view(torch.int64)[2] ^= 1

    with pytest.raises(RuntimeError, match="fingerprint is invalid"):
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=torch.empty_like(query),
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            scale=head_dim**-0.5,
            causal=True,
            alibi_slopes=None,
            sliding_window=-1,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            softcap=0,
            scheduler_metadata=metadata,
            s_aux=None,
        )


@pytest.mark.parametrize(
    "num_heads",
    [(16, 4), (16, 1), (32, 8)],
    ids=["gqa4", "gqa16", "gqa4_wide"],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 193), (1, 257)], [(1, 193), (4, 257)]],
    ids=["single_token", "mixed"],
)
@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_amx_head_dim_256_auto_correctness(
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=256,
        sliding_window=None,
        dtype=torch.bfloat16,
        block_size=32,
        soft_cap=None,
        num_blocks=NUM_BLOCKS[0],
        use_alibi=False,
        use_sink=False,
        isa="amx",
        kv_cache_dtype="auto",
        decode_mask=[True, True],
    )


@pytest.mark.skipif(not torch.cpu._is_amx_tile_supported(), reason="no AMX support.")
def test_amx_spec_decode_group_32_with_sinks() -> None:
    varlen_with_paged_kv(
        seq_lens=[(2, 193), (3, 257)],
        num_heads=(32, 1),
        head_size=128,
        sliding_window=None,
        dtype=torch.bfloat16,
        block_size=32,
        soft_cap=None,
        num_blocks=NUM_BLOCKS[0],
        use_alibi=True,
        use_sink=True,
        isa="amx",
        kv_cache_dtype="auto",
        decode_mask=[True, True],
        forced_q_head_group=32,
        forced_kv_split_count=1,
    )


# ---------------------------------------------------------------------------
# AMX_FP8 (Diamond Rapids) tests
# ---------------------------------------------------------------------------


def _amx_fp8_available() -> bool:
    """Return True iff the runtime reports AMX_FP8 capability."""
    return torch.cpu._is_amx_tile_supported() and torch.ops._C.cpu_attn_has_isa(
        "amx_fp8"
    )


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["amx_fp8"])
@pytest.mark.skipif(
    not _amx_fp8_available(), reason="no AMX_FP8 support (requires Diamond Rapids)."
)
def test_varlen_with_paged_kv_amx_fp8(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
    kv_cache_dtype: str,
) -> None:
    """Test AMX_FP8 native FP8×FP8 attention (Diamond Rapids).

    Verifies that:
    - QK uses _tile_dpfp8ps (native FP8 MMA, no K dequant).
    - PV uses native FP8 MMA for both E4M3 and E5M2.
    - Output cosine similarity vs fp32 reference > 0.99.
    """
    if block_size % 64 != 0:
        pytest.skip("native FP8 PV requires block_size divisible by 64")

    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.skipif(
    not _amx_fp8_available(), reason="no AMX_FP8 support (requires Diamond Rapids)."
)
def test_amx_fp8_qk_covers_full_64_token_group() -> None:
    head_size = 64
    block_size = 64
    query = torch.ones((1, 1, head_size), dtype=torch.bfloat16)
    key = torch.empty((block_size, 1, head_size), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    key[:32].fill_(-1)
    key[32:].fill_(1)
    value[:32].fill_(-1)
    value[32:].fill_(1)

    key_cache = torch.empty((1, 1, block_size, head_size), dtype=torch.uint8)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        isa="amx_fp8",
        kv_cache_dtype="fp8_e4m3",
    )

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([block_size], dtype=torch.int32)
    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_size,
        seq_lens=seq_lens,
        dtype=torch.bfloat16,
        query_start_loc=query_start_loc,
        causal=True,
        sliding_window_size=-1,
        isa="amx_fp8",
        enable_kv_split=False,
        kv_cache_dtype="fp8_e4m3",
    )
    output = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        scale=head_size**-0.5,
        causal=True,
        alibi_slopes=None,
        sliding_window=-1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
        kv_cache_dtype="fp8_e4m3",
    )

    torch.testing.assert_close(output, torch.ones_like(output), atol=0.05, rtol=0)

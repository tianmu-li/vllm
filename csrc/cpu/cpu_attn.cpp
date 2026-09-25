#include "cpu_attn_dispatch_generated.h"

// Runtime check for AMX-FP8, implemented in cpu_isa.cpp.
extern bool runtime_has_amx_fp8();

namespace {

void check_cpu_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

void check_contiguous_1d(const torch::Tensor& tensor, at::ScalarType dtype,
                         int64_t expected_numel, const char* name) {
  check_cpu_tensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() == expected_numel, name, " has the wrong length");
}

void check_int32_sequence(const torch::Tensor& tensor, int64_t expected_numel,
                          const char* name, bool require_zero_start,
                          bool require_monotonic, int32_t expected_end = -1) {
  check_contiguous_1d(tensor, at::ScalarType::Int, expected_numel, name);
  const auto* values = tensor.data_ptr<int32_t>();
  if (expected_numel == 0) {
    return;
  }
  if (require_zero_start) {
    TORCH_CHECK(values[0] == 0, name, " must start at zero");
  }
  for (int64_t i = 0; i < expected_numel; ++i) {
    TORCH_CHECK(values[i] >= 0, name, " must contain nonnegative values");
    if (require_monotonic && i > 0) {
      TORCH_CHECK(values[i] >= values[i - 1], name,
                  " must be monotonically nondecreasing");
    }
  }
  if (expected_end >= 0) {
    TORCH_CHECK(values[expected_numel - 1] == expected_end, name,
                " does not cover the query tensor");
  }
}

void check_bool_vector(const std::optional<torch::Tensor>& tensor,
                       int64_t expected_numel, const char* name) {
  if (tensor.has_value()) {
    check_contiguous_1d(*tensor, at::ScalarType::Bool, expected_numel, name);
  }
}

}  // namespace

// Maps kv_cache_dtype string to Fp8KVCacheDataType enum.
// "auto" -> kAuto(0); "fp8"/"fp8_e4m3" -> kFp8E4M3; "fp8_e5m2" -> kFp8E5M2.
static inline cpu_attention::Fp8KVCacheDataType parse_fp8_kv_dtype(
    const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "fp8_e5m2")
    return cpu_attention::Fp8KVCacheDataType::kFp8E5M2;
  if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8")
    return cpu_attention::Fp8KVCacheDataType::kFp8E4M3;
  if (kv_cache_dtype == "auto") return cpu_attention::Fp8KVCacheDataType::kAuto;
  TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  return cpu_attention::Fp8KVCacheDataType::kAuto;
}

bool cpu_attn_has_isa(const std::string& isa) {
  if (isa == "rvv") {
#if defined(__riscv) && defined(__riscv_v_min_vlen) && \
    (__riscv_v_min_vlen == 128 || __riscv_v_min_vlen == 256)
    return true;
#else
    return false;
#endif
  }
  if (isa == "amx_fp8") {
#ifdef CPU_CAPABILITY_AMXFP8
    // Guard with runtime CPUID check: the binary may be compiled with
    // -mamx-fp8 (CPU_CAPABILITY_AMXFP8 defined) but run on a CPU that
    // has AMX-BF16 without AMX-FP8 (e.g. Sapphire Rapids, Emerald Rapids).
    return runtime_has_amx_fp8();
#else
    return false;
#endif
  }
  return false;
}

torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool causal,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split,
    const std::optional<torch::Tensor>& dynamic_causal,
    const std::string& kv_cache_dtype,
    const std::optional<torch::Tensor>& decode_mask,
    const int64_t forced_q_head_group, const int64_t forced_kv_split_count) {
  TORCH_CHECK(num_req >= 0, "num_req must be nonnegative");
  TORCH_CHECK(num_req <= std::numeric_limits<int32_t>::max(),
              "num_req is too large");
  TORCH_CHECK(num_heads_q > 0, "num_heads_q must be positive");
  TORCH_CHECK(num_heads_kv > 0, "num_heads_kv must be positive");
  TORCH_CHECK(num_heads_q <= std::numeric_limits<int32_t>::max(),
              "num_heads_q is too large");
  TORCH_CHECK(num_heads_kv <= std::numeric_limits<int32_t>::max(),
              "num_heads_kv is too large");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(head_dim > 0, "head_dim must be positive");
  TORCH_CHECK(head_dim <= std::numeric_limits<int32_t>::max(),
              "head_dim is too large");
  TORCH_CHECK(window_size == -1 || window_size > 0,
              "window_size must be -1 or positive");
  TORCH_CHECK(window_size <= std::numeric_limits<int32_t>::max(),
              "window_size is too large");
  TORCH_CHECK(forced_q_head_group >= 0 &&
                  forced_q_head_group <= std::numeric_limits<int32_t>::max(),
              "forced_q_head_group must be nonnegative and fit int32");
  TORCH_CHECK(forced_kv_split_count >= 0 &&
                  forced_kv_split_count <= std::numeric_limits<int32_t>::max(),
              "forced_kv_split_count must be nonnegative and fit int32");
  TORCH_CHECK(dtype == at::ScalarType::Float || dtype == at::ScalarType::Half ||
                  dtype == at::ScalarType::BFloat16,
              "Unsupported CPU attention dtype: ", dtype);

  check_int32_sequence(seq_lens, num_req, "seq_lens", false, false);
  check_int32_sequence(query_start_loc, num_req + 1, "query_start_loc", true,
                       true);
  check_bool_vector(dynamic_causal, num_req, "dynamic_causal");
  check_bool_vector(decode_mask, num_req, "decode_mask");

  cpu_attention::ISA isa;
  if (isa_hint == "amx") {
    isa = cpu_attention::ISA::AMX;
  } else if (isa_hint == "amx_fp8") {
    isa = cpu_attention::ISA::AMX_FP8;
  } else if (isa_hint == "vec") {
    isa = cpu_attention::ISA::VEC;
  } else if (isa_hint == "vec16") {
    isa = cpu_attention::ISA::VEC16;
  } else if (isa_hint == "neon") {
    isa = cpu_attention::ISA::NEON;
  } else if (isa_hint == "vxe") {
    isa = cpu_attention::ISA::VXE;
  } else if (isa_hint == "rvv") {
    isa = cpu_attention::ISA::RVV;
  } else if (isa_hint == "vsx") {
    isa = cpu_attention::ISA::VSX;
  } else {
    TORCH_CHECK(false, "Unsupported CPU attention ISA hint: " + isa_hint);
  }

  cpu_attention::AttentionScheduler::ScheduleInput input;
  input.num_reqs = static_cast<int32_t>(num_req);
  input.num_heads_q = static_cast<int32_t>(num_heads_q);
  input.num_heads_kv = static_cast<int32_t>(num_heads_kv);
  input.head_dim = static_cast<int32_t>(head_dim);
  input.query_start_loc = query_start_loc.data_ptr<int32_t>();
  input.seq_lens = seq_lens.data_ptr<int32_t>();

  input.sliding_window_size = window_size;
  input.causal = causal;
  input.isa = isa;
  input.enable_kv_split = enable_kv_split;
  input.dynamic_causal =
      dynamic_causal.has_value() ? dynamic_causal->data_ptr<bool>() : nullptr;
  input.decode_mask =
      decode_mask.has_value() ? decode_mask->data_ptr<bool>() : nullptr;

  const int64_t kv_cache_idx =
      static_cast<int64_t>(parse_fp8_kv_dtype(kv_cache_dtype));
  input.fp8_kv_cache = kv_cache_idx != 0;
  input.dtype = static_cast<int32_t>(dtype);
  input.kv_cache_dtype = static_cast<int32_t>(kv_cache_idx);
  input.forced_q_head_group = forced_q_head_group;
  input.forced_kv_split_count = forced_kv_split_count;
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "get_scheduler_metadata", [&]() {
    CPU_ATTN_DISPATCH(head_dim, isa, kv_cache_idx, [&]() {
      input.elem_size = sizeof(attn_impl::kv_cache_t);
      input.q_buffer_elem_size = sizeof(attn_impl::q_buffer_t);
      input.logits_buffer_elem_size = sizeof(attn_impl::logits_buffer_t);
      input.output_buffer_elem_size =
          sizeof(attn_impl::partial_output_buffer_t);
      input.max_num_q_per_iter = attn_impl::MaxQHeadNumPerIteration;
      input.kv_block_alignment = attn_impl::BlockSizeAlignment;
    });
  });

  cpu_attention::AttentionScheduler scheduler;
  torch::Tensor metadata = scheduler.schedule(input);
  return metadata;
}

void cpu_attn_reshape_and_cache(
    const torch::Tensor& key,    // [token_num, head_num, head_size]
    const torch::Tensor& value,  // [token_num, head_num, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    const torch::Tensor& slot_mapping, const std::string& isa,
    const double k_scale = 1.0, const double v_scale = 1.0,
    const std::string& kv_cache_dtype = "auto") {
  check_cpu_tensor(key, "key");
  check_cpu_tensor(value, "value");
  check_cpu_tensor(key_cache, "key_cache");
  check_cpu_tensor(value_cache, "value_cache");
  TORCH_CHECK(key.dim() == 3, "key must be rank 3");
  TORCH_CHECK(value.dim() == 3, "value must be rank 3");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be rank 4");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be rank 4");
  TORCH_CHECK(key.stride(2) == 1, "key head dimension must be contiguous");
  TORCH_CHECK(value.stride(2) == 1, "value head dimension must be contiguous");
  TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "key and value dtypes must match");
  check_contiguous_1d(slot_mapping, at::ScalarType::Long, key.size(0),
                      "slot_mapping");
  TORCH_CHECK(
      key.scalar_type() == at::ScalarType::Float ||
          key.scalar_type() == at::ScalarType::Half ||
          key.scalar_type() == at::ScalarType::BFloat16,
      "Unsupported CPU attention cache input dtype: ", key.scalar_type());

  const int64_t kv_cache_idx =
      static_cast<int64_t>(parse_fp8_kv_dtype(kv_cache_dtype));
  const bool is_fp8 = (kv_cache_idx != 0);

  if (is_fp8) {
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Byte,
                "key_cache must be uint8 for FP8 path");
    TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::Byte,
                "value_cache must be uint8 for FP8 path");
    TORCH_CHECK(k_scale > 0, "k_scale must be positive for FP8 path");
    TORCH_CHECK(v_scale > 0, "v_scale must be positive for FP8 path");
  } else {
    TORCH_CHECK(key_cache.scalar_type() == key.scalar_type(),
                "key_cache dtype must match key dtype");
    TORCH_CHECK(value_cache.scalar_type() == key.scalar_type(),
                "value_cache dtype must match key dtype");
  }

  const float k_inv = is_fp8 ? 1.0f / static_cast<float>(k_scale) : 0.0f;
  const float v_inv = is_fp8 ? 1.0f / static_cast<float>(v_scale) : 0.0f;

  const int64_t token_num = key.size(0);
  const int64_t head_num = key.size(1);
  const int64_t head_dim = key.size(2);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_blocks_stride = key_cache.stride(0);
  const int64_t cache_head_num_stride = key_cache.stride(1);
  const int64_t block_size = key_cache.size(2);
  const int64_t block_size_stride = key_cache.stride(2);
  TORCH_CHECK(token_num > 0, "key must contain at least one token");
  TORCH_CHECK(head_num > 0, "key must contain at least one head");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "key and value cache shapes must match");
  TORCH_CHECK(key_cache.size(1) == head_num,
              "cache and input head counts must match");
  TORCH_CHECK(key_cache.size(3) == head_dim,
              "cache and input head dimensions must match");
  TORCH_CHECK(num_blocks > 0, "cache must contain at least one block");
  TORCH_CHECK(block_size > 0, "cache block size must be positive");
  const auto* slots = slot_mapping.data_ptr<int64_t>();
  for (int64_t i = 0; i < token_num; ++i) {
    TORCH_CHECK(
        slots[i] == -1 || (slots[i] >= 0 && slots[i] < num_blocks * block_size),
        "slot_mapping contains an out-of-range slot");
  }

  cpu_attention::ISA isa_tag = [&]() {
    if (isa == "amx") {
      return cpu_attention::ISA::AMX;
    } else if (isa == "amx_fp8") {
      return cpu_attention::ISA::AMX_FP8;
    } else if (isa == "vec") {
      return cpu_attention::ISA::VEC;
    } else if (isa == "vec16") {
      return cpu_attention::ISA::VEC16;
    } else if (isa == "neon") {
      return cpu_attention::ISA::NEON;
    } else if (isa == "vxe") {
      return cpu_attention::ISA::VXE;
    } else if (isa == "rvv") {
      return cpu_attention::ISA::RVV;
    } else if (isa == "vsx") {
      return cpu_attention::ISA::VSX;
    } else {
      TORCH_CHECK(false, "Invalid ISA type: " + isa);
    }
  }();

  if (is_fp8) {
    TORCH_CHECK(isa_tag == cpu_attention::ISA::AMX ||
                    isa_tag == cpu_attention::ISA::AMX_FP8 ||
                    isa_tag == cpu_attention::ISA::VEC,
                "FP8 KV cache is only supported on x86 (AMX_FP8/AMX/VEC) ISA");
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "cpu_attn_reshape_and_cache", [&]() {
        CPU_ATTN_DISPATCH(head_dim, isa_tag, kv_cache_idx, [&]() {
          using kv_t = typename attn_impl::kv_cache_t;
          attn_impl::reshape_and_cache(
              key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
              reinterpret_cast<kv_t*>(key_cache.data_ptr()),
              reinterpret_cast<kv_t*>(value_cache.data_ptr()),
              slot_mapping.data_ptr<int64_t>(), token_num, key.stride(0),
              value.stride(0), head_num, key.stride(1), value.stride(1),
              num_blocks, num_blocks_stride, cache_head_num_stride, block_size,
              block_size_stride, k_inv, v_inv);
        });
      });
}

void cpu_attention_with_kv_cache(
    const torch::Tensor& query,  // [num_tokens, num_heads, head_size]
    const torch::Tensor&
        key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    const torch::Tensor&
        value_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor& output,  // [num_tokens, num_heads, head_size]
    const torch::Tensor& query_start_loc,  // [num_tokens + 1]
    const torch::Tensor& seq_lens,         // [num_tokens]
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,  // [num_heads]
    const int64_t sliding_window,
    const torch::Tensor& block_table,  // [num_tokens, max_block_num]
    const double softcap, const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux,           // [num_heads]
    const std::optional<torch::Tensor>& dynamic_causal,  // [num_reqs]
    const double k_scale = 1.0, const double v_scale = 1.0,
    const std::string& kv_cache_dtype = "auto") {
  const int64_t kv_cache_idx =
      static_cast<int64_t>(parse_fp8_kv_dtype(kv_cache_dtype));
  const bool is_fp8 = (kv_cache_idx != 0);
  TORCH_CHECK(sliding_window == -1 || sliding_window > 0,
              "sliding_window must be -1 or positive");
  check_cpu_tensor(query, "query");
  check_cpu_tensor(key_cache, "key_cache");
  check_cpu_tensor(value_cache, "value_cache");
  check_cpu_tensor(output, "output");
  check_cpu_tensor(block_table, "block_table");
  check_cpu_tensor(scheduler_metadata, "scheduler_metadata");
  TORCH_CHECK(query.dim() == 3, "query must be rank 3");
  TORCH_CHECK(query.stride(2) == 1, "query head dimension must be contiguous");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Float ||
                  query.scalar_type() == at::ScalarType::Half ||
                  query.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported CPU attention query dtype: ", query.scalar_type());
  TORCH_CHECK(output.dim() == 3, "output must be rank 3");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output shape must match query shape");
  TORCH_CHECK(output.scalar_type() == query.scalar_type(),
              "output dtype must match query dtype");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be rank 4");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be rank 4");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "key and value cache shapes must match");
  TORCH_CHECK_EQ(key_cache.size(2), value_cache.size(2));
  TORCH_CHECK(key_cache.size(3) == query.size(2),
              "cache head dimension must match query");
  TORCH_CHECK(key_cache.size(0) > 0, "cache must contain blocks");
  TORCH_CHECK(key_cache.size(1) > 0, "cache must contain KV heads");
  TORCH_CHECK(key_cache.size(2) > 0, "cache block size must be positive");
  TORCH_CHECK(query.size(1) > 0, "query must contain heads");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "cache head dimension must be contiguous");
  if (is_fp8) {
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Byte,
                "key_cache must be uint8 for FP8 path");
    TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::Byte,
                "value_cache must be uint8 for FP8 path");
    TORCH_CHECK(k_scale > 0, "k_scale must be positive for FP8 path");
    TORCH_CHECK(v_scale > 0, "v_scale must be positive for FP8 path");
  } else {
    TORCH_CHECK(key_cache.scalar_type() == query.scalar_type(),
                "key_cache dtype must match query dtype");
    TORCH_CHECK(value_cache.scalar_type() == query.scalar_type(),
                "value_cache dtype must match query dtype");
  }

  check_cpu_tensor(query_start_loc, "query_start_loc");
  TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1-D");
  TORCH_CHECK(query_start_loc.is_contiguous(),
              "query_start_loc must be contiguous");
  TORCH_CHECK(query_start_loc.numel() >= 1,
              "query_start_loc must contain a sentinel");
  const int64_t num_reqs = query_start_loc.numel() - 1;
  TORCH_CHECK(num_reqs <= std::numeric_limits<int32_t>::max(),
              "query_start_loc has too many requests");
  TORCH_CHECK(query.size(0) <= std::numeric_limits<int32_t>::max(),
              "query has too many tokens");
  check_int32_sequence(query_start_loc, num_reqs + 1, "query_start_loc", true,
                       true, query.size(0));
  check_int32_sequence(seq_lens, num_reqs, "seq_lens", false, false);
  check_bool_vector(dynamic_causal, num_reqs, "dynamic_causal");
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int,
              "block_table must be int32");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be rank 2");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(block_table.size(0) == num_reqs,
              "block_table must have one row per request");
  TORCH_CHECK(block_table.size(1) > 0,
              "block_table must contain at least one block column");
  const auto* seq_values = seq_lens.data_ptr<int32_t>();
  const int64_t block_size = key_cache.size(2);
  for (int64_t req_id = 0; req_id < num_reqs; ++req_id) {
    const int64_t required_blocks =
        (static_cast<int64_t>(seq_values[req_id]) + block_size - 1) /
        block_size;
    TORCH_CHECK(required_blocks <= block_table.size(1),
                "block_table is too short for seq_lens");
  }
  const auto* block_values = block_table.data_ptr<int32_t>();
  for (int64_t i = 0; i < block_table.numel(); ++i) {
    TORCH_CHECK(block_values[i] >= 0 && block_values[i] < key_cache.size(0),
                "block_table contains an out-of-range block index");
  }
  if (alibi_slopes.has_value()) {
    check_contiguous_1d(*alibi_slopes, at::ScalarType::Float, query.size(1),
                        "alibi_slopes");
  }
  if (s_aux.has_value()) {
    check_cpu_tensor(*s_aux, "s_aux");
    TORCH_CHECK(s_aux->dim() == 1, "s_aux must be one-dimensional");
    TORCH_CHECK(s_aux->is_contiguous(), "s_aux must be contiguous");
    TORCH_CHECK(s_aux->numel() == query.size(1),
                "s_aux must contain one value per query head");
    TORCH_CHECK(s_aux->scalar_type() == at::ScalarType::BFloat16 ||
                    s_aux->scalar_type() == at::ScalarType::Float,
                "s_aux must be bfloat16 or float32");
  }

  TORCH_CHECK(scheduler_metadata.scalar_type() == at::ScalarType::Char,
              "scheduler_metadata must be int8");
  TORCH_CHECK(scheduler_metadata.dim() == 1,
              "scheduler_metadata must be one-dimensional");
  TORCH_CHECK(scheduler_metadata.is_contiguous(),
              "scheduler_metadata must be contiguous");
  TORCH_CHECK(
      scheduler_metadata.numel() >=
          static_cast<int64_t>(sizeof(cpu_attention::AttentionMetadata)),
      "scheduler_metadata is truncated");
  auto* metadata = reinterpret_cast<cpu_attention::AttentionMetadata*>(
      scheduler_metadata.data_ptr());
  TORCH_CHECK(metadata->magic == cpu_attention::AttentionMetadata::kMagic,
              "scheduler_metadata has an invalid magic");
  TORCH_CHECK(metadata->version == cpu_attention::AttentionMetadata::kVersion,
              "scheduler_metadata has an unsupported version");
  TORCH_CHECK(
      metadata->metadata_size == sizeof(cpu_attention::AttentionMetadata),
      "scheduler_metadata has an incompatible layout");
  TORCH_CHECK(metadata->num_reqs >= 0);
  TORCH_CHECK(metadata->num_tokens >= 0);
  TORCH_CHECK(metadata->num_heads_q >= 1);
  TORCH_CHECK(metadata->num_heads_kv >= 1);
  TORCH_CHECK(metadata->head_dim >= 1);
  TORCH_CHECK(metadata->workitem_group_num >= 0);
  TORCH_CHECK(metadata->reduction_item_num >= 0);
  TORCH_CHECK(
      metadata->workitem_group_num <=
      (scheduler_metadata.numel() - sizeof(cpu_attention::AttentionMetadata)) /
          sizeof(cpu_attention::AttentionWorkItemGroup));
  const int64_t remaining_metadata =
      scheduler_metadata.numel() - sizeof(cpu_attention::AttentionMetadata) -
      static_cast<int64_t>(metadata->workitem_group_num) *
          sizeof(cpu_attention::AttentionWorkItemGroup);
  TORCH_CHECK(metadata->reduction_item_num <=
              remaining_metadata /
                  sizeof(cpu_attention::ReductionWorkItemGroup));
  const int64_t metadata_size =
      sizeof(cpu_attention::AttentionMetadata) +
      static_cast<int64_t>(metadata->workitem_group_num) *
          sizeof(cpu_attention::AttentionWorkItemGroup) +
      static_cast<int64_t>(metadata->reduction_item_num) *
          sizeof(cpu_attention::ReductionWorkItemGroup);
  TORCH_CHECK(metadata_size <= scheduler_metadata.numel(),
              "scheduler_metadata does not contain its work items");
  TORCH_CHECK(static_cast<int32_t>(metadata->isa) >=
                      static_cast<int32_t>(cpu_attention::ISA::AMX) &&
                  static_cast<int32_t>(metadata->isa) <=
                      static_cast<int32_t>(cpu_attention::ISA::AMX_FP8),
              "scheduler_metadata has an invalid ISA");
  TORCH_CHECK(metadata->num_reqs == num_reqs,
              "scheduler_metadata request count does not match inputs");
  TORCH_CHECK(metadata->num_tokens == query.size(0),
              "scheduler_metadata token count does not match query");
  TORCH_CHECK(metadata->num_heads_q == query.size(1),
              "scheduler_metadata query head count does not match query");
  TORCH_CHECK(metadata->num_heads_kv == key_cache.size(1),
              "scheduler_metadata KV head count does not match cache");
  TORCH_CHECK(metadata->head_dim == query.size(2),
              "scheduler_metadata head dimension does not match query");
  TORCH_CHECK(metadata->dtype == static_cast<int32_t>(query.scalar_type()),
              "scheduler_metadata query dtype does not match query");
  TORCH_CHECK(metadata->kv_cache_dtype == static_cast<int32_t>(kv_cache_idx),
              "scheduler_metadata KV dtype does not match cache");
  TORCH_CHECK(
      metadata->contract_fingerprint ==
          cpu_attention::AttentionMetadata::make_contract_fingerprint(
              metadata->num_reqs, metadata->num_tokens, metadata->num_heads_q,
              metadata->num_heads_kv, metadata->head_dim, metadata->dtype,
              metadata->kv_cache_dtype, metadata->isa),
      "scheduler_metadata fingerprint is invalid");

  cpu_attention::AttentionInput input;
  input.metadata = reinterpret_cast<cpu_attention::AttentionMetadata*>(
      scheduler_metadata.data_ptr());
  input.num_tokens = query.size(0);
  input.num_heads = query.size(1);
  input.num_kv_heads = key_cache.size(1);
  input.block_size = key_cache.size(2);
  input.query = query.data_ptr();
  input.query_num_tokens_stride = query.stride(0);
  input.query_num_heads_stride = query.stride(1);
  input.cache_num_blocks_stride = key_cache.stride(0);
  input.cache_num_kv_heads_stride = key_cache.stride(1);
  input.blt_num_tokens_stride = block_table.stride(0);
  input.key_cache = key_cache.data_ptr();
  input.value_cache = value_cache.data_ptr();
  input.output = output.data_ptr();
  input.query_start_loc = query_start_loc.data_ptr<int32_t>();
  input.seq_lens = seq_lens.data_ptr<int32_t>();
  input.block_table = block_table.data_ptr<int32_t>();
  input.alibi_slopes =
      alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr;
  // Attention sinks may be bf16 (native path) or fp32. Anything that is not
  // bf16 must be provided as fp32 and is executed in full float precision.
  if (s_aux.has_value()) {
    TORCH_CHECK(s_aux->scalar_type() == at::ScalarType::BFloat16 ||
                    s_aux->scalar_type() == at::ScalarType::Float,
                "cpu_attention_with_kv_cache: s_aux (attention sinks) dtype ",
                s_aux->scalar_type(), " must be bfloat16 or float32");
    input.s_aux = s_aux->data_ptr();
    input.s_aux_is_bf16 = s_aux->scalar_type() == at::ScalarType::BFloat16;
  } else {
    input.s_aux = nullptr;
    input.s_aux_is_bf16 = false;
  }
  input.dynamic_causal =
      dynamic_causal.has_value() ? dynamic_causal->data_ptr<bool>() : nullptr;
  input.scale = scale;
  input.causal = causal;
  input.sliding_window_size = sliding_window;
  input.softcap = static_cast<float>(softcap);

  if (is_fp8) {
    input.k_scale_fp8 = static_cast<float>(k_scale);
    input.v_scale_fp8 = static_cast<float>(v_scale);
    TORCH_CHECK(input.metadata->isa == cpu_attention::ISA::AMX ||
                    input.metadata->isa == cpu_attention::ISA::AMX_FP8 ||
                    input.metadata->isa == cpu_attention::ISA::VEC,
                "FP8 KV cache is only supported on x86 (AMX_FP8/AMX/VEC) ISA");
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "cpu_attention_with_kv_cache", [&]() {
        CPU_ATTN_DISPATCH(
            query.size(2), input.metadata->isa, kv_cache_idx, [&]() {
              TORCH_CHECK_EQ(input.block_size % attn_impl::BlockSizeAlignment,
                             0);
              cpu_attention::AttentionMainLoop<attn_impl> mainloop;
              mainloop(&input);
            });
      });
}

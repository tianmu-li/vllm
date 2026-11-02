// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "dense_scheduler.h"
#include "dense_amx_plan.h"
#include "../utils.hpp"
#include "gemm.h"
#include "vec.h"
#include <cstdio>

#if defined(__AMX_INT8__) && defined(__linux__)
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

template <typename scalar_t, bool has_bias, int BLOCK_N, bool stream_output = false>
struct scale_C {
  static inline void apply(
      scalar_t* __restrict__ C,
      const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      float As,
      const float* __restrict__ Bs) {
    TORCH_CHECK(false, "scale_C: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_N, bool stream_output>
struct scale_C<at::BFloat16, has_bias, BLOCK_N, stream_output> {
  // Scales ROWS rows of one 32-column panel, reusing the column operands.
  template <int ROWS>
  static inline void apply_rows(
      at::BFloat16* __restrict__ C, const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp, const float* __restrict__ bias,
      const float* __restrict__ As, const float* __restrict__ Bs,
      int64_t ldc, int64_t cs) {
    static_assert(BLOCK_N == 32);
    const __m512 bs0 = _mm512_loadu_ps(Bs);
    const __m512 bs1 = _mm512_loadu_ps(Bs + 16);
    const __m512i comp0 = _mm512_loadu_si512(Bcomp);
    const __m512i comp1 = _mm512_loadu_si512(Bcomp + 16);
    __m512 bias0, bias1;
    if constexpr (has_bias) {
      bias0 = _mm512_loadu_ps(bias);
      bias1 = _mm512_loadu_ps(bias + 16);
    }
    Unroll<ROWS>{}([&](auto row) {
      const __m512 as = _mm512_set1_ps(As[row]);
      __m512 lo = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_loadu_si512(Ctmp + row * cs), comp0));
      __m512 hi = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_loadu_si512(Ctmp + row * cs + 16), comp1));
      if constexpr (has_bias) {
        lo = _mm512_fmadd_ps(_mm512_mul_ps(lo, as), bs0, bias0);
        hi = _mm512_fmadd_ps(_mm512_mul_ps(hi, as), bs1, bias1);
      } else {
        lo = _mm512_mul_ps(_mm512_mul_ps(lo, as), bs0);
        hi = _mm512_mul_ps(_mm512_mul_ps(hi, as), bs1);
      }
      const __m512i result = (__m512i)_mm512_cvtne2ps_pbh(hi, lo);
      if constexpr (stream_output) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(C + row * ldc), result);
      } else {
        _mm512_storeu_si512(C + row * ldc, result);
      }
    });
  }

  static inline void apply(
      at::BFloat16* __restrict__ C,
      const int32_t* __restrict__ Ctmp,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      float As,
      const float* __restrict__ Bs) {
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512 vc[COLS];
    __m512 vd0 = _mm512_set1_ps(As);

    auto compute = [&](auto col) {
      __m512 vd1 = _mm512_loadu_ps(Bs + col * 16);
      __m512i vcomp = _mm512_loadu_si512(Bcomp + col * 16);
      __m512i vc32 = _mm512_loadu_si512(Ctmp + col * 16);
      vc[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32, vcomp));
      if constexpr (has_bias) {
        __m512 vbias = _mm512_loadu_ps(bias + col * 16);
        vc[col] = _mm512_fmadd_ps(_mm512_mul_ps(vc[col], vd0), vd1, vbias);
      } else {
        vc[col] = _mm512_mul_ps(_mm512_mul_ps(vc[col], vd0), vd1);
      }
    };
    Unroll<COLS>{}(compute);

    auto storec = [&](auto col) {
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        const __m512i result = (__m512i)(_mm512_cvtne2ps_pbh(vc[col + 1], vc[col + 0]));
        if constexpr (stream_output) {
          _mm512_stream_si512(reinterpret_cast<__m512i*>(C + col * 16), result);
        } else {
          _mm512_storeu_si512(reinterpret_cast<__m512i*>(C + col * 16), result);
        }
      }
    };
    Unroll<COLS>{}(storec);
  }
};
#endif

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512i va;
    __m512i vb[COLS];
    __m512i vc[ROWS * COLS];
    __m512i vcomp[COLS];
    __m512 vd0;
    __m512 vd1[COLS];

    // oops! 4x4 spills but we use 4x2
    __m512 vbias[COLS];

    // [NOTE]: s8s8 igemm compensation in avx512-vnni
    //
    // avx512-vnni has no s8s8, so we need to change s8s8 to u8s8 with compensate:
    //
    //   a * b = (a + 128) * b - 128 * b
    //   s   s       u       s    u    s
    //
    // 1) 128 * b is pre-computed when packing B to vnni formats
    // 2) a + 128 is fused when dynamically quantize A
    //
    auto loadc = [&](auto i) { vc[i] = _mm512_set1_epi32(0); };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb;  // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb[col] = _mm512_loadu_si512(b_ptr + k * ldb4 + col * 16);
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb4 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr (col == 0) {
        vd0 = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp per 2 vectors
      // also load bias if any
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          vd1[col + 0] = _mm512_loadu_ps(Bs + col * 16);
          vd1[col + 1] = _mm512_loadu_ps(Bs + col * 16 + 16);
          vcomp[col + 0] = _mm512_loadu_si512(Bcomp + col * 16);
          vcomp[col + 1] = _mm512_loadu_si512(Bcomp + col * 16 + 16);
          if constexpr (has_bias) {
            vbias[col + 0] = _mm512_loadu_ps(bias + col * 16);
            vbias[col + 1] = _mm512_loadu_ps(bias + col * 16 + 16);
          }
        }
      }

      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 0], vcomp[col + 0]));
        __m512 vc1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 1], vcomp[col + 1]));
        if constexpr (has_bias) {
          vc0 = _mm512_fmadd_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0], vbias[col + 0]);
          vc1 = _mm512_fmadd_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1], vbias[col + 1]);
        } else {
          vc0 = _mm512_mul_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0]);
          vc1 = _mm512_mul_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1]);
        }

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)), (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * 4,                                            \
      C + mb_start * ldc + nb_start,                               \
      As + mb_start,                                               \
      Bs + nb_start,                                               \
      Bcomp + nb_start,                                            \
      has_bias ? bias + nb_start : nullptr,                        \
      K,                                                           \
      lda,                                                         \
      ldb,                                                         \
      ldc);

template <typename scalar_t, bool has_bias, bool stream_output = false>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  // B compensation
  const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * K);

  if (brg) {
    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // apply compensation and scale
    for (int64_t m = 0; m < M; ++m) {
      scale_C<scalar_t, has_bias, BLOCK_N, stream_output>::apply(C + m * ldc, Ctmp + m * BLOCK_N, Bcomp, bias, As[m], Bs);
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t, typename execution_t>
void int8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    const execution_t& execute) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<int8_t>(M);

  // K + 4 after compensation
  const int64_t packed_row_size = get_row_size<int8_t>(K);

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    execute(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      // for brgemm, use int32_t for accumulate
      alignas(64) int32_t Ctmp[BLOCK_M * BLOCK_N];

      loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        int mb_start = mb * BLOCK_M;
        int mb_size = std::min(M - mb_start, BLOCK_M);
        int nb_start = nb * BLOCK_N;
        int nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ mat1 + mb_start * K,
            /*   B */ mat2 + nb_start * packed_row_size /* nb * BLOCK_N * (K + 4) */,
            /*   C */ out + mb_start * N + nb_start,
            /* Ctmp*/ Ctmp,
            /*  As */ scales1 + mb_start,
            /*  Bs */ scales2 + nb_start,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ nb_size,
            /* ldc */ N,
            /* brg */ use_brgemm);
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

#if defined(__AMX_INT8__) && defined(__linux__)
bool amx_int8_enabled() {
  // Linux grants dynamic tile-state permission to the whole process.
  static const bool enabled = [] {
    unsigned int eax, ebx, ecx, edx;
    constexpr unsigned int capabilities = bit_AMX_TILE | bit_AMX_INT8;
    constexpr int request_xcomp_permission = 0x1023;
    constexpr int tile_data_component = 18;
    return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) &&
        (edx & capabilities) == capabilities &&
        syscall(SYS_arch_prctl, request_xcomp_permission, tile_data_component) == 0;
  }();
  return enabled;
}

struct alignas(64) Int8TileConfig {
  uint8_t palette = 1, start_row = 0, reserved[14] = {};
  uint16_t columns[16] = {};
  uint8_t rows[16] = {};

  Int8TileConfig() {
    for (int i = 0; i < 8; ++i) {
      columns[i] = 64;
      rows[i] = 16;
    }
  }
};

inline float bf16_row_amax(const at::BFloat16* __restrict__ A, int64_t K) {
  constexpr float eps = 1e-7;
  const __m512 signBit = _mm512_set1_ps(-0.0f);

  // K is 32x, no remainder
  float amax = 0.f;
  __m512 vamax0 = _mm512_set1_ps(0.f);
  __m512 vamax1 = _mm512_set1_ps(0.f);
  for (int64_t k = 0; k < K; k += 32) {
    __m512i va = _mm512_loadu_si512((void*)(A + k));
    __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
    __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
    vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, va0));
    vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, va1));
  }
  amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
  amax = std::max(amax, eps);
  return amax;
}

inline void bf16_quant_from_amax(
    uint8_t* __restrict__ Aq, float& As,
    const at::BFloat16* __restrict__ A, float amax, int64_t K) {
  const __m512i off = _mm512_set1_epi32(128);
  const float scale = amax / 127;
  const float inv_scale = 127 / amax;
  const __m512 vd = _mm512_set1_ps(inv_scale);

  for (int64_t k = 0; k < K; k += 32) {
    __m512i va = _mm512_loadu_si512((void*)(A + k));
    __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
    __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
    va0 = _mm512_mul_ps(va0, vd);
    va1 = _mm512_mul_ps(va1, vd);
    va0 = _mm512_roundscale_ps(va0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    va1 = _mm512_roundscale_ps(va1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
    __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Aq + k), _mm256_set_m128i(i1, i0));
  }
  As = scale;
}

template <bool has_bias, bool has_pending, bool stream_a, bool stream_output,
    bool distribute_pending = false, int depth = 0>
__attribute__((always_inline)) inline void dense_amx_tile_impl(
    const DenseAmxPlan& plan, const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B, int32_t* __restrict__ Ctmp,
    int64_t k_begin, int64_t k_end,
    at::BFloat16* __restrict__ pending_out,
    const float* __restrict__ pending_As, const float* __restrict__ pending_Bs,
    const float* __restrict__ pending_bias,
    const int32_t* __restrict__ pending_comp) {
  const int64_t cs = plan.c_stride();
  const int64_t a_stride = plan.a_stride;
  const int64_t output_stride = plan.output_stride;
  if (k_begin == 0) {
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
  } else {
    _tile_loadd(0, Ctmp, cs * sizeof(int32_t));
    _tile_loadd(1, Ctmp + 16, cs * sizeof(int32_t));
    _tile_loadd(2, Ctmp + 16 * cs, cs * sizeof(int32_t));
    _tile_loadd(3, Ctmp + 16 * cs + 16, cs * sizeof(int32_t));
  }
  int64_t pending_row = 0;
  const int64_t end = depth ? k_begin + depth : k_end;
  const int64_t pending_per_step = depth ? 2048 / depth : distribute_pending
      ? std::max<int64_t>(1, 2048 / (k_end - k_begin)) : 2;
  auto consume = [&]() __attribute__((always_inline)) {
    if constexpr (has_pending) {
      if (pending_row < 32) {
        scale_C<at::BFloat16, has_bias, 32, stream_output>::apply(
            pending_out + pending_row * output_stride, Ctmp + pending_row * cs,
            pending_comp, pending_bias, pending_As[pending_row], pending_Bs);
        ++pending_row;
      }
    }
  };
  auto step = [&](int64_t k) __attribute__((always_inline)) {
    if constexpr (stream_a) {
      _tile_stream_loadd(4, A + k, a_stride);
    } else {
      _tile_loadd(4, A + k, a_stride);
    }
    _tile_loadd(6, B + k * 32, 128);
    _tile_dpbusd(0, 4, 6);
    if constexpr (stream_a) {
      _tile_stream_loadd(5, A + 16 * a_stride + k, a_stride);
    } else {
      _tile_loadd(5, A + 16 * a_stride + k, a_stride);
    }
    _tile_dpbusd(2, 5, 6);
    _tile_loadd(7, B + k * 32 + 64, 128);
    _tile_dpbusd(1, 4, 7);
    _tile_dpbusd(3, 5, 7);
    if constexpr (has_pending && depth != 0) {
      constexpr int ROWS = 2048 / depth;
      const int64_t row = (k - k_begin) / 64 * ROWS;
      scale_C<at::BFloat16, has_bias, 32, stream_output>::template apply_rows<ROWS>(
          pending_out + row * output_stride, Ctmp + row * cs, pending_comp,
          pending_bias, pending_As + row, pending_Bs, output_stride, cs);
    } else if constexpr (has_pending && distribute_pending) {
      const int64_t target = std::min<int64_t>(32, pending_row + pending_per_step);
      while (pending_row < target) {
        consume();
      }
    } else {
      consume();
      consume();
    }
  };
  int64_t k = k_begin;
  for (; k + 256 <= end; k += 256) {
    Unroll<4>{}([&](auto u) { step(k + u * 64); });
  }
  for (; k < end; k += 64) {
    step(k);
  }
  // Short K blocks still consume every pending row before reusing its slot.
  if constexpr (has_pending && depth == 0) {
    while (pending_row < 32) {
      consume();
    }
  }
  _tile_stored(0, Ctmp, cs * sizeof(int32_t));
  _tile_stored(1, Ctmp + 16, cs * sizeof(int32_t));
  _tile_stored(2, Ctmp + 16 * cs, cs * sizeof(int32_t));
  _tile_stored(3, Ctmp + 16 * cs + 16, cs * sizeof(int32_t));
}

template <bool has_bias, bool has_pending, bool stream_a, bool stream_output,
    bool distribute_pending>
__attribute__((noinline)) void dense_amx_tile(
    const DenseAmxPlan& plan, const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B, int32_t* __restrict__ Ctmp,
    int64_t k_begin, int64_t k_end, at::BFloat16* __restrict__ pending_out,
    const float* __restrict__ pending_As, const float* __restrict__ pending_Bs,
    const float* __restrict__ pending_bias, const int32_t* __restrict__ pending_comp) {
  dense_amx_tile_impl<has_bias, has_pending, stream_a, stream_output, distribute_pending>(
      plan, A, B, Ctmp, k_begin, k_end, pending_out, pending_As, pending_Bs,
      pending_bias, pending_comp);
}

// With local, the worker owns its whole plan and pending rows are spread over
// constant-depth K1024/K512 blocks.
template <bool has_bias, bool stream_a, bool stream_output, bool local = false>
void dense_amx_gemm(
    const DenseAmxPlan& plan, at::BFloat16* out, const uint8_t* A,
    const int8_t* B, const float* As, const float* Bs, const float* bias,
    int32_t* workspace, int64_t worker, int64_t rank, int64_t window_phase = 0) {
  int32_t* Ctmp = workspace + worker * plan.worker_scratch();
  const int64_t cs = plan.c_stride();
  const Int8TileConfig config;
  at::native::cpublas::brgemm_release();
  _tile_loadconfig(&config);
  int64_t previous_m = 0, previous_n = 0, previous_rows = 0, previous_cols = 0;
  auto drain = [&] {
    for (int64_t n = 0; n < previous_cols; n += 32) {
      const auto* comp = reinterpret_cast<const int32_t*>(
          B + (previous_n + n) * plan.b_stride + 32 * plan.k);
      for (int64_t m = 0; m < previous_rows; ++m) {
        scale_C<at::BFloat16, has_bias, 32, stream_output>::apply(
            out + (previous_m + m) * plan.output_stride + previous_n + n,
            Ctmp + m * cs + n, comp,
            has_bias ? bias + previous_n + n : nullptr, As[previous_m + m],
            Bs + previous_n + n);
      }
    }
    previous_rows = previous_cols = 0;
  };
  auto rectangle = [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    const int64_t windows = div_up((nb1 - nb0) * 32, plan.macro_n);
    const int64_t first =
        ((mb0 / plan.ownership.group_m + window_phase) % 2) * (windows / 2);
    for (int64_t window = 0; window < windows; ++window) {
      const int64_t nc = nb0 * 32 + ((window + first) % windows) * plan.macro_n;
      const int64_t cols = std::min(plan.macro_n, nb1 * 32 - nc);
      for (int64_t mc = mb0 * 32; mc < std::min(plan.m, mb1 * 32); mc += plan.macro_m) {
        const int64_t rows = std::min(plan.macro_m, std::min(plan.m, mb1 * 32) - mc);
        if (rows % 32 != 0) {
          drain();
          _tile_release();
          alignas(64) int32_t tail_scratch[32 * 32];
          for (int64_t n = nc; n < nc + cols; n += 32) {
            for (int64_t m = mc; m < mc + rows; m += 32) {
              tinygemm_kernel<at::BFloat16, has_bias>(
                  A + m * plan.a_stride, B + n * plan.b_stride,
                  out + m * plan.output_stride + n, tail_scratch, As + m, Bs + n,
                  has_bias ? bias + n : nullptr, std::min<int64_t>(32, mc + rows - m),
                  32, plan.k, plan.a_stride, 32, plan.output_stride, true);
            }
          }
          at::native::cpublas::brgemm_release();
          _tile_loadconfig(&config);
          continue;
        }
        if (previous_rows != rows || previous_cols != cols) {
          drain();
        }
        for (int64_t kb = 0; kb < plan.k; kb += plan.k_block) {
          for (int64_t n = 0; n < cols; n += 32) {
            for (int64_t m = 0; m < rows; m += 32) {
              const int64_t old_m = previous_m + m, old_n = previous_n + n;
              AT_DISPATCH_BOOL(kb == 0 && previous_rows != 0, has_pending, [&] {
                auto tile = [&](auto fixed_depth) __attribute__((always_inline)) {
                  constexpr int depth = decltype(fixed_depth)::value;
                  auto invoke = [&](auto apply) __attribute__((always_inline)) {
                    apply(
                        plan, A + (mc + m) * plan.a_stride, B + (nc + n) * plan.b_stride,
                        Ctmp + m * cs + n, kb, std::min(plan.k, kb + plan.k_block),
                        has_pending ? out + old_m * plan.output_stride + old_n : nullptr,
                        has_pending ? As + old_m : nullptr, has_pending ? Bs + old_n : nullptr,
                        has_pending && has_bias ? bias + old_n : nullptr,
                        has_pending ? reinterpret_cast<const int32_t*>(
                            B + old_n * plan.b_stride + 32 * plan.k) : nullptr);
                  };
                  if constexpr (depth != 0) {
                    invoke(dense_amx_tile_impl<has_bias, has_pending, stream_a,
                                               stream_output, local, depth>);
                  } else {
                    invoke(dense_amx_tile<has_bias, has_pending, stream_a, stream_output,
                                          local>);
                  }
                };
                const int64_t depth = std::min(plan.k - kb, plan.k_block);
                if constexpr (local) {
                  if (depth == 1024) {
                    tile(std::integral_constant<int, 1024>{});
                  } else if (depth == 512) {
                    tile(std::integral_constant<int, 512>{});
                  } else {
                    tile(std::integral_constant<int, 0>{});
                  }
                } else {
                  tile(std::integral_constant<int, 0>{});
                }
              });
            }
          }
        }
        previous_m = mc;
        previous_n = nc;
        previous_rows = rows;
        previous_cols = cols;
      }
    }
  };
  if constexpr (local) {
    // Finish every column of each full M strip before the short final strip.
    for (int64_t m = 0; m < plan.m; m += plan.macro_m) {
      rectangle(m / 32, div_up(std::min(plan.m, m + plan.macro_m), int64_t(32)), 0,
                plan.n / 32);
    }
  } else {
    plan.ownership.each(rank, rectangle);
  }
  drain();
  _tile_release();
  if constexpr (stream_output) {
    _mm_sfence();
  }
}

#endif

}  // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Ctmp, As, Bs, nullptr, M, N, K, lda, ldb, ldc, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      const uint8_t* __restrict__ A,        \
      const int8_t* __restrict__ B,         \
      TYPE* __restrict__ C,                 \
      int32_t* __restrict__ Ctmp,           \
      const float* __restrict__ As,         \
      const float* __restrict__ Bs,         \
      int64_t M,                            \
      int64_t N,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldb,                          \
      int64_t ldc,                          \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(A);
  CHECK_DIM(2, A);

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t lda = A.stride(0);

  const auto st = A.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "per_token_quant_int8: expect A to be bfloat16 or half.");

  auto Aq = at::empty({M, K}, A.options().dtype(at::kByte));
  auto As = at::empty({M}, A.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "per_token_quant_int8", [&] {
    uint8_t* __restrict__ Aq_data = Aq.data_ptr<uint8_t>();
    float* __restrict__ As_data = As.data_ptr<float>();
    const scalar_t* __restrict__ A_data = A.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
      }
    });
  });
  return std::make_tuple(Aq, As);
}

// weight     :  static, per-channel, symmetric
// activation : dynamic,   per-token, symmetric
//
// mat1    : [M, K]
// mat2    : [N, K]
// scales1 : [M]
// scales2 : [N]
// bias    : [N]
// out     : [M, N]
//
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales1);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat1.size(1);

  // see [NOTE]: s8s8 igemm compensation in avx512-vnni
  CHECK_EQ(mat2.size(1), (int64_t)(is_vnni ? K + sizeof(int32_t) : K));
  CHECK_EQ(scales1.numel(), M);
  CHECK_EQ(scales2.numel(), N);

  TORCH_CHECK(mat1.scalar_type() == at::kByte, "int8_scaled_mm: expect mat1 to be uint8.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm: expect mat2 to be int8.");
  TORCH_CHECK(
      scales1.scalar_type() == at::kFloat && scales2.scalar_type() == at::kFloat,
      "int8_scaled_mm: expect scales to be float32.");

  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_kernel_impl", [&] {
    parallel_sgl_dense([&](const auto& execute) {
      int8_scaled_mm_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<uint8_t>(),
          packed_w.data_ptr<int8_t>(),
          scales1.data_ptr<float>(),
          scales2.data_ptr<float>(),
          bias_data,
          M,
          N,
          K,
          execute);
    });
  });
  return out;
}

namespace {
void log_sgl_dispatch(const char* kernel, int64_t team, const DenseAmxPlan* plan = nullptr,
                      int64_t activation_bytes = 0, int64_t scratch_bytes = 0) {
  std::fprintf(stderr, "SGL_DISPATCH {\"kernel\":\"%s\",\"actual_team_size\":%ld", kernel, team);
  if (plan != nullptr) {
    std::fprintf(stderr, ",\"macro_m\":%ld,\"macro_n\":%ld,\"k_block\":%ld,\"replicas\":%ld,"
                 "\"stream_a\":%d,\"stream_output\":%d", plan->macro_m, plan->macro_n,
                 plan->k_block, plan->replicas, plan->stream_a, plan->stream_output);
  }
  if (activation_bytes != 0) {
    std::fprintf(stderr, ",\"activation_bytes\":%ld,\"scratch_bytes\":%ld",
                 activation_bytes, scratch_bytes);
  }
  std::fprintf(stderr, "}\n");
}
}  // namespace

// fused `per_token_quant_int8_cpu` and `int8_scaled_mm_cpu`
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat1.size(1);
  int64_t lda = mat1.stride(0);

  // see [NOTE]: s8s8 igemm compensation in avx512-vnni
  CHECK_EQ(mat2.size(1), (int64_t)(is_vnni ? K + sizeof(int32_t) : K));
  CHECK_EQ(scales2.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "int8_scaled_mm_with_quant: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype, "int8_scaled_mm_with_quant: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm_with_quant: expect mat2 to be int8.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "int8_scaled_mm_with_quant: expect scales to be float32.");

  const bool observe = std::getenv("VLLM_CPU_SGL_DISPATCH") &&
      std::strcmp(std::getenv("VLLM_CPU_SGL_DISPATCH"), "1") == 0;
  bool amx = false;
#if defined(__AMX_INT8__) && defined(__linux__)
  amx = sgl_dense_schedule() != SGLDenseSchedule::Legacy &&
      st == at::kBFloat16 && amx_int8_enabled() &&
      2 * cpu_utils::get_available_l2_size() == DenseAmxPlan::cache_bytes;
#endif
  // AMX activations use a K + 64 row stride; the buffer also holds K-stride rows.
  const int64_t buffer_size = M * ((amx ? K + 64 : K) + sizeof(float));
  auto buffer = at::empty({buffer_size}, mat1.options().dtype(at::kByte));
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_with_quant_kernel_impl", [&] {
    const scalar_t* __restrict__ A_data = mat1.data_ptr<scalar_t>();

    auto run = [&](const auto& execute) {
      uint8_t* __restrict__ Aq_data = buffer.data_ptr<uint8_t>();
      float* __restrict__ As_data = reinterpret_cast<float*>(Aq_data + M * K);
      execute.quantize(M, [&](int64_t begin, int64_t end) __attribute__((always_inline)) {
        for (int64_t m = begin; m < end; ++m) {
          quantize_row_int8<scalar_t>(Aq_data + m * K, As_data[m], A_data + m * lda, K);
        }
      });

      int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        Aq_data,
        packed_w.data_ptr<int8_t>(),
        As_data,
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        execute);
    };
    if (sgl_dense_schedule() == SGLDenseSchedule::Legacy) {
      run(DenseLegacyExecution{});
      return;
    }
#if defined(__AMX_INT8__) && defined(__linux__)
    at::Tensor arena, accumulator;
    DenseAmxPlan invocation_plan;
    DenseAmxLocalPlan local_plan;
    bool use_local = false;
#endif
    parallel_sgl_team([&](const DenseTeamExecution& execute) {
#if defined(__AMX_INT8__) && defined(__linux__)
      const bool cohort_eligible = amx && DenseAmxPlan::eligible(M, N, K, execute.team);
      const bool local_eligible =
          amx && DenseAmxLocalPlan::eligible(M, N, K, execute.team);
      if (cohort_eligible || local_eligible) {
        if (execute.worker == 0) {
          if (cohort_eligible) {
            invocation_plan = DenseAmxPlan::cached(M, N, K, execute.team);
          }
          if (local_eligible) {
            local_plan = DenseAmxLocalPlan::make(M, N, K, execute.team);
            use_local = local_plan.preferred(cohort_eligible ? &invocation_plan : nullptr);
          }
          if (use_local) {
            const int64_t scratch = execute.team * local_plan.kernel.worker_scratch();
            if (observe) {
              log_sgl_dispatch("amx_local", execute.team, &local_plan.kernel,
                               buffer_size + local_plan.activation_bytes, 4 * scratch);
            }
            arena = at::empty({local_plan.activation_bytes}, mat1.options().dtype(at::kByte));
            accumulator = at::empty({scratch}, mat1.options().dtype(at::kInt));
          } else if (cohort_eligible) {
            const auto& p = invocation_plan;
            const int64_t replica_bytes = p.activation_bytes();
            const int64_t scratch = p.replicas == 1 ? 0 : execute.team * p.worker_scratch();
            if (observe) {
              log_sgl_dispatch(p.replicas == 1 ? "legacy_team" : "amx_cohort", execute.team,
                               &p, buffer_size + (p.replicas - 1) * replica_bytes, 4 * scratch);
            }
            if (p.replicas > 1) {
              arena = at::empty({(p.replicas - 1) * replica_bytes},
                                mat1.options().dtype(at::kByte));
              accumulator = at::empty({scratch}, mat1.options().dtype(at::kInt));
            }
          }
        }
        execute.barrier();
        if (use_local) {
          float* __restrict__ maxima =
              reinterpret_cast<float*>(buffer.data_ptr<uint8_t>() + M * K);
          execute.quantize(M, [&](int64_t begin, int64_t end) __attribute__((always_inline)) {
            for (int64_t m = begin; m < end; ++m) {
              maxima[m] = bf16_row_amax(
                  reinterpret_cast<const at::BFloat16*>(A_data + m * lda), K);
            }
          });
          const auto& owner = local_plan.workers[execute.worker];
          const auto plan = local_plan.local(execute.worker);
          uint8_t* local_A = arena.data_ptr<uint8_t>() + owner.activation_offset;
          float* local_As = reinterpret_cast<float*>(local_A + plan.m * plan.a_stride);
          for (int64_t row = owner.producer_begin; row < owner.producer_end; ++row) {
            const int64_t m = owner.m_begin + row;
            bf16_quant_from_amax(
                local_A + row * plan.a_stride, local_As[row],
                reinterpret_cast<const at::BFloat16*>(A_data + m * lda), maxima[m], K);
          }
          execute.barrier();
          AT_DISPATCH_BOOL(has_bias, with_bias, [&] {
            dense_amx_gemm<with_bias, true, true, true>(
                plan, out.data_ptr<at::BFloat16>() + owner.m_begin * N + owner.n_begin,
                local_A, packed_w.data_ptr<int8_t>() + owner.n_begin * plan.b_stride,
                local_As, scales2.data_ptr<float>() + owner.n_begin,
                has_bias ? bias_data + owner.n_begin : nullptr,
                accumulator.data_ptr<int32_t>(), execute.worker, 0,
                (owner.m_begin / 256) % 2);
          });
          return;
        }
      }
      if (cohort_eligible) {
        const auto& global_plan = invocation_plan;
        if (global_plan.replicas == 1) {
          run(DenseTeamGridExecution{execute});
          return;
        }
        int64_t replica, rank, column;
        const auto plan = global_plan.cohort(
            execute.team, execute.worker, replica, rank, column);
        uint8_t* local_A = replica == 0 ? buffer.data_ptr<uint8_t>()
            : arena.data_ptr<uint8_t>() + (replica - 1) * global_plan.activation_bytes();
        float* local_As = reinterpret_cast<float*>(local_A + M * plan.a_stride);
        const int64_t members = plan.ownership.workers;
        const int64_t begin = sgl_partition_begin(M, members, rank);
        const int64_t end = sgl_partition_begin(M, members, rank + 1);
        for (int64_t m = begin; m < end; ++m) {
          quantize_row_int8<scalar_t>(
              local_A + m * plan.a_stride, local_As[m], A_data + m * lda, K);
        }
        execute.barrier();
        AT_DISPATCH_BOOL(has_bias, with_bias, [&] {
          AT_DISPATCH_BOOL(plan.stream_a, stream_a, [&] {
            AT_DISPATCH_BOOL(plan.stream_output, stream_output, [&] {
              dense_amx_gemm<with_bias, stream_a, stream_output>(
                  plan, out.data_ptr<at::BFloat16>() + column, local_A,
                  packed_w.data_ptr<int8_t>() + column * plan.b_stride,
                  local_As, scales2.data_ptr<float>() + column,
                  has_bias ? bias_data + column : nullptr,
                  accumulator.data_ptr<int32_t>(), execute.worker, rank);
            });
          });
        });
        return;
      }
#endif
      if (execute.worker == 0 && observe) {
        log_sgl_dispatch("generic", execute.team);
      }
      run(execute);
    });
  });
  return out;
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <array>
#include <cstdlib>
#include <cstring>

namespace {

enum class SGLDenseSchedule { Legacy, Generic };

inline SGLDenseSchedule sgl_dense_schedule() {
  static const auto schedule = [] {
    const char* value = std::getenv("VLLM_CPU_SGL_SCHEDULER");
    if (value == nullptr || *value == '\0' ||
        std::strcmp(value, "legacy") == 0) {
      return SGLDenseSchedule::Legacy;
    }
    if (std::strcmp(value, "generic") == 0) {
      return SGLDenseSchedule::Generic;
    }
    TORCH_CHECK(false, "Invalid VLLM_CPU_SGL_SCHEDULER: ", value,
                ". Expected legacy or generic; restart "
                "after changing it.");
  }();
  return schedule;
}

inline int64_t sgl_partition_begin(int64_t size, int64_t parts, int64_t rank) {
  return rank * (size / parts) + std::min(rank, size % parts);
}

// parallel_2d's divisor grid: rectangle of `worker` among `team` workers.
inline std::array<int64_t, 4> sgl_grid_rectangle(int64_t mb, int64_t nb,
                                                 int64_t team, int64_t worker) {
  int64_t rows = std::ceil(std::sqrt(float(mb) / nb * team));
  while (team % rows != 0) {
    --rows;
  }
  const int64_t columns = team / rows;
  const int64_t height = div_up(mb, rows), width = div_up(nb, columns);
  const int64_t m0 = (worker / columns) * height;
  const int64_t n0 = (worker % columns) * width;
  return {m0, std::min(mb, m0 + height), n0, std::min(nb, n0 + width)};
}

struct DenseTeamExecution {
  int64_t team = 1, worker = 0;
  bool owns_team = false;

  template <typename func_t>
  void quantize(int64_t rows, const func_t& f) const {
    const int64_t begin = sgl_partition_begin(rows, team, worker);
    const int64_t end = sgl_partition_begin(rows, team, worker + 1);
    if (begin < end) {
      at::internal::ThreadIdGuard thread_guard(worker);
      c10::ParallelGuard parallel_guard(true);
      f(begin, end);
    }
    barrier();
  }

  template <typename func_t>
  void operator()(int64_t mb, int64_t nb, const func_t& f) const {
    if (mb <= 0 || nb <= 0) {
      return;
    }
    const int64_t productive = std::min(team, mb * nb);
    if (worker >= productive) {
      return;
    }
    const int64_t bands =
        std::clamp<int64_t>(std::ceil(std::sqrt(float(mb) / nb * productive)),
                            div_up(productive, nb), std::min(mb, productive));
    const int64_t large_workers = productive / bands + 1;
    const int64_t large_bands = productive % bands;
    const int64_t large_total = large_workers * large_bands;
    const int64_t band =
        worker < large_total
            ? worker / large_workers
            : large_bands + (worker - large_total) / (productive / bands);
    const int64_t first = sgl_partition_begin(productive, bands, band);
    const int64_t count =
        sgl_partition_begin(productive, bands, band + 1) - first;
    const int64_t m0 = sgl_partition_begin(mb, bands, band);
    const int64_t m1 = sgl_partition_begin(mb, bands, band + 1);
    const int64_t n0 = sgl_partition_begin(nb, count, worker - first);
    const int64_t n1 = sgl_partition_begin(nb, count, worker - first + 1);
    at::internal::ThreadIdGuard thread_guard(worker);
    c10::ParallelGuard parallel_guard(true);
    f(m0, m1, n0, n1);
  }

  void barrier() const {
#if defined(_OPENMP)
    if (owns_team) {
  #pragma omp barrier
    }
#endif
  }
};

template <typename func_t>
inline void parallel_sgl_team(const func_t& f) {
  at::internal::lazy_init_num_threads();
#if defined(_OPENMP)
  if (!at::in_parallel_region() && !omp_in_parallel() &&
      at::get_num_threads() > 1) {
  #pragma omp parallel num_threads(at::get_num_threads())
    {
      f(DenseTeamExecution{omp_get_num_threads(), omp_get_thread_num(), true});
    }
    return;
  }
#endif
  f(DenseTeamExecution{});
}

struct DenseLegacyExecution {
  template <typename func_t>
  void quantize(int64_t rows, const func_t& f) const {
    at::parallel_for(0, rows, 0, f);
  }

  template <typename func_t>
  void operator()(int64_t mb, int64_t nb, const func_t& f) const {
    if (mb <= 0 || nb <= 0) {
      return;
    }
    bool nested = at::in_parallel_region();
#if defined(_OPENMP)
    nested = nested || omp_in_parallel();
#endif
    if (nested || at::get_num_threads() == 1) {
      at::internal::ThreadIdGuard thread_guard(0);
      f(0, mb, 0, nb);
      return;
    }
    auto visit = [&](int64_t team, int64_t worker) {
      const auto [m0, m1, n0, n1] = sgl_grid_rectangle(mb, nb, team, worker);
      at::internal::ThreadIdGuard thread_guard(worker);
      f(m0, m1, n0, n1);
    };
#if defined(_OPENMP)
  #pragma omp parallel num_threads(adjust_num_threads(mb))
    {
      visit(omp_get_num_threads(), omp_get_thread_num());
    }
#else
    visit(1, 0);
#endif
  }

  void barrier() const {}
};

// Team execution with DenseLegacyExecution's quantization chunks and grid.
struct DenseTeamGridExecution : DenseTeamExecution {
  template <typename func_t>
  void quantize(int64_t rows, const func_t& f) const {
    const int64_t chunk = div_up(rows, team);
    const int64_t begin = worker * chunk;
    const int64_t end = std::min(rows, begin + chunk);
    if (begin < end) {
      at::internal::ThreadIdGuard thread_guard(worker);
      c10::ParallelGuard parallel_guard(true);
      f(begin, end);
    }
    barrier();
  }

  template <typename func_t>
  void operator()(int64_t mb, int64_t nb, const func_t& f) const {
    const int64_t participants =
        mb == 1 ? team : std::max(int64_t(1), (team / 2) * 2);
    if (mb <= 0 || nb <= 0 || worker >= participants) {
      return;
    }
    const auto [m0, m1, n0, n1] =
        sgl_grid_rectangle(mb, nb, participants, worker);
    if (m0 < m1 && n0 < n1) {
      at::internal::ThreadIdGuard thread_guard(worker);
      c10::ParallelGuard parallel_guard(true);
      f(m0, m1, n0, n1);
    }
  }
};

template <typename func_t>
inline void parallel_sgl_dense(const func_t& f) {
  if (sgl_dense_schedule() == SGLDenseSchedule::Legacy) {
    f(DenseLegacyExecution{});
  } else {
    parallel_sgl_team(f);
  }
}

}  // namespace

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

enum class SGLDenseSchedule {
  Legacy,
  Logical96,
  Flat32,
  Flat128x64,
  Flat128x128
};

inline SGLDenseSchedule sgl_dense_schedule() {
  static const auto schedule = [] {
    const char* value = std::getenv("VLLM_CPU_SGL_SCHEDULER");
    if (value == nullptr || *value == '\0' ||
        std::strcmp(value, "legacy") == 0) {
      return SGLDenseSchedule::Legacy;
    }
    if (std::strcmp(value, "logical96_dense") == 0) {
      return SGLDenseSchedule::Logical96;
    }
    if (std::strcmp(value, "flat_32x32") == 0) {
      return SGLDenseSchedule::Flat32;
    }
    if (std::strcmp(value, "flat_128x64") == 0) {
      return SGLDenseSchedule::Flat128x64;
    }
    if (std::strcmp(value, "flat_128x128") == 0) {
      return SGLDenseSchedule::Flat128x128;
    }
    TORCH_CHECK(false, "Invalid VLLM_CPU_SGL_SCHEDULER: ", value,
                ". Expected legacy, logical96_dense, flat_32x32, flat_128x64, "
                "or flat_128x128; restart after changing it.");
  }();
  return schedule;
}

struct DenseFlatGrid {
  int64_t bm, bn, rows, columns;

  DenseFlatGrid(int64_t mb, int64_t nb, SGLDenseSchedule schedule) {
    bm = schedule == SGLDenseSchedule::Flat32 ? 1 : 4;
    bn = schedule == SGLDenseSchedule::Flat32       ? 1
         : schedule == SGLDenseSchedule::Flat128x64 ? 2
                                                    : 4;
    rows = div_up(mb, bm);
    columns = div_up(nb, bn);
  }

  int64_t tasks() const { return rows * columns; }

  bool eligible(int64_t m) const {
    return m >= 128 && (bm == 1 || tasks() >= 192);
  }
};

template <typename func_t>
inline void parallel_sgl_flat(const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                              const func_t& f) {
#if defined(_OPENMP)
  #pragma omp parallel num_threads(at::get_num_threads())
  {
    const int64_t nth = omp_get_num_threads();
    const int64_t worker = omp_get_thread_num();
#else
  {
    const int64_t nth = 1;
    const int64_t worker = 0;
#endif
    const int64_t quotient = grid.tasks() / nth;
    const int64_t remainder = grid.tasks() % nth;
    const int64_t begin = worker * quotient + std::min(worker, remainder);
    const int64_t end = begin + quotient + (worker < remainder);
    f([&](const auto& rectangle) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t m0 = (task % grid.rows) * grid.bm;
        const int64_t n0 = (task / grid.rows) * grid.bn;
        rectangle(m0, std::min(mb, m0 + grid.bm), n0,
                  std::min(nb, n0 + grid.bn));
      }
    });
  }
}

struct DenseLogical96Grid {
  int64_t pm, pn, bm, bn;

  DenseLogical96Grid(int64_t mb, int64_t nb) {
    float ratio = float(mb) / nb;
    pm = std::ceil(std::sqrt(ratio * 96));
    for (; pm > 0; --pm) {
      pn = 96 / pm;
      if (pm * pn == 96) {
        break;
      }
    }
    bm = div_up(mb, pm);
    bn = div_up(nb, pn);
  }
};

template <typename func_t>
inline void parallel_sgl_dense(int64_t mb, int64_t nb, int64_t m,
                               const func_t& f) {
  const auto schedule = sgl_dense_schedule();
  if (mb == 0 || nb == 0) {
    return;
  }
  if (schedule == SGLDenseSchedule::Flat32 ||
      schedule == SGLDenseSchedule::Flat128x64 ||
      schedule == SGLDenseSchedule::Flat128x128) {
    const DenseFlatGrid grid(mb, nb, schedule);
    if (grid.eligible(m)) {
      parallel_sgl_flat(grid, mb, nb, f);
      return;
    }
  }
  if (schedule != SGLDenseSchedule::Logical96) {
    parallel_2d(mb, nb, [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
      f([&](const auto& rectangle) { rectangle(m0, m1, n0, n1); });
    });
    return;
  }
#if defined(_OPENMP)
  const DenseLogical96Grid grid(mb, nb);
  const int64_t rows = div_up(mb, grid.bm);
  std::atomic<int64_t> next{0};
  #pragma omp parallel num_threads(at::get_num_threads())
  {
    const int physical = omp_get_thread_num();
    const int owned = std::min(omp_get_num_threads(), 96);
    const int64_t row = owned / grid.pn;
    const int64_t column = std::min(nb, (owned % grid.pn) * grid.bn);
    const int64_t count =
        row >= rows ? 0 : (nb - column) + (rows - row - 1) * nb;
    f([&](const auto& rectangle) {
      if (physical < 96) {
        const int64_t m0 = (physical / grid.pn) * grid.bm;
        const int64_t n0 = (physical % grid.pn) * grid.bn;
        if (m0 < mb && n0 < nb) {
          rectangle(m0, std::min(mb, m0 + grid.bm), n0,
                    std::min(nb, n0 + grid.bn));
        }
      }
      if (count == 0) {
        return;
      }
      for (int64_t task = next.fetch_add(1, std::memory_order_relaxed);
           task < count; task = next.fetch_add(1, std::memory_order_relaxed)) {
        const int64_t offset = task + column;
        const int64_t m0 = (row + offset / nb) * grid.bm;
        const int64_t n0 = offset % nb;
        rectangle(m0, std::min(mb, m0 + grid.bm), n0, n0 + 1);
      }
    });
  }
#else
  f([&](const auto& rectangle) { rectangle(0, mb, 0, nb); });
#endif
}

}  // namespace

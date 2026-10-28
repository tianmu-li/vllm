// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

enum class SGLDenseSchedule {
  Legacy,
  Logical96,
  Flat32,
  Flat128x64,
  Flat128x128,
  FlatAuto
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
    if (std::strcmp(value, "flat_auto") == 0) {
      return SGLDenseSchedule::FlatAuto;
    }
    TORCH_CHECK(false, "Invalid VLLM_CPU_SGL_SCHEDULER: ", value,
                ". Expected legacy, logical96_dense, flat_32x32, flat_128x64, "
                "flat_128x128, or flat_auto; restart after changing it.");
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

inline SGLDenseSchedule sgl_auto_schedule(int64_t mb, int64_t nb, int64_t m,
                                          int64_t n, int64_t nth) {
  const DenseFlatGrid large(mb, nb, SGLDenseSchedule::Flat128x64);
  if (m % 128 == 0 && n % 64 == 0 && large.eligible(m) &&
      (large.tasks() % nth == 0 || large.tasks() / nth >= 8)) {
    return SGLDenseSchedule::Flat128x64;
  }
  return SGLDenseSchedule::Flat32;
}

struct DenseFlatTraversal {
  int64_t group_m, group_n;
  bool grouped, partial_m;

  static bool should_group(int64_t m, int64_t nb, int64_t panels) {
    return double(m) >= 0.25 * double(nb * 32) * panels;
  }

  DenseFlatTraversal(const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                     int64_t m, int64_t k, int64_t panels, int64_t nth)
      : group_m(std::min(grid.bm, mb)), group_n(std::min(grid.bn, nb)) {
    grouped = should_group(m, nb, panels);
    partial_m = grouped && m % 32 != 0 && mb > 1;
    if (!grouped) {
      return;
    }
    const int64_t budget =
        std::max<int64_t>(grid.bn, get_cache_blocks<int8_t>(32 * k * panels));
    const int64_t row_budget =
        std::max<int64_t>(grid.bm, get_cache_blocks<int8_t>(32 * k));
    const int64_t owned = std::max<int64_t>(1, grid.tasks() / nth);
    int64_t area = 1;
    while (area <= owned / 2) {
      area *= 2;
    }
    for (; area > 0; area /= 2) {
      const double aspect = !partial_m && grid.bm * grid.bn * area <= 16
                                ? std::clamp(double(k) / 1024, 2.0, 4.0)
                                : 2.0;
      double best = std::numeric_limits<double>::infinity();
      bool found = false;
      for (int64_t gm = 1; gm <= area; gm *= 2) {
        const int64_t gn = area / gm;
        const int64_t candidate_m = grid.bm * gm;
        const int64_t candidate_n = grid.bn * gn;
        if (candidate_m > mb || candidate_m > row_budget || candidate_n > nb ||
            candidate_n > budget) {
          continue;
        }
        const double error =
            std::abs(std::log2(double(candidate_m) / candidate_n / aspect));
        if (!found || error < best - 1e-12 ||
            (std::abs(error - best) <= 1e-12 && candidate_n > group_n)) {
          group_m = candidate_m;
          group_n = candidate_n;
          best = error;
          found = true;
        }
      }
      if (found) {
        return;
      }
    }
  }

  int64_t column_rows(const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                      int64_t k, int64_t panels, int64_t nth) const {
    if (!grouped) {
      return 0;
    }
    if (partial_m || group_m * group_n > 16 || mb % group_m != 0) {
      return 0;
    }
    int64_t rows = group_m;
    if (k <= 2048) {
      return rows;
    }
    const int64_t max_work = rows * div_up((mb / rows) * nb, nth);
    const int64_t budget = 2 * get_cache_blocks<int8_t>(32 * k);
    for (int64_t candidate = rows + grid.bm;
         candidate <= std::min(mb, max_work); candidate += grid.bm) {
      if (mb % candidate != 0) {
        continue;
      }
      const int64_t columns = div_up((mb / candidate) * nb, nth);
      if (candidate * columns <= max_work &&
          candidate + columns * panels <= budget) {
        rows = candidate;
      }
    }
    return rows;
  }

  int64_t column_width(int64_t rows, int64_t mb, int64_t nb, int64_t k,
                       int64_t nth) const {
    if (!rows || !grouped || k <= 2048) {
      return 1;
    }
    const int64_t width = div_up((mb / rows) * nb, nth);
    return (mb / rows) * div_up(nb, width) <= nth ? width : 1;
  }

  template <typename func_t>
  void each(int64_t mb, int64_t nb, int64_t begin, int64_t end,
            const func_t& f) const {
    int64_t pending_m0 = 0, pending_m1 = 0, pending_n0 = 0, pending_n1 = 0;
    bool pending = false;
    auto emit = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
      if (pending && pending_m0 == m0 && pending_m1 == m1 && pending_n1 == n0) {
        pending_n1 = n1;
      } else if (pending && pending_n0 == n0 && pending_n1 == n1 &&
                 pending_m1 == m0) {
        pending_m1 = m1;
      } else {
        if (pending) {
          f(pending_m0, pending_m1, pending_n0, pending_n1);
        }
        pending_m0 = m0;
        pending_m1 = m1;
        pending_n0 = n0;
        pending_n1 = n1;
        pending = true;
      }
    };
    auto visit = [&](int64_t body_mb, int64_t first, int64_t last) {
      while (first < last) {
        const int64_t m0 = (first / (nb * group_m)) * group_m;
        const int64_t height = std::min(group_m, body_mb - m0);
        const int64_t offset = first - m0 * nb;
        const int64_t n0 = (offset / (height * group_n)) * group_n;
        const int64_t width = std::min(group_n, nb - n0);
        const int64_t local = offset - n0 * height;
        int64_t count = std::min(last - first, height * width - local);
        first += count;
        int64_t row = local / width;
        const int64_t column = local % width;
        if (column != 0) {
          const int64_t fragment = std::min(count, width - column);
          emit(m0 + row, m0 + row + 1, n0 + column, n0 + column + fragment);
          count -= fragment;
          ++row;
        }
        const int64_t rows = count / width;
        if (rows != 0) {
          emit(m0 + row, m0 + row + rows, n0, n0 + width);
          row += rows;
          count -= rows * width;
        }
        if (count != 0) {
          emit(m0 + row, m0 + row + 1, n0, n0 + count);
        }
      }
    };
    const int64_t edge_rows = mb % group_m != 0 ? mb % group_m
                              : partial_m       ? 1
                                                : 0;
    if (edge_rows == 0) {
      visit(mb, begin, end);
    } else {
      // Spread the short final group across workers instead of a few owners.
      const int64_t body_mb = mb - edge_rows;
      while (begin < end) {
        const int64_t n = begin / mb;
        const int64_t offset = begin % mb;
        if (offset < body_mb) {
          const int64_t count = std::min(end - begin, body_mb - offset);
          const int64_t first = begin - n * edge_rows;
          visit(body_mb, first, first + count);
          begin += count;
        } else {
          const int64_t count = std::min(end - begin, mb - offset);
          emit(offset, offset + count, n, n + 1);
          begin += count;
        }
      }
    }
    if (pending) {
      f(pending_m0, pending_m1, pending_n0, pending_n1);
    }
  }
};

struct DenseFlatProblem {
  int64_t mb, nb, m, k, panels;
  SGLDenseSchedule schedule;
};

inline bool sgl_flat_matches_legacy(const DenseFlatGrid& grid, int64_t mb,
                                    int64_t nb, int64_t m, int64_t k,
                                    int64_t panels, int64_t expected) {
  if (m % 32 != 0 || mb > std::numeric_limits<int>::max() ||
      nb > std::numeric_limits<int>::max() ||
      !DenseFlatTraversal::should_group(m, nb, panels) ||
      adjust_num_threads(static_cast<int>(mb)) != expected) {
    return false;
  }
  const DenseFlatTraversal traversal(grid, mb, nb, m, k, panels, expected);
  if (traversal.group_m * traversal.group_n <= 16 ||
      mb % traversal.group_m != 0 || nb % traversal.group_n != 0 ||
      (mb / traversal.group_m) * (nb / traversal.group_n) != expected) {
    return false;
  }
  const float ratio = float(mb) / nb;
  const int64_t legacy_m = std::ceil(std::sqrt(ratio * expected));
  return legacy_m == mb / traversal.group_m;
}

#if defined(_OPENMP)
template <typename func_t>
__attribute__((noinline)) void sgl_flat_worker(const DenseFlatProblem& problem,
                                               const func_t& f) {
  const int64_t nth = omp_get_num_threads();
  const int64_t worker = omp_get_thread_num();
  const DenseFlatGrid grid(problem.mb, problem.nb, problem.schedule);
  const DenseFlatTraversal traversal(grid, problem.mb, problem.nb, problem.m,
                                     problem.k, problem.panels, nth);
  const int64_t tasks = problem.mb * problem.nb;
  const int64_t quotient = tasks / nth;
  const int64_t remainder = tasks % nth;
  const int64_t begin = worker * quotient + std::min(worker, remainder);
  const int64_t end = begin + quotient + (worker < remainder);
  f([&](const auto& rectangle) {
    traversal.each(problem.mb, problem.nb, begin, end, rectangle);
  });
}
#endif

// Keep the uniform tile loop separate from the adaptive traversal's code.
template <typename func_t>
__attribute__((noinline)) void sgl_uniform_rectangle(int64_t m0, int64_t m1,
                                                     int64_t n0, int64_t n1,
                                                     const func_t& f) {
  f([&](const auto& rectangle) { rectangle(m0, m1, n0, n1); });
}

template <bool automatic = false, typename func_t>
inline void parallel_sgl_base(const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                              const func_t& f, int64_t m = 0, int64_t n = 0) {
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
    decltype(auto) active_grid = [&]() -> decltype(auto) {
      if constexpr (automatic) {
        return DenseFlatGrid(mb, nb, sgl_auto_schedule(mb, nb, m, n, nth));
      } else {
        return (grid);
      }
    }();
    const int64_t quotient = active_grid.tasks() / nth;
    const int64_t remainder = active_grid.tasks() % nth;
    const int64_t begin = worker * quotient + std::min(worker, remainder);
    const int64_t end = begin + quotient + (worker < remainder);
    f([&](const auto& rectangle) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t m0 = (task % active_grid.rows) * active_grid.bm;
        const int64_t n0 = (task / active_grid.rows) * active_grid.bn;
        rectangle(m0, std::min(mb, m0 + active_grid.bm), n0,
                  std::min(nb, n0 + active_grid.bn));
      }
    });
  }
}

#if defined(_OPENMP)
template <typename func_t>
inline bool parallel_sgl_uniform(const DenseFlatTraversal& traversal,
                                 int64_t nb, int64_t expected,
                                 const func_t& f) {
  bool mismatch = false;
  #pragma omp parallel num_threads(expected)
  {
    const int64_t worker = omp_get_thread_num();
    if (omp_get_num_threads() != expected) {
      if (worker == 0) {
        mismatch = true;
      }
    } else {
      const int64_t m0 =
          (worker / (nb / traversal.group_n)) * traversal.group_m;
      const int64_t n0 =
          (worker % (nb / traversal.group_n)) * traversal.group_n;
      f([&](const auto& rectangle) {
        rectangle(m0, m0 + traversal.group_m, n0, n0 + traversal.group_n);
      });
    }
  }
  return !mismatch;
}
#endif

template <typename func_t>
inline void parallel_sgl_flat(const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                              int64_t m, int64_t k, int64_t panels,
                              const func_t& f) {
  if (!DenseFlatTraversal::should_group(m, nb, panels)) {
    parallel_sgl_base(grid, mb, nb, f);
    return;
  }
#if defined(_OPENMP)
  const int64_t expected = at::get_num_threads();
  if (m % 32 == 0 && (mb * nb) % expected == 0) {
    const DenseFlatTraversal traversal(grid, mb, nb, m, k, panels, expected);
    if (traversal.grouped && traversal.group_m * traversal.group_n > 16 &&
        mb % traversal.group_m == 0 && nb % traversal.group_n == 0 &&
        (mb / traversal.group_m) * (nb / traversal.group_n) == expected &&
        parallel_sgl_uniform(traversal, nb, expected, f)) {
      return;
    }
  }
  #pragma omp parallel num_threads(expected)
  {
    const int64_t nth = omp_get_num_threads();
    const int64_t worker = omp_get_thread_num();
#else
  {
    const int64_t nth = 1;
    const int64_t worker = 0;
#endif
    const DenseFlatTraversal traversal(grid, mb, nb, m, k, panels, nth);
    const int64_t column_rows =
        traversal.column_rows(grid, mb, nb, k, panels, nth);
    if (traversal.grouped && !column_rows && m % 32 == 0 &&
        mb % traversal.group_m == 0 && nb % traversal.group_n == 0 &&
        (mb / traversal.group_m) * (nb / traversal.group_n) == nth) {
      const int64_t m0 =
          (worker / (nb / traversal.group_n)) * traversal.group_m;
      const int64_t n0 =
          (worker % (nb / traversal.group_n)) * traversal.group_n;
      sgl_uniform_rectangle(m0, m0 + traversal.group_m, n0,
                            n0 + traversal.group_n, f);
    } else {
      const int64_t column_width =
          traversal.column_width(column_rows, mb, nb, k, nth);
      const int64_t columns = div_up(nb, column_width);
      const int64_t tasks =
          column_rows ? (mb / column_rows) * columns : mb * nb;
      const int64_t quotient = tasks / nth;
      const int64_t remainder = tasks % nth;
      const int64_t begin =
          column_rows ? tasks * worker / nth
                      : worker * quotient + std::min(worker, remainder);
      const int64_t end = column_rows ? tasks * (worker + 1) / nth
                                      : begin + quotient + (worker < remainder);
      f([&](const auto& rectangle) {
        if (column_rows) {
          for (int64_t rank = begin; rank < end;) {
            const int64_t m0 = (rank / columns) * column_rows;
            const int64_t column = rank % columns;
            const int64_t count = std::min(end - rank, columns - column);
            rectangle(m0, m0 + column_rows, column * column_width,
                      std::min(nb, (column + count) * column_width));
            rank += count;
          }
        } else {
          traversal.each(mb, nb, begin, end, rectangle);
        }
      });
    }
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
inline void parallel_sgl_dense(int64_t mb, int64_t nb, int64_t m, int64_t n,
                               int64_t k, int64_t panels, const func_t& f) {
  const auto selected = sgl_dense_schedule();
  const auto schedule = selected == SGLDenseSchedule::FlatAuto
                            ? SGLDenseSchedule::Flat32
                            : selected;
  if (mb == 0 || nb == 0) {
    return;
  }
  if (selected == SGLDenseSchedule::FlatAuto && m >= 128 &&
      !DenseFlatTraversal::should_group(m, nb, panels)) {
    const DenseFlatGrid grid(mb, nb, SGLDenseSchedule::Flat32);
    parallel_sgl_base<true>(grid, mb, nb, f, m, n);
    return;
  }
  bool uniform_flat = false;
  int64_t expected = 0;
  if (schedule == SGLDenseSchedule::Flat32 ||
      schedule == SGLDenseSchedule::Flat128x64 ||
      schedule == SGLDenseSchedule::Flat128x128) {
    const DenseFlatGrid grid(mb, nb, schedule);
    if (grid.eligible(m)) {
#if defined(_OPENMP)
      expected = at::get_num_threads();
      uniform_flat =
          sgl_flat_matches_legacy(grid, mb, nb, m, k, panels, expected);
#endif
      if (!uniform_flat) {
        parallel_sgl_flat(grid, mb, nb, m, k, panels, f);
        return;
      }
    }
  }
  if (schedule != SGLDenseSchedule::Logical96) {
    const DenseFlatProblem problem{mb, nb, m, k, panels, schedule};
    parallel_2d(mb, nb,
                [&f, uniform_flat, expected, &problem](int64_t m0, int64_t m1,
                                                       int64_t n0, int64_t n1) {
#if defined(_OPENMP)
                  if (uniform_flat && omp_get_num_threads() != expected) {
                    sgl_flat_worker(problem, f);
                    return;
                  }
#endif
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

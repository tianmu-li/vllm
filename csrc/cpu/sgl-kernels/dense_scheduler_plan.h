// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <cmath>
#include <tuple>

namespace {

struct DenseScheduleFootprint {
  int64_t scratch_bytes = 32 * 32 * sizeof(int32_t);
  int64_t output_bytes = 2;
  int64_t fused_input_bytes = 0;
  bool has_bias = false;
  int64_t cache_bytes = 0;
  int64_t l1_bytes = 0;
};

struct DenseScheduleProblem {
  int64_t mb, nb, m, n, k, panels;
  DenseScheduleFootprint footprint;
};

struct DenseRectangle {
  int64_t m0, m1, n0, n1;
};

enum class DenseOwnership { MFast, Blocked };

inline const char* dense_ownership_name(DenseOwnership ownership) {
  return ownership == DenseOwnership::MFast ? "m_fast" : "blocked";
}

struct DenseAutoTraversal {
  bool b_stationary;
  int64_t strip, reload_bytes;
  bool cache_feasible;
  int64_t a_bytes, b_bytes;

  static DenseAutoTraversal make(const DenseScheduleProblem& problem,
                                 const DenseRectangle& rectangle) {
    const int64_t height = rectangle.m1 - rectangle.m0;
    const int64_t width = rectangle.n1 - rectangle.n0;
    const int64_t rows =
        std::min(problem.m, rectangle.m1 * 32) - rectangle.m0 * 32;
    const int64_t columns =
        std::min(problem.n, rectangle.n1 * 32) - rectangle.n0 * 32;
    const int64_t row_tile = std::min<int64_t>(32, rows);
    const int64_t column_tile = std::min<int64_t>(32, columns);
    const int64_t a_row = problem.k + sizeof(float);
    const int64_t b_column =
        problem.panels * (problem.k + sizeof(int32_t) + sizeof(float)) +
        (problem.footprint.has_bias ? sizeof(float) : 0);
    const int64_t c_element =
        problem.footprint.output_bytes + problem.footprint.fused_input_bytes;
    const int64_t available =
        problem.footprint.cache_bytes - problem.footprint.scratch_bytes;
    const int64_t a_tile = row_tile * a_row;
    const int64_t b_tile = column_tile * b_column;
    const int64_t c_tile = row_tile * column_tile * c_element;
    const bool fits = available >= a_tile + b_tile + c_tile;
    if (!fits) {
      const int64_t a_bytes = width * rows * a_row;
      const int64_t b_bytes = height * columns * b_column;
      return {true, 1, a_bytes + b_bytes, false, a_bytes, b_bytes};
    }
    // Interleaving columns crosses adjacent A tiles before reusing a B line.
    int64_t bn = std::min(
        width,
        std::max<int64_t>(1, (available - 2 * a_tile) / (b_tile + 2 * c_tile)));
    const int64_t l1_available =
        problem.footprint.l1_bytes - problem.footprint.scratch_bytes;
    if (l1_available >= a_tile + b_tile + 2 * c_tile) {
      bn = std::min(bn, std::max<int64_t>(1, (l1_available - 2 * a_tile) /
                                                 (b_tile + 2 * c_tile)));
    }
    const int64_t bm = std::min(
        height,
        std::max<int64_t>(1, (available - 2 * b_tile) / (a_tile + 2 * c_tile)));
    const bool retain_a = rows * a_row + std::min(width, 2 * bn) * b_tile +
                              2 * height * bn * c_tile <=
                          available;
    const bool retain_b = columns * b_column +
                              std::min(height, 2 * bm) * a_tile +
                              2 * width * bm * c_tile <=
                          available;
    const int64_t b_a = (retain_a ? 1 : div_up(width, bn)) * rows * a_row;
    const int64_t b_b = columns * b_column;
    const int64_t a_a = rows * a_row;
    const int64_t a_b =
        (retain_b ? 1 : div_up(height, bm)) * columns * b_column;
    const bool b_stationary = b_a + b_b < a_a + a_b ||
                              (b_a + b_b == a_a + a_b && height * width <= 16);
    return b_stationary
               ? DenseAutoTraversal{true, bn, b_a + b_b, true, b_a, b_b}
               : DenseAutoTraversal{false, bm, a_a + a_b, true, a_a, a_b};
  }
};

struct DenseAutoPlan {
  DenseOwnership ownership = DenseOwnership::MFast;
  int64_t workers = 0, group_m = 1, group_n = 1;
  int64_t max_tiles = 0;
  int64_t mb = 0, nb = 0;
  bool partial_m = false;

  template <typename func_t>
  void each(int64_t worker, const func_t& f) const {
    if (worker >= workers) {
      return;
    }
    const int64_t tasks = mb * nb;
    const int64_t quotient = tasks / workers;
    const int64_t remainder = tasks % workers;
    int64_t begin = worker * quotient + std::min(worker, remainder);
    const int64_t end = begin + quotient + (worker < remainder);
    if (ownership == DenseOwnership::MFast) {
      if (begin % mb != 0) {
        const int64_t count = std::min(end - begin, mb - begin % mb);
        f(begin % mb, begin % mb + count, begin / mb, begin / mb + 1);
        begin += count;
      }
      const int64_t whole = (end - begin) / mb;
      if (whole != 0) {
        f(0, mb, begin / mb, begin / mb + whole);
        begin += whole * mb;
      }
      if (begin != end) {
        f(0, end - begin, begin / mb, begin / mb + 1);
      }
      return;
    }
    int64_t pm0 = 0, pm1 = 0, pn0 = 0, pn1 = 0;
    bool pending = false;
    auto emit = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
      if (pending && pm0 == m0 && pm1 == m1 && pn1 == n0) {
        pn1 = n1;
      } else if (pending && pn0 == n0 && pn1 == n1 && pm1 == m0) {
        pm1 = m1;
      } else {
        if (pending) {
          f(pm0, pm1, pn0, pn1);
        }
        pm0 = m0;
        pm1 = m1;
        pn0 = n0;
        pn1 = n1;
        pending = true;
      }
    };
    auto visit = [&](int64_t body_mb, int64_t first, int64_t last) {
      while (first < last) {
        const int64_t m0 = first / (nb * group_m) * group_m;
        const int64_t height = std::min(group_m, body_mb - m0);
        const int64_t offset = first - m0 * nb;
        const int64_t n0 = offset / (height * group_n) * group_n;
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
      // Distribute the short final row group across the balanced rank ranges.
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
      f(pm0, pm1, pn0, pn1);
    }
  }

  static DenseAutoPlan make(const DenseScheduleProblem& problem, int64_t team) {
    DenseAutoPlan plan;
    plan.workers = std::min(team, problem.mb * problem.nb);
    plan.mb = problem.mb;
    plan.nb = problem.nb;
    if (!plan.workers) {
      return plan;
    }
    plan.max_tiles = div_up(problem.mb * problem.nb, plan.workers);
    const int64_t a_tile = 32 * (problem.k + sizeof(float));
    const int64_t b_tile =
        32 * (problem.panels * (problem.k + sizeof(int32_t) + sizeof(float)) +
              (problem.footprint.has_bias ? sizeof(float) : 0));
    const int64_t c_tile =
        32 * 32 *
        (problem.footprint.output_bytes + problem.footprint.fused_input_bytes);
    const int64_t capacity = 2 * problem.footprint.cache_bytes;
    if (problem.mb == 1 || problem.nb == 1 ||
        (problem.nb >= plan.workers &&
         problem.m * (problem.k + int64_t(sizeof(float))) + b_tile +
                 2 * c_tile + problem.footprint.scratch_bytes <=
             capacity)) {
      return plan;
    }
    plan.ownership = DenseOwnership::Blocked;
    plan.partial_m = problem.m % 32 != 0 && problem.mb > 1;
    const bool l1_fits =
        a_tile + b_tile + 2 * c_tile + problem.footprint.scratch_bytes <=
        problem.footprint.l1_bytes;
    const double aspect = l1_fits
                              ? 4.0
                              : std::clamp(double(problem.k) * 32 * 32 /
                                               problem.footprint.cache_bytes,
                                           2.0, 4.0);
    int64_t area = 1;
    while (area <= plan.max_tiles / 2) {
      area *= 2;
    }
    for (; area > 0; area /= 2) {
      bool found = false;
      std::tuple<double, int64_t, int64_t> best;
      for (int64_t gm = 1; gm <= area; gm *= 2) {
        const int64_t gn = area / gm;
        if (gm > problem.mb || gn > problem.nb ||
            gm * a_tile + gn * b_tile + 2 * gn * c_tile +
                    problem.footprint.scratch_bytes >
                capacity) {
          continue;
        }
        const auto score =
            area <= 16
                ? std::make_tuple(std::abs(std::log2(double(gm) / gn / aspect)),
                                  -gn, gm)
                : std::make_tuple(
                      double(gm * problem.k + gn * problem.k * problem.panels),
                      std::max(gm, gn), -gm);
        if (!found || score < best) {
          best = score;
          plan.group_m = gm;
          plan.group_n = gn;
          found = true;
        }
      }
      if (found) {
        break;
      }
    }
    return plan;
  }
};

struct DenseAutoWorker {
  const DenseAutoPlan& plan;
  const DenseScheduleProblem& problem;
  int64_t worker;

  template <typename func_t>
  void operator()(const func_t& f) const {
    plan.each(worker, f);
  }

  template <typename func_t>
  void loop(int64_t m0, int64_t m1, int64_t n0, int64_t n1,
            const func_t& f) const {
    const auto traversal = DenseAutoTraversal::make(problem, {m0, m1, n0, n1});
    if (traversal.b_stationary) {
      for (int64_t begin = n0; begin < n1; begin += traversal.strip) {
        for (int64_t m = m0; m < m1; ++m) {
          for (int64_t n = begin; n < std::min(n1, begin + traversal.strip);
               ++n) {
            f(m, n, n - begin);
          }
        }
      }
    } else {
      for (int64_t begin = m0; begin < m1; begin += traversal.strip) {
        for (int64_t n = n0; n < n1; ++n) {
          for (int64_t m = begin; m < std::min(m1, begin + traversal.strip);
               ++m) {
            f(m, n, 0);
          }
        }
      }
    }
  }
};

struct DenseAutoUniformWorker : DenseAutoWorker {
  DenseRectangle rectangle;

  template <typename func_t>
  void operator()(const func_t& f) const {
    f(rectangle.m0, rectangle.m1, rectangle.n0, rectangle.n1);
  }
};

template <typename traversal_t, typename func_t>
inline void sgl_dense_loop(const traversal_t& traversal, int64_t m0, int64_t m1,
                           int64_t n0, int64_t n1, int64_t chunk_size,
                           const func_t& f) {
  if constexpr (std::is_same_v<std::decay_t<traversal_t>, DenseAutoWorker>) {
    traversal.loop(m0, m1, n0, n1, f);
  } else {
    loop_2d<int8_t>(m0, m1, n0, n1, chunk_size, f);
  }
}

}  // namespace

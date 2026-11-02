// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <array>
#include <limits>
#include <tuple>
#include <vector>

namespace {

struct DenseAmxOwnership {
  int64_t workers = 0, group_m = 1;
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
        const int64_t row = offset % height;
        const int64_t count = std::min(last - first, height - row);
        emit(m0 + row, m0 + row + count, offset / height, offset / height + 1);
        first += count;
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
};

struct DenseAmxPlan {
  DenseAmxOwnership ownership;
  int64_t m, n, k, a_stride, b_stride, output_stride;
  int64_t macro_m = 256, macro_n = 32, k_block = 1024;
  int64_t replicas = 1;
  bool stream_a = true, stream_output = true;
  double predicted_cold_ms = 0, predicted_warm_ms = 0;

  // The cost model was fitted for 2 MiB of per-core L2.
  static constexpr int64_t cache_bytes = 2 * 1024 * 1024;

  static bool eligible(int64_t m, int64_t n, int64_t k, int64_t team) {
    return team >= 24 && team <= 96 && m >= 256 && n >= 256 && k >= 1024 &&
           n % 32 == 0 && k % 64 == 0 && (m / 32) * (n / 32) >= 32 * team &&
           32 * (k + 64) + 32 * (k + 4) + 4 * 32 * 32 <= cache_bytes;
  }

  static DenseAmxPlan select(int64_t m, int64_t n, int64_t k, int64_t team);

  static const DenseAmxPlan& cached(int64_t m, int64_t n, int64_t k,
                                    int64_t team) {
    struct Entry {
      int64_t team = 0;
      DenseAmxPlan plan;
    };
    static thread_local std::array<Entry, 16> entries;
    static thread_local size_t next = 0;
    for (const auto& entry : entries) {
      if (entry.team == team && entry.plan.m == m && entry.plan.n == n &&
          entry.plan.k == k) {
        return entry.plan;
      }
    }
    auto& entry = entries[next];
    entry.plan = select(m, n, k, team);
    entry.team = team;
    next = (next + 1) % entries.size();
    return entry.plan;
  }

  static DenseAmxPlan make(int64_t m, int64_t n, int64_t k, int64_t team) {
    DenseAmxPlan plan;
    plan.m = m;
    plan.n = n;
    plan.output_stride = n;
    plan.k = k;
    plan.a_stride = k + 64;
    plan.b_stride = k + sizeof(int32_t);
    plan.ownership.workers = team;
    plan.ownership.mb = div_up(m, int64_t(32));
    plan.ownership.nb = n / 32;
    plan.ownership.partial_m = m % 32 != 0;
    plan.bound_ownership();
    return plan;
  }

  void bound_ownership() {
    ownership.group_m = macro_m / 32;
    while (ownership.group_m > 1 &&
           ownership.group_m * 32 * a_stride + macro_n * b_stride +
                   4 * std::min(macro_m, ownership.group_m * 32) * macro_n >
               cache_bytes) {
      ownership.group_m /= 2;
    }
  }

  int64_t c_stride() const { return macro_n + 16; }
  int64_t worker_scratch() const { return macro_m * c_stride(); }
  int64_t activation_bytes() const {
    return div_up(m * (a_stride + int64_t(sizeof(float))), int64_t(64)) * 64;
  }

  DenseAmxPlan cohort(int64_t team, int64_t worker, int64_t& replica,
                      int64_t& rank, int64_t& column) const {
    const int64_t members = team / replicas;
    const int64_t extra = team % replicas;
    const int64_t first = (members + 1) * extra;
    replica = worker < first ? worker / (members + 1)
                             : extra + (worker - first) / members;
    rank = worker - replica * members - std::min(replica, extra);
    const int64_t tiles = n / 32;
    const int64_t width = tiles / replicas;
    const int64_t remainder = tiles % replicas;
    column = 32 * (replica * width + std::min(replica, remainder));
    DenseAmxPlan local = *this;
    local.n = 32 * (width + (replica < remainder));
    local.ownership.nb = local.n / 32;
    local.ownership.workers = members + (replica < extra);
    return local;
  }

  std::pair<double, double> estimate(
      int64_t team, const std::array<double, 9>& coefficients,
      const std::array<double, 2>& quant_coefficients) const {
    double worker_ms = 0, b_delivery = 0, foreign_delivery = 0;
    int64_t max_quant_rows = 0;
    std::vector<std::pair<int64_t, int64_t>> ranges;
    const int64_t blocks = div_up(k, k_block);
    for (int64_t worker = 0; worker < team; ++worker) {
      int64_t replica, rank, column;
      const auto local = cohort(team, worker, replica, rank, column);
      const int64_t members = local.ownership.workers;
      const int64_t quant_rows = m / members + (rank < m % members);
      const int64_t quant_begin =
          rank * (m / members) + std::min(rank, m % members);
      const int64_t quant_end = quant_begin + quant_rows;
      max_quant_rows = std::max(max_quant_rows, quant_rows);
      int64_t elements = 0, partial_c = 0, b_bytes = 0;
      ranges.clear();
      local.ownership.each(rank, [&](int64_t mb0, int64_t mb1, int64_t nb0,
                                     int64_t nb1) {
        const int64_t begin = mb0 * 32, end = std::min(m, mb1 * 32);
        const int64_t rows = end - begin, columns = (nb1 - nb0) * 32;
        ranges.emplace_back(begin, end);
        elements += rows * columns;
        b_bytes += columns * k * div_up(rows, macro_m);
        const int64_t amx_rows = rows % 32 == 0 ? rows : rows - rows % macro_m;
        partial_c += 8 * amx_rows * columns * blocks;
      });
      std::sort(ranges.begin(), ranges.end());
      int64_t owned_rows = 0, local_rows = 0, begin = 0, end = 0;
      auto consume = [&] {
        owned_rows += end - begin;
        local_rows += std::max(int64_t(0), std::min(end, quant_end) -
                                               std::max(begin, quant_begin));
      };
      for (const auto& range : ranges) {
        if (range.first > end) {
          consume();
          begin = range.first;
        }
        end = std::max(end, range.second);
      }
      consume();
      const int64_t foreign_rows = owned_rows - local_rows;
      const double quant_touched = quant_rows * (3.0 * k + 4);
      const double evicted = std::max(0.0, 1.0 - cache_bytes / quant_touched);
      const double ops = 2.0 * elements * k / 1e9;
      const double output = 2.0 * elements / 1e9;
      const double a_bytes = k * (foreign_rows + local_rows * evicted) / 1e9;
      const double value =
          coefficients[0] + coefficients[stream_a ? 2 : 1] * ops +
          coefficients[3] * partial_c / 1e9 +
          (stream_output ? 0 : coefficients[4] * output) +
          coefficients[5] * b_bytes / 1e9 + coefficients[6] * a_bytes;
      worker_ms = std::max(worker_ms, value);
      b_delivery += b_bytes / 1e9;
      foreign_delivery += double(k) * foreign_rows / 1e9;
    }
    const double quant_ms = quant_coefficients[0] + quant_coefficients[1] *
                                                        double(max_quant_rows) *
                                                        k / 1e9;
    // Cache delivery proxies and physical allocation are separate from DRAM
    // traffic.
    constexpr double socket_bytes_per_ms = 600.0 * (1LL << 30) / 1000;
    const double warm =
        std::max(quant_ms + worker_ms, 2.0 * m * n / socket_bytes_per_ms);
    const double cold =
        std::max(quant_ms + worker_ms + coefficients[7] * b_delivery / 96 +
                     coefficients[8] * foreign_delivery / 96,
                 (double(n) * (k + 4) + 2.0 * m * n) / socket_bytes_per_ms);
    return {cold, warm};
  }
};

inline DenseAmxPlan DenseAmxPlan::select(int64_t m, int64_t n, int64_t k,
                                         int64_t team) {
  // Offline fit on five wide/square/tall/long-K shapes at 24, 48 and 96
  // workers. Effective costs use milliseconds, decimal giga-operations and GB.
  constexpr std::array<double, 9> coefficients = {
      0.11841670710624594, 0.29150110196911205, 0.28101257453302225,
      13.529205116676746,  65.84468821122748,   29.83490185005042,
      83.87315374503517,   49.55831281750576,   1051.1253187193824};
  constexpr std::array<double, 2> quant_coefficients = {0.0015961875918527287,
                                                        242.20650849947816};
  std::array<DenseAmxPlan, 72> candidates;
  size_t count = 0;
  double minimum = std::numeric_limits<double>::infinity();
  auto add = [&](int64_t mm, int64_t nn, int64_t kb, bool stream_a,
                 bool stream_output, int64_t replicas) {
    auto plan = make(m, n, k, team);
    plan.macro_m = mm;
    plan.macro_n = nn;
    plan.k_block = kb;
    plan.stream_a = stream_a;
    plan.stream_output = stream_output;
    plan.replicas = replicas;
    plan.bound_ownership();
    if (32 * plan.a_stride + nn * plan.b_stride + 4 * 32 * nn > cache_bytes) {
      return;
    }
    const auto latency = plan.estimate(team, coefficients, quant_coefficients);
    plan.predicted_cold_ms = latency.first;
    plan.predicted_warm_ms = latency.second;
    minimum = std::min(minimum, latency.first);
    candidates[count++] = plan;
  };
  for (int64_t mm : {128, 256}) {
    for (int64_t nn : {32, 64}) {
      for (int64_t kb : {int64_t(1024), int64_t(2048), k}) {
        for (bool stream_a : {false, true}) {
          for (bool stream_output : {false, true}) {
            add(mm, nn, kb, stream_a, stream_output, 1);
          }
        }
      }
    }
  }
  for (int64_t replicas : {2, 3, 4, 6}) {
    for (bool stream_output : {false, true}) {
      add(256, 32, 1024, true, stream_output, replicas);
      add(128, 32, k, true, stream_output, replicas);
      add(256, 64, k, false, stream_output, replicas);
    }
  }
  auto key = [&](const DenseAmxPlan& plan) {
    return std::make_tuple(plan.replicas,
                           4 * team * plan.worker_scratch() +
                               plan.replicas * plan.activation_bytes(),
                           plan.predicted_cold_ms, plan.macro_m, plan.macro_n,
                           plan.k_block, plan.stream_a, plan.stream_output);
  };
  size_t chosen = 0;
  bool found = false;
  for (size_t index = 0; index < count; ++index) {
    if (candidates[index].predicted_cold_ms <= 1.01 * minimum &&
        (!found || key(candidates[index]) < key(candidates[chosen]))) {
      chosen = index;
      found = true;
    }
  }
  return candidates[chosen];
}

// Region-local execution. Each row region's workers split its columns; worker
// pairs share one activation copy of the region's rows, which both produce.
struct DenseAmxLocalPlan {
  struct Worker {
    int64_t m_begin, m_end, n_begin, n_end;
    int64_t activation_offset, producer_begin, producer_end;
  };
  DenseAmxPlan kernel;
  std::vector<Worker> workers;
  int64_t activation_bytes = 0;

  static bool eligible(int64_t m, int64_t n, int64_t k, int64_t team) {
    return team >= 1 && team <= 96 && m >= 256 && n >= 256 && k >= 512 &&
           k <= 4096 && n % 32 == 0 && k % 64 == 0 &&
           div_up(m, int64_t(32)) * (n / 32) >= team;
  }

  static DenseAmxLocalPlan make(int64_t m, int64_t n, int64_t k, int64_t team) {
    DenseAmxLocalPlan plan;
    plan.kernel = DenseAmxPlan::make(m, n, k, team);
    plan.kernel.k_block = std::min<int64_t>(1024, k);
    plan.workers.resize(team);
    const int64_t mb = div_up(m, int64_t(32)), nb = n / 32;
    const int64_t regions = std::min(
        mb,
        std::max(std::min(div_up(m, int64_t(256)), team), div_up(team, nb)));
    auto members = [&](int64_t region) {
      return team / regions + (region < team % regions);
    };
    for (int64_t region = 0; region < regions; ++region) {
      const int64_t count = members(region);
      const int64_t copies = div_up(count, int64_t(2));
      const int64_t first =
          region * (team / regions) + std::min(region, team % regions);
      const int64_t m_begin = 32 * (region + (mb - regions) * first / team);
      const int64_t m_end = std::min(
          m, 32 * (region + 1 + (mb - regions) * (first + count) / team));
      const int64_t rows = m_end - m_begin;
      for (int64_t rank = 0; rank < count; ++rank) {
        const int64_t replica = rank % copies;
        const int64_t producer = rank / copies;
        const int64_t producers = (count - 1 - replica) / copies + 1;
        const int64_t begin = producer * (rows / producers) +
                              std::min(producer, rows % producers);
        plan.workers[region + rank * regions] = {
            m_begin,
            m_end,
            32 * (rank * (nb / count) + std::min(rank, nb % count)),
            32 * ((rank + 1) * (nb / count) + std::min(rank + 1, nb % count)),
            0,
            begin,
            begin + rows / producers + (producer < rows % producers)};
      }
    }
    // Interleave copies by replica so pair-shared segments are contiguous.
    const int64_t max_copies = div_up(div_up(team, regions), int64_t(2));
    for (int64_t replica = 0; replica < max_copies; ++replica) {
      for (int64_t region = 0; region < regions; ++region) {
        const int64_t count = members(region);
        const int64_t copies = div_up(count, int64_t(2));
        if (replica >= copies) {
          continue;
        }
        const auto& owner = plan.workers[region];
        for (int64_t rank = replica; rank < count; rank += copies) {
          plan.workers[region + rank * regions].activation_offset =
              plan.activation_bytes;
        }
        plan.activation_bytes +=
            div_up((owner.m_end - owner.m_begin) *
                       (plan.kernel.a_stride + int64_t(sizeof(float))),
                   int64_t(64)) *
            64;
      }
    }
    return plan;
  }

  // Largest worker tile count over the mean.
  double imbalance() const {
    double largest = 0, total = 0;
    for (const auto& owner : workers) {
      const double tiles =
          double(owner.m_end - owner.m_begin) * (owner.n_end - owner.n_begin);
      largest = std::max(largest, tiles);
      total += tiles;
    }
    return largest * workers.size() / total;
  }

  // Measured on the cold operator: local execution beats single-replica
  // fallback and replaces cohorts unless K1024 or uneven regions make it
  // slower. Without a cohort plan, small outputs keep the generic path, which
  // avoids per-pair activation copies that dominate warm calls.
  bool preferred(const DenseAmxPlan* cohort) const {
    if (cohort == nullptr) {
      return kernel.ownership.mb * kernel.ownership.nb >=
             32 * int64_t(workers.size());
    }
    if (cohort->replicas == 1) {
      return true;
    }
    return kernel.k > 1024 && imbalance() <= (kernel.m <= 2048 ? 1.16 : 1.05);
  }

  // The single-worker kernel plan for `worker`'s rectangle.
  DenseAmxPlan local(int64_t worker) const {
    const auto& owner = workers[worker];
    auto plan = kernel;
    plan.m = owner.m_end - owner.m_begin;
    plan.n = owner.n_end - owner.n_begin;
    plan.ownership.mb = div_up(plan.m, int64_t(32));
    plan.ownership.nb = plan.n / 32;
    plan.ownership.workers = 1;
    plan.ownership.partial_m = plan.m % 32 != 0;
    return plan;
  }
};

}  // namespace

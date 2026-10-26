// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <limits>
#include <numeric>
#include <string>

namespace {

enum class SGLKernelKind { Dense, MoE };
enum class SGLSchedule { Legacy, RoundRobin, Spread, Tile };

struct SGLSchedulerConfig {
  bool dense = true;
  bool moe = true;
  SGLSchedule mode = SGLSchedule::Legacy;

  SGLSchedulerConfig() {
    const char* value = std::getenv("VLLM_CPU_SGL_SCHEDULER");
    if (value == nullptr || *value == '\0') {
      const char* groups = std::getenv("VLLM_CPU_SGL_THREAD_GROUPS");
      mode = groups != nullptr && *groups != '\0' ? SGLSchedule::RoundRobin
                                                  : SGLSchedule::Legacy;
      return;
    }
    if (std::strcmp(value, "legacy") == 0) {
      return;
    }
    if (std::strcmp(value, "round_robin") == 0) {
      mode = SGLSchedule::RoundRobin;
      return;
    }
    const std::string name(value);
    if (name.rfind("spread_", 0) == 0) {
      mode = SGLSchedule::Spread;
    } else if (name.rfind("tile_", 0) == 0) {
      mode = SGLSchedule::Tile;
    } else {
      TORCH_CHECK(false, "Invalid VLLM_CPU_SGL_SCHEDULER: ", value);
    }
    const auto target = name.substr(name.find('_') + 1);
    TORCH_CHECK(target == "moe" || target == "dense" || target == "all",
                "Invalid VLLM_CPU_SGL_SCHEDULER: ", value);
    dense = target != "moe";
    moe = target != "dense";
    TORCH_CHECK(sgl_thread_groups().total > 0,
                "VLLM_CPU_SGL_SCHEDULER requires VLLM_CPU_SGL_THREAD_GROUPS.");
  }
};

inline SGLSchedule sgl_schedule(SGLKernelKind kind) {
  static const SGLSchedulerConfig config;
  if ((kind == SGLKernelKind::Dense && !config.dense) ||
      (kind == SGLKernelKind::MoE && !config.moe)) {
    return SGLSchedule::Legacy;
  }
  return config.mode;
}

inline bool sgl_panel_schedule(SGLSchedule mode) {
  return mode == SGLSchedule::Spread || mode == SGLSchedule::Tile;
}

struct SGLTileCost {
  double bytes = 0;
  double ops = 0;

  SGLTileCost& operator+=(const SGLTileCost& other) {
    bytes += other.bytes;
    ops += other.ops;
    return *this;
  }
};

inline SGLTileCost operator+(SGLTileCost a, const SGLTileCost& b) {
  return a += b;
}

inline double sgl_cost_score(const SGLTileCost& cost, const SGLTileCost& total,
                             double capacity = 1) {
  return std::max(total.bytes > 0 ? cost.bytes / total.bytes / capacity : 0,
                  total.ops > 0 ? cost.ops / total.ops / capacity : 0);
}

struct SGLTileTask {
  int64_t mb0, mb1, nb0, nb1;
  int expert, group;
  SGLTileCost cost;
};

struct SGLTilePlan {
  std::vector<SGLTileTask> tasks;
  std::vector<int> worker_offsets;
  std::vector<int> task_indices;
  std::vector<SGLTileCost> worker_costs;
  std::vector<int> group_sizes;
};

struct SGLMoEPlans {
  SGLTilePlan gate, down;
};

inline const SGLThreadGroups& sgl_tile_groups() {
  const auto& groups = sgl_thread_groups();
  TORCH_CHECK(groups.total > 0,
              "VLLM_CPU_SGL_SCHEDULER requires VLLM_CPU_SGL_THREAD_GROUPS.");
  groups.work_ids(at::get_num_threads());
#if !defined(_OPENMP)
  TORCH_CHECK(false, "SGL tile scheduling requires OpenMP.");
#endif
  return groups;
}

inline SGLTileCost sgl_panel_cost(int64_t rows, int64_t columns, int64_t k,
                                  int copies) {
  return {double(columns) * (k + sizeof(int32_t)) * copies,
          2.0 * rows * columns * k * copies};
}

inline void sgl_pack_worker_bins(SGLTilePlan& plan,
                                 const std::vector<int>& owners) {
  const int nth = sgl_tile_groups().total;
  plan.worker_offsets.assign(nth + 1, 0);
  plan.worker_costs.resize(nth);
  for (size_t i = 0; i < owners.size(); ++i) {
    ++plan.worker_offsets[owners[i] + 1];
    plan.worker_costs[owners[i]] += plan.tasks[i].cost;
  }
  std::partial_sum(plan.worker_offsets.begin(), plan.worker_offsets.end(),
                   plan.worker_offsets.begin());
  auto next = plan.worker_offsets;
  plan.task_indices.resize(owners.size());
  for (size_t i = 0; i < owners.size(); ++i) {
    plan.task_indices[next[owners[i]]++] = i;
  }
}

inline SGLTilePlan make_sgl_dense_plan(int64_t mb, int64_t nb, int64_t rows,
                                       int64_t columns, int64_t k, int copies) {
  const auto& groups = sgl_tile_groups();
  SGLTilePlan plan;
  plan.group_sizes = groups.sizes;
  std::vector<int> owners;
  std::vector<int64_t> counts(groups.sizes.size());
  std::vector<int> order(groups.sizes.size());
  std::iota(order.begin(), order.end(), 0);
  int64_t remainder = nb;
  for (size_t g = 0; g < counts.size(); ++g) {
    counts[g] = nb * groups.sizes[g] / groups.total;
    remainder -= counts[g];
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return nb * groups.sizes[a] % groups.total >
           nb * groups.sizes[b] % groups.total;
  });
  for (int64_t i = 0; i < remainder; ++i) {
    ++counts[order[i]];
  }
  int worker_start = 0;
  int64_t nstart = 0;
  auto add = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1, int group,
                 int worker) {
    if (m0 == m1 || n0 == n1) {
      return;
    }
    const int64_t nr = std::min(rows, m1 * 32) - m0 * 32;
    const int64_t nc = std::min(columns, n1 * 32) - n0 * 32;
    plan.tasks.push_back(
        {m0, m1, n0, n1, -1, group, sgl_panel_cost(nr, nc, k, copies)});
    owners.push_back(worker);
  };
  for (size_t g = 0; g < counts.size(); ++g) {
    const int threads = groups.sizes[g];
    const int64_t ncount = counts[g];
    for (int w = 0; w < threads && ncount > 0 && mb > 0; ++w) {
      if (ncount >= threads) {
        add(0, mb, nstart + ncount * w / threads,
            nstart + ncount * (w + 1) / threads, g, worker_start + w);
      } else {
        // Column-major intervals give at most three rectangles per worker.
        int64_t pos = mb * ncount * w / threads;
        const int64_t end = mb * ncount * (w + 1) / threads;
        while (pos < end) {
          const int64_t n0 = pos / mb;
          const int64_t m0 = pos % mb;
          if (m0 == 0 && end - pos >= mb) {
            const int64_t full = (end - pos) / mb;
            add(0, mb, nstart + n0, nstart + n0 + full, g, worker_start + w);
            pos += full * mb;
          } else {
            const int64_t m1 = std::min(mb, m0 + end - pos);
            add(m0, m1, nstart + n0, nstart + n0 + 1, g, worker_start + w);
            pos += m1 - m0;
          }
        }
      }
    }
    worker_start += threads;
    nstart += ncount;
  }
  sgl_pack_worker_bins(plan, owners);
  return plan;
}

inline SGLTileCost sgl_moe_task_cost(const SGLTileTask& task,
                                     const int32_t* offsets, int64_t columns,
                                     int64_t k, int copies) {
  return sgl_panel_cost(offsets[task.mb1] - offsets[task.mb0],
                        std::min(columns, task.nb1 * 32) - task.nb0 * 32, k,
                        copies);
}

inline std::vector<double> sgl_tile_loads(const std::vector<SGLTileTask>& tasks,
                                          const std::vector<int>& sizes,
                                          const SGLTileCost& total) {
  std::vector<SGLTileCost> costs(sizes.size());
  for (const auto& task : tasks) {
    costs[task.group] += task.cost;
  }
  const int nth = std::accumulate(sizes.begin(), sizes.end(), 0);
  std::vector<double> loads;
  for (size_t g = 0; g < sizes.size(); ++g) {
    loads.push_back(sgl_cost_score(costs[g], total, double(sizes[g]) / nth));
  }
  std::sort(loads.begin(), loads.end(), std::greater<double>());
  return loads;
}

inline int sgl_best_group(const SGLTileCost& cost,
                          const std::vector<SGLTileCost>& loads,
                          const std::vector<int>& sizes,
                          const SGLTileCost& total) {
  const int nth = std::accumulate(sizes.begin(), sizes.end(), 0);
  int best = 0;
  double score = std::numeric_limits<double>::infinity();
  for (size_t g = 0; g < sizes.size(); ++g) {
    const double next =
        sgl_cost_score(loads[g] + cost, total, double(sizes[g]) / nth);
    if (next < score) {
      best = g;
      score = next;
    }
  }
  return best;
}

inline void sgl_assign_tiles(std::vector<SGLTileTask>& tasks,
                             const std::vector<int>& sizes,
                             const SGLTileCost& total) {
  std::vector<int> order(tasks.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return sgl_cost_score(tasks[a].cost, total) >
           sgl_cost_score(tasks[b].cost, total);
  });
  std::vector<SGLTileCost> loads(sizes.size());
  for (int i : order) {
    const int best = sgl_best_group(tasks[i].cost, loads, sizes, total);
    tasks[i].group = best;
    loads[best] += tasks[i].cost;
  }
}

inline SGLTilePlan sgl_plan_moe_stage(std::vector<SGLTileTask> tiles,
                                      const int32_t* offsets, int64_t columns,
                                      int64_t k, int copies,
                                      const std::vector<int>& sizes) {
  SGLTileCost total;
  for (const auto& task : tiles) {
    total += task.cost;
  }
  auto loads = sgl_tile_loads(tiles, sizes, total);
  while (!loads.empty() && loads.front() > 1.10) {
    std::vector<SGLTileCost> costs(sizes.size());
    for (const auto& task : tiles) {
      costs[task.group] += task.cost;
    }
    const int nth = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<int> order(tiles.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return sgl_cost_score(tiles[a].cost, total) >
             sgl_cost_score(tiles[b].cost, total);
    });
    bool improved = false;
    for (int i : order) {
      const auto original = tiles[i];
      if (original.nb1 - original.nb0 < 2 ||
          sgl_cost_score(costs[original.group], total,
                         double(sizes[original.group]) / nth) <= 1.10) {
        continue;
      }
      int64_t width = div_up(original.nb1 - original.nb0, int64_t(2));
      while (true) {
        auto trial = tiles;
        auto remaining = costs;
        remaining[original.group].bytes -= original.cost.bytes;
        remaining[original.group].ops -= original.cost.ops;
        // Keep unchanged experts on the same tile in both GEMM stages.
        for (int64_t n = original.nb0; n < original.nb1; n += width) {
          auto part = original;
          part.nb0 = n;
          part.nb1 = std::min(original.nb1, n + width);
          part.cost = sgl_moe_task_cost(part, offsets, columns, k, copies);
          part.group = sgl_best_group(part.cost, remaining, sizes, total);
          remaining[part.group] += part.cost;
          if (n == original.nb0) {
            trial[i] = part;
          } else {
            trial.push_back(part);
          }
        }
        auto next = sgl_tile_loads(trial, sizes, total);
        if (next < loads) {
          tiles = std::move(trial);
          loads = std::move(next);
          improved = true;
          break;
        }
        if (width == 1) {
          break;
        }
        width = div_up(width, int64_t(2));
      }
      if (improved) {
        break;
      }
    }
    if (!improved) {
      break;
    }
  }

  SGLTilePlan plan;
  plan.group_sizes = sizes;
  std::vector<SGLTileCost> group_costs(sizes.size());
  for (const auto& task : tiles) {
    group_costs[task.group] += task.cost;
  }
  const int64_t cache_columns = get_cache_blocks<int8_t>(32 * k * copies);
  for (const auto& tile : tiles) {
    const double cost = sgl_cost_score(tile.cost, group_costs[tile.group]);
    const int64_t width = std::min(
        cache_columns,
        std::max<int64_t>(
            1, (tile.nb1 - tile.nb0) /
                   std::max(1.0, std::ceil(2 * sizes[tile.group] * cost))));
    for (int64_t n = tile.nb0; n < tile.nb1; n += width) {
      auto task = tile;
      task.nb0 = n;
      task.nb1 = std::min(tile.nb1, n + width);
      task.cost = sgl_moe_task_cost(task, offsets, columns, k, copies);
      plan.tasks.push_back(task);
    }
  }
  for (size_t g = 0; g < sizes.size(); ++g) {
    int jobs =
        std::count_if(plan.tasks.begin(), plan.tasks.end(),
                      [&](const auto& task) { return task.group == int(g); });
    while (jobs < sizes[g]) {
      int best = -1;
      double score = -1;
      for (size_t i = 0; i < plan.tasks.size(); ++i) {
        const auto& task = plan.tasks[i];
        const double value = sgl_cost_score(task.cost, group_costs[g]);
        if (task.group == int(g) && task.mb1 - task.mb0 > 1 && value > score) {
          best = i;
          score = value;
        }
      }
      if (best < 0) {
        break;
      }
      auto second = plan.tasks[best];
      const int64_t middle = (second.mb0 + second.mb1) / 2;
      plan.tasks[best].mb1 = middle;
      second.mb0 = middle;
      plan.tasks[best].cost =
          sgl_moe_task_cost(plan.tasks[best], offsets, columns, k, copies);
      second.cost = sgl_moe_task_cost(second, offsets, columns, k, copies);
      plan.tasks.push_back(second);
      ++jobs;
    }
  }
  std::vector<int> order(plan.tasks.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return sgl_cost_score(plan.tasks[a].cost, total) >
           sgl_cost_score(plan.tasks[b].cost, total);
  });
  const int nth = std::accumulate(sizes.begin(), sizes.end(), 0);
  std::vector<SGLTileCost> workers(nth);
  std::vector<int> owners(plan.tasks.size());
  for (int i : order) {
    const auto& task = plan.tasks[i];
    const int first =
        std::accumulate(sizes.begin(), sizes.begin() + task.group, 0);
    int best = first;
    double score = std::numeric_limits<double>::infinity();
    for (int w = first; w < first + sizes[task.group]; ++w) {
      const double next =
          sgl_cost_score(workers[w] + task.cost, group_costs[task.group]);
      if (next < score) {
        score = next;
        best = w;
      }
    }
    owners[i] = best;
    workers[best] += task.cost;
  }
  sgl_pack_worker_bins(plan, owners);
  return plan;
}

inline SGLMoEPlans make_sgl_moe_plans(const int32_t* expert_ids,
                                      const int32_t* offsets, int64_t mb,
                                      int64_t n, int64_t k) {
  if (!sgl_panel_schedule(sgl_schedule(SGLKernelKind::MoE))) {
    return {};
  }
  const auto& groups = sgl_tile_groups();
  std::vector<SGLTileTask> experts;
  SGLTileCost total;
  for (int64_t begin = 0; begin < mb;) {
    int64_t end = begin + 1;
    while (end < mb && expert_ids[end] == expert_ids[begin]) {
      ++end;
    }
    if (offsets[end] > offsets[begin]) {
      const int64_t rows = offsets[end] - offsets[begin];
      const auto cost =
          sgl_panel_cost(rows, n, k, 2) + sgl_panel_cost(rows, k, n, 1);
      experts.push_back(
          {begin, end, 0, div_up(n, int64_t(32)), expert_ids[begin], 0, cost});
      total += cost;
    }
    begin = end;
  }
  sgl_assign_tiles(experts, groups.sizes, total);
  auto gate = experts;
  auto down = experts;
  for (auto& task : gate) {
    task.cost = sgl_moe_task_cost(task, offsets, n, k, 2);
  }
  for (auto& task : down) {
    task.nb1 = div_up(k, int64_t(32));
    task.cost = sgl_moe_task_cost(task, offsets, k, n, 1);
  }
  return {sgl_plan_moe_stage(std::move(gate), offsets, n, k, 2, groups.sizes),
          sgl_plan_moe_stage(std::move(down), offsets, k, n, 1, groups.sizes)};
}

template <typename func_t>
inline void sgl_run_tile_plan(const SGLTilePlan& plan, SGLSchedule mode,
                              const func_t& f) {
  const auto& groups = sgl_tile_groups();
  const int nth = groups.total;
  int actual = 0;
#if defined(_OPENMP)
  #pragma omp parallel num_threads(nth)
  {
  #pragma omp single
    actual = omp_get_num_threads();
    if (actual == nth) {
      const int physical = omp_get_thread_num();
      const int worker = mode == SGLSchedule::Spread
                             ? groups.full_work_ids[physical]
                             : physical;
      f([&](const auto& rectangle) {
        for (int i = plan.worker_offsets[worker];
             i < plan.worker_offsets[worker + 1]; ++i) {
          const auto& task = plan.tasks[plan.task_indices[i]];
          rectangle(task.mb0, task.mb1, task.nb0, task.nb1);
        }
      });
    }
  }
#endif
  TORCH_CHECK(actual == nth,
              "SGL tile scheduling requires the configured OpenMP team.");
}

template <typename func_t>
inline void parallel_sgl_dense(int64_t mb, int64_t nb, int64_t rows,
                               int64_t columns, int64_t k, int copies,
                               const func_t& f) {
  const auto mode = sgl_schedule(SGLKernelKind::Dense);
  if (sgl_panel_schedule(mode)) {
    const auto plan = make_sgl_dense_plan(mb, nb, rows, columns, k, copies);
    sgl_run_tile_plan(plan, mode, f);
  } else {
    parallel_2d(
        mb, nb,
        [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
          f([&](const auto& rectangle) { rectangle(m0, m1, n0, n1); });
        },
        mode == SGLSchedule::RoundRobin);
  }
}

template <typename func_t>
inline void parallel_sgl_moe(const SGLTilePlan& plan, int64_t mb, int64_t nb,
                             const func_t& f) {
  const auto mode = sgl_schedule(SGLKernelKind::MoE);
  if (sgl_panel_schedule(mode)) {
    sgl_run_tile_plan(plan, mode, f);
  } else {
    parallel_2d(
        mb, nb,
        [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
          f([&](const auto& rectangle) { rectangle(m0, m1, n0, n1); });
        },
        mode == SGLSchedule::RoundRobin);
  }
}

}  // namespace

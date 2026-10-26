// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "common.h"
#include "tile_scheduler.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sched.h>
#include <set>
#include <sstream>
#include <tuple>

namespace {

struct Config {
  int threads;
  int m_blocks;
  int n_blocks;
  std::string kind = "dense";
  std::string phase = "gate";
  bool timing = false;
  int64_t k = 2048;
  int64_t rows = -1;
  int64_t columns = -1;
  std::vector<int64_t> expert_rows;
};

struct Observation {
  int cpu = -1;
  int work_id = -1;
  int tile = -1;
  int physical_tile = -1;
  int team_size = 0;
  std::vector<int> task_ids;
  std::vector<std::array<int64_t, 4>> rectangles;
};

std::string require_value(int& index, int argc, char** argv) {
  TORCH_CHECK(index + 1 < argc, "Missing value for ", argv[index]);
  return argv[++index];
}

int64_t parse_integer(const std::string& value, const std::string& name) {
  size_t end = 0;
  const auto parsed = std::stoll(value, &end);
  TORCH_CHECK(end == value.size(), "Invalid integer for ", name, ": ", value);
  return parsed;
}

std::vector<int64_t> parse_expert_rows(const std::string& value) {
  std::vector<int64_t> rows;
  std::stringstream input(value);
  std::string item;
  while (std::getline(input, item, ',')) {
    const int64_t count = parse_integer(item, "--expert-rows");
    TORCH_CHECK(count >= 0, "Expert row counts must be nonnegative.");
    rows.push_back(count);
  }
  TORCH_CHECK(!rows.empty(), "--expert-rows must contain at least one count.");
  return rows;
}

Config parse_config(int argc, char** argv) {
  TORCH_CHECK(argc >= 4,
              "Expected positional arguments: threads, M blocks, N blocks");
  Config config{std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])};
  for (int i = 4; i < argc; ++i) {
    const std::string option(argv[i]);
    if (option == "--timing") {
      config.timing = true;
    } else if (option == "--kind") {
      config.kind = require_value(i, argc, argv);
    } else if (option == "--phase") {
      config.phase = require_value(i, argc, argv);
    } else if (option == "--rows") {
      config.rows = parse_integer(require_value(i, argc, argv), option);
    } else if (option == "--columns") {
      config.columns = parse_integer(require_value(i, argc, argv), option);
    } else if (option == "--k") {
      config.k = parse_integer(require_value(i, argc, argv), option);
    } else if (option == "--expert-rows") {
      config.expert_rows = parse_expert_rows(require_value(i, argc, argv));
    } else {
      TORCH_CHECK(false, "Unknown option: ", option);
    }
  }
  TORCH_CHECK(config.threads > 0 && config.m_blocks > 0 &&
                  config.n_blocks > 0 && config.k > 0,
              "Threads, block counts, and K must be positive.");
  TORCH_CHECK(config.kind == "dense" || config.kind == "moe",
              "--kind must be dense or moe.");
  TORCH_CHECK(config.phase == "gate" || config.phase == "down",
              "--phase must be gate or down.");
  TORCH_CHECK(config.rows == -1 || config.rows > 0, "--rows must be positive.");
  TORCH_CHECK(config.columns == -1 || config.columns > 0,
              "--columns must be positive.");
  TORCH_CHECK(config.kind != "dense" ||
                  (config.phase == "gate" && config.expert_rows.empty()),
              "--phase and --expert-rows only apply to MoE.");
  TORCH_CHECK(config.kind != "moe" || config.rows == -1,
              "--rows only applies to dense; use --expert-rows for MoE.");
  return config;
}

const char* schedule_name(SGLSchedule mode) {
  switch (mode) {
    case SGLSchedule::Legacy:
      return "legacy";
    case SGLSchedule::RoundRobin:
      return "round_robin";
    case SGLSchedule::Spread:
      return "spread";
    case SGLSchedule::Tile:
      return "tile";
  }
  return "unknown";
}

int tile_for_worker(int worker, const std::vector<int>& group_sizes) {
  int begin = 0;
  for (size_t group = 0; group < group_sizes.size(); ++group) {
    begin += group_sizes[group];
    if (worker < begin) {
      return static_cast<int>(group);
    }
  }
  return -1;
}

void add_rectangle(std::vector<int>& coverage, int64_t mb, int64_t nb,
                   const std::array<int64_t, 4>& rectangle) {
  const auto [mb0, mb1, nb0, nb1] = rectangle;
  TORCH_CHECK(mb0 >= 0 && mb0 <= mb1 && mb1 <= mb && nb0 >= 0 && nb0 <= nb1 &&
                  nb1 <= nb,
              "Scheduled rectangle lies outside the matrix blocks.");
  for (int64_t m = mb0; m < mb1; ++m) {
    for (int64_t n = nb0; n < nb1; ++n) {
      ++coverage[static_cast<size_t>(m) * nb + n];
    }
  }
}

void check_coverage(const std::vector<int>& coverage, int64_t mb, int64_t nb,
                    const std::vector<int32_t>& offsets, bool dense) {
  for (int64_t m = 0; m < mb; ++m) {
    const bool active = dense || offsets[m + 1] > offsets[m];
    for (int64_t n = 0; n < nb; ++n) {
      const int expected = active ? 1 : 0;
      TORCH_CHECK(coverage[static_cast<size_t>(m) * nb + n] == expected,
                  "Matrix block (", m, ", ", n,
                  ") has unexpected task coverage.");
    }
  }
}

std::vector<int64_t> effective_expert_rows(const Config& config) {
  if (!config.expert_rows.empty()) {
    return config.expert_rows;
  }
  return std::vector<int64_t>(config.m_blocks, 4);
}

void pack_expert_blocks(const std::vector<int64_t>& rows,
                        std::vector<int32_t>& expert_ids,
                        std::vector<int32_t>& offsets) {
  offsets.push_back(0);
  int64_t total_rows = 0;
  for (size_t expert = 0; expert < rows.size(); ++expert) {
    int64_t remaining = rows[expert];
    total_rows += remaining;
    TORCH_CHECK(total_rows <= INT32_MAX,
                "Total expert rows exceed the supported offset range.");
    while (remaining > 0) {
      const int32_t block_rows =
          static_cast<int32_t>(std::min<int64_t>(32, remaining));
      expert_ids.push_back(static_cast<int32_t>(expert));
      offsets.push_back(offsets.back() + block_rows);
      remaining -= block_rows;
    }
  }
}

void write_cost(const SGLTileCost& cost) {
  std::cout << "{\"bytes\":" << cost.bytes << ",\"ops\":" << cost.ops << '}';
}

void write_histogram(const std::vector<int64_t>& rows) {
  std::map<int64_t, int> histogram;
  for (int64_t count : rows) {
    ++histogram[count];
  }
  std::cout << "\"row_histogram\":{";
  bool first = true;
  for (const auto& [count, experts] : histogram) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << '"' << count << "\":" << experts;
  }
  std::cout << "},\"expert_rows\":[";
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << rows[i];
  }
  std::cout << ']';
}

uint64_t plan_hash(const SGLTilePlan& plan) {
  std::ostringstream canonical;
  canonical << std::setprecision(17);
  for (int size : plan.group_sizes) {
    canonical << size << ',';
  }
  canonical << ';';
  for (const auto& task : plan.tasks) {
    canonical << task.mb0 << ',' << task.mb1 << ',' << task.nb0 << ','
              << task.nb1 << ',' << task.expert << ',' << task.group << ','
              << task.cost.bytes << ',' << task.cost.ops << ';';
  }
  canonical << '|';
  for (int value : plan.worker_offsets) {
    canonical << value << ',';
  }
  canonical << ';';
  for (int value : plan.task_indices) {
    canonical << value << ',';
  }
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : canonical.str()) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

void write_panel_report(const SGLTilePlan& plan,
                        const std::vector<Observation>& observations,
                        const std::vector<int>& task_workers,
                        const std::vector<int>& task_cpus,
                        const std::vector<int>& task_physical_tiles) {
  const int nth =
      std::accumulate(plan.group_sizes.begin(), plan.group_sizes.end(), 0);
  SGLTileCost total;
  std::vector<SGLTileCost> tile_costs(plan.group_sizes.size());
  std::vector<SGLTileCost> physical_tile_costs(plan.group_sizes.size());
  for (const auto& task : plan.tasks) {
    total += task.cost;
    tile_costs[task.group] += task.cost;
  }
  using Panel = std::tuple<int, int64_t, int64_t>;
  std::map<Panel, double> panel_bytes;
  std::map<Panel, std::set<int>> logical_panel_tiles;
  std::map<Panel, std::set<int>> physical_panel_tiles;
  for (size_t i = 0; i < plan.tasks.size(); ++i) {
    const auto& task = plan.tasks[i];
    const Panel panel{task.expert, task.nb0, task.nb1};
    panel_bytes.try_emplace(panel, task.cost.bytes);
    logical_panel_tiles[panel].insert(task.group);
    physical_panel_tiles[panel].insert(task_physical_tiles[i]);
    physical_tile_costs[task_physical_tiles[i]] += task.cost;
  }
  std::vector<int> logical_panels_by_tile(plan.group_sizes.size());
  std::vector<int> physical_panels_by_tile(plan.group_sizes.size());
  int logical_panel_instances = 0;
  int physical_panel_instances = 0;
  int physical_replicated_panels = 0;
  int extra_physical_replicas = 0;
  double logical_panel_touch_bytes = 0;
  double physical_panel_touch_bytes = 0;
  for (const auto& [panel, bytes] : panel_bytes) {
    const auto& logical_owners = logical_panel_tiles[panel];
    const auto& physical_owners = physical_panel_tiles[panel];
    logical_panel_instances += logical_owners.size();
    physical_panel_instances += physical_owners.size();
    logical_panel_touch_bytes += bytes * logical_owners.size();
    physical_panel_touch_bytes += bytes * physical_owners.size();
    if (physical_owners.size() > 1) {
      ++physical_replicated_panels;
      extra_physical_replicas += physical_owners.size() - 1;
    }
    for (int owner : logical_owners) {
      ++logical_panels_by_tile[owner];
    }
    for (int owner : physical_owners) {
      ++physical_panels_by_tile[owner];
    }
  }
  const double max_tile_load = [&] {
    double load = 0;
    for (size_t tile = 0; tile < tile_costs.size(); ++tile) {
      load =
          std::max(load, sgl_cost_score(tile_costs[tile], total,
                                        double(plan.group_sizes[tile]) / nth));
    }
    return load;
  }();
  const double max_physical_tile_load = [&] {
    double load = 0;
    for (size_t tile = 0; tile < physical_tile_costs.size(); ++tile) {
      load =
          std::max(load, sgl_cost_score(physical_tile_costs[tile], total,
                                        double(plan.group_sizes[tile]) / nth));
    }
    return load;
  }();
  const double max_worker_load = [&] {
    double load = 0;
    for (const auto& cost : plan.worker_costs) {
      load = std::max(load, sgl_cost_score(cost, total, 1.0 / nth));
    }
    return load;
  }();

  std::ostringstream hash_text;
  hash_text << std::hex << plan_hash(plan);
  std::cout << "\"plan_hash\":\"" << hash_text.str()
            << "\",\"tile_group_sizes\":[";
  for (size_t i = 0; i < plan.group_sizes.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << plan.group_sizes[i];
  }
  std::cout << "],\"panels\":{\"unique\":" << panel_bytes.size()
            << ",\"logical_tile_copies\":" << logical_panel_instances
            << ",\"physical_tile_copies\":" << physical_panel_instances
            << ",\"physical_replicated\":" << physical_replicated_panels
            << ",\"extra_physical_replicas\":" << extra_physical_replicas
            << ",\"estimated_panel_touch_bytes\":" << physical_panel_touch_bytes
            << ",\"estimated_logical_panel_touch_bytes\":"
            << logical_panel_touch_bytes << ",\"logical_by_tile\":[";
  for (size_t i = 0; i < logical_panels_by_tile.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << logical_panels_by_tile[i];
  }
  std::cout << "],\"physical_by_tile\":[";
  for (size_t i = 0; i < physical_panels_by_tile.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << physical_panels_by_tile[i];
  }
  std::cout << "]},\"cost_balance\":{\"max_tile_load\":" << max_tile_load
            << ",\"max_physical_tile_load\":" << max_physical_tile_load
            << ",\"max_worker_load\":" << max_worker_load
            << ",\"task_bytes_semantics\":\"summed per planned rectangle\""
            << ",\"owner_tiles\":[";
  for (size_t i = 0; i < tile_costs.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << "{\"tile\":" << i << ",\"threads\":" << plan.group_sizes[i]
              << ",\"load\":"
              << sgl_cost_score(tile_costs[i], total,
                                double(plan.group_sizes[i]) / nth)
              << ",\"cost\":";
    write_cost(tile_costs[i]);
    std::cout << '}';
  }
  std::cout << "],\"physical_tiles\":[";
  for (size_t i = 0; i < physical_tile_costs.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << "{\"tile\":" << i << ",\"threads\":" << plan.group_sizes[i]
              << ",\"load\":"
              << sgl_cost_score(physical_tile_costs[i], total,
                                double(plan.group_sizes[i]) / nth)
              << ",\"cost\":";
    write_cost(physical_tile_costs[i]);
    std::cout << '}';
  }
  std::cout << "],\"workers\":[";
  for (int worker = 0; worker < nth; ++worker) {
    if (worker != 0) {
      std::cout << ',';
    }
    std::cout << "{\"worker\":" << worker
              << ",\"tile\":" << tile_for_worker(worker, plan.group_sizes)
              << ",\"load\":"
              << sgl_cost_score(plan.worker_costs[worker], total, 1.0 / nth)
              << ",\"cost\":";
    write_cost(plan.worker_costs[worker]);
    std::cout << '}';
  }
  std::cout << "]},\"assignments\":[";
  for (size_t tid = 0; tid < observations.size(); ++tid) {
    if (tid != 0) {
      std::cout << ',';
    }
    const auto& observation = observations[tid];
    std::cout << "{\"thread\":" << tid << ",\"cpu\":" << observation.cpu
              << ",\"work_id\":" << observation.work_id
              << ",\"tile\":" << observation.tile
              << ",\"physical_tile\":" << observation.physical_tile
              << ",\"tasks\":[";
    for (size_t i = 0; i < observation.task_ids.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      std::cout << observation.task_ids[i];
    }
    std::cout << "]}";
  }
  std::cout << "],\"tasks\":[";
  for (size_t task_id = 0; task_id < plan.tasks.size(); ++task_id) {
    if (task_id != 0) {
      std::cout << ',';
    }
    const auto& task = plan.tasks[task_id];
    std::cout << "{\"id\":" << task_id << ",\"m\":[" << task.mb0 << ','
              << task.mb1 << "],\"n\":[" << task.nb0 << ',' << task.nb1
              << "],\"expert\":";
    if (task.expert < 0) {
      std::cout << "null";
    } else {
      std::cout << task.expert;
    }
    std::cout << ",\"tile_owner\":" << task.group
              << ",\"worker\":" << task_workers[task_id]
              << ",\"cpu\":" << task_cpus[task_id]
              << ",\"physical_tile\":" << task_physical_tiles[task_id]
              << ",\"cost\":";
    write_cost(task.cost);
    std::cout << '}';
  }
  std::cout << ']';
}

}  // namespace

// Build as a standalone executable linked to torch_cpu and c10.
// Positional arguments remain: configured threads, M blocks, N blocks.
int main(int argc, char** argv) {
  try {
    const Config config = parse_config(argc, argv);
    at::set_num_threads(config.threads);

    const bool dense = config.kind == "dense";
    const auto kind = dense ? SGLKernelKind::Dense : SGLKernelKind::MoE;
    const SGLSchedule mode = sgl_schedule(kind);
    const bool panel_mode = sgl_panel_schedule(mode);
    const int64_t rows =
        config.rows == -1 ? int64_t(32) * config.m_blocks : config.rows;
    const int64_t columns =
        config.columns == -1 ? int64_t(32) * config.n_blocks : config.columns;
    const auto expert_rows =
        dense ? std::vector<int64_t>{} : effective_expert_rows(config);
    std::vector<int32_t> expert_ids;
    std::vector<int32_t> offsets;
    if (!dense) {
      pack_expert_blocks(expert_rows, expert_ids, offsets);
    }

    const int64_t mb = dense ? div_up(rows, int64_t(32)) : expert_ids.size();
    const int64_t nb =
        dense
            ? div_up(columns, int64_t(32))
            : div_up(config.phase == "gate" ? columns : config.k, int64_t(32));
    const int64_t matrix_columns = config.phase == "gate" ? columns : config.k;
    const int64_t matrix_rows =
        dense ? rows
              : std::accumulate(expert_rows.begin(), expert_rows.end(),
                                int64_t(0));
    SGLTilePlan plan;
    int64_t plan_build_ns = 0;
    if (panel_mode) {
      const auto build_start = std::chrono::steady_clock::now();
      if (dense) {
        plan = make_sgl_dense_plan(mb, nb, rows, columns, config.k, 1);
      } else {
        auto plans = make_sgl_moe_plans(expert_ids.data(), offsets.data(), mb,
                                        columns, config.k);
        plan = config.phase == "gate" ? std::move(plans.gate)
                                      : std::move(plans.down);
      }
      if (config.timing) {
        plan_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - build_start)
                            .count();
      }
    }

    std::vector<int> coverage(static_cast<size_t>(mb) * nb);
    int active_workers = 0;
    int actual_team_size = 0;
    int64_t worker_finish_ns = 0;
    std::vector<Observation> observations;
    std::vector<int> task_workers;
    std::vector<int> task_cpus;
    std::vector<int> task_physical_tiles;
    if (panel_mode) {
      const int nth =
          std::accumulate(plan.group_sizes.begin(), plan.group_sizes.end(), 0);
      observations.resize(nth);
      std::vector<int> cursor_by_thread(nth);
      std::vector<int> team_sizes(nth);
      std::vector<uint8_t> overflow(nth);
      const auto& groups = sgl_tile_groups();
      for (int tid = 0; tid < nth; ++tid) {
        const int worker =
            mode == SGLSchedule::Spread ? groups.full_work_ids[tid] : tid;
        observations[tid].work_id = worker;
        observations[tid].tile = tile_for_worker(worker, plan.group_sizes);
        observations[tid].physical_tile =
            tile_for_worker(tid, plan.group_sizes);
        observations[tid].task_ids.reserve(plan.worker_offsets[worker + 1] -
                                           plan.worker_offsets[worker]);
      }
      const auto worker_start = std::chrono::steady_clock::now();
      sgl_run_tile_plan(plan, mode, [&](const auto& rectangles) {
        const int tid = omp_get_thread_num();
        const int worker = observations[tid].work_id;
        auto& observation = observations[tid];
        observation.cpu = sched_getcpu();
        observation.team_size = omp_get_num_threads();
        team_sizes[tid] = observation.team_size;
        int cursor = plan.worker_offsets[worker];
        rectangles([&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
          if (cursor >= plan.worker_offsets[worker + 1]) {
            overflow[tid] = 1;
            return;
          }
          const int task_id = plan.task_indices[cursor++];
          observation.task_ids.push_back(task_id);
          observation.rectangles.push_back({mb0, mb1, nb0, nb1});
        });
        cursor_by_thread[tid] = cursor;
      });
      if (config.timing) {
        worker_finish_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - worker_start)
                               .count();
      }
      actual_team_size = nth;
      std::vector<int> task_seen(plan.tasks.size());
      task_workers.assign(plan.tasks.size(), -1);
      task_cpus.assign(plan.tasks.size(), -1);
      task_physical_tiles.assign(plan.tasks.size(), -1);
      for (int tid = 0; tid < nth; ++tid) {
        auto& observation = observations[tid];
        const int worker = observation.work_id;
        TORCH_CHECK(observation.cpu >= 0 && team_sizes[tid] == nth,
                    "Observed an incomplete OpenMP team.");
        TORCH_CHECK(!overflow[tid] && cursor_by_thread[tid] ==
                                          plan.worker_offsets[worker + 1],
                    "Task executor differed from the prepared worker bins.");
        for (size_t i = 0; i < observation.task_ids.size(); ++i) {
          const int task_id = observation.task_ids[i];
          const auto& task = plan.tasks[task_id];
          const auto& rectangle = observation.rectangles[i];
          TORCH_CHECK(task.mb0 == rectangle[0] && task.mb1 == rectangle[1] &&
                          task.nb0 == rectangle[2] && task.nb1 == rectangle[3],
                      "Executed rectangle differs from its planned task.");
          ++task_seen[task_id];
          task_workers[task_id] = worker;
          task_cpus[task_id] = observation.cpu;
          task_physical_tiles[task_id] = observation.physical_tile;
          add_rectangle(coverage, mb, nb, rectangle);
        }
        active_workers += !observation.task_ids.empty();
      }
      TORCH_CHECK(std::all_of(task_seen.begin(), task_seen.end(),
                              [](int count) { return count == 1; }),
                  "Planned tasks must execute exactly once.");
    } else {
      if (mb > 0) {
        const int nth = adjust_num_threads(mb);
        observations.resize(nth);
        std::vector<std::array<int, 6>> assignments(nth);
        std::vector<int> team_sizes(nth);
        const bool remap = mode == SGLSchedule::RoundRobin;
        const auto* work_ids = remap ? sgl_work_ids(nth) : nullptr;
        const auto worker_start = std::chrono::steady_clock::now();
        parallel_2d(
            mb, nb,
            [&](int mb0, int mb1, int nb0, int nb1) {
              const int tid = omp_get_thread_num();
              const int work_id = work_ids ? (*work_ids)[tid] : tid;
              assignments[tid] = {sched_getcpu(), work_id, mb0, mb1, nb0, nb1};
              observations[tid].cpu = assignments[tid][0];
              observations[tid].work_id = work_id;
              observations[tid].team_size = omp_get_num_threads();
              team_sizes[tid] = observations[tid].team_size;
            },
            remap);
        if (config.timing) {
          worker_finish_ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - worker_start)
                  .count();
        }
        actual_team_size = nth;
        std::vector<int> seen(nth);
        for (int tid = 0; tid < nth; ++tid) {
          const auto& assignment = assignments[tid];
          TORCH_CHECK(team_sizes[tid] == nth,
                      "Actual OpenMP team is smaller than requested.");
          TORCH_CHECK(assignment[1] >= 0 && assignment[1] < nth,
                      "Work ID outside team.");
          ++seen[assignment[1]];
          const std::array<int64_t, 4> rectangle{assignment[2], assignment[3],
                                                 assignment[4], assignment[5]};
          observations[tid].rectangles.push_back(rectangle);
          add_rectangle(coverage, mb, nb, rectangle);
          active_workers +=
              rectangle[0] < rectangle[1] && rectangle[2] < rectangle[3];
        }
        TORCH_CHECK(std::all_of(seen.begin(), seen.end(),
                                [](int count) { return count == 1; }),
                    "Work mapping is not a permutation.");
      }
    }
    check_coverage(coverage, mb, nb, offsets, dense);

    std::cout << std::setprecision(12);
    std::cout << "{\"team_size\":" << actual_team_size
              << ",\"active_workers\":" << active_workers << ",\"kind\":\""
              << config.kind << "\",\"phase\":\""
              << (dense ? "dense" : config.phase) << "\",\"scheduler\":\""
              << schedule_name(mode)
              << "\",\"matrix\":{\"rows\":" << matrix_rows
              << ",\"columns\":" << matrix_columns << ",\"k\":" << config.k
              << ",\"mb\":" << mb << ",\"nb\":" << nb << ",\"intermediate_n\":";
    if (dense) {
      std::cout << "null";
    } else {
      std::cout << columns;
    }
    std::cout << "},\"actual_cpus\":[";
    for (size_t tid = 0; tid < observations.size(); ++tid) {
      if (tid != 0) {
        std::cout << ',';
      }
      std::cout << observations[tid].cpu;
    }
    std::cout << "],";
    if (panel_mode) {
      write_panel_report(plan, observations, task_workers, task_cpus,
                         task_physical_tiles);
    } else {
      std::cout << "\"assignments\":[";
      for (size_t tid = 0; tid < observations.size(); ++tid) {
        if (tid != 0) {
          std::cout << ',';
        }
        const auto& assignment = observations[tid];
        const auto& rectangle = assignment.rectangles.front();
        std::cout << "{\"thread\":" << tid << ",\"cpu\":" << assignment.cpu
                  << ",\"work_id\":" << assignment.work_id << ",\"m\":["
                  << rectangle[0] << ',' << rectangle[1] << "],\"n\":["
                  << rectangle[2] << ',' << rectangle[3] << "]}";
      }
      std::cout << "],\"tasks\":[";
      for (size_t tid = 0; tid < observations.size(); ++tid) {
        if (tid != 0) {
          std::cout << ',';
        }
        const auto& rectangle = observations[tid].rectangles.front();
        std::cout << "{\"id\":" << tid << ",\"m\":[" << rectangle[0] << ','
                  << rectangle[1] << "],\"n\":[" << rectangle[2] << ','
                  << rectangle[3]
                  << "],\"worker\":" << observations[tid].work_id
                  << ",\"cpu\":" << observations[tid].cpu
                  << ",\"tile_owner\":null}";
      }
      std::cout << ']';
    }
    if (!dense) {
      std::cout << ',';
      write_histogram(expert_rows);
    }
    if (config.timing) {
      std::cout << ",\"plan_build_ns\":" << plan_build_ns
                << ",\"worker_finish_ns\":" << worker_finish_ns;
    }
    std::cout << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

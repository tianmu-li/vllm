// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "common.h"
#include "dense_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sched.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int kLogicalWorkers = 96;
constexpr int64_t kMBlockSize = 32;
constexpr int kTimingIterations = 100;
#if defined(_OPENMP)
constexpr bool kOpenMP = true;
#else
constexpr bool kOpenMP = false;
#endif

using Scheduler = SGLDenseSchedule;

struct Config {
  int threads;
  int64_t mb;
  int64_t nb;
  int64_t actual_m;
  Scheduler scheduler = Scheduler::Legacy;
  bool timing;
};

struct Rectangle {
  int64_t m0;
  int64_t m1;
  int64_t n0;
  int64_t n1;
};

struct WorkerLog {
  int cpu = -1;
  int team_size = 0;
  bool entered = false;
  std::vector<Rectangle> rectangles;
};

struct Task {
  int id;
  int worker;
  int cpu;
  Rectangle rectangle;
  int64_t logical_owner;
  int64_t flat_task_id;
  bool overflow;
  bool empty;
};

const char* scheduler_name(Scheduler scheduler) {
  switch (scheduler) {
    case Scheduler::Legacy:
      return "legacy";
    case Scheduler::Logical96:
      return "logical96_dense";
    case Scheduler::Flat32:
      return "flat_32x32";
    case Scheduler::Flat128x64:
      return "flat_128x64";
    case Scheduler::Flat128x128:
      return "flat_128x128";
  }
  TORCH_CHECK(false, "Unknown dense scheduler.");
}

bool is_flat_schedule(Scheduler scheduler) {
  return scheduler == Scheduler::Flat32 || scheduler == Scheduler::Flat128x64 ||
         scheduler == Scheduler::Flat128x128;
}

int parse_int(const char* value, const char* name) {
  size_t end = 0;
  const auto parsed = std::stol(value, &end);
  TORCH_CHECK(end == std::string(value).size(), "Invalid ", name, ": ", value);
  TORCH_CHECK(parsed > 0 && parsed <= std::numeric_limits<int>::max(), name,
              " must be a positive integer.");
  return static_cast<int>(parsed);
}

int64_t parse_int64(const char* value, const char* name) {
  size_t end = 0;
  const auto parsed = std::stoll(value, &end);
  TORCH_CHECK(end == std::string(value).size(), "Invalid ", name, ": ", value);
  TORCH_CHECK(parsed >= 0, name, " must be nonnegative.");
  return parsed;
}

Config parse_config(int argc, char** argv) {
  TORCH_CHECK(argc >= 5,
              "Expected arguments: threads, M blocks, N blocks, scheduler");
  const std::string scheduler_arg(argv[4]);
  Scheduler scheduler = SGLDenseSchedule::Legacy;
  if (scheduler_arg == "legacy") {
    scheduler = Scheduler::Legacy;
  } else if (scheduler_arg == "logical96_dense") {
    scheduler = Scheduler::Logical96;
  } else if (scheduler_arg == "flat_32x32") {
    scheduler = Scheduler::Flat32;
  } else if (scheduler_arg == "flat_128x64") {
    scheduler = Scheduler::Flat128x64;
  } else if (scheduler_arg == "flat_128x128") {
    scheduler = Scheduler::Flat128x128;
  } else {
    TORCH_CHECK(false, "Unknown scheduler: ", scheduler_arg);
  }
  const auto threads = parse_int(argv[1], "threads");
  const auto mb = parse_int64(argv[2], "M blocks");
  const auto nb = parse_int64(argv[3], "N blocks");
  TORCH_CHECK(mb <= std::numeric_limits<int>::max() &&
                  nb <= std::numeric_limits<int>::max(),
              "Block counts must fit in int for the legacy scheduler.");
  Config config{threads, mb, nb, mb * kMBlockSize, scheduler, false};
  for (int i = 5; i < argc; ++i) {
    const std::string option(argv[i]);
    if (option == "--timing") {
      config.timing = true;
    } else if (option == "--m") {
      TORCH_CHECK(i + 1 < argc, "Missing value for --m.");
      config.actual_m = parse_int64(argv[++i], "actual M");
    } else {
      TORCH_CHECK(false, "Unknown option: ", option);
    }
  }
  const auto expected_mb =
      config.actual_m / kMBlockSize + (config.actual_m % kMBlockSize != 0);
  TORCH_CHECK(expected_mb == config.mb,
              "Actual M must satisfy M blocks = ceil(M / 32).");
  return config;
}

int current_worker_id() {
#if defined(_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

int current_team_size() {
#if defined(_OPENMP)
  return omp_get_num_threads();
#else
  return 1;
#endif
}

WorkerLog& enter_worker(std::vector<WorkerLog>& logs) {
  const int worker = current_worker_id();
  TORCH_CHECK(worker >= 0 && worker < static_cast<int>(logs.size()),
              "OpenMP worker exceeds the configured team size.");
  auto& log = logs[worker];
  TORCH_CHECK(!log.entered, "Worker entered the diagnostic more than once.");
  log.entered = true;
  log.cpu = sched_getcpu();
  log.team_size = current_team_size();
  return log;
}

void append_rectangle(WorkerLog& log, int64_t m0, int64_t m1, int64_t n0,
                      int64_t n1) {
  log.rectangles.push_back({m0, m1, n0, n1});
}

std::vector<Task> reconstruct_tasks(const std::vector<WorkerLog>& logs,
                                    int actual_team_size, bool logical96,
                                    const DenseLogical96Grid* grid,
                                    int primary_workers,
                                    const DenseFlatGrid* flat_grid) {
  std::vector<Task> tasks;
  for (int worker = 0; worker < actual_team_size; ++worker) {
    const auto& log = logs[worker];
    for (const auto& rectangle : log.rectangles) {
      const bool empty =
          rectangle.m0 >= rectangle.m1 || rectangle.n0 >= rectangle.n1;
      TORCH_CHECK(!logical96 || empty || grid != nullptr,
                  "Logical96 grid is missing for a nonempty matrix.");
      TORCH_CHECK(flat_grid == nullptr || empty || flat_grid->rows > 0,
                  "Flat grid is missing rows for a nonempty matrix.");
      const int64_t logical_owner =
          logical96 && !empty && grid != nullptr
              ? (rectangle.m0 / grid->bm) * grid->pn + rectangle.n0 / grid->bn
              : -1;
      const int64_t flat_task_id =
          flat_grid != nullptr && !empty
              ? (rectangle.n0 / flat_grid->bn) * flat_grid->rows +
                    rectangle.m0 / flat_grid->bm
              : -1;
      tasks.push_back({static_cast<int>(tasks.size()), worker, log.cpu,
                       rectangle, logical_owner, flat_task_id,
                       logical96 && logical_owner >= primary_workers, empty});
    }
  }
  return tasks;
}

void run_without_logging(int64_t mb, int64_t nb, int64_t actual_m) {
  parallel_sgl_dense(mb, nb, actual_m, [&](const auto& rectangles) {
    volatile int64_t token = 0;
    rectangles([&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
      token = m0 ^ m1 ^ n0 ^ n1;
    });
    const int64_t observed_token = token;
    (void)observed_token;
  });
}

void validate_coverage(
    const std::vector<Task>& tasks, int64_t mb, int64_t nb, bool logical96,
    const DenseLogical96Grid* grid, int primary_workers,
    std::vector<std::tuple<int64_t, int64_t, int64_t>>& overflow_claims,
    int& empty_rectangles) {
  if (mb == 0 || nb == 0) {
    TORCH_CHECK(tasks.empty(),
                "An empty matrix must not produce scheduled rectangles.");
    return;
  }
  TORCH_CHECK(static_cast<uint64_t>(mb) <= std::numeric_limits<size_t>::max() /
                                               static_cast<uint64_t>(nb),
              "Matrix block count exceeds addressable coverage storage.");
  TORCH_CHECK(!logical96 || (grid != nullptr && grid->pm > 0 && grid->pn > 0 &&
                             grid->bm > 0 && grid->bn > 0),
              "Logical96 grid dimensions must be positive.");
  std::vector<int> coverage(static_cast<size_t>(mb) * static_cast<size_t>(nb));
  for (const auto& task : tasks) {
    const auto& rect = task.rectangle;
    if (task.empty) {
      ++empty_rectangles;
      continue;
    }
    TORCH_CHECK(rect.m0 >= 0 && rect.m0 <= rect.m1 && rect.m1 <= mb &&
                    rect.n0 >= 0 && rect.n0 <= rect.n1 && rect.n1 <= nb,
                "Scheduled rectangle lies outside the matrix blocks.");
    if (logical96) {
      const int64_t m_tile = rect.m0 / grid->bm;
      const int64_t n_tile = rect.n0 / grid->bn;
      const int64_t owner = m_tile * grid->pn + n_tile;
      const int64_t expected_m0 = m_tile * grid->bm;
      const int64_t expected_m1 = std::min(expected_m0 + grid->bm, mb);
      const int64_t expected_n0 = n_tile * grid->bn;
      const int64_t expected_n1 = std::min(expected_n0 + grid->bn, nb);
      TORCH_CHECK(rect.m0 == expected_m0 && rect.m1 == expected_m1,
                  "Logical96 task must cover one complete M grid range.");
      TORCH_CHECK(owner == task.logical_owner &&
                      task.overflow == (owner >= primary_workers),
                  "Logical96 task has an invalid grid owner.");
      if (!task.overflow) {
        TORCH_CHECK(rect.n0 == expected_n0 && rect.n1 == expected_n1 &&
                        task.worker == owner,
                    "Primary task must match its owner's full grid rectangle.");
      } else {
        TORCH_CHECK(
            rect.n0 >= expected_n0 && rect.n1 == rect.n0 + 1 &&
                rect.n1 <= expected_n1,
            "Overflow tasks must cover one N block in their grid range.");
        overflow_claims.emplace_back(owner, rect.m0, rect.n0);
      }
      TORCH_CHECK(owner >= 0 && owner < grid->pm * grid->pn,
                  "Logical96 grid owner is outside the logical grid.");
      if (!task.overflow) {
        TORCH_CHECK(task.worker < kLogicalWorkers,
                    "Primary worker ID must be below 96.");
      }
    }
    for (int64_t m = rect.m0; m < rect.m1; ++m) {
      for (int64_t n = rect.n0; n < rect.n1; ++n) {
        ++coverage[static_cast<size_t>(m) * static_cast<size_t>(nb) +
                   static_cast<size_t>(n)];
      }
    }
  }
  std::sort(overflow_claims.begin(), overflow_claims.end());
  TORCH_CHECK(
      std::adjacent_find(overflow_claims.begin(), overflow_claims.end()) ==
          overflow_claims.end(),
      "An overflow N block was claimed more than once.");
  for (int64_t m = 0; m < mb; ++m) {
    for (int64_t n = 0; n < nb; ++n) {
      TORCH_CHECK(coverage[static_cast<size_t>(m) * static_cast<size_t>(nb) +
                           static_cast<size_t>(n)] == 1,
                  "Matrix block (", m, ", ", n,
                  ") does not have exact-once task coverage.");
    }
  }
}

void validate_flat_mapping(const std::vector<Task>& tasks,
                           const DenseFlatGrid& grid, int64_t mb, int64_t nb,
                           int actual_team_size,
                           const std::vector<int>& worker_task_counts) {
  const int64_t task_count = grid.tasks();
  TORCH_CHECK(
      task_count > 0 && task_count <= std::numeric_limits<size_t>::max(),
      "Flat task count must fit in addressable diagnostic storage.");
  TORCH_CHECK(static_cast<int64_t>(tasks.size()) == task_count,
              "Flat scheduler must emit each superblock exactly once.");

  const int64_t quotient = task_count / actual_team_size;
  const int64_t remainder = task_count % actual_team_size;
  std::vector<int> expected_worker(static_cast<size_t>(task_count), -1);
  int64_t begin = 0;
  for (int worker = 0; worker < actual_team_size; ++worker) {
    const int64_t count = quotient + (worker < remainder);
    TORCH_CHECK(worker_task_counts[worker] == count,
                "Flat worker load differs from its balanced static range.");
    for (int64_t task_id = begin; task_id < begin + count; ++task_id) {
      expected_worker[task_id] = worker;
    }
    begin += count;
  }
  TORCH_CHECK(begin == task_count,
              "Flat worker ranges do not cover all superblocks.");

  std::vector<int> seen(static_cast<size_t>(task_count));
  for (const auto& task : tasks) {
    const auto& rect = task.rectangle;
    TORCH_CHECK(
        !task.empty && task.flat_task_id >= 0 && task.flat_task_id < task_count,
        "Flat scheduler emitted an empty or invalid superblock.");
    const int64_t task_id = task.flat_task_id;
    const int64_t row = task_id % grid.rows;
    const int64_t column = task_id / grid.rows;
    const int64_t m0 = row * grid.bm;
    const int64_t n0 = column * grid.bn;
    TORCH_CHECK(rect.m0 == m0 && rect.m1 == std::min(mb, m0 + grid.bm) &&
                    rect.n0 == n0 && rect.n1 == std::min(nb, n0 + grid.bn),
                "Flat task does not match its clipped M-fast superblock.");
    TORCH_CHECK(expected_worker[task_id] == task.worker,
                "Flat task is assigned outside its worker's static range.");
    TORCH_CHECK(++seen[task_id] == 1,
                "Flat superblock was emitted more than once.");
  }
  TORCH_CHECK(std::all_of(seen.begin(), seen.end(),
                          [](int count) { return count == 1; }),
              "Flat superblock coverage is not exact-once.");
}

void write_rectangle(const Rectangle& rect) {
  std::cout << "{\"m\":[" << rect.m0 << ',' << rect.m1 << "],\"n\":[" << rect.n0
            << ',' << rect.n1 << "]}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_config(argc, argv);
    at::set_num_threads(config.threads);
    const auto selected_schedule = sgl_dense_schedule();
    TORCH_CHECK(selected_schedule == config.scheduler,
                "Scheduler argument differs from "
                "VLLM_CPU_SGL_SCHEDULER.");

    const bool empty_matrix = config.mb == 0 || config.nb == 0;
    const bool logical96 =
        kOpenMP && !empty_matrix && selected_schedule == Scheduler::Logical96;
    const bool flat_selected = is_flat_schedule(selected_schedule);
    std::optional<DenseFlatGrid> flat_grid;
    if (flat_selected) {
      flat_grid.emplace(config.mb, config.nb, selected_schedule);
    }
    const bool flat_eligible =
        flat_selected && !empty_matrix && flat_grid->eligible(config.actual_m);
    const bool flat_effective = flat_eligible;
    const bool legacy_fallback =
        !empty_matrix && flat_selected && !flat_eligible;
    const bool legacy_effective =
        !empty_matrix &&
        (selected_schedule == Scheduler::Legacy || legacy_fallback);
    const char* effective_schedule =
        empty_matrix                                            ? "empty"
        : !kOpenMP && selected_schedule == Scheduler::Logical96 ? "serial"
        : flat_selected && !flat_eligible                       ? "legacy"
                                          : scheduler_name(selected_schedule);
    const int legacy_required_team_size =
#if defined(_OPENMP)
        config.mb == 1 ? config.threads : std::max(1, (config.threads / 2) * 2);
#else
        1;
#endif

    std::vector<WorkerLog> logs(config.threads);
    std::optional<DenseLogical96Grid> grid;
    if (!empty_matrix) {
      grid.emplace(config.mb, config.nb);
    }
    parallel_sgl_dense(
        config.mb, config.nb, config.actual_m, [&](const auto& rectangles) {
          auto& log = enter_worker(logs);
          rectangles([&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
            append_rectangle(log, m0, m1, n0, n1);
          });
        });
    int actual_team_size = 0;
    if (empty_matrix) {
      for (const auto& log : logs) {
        TORCH_CHECK(!log.entered && log.team_size == 0,
                    "An empty matrix must not enter an OpenMP team.");
      }
    } else {
      TORCH_CHECK(!logs.empty() && logs[0].team_size > 0,
                  "The diagnostic did not observe an OpenMP team.");
      actual_team_size = logs[0].team_size;
      TORCH_CHECK(actual_team_size <= config.threads,
                  "Observed team exceeds the configured thread count.");
      for (int worker = 0; worker < actual_team_size; ++worker) {
        TORCH_CHECK(
            logs[worker].entered && logs[worker].team_size == actual_team_size,
            "Observed an incomplete or inconsistent OpenMP team.");
      }
      for (int worker = actual_team_size; worker < config.threads; ++worker) {
        TORCH_CHECK(!logs[worker].entered,
                    "Worker log exists outside the actual OpenMP team.");
      }
    }

    if (legacy_effective) {
      TORCH_CHECK(actual_team_size == legacy_required_team_size,
                  "Cannot check exact-once coverage for legacy scheduling with "
                  "a reduced OpenMP team: configured legacy team is ",
                  legacy_required_team_size, ", actual team is ",
                  actual_team_size,
                  ". Configure matching OMP thread limits or use an eligible "
                  "flat schedule for constrained-team diagnostics.");
    }

    const int primary_workers = std::min(actual_team_size, kLogicalWorkers);
    const auto tasks = reconstruct_tasks(
        logs, actual_team_size, logical96, grid ? &*grid : nullptr,
        primary_workers, flat_effective ? &*flat_grid : nullptr);
    std::vector<std::tuple<int64_t, int64_t, int64_t>> overflow_claims;
    int empty_rectangles = 0;
    validate_coverage(tasks, config.mb, config.nb, logical96,
                      grid ? &*grid : nullptr, primary_workers, overflow_claims,
                      empty_rectangles);
    if (logical96 && config.mb == 1 && (config.nb == 192 || config.nb == 288)) {
      TORCH_CHECK(grid->pm == 1 && grid->pn == 96 &&
                      grid->bn == div_up(config.nb, int64_t(96)),
                  "QKV decode must use a 1x96 logical96 grid.");
    }
    if (logical96 && config.mb == 1 && (config.nb == 32 || config.nb == 64)) {
      TORCH_CHECK(grid->pm == 2 && grid->pn == 48 &&
                      grid->bn == div_up(config.nb, int64_t(48)),
                  "Output projection must use a 2x48 logical96 grid.");
    }
    if (logical96 && config.mb == 3 && config.nb == 1 &&
        actual_team_size == 1) {
      TORCH_CHECK(grid->pm == 16 && grid->pn == 6 && !overflow_claims.empty() &&
                      std::get<0>(overflow_claims.front()) == 6,
                  "The first nonempty overflow owner should follow empty "
                  "logical96 segments.");
    }

    int useful_workers = 0;
    int useful_tasks = 0;
    std::vector<int> worker_task_counts(actual_team_size);
    for (const auto& task : tasks) {
      if (!task.empty) {
        if (worker_task_counts[task.worker] == 0) {
          ++useful_workers;
        }
        ++worker_task_counts[task.worker];
        ++useful_tasks;
      }
    }
    if (flat_effective) {
      validate_flat_mapping(tasks, *flat_grid, config.mb, config.nb,
                            actual_team_size, worker_task_counts);
    }
    if (logical96 && config.mb == 1 && (config.nb == 32 || config.nb == 64) &&
        actual_team_size >= 90) {
      TORCH_CHECK(overflow_claims.empty() && useful_tasks == 32,
                  "Output projection should have 32 primary tasks and no "
                  "overflow claims.");
    }

    double scheduler_mean_us = 0;
    if (config.timing && !empty_matrix) {
      run_without_logging(config.mb, config.nb, config.actual_m);
      const auto start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
        run_without_logging(config.mb, config.nb, config.actual_m);
      }
      const auto elapsed_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      scheduler_mean_us = double(elapsed_ns) / (kTimingIterations * 1000.0);
    }

    std::cout << "{\"selected_schedule\":\""
              << scheduler_name(selected_schedule)
              << "\",\"effective_schedule\":\"" << effective_schedule
              << "\",\"legacy_fallback\":"
              << (legacy_fallback ? "true" : "false")
              << ",\"legacy_schedule_active\":"
              << (legacy_effective ? "true" : "false")
              << ",\"legacy_expected_team_size\":";
    if (legacy_effective) {
      std::cout << legacy_required_team_size;
    } else {
      std::cout << "null";
    }
    std::cout << ",\"scheduler_eligibility\":{\"eligible\":";
    if (flat_selected) {
      std::cout << (flat_eligible ? "true" : "false");
    } else {
      std::cout << "null";
    }
    std::cout << ",\"reason\":\"";
    if (empty_matrix) {
      std::cout << "empty_matrix";
    } else if (!flat_selected) {
      std::cout << "not_applicable";
    } else if (config.actual_m < 128) {
      std::cout << "M_below_128";
    } else if (!flat_eligible) {
      std::cout << "superblock_tasks_below_192";
    } else {
      std::cout << "eligible";
    }
    std::cout << "\"},\"legacy_team_constraint\":";
    if (legacy_effective) {
      std::cout << "{\"expected_team_size\":" << legacy_required_team_size
                << ",\"actual_team_size\":" << actual_team_size
                << ",\"satisfied\":true}";
    } else {
      std::cout << "null";
    }
    std::cout << ",\"scheduler\":\"" << scheduler_name(selected_schedule)
              << "\",\"matrix\":{\"MB\":" << config.mb
              << ",\"NB\":" << config.nb << ",\"M\":" << config.actual_m
              << "},\"empty_matrix\":" << (empty_matrix ? "true" : "false")
              << ",\"actual_team_size\":" << actual_team_size
              << ",\"useful_workers\":" << useful_workers
              << ",\"useful_tasks\":" << useful_tasks
              << ",\"empty_rectangles\":" << empty_rectangles
              << ",\"overflow_claim_count\":" << overflow_claims.size()
              << ",\"overflow_claims_unique\":true,\"overflow_claims\":[";
    for (size_t i = 0; i < overflow_claims.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      std::cout << "{\"owner\":" << std::get<0>(overflow_claims[i])
                << ",\"m0\":" << std::get<1>(overflow_claims[i])
                << ",\"n0\":" << std::get<2>(overflow_claims[i]) << '}';
    }
    std::cout << "],\"coverage\":{\"exact_once\":true},";
    if (flat_selected) {
      std::cout << "\"flat_grid\":{\"bm\":" << flat_grid->bm
                << ",\"bn\":" << flat_grid->bn
                << ",\"rows\":" << flat_grid->rows
                << ",\"columns\":" << flat_grid->columns
                << ",\"candidate_task_count\":" << flat_grid->tasks()
                << ",\"task_order\":\"M_fast\""
                << ",\"effective\":" << (flat_effective ? "true" : "false")
                << "},";
    }
    if (logical96 && !empty_matrix) {
      const int64_t active_m = div_up(config.mb, grid->bm);
      const int64_t active_n = div_up(config.nb, grid->bn);
      std::cout << "\"logical96_grid\":{\"pm\":" << grid->pm
                << ",\"pn\":" << grid->pn << ",\"bm\":" << grid->bm
                << ",\"bn\":" << grid->bn << ",\"empty_grid_cells\":"
                << grid->pm * grid->pn - active_m * active_n << "},";
    }
    std::cout << "\"actual_cpus\":[";
    for (int worker = 0; worker < actual_team_size; ++worker) {
      if (worker != 0) {
        std::cout << ',';
      }
      std::cout << logs[worker].cpu;
    }
    std::cout << "],\"worker_load\":{\"minimum_tasks\":";
    if (worker_task_counts.empty()) {
      std::cout << "null,\"maximum_tasks\":null,\"idle_workers\":0},";
    } else {
      const auto [minimum, maximum] = std::minmax_element(
          worker_task_counts.begin(), worker_task_counts.end());
      const int idle_workers = static_cast<int>(
          std::count(worker_task_counts.begin(), worker_task_counts.end(), 0));
      std::cout << *minimum << ",\"maximum_tasks\":" << *maximum
                << ",\"idle_workers\":" << idle_workers << "},";
    }
    std::cout << "\"assignments\":[";
    for (int worker = 0; worker < actual_team_size; ++worker) {
      if (worker != 0) {
        std::cout << ',';
      }
      std::cout << "{\"worker\":" << worker << ",\"cpu\":" << logs[worker].cpu
                << ",\"worker_class\":\"";
      if (flat_effective) {
        std::cout << "flat";
      } else if (!logical96) {
        std::cout << "legacy";
      } else {
        std::cout << (worker < kLogicalWorkers ? "below_96" : "above_95");
      }
      std::cout << "\",\"useful_tasks\":" << worker_task_counts[worker]
                << ",\"tasks\":[";
      bool first_task = true;
      for (const auto& task : tasks) {
        if (task.worker != worker) {
          continue;
        }
        if (!first_task) {
          std::cout << ',';
        }
        first_task = false;
        std::cout << task.id;
      }
      std::cout << "]}";
    }
    std::cout << "],\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      const auto& task = tasks[i];
      std::cout << "{\"id\":" << task.id << ",\"worker\":" << task.worker
                << ",\"cpu\":" << task.cpu << ",\"role\":\"";
      if (flat_effective) {
        std::cout << "flat";
      } else if (!logical96) {
        std::cout << "legacy";
      } else {
        std::cout << (task.overflow ? "overflow" : "primary");
      }
      std::cout << "\",\"logical_owner\":";
      if (task.logical_owner < 0) {
        std::cout << "null";
      } else {
        std::cout << task.logical_owner;
      }
      std::cout << ",\"flat_task_id\":";
      if (task.flat_task_id < 0) {
        std::cout << "null";
      } else {
        std::cout << task.flat_task_id;
      }
      std::cout << ",\"empty\":" << (task.empty ? "true" : "false")
                << ",\"rectangle\":";
      write_rectangle(task.rectangle);
      std::cout << '}';
    }
    std::cout << ']';
    if (config.timing) {
      std::cout << ",\"scheduler_timing_iterations\":"
                << (empty_matrix ? 0 : kTimingIterations)
                << ",\"scheduler_mean_us\":";
      if (empty_matrix) {
        std::cout << "null";
      } else {
        std::cout << scheduler_mean_us;
      }
    }
    std::cout << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "common.h"
#include "dense_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
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
  int64_t actual_n;
  int64_t k;
  int64_t panels;
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

struct ProducerRange {
  int64_t begin;
  int64_t end;
  int worker;
  int cpu;
};

struct Task {
  int id;
  int worker;
  int cpu;
  Rectangle rectangle;
  int64_t logical_owner;
  int64_t rank_begin;
  int64_t rank_end;
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
    case Scheduler::FlatAuto:
      return "flat_auto";
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
  } else if (scheduler_arg == "flat_auto") {
    scheduler = Scheduler::FlatAuto;
  } else {
    TORCH_CHECK(false, "Unknown scheduler: ", scheduler_arg);
  }
  const auto threads = parse_int(argv[1], "threads");
  const auto mb = parse_int64(argv[2], "M blocks");
  const auto nb = parse_int64(argv[3], "N blocks");
  TORCH_CHECK(mb <= std::numeric_limits<int>::max() &&
                  nb <= std::numeric_limits<int>::max(),
              "Block counts must fit in int for the legacy scheduler.");
  Config config{threads,   mb,   nb, mb * kMBlockSize, nb * kMBlockSize, 64, 1,
                scheduler, false};
  for (int i = 5; i < argc; ++i) {
    const std::string option(argv[i]);
    if (option == "--timing") {
      config.timing = true;
    } else if (option == "--m") {
      TORCH_CHECK(i + 1 < argc, "Missing value for --m.");
      config.actual_m = parse_int64(argv[++i], "actual M");
    } else if (option == "--n") {
      TORCH_CHECK(i + 1 < argc, "Missing value for --n.");
      config.actual_n = parse_int64(argv[++i], "actual N");
    } else if (option == "--k") {
      TORCH_CHECK(i + 1 < argc, "Missing value for --k.");
      config.k = parse_int(argv[++i], "K");
    } else if (option == "--panels") {
      TORCH_CHECK(i + 1 < argc, "Missing value for --panels.");
      config.panels = parse_int(argv[++i], "panels");
    } else {
      TORCH_CHECK(false, "Unknown option: ", option);
    }
  }
  const auto expected_mb =
      config.actual_m / kMBlockSize + (config.actual_m % kMBlockSize != 0);
  TORCH_CHECK(expected_mb == config.mb,
              "Actual M must satisfy M blocks = ceil(M / 32).");
  const auto expected_nb =
      config.actual_n / kMBlockSize + (config.actual_n % kMBlockSize != 0);
  TORCH_CHECK(expected_nb == config.nb,
              "Actual N must satisfy N blocks = ceil(N / 32).");
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

std::vector<ProducerRange> record_producer_ranges(int64_t rows) {
  std::vector<ProducerRange> ranges;
  std::mutex ranges_mutex;
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    const ProducerRange range{begin, end, current_worker_id(), sched_getcpu()};
    std::lock_guard<std::mutex> lock(ranges_mutex);
    ranges.push_back(range);
  });
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.begin < rhs.begin;
  });
  int64_t covered_rows = 0;
  for (const auto& range : ranges) {
    TORCH_CHECK(range.begin == covered_rows && range.end > range.begin &&
                    range.end <= rows,
                "ATen producer ranges do not cover rows exactly once.");
    covered_rows = range.end;
  }
  TORCH_CHECK(covered_rows == rows,
              "ATen producer ranges do not cover all M rows.");
  return ranges;
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
      tasks.push_back({static_cast<int>(tasks.size()), worker, log.cpu,
                       rectangle, logical_owner, -1, -1,
                       logical96 && logical_owner >= primary_workers, empty});
    }
  }
  return tasks;
}

void run_without_logging(int64_t mb, int64_t nb, int64_t actual_m,
                         int64_t actual_n, int64_t k, int64_t panels) {
  parallel_sgl_dense(
      mb, nb, actual_m, actual_n, k, panels, [&](const auto& rectangles) {
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

void validate_flat_mapping(std::vector<Task>& tasks, const DenseFlatGrid& grid,
                           const DenseFlatTraversal& traversal, int64_t mb,
                           int64_t nb, int64_t k, int64_t panels,
                           int actual_team_size,
                           std::vector<int64_t>& worker_rank_begins,
                           std::vector<int64_t>& worker_rank_ends,
                           std::vector<int64_t>& worker_tile_counts,
                           std::vector<int64_t>& worker_tail_counts,
                           std::vector<int64_t>& worker_interleaved_counts) {
  TORCH_CHECK(
      mb > 0 && nb > 0 &&
          static_cast<uint64_t>(mb) <=
              std::numeric_limits<size_t>::max() / static_cast<uint64_t>(nb),
      "Flat compute-tile count exceeds addressable diagnostic storage.");
  const int64_t column_rows =
      traversal.column_rows(grid, mb, nb, k, panels, actual_team_size);
  const int64_t column_width =
      traversal.column_width(column_rows, mb, nb, k, actual_team_size);
  const int64_t column_chunks = div_up(nb, column_width);
  const int64_t interleaved_m_rows =
      traversal.grouped && column_rows == 0
          ? (mb % traversal.group_m != 0 ? mb % traversal.group_m
             : traversal.partial_m       ? 1
                                         : 0)
          : 0;
  const int64_t compute_tile_count = mb * nb;
  const int64_t rank_count = column_rows != 0
                                 ? (mb / column_rows) * column_chunks
                             : traversal.grouped ? compute_tile_count
                                                 : grid.tasks();
  const int64_t quotient = rank_count / actual_team_size;
  const int64_t remainder = rank_count % actual_team_size;
  std::vector<std::vector<size_t>> worker_tasks(actual_team_size);
  for (size_t index = 0; index < tasks.size(); ++index) {
    TORCH_CHECK(!tasks[index].empty && tasks[index].worker >= 0 &&
                    tasks[index].worker < actual_team_size,
                "Flat traversal emitted an empty rectangle or invalid worker.");
    worker_tasks[tasks[index].worker].push_back(index);
  }

  worker_rank_begins.resize(actual_team_size);
  worker_rank_ends.resize(actual_team_size);
  worker_tile_counts.resize(actual_team_size);
  worker_tail_counts.resize(actual_team_size);
  worker_interleaved_counts.resize(actual_team_size);
  auto expected_interleaved_count = [&](int64_t begin, int64_t end) {
    int64_t count = 0;
    const int64_t edge_begin = mb - interleaved_m_rows;
    for (int64_t n = 0; n < nb; ++n) {
      const int64_t n_begin = n * mb + edge_begin;
      const int64_t n_end = (n + 1) * mb;
      count +=
          std::max<int64_t>(0, std::min(end, n_end) - std::max(begin, n_begin));
    }
    return count;
  };
  int64_t prefix_begin = 0;
  for (int worker = 0; worker < actual_team_size; ++worker) {
    const int64_t count = quotient + (worker < remainder);
    const int64_t begin = column_rows != 0
                              ? rank_count * worker / actual_team_size
                              : prefix_begin;
    const int64_t end = column_rows != 0
                            ? rank_count * (worker + 1) / actual_team_size
                            : begin + count;
    worker_rank_begins[worker] = begin;
    worker_rank_ends[worker] = end;

    std::vector<Rectangle> expected_rectangles;
    if (column_rows != 0) {
      for (int64_t rank = begin; rank < end;) {
        const int64_t m0 = (rank / column_chunks) * column_rows;
        const int64_t column = rank % column_chunks;
        const int64_t count = std::min(end - rank, column_chunks - column);
        const int64_t n0 = column * column_width;
        const int64_t n1 = std::min(nb, (column + count) * column_width);
        expected_rectangles.push_back({m0, m0 + column_rows, n0, n1});
        rank += count;
      }
    } else if (traversal.grouped) {
      traversal.each(mb, nb, begin, end,
                     [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
                       expected_rectangles.push_back({m0, m1, n0, n1});
                     });
    } else {
      for (int64_t task_id = begin; task_id < end; ++task_id) {
        const int64_t row = task_id % grid.rows;
        const int64_t column = task_id / grid.rows;
        const int64_t m0 = row * grid.bm;
        const int64_t n0 = column * grid.bn;
        expected_rectangles.push_back(
            {m0, std::min(mb, m0 + grid.bm), n0, std::min(nb, n0 + grid.bn)});
      }
    }
    TORCH_CHECK(expected_rectangles.size() == worker_tasks[worker].size(),
                "Flat worker ", worker, " emitted ",
                worker_tasks[worker].size(), " rectangles; expected ",
                expected_rectangles.size(), " for rank interval [", begin, ",",
                end, "), column_rows=", column_rows,
                ", column_width=", column_width, ".");
    int64_t rank = begin;
    int64_t worker_tiles = 0;
    for (size_t i = 0; i < expected_rectangles.size(); ++i) {
      auto& task = tasks[worker_tasks[worker][i]];
      const auto& expected = expected_rectangles[i];
      const auto& actual = task.rectangle;
      TORCH_CHECK(actual.m0 == expected.m0 && actual.m1 == expected.m1 &&
                      actual.n0 == expected.n0 && actual.n1 == expected.n1,
                  "Flat rectangle differs from its ownership rank range.");
      const int64_t area = (actual.m1 - actual.m0) * (actual.n1 - actual.n0);
      worker_tiles += area;
      if (interleaved_m_rows != 0) {
        const int64_t overlap_rows = std::max<int64_t>(
            0, std::min(actual.m1, mb) -
                   std::max(actual.m0, mb - interleaved_m_rows));
        worker_interleaved_counts[worker] +=
            overlap_rows * (actual.n1 - actual.n0);
      }
      if (column_rows != 0) {
        task.rank_begin = rank;
        rank += div_up(actual.n1 - actual.n0, column_width);
        task.rank_end = rank;
      } else if (traversal.grouped) {
        task.rank_begin = rank;
        rank += area;
        task.rank_end = rank;
      } else {
        task.rank_begin = rank;
        task.rank_end = rank + 1;
        ++rank;
      }
      if (traversal.partial_m && actual.m0 <= mb - 1 && mb - 1 < actual.m1) {
        worker_tail_counts[worker] += actual.n1 - actual.n0;
      }
    }
    TORCH_CHECK(rank == end,
                "Flat rectangles do not cover their worker's rank interval.");
    TORCH_CHECK(worker_interleaved_counts[worker] ==
                    expected_interleaved_count(begin, end),
                "Interleaved M-edge compute tiles differ from rank interval.");
    worker_tile_counts[worker] = worker_tiles;
    prefix_begin = end;
  }
  TORCH_CHECK(prefix_begin == rank_count,
              "Flat worker rank intervals do not cover all ownership ranks.");
  if (column_rows != 0) {
    const auto [minimum, maximum] = std::minmax_element(
        worker_tile_counts.begin(), worker_tile_counts.end());
    TORCH_CHECK(*maximum - *minimum <= column_rows * column_width,
                "Column-ownership tile loads differ by more than one column "
                "chunk.");
  } else if (traversal.grouped) {
    const auto [minimum, maximum] = std::minmax_element(
        worker_tile_counts.begin(), worker_tile_counts.end());
    TORCH_CHECK(*maximum - *minimum <= 1,
                "Grouped compute-tile loads differ by more than one.");
  }
  if (traversal.partial_m) {
    const auto [minimum, maximum] = std::minmax_element(
        worker_tail_counts.begin(), worker_tail_counts.end());
    TORCH_CHECK(std::accumulate(worker_tail_counts.begin(),
                                worker_tail_counts.end(), int64_t(0)) == nb &&
                    *maximum - *minimum <= 1,
                "Grouped partial-M tail tiles are not evenly distributed.");
  }
  TORCH_CHECK(
      std::accumulate(worker_interleaved_counts.begin(),
                      worker_interleaved_counts.end(),
                      int64_t(0)) == interleaved_m_rows * nb,
      "Interleaved M-edge tile count does not match its configured rows.");
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
    const bool flat_auto = selected_schedule == Scheduler::FlatAuto;
    const bool flat_selected = is_flat_schedule(selected_schedule) || flat_auto;
    std::optional<DenseFlatGrid> flat_grid;
    if (is_flat_schedule(selected_schedule)) {
      flat_grid.emplace(config.mb, config.nb, selected_schedule);
    }
    const int legacy_required_team_size =
#if defined(_OPENMP)
        config.mb == 1 ? config.threads : std::max(1, (config.threads / 2) * 2);
#else
        1;
#endif

    std::vector<WorkerLog> logs(config.threads);
    std::optional<DenseLogical96Grid> grid;
    std::optional<DenseFlatTraversal> traversal;
    if (!empty_matrix) {
      grid.emplace(config.mb, config.nb);
    }
    parallel_sgl_dense(
        config.mb, config.nb, config.actual_m, config.actual_n, config.k,
        config.panels, [&](const auto& rectangles) {
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
    std::optional<Scheduler> auto_chosen_schedule;
    bool auto_grouped_policy = false;
    bool auto_wide_criteria_eligible = false;
    int64_t auto_wide_base_task_count = 0;
    if (flat_auto && !empty_matrix) {
      auto_grouped_policy = DenseFlatTraversal::should_group(
          config.actual_m, config.nb, config.panels);
      const DenseFlatGrid wide_candidate(config.mb, config.nb,
                                         Scheduler::Flat128x64);
      auto_wide_base_task_count = wide_candidate.tasks();
      auto_chosen_schedule = Scheduler::Flat32;
      if (!auto_grouped_policy) {
        auto_chosen_schedule =
            sgl_auto_schedule(config.mb, config.nb, config.actual_m,
                              config.actual_n, actual_team_size);
        auto_wide_criteria_eligible =
            *auto_chosen_schedule == Scheduler::Flat128x64;
      }
      flat_grid.emplace(config.mb, config.nb, *auto_chosen_schedule);
    }
    const bool flat_eligible =
        flat_selected && !empty_matrix && flat_grid->eligible(config.actual_m);
    const bool flat_effective = flat_eligible;
    const bool legacy_fallback =
        !empty_matrix && flat_selected && !flat_eligible;
    const bool legacy_effective =
        !empty_matrix &&
        (selected_schedule == Scheduler::Legacy || legacy_fallback);
    const char* effective_schedule = scheduler_name(selected_schedule);
    if (empty_matrix) {
      effective_schedule = "empty";
    } else if (!kOpenMP && selected_schedule == Scheduler::Logical96) {
      effective_schedule = "serial";
    } else if (flat_selected && !flat_eligible) {
      effective_schedule = "legacy";
    } else if (flat_auto) {
      effective_schedule = scheduler_name(*auto_chosen_schedule);
    }
    const auto producer_ranges = empty_matrix
                                     ? std::vector<ProducerRange>{}
                                     : record_producer_ranges(config.actual_m);
    if (flat_effective) {
      traversal.emplace(*flat_grid, config.mb, config.nb, config.actual_m,
                        config.k, config.panels, actual_team_size);
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
    auto tasks = reconstruct_tasks(logs, actual_team_size, logical96,
                                   grid ? &*grid : nullptr, primary_workers,
                                   flat_effective ? &*flat_grid : nullptr);
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
    std::vector<int64_t> worker_tile_counts(actual_team_size);
    for (const auto& task : tasks) {
      if (!task.empty) {
        if (worker_task_counts[task.worker] == 0) {
          ++useful_workers;
        }
        ++worker_task_counts[task.worker];
        ++useful_tasks;
        worker_tile_counts[task.worker] +=
            (task.rectangle.m1 - task.rectangle.m0) *
            (task.rectangle.n1 - task.rectangle.n0);
      }
    }
    std::vector<int64_t> worker_rank_begins;
    std::vector<int64_t> worker_rank_ends;
    std::vector<int64_t> worker_tail_counts;
    std::vector<int64_t> worker_interleaved_counts;
    if (flat_effective) {
      validate_flat_mapping(
          tasks, *flat_grid, *traversal, config.mb, config.nb, config.k,
          config.panels, actual_team_size, worker_rank_begins, worker_rank_ends,
          worker_tile_counts, worker_tail_counts, worker_interleaved_counts);
    }
    if (logical96 && config.mb == 1 && (config.nb == 32 || config.nb == 64) &&
        actual_team_size >= 90) {
      TORCH_CHECK(overflow_claims.empty() && useful_tasks == 32,
                  "Output projection should have 32 primary tasks and no "
                  "overflow claims.");
    }

    double scheduler_mean_us = 0;
    if (config.timing && !empty_matrix) {
      run_without_logging(config.mb, config.nb, config.actual_m,
                          config.actual_n, config.k, config.panels);
      const auto start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
        run_without_logging(config.mb, config.nb, config.actual_m,
                            config.actual_n, config.k, config.panels);
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
    std::cout << "\"},\"flat_auto_selection\":";
    if (!flat_auto) {
      std::cout << "null";
    } else if (empty_matrix) {
      std::cout << "{\"chosen_schedule\":null,\"team_size\":null"
                   ",\"team_size_source\":\"not_run_empty_matrix\""
                   ",\"grouped_policy\":null,\"wide_128x64_eligible\":false"
                   ",\"wide_128x64_base_task_count\":null"
                   ",\"base_geometry\":null,\"base_task_count\":null"
                   ",\"fallback_eligible\":false"
                   ",\"fallback_reason\":\"empty_matrix\"}";
    } else {
      std::cout
          << "{\"chosen_schedule\":\"" << scheduler_name(*auto_chosen_schedule)
          << "\",\"team_size\":" << actual_team_size
          << ",\"team_size_source\":\"observed OpenMP team\""
          << ",\"grouped_policy\":" << (auto_grouped_policy ? "true" : "false")
          << ",\"grouped_threshold\":0.25"
          << ",\"wide_128x64_eligible\":"
          << (auto_wide_criteria_eligible ? "true" : "false")
          << ",\"wide_128x64_selected\":"
          << (*auto_chosen_schedule == Scheduler::Flat128x64 ? "true" : "false")
          << ",\"wide_128x64_base_task_count\":" << auto_wide_base_task_count
          << ",\"base_geometry\":{\"bm\":" << flat_grid->bm
          << ",\"bn\":" << flat_grid->bn
          << "},\"base_task_count\":" << flat_grid->tasks()
          << ",\"fallback_eligible\":" << (flat_eligible ? "true" : "false")
          << ",\"fallback_reason\":\""
          << (flat_eligible           ? "eligible"
              : config.actual_m < 128 ? "M_below_128"
                                      : "superblock_tasks_below_192")
          << "\"}";
    }
    std::cout << ",\"legacy_team_constraint\":";
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
              << ",\"N\":" << config.actual_n << ",\"K\":" << config.k
              << ",\"panels\":" << config.panels
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
    if (flat_selected && flat_grid.has_value()) {
      std::cout << "\"flat_grid\":{\"bm\":" << flat_grid->bm
                << ",\"bn\":" << flat_grid->bn
                << ",\"rows\":" << flat_grid->rows
                << ",\"columns\":" << flat_grid->columns
                << ",\"candidate_task_count\":" << flat_grid->tasks()
                << ",\"effective\":" << (flat_effective ? "true" : "false")
                << "},";
      if (flat_effective) {
        const int64_t column_rows =
            traversal->column_rows(*flat_grid, config.mb, config.nb, config.k,
                                   config.panels, actual_team_size);
        const int64_t column_width = traversal->column_width(
            column_rows, config.mb, config.nb, config.k, actual_team_size);
        const int64_t column_chunks = div_up(config.nb, column_width);
        const int64_t interleaved_m_rows =
            traversal->grouped && column_rows == 0
                ? (config.mb % traversal->group_m != 0
                       ? config.mb % traversal->group_m
                   : traversal->partial_m ? 1
                                          : 0)
                : 0;
        const int64_t max_group_m = std::min(
            config.mb,
            std::max<int64_t>(flat_grid->bm,
                              get_cache_blocks<int8_t>(32 * config.k)));
        const int64_t max_group_n = std::min(
            config.nb, std::max<int64_t>(flat_grid->bn,
                                         get_cache_blocks<int8_t>(
                                             32 * config.k * config.panels)));
        const bool column_expansion_eligible =
            traversal->grouped && !traversal->partial_m &&
            traversal->group_m * traversal->group_n <= 16 &&
            config.mb % traversal->group_m == 0 && config.k > 2048;
        const int64_t column_joint_budget =
            column_expansion_eligible
                ? 2 * get_cache_blocks<int8_t>(32 * config.k)
                : 0;
        const int64_t column_peak_work =
            column_rows == 0
                ? 0
                : column_rows * div_up((config.mb / column_rows) * config.nb,
                                       int64_t(actual_team_size));
        const double target_aspect =
            traversal->grouped && !traversal->partial_m &&
                    traversal->group_m * traversal->group_n <= 16
                ? std::clamp(double(config.k) / 1024.0, 2.0, 4.0)
                : 2.0;
        std::cout << "\"flat_traversal\":{\"target_aspect\":" << target_aspect
                  << ",\"group_m\":" << traversal->group_m
                  << ",\"group_n\":" << traversal->group_n
                  << ",\"grouped\":" << (traversal->grouped ? "true" : "false")
                  << ",\"partial_m\":"
                  << (traversal->partial_m ? "true" : "false")
                  << ",\"partial_m_rows\":"
                  << (traversal->partial_m ? config.actual_m % kMBlockSize : 0)
                  << ",\"compute_tile_count\":" << config.mb * config.nb
                  << ",\"column_rows\":" << column_rows
                  << ",\"column_width\":" << column_width
                  << ",\"column_chunks_per_m_band\":";
        if (column_rows != 0) {
          std::cout << column_chunks;
        } else {
          std::cout << "null";
        }
        std::cout << ",\"interleaved_m_rows\":" << interleaved_m_rows
                  << ",\"column_peak_work\":";
        if (column_rows != 0) {
          std::cout << column_peak_work;
        } else {
          std::cout << "null";
        }
        std::cout << ",\"column_joint_budget\":";
        if (column_expansion_eligible) {
          std::cout << column_joint_budget;
        } else {
          std::cout << "null";
        }
        std::cout << ",\"max_group_m\":" << max_group_m
                  << ",\"max_group_n\":" << max_group_n
                  << ",\"ownership_unit\":\""
                  << (traversal->grouped && column_rows != 0 ? "M_group_columns"
                      : traversal->grouped                   ? "compute_tile"
                                           : "base_superblocks")
                  << "\",\"ownership_rank_count\":"
                  << (column_rows != 0
                          ? (config.mb / column_rows) * column_chunks
                      : traversal->grouped ? config.mb * config.nb
                                           : flat_grid->tasks())
                  << ",\"rank_partition\":\""
                  << (column_rows != 0 ? "proportional_floor"
                                       : "prefix_quotient_remainder")
                  << "\",\"task_order\":\""
                  << (interleaved_m_rows != 0
                          ? "N_fast_groups_with_interleaved_M_edge"
                      : traversal->grouped && column_rows != 0
                          ? (column_width > 1 ? "M_group_columns_N_chunks"
                                              : "M_group_columns_N_fast")
                      : traversal->grouped ? "N_fast_groups_then_microtiles"
                                           : "M_fast_base_superblocks")
                  << "\",\"tail_order\":\""
                  << (interleaved_m_rows != 0 ? "interleaved" : "regular")
                  << '"';
        if (traversal->partial_m) {
          const auto [minimum, maximum] = std::minmax_element(
              worker_tail_counts.begin(), worker_tail_counts.end());
          std::cout << ",\"tail_compute_tiles_per_worker\":[";
          for (size_t worker = 0; worker < worker_tail_counts.size();
               ++worker) {
            if (worker != 0) {
              std::cout << ',';
            }
            std::cout << worker_tail_counts[worker];
          }
          std::cout << "],\"tail_compute_tile_spread\":" << *maximum - *minimum;
        }
        std::cout << "},";
      }
    } else if (flat_selected) {
      std::cout << "\"flat_grid\":null,";
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
    std::cout << "],\"aten_parallel_for_ranges\":[";
    for (size_t i = 0; i < producer_ranges.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      const auto& range = producer_ranges[i];
      std::cout << "{\"rows\":[" << range.begin << ',' << range.end
                << "],\"cpu\":" << range.cpu
                << ",\"omp_worker\":" << range.worker << '}';
    }
    std::cout << "],\"worker_load\":{\"minimum_tasks\":";
    if (worker_task_counts.empty()) {
      std::cout << "null,\"maximum_tasks\":null,\"idle_workers\":0";
    } else {
      const auto [minimum, maximum] = std::minmax_element(
          worker_task_counts.begin(), worker_task_counts.end());
      const int idle_workers = static_cast<int>(
          std::count(worker_task_counts.begin(), worker_task_counts.end(), 0));
      std::cout << *minimum << ",\"maximum_tasks\":" << *maximum
                << ",\"idle_workers\":" << idle_workers;
    }
    if (worker_tile_counts.empty()) {
      std::cout << ",\"minimum_compute_tiles\":null"
                   ",\"maximum_compute_tiles\":null"
                   ",\"compute_tile_spread\":null},";
    } else {
      const auto [minimum, maximum] = std::minmax_element(
          worker_tile_counts.begin(), worker_tile_counts.end());
      const int idle_workers = static_cast<int>(
          std::count(worker_tile_counts.begin(), worker_tile_counts.end(), 0));
      std::cout << ",\"minimum_compute_tiles\":" << *minimum
                << ",\"maximum_compute_tiles\":" << *maximum
                << ",\"compute_tile_spread\":" << *maximum - *minimum
                << ",\"idle_compute_tile_workers\":" << idle_workers << "},";
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
                << ",\"compute_tiles\":" << worker_tile_counts[worker]
                << ",\"ownership_rank_range\":";
      if (flat_effective) {
        std::cout << '[' << worker_rank_begins[worker] << ','
                  << worker_rank_ends[worker] << ']';
      } else {
        std::cout << "null";
      }
      std::cout << ",\"tail_compute_tiles\":"
                << (flat_effective ? worker_tail_counts[worker] : 0);
      std::cout << ",\"tasks\":[";
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
      std::cout << ",\"ownership_rank_range\":";
      if (task.rank_begin < 0) {
        std::cout << "null";
      } else {
        std::cout << '[' << task.rank_begin << ',' << task.rank_end << ']';
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental shared-memory FP8 expert replication; no serving integration.
#include "gemm.h"
#include "moe.h"

#include <atomic>
#include <bit>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <sys/mman.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
using BF16 = at::BFloat16;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Guard pages keep each allocation's placement observable in numa_maps.
class LocalBuffer {
 public:
  LocalBuffer() = default;
  LocalBuffer(const LocalBuffer&) = delete;
  LocalBuffer& operator=(const LocalBuffer&) = delete;
  ~LocalBuffer() { release(); }
  void reserve(size_t bytes, int node, bool huge_pages = false) {
    if (bytes <= capacity_) return;
    release();
    constexpr size_t huge_page = 2 * 1024 * 1024;
    long system_page = sysconf(_SC_PAGESIZE);
    TORCH_CHECK(system_page > 0, "Cannot determine system page size");
    size_t page = system_page;
    bool use_huge_pages = huge_pages && bytes >= huge_page;
    size_t alignment = use_huge_pages ? huge_page : page;
    size_t overhead = 2 * page + (use_huge_pages ? huge_page : 0);
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() - overhead -
                             (alignment - 1),
                "NUMA allocation size overflow");
    capacity_ = ((bytes + alignment - 1) / alignment) * alignment;
    mapping_size_ = capacity_ + overhead;
    mapping_ = mmap(nullptr, mapping_size_, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TORCH_CHECK(mapping_ != MAP_FAILED, "NUMA allocation failed");
    uintptr_t address = reinterpret_cast<uintptr_t>(mapping_) + page;
    if (use_huge_pages)
      address = (address + huge_page - 1) & ~(uintptr_t(huge_page) - 1);
    data_ = reinterpret_cast<void*>(address);
    TORCH_CHECK(mprotect(data_, capacity_, PROT_READ | PROT_WRITE) == 0,
                "NUMA allocation mprotect failed");
    TORCH_CHECK(node >= 0 && node < sizeof(unsigned long) * 8,
                "NUMA allocation node is outside the policy mask");
    unsigned long node_mask = 1UL << node;
    TORCH_CHECK(mbind(data_, capacity_, MPOL_BIND, &node_mask,
                      sizeof(node_mask) * 8, 0) == 0,
                "NUMA allocation mbind failed: ", std::strerror(errno));
    if (use_huge_pages) {
      TORCH_CHECK(madvise(data_, capacity_, MADV_HUGEPAGE) == 0,
                  "NUMA allocation madvise failed: ", std::strerror(errno));
    } else {
      madvise(data_, capacity_, MADV_NOHUGEPAGE);
    }
    // The calling worker is already pinned to this node.
    std::memset(data_, 0, capacity_);
    verify(node);
  }
  void verify(int node) const {
    if (!data_) return;
    verify_pages(data_, capacity_, node);
  }
  static void verify_pages(const void* data, size_t capacity, int node) {
    std::ifstream maps("/proc/self/numa_maps");
    std::string line;
    size_t pages = 0;
    bool found = false;
    while (std::getline(maps, line)) {
      std::istringstream fields(line);
      uintptr_t address;
      fields >> std::hex >> address;
      if (address != reinterpret_cast<uintptr_t>(data)) continue;
      found = true;
      std::string field;
      while (fields >> field) {
        if (field.size() > 2 && field[0] == 'N' && std::isdigit(field[1])) {
          auto eq = field.find('=');
          int actual = std::stoi(field.substr(1, eq - 1));
          size_t count = std::stoull(field.substr(eq + 1));
          TORCH_CHECK(actual == node || count == 0,
                      "NUMA pages allocated on node ", actual, ", expected ",
                      node);
          pages += count;
        }
      }
      break;
    }
    TORCH_CHECK(found && pages == capacity / sysconf(_SC_PAGESIZE),
                "Cannot verify all NUMA allocation pages");
  }
  template <class T>
  T* ptr() {
    return reinterpret_cast<T*>(data_);
  }
  size_t pages() const { return capacity_ / sysconf(_SC_PAGESIZE); }

 private:
  void release() {
    if (mapping_ && mapping_ != MAP_FAILED) munmap(mapping_, mapping_size_);
    mapping_ = data_ = nullptr;
    capacity_ = mapping_size_ = 0;
  }
  void* mapping_ = nullptr;
  void* data_ = nullptr;
  size_t capacity_ = 0, mapping_size_ = 0;
};

class Team {
 public:
  Team(const std::vector<int64_t>& cpus, const std::atomic<int>& controller_cpu)
      : size_(cpus.size()),
        remaining_(cpus.size()),
        controller_cpu_(controller_cpu) {
    for (int tid = 0; tid < size_; ++tid) {
      threads_.emplace_back([this, tid, cpu = cpus[tid]] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        int affinity_error =
            pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        {
          std::lock_guard lock(mutex_);
          if (affinity_error) {
            error_ = std::make_exception_ptr(
                std::runtime_error("NUMA worker affinity failed"));
            failed_.store(true, std::memory_order_release);
          }
          ++ready_;
          done_.notify_one();
        }
        uint32_t previous = 0;
        for (;;) {
          auto deadline = Clock::now() + std::chrono::milliseconds(5);
          while (generation_.load(std::memory_order_acquire) == previous &&
                 Clock::now() < deadline) {
            if (cpu == controller_cpu_.load(std::memory_order_relaxed))
              std::this_thread::yield();
            else
              for (int spin = 0; spin < 64; ++spin) _mm_pause();
          }
          if (generation_.load(std::memory_order_acquire) == previous)
            generation_.wait(previous, std::memory_order_acquire);
          if (stop_.load(std::memory_order_acquire)) return;
          previous = generation_.load(std::memory_order_acquire);
          try {
            job_(tid);
          } catch (...) {
            fail(std::current_exception());
          }
          if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == size_)
            finished_.notify_one();
        }
      });
    }
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return ready_ == size_; });
    // The caller checks initialization before submitting any work.
  }
  ~Team() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& thread : threads_) thread.join();
  }
  void check() {
    if (!failed_.load(std::memory_order_acquire)) return;
    std::lock_guard lock(mutex_);
    if (error_) std::rethrow_exception(error_);
  }
  void fail(std::exception_ptr error) {
    std::lock_guard lock(mutex_);
    if (!error_) error_ = error;
    failed_.store(true, std::memory_order_release);
  }
  void submit(std::function<void(int)> job) {
    check();
    job_ = std::move(job);
    finished_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
  }
  void wait() {
    int finished = finished_.load(std::memory_order_acquire);
    while (finished != size_) {
      finished_.wait(finished, std::memory_order_acquire);
      finished = finished_.load(std::memory_order_acquire);
    }
    check();
  }
  void sync() {
    auto generation = barrier_generation_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      remaining_.store(size_, std::memory_order_relaxed);
      barrier_generation_.store(generation + 1, std::memory_order_release);
    } else {
      while (barrier_generation_.load(std::memory_order_acquire) == generation)
        _mm_pause();
    }
  }
  int size() const { return size_; }

 private:
  int size_, ready_ = 0;
  alignas(64) std::atomic<int> remaining_;
  alignas(64) std::atomic<uint64_t> barrier_generation_{0};
  alignas(64) std::atomic<int> finished_{0};
  alignas(64) std::atomic<uint32_t> generation_{0};
  std::atomic<bool> stop_{false}, failed_{false};
  const std::atomic<int>& controller_cpu_;
  std::mutex mutex_;
  std::condition_variable done_;
  std::function<void(int)> job_;
  std::exception_ptr error_;
  std::vector<std::thread> threads_;
};

struct ExpertTiles {
  std::vector<int32_t> block_offsets;
  std::vector<int64_t> work_prefix;
};

struct Executor : CpuMoeExecutor {
  Team& team;
  int tid;
  bool reduce;
  const ExpertTiles* tiles;
  const int32_t* slots;
  int64_t topk;
  bool grouped;
  const int32_t* input_offsets;
  Executor(Team& team, int tid, bool reduce = true,
           const ExpertTiles* tiles = nullptr, const int32_t* slots = nullptr,
           int64_t topk = 1, bool grouped = false,
           const int32_t* input_offsets = nullptr)
      : team(team),
        tid(tid),
        reduce(reduce),
        tiles(tiles),
        slots(slots),
        topk(topk),
        grouped(grouped),
        input_offsets(input_offsets) {}
  int thread_id() const override { return tid; }
  bool reduce_output() const override { return reduce; }
  const int32_t* routed_slots() const override { return slots; }
  int64_t input_topk() const override { return topk; }
  bool grouped_inputs() const override { return grouped; }
  const int32_t* grouped_input_offsets() const override {
    return input_offsets;
  }
  void for_1d(int64_t n, c10::function_ref<void(int64_t, int64_t)> f) override {
    int64_t begin = n * tid / team.size();
    int64_t end = n * (tid + 1) / team.size();
    try {
      f(begin, end);
    } catch (...) {
      team.fail(std::current_exception());
    }
    team.sync();
    team.check();
  }
  void for_2d(
      int m, int n,
      c10::function_ref<void(int64_t, int64_t, int64_t, int64_t)> f) override {
    if (tiles && m >= 1 && m <= team.size() &&
        m == static_cast<int>(tiles->work_prefix.size()) - 1 &&
        n >= team.size()) {
      int64_t group_work = tiles->work_prefix[1] - tiles->work_prefix[0];
      bool equal_single_mb_groups = group_work > 36;
      for (size_t group = 0;
           equal_single_mb_groups && group + 1 < tiles->work_prefix.size();
           ++group) {
        equal_single_mb_groups =
            tiles->work_prefix[group + 1] - tiles->work_prefix[group] ==
            group_work;
      }
      if (equal_single_mb_groups) {
        int64_t n0 = int64_t(n) * tid / team.size();
        int64_t n1 = int64_t(n) * (tid + 1) / team.size();
        try {
          f(0, m, n0, n1);
        } catch (...) {
          team.fail(std::current_exception());
        }
        team.sync();
        team.check();
        return;
      }
    }
    bool short_groups = grouped && tiles && tiles->work_prefix.size() - 1 <= 8;
    if (short_groups)
      for (size_t group = 0; group + 1 < tiles->block_offsets.size(); ++group)
        short_groups &=
            tiles->block_offsets[group + 1] - tiles->block_offsets[group] <= 5;
    if (tiles && n >= team.size() &&
        tiles->work_prefix.size() - 1 <= team.size() &&
        (m == static_cast<int>(tiles->work_prefix.size()) - 1 ||
         (short_groups && m == tiles->block_offsets.back()))) {
      int64_t total = tiles->work_prefix.back() * n;
      int64_t begin = (total * tid + team.size() - 1) / team.size();
      int64_t end = (total * (tid + 1) + team.size() - 1) / team.size();
      try {
        for (size_t group = 0; group + 1 < tiles->work_prefix.size(); ++group) {
          int64_t base = tiles->work_prefix[group] * n;
          int64_t limit = tiles->work_prefix[group + 1] * n;
          if (limit <= begin || base >= end) continue;
          int64_t weight =
              tiles->work_prefix[group + 1] - tiles->work_prefix[group];
          int64_t first =
              (std::max(int64_t(0), begin - base) + weight - 1) / weight;
          int64_t last = std::min(
              int64_t(n),
              (std::max(int64_t(0), end - base) + weight - 1) / weight);
          if (first < last)
            f(tiles->block_offsets[group], tiles->block_offsets[group + 1],
              first, last);
        }
      } catch (...) {
        team.fail(std::current_exception());
      }
      team.sync();
      team.check();
      return;
    }
    // Retain the original rectangle decomposition for larger expert queues.
    int tm = std::max(1, int(std::ceil(std::sqrt(float(m) / n * team.size()))));
    tm = std::min(tm, team.size());
    while (team.size() % tm) --tm;
    int tn = team.size() / tm;
    int bm = div_up(m, tm), bn = div_up(n, tn);
    int m0 = std::min(m, (tid / tn) * bm);
    int n0 = std::min(n, (tid % tn) * bn);
    try {
      f(m0, std::min(m, m0 + bm), n0, std::min(n, n0 + bn));
    } catch (...) {
      team.fail(std::current_exception());
    }
    team.sync();
    team.check();
  }
};

struct Scratch {
  LocalBuffer input, weights, ic0, ic1, ic2, output;
  LocalBuffer a_tmp, b_tmp, c_tmp;
};

struct Replica {
  int node;
  std::shared_ptr<Team> team;
  Scratch* scratch;
  LocalBuffer w1, w2, s1, s2;
  LocalBuffer shard_w1, shard_w2, shard_s1, shard_s2;
  int64_t shard_n = 0, shard_begin = -1;
  std::vector<int32_t> slots, sorted, experts, offsets;
  std::vector<int32_t> input_offsets;
  int64_t input_rows = 0;
  ExpertTiles tiles;
  double dispatch_ms = 0, expert_ms = 0, gather_ms = 0, total_ms = 0;
  int64_t padded = 0;
};

struct Pool {
  std::mutex mutex;
  std::atomic<int> controller_cpu{-1};
  std::vector<int64_t> cpus, nodes;
  std::array<LocalBuffer, 3> team_storage;
  std::array<Scratch, 3> scratch;
  std::array<Scratch, 3> large_scratch;
  std::array<std::shared_ptr<Team>, 3> teams;
};
std::mutex pool_registry_mutex;
std::weak_ptr<Pool> shared_pool;
std::shared_ptr<Pool> acquire_pool(const std::vector<int64_t>& cpus,
                                   const std::vector<int64_t>& nodes,
                                   int threads) {
  std::lock_guard lock(pool_registry_mutex);
  if (auto pool = shared_pool.lock()) {
    TORCH_CHECK(pool->cpus == cpus && pool->nodes == nodes,
                "Active NUMA layer contexts must use the same worker pool");
    return pool;
  }
  auto pool = std::make_shared<Pool>();
  pool->cpus = cpus;
  pool->nodes = nodes;
  pool->controller_cpu.store(sched_getcpu(), std::memory_order_relaxed);
  for (int i = 0; i < 3; ++i) {
    pool->team_storage[i].reserve(sizeof(Team), nodes[i]);
    auto* team = new (pool->team_storage[i].ptr<Team>())
        Team(std::vector<int64_t>(cpus.begin() + i * threads,
                                  cpus.begin() + (i + 1) * threads),
             pool->controller_cpu);
    pool->teams[i] = std::shared_ptr<Team>(team, [](Team* t) { t->~Team(); });
    pool->teams[i]->check();
  }
  shared_pool = pool;
  return pool;
}

void sum_rows(BF16* output, const BF16* const* rows, int64_t topk, int64_t K) {
  if (topk == 1) {
    copy_stub(output, rows[0], K);
    return;
  }
  using bVec = at::vec::Vectorized<BF16>;
  using fVec = at::vec::Vectorized<float>;
  int64_t channel = 0;
  for (; channel <= K - bVec::size(); channel += bVec::size()) {
    fVec sum0(0.f), sum1(0.f);
    for (int64_t slot = 0; slot < topk; ++slot) {
      auto [x0, x1] = load_float_vec2(rows[slot] + channel);
      sum0 += x0;
      sum1 += x1;
    }
    convert_from_float_ext<BF16>(sum0, sum1).store(output + channel);
  }
  for (; channel < K; ++channel) {
    float sum = 0.f;
    for (int64_t slot = 0; slot < topk; ++slot)
      sum += static_cast<float>(rows[slot][channel]);
    output[channel] = BF16(sum);
  }
}

struct CallStats {
  std::vector<int64_t> logical_counts, physical_counts;
  std::vector<int> active_experts;
  std::vector<int64_t> scratch_pages, scratch_addresses, shard_pages;
  std::array<int64_t, 3> intermediate_slices{};
  std::vector<double> node_ms;
  std::array<double, 5> phase_ms{};
  std::array<double, 3> predicted_load{};
  int64_t splits = 0, padded_rows = 0;
  int64_t M = 0, slots = 0, physical_slots = 0, dispatch_rows = 0;
  bool sharded = false, dynamic = false, calibrated = false;
};

struct Runtime {
  int64_t E, N, K, bn, bk;
  std::mutex mutex;
  std::shared_ptr<Pool> pool;
  std::array<Replica, 3> replicas;
  std::map<std::pair<int64_t, std::string>, std::vector<int>> history;
  std::vector<int64_t> cost_rows;
  std::array<std::vector<double>, 3> costs;
  std::array<double, 3> fixed_cost{};
  std::array<std::vector<double>, 3> single_costs, batch_costs;
  std::vector<int64_t> calibration_groups;
  std::array<std::array<std::vector<double>, 3>, 9> shared_costs;
  c10::Dict<std::string, at::Tensor> stats;
  CallStats last_call_stats;
  CallStats working_call_stats;
  std::vector<int> active_experts;
  std::vector<int64_t> expert_order;
  std::vector<double> expert_estimates;
  std::vector<const BF16*> contribution_rows;
  std::atomic<int> completed_nodes{0};
  std::atomic<bool> combine_ready{false}, combine_failed{false};
  Clock::time_point combine_start;
  double fused_combine_ms = 0;
  bool sharded = false;
  bool has_stats = false;
  std::vector<int64_t> weight_pages, weight_addresses;
  std::vector<std::vector<int32_t>> logical;
  std::array<std::vector<std::vector<int32_t>>, 3> groups;

  template <class T>
  at::Tensor reuse_stat(const std::string& name, const std::vector<T>& values) {
    auto found = stats.find(name);
    at::Tensor value;
    if (found != stats.end() &&
        found->value().numel() == static_cast<int64_t>(values.size()))
      value = found->value();
    else
      value =
          at::empty({static_cast<int64_t>(values.size())},
                    at::TensorOptions().dtype(
                        std::is_same_v<T, int64_t> ? at::kLong : at::kDouble));
    std::copy(values.begin(), values.end(), value.data_ptr<T>());
    return value;
  }

  void materialize_stats() {
    if (!has_stats) return;
    const auto& call = last_call_stats;
    auto phase_ms =
        std::vector<double>(call.phase_ms.begin(), call.phase_ms.end());
    auto intermediate_slices = std::vector<int64_t>(
        call.intermediate_slices.begin(), call.intermediate_slices.end());
    stats.insert("logical_counts",
                 reuse_stat<int64_t>("logical_counts", call.logical_counts));
    stats.insert("physical_counts",
                 reuse_stat<int64_t>("physical_counts", call.physical_counts)
                     .reshape({3, E}));
    stats.insert("split_groups",
                 reuse_stat<int64_t>("split_groups", {call.splits}).squeeze());
    stats.insert(
        "padded_rows",
        reuse_stat<int64_t>("padded_rows", {call.padded_rows}).squeeze());
    stats.insert("phase_ms", reuse_stat<double>("phase_ms", phase_ms));
    stats.insert("node_ms", reuse_stat<double>("node_ms", call.node_ms));
    stats.insert("weight_pages",
                 reuse_stat<int64_t>("weight_pages", weight_pages));
    stats.insert("weight_addresses",
                 reuse_stat<int64_t>("weight_addresses", weight_addresses)
                     .reshape({3, 4}));
    stats.insert("numa_nodes", reuse_stat<int64_t>("numa_nodes", pool->nodes));
    stats.insert("worker_cpus", reuse_stat<int64_t>("worker_cpus", pool->cpus));
    stats.insert("scratch_pages",
                 reuse_stat<int64_t>("scratch_pages", call.scratch_pages));
    stats.insert(
        "scratch_addresses",
        reuse_stat<int64_t>("scratch_addresses", call.scratch_addresses)
            .reshape({3, 9}));
    stats.insert("shard_pages",
                 reuse_stat<int64_t>("shard_pages", call.shard_pages));
    stats.insert(
        "intermediate_slices",
        reuse_stat<int64_t>("intermediate_slices", intermediate_slices));
    stats.insert("dispatch_bytes",
                 reuse_stat<int64_t>(
                     "dispatch_bytes",
                     {call.dispatch_rows * K * 2 + call.physical_slots * 4})
                     .squeeze());
    stats.insert("result_bytes",
                 reuse_stat<int64_t>(
                     "result_bytes",
                     {call.sharded ? call.M * K * 2 *
                                         (call.physical_slots /
                                          std::max<int64_t>(call.slots, 1))
                                   : call.slots * K * 2})
                     .squeeze());
    std::vector<int64_t> empty_rows;
    const auto& calibrated_rows = call.calibrated ? cost_rows : empty_rows;
    stats.insert_or_assign(
        "calibrated_rows",
        reuse_stat<int64_t>("calibrated_rows", calibrated_rows));
    if (call.dynamic) {
      auto predicted_load = std::vector<double>(call.predicted_load.begin(),
                                                call.predicted_load.end());
      stats.insert("predicted_node_ms",
                   reuse_stat<double>("predicted_node_ms", predicted_load));
    } else {
      (void)stats.erase("predicted_node_ms");
    }
    if (call.calibrated) {
      stats.insert(
          "fixed_node_ms",
          reuse_stat<double>("fixed_node_ms",
                             {fixed_cost[0], fixed_cost[1], fixed_cost[2]}));
      stats.insert(
          "calibration_group_counts",
          reuse_stat<int64_t>("calibration_group_counts", calibration_groups));
      for (auto entry : {std::pair{"cost_ms", &costs},
                         std::pair{"calibration_single_ms", &single_costs},
                         std::pair{"calibration_batch_ms", &batch_costs}}) {
        std::vector<double> all;
        for (auto& c : *entry.second) all.insert(all.end(), c.begin(), c.end());
        stats.insert(entry.first,
                     reuse_stat<double>(entry.first, all).reshape({3, -1}));
      }
      std::vector<double> shared;
      for (int experts = 1; experts <= std::min<int64_t>(8, E); ++experts)
        for (const auto& node : shared_costs[experts])
          shared.insert(shared.end(), node.begin(), node.end());
      stats.insert("calibration_shared_ms",
                   reuse_stat<double>("calibration_shared_ms", shared)
                       .reshape({std::min<int64_t>(8, E), 3, -1}));
    }
  }

  Runtime(const at::Tensor& w1, const at::Tensor& w2, const at::Tensor& s1,
          const at::Tensor& s2, const std::vector<int64_t>& cpus,
          const std::vector<int64_t>& nodes, int threads, int64_t block_n,
          int64_t block_k)
      : E(w1.size(0)),
        N(w1.size(1) / 2),
        K(w1.size(2)),
        bn(block_n),
        bk(block_k) {
    expert_order.reserve(E);
    active_experts.reserve(E);
    expert_estimates.resize(E, 0);
    for (auto* call : {&last_call_stats, &working_call_stats}) {
      call->logical_counts.resize(E);
      call->physical_counts.resize(3 * E);
      call->active_experts.reserve(E);
      call->node_ms.reserve(3);
      call->scratch_pages.reserve(3);
      call->scratch_addresses.reserve(27);
      call->shard_pages.reserve(3);
    }
    logical.resize(E);
    for (auto& g : groups) g.resize(E);
    pool = acquire_pool(cpus, nodes, threads);
    std::lock_guard pool_lock(pool->mutex);
    for (auto& team : pool->teams) team->check();
    for (int i = 0; i < 3; ++i) {
      auto& r = replicas[i];
      r.node = nodes[i];
      r.team = pool->teams[i];
      r.scratch = &pool->scratch[i];
    }
    for (auto& r : replicas) {
      r.team->submit([&, rp = &r](int tid) {
        if (tid != 0) return;
        auto copy = [&](LocalBuffer& b, const at::Tensor& t) {
          b.reserve(t.nbytes(), rp->node);
          std::memcpy(b.ptr<char>(), t.data_ptr(), t.nbytes());
          b.verify(rp->node);
        };
        copy(rp->w1, w1);
        copy(rp->w2, w2);
        copy(rp->s1, s1);
        copy(rp->s2, s2);
      });
    }
    wait_all();
    weight_pages.reserve(3);
    weight_addresses.reserve(12);
    for (auto& r : replicas) {
      weight_pages.push_back(r.w1.pages() + r.w2.pages() + r.s1.pages() +
                             r.s2.pages());
      for (auto* b : {&r.w1, &r.w2, &r.s1, &r.s2})
        weight_addresses.push_back(reinterpret_cast<int64_t>(b->ptr<char>()));
    }
  }

  void wait_all() {
    std::exception_ptr error;
    for (auto& r : replicas) {
      try {
        r.team->wait();
      } catch (...) {
        if (!error) error = std::current_exception();
      }
    }
    if (error) std::rethrow_exception(error);
  }

  void prepare(Replica& r, bool use_large_scratch, bool staged_inputs) {
    auto& scratch = *r.scratch;
    int64_t rows = r.slots.size(), threads = r.team->size();
    int64_t width = sharded ? r.shard_n : N;
    auto reserve = [&](LocalBuffer& b, int64_t bytes) {
      b.reserve(bytes, r.node, use_large_scratch);
    };
    if (sharded || staged_inputs)
      reserve(scratch.input, r.input_rows * K * sizeof(BF16));
    if (sharded) reserve(scratch.weights, rows * sizeof(float));
    reserve(scratch.ic0, rows * 2 * width * sizeof(BF16));
    reserve(scratch.ic1, rows * width * sizeof(BF16));
    reserve(scratch.ic2, rows * K * sizeof(BF16));
    if (sharded) reserve(scratch.output, rows * K * sizeof(BF16));
    reserve(scratch.a_tmp, threads * 32 * K * sizeof(BF16));
    reserve(scratch.b_tmp, threads * MAX_CACHE_BLOCK_SIZE * 32 *
                               std::max(K, N) * sizeof(BF16));
    reserve(scratch.c_tmp, threads * 2 * 32 * 32 * sizeof(float));
  }

  void configure_shards(const std::string& policy, int64_t layer) {
    for (auto& r : replicas) r.team->check();
    TORCH_CHECK(bn == 128 && bk == 128 && N % 256 == 0 && K % 128 == 0,
                "Sharding controls require 128x128 scales and aligned widths");
    std::array<int64_t, 3> widths{};
    if (policy == "unequal") {
      TORCH_CHECK(N == 512, "Unequal control requires intermediate size 512");
      widths.fill(128);
      widths[(layer % 3 + 3) % 3] = 256;
    } else {
      int excluded = policy == "tp2_01" ? 2 : policy == "tp2_02" ? 1 : 0;
      widths.fill(N / 2);
      widths[excluded] = 0;
    }
    int64_t begin = 0;
    for (int n = 0; n < 3; ++n) {
      auto& r = replicas[n];
      int64_t width = widths[n], offset = begin;
      begin += width;
      r.team->submit([&, rp = &r, width, offset](int tid) {
        if (tid || !width ||
            (rp->shard_n == width && rp->shard_begin == offset))
          return;
        auto& local = *rp;
        local.shard_w1.reserve(E * 2 * width * K, local.node);
        local.shard_w2.reserve(E * K * width, local.node);
        local.shard_s1.reserve(E * (2 * width / bn) * (K / bk) * 4, local.node);
        local.shard_s2.reserve(E * (K / bn) * (width / bk) * 4, local.node);
        for (int e = 0; e < E; ++e) {
          for (int half = 0; half < 2; ++half) {
            std::memcpy(
                local.shard_w1.ptr<char>() + (e * 2 * width + half * width) * K,
                local.w1.ptr<char>() + (e * 2 * N + half * N + offset) * K,
                width * K);
            std::memcpy(
                local.shard_s1.ptr<float>() +
                    (e * 2 * width / bn + half * width / bn) * (K / bk),
                local.s1.ptr<float>() +
                    (e * 2 * N / bn + (half * N + offset) / bn) * (K / bk),
                (width / bn) * (K / bk) * 4);
          }
          // Packed down weights are [K/32, N/2, 32, 2] per expert.
          for (int oc = 0; oc < K; oc += 32)
            std::memcpy(local.shard_w2.ptr<char>() + (e * K + oc) * width,
                        local.w2.ptr<char>() + (e * K + oc) * N + offset * 32,
                        width * 32);
          for (int oc = 0; oc < K / bn; ++oc)
            std::memcpy(
                local.shard_s2.ptr<float>() + (e * K / bn + oc) * (width / bk),
                local.s2.ptr<float>() + (e * K / bn + oc) * (N / bk) +
                    offset / bk,
                (width / bk) * 4);
        }
        local.shard_n = width;
        local.shard_begin = offset;
      });
    }
    wait_all();
    for (int n = 0; n < 3; ++n) replicas[n].shard_n = widths[n];
  }

  int64_t run(const at::Tensor& hidden, const at::Tensor& weights, int64_t topk,
              BF16* combined_output = nullptr) {
    const bool combine_output = combined_output && !sharded;
    completed_nodes.store(0, std::memory_order_relaxed);
    combine_ready.store(false, std::memory_order_relaxed);
    combine_failed.store(false, std::memory_order_relaxed);
    pool->controller_cpu.store(sched_getcpu(), std::memory_order_relaxed);
    for (auto& r : replicas) r.team->check();
    const bool use_large_scratch = !sharded && hidden.size(0) > 128;
    bool staged_inputs = use_large_scratch;
    if (!sharded && !use_large_scratch) {
      bool has_group = false;
      bool all_groups_large = true;
      for (const auto& r : replicas) {
        const auto& prefix = r.tiles.work_prefix;
        for (size_t group = 0; group + 1 < prefix.size(); ++group) {
          has_group = true;
          if (prefix[group + 1] - prefix[group] <= 32 + 4) {
            all_groups_large = false;
            break;
          }
        }
        if (!all_groups_large) break;
      }
      staged_inputs = has_group && all_groups_large;
    }
    auto& scratch_set = use_large_scratch ? pool->large_scratch : pool->scratch;
    for (size_t i = 0; i < replicas.size(); ++i)
      replicas[i].scratch = &scratch_set[i];
    if (!sharded) contribution_rows.resize(hidden.size(0) * topk);
    // Allocation is outside worker barriers, so an allocation failure cannot
    // strand other team members inside a stage.
    try {
      for (auto& r : replicas)
        if (!r.slots.empty()) prepare(r, use_large_scratch, staged_inputs);
    } catch (...) {
      auto error = std::current_exception();
      for (auto& r : replicas) r.team->fail(error);
      throw;
    }
    const bool direct = !sharded;
    if (direct)
      for (auto& r : replicas)
        for (size_t row = 0; row < r.slots.size(); ++row)
          contribution_rows[r.slots[row]] =
              r.scratch->ic2.ptr<BF16>() + row * K;
    for (auto& r : replicas) {
      r.dispatch_ms = r.expert_ms = r.gather_ms = r.total_ms = 0;
      r.team->submit([&, rp = &r](int tid) {
        try {
          auto& local = *rp;
          auto& scratch = *local.scratch;
          if (!local.slots.empty()) {
            Executor executor(*local.team, tid, sharded, &local.tiles,
                              direct ? local.slots.data() : nullptr, topk,
                              staged_inputs,
                              staged_inputs && !local.input_offsets.empty()
                                  ? local.input_offsets.data()
                                  : nullptr);
            int64_t width = sharded ? local.shard_n : N;
            auto start = Clock::now();
            if (staged_inputs || !direct)
              executor.for_1d(local.input_rows, [&](int64_t b, int64_t e) {
                for (int64_t row = b; row < e; ++row) {
                  int slot = local.slots[row];
                  copy_stub(scratch.input.ptr<BF16>() + row * K,
                            hidden.data_ptr<BF16>() + (slot / topk) * K, K);
                  if (!direct)
                    scratch.weights.ptr<float>()[row] =
                        weights.data_ptr<float>()[slot];
                }
              });
            if (tid == 0)
              local.dispatch_ms = direct && !staged_inputs ? 0 : ms(start);
            start = Clock::now();
            fused_experts_fp_kernel_impl<BF16, at::Float8_e4m3fn, float, false>(
                scratch.output.ptr<BF16>(), scratch.ic0.ptr<BF16>(),
                scratch.ic1.ptr<BF16>(), scratch.ic2.ptr<BF16>(),
                scratch.a_tmp.ptr<BF16>(), scratch.b_tmp.ptr<BF16>(),
                scratch.c_tmp.ptr<float>(),
                direct && !staged_inputs ? hidden.data_ptr<BF16>()
                                         : scratch.input.ptr<BF16>(),
                (sharded ? local.shard_w1 : local.w1).ptr<at::Float8_e4m3fn>(),
                (sharded ? local.shard_w2 : local.w2).ptr<at::Float8_e4m3fn>(),
                nullptr, nullptr,
                (sharded ? local.shard_s1 : local.s1).ptr<float>(),
                (sharded ? local.shard_s2 : local.s2).ptr<float>(), bn, bk,
                direct ? weights.data_ptr<float>()
                       : scratch.weights.ptr<float>(),
                local.sorted.data(), local.experts.data(), local.offsets.data(),
                local.slots.size(), width, K, E, 1, local.sorted.size(), 0, 0,
                CPUActMethod::silu_and_mul, false, &executor);
            if (tid == 0) local.expert_ms = ms(start);
            start = Clock::now();
            if (sharded) {
              executor.for_1d(local.slots.size(), [&](int64_t b, int64_t e) {
                for (int64_t row = b; row < e; ++row)
                  copy_stub(scratch.ic2.ptr<BF16>() + local.slots[row] * K,
                            scratch.output.ptr<BF16>() + row * K, K);
              });
              executor.for_1d(hidden.size(0), [&](int64_t b, int64_t e) {
                for (int64_t token = b; token < e; ++token)
                  sum_stub(scratch.input.ptr<BF16>() + token * K,
                           scratch.ic2.ptr<BF16>() + token * topk * K, topk, K);
              });
            }
            if (tid == 0) {
              local.gather_ms = sharded ? ms(start) : 0;
              local.total_ms =
                  local.dispatch_ms + local.expert_ms + local.gather_ms;
            }
          }
          if (combine_output) {
            int node = &local - replicas.data();
            int64_t begin = hidden.size(0) <= 32
                                ? (node == 0 ? 0 : hidden.size(0))
                                : hidden.size(0) * node / 3;
            int64_t end = hidden.size(0) <= 32
                              ? hidden.size(0)
                              : hidden.size(0) * (node + 1) / 3;
            if (tid != 0 && begin + tid >= end) return;
            // Publish only after every node has finished reading its inputs.
            if (tid == 0 &&
                completed_nodes.fetch_add(1, std::memory_order_acq_rel) == 2) {
              combine_start = Clock::now();
              combine_ready.store(true, std::memory_order_release);
            }
            while (!combine_ready.load(std::memory_order_acquire)) _mm_pause();
            if (combine_failed.load(std::memory_order_acquire)) return;
            for (int64_t token = begin + tid; token < end;
                 token += local.team->size())
              sum_rows(combined_output + token * K,
                       contribution_rows.data() + token * topk, topk, K);
          }
        } catch (...) {
          if (combine_output) {
            combine_failed.store(true, std::memory_order_release);
            combine_ready.store(true, std::memory_order_release);
          }
          throw;
        }
      });
    }
    wait_all();
    fused_combine_ms = combine_output ? ms(combine_start) : 0;
    int64_t dispatch_rows = 0;
    for (const auto& r : replicas)
      dispatch_rows += staged_inputs ? r.input_rows : r.slots.size();
    return dispatch_rows;
  }

  double interpolate_cost(const std::vector<double>& values,
                          int64_t rows) const {
    if (!rows) return 0;
    auto it = std::lower_bound(cost_rows.begin(), cost_rows.end(), rows);
    if (it == cost_rows.begin()) return values[0];
    if (it == cost_rows.end()) {
      size_t last = cost_rows.size() - 1;
      double slope = 0;
      if (last)
        slope = std::max(0., (values[last] - values[last - 1]) /
                                 (cost_rows[last] - cost_rows[last - 1]));
      return values.back() + slope * (rows - cost_rows.back());
    }
    auto j = it - cost_rows.begin();
    double f =
        double(rows - cost_rows[j - 1]) / (cost_rows[j] - cost_rows[j - 1]);
    return values[j - 1] * (1 - f) + values[j] * f;
  }

  double cost(int node, int64_t rows) const {
    return interpolate_cost(costs[node], rows);
  }

  double shared_cost(int node, int groups, int64_t rows) const {
    return rows && groups ? interpolate_cost(shared_costs[groups][node], rows)
                          : 0;
  }

  void group(const std::array<std::vector<std::vector<int32_t>>, 3>& groups,
             const std::vector<int>& active, int64_t topk = 1) {
    for (int node = 0; node < 3; ++node) {
      auto& r = replicas[node];
      r.slots.clear();
      r.sorted.clear();
      r.experts.clear();
      r.offsets.clear();
      r.offsets.push_back(0);
      r.padded = 0;
      r.input_offsets.clear();
      r.tiles.block_offsets.clear();
      r.tiles.block_offsets.push_back(0);
      r.tiles.work_prefix.clear();
      r.tiles.work_prefix.push_back(0);
      for (int expert : active) {
        const auto& slots = groups[node][expert];
        for (size_t begin = 0; begin < slots.size(); begin += 32) {
          int count = std::min<size_t>(32, slots.size() - begin);
          int offset = r.slots.size();
          r.experts.push_back(expert);
          for (int row = 0; row < count; ++row) {
            r.slots.push_back(slots[begin + row]);
            r.sorted.push_back(offset + row);
          }
          r.sorted.insert(r.sorted.end(), 32 - count, -1);
          r.padded += 32 - count;
          r.offsets.push_back(r.slots.size());
        }
        if (!slots.empty()) {
          r.tiles.block_offsets.push_back(r.experts.size());
          r.tiles.work_prefix.push_back(r.tiles.work_prefix.back() +
                                        slots.size() + 32);
        }
      }
      r.input_rows = r.slots.size();
      size_t group_count = r.tiles.block_offsets.size() - 1;
      if (sharded || topk == 1 || group_count < 2 || group_count > 8) continue;
      int64_t shared_rows = r.offsets[r.tiles.block_offsets[1]];
      if (shared_rows <= 4) continue;
      bool identical = true;
      for (size_t group = 1; identical && group < group_count; ++group) {
        int64_t begin = r.offsets[r.tiles.block_offsets[group]];
        int64_t end = r.offsets[r.tiles.block_offsets[group + 1]];
        identical = end - begin == shared_rows;
      }
      for (int64_t row = 0; identical && row < shared_rows; ++row) {
        int64_t first_slot = (r.slots[row] / topk) * topk;
        for (size_t group = 1; identical && group < group_count; ++group) {
          int64_t begin = r.offsets[r.tiles.block_offsets[group]];
          int64_t slot = r.slots[begin + row];
          identical = slot >= first_slot && slot < first_slot + topk;
        }
      }
      if (identical) {
        r.input_rows = shared_rows;
        r.input_offsets.resize(r.experts.size());
        for (size_t group = 0; group < group_count; ++group) {
          int first = r.tiles.block_offsets[group];
          int last = r.tiles.block_offsets[group + 1];
          for (int mb = first; mb < last; ++mb)
            r.input_offsets[mb] = r.offsets[mb] - r.offsets[first];
        }
      }
    }
  }

  void execute(const at::Tensor& hidden, const at::Tensor& ids,
               const at::Tensor& weights, at::Tensor& output,
               const std::string& policy, int64_t layer) {
    auto start = Clock::now();
    pool->controller_cpu.store(sched_getcpu(), std::memory_order_relaxed);
    sharded = policy == "unequal" || policy == "tp2_01" || policy == "tp2_02" ||
              policy == "tp2_12";
    TORCH_CHECK(policy == "static" || policy == "balanced" ||
                    policy == "split" || policy == "rows" || sharded,
                "Unknown NUMA policy");
    TORCH_CHECK(
        (policy != "balanced" && policy != "split") || !cost_rows.empty(),
        "Calibrate before dynamic scheduling");
    int64_t M = hidden.size(0), topk = ids.size(1), slots = M * topk;
    TORCH_CHECK(slots <= INT32_MAX, "Too many routed slots");
    auto& working = working_call_stats;
    for (int expert : active_experts) {
      logical[expert].clear();
      expert_estimates[expert] = 0;
      for (auto& node_groups : groups) node_groups[expert].clear();
    }
    active_experts.clear();
    for (int expert : working.active_experts) {
      working.logical_counts[expert] = 0;
      for (int node = 0; node < 3; ++node)
        working.physical_counts[node * E + expert] = 0;
    }
    working.active_experts.clear();
    const int32_t* route_ids = ids.data_ptr<int32_t>();
    for (int32_t slot = 0; slot < slots; ++slot) {
      int expert = route_ids[slot];
      TORCH_CHECK(expert >= 0 && expert < E, "Invalid logical expert ID");
      if (logical[expert].empty()) {
        active_experts.push_back(expert);
      }
      logical[expert].push_back(slot);
    }
    std::sort(active_experts.begin(), active_experts.end());
    working.active_experts.assign(active_experts.begin(), active_experts.end());

    auto& order = expert_order;
    auto& estimates = expert_estimates;
    order.assign(active_experts.begin(), active_experts.end());
    working.node_ms.clear();
    working.scratch_pages.clear();
    working.scratch_addresses.clear();
    working.shard_pages.clear();
    bool dynamic = policy == "balanced" || policy == "split";
    for (int e : active_experts) {
      working.logical_counts[e] = logical[e].size();
      if (dynamic)
        for (int n = 0; n < 3; ++n)
          estimates[e] = std::max(estimates[e], cost(n, logical[e].size()));
    }
    if (dynamic)
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (estimates[a] != estimates[b]) return estimates[a] > estimates[b];
        return a < b;
      });
    auto& previous = history[{layer, policy}];
    if (previous.empty()) previous.assign(E, -1);
    std::array<double, 3> load{};
    std::array<int, 3> group_counts{};
    int64_t splits = 0;
    if (sharded) configure_shards(policy, layer);
    for (int expert : order) {
      auto& tokens = logical[expert];
      if (tokens.empty()) continue;
      if (sharded) {
        for (int n = 0; n < 3; ++n)
          if (replicas[n].shard_n) groups[n][expert] = tokens;
        continue;
      }
      if (policy == "rows") {
        for (auto slot : tokens) {
          // Equal token rows, retaining all slots of a token on its replica.
          int node = std::min<int64_t>(2, (slot / topk) * 3 / M);
          groups[node][expert].push_back(slot);
        }
        continue;
      }
      int chosen = expert % 3;
      if (dynamic) {
        auto makespan = [&](int n, int64_t rows) {
          auto trial = load;
          trial[n] += cost(n, rows) + (group_counts[n] ? 0 : fixed_cost[n]);
          return *std::max_element(trial.begin(), trial.end());
        };
        chosen = 0;
        double best = makespan(0, tokens.size());
        for (int n = 1; n < 3; ++n) {
          double candidate = makespan(n, tokens.size());
          if (candidate < best) {
            chosen = n;
            best = candidate;
          }
        }
        int old = previous[expert];
        if (old >= 0 && makespan(old, tokens.size()) <= best * 1.01)
          chosen = old;
        groups[chosen][expert] = tokens;
        load[chosen] += cost(chosen, tokens.size()) +
                        (group_counts[chosen] ? 0 : fixed_cost[chosen]);
        ++group_counts[chosen];
        previous[expert] = chosen;
      } else {
        groups[chosen][expert] = tokens;
      }
    }
    bool shared_split = false;
    if (policy == "split" && M > 128 && order.size() >= 2 &&
        order.size() <= 8) {
      const auto& first = logical[order[0]];
      bool shared = first.size() > 32;
      for (size_t row = 0; shared && row < first.size(); ++row) {
        int token = first[row] / topk;
        if (row && token <= first[row - 1] / topk) shared = false;
        for (size_t expert = 1; shared && expert < order.size(); ++expert) {
          const auto& slots = logical[order[expert]];
          shared = slots.size() == first.size() && slots[row] / topk == token;
        }
      }
      if (shared) {
        std::array<double, 3> unsplit_load{};
        for (int n = 0; n < 3; ++n)
          unsplit_load[n] = shared_cost(n, group_counts[n], first.size());
        double best =
            *std::max_element(unsplit_load.begin(), unsplit_load.end());
        const double unsplit = best;
        std::array<int64_t, 3> best_rows{};
        std::array<double, 3> best_load{};
        // Move common token lists together so every local expert reuses the
        // same gathered inputs. Calibration charges all replicated weight
        // reads.
        for (int mask = 1; mask < 8; ++mask) {
          if (std::popcount(unsigned(mask)) < 2) continue;
          std::array<int64_t, 3> trial_rows{};
          std::array<double, 3> trial_load{};
          for (size_t row = 0; row < first.size(); row += 32) {
            int count = std::min<size_t>(32, first.size() - row);
            int target = -1;
            double finish = std::numeric_limits<double>::infinity();
            for (int n = 0; n < 3; ++n) {
              if (!(mask & (1 << n))) continue;
              auto candidate = trial_load;
              candidate[n] =
                  shared_cost(n, order.size(), trial_rows[n] + count);
              double completion =
                  *std::max_element(candidate.begin(), candidate.end());
              if (completion < finish) {
                target = n;
                finish = completion;
              }
            }
            trial_rows[target] += count;
            trial_load[target] =
                shared_cost(target, order.size(), trial_rows[target]);
          }
          int active = 0;
          for (auto rows : trial_rows) active += rows > 0;
          double candidate =
              *std::max_element(trial_load.begin(), trial_load.end());
          if (active > 1 && candidate <= unsplit * 0.95 && candidate < best) {
            best = candidate;
            best_rows = trial_rows;
            best_load = trial_load;
          }
        }
        if (best < unsplit) {
          std::array<int, 3> placement_order{0, 1, 2};
          std::sort(placement_order.begin(), placement_order.end(),
                    [&](int a, int b) {
                      auto a_tail = best_rows[a] % 32,
                           b_tail = best_rows[b] % 32;
                      return a_tail != b_tail ? a_tail < b_tail : a < b;
                    });
          for (int expert : order) {
            const auto& tokens = logical[expert];
            size_t begin = 0;
            for (int n : placement_order) {
              auto count = best_rows[n];
              groups[n][expert].assign(tokens.begin() + begin,
                                       tokens.begin() + begin + count);
              begin += count;
            }
            previous[expert] = -1;
          }
          load = best_load;
          splits = order.size();
          shared_split = true;
        }
      }
    }
    if (policy == "split" && !shared_split) {
      for (int expert : order) {
        const auto& tokens = logical[expert];
        if (tokens.size() <= 32) continue;
        int owner = previous[expert];
        auto remaining = load;
        auto remaining_counts = group_counts;
        remaining[owner] -= cost(owner, tokens.size());
        if (--remaining_counts[owner] == 0) remaining[owner] = 0;
        double unsplit = *std::max_element(load.begin(), load.end());
        if (*std::max_element(remaining.begin(), remaining.end()) >
            unsplit * 0.95)
          continue;
        double best = unsplit;
        std::array<int64_t, 3> best_rows{};
        std::array<double, 3> best_load{};
        // Evaluate against the complete queues, paying each replica's weight
        // reads and the fixed pipeline cost once per nonempty node.
        for (int mask = 1; mask < 8; ++mask) {
          if (std::popcount(unsigned(mask)) < 2) continue;
          std::array<int64_t, 3> trial_rows{};
          auto trial = remaining;
          for (size_t row = 0; row < tokens.size(); row += 32) {
            int count = std::min<size_t>(32, tokens.size() - row);
            int target = -1;
            double finish = std::numeric_limits<double>::infinity();
            for (int n = 0; n < 3; ++n) {
              if (!(mask & (1 << n))) continue;
              double candidate = remaining[n] + cost(n, trial_rows[n] + count) +
                                 (remaining_counts[n] ? 0 : fixed_cost[n]);
              if (candidate < finish) {
                finish = candidate;
                target = n;
              }
            }
            trial_rows[target] += count;
          }
          int active = 0;
          for (int n = 0; n < 3; ++n) {
            if (trial_rows[n])
              trial[n] += cost(n, trial_rows[n]) +
                          (remaining_counts[n] ? 0 : fixed_cost[n]);
            active += trial_rows[n] > 0;
          }
          double candidate = *std::max_element(trial.begin(), trial.end());
          if (active > 1 && candidate <= unsplit * 0.95 && candidate < best) {
            best = candidate;
            best_rows = trial_rows;
            best_load = trial;
          }
        }
        if (best == unsplit) continue;
        groups[owner][expert].clear();
        size_t begin = 0;
        std::array<int, 3> placement_order{0, 1, 2};
        std::sort(placement_order.begin(), placement_order.end(),
                  [&](int a, int b) {
                    auto remainder_a = best_rows[a] % 32;
                    auto remainder_b = best_rows[b] % 32;
                    if (remainder_a != remainder_b)
                      return remainder_a < remainder_b;
                    return a < b;
                  });
        for (int n : placement_order) {
          auto count = best_rows[n];
          groups[n][expert].assign(tokens.begin() + begin,
                                   tokens.begin() + begin + count);
          begin += count;
          remaining_counts[n] += count > 0;
        }
        load = best_load;
        group_counts = remaining_counts;
        ++splits;
        previous[expert] = -1;
      }
      if (splits == 0 && order.size() <= 8) {
        const double unsplit = *std::max_element(load.begin(), load.end());
        std::array<int, 3> representatives{-1, -1, -1};
        int representative_count = 0;
        for (int n = 0; n < 3; ++n) {
          if (load[n] <= unsplit * 0.95) continue;
          double largest_removal = -1;
          for (int expert : order) {
            const auto& tokens = logical[expert];
            if (tokens.size() <= 32 || previous[expert] != n) continue;
            double removal = cost(n, tokens.size()) +
                             (group_counts[n] == 1 ? fixed_cost[n] : 0);
            if (removal > largest_removal ||
                (removal == largest_removal &&
                 (representatives[n] < 0 || expert < representatives[n]))) {
              representatives[n] = expert;
              largest_removal = removal;
            }
          }
          if (representatives[n] >= 0) ++representative_count;
        }

        double best_joint_load = std::numeric_limits<double>::infinity();
        std::array<int, 2> best_experts{-1, -1};
        std::array<std::array<int64_t, 3>, 2> best_rows{};
        std::array<double, 3> best_joint_node_load{};
        for (int first_node = 0; first_node < 3; ++first_node) {
          if (representatives[first_node] < 0) continue;
          for (int second_node = first_node + 1; second_node < 3;
               ++second_node) {
            if (representatives[second_node] < 0) continue;
            std::array<int, 2> pair{representatives[first_node],
                                    representatives[second_node]};
            for (int orientation = 0; orientation < 2; ++orientation) {
              std::array<int, 2> experts =
                  orientation == 0 ? pair
                                   : std::array<int, 2>{pair[1], pair[0]};
              auto base_load = load;
              auto base_counts = group_counts;
              for (int expert : experts) {
                int owner = previous[expert];
                base_load[owner] -= cost(owner, logical[expert].size());
                if (--base_counts[owner] == 0) base_load[owner] = 0;
              }

              std::array<std::array<int64_t, 3>, 2> trial_rows{};
              std::array<double, 3> assigned_cost{};
              for (int group = 0; group < 2; ++group) {
                const auto& tokens = logical[experts[group]];
                for (size_t row = 0; row < tokens.size(); row += 32) {
                  int count = std::min<size_t>(32, tokens.size() - row);
                  int target = -1;
                  double finish = std::numeric_limits<double>::infinity();
                  for (int n = 0; n < 3; ++n) {
                    int64_t current_rows = trial_rows[group][n];
                    double candidate =
                        base_load[n] + assigned_cost[n] -
                        cost(n, current_rows) + cost(n, current_rows + count) +
                        (base_counts[n] == 0 ? fixed_cost[n] : 0);
                    if (candidate < finish) {
                      finish = candidate;
                      target = n;
                    }
                  }
                  int64_t old_rows = trial_rows[group][target];
                  trial_rows[group][target] += count;
                  assigned_cost[target] +=
                      cost(target, trial_rows[group][target]) -
                      cost(target, old_rows);
                }
              }

              bool both_split = true;
              for (int group = 0; group < 2; ++group) {
                int active_nodes = 0;
                for (int n = 0; n < 3; ++n)
                  active_nodes += trial_rows[group][n] > 0;
                both_split = both_split && active_nodes > 1;
              }
              if (!both_split) continue;
              auto trial_load = base_load;
              for (int n = 0; n < 3; ++n) {
                trial_load[n] += assigned_cost[n];
                if (base_counts[n] == 0 &&
                    (trial_rows[0][n] > 0 || trial_rows[1][n] > 0))
                  trial_load[n] += fixed_cost[n];
              }
              double candidate =
                  *std::max_element(trial_load.begin(), trial_load.end());
              if (candidate < best_joint_load) {
                best_joint_load = candidate;
                best_experts = experts;
                best_rows = trial_rows;
                best_joint_node_load = trial_load;
              }
            }
          }
        }
        if (representative_count >= 2 && best_joint_load <= unsplit * 0.95) {
          for (int group = 0; group < 2; ++group) {
            int expert = best_experts[group];
            const auto& tokens = logical[expert];
            int owner = previous[expert];
            groups[owner][expert].clear();
            --group_counts[owner];
            size_t begin = 0;
            std::array<int, 3> placement_order{0, 1, 2};
            std::sort(placement_order.begin(), placement_order.end(),
                      [&](int a, int b) {
                        auto a_tail = best_rows[group][a] % 32;
                        auto b_tail = best_rows[group][b] % 32;
                        return a_tail != b_tail ? a_tail < b_tail : a < b;
                      });
            for (int n : placement_order) {
              auto count = best_rows[group][n];
              groups[n][expert].assign(tokens.begin() + begin,
                                       tokens.begin() + begin + count);
              begin += count;
              group_counts[n] += count > 0;
            }
            previous[expert] = -1;
          }
          load = best_joint_node_load;
          splits += 2;
        }
      }
    }
    for (int n = 0; n < 3; ++n)
      for (int e : active_experts)
        working.physical_counts[n * E + e] = groups[n][e].size();
    group(groups, active_experts, topk);
    auto scheduling_ms = ms(start);
    auto dispatch_rows =
        run(hidden, weights, topk, sharded ? nullptr : output.data_ptr<BF16>());
    auto combine = Clock::now();
    // No floating-point scatter-add: restore the original top-k-slot order.
    // All input reads have finished, including when output aliases input.
    if (sharded) {
      for (int node = 0; node < 3; ++node) {
        auto& r = replicas[node];
        int64_t begin = M * node / 3, end = M * (node + 1) / 3;
        r.team->submit([&, rp = &r, begin, end](int tid) {
          Executor executor(*rp->team, tid);
          executor.for_1d(end - begin, [&](int64_t b, int64_t e) {
            for (int64_t token = begin + b; token < begin + e; ++token) {
              auto* destination = output.data_ptr<BF16>() + token * K;
              using bVec = at::vec::Vectorized<BF16>;
              using fVec = at::vec::Vectorized<float>;
              for (int64_t channel = 0; channel < K; channel += bVec::size()) {
                fVec sum0(0.f), sum1(0.f);
                for (auto& source : replicas) {
                  if (!source.shard_n) continue;
                  auto [x0, x1] = load_float_vec2(
                      source.scratch->input.ptr<BF16>() + token * K + channel);
                  sum0 += x0;
                  sum1 += x1;
                }
                convert_from_float_ext<BF16>(sum0, sum1)
                    .store(destination + channel);
              }
            }
          });
        });
      }
      wait_all();
    }
    double combine_ms = sharded ? ms(combine) : fused_combine_ms;
    double dispatch = 0, expert = 0, gather = 0;
    int64_t padded = 0;
    for (auto& r : replicas) {
      auto& scratch = *r.scratch;
      working.node_ms.push_back(r.total_ms);
      dispatch = std::max(dispatch, r.dispatch_ms);
      expert = std::max(expert, r.expert_ms);
      gather = std::max(gather, r.gather_ms);
      padded += r.padded;
      working.scratch_pages.push_back(
          scratch.input.pages() + scratch.weights.pages() +
          scratch.ic0.pages() + scratch.ic1.pages() + scratch.ic2.pages() +
          scratch.output.pages() + scratch.a_tmp.pages() +
          scratch.b_tmp.pages() + scratch.c_tmp.pages());
      for (auto* b : {&scratch.input, &scratch.weights, &scratch.ic0,
                      &scratch.ic1, &scratch.ic2, &scratch.output,
                      &scratch.a_tmp, &scratch.b_tmp, &scratch.c_tmp})
        working.scratch_addresses.push_back(
            reinterpret_cast<int64_t>(b->ptr<char>()));
      working.shard_pages.push_back(r.shard_w1.pages() + r.shard_w2.pages() +
                                    r.shard_s1.pages() + r.shard_s2.pages());
    }
    int64_t physical_slots = 0;
    for (const auto& r : replicas) physical_slots += r.slots.size();
    for (size_t node = 0; node < replicas.size(); ++node)
      working.intermediate_slices[node] = sharded ? replicas[node].shard_n : N;
    working.phase_ms = {scheduling_ms, dispatch, expert, gather + combine_ms,
                        ms(start)};
    working.predicted_load = load;
    working.splits = splits;
    working.padded_rows = padded;
    working.M = M;
    working.slots = slots;
    working.physical_slots = physical_slots;
    working.dispatch_rows = dispatch_rows;
    working.sharded = sharded;
    working.dynamic = dynamic;
    working.calibrated = !cost_rows.empty();
    std::swap(last_call_stats, working_call_stats);
    has_stats = true;
  }

  void calibrate(std::vector<int64_t> rows) {
    sharded = false;
    TORCH_CHECK(cost_rows.empty(),
                "Calibration is frozen; recreate to recalibrate");
    TORCH_CHECK(!rows.empty(), "Empty calibration row set");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    TORCH_CHECK(rows.front() > 0 && rows.back() <= INT32_MAX / 3,
                "Invalid calibration rows");
    std::vector<int> calibration_active;
    calibration_active.reserve(E);
    auto evict_calibration_weights = [&](int64_t experts) {
      for (auto& r : replicas) {
        r.team->submit([rp = &r, experts, this](int tid) {
          std::array<std::pair<LocalBuffer*, size_t>, 4> buffers{
              std::pair{&rp->w1, size_t(experts * 2 * N * K)},
              std::pair{&rp->w2, size_t(experts * N * K)},
              std::pair{&rp->s1, size_t(experts * div_up(2 * N, bn) *
                                        div_up(K, bk) * sizeof(float))},
              std::pair{&rp->s2, size_t(experts * div_up(K, bn) *
                                        div_up(N, bk) * sizeof(float))}};
          for (auto [buffer, bytes] : buffers)
            for (size_t offset = tid * 64; offset < bytes;
                 offset += rp->team->size() * 64)
              _mm_clflush(buffer->ptr<char>() + offset);
          _mm_mfence();
        });
      }
      wait_all();
    };
    for (auto count : rows) {
      int64_t batch =
          std::min<int64_t>({E, 64, std::max<int64_t>(1, 8192 / count)});
      TORCH_CHECK(count <= INT32_MAX / (3 * batch),
                  "Too many calibration slots");
      calibration_groups.push_back(batch);
      for (int64_t experts : {int64_t(1), batch}) {
        calibration_active.resize(experts);
        std::iota(calibration_active.begin(), calibration_active.end(), 0);
        auto hidden = at::zeros({3 * experts * count, K},
                                at::TensorOptions().dtype(at::kBFloat16));
        auto weights = at::ones({3 * experts * count},
                                at::TensorOptions().dtype(at::kFloat));
        std::array<std::vector<std::vector<int32_t>>, 3> groups;
        for (int n = 0; n < 3; ++n) {
          groups[n].resize(E);
          for (int e = 0; e < experts; ++e)
            for (int row = 0; row < count; ++row)
              groups[n][e].push_back((n * experts + e) * count + row);
        }
        group(groups, calibration_active);
        run(hidden, weights, 1);
        std::array<std::vector<double>, 3> samples;
        for (int repeat = 0; repeat < 5; ++repeat) {
          evict_calibration_weights(experts);
          run(hidden, weights, 1);
          for (int n = 0; n < 3; ++n)
            samples[n].push_back(replicas[n].total_ms);
        }
        for (int n = 0; n < 3; ++n) {
          std::sort(samples[n].begin(), samples[n].end());
          (experts == 1 ? single_costs[n] : batch_costs[n])
              .push_back(samples[n][2]);
        }
        if (batch == 1)
          for (int n = 0; n < 3; ++n) {
            batch_costs[n].push_back(single_costs[n].back());
          }
        if (batch == 1) break;
      }
    }
    for (int n = 0; n < 3; ++n) {
      std::vector<double> intercepts;
      for (size_t j = 0; j < rows.size(); ++j)
        if (rows[j] <= 32 && calibration_groups[j] > 1) {
          double marginal = (batch_costs[n][j] - single_costs[n][j]) /
                            (calibration_groups[j] - 1);
          intercepts.push_back(single_costs[n][j] - marginal);
        }
      if (!intercepts.empty()) {
        std::sort(intercepts.begin(), intercepts.end());
        fixed_cost[n] = std::clamp(
            intercepts[intercepts.size() / 2], 0.,
            *std::min_element(batch_costs[n].begin(), batch_costs[n].end()));
      }
      for (size_t j = 0; j < rows.size(); ++j)
        costs[n].push_back((batch_costs[n][j] - fixed_cost[n]) /
                           calibration_groups[j]);
    }
    for (int experts = 1; experts <= std::min<int64_t>(8, E); ++experts) {
      calibration_active.resize(experts);
      std::iota(calibration_active.begin(), calibration_active.end(), 0);
      for (auto count : rows) {
        TORCH_CHECK(count <= INT32_MAX / (3 * experts),
                    "Too many shared calibration slots");
        // Shared splits use the large-batch scratch and staging paths even
        // when a local slice contains only a few rows.
        int64_t input_rows = std::max<int64_t>(129, 3 * count);
        auto hidden = at::zeros({input_rows, K},
                                at::TensorOptions().dtype(at::kBFloat16));
        auto weights = at::ones({input_rows * experts},
                                at::TensorOptions().dtype(at::kFloat));
        std::array<std::vector<std::vector<int32_t>>, 3> shared_groups;
        for (int n = 0; n < 3; ++n) {
          shared_groups[n].resize(E);
          for (int e = 0; e < experts; ++e)
            for (int row = 0; row < count; ++row)
              shared_groups[n][e].push_back((n * count + row) * experts + e);
        }
        group(shared_groups, calibration_active, experts);
        run(hidden, weights, experts);
        std::array<std::vector<double>, 3> samples;
        for (int repeat = 0; repeat < 5; ++repeat) {
          evict_calibration_weights(experts);
          run(hidden, weights, experts);
          for (int n = 0; n < 3; ++n)
            samples[n].push_back(replicas[n].total_ms);
        }
        for (int n = 0; n < 3; ++n) {
          std::sort(samples[n].begin(), samples[n].end());
          shared_costs[experts][n].push_back(samples[n][2]);
        }
      }
    }
    cost_rows = std::move(rows);
    history.clear();
  }
};

std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<Runtime>> registry;
int64_t next_handle = 1;
struct LockedRuntime {
  std::shared_ptr<Runtime> runtime;
  std::unique_lock<std::mutex> lock;
  Runtime* operator->() const { return runtime.get(); }
};
LockedRuntime lookup(int64_t handle) {
  std::lock_guard lock(registry_mutex);
  auto it = registry.find(handle);
  TORCH_CHECK(it != registry.end(), "Invalid NUMA MoE runtime handle");
  return {it->second, std::unique_lock(it->second->mutex)};
}
void tensor_check(const at::Tensor& t, at::ScalarType dtype, int ndim) {
  TORCH_CHECK(
      t.device().is_cpu() && t.scalar_type() == dtype && t.dim() == ndim &&
          t.is_contiguous(),
      "NUMA MoE requires contiguous CPU tensors with matching dtype/rank");
}
}  // namespace

int64_t cpu_numa_moe_create(const at::Tensor& w1, const at::Tensor& w2,
                            const at::Tensor& s1, const at::Tensor& s2,
                            std::vector<int64_t> cpus,
                            std::vector<int64_t> nodes, int64_t threads,
                            int64_t bn, int64_t bk) {
  tensor_check(w1, at::kFloat8_e4m3fn, 3);
  tensor_check(w2, at::kFloat8_e4m3fn, 3);
  tensor_check(s1, at::kFloat, 3);
  tensor_check(s2, at::kFloat, 3);
  TORCH_CHECK(nodes.size() == 3 && threads > 0 && threads <= 32 &&
                  cpus.size() == 3 * threads,
              "Expected three physical CPU teams");
  TORCH_CHECK(
      std::set<int64_t>(nodes.begin(), nodes.end()).size() == 3 &&
          std::set<int64_t>(cpus.begin(), cpus.end()).size() == cpus.size(),
      "NUMA nodes and CPUs must be distinct");
  cpu_set_t allowed;
  TORCH_CHECK(sched_getaffinity(0, sizeof(allowed), &allowed) == 0,
              "Cannot read CPU affinity");
  std::set<std::pair<int, int>> cores;
  for (size_t i = 0; i < cpus.size(); ++i) {
    int64_t cpu = cpus[i], node = nodes[i / threads];
    TORCH_CHECK(cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed),
                "CPU outside allowed affinity");
    std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    TORCH_CHECK(
        access((root + "/node" + std::to_string(node)).c_str(), F_OK) == 0,
        "CPU does not belong to requested NUMA node");
    int package = -1, core = -1;
    std::ifstream(root + "/topology/physical_package_id") >> package;
    std::ifstream(root + "/topology/core_id") >> core;
    TORCH_CHECK(
        package >= 0 && core >= 0 && cores.emplace(package, core).second,
        "SMT siblings or unverifiable physical core topology");
  }
  int64_t E = w1.size(0), N = w1.size(1) / 2, K = w1.size(2);
  TORCH_CHECK(E > 0 && N > 0 && K > 0 && w1.size(1) == 2 * N && N % 32 == 0 &&
                  K % 32 == 0 && w2.sizes() == at::IntArrayRef({E, K, N}),
              "Invalid expert shapes");
  TORCH_CHECK(bn == 128 && bk == 128,
              "NUMA prototype requires 128x128 FP8 blocks");
  TORCH_CHECK(
      s1.sizes() == at::IntArrayRef({E, div_up(2 * N, bn), div_up(K, bk)}) &&
          s2.sizes() == at::IntArrayRef({E, div_up(K, bn), div_up(N, bk)}),
      "FP8 scales do not match expert shapes");
  auto runtime =
      std::make_shared<Runtime>(w1, w2, s1, s2, cpus, nodes, threads, bn, bk);
  std::lock_guard lock(registry_mutex);
  int64_t handle = next_handle++;
  registry.emplace(handle, std::move(runtime));
  return handle;
}

void cpu_numa_moe_execute(int64_t handle, const at::Tensor& hidden,
                          const at::Tensor& ids, const at::Tensor& weights,
                          at::Tensor& output, const std::string& policy,
                          int64_t layer) {
  auto runtime = lookup(handle);
  std::lock_guard pool_lock(runtime->pool->mutex);
  tensor_check(hidden, at::kBFloat16, 2);
  tensor_check(ids, at::kInt, 2);
  tensor_check(weights, at::kFloat, 2);
  tensor_check(output, at::kBFloat16, 2);
  TORCH_CHECK(hidden.size(1) == runtime->K &&
                  output.sizes() == hidden.sizes() &&
                  ids.size(0) == hidden.size(0) && ids.size(1) > 0 &&
                  weights.sizes() == ids.sizes(),
              "Invalid routed input shapes");
  runtime->execute(hidden, ids, weights, output, policy, layer);
}

c10::Dict<std::string, at::Tensor> cpu_numa_moe_stats(int64_t handle) {
  auto runtime = lookup(handle);
  runtime->materialize_stats();
  c10::Dict<std::string, at::Tensor> result;
  for (auto& entry : runtime->stats)
    result.insert(entry.key(), entry.value().clone());
  return result;
}

void cpu_numa_moe_calibrate(int64_t handle, std::vector<int64_t> rows) {
  auto runtime = lookup(handle);
  std::lock_guard pool_lock(runtime->pool->mutex);
  runtime->calibrate(std::move(rows));
}

void cpu_numa_moe_destroy(int64_t handle) {
  std::shared_ptr<Runtime> runtime;
  {
    std::lock_guard lock(registry_mutex);
    auto it = registry.find(handle);
    TORCH_CHECK(it != registry.end(), "Invalid NUMA MoE runtime handle");
    std::unique_lock runtime_lock(it->second->mutex);
    runtime = std::move(it->second);
    registry.erase(it);
  }
  runtime.reset();
}

void cpu_numa_moe_evict(int64_t handle) {
  auto runtime = lookup(handle);
  std::lock_guard pool_lock(runtime->pool->mutex);
  for (auto& r : runtime->replicas) r.team->check();
  for (auto& r : runtime->replicas) {
    r.team->submit([rp = &r](int tid) {
      for (auto* buffer : {&rp->w1, &rp->w2, &rp->s1, &rp->s2, &rp->shard_w1,
                           &rp->shard_w2, &rp->shard_s1, &rp->shard_s2}) {
        size_t bytes = buffer->pages() * sysconf(_SC_PAGESIZE);
        for (size_t offset = tid * 64; offset < bytes;
             offset += rp->team->size() * 64)
          _mm_clflush(buffer->ptr<char>() + offset);
      }
      _mm_mfence();
    });
  }
  runtime->wait_all();
}

void cpu_numa_moe_evict_local(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.is_contiguous(),
              "Expected a contiguous CPU tensor");
  auto* data = static_cast<const char*>(input.data_ptr());
  for (size_t offset = 0; offset < input.nbytes(); offset += 64)
    _mm_clflush(data + offset);
  _mm_mfence();
}

at::Tensor cpu_numa_moe_local_copy(const at::Tensor& input, int64_t node) {
  TORCH_CHECK(input.device().is_cpu() && input.is_contiguous() && input.numel(),
              "Expected a nonempty contiguous CPU tensor");
  auto storage = std::make_shared<LocalBuffer>();
  storage->reserve(input.nbytes(), node);
  std::memcpy(storage->ptr<char>(), input.data_ptr(), input.nbytes());
  storage->verify(node);
  return at::from_blob(
      storage->ptr<char>(), input.sizes(), [storage](void*) {},
      input.options());
}

int64_t cpu_numa_moe_verify_local(const at::Tensor& input, int64_t node) {
  TORCH_CHECK(input.device().is_cpu() && input.is_contiguous(),
              "Expected a contiguous CPU tensor from cpu_numa_moe_local_copy");
  size_t page = sysconf(_SC_PAGESIZE);
  size_t capacity = div_up(input.nbytes(), page) * page;
  LocalBuffer::verify_pages(input.data_ptr(), capacity, node);
  return capacity / page;
}

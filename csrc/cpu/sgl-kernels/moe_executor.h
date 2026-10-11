// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>
#include <functional>

// All members of a persistent team enter each stage. Stages include a barrier.
struct CpuMoeExecutor {
  virtual ~CpuMoeExecutor() = default;
  virtual int thread_id() const = 0;
  virtual bool reduce_output() const { return true; }
  virtual void for_2d(
      int m, int n,
      const std::function<void(int64_t, int64_t, int64_t, int64_t)>& f) = 0;
  virtual void for_1d(int64_t n,
                      const std::function<void(int64_t, int64_t)>& f) = 0;
  virtual const int32_t* routed_slots() const { return nullptr; }
  virtual int64_t input_topk() const { return 1; }
  virtual bool grouped_inputs() const { return false; }
  virtual const int32_t* grouped_input_offsets() const { return nullptr; }
};

# NUMA FP8 MoE tuning results, 2026-10-11

The retained experimental candidate combines bounded whole-expert column
scheduling, down-projection weight reuse within existing scratch, and shared
staged inputs for identical node-local token lists. The broad 2× complete-call
target remains unmet. Fixed ownership meets more cells than either dynamic
policy, and the dynamic policies do not meet the 5% gain over fixed ownership.
Serving promotion is rejected; numerical correctness and performance gates
remain unchanged.

## Complete-call measurements

The 64-cell matrix covers M={1,2,3,4,5,6,7,8,32,128,129,256,512,1024,2048,4096}
under uniform, reciprocal-Zipf, hot-eight, and changing-hot-eight routes. The
changing distribution contains two successive records per cell, giving 80
records per sequence. Each seed uses ten rotated paired rounds of ten full
ordered-sequence repeats, after initialization, frozen calibration, and one
warm sequence. A 10 ms untimed pause separates variants. Cell speedups compare
complete-call means with that cell's fastest valid 32-thread single-node
control. These are supplemental synthetic diagnostics, not primary timing.

| Policy | Replay ms, seeds 0 / 1 | M<8 passing cells, seeds 0 / 1 | M>128 passing cells, seeds 0 / 1 |
| --- | --- | --- | --- |
| Static | 763.24 / 764.09 | 26/28 / 24/28 | 20/24 / 21/24 |
| Balanced | 780.13 / 785.64 | 18/28 / 21/28 | 21/24 / 21/24 |
| Split | 782.75 / 788.56 | 19/28 / 21/28 | 20/24 / 20/24 |
| Rows | 1088.09 / 1090.01 | 10/28 / 8/28 | 12/24 / 12/24 |

The [seed-0 result](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_full64_seed0.json)
and [seed-1 result](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_full64_seed1.json)
record all cell means, median/p95 latency, throughput, node imbalance, splits,
padding, modeled traffic, numerical checks and source identity. Both completed
with 320 numerical comparisons and four alias comparisons; maximum absolute
error was 4.77e-7. Neither seed qualifies every requested small or large cell.

Fixed ownership's weakest small cell is M4 changing: 1.920× and 1.905×.
Its weakest large cell is M512 changing: 1.896× and 1.905×. Changing routes at
M256 and M1024 also remain below 2× in both seeds. Other near-threshold misses
vary by seed. Dynamic small hot-group placement remains sensitive: split's
seed-0 minimum is M7 hot-eight at 1.249×.

Balanced/static aggregate paired speedups are 0.97834× [0.97463, 0.98182] and
0.97257× [0.96972, 0.97560] for seeds 0 and 1.

Split/static aggregate paired speedups are 0.97512× [0.96724, 0.98104] and
0.96897× [0.96544, 0.97239] for seeds 0 and 1.

The earlier bounded-column/down-cache candidate measured static replay at
783.01/783.75 ms across its two seeds. The latest 763.24/764.09 ms is lower by
about 2.5% in these separate runs. The earlier active-experts shared-host run
measured 830.87 ms. These are descriptive comparisons: source/binary identities,
live single-node denominators, and host occupancy changed. They do not establish
paired causal gains. Two VMs observed earlier on CPUs 0–7 had exited by the
[current host snapshot](../../../build-numa-broad/campaign_20261011_broad/shared_staged_inputs_host_inventory.json).
No source update was selected by treating a per-cell union of policies as a pass.

## Retained mechanism and remaining bottlenecks

For staged queues of at most eight experts and at most five 32-row blocks per
expert, the executor partitions whole-expert column ranges by estimated work.
It preserves the original equal single-block branch and rectangular execution
for larger queues. The 160/161-row transition is an empirical scheduling bound,
not a model-shape requirement. Down projections retain more unpacked columns
only when both the existing per-thread scratch capacity and cache bound permit
it. FP8 arithmetic, scales and GEMM microkernels are unchanged.

When all 2–8 node-local expert groups contain the same ordered token list and
have more than four rows, they share one staged input copy. Matching uses the
canonical token's top-k-slot range. Separate input offsets leave intermediate
and weighted-result rows indexed by the original routed slots. Group metadata
resets every call; calibration and sharded controls disable input reuse. Gather
and final combine barriers preserve input/output aliasing and top-k reduction
order. For whole hot-eight groups at M4096, modeled BF16 staged writes decrease
from 128 MiB to 48 MiB; result rows and logical/physical route counts are unchanged.
This is a traffic model, not measured bandwidth.

The [static](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_static_live_profile.json),
[balanced](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_balanced_live_profile.json)
and [split](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_split_live_profile.json)
profiles preserve per-call phases, physical counts and each profile's own frozen
calibration tables. Static changing M256/M512/M1024 expert execution averaged
0.316/0.519/0.974 ms, with dispatch 0.035/0.080/0.129 ms and combine
0.037/0.072/0.125 ms. Node 1 completed those queues later than nodes 2 and 3.
Small changing M3/M4 expert execution averaged 0.046/0.053 ms, versus combine
about 0.008 ms. These native-only profiles locate costs; they do not independently
qualify speedups or establish their hardware cause.

In the split M512-changing profile, node 2 took 0.611 ms versus its own
0.388 ms prediction and was slowest in all 20 calls. Its four groups occupied
[5,5,16,16] row blocks. Rectangular M slices crossed three experts there, versus
at most two on the other nodes, adding weight-unpack sequences. Reordering
whole expert groups could reduce those crossings without changing placement or
within-group row blocks. This motivated the ordering trial below; the profile
does not establish the cause of the slowdown.

The subsequent [ordering trial](../../../build-numa-broad/campaign_20261011_broad/expert_order_crossings_full64_seed0.json)
searched at most 28 pair swaps and kept both projections' maximum crossing
counts at or below their initial values. It passed 11 NUMA tests and synthetic
correctness, but split
replay remained 782.64 ms versus the retained seed-0 run's 782.75 ms.
M512-changing split speedup was 1.841×; its absolute latency was about 1%
higher in these separate runs. This does not establish a significant regression
or an unpacking cause, and it does not justify retaining the search.
The trial used native source `5adf6e150ffcce0b0a4edbea92474259eef2162fcf10d3b158a18e447eddfabb`
and binary `715a33a658f4a6210df0442337c4cfdb5a91dcb1b25ed00b7a2af0a771386ec7`.
Its near-INT32_MAX block-count overflow was outside campaign sizes; the widened
candidate remains an unbuilt artifact. The trial code and its dedicated test
case were removed, and the retained source and binary were restored exactly.

An [extra M129 calibration point](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_boundary129_seed0.json)
also failed to improve aggregate replay. On the retained binary it measured
static/balanced/split at 763.26/803.02/799.49 ms. Split M129 hot-eight improved
to 1.954×, but remained below 2×; split's aggregate paired speedup over static
was 0.95465× [0.94901, 0.96118]. The original calibration table remains unchanged.
This single-seed diagnostic passed correctness and did not qualify broad speedups.

Two coordinator-placement diagnostics also failed to close the remaining gaps.
The [CPU31 trial](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_admin31_seed0.json)
used an extra physical core on node 0 while retaining all 96 expert workers on
nodes 1,2,3. Static replay increased to 877.64 ms and passed only 3/28 small and
16/24 large cells. This configuration used 97 physical cores in total; caller
placement and calibration changed, so it does not isolate CPU sharing alone.
The [CPU63 trial](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_controller63_seed0.json)
kept the original 96-core budget and shared the last node-1 worker CPU instead
of its first. Static replay measured 759.37 ms, with 23/28 small and 20/24 large
cells passing. Its weakest small cell was M3 changing at 1.813×; M256/M512/M1024
changing remained below 2×. Balanced and split replay measured 784.20/783.97 ms,
with paired speedups over static of 0.96834× [0.96478, 0.97198] and
0.96862× [0.96577, 0.97136]. Both trials passed synthetic correctness on the
unchanged retained binary. These separate single-seed measurements do not
establish a significant gain from CPU63. Caller CPU32 remains the configuration.

The separate [counter run](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_counters_seed0.json)
collected cycles, instructions, generic cache references and cache misses over
CPUs 32–127, with 100% event running time. Its enabled windows exclude creation,
calibration and correctness checks but include replay bookkeeping and any other
activity on those CPUs. Static measured 242.5 billion cycles, 134.4 billion
instructions, 6.0 billion cache references and 1.9 billion cache misses over its
instrumented window. These generic events do not establish a specific LLC cause
or kernel-only memory bandwidth. Instrumented timings are excluded from the
ordinary speedup table. Earlier IMC runs cover both sockets system-wide and
cannot establish per-NUMA or kernel-only bandwidth.

## Provenance and correctness

Source descends from c5a3a1389df9aae4be73bcac7b8eea6164af02c7. Current native
source SHA-256 values are:

- moe_numa.cpp: `2cfe4394afaccc04b463cdd1cad2a0633e8b47ee9ccded8f2fba157feb0e36c3`
- moe_fp8.cpp: `22ee5072e94b30e549a667021596e42099047dbe291e6aabaf0b0763c6ad4f3b`
- moe_executor.h: `e902ecd29f805aeb27e98e5ad296687724cb0307ce97cc519003e8da4ad1f761`
- vllm/_C.abi3.so: `a6e8c2617fb95ee33edb2850d256a9a790e7be5d70bcfe4e3d578bb2625df237`

[Build commands](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_build_commands.json)
record compilation/linking. Results retain imported paths/hashes, registered
schemas, checkpoint/corpus hashes, affinity, preload identities and calibration.
Qwen/Qwen3.6-35B-A3B-FP8 layer-12 weights use E256/top8/H2048/N512, BF16 activations
and block-FP8 scales. The weight-file hash is
`25b2726ac8d55dddc767c7e214e53072e22202cbe35f59441d9bb938122cc0e6`.

There is one batch/runtime with three local weight copies. Workers use CPUs
32–63, 64–95 and 96–127 on nodes 1,2,3, with caller CPU32 and no SMT compute
threads. Native host OMP/MKL counts are 1, PASSIVE/blocktime 0; single-node
controls use 32 threads, ACTIVE/blocktime 5 and TPAUSE 0. Commands use
`uv run --no-sync .venv/bin/python -P`, system tcmalloc and repository libiomp5.
The [placement check](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_live_placement_verified.json)
verified distinct replica storage and all 54 live native weight/scratch mappings
on their intended nodes. Large scratch had 456 MiB actual huge-page backing;
weights and small scratch used base pages.

The nearest NUMA selection passed 11 tests on nodes 1,2,3. The shared quantized
selection passed 85 tests on node 5. Added cases exercise multi-block shared
inputs with a tail, equal-size groups containing different tokens, and aliased
output across both cases. Changed native and test files passed pre-commit.
[Captured-output validation](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_primary_numerical.json)
completed 1,920 comparisons per policy over 40 layers and 48 ordered steps:
120 prefill, 840 mixed and 960 decode records per policy. Static, balanced and
split matched captured BF16 outputs exactly. Rows passed the unchanged
atol=rtol=0.01 tolerance, with maximum absolute error 0.03125. Inputs, native
source and extension stayed unchanged through validation and runtime cleanup.
This establishes captured-output parity, not an independent oracle or timing.
The frozen corpus hash is
`4160faf51bd653f43b855f951c93b6259162a4971caff25ea64c460c26c649bb`;
full captured-layer weights hash is
`bc9a62f6a3bcdf5b6c8cbbb79062d79ab1e0ee2cd6dbbd12a23dbe596d4fbea5`.

The synthetic timing campaign omits 96-thread, native TP2 and unequal-shard
controls. It does not establish the primary replay promotion gate or full-model
serving throughput. The current binary has no new primary timing qualification.

## Preserved rejected experiments

Earlier artifacts and committed reports retain the detailed trial evidence.
Follow-up active waits, expert-aligned gangs, padded input rows, a 16-worker
sparse path and caller-last dispatch did not meet the broad target. Unrestricted
weighted column scheduling regressed larger queues; expanding its down cache
alone did not resolve that regression. The five-block bound retains larger
rectangles. Delegated and combined wake protocols were also rejected; their
cold-call and lifecycle evidence remains in the previous report revision and
fresh campaign artifacts. No historical artifacts were overwritten.

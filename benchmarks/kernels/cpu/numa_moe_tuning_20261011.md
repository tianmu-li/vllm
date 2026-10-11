# NUMA FP8 MoE tuning results, 2026-10-11

The retained experimental candidate combines bounded whole-expert column
scheduling, down-projection weight reuse within existing scratch, and shared
staged inputs for identical node-local token lists. Synchronous stage callbacks
use non-owning references to eliminate per-worker closure allocations.
The broad 2× complete-call
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

### Queue-cardinality and placement diagnostics

The [56-cell static profile](../../../build-numa-broad/campaign_20261011_broad/profile_queue_cardinality_static_seed0.json)
contains correctness checks at M4 and M129. It varies node 1's expert count G
from 1 to 8, places the remaining 8-G experts on node 2, leaves node 3 empty,
and keeps each node's expert IDs in ascending order. The [cold profile](../../../build-numa-broad/campaign_20261011_broad/profile_queue_cardinality_cold_static_seed0.json)
repeats M4 and M512 for G=1..4. The M/G sweeps ran in ascending order without
counterbalancing. These are placement-conditioned profiles, not intrinsic
per-group cost curves; the frozen calibration fit excludes submit and combine
time.

| M512 groups on node 1 | Warm node 1 median | Cold node 1 median | Cold fit prediction |
| ---: | ---: | ---: | ---: |
| 2 | 0.37622 ms | 0.39683 ms | 0.27445 ms |
| 3 | 0.56368 ms | 0.60512 ms | 0.40236 ms |
| 4 | 0.63433 ms | 0.67459 ms | 0.53026 ms |

The cold M4 profile grows by roughly 17–18 µs per added node-1 group, near the
model's approximately 21 µs per group. M512 node-1 queue time, including
dispatch, rises faster than the frozen fit predicts for these placements.

The [14-cell placement-rotation diagnostic](../../../build-numa-broad/campaign_20261011_broad/profile_static_placement_rotation_seed0.json)
permuted the full expert weight axis and remapped IDs outside the timer; three
runtime contexts held nine physical weight copies. All rotated records passed
numerical checks. Over ten paired rounds, offset 1 measured a 1.028224×
aggregate speedup [1.023508, 1.032720] versus offset 0, while offset 2 measured
0.992570× [0.985694, 0.998977]. Offset 1's M4 step 0 was 0.079309 ms versus
0.076709 ms. This diagnostic did not meet the 5% target and is neither a
complete-call qualification nor a new policy; it does not establish an
intrinsic queue-cardinality curve.

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

The following trials used the same 80-record full-sequence replay and
10-round/10-repeat timing configuration, but were separate launches. Their
source and extension identities are recorded in each linked result. The
measurements are descriptive, not paired causal comparisons.

| Trial | Static replay | M<8 passes | M>128 passes |
| --- | ---: | ---: | ---: |
| [Tree arrival barrier, seed 0](../../../build-numa-broad/campaign_20261011_broad/tree_arrival_barrier_full64_seed0.json) | 761.84 ms | 24/28 | 20/24 |
| [Route-group views, seed 0](../../../build-numa-broad/campaign_20261011_broad/route_group_views_full64_seed0.json) | 758.87 ms | 23/28 | 20/24 |
| [Route-group views, seed 1](../../../build-numa-broad/campaign_20261011_broad/route_group_views_full64_seed1.json) | 790.91 ms | 6/28 | 21/24 |
| [Retained-binary control, seed 1](../../../build-numa-broad/campaign_20261011_broad/shared_staged_ranges_recheck_seed1.json) | 762.85 ms | 25/28 | 21/24 |
| [Common-launch clock, seed 0](../../../build-numa-broad/campaign_20261011_broad/node_launch_completion_full64_seed0.json) | 760.60 ms | 23/28 | 21/24 |
| [Tiny shared-input staging, seed 0](../../../build-numa-broad/campaign_20261011_broad/tiny_shared_stage_full64_seed0.json) | 764.97 ms | 24/28 | 21/24 |
| [Hierarchical counter, seed 0](../../../build-numa-broad/campaign_20261011_broad/hierarchical_counter_full64_seed0.json) | 762.18 ms | 24/28 | 20/24 |
| [Whole-expert columns through 32 row blocks, both projections, seed 0](../../../build-numa-broad/campaign_20261011_broad/shared_input_columns_bounded32_full64_seed0.json) | 768.50 ms | 26/28 | 21/24 |
| [Whole-expert columns through 32 row blocks, gate/up only, seed 0](../../../build-numa-broad/campaign_20261011_broad/shared_input_columns_up_only_full64_seed0.json) | 765.75 ms | 24/28 | 21/24 |
| [Second-arriver activation, seed 0](../../../build-numa-broad/campaign_20261011_broad/second_arriver_activation_full64_seed0.json) | 790.61 ms | 23/28 | 20/24 |

The seed-1 route-view result and same-seed retained control show why separate-run
differences are not causal evidence: the static small-cell pass count was 6/28
for route views and 25/28 for the control, while both had 21/24 large-cell
passes. The bounded whole-expert column trial let balanced and split reach
22/24 large cells, but only 20/28 small cells. Their replay means
(786.59/787.19 ms) exceeded static. Tiny shared staging still measured
M4-changing at 1.862×.
The up-only run also missed its target: M512-changing was 1.86268× and
M1024-changing was 1.80176×. Its 11 NUMA tests and 320 synthetic correctness
checks passed. After rejecting the two column trials, source, test and binary
were restored to the exact retained identities recorded in the
[restoration record](../../../build-numa-broad/campaign_20261011_broad/shared_input_columns_trials_rejection_restored.json).

The second-arriver trial used two acq_rel producer bits; the final writer
computed SiLU on 32 columns and skipped the separate activation barrier for
native calls with M≤4, at most four rows per group and eight groups per node.
Calibration and sharded calls used the retained pipeline. Its 11 NUMA tests and 320
synthetic comparisons passed, but M4-changing reached only 1.87448×. Static
M3-changing had a 1.48572061 ms mean (0.0916×), versus a 0.069288 ms median and
0.078121 ms p95; this outlier is not attributed to the source. The trial was
not retained, and source, test and binary were restored to the retained identities
in the [restoration record](../../../build-numa-broad/campaign_20261011_broad/second_arriver_activation_rejection_restored.json).

Earlier rejected experiments remain in their original artifacts. Active waits,
expert-aligned gangs, padded input rows, sparse-16 scheduling and caller-last
dispatch did not meet the broad target. Unrestricted weighted columns regressed
larger queues; expanding only the down cache did not resolve that regression.
The five-block bound retains larger rectangles. Delegated and combined wake
protocols were also rejected.

No single policy passed every requested cell, so these trials do not establish
broad 2× performance.

The [GC-overlap diagnostic](../../../build-numa-broad/campaign_20261011_broad/route_group_views_gc_overlap_summary.json)
found zero recorded GC overlap across its native call windows. A separate
[rusage trace](../../../build-numa-broad/campaign_20261011_broad/route_group_views_rusage_trace_seed1.rusage_trace.json)
did not reproduce the roughly 100 ms stalls; its 3.23 ms outlier coincided
with involuntary context switches and zero page faults. The cause remains
unresolved in these captures, which lack a thread-level timeline. These
diagnostic captures are not promotion timing, and no historical artifacts were
overwritten.

A subsequent [switch-event capture](../../../build-numa-broad/campaign_20261011_broad/retained_switch_trace_seed0_retry1.switch_summary.json)
recorded 60,819 process switch events for the retained binary and all 96 workers,
with no reported lost records. The workload completed with correctness checks
passing. The recorder finalized 3.303 MB of data, which `perf script` parsed
successfully; the original launcher classified its intentional SIGINT exit
as a recording infrastructure failure. That status file is preserved.

The capture contained one small call exceeding 1 ms: the first warmup call,
M1-uniform, took 3.598 ms. Its caller on CPU 32 was preempted for 3.334 ms.
This establishes caller preemption in that warmup delay, but does not explain
the roughly 100 ms measured outliers: none reproduced in this capture.
Worker off-CPU overlap cannot establish a critical-path delay without worker
completion markers; a worker may already have finished its contribution.

## Dense groups and worker scheduling follow-up

These are synthetic diagnostics on the same 96 physical CPUs, with one shared
worker pool and three weight replicas. They do not qualify serving integration.
All runs retain FP8 microkernels, block scales and the original top-k addition
order. Each full64 run uses 80 consecutive records, ten paired rotated rounds
and ten sequence repeats, with three separately pinned single-node references.
Historical artifacts and failed launch records remain intact.

The dense-row selector requires every token to route once to each expert in the
common set. Counts alone are insufficient: a duplicated-slot regression case
has the same counts but must retain whole groups. Three-node row partitioning
starts at M256 in `dense_rows`. The two-node diagnostic used replicas on nodes
1 and 3 and retained the existing three-team completion/combine protocol.

| Run / policy | Mean replay | Small ≥2× | Large ≥2× |
| --- | ---: | ---: | ---: |
| [Credit/row controls, seed 0](../../../build-numa-broad/campaign_20261011_broad/credit_rows2_full64_seed0.json): static | 761.38 ms | 26/28 | 20/24 |
| Same run: static_credit | 760.82 ms | 27/28 | 20/24 |
| Same run: dense_rows | 735.55 ms | 25/28 | 23/24 |
| Same run: two-node rows | 829.67 ms | 25/28 | 15/24 |
| [Credit/row controls, seed 1](../../../build-numa-broad/campaign_20261011_broad/credit_rows2_full64_seed1.json): static | 761.38 ms | 28/28 | 21/24 |
| Same run: static_credit | 759.37 ms | 28/28 | 21/24 |
| Same run: dense_rows | 734.76 ms | 27/28 | 22/24 |
| Same run: two-node rows | 828.82 ms | 26/28 | 15/24 |

Two-node row partitioning is rejected: paired speedups against static were
0.91769× [0.91644, 0.91896] and 0.91863× [0.91731, 0.92006]. Credited one-dimensional
barriers produced only 1.00073× [0.99890, 1.00284] and
1.00264× [1.00092, 1.00426]. They are an experimental control, with no broad
performance claim. Empty ranges carry no arrival count; a designated active
worker credits them, while actual job completion still requires every worker.
The preceding activation-only trial also missed the small-cell target.

`dense_groups` assigns common-set experts by sorted group position, rather than
logical ID modulo three. `dense_columns` adds weighted whole-group columns up to
eight row blocks when M≤256, credited one-dimensional barriers, and three-node
row partitioning for eligible common sets above 256. Non-common routes retain
static expert placement, but eligible six-to-eight-block groups still use the
widened executor. These selectors are diagnostic heuristics, rather than the
calibrated 5%-makespan split policy.

| Run / policy | Mean replay | Small ≥2× | Large ≥2× |
| --- | ---: | ---: | ---: |
| [Bounded columns, seed 0](../../../build-numa-broad/campaign_20261011_broad/dense_columns_full64_seed0.json): static | 759.00 ms | 24/28 | 20/24 |
| Same run: dense_groups | 764.34 ms | 23/28 | 20/24 |
| Same run: dense_columns | 731.94 ms | 24/28 | 23/24 |
| [Bounded columns, seed 1](../../../build-numa-broad/campaign_20261011_broad/dense_columns_full64_seed1.json): static | 762.22 ms | 26/28 | 21/24 |
| Same run: dense_groups | 766.60 ms | 25/28 | 20/24 |
| Same run: dense_columns | 736.15 ms | 25/28 | 23/24 |

Sorted-position assignment alone regressed aggregate time in both seeds.
The combined column candidate saved 3.57%/3.42% versus static, with paired
speedups 1.03697× [1.03492, 1.03922] and 1.03540× [1.03314, 1.03765].
M256 hot-eight improved from 1.9429× to 2.0533× in seed0, but M256 changing
remained at 1.9502×/1.9778×. Small changing-route cells also missed 2×.
The broad target and the ≥5% aggregate promotion gate remain unmet.
Both runs passed 320 numerical comparisons, four aliases and 320 routing checks.
The installation identity is in the
[column journal](../../../build-numa-broad/campaign_20261011_broad/dense_columns_install.json).

The [credited switch capture](../../../build-numa-broad/campaign_20261011_broad/credited_switch_trace_seed1.switch_summary.json)
completed correctly and recorded 110,377 events with no lost records.
Five small calls took 2.6–4.0 ms; caller preemption occupied 2.57–3.86 ms.
The captured process stream next showed worker CPU32 running, but does not
establish whether that worker had just awakened or was already runnable.
No roughly 100 ms outlier reproduced.

An initial attempt to give only the shared worker `SCHED_BATCH` matched creation
CPU34, while the timed caller later ran on CPU32. It is preserved as a placement
mismatch, not evidence for the intended treatment. Subsequent all-worker Batch
and Idle trials verified the actual 96 worker policies and affinities.
The [Idle trial](../../../build-numa-broad/campaign_20261011_broad/idle_workers_full64_seed0.json)
passed numerical checks, but regressed static to 766.54 ms and the column
candidate to 741.42 ms. It is rejected.

Starting with that trial, synthetic cells retain the already-collected native
phase/node snapshots from the final timing round. These statistics are separate
from the complete-call latency distribution over all ten rounds. For small
changing routes, gather/combine included roughly 10–13 µs; it includes worker
completion and caller wakeup, so that interval is not attributed solely to
floating-point reduction. A further control divided top-8 reduction into
32-channel blocks for M≤8, preserving one writer per output element and the
original slot order, without adding workers.

The [channel control](../../../build-numa-broad/campaign_20261011_broad/channel_columns_full64_seed0.json)
passed correctness, but its 733.41 ms aggregate did not improve on the same
run's dense-column control at 733.11 ms. Both passed only 26/28 small cells and
23/24 large cells. M4 changing remained below 2×. Reducing this part of the
combine work does not resolve the complete-call target.

The [expanded boundary matrix](../../../build-numa-broad/campaign_20261011_broad/channel_edges_seed0.json)
covered M={1,2,3,4,5,6,7,129,192,255,257,384,768,8192} with all four
distributions and ten paired rounds. Dense columns passed 25/28 small and
26/28 large cells; static passed 23/28 and 22/28. Although aggregate time
fell from 871.21 to 814.55 ms in this different corpus, M257 hot-eight
regressed from 2.077× to 1.865× against its fastest single-node reference.
The threshold that selects row partitioning above M256 is therefore unsafe
as a broad policy. At M8192 the dense-column diagnostic reached
2.608×/3.004×/3.203× for Zipf/hot-eight/changing routes. These successes do
not erase the boundary regression or qualify the original corpus.

The separate [counter diagnostic](../../../build-numa-broad/campaign_20261011_broad/channel_counters_seed0.json)
passed correctness and recorded all eight requested events in ten windows,
with 100% running time and acknowledged enable/disable operations. Two rounds
of ten full-sequence repeats gave the following totals for the native policies:

| Counter | Static | Dense columns |
| --- | ---: | ---: |
| User cycles | 480.23 billion | 465.13 billion |
| User instructions | 265.80 billion | 266.37 billion |
| LLC misses / references | 31.52% | 32.55% |
| IMC reads, sch0 + sch1 | 205,929.53 MiB | 223,946.17 MiB |
| IMC writes, sch0 + sch1 | 40,384.36 MiB | 38,301.00 MiB |

The candidate reduced cycles while increasing measured IMC reads by about
8.8%. This does not establish a memory bottleneck or a causal traffic benefit:
CPU events cover CPUs32–127 system-wide, including replay bookkeeping and the
untimed statistics getter. IMC events have package scope and include unrelated
traffic. Their PMU representative mask is 0,96; CPU96 lies in the collection
set, but CPU0 does not. These totals are not a whole-machine bandwidth claim.
The PMU supplied MiB units; modeled weight bytes were not used as measured
bandwidth. Instrumented timing is excluded from promotion.

Inspection of the installed Torch BF16×BF16 BRGEMM overload used by
`gemm_fp8.cpp` found a thread-local kernel cache on its warm path, rather than
an ATen global cache lock. Cache hits call `dnnl_brgemm_execute`; releasing the
hardware context does not clear that map. This rules out that specific proposed
lock explanation, while leaving oneDNN cache-miss internals unmeasured.

The follow-up native candidates are rejected. Restore the previously retained
runtime and its exact extension from the
[installation journal](../../../build-numa-broad/campaign_20261011_broad/retained_after_channel_restore.json);
keep only per-cell reporting of the existing final-round native statistics.
Candidate source snapshots, binaries, raw timing, correctness and counter
artifacts remain available for diagnosis. The broad 2× target remains unmet,
and no serving promotion is justified.

After restoration, all 11 existing NUMA tests passed. A
[16-cell reporting smoke](../../../build-numa-broad/campaign_20261011_broad/retained_stats_smoke.json)
at M={1,7,129,257} passed numerical/routing correctness with all three
single-node references. Each cell's reported logical and physical routed-slot
totals and final-round index were independently checked. This smoke validates
reporting, not speedup. Pre-commit checks passed for both retained changed files.

## Subset dispatch, barrier and input-placement diagnostics

The [subset-sharing trial](../../../build-numa-broad/campaign_20261011_broad/subset_staging_full64_seed0_retry1.json)
mapped each 32-row expert block to a verified contiguous subset of the first
local group's staged inputs. Its new M129 case reduced dispatched input rows
from 258 to 162 and passed aliasing and numerical checks. All 11 NUMA tests
passed. Full64 static/balanced/split replay measured 760.31/780.50/784.93 ms;
their small pass counts were 22/20/21 of 28 and large counts 20/21/20 of 24.
Split rarely exercised this reuse in the failing common-set cells. It did not
close the target, so the change is rejected. The first launch rejected an old
oracle identity before timing; that failure remains preserved.

A [dissemination-barrier trial](../../../build-numa-broad/campaign_20261011_broad/dissemination_local_full64_seed0.json)
replaced the shared arrival counter with five rounds of peer notifications.
Its flags occupied the NUMA-bound Team object; an initial heap-storage version
was corrected before timing. Twelve NUMA tests passed, including a three-thread
team to exercise non-power-of-two membership. Full64 static/balanced/split
measured 785.11/789.63/786.97 ms, with small counts 21/19/20 and large counts
20/20/20. Static M3 changing had median/p95 0.06910/0.07613 ms but mean
1.13984 ms, dominated by one round averaging 10.77312 ms. The outlier is
unexplained; M4/M5/M6 changing still missed 2×. The barrier is rejected.

Both native trials were restored to the exact retained
[source/binary identity](../../../build-numa-broad/campaign_20261011_broad/after_subset_dissemination_restored.json).

Two diagnostics then kept that binary and fixed input pages before runtime
creation, calibration, warmup and timing. Affinity alone left some allocator
pages on other nodes, so own-process `move_pages` migrated and queried all pages
covering hidden states, IDs and routing weights. This changes placement without
duplicating activations or changing tensor bytes. Both corpora retained hash
`e75710853492d4cafa8138b9bebb248f42ffd2f85eda701813e840a82f4dd5ee`.
Single-node controls kept their local inputs. Each run used static and the three
single-node references with ten paired rounds.

| Input placement | Verified input pages | Static replay | Small ≥2× | Large ≥2× |
| --- | ---: | ---: | ---: | ---: |
| [Node 1](../../../build-numa-broad/campaign_20261011_broad/fixed_input_node1_seed0.json) | 41,963 on node 1 | 764.63 ms | 21/28 | 20/24 |
| [Node 3](../../../build-numa-broad/campaign_20261011_broad/fixed_input_node3_seed0.json) | 41,962 on node 3 | 799.85 ms | 21/28 | 21/24 |

Both passed numerical checks; neither closes the target. These separate runs
establish verified placement conditions, rather than a paired causal estimate
between placements. Their page-query records and wrapper source are preserved
beside the timing artifacts.

## Synchronous callback ownership

The retained callback change replaces `std::function` parameters in
`CpuMoeExecutor` with Torch's `c10::function_ref`. Executor stages invoke these
callbacks synchronously and finish before returning; asynchronous Team jobs
still own their callbacks. No new wrapper or helper is introduced, and the FP8
kernel source and arithmetic remain unchanged.

The [retained-binary disassembly](../../../build-numa-broad/campaign_20261011_broad/CAF_fp8_callback_allocations.disasm.txt)
showed closure allocations of 232 bytes for gate/up, 224 for down and 32 for
SiLU on the executor path. With all 96 workers active, that is 288 allocations
per call. The
[new binary](../../../build-numa-broad/campaign_20261011_broad/function_ref_fp8_callback_paths.disasm.txt)
has no direct `operator new` call in that specialized FP8 function. This
establishes eliminated allocation work, not a complete-call speedup claim.

| Run | Static replay | Small ≥2× | Large ≥2× |
| --- | ---: | ---: | ---: |
| [All four policies, fixed node-1 inputs](../../../build-numa-broad/campaign_20261011_broad/function_ref_fixed_node1_seed0.json) | 764.50 ms | 23/28 | 20/24 |
| [Matching four-variant diagnostic](../../../build-numa-broad/campaign_20261011_broad/function_ref_fixed_node1_matched_seed0.json) | 763.81 ms | 22/28 | 21/24 |

The matching run used static and three single-node controls, as in the retained
fixed-input diagnostic. Aggregate time was essentially unchanged from its
764.63 ms. Several small changing calls were descriptively 1–2% faster, but
these separate launches do not establish paired causal gains. All small-cell
p95 ratios stayed within 5% regression relative to that diagnostic. Isolated
mean outliers remain, and M3 Zipf, M3/M4/M5 changing and large changing cells
still miss 2×. Balanced and split in the four-policy run were slower than static,
at 778.34/778.15 ms. The broad target and promotion gate remain unmet.

Retain this small ownership change because it demonstrably eliminates recurrent
allocations with a simpler synchronous contract and no measured aggregate
regression; continue tuning rather than treating it as qualification.
The [installation journal](../../../build-numa-broad/campaign_20261011_broad/function_ref_install.json)
records source/extension identity;
[build commands](../../../build-numa-broad/campaign_20261011_broad/function_ref_build_commands.json)
record the rebuilt native and FP8 objects.
All 96 selected NUMA, FP8 and MXFP4 tests passed, as did pre-commit checks.
[Captured-output validation](../../../build-numa-broad/campaign_20261011_broad/function_ref_primary_numerical.json)
completed all 1,920 records over 40 layers for each of the four policies.
Static, balanced and split matched captured outputs exactly; rows' maximum
absolute error was 0.03125 within the existing tolerance.
This validates captured outputs, rather than primary timing or serving throughput.

## Caller participation in the fused pipeline

The caller-participation trial parked the worker sharing the singleton caller's
CPU and executed that logical worker slot on the caller. It retained the fused
expert/combine callback, synchronous `function_ref` stages, all 32 barrier
participants per team and the existing FP8 arithmetic. The runtime serviced
pending caller work before waiting on any team. An additional last-team caller
case exercised the otherwise consequential wait-order deadlock risk.

All 11 NUMA tests passed, including caller migration, aliasing, shared runtime
creation, eviction and sharding. Independent static review found no blocking
protocol issue. These checks establish correctness, rather than performance.

The [complete fixed-node-1 run](../../../build-numa-broad/campaign_20261011_broad/inline_fused_fixed_node1_seed0.json)
used the same 80 ordered records, 64 cells, input placement, four policies,
three single-node controls and ten rotated rounds of ten repeats as the
committed synchronous-callback baseline. All numerical and routing checks
passed.

| Policy | Baseline replay | Caller participation | Small ≥2× | Large ≥2× |
| --- | ---: | ---: | ---: | ---: |
| Static | 764.50 ms | 765.68 ms | 22/28 | 20/24 |
| Balanced | 778.34 ms | 786.76 ms | 20/28 | 20/24 |
| Split | 778.15 ms | 786.53 ms | 19/28 | 21/24 |
| Rows | 1090.04 ms | 1092.34 ms | 10/28 | 12/24 |

Static still missed M3 Zipf, M3/M4/M5/M6 changing, M5 hot-eight,
M256 hot-eight and M256/M512/M1024 changing. Compared descriptively across
launches, static p95 increased by more than 10% for M2 Zipf/changing and
M3 hot-eight/changing. These observations do not establish paired causal
regressions, but provide no reason to retain the extra handoff protocol.

Reject the candidate and its candidate-specific test case.
The [restore journal](../../../build-numa-broad/campaign_20261011_broad/inline_fused_rejected_restored.json)
confirms that native source, tests, header, benchmark and installed extension
match the committed `1edaa1616c` baseline.
The [installation identity](../../../build-numa-broad/campaign_20261011_broad/inline_fused_install.json),
[build commands](../../../build-numa-broad/campaign_20261011_broad/inline_fused_build_commands.json)
and [cell analysis](../../../build-numa-broad/campaign_20261011_broad/inline_fused_fixed_node1_seed0.analysis.json)
preserve the rejected experiment. The broad 2× target remains unmet.

## Coordinated splits for shared token lists

Independent expert splits can destroy shared-input reuse. In the committed
baseline's M1024 changing cell, split gathered 6M rows per call, versus 3M for
whole ownership and M for equal row partitioning. Its node-time prediction was
about 30% below observed time. The one-expert/large-queue calibration did not
represent these small shared-input queues well.

The new split candidate measures frozen node costs for one through eight
experts sharing token lists. Five samples per point flush the participating
weights and scales before measurement, charging additional replica reads.
All shared-cost samples use the existing large-batch scratch/staging regime,
including short local slices. Calibration remains outside steady-state timing.
The original per-expert calibration serves other routes.

For large batches, the scheduler recognizes identical strictly increasing token
lists across the complete active expert set, within the existing eight-group
input-sharing limit. It evaluates two- and three-node partitions at 32-row
boundaries, moves every group's cuts together, and leaves one final tail.
It accepts a coordinated split only when measured shared-cost curves predict
at least 5% lower node makespan. Other inputs retain the existing scheduler.
FP8 arithmetic, full expert widths, weighting and original-slot reduction
are unchanged.

| Fixed node-1 input run | Static replay | Split replay | Split small ≥2× | Split large ≥2× |
| --- | ---: | ---: | ---: | ---: |
| [Initial 64-cell candidate](../../../build-numa-broad/campaign_20261011_broad/coherent_shared_split_fixed_node1_seed0.json) | 784.41 ms | 751.07 ms | 21/28 | 23/24 |
| [Refined 64-cell candidate](../../../build-numa-broad/campaign_20261011_broad/coherent_large_fixed_node1_seed0.json) | 764.22 ms | 748.52 ms | 22/28 | 22/24 |
| [Refined 56-cell boundary matrix](../../../build-numa-broad/campaign_20261011_broad/coherent_large_fixed_edges_node1_seed0.json) | 871.01 ms | 836.13 ms | 17/28 | 23/28 |

All runs used four native policies, three valid single-node controls and ten
rotated rounds of ten repeats. Refined split's paired speedup over static was
1.021×, CI [1.017, 1.025], in the main matrix and 1.042×,
CI [1.024, 1.053], in the boundary matrix. Isolated latency outliers remain;
the initial run's static mean includes a large small-call outlier.
These are synthetic diagnostics, rather than primary promotion evidence.

Changing-route speedups reached 2.08× at M512, 2.40× at M1024,
2.72× at M2048 and 3.12× at M4096. The boundary matrix reached 2.34×
at M384, 2.48× at M768 and 3.26× at M8192. These cells accepted all
eight expert splits while gathering M input rows. M256 hot-eight/changing
still missed 2× in the refined main matrix. M192 and M257 hot-eight,
and M192/M255/M257 changing, remained failures in the boundary matrix.
Those cells accepted no coordinated split. The existing whole-group model
and independent-split path still need work, as do the small-M failures.

Retain the shared-list scheduling improvement as experimental progress.
The frozen prediction covers node execution, excluding caller overhead and
final combine. The 5% node acceptance test therefore does not establish a
5% complete-step prediction. Neither aggregate result satisfies the primary
promotion gate, and the broad 2× objective remains unmet.

The existing numerical suite adds M513 shared lists with a distinct padded
tail and alias replay. All 96 selected NUMA, FP8 and MXFP4 cases passed.
[Captured-output validation](../../../build-numa-broad/campaign_20261011_broad/coherent_large_primary_numerical.json)
passed all 1,920 records for every policy: static, balanced and split matched
exactly; rows' maximum absolute error remained 0.03125 within the existing
tolerance. Pre-commit checks passed.
The [installation journal](../../../build-numa-broad/campaign_20261011_broad/coherent_large_final_install.json)
and [build commands](../../../build-numa-broad/campaign_20261011_broad/coherent_large_build_commands.json)
record the retained implementation. Shared calibration tables are preserved
in full warmup snapshots and per-cell metadata.

## Finish workers without output rows before combine

Nonleader workers whose existing output range is empty now finish before
waiting for all nodes to publish expert completion. Node leaders still publish
completion, and the runtime still joins every worker. This removes 87–91
unnecessary readiness pollers at M3–7 without changing projection arithmetic,
output ownership or alias safety. Large calls retain their existing combine.

| Fixed node-1 input run | Static replay | Split replay | Static small ≥2× | Split large ≥2× |
| --- | ---: | ---: | ---: | ---: |
| [Candidate seed 0](../../../build-numa-broad/campaign_20261011_broad/combine_waiters_fixed_node1_seed0.json) | 763.69 ms | 743.99 ms | 25/28 | 23/24 |
| [Candidate seed 1](../../../build-numa-broad/campaign_20261011_broad/combine_waiters_fixed_node1_seed1.json) | 765.06 ms | 744.62 ms | 26/28 | 22/24 |
| [Matched parent seed 1](../../../build-numa-broad/campaign_20261011_broad/coherent_large_fixed_node1_seed1.json) | 766.36 ms | 747.08 ms | 25/28 | 22/24 |

These 64-cell runs retain four distributions, four policies, three single-node
controls and ten rotated rounds. Seed 0's paired split speedup over static was
1.026×, CI [1.023, 1.030]. Against the matched seed-1 parent in separate
launches, small-call geometric mean latency ratios were 0.996 for static,
0.977 for balanced and 0.981 for split. Their p95 ratios were 0.999, 0.972
and 0.966. Static had isolated p95 increases above 10% at M1/M2 changing;
balanced and split had none. Cross-launch ratios are descriptive rather than
paired evidence of a causal improvement.

Retain the small scheduling change with its narrow benefit. Individual
small and large cells still fail 2×, and the aggregate advantage over static
remains below the promotion gate.
All 11 existing NUMA tests passed.
[Captured validation](../../../build-numa-broad/campaign_20261011_broad/combine_waiters_primary_numerical.json)
passed all 1,920 records for each policy, with exact static/balanced/split
outputs and rows' maximum absolute error 0.03125 within the existing tolerance.
Input payloads, sources and extension remained unchanged throughout validation.
The [installation journal](../../../build-numa-broad/campaign_20261011_broad/combine_waiters_install.json)
and [build commands](../../../build-numa-broad/campaign_20261011_broad/combine_waiters_build_commands.json)
preserve implementation provenance.

## Rejected paired gate/up execution

The candidate assigned matching gate/up tiles to one worker and applied the
existing SiLU stub immediately after both BF16 projection stores. It retained
the FP8 GEMM bodies and eliminated the separate activation barrier. Eligibility
required one row block per expert and enough weighted column jobs to avoid
idle workers; multiple-block groups and other executors retained the old path.
All 16 linked extension translation units depending on the executor header
were rebuilt from the recorded compiler flags. Independent static review found
no correctness blocker, and all 96 selected numerical tests passed.

The [64-cell run](../../../build-numa-broad/campaign_20261011_broad/paired_gate_up_fixed_node1_seed0.json)
passed synthetic correctness but regressed small calls. Balanced and split
small-call geometric mean latency increased by 7.16% and 6.97% against the
retained seed-0 run; each passed only 19/28 small cells. Static passed 18/28
and included a large M3 hot-eight outlier, which remains in the evidence.
Split replay was 750.70 ms versus the retained 743.99 ms. Large-cell latency
changed by less than 1% geometrically and did not close M256 changing's failure.
Separate launches do not establish paired causal effects; the repeated small
failures provide no reason to retain the candidate. The apparent paired
split/static aggregate gain is distorted by the static outlier.

Halving the column jobs can increase the maximum worker's tile count even
when every worker receives work, particularly with three equal expert groups.
Removing one barrier did not compensate in this experiment. Reject the
candidate rather than add another scheduling path for its narrow cases.
The [build journal](../../../build-numa-broad/campaign_20261011_broad/paired_gate_up_build_commands.json)
and [cell analysis](../../../build-numa-broad/campaign_20261011_broad/paired_gate_up_fixed_node1_seed0.analysis.json)
preserve it. The [restore journal](../../../build-numa-broad/campaign_20261011_broad/paired_gate_up_rejected_restored.json)
verifies native sources, tests, benchmark and installed extension against
`ba37af5210`. No captured validation or promotion claim was made for this
rejected candidate. The broad 2× objective remains unmet.

## Calibrate with nonzero activations

The [activation-data diagnostic](../../../build-numa-broad/campaign_20261011_broad/calibration_activation_diagnostic_v2.json)
compared zero and deterministic normal inputs using the first eight checkpoint
experts, fixed static ownership and verified node-1 input pages. Five rotated
rounds measured ten complete calls per case, both isolated warm calls and calls
after explicit expert-weight eviction. All outputs matched the existing
single-node operator exactly. Nonzero cold-weight complete-call medians were
8–14% higher at M129–512; M1–5 changed little. This isolates a bias in zero-input
calibration but does not qualify replay performance. The initial launcher
stopped before worker creation because the caller affinity was narrowed too
early; its script and log remain preserved.

Calibration now fills both fixture families from a local fixed-seed normal
BF16 pattern, scaled by the inverse square root of hidden width. Repeating
32 rows keeps generation inexpensive without consuming PyTorch RNG state.
The pattern is reproducible within the same standard-library implementation.
It remains an approximation of real layer activations. Calibration is still
untimed, uses five weight-evicted samples per point, and freezes its coefficients.
Execution arithmetic and FP8 microkernels are unchanged.

| Nonzero calibration run | Static replay | Split replay | Split small ≥2× | Split large ≥2× |
| --- | ---: | ---: | ---: | ---: |
| [Seed 0](../../../build-numa-broad/campaign_20261011_broad/nonzero_calibration_fixed_node1_seed0.json) | 766.70 ms | 745.99 ms | 21/28 | 22/24 |
| [Seed 1](../../../build-numa-broad/campaign_20261011_broad/nonzero_calibration_fixed_node1_seed1.json) | 765.24 ms | 741.80 ms | 21/28 | 22/24 |

Against the earlier launches with matching seeds, split aggregate time changed
by +0.27% and −0.38%. Its paired advantage over static was 1.028×,
CI [1.025, 1.031], and 1.032×, CI [1.028, 1.035].
These results establish no gain from changing calibration inputs. Several
small-cell p95 ratios exceeded 1.10 in seed 1, including 1.335 for split M6
hot-eight; none of those split increases repeated in seed 0. Balanced seed 1
also had large small-call outliers. Separate-launch comparisons are descriptive.
The small-group cost model and caller preemption still need work.

Retain the nonzero fixtures as a calibration realism correction, without a
speedup or promotion claim. All 11 existing NUMA tests passed.
[Captured validation](../../../build-numa-broad/campaign_20261011_broad/nonzero_calibration_primary_numerical.json)
passed all 1,920 records for every policy: static/balanced/split matched exactly,
and rows' maximum absolute error stayed at 0.03125 within the existing tolerance.
Input payloads, source and extension identities remained stable.
The [installation journal](../../../build-numa-broad/campaign_20261011_broad/nonzero_calibration_install.json)
and [build commands](../../../build-numa-broad/campaign_20261011_broad/nonzero_calibration_build_commands.json)
record the retained implementation. The fixture helper serves both calibration
families; existing behavior tests cover the unchanged replay interface.

## Rejected equal columns across short multiple-block groups

The candidate extended existing equal-row column ownership to grouped queues
of at most eight experts and five row blocks per expert. It retained the GEMM
loop, per-MB input offsets, weight unpacking, tails and stage barriers.
Independent review found no correctness blocker, and all 11 NUMA tests passed.
This was a bounded refinement of the earlier archived `grouped_columns` idea,
on the newer input-sharing foundation and with nonzero calibration.

The [64-cell result](../../../build-numa-broad/campaign_20261011_broad/equal_shared_columns_fixed_node1_seed0.json)
passed synthetic correctness but did not improve on nonzero calibration alone.
Split replay rose from 745.99 to 747.66 ms, while large-cell passes fell from
22/24 to 20/24. Split's large-call geometric mean latency increased by 1.11%;
M512 changing fell to 1.809×. M256 hot-eight/changing remained below 2×.
Reject this scheduling extension. Its
[cell analysis](../../../build-numa-broad/campaign_20261011_broad/equal_shared_columns_fixed_node1_seed0.analysis.json)
and [restore journal](../../../build-numa-broad/campaign_20261011_broad/equal_shared_columns_rejected_restored.json)
preserve the experiment and verify restoration to nonzero calibration alone.
The broad 2× target remains unmet.

## Rejected caller wait and placement changes

The bounded caller-wait candidate yielded at most eight times before returning
to the existing atomic wait, only while sharing a CPU with its worker team.
All 11 NUMA tests passed. Its
[64-cell run](../../../build-numa-broad/campaign_20261011_broad/caller_yield_bounded_fixed_node1_seed0.json)
passed correctness but split replay remained 745.92 ms versus 745.99 ms.
Split passed 21/28 small and 23/24 large cells; small-call geometric mean
latency increased 1.05%, and four small-cell p95 ratios exceeded 1.10.
Reject the candidate. The
[restore journal](../../../build-numa-broad/campaign_20261011_broad/caller_yield_bounded_rejected_restored.json)
confirms restoration of the committed source and nonzero-calibration binary.

A separate [caller-CPU63 run](../../../build-numa-broad/campaign_20261011_broad/controller63_early_finish_fixed_node1_seed0.json)
kept the same 96 compute CPUs, node-1 input pages and committed binary.
It tested whether early return of idle workers made CPU63 a better caller
location. Correctness passed. Split replay was 746.41 ms, with 22/28 small
and 20/24 large cells reaching 2×. Small geometric mean latency decreased
0.93%, while large latency increased 0.89%; M256 changing remained at 1.892×.
These separate launches establish no useful overall gain. Retain caller CPU32.
Neither experiment closes the broad target or qualifies primary performance.

## Rejected four-block contribution reduction

The candidate interleaved four independent channel-vector reductions, retaining
each channel's original top-k-slot addition order and final BF16 conversion.
The top-1 path and vector/scalar tails were unchanged. All 11 NUMA tests and
synthetic correctness checks passed; independent static review found no
semantic blocker. Existing numerical and alias coverage exercises this change.

The [64-cell run](../../../build-numa-broad/campaign_20261011_broad/combine_unroll4_fixed_node1_seed0.json)
did not improve complete calls: split replay was 747.17 ms versus 745.99 ms,
with 20/28 small and 22/24 large cells reaching 2×. Split's small geometric
mean latency increased 2.58%, including three p95 ratios above 1.10.
M256 changing remained at 1.865×. Several sampled combine means decreased,
including M256 changing from 39.62 to 36.19 µs, but these phase samples and
separate launches establish no overall benefit. Reject the added reduction
code. The [restore journal](../../../build-numa-broad/campaign_20261011_broad/combine_unroll4_rejected_restored.json)
verifies the committed source and retained nonzero-calibration extension.
The [cell analysis](../../../build-numa-broad/campaign_20261011_broad/combine_unroll4_fixed_node1_seed0.analysis.json)
preserves failures and tail ratios. The broad 2× objective remains unmet.

## Rejected consumer-side activation

The candidate removed the standalone SiLU stage for direct NUMA calls with
N≤K, computing the same BF16 activation in each down-projection worker's
private input scratch. The existing gate barrier, GEMMs, weighting and combine
order remained. A callback-local row-block cache reused activation across
consecutive columns; different workers and row-block revisits duplicated work.
All 11 NUMA tests and synthetic correctness checks passed.

The [64-cell diagnostic](../../../build-numa-broad/campaign_20261011_broad/consumer_activation_fixed_node1_seed0.json)
was slower: split replay increased from 745.99 to 817.54 ms. Split passed
21/28 small and 20/24 large cells; their geometric mean latencies increased
6.25% and 10.60%. Static large latency also increased 9.73%.
These separate-launch comparisons do not isolate causality, but provide no
reason to retain duplicated activation. The
[analysis](../../../build-numa-broad/campaign_20261011_broad/consumer_activation_fixed_node1_seed0.analysis.json)
preserves every cell and tail ratio.

Independent review identified a local restricted input pointer subsequently
aliasing scratch written within the callback. No miscompile was observed, but
the qualifier must be removed before any retest; this unresolved concern
precludes treating the diagnostic as qualification evidence. Reject the
candidate without a retest or captured-validation claim. The
[restore journal](../../../build-numa-broad/campaign_20261011_broad/consumer_activation_rejected_restored.json)
confirms committed source and retained extension identities. The broad target
remains unmet.

## Equal groups across row-block transitions

Extend whole-group column scheduling to equal groups of six through nine
32-row blocks when their column tasks divide evenly across the team.
This retains full-width kernels while avoiding repeated weight unpacking at
rectangle boundaries. Other queues retain their existing decomposition.
Recognized common routes now retain the calibrated whole-group alternative
when coordinated splitting is rejected, rather than falling back to independent
cuts that lose input reuse. Common-route checks also compare slot ranges after
one token division per row.

The benchmark freezes 26 calibration row counts, adding 96, 160, 161, 192,
224, 225, 255, 257, 288 and 289. Calibration is untimed. The intermediate
[eight-block run](../../../build-numa-broad/campaign_20261011_broad/equal_divisible_columns_fixed_node1_seed0.json),
[shared-choice seed 0](../../../build-numa-broad/campaign_20261011_broad/shared_choice_columns_fixed_node1_seed0.json),
[seed 1](../../../build-numa-broad/campaign_20261011_broad/shared_choice_columns_fixed_node1_seed1.json),
[transition retry](../../../build-numa-broad/campaign_20261011_broad/shared_choice_transition_fixed_edges_node1_seed0_retry1.json)
and [nine-block edges](../../../build-numa-broad/campaign_20261011_broad/nine_block_shared_choice_fixed_edges_node1_seed0.json)
remain preserved. The transition launch first rejected stale oracle identity
before timing; its failed artifact is retained.

After the slot-range simplification, split replay measured
[742.25 ms in seed 0](../../../build-numa-broad/campaign_20261011_broad/shared_slot_ranges_transition_fixed_node1_seed0.json)
and [745.25 ms in seed 1](../../../build-numa-broad/campaign_20261011_broad/shared_slot_ranges_transition_fixed_node1_seed1.json).
Small pass counts were 21/28 and 23/28; large counts were 23/24 in both.
The [boundary matrix](../../../build-numa-broad/campaign_20261011_broad/shared_slot_ranges_transition_fixed_edges_node1_seed0.json)
passed 31/36 large cells. These separate launches establish no broad 2× result.

## Combine common splits on their owning node

At M289 hot8, the preceding split assigned rows `[97,96,96]`, placing consecutive
token intervals on nodes `[1,2,0]` to keep the partial block last. Fixed output
thirds then read contributions remotely. For accepted complete common routes,
record those intervals and combine each on its owning node. Reset ranges on
every call. Keep the global input-completion barrier, original top-k addition
order, and participation of every node leader, including empty queues.

The [boundary matrix](../../../build-numa-broad/campaign_20261011_broad/owned_combine_fixed_edges_node1_seed0.json)
passed correctness and measured 912.80 ms split replay versus 928.98 ms.
M289 hot8 reached 2.058×, with sampled combine mean decreasing from
72.1 to 33.9 µs; changing reached 2.131×, with combine decreasing from
49.4 to 28.7 µs. Large pass count increased from 31/36 to 33/36.
Changing M255, M257 and M288 still missed, at 1.956×, 1.966× and 1.954×.
Large geometric mean latency decreased 0.94%, with no large p95 ratio above
1.10. Small calls were unchanged by the ownership mechanism; one M4-changing
p95 ratio was 1.713. Separate launches do not isolate its cause.

The broader matrices use the same fixed input placement, 96 compute CPUs,
three single-node controls, four policies, and ten paired rounds:

| Run | Split replay | Small ≥2× | Large ≥2× | Paired split/static, 95% CI |
| --- | ---: | ---: | ---: | --- |
| [Seed 0](../../../build-numa-broad/campaign_20261011_broad/owned_combine_fixed_node1_seed0.json) | 740.91 ms | 20/28 | 23/24 | 1.0314 [1.0286, 1.0341] |
| [Seed 1](../../../build-numa-broad/campaign_20261011_broad/owned_combine_fixed_node1_seed1.json) | 743.39 ms | 21/28 | 22/24 | 1.0272 [1.0220, 1.0321] |

Against the corresponding slot-range runs, large geometric mean latency changed
+0.14% and −0.16%; neither had large p95 ratios above 1.10. Small mean ratios
were +0.29% and +1.19%, with isolated p95 ratios above 1.10 preserved in the
analysis artifacts. Retain the measured boundary locality fix, without claiming
broad improvement. Unequal split counts may increase maximum combine work;
the frozen prediction still covers inner node execution and excludes final
combine and caller overhead.

The existing replay test now covers M257 nine-block tails and M289 combine
ownership, including aliasing followed by a changed batch. The selected
FP8/MXFP4/NUMA suite passed **96 tests**, with 100 deselected. All applicable
pre-commit hooks passed after formatting. No FP8 body or executor interface
changed. The installed extension SHA-256 is
`42e23f9b82ca9725c75cc6e7811043f3293c892b072e91059e4bae39ecb91125`;
full [installation identity](../../../build-numa-broad/campaign_20261011_broad/owned_combine_install.json)
and build commands are retained.

[Captured numerical validation](../../../build-numa-broad/campaign_20261011_broad/owned_combine_primary_numerical.json)
passed all 1,920 records under each of four policies, across all 40 layers.
Static, balanced and split maximum absolute error was zero; rows had 0.03125
maximum error and passed existing tolerances. Payload, source and extension
identities were unchanged. This is numerical validation, not primary timing
qualification. The broad 2× and original promotion gates remain unmet.

## Rejected callback allocation and cost lookup changes

The inline callback-storage trial replaced three heap-backed worker closures
with small wrappers referring to a synchronous stack callback. Disassembly
verified removal of those allocations, and all 11 NUMA tests passed.
Its [matrix](../../../build-numa-broad/campaign_20261011_broad/inline_job_storage_transition_fixed_node1_seed0.json)
measured 745.90 ms split replay versus 742.25 ms, passing 20/28 small and
23/24 large cells. Reject it; the
[restore journal](../../../build-numa-broad/campaign_20261011_broad/inline_job_storage_rejected_restored.json)
verifies the previous source and binary.

The frozen cost-cache trial precomputed unchanged interpolation values for
groups of zero through 32 rows. All 11 NUMA tests passed.
[Seed 0](../../../build-numa-broad/campaign_20261011_broad/small_row_cost_cache_transition_fixed_node1_seed0.json)
and [seed 1](../../../build-numa-broad/campaign_20261011_broad/small_row_cost_cache_transition_fixed_node1_seed1.json)
measured 740.32 and 743.18 ms split replay, passing 22/28 small and 22/24 large
cells in both. Small geometric mean latency changed −1.90% and +1.82%;
large changed +0.57% and −0.24%. Gains were inconsistent, and recorded p95
ratios included M512 hot8 at 1.155 in seed 1. Reject the extra cache;
the [restore journal](../../../build-numa-broad/campaign_20261011_broad/small_row_cost_cache_rejected_restored.json)
confirms restoration before the ownership experiment.

## Rejected shared-cost placement models

The [large shared-cost LPT trial](../../../build-numa-broad/campaign_20261011_broad/shared_lpt_fixed_edges_node1_seed0.json)
scored each candidate whole queue with the existing common-input calibration,
retaining expert order, the 1% history preference and the 5% split gate.
All 11 NUMA tests and replay correctness passed. Boundary split replay measured
913.47 ms versus 912.80 ms; large pass count fell from 33/36 to 32/36.
M257 changing reached 2.029×, but M257 hot8 selected a coordinated split and
fell to 1.852×, with mean and p95 latency ratios 1.129 and 1.128.
Reject this model change. Its
[restore journal](../../../build-numa-broad/campaign_20261011_broad/shared_lpt_rejected_restored.json)
confirms restoration of `eb43a323f0` source and retained extension.

The small shared-cost trial measured true small-scratch common-input groups
at fixed knots `{1,4,5,8}` and reused identical single-expert measurements.
It used those frozen costs only for complete common routes at M≤8.
The new curves and row axis were exported in stats and report metadata.
All 11 NUMA tests passed, including public assertions for the omitted-5 knot.
Independent source review found no semantic blocker. Matching-knot reuse,
interpolation, fixture slots, and old calibration order were preserved.

| Trial | Split replay | Small ≥2× | Large ≥2× | Small mean latency ratio |
| --- | ---: | ---: | ---: | ---: |
| [Seed 0](../../../build-numa-broad/campaign_20261011_broad/small_shared_fixed_node1_seed0.json) | 743.27 ms | 24/28 | 23/24 | 0.9723 |
| [Seed 1](../../../build-numa-broad/campaign_20261011_broad/small_shared_fixed_node1_seed1.json) | 748.54 ms | 22/28 | 22/24 | 0.9904 |

These mean ratios compare separate launches with corresponding owned-combine
runs. No p95 ratio exceeded 1.10 in seed 0; seed 1 M7 hot8 reached 1.170.
Its sixth round averaged 0.4740 ms versus about 0.10–0.11 ms in other rounds;
the outlier is unexplained. The first split replay round was also an outlier.

The [queue diagnosis](../../../build-numa-broad/campaign_20261011_broad/small_shared_m7_hot_queue_diagnosis.json)
uses existing summed physical counts: M7 hot8's average whole-group placement
changed from `[3,3,2]` to `[3,2,3]` in both seeds. Median latency increased
from 96.13 to 102.97 µs and from 95.66 to 103.93 µs. Cold calibration favored
the latter distribution, while the preceding warm replay's node-2 queue
completed sooner. These observations expose a ranking mismatch without
isolating its cause. Both sizes share placement history, so changing small
owners can also affect later large calls within the existing 1% rule.
Reject the policy rather than retain a better inner model with this tail risk.
The [restore journal](../../../build-numa-broad/campaign_20261011_broad/small_shared_rejected_restored.json)
verifies restoration of native source, benchmark, tests and extension.
No captured-validation or primary-performance claim is made for either trial.

## Small-boundary counter diagnostic

The restored binary's [instrumented replay](../../../build-numa-broad/campaign_20261011_broad/owned_small_boundaries_counters_seed1.json)
used M5/6/7/257, four routes, four native policies, three single-node controls,
five rotated rounds and 100 sequence repeats. All 35 counter windows completed
and correctness passed. Source and extension identity stayed fixed.
This is a separate diagnostic, excluded from promotion.

The [counter analysis](../../../build-numa-broad/campaign_20261011_broad/owned_small_boundaries_counters_seed1.analysis.json)
reports median split-window time of 1,150.29 ms, with 672.13 ms summed calls;
median time outside calls was 471.92 ms. Node-3 control had 1,914.22 ms windows,
1,884.09 ms summed calls and 30.41 ms outside calls. Native replay converts
full stats after every call, outside its latency timer. Control replay does
less between calls. Native workers also poll for up to 5 ms after work.
This bookkeeping gap could affect subsequent calls and warrants a separately
labeled uninterrupted replay comparison.

Split cycles/reference-cycles ratio was 1.2720 versus 1.2656 for node 3;
these aggregate counters do not support a frequency-drop diagnosis.
Split IPC was 0.602 versus 1.045, and LLC miss/reference ratio was 0.404 versus
0.553. Counters cover requested CPUs, including polling, bookkeeping and other
activity; these are not isolated GEMM metrics. No memory-controller events
were collected, and no measured bandwidth claim follows from LLC counts.
Reported migrations are system-wide events, not evidence of worker migration.
The broad 2× and original promotion gates remain unmet.

## Statistics-sampling sensitivity

The new `--native-stats-mode post-timing` diagnostic omits Python stats
conversion between timed native calls and collects a separate corpus replay
after all timing rounds, using `--diagnostic-repeats`. The default remains
`per-call`. Separate replay statistics carry their sampling contract and no
timing-round index, including in failure diagnosis. Kernel-control layouts
are installed before diagnostic collection. This mode is explicitly excluded
from promotion; its phase samples cannot attribute a particular timed miss.

The [base-policy matrix](../../../build-numa-broad/campaign_20261011_broad/stat_sampling_post_fixed_node1_seed0.json)
used the retained extension, seed 0, fixed node-1 inputs, 10 rotated paired
rounds and 10 sequence repeats. All numerical checks passed. Split replay
was 739.65 ms versus 740.91 ms in the preceding per-call launch, with 23/28
small and 23/24 large cells reaching 2×. M5 hot8/changing, M6 changing,
M7 hot8/changing and M256 changing missed. The split/static paired speedup
was 1.0349, CI [1.0332, 1.0365], below the 5% gate.
Separate-launch comparisons are descriptive: this is not a kernel improvement
and does not explain away the remaining misses.

The [layout smoke](../../../build-numa-broad/campaign_20261011_broad/stat_sampling_v2_shard_smoke_retry.json)
validated the corrected post-timing path at M1/7/129 across balanced,
unequal and all three TP2 kernel controls. Numerical checks passed, diagnostic
stats were separately labeled, and promotion remained unevaluated.
The first launcher attempt failed argument parsing before execution because
`--controls` had no value; its failure artifact is preserved.

## Scheduler and metadata overhead trials

The [per-call cost-reuse trial](../../../build-numa-broad/campaign_20261011_broad/group_cost_reuse_fixed_node1_seed0.json)
cached each active expert's three calibrated costs during estimation and
reused them for initial placement, reducing lookups from 7–8 to 3 per expert.
Independent source review confirmed unchanged values, sorting, addition order,
history preference and split thresholds. All 11 NUMA tests and replay checks
passed. Split replay was 747.23 ms versus 740.91 ms in the retained seed-0
launch; small/large geometric mean latency ratios were 1.0032/1.0051.
Small p95 ratios exceeded 1.10 at M1 zipf, M4 changing, M5 hot8 and M7 changing.
The result did not establish a consistent improvement; reject this added cache.
The [restore journal](../../../build-numa-broad/campaign_20261011_broad/group_cost_reuse_rejected_restored.json)
verifies the retained native source and extension hashes.

The buffer-page-count trial cached the existing base-page count after successful
allocation and cleared it on release, removing 39 page-size queries per call.
Placement verification and huge-page counting units remained unchanged.
Independent lifecycle review found no blocker; all 11 NUMA tests and
[synthetic replay checks](../../../build-numa-broad/campaign_20261011_broad/cached_buffer_pages_fixed_node1_seed0.json)
passed.

Its performance is **ineligible**: an unrelated oneDNN build in `vllm_local_2`
started during timing with CPU affinity 0–383. The
[affinity observation](../../../build-numa-broad/campaign_20261011_broad/cached_buffer_pages_fixed_node1_seed0.host_build_observation.json)
and [CPU observation](../../../build-numa-broad/campaign_20261011_broad/cached_buffer_pages_fixed_node1_seed0.host_cpu_observation.json)
record the overlap. Observed compiler CPUs 256/288 are SMT siblings of benchmark
CPUs 64/96. Preserve this as a shared-host correctness diagnostic, with no
performance acceptance or rejection. The
[pending-trial restore journal](../../../build-numa-broad/campaign_20261011_broad/cached_buffer_pages_pending_restored.json)
verifies restoration of the retained source and extension; candidate source,
binary and build provenance remain available for a clean rerun.

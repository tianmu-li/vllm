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

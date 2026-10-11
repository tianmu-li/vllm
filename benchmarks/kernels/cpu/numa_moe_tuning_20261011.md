# NUMA FP8 MoE tuning results, 2026-10-11

The current final native candidate is **active-experts**: `moe_numa.cpp`
SHA-256 `9551cde9d9da4cc12c2982ea90f5ef6906096c2be7137080641baebf8fdc3f9e`,
extension SHA-256
`57d27406f5a20c04660ceedf062a6a70957b5b0769971e2f5609a36fc912a665`.
The 2× complete-call target remains unmet for replay tuning: every fixed
policy misses cells in both requested bands. The per-cell any-policy indicator
is not a qualification union, and these synthetic diagnostics do not promote
the operator for serving.

The 64-cell matrix covers
`M={1,2,3,4,5,6,7,8,32,128,129,256,512,1024,2048,4096}` under uniform, Zipf,
hot8, and changing routes. This gives 28 cells for `M<8` and 24 cells for
`M>128`. Each result uses 10 rounds of 10 full ordered-sequence repeats under
the `sequence_replay` cache condition, with a 10 ms untimed pause between
variants. Cell speedups use complete-call means against that cell's fastest
numerically valid single-node control.

| Policy | Seed 0, `M<8` | Seed 0, `M>128` | Seed 1, `M<8` | Seed 1, `M>128` |
| --- | --- | --- | --- | --- |
| Static | 22/28; M1 uniform 1.744× | 19/24; M129 hot8 1.825× | 22/28; M1 hot8 1.697× | 19/24; M129 hot8 1.862× |
| Balanced | 18/28; M6 hot8 1.439× | 19/24; M129 hot8 1.821× | 18/28; M6 hot8 1.259× | 19/24; M129 hot8 1.775× |
| Split | 19/28; M6 hot8 1.431× | 20/24; M129 hot8 1.646× | 19/28; M2 hot8 1.258× | 20/24; M129 hot8 1.647× |
| Rows | 10/28; M3 changing 1.192× | 9/24; M256 uniform 1.116× | 9/28; M1 changing 1.222× | 10/24; M256 uniform 1.133× |

The table uses the [seed-0](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/active_experts_full64_seed0_retry.json) and
[seed-1](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/active_experts_full64_seed1.json) full-grid results.

The paired aggregate split/static speedup was 1.00121× [0.99924, 1.00285] for
seed 0 and 1.00790× [1.00456, 1.01116] for seed 1. These are bootstrapped
round-level confidence intervals over the 80-record synthetic diagnostic,
not per-cell intervals or timed primary-corpus results. Both active-experts
runs passed synthetic numerical checks over 80 records, 320 policy outputs,
and four alias outputs; maximum absolute error was `4.77e-7`.

The cold-leader-sleep candidate was rejected as the retained path. Its
[seed-0 run](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/cold_leader_sleep_full64_seed0.json) produced a paired
split/static aggregate of 0.99806× [0.99630, 0.99981], with split passing
18/28 small cells (weakest M6 hot8, 1.332×) and 20/24 large cells (weakest
M129 hot8, 1.796×). The matched active-experts
[seed-0 driver run](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/final_active_experts_shared_seed0.json) measured
0.99940× [0.99570, 1.00293], with split passing 21/28 small cells (weakest
M5 hot8, 1.302×) and 20/24 large cells (weakest M129 hot8, 1.717×). Both
used the same frozen corpus and weights. This is one shared-host seed, not a
causal comparison or evidence of a significant regression. The 100-cycle
[lifecycle stress check](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/cold_leader_sleep_lifecycle_stress.json) completed
with thread count 1 before and after; it establishes lifecycle behavior only.
Both runs passed synthetic correctness checks (80 records, 320 policy
outputs; maximum absolute error `4.77e-7`), with no captured outputs compared.

The cold-leader candidate used native `moe_numa.cpp` SHA-256
`c13856a19961b99b24606ec87c3065e6d33230cf1905e24fc0ae06acfed20b88` and
extension SHA-256
`6bafe24990181a6a31b9019bce25298718d073b21f96e898a564acc8f39f6954`.
The latest [host inventory](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/cold_leader_sleep_shared_host_inventory.json)
shows Firecracker VMs on CPUs 0–3 and 4–7, each about 8.4 GiB RSS; nodes 1 and
2 share a socket with node 0, and both VMs allow memory on nodes 0–5. Treat
that measurement as shared-host diagnostic evidence only.

## Driver check and remaining bottlenecks

An additional same-binary [diagnostic](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/final_active_experts_stats_off_seed0.json)
removed per-call statistics retrieval between native calls. Static coverage
was 21/28 small and 18/24 large cells, versus 22/28 and 19/24 in the adjacent
original-driver run. Static replay time was 832.719 versus 830.868 ms. This
single shared-host comparison does not establish a significant regression;
the stats-off run also failed the all-cell 2× target. Some individual misses
cleared while other cells changed to fail. Its launcher records the diagnostic shim hash.
The standard driver remains unchanged and records statistics outside the
complete-call timer.

The active-experts [profile](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/active_experts_live_placement_profile.json)
observed sparse scheduling times about 1.9–2.8 microseconds lower than the
separate hot-stage profile. This is an unpaired diagnostic.
The retained path also reuses node-local scratch and snapshots, gathers large
inputs once, schedules columns for small expert queues, and combines weighted
rows on workers in original top-k order. Larger queues retain the original
rectangular scheduling. Joint hot-group split trials retain the 32-row
boundary and 5% predicted-makespan improvement requirement.

Sparse cold diagnostics still showed an approximately 180-microsecond excess
on node 1 in the first parallel stage. Neither caller participation nor a
separate sleeping-leader wake signal resolved it. The phase evidence locates
the excess in the first parallel phase. Worker startup/barrier delay is an
interpretation; its cause remains unresolved. Sparse hot and changing routes also remain sensitive to placement and
cache state; adding copies or splits alone did not meet the broad target.
Both seeds' aggregate split gains over static remain below the required 5%.
Retain the simpler tuning as experimental replay work; reject serving
promotion and retain the numerical/performance gates.

## Provenance and validation

Full-grid timing runs use Qwen/Qwen3.6-35B-A3B-FP8 layer-12 weights,
E256/top8/H2048/N512,
BF16 activations and block-FP8 weights/scales. There is one request batch,
one persistent runtime, and three distinct local weight copies. Worker CPUs
are 32–63, 64–95 and 96–127 on NUMA nodes 1, 2 and 3 (32 physical cores each),
with caller CPU32 and no SMT compute threads. Sparse cold diagnostics used
E10 rotation runtimes containing eight selected experts. Native host OMP/MKL counts are
1, wait policy PASSIVE, blocktime 0. Single-node controls each use 32 threads,
ACTIVE waiting, blocktime 5 and TPAUSE 0. Commands use `uv run --no-sync
.venv/bin/python` and preload repository libiomp5 plus system tcmalloc.

The imported extension is `vllm/_C.abi3.so` in this worktree. The exact source
and extension hashes appear above; [build commands](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/active_experts_build_commands.json)
record compiler/link commands. Results record source identity, preload hashes,
affinity, operator schemas, calibration and input hashes. The stats-off
diagnostic omits per-call native snapshots and their derived summaries.
Source descends from `c5a3a1389df9aae4be73bcac7b8eea6164af02c7` through
prototype commit `c5abdc9dc39a970f2b1218d4dc9377764174cc71`.
Layer-12 weight-file SHA256 is
`25b2726ac8d55dddc767c7e214e53072e22202cbe35f59441d9bb938122cc0e6`;
tensor SHA256 is
`179f1ccfe059d35c1e0f957c6681440eb2fb65307dd3d6e61206b06905469895`.
The [live placement check](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/active_experts_live_placement_verified.json)
verified distinct weight storage and every live weight/scratch mapping on
its intended node. Weights and small scratch use base pages; large scratch
had 426 MiB of actual huge-page backing in that diagnostic.

Memory-controller counters were measured in a separate [hot-stage run](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/hot_stage_full64_imc.json). The four IMC aliases cover both sockets system-wide, including driver
work and unrelated traffic; they are not per-NUMA/kernel-only bandwidth or
a bandwidth-ceiling proof. LLC behavior was not measured. Inferred dispatch
and weight traffic remain models, not measured bandwidth.

[Final captured-output validation](/home/tianmuli/vllm_test/vllm_local/build-numa-broad/campaign_20261011_broad/final_active_experts_primary_numerical.json)
completed 1,920 comparisons per policy across 40 layers and 48 ordered steps:
120 prefill, 840 mixed and 960 decode records per policy. Static, balanced
and split matched the captured BF16 outputs exactly. Rows passed the stated
`atol=rtol=0.01` tolerance, with maximum absolute error 0.03125. Frozen inputs,
native source and extension were unchanged throughout validation. This is
captured-output parity, not an independent oracle or a timing measurement.
The corpus file SHA256 is
`4160faf51bd653f43b855f951c93b6259162a4971caff25ea64c460c26c649bb`;
full captured-layer weights SHA256 is
`bc9a62f6a3bcdf5b6c8cbbb79062d79ab1e0ee2cd6dbbd12a23dbe596d4fbea5`.

The nearest NUMA pytest selection passed 11 tests with all three nodes
available. An earlier 85-test shared quantized suite passed with the same
FP8/header source hashes. Changed files passed pre-commit checks. Coverage
includes changed routes, partial invalid IDs, immutable latest-call snapshots,
staging boundaries, split tails, aliasing, shared scratch and worker teardown.

The two-seed synthetic timing comparison omits the 96-thread, native TP2 and
unequal-shard controls, and is supplemental evidence. It does not establish
the primary replay promotion gate or full-model serving throughput.

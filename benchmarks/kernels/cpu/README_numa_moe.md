# CPU NUMA MoE replay harness

`benchmark_cpu_numa_moe.py` validates and times the experimental native
`cpu_numa_moe` runtime. It does not capture model traffic, start a model server,
or download checkpoint data. Supply an ordered offline replay and the raw
block-FP8 weights for its MoE layers. It fails closed unless the checkout HEAD
descends from the pinned source ancestor
`c5a3a1389df9aae4be73bcac7b8eea6164af02c7` and the imported `vllm._C`
extension resolves inside this checkout. Results record the exact HEAD and
source diff hash.

The runtime creation op receives one layer of VNNI-prepacked weights. The
harness packs each layer with `_C.convert_weight_packed`, then creates a
persistent handle for that layer. Those handles share one persistent pool of
three node-local worker teams. It preserves corpus record order for warmup and
every timed replay, dispatching each record to its layer handle. The reference
pass groups records by layer to reuse dequantized reference weights, and by
policy to reuse installed control slices. It compares every record and
selected policy before timing starts.
Its Torch host uses one thread for numerical validation, then restores the
configured count before warmup and timing. Native worker teams keep their
configured thread counts throughout.
Layer handles also share node-local scratch under the pool lock; expert weights
and scales remain separate for each layer and replica. Full-width execution
reduces weighted BF16 rows directly from this scratch in original top-k order.
Workers spin for up to 5 ms before parking. An idle worker sharing the caller's
CPU yields so the caller can submit the next stage.
Worker failures stop the shared pool: destroy every layer handle before
creating a fresh pool to recover.

## Build

Start from the pinned checkout and follow the repository's `uv` environment
setup and CPU build prerequisites. Configure and install the native extension
inside that checkout:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/cmake -S . -B build-numa -G Ninja \
  -DVLLM_TARGET_DEVICE=cpu \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD" -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build-numa --target _C --parallel 32
.venv/bin/cmake --install build-numa --component _C
```

Use one build process at a time. Keep the extension, CMake cache, compiler
commands, and source hashes with the run artifacts.

## Input files

Prepare checkpoint expert weights and fixed Sonnet token IDs, then capture from
one TP1 reference runtime in a fresh directory:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/prepare_numa_moe_capture.py \
  --output-dir /data/numa_run/inputs
PYTHONPATH="$PWD" \
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$PWD/.venv/lib/libiomp5.so" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 CPU_VISIBLE_MEMORY_NODES=1,2,3 \
VLLM_CPU_OMP_THREADS_BIND=32-127 VLLM_CPU_NUM_OF_RESERVED_CPU=0 \
OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 OMP_WAIT_POLICY=ACTIVE \
KMP_BLOCKTIME=5 KMP_TPAUSE=0 \
KMP_AFFINITY='granularity=fine,explicit,proclist=[32-127]' \
taskset -c 32-127 uv run --no-sync .venv/bin/python -P \
  benchmarks/kernels/cpu/capture_cpu_numa_moe.py \
  --checkpoint /data/checkpoint --inputs /data/numa_run/inputs \
  --output-dir /data/numa_run/capture --max-batched-tokens 1152
```

For node-local TP1 capture, pass `--reference-node 1` and set the launch affinity
to CPUs `32–63` with `OMP_NUM_THREADS=32` and `MKL_NUM_THREADS=32`. Capture rejects
missing explicit OpenMP binding. Replay still uses the three configured teams.
The existing-operator control workers use these same OpenMP wait settings.
They install explicit CPU lists before importing the extension, including a
separate list for each TP2 rank, and verify helper-thread coverage before timing.
The native runtime schedules its own persistent teams; its host's OpenMP pool
can sleep outside Torch preprocessing and calibration.

Numerical checks fail immediately by default. `--measure-invalid-controls`
retains numerical failures from row partitioning and sharding controls and
continues their diagnostic timing at the same tolerance. Reports mark these
controls invalid, exclude them from valid-control comparisons, and reject
promotion if any measured control fails. Whole-group policies, routing, aliasing,
placement, and collective failures still stop the run.

The default 1,152-token step budget permits a full 1,024-token prompt alongside
active decode requests. Capture records the actual step phase and token count.
Pass the prepared token IDs directly; decoded display text can retokenize
differently. The capture also preserves each original operator output for the
untimed replay correctness pass. Whole-group policies must match these outputs
exactly; row partitioning and intermediate slices use the stated numerical
tolerance. An independent Torch reference also models the original 32-row
small-GEMM/BRGEMM choices, branch-specific scaled-weight rounding, BF16 stage
rounding, and slot-ordered reduction. It approximates the intrinsic SiLU and
GEMM accumulation and therefore uses the stated tolerance.

Save the corpus as a PyTorch file containing `metadata` and an ordered `records`
list:

```python
{
    "metadata": {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "fixed_sonnet_prompts": True,
        "input_tokens": 1024,
        "generated_tokens": 128,
        "prompts_per_concurrency": 256,
        "concurrency": [8, 32, 128],
        "mtp_enabled": False,
        "deterministic": True,
        "captured_prefill_steps": 8,
        "captured_decode_steps": 8,
        "capture_after_warmup": True,
        "consecutive_steps": True,  # each selected window is consecutive
        "preserves_mixed_steps": True,
        "warmup_step_cutoff": 99,
        "routed_layer_order": [12, 13, 14],  # all routed layers, abbreviated
        "checkpoint_revision": "immutable checkpoint revision",
        "checkpoint_sha256": "<64 hexadecimal characters>",
        "prompt_set_sha256": "<64 hexadecimal characters>",
    },
    "records": [
        {
            "hidden": bf16_tensor,       # [tokens, 2048]
            "ids": int32_tensor,         # [tokens, 8], logical expert IDs
            "weights": fp32_tensor,      # [tokens, 8], router weights
            "layer": 12,                 # original MoE layer ID
            "phase": "prefill",          # prefill, decode, or mixed
            "concurrency": 8,
            "step": 100,                 # required source step identifier
            "capture_window": "prefill", # prefill or decode; optional for old corpora
        },
        # Keep all routed layers for each captured step in their source order.
    ],
}
```

Every concurrency must include complete records for each captured step, in
`routed_layer_order`, with one phase label (`prefill`, `decode`, or `mixed`)
shared by all routed layers in that step. Capture eight consecutive prefill
steps and eight consecutive decode steps after `warmup_step_cutoff`; the two
windows may be separated by a gap. A mixed step may be assigned to either
window, and its phase remains `mixed`. New captures should set the same
`capture_window` value (`prefill` or `decode`) on every layer record for each
step. The validator checks that the selected window agrees with the step phase.
If scheduling interrupts an incomplete window, capture starts a new window
at the next eligible step. Excluded provisional step files remain on disk;
the corpus includes only the completed windows.
For older corpora without `capture_window`, it derives both windows from phase;
a mixed step then counts in both. It reports whether the union of retained step
IDs is consecutive, but that is informational. Geometry must match 256
experts, hidden size 2048, intermediate size 512, and top-8 routing. The
revision and SHA256 values are declared provenance; hashes of the supplied
corpus and layer-weight file are computed independently.

Save raw weights separately as a layer mapping. Tensors use the existing
`FP8_W8A16` layout: `w1` has shape `[experts, 2 * intermediate, hidden]`, `w2`
has shape `[experts, hidden, intermediate]`, and scales have shape
`[experts, output_blocks, input_blocks]`. With the target geometry and 128 by
128 blocks, shapes are `w1=[E,1024,2048]`, `w2=[E,2048,512]`,
`s1=[E,8,16]`, and `s2=[E,16,4]`.
Use the default ZIP-based `torch.save` format. The harness memory-maps the
archive while installing separate, verified node-local packed copies.

```python
{
    "layers": {
        12: {"w1": fp8_tensor, "w2": fp8_tensor,
             "s1": fp32_tensor, "s2": fp32_tensor},
        # Include every MoE layer referenced by the records.
    }
}
```

Set up the repository environment with `uv` and use the repository tcmalloc and
libiomp preload for every run:

```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$PWD/.venv/lib/libiomp5.so"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
```

Freeze the corpus and checkpoint mapping before comparing policies:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/benchmark_cpu_numa_moe.py \
  --corpus /data/qwen36_replay.pt \
  --weights /data/qwen36_moe_weights.pt \
  --freeze-manifest /data/qwen36_replay.freeze.json
```

The manifest records file hashes, tensor-content hashes, layer order, and
primary workload metadata. A comparison refuses to run when the corpus or
weights no longer match the frozen hashes.
The coordinator pins its own affinity to the requested CPUs before loading
weights or records. For an additional launch-level constraint, use
`taskset --cpu-list 32-127` before the `.venv/bin/python` command; the result
records the effective affinity seen by the coordinator.
`--controller-cpu 32` pins the caller after creating the teams and keeps all
execution on nodes 1, 2, and 3. The caller runs grouping and orchestration;
the persistent expert teams still contain 32 physical threads per node.

Before a timing run, `--check-only` can validate the runtime, numerical output,
aliasing, routing counts, isolated existing-op workers, and page placement. It
writes provenance and correctness output, then exits without warmup or timing:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/benchmark_cpu_numa_moe.py \
  --corpus /data/qwen36_replay.pt \
  --weights /data/qwen36_moe_weights.pt \
  --manifest /data/qwen36_replay.freeze.json \
  --check-only \
  --output /data/qwen36_replay_correctness.json
```

## Replay and synthetic diagnostics

Run all native policies and existing-operator controls on the frozen replay:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/benchmark_cpu_numa_moe.py \
  --corpus /data/qwen36_replay.pt \
  --weights /data/qwen36_moe_weights.pt \
  --manifest /data/qwen36_replay.freeze.json \
  --controller-cpu 32 \
  --output /data/qwen36_replay_results.json
```

By default, the runtime uses physical CPUs `32-127` as three 32-thread teams
on NUMA nodes `1,2,3`, calibrates every layer with the fixed row table
`[1,4,5,8,16,31,32,33,63,64,65,128,256,512,1024,2048]`, and evaluates the four
replica policies, three experimental two-node TP2 kernel controls, and the
unequal intermediate-slice control. It also runs isolated existing-operator
controls on each individual node and a 96-thread unsharded control. The
three node pairs additionally run the existing CPU MoE operator in two
processes with matching intermediate slices and the native CPU SHM reduction.
These controls fail if the communicator falls back to Gloo reduction.
They measure supported TP2 MoE execution rather than full-model serving.
The single-node controls place one verified weight copy on the measured node. The
96-thread control also uses one verified weight copy, placed on node 1, to
isolate the effect of additional threads. Control workers use active OpenMP
waiting with a 5 ms block time and TPAUSE disabled. Every variant gets one untimed warm
replay, then five paired timing rounds with rotated order. It extends to ten
rounds if paired speedup variation exceeds 5%. `--no-calibration` is valid only
when neither `balanced` nor `split` is selected.

Calibration measures one expert group and a batch of up to 64 groups per node
at each row count. It evicts their weights and scales before each untimed sample.
The scheduler charges the fitted pipeline overhead once per active node, plus
the per-group costs. Split trials run after whole-expert assignment and require
at least a 5% reduction in the predicted complete queue makespan. Reports retain
both calibration densities, single-group fit residuals, and prediction errors
on the replay's mixed queues. These predictions cover inner node execution.
Complete-call timings also include launch, allocation, and final combine.
Full-width calls above 128 tokens gather inputs once into node-local scratch.
Smaller calls also stage inputs when every physical expert group has more than
four rows; other calls gather directly into the existing per-thread GEMM input
buffer. Dispatch records explicit staging, while direct gathering remains in
the expert phase. Arithmetic and top-k reduction order are preserved. Small
queues with one 32-row block per expert use column scheduling; larger queues
also use whole-expert column scheduling when their inputs are staged, they have
at most eight experts, each expert has at most five 32-row blocks, and group
and column counts permit partitioning across the team. Other queues retain
rectangular scheduling. Down projections reuse additional
unpacked columns only within the existing scratch and cache capacity.
Small and large calls retain separate scratch
buffers, with huge pages requested only for large full-width scratch.
Scheduling and packing traverse active experts, clearing the previous call
and statistics buffer's active entries before reuse.

Full-width node-local groups can share staged inputs when all 2–8 groups have
the same ordered token list and more than four rows. Input offsets reference
the shared copy; intermediate and weighted-result offsets remain separate
for every routed slot. Equal counts alone do not enable sharing. Metadata resets
between calls, and calibration and sharded controls retain separate inputs.
`dispatch_bytes` models the shared input rows plus all routed-slot weights;
direct gathering retains the original slot-based estimate. These are traffic
estimates, not measured memory bandwidth.

Execution captures counters and phase times before returning. The statistics
operation materializes diagnostic tensors from that latest-call snapshot and
returns independent copies. Complete-call timing includes grouping, input
movement, expert execution, and combine; diagnostic tensor materialization is
outside that timer.

Synthetic diagnostics default to token counts `1, 8, 32, 128, 512, 2048` and
uniform, Zipf-skewed, hot-eight, and changing-hot-eight routing. The
`--synthetic-token-counts` option accepts a comma-separated list of positive,
unique token counts without changing the default. A broader diagnostic grid for
small, boundary, and large calls is
`1,2,3,4,5,6,7,8,32,128,129,256,512,1024,2048,4096`. Changing-hot-eight
routing uses two ordered records per token count and changes the hot expert
group between them.

`--timing-repeats` defaults to 1. It repeats the entire ordered record sequence
within each measured round, including controls, rather than timing isolated
calls repeatedly. Each variant receives one untimed full-sequence warmup.
Throughput and work counters include all repeated calls. The synthetic
`synthetic_complete_call` report gives each `(M, routing)` cell per-call
median, p95, and mean latency across individual calls plus per-round means.
Each policy's speedup uses the fastest numerically valid measured single-node
control among nodes 1, 2, and 3 for that cell. The per-policy and cell statuses
show whether a valid policy reaches at least `2x`. The cell status is the union
of passing policies, not evidence that one selected policy passes every cell.
These are supplemental diagnostics only, and synthetic results remain outside
the primary promotion gate.

For example, time the broad grid with ten sequence repetitions per round:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/benchmark_cpu_numa_moe.py \
  --mode synthetic \
  --weights /data/qwen36_moe_weights.pt \
  --layer-id 12 \
  --synthetic-token-counts 1,2,3,4,5,6,7,8,32,128,129,256,512,1024,2048,4096 \
  --timing-repeats 10 \
  --policies static balanced split rows \
  --controls single_node_node1 single_node_node2 single_node_node3 \
  --output /data/qwen36_synthetic_results.json
```

## Results and promotion boundary

The result records the source revision and diff hash, extension path and hash,
native source hashes, registered schemas, checkpoint/corpus hashes, topology,
the coordinator's effective process affinity, preload, operator phase timings,
node completion times, route counts, physical placement counts, split groups,
padded rows, estimated dispatch/result bytes, weight-page counts, replica
storage addresses, scratch/shard page counts, and worker CPU placement.
Complete-call wall time is authoritative;
operator `phase_ms` and `node_ms` are summarized from each latest-call stats
snapshot. Routing correctness checks compare native logical counts with the
input assignments. For full-width policies, physical counts must sum to each
logical count; the TP2 and unequal-slice controls report partial intermediate
work and reduce local BF16 outputs.
The numerical reference uses dequantized block-FP8 weights and the original
top-k reduction order; output/input aliasing is also checked.
Placement passes only when every native weight allocation reports local pages
on all three nodes and the three replicas have distinct storage addresses.
Baseline controls report the local-page checks for their copied weights.
Each control also records every live thread's affinity and rejects threads
outside its assigned physical CPU team.

Paired speedups and bootstrap 95% confidence intervals compare dynamic
whole-group or split policies with fixed expert ownership. The report also
compares each primary concurrency against the fastest single-node run and
checks the strongest measured control, including `256/128/128` slicing.
Failed gates report the unmet criteria, largest reported native phase,
imbalance, split frequency, and a retain/reject decision. Phase timings guide
the bottleneck investigation; byte estimates do not establish memory bandwidth.
`tp2_01`, `tp2_02`, and `tp2_12` measure two-node intermediate-sharding kernel
controls; they do not establish native model TP2 serving. Synthetic results
and replays without the exact primary workload metadata cannot pass the
serving promotion gate. The synthetic per-cell `2x` statuses do not change
that requirement.

The primary metadata is an explicit corpus assertion. The harness can verify
observable geometry, complete per-step layer order, record phases and
concurrency, warmup-cutoff ordering, and numerical outputs. It cannot prove
that prompts were actually used, that warmup occurred, or that the supplied
checkpoint revision/hash matches a remote model identifier.

Consecutive corpus replay is the promotion timing condition. Before each
variant warmup and measured round, the coordinator waits 10 ms without timing.
Native and OpenMP workers can spin for up to 5 ms after work; the quiet interval
lets those tails end before the next variant's first, potentially tiny call.
Supplemental isolated diagnostics choose one record per phase/concurrency, warm
it once, then measure five warm-layer and five cold-weight calls per variant
(`--diagnostic-repeats` changes this count). Cold calls flush the packed weights
and scales outside the timer; other data and allocator state remain warm. TP2
cold diagnostics synchronize ranks after their local flushes. The report
retains these results separately from paired replay rounds.

Collect hardware counters in a separate replay invocation:

```bash
uv run --no-sync .venv/bin/python -P benchmarks/kernels/cpu/benchmark_cpu_numa_moe.py \
  --corpus /data/qwen36_replay.pt --weights /data/qwen36_moe_weights.pt \
  --manifest /data/qwen36_replay.freeze.json \
  --rounds 1 --max-rounds 1 --perf-output-dir /data/numa_run/counters \
  --progress-jsonl /data/numa_run/counters_progress.jsonl \
  --output /data/numa_run/counters.json
```

Each variant gets its own disabled `perf stat` collector after warmup. Control
FIFO acknowledgements bracket its full corpus replay; initialization,
calibration, numerical checks, warmup, and isolated diagnostics are excluded.
Raw CSV counters and enabled wall time are retained. The default events measure
cycles, instructions, LLC references, LLC misses, and memory-controller CAS
read/write traffic on the requested physical CPUs. Counts include replay
bookkeeping and any background activity on these CPUs and controllers.
The report derives read/write MiB/s from measured traffic and the enabled wall
interval. `--perf-events` selects other supported events. Memory-controller events
require a successful PMU probe and explicit socket attribution; they do not
automatically provide per-NUMA-node bandwidth.

Instrumented runs cannot pass promotion. Dispatch and result byte counts are
estimates, not measured memory bandwidth. Full-model TP2 and serving throughput
remain outside this milestone. Promotion also requires the specified
96 physical CPUs, three 32-thread teams, nodes 1/2/3, and the default numerical
tolerance or a stricter one.

The [broad tuning results](numa_moe_tuning_20261011.md) record the retained
implementation, correctness checks, and the remaining 2× target misses.

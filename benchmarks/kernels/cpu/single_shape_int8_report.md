# Full-K CPU INT8 single-shape experiment

## Current result: bounded 200-TOPS investigation (2026-10-30)

**200 cold full-operator TOPS was not reached.** The retained partial improvement
measures **2.534940 ms / 162.65 cold TOPS**, compared with **2.564961 ms / 160.75
TOPS** for the contemporaneous committed control. The paired geometric cold
latency change is **−1.099%**, with a process-bootstrap 95% interval of
**[−1.279%, −0.905%]**. It passes the preselected partial-improvement gate:
aggregate candidate median at least 1% lower than baseline, and paired upper
95% bound below zero. None of the five candidate process medians meets the
200-TOPS threshold of **2.06158430208 ms**.

The source baseline is commit `2782ac44fa79edc1b824eba1d7429fddd50af97b`.
Acceptance still measures the complete fused operator, including allocation,
unchanged row quantization, compensation, scaling, BF16 conversion, streaming
stores and fence. The protocol uses physical CPUs 96–191 on NUMA node 1,
96 actual workers, explicit Intel OpenMP/tcmalloc preload, performance governor
and EPP verified before and after, 33 packed-weight banks with verified 100%
huge-page backing, and more than 1.5 GiB of intervening packed weights.
Tests, builds and model checks use node 0. No outliers or samples were excluded.

### Retained implementation

Each worker has two aligned 4 KiB INT32 result buffers. While computing the
next 32×32 tile's first eight K64 steps, it converts and streams four rows of
the previous result after each step. Compensation and channel scale/bias
vectors are loaded once for that previous tile. The current tile accumulates
all 4096 K elements before its four INT32 stores. The final pending tile is
drained before tile release and the existing output fence.

The change retains the public N32 weight format, 4160-byte activation rows,
fourfold K64 unrolling, ordinary operand tile loads, 256×2048 exclusive worker
ownership and existing window traversal. Quantization, arithmetic order,
Linux tile permission, backend-state reset, exact shape/96-actual-worker guard,
opt-in interface, defaults and fallback behavior are unchanged. No serial
K blocking, software prefetch, quantization rewrite, heap workspace or
production tuning controls were retained.

Generated-code inspection confirms one first tile, 511 tiles with pending
epilogues, and one final drain per worker. Per output microtile, AMX work
remains 128 A loads, 128 B loads, 256 INT8 dots and four final INT32 stores.
There are no intermediate accumulator stores/reloads or ZMM/YMM stack spills
in the inspected functions. The compiler naturally inlines the new tile
helper; the baseline had a helper call and 4 KiB frame setup per microtile.
The selected no-bias GEMM symbol is 4713 bytes, versus 2013 bytes for the
serial-pending comparison. Register assignment, loop splitting and four-row
epilogue unrolling also change. Timing therefore establishes the benefit of
the selected implementation, rather than isolating hardware overlap alone.

### Final qualification

Five alternating candidate/current-baseline fresh-process pairs each contain
five cold and five warm samples, with a frozen legacy process interleaved in
every pair. Warm and downstream-consumption diagnostics do not determine
standalone cold acceptance.

| Pair | Candidate cold ms | Baseline cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.534940 | 2.562784 | 3.199805 |
| 2 | 2.529366 | 2.565054 | 3.279526 |
| 3 | 2.532892 | 2.564961 | 3.212580 |
| 4 | 2.541543 | 2.567402 | 3.242023 |
| 5 | 2.535779 | 2.555097 | 3.239673 |

Candidate process-median SD is **0.004460 ms**; process medians span
2.529366–2.541543 ms. Individual cold samples span 2.525000–2.562013 ms.
Aggregate warm medians are **2.270194 ms** for candidate and **2.297118 ms**
for baseline, a **1.172% decrease** in their ratio. An earlier independent
five-pair cohort also passed the partial gate: 2.536884 ms, paired
−0.977% [−1.310%, −0.679%]. Its statistics are retained separately.

### Bounded experiments and selection

**43 of 48 allowed timed optimization variants** were screened, each using five
cold samples and full-output frozen-legacy verification before timing.
The archive's `screening_summary.json` contains every variant's raw cold/warm
samples and binary hash; all negative results remain available. These screening
medians are conditional comparisons, not independent additive effects or
familywise confidence estimates. Within 1%, selection prefers fewer
mechanisms, then less scratch, then no prefetch.

| Family / representative | Variants | Cold ms | Warm ms | Decision |
| --- | ---: | ---: | ---: | --- |
| Original serial-K macroblock matrix | 23 | Best 2.612598 | 2.425328 | All slower than committed baseline |
| K2048 follow-up | 3 | Best 2.735630 | 2.520185 | Rejected |
| Compact C stride, best K1024 shape | 2 | 2.609670 | 2.448302 | Rejected |
| Cached epilogue, best K1024 shape | 2 | 2.589318 | 2.441799 | Rejected |
| Interleaved epilogue, best K1024 shape | 2 | 2.543644 | 2.351199 | Within 1%; more scratch than full-K |
| Sparse T1/T0 prefetch on K-overlap | 3 | Best 2.532989 | 2.353817 | Within 1%; more mechanisms/scratch |
| Full-K cached epilogue | 1 | 2.573393 | 2.304968 | Repeated cold comparison regressed |
| Full-K interleaved epilogue | 1 | 2.533964 | 2.269488 | Selected for final rebuild |
| Full-K helper forced inline | 1 | 2.562271 | 2.309524 | No material screening benefit |
| Two buffers, serial pending epilogue | 1 | 2.583161 | 2.310181 | Rejected |
| Explicit rounding / two-way quantization unroll | 2 | Best 2.572696 | 2.294951 | Rejected |
| Overlap plus those quantization changes | 2 | Best 2.529888 | 2.260226 | Within 1%; omit quantization change |

Serial-K experiments span M macroblocks 128/256, N macroblocks
128/256/512/1024 and K blocks 256/512/1024, excluding a footprint above the
1792 KiB screening budget; three K2048 shapes were added. They keep the
existing ownership and packing, load/store partial INT32 sums privately,
and compensate/scale only after full K. C stride is N-macroblock plus 16
INT32 elements except in the compact-stride ablation. Base workspace ranges
from 6.75 MiB to 97.5 MiB. The best K-overlap needs 13.5 MiB plus alignment
reserve, versus 8 KiB of result buffers per worker in the selected full-K
implementation. Sparse prefetch tests issue eight evenly spaced cache lines
for valid future N32/K64 panels, with T1 lookahead 128/256 or T0 lookahead 256.

The full-K cached-epilogue variant also has an archived five-pair comparison:
paired cold change **+0.252% [+0.039%, +0.497%]**. It was not retained.
The serial-pending ablation holds buffering and channel-vector caching but
changes generated loop layout/unrolling along with epilogue placement.
No causal claim rests on subtracting these screening medians.

### Roofline and phase analysis

The socket-0 side experiment's 216–242 TOPS bandwidth estimate is arithmetically
consistent with its assumed 954 MB of cold cache-response traffic and
500–560 GB/s service rate. That traffic includes quantization-associated reads,
but the estimate does not model quantization execution/dependencies, allocation,
compensation, BF16 output, barriers or AMX setup/release. Its shared-operand probe
uses about 67.4 MB of warm operands in 71.3 MB of mapped backing,
runtime K-tail checks, `std::thread`, another
socket and different observed clocks. Sequential LLC/DRAM bandwidth and
dependent-load latency measurements characterize those access patterns;
they do not identify an LLC-slice bottleneck or predict cold operator timing.
At 200 TOPS, that assumed traffic alone takes 1.908 ms at 500 GB/s or
1.704 ms at 560 GB/s, leaving 153–358 µs for other work.

Source-matched reduced diagnostics provide a closer comparison:

| Timed work | Cold ms | Warm ms |
| --- | ---: | ---: |
| Integer AMX loop, quantization prepared outside timing | 1.865904 | 1.527101 |
| Integer loop plus real output epilogue/allocation/stores/fence | 2.038841 | 1.710131 |

These are diagnostic builds, never acceptance builds. The second row is not
an epilogue-only measurement. Subtracting the rows is an unpaired marginal
difference that includes allocation, output stores and changed generated code.
The reduced integer loop still stores INT32 results and reads a scalar from each tile;
the real-epilogue variant uses the production worker and timed output
allocation/free. Their cache preparation differs from the complete operator.

Separately instrumented baseline critical-path intervals reconstruct a median
**2.559147 ms** cold call, with **0.520342 ms** outside the critical GEMM path
and **2.040887 ms** on that GEMM path. Last quantization completion is about
0.505558 ms; worker finish spread is about 0.174436 ms. Per-call reconstruction
sums correctly. Independent worker-phase maxima or medians must not be added:
they can belong to different workers/calls and overcount the critical path.
Instrumentation is excluded from qualification.

Holding that non-GEMM cost fixed would leave only **1.541242 ms** for GEMM at
200 TOPS, requiring roughly 24.5% less GEMM time. This is an illustrative
budget, not an attainable-performance prediction. Quantization variants in
this bounded iteration failed to improve cold operator latency materially.

For the best serial-K overlap, separate whole-operator counters showed
cache-response traffic **1009.75 versus 937.63 MB/call**, and L2-miss stall
cycles **201.49 versus 187.64 million/call**. Core-group total cycles were
essentially unchanged; the candidate's effective clock was about 0.84% higher.
Its small timing gain does not establish reduced cache delivery work.
Under fixed ownership, the logical B payload participation is about 805 MB
per call. This is not a physical LLC/DRAM traffic floor: cache residency,
request merging and shared sources affect hardware events.

The previous source-matched BRGEMM inspection remains below. Its equal AMX
operand counts, streaming B-load hint, scratch prefetch, dispatch/allocation
costs and larger generated bodies remain relevant; this iteration does not
remeasure or add those earlier effects to the new overlap gain.

### Selected implementation counters and output consumption

Nine counter groups ran separately for candidate and committed baseline,
with measurement-window FIFO controls, five cold samples per process and
100% event scheduling coverage. Counts below are aggregate counts divided by
timed calls; MB means decimal MB. Line-event traffic is not distinct data.
Fractions use each group's own measured cycles and may overlap.

| Metric / group | Candidate per call | Baseline per call |
| --- | ---: | ---: |
| Execution cycles / core | 727.675 million | 735.718 million |
| AMX-busy cycles / core | 201.363 million (27.67%) | 201.360 million (27.37%) |
| Load-bound cycles / core | 505.068 million (69.41%) | 518.820 million (70.52%) |
| Active clock from cycles/ref-cycles × 2.4 GHz / core | 2.9537 GHz | 2.9529 GHz |
| L2-miss stall cycles / stalls | 194.233 million (26.68%) | 186.577 million (25.46%) |
| L3-miss stall cycles / stalls | 9.339 million (1.28%) | 7.599 million (1.04%) |
| L1-miss stall cycles / l1 | 424.439 million (58.33%) | 425.059 million (57.81%) |
| L1 replacement line events / l1 | 213.202 million | 213.516 million |
| L2 fills / cache | 1011.174 MB | 1005.015 MB |
| Socket-cache response reads / cache | 949.262 MB | 942.984 MB |
| Local-DRAM response reads / cache | 53.414 MB | 53.489 MB |
| L2 demand hit events / l2 | 145.822 million | 137.735 million |
| L2 demand miss events / l2 | 1.133 million | 1.083 million |
| L2 RFO hit events / l2 | 3.277 million | 2.372 million |
| L2 RFO miss events / l2 | 168,953 | 168,517 |
| DTLB load-walk cycles / dtlb_load | 94,225 (0.013%) | 116,370 (0.016%) |
| DTLB store-walk cycles / dtlb_store | 94,514 (0.013%) | 108,350 (0.015%) |
| Frontend zero-delivery cycles / frontend | 1.361 million (0.187%) | 1.420 million (0.192%) |
| Socket IMC reads / memory | 64.256 MB | 63.758 MB |
| Socket IMC writes / memory | 105.142 MB | 104.812 MB |

Absolute load-bound cycles fall **2.651%**, with core-group active-clock difference
only about 0.026%. The narrower statement that this measured load-bound
counter decreased is supported. Cache-response bytes do not fall, and
L2/L3-miss stall cycles increase. L1 stall fraction increases even though
absolute L1 stall cycles decrease slightly. These events do not establish
reduced operand traffic, a general reduction in every load-stall counter,
or LLC-slice contention. AMX-busy cycles are essentially unchanged.
Page-walk and frontend counters remain small. IMC traffic includes socket
background activity and uses perf's scaled units.

An independent diagnostic immediately sums all 50,331,648 BF16 output
elements into FP32 after each operator call. It retains both streaming stores
and their fence and runs without perf. Candidate versus baseline medians are
**3.185526 versus 3.219791 ms cold**, and **2.964547 versus 2.988933 ms warm**.
This single-process diagnostic shows no regression for that consumer, but
does not characterize every downstream operator or enter acceptance timing.

### Correctness and validation

Every screened variant and the rebuilt selected binary match all
50,331,648 output elements exactly against seeded frozen legacy, both with
bias and without bias, with unequal channel scales and unchanged inputs.
The nearest scaled-MM suite passed **26 tests**, with two existing Swig
deprecation warnings, in 301.01 seconds. Its fresh process verifies the
selected extension SHA/path and both preloads without pre-initializing AMX.
Coverage includes actual-team fallback, mismatched shapes, strided BF16 input,
quantization boundaries/zero rows, unequal scales, bias/no-bias, first direct
execution and direct/BRGEMM alternation. No new tests or helpers were needed.

The existing offline cached RedHatAI Llama-3.1-8B W8A8 generation comparison
passed: legacy and `single_shape` produce identical greedy token IDs for
two prompts, 32 tokens each. Both processes use the selected extension.
With six actual workers and model shapes outside the specialization, this
checks serving fallback only. All test/model children and measurement
drivers completed successfully and were reaped.

Applicable pre-commit hooks pass on the changed C++ source, rendered CPU
documentation and this report; `git diff --check` also passes.

### Reproduction and provenance

The persistent archive is
`/home/tianmuli/vllm_test/snc_scaling_analysis/single_shape_200_2782ac44fa_20261030`.
Its `ARCHIVE.md` documents host dependencies, path substitution and replay.
It contains all 43 variant sources/build metadata/binaries, frozen controls
and reference tensors, raw qualification samples, phase and counter runs,
bounded disassembly, side-roofline sources/results, validation and check logs.
All 44 external link inputs are archived by SHA256. Temporary tuning and
diagnostic controls remain outside production.

| Artifact | SHA256 |
| --- | --- |
| Baseline GEMM source | `065df2fd698383aaa7c93e7fd8eb2b33a246ef13be8ea91878f0b912138f1a67` |
| Selected GEMM source | `a9b0d9909462a2af203327aa80ae65a06ee7e3f1af49d78a66828d8b6970805e` |
| Frozen committed extension | `d61b9abb2b1ca7844b0a9c22096d4038af04786180d88af75ad242382a1b5a34` |
| Rebuilt selected extension | `a1888f1c9e406212164a03f4b8657ebcf6365f6f66f25c826c2f63188b6eba67` |
| Frozen legacy extension | `0988f8960179ecdcbc5d17e866d524ccd532b2edb4842e5121021dbbfe778a1d` |

## Previous result: minimize the b23 implementation (2026-10-30)

The simplified implementation qualifies at **2.561864 ms / 160.94 cold TOPS**.
All five fresh-process medians are below **2.57698038 ms**. Relative to the
contemporaneous frozen `b23d40942152c10bdcfb4931408ed667f14763bf` control,
the paired geometric latency change is **+0.274%**, with a process-bootstrap
95% interval of **[+0.129%, +0.416%]**. This meets the preselected 1%
noninferiority limit. The purpose of this iteration is fewer special-case
changes while retaining 160 TOPS; it does not establish a new speed record.

Removed: 2 MiB activation-pointer alignment and its **2 MiB minus one byte**
allocation reserve, custom quantization ownership, N32-only window grouping,
alternate-worker reversal, worker-hoisted accumulator scratch, and explicit
`noinline`. The implementation reuses `execute.quantize` with contiguous
balanced row ownership and its dependency barrier, traverses forward 128×64
windows, and allocates aligned 4 KiB accumulator scratch in the tile helper.
The compiler still outlines the selected helper. The inherited half-partition
start offset remains.

Retained: exact shape and 96-actual-worker guard, full-K 32×32 direct AMX,
4160-byte activation rows, fourfold K64 unrolling, interleaved ordinary A/B
loads, existing output arithmetic and streaming stores/fence, Linux tile
permission and backend-state handling. The public packing, opt-in interface,
defaults and fallback behavior are unchanged. Padded rows add 256 KiB; no
software prefetch or production tuning controls are introduced.

For each complete 256×64 window, useful operand bytes including compensation
are approximately **1280.25 KiB**; the padded A address span plus B is
**1296.25 KiB**. An inner 128×64 window has **768.25 KiB** of useful operands
and a **776.25 KiB** span. Padding holes are not touched by quantization or
AMX loads. Neither address span nor useful bytes establish cache residency;
scratch, scales, output and other live data are additional.

### Selection and qualification

Screening used five cold samples per candidate, followed by five alternating
candidate/b23 fresh-process pairs with five cold samples per process.
The final cleaned-source qualification additionally interleaves frozen legacy
controls. Every candidate process median must pass
the absolute target, and the upper 95% paired process-bootstrap regression
must be at most 1%. Within that limit, prefer BRGEMM, then fewer special-case
changes, then less scratch. Warm timing and immediate output consumption are
reported separately. Effects below are conditional, not additive; there is no
claim of a global minimum or familywise statistical confidence.

All 96 measurement CPUs used the performance governor and performance EPP,
verified before and after. Placement, preloads, allocation, quantization,
packing, compensation, scaling and output timing follow the original protocol
below. Packed banks have verified huge-page backing on node 1 and more than
1.5 GiB of intervening packed weights. No samples or outliers were excluded.

| Mode | Cold ms | Cold TOPS | Warm ms | Process-median SD ms |
| --- | ---: | ---: | ---: | ---: |
| Simplified candidate | 2.561864 | 160.94 | 2.308637 | 0.003074 |
| Frozen b23 | 2.555837 | 161.32 | 2.294037 | 0.002761 |
| Frozen legacy | 3.218129 | 128.12 | 2.445057 | 0.023759 |

| Pair | Candidate ms | b23 ms | Legacy ms |
| --- | ---: | ---: | ---: |
| 1 | 2.565623 | 2.555837 | 3.207435 |
| 2 | 2.558558 | 2.557202 | 3.240898 |
| 3 | 2.560817 | 2.558679 | 3.218129 |
| 4 | 2.565543 | 2.554186 | 3.256527 |
| 5 | 2.561864 | 2.551544 | 3.199371 |

Candidate individual cold samples span 2.555715–2.596643 ms; acceptance applies
to each process median. Paired geometric warm regression is **+0.363%**;
the ratio of aggregate warm medians is +0.636%, a different statistic.

### Ablations and interactions

The following labels identify immutable source snapshots in the archive.
`no_quant` changes row ownership; quantization is always performed.
`no_align_same_size` isolates pointer placement with allocation size retained;
`no_align` removes both placement and allocation reserve. N64 grouping still
uses N32 microkernels. `brgemm128` changes backend, M aggregation, scratch size
and call cadence together; `brgemm32` preserves M32. `old_baseline` restores
the d80 GEMM source within the otherwise b23 build, rather than rebuilding the
entire d80 repository. The `simplify_layout_quant` combination removes alignment/reserve and uses
contiguous ownership. Its `_scratch` variant also restores default inlining
and helper-local scratch; `_scratch_helper` additionally reuses
`execute.quantize`. `simplify_window` adds N64 grouping, `simplify_forward`
also removes reversal, and `simplify_grouped` changes its operand load order.
`ref_stream_b` changes only the B tile-load hint in the rebuilt reference.
All intervals compare each variant with its own
contemporaneous b23 pair. A byte-identical rebuilt reference differs by
+0.246%, cautioning against interpreting tiny effects as uniquely causal.

| Variant | Cold ms | Warm ms | Cold change % | Paired 95% CI % | Passing processes |
| --- | ---: | ---: | ---: | --- | ---: |
| brgemm128_no_quant | 2.688480 | 2.339309 | +5.095 | [+4.842, +5.437] | 0/5 |
| brgemm128 | 2.677230 | 2.340275 | +4.807 | [+4.556, +5.030] | 0/5 |
| no_align_same_size | 2.554358 | 2.300349 | -0.032 | [-0.240, +0.148] | 5/5 |
| ref | 2.559438 | 2.299045 | +0.246 | [+0.033, +0.440] | 5/5 |
| no_align | 2.553168 | 2.295524 | +0.116 | [+0.058, +0.183] | 5/5 |
| window64 | 2.543800 | 2.300545 | -0.296 | [-0.475, -0.067] | 5/5 |
| local_scratch | 2.550437 | 2.302095 | -0.012 | [-0.239, +0.220] | 5/5 |
| default_inline | 2.566621 | 2.306917 | +0.341 | [-0.035, +0.594] | 5/5 |
| brgemm128_no_pad_no_align | 2.714568 | 2.443006 | +6.240 | [+5.854, +6.840] | 0/5 |
| no_pad | 2.684446 | 2.449127 | +4.751 | [+4.463, +5.134] | 0/5 |
| grouped_loads | 2.563292 | 2.294479 | +0.334 | [+0.030, +0.631] | 5/5 |
| unroll1 | 2.591720 | 2.312645 | +1.223 | [+1.166, +1.283] | 0/5 |
| window64_no_reverse | 2.566736 | 2.299751 | +0.466 | [+0.369, +0.590] | 5/5 |
| no_pad_no_align | 2.674789 | 2.439826 | +4.770 | [+4.591, +4.949] | 0/5 |
| no_quant | 2.551662 | 2.294061 | -0.034 | [-0.231, +0.162] | 5/5 |
| brgemm32 | 2.651416 | 2.357402 | +3.601 | [+3.344, +3.828] | 0/5 |
| default_inline_local_scratch | 2.552969 | 2.303789 | -0.085 | [-0.196, +0.050] | 5/5 |
| old_baseline | 2.793904 | 2.503050 | +9.161 | [+9.021, +9.374] | 0/5 |
| no_reverse | 2.574824 | 2.299543 | +0.457 | [+0.202, +0.656] | 3/5 |
| brgemm128_window64 | 2.691499 | 2.348095 | +5.607 | [+5.354, +5.867] | 0/5 |
| simplify_window | 2.548943 | 2.312054 | -0.380 | [-0.648, -0.066] | 5/5 |
| simplify_layout_quant | 2.553499 | 2.323142 | -0.160 | [-0.291, -0.055] | 5/5 |
| simplify_grouped | 2.590754 | 2.325801 | +1.481 | [+1.266, +1.724] | 0/5 |
| simplify_layout_quant_scratch_helper | 2.555333 | 2.306514 | -0.002 | [-0.099, +0.098] | 5/5 |
| ref_stream_b | 2.619114 | 2.347263 | +2.582 | [+2.249, +2.816] | 0/5 |
| simplify_layout_quant_scratch | 2.557557 | 2.315841 | +0.450 | [-0.057, +1.096] | 4/5 |
| simplify_forward | 2.567273 | 2.304024 | +0.631 | [+0.454, +0.850] | 5/5 |
| forward_no_pad | 2.692477 | 2.443894 | +5.597 | [+5.404, +5.791] | 0/5 |
| forward_brgemm128 | 2.768658 | 2.353647 | +8.357 | [+8.079, +8.611] | 0/5 |
| forward_brgemm32 | 2.697851 | 2.370673 | +5.697 | [+5.395, +6.099] | 0/5 |
| forward_unroll1 | 2.618106 | 2.322862 | +2.519 | [+2.408, +2.630] | 0/5 |
| clean_forward | 2.561864 | 2.308637 | +0.274 | [+0.129, +0.416] | 5/5 |

| Interaction | Excess latency % | 95% CI % |
| --- | ---: | --- |
| padding × alignment removal | -0.098 | [-0.493, +0.239] |
| helper scratch × default inlining | -0.413 | [-0.804, +0.037] |
| N64 window × forward traversal | +0.306 | [-0.017, +0.641] |
| BRGEMM128 × contiguous quantization | +0.310 | [-0.094, +0.737] |
| BRGEMM128 × unpadded unaligned A | -3.247 | [-3.734, -2.645] |
| BRGEMM128 × N64 window | +1.063 | [+0.663, +1.468] |

Interaction estimates are joint log regressions minus the individual log
regressions, each normalized to its contemporaneous control; intervals use
20,000 independent process-pair bootstrap draws. The backend interactions
show why individual benefits cannot simply be added. In particular, removing
padding and alignment in BRGEMM costs less than its direct-kernel joint effect, while
BRGEMM plus N64 grouping is worse than an additive prediction.

`simplify_forward` removes all selected policies together. Its combined-candidate
paired result passes at +0.631% [0.454%, 0.850%]. Removing dead alignment
constants and zero-offset expressions produces `clean_forward`; the selected
direct functions are byte-identical, and the exact cleaned source is qualified
again above. Final neighbors retain every other forward-policy choice:
removing padding, replacing direct AMX with BRGEMM M32 or M128, or reducing
unrolling to one fails all five absolute process medians. Grouped loads also
fail in the simplified combination; retain interleaving.

### Generated BRGEMM code and the measured gap

Source-matched captures use the actual public operator and loaded Torch
library, M32/M128, N32, K4096, A stride4160 and public B stride128. All 96
workers emit identical kernels for each descriptor. Full outputs match the
frozen reference. The capture, tile configurations, wrapper disassembly,
commands and hashes are in `jit-brgemm/`.

Per M32 output, both direct and JIT kernels execute **128 A tile loads, 128 B
tile loads, 256 INT8 dot products, and four INT32 accumulator stores**. M128
executes four sequential M32 blocks and reloads B four times, as do four direct
calls. Extra operand work therefore does not explain the gap by itself.

BRGEMM fully unrolls 64 K64 steps; the direct kernel loops over four-step
bodies. The M32 JIT is 4,867 bytes and M128 is 19,249 bytes. BRGEMM uses
ordinary A loads and `TILELOADDT1` B loads; direct uses ordinary loads for both.
BRGEMM issues 64 `prefetchw` instructions per M32 into INT32 accumulator
scratch, not future operands or final BF16 output, and zeros more tile
registers. Nevertheless its static executed instruction count is lower:
684 instructions for M32, or 657.75 per M32 in M128, versus 800 in the direct
K-loop body alone. Instruction-count arguments cannot explain a BRGEMM loss
without considering instruction cost and memory behavior. The M128 JIT fits
nominally inside the 64 KiB L1 instruction cache; size alone does not prove
an instruction-cache bottleneck.

The Torch u8×s8 wrapper allocates and frees a **56-byte std::function capture
on every invocation**, including cache hits, and performs descriptor lookup,
TLS/helper checks and indirect dispatch. This is real unquantified overhead.
M128 reduces calls fourfold but still loses; these measurements do not
quantify how much the remaining wrapper overhead explains. The tile helper compares the configuration and conditionally
loads it when the cached helper changes; generated code does **not** establish
a tile-configuration reload on every steady-state call. The BF16 epilogue is
inlined in both backends.

A controlled direct-kernel experiment changes only the B load hint: exactly
16 instruction bytes across bias/no-bias functions, with all addresses,
branches, registers and scheduler instructions otherwise identical. Streaming
B regresses cold latency by **2.582% [2.249%, 2.816%]** and fails all five
process medians. Separate counters show load-bound cycles increasing from
**519.959 to 539.359 million per call**, and L2-miss stall cycles from
**189.077 to 218.278 million**. AMX-busy cycles stay approximately 201.35
million; effective clocks are 2.958 and 2.939 GHz. This supports cache delivery
as one contributor to the B-hint regression, without identifying an exact
cache eviction mechanism. This isolated effect must not be added numerically
to the BRGEMM gap. Other backend differences remain coupled.

### Separate counter comparisons

Matched forward-policy BRGEMM diagnostics compare each backend with the exact
selected direct binary. Each group uses separate fresh processes; counts are
aggregate events divided by all timed calls. Millions of cycles below are
summed across workers. Fractions overlap and are not additive. Effective GHz
is execution cycles/reference cycles × 2.4. The performance governor does not
fix the measured clock.

| Metric | BRGEMM M32 | Its direct control | BRGEMM M128 | Its direct control |
| --- | ---: | ---: | ---: | ---: |
| Core-group ms/call | 2.729722 | 2.567002 | 2.778399 | 2.557455 |
| Execution cycles/call, million | 699.477 | 734.793 | 710.125 | 731.723 |
| Effective GHz | 2.646 | 2.953 | 2.639 | 2.952 |
| AMX-busy cycles/call, million | 201.423 | 201.362 | 201.387 | 201.360 |
| Load-bound cycles/call, million | 477.003 | 518.400 | 499.330 | 516.616 |
| L2-miss stall cycles/call, million | 193.209 | 187.825 | 213.887 | 188.340 |
| L3-miss stall cycles/call, million | 20.508 | 7.548 | 24.905 | 7.577 |
| L2 fills/call, million lines | 15.777 | 15.680 | 15.749 | 15.628 |
| I-cache data stall cycles/call, million | 1.258 | 0.737 | 1.854 | 0.756 |
| Frontend zero-delivery cycles/call, million | 2.065 | 1.338 | 2.015 | 1.349 |

Both backend runs have lower effective clocks and higher cache-miss stalls,
despite fewer aggregate execution cycles and essentially equal AMX work.
The isolated B-hint regression provides one plausible contributor. These
runs do not identify the clock cause or apportion backend latency. Frontend
counters do not support instruction-cache stalls as the leading explanation.
Whole-operator retired instructions include wrappers, allocator paths and
software waiting, so they are not the static kernel instruction counts above.

The matched unpadded direct variant runs at 2.963 versus 2.952 GHz but raises
load-bound cycles from 518.205 to 556.929 million per call. Separate L1-miss
stalls rise from 423.954 to 471.400 million; L2 demand misses are approximately
1.08 million in both layouts. DTLB load/store walk cycles remain small
(0.127/0.101 million versus 0.115/0.092 million). CPU96 has a 48 KiB,
64-set, 12-way L1 data cache: fixed-offset rows with stride4096 repeat a set,
while stride4160 advances one set per row. The measured L1 behavior is
consistent with that conflict hypothesis, without proving it uniquely.

Reducing K-loop unrolling from four to one leaves AMX-busy cycles unchanged
at 201.36 million and effective clocks close (2.947 versus 2.953 GHz), while
load-bound cycles rise from 517.457 to 532.592 million. Separate L2/L3-miss
stalls rise from 187.772/7.581 to 214.705/19.722 million. Machine-code audits
find no K-accumulator spills, but branch/address setup and bias-helper register
assignment also change. Treat this as a K-unroll code-generation effect,
rather than a pure change in AMX throughput.

The selected candidate versus frozen b23 is repeated in separate core/stall
processes. Absolute load-bound and cache-miss stall cycles fall in both
observations, by small amounts; this does not establish a latency improvement.

| Metric, million cycles/call | Selected | b23 | Selected repeat | b23 repeat |
| --- | ---: | ---: | ---: | ---: |
| Load-bound | 516.865 | 520.052 | 518.948 | 519.509 |
| L2-miss stalls | 186.634 | 188.626 | 186.465 | 190.170 |
| L3-miss stalls | 7.541 | 8.262 | 7.534 | 8.384 |

The first core group measures 70.36% load-bound cycles and 27.41% AMX-busy
cycles; these overlapping fractions are not a speedup budget. L2 fills are
15.625 million lines/call in the core group (approximately 953.7 MiB of event
traffic, not unique operand bytes). Separate L2 demand hits/misses are
137.696/1.093 million requests, RFO hits/misses 2.383/0.170 million, and
DTLB load/store walk cycles 0.121/0.092 million. Socket-wide DRAM counters
measure 60.158 MiB read and 100.673 MiB written per call versus
61.397/100.634 MiB for b23; these include background traffic. All detailed
cache-response, fill, frontend, event coverage and DRAM measurements are in
`diagnostics_comparison.json` and the raw per-group records. No specific LLC
slice-contention mechanism is established. Operand delivery remains a
reasonable target for further experiments, not a predicted speedup.

### Immediate output consumption

A separate diagnostic includes an immediate full-output FP32 sum after every
operator call. Each mode uses one fresh process with five cold and five warm
samples; this is not the five-pair acceptance protocol. Cold medians are
**3.215470 ms selected versus 3.175342 ms b23 (+1.264%)**; warm medians are
**3.000526 versus 2.957296 ms (+1.462%)**. All samples and generated drivers are
preserved in `chain_clean_forward/`. This diagnostic exposes a downstream
tradeoff and includes the sum's own work; it does not isolate the cost of
streaming stores. It neither overrides cold standalone selection nor
qualifies a complete model pipeline. Controller threads remained off node1
until these processes completed.

### Verification and reproducibility

The exact selected source passes **26 tests** in
`tests/kernels/quantization/test_scaled_mm_kernel_selection.py`, including
first-ever AMX execution, actual-team fallback, mismatched shapes, strided
BF16 input, quantization boundaries, unequal scales, bias/no-bias,
direct/BRGEMM alternation and input immutability. Every measured variant also
compares all 50,331,648 output elements against frozen legacy with bias and
without bias before timing. The cached Llama comparison produces identical
32-token greedy outputs for both prompts. Its shapes exercise fallback;
it does not qualify direct-kernel model performance. No new tests were needed. Applicable pre-commit hooks pass on the C++ source,
rendered documentation and this report; `git diff --check` also passes.

Selected GEMM source SHA256:
`065df2fd698383aaa7c93e7fd8eb2b33a246ef13be8ea91878f0b912138f1a67`.
Selected extension SHA256:
`d61b9abb2b1ca7844b0a9c22096d4038af04786180d88af75ad242382a1b5a34`.
Frozen b23 extension:
`843a6df1f09688d2b3c4458dc6a76818ab9e97a71b1d1af310a03f71e27c5e50`.
Loaded `libtorch_cpu.so`:
`cc3d0101c16504ba021abe6794c98122f744ec8cf077bd0178f630f320b24aeb`.

The new archive is
`/home/tianmuli/vllm_test/snc_scaling_analysis/single_shape_ablation_b23d409421_20261030`.
It preserves all screens, paired samples, qualification, counter windows,
output-consuming diagnostics, failures, frozen controls/reference outputs,
source snapshots, extension binaries, normalized compiler/linker commands,
346 source hashes and 44 linked-input hashes, JIT captures, machine-code
audits, test/model logs and replay drivers. Unchanged external build objects
were reused with hashes; this is not claimed as a clean rebuild of Torch or
all dependencies. The early stage-not-ready attempt was rejected before
verification or timing, and is retained. `ARCHIVE.md` documents replay paths
and dependency requirements. The original 2026-10-29 archive remains intact.

## Historical b23 qualification and tuning

Everything below records the original implementation committed as b23.
In these historical tables, “candidate” is b23 and “current” is d80, rather
than the simplified candidate and b23 controls used above.

### Result

**160 cold INT8 TOPS qualified.** The selected change measures **2.554976 ms / 161.38 TOPS**, with **8.40% lower cold latency** than the source-matched committed specialization. All five fresh-process candidate medians are below **2.57698038 ms**, using five cold samples per process and alternating candidate/current order with interleaved legacy controls. Earlier u4 and N32-only candidates failed this strict requirement; their results remain below and in the archive.

Acceptance times the fused operator without bias, including allocation, row quantization, compensation, scaling, BF16 conversion, streaming stores, and the output fence. Bias and no-bias correctness are both verified. Standalone acceptance excludes downstream output reads. Model generation below exercises fallback.

### Selected implementation

The opt-in interface remains `VLLM_CPU_SGL_SCHEDULER=single_shape`. Linux AMX INT8 support and process-wide tile-state permission are checked before direct execution. The exact BF16 M4096/N12288/K4096 shape requires 96 actual workers; other cases preserve `flat_auto` fallback.

The full-K 32×32 u8×s8 kernel uses four INT32 accumulators, two A tiles and two B tiles. Ordinary tile loads interleave independent load/dot operations in four K64 steps per iteration. It preserves public N32 `[K/4,32,4]` VNNI packing (B stride128), subtracts compensation once, and uses the existing epilogue. Backend tile state is reset, configuration is loaded per worker, and tiles are released afterward. Software prefetch is off.

Quantized A uses a private 4160-byte row stride, adding 256 KiB over contiguous rows. Its start is aligned to 2 MiB within a private allocation that reserves 2 MiB minus one byte of additional scratch space; this does not guarantee huge-page backing. Quantization ownership matches the 16×6 GEMM grid: six workers divide each 256-row partition exactly once and synchronize. Each worker exclusively owns 256×2048 output elements and traverses two inner 128×32 windows for each N32 step, reversing alternate workers with the half-partition offset. The complete 256×32 step has a padded operand union of **1168.125 KiB**: 1040 KiB of quantized A and 128.125 KiB of packed B plus compensation. One inner 128×32 window accounts for **648.125 KiB**. These totals exclude scratch, scales, output and other live data; they do not establish cache residency. Aligned 4 KiB accumulator scratch is allocated once per worker and passed to the tile helper, which is marked `noinline`. The four accumulators remain in AMX tiles throughout K. If eligibility holds but the actual team is smaller, fallback uses contiguous quantized rows within the same private allocation.

### Controls and protocol

Base commit: `d80bd2eeb5cb3ae7be0d83d661e723ad51491189`. Original reference: 147.48 TOPS at 2.79567 ms. Hardware: Intel Xeon 6972P, 96 physical cores per socket, 2 MiB L2 per core, 480 MiB LLC per socket. Acceptance uses physical CPUs 96–191 and NUMA node 1. Builds, correctness and model checks use the other socket.

Each fresh process preloads `/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4` and `/home/tianmuli/vllm_test/vllm_local/.venv/lib/libiomp5.so`, sets `KMP_AFFINITY=granularity=fine,proclist=[96-191],explicit`, `OMP_NUM_THREADS=96`, `OMP_THREAD_LIMIT=96`, `OMP_DYNAMIC=FALSE`, and `THP_MEM_ALLOC_ENABLE=1`. Packed banks have verified 100% huge-page backing on node 1: 33 disjoint matrices, 1,662,566,400 total bytes, 1,612,185,600 intervening bytes before revisit (>1.5 GiB). Rotation continues between samples.

Five candidate/current pairs alternate their process order, with a fresh legacy process interleaved in each pair. Each process takes five cold and five warm samples (`--sample-ms 100`). All individual samples, fault counts, command lines, memory audits and binary hashes are preserved. Profiled and output-consuming processes are separate diagnostics and are not acceptance samples.

| Binary | SHA256 |
| --- | --- |
| Selected candidate | `843a6df1f09688d2b3c4458dc6a76818ab9e97a71b1d1af310a03f71e27c5e50` |
| Committed specialization | `bcaf0d64c618a030a8529de67410204e1773e660f613a55be28c90e38e9f4346` |
| Frozen legacy | `0988f8960179ecdcbc5d17e866d524ccd532b2edb4842e5121021dbbfe778a1d` |

`window32_aligned_clean/build_metadata.json` records staged source hashes, compiler/linker commands and linked-input hashes. `integrated_source_hashes.json` also records the CMake AMX INT8 flag, tests and rendered documentation. Controls are under `/home/tianmuli/vllm_test/snc_scaling_analysis/regression_fix_20261029/{single_shape_l2_final,final}`.

### Qualification

Medians aggregate the five process medians; variability is their sample standard deviation. Controller threads are isolated on CPUs 0–15, outside the measurement set (`controller_affinity.log`). Benchmark binding, preloads, allocations, rotation and acceptance remained unchanged. No samples were excluded.

| Mode | Cold ms | Cold TOPS | Warm ms | Process median SD ms |
| --- | --- | --- | --- | --- |
| candidate | 2.554976 | 161.38 | 2.294014 | 0.007352 |
| current | 2.789365 | 147.82 | 2.490180 | 0.006032 |
| legacy | 3.216714 | 128.18 | 2.456407 | 0.014143 |

| Pair | Candidate ms | Current ms | Legacy ms | Candidate ≤ threshold |
| --- | --- | --- | --- | --- |
| 1 | 2.554976 | 2.776531 | 3.216714 | yes |
| 2 | 2.561902 | 2.791947 | 3.188306 | yes |
| 3 | 2.556059 | 2.787425 | 3.209132 | yes |
| 4 | 2.541828 | 2.789365 | 3.218093 | yes |
| 5 | 2.552678 | 2.789391 | 3.225135 | yes |

Candidate cold sample range: 2.537869–2.565112 ms. Warm latency improves by 7.88% against the committed specialization. Complete records: `qualification_window32_aligned_clean.json`.

### Historical qualification

| Candidate | Cold ms | Cold TOPS | Warm ms | Passing processes |
| --- | --- | --- | --- | --- |
| Initial u4 | 2.575473 | 160.09 | 2.298439 | 3/5 |
| Isolated-controller u4 repeat | 2.590626 | 159.16 | 2.308087 | 0/5 |
| N32 window without alignment | 2.557545 | 161.22 | 2.330786 | 4/5 |
| N32 window plus alignment, before cleanup | 2.551284 | 161.61 | 2.297186 | 5/5 |

The initial u4 aggregate above 160 TOPS did not reproduce. Its isolated-controller repeat remained 7.12% faster cold than its committed control but failed all five process medians. N32 without alignment passed four of five, with a failing first median of 2.603648 ms. These complete failures are retained, including every sample; controller isolation alone is not established as the cause of any timing change.

### Performance-governor rerun (2026-10-30)

All measurement CPUs 96–191 had `performance` governor with `intel_pstate` and `energy_performance_preference=performance`, verified before and after the run. The previous governor was not captured. Frozen binary hashes, core/NUMA placement, preloads, bank rotation and the complete five-pair protocol were retained. No source or arithmetic changes were made.

| Mode | Cold ms | Cold TOPS | Warm ms | Process median SD ms |
| --- | --- | --- | --- | --- |
| candidate | 2.554138 | 161.43 | 2.295070 | 0.003593 |
| current | 2.802465 | 147.13 | 2.496155 | 0.009659 |
| legacy | 3.200024 | 128.85 | 2.444524 | 0.008275 |

All five candidate process medians pass: 2.557218, 2.547538, 2.554138, 2.552162, 2.554436 ms.
Candidate latency changed by -0.033% versus the earlier qualification, smaller than process variability; this rerun does not establish a material governor benefit.

| Separate core-counter diagnostic | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| Effective GHz, cycles/reference cycles × 2.4 | 2.960 | 2.704 | 2.841 |

Clock counters used three additional fresh processes excluded from acceptance. A performance governor does not fix the actual clock. Earlier separate diagnostics measured 2.954/2.706/2.831 GHz respectively; these observations do not isolate governor causality.

Records: `qualification_performance_governor_20261030.json`, `performance_governor_20261030_environment{,_after}.json`, and `profiles-performance-governor-20261030/summary.json`.

### Bounded experiments

The locality experiments were motivated by [the paper's space-filling traversal and panel budget](https://arxiv.org/pdf/2601.16294); BF16 results were not treated as a predicted INT8 speedup. Inspecting source-matched oneDNN JIT code first showed full-unrolled K loops, ordinary A loads (stride4096), streaming B loads (stride128), operand reuse within M32 tiles, repeated B streams across M128, and accumulator `prefetchw`; it did not have software operand prefetch. The capture and disassembly are preserved in `jit-current/`.

Scheduling screened 21 measurements covering 19 distinct configurations: matched row ownership, strides 4096/4160/4224, full-K macroblocks 128×128/128×64/64×128, rectangular/Hilbert/Morton ownership with both curve orientations, and alternating reverse traversal for finalists. Curves discard padded coordinates and assign 96 equal contiguous ranges. No broad K-blocking sweep was repeated.

AMX screened 38 parameter configurations: 12 geometry/load/layout combinations (32×32 or 16×64; ordinary or streaming B; contiguous, padded or tile-major A), 24 prefetch combinations on two finalists, and two additional unroll factors. Tile-major storage included existing row quantization into a 4 KiB temporary followed by 64-byte scatter in timing. Prefetch covered T1/T0, 128/256/512 lookahead, eight evenly spaced or all 32 lines per future N32/K64 panel; all addresses were valid and all work timed. Fixed-source and final compiler checks then compared grouped/interleaved execution, forced inlining and hoisted scratch. No temporary tuning controls remain in production.

| Screening family | Best cold ms | Finding |
| --- | --- | --- |
| Matched padded rectangular traversal with reverse | 2.696303 | Best scheduling-only result |
| Morton/Hilbert | 3.147332 | Best curve slower than matched rectangle |
| AMX 32×32 tile-major, ordinary B, no prefetch | 2.601306 | Raw screen best |
| AMX 32×32 padded rows, ordinary B, no prefetch | 2.602732 | Within 1%; less scratch |
| AMX 16×64 | 2.738968 | Slower than 32×32 |
| Streaming B | 2.668149 | Slower than ordinary B |
| Prefetch | 2.709788 | Every screened variant slower than corresponding off |
| Tile-major unroll2 /4 | 2.630420 /2.637428 | Slower than unroll 1 |
| Fixed padded / tile-major | 2.599827 /2.609606 | Padded won 1% simplicity tie |
| Production grouped / forced-inline grouped | 2.724764 /2.711144 | Source simplification changed code generation |
| Interleaved / forced-inline interleaved | 2.585515 /2.646476 | Historical ordinary compilation |
| Hoisted worker scratch with automatic inlining | 2.638904 | Historical variant rejected |

Selection uses cold latency. Within 1%, prefer existing BRGEMM, then no prefetch, then less scratch. `final_selection.json` records the historical interleaved selection and rejected codegen checks; `selection.json` preserves the earlier fixed-layout selection. The final integrated selection is recorded separately in `integrated_selection.json`.

### Focused source-generation follow-up

After the original bounded scheduling and AMX sweeps, the production source was checked for stack and instruction-generation costs. Hoisting accumulator scratch with explicit noinline, then unrolling four K64 steps, improved the repeated result. Static inspection found no accumulator spills and 16 complete K256 iterations; compensation and epilogue semantics were unchanged. Historical `final` qualification measured 159.65 TOPS and failed the strict threshold. The subsequent u4 candidate also failed strict qualification; narrowing the window and aligning private A supplied the qualifying combination.

| Follow-up screen | Cold ms | Result |
| --- | --- | --- |
| Hoisted scratch, noinline, unroll2 | 2.581656 | Improved stack generation |
| Hoisted scratch, noinline, unroll1 | 2.633400 | Slower |
| Hoisted scratch, noinline, unroll4 | 2.574495 | Selected for repeated qualification |
| Cached epilogue parameters, unroll2 /4 | 2.580716 /2.572959 | Small screen gain; generic refactor not selected |
| Cached epilogue with restored alias qualifiers | 2.570623 | Rejected after strict qualification failed |
| Software operand pipeline | 2.648000 | Rejected |
| B-first operand loads | 2.584856 | Rejected |
| Macro 128×128 /64×128 | 2.684860 /2.686918 | Rejected |
| Row stride 4224 | 2.638798 | Rejected |
| Tile-major with selected unroll4 | 2.648599 | Rejected |
| Timed output-page touches, 2 MiB /4 KiB | 2.580282 /2.580655 | Rejected |
| Unroll8 /16 | 2.665827 /2.741346 | Rejected |
| Earlier second-A tile load | 2.575652 | Strict follow-up qualification recorded separately |

Periodic minor-fault windows are consistent with tcmalloc scavenging freed tensor storage with `madvise(MADV_DONTNEED)`. Sample logs lack fault addresses, so this does not prove tensor ownership or causation of the full timing difference. Timed output-page touches retained allocation and all arithmetic, but did not improve the selected kernel. No allocator suppression, output reuse, benchmark-window change or fault-sample exclusion was used. `allocation_notes.md` records the evidence and its limits.

### Narrow-window and allocation follow-up

These focused experiments preserve full K, exclusive output ownership, public weight packing and the timed fused operator. They do not replace the original bounded sweeps.

| Follow-up screen | Cold ms | Warm ms |
| --- | --- | --- |
| N32 window, original screen | 2.595881 | 2.334613 |
| N32 window, isolated-controller screen | 2.557855 | 2.339562 |
| Direct tile-major quantization, without temporary scatter | 2.668248 | 2.407506 |
| Four / sixteen starting-column phases | 2.704732 /2.964638 | 2.319168 /2.306344 |
| Synchronized forward / reverse traversal | 2.577170 /2.616832 | 2.337091 /2.307867 |
| Alternate-worker offsets 8 /24 | 2.587662 /2.580073 | 2.311586 /2.313911 |
| Private activation buffer aligned to 2 MiB | 2.581465 | 2.298637 |
| N32 window plus private 2 MiB alignment | 2.552387 | 2.297567 |
| Same combination, single-iteration N loop removed | 2.553908 | 2.293145 |

N32 traversal reduces the complete 256-row step's padded operand union to 1168.125 KiB and covers every output tile exactly once. Its tile helpers are instruction-byte identical to selected u4. Spreading workers across more starting columns hurt cold latency while warm latency remained closer; this supports testing temporal panel sharing but does not isolate LLC slice contention. Direct tile-major quantization preserves row arithmetic but adds scattered-store address calculations and was slower. Alignment adds up to 2 MiB of private scratch, with allocation and quantization still timed. The combined candidate passed subsequent full qualification. Removing its single-iteration column loop preserved tile instruction bytes but changed helper alignment and worker register allocation, so the cleaned source received its own fresh-process qualification.

### Verification and diagnostics

The selected scaled-MM selection/scheduler suite passed: `26 passed, 2 warnings in 291.16s`. Command: `numactl --all --physcpubind=0-95 --membind=0 <torch-venv>/bin/python -m pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py -v`, with the selected staged extension and Intel OpenMP/tcmalloc preload. Coverage includes constrained actual teams, mismatched K, strided BF16 input including padding, zero/tiny rows, BF16 quantization ties and adjacent boundary values, unequal scales, bias/no-bias, first direct execution, direct→BRGEMM→direct transitions, and unchanged inputs. Tests extend the nearest existing subprocess suite.

The selected binary matched every one of 50,331,648 elements per output against frozen legacy for seed 123, unequal per-channel scales, and bias/no-bias; activation, packed weights, scales and bias remained unchanged. Every screened candidate received the same full-output check before timing.

The existing cached `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` comparison passed with exactly identical greedy token outputs for two long prompts and 32 generated tokens. It used six workers and max-batched-token limit 1536; **this exercises fallback, not the exact-shape AMX specialization**. Both model processes and their workers terminated; records are in `window32_aligned_clean_model/terminal.json`.

Applicable pre-commit hooks and `git diff --check` pass for C++, CMake, tests, rendered docs and this report. Local fresh-agent documentation discovery found interface/default/restart behavior, shape/team guard, AMX/Linux requirements and fallback on the CPU installation page. No PR or docs preview was opened.

Counters below are separate diagnostics, enabled only for five cold windows. Counts divide by actual timed calls and aggregate cycles over inherited workers; fractions overlap. Each counter group requires ≥99% coverage. Event definitions and encodings are recorded against the Granite Rapids V1.21 event JSON. Counts and traffic are absolute per call, not inferred from cycle fractions.

| Absolute aggregate cycles per call (millions), with cycle fraction | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| Execution cycles | 730.01 | 732.24 | 883.01 |
| AMX busy | 201.35 (27.58%) | 201.37 (27.50%) | 201.40 (22.81%) |
| Load bound | 517.25 (70.86%) | 528.60 (72.19%) | 674.19 (76.35%) |
| L1 miss stalls | 425.55 (57.91%) | 445.71 (61.00%) | 618.80 (69.63%) |
| L2 miss stalls | 188.02 (25.54%) | 196.49 (26.78%) | 351.52 (39.40%) |
| L3 miss stalls | 8.27 (1.12%) | 22.78 (3.10%) | 21.64 (2.43%) |
| DTLB load walk cycles | 0.13 (0.02%) | 0.11 (0.01%) | 0.11 (0.01%) |
| DTLB store walk cycles | 0.09 (0.01%) | 0.10 (0.01%) | 0.12 (0.01%) |

| Event counts per call | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| L1 replacement lines | 213,799,367 | 111,257,779 | 109,830,403 |
| L2 demand hits | 137,943,291 | 202,044,262 | 201,462,370 |
| L2 demand misses | 1,149,611 | 1,479,602 | 1,468,467 |
| L2 RFO hits | 2,346,742 | 5,607,144 | 4,871,382 |
| L2 RFO misses | 179,332 | 161,934 | 424,907 |
| All L2 RFO | 2,530,073 | 5,770,239 | 5,297,471 |
| DTLB load walks | 426 | 395 | 379 |
| DTLB store walks | 1,064 | 1,338 | 1,499 |

| Traffic per call (MiB) | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| L2 fills | 949.59 | 952.58 | 1414.08 |
| Dirty L2 evictions | 42.95 | 44.91 | 249.33 |
| Socket cache read responses | 890.45 | 900.42 | 1349.69 |
| Local DRAM read responses | 50.68 | 50.49 | 57.05 |
| Socket-1 IMC read | 61.96 | 63.71 | 74.86 |
| Socket-1 IMC write | 100.20 | 100.35 | 5.19 |

Line traffic is event count × 64 bytes, including repeated requests; it is not unique data volume. Demand misses exclude merged misses. IMC scaled units are honored and socket-wide counters include background activity.

| Independent core/stall repeat | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| Load-bound Mcycles/call | 519.06 | 531.90 | 676.96 |
| L2-miss stall Mcycles/call | 188.06 | 197.03 | 350.80 |
| L3-miss stall Mcycles/call | 8.21 | 22.50 | 21.80 |

| Cycles/reference-cycles ratio expressed as GHz (2.4 GHz reference) | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| First diagnostic | 2.954 | 2.706 | 2.831 |
| Repeat diagnostic | 2.955 | 2.713 | 2.840 |

Changes below compare absolute cycles per call against the committed specialization, not cycle fractions.

Load-bound cycles changed by -2.15% in the first diagnostic and -2.41% in the independent repeat.
L2-miss stall cycles changed by -4.31% in the first diagnostic and -4.55% in the independent repeat.
L3-miss stall cycles changed by -63.68% in the first diagnostic and -63.48% in the independent repeat.

The frequency ratios differ between implementations. These counters do not isolate LLC slice contention or justify attributing the wall-time gain solely to L2 locality. A reduced cycle fraction alone is not evidence of reduced absolute stalls. Full fill-buffer, TLB, traffic and counter-coverage records remain in the summaries.

The separate diagnostic immediately consumes every output element with `torch.sum(output, dtype=torch.float32)`. It retains bank rotation and placement; these samples are excluded from standalone acceptance.

| Operator + output sum median ms | Candidate | Current | Legacy |
| --- | --- | --- | --- |
| Cold | 3.155473 | 3.434907 | 3.906759 |
| Warm | 2.930681 | 3.127743 | 3.226906 |

Cold chain latency improves by 8.14% and warm chain latency by 6.30% against current in this diagnostic. Each mode has one fresh process with five samples, less replication than qualification. Chain and standalone timings cannot be subtracted to isolate a store-fence cost: the consumer changes cache residency and subsequent reuse.

### Reproduction artifacts

Persistent archive: `/home/tianmuli/vllm_test/snc_scaling_analysis/single_shape_160_20261029`. Its `ARCHIVE.md` explains original-path substitution. It includes all screening samples, source snapshots, binaries, build commands/hashes, JIT capture, qualification, profile manifests, frozen outputs, test logs and model token outputs. Original root: `/tmp/sgl-160-20261029`.

With repository `.venv/bin/python`, replay `qualify_new.py <archive>/window32_aligned_clean/stage/vllm <archive>/final_config.json --label replay`, using a fresh label. Run `profile_new.py` with the same stage/config and fresh `--out`; add `--chain-only` for output-consuming timing. Torch processes use `/home/tianmuli/vllm_test/vllm_local/.venv/bin/python` and the captured bootstrap. Build snapshots include `-mamx-int8`; a full rebuild should follow the documented incremental CPU workflow.

AI assistance was used for implementation, experiments and this report. This is local work. The specialization still needs evaluation in an actual model workload with this exact shape.

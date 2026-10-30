# Full-K CPU INT8 single-shape experiment

## Result

**160 cold INT8 TOPS qualified.** The selected change measures **2.554976 ms / 161.38 TOPS**, with **8.40% lower cold latency** than the source-matched committed specialization. All five fresh-process candidate medians are below **2.57698038 ms**, using five cold samples per process and alternating candidate/current order with interleaved legacy controls. Earlier u4 and N32-only candidates failed this strict requirement; their results remain below and in the archive.

Acceptance times the fused operator without bias, including allocation, row quantization, compensation, scaling, BF16 conversion, streaming stores, and the output fence. Bias and no-bias correctness are both verified. Standalone acceptance excludes downstream output reads. Model generation below exercises fallback.

## Selected implementation

The opt-in interface remains `VLLM_CPU_SGL_SCHEDULER=single_shape`. Linux AMX INT8 support and process-wide tile-state permission are checked before direct execution. The exact BF16 M4096/N12288/K4096 shape requires 96 actual workers; other cases preserve `flat_auto` fallback.

The full-K 32×32 u8×s8 kernel uses four INT32 accumulators, two A tiles and two B tiles. Ordinary tile loads interleave independent load/dot operations in four K64 steps per iteration. It preserves public N32 `[K/4,32,4]` VNNI packing (B stride128), subtracts compensation once, and uses the existing epilogue. Backend tile state is reset, configuration is loaded per worker, and tiles are released afterward. Software prefetch is off.

Quantized A uses a private 4160-byte row stride, adding 256 KiB over contiguous rows. Its start is aligned to 2 MiB within a private allocation that reserves 2 MiB minus one byte of additional scratch space; this does not guarantee huge-page backing. Quantization ownership matches the 16×6 GEMM grid: six workers divide each 256-row partition exactly once and synchronize. Each worker exclusively owns 256×2048 output elements and traverses two inner 128×32 windows for each N32 step, reversing alternate workers with the half-partition offset. The complete 256×32 step has a padded operand union of **1168.125 KiB**: 1040 KiB of quantized A and 128.125 KiB of packed B plus compensation. One inner 128×32 window accounts for **648.125 KiB**. These totals exclude scratch, scales, output and other live data; they do not establish cache residency. Aligned 4 KiB accumulator scratch is allocated once per worker and passed to the tile helper, which is marked `noinline`. The four accumulators remain in AMX tiles throughout K. If eligibility holds but the actual team is smaller, fallback uses contiguous quantized rows within the same private allocation.

## Controls and protocol

Base commit: `d80bd2eeb5cb3ae7be0d83d661e723ad51491189`. Original reference: 147.48 TOPS at 2.79567 ms. Hardware: Intel Xeon 6972P, 96 physical cores per socket, 2 MiB L2 per core, 480 MiB LLC per socket. Acceptance uses physical CPUs 96–191 and NUMA node 1. Builds, correctness and model checks use the other socket.

Each fresh process preloads `/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4` and `/home/tianmuli/vllm_test/vllm_local/.venv/lib/libiomp5.so`, sets `KMP_AFFINITY=granularity=fine,proclist=[96-191],explicit`, `OMP_NUM_THREADS=96`, `OMP_THREAD_LIMIT=96`, `OMP_DYNAMIC=FALSE`, and `THP_MEM_ALLOC_ENABLE=1`. Packed banks have verified 100% huge-page backing on node 1: 33 disjoint matrices, 1,662,566,400 total bytes, 1,612,185,600 intervening bytes before revisit (>1.5 GiB). Rotation continues between samples.

Five candidate/current pairs alternate their process order, with a fresh legacy process interleaved in each pair. Each process takes five cold and five warm samples (`--sample-ms 100`). All individual samples, fault counts, command lines, memory audits and binary hashes are preserved. Profiled and output-consuming processes are separate diagnostics and are not acceptance samples.

| Binary | SHA256 |
| --- | --- |
| Selected candidate | `843a6df1f09688d2b3c4458dc6a76818ab9e97a71b1d1af310a03f71e27c5e50` |
| Committed specialization | `bcaf0d64c618a030a8529de67410204e1773e660f613a55be28c90e38e9f4346` |
| Frozen legacy | `0988f8960179ecdcbc5d17e866d524ccd532b2edb4842e5121021dbbfe778a1d` |

`window32_aligned_clean/build_metadata.json` records staged source hashes, compiler/linker commands and linked-input hashes. `integrated_source_hashes.json` also records the CMake AMX INT8 flag, tests and rendered documentation. Controls are under `/home/tianmuli/vllm_test/snc_scaling_analysis/regression_fix_20261029/{single_shape_l2_final,final}`.

## Qualification

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

## Historical qualification

| Candidate | Cold ms | Cold TOPS | Warm ms | Passing processes |
| --- | --- | --- | --- | --- |
| Initial u4 | 2.575473 | 160.09 | 2.298439 | 3/5 |
| Isolated-controller u4 repeat | 2.590626 | 159.16 | 2.308087 | 0/5 |
| N32 window without alignment | 2.557545 | 161.22 | 2.330786 | 4/5 |
| N32 window plus alignment, before cleanup | 2.551284 | 161.61 | 2.297186 | 5/5 |

The initial u4 aggregate above 160 TOPS did not reproduce. Its isolated-controller repeat remained 7.12% faster cold than its committed control but failed all five process medians. N32 without alignment passed four of five, with a failing first median of 2.603648 ms. These complete failures are retained, including every sample; controller isolation alone is not established as the cause of any timing change.

## Performance-governor rerun (2026-10-30)

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

## Bounded experiments

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

## Focused source-generation follow-up

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

## Narrow-window and allocation follow-up

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

## Verification and diagnostics

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

## Reproduction artifacts

Persistent archive: `/home/tianmuli/vllm_test/snc_scaling_analysis/single_shape_160_20261029`. Its `ARCHIVE.md` explains original-path substitution. It includes all screening samples, source snapshots, binaries, build commands/hashes, JIT capture, qualification, profile manifests, frozen outputs, test logs and model token outputs. Original root: `/tmp/sgl-160-20261029`.

With repository `.venv/bin/python`, replay `qualify_new.py <archive>/window32_aligned_clean/stage/vllm <archive>/final_config.json --label replay`, using a fresh label. Run `profile_new.py` with the same stage/config and fresh `--out`; add `--chain-only` for output-consuming timing. Torch processes use `/home/tianmuli/vllm_test/vllm_local/.venv/bin/python` and the captured bootstrap. Build snapshots include `-mamx-int8`; a full rebuild should follow the documented incremental CPU workflow.

AI assistance was used for implementation, experiments and this report. This is local work. The specialization still needs evaluation in an actual model workload with this exact shape.

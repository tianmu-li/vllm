# CPU INT8 single-shape experiments

## Qualified 200 TOPS: activation load hints and N32 windows (2026-10-31)

**The stateless `c_256_32_1024` implementation passes the 200-TOPS gate.** All five fresh-process cold medians are at most 2.06158430208 ms. Their median is **2.053165 ms / 200.82 INT8 TOPS**, using `2*M*N*K`, versus 2.202130 ms for frozen production `d98b499fcf`. The geometric paired latency change is -6.690%, with process-pair bootstrap 95% interval [-7.303%, -5.980%]. No samples or processes were excluded. The slowest process is 2.059102 ms; this result qualifies this host, shape and binding rather than other workloads or machines.

| Pair | Selected cold ms | Production cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.057038 | 2.225864 | 3.201001 |
| 2 | 2.053165 | 2.202130 | 3.250810 |
| 3 | 2.052633 | 2.192868 | 3.196406 |
| 4 | 2.052276 | 2.213648 | 3.208141 |
| 5 | 2.059102 | 2.176640 | 3.196017 |

Warm median latency is 1.837358 versus 1.948441 ms. Five separate alternating output-consuming pairs give 2.924602 versus 2.997481 ms cold and 2.762006 versus 2.745290 ms warm. The cold chain's geometric change is -2.928%, but its 95% interval [-5.914%, +0.106%] crosses zero. **A reliable end-to-end improvement is not established.** These chain measurements retain the immediate full-output `torch.sum(dtype=float32)` and are excluded from standalone acceptance.

### Implementation and search selection

Production changes three lines from `d98b499fcf`: activation tile loads use `tileloaddt1`, and the N macroblock is 32 columns. The M macroblock remains 256 and reduction remains four ascending K1024 blocks. Weights and partial INT32 accumulators use ordinary tile loads. The C row stride becomes 48 INT32 values, reducing each worker's private slab from 80 to 48 KiB and the fresh per-call C arena from 7.5 to **4.5 MiB**. The logical extra partial-C traffic remains **1,152 MiB per call**. The 48.8 MiB activation arena, shared maximum pass, paired conversion/publication schedule, output streaming/fence, public N32 weight packing, actual-team guard and fallback remain unchanged. There is no retained state.

The continuation used a bounded 12-configuration mechanism campaign followed by a bounded six-configuration campaign; these follow the completed 24-configuration search recorded below. Every configuration received exact frozen seeded legacy and edge-fixture checks before five-cold/five-warm fresh-process screens. The first continuation bracketed against frozen production. It selected activation-T1 loads, whose overall qualification median was 2.043328 ms / 201.79 TOPS, but **only three of five process medians passed**, so that result did not qualify 200 TOPS. All five medians [2.066426, 2.043328, 2.041989, 2.033133, 2.203563] remain recorded. A subsequent fixed-count timing diagnostic did not replace or repair that failed qualification.

The six-configuration campaign bracketed against the frozen unintegrated A-T1 stateless implementation to measure incremental gain. Four variants directly published register-packed quantized rows to three replicas; two tested smaller N32 accumulator slabs. The fastest normalized screen was non-temporal direct publication with shared maxima (ratio 0.986211), but its cold screen missed the target at 2.070597 ms. M256/N32 (ratio 0.989350) was within 1% and won the declared tie rule through smaller scratch and cached publication. Its 2.054623-ms screen triggered immediate qualification against actual frozen production. No further optimization configurations were built after the strict target pass.

| Continuation configuration | Screen cold ms | Screen warm ms | Change vs adjacent brackets |
| --- | ---: | ---: | ---: |
| `k1024_a_t1` | 2.059432 | 1.817852 | -6.511% |
| `k1024_a_t1_hoisted` | 2.061714 | 1.804692 | -7.322% |
| `k1024_rolling` | 2.155236 | 1.923448 | -4.322% |
| `k2048_rolling` | 2.255633 | 2.023954 | +0.849% |
| `k_asymmetric` | 2.330683 | 2.013648 | +4.382% |
| `k1024_rolling_a_t1` | 2.135306 | 1.950096 | -5.431% |
| `k2048_a_t1` | 2.218580 | 2.011080 | +1.501% |
| `k2048` | 2.239522 | 2.010191 | +1.853% |
| `k1024_b_t1` | 2.280407 | 1.931404 | +3.786% |
| `k2048_b_t1` | 2.300485 | 2.008071 | +3.996% |
| `k1024_hoisted` | 2.215699 | 1.951569 | -0.309% |
| `k2048_hoisted` | 2.225622 | 1.964794 | -0.405% |
| `c_128_32_1024` | 2.244241 | 2.219943 | +5.519% |
| `c_256_32_1024` | 2.054623 | 1.835652 | -1.065% |
| `d_r3_cached_shared` | 2.104087 | 1.891401 | +1.249% |
| `d_r3_cached_fused` | 2.140504 | 1.932964 | +1.158% |
| `d_r3_nt_shared` | 2.070597 | 1.797529 | -1.379% |
| `d_r3_nt_fused` | 2.202262 | 1.978164 | +7.595% |

Measurements retain CPUs96–191/node1, Intel OpenMP/tcmalloc preloads, performance governor, 33 disjoint packed-weight banks, full verified huge-page backing and **1,612,185,600 intervening packed bytes**. Allocation, preparation, GEMM, epilogue and fence remain timed. Builds, tests, cleanup and other measurements stop during performance captures. The original calibration and sample-window method is unchanged. Source-matched fixtures establish the actual 96-worker optimized path separately from requested/Torch thread counts in timing records.

### Evidence and reproduction

The integrated source SHA256 is `640f80466257867323306d3a4d95c00f632148e3fcf0a912cde47a9fe51c5a9b`, identical to the qualified snapshot. The qualified library SHA256 is `b412ecf0085cebe5cdbe258440bfabda5f182ab41a296a9998fa819f418c77cf`. Frozen production source/library hashes and all 44 source-matched link inputs remain pinned. No library code was copied from oneDNN; the bounded upstream review supplies independent evidence for separate activation/weight load hints and cache heuristics.

Archives are `/home/tianmuli/vllm_test/snc_scaling_analysis/continued_200_d98b499fcf_20261030` (12 configurations, failed A-T1 qualification and diagnostics) and `/home/tianmuli/vllm_test/snc_scaling_analysis/direct_publication_200_72e7a7b3ec_20261031` (six configurations, selected N32 qualification and consuming chain). Their sources, configuration generators, manifests, commands, raw measurements, selection and lossless cleanup/restoration journals remain separately addressable. The latter archive's `README.md` gives build, screen, qualification and selection commands; `chain_only.py c_256_32_1024` collects separate consuming-chain pairs after accepted qualification.

All six configurations match frozen seed-123 legacy outputs with and without bias. Target-shape edge fixtures also cover unequal scales, quantization boundaries, NaN/Inf, strided inputs and bytewise input immutability, with per-build activation/C allocation evidence proving the actual 96-worker optimized path. The selected existing scaled-MM suite passes **26 tests**, including team/shape fallback, repeated eligible calls, first direct AMX execution and direct/BRGEMM alternation. The cached RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 comparison passes with identical greedy outputs for two long prompts and 32 generated tokens. Its six-worker run is **fallback-only**; it does not qualify model execution of the specialization. Final accuracy records are in `verification/final_checks_terminal.json` and `verification/c_256_32_1024_correctness_model/terminal.json`.

Selected disassembly confirms complete ascending K coverage, expected INT32 reload/store traffic, full-K epilogue lifetime, pending-C consumption before overwrite and final drain before tile release/fence. Quantization loop bytes match the A-T1 base. There are no unexpected vector/AMX accumulator spills; scalar control stack traffic remains. The integrated C++ hash matches the qualified source. Existing public-entry coverage and archived fixtures cover the risks; no new repository tests or helpers were added.

### Native diagnostics and conditional EMON

All three source-matched diagnostic stages pass exact-output and scratch-layout checks. Eighteen sequential captures (three sources, six groups including clock-only) retain offcore `config1`, untimed counter initialization followed by restored 33-bank conditioning, five marked calls, actual 96-worker evidence and exact positive running/enabled equality. Raw counts are not multiplex-adjusted. Startup covers the same N256/64 full-K accumulations per worker, and remaining GEMM N1792/448 accumulations.

The selected N32 macroblock retains eight completed tiles at the startup boundary, versus 16 for the committed N64 macroblock and one for private256. Thus startup has published 56, 48 and 63 tiles respectively. Completed accumulation spans match, while output publication differs; whole-worker measurements include the final drain/fence. Pending output and scratch extents are explicit in `diagnostics/comparison.json` and per-source manifests.

| Absolute aggregate Mcycles per marked call | Production startup / remaining | Private256 startup / remaining | Selected startup / remaining |
| --- | ---: | ---: | ---: |
| AMX busy | 25.362 / 176.174 | 25.176 / 176.169 | 25.225 / 176.201 |
| Load bound | 87.559 / 339.382 | 66.718 / 326.650 | 75.201 / 249.743 |
| Execution cycles | 103.408 / 434.344 | 80.438 / 417.122 | 91.124 / 342.233 |

Clock-only median worker preparation/startup/remaining/drain durations are 139.338/369.562/1545.819/5.577 us for production and 140.314/372.080/1430.803/2.440 us for selected. Whole-worker medians are 2059.296 versus 1976.778 us. Independent event-group medians are not an additive time balance. Core-group cycles/reference-cycles ratios differ (approximately 1.208 versus 1.042 during startup), so these counts do not attribute the wall-time gain solely to cache behavior or prove L2 residency. Offcore line equivalents count requests, not unique operand bytes or LLC-array bandwidth.

The selected screen improves by 1.065% against its A-T1 brackets and qualification improves by 6.690% against production. However, startup load-bound excess versus private256 is +8.483M cycles and remaining excess is -76.908M cycles. The fraction of positive excess persisting beyond startup is **0%**, below the 25% gate. The conditional driverless EMON repeat is therefore skipped; no new saturation claim is made. All native workload and collector processes completed and were reaped.

### Cleanup and delivery

The preceding 12-configuration continuation reclaimed **3,855,892,480 allocated bytes**; this six-configuration archive reclaimed **2,483,970,048 bytes** across 36 compressed inode actions and 48 original aliases. Combined continuation reclamation is **6,339,862,528 bytes / 5.90 GiB**. Final available space is **8,606,265,344 bytes**, above the 8 GiB reserve. Gross artifact savings differ from net available-space growth because sources, raw evidence and selected binaries remain retained.

Every compressed payload was decompressed and SHA256-checked before originals or verified redundant aliases were removed. Completed-attempt, active-reference, dependency, inode, permission and whole-mirror proofs plus restoration commands are retained in the new journals. Unique sources, original commands/manifests, all frozen controls, legacy tensors, builder inputs/directories and runtime/model environments remain preserved. The later archive explicitly releases only three binary pins from its rejected provisional direct-publication selection; original dependency pinsets are unchanged. Final checks pass 13,026 original inode hashes, the preceding 805-file pinset, and the current 1,499-file/194-directory/seven-symlink pinset. Known non-dumpable service `/proc` visibility limits remain recorded. Owned comparison tensors are absent; shared-memory cleanup is separate from disk accounting.

The selected production library stays directly loadable. Obsolete diagnostic and rejected build products are losslessly compressed, with all generated assembly, raw native captures, event headers and resource-release records preserved. All owned controllers, subprocesses, model workers, collectors and cleanup processes completed and were reaped. Applicable repository pre-commit hooks and `git diff --check` pass; the final integrated C++ SHA256 still equals the qualified source. No additional optimization configurations were built after acceptance.

Historical results below remain unchanged. This work used AI assistance and has not opened a PR.

---

## Expanded search: selected stateless K-blocked kernel (2026-10-30)

**Integrate `b_256_64_1024` as a partial cold improvement; 200 TOPS was not reached.** The five-pair qualification gives **2.208672 ms / 186.68 cold TOPS**, versus 2.245464 ms for frozen `af3cabd16a`. The median latency reduction is 1.639%; the geometric paired change is -1.299%, with process-pair bootstrap 95% interval [-2.102%, -0.434%]. This passes the predeclared partial integration gate. All five finalist process medians exceed 2.06158430208 ms, so none passes the 200-TOPS gate. Another 0.147088 ms / 6.660% latency reduction is needed from this result.

The tradeoff is material: standalone warm latency is 1.957159 versus 1.939355 ms (+0.918%). A separate five-pair operator-plus-`torch.sum(output, dtype=float32)` consuming chain is 3.019710 versus 2.930580 ms cold (+3.041%), and 2.779562 versus 2.684401 ms warm (+3.545%). The chain's geometric paired regression is +2.602%, with 95% interval [+1.034%, +4.025%]. Cold standalone latency determines selection under the supplied acceptance rule. These results do not establish an end-to-end improvement, and subtracting independent chain and standalone timings cannot isolate fence cost.

### Implementation and complete bounded search

All **24 new optimization configurations** were built outside production, verified against seeded frozen legacy output before screening, and screened with five cold and five warm samples in fresh processes bracketed by fresh committed processes. Frozen and diagnostic controls do not count toward that budget. Ranking uses adjacent committed-bracket latency ratios; candidates within 1% follow the fixed preference for committed production, full-K execution, smaller scratch, cached publication, shared-max schedule, then ID. No screen reached the 200-TOPS threshold.

The eight quantize-once variants partition each M256 group's rows among its six producers, convert each row once in an aligned 4 KiB local buffer, and publish completed 64-byte chunks to R2/R3/R6 replicas. Cached and non-temporal variants retain shared maxima; two additional non-temporal R3/R6 variants fuse unchanged row maxima with conversion. Arena allocation remains worker-zero-only inside the actual-96-worker guard, with publication barriers and producer fences for NT stores. They reuse the existing row arithmetic and full-K lower-first GEMM. The new fan-out combinations trade conversion savings for approximately 32.53/48.80/97.59 MiB of activation publication. Committed preparation wins this family.

The twelve blocked variants cross M={128,256}, N={64,128,256}, K={512,1024}, using committed preparation. The selected kernel retains each worker's M256×N2048 output rectangle, `mgroup=worker%16`, `ngroup=worker/16`, half-range N phase offset, public N32 weight packing, and exact 32×32 AMX arithmetic. It traverses K in ascending four 1024 blocks, zeroes INT32 accumulators for the first block, reloads/stores partial sums thereafter, and applies compensation and floating-point epilogue only after full K. Archival macroblock overlap consumes the previous completed macroblock before overwriting its C slots; replica-local activation/scale offsets also apply to pending output and final drain. Tile release and output fence follow the final drain.

A fresh heap C tensor contributes **7.5 MiB per eligible call**, with 80 KiB per worker and `Cstride=80`. It is allocated inside the timed actual-team guard and published with the existing 48.8 MiB activation arena. There is no retained scratch in production; output remains fresh and inputs/weights remain immutable. The selected active-slice estimate is 400 KiB, versus measured 2 MiB L2 per core; this estimate does not prove residency. Relative to full-K execution, logical extra partial-C traffic is **1,152 MiB per call** for K1024 and 2,688 MiB for K512, excluding the final epilogue C read common to both.

| Configuration | Screen cold ms | Screen warm ms |
| --- | ---: | ---: |
| `a_r2_cached` | 2.738238 | 2.468694 |
| `a_r2_nt` | 2.629183 | 2.374747 |
| `a_r3_cached` | 2.383160 | 2.111935 |
| `a_r3_nt` | 2.311473 | 2.014685 |
| `a_r3_nt_fused` | 2.297494 | 1.982799 |
| `a_r6_cached` | 2.424205 | 2.155958 |
| `a_r6_nt` | 2.468632 | 2.180535 |
| `a_r6_nt_fused` | 2.520674 | 2.220503 |
| `b_128_128_1024` | 2.297005 | 2.046765 |
| `b_128_128_512` | 2.715393 | 2.632664 |
| `b_128_256_1024` | 2.516303 | 2.212586 |
| `b_128_256_512` | 2.526391 | 2.187751 |
| `b_128_64_1024` | 2.267934 | 2.084202 |
| `b_128_64_512` | 2.352829 | 2.052460 |
| `b_256_128_1024` | 2.247577 | 2.010858 |
| `b_256_128_512` | 2.308173 | 1.984514 |
| `b_256_256_1024` | 2.372471 | 2.097381 |
| `b_256_256_512` | 2.418069 | 2.117936 |
| `b_256_64_1024` | 2.209222 | 1.964007 |
| `b_256_64_512` | 2.278946 | 1.968742 |
| `r_b_256_64_1024_1` | 2.158018 | 1.894135 |
| `r_b_256_64_1024_2` | 2.166863 | 1.889993 |
| `r_committed_1` | 2.209759 | 1.921745 |
| `r_committed_2` | 2.206734 | 1.923476 |

The four retained-scratch variants cross the best full-K and blocked implementations with one or two caller-local arenas. They recompute maxima, activation bytes and scales each call, retain local Tensor ownership through team completion, and allocate fresh outputs. First-use allocation is recorded separately. The best retained finalist has 2.172175 ms / 189.82 cold TOPS against 2.205496 ms frozen stateless, median latency ratio 0.984892 and geometric change -2.262% (95% interval [-3.434%, -1.572%]). It fails both required greater-than-5% gain checks. **Reusable state is rejected**, even though its generic partial-improvement statistic passes; that statistic cannot override the retained-state gate.

### Qualification and conditioning

| Pair | Selected cold ms | Committed cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.208178 | 2.245464 | 3.196471 |
| 2 | 2.208672 | 2.241863 | 3.221924 |
| 3 | 2.220709 | 2.241151 | 3.211635 |
| 4 | 2.193512 | 2.254065 | 3.251930 |
| 5 | 2.264644 | 2.258560 | 3.204865 |

Process medians range from 2.193512 to 2.264644 ms, standard deviation 0.027201 ms. No samples or process outliers are excluded. Five separate alternating consuming-chain pairs and five alternating retained/stateless pairs are archived. Process pairs are bootstrap units; within-process windows are not independent observations.

Measurements use Intel Xeon 6972P, CPUs96–191/node1, verified Intel OpenMP/tcmalloc preloads, performance governor, 33 disjoint packed-weight banks, full packed huge-page backing, and 1,612,185,600 intervening packed bytes. Builds, tests, cleanup and diagnostic measurements stop during timing. All per-call preparation, allocation, GEMM, epilogue and fence costs are retained for the selected implementation.

### Native phases and conditional CHA evidence

Native groups retain offcore `config1`, untimed counter initialization followed by restored conditioning, five marked calls, actual 96-worker evidence, positive enabled time and exact running/enabled equality. Clock-only captures are separate. Startup covers N256 (64 full-K tiles per worker), and remaining GEMM N1792 (448 tiles). These spans match accumulation coverage, but full-K retains one pending tile while the blocked kernel retains a 16-tile macroblock. Output publication therefore differs at phase boundaries; whole-worker measurements through final drain/fence compare completed output. Independent event-set medians are not an additive elapsed-time balance.

| Absolute count, million cycles per marked call | Committed startup / remaining | Private256 startup / remaining | Selected startup / remaining |
| --- | ---: | ---: | ---: |
| AMX busy | 25.232 / 176.170 | 25.177 / 176.166 | 25.344 / 176.170 |
| Load bound | 99.800 / 331.206 | 67.507 / 327.385 | 85.642 / 340.984 |
| L2-miss stalls | 73.972 / 111.152 | 38.235 / 112.442 | 60.516 / 115.655 |
| Fill-buffer full | 53.878 / 156.421 | 35.876 / 157.231 | 42.452 / 153.170 |

Selected startup load-bound cycles decrease versus committed, but remaining load-bound cycles increase. Versus private256, 42.852% of positive load-bound excess persists beyond startup, and the bracket-normalized screen gain is 1.561%; both predeclared gates trigger the bounded driverless EMON repeat. Clock-only median worker durations for preparation/startup/remaining/drain are 140.117/401.140/1556.041/0.203 us committed and 149.995/369.851/1535.001/5.354 us selected. These instrumented durations are excluded from acceptance.

All three implementations have five bounded sequential CHA event sets, verified socket-1 topology (120 CHAs), positive clocks/activity, restored conditioning, and durable collector/workload reaping. Same-set demand-hit inserts/CHA clock are 0.006732 committed, 0.006014 private256 and 0.006682 selected; demand-miss inserts/clock are 0.000400/0.000390/0.000378. RFO-hit inserts/clock are 0.000094/0.000093/0.000112. Raw occupancy, inserts and per-CHA distributions remain archived with separate event-set interpretation. ITOM is zero with unvalidated filter activity, so it cannot establish absent output ownership traffic. These measurements do not prove L2 residency, isolate operand traffic, or establish CHA saturation.

### Correctness, generated code and reproduction

All 24 configurations match frozen seed-123 legacy output exactly with and without bias, over all 50,331,648 output elements. Target-shape fixtures cover unequal scales, quantization ties/boundaries, zero/tiny rows, NaN/Inf, strided inputs and bytewise input immutability. Per-build activation and C allocations plus actual-team evidence replace fixed arena-size recognition. Final selected fixtures additionally prove constrained-team and shape fallback. Retained experiments pass changing-input/weight/bias, prior-output immutability, arena reuse and two overlapping callers with actual 96-worker evidence on every call, despite failing performance acceptance.

Disassembly confirms complete ascending K coverage, first-block zeroing, expected partial-C loads/stores, full-K epilogue lifetime, pending macroblock consumption before overwrite, and final drain before tile release/fence. There are no vector/AMX accumulator spills; scalar pointer/control stack traffic remains. Static instruction-site counts are not runtime work reductions. Diagnostic exports and phase instrumentation remain outside production.

The nearest scaled-MM suite passes **26 tests**, covering team/shape fallback, repeated eligible calls, first direct AMX execution, strided inputs and direct/BRGEMM alternation. The cached RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 comparison passes with identical greedy output for two long prompts and 32 generated tokens. Its six-thread execution is **fallback-only**; it does not qualify model execution of the 96-worker specialization. `verification/final_checks.py b_256_64_1024` runs the scaled-MM pytest entry point and cached comparison with the frozen staged library; exact expanded argument vectors and environments are in its terminal records.

Archive: `/home/tianmuli/vllm_test/snc_scaling_analysis/expanded_200_af3cabd16a_20261030`. It retains all candidate sources, seeded/edge verification, raw five-window samples, complete commands, source/library hashes, assembly, native event descriptors/captures, EMON sets, selection and cleanup/restoration records. Selected source SHA256 is `457cd8a07fa05046a5237ad171be95163d5ee62858631ab604e707cccebe8c0a`; measured library SHA256 is `3019776decad6a1df42d8d57d73064f17ab6a38e41fdd739a92e3d801c8681c5`.

### Cleanup and final delivery

Initial cleanup reclaimed 7,226,421,248 allocated bytes and established more than 8 GiB free before building. Final cleanup compressed rejected/diagnostic libraries and objects (3,876,839,424 bytes reclaimed), deduplicated identical new-iteration source/header aliases with matching permissions (317,468,672 bytes), and compressed 76 additional completed historical libraries/objects (3,043,618,816 bytes). Total recorded gross allocated-block reclamation is **14,464,348,160 bytes / 13.47 GiB**, with **8.15 GiB free** at finalization. Gross artifact savings differ from available-space growth because this iteration retains new evidence and builds. Shared-memory tensor cleanup is recorded separately from disk reclamation.

Each compressed payload was decompressed and SHA256-checked before originals were removed; path/inode/alias hashes, compressed hashes and restoration commands are journaled. Candidate source paths remain separately addressable after deduplication. Unmatched temporary mirrors remain untouched because full canonical equivalence was unproven. Sources, raw measurements, generated assembly, original commands/manifests, frozen controls, legacy tensors, all 44 checked link inputs, required builder directories, archival blocked source, runtime/model environments and pre-existing working-tree edits are preserved. Final checks revalidate 13,026 protected inode hashes, 1,424 directories and three symlinks; two pre-existing FIFOs are observed without mutation. Readable active references were checked, while the non-dumpable systemd/PAM/SSH services' `/proc` inspection limitation is recorded. Every owned build, measurement, verification, model and collector process completed and was reaped before cleanup.

`cleanup_manifest.json` retains its original actions and points to immutable final iteration snapshots. `cleanup_iterations.json`, `supplemental_cleanup.json`, `rejected_cleanup.json` and `source_deduplication.json` provide the combined accounting; `CLEANUP_RESTORATION.md` and `FINAL_RESTORATION.md` provide verified restoration commands. The selected library stays directly loadable. Repository pre-commit hooks and `git diff --check` pass, and the integrated C++ SHA256 remains identical to the qualified source. No new repository tests/helpers were needed: existing public-entry coverage plus archived experiment fixtures cover the consequential behavior risks.

Historical results below remain unchanged. This work used AI assistance and has not opened a PR.

---

## Retained production: bounded GEMM locality experiment (2026-10-30)

**Retain `af3cabd16a` and its lower-first paired implementation.** The selected upper-first traversal reaches 2.237864 ms / 184.25 cold TOPS, versus 2.243848 ms / 183.75 TOPS for the frozen committed baseline. None of its five process medians meets the 2.06158430208 ms threshold. Its geometric paired latency change is -0.631%, with process-pair bootstrap 95% interval [-1.172%, -0.181%]. This improvement is below the greater-than-1% integration gate; the median-of-process-medians improvement is only 0.267%. The retained baseline still needs 0.182264 ms / 8.12% lower latency to reach 200 TOPS in this rerun. The original committed qualification was 2.245140 ms / 183.65 TOPS.

Only four configurations were tested: committed lower-first, constant upper-first, ownership-aware local-first, and the conditional 16-ID pairing follow-up. Production C++ is unchanged. No tuning controls enter production, and `VLLM_CPU_SGL_SCHEDULER=single_shape` retains its existing eligibility guards and fallback.

### Traversal screen and standalone qualification

The outer N64-window loop remains intact, and both M128 halves are processed inside every window. Upper-first visits the upper half then lower; local-first starts at quantization rank `(worker/16)/3`. N-window offsets, output rectangles, pending epilogue, barriers, allocation, arithmetic and full-K `32×32` AMX kernels remain unchanged. Fresh committed processes bracket each candidate screen.

| Screen | Cold median ms | Decision |
| --- | ---: | --- |
| Committed brackets, median of three processes | 2.251443 | More than 1% from the fastest screen |
| Upper-first | 2.238657 | Within 1% of local-first; preferred by the fixed tie rule |
| Local-first | 2.222271 | Rejected by the tie rule |
| Upper-first with 16-ID pairing | 2.280568 | Slower than committed brackets 2.245784 / 2.242415 |

The five-pair qualification alternates finalist/committed order, interleaves a fresh legacy control between the arms of every pair, and records five cold and five warm windows per process. Process pairs are the bootstrap units; timing windows inside a process are not independent observations. The controller reference is the frozen committed paired binary, never the old 165.8-TOPS control.

| Pair | Upper-first cold ms | Committed cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.209614 | 2.243848 | 3.185051 |
| 2 | 2.243523 | 2.251080 | 3.231254 |
| 3 | 2.213519 | 2.235913 | 3.226300 |
| 4 | 2.241288 | 2.247510 | 3.211020 |
| 5 | 2.237864 | 2.238061 | 3.232408 |

Candidate process medians range from 2.209614 to 2.243523 ms, with standard deviation 0.016247 ms. Warm medians are 1.938449 ms upper-first versus 1.931309 ms committed (+0.370% latency). Standalone cold timing determines selection.

Every timing process uses CPUs96–191/node1, explicit verified Intel OpenMP/tcmalloc preloads, checked performance governor, 33 disjoint packed-weight banks, full verified packed huge-page backing and 1,612,185,600 intervening packed bytes. Builds, tests, native diagnostics and EMON collection are stopped during timing. Source-matched actual-96-worker arena proofs establish optimized-path eligibility separately; the timing harness itself records requested/Torch thread counts rather than instrumenting the native team. A pre-existing PCM accelerator service remains recorded in the environment inventory.

### Native startup and steady traversal

Native diagnostics initialize worker perf descriptors in one untimed call, retain that initialization record separately, then restore the full 33-bank cold rotation. Five marked calls per group have 32 intervening packed banks; every marked call records 96 actual workers, positive enabled time and exact running/enabled equality. Preparation, tile setup, first N64 window, remaining 31 windows and final epilogue/fence are separate intervals. The private control also records its conversion inside GEMM separately. Counter groups and clock-only captures use separate processes; their medians do not form an additive elapsed-time balance. Diagnostic sources and event descriptors, including offcore `config1`, remain outside production.

| Absolute count per marked call | Committed first | Private first | Upper first | Committed remaining31 | Private remaining31 | Upper remaining31 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMX busy, million cycles | 6.363 | 6.293 | 6.371 | 195.052 | 195.044 | 195.045 |
| Load bound, million cycles | 49.910 | 28.690 | 50.572 | 383.166 | 367.179 | 389.684 |
| L2-miss stalls, million cycles | 46.957 | 23.076 | 45.328 | 139.886 | 132.087 | 143.056 |
| Fill-buffer full, million cycles | 26.879 | 16.292 | 25.656 | 184.351 | 175.678 | 187.885 |
| L2 fills, MiB equivalent | 140.636 | 73.366 | 141.128 | 772.376 | 796.002 | 773.535 |
| Dirty L2 evictions, MiB equivalent | 13.598 | 24.511 | 14.158 | 6.033 | 49.086 | 6.000 |
| Demand modified-peer, MiB equivalent | 6.585 | 0.001 | 6.918 | 0.002 | 0.007 | 0.004 |
| All-read modified-peer, MiB equivalent | 48.333 | 0.132 | 48.337 | 0.026 | 0.234 | 0.036 |

Modified-peer request equivalents are approximately 48 MiB in the first window and nearly absent afterward. These are request counts times 64 bytes, including repeated requests, RFO/code/prefetch for the all-read mask; they are not unique bytes or pure LLC-array bandwidth. Mandatory partner-half reads remain.

Committed paired retains +21.220M first-window and +15.987M remaining-window load-bound cycles versus private256. Upper-first retains +21.882M and +22.505M respectively, with positive total GEMM excess in all five calls and +48.005 MiB-equivalent modified-peer requests. Its steady share of positive excess is 50.702%; committed paired's is 42.967%. Both predeclared gates pass. Clock-only median worker remaining-window durations are 1775.31/1690.72/1783.19 microseconds for committed/private/upper. These instrumented durations are excluded from cold acceptance.

The sole pairing follow-up changes `replica=ngroup/2` and `rank=ngroup%2` while retaining upper-first, the original M/N output rectangles, three replicas, 128 converted rows per worker, single-worker allocation, both barriers and the 48.8 MiB arena. Worker IDs do not establish physical die proximity. Near-pair minus upper-first median load-bound differences are +2.948M first, -1.085M remaining and +3.302M complete GEMM cycles; the five call differences vary in sign. Complete-GEMM and interval medians need not add. Modified-peer equivalents remain 48.438 versus 48.370 MiB, and remaining-window clocks are 1783.91 versus 1783.19 microseconds. No reproducible delivery benefit supports the pairing change.

### Conditional CHA checkpoint

The steady-excess gate triggered driverless EMON. Committed paired, private256 and upper-first each receive five sequential bounded event-group collections after allocation, correctness, warmup and cold conditioning. Socket-1 totals and all 120 per-CHA distributions are retained in `emon/comparison.json`. Actual topology, schedules and activity are validated; hit inserts/clocks and occupancy use separate event sets. Every successful collector and workload is reaped.

| Socket-1 diagnostic | Committed | Private256 | Upper-first |
| --- | ---: | ---: | ---: |
| Demand-hit inserts, billion/s | 1.704 | 1.473 | 1.803 |
| Demand-hit occupancy, billion occupied cycles/s | 89.677 | 39.552 | 102.061 |
| Demand-miss occupancy, billion occupied cycles/s | 25.420 | 24.227 | 28.687 |
| Demand-hit occupancy per-CHA coefficient of variation | 0.123 | 0.080 | 0.128 |
| Same-set hit inserts / CHA clock | 0.006633 | 0.005949 | 0.007418 |

Upper-first retains elevated demand-hit occupancy; it does not remove the cache-delivery signal. These whole-operator captures do not isolate the steady GEMM interval or establish CHA saturation. Occupancy/inserts ratios across separate sets are not simultaneous exact latency measurements. Each group has one sequential process per control and no paired confidence interval. RFO counters are active; ITOM remains zero, so activity of that filter is unvalidated and zero is not proof of absent ownership traffic.

The collector's initial soft descriptor limit is 65,536; EMON subsequently changes its own runtime limit to 94,092, recorded in `fd_validation.json`. No driver was loaded and no persistent system setting changed. Two failed initialization/control attempts are preserved: finite paused loops could expire before conditioning, and `--driverless -resume` initialized a competing session. The successful collector runs driverless with bounded parent lifecycle and plain `-resume`/`-stop` control commands.

### Separate output-consuming chain

Five separate alternating fresh-process pairs measure the operator followed immediately by `torch.sum(result, dtype=torch.float32)`. They retain the same cold-conditioning and preload controls. Chain medians are 2.934177 ms upper-first versus 2.963215 ms committed; geometric paired latency change -0.079%, bootstrap 95% interval [-1.779%, +1.940%]. Warm chain medians are 2.735873 versus 2.749555 ms (-0.498% latency). This diagnostic does not replace standalone cold acceptance.

| Pair | Upper-first chain cold ms | Committed chain cold ms |
| --- | ---: | ---: |
| 1 | 3.068527 | 2.963215 |
| 2 | 2.929343 | 2.947146 |
| 3 | 2.914645 | 2.985892 |
| 4 | 2.934177 | 2.986570 |
| 5 | 2.971299 | 2.944606 |

### Verification and reproducibility

All three new configurations pass frozen seeded legacy comparisons through the real public operator: bias/no-bias, strided input, unequal scales, quantization boundaries, NaN/Inf cases, bytewise immutable inputs and fallback. Each compares all 100,666,464 output elements exactly and records the 51,167,232-byte arena on both eligible target calls. Source review verifies the traversal/pairing-only changes. Generated AMX instruction-site counts remain unchanged across controls: 160 `tdpbusd`, 160 `tileloadd`, 8 `tilestored`, 16 `tilezero`, 2 `ldtilecfg`, 2 `tilerelease`, 34 `vmovntdq`, 2 `sfence` across bias specializations.

The existing `tests/kernels/quantization/test_scaled_mm_kernel_selection.py` suite passes **26 tests** (2 warnings, 302.69 s) against the finalist stage. It covers strided eligible input, shape/team fallback, repeated eligible calls, first direct AMX execution and alternating direct/BRGEMM execution. No new test file or duplicate repeated-call fixture was added. The cached `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` comparison gives exact legacy/single-shape tokens for two long prompts and 32 greedy output tokens. It verifies fallback generation only: six OpenMP threads were configured, while observed spawned threads were bound to CPU0. It does not establish model performance or 96-worker model execution. All test/model processes are reaped; temporary comparison tensors were removed after successful checks.

Required pre-commit hooks and `git diff --check` are recorded in the completed archive. No production kernel, public interface or eligibility behavior changed. The retained source SHA256 is `8b54af658715cbbc940ffc382bc43906b40b069f132e883e55b3cc28330cbf7d`; committed library SHA256 `fbbb24219ac74295276d96164fbfdf1b552f7b3632f92a137fc7eaa745a3ef01`. Finalist library SHA256 is `441fd51d6072410f465dae5e55b783760b4f2fd75932f952feadb6b929f24daa`. `controls.json` binds private256 and legacy sources/libraries too.

Archive: `/home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030`. It preserves source snapshots, binaries, exact compile/link commands, generated-code review, raw samples/counters, initialization records, EMON distributions, failed collection attempts, original/compressed hashes, terminal/reaping records and final selection. `README.md` documents dependencies and restore/reproduction commands. The prior archives remain immutable. AI assistance was used for implementation, measurements and reporting.

```sh
# Archived commands; use fresh output paths/tags for a new measurement campaign.
.venv/bin/python /home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030/experiment.py qualify upper_first --tag fresh_qualification
.venv/bin/python /home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030/profile.py /home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030/upper_first/stage/vllm /home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030/empty.json --reference /home/tianmuli/vllm_test/snc_scaling_analysis/gemm_locality_af3cabd16a_20261030/controls/committed/stage/vllm --chain-only --out /path/to/fresh_chain_pair
/home/tianmuli/vllm_test/vllm_local/.venv/bin/python -m pre_commit run --files docs/getting_started/installation/cpu.md benchmarks/kernels/cpu/single_shape_int8_report.md
```

## Previous production selection: paired quantization replicas (2026-10-30)

**Selected: 2.245140 ms / 183.65 cold TOPS, with 9.841% lower latency than the frozen current control. The 200-TOPS target remains unmet.** All five candidate process medians exceed 2.06158430208 ms; another 0.183556 ms (8.18%) latency reduction is required. The paired process-bootstrap 95% interval for latency change is [-10.364%, -9.239%]. The preselected partial-improvement gate passes.

Qualification uses five alternating candidate/current fresh-process pairs, five cold and five warm samples per process, and a legacy process interleaved in each pair. All runs retain CPUs96–191/node1, actual-96-worker path evidence, explicit Intel OpenMP/tcmalloc preloads, performance governor/EPP, 33 disjoint packed weight banks, full verified packed huge-page backing, and 1,612,185,600 intervening packed bytes. Builds, tests and other measurement workloads are stopped during timing.

| Pair | Paired cold ms | Current cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.265895 | 2.486095 | 3.199072 |
| 2 | 2.251740 | 2.487379 | 3.195180 |
| 3 | 2.227629 | 2.495968 | 3.209089 |
| 4 | 2.233339 | 2.483894 | 3.206980 |
| 5 | 2.245140 | 2.495275 | 3.222466 |

Candidate process medians range from 2.227629 to 2.265895 ms, with standard deviation 0.015161 ms. Median warm latency improves from 2.193797 to 1.945290 ms. Cold performance determines selection.

The separate operator-plus-immediate-`torch.sum(result, dtype=torch.float32)` diagnostic also improves across five alternating process pairs: cold median 3.081022 ms paired versus 3.137695 ms current, geometric latency change -2.286%, paired-bootstrap 95% interval [-3.079%, -1.486%]. Warm chain medians are 2.893823 versus 2.919474 ms. This chain is diagnostic and is never mixed with standalone acceptance timing; other consumers can behave differently.

Required pre-commit hooks pass on all four changed files, including test tethering, Ruff, clang-format, rendered Markdown and SPDX checks; `git diff --check` passes. Markdown formatting was applied before the final hook run.

### Implementation and rejected alternatives

The selected implementation computes each exact BF16 row maximum once, then quantizes three cached replicas per 256-row M partition. Each replica is shared by two GEMM workers whose IDs differ by 48; each converts 128 disjoint rows, then an explicit team barrier publishes both halves and local output scales. The arena is 51,167,232 bytes (48.8 MiB), versus 102,334,464 bytes for the private256 prototype. Allocation is inside the timed eligible call, performed by worker zero after the actual-96-worker guard. The shared fallback activation allocation remains reserved but its quantized activation area is unwritten on this path.

The `16x6` ownership, `128x64` traversal, direct `32x32` full-K kernel, packed N32 weights, compensation, separate scaling divisions, bias and output arithmetic are unchanged. Cached activation stores win over streaming conversion; output streaming stores and their fence already existed. The only GEMM addressing changes make Aq and scales local to the paired M partition. No tuning environment variables, persistent arena or public weight-format change is introduced.

Quiet fresh-process screens were current 2.493595 ms, private256 2.306115 ms, paired cached 2.235173 ms and paired streaming 2.268706 ms. A single `64x16` geometry was implemented and rejected: 2.370077 ms cold, 2.126165 ms warm. It halves B tile loads but doubles A tile loads, increasing total operand tile loads by 25% for equal output. Generated code retains accumulator tiles without unexpected spills. Separate whole-operator profiles show unchanged AMX busy work (201.5 versus 201.4 million cycles), but load-bound cycles rise from 459.7 to 526.6 million and L1-miss stall cycles from 434.5 to 505.0 million. L1 replacements rise from 213.6 to 267.0 million, consistent with greater operand-load pressure; L2 fills do not rise. These groups use separate processes and differing effective frequency, so they support the delivery-cost explanation without isolating a single instruction as its cause. This experiment does not justify further broad geometry, prefetch or K-blocking sweeps.

The first paired revision exposed a concurrent Tensor assignment: an OpenMP single directive inside the dispatch macro did not guard allocation in generated BF16 code. Native traces establish the race, and the repeated-call fixture fails on its third call after two exact outputs. An ordinary `if (execute.worker == 0)` plus the existing maximum-pass barrier fixes publication. Both corrected variants pass repeated actual-path calls. Failed binaries remain diagnostic artifacts, never selection controls.

### Where the time remains

Separate native phase groups preserve absolute aggregate core counts per operator call; the GEMM interval includes the overlapped output epilogue. Counts below are independent group medians, not an additive latency balance. All phase intervals have positive enabled time and exact running/enabled equality. First-use perf setup perturbs the first barrier graph substantially; raw calls are retained, while steady timing comes from later calls and a clock-only cross-check.

| Aggregate count per call | Current | Private256 | Paired cached |
| --- | ---: | ---: | ---: |
| Whole-worker cycles, million | 674.57 | 615.48 | 588.73 |
| Whole-worker load-bound cycles, million | 544.91 | 458.78 | 453.81 |
| Conversion cycles, million | 122.30 | 115.70 | 29.18 |
| Conversion fill-buffer-full cycles, million | 101.09 | 31.76 | 6.73 |
| GEMM cycles, million | 542.81 | 486.67 | 537.58 |
| GEMM load-bound cycles, million | 438.27 | 382.61 | 433.68 |
| GEMM AMX busy cycles, million | 201.34 | 201.34 | 201.41 |
| GEMM L2-miss stall cycles, million | 181.55 | 129.87 | 187.94 |

Current conversion includes maxima discovery; the other two have a separate roughly 6–7 million-cycle maximum pass. Paired conversion instructions are about half private256's, matching three copies instead of six. The absolute whole-worker load-stall reduction is established, but most of it is preparation: paired GEMM remains close to current and slower than private256, despite the same AMX work. Representative later graphs show maxima about 38–39 µs, conversion 104–116 µs, GEMM about 2.02–2.07 ms, plus two short barriers. These instrumented intervals do not reconstruct acceptance timing.

Separate socket IMC captures measure paired reads/writes of 64.69/113.44 MiB per call, versus current 76.64/115.92 MiB, with paired combined bandwidth about 76.43 GiB/s. Socket-wide counters include background work; they do not suggest DRAM saturation. Whole-operator offcore cache-hit/peer-forward request equivalents are 1048.35 MiB paired versus 869.23 MiB current, and L2 fills 1125.36 versus 937.55 MiB-equivalent. Thus lower elapsed time does not imply less aggregate cache traffic. Offcore events include RFO and prefetch and cannot be equated to the side conversation's sequential LLC-array bandwidth ceiling.

GEMM operand delivery remains the main opportunity. A next bounded experiment should preserve paired preparation while changing traversal to consume each worker's locally converted 128-row half first, comparing original order and opposite pair-member orders. Advance only if absolute GEMM load-bound/L2-stall cycles fall and cold process qualification improves. Private GEMM demonstrates potential cache-state headroom but its sixfold conversion and downstream dirty-cache pressure make it a poor replacement. Further quantization-only tuning cannot be assumed to close the full gap.

### Output-consumer slowdown mechanism

Same-call diagnostic controls isolate operator, BF16-to-FP32 cast, reduction and release. With a preallocated, initialized FP32 destination, cold casts take 0.262 ms current, 0.545 ms private256, and 0.265 ms paired cached. Read-only native BF16 scalar reduction stays near 0.217–0.221 ms across variants, though its numerical semantics differ from FP32-output reduction. Allocating casts are 0.264/0.681/0.364 ms respectively. Allocation traces show the FP32 destination reusing the freed scratch starting address in both prototypes; that changes destination cache history but cannot alone explain private256's persistent preallocated penalty.

Cast-region profiles give nearly equal L2 fills: 297.07/296.16/296.83 MiB-equivalent per cast. Modified L2 evictions are 107.37/201.25/106.55 MiB-equivalent, and fill-buffer-full counts are 41.48/148.30/41.16 million cycles for current/private/paired. Matched unrelated-source controls also reproduce the private excess: about 201.6 versus 107.5 MiB-equivalent modified evictions, with equal fills. Cycles/reference cycles indicate essentially identical frequency in these captures. This supports FP32 destination fills displacing persistent modified cache state left by private preparation. It does not prove that every victim is activation scratch; counters do not tag addresses. The unrelated-source penalty is smaller than the result-source penalty, so output locality also matters.

FIFO acknowledgements cost 1.5–1.9 ms per profiled call and ACTIVE worker waiting contaminates aggregate region cycles/instructions. Gated casts retain the private penalty, but these captures support a mechanism rather than complete causal isolation. Modified-line evictions are L2-fill-triggered writebacks to L3, not DRAM writes. Offcore masks include RFO, prefetch, code and peer forwards; their line equivalents are not pure LLC-array bandwidth or unique DRAM bytes. Phase counters are likewise not a quantization DRAM-bandwidth measurement.

### Verification and provenance

The corrected candidates and rejected geometry pass 1,207,975,392 exact output-element comparisons against frozen legacy, including NaN positions, unequal scales, bias/no-bias, quantization boundaries, strided BF16 input, mismatched shapes, actual-team fallback, first direct AMX execution and alternating direct/BRGEMM calls. All inputs are bytewise unchanged. Eighteen observed actual-96-worker calls establish execution of the paired path. The selected candidate passes all 26 nearby scaled-MM/kernel-selection tests. The nearest scheduler test repeats eligible calls to cover the allocation/reuse failure and checks bytewise immutable inputs.

Cached Llama generation matches two 32-token sequences exactly. Its shapes exercise fallback. Six threads were requested, but spawned affinity effectively restricted execution to one physical core; this is an accuracy comparison, not a six-core performance claim or specialized-model qualification.

Selected GEMM source SHA256: `8b54af658715cbbc940ffc382bc43906b40b069f132e883e55b3cc28330cbf7d`. Selected native library SHA256: `fbbb24219ac74295276d96164fbfdf1b552f7b3632f92a137fc7eaa745a3ef01`. Frozen current source/library hashes: `e68499f88763e268b1fe4b0284d8709492800af5a1bb5d2ce2ce0755fe875d15` / `6dbf65f82da0d11b9e736da09715848b5babc67bc6effce37300a16799688b91`. Controls start from commit `64db699a4fb088de079d2d3eea2eb906dd5fa3d8` plus the initially uncommitted files preserved in the archive. No old archive or control was modified.

Reproduction sources, exact compile/link commands, native stages, source snapshots, generated-code review, raw timings/counters and terminal/reap audits are in `/home/tianmuli/vllm_test/snc_scaling_analysis/paired_quant_64db699a4f_20261030`. The archive depends on the prior immutable `private_quant_64db699a4f_20261030` controls and 44 frozen native link inputs. `experiment.py`, `run_remaining.py`, `diagnostics/PHASE_README.md` and `consumers/profiles/REVIEW.md` describe timing and profiling. The temporary path remains a symlink for recorded-command reproduction.

The historical experiment sections below describe their then-current controls and are superseded by this production selection.

## Experiment: shared maxima and private quantization (2026-10-30)

**The 256-row private panel wins standalone timing at 2.299434 ms / 179.31 cold TOPS, but the output-consuming chain regresses. Production remains unchanged.** Five alternating candidate/current process pairs establish a 7.464% geometric latency reduction, with a paired process-bootstrap 95% interval of [-7.739%, -7.188%]. The candidate passes the preselected partial-improvement gate. It does not meet 200 TOPS: every process exceeds 2.06158430208 ms; another 0.237850 ms (10.34%) latency reduction is required.

The current control is the existing working-tree specialization, including its previously qualified row-buffer/streaming-copy change, rather than HEAD alone. Its GEMM source SHA256 is `e68499f88763e268b1fe4b0284d8709492800af5a1bb5d2ce2ce0755fe875d15` and library SHA256 is `6dbf65f82da0d11b9e736da09715848b5babc67bc6effce37300a16799688b91`. Driver label `committed` refers to this frozen current control. The candidate library SHA256 is `e1c99a8df5b3706fe5e80927b8514c33b6878a0da481c476399c5b71448dc82b`. The original uncommitted production GEMM and CPU-installation documentation changes were preserved.

### Six-way screen

Five cold and five warm samples per fresh process, after builds/tests/model checks completed. Earlier concurrent-work screens, including an unusually slow baseline process, remain in the archive and are not combined with this quiet screen or qualification. No samples were excluded within any run.

| Configuration | Cold ms | Warm ms | Additional private arena |
| --- | ---: | ---: | ---: |
| Current baseline | 2.495001 | 2.198133 | 0 |
| Shared split, 256 rows | 2.532749 | 2.264189 | 0 |
| Shared split, 128 rows | 2.838763 | 2.507771 | 0 |
| Private, 256 rows | 2.312863 | 1.938314 | 97.6 MiB |
| Private, 128 rows | 2.471532 | 2.101655 | 48.8 MiB |
| Adjacent A readers, shared Aq | 4.339378 | 1.930639 | 0 |

The shared-split controls isolate the extra maximum pass/barrier from private conversion. The two 128-row variants use identical M-panel-outer traversal and pending-epilogue flush boundaries. Adjacent placement maps physical worker `p` to logical worker `(p / 6) + 16 * (p % 6)` while retaining quantization ownership. It improves warm execution and quantization, but substantially worsens cold GEMM; it is rejected under the cold criterion. The smaller private panel is also rejected in favor of the faster 256-row panel. Neither shared split is beneficial.

### Selected experimental implementation

Compute each row's maximum once into the existing shared scalar area, preserving both FP32 maximum recurrences, operand order, reduction and epsilon. After that dependency barrier, each GEMM worker derives `amax / 127` and `127 / amax` separately, converts its own 256-row M partition once into cached private INT8 storage, and reuses it across its 2048 assigned N columns. Conversion occurs six times per global activation row. Shared Aq remains unwritten; the existing fallback allocation remains reserved.

A tensor-backed arena is allocated inside every timed operator call before OpenMP. Its 96 exclusive, 64-byte-aligned segments contain rows at stride 4160 and local output scales. The 256-row arena is 102,334,464 bytes; the 128-row arena is 51,167,232 bytes. No persistent scratch cache, untimed initialization, public selector or weight-format change is introduced. Both private variants retain compensation and full-K arithmetic, release tile state and fence output stores. The 128-row variant finishes its deferred epilogue and clears pending state before overwriting scratch or scales.

Generated code preserves separate divisions, maximum operand ordering, rounding and byte packing. Conversion helpers inline into GEMM, use cached scratch stores, and introduce no vector spills in bias/no-bias GEMM specializations. Tile configuration/release counts stay at 96 per call. Both 128-row variants add a second pending-output flush per worker without changing the 49,152 tiles per operator. The prototype allocates its full arena for exact-shape calls that later fall back because of a smaller actual team, and constructs a zero-sized arena tensor for other shapes; fallback output checks do not establish fallback latency.

### Qualification and downstream consumption

All processes use physical CPUs 96–191, NUMA node 1, 96 requested workers, explicit Intel OpenMP/tcmalloc preloads, performance governor/EPP, ACTIVE wait policy, 33 disjoint weight banks, 100% verified packed-weight huge-page backing and 1,612,185,600 intervening packed bytes. Instrumented records establish 96 distinct actual workers. Five cold samples and five warm samples are recorded per acceptance process; a frozen legacy process is interleaved in each pair.

| Pair | Candidate cold ms | Current cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.299434 | 2.484869 | 3.205299 |
| 2 | 2.296528 | 2.495276 | 3.203036 |
| 3 | 2.303970 | 2.481417 | 3.204811 |
| 4 | 2.311023 | 2.487342 | 3.224053 |
| 5 | 2.297187 | 2.487428 | 3.212732 |

Candidate process-median range is 2.296528–2.311023 ms, standard deviation 0.006006 ms. Median warm latency is 1.926183 versus 2.195904 ms.

An immediate full BF16 output sum into FP32 measures consumption separately from standalone acceptance. Its initial pair suggested a cold regression, so five alternating fresh-process pairs were run. All five pairs regress cold: process-median medians are 3.354985 versus 3.129213 ms, with a paired geometric regression of **5.976%**, 95% interval [4.534%, 7.439%]. Warm chain medians are 2.977891 versus 2.908952 ms. This diagnostic does not establish a regression for every possible downstream consumer, but the standalone gain does not carry through this chain.

### Phase and counter interpretation

Separate instrumented binaries record 25 calls and 96 workers per configuration. Times below are medians across worker-call observations; repeated 128-row conversion and GEMM intervals are summed within each worker-call before taking its median. Columns cannot be added to reconstruct operator latency. Timestamp/fault-query instrumentation perturbs execution, and AMX overlaps the epilogue.

| Configuration | Maximum pass µs | Conversion µs | AMX plus overlapped epilogue µs |
| --- | ---: | ---: | ---: |
| Current baseline | fused | 397.5 | 1909.6 |
| Shared split, 256 rows | 20.9 | 330.6 | 1916.9 |
| Shared split, 128 rows | 20.9 | 360.4 | 2209.4 |
| Private, 256 rows | 21.4 | 418.1 | 1732.8 |
| Private, 128 rows | 20.8 | 165.0 | 2117.6 |
| Adjacent A readers, shared Aq | fused | 109.7 | 4000.0 |

Private 256-row preparation is not faster: it adds the shared maximum pass and performs much more conversion per worker. Its benefit is the shorter mixed GEMM/epilogue region. Private 128-row conversion is faster, but the longer weight reuse distance increases the mixed region and cache traffic. This constrains the explanation without attributing the entire gain to one coherence queue.

Separate core/cache/stall groups report absolute counts per call and 100% counter coverage. Cache-group L2 fill traffic estimates are 1.002 GB for current, 1.155 GB for private256 and 1.891 GB for private128, counting repeated fills rather than distinct data or DRAM traffic. Private256 therefore increases total L2 fills while reducing demand misses and waiting. AMX busy counts remain approximately 201.34 million cycles per call. Core-group load-bound cycles fall from 541.05 to 467.53 million; L2-miss stalls fall from 214.27 to 189.73 million in their separate group. These overlapping events cannot be added.

The first baseline FB-full profile had extra runtime/spin variability. A separate quiet repeat resolves that confound:

| Quiet FB-full group | Current, million cycles/call | Private256, million cycles/call |
| --- | ---: | ---: |
| Fill-buffer full, raw 0x0248 cmask 0 | 308.458 | 228.194 |
| Load-bound | 542.891 | 468.493 |
| L1-miss stalls | 447.398 | 437.741 |
| AMX busy | 201.338 | 201.335 |

Absolute FB-full cycles decrease 26.02% and load-bound cycles decrease 13.70% in this repeat. This supports reduced waiting over the complete operator; it does not isolate quantization or prove an invalidation/admission-queue mechanism. Validated socket-1 CHA local requests are measured separately: current/private256 local reads are 1.123/0.926 million per call and local writes 1.978/1.853 million. These broad events count requests rather than cache-fill bytes, invalidations or WCILF residence. All measured worker phase/fault windows report zero minor faults. Scratch VMA backing and NUMA records are retained; VMA aggregates do not prove exact arena huge-page coverage.

No 480 GiB/s DRAM quantization claim is made. Resident private conversion, sixfold conversion work and the shared maximum pass are different traffic scopes from the original 48 MiB useful-traffic metric.

### Verification, recommendation and reproduction

All six variants match legacy across 251,664,576 compared output elements per variant, including unequal channel scales, bias/no-bias, strided BF16 input, boundary values, NaN/Inf patterns, smaller actual teams, mismatched shapes, first direct AMX execution and alternating AMX/BRGEMM. Finite outputs use zero tolerances; NaN outputs match by position with `equal_nan=True`, without checking their payload bits. Inputs remain unchanged at the bit level. Split conversion matches 278,462,464 INT8 bytes and 67,984 raw scale-bit values, including every BF16 encoding. The nearest scheduler fixture adds distinct scales around 128-row panel reuse boundaries. The selected candidate passes all 26 scaled-MM kernel-selection tests. Cached Llama generation matches both 32-token outputs; it exercises fallback. Required hooks and archive checks are recorded with the artifacts.

Retain private256 as the measured standalone winner and a reviewable experimental patch. Retire the other four candidates under the cold criterion. Its 97.6 MiB arena and confirmed consuming-chain regression are material tradeoffs. The experiment leaves production unchanged; it does not justify a general performance claim or claim 200 TOPS.

Persistent artifacts: [/home/tianmuli/vllm_test/snc_scaling_analysis/private_quant_64db699a4f_20261030/README.md](/home/tianmuli/vllm_test/snc_scaling_analysis/private_quant_64db699a4f_20261030/README.md). They include all six source/library snapshots, five patches, hashes, 44 frozen link inputs, frozen legacy control, raw initial and quiet screens, five standalone qualification pairs, five consuming-chain pairs, phase timestamps/faults/mappings, independent measurement review, raw perf/CHA encodings and coverage, correctness/model/hook logs and reproduction commands. `/tmp/sgl-private-quant-20261030` resolves to the archive so recorded commands remain valid.

## Retained production result: quantization pipeline (2026-10-30)

**The 480 GiB/s quantization target was not reached.** The retained partial improvement measures **2.513397 ms / 164.05 cold TOPS**, against **2.552284 ms** for source baseline `64db699a4fb088de079d2d3eea2eb906dd5fa3d8` in the same qualification run.
The paired geometric latency change is **-1.553%**, with a paired process-bootstrap 95% interval of **[-2.021%, -1.158%]**. This passes the preselected partial-improvement gate: aggregate candidate median at least 1% lower and the paired upper bound below zero. None of the five processes meets the 200-TOPS limit of 2.06158430208 ms.

### Retained change

Only the guarded exact-shape, 96-actual-worker quantization store path changes. Each productive worker uses the unchanged row quantizer into an aligned private 4 KiB stack buffer, then copies its 4096 bytes with 64 aligned 64-byte streaming stores. One worker fence precedes the existing GEMM barrier. The row buffer adds 384 KiB across 96 workers and is reused within the call; no allocation, page preparation, or arithmetic is moved outside operator timing. The existing heap allocation, 4160-byte row stride, balanced row partitions, quantization arithmetic, public weights, compensation, AMX loop and output epilogue remain unchanged. Defaults and fallback remain unchanged.

Generated code retains both FP32 maximum recurrences and their operand order, horizontal reduction, scale divisions, rounding and integer conversion. The copy loop emits full ZMM `vmovntdq` stores and `sfence`, with no quantization register spills. Normalized bias/no-bias AMX instruction streams match baseline structurally (817/807 instructions, 94 static tile instructions each). No new production helper, tuning control or compiler flag is added.

### Qualification

Five alternating candidate/current fresh-process pairs, five cold and five warm samples per process, with one interleaved frozen legacy control per pair. Physical CPUs 96–191, NUMA node 1, 96 workers, explicit Intel OpenMP/tcmalloc preloads, performance governor/EPP audits, 33 disjoint packed-weight banks and verified 100% packed huge-page backing are retained. Intervening packed weights exceed 1.5 GiB. No samples were excluded. Node 0 handles tests/builds/model checks; socket 1 measurements run sequentially.

| Pair | Candidate cold ms | Current cold ms | Legacy cold ms |
| --- | ---: | ---: | ---: |
| 1 | 2.504523 | 2.547750 | 3.208818 |
| 2 | 2.513397 | 2.544150 | 3.301655 |
| 3 | 2.491066 | 2.552284 | 3.203769 |
| 4 | 2.523940 | 2.562383 | 3.221851 |
| 5 | 2.532281 | 2.556703 | 3.221504 |

Candidate process-median standard deviation is 0.016160 ms; range 2.491066–2.532281 ms. Warm median is 2.235358 versus 2.299980 ms (-2.81%).

### Quantization bandwidth and mechanism

Useful traffic is 4096×4096×3 = 50,331,648 bytes (48 MiB): one logical BF16 input read and one byte output. At 480 GiB/s this requires **97.65625 µs**. This is useful bandwidth, not measured DRAM traffic: the unchanged two-pass quantizer logically reads input twice. Row staging adds one private scratch write and read, making the selected logical traffic 7 bytes/value rather than baseline 5; most extra scratch traffic stays in L1.

Separate instrumented binaries record earliest worker quantization-work entry through latest work completion, including the copy and fence. Five alternating candidate/baseline process pairs each record 25 calls. These diagnostic clocks exclude dispatch/barrier and fault-query overhead and are not acceptance timings.

| Process | Candidate µs | Candidate useful GiB/s | Baseline µs | Baseline useful GiB/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 451.378 | 103.85 | 507.684 | 92.33 |
| 2 | 448.661 | 104.48 | 502.311 | 93.32 |
| 3 | 449.383 | 104.31 | 502.845 | 93.22 |
| 4 | 450.556 | 104.04 | 502.456 | 93.29 |
| 5 | 447.653 | 104.71 | 507.857 | 92.30 |

Median of process medians: **449.383 µs / 104.31 GiB/s**, versus 502.845 µs / 93.22 GiB/s. All 250 calls and 24,000 worker quantization windows report zero minor faults. Neither alignment/huge-page advice nor moving allocation work explains the actual fused slowdown.

A separate native control retains the same rows and padded Aq storage. Owner-private resident quantization is fast; after six workers read each 256-row Aq partition, cached writes slow substantially. The reader touches every cache line and barriers before quantization. This reproduces the store-ownership constraint without AMX. Reused-output medians from the final frozen native binary:

| Native control | Owner-private µs | After six shared readers µs |
| --- | ---: | ---: |
| legacy | 47.296 | 461.503 |
| minimum | 44.219 | 531.556 |
| copy | 22.614 | 423.209 |
| max | 22.556 | 25.207 |
| convert | 40.307 | 435.361 |

Separately gated native controls show demand RFO requests rising from 1,999 to 191,837 per iteration, and L2 RFO misses from 1,971 to 130,743. These are requests, not cycles or bytes. The counter region includes the 96 MiB shared-reader loads, barriers and metadata per iteration, so its aggregate cycles are not pure quantization cycles. All events have 100% coverage. Resident controls cannot establish socket-wide DRAM bandwidth.

### Quantization-only counter follow-up

Per-worker `perf_event_open` groups run continuously, with group reads directly before and after each quantization loop. Each of four controls uses one fresh process per group and 100 samples. Counts exclude shared-reader setup, cache preparation, AMX, barriers and kernel/hypervisor execution. User syscall-wrapper instructions and timestamp fences remain inside the counter boundaries; work timestamps exclude the counter reads. All 96 workers and every sample have 100% enabled/running coverage.

| Quantization-only event per call | Private legacy | Shared legacy | Private streaming | Shared streaming |
| --- | ---: | ---: | ---: | ---: |
| Execution Mcycles | 8.584 | 112.236 | 11.450 | 107.494 |
| Store-bound Mcycles | 0.000 | 56.177 | 0.000 | 0.379 |
| Store-buffer-full Mcycles | 0.000 | 56.221 | 0.000 | 0.383 |
| Cycles with pending RFO, Mcycles | 0.145 | 111.220 | 0.152 | 2.603 |
| Demand RFO requests | 461 | 191,623 | 388 | 750 |
| L2 RFO misses | 457 | 130,514 | 388 | 737 |
| Load-bound Mcycles | 0.230 | 46.198 | 2.219 | 95.478 |
| L2-miss stall Mcycles | 0.002 | 0.473 | 0.004 | 0.596 |
| L3-miss stall Mcycles | 0.000 | 0.011 | 0.001 | 0.002 |
| Fill-buffer-full Mcycles | 0.000 | 46.892 | 2.850 | 96.259 |
| False address dependency events | 0 | 8 | 1 | 3 |
| Blocked store-forwarding events | 0 | 0 | 3 | 4 |
| Memory-ordering machine clears | 0 | 1 | 427 | 411 |

Streaming removes most demand RFOs and conventional store-buffer stalls, but the shared output still stalls the quantization pipeline: fill-buffer-full cycles are **96.259 Mcycles / 107.387 M execution cycles (89.6%)**, versus 2.850 / 11.480 Mcycles (24.8%) with private output. The buffer-group process medians are 446.471 versus 52.307 µs for streaming, and 435.139 versus 44.556 µs for legacy. False address dependencies and store-forwarding blocks are negligible. Memory-ordering machine clears increase with streaming, but event counts alone do not establish their cycle cost.

This is strong evidence for cache-coherent write delivery and limited fill-buffer capacity as the remaining constraint in the reproduced shared-operand state. Replacing the maximum reduction cannot remove this constraint. The selected kernel is not simply saturating socket DRAM bandwidth: the same arithmetic/store loop is fast with private output, input-only maximum is fast under shared-output setup, and L2/L3 load-miss stall cycles remain small. These controls do not isolate a particular coherence response, LLC slice, or exact DRAM rate during fused quantization. Native standalone controls also do not predict the full-operator gain; only the repeated operator qualification selects production.

The fill-buffer cycle event uses `L1D_PEND_MISS.FB_FULL`, event 0x48 / umask 0x02 / **cmask 0**. The whole-operator group uses cmask 1, which counts full-buffer periods; its derived summary therefore reports a count rather than a stall-cycle fraction. Raw encodings, per-worker deltas, event-catalog provenance and this unit correction are retained in the archive.

### Shared-output root-cause follow-up

Further controls narrow the bottleneck to **placement-dependent shared-output ownership and uncore write handling**. Fill-buffer pressure is the observed backpressure, rather than a sufficient explanation of its source. This investigation changes no production kernel code or system settings.

The native control divides output among W exclusive writers and maps F readers per writer as `(writer + offset + j*step) % 96`. Every reader loads every logical cache line and finishes before the writer barrier. Default step is `96/F`; six readers spaced by 16 correspond to the current AMX A-sharing worker spacing. Explicit step 1 clusters readers on adjacent physical-core IDs, without assuming physical mesh adjacency. Constant full-line store controls remove all input loads, quantization arithmetic, scale writes and scratch writes.

| 4096-row native control | No readers, µs | One remote reader, µs | Six spaced readers, µs |
| --- | ---: | ---: | ---: |
| Pure streaming stores | 41.343 | 46.818 | 427.212 |
| Pure cached stores | 22.260 | 34.500 | 421.529 |
| Selected row-staged quantizer | 53.231 | 65.771 | 449.174 |
| Batch-staged quantizer | 70.911 | 77.163 | 460.338 |

Both store types reproduce the slowdown. Staging all rows before streaming does not rescue it. Two spaced readers remain fast; three and four trigger sharp increases. Fanout also changes reader spacing, so those results do not establish a fixed hardware sharer-count threshold.

Holding six readers fixed demonstrates a substantial placement effect. Five alternating fresh-process pairs with explicit NUMA-node-1 memory binding and 50 samples per process confirm:

| Native control | Readers spaced by 16, µs | Adjacent readers, µs |
| --- | ---: | ---: |
| Pure streaming stores | 427.663 | 140.774 |
| Selected quantizer | 447.084 | 146.254 |

These are medians of process medians. Selected-quantizer process ranges are 445.703–448.610 and 143.538–146.655 µs respectively. A separate five-pair owner-first-touch run gives the same result. No samples were excluded.

Writer scaling keeps 32 rows per writer, avoiding a changing per-core cache footprint. With six spaced readers, pure streaming stores take 53.282/106.449/160.634/212.972/323.072 µs at 16/32/48/64/96 writers: approximately **39 GB/s logical output** throughout. Unshared times are 11.206/12.131/19.263/22.833/32.109 µs. The slowdown depends on both sharing and concurrent writers, supporting a socket-wide throughput bottleneck. Passive OpenMP barriers produce large wakeup skew and are excluded from this inference; active-policy per-worker timestamps confirm that shared writes themselves take hundreds of microseconds.

The streaming-store opcode is positively identified by Intel GNR's **WCILF** events. Every tested NT layout issues exactly **262,144 WCILF requests per call**, including selected quantization. Paired `TOR_OCCUPANCY.IA_WCILF` and `TOR_INSERTS.IA_WCILF` counts give observed mean request-table residence in CHA cycles:

| Pure streaming-store layout | Write µs | Mean TOR residence, CHA cycles |
| --- | ---: | ---: |
| No readers | 41.194 | 514.4 |
| Two readers, step 48 | 49.939 | 594.6 |
| Four readers, step 24 | 605.870 | 1,878.7 |
| Six readers, step 16 | 427.774 | 1,521.1 |
| Six readers, step 1 | 141.246 | 895.1 |

For selected quantization, step 16 measures 447.207 µs / 1,672.5 CHA cycles versus 145.534 µs / 852.6 cycles at step 1. All 120 CHAs have higher residence than the unshared control, while insert counts remain broadly distributed (3.4–3.9% coefficient of variation). Absolute pure-NT fill-buffer-full cycles rise from 7.596 to 116.014 Mcycles per call with six spaced readers; selected-quantizer cycles rise from 2.817 to 98.052 Mcycles. Every core/CHA group has 100% enabled/running coverage, and every collected native timing has zero page faults and valid writer affinity.

CHA sample-block gating includes reader preparation and barriers but excludes startup and warmups; the exact WCILF insert count attributes these requests to timed NT output. Occupancy uses event 0x36, inserts 0x35, low mask 0x01 and extended mask 0x00C867FF. Occupancy is constrained to counter 0 and appears first in each group. Uncore clocks vary during gating/setup, so residence is reported in cycles rather than a precise nanosecond conversion. It measures time inside TOR, not total store latency.

The earlier ItoM selector does not observe these NT requests. The particular `OSB.RFO_HITS_SNP_BCAST` event remains zero; **snoop broadcast or compressed-directory overflow is not established**. Positive WCILF and broad request controls validate counter sensitivity and socket routing. Longer TOR residence, fewer completed writes per unit time, placement dependence and writer saturation support an uncore write-handling constraint, but do not identify its exact admission, snoop-service or downstream queue.

The next bounded production experiment is to map the six A-sharing GEMM workers to adjacent cores while preserving the 16×6 work assignment and exclusive outputs, then measure quantization and AMX separately and qualify the full cold operator. The native 146-µs result does not establish a full-operator gain, 200 TOPS, or 480 GiB/s DRAM bandwidth.

Sources, controls, raw samples, counters, independent review and replay scripts are preserved in `/home/tianmuli/vllm_test/snc_scaling_analysis/quant_sharing_64db699a4f_20261030/README.md`.

### Bounded screening and nonfinite semantics

Fifteen configurations were screened in fresh processes using five cold samples and full frozen-legacy output verification before timing. Each row reports its contemporaneous committed control; screen winners are not qualified results.

| Configuration | Cold ms | Control ms | Disposition |
| --- | ---: | ---: | --- |
| `aligned` | 2.554639 | 2.539933 | No useful cold benefit |
| `baseline_huge` | 2.535022 | 2.539933 | No useful cold benefit |
| `huge` | 2.551914 | 2.539933 | No useful cold benefit |
| `demote` | 2.617210 | 2.547991 | No useful cold benefit |
| `rotating8` | 2.523303 | 2.547991 | Unstable; warm regression; more scratch |
| `ownership` | 2.529652 | 2.530280 | No useful cold benefit |
| `stream` | 2.574006 | 2.530280 | No useful cold benefit |
| `write_prefetch2` | 2.533224 | 2.537326 | No useful cold benefit |
| `copy_direct` | 2.486786 | 2.537326 | Within 0.1% of selected screen; more complexity |
| `write_prefetch8` | 2.534611 | 2.537326 | No useful cold benefit |
| `minimum2` | 2.567778 | 2.535964 | No useful cold benefit |
| `rotating4` | 2.545941 | 2.551198 | No useful cold benefit |
| `rotating` | 2.547521 | 2.539669 | No useful cold benefit |
| `copy_cached` | 2.519279 | 2.538257 | Below 1% gate |
| `copy_stream` | 2.484844 | 2.538257 | Selected and qualified |

`minimum2` uses an unsigned BF16 integer magnitude maximum; `aligned`/`huge` add aligned/advised scratch; `baseline_huge` retains legacy arithmetic; `ownership` matches producer rows to 16×6 consumers; `stream` streams 32-byte quantizer stores; `copy_cached`/`copy_stream` stage unchanged arithmetic with cached/streaming full-line copy; `rotating`/`rotating4`/`rotating8` use 2/4/8 Aq slots within each call; `demote` demotes consumer lines; `write_prefetch2`/`write_prefetch8` prefetch write ownership; `copy_direct` uses MOVDIR64B. These alternatives are absent from production.

Unsigned sign-cleared BF16 codes correctly order finite magnitudes and infinity. NaNs exceed `0x7f80` and cannot simply join the integer maximum: legacy `MAXPS` can forget an earlier maximum after a NaN in the same lane recurrence, or propagate NaN during reduction. For example, successive lane values 10, NaN, 1 can leave 1. The experimental integer helper detects NaNs and reruns the unchanged quantizer; infinities retain the original FP conversion behavior. Exhaustive native testing covers all 65,536 BF16 codes and signed NaN payload/order patterns, comparing 278,462,464 Aq bytes and 67,984 scale patterns with zero mismatches. Its full-operator timing was slower, so the selected implementation retains the original arithmetic directly and needs no nonfinite fast-path logic.

### Separate counters and downstream consumption

Each counter group is collected in a separate fresh process, normalized by timed calls, with at least 99% coverage. Cycle fractions overlap; changes below use absolute cycles/call.

| Event, Mcycles/call | Candidate | Current |
| --- | ---: | ---: |
| Execution cycles | 728.682 | 732.180 |
| AMX busy | 201.338 | 201.362 |
| Load bound | 542.386 | 505.265 |
| L1 miss stalls | 446.882 | 426.127 |
| L2 miss stalls | 212.972 | 193.974 |
| L3 miss stalls | 10.800 | 9.304 |
| Load DTLB walk | 0.085 | 0.100 |
| Store DTLB walk | 0.087 | 0.107 |

| Requests/call | Candidate | Current |
| --- | ---: | ---: |
| L2_DEMAND_HIT | 145,831,593 | 145,856,687 |
| L2_DEMAND_MISS | 1,058,922 | 1,139,982 |
| L2_RFO_HIT | 3,385,529 | 3,278,259 |
| L2_RFO_MISS | 168,388 | 168,917 |
| L2_ALL_RFO | 3,180,885 | 3,446,993 |

| Line traffic/call, MiB | Candidate | Current |
| --- | ---: | ---: |
| L2_LINES_IN | 951.85 | 951.98 |
| L2_DIRTY_EVICTIONS | 49.08 | 44.36 |
| SOCKET_CACHE_READS | 876.39 | 903.74 |
| LOCAL_DRAM_READS | 66.81 | 50.99 |
| Socket 1 IMC read | 76.14 | 61.35 |
| Socket 1 IMC write | 115.37 | 100.25 |

Absolute load-bound cycles change +7.35%, L2-miss stall cycles +9.79%, and L3-miss stall cycles +16.08%. Line traffic is count×64 and includes repeated requests, not unique payload. Demand misses exclude merged misses; IMC units are honored and socket-wide counters include background activity. These measurements support a cache-delivery diagnosis but do not isolate LLC slices or establish a quantization-only DRAM rate.

Core diagnostic cycles/reference-cycles ratio: candidate 2.974 GHz equivalent (2.4 GHz reference).
Core diagnostic cycles/reference-cycles ratio: current 2.955 GHz equivalent (2.4 GHz reference).

A separate chain immediately sums every BF16 output in FP32. It is diagnostic only and excluded from acceptance; each mode uses one fresh process and five samples.

| Operator + output sum | Candidate ms | Current ms |
| --- | ---: | ---: |
| cold | 3.115377 | 3.160214 |
| warm | 2.861827 | 2.917406 |

### Verification and reproducibility

The complete scaled-MM kernel-selection suite passes 26 tests, including exact frozen legacy output comparison, unequal scales, bias/no-bias, boundary and signed quiet/signaling NaN/Inf rows, bitwise input preservation, strided BF16 input, first direct AMX execution, alternating AMX/BRGEMM, mismatched shapes and constrained actual-team fallback. The diagnostic phase binary also passes full-operator legacy comparison and input immutability. Required pre-commit hooks and `git diff --check` pass. The existing cached Llama greedy-generation comparison matches token IDs/text under legacy and single_shape; this model exercises fallback, not the exact specialization.

Selected `gemm_int8.cpp` SHA256: `e68499f88763e268b1fe4b0284d8709492800af5a1bb5d2ce2ce0755fe875d15`.
Selected extension SHA256: `6dbf65f82da0d11b9e736da09715848b5babc67bc6effce37300a16799688b91`.
Current baseline extension SHA256: `a1888f1c9e406212164a03f4b8657ebcf6365f6f66f25c826c2f63188b6eba67`.

Persistent archive: `/home/tianmuli/vllm_test/snc_scaling_analysis/quant_pipeline_64db699a4f_20261030`. Its `README.md` records replay commands, original-path substitutions and dependencies. Source snapshots, all raw cold/warm samples, failed experiments, build commands and hashes, frozen link inputs/control binaries, phase/RFO evidence, codegen review, test/model/hook logs and a SHA256 integrity manifest are retained. Original root: `/tmp/sgl-quant-64db699a4f-20261030`. No system settings were changed. AI assistance was used.

## Previous result: bounded 200-TOPS investigation (2026-10-30)

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

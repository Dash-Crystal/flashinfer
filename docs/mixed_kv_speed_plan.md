# Mixed-format paged KV — implementation plan (synthesis of 5 reads x judges)

Input caveat: the judge payload I received held two score sets (the second truncated after item [38]); no third set was present. The sm120 read was truncated after its L1; its L2-L6 are reconstructed from the judges' verified reasons ([26]-[31]). Lever indices below are the judges' ([0]-[11] sm90-consumer L1-L12, [12]-[19] sm90-io L1-L8, [20]-[24] fa3 L0-L4, [25]-[31] sm120 L0-L6, [32]-[39] cross-cutting read). Code anchors re-checked locally: `csrc/xqa/mha_sm90.cu` gemm1 commit/wait at :1657/:1661, setmaxnreg 24/56 at :1252-1256, `chooseNbSubSeq` at :4070, `nbKBuf = (ENABLE_MIXED_KV_CACHE && !SPEC_DEC) ? MIXED_KV_KDEPTH : 2` at :181, `nbScaleTiles = 4` at :318; `csrc/xqa/mha.cu` SPEC_DEC `nbCtaPerSM = 2` for sm90/M_TILESIZE 16 at :3508-3512, `preferedKHeadPartBytes = 64` / `cacheVTileSeqLen = 32` for sm120 at :120-125; `flashinfer/xqa.py` forces `run_sm90_fp8_mha = False` for q>1 when `swap_ab_eligible` at :566-575; FA3 `kProducerRegs 104 / kConsumerRegs 200` at `sparse_mixed_mainloop.cuh:176-177`; PDL disabled for the mixed build at `mha_sm90.cu:4226/4338`, `mha.cu:3676`.

Accounting convention: sm90 wall = (tile-times per slot) x T_tile + fixed. Reads disagree on the multiplier (33 = 8704/264 vs 39 = 3 rounds x 13 at n=5) and on the per-tile converter period (3.04 us from level-3 stamps vs 96/39 = 2.46 us). Converter-paced predictions are therefore quoted as ratios of the measured period applied to the measured wall (e.g. 3.0 -> 2.83 gives 96 -> 91), and P0.3 fixes the multiplier before any absolute total is treated as a prediction.

---

## Phase 0 — Attribution and calibration (no production code; env vars, trace builds, ncu). All items run in parallel; every later phase cites one of these as its gate.

**P0.1 [32] Kernel family, HBM ceiling, co-tenant** (both judges: do-first/do)
- Files: `benchmarks/bench_xqa_mixed_page_transport.py`, `flashinfer/xqa.py:525-533`.
- Mechanism: bench prints kernel family (mha.cu vs mha_sm90.cu) and module URI for every mode; run a streaming-read probe on nkcut2 for the achievable TB/s; check `nvidia-smi` for co-tenants and locked clocks (cutlass_references.md section 7 records a VLLM co-tenant during earlier sm90 runs).
- Model: A16 baseline (83 us) is mha.cu on sm90 at 3.4 TB/s; mixed-build transport_a16 sits at the same 3.4 TB/s. If the host ceiling is ~3.4 TB/s the byte rooflines move (A16 59 -> ~84, FP8 32 -> 45, FP4 17 -> 24, mixed 36 -> 51) but the gate targets do not (they are relative to measured A16).
- Artifact / accept: family + URI in every bench line; probe TB/s recorded in backends.md; all later H200 numbers taken with no co-tenant.
- If the probe shows ~4.5+ TB/s: A16's 3.4 TB/s plateau is a per-tile issue limiter (16 TMA boxes/tile elected-lane issue ~1.8 us) and becomes an A16-side item for the loader (fold into [15]/[34]).

**P0.2 [39] SPEC_DEC attribution** (J1 do-first; J2 truncated)
- Files: `flashinfer/xqa.py:566-575`, `csrc/xqa/mha.cu:3508-3512`, JIT cubin of the q=4 mixed module.
- Mechanism: confirm q=4 mixed/fp4 (935/276 us) run mha.cu with SPEC_DEC (not mha_sm90); `cuobjdump -res-usage` REG/STACK and `cuobjdump -sass | grep -c LDL` for the q=4 module; record `nbCtaPerSM` (2 -> 128-reg cap on sm90 for M_TILESIZE 16).
- Accept: attribution line in backends.md; REG/STACK/LDL numbers per format module (fp4 vs mixed).
- Decides Track S step 1 vs step 2 (below).

**P0.3 [11] + missing-lever: consumer trace, tile-time multiplier, RT constants**
- Files: `mha_sm90.cu` MIXED_KV_TRACE=1 slots 0-7 (CTA 0, tiles 2-7); extend trace with per-CTA %globaltimer start/end (2 words per CTA); a 50-line clock64 microbenchmark for RT_mbarrier (128-thread arrive_and_wait), RT_bar.sync(128), and the m64n8k16 SS wgmma drain.
- Model: recurrence at X depth 2 gives cadence = max(T_g0, T_g1) unless slot3-slot2 (gemm0 xBar.consumed wait) or slot5-slot4 (gemm1 xBar.produced wait) contain a wait on the other group; the SharedMem comment claims lock-step (sum). Multiplier: 62 us = 39 x 1.5 + 3.5 or 33 x 1.88.
- Zero-code calibration: `XQA_NB_SUB_SEQ=3` vs default 5 -> predicted duration ratio 44/39 = 1.13 if the wave model is right.
- Accept: (i) cadence hypothesis stated (sum vs max) with the two slot differences; (ii) per-slot tile-times histogram -> multiplier; (iii) RT_mbarrier, RT_bar.sync, L_mma in cycles.
- Decides: Phase 1's wall-clock expectations (sum -> [0][1] show on stopwatch; max -> only gemm0 levers [2][4][33c] do), and whether [11] X depth 3 is built at all.

**P0.4 [12] Converter PC sampling + SASS class count** (both do-first)
- Files: `mha_sm90.cu expandPackedStage (:2734-2819)`, `issueCompressedPageCopies (:2538-2607)`; JIT cubin.
- Mechanism: `cuobjdump -sass` count of the converter body per lane-tile (expect ~310 fp4 / ~245 fp8-bf16; class counts 96 PRMT + 32 HMUL2 + 8 STS.128 fp4; 64 HADD2.F32 + 64 F2FP + 32 HMUL2 fp8); ncu PC sampling restricted to that PC range: stall_mio_throttle / lg_throttle / short_scoreboard vs stall_wait vs no_instruction vs not_selected.
- Model: 3350 cyc / 310 instr = 10.8 cyc/instr with 1.7 eligible warps -> the converter is self-stalled, not starved. Stall class selects the Phase 2/4 path: MIO/drain -> [14] overlap; wait/ILP -> [15] 64-reg layout; no_instruction -> code-size cuts; not_selected -> only count levers ([16], audit).
- Accept: per-class stall shares for the expansion and copy-issue PC ranges; SASS class counts recorded (this is the missing measured record).

**P0.5 [34] Fair-share vs latency-bound discriminator** (J1: "single most informative cheap experiment")
- Mechanism: FP4 build, `MIXED_KV_EXPERIMENT=1` (converters skipped), `XQA_NB_SUB_SEQ=1` -> 136 CTAs at 1 CTA/SM (5 warps/scheduler) vs default (10 warps/scheduler). Read per-tile consumer cadence from the trace, not the stopwatch.
- Model: L_chain(5) ~1.0 us -> fair-share (chain scales with warps/scheduler): spin removal and [15]/[34] pay, [38] arithmetic is live. L_chain(5) ~1.7 us -> latency-bound: only RT-removal levers pay, [9]/[38] rejected for good.
- Accept: the two cadence numbers in backends.md.

**P0.6 [5] step 1: elected converter arrive** (one-site edit, trace read)
- Files: `mha_sm90.cu` K/V converter `kBar[s].produced.arrive()` (:2114-2116 region); `barriers.cuh MBarrier::arrive(update)`.
- Mechanism: `__syncwarp(); if (lane==0) arrive(32)` instead of 32 per-lane arrives; counts unchanged. Read level-3 segment 13->8 (fence + arrive, 0.45 us today).
- Model: if 128 same-address arrivals serialize at c cyc/lane the segment drops to <0.1 us; if it is the release-arrive waiting on the 8 x STS.128/lane drain (J1's reading) it stays ~0.45.
- Accept/branch: segment < 0.15 us -> apply elected arrivals to all 17 consumer/loader sites (Phase 2 item [5] step 2, predicted -0.1..-0.3 us/tile per role). Segment unchanged -> [5] closed as a negative result; the drain is attacked only by [14].

**P0.7 [24] FA3 fin sub-stamps + producer PC sampling** (both do-first)
- Files: `sparse_mixed_mainloop.cuh` MIXED_FA3_TRACE blocks (:683-690, :696-738, :782-793).
- Mechanism: stamps after cp.async.wait_group, bar.sync B, expand K, expand V, fence.proxy.async, commits; acq split K/V; ncu smsp__pcsamp_warps_issue_stalled_{mio_throttle,lg_throttle,short_scoreboard,barrier,membar,long_scoreboard} on WG0; l1tex shared bank conflicts.
- Model: source count predicts 1.0-1.3 us/pair A16 and 2.5-3.3 compressed; measured 2.45 and 7-8.4. Candidates: landing wait, expansion IPC under smem-port contention with wgmma operand reads (~1024 cyc/tile of the 128 B/clk port), fence cost, consumer slowed by the 200-reg cap (acq large).
- Accept: sub-segments sum to fin within 2%; stall mix for WG0. Decides whether [23]'s 0.5-0.6 us/pair target holds or must be re-derived from smem wavefronts (producer ~600 + consumer ~1024 of 2800 cycles).

**P0.8 [25] sm120 request-rate vs per-SM-latency gate** (both do-first)
- Files: `csrc/xqa/mha.cu` launcher (:3760-3770, XQA_NB_SUB_SEQ honoured), ~30 lines behind a macro for probe (c).
- Mechanism: (a) `XQA_NB_SUB_SEQ=5` -> 680 CTAs = exactly 4 waves on 170 SMs (12.8 vs 16 CTA-tiles per SM); (b) ncu dram__sectors_read.sum, lts__t_sectors_srcunit_tex_op_read(.sum/_lookup_miss), dram__cycles_active pct, smsp__issue_active, stalled_long_scoreboard vs no_instruction; (c) FP8 K-part copies fetch 64 B/token (2x bytes, same request count).
- Model: per-SM-latency-bound -> (a) gives 0.80x on compressed modes (FP8 139 -> ~112, mixed 146 -> ~118, FP4 84 -> ~68; A16 -> ~160); request-bound -> (a) flat within 5%, (c) flat at ~139 with 2x bytes.
- Accept: elapsed and dram sectors per mode for (a) and (c). Decides Track W branch.

---

## Phase 1 — sm90 consumer skeleton, cheap set (worktree A: `mha_sm90.cu` warpIdx.z 0/1 branches, SharedMem consumer fields, `barriers.cuh`)

Predicted effect (per tile per CTA, and wall): consumer floor (converters-skipped experiment) 1.5 -> ~1.0-1.1 us/tile -> 62 -> 45-50 us under the sum hypothesis; 55-58 us under max() (gemm0-side levers only). Production FP8/FP4/mixed: 0 +-2 us today (converter-paced at 3.0 us/tile). A16: 0 (DRAM-bound). This phase is verified by SASS + trace + the experiment-bit floor, not by the production stopwatch.

**[4] Hardware named barriers for the two intra-warp-group syncs** (both do-first; 0.5 d)
- Files: `computeWarpGrpColMax_sync (:2960-3013)`, `rescaleGemm1AccForNewColMax_sync (:3564-3640)`, finalize warpGrpBar sites; `barriers.cuh NamedBarrier (:405)`, ids 3/4 (1/2 free since expandPackedStage uses __syncwarp).
- Mechanism: replace mbarrier arrive_and_wait pairs with `bar.sync id,128`. J1 correction: the split arrive->work->wait in rescale cannot be expressed with bar.sync (double count) -> becomes a plain sync at the wait point (or disappears with [1]).
- Model: per sync RT_mbarrier (150-400 cyc) -> ~40 cyc; 4 syncs/tile -> 0.22-0.7 us/tile alone; 0.06-0.18 residual after [1][2]. Removes 512 thread-arrivals and 4 poll loops per tile from SYNCS/issue.
- Verify: SASS shows BAR.SYNC in both consumer regions, SYNCS.PHASECHK sites drop by 4/tile; trace slot2-slot1 shrinks by (RT_mbarrier - RT_bar.sync) x 2 (this is the RT calibration if P0.3's microbenchmark is delayed); ncu sm__issue_active falls while duration does not rise.
- Accept: slot2-slot1 <= 300 cyc + shuffles; bit-exact matrix unchanged.
- If fails (slot2-slot1 unchanged): RT is not mbarrier-dominated -> read PC samples on the shuffle/ATOMS lines; proceed to [2] anyway (removes the ATOMS).

**[0] Single commit / single wait in gemm1 PV** (both do-first; 0.5 d)
- Files: `mha_sm90.cu:1640-1661` (SWAP_AB, CACHE_ELEM_ENUM 0/5 gemm1 idxInstK loop).
- Mechanism: issue all 8 HGMMA, one commit, one wait_group<0> at tile end; arrives untouched (no protocol change). Deferred-to-next-tile wait is a second step only after [1].
- Model: saves 3 x (issue bubble + drain); J1 sizes the m64n8k16 SS drain at 60-150 cyc -> 200-450 cyc = 0.1-0.25 us off T_g1.
- Verify: SASS WARPGROUP.DEPBAR count in gemm1 region 4 -> 1 per tile, HGMMA = 8; trace slot7-slot6 collapses toward issue-only (~150 cyc).
- Accept: slot7-slot6 <= 250 cyc. Wall effect 0 if P0.3 says max() with gemm0 binding — that is expected, not a failure.
- If slot7-slot6 does not shrink: the HGMMAs are smem-read-bound (2 KB A per instr at 128 B/clk = 16 cyc each, 8 = 128 cyc floor) — record and stop.

**[2] One barrier per tile for gemm0 colMax (per-warp parity slots, register running max)** (both do; 1 d)
- Files: `computeWarpGrpColMax_sync`, `SharedMem::gemm0CurrentSeqMax` -> `[2][4][8]` slots, safeInitRowMax path.
- Mechanism: lanes<4 STS tile-local max into slot[warp][t%2]; one sync ([4]'s bar.sync); every warp LDS 4 slots, max with register running max. WAR-safe (slot rewritten at t+2 after the t+1 sync).
- Model: -1 RT - 2 ATOMS round trips (~70 cyc) = 0.1-0.25 us off T_g0.
- Verify: SASS ATOMS count in gemm0 region 0; sync sites per tile 4 -> 3; slot2-slot1 = shuffles (~150 cyc) + 1 bar.sync.
- Accept: slot2-slot1 <= 250 cyc. If not: PC-sample the SHFL chain (6 dependent rounds ~150 cyc is the floor; anything above is issue starvation -> P0.5 result decides).

**[1] Register-resident running colMax/colSum in gemm1** (both do; 1-1.5 d)
- Files: `rescaleGemm1AccForNewColMax_sync`, `finalizeAndWriteOut_sync` (SWAP_AB), multi-block save, `SharedMem::gemm1AccColMax/Sum`.
- Mechanism: N=8 -> 1 float/lane; every warp reads xColMax[b] and xColSum[b][0..3] (published under xBar.produced release/acquire) and updates its own copy in the same order (bit-exact); single STS + one sync before finalize. Ballot-uniform rescale branch kept. +2 regs within 48.
- Model: -2 RTs - ~100 cyc LDS/STS chain = 0.2-0.45 us off T_g1.
- Verify: SASS SYNCS.ARRIVE/PHASECHK sites in gemm1 region 4 -> 2 per tile; trace slot6-slot5 <= 100 cyc.
- Accept: slot6-slot5 <= 100 cyc; matrix bit-exact.

**[33c] Split xBar.consumed arrive/wait around softmax** (J1/J2 do as part of [33]; 0.5 d)
- Files: `storeGemm0AccToShm (:3123-3160)`, gemm0 tile loop top.
- Mechanism: arrive at loop top, wait immediately before stmatrix. Protocol check (J1): gemm0's early arrive cannot complete a phase alone; gemm1(t)'s release follows xBar.produced(t) which follows the wait.
- Model: hides 1 RT (0.05-0.15 us) behind softmax. Verify: trace slot3-slot2 <= 100 cyc + STSM. Accept as above; if slot3-slot2 stays large it is a genuine wait on gemm1(t-2) -> that is the [11] signal.

**[11] X ring depth 3 (+ nbScaleTiles 4 -> 3)** — build only if P0.3 shows slot3-slot2 or slot5-slot4 waiting on the other group (both judges: cheap, hypothesis-gated)
- Files: `MIXED_KV_XDEPTH (:120)`, `SharedMem::nbScaleTiles (:318)`, kScales/vScales indexing.
- Model: sum -> max: 1.5 -> 0.8-1.1 us/tile (62 -> ~50). Verify: occupancy API still 2 CTAs/SM (111.6 KB); slot3-slot2 and slot5-slot4 -> ~RT only. If no change: cadence was max() already; revert (smem is needed by L8).

**[35] Record RS-decode negative result; delete or #if 0 the dead RS fragment loaders (`mha_sm90.cu:614-847`)** (both do; 0.25-0.5 d). Model: 35-50 decode instr per k-step against a 10-20 cyc m64n8k16 wgmma cannot hide; +1.2-1.4 us/group/tile at 4 cyc/instr, 2-3x worse at the measured 10.8-13.3. Verify: SASS size of the sm90 TU drops; no HGMMA R-form appears. Closes [18].

Deferred inside Phase 1: **[3]** colSum-to-epilogue (both maybe; ~0.05 us/tile, order change breaks bit-exactness against external references; do only as part of L8). **[6]** gemm0 software pipelining (both maybe): only with a `test_wait.parity`-guarded prefetch (issue QK(t+1) early only if kBar[t+1].produced is complete), otherwise it lengthens the chain while converters pace; revisit after Phase 2 when the trace shows converters leading.

---

## Phase 2 — sm90 IO/converter cheap set (worktree B: `mha_sm90.cu` warpIdx.z 2/3/4 branches, loader, barrier init :1185-1220; `mhaUtils.cuh` convert helpers under an sm90 arch guard)

Predicted effect (converter period 3.0 -> 2.3-2.55 us; ratios applied to measured walls): FP4 96 -> 74-82; FP8 91 -> 68-76 (with [16]); mixed 114 -> 85-95 ([13] also decouples converters from A16 TMA landing, which is the mixed-specific excess over both pure modes); A16 83 -> 83. Consumer floor unchanged. These are the first levers that move the production stopwatch.

**[13] Remove the per-tile kLoadReady rendezvous** (both do; 1.5 d)
- Files: barrier init `:1190-1199`, K loader `:1857-1891`, converters `:2104-2113 / :2153-2162`, `expandPackedStage` formats argument, `fillTileMeta` chunk ordering (:1882-1885).
- Mechanism: converters already read the tag in `issueCompressedPageCopies` two tiles ahead -> keep it in rotating registers; static compressed builds drop kLoadReady/vLoadReady at compile time; mixed streams retarget A16 TMA tx bytes to kBar[s].produced (loader `arrive_tx` + 32 arrivals; gemm0's single wait covers TMA landing and expansion; A16 rows and expanded rows are disjoint). Gap (J1): kLoadReady also orders the metadata chunk refill for converter reads -> add a per-chunk ready barrier (count 32, loader arrives after fill; converters wait once per 16 tiles).
- Model: -0.17 us of 3.0 -> FP4 96 -> ~91, FP8 91 -> ~86, mixed 114 -> ~108; mixed additionally stops exposing TMA landing to the converters.
- Verify: trace slot 12-11 -> ~0; SASS: no SYNCS.ARRIVE on kLoadReady in the converter body; SYNCS.ARRIVE.TRANS64 in the loader; sizeof(SharedMem) -6 x 8 B for static builds. Timeout-bounded matrix incl. nbIters < 3 tails (C7 class).
- Accept: FP4 <= 92. If the converter period does not drop 0.17: the wait was already overlapped -> the 0.17 was the format LDS/publish, look at slot 11-10.

**[16] FP8->BF16 decode 5 -> 4 SASS per pair + hoist lane-constant store offsets** (both do; 1 d)
- Files: `mhaUtils.cuh convertE4M3x2ToA16 (:376)`, `expandCompressedBlock16WithScale (:559)`, `expandPackedStage` store addressing (:2812-2814). Arch-guard: sm120 keeps its native cvt path.
- Mechanism (J1 correction: sm_90 has no f16x2->bf16x2 cvt): `cvt.rn.f16x2.e4m3x2` -> `SHF.R.U32` + `LOP3` (sign | ((h>>3) & 0x0FFF0FFF)) = bf16(x * 2^-112) exactly -> `HMUL2.BF16` with 2^112 folded into the per-block scale word (exact while block_scale * global < 2^16: host assert, else fall back to the 5-instr path). Hoist the 8 store chunk XORs ((2b+g)^(token%8) is a lane constant) to 8 registers or immediate offsets; drop `asm volatile` from pure conversions.
- Model: fp8 lane-tile 245 -> ~213 (-13%); at the observed 10.8-13.7 cyc/instr expansion 1.6 -> ~1.35 us -> period -0.2..-0.25 us; fp4 -16 instr (-0.09 us).
- Verify: SASS HADD2.F32 count in converter body 64 -> 0; F2FP count halves; LOP3/LEA for stores 16 -> <= 8; STACK:0; bit-exact matrix (add pages with e4m3 subnormal values and max scales).
- Accept: FP8 wall -5 us or more relative to post-[13]. If the expansion segment does not scale with the count: rate-bound (P0.4 says MIO) -> only [14]/[15] help; stop count levers here.

**[14] Software-pipeline the store drain: defer fence + produced.arrive(t) past tile t+1's loads and the copy-issue chain** (J1 maybe 0.1-0.2, J2 do 0.45; 2 d; fp4/mixed first, fp8 after [15] or with partial early loads)
- Files: converter loops `:2082-2124`, `:2131-2166`; split `expandPackedStage` into loadPacked()/decodeStore().
- Mechanism: wait landed(t+1) -> LDS packed rows + scale word of t+1 into registers -> issueCopies(t+3) -> fence.proxy.async + produced.arrive(t) -> decode/STS of t+1. landed(t+1) depends only on the converter's own cp.async groups (no cycle with gemm0). J1 caveat: same-warp smem ops are in order through MIO, so only the non-MIO part of the copy chain (IMAD page bases, wait) overlaps the drain.
- Model: serial 0.45 + 0.64 -> max() plus residual: J2 -0.45, J1 -0.1..-0.2. Plan uses -0.2 (period 2.83 -> ~2.6) and lets the trace decide.
- Verify: trace segment 13->8 < 0.1 us and 8->9 overlapping; SASS FENCE.VIEW.ASYNC after the LDGSTS group; instruction count +-5; REG <= 56, STACK:0.
- Accept: period -0.2 us. If < 0.1: the drain is MIO-serialized behind the next LDS -> move the packed LDS of t+1 after the fence (keep only the IMAD/wait overlap) and record.

**[19] Copy-issue precompute only** (J1 maybe, J2 do; 0.5 d, after [14]) — `test_wait` fast path rejected (same SYNCS.PHASECHK path, J1). Precompute page/scale bases for t+2 during the initial LDS window. Gain <= 0.05-0.1 us; do only if segment 8->9 stays exposed after [14].

**[5] step 2 — elected arrivals at all consumer/loader sites** — only if P0.6 segment dropped below 0.15 us. Model -0.1..-0.3 us/tile per role; verify SYNCS.ARRIVE warp-instr per tile 68 -> ~17, PC-sampling stall on arrive lines ~0.

**Converter SASS audit (J1 missing lever; 1 d, output feeds Phase 4)**
- Mechanism: from P0.4's SASS, attribute the ~170 non-essential fp4 instructions (essential ~140: 48 PRMT-LUT, 32 HMUL2, 8 STS.128, 2 LDS.128, ~14 scale prep, ~20 addressing): swizzle XOR per store, format predication, 64-bit pointer math, loop/unswitch overhead. Each removal is a count lever with prediction (instr x 10.8 cyc).
- Accept: fp4 lane-tile <= 180 SASS with bit-exact matrix. This is the only converter-side route toward FP4 <= 36 alongside L8; its result is quoted in the gate check.

---

## Phase 3 — sm90 scheduling and ramp (worktree C: `chooseNbSubSeq` host, kernel entry indices, prologue, ScratchMem). Independent of cadence hypotheses; absolute savings persist after every other lever.

Predicted effect at converter-paced cadence after Phase 2: -15% of steady state (39 -> 33 tile-times) and -2..-4 us ramp: FP8 68-76 -> 56-63; FP4 74-82 -> 61-68; mixed 85-95 -> 70-79; A16 83 -> 78-83 (ramp only; DRAM-bound).

**[8] Persistent, balanced CTA scheduling** (both do; 4-5 d)
- Files: `chooseNbSubSeq (:4070-4108)`, kernel_mha grid/index derivation (idxReq/idxHeadGrp/idxSubSeq from a global atomic work counter), `fillTileMeta` cross-item prefetch, ScratchMem sizing, `launchHopperF8MHAFlashInfer` grid = 264.
- Mechanism: 264 CTAs pull (seq-head, tile-range) items so each processes ~33 tiles; finalize -> partial write -> next item via the existing multi-block scratch/merge; barrier parities become per-CTA running counters across items (cf. FA3 amendment A5 cross-item K(last) hazard); next item's metadata + first copies prefetched during the current item's last tiles.
- Model: tile-times 39 -> 33; fills per slot 3 -> 1. Calibrated first by P0.3's `XQA_NB_SUB_SEQ=3` ratio (1.13 predicted).
- Verify: per-CTA %globaltimer start/end histogram (slot idle + tail), ncu sm__cycles_active.max vs .avg, launch stats 264 CTAs; duration vs 33 x T_tile + f.
- Accept: -12% or better on every converter-paced mode at unchanged T_tile (trace). If the histogram shows tail slots finishing early anyway (J1: lone CTAs run faster), the upper bound was 9 us; take what the histogram gives and stop.

**[36a][36b] Ramp cuts** (both do; 1 d): defer metadata chunk 1 fill to iteration ~12 (first needed at 14, `:1845-1850`); move the Q load off warp 0 (currently both Q loader and K loader, so the first K TMA waits behind the Q round trip). Model: -2..-4 us (2-4 us ramp x 2.58 lifetimes half-hidden). Verify: per-CTA start-to-first-kBar.produced stamp shrinks by one DRAM round trip.

**[37] PDL for the mixed build** (J1 maybe, J2 do; 1 d): flip `enable_pdl && !ENABLE_MIXED_KV_CACHE` after placing `acqBulk` before the first page-table/transport read in every role. J1 is right that the single-node CUDA-graph bench cannot show it (no predecessor kernel); it is a production-graph item (2-5 us) and does not enter the gate numbers. Verify: griddepcontrol instructions in SASS; a two-kernel graph bench shows overlap.

---

## Phase 4 — sm90 structural (sequential; after A, B, C have merged; touches all roles)

**[15]/[34] Four-warp-group layout, 64 registers uniform, loader merged into converters** (J1 do, J2 maybe; 7-8 d) — gate: P0.4 stall class (wait/ILP -> up to -0.8 us of 1.69; MIO -> only [13]'s share) and P0.5 (fair-share -> spin/warp removal pays).
- Files: `ctaWarpGroups (:137)`, SharedMem barriers (:330-339), role dispatch (:1252-2170), `KVTilePartLoader` (:401-480, :2466-2900), setmaxnreg block (:1244-1257, removed).
- Mechanism: gemm0, gemm1, K-side (converters that also issue their own page's A16 TMA from an elected lane; Q by warp 0 once), V-side. 512 threads x 2 CTAs -> 64 regs, no setmaxnreg; kBar.consumed count 128; 8 fewer resident warps/SM.
- Model: converter period 2.3-2.55 -> 2.0-2.2 us; consumer groups get 64 regs (needed by L8). Predicted (with Phase 3): FP8 54-60, FP4 55-60, mixed 60-68.
- Verify: `cuobjdump -res-usage` REG:64 STACK:0, no USETMAXREG; SASS of converter body shows interleaved PRMT/HMUL2/STS across blocks; ncu launch stats 16 warps/CTA, 2 CTAs/SM; trace converter period.
- Accept: converter period <= 2.2 us. If ILP does not materialize (SASS still serial): ptxas scheduling, not registers, is the limit -> manual two-block interleave in source, verified by SASS order.

**[7] L8 fused consumer warp groups (FA3 pattern, alternate tiles per group)** (J1 do, J2 maybe; 7-10 d) — gate: after Phases 2-4 the trace must show the consumer above or within 20% of the converter period; otherwise its wall effect is 0 and it is built only for the FP4 pursuit.
- Files: warpIdx.z 0/1 branches -> one body parameterized by group; `reusedXVOutSwizzleBuf/xColMax/xColSum` -> 2x2 private P buffers; xBar removed; kBar/vBar counts unchanged; in-CTA merge of two (O, max, sum) via the multi-block combine arithmetic. Reference `mainloop_mma.cuh`.
- Mechanism: kBar wait -> 8 HGMMA QK -> max exchange via bar.sync ([2]) -> exp2 + per-thread partial sums ([3]) -> bf16 P STSM -> bar.sync + fence.proxy.async -> vBar wait -> 8 HGMMA PV (single commit, wait deferred) -> elected consumed arrives.
- Model: ~900 cyc per tile per group, two groups interleaved -> 0.25-0.3 us/tile/CTA, bounded by smem port (~80 KB per CTA-tile -> 0.33 us/tile-time at 2 CTAs/SM). Consumer floor 62 -> 9-21 us. Register cap: J1 correction — setmaxnreg operands are multiples of 8, so 52 is illegal; consumers stay at 48 in the 5-group layout, or 64 after [15] (which is why [15] precedes this).
- Verify: SASS consumer region instr/tile ~halves, HGMMA 16, SYNCS.PHASECHK 2, BAR.SYNC 2, STSM 1, FENCE.VIEW.ASYNC 1; REG <= cap, STACK:0, no LDL; new per-group trace stamps ~900 cyc/tile; bit-exact matrix incl. >1 item per CTA.
- Accept: converters-skipped floor <= 25 us. If REG spills at the cap: fall back to the 5-group layout with [15]'s 64-reg budget is unavailable -> hold P in smem only (no register P words) and re-measure; if still spilling, [38] (128-token tiles at 1 CTA/SM) is the fallback only under P0.5 = fair-share.

Rejected on arithmetic (both judges; record in backends.md): **[9]** 128-token stages at 2 CTAs/SM (192 KB -> 1 CTA/SM; +10-30%); **[10]** 3 CTAs/SM (34 regs, 74.6 KB); **[17]** convert-once/larger-stores/no-fence/extra IO lanes (setmaxnreg is warpgroup-uniform; 33792 > 30720); **[18]** RS-K decode in gemm0 (see [35]); **[38]** 128-token tile at 1 CTA/SM (J1 maybe/J2 reject; issue budget tile-invariant, ~41 us floor, 10+ d) kept only as the L8 fallback named above.

---

## Track S — SPEC_DEC q=4 (mandatory; worktree E shares `mha.cu` with Track W)

Step 1 (after P0.2; 0.5 d): **launch bounds for the mixed SPEC_DEC build.** File `mha.cu:3508-3512`: `nbCtaPerSM = 1` when `ENABLE_MIXED_KV_CACHE && SPEC_DEC` (cap 128 -> 255 regs). Model: static demand 226 fits; STACK 0; LDL count -> 0 for the register-array spills; occupancy 2 -> 1 CTA/SM on sm90 (the pathology is 3.4x fp4, so the trade is favorable if the LDL is the cause). Verify: `cuobjdump -res-usage` REG <= 255, STACK:0, LDL count; ncu sm__inst_executed_pipe_lsu drop. Accept: mixed q=4 <= 1.5x fp4 q=4 (<= ~415 us). If REG/STACK are clean but time stays > 2x fp4: the second cause is the copy/expand shape, go to step 2.

Step 2 (2 d): **[29] C2 fix in `mhaUtils.cuh copyMixedPartialHeadsAsync (:232-372)` / `expandMixedPartialHeadsInPlace (:603-645)`**: `tokenOffset` is provably 0 (warpTile.x 64 and cacheVTileSeqLen 32 are multiples of 16) so `pages[localPage]`/`formats[]` become compile-time indices (removes the 118-338 LDL common to sm120 and sm90 SPEC_DEC); delete the dead zero-fill cp.async (13 -> 9/5 LDGSTS per part). Verify: LDL = 0 in both static and dynamic modules; LDGSTS count 9 (fp8) / 5 (fp4) per part per lane. This step also serves Track W.

Result (2026-09-04, worktree E; details in backends.md "P0.2 [39] SPEC_DEC attribution, Track S step 1 (no-op), Track W [29] C2 fix"): step 1 is a no-op - `M_TILESIZE` is the default 32, so the sm90 SPEC_DEC build already had `nbCtaPerSM = 1` / 255 registers (fp8 module REG:255), and the STACK 48/112 + LDL 114/378 were the C2 register arrays (identical STACK on sm120 at REG 166-242). Step 2 done, extended to `HeadPtr::operator+` (the dynamic module's A16 path held the other half of its LDLs): LDL 0 / STL 0 / STACK 0 in every static and dynamic module on both hosts; LDGSTS 3 -> 2 (fp8) / 1 (fp4) per block; 34/34 bit-exact on both hosts. sm120: FP4 83.4 -> 65.5, FP8 139.5 -> 118.8, mixed 145.8 -> 126.4 us (all three sm120 targets pass from [29] alone). sm90 q=4 (3 locked rounds, co-tenant present): fp4 276.5 -> 239.5, fp8 231.2 -> 198.2, a16 137.7 -> 131.4, mixed 436.9 -> 422.8 = 1.77x fp4: **acceptance (<= 1.5x) not met by steps 1-2**. ncu: mixed executes the same instruction count as fp4 but at 10.0 vs 5.1 warp-cycles per issued instruction, 6.3 of them instruction-fetch (no_instruction) stalls of the 17.9 K-instruction dynamic kernel; grid 136 on 132 SMs at 1 CTA/SM (2 waves). `XQA_NB_SUB_SEQ=4` gives 0.85x on every q=4 mode (a16 112, fp8 168, fp4 205, mixed 359) but leaves the ratio at 1.75x. Next: step 3 (sizeof(SharedMem) arithmetic for the mha_sm90 SPEC_DEC route) or a code-size/dispatch lever for the dynamic path; a host `nbSubSeqPerSeq` default for the sm90 SPEC_DEC shape is a separate ~15 % item once the per-CTA fixed cost is modelled.

Step 3 (only if steps 1-2 leave q=4 mixed > 1.5x fp4): route q=4 to `mha_sm90.cu` SPEC_DEC (relax `xqa.py:566-575` for mixed pages) after fixing its `nbKBuf = nbVBuf = 2` for SPEC_DEC (`:181/:207`; kAhead = 1 exposes the ~1.9 us landing every tile) — needs the SPEC_DEC smem budget re-summed (ctaNbQHeads 16 -> X entry 2 KB, mask buffers) against the 112.5 KB cap; predicted q=4 at q=1-like cadence (~100-120 us at today's converters). Decided by the sizeof(SharedMem) arithmetic before any build.

Result (2026-09-04, worktree E; details in backends.md "Track S step 3 — [40] per-page format dispatch in the mha.cu dynamic path"): the arithmetic says K3/V3 does **not** fit two CTAs per SM at q=4 (`sizeof(SharedMem)` = 117,760 B vs the 115,200 B cap; K2/V2 84,992, K3/V2 101,376 fit), and the route is blocked before smem: the mixed q=4 module's `mha_sm90.cu` kernel is a 16-instruction stub (`:1089` guard, `IS_SUPPORTED_F16_CASE` requires `!SPEC_DEC` for enum 5) on top of the SWAP_AB full-draft-mask defect — a new kernel path in an A/B file, not a routing change; predicted 95-135 us if ever built.  Route taken: **[40] per-page format dispatch** in `mhaUtils.cuh copyMixedPartialHeadsAsync / expandMixedPartialHeadsInPlace` (page-outer loops, one format branch per page, format-specialised bodies, page loop rolled in the dynamic module, dynamic module without the separate A16 copy path).  Attribution (lineinfo cubins) had put the whole 7.9 K-instruction excess of the dynamic module in the copy/expansion helpers (3 predicated LDGSTS variants x 8 unrolled iterations, 2 expansion bodies x 8, plus the stock A16 path at every site).  Artifact: dyn SASS 17,912 -> 8,824 (fp4 9,968 -> 9,792, fp8 9,016 -> 8,912, a16 identical), LDGSTS static 455 -> 137, LDL/STL/STACK 0; ncu mixed q=4 warp-cycles per issued instruction 10.04 -> 4.52 (fp4 5.06), no_instruction stall 6.3 -> 0.52 warps per issue-active cycle (fp4 0.77), issued LDGSTS 924,800 -> 555,648, executed instructions 48.1 M -> 46.4 M.  34/34 bit-exact.  sm90 q=4 (3 locked rounds): mixed 422.8 -> **216.4 us (0.51x), 0.915x fp4** (fp4 236.1, fp8 198.7, a16 136.4 with byte-identical SASS = session offset +3.8 %); q=1 control unchanged (83 / 91 / 96 / 116).  **Track S acceptance (mixed <= 1.5x fp4) met.**  The q=4 targets (94 / 59 / 101) remain open for every mode: all run at 1 CTA/SM (255-register SPEC_DEC build, 0.4 issued instr per scheduler-cycle, DRAM 7-14 %) — the next levers are the host `nbSubSeqPerSeq` default (0.85x measured) and a 2-CTA/SM SPEC_DEC build, not the mixed dispatch.

---

## Track F — FA3 prefill parity (mandatory; worktree D: `sparse_mixed_mainloop.cuh`, `kernel_traits.cuh`, `flashinfer/mixed_page_prefill.py`, jit module naming; no XQA files)

Order: P0.7 -> [21] -> [22] -> [23] -> [20]. Predicted: A16 528 -> 300 +-3% (consumer-bound, parity); FP8/FP4 static 1500-1800 -> 300-320; mixed -> 300-330; requirement <= 1.05x stock A16 / <= 1.1x compressed. Tail (5th scheduler round, 16 items) is outside the producer and identical for stock.

**[21] Per-item 16-tile page-metadata chunk table** (both do; 2-2.5 d)
- Files: `gather_tile_meta (:309-331)`, load() ring lambdas (:640-660, :687-709), `produce_pair`; `SharedStorageQKVOMixed mixed_meta_*` (+2.4 KB -> 188 KB).
- Mechanism: threads 0..95 load one kv_index + tag per chunk; STS at the chunk's 15th pair; one bar.sync per 16 pairs; per pair 3 vector LDS; barrier A and warp 0's exposed LDG chain disappear (43 round trips/item -> 3).
- Model: -0.4..-1.0 us/pair exposed latency, -1 bar.sync/pair, -50 instr/thread/pair.
- Verify: SASS BAR.SYNC in pair body = 0 (A16 module), LDS <= 6/pair, no LDG between acquires and LDGSTS; trace gat = 0, iss <= 0.3 us; ncu stalled_barrier/membar on WG0 ~0.
- If A16 is still > 1.1x stock with iss small: acq is large -> the consumer is slower than stock -> [20] moves up.

**[22] JIT-constant static format + D6 copy issue <= 4 instr per 16 B** (both do; 2 d)
- Files: `issue_tile_copies/issue_operand (:369-441)`, `Params.static_format` -> template constant carried in the module URI; `flashinfer/mixed_page_prefill.py`, jit naming.
- Mechanism: per-operand bases (a16_src0, fp8_base, fp4_base, scale_base) computed once per item; per page one IMAD.WIDE.U32 + immediate smem offsets; hoisted `valid_tokens == 96` branch to an unpredicated body; drop f8/f4 swizzle evaluations from the a16 module.
- Model: A16 ~100 instr/thread/pair (~400 warp-instr vs stock 780, now ~1650) -> iss 0.1-0.15 us/pair; compressed T_i 0.6 -> ~0.05.
- Verify: a16 module SASS in the pair body: LDGSTS 24, IMAD.WIDE.U32 12, LDC 0 in loop, no F2FP/PRMT/HMUL2 in the producer region, SEL <= 2 outside the tail branch, producer region <= ~600 SASS (from 3.9K); fp8 module LDGSTS 12 (+12 scale words after [23]).
- Accept: A16 <= 315 us with [21]. If not and acq > 0 (producer ahead): consumer is the bound -> [20].

**[23] Block-granular copy-owner expansion (no barrier B, 128 threads, immediate addresses)** (both do; 6 d; after P0.7 attributes the 2.5x residual)
- Files: `issue_tile_copies` (ownership r = t/8, b = t%8; landing slots), `expand_token -> expand_page_block`, `decode_block/e4m3x2_to_a16/block_scale_a16x2`, `finish_pending/finish_pending_pair` (drop group_barrier B, single fence), docs D2/D3/A2/C3 amendments.
- Mechanism: FP8 16 B / FP4 8 B (cp.async.ca) of row r block b land in chunk 2b of row r; each thread copies its own 4-B scale word; after wait_group the owner decodes (FP8: cvt + SHF/LOP3 shift with 2^112 folded into the scale; FP4: nibble->e4m3 PRMT LUT then the FP8 tail) and STS.128 to chunks 2b, 2b+1 at immediate offsets; one fence per thread per pair.
- Model: FP8 552 / FP4 648 instr/thread/pair + ~90 protocol -> 2560/2960 warp-instr/pair on 4 warps (now ~6500-7000 on 3); issue share per SMSP 50-54% of the 2800-cycle tile; smem port ~57%; P ~0.5-0.65 us/pair <= Tc 1.4 with >2x margin.
- Verify: SASS BAR.SYNC in pair body 0, FENCE.VIEW.ASYNC 1/pair, per page exactly 8 F2FP.E4M3 + 8 SHF/LOP3 pairs + 8 HMUL2.BF16 + 2 STS.128 with immediate offsets and 0 address ALU; LDGSTS 12 + 12; no LDL/STL; trace fin <= 0.6 us/pair, acq > 0.5; ncu WG0 inst_executed <= 3000/tile, shared st bank conflicts ~0, tensor pipe active equal to stock within 3%; bit-exact matrix incl. many-items, subnormal scales/values, and a NaN-payload case if the quantizer can emit NaN (else host assert `block_scale*global < 2^16`).
- Accept: FP8/FP4/mixed <= 330 us. If fin stays > 1 us with counts as predicted: smem-port bound (P0.7 (b)) -> re-derive from wavefronts; the only remaining lever is spreading STS across the pair (already the design) — record and re-state the requirement from the wavefront budget.

**[20] Restore stock register split 72/216** (both do, low time confidence; 0.25 d, after [23] frees the producer to ~30-40 regs)
- Files: `sparse_mixed_mainloop.cuh:176-177`, `prefill_sm90.cuh:172-176, 217-221`.
- Verify: USETMAXREG.DEALLOC 72 / ALLOC 216; consumer region SASS textually identical to stock modulo addresses; STACK:0 for both regions (ptxas -v at 72 for the producer). Accept: consumer diff empty. If the trace's acq segment was large before [21]/[22], this is the item that closes A16 parity.

---

Step 4 (after step 3; the q=4 targets stay open at 1 CTA/SM): **[41] 2-CTA/SM SPEC_DEC build + [42] host `nbSubSeqPerSeq` default.** Arithmetic first: the sm90 q=4 modules are limited to 1 CTA/SM by registers (239-249 under the 255 cap) *and* by smem (163.6 KB of 228 KB), so both must move: `M_TILESIZE 16` (q x GQA = 16 rows; the 32-row tile is half padding) halves the accumulators (gemm0 32 x 64 -> 16 x 64 fp32, gemm1 likewise) for the 128-register `__launch_bounds__(256, 2)`, and the sm120 K/V ring (64 B K parts, 32-row V tiles) with the 16-row Q/X halves SharedMem to ~82 KB (K 128 B / V 32 rows at M 16 is 116.5 KB: 1.5 KB over the 2-CTA cap). Then the host default: cost(n) = ceil(nbSeq x n / (SMs x CTAs/SM)) x (1 + nbTiles / n), n > 1 only above 5 % modelled gain. Verify: cuobjdump REG 128 STACK 0 per module; ncu occupancy limits 2 / 2, grid 680; 34/34 both hosts; sm120 q=1 unchanged.

Result (2026-09-03, worktree E; details in backends.md "Track S step 4 — [41] 2-CTA/SM SPEC_DEC build and [42] host nbSubSeqPerSeq default"): all four sm90 q=4 modules **REG 128, STACK 0, LDL 0, STL 0** (predicted 128 / 0-32 / spills outside the loop); `launch__shared_mem_per_block_dynamic` 167,504 -> **83,712 B**, `launch__occupancy_limit_registers` / `_shared_mem` 1 / 1 -> **2 / 2**, warps active 12.2 % -> 21-22 % (theoretical 25 %), grid 136 -> 680 (2.58 waves); the XQA_NB_SUB_SEQ sweep on the new build calibrated the host model to the mean tiles per CTA (n = 5 best on every mode; n = 2/4/6, the "just over an integer wave" counts, pay a full tail wave) while ws-1 stays at n = 1 (grid `[1, 8, 17]`). sm90 q=4 (3 locked rounds): a16 136.4 -> **99.7**, fp8 198.7 -> **124.3**, fp4 236.1 -> **144.3**, mixed 216.4 -> **137.8** us (0.61-0.73x; every mode inside its predicted band; A16 target 135 passes; FP8 / FP4 / mixed targets still open at 1.32x / 2.45x / 1.36x). q=1 control unchanged (mha_sm90.cu). sm120: q=1 unchanged (100.6 / 59.8 / 113.5 vs 100.5 / 59.5 / 113.5), q=4 FP8 125.0 -> 115.1, FP4 81.9 -> 65.7, mixed 132.3 -> 119.0 (all three sm120 q=4 targets pass). 34/34 bit-exact on both hosts, with the default and with `XQA_NB_SUB_SEQ=2` (merge path). Next for the sm90 q=4 targets: occupancy is exhausted (issue-active 42-44 %, i-fetch stall 2.2-2.8 warps per issue cycle with two co-resident CTAs); the levers are the per-tile instruction count / code footprint of the 4-round K pipeline, or the mha_sm90.cu SPEC_DEC route (K3/V2 at 99 KB fits 2 CTAs/SM).

Step 5 (after step 4; occupancy exhausted, i-fetch bound): **[43] 128 B K parts + rolled tile loops + single copy body** for the sm90 SPEC_DEC M16 mixed build.  Arithmetic first: SharedMem with 128 B K parts / 32-row V at M 16 is 116,480 B, 768 B over the 2-CTA cap (2 x (s + 1 KB) <= 228 KB -> s <= 115,712); the V scale rows are staged at the copy's 4 B stride but allocated at 8 B (`mixedVScaleBytes`), so shrinking them (-1,056 B) gives 115,456 B (256 B under the cap).  64-row V tiles do not fit (147 KB).  Then the code footprint: roll the gemm0 K-part loop and the gemm1 X-tile loop (`kCompactTileLoops`), drop the `isFull` copy specialisations.  Predicted: gemm0 rounds 4 -> 2 per 64-token K tile, hot-loop SASS 6.7 K -> ~2.5 K (dyn), executed instructions -5 %, no_instruction stall 2.2-2.8 -> <= 0.8, q=4 mixed 95-115 / fp8 88-105 / fp4 100-125 / a16 90-100 us.

Result (2026-09-04, worktree E; details in backends.md "Track S step 5 — [43]"): `launch__shared_mem_per_block_dynamic` 83,712 -> **115,456 B** (= the sum), occupancy limits 2 / 2, REG 124-128 STACK 0 LDL 0 STL 0 on every sm90 q=4 module.  Hot-loop SASS (gemm0 + gemm1 loops): dyn 3,239 + 3,500 -> **1,588 + 888**, fp4 6,647 -> 2,589, fp8 5,694 -> 2,243, a16 4,426 -> 1,648 (module totals 9,288 / 9,104 / 8,160 / 7,008 -> 4,968 / 4,688 / 4,352 / 3,928).  ncu: no_instruction 2.16 -> **0.51** (mixed), 2.42 -> 0.30 (fp4), 1.64 -> 0.15 (fp8), 1.86 -> 0.14 (a16); warp-cycles per issued instruction 7.6-8.3 -> 5.8-6.6; issue-active 42-44 % -> 50-58 %; executed instructions +2.5 % (mixed, rolled-loop control offsets the round saving), -5.6 % (fp8).  sm90 q=4 (3 locked rounds): a16 99.7 -> **86.1**, fp8 124.3 -> **114.0**, fp4 144.3 -> **115.9**, mixed 137.8 -> **116.1** us (0.80-0.86x; fp4 and mixed inside their bands, fp8 9 us above its band).  Targets 94 / 59 / 101 stay open at 1.21x / 1.96x / 1.15x; the kernel is no longer fetch-bound (issue-active 58 %, wait 1.0-1.2 and long-scoreboard 0.9-1.4 warps per issue cycle lead), so the next lever is the executed instruction count per tile (51 M for mixed vs 30 M a16).  q=1 control unchanged.  34/34 bit-exact on both hosts (default and `XQA_NB_SUB_SEQ=2`); sm120 SASS byte-identical to a pristine e777da96 build for 7 of 8 modules (the 8th varies between two pristine builds); interleaved base / [43] rounds on ws-1 equal within 0.9 us (q=4 fp8 116.4 / fp4 65.9 / mixed 123.5 vs 116.9 / 65.9 / 123.2 with a VLLM co-tenant resident; idle-GPU records 115.1 / 65.7 / 119.0).

## Track W — RTX 5090 sm120 decode (worktree E; `mha.cu`, `mhaUtils.cuh` copy/expand, `defines.h`)

Branch on P0.8:

Branch (a) — per-SM-latency-bound (n=5 probe gives ~0.8x): set the host default `nbSubSeqPerSeq` on sm120 to fill 4 full waves (mha.cu launcher :3760-3770). Predicted FP8 139 -> ~112 (<= 125 pass), mixed 146 -> ~118 (<= 135 pass), FP4 84 -> ~68 (<= 79 pass), A16 180 -> ~160. Verify: launch stats grid 680, sm__cycles_active.avg/.max ~1.0; merge overhead (~1-2 us/seq) in the per-CTA stamps. Then still do [29].

Branch (b) — request-rate-bound (probe flat, (c) flat at 2x bytes):
- **[26] 128 B K parts on sm120** (both do; 2.5-3 d): `mha.cu:120-125` -> `preferedKHeadPartBytes = 128`, `cacheVTileSeqLen = 16` to keep SharedMem under the 99 KB static_assert (K 64 KB + V 16 KB; recheck kScales/vScales sizes). Model: FP8 K requests 32 B -> 64 B full sectors (1024 -> 512 per SM-tile), FP4 16 B -> 32 B; FP8 139 -> 96-105, mixed 146 -> 110-120, FP4 84 -> 68-75. Verify: dram__sectors_read per SM-tile matches the request budget; static_assert passes; REG <= 255 STACK:0 at launch_bounds(256,1). If the ratio elapsed/dram_sectors does not hold: latency-bound after all -> [30b] third K buffer.
- **[27] GRP_LOAD_V for CACHE_ELEM_ENUM 5** (both do; 1.5 d): `defines.h:203`; re-read the @fixme at `mha.cu:145-148` (nbVBuffers=2 refill hazard with group loads) against the vBarrier/mixedVExpandBarriers protocol; vScales to group scope. Model: whole-row V per token halves V requests; FP8 -> 115-125 alone, ~96 with [26]; mixed -> 125-130 alone, ~112 with [26]. Verify: LDGSTS count per V tile halves; V request sectors halve in ncu.
- **[29] C2 fix** (Track S step 2, same code): LDL 0; FP4 84 -> 78-80 on its own.
  - Result 2026-09-04: LDL 0 / STACK 0 in all modules; measured (3 locked rounds) FP4 83.4 -> 65.5, FP8 139.5 -> 118.8, mixed 145.8 -> 126.4, A16 180.3 -> 174.7 - the sm120 gate passes on [29] alone; [25]/[26]/[27] become optional headroom.
- **[28] 8-B scale word loads** (both maybe; 1 d): only if P0.8 (b) shows scale sectors missing L2 (else issue-only, kernel not issue-bound). Model 15-25% on FP8/mixed if misses, 0-5% otherwise.

Rejected for sm120 (both judges): **[30a][30c]** dedicated expander warps / register-side FP8 (P = (L + T_x + T_m)/2 regardless of who runs T_x; FP8 compact 137.5 vs 139.8 already measured flat); **[31]** 2 CTAs/SM or token-tile doubling (infeasible in 99 KB; <= 3%). **[30b]** third K buffer kept only for the latency-bound branch.

---

## Parallelism map

Independent worktrees (no shared files):
- A: sm90 consumer (Phase 1) — `mha_sm90.cu` z=0/1 bodies, consumer SharedMem fields, `barriers.cuh` NamedBarrier ids 3/4.
- B: sm90 IO (Phase 2) — `mha_sm90.cu` z=2/3/4 bodies, loader, `mhaUtils.cuh` sm90 convert path.
- C: sm90 scheduling (Phase 3) — host `chooseNbSubSeq`, kernel entry/indices, prologue, ScratchMem, PDL launch flags.
- D: FA3 (Track F) — hopper headers + `mixed_page_prefill.py`/jit only.
- E: sm120 + SPEC_DEC (Tracks S, W) — `mha.cu`, `defines.h`, `mhaUtils.cuh` copy/expand helpers.

Conflicts on the same code (serialize or rebase):
- A vs B: `SharedMem` struct (barrier list; [13] removes kLoadReady, [2] adds slots, [11] changes X depth/nbScaleTiles), barrier init block `:1185-1220`, setmaxnreg block. Rule: B owns the barrier init block; A adds its slots via a single appended member; [11] is built only after P0.3 and lands last.
- B vs E: `mhaUtils.cuh convertE4M3x2ToA16` ([16] sm90 shift path vs sm120 native cvt) — [16] must be `__CUDA_ARCH__`-guarded; E's [29] touches `copyMixedPartialHeadsAsync/expandMixedPartialHeadsInPlace` which sm90 does not use (mha_sm90 has its own expandPackedStage) — no overlap beyond the file.
- C vs A/B: [8] makes barrier parities running counters across items; it must rebase onto A+B's final barrier set. [36b] (Q load off warp 0) touches the loader (B) — hand it to B.
- Phase 4 ([15]/[34], [7]) touches every role: starts only after A, B, C merge; [15] before [7] (64-reg consumers).
- Track S step 3 (re-route q=4 to mha_sm90) touches `mha_sm90.cu` SPEC_DEC stage depth — coordinate with A/B, do last.

Judge agreement summary: unanimous do-first/do — [0], [1], [2], [4], [8], [11] (as trace read), [12], [13], [16], [21], [22], [23], [24], [25], [26], [27], [29], [32], [33], [35], [36]. Split — [7] (do/maybe), [14] (maybe/do), [15]/[34] (do/maybe), [19] (maybe/do), [37] (maybe/do), [5] (maybe/do-first: step 1 only). Unanimous maybe — [3], [6], [28], [30]. Unanimous reject — [9], [10], [17], [18], [31]; [38] maybe/reject.

---

## Gate check (50% of analytic speedup = the stated targets)

H200 sm90 decode (A16 83; targets FP8 <= 58, FP4 <= 36, mixed <= 62):
- FP8: after Phases 2+3 predicted 56-63; after Phase 4 ([15]) 54-60. Borderline pass; passes with confidence only if P0.4 returns wait/ILP-bound (then [15]'s -0.3..-0.5 us/tile is real) or P0.5 returns fair-share. Decided by the Phase 2 converter period: if it lands <= 2.4 us before Phase 3, FP8 passes on [8] alone.
- Mixed: after Phases 2+3 predicted 70-79; after Phase 4 60-68. Borderline; passing requires [13]'s TMA retarget to remove mixed's 20 us excess over the pure modes (verify in the Phase 2 trace: converter period for mixed <= fp8's) and [15]. If the mixed period stays above fp8's after [13], the A16 TMA issue path (16 boxes/tile by one elected lane) is the residual and needs the per-warp TMA issue of [15].
- FP4: analytically not reachable in this plan. The sm90-io budget needs ~4.4 IPC at 2 CTAs/SM for the software E2M1 decode (~2000 warp-instr per CTA-tile) plus today's consumer (2100); hardware max 4.0, measured 2.78. Every lever combined (Phases 1-4 incl. L8, SASS audit to <= 180 instr/lane-tile, persistence) gives a predicted 40-50 us (converter ~1.0-1.3 us/tile, consumer 0.3, 33 tile-times + 3 us). The smem floor of expand-to-A16 (81 KB per CTA-tile -> ~21 us) is below 36, so the gap is issue-side only; 36 would need the fp4 lane-tile at ~140 SASS and IPC ~3.5 sustained, which no measured number supports. Report FP4 sm90 as failing the gate unless the Phase 2 SASS audit finds the lane-tile can go below 150 with the consumer at L8's 0.3 us.
- A16: 83, DRAM/host-ceiling bound; parity holds (transport_a16 = baseline).

RTX 5090 sm120 (A16 180; targets FP8 <= 125, FP4 <= 79, mixed <= 135):
- Both P0.8 branches predict pass: (a) FP8 ~112 / FP4 ~68 / mixed ~118; (b) FP8 96-105 / FP4 68-75 / mixed 110-120. FP4 (84 today, the task text counts it as passing; plan acceptance stays <= 79) passes under both branches. Risk: branch (b) [26] hinges on the 99 KB static_assert with cacheVTileSeqLen 16; if it fails, Option B (3-deep K ring with double-part packed loads) is the fallback with the same request model.

H200 FA3 prefill (stock 300; requirement parity):
- A16 528 -> 300 +-3% predicted after [21]+[22] (+[20] if acq-dominated): pass.
- FP8/FP4/mixed 1500-1800 -> 300-330 predicted after [23]: pass at <= 1.1x, conditional on P0.7 not revealing an smem-port bound above ~1600 of 2800 cycles per tile; if it does, parity is still predicted (57% port utilization) but the margin shrinks to ~1.3x and must be re-derived from wavefront counts. Compressed streams cannot be faster than stock on this host (consumer issue-bound, 1.8x the tensor floor); no byte-proportional speedup is analytically available there, by the record's own C5.

SPEC_DEC q=4 (mixed 935 vs fp4 276):
- Predicted fixed by Track S steps 1-2 (REG <= 255, STACK 0, LDL 0; mixed <= 1.5x fp4). Not a gate target but mandatory; the residual after step 2, if any, is the exposed cp.async landing at kAhead = 1 in mha_sm90's SPEC_DEC path (step 3), decided by sizeof(SharedMem) arithmetic before any build.
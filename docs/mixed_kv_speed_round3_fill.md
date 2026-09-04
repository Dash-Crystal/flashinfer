# Round 3, lever "fill / prologue cut" — attribution, design (rev 3), as written

Kernel: `csrc/xqa/mha_sm90.cu`, persistent q=1 mixed build ([8] merged at
`039ba5c7`, tip `e113026a`), 5 warp groups (z=0 gemm0, z=1 gemm1, z=2 IO:
warp 0 K loader, 1 V loader, 2 Q warp, 3 scan + merge; z=3 K converters, z=4
V converters), 64-token tiles, K/V depth 3, P = 264 CTAs, 33 tiles per CTA.
Bench B=17, S=4096, 8 KV heads, GQA 4, D=128.  Line references are into
`mha_sm90.cu` of this worktree (`wt/r3fill`) AFTER the change (post-change
numbering, refreshed 2026-09-04 against the rev-3 code; `barriers.cuh`,
`decode.py` references unchanged).

State after [8] (nkcut2 H200, locked 5x5 medians): a16 78.8, fp8 67.8, fp4
60.5, mixed 64.4 us.  Wall model (fp8): 8.5 fill + 57.1 slow-member body +
2.8 tail.  This document attributes the fill by stamps and specifies the cut.

**Revision 2 (after the judge review).**  Three blockers changed the design
and its numbers: (1) the first design left the converters' first copy gated
on the *loader's* `consumed` arrive — program-ordered after the cold generic
walk — so the fast records alone would have cut ~0.4 us, not 3.7; the
`consumed` phase-0 protocol is now amended (converters do not wait `consumed`
for t < 3; section 3.1, C8').  (2) All "today" numbers and every accept /
reject threshold are now medians over the 3 launches of ONE final-layout log
(`/tmp/r3fill/trace5.log`, the only log the committed parser describes), with
the per-launch spread stated, and thresholds set outside that spread.  (3)
The 1.2 table is regenerated from that log by `parse_xqa_prologue.py
--summary`; run 3 / run 4 numbers are no longer cited.  Notes taken: finding 3
(gemm0 gap) is restated as *unattributed* with a new true-completion stamp;
`metaReadyFirst` is dropped (redundant with the `__syncthreads`); R6's
overflow condition is `P*T + (P-1) <= 2^32-1`; the scan runs at the launch
budget (48), not 40; the loader skip predicate is `gBeg == 0 && i0 < nbFast`.

**Revision 3 (verify-phase review, 2026-09-04; no kernel data-flow change).**
(1) C8' was incomplete: the converters' first real `consumed` wait (t = 3+s)
is a parity-1 wait, which returns immediately on a barrier still in phase 0;
section 4 now writes out the chain that makes it sound (the `kMetaReady[0]`
arrive sits after the loader's g < nbFast `arrive_and_wait`s) and states it
as the explicit invariant I-fill.  (2) The section-6 fallback "fill at
iteration 0 before the first arrive" violated I-fill; it is replaced by the
three sound variants (a)-(c) and the section-7 / section-8 references
updated.  (3) The C14 claim "debug builds assert" was dead code
(`-DNDEBUG=1` in `flashinfer/jit/xqa.py`); the check is now a trace-build
`trap;` and the 66-case matrix is named as the bench modules' only check.
(4) a16 `kl_iss(2)` and mixed `kc_ready(2)` rows with thresholds were added
to the section-7 table (the `g == nbFast` placement delays tile 2's issue
behind the walk in the TMA branch).  Two review items needed no change and
are recorded here so they are not re-raised: `mixedKvCtaTrace` (4096 x 48
u64) and `mixedKvCtaStamp` were already inside `#if MIXED_KV_TRACE`
(`mha_sm90.cu` :167-:217; the production cubin has no such symbol — checked
with `cuobjdump -elf` in section 10), and the runner docstring already counts
66 = 32 + 2 + 30 + 2 (`mixedkv_remote_run.sh` greps the runner's own
"passed" line and hard-codes no count).

**Revision 3a (implement-phase review, 2026-09-04; no blockers, notes only;
no kernel data-flow change).**  Notes taken into the text: (1) every
`mha_sm90.cu` line reference is now the post-change line (they were pre-change
numbers, so I-fill pointed at the wrong lines); (2) the attribution of the
13 -> 14 segment to the loader gate is an *inference* (trace5 has no stamp at
the converters' `issue(0)`) and the alternative reading is stated with the
stamp that decides it (section 6); (3) the "cold instruction fetch" model has
two readings (per-line latency on the chain vs. i-fetch bandwidth shared by
the SM's 40 warps) and the 0.30 budget for `issue(0)` is optimistic under the
second (finding 1, section 6); (4) R5's -0.4 us wall credit is optimistic
(no pacing role; the rings may absorb a one-time stall) and the mixed margin
is restated with it removed (section 6); (5) the IO group's register headroom
under `.dec 40` is < 8 registers by history and the loader now carries two
inlined `fillTileMeta` sites in its tile loop — the gating in sections 7 / 8
is what protects it (section 5); (6) `mixedKvTraceUse` is an empty `asm
volatile` and orders the stamp only in practice, not by construction (1.1
(a)); (7) the trace-only `test_wait` poll diverges converter warp 0's lane 0
(section 9); (8) C14 is enforced only by the trace-build trap and the 66-case
matrix (section 4).  The kernel's stale `fillTileMeta` header comment ("debug
builds assert") was corrected to the trace-build trap.

## 1. Attribution evidence

### 1.1 Method

Trace-only build (`#define MIXED_KV_TRACE 1`, checkout
`/home/bigboi/dash-flashinfer-claude-r3fill-trace`, JIT workspace
`/tmp/mixedkv-r3fill-trace`), `python benchmarks/xqa_mixed_trace_once.py
--modes fp8 fp4 transport_a16 mixed --q-len 1 --launches 3` under
`flock /tmp/mixedkv-gpu0.lock`.  Per-CTA `%globaltimer` stamps
(`mixedKvCtaTrace[cta][48]`; words 4..27 on the `TRACE ctarec ... prolog`
line, 28..39 on `TRACE ctaprolog2`, 40..47 on `TRACE ctaprolog3` — device
printf takes at most 32 arguments) taken by lane 0 of the role warp named in
the table; parsed by `benchmarks/parse_xqa_prologue.py` (per launch: median
over the 256 CTAs after dropping the 8 largest fills = co-tenant time slices,
`VLLM::EngineCore` at 100 % `utilization.gpu` throughout; `--summary`: the
per-launch medians and their median over launches, which is the form every
"today" and "accept" number below uses).  Per-tile `clock64` stamps have two
extra slots (16 = gemm0 after the item's Q wait, 17 = gemm0 after the QK
HGMMA commit, before `wait_group`), parsed by `benchmarks/parse_xqa_trace.py`.

Evidence log: `/tmp/r3fill/trace5.log` (nkcut2 `/tmp/r3fill-trace-5.log`),
the ONLY log whose stamp layout is the committed parser's (words 4..39; the
words 40..47 of this revision are not in it and read as absent).  Earlier logs
(`trace3.log`, `trace4.log`: intermediate stamp layouts; `trace27.log`: CTA 1
tiles 27-34 per-tile; `spin.log`: suspend-hint experiment) are kept as raw
material only; no number below is taken from them except the per-tile item
boundary cycles of finding 4 (from `trace27.log`, a per-tile `clock64` trace
the committed `parse_xqa_trace.py` reads).

Two stamp-placement facts, recorded so they are not re-learned: (a) `CS2R
SR_GLOBALTIMERLO` does **not** wait for `BAR.SYNC.DEFER_BLOCKING` — a stamp
placed right after `__syncthreads()` records the warp's *arrival*, not the
release; every "after X" stamp here is ordered by a dependent instruction
(`mixedKvTraceUse(loaded value)` or the predicate branch of a `try_wait`
loop).  Caveat: `mixedKvTraceUse` is `asm volatile("" :: "r"(v) : "memory")`
— it forces the value to exist (the load cannot be dropped) but emits no SASS
instruction, so ptxas is free to schedule the `CS2R SR_GLOBALTIMERLO` before
the load's consuming scoreboard wait.  The measured RT2 0.16 / RT3 0.35 are
real L2-hit latencies, so the stamps evidently did wait, but this is not
guaranteed by construction; if a future "after load" stamp reads implausibly
small, check the SASS order around the `CS2R` before believing it.  (b) An "arrive" stamp records the `SYNCS.ARRIVE` *issue*, not the
moment the release-arrive performs; the phase-completion time needs the
arrive token and a `test_wait` poll (new stamps 42 / 44, section 1.3 finding
3).  (c) The trace build costs ~0.5-0.7 us of fill (extra cold code and printf
frames): fp8 fill 8.99 here against 8.5 in the [8] histogram; all gains are
stated as differences inside the same build.  Trace build `ptxas -v` (fp8
module): no C7507, 48 registers, `USETMAXREG` 2, 256 B stack (printf frames)
with 4 B spill — trace only; the production TU of the trace-stamps commit was
byte-identical to `e113026a`.

Resolution: every prologue delta in the raw logs is a multiple of 32 ns (gcd
32 over 1916 values), so the sub-microsecond segment medians below are real.

### 1.2 The fill, stamp by stamp (trace5, 3 launches; per-launch medians of 256 CTAs | median over launches; us from CTA start)

fp8 (the wall's slow member is fp8/fp4; a16, mixed columns at the end):

| stamp (warp) | launch 0 | 1 | 2 | **median** | segment (us, median over launches; per-launch range) |
|---|---:|---:|---:|---:|---|
| 4 seqLen loads returned (IO warp 3) | 0.80 | 0.83 | 0.96 | **0.83** | RT1: warp launch + 17 lane-strided `ld.global.nc` (L2) |
| 5 `smem.sched` published (IO warp 3) | 1.44 | 1.47 | 1.60 | **1.47** | scan ALU **0.64** (0.64-0.64): 2 x `div_u64` CALL, pass 2, `seqLen1` load — cold code |
| 37 `__syncthreads` released (thread 0, dependent LDS) | 1.63 | 1.66 | 1.79 | **1.66** | published -> released **0.16** (0.16-0.16) |
| 30 gemm1 warp 0 at `vBar[0].produced` wait | | | | 1.92 | sync -> 0.26 (setmaxnreg.dec + pre-arrives) |
| 29 gemm0 warp 0 at `kBar[0].produced` wait (:1615) | 2.08 | 2.10 | 2.21 | **2.10** | sync -> **0.45** (0.42-0.45); warps 1-3 arrive at the same time (seg 29->27 = 0.00) |
| 38 Q warp `cursor.next` done | 2.43 | 2.50 | 2.59 | 2.50 | sync -> 0.80 (0.80-0.83) |
| 21 / 22 Q warp `qBar[0].consumed` passed / `produced.arrive` | 2.66 / 3.01 | 2.69 / 3.07 | 2.78 / 3.20 | 2.69 / 3.07 | Q is in smem 6 us before gemm0 needs it |
| 26 K fill phase A done, cold pass (IO warp 0) | 3.68 | 3.71 | 3.84 | **3.71** | sync -> A done **2.05** (2.05-2.05): setmaxnreg, preExit, cursor init, generic walk — cold |
| 7 K fill phase A, warm re-run of the same code | 3.90 | 3.94 | 4.06 | 3.94 | **0.22** (0.22-0.22): the same instructions warm |
| 8 page indices returned | 4.06 | 4.10 | 4.26 | 4.10 | RT2 **0.16** (0.16-0.16), L2 hit |
| 9 page formats returned | 4.45 | 4.38 | 4.61 | 4.45 | RT3 **0.35** (0.30-0.42), L2 hit; dead data in the static modules |
| 13 `kMetaReady[0].arrive` issued (:2316) | 4.83 | 4.70 | 4.99 | **4.83** | gather + STS + arrive **0.35** (0.32-0.38) — cold |
| 14 K conv warp 0: tile-0 copies committed | 5.31 | 5.22 | 5.44 | **5.31** | kMetaReady -> committed **0.45** (0.45-0.48): the `kMetaReady` wake, the **`consumed[0]` wait (completed only by the loader's arrive, program-ordered after stamp 13)**, and the cold issue code |
| 15 K conv warp 0: copies landed | 6.75 | 6.91 | 6.94 | **6.91** | RT4 **1.54** (1.46-1.70): start burst, 264 CTAs x 3 tiles x K+V = 13.5 MB in flight |
| 39 K conv warp 0: expanded | 7.26 | 7.42 | 7.46 | 7.42 | expand **0.48** (0.48-0.51) |
| 16 K conv warp 0: `produced.arrive` issued | 7.39 | 7.49 | 7.52 | **7.49** | fence + arrive issue 0.03; landed -> arrive **0.58** (0.51-0.58) |
| 31 / 32 / 24 K conv warps 1 / 2 / 3 arrive issued | 7.42 / 7.30 / 7.42 | 7.42 / 7.39 / 7.41 | 7.52 / 7.49 / 7.54 | 7.42 / 7.39 / 7.42 | inter-warp arrive-issue skew **0.00-0.03** (warp 0 is not the last) |
| 35 / 36 / 28 gemm0 warps 1 / 2 / 3 pass `kBar[0].produced` | 8.29 | 8.38 | 8.77 | **8.38** | last arrive issued -> pass **0.96** (0.87-1.25): the "gap", see finding 3 |
| 20 gemm0 warp 0 passes (after its per-tile `TRACE_STAMP`) | 8.64 | 8.74 | 8.90 | 8.74 | +0.19-0.36 after warps 1-3 = the warp-0 trace stamp code, cold |
| 1 firstk (after the Q wait, :1642) | 8.90 | 8.99 | 9.25 | **8.99** | Q `arrive_and_wait` **0.19** (0.19-0.19), phase long complete |

V mirrors K within 0.1 us (committed 5.31, landed 6.98 (RT4 1.60), arrive
issued 7.42, gemm1 passes 8.96 — launch 1: 8.35).

Other modes (same stamps, medians over 3 launches; per-launch ranges where
they matter):

| quantity | fp4 | mixed | a16 |
|---|---:|---:|---:|
| RT1 (4) | 0.77 | 0.74 | 0.74 |
| published (5) / scan ALU | 1.41 / 0.64 | 1.41 / 0.67 | 1.44 / 0.67 |
| sync released (37) | 1.57 | 1.60 | 1.60 |
| K fill A cold done (26) / sync -> A | 3.01 / 1.44 (1.34-1.54) | 3.46 / 1.86 (1.76-1.89) | 2.78 / 1.18 |
| A warm re-run (7-26) | 0.38 (0.22-0.42) | 0.45 (0.42-0.51) | 0.22 |
| RT2 / RT3 | 0.16 / 0.29 | 0.16 / 0.45 (0.29-0.48) | 0.16 / 0.29 |
| kMetaReady arrive (13) | 4.19 | 4.77 | 3.84 |
| K conv committed (14) / kMetaReady -> committed | 4.74 / 0.51 | 5.47 / 0.42 (0.38-0.74) | 3.97 (nothing to copy) |
| RT4 (15-14) | 1.70 (1.57-1.73) | 2.82 (2.78-2.91) | (0.38, empty group) |
| expand (39-15) / landed -> arrive (16-15) | 0.45 / 0.48 | 1.28 / 1.34 | — |
| gemm0 warps 1-3 pass (28/35/36) | 7.84 | 9.82 | 9.54 |
| gap: last conv arrive issued -> gemm0 warps 1-3 pass | 0.96 (0.90-1.25) | 0.32 (0.29-0.39) | 5.12 = TMA landing of tile 0 in a 264 x 3 x 32 KB = 25 MB burst |
| firstk (1) | 8.21 (8.00-8.45) | 10.05 (9.86-10.18) | 9.71 (9.38-9.76) |

a16 is DRAM-bound from its first TMA issue (loader iteration 0, right after
stamp 13 at ~3.9 us) on; only the ~3.9 us before that issue is a16's fill lever.

### 1.3 Findings, ranked by size

1. **Cold instruction fetch is the largest single component: ~3.3 us of the
   9.0 us fill on fp8.**  Every straight-line prologue segment runs 5-9x
   slower than the same code warm: phase A 2.05 vs 0.22 (fp8; the same
   instructions re-run under `#pragma unroll 1`; 1.44 vs 0.38 fp4, 1.86 vs
   0.45 mixed, 1.18 vs 0.22 a16), the scan's 0.64, the fill epilogue 0.35,
   kMetaReady -> committed 0.45 (steady-state `kl_iss - kl_start` 0.06-0.09),
   expand 0.48 (steady 0.40).  The kernel is 4696 SASS instructions (75 KB) in
   the fp8 module; each role's prologue runs once per CTA on 264 CTAs that
   start within 0.3 us, so every instruction line misses the SM's instruction
   cache.  ncu PC sampling cannot see it (0 samples in the prologue regions;
   94.7 % of all samples are `BAR.SYNC` waits, `/tmp/r3fill/r3fill_ncu_src.csv`).
   The cold cost is *not* a usable function of the instruction count: the
   identical phase-A code costs 2.05 / 1.44 / 1.18 / 1.86 us in the four
   modules; predictions in section 6 are therefore segment-based (measured
   segments reused or scaled), never count-based.  What the design controls is
   *which* segments sit on the chain before the first copy is issued.
   Two readings of "cold" are consistent with the data and are NOT separated
   by trace5: (i) per-line i-cache miss latency on each role's own chain, or
   (ii) i-fetch bandwidth shared by the 40 warps of the SM (2 CTAs x 20 warps
   fetching distinct code regions concurrently at start).  Under (ii) the
   converters' `issue(0)` code competes with the loader's walk being fetched
   in parallel, so moving the walk off the converters' *dependency* chain
   helps less than the segment arithmetic of section 6 says; the section-7
   gates (stamps 41 / 14) decide, not the model.  Also part of the "cold
   walk": `ItemCursor::next` issues a dependent `getCacheSeqLen(req + 1)`
   load whenever the chunk-0 walk crosses a request (~3 % of CTAs at B = 17,
   S = 4096; irrelevant to the median here, relevant for small-S batches).
2. **The converters' first copy waits for the loader twice (blocker 1).**
   `issueKCopies(0)` waits `kMetaReady[0]` (the loader's chunk-0 fill) and
   then `kBar[0].consumed.wait_parity(false)`; `consumed` is initialised with
   count 128 + 32 (:1436-1440), so phase 0 completes only when the loader's
   `consumed.arrive_and_wait()` at its iteration g = 0 (:2354 / :2423) has
   performed — program-ordered after `fillTileMeta(0)` + `metaReady[0].arrive`.
   Removing the `kMetaReady` dependency alone (rev 1's R1) leaves the copy
   gated on the same cold walk through `consumed`.  Both gates are on the
   loader's chain, ~4.6-4.9 us after start; the design must take the loader
   off the converters' tile-0..2 chain entirely (3.1).
3. **Start burst: RT4 = 1.54 us (1.46-1.70) for fp8 tile 0, 1.70 fp4, 2.82
   mixed**, against 1.2-2.0 in steady state, because every CTA issues tiles 0,
   1 and 2 of both operands within 0.4 us (13.5 MB fp8, 7 MB fp4; 3.4 / 1.8 us
   at 4 TB/s); tile 0's pages share the DRAM queues with tiles 1-2 of all 264
   CTAs.  a16's TMA loader issues 8 boxes per tile at ~0.1 us each, which
   serialises its burst by tile; the cp.async converters do not.
4. **gemm0 passes its first `produced` wait 0.96 us (0.87-1.25) after the
   last converter warp's arrive was *issued* (fp8; fp4 0.96, mixed 0.32) —
   unattributed.**  Rev 1 called this "cold post-wait code"; that does not
   hold: the code between the `try_wait` exit and stamp 28/35/36 is a handful
   of instructions (~0.1 us even at the cold rate), and the arrive stamps are
   *issue* stamps (1.1 (b)) — a release-arrive performs only after the warp's
   expansion STS have performed behind the SM's MIO / LDGSTS traffic, so the
   phase may complete well after the last `SYNCS.ARRIVE` issued.  Consistent
   with that: the gap follows the dispatch slot (the CTA whose converters
   finish while the co-resident CTA's tile-0..2 copies are still landing sees
   the larger gap) and the module (mixed 0.32 with its copies long landed).
   The suspend-hint experiment (`kSUSPEND_TIME_HINT` 0x80, `spin.log`) is
   consistent with either reading.  Consequence: the gap is *not* declared
   structural; R3 (smaller burst, fewer in-flight LDGSTS at the arrive) may
   move it.  Stamps 42 / 44 (converter warp 0 lane 0 keeps its arrive token
   and polls `test_wait` — the true phase completion) split the gap into
   "arrive performs" and "gemm0 wakes" in the confirmation trace; steady-state
   tiles need the same split before the gap is called "prologue only" (per-tile
   trace: `kc:done(warp 0) -> g0:kwait` is 0.9-1.6 us in steady state too, and
   warp 0 is not the last of the four).
5. **Item boundary +0.4 us on gemm0 (K-wait -> mma 1869 / 1808 cyc) is the
   `qBar[j&1].produced.arrive_and_wait()` at the item's first tile** (:1642):
   `trace27.log` (CTA 1 tiles 27-34, boundary 30 -> 31): `g0_kwait ->
   g0_qwait` 1113-1351 cyc on tile 31 against 175-560 on tiles 27-30 and 32 in
   every launch (fp8 and fp4), while `qwait -> issued` (633-696) and `issued
   -> mma` (23-41) are unchanged.  The phase is long complete (Q stored at ~3
   us); the cost is 128 gemm0 arrivals on one mbarrier word plus the token
   wait — the same mechanism costs 0.19 us on tile 0 of every CTA.
6. **RT1 (0.83 us) is the only real round trip on the chain**; RT2 / RT3 are
   L2 hits (0.16 / 0.29-0.45), not the 1-1.5 us of the [8] design.
7. **Q is not on the path** (Q(0) in smem at 3.07, needed at 8.7; after the
   change needed at ~5.4 — still 2 us of slack).
8. **The scan warp has `(req0, head0, tile0, seqLen0, seqLen1)` at 1.47 us**
   and can compute tile 0/1's page-table addresses in ~20 instructions — 2.2
   us before the loader's generic walk finishes.  This is the design's main
   lever (R1).
9. Steady state is unchanged by the trace build: fp8 gemm0 1.26 / gemm1 1.32
   / K-conv 1.23 us, fp4 1.29 / 1.26 / 1.31 (tiles 3-7, CTA 0).

### 1.4 What was ruled out

- `qScalePtr` / `kvScalePtr` derefs at entry (:1281-1282): the bench passes floats
  (`flashinfer/decode.py:3676-3677`), so both pointers are null; no round trip.
  (A tensor `bmm1_scale` in production adds one dependent L2 round trip for
  every thread before the scan — noted for PDL, 3.7, and 8.10.)
- DRAM latency of the page table / formats: L2 hits (0.16-0.45 us).
- mbarrier suspend granularity (experiment, finding 4).
- Late gemm0 warps (warps 0-3 at the wait within 0.05 us of each other).
- Rev 1's claim "the loader's `consumed` waits are pre-arrived" was about the
  *loader's own* wait (gemm0's pre-arrive), not about the loader's *arrival*
  the converters need — withdrawn (finding 2).

## 2. Current data flow and control flow of the touched roles

Prologue chain today (fp8; role, lines):

    all 640 thr   qScale/kvScale deref (:1281-1282, null) ; warpIdx ; wid 0: tensor-map prefetch
    wids 0..8     barrier init: kBar/vBar produced 128+128 (+32 TMA), consumed 128+32; xBar 256/256;
                  qBar[2] 160/160; gemm0/gemm1 named bars; kMetaReady/vMetaReady[2] count 32
    IO warp 3     persistentPrologueScan: pass 1 seqLen loads + warpSum (RT1);
                  T ; x0 = ceil(c*T/P), x1 (two div_u64 CALLs) ; pass 2 ; seqLen1 load ; lane 0 publishes sched
    all           __syncthreads  -> released 1.66 us
    all           nbCtaTiles = sched.x1 - sched.x0 ; setmaxnreg .dec 40 (z<=2) / .inc 56 (z>=3)
    gemm0         pre-arrive qBar[*].consumed, kBar[*].consumed ; kBar[0].produced.arrive_and_wait
                  ; LDS tile word ; if first: runningColMax reset, qBar[b].produced.arrive_and_wait ; HGMMA
    gemm1         pre-arrive vBar/xBar consumed ; vBar[0].produced.arrive_and_wait
    IO warps 0/1  preExit ; ItemCursor::init(sched) ; fillTileMeta(chunk 0):
                  A. cursor walk + capture x2  [2.05 us cold / 0.22 warm]
                  B. page loads (RT2 0.16) then page_format loads (RT3 0.35; dead in static modules)
                  C. shfl gather, STS.32 x2, STS.128 x2, __syncwarp
                  metaReady[0].arrive ; per-tile loop: consumed.arrive_and_wait, (TMA boxes), chunk refill at g+4
    IO warp 2     preExit ; cursor ; per item: QCvt::load ; qBar[b].consumed.arrive_and_wait ; store ; fence ; produced.arrive
    converters    global scales (one L2 RT) ; makeExpandLane / makeExpandScales ;
                  prologue: for t in {0,1}: issue(t) = [t%16==0 -> kMetaReady wait_parity] ;
                  kBar[t%3].consumed.wait_parity ; issueCompressedPageCopies ; commitGroup
                  loop: waitGroup<1> ; __syncwarp ; expand ; fence ; produced.arrive ; issue(t+2) ; commit
    host          dimGrid {P,1,1} ; makeLaunchConfig(..., enable_pdl && !ENABLE_MIXED_KV_CACHE)

The chain: converters' `issue(0)` needs `kMetaReady[0]` **and** `consumed[0]`
phase 0; both complete only after the loader's `fillTileMeta(0)` (cold
generic walk + RT2 + RT3 + gather) and its arrive at g = 0; the loader needs
`sched`, which needs the scan; gemm0 needs 128 converter arrivals plus its own
128 and then the Q arrive-and-wait.

## 3. New data flow and control flow

Seven items; R1-R6 are the round-3 change, R7 is the PDL design (outside
gate numbers).

### 3.1 R1 — fast-path records for tiles 0 and 1 from the scan warp; loader off the tile-0..2 chain

IO warp 3, after pass 2 and before it publishes `sched` and arrives at the
`__syncthreads` (no new barrier):

    have: x0, x1, req0, head0, tile0, Lseq0, seqLen0, seqLen1 (lane-uniform)
    G = x1 - x0 ; nbFast = min(2, G)
    piece g = 0:  (req0, head0, t = tile0, tl0 = tiles(seqLen0), Lseq0, seqLen0)
    piece g = 1:  if tile0 + 1 < tl0:           same sequence, t = tile0 + 1
                  else if head0 + 1 < H:        (req0, head0 + 1, t = 0, Lseq0 + tl0, seqLen0)
                  else if tiles(seqLen1) > 0:   (req0 + 1, 0, t = 0, Lseq0 + tl0, seqLen1)
                  else:                          nbFast = min(nbFast, 1)    (generic skip loop needed)
    lane l = 4g + j, active iff l < 4*nbFast:
        idxPage = 4 * (skipTiles(seqLen_g) + t_g) + j
        page = idxPage < divUp(seqLen_g, 16) ? kvCachePageList[req_g * maxNbPagesPerSeq + idxPage] : kBAD  (RT2)
        fmt  = page == kBAD ? kMixedBadPageFormat : (static module: MIXED_PAGE_STATIC_FORMAT | mixed: page_format[page])
    formats_g = shfl gather into lane 4g ; word_g = makeTileWord(x0 + g, t_g, tl_g, x0, x1, seqLen_g, sw, tileItemBits(Lseq_g, tl_g, x0, x1))
    active lanes: STS.32 page into meta[K][0][g].pages[j] and meta[V][0][g].pages[j]
    lane 4g:      STS.128 {formats, word, req, head} into both operands' records
    lane 0: sched.nbFast = nbFast ; publish sched as today ; fall through to the __syncthreads

Converters (`issueKCopies` / `issueVCopies` and the prologue; the loop body
is unchanged):

    issue(t):  if (t >= nbFast && (t == nbFast || t % 16 == 0))
                   kMetaReady[(t/16)%2].wait_parity(toParity<1>(t/32))      // chunk 0's first generic tile is t = nbFast
               if (t >= 3) kBar[t%3].consumed.wait_parity(toParity<3>(t))    // C8': stages 0..2 are initially free
               issueCompressedPageCopies(record t)                             // unchanged
    prologue:  issue(0) ; commit ; [R2 constants] ; waitGroup<0> ; issue(1) ; commit   // R3
    loop:      as today

`nbFast` is one warp-uniform LDS of `smem.sched.nbFast` after the sync.

Loader (both branches; `fillChunk0()` = `fillTileMeta(op, gBeg 0, nbSkip =
nbFast, cursor)` + `metaReady[0].arrive`):

    for g in [0, G):  if (g == nbFast) fillChunk0()
                      [TMA branch: LDS record g (scan warp's for g < nbFast)]
                      consumed[g%3].arrive_and_wait ; [TMA boxes] ; chunk refill at g + 4 as today

`fillTileMeta` skips the STS of entry 0 for lanes with `i0 < nbSkip`
(`gBeg == 0 && i0 < nbFast`, entry 0 only: chunk-0 slots 0-1 are legitimately
rewritten by the fill for tiles 32-47 at iteration 28 with `nbSkip = 0`); it
still computes those entries, and trace builds (`MIXED_KV_TRACE`) trap if they
differ from the fast records (C14; no bench module has a check:
`flashinfer/jit/xqa.py` passes `-DNDEBUG=1`, so an `assert` would be dead).
Static modules take the tag from the compile-time format (R4).

Why the loader moves: (i) a16 / mixed — the TMA loader's tile-0 and tile-1
boxes are issued from the fast records at iterations 0-1 *before* the cold
generic walk, which is a16's only fill lever (its first TMA issue moves from
~3.9 us to ~2.5); (ii) the fill's position does not matter for the static
modules (iterations 0-2 are one arrive each) and one rule for both branches
is preferred over two.  The chunk-0 fill still happens exactly once per CTA
with G > nbFast; a CTA with G <= 2 has every tile fast and never fills or
waits chunk 0 (kMetaReady[0] then has no waiter; C8 holds vacuously).

Effect: the converters' first copy waits for the scan warp's fast path
(~1.7 us) + the sync + their own `setmaxnreg.inc` instead of the loader's
cold walk + RT2 + RT3 + gather + arrive (4.83).  Alternative considered and
rejected (3.9): the loader pre-arrives `consumed[s]` for s < 3 before its fill
and waits by parity at g < 3 — it works, but adds a second arrive protocol
for the same barrier and needs its WAR argument restated; "no wait for t < 3"
is one line and its argument is C8' below.

### 3.2 R2 — converters issue before loading their expansion constants

The global-scale loads (:2900-2907) and `makeExpandLane` / `makeExpandScales`
(:2914-2915) move to right after the first `commitGroup` (before the R3
`waitGroup<0>`).  They are needed by `expandPackedStage` only; the tile-0
landing (~1.1 us) covers the L2 round trip.  Register liveness during
`issue(1)` is as today (the sets were live across both issues before).

### 3.3 R3 — tile-ordered start burst

Converter prologue: `issue(0) ; commit ; ldgsts::waitGroup<0>() ; issue(1) ;
commit`, then the loop as today.  The loop invariant "at iteration i the
outstanding groups are tiles i and i+1; `waitGroup<kAhead-1>` completes tile
i" is unchanged: the prologue still commits exactly two groups; the inserted
`waitGroup<0>` is a wait, not a commit (C16).  DRAM sees 264 x 2 x 8.5 KB =
4.5 MB (fp8) / 2.6 MB (fp4) before any tile-0 byte is queued behind a tile-1
byte.  Tile 1 is issued at K0-landed and lands by ~4.9-5.5 against gemm0's
need at firstk + 1.2 ~ 6.6; tile 2 is issued at loop iteration 0 as today.
Not applied to the a16 TMA loader (DRAM-bound; its boxes already serialise by
tile) — recorded as rejected.

### 3.4 R4 — static-format modules skip the `page_format` load

`fillTileMeta` phase B and the R1 fast path: under `MIXED_PAGE_STATIC_FORMAT
>= 0`, `fmt = (page == kBAD_PAGE_INDEX) ? kMixedBadPageFormat :
MIXED_PAGE_STATIC_FORMAT` with no load.  The consumers already override the
tag (:2371-2377 TMA loader, :3470-3472 converters), so the records carry the same
bytes as today whenever `page_format` agrees with the static tag — the only
valid configuration of a static module (C14).  Saves RT3 (0.29-0.35) on every
fill of the static modules — off the critical path once R1 exists, but it
moves the loader's chunk-0 completion earlier, which is the slack the
converters' `issue(2)` (first in-loop issue, at expand(0) done) waits on.

### 3.5 R5 — gemm0's Q wait becomes a parity wait

`qBar[b].produced` init count 160 -> 32 in the persistent build (:1467:
`nbQLdThrds`; `consumed` stays 160; the non-persistent build keeps 160/160 and
its `arrive_and_wait` at :1585 under `#else`).  gemm0 at the item's first tile
(:1642): `qBar[idxQBuf].produced.wait_parity(toParity<2>(idxItem))`.  Item j
uses buffer j & 1 and is that buffer's phase j >> 1; the Q warp's 32 arrivals
complete it.  Replaces 128 arrivals + a token wait by one `try_wait` per
thread that returns on the first test (the phase is complete ~40 us early):
-0.4 us at every item boundary on gemm0's path and -0.15 on tile 0 (C17).

### 3.6 R6 — 32-bit partition arithmetic when it fits

`x_c = ceil(c*T/P)` (:3580-3586): the largest numerator is c = P: `P*T +
(P-1)`.  If `T*P + (P-1) <= 2^32-1` (64-bit multiply, no divide; bench T =
8704, P = 264), both divisions are inline u32; else the 64-bit path as today.
(Rev 1's `T <= 0xFFFFFFFF / P` was off by one: P = 5, T = 858993459 wraps the
last CTA's x1.)  Warp-uniform branch; removes two cold 68-instruction
subroutine executions from the chain.  The merge warp's three `div_u64` sites
are off the path and stay.

### 3.7 R7 — PDL for the mixed build ([37], design only, unchanged from rev 1)

Host: `makeLaunchConfig(..., enable_pdl)` for the mixed build (:5518, :5639
drop `&& !ENABLE_MIXED_KV_CACHE`).  Kernel: the first dependent global access
of every warp must follow `griddepcontrol.wait` (`acqBulk()`):

| warp(s) | first dependent global access today | placement of `acqBulk` |
|---|---|---|
| all threads | `qScalePtr[0]` / `kvScalePtr[0]` (:1281-1282) | move the two derefs after the wait |
| IO warp 3 | `getCacheSeqLen` in the scan | first instruction of `persistentPrologueScan` (the fast path's page-table loads follow it) |
| IO warps 0/1 | page table in `fillTileMeta` | after `setmaxnreg`, before `ItemCursor::init` |
| IO warp 2 | `QCvt::load` | same point |
| converters | page copies (first), global scales (after R2) | after `setmaxnreg.inc` |
| gemm0 / gemm1 | none (smem only) until the output / scratch writes | after `setmaxnreg` |

`preExit()` stays at the IO group start.  Verification is SASS (`ACQBULK` in
every role's path, one `PREEXIT`) and an nsys trace of a two-kernel stream;
not in this round's gate numbers.

### 3.8 One flag function (code-shape rule for R1)

`tileItemBits(Lseq, tl, x0, xEnd)` (partial / ctaLast) and `makeTileWord(x,
t, tl, x0, xEnd, seqLen, slidingWinSize, itemBits)` are `__forceinline__`
helpers used by `fillTileMeta`'s capture, by the fast path, and (item bits)
by `ItemCursor::next`, so the three places that derive item / tile flags share
one formula; `seqTilesInUse` / `seqSkipTokens` are the shared primitives.

### 3.9 Rejected alternatives (so they are not re-proposed)

- `metaReadyFirst` (rev 1, C15): a 32-count barrier initialised and arrived by
  IO warp 3 after its STS — redundant: the STS precede the warp's
  `__syncthreads` arrival and every converter wait is after the sync, so the
  sync gives the happens-before already.  Dropping it removes an init, 32
  arrives and one cold PHASECHK from the chain.
- Loader pre-arrives `consumed[s]` (s < 3) before its fill and waits by parity
  at g < 3: correct but a second protocol on one barrier (3.1).
- Loaders run their own scan and issue RT2 before the sync ("F2"): saves the
  0.16 us sync only; the cold walk stays on the chain.
- Converters compute their own tile-0/1 page addresses from `sched` (no
  record write): breaks C10 for g < 2 (gemm0's record read would no longer be
  ordered after the writer) and needs a gemm0/gemm1 wait on `kMetaReady[0]`;
  same cold-code count as R1.  Fallback if the scan warp spills at 48.
- Generic `fillTileMeta(limit = x0 + 2)` in the scan warp: the cold cost of
  the generic walk is per instruction line, not per iteration — it would move,
  not shrink.
- Suspend-hint / polling changes to the mbarrier waits: no effect measured.
- V converters delaying their first issue behind `kBar[0].produced`: a
  cross-role wait for ~0.5 us of burst; R3 already takes the burst to 4.5 MB.
- Pre-warming gemm0's first-tile code or the converter's issue code by a
  dummy execution: the code has side effects (wgmma, cp.async, barriers).

## 4. Barrier, parity and ownership invariants

D1-D6 and C1-C13 (`mixed_kv_page_transport_dataflow.md`,
`mixed_kv_speed_round2_lever8.md` sections 3 and 8) are unchanged in
statement except C8 for `consumed` (restated as C8') and C10 for g < nbFast;
C14, C16, C17, C18 are new; rev 1's C15 is withdrawn.

- **C8' (`kBar/vBar.consumed`, consumer-gated parity with a free start).**
  `consumed[s]` keeps count 160 (128 gemm0 + 32 K loader; V: 128 gemm1 + 32 V
  loader).  Phase 0 completes with the GEMM group's pre-arrive (:1559 /
  :1822) and the loader's `arrive_and_wait` at its iteration g = s; phase
  k >= 1 with the GEMM group's end-of-tile arrive for tile 3(k-1)+s and the
  loader's arrive at g = 3k+s.  Converters wait phase k-1 (parity
  `toParity<3>(t)` = `t % 6 / 3`) at `issue(t)` for t = 3k+s, k >= 1 —
  unchanged — and do **not** wait for t < 3: the stage is initially free and
  no reader of stage s exists before tile s (gemm0 reads stage s first at tile
  s, after the converters' `produced` arrive for tile s).
  **Soundness of the first real wait (t = 3+s, parity 1).**  `wait_parity(p)`
  returns as soon as the barrier's current phase parity differs from p
  (barriers.cuh :350-358, "starting from parity = false"); on a barrier still
  in phase 0 a parity-1 wait returns immediately.  So the wait at t = 3+s
  guards stage s only if phase 0 of `consumed[s]` has already completed when
  the converter reaches it.  Nothing in the converter waits for that phase
  directly; it is guaranteed by this chain and by nothing else:
  1. `issue(3+s)` is program-ordered after `issue(nbFast)` (nbFast <= 2 < 3),
     and `issue(nbFast)` waits `kMetaReady[0]` phase 0 (C8).
  2. `kMetaReady[0]`'s only arrive is inside `fillChunk0()`, which the loader
     calls at the TOP of iteration g == nbFast, i.e. AFTER its
     `consumed[g % 3].arrive_and_wait` for every g < nbFast — each of which
     returns only when phase 0 of `consumed[g]` is complete (it includes
     gemm0's pre-arrive).  Hence for every stage s < nbFast phase 0 is
     complete before any converter passes `issue(nbFast)`, and the parity-1
     wait at t = 3+s is a real wait for phase 1.
  3. Stages s >= nbFast (s = nbFast..2) are covered transitively by the wait
     at t = 3 on stage 0 (nbFast >= 1 whenever G >= 1 — `nbFast = min(2, G)`,
     lowered to 1 only by the empty-next-request branch — so stage 0 is in
     case 2 and the t = 3 wait is real): phase 1 of `consumed[0]` needs the
     loader's arrive at g = 3, program-ordered after its `arrive_and_wait` at
     g = 1 and g = 2 (phase 0 of `consumed[1]`, `consumed[2]` complete), and
     gemm0's end-of-tile-0 arrive, program-ordered after its pre-arrives;
     `issue(4)`, `issue(5)` are program-ordered after `issue(3)`.  V: the same
     chain with the V loader, gemm1 and `vMetaReady[0]`.
  4. CTAs with G <= nbFast never reach t = 3.  A CTA with G > nbFast always
     runs iterations 0..nbFast-1 before `fillChunk0()`.
  **Invariant I-fill (explicit; load-bearing for C8').**  In both loader
  branches `fillChunk0()` — and with it the only `metaReady[0].arrive` — is
  placed after iteration nbFast-1's `consumed.arrive_and_wait` and before
  iteration nbFast's own wait (`if (g == nbFast) fillChunk0();` at the top of
  the loop body, :2342 / :2420).  Moving the arrive earlier (before the g = 0
  wait) deletes step 2 and re-opens the race: a converter could pass
  `issue(nbFast)` and the t = 3 parity-1 wait before gemm0's pre-arrive or the
  loader's g = 0 arrive has performed, and issue tile 3 into stage 0 while
  tile 0 is still being read.  The fill's STS may move earlier; the arrive may
  not.  Any change that moves the arrive must restore the converters'
  t < nbKBuf `consumed` waits (parity 0) or adopt the loader pre-arrive
  protocol of 3.9 (section 6, fallback (b) / (c)).
  Aliasing: the wait at t = 3k+s precedes the completion of phase k+1 because
  that needs gemm0's arrive for tile 3k+s, which needs `produced` for tile
  3k+s, which needs this converter's arrive after `issue(3k+s)`.  The loader's
  own wait at g < 3 is unchanged (completes with its own arrive plus gemm0's
  pre-arrive) and its WAR role for the chunk refills (lever [8] 8.1: the wait
  at iteration g orders the refill after gemm0's `consumed.arrive(g-3)`) is
  untouched because refills happen at g = 12, 28, ... only.
- **C8 (`kMetaReady`)** — chunk 0's first wait moves from t = 0 to t = nbFast
  (1 or 2); fill 0 of chunk 0 (loader iteration g = nbFast) completes phase 0
  exactly once; the wait at t = nbFast precedes the next completion (fill 1 at
  loader iteration 28, gated on `consumed[1]` phase 9 -> gemm0 tile 25 -> the
  converters' `issue(25)`, after `issue(nbFast)`).  CTAs with G <= nbFast
  never fill and never wait chunk 0.  Chunks 1, 0, 1, ... at t = 16, 32, 48 as
  before.
- **C10 (record visibility and reuse)** — for g >= nbFast unchanged (writer:
  loader fill; readers after the wait the writer chain feeds).  For g < nbFast
  the writer is IO warp 3 (STS before its `__syncthreads` arrival); every
  reader (converters at `issue(g)`, gemm0 after `kBar[g%3].produced`, gemm1
  after `vBar`, the TMA loader at its iteration g) runs after the sync, which
  orders CTA shared memory.  Reuse: slot g < 2 is next written by the loader's
  fill for tiles 32-47 at iteration 28 (`nbSkip = 0`) — the WAR argument of
  [8] 8.1 is unchanged (last readers of tiles 0-1 precede gemm0's
  `consumed.arrive(25)`).
- **C14 (fast records equal generic records).**  Both writers compute the
  word through `makeTileWord` / `tileItemBits` from the same primitives on the
  same inputs; pages come from the same table addresses; the tag is the static
  format or the same `page_format` load.  Checks: none in any bench module
  (`flashinfer/jit/xqa.py` passes `-DNDEBUG=1`, so an `assert` never runs in
  a build in use); the trace build (`MIXED_KV_TRACE >= 1`) recomputes entry 0
  of tiles g < nbFast inside the loader's chunk-0 fill and traps on a mismatch
  with the fast record (`trap;` — a launch failure of
  `xqa_mixed_trace_once.py`); the 66-case conformance matrix (nbFast = 2 via
  the same-head / next-head / next-request branches, nbFast = 1 via midempty,
  nbFast in {0, 1} for T < P) is the only check of the bench modules.  Static
  modules: the recorded tag is the compile-time tag or `kMixedBadPageFormat`,
  which is exactly what every reader derives from a loaded tag today.
  Enforcement summary: C14 has NO production-build check; it rests on (i) the
  shared helpers (one formula, 3.8), (ii) the trace-build `trap;`, (iii) the
  66-case matrix.  A future edit to `ItemCursor::next` or the fast-path piece
  derivation must run the trace build once (a trap is a launch failure).
- **C16 (tile-ordered prologue issue keeps the cp.async group accounting).**
  Two commits in the prologue in both builds; `waitGroup<0>` is a wait;
  `waitGroup<kAhead-1>` at loop iteration i completes tile i.  Tiles past G
  still commit empty groups.  D3/D6 (ownership) untouched.
- **C17 (`qBar.produced` parity is unambiguous).**  Phase m of
  `qBar[b].produced` completes with the Q warp's store of item 2m+b; the Q
  warp stores item j+2 into buffer b only after `consumed[b]` completes, which
  needs all 128 gemm0 end-of-item arrives for item j (:1812), program-ordered
  after every gemm0 warp's wait on phase j>>1 — never two phases behind.
  `wait_parity` starts from parity false (barriers.cuh :352); `consumed` keeps
  160; the non-persistent build is under `#else`.
- **C18 (PDL ordering, R7).**  As rev 1; the fast path adds page-table loads
  to IO warp 3's first-access list — after the scan's `acqBulk`.
- **C4 (barrier accounting)** — `qBar[b].produced` 160 -> 32 (persistent
  build only); no barrier added.  `used 5 barriers` (named) unchanged.
- **C9 (item-agnostic tile stream)** holds: `nbFast` is a property of the
  CTA's first two tiles, read once per role; no ring, stage, barrier count or
  group count depends on items.
- **C11-C13** (scratch slot rule, merge hand-off, balanced partition) are not
  touched; R6 changes the arithmetic of x_c, not its value.

## 5. Register, shared-memory and issue budgets

Registers (`__launch_bounds__(640,2)`, pool 30720; split 3 x 128 x 40 + 2 x
128 x 56 = 29696 unchanged).  The scan and the fast path execute BEFORE
`setmaxnreg.dec` (the sync precedes :1541-1543), so their ceiling is the
launch allocation 48, not 40:

| role | change | live-set arithmetic | check |
|---|---|---|---|
| IO warp 3 (scan + fast path, at 48) | +fast-path state, transient before publish | scan state (~14) + per lane one page, one fmt, `t, tl, Lseq, len, req, head` for its own g (8), word, formats (2) -> ~26 + addresses; the merge state is not live yet | `ptxas -v`: 0 spill; fallback: converter-side fast path (3.9) |
| IO warps 0/1 (at 40) | `nbFast` (1, live across the tile loop), skip predicate (1); a SECOND inlined `fillTileMeta` site inside the tile loop (`fillChunk0` at `g == nbFast`, plus the refill) | history (`mixed_kv_page_transport_backends.md` :1316-1324): IO at `.dec 32` spilled 120 B, at 40 zero — the group's headroom under 40 is < 8 registers; the two sites are not simultaneously live (both are in the loop body, sequential), so the peak is one fill's live set + `nbFast` + the loop state | 0 spill, no C7507 — this is THE unverified risk of the change: a C7507 here drops every `setmaxnreg` (converters back to 48 with spills); gated by section 7 item 1 / section 8 items 2, 8; fallback: `fillChunk0` outlined (`__noinline__`) or the refill and chunk-0 fill merged into one call site with a runtime `gBeg` |
| converters (at 56) | `nbFast` (1); constants after issue(0) | ExpandLane/ExpandScales live across issue(1) as today | 0 spill; expansion SASS count unchanged (188 / 187) |
| gemm0 | parity wait replaces token wait (-1) | unchanged | 0 spill |
| gemm1, Q warp, merge | none | | |

Shared memory: `sched` 36 -> 40 B (`nbFast`); `sizeof(SharedMem)` 113 664 ->
113 664 or 113 792 (128-B alignment) <= 115 712; trace build (18 slots x 8
tiles) <= 115 072.  `static_assert(smemSize + 1024 <= 233472 / 2)` guards it.

Issue budget (C6): converters +3 ALU per issue (`t >= nbFast`, `t == nbFast`,
`t >= 3`), once per tile per warp against ~310; IO warp 3 +~60 instructions
once per CTA; gemm0 -127 arrivals per item.  No per-tile path gains a barrier
site: gemm0's `SYNCS.PHASECHK` count 8 -> 8 (arrive_and_wait -> wait_parity
is one PHASECHK either way, minus one `SYNCS.ARRIVE`); converters' PHASECHK
sites unchanged (the two waits gain predicates, no new site).

## 6. Predicted periods and wall

Steady-state periods do not change (no per-tile path is touched beyond three
predicates in the converters and a cheaper per-item Q wait in gemm0): fp8
gemm0 1.21-1.26 / gemm1 1.20-1.32 / K-conv 1.19-1.23, fp4 1.20-1.29 us.

Fill chain after R1-R6 (fp8, trace-build equivalent; every term is a
measured trace5 segment reused or scaled; "today" = trace5 median):

    0.83  RT1 (unchanged)                                                          -> 0.83
    0.40  scan ALU without the two div_u64 executions (today 0.64; ~2 x 68 cold instr)  -> 1.23
    0.25  fast-path arithmetic (~60 instr, cold, at the scan segment's own rate)   -> 1.48
    0.16  RT2 (L2, today 0.16)                                                     -> 1.64
    0.05  fmt / gather / STS / publish                                             -> 1.69   IO warp 3 arrives at the sync
    0.16  sync release (today 0.16)                                                -> 1.85
    0.45  converters' setmaxnreg.inc + nbFast LDS (today's sync -> gemm0-at-wait 0.45) -> 2.30
          [borrowed from gemm0's .dec path; the .inc path is NOT measured in trace5 -- stamp 41 measures it]
    0.30  issue(0) + commit, cold (today's kMetaReady -> committed 0.45 minus the consumed wait) -> 2.60   first K0 byte requested (today 5.31: -2.7)
          [optimistic under the i-fetch-bandwidth reading of finding 1; stamp 14 - 41 measures it]
    1.15  RT4, tile-0-only burst (4.5 MB; today 1.54 with 13.5 MB)                 -> 3.75
    0.58  expand + fence + arrive (today 0.58)                                     -> 4.33
    0.10  arrive skew inside the smaller burst (today 0.00-0.03 issue skew)        -> 4.43
    0.96  gap: last arrive issued -> gemm0 passes (today 0.96; NOT designed away)  -> 5.39
    0.05  Q parity wait (R5; today 0.19)                                           -> 5.44   firstk

Attribution gap (stated, not a blocker): trace5 has NO stamp for the moment
the converters reach `issue(0)`; "the converters are gated on the loader at
4.83" is inferred from the 13 -> 14 segment (0.45, per-launch range
0.29-0.70, tighter than stamp 13's own 4.48-5.12 spread across CTAs —
supportive but indirect).  Alternative reading: the converters' own cold
prologue (`setmaxnreg.inc` behind three `.dec` groups, the global-scale L2
round trip inside the start burst, cold `makeExpandLane`) ends at ~3-3.5 us
today; R1 then still helps, but by less than the predicted -2.7 on stamp 14.
The confirmation trace separates the two with stamp 41 (converter warp 0 at
`issueKCopies(0)`, after `setmaxnreg.inc` + the `nbFast` LDS): under the
design's reading 41 - 37 <= 0.6 and 14 - 41 <= 0.5; under the alternative
41 - 37 is ~1.5-2 and the fast records buy only the difference.  The reject
threshold on stamp 14 (> 3.8) catches either outcome; the 41 - 37 row in
section 7 attributes it.

| mode | fill today (trace5) | predicted | change | bench-equivalent ([8] histogram) -> predicted |
|---|---:|---:|---:|---|
| fp8 | 8.99 (8.90-9.25) | 5.4 (5.1-5.9 with the gap 0.7-1.25) | -3.5 | 8.5 -> ~5.0 |
| fp4 | 8.21 (8.00-8.45) | 5.1 (RT4 1.0 with 2.6 MB; expand 0.48; gap 0.96) | -3.1 | 7.4 -> ~4.5 |
| mixed | 10.05 (9.86-10.18) | 6.3 (+0.30 RT3 on the fast path; RT4 1.6; expand 1.34; gap 0.32) | -3.7 | ~9.5 -> ~5.9 |
| a16 | 9.71 (9.38-9.76) | first TMA issue ~3.9 -> ~2.5 (sync 1.85 + setmaxnreg/cursor 0.45 + LDS/wait/issue 0.2); DRAM-bound after it | -1.4 | 6.6 -> ~5.2 |

Chunk-0 slack, fp8 / fp4 (the converters' `issue(2)` at expand(0) done ~4.4
waits `kMetaReady[0]`): the loader enters `fillChunk0` at iteration 2 (~sync +
0.45 + 2 x ~0.05) ~2.4, walk 2.05 cold, RT2 0.16, gather/STS/arrive 0.35 ->
~5.0 (R4 removes RT3 0.35 from today's 4.83 + 0.15 of iterations).  That is
~0.6 us LATER than issue(2) wants it: the converters' iteration 0 stalls ~0.6
us at `issue(2)`; tile 2 is needed by gemm0 at firstk + 2.4 ~ 7.8 and lands
by ~6.3, so no consumer stall is predicted, but the confirmation trace must
show `kc_ready(2)` within the steady period (section 7).

Chunk-0 slack, a16 and mixed (the `g == nbFast` placement puts the cold walk
between the TMA issue of tile 1 and that of tile 2): a16 `kl_iss(2)` ~ sync
1.85 + setmaxnreg / cursor 0.45 + two iterations 0.2 + walk 1.18 (no RT3 in
the a16 module) + RT2 0.16 + gather / STS / arrive 0.35 ~ 4.2-4.3 us against
~4.0 today (today the walk precedes tile 0's issue and tile 2 follows tile 1
by one iteration).  gemm0 needs tile 2 two a16 periods (~2 x 2.1 us) after
tile 0 lands (> 7 us), so +0.3 on tile 2 is absorbed by the depth-3 ring and
the gain is tiles 0-1 issued ~1.4 us earlier.  mixed: the converters'
`issue(2)` is gated by `kMetaReady[0]` at ~4.3 + RT3 0.45 ~ 5.3 us (the mixed
module keeps the `page_format` load in the walk) against issue(2)'s want at
~4.9 (RT4 1.6): a ~0.4-0.5 us stall of the converters' iteration 0, and tile
2's A16 boxes also go out at ~5.3; tile 2 is needed at firstk + 2.4 ~ 8.7 and
lands by ~7.0 -> no consumer stall predicted.  Both have explicit rows with
thresholds in the section-7 table.

Fallback if `kc_ready(2)` is late in the static modules.  The obvious
one-liner — "fill chunk 0 at iteration 0, before the first arrive, as today"
— is **unsound as stated** and must not be built: it moves the
`kMetaReady[0]` arrive before the loader's g = 0 `arrive_and_wait`, deletes
step 2 of C8' (I-fill) and re-opens the stage-0 race — a converter can pass
`issue(nbFast)` and the parity-1 wait at t = 3 while `consumed[0]` is still
in phase 0, i.e. before gemm0's pre-arrive / the loader's g = 0 arrive has
performed, and issue tile 3 into stage 0 under gemm0's read of tile 0.  The
sound variants are:

(a) split the fill: the `fillTileMeta(nbSkip = nbFast)` STS at iteration 0
    before the first wait, `metaReady[0].arrive` kept at the top of iteration
    g == nbFast (I-fill holds; chunk-0 entries >= nbFast have no reader before
    the arrive).  Static modules gain only the g < nbFast iterations (~0.1
    us), so it cannot close a 0.6 us gap; a16 / mixed lose the
    tiles-0-1-before-walk ordering — not for them.
(b) fill + arrive at iteration 0 paired with restoring the converters'
    `consumed` waits for t < nbKBuf (parity 0; the [8] code): sound, but the
    converters' `issue(0)` then waits for the loader's g = 0 arrive, which
    sits behind the walk — that is the [8] chain; R1's gain on the static
    modules is lost.
(c) fill + arrive at iteration 0 paired with the loader pre-arrive protocol
    of 3.9: the loader arrives `consumed[s]` for s < 3 before the fill and at
    g < 3 waits by parity without arriving; the converters restore the
    t < nbKBuf waits with parity 0, which now complete with the two prologue
    arrives (gemm0's and the loader's, both before any walk) and are cheap.
    Sound and keeps the chain; costs a second arrive protocol on `consumed`
    and a restated WAR argument (3.9).

If the confirmation trace triggers the fallback, (c) is the one to design
and review before building; (a) and (b) are listed so that neither is
proposed as a one-line fix.

Item boundary (R5): -0.4 us on gemm0's path per boundary; ~half the CTAs have
one (33-tile ranges cross a 64-tile sequence boundary with probability 33/64),
so the slow member of the wall almost surely does.  The WALL credit is
optimistic: all roles run at the same ~1.2 us cadence with no pacing role, so
a one-time 0.4 us gemm0 stall propagates to the CTA's end only to the extent
the depth-3 K ring / X ring 2 cannot absorb it (the converters keep issuing
while gemm0 stalls; the ring has one free stage at the boundary).  The
per-tile accept row (`g0_kwait -> g0_qwait` <= 600 cyc) verifies the
mechanism, not the wall credit; the wall table below is therefore read as
"-0.4 optimistic": without it fp8 64.6 / fp4 57.7 / mixed 61.1 — mixed still
a predicted pass with ~0.9 us margin instead of 1.3.

Wall (the fill is paid once per CTA and lies on the wall; body and tail
unchanged; bench-equivalent cut = trace cut less 10 % for the trace build's
own cold code):

| mode | today | fill cut | boundary (R5) | predicted | target |
|---|---:|---:|---:|---:|---|
| fp8 | 67.8 | -3.2 | -0.4 | **64.2** (63.6-64.8) | <= 58 (no) |
| fp4 | 60.5 | -2.8 | -0.4 | **57.3** (56.8-57.9) | <= 36 (no) |
| mixed | 64.4 | -3.3 | -0.4 (optimistic; 0 -> 61.1) | **60.7** (60.0-61.6) | <= 62 (predicted pass, 1.3 us margin; 0.9 without the R5 credit) |
| a16 | 78.8 | -1.4 | 0 | **77.4** | parity |

This is short of the -4..-5 us the round-3 brief assumed and short of the
fp8 / fp4 targets: RT2/RT3 are L2 hits, the fill is cold instruction fetch
(~3.3 us, of which this design removes ~1.9: the generic walk, the fill
epilogue, the consumed gate, the subroutine executions) plus the start burst
(~0.4 removed of 1.54) plus a ~1 us converter-arrive -> gemm0-pass gap the
design does not reach and cannot yet attribute (finding 4).  The residual
~5 us is paid once per CTA; PDL (R7) is the lever that hides it in a graph.

## 7. Verification artifacts and accept / reject

Build (each of the four q=1 modules, `ptxas -v` on the TU with the ninja
flags via `/tmp/main_ptx/ninja_flags.py`, then `cuobjdump`):

1. No C7507; 0 bytes stack, 0 spill stores / loads; `USETMAXREG` = 2 (`0x28`,
   `0x38`); `LDL` = `STL` = 0; one `ATOMG.INC`; `HGMMA` 16; `UTMALDG` 8 (a16,
   mixed) / 0 (fp8, fp4).
2. `CALL.REL.NOINC`: 9 today; the two scan sites stay compiled (64-bit
   fallback) so the count is unchanged, but the fp8 SASS between the scan's
   first `LDG` and the `sched` STS must contain the inline u32 divide path
   (`I2F` / `MUFU.RCP` / `IMAD` sequence) reached when `P*T + P-1 < 2^32`.
3. Per-role split (`/tmp/r2p8_role_split.py` on nvdisasm line info): gemm0
   `SYNCS.PHASECHK` 8, `ARRIVE` 10 (11 - the Q token arrive), `BAR.SYNC` 1,
   HGMMA 8; converters: the two prologue `LDGSTS` groups separated by a
   `DEPBAR.LE SB0, 0x0` (the `waitGroup<0>`), the global-scale `LDG`s after
   the first group in program order, PHASECHK site count unchanged; IO warp 3's
   region contains the fast path's 8 `LDG` (page table; +8 `LDG` page_format in
   the mixed module only) before its `sched` STS; static modules' `fillTileMeta`
   region has no `page_format` `LDG`; the loader's `fillTileMeta` call sits
   inside the tile loop under a `g == nbFast` predicate.
4. `sizeof(SharedMem)` <= 113 792 (trace <= 115 072); occupancy 2 by
   registers and by shared memory (ncu `launch__occupancy_limit_*`).

Conformance: `python tests/attention/run_xqa_mixed_page_transport.py` -> 66
passed, 0 failed (60 + the two round-3 tail cases x 3 modes: (130, P = 5,
normal) — CTA 2 owns x in [5, 8), x0 = 5 is the last tile of head 1 of
request 0, so tile 1 is tile 0 of the next request (nbFast = 2 through the
next-request branch); (130, P = 5, midempty) — batch 3 with seq_lens [130,
0, 130]: the same CTA's tile 1 needs the empty-request skip loop, so nbFast =
1 and the loader's fill does tile 1; the empty request's output row is
excluded from the comparison.  The existing T < P cases exercise nbFast in
{0, 1}; (2200, P = 1) has tile 1 in the same head; (4096, P = 5) has tile 1
in the same head with partial items.)

Trace (`MIXED_KV_TRACE 1` copy of the checkout, `xqa_mixed_trace_once.py
--modes fp8 fp4 transport_a16 mixed --q-len 1 --launches 3`,
`parse_xqa_prologue.py --summary`).  "today" = trace5 median over 3 launches
(per-launch range); accept = the median over the 3 launches of the
confirmation run; reject thresholds lie outside today's launch-to-launch
spread so that one run can decide:

| quantity (fp8 unless stated) | today | accept | reject / action |
|---|---:|---|---|
| first K copy committed (14) | 5.31 (5.22-5.44) | <= 3.2 | > 3.8: the fast path is not on the chain designed; re-stamp (40, 41, 47) before any tuning |
| scan -> fast pages returned (40 - 4) | — | <= 0.9 | > 1.2: fast-path arithmetic is heavier than budgeted; inspect its SASS |
| sync released -> conv at issue(0) (41 - 37) | — | <= 0.6 | > 0.9: setmaxnreg.inc waits on a late .dec; look at the other groups' pre-sync code |
| RT4 = landed - committed (15 - 14) | 1.54 (1.46-1.70) | <= 1.30 | >= 1.45 (inside today's spread): tile ordering not effective; check the `DEPBAR` placement |
| tile 1: committed - K0 landed (43 - 15) | — | <= 0.30 | > 0.5: R2 constants are on the tile-1 path; move them after issue(1) |
| tile 1 landed (45) | — | <= firstk + 0.6 | later: gemm0 stalls on tile 1 -> F3c (issue tile 2 only after tile 1 landed) or revert R3 |
| true gap: produced(0) complete -> gemm0 passes (20 - 42) | — | report | this splits finding 4; no gate |
| arrive issued -> phase complete (42 - 16) | — | report | > 0.5: the release-arrive waits on MIO; a steady-state item |
| firstk (fill) | 8.99 (8.90-9.25) | <= 6.0 (fp4 <= 5.7, mixed <= 7.0) | > 6.8 (fp4 6.5, mixed 7.8): stop, re-attribute |
| loader chunk 0 done (13) vs converters' issue(2) | 4.83 | `kc_ready(2)` within +10 % of the steady period | later: section-6 fallback (c) (loader pre-arrive + iteration-0 fill), designed and reviewed first; NOT the bare iteration-0 fill (unsound: C8' I-fill) |
| a16 `kl_iss(2)` (per-tile trace, TILE0 = 0: TMA issue of tile 2 behind the walk) | ~4.0 | <= 4.6 | > 5.0, or a16 `g0_kwait(2) -> g0_mma(2)` above steady + 0.6: the TMA-branch walk is longer than budgeted; re-stamp (46, 13); the lever is nbFast = 3 in the scan warp (register budget first), not a fill move |
| mixed `kc_ready(2) - kc_ready(1)` (converters' issue(2) gated by `kMetaReady[0]` at ~5.3) | — | <= steady period + 0.5 us | > steady + 1.0: RT3 is on the chunk-0 chain in the mixed module; report (nbFast = 3 or a page_format-free tag path are separate designs); no tuning |
| `g0_kwait(1) - g0_xarr(0)` (tile-1 stall on gemm0) | steady | <= steady + 0.6 us | > +0.6: F3c or revert R3 |
| `kc_ready(1)`, `kc_ready(2)`, `kc_ready(16)`, `kc_ready(32)` | steady | steady-state period +-10 % | chunk completion late: check `kMetaReady` slack (R4, fallback above) |
| item boundary `g0_kwait -> g0_qwait` (CTA 1, tile 31) | 1113-1351 cyc | <= 600 cyc | |
| steady-state role periods (tiles 3-7) | fp8 1.21-1.32, fp4 1.20-1.31 | within +-3 % | moved: a per-tile path changed; find the site in the per-role SASS split |
| per-CTA histogram fill median (bench-equivalent build) | fp8 8.5 / fp4 7.4 / a16 6.6 | <= 5.5 / 5.0 / 5.5 | |
| body median, tail | unchanged | +-3 % | |
| a16 first TMA issue (loader iteration 0; stamp 13 no longer marks it — use `kl_start(0)` of the per-tile trace with TILE0 = 0) | ~3.9 | <= 2.8 | > 3.4: the a16 loader is still behind the fill; check the `g == nbFast` placement |

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
<checkout> r3fill sm90 transport_a16 fp8 fp4 mixed`; 5 x 5; report min /
median / max; today's spread is +-0.2-0.3 us, so the reject bounds are >= 1.3
us above the accept bounds):

| mode | today (5x5 median) | accept (median) | predicted | reject if |
|---|---:|---|---:|---|
| fp8 | 67.8 | <= 65.0 | 64.2 | > 66.3 (cut < 1.5 us: less than half the prediction) |
| fp4 | 60.5 | <= 58.0 | 57.3 | > 59.2 |
| mixed | 64.4 | <= 62.0 (target) | 60.7 | > 63.0 |
| a16 | 78.8 | <= 78.8 (no regression) | 77.4 | > 79.5 |
| q=4 rows | unchanged within spread (untouched SPEC_DEC path) | | | |

PDL (R7, when built): SASS `ACQBULK` in every role region and one `PREEXIT`;
`nsys` on a two-kernel stream shows the second kernel's CTAs starting before
the first ends; the single-kernel bench must be unchanged.

## 8. Do not build if

1. The fast path and the generic capture cannot share `makeTileWord` /
   `tileItemBits` (3.8) — two flag formulas are not acceptable.
2. `ptxas -v` reports a spill in IO warp 3 (at the launch budget 48, before
   `setmaxnreg`) with the fast path, **and** the converter-side variant (3.9)
   also spills: the register story then needs [15]'s layout first.
3. Any reader of records 0..nbFast-1 can run before the `__syncthreads` (e.g.
   if a role's prologue is moved ahead of it) — C10 for g < nbFast depends on
   the sync.
4. `sizeof(SharedMem)` would exceed 115 712 B for the trace build.
5. The confirmation trace shows any steady-state period moved by more than
   3 % — the premise is "prologue only".
6. The confirmation trace shows the first copy committed later than 3.8 us
   or firstk above 6.8 us (fp8): the chain is not the one designed here;
   re-stamp instead of tuning.
7. Tile 1 stalls gemm0 by more than 0.6 us after R3 and the F3c fallback
   (issue tile 2 only after tile 1 landed) does not remove it: revert R3
   (tile ordering) rather than adding depth.
8. R5's count change leaks into the non-persistent build (`qBar` init is
   shared code at :1467): the non-persistent path must keep 160/160 and its
   `arrive_and_wait` (:1585) — a build where `nbQBuf == 1` must be checked.
9. The conformance matrix cannot express the nbFast = 1 fallback (empty next
   request with G >= 2): without it C14's fallback branch is untested.
10. Production passes `bmm1_scale` / `bmm2_scale` as tensors: the
    `qScalePtr` derefs at :1281-1282 become a dependent round trip for every
    thread before the scan; that is a separate item and must not be folded
    into this change blind.
11. The loader's chunk-0 fill at `g == nbFast` makes `kc_ready(2)` late by
    more than 10 % of the steady period in the static modules and the sound
    fallback (section 6 (c)) does not fix it: the chunk-0 protocol,
    not its position, is then the issue.

## 9. As written (kernel state after this change; not built or run in this phase)

**Data flow.**

    seqLenList[B] --scan (IO warp 3)--> smem.sched {x0, x1, T, req0, head0, tile0, Lseq0, seqLen0, seqLen1, nbFast}
                  --fast path (IO warp 3, before the sync)--> meta[K][0][g], meta[V][0][g] for g < nbFast
                                                             (pages: page table RT2; tag: static format or page_format)
    sched --> ItemCursor (registers) in IO warps 0 (K), 1 (V), 2 (Q), 3 (merge)
    K loader: at iteration g == nbFast: cursor + page table [+ page_format, mixed module] --fillTileMeta(nbSkip = nbFast)-->
              meta[K][0][nbFast..15] --kMetaReady[0].arrive--> ; refills at g = 12, 28, ... as before (nbSkip = 0)
    K converters: meta[K][t % 32] {page, tag, head} --cp.async--> packed rows / scales --expand--> k stage --kBar.produced.arrive-->
    gemm0: kBar.produced.wait -> meta[K][g].tile ; if first: qBar[j&1].produced.wait_parity(j >> 1) ; q[j&1] ; X(g%2)
    Q warp: q[(req, head)] --regs--> smem.q[j&1]  (qBar[j&1]: produced count 32, consumed 160)
    gemm1 / merge: unchanged

**Control flow per role** (g = CTA-local tile counter, G = x1 - x0, t = tile
being issued):

    scan (IO 3)   pass 1 ; x0/x1 (u32 when P*T + P-1 < 2^32, else u64) ; pass 2 ;
                  nbFast = min(2, G) (1 if tile 1 needs the empty-request skip) ; lanes < 4*nbFast: page LDG ;
                  fmt ; gather ; makeTileWord ; STS both operands ; lane 0 publishes sched ; __syncthreads
    gemm0         pre-arrive qBar[0..1].consumed, kBar[*].consumed
                  for g: kBar[g%3].produced.arrive_and_wait ; tile = LDS meta[K][g].tile
                         if first: runningColMax = -inf ; qBar[j&1].produced.wait_parity(toParity<2>(j))
                         QK HGMMA ; ... ; if last: qBar[j&1].consumed.arrive ; j++      (rest unchanged)
    K loader (IO 0)   cursor ; for g: if g == nbFast: fillTileMeta(chunk 0, nbSkip = nbFast) ; kMetaReady[0].arrive
                             [a16/mixed: LDS meta[K][g]] ; kBar[g%3].consumed.arrive_and_wait ; [TMA boxes]
                             if (g+4) % 16 == 0 && g+4 < G: fillTileMeta(chunk, nbSkip = 0) ; kMetaReady[..].arrive
    V loader (IO 1)   mirror
    converters    nbFast = LDS sched.nbFast
                  issue(t): if t >= nbFast && (t == nbFast || t % 16 == 0): kMetaReady[(t/16)%2].wait_parity(toParity<1>(t/32))
                            if t >= 3: kBar[t%3].consumed.wait_parity(toParity<3>(t))
                            issueCompressedPageCopies(meta[op][t % 32])
                  issue(0) ; commit ; global scales LDG ; ExpandLane / ExpandScales ; waitGroup<0> ; issue(1) ; commit
                  for g: waitGroup<1> ; __syncwarp ; expand ; fence ; produced.arrive ; issue(g+2) ; commit   (unchanged)

Code shape rules applied: the fast path selects its piece's scalars per lane
(`g == 0 ? a : b` on registers, never on K/V struct references); records are
shared-window `st.shared` at `metaBase[op] + g * 32 + imm`; the two prologue
issues are written out (no loop) so the R3 wait sits between them; the
`kAhead == 2` assumption is a `static_assert`; every existing
`MIXED_KV_TRACE` stamp keeps its slot and meaning (13 is still the K loader's
`kMetaReady[0].arrive`, now inside the loop; 14/17 the first commit; 7/26/8/9
the chunk-0 fill phases), and eight stamps are added (40-47, printed on a
third line `TRACE ctaprolog3`, parsed by `parse_xqa_prologue.py`).  The
trace-only `test_wait` poll behind stamps 42 / 44 delays converter warp 0's
lane 0 by at most the arrive skew of the other three warps and exists only
under `MIXED_KV_TRACE`.  It diverges lane 0 from the warp's other 31 lanes
across the following `issueKCopies(idxIter + 2)` (cp.async issue is per lane,
no `__syncwarp` inside), and the warp reconverges at the loop's `__syncwarp`;
the phase completion polled does not depend on that warp's other lanes (their
arrives are already issued), so no deadlock — trace-only, recorded so the
divergence is not mistaken for a bug in a SASS read.  `mixedKvTraceUse`
orders a stamp after a load only in practice (1.1 (a) caveat).

Files: `csrc/xqa/mha_sm90.cu` (helpers `tileItemBits` / `makeTileWord`;
`PersistentSched::nbFast`; `qBar` init; gemm0 Q parity wait; loader
`fillChunk0` at `g == nbFast`; converter prologues; scan R6 + fast path;
`fillTileMeta(nbSkip)` + R4; trace words 40-47),
`benchmarks/parse_xqa_prologue.py` (words 40-47, `--summary`),
`tests/attention/test_xqa_mixed_page_transport.py` +
`run_xqa_mixed_page_transport.py` (two tail cases x 3 modes: 66 cases).

Verification for this change is section 7.  Not run in this phase (review
by reading first): `ptxas -v` (no C7507, 0 spill, IO warp 3 at 48 with the
fast path), `USETMAXREG` = 2, `LDL`/`STL` = 0, the 66-case matrix, the locked
bench, the same-layout trace (`--summary`) and the per-CTA histogram.

## 10. Confirmation results (2026-09-04, nkcut2 H200, wt/r3fill @ 6dbc6ae3)

Full tables (section-7 accept table filled, per-mode attribution, hang and
spill facts of the trace build) are in `mixed_kv_page_transport_backends.md`,
section "Round 3, lever "fill / prologue cut" (R1-R6): confirmation and
attribution".  Summary against sections 6-8:

- **Build (7.1-7.4)**: production modules pass every gate (no C7507, 0
  stack / spill, `USETMAXREG` 2, `LDL` = `STL` = 0).  Conformance 66 / 66.
- **Timing (7, locked 5 x 5 medians)**: a16 78.8 -> 77.7 (pass, -1.1), fp8
  67.8 -> 67.3 (reject: <= 65.0), fp4 60.5 -> 60.4 (reject: <= 58.0), mixed
  64.4 -> **66.7 (+2.3, reject)**.
- **Trace build**: hangs in 5 of ~14 compressed-mode launches (host spins in
  the synchronize; 1-14 CTAs of a launch never finish; never seen in [8]'s
  trace build or in the production run's > 300 launches; 0 in 8 a16 / mixed
  launches) — **unattributed**; and spills (`STACK:320`, `LDL` 34 / `STL`
  103, mostly printf frames, vs 0 in production), so its body / period
  changes (fp8 body 58.9 -> 66.5) are not evidence; section-7's "periods
  +-3 %" and "body +-3 %" rows are undecidable from this run.  Complete
  launches used: fp8 4, fp4 2, a16 2, mixed 2.
- **Fill chain (7)**: stamp 14 as designed (fp8 5.31 -> 3.07, fp4 4.74 ->
  2.99), 40-4 0.67, 41-37 0.29, 43-15 <= 0 — R1 / R2 / R6 behave as
  written.  firstk fp8 9.0 -> 7.3 (design 5.4; reject > 6.8), fp4 8.2 -> 6.4
  (design 5.1), mixed 10.05 -> 8.11, a16 9.71 -> 6.82.  Shortfall: RT4 did
  **not** shrink with the tile-0-only burst (2.25 / 1.98 vs 1.54 / 1.70
  today; reject row), the arrive-perform latency stayed 0.6-0.7 us (the
  "gap" of finding 4 is now attributed: `produced(0)` complete -> gemm0
  passes is 0.01-0.10 us, arrive issued -> complete 0.58-0.74), and on fp4
  the Q warp now publishes after gemm0's K wait (Q wait 0.54).
- **The decisive failure — chunk 0 (7 row "13 vs issue(2)", 8.11)**: with the
  fill at `g == nbFast`, the loader's two preceding iterations (one
  pre-arrived `arrive_and_wait` each) take 2.5 + 1.7 us behind the
  converters' tile-0/1 LDGSTS burst that R1 moved ahead of the fill (MIO
  pipe), so `fillChunk0` is entered at 7.5 / 7.2 / 4.6 / 8.4 us (fp8 / fp4 /
  a16 / mixed) and `kMetaReady[0]` arrives at **9.4 / 9.1 / 8.6 / 10.9** (today
  4.8 / 4.2 / 3.8 / 4.8).  The converters' `issue(2)` in loop iteration 0
  waits for it, tile 1's expansion starts ~4 us late (`kc_ready(1) -
  kc_ready(0)` fp8 3.4-4.7, fp4 3.6-3.8, mixed 6.7-6.9 vs 0.9-2.6 today) and
  gemm0 stalls on tile 1 by fp8 1.9-3.9 (median 2.8), fp4 2.7-2.9, mixed
  3.1-4.1 us (today 0.1-1.0) — more than the fill cut in every compressed
  mode.  Items 8.6, 8.7 (cause is the gate, not R3) and 8.11 are triggered.
- **Mixed +2.3**: this gate (kMetaReady at 10.9 behind the two TMA iterations
  + walk + RT3) -> tile-1 stall 3.1-4.1 + tile-2 stall ~1 against firstk
  -1.9.  Not a period change (undecidable, and not needed to explain the
  wall).
- **fp8 / fp4 missing gain**: the cut lands on both dispatch sets equally
  (fill slow / fast 7.3 / 7.3 vs 9.1 / 8.7; fp4 6.4 / 6.4 vs 8.1 / 7.9), so
  it *is* on the slow member's wall and is cancelled by the tile-1 stall.
- **a16 pass**: from tile *ordering* (TMA tiles 0-1 before the walk, tile 2
  after: `kl_iss(2)` ~8.9, reject row), not from an earlier first issue
  (`kl_start(0)` ~3.8-4.2 vs ~3.9; reject row); a16 fill now follows the
  dispatch slot (6.5 slow / 8.9 fast).  Needs neither fast records nor C8'.

**Decision: keep nothing from R1-R6 now.**  The trace-build hang must be
attributed before any protocol change of this lever (C8', `qBar` counts) is
merged, a16 included.  Fallback (c) alone is not sufficient (the fill would
still run inside the burst; ~5-6.5 vs `issue(2)`'s want at ~6.0; and it
gives back a16's ordering).  To design, not build: **(c) + `nbFast = 3`**
(no `kMetaReady` gate before tile 3; scan-warp register budget first) for the
compressed modules, and the loader-only ordering "TMA tiles 0-1, walk, tile
2" for the a16 TMA branch; keep R4 / R6 as pure carry-along items; the Q
warp's placement behind the LDGSTS burst (fp4) and the unchanged RT4 (the
start-burst model is ordering, not bytes) are recorded as separate items.

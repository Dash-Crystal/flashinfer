# Mixed KV page transport: speed targets and the gate

The KV pages are quantized **for speed**, the way SageAttention quantizes Q/K
and P/V for speed rather than for footprint: attention over a long context is
dominated by streaming the activation sequence, so a page that is 2x or 4x
smaller must make the attention operation proportionally faster. A compressed
page that does not speed up attention is pointless (the analogical dual of
SageAttention here: the router quantizes stored pages by outlier statistics
once, and every later attention call streams fewer bytes).

**Gate.** No divergence or acceptance measurement starts until at least 50 % of
the analytic speedup has materialized on each host that serves the shape.

Analytic speedup vs A16 = byte ratio: FP8 1.87x, FP4 3.56x, the interleaved
mixed stream used by the benchmarks 1.66x (172 MB vs 285 MB).

| host (B=17, S=4096, 8 KV heads, GQA 4, D=128) | A16 | FP8 target | FP4 target | mixed target | measured now |
|---|---|---|---|---|---|
| H200 sm90 XQA decode, q=1 | 83 us (transport_a16 78.8 after [8]) | <= 58 | <= 36 | <= 62 | FP8 91, FP4 96, mixed 114; after wt/A+wt/B + setmaxnreg fix (fe2e9a33): 76.9 / 70.7 / 79.5; **after [8] persistent scheduling (039ba5c7): FP8 67.8, FP4 60.5, mixed 64.4** (fp8/mixed within 10 / 2.4 us of target; the wall is now set by the slow member of each co-resident CTA pair, 46.5 vs 57.1 us fp8 body, + 8.5 us fill) |
| H200 sm90 XQA, q=4 (SPEC_DEC, runs mha.cu) | 135 | <= 94 | <= 59 | <= 101 | FP8 231, FP4 277, mixed 437 (co-tenant-corrected; after [29]: 198/240/420; after [40]: 198 / 236 / 216; after [41][42] (2 CTAs/SM, M tile 16, 680 CTAs): A16 99.7, FP8 124.3, FP4 144.3, mixed 137.8; **after [43] (128 B K parts, rolled tile loops, one copy body): A16 86.1, FP8 114.0, FP4 115.9, mixed 116.1** — A16 passes; FP8 / FP4 / mixed at 1.21x / 1.96x / 1.15x of target, i-fetch stall gone (no_instruction 0.15-0.5), issue-active 50-58 %; after step 6 [44] (placement decode, half-row owner cut, hoisted copy constants): A16 86.4, FP8 116.7 (+2.7, regression kept: reject rule not triggered), FP4 103.6, mixed 110.3 — instruction counts at the design numbers (0.84x / 0.72x / 0.82x) but time tracks stall structure (short/long scoreboard rose), fp8 instruction-count lever exhausted at this pipeline depth) |
| RTX 5090 sm120 XQA decode, q=1 | 174 (A16) | <= 125 | <= 79 | <= 135 | **after [29][26][27]: FP8 100.5, FP4 59.5, mixed 113.5** (were 139/84/146; 2.93x / 1.73x / 1.53x vs A16 against analytic 3.56 / 1.87 / 1.66) |
| RTX 5090 sm120 XQA, q=4 | 184 (179) | <= 128 | <= 81 | <= 138 | after [29]: FP8 125.0, FP4 81.9, mixed 132.3; **after [41] (M tile 16): A16 176.1, FP8 115.1, FP4 65.7, mixed 119.0 (all three pass)**; [43] is sm90-only (sm120 SASS byte-identical): interleaved base vs [43], VLLM co-tenant resident, 176.4 / 116.3 / 65.8 / 123.4 vs 176.2 / 116.7 / 65.8 / 123.2 |
| H200 FA3 prefill, q>=64 | 300 (stock) | parity | parity | parity | **after [21][22]: A16 282-287 (0.94x stock, passes)**; after [23]: FP8 474-483, FP4 507-517, mixed 880-907 (were 737-748 / 944-964 / 1760; producer issue-bound at ~0.22 IPC per producer warp, see dataflow A7); **after [24] (F24a E4M3 decode floor + F24b second producer warp group + F24c dynamic page masks, wt/F24 @ 35706f8a): FP8 460 / 476, FP4 496 / 512, mixed 718 / 728** (q=1 / q=64 medians; target <= 330 not met: the producer still paces - trace acq 0.1 us, 4432 producer warp-instr per pair = +30 % vs [23] at issue-active 52.9 %; F24a alone 495 / 512, slower than [23]; a16 282 / 288 unchanged, module byte-identical; 88/88 bit-exact); **after [25] (F25a-e: one 12-warp producer at 136 / 184 registers, one fold vote per operand with branch-free bodies, per-item copy bases, E2M1 placement decode, predicated dynamic copies; wt/F25 @ dd583e36): FP8 403 / 415, FP4 422 / 430, mixed 650 / 664** (q=1 / q=64 medians; target <= 330 not met: consumer K-wait 10 / 18 / 37 %, producer 716 warp-instr per pair at IPC 0.23 vs the design's 655 at 0.27; a16 282.8 / 289.5, stock 300.9 / 310.9; 104/104 bit-exact; stock kernel byte-identical, a16 module identical up to a uniform-register permutation) |

Byte rooflines, corrected by the P0.1 host probe (measured achievable streaming
read at each footprint on nkcut2: 4.23 TB/s at 285 MB, 4.5 sustained): A16 67.5,
FP8 38.6, FP4 23.0, mixed 43.0 us (the 4.8 TB/s paper values 59/32/17/36 are not
reachable).  "A16 83 us" is transport_a16 through mha_sm90.cu; the stock mha.cu
A16 baseline at q=1 is 108-110 us.  At q=4 every mode runs mha.cu SPEC_DEC; the
recorded 935 us for mixed q=4 was co-tenant time slicing (bursts > ~2 ms are
inflated 1.8-2.1x) - the kernel takes 441-452 us (fp4 277-285).  The sm90 XQA
consumer chain floor is 1.00 us/tile with converters skipped (0.84 at 1 CTA/SM),
i.e. latency-bound on round trips, not issue-bound.

Measurement rule (H200): keep repeats x kernel time < 1.5 ms per event pair and
report min/median/max; --repeats 1 carries a 5-15 us launch gap.

Method: every change is a lever with an analytic model that predicts its gain
from the measured record (docs/mixed_kv_page_transport_backends.md) and a
verification artifact that reads the mechanism (SASS counts, per-tile trace,
ncu launch/issue statistics); the stopwatch confirms, it does not steer.

## RTX 5090 (sm120): gate PASSED — all six rows (2026-09-04, main @ 67a6b4aa, 72/72 bit-exact)

Kernel: `csrc/xqa/mha.cu` (compute warps load + expand pages in place), levers
[29] C2 fix, [26] 128 B K parts, [27] GRP_LOAD_V, [40] page-outer format
dispatch, [41] M tile 16 / 2 CTAs per SM, [42] occupancy-aware
`nbSubSeqPerSeq`; sm90-only changes ([43], [44]) verified SASS byte-identical
on sm120.  Locked bench on ws-1, standard script (5 x 5), B=17, S=4096, 8 KV
heads, GQA 4, D=128; min / median / max us and the fraction of the analytic
byte-ratio speedup realised (gate = 50 %):

| q | mode | time (us) | speedup vs A16 | analytic | realised | target | gate |
|---|---|---|---|---|---|---|---|
| 1 | A16 (transport) | 172.7 / **173.3** / 174.1 | 1.00x | — | — | — | reference |
| 1 | FP8 | 100.2 / **100.7** / 100.8 | 1.72x | 1.87x | **83 %** | <= 125 | PASS |
| 1 | FP4 | 59.8 / **59.9** / 60.4 | 2.89x | 3.56x | **74 %** | <= 79 | PASS |
| 1 | mixed | 112.8 / **113.6** / 114.0 | 1.53x | 1.66x | **80 %** | <= 135 | PASS |
| 4 | A16 (transport) | 175.6 / **176.0** / 177.0 | 1.00x | — | — | — | reference |
| 4 | FP8 | 114.6 / **116.3** / 116.6 | 1.51x | 1.87x | **59 %** | <= 128 | PASS |
| 4 | FP4 | 65.5 / **65.9** / 66.2 | 2.67x | 3.56x | **65 %** | <= 81 | PASS |
| 4 | mixed | 119.2 / **119.8** / 120.1 | 1.47x | 1.66x | **71 %** | <= 138 | PASS |

Realised = (measured speedup - 1) / (analytic speedup - 1).  Correctness: the
72-case matrix (32 register-expansion, 2 native block-FP8, 36 tail /
value-range incl. E4M3 subnormals, +-448, maximal and sub-2^-117 block/global
scales at q=1 and q=4, 2 independent stock-decode references) is bit-exact
against the A16-expansion reference on every case.  Per the project rule, the
sm120 host is cleared for behavioural (divergence / acceptance) measurement.

## Offloaded KV cache (sm120 and any host reading pages over a link)

When pages stream from a host tier (PCIe / NVLink) rather than local device
memory, the byte ratio is the wall: the link is 10-100x slower than DRAM, so
FP4/FP8 pages decide between a pipeline stall and none, and the on-device
request-pattern effects (sector utilization, L2 neighbour hits) are second order.
Requirement carried by every host, independent of whether it is currently
request-pattern-bound: the expansion path must consume packed pages at whatever
rate they land without adding a stall of its own - lean, issue-light,
shared-window `LDS`/`STS` with immediate offsets and independent block bodies,
verified in SASS (no generic `LD.E`/`ST.E` for the expansion, no `LDL`/`STL`).
On sm120 the compute warps are the expanders, so a lean expansion returns issue
slots to the MMA directly; Track W's [29] adopts the same discipline as FA3's [23].

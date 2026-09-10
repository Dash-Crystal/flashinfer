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
| H200 sm90 XQA, q=4 (SPEC_DEC, runs mha.cu) | 135 | <= 94 | <= 59 | <= 101 | FP8 231, FP4 277, mixed 437 (co-tenant-corrected; after [29]: 198/240/420; after [40]: 198 / 236 / 216; after [41][42] (2 CTAs/SM, M tile 16, 680 CTAs): A16 99.7, FP8 124.3, FP4 144.3, mixed 137.8; **after [43] (128 B K parts, rolled tile loops, one copy body): A16 86.1, FP8 114.0, FP4 115.9, mixed 116.1** — A16 passes; FP8 / FP4 / mixed at 1.21x / 1.96x / 1.15x of target, i-fetch stall gone (no_instruction 0.15-0.5), issue-active 50-58 %; after step 6 [44] (placement decode, half-row owner cut, hoisted copy constants): A16 86.4, FP8 116.7 (+2.7, regression kept: reject rule not triggered), FP4 103.6, mixed 110.3 — instruction counts at the design numbers (0.84x / 0.72x / 0.82x) but time tracks stall structure (short/long scoreboard rose), fp8 instruction-count lever exhausted at this pipeline depth); **after step 7 [45] (`d453cbcb`: tags / flags / formats in registers, rowSum constant mask, ungated page-list load, one fold vote per call in the static modules; [45b] off by the WAR gate, [45d] reverted, dyn per-span vote): A16 85.3, FP8 110.3, FP4 95.2, mixed 102.5** vs the same-session pristine base 86.7 / 116.5 / 103.7 / 110.3 (0.98x / 0.95x / 0.92x / 0.93x; the base runs 2-3 us above the record this session) — instruction counts 0.90-0.93x, stall-per-instruction did not fall (long_scoreboard rose: the page-index round trip is now exposed on the BAD SEL, 4-6 %); FP8 / FP4 / mixed targets stay open (1.17x / 1.61x / 1.01x) |
| RTX 5090 sm120 XQA decode, q=1 | 174 (A16) | <= 125 | <= 79 | <= 135 | **after [29][26][27]: FP8 100.5, FP4 59.5, mixed 113.5** (were 139/84/146; 2.93x / 1.73x / 1.53x vs A16 against analytic 3.56 / 1.87 / 1.66) |
| RTX 5090 sm120 XQA, q=4 | 184 (179) | <= 128 | <= 81 | <= 138 | after [29]: FP8 125.0, FP4 81.9, mixed 132.3; **after [41] (M tile 16): A16 176.1, FP8 115.1, FP4 65.7, mixed 119.0 (all three pass)**; [43] is sm90-only (sm120 SASS byte-identical): interleaved base vs [43], VLLM co-tenant resident, 176.4 / 116.3 / 65.8 / 123.4 vs 176.2 / 116.7 / 65.8 / 123.2 |
| H200 FA3 prefill, q>=64 | 300 (stock) | parity | parity | parity | **after [21][22]: A16 282-287 (0.94x stock, passes)**; after [23]: FP8 474-483, FP4 507-517, mixed 880-907 (were 737-748 / 944-964 / 1760; producer issue-bound at ~0.22 IPC per producer warp, see dataflow A7); **after [24] (F24a E4M3 decode floor + F24b second producer warp group + F24c dynamic page masks, wt/F24 @ 35706f8a): FP8 460 / 476, FP4 496 / 512, mixed 718 / 728** (q=1 / q=64 medians; target <= 330 not met: the producer still paces - trace acq 0.1 us, 4432 producer warp-instr per pair = +30 % vs [23] at issue-active 52.9 %; F24a alone 495 / 512, slower than [23]; a16 282 / 288 unchanged, module byte-identical; 88/88 bit-exact); **after [25] (F25a-e: one 12-warp producer at 136 / 184 registers, one fold vote per operand with branch-free bodies, per-item copy bases, E2M1 placement decode, predicated dynamic copies; wt/F25 @ dd583e36): FP8 403 / 415, FP4 422 / 430, mixed 650 / 664** (q=1 / q=64 medians; target <= 330 not met: consumer K-wait 10 / 18 / 37 %, producer 716 warp-instr per pair at IPC 0.23 vs the design's 655 at 0.27; a16 282.8 / 289.5, stock 300.9 / 310.9; 104/104 bit-exact; stock kernel byte-identical, a16 module identical up to a uniform-register permutation); **after F26a+b (merged): FP8 366 / 389, FP4 393 / 400, mixed 635 / 645** (F26c dynamic per-slot bodies rejected: mixed 1111) |

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
The streaming contract keeps payloads packed through shared memory and converts
them directly into matrix operand registers. A tile-wide A16 shared expansion
adds traffic and a producer/consumer boundary that the implementation must remove.
The historical expansion measurements above remain receipts, not a requirement
to retain that staging scheme.

SM12x now lowers mixed pages through `xqa/mixed_kv_fragments.cuh`. K uses native
packed matrix loads and reuses each converted block scale. The existing Q
permutation puts coefficients into the packed K fragment order once per CTA;
A16 pages gather four contiguous coefficients into the same order. V uses native transposed
8-bit matrix loads, including hardware nibble unpacking for FP4. Its asynchronous
copy permutes token rows within each 16-token tile into MMA pair order; A16 loads
use the same row mapping, while scales retain logical token indexing. No expanded
KV tile is written back to shared memory. The existing global double buffering,
mask closure, codecs, and page-format decisions are retained.

The former dormant packed consumer duplicated converters inside unrolled
block/page loops and assumed a 16-token page and one V head slice. The replacement
dispatches a page format outside K's rolled reduction loop, retains static
accumulator indices, and indexes V's head slice separately from its token tile.
It covers D256/page16 and D512/page32, including grouped V copies with their
per-warp scale-row gaps. Full-model graph execution, task outputs, latency, and
compiled register/local-memory use are the validation surface. The first
register-consumer full-model trace retained 673 kernels per decode execution,
with no local-memory spills (D256: 148 registers; D512: 217). Its early
short-context graph span remained about 33 ms. The native K matrix-load followup
ran in V63: D256/D512 use 149/217 registers and zero local bytes. Its captured
padded152 decode graph averages 33.539 ms across 42 executions. This is a
different work coordinate from V62, not a matched speedup measurement. Other
architecture paths still require equivalent removal of shared expansion.

The completed-page producer also distributes the original 16-lane codec across
32 warps per page, reducing the serialized codec iterations eightfold for
Gemma's 32768-value pages. Routing reuses one coefficient read for the adjacent
token moments and block signature, and reduces the four moments together.
Thresholds, nine-candidate scale selection, and encoded formats are unchanged.
The rectangular producer uses the same flattened K/V signature indexing as the
arena, including head widths divisible by 16 but not 32. V64's delayed
full-model padded96 decode trace averages 26.930 ms across 61 executions.
Summed sealing time is 3.982 ms per forward; its intervals without another
kernel name active in the same execution total 0.018 ms. Sealing uses 62
registers with zero local bytes. V61's earlier padded144 trace recorded
22.121 ms summed sealing and 2.371 ms exclusive intervals. Those different
work coordinates demonstrate the overlap mechanism, not a causal speedup.

V64 retained 86,024 pages per rank at 9.001/8.951 bits per value including
scales and A16 tails, with zero allocation failures and duplicate addresses.
The canonical persistent SVG/video mixture reports 185 SVG turns, 62 invalid
program turns, zero public exchange failures, MSE 0.10956 and SSIM 0.61863.
All 111 decoder and 25 encoder graphs captured. Maximum completed SVG prompt
length is 3,706; the direct >4K end-to-end compression objective remains open.
The vLLM TP2 execution review records the retained trace and moment artifacts.

The V66 SM12x D256 decode lowering divides the eight-warp CTA into two
independent four-warp CTAs. Each keeps two K and V pipeline buffers and owns
two existing V head slices per PV warp. A 64-byte K part makes the main shared
arrays 16 KiB K, 16 KiB V, 4 KiB Q and 4 KiB X, leaving space for scales and
barriers while allowing two CTAs in the SM's shared-memory budget. Launch
geometry and split-KV planning read the compiled geometry exports. The layer
shape selects this lowering at compilation; D512 and continuation retain
their existing plans. Smaller K transfers increase the number of copy rounds,
so the overlap benefit and register footprint require full-model measurement.

V66's full-model trace measures 43,392 shared bytes, 208 registers and zero
local bytes for D256. Its padded80 decode span is 26.069 ms; D256 attention
accounts for 8.075 ms and D512 for 3.140 ms. The late padded88 cohort takes
31.425 ms at 85.57 useful rows and 216,077 causal pairs, compared with V64's
30.706 ms at 86.40 rows and 224,480 pairs. Independent CTA progress has not
established a latency gain. Two four-warp CTAs still supply eight resident warps,
the same as the previous eight-warp CTA. The census records 90,848 pages per rank,
9.070 bits/value on rank zero, and no allocation failures or duplicate addresses.

The following implementation separates K's logical copy extent from its
physical shared row. SM12x uses 128 logical A16 bytes per part with 64 shared
bytes for packed FP8/FP4. A16 pages load coalesced 64-bit MMA register fragments
directly from their authoritative page address. The K ring retains that address,
including its format, instead of another format-only record. Logical bounds and
masked tails remain explicit. D256 retains its smaller CTA allocation without
doubling copy rounds; D512 and continuation also use wider logical parts.
V's existing packed matrix loads and double buffering are unchanged. Full-model
compilation, output metrics and latency measurements remain pending for this
layout; the V66 results do not measure it.

V68 full-model warmup exposed a device compilation error from binding `mha::min`
to a host constexpr geometry member. The value-based comparison correction is
running in V69. Its loaded decode binaries use 208/217 registers for D256/D512
and zero local bytes; the query-span binaries use 168/223 registers without
local bytes. Binary receipts are copied while the server runs, because the
shared JIT directory is rebuilt when the previous serving source is restored.

The following D256 ownership change preserves the 128-token CTA extent and
128-byte logical K parts, distributing each pipeline over four 32-token warps
instead of two 64-token warps. The main shared arrays remain 40 KiB. Two such
CTAs supply sixteen resident warps if the compiler meets the 128-register budget.
Each PV warp owns 64 output coefficients rather than 128, and each QK warp
owns half as many token accumulators and page references. This increases the
available warp concurrency rather than only the CTA count. Page-relative K
offsets also cover warp tiles smaller than a page; pages wider than the smaller
CTA retain the wider CTA geometry. Compilation, spill placement and full-model
performance still need measurement; V69 does not include this ownership change.
V70's loaded D256 binary uses 121 registers and zero local bytes, fitting two
eight-warp CTAs within the register budget. The full model has captured its graphs
and is serving the canonical workload; its latency comparison is still running.

The next register pipeline separates K fragment fetch from conversion and MMA.
Two packed fragments alternate so the next matrix load precedes the current
fragment's conversion and multiply. The existing four-byte scale row is loaded
once per token into a register, replacing one byte load for every reduction
block. Conversion still rounds the block scale times the global scale to A16
before multiplying the payload. The final iteration consumes the last fragment
without an out-of-range prefetch. No shared expansion or additional barrier is
introduced. Full-model compilation, resource placement and performance are
pending for this followup; the V70 binary does not contain it.

V uses the same fragment pipeline with compile-time unrolling, preserving static
output-accumulator indices. Four scale words cover its two adjacent token pairs
across the output blocks; each byte is extracted from the existing row layout,
including group-buffer dump rows. K and V retain their distinct matrix-load and
scale-pair arrangements while sharing the pipeline schedule. The staged K-only
V71 runtime has not executed; V72 validates the composed K/V change.

V72's loaded D256/D512 decode binaries use 118/203 registers; query spans use
168/212. All four report zero local bytes. Its completed canonical workload's
shared-cost fit predicts 9–18% lower 8K/1K latency than A16 across B48–96.
The comparison and its limitations are recorded in vLLM's execution review;
this is not a matched replay measurement or an established 20% latency gain.
The emitted K loop fetches the next packed fragment before the current MMA,
but some current conversion precedes that fetch despite the source ordering.

The capacity snapshot's final scalar now counts reusable bytes across all size
classes and empty slabs. Empty-slab count and free bytes share one block reduction.
vLLM uses A16-slot capacity for immediate writes and total reusable bytes for its
compressed admission forecast, reserving outstanding writes and writable tails
at A16. The paired snapshot ABI is six scalars plus two per geometry; the last
scalar follows the existing per-geometry arrays. V72 predates this accounting
change, which requires its own native scheduler build and serving validation.

The next consumer change pairs adjacent E4M3 block-scale conversions. It shares
the exact E4M3-to-FP16-to-FP32 embedding with the existing scale helper, multiplies
each scale by its FP32 global scale, then packs the two rounded A16 values in one
conversion. K broadcasts the two result halves; V consumes the pair directly.
The ordinary paired scaling helper uses this implementation too. V72 SASS had
two separate E4M3 conversions and two scalar BF16 conversions at this boundary.
The paired source needs full-model compilation, emitted-instruction inspection
and latency measurement; no performance gain is inferred from the source alone.

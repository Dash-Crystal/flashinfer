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
| H200 sm90 XQA decode, q=1 | 83 us | <= 58 | <= 36 | <= 62 | FP8 91, FP4 96, mixed 114 |
| H200 sm90 XQA, q=4 (SPEC_DEC, runs mha.cu) | 135 | <= 94 | <= 59 | <= 101 | FP8 231, FP4 277, mixed 437 (co-tenant-corrected; after [29]: 198/240/420; after [40]: 198 / 236 / 216; **after [41][42] (2 CTAs/SM, M tile 16, 680 CTAs): A16 99.7, FP8 124.3, FP4 144.3, mixed 137.8** — A16 passes, mixed 0.95x fp4; FP8 / FP4 / mixed at 1.32x / 2.45x / 1.36x of target with occupancy exhausted) |
| RTX 5090 sm120 XQA decode, q=1 | 174 (A16) | <= 125 | <= 79 | <= 135 | **after [29][26][27]: FP8 100.5, FP4 59.5, mixed 113.5** (were 139/84/146; 2.93x / 1.73x / 1.53x vs A16 against analytic 3.56 / 1.87 / 1.66) |
| RTX 5090 sm120 XQA, q=4 | 184 (179) | <= 128 | <= 81 | <= 138 | after [29]: FP8 125.0, FP4 81.9, mixed 132.3; **after [41] (M tile 16): A16 176.1, FP8 115.1, FP4 65.7, mixed 119.0 (all three pass)** |
| H200 FA3 prefill, q>=64 | 300 (stock) | parity | parity | parity | **after [21][22]: A16 282-287 (0.94x stock, passes)**, FP8 737-748, FP4 944-964, mixed 1760 (awaiting [23]) |

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

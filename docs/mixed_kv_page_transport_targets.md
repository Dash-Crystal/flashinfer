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
| H200 sm90 XQA, q=4 (SPEC_DEC) | 135 | <= 94 | <= 59 | <= 101 | FP8 229, FP4 275, mixed 935 |
| RTX 5090 sm120 XQA decode, q=1 | 180 | <= 125 | <= 79 | <= 135 | FP8 139, FP4 84 (passes), mixed 146 |
| RTX 5090 sm120 XQA, q=4 | 184 | <= 128 | <= 81 | <= 138 | FP8 155, FP4 101, mixed 157 |
| H200 FA3 prefill, q>=64 | 300 | parity | parity | parity | A16 528, compressed ~1500-1800 |

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

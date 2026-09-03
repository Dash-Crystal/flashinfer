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

Byte rooflines (H200 4.8 TB/s): A16 59, FP8 32, FP4 17, mixed 36 us. The sm90
XQA consumer chain floor measured ~62 us (converters skipped), so the FP4 target
requires the consumer chain itself to get faster, not only the transport.

Method: every change is a lever with an analytic model that predicts its gain
from the measured record (docs/mixed_kv_page_transport_backends.md) and a
verification artifact that reads the mechanism (SASS counts, per-tile trace,
ncu launch/issue statistics); the stopwatch confirms, it does not steer.

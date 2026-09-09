# GEMM tiles with live row operands

`mm_masked_tiles(x, weight, out, is_padding, row_offset=0)` reuses the
caller's row visibility tensor. A CTA checks its 128-row tile and returns
before operand loading or matrix multiplication if every row is padding.
Partially visible tiles execute ordinary GEMM arithmetic. The operation
accepts holes in the mask and strided row-major A/output and column-major B.
The leading dimensions and K/N extents require eight-value alignment.

Wholly masked tiles leave output storage untouched. Consumers must use the
same row mask; masked output values are unspecified. The vLLM projection
pipeline's peer consumer already suppresses those reads and writes zero
padding outputs. No pointer table, segment array, preparation launch or
output-clearing launch is introduced.

The CUTLASS mainloop uses the observed SM120 producer geometry: 128x64x64
CTAs, 64x32x64 warp tiles, three asynchronous-copy stages, FP16/BF16 inputs
and output, and FP32 accumulation. Shared-memory attributes are prepared
once per device and dtype. The source-derived JIT identity keeps revised
modules distinct during concurrent serving, and the module is registered
for AOT packaging. This is a fixed recipe, without an algorithm sweep.

Validation runs through the complete Gemma4 W16/A16 TP2 server and the
canonical adaptive SVG/video clients. Operator measurements compare the
projection, reduction and consumer with unsplit matrix multiplication and
ordinary TP reduction. Graph traces measure the surviving device work;
unprofiled joint moments compare context and live-query work. The low-level
primitive has no automatic tensor dump or sampling wrapper. It introduces
no standalone kernel benchmark or generated trace fixture.

The first isolated SM120 module build completes on September 9. Numerical,
graph and serving campaigns are tracked under
`/data/h3-runtime/tp2-masked-gemm-v18-20260909` on ws-1. vLLM selects this
recipe for SM120 projection plans; other architectures retain their producer.
The first numerical campaign completes at 18:13:20 UTC: 13 samples and
31,134,720 measured output values per rank. Projection samples include
759 rows, K=2,048/4,096/7,680 and N=3,840. Relative RMS is 0--0.4024%,
maximum absolute error is 2.0, and nonfinite count is zero against the
unsplit projection/reduction/consumer reference. This uncaptured numerical
mode has no padding; full graph replay exercises the changing tile mask.
Full-model graph and adaptive serving measurements now complete. At matched
live-query/context work against the preceding row pipeline, target execution
through sampling changes 64.973 to 63.729 ms at capacity 512, 116.920 to
111.351 ms at 1,024, and 192.520 to 172.016 ms at 2,048: estimated latency
reductions of 1.91%, 4.76% and 10.65%. Working standard errors are undefined,
1.89 and 2.67 percentage points, excluding temporal dependence and model error.
Decode is unchanged by this producer selection; its sparse estimates do not
establish a uniform frontier gain. These are full-server comparisons, not
standalone GEMM timings or measured DRAM throughput.

A column-partition composition using the same kernel measures 1.73%, 2.73%
and 2.92% higher matched latency than the row composition. Both results are
retained in the vLLM serving review; the canonical service adopts the masked
row composition. Its numerical reference remains the ordinary unsplit GEMM
and reduction, with the error metrics above.

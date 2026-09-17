# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM120 2:4 NVFP4 weights, BF16 activations, and sparse BF16 warp MMA."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def decode_pair(value: Uint32, scale: Uint32, *, loc=None, ip=None) -> Uint32:
    """Decode E2M1 and S0E5M3 directly to BF16 without narrowing activations."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
            """{
        .reg .b8 byte;
        .reg .b16 lo, hi, blo, bhi;
        .reg .b32 values, scales;
        cvt.u8.u32 byte, $1;
        cvt.rn.f16x2.e2m1x2 values, byte;
        mul.lo.u32 scales, $2, 0x00800080;
        mul.f16x2 values, values, scales;
        mov.b32 {lo, hi}, values;
        cvt.rn.bf16.f16 blo, lo;
        cvt.rn.bf16.f16 bhi, hi;
        mov.b32 $0, {blo, bhi};
        }""",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sparse_mma(a0, a1, a2, a3, b0, b1, b2, b3, e, c0, c1, c2, c3, *, loc=None, ip=None):
    # PTX's selector 0 assigns each row pair's K halves to lanes 4g and 4g+1.
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            v.ir_value(loc=loc, ip=ip)
            for v in (a0, a1, a2, a3, b0, b1, b2, b3, e, c0, c1, c2, c3)
        ],
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9,$10,$11}, "
        "{$13,$14,$15,$16}, $12, 0;",
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def permute_pair(
    a: Uint32, b: Uint32, selector: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip) for v in (a, b, selector)],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class SparseGemmBf16Fp4:
    def __init__(self, m, n, k, m_tiles=2, split_k=1, paired=False, prepared_a=False):
        self.m, self.n, self.k = m, n, k
        self.m_tiles, self.split_k = m_tiles, split_k
        self.paired = paired
        self.prepared_a = prepared_a

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        sf: cute.Tensor,
        meta: cute.Tensor,
        alpha: cute.Tensor,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, w, sf, meta, alpha, y).launch(
            grid=(self.n // 64, cute.ceil_div(self.m, self.m_tiles * 8), self.split_k),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        sf: cute.Tensor,
        meta: cute.Tensor,
        alpha: cute.Tensor,
        y: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bn, bm, bk = cute.arch.block_idx()
        lane = tid % 32
        group, t = lane // 4, lane % 4
        nt = bn * 4 + tid // 32
        m_base = bm * self.m_tiles * 8
        acc = cute.make_rmem_tensor((self.m_tiles, 4), Float32)
        acc.fill(0.0)
        chunk = cute.ceil_div(self.k // 32, self.split_k)
        for kt in cutlass.range(
            bk * chunk, cutlass.min((bk + 1) * chunk, self.k // 32), unroll=4
        ):
            packed = Uint32(w[nt, kt, lane])
            scales = Uint32(sf[nt, kt, group])
            a0 = decode_pair(packed & 255, scales & 255)
            a1 = decode_pair((packed >> 8) & 255, (scales >> 8) & 255)
            a2 = decode_pair((packed >> 16) & 255, (scales >> 16) & 255)
            a3 = decode_pair(packed >> 24, scales >> 24)
            if cutlass.const_expr(self.paired):
                e = Uint32(meta[nt, kt, group]) >> (t % 2 * 8)
                e = (e & Uint32(0x000F000F)) * Uint32(0x11) | (
                    e & Uint32(0x00F000F0)
                ) * Uint32(0x110)
            else:
                e = Uint32(meta[nt, kt, group * 2 + t % 2])
            for mi in cutlass.range_constexpr(self.m_tiles):
                m = m_base + mi * 8 + group
                b0, b1, b2, b3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if cutlass.const_expr(self.prepared_a):
                    base = ((m_base // 8 + mi) * (self.k // 32) + kt) * 4
                    b0 = Uint32(x[base, lane])
                    b1 = Uint32(x[base + 1, lane])
                    b2 = Uint32(x[base + 2, lane])
                    b3 = Uint32(x[base + 3, lane])
                elif m < self.m:
                    if cutlass.const_expr(self.paired):
                        col = kt * 16 + t % 2 * 2
                        selector = Uint32(0x5410 + t // 2 * 0x2222)
                        b0 = permute_pair(
                            Uint32(x[m, col]), Uint32(x[m, col + 1]), selector
                        )
                        b1 = permute_pair(
                            Uint32(x[m, col + 4]), Uint32(x[m, col + 5]), selector
                        )
                        b2 = permute_pair(
                            Uint32(x[m, col + 8]), Uint32(x[m, col + 9]), selector
                        )
                        b3 = permute_pair(
                            Uint32(x[m, col + 12]), Uint32(x[m, col + 13]), selector
                        )
                    else:
                        b0 = Uint32(x[m, kt * 16 + t])
                        b1 = Uint32(x[m, kt * 16 + t + 4])
                        b2 = Uint32(x[m, kt * 16 + t + 8])
                        b3 = Uint32(x[m, kt * 16 + t + 12])
                d0, d1, d2, d3 = sparse_mma(
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    b2,
                    b3,
                    e,
                    acc[mi, 0],
                    acc[mi, 1],
                    acc[mi, 2],
                    acc[mi, 3],
                )
                acc[mi, 0], acc[mi, 1] = d0, d1
                acc[mi, 2], acc[mi, 3] = d2, d3
        for mi in cutlass.range_constexpr(self.m_tiles):
            for ri in cutlass.range_constexpr(2):
                m = m_base + mi * 8 + t * 2 + ri
                if m < self.m:
                    y[bk, m, nt * 16 + group] = (acc[mi, ri] * alpha[0]).to(
                        y.element_type
                    )
                    y[bk, m, nt * 16 + group + 8] = (acc[mi, ri + 2] * alpha[0]).to(
                        y.element_type
                    )


class PackActivations:
    def __init__(self, m, k, padded_m, paired):
        self.m, self.k, self.padded_m, self.paired = m, k, padded_m, paired

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, y).launch(
            grid=(cute.ceil_div(self.padded_m * self.k // 2, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, y: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = bid * 256 + tid
        if index < self.padded_m * self.k // 2:
            lane = index % 32
            q = index // 32 % 4
            kt = index // 128 % (self.k // 32)
            m = index // (self.k * 4) * 8 + lane // 4
            t = lane % 4
            value = Uint32(0)
            if m < self.m:
                if cutlass.const_expr(self.paired):
                    col = kt * 16 + q * 4 + t % 2 * 2
                    value = permute_pair(
                        Uint32(x[m, col]),
                        Uint32(x[m, col + 1]),
                        Uint32(0x5410 + t // 2 * 0x2222),
                    )
                else:
                    value = Uint32(x[m, kt * 16 + q * 4 + t])
            y[index // 32, lane] = value.to(y.element_type)

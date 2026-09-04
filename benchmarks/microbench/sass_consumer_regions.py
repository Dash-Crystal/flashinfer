"""Opcode counts for the two consumer warp-group regions of the sm90 XQA kernel.

Usage: python sass_consumer_regions.py <cuobjdump -sass dump> [<dump> ...]

Region detection (kernel_mha, mixed build): ptxas lays the role bodies out as
IO/converters, gemm1, gemm0, epilogue.  The gemm1 body starts with the
vBar/xBar `consumed` pre-arrives (>= 5 SYNCS.ARRIVE.TRANS64.RED.A1T0 within
16 lines) and the gemm0 body with the qBar/kBar pre-arrives (>= 4 within 16
lines) that precede the QK HGMMA cluster (the PV cluster is the one whose
HGMMAs carry `.tnspA`).  The gemm0 body ends at its tile loop back-edge.  Everything is verified by printing the
anchors; if the layout changes, the anchors are wrong and this says so.
"""
import collections
import re
import sys

OPS = ["HGMMA", "WARPGROUP.DEPBAR", "WARPGROUP.ARRIVE", "BAR.SYNC", "SYNCS.PHASECHK",
       "SYNCS.ARRIVE", "ATOMS", "STSM", "LDS", "STS", "SHFL", "MUFU", "FENCE", "LDSM",
       "REDUX", "VOTE", "MEMBAR", "ERRBAR", "BRA", "USETMAXREG"]


ADDRS = []


def load(path):
    lines = []
    ADDRS.clear()
    for raw in open(path, errors="replace"):
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;", raw)
        if m:
            ADDRS.append(int(m.group(1), 16))
            lines.append(m.group(2))
    return lines


def is_back_branch(lines, i):
    m = re.search(r"\bBRA\b.*?0x([0-9a-f]+)", lines[i])
    return bool(m) and int(m.group(1), 16) < ADDRS[i]


def opcode(instr):
    s = re.sub(r"^@!?U?P\w+\s+", "", instr)
    return s.split()[0] if s else ""


def arrive_blocks(lines, min_count):
    idx = [i for i, l in enumerate(lines) if "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ" in l]
    blocks = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[i] < 16:
            j += 1
        if j - i + 1 >= min_count:
            blocks.append(idx[i])
        i = j + 1
    return blocks


def regions(lines):
    hg = [i for i, l in enumerate(lines) if opcode(l).startswith("HGMMA")]
    pv = [i for i in hg if "tnspA" in lines[i]]
    qk = [i for i in hg if "tnspA" not in lines[i]]
    b5 = [b for b in arrive_blocks(lines, 5) if b < pv[0]]
    b4 = [b for b in arrive_blocks(lines, 4) if pv[-1] < b < qk[0]]
    g1_start = b5[-1]
    g0_start = b4[-1]
    # The gemm0 body ends at its tile loop's back-edge (the only backward branch
    # after the QK cluster; try_wait retry loops live out of line at the kernel
    # end), plus the trailing qBar.consumed arrive and the jump out.
    back = next(i for i in range(qk[-1], len(lines)) if is_back_branch(lines, i))
    g0_end = back + 1
    for i in range(back + 1, min(back + 8, len(lines))):
        g0_end = i + 1
        if opcode(lines[i]) in ("BRA", "EXIT"):
            break
    return {"gemm1": (g1_start, g0_start), "gemm0": (g0_start, g0_end)}, pv, qk


def count(lines, lo, hi):
    c = collections.Counter()
    for l in lines[lo:hi]:
        op = opcode(l)
        for o in OPS:
            if op.startswith(o):
                c[o] += 1
        c["_instr"] += 1
    return c


def main():
    for path in sys.argv[1:]:
        lines = load(path)
        regs, pv, qk = regions(lines)
        print(f"== {path}: {len(lines)} instr; PV HGMMA {len(pv)} @[{pv[0]},{pv[-1]}], QK HGMMA {len(qk)} @[{qk[0]},{qk[-1]}]")
        for name, (lo, hi) in regs.items():
            c = count(lines, lo, hi)
            print(f"  {name:6s} [{lo:5d},{hi:5d}) instr={c['_instr']:5d} " +
                  " ".join(f"{o}={c[o]}" for o in OPS if c[o]))
            print(f"         first: {lines[lo][:60]!r}  last: {lines[hi - 1][:60]!r}")
        c = count(lines, 0, len(lines))
        print(f"  kernel               instr={c['_instr']:5d} " +
              " ".join(f"{o}={c[o]}" for o in OPS if c[o]))


if __name__ == "__main__":
    main()

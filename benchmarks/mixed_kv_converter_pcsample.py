"""P0.4 artifact: sm90 XQA mixed-KV converter SASS class counts + ncu PC-sampling by PC range.

Inputs (produced on the GPU host; see docs/mixed_kv_page_transport_backends.md, P0.4):
  nvdisasm --print-line-info --print-line-info-inline --print-code mha_sm90.sm_90a.cubin > nvdis.txt
      (cubin from `cuobjdump -xelf all xqa_mha_sm90.cuda.o` of a FLASHINFER_JIT_LINEINFO=1 build;
       -lineinfo does not change the SASS: verified addr+text identical to the production build)
  ncu --set full (or --section SourceCounters --warp-sampling-interval 4) -k regex:kernel_mha -s 3 -c 1 ...
  ncu --import x.ncu-rep --page source --csv --print-source sass > x.source.csv

Usage:
  python benchmarks/mixed_kv_converter_pcsample.py sass   nvdis.txt
  python benchmarks/mixed_kv_converter_pcsample.py sample nvdis.txt x.source.csv [--warp-tiles 34816]

Regions are attributed by the mha_sm90.cu line of the outermost inline frame (the converter loop
call sites) and by any frame inside expandPackedStage / issueCompressedPageCopies.  Adjust the
LINE_* constants when the converter loop in csrc/xqa/mha_sm90.cu moves.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys

# csrc/xqa/mha_sm90.cu anchors (commit 5d8519a0 layout).
LINE_EXPAND = (2734, 2819)  # expandPackedStage body
LINE_COPY = (2538, 2607)  # issueCompressedPageCopies body
CALL_K_EXPAND, CALL_V_EXPAND = 2108, 2158
CALL_K_COPY, CALL_V_COPY = 2117, 2164  # steady-state issue{K,V}Copies(idxIter + kAhead)
CALL_K_PROLOGUE, CALL_V_PROLOGUE = 2096, 2149
LOOP_K = (2099, 2100, 2101, 2102, 2106, 2112, 2113, 2121)
LOOP_V = (2150, 2152, 2153, 2154, 2155, 2156, 2161, 2162, 2165)
ROLE_K = (2064, 2126)
ROLE_V = (2127, 2168)

CLASSES = ["PRMT", "LOP3", "SHF", "HMUL2", "HADD2", "F2FP", "LDS", "STS", "LDGSTS", "IMAD", "SEL", "BRA", "SYNCS"]
REASONS = [
    "stall_barrier", "stall_branch_resolving", "stall_dispatch", "stall_drain", "stall_imc", "stall_lg",
    "stall_long_sb", "stall_math", "stall_membar", "stall_mio", "stall_misc", "stall_no_inst",
    "stall_not_selected", "stall_selected", "stall_short_sb", "stall_sleep", "stall_tex", "stall_wait",
]

_INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*$")
_FILE = re.compile(r'//## File "([^"]+)", line (\d+)(?: inlined at "([^"]+)", line (\d+))?')


def parse_nvdis(path: str) -> list[dict]:
    frames: list[tuple[str, int]] = []
    pending = False
    out = []
    for line in open(path):
        m = _FILE.search(line)
        if m:
            if not pending:
                frames, pending = [], True
            frames.append((m.group(1).rsplit("/", 1)[-1], int(m.group(2))))
            if m.group(3):
                frames.append((m.group(3).rsplit("/", 1)[-1], int(m.group(4))))
            continue
        m = _INS.match(line)
        if not m:
            continue
        pending = False
        body = m.group(2).strip()
        pred = ""
        if body.startswith("@"):
            pred, body = body.split(None, 1)
        outer = next((l for f, l in reversed(frames) if f == "mha_sm90.cu"), None)
        reg = set()
        for f, l in frames:
            if f == "mha_sm90.cu" and LINE_EXPAND[0] <= l <= LINE_EXPAND[1]:
                reg.add("expand")
            if f == "mha_sm90.cu" and LINE_COPY[0] <= l <= LINE_COPY[1]:
                reg.add("copy")
        out.append(dict(addr=int(m.group(1), 16), pred=pred, op=body.split()[0], body=body,
                        frames=list(frames), outer=outer, reg=reg))
    return out


def regions():
    def outer_is(*lines):
        return lambda i: i["outer"] in lines

    def outer_in(lo, hi):
        return lambda i: i["outer"] is not None and lo <= i["outer"] <= hi

    return [
        ("K expansion (expandPackedStage)", outer_is(CALL_K_EXPAND)),
        ("V expansion", outer_is(CALL_V_EXPAND)),
        ("K copy-issue body (issueCompressedPageCopies)", lambda i: i["outer"] == CALL_K_COPY and "copy" in i["reg"]),
        ("K copy-issue consumed.wait_parity", lambda i: i["outer"] == CALL_K_COPY and "copy" not in i["reg"]),
        ("V copy-issue body", lambda i: i["outer"] == CALL_V_COPY and "copy" in i["reg"]),
        ("V copy-issue consumed.wait_parity", lambda i: i["outer"] == CALL_V_COPY and "copy" not in i["reg"]),
        ("K prologue copies", outer_is(CALL_K_PROLOGUE)),
        ("V prologue copies", outer_is(CALL_V_PROLOGUE)),
        ("K loop overhead (waitGroup, syncwarp, kLoadReady wait, fence, produced.arrive, commit)", outer_is(*LOOP_K)),
        ("V loop overhead", outer_is(*LOOP_V)),
        ("K converter total", outer_in(*ROLE_K)),
        ("V converter total", outer_in(*ROLE_V)),
        ("rest of kernel", lambda i: not (outer_in(*ROLE_K)(i) or outer_in(*ROLE_V)(i))),
    ]


def hist(sel, weight=lambda i: 1.0):
    c = collections.Counter()
    for i in sel:
        c[i["op"].split(".")[0]] += weight(i)
    return c


def fmt_hist(c, scale=1.0):
    main = " ".join(f"{k}={c.get(k, 0) / scale:.1f}" for k in CLASSES)
    other = {k: round(v / scale, 1) for k, v in c.items() if k not in CLASSES and v / scale >= 0.5}
    return f"{main} | other={dict(sorted(other.items(), key=lambda x: -x[1]))}"


def cmd_sass(args):
    ins = parse_nvdis(args.nvdis)
    print(f"{len(ins)} SASS instructions")
    for name, pred in regions():
        sel = [i for i in ins if pred(i)]
        if not sel:
            continue
        addrs = [i["addr"] for i in sel]
        cold = [i for i in sel if i["addr"] >= args.cold]
        print(f"-- {name}: static={len(sel)} PC 0x{min(addrs):05x}-0x{max(addrs):05x}"
              + (f" (cold blocks >=0x{args.cold:x}: {len(cold)})" if cold else ""))
        print("   " + fmt_hist(hist(sel)) + f" | total={len(sel)}")


def cmd_sample(args):
    ins = {i["addr"]: i for i in parse_nvdis(args.nvdis)}
    rows = list(csv.reader(open(args.source_csv)))
    hdr = rows[1]
    data = [dict(zip(hdr, r)) for r in rows[2:] if len(r) >= len(hdr)]
    base = min(int(d["Address"], 16) for d in data)
    for d in data:
        d["_off"] = int(d["Address"], 16) - base
    unmatched = sum(1 for d in data if d["_off"] not in ins)
    tot = sum(float(d["Warp Stall Sampling (All Samples)"]) for d in data)
    print(f"{len(data)} sampled PCs ({unmatched} not in nvdis), {tot:.0f} samples total")
    wt = args.warp_tiles
    for name, pred in regions():
        sel = [d for d in data if d["_off"] in ins and pred(ins[d["_off"]])]
        if not sel:
            continue
        s = sum(float(d["Warp Stall Sampling (All Samples)"]) for d in sel)
        ni = sum(float(d["Warp Stall Sampling (Not-issued Samples)"]) for d in sel)
        ie = sum(float(d["Instructions Executed"]) for d in sel)
        reasons = {r: sum(float(d[r]) for d in sel) for r in REASONS}
        print(f"-- {name}: static={len(sel)} samples={s:.0f} ({100 * s / tot:.1f}% of kernel) not-issued={ni:.0f}"
              f" warp-instr executed={ie:.0f} = {ie / wt:.1f} per warp-tile")
        if s:
            print("   stall shares (all samples): " + ", ".join(
                f"{r[6:]}={100 * v / s:.1f}%" for r, v in sorted(reasons.items(), key=lambda x: -x[1]) if v))
        dyn = collections.Counter()
        for d in sel:
            dyn[ins[d["_off"]]["op"].split(".")[0]] += float(d["Instructions Executed"])
        print("   executed per warp-tile: " + fmt_hist(dyn, wt))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sass")
    a.add_argument("nvdis")
    a.add_argument("--cold", type=lambda x: int(x, 0), default=0xB000, help="first address of out-of-line cold blocks")
    a.set_defaults(fn=cmd_sass)
    b = sub.add_parser("sample")
    b.add_argument("nvdis")
    b.add_argument("source_csv")
    b.add_argument("--warp-tiles", type=int, default=8704 * 4,
                   help="converter warp-tiles per operand (4 warps x tiles; 8704 tiles for B=17,S=4096,8 heads)")
    b.set_defaults(fn=cmd_sample)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

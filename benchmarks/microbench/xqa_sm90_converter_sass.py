"""Per-region SASS class counts for the sm90 XQA mixed-KV converter warps (plan Phase 2 artifact).

Regions are derived from the *source* (csrc/xqa/mha_sm90.cu) so the tool does not go stale when
lines move: the K/V converter role blocks (`warpIdx.z == 3` / `== 4`), the `expandPackedStage`
and `issueCompressedPageCopies` bodies, and (in mhaUtils.cuh) the block decode helpers.  Every
SASS instruction is attributed by its outermost mha_sm90.cu frame (the converter-loop line that
called into the helper) and by whether any inline frame lies inside the helper bodies.

Inputs: an nvdisasm listing with inline line info of the FLASHINFER_JIT_LINEINFO=1 build
(SASS is byte-identical to the production build; see P0.4 in
docs/mixed_kv_page_transport_backends.md):

  cuobjdump -xelf all xqa_mha_sm90.cuda.o
  nvdisasm --print-line-info --print-line-info-inline --print-code mha_sm90.sm_90a.cubin > x.nvdis
  python benchmarks/microbench/xqa_sm90_converter_sass.py x.nvdis csrc/xqa/mha_sm90.cu

Static counts equal dynamic counts inside the expansion (no branch is taken there), so "per lane
per tile" is the static count of the steady-state loop body.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

_INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*$")
_FILE = re.compile(r'//## File "([^"]+)", line (\d+)(?: inlined at "([^"]+)", line (\d+))?')

CLASSES = ["PRMT", "LOP3", "SHF", "IMAD", "IADD3", "LEA", "HMUL2", "HADD2", "F2FP", "FMUL", "FMNMX",
           "LDS", "STS", "LDGSTS", "SEL", "ISETP", "FSETP", "CS2R", "MOV", "UMOV", "BRA", "BSSY",
           "BSYNC", "SYNCS", "VOTE", "MEMBAR", "FENCE", "DEPBAR", "WARPSYNC", "NOP"]


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
        out.append(dict(addr=int(m.group(1), 16), pred=pred, op=body.split()[0], body=body,
                        frames=list(frames), outer=outer))
    return out


def _find(lines: list[str], pattern: str, start: int = 0, definition: bool = False) -> int:
    """1-based line of the first match; with definition=True skip matches whose statement ends
    in ';' (forward declarations) and return the one whose signature ends in '{'."""
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            if definition:
                j = i
                while ";" not in lines[j] and "{" not in lines[j]:
                    j += 1
                if ";" in lines[j] and "{" not in lines[j]:
                    continue
            return i + 1  # 1-based
    raise SystemExit(f"anchor not found: {pattern!r}")


def _body_end(lines: list[str], start_1based: int) -> int:
    """First line matching '^}' after start (1-based, inclusive)."""
    for i in range(start_1based, len(lines)):
        if lines[i].startswith("}"):
            return i + 1
    raise SystemExit("body end not found")


def source_regions(src_path: str) -> dict[str, tuple[int, int]]:
    lines = open(src_path).read().split("\n")
    r = {}
    s = _find(lines, r"^__device__ __forceinline__ void expandPackedStage\(", definition=True)
    r["expand"] = (s, _body_end(lines, s))
    s = _find(lines, r"^__device__ __forceinline__ (uint32_t|void) issueCompressedPageCopies\(",
              definition=True)
    r["copy"] = (s, _body_end(lines, s))
    k = _find(lines, r"\} else if \(warpIdx\.z == 3\) \{")
    v = _find(lines, r"assert\(warpIdx\.z == 4\);")
    end = _find(lines, r"^#if MIXED_KV_TRACE$", v)
    r["roleK"] = (k, v - 1)
    r["roleV"] = (v, end)
    return r


def hist(sel):
    c = collections.Counter()
    for i in sel:
        c[i["op"].split(".")[0]] += 1
    return c


def fmt_hist(c):
    main = " ".join(f"{k}={c[k]}" for k in CLASSES if c.get(k))
    other = {k: v for k, v in c.items() if k not in CLASSES}
    return main + (f" | other={dict(sorted(other.items(), key=lambda x: -x[1]))}" if other else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nvdis")
    ap.add_argument("source", help="csrc/xqa/mha_sm90.cu matching the build")
    ap.add_argument("--dump", help="write the attributed SASS of this region (e.g. K-expand) to stdout")
    ap.add_argument("--paths", action="store_true",
                    help="split the K-expand region at its branches into the executed paths of a "
                         "static fp8/fp4 build (bad-page test, fold vote, fold loop, two-multiply loop, "
                         "zero fill) and print per-path class counts; the fold path is the steady state")
    args = ap.parse_args()
    ins = parse_nvdis(args.nvdis)
    reg = source_regions(args.source)
    print(f"{len(ins)} SASS instructions; source regions: " +
          ", ".join(f"{k}={v[0]}-{v[1]}" for k, v in reg.items()))

    def in_range(i, name):
        lo, hi = reg[name]
        return i["outer"] is not None and lo <= i["outer"] <= hi

    def has_frame(i, name):
        lo, hi = reg[name]
        return any(f == "mha_sm90.cu" and lo <= l <= hi for f, l in i["frames"])

    groups = []
    for role in ("K", "V"):
        rn = "role" + role
        role_ins = [i for i in ins if in_range(i, rn)]
        exp = [i for i in role_ins if has_frame(i, "expand")]
        cpy = [i for i in role_ins if has_frame(i, "copy") and not has_frame(i, "expand")]
        rest = [i for i in role_ins if not has_frame(i, "expand") and not has_frame(i, "copy")]
        groups += [(f"{role}-expand (expandPackedStage, all call sites)", exp),
                   (f"{role}-copy-issue (issueCompressedPageCopies body, prologue + steady)", cpy),
                   (f"{role}-loop-overhead (waits, syncwarp, fence, arrive, commit, tag rotation)", rest),
                   (f"{role}-role total", role_ins)]
    for name, sel in groups:
        if not sel:
            continue
        addrs = [i["addr"] for i in sel]
        print(f"-- {name}: {len(sel)} SASS, PC 0x{min(addrs):05x}-0x{max(addrs):05x}")
        print("   " + fmt_hist(hist(sel)))
    if args.paths:
        exp = [i for i in ins if in_range(i, "roleK") and has_frame(i, "expand")]
        bras = [i["addr"] for i in exp if i["op"].startswith("BRA")]
        if len(bras) < 4:
            raise SystemExit(f"expected >= 4 branches in K-expand (static fp8/fp4 build), found {len(bras)}")
        b0, b1, b2, b3 = bras[:4]
        seg = {"prologue (to the bad-page branch)": [i for i in exp if i["addr"] <= b0],
               "scale prep + fold vote": [i for i in exp if b0 < i["addr"] <= b1],
               "loop A": [i for i in exp if b1 < i["addr"] <= b2],
               "loop B": [i for i in exp if b2 < i["addr"] <= b3],
               "zero fill (past the sequence end)": [i for i in exp if i["addr"] > b3]}
        n_hmul = lambda sel: sum(1 for i in sel if i["op"].startswith("HMUL2"))
        fold, two = (seg["loop A"], seg["loop B"]) if n_hmul(seg["loop A"]) == 32 else (seg["loop B"], seg["loop A"])
        for name, sel in seg.items():
            print(f"-- {name}: {len(sel)}  {fmt_hist(hist(sel))}")
        steady = seg["prologue (to the bad-page branch)"] + seg["scale prep + fold vote"] + fold
        print(f"== executed per lane per tile, folded scale (steady state): {len(steady)}")
        print("   " + fmt_hist(hist(steady)))
        print(f"== executed per lane per tile, two-multiply fallback: "
              f"{len(steady) - len(fold) + len(two)}")
    if args.dump:
        role, part = args.dump.split("-")
        rn = "role" + role
        for i in ins:
            if in_range(i, rn) and ((part == "expand" and has_frame(i, "expand")) or
                                    (part == "copy" and has_frame(i, "copy") and not has_frame(i, "expand")) or
                                    (part == "overhead" and not has_frame(i, "expand") and not has_frame(i, "copy"))):
                fr = ",".join(f"{f}:{l}" for f, l in i["frames"] if f != "mha_sm90.cu" or l == i["outer"])
                print(f"/*{i['addr']:05x}*/ {i['pred']:>4} {i['body']:<60} ; {fr}")


if __name__ == "__main__":
    main()

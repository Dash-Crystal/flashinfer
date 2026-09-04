"""Per-CTA prologue / fill attribution of the persistent sm90 mixed build
(round 3, docs/mixed_kv_speed_round3_fill.md).

Reads 'TRACE ctarec ... prolog <24 ints>' lines written by a MIXED_KV_TRACE=1
build (mixedKvCtaTrace words 4..27, ns relative to the CTA's start stamp), the
'TRACE ctaprolog2 <cta> <12 ints>' (words 28..39) and 'TRACE ctaprolog3 <cta>
<8 ints>' (words 40..47, round-3 fill cut) continuation lines from
xqa_mixed_trace_once.py output, and prints per mode / per launch the median
(and min / max) over CTAs of each stamp and of the dependent segments, split by
dispatch slot (CTA ids < P/2 vs >= P/2) when --split is given.  --summary
prints, per mode, each quantity's per-launch medians and the median over the
launches (the form the design's "today" and accept columns use).
"""
import argparse
import re
import statistics

NAMES = {
    4: "scan_seqlens_returned", 5: "scan_published", 6: "syncthreads_passed",
    7: "Kfill_A_done_warm_rerun", 8: "Kfill_pages_returned", 9: "Kfill_formats_returned",
    10: "Vfill_A_done", 11: "Vfill_pages_returned", 12: "Vfill_formats_returned",
    13: "Kmeta_arrive", 14: "Kconv0_tile0_committed", 15: "Kconv0_tile0_landed",
    16: "Kconv0_tile0_produced", 17: "Vconv0_tile0_committed", 18: "Vconv0_tile0_landed",
    19: "Vconv0_tile0_produced", 20: "gemm0_kwait0_passed", 21: "Qwarp_consumed_passed",
    22: "Qwarp_produced_arrive", 23: "gemm1_vwait0_passed", 24: "Kconv3_tile0_produced",
    25: "Vconv3_tile0_produced", 26: "Kfill_A_done_cold", 27: "gemm0w3_before_kwait0", 28: "gemm0w3_kwait0_passed",
    29: "gemm0_before_kwait0", 30: "gemm1_before_vwait0", 31: "Kconv1_tile0_produced",
    32: "Kconv2_tile0_produced", 33: "Vconv1_tile0_produced", 34: "Vconv2_tile0_produced",
    35: "gemm0w1_kwait0_passed", 36: "gemm0w2_kwait0_passed", 37: "syncthreads_released",
    38: "Qwarp_cursor_next_done", 39: "Kconv0_tile0_expanded",
    # round-3 fill cut (third printf line)
    40: "scan_fast_pages_returned", 41: "Kconv0_at_issue0", 42: "Kconv0_tile0_phase_complete",
    43: "Kconv0_tile1_committed", 44: "Vconv0_tile0_phase_complete", 45: "Kconv0_tile1_landed",
    46: "Kfill_chunk0_entered", 47: "scan_fast_records_stored",
}
NWORDS = 36
# dependent chain segments (from, to) -> label
SEGS = [
    (0, 4, "start -> seqLens returned (RT1)"),
    (4, 5, "seqLens -> sched published (div_u64 x2, pass 2)"),
    (5, 6, "published -> __syncthreads ARRIVAL of thread 0 (slot 6; not the release)"),
    (5, 37, "published -> __syncthreads released (dependent LDS)"),
    (37, 26, "sync released -> K fill phase A done, cold pass (setmaxnreg, preExit, cursor init, walk)"),
    (26, 7, "K fill phase A, warm re-run of the same code"),
    (29, 27, "gemm0 warp 0 -> warp 3 arrival at the kBar(0) wait"),
    (20, 28, "gemm0 warp 0 -> warp 3 kBar(0) passed"),
    (20, 35, "gemm0 warp 0 -> warp 1 kBar(0) passed"),
    (20, 36, "gemm0 warp 0 -> warp 2 kBar(0) passed"),
    (15, 39, "K conv warp 0: expand (landed -> expanded)"),
    (39, 16, "K conv warp 0: fence.proxy.async + produced.arrive"),
    (7, 8, "K fill: page indices returned (RT2)"),
    (8, 9, "K fill: page formats returned (RT3)"),
    (9, 13, "K formats -> kMetaReady[0].arrive"),
    (13, 14, "kMetaReady -> K conv warp 0 tile 0 committed (parity wait + issue)"),
    (14, 15, "K conv warp 0: tile 0 copies landed (RT4)"),
    (15, 16, "K conv warp 0: expand + fence + arrive"),
    (16, 24, "K conv warp 0 produced -> warp 3 produced (page landing skew)"),
    (16, 31, "K conv warp 0 produced -> warp 1 produced"),
    (16, 32, "K conv warp 0 produced -> warp 2 produced"),
    (24, 20, "K conv warp 3 produced -> gemm0 kBar(0) passed"),
    (37, 29, "sync released -> gemm0 before kBar(0) wait (setmaxnreg + pre-arrives)"),
    (37, 38, "sync released -> Q warp cursor.next done"),
    (38, 21, "Q warp: Q load RT + qBar[0].consumed wait"),
    (20, 1, "gemm0: Q wait (kwait0 -> firstk)"),
    (0, 1, "start -> firstk (fill)"),
    (6, 21, "sync -> Q warp qBar.consumed passed (Q load RT)"),
    (21, 22, "Q warp store + fence + arrive"),
    (10, 11, "V fill: page indices returned"),
    (11, 12, "V fill: page formats returned"),
    (17, 18, "V conv warp 0: tile 0 copies landed"),
    (18, 19, "V conv warp 0: expand + fence + arrive"),
    (19, 25, "V conv warp 0 produced -> warp 3 produced"),
    (19, 33, "V conv warp 0 produced -> warp 1 produced"),
    (19, 34, "V conv warp 0 produced -> warp 2 produced"),
    (25, 23, "V conv warp 3 produced -> gemm1 vBar(0) passed"),
    (37, 30, "sync released -> gemm1 before vBar(0) wait"),
    (0, 23, "start -> gemm1 vBar(0) passed"),
    # round 3 (fast path / tile-ordered burst / true phase completion)
    (4, 40, "scan: seqLens -> fast-path pages returned (scan ALU + fast arithmetic + RT2)"),
    (40, 47, "scan: fast pages -> fast records stored (fmt, gather, STS)"),
    (47, 5, "scan: fast records stored -> sched published"),
    (37, 41, "sync released -> K conv warp 0 at issue(0) (setmaxnreg.inc + nbFast LDS)"),
    (41, 14, "K conv warp 0: issue(0) + commit (cold)"),
    (0, 14, "start -> first K copy committed (chain head)"),
    (15, 43, "K conv warp 0: tile 0 landed -> tile 1 committed (R2 constants + waitGroup<0> + issue)"),
    (43, 45, "K conv warp 0: tile 1 committed -> landed"),
    (16, 42, "K conv warp 0: arrive issued -> kBar(0).produced phase complete"),
    (42, 20, "K conv: produced(0) complete -> gemm0 kBar(0) passed (true gap)"),
    (19, 44, "V conv warp 0: arrive issued -> vBar(0).produced phase complete"),
    (44, 23, "V conv: produced(0) complete -> gemm1 vBar(0) passed (true gap)"),
    (37, 46, "sync released -> K loader fillTileMeta(chunk 0) entered"),
    (46, 13, "K loader: chunk 0 fill (walk + RT2 [+ RT3] + STS) -> kMetaReady[0].arrive"),
]
PAT = re.compile(r"TRACE ctarec (\d+) start (\d+) firstk (\d+) last (\d+) end (\d+) tiles (\d+)"
                 r"(?: smid (\d+) range (\d+))? prolog ((?:-?\d+ ?){24})")
PAT2 = re.compile(r"TRACE ctaprolog2 (\d+) ((?:-?\d+ ?){12})")
PAT3 = re.compile(r"TRACE ctaprolog3 (\d+) ((?:-?\d+ ?){8})")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def parse(path):
    text = open(path, errors="replace").read()
    parts = re.split(r"^=== MODE (\S+) q_len (\d+) launch (\d+) ===$", text, flags=re.M)
    out = []
    for i in range(1, len(parts), 4):
        mode, launch, body = parts[i], int(parts[i + 2]), parts[i + 3].split("=== END")[0]
        recs = {}
        for m in PAT.finditer(body):
            cta = int(m.group(1))
            start, firstk = int(m.group(2)), int(m.group(3))
            words = [int(x) for x in m.group(9).split()]
            rec = {0: 0, 1: firstk - start}
            for k, w in enumerate(words):
                rec[4 + k] = w
            rec["tiles"] = int(m.group(6))
            recs[cta] = rec
        for m in PAT2.finditer(body):
            cta = int(m.group(1))
            if cta in recs:
                for k, w in enumerate(int(x) for x in m.group(2).split()):
                    recs[cta][28 + k] = w
        for m in PAT3.finditer(body):
            cta = int(m.group(1))
            if cta in recs:
                for k, w in enumerate(int(x) for x in m.group(2).split()):
                    recs[cta][40 + k] = w
        out.append((mode, launch, recs))
    return out


def report(recs, label):
    recs = {c: r for c, r in recs.items() if r["tiles"] > 0}
    if not recs:
        return
    print(f"  [{label}] ctas {len(recs)}")
    print("   stamp (us from start): median  (min / max)")
    for slot in sorted(NAMES):
        v = [r[slot] / 1e3 for r in recs.values() if slot in r]
        if not v:
            continue
        print(f"    {slot:2d} {NAMES[slot]:<28s} {med(v):6.2f}  ({min(v):6.2f} / {max(v):6.2f})")
    print("   segment (us): median  (min / max)")
    for a, b, lab in SEGS:
        v = [(r[b] - r[a]) / 1e3 for r in recs.values() if a in r and b in r]
        if not v:
            continue
        print(f"    {lab:<64s} {med(v):6.2f}  ({min(v):6.2f} / {max(v):6.2f})")


def _drop_top(recs, n):
    top = sorted(recs, key=lambda c: recs[c][1], reverse=True)[:n]
    return {c: r for c, r in recs.items() if c not in top and r["tiles"] > 0}


def summary(parsed, exclude_top):
    """Per mode: per-launch medians of every stamp / segment and their median over
    launches (the 'today' / accept form of docs/mixed_kv_speed_round3_fill.md)."""
    modes = []
    for mode, _, _ in parsed:
        if mode not in modes:
            modes.append(mode)
    for mode in modes:
        launches = [_drop_top(r, exclude_top) for m, _, r in parsed if m == mode and r]
        launches = [r for r in launches if r]
        if not launches:
            continue
        print(f"===== {mode}: {len(launches)} launches, per-launch medians | median over launches")
        for slot in sorted(NAMES):
            per = [med([r[slot] / 1e3 for r in recs.values() if slot in r]) for recs in launches]
            per = [v for v in per if v == v]
            if per:
                print(f"  stamp {slot:2d} {NAMES[slot]:<30s} "
                      f"{' '.join(f'{v:6.2f}' for v in per)} | {med(per):6.2f}")
        for a, b, lab in SEGS:
            per = [med([(r[b] - r[a]) / 1e3 for r in recs.values() if a in r and b in r])
                   for recs in launches]
            per = [v for v in per if v == v]
            if per:
                print(f"  seg {lab:<78s} {' '.join(f'{v:6.2f}' for v in per)} | {med(per):6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--split", action="store_true", help="also split by dispatch slot (id < P/2)")
    ap.add_argument("--exclude-top", type=int, default=8,
                    help="drop the N CTAs with the largest fill (co-tenant time slices)")
    ap.add_argument("--summary", action="store_true",
                    help="per mode: per-launch medians and the median over launches only")
    a = ap.parse_args()
    parsed = parse(a.log)
    if a.summary:
        summary(parsed, a.exclude_top)
        return
    for mode, launch, recs in parsed:
        if not recs:
            continue
        recs = _drop_top(recs, a.exclude_top)
        print(f"##### {mode} launch {launch}")
        report(recs, "all")
        if a.split:
            P = max(recs) + 1
            report({c: r for c, r in recs.items() if c < P // 2}, "ids < P/2")
            report({c: r for c, r in recs.items() if c >= P // 2}, "ids >= P/2")


if __name__ == "__main__":
    main()

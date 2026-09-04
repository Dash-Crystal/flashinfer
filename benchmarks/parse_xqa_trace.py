"""Parse MIXED_KV_TRACE=1 'TRACE tile' lines from a bench log.

Groups lines into launches (a new launch starts when the tile index does not
increase, so a window moved by -DMIXED_KV_TRACE_TILE0 works), computes per-slot
differences per tile, and prints medians over launches for the steady-state
tiles.  Cycle -> us conversion uses --sm-mhz (default 1980, H200 boost).

Persistent build (lever [8]): 'TRACE ctarec <cta> start <ns> firstk <ns> last
<ns> end <ns> tiles <n>' lines (one per CTA per launch) give the per-CTA
histogram: start spread, body (firstk -> last), end spread, idle fraction.
"""
import argparse
import re
import statistics
import sys

SLOT_NAMES = [
    "g0_kwait", "g0_mma", "g0_smax", "g0_xarr",
    "g1_vwait", "g1_xwait", "g1_rs", "g1_mma",
    "kl_start", "kl_iss", "vl_start", "vl_iss",
    "kc_ready", "kc_done", "vc_ready", "vc_done",
]
PAT = re.compile(r"TRACE tile (\d+) (.*)")
PAT_CTA = re.compile(
    r"TRACE ctarec (\d+) start (\d+) firstk (\d+) last (\d+) end (\d+) tiles (\d+)")
# 9ce501fe appends " smid <n> range <c>"; the fields above are unchanged.


def _launch_end(line):
    """Mode name if this line ends a launch group: a bench JSON result line
    (bench_xqa_mixed_page_transport.py) or an '=== END <mode> ===' marker
    (xqa_mixed_trace_once.py); else None."""
    if line.startswith("{\"q_len\""):
        import json
        return json.loads(line)["mode"]
    m = re.match(r"=== END (\S+) ===", line)
    return m.group(1) if m else None


def parse(path, mode=None):
    """Launches are attributed to the mode of the JSON result line that follows them."""
    launches = []
    pending = []
    cur = None
    last_tile = None
    for line in open(path, errors="replace"):
        m_end = _launch_end(line)
        if m_end is not None:
            if mode is None or m_end == mode:
                launches.extend(pending)
            pending = []
            cur = None
            last_tile = None
            continue
        m = PAT.search(line)
        if not m:
            continue
        tile = int(m.group(1))
        nums = [int(t) for t in m.group(2).split() if re.fullmatch(r"-?\d+", t)]
        if len(nums) != 16:
            continue
        if last_tile is None or tile <= last_tile:
            cur = {}
            pending.append(cur)
        last_tile = tile
        cur[tile] = nums
    return launches


def parse_cta_records(path, mode=None):
    """Per-launch lists of (cta, start, firstk, last, end, tiles); a launch's
    records are attributed to the JSON result line that follows them."""
    launches = []
    pending = []
    cur_mode = None  # set by the '=== MODE <mode> ...' markers of xqa_mixed_trace_once.py
    for line in open(path, errors="replace"):
        m_end = _launch_end(line)
        m_beg = re.match(r"=== MODE (\S+) ", line)
        if m_end is not None or m_beg is not None:
            # one launch per MODE marker; a bench JSON line closes the group too
            group_mode = m_end if m_end is not None else cur_mode
            if pending and (mode is None or group_mode == mode):
                launches.append(pending)
            pending = []
            if m_beg is not None:
                cur_mode = m_beg.group(1)
            continue
        m = PAT_CTA.search(line)
        if m:
            pending.append(tuple(int(g) for g in m.groups()))
    return launches


def print_cta_histogram(launches):
    for i, recs in enumerate(launches):
        recs = [r for r in recs if r[5] > 0]
        if not recs:
            continue
        t0 = min(r[1] for r in recs)
        starts = [(r[1] - t0) / 1e3 for r in recs]
        ends = [(r[4] - t0) / 1e3 for r in recs]
        bodies = [(r[3] - r[2]) / 1e3 for r in recs]
        fills = [(r[2] - r[1]) / 1e3 for r in recs]
        wall = max(ends)
        idle = [(wall - b - f) / wall for b, f in zip(bodies, fills)]
        tiles = [r[5] for r in recs]
        print(f"launch {i}: ctas {len(recs)} tiles {min(tiles)}..{max(tiles)} | "
              f"start spread {max(starts) - min(starts):.2f} us | fill median {med(fills):.2f} us | "
              f"body median {med(bodies):.2f} us (min {min(bodies):.2f} max {max(bodies):.2f}, "
              f"per tile {med(bodies) / max(1, med(tiles)):.3f}) | "
              f"end median {med(ends):.2f} max {wall:.2f} (spread {wall - med(ends):.2f}) | "
              f"idle fraction median {med(idle):.3f}")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--sm-mhz", type=float, default=1980.0)
    ap.add_argument("--first-tile", type=int, default=2, help="first steady-state tile (inclusive)")
    ap.add_argument("--last-tile", type=int, default=7, help="last tile (inclusive)")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--skip-launches", type=int, default=0, help="ignore the first N launches (warmup)")
    ap.add_argument("--mode", default=None, help="only launches attributed to this bench mode")
    a = ap.parse_args()
    cta_launches = parse_cta_records(a.log, a.mode)[a.skip_launches:]
    if cta_launches:
        print(f"per-CTA record launches parsed: {len(cta_launches)}")
        print_cta_histogram(cta_launches)
    launches = parse(a.log, a.mode)[a.skip_launches:]
    print(f"launches parsed: {len(launches)}")
    if not launches:
        return
    ntiles = min(len(l) for l in launches)
    print(f"tiles per launch (min): {ntiles}")
    cyc2us = 1.0 / a.sm_mhz

    # cadence = slot[t+1] - slot[t] for the chosen slot, steady-state tiles
    for slot in (a.slot, 7, 8, 13):
        diffs_all = []
        per_tile = {}
        for l in launches:
            for t in range(a.first_tile, min(a.last_tile, ntiles - 1)):
                if t in l and t + 1 in l:
                    d = l[t + 1][slot] - l[t][slot]
                    diffs_all.append(d)
                    per_tile.setdefault(t, []).append(d)
        if diffs_all:
            print(f"cadence slot{slot} ({SLOT_NAMES[slot]}): median {med(diffs_all):.0f} cyc = {med(diffs_all)*cyc2us:.3f} us"
                  f" | min {min(diffs_all)} max {max(diffs_all)} n={len(diffs_all)}")
            print("   per-tile medians (cyc): " + " ".join(f"t{t}->{t+1}:{med(v):.0f}" for t, v in sorted(per_tile.items())))

    # intra-tile segments for consumer slots
    print("intra-tile segment medians (cyc), steady-state tiles:")
    segs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7), (12, 13), (14, 15), (8, 9), (10, 11)]
    for s0, s1 in segs:
        vals = []
        for l in launches:
            for t in range(a.first_tile, min(a.last_tile + 1, ntiles)):
                if t in l:
                    vals.append(l[t][s1] - l[t][s0])
        if vals:
            print(f"  slot{s1}-slot{s0} ({SLOT_NAMES[s1]}-{SLOT_NAMES[s0]}): median {med(vals):.0f} cyc = {med(vals)*cyc2us:.3f} us  [min {min(vals)} max {max(vals)}]")

    # absolute slot0 per tile for the first launch (to see fill)
    l = launches[len(launches) // 2]
    print("mid-launch slot0 per tile (cyc from t0): " + " ".join(f"t{t}:{l[t][0]}" for t in sorted(l)))
    print("mid-launch slot7 per tile (cyc from t0): " + " ".join(f"t{t}:{l[t][7]}" for t in sorted(l)))


if __name__ == "__main__":
    main()

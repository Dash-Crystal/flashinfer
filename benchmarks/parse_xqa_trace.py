"""Parse MIXED_KV_TRACE=1 'TRACE tile' lines from a bench log.

Groups lines into launches (a new launch starts at tile 0), computes per-slot
differences per tile, and prints medians over launches for the steady-state
tiles.  Cycle -> us conversion uses --sm-mhz (default 1980, H200 boost).
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


def parse(path, mode=None):
    """Launches are attributed to the mode of the JSON result line that follows them."""
    launches = []
    pending = []
    cur = None
    for line in open(path, errors="replace"):
        if line.startswith("{\"q_len\""):
            import json
            rec = json.loads(line)
            if mode is None or rec["mode"] == mode:
                launches.extend(pending)
            pending = []
            cur = None
            continue
        m = PAT.search(line)
        if not m:
            continue
        tile = int(m.group(1))
        nums = [int(t) for t in m.group(2).split() if re.fullmatch(r"-?\d+", t)]
        if len(nums) != 16:
            continue
        if tile == 0:
            cur = {}
            pending.append(cur)
        if cur is None:
            continue
        cur[tile] = nums
    return launches


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

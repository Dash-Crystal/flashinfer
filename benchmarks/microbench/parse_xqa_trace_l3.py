"""Parse MIXED_KV_TRACE=3 output of benchmarks/xqa_mixed_trace_once.py: K-converter segments.

At trace level 3 the 16 per-tile slots of CTA 0 are (csrc/xqa/mha_sm90.cu):
  0 g0:kwait  1 g0:mma  2 g0:smax  3 g0:xarr | 4 g1:vwait 5 g1:xwait 6 g1:rs 7 g1:mma |
  8  K converter: fence.proxy.async + kBar.produced.arrive done   (printed as kl:start)
  9  K converter: copies for tile t+kAhead issued, before commit  (kl:iss)
  10 K converter: after cp.async.commit_group                      (vl:start)
  11 K converter: this tile's copies landed (wait_group + syncwarp) (vl:iss)
  12 K converter: ready to expand (after the rendezvous, if any)   (kc:ready)
  13 K converter: expansion done                                   (kc:done)
  14/15 V converter ready / done.

Per tile t (steady state tiles 2..7, medians over tiles and launches):
  landed->ready  = s12 - s11          expansion = s13 - s12       fence+arrive = s8 - s13
  copy issue     = s9 - s8            commit    = s10 - s9        commit->next landed = s11(t+1) - s10(t)
  converter period = s13(t) - s13(t-1);  gemm0 cadence = s0(t) - s0(t-1);  K-wait = s0(t) - s3(t-1)
Cycles are clock64 at the SM clock measured from the CTA lines (clk / globaltimer).
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict

SEGS = [
    ("landed->ready (s12-s11)", lambda s, p, n: s[12] - s[11]),
    ("expansion (s13-s12)", lambda s, p, n: s[13] - s[12]),
    ("fence+arrive (s8-s13)", lambda s, p, n: s[8] - s[13]),
    ("copy issue (s9-s8)", lambda s, p, n: s[9] - s[8]),
    ("commit (s10-s9)", lambda s, p, n: s[10] - s[9]),
    ("commit->next landed (s11(t+1)-s10)", lambda s, p, n: n[11] - s[10] if n else None),
    ("K conv period (s13(t)-s13(t-1))", lambda s, p, n: s[13] - p[13] if p else None),
    ("V conv period (s15(t)-s15(t-1))", lambda s, p, n: s[15] - p[15] if p else None),
    ("V expansion (s15-s14)", lambda s, p, n: s[15] - s[14]),
    ("gemm0 cadence (s0(t)-s0(t-1))", lambda s, p, n: s[0] - p[0] if p else None),
    ("gemm0 K-wait (s0(t)-s3(t-1))", lambda s, p, n: s[0] - p[3] if p else None),
    ("kc:done -> g0 kwait done (s0-s13)", lambda s, p, n: s[0] - s[13]),
]


def main():
    text = open(sys.argv[1]).read()
    sections = re.split(r"^=== MODE (\S+) q_len (\d+) launch (\d+) ===$", text, flags=re.M)
    per_mode = defaultdict(lambda: defaultdict(list))
    ghz_by_mode = defaultdict(list)
    for i in range(1, len(sections), 4):
        mode, launch, body = sections[i], sections[i + 2], sections[i + 3].split("=== END")[0]
        ctas = [tuple(int(x) for x in m.groups()) for m in re.finditer(
            r"^CTA (\d+) (\d+) (\d+) sm (\d+) iters (\d+) sub (\d+) gt (\d+) (\d+) clk (-?\d+) (-?\d+)$", body, flags=re.M)]
        if ctas:
            ghz = statistics.median((c[9] - c[8]) / (c[7] - c[6]) for c in ctas if c[7] > c[6])
            span = (max(c[7] for c in ctas) - min(c[6] for c in ctas)) / 1e3
            ghz_by_mode[mode].append(ghz)
        else:
            # No per-CTA globaltimer records in this build: nkcut2 runs with the SM clock locked
            # at 1980 MHz (nvidia-smi); P0.3 measured 1.79-1.80 GHz effective under the co-tenant.
            ghz, span = 1.98, float("nan")
        tr = {}
        for m in re.finditer(r"^TRACE tile (\d+) (.*)$", body, flags=re.M):
            nums = [int(tok) for tok in m.group(2).split() if re.fullmatch(r"-?\d+", tok)]
            if len(nums) >= 16:
                tr[int(m.group(1))] = nums[:16]
        if not tr:
            continue
        print(f"## mode={mode} launch={launch} clock {ghz:.3f} GHz span {span:.1f} us tiles {sorted(tr)}")
        for name, f in SEGS:
            vals = []
            for t in sorted(tr):
                if not (2 <= t <= 7):
                    continue
                v = f(tr[t], tr.get(t - 1), tr.get(t + 1))
                if v is not None:
                    vals.append(v)
            if vals:
                med = statistics.median(vals)
                per_mode[mode][name].append(med)
                print(f"   {name:<40} median {med:7.0f} cyc  ({min(vals):.0f}..{max(vals):.0f})")
    print("\n===== per mode: median over launches of the per-launch medians (cycles; us at the measured clock) =====")
    for mode, segs in per_mode.items():
        ghz = statistics.median(ghz_by_mode[mode]) if ghz_by_mode[mode] else 1.98
        print(f"\n{mode}: {len(next(iter(segs.values())))} launches, clock {ghz:.3f} GHz")
        for name, vals in segs.items():
            med = statistics.median(vals)
            print(f"   {name:<40} {med:7.0f} cyc = {med / ghz / 1e3:6.3f} us   (launch medians {min(vals):.0f}..{max(vals):.0f})")


if __name__ == "__main__":
    main()

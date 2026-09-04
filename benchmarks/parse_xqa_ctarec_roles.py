"""Per-CTA ctarec analysis for the sm90 mixed-KV trace build (round 3).

Reads the `TRACE ctarec` lines of `benchmarks/xqa_mixed_trace_once.py` output
(optionally with the pair track's per-role `acc` accumulators) and prints,
per mode and launch: fill / body / tail / end medians over the CTAs, body
deciles (bimodality), and the per-tile per-role segment medians (acc / tiles).
Co-tenant time-slice outliers (tail > 15 us) are excluded.

Usage: python3 benchmarks/parse_xqa_ctarec_roles.py /tmp/r3pair_trace132.log
"""
import re, sys, statistics
med=statistics.median
text=open(sys.argv[1], errors="replace").read()
parts = re.split(r"^=== MODE (\S+) q_len (\d+) launch (\d+) ===$", text, flags=re.M)
for i in range(1,len(parts),4):
    mode, launch, body = parts[i], int(parts[i+2]), parts[i+3].split("=== END")[0]
    recs=[]
    for m in re.finditer(r"TRACE ctarec (\d+) start (\d+) firstk (\d+) last (\d+) end (\d+) tiles (\d+) smid (\d+) range (\d+)(?: warpid (\d+) acc ([\d ]+))?", body):
        c,st,fk,la,en,ti,sm,rg=[int(x) for x in m.groups()[:8]]
        acc=[int(x) for x in m.group(10).split()] if m.group(10) else None
        recs.append((c,st,fk,la,en,ti,sm,rg,acc))
    if not recs: continue
    t0=min(r[1] for r in recs)
    fill=[(r[2]-r[1])/1e3 for r in recs]; bod=[(r[3]-r[2])/1e3 for r in recs]; tail=[(r[4]-r[3])/1e3 for r in recs]; end=[(r[4]-t0)/1e3 for r in recs]
    tiles=[r[5] for r in recs]
    # exclude co-tenant outliers: tail > 15us
    ok=[k for k in range(len(recs)) if tail[k] < 15]
    def s(xs): xs=[xs[k] for k in ok]; return f"{min(xs):.1f}/{med(xs):.1f}/{max(xs):.1f}"
    print(f"### {mode} launch {launch}: n={len(recs)} tiles {min(tiles)}..{max(tiles)} outliers {len(recs)-len(ok)} | start spread {(max(r[1] for r in recs)-t0)/1e3:.1f} | fill {s(fill)} | body {s(bod)} | body/tile {med([bod[k]/tiles[k] for k in ok]):.3f} | tail {s(tail)} | end {s(end)}")
    bs=sorted(bod[k] for k in ok)
    print("   body deciles: "+" ".join(f"{bs[int(q*(len(bs)-1))]:.1f}" for q in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]))
    if recs[0][8]:
        names=["g0 kwait","g0 mma","g0 smax","g0 xarr","g1 vwait","g1 xwait","g1 rs","g1 mma","kl start","kl iss","vl start","vl iss","kc ready","kc done","vc ready","vc done"]
        accs=[recs[k][8] for k in ok]
        medacc=[med(a[j] for a in accs)/1980.0 for j in range(16)]  # us total at 1.98GHz
        nt=med(tiles)
        print("   per-tile role segment medians (us, acc/tiles): "+" | ".join(f"{names[j]} {medacc[j]/nt:.3f}" for j in range(16)))
        roles={"gemm0":[0,1,2,3],"gemm1":[4,5,6,7],"kload":[8,9],"vload":[10,11],"kconv":[12,13],"vconv":[14,15]}
        print("   per-tile role totals (us): "+" ".join(f"{r} {sum(medacc[j] for j in js)/nt:.3f}" for r,js in roles.items()))

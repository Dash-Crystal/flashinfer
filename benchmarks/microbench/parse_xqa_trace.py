"""Parse MIXED_KV_TRACE=1 output (TRACE tile / CTA / MERGE lines) from
benchmarks/xqa_mixed_trace_once.py.

Per launch: CTA-0 per-tile slot differences (cycles) and steady-state medians;
per-CTA lifetime histogram (tile-time multiplier, fixed cost per CTA, tail).
The SM clock is measured from the CTA lines (clock64 delta / globaltimer delta).
Launches whose kernel span is inflated by co-tenant preemption are flagged; the
summary at the end uses the launch with the smallest span per mode."""
import re
import statistics
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
verbose = "-v" in sys.argv
sections = re.split(r"^=== MODE (\S+) q_len (\d+) launch (\d+) ===$", text, flags=re.M)
summary = defaultdict(list)
for i in range(1, len(sections), 4):
    mode, q, launch, body = sections[i], sections[i + 1], sections[i + 2], sections[i + 3]
    body = body.split("=== END")[0]
    ctas = []
    for m in re.finditer(r"^CTA (\d+) (\d+) (\d+) sm (\d+) iters (\d+) sub (\d+) gt (\d+) (\d+) clk (-?\d+) (-?\d+)$", body, flags=re.M):
        bx, by, bz, sm, it, sub, g0, g1, c0, c1 = [int(x) for x in m.groups()]
        ctas.append(dict(bx=bx, by=by, bz=bz, sm=sm, it=it, sub=sub, g0=g0, g1=g1, c0=c0, c1=c1))
    merges = [(int(m.group(3)), int(m.group(4))) for m in re.finditer(r"^MERGE (\d+) (\d+) sm (\d+) gt (\d+)$", body, flags=re.M)]
    if not ctas:
        continue
    ghz = statistics.median((c["c1"] - c["c0"]) / (c["g1"] - c["g0"]) for c in ctas if c["g1"] - c["g0"] > 0)
    gmin = min(c["g0"] for c in ctas); gmax = max(c["g1"] for c in ctas)
    mmax = max((g for _, g in merges), default=gmax)
    span = (mmax - gmin) / 1e3
    print(f"\n##### mode={mode} q={q} launch={launch}  SM clock {ghz:.3f} GHz (clk/gt)  kernel span {span:.1f} us{'  [PREEMPTED by co-tenant]' if span > 300 else ''}")
    tr = {}
    for m in re.finditer(r"^TRACE tile (\d+) g0:kwait (-?\d+) mma (-?\d+) smax (-?\d+) xarr (-?\d+) \| g1:vwait (-?\d+) xwait (-?\d+) rs (-?\d+) mma (-?\d+) \| kl:start (-?\d+) iss (-?\d+) \| vl:start (-?\d+) iss (-?\d+) \| kc:ready (-?\d+) done (-?\d+) \| vc:ready (-?\d+) done (-?\d+)", body, flags=re.M):
        t = int(m.group(1)); tr[t] = [int(x) for x in m.groups()[1:]]
    rec = dict(mode=mode, launch=launch, span=span, ghz=ghz)
    if tr:
        if verbose:
            print("CTA 0 per-tile slot differences (cycles).  slots: 0 kwait(K ready) 1 mma(gemm0 wgmma done) 2 smax(colMax sync + softmax) 3 xarr(X stored + xBar.produced.arrive) | 4 vwait(V ready) 5 xwait(X ready) 6 rs(rescale done) 7 mma(PV done)")
            print(f"{'tile':>4} {'s1-s0':>6} {'s2-s1':>6} {'s3-s2':>6} {'s5-s4':>6} {'s6-s5':>6} {'s7-s6':>6} | {'T_g0=s3-s0':>10} {'T_g1=s7-s4':>10} {'g1work=s7-s5':>12} | {'Kwait=s0(t)-s3(t-1)':>19} {'Vwait=s4(t)-s7(t-1)':>19} {'s5(t)-s3(t)':>11} | {'cad s0':>7} {'cad s7':>7} | {'kc:done-s0':>10} {'vc:done-s4':>10}")
            for t in sorted(tr):
                s = tr[t]; prev = tr.get(t - 1)
                d = lambda a, b: s[a] - s[b]
                kw = f"{s[0]-prev[3]:>19}" if prev else f"{'':>19}"
                vw = f"{s[4]-prev[7]:>19}" if prev else f"{'':>19}"
                c0 = f"{s[0]-prev[0]:>7}" if prev else f"{'':>7}"
                c7 = f"{s[7]-prev[7]:>7}" if prev else f"{'':>7}"
                print(f"{t:>4} {d(1,0):>6} {d(2,1):>6} {d(3,2):>6} {d(5,4):>6} {d(6,5):>6} {d(7,6):>6} | {d(3,0):>10} {d(7,4):>10} {d(7,5):>12} | {kw} {vw} {d(5,3):>11} | {c0} {c7} | {s[13]-s[0]:>10} {s[15]-s[4]:>10}")
        ss = [t for t in sorted(tr) if 2 <= t <= 7 and t - 1 in tr]
        if ss:
            med = lambda f: statistics.median(f(tr[t], tr[t - 1]) for t in ss)
            vals = dict(
                s10=med(lambda s, p: s[1]-s[0]), s21=med(lambda s, p: s[2]-s[1]), s32=med(lambda s, p: s[3]-s[2]),
                s54=med(lambda s, p: s[5]-s[4]), s65=med(lambda s, p: s[6]-s[5]), s76=med(lambda s, p: s[7]-s[6]),
                Tg0=med(lambda s, p: s[3]-s[0]), Tg1=med(lambda s, p: s[7]-s[4]), g1work=med(lambda s, p: s[7]-s[5]),
                kwait=med(lambda s, p: s[0]-p[3]), vwait=med(lambda s, p: s[4]-p[7]), xlag=med(lambda s, p: s[5]-s[3]),
                cad=med(lambda s, p: s[0]-p[0]), kcdone=med(lambda s, p: s[13]-s[0]))
            rec.update(vals)
            print("CTA0 steady-state medians (tiles 2-7, cyc): s1-s0 %(s10).0f  s2-s1 %(s21).0f  s3-s2 %(s32).0f | s5-s4 %(s54).0f  s6-s5 %(s65).0f  s7-s6 %(s76).0f | T_g0 %(Tg0).0f  T_g1 %(Tg1).0f  gemm1 work(s7-s5) %(g1work).0f | gemm0 K-wait(s0(t)-s3(t-1)) %(kwait).0f  gemm1 V-wait(s4(t)-s7(t-1)) %(vwait).0f  X handoff lag(s5-s3) %(xlag).0f | cadence %(cad).0f cyc" % vals + f" = {vals['cad']/ghz/1e3:.2f} us")
    # per-CTA / per-slot
    first_wave = [c for c in ctas if (c["g0"] - gmin) < 2000]
    life_fw = sorted((c["c1"] - c["c0"]) for c in first_wave)
    life_all = sorted((c["c1"] - c["c0"]) for c in ctas)
    c0 = [c for c in ctas if c["bx"] == 0 and c["by"] == 0 and c["bz"] == 0][0]
    print(f"CTAs {len(ctas)} nbSubSeq {ctas[0]['sub']} tiles/CTA {sorted(set(c['it'] for c in ctas))} total tiles {sum(c['it'] for c in ctas)}; first wave {len(first_wave)} CTAs; 136 merges" if len(merges) == 136 else f"CTAs {len(ctas)} merges {len(merges)}")
    print(f"CTA lifetime (13-tile CTAs, cyc->us at {ghz:.2f} GHz): first wave median {statistics.median(life_fw)/ghz/1e3:.2f} min {life_fw[0]/ghz/1e3:.2f} max {life_fw[-1]/ghz/1e3:.2f}; all CTAs median {statistics.median(life_all)/ghz/1e3:.2f} p90 {life_all[9*len(life_all)//10]/ghz/1e3:.2f} max {life_all[-1]/ghz/1e3:.2f}; CTA0 {(c0['c1']-c0['c0'])/ghz/1e3:.2f} us for {c0['it']} tiles")
    if tr and ss:
        F0 = (c0["c1"] - c0["c0"]) - c0["it"] * vals["cad"]
        Fmed = statistics.median(life_fw) - 13 * vals["cad"]
        rec.update(F0=F0 / ghz / 1e3, Fmed=Fmed / ghz / 1e3, life_fw=statistics.median(life_fw) / ghz / 1e3)
        print(f"fixed cost per CTA (lifetime - tiles x CTA0 cadence): CTA0 {F0/ghz/1e3:.2f} us; first-wave median CTA {Fmed/ghz/1e3:.2f} us")
    by_sm = defaultdict(list)
    for c in ctas: by_sm[c["sm"]].append(c)
    slot_tiles = []; slot_end = []; slot_busy = []
    for sm, v in by_sm.items():
        v = sorted(v, key=lambda c: c["g0"])
        slots = []
        for c in v:
            for s in slots:
                if s[0] <= c["g0"] + 3000:
                    s[0] = c["g1"]; s[1] += c["it"]; s[2] += c["g1"] - c["g0"]; break
            else:
                slots.append([c["g1"], c["it"], c["g1"] - c["g0"]])
        for s in slots:
            slot_tiles.append(s[1]); slot_end.append((s[0] - gmin) / 1e3); slot_busy.append(s[2] / 1e3)
    hist = defaultdict(int)
    for t in slot_tiles: hist[t] += 1
    slot_end.sort()
    print(f"slots reconstructed {len(slot_tiles)}; tiles-per-slot histogram {dict(sorted(hist.items()))}")
    print(f"slot end times (us from first start): min {slot_end[0]:.1f} median {statistics.median(slot_end):.1f} p90 {slot_end[9*len(slot_end)//10]:.1f} max {slot_end[-1]:.1f}; last main-loop end {(gmax-gmin)/1e3:.1f}; last MERGE end {span:.1f}")
    # slot end by tiles-per-slot class
    cls = defaultdict(list)
    for t, e in zip(slot_tiles, [ (0) for _ in slot_tiles]): pass
    cls2 = defaultdict(list)
    idx = 0
    for sm, v in by_sm.items():
        pass
    ends_by_tiles = defaultdict(list)
    # recompute pairing tiles->end
    for sm, v in by_sm.items():
        v = sorted(v, key=lambda c: c["g0"]); slots = []
        for c in v:
            for s in slots:
                if s[0] <= c["g0"] + 3000:
                    s[0] = c["g1"]; s[1] += c["it"]; break
            else:
                slots.append([c["g1"], c["it"]])
        for s in slots: ends_by_tiles[s[1]].append((s[0] - gmin) / 1e3)
    print("slot end (median us) by tiles-per-slot:", {k: round(statistics.median(v), 1) for k, v in sorted(ends_by_tiles.items())})
    rec.update(slot_hist=dict(sorted(hist.items())), slot_end_med=statistics.median(slot_end), slot_end_max=slot_end[-1], mainloop_end=(gmax - gmin) / 1e3,
               ends_by_tiles={k: round(statistics.median(v), 1) for k, v in sorted(ends_by_tiles.items())})
    summary[mode].append(rec)

print("\n\n===== SUMMARY: cleanest launch per mode (smallest kernel span) =====")
for mode, recs in summary.items():
    best = min(recs, key=lambda r: r["span"])
    print(f"\n{mode}: launch {best['launch']} span {best['span']:.1f} us (spans of all launches: {[round(r['span'],1) for r in recs]}) clock {best['ghz']:.3f} GHz")
    if "cad" in best:
        print("  CTA0 steady tiles 2-7 (cyc): s1-s0 %(s10).0f s2-s1 %(s21).0f s3-s2 %(s32).0f | s5-s4 %(s54).0f s6-s5 %(s65).0f s7-s6 %(s76).0f | T_g0 %(Tg0).0f T_g1 %(Tg1).0f g1work %(g1work).0f | K-wait %(kwait).0f V-wait %(vwait).0f X-lag %(xlag).0f | cadence %(cad).0f" % best + f" cyc = {best['cad']/best['ghz']/1e3:.2f} us; converter done->K ready {best['kcdone']:.0f}")
        print(f"  fixed per CTA: CTA0 {best['F0']:.2f} us, first-wave median {best['Fmed']:.2f} us; first-wave CTA lifetime median {best['life_fw']:.2f} us")
    print(f"  tiles-per-slot histogram {best['slot_hist']}; slot end median {best['slot_end_med']:.1f} max {best['slot_end_max']:.1f}; main-loop end {best['mainloop_end']:.1f}; merge end {best['span']:.1f}; slot end by class {best['ends_by_tiles']}")
    # medians over clean launches
    clean = [r for r in recs if r["span"] < 300 and "cad" in r]
    if len(clean) >= 2:
        print(f"  over {len(clean)} clean launches: cadence median {statistics.median(r['cad'] for r in clean):.0f} cyc, T_g0 {statistics.median(r['Tg0'] for r in clean):.0f}, g1work {statistics.median(r['g1work'] for r in clean):.0f}, s2-s1 {statistics.median(r['s21'] for r in clean):.0f}, s3-s2 {statistics.median(r['s32'] for r in clean):.0f}, s5-s4 {statistics.median(r['s54'] for r in clean):.0f}, s6-s5 {statistics.median(r['s65'] for r in clean):.0f}, s7-s6 {statistics.median(r['s76'] for r in clean):.0f}, K-wait {statistics.median(r['kwait'] for r in clean):.0f}; span median {statistics.median(r['span'] for r in clean):.1f} us; first-wave CTA lifetime median {statistics.median(r['life_fw'] for r in clean):.2f} us; Fmed {statistics.median(r['Fmed'] for r in clean):.2f} us")

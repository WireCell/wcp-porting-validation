#!/usr/bin/env python3
"""doc pr/11 sec 3/4: sequential (PR_JOBS=1) latency arm vs the 20-way census.

Joins the 30-event sequential table to the census rows for the same events and
prints, per event, wall/core/RSS in both arms.  The census arm is
throughput-under-load; only this arm carries a latency claim.
"""
import csv
import sys

SEQ = sys.argv[1]
CENSUS = sys.argv[2]


def load(path, key_sample=None):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if key_sample and r['sample'] not in key_sample:
                continue
            out[(r['run'], r['subrun'], r['event'])] = r
    return out


def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


seq = load(SEQ)
cen = load(CENSUS)

GROUPS = {'seq_mcp1k': 'mcp1k data', 'seq_nuecc': 'nueCC48'}
rows = []
print(f"{'evt':>8} {'sample':<10} {'seq_wall':>9} {'cen_wall':>9} {'seq_core':>9} "
      f"{'cen_core':>9} {'seq_RSS_GB':>11} {'cen_RSS_GB':>11}")
for k, r in sorted(seq.items(), key=lambda kv: -(f(kv[1], 'core_s') or 0)):
    c = cen.get(k)
    sw, sc = f(r, 'wall_s'), f(r, 'core_s')
    cw, cc = (f(c, 'wall_s'), f(c, 'core_s')) if c else (None, None)
    sr = (f(r, 'maxrss_kb') or 0) / 1048576.0
    cr = (f(c, 'maxrss_kb') or 0) / 1048576.0 if c else 0
    rows.append((r['sample'], sw, cw, sc, cc, sr, cr))
    fmt = lambda v: f"{v:>9.2f}" if v is not None else f"{'-':>9}"
    print(f"{k[2]:>8} {r['sample']:<10} {fmt(sw)} {fmt(cw)} {fmt(sc)} {fmt(cc)} "
          f"{sr:>11.3f} {cr:>11.3f}")


def summarise(tag, sel):
    sub = [x for x in rows if sel(x)]
    if not sub:
        return
    sw = sorted(x[1] for x in sub if x[1] is not None)
    cw = sorted(x[2] for x in sub if x[2] is not None)
    sr = sorted(x[5] for x in sub if x[5])
    print(f"\n  {tag} (n={len(sub)})")
    if sw:
        print(f"    seq wall   : min={sw[0]:.2f} p50={sw[len(sw)//2]:.2f} max={sw[-1]:.2f} s")
    if cw:
        print(f"    census wall: min={cw[0]:.2f} p50={cw[len(cw)//2]:.2f} max={cw[-1]:.2f} s")
    if sw and cw:
        ratios = sorted(x[2] / x[1] for x in sub
                        if x[1] and x[2] and x[1] > 0)
        if ratios:
            print(f"    census/seq wall ratio: min={ratios[0]:.2f} "
                  f"p50={ratios[len(ratios)//2]:.2f} max={ratios[-1]:.2f}")
    if sr:
        print(f"    seq peak RSS: min={sr[0]:.3f} p50={sr[len(sr)//2]:.3f} max={sr[-1]:.3f} GB")


print("\n===== summary =====")
summarise('mcp1k heaviest+median (20)', lambda x: x[0] == 'seq_mcp1k')
summarise('nueCC48 heaviest (10)', lambda x: x[0] == 'seq_nuecc')
summarise('ALL 30', lambda x: True)

#!/usr/bin/env python3
"""doc pr/11: turn the merged 1071-row census into the numbers doc sections 2-4 quote.

Reads docs/pr/11_scores-table.tsv (written by pr_scores_table.py --merge).
Prints, per sample and overall: label breakdown, score distributions on the
nu_evaluated subset, wall/core/cw_ratio percentiles, peak-RSS percentiles.
Also emits the event lists Phase 4 needs (heaviest / median data events).
"""
import csv
import statistics as st
import sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else 'docs/pr/11_scores-table.tsv'

rows = []
with open(PATH) as fh:
    for r in csv.DictReader(fh, delimiter='\t'):
        rows.append(r)


def fnum(r, k):
    v = r.get(k, '')
    if v in ('', 'nan', 'None', '-'):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def pct(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]


def describe(vals, unit='', nd=2):
    if not vals:
        return 'n=0'
    return (f"n={len(vals)} min={min(vals):.{nd}f} p50={pct(vals,50):.{nd}f} "
            f"p90={pct(vals,90):.{nd}f} p99={pct(vals,99):.{nd}f} "
            f"max={max(vals):.{nd}f} mean={st.mean(vals):.{nd}f}{unit}")


by_sample = defaultdict(list)
for r in rows:
    by_sample[r['sample']].append(r)

ORDER = ['mcp1k', 'nuecc48', 'r1qlmc', 'r2mc']
samples = [s for s in ORDER if s in by_sample] + \
          [s for s in by_sample if s not in ORDER]

print(f"===== doc pr/11 census: {len(rows)} events =====\n")

# ---------- section 2: labels + scores ----------
print("--- label breakdown (per sample) ---")
labels = sorted({r['event_label'] for r in rows})
hdr = f"{'sample':<10} {'n':>5} " + ' '.join(f"{l:>15}" for l in labels)
print(hdr)
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    c = Counter(r['event_label'] for r in rs)
    line = f"{s:<10} {len(rs):>5} " + ' '.join(
        f"{c.get(l,0):>7} ({100.0*c.get(l,0)/len(rs):>4.1f}%)" for l in labels)
    print(line)

print("\n--- nu_evaluated (scores meaningful) ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    n = sum(1 for r in rs if r['nu_evaluated'] == '1')
    print(f"{s:<10} {n:>5}/{len(rs):<5} ({100.0*n/len(rs):.1f}%)")

print("\n--- score distributions on nu_evaluated==1 ---")
for key in ('numu_score', 'nue_score', 'cosmict_10_score', 'kine_reco_Enu_MeV'):
    print(f"\n  [{key}]")
    for s in samples + ['ALL']:
        rs = rows if s == 'ALL' else by_sample[s]
        vals = [fnum(r, key) for r in rs if r['nu_evaluated'] == '1']
        vals = [v for v in vals if v is not None]
        print(f"    {s:<10} {describe(vals, nd=3)}")

print("\n--- nue_score saturation (|s| >= 4.30) ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    vals = [fnum(r, 'nue_score') for r in rs if r['nu_evaluated'] == '1']
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"    {s:<10} n=0")
        continue
    hi = sum(1 for v in vals if v >= 4.30)
    lo = sum(1 for v in vals if v <= -4.30)
    print(f"    {s:<10} n={len(vals):<5} at +clamp={hi:<4} at -clamp={lo:<4} "
          f"({100.0*(hi+lo)/len(vals):.1f}% clamped)")

print("\n--- cosmict_score (expected structurally 0) ---")
vals = [fnum(r, 'cosmict_score') for r in rows]
vals = [v for v in vals if v is not None]
nz = [v for v in vals if v != 0.0]
print(f"    n={len(vals)} nonzero={len(nz)}")

print("\n--- cosmic-side flags on nu_evaluated==1 ---")
for key in ('cosmic_flag', 'cosmict_flag'):
    c = Counter(r[key] for r in rows if r['nu_evaluated'] == '1')
    print(f"    {key}: {dict(sorted(c.items()))}")

# ---------- section 3: runtime ----------
print("\n\n--- wall_s (WCT Timer total, 20-way concurrency = THROUGHPUT not latency) ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    vals = [v for v in (fnum(r, 'wall_s') for r in rs) if v is not None]
    print(f"    {s:<10} {describe(vals)}")

print("\n--- core_s ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    vals = [v for v in (fnum(r, 'core_s') for r in rs) if v is not None]
    print(f"    {s:<10} {describe(vals)}")

print("\n--- cw_ratio = core_s/wall_s (DL engagement discriminator, flag < 1.3) ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    vals = []
    for r in rs:
        w, c = fnum(r, 'wall_s'), fnum(r, 'core_s')
        if w and c and w > 0:
            vals.append(c / w)
    lowv = sum(1 for v in vals if v < 1.3)
    print(f"    {s:<10} {describe(vals, nd=3)}  below1.3={lowv}")

# ---------- section 4: memory ----------
print("\n\n--- maxrss_kb -> GB (per-process peak, concurrency independent) ---")
for s in samples + ['ALL']:
    rs = rows if s == 'ALL' else by_sample[s]
    vals = [v / 1048576.0 for v in (fnum(r, 'maxrss_kb') for r in rs)
            if v is not None]
    print(f"    {s:<10} {describe(vals, unit=' GB', nd=3)}")

# ---------- phase 4 event picks ----------
print("\n\n--- Phase 4 picks (sequential latency arm) ---")
data = [r for r in by_sample.get('mcp1k', [])
        if fnum(r, 'wall_s') is not None]
data.sort(key=lambda r: fnum(r, 'wall_s'))
heaviest = data[-10:][::-1]
mid = len(data) // 2
median10 = data[mid - 5:mid + 5]
nue = sorted(by_sample.get('nuecc48', []),
             key=lambda r: -(fnum(r, 'wall_s') or 0))[:10]


def show(tag, rs):
    print(f"  {tag}:")
    for r in rs:
        print(f"    {r['sample']:<8} evt {r['event']:<8} run {r['run']}/{r['subrun']:<4} "
              f"wall={fnum(r,'wall_s'):>7.2f}s core={fnum(r,'core_s'):>7.2f}s "
              f"rss={fnum(r,'maxrss_kb')/1048576.0:.2f}GB  {r['event_label']}")
    print(f"    -> EVENTS: {' '.join(r['event'] for r in rs)}")


show('10 heaviest mcp1k', heaviest)
show('10 median mcp1k', median10)
show('10 heaviest nuecc48', nue)

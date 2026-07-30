#!/usr/bin/env python3
"""Per-event beam-window detail for the 48-event nueCC-candidate sample.

n=48, so name every event rather than only quoting percentages.
"""
import glob, os, sys
from collections import Counter

def read_tsv(path):
    with open(path, errors='replace') as f:
        rows = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    if not rows:
        return []
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:] if len(r) == len(head)]

root = sys.argv[1]
tsvs = sorted(glob.glob(os.path.join(root, 'nusel_evt*', 'nusel-evt*.tsv')))
print(f"tables: {len(tsvs)}")
lab = Counter(); ev_class = Counter()
nobundle_ev = []; noinbeam_ev = []; cosmic_rows = []
print(f"\n{'run':>6} {'event':>8}  {'in-beam labels (len_cm, fc, t0_us)':<60}")
for p in tsvs:
    evid = os.path.basename(p)[len('nusel-evt'):-len('.tsv')]
    rows = read_tsv(p)
    run = rows[0]['run'] if rows else '?'
    inbeam = [r for r in rows if r['in_beam'] == '1' and r['label'] != 'no-bundle']
    nb = [r for r in rows if r['in_beam'] == '1' and r['label'] == 'no-bundle']
    desc = []
    for r in inbeam:
        lab[r['label']] += 1
        desc.append(f"{r['label']}({r['len_main_cm']}cm fc={r['fc']} t={r['flash_time_us']})")
        if r['label'] in ('TGM', 'STM', 'LM'):
            cosmic_rows.append((run, evid, r['main_id'], r['label'], r['len_main_cm'],
                                r['fc'], r['flash_time_us'], r['stmfit'], r['lm']))
    if not inbeam:
        noinbeam_ev.append((run, evid, len(nb)))
        desc.append(f"-- NO in-beam bundle ({len(nb)} in-beam no-bundle flash row(s))")
    if nb:
        nobundle_ev.append((run, evid, len(nb)))
    L = set(r['label'] for r in inbeam)
    if not inbeam:
        ev_class['no-in-beam-bundle'] += 1
    elif L & {'TGM', 'LM'} and L & {'STM', 'nu-candidate'}:
        ev_class['mixed (cosmic + keepable)'] += 1
    elif L <= {'TGM'}:
        ev_class['TGM only'] += 1
    elif L <= {'LM'}:
        ev_class['LM only'] += 1
    elif L <= {'TGM', 'LM'}:
        ev_class['TGM+LM'] += 1
    elif 'STM' in L:
        ev_class['STM (no TGM/LM)'] += 1
    else:
        ev_class['all nu-candidate'] += 1
    print(f"{run:>6} {evid:>8}  " + "  ".join(desc))

tot = sum(lab.values())
print(f"\n-- per IN-BEAM BUNDLE (n={tot})")
for k in ('TGM', 'STM', 'LM', 'nu-candidate'):
    print(f"   {k:<14} {lab[k]:4d}   {100.0*lab[k]/tot if tot else 0:6.2f} %")
cos = lab['TGM'] + lab['STM'] + lab['LM']
print(f"   {'cosmic total':<14} {cos:4d}   {100.0*cos/tot if tot else 0:6.2f} %")

print(f"\n-- per EVENT (n={len(tsvs)})")
for k, n in sorted(ev_class.items(), key=lambda kv: -kv[1]):
    print(f"   {k:<28} {n:3d}   {100.0*n/len(tsvs):6.2f} %")

print(f"\n-- events with NO in-beam bundle ({len(noinbeam_ev)}): "
      + ", ".join(f"{r}/{e}(nb={n})" for r, e, n in noinbeam_ev))
print(f"\n-- every cosmic-tagged in-beam bundle ({len(cosmic_rows)}):")
print(f"   {'run':>6} {'event':>8} {'main':>5} {'label':<5} {'len_cm':>8} {'fc':>3} {'t0_us':>9} {'stmfit':<12} {'lm':>3}")
for r in cosmic_rows:
    print("   " + " ".join(f"{v:>{w}}" for v, w in zip(r, (6, 8, 5, 5, 8, 3, 9, 12, 3))))

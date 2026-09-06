#!/usr/bin/env python3
"""doc sbnd_xin/pr/144 -- classify the OFF->ON movers and pick the Bee hand-scan set.

Sits on top of pr142_campaign_ab.py: that tool decides WHAT moved and writes the
movers TSV; this one applies doc 144's own severity classes, reports the census,
and picks a stratified subset for the owner to judge in Bee.

The classes are fixed in the doc-144 plan BEFORE the census is read, so the
selection rule cannot be tuned to the answer:

  S  selection   event_label change, nu_evaluated flip, or a BDT working-point
                 crossing (numu>0.9, nue>7.0 / >4.30103 / >0.7)
  V  vertex      |vertex move| > 5 cm
  E  energy      |dEnu| > 200 MeV
  N  pathology   kine_reco_Enu NaN appearing, or rc != 0

N is the one that decides whether the kine guard was needed: on PDVD, turning
excl_t0_frame on took NaN Enu from 10 to 72.  The ON arm here carries the guard,
so a nonzero "NaN only in OFF" count means the guard FIRED and the pair is not
equivalent to frame-only; "NaN in neither" means it never fired and the two are
the same measurement (doc pdvd/45 sec 11 measured it inert with the frame off).

Usage:
  pr144_pick_movers.py --movers docs/pr/pr144-movers.tsv \\
      --a products/d144off/*.tsv --b products/d144on/*.tsv \\
      [--n 12] [--pick-tsv docs/pr/pr144-beepick.tsv]
"""
import argparse, csv, math, os, sys
from collections import defaultdict

NUMU_SEL, NUE_UB, NUE_CLAMP, NUE_LOOSE = 0.9, 7.0, 4.30103, 0.7
VTX_CM, ENU_MEV = 5.0, 200.0

def f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None

def isnan(x):
    v = f(x)
    return v is not None and math.isnan(v)

def load(paths):
    rows = {}
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                if r.get('event') in (None, 'event'):
                    continue
                rows[(r['sample'], r['event'])] = r
    return rows

def degenerate(r):
    """doc 85: kine_reco_Enu exactly 0.0 AND vertex exactly (0,0,0) -- an unmerge
    shard the tagger selected, KineInfo never filled.  Carries no reconstruction."""
    e = f(r.get('kine_reco_Enu_MeV'))
    xs = [f(r.get(k)) for k in ('nu_x_cm', 'nu_y_cm', 'nu_z_cm')]
    return e == 0.0 and all(v == 0.0 for v in xs if v is not None) and all(v is not None for v in xs)

def crossed(a, b, thr):
    if a is None or b is None:
        return False
    return (a > thr) != (b > thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movers', required=True)
    ap.add_argument('--a', nargs='+', required=True)
    ap.add_argument('--b', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--pick-tsv')
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    print(f"score rows: A {len(A)}  B {len(B)}  common {len(common)}  "
          f"only-A {len(set(A)-set(B))}  only-B {len(set(B)-set(A))}")

    # ---- N: pathology census, over the WHOLE population (not just movers) ----
    nan_a = [k for k in common if isnan(A[k].get('kine_reco_Enu_MeV'))]
    nan_b = [k for k in common if isnan(B[k].get('kine_reco_Enu_MeV'))]
    rc_bad = [k for k in common if (A[k].get('rc') or '0') not in ('0', '') or
                                    (B[k].get('rc') or '0') not in ('0', '')]
    print(f"\n[N] NaN kine_reco_Enu: OFF {len(nan_a)}  ON {len(nan_b)}  "
          f"OFF-only {len(set(nan_a)-set(nan_b))}  ON-only {len(set(nan_b)-set(nan_a))}")
    print(f"[N] rc != 0 on either arm: {len(rc_bad)}")
    if set(nan_a) - set(nan_b):
        print("    guard FIRED -- frame+guard is NOT equivalent to frame-only on:",
              ' '.join(f"{s}/{e}" for s, e in sorted(set(nan_a) - set(nan_b))[:20]))
    elif not nan_a and not nan_b:
        print("    guard never fired and no NaN exists on either arm ->"
              " frame+guard == frame-only on this population.")

    # ---- S / V / E from the movers table, re-derived from the score tables ----
    mov = [r for r in csv.DictReader(open(a.movers), delimiter='\t')
           if r.get('event') not in (None, 'event')]
    cls = defaultdict(list)
    detail = {}
    for r in mov:
        k = (r['sample'], r['event'])
        if k not in A or k not in B:
            continue
        ra, rb = A[k], B[k]
        if degenerate(ra) or degenerate(rb):
            continue
        s = []
        if ra.get('event_label') != rb.get('event_label'):
            s.append('label')
        if ra.get('nu_evaluated') != rb.get('nu_evaluated'):
            s.append('eval')
        for nm, col, thr in (('numu>0.9', 'numu_score', NUMU_SEL),
                             ('nue>7.0', 'nue_score', NUE_UB),
                             ('nue>4.3', 'nue_score', NUE_CLAMP),
                             ('nue>0.7', 'nue_score', NUE_LOOSE)):
            if crossed(f(ra.get(col)), f(rb.get(col)), thr):
                s.append(nm)
        v = f(r.get('vtx_move_cm'))
        d = f(r.get('denu'))
        C = set()
        if s: C.add('S')
        if v is not None and v > VTX_CM: C.add('V')
        if d is not None and abs(d) > ENU_MEV: C.add('E')
        if isnan(ra.get('kine_reco_Enu_MeV')) != isnan(rb.get('kine_reco_Enu_MeV')): C.add('N')
        if not C:
            continue
        detail[k] = dict(sample=r['sample'], event=r['event'], run=r.get('run'),
                         subrun=r.get('subrun'), classes=''.join(sorted(C)),
                         sflags=','.join(s), vtx=v, denu=d,
                         enu_a=f(ra.get('kine_reco_Enu_MeV')), enu_b=f(rb.get('kine_reco_Enu_MeV')),
                         numu_a=f(ra.get('numu_score')), numu_b=f(rb.get('numu_score')),
                         nue_a=f(ra.get('nue_score')), nue_b=f(rb.get('nue_score')))
        for c in C:
            cls[c].append(k)

    print(f"\nmovers past the doc-144 thresholds: {len(detail)} of {len(mov)} "
          f"pr142-level movers, population {len(common)}")
    for c in 'SVEN':
        print(f"  [{c}] {len(cls[c]):5d}")

    # ---- the stratified Bee pick ----
    pick, seen = [], set()
    def take(keys, why):
        for k in keys:
            if k in seen or len(pick) >= a.n:
                continue
            seen.add(k); d = dict(detail[k]); d['why'] = why; pick.append(d)
    take(sorted(cls['N']), 'pathology')
    take(sorted(cls['S']), 'selection')
    take(sorted(cls['V'], key=lambda k: -(detail[k]['vtx'] or 0)), 'largest vertex move')
    take(sorted(cls['E'], key=lambda k: -abs(detail[k]['denu'] or 0)), 'largest |dEnu|')

    print(f"\nBee pick ({len(pick)} of {a.n} requested):")
    hdr = f"{'#':>2} {'sample':9} {'event':>8} {'cls':5} {'why':20} {'vtx_cm':>7} {'dEnu':>9}  evidence"
    print(hdr); print('-' * len(hdr))
    for i, d in enumerate(pick):
        ev = []
        if d['denu'] is not None and d['enu_a'] is not None:
            ev.append(f"Enu {d['enu_a']:.1f}->{d['enu_b']:.1f} MeV")
        if d['vtx']:
            ev.append(f"vertex {d['vtx']:.1f} cm")
        if d['sflags']:
            ev.append(d['sflags'])
        print(f"{i:>2} {d['sample']:9} {d['event']:>8} {d['classes']:5} {d['why']:20} "
              f"{(d['vtx'] or 0):7.1f} {(d['denu'] or 0):9.1f}  {'; '.join(ev)}")

    if a.pick_tsv:
        with open(a.pick_tsv, 'w') as fh:
            w = csv.DictWriter(fh, delimiter='\t', extrasaction='ignore',
                               fieldnames=['sample', 'run', 'subrun', 'event', 'classes',
                                           'why', 'sflags', 'vtx', 'denu', 'enu_a', 'enu_b',
                                           'numu_a', 'numu_b', 'nue_a', 'nue_b'])
            w.writeheader()
            for d in pick:
                w.writerow(d)
        print(f"\nwrote {a.pick_tsv}")
        print("Bee event list (index order): " + ' '.join(d['event'] for d in pick))

if __name__ == '__main__':
    main()

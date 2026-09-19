#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 2 -- select the "missing particle near the neutrino candidate" components from the
census TSVs, print the class x sample tables, the cut sweeps, and the ranked exhibit list for the scan.

Selection (the doc's default; every cut is a flag so the sweep table can vary it):
  size        ncell >= --min-cell (12), Q >= --min-q (2e4 e), wext >= --min-ext (6) or sext >= --min-ext
  geometry    not a stripe: NOT (wext > 60 and sext < 3) and NOT (sext > 40 and wext <= 2); not a hot channel:
              NOT (f_topwire > 0.8 and sext >= 6) and qmax <= 40 x floor (floor = 0.5 x median active-cell charge)
  near        d_bundle <= --dmax (25 cells; the (wire, slice) metric is ~isotropic: 0.3 cm pitch vs 0.32 cm/slice)
  coincident  a same-(apa) component of the same class family (level) in >= 1 OTHER plane whose slice range overlaps
              this one's by >= --ovl (0.5) of the shorter range, itself passing size + near
  charge      Qfrac = Q / Qbundle_plane >= --min-qfrac (0.03) -- the bundle's own L0 charge in that plane
Classes: THRESH (below the slicing threshold), NOBLOB-SLICE (slice has no blob in any branch), NOBLOB-INSLICE
(the slice has blobs elsewhere, none here), BUNDLE (imaged, not in the candidate).  A component passing everything
is one FLAG; an event is flagged per class if it carries >= 1 flag of that class.

Usage: d113_summary.py [--census docs/113_figs/census] [--out docs/113_figs/113_census_summary.txt]
                       [--exhibits docs/113_figs/113_exhibits.tsv] [--dmax 25] [--ovl 0.5] ...
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C

CLASSES = ('THRESH', 'NOBLOB-SLICE', 'NOBLOB-INSLICE', 'BUNDLE')
FAMILY = {'THRESH': 'img', 'NOBLOB-SLICE': 'img', 'NOBLOB-INSLICE': 'img', 'BUNDLE': 'pr'}


def load(census):
    rows = []
    for f in sorted(glob.glob(f'{census}/*.tsv')):
        for r in csv.DictReader(open(f), delimiter='\t'):
            for k in ('event', 'level', 'apa', 'plane', 'comp', 'ncell', 'wmin', 'wmax', 'smin', 'smax', 'wext', 'sext', 'has_bundle'):
                r[k] = int(r[k])
            for k in ('Q', 'cw', 'cs', 'f_act', 'f_noblobslice', 'f_zsum', 'nplane_act', 'd_bundle', 'd_scope', 'Qbundle_plane', 'floor', 'pca_rms',
                      'qmax', 'f_topwire', 'f_topslice', 'f_cov4', 'fq_cov4'):
                r[k] = float(r[k])
            r['nwire_q'] = int(r['nwire_q'])
            rows.append(r)
    evs = []
    for f in sorted(glob.glob(f'{census}/*.events.json')):
        evs.extend(json.load(open(f)))
    return rows, evs


def select(rows, a):
    # index by (sample, event, apa, family) for the coincidence test
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['sample'], r['event'], r['apa'], FAMILY[r['cls']])].append(r)
    flags = []
    for r in rows:
        if not passes_basic(r, a):
            continue
        partners = [p for p in by[(r['sample'], r['event'], r['apa'], FAMILY[r['cls']])]
                    if p['plane'] != r['plane'] and passes_basic(p, a) and overlap(r, p) >= a.ovl]
        if not partners:
            continue
        r = dict(r); r['npartner_planes'] = len({p['plane'] for p in partners})
        r['Qfrac'] = r['Q'] / r['Qbundle_plane'] if r['Qbundle_plane'] > 0 else float('nan')
        if not (r['Qfrac'] >= a.min_qfrac):
            continue
        flags.append(r)
    return flags


def passes_basic(r, a):
    if r['ncell'] < a.min_cell or r['Q'] < a.min_q:
        return False
    if max(r['wext'], r['sext']) < a.min_ext:
        return False
    if (r['wext'] > 60 and r['sext'] < 3) or (r['sext'] > 40 and r['wext'] <= 2):
        return False
    # hot / sticky channel: one wire holds > 80 % of the charge, or a cell carries > 20 x the plane's typical
    # active-cell charge (floor = 0.5 x median); ionisation never does either
    if (r['f_topwire'] > 0.8 and r['sext'] >= 6) or r['qmax'] > 40.0 * r['floor']:
        return False
    if not (r['has_bundle'] and r['d_bundle'] <= a.dmax):
        return False
    return True


def overlap(r, p):
    lo, hi = max(r['smin'], p['smin']), min(r['smax'], p['smax'])
    if hi < lo:
        return 0.0
    return (hi - lo + 1) / max(1, min(r['sext'], p['sext']))


def table(flags, evs, a, log):
    nev = collections.Counter(e['sample'] for e in evs if 'error' not in e)
    nev_b = collections.Counter(e['sample'] for e in evs if 'error' not in e and e.get('has_bundle'))
    log(f'events per sample: {dict(nev)}; with a PR candidate: {dict(nev_b)}; errors: {sum(1 for e in evs if "error" in e)}')
    log('')
    log('class x sample: flagged EVENTS (>= 1 passing component of the class) / events with a candidate, and the median Qfrac of the flags')
    hdr = f'{"class":16s}' + ''.join(f'{s:>22s}' for s in C.SAMPLES) + f'{"all":>22s}'
    log(hdr)
    for cls in CLASSES:
        line = f'{cls:16s}'; tot_e = tot_n = 0; qf_all = []
        for s in C.SAMPLES:
            fl = [f for f in flags if f['sample'] == s and f['cls'] == cls]
            e = len({f['event'] for f in fl}); n = nev_b.get(s, 0)
            qf = [f['Qfrac'] for f in fl]
            line += f'{e:>6d}/{n:<5d} {100.0*e/max(1,n):5.1f}% {np.median(qf) if qf else 0:5.3f}'
            tot_e += e; tot_n += n; qf_all += qf
        line += f'{tot_e:>6d}/{tot_n:<5d} {100.0*tot_e/max(1,tot_n):5.1f}% {np.median(qf_all) if qf_all else 0:5.3f}'
        log(line)
    fl_img = [f for f in flags if FAMILY[f['cls']] == 'img']
    e_img = len({(f['sample'], f['event']) for f in fl_img})
    log(f'any imaging class (THRESH | NOBLOB-*): {e_img} events / {sum(nev_b.values())} = {100.0*e_img/max(1,sum(nev_b.values())):.1f} %')
    log('')
    # charge-weighted view: per flagged event, the largest imaging-class Qfrac
    per_ev = collections.defaultdict(float)
    for f in fl_img:
        k = (f['sample'], f['event']); per_ev[k] = max(per_ev[k], f['Qfrac'])
    v = np.array(list(per_ev.values()))
    if len(v):
        log(f'largest imaging-class Qfrac per flagged event: median {np.median(v):.3f}, p90 {np.percentile(v,90):.3f}, max {v.max():.3f}; '
            f'>= 0.10: {(v>=0.10).sum()} events, >= 0.05: {(v>=0.05).sum()} events')
    # how much of the flagged imaging-class charge the PR candidate holds anyway (its re-tiling from ctpc)
    if fl_img:
        fq = np.array([f['fq_cov4'] for f in fl_img])
        lost = [f for f in fl_img if f['fq_cov4'] < 0.5]
        e_lost = len({(f['sample'], f['event']) for f in lost})
        log(f'PR re-tiling recovery of the flagged imaging-class components: share of charge in T_proj_data median {np.median(fq):.2f}; '
            f'components with < 50 % recovered: {len(lost)} / {len(fl_img)} in {e_lost} events (the energy-relevant loss)')
    # sub-readings for the NOBLOB-SLICE class: how many planes carried activity in those slices (2 = tiling bail-out signature)
    for cls in ('NOBLOB-SLICE', 'NOBLOB-INSLICE', 'THRESH'):
        fl = [f for f in flags if f['cls'] == cls]
        if not fl:
            continue
        npl = collections.Counter(int(round(f['nplane_act'])) for f in fl)
        zs = np.array([f['f_zsum'] for f in fl])
        log(f'{cls}: {len(fl)} flags; planes with activity in their slices (median per component): {dict(sorted(npl.items()))}; '
            f'zero-summary cell fraction: median {np.median(zs):.2f}, share of flags with > 0.5: {(zs>0.5).mean():.2f}')
    log('')


def sweep(rows, evs, a, log):
    nb = sum(1 for e in evs if 'error' not in e and e.get('has_bundle'))
    log('cut sweep (imaging-class flagged events / events with a candidate):')
    base = dict(dmax=a.dmax, ovl=a.ovl, min_qfrac=a.min_qfrac, min_cell=a.min_cell, min_q=a.min_q, min_ext=a.min_ext)
    for name, vals in (('dmax', (12, 25, 40)), ('ovl', (0.3, 0.5, 0.7)), ('min_qfrac', (0.01, 0.03, 0.10)), ('min_cell', (8, 12, 24))):
        line = f'  {name:10s}'
        for v in vals:
            b = argparse.Namespace(**{**base, name: v})
            fl = select(rows, b)
            e = len({(f['sample'], f['event']) for f in fl if FAMILY[f['cls']] == 'img'})
            line += f'  {v}: {e}/{nb} ({100.0*e/max(1,nb):.1f}%)'
        log(line)
    log('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', default=f'{C.SX}/docs/113_figs/census')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_census_summary.txt')
    ap.add_argument('--exhibits', default=f'{C.SX}/docs/113_figs/113_exhibits.tsv')
    ap.add_argument('--flags', default=f'{C.SX}/docs/113_figs/113_flags.tsv')
    ap.add_argument('--dmax', type=float, default=25.0)
    ap.add_argument('--ovl', type=float, default=0.5)
    ap.add_argument('--min-qfrac', dest='min_qfrac', type=float, default=0.03)
    ap.add_argument('--min-cell', dest='min_cell', type=int, default=12)
    ap.add_argument('--min-q', dest='min_q', type=float, default=2e4)
    ap.add_argument('--min-ext', dest='min_ext', type=int, default=6)
    ap.add_argument('--top', type=int, default=60)
    a = ap.parse_args()
    rows, evs = load(a.census)
    lines = []
    def log(s):
        print(s); lines.append(s)
    log(f'census rows {len(rows)}; cuts: dmax {a.dmax} ovl {a.ovl} min_qfrac {a.min_qfrac} min_cell {a.min_cell} min_q {a.min_q:g} min_ext {a.min_ext}')
    flags = select(rows, a)
    table(flags, evs, a, log)
    sweep(rows, evs, a, log)
    # flags TSV
    keys = ['sample', 'event', 'cls', 'apa', 'plane', 'comp', 'ncell', 'Q', 'Qfrac', 'wmin', 'wmax', 'smin', 'smax', 'wext', 'sext', 'cw', 'cs',
            'f_act', 'f_noblobslice', 'f_zsum', 'nplane_act', 'd_bundle', 'd_scope', 'Qbundle_plane', 'pca_rms', 'npartner_planes',
            'qmax', 'f_topwire', 'f_topslice', 'nwire_q', 'floor', 'f_cov4', 'fq_cov4']
    with open(a.flags, 'w') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore'); w.writeheader()
        for r in sorted(flags, key=lambda r: (r['sample'], r['event'], r['cls'], -r['Q'])):
            w.writerow(r)
    # exhibits: per (sample, event, cls) the largest-Q flag; ranked by Q; imaging classes first, top N
    best = {}
    for r in flags:
        k = (r['sample'], r['event'], r['cls'])
        if k not in best or r['Q'] > best[k]['Q']:
            best[k] = r
    ex = sorted(best.values(), key=lambda r: (FAMILY[r['cls']] != 'img', -r['Q']))
    with open(a.exhibits, 'w') as f:
        w = csv.DictWriter(f, fieldnames=['rank'] + keys, delimiter='\t', extrasaction='ignore'); w.writeheader()
        for i, r in enumerate(ex[:a.top]):
            w.writerow({'rank': i, **r})
    log(f'flags {len(flags)} -> {a.flags}; exhibits {min(a.top, len(ex))} -> {a.exhibits}')
    log('top exhibits (rank sample event class apa plane ncell Q Qfrac wext sext f_act f_zsum nplane d_bundle):')
    for i, r in enumerate(ex[:25]):
        log(f'  {i:>3} {r["sample"]:8s} {r["event"]:>7d} {r["cls"]:15s} {r["apa"]} {r["plane"]} {r["ncell"]:>5d} {r["Q"]:>10.3g} {r["Qfrac"]:6.3f} '
            f'{r["wext"]:>4d} {r["sext"]:>4d} {r["f_act"]:.2f} {r["f_zsum"]:.2f} {r["nplane_act"]:.0f} {r["d_bundle"]:.1f}')
    open(a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

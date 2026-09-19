#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 6 -- select and tabulate the PR-stage losses from d113_pr_uncover.py's output.

UNHELD groups (the candidate's own image points no particle holds): flagged when npts >= --min-pts (40), Qfrac >=
--min-qfrac (0.03), ext >= --min-ext (5 cm) and dvtx <= --dvtx (15 cm; pr/96's calibrated prong cut), with the
straightness rms reported (track-like <= 0.8 cm).  NEARBY clusters (not in the bundle, within --near of the
candidate): flagged when cl_npts >= --min-pts and dvtx <= --dvtx-near (30 cm), split by class (COMPANION / OTHER /
UNMATCHED).  Prints class x sample tables, sweeps, and the ranked exhibit list.

Usage: d113_pr_summary.py [--census docs/113_figs/pr_census] [--out ...] [--exhibits ...] [--flags ...]
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C

NUM = ('npts', 'Q', 'Qfrac', 'ext', 'rms', 'dvtx', 'dheld', 'cx', 'cy', 'cz', 'cl_npts', 'dcand', 'n_cand_pts', 'Qcand', 'f_absorbed', 'dfit', 'dfit_max', 'dfit_p90', 'f_far12')


def load(census):
    rows, evs = [], []
    for f in sorted(glob.glob(f'{census}/*.tsv')):
        for r in csv.DictReader(open(f), delimiter='\t'):
            for k in NUM:
                r[k] = float(r[k]) if r[k] not in ('', 'nan') else float('nan')
            r['event'] = int(r['event']); rows.append(r)
    for f in sorted(glob.glob(f'{census}/*.events.json')):
        evs += json.load(open(f))
    return rows, evs


def select(rows, a):
    fl = []
    for r in rows:
        if r['kind'] == 'UNHELD':
            if r['npts'] >= a.min_pts and r['Qfrac'] >= a.min_qfrac and r['ext'] >= a.min_ext and (np.isnan(r['dvtx']) or r['dvtx'] <= a.dvtx):
                # halo veto: a group that never gets farther than --halo cm from a fitted trajectory is the image band
                # of a fitted track (its energy is a range / dQ/dx energy; the band is not a particle)
                if not np.isnan(r['dfit_p90']) and r['dfit_p90'] < a.halo:
                    continue
                r = dict(r); r['cls'] = 'UNHELD-TRACK' if r['rms'] <= a.rms else 'UNHELD-WIDE'; fl.append(r)
        else:
            if r['cl_npts'] >= a.min_pts and (np.isnan(r['dvtx']) or r['dvtx'] <= a.dvtx_near) and r['Qfrac'] >= a.min_qfrac:
                r = dict(r); r['cls'] = 'NEARBY-' + r['cl_class']; fl.append(r)
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', default=f'{C.SX}/docs/113_figs/pr_census')
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_pr_summary.txt')
    ap.add_argument('--flags', default=f'{C.SX}/docs/113_figs/113_pr_flags.tsv')
    ap.add_argument('--exhibits', default=f'{C.SX}/docs/113_figs/113_pr_exhibits.tsv')
    ap.add_argument('--min-pts', dest='min_pts', type=float, default=40)
    ap.add_argument('--min-qfrac', dest='min_qfrac', type=float, default=0.03)
    ap.add_argument('--min-ext', dest='min_ext', type=float, default=5.0)
    ap.add_argument('--rms', type=float, default=0.8)
    ap.add_argument('--dvtx', type=float, default=15.0)
    ap.add_argument('--dvtx-near', dest='dvtx_near', type=float, default=30.0)
    ap.add_argument('--top', type=int, default=60)
    ap.add_argument('--halo', type=float, default=12.0, help='UNHELD group must reach this far (p90 of point-to-fit distance, cm) from every fitted trajectory')
    a = ap.parse_args()
    rows, evs = load(a.census)
    lines = []
    log = lambda s: (print(s), lines.append(s))
    nev = collections.Counter(e['sample'] for e in evs if e.get('has'))
    log(f'events with a candidate and layers: {dict(nev)} (total {sum(nev.values())}); rows {len(rows)}; cuts: min_pts {a.min_pts:g} min_qfrac {a.min_qfrac} min_ext {a.min_ext} rms {a.rms} dvtx {a.dvtx} dvtx_near {a.dvtx_near}')
    # event-level unheld fraction
    fu = np.array([e.get('f_unheld_q', np.nan) for e in evs if e.get('has')], float)
    log(f'unheld charge fraction of the candidate (points > 3 cm from every held/fitted point): median {np.nanmedian(fu):.3f}, p90 {np.nanpercentile(fu,90):.3f}, > 0.10 in {(fu>0.10).sum()} events, > 0.25 in {(fu>0.25).sum()}')
    fl = select(rows, a)
    log('\nclass x sample: flagged events / events with a candidate (median Qfrac of the flags)')
    classes = ['UNHELD-TRACK', 'UNHELD-WIDE', 'NEARBY-COMPANION', 'NEARBY-OTHER', 'NEARBY-UNMATCHED']
    hdr = f'{"class":18s}' + ''.join(f'{s:>22s}' for s in C.SAMPLES) + f'{"all":>22s}'; log(hdr)
    for cls in classes:
        line = f'{cls:18s}'; te = tn = 0; qa = []
        for s in C.SAMPLES:
            f = [x for x in fl if x['sample'] == s and x['cls'] == cls]; e = len({x['event'] for x in f}); n = nev.get(s, 0)
            qf = [x['Qfrac'] for x in f]; line += f'{e:>6d}/{n:<5d} {100*e/max(1,n):5.1f}% {np.median(qf) if qf else 0:5.3f}'; te += e; tn += n; qa += qf
        line += f'{te:>6d}/{tn:<5d} {100*te/max(1,tn):5.1f}% {np.median(qa) if qa else 0:5.3f}'; log(line)
    for cls in classes[:2]:
        f = [x for x in fl if x['cls'] == cls]
        if f:
            fa = np.array([x['f_absorbed'] for x in f])
            log(f'{cls}: {len(f)} groups; npts median {np.median([x["npts"] for x in f]):.0f}; ext median {np.median([x["ext"] for x in f]):.1f} cm; dvtx median {np.nanmedian([x["dvtx"] for x in f]):.1f} cm; Qfrac >= 0.10: {sum(1 for x in f if x["Qfrac"]>=0.1)}; p90 dist to fit median {np.nanmedian([x["dfit_p90"] for x in f]):.1f} cm; '
                f'absorbed into a track association (>= 50 % of points within 3 cm of a track-associated point): {(fa>=0.5).sum()}, held by nothing (< 10 %): {(fa<0.1).sum()}')
    log('\nsweep (UNHELD-* flagged events): ' + '  '.join(f'dvtx {d}: {len({(x["sample"],x["event"]) for x in select(rows, argparse.Namespace(**{**vars(a), "dvtx": d})) if x["cls"].startswith("UNHELD")})}' for d in (10, 15, 25, 1e9))
        + '  |  ' + '  '.join(f'halo {h}: {len({(x["sample"],x["event"]) for x in select(rows, argparse.Namespace(**{**vars(a), "halo": h})) if x["cls"].startswith("UNHELD")})}' for h in (0, 8, 12, 20)))
    with open(a.flags, 'w') as f:
        keys = ['sample', 'event', 'cls', 'kind', 'gid', 'cluster', 'cl_class', 'cl_gid', 'npts', 'cl_npts', 'Q', 'Qfrac', 'ext', 'rms', 'dvtx', 'dheld', 'dcand', 'cx', 'cy', 'cz', 'n_cand_pts', 'Qcand', 'f_absorbed', 'dfit', 'dfit_max', 'dfit_p90', 'f_far12']
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore'); w.writeheader()
        for r in sorted(fl, key=lambda r: (r['sample'], r['event'], r['cls'], -r['Q'])):
            w.writerow(r)
    best = {}
    for r in fl:
        k = (r['sample'], r['event'], r['cls'])
        if k not in best or r['Q'] > best[k]['Q']:
            best[k] = r
    ex = sorted(best.values(), key=lambda r: (not r['cls'].startswith('UNHELD'), -r['Q']))
    with open(a.exhibits, 'w') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'sample', 'event', 'cls', 'kind', 'gid', 'cluster', 'cl_class', 'npts', 'cl_npts', 'Q', 'Qfrac', 'ext', 'rms', 'dvtx', 'dheld', 'dcand', 'cx', 'cy', 'cz', 'f_absorbed', 'dfit', 'dfit_max', 'dfit_p90', 'f_far12'], delimiter='\t', extrasaction='ignore'); w.writeheader()
        for i, r in enumerate(ex[:a.top]):
            w.writerow({'rank': i, **r})
    log(f'\nflags {len(fl)} -> {a.flags}; exhibits {min(a.top, len(ex))} -> {a.exhibits}')
    log('top exhibits (rank sample event class npts Q Qfrac ext rms dvtx dcand cluster):')
    for i, r in enumerate(ex[:25]):
        log(f'  {i:>3} {r["sample"]:8s} {r["event"]:>7d} {r["cls"]:17s} {r["npts"]:>5.0f} {r["Q"]:>10.3g} {r["Qfrac"]:6.3f} {r["ext"]:6.1f} {r["rms"]:5.2f} {r["dvtx"]:6.1f} {r["dcand"]:5.1f} {r["cluster"]}')
    open(a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

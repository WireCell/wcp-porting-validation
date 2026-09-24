#!/usr/bin/env python3
"""iso_baseline.py -- the isochronous baseline of the current imaging chain on the workspace
iso-track gun (wcfm/docs/01 sec 6 W5, wcfm/docs/02 sec 5).

Reads, per event work dir and anode, the imaging cluster files (numpy ClusterFileSink
schema, blob rows: desc, ident, val, unc, faceid, sliceid, start, span, u0,u1,v0,v1,w0,w1,
ncorners, ...) of three tiers:
  clusters-apa-anode<N>-ms-active   reconstructed, charge-solved survivors (val = solved charge)
  clusters-tru-anode<N>-ms-active   the SAME blobs (desc-aligned) with val = BlobDepoFill true charge
  clusters-tru0-anode<N>-ms-active  every tiled blob before any deghosting, val = true charge
and the drifted depos (sim-depos.tar.bz2, gen 0 = drifted) for the total true charge.

A blob with true charge exactly 0 is a ghost (BlobDepoFill writes {0,0} when no depo overlaps).
Per (anode, face, slice): blob counts and ghost fractions at both tiers, the maximum wire-range
width per plane (half-open bounds, end-beg), solved vs true charge.  Per event: captured
true-charge fraction, charge error, joined to the event JSON (angle, multiplicity, kind).

    python3 scripts/iso_baseline.py --out docs/02_tables work/000001_*
    python3 scripts/iso_baseline.py --quick work/000001_1_toff*     # time_offset scan summary
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from wirecell.util import ario

COLS = dict(desc=0, ident=1, val=2, unc=3, faceid=4, sliceid=5, start=6, span=7, u0=8, u1=9, v0=10, v1=11, w0=12, w1=13, ncorners=14)


def blob_rows(fname):
    """Concatenate the blob-node arrays of every cluster in a numpy cluster file."""
    arf = ario.load(fname)
    rows = [arf[k] for k in sorted(arf.keys()) if k.endswith('_bnodes')]
    if not rows:
        return np.zeros((0, 15))
    return np.concatenate([r.reshape(-1, r.shape[-1]) for r in rows], axis=0)


def face_of(faceid):
    return (int(faceid) >> 3) & 1


def anode_of(faceid):
    return int(faceid) >> 4


def depo_true_charge(fname):
    arf = ario.load(fname)
    q = 0.0
    for k in arf.keys():
        if k.startswith('depo_data_'):
            d = arf[k]; i = arf[k.replace('data', 'info')]
            if d.shape[0] != 7 and d.shape[1] == 7:
                pass
            elif d.shape[0] == 7:
                d = d.T; i = i.T
            q += float(np.abs(d[i[:, 2] == 0, 1]).sum())
    return q


def load_event(workdir, evdir):
    base = os.path.basename(workdir.rstrip('/'))
    run6, rest = base.split('_', 1)
    evt = int(rest.split('_')[0])
    ev = json.load(open(os.path.join(evdir, f'{run6}_{evt}.json')))
    theta = max(t['angle_deg'] if t['angle_deg'] is not None else -1 for t in ev['tracks'])
    depofile = os.path.join(workdir, 'sim-depos.tar.bz2')
    if not os.path.exists(depofile):
        depofile = os.path.join(os.path.dirname(workdir.rstrip('/')), f'{run6}_{evt}', 'sim-depos.tar.bz2')
    q_depo = depo_true_charge(depofile)
    anodes = sorted(int(os.path.basename(f).split('anode')[1].split('-')[0])
                    for f in glob.glob(os.path.join(workdir, 'clusters-apa-anode*-ms-active.tar.gz')))
    per_slice = []
    tot = dict(nblob0=0, nghost0=0, nblob=0, nghost=0, nsoft=0, q_reco=0.0, q_true=0.0, q_true0=0.0)
    for a in anodes:
        reco = blob_rows(os.path.join(workdir, f'clusters-apa-anode{a}-ms-active.tar.gz'))
        tru = blob_rows(os.path.join(workdir, f'clusters-tru-anode{a}-ms-active.tar.gz'))
        tru0 = blob_rows(os.path.join(workdir, f'clusters-tru0-anode{a}-ms-active.tar.gz'))
        if reco.shape[0] != tru.shape[0] or (reco.shape[0] and not np.array_equal(reco[:, COLS['desc']], tru[:, COLS['desc']])):
            sys.exit(f'{workdir} anode {a}: tru is not desc-aligned with the ms-active file ({reco.shape[0]} vs {tru.shape[0]} blobs)')
        # geometric sanity: same bounds/slice per desc
        if reco.shape[0] and not np.array_equal(reco[:, 4:14], tru[:, 4:14]):
            sys.exit(f'{workdir} anode {a}: tru/ms-active blob geometry differs')
        groups = defaultdict(lambda: dict(nblob0=0, nghost0=0, nblob=0, nghost=0, nsoft=0, q_reco=0.0, q_true=0.0, q_true0=0.0,
                                          wmax0=[0, 0, 0], wmax=[0, 0, 0], wsum=[0, 0, 0]))
        for r in tru0:
            key = (a, face_of(r[COLS['faceid']]), int(r[COLS['sliceid']]))
            g = groups[key]
            g['nblob0'] += 1
            g['nghost0'] += int(r[COLS['val']] == 0)
            g['q_true0'] += float(r[COLS['val']])
            for p, (lo, hi) in enumerate(((8, 9), (10, 11), (12, 13))):
                g['wmax0'][p] = max(g['wmax0'][p], int(r[hi] - r[lo]))
        for r, t in zip(reco, tru):
            key = (a, face_of(r[COLS['faceid']]), int(r[COLS['sliceid']]))
            g = groups[key]
            g['nblob'] += 1
            g['nghost'] += int(t[COLS['val']] == 0)
            g['nsoft'] += int(t[COLS['val']] < 0.05 * max(r[COLS['val']], 1e-9))
            g['q_reco'] += float(r[COLS['val']])
            g['q_true'] += float(t[COLS['val']])
            for p, (lo, hi) in enumerate(((8, 9), (10, 11), (12, 13))):
                w = int(r[hi] - r[lo])
                g['wmax'][p] = max(g['wmax'][p], w)
                g['wsum'][p] += w
        for (aa, face, sl), g in sorted(groups.items()):
            per_slice.append(dict(run=int(run6), event=evt, kind=ev['kind'], theta=theta, ntrack=len(ev['tracks']),
                                  anode=aa, face=face, slice=sl, **{k: (v if not isinstance(v, list) else v) for k, v in g.items()}))
            for k in tot:
                tot[k] += g[k]
    return dict(run=int(run6), event=evt, kind=ev['kind'], theta=theta, ntrack=len(ev['tracks']), anodes=anodes,
                q_depo=q_depo, **tot), per_slice


def fmt_event(e):
    cap = e['q_true'] / e['q_depo'] if e['q_depo'] else float('nan')
    cap0 = e['q_true0'] / e['q_depo'] if e['q_depo'] else float('nan')
    gf0 = e['nghost0'] / e['nblob0'] if e['nblob0'] else float('nan')
    gf = e['nghost'] / e['nblob'] if e['nblob'] else float('nan')
    err = (e['q_reco'] - e['q_true']) / e['q_true'] if e['q_true'] else float('nan')
    return cap, cap0, gf0, gf, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdirs', nargs='+')
    ap.add_argument('--events', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'events'))
    ap.add_argument('--out', default='')
    ap.add_argument('--quick', action='store_true', help='per-work-dir totals only (time_offset scan)')
    a = ap.parse_args()
    events, slices = [], []
    for wd in a.workdirs:
        e, ps = load_event(wd, a.events)
        e['workdir'] = os.path.basename(wd.rstrip('/'))
        events.append(e); slices.extend(ps)
        cap, cap0, gf0, gf, err = fmt_event(e)
        print(f"{e['workdir']:<22} {e['kind']:<6} theta {e['theta']:5.1f} ntrk {e['ntrack']} anodes {e['anodes']}  "
              f"blobs tru0/final {e['nblob0']:6d}/{e['nblob']:6d}  ghost0 {gf0:6.3f} ghost {gf:6.3f} soft {e['nsoft']/max(e['nblob'],1):6.3f}  "
              f"captured0 {cap0:6.3f} captured {cap:6.3f}  q_reco/q_true-1 {err:+7.3f}")
    if a.quick or not a.out:
        return
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'iso_baseline_events.tsv'), 'w') as f:
        keys = ['workdir', 'kind', 'theta', 'ntrack', 'anodes', 'q_depo', 'nblob0', 'nghost0', 'nblob', 'nghost', 'nsoft', 'q_true0', 'q_true', 'q_reco']
        f.write('\t'.join(keys + ['captured0', 'captured', 'ghost0', 'ghost', 'qerr']) + '\n')
        for e in events:
            cap, cap0, gf0, gf, err = fmt_event(e)
            f.write('\t'.join(str(e[k]) for k in keys) + f'\t{cap0:.4f}\t{cap:.4f}\t{gf0:.4f}\t{gf:.4f}\t{err:.4f}\n')
    with open(os.path.join(a.out, 'iso_baseline_slices.tsv'), 'w') as f:
        keys = ['run', 'event', 'kind', 'theta', 'ntrack', 'anode', 'face', 'slice', 'nblob0', 'nghost0', 'nblob', 'nghost', 'nsoft', 'q_true0', 'q_true', 'q_reco']
        f.write('\t'.join(keys + ['wmax0_u', 'wmax0_v', 'wmax0_w', 'wmax_u', 'wmax_v', 'wmax_w', 'wmean_u', 'wmean_v', 'wmean_w']) + '\n')
        for s in slices:
            wm = [s['wsum'][p] / s['nblob'] if s['nblob'] else 0 for p in range(3)]
            f.write('\t'.join(str(s[k]) for k in keys) + '\t' + '\t'.join(str(x) for x in s['wmax0'] + s['wmax']) + '\t' + '\t'.join(f'{x:.1f}' for x in wm) + '\n')
    # summary by (kind, theta, ntrack): slice-level medians/maxima over slices holding >= 1 final blob
    summ = defaultdict(list)
    for s in slices:
        summ[(s['kind'], s['theta'], s['ntrack'], s['event'])].append(s)
    lines = ['| event | kind | theta | ntrk | slices | blobs tru0 | blobs final | ghost0 | ghost | max strip U/V/W (wires) | median blobs/slice | captured | q_reco/q_true-1 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (kind, theta, ntrk, evt), ss in sorted(summ.items(), key=lambda kv: kv[0][3]):
        e = [x for x in events if x['event'] == evt][0]
        cap, cap0, gf0, gf, err = fmt_event(e)
        nb = [s['nblob'] for s in ss if s['nblob']]
        wmax = [max(s['wmax'][p] for s in ss) for p in range(3)]
        lines.append(f"| {evt} | {kind} | {theta:.0f} | {ntrk} | {len(ss)} | {e['nblob0']} | {e['nblob']} | {gf0:.3f} | {gf:.3f} | "
                     f"{wmax[0]}/{wmax[1]}/{wmax[2]} | {np.median(nb) if nb else 0:.0f} | {cap:.3f} | {err:+.3f} |")
    open(os.path.join(a.out, 'iso_baseline_summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        iso = [e for e in events if e['kind'] == 'iso']
        cos = [e for e in events if e['kind'] == 'cosmic']
        for ax, (lab, fn) in zip(axes, [('ghost fraction (tru0 tier)', lambda e: fmt_event(e)[2]),
                                         ('ghost fraction (final)', lambda e: fmt_event(e)[3]),
                                         ('blobs per event (final)', lambda e: e['nblob'])]):
            ax.scatter([e['theta'] for e in iso], [fn(e) for e in iso], c=[e['ntrack'] for e in iso], cmap='viridis', s=60, label='iso (color = n tracks)')
            for e in cos:
                ax.axhline(fn(e), ls='--', c='gray', lw=0.8)
            ax.set_xlabel('angle to anode plane (deg); dashed = cosmic controls'); ax.set_title(lab)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'iso_baseline_vs_angle.png'), dpi=110)
        # strip width distribution per plane, iso vs cosmic (final tier, per slice max)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for p, ax in enumerate(axes):
            for kind, c in (('iso', 'C0'), ('cosmic', 'C1')):
                w = [s['wmax'][p] for s in slices if s['kind'] == kind and s['nblob']]
                if w:
                    ax.hist(w, bins=40, range=(0, max(w) + 1), histtype='step', color=c, label=f'{kind} ({len(w)} slices)', density=True)
            ax.set_xlabel(f'max blob wire-range width in slice, plane {"UVW"[p]}'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'iso_baseline_strip_width.png'), dpi=110)
    except Exception as ex:  # noqa
        print('plots skipped:', ex)


if __name__ == '__main__':
    main()

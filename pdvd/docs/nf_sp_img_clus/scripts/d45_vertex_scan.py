#!/usr/bin/env python3
"""doc pdvd/45 sec 6 follow-up -- the moved main vertices, measured and hand-scanned BLIND.

For every neutrino candidate present in both arms (T_tagger rows matched on cluster_id) whose
main vertex moved by more than --min-shift cm, compute for EACH arm's vertex:
  d_charge   distance to the nearest point of its own cluster (Bee clustering layer, identical in both arms)
  d_stm_end  distance to the nearest END of the STM trajectory of that cluster (stm_fit layer; the
             single-track fit's two extreme points -- a cosmic muon's entry/exit)
  d_stm_path distance to the nearest stm_fit point at all
and write movers.tsv.  With --n N > 0 also render N blinded sheets: cluster charge (grey) in the
z-y and z-x projections with the two vertices as markers A and B, whose assignment to base/arm is
RANDOM per sheet and recorded only in key.tsv (feedback_blind_the_scan_sheet: no proxy value and
no arm name appears on the image).  --unblind labels.tsv joins hand labels (sheet, label_A,
label_B) with key.tsv and reports per arm.
Usage: d45_vertex_scan.py <base_tag> <arm_tag> [events.txt] --out DIR [--n 40] [--seed 45] [--min-shift 5]
       d45_vertex_scan.py --unblind DIR/labels.tsv --out DIR
"""
import sys, os, json, zipfile, glob, argparse, random
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('base', nargs='?'); ap.add_argument('arm', nargs='?'); ap.add_argument('events', nargs='?')
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
ap.add_argument('--seed', type=int, default=45); ap.add_argument('--min-shift', type=float, default=5.0)
ap.add_argument('--unblind')
a = ap.parse_args()
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W = os.path.join(PDVD, 'work')
os.makedirs(a.out, exist_ok=True)

if a.unblind:
    key = {}
    for l in open(os.path.join(a.out, 'key.tsv')):
        if l.startswith('sheet'): continue
        s, ev, cl, A, B = l.rstrip('\n').split('\t')[:5]
        key[s] = (ev, cl, A, B)
    from collections import Counter
    per = {'base': Counter(), 'arm': Counter()}; pairs = Counter(); rows = []
    for l in open(a.unblind):
        if l.startswith('#') or l.startswith('sheet') or not l.strip(): continue
        f = l.rstrip('\n').split('\t'); s, la, lb = f[0], f[1].strip(), f[2].strip()
        ev, cl, A, B = key[s]
        lab = {A: la, B: lb}          # A/B -> which arm
        per['base'][lab['base']] += 1; per['arm'][lab['arm']] += 1
        pairs[(lab['base'], lab['arm'])] += 1
        rows.append((s, ev, cl, lab['base'], lab['arm'], f[3] if len(f) > 3 else ''))
    print('labels per arm (END = at a track end, KINK = at a kink/branch point, MID = mid-track on charge, OFF = off the charge):')
    for k in ('base', 'arm'): print(f'  {k:5s}', dict(sorted(per[k].items())))
    print('pairs (base -> arm):'); [print(f'  {b:4s} -> {r:4s}: {n}') for (b, r), n in sorted(pairs.items(), key=lambda x: -x[1])]
    with open(os.path.join(a.out, 'unblinded.tsv'), 'w') as o:
        o.write('sheet\tevent\tcluster\tlabel_base\tlabel_arm\tnote\n')
        for r in rows: o.write('\t'.join(r) + '\n')
    print('wrote', os.path.join(a.out, 'unblinded.tsv')); sys.exit(0)

import uproot
ev_file = a.events or os.path.join(PDVD, 'stm', 'events.txt')
events = ['%06d_%s' % (int(l.split()[0]), l.split()[1]) for l in open(ev_file) if l.strip() and not l.startswith('#')]

def cands(wd):
    p = os.path.join(wd, 'tracking-pr.root')
    if not os.path.exists(p): return None
    f = uproot.open(p)
    if 'T_tagger' not in f: return {}
    t = f['T_tagger'].arrays(['cluster_id', 'nu_x', 'nu_y', 'nu_z'], library='np')
    return {int(c): np.array([x, y, z], float) for c, x, y, z in zip(t['cluster_id'], t['nu_x'], t['nu_y'], t['nu_z'])}

def layer(wd, name):
    z = zipfile.ZipFile(os.path.join(wd, 'mabc-pr.zip'))
    for n in z.namelist():
        if n.endswith(f'-{name}-global.json'):
            d = json.loads(z.read(n)); return np.array(d['x'], float), np.array(d['y'], float), np.array(d['z'], float), np.array(d['real_cluster_id'], int)
    return None

def ends(pts):
    # the stm_fit layer is written along the trajectory; take the two points farthest apart as a guard
    if len(pts) < 2: return pts
    i = np.argmax(np.linalg.norm(pts - pts[0], axis=1)); j = np.argmax(np.linalg.norm(pts - pts[i], axis=1))
    return pts[[i, j]]

def nearest(p, pts):
    return float(np.min(np.linalg.norm(pts - p, axis=1))) if len(pts) else float('nan')

movers = []
for e in events:
    b, r = os.path.join(W, f'{e}_{a.base}'), os.path.join(W, f'{e}_{a.arm}')
    cb, cr = cands(b), cands(r)
    if not cb or not cr: continue
    cl_layer = layer(b, 'clustering'); stm_layer = layer(b, 'stm_fit')
    for c in sorted(set(cb) & set(cr)):
        d = float(np.linalg.norm(cb[c] - cr[c]))
        if d <= a.min_shift: continue
        cpts = np.c_[cl_layer[0], cl_layer[1], cl_layer[2]][cl_layer[3] == c] if cl_layer else np.zeros((0, 3))
        spts = np.c_[stm_layer[0], stm_layer[1], stm_layer[2]][stm_layer[3] == c] if stm_layer else np.zeros((0, 3))
        se = ends(spts)
        row = dict(event=e, cluster=c, shift=d, ncl=len(cpts), nstm=len(spts))
        for tag, v in (('base', cb[c]), ('arm', cr[c])):
            row[f'{tag}_x'], row[f'{tag}_y'], row[f'{tag}_z'] = v
            row[f'{tag}_d_charge'] = nearest(v, cpts); row[f'{tag}_d_stm_end'] = nearest(v, se); row[f'{tag}_d_stm_path'] = nearest(v, spts)
        movers.append((row, cpts, spts))

cols = ['event', 'cluster', 'shift', 'ncl', 'nstm'] + [f'{t}_{k}' for t in ('base', 'arm') for k in ('x', 'y', 'z', 'd_charge', 'd_stm_end', 'd_stm_path')]
with open(os.path.join(a.out, 'movers.tsv'), 'w') as o:
    o.write('\t'.join(cols) + '\n')
    for row, _, _ in movers: o.write('\t'.join(f'{row[k]:.2f}' if isinstance(row[k], float) else str(row[k]) for k in cols) + '\n')
n = len(movers)
print(f'{a.base} -> {a.arm}: {n} candidates moved > {a.min_shift} cm (movers.tsv)')
if n:
    for k in ('d_charge', 'd_stm_end', 'd_stm_path'):
        vb = np.array([m[0][f'base_{k}'] for m in movers]); va = np.array([m[0][f'arm_{k}'] for m in movers])
        ok = np.isfinite(vb) & np.isfinite(va)
        print(f'  {k:10s} base median {np.median(vb[ok]):6.2f} (<=1 cm {np.mean(vb[ok] <= 1):.2f}, <=3 cm {np.mean(vb[ok] <= 3):.2f}) | '
              f'arm median {np.median(va[ok]):6.2f} (<=1 cm {np.mean(va[ok] <= 1):.2f}, <=3 cm {np.mean(va[ok] <= 3):.2f}) | n {ok.sum()}')
    vb = np.array([m[0]['base_d_stm_end'] for m in movers]); va = np.array([m[0]['arm_d_stm_end'] for m in movers]); ok = np.isfinite(vb) & np.isfinite(va)
    print(f'  at an STM end (<= 3 cm): base only {np.sum((vb <= 3) & (va > 3))}, arm only {np.sum((va <= 3) & (vb > 3))}, both {np.sum((vb <= 3) & (va <= 3))}, neither {np.sum((vb > 3) & (va > 3) & ok)}')

if a.n > 0 and n:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rng = random.Random(a.seed); idx = list(range(n)); rng.shuffle(idx); idx = idx[:a.n]
    sheets = os.path.join(a.out, 'sheets'); os.makedirs(sheets, exist_ok=True)
    with open(os.path.join(a.out, 'key.tsv'), 'w') as k:
        k.write('sheet\tevent\tcluster\tA\tB\tshift_cm\n')
        for s, i in enumerate(idx):
            row, cpts, spts = movers[i]
            flip = rng.random() < 0.5
            A, B = ('arm', 'base') if flip else ('base', 'arm')
            pa = np.array([row[f'{A}_x'], row[f'{A}_y'], row[f'{A}_z']]); pb = np.array([row[f'{B}_x'], row[f'{B}_y'], row[f'{B}_z']])
            k.write(f'{s:02d}\t{row["event"]}\t{row["cluster"]}\t{A}\t{B}\t{row["shift"]:.1f}\n')
            fig, ax = plt.subplots(2, 3, figsize=(16, 9))
            def panel(axh, ix, iy, lab, lim=None, title=''):
                axh.scatter(cpts[:, ix], cpts[:, iy], s=2, c='0.55', lw=0)
                axh.scatter([pa[ix]], [pa[iy]], s=140, marker='o', facecolors='none', edgecolors='tab:red', lw=2)
                axh.text(pa[ix], pa[iy], ' A', color='tab:red', fontsize=13, weight='bold', clip_on=True)
                axh.scatter([pb[ix]], [pb[iy]], s=140, marker='s', facecolors='none', edgecolors='tab:blue', lw=2)
                axh.text(pb[ix], pb[iy], ' B', color='tab:blue', fontsize=13, weight='bold', clip_on=True)
                axh.set_xlabel(lab[0] + ' [cm]'); axh.set_ylabel(lab[1] + ' [cm]'); axh.set_title(title)
                if lim is not None: axh.set_xlim(lim[0]); axh.set_ylim(lim[1]); axh.set_aspect('equal')
                else: axh.set_aspect('equal', adjustable='datalim')
            panel(ax[0, 0], 2, 1, ('z', 'y'), title='cluster, z-y'); panel(ax[1, 0], 2, 0, ('z', 'x'), title='cluster, z-x')
            for col, (p, nm) in enumerate(((pa, 'A'), (pb, 'B')), start=1):
                panel(ax[0, col], 2, 1, ('z', 'y'), lim=((p[2] - 25, p[2] + 25), (p[1] - 25, p[1] + 25)), title=f'zoom {nm}, z-y (+-25 cm)')
                panel(ax[1, col], 2, 0, ('z', 'x'), lim=((p[2] - 25, p[2] + 25), (p[0] - 25, p[0] + 25)), title=f'zoom {nm}, z-x (+-25 cm)')
            fig.suptitle(f'sheet {s:02d}   PDVD {row["event"]} cluster {row["cluster"]}   -- which vertex sits at a track END / KINK / MID-track / OFF charge?')
            fig.tight_layout(); fig.savefig(os.path.join(sheets, f'{s:02d}.png'), dpi=80); plt.close(fig)
    with open(os.path.join(a.out, 'labels.tsv'), 'w') as o:
        o.write('# hand labels, filled in BLIND from sheets/NN.png; one of END KINK MID OFF UNCLEAR per marker\n')
        o.write('sheet\tlabel_A\tlabel_B\tnote\n')
    print(f'  wrote {len(idx)} sheets to {sheets}, key.tsv (do not read before labelling), labels.tsv template')

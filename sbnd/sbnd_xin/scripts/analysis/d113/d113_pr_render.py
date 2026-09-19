#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 6 -- 3-D panels (x-y, x-z, y-z projections) for the PR-stage exhibits: the candidate's
stage-A image points (grey, charge-weighted), the points the PR holds (green), the item under question (red:
an UNHELD group of the candidate's own image, or a NEARBY cluster the bundle does not contain), the main vertex
(star), and every other image cluster within the window (light blue).  Class hidden on the image (sidecar json).

Usage: d113_pr_render.py [--exhibits docs/113_figs/113_pr_exhibits.tsv] [--outdir docs/113_figs/pr_panels] [--ranks 0-59]
                         [--controls N --seed S] [--blind]
"""
import argparse, csv, glob, json, os, random, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C
from d113_pr_uncover import layers, image_points, single_link, held_points, MAIN_VTX_Q


def render_item(x, outdir, blind):
    s, ev = x['sample'], x['event']
    d = f'{C.SX}/work-{s}-pr150s0/pr_evt{ev}'
    B = C.load_bundle(d, ev); L = layers(f'{d}/mabc-pr.zip')
    pt = C.PCTree(f'{d}/pctree-pr-evt{ev}.tar.gz', ev)
    P, q, ident = image_points(pt, ev); ok = np.abs(P[:, 0]) < 1e4
    inb = np.isin(ident, list(B['union'])) & ok
    held = held_points(L, 'fit+shower')
    vtx = None
    if 'vertices-global' in L:
        v = L['vertices-global']; m = np.asarray(v['q'], float) == MAIN_VTX_Q
        if m.any():
            vtx = np.c_[v['x'], v['y'], v['z']][m][0]
    # the item's points
    if x['kind'] == 'UNHELD':
        Pc = P[inb]
        dh, _ = cKDTree(held).query(Pc) if len(held) else (np.full(len(Pc), 1e9), None)
        unh = dh > 3.0
        lab, k = single_link(Pc[unh], 2.0) if unh.sum() >= 3 else (np.zeros(0, int), 0)
        item = Pc[unh][lab == int(x['gid'])] if k else np.zeros((0, 3))
    elif x['kind'] == 'NEARBY':
        item = P[(ident == int(x['cluster'])) & ok]
    else:   # control: a held region of the candidate
        Pc = P[inb]; c = np.array([x['cx'], x['cy'], x['cz']])
        item = Pc[np.linalg.norm(Pc - c, axis=1) < 8.0]
    if len(item) == 0:
        return None
    c = item.mean(axis=0); R = max(40.0, 1.3 * np.max(np.linalg.norm(item - c, axis=1)) + 15.0)
    win = lambda X: X[np.all(np.abs(X - c) < R, axis=1)] if len(X) else X
    cand = win(P[inb]); qcand = q[inb][np.all(np.abs(P[inb] - c) < R, axis=1)] if inb.any() else np.zeros(0)
    others = win(P[~inb & ok]); heldw = win(held)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j, lab) in zip(axes, ((0, 1, 'x (drift) [cm] vs y'), (0, 2, 'x vs z'), (2, 1, 'z vs y'))):
        if len(others): ax.scatter(others[:, i], others[:, j], s=2, c='#9ecae1', alpha=0.6, linewidths=0, label='other image clusters')
        if len(cand): ax.scatter(cand[:, i], cand[:, j], s=3 + 6 * (qcand / max(1.0, np.percentile(qcand, 95))), c='0.35', alpha=0.7, linewidths=0, label='candidate image')
        if len(heldw): ax.scatter(heldw[:, i], heldw[:, j], s=4, c='tab:green', alpha=0.6, linewidths=0, label='held by a particle')
        ax.scatter(item[:, i], item[:, j], s=10, facecolors='none', edgecolors='red', linewidths=0.6, label='item')
        if vtx is not None and np.all(np.abs(vtx - c) < R): ax.scatter([vtx[i]], [vtx[j]], marker='*', s=160, c='gold', edgecolors='k', zorder=5, label='main vertex')
        ax.set_xlim(c[i] - R, c[i] + R); ax.set_ylim(c[j] - R, c[j] + R); ax.set_aspect('equal'); ax.set_title(lab)
    axes[0].legend(loc='upper left', fontsize=8, markerscale=2)
    tag = f'{s}-{ev}-p{int(x["rank"]):03d}'
    fig.suptitle(f'item {tag}' if blind else f'{tag} {x["cls"]} Q={float(x["Q"]):.3g} Qfrac={float(x["Qfrac"]):.3f} dvtx={float(x["dvtx"]):.1f}')
    fig.tight_layout(); out = f'{outdir}/{tag}.png'; fig.savefig(out, dpi=100); plt.close(fig)
    json.dump({k: x[k] for k in x}, open(f'{outdir}/{tag}.json', 'w'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exhibits', default=f'{C.SX}/docs/113_figs/113_pr_exhibits.tsv')
    ap.add_argument('--outdir', default=f'{C.SX}/docs/113_figs/pr_panels')
    ap.add_argument('--ranks', default=None)
    ap.add_argument('--controls', type=int, default=0)
    ap.add_argument('--seed', type=int, default=113)
    ap.add_argument('--blind', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ex = list(csv.DictReader(open(a.exhibits), delimiter='\t'))
    for r in ex:
        r['rank'] = int(r['rank']); r['event'] = int(r['event'])
    if a.controls:
        rng = random.Random(a.seed)
        flagged = {(r['sample'], r['event']) for r in ex}
        evs = []
        for f in glob.glob(f'{C.SX}/docs/113_figs/pr_census/*.events.json'):
            evs += [e for e in json.load(open(f)) if e.get('has') and (e['sample'], e['event']) not in flagged and e.get('n_held', 0) > 200]
        rng.shuffle(evs); made = 0
        for e in evs:
            if made >= a.controls:
                break
            d = f'{C.SX}/work-{e["sample"]}-pr150s0/pr_evt{e["event"]}'; L = layers(f'{d}/mabc-pr.zip')
            st = L.get('track_fit-global')          # controls sit ON a fitted trajectory (held by construction)
            if not st or len(st.get('x', [])) < 20:
                continue
            i = rng.randrange(10, len(st['x']) - 10)   # not a trajectory end
            x = dict(rank=900 + made, sample=e['sample'], event=e['event'], cls='CONTROL', kind='CONTROL', gid=-1, cluster=-1, Q=0.0, Qfrac=0.0, dvtx=0.0,
                     cx=st['x'][i], cy=st['y'][i], cz=st['z'][i])
            if render_item(x, a.outdir, a.blind):
                made += 1
        print(made, 'control panels'); return
    if a.ranks:
        lo, hi = [int(v) for v in a.ranks.split('-')]; ex = [r for r in ex if lo <= r['rank'] <= hi]
    n = 0
    for x in ex:
        if render_item(x, a.outdir, a.blind):
            n += 1
    print(n, 'panels')


if __name__ == '__main__':
    main()

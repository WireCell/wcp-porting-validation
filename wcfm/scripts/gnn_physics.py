#!/usr/bin/env python3
"""gnn_physics.py -- the doc 02 physics metrics of a keep/drop decision on the sub-blob graphs (wcfm/docs/05 sec 8).

For the legacy chain (kept <=> present in clusters-apa with val > 0) and for each GNN run (mean logit over
its seeds, P(real) >= threshold), per population and per event:

  charge recall     = sum q_true(kept) / sum q_true(all)                  (doc 02 "captured charge")
  track-cell recall = fraction of cells with q_true >= Q kept              (Q = --track-q, 1000 e)
  ghost fraction    = fraction of kept cells with q_true == 0
  ghost charge      = sum of the measured (packed) W-plane charge of kept ghost cells / that of all kept cells
  tail kept         = fraction of the 0 < q_true < Q "diffusion tail" cells kept

and y-z views (cell centres from the cluster-file corners) of chosen anode-events: legacy vs GNN keep set,
track cells / ghosts / tail cells.

    python3 scripts/gnn_physics.py --graphs /home/xqian/tmp/wcfm-gnn/graphs --out /home/xqian/tmp/wcfm-gnn/physics \
        --run charge=/home/xqian/tmp/wcfm-gnn/abl_charge --run fm=/home/xqian/tmp/wcfm-gnn/abl_fm --plot 1:10,9:9,17:all
"""
import argparse
import glob
import io
import os
import tarfile
from collections import defaultdict

import numpy as np

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(gdir):
    G = {}
    for fn in sorted(glob.glob(os.path.join(gdir, 'graph-*.npz'))):
        z = np.load(fn)
        G[os.path.basename(fn)[6:-4]] = dict(event=int(z['event']), anode=int(z['anode']), kind=str(z['kind']),
                                             q=z['b_qtrue'].astype(np.float64), kept=z['b_kept'].astype(bool),
                                             wsum=np.expm1(z['b_charge'][:, 10].astype(np.float64)), slice=z['b_slice'])
    return G


def run_scores(G, rdir):
    files = sorted(glob.glob(os.path.join(rdir, 'scores_*_s*.npz')))
    S = [np.load(f) for f in files]
    return {k: 1.0 / (1.0 + np.exp(np.mean([s[k] for s in S], axis=0))) for k in G}, len(files)


def metrics(G, keep_of, sel, Q):
    q, kept, ws = [], [], []
    for k, g in G.items():
        if not sel(g):
            continue
        q.append(g['q']); kept.append(keep_of(k, g)); ws.append(g['wsum'])
    if not q:
        return None
    q = np.concatenate(q); kept = np.concatenate(kept); ws = np.concatenate(ws)
    ghost = q == 0; track = q >= Q; tail = (q > 0) & (q < Q)
    return dict(cells=int(len(q)), kept=int(kept.sum()), kept_frac=kept.mean(),
                charge_recall=q[kept].sum() / q.sum(), track_recall=kept[track].mean() if track.any() else float('nan'),
                ghost_frac=ghost[kept].mean() if kept.any() else float('nan'),
                ghost_charge=ws[kept & ghost].sum() / max(ws[kept].sum(), 1e-9), tail_kept=kept[tail].mean() if tail.any() else float('nan'),
                track_cells=int(track.sum()))


def fmt(m):
    return (f"{m['kept']:,} ({m['kept_frac'] * 100:.1f} %) | {m['charge_recall']:.4f} | {m['track_recall']:.3f} | "
            f"{m['ghost_frac']:.3f} | {m['ghost_charge']:.3f} | {m['tail_kept']:.3f}")


def centres(event, anode, sub):
    fn = os.path.join(WCFM_DIR, 'work', f'{1:06d}_{event}{sub}', f'clusters-tru0-anode{anode}-ms-active.tar.gz')
    t = tarfile.open(fn)
    name = [n for n in t.getnames() if n.endswith('bnodes.npy')][0]
    b = np.load(io.BytesIO(t.extractfile(name).read()))
    nc = b[:, 14].astype(int); yz = np.zeros((len(b), 2))
    for i in range(len(b)):
        c = b[i, 15:15 + 2 * nc[i]].reshape(-1, 2); yz[i] = c.mean(axis=0)
    return yz / 10.0   # cm


def plot(G, rules, event, anode_spec, Q, sub, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [k for k, g in G.items() if g['event'] == event and (anode_spec == 'all' or g['anode'] == int(anode_spec))]
    for k in keys:
        g = G[k]; yz = centres(g['event'], g['anode'], sub)
        q = g['q']; ghost = q == 0; track = q >= Q; tail = (q > 0) & (q < Q)
        fig, axes = plt.subplots(1, len(rules) + 1, figsize=(5 * (len(rules) + 1), 5), sharex=True, sharey=True)
        ax = axes[0]
        ax.scatter(yz[ghost, 1], yz[ghost, 0], s=1, c='lightgray', label=f'ghost ({ghost.sum():,})')
        ax.scatter(yz[tail, 1], yz[tail, 0], s=1, c='tab:orange', label=f'tail 0<q<{Q:.0f} ({tail.sum():,})')
        ax.scatter(yz[track, 1], yz[track, 0], s=2, c='tab:blue', label=f'track q>={Q:.0f} ({track.sum():,})')
        ax.set_title(f'{k}: all sub-blobs (truth)'); ax.legend(fontsize=7, markerscale=5); ax.set_xlabel('z [cm]'); ax.set_ylabel('y [cm]')
        for ax, (name, keep_of) in zip(axes[1:], rules.items()):
            keep = keep_of(k, g)
            ax.scatter(yz[keep & ghost, 1], yz[keep & ghost, 0], s=2, c='tab:red', label=f'kept ghost ({(keep & ghost).sum():,})')
            ax.scatter(yz[keep & tail, 1], yz[keep & tail, 0], s=1, c='tab:orange', label=f'kept tail ({(keep & tail).sum():,})')
            ax.scatter(yz[keep & track, 1], yz[keep & track, 0], s=2, c='tab:blue', label=f'kept track ({(keep & track).sum():,})')
            ax.scatter(yz[~keep & track, 1], yz[~keep & track, 0], s=4, c='black', marker='x', label=f'dropped track ({(~keep & track).sum():,})')
            ax.set_title(f'{name}: keep set'); ax.legend(fontsize=7, markerscale=4); ax.set_xlabel('z [cm]')
        fig.tight_layout(); fn = os.path.join(out, f'view-{k}.png'); fig.savefig(fn, dpi=110); plt.close(fig)
        print(f'[physics] wrote {fn}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--run', action='append', default=[], help='name=dir with scores_*_s*.npz (mean logit over seeds)')
    ap.add_argument('--threshold', type=float, default=0.5, help='keep <=> P(real) >= threshold')
    ap.add_argument('--track-q', type=float, default=1000.0)
    ap.add_argument('--sub', default='_sub4')
    ap.add_argument('--plot', default='', help='event:anode|all, comma separated')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = load(a.graphs); Q = a.track_q
    rules = {'legacy chain': lambda k, g: g['kept']}
    for spec in a.run:
        name, rdir = spec.split('=', 1)
        P, nseed = run_scores(G, rdir)
        rules[f'{name} GNN P(real)>={a.threshold} ({nseed} seeds)'] = (lambda P: (lambda k, g: P[k] >= a.threshold))(P)
    pops = {'all': lambda g: True, 'iso': lambda g: g['kind'] == 'iso', 'cosmic': lambda g: g['kind'] == 'cosmic'}
    lines = ['| rule | population | cells kept | charge recall | track-cell recall | ghost fraction of kept | ghost share of kept measured charge | tail cells kept |',
             '|---|---|---|---|---|---|---|---|']
    for rname, keep_of in rules.items():
        for pname, sel in pops.items():
            m = metrics(G, keep_of, sel, Q)
            lines.append(f'| {rname} | {pname} ({m["cells"]:,} cells, {m["track_cells"]:,} track cells) | {fmt(m)} |')
    lines += ['', f'Per event (doc 02 events 1-10 first; Q = {Q:.0f} e):', '',
              '| event | kind | cells | track cells | ' + ' | '.join(f'{r}: charge recall / ghost frac' for r in rules) + ' |',
              '|---|---|---|---|' + '---|' * len(rules)]
    events = sorted({g['event'] for g in G.values()})
    for e in events:
        sel = lambda g, e=e: g['event'] == e
        cells = [fmt_e for fmt_e in [metrics(G, keep_of, sel, Q) for keep_of in rules.values()]]
        kind = next(g['kind'] for g in G.values() if g['event'] == e)
        lines.append(f'| {e} | {kind} | {cells[0]["cells"]:,} | {cells[0]["track_cells"]:,} | ' +
                     ' | '.join(f'{m["charge_recall"]:.3f} / {m["ghost_frac"]:.3f}' for m in cells) + ' |')
    with open(os.path.join(a.out, 'physics.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:2 + 3 * len(rules)]))
    for spec in [s for s in a.plot.split(',') if s]:
        ev, an = spec.split(':'); plot(G, rules, int(ev), an, Q, a.sub, a.out)
    print(f'[physics] wrote {a.out}/physics.md')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""iso_slab_probe.py -- what an exactly isochronous (0 deg) slab looks like to the doc 05 chain (wcfm/docs/06).

Read-only over the fixed-truth sub-blob graphs of gnn_dataset.py (--graphs), the charge-GNN scores of gnn_train.py
(--scores), and the _sub4f cluster archives (tru0 for cell centres, apa for the legacy solver's measure nodes).
For every single-track 0 deg event (and the single 0.5 deg events as the contrast) it writes, per anode-event:

  slab_census.md         slices holding true charge, real cells and candidates per slice, wires under the track, fan-in
  consistency.md         AP of a three-view charge-consistency statistic (real vs ghost) against the prior, wire-charge CV
  fm_similarity.md       cosine similarity of the FM wire descriptor vs channel separation along the track, on vs off track
  gnn_by_end_distance.md the charge GNN's P(real) and recall by distance from the track ends; legacy and GNN keep counts
  line_rule.md           a truth-free line-hypothesis rule (the hexagon chord covering all three channel intervals) vs legacy and GNN in the same slab
  measures.md            the legacy solver's measure nodes per slab slice (apa archive: one row per connected channel group)
  view-<key>.png         y-z of the busiest slab slice: truth / GNN keep / line rule

    python3 scripts/iso_slab_probe.py --out docs/06_tables
"""
import argparse
import glob
import io
import json
import math
import os
import sys
import tarfile

import numpy as np

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WCFM_DIR, 'scripts'))
from gnn_physics import centres  # noqa: E402


def ap(score, y):
    o = np.argsort(-score); y = y[o].astype(float)
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def load_events():
    ev = {}
    for fn in glob.glob(os.path.join(WCFM_DIR, 'events', '000001_*.json')):
        e = json.load(open(fn)); ev[int(e['event'])] = e
    return ev


def single_iso(ev, theta):
    return sorted(k for k, e in ev.items() if e['kind'] == 'iso' and len(e['tracks']) == 1 and e['tracks'][0]['angle_deg'] == theta)


WIRE_ANGLE_DEG = (54.0, 126.0, 90.0)   # FD-HD 1x2x6 wire directions in the y-z plane, from +z (tru0 wnodes tail/head)


def pitch_coords(yz):
    """the three pitch coordinates (cm) of y-z points: p = y cos(theta) - z sin(theta), perpendicular to each wire direction.
    Channel numbers are NOT a position coordinate on the wrapped U/V planes (a 4 m track on anode 9 of event 27 'covers'
    all 800 U channels), so the rule works in position space."""
    th = np.radians(WIRE_ANGLE_DEG)
    return np.stack([yz[:, 0] * np.cos(t) - yz[:, 1] * np.sin(t) for t in th], axis=1)


def components(cells, bw_src, bw_dst, nw):
    """wire-connected components of the given cells (union-find over shared wires)."""
    idx = {c: i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    m = np.isin(bw_src, cells)
    first = {}
    for b, w in zip(bw_src[m], bw_dst[m]):
        i = idx[b]
        if w in first:
            ra, rb = find(first[w]), find(i)
            if ra != rb:
                parent[ra] = rb
        else:
            first[w] = i
    roots = np.array([find(i) for i in range(len(cells))])
    return [cells[roots == r] for r in np.unique(roots)]


def line_rule(z, cand, yz, nend=3, min_cells=50):
    """truth-free: per wire-connected component the candidates fill the intersection of three wire strips (a hexagon);
    a straight track is the chord whose projections cover all three pitch intervals.  Endpoint candidates = the nend
    lowest / highest cells per pitch coordinate; the chord = the pair maximising the minimum (over planes) interval
    coverage.  Returns dist-to-chord (cm), along-chord position and chord length per cell (nan when no chord)."""
    dist = np.full(len(cand), np.nan); tpar = np.full(len(cand), np.nan); L_of = np.full(len(cand), np.nan)
    pos = {c: i for i, c in enumerate(cand)}
    pc_all = pitch_coords(yz)
    ncomp = 0
    for comp in components(cand, z['bw_src'], z['bw_dst'], len(z['w_plane'])):
        if len(comp) < min_cells:
            continue
        ii = np.array([pos[c] for c in comp]); P = yz[ii]; pc = pc_all[ii]
        width = np.maximum(pc.max(0) - pc.min(0), 1e-3)
        ends = set()
        for p in range(3):
            ends.update(np.argsort(pc[:, p])[:nend].tolist()); ends.update(np.argsort(-pc[:, p])[:nend].tolist())
        ends = sorted(ends)
        best, pair = -1.0, None
        for x in range(len(ends)):
            for y in range(x + 1, len(ends)):
                score = (np.abs(pc[ends[x]] - pc[ends[y]]) / width).min()
                if score > best:
                    best, pair = score, (ends[x], ends[y])
        if pair is None:
            continue
        a, b = P[pair[0]], P[pair[1]]; v = b - a; L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        v /= L; ncomp += 1
        X = P - a
        tpar[ii] = X @ v; dist[ii] = np.abs(X[:, 0] * v[1] - X[:, 1] * v[0]); L_of[ii] = L
    return dist, tpar, L_of, ncomp


def keep_of_rule(dist, tpar, L_of, rad):
    return np.isfinite(dist) & (dist < rad) & (tpar > -rad) & (tpar < L_of + rad)


def measures(event, anode, sub):
    fn = os.path.join(WCFM_DIR, 'work', f'{1:06d}_{event}{sub}', f'clusters-apa-anode{anode}-ms-active.tar.gz')
    if not os.path.exists(fn):
        return None
    t = tarfile.open(fn)
    names = t.getnames()

    def L(suffix):
        n = [m for m in names if m.endswith(suffix)]
        return np.load(io.BytesIO(t.extractfile(n[0]).read())) if n else None
    b, m, bm = L('bnodes.npy'), L('mnodes.npy'), L('bmedges.npy')
    if b is None or m is None or bm is None:
        return None
    return b, m, bm


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--graphs', default='/home/xqian/tmp/wcfm-gnn/graphs_f')
    ap_.add_argument('--scores', default='/home/xqian/tmp/wcfm-gnn/ablf_charge', help='dir with scores_*_s*.npz of the charge GNN')
    ap_.add_argument('--sub', default='_sub4f')
    ap_.add_argument('--out', default=os.path.join(WCFM_DIR, 'docs', '06_tables'))
    ap_.add_argument('--track-q', type=float, default=1000.0)
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = load_events()
    S = [np.load(f) for f in sorted(glob.glob(os.path.join(a.scores, 'scores_*_s*.npz')))]
    zero, half = single_iso(ev, 0.0), single_iso(ev, 0.5)
    print('single 0 deg events', zero, '; single 0.5 deg events', half)

    T = {k: [] for k in ('census', 'consistency', 'fm', 'gnn', 'line', 'meas')}
    T['census'].append('| event / anode | theta | phi_yz [deg] | length [cm] | slices with q_true | real cells per slice | candidates per slice | wires under the track U / V / W (busiest slice) | fan-in median (max) U / V / W |')
    T['census'].append('|---|---|---|---|---|---|---|---|---|')
    T['consistency'].append('| event / anode | busiest slice | candidates | real | prior | AP of the 3-view consistency statistic | wire-charge CV along the track U / V / W |')
    T['consistency'].append('|---|---|---|---|---|---|---|')
    T['fm'].append('| event / anode | plane | on-track wires | cos-sim at 1 / 5 / 20 / 100 channels | on-track vs off-track (same slice) |')
    T['fm'].append('|---|---|---|---|---|')
    T['gnn'].append('| event / anode | slab cells | real | GNN kept (real) | legacy kept (real) | mean P(real) real / ghost | recall and mean P(real) of real cells by distance from an end: <10 / 10-30 / 30-80 / >80 cm |')
    T['gnn'].append('|---|---|---|---|---|---|---|')
    T['line'].append('| event / anode | slab cells | real | rule components | line r<1.5: kept, cell recall, precision, charge recall | line r<3: kept, recall, precision, charge recall | GNN P>=0.5: recall, precision, charge recall | legacy: recall, precision, charge recall |')
    T['line'].append('|---|---|---|---|---|---|---|---|')
    T['meas'].append('| event / anode | slice | apa blobs | measures (rows of the solver) | wpid of each measure | sum of measure values |')
    T['meas'].append('|---|---|---|---|---|---|')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for theta, events in ((0.0, zero), (0.5, half)):
        for k in events:
            t = ev[k]['tracks'][0]
            d = np.array(t['head']) - np.array(t['tail']); d /= np.linalg.norm(d)
            phi = math.degrees(math.atan2(d[1], d[2]))
            for fn in sorted(glob.glob(os.path.join(a.graphs, f'graph-000001_{k}-anode*.npz'))):
                key = os.path.basename(fn)[6:-4]; anode = int(key.split('anode')[1])
                z = np.load(fn); q = z['b_qtrue'].astype(float); real = q > 0; sl = z['b_slice']
                if real.sum() < 20:
                    continue
                wp, wc, wq = z['w_plane'], z['w_channel'], z['w_q'].astype(float)
                bw_src, bw_dst = z['bw_src'], z['bw_dst']
                us, cnt = np.unique(sl[real], return_counts=True); s0 = us[np.argmax(cnt)]
                us2, cnt2 = np.unique(sl, return_counts=True); ncand = {int(s): int(cnt2[list(us2).index(s)]) for s in us}
                cand = np.where(sl == s0)[0]; r = real[cand]
                slab = np.where(np.isin(sl, us))[0]; rs = real[slab]
                label = f'{k} / {anode}'
                # --- census
                rb = cand[r]; wsel = np.unique(bw_dst[np.isin(bw_src, rb)]); deg = np.bincount(bw_dst, minlength=len(wp))
                nw = [int((wp[wsel] == p).sum()) for p in range(3)]
                fan = [f'{np.median(deg[wsel[wp[wsel] == p]]):.0f} ({deg[wsel[wp[wsel] == p]].max()})' if nw[p] else '-' for p in range(3)]
                T['census'].append(f'| {label} | {theta} | {phi:.1f} | {t["length_cm"]} | {len(us)} | ' + ' / '.join(str(c) for c in cnt) + ' | '
                                   + ' / '.join(str(ncand[int(s)]) for s in us) + f' | {nw[0]} / {nw[1]} / {nw[2]} | ' + ' / '.join(fan) + ' |')
                # --- consistency statistic
                cq = np.zeros((len(q), 3)); cn = np.zeros((len(q), 3))
                np.add.at(cq, (bw_src, wp[bw_dst]), wq[bw_dst]); np.add.at(cn, (bw_src, wp[bw_dst]), 1)
                cqm = cq / np.maximum(cn, 1)
                med = [np.median(cqm[cand][:, p][cqm[cand][:, p] > 0]) if (cqm[cand][:, p] > 0).any() else 1.0 for p in range(3)]
                lg = np.log(np.maximum(cqm[cand], 1) / np.array(med)[None, :]); stat = -lg.var(axis=1)
                cv = []
                for p in range(3):
                    qq = wq[wsel[wp[wsel] == p]]; qq = qq[qq > 0]
                    cv.append(f'{qq.std() / qq.mean():.3f}' if len(qq) > 1 else '-')
                T['consistency'].append(f'| {label} | {s0} | {len(cand)} | {r.sum()} | {r.mean():.3f} | {ap(stat, r):.3f} | ' + ' / '.join(cv) + ' |')
                # --- FM similarity
                if 'w_fm' in z:
                    fm = z['w_fm'].astype(np.float32)
                    allw = np.unique(bw_dst[np.isin(bw_src, cand)]); off = np.setdiff1d(allw, wsel)
                    for p in range(3):
                        w = wsel[wp[wsel] == p]; w = w[np.argsort(wc[w])]
                        if len(w) < 6:
                            continue
                        f = fm[w]; f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)
                        sims = [f'{(f[:-dd] * f[dd:]).sum(1).mean():.3f}' if len(w) > dd else '-' for dd in (1, 5, 20, 100)]
                        fo = fm[off[wp[off] == p]]
                        so = f'{(f @ (fo / np.maximum(np.linalg.norm(fo, axis=1, keepdims=True), 1e-6)).T).mean():.3f} ({len(fo)} wires)' if len(fo) else '- (0 wires)'
                        T['fm'].append(f'| {label} | {"UVW"[p]} | {len(w)} | ' + ' / '.join(sims) + f' | {so} |')
                # --- GNN by end distance (slab level)
                P = 1.0 / (1.0 + np.exp(np.mean([s[key] for s in S], axis=0))) if S else np.zeros(len(q))
                yz_all = centres(k, anode, a.sub)
                A = np.array(t['tail'][1:]); B = np.array(t['head'][1:]); v = (B - A) / np.linalg.norm(B - A); Ltr = np.linalg.norm(B - A)
                tp = (yz_all[slab] - A) @ v; dend = np.minimum(tp, Ltr - tp)
                keep = P[slab] >= 0.5; kept_leg = z['b_kept'][slab].astype(bool)
                bins = []
                for lo, hi in ((0, 10), (10, 30), (30, 80), (80, 1e9)):
                    m = rs & (dend >= lo) & (dend < hi)
                    bins.append(f'{(keep & m).sum() / m.sum():.2f} ({P[slab][m].mean():.2f}, n={m.sum()})' if m.sum() else '- ')
                T['gnn'].append(f'| {label} | {len(slab)} | {rs.sum()} | {keep.sum()} ({(keep & rs).sum()}) | {kept_leg.sum()} ({(kept_leg & rs).sum()}) | '
                                f'{P[slab][rs].mean():.3f} / {P[slab][~rs].mean():.3f} | ' + ' / '.join(bins) + ' |')
                # --- line rule (per slice of the slab, truth-free), evaluated on the slab
                keep_rule = {1.5: np.zeros(len(slab), bool), 3.0: np.zeros(len(slab), bool)}; ncomp = 0
                pos_slab = {c: i for i, c in enumerate(slab)}
                for s in us:
                    cs = np.where(sl == s)[0]
                    dist, tpar, L_of, nc = line_rule(z, cs, yz_all[cs]); ncomp += nc
                    for rad in keep_rule:
                        kk = keep_of_rule(dist, tpar, L_of, rad)
                        keep_rule[rad][[pos_slab[c] for c in cs[kk]]] = True

                def prf(kk):
                    return f'{(kk & rs).sum() / rs.sum():.3f}, {(kk & rs).sum() / max(kk.sum(), 1):.3f}, {q[slab][kk].sum() / q[slab].sum():.3f}'
                T['line'].append(f'| {label} | {len(slab)} | {rs.sum()} | {ncomp} | {keep_rule[1.5].sum()}, ' + prf(keep_rule[1.5]) + f' | {keep_rule[3.0].sum()}, '
                                 + prf(keep_rule[3.0]) + ' | ' + prf(keep) + ' | ' + prf(kept_leg) + ' |')
                # --- legacy measures
                mm = measures(k, anode, a.sub)
                if mm is not None:
                    b, m, bm = mm
                    for s in us:
                        bi = np.where(b[:, 5] == s)[0]; e = bm[np.isin(bm[:, 1], bi)]; mi = np.unique(e[:, 2])
                        T['meas'].append(f'| {label} | {int(s)} | {len(bi)} | {len(mi)} | ' + ', '.join(str(int(m[i, 4])) for i in mi) + f' | {m[mi, 2].sum():.3g} |')
                # --- figure of the busiest slice
                if theta == 0.0:
                    yz = yz_all[cand]; kp = P[cand] >= 0.5
                    dist, tpar, L_of, _ = line_rule(z, cand, yz); kr = keep_of_rule(dist, tpar, L_of, 1.5)
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
                    for ax, (title, kk) in zip(axes, (('truth', None), ('charge GNN P(real)>=0.5', kp), ('line rule r<1.5 cm', kr))):
                        if kk is None:
                            ax.scatter(yz[~r, 1], yz[~r, 0], s=2, c='lightgray', label=f'ghost ({(~r).sum()})')
                            ax.scatter(yz[r, 1], yz[r, 0], s=3, c='tab:blue', label=f'real ({r.sum()})')
                        else:
                            ax.scatter(yz[~kk & ~r, 1], yz[~kk & ~r, 0], s=1, c='lightgray', label='dropped ghost')
                            ax.scatter(yz[kk & ~r, 1], yz[kk & ~r, 0], s=3, c='tab:red', label=f'kept ghost ({(kk & ~r).sum()})')
                            ax.scatter(yz[kk & r, 1], yz[kk & r, 0], s=3, c='tab:blue', label=f'kept real ({(kk & r).sum()})')
                            ax.scatter(yz[~kk & r, 1], yz[~kk & r, 0], s=6, c='black', marker='x', label=f'dropped real ({(~kk & r).sum()})')
                        ax.set_title(f'{key} slice {s0}: {title}'); ax.legend(fontsize=7, markerscale=4); ax.set_xlabel('z [cm]')
                    axes[0].set_ylabel('y [cm]'); fig.tight_layout()
                    fig.savefig(os.path.join(a.out, f'view-{key}.png'), dpi=110); plt.close(fig)
                print(f'[probe] {label} theta {theta}: slab {len(slab)} cells, {rs.sum()} real; slice {s0} {len(cand)} / {r.sum()}')

    names = dict(census='slab_census.md', consistency='consistency.md', fm='fm_similarity.md', gnn='gnn_by_end_distance.md', line='line_rule.md', meas='measures.md')
    for k, fn in names.items():
        with open(os.path.join(a.out, fn), 'w') as f:
            f.write('\n'.join(T[k]) + '\n')
        print(f'[probe] wrote {a.out}/{fn}')


if __name__ == '__main__':
    main()

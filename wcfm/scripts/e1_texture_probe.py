#!/usr/bin/env python3
"""e1_texture_probe.py -- the doc 06 E1 go/no-go on real events (wcfm/docs/07).

Read-only over the fixed-truth sub-blob graphs of gnn_dataset.py (--graphs, the E1 events; --ref-graphs, the
gun events of doc 06 = reference rows and null control), the _sub4f tru0 archives (cell wire-in-plane ranges,
wire geometry) and the sim frames (packed wire charge).

Criterion (a): the doc 06 sec 3.1 three-view consistency statistic (code copied from iso_slab_probe.main): per cell
   the mean packed wire charge of its wires in each plane, normalised by the slice-plane median, scored by minus the
   variance of the three log-ratios; AP(real vs ghost) against the prior on the busiest slice of the muon slab.
   For the neutrino events the slab is the primary muon's: real cells farther than --min-dist (cm, y-z) from the
   vertex, so the hadronic vertex region does not enter.  Go = AP > 2x prior on the 0 deg slabs.
Criterion (b): the 1-D alignment probe.  q_U(u), q_V(v), q_W(w) are the packed wire charges of the slab's slices
   indexed by wire-in-plane; the truth correspondence v_true(u), w_true(u) is the straight line fitted to the real
   muon cells' (u,v,w) centres.  Two searches, both scored by the Pearson correlation of q_U with q_V (or q_W)
   resampled along the hypothesis v = a + b u:
     local  -- windows of --window wires slid along the muon (interior texture only), slope b from the truth
               (oracle direction), offset a searched over the slab's own V (W) range (= the deghosting ambiguity);
     global -- the whole u range, direction phi_yz and offset both free.
   A window / a wire is "recovered" if |v_hat - v_true| <= --tol wires (one 4-wire cell).  Go = >= 80 % of the
   length.  Chance = 2*tol+1 over the search range, reported alongside.
Context: the legacy solver's charge recall on the slab, the doc 06 line rule, wire-charge CV per plane.

    python3 scripts/e1_texture_probe.py --graphs /home/xqian/tmp/wcfm-gnn/graphs_e1 --out docs/07_tables
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
from gnn_physics import centres                                   # noqa: E402
from iso_slab_probe import ap, load_events, single_iso, pitch_coords, WIRE_ANGLE_DEG, line_rule, keep_of_rule  # noqa: E402
from crossview_probe import load_charge, COLS, LAYER              # noqa: E402

RUN = 1


def load_tru0(event, anode, sub):
    fn = os.path.join(WCFM_DIR, 'work', f'{RUN:06d}_{event}{sub}', f'clusters-tru0-anode{anode}-ms-active.tar.gz')
    t = tarfile.open(fn); names = t.getnames()
    b = np.load(io.BytesIO(t.extractfile([n for n in names if n.endswith('bnodes.npy')][0]).read()))
    w = np.load(io.BytesIO(t.extractfile([n for n in names if n.endswith('wnodes.npy')][0]).read()))
    return b, w


def consistency(z, cand, real):
    """doc 06 sec 3.1 statistic on the candidate cells `cand` (indices) -> (AP, prior)."""
    q = z['b_qtrue'].astype(float); wp = z['w_plane']; wq = z['w_q'].astype(float)
    bw_src, bw_dst = z['bw_src'], z['bw_dst']
    cq = np.zeros((len(q), 3)); cn = np.zeros((len(q), 3))
    np.add.at(cq, (bw_src, wp[bw_dst]), wq[bw_dst]); np.add.at(cn, (bw_src, wp[bw_dst]), 1)
    cqm = cq / np.maximum(cn, 1)
    med = [np.median(cqm[cand][:, p][cqm[cand][:, p] > 0]) if (cqm[cand][:, p] > 0).any() else 1.0 for p in range(3)]
    lg = np.log(np.maximum(cqm[cand], 1) / np.array(med)[None, :]); stat = -lg.var(axis=1)
    r = real[cand]
    return ap(stat, r), float(r.mean())


def wire_cv(z, cells):
    wp = z['w_plane']; wq = z['w_q'].astype(float)
    wsel = np.unique(z['bw_dst'][np.isin(z['bw_src'], cells)])
    out = []
    for p in range(3):
        qq = wq[wsel[wp[wsel] == p]]; qq = qq[qq > 0]
        out.append(qq.std() / qq.mean() if len(qq) > 1 else float('nan'))
    return out


class PlaneGeom:
    """wire-in-plane <-> channel and <-> pitch coordinate for one (anode, face), from the tru0 wnodes."""

    def __init__(self, wn, anode, face):
        self.chan = {}; self.fit = {}
        th = np.radians(WIRE_ANGLE_DEG)
        for p in range(3):
            pid = (anode << 4) | (face << 3) | LAYER[p]
            rows = wn[wn[:, 5] == pid]
            if len(rows) == 0:
                continue
            n = int(rows[:, 2].max()) + 1
            arr = np.full(n, -1, int); arr[rows[:, 2].astype(int)] = rows[:, 4].astype(int)
            self.chan[p] = arr
            mid = (rows[:, 6:9] + rows[:, 9:12]) / 2.0 / 10.0     # cm
            pc = mid[:, 1] * np.cos(th[p]) - mid[:, 2] * np.sin(th[p])
            self.fit[p] = np.polyfit(pc, rows[:, 2], 1)          # wip = c1 * pitch + c0

    def wip_of_yz(self, p, yz):
        th = np.radians(WIRE_ANGLE_DEG[p]); pc = yz[:, 0] * np.cos(th) - yz[:, 1] * np.sin(th)
        return np.polyval(self.fit[p], pc)

    def sequence(self, p, qmap, anode, slices):
        ch = self.chan[p]; out = np.zeros(len(ch))
        ok = ch >= 0
        out[ok] = qmap[ch[ok] - anode * 2560][:, slices].sum(axis=1)
        return out


def pearson_rows(T, M):
    """Pearson correlation of template T (n,) with every row of M (m, n)."""
    T = T - T.mean(); M = M - M.mean(axis=1, keepdims=True)
    den = np.linalg.norm(T) * np.linalg.norm(M, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (M @ T) / den
    r[~np.isfinite(r)] = -1.0
    return r


def align_local(qu, qv, u_lo, u_hi, b_true, a_true, v_rng, window, step, tol):
    """slide windows along u; slope known; offset a searched so that v(u_mid) in v_rng. -> (n windows, n correct, chance)"""
    n = 0; ok = 0; chance = []
    for u0 in range(int(u_lo), int(u_hi) - window + 1, step):
        uu = np.arange(u0, u0 + window); T = qu[u0:u0 + window]
        if T.max() <= 0 or T.std() == 0:
            continue
        umid = u0 + window / 2.0
        a_grid = np.arange(v_rng[0] - b_true * umid, v_rng[1] - b_true * umid + 1)
        pos = a_grid[:, None] + b_true * uu[None, :]
        M = np.interp(pos, np.arange(len(qv)), qv, left=0, right=0)
        r = pearson_rows(T, M)
        a_hat = a_grid[int(np.argmax(r))]
        n += 1; ok += int(abs((a_hat + b_true * umid) - (a_true + b_true * umid)) <= tol)
        chance.append((2 * tol + 1) / max(len(a_grid), 1))
    return n, ok, (float(np.mean(chance)) if chance else float('nan'))


def align_global(qu, qv, u_lo, u_hi, geom, pu, pv, v_rng, v_true_fn, tol, phi_step=0.5):
    """direction and offset free over the whole u range -> (fraction of u wires within tol, b_hat, best r)."""
    uu = np.arange(int(u_lo), int(u_hi)); T = qu[uu[0]:uu[-1] + 1]
    if T.std() == 0:
        return float('nan'), float('nan'), float('nan')
    best = (-2, None, None)
    P0 = np.zeros((1, 2))
    for phi in np.arange(0.0, 180.0, phi_step):
        P1 = np.array([[math.sin(math.radians(phi)), math.cos(math.radians(phi))]]) * 100.0
        du = geom.wip_of_yz(pu, P1)[0] - geom.wip_of_yz(pu, P0)[0]
        dv = geom.wip_of_yz(pv, P1)[0] - geom.wip_of_yz(pv, P0)[0]
        if abs(du) < 1e-6:
            continue
        b = dv / du
        umid = uu.mean()
        a_grid = np.arange(v_rng[0] - b * umid, v_rng[1] - b * umid + 1)
        if len(a_grid) == 0:
            continue
        pos = a_grid[:, None] + b * uu[None, :]
        M = np.interp(pos, np.arange(len(qv)), qv, left=0, right=0)
        r = pearson_rows(T, M); i = int(np.argmax(r))
        if r[i] > best[0]:
            best = (float(r[i]), float(a_grid[i]), b, phi)
    if best[1] is None:
        return float('nan'), float('nan'), float('nan')
    r, a, b, phi = best
    v_hat = a + b * uu; frac = float((np.abs(v_hat - v_true_fn(uu)) <= tol).mean())
    return frac, b, r


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--graphs', default='/home/xqian/tmp/wcfm-gnn/graphs_e1')
    ap_.add_argument('--ref-graphs', default='/home/xqian/tmp/wcfm-gnn/graphs_f', help='doc 06 gun graphs: reference rows + null control')
    ap_.add_argument('--sub', default='_sub4f')
    ap_.add_argument('--out', default=os.path.join(WCFM_DIR, 'docs', '07_tables'))
    ap_.add_argument('--min-dist', type=float, default=30.0, help='cm from the vertex (neutrino events)')
    ap_.add_argument('--window', type=int, default=32)
    ap_.add_argument('--step', type=int, default=8)
    ap_.add_argument('--tol', type=float, default=4.0)
    ap_.add_argument('--min-real', type=int, default=20)
    ap_.add_argument('--max-slab', type=int, default=4, help='slices on each side of the busiest one the slab may extend')
    ap_.add_argument('--no-figs', action='store_true')
    ap_.add_argument('--pops', default='all', help='comma list of population indices (0 gun 0, 1 gun 0.5, 2 rot 0, 3 rot 0.5, 4 natural) to run')
    ap_.add_argument('--events', default='all', help='comma list of event numbers to keep (after --pops)')
    ap_.add_argument('--phi-step', type=float, default=1.0, help='degrees, global search grid')
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = load_events()
    if not a.no_figs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    T = {k: [] for k in ('consistency', 'alignment', 'context')}
    T['consistency'].append('| stratum | event / anode | angle to anode [deg] | busiest slice | slab slices | candidates | real | prior | AP | AP / prior | wire-charge CV along the muon U / V / W |')
    T['consistency'].append('|---|---|---|---|---|---|---|---|---|---|---|')
    T['alignment'].append('| stratum | event / anode | u wires along the muon | V search range [wires] | local U-V: windows recovered (chance) | local U-W | global U-V: length within 1 cell, slope b_hat / b_true, r | global U-W |')
    T['alignment'].append('|---|---|---|---|---|---|---|---|')
    T['context'].append('| stratum | event / anode | slab cells | real | legacy kept (real): cell recall, precision, charge recall | line rule r<1.5: recall, precision, charge recall | line rule r<3 |')
    T['context'].append('|---|---|---|---|---|---|---|')

    # populations: (stratum label, event list, graphs dir, is_gun)
    e1 = lambda lo, hi: [k for k in sorted(ev) if lo <= k <= hi and os.path.isdir(os.path.join(WCFM_DIR, 'work', f'{RUN:06d}_{k}{a.sub}'))]
    pops = [('gun 0 deg (doc 06 reference / null control)', single_iso(ev, 0.0), a.ref_graphs, True),
            ('gun 0.5 deg (doc 06 reference)', single_iso(ev, 0.5), a.ref_graphs, True),
            ('numu rotated 0 deg', e1(201, 220), a.graphs, False),
            ('numu rotated 0.5 deg', e1(301, 320), a.graphs, False),
            ('numu natural', e1(101, 120), a.graphs, False)]
    if a.pops != 'all':
        pops = [pops[int(i)] for i in a.pops.split(',')]
    for stratum, events, gdir, gun in pops:
        if a.events != 'all':
            events = [k for k in events if str(k) in a.events.split(',')]
        for k in events:
            t = ev[k]['tracks'][0]
            angle = t['angle_deg']
            vtx = np.array(t['tail'][1:])
            for fn in sorted(glob.glob(os.path.join(gdir, f'graph-{RUN:06d}_{k}-anode*.npz'))):
                anode = int(os.path.basename(fn)[6:-4].split('anode')[1])
                z = np.load(fn); q = z['b_qtrue'].astype(float); real = q > 0; sl = z['b_slice']
                yz = centres(k, anode, a.sub)
                far = np.ones(len(q), bool) if gun else (np.linalg.norm(yz - vtx, axis=1) > a.min_dist)
                rf = real & far
                if rf.sum() < a.min_real:
                    continue
                us, cnt = np.unique(sl[rf], return_counts=True); s0 = us[np.argmax(cnt)]
                # the slab = the contiguous run of slices around the busiest one holding >= 10 % of its real cells
                # (an isochronous track spans 2-4 slices; a neutrino event's other particles span hundreds)
                cmap = dict(zip(us.tolist(), cnt.tolist())); run = [int(s0)]
                for step in (1, -1):
                    s = int(s0) + step
                    while abs(s - int(s0)) <= a.max_slab and cmap.get(s, 0) >= 0.1 * cmap[int(s0)]:
                        run.append(s); s += step
                us = np.array(sorted(run)); cnt = np.array([cmap[s] for s in us])
                cand = np.where((sl == s0) & far)[0]
                slab = np.where(np.isin(sl, us) & far)[0]; rs = real[slab]
                label = f'{k} / {anode}'
                # ---- (a) consistency
                A, prior = consistency(z, cand, real)
                cv = wire_cv(z, cand[real[cand]])
                T['consistency'].append(f'| {stratum} | {label} | {angle} | {int(s0)} | {len(us)} | {len(cand)} | {int(real[cand].sum())} | {prior:.3f} | {A:.3f} | {A / prior:.2f} | '
                                        + ' / '.join(f'{c:.2f}' for c in cv) + ' |')
                # ---- context: legacy + line rule on the slab
                kept_leg = z['b_kept'][slab].astype(bool)

                def prf(kk):
                    return f'{(kk & rs).sum() / rs.sum():.3f}, {(kk & rs).sum() / max(kk.sum(), 1):.3f}, {q[slab][kk].sum() / q[slab].sum():.3f}'
                keep_rule = {1.5: np.zeros(len(slab), bool), 3.0: np.zeros(len(slab), bool)}
                pos_slab = {c: i for i, c in enumerate(slab)}
                for s in us:
                    cs = np.where((sl == s) & far)[0]
                    dist, tpar, L_of, nc = line_rule(z, cs, yz[cs])
                    for rad in keep_rule:
                        kk = keep_of_rule(dist, tpar, L_of, rad)
                        keep_rule[rad][[pos_slab[c] for c in cs[kk]]] = True
                T['context'].append(f'| {stratum} | {label} | {len(slab)} | {rs.sum()} | {kept_leg.sum()} ({(kept_leg & rs).sum()}): ' + prf(kept_leg)
                                    + ' | ' + prf(keep_rule[1.5]) + ' | ' + prf(keep_rule[3.0]) + ' |')
                # ---- (b) alignment (0 deg strata only: the isochronous question)
                if angle is None or angle > 0.75:
                    continue
                bn, wn = load_tru0(k, anode, a.sub)
                assert len(bn) == len(q), (len(bn), len(q))
                faces = z['b_face'][slab[rs]]
                face = int(np.bincount(faces).argmax())
                geom = PlaneGeom(wn, anode, face)
                if any(p not in geom.chan for p in range(3)):
                    continue
                qmap = load_charge(f'{RUN:06d}_{k}', anode)
                seqs = [geom.sequence(p, qmap, anode, us.astype(int)) for p in range(3)]
                rc = slab[rs & (z['b_face'][slab] == face)]
                mid = np.stack([(bn[rc, COLS['u0']] + bn[rc, COLS['u1']] - 1) / 2.0,
                                (bn[rc, COLS['v0']] + bn[rc, COLS['v1']] - 1) / 2.0,
                                (bn[rc, COLS['w0']] + bn[rc, COLS['w1']] - 1) / 2.0], axis=1)
                u_lo, u_hi = bn[rc, COLS['u0']].min(), bn[rc, COLS['u1']].max()
                if u_hi - u_lo < a.window + a.step:
                    continue
                allc = slab[z['b_face'][slab] == face]
                cols = {1: ('v0', 'v1'), 2: ('w0', 'w1')}
                res = {}
                for p in (1, 2):
                    fit = np.polyfit(mid[:, 0], mid[:, p], 1)          # v_true(u) = fit[0] u + fit[1]
                    v_rng = (bn[allc, COLS[cols[p][0]]].min(), bn[allc, COLS[cols[p][1]]].max())
                    n, ok, ch = align_local(seqs[0], seqs[p], u_lo, u_hi, fit[0], fit[1], v_rng, a.window, a.step, a.tol)
                    frac, b_hat, r = align_global(seqs[0], seqs[p], u_lo, u_hi, geom, 0, p, v_rng,
                                                  lambda uu, f=fit: np.polyval(f, uu), a.tol, a.phi_step)
                    res[p] = (n, ok, ch, frac, b_hat, r, v_rng, fit[0])
                T['alignment'].append(f'| {stratum} | {label} | {int(u_hi - u_lo)} | {int(res[1][6][1] - res[1][6][0])} | '
                                      f'{res[1][1]}/{res[1][0]} = {res[1][1] / max(res[1][0], 1):.2f} ({res[1][2]:.3f}) | '
                                      f'{res[2][1]}/{res[2][0]} = {res[2][1] / max(res[2][0], 1):.2f} ({res[2][2]:.3f}) | '
                                      f'{res[1][3]:.2f}, {res[1][4]:.2f} / {res[1][7]:.2f}, r={res[1][5]:.2f} | '
                                      f'{res[2][3]:.2f}, {res[2][4]:.2f} / {res[2][7]:.2f}, r={res[2][5]:.2f} |')
                if not a.no_figs and not gun:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
                    uu = np.arange(int(u_lo), int(u_hi))
                    axes[0].plot(uu, seqs[0][uu], label='q_U(u)', lw=0.8)
                    for p, nm in ((1, 'V'), (2, 'W')):
                        fit = np.polyfit(mid[:, 0], mid[:, p], 1)
                        axes[0].plot(uu, np.interp(np.polyval(fit, uu), np.arange(len(seqs[p])), seqs[p]), label=f'q_{nm}({nm.lower()}_true(u))', lw=0.8)
                    axes[0].set_xlabel('U wire-in-plane'); axes[0].set_ylabel('packed charge (slab slices)'); axes[0].legend(fontsize=8)
                    axes[0].set_title(f'{stratum}: event {k} anode {anode} face {face}: the three views along the truth line')
                    cc = np.where(sl == s0)[0]; r0 = real[cc]
                    axes[1].scatter(yz[cc][~r0, 1], yz[cc][~r0, 0], s=2, c='lightgray', label=f'ghost ({(~r0).sum()})')
                    axes[1].scatter(yz[cc][r0, 1], yz[cc][r0, 0], s=3, c='tab:blue', label=f'real ({r0.sum()})')
                    axes[1].scatter([vtx[1]], [vtx[0]], marker='*', s=80, c='red', label='vertex')
                    axes[1].set_xlabel('z [cm]'); axes[1].set_ylabel('y [cm]'); axes[1].legend(fontsize=8); axes[1].set_aspect('equal')
                    fig.tight_layout(); fig.savefig(os.path.join(a.out, f'e1-{k}-anode{anode}.png'), dpi=110); plt.close(fig)
    for k, rows in T.items():
        open(os.path.join(a.out, k + '.md'), 'w').write('\n'.join(rows) + '\n')
        print(k, len(rows) - 2, 'rows')


if __name__ == '__main__':
    main()

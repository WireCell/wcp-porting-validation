#!/usr/bin/env python3
"""doc pdhd/03 -- static per-candidate panels for the CheckSTM_Michel output, so the
verdicts can be scanned WITHOUT a Bee upload (ask-first).  One PNG per STM candidate:

  top row    : whole cluster (Bee 'stm' layer, t0-corrected cm) in z-y and z-x with the
               chain roles overlaid (muon red, delta orange, Michel blue, dot magenta,
               vertices black), entry (green ^), tagger stop (green x), PR stop (green v);
               dQ/dx vs residual range of the chain with the tail/plateau windows.
  bottom row : +-ZOOM cm around the PR stop in z-y / z-x / y-x with EVERY cluster of the
               event (Bee 'clustering' layer) in grey, so unattached dots and
               continuations past the stop are visible.

Usage: d03_render_candidates.py <tag> [--out DIR] [--events 0,6,..] [--zoom 30] [--only-stm]
Report only -- nothing here changes a verdict.
"""
import sys, os, glob, json, zipfile, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot

PDHD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(PDHD, 'work')
BITS = [(1, 'no_chain'), (2, 'stop_unmatched'), (4, 'no_bragg'), (8, 'shape_flat'), (16, 'not_muon_pid'),
        (32, 'continuation'), (64, 'stop_near_boundary'), (128, 'vertex_hadron'), (256, 'short'),
        (512, 'profile_sparse'), (1024, 'plateau_off_mip'), (2048, 'stop_into_dead'), (4096, 'cluster_not_track')]
ROLE = {0: ('k', '*', 60, 'vertex'), 1: ('r', '.', 9, 'muon'), 2: ('orange', '.', 14, 'delta'),
        3: ('b', '.', 14, 'michel'), 4: ('m', 'o', 18, 'dot')}

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('--out', default=os.path.join('/home/xqian/tmp/d03_render'))
ap.add_argument('--events', default=''); ap.add_argument('--zoom', type=float, default=30.0)
ap.add_argument('--only-stm', action='store_true'); ap.add_argument('--mip', type=float, default=56000.0)
a = ap.parse_args()
out = os.path.join(a.out, a.tag); os.makedirs(out, exist_ok=True)
dirs = sorted(glob.glob(os.path.join(W, '*_' + a.tag)), key=lambda d: int(os.path.basename(d).split('_')[1]))
if a.events:
    keep = set(a.events.split(','))
    dirs = [d for d in dirs if os.path.basename(d).split('_')[1] in keep]

def bee_layer(z, name):
    n = [m for m in z.namelist() if m.endswith('-%s-global.json' % name)]
    if not n: return None
    d = json.loads(z.read(n[0]))
    return {k: np.asarray(d[k]) for k in ('x', 'y', 'z', 'q', 'cluster_id', 'real_cluster_id')}

def bits_str(b):
    return '+'.join(n for v, n in BITS if b & v) or 'STM'

index = []
for d in dirs:
    ev = os.path.basename(d)[:-len(a.tag) - 1]
    f = os.path.join(d, 'tracking-pr.root'); zp = os.path.join(d, 'mabc-pr.zip')
    if not (os.path.isfile(f) and os.path.isfile(zp)): continue
    with uproot.open(f) as rf:
        if 'T_stm_michel' not in rf: continue
        T = rf['T_stm_michel'].arrays(library='np'); P = rf['T_stm_michel_pts'].arrays(library='np')
    TG = None
    fs = os.path.join(d, 'tracking-stm.root')
    if os.path.isfile(fs):
        with uproot.open(fs) as rs:
            if 'T_rec_charge' in rs:
                TG = rs['T_rec_charge'].arrays(['x', 'y', 'z', 'q', 'rr', 'cluster_id', 'status', 'pass'], library='np')
    z = zipfile.ZipFile(zp)
    stm = bee_layer(z, 'stm'); allc = bee_layer(z, 'clustering')
    for i in range(len(T['cluster_id'])):
        r = {k: v[i] for k, v in T.items()}
        if a.only_stm and r['reject_bits'] != 0: continue
        cid = int(r['cluster_id'])
        sel = P['cluster_id'] == cid
        px, py, pz, pq, pL, prr, prole = (P[k][sel] for k in ('x', 'y', 'z', 'q', 'L', 'rr', 'role'))
        stop = np.array([r['stop_x'], r['stop_y'], r['stop_z']]); ent = np.array([r['entry_x'], r['entry_y'], r['entry_z']])
        tstop = np.array([r['tagger_stop_x'], r['tagger_stop_y'], r['tagger_stop_z']])
        fig, ax = plt.subplots(2, 3, figsize=(18, 11))
        # whole cluster
        if stm is not None:
            m = stm['cluster_id'] == cid
            cx, cy, cz = stm['x'][m], stm['y'][m], stm['z'][m]
        else:
            cx = cy = cz = np.array([])
        tg = None
        if TG is not None:
            mt = (TG['cluster_id'] == cid) & (TG['status'] == 0) & (TG['pass'] == r['pass'])
            if mt.any(): tg = {k: TG[k][mt] for k in ('x', 'y', 'z', 'q', 'rr')}
        for (axi, X, Y, xl, yl) in ((ax[0, 0], cz, cy, 'z [cm]', 'y [cm]'), (ax[0, 1], cz, cx, 'z [cm]', 'x [cm]')):
            axi.scatter(X, Y, s=2, c='0.75', label='cluster %d (%d pts)' % (cid, len(cx)))
            if tg is not None:
                axi.plot(tg['z'], tg['y'] if yl.startswith('y') else tg['x'], '-', c='c', lw=1.2, alpha=.8, label='tagger fit (%d)' % len(tg['z']), zorder=2)
            for role, (c, mk, sz, nm) in ROLE.items():
                mm = prole == role
                if mm.any():
                    XX = pz[mm]; YY = py[mm] if yl.startswith('y') else px[mm]
                    axi.scatter(XX, YY, s=sz, c=c, marker=mk, label='%s (%d)' % (nm, mm.sum()), zorder=3)
            yi = 1 if yl.startswith('y') else 0
            axi.plot(ent[2], ent[yi], 'g^', ms=12, mfc='none', mew=2, label='entry'); axi.plot(tstop[2], tstop[yi], 'gx', ms=12, mew=2, label='tagger stop')
            axi.plot(stop[2], stop[yi], 'gv', ms=12, mfc='none', mew=2, label='PR stop')
            axi.set_xlabel(xl); axi.set_ylabel(yl); axi.set_aspect('equal', adjustable='datalim'); axi.grid(alpha=.3)
        ax[0, 0].legend(fontsize=7, loc='best')
        # profile
        pr = ax[0, 2]; mu = prole == 1
        if mu.any():
            o = np.argsort(prr[mu]); pr.plot(prr[mu][o], pq[mu][o] / 1e3, 'r.-', ms=4, lw=.8, label='chain dQ/dx')
        for role in (2, 3, 4):
            mm = prole == role
            if mm.any(): pr.plot(prr[mm], pq[mm] / 1e3, ROLE[role][1], c=ROLE[role][0], ms=5, label=ROLE[role][3])
        if tg is not None:
            o = np.argsort(tg['rr']); pr.plot(tg['rr'][o], tg['q'][o] / 1e3, '-', c='c', lw=1, alpha=.7, label='tagger fit dQ (raw, per row)')
        pr.axhline(a.mip / 1e3, c='k', ls='--', lw=.8, label='mip %.0f ke/cm' % (a.mip / 1e3))
        pr.axvspan(0.5, 3, color='r', alpha=.08); pr.axvspan(20, 40, color='b', alpha=.08)
        pr.set_xlabel('residual range from PR stop [cm]'); pr.set_ylabel('dQ/dx [ke/cm]'); pr.grid(alpha=.3)
        ymax = np.percentile(pq[mu] / 1e3, 99) * 1.3 if mu.any() else 200
        pr.set_ylim(0, max(ymax, 3 * a.mip / 1e3)); pr.set_xlim(-2, max(60, (prr[mu].max() if mu.any() else 60) * 1.02))
        pr.legend(fontsize=7)
        pr.set_title('contrast %.2f (exp %.2f, tail %.0f / plat %.0f ke)  ks_mu %.3f ks_flat %.3f\ncomp_fwd gate %.0f mu %.2f p %.2f e %.2f | bwd gate %.0f mu %.2f p %.2f e %.2f' % (
            r['contrast'], r['contrast_expected'], r['tail_med'] / 1e3, r['plateau_med'] / 1e3, r['ks_mu'], r['ks_flat'],
            r['comp_fwd0'], r['comp_fwd1'], r['comp_fwd2'], r['comp_fwd3'], r['comp_bwd0'], r['comp_bwd1'], r['comp_bwd2'], r['comp_bwd3']), fontsize=8)
        # zoom
        Z = a.zoom
        if allc is not None:
            near = (np.abs(allc['x'] - stop[0]) < Z) & (np.abs(allc['y'] - stop[1]) < Z) & (np.abs(allc['z'] - stop[2]) < Z)
        for (axi, i1, i2, l1, l2) in ((ax[1, 0], 2, 1, 'z', 'y'), (ax[1, 1], 2, 0, 'z', 'x'), (ax[1, 2], 1, 0, 'y', 'x')):
            if allc is not None and near.any():
                A = np.stack([allc['x'][near], allc['y'][near], allc['z'][near]]); own = allc['cluster_id'][near] == cid
                axi.scatter(A[i1][~own], A[i2][~own], s=6, c='0.55', marker='s', label='other clusters (%d)' % (~own).sum())
                axi.scatter(A[i1][own], A[i2][own], s=4, c='0.8', label='this cluster')
            PP = np.stack([px, py, pz])
            for role, (c, mk, sz, nm) in ROLE.items():
                mm = prole == role
                if mm.any(): axi.scatter(PP[i1][mm], PP[i2][mm], s=sz + 6, c=c, marker=mk, zorder=3)
            axi.plot(tstop[i1], tstop[i2], 'gx', ms=14, mew=2); axi.plot(stop[i1], stop[i2], 'gv', ms=14, mfc='none', mew=2)
            axi.set_xlim(stop[i1] - Z, stop[i1] + Z); axi.set_ylim(stop[i2] - Z, stop[i2] + Z)
            axi.set_xlabel(l1 + ' [cm]'); axi.set_ylabel(l2 + ' [cm]'); axi.set_aspect('equal'); axi.grid(alpha=.3)
        ax[1, 0].legend(fontsize=7, loc='best')
        ttl = '%s cluster %d gid %d  verdict %s (bits %d)  muon %.1f cm / %d segs, stop_dis %.1f cm, in_fv %d | delta %d hadron %d other %d | michel %d (%.1f cm, mip %.2f, kink %.0f deg, KE %.1f MeV, conn %d) dots %d unfit %d | cont %.1f cm @ %.0f deg' % (
            ev, cid, r['gid'], bits_str(int(r['reject_bits'])), r['reject_bits'], r['muon_len'], r['n_chain_segs'], r['stop_dis'], r['in_fv'],
            r['n_delta'], r['n_body_hadron'], r['n_body_other'], r['michel_found'], r['michel_len'], r['michel_mip'], r['michel_kink_deg'], r['michel_ke_best'], r['michel_conn_type'],
            r['n_dots'], r['n_dot_clusters_unfit'], r['cont_len'], r['cont_angle_deg'])
        fig.suptitle(ttl, fontsize=9); fig.tight_layout(rect=(0, 0, 1, 0.96))
        fn = os.path.join(out, '%s_c%03d_%s.png' % (ev, cid, 'STM' if r['reject_bits'] == 0 else 'rej%d' % r['reject_bits']))
        fig.savefig(fn, dpi=80); plt.close(fig)
        index.append((ev, cid, int(r['reject_bits']), bits_str(int(r['reject_bits'])), int(r['michel_found']), fn))
with open(os.path.join(out, 'index.tsv'), 'w') as fo:
    fo.write('event\tcluster\tbits\tverdict\tmichel\tpng\n')
    for row in index: fo.write('\t'.join(str(x) for x in row) + '\n')
print('%d panels -> %s' % (len(index), out))

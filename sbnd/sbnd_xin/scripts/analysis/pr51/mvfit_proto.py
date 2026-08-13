#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/51 round 7: offline prototype of the robust vertex fit
(`mvfit_robust`) — dynamic per-leg direction windows, disagreement-gated,
multiplicity-aware prior — replayed over whole arms so the knob defaults can be
tuned before any C++ is written.

Per main (neutrino) vertex it replays MyFCN twice:
  production : per-leg PCA over (1.5, 6] cm ((0.9, 6] for chord<=3), prior
               npoints/0.43cm^2 — the shipped fit;
  robust     : for each non-shower leg with chord >= MIN_LEN, an outer annulus
               (r_in, r_out] with r_in = max(1.5, reseat+RIN_MARGIN)
               (reseat = 4 cm iff the leg's far end is > 8 cm, mirroring
               UpdateInfo) and r_out = clamp(ROUT_FRAC*chord, ROUT_MIN,
               ROUT_MAX); the leg's PCA/center are SUBSTITUTED by the outer
               window's iff the folded inner-vs-outer axis angle > ANGLE, the
               outer window has >= MIN_PTS points and anisotropy
               sqrt(l0/l1) >= MIN_ANISO.  npoints (the prior weight) stays the
               production inner count.  If >=1 leg fired and exactly 2 legs
               contribute, the prior range relaxes 0.43 -> PRIOR2 cm.

CAVEAT (same as vtx_fit_standalone.py / myfcn_offline.py): the zip trajectories
are the FINAL fits, so this is a characterization at the converged vertex, not
a bit-level replay of the in-toolkit solve.  The production replay's own move
is therefore ~0 by construction; the robust-vs-production DIFFERENCE is the
observable being prototyped.

Inputs per event (mabc-pr.zip): vertices-global (q=15000 = main vertex),
track_fit-global (= Segment::fits(), the layer MyFCN reads),
shower_track-global (per-point q flag 15000 = shower-typed segment).

Usage: mvfit_proto.py <ARM> [ARM ...] [--tsv OUT.tsv] [--evt N]
         [--angle 18] [--rin-margin 2.0] [--rout-frac 0.5] [--rout-min 9]
         [--rout-max 18] [--min-len 10] [--min-pts 5] [--min-aniso 3.0]
         [--prior 0.43] [--prior2 1.0] [--all-multiplicity] [-v]
"""
import sys, os, json, glob, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
FLOOR = 0.15 ** 2
DMIN = 1.0        # leg incident if a fit point is this close to the vertex
PREFILTER = 1.0   # m_fit_vertex_min_seg_length SBND operating point (cm)

P = dict(angle=18.0, rin_margin=2.0, rout_frac=0.5, rout_min=9.0,
         rout_max=18.0, min_len=10.0, min_pts=5, min_aniso=3.0,
         prior=0.43, prior2=1.0)

tsv = None; only_evt = None; verbose = False
args = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a == '--tsv':
        tsv = argv[i + 1]; i += 2; continue
    if a == '--evt':
        only_evt = int(argv[i + 1]); i += 2; continue
    if a == '-v':
        verbose = True; i += 1; continue
    key = a[2:].replace('-', '_')
    if a.startswith('--') and key in P:
        P[key] = float(argv[i + 1]); i += 2; continue
    if a.startswith('--'):
        i += 1; continue
    args.append(a); i += 1
if not args:
    sys.exit(__doc__)


def leg_pca(pts, V, r1, r2):
    """MyFCN::AddSegment for one leg: (r1, r2] shell, PCA about the kept point
    closest to V, eigenvalues + (0.15cm)^2 floor, descending order."""
    d = np.linalg.norm(pts - V, axis=1)
    sel = pts[(d >= r1) & (d <= r2)]
    if len(sel) < 2:
        return None
    center = sel[np.argmin(np.linalg.norm(sel - V, axis=1))]
    cov = np.zeros((3, 3))
    for p in sel:
        dd = p - center
        cov += np.outer(dd, dd)
    cov /= len(sel)
    w, vecs = np.linalg.eigh(cov)
    lam = w[::-1] + FLOOR
    ax = vecs[:, ::-1].T
    return dict(n=len(sel), lam=lam, ax=ax, center=center)


def myfcn_solve(pcas, V, rng):
    """MyFCN::FitVertex normal equations (row 0 zeroed, sqrt(l0/lk) weights,
    isotropic prior npoints/rng^2)."""
    A = np.zeros((3, 3)); b = np.zeros(3); npts = 0
    for pc in pcas:
        lam, ax, c = pc['lam'], pc['ax'], pc['center']
        npts += pc['n']
        R = np.zeros((3, 3))
        R[1] = np.sqrt(lam[0] / lam[1]) * ax[1]
        R[2] = np.sqrt(lam[0] / lam[2]) * ax[2]
        A += R.T @ R
        b += R.T @ (R @ c)
    if rng is not None:
        k = npts / rng ** 2
        A += np.eye(3) * k
        b += V * k
    return np.linalg.solve(A, b), A, npts


def fold(a, b):
    """angle between two axes folded to [0, 90] deg (eigenvector signs are
    arbitrary)."""
    return np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b))))))


leg_rows = []   # per-leg TSV rows
vtx_rows = []   # per-vertex summaries

for arm in args:
    for d in sorted(glob.glob(os.path.join(SB, arm, 'pr_evt*'))):
        evt = int(os.path.basename(d)[6:])
        if only_evt is not None and evt != only_evt:
            continue
        zp = os.path.join(d, 'mabc-pr.zip')
        if not os.path.exists(zp):
            continue
        try:
            z = zipfile.ZipFile(zp)
            vd = json.loads(z.read('data/0/0-vertices-global.json'))
            fd = json.loads(z.read('data/0/0-track_fit-global.json'))
            sd = json.loads(z.read('data/0/0-shower_track-global.json'))
        except Exception as e:
            print('  [skip] %s evt %d: %s' % (arm, evt, e), file=sys.stderr)
            continue
        vq = np.array(vd['q'])
        if not (vq == 15000).any():
            continue
        V = np.array([vd['x'], vd['y'], vd['z']]).T[vq == 15000][0]
        FP = np.array([fd['x'], fd['y'], fd['z']]).T
        FR = np.array(fd['real_cluster_id'])
        SR = np.array(sd['real_cluster_id'])
        SQ = np.array(sd['q'])
        shower_ids = set(SR[SQ == 15000].tolist())

        # incident legs (production semantics: every leg with a fit point at
        # the vertex and chord above the SBND fit_vertex_min_seg_length cut)
        legs = []
        for L in sorted(set(FR[FR >= 0].tolist())):
            pts = FP[FR == L]
            if len(pts) < 2:
                continue
            if np.linalg.norm(pts - V, axis=1).min() > DMIN:
                continue
            chord = np.linalg.norm(pts[0] - pts[-1])
            if chord < PREFILTER:
                continue
            legs.append((L, pts, chord))
        if len(legs) < 2:
            continue

        # production per-leg PCA
        prod, robust, fired_legs = [], [], []
        for L, pts, chord in legs:
            r1 = 1.5 if chord > 3.0 else 0.9
            pc_in = leg_pca(pts, V, r1, 6.0)
            if pc_in is None:
                continue
            entry = dict(rcid=L, chord=chord, pc=pc_in,
                         shower=(L in shower_ids))
            prod.append(entry)

        ntracks = len(prod)
        if ntracks < 2:
            continue

        # robust substitution pass (mirrors the planned C++ order: the
        # substitution happens in AddSegment's epilogue, BEFORE FitVertex's
        # pair-angle census — so substituted axes can open the gate for
        # hairpin vertices whose inner axes agree, e.g. 57903)
        for e in prod:
            pts = FP[FR == e['rcid']]
            sub = None
            ang = -1.0; aniso = -1.0; n_out = 0; rin = -1.0; rout = -1.0
            if (not e['shower']) and e['chord'] >= P['min_len']:
                far = np.linalg.norm(pts - V, axis=1).max()
                reseat = 4.0 if far > 8.0 else 0.0
                rin = max(1.5, reseat + P['rin_margin'])
                rout = min(max(P['rout_frac'] * e['chord'], P['rout_min']),
                           P['rout_max'])
                pc_out = leg_pca(pts, V, rin, rout)
                if pc_out is not None:
                    n_out = pc_out['n']
                    aniso = np.sqrt(pc_out['lam'][0] / pc_out['lam'][1])
                    ang = fold(e['pc']['ax'][0], pc_out['ax'][0])
                    if (ang > P['angle'] and n_out >= P['min_pts']
                            and aniso >= P['min_aniso']):
                        # substitute PCA/center, KEEP production npoints;
                        # hemisphere-orient axes toward production counterparts
                        ax = pc_out['ax'].copy()
                        for k in range(3):
                            if np.dot(ax[k], e['pc']['ax'][k]) < 0:
                                ax[k] = -ax[k]
                        sub = dict(n=e['pc']['n'], lam=pc_out['lam'],
                                   ax=ax, center=pc_out['center'])
            e['ang'] = ang; e['aniso'] = aniso; e['n_out'] = n_out
            e['rin'] = rin; e['rout'] = rout
            e['sub'] = sub
            robust.append(sub if sub is not None else e['pc'])
            if sub is not None:
                fired_legs.append(e['rcid'])

        def wide_pairs(pcas):
            """>=1 pair opening angle > 15 deg with each leg's axis oriented
            AWAY from the vertex (physical opening; approximates the C++ pair
            loop whose raw eigenvector signs are arbitrary)."""
            away = []
            for e, pc in zip(prod, pcas):
                pts = FP[FR == e['rcid']]
                far = pts[np.argmax(np.linalg.norm(pts - V, axis=1))]
                ax0 = pc['ax'][0].copy()
                if np.dot(ax0, far - pc['center']) < 0:
                    ax0 = -ax0
                away.append(ax0)
            wide = 0
            for a in range(len(away)):
                for b in range(a + 1, len(away)):
                    ang_ab = np.degrees(np.arccos(
                        np.clip(float(np.dot(away[a], away[b])), -1, 1)))
                    if ang_ab > 15:
                        wide += 1
            return wide

        gate_p = wide_pairs([e['pc'] for e in prod]) >= 1
        gate_r = wide_pairs(robust) >= 1
        if not gate_p and not gate_r:
            continue

        sol_p = (myfcn_solve([e['pc'] for e in prod], V, P['prior'])[0]
                 if gate_p else V.copy())
        rng2 = P['prior2'] if (fired_legs and ntracks == 2) else P['prior']
        sol_r = myfcn_solve(robust, V, rng2)[0] if gate_r else V.copy()

        move_p = np.linalg.norm(sol_p - V)
        move_r = np.linalg.norm(sol_r - V)
        dsol = np.linalg.norm(sol_r - sol_p)

        for e in prod:
            leg_rows.append((arm, evt, e['rcid'], ntracks, e['chord'],
                             int(e['shower']), e['pc']['n'], e['n_out'],
                             e['rin'], e['rout'], e['ang'], e['aniso'],
                             int(e['sub'] is not None)))
        vtx_rows.append((arm, evt, ntracks, len(fired_legs), rng2,
                         move_p, move_r, dsol,
                         sol_r[0], sol_r[1], sol_r[2], gate_p, gate_r))
        if verbose and fired_legs:
            print('%s evt %d: nlegs=%d fired=%s prior=%.2f gate p/r=%d/%d  '
                  'move prod=%.2f robust=%.2f  d(sol)=%.2f cm  -> (%.2f,%.2f,%.2f)'
                  % (arm, evt, ntracks, fired_legs, rng2, gate_p, gate_r,
                     move_p, move_r, dsol, sol_r[0], sol_r[1], sol_r[2]))

if tsv:
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(['arm', 'event', 'rcid', 'nlegs', 'chord_cm',
                            'shower', 'n_inner', 'n_outer', 'rin_cm', 'rout_cm',
                            'angle_in_out_deg', 'aniso_outer', 'fired']) + '\n')
        for r in leg_rows:
            fh.write('%s\t%d\t%d\t%d\t%.2f\t%d\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%d\n' % r)
    print('wrote', tsv)

nv = len(vtx_rows)
if nv == 0:
    print('no fittable main vertices')
    sys.exit(0)

fired = [r for r in vtx_rows if r[3] > 0]
d_fired = np.array([r[7] for r in fired]) if fired else np.array([])
d_all = np.array([r[7] for r in vtx_rows])
print('params: %s' % ' '.join('%s=%g' % kv for kv in sorted(P.items())))
print('fittable main vertices: %d   legs: %d   substituted legs: %d   '
      'vertices with >=1 fired leg: %d (%.1f%%)'
      % (nv, len(leg_rows), sum(r[12] for r in leg_rows), len(fired),
         100.0 * len(fired) / nv))
print('robust-vs-production solution distance: fired median %.2f cm max %.2f | '
      'unfired max %.3f cm'
      % (np.median(d_fired) if len(d_fired) else 0,
         d_fired.max() if len(d_fired) else 0,
         max((r[7] for r in vtx_rows if r[3] == 0), default=0.0)))
print()
print('fired vertices (sorted by |robust - production| solution distance):')
print('%-22s %8s %6s %6s %6s %9s %9s %9s  %s' %
      ('arm', 'event', 'nlegs', 'nfired', 'prior', 'move_p', 'move_r', 'd(sol)',
       'gate p/r'))
for r in sorted(fired, key=lambda r: -r[7]):
    print('%-22s %8d %6d %6d %6.2f %8.2f %9.2f %9.2f  %d/%d'
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
             r[11], r[12]))

#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/51 follow-up 3: how often does the MyFCN direction
window overlap MyFCN::UpdateInfo's own re-seat, and how often does that matter?

For every event of one or more arms: find the main (neutrino) vertex, find the
legs incident on it (track_fit-global segments with a fitted point within
DMIN cm), and for each leg compare
    inner axis  = PCA over the production window   (1.5, 6] cm      <- what the
                  fit actually uses, and which UpdateInfo rewrites out to 4 cm
    outer axis  = PCA over                          (R_OUT, 18] cm  <- untouched
Reports the angle between them and each axis's impact parameter w.r.t. the
reconstructed vertex.  A large angle on a long leg is the signature of the
self-confirming re-seat described in follow-up 3.

Read-only: consumes each arm's existing mabc-pr.zip, runs nothing.

Usage: vtx_dir_bias_census.py <ARM> [ARM ...] [--tsv OUT.tsv]
"""
import sys, os, json, glob, zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
INNER = (1.5, 6.0)     # MyFCN production window
R_OUT = 5.0            # > default_dis_cut (4 cm), so outside the re-seat
OUTER = (R_OUT, 18.0)
DMIN = 1.0             # a leg is "incident" if a fit point is this close
MINLEN = 10.0          # only legs long enough to have an outer window
FLOOR = 0.15 ** 2

tsv = None
argv = sys.argv[1:]
args = []
i = 0
while i < len(argv):
    if argv[i] == '--tsv':
        tsv = argv[i + 1]      # consume the flag AND its value
        i += 2
        continue
    if argv[i].startswith('--'):
        i += 1
        continue
    args.append(argv[i])
    i += 1
if not args:
    sys.exit(__doc__)


def pca(P, V, r1, r2):
    d = np.linalg.norm(P - V, axis=1)
    s = P[(d >= r1) & (d <= r2)]
    if len(s) < 3:
        return None
    c = s[np.argmin(np.linalg.norm(s - V, axis=1))]
    cov = np.zeros((3, 3))
    for p in s:
        cov += np.outer(p - c, p - c)
    cov /= len(s)
    w, e = np.linalg.eigh(cov)
    ax = e[:, -1]
    w_ = V - c
    return dict(n=len(s), ax=ax, center=c,
                imp=np.linalg.norm(w_ - np.dot(w_, ax) * ax))


rows = []
for arm in args:
    for d in sorted(glob.glob(os.path.join(SB, arm, 'pr_evt*'))):
        evt = int(os.path.basename(d)[6:])
        zp = os.path.join(d, 'mabc-pr.zip')
        if not os.path.exists(zp):
            continue
        try:
            z = zipfile.ZipFile(zp)
            vd = json.loads(z.read('data/0/0-vertices-global.json'))
            fd = json.loads(z.read('data/0/0-track_fit-global.json'))
        except Exception as e:
            print('  [skip] %s evt %d: %s' % (arm, evt, e), file=sys.stderr)
            continue
        vq = np.array(vd['q'])
        if not (vq == 15000).any():
            continue
        V = np.array([vd['x'], vd['y'], vd['z']]).T[vq == 15000][0]
        FP = np.array([fd['x'], fd['y'], fd['z']]).T
        FR = np.array(fd['real_cluster_id'])
        for L in sorted(set(FR[FR >= 0].tolist())):
            P = FP[FR == L]
            if len(P) < 4:
                continue
            dd = np.linalg.norm(P - V, axis=1)
            if dd.min() > DMIN:
                continue
            if np.linalg.norm(P[0] - P[-1]) < MINLEN:
                continue
            a_in, a_out = pca(P, V, *INNER), pca(P, V, *OUTER)
            if a_in is None or a_out is None:
                continue
            ang = np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(a_in['ax'], a_out['ax']))))))
            rows.append((arm, evt, L, len(P), a_in['n'], a_out['n'],
                         ang, a_in['imp'], a_out['imp']))

hdr = ['arm', 'event', 'rcid', 'npts', 'n_inner', 'n_outer',
       'angle_in_out_deg', 'imp_inner_cm', 'imp_outer_cm']
if tsv:
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(hdr) + '\n')
        for r in rows:
            fh.write('%s\t%d\t%d\t%d\t%d\t%d\t%.2f\t%.3f\t%.3f\n' % r)
    print('wrote', tsv)

if not rows:
    print('no legs passed the selection')
    sys.exit(0)

ang = np.array([r[6] for r in rows])
imp_i = np.array([r[7] for r in rows])
imp_o = np.array([r[8] for r in rows])
evts = set((r[0], r[1]) for r in rows)
print('arms: %s' % ', '.join(args))
print('main vertices with >=1 qualifying leg : %d' % len(evts))
print('qualifying legs (>= %.0f cm, incident within %.1f cm) : %d'
      % (MINLEN, DMIN, len(rows)))
print()
print('inner window %s vs outer window %s' % (INNER, OUTER))
print('  angle(inner, outer): median %.1f deg, 75%% %.1f, 90%% %.1f, max %.1f'
      % (np.median(ang), np.percentile(ang, 75), np.percentile(ang, 90), ang.max()))
print('  impact parameter, inner axis: median %.2f cm, 90%% %.2f'
      % (np.median(imp_i), np.percentile(imp_i, 90)))
print('  impact parameter, outer axis: median %.2f cm, 90%% %.2f'
      % (np.median(imp_o), np.percentile(imp_o, 90)))
print()
print('%-10s %8s %8s   %s' % ('threshold', 'legs', 'vertices', 'fraction of vertices'))
for t in (10, 15, 20, 25, 30, 40):
    m = ang > t
    ve = set((rows[i][0], rows[i][1]) for i in np.where(m)[0])
    print('%-10s %8d %8d   %5.1f %%'
          % ('> %d deg' % t, m.sum(), len(ve), 100.0 * len(ve) / len(evts)))
print()
print('worst 15 legs by angle:')
print('%-22s %8s %8s %7s %9s %9s' %
      ('arm', 'event', 'rcid', 'angle', 'imp_in', 'imp_out'))
for r in sorted(rows, key=lambda r: -r[6])[:15]:
    print('%-22s %8d %8d %6.1f  %8.2f  %8.2f' % (r[0], r[1], r[2], r[6], r[7], r[8]))

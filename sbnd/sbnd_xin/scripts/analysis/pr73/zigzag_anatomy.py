#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73: anatomy of the three owner-reported trajectory
zigzags, and the figure.

Prints, and then plots:

  (E1) the fitted-polyline step length, to confirm it is the 0.6 cm path that
       organize_segments_path_3rd produces and that dQ_dx_multi_fit consumes
       (TrackFitting.cxx:8570 reassigns low_dis_limit = 0.6*units::cm).

  (E3) the image cloud around 18255-57903 seg 14001, selected FIT-INDEPENDENTLY
       by a cylinder about the segment's own chord, decomposed into the drift
       direction and the remaining in-plane transverse direction; plus the
       number of distinct drift slices the segment's charge occupies, and the
       charge captured by a 3 cm tube about the chord vs about the fit.

  (E4) the same segment across arms carrying different upstream (skeleton)
       knobs -- the discriminator between "the fit creates the zigzag" and
       "the fit inherits it".

  (E5) per-point q_bee and turn angle around each owner point, with the
       distance to the nearest image point, for 53427 / 54351 / 57903.

q_bee = fit.dQ * dQdx_scale + dQdx_offset clipped at 0
(MultiAlgBlobClustering.cxx:955-957); SBND 0.1 / -1000, so
dQ[e] = (q_bee + 1000) / 0.1 and q_bee == 0 means dQ <= 1e4 e over 0.6 cm.

Read-only.  Usage:
    zigzag_anatomy.py [--png OUT.png]
"""
import sys
import json
import os
import zipfile
import numpy as np

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
DQ_SCALE, DQ_OFFSET = 0.1, -1000.0          # cfg/.../sbnd/clus.jsonnet:1768

# the owner's three points, and the arm whose Bee content he was looking at
CASES = [
    dict(evt=53427, arm='work-pr64r4-on1k',   pt=(-24.6, -17.1, 446.1),
         bee='f8203fcd idx 13'),
    dict(evt=54351, arm='work-pr64r4-on1k',   pt=(-150.3, 82.4, 196.2),
         bee='f8203fcd idx 17'),
    dict(evt=57903, arm='work-pr51r6-flip50', pt=(-16.5, -65.6, 293.6),
         bee='deb8abf5 idx 0 (round-6 production)'),
]

# 57903 across upstream-knob arms, oldest first
ARMS_57903 = [
    ('work-pr67f-off50', 'pre-round-5 (steiner_gap_penalty off)'),
    ('work-pr51r5-flip50', 'round 5 (steiner_gap_penalty = 2.0)'),
    ('work-pr51r6-flip50', 'round 6 (+ sgp_weak_scale = 5.0)'),
    ('work-pr51r7-on50', 'round 7 = current production (+ mvfit_robust)'),
]

png = None
if '--png' in sys.argv:
    png = sys.argv[sys.argv.index('--png') + 1]


def load(arm, evt):
    z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip'))
    return lambda n: json.loads(z.read('data/0/0-%s.json' % n))


def fitlayer(arm, evt):
    L = load(arm, evt)
    d = L('track_fit-global')
    return (np.array([d['x'], d['y'], d['z']]).T,
            np.array(d['real_cluster_id']),
            np.array(d['q'], dtype=float))


def imglayer(arm, evt):
    L = load(arm, evt)
    d = L('clustering-global')
    return (np.array([d['x'], d['y'], d['z']]).T,
            np.clip(np.array(d['q'], dtype=float), 0, None),
            np.array(d['cluster_id']))


def frame(S):
    """Chord frame of a polyline: unit chord u, drift-ish e_dr, in-plane e_iso."""
    u = S[-1] - S[0]
    u = u / np.linalg.norm(u)
    ex = np.array([1.0, 0.0, 0.0])
    e_dr = ex - np.dot(ex, u) * u
    e_dr /= np.linalg.norm(e_dr)
    return u, e_dr, np.cross(u, e_dr)


def turn_angles(S):
    st = np.diff(S, axis=0)
    n = np.linalg.norm(st, axis=1)
    ok = n > 1e-6
    t = st[ok] / n[ok, None]
    return np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', t[:-1], t[1:]), -1, 1)))


def owner_segment(arm, evt, pt, min_pts=10):
    """The segment whose fitted points come closest to the owner's point."""
    P, R, Q = fitlayer(arm, evt)
    best = None
    for s in sorted(set(R[R >= 0].tolist())):
        m = R == s
        if m.sum() < min_pts:
            continue
        d = float(np.linalg.norm(P[m] - np.array(pt), axis=1).min())
        if best is None or d < best[0]:
            best = (d, int(s), P[m], Q[m])
    return best


print('=' * 78)
print('E1 -- fitted-polyline step length (expect a uniform 0.600 cm)')
print('=' * 78)
for c in CASES:
    d, s, S, q = owner_segment(c['arm'], c['evt'], c['pt'])
    st = np.linalg.norm(np.diff(S, axis=0), axis=1)
    print('  evt %d seg %-6d n=%3d  step min %.3f  median %.3f  max %.3f cm'
          % (c['evt'], s, len(S), st.min(), np.median(st), st.max()))

print()
print('=' * 78)
print('E4 -- 18255-57903, the owner segment across upstream-knob arms')
print('=' * 78)
print('%-22s %-46s %6s %5s %8s %8s %7s %8s'
      % ('arm', 'what changed upstream', 'seg', 'n', 'chord', 'path', 'ratio', 'max_iso'))
E4 = []
for arm, what in ARMS_57903:
    d, s, S, q = owner_segment(arm, 57903, (-16.5, -65.6, 293.6))
    chord = np.linalg.norm(S[0] - S[-1])
    path = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    u, e_dr, e_iso = frame(S)
    r = S - S[0]
    perp = r - np.outer(r @ u, u)
    print('%-22s %-46s %6d %5d %8.2f %8.2f %7.3f %8.3f'
          % (arm, what, s, len(S), chord, path, path / chord,
             np.abs(perp @ e_iso).max()))
    E4.append((arm, what, s, S, q))

# main vertex + charge coverage per arm
print()
print('%-22s %-34s %s' % ('arm', 'main vertex (x,y,z) cm', 'cluster charge within 1.5 / 3.0 cm of any fit point'))
for arm, what, s, S, q in E4:
    L = load(arm, 57903)
    v = L('vertices-global')
    VP = np.array([v['x'], v['y'], v['z']]).T
    vq = np.array(v['q'])
    mv = VP[vq == 15000][0] if (vq == 15000).any() else None
    P, R, _ = fitlayer(arm, 57903)
    F = np.vstack([P[R == t] for t in sorted(set(R[R >= 0].tolist()))
                   if (R == t).sum() >= 10])
    IP, IQ, IC = imglayer(arm, 57903)
    # the cluster the owner segment lives in
    dmin = np.min(np.linalg.norm(IP[:, None, :] - S[None, :, :], axis=2), axis=1)
    cid = int(np.bincount(IC[dmin < 4.0]).argmax())
    m = IC == cid
    dfit = np.min(np.linalg.norm(IP[m][:, None, :] - F[None, :, :], axis=2), axis=1)
    tot = IQ[m].sum()
    print('%-22s %-34s cluster %d: %.1f %% / %.1f %%'
          % (arm, np.array2string(mv, precision=2) if mv is not None else 'none',
             cid, 100 * IQ[m][dfit < 1.5].sum() / tot,
             100 * IQ[m][dfit < 3.0].sum() / tot))

# the near-vertex opening angle between the two hairpin legs
print()
print('  opening angle between the two legs at the main vertex (mean direction over 1-8 cm):')
for arm, what in ARMS_57903:
    P, R, _ = fitlayer(arm, 57903)
    L = load(arm, 57903)
    v = L('vertices-global')
    vq = np.array(v['q'])
    if not (vq == 15000).any():
        continue
    V = np.array([v['x'], v['y'], v['z']]).T[vq == 15000][0]
    dirs = []
    for s in sorted(set(R[R >= 0].tolist())):
        Q = P[R == s]
        if len(Q) < 10:
            continue
        dd = np.linalg.norm(Q - V, axis=1)
        if dd.min() > 1.0:
            continue
        sel = Q[(dd > 1.0) & (dd < 8.0)]
        if len(sel) < 3:
            continue
        u = sel.mean(axis=0) - V
        dirs.append((int(s), u / np.linalg.norm(u)))
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            print('     %-22s seg %d vs %d : %.1f deg'
                  % (arm, dirs[i][0], dirs[j][0],
                     np.degrees(np.arccos(np.clip(np.dot(dirs[i][1], dirs[j][1]), -1, 1)))))

print()
print('=' * 78)
print('E3 -- 18255-57903 seg 14001: the image is a ribbon (fit-independent)')
print('=' * 78)
ARM6 = 'work-pr51r6-flip50'
d6, s6, S6, q6 = owner_segment(ARM6, 57903, (-16.5, -65.6, 293.6))
IP, IQ, IC = imglayer(ARM6, 57903)
dmin = np.min(np.linalg.norm(IP[:, None, :] - S6[None, :, :], axis=2), axis=1)
CID = int(np.bincount(IC[dmin < 4.0]).argmax())
slices = np.unique(np.round(IP[dmin < 4.0][:, 0], 3))
print('  seg %d: chord %.2f cm; its charge occupies %d distinct drift slices'
      % (s6, np.linalg.norm(S6[0] - S6[-1]), len(slices)))
print('  drift slice pitch %.4f cm  ->  %.1f cm of track per slice'
      % (np.diff(slices).min(), np.linalg.norm(S6[0] - S6[-1]) / len(slices)))
for tag, seg in (('14006 (iso 26 deg, ratio 1.03)', None),):
    pass
A, B = S6[0], S6[-1]
u, e_dr, e_iso = frame(S6)
Lc = np.linalg.norm(B - A)
r = IP - A
t = r @ u
pv = r - np.outer(t, u)
rho = np.linalg.norm(pv, axis=1)
inband = (IC == CID) & (t >= -1) & (t <= Lc + 1)
cyl = inband & (rho < 8.0)
print('  within an 8 cm cylinder about the CHORD (no reference to the fit):')
print('     %d image points; drift-direction rms %.2f cm, in-plane rms %.2f cm'
      % (cyl.sum(), (pv[cyl] @ e_dr).std(), (pv[cyl] @ e_iso).std()))
dfit6 = np.min(np.linalg.norm(IP[:, None, :] - S6[None, :, :], axis=2), axis=1)
tot = IQ[inband].sum()
print('  charge within 3 cm of the straight CHORD : %9.0f  (%.1f %% of in-band cluster charge)'
      % (IQ[inband & (rho < 3)].sum(), 100 * IQ[inband & (rho < 3)].sum() / tot))
print('  charge within 3 cm of the zigzag FIT     : %9.0f  (%.1f %%)'
      % (IQ[inband & (dfit6 < 3)].sum(), 100 * IQ[inband & (dfit6 < 3)].sum() / tot))
print('  -> a straight chord between the CURRENT endpoints captures LESS charge,')
print('     so "straight line between the two ends" is not literally the target;')
print('     the target is a smooth line through the charge ridge.')

print()
print('  transverse profile along the chord (charge-weighted, fit-independent):')
print('  %7s %6s %11s %10s %9s   %s'
      % ('t_cm', 'nimg', 'q', 'iso_qmean', 'iso_rms', 'fit_iso range'))
rf = S6 - A
tf = rf @ u
iof = (rf - np.outer(tf, u)) @ e_iso
io = pv @ e_iso
PROF = []
for lo in np.arange(-1, Lc + 1, 2.0):
    m = cyl & (t >= lo) & (t < lo + 2)
    mf = (tf >= lo) & (tf < lo + 2)
    if m.sum() == 0:
        continue
    w = IQ[m]
    qm = float((io[m] * w).sum() / max(w.sum(), 1e-9))
    PROF.append((lo + 1.0, qm, io[m].std()))
    print('  %7.1f %6d %11.0f %10.2f %9.2f   %s'
          % (lo, m.sum(), w.sum(), qm, io[m].std(),
             '[%.1f, %.1f]' % (iof[mf].min(), iof[mf].max()) if mf.sum() else '-'))

print()
print('=' * 78)
print('E4b -- charge coverage restricted to the CHANGED region only')
print('=' * 78)
# Fix the region once, from the round-6 arm, then apply it unchanged to every
# arm: the whole-cluster number of the previous block includes seg 14006, which
# no knob moves, and that dilutes a local change.
_P, _R, _ = fitlayer('work-pr51r6-flip50', 57903)
_REG = np.vstack([_P[_R == 14001], _P[_R == 14007]])
zlo, zhi = _REG[:, 2].min(), _REG[:, 2].max()
print('  region: cluster %d image points with z in [%.2f, %.2f] cm'
      % (CID if 'CID' in dir() else 14, zlo, zhi))
print('  (the union z-range of round-6 segments 14001 and 14007)')
print('  %-22s %8s %11s %9s %9s' % ('arm', 'nimg', 'charge', '<1.5 cm', '<3.0 cm'))
for arm, what in ARMS_57903:
    IP_, IQ_, IC_ = imglayer(arm, 57903)
    P_, R_, _ = fitlayer(arm, 57903)
    F = np.vstack([P_[R_ == t] for t in sorted(set(R_[R_ >= 0].tolist()))
                   if (R_ == t).sum() >= 10])
    m = (IC_ == 14) & (IP_[:, 2] >= zlo) & (IP_[:, 2] <= zhi)
    dfit = np.min(np.linalg.norm(IP_[m][:, None, :] - F[None, :, :], axis=2), axis=1)
    tot = IQ_[m].sum()
    print('  %-22s %8d %11.0f %8.1f %% %8.1f %%'
          % (arm, m.sum(), tot,
             100 * IQ_[m][dfit < 1.5].sum() / tot,
             100 * IQ_[m][dfit < 3.0].sum() / tot))

print()
print('=' * 78)
print('E4c -- bow versus jitter: the two are different phenomena')
print('=' * 78)
# path/chord and the fold-back fraction both grow with path length, but a smooth
# large-amplitude excursion (a bow) and a small-amplitude sawtooth (jitter)
# demand different fixes.  Split them: fit a degree-4 polynomial in arclength to
# each transverse component, then report path/chord OF the smooth curve and
# path/chord of the raw path ABOUT that curve.
DEG = 4
DEG_SWEEP = (2, 4, 6)


def bow_jitter(S, deg):
    chord = np.linalg.norm(S[0] - S[-1])
    u = (S[-1] - S[0]) / chord
    t = (S - S[0]) @ u
    ex = np.array([1.0, 0.0, 0.0])
    e1 = ex - np.dot(ex, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    perp = (S - S[0]) - np.outer(t, u)
    SM = S[0] + np.outer(t, u)
    res = np.zeros_like(S)
    for e in (e1, e2):
        c = perp @ e
        f = np.polyval(np.polyfit(t, c, deg), t)
        SM = SM + np.outer(f, e)
        res = res + np.outer(c - f, e)
    path = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    pb = np.linalg.norm(np.diff(SM, axis=0), axis=1).sum()
    Lb = np.linalg.norm(SM[0] - SM[-1])
    bow = float(np.abs(np.linalg.norm(SM - (S[0] + np.outer(t, u)), axis=1)).max())
    return dict(chord=chord, ratio=path / chord, bow=bow, ratio_bow=pb / Lb,
                ratio_jit=path / pb, jit_rms=float(np.linalg.norm(res, axis=1).std()))


print('  degree-%d polynomial in arclength, per transverse component' % DEG)
print('  %-24s %6s %7s %7s %9s %10s %10s %9s'
      % ('arm / event', 'seg', 'chord', 'ratio', 'bow_amp', 'ratio_bow', 'ratio_jit', 'jit_rms'))
CASE_SEGS = ([(c['arm'], c['evt'], c['pt']) for c in CASES]
             + [('work-pr51r7-on50', 57903, CASES[2]['pt']),
                ('work-pr67f-off50', 57903, CASES[2]['pt'])])
for arm, evt, pt in CASE_SEGS:
    _d, s, S, _q = owner_segment(arm, evt, pt)
    r = bow_jitter(S, DEG)
    print('  %-24s %6d %7.2f %7.3f %9.2f %10.3f %10.3f %9.3f'
          % ('%s / %d' % (arm.replace('work-', ''), evt), s, r['chord'],
             r['ratio'], r['bow'], r['ratio_bow'], r['ratio_jit'], r['jit_rms']))
print('  ratio_bow x ratio_jit = ratio.  57903 is bow-dominated; 53427/54351 are jitter.')
print()
print('  degree sensitivity (ratio_bow / ratio_jit) -- the split is NOT degree-free:')
print('  %-24s %s' % ('', '   '.join('deg %d' % g for g in DEG_SWEEP)))
for arm, evt, pt in CASE_SEGS[:3]:
    _d, s, S, _q = owner_segment(arm, evt, pt)
    print('  %-24s %s'
          % ('%s / %d' % (arm.replace('work-', ''), evt),
             '   '.join('%.3f/%.3f' % (lambda r: (r['ratio_bow'], r['ratio_jit']))(bow_jitter(S, g))
                        for g in DEG_SWEEP)))
print('  a quadratic cannot represent 57903\'s excursion and dumps it into "jitter";')
print('  degrees >= 4 agree.  The bow claim rests on the raw transverse profile and')
print('  on the endpoint pinning, not on this polynomial.')
print('  NB multi_trajectory_fit pins both endpoints to vertex->fit().point and')
print('  never fits them (TrackFitting.cxx:4246-4259), so a vertex off the charge')
print('  ridge FORCES a bow -- interior-only smoothing cannot remove it.')

print()
print('=' * 78)
print('E5 -- per-point q_bee and turn angle around each owner point')
print('=' * 78)
E5 = []
for c in CASES:
    d, s, S, q = owner_segment(c['arm'], c['evt'], c['pt'])
    IPc, IQc, _ = imglayer(c['arm'], c['evt'])
    j = int(np.linalg.norm(S - np.array(c['pt']), axis=1).argmin())
    ang = np.concatenate(([0.0], turn_angles(S), [0.0]))
    a, b = max(0, j - 10), min(len(S), j + 11)
    print('  evt %d (%s) seg %d, owner point at index %d of %d'
          % (c['evt'], c['bee'], s, j, len(S)))
    print('  %5s %9s %11s %8s %9s %11s'
          % ('i', 'q_bee', 'dQ [e]', 'turn deg', 'd_img cm', 'q_img<1.5cm'))
    for k in range(a, b):
        dd = np.linalg.norm(IPc - S[k], axis=1)
        print('  %5d %9.0f %11.0f %8.0f %9.2f %11.0f'
              % (k, q[k], (q[k] - DQ_OFFSET) / DQ_SCALE, ang[k],
                 dd.min(), IQc[dd < 1.5].sum()))
    E5.append((c, s, S, q, ang, j))
    print()

# ------------------------------------------------------------------ figure
if png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(18.0, 10.4))
    cols = ['#000000', '#2ca02c', '#d62728', '#1f77b4']
    styles = ['--', '-', '-', '-']
    lws = [2.6, 1.6, 1.8, 1.8]

    # (a) z-y and (b) z-x overlay of the four arms, 57903, zoomed on the
    # owner segment's own neighbourhood
    m = IC == CID
    zlo, zhi = S6[:, 2].min() - 12, S6[:, 2].max() + 8
    box = m & (IP[:, 2] > zlo) & (IP[:, 2] < zhi)
    for col, proj, lab, a_ in ((0, lambda P: (P[:, 2], P[:, 1]), ('z (cm)', 'y (cm)'), ax[0, 0]),
                               (1, lambda P: (P[:, 2], P[:, 0]), ('z (cm)', 'x (cm)  [drift]'), ax[0, 1])):
        X, Y = proj(IP[box])
        a_.scatter(X, Y, s=13, c=np.clip(IQ[box], 0, np.percentile(IQ[box], 98)),
                   cmap='Greys', alpha=0.6, linewidths=0, zorder=1)
        for i, (arm, what, s, S, q) in enumerate(E4):
            x, y = proj(S)
            a_.plot(x, y, styles[i], lw=lws[i], color=cols[i],
                    alpha=0.95, zorder=3 + (4 - i),
                    label=what if col == 0 else None)
        x, y = proj(np.atleast_2d(np.array(CASES[2]['pt'])))
        a_.plot(x, y, '*', ms=17, color='#ff7f0e', mec='k', mew=0.7, zorder=9,
                label='owner point' if col == 0 else None)
        a_.set_xlabel(lab[0]); a_.set_ylabel(lab[1])
        a_.set_xlim(zlo, zhi)
        a_.grid(alpha=0.25, lw=0.4)
    ax[0, 0].set_ylim(S6[:, 1].min() - 10, S6[:, 1].max() + 6)
    ax[0, 0].set_aspect('equal', 'box')
    ax[0, 1].set_ylim(S6[:, 0].mean() - 9, S6[:, 0].mean() + 9)
    ax[0, 0].set_title('18255-57903 cluster %d: the owner segment across upstream-knob arms'
                       % CID, fontsize=10, loc='left')
    ax[0, 1].set_title('same, drift view (y-axis expanded): the whole segment is ~6 drift slices thick',
                       fontsize=10, loc='left')
    ax[0, 0].legend(fontsize=8, loc='lower left', framealpha=0.92)

    # (c) transverse profile about the chord
    a_ = ax[0, 2]
    P_ = np.array(PROF)
    a_.errorbar(P_[:, 0], P_[:, 1], yerr=P_[:, 2], fmt='o-', color='#7f7f7f',
                ms=4, lw=1.4, capsize=3,
                label='image charge centroid about the chord\n(fit-independent; bars = rms width)')
    a_.plot(tf, iof, '-', color=cols[2], lw=1.8,
            label='round-6 production fit (seg %d)' % s6)
    a_.axhline(0, color='k', lw=1.0, ls='--', label='straight chord between the two ends')
    a_.set_xlabel('distance along the chord (cm)')
    a_.set_ylabel('transverse offset, in-plane (cm)')
    a_.grid(alpha=0.25, lw=0.4)
    a_.legend(fontsize=8, loc='best', framealpha=0.92)
    a_.set_title('the charge is a ribbon 2.6 cm rms wide -- there is no per-point 3-D anchor\n'
                 '(the fit is 0 at both ends by construction: it is measured about its own chord)',
                 fontsize=9.5, loc='left')

    # (d-f) q_bee and turn angle vs index, one per owner case
    for k, (c, s, S, q, ang, j) in enumerate(E5):
        a_ = ax[1, k]
        idx = np.arange(len(S))
        a_.plot(idx, q, '-', color='#1f77b4', lw=1.4, label='q_bee  (0 <=> dQ <= 1e4 e)')
        a_.axvline(j, color='#ff7f0e', lw=1.6, ls='--', label='owner point')
        a_.set_xlabel('fitted point index (0.6 cm apart)')
        a_.set_ylabel('q_bee', color='#1f77b4')
        a_.tick_params(axis='y', labelcolor='#1f77b4')
        a_.grid(alpha=0.25, lw=0.4)
        a2 = a_.twinx()
        a2.plot(idx, ang, '-', color='#d62728', lw=1.0, alpha=0.75)
        a2.set_ylabel('turn angle between steps (deg)', color='#d62728')
        a2.tick_params(axis='y', labelcolor='#d62728')
        a2.set_ylim(0, 185)
        lo, hi = max(0, j - 45), min(len(S), j + 46)
        a_.set_xlim(lo, hi)
        a_.set_title('18255-%d seg %d  (%s)' % (c['evt'], s, c['bee']),
                     fontsize=10, loc='left')
        if k == 0:
            a_.legend(fontsize=8, loc='upper left', framealpha=0.92)

    fig.suptitle('doc pr/73 -- the fitted trajectory zigzags because its seed does, and every anti-zigzag guard in the fit is measured against that same seed.\n'
                 'Top: before pr/51 round 5 this region was ONE smooth 48 cm segment (path/chord 1.026) at 0.6 deg from isochronous;\n'
                 'today it is a hairpin pair (27 deg opening at the main vertex in round 6, 69 deg in round 7) at path/chord 1.32 / 1.25.\n'
                 'Bottom: two distinct failures -- 53427/54351 keep the trajectory within 0.5 cm of image charge and lose dQ only at fold-back kinks (130 deg / 70 deg);\n'
                 '57903 instead walks off the charge ridge for 11 consecutive points (6.6 cm) and dQ floors there.',
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png, dpi=135)
    print('wrote', png)

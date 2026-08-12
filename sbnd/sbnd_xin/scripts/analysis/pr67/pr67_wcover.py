#!/usr/bin/env python3
"""doc pr/67: measure the owner's own detection criterion -- "I cannot have
track trajectory covering the W plane channels" -- as a number.

The owner finds these cases by looking at the COLLECTION (W) plane and seeing
image charge on W channels that no fitted trajectory projects onto.  This
script reproduces that test directly:

  * take the target cluster's image points from the PR-stage
    `0-clustering-global.json` inside `pr_evt<N>/mabc-pr.zip`,
  * take the fitted polyline(s) from `0-track_fit-global.json`, resampled
    densely so a straight segment does not skip channels,
  * project both into the event's own per-(apa,face) W wire index using
    `oc53_probe.Loader.wind()` (the same `wind = m*y2d + c` fit against the
    event's real `ctpc_a<A>f0pW` arrays that doc pr/53 used, so the channel
    numbers are the detector's, not a geometry guess),
  * report the W channels that carry image charge but have NO fit projection,
    both channel-level and (channel, drift-x bin)-level.

FRAME WARNING (doc pr/13, and measured again in this round).  The owner reads
coordinates off the Bee `img-global` layer, which is the ONLY raw layer.
`clustering-global` / `shower_track-global` / `track_fit-global` and the
pctree are all T0-corrected and carry a PER-CLUSTER drift-x offset -- measured
in this round as ~1 cm for the neutrino candidates but 99 cm for a cosmic in
the same event (18255-58717).  `--owner-point` is therefore interpreted in the
img-global frame and mapped across by the local offset before use; the offset
actually applied is always printed.  Never compare the two frames directly.

Usage:
  pr67_wcover.py <pr_evt_dir> [--owner-point X Y Z] [--cid N] [--xbin 0.5]
                              [--ql-dir DIR] [--step 0.2] [--tol 1]

  <pr_evt_dir>  a PR-stage event dir holding mabc-pr.zip and
                pctree-pr-evt<N>.tar.gz, e.g. work-pr67-base48/pr_evt137238
  --owner-point coordinates AS READ OFF img-global (cm); selects the target
                cluster and anchors the report.  Requires --ql-dir (or an
                img-global next to it) to do the frame mapping.
  --cid         target cluster id in the clustering-global frame, if you
                already know it (skips the owner-point lookup)
  --xbin        drift-x bin width in cm for the (channel, x) test (default 0.5)
  --step        polyline resampling step in cm (default 0.2)
  --tol         channel tolerance: a W channel counts as covered if the fit
                touches it within +/- tol channels (default 1)

Repro (doc pr/67):
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 scripts/analysis/pr67/pr67_wcover.py work-pr67-base48/pr_evt137238 \
      --owner-point -122.0 22.5 423.2 --ql-dir work-nuecc48-cb0805/ql_evt137238
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pr53'))
from oc53_probe import Loader, _y2d  # noqa: E402


def load_layers(pr_evt_dir):
    """The three PR Bee layers, read from the PR-stage archive directly.

    Deliberately NOT the bee/*.zip upload copy: make_pr_bee.py rewrites
    cluster_id into img-global's id space there (doc pr/55 gotcha), which
    breaks any cluster-keyed analysis.
    """
    zp = os.path.join(pr_evt_dir, 'mabc-pr.zip')
    z = zipfile.ZipFile(zp)
    out = {}
    for key in ('clustering-global', 'track_fit-global', 'shower_track-global'):
        out[key] = json.loads(z.read('data/0/0-%s.json' % key))
    return out


def pts(d):
    return np.c_[d['x'], d['y'], d['z']].astype(float)


def frame_offset(img_pts, clus_pts, p_img):
    """Local img-global -> clustering-global drift-x offset near p_img.

    Matched on (y,z) only, because that pair is invariant under the T0
    correction while x is exactly what the correction moves.  Returns
    (dx, matched_img_point).  The caller prints it; a large value is a signal
    the target is a T0-shifted (cosmic-tagged) cluster, not an error.
    """
    from scipy.spatial import cKDTree
    k = int(np.argmin(np.linalg.norm(img_pts - p_img, axis=1)))
    tree = cKDTree(clus_pts[:, 1:])
    _, j = tree.query(img_pts[k, 1:])
    return float(img_pts[k, 0] - clus_pts[j, 0]), img_pts[k]


def resample(poly, step):
    """Densify a polyline so consecutive samples are <= step apart.

    Without this, a 20 cm straight segment contributes 2 samples and appears
    to "cover" no channels between its ends -- the metric would then report a
    fictitious hole wherever the fit is simply sparse.
    """
    if len(poly) < 2:
        return poly
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        d = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(d / step)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out)


def wcells(ld, P, xbin):
    """Map 3D points (cm) to {(apa, wchan)} and {(apa, wchan, xbin)}."""
    ch, cell = set(), set()
    for p in P:
        x, y, z = p * 10.0                      # cm -> mm, the ctpc/wind units
        apa = 0 if x < 0 else 1
        w = ld.wind(apa, 2, _y2d(2, y, z))
        ch.add((apa, w))
        cell.add((apa, w, int(np.floor(p[0] / xbin))))
    return ch, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pr_evt_dir')
    ap.add_argument('--owner-point', nargs=3, type=float, default=None)
    ap.add_argument('--cid', type=int, default=None)
    ap.add_argument('--ql-dir', default=None)
    ap.add_argument('--xbin', type=float, default=0.5)
    ap.add_argument('--step', type=float, default=0.2)
    ap.add_argument('--tol', type=int, default=1)
    a = ap.parse_args()

    L = load_layers(a.pr_evt_dir)
    C = pts(L['clustering-global'])
    cid = np.array(L['clustering-global']['cluster_id'])
    F = pts(L['track_fit-global'])
    frc = np.array(L['track_fit-global']['real_cluster_id'])

    ld = Loader(a.pr_evt_dir)

    p_clus = None
    if a.owner_point is not None:
        p_img = np.array(a.owner_point, float)
        qd = a.ql_dir
        if qd is None:
            sys.exit('--owner-point needs --ql-dir (the ql_evt<N> dir with mabc-all-apa.zip)')
        zq = zipfile.ZipFile(os.path.join(qd, 'mabc-all-apa.zip'))
        img = json.loads(zq.read('data/0/0-img-global.json'))
        I = pts(img)
        dx, hit = frame_offset(I, C, p_img)
        p_clus = p_img - np.array([dx, 0, 0])
        print('owner point (img-global) %s -> nearest img pt %s (%.2f cm)'
              % (p_img.round(2), hit.round(2), np.linalg.norm(hit - p_img)))
        print('frame offset applied dx = %.2f cm -> clustering frame %s' % (dx, p_clus.round(2)))

    if a.cid is None:
        if p_clus is None:
            sys.exit('need --cid or --owner-point')
        a.cid = int(cid[int(np.argmin(np.linalg.norm(C - p_clus, axis=1)))])
    print('target cluster_id = %d' % a.cid)

    sel = np.where(cid == a.cid)[0]
    IMG = C[sel]
    segs = sorted({int(s) for s in frc if s > 0 and s // 1000 == a.cid})
    FIT = F[np.isin(frc, segs)] if segs else np.empty((0, 3))
    print('image points %d ; fitted segments %s ; fit points %d'
          % (len(IMG), segs, len(FIT)))

    # Resample each segment's polyline separately -- storage order is
    # trajectory order within a segment, but NOT across segments, so a global
    # resample would draw phantom lines between unrelated segment ends.
    dense = []
    for s in segs:
        dense.append(resample(F[frc == s], a.step))
    DENSE = np.vstack(dense) if dense else np.empty((0, 3))

    img_ch, img_cell = wcells(ld, IMG, a.xbin)
    fit_ch, fit_cell = wcells(ld, DENSE, a.xbin)

    def covered_ch(key):
        apa, w = key
        return any((apa, w + d) in fit_ch for d in range(-a.tol, a.tol + 1))

    def covered_cell(key):
        apa, w, xb = key
        return any((apa, w + d, xb + e) in fit_cell
                   for d in range(-a.tol, a.tol + 1) for e in (-1, 0, 1))

    unc_ch = sorted(k for k in img_ch if not covered_ch(k))
    unc_cell = sorted(k for k in img_cell if not covered_cell(k))
    print('\nW channels with image charge:      %d' % len(img_ch))
    print('W channels NOT covered by the fit: %d  (%.1f%%)'
          % (len(unc_ch), 100.0 * len(unc_ch) / max(1, len(img_ch))))
    print('(W chan, x-bin) cells with charge: %d' % len(img_cell))
    print('(W chan, x-bin) NOT covered:       %d  (%.1f%%)'
          % (len(unc_cell), 100.0 * len(unc_cell) / max(1, len(img_cell))))

    # Longest contiguous uncovered W-channel run per apa -- this is the shape
    # the owner actually sees on the plane view, and it localizes the hole far
    # better than the bare percentage.
    for apa in (0, 1):
        ws = sorted(w for (p, w) in unc_ch if p == apa)
        if not ws:
            continue
        runs, cur = [], [ws[0]]
        for w in ws[1:]:
            if w - cur[-1] <= 1:
                cur.append(w)
            else:
                runs.append(cur)
                cur = [w]
        runs.append(cur)
        runs.sort(key=len, reverse=True)
        print('\napa%d: %d uncovered W channels, longest runs %s'
              % (apa, len(ws), [(r[0], r[-1], len(r)) for r in runs[:5]]))
        # 3D bounding box of the image points sitting on the longest run
        top = set(runs[0])
        keep = []
        for p in IMG:
            x, y, z = p * 10.0
            if (0 if x < 0 else 1) != apa:
                continue
            if ld.wind(apa, 2, _y2d(2, y, z)) in top:
                keep.append(p)
        if keep:
            K = np.array(keep)
            print('      longest run holds %d image pts, bbox %s .. %s'
                  % (len(K), K.min(0).round(1), K.max(0).round(1)))

    if p_clus is not None and len(FIT):
        print('\nowner point -> nearest fit point %.2f cm'
              % np.linalg.norm(FIT - p_clus, axis=1).min())
    ld.cleanup()


if __name__ == '__main__':
    main()

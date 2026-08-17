#!/usr/bin/env python3
"""Find events where a cathode-crossing track was Q/L matched on ONE side only.

THE DEFECT.  A cosmic (or any long track) crosses the SBND central cathode and is
imaged as two pieces, one per drift volume.  Q/L matching assigns a flash T0 to a
cluster; the T0 correction then slides that cluster in x by

    dx = sign_offset(TPC) * v_drift * t0        sign_offset = -1 (TPC0, x<0)
                                                             = +1 (TPC1, x>0)

VERIFIED, not assumed (evt100002): the three cross-cathode clusters carry
equal-and-opposite dx on their two halves, and dx/v_drift reproduces the matched
flash's `op_t` to 0.1 us on every clean cluster.

`cathode_connect` joins the two halves only when BOTH carry the same flash T0.  If
only one half was matched, the other keeps dx = 0, stays parked at its raw drift
position -- often >100 cm from where it belongs -- and the reconstructed track just
stops dead at the cathode.  This script finds those events.

METHOD (everything comes from one Q/L-stage event dir; no PR products needed)

  ql_evt<ID>/mabc-all-apa.zip     0-img-global.json        RAW x (t0=0), img cids
                                  0-clustering-global.json T0-CORRECTED, all-APA cids
                                  0-op.json                flash -> img cluster ids
  ql_evt<ID>/mabc-apa{0,1}-face0.zip                       img cid -> APA (= TPC)

  1. join img <-> clustering on q (the per-point charge; no stage rewrites it) to
     get, per all-APA cluster K, its rigid dx_K and the img cids feeding it.
     NOT on (y,z): measured on evt100002, the correction also carries a small
     per-TPC transverse shift (dy ~ -0.11, dz ~ +0.67 cm), so a (y,z,q) key joins
     ZERO points.  q collides for a few % of points; those are dropped.
  2. K is a CROSSER if its img cids span both APAs -- the control bucket, and the
     proof the frame arithmetic is right.
  3. otherwise K is a half-track candidate if it is flash-matched, long enough, and
     its nearest approach to the cathode plane is within CATH_NEAR.
  4. take K's local PCA axis at that cathode end and extrapolate to x = 0, giving
     the predicted (y,z) where the track pierces the cathode.
  5. re-place every RAW img cluster of the OTHER APA under K's own t0 hypothesis
     (x' = x_raw - dx_K, the mirror) and keep those that (a) land inside the other
     TPC, (b) start at the cathode, (c) pierce x = 0 within GAP2D of K's
     prediction, (d) run parallel to K within ANGLE.
  6. classify the partner by its OWN dx: mirror of dx_K = crosser already handled
     (should be empty here, since (2) removed those); ~0 = UNMATCHED, the defect;
     anything else = matched to a DIFFERENT flash.

The (y,z)-at-the-cathode test is the discriminating one.  doc 14 measured the real
data cathode mismatch as a small transverse y-z offset on top of a ~0.9 cm dead-gap
floor, so a genuine partner lands within a few cm.  Counting charge in a
perpendicular corridor instead would sweep up unrelated cosmics, because the mirror
shift moves ALL of the other TPC's charge by up to ~150 cm.

TWO ID-SPACE TRAPS, both hit and both fixed here:
  * the per-APA Bee layers number their clusters from 1, so their ids overlap
    img-global's and each other's.  Taking them at face value labels every cluster
    APA1 and hides every crosser (82 -> 0 in a 30-event probe).  Join on q.
  * clustering-global cluster ids are a different id space again (doc pr/55).
    Never compare ids across layers; always go through the q join.

LIMITATION.  img-global carries the ACTIVE imaged blobs only.  A partner that falls
in a dead region lives in icluster-apa*-masked.npz and is invisible here.

Repro:
  cd .../sbnd/sbnd_xin
  python3 cathode_halfmatch.py --arm work-mcp1k-cb0805 --out mcp1k.tsv
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

# ---- SBND geometry, read off any calib-evt*.json 'geometry' (same every event)
V_DRIFT = 0.1563          # cm/us
ANODE = {0: -201.45, 1: 201.45}
CATH = {0: -0.45, 1: 0.45}
SIGN = {0: -1.0, 1: 1.0}  # sign_offset: dx = SIGN[apa] * V_DRIFT * t0

# ---- cuts.  Every one is also reported per row, so they can be retuned offline
#      from the TSV without re-running the sweep.
K_MIN_PTS = 100      # matched half: points
K_MIN_LEN = 25.0     # matched half: extent along its principal axis, cm
CATH_NEAR = 10.0     # matched half: nearest approach to the cathode plane, cm
DIRX_MIN = 0.30      # |dir_x| of the matched half at the cathode end
LOCAL_R = 20.0       # radius of the local PCA fit at a cathode end, cm
RIGID_TOL = 1.0      # cm, dx spread allowed inside one all-APA cluster
RIGID_FRAC = 0.95    # fraction of joined points that must sit inside RIGID_TOL
P_MIN_PTS = 30       # partner: points
P_MIN_LEN = 10.0     # partner: extent, cm
P_CATH_NEAR = 8.0    # partner: nearest approach to the cathode under K's t0, cm
P_INRANGE = 0.90     # partner: fraction of points inside the other TPC under K's t0
CORRIDOR = 5.0       # cm, perpendicular half-width for the "missing length" measure
GAP2D = 15.0         # (y,z) distance between the two cathode piercings, cm
ANGLE = 20.0         # deg between the two local axes
MIRROR_TOL = 3.0     # cm, |dx_partner + dx_K| below this = already-handled crosser
ZERO_TOL = 1.0       # cm, |dx_partner| below this = never T0-corrected


def _read(zf, base):
    for n in zf.namelist():
        if os.path.basename(n) == base:
            return json.loads(zf.read(n))
    return None


def load_event(ql_dir):
    z = zipfile.ZipFile(os.path.join(ql_dir, 'mabc-all-apa.zip'))
    img = _read(z, '0-img-global.json')
    clus = _read(z, '0-clustering-global.json')
    op = _read(z, '0-op.json')
    if img is None or clus is None or op is None:
        return None

    iq4 = np.round(img['q'], 4)
    vote = collections.defaultdict(collections.Counter)
    for apa in (0, 1):
        p = os.path.join(ql_dir, 'mabc-apa%d-face0.zip' % apa)
        if not os.path.isfile(p):
            return None
        d = _read(zipfile.ZipFile(p), '0-clustering-apa%d-face0.json' % apa)
        if d is None:
            return None
        qs = set(np.round(d['q'], 4).tolist())
        for q, c in zip(iq4, img['cluster_id']):
            if q in qs:
                vote[c][apa] += 1
    cid2apa = {c: v.most_common(1)[0][0] for c, v in vote.items()}
    return img, clus, op, cid2apa


def pca(P):
    """(centroid, unit principal axis, extent along it) of an (N,3) array."""
    c = P.mean(axis=0)
    Q = P - c
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    d = vt[0]
    t = Q @ d
    return c, d, float(t.max() - t.min())


def cathode_end(P, cath_x):
    """Local axis of P at its end nearest the cathode plane.

    Returns (end point, unit axis oriented toward x=0, min |x - cath_x|) or None
    when that end is too sparse to fit.
    """
    dcath = np.abs(P[:, 0] - cath_x)
    end = P[np.argsort(dcath)[:20]].mean(axis=0)
    sel = np.linalg.norm(P - end, axis=1) < LOCAL_R
    if sel.sum() < 10:
        return None
    _, d, _ = pca(P[sel])
    if d[0] * (0.0 - end[0]) < 0:            # orient toward the cathode plane
        d = -d
    return end, d, float(dcath.min())


def pierce(end, d):
    """(y, z) where the local axis crosses x = 0, or None if it never does."""
    if abs(d[0]) < 1e-6:
        return None
    t = (0.0 - end[0]) / d[0]
    return end[1] + t * d[1], end[2] + t * d[2]


def corridor_extent(P, end, d):
    """Extent of the points lying within CORRIDOR of the line (end, d), cm.

    This is the "how much track is actually missing" number: the whole-cluster
    extent would count an unrelated blob that happens to share the cluster.
    """
    Q = P - end
    t = Q @ d
    perp = np.linalg.norm(Q - np.outer(t, d), axis=1)
    sel = perp < CORRIDOR
    if sel.sum() < 3:
        return 0.0, 0
    return float(t[sel].max() - t[sel].min()), int(sel.sum())


def analyse(args):
    ql_dir, evt = args
    try:
        got = load_event(ql_dir)
    except Exception as exc:                       # report, never die mid-sweep
        return [], 'load-error: %s' % exc, collections.Counter()
    if got is None:
        return [], 'incomplete', collections.Counter()
    img, clus, op, cid2apa = got
    fun = collections.Counter()

    ix = np.array(img['x'], float)
    iy = np.array(img['y'], float)
    iz = np.array(img['z'], float)
    iq = np.array(img['q'], float)
    icid = np.array(img['cluster_id'], int)

    key = {}                                        # q -> img point index
    for i in range(len(ix)):
        k = round(iq[i], 4)
        key[k] = -1 if k in key else i

    cx = np.array(clus['x'], float)
    cy = np.array(clus['y'], float)
    cz = np.array(clus['z'], float)
    cq = np.array(clus['q'], float)
    ccid = np.array(clus['cluster_id'], int)

    gdx = collections.defaultdict(list)             # all-APA cluster -> dx samples
    members = collections.defaultdict(collections.Counter)
    idx_dx = collections.defaultdict(list)          # img cid -> dx samples
    dyz = collections.defaultdict(list)             # APA -> (dy, dz) samples
    for j in range(len(cx)):
        i = key.get(round(cq[j], 4), -1)
        if i < 0:
            continue
        gdx[ccid[j]].append(cx[j] - ix[i])
        members[ccid[j]][icid[i]] += 1
        idx_dx[icid[i]].append(cx[j] - ix[i])
        apa_i = cid2apa.get(icid[i])
        if apa_i is not None:
            dyz[apa_i].append((cy[j] - iy[i], cz[j] - iz[i]))

    # The transverse half of the same correction.  It FLIPS SIGN between the two
    # drift volumes (measured over 150 events: APA0 dy=-0.110 dz=+0.670, APA1
    # dy=+0.110 dz=-0.670, IQR 0.000 both), so it must be taken per APA -- an
    # event-wide median would cancel to ~0 and then be applied to the wrong TPC.
    dyz0 = {}
    for apa in (0, 1):
        v = dyz.get(apa)
        dyz0[apa] = ((float(np.median([a[0] for a in v])),
                      float(np.median([a[1] for a in v]))) if v else (0.0, 0.0))

    matched = set()
    for lst in op['op_cluster_ids']:
        matched.update(int(c) for c in lst)

    img_pts = {}
    for c in set(icid.tolist()):
        m = icid == c
        img_pts[c] = np.c_[ix[m], iy[m], iz[m]]
    img_dx, img_dx_rigid = {}, {}
    for c, v in idx_dx.items():
        a = np.asarray(v)
        img_dx[c] = float(np.median(a))
        # same guard gdx gets: a bimodal median would make the kind label
        # arbitrary.  Measured 0/3047 failures over 150 events, so this is a
        # tripwire, not a filter -- rows that fail are labelled, not dropped.
        img_dx_rigid[c] = float(np.mean(np.abs(a - img_dx[c]) < RIGID_TOL)) >= RIGID_FRAC

    rows = []
    for gid, cnt in members.items():
        fun['gcluster'] += 1
        apas = {cid2apa.get(c) for c in cnt}
        apas.discard(None)
        if len(apas) != 1:
            fun['crosser'] += 1                     # control bucket
            continue
        apa = apas.pop()
        if not (set(cnt) & matched):
            fun['half-unmatched'] += 1
            continue
        dxk = float(np.median(gdx[gid]))
        a = np.asarray(gdx[gid])
        if float(np.mean(np.abs(a - dxk) < RIGID_TOL)) < RIGID_FRAC:
            fun['non-rigid'] += 1
            continue
        t0 = dxk / (SIGN[apa] * V_DRIFT)

        m = ccid == gid
        P = np.c_[cx[m], cy[m], cz[m]]
        if len(P) < K_MIN_PTS:
            fun['too-few-points'] += 1
            continue
        _, _, length = pca(P)
        if length < K_MIN_LEN:
            fun['too-short'] += 1
            continue
        ce = cathode_end(P, CATH[apa])
        if ce is None:
            fun['no-end-fit'] += 1
            continue
        end, d, dmin = ce
        if dmin > CATH_NEAR:
            fun['far-from-cathode'] += 1
            continue
        if abs(d[0]) < DIRX_MIN:
            fun['isochronous'] += 1                 # no lever arm; reported, not hidden
            continue
        pk = pierce(end, d)
        if pk is None:
            continue
        yk, zk = pk
        fun['candidate-half'] += 1

        other = 1 - apa
        lo, hi = ((ANODE[other], CATH[other]) if other == 0
                  else (CATH[other], ANODE[other]))
        found = 0
        for c, apc in cid2apa.items():
            if apc != other or c not in img_pts:
                continue
            Praw = img_pts[c]
            if len(Praw) < P_MIN_PTS:
                continue
            Q = Praw.copy()
            Q[:, 0] -= dxk                          # mirror: dx_other = -dx_K
            Q[:, 1] += dyz0[other][0]
            Q[:, 2] += dyz0[other][1]
            frac = float(np.mean((Q[:, 0] >= lo) & (Q[:, 0] <= hi)))
            if frac < P_INRANGE:
                continue
            _, _, plen = pca(Q)
            if plen < P_MIN_LEN:
                continue
            pe = cathode_end(Q, CATH[other])
            if pe is None:
                continue
            end2, d2, dmin2 = pe
            if dmin2 > P_CATH_NEAR:
                continue
            pp = pierce(end2, d2)
            if pp is None:
                continue
            y2, z2 = pp
            gap = math.hypot(yk - y2, zk - z2)
            if gap > GAP2D:
                continue
            ang = math.degrees(math.acos(min(1.0, abs(float(d @ d2)))))
            if ang > ANGLE:
                continue
            ext, next_ = corridor_extent(Q, end2, d2)

            # disp = how far the partner sits from where this hypothesis says it
            # belongs.  Its own T0 puts it at x_raw + dxp; the matched half's T0
            # puts it at x_raw - dxk.  disp = |dxp + dxk|, and that -- not dxp --
            # is the size of the defect.
            dxp = img_dx.get(c, 0.0)
            disp = abs(dxp + dxk)
            if disp < MIRROR_TOL:
                kind = 'crosser-ok'                 # placed right, just not joined
            elif abs(dxp) < ZERO_TOL and c not in matched:
                kind = 'UNMATCHED'                  # no flash at all
            elif c not in matched:
                kind = 'unmatched-shifted'
            else:
                kind = 'other-flash'                # matched to the wrong flash
            if not img_dx_rigid.get(c, True):
                kind = 'ambiguous-dx'               # tripwire; never seen so far
            found += 1
            rows.append(dict(
                event=evt, kind=kind, gid=int(gid), apa=apa, t0=t0, dx=dxk,
                k_npts=int(len(P)), k_len=length, k_dcath=dmin, k_dirx=abs(d[0]),
                p_cid=int(c), p_npts=int(len(Praw)), p_len=plen, p_ext=ext,
                p_ncorr=next_, p_dcath=dmin2, p_dx=dxp, disp=disp, p_inrange=frac,
                gap2d=gap, angle=ang, y_cath=0.5 * (yk + y2), z_cath=0.5 * (zk + z2)))
        if not found:
            fun['no-partner'] += 1
    return rows, 'ok', fun


COLS = ['event', 'kind', 'gid', 'apa', 't0', 'dx', 'k_npts', 'k_len', 'k_dcath',
        'k_dirx', 'p_cid', 'p_npts', 'p_len', 'p_ext', 'p_ncorr', 'p_dcath',
        'p_dx', 'disp', 'p_inrange', 'gap2d', 'angle', 'y_cath', 'z_cath']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, help='work-*-<tag> dir holding ql_evt*/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.arm, 'ql_evt*')))
    if a.limit:
        dirs = dirs[:a.limit]
    tasks = [(d, int(re.search(r'ql_evt(\d+)$', d).group(1))) for d in dirs]

    rows, stats, fun = [], collections.Counter(), collections.Counter()
    with Pool(a.jobs) as pool:
        for i, (rr, st, ff) in enumerate(pool.imap_unordered(analyse, tasks, chunksize=4)):
            stats[st.split(':')[0]] += 1
            fun.update(ff)
            rows.extend(rr)
            if (i + 1) % 200 == 0:
                print('  %d/%d events, %d rows' % (i + 1, len(tasks), len(rows)),
                      file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (-r['p_ext'], r['gap2d']))
    with open(a.out, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(('%.4g' % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in COLS) + '\n')

    print('events   : %s' % dict(stats))
    print('funnel   : %s' % dict(fun))
    byk = collections.Counter(r['kind'] for r in rows)
    print('rows     : %s' % dict(byk))
    print('events/kind: %s' % {k: len({r['event'] for r in rows if r['kind'] == k})
                               for k in byk})
    print('wrote %s' % a.out)


if __name__ == '__main__':
    main()

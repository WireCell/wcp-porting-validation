#!/usr/bin/env python3
"""In-beam candidates whose track is cut at the cathode: PR only got one TPC's half.

THE QUESTION (owner, 2026-08-17).  A beam-window track crosses the central cathode.
Q/L matching puts only the half in one TPC into the in-beam bundle, so pattern
recognition never sees the other half and the reconstructed track stops at x = 0.
This is NOT about a large displacement between the two halves -- the beam flash is at
t ~ 0, so **the raw image already shows one consistent track through the cathode**.
The whole search therefore runs in RAW image coordinates with no T0 shift anywhere.

  beam window [0.2, 2.2) us  (cfg/pgrapher/experiment/sbnd/clus.jsonnet:518)
  => |dx| = v_drift * t0 <= 0.34 cm.  Sub-blob; ignored, not corrected.

WHAT PR ACTUALLY SEES is the in-beam bundle's **main** cluster, and only that: the PR
job runs `unmerge_bundle, unmerge_assoc` before Steiner (doc 53, doc 45), which strips
the associated clusters. So the object under test is

    nusel-table.tsv row with in_beam=1 and label != 'no-bundle'
      -> main_id, which is a `clustering-global` cluster id

VERIFIED, not assumed (evt409546): clustering-global clusters 6/7/9/11 have
956/196/283/2708 points and the nusel rows for main_id 6/7/9/11 carry npts_main
956/196/283/2708.  `op_cluster_ids` is deliberately NOT used for bundle membership --
it is a list of *img* cluster ids and agrees with `n_bundle` on only 54 % of in-beam
events (measured over 300), so it would invent missing halves on events where Q/L did
the right thing.

METHOD

  1. K = the in-beam main cluster, in clustering-global.
  2. for EACH side of the cathode on which K has charge: find K's end nearest the
     cathode plane, fit a local axis there, require it to be track-like (residual
     RMS < LIN_RMS) and to actually head across (|dir_x| >= DIRX_MIN); extrapolate
     to x = 0 for the predicted piercing (y, z).
  3. in the RAW image, take every img cluster of the OTHER TPC and ask whether it
     starts at the cathode and continues that line -- (y,z) piercing within GAP2D,
     direction within ANGLE.
  4. a continuation already inside K is `already-in-main`: the beam crosser the chain
     handled correctly, and the control.  One outside K is the defect.
  5. report how much track that is (extent inside a 5 cm corridor around its axis).

  Step 2 deliberately does NOT skip a main that merely HAS charge in both TPCs.  An
  in-beam main can be a scattered multi-fragment object (evt409546 main 9: 7
  fragments, x from -201 to +167), so "spans both TPCs" is not the same as "the
  continuation at this cathode end is present" -- taking it as such skipped 430 of
  837 in-beam mains wholesale and could hide a defect behind an unrelated fragment.

Sub-class worth separating: the other TPC HAS an in-beam flash which matched no
cluster at all.  That is a different Q/L failure from "matched it to a cosmic".

LIMITATION.  img-global carries the ACTIVE imaged blobs only; a partner in a dead
region lives in icluster-apa*-masked.npz and is invisible here.

Repro:
  cd .../sbnd/sbnd_xin
  python3 scripts/analysis/cathode/beam_cathode_split.py \
      --arm work-mcp1k-cb0805 --out products/beam-cathode-split-mcp1k.tsv --jobs 8
"""
import argparse
import collections
import glob
import math
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cathode_halfmatch import (ANODE, CATH, load_event, pca)   # noqa: E402

BEAM_LO, BEAM_HI = 0.2, 2.2      # us, SBND production beam window

K_MIN_PTS = 50       # in-beam main: points.  In-beam mains are long (median
K_MIN_LEN = 15.0     # 130 cm, 92 % >= 25 cm) but a short stub reaching the
                     # cathode is exactly the case worth finding, so these sit
                     # well below the bulk rather than at it.
CATH_NEAR = 10.0     # in-beam main: nearest approach to the cathode plane, cm
DIRX_MIN = 0.30      # |dir_x| at that end; below this there is no lever arm
LOCAL_R = 20.0       # radius of the local axis fit, cm
LIN_RMS = 3.0        # cm, residual RMS of that fit -- an in-beam main can be a
                     # scattered multi-fragment object (evt409546 main 9 has 7
                     # fragments spanning the whole detector), and fitting a blob
                     # would point the extrapolation anywhere
P_MIN_PTS = 20       # partner: points
P_MIN_LEN = 8.0      # partner: extent, cm
P_CATH_NEAR = 8.0    # partner: nearest approach to the cathode, cm
P_INRANGE = 0.90     # partner: fraction of points inside its own TPC
CORRIDOR = 5.0       # cm, half-width for the missing-length measure
GAP2D = 15.0         # cm between the two cathode piercings
ANGLE = 20.0         # deg between the two local axes


# Every cut above can be overridden from the environment (BCS_<NAME>=value) so a
# looser working point can be swept without editing the file.  Used to size the
# validation pool in doc 72 sec A.6.
for _k, _v in list(os.environ.items()):
    if _k.startswith('BCS_') and _k[4:] in globals():
        globals()[_k[4:]] = type(globals()[_k[4:]])(_v)


def read_nusel(arm):
    """event -> list of in-beam bundle rows.  Whitespace-aligned, not TSV.

    Prefers the arm's aggregated nusel-table.tsv; falls back to the per-event
    nusel_evt<ID>/nusel-evt<ID>.tsv files.  Deliberately never reads another arm's
    table: main_id is a clustering-global cluster id and only means anything
    against the clustering-global in the SAME arm.
    """
    paths = [os.path.join(arm, 'nusel-table.tsv')]
    if not os.path.isfile(paths[0]):
        paths = sorted(glob.glob(os.path.join(arm, 'nusel_evt*', 'nusel-evt*.tsv')))
    if not paths:
        raise SystemExit('no nusel table under ' + arm)
    out = collections.defaultdict(list)
    for path in paths:
        with open(path) as fh:
            hdr = fh.readline().split()
            for line in fh:
                f = line.split()
                if len(f) != len(hdr):
                    continue
                r = dict(zip(hdr, f))
                if r['in_beam'] == '1' and r['label'] != 'no-bundle':
                    out[int(r['event'])].append(r)
    return out


def local_axis(P, cath_x):
    """(end, unit axis toward x=0, min |x-cath|, residual RMS) at the cathode end."""
    dcath = np.abs(P[:, 0] - cath_x)
    end = P[np.argsort(dcath)[:20]].mean(axis=0)
    sel = np.linalg.norm(P - end, axis=1) < LOCAL_R
    if sel.sum() < 10:
        return None
    Q = P[sel]
    _, d, _ = pca(Q)
    R = Q - Q.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((R - np.outer(R @ d, d)) ** 2, axis=1))))
    if d[0] * (0.0 - end[0]) < 0:
        d = -d
    return end, d, float(dcath.min()), rms


def pierce(end, d):
    if abs(d[0]) < 1e-6:
        return None
    t = (0.0 - end[0]) / d[0]
    return end[1] + t * d[1], end[2] + t * d[2]


def corridor_extent(P, end, d):
    Q = P - end
    t = Q @ d
    perp = np.linalg.norm(Q - np.outer(t, d), axis=1)
    sel = perp < CORRIDOR
    if sel.sum() < 3:
        return 0.0, 0
    return float(t[sel].max() - t[sel].min()), int(sel.sum())


def analyse(args):
    ql_dir, evt, rows = args
    fun = collections.Counter()
    if not rows:
        fun['no-in-beam-bundle'] += 1
        return [], fun
    try:
        got = load_event(ql_dir)
    except Exception as exc:
        fun['load-error'] += 1
        return [], fun
    if got is None:
        fun['incomplete'] += 1
        return [], fun
    img, clus, op, cid2apa = got

    ix = np.array(img['x'], float)
    iy = np.array(img['y'], float)
    iz = np.array(img['z'], float)
    iq = np.array(img['q'], float)
    icid = np.array(img['cluster_id'], int)
    key = {}
    for i in range(len(ix)):
        k = round(iq[i], 4)
        key[k] = -1 if k in key else i

    cx = np.array(clus['x'], float)
    cy = np.array(clus['y'], float)
    cz = np.array(clus['z'], float)
    cq = np.array(clus['q'], float)
    ccid = np.array(clus['cluster_id'], int)

    members = collections.defaultdict(collections.Counter)
    for j in range(len(cx)):
        i = key.get(round(cq[j], 4), -1)
        if i >= 0:
            members[ccid[j]][icid[i]] += 1

    img_pts = {}
    for c in set(icid.tolist()):
        m = icid == c
        img_pts[c] = np.c_[ix[m], iy[m], iz[m]]

    matched, beam_matched = set(), set()
    for t, lst in zip(op['op_t'], op['op_cluster_ids']):
        matched.update(int(c) for c in lst)
        if BEAM_LO <= t < BEAM_HI:
            beam_matched.update(int(c) for c in lst)

    # does the other TPC have an in-beam flash that matched nothing?
    beam_empty_apa = set()
    for t, a, ids in zip(op['op_t'], op['apa'], op['op_cluster_ids']):
        if BEAM_LO <= t < BEAM_HI and not ids:
            beam_empty_apa.add(int(a))

    out = []
    for r in rows:
        gid = int(r['main_id'])
        fun['in-beam-main'] += 1
        cnt = members.get(gid)
        if not cnt:
            fun['main-not-in-clustering'] += 1
            continue
        kimg = set(cnt)                          # img clusters feeding this main

        m = ccid == gid
        KP = np.c_[cx[m], cy[m], cz[m]]
        if len(KP) < K_MIN_PTS:
            fun['main-too-few-points'] += 1
            continue

        hit_any = False
        for side in (0, 1):
            # the beam bundle carries |dx| <= 0.34 cm, so the corrected x sign is
            # the drift volume; no per-blob APA lookup is needed on K's own points
            P = KP[KP[:, 0] < 0] if side == 0 else KP[KP[:, 0] > 0]
            if len(P) < K_MIN_PTS:
                continue
            _, _, klen = pca(P)
            if klen < K_MIN_LEN:
                fun['side-too-short'] += 1
                continue
            la = local_axis(P, CATH[side])
            if la is None:
                fun['no-end-fit'] += 1
                continue
            end, d, dmin, rms = la
            if dmin > CATH_NEAR:
                fun['side-far-from-cathode'] += 1
                continue
            if rms > LIN_RMS:
                fun['cathode-end-not-track-like'] += 1
                continue
            if abs(d[0]) < DIRX_MIN:
                fun['isochronous'] += 1
                continue
            pk = pierce(end, d)
            if pk is None:
                continue
            yk, zk = pk
            fun['candidate-end'] += 1

            other = 1 - side
            lo, hi = ((ANODE[other], CATH[other]) if other == 0
                      else (CATH[other], ANODE[other]))
            found = 0
            for c, apc in cid2apa.items():
                if apc != other or c not in img_pts:
                    continue
                Q = img_pts[c]                   # RAW; no shift, t0 ~ 0
                if len(Q) < P_MIN_PTS:
                    continue
                if float(np.mean((Q[:, 0] >= lo) & (Q[:, 0] <= hi))) < P_INRANGE:
                    continue
                _, _, plen = pca(Q)
                if plen < P_MIN_LEN:
                    continue
                pa = local_axis(Q, CATH[other])
                if pa is None:
                    continue
                end2, d2, dmin2, rms2 = pa
                if dmin2 > P_CATH_NEAR or rms2 > LIN_RMS:
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
                ext, ncorr = corridor_extent(Q, end2, d2)
                kind = 'already-in-main' if c in kimg else 'MISSING'
                fun[kind] += 1
                found += 1
                out.append(dict(
                    event=evt, kind=kind, main_id=gid, side=side,
                    flash_t=float(r['flash_time_us']), label=r['label'],
                    k_npts=int(len(P)), k_len=klen,
                    k_len_nusel=float(r['len_main_cm']), k_dcath=dmin,
                    k_dirx=abs(d[0]), k_rms=rms, p_cid=int(c), p_npts=int(len(Q)),
                    p_len=plen, p_ext=ext, p_ncorr=ncorr, p_dcath=dmin2, p_rms=rms2,
                    p_matched=int(c in matched), p_beam=int(c in beam_matched),
                    gap2d=gap, angle=ang,
                    other_beam_flash_empty=int(other in beam_empty_apa),
                    y_cath=0.5 * (yk + y2), z_cath=0.5 * (zk + z2)))
            if found:
                hit_any = True
            else:
                fun['no-continuation-found'] += 1
        if not hit_any:
            fun['main-with-no-hit'] += 1

    # A main that DOES have a correct continuation somewhere is not a one-sided
    # track: its other cathode end fit picked a different local fragment and the
    # "missing" partner is an unrelated track.  Measured on evt407798, where main 25
    # carries already-in-main (gap 2.0 cm, 0.3 deg) on side 1 and a spurious MISSING
    # (gap 13.0 cm, 13.2 deg) on side 0.  Demoted, not deleted.
    good = {r['main_id'] for r in out if r['kind'] == 'already-in-main'}
    for r in out:
        if r['kind'] == 'MISSING' and r['main_id'] in good:
            r['kind'] = 'MISSING-main-also-crosses'
            fun['MISSING'] -= 1
            fun['MISSING-main-also-crosses'] += 1
    return out, fun


COLS = ['event', 'kind', 'main_id', 'side', 'flash_t', 'label', 'k_npts', 'k_len',
        'k_len_nusel', 'k_dcath', 'k_dirx', 'k_rms', 'p_cid', 'p_npts', 'p_len',
        'p_ext', 'p_ncorr', 'p_dcath', 'p_rms', 'p_matched', 'p_beam', 'gap2d', 'angle',
        'other_beam_flash_empty', 'y_cath', 'z_cath']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()

    nus = read_nusel(a.arm)
    dirs = sorted(glob.glob(os.path.join(a.arm, 'ql_evt*')))
    if a.limit:
        dirs = dirs[:a.limit]
    tasks = []
    for d in dirs:
        evt = int(re.search(r'ql_evt(\d+)$', d).group(1))
        tasks.append((d, evt, nus.get(evt, [])))

    rows, fun = [], collections.Counter()
    with Pool(a.jobs) as pool:
        for i, (rr, ff) in enumerate(pool.imap_unordered(analyse, tasks, chunksize=4)):
            rows.extend(rr)
            fun.update(ff)
            if (i + 1) % 250 == 0:
                print('  %d/%d' % (i + 1, len(tasks)), file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (r['kind'] != 'MISSING', -r['p_ext'], r['gap2d']))
    with open(a.out, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(('%.4g' % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in COLS) + '\n')
    print('funnel: %s' % dict(fun))
    miss = [r for r in rows if r['kind'] == 'MISSING']
    ctrl = [r for r in rows if r['kind'] == 'already-in-main']
    print('MISSING        : %d rows over %d events (%d with an empty in-beam flash '
          'in the other TPC)' % (len(miss), len({r['event'] for r in miss}),
                                 len({r['event'] for r in miss
                                      if r['other_beam_flash_empty']})))
    print('already-in-main: %d rows over %d events   <- control'
          % (len(ctrl), len({r['event'] for r in ctrl})))
    print('wrote %s' % a.out)


if __name__ == '__main__':
    main()

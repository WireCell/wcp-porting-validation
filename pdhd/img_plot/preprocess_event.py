#!/usr/bin/env python
"""Consolidate one PDHD event's imaging results into a compact viewer artifact.

Inputs (per event):
  * wire geometry         protodunehd-wires-larsoft-v1.json.bz2
  * imaging cluster files clusters-apa-apa{0..3}-ms-active.tar.gz   (blob bounds)
  * stepped Bee points    mabc-all-apa.zip : 0-clustering-group02/group13.json

Output: <out>.npz  (numeric arrays) + <out>.json (scalars / metadata / gate results)

Sampling points come from the toolkit "stepped" BlobSampler (threshold 0), as
written by MultiAlgBlobClustering's pre-clustering ``name:"img"`` Bee hook into
the per-drift-side instances ``0-clustering-group02.json`` (APAs 0+2, drift -x)
and ``0-clustering-group13.json`` (APAs 1+3, drift +x).  These are captured
BEFORE the clustering pipeline (PDHD/PDVD have no T0 correction, so
x == x_t0cor).

Frame: everything is stored in the MABC point convention (cm) so the stepped
points and the blob outlines share one plane.  The drift x is computed from the
blob slice time with the toolkit BlobSampler ``time2drift`` formula
(BlobSampler.cxx):
    x = xorig + xsign*(t + time_offset)*drift_speed
with time_offset = 0 (no preset T0; the clustering chain zeroed its -250 us
readout-tick0 compensation), drift_speed = 1.580 mm/us, xorig = the collection-plane
(W) wire-center x of that (anode,face), and xsign = anodeface->dirx() (resolved
empirically per (anode,face) against the points; +1 for APAs 0/2, -1 for 1/3).
Geometry y,z are mm/10 (units.cm == 10).  Blob start/span are in ns.

Two correctness gates run automatically:
  Gate 1  points-in-polygon: stepped points (matched to a slice by drift-x
          window) must fall inside a blob's Y-Z polygon -> validates the frame
          transform and the faceid/WIP geometry.
  Gate 2  channel numbering: every wire-band channel must lie within the Magnify
          ROOT channel range for its anode (skipped with a warning if ROOTs absent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np

import wirecell.img.tap as tap
import geom as G

# BlobSampler time2drift constants (internal units: mm, ns), from
# cfg/pgrapher/experiment/pdhd/clus.jsonnet
DRIFT_MM_PER_NS = 1.580 / 1000.0     # 1.580 mm/us (PDHD cathode-end centered, n=4 evt-983 crossers; was 1.585, 1.565, 1.6)
TIME_OFFSET_NS = 0.0                 # no preset T0 (was -250 * us)
TICK_NS = 500.0


def time2drift_cm(xorig_cm, xsign, t_ns):
    """Toolkit BlobSampler x convention (cm) for a slice time t_ns."""
    return xorig_cm + xsign * (t_ns + TIME_OFFSET_NS) * DRIFT_MM_PER_NS * G.MM2CM


def load_group_points(mabc_zip):
    """Read the stepped img-stage points from mabc-all-apa.zip.

    Returns parallel arrays x,y,z,q,cluster_id (cm / charge) and group (0 for
    APAs 0+2, 1 for APAs 1+3), plus the bee RSE metadata.
    """
    group_members = {0: "0-clustering-group02.json",
                     1: "0-clustering-group13.json"}
    px, py, pz, pq, pcid, pgrp = [], [], [], [], [], []
    meta = {}
    with zipfile.ZipFile(mabc_zip) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for grp, fn in group_members.items():
            if fn not in names:
                print(f"[warn] {fn} not in {mabc_zip}", file=sys.stderr)
                continue
            d = json.loads(z.read(names[fn]))
            meta = dict(run=d.get("runNo"), subrun=d.get("subRunNo"),
                        event=d.get("eventNo"), geom=d.get("geom"))
            n = len(d["x"])
            px.extend(d["x"]); py.extend(d["y"]); pz.extend(d["z"])
            pq.extend(d["q"]); pcid.extend(d["cluster_id"])
            pgrp.extend([grp] * n)
            print(f"[pts] group {grp}: {n} stepped points from {fn}")
    return (np.asarray(px), np.asarray(py), np.asarray(pz), np.asarray(pq),
            np.asarray(pcid, dtype=np.int64), np.asarray(pgrp, dtype=np.int8), meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--wires",
                    default="/nfs/data/1/xqian/toolkit-dev/wire-cell-data/"
                            "protodunehd-wires-larsoft-v1.json.bz2")
    ap.add_argument("--clusters-dir",
                    default="/home/xqian/work/scratch_wcgpu1/toolkit-dev/toolkit/"
                            "pdhd/work/027409_0",
                    help="dir holding clusters-apa-apa{N}-ms-active.tar.gz "
                         "and mabc-all-apa.zip")
    ap.add_argument("--mabc-zip", default="",
                    help="mabc-all-apa.zip with the stepped img-stage points "
                         "(default: <clusters-dir>/mabc-all-apa.zip)")
    ap.add_argument("--magnify-template", default="",
                    help="Magnify ROOT path template with {anode} for Gate 2 "
                         "(empty = skip Gate 2)")
    ap.add_argument("--anodes", default="0,1,2,3")
    ap.add_argument("--out", default=os.path.join(here, "cache", "evt0.npz"))
    args = ap.parse_args()

    anodes = [int(a) for a in args.anodes.split(",")]
    mabc_zip = args.mabc_zip or os.path.join(args.clusters_dir, "mabc-all-apa.zip")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[geom] loading {args.wires}")
    store = G.load_store(args.wires)
    pg_cache = {}

    def plane_geom(anode, face, plane):
        key = (anode, face, plane)
        if key not in pg_cache:
            pg_cache[key] = G.PlaneGeom(store, anode, face, plane)
        return pg_cache[key]

    # ---- blob & band accumulators (x deferred until xsign resolved) --------
    b_anode, b_face, b_slice = [], [], []
    b_start, b_span, b_val = [], [], []
    poly_xy, poly_off = [], [0]
    band_blob, band_plane, band_chan, band_wip = [], [], [], []
    band_quad = []                       # (Nw,4,2)
    chmap = {}                           # (anode,plane) -> [min,max] channel

    for anode in anodes:
        cf = os.path.join(args.clusters_dir,
                          f"clusters-apa-apa{anode}-ms-active.tar.gz")
        if not os.path.isfile(cf):
            print(f"[warn] missing {cf}; skipping anode {anode}", file=sys.stderr)
            continue
        graphs = list(tap.load(cf))
        if not graphs:
            print(f"[warn] empty graph {cf}", file=sys.stderr)
            continue
        gr = graphs[0]
        nblob = 0
        for n, d in gr.nodes(data=True):
            if d.get("code") != "b":
                continue
            a_chk, face = G.faceid_to_anode_face(d["faceid"])
            if a_chk != anode:
                print(f"[gate] FAIL faceid->anode {a_chk} != filename {anode}",
                      file=sys.stderr)
                sys.exit(2)
            bi = len(b_anode)
            b_anode.append(anode); b_face.append(face)
            b_slice.append(int(d["sliceid"]))
            b_start.append(float(d["start"])); b_span.append(float(d["span"]))
            b_val.append(float(d.get("val", 0.0)))

            corners = np.asarray(d["corners"], dtype=float)
            yz = corners[:, 1:3] * G.MM2CM
            yz = G.order_polygon(yz)
            poly_xy.extend(yz.tolist())
            poly_off.append(len(poly_xy))

            bounds = d["bounds"]   # [U,V,W] {beg,end}
            for plane in (0, 1, 2):
                beg = int(bounds[plane]["beg"]); end = int(bounds[plane]["end"])
                pgm = plane_geom(anode, face, plane)
                for wip in range(beg, min(end, len(pgm.chans))):
                    band_blob.append(bi)
                    band_plane.append(plane)
                    band_wip.append(wip)
                    ch = pgm.channel(wip)
                    band_chan.append(ch)
                    band_quad.append(pgm.band_quad(wip))
                    lo, hi = chmap.get((anode, plane), (ch, ch))
                    chmap[(anode, plane)] = (min(lo, ch), max(hi, ch))
            nblob += 1
        print(f"[blob] anode {anode}: {nblob} blobs")

    b_anode = np.asarray(b_anode, dtype=np.int16)
    b_face = np.asarray(b_face, dtype=np.int16)
    b_start = np.asarray(b_start)
    b_span = np.asarray(b_span)
    poly_xy = np.asarray(poly_xy, dtype=np.float32)
    poly_off = np.asarray(poly_off, dtype=np.int64)

    # ---- stepped img-stage points -----------------------------------------
    pts_x, pts_y, pts_z, pts_q, pts_cid, pts_grp, bee_meta = \
        load_group_points(mabc_zip)

    # ---- resolve x convention (xorig, xsign) per (anode,face) -------------
    xconv = {}            # (anode,face) -> dict(xorig_cm, xsign, resid_cm, n)
    sign_arr = np.ones(b_anode.size)       # per-blob xsign
    xorig_arr = np.zeros(b_anode.size)     # per-blob xorig
    for af in sorted(set(zip(b_anode.tolist(), b_face.tolist()))):
        anode, face = af
        xorig = G.wplane_x_cm(store, anode, face)
        idx = np.where((b_anode == anode) & (b_face == face))[0]
        # fattest blob of this (anode,face)
        areas = [np.ptp(poly_xy[poly_off[bi]:poly_off[bi + 1], 0]) *
                 np.ptp(poly_xy[poly_off[bi]:poly_off[bi + 1], 1]) for bi in idx]
        bi = idx[int(np.argmax(areas))]
        poly = poly_xy[poly_off[bi]:poly_off[bi + 1]]
        grp = 0 if anode in (0, 2) else 1
        m = pts_grp == grp
        gx, gy, gz = pts_x[m], pts_y[m], pts_z[m]
        inx = [gx[i] for i in range(gx.size)
               if G.point_in_poly(gy[i], gz[i], poly)]
        if inx:
            med = float(np.median(inx))
            cands = {s: abs(time2drift_cm(xorig, s, b_start[bi]) - med)
                     for s in (+1, -1)}
            xsign = min(cands, key=cands.get)
            resid = cands[xsign]
        else:
            xsign = 1 if anode in (0, 2) else -1
            resid = None
        xconv[af] = dict(xorig_cm=float(xorig), xsign=int(xsign),
                         resid_cm=(None if resid is None else float(resid)),
                         n_points=len(inx))
        sign_arr[idx] = xsign
        xorig_arr[idx] = xorig

    # ---- per-blob drift x via time2drift ----------------------------------
    xa = time2drift_cm(xorig_arr, sign_arr, b_start)
    xb = time2drift_cm(xorig_arr, sign_arr, b_start + b_span)
    b_xlo = np.minimum(xa, xb)
    b_xhi = np.maximum(xa, xb)
    b_xc = 0.5 * (b_xlo + b_xhi)
    # x of the slice START -- the stepped sampler emits all of a slice's points
    # exactly at this x (one x per slice, 0.313 cm apart), so the viewer matches
    # points to a slice by |pts_x - x_start| < half the spacing.
    b_xstart = xa

    # ---- Gate 1: points-in-polygon ----------------------------------------
    gate1 = gate_points_in_poly(pts_x, pts_y, pts_z, pts_grp,
                                b_anode, b_xlo, b_xhi, poly_xy, poly_off)

    # ---- Gate 2: channel numbering ----------------------------------------
    gate2 = gate_channels(args.magnify_template, chmap)

    # ---- write ------------------------------------------------------------
    np.savez_compressed(
        args.out,
        blob_anode=b_anode,
        blob_face=b_face,
        blob_sliceid=np.asarray(b_slice, dtype=np.int32),
        blob_start_ns=b_start,
        blob_span_ns=b_span,
        blob_x_lo=b_xlo, blob_x_hi=b_xhi, blob_xc=b_xc,
        blob_x_start=b_xstart,
        blob_val=np.asarray(b_val),
        blob_poly_xy=poly_xy, blob_poly_off=poly_off,
        band_blob=np.asarray(band_blob, dtype=np.int64),
        band_plane=np.asarray(band_plane, dtype=np.int8),
        band_wip=np.asarray(band_wip, dtype=np.int32),
        band_channel=np.asarray(band_chan, dtype=np.int64),
        band_quad_yz=np.asarray(band_quad, dtype=np.float32),
        pts_x=pts_x, pts_y=pts_y, pts_z=pts_z,
        pts_q=pts_q, pts_cluster_id=pts_cid, pts_group=pts_grp,
    )
    sidecar = os.path.splitext(args.out)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump(dict(
            **bee_meta,
            anodes=anodes,
            n_blobs=int(b_anode.size),
            n_bands=int(len(band_blob)),
            n_points=int(pts_x.size),
            tick_ns=TICK_NS,
            drift_mm_per_ns=DRIFT_MM_PER_NS, time_offset_ns=TIME_OFFSET_NS,
            xconv={f"{a}_{fc}": v for (a, fc), v in sorted(xconv.items())},
            channel_map={f"{a}_{p}": list(map(int, mm))
                         for (a, p), mm in sorted(chmap.items())},
            gate1=gate1, gate2=gate2,
        ), f, indent=2)

    print(f"\n[out] {args.out}")
    print(f"[out] {sidecar}")
    worst = max((v["resid_cm"] for v in xconv.values()
                 if v["resid_cm"] is not None), default=None)
    print(f"[xconv] resolved {len(xconv)} (anode,face); worst sign residual "
          f"{worst:.2f} cm" if worst is not None else "[xconv] no points")
    print(f"[gate1] points-inside fraction: {gate1['inside_frac']:.3f} "
          f"(n={gate1['n_tested']})  -> {gate1['verdict']}")
    print(f"[gate2] {gate2['verdict']}")


def gate_points_in_poly(px, py, pz, pgrp, b_anode, b_xlo, b_xhi,
                        poly_xy, poly_off, sample=4000, pad=0.5):
    """Fraction of stepped points that fall inside some same-slice blob polygon."""
    if px.size == 0 or b_anode.size == 0:
        return dict(inside_frac=0.0, n_tested=0, verdict="SKIP (no data)")
    grp_of_blob = np.isin(b_anode, (1, 3)).astype(np.int8)
    rng = np.random.default_rng(0)
    idx = rng.choice(px.size, size=min(sample, px.size), replace=False)
    inside = 0
    for i in idx:
        g = pgrp[i]
        x, y, z = px[i], py[i], pz[i]
        cand = np.where((grp_of_blob == g) &
                        (b_xlo - pad <= x) & (x <= b_xhi + pad))[0]
        hit = False
        for bi in cand:
            poly = poly_xy[poly_off[bi]:poly_off[bi + 1]]
            if G.point_in_poly(y, z, poly):
                hit = True
                break
        inside += hit
    frac = inside / len(idx)
    verdict = "PASS" if frac >= 0.80 else "FAIL (check frame transform / geometry)"
    return dict(inside_frac=float(frac), n_tested=int(len(idx)), verdict=verdict)


def gate_channels(template, chmap):
    """Every band channel must be within the Magnify ROOT channel range."""
    if not template:
        return dict(verdict="SKIP (no --magnify-template)")
    try:
        import uproot
    except Exception as e:                                    # pragma: no cover
        return dict(verdict=f"SKIP (uproot unavailable: {e})")
    PFX = {0: "hu_", 1: "hv_", 2: "hw_"}
    FRAMES = ("raw", "gauss", "orig")    # first present wins
    problems = []
    checked = 0
    for (anode, plane), (cmin, cmax) in sorted(chmap.items()):
        path = template.format(anode=anode)
        if not os.path.isfile(path):
            return dict(verdict=f"SKIP (Magnify ROOT absent: {path})")
        try:
            with uproot.open(path) as f:
                keys = {k.split(';')[0] for k in f.keys()}
                name = next((f"{PFX[plane]}{fr}{anode}" for fr in FRAMES
                             if f"{PFX[plane]}{fr}{anode}" in keys), None)
                if name is None:
                    problems.append(f"a{anode}p{plane}: no {PFX[plane]}*{anode}")
                    continue
                h = f[name]
                off = int(round(h.to_numpy()[1][0]))
                nrow = h.values().shape[0]
                checked += 1
                if cmin < off or cmax >= off + nrow:
                    problems.append(
                        f"a{anode}p{plane}: bands [{cmin},{cmax}] outside "
                        f"ROOT [{off},{off + nrow})")
        except Exception as e:                                # pragma: no cover
            problems.append(f"a{anode}p{plane}: {e}")
    if not checked:
        return dict(verdict="SKIP (no usable Magnify ROOTs)")
    verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems[:6])
    return dict(verdict=verdict, n_checked=checked, problems=problems)


if __name__ == "__main__":
    main()

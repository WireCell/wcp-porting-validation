#!/usr/bin/env python
"""Consolidate one PDVD event's imaging results into a compact viewer artifact.

Inputs (per event):
  * wire geometry         protodunevd-wires-larsoft-v5.json.bz2
  * imaging cluster files clusters-apa-anode{0..7}-ms-active.tar.gz   (blob bounds)
  * Bee JSON points       {idx}-imaging-group0123.json / group4567.json (as-is)

Output: <out>.npz  (numeric arrays) + <out>.json (scalars / metadata / gate results)

Everything is stored in the **Bee frame** (cm): geometry mm / 10 for (y,z); the
drift x is undrifted exactly as ``wirecell-img bee-blobs`` did, using the per-
drift-side constants from build_v4_bee_evt0to4.sh:
    anodes 0-3 (bottom): speed = -1.56 mm/us, x0 = -341.5 cm, t0 = 0
    anodes 4-7 (top):    speed = +1.56 mm/us, x0 = +341.5 cm, t0 = 0
Blob start/span are in ns; internal length unit is mm (units.cm == 10).

Two correctness gates run automatically:
  Gate 1  points-in-polygon: Bee points (matched to a slice by drift-x window)
          must fall inside a blob's Y-Z polygon -> validates the mm->cm/undrift
          frame transform and the faceid/WIP geometry.
  Gate 2  channel numbering: every wire-band channel must lie within the Magnify
          ROOT channel range for its anode (skipped with a warning if ROOTs absent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import wirecell.img.tap as tap
import geom as G

# drift-side undrift constants (internal units: mm, ns ; cm == 10, us == 1000)
SPEED_MM_PER_NS = 1.56 * 1.0 / 1000.0     # 1.56 * mm/us
X0_MM = 341.5 * 10.0                       # 341.5 * cm
TICK_NS = 500.0


def drift_params(anode):
    """(speed_mm_per_ns, x0_mm) for an anode's drift side, matching bee-blobs."""
    if anode <= 3:
        return -SPEED_MM_PER_NS, -X0_MM
    return SPEED_MM_PER_NS, X0_MM


def undrift_x_cm(anode, t_ns):
    speed, x0 = drift_params(anode)
    return (x0 - speed * t_ns) * G.MM2CM


def load_bee(path):
    with open(path) as f:
        d = json.load(f)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--wires",
                    default="/nfs/data/1/xqian/toolkit-dev/wire-cell-data/"
                            "protodunevd-wires-larsoft-v5.json.bz2")
    ap.add_argument("--clusters-dir",
                    default="/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/work/039324_0",
                    help="dir holding clusters-apa-anode{N}-ms-active.tar.gz")
    ap.add_argument("--bee-dir",
                    default="/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/data/0",
                    help="dir holding {idx}-imaging-group0123/4567.json")
    ap.add_argument("--bee-idx", default="0",
                    help="filename prefix of the Bee group files (e.g. 0)")
    ap.add_argument("--magnify-template", default="",
                    help="Magnify ROOT path template with {anode} for Gate 2 "
                         "(empty = skip Gate 2)")
    ap.add_argument("--anodes", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default=os.path.join(here, "cache", "evt0.npz"))
    args = ap.parse_args()

    anodes = [int(a) for a in args.anodes.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[geom] loading {args.wires}")
    store = G.load_store(args.wires)
    # PlaneGeom cache, built lazily per (anode,face,plane)
    pg_cache = {}

    def plane_geom(anode, face, plane):
        key = (anode, face, plane)
        if key not in pg_cache:
            pg_cache[key] = G.PlaneGeom(store, anode, face, plane)
        return pg_cache[key]

    # ---- blob & band accumulators -----------------------------------------
    b_anode, b_face, b_slice = [], [], []
    b_start, b_span, b_xlo, b_xhi, b_xc, b_val = [], [], [], [], [], []
    poly_xy, poly_off = [], [0]
    band_blob, band_plane, band_chan = [], [], []
    band_quad = []                       # (Nw,4,2)
    chmap = {}                           # (anode,plane) -> [min,max] channel

    for anode in anodes:
        cf = os.path.join(args.clusters_dir,
                          f"clusters-apa-anode{anode}-ms-active.tar.gz")
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
            sliceid = int(d["sliceid"])
            start = float(d["start"])
            span = float(d["span"])
            xa = undrift_x_cm(anode, start)
            xb = undrift_x_cm(anode, start + span)
            xlo, xhi = (xa, xb) if xa <= xb else (xb, xa)

            bi = len(b_anode)
            b_anode.append(anode); b_face.append(face); b_slice.append(sliceid)
            b_start.append(start); b_span.append(span)
            b_xlo.append(xlo); b_xhi.append(xhi); b_xc.append(0.5 * (xlo + xhi))
            b_val.append(float(d.get("val", 0.0)))

            # blob polygon from precomputed corners (cols 1,2 = y,z in mm)
            corners = np.asarray(d["corners"], dtype=float)
            yz = corners[:, 1:3] * G.MM2CM
            yz = G.order_polygon(yz)
            poly_xy.extend(yz.tolist())
            poly_off.append(len(poly_xy))

            # per-fired-wire bands from the wire store
            bounds = d["bounds"]   # [U,V,W] {beg,end}
            for plane in (0, 1, 2):
                beg = int(bounds[plane]["beg"]); end = int(bounds[plane]["end"])
                pgm = plane_geom(anode, face, plane)
                for wip in range(beg, min(end, len(pgm.chans))):
                    band_blob.append(bi)
                    band_plane.append(plane)
                    ch = pgm.channel(wip)
                    band_chan.append(ch)
                    band_quad.append(pgm.band_quad(wip))
                    lo, hi = chmap.get((anode, plane), (ch, ch))
                    chmap[(anode, plane)] = (min(lo, ch), max(hi, ch))
            nblob += 1
        print(f"[blob] anode {anode}: {nblob} blobs")

    # ---- Bee points (as-is) -----------------------------------------------
    pts_x, pts_y, pts_z, pts_q, pts_cid, pts_grp = [], [], [], [], [], []
    group_files = {0: f"{args.bee_idx}-imaging-group0123.json",
                   1: f"{args.bee_idx}-imaging-group4567.json"}
    bee_meta = {}
    for grp, fn in group_files.items():
        path = os.path.join(args.bee_dir, fn)
        if not os.path.isfile(path):
            print(f"[warn] missing Bee file {path}", file=sys.stderr)
            continue
        bee = load_bee(path)
        bee_meta = dict(run=bee.get("runNo"), subrun=bee.get("subRunNo"),
                        event=bee.get("eventNo"), geom=bee.get("geom"))
        npt = len(bee["x"])
        pts_x.extend(bee["x"]); pts_y.extend(bee["y"]); pts_z.extend(bee["z"])
        pts_q.extend(bee["q"]); pts_cid.extend(bee["cluster_id"])
        pts_grp.extend([grp] * npt)
        print(f"[bee] group {grp}: {npt} points from {fn}")

    pts_x = np.asarray(pts_x); pts_y = np.asarray(pts_y); pts_z = np.asarray(pts_z)
    pts_grp = np.asarray(pts_grp, dtype=np.int8)

    b_anode = np.asarray(b_anode, dtype=np.int16)
    b_xlo = np.asarray(b_xlo); b_xhi = np.asarray(b_xhi)
    poly_xy = np.asarray(poly_xy, dtype=np.float32)
    poly_off = np.asarray(poly_off, dtype=np.int64)

    # ---- Gate 1: points-in-polygon ----------------------------------------
    gate1 = gate_points_in_poly(pts_x, pts_y, pts_z, pts_grp,
                                b_anode, b_xlo, b_xhi, poly_xy, poly_off)

    # ---- Gate 2: channel numbering ----------------------------------------
    gate2 = gate_channels(args.magnify_template, chmap)

    # ---- write ------------------------------------------------------------
    np.savez_compressed(
        args.out,
        blob_anode=b_anode,
        blob_face=np.asarray(b_face, dtype=np.int16),
        blob_sliceid=np.asarray(b_slice, dtype=np.int32),
        blob_start_ns=np.asarray(b_start),
        blob_span_ns=np.asarray(b_span),
        blob_x_lo=b_xlo, blob_x_hi=b_xhi, blob_xc=np.asarray(b_xc),
        blob_val=np.asarray(b_val),
        blob_poly_xy=poly_xy, blob_poly_off=poly_off,
        band_blob=np.asarray(band_blob, dtype=np.int64),
        band_plane=np.asarray(band_plane, dtype=np.int8),
        band_channel=np.asarray(band_chan, dtype=np.int64),
        band_quad_yz=np.asarray(band_quad, dtype=np.float32),
        pts_x=pts_x, pts_y=pts_y, pts_z=pts_z,
        pts_q=np.asarray(pts_q), pts_cluster_id=np.asarray(pts_cid, dtype=np.int64),
        pts_group=pts_grp,
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
            speed_mm_per_ns=SPEED_MM_PER_NS, x0_mm=X0_MM,
            channel_map={f"{a}_{p}": list(map(int, mm))
                         for (a, p), mm in sorted(chmap.items())},
            gate1=gate1, gate2=gate2,
        ), f, indent=2)

    print(f"\n[out] {args.out}")
    print(f"[out] {sidecar}")
    print(f"[gate1] points-inside fraction: {gate1['inside_frac']:.3f} "
          f"(n={gate1['n_tested']})  -> {gate1['verdict']}")
    print(f"[gate2] {gate2['verdict']}")


def gate_points_in_poly(px, py, pz, pgrp, b_anode, b_xlo, b_xhi,
                        poly_xy, poly_off, sample=4000, pad=0.5):
    """Fraction of Bee points that fall inside some same-slice blob polygon."""
    if px.size == 0 or b_anode.size == 0:
        return dict(inside_frac=0.0, n_tested=0, verdict="SKIP (no data)")
    # group 0 -> anodes 0-3, group 1 -> anodes 4-7
    grp_of_blob = (b_anode > 3).astype(np.int8)
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
    PFX = {0: "hu_raw", 1: "hv_raw", 2: "hw_raw"}
    problems = []
    checked = 0
    for (anode, plane), (cmin, cmax) in sorted(chmap.items()):
        path = template.format(anode=anode)
        if not os.path.isfile(path):
            return dict(verdict=f"SKIP (Magnify ROOT absent: {path})")
        try:
            with uproot.open(path) as f:
                keys = {k.split(';')[0] for k in f.keys()}
                name = next((f"{PFX[plane]}{anode}" for _ in [0]
                             if f"{PFX[plane]}{anode}" in keys), None)
                if name is None:
                    problems.append(f"a{anode}p{plane}: no {PFX[plane]}{anode}")
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

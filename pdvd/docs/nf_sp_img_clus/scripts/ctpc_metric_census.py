#!/usr/bin/env python3
"""doc pdvd/34 -- the ctpc's two axes are not the same size.

The ctpc is an exact 2-D lattice whose two cell edges are set by DIFFERENT
detector quantities:

    PointTreeBuilding.cxx:312   x = time2drift(slice->start())      <- drift step
                                                                       = nticks_live_slice * tick * drift_speed
    PointTreeBuilding.cxx:313   y = pitch * (wind + 0.5) + center   <- wire pitch

Every distance taken in that cloud today is plain isotropic Euclidean in mm
(`Grouping::get_closest_points` / `has_closest_point`, Facade_Grouping.cxx:680
and :693, both `skd.<query>(radius*radius, {x, y})`).  That is harmless where
the two cell edges are nearly equal -- SBND 3.126 x 3.000 mm -- and it is not
harmless on PDVD, where U/V are 2.9615 x 7.650 mm, a 2.58:1 cell.

This script measures the consequence and grades the proposed fix.  The
anisotropic metric scales ONLY the pitch axis,

    s = min(1, drift_step / pitch)          per (apa, face, plane)
    d^2 = dx^2 + (s * dy)^2

so a radius keeps its exact drift-axis meaning and only the pitch tolerance
widens.  The clamp at 1 means the metric can only ever RELAX a plane coarser
than the drift step, never tighten a finer one; with it SBND (s = 1.042 -> 1.0)
is bit-identical even with the knob on.

Compared arms, all at the same query points:

  iso r          the legacy isotropic radius, what ships today off-PDVD
  frac F         doc 32's shipped floor, r = max(radius, F * pitch), isotropic
  aniso r        this proposal

Query points are INTERIOR trajectory points -- the first and last `--edge` of
each polyline are dropped, because those are exactly the points the
`examine_end_ps_vec` pop loop accepted, and including them measures
survivorship instead of geometry (doc 32 sec 10).

Everything is read-only: existing arms are opened, nothing is written or
regenerated (M13).  The `Detector` class is imported from doc 32's
`goodpoint_pitch_census.py` so the numbers here are directly comparable with
that document's.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/ctpc_metric_census.py \
      PDVD:work/039252_2_d31r6e2e \
      SBND:../sbnd/sbnd_xin/work-mcp2k-d97fvpr2/pr_evt100002
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goodpoint_pitch_census import (Detector, trajectories, lattice_model,   # noqa: E402
                                    PLANES)

CM = 10.0                      # WCT internal length unit is mm


def find_dump(workdir):
    """The pctree dump of an arm, whatever the runner named it."""
    for pat in ("pctree-pr-evt*.tar.gz", "pctree-evt*.tar.gz", "pctree*.tar.gz"):
        hits = sorted(glob.glob(os.path.join(workdir, pat)))
        if hits:
            return hits[0]
    return None


def scale_of(det, key):
    """s = min(1, drift_step/pitch) for one (apa, face, plane), or 1.0."""
    p, dx = det.pitch.get(key, np.nan), det.dx.get(key, np.nan)
    if not (np.isfinite(p) and np.isfinite(dx)) or p <= 0:
        return 1.0
    return float(min(1.0, dx / p))


def scaled_trees(det):
    """A cKDTree per (apa,face,plane) with the pitch axis scaled by s."""
    trees, svec = {}, {}
    for key, t in det.tree.items():
        s = scale_of(det, key)
        svec[key] = s
        pts = t.data.copy()
        pts[:, 1] *= s
        trees[key] = cKDTree(pts)
    return trees, svec


# ------------------------------------------------------------------ geometry

def geometry_table(det, tag):
    """Per-plane lattice constants, and the radius each metric needs.

    Worst-case ("corner") coverage of every lattice phase needs a radius
    reaching the far corner of a cell from its centre:

        isotropic     r >= sqrt((dx/2)^2 + (pitch/2)^2)
        anisotropic   r >= sqrt((dx/2)^2 + (s*pitch/2)^2) = dx/sqrt(2)

    because with s = dx/pitch both terms collapse to dx/2.  The anisotropic
    requirement is therefore a property of the DRIFT STEP ALONE -- the same
    number on every plane of a detector, and nearly the same number on every
    detector.  That is the whole argument in one line.
    """
    print("\n### %s -- lattice constants and the radius each metric needs" % tag)
    print("%-6s %10s %10s %8s %12s %12s" % (
        "plane", "drift/mm", "pitch/mm", "ratio", "iso r/mm", "aniso r/mm"))
    seen = {}
    for key in sorted(det.pitch):
        seen.setdefault(key[2], []).append(key)
    for p in sorted(seen):
        keys = seen[p]
        dx = float(np.nanmedian([det.dx[k] for k in keys]))
        pit = float(np.nanmedian([det.pitch[k] for k in keys]))
        if not (np.isfinite(dx) and np.isfinite(pit)):
            continue
        s = min(1.0, dx / pit)
        r_iso = math.hypot(dx / 2.0, pit / 2.0)
        r_ani = math.hypot(dx / 2.0, s * pit / 2.0)
        print("%-6s %10.4f %10.4f %8.3f %12.3f %12.3f" % (
            PLANES[p], dx, pit, pit / dx, r_iso, r_ani))


# ------------------------------------------------------------------ the arms

def arm_defs(radius, fracs):
    arms = [("iso r=%.2fcm" % (radius / CM), ("iso", radius))]
    for f in fracs:
        arms.append(("iso frac %.2f" % f, ("frac", f)))
    arms.append(("aniso r=%.2fcm" % (radius / CM), ("ani", radius)))
    return arms


def measure(det, workdir, arms, edge, sources, count_candidates=False):
    """Per-plane and all-three pass masks for every arm, over interior points.

    Returns (npoints, {arm: bool[n,3]}, inflation) where `inflation` is the
    mean number of ctpc points the circumscribing-circle query must enumerate
    per (point, plane) against the mean number the ellipse actually accepts --
    the cost bound for the two-round implementation.
    """
    strees, svec = scaled_trees(det)
    acc = {nm: [] for nm, _ in arms}
    cand_out, cand_in, ncand = 0.0, 0.0, 0
    used = {}
    for src, _cid, P in trajectories(workdir):
        if src not in sources or len(P) < 2 * edge + 3:
            continue
        used[src] = used.get(src, 0) + 1
        P = P[edge:-edge]
        shift, _ = det.offset(P)
        _, i = det.ktree.query(P)
        apa, face = det.apa[i], det.face[i]
        xraw = P[:, 0] + shift
        res = {nm: np.zeros((len(P), 3), bool) for nm, _ in arms}
        for vol in np.unique(apa * 2 + face):
            aa, ff = int(vol) // 2, int(vol) % 2
            m = (apa == aa) & (face == ff)
            for p in range(3):
                key = (aa, ff, p)
                if key not in det.tree or key not in det.proj:
                    continue
                sol = det.proj[key]
                yp = sol[0] * P[m, 1] + sol[1] * P[m, 2] + sol[2]
                s = svec[key]
                d_iso, _ = det.tree[key].query(np.stack([xraw[m], yp], 1))
                d_ani, _ = strees[key].query(np.stack([xraw[m], yp * s], 1))
                for nm, (kind, val) in arms:
                    if kind == "iso":
                        res[nm][m, p] = d_iso < val
                    elif kind == "frac":
                        res[nm][m, p] = d_iso < max(arms[0][1][1], val * det.pitch[key])
                    else:
                        res[nm][m, p] = d_ani < val
                if count_candidates and s < 1.0:
                    r = arms[-1][1][1]
                    q = np.stack([xraw[m], yp], 1)
                    # circumscribing circle of the ellipse (semi-axes r, r/s)
                    out = det.tree[key].query_ball_point(q, r / s)
                    cand_out += sum(len(o) for o in out)
                    inn = strees[key].query_ball_point(
                        np.stack([xraw[m], yp * s], 1), r)
                    cand_in += sum(len(o) for o in inn)
                    ncand += len(q)
        for nm, _ in arms:
            acc[nm].append(res[nm])
    if not acc[arms[0][0]]:
        return 0, {}, None, used
    out = {nm: np.concatenate(v) for nm, v in acc.items()}
    infl = None
    if ncand:
        infl = (cand_out / ncand, cand_in / ncand)
    return len(out[arms[0][0]]), out, infl, used


def ceilings(det, arms):
    """Uniform-phase upper bound per (arm, plane): the best any event can give.

    doc 32's `lattice_model(pitch, dxs, radius)` is the fraction of a cell's
    area covered by a disc of that radius when EVERY neighbouring cell is
    occupied.  Under the anisotropic metric the cell is (dxs x s*pitch) =
    (dxs x dxs), so the same model applies with the scaled pitch.  Comparing
    the measured rate against this bound is what separates "limited by the
    lattice" from "limited by the charge" (doc 32 sec 11).
    """
    out = {}
    per_plane = {}
    for key in sorted(det.pitch):
        per_plane.setdefault(key[2], []).append(key)
    for nm, (kind, val) in arms:
        row = []
        for p in sorted(per_plane):
            keys = per_plane[p]
            dxs = float(np.nanmedian([det.dx[k] for k in keys]))
            pit = float(np.nanmedian([det.pitch[k] for k in keys]))
            if not (np.isfinite(dxs) and np.isfinite(pit)):
                row.append(float("nan"))
                continue
            if kind == "ani":
                row.append(lattice_model(min(1.0, dxs / pit) * pit, dxs, val))
            elif kind == "frac":
                row.append(lattice_model(pit, dxs, max(arms[0][1][1], val * pit)))
            else:
                row.append(lattice_model(pit, dxs, val))
        out[nm] = row
    return out


def report(tag, n, masks, arms, infl, used, ceil):
    src = ", ".join("%s x%d" % (k, v) for k, v in sorted(used.items())) or "none"
    print("\n### %s -- interior trajectory points, n = %d  (%s)" % (tag, n, src))
    if not n:
        print("    (no trajectories of the requested source in this arm)")
        return
    print("%-22s %8s   %-21s %s" % ("arm", "all-3", "U      V      W", "ceiling U/V/W"))
    for nm, _ in arms:
        A = masks[nm]
        print("%-22s %8.3f   %-21s %s" % (
            nm, A.all(1).mean(),
            "  ".join("%.3f" % A[:, p].mean() for p in range(3)),
            "  ".join("%.3f" % c for c in ceil[nm])))
    if infl:
        print("  two-round cost: circumscribing circle enumerates %.2f ctpc points "
              "per (point, plane); the ellipse accepts %.2f  -> %.2fx" % (
                  infl[0], infl[1], infl[0] / max(infl[1], 1e-9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="TAG:workdir",
                    help="e.g. PDVD:work/039252_2_d31r6e2e")
    ap.add_argument("--radius", type=float, default=2.0,
                    help="query radius in mm (default 2.0 = the 0.2 cm the PR chain passes)")
    ap.add_argument("--frac", type=float, nargs="*", default=[0.35, 0.50],
                    help="good_point_pitch_frac values to compare (default 0.35 0.50)")
    ap.add_argument("--edge", type=int, default=3,
                    help="trajectory points dropped at each end (survivorship, default 3)")
    ap.add_argument("--source", nargs="*", default=["stm_fit", "segment"],
                    help="trajectory sources: stm_fit (Bee mabc layer) and/or "
                         "segment (calib dump). Default both -- PDVD arms carry "
                         "stm_fit, SBND arms carry segment.")
    ap.add_argument("--candidates", action="store_true",
                    help="also measure the two-round enumeration cost")
    args = ap.parse_args()

    arms = arm_defs(args.radius, args.frac)
    print("radius = %.2f mm; anisotropic scale s = min(1, drift_step/pitch) per plane" % args.radius)

    for spec in args.specs:
        tag, workdir = spec.split(":", 1)
        dump = find_dump(workdir)
        if dump is None:
            print("\n### %s -- no pctree dump under %s, skipped" % (tag, workdir))
            continue
        det = Detector(dump)
        print("\n%s  evt %s  (%s)" % ("=" * 60, det.evt, dump))
        worst = max(det.projres.values()) if det.projres else float("nan")
        print("projection refit gate: max |residual| = %.3g mm (must be ~1e-12, not 'small')" % worst)
        geometry_table(det, tag)
        n, masks, infl, used = measure(det, workdir, arms, args.edge,
                                       set(args.source), args.candidates)
        report(tag, n, masks, arms, infl, used, ceilings(det, arms))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/32 round 2 -- is `is_good_point(..., 0.2 cm, 0, 0)` pitch-blind?

Doc 32 pinned the STM trajectory's missing track ends on `examine_end_ps_vec`
(TrackFitting.cxx:2275), whose pop loop drops a trajectory point unless ALL
THREE planes have a ctpc hit within 0.2 cm.  It left the *reason* as an
explicitly unproven hypothesis: 2 x 0.2 cm spans 1.33 of SBND's 0.300 cm wire
pitch but only 0.52 of PDVD's 0.765 cm.  This script settles it.

The measurement is possible offline because the ctpc is an exact lattice:

    PointTreeBuilding.cxx:313   y = pitch * (wind + 0.5) + proj_center
    PointTreeBuilding.cxx:312   x = time2drift(slice->start())

so `Grouping::has_closest_point` (Facade_Grouping.cxx:687) is a 2-D nearest-
neighbour query against occupied cells of a (drift-step x pitch) grid.  A
trajectory point lands at an arbitrary phase in that grid; if half the pitch
exceeds the radius there is a band of phases that CANNOT match, whatever the
charge.

What is replicated, and how each piece is checked:

  * the projection y_proj = cos(a)*z - sin(a)*y is refitted from the dump's own
    `2dp{p}_y` arrays against (y, z).  It is linear by construction, so the
    max |residual| is printed as a gate -- it must be ~1e-12 mm, not "small".
  * the ctpc (x, y) arrays are used as-is, so pitch and drift step are measured
    from the data rather than assumed.
  * the corrected->raw x offset is read off the 3d point cloud (`x` minus
    `x_t0cor` at the nearest point, mode over the trajectory).  The pctree dump
    is written at the CLUSTERING stage, so its cluster idents do not match the
    PR/Bee ids and `cluster_scalar/cluster_t0` cannot be indexed by them; the
    printed purity is the fraction of trajectory points agreeing on the offset.

Query points are INTERIOR trajectory points -- the first and last `--edge` of
each polyline are dropped, because those are exactly the points the pop loop
accepted, and including them would measure survivorship instead of geometry.
Everything else on a fitted trajectory was never subjected to the test.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/goodpoint_pitch_census.py \
      PDVD:work/039252_2_d31r6e2e \
      SBND:../sbnd/sbnd_xin/work-mcp2k-d97fvpr2/pr_evt100002 \
      SBND:../sbnd/sbnd_xin/work-mcp2k-d97fvpr2/pr_evt100032
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree

PLANES = ["U", "V", "W"]
CM = 10.0                      # WCT internal length unit is mm
RADII_CM = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]


# ----------------------------------------------------------------- loading

def load_pct(tgz, want):
    """{datapath: ndarray} for the pcarray tensors whose datapath passes `want`."""
    out = {}
    with tarfile.open(tgz, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for name, m in members.items():
            if not name.endswith("_metadata.json"):
                continue
            md = json.load(io.BytesIO(tf.extractfile(m).read()))
            if md.get("datatype") != "pcarray":
                continue
            dp = md.get("datapath", "")
            if not want(dp):
                continue
            an = name.replace("_metadata.json", "_array.npy")
            if an in members:
                out[dp] = np.load(io.BytesIO(tf.extractfile(members[an]).read()))
    return out


class Detector:
    """The ctpc lattices and projection maps of one event's pctree dump."""

    def __init__(self, tgz):
        with tarfile.open(tgz, "r:gz") as tf:
            first = tf.next().name
        evt = re.search(r"_(\d+)_", first + "_").group(1)
        B = f"pointtrees/{evt}/live/"
        keep = ("x", "x_t0cor", "y", "z", "wpid",
                "2dp0_y", "2dp1_y", "2dp2_y")
        d = load_pct(tgz, lambda p: p.startswith(B) and (
            "/namedpcs/ctpc_" in p
            or ("/namedpcs/3d/arrays/" in p and p.rsplit("/", 1)[-1] in keep)))
        g = lambda n: d[B + "pointclouds/namedpcs/3d/arrays/" + n]
        self.evt = evt
        self.x, self.xc = g("x"), g("x_t0cor")
        self.y, self.z = g("y"), g("z")
        wpid = g("wpid")
        self.apa, self.face = wpid >> 4, (wpid >> 3) & 1   # WirePlaneId.cxx:36,37

        self.proj, self.projres = {}, {}
        for key in np.unique(self.apa * 2 + self.face):
            a, f = int(key) // 2, int(key) % 2
            m = (self.apa == a) & (self.face == f)
            if m.sum() < 100:
                continue
            A = np.stack([self.y[m], self.z[m], np.ones(int(m.sum()))], 1)
            for p in range(3):
                sol, *_ = np.linalg.lstsq(A, g(f"2dp{p}_y")[m], rcond=None)
                self.proj[(a, f, p)] = sol
                self.projres[(a, f, p)] = float(np.abs(A @ sol - g(f"2dp{p}_y")[m]).max())

        self.tree, self.pitch, self.dx = {}, {}, {}
        pref = B + "pointclouds/namedpcs/ctpc_"
        for nm in sorted({k[len(pref):].split("/")[0] for k in d if k.startswith(pref)}):
            a = int(nm[1:nm.index("f")])
            f = int(nm[nm.index("f") + 1:nm.index("p")])
            p = PLANES.index(nm[-1])
            xs, ys = d[pref + nm + "/arrays/x"], d[pref + nm + "/arrays/y"]
            if len(xs) < 3:
                continue
            self.tree[(a, f, p)] = cKDTree(np.stack([xs, ys], 1))
            uy, ux = np.unique(np.round(ys, 4)), np.unique(np.round(xs, 4))
            self.pitch[(a, f, p)] = float(np.median(np.diff(uy))) if len(uy) > 2 else np.nan
            self.dx[(a, f, p)] = float(np.median(np.diff(ux))) if len(ux) > 2 else np.nan

        self.ktree = cKDTree(np.stack([self.xc, self.y, self.z], 1))

    def offset(self, P):
        """(raw-minus-corrected x, purity) for a trajectory, from the 3d PC."""
        _, i = self.ktree.query(P)
        sh = np.round(self.x[i] - self.xc[i], 3)
        vals, cnt = np.unique(sh, return_counts=True)
        return float(vals[np.argmax(cnt)]), float(cnt.max() / len(sh))

    def measure(self, P):
        """Per-plane ctpc distance (mm) for corrected-frame points P.

        Returns (dist[n,3], apa[n], face[n], phase_rate[n,3]); phase_rate is
        d(pitch coordinate)/d(path length), i.e. how fast the lattice phase
        advances along the trajectory -- 0 means a frozen phase.
        """
        shift, _ = self.offset(P)
        _, i = self.ktree.query(P)
        a, f = self.apa[i], self.face[i]
        xraw = P[:, 0] + shift
        dist = np.full((len(P), 3), np.nan)
        rate = np.full((len(P), 3), np.nan)
        step = np.gradient(P, axis=0)
        step /= np.maximum(np.linalg.norm(step, axis=1), 1e-9)[:, None]
        for key in np.unique(a * 2 + f):
            aa, ff = int(key) // 2, int(key) % 2
            m = (a == aa) & (f == ff)
            for p in range(3):
                if (aa, ff, p) not in self.tree or (aa, ff, p) not in self.proj:
                    continue
                sol = self.proj[(aa, ff, p)]
                yp = sol[0] * P[m, 1] + sol[1] * P[m, 2] + sol[2]
                dd, _ = self.tree[(aa, ff, p)].query(np.stack([xraw[m], yp], 1))
                dist[m, p] = dd
                rate[m, p] = np.abs(sol[0] * step[m, 1] + sol[1] * step[m, 2])
        return dist, a, f, rate


# ------------------------------------------------------------- trajectories

def trajectories(workdir):
    """Yield (source, cluster_id, points[n,3] in mm, corrected frame)."""
    zips = glob.glob(os.path.join(workdir, "mabc-pr.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            if "data/0/0-stm_fit-global.json" in z.namelist():
                b = json.loads(z.read("data/0/0-stm_fit-global.json"))
                cid = np.asarray(b["cluster_id"])
                P = np.stack([b["x"], b["y"], b["z"]], 1).astype(float) * CM
                for c in sorted(set(cid.tolist())):
                    yield "stm_fit", int(c), P[cid == c]
    for cal in sorted(glob.glob(os.path.join(workdir, "calib-pr-evt*.json"))):
        d = json.load(open(cal))
        for s in d.get("segments", []):
            pts = s.get("points") or []
            if len(pts) < 4:
                continue
            P = np.array([[q["x"], q["y"], q["z"]] for q in pts], float) * CM
            yield "segment", int(s.get("cluster_id", -1)), P


# ------------------------------------------------------------------- model

def lattice_model(pitch, dxs, radius):
    """Best-case per-plane pass rate for a uniform phase on a (dxs x pitch) grid.

    A point is covered iff some occupied cell is within `radius`; with every
    neighbouring cell occupied this is the fraction of the cell's area inside a
    disc of that radius, i.e. an UPPER bound on what any real event can give.
    """
    if not np.isfinite(pitch) or not np.isfinite(dxs):
        return np.nan
    hy, hx = pitch / 2.0, dxs / 2.0
    ys = np.linspace(0, hy, 4001)
    cov = np.zeros_like(ys)
    ok = ys < radius
    cov[ok] = np.minimum(1.0, np.sqrt(radius ** 2 - ys[ok] ** 2) / hx)
    return float(np.trapezoid(cov, ys) / hy)


def runs_of_false(mask):
    """Lengths of maximal runs of False in a 1-D boolean array."""
    out, n = [], 0
    for v in mask:
        if v:
            if n:
                out.append(n)
            n = 0
        else:
            n += 1
    if n:
        out.append(n)
    return out


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="TAG:workdir")
    ap.add_argument("--edge", type=int, default=3,
                    help="trajectory points dropped at each end (survivorship)")
    ap.add_argument("--minpts", type=int, default=12)
    ap.add_argument("--extend", type=float, default=200.0,
                    help="mm to extend past each trajectory end (0 disables)")
    ap.add_argument("--near", type=float, default=15.0,
                    help="mm: an extension point counts only if a 3d point is "
                         "this close, i.e. only where charge demonstrably is")
    args = ap.parse_args()

    pooled = {}
    for spec in args.arms:
        tag, wd = spec.split(":", 1)
        tgz = sorted(glob.glob(os.path.join(wd, "pctree*.tar.gz")))
        if not tgz:
            print(f"  skip {spec}: no pctree dump", file=sys.stderr)
            continue
        det = Detector(tgz[0])
        acc = pooled.setdefault(tag, {"d": [], "rate": [], "runs": [], "res": 0.0,
                                      "pitch": {}, "dx": {}, "ncl": 0, "src": set(),
                                      "ext": [], "extlen": []})
        acc["res"] = max(acc["res"], max(det.projres.values()) if det.projres else 0.0)
        for k, v in det.pitch.items():
            acc["pitch"].setdefault(round(v, 4), 0)
            acc["pitch"][round(v, 4)] += 1
        for k, v in det.dx.items():
            acc["dx"].setdefault(round(v, 4), 0)
            acc["dx"][round(v, 4)] += 1
        for src, cid, P in trajectories(wd):
            if len(P) < args.minpts:
                continue
            e = args.edge
            Q = P[e:-e] if len(P) > 2 * e + 2 else P
            dist, a, f, rate = det.measure(Q)
            if not np.isfinite(dist).any():
                continue
            acc["d"].append(dist)
            acc["rate"].append(rate)
            acc["runs"].append((cid, dist))
            acc["ncl"] += 1
            acc["src"].add(src)

            # The amputated extension: walk PAST each end along the local
            # direction and keep only steps that have charge within --near.
            # Those points are, by construction, places where charge exists and
            # the fit does not -- exactly the region examine_end_ps_vec removed.
            if args.extend > 0 and len(P) >= 8:
                for end, sgn in ((0, -1.0), (len(P) - 1, +1.0)):
                    seg = P[:8] if end == 0 else P[-8:]
                    d3 = seg[0] - seg[-1] if end == 0 else seg[-1] - seg[0]
                    n3 = np.linalg.norm(d3)
                    if n3 < 1e-6:
                        continue
                    d3 = d3 / n3
                    steps = np.arange(6.0, args.extend + 1e-9, 6.0)
                    E = P[end] + np.outer(steps, d3)
                    dnear, _ = det.ktree.query(E)
                    keep = dnear <= args.near
                    if keep.sum() < 2:
                        continue
                    # contiguous run from the tip only: past the first gap the
                    # extension has left the track
                    stop = int(np.argmax(~keep)) if (~keep).any() else len(keep)
                    if stop < 2:
                        continue
                    ed, _, _, _ = det.measure(E[:stop])
                    acc["ext"].append(ed)
                    acc["extlen"].append(steps[stop - 1])

    for tag, acc in pooled.items():
        D = np.concatenate(acc["d"]) if acc["d"] else np.zeros((0, 3))
        R = np.concatenate(acc["rate"]) if acc["rate"] else np.zeros((0, 3))
        pitch = sorted(acc["pitch"])
        dxs = sorted(acc["dx"])
        print(f"\n=== {tag}: {acc['ncl']} trajectories, {len(D)} interior points"
              f"  [{', '.join(sorted(acc['src']))}]")
        print(f"  gate: max |projection-fit residual| = {acc['res']:.2e} mm"
              f"   (must be ~1e-12: the map is linear by construction)")
        print(f"  lattice measured from the dump: pitch {pitch} mm, drift step {dxs} mm")

        print("\n  -- distance from an interior trajectory point to the nearest ctpc cell (mm)")
        print("     plane    median      p75      p90   |  pass@2.0mm   model(best case)")
        dxm = dxs[0]
        for p in range(3):
            v = D[:, p][np.isfinite(D[:, p])]
            if not len(v):
                continue
            pm = pitch[0] if len(pitch) == 1 else (pitch[-1] if p < 2 else pitch[0])
            print("       %s    %7.3f  %7.3f  %7.3f  |    %6.3f        %6.3f"
                  % (PLANES[p], np.median(v), np.percentile(v, 75),
                     np.percentile(v, 90), (v <= 2.0).mean(),
                     lattice_model(pm, dxm, 2.0)))

        print("\n  -- is_good_point(radius, ch_range=0, allowed_bad=0): all three planes")
        print("     radius(cm)   U      V      W    all-three")
        for rc in RADII_CM:
            r = rc * CM
            per = [float((D[:, p][np.isfinite(D[:, p])] <= r).mean()) for p in range(3)]
            allp = float((D <= r).all(1).mean())
            print("       %4.2f      %5.3f  %5.3f  %5.3f    %5.3f"
                  % (rc, per[0], per[1], per[2], allp))

        # run lengths of consecutive failures at the production radius
        runs = []
        for cid, dist in acc["runs"]:
            ok = (dist <= 2.0).all(1)
            runs += runs_of_false(ok)
        if runs:
            runs = np.array(runs)
            print("\n  -- consecutive-failure runs at 0.2 cm (points; ~0.6 cm apart)")
            print("     n=%d  median %.0f  p90 %.0f  max %d   >=10 points: %.3f"
                  % (len(runs), np.median(runs), np.percentile(runs, 90),
                     runs.max(), (runs >= 10).mean()))

        X = np.concatenate(acc["ext"]) if acc["ext"] else np.zeros((0, 3))
        if len(X):
            EL = np.array(acc["extlen"])
            print("\n  -- past the trajectory tip, where charge IS present"
                  " (a 3d point within %.0f mm)" % args.near)
            print("     %d ends extend a median %.1f mm (p90 %.1f, max %.1f)"
                  " before the charge runs out"
                  % (len(EL), np.median(EL), np.percentile(EL, 90), EL.max()))
            print("     radius(cm)   U      V      W    all-three")
            for rc in RADII_CM:
                r = rc * CM
                per = [float((X[:, p][np.isfinite(X[:, p])] <= r).mean()) for p in range(3)]
                print("       %4.2f      %5.3f  %5.3f  %5.3f    %5.3f"
                      % (rc, per[0], per[1], per[2], float((X <= r).all(1).mean())))

        # phase-advance dependence: a frozen phase cannot be rescued by walking
        fin = np.isfinite(D) & np.isfinite(R)
        if fin.any():
            print("\n  -- pass rate at 0.2 cm vs lattice phase advance along the track")
            print("     |dphase/ds| (pitch/mm)   n      pass")
            for lo, hi in [(0, .05), (.05, .15), (.15, .3), (.3, .6), (.6, 1.01)]:
                m = fin & (np.abs(R) >= lo) & (np.abs(R) < hi)
                if m.sum() < 50:
                    continue
                print("       %.2f - %.2f            %6d   %5.3f"
                      % (lo, hi, int(m.sum()), float((D[m] <= 2.0).mean())))


if __name__ == "__main__":
    main()

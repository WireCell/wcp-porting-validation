#!/usr/bin/env python3
"""Doc pdvd/43 -- the EXIT census: where does a long track's end stop relative to
the wall its own direction is heading for?

Doc 41 sec 13 built its endpoint table from "the closest approach of a long
cluster to a wall, within a cap".  That sample is contaminated by tracks that
merely PASS a wall on their way out through another one, and the contamination
is exactly the tail a quantile reads: raising the cap from 25 to 40 cm moved the
pooled anode-half p90 from 15 to 26 cm.  A fiducial surface for an endpoint test
has to be built from exits, so here every end of every long cluster is assigned
to ONE boundary surface -- the first of the six (y+-, z-+, the two anode faces)
that the ray from the end along the track's outward local direction crosses --
and the record is the perpendicular gap between the end and that surface.  The
cathode slab is not a boundary (both fiducials span it).

The anode faces are the CONTROL: charge reaching the anode plane goes past it
(doc 41 sec 11.3), so an instrument that reads a 10 cm gap there is broken.

Per end the record also carries the readout-window position in the raw frame
(doc 41 sec 11.2: an end at the window edge is truncated by the READOUT, not by
imaging, and must not enter the surface) and the OFF arm's TGM/STM/FC verdicts
for the cluster (a stopping muon's stop end is a legitimate non-exit).

Row keys are those of fv_curved_zapproach.py (wall, x, dmin, cos, half, ...)
so fv_quantile_surface.py consumes either.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_exit_census.py --tag d41fvoff \
      --out /home/xqian/tmp/doc43/exits
"""
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_map import XW, YW, ZLO, ZHI, WALLS, wall_dist
from fv_curved_longloss import event_points, RAW_LATE, RAW_EARLY
from fv_curved_ab import read_arm

NORMAL = {"y+": np.array([0, 1., 0]), "y-": np.array([0, -1., 0]),
          "z-": np.array([0, 0, -1.]), "z+": np.array([0, 0, 1.]),
          "anode": None}


def first_exit(e, u):
    """The first of the six boundary planes the ray e + t*u (t >= 0) crosses.
    An end already AT or BEYOND a plane it is heading out of (signed gap <= 0, the
    imaged charge reaches or overshoots the wall) is an immediate hit, t = 0 --
    without that clamp such an end is handed to the next plane along the ray,
    hundreds of cm away.  Returns (wall, t_path, signed perpendicular gap)."""
    cands = []
    for w, ax, pos, n in (("y+", 1, YW, 1), ("y-", 1, -YW, -1), ("z-", 2, ZLO, -1), ("z+", 2, ZHI, 1),
                          ("anode", 0, XW, 1), ("anode", 0, -XW, -1)):
        un = u[ax] * n                      # outward component of the direction
        if un < 1e-9:
            continue                        # heading away from this plane
        g = (pos - e[ax]) * n               # signed gap, > 0 inside the plane
        cands.append((max(g / un, 0.0), w, g))
    if not cands:
        return None
    t, w, g = min(cands)
    return w, float(t), float(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--tag", default="d41fvoff")
    ap.add_argument("--minlen", type=float, default=200.0)
    ap.add_argument("--minpts", type=int, default=40)
    ap.add_argument("--out", default="/home/xqian/tmp/doc43/exits")
    a = ap.parse_args()

    arm = read_arm(a.pdvd, a.tag)
    rows = []
    nev = 0
    for wd in sorted(glob.glob(os.path.join(a.pdvd, "work", f"*_{a.tag}"))):
        base = os.path.basename(wd)
        run, idx, _ = base.split("_", 2)
        if (run, idx) not in arm:
            continue
        try:
            E = event_points(wd)
        except Exception as ex:
            print("  skip", wd, ex); continue
        nev += 1
        P, cid_all, ph = E["P"], E["cid"], E["phys"]
        for cid in np.unique(cid_all[ph]):
            m = (cid_all == cid) & ph
            if m.sum() < a.minpts:
                continue
            Q = P[m]
            c = Q - Q.mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            t = c @ axis
            length = float(t.max() - t.min())
            if length < a.minlen:
                continue
            v = arm[(run, idx)].get(int(cid), {})
            gidx = np.flatnonzero(m)
            for iend, ii in enumerate((int(np.argmax(t)), int(np.argmin(t)))):
                e = Q[ii]; gi = int(gidx[ii])
                dd = np.linalg.norm(Q - e, axis=1)
                near = Q[np.argsort(dd)[:40]]
                lc = near - near.mean(0)
                u = np.linalg.svd(lc, full_matrices=False)[2][0]
                # orient OUTWARD by the cluster's global axis (end 0 = +axis end), and
                # fall back to that axis when the last 40 points do not follow it (a
                # delta ray or a kink at the end): the local fit can point back INTO
                # the detector and hand the end to a wall hundreds of cm away.
                gax = axis if iend == 0 else -axis
                if np.dot(u, gax) < 0:
                    u = -u
                local_ok = bool(abs(np.dot(u, gax)) >= 0.7)
                if not local_ok:
                    u = gax
                hit = first_exit(e, u)
                if hit is None:
                    continue
                w, path, gap = hit
                # readout window, raw frame (doc 41 sec 11.2)
                xr = float(E["xraw"][gi]); sg = int(E["side"][gi])
                late = RAW_LATE if sg < 0 else -RAW_LATE
                early = -RAW_EARLY if sg < 0 else RAW_EARLY
                # the nearest wall by perpendicular distance, for comparison with sec 13
                per = {ww: float(wall_dist(ww, e[1], e[2])) for ww in WALLS}
                per["anode"] = float(XW - abs(e[0]))
                cw = min(per, key=per.get)
                nrm = NORMAL[w] if w != "anode" else np.array([np.sign(e[0]), 0, 0])
                rows.append(dict(run=run, idx=idx, cid=int(cid), end=iend, wall=w,
                                 dmin=round(gap, 3), path=round(path, 2),
                                 x=round(float(e[0]), 3), y=round(float(e[1]), 3), z=round(float(e[2]), 3),
                                 half=("cathode" if abs(e[0]) < 170 else "anode"),
                                 cos=round(float(abs(np.dot(u, nrm))), 4),
                                 ux=round(float(u[0]), 4), uy=round(float(u[1]), 4), uz=round(float(u[2]), 4),
                                 closest_wall=cw, closest_d=round(per[cw], 3),
                                 d_late=round(abs(xr - late), 2), d_early=round(abs(xr - early), 2),
                                 side=sg, npts=int(m.sum()), len_cm=round(length, 1),
                                 local_dir=local_ok,
                                 tgm=bool(v.get("tgm")), stm=bool(v.get("stm")), fc=bool(v.get("fc"))))
    json.dump(rows, open(a.out + "_rows.json", "w"), indent=0)
    print(f"{len(rows)} ends of long (>= {a.minlen:.0f} cm) clusters in {nev} events -> {a.out}_rows.json")

    # a first look: per assigned wall, the gap distribution, and the readout / STM shares
    for w in WALLS + ["anode"]:
        R = [r for r in rows if r["wall"] == w]
        if not R:
            continue
        g = np.array([r["dmin"] for r in R])
        ro = np.array([min(r["d_late"], r["d_early"]) < 5 for r in R])
        st = np.array([r["stm"] for r in R])
        print(f"  {w:5s} n {len(R):5d}  median {np.median(g):5.1f}  p80 {np.percentile(g, 80):5.1f}  "
              f"p90 {np.percentile(g, 90):5.1f}  >25 cm {100*np.mean(g > 25):4.1f} %  "
              f"at a readout edge {100*ro.mean():4.1f} %  cluster STM-tagged {100*st.mean():4.1f} %  "
              f"assigned wall == closest wall {100*np.mean([r['closest_wall'] == w for r in R]):4.1f} %")


if __name__ == "__main__":
    main()

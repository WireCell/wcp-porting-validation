#!/usr/bin/env python3
"""How connected is the cluster the STM tagger actually fits?

ClusteringUnmergeBundle (doc 45) undoes the Q/L *flash* merge, so the main
cluster it hands the STM tagger contains exactly one pre-merge
`real_cluster_id`.  It does NOT undo the *clustering-chain* merges (all-APA
merge passes), so that main can still be several spatially detached clumps --
and `connect_graph()` bridges them with an unbounded MST edge, exactly like the
prototype's `PR3DCluster::Connect_graph` (pid/src/PR3DCluster_graph.h:1166).
The fitted trajectory then flies through empty space.

This measures, for every cluster the STM fit actually ran on:
  - the number of >=`--gap` cm-separated charge components in the POST-unmerge
    main (nusel_evt<ID>/mabc-pr.zip, i.e. the tagger's own scope),
  - the largest inter-component gap,
  - `stray`: how far the fitted trajectory gets from ANY charge of its own
    cluster.  Unlike the component count this needs no linkage threshold, so it
    is the number to quote,
  - whether the trajectory *terminates* in a detached clump (the case that
    moves the STM verdict: the exit point is then not on the real track),
  - `dead%`: how much of the bridged gap lies inside a dead-channel region.
    `connect_graph_ctpc` bridges dead regions on purpose, so a high dead% means
    the algorithm working, not over-merge.

NOTE the geometry here is the nusel job's own POST-unmerge dump.  The scan
viewer draws ql_evt<ID>/mabc-all-apa.zip, which is PRE-unmerge -- see doc 50 §1.

Repro:
  cd sbnd_xin
  python3 stm_main_connectivity.py work-mcp10-d49son [work-mcp1000-d49son ...]
  python3 stm_main_connectivity.py --gap 15 work-mcp10-d49son      # sensitivity
  python3 stm_main_connectivity.py --detail 284657:27 work-mcp10-d49son
  python3 stm_main_connectivity.py --detail 285185:21 work-mcp10-d49son

Full write-up: docs/50_stm-fit-scope-and-unmerge.md
"""
import argparse
import glob
import json
import os
import zipfile

import numpy as np
import uproot
from matplotlib.path import Path as MplPath
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

AP = argparse.ArgumentParser()
AP.add_argument("roots", nargs="+", help="work-<tag> directories")
AP.add_argument("--gap", type=float, default=5.0,
                help="single-linkage threshold, cm (default 5)")
AP.add_argument("--stray", type=float, default=5.0,
                help="a trajectory point this far from any charge of its own "
                     "cluster is 'flying' (cm, default 5)")
AP.add_argument("--end", type=float, default=3.0,
                help="trajectory-end tolerance for 'terminates in a detached "
                     "clump' (cm, default 3)")
AP.add_argument("--detail", default=None, metavar="EVT:CID",
                help="print the full per-component breakdown for one cluster")
A = AP.parse_args()


def bee_clusters(path):
    """{cluster_id: (N,3) points} from a Bee clustering-global layer."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("-clustering-global.json")]
        if not names:
            return None
        d = json.loads(z.read(names[0]))
    cid = np.array(d["cluster_id"], int)
    P = np.c_[np.array(d["x"], float), np.array(d["y"], float),
              np.array(d["z"], float)]
    return {int(c): P[cid == c] for c in np.unique(cid)}


def components(P, thr):
    """single-linkage labels of P at threshold thr, via a KD-tree (no O(n^2))."""
    t = cKDTree(P)
    A = t.sparse_distance_matrix(t, thr, output_type="coo_matrix")
    nc, lab = connected_components(A, directed=False)
    return nc, lab


def dead_paths(path):
    """{tpc: [matplotlib Path]} from the Bee channel-deadarea layers.

    A dead wire region is a strip in (y, z) extended along the drift, so the
    polygons are (y, z) and the test is a 2-D point-in-polygon on the point's
    own TPC.  This matters because `connect_graph_ctpc` bridges gaps THROUGH
    dead channels on purpose -- a gap inside a dead region is the algorithm
    working, not over-merge.
    """
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if "-channel-deadarea-" not in n:
                continue
            d = json.loads(z.read(n))
            out.setdefault(int(d["tpc"]), []).extend(
                MplPath(np.array(p, float)) for p in d["polygons"] if len(p) >= 3)
    return out


def dead_fraction(A3, B3, dead, step=0.5):
    """Fraction of the straight A3->B3 segment whose (y,z) lies in a dead region.

    SBND: TPC0 is x<0, TPC1 is x>0 (cathode at x=0).  A segment that changes
    sign is tested against each side's own polygons.
    """
    if not dead:
        return float("nan")
    L = float(np.linalg.norm(B3 - A3))
    n = max(2, int(L / step))
    t = np.linspace(0, 1, n)[:, None]
    S = A3[None, :] * (1 - t) + B3[None, :] * t
    inside = np.zeros(len(S), bool)
    for tpc, paths in dead.items():
        m = (S[:, 0] > 0) if tpc == 1 else (S[:, 0] < 0)
        if not m.any():
            continue
        yz = S[m][:, 1:]
        hit = np.zeros(m.sum(), bool)
        for p in paths:
            hit |= p.contains_points(yz)
        inside[m] = hit
    return float(inside.mean())


rows = []
for root in A.roots:
    for stmroot in sorted(glob.glob(os.path.join(root, "nusel_evt*", "tracking-stm.root"))):
        nudir = os.path.dirname(stmroot)
        evt = os.path.basename(nudir).replace("nusel_evt", "")
        zp = os.path.join(nudir, "mabc-pr.zip")
        if not os.path.isfile(zp):
            continue
        try:
            geo = bee_clusters(zp)
            dead = dead_paths(zp)
            rec = uproot.open(stmroot)["T_rec_charge"].arrays(
                ["x", "y", "z", "rr", "cluster_id", "pass", "status"], library="np")
        except Exception as e:
            print(f"  [skip] evt{evt}: {e}")
            continue
        if geo is None:
            continue
        cids = rec["cluster_id"].astype(int)
        for cid in sorted(set(cids.tolist())):
            P = geo.get(cid)
            if P is None or len(P) < 2:
                continue
            nc, lab = components(P, A.gap)
            sizes = sorted(((lab == l).sum(), l) for l in set(lab.tolist()))[::-1]
            # largest gap between the dominant component and any other
            big = sizes[0][1]
            maxgap = 0.0
            tbig = cKDTree(P[lab == big])
            for n, l in sizes[1:]:
                d, _ = tbig.query(P[lab == l])
                maxgap = max(maxgap, d.min())
            # the fit: last pass only (the one the verdict rests on)
            s = cids == cid
            passes = sorted(set(rec["pass"][s].astype(int).tolist()))
            fp = passes[-1]
            s = s & (rec["pass"].astype(int) == fp)
            T = np.c_[rec["x"][s], rec["y"][s], rec["z"][s]]
            T = T[np.argsort(-rec["rr"][s])]
            tp = cKDTree(P)
            dn, _ = tp.query(T)
            L = np.linalg.norm(np.diff(T, axis=0), axis=1).sum() if len(T) > 1 else 0.0
            # does an END of the trajectory sit in a NON-dominant component?
            # `endgap` is the gap that component sits behind -- the number that
            # decides whether the fit crossed a plausible dead/blob gap or two
            # physically distinct objects.
            ends_detached = []
            endgap = 0.0
            deadfrac = float("nan")
            for k in (0, len(T) - 1):
                d, i = tp.query(T[k])
                if d < A.end and lab[i] != big:
                    ends_detached.append(k)
                    C = P[lab == lab[i]]
                    dg, jg = tbig.query(C)
                    a = int(np.argmin(dg))
                    if dg[a] > endgap:
                        endgap = dg[a]
                        deadfrac = dead_fraction(P[lab == big][jg[a]], C[a], dead)
            rows.append(dict(root=os.path.basename(root), evt=evt, cid=cid,
                             npts=len(P), ncomp=nc, maxgap=maxgap,
                             L=L, ntraj=len(T), stray=dn.max(),
                             nstray=int((dn > A.stray).sum()),
                             endgap=endgap, deadfrac=deadfrac, status=int(rec["status"][s][0]) if s.any() else -1,
                             passno=fp, ends=len(ends_detached)))
            if A.detail and A.detail == f"{evt}:{cid}":
                print(f"--- evt{evt} cluster {cid} (POST-unmerge, {len(P)} pts) ---")
                for n, l in sizes:
                    Q = P[lab == l]
                    d, _ = tbig.query(Q)
                    print(f"   comp n={n:5d} centroid={np.round(Q.mean(0), 1)} "
                          f"span={np.linalg.norm(Q.max(0) - Q.min(0)):6.2f} cm "
                          f"gap_to_dominant={d.min():6.2f} cm")
                print(f"   fit pass {fp}: {len(T)} pts, L={L:.2f} cm, "
                      f"status={rows[-1]['status']}")
                print(f"   ends {np.round(T[0], 2)} -> {np.round(T[-1], 2)}")
                print(f"   traj->own charge: median {np.median(dn):.2f} "
                      f"max {dn.max():.2f} cm, {int((dn > A.stray).sum())} pt(s) "
                      f"> {A.stray} cm")
                print(f"   trajectory ends inside a detached clump: "
                      f"{len(ends_detached)} of 2")

if not rows:
    raise SystemExit("no fitted clusters found (need -stm-fit rounds)")

print(f"\n=== STM-fitted main clusters: {len(rows)} "
      f"(linkage {A.gap} cm, stray {A.stray} cm) ===")
multi = [r for r in rows if r["ncomp"] > 1]
fly = [r for r in rows if r["nstray"] > 0]
endd = [r for r in rows if r["ends"] > 0]
print(f"  {len(multi):4d} ({100*len(multi)/len(rows):.0f} %) are NOT connected "
      f"post-unmerge  (>1 component at {A.gap} cm)")
print(f"  {len(fly):4d} ({100*len(fly)/len(rows):.0f} %) have a fitted "
      f"trajectory point > {A.stray} cm from any charge of their own cluster")
print(f"  {len(endd):4d} ({100*len(endd)/len(rows):.0f} %) have a trajectory "
      f"END inside a detached clump  <-- these can move the verdict")
# A gap is not automatically wrong: connect_graph exists precisely to bridge
# dead-channel and sparse-blob gaps, and the prototype does the same.  Split the
# end-in-clump cases by how far the fit had to fly to get there.
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 1e9)):
    n = len([r for r in endd if lo <= r["endgap"] < hi])
    print(f"        endgap {lo:>3}-{'inf' if hi > 1e8 else int(hi):>3} cm: {n:3d}")
if multi:
    g = np.array([r["maxgap"] for r in multi])
    print(f"  gap to the dominant component: median {np.median(g):.1f} "
          f"max {g.max():.1f} cm")

print(f"\n{'evt':>8} {'cid':>4} {'pts':>6} {'ncmp':>5} {'maxgap':>7} {'endgap':>7} {'dead%':>6} "
      f"{'L':>7} {'stray':>6} {'nfly':>5} {'endclump':>8} {'st':>3}  root")
for r in sorted(rows, key=lambda r: (-r["ends"], -r["stray"])):
    if r["ncomp"] == 1 and r["nstray"] == 0:
        continue
    print(f"{r['evt']:>8} {r['cid']:>4} {r['npts']:>6} {r['ncomp']:>5} "
          f"{r['maxgap']:>7.2f} {r['endgap']:>7.2f} {100*r['deadfrac']:>5.0f}% {r['L']:>7.1f} {r['stray']:>6.2f} "
          f"{r['nstray']:>5} {r['ends']:>8} {r['status']:>3}  {r['root']}")

#!/usr/bin/env python3
"""At WHICH clustering step did two detached pieces become one cluster?

Uses the per-step Bee layers written by run_ql_evt.sh -trace-bee (jsonnet
`trace_bee`, cfg/.../sbnd/clus.jsonnet trace_sets()).  Cluster ids are
renumbered after every step (`cluster_id_order: 'tree'`), so the two pieces are
identified by POINT COORDINATES taken from the final geometry and then looked up
in every layer.

For each step it reports whether the body points and the clump points sit in the
same cluster, and -- once they do -- how large the joining cluster is, so a
merge is distinguished from a rename.

Repro:
  cd sbnd_xin
  python3 stm_merge_attribution.py work-mcp10-trace51 285185 21
  python3 stm_merge_attribution.py work-mcp10-trace51 284657 27
Full write-up: docs/51_clustering-merge-attribution.md
"""
import argparse
import json
import os
import zipfile

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

AP = argparse.ArgumentParser()
AP.add_argument("root", help="work-<tag> dir holding ql_evt<ID>/ with trace layers")
AP.add_argument("evt")
AP.add_argument("cid", type=int, help="cluster ident in the FINAL (post-unmerge) geometry")
AP.add_argument("--gap", type=float, default=5.0, help="linkage threshold, cm")
AP.add_argument("--tol", type=float, default=0.5,
                help="3-D match tolerance inside the corrected (all-APA) scope, cm")
AP.add_argument("--tol-raw", type=float, default=1.5,
                help="3-D match tolerance for the RAW per-APA layers, cm.  The "
                     "per-APA scope differs from the post-QL reference by the T0 "
                     "drift correction (sub-cm on these events) plus the data "
                     "transverse offset (<=0.7 cm), so it needs slack.")
AP.add_argument("--pr-root", default=None,
                help="work root holding nusel_evt<ID>/mabc-pr.zip for the "
                     "reference geometry (default: same as root, else the "
                     "d49son tag)")
A = AP.parse_args()


def layers(path):
    """{layer_name: (cluster_id array, (N,3) points)} for every *-global.json."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith("-global.json"):
                continue
            base = os.path.basename(n)[:-len("-global.json")]
            name = base.split("-", 1)[1] if "-" in base else base
            d = json.loads(z.read(n))
            if "cluster_id" not in d or "x" not in d:
                continue
            out[name] = (np.array(d["cluster_id"], int),
                         np.c_[np.array(d["x"], float), np.array(d["y"], float),
                               np.array(d["z"], float)])
    return out


# --- reference geometry: the two pieces, from the post-unmerge PR dump ---------
pr_root = A.pr_root or A.root
prz = os.path.join(pr_root, f"nusel_evt{A.evt}", "mabc-pr.zip")
if not os.path.isfile(prz):
    alt = os.path.join(os.path.dirname(A.root.rstrip("/")) or ".",
                       "work-mcp10-d49son", f"nusel_evt{A.evt}", "mabc-pr.zip")
    prz = alt
ref = layers(prz)["clustering"]
m = ref[0] == A.cid
P = ref[1][m]
t = cKDTree(P)
nc, lab = connected_components(
    t.sparse_distance_matrix(t, A.gap, output_type="coo_matrix"), directed=False)
sizes = sorted(((lab == l).sum(), l) for l in set(lab.tolist()))[::-1]
big, small = sizes[0][1], sizes[1][1]
BODY, CLUMP = P[lab == big], P[lab == small]
tb = cKDTree(BODY)
print(f"reference (post-unmerge {os.path.basename(prz)} cluster {A.cid}): "
      f"body {len(BODY)} pts, clump {len(CLUMP)} pts, "
      f"gap {tb.query(CLUMP)[0].min():.2f} cm")
print(f"   clump centroid {np.round(CLUMP.mean(0), 2)}")


def ids_at(pts, cid_arr, layer_pts, tol):
    """Cluster ids holding `pts` in a layer, matched by coordinate.

    Returns (majority ids, matched, total).  "Majority" = every id holding at
    least 20 % of the matched points, so a handful of stray nearest-neighbour
    hits from an adjacent cluster cannot fake a merge.
    """
    kt = cKDTree(layer_pts)
    d, i = kt.query(pts)
    ok = d < tol
    if not ok.any():
        return set(), 0, len(pts)
    ids, cnt = np.unique(cid_arr[i[ok]], return_counts=True)
    keep = cnt >= 0.2 * cnt.sum()
    return set(ids[keep].tolist()), int(ok.sum()), len(pts)


# NOTE on scopes: the per-APA layers (and the all-APA 'img' layer) are dumped in
# RAW coords; every all-APA layer from switch_scope onward is in the T0-corrected
# scope.  The offset is the cluster's own drift correction, which is NOT small --
# evt284657 grp 5 sits at t0 = -841 us, i.e. ~131 cm of x.  So calibrate the shift
# from the data instead of assuming it: histogram dx over (y,z)-matched pairs and
# take the peak.  The true counterpart gives a sharp peak; unrelated clusters at
# the same (y,z) spread out and lose.
def estimate_dx(ref_pts, layer_pts, tol_yz=1.5, binw=1.0):
    kt = cKDTree(layer_pts[:, 1:])
    dx = []
    for j, hits in enumerate(kt.query_ball_point(ref_pts[:, 1:], tol_yz)):
        for h in hits:
            dx.append(layer_pts[h, 0] - ref_pts[j, 0])
    if not dx:
        return None
    dx = np.array(dx)
    lo, hi = dx.min(), dx.max() + binw
    nb = max(1, int(np.ceil((hi - lo) / binw)))
    cnt, edges = np.histogram(dx, bins=nb, range=(lo, lo + nb * binw))
    k = int(np.argmax(cnt))
    peak = 0.5 * (edges[k] + edges[k + 1])
    near = dx[np.abs(dx - peak) < 2 * binw]
    return float(np.median(near))
# --- walk every layer of every zip, in pipeline order -------------------------
zips = [("per-APA apa0", f"mabc-apa0-face0.zip"),
        ("per-APA apa1", f"mabc-apa1-face0.zip"),
        ("all-APA", "mabc-all-apa.zip")]
print(f"\n{'stage':<13} {'layer':<34} {'body cid(s)':<14} {'clump cid(s)':<14} "
      f"{'joined':<7} n_in_joined")
first = None
for stage, zn in zips:
    zp = os.path.join(A.root, f"ql_evt{A.evt}", zn)
    if not os.path.isfile(zp):
        continue
    L = layers(zp)
    names = sorted(n for n in L if n.startswith("tr"))
    # the pre-pipeline 'img' layer and the final 'clustering' layer bracket them
    order = (["img"] if "img" in L else []) + names + (["clustering"] if "clustering" in L else [])
    raw_stage = stage.startswith("per-APA")
    for name in order:
        cid_arr, pts = L[name]
        # the all-APA 'img' layer is pre-pipeline, hence RAW like the per-APA ones
        raw = raw_stage or name == "img"
        t_here, B, C = A.tol, BODY, CLUMP
        if raw:
            dx = estimate_dx(BODY, pts)
            if dx is None:
                print(f"{stage:<13} {name:<34} {'-':<14} {'(no y-z overlap)':<14}")
                continue
            sh = np.array([dx, 0.0, 0.0])
            B, C, t_here = BODY + sh, CLUMP + sh, A.tol_raw
        bset, bn, bt = ids_at(B, cid_arr, pts, t_here)
        cset, cn, ct = ids_at(C, cid_arr, pts, t_here)
        if cn == 0:
            print(f"{stage:<13} {name:<34} {'-':<14} {'(clump absent)':<14}")
            continue
        joined = bool(bset & cset) and len(bset) > 0
        nj = ""
        if joined:
            j = sorted(bset & cset)[0]
            nj = str(int((cid_arr == j).sum()))
            if first is None:
                first = (stage, name, j, nj)
        print(f"{stage:<13} {name:<34} "
              f"{','.join(str(x) for x in sorted(bset)):<14} "
              f"{','.join(str(x) for x in sorted(cset)):<14} "
              f"{'YES' if joined else 'no':<7} {nj}"
              + (f"   [dx {dx:+.2f} cm]" if raw else "")
              + ("" if bn == bt and cn == ct else
                 f"   [matched body {bn}/{bt}, clump {cn}/{ct}]"))

if first:
    print(f"\n==> FIRST layer where body and clump share a cluster: "
          f"{first[0]} / {first[1]}  (cluster {first[2]}, {first[3]} pts)")
else:
    print("\n==> body and clump are never in the same cluster in these layers")

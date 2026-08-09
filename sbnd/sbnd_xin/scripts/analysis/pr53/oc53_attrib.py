#!/usr/bin/env python3
"""doc pr/53 round 4: at WHICH clustering step do an owner-flagged pair of
(x,y,z) points A/B first become one cluster?

Reuses the layer reader / coordinate-matching machinery of
scripts/analysis/stm/stm_merge_attribution.py (doc 51), but is seeded from the
owner's OWN (Ax,Ay,Az),(Bx,By,Bz) pair -- as used directly by
oc53_probe.py -- instead of a final PR cluster id + connected-components split.
That is the right seed for this question: we are not isolating a "body" and
"clump" inside one big cluster, we are asking when two specific points the
owner pointed at first joined.

Walks BOTH clustering pipelines in order (per-APA apa0, per-APA apa1, all-APA),
using the per-step Bee layers written by run_ql_evt.sh -trace-bee (jsonnet
`trace_bee`, cfg/.../sbnd/clus.jsonnet trace_sets()) -- default OFF, no
production behaviour touched.

Two scope traps (both handled, per doc 51 sec 1):
  - cluster ids are renumbered after every step (`cluster_id_order: 'tree'`),
    so points are matched by 3-D coordinate, not id.
  - the per-APA layers (and the all-APA pre-pipeline 'img' layer) are in the
    RAW scope; every all-APA layer from switch_scope onward carries the T0
    drift correction -- the shift is calibrated from the data (estimate_dx),
    not assumed, exactly as stm_merge_attribution.py does.

Usage:
    oc53_attrib.py <trace_root> <evt> \\
        <Ax> <Ay> <Az> <Bx> <By> <Bz> [label] [--tol T] [--tol-raw T]

<trace_root> is a work root produced with `run_ql_evt.sh ... -trace-bee
-save-pctree -save-rcid -save-assoc`, holding ql_evt<evt>/mabc-{apa0-face0,
apa1-face0,all-apa}.zip.

Repro (doc pr/53 round 4):
    cd sbnd_xin
    python3 scripts/analysis/pr53/oc53_attrib.py \\
        work-oc53r4-trace 422851 \\
        -107.9 -55.1 344.6 -110.4 -61.9 350.0 "18255-422851"
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

AP = argparse.ArgumentParser()
AP.add_argument("root", help="work-<tag> dir holding ql_evt<ID>/ with trace layers")
AP.add_argument("evt")
AP.add_argument("Ax", type=float); AP.add_argument("Ay", type=float); AP.add_argument("Az", type=float)
AP.add_argument("Bx", type=float); AP.add_argument("By", type=float); AP.add_argument("Bz", type=float)
AP.add_argument("label", nargs="?", default="")
AP.add_argument("--seed-radius", type=float, default=1.5,
                 help="radius (cm) around each of A/B, in the FINAL clustering "
                      "layer, used to build the local neighbourhood tracked "
                      "back through every earlier layer")
AP.add_argument("--tol", type=float, default=0.5,
                 help="3-D match tolerance in the corrected (all-APA) scope, cm")
AP.add_argument("--tol-raw", type=float, default=1.5,
                 help="3-D match tolerance for the RAW per-APA layers, cm "
                      "(matches stm_merge_attribution.py's default -- the raw "
                      "scope differs from the corrected one by T0 + transverse "
                      "offset, sub-cm to <=0.7cm here)")
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


def estimate_dx(ref_pts, layer_pts, tol_yz=1.5, binw=1.0):
    """Calibrate the RAW->corrected x-shift from (y,z)-matched pairs (doc 51):
    a real counterpart gives a sharp peak in dx, unrelated points at the same
    (y,z) spread out and lose."""
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


def ids_at(pts, cid_arr, layer_pts, tol):
    """Cluster ids holding >=20% of the matched `pts` in this layer (doc 51's
    majority rule: a few stray nearest-neighbour hits cannot fake a merge)."""
    kt = cKDTree(layer_pts)
    d, i = kt.query(pts)
    ok = d < tol
    if not ok.any():
        return set(), 0, len(pts)
    ids, cnt = np.unique(cid_arr[i[ok]], return_counts=True)
    keep = cnt >= 0.2 * cnt.sum()
    return set(ids[keep].tolist()), int(ok.sum()), len(pts)


ql_dir = os.path.join(A.root, f"ql_evt{A.evt}")
all_zip = os.path.join(ql_dir, "mabc-all-apa.zip")
if not os.path.isfile(all_zip):
    sys.exit(f"ERROR: missing {all_zip} -- run with -trace-bee first")

# --- seed: local neighbourhoods around A and B in the FINAL clustering layer ---
Lfinal = layers(all_zip)
if "clustering" not in Lfinal:
    sys.exit("ERROR: no final 'clustering' layer in the all-APA zip")
cid_f, pts_f = Lfinal["clustering"]
Apt = np.array([A.Ax, A.Ay, A.Az])
Bpt = np.array([A.Bx, A.By, A.Bz])
tf = cKDTree(pts_f)
seedA = pts_f[tf.query_ball_point(Apt, A.seed_radius)]
seedB = pts_f[tf.query_ball_point(Bpt, A.seed_radius)]
if len(seedA) == 0 or len(seedB) == 0:
    sys.exit(f"ERROR: no final-layer charge within {A.seed_radius}cm of "
              f"{'A' if len(seedA)==0 else 'B'} -- check coordinates/frame")
label = A.label or f"evt{A.evt}"
print(f"=== {label} (evt {A.evt}) ===")
print(f"seed A: {len(seedA)} pts within {A.seed_radius}cm of {Apt.tolist()}")
print(f"seed B: {len(seedB)} pts within {A.seed_radius}cm of {Bpt.tolist()}")
cidA_final = cid_f[tf.query(Apt)[1]]
cidB_final = cid_f[tf.query(Bpt)[1]]
print(f"final (post-QL) cluster ids: A={cidA_final} B={cidB_final} "
      f"{'SAME' if cidA_final == cidB_final else 'DIFFERENT'}")

zips = [("per-APA apa0", "mabc-apa0-face0.zip"),
        ("per-APA apa1", "mabc-apa1-face0.zip"),
        ("all-APA", "mabc-all-apa.zip")]
print(f"\n{'stage':<13} {'layer':<34} {'A cid(s)':<14} {'B cid(s)':<14} "
      f"{'joined':<7} n_in_joined")
first = None
for stage, zn in zips:
    zp = os.path.join(ql_dir, zn)
    if not os.path.isfile(zp):
        continue
    L = layers(zp)
    names = sorted(n for n in L if n.startswith("tr"))
    order = (["img"] if "img" in L else []) + names + (["clustering"] if "clustering" in L else [])
    raw_stage = stage.startswith("per-APA")
    for name in order:
        cid_arr, pts = L[name]
        raw = raw_stage or name == "img"
        t_here, Aq, Bq = A.tol, seedA, seedB
        dx = None
        if raw:
            dx = estimate_dx(seedA, pts)
            if dx is None:
                dx = estimate_dx(seedB, pts)
            if dx is None:
                print(f"{stage:<13} {name:<34} {'-':<14} {'(no y-z overlap)':<14}")
                continue
            sh = np.array([dx, 0.0, 0.0])
            Aq, Bq, t_here = seedA + sh, seedB + sh, A.tol_raw
        aset, an, at = ids_at(Aq, cid_arr, pts, t_here)
        bset, bn, bt = ids_at(Bq, cid_arr, pts, t_here)
        if an == 0 or bn == 0:
            miss = "A" if an == 0 else "B"
            print(f"{stage:<13} {name:<34} {'-':<14} {f'({miss} absent)':<14}")
            continue
        joined = bool(aset & bset) and len(aset) > 0
        nj = ""
        if joined:
            j = sorted(aset & bset)[0]
            nj = str(int((cid_arr == j).sum()))
            if first is None:
                first = (stage, name, j, nj)
        print(f"{stage:<13} {name:<34} "
              f"{','.join(str(x) for x in sorted(aset)):<14} "
              f"{','.join(str(x) for x in sorted(bset)):<14} "
              f"{'YES' if joined else 'no':<7} {nj}"
              + (f"   [dx {dx:+.2f} cm]" if raw else "")
              + ("" if an == at and bn == bt else
                 f"   [matched A {an}/{at}, B {bn}/{bt}]"))

if first:
    print(f"\n==> FIRST layer where A and B share a cluster: "
          f"{first[0]} / {first[1]}  (cluster {first[2]}, {first[3]} pts)")
else:
    print("\n==> A and B are never in the same cluster in these layers "
          "(check --seed-radius / coordinates, or they may join only in a "
          "later PR-chain stage not covered by trace_bee)")

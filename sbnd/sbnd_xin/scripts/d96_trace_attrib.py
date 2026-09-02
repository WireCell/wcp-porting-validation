#!/usr/bin/env python3
"""doc 96 -- at WHICH clustering step did the final in-beam main become one cluster?

Generalises scripts/analysis/stm/stm_merge_attribution.py (doc 51), which
identifies the two pieces by connected components at a 5 cm gap.  That cannot
work here: the owner's finding is over-clustering of *touching* tracks, so the
pieces are one connected component in the final geometry by construction.

Instead: take the final production main's point cloud as the reference set, and
for every trace layer report how that SAME set of points was partitioned into
clusters at that step.  The merge step is the one where two comparable-mass
groups collapse into one -- not merely where the cluster count drops, since a
rename (`cluster_id_order: 'tree'` renumbers after every step) or a 5-point
crumb joining also moves the count.

Scope note (doc 51 sec 1): the per-APA layers are RAW; every all-APA layer from
switch_scope on carries the T0 drift correction.  These bundles sit at
t0 ~ 1 us => |dx| = t0 * 1.563 mm/us < 0.3 cm, and reality=sim means no
transverse pos_offset, so a 1.5 cm tolerance covers it.  The per-step match
fraction is printed so a silent scope mismatch shows up as a collapse in
`nmatch`, not as a wrong verdict.

Repro:
  python3 scripts/d96_trace_attrib.py work-dbg25a-ql work-dbg25a-trace95 30 5
  python3 scripts/d96_trace_attrib.py work-dbg25a-ql work-dbg25a-trace95 21 15
  python3 scripts/d96_trace_attrib.py work-dbg25a-ql work-dbg25a-trace95 5 18
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

AP = argparse.ArgumentParser()
AP.add_argument("prod_root", help="production Q/L work root (the reference clustering)")
AP.add_argument("trace_root", help="work root of the -trace-bee re-run")
AP.add_argument("evt")
AP.add_argument("cid", type=int, help="in-beam main cluster id in the FINAL geometry")
AP.add_argument("--tol", type=float, default=1.5, help="3-D match tolerance, cm")
AP.add_argument("--top", type=int, default=5, help="how many largest groups to print")
A = AP.parse_args()


def read_layer(zf, member):
    d = json.loads(zf.read(member))
    if "cluster_id" not in d or "x" not in d:
        return None
    return (np.array(d["cluster_id"], int),
            np.c_[np.array(d["x"], float), np.array(d["y"], float),
                  np.array(d["z"], float)])


def farpair(P):
    a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
    b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
    return float(np.linalg.norm(P[a] - P[b]))


def final_main(root, evt, cid):
    zp = os.path.join(root, f"ql_evt{evt}", "mabc-all-apa.zip")
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.endswith("-clustering-global.json")][0]
        c, P = read_layer(z, name)
    m = c == cid
    if not m.any():
        sys.exit(f"ERROR: cluster {cid} absent from {zp}")
    return P[m]


# --- reference + reproduction check (doc 51: the trace run is a separate job) --
ref = final_main(A.prod_root, A.evt, A.cid)
try:
    rep = final_main(A.trace_root, A.evt, A.cid)
except SystemExit:
    rep = None
print(f"evt{A.evt} main cid={A.cid}")
print(f"  production : {len(ref):6d} pts  far-pair {farpair(ref):7.1f} cm")
if rep is None:
    print("  trace      : cluster id absent -- REPRODUCTION CHECK FAILED")
else:
    same = len(rep) == len(ref) and farpair(rep) == farpair(ref)
    print(f"  trace      : {len(rep):6d} pts  far-pair {farpair(rep):7.1f} cm"
          f"   {'REPRODUCES' if same else 'DIFFERS -- attribution is unsafe'}")

tree = cKDTree(ref)

rows = []
for zipname in ("mabc-apa0-face0.zip", "mabc-apa1-face0.zip", "mabc-all-apa.zip"):
    zp = os.path.join(A.trace_root, f"ql_evt{A.evt}", zipname)
    if not os.path.isfile(zp):
        continue
    scope = "all-apa" if "all-apa" in zipname else zipname.split("-")[1]
    with zipfile.ZipFile(zp) as z:
        members = sorted(n for n in z.namelist()
                         if os.path.basename(n).split("-", 1)[1].startswith("tr"))
        for n in members:
            got = read_layer(z, n)
            if got is None:
                continue
            cl, P = got
            step = os.path.basename(n).split("-", 1)[1].split("-global")[0]
            # which layer point is each reference point?  (nearest, within tol)
            t2 = cKDTree(P)
            d, i = t2.query(ref, distance_upper_bound=A.tol)
            ok = np.isfinite(d)
            ids = cl[i[ok]]
            u, cnt = np.unique(ids, return_counts=True)
            o = np.argsort(-cnt)
            rows.append((scope, step, int(ok.sum()), len(ref), len(u),
                         [(int(u[j]), int(cnt[j])) for j in o[:A.top]]))

w = max(len(r[1]) for r in rows) if rows else 10
print(f"\n{'scope':<8}{'step':<{w+2}}{'matched':>9}{'ngrp':>6}  largest groups (id:npts)")
prev = None
for scope, step, nm, ntot, ng, top in rows:
    tops = "  ".join(f"{i}:{c}" for i, c in top)
    mark = ""
    if prev is not None and len(prev) >= 2 and len(top) >= 1:
        # merge = the previous step's top-2 both land inside this step's top-1
        if prev[1][1] >= 0.15 * ntot and (len(top) < 2 or top[1][1] < 0.5 * prev[1][1]):
            if top[0][1] >= 0.8 * (prev[0][1] + prev[1][1]):
                mark = "   <== MERGE"
    print(f"{scope:<8}{step:<{w+2}}{nm:>5}/{ntot:<4}{ng:>6}  {tops}{mark}")
    prev = top

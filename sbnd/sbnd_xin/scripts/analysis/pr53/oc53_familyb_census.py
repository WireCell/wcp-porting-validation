#!/usr/bin/env python3
"""doc pr/53 round 4: how big is the family-B ("clustering_isolated" absorb)
residual, detector-wide?

Section 13 of the doc found that a SINGLE event (71372) has its
clustering_isolated absorb correctly reversed by `unmerge_assoc`
(`ClusteringUnmergeBundle` name `assoc`) before any tagger runs. Planning-phase
code reading (round 4) found that reversal is READER-side conditional:
`unmerge_assoc` only processes a cluster that still carries
`Flags::main_cluster`, and the immediately preceding `unmerge_bundle` stage
clears that flag on every flash-bundle companion it splits off
(`ClusteringUnmergeBundle.cxx:180,403`). This script turns "family B is
retired" into a measured rate over the 48-event nueCC manifest.

Method, per event:
  1. Load the QL-stage pctree (production, `save_assoc=true` per doc 68):
     per-blob `real_cluster_id` (QL final cluster) and `assoc_cluster_id`
     (pre-isolated-merge partition, ClusteringIsolated::save_assoc_id).
  2. Group blobs by `real_cluster_id`. Any group spanning >1 distinct
     `assoc_cluster_id` had an isolated-merge fold two or more pre-existing
     clusters together -- this is the population of "family B" merges.
  3. For each such group, match EVERY point of each `assoc_cluster_id`
     subgroup (not just its centroid -- a centroid nearest-neighbour lookup
     mis-attributes small/sparse subgroups to whichever cluster is
     spatially nearest, which is the nearby main cluster BY CONSTRUCTION for
     an isolated-absorb neighbour; verified against a case the PR log proved
     was actually split) against the FINAL PR-chain `clustering-global` layer
     (bare production, `mabc-pr.zip`) within `--tol` cm, majority-vote
     (>= `--min-match-frac` of points must land within tolerance, else the
     subgroup is "ambiguous" and the whole group is excluded from the rate
     rather than guessed). If every confidently-matched subgroup maps to the
     SAME final cluster id, the merge is still standing (residual); if they
     map to DIFFERENT ids, `unmerge_assoc` (or a later stage) split it apart.
     Both pctree "3d" and mabc-pr.zip "clustering" are all-APA, post-
     switch_scope coordinates (production `save_tensors` is wired to the
     all-APA MABC node, `wct-clus-matching-perevt.jsonnet:348/383`) -- no
     RAW/corrected scope shift is needed here, unlike the per-APA trace_bee
     layers in oc53_attrib.py.

Repro (doc pr/53 round 4):
    cd sbnd_xin
    python3 scripts/analysis/pr53/oc53_familyb_census.py \\
        work-nuecc48-cb0805 work-pr51-off48e
"""
import argparse
import glob
import json
import os
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree

AP = argparse.ArgumentParser()
AP.add_argument("ql_root", help="QL-stage work root (production, save_assoc=true)")
AP.add_argument("pr_root", help="bare-production PR-chain work root (mabc-pr.zip per event)")
AP.add_argument("--events", default=None,
                 help="comma-separated event id subset (default: every ql_evt<ID> "
                      "under ql_root that also has a pr_evt<ID> under pr_root)")
AP.add_argument("--min-points", type=int, default=10,
                 help="materiality filter: only count a merge group in the "
                      "filtered rate if its SMALLER assoc_cluster_id subgroup "
                      "has at least this many points (the small<->small 5cm/"
                      "50cm chains in clustering_isolated.cxx absorb a large "
                      "number of few-point noise fragments that dilute the "
                      "raw rate; the owner's 71372 example was a 14pt/15pt "
                      "fragment, so 10 is a conservative materiality floor)")
AP.add_argument("--tol", type=float, default=1.5,
                 help="3-D match tolerance (cm) against the final PR "
                      "clustering-global layer; loose enough to survive a "
                      "Steiner-node snap without crossing to a genuinely "
                      "different nearby cluster")
AP.add_argument("--min-match-frac", type=float, default=0.3,
                 help="minimum fraction of a subgroup's own points that must "
                      "match some final cluster id within --tol for that "
                      "subgroup's attribution to be trusted; below this the "
                      "whole group is marked ambiguous and excluded from "
                      "the rate rather than guessed via nearest-neighbour")
A = AP.parse_args()


def A_(idx, path):
    return np.load(idx[path])


def load_pctree_perblob(ql_evt_dir, evt):
    tgz = os.path.join(ql_evt_dir, f"pctree-evt{evt}.tar.gz")
    tmp = tempfile.mkdtemp(prefix="oc53fb_")
    try:
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        idx = {}
        for f in glob.glob(os.path.join(tmp, "*_metadata.json")):
            m = json.load(open(f))
            if m.get("datatype") == "pcarray":
                idx[m["datapath"]] = f.replace("_metadata.json", "_array.npy")
        pre = f"pointtrees/{evt}/live/"
        g3d = pre + "pointclouds/namedpcs/3d/arrays/"
        gpb = pre + "pointclouds/namedpcs/perblob/arrays/"
        gsc = pre + "pointclouds/namedpcs/scalar/arrays/"
        if gpb + "assoc_cluster_id" not in idx:
            return None  # save_assoc arrays absent for this event
        P = np.c_[A_(idx, g3d + "x"), A_(idx, g3d + "y"), A_(idx, g3d + "z")] / 10.0  # mm->cm
        npts = A_(idx, gsc + "npoints")
        blob_of_point = np.repeat(np.arange(len(npts)), npts)
        real_id = A_(idx, gpb + "real_cluster_id")
        assoc_id = A_(idx, gpb + "assoc_cluster_id")
        return P, blob_of_point, real_id, assoc_id
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def load_pr_clustering(pr_zip):
    with zipfile.ZipFile(pr_zip) as z:
        names = [n for n in z.namelist() if n.endswith("clustering-global.json")]
        if not names:
            return None
        d = json.loads(z.read(names[0]))
        pts = np.c_[d["x"], d["y"], d["z"]]
        cid = np.array(d["cluster_id"])
        return cKDTree(pts), cid


if A.events:
    events = A.events.split(",")
else:
    events = sorted(
        os.path.basename(d)[len("ql_evt"):]
        for d in glob.glob(os.path.join(A.ql_root, "ql_evt*"))
        if os.path.isdir(d)
        and os.path.isdir(os.path.join(A.pr_root, "pr_evt" + os.path.basename(d)[len("ql_evt"):]))
    )

print(f"scanning {len(events)} events: ql_root={A.ql_root} pr_root={A.pr_root}\n")

n_events_with_merge = 0
n_merges_total = 0
n_merges_resolved = 0
n_merges_residual = 0
n_merges_total_f = 0
n_merges_resolved_f = 0
n_merges_residual_f = 0
detail = []

for evt in events:
    ql_dir = os.path.join(A.ql_root, f"ql_evt{evt}")
    pr_zip = os.path.join(A.pr_root, f"pr_evt{evt}", "mabc-pr.zip")
    if not (os.path.isdir(ql_dir) and os.path.isfile(pr_zip)):
        print(f"  [skip] evt {evt}: missing ql dir or pr zip")
        continue
    loaded = load_pctree_perblob(ql_dir, evt)
    if loaded is None:
        print(f"  [skip] evt {evt}: no assoc_cluster_id array (save_assoc off?)")
        continue
    P, blob_of_point, real_id, assoc_id = loaded
    prload = load_pr_clustering(pr_zip)
    if prload is None:
        print(f"  [skip] evt {evt}: no clustering-global layer in mabc-pr.zip")
        continue
    pr_tree, pr_cid = prload
    n_blobs = len(real_id)
    point_of_blob = blob_of_point  # per-POINT blob index, len == len(P)

    real_groups = {}
    for b in range(n_blobs):
        real_groups.setdefault(int(real_id[b]), []).append(b)

    def match_ids(sub_pts):
        """Majority-vote final cluster id(s) for a point set, gated on
        --min-match-frac points landing within --tol (doc 51's ids_at()
        idiom). Returns (ids_set, confident, n_matched, n_total)."""
        if len(sub_pts) == 0:
            return set(), False, 0, 0
        d, i = pr_tree.query(sub_pts)
        ok = d < A.tol
        n_matched = int(ok.sum())
        if n_matched == 0 or n_matched < A.min_match_frac * len(sub_pts):
            return set(), False, n_matched, len(sub_pts)
        ids, cnt = np.unique(pr_cid[i[ok]], return_counts=True)
        keep = cnt >= 0.2 * cnt.sum()
        return set(ids[keep].tolist()), True, n_matched, len(sub_pts)

    evt_has_merge = False
    for rid, blobs in real_groups.items():
        aids = sorted(set(int(assoc_id[b]) for b in blobs))
        if len(aids) <= 1:
            continue  # no isolated-merge in this QL cluster
        evt_has_merge = True
        n_merges_total += 1
        sub_npts = []
        sub_id_sets = []
        all_confident = True
        for aid in aids:
            sub_blobs = set(b for b in blobs if int(assoc_id[b]) == aid)
            mask = np.isin(point_of_blob, list(sub_blobs))
            sub_pts = P[mask]
            sub_npts.append(int(mask.sum()))
            ids, confident, nm, nt = match_ids(sub_pts)
            sub_id_sets.append(ids)
            all_confident = all_confident and confident
        if not all_confident:
            detail.append((evt, rid, len(aids), len(blobs), sub_npts, "ambiguous", [], False))
            continue
        union_shared = set.intersection(*sub_id_sets) if sub_id_sets else set()
        still_merged = len(union_shared) > 0  # every subgroup agrees on >=1 common id
        if still_merged:
            n_merges_residual += 1
        else:
            n_merges_resolved += 1
        material = min(sub_npts) >= A.min_points
        if material:
            n_merges_total_f += 1
            if still_merged:
                n_merges_residual_f += 1
            else:
                n_merges_resolved_f += 1
        status = "RESIDUAL" if still_merged else "resolved"
        detail.append((evt, rid, len(aids), len(blobs), sub_npts, status,
                        [sorted(s) for s in sub_id_sets], material))
    if evt_has_merge:
        n_events_with_merge += 1

n_ambiguous = sum(1 for d in detail if d[5] == "ambiguous")
print(f"{'evt':<8} {'QL rid':<8} {'#assoc':<7} {'#blobs':<7} {'subgrp pts':<24} {'status':<10} {'material':<9} final PR cid sets")
for evt, rid, naids, nblobs, sub_npts, status, id_sets, material in detail:
    print(f"{evt:<8} {rid:<8} {naids:<7} {nblobs:<7} {str(sub_npts):<24} {status:<10} "
          f"{'yes' if material else 'no':<9} {id_sets}")

print(f"\n=== summary (raw, every isolated-merge group regardless of size) ===")
print(f"events scanned:              {len(events)}")
print(f"events with >=1 isolated-merge: {n_events_with_merge}")
print(f"total isolated-merge groups:  {n_merges_total}")
print(f"  ambiguous (no subgroup confidently matched final output, excluded): {n_ambiguous}")
print(f"  resolved (unmerge_assoc or later stage split it back): {n_merges_resolved}")
print(f"  RESIDUAL (every subgroup still shares a final cluster id): {n_merges_residual}")
n_decided = n_merges_resolved + n_merges_residual
if n_decided:
    print(f"residual rate (of decided groups): {100.0*n_merges_residual/n_decided:.1f}%")

print(f"\n=== summary (materiality filter: smaller subgroup >= {A.min_points} points) ===")
print(f"total material, decided isolated-merge groups: {n_merges_total_f}")
print(f"  resolved: {n_merges_resolved_f}")
print(f"  RESIDUAL: {n_merges_residual_f}")
if n_merges_total_f:
    print(f"residual rate (material, decided only): {100.0*n_merges_residual_f/n_merges_total_f:.1f}%")

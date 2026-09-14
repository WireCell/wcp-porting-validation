#!/usr/bin/env python3
"""doc 108 sec 3.3: can the production Bee zips already give a charge-based
truth match for a PR candidate?

Usage:
  python3 d108_bee_label_match.py [--cands products/d107/candidates.tsv]
      [--bee /nfs/data/1/xqian/sbnd_data/run/bee] [--n 25] [--seed 5] [--layer-seed 3]

Part A (layers): over --n random events, which Bee point layer the
truth_trackid_labeled + truth_unlabeled points coincide with (< 0.05 cm).
Part B (candidates): for --n random T_tagger rows with a vertex, take the
points of PR cluster T_tagger.cluster_id in clustering-pr-global, find the
nearest labeled/unlabeled point, and attribute the points within 1 cm to a
truth interaction via the mc.json tree (label id -> top node 9000000+nu_idx).
"non-nu tree" = a label id with no node under a truth neutrino (cosmic, or a
beam-nu particle below the tree's 10 MeV KE cut).
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import zipfile

import numpy as np
from scipy.spatial import cKDTree


def layers(z):
    names = z.namelist()

    def L(n):
        m = [x for x in names if x.endswith("-" + n + ".json")]
        return json.loads(z.read(m[0])) if m else None
    return L


def xyz(j):
    return np.array([j["x"], j["y"], j["z"]]).T


def part_a(bee, n, seed):
    files = sorted(glob.glob(os.path.join(bee, "bee_*.zip")))
    random.seed(seed)
    random.shuffle(files)
    tot = collections.Counter()
    for p in files[:n]:
        L = layers(zipfile.ZipFile(p))
        lab, unl = L("truth_trackid_labeled"), L("truth_unlabeled")
        if lab is None:
            tot["no_truth_layer"] += 1
            continue
        Q = np.vstack([xyz(lab)] + ([xyz(unl)] if unl and len(unl["x"]) else []))
        tot["truth_pts"] += len(Q)
        for other in ("img-global", "clustering-global", "clustering-pr-global"):
            o = L(other)
            if o is None:
                continue
            d, _ = cKDTree(xyz(o)).query(Q)
            tot[f"truth_pts_within_0.05cm_of_{other}"] += int((d < 0.05).sum())
    print("== A: layer membership of the truth label points,", n, "events")
    for k, v in tot.items():
        print("  ", k, v)


def part_b(cands, bee, n, seed):
    C = list(csv.DictReader(open(cands), delimiter="\t"))
    random.seed(seed)
    rows = random.sample([r for r in C if r["vertex_default"] == "0"], n)
    summ = collections.Counter()
    print("== B: selected PR cluster vs truth labels,", n, "T_tagger rows")
    print("tag\tcluster\tnpts\tmedian_nn_cm\tfrac_within_1cm\ttop_attribution\tnearest_truth_idx\tnearest_truth_dist_cm")
    for r in rows:
        tag = f"r{r['run']}_s{r['subrun']}_e{r['event']}"
        L = layers(zipfile.ZipFile(os.path.join(bee, f"bee_{tag}.zip")))
        pr, lab, unl, mc = L("clustering-pr-global"), L("truth_trackid_labeled"), L("truth_unlabeled"), L("mc")
        Q = np.vstack([xyz(lab)] + ([xyz(unl)] if unl and len(unl["x"]) else []))
        ql = np.array(list(lab["cluster_id"]) + ([-1] * len(unl["x"]) if unl else []))
        cid = int(r["cluster_id"])
        P = xyz(pr)[np.array(pr["cluster_id"]) == cid]
        d, i = cKDTree(Q).query(P)
        near = d < 1.0
        top = {}

        def walk(node, root):
            top[node["id"]] = root
            for c in node.get("children", []):
                walk(c, root)
        for t in (mc if isinstance(mc, list) else [mc]):
            walk(t, t["id"])
        att = collections.Counter()
        for lbl, c in collections.Counter(ql[i[near]].tolist()).items():
            rt = top.get(lbl)
            if lbl == -1:
                att["unlabeled"] += c
            elif rt is not None and 9000000 <= rt < 9100000:
                att["nu%d" % (rt - 9000000)] += c
            else:
                att["non-nu tree"] += c
        tot = sum(att.values())
        frac = {k: round(v / tot, 2) for k, v in att.most_common(3)}
        best = att.most_common(1)[0]
        summ["rows"] += 1
        summ["all_pts_within_1cm"] += int(near.all())
        summ["dominant_" + ("nu" if best[0].startswith("nu") else best[0])] += 1
        if best[0].startswith("nu"):
            summ["min_nu_dominant_frac_x100"] = min(summ.get("min_nu_dominant_frac_x100", 100),
                                                    round(100 * best[1] / tot))
            # 1-based truth.tsv idx vs 0-based nu_idx of the dominant label
            summ["dominant_nu_equals_nearest_truth"] += int(int(best[0][2:]) + 1 == int(r["t_idx"]))
        print(f"{tag}\t{cid}\t{len(P)}\t{np.median(d):.2f}\t{near.mean():.3f}\t{frac}\t{r['t_idx']}\t{r['t_dist_cm']}")
    print("summary", dict(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="products/d107/candidates.tsv")
    ap.add_argument("--bee", default="/nfs/data/1/xqian/sbnd_data/run/bee")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--layer-seed", type=int, default=3)
    a = ap.parse_args()
    part_a(a.bee, a.n, a.layer_seed)
    part_b(a.cands, a.bee, a.n, a.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pr/66: trace evt 18255-10550's nu/TGM re-fusion to the all-APA merge
chain and rule out `examine_bundles` (doc pr/18 §6/§9's attribution).

Read-only. Reproduces every number quoted in the doc from the Bee zips of two
already-existing arms:
  - `work-nuecc48-cb0805/ql_evt10550/mabc-all-apa.zip` -- the QL (all-APA)
    output: `img-global` (end of the per-APA chain, pre-merge) and
    `clustering-global` (end of the all-APA chain, post-merge).
  - `work-pr64r4-on48/pr_evt10550/mabc-pr.zip` -- the PR output:
    `clustering-global` (post `ClusteringUnmergeBundle`).

Three checks:
  1. census -- match the img-global cluster holding each owner point
     (84.0, 79.2, 140.7) and (69.4, 57.4, 143.9), report its size/length/
     drift-x extent and the pr/18 iso_band_like() verdict, and the closest
     approach between the two.
  2. real_cluster_id -- inside QL clustering-global cluster 11 (the fused
     object), tabulate the perblob `real_cluster_id` array `examine_bundles`
     itself writes. If cluster 11 already existed at size ~12292 (band+nu)
     BEFORE examine_bundles ran (only 4 small members carry a distinct id),
     the merge predates that stage.
  3. pf_fate -- map every img-global nu-candidate point into the PR
     clustering-global layer and report how many stayed inside the TGM main
     (skipped by nu_skip_cosmic) vs. were carved out by unmerge_bundle.

Usage (from sbnd_xin/):
    python3 scripts/analysis/pr66/oc66_layer_trace.py

Requires numpy + scipy (already used elsewhere in scripts/analysis/).
"""
import collections
import json
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

QL_ZIP = "work-nuecc48-cb0805/ql_evt10550/mabc-all-apa.zip"
PR_ZIP = "work-pr64r4-on48/pr_evt10550/mabc-pr.zip"

OWNER_POINTS = [(84.0, 79.2, 140.7), (69.4, 57.4, 143.9)]


def load_layer(zp, name):
    with zipfile.ZipFile(zp) as z:
        return json.load(z.open(f"data/0/0-{name}.json"))


def iso_band_like(pts):
    """Port of clustering_neutrino.cxx's iso_band_like(): PCA length >= 80 cm
    and blob-center drift-x extent < max(25 cm, 0.18*length)."""
    if len(pts) == 0:
        return False, 0.0, 0.0
    ctr = pts.mean(0)
    c = pts - ctr
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    length = float(np.ptp(c @ vt[0]))
    xext = float(pts[:, 0].max() - pts[:, 0].min())
    return (length >= 80.0 and xext < max(25.0, 0.18 * length)), length, xext


def check1_census():
    print("=== check 1: img-global census + iso_band_like at the owner points ===")
    img = load_layer(QL_ZIP, "img-global")
    P = np.c_[img["x"], img["y"], img["z"]]
    C = np.array(img["cluster_id"])
    tree = cKDTree(P)
    hit_cids = []
    for p in OWNER_POINTS:
        d, i = tree.query([p], k=1)
        cid = int(C[i[0]])
        hit_cids.append(cid)
        print(f"  owner point {p} -> img-global cluster {cid} (dist {d[0]:.2f} cm)")
    for cid in sorted(set(hit_cids)):
        pts = P[C == cid]
        band, length, xext = iso_band_like(pts)
        print(f"  cluster {cid}: n={len(pts)} len={length:.1f} cm xext={xext:.1f} cm "
              f"iso_band_like={band}")
    if len(set(hit_cids)) == 2:
        a, b = sorted(set(hit_cids))
        pa, pb = P[C == a], P[C == b]
        d, _ = cKDTree(pb).query(pa, k=1)
        print(f"  closest approach {a}<->{b} = {d.min():.2f} cm")
        print(f"  combined size {a}+{b} = {(C == a).sum() + (C == b).sum()}")
    return hit_cids


def check2_real_cluster_id():
    print("\n=== check 2: real_cluster_id inside QL clustering-global cluster 11 ===")
    qcl = load_layer(QL_ZIP, "clustering-global")
    cid = np.array(qcl["cluster_id"])
    rcid = np.array(qcl["real_cluster_id"])
    sel = cid == 11
    cnt = collections.Counter(rcid[sel].tolist())
    print(f"  cluster 11 total size: {sel.sum()}")
    print(f"  real_cluster_id breakdown: {sorted(cnt.items())}")
    main = cnt.get(11, 0)
    others = sum(v for k, v in cnt.items() if k != 11)
    print(f"  pre-examine_bundles size of real id 11: {main}  "
          f"(+{others} folded in by examine_bundles = {main + others} total)")


def check3_pf_fate(nu_cid):
    print(f"\n=== check 3: fate of img-global nu cluster {nu_cid} in the PR layer ===")
    img = load_layer(QL_ZIP, "img-global")
    IP = np.c_[img["x"], img["y"], img["z"]]
    IC = np.array(img["cluster_id"])
    prc = load_layer(PR_ZIP, "clustering-global")
    PP = np.c_[prc["x"], prc["y"], prc["z"]]
    PC = np.array(prc["cluster_id"])
    tree = cKDTree(PP)
    sel = IC == nu_cid
    d, i = tree.query(IP[sel], k=1)
    matched = d < 1.5
    cnt = collections.Counter(PC[i[matched]].tolist())
    print(f"  nu candidate (img c{nu_cid}, {sel.sum()} pts) lands on PR clusters: "
          f"{cnt.most_common(10)}")
    in_tgm = cnt.get(11, 0)
    print(f"  -> {in_tgm}/{sel.sum()} pts ({100.0*in_tgm/sel.sum():.0f}%) remain inside "
          f"PR main cluster 11 (the TGM-tagged, nu_skip_cosmic'd bundle)")
    print(f"  -> {sel.sum()-in_tgm}/{sel.sum()} pts carved out by ClusteringUnmergeBundle "
          f"into {len(cnt)-1} small associated cluster(s)")


def check4_flash_group_bridge(band_cid, nu_cid):
    """Enumerate every img-global cluster that maps into QL clustering-global
    cluster 11 (the fused band+nu+specks object), and the pairwise closest
    approach among the band, the nu, and each speck -- checking whether any
    speck sits within reach of BOTH (a transitive-bridge risk for a
    pairwise-only merge veto)."""
    print("\n=== check 4: flash-group members feeding QL cluster 11 + pairwise "
          "closest approach ===")
    img = load_layer(QL_ZIP, "img-global")
    P = np.c_[img["x"], img["y"], img["z"]]
    C = np.array(img["cluster_id"])
    qcl = load_layer(QL_ZIP, "clustering-global")
    QP = np.c_[qcl["x"], qcl["y"], qcl["z"]]
    QC = np.array(qcl["cluster_id"])
    tree = cKDTree(QP)
    d, i = tree.query(P, k=1)
    sel = (QC[i] == 11) & (d < 1.5)
    members = sorted(set(C[sel].tolist()), key=lambda c: -(C[sel] == c).sum())
    print(f"  img-global clusters mapping into QL cluster 11: "
          f"{[(m, int((C[sel] == m).sum())) for m in members]}")
    trees = {m: cKDTree(P[C == m]) for m in members}
    print("  pairwise closest approach (cm):")
    header = "        " + "".join(f"c{m:<7}" for m in members)
    print(header)
    for a in members:
        row = []
        for b in members:
            if a == b:
                row.append("   -   ")
            else:
                dd, _ = trees[b].query(P[C == a], k=1)
                row.append(f"{dd.min():7.2f}")
        print(f"  c{a:<5} " + " ".join(row))
    others = [m for m in members if m not in (band_cid, nu_cid)]
    bridge = False
    for m in others:
        d_band, _ = trees[band_cid].query(P[C == m], k=1)
        d_nu, _ = trees[nu_cid].query(P[C == m], k=1)
        near_both = d_band.min() < 1.2 and d_nu.min() < 1.2
        bridge = bridge or near_both
        print(f"  speck c{m}: {d_band.min():.1f} cm from band, {d_nu.min():.1f} cm "
              f"from nu -> transitive bridge risk at 1.2cm cut: {near_both}")
    print(f"  ANY speck bridges band<->nu at the close:all 1.2cm cut: {bridge}")


def main():
    hit_cids = check1_census()
    check2_real_cluster_id()
    if len(set(hit_cids)) == 2:
        # the smaller cluster is the nu candidate (iso_band_like == False)
        img = load_layer(QL_ZIP, "img-global")
        C = np.array(img["cluster_id"])
        sizes = {c: (C == c).sum() for c in set(hit_cids)}
        nu_cid = min(sizes, key=sizes.get)
        band_cid = max(sizes, key=sizes.get)
        check3_pf_fate(nu_cid)
        check4_flash_group_bridge(band_cid, nu_cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

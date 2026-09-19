#!/usr/bin/env python3
"""doc pdvd/116 sec 10.3 (d) -- the flip gate of the doc-116 flip.  Fork by duplication of d103_flip_gate.py (untouched)
with ONE addition: an event whose file lacks the tree in BOTH arms with the same set of trees (the known "no candidate"
events 028084_13 / 039252_17, which write no tagger table) is reported as `absent in both` and counts as a match; a
tree absent in only one arm is still a divergence.  Read-only.

Original header:
doc pdvd/103 sec 14 -- the PDVD flip gate: does the FLIPPED CONFIG reproduce the graded arm?  Read-only.

The graded arm d103v1 was produced with TLAs (-A trackfitting_config=<figs/101_tf_prod_pdvd_kf.json>
-S retile_sampler_strategy='charge_stepped').  The proposed flip instead changes two production files
(figs/103_flip_pdvd.patch for the retile default, and the two lattice keys in pdvd_track_fitting.json), so the flip is
only as good as the claim that the two produce the same chain.  This script compares the tagger output of a flip-config
arm against the graded arm, event by event and cluster by cluster.

    python3 d103_flip_gate.py --a d103v1 --b d103vflip [--det pdvd] [--tree T_stm_michel]

Compares every branch of T_stm_michel (the tagger's verdict table) keyed by (event, cluster_id), and reports the first
divergence.  Exit status 0 only when every event, every cluster and every branch match.
"""
import argparse, glob, os, sys
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"


def arm_events(det, arm):
    out = {}
    for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}")):
        out[os.path.basename(d)[:-len(arm) - 1]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the graded arm (TLA-driven)")
    ap.add_argument("--b", required=True, help="the flip-config arm")
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--tree", default="T_stm_michel")
    a = ap.parse_args()
    A, B = arm_events(a.det, a.a), arm_events(a.det, a.b)
    print(f"# doc pdvd/116 flip gate ({a.det}): {a.a} (graded, TLAs) vs {a.b} (flipped config); tree {a.tree}")
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print(f"  event dirs only in {a.a}: {only_a}\n  only in {a.b}: {only_b}")
    bad, n_cl, n_ev, absent = [], 0, 0, []
    for ev in sorted(set(A) & set(B)):
        try:
            fa, fb = uproot.open(f"{A[ev]}/tracking-pr.root"), uproot.open(f"{B[ev]}/tracking-pr.root")
            ka, kb = {k.split(";")[0] for k in fa.keys()}, {k.split(";")[0] for k in fb.keys()}
            if a.tree not in ka and a.tree not in kb:
                if ka == kb:
                    absent.append(f"{ev} (trees {sorted(ka)})")
                else:
                    bad.append(f"{ev}: {a.tree} absent in both but the tree sets differ: {sorted(ka ^ kb)}")
                continue
            ta = fa[a.tree].arrays(library="np")
            tb = fb[a.tree].arrays(library="np")
        except Exception as e:
            bad.append(f"{ev}: cannot read ({type(e).__name__})")
            continue
        n_ev += 1
        if set(ta) != set(tb):
            bad.append(f"{ev}: branch sets differ: {sorted(set(ta) ^ set(tb))}")
            continue
        if len(ta["cluster_id"]) != len(tb["cluster_id"]):
            bad.append(f"{ev}: {len(ta['cluster_id'])} rows vs {len(tb['cluster_id'])}")
            continue
        oa, ob = np.argsort(ta["cluster_id"]), np.argsort(tb["cluster_id"])
        n_cl += len(oa)
        for br in sorted(ta):
            va, vb = np.asarray(ta[br])[oa], np.asarray(tb[br])[ob]
            if va.dtype.kind in "fc":
                same = np.allclose(va, vb, rtol=0, atol=0, equal_nan=True)
            else:
                same = np.array_equal(va, vb)
            if not same:
                i = int(np.argmax(va != vb)) if va.shape == vb.shape else -1
                bad.append(f"{ev}: branch {br} differs (first at cluster "
                           f"{int(ta['cluster_id'][oa][i]) if i >= 0 else '?'}: {va[i]!r} vs {vb[i]!r})")
                break
    print(f"  events compared {n_ev} (dirs {len(A)} / {len(B)}), clusters {n_cl}, branches {len(set(ta)) if n_ev else 0}")
    if absent:
        print(f"  {a.tree} absent in BOTH arms with the same tree set (no candidate; counted as a match): {absent}")
    if bad:
        print(f"  DIVERGENCES {len(bad)}:")
        for x in bad[:20]:
            print(f"    {x}")
        sys.exit(1)
    print("  IDENTICAL: every event, cluster and branch of the tagger table matches.")


if __name__ == "__main__":
    main()

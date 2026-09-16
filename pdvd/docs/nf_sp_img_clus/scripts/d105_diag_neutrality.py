#!/usr/bin/env python3
"""doc pdvd/105 sec 3 -- is the diagnostic arm the same reconstruction as the graded arm?  (read-only)

Doc 104 sec 7 proposed one PDHD arm with survey_enable + publish_other_arms to expose the companion-piece geometry
(rej / d_stop / d_body columns, role-6 survey rows, role-7 rejected-arm rows) that the stock output does not carry.
Whether those columns describe the GRADED arm depends on the knobs being pure writers, and

    cfg/pgrapher/experiment/pdhd/pr.jsonnet says they are NOT:
    "OWNER RULING 2026-09-08: it stays FALSE here.  The survey costs a 20-25 % mover rate on the muon's OWN
     profile branches through preload_clusters (doc pdvd/53 sec 6.2) for a feature with no physics value"

so this is measured, not assumed, BEFORE any number from the diagnostic arm is used.  The comparison is per
(event, cluster_id) over every branch the two trees share; the diagnostic arm's four extra branches
(n_survey_clusters / n_survey_segs / n_survey_unfit / n_other_published) are listed and skipped.

WHAT THE ANSWER CHANGES:
  * verdict branches (is_stm, reject_bits, michel_found, michel_conn_type) identical => the geometry columns
    describe the graded arm and can be used to size a discriminator directly;
  * verdict branches move => the arm is a different reconstruction.  The columns still describe THAT arm, so they
    can characterise the companion-piece population, but no cut sized on them can be quoted against the doc 103
    grade without re-running the graded arm with the cut itself.

    python3 d105_diag_neutrality.py --a d102hcs --b d105hdiag > figs/105_diag_neutrality.txt
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# the branches that ARE the tagger's answer; everything else is descriptive
VERDICT = ["is_stm", "reject_bits", "michel_found", "michel_conn_type", "michel_len", "michel_ke_best",
           "michel_dis_cm", "michel_n_pieces", "n_dots", "muon_len", "stop_x", "stop_y", "stop_z"]


def load(det, arm, tree="T_stm_michel"):
    out, keys = {}, None
    for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}")):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")[tree]
            k = [x for x in t.keys()]
            a = t.arrays(k, library="np")
        except Exception:
            continue
        keys = k if keys is None else keys
        for i in range(len(a["cluster_id"])):
            out[f"{ev}/{int(a['cluster_id'][i])}"] = {c: a[c][i] for c in k}
    return out, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdhd")
    ap.add_argument("--a", required=True, help="the graded arm")
    ap.add_argument("--b", required=True, help="the diagnostic arm")
    a = ap.parse_args()
    A, ka = load(a.det, a.a)
    B, kb = load(a.det, a.b)
    print(f"# doc pdvd/105 sec 3: does {a.b} reconstruct the same as {a.a}?  ({a.det})")
    print(f"# rows: {a.a} {len(A)}, {a.b} {len(B)}; branches {len(ka)} vs {len(kb)}")
    extra = [k for k in kb if k not in ka]
    missing = [k for k in ka if k not in kb]
    print(f"# extra branches in {a.b} (skipped): {extra}")
    if missing:
        print(f"# MISSING from {a.b}: {missing}")
    common = [k for k in ka if k in kb]
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    shared = sorted(set(A) & set(B))
    print(f"# candidates: shared {len(shared)}, only in {a.a} {len(only_a)}, only in {a.b} {len(only_b)}")
    if only_a[:5] or only_b[:5]:
        print(f"#   e.g. only-{a.a}: {only_a[:5]}   only-{a.b}: {only_b[:5]}")

    moved = {}
    for k in shared:
        for c in common:
            x, y = A[k][c], B[k][c]
            same = bool(np.all(x == y)) if np.ndim(x) or np.ndim(y) else (x == y or (x != x and y != y))
            if not same:
                moved.setdefault(c, []).append(k)
    print(f"\n== branches that move on at least one shared candidate: {len(moved)} of {len(common)}")
    for c, ks in sorted(moved.items(), key=lambda kv: -len(kv[1])):
        flag = "  <== VERDICT" if c in VERDICT else ""
        print(f"   {c:28s} {len(ks):5d} of {len(shared)} ({100 * len(ks) / max(1, len(shared)):5.1f} %){flag}")
    vm = [c for c in moved if c in VERDICT]
    print(f"\n== reading: {'the two arms are the same reconstruction on every verdict branch' if not vm and not only_a and not only_b else 'the diagnostic arm is a DIFFERENT reconstruction'}")
    if vm or only_a or only_b:
        print(f"   verdict branches that move: {vm}")
        print(f"   candidate-set differences: only-{a.a} {len(only_a)}, only-{a.b} {len(only_b)}")
        print("   => the geometry columns describe the DIAGNOSTIC arm, not the graded one.  They may characterise")
        print("      the companion-piece population; a cut sized on them must be re-run in the graded arm before")
        print("      any number is quoted against the doc 103 grade.")


if __name__ == "__main__":
    main()

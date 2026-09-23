#!/usr/bin/env python3
"""doc qlmatch/31 -- hand-scan agreement restricted to the flashes ToT can change.

A flash is "cathode-railed" if its sat flag is set on a cathode X-Arapuca OpDet (4..11): only those flashes carry a
cathode rail run, so only their PE can move under saturation_repair_mode tot (membrane/PMT keep twoside).  For each
tag: judged autos (agree/phantom, objective tiers, long tracks) on cathode-railed flashes, and scan positives whose
flash time matches a cathode-railed flash (covered/missed).  Same truth, uid map, tolerances and long-track cut as
ql_display/ql_agree_score.py, which is imported read-only.

    cd pdvd && python3 docs/qlmatch/scripts/d31_scan_subset.py --work-root /home/xqian/tmp/p31/wroot --tag q29flip ...
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(PDVD, "ql_display"))
import ql_agree_score as Q          # noqa: E402

CATH_ODS = set(range(4, 12))


def cath_railed(f):
    return any(s > 0 for k, s in enumerate(f.get("sat", [])) if k in CATH_ODS)


def one_tag(tag, truth, tol, tmpd, root):
    tot = dict(j_agree=0, j_phantom=0, p_covered=0, p_missed=0, n_flash=0, n_railed=0)
    for idx in range(Q.NEVT):
        evt = Q.evt_of_idx(idx)
        calib = os.path.join(root, f"{Q.RUN}_{idx}_{tag}", f"calib-evt{evt}.json")
        if not os.path.exists(calib):
            print(f"[warn] missing {calib}")
            continue
        old = os.path.join(root, f"{Q.RUN}_{idx}_keep", f"calib-evt{evt}.json")
        umap = Q.build_uid_map(old, calib)
        tev = [dict(e, uid=umap[e["uid"]]) for e in truth[evt] if e["uid"] in umap]
        c = json.load(open(calib))
        for f in c["flashes"]:          # sat kept only on cathode OpDets -> score_event's sat_flash == cathode-railed
            tot["n_flash"] += 1
            if cath_railed(f):
                tot["n_railed"] += 1
            f["sat"] = [s if k in CATH_ODS else 0 for k, s in enumerate(f.get("sat", []))]
        p = os.path.join(tmpd, f"calib-{tag}-{evt}.json")
        json.dump(c, open(p, "w"))
        r = Q.score_event(p, tev, tol, 25.0, 100, Q.OBJECTIVE_TIERS)
        tot["j_agree"] += r["flag_judged"]["sat_flash"] - r["flag_phantom"]["sat_flash"]
        tot["j_phantom"] += r["flag_phantom"]["sat_flash"]
        railed_t = [f["time"] for f in c["flashes"] if any(f["sat"])]
        missed = {(m["uid"], m["time"]) for m in r["missed_list"]}
        clusters = {cl["uid"]: cl for cl in c["clusters"]}
        for e in tev:
            if not e["positive"] or e["conf"] not in Q.OBJECTIVE_TIERS or e["uid"] not in clusters:
                continue
            cl = clusters[e["uid"]]
            if not (Q.cluster_len_cm(cl) >= 25.0 or cl["npoints"] >= 100):
                continue
            if not any(abs(t - e["time"]) <= tol for t in railed_t):
                continue
            tot["p_missed" if (e["uid"], e["time"]) in missed else "p_covered"] += 1
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--work-root", default="work",
                    help="dir holding <run6>_<idx>_<tag>/ and _keep/ with plain calib-evt*.json (the _keep dumps are "
                         ".zst in work/; point this at a scratch root of symlinks + decompressed copies)")
    a = ap.parse_args()
    os.chdir(PDVD)
    truth = Q.load_truth("work/ql_labels/wfresc/labels-evt298567.json", "ql_display/decisions-cathxa")
    print("# cathode-railed subset (sat on OpDet 4..11); objective tiers, long tracks, tol %.1f us" % a.tol)
    print(f"{'tag':>10} {'flashes':>8} {'railed':>7} | {'judged':>6} {'agree':>6} {'phantom':>7} | "
          f"{'pos':>4} {'covered':>7} {'missed':>6}")
    with tempfile.TemporaryDirectory(dir="/home/xqian/tmp") as tmpd:
        for tag in a.tag:
            t = one_tag(tag, truth, a.tol, tmpd, os.path.abspath(a.work_root))
            print(f"{tag:>10} {t['n_flash']:>8} {t['n_railed']:>7} | {t['j_agree'] + t['j_phantom']:>6} "
                  f"{t['j_agree']:>6} {t['j_phantom']:>7} | {t['p_covered'] + t['p_missed']:>4} "
                  f"{t['p_covered']:>7} {t['p_missed']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

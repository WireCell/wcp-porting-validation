#!/usr/bin/env python3
"""doc pdvd/102 sec 7.2 -- adjudicate every track that fails the pre-registered segment clause (c4).

c4 counts PR segments (tracking-pr.root T_rec_charge, sub_cluster_id with flag_vertex == 0) and fails a
setting if ANY ISO track has more segments than the baseline.  It was written for doc 101's PDHD k3 mode
(a spurious segment AND the main pulled off the track).  This script does not change the frozen verdict; it
classifies each flagged track so the owner can rule:
  stub_beside_improved_main   no extra >= 10-point muon segment, res_t p90 improves
  stub_beside_worse_main      no extra >= 10-point muon segment, res_t p90 worsens
  fragmented_main             an extra >= 10-point pid-13 segment
and, per arm segment: its median distance to the arm's own clustering image (Bee clustering layer), its
median / max distance from the TRUE muon line (truth json tail/head), and its median Bee q.
A segment on the image but far from the truth line is real deposited charge off the muon (eg a delta ray,
which the truth json does not list); a segment off the image is a reconstruction artefact.

Usage: d102_c4_decompose.py [--arms cs1,cs2,cs3] [--eval-dir /home/xqian/tmp/d102/eval]
"""
import argparse, csv, json, zipfile
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
RUN = {"pdvd": 900103, "pdhd": 900104}


def ev(d, det, arm):
    return {int(r["k"]): r for r in csv.DictReader(open(f"{d}/eval_{det}_{arm}.tsv"), delimiter="\t")}


def segments(det, k, arm):
    w = f"{IMG}/{det}/work/{RUN[det]}_{k}_{arm}"
    t = uproot.open(f"{w}/tracking-pr.root")["T_rec_charge"].arrays(
        ["x", "y", "z", "q", "sub_cluster_id", "flag_vertex", "particle_id"], library="np")
    m = t["flag_vertex"] == 0
    out = []
    for s in sorted(set(t["sub_cluster_id"][m].tolist())):
        mm = m & (t["sub_cluster_id"] == s)
        out.append(dict(n=int(mm.sum()), pid=int(t["particle_id"][mm][0]),
                        P=np.c_[t["x"][mm], t["y"][mm], t["z"][mm]], q=float(np.median(t["q"][mm]))))
    img = json.loads(zipfile.ZipFile(f"{w}/mabc-pr.zip").read("data/0/0-clustering-global.json"))
    return sorted(out, key=lambda s: -s["n"]), np.c_[img["x"], img["y"], img["z"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="cs1,cs2,cs3")
    ap.add_argument("--eval-dir", default="/home/xqian/tmp/d102/eval")
    a = ap.parse_args()
    for det in ("pdvd", "pdhd"):
        truth = {e["k"]: e for e in json.load(open(f"/home/xqian/tmp/d102/sim/truth_{det}.json"))["events"]}
        B = ev(a.eval_dir, det, "d102b")
        for s in a.arms.split(","):
            arm = "d102" + s
            A = ev(a.eval_dir, det, arm)
            counts = {"stub_beside_improved_main": 0, "stub_beside_worse_main": 0, "fragmented_main": 0}
            lines = []
            for k in range(16):
                sb, _ = segments(det, k, "d102b")
                sa, Ia = segments(det, k, arm)
                if len(sa) <= len(sb):
                    continue
                dp90 = float(A[k]["res_t_p90"]) - float(B[k]["res_t_p90"])
                mu = lambda ss: sum(1 for x in ss if x["pid"] == 13 and x["n"] >= 10)
                cls = ("fragmented_main" if mu(sa) > mu(sb) else
                       "stub_beside_improved_main" if dp90 < 0 else "stub_beside_worse_main")
                counts[cls] += 1
                tail = np.array(truth[k]["tail_cm"]); head = np.array(truth[k]["head_cm"])
                u = (head - tail) / np.linalg.norm(head - tail)
                ti = cKDTree(Ia)
                lines.append(f"    k{k} theta {truth[k]['theta_deg']} {cls}: d res_t_p90 {dp90:+.2f} mm; "
                             f"base {[(x['n'], x['pid']) for x in sb]} -> arm {[(x['n'], x['pid']) for x in sa]}")
                for x in sa:
                    rel = x["P"] - tail; along = rel @ u
                    perp = np.linalg.norm(rel - np.outer(along, u), axis=1)
                    lines.append(f"        seg n {x['n']:3d} pid {x['pid']:3d} along truth {along.min():6.1f}..{along.max():6.1f} cm | "
                                 f"to image med {np.median(ti.query(x['P'])[0]):.2f} cm | from truth line med {np.median(perp):.2f} "
                                 f"max {perp.max():.2f} cm | Bee q med {x['q']:.0f}")
            print(f"{det} {arm}: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
            print("\n".join(lines))


if __name__ == "__main__":
    main()

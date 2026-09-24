#!/usr/bin/env python3
"""doc sbnd_xin/123 round 3 -- compare the neutrino-selection output of two stage-B arms.

Input: two product dirs written by d115_truth_join.py (--arm <pr root> [--truth truth_base.tsv]
--out <dir>): candidates.tsv (one row per neutrino candidate with numu_score, nue_score, reco_Enu,
nu_x/y/z and, on MC, the matched truth interaction) and, on MC, truth.tsv.  Events are joined on
(run, subrun, event); the per-event record is the candidate with the highest numu_score (an event
has 0 or 1 candidate in this chain; the max is a guard).

Reports: candidates per arm, events passing numu > 0.9 and nue > 7 (and 4) per arm, the flips
(pass in one arm only), candidates that appear/disappear, vertex moves > --vtx-cm, and on MC the
truth class of every flip (signal numuCC / nueCC in the FV, or background) so an efficiency change
is read against its purity change.

usage: r3_pr_compare.py <products_A> <products_B> [--vtx-cm 5] [--tsv flips.tsv] [--label A,B]
"""
import argparse, csv, json, math, os, sys

NUMU_CUT, NUE_CUT, NUE_CUT2 = 0.9, 7.0, 4.0


def load(d):
    p = os.path.join(d, "candidates.tsv")
    if not os.path.exists(p):
        sys.exit("no %s" % p)
    ev = {}
    with open(p) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (int(r["run"]), int(r["subrun"]), int(r["event"]))
            try:
                s = float(r["numu_score"])
            except ValueError:
                s = -999.0
            if key not in ev or s > ev[key]["numu"]:
                ev[key] = dict(numu=s, nue=float(r.get("nue_score", -999) or -999),
                               enu=float(r.get("reco_Enu", 0) or 0),
                               x=float(r.get("nu_x", 0) or 0), y=float(r.get("nu_y", 0) or 0), z=float(r.get("nu_z", 0) or 0),
                               t_flav=r.get("t_flav", ""), t_ccnc=r.get("t_ccnc", ""), t_idx=r.get("t_idx", ""),
                               cid=r.get("cluster_id", ""))
    return ev


def in_fv(x, y, z):
    return 5 < abs(x) < 190 and abs(y) < 190 and 10 < z < 450


def truth_signal(d):
    """{(run,subrun,event): set of signal classes} from truth.tsv when present"""
    p = os.path.join(d, "truth.tsv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (int(r["run"]), int(r["subrun"]), int(r["event"]))
            flav, ccnc = r.get("flav", r.get("t_flav", "")), r.get("ccnc", r.get("t_ccnc", ""))
            try:
                fv = in_fv(float(r["vx"]), float(r["vy"]), float(r["vz"]))
            except (KeyError, ValueError):
                fv = True
            if ccnc == "CC" and fv:
                out.setdefault(key, set()).add(flav)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--vtx-cm", type=float, default=5.0)
    ap.add_argument("--tsv"); ap.add_argument("--label", default="A,B")
    a = ap.parse_args()
    la, lb = a.label.split(",")
    A, B = load(a.a), load(a.b)
    truth = truth_signal(a.a) or truth_signal(a.b)
    keys = sorted(set(A) | set(B))
    S = {"events_with_candidate_" + la: len(A), "events_with_candidate_" + lb: len(B),
         "candidate_only_" + la: 0, "candidate_only_" + lb: 0, "both": 0,
         "vertex_moved_gt_%.0fcm" % a.vtx_cm: 0}
    for cut, name in ((("numu", NUMU_CUT), "numu>0.9"), (("nue", NUE_CUT), "nue>7"), (("nue", NUE_CUT2), "nue>4")):
        S["pass_%s_%s" % (name, la)] = 0; S["pass_%s_%s" % (name, lb)] = 0
        S["flip_%s_%s_only" % (name, la)] = 0; S["flip_%s_%s_only" % (name, lb)] = 0
        if truth is not None:
            for who in (la, lb):
                S["flip_%s_%s_only_signal" % (name, who)] = 0
    rows = []
    for k in keys:
        xa, xb = A.get(k), B.get(k)
        if xa and xb:
            S["both"] += 1
            d = math.sqrt((xa["x"] - xb["x"]) ** 2 + (xa["y"] - xb["y"]) ** 2 + (xa["z"] - xb["z"]) ** 2)
            if d > a.vtx_cm:
                S["vertex_moved_gt_%.0fcm" % a.vtx_cm] += 1
        elif xa:
            S["candidate_only_" + la] += 1
        else:
            S["candidate_only_" + lb] += 1
        sig = ",".join(sorted(truth.get(k, []))) if truth is not None else ""
        flip_any = (xa is None) != (xb is None)
        for cut, name in ((("numu", NUMU_CUT), "numu>0.9"), (("nue", NUE_CUT), "nue>7"), (("nue", NUE_CUT2), "nue>4")):
            pa = xa is not None and xa[cut[0]] > cut[1]
            pb = xb is not None and xb[cut[0]] > cut[1]
            S["pass_%s_%s" % (name, la)] += pa; S["pass_%s_%s" % (name, lb)] += pb
            if pa != pb:
                flip_any = True
                who = la if pa else lb
                S["flip_%s_%s_only" % (name, who)] += 1
                if truth is not None and sig:
                    S["flip_%s_%s_only_signal" % (name, who)] += 1
        if flip_any or (xa and xb and math.sqrt((xa["x"] - xb["x"]) ** 2 + (xa["y"] - xb["y"]) ** 2 + (xa["z"] - xb["z"]) ** 2) > a.vtx_cm):
            rows.append((k, xa, xb, sig))
    print(json.dumps(S, indent=1))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("run\tsubrun\tevent\t%s_numu\t%s_nue\t%s_Enu\t%s_x\t%s_y\t%s_z\t%s_numu\t%s_nue\t%s_Enu\t%s_x\t%s_y\t%s_z\ttruth_signal\n"
                    % ((la,) * 6 + (lb,) * 6))
            for k, xa, xb, sig in rows:
                def col(x):
                    return "%.3f\t%.3f\t%.1f\t%.1f\t%.1f\t%.1f" % ((x["numu"], x["nue"], x["enu"], x["x"], x["y"], x["z"]) if x else (-999, -999, 0, 0, 0, 0))
                f.write("%d\t%d\t%d\t%s\t%s\t%s\n" % (k[0], k[1], k[2], col(xa), col(xb), sig))


if __name__ == "__main__":
    main()

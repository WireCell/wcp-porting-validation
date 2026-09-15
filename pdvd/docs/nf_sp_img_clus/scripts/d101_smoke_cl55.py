#!/usr/bin/env python3
"""doc pdvd/101 sec 6.3 -- the knob-ON smoke on the owner's Bee event, PDHD 028084_21 cluster 55.

Reads T_rec_charge of tracking-stm.root (TaggerCheckSTM single-track fit; the Bee display's
trajectory) and tracking-pr.root (PR multi-track fit) in two tags that consumed the SAME h28prod
pctree on the SAME pin (libpin_k): the knob-off tag and the knob-on tag.  Grades cluster CL with
d101_census.measure, i.e. the ruler of sec 1 (chord deviation, dQ/dx CV, dips, ACF(1),
corr(deviation, dQ/dx), lattice snap per plane).

Usage: d101_smoke_cl55.py [--cl 55] [--off d101smoff] [--on d101smkf]
"""
import argparse, os, sys
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d101_census as C  # noqa: E402

W = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/work/028084_21_"


def grade(tag, kind, cl):
    fn = W + tag + "/tracking-%s.root" % kind
    f = uproot.open(fn)
    tr = f["T_rec_charge"]
    br = ["x", "y", "z", "q", "nq", "pu", "pv", "pw", "pt", "cluster_id"]
    have = set(tr.keys())
    for extra in ("status", "pass", "flag_shower", "sub_cluster_id"):
        if extra in have:
            br.append(extra)
    t = tr.arrays(br, library="np")
    run = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    scale, off = float(run["dQdx_scale"][0]), float(run["dQdx_offset"][0])
    rows = []
    sel_cl = t["cluster_id"] == cl
    if kind == "stm":
        for p in sorted(set(t["pass"][sel_cl].tolist())):
            sel = sel_cl & (t["pass"] == p)
            for r in C.measure(t, scale, off, sel, "%s pass %d" % (kind, p), fn):
                rows.append(r)
    else:
        for s in sorted(set(t["sub_cluster_id"][sel_cl].tolist())) if "sub_cluster_id" in t else [None]:
            sel = sel_cl if s is None else sel_cl & (t["sub_cluster_id"] == s)
            for r in C.measure(t, scale, off, sel, "%s seg %s" % (kind, s), fn):
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cl", type=int, default=55)
    ap.add_argument("--off", default="d101smoff")
    ap.add_argument("--on", default="d101smkf")
    a = ap.parse_args()
    cols = ["n", "L", "dev_med", "dev_p90", "cv", "dip", "acf1", "r_dev_q", "snap_pu", "snap_pv", "snap_pw"]
    for kind in ("stm", "pr"):
        for tag in (a.off, a.on):
            for r in grade(tag, kind, a.cl):
                print("%-10s %-14s " % (tag, r["label"]) + "  ".join(
                    "%s=%.3f" % (c, r[c]) if isinstance(r[c], float) else "%s=%s" % (c, r[c]) for c in cols))


if __name__ == "__main__":
    main()

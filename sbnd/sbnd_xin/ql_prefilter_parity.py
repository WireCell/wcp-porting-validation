#!/usr/bin/env python3
"""C++/Python parity check for the over-prediction prefilter (post-build re-dump).

Reads the freshly regenerated dumps work/ql_evt<ID>/calib-evt<ID>.nl.json (produced
with reject_overpred=true, overpred_total_ratio=2.9, overpred_maxch_ratio=4.3) and
confirms:
  1. every bundle the Python metric predicts as an over-prediction reject carries the
     C++ potential_bad_match flag (the C++ actually dropped it) -- C++/Python parity;
  2. no ground-truth (hand-scanned) bundle is an over-prediction reject -- the hard
     constraint survives the real pipeline.

Reuses the exact metric + mask definitions from ql_prefilter_tune.py.
"""
import json
import os
import sys

import numpy as np

import ql_prefilter_tune as T

HERE = os.path.dirname(os.path.abspath(__file__))
CT, CM = 2.9, 4.3                      # the jsonnet cuts


def new_dump(ev):
    return os.path.join(HERE, "work", "ql_evt%d" % ev, "calib-evt%d.nl.json" % ev)


def check_mode(mode):
    events = T.events_for(mode)
    n_predict = n_parity_ok = n_gt_flagged = 0
    n_cpp_flag = 0
    for ev in events:
        state = os.path.join(HERE, "work", "ql_labels", mode, ".scan_state-evt%d.json" % ev)
        want = {tuple(k) for k in json.load(open(state))["selected"]}
        cal = json.load(T.calib_open(new_dump(ev)))
        ch_apa = np.array([o["apa"] for o in cal["opdets"]])
        ch_type = np.array([o["type"] for o in cal["opdets"]])
        ch_active = np.array([bool(o["active"]) for o in cal["opdets"]])
        flash_pe = {f["gid"]: np.asarray(f["pe"], float) for f in cal["flashes"]}
        flash_apa = {f["gid"]: f["apa"] for f in cal["flashes"]}
        apas = set(flash_apa.values())
        keep = {a: (ch_type == T.PMT_TYPE) & (ch_apa == a) & ch_active for a in apas}

        for b in cal["bundles"]:
            boundary = any(b[f] for f in T.BOUNDARY)
            m = T.bundle_metrics(b, flash_pe, flash_apa, keep)
            predict = (not boundary) and (m["r_total"] > CT or m["r_max"] > CM)
            is_gt = (b["flash_gid"], b["main_cluster"]) in want
            if b["potential_bad_match"]:
                n_cpp_flag += 1
            if predict:
                n_predict += 1
                if b["potential_bad_match"]:        # C++ also flagged it
                    n_parity_ok += 1
                else:
                    print("  PARITY MISS evt%d flash %s clu %s R_total=%.2f R_max=%.2f "
                          "(Python rejects, C++ kept)"
                          % (ev, b["flash_gid"], b["main_cluster"], m["r_total"], m["r_max"]))
                if is_gt:
                    n_gt_flagged += 1
                    print("  !! GT REJECTED evt%d flash %s clu %s"
                          % (ev, b["flash_gid"], b["main_cluster"]))
    print("[%s] python-predicted over-pred rejects=%d  parity(C++ also flagged)=%d/%d  "
          "GT rejected=%d  (C++ total potential_bad_match flags=%d)"
          % (mode, n_predict, n_parity_ok, n_predict, n_gt_flagged, n_cpp_flag))
    return n_predict == n_parity_ok and n_gt_flagged == 0


if __name__ == "__main__":
    ok = all(check_mode(m) for m in (sys.argv[1:] or ["data", "mc"]))
    print("\n%s" % ("PASS: full parity, no GT rejected" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

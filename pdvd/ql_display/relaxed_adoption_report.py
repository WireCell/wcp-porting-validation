#!/usr/bin/env python
"""Classify the REAL relaxed-tier adoptions of nm4* tags vs the frozen truth,
and verify strict additivity vs nm3 (no pre-existing decision may change).

Run from pdvd/:  python <this> nm4a nm4b nm4c
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "ql_display")
from ql_agree_score import RUN, NEVT, evt_of_idx, load_truth, find_truth, OBJECTIVE_TIERS

TOL = 0.5
truth = load_truth("work/ql_labels/wfresc/labels-evt298567.json",
                   "ql_display/decisions-cathxa")


def load(tag, idx):
    evt = evt_of_idx(idx)
    p = f"work/{RUN}_{idx}_{tag}/calib-evt{evt}.json"
    return json.load(open(p)), evt


def auto_pairs(calib):
    fb = {f["gid"]: f["time"] for f in calib["flashes"]}
    return {(b["main_cluster"], round(fb[b["flash_gid"]], 3))
            for b in calib["bundles"] if b.get("auto_selected")}


for tag in sys.argv[1:]:
    counts = defaultdict(int)
    details = []
    additivity_ok = True
    for idx in range(NEVT):
        calib, evt = load(tag, idx)
        nm3_calib, _ = load("nm3", idx)
        pairs_nm3 = auto_pairs(nm3_calib)
        pairs_new = auto_pairs(calib)
        lost = pairs_nm3 - pairs_new
        if lost:
            additivity_ok = False
            print(f"[{tag}] evt {evt} REGRESSION: nm3 pairs lost: {sorted(lost)}")
        fb = {f["gid"]: f["time"] for f in calib["flashes"]}
        ebu = defaultdict(list)
        for e in truth[evt]:
            ebu[e["uid"]].append(e)
        pos_times = {e["uid"] for e in truth[evt]
                     if e["positive"] and e["conf"] in OBJECTIVE_TIERS}
        for b in calib["bundles"]:
            if not b.get("cluster_rescue_relaxed"):
                continue
            uid, t = b["main_cluster"], fb[b["flash_gid"]]
            e = find_truth(ebu, uid, t, TOL)
            if e is None:
                cls = "wrongflash" if uid in pos_times else "unlabeled"
            elif e["positive"] and e["conf"] in OBJECTIVE_TIERS:
                cls = "recovered"
            elif not e["positive"] and e["conf"] in OBJECTIVE_TIERS:
                cls = "phantom"
            else:
                cls = "unlabeled"
            counts[cls] += 1
            details.append((cls, evt, uid, round(t, 1)))
    n = sum(counts.values())
    known = counts["recovered"] + counts["phantom"] + counts["wrongflash"]
    prec = f"{counts['recovered']/known:.2f}" if known else "n/a"
    print(f"[{tag}] adoptions {n}: " +
          ", ".join(f"{k}={counts[k]}" for k in
                    ("recovered", "phantom", "wrongflash", "unlabeled")) +
          f", precision {prec}, additivity {'OK' if additivity_ok else 'VIOLATED'}")
    for cls, evt, uid, t in details:
        if cls != "unlabeled":
            print(f"    {cls}: evt {evt} uid {uid} @ {t} us")

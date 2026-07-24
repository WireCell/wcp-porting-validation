#!/usr/bin/env python3
"""Doc 28 phase 0: classify the rc14 residual losses by governing lever.

For each (evt, uid) missed at rc14 but not at tm0 (uid in tm0k truth space):
map the uid into the rc14 cluster space (geometric map, same machinery as the
scorer), locate the truth-flash bundle in the rc14 calib dump, the bundle the
cluster actually auto-selected, and the auto-selected occupant(s) of the truth
flash.  Emit one row per pair plus aggregate lever statistics.

Run from pdvd/:  python3 docs/qlmatch/scripts/rl_forensics.py [base_tag=rc14]
"""
import json
import sys
from collections import Counter

EVT0, STEP = 298567, 14
TOL = 0.5  # us, same as scorer
BASE = sys.argv[1] if len(sys.argv) > 1 else "rc14"

FLAG_CH = {"at_cathode": "C", "at_x_boundary": "B", "two_boundary": "2",
           "close_to_PMT": "P", "window_truncated": "W", "consistent": "c",
           "xtpc_consistent": "x", "xtpc_pin": "p", "xtpc_scenario1": "1",
           "xtpc_cathode_rescued": "r", "spec_end": "s"}


def detail(tag):
    return json.load(open(f"work/ql_scores/{tag}/scores.json"))["detail"]


def pairs(det, key):
    out = {}
    for idx in range(18):
        evt = f"evt{EVT0 + STEP * idx}"
        for e in det[evt][key]:
            out[(evt, e["uid"])] = e
    return out


def brow(b, ftime):
    c2n = b["chi2"] / max(b["ndf"], 1)
    fl = "".join(ch for k, ch in FLAG_CH.items() if b.get(k))
    return (f"t={ftime:9.1f} ks={b['ks_dis']:.3f} c2n={c2n:6.1f} "
            f"str={b['strength']:.3f} cont={int(b['contained'])} fl=[{fl}]")


ref = pairs(detail("tm0"), "missed_list")
cur = pairs(detail(BASE), "missed_list")
lost = sorted(set(cur) - set(ref), key=lambda p: (int(p[0][3:]), p[1]))
print(f"{len(lost)} residual pairs (missed at {BASE}, not at tm0)\n")

cls = Counter()
truth_flags = Counter()
econ_strengths = []

by_evt = {}
for (evt, uid) in lost:
    by_evt.setdefault(evt, []).append(uid)

for evt in sorted(by_evt, key=lambda e: int(e[3:])):
    evtno = int(evt[3:])
    idx = (evtno - EVT0) // STEP
    dump = json.load(open(f"work/039252_{idx}_{BASE}/calib-evt{evtno}.json"))
    ftime = {f["gid"]: f["time"] for f in dump["flashes"]}
    for uid in by_evt[evt]:
        # NB missed_list uids are already in the scored tag's cluster space
        # (the scorer maps truth->tag and drops unmapped entries up front).
        ent = cur[(evt, uid)]
        muid = uid
        tag = f"{evt} uid {uid:7d} truth t={ent['time']:9.1f}"
        mine = [b for b in dump["bundles"] if b["main_cluster"] == muid]
        truth_b = [b for b in mine
                   if abs(ftime.get(b["flash_gid"], 1e9) - ent["time"]) <= TOL]
        sel_b = [b for b in mine if b["auto_selected"]]
        occupants = [b for b in dump["bundles"]
                     if b["auto_selected"] and b["main_cluster"] != muid
                     and abs(ftime.get(b["flash_gid"], 1e9) - ent["time"]) <= TOL]
        if not truth_b:
            cls["no-truth-bundle"] += 1
            print(f"{tag}  NO-TRUTH-BUNDLE ({len(mine)} bundles elsewhere,"
                  f" sel={len(sel_b)})")
            continue
        tb = max(truth_b, key=lambda b: b["strength"])
        for k in FLAG_CH:
            if tb.get(k):
                truth_flags[k] += 1
        if not tb["contained"]:
            truth_flags["NOT-contained"] += 1
        if not sel_b:
            cls["unmatched"] += 1
            print(f"{tag}  UNMATCHED   truth: {brow(tb, ent['time'])}"
                  f"  occ={len(occupants)}")
            econ_strengths.append(tb["strength"])
            continue
        sb = sel_b[0]
        st = ftime.get(sb["flash_gid"], float("nan"))
        cls["mis-pick"] += 1
        econ_strengths.append(tb["strength"])
        tks, sks = tb["ks_dis"], sb["ks_dis"]
        rel = "truth-ks-better" if tks < sks else "winner-ks-better"
        cls[rel] += 1
        if sb.get("xtpc_pin"):
            cls["winner-pinned"] += 1
        elif sb.get("xtpc_scenario1"):
            cls["winner-sc1-nopin"] += 1
        elif sb["strength"] == 0.0:
            cls["winner-str0-rescuepick"] += 1
        else:
            cls["winner-plain-lasso"] += 1
        print(f"{tag}  MIS-PICK ({rel})")
        print(f"    truth : {brow(tb, ent['time'])}  occ={len(occupants)}")
        print(f"    picked: {brow(sb, st)}")

print(f"\nclasses: {dict(cls)}")
print(f"truth-bundle flags: {dict(truth_flags)}")
zeros = sum(1 for s in econ_strengths if s == 0.0)
sub = sum(1 for s in econ_strengths if 0.0 < s < 0.05)
print(f"truth strengths: n={len(econ_strengths)} zero={zeros} "
      f"sub-cutoff(0-0.05)={sub} ok={len(econ_strengths)-zeros-sub}")

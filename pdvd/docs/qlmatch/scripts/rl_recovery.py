#!/usr/bin/env python3
"""Doc 28: per-variant recovery of the 50 residual pairs (missed at rc14,
not at tm0) for the LASSO-economy retune sweep.

Same machinery as doc 27's rc_recovery.py, rebased: reference = tm0
(production frame), base = rc14 (doc 27 op point).  Reads
work/ql_scores/<tag>/scores.json (run ql_agree_score.py on the tag first,
same truth flags as rc14).

Run from pdvd/:  python3 docs/qlmatch/scripts/rl_recovery.py rl1 rl2 ...
"""
import json
import sys

EVT0, STEP = 298567, 14
BASE = "rc14"


def detail(tag):
    return json.load(open(f"work/ql_scores/{tag}/scores.json"))["detail"]


def pairs(det, key):
    out = set()
    for idx in range(18):
        evt = f"evt{EVT0+STEP*idx}"
        for e in det[evt][key]:
            out.add((evt, e["uid"]))
    return out


def counts(det):
    a = p = m = 0
    for idx in range(18):
        evt = f"evt{EVT0+STEP*idx}"
        p += len(det[evt]["phantom_list"])
        m += len(det[evt]["missed_list"])
    return p, m


ref = detail("tm0")
base = detail(BASE)
lost = pairs(base, "missed_list") - pairs(ref, "missed_list")
print(f"{len(lost)} residual pairs (missed at {BASE}, not at tm0)\n")
print(f"{'tag':10s} {'recovered':>9s} {'still-lost':>10s} "
      f"{'new-missed':>10s} {'new-phantom':>11s}")
for tag in sys.argv[1:]:
    d = detail(tag)
    m = pairs(d, "missed_list")
    p_new = pairs(d, "phantom_list") - pairs(base, "phantom_list")
    m_new = m - pairs(base, "missed_list")
    rec = lost - m
    print(f"{tag:10s} {len(rec):9d} {len(lost & m):10d} "
          f"{len(m_new):10d} {len(p_new):11d}")

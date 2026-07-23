#!/usr/bin/env python3
"""Doc 27: per-variant recovery of the 59 pairs newly-missed at rtp1 vs tm0.

For each candidate tag, counts how many of those (evt, uid) pairs are no
longer missed (recovered) and how many NEW misses/phantoms the variant
introduces relative to rtp1.  Reads work/ql_scores/<tag>/scores.json (run
ql_agree_score.py on the tag first, same truth flags as rtp1).

Run from pdvd/:  python3 docs/qlmatch/scripts/rc_recovery.py rc1 rc2 rc3 rc4
"""
import json
import sys

EVT0, STEP = 298567, 14


def detail(tag):
    return json.load(open(f"work/ql_scores/{tag}/scores.json"))["detail"]


def pairs(det, key):
    out = set()
    for idx in range(18):
        evt = f"evt{EVT0+STEP*idx}"
        for e in det[evt][key]:
            out.add((evt, e["uid"]))
    return out


ref = detail("tm0")
rt = detail("rtp1")
lost = pairs(rt, "missed_list") - pairs(ref, "missed_list")
print(f"{len(lost)} target pairs (missed at rtp1, not at tm0)\n")
print(f"{'tag':8s} {'recovered':>9s} {'still-lost':>10s} {'new-missed':>10s} {'new-phantom':>11s}")
for tag in sys.argv[1:]:
    d = detail(tag)
    m = pairs(d, "missed_list")
    p_new = pairs(d, "phantom_list") - pairs(rt, "phantom_list")
    m_new = m - pairs(rt, "missed_list")
    rec = lost - m
    print(f"{tag:8s} {len(rec):9d} {len(lost & m):10d} {len(m_new):10d} {len(p_new):11d}")

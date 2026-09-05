#!/usr/bin/env python3
"""Sizing for the sec 13.2 FRAG category -- doc pdhd/docs/stm-tagger-chain.md.

WARNING: this reads the ANSWER KEY (for the stratum / direction columns only).
Do not run it while scanning.  It is deliberately NOT importable by the viewer
and lives outside stm_scan/ for that reason.

Repro:
  cd wcp-porting-img/pdhd && python3 docs/scripts/d45_frag_prevalence.py

Proxy for "the cluster is a piece of something longer": PCA the cluster, take
its two extreme points along the principal axis, and count OTHER charge within
R of each extreme.  An upper bound only -- a crossing cosmic buries an end too.
Kept off the scan sheet and off the display (feedback_blind_the_scan_sheet).
"""
import csv, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree

PDHD = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd"
WORK = os.path.join(PDHD, "work")
SHEET = os.path.join(PDHD, "docs", "scan", "pdhd_retile_scan_sheet.tsv")
KEY = os.path.join(PDHD, "docs", "scan", "pdhd_retile_scan_key.tsv")
R = 30.0

def rows(path):
    with open(path) as fh:
        return list(csv.DictReader([l for l in fh if not l.startswith("#")], delimiter="\t"))

items = rows(SHEET)
key = {"%s/%s" % (r["event"], r["cluster"]): r for r in rows(KEY)}

_c = {}
def charge(ev):
    if ev in _c: return _c[ev]
    zp = os.path.join(WORK, "029107_%s_stm0" % ev, "mabc-pr.zip")
    if not os.path.exists(zp):
        _c[ev] = None; return None
    z = zipfile.ZipFile(zp)
    n = [n for n in z.namelist() if "clustering-global" in n]
    if not n:
        _c[ev] = None; return None
    d = json.loads(z.read(n[0]))
    _c[ev] = (np.c_[d["x"], d["y"], d["z"]].astype(float), np.asarray(d["cluster_id"], int))
    return _c[ev]

out = []
for it in items:
    ch = charge(it["event"])
    if ch is None: continue
    P, C = ch
    m = C == int(it["cluster"])
    if m.sum() < 3: continue
    pts = P[m]; oth = P[~m]
    if len(oth) == 0: continue
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    t = (pts - c) @ ax
    ends = np.array([pts[t.argmin()], pts[t.argmax()]])
    tr = cKDTree(oth)
    near = [len(tr.query_ball_point(e, R)) for e in ends]
    k = key.get("%s/%s" % (it["event"], it["cluster"]))
    out.append(dict(scan_id=int(it["scan_id"]), tranche=int(it["tranche"]),
                    stratum=k["stratum"] if k else "?",
                    gain=(k and k["stm_retiler_on"] == "1"),
                    npts=int(it["npts"]), n0=near[0], n1=near[1],
                    both=min(near), either=max(near)))

print("items measured: %d / %d" % (len(out), len(items)))
def frac(sel, thr=20):
    s = [o for o in out if sel(o)]
    if not s: return "n=0"
    f = sum(1 for o in s if o["either"] >= thr)
    b = sum(1 for o in s if o["both"] >= thr)
    return "n=%3d  one end buried %3d (%2.0f%%)  both ends buried %3d (%2.0f%%)" % (
        len(s), f, 100.*f/len(s), b, 100.*b/len(s))

print("\nother charge within %.0f cm of a PCA extreme (>=20 pts = 'buried'):" % R)
print("  all                 ", frac(lambda o: True))
print("  stratum A (n>=200)  ", frac(lambda o: o["stratum"] == "A"))
print("  stratum B (n< 200)  ", frac(lambda o: o["stratum"] == "B"))
print("  knob GAINS the tag  ", frac(lambda o: o["gain"]))
print("  knob LOSES the tag  ", frac(lambda o: not o["gain"]))
print("  A & gains           ", frac(lambda o: o["stratum"]=="A" and o["gain"]))
print("  B & gains           ", frac(lambda o: o["stratum"]=="B" and o["gain"]))
print("  A & loses           ", frac(lambda o: o["stratum"]=="A" and not o["gain"]))
print("  B & loses           ", frac(lambda o: o["stratum"]=="B" and not o["gain"]))
print("\ntranche 1 only (first 63):")
print("  A & gains           ", frac(lambda o: o["tranche"]==1 and o["stratum"]=="A" and o["gain"]))
print("  B & gains           ", frac(lambda o: o["tranche"]==1 and o["stratum"]=="B" and o["gain"]))
print("  A & loses           ", frac(lambda o: o["tranche"]==1 and o["stratum"]=="A" and not o["gain"]))
print("  B & loses           ", frac(lambda o: o["tranche"]==1 and o["stratum"]=="B" and not o["gain"]))

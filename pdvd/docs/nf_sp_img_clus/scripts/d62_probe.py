#!/usr/bin/env python3
"""doc pdvd/62 (T3) -- the probe before any C++.

1. Every `pr54 isolated-residual drop` line in the baseline arm's logs, matched
   to the scan candidate of the SAME cluster (not just the same event), with
   the distance of the nearer residual endpoint to that candidate's stop.
   Companion clusters (a drop line naming a cluster that is not a candidate)
   are matched to the nearest candidate stop in the event and flagged.
2. The recoverable set: scan STM_MICHEL & michel_found==0 (class D) with a
   drop within R cm of the stop; the risk set: THRU with such a drop.
3. Doc 55 sec 15.1's range-energy cut recomputed on this arm, by name, at
   dis > 3 and dis > 5 cm.
Usage: python3 d62_probe.py <arm> <prep_dir>
"""
import sys, os, re, glob, collections
import numpy as np
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

arm, prep = sys.argv[1], sys.argv[2]
R = 20.0
rec = C.load_record()
P, missing = C.load_payloads(prep, rec)
keys = [k for k in rec if k in P]
print("arm %s: %d payloads, %d unmatched" % (arm, len(keys), len(missing)))

# ---- 1. drop lines --------------------------------------------------------
pat = re.compile(r"pr54 isolated-residual drop: cluster (\d+) n_points=(\d+) length=([\d.]+) cm dir_mag=([\d.]+) cm "
                 r"v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
drops = []   # (event, cluster, n, len, v1, v2)
for d in sorted(glob.glob(C.WORK + "/*_" + arm)):
    ev = os.path.basename(d)[: -len("_" + arm)]
    for lg in glob.glob(d + "/wct_pr_*.log"):
        for line in open(lg, errors="replace"):
            m = pat.search(line)
            if not m: continue
            g = m.groups()
            drops.append((ev, int(g[0]), int(g[1]), float(g[2]),
                          np.array([float(g[4]), float(g[5]), float(g[6])]),
                          np.array([float(g[7]), float(g[8]), float(g[9])])))
print("pr54 drops: %d in %d events" % (len(drops), len({d[0] for d in drops})))

stops = {}   # key -> stop xyz (the chain's stop, payload muon rr-min)
for k in keys:
    rr, q, xyz = C.profile(P[k]); stops[k] = xyz[0]
by_ev = collections.defaultdict(list)
for k in keys: by_ev[k.split("/")[0]].append(k)

own = 0; comp = 0; near_own = []; near_comp = []
for ev, cid, n, ln, v1, v2 in drops:
    cands = by_ev.get(ev, [])
    if not cands: continue
    key_own = "%s/%d" % (ev, cid)
    if key_own in stops:
        own += 1
        d = min(np.linalg.norm(v1 - stops[key_own]), np.linalg.norm(v2 - stops[key_own]))
        if d <= R: near_own.append((key_own, n, ln, d))
    else:
        comp += 1
        best = min(cands, key=lambda k: min(np.linalg.norm(v1 - stops[k]), np.linalg.norm(v2 - stops[k])))
        d = min(np.linalg.norm(v1 - stops[best]), np.linalg.norm(v2 - stops[best]))
        if d <= R: near_comp.append((best, cid, n, ln, d))
print("drops in a CANDIDATE cluster: %d (within %.0f cm of its own stop: %d)" % (own, R, len(near_own)))
print("drops in a non-candidate (companion/other) cluster: %d (within %.0f cm of some candidate stop: %d)" % (comp, R, len(near_comp)))

def verdict_of(k):
    r = rec[k]; v = P[k]["verdict"]
    return C.bare(r["verdict"]), int(v["michel_found"]), int(v["is_stm"])

tab = collections.Counter()
items_own = collections.defaultdict(list)
for k, n, ln, d in near_own: items_own[k].append((n, ln, d))
items_comp = collections.defaultdict(list)
for k, cid, n, ln, d in near_comp: items_comp[k].append((cid, n, ln, d))
allitems = set(items_own) | set(items_comp)
print("\ncandidates with >= 1 drop within %.0f cm of the stop (own cluster or companion): %d" % (R, len(allitems)))
print("%-16s %-12s mf is_stm  own-drops(n,len,d)                     comp-drops(cid,n,len,d)" % ("item", "scan"))
for k in sorted(allitems):
    sc, mf, st = verdict_of(k)
    tab[(sc, mf)] += 1
    o = " ".join("(%d,%.1f,%.1f)" % t for t in items_own.get(k, []))
    c = " ".join("(c%d,%d,%.1f,%.1f)" % t for t in items_comp.get(k, []))
    print("%-16s %-12s %d  %d      %-38s %s" % (k, sc, mf, st, o, c))
print("\nby (scan verdict, michel_found):")
for (sc, mf), n in sorted(tab.items()): print("   %-12s michel_found=%d : %d" % (sc, mf, n))
D = [k for k in allitems if C.judged(rec[k]) and C.is_michel(rec[k]) and not P[k]["verdict"]["michel_found"]]
T = [k for k in allitems if C.judged(rec[k]) and C.bare(rec[k]["verdict"]) == "THRU"]
print("RECOVERABLE (class D, scan michel & michel_found==0) with a nearby drop: %d  %s" % (len(D), sorted(D)))
print("RISK (THRU) with a nearby drop: %d" % len(T))
nD = sum(1 for k in keys if C.judged(rec[k]) and C.is_michel(rec[k]) and not P[k]["verdict"]["michel_found"])
print("class D total on this arm: %d" % nD)

# ---- 3. doc 55 sec 15.1 on this arm --------------------------------------
print("\n=== doc 55 sec 15.1 range-energy cut on %s (judged items) ===" % arm)
J = [k for k in keys if C.judged(rec[k])]
def score(pred):
    tp = fp = fn = tn = 0
    for k in J:
        mf = bool(P[k]["verdict"]["michel_found"]) and not pred(P[k]["verdict"])
        mi = C.is_michel(rec[k])
        if mf and mi: tp += 1
        elif mf and not mi: fp += 1
        elif not mf and mi: fn += 1
        else: tn += 1
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
    return tp, fp, fn, tn, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)
print("%-34s TP  FP  FN  TN  purity eff   F1" % "cut")
for name, pred in (("as shipped", lambda v: False),
                   ("drop conn2/3 & dis>5 & KE<10", lambda v: v["michel_conn_type"] in (2, 3) and (v["michel_dis_cm"] or 0) > 5.0 and (v["michel_ke_best"] or 0) < 10.0),
                   ("drop conn2/3 & dis>3 & KE<10", lambda v: v["michel_conn_type"] in (2, 3) and (v["michel_dis_cm"] or 0) > 3.0 and (v["michel_ke_best"] or 0) < 10.0),
                   ("drop any dis>3 & KE<10 (class E)", lambda v: (v["michel_dis_cm"] or 0) > 3.0 and (v["michel_ke_best"] or 0) < 10.0)):
    print("%-34s %3d %3d %3d %3d  %.3f %.3f %.3f" % ((name,) + score(pred)))
for dcut in (5.0, 3.0):
    print("\nitems the conn2/3 & dis>%.0f & KE<10 cut removes (name, scan, conn, dis, KE):" % dcut)
    for k in sorted(J):
        v = P[k]["verdict"]
        if v["michel_found"] and v["michel_conn_type"] in (2, 3) and (v["michel_dis_cm"] or 0) > dcut and (v["michel_ke_best"] or 0) < 10.0:
            print("   %-16s %-12s conn %d dis %.2f KE %.2f  %s" % (k, C.bare(rec[k]["verdict"]), v["michel_conn_type"], v["michel_dis_cm"], v["michel_ke_best"],
                  "** CASUALTY (scan michel)" if C.is_michel(rec[k]) else ""))
# conn_type histogram
h = collections.Counter(P[k]["verdict"]["michel_conn_type"] for k in keys)
print("\nmichel_conn_type histogram:", dict(sorted(h.items())))

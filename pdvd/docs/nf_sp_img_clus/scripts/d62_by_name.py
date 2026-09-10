#!/usr/bin/env python3
"""doc pdvd/62 (T3) -- set-by-name diffs between two arms' preps, with the
three T3 mechanisms traced item by item.

Usage: python3 d62_by_name.py <prep_before> <prep_after>

Prints: michel_found FP/TP removed/new by name (with the T3 counters on the
after arm), is_stm flips by name split into (a) items where only the residual
keep fired (pure refit perturbation), (b) items where a local piece was
admitted, (c) neither; the fire counts of every counter; and the C2
Michel-attachment census (scan-tagged michel segments matched by geometry
from the BEFORE arm into both arms) with the segments that changed role
named.
"""
import sys, os, collections
import numpy as np
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

def match_segment(pts0, pay):
    """census_score.py's rule: the arm segment whose points sit on the baseline segment (mean nearest-distance < 1.5 cm)."""
    best, bd = None, 1.5
    for s in pay["pf"]["seg"]:
        pts = C.seg_points(s)
        d = np.mean([np.linalg.norm(pts0 - p, axis=1).min() for p in pts[:: max(1, len(pts) // 12)]])
        if d < bd:
            best, bd = s, d
    return best

def sets(prep):
    rec = C.load_record()
    P, missing = C.load_payloads(prep, rec)
    keys = [k for k in rec if k in P]
    out = {}
    for k in keys:
        r = dict(rec[k]); v = P[k]["verdict"]
        r["_v"] = v; r["_is_stm"] = bool(v["is_stm"]); r["_mi"] = bool(v["michel_found"])
        out[k] = r
    J = [k for k in keys if C.judged(out[k])]
    mfp = {k for k in J if out[k]["_mi"] and not C.is_michel(out[k])}
    mtp = {k for k in J if out[k]["_mi"] and C.is_michel(out[k])}
    sfp = {k for k in J if out[k]["_is_stm"] and not C.is_stopper(out[k])}
    stp = {k for k in J if out[k]["_is_stm"] and C.is_stopper(out[k])}
    return out, mfp, mtp, sfp, stp, set(missing), P, rec

before, after = sys.argv[1], sys.argv[2]
ob, bmfp, bmtp, bsfp, bstp, bmiss, bP, rec = sets(before)
oa, amfp, amtp, asfp, astp, amiss, aP, _ = sets(after)
def cnt(v, k): return v.get(k) or 0
def trace(k):
    v = oa[k]["_v"]
    return "kept_main=%s kept_comp=%s local=%s range_veto=%s veto=%s conn=%s dis=%.2f ke=%.2f" % (
        v.get("n_kept_near_stop_main"), v.get("n_kept_near_stop_comp"), v.get("n_local_pieces"),
        v.get("n_michel_range_veto"), v.get("n_michel_veto"), v.get("michel_conn_type"),
        v.get("michel_dis_cm") or 0, v.get("michel_ke_best") or 0)

def f1(tp, fp, fn):
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1); return pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)
def census(o, tp, fp):
    J = [k for k in o if C.judged(o[k])]
    fn_m = sum(1 for k in J if C.is_michel(o[k]) and not o[k]["_mi"])
    return tp, fp, fn_m
print("=== michel_found census ===")
for nm, o, tp, fp in (("before", ob, bmtp, bmfp), ("after", oa, amtp, amfp)):
    t, f, n = census(o, len(tp), len(fp)); pu, ef, F = f1(t, f, n)
    print("%-7s TP %3d FP %3d FN %3d  purity %.3f eff %.3f F1 %.3f  (unmatched %d)" % (nm, t, f, n, pu, ef, F, len(bmiss if nm == "before" else amiss)))
print("\nmichel FP REMOVED:"); [print("   ", k, C.bare(rec[k]["verdict"]), trace(k)) for k in sorted(bmfp - amfp) if k in oa]
print("michel FP NEW:");     [print("   ", k, C.bare(rec[k]["verdict"]), trace(k)) for k in sorted(amfp - bmfp)]
print("michel TP LOST:");    [print("   ", k, C.bare(rec[k]["verdict"]), trace(k)) for k in sorted(bmtp - amtp) if k in oa]
print("michel TP NEW:");     [print("   ", k, C.bare(rec[k]["verdict"]), trace(k)) for k in sorted(amtp - bmtp)]

print("\n=== is_stm census ===")
for nm, o, tp, fp in (("before", ob, bstp, bsfp), ("after", oa, astp, asfp)):
    J = [k for k in o if C.judged(o[k])]
    fn_s = sum(1 for k in J if C.is_stopper(o[k]) and not o[k]["_is_stm"])
    pu, ef, F = f1(len(tp), len(fp), fn_s)
    print("%-7s TP %3d FP %3d FN %3d  purity %.3f eff %.3f F1 %.3f" % (nm, len(tp), len(fp), fn_s, pu, ef, F))
common = sorted(set(ob) & set(oa))
flips = [k for k in common if ob[k]["_is_stm"] != oa[k]["_is_stm"]]
print("is_stm differs on %d of %d common items" % (len(flips), len(common)))
cat = collections.Counter()
for k in flips:
    v = oa[k]["_v"]
    kept = cnt(v, "n_kept_near_stop_main") + cnt(v, "n_kept_near_stop_comp")
    c = "refit-only (kept>0, local==0)" if kept > 0 and cnt(v, "n_local_pieces") == 0 else ("local piece admitted" if cnt(v, "n_local_pieces") > 0 else "neither fired")
    cat[c] += 1
    print("   %-16s %-12s %d -> %d  [%s]  %s  rej_before=%s rej_after=%s" % (k, C.bare(rec[k]["verdict"]), ob[k]["_is_stm"], oa[k]["_is_stm"], c, trace(k),
          ob[k]["_v"].get("reject_names"), v.get("reject_names")))
print("is_stm flips by category:", dict(cat))
print("is_stm FP NEW:", sorted(asfp - bsfp)); print("is_stm FP REMOVED:", sorted(bsfp - asfp))
print("is_stm TP NEW:", sorted(astp - bstp)); print("is_stm TP LOST:", sorted(bstp - astp))

print("\n=== counters on the AFTER arm (all matched items) ===")
for f in ("n_kept_near_stop_main", "n_kept_near_stop_comp", "n_local_pieces", "n_michel_range_veto", "n_michel_veto"):
    items = [k for k in oa if cnt(oa[k]["_v"], f) > 0]
    print("%-22s fired on %3d items, %4d fires total" % (f, len(items), sum(cnt(oa[k]["_v"], f) for k in items)))
for f in ("n_local_pieces", "n_michel_range_veto"):
    items = [k for k in oa if cnt(oa[k]["_v"], f) > 0]
    if items:
        print("  %s items:" % f)
        for k in sorted(items):
            print("     %-16s %-12s mf %d->%d  %s" % (k, C.bare(rec[k]["verdict"]), ob.get(k, {}).get("_mi", -1), oa[k]["_mi"], trace(k)))
print("michel_conn_type histogram before:", dict(sorted(collections.Counter(ob[k]["_v"]["michel_conn_type"] for k in ob).items())))
print("michel_conn_type histogram after: ", dict(sorted(collections.Counter(oa[k]["_v"]["michel_conn_type"] for k in oa).items())))

print("\n=== C2. Michel attachment by NAME (scan-tagged michel segments, geometry from the BEFORE arm) ===")
names = {3: "role 3 michel", 1: "role 1 muon (swallowed)", 2: "role 2 delta", 5: "role 5 gamma", 6: "role 6 survey", None: "no role"}
att_b = collections.Counter(); att_a = collections.Counter(); changed = []
for k in common:
    if not C.is_michel(rec[k]): continue
    segs0, _ = C.seg_index(bP[k])
    for t, tag in rec[k]["tags"].items():
        if tag != "michel" or t not in segs0: continue
        pts0 = C.seg_points(segs0[t])
        def role_in(P):
            s = match_segment(pts0, P[k])
            if s is None: return "lost (no fitted segment there)"
            return names.get(P[k]["pf"]["chain_role"].get(str(s["id"])), "role ?")
        rb, ra = role_in(bP), role_in(aP)
        att_b[rb] += 1; att_a[ra] += 1
        if rb != ra: changed.append((k, t, rb, ra))
print("%-30s %6s %6s" % ("", "before", "after"))
for nm in sorted(set(att_b) | set(att_a), key=lambda n: -att_b[n]):
    print("%-30s %6d %6d" % (nm, att_b[nm], att_a[nm]))
print("segments whose role changed (%d):" % len(changed))
for k, t, rb, ra in changed:
    print("   %-16s seg %-8s %-30s -> %s   %s" % (k, t, rb, ra, trace(k)))

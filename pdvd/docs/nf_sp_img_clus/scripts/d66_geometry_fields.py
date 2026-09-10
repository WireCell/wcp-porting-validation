#!/usr/bin/env python3
"""doc pdvd/66 (T8) -- agreement between the chain's new profile-geometry fields
and the offline failure classes they were built to replace.

  end_arc_span >= 1.5     vs  class G (census_score.py: arc/span over the last
                              20 cm of the chain profile, all-pairs diameter
                              for <= 60 points, endpoint chord above)
  n_unsupported_segs > 0  vs  class H (a pf.seg segment >= 20 cm with dqdx_med
                              < 0.25 x median(q > 0) of the chain profile)

Every disagreement is listed by name with both sides' numbers.  Usage:
python3 d66_geometry_fields.py <prep_dir>
"""
import sys, collections
import numpy as np
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

prep = sys.argv[1]
rec = C.load_record()
P, missing = C.load_payloads(prep, rec)
keys = [k for k in rec if k in P]

def offline_G(pay):
    rr, q, xyz = C.profile(pay)
    take = np.where(rr <= 20.0)[0]
    if len(take) < 6: return None
    L = np.asarray(pay["muon"]["L"], float)[np.argsort(np.asarray(pay["muon"]["rr"], float))]
    arc = abs(L[take[-1]] - L[take[0]]); pts = xyz[take]
    if len(take) <= 60:
        span = max(np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))
    else:
        span = float(np.linalg.norm(pts[0] - pts[-1]))
    return (arc / span if span > 0.01 else None, arc, span, len(take))

def offline_H(pay):
    rr, q, xyz = C.profile(pay); live = q[q > 0]
    if len(live) < 20: return None
    plat = float(np.median(live)); out = []
    for s in pay["pf"]["seg"]:
        if (s["len_cm"] or 0) >= 20.0 and 0 < (s["dqdx_med"] or 0) < 0.25 * plat:
            out.append((s["id"], s["len_cm"], s["dqdx_med"] / plat))
    return out, plat

print("=== G: end_arc_span (C++) vs offline class G ===")
agree = collections.Counter(); dis = []
for k in keys:
    v = P[k]["verdict"]; cpp = v.get("end_arc_span"); off = offline_G(P[k])
    if cpp is None: continue
    c_flag = cpp >= 1.5; o_flag = bool(off and off[0] is not None and off[0] >= 1.5)
    agree[(c_flag, o_flag)] += 1
    if c_flag != o_flag: dis.append((k, cpp, v.get("n_end_pts"), off))
print("  (C++ flag, offline flag) counts:", dict(agree))
for d in dis: print("   ", d[0], "cpp arc/span %.2f n_end %s | offline %s" % (d[1], d[2], d[3]))
vals = [(P[k]["verdict"]["end_arc_span"], (offline_G(P[k]) or [None])[0]) for k in keys if P[k]["verdict"].get("end_arc_span") is not None]
both = [(a, b) for a, b in vals if a is not None and b is not None and a > 0]
if both:
    d = np.array([abs(a - b) for a, b in both]); print("  |cpp - offline| arc/span: median %.4f p90 %.4f max %.3f (n %d)" % (np.median(d), np.percentile(d, 90), d.max(), len(d)))

print("\n=== H: n_unsupported_segs (C++) vs offline class H ===")
agree = collections.Counter(); dis = []
for k in keys:
    v = P[k]["verdict"]; cpp = v.get("n_unsupported_segs"); off = offline_H(P[k])
    if cpp is None or cpp < 0: continue
    c_flag = cpp > 0; o_flag = bool(off and off[0])
    agree[(c_flag, o_flag)] += 1
    if c_flag != o_flag: dis.append((k, cpp, v.get("unsupported_frac_min"), v.get("plateau_med"), off))
print("  (C++ flag, offline flag) counts:", dict(agree))
for d in dis: print("   ", d[0], "cpp n %s frac_min %s plateau_med(20-40 live) %s | offline (segs, plateau=median q>0) %s" % (d[1], d[2], d[3], d[4]))
# the two plateau definitions differ: the chain's bragg plateau_med (20-40 cm, live rows) vs the offline median of all q>0
pl = []
for k in keys:
    v = P[k]["verdict"]; rr, q, xyz = C.profile(P[k]); live = q[q > 0]
    if (v.get("plateau_med") or 0) > 0 and len(live) >= 20: pl.append(v["plateau_med"] / float(np.median(live)))
print("  chain plateau_med / offline median(q>0): median %.3f, p10 %.3f, p90 %.3f (n %d)" % (np.median(pl), np.percentile(pl, 10), np.percentile(pl, 90), len(pl)))

print("\n=== the guard's would-be firing set (fields only; the knob is off on this arm) ===")
J = [k for k in keys if C.judged(rec[k])]
fire = [k for k in J if (P[k]["verdict"].get("end_arc_span") or 0) >= 1.5 or (P[k]["verdict"].get("n_unsupported_segs") or 0) > 0]
tp = [k for k in fire if P[k]["verdict"]["is_stm"] and C.is_stopper(rec[k])]
fp = [k for k in fire if P[k]["verdict"]["is_stm"] and not C.is_stopper(rec[k])]
print("  items the guard would fire on: %d of %d judged; of those is_stm=1: %d (TP %d, FP %d)" % (len(fire), len(J), len(tp) + len(fp), len(tp), len(fp)))
print("  is_stm TPs it would cost:", sorted(tp)); print("  is_stm FPs it would remove:", sorted(fp))

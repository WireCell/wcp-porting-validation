#!/usr/bin/env python3
"""doc pdvd/87 (doc 78 action item 7) -- the stop-local residual keep, arm by arm.

Usage: python3 d87_keep_census.py --arm <arm> --prep <arm prep> --base <arm> --base-prep <prep>

The keep-fire SET comes from the tree: n_kept_near_stop_main / _comp (always
persisted, doc 62) and n_floored_near_stop (doc 87, present only with a floor).
The log lines ("pr54 keep-isolated near-anchor:" and "... near-anchor floored:")
supply only each residual's terminals, length and d_anchor; they are joined at
CLUSTER level (the candidate's own cluster id; a line naming another cluster is a
companion's) and cross-checked against the tree counts.

  A  every candidate with a keep fire or a floored residual: the residuals, the
     record cell, and is_stm / michel_found / reject_bits / michel_conn_type /
     michel_ke_best / n_local_pieces base -> arm.
  B  candidates that moved on any verdict scalar WITHOUT a keep fire (must be 0).
  C  the merged-record census, base and arm (michel_found and is_stm TP/FP/FN,
     purity, efficiency, F1), and the changes by name.
The record is $STM_SCAN_RECORD (census_lib); the script prints the path it read.
"""
import argparse, collections, glob, os, re, sys
import numpy as np
import uproot
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--prep", required=True)
ap.add_argument("--base", default="p85vprod")
ap.add_argument("--base-prep", required=True)
a = ap.parse_args()
print("record: %s" % C.REC)
if not C.REC.endswith("smx1a_smx3_smx4_smx5_verdicts.json"):
    print("WARNING: not the merged smx5 record -- 039253_0/44 and 039349_30/45 would grade wrong")

KEEP = re.compile(r"pr54 keep-isolated near-anchor: cluster (\d+) n_points=(\d+) length=([\d.]+) cm d_anchor=([\d.]+) cm")
FLOOR = re.compile(r"pr54 keep-isolated near-anchor floored: cluster (\d+) n_points=(\d+) length=([\d.]+) cm d_anchor=([\d.]+) cm")
SC = ["is_stm", "michel_found", "reject_bits", "michel_conn_type", "michel_ke_best", "n_local_pieces",
      "n_kept_near_stop_main", "n_kept_near_stop_comp", "n_floored_near_stop", "stop_dis", "michel_dis_cm"]

def tree(arm):
    out = {}
    for d in sorted(glob.glob(C.WORK + "/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        try:
            f = uproot.open(fn)
            if "T_stm_michel" not in f: continue
            t = f["T_stm_michel"].arrays(library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, t["cluster_id"][i])] = {k: (t[k][i].item() if hasattr(t[k][i], "item") else t[k][i])
                                                       for k in SC if k in t}
    return out

def logs(arm):
    keep, flo = collections.defaultdict(list), collections.defaultdict(list)
    for d in sorted(glob.glob(C.WORK + "/*_" + arm)):
        ev = os.path.basename(d)[: -len(arm) - 1]
        for lg in sorted(glob.glob(d + "/wct_pr_*.log")):
            for line in open(lg, errors="replace"):
                for pat, dst in ((KEEP, keep), (FLOOR, flo)):
                    m = pat.search(line)
                    if m:
                        g = m.groups()
                        dst[(ev, int(g[0]))].append((int(g[1]), float(g[2]), float(g[3])))
    return keep, flo

TB, TA = tree(a.base), tree(a.arm)
LK, LF = logs(a.arm)
rec = C.load_record()
def cell(k):
    r = rec.get(k)
    if not r: return "unjudged"
    return r["verdict"]
print("base %s: %d candidates | arm %s: %d candidates" % (a.base, len(TB), a.arm, len(TA)))

# ---- A ----------------------------------------------------------------------
print("\n=== A. candidates with a keep fire or a floored residual (tree counts; residuals from the log) ===")
fire = sorted(k for k, v in TA.items() if v.get("n_kept_near_stop_main", 0) + v.get("n_kept_near_stop_comp", 0) > 0)
flrd = sorted(k for k, v in TA.items() if v.get("n_floored_near_stop", 0) > 0)
print("keep fires on %d candidates (%d residuals main, %d companion); floored on %d candidates (%d residuals)" % (
    len(fire), sum(TA[k]["n_kept_near_stop_main"] for k in fire), sum(TA[k]["n_kept_near_stop_comp"] for k in fire),
    len(flrd), sum(TA[k].get("n_floored_near_stop", 0) for k in flrd)))
cells = collections.Counter("%s/is_stm%d" % (cell(k), int(TB.get(k, TA[k])["is_stm"])) for k in fire)
print("keep-fire cells (record / base is_stm): %s" % dict(sorted(cells.items())))
xcheck = []
for k in sorted(set(fire) | set(flrd)):
    ev, c = k.split("/")
    v, b = TA[k], TB.get(k, {})
    own_k = LK.get((ev, int(c)), []); own_f = LF.get((ev, int(c)), [])
    if len(own_k) > v["n_kept_near_stop_main"] or len(own_f) > v.get("n_floored_near_stop", 0):
        xcheck.append(k)
    def ch(n, fmt="%d"):
        x, y = b.get(n), v.get(n)
        if x is None or y is None: return "-"
        return (fmt % x) + ("" if x == y else ("->" + fmt % y))
    print("  %-14s %-10s kept main %d comp %d floored %d | kept %s floored %s | is_stm %s mf %s bits %s conn %s ke %s local %s stop_dis %s" % (
        k, cell(k), v["n_kept_near_stop_main"], v["n_kept_near_stop_comp"], v.get("n_floored_near_stop", 0),
        [(n, round(l, 2), round(d, 1)) for n, l, d in own_k], [(n, round(l, 2), round(d, 1)) for n, l, d in own_f],
        ch("is_stm"), ch("michel_found"), ch("reject_bits"), ch("michel_conn_type"), ch("michel_ke_best", "%.1f"),
        ch("n_local_pieces"), ch("stop_dis", "%.1f")))
comp_lines = [(ev, c, x) for (ev, c), xs in LK.items() for x in xs if "%s/%d" % (ev, c) not in TA]
print("keep lines naming a non-candidate cluster (companions): %d %s" % (len(comp_lines), comp_lines[:8]))
print("cross-check: candidates with more own-cluster log lines than the tree count: %d %s" % (len(xcheck), xcheck))

# ---- B ----------------------------------------------------------------------
print("\n=== B. verdict scalars moved on a candidate with NO keep fire (must be 0) ===")
nofire = []
for k, v in sorted(TA.items()):
    if k in fire or k not in TB: continue
    moved = [n for n in ("is_stm", "michel_found", "reject_bits", "michel_conn_type", "michel_ke_best", "n_local_pieces")
             if n in v and n in TB[k] and not (v[n] == TB[k][n] or (isinstance(v[n], float) and np.isnan(v[n]) and np.isnan(TB[k][n])))]
    if moved: nofire.append((k, moved))
print("  %d %s" % (len(nofire), nofire[:20]))
print("  candidates only in base: %s | only in arm: %s" % (sorted(set(TB) - set(TA)), sorted(set(TA) - set(TB))))

# ---- C ----------------------------------------------------------------------
def sets(prep):
    P, missing = C.load_payloads(prep, rec)
    J = [k for k in rec if k in P and C.judged(rec[k])]
    mi = {k for k in J if P[k]["verdict"]["michel_found"]}
    st = {k for k in J if P[k]["verdict"]["is_stm"]}
    M = {k for k in J if C.is_michel(rec[k])}; S = {k for k in J if C.is_stopper(rec[k])}
    return set(J), mi, st, M, S
def f1(tp, fp, fn):
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1); return pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)
print("\n=== C. merged-record census, %s -> %s ===" % (a.base, a.arm))
Jb, mib, stb, Mb, Sb = sets(a.base_prep)
Ja, mia, sta, Ma, Sa = sets(a.prep)
J = Jb & Ja
print("  judged items with a payload on both: %d (base only %d, arm only %d)" % (len(J), len(Jb - Ja), len(Ja - Jb)))
for lab, xb, xa, T in (("michel_found", mib, mia, Mb), ("is_stm", stb, sta, Sb)):
    for nm, x in (("base", xb), ("arm ", xa)):
        tp, fp, fn = len(x & T & J), len((x - T) & J), len((T - x) & J)
        pu, ef, f = f1(tp, fp, fn)
        print("  %-12s %s TP %3d FP %3d FN %3d | purity %.3f eff %.3f F1 %.3f" % (lab, nm, tp, fp, fn, pu, ef, f))
    print("    TP gained %s" % sorted((xa - xb) & T & J))
    print("    TP lost   %s" % sorted((xb - xa) & T & J))
    print("    FP new    %s" % ["%s(%s)" % (k, cell(k)) for k in sorted((xa - xb) - T) if k in J])
    print("    FP gone   %s" % ["%s(%s)" % (k, cell(k)) for k in sorted((xb - xa) - T) if k in J])
thru = sorted(k for k in fire if cell(k) == "THRU")
print("  THRU items with a keep fire: %d %s" % (len(thru), thru))

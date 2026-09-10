#!/usr/bin/env python3
"""doc pdvd/70 -- size the second-round proposals on the CURRENT production arm.

Read-only.  Every table in doc 70 sec 3-5 comes from this script's stdout.

Inputs (all pre-existing, nothing is written except --out):
  * the merged grading record (STM_SCAN_RECORD, doc pdvd/68 sec 2);
  * the production arm's payloads (--prep, default the anchor arm d68a3 = today's bag);
  * the two baseline preps the scan tags refer to (prep-pdvd for smx1a items,
    prep-pdvd-smx3 for the owner's smx3 items) -- a scan tag names a segment of the
    payload the scanner looked at, so it is matched into the arm by geometry, the
    same rule census_score.py's C2 uses;
  * the arm's per-event PR logs (--work, for the classifier's per-stop-arm DEBUG line,
    doc pdvd/63).

Sections
  A  census + the four is_stm cells split by whether a Michel was found (Q2)
  B  the Michel-quality grid: FN recovered vs THRU admitted (Q2, proposal P1)
  C  the found-stopper Michel misses, one block per item, with the scan's michel
     segments matched into the arm and the C++ stop-arm line (Q3)
  D  gamma-tagged scan segments: role in the arm, distance to the stop, length (Q4)

Usage:
  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_verdicts.json \
  python3 d70_sizing.py --prep /home/xqian/tmp/d68/prep_d68a3 --arm d68a3 [--out DIR]
"""
import argparse, collections, glob, json, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/d68/prep_d68a3")
ap.add_argument("--arm", default="d68a3")
ap.add_argument("--work", default=IMG + "/pdvd/work")
ap.add_argument("--base1", default=C.PREP_DEFAULT, help="baseline prep of the smx1a items")
ap.add_argument("--base2", default=IMG + "/pdhd/stm_michel_scan/prep-pdvd-smx3", help="baseline prep of the smx3 items")
ap.add_argument("--out", default=None, help="scratch dir for the per-item JSON")
a = ap.parse_args()

R = C.load_record()
P, missing = C.load_payloads(a.prep, R)
B1, _ = C.load_payloads(a.base1, R)
B2, _ = C.load_payloads(a.base2, R)
V = lambda k: P[k]["verdict"]
J = sorted(k for k in R if C.judged(R[k]) and k in P)
print("record %s: %d records, %d judged with a payload in %s (%d unmatched)" % (os.path.basename(C.REC), len(R), len(J), a.prep, len(missing)))


def match_segment(pts0, pay):
    """census_score.py's rule: the arm segment whose points sit on the baseline segment
    (best mean nearest-distance under 1.5 cm)."""
    best, bd = None, 1.5
    for s in pay["pf"]["seg"]:
        pts = C.seg_points(s)
        d = np.mean([np.linalg.norm(pts0 - p, axis=1).min() for p in pts[:: max(1, len(pts) // 12)]])
        if d < bd:
            best, bd = s, d
    return best


def tagged(k, tag):
    """(baseline segment id, matched arm segment or None) for every scan segment tagged `tag`."""
    B = B1.get(k) or B2.get(k)
    if B is None:
        return []
    segs0, _ = C.seg_index(B)
    out = []
    for t, tg in (R[k].get("tags") or {}).items():
        if tg == tag and t in segs0:
            out.append((t, match_segment(C.seg_points(segs0[t]), P[k])))
    return out


# ------------------------------------------------------------------ A
STOP = {k: C.is_stopper(R[k]) for k in J}
cell = collections.defaultdict(list)
for k in J:
    cell[("TP" if STOP[k] else "FP") if V(k)["is_stm"] else ("FN" if STOP[k] else "TN")].append(k)
print("\n=== A. is_stm cells on the arm, split by michel_found / a stop arm / bragg_valid ===")
print("%-4s %5s %14s %12s %12s %12s" % ("cell", "n", "scan-michel", "michel_found", "stop_arm>0", "bragg_valid"))
for c in ("TP", "FP", "FN", "TN"):
    S = cell[c]
    print("%-4s %5d %14d %12d %12d %12d" % (c, len(S), sum(C.is_michel(R[k]) for k in S), sum(V(k)["michel_found"] for k in S),
                                             sum(V(k)["n_stop_arms"] > 0 for k in S), sum(bool(V(k)["bragg_valid"]) for k in S)))
FNm = [k for k in cell["FN"] if V(k)["michel_found"]]
TNm = [k for k in cell["TN"] if V(k)["michel_found"]]
print("\nFN reject-bit combinations:")
for combo, n in collections.Counter(tuple(C.reject_names(V(k))) for k in cell["FN"]).most_common():
    print("  %3d  %s" % (n, ",".join(combo)))
print("FN with michel_found=1 (%d), by reject bits:" % len(FNm))
for combo, n in collections.Counter(tuple(C.reject_names(V(k))) for k in FNm).most_common():
    print("  %3d  %s" % (n, ",".join(combo)))
print("FN with michel_found=1:", " ".join(FNm))
print("TN with michel_found=1 (%d):" % len(TNm), " ".join(TNm))
print("FP items:", " ".join("%s(%s)" % (k, R[k]["verdict"]) for k in cell["FP"]))

# ------------------------------------------------------------------ B
print("\n=== B. Michel quality on the two Michel-carrying populations (P1 sizing) ===")
cols = ("michel_conn_type", "michel_ke_best", "michel_len", "michel_kink_deg", "michel_dis_cm", "michel_mip", "n_dots", "michel_n_clusters")
for nm, S in (("FN + Michel (stopper missed, Michel found)", FNm), ("TN + Michel (THRU by scan, Michel found)", TNm)):
    print("\n-- %s, n=%d" % (nm, len(S)))
    print("%-14s %-9s %-6s " % ("key", "scan", "conf") + " ".join("%8s" % c.replace("michel_", "")[:8] for c in cols) + "  reject")
    for k in sorted(S):
        v = V(k)
        print("%-14s %-9s %-6s " % (k, R[k]["verdict"][:9], str(R[k].get("confidence"))[:6]) +
              " ".join("%8s" % (("%.2f" % v[c]) if isinstance(v[c], float) else str(v[c])) for c in cols) + "  " + ",".join(C.reject_names(v)))
TPm = [k for k in cell["TP"] if V(k)["michel_found"]]


def q(S, c):
    return np.percentile(np.array([V(k)[c] or 0 for k in S], float), [10, 50, 90])


print("\nquantiles 10/50/90:        TP+Michel            FN+Michel            TN+Michel")
for c in ("michel_ke_best", "michel_len", "michel_kink_deg"):
    print("  %-16s %s %s %s" % (c, np.round(q(TPm, c), 1), np.round(q(FNm, c), 1), np.round(q(TNm, c), 1)))
print("  conn_type       TP %s  FN %s  TN %s" % (dict(collections.Counter(V(k)["michel_conn_type"] for k in TPm)),
                                                 dict(collections.Counter(V(k)["michel_conn_type"] for k in FNm)),
                                                 dict(collections.Counter(V(k)["michel_conn_type"] for k in TNm))))


def gate(k, ke, ln):
    # P1 clears no_bragg + shape_flat ONLY: an item keeping any other bit stays
    # rejected.  (The first version of this gate, used for doc 70's first table,
    # omitted that test and counted 5 profile_sparse / continuation items; doc 70
    # sec 9 corrects the table.)  The boundary clause is then redundant, kept.
    v = V(k)
    return v["michel_conn_type"] in (1, 2) and (v["michel_ke_best"] or 0) >= ke and (v["michel_len"] or 0) >= ln \
        and "stop_near_boundary" not in C.reject_names(v) \
        and not (set(C.reject_names(v)) - {"no_bragg", "shape_flat"})


nTP, nFP = len(cell["TP"]), len(cell["FP"])
nS = nTP + len(cell["FN"])
print("\nP1 grid: conn_type in {1,2} & KE >= ke & len >= ln & not stop_near_boundary -> clears no_bragg + shape_flat")
print("%6s %6s %10s %10s %8s %8s %8s %8s" % ("ke", "len", "FN->TP", "TN->FP", "TP", "FP", "purity", "eff"))
for ke in (5, 8, 10, 12, 15):
    for ln in (0, 2, 3, 5):
        g = sum(gate(k, ke, ln) for k in FNm)
        b = sum(gate(k, ke, ln) for k in TNm)
        print("%6d %6d %10d %10d %8d %8d %8.3f %8.3f" % (ke, ln, g, b, nTP + g, nFP + b, (nTP + g) / (nTP + g + nFP + b), (nTP + g) / nS))
print("TN admitted at KE>=10, len>=3:", " ".join("%s(%s,%.1f MeV,%.1f cm)" % (k, R[k].get("confidence"), V(k)["michel_ke_best"], V(k)["michel_len"]) for k in TNm if gate(k, 10, 3)))
print("FN NOT recovered at KE>=10, len>=3:", " ".join("%s(%.1f MeV,%.1f cm,%s)" % (k, V(k)["michel_ke_best"], V(k)["michel_len"], ",".join(C.reject_names(V(k)))) for k in FNm if not gate(k, 10, 3)))

# ------------------------------------------------------------------ C
print("\n=== C. the found-stopper Michel misses (is_stm=1, michel_found=0) ===")
D = [k for k in J if C.is_michel(R[k]) and V(k)["is_stm"] and not V(k)["michel_found"]]
Dall = [k for k in R if C.judged(R[k]) and C.is_michel(R[k]) and (k not in P or not V(k)["michel_found"])]
print("michel FN on the union: %d; no candidate on the arm: %s" % (len(Dall), " ".join(k for k in Dall if k not in P)))
print("michel FN with is_stm=0 (stopper missed too): %d; with is_stm=1: %d" % (len([k for k in Dall if k in P and not V(k)["is_stm"]]), len(D)))
LOGLINE = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg shower (\d) terminal (\d)")


def stop_arm_lines(k):
    ev, cl = k.split("/")
    logs = glob.glob(os.path.join(a.work, "%s_%s" % (ev, a.arm), "wct_pr_*.log"))
    out = {}
    for lg in logs:
        with open(lg, errors="replace") as fh:
            for line in fh:
                m = LOGLINE.search(line)
                if m and m.group(1) == cl:
                    out[m.group(2)] = "kind %s len %s far_len %s mip %s kink %s shower %s" % (m.group(3), m.group(4), m.group(5), m.group(6), m.group(7), m.group(8))
    return out


for k in sorted(D):
    pay = P[k]; v = V(k); role = pay["pf"]["chain_role"]; segs, cl = C.seg_index(pay)
    rr, qq, xyz = C.profile(pay); stop = xyz[0]
    print("\n%s %s michel_kind=%s src=%s conf=%s | n_stop_arms=%s n_body_other=%s n_stop_other=%s n_michel_veto=%s n_michel_range_veto=%s n_dots=%s n_local_pieces=%s n_stop_gammas=%s"
          % (k, R[k]["verdict"], R[k].get("michel_kind"), R[k].get("source", "smx1a"), R[k].get("confidence"), v["n_stop_arms"], v["n_body_other"], v["n_stop_other"],
             v["n_michel_veto"], v["n_michel_range_veto"], v["n_dots"], v["n_local_pieces"], v["n_stop_gammas"]))
    note = (R[k].get("evidence") or "").strip()
    if note:
        print("   scan note: %s" % note[:200])
    arms = stop_arm_lines(k)
    for t, s in tagged(k, "michel"):
        if s is None:
            print("   michel-seg %s: no fitted segment there in the arm" % t); continue
        fails, mip, kink = C.michel_gate_failures(pay, s)
        print("   michel-seg %s -> arm seg %s role=%s len=%.1f mip=%.2f offline-kink=%.0f d_end=%.1f offline-gates=%s%s"
              % (t, s["id"], role.get(str(s["id"])), s["len_cm"], mip, kink, C.arm_at_fit_end(pay, s), fails or ["pass"],
                 ("  C++ stop-arm: " + arms[str(s["id"])]) if str(s["id"]) in arms else ""))
    for sid, ln in arms.items():
        if sid not in {str(s["id"]) for _, s in tagged(k, "michel") if s is not None}:
            print("   C++ stop-arm (untagged) seg %s: %s role=%s" % (sid, ln, role.get(sid)))
    near = [(s["id"], round(s["len_cm"], 1), round((s.get("dqdx_med") or 0) / C.MIP_MEDIAN, 2),
             round(float(np.linalg.norm(C.seg_points(s) - stop, axis=1).min()), 1), role.get(str(s["id"])))
            for s in pay["pf"]["seg"] if np.linalg.norm(C.seg_points(s) - stop, axis=1).min() < 8 and role.get(str(s["id"])) != 1]
    print("   segments within 8 cm of the stop (id, len, mip, d, role):", near[:8] or "none")

# ------------------------------------------------------------------ D
print("\n=== D. gamma-tagged scan segments in the arm (P4 sizing) ===")
tagc = collections.Counter()
for k in R:
    tagc.update((R[k].get("tags") or {}).values())
print("segment tags over the record:", dict(tagc))
print("items with >= 1 gamma tag: %d, of which scan-Michel: %d" % (sum(1 for k in R if "gamma" in (R[k].get("tags") or {}).values()),
                                                                     sum(1 for k in R if "gamma" in (R[k].get("tags") or {}).values() and C.is_michel(R[k]))))
print("michel_kind among scan-Michel items:", dict(collections.Counter(str(R[k].get("michel_kind")) for k in R if C.is_michel(R[k]))))
rows = []
for k in J:
    rr, qq, xyz = C.profile(P[k]); stop = xyz[0]; role = P[k]["pf"]["chain_role"]
    for t, s in tagged(k, "gamma"):
        if s is None:
            rows.append(dict(key=k, michel=C.is_michel(R[k]), found=int(V(k)["michel_found"]), role="lost", d=None, len=None)); continue
        rows.append(dict(key=k, michel=C.is_michel(R[k]), found=int(V(k)["michel_found"]), role=role.get(str(s["id"])),
                         d=float(np.linalg.norm(C.seg_points(s) - stop, axis=1).min()), len=s["len_cm"]))
ROLE = {3: "role 3 michel", 5: "role 5 capture gamma", 6: "role 6 survey", 4: "role 4", 7: "role 7 other", 2: "role 2 delta", 1: "role 1 muon", None: "no role", "lost": "lost"}
print("gamma segments matched into the arm: %d" % len(rows))
for nm, sel in (("all", lambda x: True), ("on scan-Michel items", lambda x: x["michel"]), ("on non-Michel items", lambda x: not x["michel"]),
                ("on scan-Michel items with michel_found=1", lambda x: x["michel"] and x["found"])):
    sub = [x for x in rows if sel(x)]
    print("\n-- %s: %d segments on %d items" % (nm, len(sub), len({x["key"] for x in sub})))
    for rl, n in collections.Counter(x["role"] for x in sub).most_common():
        print("   %-22s %4d" % (ROLE.get(rl, str(rl)), n))
    ds = np.array([x["d"] for x in sub if x["d"] is not None])
    if len(ds):
        print("   d_stop q25/50/75/90/max: %.1f / %.1f / %.1f / %.1f / %.1f cm" % tuple(np.percentile(ds, [25, 50, 75, 90, 100])))
        for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 35), (35, 60), (60, 1e9)):
            m = [x for x in sub if x["d"] is not None and lo <= x["d"] < hi]
            print("   d in [%2g,%3g) cm: %3d   %s" % (lo, hi, len(m), dict(collections.Counter(ROLE.get(x["role"], str(x["role"])) for x in m))))
        ls = np.array([x["len"] for x in sub if x["len"] is not None])
        print("   len q50/90/max: %.1f / %.1f / %.1f cm" % tuple(np.percentile(ls, [50, 90, 100])))
outside = [x for x in rows if x["michel"] and x["found"] and x["role"] != 3]
print("\nscan-Michel items with michel_found=1 whose gamma segments sit OUTSIDE the Michel object: %d segments on %d items"
      % (len(outside), len({x["key"] for x in outside})))
ng = [k for k in J if not C.is_michel(R[k]) and R[k].get("michel_kind") == "detached dots" and V(k)["n_stop_gammas"] >= 1]
print("STM_ONLY / 'detached dots' items whose dots are reconstructed as capture gamma (n_stop_gammas >= 1): %d" % len(ng))
print("  ", " ".join(ng))

if a.out:
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "d70_cells.json"), "w") as fh:
        json.dump(dict(cells={c: cell[c] for c in cell}, FN_michel=FNm, TN_michel=TNm, D=D, gamma_rows=rows), fh, indent=1)
    print("\nwrote", os.path.join(a.out, "d70_cells.json"))

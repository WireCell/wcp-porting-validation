#!/usr/bin/env python3
"""docs pdvd/72 (P3b) + pdvd/73 (P2) sec 2 -- the production re-measurement that
set both designs, read-only over existing arms' PR logs and one prep.

  1  the moved-stop veto's fires on production (p4v50) and each item's stop-arm
     kink on six arms (the 60 cm survey arms d68a3 / d71vsp, bare 35 cm p4voff /
     p4v35, bare 50 cm p4v50 = production, bare 60 cm p4v60), with the record's
     verdict; the decision interval for a kink threshold.
  2  every kOther stop arm on p4v50 re-judged with P2 (a) 0.15 MIP when kink >=
     60 deg and (b) a shower-flagged far_len cap, from the DEBUG line (which
     carries every classifier input; far_len > 25 cm there is the capped
     walk's partial sum, a lower bound).
  3  doc 70's four named P2 items on production.
  4  039349_11/19 arm 19004's whole far subtree, by payload endpoint matching.
  5  why (c) cannot be sized from payloads: the 15 cm kink re-derived from the
     payload's segment points against the C++ one.

Usage:  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json python3 d72_sizing.py
"""
import collections, glob, json, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# census_lib reads STM_SCAN_RECORD when it is imported: default it FIRST
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

R = C.load_record()
PREP = "/home/xqian/tmp/p4/prep_p4v50"
A = C.load_payloads(PREP, R)[0]
LL = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg shower (\d) terminal (\d)")


def rv(k):
    return ("%s/%s" % (R[k]["verdict"], R[k].get("michel_kind"))) if k in R else "not-judged"


def arms(arm):
    out = collections.defaultdict(list)
    for f in glob.glob(IMG + "/pdvd/work/*_%s/wct_pr_*.log" % arm):
        ev = f.split("/")[-2][: -len(arm) - 1]
        for ln in open(f, errors="replace"):
            m = LL.search(ln)
            if m:
                g = m.groups()
                out["%s/%s" % (ev, g[0])].append((int(g[1]), int(g[2])) + tuple(float(x) for x in g[3:7]) + (int(g[7]), int(g[8])))
    return out


# ------------------------------------------------------------------ 1
print("=== 1. the moved-stop veto on production (p4v50), and the kink on six arms ===")
SIX = ("d68a3", "d71vsp", "p4voff", "p4v35", "p4v50", "p4v60")
AR = {a: arms(a) for a in SIX}
fires = sorted(k for k, p in A.items() if p["verdict"]["n_michel_veto"] > 0)
print("%-14s %-24s %6s | %s" % ("item", "record", "KE", " ".join("%-14s" % a for a in SIX)))
km, kx = [], []
for k in fires:
    v = A[k]["verdict"]
    cells = []
    for a in SIX:
        L = [x for x in AR[a].get(k, []) if x[1] == 1]
        cells.append("%5.1f (%s)" % (L[0][5], L[0][0]) if L else "-")
    print("%-14s %-24s %6.2f | %s" % (k, rv(k), v["michel_ke_best"], " ".join("%-14s" % c for c in cells)))
    kk = v["michel_kink_deg"]
    (km if (k in R and C.is_michel(R[k])) else kx).append(kk)
print("record Michels' kinks %s; the others' %s" % (sorted(km), sorted(kx)))
if km and kx:
    # a threshold t spares every item with kink >= t
    lo = max(kx)
    edges = sorted(x for x in km if x > lo)
    prev = lo
    for x in edges:
        n_sp = sum(1 for y in km if y >= x)
        print("a threshold in (%.1f, %.1f] spares %d record Michel(s) and no other item on p4v50" % (prev, x, n_sp))
        prev = x
    print("60 deg spares %d (%s); it sits %.1f deg above the nearest non-Michel"
          % (sum(1 for y in km if y >= 60), ", ".join("%.1f" % y for y in km if y >= 60), 60 - lo))
    print("across the six arms the non-Michels' largest kink is %.1f" % max(x[5] for a in SIX for k in fires if not (k in R and C.is_michel(R[k])) for x in AR[a].get(k, []) if x[1] == 1))

# ------------------------------------------------------------------ 2
print("\n=== 2. kOther stop arms on p4v50 that P2 (a) 0.15 MIP @ kink >= 60 and (b) the shower far_len cap would admit ===")
P50 = AR["p4v50"]


def admit(ln, far, mip, kink, sh, mlt=-1.0, fsm=-1.0):
    kink_ok = kink >= 0
    if kink_ok and kink < 20 and ln > 3 and 0.7 <= mip <= 1.3:
        return False
    reach = (ln <= 25 and far <= fsm) if (fsm >= 0 and sh) else (ln + far <= 25)
    lo = mip > 0.3 or (mlt >= 0 and kink_ok and kink >= 60 and mip > mlt)
    return reach and lo and mip < 2.0 and ((sh and (not kink_ok or kink >= 15)) or (kink_ok and kink >= 30))


for fsm in (40.0, 60.0, 100.0):
    rows = []
    for k in sorted(P50):
        for (seg, kind, ln, far, mip, kink, sh, term) in P50[k]:
            if kind != 0:
                continue
            a_ = admit(ln, far, mip, kink, sh, mlt=0.15); b_ = admit(ln, far, mip, kink, sh, fsm=fsm); ab = admit(ln, far, mip, kink, sh, 0.15, fsm)
            if ab:
                v = A[k]["verdict"] if k in A else None
                rows.append("  %-14s seg %-7d len %5.2f far %6.2f mip %.2f kink %5.1f sh %d | a %d b %d | %-24s mf %s conn %s retreat %s split %s ke %.2f"
                            % (k, seg, ln, far, mip, kink, sh, a_, b_, rv(k), v and v["michel_found"], v and v["michel_conn_type"],
                               v and v["n_retreat"], v and v["n_split"], v["michel_ke_best"] if v else -1))
    print("(a)+(b) with the cap at %g cm: %d arms (far_len > 25 is a lower bound here -- the arm's diagnostic far_full settles it)" % (fsm, len(rows)))
    for r in rows:
        print(r)

# ------------------------------------------------------------------ 3
print("\n=== 3. doc 70's four named P2 items on production ===")
for k in ("039349_11/19", "039349_69/56", "039253_3/61", "039349_32/63"):
    v = A[k]["verdict"] if k in A else None
    L = P50.get(k, [])
    print("  %-14s %-24s michel_found %s n_stop_arms %s | %s" % (k, rv(k), v and v["michel_found"], v and v["n_stop_arms"],
                                                                "; ".join("seg %d kind %d len %.2f far %.2f mip %.2f kink %.1f sh %d term %d" % x for x in L) or "NO stop arm"))


# ------------------------------------------------------------------ 4
def pts(s):
    return np.c_[s["x"], s["y"], s["z"]].astype(float)


p = json.load(open(PREP + "/smprep-039349_11-c19.json"))
S = {s["id"]: s for s in p["pf"]["seg"]}
sp = np.array([p["verdict"]["stop_x"], p["verdict"]["stop_y"], p["verdict"]["stop_z"]], float)
a19 = pts(S[19004])
far_end = a19[-1] if np.linalg.norm(a19[0] - sp) < np.linalg.norm(a19[-1] - sp) else a19[0]
used = {19004} | set(p["pf"]["chain_segs"])
front, tot, n = [far_end], 0.0, 0
while front:
    q = front.pop()
    for i, s in S.items():
        if i in used:
            continue
        P_ = pts(s)
        for e0, e1 in ((P_[0], P_[-1]), (P_[-1], P_[0])):
            if np.linalg.norm(e0 - q) < 1.0:
                used.add(i); tot += s["len_cm"]; n += 1; front.append(e1)
                break
print("\n=== 4. 039349_11/19 arm 19004: far subtree by payload endpoint matching (1 cm) = %.1f cm over %d segments (the C++ capped walk printed 54.37) ===" % (tot, n))


# ------------------------------------------------------------------ 5
def cdir(P_, vp, w):
    d = np.linalg.norm(P_ - vp, axis=1); sel = d < w
    if not sel.any():
        return None
    v = P_[sel].mean(0) - vp; nn = np.linalg.norm(v)
    return v / nn if nn > 0 else None


diffs = []
for key, L in sorted(P50.items()):
    ev, cl = key.split("/")
    fs = glob.glob(PREP + "/smprep-%s-c%s.json" % (ev, cl))
    if not fs:
        continue
    p = json.load(open(fs[0]))
    S = {s["id"]: s for s in p["pf"]["seg"]}
    chain = [S[i] for i in p["pf"]["chain_segs"] if i in S]
    if not chain:
        continue
    VX = np.c_[p["pf"]["vtx"]["x"], p["pf"]["vtx"]["y"], p["pf"]["vtx"]["z"]].astype(float)
    CE = np.array([e for s in chain for e in pts(s)[[0, -1]]])
    for x in L:
        if x[0] not in S:
            continue
        Pa = pts(S[x[0]])
        ends = Pa[[0, -1]]
        ea = ends[int(np.argmin([np.linalg.norm(CE - e, axis=1).min() for e in ends]))]
        vp = VX[np.argmin(np.linalg.norm(VX - ea, axis=1))] if len(VX) else ea
        last = min(chain, key=lambda s: np.linalg.norm(pts(s)[[0, -1]] - vp, axis=1).min())
        a_, b_ = cdir(pts(last), vp, 15), cdir(Pa, vp, 15)
        if a_ is None or b_ is None:
            continue
        diffs.append(abs(180 - np.degrees(np.arccos(np.clip(a_ @ b_, -1, 1))) - x[5]))
d = np.array(diffs)
print("\n=== 5. the 15 cm kink from the payload's segment points vs the C++: %d arms, within 1 deg %d, median |diff| %.1f deg -> the payload points are not the fits; (c) is sized on an arm ==="
      % (len(d), int((d < 1).sum()), float(np.median(d)) if len(d) else float("nan")))

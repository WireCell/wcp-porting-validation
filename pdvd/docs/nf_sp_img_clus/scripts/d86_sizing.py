#!/usr/bin/env python3
"""doc pdvd/86 sec 8 -- size the two mechanisms the owner's smx5 answers name, over the
whole merged record, read-only, on the production-equivalent prep (p85vwh).

  U  "unfitted": imaged charge of the muon's OWN cluster near the fit end that no PR
     segment follows (039253_0/44, 039349_30/45).  n_off / q_off = own-cluster image
     points within R cm of the fit end lying > D cm from every fit point (the chain's
     muon rows plus every PR segment of the cluster).  D = 2 cm is past the body
     halo (d86_offfit.py's control: max 1.3-1.7 cm on the four items; re-measured
     here over the population as the halo p99).
  A  "fit-through": the Michel is the fit's own tail past the Bragg peak
     (039349_43/66, 039349_72/11).  The production verdict already locates that peak:
     bragg_anchor_shift_cm (doc 68) is the residual range of the anchored end, i.e. the
     length of fit past the peak.  Tail = the fit rows inside it; its median dQ/dx is
     read against the plateau.  bragg_anchor_fallback 1 rows (doc 75, the geometric
     reading stood) are reported separately.

Classes come from the record (verdict only: STM_MICHEL / STM_ONLY / THRU, FRAG_
folded).  Every count is by name-able item; the lists print the names.

Usage: STM_SCAN_RECORD=<merged record> d86_sizing.py --prep PREP [--R 10] [--D 2]
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--R", type=float, default=10.0)
ap.add_argument("--D", type=float, default=2.0)
ap.add_argument("--arm", default="p85vwh", help="arm whose wct_pr logs carry the pr54 drop lines")
a = ap.parse_args()
if "smx1a_smx3_smx4" not in os.environ.get("STM_SCAN_RECORD", ""):
    sys.exit("STM_SCAN_RECORD must name a merged record (smx1a+smx3+smx4[+smx5])")
rec = C.load_record()
P1_CLEAR = {"no_bragg", "shape_flat", "profile_sparse"}   # topology_stop_evidence + topology_clears_sparse

rows, halo = [], []
for k in sorted(rec):
    r = rec[k]
    if not C.judged(r):
        continue
    fn = C.payload_path(a.prep, k)
    if not os.path.exists(fn):
        continue
    p = json.load(open(fn))
    v, m = p["verdict"], p["muon"]
    M = np.c_[m["x"], m["y"], m["z"]]
    rr = np.asarray(m["rr"], float)
    q_mu = np.asarray(m["q"], float)
    i = int(np.argmin(rr))
    stop = M[i]
    im = p["image_near"]
    X = np.c_[im["x"], im["y"], im["z"]]
    cl = np.asarray(im["c"], int)
    q = np.asarray(im["q"], float)
    F = np.vstack([M] + [np.c_[s["x"], s["y"], s["z"]] for s in p["pf"]["seg"]])
    dfit = cKDTree(F).query(X)[0] if len(X) else np.zeros(0)
    dstop = np.linalg.norm(X - stop, axis=1) if len(X) else np.zeros(0)
    own = cl == int(p["cluster_id"])
    near = own & (dstop <= a.R)
    off = near & (dfit > a.D)
    # the halo control: own points beside a body stretch rr 30-40 cm upstream
    body = M[(rr > rr[i] + 30) & (rr < rr[i] + 40)]
    if len(body) and len(X):
        ctl = own & (cKDTree(body).query(X)[0] <= 5.0)
        if ctl.any():
            halo.append(float(dfit[ctl].max()))
    # the same count in three 10 cm spheres centred on the fit 25 / 35 / 45 cm upstream:
    # the item's own halo + delta rate with no Michel in it; excess = stop - max(body)
    bref = []
    for c0 in (25.0, 35.0, 45.0):
        j = np.where(rr - rr[i] >= c0)[0]
        if len(j) and len(X):
            ctr = M[j[np.argmin(rr[j])]]
            bref.append(int((own & (np.linalg.norm(X - ctr, axis=1) <= a.R) & (dfit > a.D)).sum()))
    # the fit's tail past the anchored peak
    sh = float(v.get("bragg_anchor_shift_cm") or 0.0)
    tail = (rr - rr[i]) < sh
    tq = float(np.median(q_mu[tail])) if tail.any() else float("nan")
    rows.append(dict(
        key=k, cls="THRU" if not C.is_stopper(r) else C.bare(r["verdict"]), src=r.get("source", "smx1a"),
        is_stm=int(v["is_stm"]), mf=int(v["michel_found"]), rej=C.reject_names(v),
        n_near=int(near.sum()), n_off=int(off.sum()), q_off=float(q[off].sum()),
        excess=(int(off.sum()) - max(bref)) if bref else None, stop=stop, cid=int(p['cluster_id']),
        sh=sh, fb=int(v.get("bragg_anchor_fallback") or 0), ntail=int(tail.sum()),
        tail_rel=tq / float(v["plateau_med"]) if v["plateau_med"] else float("nan")))

print("rows: %d judged items with a payload on %s (R %.0f cm, D %.1f cm)" % (len(rows), a.prep, a.R, a.D))
print("halo control (own points within 5 cm of a body stretch rr 30-40): %d items, max off-fit p50 %.2f p90 %.2f p99 %.2f max %.2f cm"
      % (len(halo), *np.percentile(halo, [50, 90, 99]), max(halo)))
print("own points within R of the fit end: items with none = %d (an id mismatch would show here)"
      % sum(1 for x in rows if x["n_near"] == 0))


def cell(x):
    if x["cls"] == "STM_MICHEL":
        return "STM_MICHEL mf%d" % x["mf"]
    return x["cls"]


CELLS = ["STM_MICHEL mf0", "STM_MICHEL mf1", "STM_ONLY", "THRU"]

# ---- U ------------------------------------------------------------------------
print("\n=== U. unfitted own-cluster charge near the fit end (n_off = points > D off every fit) ===")
print("population by cell and is_stm:")
for c in CELLS:
    for s in (1, 0):
        sel = [x for x in rows if cell(x) == c and x["is_stm"] == s]
        if not sel:
            continue
        n = np.array([x["n_off"] for x in sel])
        print("  %-15s is_stm %d: %3d items | n_off = 0: %3d | >=5: %3d | >=10: %3d | >=20: %3d | >=40: %3d"
              % (c, s, len(sel), (n == 0).sum(), (n >= 5).sum(), (n >= 10).sum(), (n >= 20).sum(), (n >= 40).sum()))
for thr in (10, 20):
    print("\nthreshold n_off >= %d:" % thr)
    for c in CELLS:
        sel = [x for x in rows if cell(x) == c and x["n_off"] >= thr]
        print("  %-15s %3d: %s" % (c, len(sel), ", ".join(
            "%s(%s%d, %.0fk)" % (x["key"], "" if x["is_stm"] else "is_stm0 ", x["n_off"], x["q_off"] / 1e3)
            for x in sorted(sel, key=lambda t: -t["n_off"])) if c != "STM_MICHEL mf1" else "(Michel already found)"))
print("\nthe michel_found-0 stoppers the record calls STM_MICHEL, every one:")
for x in sorted([x for x in rows if cell(x) == "STM_MICHEL mf0"], key=lambda t: (-t["is_stm"], -t["n_off"])):
    print("  %-14s is_stm %d  n_off %3d q_off %5.0fk | anchor shift %.2f fb %d tail %d rows %.2f plateau | %s"
          % (x["key"], x["is_stm"], x["n_off"], x["q_off"] / 1e3, x["sh"], x["fb"], x["ntail"], x["tail_rel"], x["src"]))
print("\nP1 risk: is_stm-0 THRU items with n_off >= 10 whose reject bits P1 could clear entirely:")
risk = [x for x in rows if x["cls"] == "THRU" and x["is_stm"] == 0 and x["n_off"] >= 10
        and x["rej"] and set(x["rej"]) <= P1_CLEAR]
print("  %d: %s" % (len(risk), ", ".join("%s(%s)" % (x["key"], "+".join(x["rej"])) for x in risk)))

# ---- U2 -----------------------------------------------------------------------
print("\n=== U2. the same count minus the item's own body rate (max over three 10 cm body spheres) ===")
for thr in (5, 10, 20):
    line = []
    for c in CELLS:
        for s_ in (1, 0):
            sel = [x for x in rows if cell(x) == c and x["is_stm"] == s_ and x["excess"] is not None]
            if sel:
                line.append("%s/is_stm%d %d/%d" % (c, s_, sum(1 for x in sel if x["excess"] >= thr), len(sel)))
    print("  excess >= %2d: %s" % (thr, " | ".join(line)))
print("  is_stm-1 items at excess >= 10, michel_found 0: %s" % ", ".join(
    "%s %s(%d)" % (cell(x), x["key"], x["excess"]) for x in rows
    if x["is_stm"] == 1 and x["mf"] == 0 and x["excess"] is not None and x["excess"] >= 10))

# ---- R ------------------------------------------------------------------------
# doc pdvd/62 T3a's object: the PR residual NeutrinoOtherSegments fits and then drops as
# isolated ("pr54 isolated-residual drop"); stop_local_residual_cm keeps it when an
# endpoint lies within the radius of the tagger's stop, with NO size floor.
print("\n=== R. pr54 isolated-residual drops of the candidate's own cluster near its fit end (arm %s logs) ===" % a.arm)
pat = re.compile(r"pr54 isolated-residual drop: cluster (\d+) n_points=(\d+) length=([\d.]+) cm dir_mag=([\d.]+) cm "
                 r"v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
drops = defaultdict(list)
for d in sorted(glob.glob(C.WORK + "/*_" + a.arm)):
    ev = os.path.basename(d)[: -len("_" + a.arm)]
    for lg in sorted(glob.glob(d + "/wct_pr_*.log")):
        for line in open(lg, errors="replace"):
            mm = pat.search(line)
            if mm:
                g = mm.groups()
                drops[(ev, int(g[0]))].append((int(g[1]), float(g[2]),
                    np.array([float(g[4]), float(g[5]), float(g[6])]), np.array([float(g[7]), float(g[8]), float(g[9])])))
byk = {x["key"]: x for x in rows}
near = []
for x in rows:
    ev, c = x["key"].split("/")
    for n, ln, v1, v2 in drops.get((ev, int(c)), []):
        dd = min(np.linalg.norm(v1 - x["stop"]), np.linalg.norm(v2 - x["stop"]))
        near.append((x["key"], n, ln, dd))
for R_, nmin, lmin, lab in ((20, 0, 0, "T3a as measured by doc 62 (20 cm, no floor)"), (5, 0, 0, "5 cm, no floor"),
                            (5, 5, 5.0, "5 cm, n >= 5 and length >= 5 cm")):
    hit = sorted({k for k, n, ln, dd in near if dd <= R_ and n >= nmin and ln >= lmin})
    cnt = Counter("%s/is_stm%d" % (cell(byk[k]), byk[k]["is_stm"]) for k in hit)
    print("  %-40s %2d items: %s" % (lab, len(hit), dict(sorted(cnt.items()))))
    for k in hit:
        # the drop line is printed once per PR pass over the cluster (the tagger's and the chain's); dedupe
        dl = sorted({(n, round(ln, 2), round(float(dd), 1)) for kk, n, ln, dd in near
                     if kk == k and dd <= R_ and n >= nmin and ln >= lmin})
        if R_ <= 5:
            print("      %-14s %-16s %s" % (k, "%s is_stm%d" % (cell(byk[k]), byk[k]["is_stm"]), dl))

# ---- A ------------------------------------------------------------------------
print("\n=== A. the fit's tail past the anchored Bragg peak (is_stm 1, michel_found 0) ===")
for fb in (0, 1):
    print("bragg_anchor_fallback %d:" % fb)
    for c in CELLS:
        sel = [x for x in rows if cell(x) == c and x["is_stm"] == 1 and x["mf"] == 0 and x["fb"] == fb]
        if not sel:
            continue
        s = np.array([x["sh"] for x in sel])
        print("  %-15s %3d items | shift < 1: %3d | 1-2: %3d | 2-3: %3d | >= 3: %3d | median %.2f cm"
              % (c, len(sel), (s < 1).sum(), ((s >= 1) & (s < 2)).sum(), ((s >= 2) & (s < 3)).sum(), (s >= 3).sum(), np.median(s)))
print("rule sketch: fallback 0, shift >= S, tail median dQ/dx <= T x plateau (is_stm 1, michel_found 0):")
for S in (1.5, 2.0, 2.5):
    for T in (1.0, 1.5):
        hit = [x for x in rows if x["is_stm"] == 1 and x["mf"] == 0 and x["fb"] == 0 and x["sh"] >= S and x["tail_rel"] <= T]
        cnt = Counter(cell(x) for x in hit)
        print("  S %.1f T %.1f: %s | STM_MICHEL mf0 %s | STM_ONLY %s | THRU %s" % (
            S, T, dict(sorted(cnt.items())),
            ", ".join(x["key"] for x in hit if cell(x) == "STM_MICHEL mf0"),
            ", ".join(x["key"] for x in hit if x["cls"] == "STM_ONLY"),
            ", ".join(x["key"] for x in hit if x["cls"] == "THRU")))
print("the same shift on the is_stm-1 stoppers whose Michel IS found (the found Michel starts at the fit end or past it):")
sel = [x for x in rows if cell(x) == "STM_MICHEL mf1" and x["is_stm"] == 1 and x["fb"] == 0]
s = np.array([x["sh"] for x in sel])
print("  %d items | shift < 1: %d | 1-2: %d | 2-3: %d | >= 3: %d | median %.2f cm"
      % (len(sel), (s < 1).sum(), ((s >= 1) & (s < 2)).sum(), ((s >= 2) & (s < 3)).sum(), (s >= 3).sum(), np.median(s)))

#!/usr/bin/env python3
"""doc pdvd/74 (P3) -- size "the Michel inside the muon chain" on today's production arm.

Read-only over one arm's prep (default: doc 72's bare-production confirmation arm,
/home/xqian/tmp/p72/prep_p72vprod) and the grading record.  Every number in doc 74
sec 2 comes from this script's stdout.

The offline twins (clus/src/StmMichelFunctions.cxx, PDVD production constants):
  bragg_confirmed  stm_michel_bragg_contrast on the live profile at the GEOMETRIC origin
                   (CheckSTM_Michel's chain-walk lambda): contrast >= 0.6 x expected
  retreat          stm_michel_stop_retreat: max_drop 2, collapse 0.5 x plateau, a peak
                   >= 1.4 x plateau within 15 cm of the new end, max_drop_len 25 cm,
                   min_tail_pts 2 -- its tail read four ways: doc57 (as built), strict
                   (rows past the vertex), sublive (the live floor not applied to the
                   tail), strict+sublive
  split            stm_michel_stop_split: rows inside the last chain segment, rr in
                   [3, 25] cm, the 5 cm row kink >= 15 deg (also >= 0, kink-free),
                   min_tail_pts 3
The payload's muon profile IS the C++ profile: every row, dead ones included, in
ascending L from the entry.  A chain vertex is written twice (same L, same point), so
seg_idx is rebuilt by counting those duplicates -- checked against n_chain_segs (sec 0).

Sections
  0  validation: the segment rebuild; the geometric contrast against the payload's
     `contrast` where T7's anchor shift is 0, and what the outliers are
  1  who is Bragg-confirmed, by record group; doc 70's P3 as proposed (the retreat /
     split let loose on a Bragg-confirmed chain), every fire named
  2  the retreat's tail read four ways, on every chain the doc 57 guard admits; fires named
  3  doc 70's four named items and the 25 scan-pinned items: the tail composition
  4  the 18 items where the retreat or the split already fired in production -- not
     replayable offline (their chain is already mutated); listed for the arm to decide
  5  the KE the new stop's Michel would carry (039252_16/32's 32009), from production's
     own attached single-segment Michels

Usage:
  python3 d74_sizing.py [--prep DIR] [--json FILE]   (STM_SCAN_RECORD defaults to smx1a+smx3+smx4)
"""
import argparse, collections, glob, json, os, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# before census_lib is imported: it reads the record path at import time (doc 72's trap)
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/p72/prep_p72vprod")
ap.add_argument("--json", default=None, help="write the predicted fire lists (d74_score.py section G reads them)")
a = ap.parse_args()

MIP = 55000.0            # mip_dqdx, PDVD production
LIVE = 0.15 * MIP        # profile_min_dqdx_frac
CMIN = 0.6               # bragg_contrast_min
NAMED = ["039252_16/32", "039253_3/61", "039349_60/40", "039349_64/65"]
REF = json.load(open(os.path.join(a.prep, "dqdx_ref_pdvd.json")))
_g = REF["grid"]
GX = _g["start"] + _g["step"] * np.arange(_g["n"])
GY = np.asarray(REF["muon"], float)
R = C.load_record()
print("record %s (%d records); prep %s" % (os.path.basename(C.REC), len(R), a.prep))


def mu(r):
    return np.interp(r, GX, GY)


def contrast(rr, q, tot):
    """(contrast, expected) of stm_michel_bragg_contrast on the live rows, or None."""
    m = q >= LIVE
    r, qq = rr[m], q[m]
    plo, phi = (20.0, 40.0) if tot >= 40 else (10.0, 20.0)
    tm = (r >= 0.5) & (r <= 3.0)
    pm = (r >= plo) & (r <= phi)
    if tm.sum() < 3 or pm.sum() < 3 or np.median(qq[pm]) <= 0:
        return None
    return float(np.median(qq[tm]) / np.median(qq[pm])), float(np.median(mu(r[tm])) / np.median(mu(r[pm])))


def rm3(v):
    """StmMichelFunctions.cxx running_median3: window i-1..i+1, w[size/2] of the sorted window."""
    out = []
    for i in range(len(v)):
        w = sorted(v[max(0, i - 1):min(len(v) - 1, i + 1) + 1])
        out.append(w[len(w) // 2])
    return np.array(out)


def plateau(rr, q, tot):
    lo, hi = (20.0, 40.0) if tot >= 40 else (10.0, 20.0)
    pl = q[(q >= LIVE) & (rr >= lo) & (rr <= hi)]
    return float(np.median(pl)) if len(pl) >= 3 and np.median(pl) > 0 else 0.0


def peak_before(L, q, bL, win=15.0):
    k = (L <= bL) & (q >= LIVE)
    if k.sum() < 3:
        return None
    r = rm3(list(q[k]))
    w = (bL - L[k]) <= win
    return float(r[w].max()) if w.sum() >= 3 else None


def tail_rows(L, q, bL, mode):
    if mode == "doc57":
        sel = (L >= bL) & (q >= LIVE)
    elif mode == "strict":
        sel = (L > bL) & (q >= LIVE)
    elif mode == "sublive":
        sel = L >= bL
    else:  # strict+sublive
        sel = L > bL
    return q[sel]


def retreat(it, mode):
    """-> (n_drop, drop_len, tail/plateau, peak/plateau) of the last accepted drop, or None."""
    L, q, seg, nseg, tot, pla = it["L"], it["q"], it["seg"], it["nseg"], it["tot"], it["pla"]
    if nseg < 2 or pla <= 0 or not it["segok"]:
        return None
    res = None
    for nd in (1, 2):
        cut = nseg - nd
        if cut <= 0:
            break
        idx = np.where(seg >= cut)[0]
        if not len(idx):
            break
        bL = L[idx[0]]
        dl = it["tot"] - bL
        if dl > 25:
            break
        tail = tail_rows(L, q, bL, mode)
        if len(tail) < 2:
            break
        tm = float(np.median(tail))
        if not tm < 0.5 * pla:
            break
        pk = peak_before(L, q, bL)
        if pk is None or pk < 1.4 * pla:
            break
        res = (nd, round(dl, 2), round(tm / pla, 2), round(pk / pla, 2))
    return res


def rowkink(L, P, i, win=5.0):
    jl, hl = i, False
    while jl > 0:
        jl -= 1
        if L[i] - L[jl] >= win:
            hl = True
            break
    jh, hh = i, False
    while jh + 1 < len(L):
        jh += 1
        if L[jh] - L[i] >= win:
            hh = True
            break
    if not (hl and hh):
        return -1.0
    u, w = P[i] - P[jl], P[jh] - P[i]
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu <= 0 or nw <= 0:
        return -1.0
    return float(np.degrees(np.arccos(np.clip(u @ w / (nu * nw), -1, 1))))


def split(it, kmin):
    L, q, P, seg, rr, pla = it["L"], it["q"], it["P"], it["seg"], it["rr"], it["pla"]
    if pla <= 0 or not it["segok"] or it["nlast"] < 4:
        return None
    best = None
    for i in range(len(L)):
        if seg[i] != seg[-1] or rr[i] < 3 or rr[i] > 25:
            continue
        kk = rowkink(L, P, i)
        if kk < kmin:
            continue
        tail = q[(L >= L[i]) & (q >= LIVE)]
        if len(tail) < 3:
            continue
        tm = float(np.median(tail))
        if not tm < 0.5 * pla:
            continue
        pk = peak_before(L, q, L[i])
        if pk is None or pk < 1.4 * pla:
            continue
        if best is None or kk > best[1]:
            best = (round(float(rr[i]), 2), round(kk, 1), round(tm / pla, 2))
    return best


def group(k, v):
    r = R.get(k)
    if not (r and C.judged(r)):
        return "not judged"
    return ("stopper" if C.is_stopper(r) else "THRU") + (" is_stm 1" if v["is_stm"] else " is_stm 0")


IT, bad_contrast = {}, []
for f in sorted(glob.glob(os.path.join(a.prep, "smprep-*.json"))):
    k = os.path.basename(f)[7:-5].replace("-c", "/")
    d = json.load(open(f))
    v, m = d["verdict"], d["muon"]
    L = np.asarray(m["L"], float); q = np.asarray(m["q"], float); P = np.c_[m["x"], m["y"], m["z"]].astype(float)
    if len(L) < 5:
        continue
    tot = float(L[-1]); rr = tot - L
    dup = np.r_[False, (np.diff(L) == 0) & (np.linalg.norm(np.diff(P, axis=0), axis=1) < 1e-6)]
    seg = np.cumsum(dup)
    nseg = int(v["n_chain_segs"])
    c = contrast(rr, q, tot)
    sh = float(v.get("bragg_anchor_shift_cm") or 0)
    if c and abs(sh) < 1e-9 and v.get("contrast"):
        dc = c[0] - float(v["contrast"])
        c2 = contrast(rr + 0.2, q, tot)   # T7: end_L = L[peak] + 0.2 cm, the shift clamped at 0
        bad_contrast.append((k, dc, (c2[0] - float(v["contrast"])) if c2 else None, c, v))
    IT[k] = dict(v=v, L=L, q=q, P=P, rr=rr, tot=tot, seg=seg, nseg=nseg, segok=int(dup.sum()) == max(0, nseg - 1),
                 nlast=int((seg == seg[-1]).sum()), bc=bool(c and c[1] > 0 and c[0] >= CMIN * c[1]), c=c, sh=sh,
                 pla=plateau(rr, q, tot), grp=group(k, v), moved=bool(v["n_retreat"] or v["n_split"]))

# ------------------------------------------------------------------ 0
print("\n=== 0. validation ===")
n = len(IT)
print("candidates with a profile: %d; seg_idx rebuilt from duplicate rows matches n_chain_segs on %d (the rest have a chain segment with no rows)"
      % (n, sum(it["segok"] for it in IT.values())))
dd = np.array([abs(x[1]) for x in bad_contrast])
out = [x for x in bad_contrast if abs(x[1]) > 1e-3]
fix = [x for x in out if x[2] is not None and abs(x[2]) <= 1e-3]
rest = [x for x in out if not (x[2] is not None and abs(x[2]) <= 1e-3)]
print("geometric contrast vs the payload at anchor shift 0: n=%d, median |d| %.1e; %d above 1e-3, of which %d are T7's +0.2 cm origin (shift clamped at 0)"
      % (len(dd), np.median(dd), len(out), len(fix)))
near = [x for x in rest if x[3][1] > 0 and abs(float(x[4]["contrast"]) / x[3][1] - CMIN) < 0.1]
print("  the other %d: max |d| %.3f; within 0.1 of the 0.6 x expected cut: %s"
      % (len(rest), max([abs(x[1]) for x in rest] or [0]), [x[0] for x in near] or "none"))

# ------------------------------------------------------------------ 1
print("\n=== 1. Bragg-confirmed at the geometric origin, by record group; doc 70's P3 as proposed ===")
G = collections.defaultdict(list)
for k, it in IT.items():
    G[it["grp"]].append(k)
print("%-18s %5s %6s %20s %22s" % ("group", "n", "bragg", "anchor shift > 0", "not bragg but is_stm 1"))
for g_ in sorted(G):
    ks = G[g_]
    print("%-18s %5d %6d %20d %22d" % (g_, len(ks), sum(IT[k]["bc"] for k in ks), sum(IT[k]["sh"] > 0 for k in ks),
                                       sum((not IT[k]["bc"]) and IT[k]["v"]["is_stm"] for k in ks)))
cs_fires = []
for k, it in sorted(IT.items()):
    if not it["bc"] or it["moved"]:
        continue
    r = retreat(it, "doc57")
    s15 = None if r else split(it, 15)
    s0 = split(it, 0)
    if r or s15 or s0:
        print("  %-14s %-18s retreat %s | split(15 deg) %s | split(kink-free) %s" % (k, it["grp"], r, s15, s0))
    if r or s15:
        cs_fires.append(k)
print("doc 70's P3 (retreat, else split at 15 deg, on a Bragg-confirmed chain): %d fires %s" % (len(cs_fires), cs_fires))

# ------------------------------------------------------------------ 2
print("\n=== 2. the retreat's tail read four ways, on chains the doc 57 guard admits (not Bragg-confirmed, not already moved) ===")
MODES = ("doc57", "strict", "sublive", "strict+sublive")
F = {m_: [] for m_ in MODES}
for k, it in sorted(IT.items()):
    if it["bc"] or it["moved"]:
        continue
    for m_ in MODES:
        r = retreat(it, m_)
        if r:
            F[m_].append((k, it["grp"], r))
print("%-16s %s" % ("reading", "  ".join("%-18s" % g_ for g_ in sorted(G))))
for m_ in MODES:
    c_ = collections.Counter(x[1] for x in F[m_])
    print("%-16s %s" % (m_, "  ".join("%-18d" % c_[g_] for g_ in sorted(G))))
legacy = {x[0] for x in F["doc57"]}
for m_ in MODES[1:]:
    new = [x for x in F[m_] if x[0] not in legacy]
    print("\n%s: fires the doc 57 reading does not (%d), THRU first:" % (m_, len(new)))
    for k, g_, r in sorted(new, key=lambda x: (not x[1].startswith("THRU"), x[0])):
        v = IT[k]["v"]
        print("   %-14s %-18s n_drop %d drop %.2f cm tail %.2f peak %.2f x plateau | michel_found %d conn %d | %s"
              % (k, g_, r[0], r[1], r[2], r[3], v["michel_found"], v["michel_conn_type"], ",".join(C.reject_names(v)) or "-"))
PRED = {"cs": cs_fires,
        "ts": sorted(x[0] for x in F["strict"] if x[0] not in legacy),
        "tl": sorted(x[0] for x in F["strict+sublive"] if x[0] not in legacy)}

# ------------------------------------------------------------------ 3
print("\n=== 3. doc 70's named items and the scan-pinned items ===")


def compo(it):
    """the n_drop = 1 tail, read the four ways, as median / plateau (and the row count)."""
    idx = np.where(it["seg"] >= it["nseg"] - 1)[0]
    if it["nseg"] < 2 or it["pla"] <= 0 or not it["segok"] or not len(idx):
        return "- (%s)" % ("one segment" if it["nseg"] < 2 else "segment rows not rebuilt" if not it["segok"] else "no plateau")
    bL = it["L"][idx[0]]
    parts = []
    for m_ in MODES:
        t = tail_rows(it["L"], it["q"], bL, m_)
        parts.append("%s %s(%d)" % (m_, ("%.2f" % (np.median(t) / it["pla"])) if len(t) else "-", len(t)))
    return "drop %.2f cm: " % (it["tot"] - bL) + ", ".join(parts)


for k in NAMED + sorted(x for x in IT if R.get(x, {}).get("pin_rr") is not None and x not in NAMED):
    if k not in IT:
        print("  %s: no payload" % k)
        continue
    it = IT[k]; v = it["v"]
    print("  %-14s %-18s pin %-5s bragg %d shift %.2f nseg %d n_retreat %d n_split %d stub %d is_stm %d mf %d | %s"
          % (k, it["grp"], R.get(k, {}).get("pin_rr"), it["bc"], it["sh"], it["nseg"], v["n_retreat"], v["n_split"],
             v["n_stub_absorb"], v["is_stm"], v["michel_found"], compo(it)))

# ------------------------------------------------------------------ 4
print("\n=== 4. already moved in production (retreat or split fired) -- the arm decides these ===")
MOVED = sorted(k for k, it in IT.items() if it["moved"])
for k in MOVED:
    v = IT[k]["v"]
    print("  %-14s %-18s n_retreat %d (%.2f cm) n_split %d (%.2f cm) nseg %d is_stm %d mf %d conn %d"
          % (k, IT[k]["grp"], v["n_retreat"], v["retreat_len"], v["n_split"], v["split_len"], v["n_chain_segs"],
             v["is_stm"], v["michel_found"], v["michel_conn_type"]))
print("  %d items" % len(MOVED))
PRED["moved"] = MOVED

# ------------------------------------------------------------------ 5
print("\n=== 5. the KE of an attached single-segment Michel, by length and charge (production) ===")
X = np.array([(it["v"]["michel_len"], it["v"]["michel_mip"], it["v"]["michel_ke_dqdx"]) for it in IT.values()
              if it["v"]["michel_conn_type"] == 1 and it["v"]["n_michel_segs"] == 1 and it["v"]["michel_len"] > 0
              and it["v"]["michel_ke_dqdx"] > 0 and it["v"].get("n_dots", 0) == 0], float)
cf, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X[:, 1]], X[:, 2] / X[:, 0], rcond=None)
nb = X[(np.abs(X[:, 1] - 0.90) < 0.1) & (np.abs(X[:, 0] - 6.8) < 3)]
print("n %d; KE/len = %.3f + %.3f x mip -> 6.79 cm at 0.90 MIP: %.1f MeV; neighbours (len 3.8-9.8, mip 0.8-1.0): %s MeV"
      % (len(X), cf[0], cf[1], 6.79 * (cf[0] + cf[1] * 0.90), sorted(np.round(nb[:, 2], 1).tolist())))

if a.json:
    json.dump(PRED, open(a.json, "w"), indent=1)
    print("\nwrote", a.json, {k_: len(v_) for k_, v_ in PRED.items()})

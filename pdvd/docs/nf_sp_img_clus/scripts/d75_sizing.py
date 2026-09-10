#!/usr/bin/env python3
"""doc pdvd/75 (P1b) -- offline twin of CheckSTM_Michel's verdict-stage shape
tests, run on a bare-production prep, used to (1) validate itself against the
payload's own verdict, (2) size doc 65 sec 5.1's "rise precondition" on the peak
anchor and every one-sided variant of it, and (3) size the two-sided rule that
was built (bragg_anchor_geo_fallback), writing the predicted fire list.

The twin (CheckSTM_Michel.cxx, PDVD production values):
  live       rows with dQ/dx >= profile_min_dqdx_frac (0.15) x mip_dqdx (55000)
  anchor     bragg_peak_anchor: over live rows within bragg_peak_search_cm (3.0)
             of the end, the row maximising the 5-point running mean of dQ/dx
             (edges clipped); end_L = L[peak] + 0.2 cm; rows past end_L dropped;
             needs >= 5 live rows, end_L < total_length, >= 3 rows kept
  contrast   stm_michel_bragg_contrast: tail rr in [0.5, 3.0], plateau rr in
             [20, 40] ([10, 20] when the profile is shorter than 40 cm); >= 3
             rows each; expected from the muon dQ/dx-vs-rr table
  no_bragg   contrast < bragg_contrast_min (0.6) x expected
  plateau    plateau_med / mip outside [plateau_mip_lo 0.6, plateau_mip_hi 1.6]
  shape_flat kslike_compare on rows with rr <= compare_range_cm (45) against the
             muon table and against a flat mip: ks_mu + ks_margin (-0.02) >= ks_flat
  sparse     contrast invalid, or fewer than 3 KS rows

Known limit: the payload's L is rounded to 0.01 cm, so a row at exactly rr = 3.0 /
20 / 40 can fall on the other side of a window edge than in the C++.  Section 1
counts the items where this changes the published verdict (4 of 568 on p74vprod2);
they are named, and the arm decides them.

The other reject bits (boundary, continuation, PID, ...) are taken from the
payload; topology_cleared_bits (P1) is re-applied as the C++ does.

Usage: d75_sizing.py [--prep DIR] [--json OUT]
"""
import argparse, collections, json, os, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/p74/prep_p74vprod2")
ap.add_argument("--json", default=None, help="write the predicted fire / mover lists (d75_score.py section G reads them)")
ap.add_argument("--rise", type=float, default=1.5, help="bragg_anchor_rise_min")
a = ap.parse_args()

MIP = 55000.0; LIVE = 0.15 * MIP; CMIN = 0.6; KSM = -0.02; CMP = 45.0; SEARCH = 3.0; PLO, PHI = 0.6, 1.6
NB, SF, PS, PM = 1 << 2, 1 << 3, 1 << 9, 1 << 10
SHAPE = NB | SF | PS | PM
TARGETS = ["039253_0/44", "039253_6/85", "039349_15/23", "039349_76/75"]   # doc 70 sec 3.4
P1SIDE = ["039253_13/73", "039253_3/66", "039349_36/46"]                    # the anchor's other 3 lost stoppers, inside P1's 40

rec = C.load_record()
P, _ = C.load_payloads(a.prep, rec)
REF = json.load(open(os.path.join(a.prep, "dqdx_ref_pdvd.json")))
_g = REF["grid"]
GX = _g["start"] + _g["step"] * np.arange(_g["n"]); GY = np.asarray(REF["muon"], float)
mu = lambda r: np.interp(r, GX, GY)  # noqa: E731
print("record %s (%d records); prep %s" % (os.path.basename(C.REC), len(rec), a.prep))


def kslike(t, r):
    t = np.asarray(t); r = np.asarray(r)
    return float(np.abs(np.cumsum(t) / t.sum() - np.cumsum(r) / r.sum()).max())


def anchor(L, q, tot):
    """T7 on the live rows -> dict(end_L, shift, peak5, ndrop, drop_med, last_q) or None (not fired)."""
    m = q >= LIVE; L = L[m]; q = q[m]; rr = tot - L; n = len(L)
    if n < 5:
        return None
    best = -1; ib = n - 1
    for i in range(n):
        if rr[i] > SEARCH + 1e-9:
            continue
        lo = max(0, i - 2); hi = min(n - 1, i + 2); s = q[lo:hi + 1].mean()
        if s > best:
            best = s; ib = i
    end_L = L[ib] + 0.2
    if not end_L < tot:
        return None
    keep = (end_L - L) >= 0
    if keep.sum() < 3:
        return None
    drop = q[~keep]
    return dict(end_L=end_L, shift=tot - end_L, peak5=best, ndrop=int(len(drop)),
                drop_med=(float(np.median(drop)) if len(drop) else 0.0), last_q=float(q[-1]))


def shape(L, q, tot, end_L=None):
    """the four shape bits (+ plateau median, ks_flat - ks_mu) at the geometric (end_L None) or anchored origin"""
    m = q >= LIVE; L = L[m]; q = q[m]
    if end_L is not None:
        k = (end_L - L) >= 0; L = L[k]; q = q[k]; rr = end_L - L; T = end_L
    else:
        rr = tot - L; T = tot
    b = 0; plo, phi = (20.0, 40.0) if T >= 40 else (10.0, 20.0)
    tm = (rr >= 0.5) & (rr <= 3.0); pm = (rr >= plo) & (rr <= phi); pl = None; c = e = None
    if tm.sum() < 3 or pm.sum() < 3 or np.median(q[pm]) <= 0:
        b |= PS
    else:
        pl = float(np.median(q[pm])); c = float(np.median(q[tm]) / pl); e = float(np.median(mu(rr[tm])) / np.median(mu(rr[pm])))
        if c < CMIN * e:
            b |= NB
        if pl / MIP < PLO or pl / MIP > PHI:
            b |= PM
    cm = rr <= CMP; kd = None
    if cm.sum() >= 3:
        km = kslike(q[cm], mu(rr[cm])); kf = kslike(q[cm], np.full(cm.sum(), MIP)); kd = kf - km
        if km + KSM >= kf:
            b |= SF
    else:
        b |= PS
    return b, pl, kd, c, e


rows = []
for k, p in sorted(P.items()):
    v = p["verdict"]; m = p["muon"]
    if not m or not m.get("L"):
        continue
    L = np.asarray(m["L"], float); q = np.asarray(m["q"], float); tot = float(L[-1])
    an = anchor(L, q, tot)
    r = rec.get(k); judged = r is not None and C.judged(r); truth = C.is_stopper(r) if judged else None
    b = int(v["reject_bits"] or 0); tcb = int(v.get("topology_cleared_bits") or 0); other = b & ~SHAPE
    bA, plA, kdA, cA, eA = shape(L, q, tot, an["end_L"] if an else None)
    bG, plG, kdG, cG, eG = shape(L, q, tot, None)
    rows.append(dict(k=k, pub=int(v["is_stm"]), pubshift=float(v["bragg_anchor_shift_cm"]), judged=judged, truth=truth,
                     conf=(r["confidence"] if r else ""), an=an, other=other, tcb=tcb,
                     anch=int(other == 0 and (bA & ~tcb) == 0), geo=int(other == 0 and (bG & ~tcb) == 0),
                     bA=bA, bG=bG, plA=plA, plG=plG, kdA=kdA, kdG=kdG, cA=cA, eA=eA, cG=cG,
                     michel=int(v["michel_found"])))


def prom(r):   # the anchored peak against the anchored plateau -- the C++ condition
    return (r["an"]["peak5"] / r["plA"]) if (r["an"] and r["plA"]) else None


def fall(r):
    return (r["an"]["drop_med"] / r["an"]["peak5"]) if r["an"] else None


def lastf(r):
    return (r["an"]["last_q"] / r["an"]["peak5"]) if r["an"] else None


def tr(r):
    return "STOP" if r["truth"] else ("THRU" if r["truth"] is not None else "n/j")


J = [r for r in rows if r["judged"]]


def score(sel):
    tp = sum(1 for r in J if sel(r) and r["truth"]); fp = sum(1 for r in J if sel(r) and not r["truth"])
    fn = sum(1 for r in J if not sel(r) and r["truth"]); return tp, fp, fn


def movers(sel):
    g = [r["k"] for r in J if sel(r) and not r["anch"] and r["truth"]]; nf = [r["k"] for r in J if sel(r) and not r["anch"] and not r["truth"]]
    lo = [r["k"] for r in J if not sel(r) and r["anch"] and r["truth"]]; rf = [r["k"] for r in J if not sel(r) and r["anch"] and not r["truth"]]
    return "+TP %s | newFP %s | lostTP %s | rmFP %s" % (g, nf, lo, rf)


# ---- 1. validation -------------------------------------------------------------
print("\n=== 1. the twin against the payload ===")
same = [r for r in rows if r["anch"] == r["pub"]]
print("candidates with a profile %d; anchored twin is_stm == published on %d" % (len(rows), len(same)))
for r in rows:
    if r["anch"] != r["pub"]:
        print("   twin != published  %-14s %-4s pub %d twin %d  shift twin %.2f payload %.2f  anchored bits %d other %d" %
              (r["k"], tr(r), r["pub"], r["anch"], r["an"]["shift"] if r["an"] else 0, r["pubshift"], r["bA"], r["other"]))
nfire = sum(1 for r in rows if r["an"]); npay = sum(1 for r in rows if r["pubshift"] > 0)
print("anchor fired: twin %d, payload %d; shift within 0.01 cm on %d" %
      (nfire, npay, sum(1 for r in rows if abs((r["an"]["shift"] if r["an"] else 0) - r["pubshift"]) <= 0.011)))
print("production (anchored twin) is_stm TP/FP/FN %s; geometric origin everywhere %s" % (score(lambda r: r["anch"]), score(lambda r: r["geo"])))

# ---- 2. the population a precondition can touch ---------------------------------
print("\n=== 2. anchor-rejected / geometric-accepted judged items (anch 0, geo 1) ===")
print("%-14s truth conf   shift ndrop peak5/plA drop_med/peak5 last/peak5   kdA    kdG  michel" % "item")
for r in sorted([r for r in J if r["anch"] == 0 and r["geo"] == 1], key=lambda r: -(prom(r) or 0)):
    an = r["an"]
    print("%-14s %-4s %-6s %.2f   %d    %5.2f      %5.2f       %5.2f    %+.3f %+.3f  %d" %
          (r["k"], tr(r), r["conf"], an["shift"], an["ndrop"], prom(r) or -1, fall(r), lastf(r), r["kdA"], r["kdG"], r["michel"]))
print("\n=== 2b. anchor-accepted / geometric-rejected (anch 1, geo 0): what a rule must not touch ===")
AG = [r for r in J if r["anch"] == 1 and r["geo"] == 0]
for nm, f in (("peak5/plA", prom), ("drop_med/peak5", fall), ("last/peak5", lastf)):
    vs = sorted([f(r) for r in AG if f(r) is not None])
    print("%-15s n=%d min %.2f p10 %.2f p50 %.2f p90 %.2f max %.2f" % (nm, len(vs), vs[0], vs[len(vs) // 10], vs[len(vs) // 2], vs[9 * len(vs) // 10], vs[-1]))
    print("   THRU items:", [(r["k"], round(f(r), 2) if f(r) is not None else None) for r in AG if not r["truth"]])
print("\n=== 2c. the doc 70 sec 3.4 targets and the P1-side items ===")
for k in TARGETS + P1SIDE:
    r = [x for x in rows if x["k"] == k]
    if not r:
        print(k, "no payload"); continue
    r = r[0]
    print("%-14s pub %d anch %d geo %d shift %.2f peak5/plA %s cA/eA %s/%s cG %s kdA %s kdG %s bitsA %d bitsG %d other %d" %
          (k, r["pub"], r["anch"], r["geo"], r["an"]["shift"] if r["an"] else 0, "%.2f" % prom(r) if prom(r) else "-",
           "%.2f" % r["cA"] if r["cA"] else "-", "%.2f" % r["eA"] if r["eA"] else "-", "%.2f" % r["cG"] if r["cG"] else "-",
           "%+.3f" % r["kdA"] if r["kdA"] is not None else "-", "%+.3f" % r["kdG"] if r["kdG"] is not None else "-", r["bA"], r["bG"], r["other"]))

# ---- 3. one-sided preconditions (doc 65 sec 5.1 as written, and its variants) ----
print("\n=== 3a. doc 65's precondition as written: anchor only if peak5/plateau >= T, else geometric ===")
for T in (1.0, 1.2, 1.5, 1.75, 2.0):
    sel = lambda r, T=T: r["anch"] if (prom(r) is not None and prom(r) >= T) else r["geo"]  # noqa: E731
    print("T=%.2f TP/FP/FN %s  %s" % (T, score(sel), movers(sel)))
print("\n=== 3b. anchor only if what it dropped is a real fall: drop_med/peak5 <= F ===")
for F in (0.5, 0.7, 0.8, 0.9, 0.95):
    sel = lambda r, F=F: r["anch"] if (r["an"] is None or fall(r) <= F) else r["geo"]  # noqa: E731
    print("F=%.2f TP/FP/FN %s  %s" % (F, score(sel), movers(sel)))
print("\n=== 3c. anchor only if the last live row is below the peak: last/peak5 <= X ===")
for X in (0.5, 0.7, 0.9):
    sel = lambda r, X=X: r["anch"] if (r["an"] is None or lastf(r) <= X) else r["geo"]  # noqa: E731
    print("X=%.2f TP/FP/FN %s  %s" % (X, score(sel), movers(sel)))

# ---- 4. the two-sided rule that was built -----------------------------------------
print("\n=== 4. bragg_anchor_geo_fallback: is_stm = anchored, or geometric when the anchor rejected on a shape bit and peak5 >= T x anchored plateau ===")


def fb(r, T):
    if r["anch"]:
        return 1, False
    if r["an"] is None or r["plA"] is None or prom(r) is None or prom(r) < T:
        return 0, False
    if not (r["bA"] & SHAPE):
        return 0, False
    fired = r["other"] == 0 and (r["bG"] & SHAPE) == 0     # the C++ fires only when the geometric reading clears all four
    return (1 if fired else 0), fired


for T in (0.0, 1.0, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0):
    sel = lambda r, T=T: fb(r, T)[0]  # noqa: E731
    print("T=%.2f TP/FP/FN %s  %s" % (T, score(sel), movers(sel)))
pred = {}
for T, nm in ((a.rise, "fb"), (0.0, "fb0")):
    fires = [r["k"] for r in rows if fb(r, T)[1]]
    # the C++ fires on the RAW anchored bits, before P1's topology clearance and
    # regardless of the other bits: name every such item (some fire with is_stm
    # already 1 through P1, some with is_stm staying 0 through another bit).
    # (Corrected 2026-09-10 after the first DEBUG lines of p75vfb showed 039253_3/66
    # firing -- a P1-cleared item the first version left out; the is_stm_gain list
    # below is the pre-launch prediction and is unchanged.)
    fires_any = [r["k"] for r in rows if r["an"] and r["plA"] and prom(r) is not None and prom(r) >= T
                 and (r["bA"] & SHAPE) and (r["bG"] & SHAPE) == 0]
    pred[nm] = dict(rise=T, fires=sorted(fires_any), is_stm_gain=sorted(fires),
                    judged=[(k, tr(next(r for r in rows if r["k"] == k))) for k in sorted(fires_any)])
    print("\n%s (rise %.2f): predicted bragg_anchor_fallback fires %d: %s" % (nm, T, len(fires_any), pred[nm]["judged"]))
print("\ntwin-uncertain (published != twin): %s" % [r["k"] for r in rows if r["anch"] != r["pub"]])
if a.json:
    json.dump(pred, open(a.json, "w"), indent=1)
    print("wrote", a.json)

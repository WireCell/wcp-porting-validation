#!/usr/bin/env python3
"""doc pdvd/82 -- doc 78 action item 2 ("a peak-then-drop stop mover") sized offline
on one bare-production arm against the smx1a+smx3+smx4 record, read-only.

Sections:
  0  the twin's OWN validation: on every payload where NO production mover fired,
     how often does this offline reading say one would have?  (the number that
     tells you how much the control counts below are inflated)
  1  the literal doc 78 rule -- tail median <= f x the surviving PEAK -- swept
     against the 21 pin_rr items production did not move and the through-going
     control.  Shows why it is not a collapse test at all: with peak/plateau in
     1.4-3, f = 0.7 puts the threshold at 0.98-2.1 x plateau.
  2  WHY production's movers are silent on those items: CheckSTM_Michel skips
     the retreat AND the split whenever the chain is already Bragg-confirmed
     (:2270-2273, :2328-2332; michel_collinear_split is OFF in production).
     Counts the Bragg-confirmed fraction of every verdict class.
  3  the design this leads to: the peak-relative tail test gated on a
     Bragg-confirmed chain, swept in (f, kink) against the same two populations.
  4  the two side-sweeps doc 78 item 2 asked for on EXISTING knobs:
     split_kink_min_deg lowered, retreat_peak_frac lowered.

The chain profile is read from the prep payload's role-1 muon rows (q = dQ/dx in
e/cm).  The payload's profile is the FINAL chain, so an item whose stop production
already moved cannot be re-read here -- those items are excluded from every
population and counted separately.

Each candidate row is judged with the production thresholds
(CheckSTM_Michel.cxx defaults + pdvd/wct-pr-perevt.jsonnet):
  plateau      median dQ/dx over rr in [20, 40] cm (halved under 40 cm of track),
               live rows only (q >= profile_min_dqdx_frac * mip_dqdx = 0.15*55k; 50k through doc 82)
  peak         max of the 3-row running median of the SURVIVING rows within
               retreat_peak_window_cm = 15 cm of the candidate row
  collapse     tail median < retreat_collapse_frac = 0.5 x plateau   (production)
  peak gate    peak >= retreat_peak_frac = 1.4 x plateau             (production)
  bend         stm_michel_row_kink_deg's reading: 5 cm arms either side
Candidate rows are restricted to split_min_drop_cm = 3 .. michel_max_len_cm = 25
and, where stated, to the chain's LAST segment -- the only rows stop_split can
reach (stm_michel_stop_split: seg_idx[i] == n_chain_segs - 1).

tail_strict: production runs retreat_tail_strict:true, so the RETREAT's tail
excludes the boundary row; the SPLIT has no such flag and includes it.  --strict
picks the reading; the default (False) is the split's, since the split is the
mechanism that reaches a row with no vertex on it.

Repro (doc 82 sec 0):
  python3 d82_sizing.py [--prep /home/xqian/tmp/p79/prep_p79vprod] > /home/xqian/tmp/p82/sizing.txt
"""
import argparse, glob, json, math
import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/p79/prep_p79vprod")
ap.add_argument("--record", default=IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
ap.add_argument("--strict", action="store_true", help="read the tail the RETREAT's way (exclude the boundary row)")
args = ap.parse_args()

# production operating point
MINLIVE = 0.15 * 55000.0          # profile_min_dqdx_frac * mip_dqdx -- the COMPILED PDVD value (wct-pr-perevt.jsonnet:578); 50000 through doc 82, corrected in doc 83
COLLAPSE, PEAK_FRAC, PEAK_WIN = 0.5, 1.4, 15.0
MIN_DROP, MAX_DROP = 3.0, 25.0    # split_min_drop_cm, michel_max_len_cm
DIR_WIN = 5.0                     # split_dir_window_cm
CMIN = 0.6                        # bragg_contrast_min
PLAT_LO, PLAT_HI = 20.0, 40.0     # bragg_plateau_{lo,hi}_cm

R = {v["key"]: v for v in json.load(open(args.record))}
P = {}
for fn in glob.glob(args.prep + "/smprep-*.json"):
    d = json.load(open(fn))
    P["%s/%d" % (d["event"], d["cluster_id"])] = d
print("prep %s: record %d items, payloads %d, both %d" % (args.prep, len(R), len(P), len(set(R) & set(P))))
print("tail reading: %s" % ("RETREAT (strict, boundary row excluded)" if args.strict else "SPLIT (boundary row included)"))


def prof(p):
    m = p["muon"]; o = np.argsort(np.asarray(m["rr"], float))
    return {k: np.asarray(m[k], float)[o] for k in ("x", "y", "z", "q", "rr", "L")}


def plateau_of(m):
    tot = float(m["L"].max() - m["L"].min()) if m["L"].size else 0.0
    lo, hi = (PLAT_LO, PLAT_HI) if tot >= PLAT_HI else (PLAT_LO / 2, PLAT_HI / 2)
    sel = (m["rr"] >= lo) & (m["rr"] <= hi) & (m["q"] >= MINLIVE)
    return float(np.median(m["q"][sel])) if sel.sum() >= 3 else 0.0


def rm3(v):
    out = np.empty(v.size)
    for i in range(v.size):
        out[i] = np.median(v[max(0, i - 1):min(v.size, i + 2)])
    return out


def bend_at(m, i, w=DIR_WIN):
    """stm_michel_row_kink_deg's reading at row i: w cm of arc either side."""
    P3 = np.c_[m["x"], m["y"], m["z"]]; r = m["rr"][i]
    a = P3[(m["rr"] > r) & (m["rr"] <= r + w)]      # incoming (larger rr)
    b = P3[(m["rr"] < r) & (m["rr"] >= r - w)]      # outgoing (smaller rr)
    if len(a) < 2 or len(b) < 2: return -1.0
    da = a[0] - a[-1]; db = b[0] - b[-1]
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na == 0 or nb == 0: return -1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(da, db) / na / nb)))))


def last_seg_max_rr(p):
    """largest rr still inside the chain's LAST segment (the one holding rr = 0)."""
    ids = [int(k) for k, v in p["pf"]["chain_role"].items() if v == 1 and not str(k).startswith("C")]
    best = None
    for s in p["pf"]["seg"]:
        if int(s["id"]) not in ids: continue
        rr = np.asarray(s["rr"], float); rr = rr[rr >= 0]
        if rr.size == 0: continue
        if best is None or rr.min() < best[0]: best = (float(rr.min()), float(rr.max()))
    return best[1] if best else -1.0


def bragg_confirmed(p):
    """the chain walk's own gate (CheckSTM_Michel.cxx:2210), read from the published
    contrast.  NOTE: published contrast is at the ANCHORED origin; the chain walk
    reads it at the GEOMETRIC one, so this is an UPPER bound on the true count."""
    v = p["verdict"]; c = v.get("contrast"); e = v.get("contrast_expected")
    return bool(v.get("bragg_valid")) and c is not None and e and e > 0 and c >= CMIN * e


_C = {}
def rows(k, peak_frac=PEAK_FRAC, last_seg_only=True):
    """every candidate row with its five readings."""
    key = (k, peak_frac, last_seg_only)
    if key in _C: return _C[key]
    p = P[k]; m = prof(p); pl = plateau_of(m); lm = last_seg_max_rr(p); out = []
    if pl > 0 and (lm > 0 or not last_seg_only):
        for i in range(m["rr"].size):
            r = float(m["rr"][i])
            if r < MIN_DROP or r > MAX_DROP: continue
            if last_seg_only and r > lm: continue
            tsel = (m["rr"] < r) if args.strict else (m["rr"] <= r)
            tl = m["q"][tsel & (m["q"] >= MINLIVE)]
            if tl.size < 3: continue
            tmed = float(np.median(tl))
            kept = (m["rr"] >= r) & (m["q"] >= MINLIVE)
            kq = m["q"][kept]; krr = m["rr"][kept] - r
            if kq.size < 3: continue
            r3 = rm3(kq); win = krr <= PEAK_WIN
            if win.sum() < 3: continue
            pk = float(r3[win].max())
            if pk < peak_frac * pl: continue
            out.append(dict(rr=r, tp=tmed / pl, pp=pk / pl, tk=tmed / pk, bd=bend_at(m, i)))
    _C[key] = out; return out


def fires(k, f, kmin, peak_frac=PEAK_FRAC, last_seg_only=True):
    """the admitted rows: the production plateau rule OR the peak-relative rule."""
    return [z for z in rows(k, peak_frac, last_seg_only)
            if (z["tp"] < COLLAPSE) or (f > 0 and z["tk"] <= f and z["bd"] >= kmin)]


MOVED = [k for k in P if (P[k]["verdict"]["n_retreat"] or P[k]["verdict"]["n_split"])]
NOMOVE = [k for k in P if k not in MOVED]
prr = sorted((x for x in R if R[x].get("pin_rr") is not None and x in P), key=lambda x: -R[x]["pin_rr"])
SIG = [x for x in prr if x in NOMOVE]
CTL = [x for x in R if R[x]["verdict"] == "THRU" and x in NOMOVE]

# ============================ 0. the twin's validation ============================
print("\n=== 0. TWIN VALIDATION -- production's movers fired on %d of %d payloads (%d pin_rr, %d THRU)" % (
    len(MOVED), len(P), sum(1 for k in MOVED if R.get(k, {}).get("pin_rr") is not None),
    sum(1 for k in MOVED if R.get(k, {}).get("verdict") == "THRU")))
print("    A fired mover cannot be re-read here: the dropped tail is not in the published profile.")
print("    What IS checkable is the converse -- on the %d payloads where NO mover fired, how often does" % len(NOMOVE))
print("    this reading of the PRODUCTION rule (plateau collapse + peak + bend >= split_kink_min_deg 15) say one would have?")
for nm, sel in (("ALL", NOMOVE),
                ("Bragg-CONFIRMED (production skips BOTH movers here)", [k for k in NOMOVE if bragg_confirmed(P[k])]),
                ("not Bragg-confirmed (production DID try them)", [k for k in NOMOVE if not bragg_confirmed(P[k])])):
    n = sum(1 for k in sel if any(z["tp"] < COLLAPSE and z["bd"] >= 15.0 for z in rows(k)))
    print("      %-52s %3d items, twin says fire on %3d (%.1f%%)" % (nm, len(sel), n, 100.0 * n / max(1, len(sel))))
print("    -> the residual is the graph constraints the twin cannot see (fits() >= 4, break_segment success,")
print("       stop_split_max 1, an on-chain vertex for the retreat): production's split fired on 5 of that")
print("       last group where the twin says 10, so every control count below is inflated by about 2x.")

# ============================ 1. the literal doc 78 rule ============================
print("\n=== 1. doc 78 item 2 AS WRITTEN: tail median <= f x the surviving peak, bend >= k")
print("    signal = the %d pin_rr items production did not move; control = the %d THRU items it did not move" % (len(SIG), len(CTL)))
print("    (no Bragg gate, rows restricted to the chain's last segment -- what stop_split can reach)")
print("      f    k | signal  within 2cm of the pin | THRU control | control at plateau-only")
base_s = sum(1 for x in SIG if fires(x, 0, 99))
base_c = sum(1 for x in CTL if fires(x, 0, 99))
for f in (0.5, 0.6, 0.7):
    for kk in (15.0, 20.0, 25.0, 30.0):
        ns = nn = 0
        for x in SIG:
            c = fires(x, f, kk)
            if c:
                ns += 1
                if abs(max(c, key=lambda z: z["bd"])["rr"] - R[x]["pin_rr"]) <= 2.0: nn += 1
        nc = sum(1 for x in CTL if fires(x, f, kk))
        print("    %.2f %4.0f | %2d  %2d | %3d | %3d" % (f, kk, ns, nn, nc, base_c))
print("    production rule alone (f = 0): signal %d, control %d" % (base_s, base_c))
print("    WHY it is not a collapse test: with peak/plateau in 1.4-3, tail <= 0.7 x peak is tail <= 0.98-2.1 x")
print("    plateau -- a muon that simply keeps going passes it.")

# ============================ 2. what really blocks the movers ============================
print("\n=== 2. the Bragg gate: CheckSTM_Michel runs neither mover on a Bragg-confirmed chain")
cls = {}
for k, p in P.items():
    v = R.get(k, {}).get("verdict", "UNJUDGED")
    c = cls.setdefault(v, [0, 0]); c[0] += 1
    if bragg_confirmed(p): c[1] += 1
for v in sorted(cls, key=lambda x: -cls[x][0]):
    n, b = cls[v]
    print("    %-16s %3d payloads, %3d Bragg-confirmed (%3.0f%%)" % (v, n, b, 100.0 * b / n))
print("    the %d pin_rr items production did not move: %d Bragg-confirmed -- the movers were never tried on them" % (
    len(SIG), sum(1 for k in SIG if bragg_confirmed(P[k]))))
print("    the %d THRU control items:                  %d Bragg-confirmed" % (
    len(CTL), sum(1 for k in CTL if bragg_confirmed(P[k]))))

# ============================ 3. the design ============================
BC_SIG = [x for x in SIG if bragg_confirmed(P[x])]
BC_CTL = [x for x in CTL if bragg_confirmed(P[x])]
print("\n=== 3. the peak-relative tail test on a Bragg-confirmed chain (michel_collinear_split + stop_tail_peak_frac)")
print("    signal = %d, control = %d Bragg-confirmed THRU" % (len(BC_SIG), len(BC_CTL)))
print("      f    k | signal  within 2cm | control | the signal items by name")
for f in (0.5, 0.6, 0.7):
    for kk in (15.0, 20.0, 25.0, 30.0):
        ns = nn = 0; names = []
        for x in BC_SIG:
            c = fires(x, f, kk)
            if c:
                ns += 1; names.append(x)
                if abs(max(c, key=lambda z: z["bd"])["rr"] - R[x]["pin_rr"]) <= 2.0: nn += 1
        nc = sum(1 for x in BC_CTL if fires(x, f, kk))
        print("    %.2f %4.0f | %2d  %2d | %2d | %s" % (f, kk, ns, nn, nc, " ".join(names)))

F_OP, K_OP = 0.5, 25.0
print("\n    the operating point f = %.2f, kink = %.0f, item by item over the %d Bragg-confirmed signal items:" % (F_OP, K_OP, len(BC_SIG)))
for x in BC_SIG:
    vd = P[x]["verdict"]; c = fires(x, F_OP, K_OP)
    if not c:
        print("      %-14s pin_rr %5.1f  -- no qualifying row | is_stm %d michel_found %d" % (x, R[x]["pin_rr"], vd["is_stm"], vd["michel_found"]))
        continue
    b = max(c, key=lambda z: z["bd"])
    print("      %-14s pin_rr %5.1f -> rr %5.1f (%+5.1f) tail/plateau %.2f peak/plateau %.2f tail/peak %.2f bend %5.1f | is_stm %d michel_found %d" % (
        x, R[x]["pin_rr"], b["rr"], b["rr"] - R[x]["pin_rr"], b["tp"], b["pp"], b["tk"], b["bd"], vd["is_stm"], vd["michel_found"]))
print("    the Bragg-confirmed THRU items that fire (the FP risk list):")
for x in BC_CTL:
    c = fires(x, F_OP, K_OP)
    if c:
        b = max(c, key=lambda z: z["bd"])
        print("      %-14s rr %5.1f tail/plateau %.2f peak/plateau %.2f tail/peak %.2f bend %5.1f is_stm %d" % (
            x, b["rr"], b["tp"], b["pp"], b["tk"], b["bd"], P[x]["verdict"]["is_stm"]))
print("    what the Bragg gate buys: the same rule with NO gate fires on %d of the %d THRU control items" % (
    sum(1 for x in CTL if fires(x, F_OP, K_OP)), len(CTL)))

# ============================ 4. the side-sweeps on existing knobs ============================
NB_SIG = [x for x in SIG if not bragg_confirmed(P[x])]
NB_CTL = [x for x in CTL if not bragg_confirmed(P[x])]
print("\n=== 4. doc 78 item 2's two side-sweeps, on EXISTING knobs (no code)")
print("    only the not-Bragg-confirmed items can be reached (%d signal, %d control)" % (len(NB_SIG), len(NB_CTL)))
print("  A) split_kink_min_deg lowered, production collapse rule unchanged")
for kk in (15.0, 12.0, 10.0, 8.0, 5.0):
    ns = [x for x in NB_SIG if any(z["tp"] < COLLAPSE and z["bd"] >= kk for z in rows(x))]
    nc = sum(1 for x in NB_CTL if any(z["tp"] < COLLAPSE and z["bd"] >= kk for z in rows(x)))
    print("    kink >= %4.0f | signal %2d  %-40s | control %3d" % (kk, len(ns), " ".join(n.split("_")[1] for n in ns), nc))
print("  B) retreat_peak_frac lowered (doc 78 asked for 1.25 to reach 039253_3/61)")
for pf in (1.4, 1.3, 1.25, 1.2):
    ns = [x for x in NB_SIG if any(z["tp"] < COLLAPSE and z["bd"] >= 15.0 for z in rows(x, pf))]
    nc = sum(1 for x in NB_CTL if any(z["tp"] < COLLAPSE and z["bd"] >= 15.0 for z in rows(x, pf)))
    print("    peak_frac %.2f | signal %2d  %-40s | control %3d" % (pf, len(ns), " ".join(n.split("_")[1] for n in ns), nc))
print("    039253_3/61 is Bragg-confirmed: %s -- the retreat is never tried on it, so retreat_peak_frac is not what blocks it" % (
    bragg_confirmed(P["039253_3/61"]) if "039253_3/61" in P else "no payload"))
print("\n  the 8 is_stm-0 pin_rr items and whether the Bragg-gated rule can reach them:")
for x in SIG:
    vd = P[x]["verdict"]
    if vd["is_stm"] == 0:
        print("    %-14s pin_rr %5.1f Bragg-confirmed %-5s reject_bits %4d michel_found %d" % (
            x, R[x]["pin_rr"], bragg_confirmed(P[x]), vd["reject_bits"], vd["michel_found"]))

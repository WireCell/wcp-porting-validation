#!/usr/bin/env python3
"""doc pdvd/92 -- the exact twin of the wide Bragg-peak read, written BEFORE the arms run, and the
decision set for the smx8 blind re-judge (doc 91 sec 10 item 1).

Doc 91 sized the rule offline and pre-registered W 8 cm / R 1.3 / D 0.8 as the operating point, with
R 1.5 as the conservative alternative.  This script predicts, item by item, what the two ON arms must
publish, so the gate is "exactly this and nothing else" rather than "the counts look right".

WHAT THE C++ DOES (CheckSTM_Michel.cxx, doc pdvd/92 block after doc 75's fallback):
  It runs where the reading that stands still carries a shape bit (no_bragg | shape_flat |
  plateau_off_mip | profile_sparse), re-anchors at the 5-point running-mean maximum of the LIVE
  GEOMETRIC rows within bragg_wide_anchor_cm of the fit end (T7's recipe: end_L = L[max] + 0.2 cm),
  reads the same four tests there, and clears all four only if that reading sets none of them AND
    R: the wide peak's 5-point mean >= bragg_wide_anchor_rise_min x the wide plateau median,
    D: >= 3 live rows lie past the wide peak and their median <= bragg_wide_anchor_tail_max x it.
  It never sets a bit.  It does NOT replace rec.bragg / ks_* / ratio_* -- unlike doc 75's fallback --
  because rec.bragg is read again at :2950 (bragg_here, the continuation-arm demotion under
  michel_guards_stop) and at :2869 (T8's charge support).  Replacing it would couple the wide read to
  R_CONTINUATION, which this twin does not model.  Leaving it alone makes the C++ exactly
  "published bits with the four shape bits cleared, then P1", which is what section 2 predicts, and
  keeps every shape field byte-identical on every item including the movers.

WHAT THIS MEANS FOR THE DIFF (the gate's expected set, section 3):
  * reject_bits / topology_cleared_bits move on EVERY firing item -- including items P1 was going to
    clear anyway, where is_stm does not move but topology_cleared_bits goes to 0 because the wide
    read cleared the bits before P1 could.  Those are NOT is_stm movers and are easy to misread as a
    gate failure.
  * bragg_wide_fired / bragg_wide_shift_cm are new branches, written only when the knob is on.
  * role-5 capture-gamma rows and n_stop_gammas may appear on firing items: stop_gamma_require_stm
    is ON in PDVD production, so a candidate whose verdict stops rejecting may now publish a gamma.
    Point-row geometry therefore changes on firing items.
  * michel_found must NOT move anywhere: the rule makes no Michel object.  Asserted by the gate.

Fork by duplication (CLAUDE.md M10) of d91_wide_anchor.py (its anchor / shape / kslike / p1 twin and
its threshold reader); d91 stays as doc 91's record.  The median convention was checked, not assumed:
stm_michel_median averages the two middle values for an even count (StmMichelFunctions.cxx:208-215),
which is numpy's convention, so np.median here and the C++ agree on the D guard's drop median.

Usage: STM_SCAN_RECORD=<smx1a..smx7 record> d92_twin.py --prep DIR --cfg post.json --root-arm ARM
       [--json OUT]
"""
import argparse, collections, csv, json, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
TK = "/home/xqian/toolkit-dev/toolkit/clus/src"
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5+smx6+smx7 record (doc 90)")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True, help="the production prep (doc 90: prep_p90vprod)")
ap.add_argument("--cfg", required=True, help="the compiled production config (wcsonnet output)")
ap.add_argument("--root-arm", required=True, help="read the unrounded profile rows from this arm's tracking-pr.root")
ap.add_argument("--json", default=None)
a = ap.parse_args()

# The two pre-registered operating points (doc 91 sec 4 / sec 6).  W and D are common; R is the pick.
ARMS = [("p92v13", 8.0, 1.3, 0.8), ("p92v15", 8.0, 1.5, 0.8)]
# The points whose movers the owner's scan could still change (doc 92 sec 4): the selection, the
# conservative alternative, the D neighbour and the W neighbour.
DECISION = [(8.0, 1.3, 0.8), (8.0, 1.5, 0.8), (8.0, 1.3, 1.0), (10.0, 1.3, 0.8)]

NB, SF, PS, PM = 1 << 2, 1 << 3, 1 << 9, 1 << 10
SHAPE = NB | SF | PS | PM


# ---------------------------------------------------------------- 0 thresholds (d89/d91's reader)
def cfg_node(cfg, typ):
    stack = [cfg]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get("type") == typ:
                return x.get("data", {})
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    sys.exit("no %s node in %s" % (typ, a.cfg))


SRC = open(os.path.join(TK, "CheckSTM_Michel.cxx")).read()


def cpp_default(member):
    m = re.search(r"\b%s\{([^}]*)\}" % re.escape(member), SRC)
    if not m:
        sys.exit("no initializer for %s in CheckSTM_Michel.cxx" % member)
    s = m.group(1).strip()
    if s in ("true", "false"):
        return s == "true"
    scale = 10.0 if "units::cm" in s else 1.0
    return float(s.split("*")[0]) * scale


CK = cfg_node(json.load(open(a.cfg)), "CheckSTM_Michel")
TH, THSRC = {}, {}


def th(key):
    if key in CK:
        TH[key], THSRC[key] = CK[key], "compiled"
    else:
        TH[key], THSRC[key] = cpp_default("m_" + key), "C++ default"
    return TH[key]


MIP = th("mip_dqdx"); LIVE = th("profile_min_dqdx_frac") * MIP
ANCHOR = th("bragg_peak_anchor"); SEARCH = th("bragg_peak_search_cm")
FALLBACK = th("bragg_anchor_geo_fallback"); RISE = th("bragg_anchor_rise_min")
TLO = th("bragg_tail_lo_cm"); THI = th("bragg_tail_hi_cm")
BPLO = th("bragg_plateau_lo_cm"); BPHI = th("bragg_plateau_hi_cm")
CMIN = th("bragg_contrast_min"); CMP = th("compare_range_cm"); OFF = th("offset_length_cm")
KSM = th("ks_margin"); PLO = th("plateau_mip_lo"); PHI = th("plateau_mip_hi")
KE = th("topology_michel_ke_min"); LEN = th("topology_michel_len_min_cm")
P1ON = th("topology_stop_evidence"); P1C = NB | SF | (PS if th("topology_clears_sparse") else 0)
print("=== 0. thresholds, with their source (%s) ===" % a.cfg)
for k in TH:
    print("  %-28s %-8s %s" % (k, TH[k], THSRC[k]))
if not (ANCHOR and FALLBACK and P1ON):
    sys.exit("the compiled config is not today's production (anchor, fallback and P1 all on)")

REF = json.load(open(os.path.join(a.prep, "dqdx_ref_pdvd.json")))
_g = REF["grid"]
GX = _g["start"] + _g["step"] * np.arange(_g["n"]); GY = np.asarray(REF["muon"], float)


def mu(r):
    return np.interp(r, GX, GY)


# ---------------------------------------------------------------- the twin (d91's, unchanged)
def kslike(t, r):
    t = np.asarray(t); r = np.asarray(r)
    return float(np.abs(np.cumsum(t) / t.sum() - np.cumsum(r) / r.sum()).max())


def anchor(L, rr, q, tot, W):
    """T7 on the live rows within W cm of the end -> dict or None (not fired)."""
    n = len(L)
    if n < 5:
        return None
    best, ib = -1.0, n - 1
    for i in range(n):
        if rr[i] > W:
            continue
        s = q[max(0, i - 2):min(n - 1, i + 2) + 1].mean()
        if s > best:
            best, ib = s, i
    end_L = L[ib] + 0.2
    if not end_L < tot:
        return None
    keep = (end_L - L) >= 0
    if keep.sum() < 3:
        return None
    return dict(end_L=end_L, shift=tot - end_L, peak5=best, drop=q[~keep])


def shape(L, rr, q, tot, end_L=None):
    """the four shape bits and the plateau median, at the anchored (end_L) or geometric origin"""
    if end_L is not None:
        k = (end_L - L) >= 0; L = L[k]; q = q[k]; rr = end_L - L; T = end_L
    else:
        T = tot
    b = 0
    plo, phi = (BPLO, BPHI) if T >= BPHI else (0.5 * BPLO, 0.5 * BPHI)
    tm = (rr >= TLO) & (rr <= THI); pm = (rr >= plo) & (rr <= phi); pl = None
    if tm.sum() < 3 or pm.sum() < 3 or np.median(q[pm]) <= 0:
        b |= PS
    else:
        pl = float(np.median(q[pm])); c = float(np.median(q[tm])) / pl
        e = float(np.median(mu(rr[tm])) / np.median(mu(rr[pm])))
        if e <= 0:
            b |= PS
        elif c < CMIN * e:
            b |= NB
        if pl / MIP < PLO or pl / MIP > PHI:
            b |= PM
    cm = rr <= CMP
    if cm.sum() >= 3:
        if kslike(q[cm], mu(rr[cm] + OFF)) + KSM >= kslike(q[cm], np.full(cm.sum(), MIP)):
            b |= SF
    else:
        b |= PS
    return b, pl


def p1(x, b):
    if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= KE and x["michel_len"] >= LEN:
        return b & P1C
    return 0


def names(b):
    return "+".join(n for i, n in enumerate(C.STM_BITS) if b >> i & 1) or "STM"


# ---------------------------------------------------------------- the record and the profiles
R = C.load_record()
J = [k for k in R if C.judged(R[k])]


def cls(k):
    if k not in R:
        return "unjudged"
    if not C.judged(R[k]):
        return "MESSY/UNCLEAR"
    return "stopper" if C.is_stopper(R[k]) else "THRU"


V, PRO, MU = {}, {}, {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        p = json.load(open(os.path.join(a.prep, f)))
        V[f[7:-5].replace("-c", "/")] = p["verdict"]
        if p["muon"] and p["muon"].get("L"):
            MU[f[7:-5].replace("-c", "/")] = p["muon"]
import uproot  # noqa: E402
byev = collections.defaultdict(list)
for k in MU:
    byev[k.split("/")[0]].append(k)
nrow = 0; dLmax = 0.0
for ev, ks in sorted(byev.items()):
    t = uproot.open(os.path.join(C.WORK, "%s_%s" % (ev, a.root_arm), "tracking-pr.root"))["T_stm_michel_pts"]
    p = t.arrays(["L", "rr", "q", "role", "cluster_id"], library="np")
    for k in ks:
        sel = (p["cluster_id"] == int(k.split("/")[1])) & (p["role"] == 1)
        L, rr, q = p["L"][sel].astype(float), p["rr"][sel].astype(float), p["q"][sel].astype(float)
        Lp = np.asarray(MU[k]["L"], float)
        if len(L) != len(Lp):
            sys.exit("%s: %d rows in %s's tree, %d in the payload" % (k, len(L), a.root_arm, len(Lp)))
        dLmax = max(dLmax, float(np.abs(L - Lp).max())); nrow += len(L)
        tot = float(L[-1] + rr[-1]); lv = q >= LIVE
        PRO[k] = (L[lv], rr[lv], q[lv], tot)
print("\nprofiles: unrounded rows from %s's T_stm_michel_pts, %d rows on %d candidates; max |L - payload L| %.4f cm"
      % (a.root_arm, nrow, len(PRO), dLmax))
if dLmax > 0.0051:
    sys.exit("the tree and the payload are not the same rows")
print("record %s: %d records, %d judged; prep %s: %d candidates, %d with a profile" % (
    os.path.basename(C.REC), len(R), len(J), a.prep, len(V), len(PRO)))


def pay_bits(x):
    return int(x["reject_bits"] or 0), int(x.get("topology_cleared_bits") or 0)


# ---------------------------------------------------------------- 1 the twin against production
print("\n=== 1. the twin against the payload (production's own reading: anchor %.1f cm, fallback, P1) ===" % SEARCH)
BAD = set()
n_stm = n_bits = n_shift = n_fb = 0
for k, x in sorted(V.items()):
    rb, tcb = pay_bits(x)
    if k not in PRO:
        BAD.add(k); print("   no profile: %s [%s] published %s" % (k, cls(k), names(rb))); continue
    L, rr, q, tot = PRO[k]
    an = anchor(L, rr, q, tot, SEARCH)
    bA, plA = shape(L, rr, q, tot, an["end_L"] if an else None)
    fb = 0
    if FALLBACK and an and plA and an["peak5"] >= RISE * plA and bA & SHAPE:
        bG, _ = shape(L, rr, q, tot, None)
        if not bG & SHAPE:
            bA, fb = bG, 1
    pre = ((rb | tcb) & ~SHAPE) | (bA & SHAPE)
    clr = p1(x, pre); fin = pre & ~clr
    ok_stm = (fin == 0) == bool(x["is_stm"]); ok_bits = (fin, clr) == (rb, tcb)
    tsh = an["shift"] if an else 0.0; psh = float(x["bragg_anchor_shift_cm"] or 0)
    n_stm += ok_stm; n_bits += ok_bits
    n_shift += abs(tsh - psh) <= 0.011; n_fb += fb == int(x.get("bragg_anchor_fallback") or 0)
    if not (ok_stm and ok_bits) or abs(tsh - psh) > 0.011:
        BAD.add(k)
        print("   %-14s [%-13s] published is_stm %d %-30s | twin is_stm %d %-30s | shift %.2f/%.2f | fb %d/%d" % (
            k, cls(k), x["is_stm"], names(rb), fin == 0, names(fin), tsh, psh, fb, int(x.get("bragg_anchor_fallback") or 0)))
N = len(PRO)
print("is_stm agrees on %d / %d; (reject_bits, topology_cleared_bits) on %d / %d; shift within 0.01 cm on %d; fallback flag on %d"
      % (n_stm, N, n_bits, N, n_shift, n_fb))
if len(BAD):
    print("!! %d item(s) disagree; the arms cannot be graded item by item until this is 0" % len(BAD))

PRE = {k: pay_bits(x)[0] | pay_bits(x)[1] for k, x in V.items()}   # production's bits before P1


# ---------------------------------------------------------------- 2 the prediction, per arm
# EVERY item the C++ block can fire on: a shape bit is set where the block sits.  NOT restricted to
# is_stm 0 -- an item P1 was going to clear is fired on too, and its topology_cleared_bits moves.
FIRE = sorted(k for k in V if (PRE[k] & SHAPE) and k in PRO)
WI = {}
for k in FIRE:
    L, rr, q, tot = PRO[k]
    for W in sorted({w for _, w, _, _ in ARMS} | {w for w, _, _ in DECISION}):
        an = anchor(L, rr, q, tot, W)
        if an is None:
            WI[(k, W)] = None; continue
        bW, plW = shape(L, rr, q, tot, an["end_L"])
        d = an["drop"]
        WI[(k, W)] = dict(shift=an["shift"], peak5=an["peak5"], bW=bW, plW=plW, nd=int(len(d)),
                          dmed=(float(np.median(d)) if len(d) else float("nan")))


def fires(k, W, R_, D_):
    w = WI.get((k, W))
    if w is None or w["bW"] & SHAPE or not w["plW"]:
        return None
    if R_ and w["peak5"] < R_ * w["plW"]:
        return None
    if D_ and (w["nd"] < 3 or w["dmed"] > D_ * w["plW"]):
        return None
    return w


print("\n=== 2. the prediction for each ON arm, item by item (every firing item, not only is_stm movers) ===")
PRED = {}
for arm, W, R_, D_ in ARMS:
    rows = {}
    for k in FIRE:
        w = fires(k, W, R_, D_)
        if w is None:
            continue
        rb, tcb = pay_bits(V[k])
        b = PRE[k] & ~SHAPE
        clr = p1(V[k], b); fin = b & ~clr
        rows[k] = dict(bragg_wide_fired=1, bragg_wide_shift_cm=round(w["shift"], 4),
                       reject_bits_before=rb, reject_bits_after=fin,
                       topology_cleared_before=tcb, topology_cleared_after=clr,
                       is_stm_before=int(bool(V[k]["is_stm"])), is_stm_after=int(fin == 0),
                       verdict=cls(k), p=round(w["peak5"] / w["plW"], 3), d=round(w["dmed"] / w["plW"], 3), nd=w["nd"])
    mov = sorted(k for k in rows if rows[k]["is_stm_after"] != rows[k]["is_stm_before"])
    quiet = sorted(k for k in rows if k not in mov and
                   (rows[k]["reject_bits_after"], rows[k]["topology_cleared_after"]) !=
                   (rows[k]["reject_bits_before"], rows[k]["topology_cleared_before"]))
    same = sorted(k for k in rows if k not in mov and k not in quiet)
    PRED[arm] = dict(W=W, R=R_, D=D_, items=rows, movers=mov, bits_only=quiet, no_change=same)
    by = collections.defaultdict(list)
    for k in mov:
        by[cls(k)].append(k)
    print("\n  %s  (W %.1f, R %.1f, D %.1f): fires on %d item(s)" % (arm, W, R_, D_, len(rows)))
    print("    is_stm 0 -> 1 : %d  [stoppers %d, THRU %d, MESSY %d, unjudged %d]" % (
        len(mov), len(by["stopper"]), len(by["THRU"]), len(by["MESSY/UNCLEAR"]), len(by["unjudged"])))
    for k in mov:
        r = rows[k]
        print("      %-14s %-9s %-26s -> STM   shift %.2f p %.2f d %.2f(%d)" % (
            k, r["verdict"], names(r["reject_bits_before"]), r["bragg_wide_shift_cm"], r["p"], r["d"], r["nd"]))
    print("    bits move, is_stm does NOT (P1 would have cleared these anyway): %d" % len(quiet))
    for k in quiet:
        r = rows[k]
        print("      %-14s %-9s cleared_bits %s -> %s" % (
            k, r["verdict"], names(r["topology_cleared_before"]), names(r["topology_cleared_after"])))
    print("    fires but publishes exactly what it did before: %d" % len(same))
    if any(rows[k]["is_stm_before"] and not rows[k]["is_stm_after"] for k in rows):
        sys.exit("BUG: the rule lost a stopper; it is clears-only and cannot")

# the census at each point, on the judged record
for arm, W, R_, D_ in ARMS:
    mov = set(PRED[arm]["movers"])
    def stm(k):
        return bool(V[k]["is_stm"]) if k in V else False
    tp = sum(1 for k in J if (stm(k) or k in mov) and C.is_stopper(R[k]))
    fp = sum(1 for k in J if (stm(k) or k in mov) and not C.is_stopper(R[k]))
    fn = sum(1 for k in J if not (stm(k) or k in mov) and C.is_stopper(R[k]))
    tp0 = sum(1 for k in J if stm(k) and C.is_stopper(R[k]))
    fp0 = sum(1 for k in J if stm(k) and not C.is_stopper(R[k]))
    fn0 = sum(1 for k in J if not stm(k) and C.is_stopper(R[k]))
    PRED[arm]["census"] = [tp, fp, fn]; PRED[arm]["census_before"] = [tp0, fp0, fn0]
    print("  %s census on the %d judged: %d/%d/%d (eff %.3f pur %.3f)  <- production %d/%d/%d (eff %.3f pur %.3f)" % (
        arm, len(J), tp, fp, fn, tp / (tp + fn), tp / (tp + fp), tp0, fp0, fn0, tp0 / (tp0 + fn0), tp0 / (tp0 + fp0)))


# ---------------------------------------------------------------- 3 the expected diff set
print("\n=== 3. the gate's expected diff set (anything outside this is a FAILURE) ===")
print("  branches expected ONLY in the ON arms : bragg_wide_fired, bragg_wide_shift_cm")
print("  branches expected to MOVE, and where  : reject_bits / topology_cleared_bits on the firing items above;")
print("                                          is_stm on the movers only")
print("  point rows expected to move           : role-5 capture-gamma rows on movers (stop_gamma_require_stm is ON,")
print("                                          so a candidate whose verdict stops rejecting may publish a gamma)")
print("  branches that must NOT move anywhere  : michel_found, michel_ke_best, michel_len, michel_conn_type,")
print("                                          bragg_anchor_shift_cm, bragg_anchor_fallback, ks_mu, ks_flat,")
print("                                          ratio_mu, ratio_flat, tail_med, plateau_med, n_live_pts,")
print("                                          dead_frac_cmp, comp_fwd*, comp_bwd*  (rec.bragg is NOT replaced)")
print("  the OFF arms                          : byte-identical to production on every branch and every point row")


# ---------------------------------------------------------------- 4 the smx8 decision set
# The items whose verdict could still change the owner's pick: a mover at ANY of the four candidate
# points.  Chosen BEFORE the scan and frozen; the scan answers a factual question about each, it does
# not re-select the operating point (that is doc 91's pre-registered W 8 / R 1.3 / D 0.8).
print("\n=== 4. the smx8 decision set: every item a candidate operating point would move ===")
DEC = {}
for W, R_, D_ in DECISION:
    for k in FIRE:
        w = fires(k, W, R_, D_)
        if w is None:
            continue
        b = PRE[k] & ~SHAPE
        if (b & ~p1(V[k], b)) != 0:
            continue                      # not an is_stm mover at this point
        if V[k]["is_stm"]:
            continue
        DEC.setdefault(k, []).append([W, R_, D_])
print("  %d item(s), with the points that move them:" % len(DEC))
for k in sorted(DEC, key=lambda k: (cls(k), k)):
    r = R.get(k, {})
    w8 = WI.get((k, 8.0)); w10 = WI.get((k, 10.0))
    print("    %-14s %-8s %-6s %-26s | W8 %s | W10 %s | moved at %s" % (
        k, cls(k), r.get("confidence", "-"), names(PRE[k]),
        ("p %.2f d %.2f(%2d)" % (w8["peak5"] / w8["plW"], w8["dmed"] / w8["plW"], w8["nd"])) if w8 and w8["plW"] else "-",
        ("p %.2f d %.2f(%2d)" % (w10["peak5"] / w10["plW"], w10["dmed"] / w10["plW"], w10["nd"])) if w10 and w10["plW"] else "-",
        " ".join("%g/%g/%g" % tuple(t) for t in DEC[k])))
print("  the two factual questions the scan answers (it does NOT re-select the operating point):")
print("    Q1  is 039349_81/25 a through-going track?   -> R 1.3 (it stays out by 0.02) vs R 1.5 (out by 0.22)")
print("    Q2  are the four d 0.93-0.97 items THRU?      -> whether D 1.0 could ever be admissible")

if a.json:
    json.dump(dict(profiles=a.root_arm, thresholds={k: [TH[k], THSRC[k]] for k in TH}, bad=sorted(BAD),
                   arms={k: {kk: vv for kk, vv in v.items()} for k, v in PRED.items()},
                   decision={k: DEC[k] for k in DEC},
                   fire=FIRE), open(a.json, "w"), indent=1)
    print("\nwrote", a.json)

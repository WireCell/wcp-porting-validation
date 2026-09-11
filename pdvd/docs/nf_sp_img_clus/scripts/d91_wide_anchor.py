#!/usr/bin/env python3
"""doc pdvd/91 -- a second, wider Bragg-peak read for the stoppers production rejects on the shape
tests, sized offline before any build (doc 90 sec 8 item 1).  Read-only.

Production (CheckSTM_Michel.cxx:2687-2847): the anchor (doc 65 T7) puts the residual-range origin at
the 5-point running-mean maximum of the live profile within bragg_peak_search_cm (3 cm) of the fit
end; doc 75's fallback re-reads the four shape tests at the geometric end when the anchored reading
keeps a shape bit and the anchor's peak is >= bragg_anchor_rise_min x the anchored plateau.  On the
owner's fit-through Michels (doc 90 sec 6) the Bragg peak sits 3.6-7.2 cm back from the fit end.

The rule sized here is a candidate for a NEW default-OFF knob (bragg_peak_search_cm is not
touched).  Placement: after the fallback, before P1 (stm_michel_topology_clear, :3989).  It runs only
where the reading that stands still carries a shape bit (no_bragg | shape_flat | plateau_off_mip |
profile_sparse), re-anchors at the 5-point maximum within W cm of the geometric end with the same
recipe on the same live rows, reads the same four tests there, and clears all four only if
  - that reading sets none of them,
  - R: its peak's 5-point mean >= R x its plateau median (R 0 = no guard),
  - D: >= 3 live rows lie past the wide peak and their median is <= D x its plateau median
    (D 0 = no guard; doc 90 sec 8's "the tail past the peak reads below the plateau").
It never sets a bit, so it cannot lose a stopper.  PID (do_track_comp), the dead-volume probe,
n_live_pts and dead_frac_cmp keep the anchored profile, so every non-shape bit is the payload's.
Then P1 is re-applied at the compiled floors.  It acts on is_stm only: no Michel object is made.

Section 1 validates the twin before anything is predicted: production's own reading (the anchor at
bragg_peak_search_cm, the fallback, P1) must reproduce the published reject_bits /
topology_cleared_bits / is_stm.  Disagreements are named, and a gain on one of them is flagged "*",
not counted as a prediction.  Known limit (doc 75): the payload's L is rounded to 0.01 cm, so a row
at exactly a window edge can fall on the other side than in the C++.

Fork by duplication of d75_sizing.py's twin (anchor / shape / kslike), d89_miss_census.py's
threshold reader and d90_twin.py's P1 re-application; those scripts stay as their docs' records.
The grid and the selection rule are pre-registered in /home/xqian/tmp/p91/pred.txt.

Profiles: with --root-arm the twin reads each candidate's unrounded role-1 rows (L, rr, q; 8-byte
doubles) from T_stm_michel_pts in pdvd/work/<event>_<arm>/tracking-pr.root, joined to the payload row
for row, and compares rr against the window edges with no tolerance -- the rows sit on exact 0.6 cm
steps, so nominal rr 3.0 / 6.0 / 20.0 fall on an edge that only the unrounded value decides.  Without it
the payload's rounded L is used as in doc 75 (rr = total - L, edges inclusive by 1e-9).

Usage: STM_SCAN_RECORD=<smx1a..smx7 record> d91_wide_anchor.py --prep DIR --cfg post.json
       [--root-arm ARM] [--json OUT]
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
ap.add_argument("--prep", required=True, help="a bare-production prep (doc 90: prep_p90vprod)")
ap.add_argument("--cfg", required=True, help="the arm's compiled config (wcsonnet output)")
ap.add_argument("--root-arm", default=None, help="read the unrounded profile rows from this arm's tracking-pr.root")
ap.add_argument("--json", default=None)
a = ap.parse_args()
EPS = 0.0 if a.root_arm else 1e-9

GRID_W = (4.0, 5.0, 6.0, 8.0, 10.0)
GRID_R = (0.0, 1.3, 1.5, 1.8)
GRID_D = (0.0, 1.0, 0.8)          # 0 = off; listed loosest to strictest
NB, SF, PS, PM = 1 << 2, 1 << 3, 1 << 9, 1 << 10
SHAPE = NB | SF | PS | PM
KEY7 = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx7_key.tsv"
LAB7 = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx7_labels.json"


# ---------------------------------------------------------------- 0 thresholds (d89_miss_census.py's reader)
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
    scale = 10.0 if "units::cm" in s else 1.0          # WCT internal length unit is mm
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


# ---------------------------------------------------------------- the twin (d75_sizing.py's, constants above)
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
        if rr[i] > W + EPS:
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
    """the bits P1 clears (d90_twin.py's verdict(), validated 0 mismatches on doc 90's production)"""
    if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= KE and x["michel_len"] >= LEN:
        return b & P1C
    return 0


def names(b):
    return "+".join(n for i, n in enumerate(C.STM_BITS) if b >> i & 1) or "STM"


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
if a.root_arm:
    import uproot
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
    print("profiles: unrounded rows from %s's T_stm_michel_pts, %d rows on %d candidates; max |L - payload L| %.4f cm"
          % (a.root_arm, nrow, len(PRO), dLmax))
    if dLmax > 0.0051:
        sys.exit("the tree and the payload are not the same rows")
else:
    for k, m in MU.items():
        L = np.asarray(m["L"], float); q = np.asarray(m["q"], float); tot = float(L[-1]); lv = q >= LIVE
        PRO[k] = (L[lv], tot - L[lv], q[lv], tot)
    print("profiles: the payload's rounded rows (doc 75's limit applies)")
print("\nrecord %s: %d records, %d judged; prep %s: %d candidates, %d with a profile" % (
    os.path.basename(C.REC), len(R), len(J), a.prep, len(V), len(PRO)))


def pay_bits(x):
    return int(x["reject_bits"] or 0), int(x.get("topology_cleared_bits") or 0)


# ---------------------------------------------------------------- 1 validation
print("\n=== 1. the twin against the payload (production's own reading: anchor %.1f cm, fallback, P1) ===" % SEARCH)
BAD = set()
n_stm = n_bits = n_fire_t = n_fire_p = n_shift = n_fb = 0
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
    n_stm += ok_stm; n_bits += ok_bits; n_fire_t += bool(an); n_fire_p += psh > 0
    n_shift += abs(tsh - psh) <= 0.011; n_fb += fb == int(x.get("bragg_anchor_fallback") or 0)
    if not (ok_stm and ok_bits) or abs(tsh - psh) > 0.011:
        BAD.add(k)
        print("   %-14s [%-13s] published is_stm %d %-32s | twin is_stm %d %-32s | shift twin %.2f published %.2f | fallback twin %d published %d" % (
            k, cls(k), x["is_stm"], names(rb), fin == 0, names(fin), tsh, psh, fb, int(x.get("bragg_anchor_fallback") or 0)))
N = len(PRO)
print("is_stm agrees on %d / %d; (reject_bits, topology_cleared_bits) on %d / %d; anchor fired twin %d published %d; "
      "shift within 0.01 cm on %d; fallback flag on %d" % (n_stm, N, n_bits, N, n_fire_t, n_fire_p, n_shift, n_fb))
print("items flagged '*' below (any disagreement above): %d" % len(BAD))


# ---------------------------------------------------------------- 2 the population the rule can touch
def pub_stm(k):
    return bool(V[k]["is_stm"]) if k in V else False


def census(stm):
    tp = sum(1 for k in J if stm(k) and C.is_stopper(R[k])); fp = sum(1 for k in J if stm(k) and not C.is_stopper(R[k]))
    fn = sum(1 for k in J if not stm(k) and C.is_stopper(R[k]))
    return tp, fp, fn


PRE = {k: pay_bits(x)[0] | pay_bits(x)[1] for k, x in V.items()}   # production's bits before P1 (exact)
tp0, fp0, fn0 = census(pub_stm)
print("\n=== 2. production on the record, and the population the rule can touch ===")
print("production (published) is_stm on the %d judged, no candidate = not a stopper: %d / %d / %d (eff %.3f, purity %.3f)" % (
    len(J), tp0, fp0, fn0, tp0 / (tp0 + fn0), tp0 / (tp0 + fp0)))
touch = [k for k in V if not V[k]["is_stm"] and PRE[k] & SHAPE]
only = [k for k in touch if not (PRE[k] & ~SHAPE) & ~p1(V[k], PRE[k] & ~SHAPE)]
for nm, ks in (("is_stm 0 with a shape bit before P1", touch), ("... and no other bit left after P1 (the ceiling)", only)):
    cc = collections.Counter(cls(k) for k in ks)
    print("  %-50s %3d: %s" % (nm, len(ks), dict(sorted(cc.items()))))
print("  the ceiling's stoppers: %s" % " ".join(sorted(k for k in only if cls(k) == "stopper")))

# ---------------------------------------------------------------- 3 the grid
WI = {}
for k in touch:
    if k not in PRO:
        continue
    L, rr, q, tot = PRO[k]
    for W in GRID_W:
        an = anchor(L, rr, q, tot, W)
        if an is None:
            WI[(k, W)] = None; continue
        bW, plW = shape(L, rr, q, tot, an["end_L"])
        d = an["drop"]
        WI[(k, W)] = dict(shift=an["shift"], peak5=an["peak5"], bW=bW, plW=plW, nd=len(d),
                          dmed=(float(np.median(d)) if len(d) else float("nan")))


def wide(k, W, R_, D_):
    """the published bits under the rule, or None when it does not fire"""
    w = WI.get((k, W))
    if w is None or w["bW"] & SHAPE:
        return None
    if R_ and w["peak5"] < R_ * w["plW"]:
        return None
    if D_ and (w["nd"] < 3 or w["dmed"] > D_ * w["plW"]):
        return None
    b = PRE[k] & ~SHAPE
    return b & ~p1(V[k], b)


def star(k):
    return k + ("*" if k in BAD else "")


print("\n=== 3. the grid (W cm, R x plateau, D x plateau; 0 = guard off) ===")
print("fires = shape bits cleared; movers = is_stm 0 -> 1.  '*' = the twin disagrees with production on that item (section 1)")
G = []
for W in GRID_W:
    for R_ in GRID_R:
        for D_ in GRID_D:
            fires, mov = [], []
            for k in touch:
                b = wide(k, W, R_, D_)
                if b is None:
                    continue
                fires.append(k)
                if b == 0:
                    mov.append(k)
            by = collections.defaultdict(list)
            for k in mov:
                by[cls(k)].append(k)
            gain = [k for k in by["stopper"] if k not in BAD]
            tp, fp, fn = census(lambda k: pub_stm(k) or k in mov)
            g = dict(W=W, R=R_, D=D_, fires=len(fires), movers=sorted(mov), stoppers=sorted(by["stopper"]),
                     thru=sorted(by["THRU"]), messy=sorted(by["MESSY/UNCLEAR"]), unjudged=sorted(by["unjudged"]),
                     gain_counted=len(gain), census=[tp, fp, fn])
            G.append(g)
            print("W %4.1f R %.1f D %.1f | fires %3d movers %3d | +stoppers %2d +THRU %2d MESSY %d unjudged %2d | %d/%d/%d eff %.3f pur %.3f" % (
                W, R_, D_, len(fires), len(mov), len(by["stopper"]), len(by["THRU"]), len(by["MESSY/UNCLEAR"]),
                len(by["unjudged"]), tp, fp, fn, tp / (tp + fn), tp / (tp + fp)))
lost = [k for k in V if V[k]["is_stm"] and any(wide(k, W, 0, 0) not in (None, 0) for W in GRID_W)]
print("is_stm 1 -> 0 anywhere on the grid: %d (the rule runs only where is_stm is 0)" % len(lost))

# ---------------------------------------------------------------- 4 the pre-registered selection
print("\n=== 4. the pre-registered selection: 0 judged THRU, most judged stoppers (counted, no '*') ===")
DSTRICT = {0.0: 2, 1.0: 1, 0.8: 0}
ok = [g for g in G if not g["thru"] and g["gain_counted"] > 0]


def show(g, tag=""):
    print("%sW %.1f R %.1f D %.1f: +%d stoppers %s | THRU %s | MESSY %s | unjudged %d %s | census %d/%d/%d" % (
        tag, g["W"], g["R"], g["D"], len(g["stoppers"]), " ".join(star(k) for k in g["stoppers"]) or "-",
        " ".join(g["thru"]) or "-", " ".join(g["messy"]) or "-", len(g["unjudged"]), " ".join(g["unjudged"]), *g["census"]))


SEL = None
if not ok:
    print("no grid point gains a counted stopper at 0 judged THRU")
else:
    SEL = sorted(ok, key=lambda g: (-g["gain_counted"], len(g["messy"]) + len(g["unjudged"]), g["W"], -g["R"], DSTRICT[g["D"]]))[0]
    show(SEL, "SELECTED  ")
    iw, ir, idd = GRID_W.index(SEL["W"]), GRID_R.index(SEL["R"]), GRID_D.index(SEL["D"])
    nb = []
    for dw, dr, dd in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        j = (iw + dw, ir + dr, idd + dd)
        if 0 <= j[0] < len(GRID_W) and 0 <= j[1] < len(GRID_R) and 0 <= j[2] < len(GRID_D):
            nb.append([g for g in G if (g["W"], g["R"], g["D"]) == (GRID_W[j[0]], GRID_R[j[1]], GRID_D[j[2]])][0])
    for g in nb:
        show(g, "neighbour ")
    plateau = any(not g["thru"] and g["gain_counted"] > 0 for g in nb)
    print("bar (pred.txt): >= 1 counted stopper at 0 THRU [%s] and a one-step neighbour also at 0 THRU with a gain [%s]" % (
        "met" if SEL["gain_counted"] else "not met", "met" if plateau else "not met"))
print("\nevery judged THRU admitted anywhere on the grid, with the first (loosest-guard) point that admits it:")
seen = {}
for g in G:
    for k in g["thru"]:
        seen.setdefault(k, g)
for k, g in sorted(seen.items()):
    admitted = [(x["W"], x["R"], x["D"]) for x in G if k in x["thru"]]
    print("  %-14s admitted at %2d points; at R >= 1.3: %d; at D > 0: %d" % (
        k, len(admitted), sum(1 for t in admitted if t[1] >= 1.3), sum(1 for t in admitted if t[2] > 0)))

# ---------------------------------------------------------------- 5 per item
S7 = {r["key"]: r for r in csv.DictReader([l for l in open(KEY7) if not l.startswith("#")], delimiter="\t")}
L7 = json.load(open(LAB7))["labels"]
b1 = sorted((k for k in S7 if S7[k]["group"] == "b1"), key=lambda k: (L7[k]["choice"], int(S7[k]["scan_id"])))
moved = sorted({k for g in G for k in g["movers"]} - set(b1))


def pin_rr(k):
    r = R.get(k) or {}
    pin = r.get("pin") or {}
    if pin.get("placed"):
        return pin.get("rr", r.get("pin_rr"))
    return r.get("pin_rr")


def item(k):
    x = V[k]
    cells = []
    for W in GRID_W:
        w = WI.get((k, W))
        if w is None:
            cells.append("W%g: -" % W); continue
        cells.append("W%g: o %.1f p %.2f d %.2f(%d) %s" % (W, w["shift"], w["peak5"] / w["plW"] if w["plW"] else float("nan"),
                                                           w["dmed"] / w["plW"] if w["plW"] else float("nan"), w["nd"],
                                                           names(w["bW"] & SHAPE) if w["bW"] & SHAPE else "clear"))
    pr = pin_rr(k)
    print("  %-14s %-10s pin %-5s published %-28s shift %.2f fb %d\n      %s" % (
        star(k), (L7[k]["choice"] if k in L7 else cls(k)), "%.1f" % pr if pr else "-", names(pay_bits(x)[0]),
        float(x["bragg_anchor_shift_cm"] or 0), int(x.get("bragg_anchor_fallback") or 0), " | ".join(cells)))


print("\n=== 5. per item: the wide origin o (cm back from the fit end), p = peak5 / plateau, d = median past the peak / plateau (rows), bits ===")
print("smx7 part (b1), by the owner's call:")
for k in b1:
    if k in touch:
        item(k)
    else:
        print("  %-14s not touched (is_stm %d, bits %s)" % (k, V[k]["is_stm"], names(PRE[k])))
print("every other candidate that moves somewhere on the grid (%d):" % len(moved))
for k in moved:
    item(k)

# ---------------------------------------------------------------- 6 the pinned five
print("\n=== 6. the 5 pinned fit-through Michels: the wide origin against the owner's pin ===")
for k in [k for k in b1 if L7[k]["choice"] == "STM_MICHEL" and (L7[k].get("pin") or {}).get("placed")] or \
         [k for k in b1 if L7[k]["choice"] == "STM_MICHEL" and pin_rr(k)]:
    pr = pin_rr(k)
    row = []
    for W in GRID_W:
        w = WI.get((k, W))
        row.append("W%g %s" % (W, ("%.1f (%+.1f)" % (w["shift"], w["shift"] - pr)) if w else "-"))
    gp = [(g["W"], g["R"], g["D"]) for g in G if k in g["movers"]]
    print("  %-14s pin %.1f | %s | gained at %d grid points%s" % (star(k), pr, "  ".join(row), len(gp),
                                                                   (", first %s" % (gp[0],)) if gp else ""))

if a.json:
    json.dump(dict(profiles=a.root_arm or "payload", thresholds={k: [TH[k], THSRC[k]] for k in TH}, bad=sorted(BAD), grid=G, selected=SEL,
                   touch=sorted(touch), ceiling=sorted(only)), open(a.json, "w"), indent=1)
    print("\nwrote", a.json)

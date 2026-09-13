#!/usr/bin/env python3
"""doc pdhd/29 sec 8-9 -- the owner's follow-up: is PDVD's bottom volume (BDE, PDHD's cold electronics) consistent with
PDHD, so that the top/bottom 0.889 can be read as the top (TDE) electronics gain; and is the field dependence of
recombination carried into each chain's Michel charge -> energy conversion?

    python3 d29_gain_recomb.py [--fig ../../figs/29_gain_recomb.png] [--skip-frames] [--skip-michel] > gain_recomb.txt

A  EXPECTATION AUDIT.  Rebuild the muon dQ/dx-vs-rr table from the CSDA source with convert_field.C's own formula
   (energy_loss/pion_travel/convert_field.C:38-42 constants, :71 formula, :90-100 1-cm bins of 10-point averages,
   TGraph::Eval linear) at 0.4959 / 0.45 / 0.4393 kV/cm and compare to the committed tables and to the chain's own
   dqdx_ref block (calib-pr-evt*.json of the production arms h28prod / p96vprod).
B  PDVD BOTTOM AGAINST PDHD on the committed per-track plateaus of doc pdhd/27 sec 5 (d27/dqdx_drift.json; median
   dQ/dx / expected over rr 40-60 cm, no free scale) and the per-track W-plane measured/predicted of
   d27/dqdx_rr_vs_drift.json (mpW; doc 27 sec 5.3).  Medians, ratios, matched drift TIME, angle, and a systematic
   budget.  The 1-D space-charge toy is COPIED from d29_space_charge.py:41-63 (that script runs its analysis and
   writes its figure at import, so it cannot be imported) and self-checked against space_charge.txt.
C  WHAT SP ASSUMES.  Electronics response per unit from the config (ColdElec formula util/src/Response.cxx:311-335;
   the top JsonElecResponse file) -- peak and area -- and, on real frames, SP's effective electrons per ADC tick on
   collection signals: sum(frame_gauss) / sum(post-NF frame_raw) over channels whose raw signal in the dilated gauss
   ROI is unipolar.  The SP archives are the ones the production arms' imaging was built from.
D  RECOMBINATION IN THE MICHEL CONVERSION.  R(dE/dx, E) of the Modified Box at each field; the PowerBoxRecombination
   k and C bound to CheckSTM_Michel (read from the jsonnet and the compiled PDHD config); MeV per electron
   = Wi / (C R(michel_unfit_dedx 2.1)); its field x C split; the MIP-at-2.1 bound; the PDVD Michel region energy by
   drift volume on p96vprod (hand-Michel population of doc pdhd/26 sec 3).
Bootstraps are over tracks / items, fixed seeds, [p16, p84] unless marked 95 %.
"""
import argparse, concurrent.futures, glob, io, json, os, re, sys, tarfile
import numpy as np
import uproot
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOPDIR = "/nfs/data/1/xqian/toolkit-dev"
IMG = TOPDIR + "/wcp-porting-img"
TK = TOPDIR + "/toolkit"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--fig", default=os.path.join(HERE, "..", "..", "figs", "29_gain_recomb.png"))
ap.add_argument("--skip-frames", action="store_true")
ap.add_argument("--skip-michel", action="store_true")
a = ap.parse_args()

# ---------------------------------------------------------------- constants (cited)
A_BOX, B_BOX, RHO, WION, FUDGE = 0.93, 0.212, 1.38, 23.6e-6, 0.85     # convert_field.C:38-42
F_HD, F_VD, F_VD_V = 0.4959, 0.45, 0.4393   # kV/cm: pdhd/particle_dataset.jsonnet:18; protodunevd/params.jsonnet:133-141;
                                            # PDVD 1.48073 mm/us at 87.68 K (pdvd/stm/pdvd_transport.tsv)
V_HD, V_VD = 1.576, 1.48073                 # mm/us: pdhd tlas drift_speed_mmus; pdvd/run_pr_evt.sh:229-230
ANODE_BOUND = {"pdvd": {"x<0": (0, 1, 2, 3), "x>0": (4, 5, 6, 7)}}
E_PER_FC = 6241.509074                      # electrons per fC
MVFC_WCT = 1e-9 / E_PER_FC                  # 1 mV/fC in WCT units (1.602e-13)
NB = 2000


def bmed(v, seed=29, n=NB):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(m, 16)), float(np.percentile(m, 84)), m


def bratio(u, w, seed=31, n=NB):
    mu, mw = bmed(u, seed, n)[3], bmed(w, seed + 1, n)[3]
    r = mu / mw
    return (float(np.median(np.asarray(u)[np.isfinite(u)]) / np.median(np.asarray(w)[np.isfinite(w)])),
            float(np.percentile(r, 16)), float(np.percentile(r, 84)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def fm(t):
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


# ================================================================= A. expectation audit
print("=== A. the expected muon dQ/dx vs rr: rebuilt from the CSDA source with convert_field.C's formula")


def tgraph_eval(x, y, xs):
    o = np.argsort(x, kind="stable"); x, y = x[o], y[o]
    v = np.interp(xs, x, y)
    lo, hi = xs < x[0], xs > x[-1]
    v[lo] = y[0] + (xs[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    v[hi] = y[-1] + (xs[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return v


SRC = uproot.open(TOPDIR + "/energy_loss/pion_travel/stopping.root")["muon"]
MX, MY = np.asarray(SRC.values(axis="x"), float), np.asarray(SRC.values(axis="y"), float)


def rebuild(E):
    bp = B_BOX / RHO / E
    q = np.log(A_BOX + bp * MY) / (WION * bp) * FUDGE
    return np.array([tgraph_eval(MX, q, (0.5 + i) - 0.5 + 0.05 + np.arange(10) / 10.0).mean() for i in range(60)])


print(f"  source stopping.root muon: {len(MX)} points, rr {MX.min():.3f}-{MX.max():.1f} cm, dE/dx at rr 50 {np.interp(50, MX, MY):.3f} MeV/cm")
TAB = {E: rebuild(E) for E in (F_HD, F_VD, F_VD_V, 0.44)}
for f, E in (("pdhd/stm/pdhd_ref_dqdx.json", F_HD), ("pdvd/stm/pdvd_ref_dqdx_045.json", F_VD), ("pdvd/stm/pdvd_ref_dqdx.json", 0.44)):
    J = json.load(open(IMG + "/" + f))
    js = np.asarray(J["MuonDeDx"]["values"], float)
    print(f"  {f:34s} _meta E {J['_meta']['E_field_kVcm']}: max |rebuilt/json - 1| = {np.max(np.abs(TAB[E] / js - 1)):.2e}")
for det, arm, E in (("pdhd", "h28prod", F_HD), ("pdvd", "p96vprod", F_VD)):
    j = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/calib-pr-evt*.json"))[0]
    r = json.load(open(j))["dqdx_ref"]; g = r["grid"]
    idx = np.round((0.5 + np.arange(60) - g["start"]) / g["step"]).astype(int)
    ch = np.asarray(r["muon"], float)[idx]
    print(f"  chain dqdx_ref {arm:9s} ({os.path.basename(j)}): max |chain/rebuilt({E}) - 1| at rr 0.5+i = {np.max(np.abs(ch / TAB[E] - 1)):.2e}"
          "  (jsonnet carries 6 significant figures)")
CX = 0.5 + np.arange(60)
print("  rr (cm)        " + "  ".join(f"{x:>7g}" for x in (1, 5, 10, 30, 50)))
for E in (F_HD, F_VD, F_VD_V):
    print(f"  E {E:.4f} e/cm " + "  ".join(f"{np.interp(x, CX, TAB[E]):7.0f}" for x in (1, 5, 10, 30, 50)))
print("  PDHD/PDVD      " + "  ".join(f"{np.interp(x, CX, TAB[F_HD]) / np.interp(x, CX, TAB[F_VD]):7.4f}" for x in (1, 5, 10, 30, 50)))
PL = (CX >= 40) & (CX < 60)
sh = TAB[F_VD][PL].mean() / TAB[F_VD_V][PL].mean()
print(f"  plateau (rr 40-60) expectation 0.45 / 0.4393 = {sh:.4f}: if PDVD's true field is the one its drift speed implies, "
      f"PDVD plateau ratios read {100 * (sh - 1):.2f} % LOW")
print("  identical inputs on both detectors: dE/dx table, A, B, rho, W, x0.85, binning; only E differs.")

# ================================================================= B. PDVD bottom vs PDHD
print("\n=== B. PDVD bottom against PDHD, per-track plateau (median dQ/dx / expected, rr 40-60, no free scale)")
D = json.load(open(os.path.join(HERE, "..", "d27", "dqdx_drift.json")))
RV = json.load(open(os.path.join(HERE, "..", "d27", "dqdx_rr_vs_drift.json")))
MPW = {(det, r["key"]): r.get("mpW", np.nan) for det in RV for r in RV[det]}
for det in D:
    for t in D[det]:
        t["mpW"] = MPW.get((det, t["key"]), np.nan)
        t["pc"] = t["plat"] * t["mpW"] if np.isfinite(t["mpW"] if t["mpW"] is not None else np.nan) else np.nan
        t["tms"] = t["drift"] * 10.0 / ((V_HD if det == "pdhd" else V_VD) * 1000.0)   # cm / (mm/us) -> ms
BOT = [t for t in D["pdvd"] if t["side"] == "x<0"]
TOPV = [t for t in D["pdvd"] if t["side"] == "x>0"]
HD = list(D["pdhd"])
UNITS = [("PDVD bottom (BDE)", BOT), ("PDVD top (TDE)", TOPV), ("PDHD APA1-3", HD)]
print(f"  population: PDVD p96vprod every record item with a candidate; PDHD h26q2dprod APA0 strict (apa field = majority 2-D cell): "
      f"PDHD APAs {sorted(set(t['apa'] for t in HD))}")
print("  unit                         n   plateau median [p16,p84]   plateau x W meas/pred (n)       median drift time ms   median |cos_x|")
SUM = {}
for nm, tr in UNITS:
    p = bmed([t["plat"] for t in tr]); c = bmed([t["pc"] for t in tr])
    nc = int(np.isfinite([t["pc"] for t in tr]).sum())
    SUM[nm] = (p, c)
    print(f"  {nm:26s} {len(tr):3d}   {fm(p):24s}   {fm(c):22s} ({nc:3d})   {np.median([t['tms'] for t in tr]):6.2f}"
          f"               {np.median([abs(t['cosx']) for t in tr]):.2f}")
print("  split:")
for lab, tr in ([("PDHD run " + r, [t for t in HD if t["key"].startswith(r)]) for r in ("028084", "029107")]
                + [("PDHD APA %d" % k, [t for t in HD if t["apa"] == k]) for k in (1, 2, 3)]
                + [("PDVD bottom anode %d" % k, [t for t in BOT if t["apa"] == k]) for k in (0, 1, 2, 3)]
                + [("PDVD top anode %d" % k, [t for t in TOPV if t["apa"] == k]) for k in (4, 5, 6, 7)]
                + [("PDVD bottom run " + r, [t for t in BOT if t["key"].startswith(r)]) for r in ("039252", "039253", "039349")]
                + [("PDVD top run " + r, [t for t in TOPV if t["key"].startswith(r)]) for r in ("039252", "039253", "039349")]):
    if len(tr) < 3:
        print(f"    {lab:26s} n {len(tr):3d}"); continue
    p = bmed([t["plat"] for t in tr]); c = bmed([t["pc"] for t in tr])
    print(f"    {lab:26s} n {len(tr):3d}  plateau {fm(p)}   x W meas/pred {fm(c)}")
print("  ratios (median/median; bootstrap [p16,p84] and 95 %):")
RAT = {}
for lab, u, w in (("bottom / PDHD", BOT, HD), ("top / PDHD", TOPV, HD), ("top / bottom", TOPV, BOT)):
    for est in ("plat", "pc"):
        r = bratio([t[est] for t in u], [t[est] for t in w])
        RAT[(lab, est)] = r
        print(f"    {lab:14s} {'plateau' if est == 'plat' else 'plateau x W meas/pred':24s} {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]  95 % [{r[3]:.3f},{r[4]:.3f}]")

print("\n  matched drift TIME (attachment scales with time; PDHD v 1.576, PDVD 1.48073 mm/us):")
TE = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
for i in range(len(TE) - 1):
    s = lambda tr: [t["plat"] for t in tr if TE[i] <= t["tms"] < TE[i + 1]]
    b, h, tp = s(BOT), s(HD), s(TOPV)
    row = f"    {TE[i]:.1f}-{TE[i + 1]:.1f} ms: bottom {np.median(b) if len(b) else np.nan:.3f} ({len(b):2d})  top {np.median(tp) if len(tp) else np.nan:.3f} ({len(tp):3d})  PDHD {np.median(h) if len(h) else np.nan:.3f} ({len(h):2d})"
    if len(b) >= 3 and len(h) >= 3:
        row += f"  bottom/PDHD {np.median(b) / np.median(h):.3f}"
    print(row)
tb, th = np.median([t["tms"] for t in BOT]), np.median([t["tms"] for t in HD])
rbh = RAT[("bottom / PDHD", "plat")][0]
tau = (th - tb) / np.log(rbh) if rbh > 1 and th > tb else np.nan
print(f"  median drift time bottom {tb:.2f} ms, PDHD {th:.2f} ms: an uncorrected lifetime common to both closes bottom/PDHD {rbh:.3f} "
      f"at tau = {tau:.1f} ms (illustration only; tau is not measured and need not be common)")
for T in (10.0, 30.0, 100.0):
    print(f"    tau {T:5.0f} ms: exp(-(t_PDHD - t_bottom)/tau) = {np.exp(-(th - tb) / T):.3f}")

# --- the doc 29 toy, copied (d29_space_charge.py:35-63), self-checked against space_charge.txt
TA, TB_, TRHO, TDEDX = 0.93, 0.212, 1.3954, 2.4


def R_toy(dedx, E):
    xi = TB_ * dedx / (TRHO * E)
    return np.log(TA + xi) / xi


def vW(E, T=87.5):
    P1, P2, P3, P4, P5, P6, T0 = -0.04640, 0.01712, 1.88125, 0.99408, 0.01172, 4.20214, 105.749
    return (P1 * (T - T0) + 1) * (P3 * E * np.log(1 + P4 / E) + P5 * E ** P6) + P2 * (T - T0)


def model(drift, c2, delta, E0, L, pitch=True):
    x = np.clip(np.asarray(drift, float), 0, L)
    E = E0 * (1 - delta + 3 * delta * (x / L) ** 2)
    q = R_toy(TDEDX, E) / R_toy(TDEDX, E0)
    if not pitch:
        return q
    v = vW(E) / vW(E0)
    return q / np.sqrt(np.asarray(c2) / v ** 2 + (1 - np.asarray(c2)))


chk = open(os.path.join(HERE, "space_charge.txt")).read()
mline = re.search(r"pdvd delta 0\.10 cos\^2 0\.72: 0:([\d.]+) .* 340:([\d.]+)", chk)
mine = (model(0.0, 0.72, 0.10, 0.45, 340.0), model(340.0, 0.72, 0.10, 0.45, 340.0))
ok = mline and abs(float(mline.group(1)) - mine[0]) < 6e-4 and abs(float(mline.group(2)) - mine[1]) < 6e-4
print(f"\n  toy copy self-check vs space_charge.txt (pdvd delta 0.10 cos^2 0.72 at 0 / 340 cm): {mine[0]:.3f} / {mine[1]:.3f} -> {'OK' if ok else 'MISMATCH'}")
if not ok:
    sys.exit("toy copy does not reproduce d29_space_charge.py")
DETP = {"pdvd": (0.45, 340.0), "pdhd": (0.4959, 350.0)}
print("  angle x space charge: median toy factor per unit (recombination + drift-velocity pitch, A = 1) at each track's own drift and angle")
TOYF = {}
for nm, tr, det in (("PDVD bottom (BDE)", BOT, "pdvd"), ("PDVD top (TDE)", TOPV, "pdvd"), ("PDHD APA1-3", HD, "pdhd")):
    E0, L = DETP[det]
    row = f"    {nm:22s}"
    for dl in (0.07, 0.11):
        f = model([t["drift"] for t in tr], [t["cosx"] ** 2 for t in tr], dl, E0, L)
        TOYF[(nm, dl)] = float(np.median(f))
        row += f"  delta {dl:.2f}: {np.median(f):.3f}"
    print(row)
for dl in (0.07, 0.11):
    print(f"    delta {dl:.2f}: toy bottom/PDHD {TOYF[('PDVD bottom (BDE)', dl)] / TOYF[('PDHD APA1-3', dl)]:.3f}, "
          f"top/bottom {TOYF[('PDVD top (TDE)', dl)] / TOYF[('PDVD bottom (BDE)', dl)]:.3f}")

# --- PDVD field vs drift speed (BNL parameterisation, 87.68 K rows of pdvd/stm/pdvd_transport.tsv)
rows = [l.split("\t") for l in open(IMG + "/pdvd/stm/pdvd_transport.tsv") if l[:1].isdigit()]
r87 = sorted([(float(r[2]), float(r[1])) for r in rows if abs(float(r[0]) - 87.68) < 1e-6])   # (E, v)
(e1, v1), (e2, v2) = r87[0], r87[-1]
vE = v1 * (F_VD / e1) ** (np.log(v2 / v1) / np.log(e2 / e1))
rr_ = V_VD / vE
c2b = np.median([t["cosx"] ** 2 for t in BOT])
fpitch = 1 / np.sqrt(c2b * rr_ ** 2 + 1 - c2b)
print(f"\n  PDVD field provenance: the chain drifts at {V_VD} mm/us; at 87.68 K that is {e1:.4f} kV/cm, while the tables use {F_VD}.")
print(f"    (a) field 0.45 true -> true v {vE:.4f} mm/us, chain v0/v {rr_:.4f}: along-drift dQ/dx reads x{fpitch:.4f} on bottom's median cos^2 {c2b:.2f}")
print(f"    (b) drift speed true -> field {e1:.4f}: expectation too high, plateau reads x{1 / sh:.4f}")

print("\n  SYSTEMATIC BUDGET for bottom/PDHD (each term: how far it can move the ratio; sign where known)")
rc = RAT[("bottom / PDHD", "pc")][0]
hr = [np.median([t["plat"] for t in HD if t["key"].startswith(r)]) for r in ("028084", "029107")]
print(f"    observed bottom/PDHD plateau                      {fm(RAT[('bottom / PDHD', 'plat')])}  95 % [{RAT[('bottom / PDHD', 'plat')][3]:.3f},{RAT[('bottom / PDHD', 'plat')][4]:.3f}]")
print(f"    1 PDHD run-to-run (028084 / 029107 medians)       {hr[0]:.3f} / {hr[1]:.3f}: +-{abs(hr[0] - hr[1]) / 2 / np.mean(hr) * 100:.1f} %")
print(f"    2 PDVD field vs drift speed                        (a) {100 * (fpitch - 1):+.1f} % on bottom  /  (b) {100 * (1 / sh - 1):+.1f} %")
print(f"    3 space charge x angle (toy, delta 0.07 / 0.11)    {100 * (TOYF[('PDVD bottom (BDE)', 0.07)] / TOYF[('PDHD APA1-3', 0.07)] - 1):+.1f} % / "
      f"{100 * (TOYF[('PDVD bottom (BDE)', 0.11)] / TOYF[('PDHD APA1-3', 0.11)] - 1):+.1f} %")
print(f"    4 the fit's W bias (plateau x W meas/pred)         bottom/PDHD {rc:.3f} instead of {rbh:.3f}")
print(f"    5 uncorrected attachment                           closes the gap at tau {tau:.1f} ms; tau 30 ms moves it {100 * (np.exp(-(th - tb) / 30.0) - 1):+.1f} %")
print("    6 charge recovery (doc pdhd/17 sec 4, pooled PDVD) 0.941 PDHD / 0.921 PDVD: 2 %, not split by volume")

# ================================================================= C. what SP assumes
print("\n=== C. what signal processing assumes, per unit")


def coldelec(t, gain, shaping=2.2):   # util/src/Response.cxx:311-335; t, shaping in us; gain in mV/fC; returns mV/fC
    t = np.asarray(t, float); r = t / shaping; g = gain * 10 * 1.012
    e, c, s = np.exp, np.cos, np.sin
    v = (4.31054 * e(-2.94809 * r) * g - 2.6202 * e(-2.82833 * r) * c(1.19361 * r) * g
         - 2.6202 * e(-2.82833 * r) * c(1.19361 * r) * c(2.38722 * r) * g
         + 0.464924 * e(-2.40318 * r) * c(2.5928 * r) * g + 0.464924 * e(-2.40318 * r) * c(2.5928 * r) * c(5.18561 * r) * g
         + 0.762456 * e(-2.82833 * r) * s(1.19361 * r) * g - 0.762456 * e(-2.82833 * r) * c(2.38722 * r) * s(1.19361 * r) * g
         + 0.762456 * e(-2.82833 * r) * c(1.19361 * r) * s(2.38722 * r) * g - 2.620200 * e(-2.82833 * r) * s(1.19361 * r) * s(2.38722 * r) * g
         - 0.327684 * e(-2.40318 * r) * s(2.5928 * r) * g + 0.327684 * e(-2.40318 * r) * c(5.18561 * r) * s(2.5928 * r) * g
         - 0.327684 * e(-2.40318 * r) * c(2.5928 * r) * s(5.18561 * r) * g + 0.464924 * e(-2.40318 * r) * s(2.5928 * r) * s(5.18561 * r) * g)
    return np.where((t > 0) & (t < 10.0), v, 0.0)


trapz = getattr(np, "trapezoid", None) or np.trapz
tt = np.arange(0, 20, 0.001)
ce78, ce14 = coldelec(tt, 7.8), coldelec(tt, 14.0)
TJ = json.load(__import__("bz2").open(TOPDIR + "/wire-cell-data/dunevd-coldbox-elecresp-top-psnorm_400.json.bz2"))
tj_t = np.asarray(TJ["times"], float) / 1000.0            # ns -> us
tj_a = np.asarray(TJ["amplitudes"], float) / MVFC_WCT      # WCT -> mV/fC
ADCMV_BOT = ((1 << 14) - 1) / 1400.0                       # pdhd & protodunevd params adc 0.2-1.6 V, 14 bit
ADCMV_TOP = ((1 << 14) - 1) / 2000.0                       # protodunevd/sp.jsonnet:96-98 (2.0 V for ident >= 4)
RESP = {
    "PDHD / PDVD bottom at 7.8": (ce78.max(), trapz(ce78, tt), 1.0, ADCMV_BOT),
    "PDHD at 14":                (ce14.max(), trapz(ce14, tt), 1.0, ADCMV_BOT),
    "PDVD top (JSON x 1.36)":    (tj_a.max(), trapz(tj_a, tj_t), 1.36, ADCMV_TOP),
}
print("  electronics response   peak mV/fC  area mV.us/fC  area/peak us  postgain  ADC/mV   peak ADC/fC  area ADC.tick/e   e per ADC.tick if SP normalises by area")
PRED = {}
for k, (pk, ar, pg, am) in RESP.items():
    adc_tick_per_e = ar * pg * am / E_PER_FC / 0.5
    PRED[k] = 1.0 / adc_tick_per_e
    print(f"  {k:24s} {pk:8.3f}   {ar:10.3f}    {ar / pk:8.2f}     {pg:5.2f}   {am:6.3f}   {pk * pg * am:8.1f}      {adc_tick_per_e:.5f}        {PRED[k]:6.2f}")
t_ = RESP["PDVD top (JSON x 1.36)"]; b_ = RESP["PDHD / PDVD bottom at 7.8"]
print(f"  assumed top/bottom: by peak {t_[0] * t_[2] * t_[3] / (b_[0] * b_[3]):.3f}, by area {t_[1] * t_[2] * t_[3] / (b_[1] * b_[3]):.3f}"
      f" (ADC tick per fC; in electrons per ADC tick {PRED['PDVD top (JSON x 1.36)'] / PRED['PDHD / PDVD bottom at 7.8']:.3f})"
      " -- NEITHER is SP's effective charge scale: the deconvolved charge depends on the response in the pass band of the"
      " software filters, and the two response shapes differ (top area/peak above). The frame check below measures it.")
print("  PDVD BDE and PDHD share ColdElec, 2.2 us shaping, 14 bit over 0.2-1.6 V, postgain 1.0; PDHD's gain follows the"
      " run's hardware (run_nf_sp_evt.sh -g / input-root name); PDVD bottom is 7.8 for every run (META: all runs 7.8).")
print("  PDHD META.json (input_data_7p8_new_coh_grouping): run028084 14.0 mV/fC, run029107 7.8 (inferred from orig-frame ADC RMS);"
      " PDVD input_data/META.json: 039252/039253/039349 7.8.")
print(f"  postgain history (toolkit 546b2dac, 2026-05-03): the shared field-response file's W normalisation was fixed (x1.117) and"
      f" both postgains de-compensated, top 1.52 -> 1.36, bottom 1.1365 -> 1.0.  Top relative to bottom, SP's assumed response"
      f" moved x{(1.36 / 1.0) / (1.52 / 1.1365):.4f}, i.e. top charge relative to bottom x{(1.52 / 1.1365) / (1.36 / 1.0):.4f};"
      f" a mis-set 7.8 vs 14 mV/fC is x{14 / 7.8:.2f}")

FRAMES = [
    ("PDHD 028084 apa1", ["pdhd/work/028084_0_d09/protodunehd-sp-dnnroi-frames-anode1.tar.bz2",
                          "pdhd/work/028084_1_d09/protodunehd-sp-dnnroi-frames-anode1.tar.bz2"]),
    ("PDHD 029107 apa1", ["pdhd/work/029107_0_d09ctl/protodunehd-sp-dnnroi-frames-anode1.tar.bz2",
                          "pdhd/work/029107_1_d09ctl/protodunehd-sp-dnnroi-frames-anode1.tar.bz2"]),
    ("PDVD bottom anode0", ["pdvd/work/039252_0_d27fresh/protodune-sp-dnnroi-frames-anode0.tar.bz2",
                            "pdvd/work/039253_0_d27fresh/protodune-sp-dnnroi-frames-anode0.tar.bz2"]),
    ("PDVD top anode4", ["pdvd/work/039252_0_d27fresh/protodune-sp-dnnroi-frames-anode4.tar.bz2",
                         "pdvd/work/039253_0_d27fresh/protodune-sp-dnnroi-frames-anode4.tar.bz2"]),
]
DIL = (0, 20, 60)


def frame_ratio(rel):
    arr = {}
    with tarfile.open(IMG + "/" + rel, "r:bz2") as tf:
        for m in tf:
            for pre in ("frame_gauss", "frame_raw", "channels_gauss", "channels_raw", "tickinfo_gauss", "tickinfo_raw"):
                if m.name.startswith(pre):
                    arr[pre] = np.load(io.BytesIO(tf.extractfile(m).read()))
    G, Rw = arr["frame_gauss"].astype(float), arr["frame_raw"].astype(float)
    assert np.array_equal(arr["channels_gauss"], arr["channels_raw"]) and np.array_equal(arr["tickinfo_gauss"], arr["tickinfo_raw"])
    assert G.shape == Rw.shape
    out = {}
    for d in DIL:
        roi = G > 0
        if d:
            roi = ndimage.binary_dilation(roi, structure=np.ones((1, 2 * d + 1), bool))
        sg = (G * roi).sum(axis=1); sr = (Rw * roi).sum(axis=1); sa = (np.abs(Rw) * roi).sum(axis=1)
        sel = (sg > 3e4) & (sr > 0.8 * sa)
        out[d] = (int(sel.sum()), float(sg[sel].sum()), float(sr[sel].sum()),
                  float(np.median(sg[sel] / sr[sel])) if sel.any() else np.nan)
    return rel, out


FR = {}
if not a.skip_frames:
    jobs = [rel for _, rels in FRAMES for rel in rels]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for rel, out in ex.map(frame_ratio, jobs):
            FR[rel] = out
    print("  FRAME CHECK: SP's effective electrons per ADC tick on collection signals")
    print("    selection: channel sum(gauss) > 3e4 e in the gauss ROI dilated by +-N ticks, and sum(raw) > 0.8 sum|raw| there (unipolar)")
    for nm, rels in FRAMES:
        for d in DIL:
            n = sum(FR[r][d][0] for r in rels); sg = sum(FR[r][d][1] for r in rels); sr = sum(FR[r][d][2] for r in rels)
            FR[(nm, d)] = sg / sr
            print(f"    {nm:20s} dilation +-{d:2d} ticks: channels {n:4d}  pooled e/(ADC tick) {sg / sr:6.2f}  per-event medians "
                  + " / ".join(f"{FR[r][d][3]:.2f}" for r in rels))
    for d in DIL:
        print(f"    dilation +-{d:2d}: 028084/029107 {FR[('PDHD 028084 apa1', d)] / FR[('PDHD 029107 apa1', d)]:.3f} (7.8/14 would be 0.557)"
              f"   top/bottom {FR[('PDVD top anode4', d)] / FR[('PDVD bottom anode0', d)]:.3f}"
              f"   PDVD bottom / PDHD 029107 {FR[('PDVD bottom anode0', d)] / FR[('PDHD 029107 apa1', d)]:.3f}")
    print(f"    area-normalisation prediction (collection FR integrating to one electron): 7.8 mV/fC {PRED['PDHD / PDVD bottom at 7.8']:.2f}, "
          f"14 mV/fC {PRED['PDHD at 14']:.2f}, top {PRED['PDVD top (JSON x 1.36)']:.2f} e/(ADC tick)")
    print("    e/(ADC tick) top/bottom is SP's CONVERSION FACTOR, not a charge ratio: SP credits a top ADC tick with that fraction of a"
          " bottom tick's electrons because it expects 1/that more ADC per electron from the top.  If everything after SP treats"
          " the volumes alike, the measured dQ/dx top/bottom then implies the top's real ADC per electron relative to the bottom"
          " (collection plane, in band, gain and shape together):")
    for d in (20, 60):
        fr_tb = FR[("PDVD top anode4", d)] / FR[("PDVD bottom anode0", d)]
        print(f"      dilation +-{d:2d}: SP expects x{1 / fr_tb:.3f}; dQ/dx says x{RAT[('top / bottom', 'plat')][0] / fr_tb:.3f} (plateau) / "
              f"x{RAT[('top / bottom', 'pc')][0] / fr_tb:.3f} (x W meas/pred)")

# ================================================================= D. recombination in the Michel conversion
print("\n=== D. the field in the Michel charge -> energy conversion")


def R_box(dedx, E, rho=RHO):
    xi = B_BOX * dedx / (rho * E)
    return np.log(A_BOX + xi) / xi


print("  Modified Box survival R (A 0.93, B 0.212, rho 1.38)")
print("  dE/dx MeV/cm    E 0.4500   E 0.4393   E 0.4959   R(0.4959)/R(0.45)")
for x in (2.1, 3.0, 5.0, 10.0):
    print(f"  {x:10.1f}      {R_box(x, F_VD):.4f}     {R_box(x, F_VD_V):.4f}     {R_box(x, F_HD):.4f}     {R_box(x, F_HD) / R_box(x, F_VD):.4f}")
cfg = {}
for det, fn in (("pdhd", "pdhd/pr.jsonnet"), ("pdvd", "protodunevd/pr.jsonnet")):
    txt = open(f"{TK}/cfg/pgrapher/experiment/{fn}").read()
    mb = re.search(r"name: '%s_box_recomb',\s*data: \{ A: ([\d.]+), B: ([\d.]+), Efield: ([\d.]+), rho: ([\d.]+)" % det, txt)
    ms = re.search(r"name: '%s_stm_recomb',\s*data: \{ A: ([\d.]+), k: ([\d.]+), p: ([\d.]+), C: ([\d.]+),\s*pivot: ([\d.]+)" % det, txt)
    cfg[det] = dict(E=float(mb.group(3)), rho=float(mb.group(4)), A=float(ms.group(1)), k=float(ms.group(2)), p=float(ms.group(3)),
                    C=float(ms.group(4)), pivot=float(ms.group(5)))
comp = [n for n in json.load(open(IMG + "/pdhd/work/029107_17_h28cfgoff/.wct-pr_h28cfgoff.json"))
        if isinstance(n, dict) and n.get("name") in ("pdhd_stm_recomb", "pdhd_box_recomb")]
cc = {n["name"]: n["data"] for n in comp}
print(f"  compiled PDHD config (h28cfgoff): box Efield {cc['pdhd_box_recomb']['Efield']}, stm k {cc['pdhd_stm_recomb']['k']}, C {cc['pdhd_stm_recomb']['C']}")
K = {}
for det in ("pdhd", "pdvd"):
    c = cfg[det]
    kre = B_BOX * c["pivot"] / (c["rho"] * c["E"])
    u = c["k"] * (2.1 / c["pivot"]) ** c["p"]
    R21 = np.log(c["A"] + u) / u
    K[det] = WION / (c["C"] * R21)
    c["R21"] = R21
    print(f"  {det}: jsonnet E {c['E']}  k {c['k']:.10f}  k recomputed B*pivot/(rho E) {kre:.10f} (diff {abs(kre - c['k']):.1e})  "
          f"C {c['C']}  R(2.1) {R21:.4f}  MeV/electron = Wi/(C R) {K[det]:.5e}")
m96 = re.search(r"`K` = ([\d.e+-]+) MeV/electron", open(IMG + "/pdvd/docs/nf_sp_img_clus/96_region-scope-and-prediction-bias.md").read())
print(f"  doc pdvd/96 measured K on p96vprod candidates: {m96.group(1)} (this script {K['pdvd']:.5e})")
fr = cfg["pdhd"]["R21"] / cfg["pdvd"]["R21"]; cr = cfg["pdhd"]["C"] / cfg["pdvd"]["C"]
print(f"  PDVD / PDHD MeV per electron = {K['pdvd'] / K['pdhd']:.4f} = field (R ratio) {fr:.4f} x C ratio {cr:.4f}")
for x in (3.0, 5.0):
    miss = (R_box(x, F_HD) / R_box(x, F_VD)) / (R_box(2.1, F_HD) / R_box(2.1, F_VD))
    print(f"  if Michel charge were deposited at {x:.0f} MeV/cm, the field correction the 2.1 MeV/cm conversion misses: {100 * (miss - 1):.1f} % "
          f"(PDHD energy reads that much HIGH relative to PDVD)")

MI = None
if not a.skip_michel:
    os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json")
    sys.path.insert(0, os.path.join(HERE, "..", "d26"))
    import d26_michel_energy as M
    if "stop_x" not in M.BR:
        M.BR.append("stop_x")
    DV = M.read("pdvd", "p96vprod")
    its, miss = M.BM.items("pdvd", "p96vprod", "all")
    judged = {x[0]: x for x in its}
    hm = [d for k, d in DV.items() if k in judged and int(d["is_stm"]) == 1 and judged[k][1] in M.BM.STOP and judged[k][2] in M.BM.MICHEL_KINDS]
    allsel = [d for d in DV.values() if int(d["is_stm"]) == 1]
    print("\n  PDVD Michel energy by drift volume, p96vprod, split on stop_x (x>0 top/TDE, x<0 bottom/BDE); median MeV [p16,p84]")
    MI = {}
    for lab, pop in (("hand Michel (doc 26 sec 3)", hm), ("all chain STM+Michel", allsel)):
        for est in ("michel_ke_q2d_region", "michel_ke_q2d_ctl", "michel_ke_best"):
            tp = [float(d[est]) for d in pop if float(d["stop_x"]) > 0]
            bt = [float(d[est]) for d in pop if float(d["stop_x"]) < 0]
            mt, mb_ = bmed(tp, 41), bmed(bt, 43)
            r = bratio(tp, bt, 45) if np.median(bt) > 0 else (np.nan,) * 5
            MI[(lab, est)] = (tp, bt, mt, mb_, r)
            print(f"    {lab:26s} {est:21s} top n {len(tp):3d} {fm(mt)}  bottom n {len(bt):3d} {fm(mb_)}  top/bottom {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] 95 % [{r[3]:.3f},{r[4]:.3f}]")
    print("    gain reading predicts top/bottom ~0.89 on region and control alike (both are charge through the same SP)")

# ================================================================= figure
fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
# (a) per-unit plateaus
labs, P, C_ = [], [], []
units = ([("PDHD run 028084", [t for t in HD if t["key"].startswith("028084")]), ("PDHD run 029107", [t for t in HD if t["key"].startswith("029107")])]
         + [("PDHD APA%d" % k, [t for t in HD if t["apa"] == k]) for k in (1, 2, 3)] + [("PDHD all", HD)]
         + [("VD bottom a%d" % k, [t for t in BOT if t["apa"] == k]) for k in (0, 1, 2, 3)] + [("VD bottom", BOT)]
         + [("VD top a%d" % k, [t for t in TOPV if t["apa"] == k]) for k in (4, 5, 6, 7)] + [("VD top", TOPV)])
for i, (lab, tr) in enumerate(units):
    p = bmed([t["plat"] for t in tr]); c = bmed([t["pc"] for t in tr])
    col = "C0" if lab.startswith("PDHD") else ("C2" if "bottom" in lab else "C3")
    ax[0, 0].errorbar(i - 0.12, p[0], yerr=[[p[0] - p[1]], [p[2] - p[0]]], fmt="o", color=col, ms=6 if "a" in lab.split()[-1][:1] or "APA" in lab or "run" in lab else 9)
    ax[0, 0].errorbar(i + 0.12, c[0], yerr=[[c[0] - c[1]], [c[2] - c[0]]], fmt="s", mfc="none", color=col, ms=6)
    labs.append(f"{lab}\n(n {len(tr)})")
ax[0, 0].set_xticks(range(len(units))); ax[0, 0].set_xticklabels(labs, rotation=70, fontsize=7)
ax[0, 0].axhline(1, color="k", lw=0.6)
ax[0, 0].plot([], [], "ko", label="plateau (no free scale)"); ax[0, 0].plot([], [], "ks", mfc="none", label="plateau x W measured/predicted")
ax[0, 0].legend(fontsize=8, loc="upper right"); ax[0, 0].set_ylabel("median, [p16, p84]")
ax[0, 0].set_title("(a) per-track plateau by unit: blue PDHD, green PDVD bottom (BDE), red PDVD top (TDE)", fontsize=9)
# (b) plateau vs drift time
for tr, col, lab in ((HD, "C0", "PDHD APA1-3"), (BOT, "C2", "PDVD bottom (BDE)"), (TOPV, "C3", "PDVD top (TDE)")):
    ax[0, 1].scatter([t["tms"] for t in tr], [t["plat"] for t in tr], s=9, color=col, alpha=0.35)
    xs, ys = [], []
    for i in range(len(TE) - 1):
        s = [t["plat"] for t in tr if TE[i] <= t["tms"] < TE[i + 1]]
        if len(s) >= 3:
            xs.append((TE[i] + TE[i + 1]) / 2); ys.append(np.median(s))
    ax[0, 1].plot(xs, ys, "-o", color=col, label=lab + " (bin medians, n>=3)")
ax[0, 1].axhline(1, color="k", lw=0.6); ax[0, 1].set_ylim(0.5, 1.5)
ax[0, 1].set_xlabel("plateau drift time (ms) = drift distance / chain drift speed"); ax[0, 1].set_ylabel("per-track plateau")
ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("(b) plateau against drift time (an uncorrected lifetime acts on time)", fontsize=9)
# (c) recombination field ratio
xx = np.linspace(1.5, 12, 200)
ax[1, 0].plot(xx, R_box(xx, F_HD) / R_box(xx, F_VD), "k-", label="R(0.4959) / R(0.45)")
ax[1, 0].plot(xx, R_box(xx, F_VD) / R_box(xx, F_VD_V), "k--", label="R(0.45) / R(0.4393)")
ax[1, 0].axvline(2.1, color="C1", lw=1, label="michel_unfit_dedx 2.1 (both chains)")
ax[1, 0].axhline(K["pdvd"] / K["pdhd"], color="C4", lw=1, ls=":", label=f"PDVD/PDHD MeV per electron {K['pdvd'] / K['pdhd']:.3f} (field x C)")
ax[1, 0].set_xlabel("dE/dx (MeV/cm)"); ax[1, 0].set_ylabel("ratio"); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("(d) recombination: field ratio against dE/dx, and the two chains' constants", fontsize=9)
# (c) the frame check
if FR:
    names = [nm for nm, _ in FRAMES]
    pred = [PRED["PDHD at 14"], PRED["PDHD / PDVD bottom at 7.8"], PRED["PDHD / PDVD bottom at 7.8"], PRED["PDVD top (JSON x 1.36)"]]
    for j, d in enumerate(DIL):
        ax[0, 2].bar(np.arange(4) + (j - 1) * 0.25, [FR[(nm, d)] for nm in names], width=0.25, color=f"C{j + 4}",
                     label=f"measured, gauss ROI dilated +-{d} ticks")
    for i, p in enumerate(pred):
        ax[0, 2].plot([i - 0.4, i + 0.4], [p, p], "k-", lw=2, zorder=5, label="if SP normalised by the response area" if i == 0 else None)
    ax[0, 2].set_xticks(range(4))
    ax[0, 2].set_xticklabels(["PDHD 028084\nMETA 14 mV/fC", "PDHD 029107\nMETA 7.8 mV/fC", "PDVD bottom\nBDE 7.8 mV/fC", "PDVD top\nTDE JSON x 1.36"], fontsize=8)
    ax[0, 2].set_ylabel("SP electrons per raw ADC tick, collection signals")
    ax[0, 2].legend(fontsize=8, loc="upper right"); ax[0, 2].set_ylim(0, 19.5)
    ax[0, 2].set_title(f"(c) what SP applies: 028084/029107 {FR[('PDHD 028084 apa1', 60)] / FR[('PDHD 029107 apa1', 60)]:.3f} (7.8/14 = 0.557), "
                       f"top/bottom {FR[('PDVD top anode4', 60)] / FR[('PDVD bottom anode0', 60)]:.3f}", fontsize=9)
# (f) summary
rb, rp, rt = RAT[("bottom / PDHD", "plat")], RAT[("bottom / PDHD", "pc")], RAT[("top / bottom", "plat")]
lines = ["bottom / PDHD",
         f"  plateau                {rb[0]:.3f} [{rb[1]:.3f},{rb[2]:.3f}]",
         f"  plateau x W meas/pred  {rp[0]:.3f} [{rp[1]:.3f},{rp[2]:.3f}]",
         "top / bottom",
         f"  plateau                {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}]",
         f"  plateau x W meas/pred  {RAT[('top / bottom', 'pc')][0]:.3f} [{RAT[('top / bottom', 'pc')][1]:.3f},{RAT[('top / bottom', 'pc')][2]:.3f}]",
         "",
         "what can move bottom / PDHD",
         f"  PDHD run to run         +-{abs(hr[0] - hr[1]) / 2 / np.mean(hr) * 100:.1f} %",
         f"  PDVD field vs speed     {100 * (fpitch - 1):+.1f} % / {100 * (1 / sh - 1):+.1f} %",
         f"  space charge x angle    {100 * (TOYF[('PDVD bottom (BDE)', 0.07)] / TOYF[('PDHD APA1-3', 0.07)] - 1):+.1f} % (delta 0.07)",
         f"  fit W bias              {rb[0]:.3f} -> {rp[0]:.3f}",
         f"  lifetime                closes it at tau {tau:.0f} ms",
         "",
         "Michel conversion, PDVD / PDHD MeV per electron",
         f"  {K['pdvd'] / K['pdhd']:.4f} = field {fr:.4f} x C {cr:.4f}"]
ax[1, 2].axis("off")
ax[1, 2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10, transform=ax[1, 2].transAxes)
ax[1, 2].set_title("(f) summary (every number from gain_recomb.txt)", fontsize=9)
# (d) Michel energy by volume
if MI:
    tp, bt, mt, mb_, r = MI[("hand Michel (doc 26 sec 3)", "michel_ke_q2d_region")]
    for v, col, lab, m in ((tp, "C3", "top (TDE)", mt), (bt, "C2", "bottom (BDE)", mb_)):
        s = np.sort(v)
        ax[1, 1].step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=col, label=f"{lab} n {len(s)}, median {m[0]:.1f} [{m[1]:.1f},{m[2]:.1f}]")
    ax[1, 1].axvline(52.83, color="k", lw=0.6, ls=":")
    ax[1, 1].set_xlim(0, 90); ax[1, 1].set_xlabel("michel_ke_q2d_region (MeV)"); ax[1, 1].set_ylabel("cumulative fraction")
    ax[1, 1].legend(fontsize=8, loc="lower right")
    ax[1, 1].set_title(f"(e) PDVD hand-Michel region energy by volume: top/bottom median {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]", fontsize=9)
fig.tight_layout()
fig.savefig(a.fig, dpi=110)
print(f"\nfigure: {os.path.relpath(a.fig, HERE)}")

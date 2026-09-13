#!/usr/bin/env python3
"""doc pdhd/27 figures, from the JSON the analysis scripts write next to this file.

    python3 d27_figs.py [--figdir ../../figs]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"pdhd": "C0", "pdvd": "C3"}
ap = argparse.ArgumentParser(); ap.add_argument("--figdir", default=os.path.join(HERE, "..", "..", "figs")); a = ap.parse_args()
J = lambda n: json.load(open(os.path.join(HERE, n)))
WF, NC, GT, DD, DR = J("wrap_face.json"), J("neutral_ctl.json"), J("ghost_twin.json"), J("dqdx_drift.json"), J("dqdx_rr_vs_drift.json")

# ---- figure 1: the body control -------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
ax = axs[0, 0]
B = np.array(WF["percell_bins"])
sty = {"active face, channel has 1 active-face wire": ("C0", "-"), "active face, 2 active-face wires, recorded the first": ("C0", "--"),
       "other face (first-listed wire is on face 0)": ("C1", "-")}
for k, cnt in WF["percell"].items():
    c, ls = sty.get(k, ("0.5", ":"))
    ax.stairs(np.array(cnt) / max(1, sum(cnt)), B, color=c, ls=ls, lw=1.8, label=f"PDHD {k} (n {sum(cnt)})")
vv = np.sum([np.array(v) for v in WF["percell_pdvd"].values()], axis=0)
ax.stairs(vv / vv.sum(), B, color=COL["pdvd"], lw=1.8, label=f"PDVD U/V/W (n {int(vv.sum())})")
ax.axvline(10, color="0.4", lw=1); ax.set_xscale("log")
ax.set_xlabel("recorded d_stop_cm of role-1 (muon footprint) cells [cm]"); ax.set_ylabel("fraction of cells")
ax.set_title("(a) PDHD U/V distances are taken on the channel's first-listed wire", fontsize=9.5)
ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.3)

ax = axs[0, 1]
for det, lab in (("pdhd", "PDHD"), ("pdvd", "PDVD")):
    r = [x for x in NC[det] if x["cosv"] == x["cosv"]]
    cv = np.array([x["cosv"] for x in r]); ch = np.array([x["ctl"] for x in r]); cl = np.array([x["ctlW_clean"] for x in r])
    ax.scatter(cv, np.clip(ch, 0.3, 200), s=22, color=COL[det], alpha=0.75, label=f"{lab}: chain control michel_ke_q2d_ctl")
    ax.scatter(cv, np.clip(cl, 0.3, 200), s=22, facecolors="none", edgecolors=COL[det], alpha=0.75, label=f"{lab}: W only, stop-region cells removed")
ax.set_yscale("log"); ax.set_xlabel("|cos| of the muon at the control to the vertical\n(PDHD: y, the W wire direction; PDVD: x, the drift direction)")
ax.set_ylabel("MeV (floored at 0.3 for the log axis)"); ax.set_title("(b) the control against muon steepness, hand Michel items", fontsize=9.5)
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

ax = axs[1, 0]
b = np.linspace(-0.3, 0.6, 37)
for det, lab in (("pdhd", "PDHD"), ("pdvd", "PDVD")):
    v = [x["ctlW_clean"] / x["ctlW_clean_mu"] for x in NC[det] if x["ctlW_clean_mu"] > 0]
    ax.hist(np.clip(v, b[0], b[-1] - 1e-6), bins=b, histtype="step", lw=1.8, density=True, color=COL[det],
            label=f"{lab} (n {len(v)}, median {np.median(v):+.3f})")
ax.axvline(0, color="0.4", lw=1)
ax.set_xlabel("W-only control with stop-region cells removed, over its own muon prediction: (measured - fit) / fit")
ax.set_ylabel("normalised"); ax.set_title("(c) the floor per unit of muon charge, a detector-neutral control", fontsize=9.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 1]
g1 = [x for x in GT if x["face"] == 1 and x["corrected_ok"]]; g0 = [x for x in GT if x["face"] == 0 and x["corrected_ok"]]
for g, mk, lab in ((g0, "s", "APA0/2"), (g1, "o", "APA1/3")):
    ax.scatter([x["rec_ctl"] for x in g], [x["fix_ctl"] for x in g], marker=mk, s=26, color="C2", alpha=0.8, label=f"control, {lab} (n {len(g)})")
    ax.scatter([x["rec_stop"] for x in g], [x["fix_stop"] for x in g], marker=mk, s=26, color="C1", alpha=0.8, label=f"region, {lab}")
    if True:
        for x in g:
            ax.plot([x["rec_ctl"]] * 2, [x["fixlo_ctl"], x["fix_ctl"]], color="C2", lw=0.6, alpha=0.5)
            ax.plot([x["rec_stop"]] * 2, [x["fixlo_stop"], x["fix_stop"]], color="C1", lw=0.6, alpha=0.5)
lim = [0, 120]; ax.plot(lim, lim, "k:", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("as the chain recorded it [MeV]"); ax.set_ylabel("U/V distances on the muon's face [MeV] (bar: lower bound)")
ax.set_title("(d) U/V distances re-measured on the wire the muon crossed, every is_stm Michel", fontsize=9.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); f1 = os.path.join(a.figdir, "27_ctl_floor.png"); fig.savefig(f1, dpi=110); plt.close(fig)

# ---- figure 2: PDVD near-anode dQ/dx -------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
ax = axs[0]
for side, mk in (("x<0", "v"), ("x>0", "^")):
    r = [x for x in DD["pdvd"] if x["side"] == side]
    sc = ax.scatter([x["drift"] for x in r], [x["plat"] for x in r], c=[x["cosx"] for x in r], cmap="viridis", vmin=0, vmax=1, marker=mk, s=30,
                    label=f"PDVD {side} (n {len(r)})")
r = DD["pdhd"]; ax.scatter([x["drift"] for x in r], [x["plat"] for x in r], color="0.6", marker=".", s=30, label=f"PDHD APA0 strict (n {len(r)})")
plt.colorbar(sc, ax=ax, label="|cos| to the drift axis")
ax.axhline(1, color="0.4", lw=1); ax.set_ylim(0.6, 1.4)
ax.set_xlabel("drift distance = x_anode - |x| over the plateau points [cm]"); ax.set_ylabel("per-track plateau dQ/dx / expected (rr 40-60 cm)")
ax.set_title("(a) plateau against drift, coloured by steepness", fontsize=9.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axs[1]
for side, c in (("x<0", "C0"), ("x>0", "C3")):
    r = [x for x in DR["pdvd"] if x["side"] == side and "s_rr" in x]
    k = np.array([x["ddrr"] for x in r]); s = np.array([x["s_rr"] for x in r])
    ax.scatter(k, s, s=18, color=c, alpha=0.7, label=f"PDVD {side} (n {len(r)})")
    cf = np.linalg.lstsq(np.c_[np.ones(len(k)), k], s, rcond=None)[0]
    kk = np.linspace(-1, 1, 10); ax.plot(kk, cf[0] + cf[1] * kk, color=c, lw=1.8, label=f"{side}: alpha (rr shape) {cf[0]:+.3f}, beta (drift) {cf[1]:+.3f}")
ax.axhline(0, color="0.4", lw=1); ax.set_ylim(-1, 1)
ax.set_xlabel("d drift / d rr along the track (cm per cm)"); ax.set_ylabel("per-track slope of normalised dQ/dx in rr [per 100 cm]")
ax.set_title("(b) within-track: residual-range shape vs drift, rr 20-100 cm", fontsize=9.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axs[2]
E = np.array([0, 50, 100, 150, 200, 250, 360])
for side, c in (("x<0", "C0"), ("x>0", "C3")):
    r = [x for x in DR["pdvd"] if x["side"] == side and x["mpW"] == x["mpW"]]
    dr = np.array([x["drift"] for x in r]); pl = np.array([x["plat"] for x in r]); mw = np.array([x["mpW"] for x in r])
    xs, y0, y1 = [], [], []
    for lo, hi in zip(E[:-1], E[1:]):
        s = (dr >= lo) & (dr < hi)
        if s.sum() >= 4: xs.append((lo + hi) / 2); y0.append(np.median(pl[s])); y1.append(np.median(pl[s] * mw[s]))
    ax.plot(xs, y0, "o-", color=c, label=f"PDVD {side}: fitted plateau")
    ax.plot(xs, y1, "s--", color=c, mfc="none", label=f"PDVD {side}: plateau x (W measured / W fit prediction)")
ax.axhline(1, color="0.4", lw=1)
ax.set_xlabel("drift distance [cm] (bins of 50 cm, >= 4 tracks)"); ax.set_ylabel("median over tracks")
ax.set_title("(c) how much of the drift trend the fit's own W bias carries", fontsize=9.5); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); f2 = os.path.join(a.figdir, "27_pdvd_near_anode.png"); fig.savefig(f2, dpi=110); plt.close(fig)
print("figures:", os.path.abspath(f1), os.path.abspath(f2))

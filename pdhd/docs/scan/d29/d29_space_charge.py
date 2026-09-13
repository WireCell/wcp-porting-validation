#!/usr/bin/env python3
"""doc pdhd/29 -- can the space-charge effect explain the dQ/dx position dependences of docs pdhd/26 sec 4.2 and 27 sec 5?

    python3 d29_space_charge.py [--json ../d27/dqdx_drift.json] [--fig ../../figs/29_space_charge.png] > space_charge.txt

INPUT: the committed per-track table of doc pdhd/27 sec 5 (d27/dqdx_drift.json): hand stoppers the chain accepted on
its own Bragg reading, per-track plateau = median dQ/dx / expected over rr 40-60 cm (no free scale), drift distance of
the plateau point from its own anode (x_anode - |x|), |cos_x| to the drift axis.  PDVD = p96vprod, PDHD = h26q2dprod
(identical T_stm_michel_pts on h28prod: doc pdhd/28 changes no point row).

MODEL (1-D, stated as a toy): positive ions produced uniformly drift to the cathode, so their density grows linearly
with the distance x from the anode; with the anode-cathode voltage fixed
    E(x) = E0 * (1 - delta + 3 delta (x/L)^2)          -delta at the anode, +2 delta at the cathode, E = E0 at L/sqrt(3)
(ProtoDUNE-SP: about -10 % / +20 %, i.e. delta ~ 0.10).  Two couplings to the measured dQ/dx:
  recombination   Modified Box (ArgoNeuT alpha 0.93, beta 0.212, rho 1.3954), plateau dE/dx 2.4 MeV/cm
  drift velocity  the chain converts time to x with one fixed speed; where electrons really drift at v(E) the
                  reconstructed pitch along drift is scaled by v0/v, so dQ/dx scales by 1/sqrt(cos^2 (v0/v)^2 + sin^2).
                  v(E): Walkowiak (NIM A 449 (2000) 288) at 87.5 K, the form LArSoft uses below 0.619 kV/cm.
PDVD E0 0.45 kV/cm, L 340 cm per volume (anode |x| 339.91, cathode drift face |x| 3.0 -> ~337 cm active drift,
protodunevd/params.jsonnet; the 1 % difference is below the quoted precision); PDHD E0 0.4959 kV/cm, L 350 cm.
Nothing here is fitted to the chain; delta and a per-volume scale A are fitted to the per-track plateaus (L1 loss).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--json", default=os.path.join(HERE, "..", "d27", "dqdx_drift.json"))
ap.add_argument("--fig", default=os.path.join(HERE, "..", "..", "figs", "29_space_charge.png"))
a = ap.parse_args()

ALPHA, BETA, RHO, DEDX = 0.93, 0.212, 1.3954, 2.4
DET = {"pdvd": dict(E0=0.45, L=340.0), "pdhd": dict(E0=0.4959, L=350.0)}
VOL = {("pdvd", "x>0"): "PDVD top (x>0)", ("pdvd", "x<0"): "PDVD bottom (x<0)",
       ("pdhd", "x>0"): "PDHD x>0", ("pdhd", "x<0"): "PDHD x<0"}


def R(dedx, E):
    xi = BETA * dedx / (RHO * E)
    return np.log(ALPHA + xi) / xi


def vW(E, T=87.5):
    P1, P2, P3, P4, P5, P6, T0 = -0.04640, 0.01712, 1.88125, 0.99408, 0.01172, 4.20214, 105.749
    return (P1 * (T - T0) + 1) * (P3 * E * np.log(1 + P4 / E) + P5 * E ** P6) + P2 * (T - T0)


def slog(f, E, h=1e-4):
    return (np.log(f(E * (1 + h))) - np.log(f(E * (1 - h)))) / (2 * h)


def model(drift, c2, delta, E0, L, pitch=True):
    x = np.clip(np.asarray(drift, float), 0, L)
    E = E0 * (1 - delta + 3 * delta * (x / L) ** 2)
    q = R(DEDX, E) / R(DEDX, E0)
    if not pitch:
        return q
    v = vW(E) / vW(E0)
    return q / np.sqrt(np.asarray(c2) / v ** 2 + (1 - np.asarray(c2)))


D = json.load(open(a.json))
V = {(det, s): [t for t in D[det] if t["side"] == s and np.isfinite(t["plat"])] for det in DET for s in ("x>0", "x<0")}
arr = lambda tr, k: np.array([t[k] for t in tr], float)
EDGES = [0, 60, 120, 180, 240, 300, 360]

print("=== 1. sensitivities")
for E in (0.40, 0.45, 0.4959, 0.55):
    print(f"  E {E:.4f} kV/cm: v {vW(E):.3f} mm/us, dlnv/dlnE {slog(vW, E):.3f} | recombination dlnQ/dlnE at dE/dx "
          + ", ".join(f"{d}: {slog(lambda e: R(d, e), E):.3f}" for d in (2.1, 2.4, 5.0, 10.0, 20.0)))

print("\n=== 2. the toy, dQ/dx ratio vs drift distance from the anode (A = 1)")
xs = np.array([0, 60, 120, 180, 240, 300, 340])
for det in DET:
    for delta in (0.05, 0.10):
        for c2 in (0.0, 0.06, 0.72, 1.0):
            m = model(xs, c2, delta, **DET[det])
            print(f"  {det} delta {delta:.2f} cos^2 {c2:.2f}: " + " ".join(f"{x}:{v:.3f}" for x, v in zip(xs, m))
                  + f" | end-to-end {m[-1] - m[0]:+.3f}")
for delta in (0.05, 0.10):
    m = model(np.arange(0, 341, 60), 0.72, delta, **DET["pdvd"])
    print(f"  pdvd delta {delta:.2f} cos^2 0.72 local slope per 100 cm by 60 cm step: "
          + " ".join(f"{(m[i + 1] - m[i]) / 0.6:+.3f}" for i in range(len(m) - 1)))

print("\n=== 3. per volume: where the plateau sits, the drift trend by angle, and the fitted delta")
rng = np.random.default_rng(5)
dg = np.linspace(-0.3, 0.8, 111)


def fit_delta(tr, E0, L, pitch=True):
    y = arr(tr, "plat"); d = arr(tr, "drift"); c2 = arr(tr, "cosx") ** 2
    best = (np.inf, None, None)
    for dl in dg:
        m = model(d, c2, dl, E0, L, pitch); A = np.median(y / m); loss = np.abs(y - A * m).sum()
        if loss < best[0]:
            best = (loss, dl, A)
    return best[1], best[2]


FITS = {}
for key, tr in V.items():
    det, side = key; E0, L = DET[det]["E0"], DET[det]["L"]
    d = arr(tr, "drift"); y = arr(tr, "plat"); cx = np.abs(arr(tr, "cosx"))
    print(f"  == {VOL[key]} n {len(tr)} | |cos_x| p50 {np.median(cx):.2f} cos^2 p50 {np.median(cx ** 2):.2f} | plateau drift p10/p25/p50/p75/p90 "
          + "/".join(f"{np.percentile(d, q):.0f}" for q in (10, 25, 50, 75, 90)) + " cm")
    print("     binned median plateau (n): " + " ".join(
        f"{EDGES[i]}-{EDGES[i + 1]}:{np.median(y[(d >= EDGES[i]) & (d < EDGES[i + 1])]):.3f}({int(((d >= EDGES[i]) & (d < EDGES[i + 1])).sum())})"
        if ((d >= EDGES[i]) & (d < EDGES[i + 1])).sum() >= 3 else f"{EDGES[i]}-{EDGES[i + 1]}:-({int(((d >= EDGES[i]) & (d < EDGES[i + 1])).sum())})"
        for i in range(len(EDGES) - 1)))
    for lo, hi in ((0, 0.5), (0.5, 0.85), (0.85, 1.01)):
        s = [t for t in tr if lo <= abs(t["cosx"]) < hi]
        if len(s) < 8:
            print(f"     |cos_x| {lo:.2f}-{hi:.2f}: n {len(s)} (too few)"); continue
        ds, ys = arr(s, "drift") / 100, arr(s, "plat")
        p = np.polyfit(ds, ys, 1)[0]
        bs = [np.polyfit(ds[i], ys[i], 1)[0] for i in (rng.integers(0, len(s), len(s)) for _ in range(2000))]
        pred = np.polyfit(ds, model(ds * 100, arr(s, "cosx") ** 2, 0.10, E0, L), 1)[0]
        print(f"     |cos_x| {lo:.2f}-{hi:.2f}: n {len(s):3d} observed slope per 100 cm {p:+.3f} [{np.percentile(bs, 16):+.3f},{np.percentile(bs, 84):+.3f}]"
              f" | toy delta 0.10 predicts {pred:+.3f}")
    for nm, X in (("linear in drift", d / 100), ("pure (drift)^2", (d / 100) ** 2)):
        M = np.c_[np.ones(len(d)), X]; p_ = np.linalg.lstsq(M, y, rcond=None)[0]; r_ = y - M @ p_
        print(f"     shape, {nm:15s}: coefficient {p_[1]:+.4f}, rms residual {np.sqrt(np.mean(r_ ** 2)):.4f}, MAD {np.median(np.abs(r_ - np.median(r_))):.4f}")
    dl, A = fit_delta(tr, E0, L)
    bs = [fit_delta([tr[i] for i in rng.integers(0, len(tr), len(tr))], E0, L)[0] for _ in range(300)]
    dl0, _ = fit_delta(tr, E0, L, pitch=False)
    FITS[key] = (dl, A)
    print(f"     fitted delta {dl:+.3f} [{np.percentile(bs, 16):+.3f},{np.percentile(bs, 84):+.3f}] (A {A:.3f}) | recombination-only fit needs delta {dl0:+.3f}")

print("\n=== 4. PDVD top vs bottom: is the offset a position effect inside each half-volume?")
T, B = V[("pdvd", "x>0")], V[("pdvd", "x<0")]
print("  matched drift distance from each volume's own anode: median plateau top | bottom | top/bottom")
for i in range(len(EDGES) - 1):
    yt = [t["plat"] for t in T if EDGES[i] <= t["drift"] < EDGES[i + 1]]; yb = [t["plat"] for t in B if EDGES[i] <= t["drift"] < EDGES[i + 1]]
    mt = np.median(yt) if len(yt) >= 3 else np.nan; mb = np.median(yb) if len(yb) >= 3 else np.nan
    print(f"    {EDGES[i]:3d}-{EDGES[i + 1]:3d} cm: {mt:.3f} ({len(yt):3d}) | {mb:.3f} ({len(yb):3d}) | {mt / mb:.3f}")
for lo, hi in ((0.5, 0.85), (0.85, 1.01)):
    yt = [t["plat"] for t in T if lo <= abs(t["cosx"]) < hi and t["drift"] < 240]; yb = [t["plat"] for t in B if lo <= abs(t["cosx"]) < hi and t["drift"] < 240]
    print(f"    matched angle |cos_x| {lo}-{hi}, drift 0-240 cm: top {np.median(yt):.3f} ({len(yt)}) | bottom {np.median(yb):.3f} ({len(yb)})")
print("  plateau by anode: top " + ", ".join(f"{k}: {np.median([t['plat'] for t in T if t['apa'] == k]):.3f} ({sum(1 for t in T if t['apa'] == k)})" for k in (4, 5, 6, 7))
      + " | bottom " + ", ".join(f"{k}: {np.median([t['plat'] for t in B if t['apa'] == k]):.3f} ({sum(1 for t in B if t['apa'] == k)})" for k in (0, 1, 2, 3)))

Ag = np.linspace(0.7, 1.3, 121); dG = np.linspace(-0.2, 0.8, 101)
E0, L = DET["pdvd"]["E0"], DET["pdvd"]["L"]


def comp(tT, tB):
    def prep(tr):
        y = arr(tr, "plat"); M = np.array([model(arr(tr, "drift"), arr(tr, "cosx") ** 2, dl, E0, L) for dl in dG])
        return np.abs(y[None, None, :] - Ag[:, None, None] * M[None, :, :]).sum(-1)   # [A, delta]
    LT, LB = prep(tT), prep(tB); o = {}
    L1 = LT + LB; i = np.unravel_index(np.argmin(L1), L1.shape); o["M1"] = (L1[i], Ag[i[0]], dG[i[1]])
    b3 = LT[:, :, None] + LB[:, None, :]; i = np.unravel_index(np.argmin(b3), b3.shape); o["M3"] = (b3[i], Ag[i[0]], dG[i[1]], dG[i[2]])
    m2 = LT.min(0) + LB.min(0); j = int(np.argmin(m2)); o["M2"] = (m2[j], Ag[np.argmin(LT[:, j])], Ag[np.argmin(LB[:, j])], dG[j])
    iT = np.unravel_index(np.argmin(LT), LT.shape); iB = np.unravel_index(np.argmin(LB), LB.shape)
    o["M4"] = (LT[iT] + LB[iB], Ag[iT[0]], dG[iT[1]], Ag[iB[0]], dG[iB[1]])
    return o


o = comp(T, B)
print(f"\n  model comparison over both PDVD volumes (n {len(T) + len(B)}), L1 loss = sum |plateau - A x toy| (lower is better)")
print(f"    M1 common A, common delta            : loss {o['M1'][0]:.2f}  A {o['M1'][1]:.3f} delta {o['M1'][2]:+.3f}")
print(f"    M3 common A, delta_top, delta_bottom : loss {o['M3'][0]:.2f}  A {o['M3'][1]:.3f} delta_top {o['M3'][2]:+.3f} delta_bottom {o['M3'][3]:+.3f}"
      "   <- a position effect only, SCE size free per volume, no volume offset")
print(f"    M2 A_top, A_bottom, common delta     : loss {o['M2'][0]:.2f}  A_top {o['M2'][1]:.3f} A_bottom {o['M2'][2]:.3f} (ratio {o['M2'][1] / o['M2'][2]:.3f}) delta {o['M2'][3]:+.3f}")
print(f"    M4 everything separate               : loss {o['M4'][0]:.2f}  top A {o['M4'][1]:.3f} delta {o['M4'][2]:+.3f} | bottom A {o['M4'][3]:.3f} delta {o['M4'][4]:+.3f}")
rng2 = np.random.default_rng(11); r2, d3, g = [], [], []
for _ in range(150):
    q = comp([T[i] for i in rng2.integers(0, len(T), len(T))], [B[i] for i in rng2.integers(0, len(B), len(B))])
    r2.append(q["M2"][1] / q["M2"][2]); d3.append((q["M3"][2], q["M3"][3])); g.append(q["M3"][0] - q["M2"][0])
d3 = np.array(d3)
print(f"    bootstrap (150): M2 A_top/A_bottom {np.median(r2):.3f} [{np.percentile(r2, 16):.3f},{np.percentile(r2, 84):.3f}] | "
      f"M3 delta_top {np.median(d3[:, 0]):+.2f} [{np.percentile(d3[:, 0], 16):+.2f},{np.percentile(d3[:, 0], 84):+.2f}] "
      f"delta_bottom {np.median(d3[:, 1]):+.2f} [{np.percentile(d3[:, 1], 16):+.2f},{np.percentile(d3[:, 1], 84):+.2f}] | "
      f"loss(M3) - loss(M2) {np.median(g):+.2f} [{np.percentile(g, 16):+.2f},{np.percentile(g, 84):+.2f}]")
A3, dT3, dB3 = o["M3"][1], o["M3"][2], o["M3"][3]
for tr, nm, dl in ((T, "top", dT3), (B, "bottom", dB3)):
    c2 = np.median(arr(tr, "cosx") ** 2); d = arr(tr, "drift"); y = arr(tr, "plat")
    print(f"    M3 vs data, {nm} (delta {dl:+.2f}: field {1 - dl:.2f} E0 at the anode, {1 + 2 * dl:.2f} E0 at the cathode): " + " ".join(
        f"{EDGES[i]}-{EDGES[i + 1]} data {np.median(y[(d >= EDGES[i]) & (d < EDGES[i + 1])]):.3f} pred {A3 * model((EDGES[i] + EDGES[i + 1]) / 2, c2, dl, E0, L):.3f}"
        for i in range(len(EDGES) - 1) if ((d >= EDGES[i]) & (d < EDGES[i + 1])).sum() >= 3))

# ---- figure ----------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
ax = axs[0, 0]; xx = np.linspace(0, 340, 200)
for c2, col, lab in ((0.0, "0.4", "recombination only (cos$^2$ 0)"), (0.06, "C2", "cos$^2$ 0.06 (PDHD stopper median)"), (0.72, "C0", "cos$^2$ 0.72 (PDVD top stopper median)")):
    ax.plot(xx, model(xx, c2, 0.10, **DET["pdvd"]), color=col, lw=2, label=lab + ", $\\delta$ 0.10")
    ax.plot(xx, model(xx, c2, 0.05, **DET["pdvd"]), color=col, lw=1, ls="--")
ax.axhline(1, color="k", lw=0.6); ax.set_xlabel("drift distance from the anode [cm]"); ax.set_ylabel("dQ/dx ratio (toy, A = 1)")
ax.set_title("(a) 1-D space-charge toy at PDVD's 0.45 kV/cm: solid $\\delta$ 0.10 (field -10 % / +20 %), dashed 0.05", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=8)


def binned(tr, n=500, seed=3):
    r = np.random.default_rng(seed); d = arr(tr, "drift"); y = arr(tr, "plat"); out = []
    for i in range(len(EDGES) - 1):
        s = y[(d >= EDGES[i]) & (d < EDGES[i + 1])]
        if len(s) < 3: continue
        b = [np.median(s[r.integers(0, len(s), len(s))]) for _ in range(n)]
        out.append(((EDGES[i] + EDGES[i + 1]) / 2, np.median(s), np.percentile(b, 16), np.percentile(b, 84), len(s)))
    return np.array(out)


ax = axs[0, 1]
for tr, col, nm, A2, A3_ in ((T, "C0", "top (x>0)", o["M2"][1], A3), (B, "C3", "bottom (x<0)", o["M2"][2], A3)):
    ax.plot(arr(tr, "drift"), arr(tr, "plat"), ".", color=col, alpha=0.25, ms=4)
    bb = binned(tr); ax.errorbar(bb[:, 0], bb[:, 1], yerr=[bb[:, 1] - bb[:, 2], bb[:, 3] - bb[:, 1]], fmt="o", color=col, ms=6, capsize=3, label=f"PDVD {nm}: median per 60 cm (n {len(tr)})")
    c2 = np.median(arr(tr, "cosx") ** 2)
    ax.plot(xx, A2 * model(xx, c2, o["M2"][3], **DET["pdvd"]), color=col, lw=2, label=f"M2: common $\\delta$ {o['M2'][3]:+.2f}, A {A2:.3f}")
    dl = dT3 if tr is T else dB3
    ax.plot(xx, A3_ * model(xx, c2, dl, **DET["pdvd"]), color=col, lw=1.2, ls="--", label=f"M3: no offset, $\\delta$ {dl:+.2f}")
ax.set_ylim(0.6, 1.4); ax.axhline(1, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=1, loc="upper left")
ax.set_xlabel("drift distance from its own anode [cm]"); ax.set_ylabel("per-track plateau (dQ/dx / expected, rr 40-60 cm)")
ax.set_title("(b) PDVD top vs bottom at matched drift distance; M3 (position effect only) needs $\\delta_{bottom}$ < 0", fontsize=9)

ax = axs[1, 0]
cls = ((0, 0.5), (0.5, 0.85), (0.85, 1.01)); xs_ = np.arange(3); rng3 = np.random.default_rng(5)
for k, (tr, col, nm) in enumerate(((T, "C0", "PDVD top"), (V[("pdhd", "x>0")], "C2", "PDHD x>0"))):
    det = "pdvd" if tr is T else "pdhd"
    for j, (lo, hi) in enumerate(cls):
        s = [t for t in tr if lo <= abs(t["cosx"]) < hi]
        if len(s) < 10: continue   # the figure drops classes under 10 tracks (PDHD 0.5-0.85 has 9); the text table keeps them
        ds, ys = arr(s, "drift") / 100, arr(s, "plat"); p = np.polyfit(ds, ys, 1)[0]
        bs = [np.polyfit(ds[i], ys[i], 1)[0] for i in (rng3.integers(0, len(s), len(s)) for _ in range(1000))]
        pred = np.polyfit(ds, model(ds * 100, arr(s, "cosx") ** 2, 0.10, **DET[det]), 1)[0]
        ax.errorbar(j + (k - 0.5) * 0.25, p, yerr=[[p - np.percentile(bs, 16)], [np.percentile(bs, 84) - p]], fmt="o", color=col, capsize=3,
                    label=f"{nm} observed" if j == (0 if k == 0 else 0) else None)
        ax.plot(j + (k - 0.5) * 0.25, pred, "_", color=col, ms=18, mew=2.5, label=f"{nm} toy $\\delta$ 0.10" if j == 0 else None)
ax.set_xticks(xs_); ax.set_xticklabels(["|cos$_x$| 0-0.5", "0.5-0.85", "0.85-1"]); ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("slope of plateau vs drift distance [per 100 cm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("(c) the drift trend by track angle: observed (points, 68 %) vs toy (bars)", fontsize=9)

ax = axs[1, 1]
for side, col in (("x>0", "C2"), ("x<0", "C1")):
    tr = V[("pdhd", side)]; ax.plot(arr(tr, "drift"), arr(tr, "plat"), ".", color=col, alpha=0.3, ms=4)
    bb = binned(tr); ax.errorbar(bb[:, 0], bb[:, 1], yerr=[bb[:, 1] - bb[:, 2], bb[:, 3] - bb[:, 1]], fmt="s", color=col, ms=5, capsize=3, label=f"PDHD {side}: median per 60 cm (n {len(tr)})")
    A = np.median(arr(tr, "plat") / model(arr(tr, "drift"), arr(tr, "cosx") ** 2, 0.10, **DET["pdhd"]))
    ax.plot(xx, A * model(xx, np.median(arr(tr, "cosx") ** 2), 0.10, **DET["pdhd"]), color=col, lw=1.5, label=f"toy $\\delta$ 0.10 at its cos$^2$ {np.median(arr(tr, 'cosx') ** 2):.2f}")
ax.set_ylim(0.6, 1.4); ax.axhline(1, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
ax.set_xlabel("drift distance from its own anode [cm]"); ax.set_ylabel("per-track plateau")
ax.set_title("(d) PDHD: stoppers cross the drift at near right angles -> the toy moves dQ/dx ~4 %", fontsize=9)
fig.suptitle("doc pdhd/29 -- space charge against the stopping-muon dQ/dx plateau (d27/dqdx_drift.json)", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.dirname(a.fig), exist_ok=True); fig.savefig(a.fig, dpi=130)
print(f"\nfigure: {os.path.abspath(a.fig)}")

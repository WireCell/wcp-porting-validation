#!/usr/bin/env python3
"""doc pdvd/98 -- the pre-registered readouts and figures: does the regressor's mu correlate with the Q-L drift?

    python3 d98_plots.py [--tag round1]      # scan/d98/stats_<tag>.txt + figs/98_*.png

Readouts (per detector, per tier A / B, in-range events only unless noted), for every crop variant:
  Pearson r with a bootstrap 68 % interval, Spearman rho, a permutation null (N_PERM label shuffles -> one-sided
  p for rho > 0), an OLS slope/intercept of mu on drift with bootstrap 68 % intervals, and the pull width
  (mu - fit)/sigma.  Expected slope from the training constants (doc sec 7): PDVD ~1.3, PDHD ~1.6.
Controls: r(q_keep, drift), r(n_pix, drift), r(mu_Z, q_keep) -- is mu tracking amplitude rather than width?
Success criterion (fixed before the numbers): rho > 0 with p < 0.01 on PDVD tier B, variant Z.
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
SCAN = IMG + "/pdvd/docs/scan/d98"
FIGS = IMG + "/pdvd/docs/nf_sp_img_clus/figs"
OUT_TMP = "/home/xqian/tmp/d98"
VARIANTS = ("Z", "S", "M", "Zw")
EXPECTED_SLOPE = {"pdhd": 1.6, "pdvd": 1.3}
N_PERM, N_BOOT, SEED = 10000, 2000, 20260913
DRIFT_LO, DRIFT_HI = 80.0, 340.0


def load():
    rows = list(csv.DictReader((l for l in open(SCAN + "/scores.tsv") if not l.startswith("#")), delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            if k not in ("det", "key", "tier", "kind", "source", "d97"):
                r[k] = float(v)
    return rows


def readout(x, y, sig, rng):
    """-> dict of the pre-registered statistics of y (mu) against x (drift)"""
    n = len(x)
    if n < 4:
        return dict(n=n)
    r = stats.pearsonr(x, y)[0]; rho = stats.spearmanr(x, y)[0]
    perm = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERM)])
    p = float((perm >= rho).sum() + 1) / (N_PERM + 1)
    bs_r, bs_s, bs_i = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0:
            continue
        bs_r.append(stats.pearsonr(x[i], y[i])[0])
        s, c = np.polyfit(x[i], y[i], 1); bs_s.append(s); bs_i.append(c)
    slope, icpt = np.polyfit(x, y, 1)
    pull = (y - (slope * x + icpt)) / sig
    q = lambda a: tuple(np.quantile(a, [0.16, 0.84]))
    return dict(n=n, r=r, r_lo=q(bs_r)[0], r_hi=q(bs_r)[1], rho=rho, p_perm=p, slope=slope,
                slope_lo=q(bs_s)[0], slope_hi=q(bs_s)[1], icpt=icpt, icpt_lo=q(bs_i)[0], icpt_hi=q(bs_i)[1],
                pull_sd=float(np.std(pull)), resid_rms=float(np.std(y - (slope * x + icpt))),
                mad_identity=float(np.median(np.abs(y - x))))


def fmt(s):
    if s.get("n", 0) < 4:
        return f"n={s.get('n', 0)} (too few)"
    return (f"n={s['n']:3d}  r={s['r']:+.2f} [{s['r_lo']:+.2f},{s['r_hi']:+.2f}]  rho={s['rho']:+.2f}  p_perm={s['p_perm']:.4f}  "
            f"slope={s['slope']:+.2f} [{s['slope_lo']:+.2f},{s['slope_hi']:+.2f}]  icpt={s['icpt']:+.0f}  "
            f"resid_rms={s['resid_rms']:.0f} cm  pull_sd={s['pull_sd']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="round1")
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    rows = load()
    rng = np.random.default_rng(SEED)
    lines = [f"# doc pdvd/98 -- readouts ({a.tag}); d98_plots.py on scan/d98/scores.tsv; perm N={N_PERM}, boot N={N_BOOT}, seed {SEED}",
             "# x = drift_tick (Q-L label), y = mu of the named variant; in-range = 80 <= drift <= 340 cm unless 'all'"]
    S = {}
    for det in ("pdhd", "pdvd"):
        R = [r for r in rows if r["det"] == det]
        lines.append(f"\n== {det}: {len(R)} scored, tier A {sum(r['tier'] == 'A' for r in R)}, "
                     f"in-range {sum(r['inrange'] > 0 for r in R)}; expected slope ~{EXPECTED_SLOPE[det]}")
        for tier in ("B", "A"):
            for rng_name, sel in (("in-range", lambda r: r["inrange"] > 0), ("all", lambda r: True)):
                T = [r for r in R if sel(r) and (tier == "B" or r["tier"] == "A")]
                x = np.array([r["drift_tick"] for r in T])
                for v in VARIANTS:
                    y = np.array([r[f"mu_{v}"] for r in T]); sg = np.array([r[f"sig_{v}"] for r in T])
                    s = readout(x, y, sg, rng); S[(det, tier, rng_name, v)] = s
                    lines.append(f"  tier {tier} {rng_name:8s} {v:2s}: {fmt(s)}")
                if rng_name == "in-range" and len(T) >= 4:
                    qk = np.array([r["q_keep"] for r in T]); npx = np.array([r["n_pix"] for r in T])
                    muz = np.array([r["mu_Z"] for r in T])
                    lines.append(f"  tier {tier} controls: r(q_keep,drift)={stats.pearsonr(qk, x)[0]:+.2f}  "
                                 f"r(n_pix,drift)={stats.pearsonr(npx, x)[0]:+.2f}  r(mu_Z,q_keep)={stats.pearsonr(muz, qk)[0]:+.2f}  "
                                 f"r(mu_Z,ke_best)={stats.pearsonr(muz, np.array([r['ke_best'] for r in T]))[0]:+.2f}")
        # binned medians of mu_Z, tier B in-range, and the doc 97 showcase subset
        T = [r for r in R if r["inrange"] > 0]
        x = np.array([r["drift_tick"] for r in T]); y = np.array([r["mu_Z"] for r in T])
        for lo, hi in ((80, 150), (150, 220), (220, 340)):
            s = (x >= lo) & (x < hi)
            if s.sum():
                lines.append(f"  drift [{lo},{hi}): n={s.sum():2d}  median mu_Z={np.median(y[s]):.0f}  "
                             f"q16/q84={np.quantile(y[s], .16):.0f}/{np.quantile(y[s], .84):.0f}  median drift={np.median(x[s]):.0f}")
        for r in R:
            if r["d97"]:
                lines.append(f"  doc 97 pick {r['key']} ({r['d97']}): drift {r['drift_tick']:.0f}  mu_Z {r['mu_Z']:.0f} +- {r['sig_Z']:.0f}  "
                             f"mu_M {r['mu_M']:.0f}  ke {r['ke_best']:.0f} MeV  frac_lost {r['frac_lost']:.2f}  tier {r['tier']}")
        # saturation census
        muz = np.array([r["mu_Z"] for r in R])
        lines.append(f"  mu_Z < 90 cm (floor-saturated): {(muz < 90).sum()}; mu_Z > 540: {(muz > 540).sum()}; "
                     f"drift < 80: {sum(r['drift_tick'] < 80 for r in R)}")
    crit = S.get(("pdvd", "B", "in-range", "Z"), {})
    ok = crit.get("n", 0) >= 4 and crit["rho"] > 0 and crit["p_perm"] < 0.01
    lines.append(f"\nSUCCESS CRITERION (PDVD tier B in-range Z: rho>0 and p_perm<0.01): {'MET' if ok else 'NOT MET'}"
                 + (f"  rho={crit['rho']:+.2f} p={crit['p_perm']:.4f}" if crit.get("n", 0) >= 4 else ""))
    txt = "\n".join(lines) + "\n"
    open(f"{SCAN}/stats_{a.tag}.txt", "w").write(txt)
    print(txt)
    json.dump({"|".join(k): v for k, v in S.items()}, open(f"{OUT_TMP}/stats_{a.tag}.json", "w"), indent=1)

    # ---- figures
    for det in ("pdhd", "pdvd"):
        R = [r for r in rows if r["det"] == det]
        if not R:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
        for k, v in enumerate(("Z", "M")):
            A = ax[k]
            for tier, mk, col in (("B", "o", "tab:blue"), ("A", "s", "tab:red")):
                T = [r for r in R if r["tier"] == tier]
                if not T:
                    continue
                x = [r["drift_tick"] for r in T]; y = [r[f"mu_{v}"] for r in T]; e = [r[f"sig_{v}"] for r in T]
                inr = np.array([r["inrange"] > 0 for r in T])
                A.errorbar(x, y, yerr=e, fmt=mk, color=col, ms=5, alpha=0.75, lw=0.8, capsize=0,
                           label=f"tier {tier} (n={len(T)})")
                if (~inr).any():
                    A.plot(np.array(x)[~inr], np.array(y)[~inr], "x", color="k", ms=9, mew=1.2,
                           label="out of range (drift<80)" if tier == "B" else None)
            xx = np.array([0, 350]); A.plot(xx, xx, "k--", lw=1, label="identity")
            A.plot(xx, EXPECTED_SLOPE[det] * xx, "g:", lw=1.5, label=f"slope {EXPECTED_SLOPE[det]} (D_L, v_drift)")
            s = S.get((det, "B", "in-range", v), {})
            if s.get("n", 0) >= 4:
                A.plot(xx, s["slope"] * xx + s["icpt"], "-", color="tab:blue", lw=1.5,
                       label=f"fit B: slope {s['slope']:.2f}, r={s['r']:.2f}, p={s['p_perm']:.3f}")
            A.axvspan(0, DRIFT_LO, color="0.9", zorder=0)
            A.set_xlim(0, 350); A.set_ylim(0, 600)
            A.set_xlabel("Q-L drift distance of the Michel  [cm]  (tick route)")
            A.set_ylabel("regressor mu  [cm]  (+- sigma)")
            A.set_title(f"{det.upper()}  variant {v}: " + ("Michel only, muon removed" if v == "Z" else "muon left in"))
            A.legend(fontsize=8, loc="upper left")
        fig.suptitle("doc pdvd/98 -- m3-200k-w drift regressor on real STM+Michel crops")
        fig.tight_layout(); fig.savefig(f"{FIGS}/98_mu_vs_drift_{det}.png", dpi=110); plt.close(fig)

        # residuals and the crop diagnostics
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        T = [r for r in R if r["inrange"] > 0]
        x = np.array([r["drift_tick"] for r in T]); y = np.array([r["mu_Z"] for r in T])
        s = S.get((det, "B", "in-range", "Z"), {})
        if s.get("n", 0) >= 4:
            ax[0].plot(x, y - (s["slope"] * x + s["icpt"]), "o", ms=4)
            ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("drift [cm]"); ax[0].set_ylabel("mu_Z - fit [cm]")
            ax[0].set_title(f"{det.upper()} residual to the tier-B fit, rms {s['resid_rms']:.0f} cm")
        ax[1].scatter([r["frac_lost"] for r in T], y - x, c=[r["ke_best"] for r in T], s=16, cmap="viridis")
        ax[1].set_xlabel("frac_lost (Michel cell charge removed by the muon mask)"); ax[1].set_ylabel("mu_Z - drift [cm]")
        ax[1].set_title("colour = michel_ke_best [MeV]"); ax[1].axhline(0, color="k", lw=0.8)
        ax[2].scatter([r["ke_best"] for r in T], y - x, c=x, s=16, cmap="plasma")
        ax[2].set_xlabel("michel_ke_best [MeV]"); ax[2].set_ylabel("mu_Z - drift [cm]"); ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set_title("colour = drift [cm]")
        fig.tight_layout(); fig.savefig(f"{FIGS}/98_resid_{det}.png", dpi=110); plt.close(fig)

    # model-free width check: per-channel tick RMS^2 of the Z crop within +-WIN ticks of the column's peak (so a
    # second blob on the same channel does not count), 25th percentile over the Michel's channels (the columns least
    # tilted into the drift), against drift.  sigma_t^2 = sigma_0^2 + 2 D_L x / v^3 in ticks^2.
    WIN = 12
    wl = [f"\n# model-free width: w2 = 25th percentile over channels of the per-channel tick RMS^2 within +-{WIN} ticks of the column peak, Z crop [ticks^2]"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for k, det in enumerate(("pdhd", "pdvd")):
        fn = f"{OUT_TMP}/crops_{det}.npz"
        R = [r for r in rows if r["det"] == det]
        if not os.path.exists(fn) or not R:
            continue
        z = np.load(fn); keys = [str(kk) for kk in z["keys"]]; Zc = z["Z"]
        w2 = {}
        for r in R:
            img = Zc[keys.index(r["key"])]
            q = img.sum(axis=1); cols = np.nonzero(q > 0.05 * q.max())[0]
            v = []
            for c in cols:
                pk = int(np.argmax(img[c])); lo, hi = max(0, pk - WIN), min(img.shape[1], pk + WIN + 1)
                p = img[c, lo:hi]; t = np.arange(lo, hi); mu_ = (p * t).sum() / p.sum()
                v.append(((p * (t - mu_) ** 2).sum() / p.sum()))
            if len(v) >= 3:
                w2[r["key"]] = float(np.quantile(v, 0.25))
        T = [r for r in R if r["key"] in w2]
        x = np.array([r["drift_tick"] for r in T]); y = np.array([w2[r["key"]] for r in T])
        vd = {"pdhd": 0.1576, "pdvd": 0.148073}[det]; dl = {"pdhd": 6.2e-6, "pdvd": 4.1307e-6}[det]   # cm/us, cm2/us
        exp_slope = 2 * dl / vd ** 3 / 0.25                                                          # ticks^2 per cm
        rr = stats.pearsonr(x, y)[0]; rho = stats.spearmanr(x, y)[0]
        perm = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERM)])
        p = float((perm >= rho).sum() + 1) / (N_PERM + 1)
        sl, ic = np.polyfit(x, y, 1)
        wl.append(f"{det}: n={len(T)} r={rr:+.2f} rho={rho:+.2f} p_perm={p:.4f} slope={sl:.4f} ticks^2/cm "
                  f"(expected 2 D_L/v^3 = {exp_slope:.4f} with D_L {dl * 1e6:.2f} cm2/s, v {vd * 10:.3f} mm/us) icpt={ic:.1f}")
        ax[k].plot(x, y, "o", ms=4); xx = np.array([0, 350])
        ax[k].plot(xx, sl * xx + ic, "-", label=f"fit slope {sl:.4f} ticks$^2$/cm, r={rr:.2f}, p={p:.3f}")
        ax[k].plot(xx, exp_slope * xx + ic, "g:", label=f"expected 2D$_L$/v$^3$ = {exp_slope:.4f}")
        ax[k].set_xlabel("Q-L drift [cm]"); ax[k].set_ylabel("Michel tick RMS$^2$, 25th pct over channels [ticks$^2$]")
        ax[k].set_title(f"{det.upper()} model-free width vs drift (all scored)"); ax[k].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/98_width_vs_drift.png", dpi=110); plt.close(fig)
    txt2 = "\n".join(wl) + "\n"
    open(f"{SCAN}/stats_{a.tag}.txt", "a").write(txt2); print(txt2)

    # the headline correlation figure: primary variant only, one panel per detector, binned medians with q16-q84
    # bands, tier A marked, the tier-B fit, identity, and the readouts written on the panel
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.8))
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.2))
    for k, det in enumerate(("pdvd", "pdhd")):
        R = [r for r in rows if r["det"] == det and r["inrange"] > 0]
        if len(R) < 4:
            continue
        A = ax[k]
        x = np.array([r["drift_tick"] for r in R]); y = np.array([r["mu_Z"] for r in R])
        e = np.array([r["sig_Z"] for r in R]); ta = np.array([r["tier"] == "A" for r in R])
        A.errorbar(x[~ta], y[~ta], yerr=e[~ta], fmt="o", color="tab:blue", ms=5, alpha=0.6, lw=0.7, label=f"tier B (n={(~ta).sum()})")
        if ta.any():
            A.errorbar(x[ta], y[ta], yerr=e[ta], fmt="s", color="tab:red", ms=6, alpha=0.9, lw=0.8, label=f"tier A, clean (n={ta.sum()})")
        edges = [80, 150, 220, 340]
        bx, bm, blo, bhi = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x >= lo) & (x < hi)
            if s.sum() >= 3:
                bx.append(np.median(x[s])); bm.append(np.median(y[s]))
                blo.append(np.quantile(y[s], .16)); bhi.append(np.quantile(y[s], .84))
        A.fill_between(bx, blo, bhi, color="orange", alpha=0.2, label="binned q16-q84")
        A.plot(bx, bm, "D-", color="darkorange", ms=9, lw=2, label="binned median", zorder=5)
        xx = np.array([80, 340]); A.plot(xx, xx, "k--", lw=1, label="identity")
        s = S.get((det, "B", "in-range", "Z"), {})
        A.plot(xx, s["slope"] * xx + s["icpt"], "-", color="tab:blue", lw=2,
               label=f"OLS fit: slope {s['slope']:.2f} [{s['slope_lo']:.2f}, {s['slope_hi']:.2f}]")
        A.axhspan(0, 90, color="0.92", zorder=0)
        A.text(0.03, 0.97, f"Pearson r = {s['r']:+.2f} [{s['r_lo']:+.2f}, {s['r_hi']:+.2f}]\nSpearman rho = {s['rho']:+.2f}\n"
                           f"permutation p = {s['p_perm']:.1g}\nn = {s['n']} Michels, 80-340 cm",
               transform=A.transAxes, va="top", fontsize=10, bbox=dict(fc="white", ec="0.5", alpha=0.9))
        A.set_xlim(70, 350); A.set_ylim(0, 360)
        A.set_xlabel("true drift distance of the Michel from Q-L matching  [cm]", fontsize=11)
        A.set_ylabel("regressor prediction mu  [cm]  (bars: the model's own sigma)", fontsize=11)
        A.set_title(f"{det.upper()} ({'p96vprod' if det == 'pdvd' else 'h28prod'}): Michel-only crop, muon removed", fontsize=12)
        A.legend(fontsize=8.5, loc="lower right")
        # rank-rank view (what Spearman sees)
        B = ax2[k]
        rx = stats.rankdata(x); ry = stats.rankdata(y)
        B.plot(rx[~ta], ry[~ta], "o", color="tab:blue", ms=5, alpha=0.7, label="tier B")
        if ta.any():
            B.plot(rx[ta], ry[ta], "s", color="tab:red", ms=6, label="tier A")
        B.plot([1, len(x)], [1, len(x)], "k--", lw=1)
        B.set_xlabel("rank of true drift (1 = nearest anode)"); B.set_ylabel("rank of predicted mu")
        B.set_title(f"{det.upper()}: rank vs rank, rho = {s['rho']:+.2f}, p = {s['p_perm']:.1g}")
        B.legend(fontsize=9, loc="upper left")
    fig.suptitle("doc pdvd/98 -- DUNE-VD-simulation-trained diffusion drift regressor (m3-200k-w) on real STM+Michel electrons", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIGS}/98_correlation_summary.png", dpi=120); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(f"{FIGS}/98_rank_rank.png", dpi=110); plt.close(fig2)

    # topology splits: the same reco-vs-true axes, one panel per split, each group with its own rho / n
    SPLITS = [("muon overlap", lambda r: r["frac_lost"] < 0.2, "Michel loses <20 % to the muon", "loses >=20 % (overlapping)"),
              ("connection", lambda r: r["conn_type"] == 1, "attached to the stop", "bridged (gap to the stop)"),
              ("hand kind", lambda r: r["kind"] == "attached", "Michel only (attached)", "Michel + gamma pieces (both)"),
              ("energy", lambda r: r["ke_best"] >= 20, "michel_ke_best >= 20 MeV", "< 20 MeV"),
              ("Bragg peak", lambda r: r["ratio"] >= 0.8, "contrast ratio >= 0.8 (clear)", "< 0.8 (weak)"),
              ("tier", lambda r: r["tier"] == "A", "tier A (clean by every rule)", "tier B rest")]
    tl = ["\n# topology splits, variant Z, in-range: per-group n / r / rho / p_perm / slope"]
    for det in ("pdvd", "pdhd"):
        R = [r for r in rows if r["det"] == det and r["inrange"] > 0]
        if len(R) < 4:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
        tl.append(f"== {det}")
        for A, (name, pred, lab1, lab0) in zip(axs.ravel(), SPLITS):
            xx = np.array([80, 340]); A.plot(xx, xx, "k--", lw=1, label="identity")
            for grp, col, mk in ((True, "tab:green", "o"), (False, "tab:purple", "^")):
                G = [r for r in R if bool(pred(r)) == grp]
                lab = lab1 if grp else lab0
                if not G:
                    tl.append(f"  {name:12s} {lab:36s}: n=0"); continue
                x = np.array([r["drift_tick"] for r in G]); y = np.array([r["mu_Z"] for r in G]); e = np.array([r["sig_Z"] for r in G])
                s = readout(x, y, e, rng)
                txt = (f"{lab} (n={len(G)}): rho={s['rho']:+.2f}, p={s['p_perm']:.2g}, slope {s['slope']:.2f}"
                       if s.get("n", 0) >= 4 else f"{lab} (n={len(G)})")
                A.errorbar(x, y, yerr=e, fmt=mk, color=col, ms=6, alpha=0.7, lw=0.6, label=txt)
                if s.get("n", 0) >= 4:
                    A.plot(xx, s["slope"] * xx + s["icpt"], "-", color=col, lw=1.5)
                    tl.append(f"  {name:12s} {lab:36s}: {fmt(s)}")
                else:
                    tl.append(f"  {name:12s} {lab:36s}: n={len(G)} (too few)")
            A.axhspan(0, 90, color="0.92", zorder=0)
            A.set_xlim(70, 350); A.set_ylim(0, 360)
            A.set_xlabel("true drift from Q-L matching [cm]"); A.set_ylabel("regressor mu [cm]")
            A.set_title(f"{det.upper()} split by {name}", fontsize=11); A.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"doc pdvd/98 -- {det.upper()}: reco vs true drift by topology (variant Z, Michel only, 80-340 cm)", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{FIGS}/98_topology_{det}.png", dpi=105); plt.close(fig)
    open(f"{SCAN}/stats_{a.tag}.txt", "a").write("\n".join(tl) + "\n"); print("\n".join(tl))

    # label check
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for k, det in enumerate(("pdhd", "pdvd")):
        R = [r for r in rows if r["det"] == det]
        if not R:
            continue
        a_ = np.array([r["drift_fit"] for r in R]); b_ = np.array([r["drift_tick"] for r in R])
        ax[k].plot(a_, b_, "o", ms=4); ax[k].plot([0, 350], [0, 350], "k--", lw=0.8)
        ax[k].set_xlabel("fit route: x_anode(W) - |x| over Michel points [cm]")
        ax[k].set_ylabel("tick route: (tick*0.5 - t0 - trig)*v [cm]")
        ok = np.isfinite(a_)
        ax[k].set_title(f"{det.upper()} label check, median diff {np.median(b_[ok] - a_[ok]):+.1f} cm, sd {np.std(b_[ok] - a_[ok]):.1f}")
    fig.tight_layout(); fig.savefig(f"{FIGS}/98_label_check.png", dpi=110); plt.close(fig)

    # example crops: near / far, Z beside M, zoomed to the nonzero box of M
    for det in ("pdhd", "pdvd"):
        fn = f"{OUT_TMP}/crops_{det}.npz"
        R = [r for r in rows if r["det"] == det and r["inrange"] > 0]
        if not os.path.exists(fn) or len(R) < 2:
            continue
        z = np.load(fn); keys = [str(k) for k in z["keys"]]
        R.sort(key=lambda r: r["drift_tick"])
        picks = [R[0], R[len(R) // 3], R[2 * len(R) // 3], R[-1]]
        fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
        for k, r in enumerate(picks):
            i = keys.index(r["key"])
            M, Z = z["M"][i], z["Z"][i]
            nz = np.nonzero(M)
            if not len(nz[0]):
                continue
            c_lo, c_hi = max(nz[0].min() - 6, 0), min(nz[0].max() + 7, 256)
            t_lo, t_hi = max(nz[1].min() - 20, 0), min(nz[1].max() + 21, 1024)
            vmax = max(M[c_lo:c_hi, t_lo:t_hi].max(), 1)
            for j, (img, nm) in enumerate(((M, "M: muon left in"), (Z, "Z: muon removed"))):
                A = ax[j, k]
                A.imshow(img[c_lo:c_hi, t_lo:t_hi], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                         extent=[t_lo, t_hi, c_lo, c_hi])
                A.set_xlabel("crop tick (0.5 us)"); A.set_ylabel("crop channel")
                A.set_title(f"{r['key']} {nm}\n" + (f"drift {r['drift_tick']:.0f} cm, mu_Z {r['mu_Z']:.0f}+-{r['sig_Z']:.0f}, "
                                                    f"mu_M {r['mu_M']:.0f}; ke {r['ke_best']:.0f} MeV; lost {r['frac_lost']:.2f}"),
                            fontsize=8)
        fig.suptitle(f"doc pdvd/98 -- {det.upper()} example crops (zoomed to the Michel's box; full crop is 256 ch x 1024 ticks)")
        fig.tight_layout(); fig.savefig(f"{FIGS}/98_crops_{det}.png", dpi=100); plt.close(fig)
    print("figures in", FIGS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

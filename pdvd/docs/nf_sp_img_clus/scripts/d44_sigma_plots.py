#!/usr/bin/env python3
"""doc pdvd/44 -- figures for the effective transverse width derivation.

  <out>_sigma_fit.png        sigma_eff^2 vs drift time per plane, both detectors, both
                             estimators, with the fitted lines and the configured model
  <out>_sigma_occasions.png  c_eff and DT_eff per validation occasion (joint fits)
  <out>_sigma_shape.png      measured ring shares vs the binned Gaussians

Usage: d44_sigma_plots.py --pdvd figs/44_sigma_pdvd --sbnd figs/44_sigma_sbnd --out figs/44
       [--extra label=prefix ...]   extra bins/fit TSV prefixes (window / advance controls) for the occasions panel
"""
import argparse, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PITCH = {"pdvd": (7.65, 7.65, 5.10), "sbnd": (3.0, 3.0, 3.0)}


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def fig_fit(dets, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex="row")
    for i, (det, pre) in enumerate(dets.items()):
        bins = read(pre + "_bins.tsv"); fits = read(pre + "_fit.tsv")
        for j, p in enumerate("UVW"):
            ax = axes[i, j]
            for est, col, mk in (("rms", "C0", "o"), ("share", "C3", "s")):
                b = [r for r in bins if r["label"] == "all" and r["est"] == est and r["plane"] == p]
                if not b:
                    continue
                t = np.array([r["t_us"] for r in b]); s2 = np.array([r["sig2_eff_mm2"] for r in b])
                e = np.array([r["sig2_err"] for r in b])
                ax.errorbar(t, s2, yerr=e, fmt=mk, color=col, ms=5, capsize=2, label="measured (%s)" % est)
                f = [r for r in fits if r["label"] == "all" and r["est"] == est and r["plane"] == p]
                fj = [r for r in fits if r["label"] == "all" and r["est"] == est and r["plane"] == p + "(joint)"]
                tt = np.linspace(0, max(t) * 1.05, 50)
                if f:
                    ax.plot(tt, 2 * f[0]["DT_json"] * tt * 1e3 + f[0]["c2_mm2"], "-", color=col, lw=1,
                            label="fit %s: D=%.1f, c=%.2f mm" % (est, f[0]["DT_eff_cm2s"], f[0]["c_eff_mm"]))
                if fj:
                    ax.plot(tt, 2 * fj[0]["DT_json"] * tt * 1e3 + fj[0]["c2_mm2"], "--", color=col, lw=1,
                            label="joint %s: D=%.1f, c=%.2f mm" % (est, fj[0]["DT_eff_cm2s"], fj[0]["c_eff_mm"]))
            b0 = [r for r in bins if r["label"] == "all" and r["est"] == "rms" and r["plane"] == p]
            if b0:
                f0 = [r for r in fits if r["label"] == "all" and r["plane"] == p][0]
                tt = np.linspace(0, max(r["t_us"] for r in b0) * 1.05, 50)
                ax.plot(tt, 2 * f0["DT_model_cm2s"] * 1e-7 * tt * 1e3 + f0["c_model_mm"] ** 2, "k:", lw=1.5,
                        label="configured: D=%.1f, c=%.3f mm" % (f0["DT_model_cm2s"], f0["c_model_mm"]))
            ax.set_title("%s %s  (pitch %.2f mm)" % (det.upper(), p, PITCH[det][j]))
            ax.set_xlabel("drift time [us]"); ax.set_ylabel("sigma_T^2 [mm^2]")
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
            ax.set_ylim(bottom=0)
    fig.suptitle("effective transverse width vs drift: sigma_eff^2 = 2 D_T t + c^2 (prolonged segments, own-centroid, unfolded)")
    fig.tight_layout(); fig.savefig(out + "_sigma_fit.png", dpi=110); plt.close(fig)


def fig_occasions(dets, extra, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, (det, pre) in enumerate(dets.items()):
        fits = read(pre + "_fit.tsv")
        for lab, p2 in extra.get(det, []):
            for r in read(p2 + "_fit.tsv"):
                if r["label"] == "all":
                    r["label"] = lab; fits.append(r)
        labels = []
        for r in fits:
            if r["label"] not in labels:
                labels.append(r["label"])
        for k, est in enumerate(("rms", "share")):
            ax = axes[i, k]
            y = np.arange(len(labels))
            for j, p in enumerate("UVW"):
                c = np.full(len(labels), np.nan); e = np.full(len(labels), np.nan)
                for li, lab in enumerate(labels):
                    f = [r for r in fits if r["label"] == lab and r["est"] == est and r["plane"] == p + "(joint)"]
                    if f:
                        c[li] = f[0]["c_eff_mm"]; e[li] = f[0]["c_err"]
                ax.errorbar(c, y + (j - 1) * 0.22, xerr=e, fmt="o", ms=4, capsize=2, label="c_eff %s" % p)
            d = np.full(len(labels), np.nan); de = np.full(len(labels), np.nan)
            for li, lab in enumerate(labels):
                f = [r for r in fits if r["label"] == lab and r["est"] == est and r["plane"] == "U(joint)"]
                if f:
                    d[li] = f[0]["DT_eff_cm2s"]; de[li] = f[0]["DT_err"]
            ax2 = ax.twiny()
            ax2.errorbar(d, y, xerr=de, fmt="k^", ms=4, capsize=2, alpha=0.6, label="D_T eff [cm2/s] (top axis)")
            f0 = [r for r in fits if r["label"] == "all" and r["plane"] == "U(joint)"]
            if f0:
                ax2.axvline(f0[0]["DT_model_cm2s"], color="k", ls=":", lw=1)
            ax2.set_xlim(-10, 25); ax2.set_xlabel("D_T eff [cm2/s]  (black triangles; dotted = configured)")
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("c_eff [mm]"); ax.set_title("%s, %s estimator (joint fits)" % (det.upper(), est))
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right"); ax.invert_yaxis()
    fig.suptitle("the derived constants across validation occasions")
    fig.tight_layout(); fig.savefig(out + "_sigma_occasions.png", dpi=110); plt.close(fig)


def fig_shape(dets, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (det, pre) in enumerate(dets.items()):
        sh = read(pre + "_shape.tsv"); ax = axes[i]
        x = np.arange(4); w = 0.2
        names = ["centre", "+-1", "+-2", "beyond"]
        for j, r in enumerate(sh):
            base = j * 5
            ax.bar(base + x - 1.5 * w, [r["meas_centre"], r["meas_pm1"], r["meas_pm2"], r["meas_beyond"]], w, color="k", label="measured" if j == 0 else None)
            ax.bar(base + x - 0.5 * w, [r["gaus_model_centre"], r["gaus_model_pm1"], r["gaus_model_pm2"], r["gaus_model_beyond"]], w, color="C7", label="Gaussian, configured sigma" if j == 0 else None)
            ax.bar(base + x + 0.5 * w, [r["gaus_fit_centre"], r["gaus_fit_pm1"], r["gaus_fit_pm2"], r["gaus_fit_beyond"]], w, color="C0", label="Gaussian, rms-matched" if j == 0 else None)
            ax.bar(base + x + 1.5 * w, [r["gaus_share_centre"], r["gaus_share_pm1"], r["gaus_share_pm2"], r["gaus_share_beyond"]], w, color="C3", label="Gaussian, share-matched" if j == 0 else None)
            ax.text(base + 1.5, 0.86 - 0.05 * (j % 2), "%s: stacked mismatch\nconfigured %.3f / rms %.3f / share %.3f" % (
                r["plane"], r["U_stack_model"], r["U_stack_rms"], r["U_stack_share"]), ha="center", fontsize=7)
        ax.set_xticks([j * 5 + k for j in range(len(sh)) for k in x]); ax.set_xticklabels(names * len(sh), fontsize=7)
        ax.set_ylim(0, 1.02); ax.set_ylabel("share of the profile's charge"); ax.set_yscale("linear")
        ax.set_title("%s: ring shares about the centroid (prolonged)" % det.upper()); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out + "_sigma_shape.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", required=True); ap.add_argument("--sbnd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="det:label=prefix, e.g. pdvd:window+-2=/tmp/44_sigma_pdvd_hw2")
    a = ap.parse_args()
    dets = {"pdvd": a.pdvd, "sbnd": a.sbnd}
    extra = {}
    for e in a.extra:
        det, rest = e.split(":", 1); lab, pre = rest.split("=", 1)
        extra.setdefault(det, []).append((lab, pre))
    fig_fit(dets, a.out); fig_occasions(dets, extra, a.out); fig_shape(dets, a.out)
    print("->", a.out + "_sigma_{fit,occasions,shape}.png")


if __name__ == "__main__":
    sys.exit(main())

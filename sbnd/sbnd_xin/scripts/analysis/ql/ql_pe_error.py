#!/usr/bin/env python3
"""Measure the per-PMT light-error fraction `a` from the 10 data hand-scans.

The Q/L matcher assumes a per-PMT light error of 30% (m_pe_err_frac=0.3,
QLMatching.h), entering the chi2 as sigma^2 = meas + (0.3*meas)^2.  Using the
trusted (flash <-> cluster) matches the user hand-scanned in the ql_scan viewer,
we MEASURE the fractional error by modelling the per-PMT light variance as

    E[(pred - meas)^2] = meas + a * pred^2          (a = fractional-error^2)

and fitting `a`.  30% would give a = 0.09.

This script produces three variants (see main()):
  ql_pe_error_data.png            NL-on  predicted light, all kept matches
  ql_pe_error_data_nloff.png      NL-off predicted light, all kept matches
  ql_pe_error_data_consistent.png NL-on, high-consistency subset (chi2/ndf & ks cut)

NL-on  pred is read from work/ql_nl_study/calib-evt<ID>.nl.json (PMT_NL=true).
NL-off pred is read from work/ql_evt<ID>/calib-evt<ID>.json     (PMT_NL=false).
Clusters/flashes are identical between the two (verified); only pred differs.

Selection (the user's cuts):
  - reconstruct selected bundles by matching (flash_gid, main_cluster);
  - aggregate per FLASH: pred summed over all selected bundles sharing a flash_gid,
    meas counted once (some flashes have two hand-selected clusters);
  - drop a flash if ANY contributing bundle is window_truncated or close_to_PMT;
  - [consistent variant] additionally require ALL contributing bundles to have
    chi2/ndf < CHI2NDF_MAX and ks_dis < KS_MAX;
  - keep PMT channels only (opdet type==1), in the flash's own TPC, meas>0 or pred>0.

NOTE: in this sample pred systematically OVER-predicts (median pred/meas ~ 1.3); that
coherent +30% normalization offset, not random per-PMT scatter, is most of what drives
`a` above 0.09.  The pred-vs-meas panel makes that offset visible.

Usage:  python3 ql_pe_error.py
"""
import glob
import gzip
import json
import os
import re


def calib_open(path):
    """Open a calib dump, transparently accepting a gzipped `<path>.gz` sibling.

    The 2026-07-24 work-tree consolidation gzipped the archived dumps (~6x);
    fresh -calib runs still write plain .json, so both names must resolve.
    """
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PICS = os.path.join(HERE, "pics")
NL_DIR = os.path.join(HERE, "work", "ql_nl_study")   # NL dumps for data+mc (IDs unique)

A_REF = 0.09          # the assumed 30% fractional error, squared
PMT_TYPE = 1          # opdet type for PMTs (type 0 = X-ARAPUCA, always 0 light)
CHI2NDF_MAX = 5.0     # "high-consistency" subset cuts
KS_MAX = 0.10


def robust_width(x):
    """1.4826 * MAD -- a Gaussian-sigma estimate insensitive to heavy tails."""
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def calib_path(ev, nl):
    return (os.path.join(NL_DIR, "calib-evt%d.nl.json" % ev) if nl
            else os.path.join(HERE, "work", "ql_evt%d" % ev, "calib-evt%d.json" % ev))


def load_event(ev, nl, consistent, state_dir):
    """Per-PMT (pred, meas) arrays for one event's surviving hand-scan flashes."""
    want = {tuple(k) for k in json.load(open(
        os.path.join(state_dir, ".scan_state-evt%d.json" % ev)))["selected"]}
    cal = json.load(calib_open(calib_path(ev, nl)))
    ch_apa = np.array([o["apa"] for o in cal["opdets"]])
    ch_type = np.array([o["type"] for o in cal["opdets"]])
    flash_pe = {f["gid"]: np.asarray(f["pe"], float) for f in cal["flashes"]}
    flash_apa = {f["gid"]: f["apa"] for f in cal["flashes"]}

    groups = {}
    for b in cal["bundles"]:
        if (b["flash_gid"], b["main_cluster"]) in want:
            groups.setdefault(b["flash_gid"], []).append(b)

    n_sel = len(groups)
    n_drop_flag = n_drop_consist = 0
    preds, meass = [], []
    for gid, bundles in groups.items():
        if any(b["close_to_PMT"] or b["window_truncated"] for b in bundles):
            n_drop_flag += 1
            continue
        if consistent and not all(
                (b["chi2"] / max(b["ndf"], 1)) < CHI2NDF_MAX and b["ks_dis"] < KS_MAX
                for b in bundles):
            n_drop_consist += 1
            continue
        pred = np.sum([np.asarray(b["pred_pe"], float) for b in bundles], axis=0)
        meas = flash_pe[gid]
        keep = (ch_type == PMT_TYPE) & (ch_apa == flash_apa[gid]) & ((meas > 0) | (pred > 0))
        preds.append(pred[keep])
        meass.append(meas[keep])
    pred = np.concatenate(preds) if preds else np.array([])
    meas = np.concatenate(meass) if meass else np.array([])
    return dict(pred=pred, meas=meas, n_sel=n_sel,
                n_kept=n_sel - n_drop_flag - n_drop_consist)


def make_figure(events, nl, consistent, outname, subtitle, state_dir):
    pred_all, meas_all, n_kept = [], [], 0
    for ev in events:
        d = load_event(ev, nl, consistent, state_dir)
        pred_all.append(d["pred"])
        meas_all.append(d["meas"])
        n_kept += d["n_kept"]
    pred = np.concatenate(pred_all)
    meas = np.concatenate(meas_all)

    fit = pred > 0
    pf, mf = pred[fit], meas[fit]
    R = (pf - mf) ** 2
    Y = R - mf
    a_hat = float(np.sum(Y) / np.sum(pf ** 2))
    frac_hat = 100.0 * np.sqrt(max(a_hat, 0.0))
    norm = float(np.sum(mf) / np.sum(pf))            # meas/pred normalization
    ratio_med = float(np.median(mf[mf > 0] / pf[mf > 0])) if np.any(mf > 0) else float("nan")

    def pull_width(a):
        return robust_width((pf - mf) / np.sqrt(mf + a * pf ** 2))
    w_hat, w_ref = pull_width(a_hat), pull_width(A_REF)

    # binned profile in pred (log bins), bin MEANS (R ~ chi2_1 => use mean, not median)
    lo = max(pf.min(), 1e-1)
    edges = np.logspace(np.log10(lo), np.log10(pf.max()), 16)
    idx = np.digitize(pf, edges)
    bx, b_a, b_err, b_meanY, b_meanP2, b_predmeas = [], [], [], [], [], []
    for i in range(1, len(edges)):
        sel = idx == i
        n = int(np.sum(sel))
        if n < 5:
            continue
        yb, p2 = Y[sel], pf[sel] ** 2
        mp2 = np.mean(p2)
        bx.append(np.mean(pf[sel]))
        b_a.append(np.mean(yb) / mp2)
        b_err.append((np.std(yb) / np.sqrt(n)) / mp2)
        b_meanY.append(np.mean(yb)); b_meanP2.append(mp2)
        b_predmeas.append(np.mean(mf[sel]) / np.mean(pf[sel]))
    bx = np.array(bx); b_a = np.array(b_a); b_err = np.array(b_err)
    b_meanY = np.array(b_meanY); b_meanP2 = np.array(b_meanP2); b_predmeas = np.array(b_predmeas)

    # -------- figure --------
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 1. pred vs meas (shows the normalization offset directly)
    a0 = ax[0, 0]
    a0.scatter(pf, mf, s=4, alpha=0.15, color="C0", rasterized=True)
    lim = [max(min(pf.min(), mf[mf > 0].min()) * 0.5, 1e-1), max(pf.max(), mf.max()) * 2]
    a0.plot(lim, lim, "k--", lw=1, label="meas = pred")
    a0.plot(lim, [norm * x for x in lim], "C3-", lw=1.5, label="meas = %.2f pred (fit)" % norm)
    a0.set_xscale("log"); a0.set_yscale("log"); a0.set_xlim(lim); a0.set_ylim(lim)
    a0.set_xlabel("pred  (PE)"); a0.set_ylabel("meas  (PE)")
    direction = "over-predicts" if ratio_med < 1 else "under-predicts"
    a0.set_title("Measured vs predicted per-PMT light\n(median meas/pred = %.2f -> pred %s)" % (ratio_med, direction))
    a0.legend(loc="upper left")

    # 2. local a vs pred
    a1 = ax[0, 1]
    a1.errorbar(bx, b_a, yerr=b_err, fmt="o-", color="C0", capsize=3, label="binned local a")
    a1.axhline(a_hat, color="C3", ls="-", lw=1.5, label=r"global $\hat{a}=%.3f$ (%.0f%%)" % (a_hat, frac_hat))
    a1.axhline(A_REF, color="k", ls="--", lw=1.5, label="a = 0.09 (30%)")
    a1.set_xscale("log"); a1.set_xlabel("pred  (PE)")
    a1.set_ylabel(r"local $a$ = mean(Y) / mean(pred$^2$)")
    a1.set_title("Local fractional-error$^2$ vs predicted light")
    a1.legend(loc="best")
    a1.set_ylim(-0.05, max(0.40, a_hat * 1.6))

    # 3. Y vs pred^2 linearity
    a2 = ax[1, 0]
    a2.scatter(pf ** 2, Y, s=4, alpha=0.12, color="C0", rasterized=True, label="per-PMT")
    a2.plot(b_meanP2, b_meanY, "s-", color="C1", label="bin mean")
    xx = np.array([0, pf.max() ** 2])
    a2.plot(xx, a_hat * xx, "C3-", lw=1.5, label=r"$\hat{a}\,$pred$^2$")
    a2.plot(xx, A_REF * xx, "k--", lw=1.2, label=r"$0.09\,$pred$^2$")
    a2.set_xscale("symlog"); a2.set_yscale("symlog")
    a2.set_xlabel(r"pred$^2$  (PE$^2$)")
    a2.set_ylabel(r"Y = $(\mathrm{pred}-\mathrm{meas})^2 - \mathrm{meas}$")
    a2.set_title("Excess squared residual vs pred$^2$ (slope = a)")
    a2.legend(loc="upper left")

    # 4. pull histograms
    a3 = ax[1, 1]
    for a, c, lab in [(a_hat, "C3", r"$\hat{a}=%.3f$: width=%.2f" % (a_hat, w_hat)),
                      (A_REF, "k", "a=0.09: width=%.2f" % w_ref)]:
        pull = (pf - mf) / np.sqrt(mf + a * pf ** 2)
        a3.hist(np.clip(pull, -6, 6), bins=60, histtype="step", color=c, lw=1.6, label=lab, density=True)
    xs = np.linspace(-6, 6, 200)
    a3.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "C7:", lw=1, label="unit Gaussian")
    a3.axvline(0, color="0.7", lw=0.8)
    a3.set_xlabel(r"pull = $(\mathrm{pred}-\mathrm{meas})/\sqrt{\mathrm{meas}+a\,\mathrm{pred}^2}$")
    a3.set_ylabel("density")
    a3.set_title("Pull (robust width ~ 1 when a is right; offset from 0 = norm bias)")
    a3.legend(loc="upper left", fontsize=9)

    fig.suptitle("SBND data Q/L per-PMT light error  |  %s  |  "
                 r"$\hat{a}=%.3f$ ($\sqrt{a}=%.0f\%%$),  meas/pred=%.2f,  N$_{PMT}$=%d,  N$_{flash}$=%d"
                 % (subtitle, a_hat, frac_hat, norm, pf.size, n_kept), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(PICS, outname)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("  %-34s a_hat=%.3f (%.0f%%)  meas/pred=%.2f  pull[a_hat]=%.2f pull[0.09]=%.2f  N_pmt=%d N_flash=%d"
          % (outname, a_hat, frac_hat, norm, w_hat, w_ref, pf.size, n_kept))
    print("    wrote", out)


def main(mode):
    os.makedirs(PICS, exist_ok=True)
    state_dir = os.path.join(HERE, "work", "ql_labels", mode)
    events = sorted(int(re.search(r"evt(\d+)", os.path.basename(p)).group(1))
                    for p in glob.glob(os.path.join(state_dir, ".scan_state-evt*.json")))
    print("[%s] events: %s" % (mode, events))
    make_figure(events, nl=True, consistent=False,
                outname="ql_pe_error_%s.png" % mode,
                subtitle="%s: 10 hand-scan events, NL-corrected pred" % mode, state_dir=state_dir)
    make_figure(events, nl=False, consistent=False,
                outname="ql_pe_error_%s_nloff.png" % mode,
                subtitle="%s: 10 hand-scan events, NL OFF (linear pred)" % mode, state_dir=state_dir)
    make_figure(events, nl=True, consistent=True,
                outname="ql_pe_error_%s_consistent.png" % mode,
                subtitle="%s: high-consistency subset (chi2/ndf<%.0f & ks<%.2f), NL pred"
                         % (mode, CHI2NDF_MAX, KS_MAX), state_dir=state_dir)


if __name__ == "__main__":
    modes = sys.argv[1:] or ["data", "mc"]
    for m in modes:
        main(m)

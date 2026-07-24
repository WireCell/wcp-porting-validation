#!/usr/bin/env python3
"""Compare data vs MC Q/L match quality on the hand-scans, AFTER the data recipe.

Recipe (cfg/.../sbnd/qlmatching.jsonnet): DATA prediction scaled by QtoL=0.86 (sim 1.0);
per-PMT light error sigma^2 = meas + max(5 PE, 0.25*pred)^2 (pe_err_on_pred, both modes).
This script reads the regenerated calib dumps (work/ql_recipe/) and compares the updated
chi2/ndf, ks, and meas/pred for the hand-scan-selected matches, data vs MC.

Selection mirrors ql_pe_error.py: reconstruct selected bundles by (flash_gid, main_cluster);
aggregate per flash; drop window_truncated / close_to_PMT flashes; PMT channels (opdet type 1)
in the flash TPC.

Output: pics/ql_recipe_data_vs_mc.png  + a printed summary.
"""
import glob, gzip, json, os, re


def calib_open(path):
    """Open a calib dump, transparently accepting a gzipped `<path>.gz` sibling.

    The 2026-07-24 work-tree consolidation gzipped the archived dumps (~6x);
    fresh -calib runs still write plain .json, so both names must resolve.
    """
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)


import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PICS = os.path.join(HERE, "pics")
RECIPE_DIR = os.path.join(HERE, "work", "ql_recipe")


def collect(mode):
    sd = os.path.join(HERE, "work", "ql_labels", mode)
    evs = sorted(int(re.search(r"evt(\d+)", os.path.basename(p)).group(1))
                 for p in glob.glob(os.path.join(sd, ".scan_state-evt*.json")))
    chi2ndf, ks, mp, totpe = [], [], [], []
    for ev in evs:
        want = {tuple(k) for k in json.load(open("%s/.scan_state-evt%d.json" % (sd, ev)))["selected"]}
        c = json.load(calib_open(os.path.join(RECIPE_DIR, "calib-evt%d.json" % ev)))
        fpe = {f["gid"]: np.asarray(f["pe"], float) for f in c["flashes"]}
        fap = {f["gid"]: f["apa"] for f in c["flashes"]}
        ca = np.array([o["apa"] for o in c["opdets"]]); ct = np.array([o["type"] for o in c["opdets"]])
        gr = {}
        for b in c["bundles"]:
            if (b["flash_gid"], b["main_cluster"]) in want:
                gr.setdefault(b["flash_gid"], []).append(b)
        for gid, bs in gr.items():
            if any(b["close_to_PMT"] or b["window_truncated"] for b in bs):
                continue
            for b in bs:
                chi2ndf.append(b["chi2"] / max(b["ndf"], 1)); ks.append(b["ks_dis"])
            pred = np.sum([np.asarray(b["pred_pe"], float) for b in bs], axis=0); meas = fpe[gid]
            k = (ct == 1) & (ca == fap[gid]) & ((meas > 0) | (pred > 0))
            if pred[k].sum() > 0:
                mp.append(meas[k].sum() / pred[k].sum()); totpe.append(meas[k].sum())
    return dict(chi2ndf=np.array(chi2ndf), ks=np.array(ks), mp=np.array(mp), totpe=np.array(totpe))


def main():
    os.makedirs(PICS, exist_ok=True)
    D, M = collect("data"), collect("mc")
    for name, d in [("DATA", D), ("MC", M)]:
        print("%s: %d bundles | median chi2/ndf=%.2f | median ks=%.3f | median meas/pred=%.2f"
              % (name, d["chi2ndf"].size, np.median(d["chi2ndf"]), np.median(d["ks"]), np.median(d["mp"])))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    C = {"DATA": "C3", "MC": "C0"}

    # 1. chi2/ndf
    a = ax[0, 0]
    bins = np.logspace(-1, 2.5, 40)
    for n, d in [("DATA", D), ("MC", M)]:
        a.hist(np.clip(d["chi2ndf"], bins[0], bins[-1]), bins=bins, histtype="step", lw=1.8,
               color=C[n], density=True, label="%s (median %.2f)" % (n, np.median(d["chi2ndf"])))
    a.axvline(1.0, color="0.6", ls=":")
    a.set_xscale("log"); a.set_xlabel("bundle chi2/ndf"); a.set_ylabel("density")
    a.set_title("Match chi2/ndf (recipe: pred-based error)"); a.legend()

    # 2. ks
    a = ax[0, 1]
    for n, d in [("DATA", D), ("MC", M)]:
        a.hist(d["ks"], bins=np.linspace(0, 0.5, 40), histtype="step", lw=1.8, color=C[n],
               density=True, label="%s (median %.3f)" % (n, np.median(d["ks"])))
    a.set_xlabel("bundle ks_dis (shape)"); a.set_ylabel("density")
    a.set_title("KS shape distance (scale-invariant -> ~unchanged)"); a.legend()

    # 3. meas/pred per flash
    a = ax[1, 0]
    for n, d in [("DATA", D), ("MC", M)]:
        a.hist(d["mp"], bins=np.linspace(0, 3, 40), histtype="step", lw=1.8, color=C[n],
               density=True, label="%s (median %.2f)" % (n, np.median(d["mp"])))
    a.axvline(1.0, color="0.6", ls=":")
    a.set_xlabel("meas / pred  per flash"); a.set_ylabel("density")
    a.set_title("Normalization: data scaled to ~1; MC unscaled (under-predicts)"); a.legend()

    # 4. chi2/ndf vs ks
    a = ax[1, 1]
    for n, d in [("DATA", D), ("MC", M)]:
        a.scatter(d["ks"], np.clip(d["chi2ndf"], 0.1, 300), s=18, alpha=0.6, color=C[n], label=n)
    a.set_yscale("log"); a.set_xlabel("ks_dis"); a.set_ylabel("chi2/ndf")
    a.set_title("Joint match quality (low-left = good)"); a.legend()

    fig.suptitle("SBND Q/L hand-scan match quality: DATA vs MC, after the data recipe "
                 "(QtoL 0.86 data; sigma^2 = meas + max(5 PE, 0.25 pred)^2)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(PICS, "ql_recipe_data_vs_mc.png")
    fig.savefig(out, dpi=110); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()

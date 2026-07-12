#!/usr/bin/env python3
"""Which ANN doping is the PDFastSimPVS voxel library more consistent with?

Both the voxel library (extract_photlib.py) and the ANN (sample_ann.py) exist
in two dopings -- Argon/128nm and Xenon/175nm -- for the v4 geometry. This
script builds the full 2x2 consistency matrix {Ar-lib, Xe-lib} x {128nm-ANN,
175nm-ANN} at the library's own 15625 voxel centers, plus a per-channel
breakdown that isolates the channels where the two dopings are supposed to
disagree (the "Ar-blind" channels 16/29/39 -- eff_Ar=0, eff_Xe>0 in
pdvd-photlib-chanmap.json).

Reuses (does not regenerate) the existing pipeline outputs:
  work/photlib_vis_{Ar,Xe}.npy   -- extract_photlib.py
  work/photlib_grid.json         -- extract_photlib.py (v4 grid, mm)
  work/ann_v4_at_voxels.npy      -- sample_ann.py stage_checkv4 (v4 128nm ANN)
Only new computation: the v4 175nm ANN at the same voxel centers (cached to
work/ann_v4_175nm_at_voxels.npy so TF only runs once).

Run inside the TF venv:  /home/xqian/tmp/tfvenv/bin/python compare_lib_ann.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
PICS = os.path.join(HERE, "pics")
NCH = 40


def voxel_centers_cm():
    gj = json.load(open(os.path.join(WORK, "photlib_grid.json")))
    n, org, stp = gj["n"], np.array(gj["origin_mm"]), np.array(gj["step_mm"])
    ids = np.arange(n[0] * n[1] * n[2])
    ix, iy, iz = ids % n[0], (ids // n[0]) % n[1], ids // (n[0] * n[1])
    return (org + (np.stack([ix, iy, iz], 1) + 0.5) * stp) / 10.0  # mm -> cm


def get_ann_175nm(centers_cm):
    cache = os.path.join(WORK, "ann_v4_175nm_at_voxels.npy")
    if os.path.exists(cache):
        print(f"  reusing cached {cache}")
        return np.load(cache)
    from sample_ann import load_model
    print("  sampling v4 175nm ANN at the 15625 voxel centers (TF)...")
    run = load_model("protodune_vd_v4_175nm_tf2.6")
    ann = run(centers_cm)
    np.save(cache, ann)
    print(f"  wrote {cache}")
    return ann


def pair_stats(lib, ann):
    both = (lib > 1e-8) & (ann > 1e-8)
    n = int(both.sum())
    llib, lann = np.log(lib[both]), np.log(ann[both])
    r = float(np.corrcoef(llib, lann)[0, 1])
    ratio = ann[both] / lib[both]
    q16, q50, q84 = np.percentile(ratio, [16, 50, 84])
    rms = float(np.sqrt(np.mean((lann - llib) ** 2)))
    return {"n_entries": n, "log_corr": r, "ratio_p16": float(q16),
            "ratio_p50": float(q50), "ratio_p84": float(q84),
            "log_ratio_rms": rms}, both, (lann - llib)


def main():
    os.makedirs(PICS, exist_ok=True)
    print("== voxel centers + inputs")
    centers = voxel_centers_cm()
    ar = np.load(os.path.join(WORK, "photlib_vis_Ar.npy"))
    xe = np.load(os.path.join(WORK, "photlib_vis_Xe.npy"))
    ann128 = np.load(os.path.join(WORK, "ann_v4_at_voxels.npy"))
    ann175 = get_ann_175nm(centers)

    libs = {"Ar": ar, "Xe": xe}
    anns = {"128nm": ann128, "175nm": ann175}

    print("\n== 2x2 global consistency (log-visibility, shared support lib>0 & ann>0)")
    matrix = {}
    diffs = {}
    masks = {}
    for lname, lib in libs.items():
        for aname, ann in anns.items():
            stats, mask, diff = pair_stats(lib, ann)
            matrix[f"{lname}_lib_vs_{aname}_ann"] = stats
            diffs[(lname, aname)] = diff
            masks[(lname, aname)] = mask
            print(f"  {lname} lib vs {aname} ANN: n={stats['n_entries']:6d}  "
                  f"log-corr={stats['log_corr']:.3f}  "
                  f"ratio 16/50/84%={stats['ratio_p16']:.2f}/{stats['ratio_p50']:.2f}/{stats['ratio_p84']:.2f}  "
                  f"log-ratio RMS={stats['log_ratio_rms']:.3f}")

    verdict = {}
    for lname in libs:
        corr128 = matrix[f"{lname}_lib_vs_128nm_ann"]["log_corr"]
        corr175 = matrix[f"{lname}_lib_vs_175nm_ann"]["log_corr"]
        rms128 = matrix[f"{lname}_lib_vs_128nm_ann"]["log_ratio_rms"]
        rms175 = matrix[f"{lname}_lib_vs_175nm_ann"]["log_ratio_rms"]
        by_corr = "128nm" if corr128 > corr175 else "175nm"
        by_rms = "128nm" if rms128 < rms175 else "175nm"
        verdict[lname] = {"closer_by_corr": by_corr, "closer_by_rms": by_rms,
                           "corr_128nm": corr128, "corr_175nm": corr175,
                           "rms_128nm": rms128, "rms_175nm": rms175}
        print(f"\n  {lname} library is closer to: corr says {by_corr}, RMS says {by_rms} "
              f"(corr 128={corr128:.4f} vs 175={corr175:.4f}; "
              f"RMS 128={rms128:.4f} vs 175={rms175:.4f})")

    # --- per-(voxel,ch) discriminator: for entries seen by BOTH libraries and
    # BOTH ANNs, which ANN flavor is each library closer to in |log-ratio|?
    print("\n== per-entry discriminator (common support across all 4 pairs)")
    common = masks[("Ar", "128nm")] & masks[("Ar", "175nm")] & masks[("Xe", "128nm")] & masks[("Xe", "175nm")]
    disc = {}
    for lname in libs:
        d128 = np.abs(diffs[(lname, "128nm")][ _reindex(masks[(lname, "128nm")], common) ])
        d175 = np.abs(diffs[(lname, "175nm")][ _reindex(masks[(lname, "175nm")], common) ])
        prefers_128 = float(np.mean(d128 < d175))
        disc[lname] = prefers_128
        print(f"  {lname} lib: {prefers_128*100:.1f}% of common entries closer to 128nm ANN, "
              f"{(1-prefers_128)*100:.1f}% closer to 175nm ANN")

    # --- per-channel median ratio (native channel index), flags the known
    # Ar-blind channels (16, 29, 39: eff_Ar=0, eff_Xe>0 per the official map)
    print("\n== per-channel median ANN/lib ratio (native channel index)")
    perchan = {"Ar_vs_128nm": [], "Ar_vs_175nm": [], "Xe_vs_128nm": [], "Xe_vs_175nm": []}
    ar_blind = [16, 29, 39]
    for ch in range(NCH):
        row = {}
        for lname, lib in libs.items():
            for aname, ann in anns.items():
                l, a = lib[:, ch], ann[:, ch]
                m = (l > 1e-8) & (a > 1e-8)
                med = float(np.median(a[m] / l[m])) if m.sum() >= 5 else None
                perchan[f"{lname}_vs_{aname}"].append(med)
                row[f"{lname}_vs_{aname}"] = med
        flag = " <- Ar-blind (eff_Ar=0)" if ch in ar_blind else ""
        print(f"  ch{ch:2d}: Ar/128={_fmt(row['Ar_vs_128nm'])} Ar/175={_fmt(row['Ar_vs_175nm'])} "
              f"Xe/128={_fmt(row['Xe_vs_128nm'])} Xe/175={_fmt(row['Xe_vs_175nm'])}{flag}")

    summary = {"matrix": matrix, "verdict": verdict, "discriminator_frac_prefers_128nm": disc,
               "ar_blind_channels": ar_blind, "per_channel_median_ratio": perchan}
    out = os.path.join(WORK, "lib_ann_compare.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {out}")

    make_plots(libs, anns, matrix)


def _reindex(mask_pair, common):
    """Index array selecting, within the entries kept by mask_pair (True/False
    over the full array), those that are also in `common`."""
    # positions (in the compact 'both' array) that correspond to `common`
    return common[mask_pair]


def _fmt(v):
    return f"{v:.2f}" if v is not None else " n/a"


def make_plots(libs, anns, matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n== plots")
    lnames = list(libs.keys())
    anames = list(anns.keys())

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    for i, lname in enumerate(lnames):
        for j, aname in enumerate(anames):
            ax = axes[i, j]
            lib, ann = libs[lname], anns[aname]
            m = (lib > 1e-8) & (ann > 1e-8)
            x, y = np.log10(lib[m]), np.log10(ann[m])
            ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
            lo, hi = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
            ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
            s = matrix[f"{lname}_lib_vs_{aname}_ann"]
            ax.set_title(f"{lname} lib vs {aname} ANN\n"
                         f"corr={s['log_corr']:.3f}  ratio50={s['ratio_p50']:.2f}", fontsize=10)
            if i == 1:
                ax.set_xlabel(f"log10({lname} lib visibility)")
            if j == 0:
                ax.set_ylabel("log10(ANN visibility)")
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("PDVD v4 voxel library vs ANN, per-doping consistency (15625 voxels x 40 ch)")
    fig.tight_layout()
    p1 = os.path.join(PICS, "lib_vs_ann_scatter.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"  wrote {p1}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for i, lname in enumerate(lnames):
        for aname in anames:
            lib, ann = libs[lname], anns[aname]
            m = (lib > 1e-8) & (ann > 1e-8)
            lr = np.log10(ann[m] / lib[m])
            ax[i].hist(lr, bins=100, range=(-2, 2), histtype="step", lw=1.6,
                       label=f"vs {aname} ANN", density=True)
        ax[i].axvline(0, color="k", lw=0.8, ls=":")
        ax[i].set_title(f"{lname} library")
        ax[i].set_xlabel("log10(ANN / library)")
        ax[i].legend(fontsize=9)
    ax[0].set_ylabel("density")
    fig.suptitle("log-ratio distributions: which ANN doping does each library center on?")
    fig.tight_layout()
    p2 = os.path.join(PICS, "lib_vs_ann_ratio.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print(f"  wrote {p2}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    chans = np.arange(NCH)
    for lname in lnames:
        for aname in anames:
            lib, ann = libs[lname], anns[aname]
            meds = []
            for ch in range(NCH):
                l, a = lib[:, ch], ann[:, ch]
                m = (l > 1e-8) & (a > 1e-8)
                meds.append(np.median(a[m] / l[m]) if m.sum() >= 5 else np.nan)
            ax.plot(chans, meds, marker="o", ms=3, lw=1, label=f"{lname} lib vs {aname} ANN")
    for ch in (16, 29, 39):
        ax.axvline(ch, color="gray", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("library/ANN native channel index (dashed = Ar-blind channels 16,29,39)")
    ax.set_ylabel("median ANN/library ratio")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Per-channel ANN/library visibility ratio")
    fig.tight_layout()
    p3 = os.path.join(PICS, "lib_ann_perchannel.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    print(f"  wrote {p3}")


if __name__ == "__main__":
    main()

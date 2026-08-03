#!/usr/bin/env python3
"""SBND matched Q-L light: measured vs predicted PE (data vs simulation).

The charge-light matcher (QLMatching) writes, per event, a Bee optical JSON
<n>-op.json inside mabc.zip.  Each flash row carries:
    op_t            flash time (us)
    op_peTotal      total measured PE
    op_pes[312]     measured PE per optical channel
    op_pes_pred[312]predicted PE per channel (filled only for MATCHED flashes,
                    and only on the active PMT channels of that flash's TPC side)
    op_cluster_ids  matched cluster ids (empty => unmatched flash)
    apa             "0"/"1", the flash's TPC side

A MATCHED bundle = a row with non-empty op_cluster_ids (=> op_pes_pred filled).

For every matched bundle we compare measured vs predicted light over the active
channel set A = {ch : op_pes_pred[ch] > 0} (self-consistent: we only compare
where the model predicted).  We compute:
    meas  = sum_A op_pes          pred = sum_A op_pes_pred
    ratio = meas / pred           (normalization: how much brighter measured is)
    cos   = <m,p> / (|m||p|)      (pattern similarity over A; the user's metric
                                   for "reasonably matched / pattern similar")

Outputs:
  ql_light_dump.csv   one row per matched bundle
  ql_perpmt_dump.csv  one row per (mode, channel): summed meas/pred over bundles
  pics/ql_*.png       the 5 comparison figures

Usage:
    python3 ql_light_compare.py [--cos-cut 0.9]
"""
import argparse
import json
import os
import re
import zipfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PICS = os.path.join(HERE, "pics")
BUNDLE_DUMP = os.path.join(HERE, "ql_light_dump.csv")
PERPMT_DUMP = os.path.join(HERE, "ql_perpmt_dump.csv")

# mode -> (Bee zip, real event ids in streaming/dir-index order)
MODES = {
    "mc":   (os.path.join(HERE, "work/ql_mc/mabc.zip"),
             [2, 9, 11, 12, 14, 18, 31, 35, 41, 42]),
    "data": (os.path.join(HERE, "work/ql_data/mabc.zip"),
             [686, 1258, 1302, 1346, 1698,
              1720, 1808, 1852, 2028, 2050]),
}

NCHAN = 312
OP_MEMBER_RE = re.compile(r"data/(\d+)/\d+-op\.json$")
COLORS = {"mc": "C0", "data": "C3"}


def load_matched_bundles(mode):
    """Return list of dicts, one per matched bundle, plus per-channel sums.

    Per-channel accumulators (meas_sum, pred_sum, nbundles, apa_side) are over
    the active set of every matched bundle.
    """
    zpath, event_ids = MODES[mode]
    if not os.path.isfile(zpath):
        raise SystemExit("missing matcher output: %s (run ./run_clust_QL_evt.sh %s)"
                         % (zpath, mode))

    bundles = []
    meas_sum = np.zeros(NCHAN)
    pred_sum = np.zeros(NCHAN)
    nbundles = np.zeros(NCHAN, dtype=int)
    apa_side = np.full(NCHAN, -1, dtype=int)

    with zipfile.ZipFile(zpath) as z:
        members = sorted(n for n in z.namelist() if OP_MEMBER_RE.search(n))
        for name in members:
            idx = int(OP_MEMBER_RE.search(name).group(1))
            ev_id = event_ids[idx] if idx < len(event_ids) else -1
            d = json.loads(z.read(name))
            n = len(d["op_t"])
            for i in range(n):
                if not d["op_cluster_ids"][i]:
                    continue  # unmatched flash
                pred = np.asarray(d["op_pes_pred"][i], dtype=float)
                meas = np.asarray(d["op_pes"][i], dtype=float)
                if pred.size != NCHAN or meas.size != NCHAN:
                    continue
                active = pred > 0.0
                if not active.any():
                    continue
                m = meas[active]
                p = pred[active]
                meas_pe = float(m.sum())
                pred_pe = float(p.sum())
                denom = np.linalg.norm(m) * np.linalg.norm(p)
                cos = float(m.dot(p) / denom) if denom > 0 else 0.0
                apa = int(d["apa"][i])
                bundles.append(dict(
                    mode=mode, event_idx=idx, event_id=ev_id, apa=apa,
                    flash_t=float(d["op_t"][i]), n_active=int(active.sum()),
                    meas_pe=meas_pe, pred_pe=pred_pe,
                    ratio=meas_pe / pred_pe, cos=cos))
                meas_sum[active] += m
                pred_sum[active] += p
                nbundles[active] += 1
                apa_side[active] = apa
    return bundles, meas_sum, pred_sum, nbundles, apa_side


def write_bundle_csv(all_bundles):
    cols = ["mode", "event_idx", "event_id", "apa", "flash_t_us",
            "n_active_ch", "meas_pe", "pred_pe", "ratio", "cos_sim"]
    with open(BUNDLE_DUMP, "w") as f:
        f.write(",".join(cols) + "\n")
        for b in all_bundles:
            f.write("%s,%d,%d,%d,%.6g,%d,%.6g,%.6g,%.6g,%.6g\n" % (
                b["mode"], b["event_idx"], b["event_id"], b["apa"], b["flash_t"],
                b["n_active"], b["meas_pe"], b["pred_pe"], b["ratio"], b["cos"]))
    print("wrote", BUNDLE_DUMP, "(%d matched bundles)" % len(all_bundles))


def write_perpmt_csv(perpmt):
    with open(PERPMT_DUMP, "w") as f:
        f.write("mode,channel,apa_side,meas_sum,pred_sum,n_bundles\n")
        for mode, (meas_sum, pred_sum, nbundles, apa_side) in perpmt.items():
            for ch in range(NCHAN):
                if nbundles[ch] == 0:
                    continue
                f.write("%s,%d,%d,%.6g,%.6g,%d\n" % (
                    mode, ch, apa_side[ch], meas_sum[ch], pred_sum[ch],
                    nbundles[ch]))
    print("wrote", PERPMT_DUMP)


def stats(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3, 0
    return (np.median(v), np.percentile(v, 16), np.percentile(v, 84)), v.size


# ---------------------------------------------------------------- plots

def plot_event_apa_sumpe(all_bundles):
    """req 1: per (event, APA) summed measured vs predicted, MC vs data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mode in zip(axes, ("mc", "data")):
        agg = {}  # (event_idx, apa) -> [meas, pred]
        for b in all_bundles:
            if b["mode"] != mode:
                continue
            k = (b["event_idx"], b["apa"])
            agg.setdefault(k, [0.0, 0.0])
            agg[k][0] += b["meas_pe"]
            agg[k][1] += b["pred_pe"]
        for apa, mk in ((0, "o"), (1, "s")):
            xs = [v[1] for (e, a), v in agg.items() if a == apa]
            ys = [v[0] for (e, a), v in agg.items() if a == apa]
            ax.scatter(xs, ys, marker=mk, label="APA%d" % apa, alpha=0.8)
        lim = [50, 5e5]
        ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
               xlabel="predicted PE (sum)", ylabel="measured PE (sum)",
               title="%s: per event / APA" % mode.upper())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(PICS, "ql_event_apa_sumPE.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def plot_perpmt_sumpe(perpmt):
    """req 2: per-PMT summed measured vs predicted, MC vs data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mode in zip(axes, ("mc", "data")):
        meas_sum, pred_sum, nbundles, apa_side = perpmt[mode]
        sel = nbundles > 0
        for apa, c in ((0, "C0"), (1, "C1")):
            m = sel & (apa_side == apa)
            ax.scatter(pred_sum[m], meas_sum[m], s=18, c=c, alpha=0.7,
                       label="APA%d side" % apa)
        lim = [1, 1e6]
        ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
               xlabel="predicted PE (sum over bundles)",
               ylabel="measured PE (sum over bundles)",
               title="%s: per PMT channel" % mode.upper())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(PICS, "ql_perpmt_sumPE.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def plot_cosine_dist(all_bundles, cos_cut):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 41)
    for mode in ("mc", "data"):
        c = [b["cos"] for b in all_bundles if b["mode"] == mode]
        ax.hist(c, bins=bins, histtype="step", lw=1.8, color=COLORS[mode],
                label="%s (N=%d)" % (mode.upper(), len(c)))
    ax.axvline(cos_cut, color="k", ls="--", lw=1,
               label="cut = %.2f" % cos_cut)
    ax.set(xlabel="cosine similarity (measured vs predicted pattern)",
           ylabel="matched bundles",
           title="Pattern similarity of matched Q-L bundles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PICS, "ql_cosine_dist.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def plot_ratio_dist(all_bundles, cos_cut):
    """req 5: normalization (meas/pred) for reasonably-matched bundles."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.logspace(np.log10(0.1), np.log10(300), 50)
    txt = []
    for mode in ("mc", "data"):
        kept = [b["ratio"] for b in all_bundles
                if b["mode"] == mode and b["cos"] >= cos_cut]
        allr = [b["ratio"] for b in all_bundles if b["mode"] == mode]
        ax.hist(kept, bins=bins, histtype="stepfilled", alpha=0.35,
                color=COLORS[mode])
        ax.hist(kept, bins=bins, histtype="step", lw=1.8, color=COLORS[mode],
                label="%s, cos>=%.2f (N=%d)" % (mode.upper(), cos_cut, len(kept)))
        (med, lo, hi), _ = stats(kept)
        (amed, alo, ahi), _ = stats(allr)
        ax.axvline(med, color=COLORS[mode], ls="--", lw=1)
        txt.append("%s  reasonable: med=%.2f  [%.2f, %.2f]   (all: med=%.2f)"
                   % (mode.upper(), med, lo, hi, amed))
    ax.set(xscale="log", xlabel="measured / predicted PE  (ratio)",
           ylabel="matched bundles",
           title="Normalization: measured / predicted light")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    out = os.path.join(PICS, "ql_ratio_dist.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def plot_ratio_vs_cos(all_bundles, cos_cut):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mode in ("mc", "data"):
        xs = [b["cos"] for b in all_bundles if b["mode"] == mode]
        ys = [b["ratio"] for b in all_bundles if b["mode"] == mode]
        ax.scatter(xs, ys, s=20, c=COLORS[mode], alpha=0.6, label=mode.upper())
    ax.axvline(cos_cut, color="k", ls="--", lw=1, label="cut = %.2f" % cos_cut)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set(yscale="log", xlabel="cosine similarity",
           ylabel="measured / predicted PE",
           title="Normalization vs pattern similarity")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(PICS, "ql_ratio_vs_cos.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cos-cut", type=float, default=0.85,
                    help="cosine-similarity threshold for 'reasonably matched' "
                         "(default 0.85)")
    args = ap.parse_args()

    os.makedirs(PICS, exist_ok=True)

    all_bundles = []
    perpmt = {}
    for mode in ("mc", "data"):
        bundles, meas_sum, pred_sum, nbundles, apa_side = load_matched_bundles(mode)
        all_bundles.extend(bundles)
        perpmt[mode] = (meas_sum, pred_sum, nbundles, apa_side)
        print("%-4s: %d matched bundles, %d active channels"
              % (mode, len(bundles), int((nbundles > 0).sum())))

    write_bundle_csv(all_bundles)
    write_perpmt_csv(perpmt)

    plot_event_apa_sumpe(all_bundles)
    plot_perpmt_sumpe(perpmt)
    plot_cosine_dist(all_bundles, args.cos_cut)
    plot_ratio_dist(all_bundles, args.cos_cut)
    plot_ratio_vs_cos(all_bundles, args.cos_cut)

    # console summary
    print("\n--- normalization summary (ratio = measured / predicted) ---")
    for mode in ("mc", "data"):
        kept = [b["ratio"] for b in all_bundles
                if b["mode"] == mode and b["cos"] >= args.cos_cut]
        (med, lo, hi), n = stats(kept)
        print("%-4s  cos>=%.2f  N=%-3d  median=%.2f  16-84%%=[%.2f, %.2f]"
              % (mode, args.cos_cut, n, med, lo, hi))


if __name__ == "__main__":
    main()

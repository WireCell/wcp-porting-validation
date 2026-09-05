#!/usr/bin/env python3
"""doc pdvd/42 -- dQ/dx vs residual range figures, PDVD against SBND, from the
d42_dqdx_rr.py outputs (points / tracks / summary TSVs) and the two reference
tables.

Usage:
  d42_dqdx_plots.py --pdvd figs/42_dqdx_pdvd --pdvd-ref stm/pdvd_ref_dqdx_045.json:MuonDeDx \
      --sbnd figs/42_dqdx_sbnd --sbnd-ref ../sbnd/sbnd_xin/nusel_display/stm_ref_dqdx.json:MuonDeDxBox \
      --out figs/42 [--tier all_status0]
Figures:
  <out>_dqdx_rr_2d.png     per detector: 2-D histogram of dQ/dx vs rr (kink-anchored, leftover removed) with the muon table
  <out>_dqdx_rr_ratio.png  per-bin median data/(k_pop table) for each tier, both detectors
  <out>_dqdx_tracks.png    per-track k, shape rms, Bragg contrast, median reduced chi2 distributions
  <out>_dqdx_anchor.png    kink-anchored vs end-anchored profile (the Michel/leftover removal), per detector
Prints the summary tables as markdown.
"""
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
COLS = {"pdvd": "tab:red", "sbnd": "tab:blue"}


def read_tsv(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    hdr = rows[0]; out = {h: [] for h in hdr}
    for r in rows[1:]:
        for h, v in zip(hdr, r): out[h].append(v)
    def col(h, f=float):
        if f is str: return np.array(out[h])
        return np.array([float(v) if v not in ("nan", "") else np.nan for v in out[h]])
    return out, col


def load_ref(spec):
    path, key = spec.rsplit(":", 1)
    t = json.load(open(path))[key]
    x = t["start"] + t["step"] * np.arange(len(t["values"])); y = np.asarray(t["values"], float)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", required=True); ap.add_argument("--pdvd-ref", required=True)
    ap.add_argument("--sbnd", required=True); ap.add_argument("--sbnd-ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tier", default="all_status0")
    a = ap.parse_args()
    D = {}
    for det, pre, refspec in (("pdvd", a.pdvd, a.pdvd_ref), ("sbnd", a.sbnd, a.sbnd_ref)):
        _, pc = read_tsv(pre + "_points.tsv"); _, tc = read_tsv(pre + "_tracks.tsv")
        sraw, sc = read_tsv(pre + "_summary.tsv")
        D[det] = dict(rr=pc("rr_kink"), rre=pc("rr_end"), v=pc("dqdx"), past=pc("past_kink"), ev=pc("event", str), blk=pc("block"),
                      tk=dict(k=tc("k"), shape=tc("shape"), contrast=tc("contrast"), chi2=tc("med_chi2"), hasleft=tc("has_left"),
                              leftL=tc("left_L"), npts=tc("npts"), doc55=tc("doc55"), ev=tc("event", str), blk=tc("block")),
                      summ=(sraw, sc), ref=load_ref(refspec))

    # ---- table: tiers
    print("| det | tier | tracks | points | k_pop | chi2/nbins | k med +- rms | contrast med | leftover frac |")
    print("|---|---|---|---|---|---|---|---|---|")
    for det in ("pdvd", "sbnd"):
        sraw, sc = D[det]["summ"]
        for i, tier in enumerate(sraw["tier"]):
            if sraw["ntracks"][i] == "0": print("| %s | %s | 0 | | | | | | |" % (det, tier)); continue
            print("| %s | %s | %s | %s | %s | %.1f/%s | %s +- %s | %s | %s |" % (det, tier, sraw["ntracks"][i], sraw["npoints"][i], sraw["k_pop"][i],
                  float(sraw["chi2"][i]), sraw["nbins"][i], sraw["k_med"][i], sraw["k_rms"][i], sraw["contrast_med"][i], sraw["has_left_frac"][i]))
    print("\nper-bin median data/(k_pop x table), tier %s:" % a.tier)
    print("| det | " + " | ".join("%d-%d" % b for b in BINS) + " |"); print("|---|" + "---|" * len(BINS))
    for det in ("pdvd", "sbnd"):
        sraw, sc = D[det]["summ"]; i = sraw["tier"].index(a.tier)
        print("| %s | " % det + " | ".join("%s +- %s" % (sraw["r_%d_%d" % b][i], sraw["e_%d_%d" % b][i]) for b in BINS) + " |")

    # ---- fig 1: 2-D hist with table
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for k_, det in enumerate(("pdvd", "sbnd")):
        d = D[det]; m = (d["past"] == 0) & np.isfinite(d["v"]) & (d["v"] > 0)
        if a.tier != "all_status0":
            tk = d["tk"]
            sel = {"contrast_ge2": tk["contrast"] >= 2.0, "doc55_cuts": tk["doc55"] == 1,
                   "doc55_muon": (tk["doc55"] == 1) & (tk["k"] >= 0.85) & (tk["k"] <= 1.25)}[a.tier]
            keys = set(zip(tk["ev"][sel], tk["blk"][sel]))
            m &= np.array([(e, b) in keys for e, b in zip(d["ev"], d["blk"])])
        h = ax[k_].hist2d(d["rr"][m], d["v"][m] / 1e3, bins=[np.linspace(0, 60, 61), np.linspace(0, 250, 101)], cmap="Greys", cmin=1)
        x, y = d["ref"]; ax[k_].plot(x, y / 1e3, "-", color=COLS[det], lw=2, label="muon table (config)")
        # binned medians
        cen = [(lo + hi) / 2 for lo, hi in BINS]; med = []
        for lo, hi in BINS:
            mm = m & (d["rr"] >= lo) & (d["rr"] < hi); med.append(np.median(d["v"][mm]) / 1e3 if mm.sum() >= 5 else np.nan)
        ax[k_].plot(cen, med, "o", color="k", ms=5, label="median per bin")
        ax[k_].set_xlabel("residual range from the tagger's stop (cm)"); ax[k_].set_ylabel("dQ/dx (ke/cm)"); ax[k_].set_ylim(0, 250)
        ax[k_].set_title("%s: %d points, %d accepted passes (%s)" % (det.upper(), m.sum(), len(set(zip(d["ev"][m], d["blk"][m]))), a.tier)); ax[k_].legend()
    fig.tight_layout(); fig.savefig(a.out + "_dqdx_rr_2d.png", dpi=110); plt.close(fig)

    # ---- fig 2: ratio per bin per tier
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    cen = np.array([(lo + hi) / 2 for lo, hi in BINS])
    for j, tier in enumerate(("all_status0", "contrast_ge2", "doc55_cuts", "doc55_muon")):
        for det in ("pdvd", "sbnd"):
            sraw, sc = D[det]["summ"]
            if tier not in sraw["tier"]: continue
            i = sraw["tier"].index(tier)
            if sraw["ntracks"][i] == "0": continue
            r = np.array([float(sraw["r_%d_%d" % b][i]) for b in BINS]); e = np.array([float(sraw["e_%d_%d" % b][i]) for b in BINS])
            ax[j].errorbar(cen, r, yerr=e, fmt="o-", color=COLS[det], capsize=2,
                           label="%s n=%s k_pop=%s chi2=%.1f/%s" % (det.upper(), sraw["ntracks"][i], sraw["k_pop"][i], float(sraw["chi2"][i]), sraw["nbins"][i]))
        ax[j].axhline(1, color="k", lw=0.6); ax[j].set_ylim(0.6, 1.4); ax[j].set_title(tier); ax[j].set_xlabel("residual range (cm)"); ax[j].legend(fontsize=8)
    ax[0].set_ylabel("median dQ/dx / (k_pop x muon table)")
    fig.tight_layout(); fig.savefig(a.out + "_dqdx_rr_ratio.png", dpi=110); plt.close(fig)

    # ---- fig 3: per-track distributions
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for k_, (key, lab, rng) in enumerate((("k", "per-track scale k vs muon table", (0, 2.5)), ("shape", "shape rms of log(median/(k table))", (0, 1.0)),
                                          ("contrast", "Bragg contrast med(rr<2)/med(20-40)", (0, 5)), ("chi2", "median reduced chi2 of the fit", (0, 8)))):
        for det in ("pdvd", "sbnd"):
            v = D[det]["tk"][key]; v = v[np.isfinite(v)]
            ax[k_].hist(np.clip(v, rng[0], rng[1]), bins=40, range=rng, histtype="step", lw=1.6, color=COLS[det], density=True,
                        label="%s n=%d med %.2f" % (det.upper(), len(v), np.median(v) if len(v) else np.nan))
        ax[k_].set_xlabel(lab); ax[k_].legend(fontsize=8)
    fig.suptitle("STM accepted passes: per-track dQ/dx-vs-rr quantities"); fig.tight_layout(); fig.savefig(a.out + "_dqdx_tracks.png", dpi=110); plt.close(fig)

    # ---- fig 4: anchor comparison
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k_, det in enumerate(("pdvd", "sbnd")):
        d = D[det]; x, y = d["ref"]
        for anchor, lab, sty in (("rr", "kink-anchored, leftover removed", "o-"), ("rre", "end-anchored (persist_stm_fit rr)", "s--")):
            m = np.isfinite(d["v"]) & (d["v"] > 0) & ((d["past"] == 0) if anchor == "rr" else True)
            cen = []; med = []; lo_ = []; hi_ = []
            for lo, hi in BINS:
                mm = m & (d[anchor] >= lo) & (d[anchor] < hi)
                if mm.sum() < 5: continue
                cen.append((lo + hi) / 2); med.append(np.median(d["v"][mm]) / 1e3); lo_.append(np.percentile(d["v"][mm], 25) / 1e3); hi_.append(np.percentile(d["v"][mm], 75) / 1e3)
            ax[k_].errorbar(cen, med, yerr=[np.array(med) - np.array(lo_), np.array(hi_) - np.array(med)], fmt=sty, capsize=2, label=lab, alpha=0.85)
        ax[k_].plot(x, y / 1e3, "k-", lw=1.5, label="muon table (config)")
        nleft = int(np.sum(d["tk"]["hasleft"])); ax[k_].set_title("%s: %d of %d accepted passes carry a leftover past the kink (median left_L %.1f cm)" % (
            det.upper(), nleft, len(d["tk"]["hasleft"]), np.median(d["tk"]["leftL"][d["tk"]["hasleft"] == 1]) if nleft else np.nan), fontsize=9)
        ax[k_].set_xlabel("residual range (cm)"); ax[k_].set_ylabel("median dQ/dx (ke/cm), IQR bars"); ax[k_].set_ylim(0, 200); ax[k_].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out + "_dqdx_anchor.png", dpi=110); plt.close(fig)
    print("\nwrote", a.out + "_dqdx_{rr_2d,rr_ratio,tracks,anchor}.png")


if __name__ == "__main__":
    sys.exit(main())

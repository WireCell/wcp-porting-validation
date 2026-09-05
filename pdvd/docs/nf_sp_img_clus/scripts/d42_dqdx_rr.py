#!/usr/bin/env python3
"""doc pdvd/42 -- dQ/dx vs residual range of the STM-ACCEPTED passes against the
detector's own muon expectation table, PDVD and SBND through one code path.

Reads tracking-stm.root (T_rec_charge / T_stm_pass / Trun).  Per accepted pass
(status 0):
  dQ/dx = ((q - dQdx_offset)/dQdx_scale) / nq          e/cm  (nq is dx in cm)
  Michel / leftover removal: the tagger's own stop is the kink.  When the pass
  carries a leftover (left_L > 0, kink_num inside the path) the points PAST the
  kink are dropped and rr is re-anchored there: rr_kink = rr - rr[kink_num].
  The end-anchored rr (rr_end, the persist_stm_fit convention) is kept for the
  cross-check.  Points with |x| > --max-abs-x (PDVD: 305 cm, the near-CRP rise
  of doc 25 sec 13.4) are excluded from every profile.
Per track: binned medians (doc-25 bins), Bragg contrast = med(rr<2)/med(20-40),
(tiers: all status 0; contrast >= 2; the doc-55 five cuts; doc-55 cuts + muon-like
k in [0.85, 1.25], which is how collect_dqdx_rr_sample.py separates protons)
free scale k = geometric mean of median/ref over populated bins, shape rms of
log(median/(k ref)), median reduced_chi2; the doc-55 five-cut tier flag.
Population (per tier): one global scale k_pop = exp(median log(dqdx/ref)) over
all points, then the per-bin median of dqdx/(k_pop ref) with a 1.2533*MAD/sqrt(n)
error and a 3 % systematic floor (doc 25 sec 13.6), chi2 over bins.

Usage:
  d42_dqdx_rr.py --det pdvd --ref stm/pdvd_ref_dqdx_045.json --ref-key MuonDeDx \
      --out figs/42_dqdx_pdvd work/*_d42fit/tracking-stm.root
  d42_dqdx_rr.py --det sbnd --ref ../sbnd/sbnd_xin/nusel_display/stm_ref_dqdx.json --ref-key MuonDeDxBox \
      --max-abs-x 1e9 --out figs/42_dqdx_sbnd work-stmcamp-d42fit/nusel_evt*/tracking-stm.root
Writes <out>_tracks.tsv, <out>_points.tsv, <out>_summary.tsv.
"""
import argparse, json, os, sys
import numpy as np
import uproot

BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
SYS_FLOOR = 0.03


def load_ref(path, key):
    t = json.load(open(path))[key]
    x = t["start"] + t["step"] * np.arange(len(t["values"]))
    y = np.asarray(t["values"], float)
    return lambda rr: np.interp(rr, x, y)   # clamped outside, like LinterpFunction


def binned_median(rr, v):
    out = []
    for lo, hi in BINS:
        m = (rr >= lo) & (rr < hi) & np.isfinite(v)
        out.append((float(np.median(v[m])) if m.sum() >= 3 else float("nan"), int(m.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--det", required=True)
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-key", default="MuonDeDx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-abs-x", type=float, default=305.0)
    ap.add_argument("--min-npts", type=int, default=10)
    a = ap.parse_args()
    ref = load_ref(a.ref, a.ref_key)

    tracks = []; points = []
    for path in a.roots:
        try:
            f = uproot.open(path)
            t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf", "status", "pass", "reduced_chi2"], library="np")
            tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
            sp = f["T_stm_pass"].arrays(library="np")
        except Exception as ex:
            print("skip", path, ex, file=sys.stderr); continue
        ev = os.path.basename(os.path.dirname(path))
        pinfo = {(int(c) * 10 + int(p)): (int(k), float(eL), float(lL))
                 for c, p, k, eL, lL in zip(sp["cluster_id"], sp["pass"], sp["kink_num"], sp["exit_L"], sp["left_L"])}
        for blk in sorted(set(t["ndf"].tolist())):
            mk = t["ndf"] == blk
            if int(t["status"][mk][0]) != 0 or mk.sum() < a.min_npts: continue
            n = int(mk.sum())
            dQ = (t["q"][mk] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
            dx = t["nq"][mk]; x = t["x"][mk]; rr_end = t["rr"][mk]; chi2 = t["reduced_chi2"][mk]
            with np.errstate(divide="ignore", invalid="ignore"):
                dqdx = np.where(dx > 0, dQ / dx, np.nan)
            dqdx = np.where(np.abs(x) > a.max_abs_x, np.nan, dqdx)
            kink, exL, lfL = pinfo.get(int(blk), (-1, float("nan"), 0.0))
            has_left = (lfL > 0) and (0 <= kink < n - 1)
            keep = np.ones(n, bool)
            if has_left:
                keep[kink + 1:] = False
                rr_k = rr_end - rr_end[kink]
            else:
                rr_k = rr_end.copy()
            rr_use = rr_k[keep]; v = dqdx[keep]
            good = np.isfinite(v) & (v > 0)
            bm = binned_median(rr_use[good], v[good])
            meds = np.array([m for m, _ in bm]); cnt = np.array([c for _, c in bm])
            centers = np.array([(lo + hi) / 2 for lo, hi in BINS])
            pop = np.isfinite(meds)
            k = float(np.exp(np.mean(np.log(meds[pop] / ref(centers[pop]))))) if pop.sum() >= 2 else float("nan")
            shape = float(np.sqrt(np.mean(np.log(meds[pop] / (k * ref(centers[pop]))) ** 2))) if pop.sum() >= 2 else float("nan")
            bragg = v[good & (rr_use < 2)]; plat = v[good & (rr_use >= 20) & (rr_use < 40)]
            contrast = float(np.median(bragg) / np.median(plat)) if len(bragg) >= 3 and len(plat) >= 3 else float("nan")
            mchi2 = float(np.nanmedian(chi2[keep])) if keep.sum() else float("nan")
            rrmin = float(rr_use[good].min()) if good.sum() else float("nan"); rrmax = float(rr_use[good].max()) if good.sum() else float("nan")
            doc55 = (n >= 40 and pop.sum() >= 6 and rrmin < 2.0 and rrmax >= 22.0 and contrast >= 2.0 and mchi2 <= 2.5 and shape <= 0.10)
            tracks.append(dict(det=a.det, event=ev, block=int(blk), cluster=int(blk) // 10, npts=n, nkept=int(keep.sum()),
                               length=float(rr_end.max()), kink=kink, exit_L=exL, left_L=lfL, has_left=int(has_left),
                               ngood=int(good.sum()), rrmin=rrmin, rrmax=rrmax, nbins=int(pop.sum()), contrast=contrast,
                               k=k, shape=shape, med_chi2=mchi2, doc55=int(doc55),
                               meds=meds, cnt=cnt))
            for i in range(n):
                points.append((a.det, ev, int(blk), i, "%.2f" % rr_k[i], "%.2f" % rr_end[i], "%.1f" % dqdx[i] if np.isfinite(dqdx[i]) else "nan",
                               "%.3f" % dx[i], "%.2f" % x[i], "%.3f" % chi2[i], int(not keep[i])))

    # --- write tracks / points
    with open(a.out + "_tracks.tsv", "w") as fh:
        fh.write("# doc pdvd/42 d42_dqdx_rr.py det=%s ref=%s:%s max_abs_x=%g files=%d\n" % (a.det, a.ref, a.ref_key, a.max_abs_x, len(a.roots)))
        fh.write("det\tevent\tblock\tcluster\tnpts\tnkept\tlength_cm\tkink\texit_L\tleft_L\thas_left\tngood\trrmin\trrmax\tnbins\tcontrast\tk\tshape\tmed_chi2\tdoc55\t"
                 + "\t".join("med_%d_%d" % b for b in BINS) + "\n")
        for tk in tracks:
            fh.write("\t".join(str(tk[c]) if not isinstance(tk[c], float) else "%.4f" % tk[c] for c in
                              ["det", "event", "block", "cluster", "npts", "nkept", "length", "kink", "exit_L", "left_L", "has_left", "ngood", "rrmin", "rrmax", "nbins", "contrast", "k", "shape", "med_chi2", "doc55"])
                     + "\t" + "\t".join("%.0f" % m if np.isfinite(m) else "nan" for m in tk["meds"]) + "\n")
    with open(a.out + "_points.tsv", "w") as fh:
        fh.write("det\tevent\tblock\ti\trr_kink\trr_end\tdqdx\tdx\tx\treduced_chi2\tpast_kink\n")
        for p in points: fh.write("\t".join(str(v) for v in p) + "\n")

    # --- population summary per tier
    pts = np.array([(float(p[4]), float(p[6]) if p[6] != "nan" else np.nan, p[2], p[1], int(p[10])) for p in points], dtype=object)
    tier_sets = {"all_status0": {(tk["event"], tk["block"]) for tk in tracks},
                 "contrast_ge2": {(tk["event"], tk["block"]) for tk in tracks if tk["contrast"] >= 2.0},
                 "doc55_cuts": {(tk["event"], tk["block"]) for tk in tracks if tk["doc55"]},
                 # collect_dqdx_rr_sample.py assigns the particle by the free scale: muon-like k in [0.85, 1.25]
                 "doc55_muon": {(tk["event"], tk["block"]) for tk in tracks if tk["doc55"] and 0.85 <= tk["k"] <= 1.25}}
    with open(a.out + "_summary.tsv", "w") as fh:
        fh.write("# per tier: ntracks, npoints, k_pop, then per-bin median(dqdx/(k_pop*ref)) +- err (3%% floor), chi2/nbins, per-track k median/rms, contrast median\n")
        fh.write("tier\tntracks\tnpoints\tk_pop\tchi2\tnbins\tk_med\tk_rms\tcontrast_med\thas_left_frac\t" + "\t".join("r_%d_%d" % b for b in BINS) + "\t" + "\t".join("e_%d_%d" % b for b in BINS) + "\n")
        for name, keys in tier_sets.items():
            sel = [i for i, p in enumerate(points) if (p[1], p[2]) in keys and p[10] == 0 and p[6] != "nan"]
            if not sel:
                fh.write("%s\t0\n" % name); continue
            rr = np.array([float(points[i][4]) for i in sel]); v = np.array([float(points[i][6]) for i in sel])
            ok = (v > 0) & (rr >= 0)
            rr, v = rr[ok], v[ok]
            lr = np.log(v / ref(rr))
            k_pop = float(np.exp(np.median(lr)))
            ratios = []; errs = []; chi2 = 0.0; nb = 0
            for lo, hi in BINS:
                m = (rr >= lo) & (rr < hi)
                if m.sum() < 5: ratios.append(float("nan")); errs.append(float("nan")); continue
                r = v[m] / (k_pop * ref(rr[m]))
                med = float(np.median(r)); err = float(1.2533 * 1.4826 * np.median(np.abs(r - med)) / np.sqrt(m.sum()))
                err = float(np.hypot(err, SYS_FLOOR * med))
                ratios.append(med); errs.append(err); chi2 += ((med - 1) / err) ** 2; nb += 1
            tks = [tk for tk in tracks if (tk["event"], tk["block"]) in keys]
            ks = np.array([tk["k"] for tk in tks]); ks = ks[np.isfinite(ks)]
            cs = np.array([tk["contrast"] for tk in tks]); cs = cs[np.isfinite(cs)]
            fh.write("%s\t%d\t%d\t%.4f\t%.2f\t%d\t%.4f\t%.4f\t%.3f\t%.3f\t" % (name, len(tks), len(v), k_pop, chi2, nb,
                     np.median(ks) if len(ks) else float("nan"), np.std(ks) if len(ks) else float("nan"),
                     np.median(cs) if len(cs) else float("nan"), np.mean([tk["has_left"] for tk in tks]))
                     + "\t".join("%.4f" % r for r in ratios) + "\t" + "\t".join("%.4f" % e for e in errs) + "\n")
    print("%s: %d accepted passes from %d files -> %s_{tracks,points,summary}.tsv" % (a.det, len(tracks), len(a.roots), a.out))
    for line in open(a.out + "_summary.tsv"):
        if not line.startswith("#"): print("  " + "\t".join(line.split("\t")[:10]).rstrip())


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc pdvd/115 -- the dQ/dx fit result of a knob arm against its base arm, on the COMMON accepted STM passes
(event, cluster*10+pass with status in --status in BOTH arms), through doc pdvd/50's engine (d50_dqdx_rr_cross.py:
load_ref, scale_and_shape, BINS, SYS_FLOOR imported; the per-track block duplicated so it can run on one record).

Per track (both arms): dQ/dx [e/cm] = (q - dQdx_offset)/dQdx_scale / nq, rr re-anchored at the tagger's kink when
left_L > 0 (Michel / leftover removed), live points = finite dQ/dx > 0; f_low (share of live points below
--low-frac x the muon plateau), the muon-hypothesis free scale k and scale-free SHAPE rms over the populated rr
bins, the Bragg contrast med(rr<2)/med(20-40), the share of q < 0 rows and the Bee-visible holes (>= 3 consecutive
q < 0 rows).  Per population: k_pop = exp(median log(dQ/dx / ref)), chi2 over the 11 rr bins against the
detector's own muon table with the 3 % floor, per-bin median ratios.  Paired per track (arm - base): f_low, shape,
contrast sign counts and medians.

Usage:
  d115_dqdx_compare.py --det pdhd --base d115hoff --arm d115hbwa --out figs/115_dqdx_pdhd_bwa [--status 0]
Writes <out>.txt, <out>.json, <out>_tracks.tsv and <out>.png (ratio to the reference vs rr, both arms).
"""
import argparse, glob, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
from d50_dqdx_rr_cross import load_ref, scale_and_shape, BINS, SYS_FLOOR

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
REF = {"pdhd": f"{IMG}/pdhd/stm/pdhd_ref_dqdx.json", "pdvd": f"{IMG}/pdvd/stm/pdvd_ref_dqdx_045.json"}


def holes(q, minrows=3):
    n = 0; run = 0
    for v in q:
        if v < 0:
            run += 1
        else:
            if run >= minrows: n += 1
            run = 0
    if run >= minrows: n += 1
    return n


def load(d):
    f = uproot.open(f"{d}/tracking-stm.root")
    t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf", "status", "pass", "reduced_chi2"], library="np")
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    sp = f["T_stm_pass"].arrays(library="np")
    pinfo = {(int(c) * 10 + int(p)): (int(k), float(eL), float(lL))
             for c, p, k, eL, lL in zip(sp["cluster_id"], sp["pass"], sp["kink_num"], sp["exit_L"], sp["left_L"])}
    return t, tr, pinfo


def track(t, tr, pinfo, blk, ref, low_thr):
    mk = t["ndf"] == blk
    n = int(mk.sum())
    dQ = (t["q"][mk] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = t["nq"][mk]; rr_end = t["rr"][mk]; qraw = t["q"][mk]
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdx = np.where(dx > 0, dQ / dx, np.nan)
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
    f_low = float((v[good] < low_thr).mean()) if good.sum() else float("nan")
    meds, cen = [], []
    for lo, hi in BINS:
        b = good & (rr_use >= lo) & (rr_use < hi)
        meds.append(float(np.median(v[b])) if b.sum() >= 3 else float("nan")); cen.append((lo + hi) / 2)
    meds = np.array(meds); cen = np.array(cen); pop = np.isfinite(meds)
    k, shape = scale_and_shape(meds[pop], cen[pop], ref) if pop.sum() >= 2 else (float("nan"), float("nan"))
    bragg = v[good & (rr_use < 2)]; plat = v[good & (rr_use >= 20) & (rr_use < 40)]
    contrast = float(np.median(bragg) / np.median(plat)) if len(bragg) >= 3 and len(plat) >= 3 else float("nan")
    return dict(npts=n, nkept=int(keep.sum()), ngood=int(good.sum()), length=float(rr_end.max()) if n else 0.0,
                has_left=int(has_left), f_low=f_low, nbins=int(pop.sum()), k=k, shape=shape, contrast=contrast,
                qneg=int((qraw < 0).sum()), holes=holes(qraw), rr=rr_use[good], v=v[good])


def population(tracks, ref):
    rr = np.concatenate([tk["rr"] for tk in tracks]) if tracks else np.array([])
    v = np.concatenate([tk["v"] for tk in tracks]) if tracks else np.array([])
    ok = (v > 0) & (rr >= 0); rr, v = rr[ok], v[ok]
    out = dict(ntracks=len(tracks), npoints=int(len(v)))
    if len(v) == 0:
        return out
    k_pop = float(np.exp(np.median(np.log(v / ref(rr)))))
    ratios, errs, ns, chi2, nb = [], [], [], 0.0, 0
    for lo, hi in BINS:
        m = (rr >= lo) & (rr < hi); ns.append(int(m.sum()))
        if m.sum() < 5:
            ratios.append(float("nan")); errs.append(float("nan")); continue
        r = v[m] / (k_pop * ref(rr[m])); med = float(np.median(r))
        err = float(1.2533 * 1.4826 * np.median(np.abs(r - med)) / np.sqrt(m.sum())); err = float(np.hypot(err, SYS_FLOOR * med))
        ratios.append(med); errs.append(err); chi2 += ((med - 1) / err) ** 2; nb += 1
    med = lambda key: float(np.nanmedian([tk[key] for tk in tracks]))
    out.update(k_pop=k_pop, chi2=chi2, nbins=nb, ratios=ratios, errs=errs, ns=ns,
               f_low_med=med("f_low"), shape_med=med("shape"), k_med=med("k"), contrast_med=med("contrast"),
               qneg_share=sum(tk["qneg"] for tk in tracks) / max(1, sum(tk["npts"] for tk in tracks)),
               holes_per_10m=sum(tk["holes"] for tk in tracks) / max(1e-9, sum(tk["length"] for tk in tracks) / 1000.0),
               rows=sum(tk["npts"] for tk in tracks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--base", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--status", default="0"); ap.add_argument("--min-npts", type=int, default=10)
    ap.add_argument("--low-frac", type=float, default=0.40); ap.add_argument("--ref")
    a = ap.parse_args()
    ref = load_ref(a.ref or REF[a.det], "MuonDeDx")
    plateau = float(ref(np.array([59.5]))[0]); low_thr = a.low_frac * plateau
    want = {int(s) for s in a.status.split(",")}
    evB = {os.path.basename(p)[:-len(a.base) - 1]: p for p in glob.glob(f"{IMG}/{a.det}/work/*_{a.base}")}
    evA = {os.path.basename(p)[:-len(a.arm) - 1]: p for p in glob.glob(f"{IMG}/{a.det}/work/*_{a.arm}")}
    TA, TB, pairs = [], [], []
    nev = 0
    for ev in sorted(set(evA) & set(evB)):
        try:
            tb, trb, pb = load(evB[ev]); ta, tra, pa = load(evA[ev])
        except Exception as ex:
            print("skip", ev, ex, file=sys.stderr); continue
        nev += 1
        def blocks(t):
            out = {}
            for blk in sorted(set(t["ndf"].tolist())):
                mk = t["ndf"] == blk
                if int(t["status"][mk][0]) in want and mk.sum() >= a.min_npts:
                    out[int(blk)] = True
            return out
        for blk in sorted(set(blocks(tb)) & set(blocks(ta))):
            kb = track(tb, trb, pb, blk, ref, low_thr); ka = track(ta, tra, pa, blk, ref, low_thr)
            kb["event"] = ka["event"] = ev; kb["block"] = ka["block"] = blk
            TB.append(kb); TA.append(ka); pairs.append((kb, ka))
    PA, PB = population(TA, ref), population(TB, ref)
    paired = {}
    for key, lower_is_better in (("f_low", True), ("shape", True), ("contrast", False)):
        d = np.array([ka[key] - kb[key] for kb, ka in pairs]); d = d[np.isfinite(d)]
        better = int((d < -1e-9).sum()) if lower_is_better else int((d > 1e-9).sum())
        worse = int((d > 1e-9).sum()) if lower_is_better else int((d < -1e-9).sum())
        paired[key] = dict(n=int(len(d)), better=better, worse=worse, same=int(len(d) - better - worse),
                           median=float(np.median(d)) if len(d) else float("nan"))
    with open(a.out + ".txt", "w") as fh:
        P = lambda s="": fh.write(s + "\n")
        P(f"# doc pdvd/115 dQ/dx compare: det={a.det} base={a.base} arm={a.arm} events={nev} status={a.status} "
          f"min_npts={a.min_npts} low_thr={low_thr:.0f} e/cm ({a.low_frac} x plateau {plateau:.0f}); common accepted passes {len(pairs)}")
        P()
        P("| metric | base | arm |")
        P("|---|---|---|")
        for kk, name, fmt in (("npoints", "live points", "%d"), ("k_pop", "k_pop (scale vs the muon table)", "%.4f"),
                              ("chi2", "chi2 over populated rr bins (3 % floor)", "%.2f"), ("nbins", "populated bins", "%d"),
                              ("f_low_med", "median f_low (charge completeness; lower is better)", "%.4f"),
                              ("shape_med", "median per-track shape rms (lower is better)", "%.4f"),
                              ("k_med", "median per-track k", "%.4f"), ("contrast_med", "median Bragg contrast", "%.3f"),
                              ("qneg_share", "share of rows with q < 0", "%.4f"), ("holes_per_10m", "Bee holes per 10 m", "%.3f")):
            if kk in PB and kk in PA:
                P(f"| {name} | {fmt % PB[kk]} | {fmt % PA[kk]} |")
        P()
        P("| rr bin (cm) | base ratio | arm ratio | base n | arm n |")
        P("|---|---|---|---|---|")
        if "ratios" in PB:
            for i, (lo, hi) in enumerate(BINS):
                P(f"| {lo}-{hi} | {PB['ratios'][i]:.4f} +- {PB['errs'][i]:.4f} | {PA['ratios'][i]:.4f} +- {PA['errs'][i]:.4f} | {PB['ns'][i]} | {PA['ns'][i]} |")
        P()
        for key in ("f_low", "shape", "contrast"):
            p = paired[key]
            P(f"paired {key} (arm - base) on {p['n']} tracks: better {p['better']}, worse {p['worse']}, same {p['same']}; median {p['median']:+.4f}")
    json.dump(dict(det=a.det, base=a.base, arm=a.arm, events=nev, npairs=len(pairs), A=PB, B=PA, paired=paired),
              open(a.out + ".json", "w"), indent=1, default=lambda o: None)
    with open(a.out + "_tracks.tsv", "w") as fh:
        cols = ["event", "block", "npts", "ngood", "length", "has_left", "f_low", "nbins", "k", "shape", "contrast", "qneg", "holes"]
        fh.write("arm\t" + "\t".join(cols) + "\n")
        for lab, T in (("base", TB), ("arm", TA)):
            for tk in T:
                fh.write(lab + "\t" + "\t".join(("%.4f" % tk[c]) if isinstance(tk[c], float) else str(tk[c]) for c in cols) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "ratios" in PB:
            cen = np.array([(lo + hi) / 2 for lo, hi in BINS])
            fig, ax = plt.subplots(figsize=(7, 4))
            for lab, Pp, c in (("base " + a.base, PB, "k"), ("arm " + a.arm, PA, "C3")):
                ax.errorbar(cen, Pp["ratios"], yerr=Pp["errs"], fmt="o-", color=c, label=f"{lab}: k_pop {Pp['k_pop']:.3f}, chi2 {Pp['chi2']:.1f}")
            ax.axhline(1, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("residual range (cm)")
            ax.set_ylabel("median dQ/dx / (k_pop x muon table)"); ax.set_title(f"{a.det}: common accepted passes {len(pairs)}")
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    except Exception as ex:
        print("no figure:", ex, file=sys.stderr)
    print(open(a.out + ".txt").read())


if __name__ == "__main__":
    main()

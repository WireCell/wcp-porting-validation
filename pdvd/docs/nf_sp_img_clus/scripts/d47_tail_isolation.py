#!/usr/bin/env python3
"""doc pdvd/47 sec 9 -- where the >= 2-wire tail of sec 8.5(b) comes from.

Sec 8.5(b) found that the data profile is the simulated core PLUS a component beyond
+-2 wires that the simulation does not produce on any detector (data 0.002-0.017 of the
charge, simulation 0.000).  Sec 8.6 named the two non-instrumental sources that must be
bounded before that can be called instrumental: other activity inside the +-3 window
(delta rays, a second track) and the segment's own straightness.  This script answers
that in three steps, ordered by how few assumptions each needs.

  A. CONCENTRATION -- no cut, no threshold, nothing to tune.  A delta ray or a second
     track is rare and heavy: a few profiles carry a large tail, most carry exactly
     zero.  A per-channel effect (cross-talk, the field response's far lobes) is
     ubiquitous and proportional: nearly every profile carries a similar small share.
     Reported as the fraction of profiles with no tail charge at all and the share of
     the total tail charge held by the top 1/5/10 % of profiles, against the SAME
     statistic computed on the profile charge itself -- the null for anything strictly
     proportional to the signal.

  B. CHARGE DEPENDENCE -- tail SHARE against the profile's charge, in deciles.  The
     estimator clips negatives (y = max(Q, 0), doc 44), so positive noise excursions on
     the outer wires make a tail with no physical source at all.  That tail is roughly
     constant in ABSOLUTE size, i.e. share ~ 1/y: log-log slope -1.  Anything
     proportional to the signal gives slope 0.  This is the third hypothesis sec 8.6
     did not have.

  C. THE CUTS -- sec 8.6's `--max-foff 0.15` (isolation) and `--max-advance 0.10`
     (straightness), plus `rr > 30 cm`, each with the tail measured against the
     Gaussian at THAT selection's own fitted sigma and mean extent.  The raw share is
     not the verdict: every one of these cuts moves sigma, and a narrower sigma lowers
     the raw beyond-+-2 share with the same underlying physics.
     `foff` is the block's off-trajectory charge fraction ON W, so it is an independent
     isolation tag for a U or V profile and a circular one for a W profile.  For W the
     script uses `oth` instead: the same quantity measured on the OTHER two planes at
     the same time slice (d44_sigma_fit.py collect(), doc 47 sec 9), which cannot see
     the charge whose own tail is being measured.

Usage:
  d47_tail_isolation.py --det pdvd --out figs/47_tail_pdvd  pdvd/work/*_d42fit/tracking-stm.root
  d47_tail_isolation.py --det pdhd --sigma-fit pdhd/docs/scripts/d44_sigma_fit.py \
      --max-advance 0.436 --out figs/47_tail_pdhd  pdhd/work/*_stmwc/tracking-stm.root
"""
import argparse, importlib.util, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_sigma_fit(path):
    """the doc-44 estimator, imported (never copied).  The PDVD copy knows pdvd/sbnd;
    PDHD needs its fork, which carries the extra DET entry."""
    spec = importlib.util.spec_from_file_location("d44_sigma_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def top_share(w, fracs=(0.01, 0.05, 0.10)):
    """share of sum(w) held by the largest frac of the entries."""
    if len(w) == 0 or w.sum() <= 0:
        return [np.nan] * len(fracs)
    v = np.sort(w)[::-1]
    c = np.cumsum(v) / v.sum()
    return [float(c[min(len(v) - 1, max(0, int(round(f * len(v))) - 1))]) for f in fracs]


def anatomy(R, sel, Pi, det, label="base"):
    """A and B on two quantities: the whole >= 2-wire tail (r2 + r3, the charge that
    carries the width excess of sec 8.5) and the far tail alone (r3, beyond +-2)."""
    s = sel & (R["plane"] == Pi)
    y = R["y"][s]; r2 = R["r2"][s]; r3 = R["r3"][s]
    row = dict(sel=label, plane="UVW"[Pi], n=int(s.sum()), q_tot=float(y.sum()))
    for lab, w in (("t2p", r2 + r3), ("t3", r3)):
        row["tail_" + lab] = float(w.sum() / y.sum())
        row["frac_zero_" + lab] = float((w <= 0).mean())
        row["frac_over10pct_" + lab] = float((w > 0.10 * y).mean())
        for f, v in zip((1, 5, 10), top_share(w)):
            row["top%d_%s" % (f, lab)] = v
    for f, v in zip((1, 5, 10), top_share(y)):
        row["top%d_q" % f] = v
    # the load-bearing number: concentration measured against the null for anything
    # strictly proportional to the signal (the same statistic on the charge itself).
    # Ratios normalise away part of the data/sim difference in profile density.
    for lab in ("t2p", "t3"):
        for f in (1, 5):
            row["r%d_%s" % (f, lab)] = float(row["top%d_%s" % (f, lab)] / max(row["top%d_q" % f], 1e-9))
    # B: tail share in deciles of the profile charge
    o = np.argsort(y)
    for lab, w in (("t2p", r2 + r3), ("t3", r3)):
        xs, ys = [], []
        for k, ix in enumerate(np.array_split(o, 10)):
            qs = y[ix].sum()
            sh = float(w[ix].sum() / qs) if qs > 0 else np.nan
            row["dec%d_q" % k] = float(np.median(y[ix]))
            row["dec%d_%s" % (k, lab)] = sh
            if np.isfinite(sh) and sh > 0 and np.median(y[ix]) > 0:
                xs.append(np.log(np.median(y[ix]))); ys.append(np.log(sh))
        row["slope_" + lab] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else np.nan
        # the deciles are U-shaped (clipped noise at the bottom, whatever the tail is at
        # the top), so a single log-log slope is a poor summary: keep the three anchors
        # and the share of the WHOLE tail that the top charge decile carries
        row["dec_lo_" + lab] = row["dec0_" + lab]
        row["dec_mid_" + lab] = float(np.nanmean([row["dec%d_%s" % (k, lab)] for k in (4, 5)]))
        row["dec_hi_" + lab] = row["dec9_" + lab]
        dec = np.array_split(o, 10)
        row["tail_from_hidec_" + lab] = float(w[dec[9]].sum() / max(w.sum(), 1e-12))
        row["tail_from_lodec_" + lab] = float(w[dec[0]].sum() / max(w.sum(), 1e-12))
    return row


def shape_of(mod, R, sel, det, model, Pi, nbins, nboot, rng, label):
    """ring shares of the stacked profile against the Gaussian at THIS selection's own
    fitted sigma and mean extent -- the d44 _shape.tsv block, per selection."""
    s = sel & (R["plane"] == Pi)
    if s.sum() < 20:
        return None
    bins, fits = mod.bin_and_fit(R, sel, det, model, nbins, nboot, rng, label)
    pl = "UVW"[Pi]
    fr = [f for f in fits if f["label"] == label and f["est"] == "share" and f["plane"] == pl]
    br = [b for b in bins if b["label"] == label and b["est"] == "share" and b["plane"] == pl]
    if not fr or not br:
        return None
    y = R["y"][s]
    meas = np.array([R[k][s].sum() for k in ("r0", "r1", "r2", "r3")]) / y.sum()
    tmean = np.average([b["t_us"] for b in br], weights=[b["q"] for b in br]) * 1e3
    ext = np.average([b["extent"] for b in br], weights=[b["q"] for b in br])
    pitch = det["pitch"][Pi]
    sig = np.sqrt(max(2 * fr[0]["DT_json"] * tmean + fr[0]["c2_mm2"], 1e-6)) / pitch
    g = mod.ring_shares(sig, ext)
    # what the >= 2-wire excess alone is worth as a WIDTH: the second moment about the
    # centroid that this charge carries, over and above the fitted Gaussian's own.  The
    # beyond-+-2 ring is charged at 3 wires, its smallest possible offset, so the number
    # is a lower bound.  This is the same charge that makes the profile "more peaked
    # than a Gaussian of equal rms" (doc 44 sec 3.3, doc 47 sec 8.5(a)): the
    # share-matched sigma is fitted to the CENTRE share and knows nothing about it.
    dvar = max(meas[2] - g[2], 0) * (2 * pitch) ** 2 + max(meas[3] - g[3], 0) * (3 * pitch) ** 2
    return dict(sel=label, plane=pl, n=int(s.sum()), q_frac=np.nan,
                c_eff_mm=fr[0]["c_eff_mm"], DT_eff=fr[0]["DT_eff_cm2s"],
                sig_mm=sig * pitch, extent=ext, t_us=tmean / 1e3,
                meas_pm2=meas[2], meas_beyond=meas[3],
                gaus_pm2=g[2], gaus_beyond=g[3],
                exc_pm2=meas[2] - g[2], exc_beyond=meas[3] - g[3],
                exc_tail=(meas[2] + meas[3]) - (g[2] + g[3]),
                sig_excess_mm=float(np.sqrt(dvar)))


def rows_mode(mod, a):
    """A and B on the simulation's own per-profile rows.  No cut applies (the simulated
    tracks are isolated and straight by construction) and there is nothing to re-fit;
    what this gives is the NULL for part B -- the charge dependence of a tail that is
    known to be the Gaussian core plus clipped noise and nothing else."""
    hdr = open(a.rows).readline().rstrip("\n").split("\t")
    dat = np.genfromtxt(a.rows, delimiter="\t", skip_header=1)
    R = {k: dat[:, i] for i, k in enumerate(hdr)}
    R["plane"] = R["plane"].astype(int)
    sel = np.ones(len(R["y"]), bool)
    arows = [anatomy(R, sel, Pi, None, a.tag) for Pi in range(3)]
    arows = [r for r in arows if r and r["n"] >= 20]
    path = a.out + "_anatomy.tsv"
    if a.append and os.path.exists(path):
        keys = open(path).readline().rstrip("\n").split("\t")
        with open(path, "a") as fo:
            for r in arows:
                fo.write("\t".join("%.6g" % r[k] if isinstance(r[k], float) else str(r[k])
                                    for k in keys) + "\n")
    else:
        mod.write_tsv(path, arows)
    print("simulation rows %s  [%s]" % (a.rows, a.tag))
    print("     %-2s %7s | %8s %7s %7s %7s | %8s %7s %7s | %7s" % (
        "pl", "n", "t2p", "zero", "top1", "top5", "t3", "zero", "top1", "q:top1"))
    for r in arows:
        print("     %-2s %7d | %8.4f %7.3f %7.3f %7.3f | %8.4f %7.3f %7.3f | %7.3f" % (
            r["plane"], r["n"], r["tail_t2p"], r["frac_zero_t2p"], r["top1_t2p"], r["top5_t2p"],
            r["tail_t3"], r["frac_zero_t3"], r["top1_t3"], r["top1_q"]))
    for r in arows:
        print("     %-2s t2p %s  slope %+.2f  top decile carries %.2f" % (
            r["plane"], " ".join("%.4f" % r["dec%d_t2p" % k] for k in range(10)),
            r["slope_t2p"], r["tail_from_hidec_t2p"]))
    print("  -> %s_anatomy.tsv" % a.out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-fit", default=os.path.join(HERE, "d44_sigma_fit.py"))
    ap.add_argument("--model-json", default=None)
    ap.add_argument("--status", type=int, default=0)
    ap.add_argument("--ticks-per-slice", type=float, default=4.0)
    ap.add_argument("--max-advance", type=float, default=0.25)
    ap.add_argument("--halfwidth", type=float, default=3.0)
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--nboot", type=int, default=50)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--foff-cut", type=float, default=0.15, help="sec 8.6 isolation cut (W-derived)")
    ap.add_argument("--oth-cut", type=float, default=0.05,
                    help="isolation from the OTHER two planes at the same slice")
    ap.add_argument("--max-foff", type=float, default=1.1)   # collect() reads none of these,
    ap.add_argument("--split", default="")                   # but keep the namespace compatible
    ap.add_argument("--phase-split", action="store_true")
    ap.add_argument("--tag", default="sim", help="--rows: the arm label written to the TSV")
    ap.add_argument("--append", action="store_true", help="--rows: append to an existing TSV")
    ap.add_argument("--rows", default=None,
                    help="A and B only, on a d47_sim_transverse_profile.py _rows.tsv "
                         "(the simulation's own profiles: no fit, no cuts to apply)")
    a = ap.parse_args()

    mod = load_sigma_fit(a.sigma_fit)
    if a.rows:
        return rows_mode(mod, a)
    det = mod.DET[a.det]
    model = mod.load_model(a.model_json or det["json"])
    rng = np.random.default_rng(a.seed)
    R, blocks = mod.collect(a, det, model)
    if len(R["bid"]) == 0:
        print("no profiles", file=sys.stderr); return 1
    bf = np.array([blocks[b]["foff"] for b in R["bid"]])
    # the isolation tag: charge in the OTHER two planes at this slice more than FOUR
    # wires from the trajectory.  At two wires the tag is dominated by the track's own
    # width (PDHD: a clean 0.7-pitch profile puts ~8 % there), so a "> 2 wires" tag --
    # and the block-level foff, which is the same quantity on W -- selects narrow
    # slices rather than isolated ones.  See doc 46 on f_off mixing width and coverage.
    oth = np.nan_to_num(R["oth4"], nan=1.0)
    oth2 = np.nan_to_num(R["oth"], nan=1.0)
    base = R["adv"] < a.max_advance
    print("%s: %d blocks, %d profiles, %d prolonged (adv<%.3f)" % (
        a.det.upper(), len(blocks), len(base), base.sum(), a.max_advance))
    print("  foff (block, on W) quantiles 10/25/50/75/90%%: %s" % np.round(
        np.nanpercentile(bf, [10, 25, 50, 75, 90]), 3))
    print("  oth2 (other planes, >2 wires off the trajectory) 10/25/50/75/90%%: %s" %
          np.round(np.nanpercentile(R["oth"], [10, 25, 50, 75, 90]), 3))
    print("  oth4 (other planes, >4 wires -- the isolation tag) 10/25/50/75/90%%: %s   nan %.3f" % (
        np.round(np.nanpercentile(R["oth4"], [10, 25, 50, 75, 90]), 3),
        float(np.isnan(R["oth4"]).mean())))

    # ---------------------------------------------------------------- A and B
    tight = base & (oth <= a.oth_cut) & (R["adv"] < 0.10) & (R["rr"] > 30)
    arows = [anatomy(R, base, Pi, det, "base") for Pi in range(3)]
    arows += [anatomy(R, tight, Pi, det, "iso+straight+rr") for Pi in range(3)]
    arows = [r for r in arows if r and r["n"] >= 20]
    mod.write_tsv(a.out + "_anatomy.tsv", arows)
    print("  A. concentration of the tail (prolonged, all t).  t2p = charge >= 2 wires from")
    print("     the centroid, t3 = beyond +-2; q: = the same statistic on the profile charge,")
    print("     i.e. the null for anything strictly proportional to the signal.")
    print("     %-16s %-2s %7s | %8s %7s %7s %7s %7s | %8s %7s %7s | %7s %7s" % (
        "selection", "pl", "n", "t2p", "zero", "top1", "top5", ">10%", "t3", "zero", "top1",
        "q:top1", "q:top5"))
    for r in arows:
        print("     %-16s %-2s %7d | %8.4f %7.3f %7.3f %7.3f %7.4f | %8.4f %7.3f %7.3f | %7.3f %7.3f" % (
            r["sel"], r["plane"], r["n"], r["tail_t2p"], r["frac_zero_t2p"], r["top1_t2p"],
            r["top5_t2p"], r["frac_over10pct_t2p"], r["tail_t3"], r["frac_zero_t3"],
            r["top1_t3"], r["top1_q"], r["top5_q"]))
    print("  B. tail share vs profile charge, deciles (clipped noise -> slope -1, a")
    print("     signal-proportional effect -> 0)")
    for r in arows:
        if r["sel"] != "base":
            continue
        print("     %-2s t2p %s  slope %+.2f  top decile carries %.2f of the tail" % (
            r["plane"], " ".join("%.4f" % r["dec%d_t2p" % k] for k in range(10)),
            r["slope_t2p"], r["tail_from_hidec_t2p"]))
        print("        t3  %s  slope %+.2f  top decile carries %.2f" % (
            " ".join("%.4f" % r["dec%d_t3" % k] for k in range(10)),
            r["slope_t3"], r["tail_from_hidec_t3"]))

    # ---------------------------------------------------------------- C
    sels = [("base", base),
            ("foff<%.2f" % a.foff_cut, base & (np.nan_to_num(bf, nan=1.0) <= a.foff_cut)),
            ("oth2<%.2f" % a.oth_cut, base & (oth2 <= a.oth_cut)),
            ("oth4<%.2f" % a.oth_cut, base & (oth <= a.oth_cut)),
            ("adv<0.10", base & (R["adv"] < 0.10)),
            ("rr>30cm", base & (R["rr"] > 30)),
            ("iso+straight+rr", tight)]
    crows = []
    qtot = {Pi: R["y"][base & (R["plane"] == Pi)].sum() for Pi in range(3)}
    for name, sel in sels:
        for Pi in range(3):
            row = shape_of(mod, R, sel, det, model, Pi, a.nbins, a.nboot, rng, name)
            if row is None:
                continue
            row["q_frac"] = float(R["y"][sel & (R["plane"] == Pi)].sum() / max(qtot[Pi], 1e-9))
            crows.append(row)
    mod.write_tsv(a.out + "_cuts.tsv", crows)
    print("  C. the tail under the sec 8.6 cuts, each against its OWN fitted Gaussian")
    print("     (c from a sub-selection is not reliable -- the cuts are drift-correlated and")
    print("      the lever arm collapses; read sig[mm], the charge-weighted width, instead)")
    print("     %-16s %-2s %7s %6s %7s %7s | %8s %8s %8s | %8s %8s %8s | %7s" % (
        "selection", "pl", "n", "qfrac", "c[mm]", "sig[mm]", "meas pm2", "gaus pm2", "excess",
        "meas >2", "gaus >2", "excess", "sig_exc"))
    for r in crows:
        print("     %-16s %-2s %7d %6.3f %7.2f %7.2f | %8.4f %8.4f %+8.4f | %8.4f %8.4f %+8.4f | %7.2f" % (
            r["sel"], r["plane"], r["n"], r["q_frac"], r["c_eff_mm"], r["sig_mm"],
            r["meas_pm2"], r["gaus_pm2"], r["exc_pm2"],
            r["meas_beyond"], r["gaus_beyond"], r["exc_beyond"], r["sig_excess_mm"]))
    print("  -> %s_{anatomy,cuts}.tsv" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

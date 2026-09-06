#!/usr/bin/env python3
"""doc pdvd/47 sec 8 -- build the phase-window tables from a `_bins.tsv` that carries the
`phase:*` labels (d44_sigma_fit.py --phase-split on data, d47_sim_transverse_profile.py
--phase-split on simulation).

The line intercept c is NOT the right statistic to compare across phase windows: on the
angled simulated tracks the sub-pitch phase advances with drift, so a phase window samples
a periodic subset of drift slices and the fitted D_T,eff and c trade off against each other
(the same D/c degeneracy doc 44 sec 2.2 fights).  What is comparable is the WIDTH ITSELF in
each drift bin.  This prints, per plane and window, the charge-weighted mean of sigma_eff
over the drift bins (and the fitted c beside it, for the record).

Usage:
  d47_phase_table.py --bins figs/47_phase_pdvd_bins.tsv --fit figs/47_phase_pdvd_fit.tsv \\
      --est share --label-prefix phase: [--artefact figs/47_phase_artefact_pdvd.tsv] [--out ...]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import write_tsv  # noqa: E402
from d47_phase_artefact import read_tsv  # noqa: E402

WINS = ["full", "q1", "q2", "q3", "q4", "centre", "edge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--est", default="share", choices=("rms", "share", "truth"))
    ap.add_argument("--artefact", default=None, help="d47_phase_artefact.py TSV (same det/est)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--append", action="store_true", help="append to --out (one combined table)")
    a = ap.parse_args()
    bins = read_tsv(a.bins)
    fits = read_tsv(a.fit) if a.fit else []
    art = read_tsv(a.artefact) if a.artefact else []
    rows = []
    for pl in "UVW":
        for w in WINS:
            br = [b for b in bins if b["label"] == "phase:" + w and b["est"] == a.est and b["plane"] == pl]
            if not br:
                continue
            q = np.array([b["q"] for b in br]); s = np.sqrt(np.maximum([b["sig2_eff_mm2"] for b in br], 0))
            e = np.array([b["sig2_err"] for b in br]) / (2 * np.maximum(s, 1e-6))
            fr = [f for f in fits if f["label"] == "phase:" + w and f["est"] == a.est and f["plane"] == pl + "(joint)"]
            ar = [x for x in art if x["window"] == w and x["plane"] == pl and x["inversion"] == "bin"]
            row = dict(tag=a.tag, est=a.est, plane=pl, window=w, n_bins=len(br),
                       sig_mean_mm=float(np.average(s, weights=q)),
                       sig_err_mm=float(np.sqrt(np.average(e ** 2, weights=q) / len(br))),
                       q_frac=float(q.sum()),
                       c_fit_mm=float(fr[0]["c_eff_mm"]) if fr else np.nan,
                       DT_fit_cm2s=float(fr[0]["DT_eff_cm2s"]) if fr else np.nan,
                       chi2_ndf=float(fr[0]["chi2"] / max(fr[0]["ndf"], 1)) if fr else np.nan)
            if ar:
                # the artefact prediction sampled on the reference run's drift bins
                aq = np.ones(len(ar))
                row["sig_artefact_old_mm"] = float(np.average([x["sig_old_mm"] for x in ar], weights=aq))
                row["sig_artefact_true_mm"] = float(np.average([x["sig_true_mm"] for x in ar], weights=aq))
            rows.append(row)
    qtot = {pl: sum(r["q_frac"] for r in rows if r["plane"] == pl and r["window"] == "full") for pl in "UVW"}
    for r in rows:
        r["q_frac"] = r["q_frac"] / max(qtot[r["plane"]], 1e-9)
    if a.out:
        if a.append and os.path.exists(a.out):
            keys = open(a.out).readline().rstrip("\n").split("\t")
            with open(a.out, "a") as fo:
                for r_ in rows:
                    fo.write("\t".join("%.6g" % r_[k] if isinstance(r_.get(k), float) else str(r_.get(k, "")) for k in keys) + "\n")
        else:
            write_tsv(a.out, rows)
    print("%s  est=%s   <sigma_eff> [mm] charge-weighted over the drift bins (c of the line beside it)" % (a.tag, a.est))
    print("  plane %s" % "".join("%14s" % w for w in WINS))
    for pl in "UVW":
        rr = {r["window"]: r for r in rows if r["plane"] == pl}
        if not rr:
            continue
        print("  %s     %s" % (pl, "".join("%8.2f+-%-5.2f" % (rr[w]["sig_mean_mm"], rr[w]["sig_err_mm"]) if w in rr else "%14s" % "-" for w in WINS)))
        print("    c     %s" % "".join("%14.2f" % rr[w]["c_fit_mm"] if w in rr else "%14s" % "-" for w in WINS))
        if any("sig_artefact_old_mm" in rr.get(w, {}) for w in WINS):
            print("    artefact (phase-independent sigma, OLD inversion): %s" %
                  "".join("%9.2f" % rr[w]["sig_artefact_old_mm"] if w in rr and "sig_artefact_old_mm" in rr[w] else "%9s" % "-" for w in WINS))
        if "edge" in rr and "centre" in rr:
            print("    edge/centre = %.3f" % (rr["edge"]["sig_mean_mm"] / max(rr["centre"]["sig_mean_mm"], 1e-9)))
    if a.out:
        print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

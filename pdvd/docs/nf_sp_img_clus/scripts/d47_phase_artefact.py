#!/usr/bin/env python3
"""doc pdvd/47 sec 8.1 -- what the PHASE-AVERAGED inversion reports on a phase-SELECTED
subset when the true sigma is phase INDEPENDENT.

The estimator of doc 44 measures the charge profile's rms about its own centroid (and the
share in the centre wire) and inverts it against a binned-Gaussian model averaged over the
source's sub-pitch position (`apparent_rms`/`ring_shares` marginalise over `_U`).  Both
statistics depend strongly on that position: a source at a wire-bin centre puts everything
in one bin (rms 0) and one at a bin boundary splits it in two (rms 0.5 pitch).  So on a
subset selected by phase, the phase-averaged model is the wrong model, and it manufactures
a centre/boundary contrast out of nothing.  This script quantifies exactly that: it takes a
published `all` fit (sigma^2 = 2 D t + c^2) as the TRUTH, assumes sigma is phase
independent, computes the statistic each phase window would then measure, inverts it the
old (phase-averaged) way, and refits the line -> the c the old machinery would have
reported.  The same numbers with the corrected inversion return the input c (printed as the
closure check).

Usage:
  d47_phase_artefact.py --det pdvd --fit figs/44_sigma_pdvd_fit.tsv --bins figs/44_sigma_pdvd_bins.tsv \\
      --est share --out figs/47_phase_artefact_pdvd.tsv
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import apparent_rms, ring_shares, _bisect, wlsq, write_tsv  # noqa: E402

PITCH = {"pdvd": (7.65, 7.65, 5.10), "sbnd": (3.00, 3.00, 3.00), "pdhd": (4.6693, 4.6693, 4.7920)}
WINS = [("full", -0.5, 0.5), ("q1", -0.5, -0.25), ("q2", -0.25, 0.0), ("q3", 0.0, 0.25),
        ("q4", 0.25, 0.5), ("centre", -0.25, 0.25), ("edge", 0.25, 0.5)]


def read_tsv(path):
    with open(path) as f:
        keys = f.readline().rstrip("\n").split("\t")
        out = []
        for line in f:
            v = line.rstrip("\n").split("\t")
            d = {}
            for k, x in zip(keys, v):
                try:
                    d[k] = float(x)
                except ValueError:
                    d[k] = x
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=PITCH)
    ap.add_argument("--fit", required=True, help="_fit.tsv of the reference run")
    ap.add_argument("--bins", required=True, help="_bins.tsv of the same run (t, extent, errors)")
    ap.add_argument("--est", default="share", choices=("rms", "share"))
    ap.add_argument("--label", default="all")
    ap.add_argument("--joint", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fits = read_tsv(a.fit); bins = read_tsv(a.bins)
    rows = []
    for Pi, pl in enumerate("UVW"):
        want = pl + "(joint)" if a.joint else pl
        fr = [f for f in fits if f["label"] == a.label and f["est"] == a.est and f["plane"] == want]
        br = [b for b in bins if b["label"] == a.label and b["est"] == a.est and b["plane"] == pl]
        if not fr or len(br) < 3:
            continue
        D, c2 = fr[0]["DT_json"], fr[0]["c2_mm2"]          # mm^2/ns, mm^2
        pitch = PITCH[a.det][Pi]
        t = np.array([b["t_us"] * 1e3 for b in br])
        ext = np.array([b["extent"] for b in br])
        err = np.array([b["sig2_err"] for b in br])
        sig_true = np.sqrt(np.maximum(2 * D * t + c2, 1e-9)) / pitch     # pitch units, phase INDEPENDENT
        for name, lo, hi in WINS:
            rep, clo = [], []
            for k in range(len(t)):
                s, e = sig_true[k], ext[k]
                if a.est == "rms":
                    meas = apparent_rms(s, e, None, (lo, hi))
                    old, _ = _bisect(lambda g: apparent_rms(g, e), meas, 0.02, 3.0)
                    new, _ = _bisect(lambda g: apparent_rms(g, e, None, (lo, hi)), meas, 0.02, 3.0)
                else:
                    meas = ring_shares(s, e, None, (lo, hi))[0]
                    old, _ = _bisect(lambda g: -ring_shares(g, e)[0], -meas, 0.02, 3.0)
                    new, _ = _bisect(lambda g: -ring_shares(g, e, None, (lo, hi))[0], -meas, 0.02, 3.0)
                rep.append(old * pitch); clo.append(new * pitch)
            for k in range(len(t)):
                rows.append(dict(det=a.det, est=a.est, plane=pl, window=name, inversion="bin",
                                 tbin=k, t_us=t[k] / 1e3, extent=ext[k],
                                 sig_true_mm=sig_true[k] * pitch, sig_old_mm=rep[k], sig_corr_mm=clo[k]))
            for tag, val in (("old", np.array(rep)), ("corrected", np.array(clo))):
                A = np.column_stack([2 * t, np.ones(len(t))])
                beta, cov, chi2 = wlsq(A, val ** 2, 1.0 / err)
                c = np.sqrt(beta[1]) if beta[1] >= 0 else -np.sqrt(-beta[1])
                rows.append(dict(det=a.det, est=a.est, plane=pl, window=name, inversion=tag,
                                 tbin=-1, t_us=np.nan, extent=np.nan, sig_true_mm=np.nan,
                                 sig_old_mm=np.nan, sig_corr_mm=np.nan,
                                 c_pred_mm=c, DT_pred_cm2s=beta[0] * 1e7,
                                 c_input_mm=(np.sqrt(c2) if c2 >= 0 else -np.sqrt(-c2)),
                                 DT_input_cm2s=D * 1e7,
                                 sig_true_lo_pitch=sig_true.min(), sig_true_hi_pitch=sig_true.max()))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    rows = [{k: r.get(k, np.nan) for k in keys} for r in rows]
    write_tsv(a.out, rows)
    print("%s %s: c [mm] the phase-averaged inversion would report for a phase-INDEPENDENT sigma" % (a.det, a.est))
    print("  plane  input   %s" % "  ".join("%8s" % w[0] for w in WINS))
    for pl in "UVW":
        r = {x["window"]: x for x in rows if x["plane"] == pl and x["inversion"] == "old"}
        n = {x["window"]: x for x in rows if x["plane"] == pl and x["inversion"] == "corrected"}
        if not r:
            continue
        print("  %s     %5.2f   %s" % (pl, r["full"]["c_input_mm"], "  ".join("%8.2f" % r[w[0]]["c_pred_mm"] for w in WINS)))
        print("    closure (corrected inversion, must return the input): %s" %
              "  ".join("%5.2f" % n[w[0]]["c_pred_mm"] for w in WINS))
    print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

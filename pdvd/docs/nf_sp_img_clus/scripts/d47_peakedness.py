#!/usr/bin/env python3
"""doc pdvd/47 sec 8.5 -- a phase-FREE shape test: how peaked the stacked transverse
profile is, at matched width, in data and in simulation.

Doc 44 sec 2.3 found the measured profile more peaked than a Gaussian of the same rms, and
kept two sigmas per drift bin: the one matched to the profile's rms (`est=rms`) and the one
matched to the share of charge in the centre wire (`est=share`).  Their RATIO
    rho = sigma_rms / sigma_share
is a dimensionless shape statistic: 1 for a Gaussian, > 1 for a narrow core with tails.  It
needs no knowledge of the sub-pitch phase, so unlike the phase split (sec 8.3) it is not
diluted by the fitted trajectory's phase resolution.  Comparing rho at the SAME sigma_share
asks whether the extra width the ProtoDUNE data carry over the simulation (sec 4.4) has the
same shape as the width the simulation produces, or a different one.

Usage:
  d47_peakedness.py --out figs/47_peakedness.tsv \\
     data:pdvd:figs/44_sigma_pdvd_bins.tsv sim:pdvd:/home/xqian/tmp/xtrack/pdvd/ana/S1_gauss_bins.tsv ...
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import write_tsv  # noqa: E402
from d47_phase_artefact import read_tsv  # noqa: E402


def curve(path, label="all"):
    """-> {plane: (sigma_share_mm, rho, q)} over the drift bins of `label`"""
    B = read_tsv(path)
    out = {}
    for pl in "UVW":
        r = [b for b in B if b["label"] == label and b["est"] == "rms" and b["plane"] == pl]
        s = {b["tbin"]: b for b in B if b["label"] == label and b["est"] == "share" and b["plane"] == pl}
        x, y, q = [], [], []
        for b in r:
            t = s.get(b["tbin"])
            if not t or t["sig2_eff_mm2"] <= 0 or b["sig2_eff_mm2"] <= 0:
                continue
            x.append(np.sqrt(t["sig2_eff_mm2"])); y.append(np.sqrt(b["sig2_eff_mm2"] / t["sig2_eff_mm2"]))
            q.append(b["q"])
        if x:
            out[pl] = (np.array(x), np.array(y), np.array(q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="kind:det:path (kind = data|sim, path = a _bins.tsv)")
    ap.add_argument("--label", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = {}
    for spec in a.sets:
        kind, det, path = spec.split(":", 2)
        C[(kind, det)] = curve(path, a.label)
    rows = []
    for (kind, det), cur in C.items():
        for pl, (x, y, q) in cur.items():
            for k in range(len(x)):
                rows.append(dict(kind=kind, det=det, plane=pl, tbin=k, sigma_share_mm=x[k], rho=y[k], q=q[k]))
    write_tsv(a.out, rows)
    print("rho = sigma(rms-matched) / sigma(share-matched): 1 = Gaussian, > 1 = peaked core + tails")
    print("  det  plane |   data: <sigma> <rho>  |   sim: <sigma> <rho>  | sim rho AT the data's sigma | data-sim")
    for det in sorted({d for (_, d) in C}):
        for pl in "UVW":
            d = C.get(("data", det), {}).get(pl); s = C.get(("sim", det), {}).get(pl)
            if not d or not s:
                continue
            sd = np.average(d[0], weights=d[2]); rd = np.average(d[1], weights=d[2])
            ss = np.average(s[0], weights=s[2]); rs = np.average(s[1], weights=s[2])
            # sim rho evaluated at the data's sigma (linear in sigma over the sim's drift bins)
            A = np.column_stack([s[0], np.ones(len(s[0]))])
            beta, *_ = np.linalg.lstsq(A * np.sqrt(s[2])[:, None], s[1] * np.sqrt(s[2]), rcond=None)
            rs_at = beta[0] * sd + beta[1]
            print("  %-4s   %s   |  %6.2f  %6.3f     |  %6.2f  %6.3f    |        %6.3f            | %+.3f" % (
                det, pl, sd, rd, ss, rs, rs_at, rd - rs_at))
    print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

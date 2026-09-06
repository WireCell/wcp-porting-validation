#!/usr/bin/env python3
"""doc pdvd/47 sec 8.4 -- the transverse POSITION bias of the signal-processing output as a
function of the charge's sub-pitch phase, measured on the simulated tracks (truth known).

`d47_sim_transverse_profile.py` records, per (track, plane, slice), the true sub-pitch phase
`phase` of the trajectory inside its wire bin and `off` = (charge centroid of the +-3 window)
- phase, in wires.  A deconvolution that is exact for every impact position would give
<off> = 0 at every phase.  What the production chain actually gives is a centroid pulled
TOWARD the nearest wire centre, linear in the phase: <off> = s * phase with s < 0.  That is
the impact-position residual of deconvolving against the pitch-averaged field response
(Response::wire_region_average) -- it appears as a position BIAS, not as extra width
(sec 8.2), and it is what makes the reconstructed (and hence fitted) trajectory phase pile
up at wire centres in data (sec 8.3).

Usage:
  d47_phase_bias.py --out figs/47_phase_bias.tsv \\
     pdhd:S1:/home/xqian/tmp/xtrack/pdhd/ana/S1_gauss_ph2_rows.tsv ...
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import write_tsv  # noqa: E402

NB = 8


def load(path):
    lines = open(path).read().splitlines()
    keys = lines[0].split("\t")
    cols = {k: [] for k in keys}
    for l in lines[1:]:
        for k, v in zip(keys, l.split("\t")):
            cols[k].append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="det:arm:path to a _rows.tsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    print("SP transverse position bias: <centroid - truth> [wire] vs the true sub-pitch phase")
    print("  det  arm  pl |" + "".join("%7.2f" % x for x in (np.arange(NB) + 0.5) / NB - 0.5) +
          " |  slope   rms(off)  peak/flat of the CENTROID phase")
    for spec in a.sets:
        det, arm, path = spec.split(":", 2)
        if not os.path.exists(path):
            print("  missing", path, file=sys.stderr); continue
        R = load(path)
        if "off" not in R:
            print("  %s: no `off` column (written by a pre-sec-8 run) -- rerun the profile script"
                  % path, file=sys.stderr); continue
        for Pi, pl in enumerate("UVW"):
            m = R["plane"] == Pi
            if m.sum() < 50:
                continue
            ph = R["phase"][m]; off = R["off"][m]; w = np.maximum(R["y"][m], 0)
            e = np.linspace(-0.5, 0.5, NB + 1); b = np.clip(np.digitize(ph, e) - 1, 0, NB - 1)
            mu = [float(np.average(off[b == i], weights=w[b == i])) if (b == i).sum() > 5 else np.nan for i in range(NB)]
            slope, icpt = np.polyfit(ph, off, 1, w=w)
            phm = ph + off; phm = phm - np.round(phm)              # the phase a charge-driven estimator sees
            h, _ = np.histogram(phm, bins=10, range=(-0.5, 0.5), weights=w)
            h = h / h.sum()
            rows.append(dict(det=det, arm=arm, plane=pl, n=int(m.sum()), slope=float(slope), intercept=float(icpt),
                             rms_off=float(np.sqrt(np.average(off ** 2, weights=w))),
                             centroid_phase_peak=float(h[4:6].sum() / 0.2),
                             **{"bias_b%d" % i: mu[i] for i in range(NB)},
                             **{"cph_b%d" % i: float(h[i]) for i in range(10)}))
            print("  %-4s %-4s %s |%s | %+.3f   %.3f    %.2fx" % (
                det, arm, pl, "".join("%7.3f" % x for x in mu), slope,
                rows[-1]["rms_off"], rows[-1]["centroid_phase_peak"]))
    write_tsv(a.out, rows)
    print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

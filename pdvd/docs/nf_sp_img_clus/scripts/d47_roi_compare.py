#!/usr/bin/env python3
"""doc 47 sec 11 -- splitting the ROI stage's rejection into admission and amplitude.

Section 9.5 left one number unexplained and it is the largest in the document: the
2-D deconvolution feeds ROI formation the SAME cross-channel amplitude in data and in
simulation (measured on |q|, they agree to a few per cent), and the stage that follows
removes 99.3 % of it in simulation against 93-95 % in the data.  A factor of ~100 in
what one stage rejects.

`gauss` is EXACTLY zero outside an ROI and never negative (verified on both sides:
97.9 % of the frame is exact zero, min == 0), so for a cell at offset D from the fitted
trajectory the slice sum is a clean in-ROI flag.  That splits the survival of the
cross-channel charge into two factors that have different causes and different next
steps:

    share(D)  ~  P(D) x E(D)

  P(D) = fraction of profiles where the cell D wires from the track is INSIDE AN ROI
         -- "does the stage admit that channel at all"
  E(D) = mean charge there GIVEN it is admitted, normalised to the centre cell
         -- "how much of what is admitted the filters keep"

P differing between data and simulation means the stage ADMITS more; E differing means
the filters KEEP more of what is admitted.

The offsets are measured from the TRAJECTORY, never from the profile's own centroid:
the question is whether an ROI exists that far from the track, and that must not be
defined by the charge being measured.

The noise arms (S1n05, S1n2) give P(D) as a function of the noise-scaled threshold, so
the data can be asked whether it sits ON that curve -- in which case the factor of 100
is a signal-to-noise configuration difference and not a detector effect.

Usage:
  d47_roi_compare.py --data <roi tsv> --sim label=<roi tsv> [--sim ...] [--out <prefix>]
"""
import argparse
import csv
import os
import sys

import numpy as np

PLANES = ("U", "V", "W")


def read(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows:
        raise SystemExit("empty: " + path)
    offs = sorted((k for k in rows[0] if k.startswith("q") and k[1] in "+-"),
                  key=lambda k: int(k[1:]))
    d = dict(plane=np.array([int(float(r["plane"])) for r in rows]),
             t_ns=np.array([float(r["t_ns"]) for r in rows]),
             thresh=np.array([float(r["thresh"]) if r["thresh"] not in ("", "nan") else np.nan
                              for r in rows]),
             q=np.array([[float(r[k]) for k in offs] for r in rows]),
             offs=np.array([int(k[1:]) for k in offs]))
    return d


def pe(d, sel):
    """-> (P, E, n) per offset for the selected profiles; E normalised to the centre."""
    q = d["q"][sel]
    if len(q) == 0:
        return None
    inroi = q != 0.0
    P = inroi.mean(axis=0)
    E = np.array([q[inroi[:, j], j].mean() if inroi[:, j].any() else 0.0
                  for j in range(q.shape[1])])
    c = np.where(d["offs"] == 0)[0][0]
    return P, (E / E[c] if E[c] != 0 else E * np.nan), len(q)


def drift_weights(dref, dsim, pi, nbin=6):
    """weights that make the simulation's drift distribution match the data's, in the
    window the two share.  Doc 47 sec 10.4: ROI admission grows with the diffusion width,
    so a pooled P inherits the same drift-mixing artefact that flipped the SBND core
    claim -- match or say you did not."""
    a, b = dref["t_ns"][dref["plane"] == pi], dsim["t_ns"][dsim["plane"] == pi]
    if len(a) == 0 or len(b) == 0:
        return None
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, nbin + 1)
    na, _ = np.histogram(a, edges)
    nb, _ = np.histogram(b, edges)
    w = np.zeros(nbin)
    ok = nb > 0
    w[ok] = na[ok] / nb[ok]
    idx = np.clip(np.digitize(dsim["t_ns"], edges) - 1, 0, nbin - 1)
    inw = (dsim["t_ns"] >= lo) & (dsim["t_ns"] <= hi)
    return w[idx] * inw, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sim", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = read(a.data)
    sims = {}
    for spec in a.sim:
        lab, _, path = spec.partition("=")
        sims[lab] = read(path)

    rows = []
    print("A. admission P(D) -- fraction of profiles whose cell D wires from the TRAJECTORY")
    print("   lies inside an ROI at all.  `gauss` is exactly zero outside an ROI, so this is")
    print("   an exact flag, not a threshold on charge.")
    hdr = "   %-16s %-2s %7s | " + " ".join("%7s" % ("D=%+d" % o) for o in data["offs"])
    print(hdr % ("sample", "pl", "n"))
    for lab, d in [("data (PDVD)", data)] + list(sims.items()):
        for pi, pl in enumerate(PLANES):
            r = pe(d, d["plane"] == pi)
            if r is None:
                continue
            P, E, n = r
            print(("   %-16s %-2s %7d | " + " ".join("%7.4f" % x for x in P)) % (lab, pl, n))
            rows.append(dict(sample=lab, plane=pl, kind="P", n=n,
                             **{"d%+d" % o: "%.5g" % x for o, x in zip(d["offs"], P)}))

    print()
    print("B. amplitude E(D) -- mean charge at D GIVEN the cell is admitted, over the same")
    print("   at D=0.  This is what the ROI's own filters keep of what they let in.")
    print(hdr % ("sample", "pl", "n"))
    for lab, d in [("data (PDVD)", data)] + list(sims.items()):
        for pi, pl in enumerate(PLANES):
            r = pe(d, d["plane"] == pi)
            if r is None:
                continue
            P, E, n = r
            print(("   %-16s %-2s %7d | " + " ".join("%7.4f" % x for x in E)) % (lab, pl, n))
            rows.append(dict(sample=lab, plane=pl, kind="E", n=n,
                             **{"d%+d" % o: "%.5g" % x for o, x in zip(d["offs"], E)}))

    print()
    print("C. the split, at |D| = 2 and 3, in MATCHED DRIFT WINDOWS (the simulation reweighted")
    print("   to the data's drift distribution over the window the two share; ROI admission")
    print("   grows with the diffusion width, so a pooled ratio is not safe -- sec 10.4).")
    print("   %-16s %-2s | %-19s | %-19s | %s"
          % ("sample", "pl", "P(|D|=2)  ratio", "P(|D|=3)  ratio", "E(|D|=2)   ratio   drift window"))
    for pi, pl in enumerate(PLANES):
        dref = pe(data, data["plane"] == pi)
        if dref is None:
            continue
        Pd, Ed, nd = dref
        def band(P, k):
            j = [i for i, o in enumerate(data["offs"]) if abs(o) == k]
            return float(np.mean([P[i] for i in j]))
        print("   %-16s %-2s | %8.4f  %7s | %8.4f  %7s | %8.4f %7s"
              % ("data (PDVD)", pl, band(Pd, 2), "-", band(Pd, 3), "-", band(Ed, 2), "-"))
        for lab, d in sims.items():
            dw = drift_weights(data, d, pi)
            if dw is None:
                continue
            w, (lo, hi) = dw
            sel = (d["plane"] == pi) & (w > 0)
            if sel.sum() < 20:
                continue
            q = d["q"][sel]
            ww = w[sel]
            inroi = q != 0.0
            P = (inroi * ww[:, None]).sum(axis=0) / ww.sum()
            E = np.array([np.average(q[inroi[:, j], j], weights=ww[inroi[:, j]])
                          if inroi[:, j].any() else 0.0 for j in range(q.shape[1])])
            c = np.where(d["offs"] == 0)[0][0]
            E = E / E[c] if E[c] != 0 else E * np.nan
            p2, p3, e2 = band(P, 2), band(P, 3), band(E, 2)
            print("   %-16s %-2s | %8.4f  %7s | %8.4f  %7s | %8.4f %7s   %.0f-%.0f us  n=%d"
                  % (lab, pl, p2, "x%.1f" % (band(Pd, 2) / p2) if p2 > 0 else "inf",
                     p3, "x%.1f" % (band(Pd, 3) / p3) if p3 > 0 else "inf",
                     e2, "x%.2f" % (band(Ed, 2) / e2) if e2 > 0 else "inf",
                     lo * 1e-3, hi * 1e-3, int(sel.sum())))
            rows.append(dict(sample=lab + " (drift-matched)", plane=pl, kind="P_matched",
                             n=int(sel.sum()),
                             **{"d%+d" % o: "%.5g" % x for o, x in zip(d["offs"], P)}))

    print()
    print("D. the ROI threshold basis at the profile's own centre channel (cal_RMS, the")
    print("   `wiener` trace summary).  On DATA this is a percentile spread computed on")
    print("   waveforms that contain signal, so it is contaminated by occupancy -- read it")
    print("   as an upper bound on the noise, not as the noise.")
    print("   %-16s %-2s %10s %10s %10s" % ("sample", "pl", "thresh med", "q(D=0) med", "ratio"))
    for lab, d in [("data (PDVD)", data)] + list(sims.items()):
        c = np.where(d["offs"] == 0)[0][0]
        for pi, pl in enumerate(PLANES):
            m = (d["plane"] == pi) & np.isfinite(d["thresh"]) & (d["q"][:, c] != 0)
            if m.sum() < 20:
                continue
            th, q0 = d["thresh"][m], d["q"][m, c]
            print("   %-16s %-2s %10.1f %10.1f %10.1f"
                  % (lab, pl, np.median(th), np.median(q0), np.median(q0 / th)))
            rows.append(dict(sample=lab, plane=pl, kind="thresh", n=int(m.sum()),
                             thresh_med="%.5g" % np.median(th), q0_med="%.5g" % np.median(q0),
                             q0_over_thresh="%.5g" % np.median(q0 / th)))

    if a.out:
        keys = ["sample", "plane", "kind", "n"] + sorted({k for r in rows for k in r} -
                                                         {"sample", "plane", "kind", "n"})
        with open(a.out + ".tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", restval="")
            w.writeheader()
            w.writerows(rows)
        print("\n  -> %s.tsv" % a.out)


if __name__ == "__main__":
    sys.exit(main())

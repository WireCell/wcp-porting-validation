#!/usr/bin/env python3
"""pr/132: refresh the pi0 mass-peak fit with the pairing-pass sample.

Reuses pr126_pi0_peak.py's estimator (bounded unbinned truncated-Gaussian ML
on [100,185], bootstrapped whole) unchanged.  Three cells:

  base       the pr/126 pooled sample (all origins, vertex convention, `now`
             energies, min(E)>15) -- the 0.833 [0.808,0.855] cell of sec 4h
  overlay    the pr/132 pairing-pass pairs (pi0scan-0829-agent) with a
             vertex-convention mass and both gammas > 15 MeV
  union      base + overlay

Each is reported at the 0.80 scale (implied fudge) and re-expressed at 0.84
(where would the peak sit after the flip -- the alignment check).

READ-ONLY.  ./pr132_pi0_peak_refresh.py [--tsv docs/pr/pr132-pi0-peak.tsv]
"""
import argparse, csv, glob, json, os, sys, importlib.util
import numpy as np

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_peak", os.path.join(SX, "scripts", "pr126_pi0_peak.py"))
PK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PK)

TAG = "pi0scan-0829-agent"


def load_overlay():
    out = []
    for f in sorted(glob.glob(os.path.join(SX, "em_labels", TAG, "labels-evt*.json"))):
        d = json.load(open(f))
        pio = d.get("pio")
        if not pio:
            continue
        m = pio.get("mass_vertex_convention")
        g = pio.get("gammas") or {}
        e1 = (g.get("1") or {}).get("energy") or 0
        e2 = (g.get("2") or {}).get("energy") or 0
        conf = d.get("confidence")
        if m and m > 0 and min(e1, e2) > PK.E_MIN:
            out.append((d["eventNo"], m, conf))
    return out


def cell(name, x):
    x = np.asarray(x, float)
    mu = PK.peak_fit(x)
    lo, hi = PK.boot(x, PK.peak_fit)
    n_in = int(((x >= PK.WIN[0]) & (x <= PK.WIN[1])).sum())
    fudge = PK.to_fudge(mu)
    # after the 0.84 flip every mass scales by 0.80/0.84
    mu84 = mu * 0.80 / 0.84
    return dict(cell=name, n=len(x), n_in=n_in,
                peak="%.1f" % mu, ci68="[%.1f,%.1f]" % (lo, hi),
                fudge="%.3f" % fudge,
                fudge_ci="[%.3f,%.3f]" % (PK.to_fudge(lo), PK.to_fudge(hi)),
                peak_at_084="%.1f" % mu84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("--exclude-low-conf", action="store_true",
                    help="drop confidence=low overlay pairs")
    a = ap.parse_args()

    base = PK.load("vtx", "now", gate="all")
    ov = load_overlay()
    if a.exclude_low_conf:
        ov = [t for t in ov if t[2] != "low"]
    ov_m = [t[1] for t in ov]
    rows = [cell("base-pooled(pr126-4h)", base),
            cell("overlay(pi0scan)", ov_m),
            cell("union", np.concatenate([base, ov_m]) if ov_m else base)]
    print("overlay pairs used: %s" % [(e, "%.1f" % m, c) for e, m, c in ov])
    for r in rows:
        print("  %-22s n=%-3d n_in=%-3d peak=%s %s fudge=%s %s peak@0.84=%s"
              % (r["cell"], r["n"], r["n_in"], r["peak"], r["ci68"],
                 r["fudge"], r["fudge_ci"], r["peak_at_084"]))
    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

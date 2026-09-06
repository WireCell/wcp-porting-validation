#!/usr/bin/env python3
"""doc pdvd/47 -- write amplitude-scaled copies of a WCT EmpiricalNoiseModel spectra file
(every entry's 'amps' and 'const' multiplied by k) so a sim arm can test how the SP
transverse width depends on the noise level through the ROI thresholds.

Usage: d47_scale_noise.py --k 0.5 --outdir /home/xqian/tmp/xtrack/noise <spectra.json.bz2> ...
  -> <outdir>/<basename>.x<k>.json.bz2  (referenced by the drivers' noise_tag='x<k>' TLA)
"""
import argparse, bz2, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--k", type=float, required=True); ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for f in a.files:
        j = json.load(bz2.open(f))
        for e in j:
            e["amps"] = [x * a.k for x in e["amps"]]; e["const"] = e["const"] * a.k
        out = os.path.join(a.outdir, os.path.basename(f).replace(".json.bz2", ".x%g.json.bz2" % a.k))
        with bz2.open(out, "wt") as fo:
            json.dump(j, fo)
        print(out, len(j), "entries x%g" % a.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())

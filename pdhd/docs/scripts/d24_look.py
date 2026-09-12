#!/usr/bin/env python3
"""One picture: q(L) against every candidate clock, for one chain.

Top panel  : dQ/dx vs path length L, with the Bragg reference overlaid.
Lower panels: the wire/time coordinate of the same points vs L, with integer
wire crossings marked, so "does an extremum sit on a wire crossing" is a look,
not a periodogram.
"""
import json, sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")

def arr(v):
    return np.array([np.nan if t is None else t for t in v], float)

def main(event, cid, out):
    d = json.load(open("%s/smprep-%s-c%s.json" % (PREP, event, cid)))
    m = d["muon"]
    L, q, rr = arr(m["L"]), arr(m["q"]), arr(m["rr"])
    pw, pu, pv, pt = (arr(m[k]) for k in ("pw", "pu", "pv", "pt"))
    ref = json.load(open("%s/dqdx_ref_pdhd.json" % PREP))
    g = ref["grid"]
    rg = np.linspace(g["lo"], g["hi"], len(ref["muon"])) if "lo" in g else np.arange(len(ref["muon"]))*0.25
    fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    ax[0].plot(L, q/1000.0, ".-", lw=0.8, ms=3, color="k")
    ax[0].plot(L, np.interp(rr, rg, np.array(ref["muon"])/1000.0), "-", color="tab:red",
               lw=1.2, label="muon Bragg reference (rr)")
    ax[0].set_ylabel("dQ/dx [ke/cm]"); ax[0].legend(fontsize=8)
    ax[0].set_title("%s/%s  %d pts  %.1f cm   (stop at L=%.1f)" % (event, cid, len(L), L[-1], L[-1]))
    for i, (p, nm) in enumerate([(pw, "pw  (W / collection wire)"),
                                 (pu, "pu  (U wire)"),
                                 (pv, "pv  (V wire)"),
                                 (pt, "pt  (time slice)")]):
        a = ax[i+1]
        a.plot(L, p - np.nanmin(p), "-", lw=1.0, color="tab:blue")
        a.set_ylabel(nm, fontsize=8)
        span = np.nanmax(p) - np.nanmin(p)
        if np.isfinite(span) and span < 200:   # mark integer crossings when legible
            base = np.floor(np.nanmin(p))
            for k in np.arange(0, span + 2):
                a.axhline(k + (base - np.nanmin(p)) + 0, color="grey", lw=0.4, alpha=0.5)
        a.text(0.99, 0.9, "span %.2f  => %.2f cm per unit" % (span, L[-1]/max(span, 1e-9)),
               transform=a.transAxes, ha="right", fontsize=8)
    ax[-1].set_xlabel("L along chain [cm]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)

main(sys.argv[1], sys.argv[2], sys.argv[3])

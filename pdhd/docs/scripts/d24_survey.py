#!/usr/bin/env python3
"""Per-item geometry survey of the STM chain payloads.

For every smprep-*.json: the chain's advance per wire on each plane (the
"which clock" candidates), its angle to the collection-wire direction, and
basic dQ/dx stats.  Wrapped PDHD induction planes make pu/pv jump by large
integers at a wrap, so every rate is a MEDIAN of |dp|/dL over steps, with the
top 10% of |dp| discarded first.
"""
import json, glob, sys, math
import os
import numpy as np

def rate(p, dL):
    """median |dp/dL| over steps, robust to wrap jumps."""
    p = np.array([np.nan if v is None else v for v in p], float)
    d = np.abs(np.diff(p))
    g = np.isfinite(d) & (dL > 1e-6)
    if g.sum() < 5:
        return np.nan
    dd = d[g] / dL[g]
    cut = np.percentile(dd, 90)
    keep = dd[dd <= cut]
    return float(np.median(keep)) if keep.size else np.nan

def main(prepdir, out):
    rows = []
    for f in sorted(glob.glob(prepdir + "/smprep-*.json")):
        d = json.load(open(f))
        m = d["muon"]
        n = len(m["L"])
        if n < 30:
            continue
        L = np.array(m["L"], float)
        dL = np.diff(L)
        q = np.array(m["q"], float)
        x, y, z = (np.array(m[k], float) for k in "xyz")
        # end-to-end direction
        v = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
        vn = v / (np.linalg.norm(v) + 1e-12)
        rw = rate(m["pw"], dL); ru = rate(m["pu"], dL); rv = rate(m["pv"], dL)
        rt = rate(m["pt"], dL)
        rows.append(dict(
            key=d["event"] + "/" + str(d["cluster_id"]), f=f,
            npts=n, len_cm=float(L[-1]),
            dpw=rw, dpu=ru, dpv=rv, dpt=rt,
            cm_w=(np.nan if not rw or not np.isfinite(rw) or rw <= 0 else 1.0/rw),
            cm_u=(np.nan if not ru or not np.isfinite(ru) or ru <= 0 else 1.0/ru),
            cm_v=(np.nan if not rv or not np.isfinite(rv) or rv <= 0 else 1.0/rv),
            q_med=float(np.median(q[np.isfinite(q)])) if np.isfinite(q).any() else np.nan,
            q_rms=float(np.std(q[np.isfinite(q)])) if np.isfinite(q).any() else np.nan,
            dirx=float(vn[0]), diry=float(vn[1]), dirz=float(vn[2]),
        ))
    keys = ["key","npts","len_cm","dpw","dpu","dpv","dpt","cm_w","cm_u","cm_v",
            "q_med","q_rms","dirx","diry","dirz","f"]
    with open(out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4g" % r[k] if isinstance(r[k], float) else str(r[k]))
                               for k in keys) + "\n")
    print("wrote %s: %d items" % (out, len(rows)))
    cw = np.array([r["cm_w"] for r in rows], float)
    cw = cw[np.isfinite(cw)]
    print("cm per W wire: median %.2f  q90 %.2f  q99 %.2f  max %.1f  n>=4cm/wire %d/%d"
          % (np.median(cw), np.percentile(cw,90), np.percentile(cw,99), cw.max(),
             (cw>=4).sum(), cw.size))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""The statistic of doc 24 sec 3c: lambda / cm-per-cell.

A lattice origin makes the dQ/dx correlation length lambda proportional to the
track's cm-per-cell, i.e. holds this ratio constant (= 1 if the scale is one
cell).  Quoting the two ranges separately invites "your top bucket is 21 tracks
with a wide internal spread", so the bucket q25-q75 is printed with every row.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")
PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
from d24_fold import collect

MAXLAG = 40

def lam(sel):
    acc = np.zeros(MAXLAG+1); cnt = np.zeros(MAXLAG+1)
    for it in sel:
        r = it["r"]-it["r"].mean(); v = np.var(r); n = r.size
        if v <= 0: continue
        for k in range(MAXLAG+1):
            if n-k < 20: break
            acc[k] += np.mean(r[:n-k]*r[k:])/v*(n-k); cnt[k] += (n-k)
    a = acc/np.maximum(cnt, 1); b = np.where(a < 1/np.e)[0]
    if b.size and b[0] > 0:
        k = b[0]; return (k-1 + (a[k-1]-1/np.e)/(a[k-1]-a[k]))*0.6
    return MAXLAG*0.6

items = collect(PREP)
out = ["lambda / cm-per-cell -- 1.0 is what a geometric (one-cell) scale requires.", ""]
for key, lab in (("pw_cm", "collection wire"), ("pt_cm", "drift time slice")):
    out.append(lab)
    for lo, hi in ((0, 0.8), (0.8, 1.5), (1.5, 3.0), (3.0, 8.0), (8.0, 1e9)):
        sel = [it for it in items if lo <= it[key] < hi]
        if len(sel) < 8: continue
        c = np.array([it[key] for it in sel]); L = lam(sel)
        out.append("   cm/cell [%-4g,%-5g) n=%3d  med %6.2f (q25 %5.2f q75 %6.2f)"
                   "  lambda %.2f cm  ratio %.2f"
                   % (lo, hi, len(sel), np.median(c), np.percentile(c, 25),
                      np.percentile(c, 75), L, L/np.median(c)))
txt = "\n".join(out); print(txt)
open(os.path.join(FIGS, "24_ratio.txt"), "w").write(txt+"\n")

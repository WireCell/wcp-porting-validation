#!/usr/bin/env python3
"""doc pdvd/113 sec 7 -- measure the sideways step of the v3 seed in the reduced levels (read-only).

Same frame as d113_spot_figs.py (image-only charge-weighted PCA within the window, built on the production arm).  In
slabs of 1 cm along s: the image points' e2 range and charge-weighted e2 centroid, the seed's e2 and ridge offset, and the
fit's e2, for production and for retile_mode none.

Usage: d113_v3_step.py > figs/113_v3_step.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import d111s_common as C
from d111_spot_frame import frame
from d111s_replay import SPOTS
from d113_spot_figs import load

det, ev, cl, click, win = SPOTS["v3"]
click = np.array(click)
arms = [("d113vbase", "production"), ("d113vnone", "none")]
X = {a: load(det, a, f"/home/xqian/tmp/d113/arm_{a}", ev, cl, click, win) for a, _ in arms}
near = np.linalg.norm(X["d113vbase"]["Pimg"] - click, axis=1) < win
c, e1, e2, e3 = frame(X["d113vbase"]["Pimg"][near], X["d113vbase"]["qimg"][near])
pr = lambda P: ((P - c) @ e1, (P - c) @ e2, (P - c) @ e3)
si, ai, bi = pr(X["d113vbase"]["Pimg"][near]); qi = X["d113vbase"]["qimg"][near]
print(f"# doc pdvd/113 v3 step: {det} {ev} cl{cl}, frame from the production image within {win:.0f} cm of the click")
print("| s slab (cm) | image n | image e2 min / max | image e2 centroid (q-weighted) | image e2 share > +1 cm | "
      + " | ".join(f"{lab} seed e2 / ridge off | {lab} fit e2" for _, lab in arms) + " |")
print("|---|---|---|---|---|" + "---|---|" * len(arms))
for s0 in np.arange(-8.0, 3.0, 1.0):
    m = (si >= s0) & (si < s0 + 1)
    row = [f"[{s0:+.0f}, {s0+1:+.0f})", f"{m.sum()}"]
    if m.sum():
        row += [f"{ai[m].min():+.2f} / {ai[m].max():+.2f}", f"{np.average(ai[m], weights=np.clip(qi[m], 1, None)):+.2f}",
                f"{100*np.mean(ai[m] > 1.0):.0f} %"]
    else:
        row += ["-", "-", "-"]
    for a, lab in arms:
        seed = X[a]["seed"]; R = X[a]["R"]
        Xs, _ = C.polyline_samples(seed, 0.3)
        ss, sa, _ = pr(Xs)
        ms = (ss >= s0) & (ss < s0 + 1) & (np.linalg.norm(Xs - click, axis=1) <= 1.3 * win)
        row.append(f"{np.mean(sa[ms]):+.2f} / {np.max(R.offset(Xs[ms])):.2f}" if ms.any() else "-")
        F = X[a]["fit"]
        fs, fa, _ = pr(F)
        mf = (fs >= s0) & (fs < s0 + 1)
        row.append(f"{np.mean(fa[mf]):+.2f}" if mf.any() else "-")
    print("| " + " | ".join(row) + " |")

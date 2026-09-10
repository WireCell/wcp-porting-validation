#!/usr/bin/env python3
"""doc pdvd/65 (T7) -- the profile's autocorrelation length, per detector.

Doc 56 sec 6: adjacent 0.6 cm fit points are positively correlated (rebinning
RAISES the point-to-point scatter), so "the right first measurement is the
profile's autocorrelation length per detector, not a step change".  Here it
is: on the plateau (rr 20-80 cm, q > 0), the residual of log q about a
running median (window 9 points), lag-k autocorrelation for k = 1..8 points,
pooled over items; the lag where it first drops below 1/e, in points and cm
(median L step).  Usage: python3 d65_autocorr.py <prep_dir> [<prep_dir> ...]
"""
import sys, glob, json, os
import numpy as np

def profile_arrays(pay):
    m = pay["muon"]; rr = np.asarray(m["rr"], float); q = np.asarray(m["q"], float); L = np.asarray(m["L"], float)
    o = np.argsort(rr); return rr[o], q[o], L[o]

def running_median(x, w=9):
    h = w // 2; out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - h): i + h + 1])
    return out

for prep in sys.argv[1:]:
    files = [f for f in glob.glob(prep + "/smprep-*.json")]
    acc = {k: [] for k in range(1, 9)}; steps = []; nitems = 0; scat = []
    for f in files:
        pay = json.load(open(f))
        if not pay.get("muon") or not pay["muon"].get("q"): continue
        rr, q, L = profile_arrays(pay)
        sel = (rr > 20) & (rr < 80) & (q > 0)
        if sel.sum() < 30: continue
        nitems += 1
        lq = np.log(q[sel]); r = lq - running_median(lq)
        r = r - r.mean()
        var = float((r * r).mean())
        if var <= 0: continue
        for k in acc:
            if len(r) > k + 5:
                acc[k].append(float((r[:-k] * r[k:]).mean() / var))
        steps.append(float(np.median(np.abs(np.diff(L[sel])))))
        scat.append(float(np.median(np.abs(np.diff(lq)))))
    step = float(np.median(steps)) if steps else float("nan")
    print("%s: %d items with >= 30 plateau points; median L step %.3f cm; median |d log q| point-to-point %.3f" % (os.path.basename(prep.rstrip("/")), nitems, step, np.median(scat) if scat else float("nan")))
    print("  lag(points)  lag(cm)  autocorr median  [p25, p75]")
    below = None
    for k in acc:
        a = np.array(acc[k])
        if a.size == 0: continue
        med = float(np.median(a))
        print("  %5d  %8.2f  %8.3f   [%.3f, %.3f]" % (k, k * step, med, np.percentile(a, 25), np.percentile(a, 75)))
        if below is None and med < np.exp(-1): below = k
    print("  first lag with median autocorrelation below 1/e: %s points (%s cm)" % (below, ("%.2f" % (below * step)) if below else "n/a"))

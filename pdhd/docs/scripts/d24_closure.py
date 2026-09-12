#!/usr/bin/env python3
"""Does the cell modulation ACCOUNT for the observed wave period?

If the wave is the cell fold, then in a stratum where a cell is resolvable the
Lomb-Scargle peak of the residual against that cell coordinate must sit at
period = 1 cell, and the peak in path length must sit at cm-per-cell.
Both are checked, against the circular-shift null and against the OTHER
plane's coordinate as a control.
"""
import os
import numpy as np

FIGS = os.environ.get("D24_FIGS", "../figs")

from scipy.signal import lombscargle
from d24_fold import collect

PREP = os.environ.get("D24_PREP",
                      "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/prep-pdhd")
items = collect(PREP)
rng = np.random.default_rng(20260912)

def ls_peak(c, r, pmin, pmax, n=3000):
    f = np.linspace(1.0/pmax, 1.0/pmin, n)
    p = lombscargle(c.astype(float), r.astype(float), 2*np.pi*f, normalize=True)
    k = int(np.argmax(p))
    return 1.0/f[k], float(p[k])

out = []
for coord, lab, other in (("pw", "W wire", "pt"), ("pt", "time slice", "pw")):
    sel = [it for it in items if it[coord+"_spc"] >= 6 and it[other+"_spc"] < 6]
    out.append("")
    out.append("=== %s: stratum where the %s cell is resolvable and the other is not (%d items) ==="
               % (coord, lab, len(sel)))
    Ts, Tc, Tl, cmcell = [], [], [], []
    for it in sel:
        c = it[coord] - it[coord][0]
        span = abs(c[-1]-c[0])
        if span < 3: continue                        # need >= 3 cycles of a 1-cell period
        T, pk = ls_peak(np.abs(c), it["r"], 0.3, min(8.0, span/3.0))
        Tc.append(T)
        TL, _ = ls_peak(it["L"], it["r"], 1.8, it["L"][-1]/3.0)
        Tl.append(TL)
        cmcell.append(it["L"][-1]/max(span, 1e-9))
    Tc = np.array(Tc); Tl = np.array(Tl); cmcell = np.array(cmcell)
    if Tc.size < 5:
        out.append("  too few items with >= 3 cells crossed"); continue
    out.append("  items with >= 3 cells crossed: %d" % Tc.size)
    out.append("  LS peak period in %s units: median %.3f cells   [q25 %.3f, q75 %.3f]   frac in [0.8,1.25]: %.2f"
               % (coord, np.median(Tc), np.percentile(Tc,25), np.percentile(Tc,75),
                  float(((Tc>=0.8)&(Tc<=1.25)).mean())))
    rat = Tl/cmcell
    out.append("  LS peak period in cm:        median %.2f cm ; cm per cell median %.2f cm"
               % (np.median(Tl), np.median(cmcell)))
    out.append("  ratio (period_cm / cm_per_cell): median %.3f  [q25 %.3f, q75 %.3f]  frac in [0.8,1.25]: %.2f"
               % (np.median(rat), np.percentile(rat,25), np.percentile(rat,75),
                  float(((rat>=0.8)&(rat<=1.25)).mean())))
    # null: same statistic on circularly shifted residuals
    nullfrac = []
    for _ in range(200):
        acc = []
        for it in sel:
            c = it[coord]-it[coord][0]; span = abs(c[-1]-c[0])
            if span < 3: continue
            r = np.roll(it["r"], rng.integers(1, it["r"].size))
            T, _ = ls_peak(np.abs(c), r, 0.3, min(8.0, span/3.0))
            acc.append(T)
        acc = np.array(acc); nullfrac.append(float(((acc>=0.8)&(acc<=1.25)).mean()))
    nullfrac = np.array(nullfrac)
    obs = float(((Tc>=0.8)&(Tc<=1.25)).mean())
    out.append("  NULL (circular shift): frac in [0.8,1.25] = %.3f +- %.3f ; observed %.3f ; p=%.4f"
               % (nullfrac.mean(), nullfrac.std(), obs, ((nullfrac>=obs).sum()+1)/201.0))
txt = "\n".join(out); print(txt); open(os.path.join(FIGS, "24_closure.txt"), "w").write(txt+"\n")

#!/usr/bin/env python3
"""Waveforms at cluster 97's CROSSING-POINT pile (y=175.58, z=219.04, u 339->342 at
flash-96 T0): is the ~22us-long charge at the crossing point real continuous signal
(track segment along drift / late charge) or a random coincident blip?

Answer: continuous with the track's own crossing pulse on all three planes (the W
raw trace never returns to baseline between the main peak and the late charge) =>
the crosser's own late-arriving charge, NOT a coincidence.
See 16_pdvd-clus97-crosser-evt298567.md.

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    OMP_NUM_THREADS=4 python3 docs/qlmatch/scripts/check_clus97_crossing_pile_waveform.py
"""
import sys
import numpy as np

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/scripts")
import check_clus97_tail_waveforms as W   # reuse loaders/geometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# crossing-point pile: raw x -212.84 .. -216.09, fixed (y,z)
Y, Z = 175.58, 219.04
XDEEP, XIN = -216.09, -212.84          # deepest & shallowest raw x of the pile
a, f = W.which_af(Y, Z)
print("anode", a, "face", f)

V = None
import json
d = json.load(open(W.DUMP)); V = d["drift_speed"]

wx = W.geom.wplane_x_cm(W.STORE, a, f)
t_deep = (XDEEP - wx) * (-1) / V / W.TICK_US
t_in   = (XIN   - wx) * (-1) / V / W.TICK_US
print(f"pile predicted ticks: {t_in:.0f} .. {t_deep:.0f}  (span {abs(t_deep-t_in):.0f} ticks)")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
for cc, (p, nm) in enumerate(((0, "U"), (1, "V"), (2, "W"))):
    ax = axes[cc]
    ch = W.wire_of(a, f, p, Y, Z)
    if ch is None:
        print(nm, "no wire"); continue
    lo, hi = int(min(t_in, t_deep)) - 150, int(max(t_in, t_deep)) + 150
    t = np.arange(lo, hi)
    (Fo, Ro), (Fg, Rg) = W.frames(a)["o"], W.frames(a)["g"]
    ax2 = ax.twinx()
    if ch in Ro:
        w = Fo[Ro[ch]][lo:hi]
        ax.plot(t, w, color="0.45", lw=0.9, label="raw ADC (orig, pre-NF)")
        print(f"{nm} ch {ch}: raw peak {w.max():.0f} min {w.min():.0f} in window")
    if ch in Rg:
        w = Fg[Rg[ch]][lo:hi]
        ax2.plot(t, w, color="tab:red", lw=1.4, label="decon (gauss)")
        print(f"{nm} ch {ch}: decon peak {w.max():.0f}")
    ax.axvspan(min(t_in, t_deep), max(t_in, t_deep), color="tab:blue", alpha=0.18,
               label="pile tick span (past gate)")
    ax.set_xlabel("tick"); ax.set_ylabel("raw ADC", color="0.35")
    ax2.set_ylabel("decon", color="tab:red")
    ax.set_title(f"{nm} ch {ch} @ crossing point (y={Y}, z={Z})", fontsize=10)
    ax.grid(alpha=0.25)
    if cc == 0:
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.suptitle("PDVD evt298567 clus97: waveforms at the cathode-crossing point "
             "(the 13-pt pile past the containment gate)", fontsize=12)
fig.tight_layout()
out = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/pics/clus97_crossing_pile_waveforms.png"
fig.savefig(out, dpi=105)
print("wrote", out)

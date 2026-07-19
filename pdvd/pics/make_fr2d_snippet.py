#!/usr/bin/env python3
"""2D time-domain field-response inset (V plane) for the sim-chain diagram.

Loads the PD-VD field-response file and renders the induced-current response
as a wire-offset x time image for the V (induction) plane -- the classic
bipolar field-response signature.  Per-wire response = mean over the 10
sub-pitch impact bins (matching field_response_2d.py).

Output: pdvd/pics/sim_chain_src/field_response_2d_time_V.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from wirecell.sigproc.response import persist
from wirecell.sigproc.response.arrays import pr2array
from wirecell.util.fileio import wirecell_path

FR_FILE = "protodunevd_FR_imbalance3p_260501.json.bz2"
PLANE, PNAME = 1, "V"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sim_chain_src", "field_response_2d_time_V.png")


def find_wcdata(fn):
    for d in wirecell_path():
        c = os.path.join(d, fn)
        if os.path.exists(c):
            return c
    raise FileNotFoundError(fn)


def main():
    fr = persist.load(find_wcdata(FR_FILE))
    pr = [p for p in fr.planes if p.planeid == PLANE][0]
    r210, _ = pr2array(pr)                       # (210, nt)
    nt = r210.shape[1]
    per_wire = r210.reshape(-1, 10, nt).mean(axis=1)   # (21, nt)
    nw = per_wire.shape[0]

    period_us = fr.period / 1000.0               # ns -> us
    t = np.arange(nt) * period_us
    # zoom to the active window (where |response| is non-negligible)
    env = np.abs(per_wire).sum(axis=0)
    active = np.where(env > 0.02 * env.max())[0]
    t0, t1 = max(0, active[0] - 8), min(nt, active[-1] + 12)
    # restrict to a few wires either side of the track so the neighbour
    # induction wings are visible (not swamped by empty rows)
    wire_off = np.arange(nw) - nw // 2
    keep = np.abs(wire_off) <= 6
    disp = per_wire[keep, t0:t1]
    wire_off = wire_off[keep]
    t = t - t[t0]                                # start the window at 0 µs

    # vmax below the central-wire peak so the neighbour wings show contrast
    vmax = np.percentile(np.abs(disp), 88.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    extent = (t[t0], t[t1 - 1], wire_off[0] - 0.5, wire_off[-1] + 0.5)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(disp, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xlabel("time  [µs]", fontsize=9)
    ax.set_ylabel("wire offset from track", fontsize=9)
    ax.set_title("field response  —  %s plane (induction)" % PNAME,
                 fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("induced current  [a.u.]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT, "nt=%d zoom=[%d,%d] vmax=%.3g" % (nt, t0, t1, vmax))


if __name__ == "__main__":
    main()

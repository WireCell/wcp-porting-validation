#!/usr/bin/env python3
"""PDVD 039252 evt298567: is cluster 97's tail real charge or an imaging artefact?

Reads the RAW ADC (orig, pre-NF) and the DECONVOLVED (DNN-ROI gauss) U/V/W waveforms at
each tail clump's predicted (channel, tick) and compares them with the cluster's own real
track end. Answer: the tail is REAL charge -- clean bipolar induction pulses on U and V
and a clean unipolar collection pulse on W, coincident at the predicted tick, and
BRIGHTER than the real track end. See 16_pdvd-clus97-crosser-evt298567.md §4.

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    OMP_NUM_THREADS=4 python3 docs/qlmatch/scripts/check_clus97_tail_waveforms.py

Inputs (read-only):
    work/039252_0_walkfloor/calib-evt298567.json                  3-D points
    input_data/run039252/evt_0/protodune-orig-frames-anode{6,7}.tar.bz2   raw ADC (pre-NF)
    work/039252_0_keep/protodune-sp-dnnroi-frames-anode{6,7}.tar.bz2      decon (gauss)

Gotchas (cost real time -- see doc §9):
  - M8: `orig` = PRE-NF raw ADC; `raw` inside the SP archive = POST-NF. Not the same thing.
  - The orig archive's tickinfo says 512 ns, the SP archive says 500 ns (the known PDVD
    top-CRP tick mislabel). The SP chain read them at 500 ns, so the tick INDEX is common
    to both; only the label differs. Index-based comparison is safe.
  - The wire index for the ANGLED U/V planes cannot be found by projecting onto the pitch
    direction from wire 0's centre: angled wires are clipped by the face boundary, so their
    centres shift along the wire and the projection is not linear in wire index. It gives a
    plausible-but-wrong channel (it round-trips for W and fails for U/V). Brute-force the
    band polygons instead -- that is what `wire_of` does.
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom  # noqa: E402

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
DUMP = os.path.join(BASE, "work/039252_0_walkfloor/calib-evt298567.json")
ORIG = os.path.join(BASE, "input_data/run039252/evt_0/protodune-orig-frames-anode%d.tar.bz2")
SPDN = os.path.join(BASE, "work/039252_0_keep/protodune-sp-dnnroi-frames-anode%d.tar.bz2")
WIRES = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"
PICS = os.path.join(BASE, "docs/qlmatch/pics")
TICK_US = 0.5
TAIL_X = -228.0

STORE = geom.load_store(WIRES)
_PG = {}


def pg(a, f, p):
    if (a, f, p) not in _PG:
        _PG[(a, f, p)] = geom.PlaneGeom(STORE, a, f, p)
    return _PG[(a, f, p)]


def wire_of(a, f, p, y, z):
    """Channel whose +-half-pitch band contains (y,z). Brute force over band polygons:
    the pitch-projection shortcut is wrong for the angled U/V planes (see module docstring)."""
    g = pg(a, f, p)
    for k in range(len(g.chans)):
        if geom.point_in_poly(y, z, geom.order_polygon(g.band_quad(k))):
            return g.channel(k)
    return None


def which_af(y, z):
    """(anode,face) of the TOP volume whose W-plane y-z bbox contains (y,z)."""
    for a in (4, 5, 6, 7):
        for f in (0, 1):
            g = pg(a, f, 2)
            ys = np.concatenate([g.tails[:, 0], g.heads[:, 0]])
            zs = np.concatenate([g.tails[:, 1], g.heads[:, 1]])
            if ys.min() - .3 <= y <= ys.max() + .3 and zs.min() - .3 <= z <= zs.max() + .3:
                return a, f
    return None, None


def _load(path, prefixes):
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            for k in prefixes:
                if m.name.startswith(k + "_"):
                    out[k] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return out


_FR = {}


def frames(a):
    """{'o': (raw-ADC frame pedestal-subtracted, chan->row), 'g': (decon frame, chan->row)}"""
    if a not in _FR:
        o = _load(ORIG % a, ["frame_orig", "channels_orig"])
        s = _load(SPDN % a, ["frame_gauss%d" % a, "channels_gauss%d" % a])
        Fo = o["frame_orig"] - np.median(o["frame_orig"], axis=1, keepdims=True)
        _FR[a] = dict(o=(Fo, {c: i for i, c in enumerate(o["channels_orig"])}),
                      g=(s["frame_gauss%d" % a],
                         {c: i for i, c in enumerate(s["channels_gauss%d" % a])}))
    return _FR[a]


def clumps_of(P, link=6.0):
    used = np.zeros(len(P), bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        g = [i]
        used[i] = True
        grew = True
        while grew:
            grew = False
            for j in range(len(P)):
                if used[j]:
                    continue
                if min(np.linalg.norm(P[j] - P[k]) for k in g) < link:
                    g.append(j)
                    used[j] = True
                    grew = True
        out.append(g)
    return sorted(out, key=len, reverse=True)


def main():
    d = json.load(open(DUMP))
    V = d["drift_speed"]
    c = [z for z in d["clusters"] if z["uid"] == 4000097][0]
    x, y, z = (np.array(c[k]) for k in "xyz")
    tail = x < TAIL_X
    P = np.vstack([x[tail], y[tail], z[tail]]).T
    cl = clumps_of(P)

    cases = []
    for n in (0, 1, 3):                      # clumps 1, 2, 4
        g = cl[n]
        cen = P[g].mean(0)
        cases.append(("TAIL clump %d (%d pts)" % (n + 1, len(g)), cen[0], cen[1], cen[2]))
    i = np.argmin(x[~tail])                  # cluster 97's own deepest real point
    cases.append(("BODY end (real track)", x[~tail][i], y[~tail][i], z[~tail][i]))

    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 3.3 * len(cases)))
    print("%-24s %-6s %-8s %8s %8s" % ("case", "plane", "chan", "rawADC", "decon"))
    for r, (lbl, xx, yy, zz) in enumerate(cases):
        a, f = which_af(yy, zz)
        ti = (xx - geom.wplane_x_cm(STORE, a, f)) * (-1) / V / TICK_US
        for cc, (p, nm) in enumerate(((0, "U"), (1, "V"), (2, "W"))):
            ax = axes[r][cc]
            ch = wire_of(a, f, p, yy, zz)
            if ch is None:
                ax.set_title("%s - %s: no wire" % (lbl, nm), fontsize=9)
                continue
            lo, hi = int(ti - 120), int(ti + 120)
            t = np.arange(lo, hi)
            (Fo, Ro), (Fg, Rg) = frames(a)["o"], frames(a)["g"]
            ax2 = ax.twinx()
            pk_o = pk_g = float("nan")
            if ch in Ro:
                w = Fo[Ro[ch]][lo:hi]
                pk_o = float(w.max())
                ax.plot(t, w, color="0.45", lw=0.9, label="raw ADC (orig, pre-NF)")
                ax.set_ylim(-80, max(120, pk_o * 1.2))
            if ch in Rg:
                w = Fg[Rg[ch]][lo:hi]
                pk_g = float(w.max())
                ax2.plot(t, w, color="tab:red", lw=1.4, label="deconvolved (gauss)")
                ax2.set_ylim(-800, max(2000, pk_g * 1.2))
            ax.axvline(ti, color="tab:blue", ls="--", lw=1.5)
            ax.set_xlabel("tick")
            ax.set_ylabel("raw ADC", color="0.35")
            ax2.set_ylabel("decon", color="tab:red")
            ax.set_title("%s - %s ch %d (pred tick %.0f)" % (lbl, nm, ch, ti), fontsize=9)
            ax.grid(alpha=0.25)
            if r == 0 and cc == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
            print("%-24s %-6s %-8d %8.1f %8.1f" % (lbl, nm, ch, pk_o, pk_g))
    fig.suptitle("PDVD evt298567: raw + deconvolved U/V/W at cluster 97's tail clumps vs "
                 "the real body end\n(blue dashed = tick predicted from the 3-D point; "
                 "U/V bipolar = induction, W unipolar = collection)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PICS, "clus97_tail_uvw_waveforms.png")
    fig.savefig(out, dpi=105)
    print("\nwrote", out)


if __name__ == "__main__":
    main()

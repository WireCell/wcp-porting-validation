#!/usr/bin/env python3
"""Waveform check of a census tail case: is the past-cathode-face charge real
and continuous with the track's own pulse (evt298567 &10.2 signature)?

Usage: tail_waveform_check.py RUN IDX UID GID OUT.png
Adapted from docs/qlmatch/scripts/check_clus97_tail_waveforms.py (loaders,
wire_of brute force; gotchas M8 + tick-mislabel + angled-wire lookup apply).
Extended to the BOTTOM volume (anodes 0-3, opposite drift sign).
"""
import io
import json
import os
import glob
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cathode_tail_census as C

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
WIRES = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"
TICK_US = 0.5

run, idx, uid, gid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out = sys.argv[5]

STORE = geom.load_store(WIRES)
_PG = {}


def pg(a, f, p):
    if (a, f, p) not in _PG:
        _PG[(a, f, p)] = geom.PlaneGeom(STORE, a, f, p)
    return _PG[(a, f, p)]


def wire_of(a, f, p, y, z):
    g = pg(a, f, p)
    for k in range(len(g.chans)):
        if geom.point_in_poly(y, z, geom.order_polygon(g.band_quad(k))):
            return g.channel(k)
    return None


def which_af(anodes, y, z):
    for a in anodes:
        for f in (0, 1):
            g = pg(a, f, 2)
            ys = np.concatenate([g.tails[:, 0], g.heads[:, 0]])
            zs = np.concatenate([g.tails[:, 1], g.heads[:, 1]])
            if ys.min() - .3 <= y <= ys.max() + .3 and zs.min() - .3 <= z <= zs.max() + .3:
                return a, f
    return None, None


def load_tar(path, prefixes):
    outd = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            for k in prefixes:
                if m.name.startswith(k + "_"):
                    outd[k] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return outd


def frames(a):
    orig = f"{BASE}/input_data/run{run}/evt_{idx}/protodune-orig-frames-anode{a}.tar.bz2"
    spdn = f"{BASE}/work/{run}_{idx}_keep/protodune-sp-dnnroi-frames-anode{a}.tar.bz2"
    o = load_tar(orig, ["frame_orig", "channels_orig"])
    s = load_tar(spdn, [f"frame_gauss{a}", f"channels_gauss{a}"])
    Fo = o["frame_orig"] - np.median(o["frame_orig"], axis=1, keepdims=True)
    return (Fo, {c: i for i, c in enumerate(o["channels_orig"])}), \
           (s[f"frame_gauss{a}"], {c: i for i, c in enumerate(s[f"channels_gauss{a}"])})


dump = glob.glob(f"{C.BASE}/{run}_{idx}_keep/calib-evt*.json")[0]
evt = dump.split("calib-evt")[1][:-5]
d = json.load(open(dump))
V = d["drift_speed"]
top = uid >= 4000000
g = d["geometry"]["4" if top else "0"]
f = {x["gid"]: x for x in d["flashes"]}[gid]
t_fl = f["time1"] if top else f["time"]
ci = C.cluster_info({c["uid"]: c for c in d["clusters"]}[uid], g)
X = ci["x_raw"] + g["sign_offset"] * t_fl * V
face, sgn = g["cathode_x"], (1 if top else -1)
past = ci["tube"] & (sgn * (X - face) < 0)
print(f"evt{evt} uid{uid} flash {gid} ({f['total_PE']:.0f} PE): {past.sum()} pts past face, "
      f"pen {np.max(sgn*(face-X[ci['tube']])):.2f} cm")

# representative points: deepest past-face, mid past-face, body reference near face
ip = np.where(past)[0]
iord = ip[np.argsort(sgn * (face - X[ip]))]
deep = iord[-1]
mid = iord[len(iord) // 2]
body = ci["tube"] & ~past
ib = np.where(body)[0]
iref = ib[np.argmin(np.abs(X[ib] - face))]
cases = [("body @ face", iref), ("past-face mid", mid), ("past-face deepest", deep)]

anodes = (4, 5, 6, 7) if top else (0, 1, 2, 3)
sgn_t = -1 if top else 1
fig, axes = plt.subplots(len(cases), 3, figsize=(18, 3.4 * len(cases)))
FR = {}
for r, (lbl, i) in enumerate(cases):
    xx, yy, zz = ci["x_raw"][i], ci["y"][i], ci["z"][i]
    a, fc = which_af(anodes, yy, zz)
    if a is None:
        print(f"{lbl}: no (anode,face) for y={yy:.1f} z={zz:.1f}")
        continue
    if a not in FR:
        FR[a] = frames(a)
    (Fo, Ro), (Fg, Rg) = FR[a]
    wx = geom.wplane_x_cm(STORE, a, fc)
    ti = (xx - wx) * sgn_t / V / TICK_US
    # window: include the tick of the body reference to show continuity
    ti_ref = (ci["x_raw"][iref] - wx) * sgn_t / V / TICK_US
    nt = FR[a][0][0].shape[1]
    lo = max(0, int(min(ti, ti_ref)) - 100)
    hi = min(nt, int(max(ti, ti_ref)) + 100)
    for cc, (p, nm) in enumerate(((0, "U"), (1, "V"), (2, "W"))):
        ax = axes[r][cc]
        ch = wire_of(a, fc, p, yy, zz)
        if ch is None:
            ax.set_title(f"{lbl} - {nm}: no wire", fontsize=9)
            continue
        t = np.arange(lo, hi)
        ax2 = ax.twinx()
        pk_o = pk_g = float("nan")
        if ch in Ro:
            w = Fo[Ro[ch]][lo:hi]
            pk_o = float(w.max())
            ax.plot(t, w, color="0.45", lw=0.8)
        if ch in Rg:
            w = Fg[Rg[ch]][lo:hi]
            pk_g = float(w.max())
            ax2.plot(t, w, color="tab:red", lw=1.2)
        ax.axvline(ti, color="tab:blue", ls="--", lw=1.4)
        ax.axvline(ti_ref, color="tab:green", ls=":", lw=1.2)
        ax.set_title(f"{lbl} - {nm} ch {ch} a{a}f{fc} (pred tick {ti:.0f}, x={xx:.1f})",
                     fontsize=9)
        ax.set_xlabel("tick")
        ax.grid(alpha=0.25)
        print(f"{lbl:22s} {nm} ch {ch:5d} a{a}f{fc} tick {ti:7.0f} rawpk {pk_o:7.1f} deconpk {pk_g:8.1f}")
fig.suptitle(f"PDVD {run} evt{evt} uid {uid} @ flash {gid}: waveforms at past-cathode-face "
             "points vs the body end\n(blue dashed = point's predicted tick; green dotted = "
             "body-at-face tick; raw ADC grey, decon red)", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)

#!/usr/bin/env python3
"""W-streak endpoint reads for the two anode<->cathode(<->anode) crossers of
run 039252 evt 298609 (art evt_3), ccprod tag.  Companion to
06_pdvd-drift-crosser-298651.md (which did evt 298651) and to the light-side
closure check aca_light_check.py / 23_pdvd-light-timing-check.md.

Tracks (Bee img-global ccprod numbering -> calib (apa,ident)):
  track A (flash #117, 1019.45 us folded, 22117 PE):
    bee 37 = (0,5)   bottom half, full drift  (anode -> cathode)
    bee 79 = (4,3)   top half,    full drift
  track B (flash #163, 2899.69 us folded, 24384 PE -- matcher left it
    UNASSIGNED, op_cluster_ids=[]):
    bee 83 = (4,21)  top half,    cathode end TRUNCATED at readout edge
    bee 50 = (0,102) bottom half, cathode end TRUNCATED at readout edge

Method identical to crosser_drifttime_298651.py: the deconvolved W
(collection) trace records the charge-arrival tick independent of the
processing drift speed; imaging points only build a (y,z)->W-channel corridor
that rejects other cosmics.  Generalized here to halves spanning SEVERAL
anodes (per-(anode) bands combined on the common crate tick axis) and to
magnify ROOT input (hw_gauss<N>, all channels) instead of SP-frame tarballs.

Repro:
    cd pdvd/docs/qlmatch
    OMP_NUM_THREADS=4 python3 scripts/aca_crossers_298609.py
    # writes pics/track_298609_{A,B}_xyz.png, pics/wdecon_298609_<half>.png,
    #        pics/driftprofile_298609_{A,B}.png, scripts/aca_298609_endpoints.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
EV = "298609"
CALIB = os.path.join(WORK, "039252_3_ccprod", "calib-evt%s.json" % EV)
MAGDIR = os.path.join(WORK, "039252_3_magnify")
STORE = geom.load_store("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "pics")
TICK_US = 0.5
WIN = 60          # +-tick corridor window around each imaging-predicted tick
THR = 400.0       # gauss charge threshold for "on the streak"
GAP = 700.0       # channel isolated by >GAP ticks from the band = other cosmic

# (label, apa_key, ident, side, track)
HALVES = [
    ("A_bot37", 0,   5, "bot", "A"),
    ("A_top79", 4,   3, "top", "A"),
    ("B_top83", 4,  21, "top", "B"),
    ("B_bot50", 0, 102, "bot", "B"),
]

PG = {}
def pg(a, f):
    if (a, f) not in PG:
        PG[(a, f)] = geom.PlaneGeom(STORE, a, f, 2)      # plane 2 = W
    return PG[(a, f)]

def which_af(y, z, side):
    for a in ((0, 1, 2, 3) if side == "bot" else (4, 5, 6, 7)):
        for f in (0, 1):
            g = pg(a, f)
            ys = np.concatenate([g.tails[:, 0], g.heads[:, 0]])
            zs = np.concatenate([g.tails[:, 1], g.heads[:, 1]])
            if ys.min()-0.3 <= y <= ys.max()+0.3 and zs.min()-0.3 <= z <= zs.max()+0.3:
                return a, f
    return None, None

def chan_tick(x, y, z, side, V):
    a, f = which_af(y, z, side)
    if a is None:
        return None
    g = pg(a, f)
    centers = 0.5 * (g.tails + g.heads)
    k = int(round((z - centers[0, 1]) / g.pitch_cm))
    if not (0 <= k < len(g.chans)):
        return None
    dirx = 1 if side == "bot" else -1
    ti = (x - geom.wplane_x_cm(STORE, a, f)) * dirx / V / TICK_US
    return a, int(g.chans[k]), ti

_mag = {}
def magnify(anode):
    """(values (nch, 10000), chan ids (nch,)) from hw_gauss<anode>."""
    if anode not in _mag:
        f = uproot.open(os.path.join(MAGDIR,
            "magnify-run039252-evt%d-anode%d-dnnroi_magnify.root" % (3, anode)))
        h = f["hw_gauss%d" % anode]
        _mag[anode] = (h.values(), np.round(h.axis(0).centers()).astype(int))
    return _mag[anode]

d = json.load(open(CALIB))
V = d["drift_speed"]
TRIG = d.get("trigger_offsets_us", [0.0, 0.0])
CL = {(c["apa"], c["ident"]): c for c in d["clusters"]}
print("calib %s  drift_speed=%.6f cm/us  trigger_offsets_us=%s" % (CALIB, V, TRIG))

def cluster(apa, ident):
    c = CL[(apa, ident)]
    return (np.asarray(c["x"], float), np.asarray(c["y"], float),
            np.asarray(c["z"], float))

def endpoints(apa, ident, side):
    """Corridor endpoint read; per-anode bands combined on the crate tick axis."""
    x, y, z = cluster(apa, ident)
    pred = {}                                  # (anode, chan) -> [imaging ticks]
    for xi, yi, zi in zip(x, y, z):
        r = chan_tick(xi, yi, zi, side, V)
        if r:
            pred.setdefault((r[0], r[1]), []).append(r[2])
    # band isolation per anode: order that anode's channels by median tick and
    # drop channels isolated by >GAP (junk satellite points / other cosmics)
    band = []
    for a in sorted(set(k[0] for k in pred)):
        chans = sorted((k for k in pred if k[0] == a), key=lambda k: np.median(pred[k]))
        med = np.array([np.median(pred[k]) for k in chans])
        for i, k in enumerate(chans):
            gl = med[i]-med[i-1] if i > 0 else 0
            gr = med[i+1]-med[i] if i < len(chans)-1 else 0
            iso = (i == 0 and gr > GAP) or (i == len(chans)-1 and gl > GAP) or (gl > GAP and gr > GAP)
            if not iso:
                band.append(k)
    prof = np.zeros(10000)
    nch_used = 0
    for (a, ch) in band:
        F, C = magnify(a)
        rr = np.where(C == ch)[0]
        if not len(rr):
            continue
        row = F[rr[0]]
        mask = np.zeros(10000, bool)
        for t in pred[(a, ch)]:
            lo = max(0, int(t)-WIN); hi = min(10000, int(t)+WIN)
            mask[lo:hi] = True
        seg = np.where(mask, row, 0.0)
        seg[seg < THR] = 0
        prof += seg
        nch_used += 1
    k = 51; ps = np.convolve(prof, np.ones(k)/k, mode="same")
    lvl = 0.5 * np.median(ps[ps > 0.2*ps.max()])
    ons = np.where(ps > lvl)[0]
    return dict(band=band, pred=pred, prof=prof, ps=ps,
                t_lo=float(ons.min()), t_hi=float(ons.max()),
                nch=nch_used, x=x, y=y, z=z,
                anodes=sorted(set(k[0] for k in band)))

R = {}
for lab, a, i, side, trk in HALVES:
    R[lab] = endpoints(a, i, side)
    R[lab].update(side=side, apa=a, ident=i, track=trk)
    r = R[lab]
    print("%-8s (apa%d id%3d %s) anodes=%s nch=%3d  corridor ticks [%6.1f, %6.1f]"
          "  = [%7.1f, %7.1f] us  span %7.1f us"
          % (lab, a, i, side, r["anodes"], r["nch"], r["t_lo"], r["t_hi"],
             r["t_lo"]*TICK_US, r["t_hi"]*TICK_US, (r["t_hi"]-r["t_lo"])*TICK_US))

# ---------- figures ----------
for trk, labs in (("A", ["A_bot37", "A_top79"]), ("B", ["B_top83", "B_bot50"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for lab, col in zip(labs, ("tab:blue", "tab:red")):
        r = R[lab]
        axs[0].plot(r["x"], r["y"], ".", ms=1.5, color=col, label=lab)
        axs[1].plot(r["z"], r["y"], ".", ms=1.5, color=col)
        axs[2].plot(r["x"], r["z"], ".", ms=1.5, color=col)
    axs[0].set_xlabel("x app (cm)"); axs[0].set_ylabel("y (cm)"); axs[0].legend()
    axs[1].set_xlabel("z (cm)"); axs[1].set_ylabel("y (cm)")
    axs[2].set_xlabel("x app (cm)"); axs[2].set_ylabel("z (cm)")
    fig.suptitle("evt 298609 track %s (ccprod, apparent coords)" % trk)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "track_298609_%s_xyz.png" % trk), dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, lab in zip(axs, labs):
        r = R[lab]
        ax.plot(np.arange(10000), r["ps"], "k-", lw=0.8)
        ax.axvline(r["t_lo"], color="c", ls="--", lw=1, label="t_lo %.0f" % r["t_lo"])
        ax.axvline(r["t_hi"], color="m", ls=":", lw=1, label="t_hi %.0f" % r["t_hi"])
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    axs[1].set_xlabel("tick (0.5 us)")
    fig.suptitle("evt 298609 track %s: W corridor charge profiles" % trk)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "driftprofile_298609_%s.png" % trk), dpi=110)
    plt.close(fig)

# W-streak displays: per half, all band channels stacked (chan vs tick scatter of
# gauss>THR within corridor), imaging preds overlaid
for lab in R:
    r = R[lab]
    fig, ax = plt.subplots(figsize=(13, 5))
    for (a, ch) in r["band"]:
        F, C = magnify(a)
        rr = np.where(C == ch)[0]
        if not len(rr):
            continue
        row = F[rr[0]]
        on = np.where(row > THR)[0]
        if len(on):
            ax.plot(on, np.full(len(on), ch), ".", ms=1, color="0.6")
        for t in r["pred"][(a, ch)]:
            ax.plot([t], [ch], "r.", ms=2)
    ax.axvline(r["t_lo"], color="c", ls="--", lw=1.2, label="t_lo %.0f" % r["t_lo"])
    ax.axvline(r["t_hi"], color="m", ls=":", lw=1.2, label="t_hi %.0f" % r["t_hi"])
    ax.set_xlabel("tick (0.5 us)"); ax.set_ylabel("W channel")
    ax.set_title("evt 298609 %s: W gauss>%.0f (grey), imaging corridor (red)" % (lab, THR))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "wdecon_298609_%s.png" % lab), dpi=110)
    plt.close(fig)

out = {lab: dict(track=R[lab]["track"], side=R[lab]["side"], apa=R[lab]["apa"],
                 ident=R[lab]["ident"], anodes=R[lab]["anodes"],
                 t_lo=R[lab]["t_lo"], t_hi=R[lab]["t_hi"])
       for lab in R}
out["_meta"] = dict(event=int(EV), calib=CALIB, drift_speed=V,
                    trigger_offsets_us=TRIG, tick_us=TICK_US)
with open(os.path.join(HERE, "aca_298609_endpoints.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote aca_298609_endpoints.json and pics/*298609*.png")

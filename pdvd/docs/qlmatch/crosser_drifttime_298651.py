#!/usr/bin/env python3
"""Drift-velocity cross-check from the anode<->cathode crossing cosmic in
run 039252 event 298651: flash gid=88 (t=364.7us, group=89) <- clusters
[bot:34, top:169].  See pdvd-drift-crosser-298651.md.

Idea (imaging-INDEPENDENT): the deconvolved W (collection) waveform records the
charge-arrival tick regardless of the processing drift speed (v enters only the
later tick->x conversion, never the waveform).  For each half of the crosser we
read the drift-time of the two track ends off the W streak and take the
difference dt = t_cathode - t_anode (a difference, so ctoffset and per-crate
trigger offsets cancel).  With the known full drift distance DFULL = 336.91 cm,
v = DFULL / dt.

Endpoints are read from the CHARGE, not from the 3-D imaging points: top:169's
3-D imaging is gappy at the anode (induction-plane SP), so its cluster loses the
anode-end charge -- but the W collection charge is there on wires the imaging
never assigned.  We recover it and show it.

Repro:
    cd pdvd/docs/qlmatch
    OMP_NUM_THREADS=4 python3 crosser_drifttime_298651.py
    # writes track_298651_xyz.png, wdecon_298651_bot34.png,
    #        wdecon_298651_top169.png, driftprofile_298651.png  and prints a table
"""
import json, os, io, tarfile, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
EV = "298651"
GID = 88
CALIB = os.path.join(WORK, "039252_6_v153", "calib-evt%s.json" % EV)
PDIR = os.path.join(WORK, "039252_6")                 # real SP-frame tarballs
STORE = geom.load_store("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
# Full drift traversed by charge collected at the W plane = cathode -> W collection
# plane.  u_cathode (336.91 cm) is cathode -> shield-plane anode edge (v6 shield FV);
# the W collection plane sits 1.64 cm deeper (W |x|=341.55 vs shield 339.91), and the
# gauss pulse we time is the arrival at W, so the relevant distance is 336.91+1.64.
DFULL = 338.55          # cm  (= 336.91 u_cathode + 1.64 shield->W)
TICK_US = 0.5
# endpoint-read parameters
WIN = 60                # +-tick window around each imaging-predicted tick (corridor)
THR = 400.0             # gauss charge threshold for "on the streak"
GAP = 700.0             # a channel isolated by >GAP ticks from the band is a crosser
AMP_STUB = 1500.0       # min gauss amplitude for anode-stub charge recovery
WID_STUB = 6            # min tick-width (>0.4 peak) for stub to count as track-like
TRACKS = [(0, 34, "bot"), (4, 169, "top")]

# ---------- geometry / frame helpers (reused from check_chain_consistency.py) ----------
PG = {}
def pg(a, f):
    if (a, f) not in PG:
        PG[(a, f)] = geom.PlaneGeom(STORE, a, f, 2)   # plane ident 2 = W (collection)
    return PG[(a, f)]

def which_af(y, z, side):
    """(anode,face) whose W-wire y-z bbox contains (y,z); bot=anodes0-3, top=4-7."""
    for a in ((0, 1, 2, 3) if side == "bot" else (4, 5, 6, 7)):
        for f in (0, 1):
            g = pg(a, f)
            ys = np.concatenate([g.tails[:, 0], g.heads[:, 0]])
            zs = np.concatenate([g.tails[:, 1], g.heads[:, 1]])
            if ys.min()-0.3 <= y <= ys.max()+0.3 and zs.min()-0.3 <= z <= zs.max()+0.3:
                return a, f
    return None, None

def chan_tick(x, y, z, side, V):
    """3-D point -> (anode, W-channel, imaging-predicted tick).  Used only to
    build the spatial corridor (reject other cosmics), NOT to define the tick
    the velocity is read from."""
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

def load_gauss(anode, ev):
    """(frame_gauss (nch,nt), channels_gauss (nch,)) for one anode."""
    F = C = None
    with tarfile.open(os.path.join(PDIR, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)) as tf:
        for m in tf.getmembers():
            if m.name == "frame_gauss%d_%s.npy" % (anode, ev):
                F = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif m.name == "channels_gauss%d_%s.npy" % (anode, ev):
                C = np.load(io.BytesIO(tf.extractfile(m).read()))
    return F, C

# ---------- load event ----------
d = json.load(open(CALIB))
V = d["drift_speed"]                                   # cm/us used by this processing
GEO = {int(k): v for k, v in d["geometry"].items()}
CL = {(c["apa"], c["ident"]): c for c in d["clusters"]}
TRIG = d.get("trigger_offsets_us", [0.0, 0.0])
_frames = {}
def gof(a):
    if a not in _frames:
        _frames[a] = load_gauss(a, EV)
    return _frames[a]

def cluster(apa, ident):
    c = CL[(apa, ident)]
    return (np.asarray(c["x"], float), np.asarray(c["y"], float),
            np.asarray(c["z"], float), np.asarray(c["q"], float))

# ---------- W-streak endpoint reader ----------
def endpoints(apa, ident, side):
    """Return dict with the anode/cathode drift-time ticks read from the W decon
    streak, plus intermediate products for plotting."""
    x, y, z, q = cluster(apa, ident)
    pred = {}                                          # W-chan -> list of imaging ticks
    for xi, yi, zi in zip(x, y, z):
        r = chan_tick(xi, yi, zi, side, V)
        if r:
            pred.setdefault(r[1], []).append(r[2])
    anode = which_af(y[0], z[0], side)[0]
    F, C = gof(anode)
    nt = F.shape[1]
    def row_of(ch):
        rr = np.where(C == ch)[0]
        return F[rr[0]] if len(rr) else None
    # reject channels isolated in the (median-tick) band ordering = crossing cosmics
    chans = sorted(pred, key=lambda ch: np.median(pred[ch]))
    med = np.array([np.median(pred[ch]) for ch in chans])
    band = []
    for i, ch in enumerate(chans):
        gl = med[i]-med[i-1] if i > 0 else 0
        gr = med[i+1]-med[i] if i < len(chans)-1 else 0
        iso = (i == 0 and gr > GAP) or (i == len(chans)-1 and gl > GAP) or (gl > GAP and gr > GAP)
        if not iso:
            band.append(ch)
    # charge-vs-tick profile over the band (union of +-WIN windows around imaging ticks)
    prof = np.zeros(nt)
    for ch in band:
        row = row_of(ch)
        if row is None:
            continue
        mask = np.zeros(nt, bool)
        for t in pred[ch]:
            a0 = max(0, int(t)-WIN); b0 = min(nt, int(t)+WIN)
            mask[a0:b0] = True
        seg = np.where(mask, row, 0.0)
        seg[seg < THR] = 0
        prof += seg
    on = np.where(prof > 0)[0]
    k = 51; ps = np.convolve(prof, np.ones(k)/k, mode="same")
    lvl = 0.5 * np.median(ps[ps > 0.2*ps.max()])       # 50% of typical band level
    ons = np.where(ps > lvl)[0]
    t_anode_corr = float(ons.min())                    # corridor anode edge
    t_cathode = float(ons.max())                       # corridor cathode edge (robust)
    # anode-stub recovery: broad W charge on wires adjacent to the band that
    # imaging never assigned, continuing the streak below t_anode_corr
    band_ids = [int(c) for c in band]
    wall = set()
    for f in (0, 1):
        wall |= set(int(c) for c in pg(anode, f).chans)
    stub = []
    for ch in sorted(wall):
        if ch in band or not (min(band_ids)-3 <= ch <= max(band_ids)+3):
            continue
        if not any(abs(ch-b) <= 2 for b in band_ids):
            continue
        row = row_of(ch)
        if row is None:
            continue
        seg = row[:int(t_anode_corr)]
        if len(seg) == 0 or seg.max() < AMP_STUB:
            continue
        j = int(seg.argmax())
        if (row[max(0, j-30):j+30] > 0.4*row[j]).sum() < WID_STUB:
            continue
        stub.append((ch, j, float(row[j])))
    t_anode = min([t_anode_corr] + [s[1] for s in stub])
    return dict(anode=anode, band=band, pred=pred, prof=prof, F=F, C=C,
                t_anode_corr=t_anode_corr, t_anode=t_anode, t_cathode=t_cathode,
                stub=stub, x=x, y=y, z=z)

# ---------- run both halves ----------
R = {s: endpoints(a, i, s) for (a, i, s) in TRACKS}

# Owner Magnify hand-scan endpoints (deconvolved W, hw_gauss1/hw_gauss5), adopted
# as the authoritative reads.  These supersede the automated corridor reads (kept
# as 't_anode_auto'/'t_cathode_auto' for provenance).  ticks, 0.5 us each.
HANDSCAN = {"bot": (576.0, 5166.0), "top": (590.0, 5164.0)}
for side, (ta, tc) in HANDSCAN.items():
    R[side]["t_anode_auto"] = R[side]["t_anode"]
    R[side]["t_cathode_auto"] = R[side]["t_cathode"]
    R[side]["t_anode"] = ta
    R[side]["t_cathode"] = tc

def vel(t_lo, t_hi):
    return DFULL / ((t_hi - t_lo) * TICK_US)

print("\n============ evt %s  flash gid=%d  A<->C crosser  drift-time cross-check ============" % (EV, GID))
print("processing drift_speed = %.4f cm/us (%.3f mm/us); DFULL = %.2f cm; tick = %.1f us" %
      (V, V*10, DFULL, TICK_US))
print("trigger_offsets_us = %s  (crate difference %.1f us = %.1f ticks)\n" %
      (TRIG, TRIG[1]-TRIG[0], (TRIG[1]-TRIG[0])/TICK_US))
print("Endpoints = owner Magnify hand-scan of the deconvolved W (hw_gauss1/hw_gauss5).\n")
print("%-8s %5s | %8s %10s | %8s %10s %9s | %s" %
      ("half", "anode", "t_anode", "t_cathode", "dt(us)", "v(cm/us)", "v(mm/us)", "auto-read (provenance)"))
print("-"*104)
for (apa, ident, side) in TRACKS:
    r = R[side]
    dt = (r["t_cathode"]-r["t_anode"])*TICK_US
    v_i = vel(r["t_anode"], r["t_cathode"])
    print("%-8s %5d | %8.0f %10.0f | %8.1f %10.5f %9.3f | anode %.0f cathode %.0f" %
          ("%s:%d" % (side, ident), r["anode"], r["t_anode"], r["t_cathode"], dt, v_i, v_i*10,
           r["t_anode_auto"], r["t_cathode_auto"]))
print("-"*104)

rb, rt = R["bot"], R["top"]
print("\nvalidity checks:")
print("  cathode-end coincidence : bot %.0f  top %.0f  -> %.0f ticks (%.1f us)   [same physical cathode]" %
      (rb["t_cathode"], rt["t_cathode"], rb["t_cathode"]-rt["t_cathode"], (rb["t_cathode"]-rt["t_cathode"])*TICK_US))
print("  anode-end coincidence   : bot %.0f  top %.0f  -> %.0f ticks (%.1f us)   [expect ~ crate trigger diff %.1f ticks]" %
      (rb["t_anode"], rt["t_anode"], rb["t_anode"]-rt["t_anode"], (rb["t_anode"]-rt["t_anode"])*TICK_US, (TRIG[1]-TRIG[0])/TICK_US))
vb = vel(rb["t_anode"], rb["t_cathode"]); vt = vel(rt["t_anode"], rt["t_cathode"])
vmean = 0.5*(vb+vt)
print("\n  v(bot:34)  = %.5f cm/us (%.3f mm/us)" % (vb, vb*10))
print("  v(top:169) = %.5f cm/us (%.3f mm/us)" % (vt, vt*10))
print("  --------------------------------------------------")
print("  v = %.5f cm/us (%.3f mm/us)  (mean; half-spread %.5f cm/us)   vs config 0.153 / toolkit 0.1568 / conv 0.1586 cm/us" %
      (vmean, vmean*10, abs(vb-vt)/2))

# ================================ PLOTS ================================
# ---- 1) X-Y / Y-Z / X-Z projections of the two clusters (render_groups house style) ----
def draw_projection_panels():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cols = {"bot": "tab:blue", "top": "tab:red"}
    for (apa, ident, side) in TRACKS:
        x, y, z, q = cluster(apa, ident)
        n = len(x)
        if n > 3000:
            s = np.linspace(0, n-1, 3000).astype(int); x, y, z = x[s], y[s], z[s]
        kw = dict(s=3, color=cols[side], alpha=0.7, lw=0, label="%s:%d (%d pts)" % (side, ident, n))
        axs[0].scatter(z, x, **kw); axs[1].scatter(y, x, **kw); axs[2].scatter(z, y, **kw)
    for a in (0, 4):
        g = GEO[a]
        for ax, h in ((axs[0], ("z_lo", "z_hi")), (axs[1], ("y_lo", "y_hi"))):
            ax.plot([g[h[0]], g[h[1]], g[h[1]], g[h[0]], g[h[0]]],
                    [g["anode_x"], g["anode_x"], g["cathode_x"], g["cathode_x"], g["anode_x"]],
                    color="0.4", lw=0.7, alpha=0.7)
    g0, g4 = GEO[0], GEO[4]
    axs[2].plot([g0["z_lo"], g0["z_hi"], g0["z_hi"], g0["z_lo"], g0["z_lo"]],
                [g0["y_lo"], g0["y_lo"], g0["y_hi"], g0["y_hi"], g0["y_lo"]], color="0.4", lw=0.7, alpha=0.7)
    for ax, (xl, yl, tt) in zip(axs, (("z (cm)", "x (cm) drift", "X-Z"),
                                      ("y (cm)", "x (cm) drift", "X-Y"),
                                      ("z (cm)", "y (cm)", "Y-Z"))):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(tt); ax.legend(fontsize=8)
    fig.suptitle("evt %s  flash gid=%d  anode<->cathode crosser  bot:34 + top:169 (imaging 3-D points)" % (EV, GID))
    fig.tight_layout()
    p = os.path.join(OUTDIR, "track_298651_xyz.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print("  wrote", os.path.basename(p))

# ---- 2/3) per-half annotated W-decon heatmap (band channels + imaging + termini + stub) ----
def draw_wdecon(side, ident):
    r = R[side]; F, C = r["F"], r["C"]; band = r["band"]
    ids = sorted(int(c) for c in band) + [s[0] for s in r["stub"]]
    lo_id, hi_id = min(ids)-4, max(ids)+4
    wall = set()
    for f in (0, 1): wall |= set(int(c) for c in pg(r["anode"], f).chans)
    winchans = sorted(ch for ch in wall if lo_id <= ch <= hi_id)
    c2i = {ch: k for k, ch in enumerate(winchans)}
    img = np.zeros((len(winchans), F.shape[1]))
    for ch in winchans:
        rr = np.where(C == ch)[0]
        if len(rr): img[c2i[ch]] = F[rr[0]]
    px, py = [], []
    for ch, tks in r["pred"].items():
        if ch in c2i:
            for t in tks: px.append(t); py.append(c2i[ch]+0.5)
    fig, axs = plt.subplots(2, 1, figsize=(13, 8))
    vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
    for ax, tl, ttl in ((axs[0], (0, F.shape[1]), "full readout"),
                        (axs[1], (max(0, int(r["t_anode"])-350), int(r["t_anode"])+600), "anode-end zoom")):
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, F.shape[1], 0, len(winchans)], vmax=vmax, vmin=0)
        ax.plot(px, py, "r.", ms=1.5, alpha=0.45, label="imaging-assigned (corridor)")
        for s in r["stub"]:
            ax.plot(s[1], c2i[s[0]]+0.5, "m*", ms=13,
                    label="imaging-MISSED W charge (anode)" if s is r["stub"][0] else None)
        ax.axvline(r["t_anode"], color="cyan", ls="--", lw=1.6, label="t_anode=%.0f (hand-scan)" % r["t_anode"])
        ax.axvline(r["t_cathode"], color="magenta", ls="--", lw=1.6, label="t_cathode=%.0f (hand-scan)" % r["t_cathode"])
        ax.axvline(r["t_anode_corr"], color="0.7", ls=":", lw=1.2, label="auto anode read=%.0f" % r["t_anode_corr"])
        ax.set_xlim(*tl); ax.set_ylabel("W channel idx"); ax.set_title("%s:%d  anode%d  %s" % (side, ident, r["anode"], ttl))
        ax.legend(fontsize=7, loc="upper right")
    dt = (r["t_cathode"]-r["t_anode"])*TICK_US
    axs[1].set_xlabel("tick (0.5 us)")
    fig.suptitle("%s:%d W deconvolved streak   dt = %.0f ticks = %.0f us   v = %.3f mm/us" %
                 (side, ident, r["t_cathode"]-r["t_anode"], dt, vel(r["t_anode"], r["t_cathode"])*10))
    fig.tight_layout()
    p = os.path.join(OUTDIR, "wdecon_298651_%s%d.png" % (side, ident)); fig.savefig(p, dpi=110); plt.close(fig)
    print("  wrote", os.path.basename(p))

# ---- 4) charge-vs-tick profiles overlaid, termini marked ----
def draw_profiles():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    cols = {"bot": "tab:blue", "top": "tab:red"}
    for side in ("bot", "top"):
        r = R[side]; p = r["prof"]; on = np.where(p > 0)[0]
        ax.plot(np.arange(len(p)), p/ p.max(), color=cols[side], lw=0.9,
                label="%s:%d  dt=%.0f ticks  v=%.3f mm/us" %
                (side, dict(bot=34, top=169)[side], r["t_cathode"]-r["t_anode"], vel(r["t_anode"], r["t_cathode"])*10))
        ax.axvline(r["t_anode"], color=cols[side], ls="--", lw=1.2)
        ax.axvline(r["t_cathode"], color=cols[side], ls=":", lw=1.2)
    ax.set_xlabel("tick (0.5 us)"); ax.set_ylabel("W corridor charge (norm)")
    ax.set_title("evt %s  drift-time profiles: dashed=t_anode, dotted=t_cathode (cathode ends coincide)" % EV)
    ax.legend(fontsize=9); ax.set_xlim(0, 6000)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "driftprofile_298651.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print("  wrote", os.path.basename(p))

print("\nplots:")
draw_projection_panels()
draw_wdecon("bot", 34)
draw_wdecon("top", 169)
draw_profiles()
print("\ndone.")

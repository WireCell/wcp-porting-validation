#!/usr/bin/env python
"""
PDVD U/V -> W wire-offset calibration from real tracks (run 39324).

For an isolated track, at each drift tick the track sits at one (y,z) point.
The charge-weighted U and V wire lines must cross at that point, and the W wire
through the crossing must be the hit W wire.  A constant channel<->wire ordering
error in the (mirror-symmetric) induction planes shows up as a single offset
delta applied EQUALLY to U and V, producing a constant shift of the predicted W
relative to the measured W (along the W pitch).

This script measures, per anode and per geometry (v3, v4):
  - residual r(t) = W_measured(t) - W_predicted-from-U,V(t)   [W channels]
  - the symmetric U/V channel offset delta that minimises RMS(r)

Outputs plots + a printed summary to /home/xqian/tmp/.
Read-only on all inputs; no geometry file is modified.
"""
import os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wirecell.util.wires import persist

OUT = "/home/xqian/tmp"
WIREDIR = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data"
GEOMS = {
    "v3": f"{WIREDIR}/protodunevd-wires-larsoft-v3.json.bz2",
    "v4": f"{WIREDIR}/protodunevd-wires-larsoft-v4.json.bz2",
}
MAGBASE = ("/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/"
           "039324_{E}/magnify-run039324-evt{E}-anode{A}-dnnroi.root")

W_PITCH_MM = 5.10
UV_PITCH_MM = 7.65

# Per-anode track definition (run 39324, event 0).
CONFIGS = {
    0: dict(event=0, anode=0,
            win=dict(U=(75, 125), V=(1010, 1090), W=(2020, 2070)),
            tickwin=(0, 5000), label="anode 0 (bottom CRP)"),
    4: dict(event=0, anode=4,
            win=dict(U=(6530, 6610), V=(7460, 7550), W=(8480, 8630)),
            tickwin=(0, 4500), label="anode 4 (top CRP)"),
}


# ---------------------------------------------------------------- geometry ----
def wire_mid_dir(store, w):
    t = store.points[w.tail]; h = store.points[w.head]
    mid = np.array([(t.y + h.y) / 2.0, (t.z + h.z) / 2.0])
    d = np.array([h.y - t.y, h.z - t.z]); d = d / np.linalg.norm(d)
    return mid, d


def plane_table(store, plane_idx):
    """Return per-wire arrays sorted by channel: chan, pitch-coord, pitch-dir, wire-dir.
    pitch coordinate = midpoint projected on the in-plane pitch direction (perp to wire)."""
    pl = store.planes[plane_idx]
    ws = [store.wires[i] for i in pl.wires]
    mids = []; dirs = []; chans = []
    for w in ws:
        mid, d = wire_mid_dir(store, w)
        mids.append(mid); dirs.append(d); chans.append(w.channel)
    mids = np.array(mids); dirs = np.array(dirs); chans = np.array(chans)
    wdir = dirs.mean(0); wdir /= np.linalg.norm(wdir)
    pdir = np.array([wdir[1], -wdir[0]])          # perpendicular, in YZ
    pitch = mids @ pdir
    order = np.argsort(chans)
    # zmid: z-position of each wire's midpoint (well-defined / constant for the
    # vertical W collection wires; used to measure the W-plane position).
    return dict(chan=chans[order], pitch=pitch[order], pdir=pdir, wdir=wdir,
                zmid=mids[order, 1], ymid=mids[order, 0])


def find_face_planes(store, anode_ident, w_window):
    """Pick the face of the anode whose W (collection) plane contains the W window;
    return the three plane tables (U, V, W) for that face."""
    a = [an for an in store.anodes if an.ident == anode_ident][0]
    for fi in a.faces:
        face = store.faces[fi]
        planes = [store.planes[pi] for pi in face.planes]
        # plane ident 2 == collection (W)
        wpl_idx = [pi for pi in face.planes if store.planes[pi].ident == 2][0]
        wchans = [store.wires[i].channel for i in store.planes[wpl_idx].wires]
        if min(wchans) <= w_window[0] and w_window[1] <= max(wchans):
            tabs = {}
            for pi in face.planes:
                ident = store.planes[pi].ident          # 0=U,1=V,2=W
                tabs[{0: "U", 1: "V", 2: "W"}[ident]] = plane_table(store, pi)
            return fi, tabs
    raise RuntimeError(f"no face of anode {anode_ident} contains W window {w_window}")


def chan_to_pitch(tab, chan):
    """Real-valued channel -> pitch coordinate (linear interp; mapping is affine & 1:1)."""
    return np.interp(chan, tab["chan"], tab["pitch"])


def w_chan_to_z(tab, chan):
    """W collection channel -> wire z position (vertical wires, constant z)."""
    return np.interp(chan, tab["chan"], tab["zmid"])


def pitch_to_chan(tab, pitch):
    p = tab["pitch"]; c = tab["chan"]
    if p[0] > p[-1]:
        p = p[::-1]; c = c[::-1]
    return np.interp(pitch, p, c)


def predict_w(tabs, u_chan, v_chan, dp=0.0):
    """Given measured U & V centroid channels, intersect the two wire lines and
    return the predicted (real-valued) W channel.

    `dp` is a SYMMETRIC pitch offset (mm) added equally to the U and V pitch
    coordinates.  Because the induction planes are mirror-symmetric (pdir_U and
    pdir_V share the same component along the wire, opposite along W pitch), this
    equal-pitch shift moves the U/V crossing PURELY along the W pitch -- i.e. it
    is exactly "the same offset for the U and V strips" producing "a constant
    shift along W pitch".  (In channel units it is opposite-signed on U vs V,
    since their channel numbering runs in opposite pitch senses.)"""
    pU = chan_to_pitch(tabs["U"], u_chan) + dp
    pV = chan_to_pitch(tabs["V"], v_chan) + dp
    A = np.vstack([tabs["U"]["pdir"], tabs["V"]["pdir"]])   # 2x2
    yz = np.linalg.solve(A, np.array([pU, pV]))             # (y,z) crossing
    pW = tabs["W"]["pdir"] @ yz
    return pitch_to_chan(tabs["W"], pW), yz


# ----------------------------------------------------------------- signals ----
def load_hist(fn, name):
    h = uproot.open(fn)[name]
    v = h.values()                       # (nchan, ntick)
    xe = h.axis(0).edges()
    chans = (xe[:-1] + xe[1:]) / 2.0
    return v, chans


def centroids(fn, suffix, cfg):
    """Per-tick charge-weighted centroid channel for U,V,W within the windows.
    Returns ticks plus u,v,w arrays and the per-plane spreads; masked to ticks
    where all three planes have charge above threshold and are reasonably narrow."""
    t0, t1 = cfg["tickwin"]
    res = {}
    for pl in ("U", "V", "W"):
        v, chans = load_hist(fn, f"h{pl.lower()}_gauss{suffix}")
        c0, c1 = cfg["win"][pl]
        cm = (chans >= c0) & (chans <= c1)
        sub = np.clip(v[cm, t0:t1], 0, None)          # (nc, nt) charge
        cc = chans[cm]
        q = sub.sum(0)                                # per-tick total charge
        cen = np.where(q > 0, (sub * cc[:, None]).sum(0) / np.where(q > 0, q, 1), np.nan)
        var = np.where(q > 0,
                       (sub * (cc[:, None] - cen[None, :]) ** 2).sum(0) / np.where(q > 0, q, 1),
                       np.nan)
        res[pl] = dict(cen=cen, q=q, spread=np.sqrt(var))
    ticks = np.arange(t0, t1)
    return ticks, res


def good_mask(res, qmin_frac=0.05, spread_max=4.0):
    """Ticks where all three planes have charge above a fraction of their peak and
    a narrow (single-track) charge profile."""
    m = np.ones_like(res["U"]["cen"], bool)
    for pl in ("U", "V", "W"):
        q = res[pl]["q"]; sp = res[pl]["spread"]
        thr = qmin_frac * np.nanmax(q)
        m &= (q > thr) & np.isfinite(res[pl]["cen"]) & (sp < spread_max)
    return m


# -------------------------------------------------------------- per-anode  ----
def analyse_anode(akey):
    cfg = CONFIGS[akey]
    fn = MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    suffix = cfg["anode"]
    ticks, res = centroids(fn, suffix, cfg)
    gm = good_mask(res)
    print(f"\n==== {cfg['label']}  (evt{cfg['event']}, {fn.split('/')[-1]}) ====")
    print(f"  usable ticks: {gm.sum()} / {len(ticks)}")

    uc = res["U"]["cen"][gm]; vc = res["V"]["cen"][gm]; wc = res["W"]["cen"][gm]
    tt = ticks[gm]

    half = W_PITCH_MM / 2.0
    geoms = {}
    for gname, gpath in GEOMS.items():
        store = persist.load(gpath)
        fi, tabs = find_face_planes(store, cfg["anode"], cfg["win"]["W"])

        # ---- primary measure: keep U,V geometry fixed; per tick compare the
        #      z of the U/∩V crossing to the z of the measured W wire.
        #      dz_W(t) = z(U∩V) - z(W wire)  is the W-plane shift that would match.
        zc = np.array([predict_w(tabs, u, v)[1][1] for u, v in zip(uc, vc)])  # crossing z
        zw = w_chan_to_z(tabs["W"], wc)                                       # measured W z
        dzW = zc - zw                                                         # mm
        med_dzW = float(np.median(dzW)); rms_dzW = float(np.std(dzW))
        # consistency to +-half pitch (the imaging tolerance)
        frac_before = float(np.mean(np.abs(dzW) < half))
        frac_after = float(np.mean(np.abs(dzW - med_dzW) < half))

        # ---- equivalent (secondary) framing: residual in W-wire units, and the
        #      symmetric U/V strip offset that nulls it (crossing moves -2*dp in z).
        r0 = (wc - np.array([predict_w(tabs, u, v)[0] for u, v in zip(uc, vc)]))  # W-wires
        best_dp = med_dzW / 2.0                       # symmetric U/V pitch offset (mm)

        geoms[gname] = dict(face=fi, dzW=dzW, med_dzW=med_dzW, rms_dzW=rms_dzW,
                            frac_before=frac_before, frac_after=frac_after,
                            r0=r0, best_dp=best_dp, store=store, tabs=tabs)
        print(f"  [{gname}] face={fi}  W-plane shift needed dZ_W = {med_dzW:+.2f} mm "
              f"(= {med_dzW/W_PITCH_MM:+.2f} W-pitch); per-tick RMS={rms_dzW:.2f} mm")
        print(f"        consistency within +-1/2 W-pitch: {frac_before*100:.0f}% before, "
              f"{frac_after*100:.0f}% after applying the shift")
        print(f"        equivalent symmetric U/V strip offset dp = {best_dp:+.2f} mm "
              f"({best_dp/UV_PITCH_MM:+.3f} induction strips)")

    make_plots(akey, cfg, tt, uc, vc, wc, res, ticks, gm, geoms)
    return cfg, geoms, (tt, uc, vc, wc)


# ----------------------------------------------------------------- plotting ---
def make_plots(akey, cfg, tt, uc, vc, wc, res, ticks, gm, geoms):
    # (a) track centroids vs tick
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, pl in enumerate(("U", "V", "W")):
        ax[i].plot(ticks, res[pl]["cen"], ',', color="0.7")
        ax[i].plot(tt, res[pl]["cen"][gm], '.', ms=2, color="C0")
        ax[i].set_ylabel(f"{pl} chan"); ax[i].grid(alpha=.3)
    ax[0].set_title(f"{cfg['label']}: charge-weighted centroid vs tick (evt{cfg['event']})")
    ax[-1].set_xlabel("tick")
    fig.tight_layout(); fig.savefig(f"{OUT}/pdvd_offset_track_anode{akey}.png", dpi=110)
    plt.close(fig)

    # (b) W-plane mismatch dZ_W vs tick (keep U,V fixed) + (c) its distribution
    half = W_PITCH_MM / 2.0
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for gname, g in geoms.items():
        ax[0].plot(tt, g["dzW"], '.', ms=2, label=f"{gname} (med {g['med_dzW']:+.1f} mm)")
    ax[0].axhspan(-half, half, color='g', alpha=.15, label="±½ W-pitch (consistent)")
    ax[0].axhline(0, color='k', lw=.8); ax[0].set_xlabel("tick")
    ax[0].set_ylabel("dZ_W = z(U∩V) - z(W wire)  [mm]")
    ax[0].set_title(f"{cfg['label']}: per-tick W-plane mismatch (U,V fixed)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for gname, g in geoms.items():
        ax[1].hist(g["dzW"], bins=80, histtype="step", label=f"{gname}")
        ax[1].axvline(g["med_dzW"], ls='--', color='0.5')
    ax[1].axvspan(-half, half, color='g', alpha=.15)
    ax[1].set_xlabel("dZ_W  [mm]"); ax[1].set_ylabel("ticks")
    ax[1].set_title("required W-plane shift = median dZ_W"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pdvd_offset_residual_anode{akey}.png", dpi=110)
    plt.close(fig)

    # (d) YZ crossing illustration for a few ticks (v3, delta=0)
    g = geoms["v3"]; tabs = g["tabs"]
    fig, axx = plt.subplots(figsize=(6.5, 6))
    sel = np.linspace(0, len(tt) - 1, min(8, len(tt))).astype(int)
    for k in sel:
        _, yz = predict_w(tabs, uc[k], vc[k])
        axx.plot(yz[1], yz[0], 'o', color='C3', ms=5)
    axx.set_xlabel("z [mm]"); axx.set_ylabel("y [mm]")
    axx.set_title(f"{cfg['label']}: U∩V crossings (v3)"); axx.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pdvd_offset_yz_anode{akey}.png", dpi=110)
    plt.close(fig)


def main():
    summary = {}
    for akey in (0, 4):
        cfg, geoms, _ = analyse_anode(akey)
        summary[akey] = (cfg, geoms)
    print("\n================ SUMMARY ================")
    for akey, (cfg, geoms) in summary.items():
        print(f"\n{cfg['label']}:")
        for gname, g in geoms.items():
            print(f"  {gname}: W-plane shift dZ_W = {g['med_dzW']:+.2f} mm "
                  f"({g['med_dzW']/W_PITCH_MM:+.2f} W-pitch); "
                  f"consistency ±½pitch {g['frac_before']*100:.0f}%→{g['frac_after']*100:.0f}%; "
                  f"equiv. U/V strip offset {g['best_dp']:+.2f} mm")
    print("\nplots written to", OUT)


if __name__ == "__main__":
    main()

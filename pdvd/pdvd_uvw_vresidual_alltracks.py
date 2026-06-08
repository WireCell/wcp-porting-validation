#!/usr/bin/env python
"""Same V fine-tune diagnostic as pdvd_uvw_vresidual.py, but using ALL tracks in
the plane (not just the one calibration track), with CORRECT per-face mapping.

Per tick: take the single dominant peak in U, V, W (one isolated track active).
The W peak channel selects the face (W is unwrapped -> face-unambiguous); use that
face's U/V/W wire maps.  Intersect the U wire and W wire, predict V there, and
histogram (V_predicted - V_measured) over all such ticks, after the common-mode
shift, with U and W fixed.  Centred on 0 => V has no room to move (for ALL tracks).

Writes the plot to /home/xqian/tmp and pdvd/pics.  Read-only on inputs.
"""
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from wirecell.util.wires import persist

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O
import pdvd_uvw_2dscan as S2

OUT = "/home/xqian/tmp"
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/pics"
HALF_V = O.UV_PITCH_MM / 2.0
DOM = 3.0

# per-anode common-mode basis s* (measured on the calibration track, vresidual.py)
SSTAR = {0: +13.21, 4: -9.80}
# full-plane channel windows (both faces) and tick windows
PLANE = {
    0: dict(event=0, U=(0, 476), V=(952, 1428), W=(1904, 2488), tick=(0, 6400)),
    4: dict(event=0, U=(6144, 6620), V=(7096, 7572), W=(8048, 8632), tick=(0, 5000)),
}


def build_faces(store, aid):
    a = [an for an in store.anodes if an.ident == aid][0]
    faces = {}
    for fi in a.faces:
        tabs = {}
        for pi in store.faces[fi].planes:
            ident = store.planes[pi].ident
            tabs[{0: "U", 1: "V", 2: "W"}[ident]] = O.plane_table(store, pi)
        faces[fi] = tabs
    return faces


def face_for_w(faces, wchan):
    for fi, tabs in faces.items():
        wc = tabs["W"]["chan"]
        if wc.min() <= wchan <= wc.max():
            return fi
    return None


def predict_v_pitch(tabs, u_chan, w_chan, sU):
    pdU = tabs["U"]["pdir"]; pdW = tabs["W"]["pdir"]; pdV = tabs["V"]["pdir"]
    pU = O.chan_to_pitch(tabs["U"], u_chan) + sU * pdU[1]
    pW = O.chan_to_pitch(tabs["W"], w_chan)
    P = np.linalg.solve(np.vstack([pdU, pdW]), np.array([pU, pW]))
    return pdV @ P


def collect(aid, store):
    faces = build_faces(store, aid)
    pc = PLANE[aid]; sU = SSTAR[aid]
    fn = O.MAGBASE.format(E=pc["event"], A=aid)
    f = uproot.open(fn)
    prof = {}
    for pl in ("U", "V", "W"):
        h = f[f"h{pl.lower()}_gauss{aid}"]
        v = np.clip(h.values(), 0, None)
        xe = h.axis(0).edges(); ch = (xe[:-1] + xe[1:]) / 2.0
        lo, hi = pc[pl]
        sel = (ch >= lo) & (ch <= hi)
        prof[pl] = (v[sel], ch[sel])
    thr = {pl: 0.08 * np.percentile(prof[pl][0][prof[pl][0] > 0], 99.5)
           for pl in ("U", "V", "W")}

    rV = []
    t0, t1 = pc["tick"]
    for t in range(t0, t1):
        peak = {}
        ok = True
        for pl in ("U", "V", "W"):
            v, ch = prof[pl]
            col = v[:, t]
            idx, _ = find_peaks(col, height=thr[pl], distance=3)
            if len(idx) == 0:
                ok = False; break
            hgt = col[idx]; order = np.argsort(hgt)[::-1]
            if len(idx) > 1 and hgt[order[0]] < DOM * hgt[order[1]]:
                ok = False; break             # dominant peak only
            peak[pl] = ch[idx[order[0]]]
        if not ok:
            continue
        fi = face_for_w(faces, peak["W"])
        if fi is None:
            continue
        tabs = faces[fi]
        # require U,V peaks to belong to this face's wire range
        if not (tabs["U"]["chan"].min() <= peak["U"] <= tabs["U"]["chan"].max()):
            continue
        if not (tabs["V"]["chan"].min() <= peak["V"] <= tabs["V"]["chan"].max()):
            continue
        pV_pred = predict_v_pitch(tabs, peak["U"], peak["W"], sU)
        pV_meas = O.chan_to_pitch(tabs["V"], peak["V"]) + sU * tabs["V"]["pdir"][1]
        r = pV_pred - pV_meas
        if abs(r) < 2.0 * O.UV_PITCH_MM:       # third-view gate: V prediction near a real V peak
            rV.append(r)
    return np.array(rV)


def main():
    store = persist.load(S2.V4)
    print("All-tracks V fine-tune diagnostic (per-face mapping, single-dominant peaks).")
    out = {}
    for aid in (0, 4):
        rV = collect(aid, store)
        med = float(np.median(rV)); rms = float(np.std(rV))
        rsig = 1.4826 * float(np.median(np.abs(rV - np.median(rV))))
        frac = float(np.mean(np.abs(rV) < HALF_V))
        faces = build_faces(store, aid)
        pdV_z = faces[list(faces)[0]]["V"]["pdir"][1]
        dV = med / (-pdV_z)
        out[aid] = (rV, med, rms, rsig, frac, dV)
        print(f"\n==== anode {aid} ====")
        print(f"  {len(rV)} all-track clean ticks")
        print(f"  V_pred-V_meas: median {med:+.3f} mm, RMS {rms:.2f}, robust-sigma {rsig:.2f} mm")
        print(f"  within ±½ V-pitch: {frac*100:.1f}%")
        print(f"  V fine-tune to re-centre: dV = {dV:+.3f} mm  [~0 => no room]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, aid in zip(axes, (0, 4)):
        rV, med, rms, rsig, frac, dV = out[aid]
        ax.hist(rV, bins=80, range=(-3 * O.UV_PITCH_MM, 3 * O.UV_PITCH_MM),
                color="C0", alpha=.85)
        ax.axvspan(-HALF_V, HALF_V, color="g", alpha=.15, label="±½ V-pitch")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(med, color="r", ls="--", lw=1.5,
                   label=f"median {med:+.2f} mm (V fine-tune {dV:+.2f} mm)")
        ax.set_xlabel("V$_{pred}$(from U∩W) − V$_{meas}$   [mm along V pitch]")
        ax.set_ylabel("ticks")
        ax.set_title(f"anode {aid}: ALL tracks ({len(rV)} ticks)\n"
                     f"robust-σ {rsig:.2f} mm, {frac*100:.0f}% within ½ V-pitch")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Does V have room to fine-tune? — ALL tracks, per-face mapping "
                 "(U & W fixed, common mode applied)", fontsize=11)
    fig.tight_layout()
    for d in (OUT, PICS):
        fig.savefig(f"{d}/pdvd_v_finetune_residual_alltracks.png", dpi=120)
    print(f"\nwrote {OUT}/pdvd_v_finetune_residual_alltracks.png and {PICS}/")


if __name__ == "__main__":
    main()

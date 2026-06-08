#!/usr/bin/env python
"""Whole-plane multi-track separability test (user's algorithm).

For a full anode plane (run 39324 evt0), at each tick:
  - find charge peaks in U, V, W (many tracks at many angles);
  - pair every U-peak with every V-peak, predict the crossing (y,z);
  - KEEP the combination only if a W peak lies within a couple wire pitches of
    the predicted W (the "check the third view" gate);
  - collect every surviving (U,V,W) triple's W-mismatch.

Then fit a SEPARATE fine shift (ddzU, ddzV) of the U and V planes (pure z, the W
pitch direction) on top of the coarse common-mode shift, over all triples, and
scan it in 2-D.  This is exactly the "use the whole plane / many tracks to
determine the fine U-vs-V shift" recipe.

Outputs the 2-D objective (consistent-triple count) + best fit to /home/xqian/tmp/.
Read-only on inputs.
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

OUT = "/home/xqian/tmp"
V4 = O.GEOMS["v4"]
HALF = O.W_PITCH_MM / 2.0                     # 2.55 mm
GATE = 2.5 * O.W_PITCH_MM                     # "couple wire pitches" third-view gate

# whole-plane configs: coarse common-mode z-shift = sum/2 (anode0 +26/2, anode4 -20/2)
PCFG = {
    0: dict(anode=0, event=0, tickwin=(0, 6400), coarse=+13.0, label="anode 0 (bottom CRP)"),
    4: dict(anode=4, event=0, tickwin=(0, 5000), coarse=-10.0, label="anode 4 (top CRP)"),
}


def crossing_coeffs(tabs):
    A = np.vstack([tabs["U"]["pdir"], tabs["V"]["pdir"]])
    Ainv = np.linalg.inv(A)
    a, b = Ainv[1, 0], Ainv[1, 1]
    return a, b, a * tabs["U"]["pdir"][1], b * tabs["V"]["pdir"][1]   # a,b,cU,cV


def collect_triples(pc, tabs, a, b):
    fn = O.MAGBASE.format(E=pc["event"], A=pc["anode"])
    f = uproot.open(fn)
    prof = {}
    for pl in ("U", "V", "W"):
        h = f[f"h{pl.lower()}_gauss{pc['anode']}"]
        v = np.clip(h.values(), 0, None)                    # (nchan, ntick)
        xe = h.axis(0).edges(); ch = (xe[:-1] + xe[1:]) / 2.0
        lo, hi = tabs[pl]["chan"].min(), tabs[pl]["chan"].max()
        sel = (ch >= lo) & (ch <= hi)
        prof[pl] = (v[sel], ch[sel])
    t0, t1 = pc["tickwin"]
    # per-plane peak height threshold
    thr = {pl: 0.08 * np.percentile(prof[pl][0][prof[pl][0] > 0], 99.5)
           for pl in ("U", "V", "W")}

    residuals = []
    for t in range(t0, t1):
        pk = {}
        for pl in ("U", "V", "W"):
            v, ch = prof[pl]
            col = v[:, t]
            idx, _ = find_peaks(col, height=thr[pl], distance=3)
            pk[pl] = ch[idx]
        if len(pk["U"]) == 0 or len(pk["V"]) == 0 or len(pk["W"]) == 0:
            continue
        pU = O.chan_to_pitch(tabs["U"], pk["U"])             # (nu,)
        pV = O.chan_to_pitch(tabs["V"], pk["V"])             # (nv,)
        zc = a * pU[:, None] + b * pV[None, :] + pc["coarse"]  # (nu,nv) coarse crossing z
        wz = O.w_chan_to_z(tabs["W"], pk["W"])               # (nw,) measured W z
        # nearest W peak to each crossing
        d = zc[:, :, None] - wz[None, None, :]               # (nu,nv,nw)
        k = np.argmin(np.abs(d), axis=2)
        nearest = np.take_along_axis(d, k[:, :, None], axis=2)[:, :, 0]   # (nu,nv)
        good = np.abs(nearest) < GATE                        # third-view gate
        residuals.extend(nearest[good].ravel().tolist())
    return np.array(residuals)


def analyse(pc, store):
    fi, tabs = O.find_face_planes(store, pc["anode"], (tabs_w_window(store, pc)))
    a, b, cU, cV = crossing_coeffs(tabs)
    res0 = collect_triples(pc, tabs, a, b)
    print(f"\n==== {pc['label']}  face {fi} ====")
    print(f"  geometry sensitivities cU={cU:.4f}  cV={cV:.4f}")
    print(f"  selected {len(res0)} whole-plane triples (W within {GATE:.1f} mm)")

    # fine 2-D scan of separate (ddzU, ddzV) on top of the coarse common-mode shift
    g = np.arange(-8.0, 8.001, 0.25)
    C = np.empty((len(g), len(g)))
    for i, du in enumerate(g):
        for j, dv in enumerate(g):
            C[j, i] = np.mean(np.abs(res0 + cU * du + cV * dv) < HALF)
    cmax = C.max()
    best = np.argwhere(C >= cmax - 1e-9)
    sums = sorted({round(g[i] + g[j], 3) for j, i in best})
    diffs = sorted({round(g[i] - g[j], 3) for j, i in best})
    print(f"  fine-fit max consistent-triple frac = {cmax*100:.1f}%")
    print(f"    best (ddzU+ddzV) values = {sums}   (single value => sum-only)")
    print(f"    best (ddzU-ddzV) values span = [{min(diffs)}, {max(diffs)}]  "
          f"(wide/full-range => difference UNCONSTRAINED)")
    return pc, g, C, cmax, (cU, cV), len(res0)


def tabs_w_window(store, pc):
    # full W coverage of the anode's collection face (pick any W channel of it)
    a = [an for an in store.anodes if an.ident == pc["anode"]][0]
    for fi in a.faces:
        face = store.faces[fi]
        wpl = [pi for pi in face.planes if store.planes[pi].ident == 2]
        if not wpl:
            continue
        wch = [store.wires[i].channel for i in store.planes[wpl[0]].wires]
        # the bottom/top face we imaged; use its middle channel as the locator
        return (int(np.median(wch)), int(np.median(wch)))
    raise RuntimeError("no W face")


def plot(pc, g, C, cmax, nt):
    GU, GV = np.meshgrid(g, g)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cs = ax.contourf(GU, GV, C * 100, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="consistent whole-plane triples [%]")
    ax.plot(g, -g, "r--", lw=1, label="ddzU+ddzV = 0 (coarse already optimal)")
    ax.set_xlabel("ddzU  (extra U z-shift, mm)")
    ax.set_ylabel("ddzV  (extra V z-shift, mm)")
    ax.set_title(f"{pc['label']}: whole-plane fine fit\n{nt} triples, {C.size} grid pts")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = f"{OUT}/pdvd_wholeplane_anode{pc['anode']}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")


def main():
    store = persist.load(V4)
    print("Whole-plane multi-track separability test (v4 geometry, separate U,V fine shift).")
    for akey in (0, 4):
        pc, g, C, cmax, cuv, nt = analyse(PCFG[akey], store)
        plot(pc, g, C, cmax, nt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Decisive separability test: do the two real tracks constrain SEPARATE U and V
z-shifts, or only their sum?

Motivation (user): a single track has many ticks and is wiggled, so a 2-D scan of
(dzU, dzV) -- separate pure-z shifts of the U and V wire lines, both along the W
pitch direction -- with a coordinate descent should reveal whether U and V want
different shifts.

What this script does (genuine empirical test, no shortcut):
  - rebuild, from the ACTUAL v4 wire geometry, the per-tick z of the U/\\cap V crossing
    as a function of separate (dzU, dzV), and the W-mismatch dZ_W(t)=z_cross-z_W.
  - report the linear sensitivities cU=d z_cross/d dzU and cV=d z_cross/d dzV
    measured from the real pdir vectors (if the U/V planes are mirror-symmetric
    about the vertical W, both are exactly 1/2 -> only dzU+dzV is observable).
  - scan a 2-D grid over (dzU,dzV); objective A = consistency fraction
    (|dZ_W|<1/2 W-pitch), objective B = RMS(dZ_W). Plot contour maps.
  - run the user's coordinate descent (joint -> fix U opt V -> fix V opt U, rounds)
    from several seeds; log the trajectory.

Read-only on all inputs; writes plots+summary to /home/xqian/tmp/.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wirecell.util.wires import persist

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd")
import pdvd_uvw_offset as O

OUT = "/home/xqian/tmp"
V4 = O.GEOMS["v4"]
HALF = O.W_PITCH_MM / 2.0


def crossing_z_coeffs(tabs):
    """From the real wire geometry, return (z0_fn, cU, cV) such that the z of the
    U/\\cap V crossing for measured channels (u,v) under separate pure-z shifts
    (dzU,dzV) is   z_cross = z0(u,v) + cU*dzU + cV*dzV.

    Shifting a wire's endpoints by (0,dz) in (y,z) moves its pitch coordinate by
    dz*pdir_z.  The crossing solves [pdir_U; pdir_V] (y,z) = (pU,pV); the z-row of
    the inverse gives d z_cross/d pU = a, /d pV = b, so cU=a*pdir_U_z, cV=b*pdir_V_z.
    """
    A = np.vstack([tabs["U"]["pdir"], tabs["V"]["pdir"]])     # 2x2
    Ainv = np.linalg.inv(A)
    a, b = Ainv[1, 0], Ainv[1, 1]                            # z-row
    cU = a * tabs["U"]["pdir"][1]
    cV = b * tabs["V"]["pdir"][1]

    def z0(uc, vc):
        pU = O.chan_to_pitch(tabs["U"], uc)
        pV = O.chan_to_pitch(tabs["V"], vc)
        return a * pU + b * pV
    return z0, cU, cV


def analyse(akey, store):
    cfg = O.CONFIGS[akey]
    fn = O.MAGBASE.format(E=cfg["event"], A=cfg["anode"])
    ticks, res = O.centroids(fn, cfg["anode"], cfg)
    gm = O.good_mask(res)
    uc = res["U"]["cen"][gm]; vc = res["V"]["cen"][gm]; wc = res["W"]["cen"][gm]

    fi, tabs = O.find_face_planes(store, cfg["anode"], cfg["win"]["W"])
    z0fn, cU, cV = crossing_z_coeffs(tabs)
    z0 = z0fn(uc, vc)                              # crossing z, unshifted (mm)
    zw = O.w_chan_to_z(tabs["W"], wc)             # measured W z (mm)
    base = z0 - zw                                 # dZ_W at (dzU,dzV)=(0,0)

    # dZ_W(t; dzU,dzV) = base(t) + cU*dzU + cV*dzV
    def frac_consistent(dzU, dzV):
        d = base + cU * dzU + cV * dzV
        return float(np.mean(np.abs(d) < HALF))

    def rms(dzU, dzV):
        d = base + cU * dzU + cV * dzV
        return float(np.sqrt(np.mean(d ** 2)))

    print(f"\n==== {cfg['label']}  ({gm.sum()} usable ticks, face {fi}) ====")
    print(f"  measured-from-geometry sensitivities:  "
          f"cU = dz_cross/d(dzU) = {cU:.4f},  cV = {cV:.4f}")
    print(f"  pdir_U = ({tabs['U']['pdir'][0]:+.3f},{tabs['U']['pdir'][1]:+.3f})  "
          f"pdir_V = ({tabs['V']['pdir'][0]:+.3f},{tabs['V']['pdir'][1]:+.3f})")
    if abs(cU - cV) < 1e-6:
        print(f"  -> cU == cV exactly: the objective depends on (dzU,dzV) ONLY through "
              f"the combination cU*(dzU+dzV); the difference is UNOBSERVABLE.")

    # ---- 2-D grid scan ----------------------------------------------------
    g = np.arange(-20.0, 20.01, 0.5)
    F = np.empty((len(g), len(g))); R = np.empty_like(F)
    for i, du in enumerate(g):
        for j, dv in enumerate(g):
            F[j, i] = frac_consistent(du, dv)     # rows=dzV, cols=dzU
            R[j, i] = rms(du, dv)
    fmax = F.max()
    best = np.argwhere(F >= fmax - 1e-9)
    sums = sorted({round(g[i] + g[j], 2) for j, i in best})
    print(f"  grid max consistency = {fmax*100:.1f}% ; reached on {len(best)} grid "
          f"cells; their (dzU+dzV) values = {sums}  (a single value => diagonal valley)")

    # ---- coordinate descent (user's recipe) from several seeds ------------
    fine = np.arange(-20.0, 20.001, 0.25)
    def opt_axis(fix_val, which):
        # maximize consistency over one axis
        vals = [frac_consistent(fix_val, x) if which == "V" else frac_consistent(x, fix_val)
                for x in fine]
        return float(fine[int(np.argmax(vals))])
    print("  coordinate descent (maximize consistency), trajectories:")
    for seed in [(0.0, 0.0), (-10.0, 10.0), (15.0, -5.0)]:
        du, dv = seed
        # step 1: joint move along the diagonal (shift U,V together) to best sum
        s = [frac_consistent(t, t) for t in fine]
        t0 = float(fine[int(np.argmax(s))]); du = dv = t0
        traj = [(du, dv)]
        for _ in range(4):
            dv = opt_axis(du, "V"); traj.append((du, dv))
            du = opt_axis(dv, "U"); traj.append((du, dv))
        tr = " -> ".join(f"({a:+.2f},{b:+.2f})" for a, b in traj)
        print(f"    seed {seed}:  {tr}   final sum={du+dv:+.2f} mm, "
              f"diff={du-dv:+.2f} mm, frac={frac_consistent(du,dv)*100:.1f}%")

    return cfg, g, F, R, fmax, (cU, cV)


def plot(akey, cfg, g, F, R, fmax):
    GU, GV = np.meshgrid(g, g)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    cs = ax[0].contourf(GU, GV, F * 100, levels=20, cmap="viridis")
    ax[0].contour(GU, GV, F * 100, levels=[fmax * 100 - 1], colors="w", linewidths=1)
    fig.colorbar(cs, ax=ax[0], label="consistency  |dZ_W|<½ W-pitch  [%]")
    ax[0].set_title(f"{cfg['label']}: consistency vs (dzU,dzV)")
    cs2 = ax[1].contourf(GU, GV, R, levels=20, cmap="magma_r")
    fig.colorbar(cs2, ax=ax[1], label="RMS(dZ_W)  [mm]")
    ax[1].set_title(f"{cfg['label']}: RMS(dZ_W) vs (dzU,dzV)")
    for a in ax:
        # diagonal dzU+dzV = const passing through the optimum
        a.plot(g, -g + g[np.unravel_index(F.argmax(), F.shape)[1]] * 0 +
               (g[np.unravel_index(F.argmax(), F.shape)[1]] +
                g[np.unravel_index(F.argmax(), F.shape)[0]]), "r--", lw=1,
               label="dzU+dzV = const (best)")
        a.set_xlabel("dzU  (U z-shift, mm)"); a.set_ylabel("dzV  (V z-shift, mm)")
        a.set_aspect("equal"); a.legend(loc="upper right", fontsize=8)
        a.set_xlim(g[0], g[-1]); a.set_ylim(g[0], g[-1])
    fig.tight_layout()
    p = f"{OUT}/pdvd_uvw_2dscan_anode{akey}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")


def main():
    store = persist.load(V4)
    print("2-D separability scan on the v4 geometry (separate pure-z U,V shifts).")
    for akey in (0, 4):
        cfg, g, F, R, fmax, (cU, cV) = analyse(akey, store)
        plot(akey, cfg, g, F, R, fmax)
    print("\nIf cU==cV and the consistency optimum is a single diagonal line "
          "dzU+dzV=const, the two tracks fix only the SUM; the U-vs-V difference "
          "is geometrically unconstrained (it only translates blobs in y).")


if __name__ == "__main__":
    main()

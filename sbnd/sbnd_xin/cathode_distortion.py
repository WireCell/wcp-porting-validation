#!/usr/bin/env python3
"""Map the SBND cathode-plane field-cage distortion across the (Y,Z) plane.

After Q/L matching + the per-cluster T0 correction every matched cosmic is
reconstructed at its true drift position, so the geometry near the central
cathode (x = cathode_x ~ +-0.45 cm) can be used to *map* the distortion and
localise the suspected top-corner field-cage problem.

Data source: the per-event calib dump (work/ql_evt<ID>/calib-evt<ID>.json,
written by QLMatching::dump_calib with run_ql_evt.sh -calib).  Per event it has
  drift_speed (cm/us); geometry["<apa>"] = {cathode_x, sign_offset, y/z bounds};
  clusters[] = {uid, apa, x,y,z,q} RAW un-shifted points (cm);
  bundles[]  = {main_cluster, other_clusters, flash_gid, auto_selected, ...};
  flashes[]  = {gid, time(us), group, ...}.
The T0-corrected x of a cluster in its auto_selected bundle is
  x_corr = x_raw + sign_offset * flash_time_us * drift_speed_cm_per_us
(y,z unchanged).  This is the exact convention dump_calib documents.  We use
*every* matched (auto_selected) track's T0-corrected points.

The cleanest *absolute* transverse distortion comes from cathode-crossing
(xTPC) tracks (the two TPC halves must meet at the cathode); that uses the
vetted QLCATHODE log lines (run_ql_evt.sh -cathode-diag).  Single-track
observables (off-axis residual, dx) are higher-statistics but weaker (curvature
/ T0-degenerate) — read them against the MC reference, not as absolutes.

Outputs (pics/):
  cathode_coverage_yz.png             #1  near-cathode SLAB occupancy of ALL matched-track points
  cathode_xresidual_yz.png            #2  mean dx = x_corr,end - cathode_x over (y,z)  (drift, T0-degenerate)
  cathode_transverse_residual_yz.png  #3  per-track cathode-end off-axis transverse residual (curvature)
  cathode_xtpc_perp_yz.png            #4  xTPC transverse offset scatter+quiver (the clean, T0-immune signal)
  cathode_profiles.png                #5  transverse/perp residual vs Y and vs Z
  cathode_furthest_points_yz.png      #6  long-track cathode-most point map, coloured by dx
  cathode_surface_3d.png              #7  apparent cathode as a 2-D surface in 3-D (drift residual over Y,Z)
  cathode_closest_yz.png              #8  per-(Y,Z)-bin closest approach of charge to the cathode

Usage:  python3 cathode_distortion.py [-j NPROC]
"""
import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PICS = os.path.join(HERE, "pics")
CALIB_GLOB = os.path.join(HERE, "work", "ql_evt*", "calib-evt*.json")

# MC reference: the original 10 mc events (small ids) + the input-10files-mc
# baseline, which run_mcbase.sh remaps to globally-unique ids >= 700000.
# Everything else (lan-reco2 + input-1file) is data.
MC_IDS = {2, 9, 11, 12, 14, 18, 31, 35, 41, 42}
MC_BASELINE_LO = 700000


def is_mc(evt_id):
    return evt_id in MC_IDS or evt_id >= MC_BASELINE_LO


# cathode-plane drift position per TPC (cm); constant in the calib geometry block.
CATHODE_X = {0: -0.45, 1: 0.45}

# (Y,Z) plane extent (cm).  Y vertical, Z beam.
Y_LO, Y_HI = -200.0, 200.0
Z_LO, Z_HI = 0.0, 500.0
NY, NZ = 8, 10            # coarse bins for the per-bin mean maps (#2, #3)
OY, OZ = 40, 50           # fine bins for the point-occupancy map (#1)

BAND = 15.0       # |x_corr - cathode_x| < BAND => near-cathode slab / "cathode-reaching"
NPTS_MIN = 20     # minimum cluster points for a stable PCA axis
LONG_MIN = 60.0   # principal-axis extent (cm) for the long-track samples (#3, #6)
RES_MAX = 20.0    # drop pathological off-axis residuals

YEDGES = np.linspace(Y_LO, Y_HI, NY + 1)
ZEDGES = np.linspace(Z_LO, Z_HI, NZ + 1)
OYEDGES = np.linspace(Y_LO, Y_HI, OY + 1)
OZEDGES = np.linspace(Z_LO, Z_HI, OZ + 1)


# ---------------------------------------------------------------------------
# Per-event extraction (run in a worker process)
# ---------------------------------------------------------------------------
def evt_id_of(path):
    m = re.search(r"calib-evt(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def pca_dir(P):
    """First principal axis (unit) of an (N,3) point cloud."""
    c = P.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return V[:, np.argmax(w)]


def process_event(path):
    """Per-event cathode observables from all auto_selected (matched) tracks.

    Returns (cover, trans, furth, occ, mind):
      cover : list (apa, y_end, z_end, dx)           cathode-reaching clusters (1 pt each)
      trans : list (apa, y_end, z_end, res_y, res_z) long-track cathode-end off-axis residual
      furth : list (apa, y_end, z_end, dx, extent)   long-track cathode-most point
      occ   : (OY,OZ) histogram of ALL matched-track points in the near-cathode slab
      mind  : {apa: (OY,OZ)} per-(Y,Z)-bin MIN |x_corr - cathode_x| over all matched-track
              points = closest approach of charge to the cathode (large ⇒ charge stops short)
    """
    occ = np.zeros((OY, OZ))
    mind = {0: np.full((OY, OZ), np.inf), 1: np.full((OY, OZ), np.inf)}
    try:
        d = json.load(open(path))
    except Exception:
        return [], [], [], occ, mind
    drift = d["drift_speed"]                                   # cm/us
    geom = {int(k): v for k, v in d["geometry"].items()}
    ftime = {f["gid"]: f["time"] for f in d["flashes"]}        # us
    clu = {c["uid"]: c for c in d["clusters"]}

    cover, trans, furth = [], [], []
    seen = set()
    for b in d["bundles"]:
        if not b["auto_selected"]:
            continue
        gid = b["flash_gid"]
        if gid not in ftime:
            continue
        for uid in [b["main_cluster"], *b["other_clusters"]]:
            if uid in seen or uid not in clu:
                continue
            seen.add(uid)
            c = clu[uid]
            apa = c["apa"]
            if apa not in geom:
                continue
            g = geom[apa]
            cath = g["cathode_x"]
            X = np.asarray(c["x"], float) + g["sign_offset"] * ftime[gid] * drift  # T0-corrected
            Y = np.asarray(c["y"], float)
            Z = np.asarray(c["z"], float)
            if X.size < NPTS_MIN:
                continue
            dcath = np.abs(X - cath)
            # closest-approach map: per (Y,Z) bin, MIN |x_corr-cath| over ALL track points
            inb = (Z >= Z_LO) & (Z < Z_HI) & (Y >= Y_LO) & (Y < Y_HI)
            if inb.any():
                iz = ((Z[inb] - Z_LO) / (Z_HI - Z_LO) * OZ).astype(int)
                iy = ((Y[inb] - Y_LO) / (Y_HI - Y_LO) * OY).astype(int)
                np.minimum.at(mind[apa], (iy, iz), dcath[inb])
            # slab occupancy: ALL points of this track within the near-cathode band
            slab = dcath < BAND
            if slab.any():
                h, _, _ = np.histogram2d(Z[slab], Y[slab], [OZEDGES, OYEDGES])
                occ += h.T
            ei = int(np.argmin(dcath))
            if dcath[ei] > BAND:                               # track does not reach the cathode
                continue
            y_end, z_end, dx = Y[ei], Z[ei], X[ei] - cath
            cover.append((apa, y_end, z_end, dx))

            P = np.column_stack([X, Y, Z])
            ctr = P.mean(0)
            dirv = pca_dir(P)
            ext = float(np.ptp((P - ctr) @ dirv))
            if ext < LONG_MIN:
                continue
            furth.append((apa, y_end, z_end, dx, ext))
            # off-axis transverse residual of the cathode-end vs the full-track axis
            # (no extrapolation -> physical scale; straight track -> ~0; curvature -> >0)
            v = P[ei] - ctr
            perp = v - (v @ dirv) * dirv
            if np.hypot(perp[1], perp[2]) <= RES_MAX:
                trans.append((apa, y_end, z_end, perp[1], perp[2]))
    return cover, trans, furth, occ, mind


# ---------------------------------------------------------------------------
# QLCATHODE log parsing (xTPC crossers, the clean transverse signal)
# ---------------------------------------------------------------------------
QLC_RE = re.compile(
    r"QLCATHODE\s+\d+\s+0/\d+\s+1/\d+\s+[-\d.]+\s+[-\d.]+\s+d=([-\d.]+)\s+"
    r"p0=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+p1=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"dir0=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+dir1=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"conn=\(([-\d.]+),([-\d.]+),([-\d.]+)\)"
)


def parse_qlcathode(logpaths):
    """Per xTPC pair: (y_mid, z_mid, perp_y, perp_z, perp_mag, d).

    perp = conn minus its component along dhat = unit(dir0n + dir1n); the
    transverse (y,z) part is the artifact-immune distortion (x is T0-degenerate).
    """
    out = []
    for lp in logpaths:
        try:
            txt = open(lp, errors="ignore").read()
        except Exception:
            continue
        for m in QLC_RE.finditer(txt):
            g = list(map(float, m.groups()))
            d = g[0]
            p0 = np.array(g[1:4]); p1 = np.array(g[4:7])
            dir0 = np.array(g[7:10]); dir1 = np.array(g[10:13])
            conn = np.array(g[13:16])
            n0 = dir0 / (np.linalg.norm(dir0) or 1)
            n1 = dir1 / (np.linalg.norm(dir1) or 1)
            if n0 @ conn < 0:
                n0 = -n0
            if n1 @ conn < 0:
                n1 = -n1
            dhat = n0 + n1
            dhat = dhat / (np.linalg.norm(dhat) or 1)
            perp = conn - (conn @ dhat) * dhat
            ymid, zmid = 0.5 * (p0[1] + p1[1]), 0.5 * (p0[2] + p1[2])
            out.append((ymid, zmid, perp[1], perp[2], np.hypot(perp[1], perp[2]), d))
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def binned_mean(z_coord, y_coord, val, mincount=3):
    """Mean of `val` per (z,y) bin; returns (y,z) array, NaN where < mincount."""
    s, _, _ = np.histogram2d(z_coord, y_coord, [ZEDGES, YEDGES], weights=val)
    n, _, _ = np.histogram2d(z_coord, y_coord, [ZEDGES, YEDGES])
    with np.errstate(invalid="ignore"):
        m = s / n
    m[n < mincount] = np.nan
    return m.T, n.T


def corners_annotate(ax):
    for zz, yy, lab in [(Z_LO, Y_HI, "top-L"), (Z_HI, Y_HI, "top-R"),
                        (Z_LO, Y_LO, "bot-L"), (Z_HI, Y_LO, "bot-R")]:
        ax.plot(zz, yy, "k+", ms=9, mew=1.5)
        ax.annotate(lab, (zz, yy), fontsize=7,
                    ha="left" if zz == Z_LO else "right",
                    va="top" if yy == Y_HI else "bottom")


def extent():
    return [Z_LO, Z_HI, Y_LO, Y_HI]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_coverage(occ):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for cc, mode in enumerate(["data", "mc"]):
        a = ax[cc]
        m = occ[mode]
        norm = m / m[m > 0].mean() if (m > 0).any() else m       # normalise to mean occupancy
        im = a.imshow(norm, origin="lower", aspect="auto", extent=extent(),
                      cmap="viridis", vmin=0, vmax=3)
        fig.colorbar(im, ax=a, label="points / mean")
        corners_annotate(a)
        a.set_title(f"{mode}  near-cathode slab |x-x_cath|<{BAND:.0f}cm  ({int(m.sum())} pts)")
        a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("Near-cathode occupancy of all matched-track points — depletion flags distortion/dead region")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_coverage_yz.png"), dpi=110)
    plt.close(fig)


def plot_xresidual(samp):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for r, mode in enumerate(["data", "mc"]):
        for cc, apa in enumerate([0, 1]):
            a = ax[r, cc]
            s = np.array([t[1:4] for t in samp[mode]["cover"] if t[0] == apa])
            if len(s):
                m, _ = binned_mean(s[:, 1], s[:, 0], s[:, 2])
                im = a.imshow(m, origin="lower", aspect="auto", extent=extent(),
                              cmap="coolwarm", vmin=-6, vmax=6)
                fig.colorbar(im, ax=a, label=r"$\langle x_{corr,end}-x_{cath}\rangle$ [cm]")
            corners_annotate(a)
            a.set_title(f"{mode}  TPC{apa}")
            a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("Cathode-end drift residual dx (degenerate with T0/v for single tracks — read vs MC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_xresidual_yz.png"), dpi=110)
    plt.close(fig)


def plot_transverse(samp):
    # Data-only spatial map: 10 MC events cannot fill an 8x10 grid (MC contrast is
    # carried by the 1-D profiles and the aggregate medians instead).
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    hot = {}
    for cc, apa in enumerate([0, 1]):
        a = ax[cc]
        s = np.array([(t[1], t[2], np.hypot(t[3], t[4]), t[3], t[4])
                      for t in samp["data"]["trans"] if t[0] == apa])
        if len(s):
            m, n = binned_mean(s[:, 1], s[:, 0], s[:, 2])
            im = a.imshow(m, origin="lower", aspect="auto", extent=extent(),
                          cmap="magma", vmin=0, vmax=3)
            fig.colorbar(im, ax=a, label=r"$\langle|\Delta_\perp|\rangle$ [cm]")
            qy, _ = binned_mean(s[:, 1], s[:, 0], s[:, 3])
            qz, _ = binned_mean(s[:, 1], s[:, 0], s[:, 4])
            yc = 0.5 * (YEDGES[:-1] + YEDGES[1:]); zc = 0.5 * (ZEDGES[:-1] + ZEDGES[1:])
            ZZ, YY = np.meshgrid(zc, yc)
            a.quiver(ZZ, YY, qz, qy, angles="xy", color="cyan", scale=30, width=0.005)
            # hotspot = largest well-populated data bin
            mh = np.where(n >= 5, m, np.nan)
            if np.isfinite(mh).any():
                iy, iz = np.unravel_index(np.nanargmax(mh), mh.shape)
                z0, y0 = 0.5 * (ZEDGES[iz] + ZEDGES[iz + 1]), 0.5 * (YEDGES[iy] + YEDGES[iy + 1])
                hot[apa] = (z0, y0, float(mh[iy, iz]))
                a.plot(z0, y0, "o", mfc="none", mec="lime", ms=18, mew=2)
        corners_annotate(a)
        a.set_title(f"data  TPC{apa}  (N={len(s)})")
        a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("Per-track cathode-end off-axis transverse residual, DATA (curvature); "
                 "cyan=mean (Δz,Δy), green=hotspot.  MC too sparse to bin (see profiles)")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_transverse_residual_yz.png"), dpi=110)
    plt.close(fig)
    return hot


def plot_xtpc(xtpc):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for cc, mode in enumerate(["data", "mc"]):
        a = ax[cc]
        s = np.array(xtpc[mode]) if xtpc[mode] else np.empty((0, 6))
        if len(s):
            sc = a.scatter(s[:, 1], s[:, 0], c=s[:, 4], cmap="magma", vmin=0, vmax=4,
                           s=24, edgecolor="k", linewidth=0.3)
            fig.colorbar(sc, ax=a, label="|perp⊥| [cm]")
            a.quiver(s[:, 1], s[:, 0], s[:, 3], s[:, 2], angles="xy",
                     color="tab:cyan", scale=40, width=0.005)
            med = np.median(s[:, 4])
            a.text(0.02, 0.98, f"N={len(s)} crossers\nmedian |perp|={med:.2f} cm",
                   transform=a.transAxes, va="top", fontsize=9,
                   bbox=dict(fc="white", alpha=0.7))
        corners_annotate(a)
        a.set_xlim(Z_LO, Z_HI); a.set_ylim(Y_LO, Y_HI)
        a.set_title(f"{mode}  xTPC cathode crossers")
        a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("xTPC cathode-crosser transverse offset (clean, T0-immune) — cyan = (Δz,Δy) perp")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_xtpc_perp_yz.png"), dpi=110)
    plt.close(fig)


def profile(ax, coord, val, edges, label, color, ls="-"):
    idx = np.digitize(coord, edges) - 1
    xc = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(len(xc), np.nan); err = np.full(len(xc), np.nan)
    for i in range(len(xc)):
        v = val[idx == i]
        if len(v) >= 3:
            mean[i] = v.mean(); err[i] = v.std() / np.sqrt(len(v))
    ax.errorbar(xc, mean, yerr=err, marker="o", ms=4, label=label, color=color,
                capsize=2, ls=ls)


def plot_profiles(samp, xtpc):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mode, col in [("data", "tab:red"), ("mc", "tab:blue")]:
        t = np.array([(x[1], x[2], np.hypot(x[3], x[4])) for x in samp[mode]["trans"]])
        if len(t):
            profile(ax[0], t[:, 0], t[:, 2], YEDGES, f"{mode} per-track", col, "--")
            profile(ax[1], t[:, 1], t[:, 2], ZEDGES, f"{mode} per-track", col, "--")
        x = np.array(xtpc[mode]) if xtpc[mode] else np.empty((0, 6))
        if len(x):
            profile(ax[0], x[:, 0], x[:, 4], YEDGES, f"{mode} xTPC", col, "-")
            profile(ax[1], x[:, 1], x[:, 4], ZEDGES, f"{mode} xTPC", col, "-")
    ax[0].set_xlabel("Y [cm]"); ax[1].set_xlabel("Z [cm]")
    for a in ax:
        a.set_ylabel("transverse residual [cm]"); a.grid(alpha=0.3); a.legend(fontsize=8)
        a.set_ylim(0, None)
    fig.suptitle("Transverse-residual profiles vs Y and vs Z (solid=xTPC absolute, dashed=per-track curvature)")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_profiles.png"), dpi=110)
    plt.close(fig)


def plot_closest(mind, cy=10, cz=10):
    """(Y,Z) map of the closest approach of charge to the cathode per bin
    (min |x_corr - cathode_x|). 0 ⇒ charge reaches the cathode; large ⇒ charge
    stops short there — a candidate distortion / dead region.  The fine (OY,OZ)
    min grid is block-reduced to (cy,cz) so each bin aggregates many tracks
    (otherwise a bin is large simply where no cosmic happened to graze)."""
    by, bz = OY // cy, OZ // cz                          # block sizes (5x5 for 40x50->8x10? here 4x5)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for r, mode in enumerate(["data", "mc"]):
        for cc, apa in enumerate([0, 1]):
            a = ax[r, cc]
            m = mind[mode][apa].reshape(cy, by, cz, bz).min(axis=(1, 3))
            m = np.where(np.isfinite(m), m, np.nan)
            im = a.imshow(m, origin="lower", aspect="auto", extent=extent(),
                          cmap="turbo", vmin=0, vmax=10)
            fig.colorbar(im, ax=a, label="closest |x-x_cath| [cm]")
            corners_annotate(a)
            a.set_title(f"{mode}  TPC{apa}")
            a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("Closest approach of charge to the cathode per (Y,Z) bin (rebinned) — "
                 "large = charge stops short (distortion or dead region)")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_closest_yz.png"), dpi=110)
    plt.close(fig)


def plot_surface_3d(samp):
    """The apparent cathode as a 2-D surface in 3-D: mean drift residual
    dx = x_corr,end - cathode_x of all cathode-reaching tracks, per (Y,Z) cell.
    A flat sheet at dx=0 ⇒ no distortion; warping ⇒ drift-direction distortion
    (per-track random T0 errors average out over the many tracks in each cell)."""
    yc = 0.5 * (YEDGES[:-1] + YEDGES[1:]); zc = 0.5 * (ZEDGES[:-1] + ZEDGES[1:])
    ZZc, YYc = np.meshgrid(zc, yc)
    fig = plt.figure(figsize=(14, 10))
    for r, mode in enumerate(["data", "mc"]):
        for cc, apa in enumerate([0, 1]):
            ax = fig.add_subplot(2, 2, r * 2 + cc + 1, projection="3d")
            s = np.array([t[1:4] for t in samp[mode]["cover"] if t[0] == apa])  # y,z,dx
            if len(s):
                m, n = binned_mean(s[:, 1], s[:, 0], s[:, 2], mincount=3)  # (y,z)
                ok = np.isfinite(m)
                if ok.sum() >= 3:
                    ax.plot_trisurf(ZZc[ok], YYc[ok], m[ok], cmap="coolwarm",
                                    vmin=-6, vmax=6, linewidth=0.1, antialiased=True)
            ax.plot_surface(ZZc, YYc, np.zeros_like(ZZc), color="gray", alpha=0.12)
            ax.set_zlim(-6, 6)
            ax.set_title(f"{mode}  TPC{apa}  (N={len(s)})")
            ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
            ax.set_zlabel(r"$\langle x_{end}-x_{cath}\rangle$ [cm]")
            ax.view_init(elev=22, azim=-60)
    fig.suptitle("Apparent cathode surface in 3D (drift residual over Y,Z) — flat=undistorted, warped=distortion")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_surface_3d.png"), dpi=110)
    plt.close(fig)


def plot_furthest(samp):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for cc, mode in enumerate(["data", "mc"]):
        a = ax[cc]
        s = np.array([t[1:5] for t in samp[mode]["furth"]])  # y,z,dx,ext
        if len(s):
            sc = a.scatter(s[:, 1], s[:, 0], c=s[:, 2], cmap="coolwarm", vmin=-6, vmax=6,
                           s=14, edgecolor="none")
            fig.colorbar(sc, ax=a, label=r"$x_{corr,end}-x_{cath}$ [cm]")
        corners_annotate(a)
        a.set_xlim(Z_LO, Z_HI); a.set_ylim(Y_LO, Y_HI)
        a.set_title(f"{mode}  long tracks (ext>{LONG_MIN:.0f}cm, N={len(s)})")
        a.set_xlabel("Z [cm]"); a.set_ylabel("Y [cm]")
    fig.suptitle("Furthest (cathode-most) point of long tracks in (Y,Z), coloured by drift residual")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "cathode_furthest_points_yz.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--nproc", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()
    os.makedirs(PICS, exist_ok=True)

    paths = sorted(glob.glob(CALIB_GLOB))
    print(f"calib files: {len(paths)}  (nproc={args.nproc})")
    with Pool(args.nproc) as pool:
        results = pool.map(process_event, paths)

    samp = {m: {"cover": [], "trans": [], "furth": []} for m in ("data", "mc")}
    occ = {m: np.zeros((OY, OZ)) for m in ("data", "mc")}
    mind = {m: {0: np.full((OY, OZ), np.inf), 1: np.full((OY, OZ), np.inf)} for m in ("data", "mc")}
    xtpc = {"data": [], "mc": []}
    for path, (cover, trans, furth, oc, md) in zip(paths, results):
        mode = "mc" if is_mc(evt_id_of(path)) else "data"
        samp[mode]["cover"] += cover
        samp[mode]["trans"] += trans
        samp[mode]["furth"] += furth
        occ[mode] += oc
        for apa in (0, 1):
            np.minimum(mind[mode][apa], md[apa], out=mind[mode][apa])
    for mode in ("data", "mc"):
        ids = [p for p in paths if is_mc(evt_id_of(p)) == (mode == "mc")]
        logs = [g for p in ids for g in glob.glob(os.path.join(os.path.dirname(p), "*.log"))]
        xtpc[mode] = parse_qlcathode(logs)
        print(f"  {mode}: cover={len(samp[mode]['cover'])} trans={len(samp[mode]['trans'])} "
              f"furth={len(samp[mode]['furth'])} xtpc={len(xtpc[mode])} slab_pts={int(occ[mode].sum())}")

    plot_coverage(occ)
    plot_xresidual(samp)
    hot = plot_transverse(samp)
    plot_xtpc(xtpc)
    plot_profiles(samp, xtpc)
    plot_furthest(samp)
    plot_surface_3d(samp)
    plot_closest(mind)

    for mode in ("data", "mc"):
        x = np.array(xtpc[mode]) if xtpc[mode] else np.empty((0, 6))
        if len(x):
            print(f"  {mode} xTPC |perp| median={np.median(x[:,4]):.2f} cm")
    # clean-signal hotspot: coarse (4x5) data xTPC |perp| max bin
    xd = np.array(xtpc["data"])
    if len(xd):
        ze = np.linspace(Z_LO, Z_HI, 6); ye = np.linspace(Y_LO, Y_HI, 5)
        s, _, _ = np.histogram2d(xd[:, 1], xd[:, 0], [ze, ye], weights=xd[:, 4])
        n, _, _ = np.histogram2d(xd[:, 1], xd[:, 0], [ze, ye])
        with np.errstate(invalid="ignore"):
            mm = np.where(n >= 4, s / n, np.nan)
        if np.isfinite(mm).any():
            iz, iy = np.unravel_index(np.nanargmax(mm), mm.shape)
            print(f"  HOTSPOT (xTPC data): max |perp|={mm[iz,iy]:.2f} cm at "
                  f"(Z={0.5*(ze[iz]+ze[iz+1]):.0f}, Y={0.5*(ye[iy]+ye[iy+1]):.0f}) cm")
    for apa, (z, y, dd) in sorted(hot.items()):
        print(f"  HOTSPOT (off-axis) TPC{apa}: max data residual {dd:.2f} cm at (Z={z:.0f}, Y={y:.0f}) cm")
    print(f"wrote 8 plots to {PICS}")


if __name__ == "__main__":
    main()

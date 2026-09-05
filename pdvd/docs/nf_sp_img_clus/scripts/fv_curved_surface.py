#!/usr/bin/env python3
"""Doc pdvd/41 sec 9 -- turn the calibrated surface of sec 5-7 into a fiducial volume.

Three products:

  1. the surface, refitted with a trapezoid (M3: a cathode-side PLATEAU of dc out to
     |x| = x1, then a linear ramp to 0 at |x| = x2).  M1 (the MicroBooNE / doc 41 sec 7
     shape) is M3 with x1 = the cathode face, so the two are nested and comparable by
     dchi2.  A trapezoid is used because a fiducial surface must not sit INSIDE the
     measured one by more than the cushion, and a single ramp forced through a
     saturating profile does exactly that on the top z- wall (10.4 cm measured at
     |x| = 50 vs 3.6 cm from M1);

  2. the polygons: ONE (x,y) polygon and ONE (z,x) polygon spanning BOTH drift volumes
     and joined across the cathode slab, so a cathode crosser is not an exiter -- the
     PolyFiducial{axis:2} + PolyFiducial{axis:1} + CompositeFiducial{logic:'and'} form,
     with an optional cushion (a pure transverse inset of the whole surface);

  3. the census: an offline replica of PolyFiducial::contained (the same pnpoly) and of
     FiducialUtils::inside_fiducial_volume's tolerance probe, run on the 120-event arm's
     points and PCA track ends, against today's box + margins.  This is a PROXY for the
     taggers (which test extreme / steiner boundary points, not PCA ends) -- it says
     which population moves and where, not what TGM will do.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_surface.py \
      /home/xqian/tmp/doc41/map20_result.json \
      --npz /home/xqian/tmp/doc41/points_d28dlfp.npz \
      --out /home/xqian/tmp/doc41/fv
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_map import (XW, YW, ZLO, ZHI, CATH, WALLS, Tracks, endpoint_table, m1)

# Today's PDVD tagger volume (pr.jsonnet pdvd_pr_fv + the driver's margins, doc 35)
BOX = dict(x=(-339.91, 339.91), y=(-336.4, 336.4), z=(0.05, 299.25))
BOX_MARGIN = dict(x=2.5, y=17.5, z=18.0)
# The proposed cushion, uBooNE's tagger regime (Cosmic_tagger.h stm/tgm tol vectors)
NEW_MARGIN = dict(x=2.5, y=3.0, z=3.0)

VOLS = [("bot", 0), ("top", 1)]


# ------------------------------------------------------------------ the surface
def m3(xabs, dc, x1, f):
    """Trapezoid: dc for |x| <= x1, linear to 0 at x2 = x1 + f*(XW - x1), 0 beyond.

    The ramp foot is parametrized as a FRACTION of the remaining drift so that x2 can
    never exceed the anode face: the displacement a drifting charge accumulates is zero
    at the anode by construction, and a fit that puts the foot beyond it (y- top, M3 at
    x2 = 476 cm with a free width) would give the anode plane a nonzero inset."""
    w = np.maximum(f * (XW - x1), 1e-6)
    return dc * np.clip((x1 + w - xabs) / w, 0, 1)


def fit_surface(xc, d50, err):
    """M1 and M3 on one (wall, volume) profile; the FV model choice."""
    m = np.isfinite(d50) & np.isfinite(err) & (err > 0)
    x, d, e = xc[m], d50[m], err[m]
    out = {}
    p, cov = curve_fit(m1, x, d, p0=[10, 150], sigma=e, absolute_sigma=True,
                       bounds=([-50, CATH + 5], [80, XW]), maxfev=40000)
    out["M1"] = dict(dc=float(p[0]), x1=CATH, x2=float(p[1]),
                     err=[float(v) for v in np.sqrt(np.diag(cov))],
                     chi2=float((((m1(x, *p) - d) / e) ** 2).sum()), ndf=int(len(x) - 2))
    try:
        p3, cov3 = curve_fit(m3, x, d, p0=[max(p[0], 1.0), 30.0, 0.5], sigma=e,
                             absolute_sigma=True, bounds=([-50, CATH, 0.02], [80, XW - 5, 1.0]), maxfev=40000)
        out["M3"] = dict(dc=float(p3[0]), x1=float(p3[1]), x2=float(p3[1] + p3[2] * (XW - p3[1])),
                         err=[float(v) for v in np.sqrt(np.diag(cov3))],
                         chi2=float((((m3(x, *p3) - d) / e) ** 2).sum()), ndf=int(len(x) - 3))
    except Exception as ex:
        out["M3"] = dict(error=str(ex))
    return out


def choose(fits, sigma_flat=2.0, dchi2=4.0):
    """FV model per profile: flat when the amplitude is under 2 sigma, else M1 unless the
    trapezoid buys more than dchi2 (one extra parameter)."""
    a, b = fits["M1"], fits.get("M3", {})
    if a["dc"] < sigma_flat * a["err"][0]:
        return dict(model="flat", dc=0.0, x1=CATH, x2=CATH, why="amplitude < 2 sigma")
    if "chi2" in b and a["chi2"] - b["chi2"] > dchi2 and b["dc"] > sigma_flat * b["err"][0]:
        return dict(model="M3", dc=b["dc"], x1=b["x1"], x2=b["x2"],
                    why=f"dchi2 {a['chi2'] - b['chi2']:.1f} > {dchi2}")
    return dict(model="M1", dc=a["dc"], x1=a["x1"], x2=a["x2"], why="single ramp adequate")


def inset(par, xabs):
    """The chosen surface's inset (cm) at |x|.  (x1, x2) -> m3's fractional foot."""
    x1, x2 = par["x1"], par["x2"]
    f = (x2 - x1) / max(XW - x1, 1e-9)
    return m3(np.asarray(xabs, float), par["dc"], x1, f)


# ------------------------------------------------------------------ the polygons
def wall_vertices(par):
    """(|x|, inset) knots of one wall of one volume, anode face -> cathode face."""
    knots = [(XW, float(inset(par, XW)))]
    if par["x2"] < XW:                      # the foot of the ramp, if it lands inside
        knots.append((par["x2"], 0.0))
    if CATH < par["x1"] < XW:               # the shoulder of the plateau (M3 only)
        knots.append((par["x1"], par["dc"]))
    knots.append((CATH, float(inset(par, CATH))))
    return knots


def build_polygons(sur, cushion_y=0.0, cushion_z=0.0, cushion_x=0.0):
    """One (x,y) polygon and one (x,z) polygon over BOTH drifts, joined across the cathode
    slab so a cathode crosser is not an exiter -- exactly what today's box does.

    The cushion is a pure transverse inset of the whole surface; in x it insets the two
    anode faces only.  Applying it here and applying it through the tagger's fv_tolerance
    probes give the same volume to within cos(ramp angle) = 0.997."""
    xa = XW - cushion_x

    def side_pts(wall, vol, wallpos, inward, cush):
        """(x, coord) knots on one wall of one volume, anode -> cathode face.
        `inward` = +1 when the interior is at larger coord (a low wall), -1 otherwise."""
        s = -1.0 if vol == "bot" else 1.0
        return [(s * min(xabs, xa), wallpos + inward * (d + cush))
                for xabs, d in wall_vertices(sur[wall][vol]["fv"])]

    ylo_bot = side_pts("y-", "bot", -YW, +1, cushion_y)
    ylo_top = side_pts("y-", "top", -YW, +1, cushion_y)
    yhi_bot = side_pts("y+", "bot", +YW, -1, cushion_y)
    yhi_top = side_pts("y+", "top", +YW, -1, cushion_y)
    xy = ylo_bot + ylo_top[::-1] + yhi_top + yhi_bot[::-1]

    zlo_bot = side_pts("z-", "bot", ZLO, +1, cushion_z)
    zlo_top = side_pts("z-", "top", ZLO, +1, cushion_z)
    zhi_bot = side_pts("z+", "bot", ZHI, -1, cushion_z)
    zhi_top = side_pts("z+", "top", ZHI, -1, cushion_z)
    xz = zlo_bot + zlo_top[::-1] + zhi_top + zhi_bot[::-1]

    def dedup(P):
        return [p for i, p in enumerate(P)
                if i == 0 or abs(p[0] - P[i - 1][0]) > 1e-9 or abs(p[1] - P[i - 1][1]) > 1e-9]

    return dedup(xy), dedup(xz)


# ------------------------------------------------------------------ the replica
def pnpoly(px, py, cx, cy):
    """PolyFiducial's is_inside (aux/src/PolyFiducial.cxx:43-56), vectorized."""
    px = np.asarray(px, float); py = np.asarray(py, float)
    inside = np.zeros(px.shape, bool)
    n = len(cx)
    j = n - 1
    for i in range(n):
        cond = ((cy[i] > py) != (cy[j] > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (cx[j] - cx[i]) * (py - cy[i]) / (cy[j] - cy[i]) + cx[i]
        inside ^= cond & (px < xint)
        j = i
    return inside


class CurvedFV:
    """CompositeFiducial{and} of PolyFiducial{axis:2}(x,y) and PolyFiducial{axis:1}(z,x)."""

    def __init__(self, xy, xz):
        self.xyx = np.array([p[0] for p in xy]); self.xyy = np.array([p[1] for p in xy])
        self.xzx = np.array([p[0] for p in xz]); self.xzz = np.array([p[1] for p in xz])

    def contained(self, x, y, z):
        return pnpoly(x, y, self.xyx, self.xyy) & pnpoly(x, z, self.xzx, self.xzz)


class BoxFV:
    def __init__(self, box=BOX):
        self.box = box

    def contained(self, x, y, z):
        b = self.box
        return ((x >= b["x"][0]) & (x <= b["x"][1]) & (y >= b["y"][0]) & (y <= b["y"][1])
                & (z >= b["z"][0]) & (z <= b["z"][1]))


def probe_fail(fv, x, y, z, mx, my, mz):
    """Which of FiducialUtils' seven tests fail, per point.  The axis decomposition doc 35
    sec 5.2 insists on: a margin verdict that is not split by axis lies about its cause."""
    base = ~fv.contained(x, y, z)
    out = {"base": base}
    for nm, (dx, dy, dz) in (("x_lo", (-mx, 0, 0)), ("x_hi", (mx, 0, 0)),
                             ("y_lo", (0, -my, 0)), ("y_hi", (0, my, 0)),
                             ("z_lo", (0, 0, -mz)), ("z_hi", (0, 0, mz))):
        # conditional on the point itself being contained, else a point outside the
        # envelope fails all six probes and the decomposition says nothing.
        out[nm] = (~fv.contained(x + dx, y + dy, z + dz)) & ~base
    return out


def inside_with_margin(fv, x, y, z, mx, my, mz):
    """FiducialUtils::inside_fiducial_volume(p, tol): the point shifted OUTWARD by the
    margin along each axis must still be contained (six probes)."""
    ok = fv.contained(x, y, z)
    for dx, dy, dz in ((mx, 0, 0), (-mx, 0, 0), (0, my, 0), (0, -my, 0), (0, 0, mz), (0, 0, -mz)):
        ok &= fv.contained(x + dx, y + dy, z + dz)
    return ok


# ------------------------------------------------------------------ the figure
def make_figure(sur, xc, cush, path):
    """The proposed volume: both polygons at true scale, then the four walls with the
    measurement, the surface, and the cushioned surface that IS the fiducial boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy0, xz0 = build_polygons(sur, 0.0, 0.0, 0.0)
    xyc, xzc = build_polygons(sur, cush, cush, 0.0)
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.05, 1.25], hspace=0.35, wspace=0.30)

    for col, (poly0, polyc, lo, hi, lab, wlo, whi) in enumerate((
            (xy0, xyc, -YW, YW, "y", -YW, YW), (xz0, xzc, ZLO, ZHI, "z", ZLO, ZHI))):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        ax.add_patch(plt.Rectangle((-XW, lo), 2 * XW, hi - lo, fill=False, ec="k", lw=1.2, label="sensitive volume"))
        m = 17.5 if lab == "y" else 18.0
        ax.add_patch(plt.Rectangle((-XW + 2.5, lo + m), 2 * (XW - 2.5), (hi - lo) - 2 * m,
                                   fill=False, ec="tab:brown", lw=1.4, ls="-.", label="today: box + 15 cm + margin"))
        ax.plot([p[0] for p in poly0] + [poly0[0][0]], [p[1] for p in poly0] + [poly0[0][1]],
                color="0.55", lw=1.0, ls=":", label="calibrated surface")
        ax.plot([p[0] for p in polyc] + [polyc[0][0]], [p[1] for p in polyc] + [polyc[0][1]],
                color="tab:blue", lw=1.8, label=f"proposed FV (+{cush:g} cm cushion)")
        ax.set_xlabel("x [cm]  (bottom drift < 0, top drift > 0)"); ax.set_ylabel(f"{lab} [cm]")
        ax.set_title(f"X-{lab.upper()} plane, true scale")
        ax.legend(fontsize=7, loc="center")
        ax.set_xlim(-XW - 15, XW + 15); ax.set_ylim(lo - 25, hi + 25)

    for i, wall in enumerate(WALLS):
        ax = fig.add_subplot(gs[1 + i // 2, 2 * (i % 2): 2 * (i % 2) + 2]) if False else fig.add_subplot(gs[1 + i // 4, i % 4])
        for vn, col in (("bot", "tab:blue"), ("top", "tab:red")):
            s = sur[wall][vn]
            ax.errorbar(xc, s["d50"], yerr=s["d50_err"], fmt="o", ms=3, lw=1, color=col, alpha=0.65,
                        label=f"d50 {vn}")
            xx = np.linspace(CATH, XW, 400)
            ax.plot(xx, inset(s["fv"], xx), color=col, lw=1.4, label=f"surface {vn} ({s['fv']['model']})")
            ax.plot(xx, inset(s["fv"], xx) + cush, color=col, lw=1.6, ls="--", label=f"FV {vn} (+{cush:g})")
        ax.axhline(17.5 if wall[0] == "y" else 18.0, color="tab:brown", ls="-.", lw=1.2, label="today")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{wall} wall"); ax.set_xlabel("|x| [cm]"); ax.set_ylabel("inset from the nominal wall [cm]")
        ax.set_ylim(-6, 26); ax.set_xlim(0, XW)
        if i == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("PDVD doc 41 sec 9 -- fiducial volume from the calibrated surface (120 Q/L-matched cosmic events)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="map20_result.json from fv_curved_map.py")
    ap.add_argument("--npz", default=None, help="points cache for the census")
    ap.add_argument("--out", default="/home/xqian/tmp/doc41/fv")
    ap.add_argument("--cushion", type=float, default=3.0, help="cushion used in the census (cm, y/z)")
    ap.add_argument("--nsample", type=int, default=2000000)
    ap.add_argument("--fig", default=None, help="write the summary figure here")
    a = ap.parse_args()

    R = json.load(open(a.result))
    xc = np.array(R["xcenter"], float)
    sur = {}
    for wall in WALLS:
        P = R["walls"][wall]["profile"]
        sur[wall] = {}
        for vn, vi in VOLS:
            d50 = np.array(P["d50"][vi], float); err = np.array(P["d50_err"][vi], float)
            fits = fit_surface(xc, d50, err)
            fv = choose(fits)
            # how far the chosen surface sits INSIDE the measurement, per bin
            resid = d50 - inset(fv, xc)           # >0 = the surface sits INSIDE the measurement
            ia = int(np.nanargmax(resid)); isig = int(np.nanargmax(resid / err))
            sur[wall][vn] = dict(fits=fits, fv=fv,
                                 d50=[float(v) for v in d50], d50_err=[float(v) for v in err],
                                 surface=[float(v) for v in inset(fv, xc)],
                                 max_deficit=float(resid[ia]), max_deficit_x=float(xc[ia]),
                                 max_deficit_sig_there=float(resid[ia] / err[ia]),
                                 max_sig=float(resid[isig] / err[isig]), max_sig_x=float(xc[isig]),
                                 max_sig_deficit=float(resid[isig]))

    out = dict(surface={w: {v: sur[w][v] for v, _ in VOLS} for w in WALLS})

    for cush in (0.0, a.cushion):
        xy, xz = build_polygons(sur, cushion_y=cush, cushion_z=cush, cushion_x=0.0)
        key = "polygons_cushion_%g" % cush
        out[key] = dict(xy=[[round(p[0], 3), round(p[1], 3)] for p in xy],
                        xz=[[round(p[0], 3), round(p[1], 3)] for p in xz],
                        # PolyFiducial corner order: axis=2 -> (x,y); axis=1 -> (z,x)
                        poly_axis2_corners=[[round(p[0], 3), round(p[1], 3)] for p in xy],
                        poly_axis1_corners=[[round(p[1], 3), round(p[0], 3)] for p in xz])

    xy0, xz0 = build_polygons(sur, 0.0, 0.0, 0.0)
    curved = CurvedFV(xy0, xz0)
    box = BoxFV()

    # --- transverse area / volume accounting, analytic on the two surfaces
    xs = np.linspace(-XW, XW, 1361)
    a_nom = (2 * YW) * (ZHI - ZLO)
    A_old, A_new = [], []
    # transverse area only: the x margin is 2.5 cm in BOTH schemes and cancels.
    for xv in xs:
        ylo = BOX["y"][0] + BOX_MARGIN["y"]; yhi = BOX["y"][1] - BOX_MARGIN["y"]
        zlo = BOX["z"][0] + BOX_MARGIN["z"]; zhi = BOX["z"][1] - BOX_MARGIN["z"]
        A_old.append((yhi - ylo) * (zhi - zlo))
        vn = "bot" if xv < 0 else "top"
        dy1 = inset(sur["y-"][vn]["fv"], abs(xv)) + a.cushion
        dy2 = inset(sur["y+"][vn]["fv"], abs(xv)) + a.cushion
        dz1 = inset(sur["z-"][vn]["fv"], abs(xv)) + a.cushion
        dz2 = inset(sur["z+"][vn]["fv"], abs(xv)) + a.cushion
        A_new.append(max(0.0, (2 * YW - dy1 - dy2)) * max(0.0, (ZHI - ZLO - dz1 - dz2)))
    A_old = np.array(A_old); A_new = np.array(A_new)
    out["area"] = dict(nominal_cm2=a_nom,
                       old_frac_mean=float(np.mean(A_old) / a_nom),
                       new_frac_mean=float(np.mean(A_new) / a_nom),
                       old_frac_at_cathode=float(A_old[np.argmin(np.abs(xs - 10))] / a_nom),
                       new_frac_at_cathode=float(A_new[np.argmin(np.abs(xs - 10))] / a_nom),
                       old_frac_at_anode=float(A_old[np.argmin(np.abs(xs - 300))] / a_nom),
                       new_frac_at_anode=float(A_new[np.argmin(np.abs(xs - 300))] / a_nom))

    # --- census on the arm
    if a.npz:
        T = Tracks(a.npz)
        rng = np.random.default_rng(41)
        n = len(T.x)
        idx = rng.choice(n, size=min(a.nsample, n), replace=False)
        x, y, z = T.x[idx], T.y[idx], T.z[idx]
        io = inside_with_margin(box, x, y, z, BOX_MARGIN["x"], BOX_MARGIN["y"], BOX_MARGIN["z"])
        inw = inside_with_margin(curved, x, y, z, NEW_MARGIN["x"], NEW_MARGIN["y"], NEW_MARGIN["z"])
        cath = np.abs(x) < 170
        out["points"] = dict(n=int(len(x)), n_total_used=int(n),
                             in_old=int(io.sum()), in_new=int(inw.sum()),
                             gain=int((inw & ~io).sum()), loss=int((io & ~inw).sum()),
                             gain_cathode=int((inw & ~io & cath).sum()),
                             gain_anode=int((inw & ~io & ~cath).sum()),
                             loss_cathode=int((io & ~inw & cath).sum()),
                             loss_anode=int((io & ~inw & ~cath).sum()))

        E = endpoint_table(T)
        ex, ey, ez = E["x"], E["y"], E["z"]
        eo = inside_with_margin(box, ex, ey, ez, BOX_MARGIN["x"], BOX_MARGIN["y"], BOX_MARGIN["z"])
        en = inside_with_margin(curved, ex, ey, ez, NEW_MARGIN["x"], NEW_MARGIN["y"], NEW_MARGIN["z"])
        # which wall is nearest (transverse only), and the drift half
        dists = np.vstack([YW - ey, YW + ey, ez - ZLO, ZHI - ez])
        near = np.array(WALLS)[np.argmin(dists, axis=0)]
        cath = np.abs(ex) < 170
        rows = {}
        for w in WALLS:
            for half, hm in (("cathode", cath), ("anode", ~cath)):
                m = (near == w) & hm
                rows[f"{w}|{half}"] = dict(n=int(m.sum()), out_old=int((~eo & m).sum()), out_new=int((~en & m).sum()),
                                           newly_out=int((eo & ~en & m).sum()), newly_in=int((~eo & en & m).sum()))
        env = BoxFV(dict(x=(-XW, XW), y=(-YW, YW), z=(ZLO, ZHI)))   # the sensvol envelope
        out["envelope_base_out"] = int((~env.contained(ex, ey, ez)).sum())
        po = probe_fail(box, ex, ey, ez, BOX_MARGIN["x"], BOX_MARGIN["y"], BOX_MARGIN["z"])
        pn = probe_fail(curved, ex, ey, ez, NEW_MARGIN["x"], NEW_MARGIN["y"], NEW_MARGIN["z"])
        good = ~E["atwin"]                                  # ends at the readout window: x suspect
        out["ends"] = dict(n=int(len(ex)), at_window=int(E["atwin"].sum()),
                           out_old=int((~eo).sum()), out_new=int((~en).sum()),
                           newly_out=int((eo & ~en).sum()), newly_in=int((~eo & en).sum()),
                           n_nowin=int(good.sum()),
                           out_old_nowin=int((~eo & good).sum()), out_new_nowin=int((~en & good).sum()),
                           newly_in_nowin=int((~eo & en & good).sum()),
                           newly_out_nowin=int((eo & ~en & good).sum()),
                           by_wall=rows,
                           by_probe_old={k: int(v.sum()) for k, v in po.items()},
                           by_probe_new={k: int(v.sum()) for k, v in pn.items()},
                           by_probe_old_nowin={k: int((v & good).sum()) for k, v in po.items()},
                           by_probe_new_nowin={k: int((v & good).sum()) for k, v in pn.items()})

    if a.fig:
        make_figure(sur, xc, a.cushion, a.fig)

    with open(a.out + "_surface.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out + "_surface.json")

    # human-readable summary
    print("\nwall vol  model     dc(cm)   x1     x2    chi2/ndf   max deficit vs d50")
    for wall in WALLS:
        for vn, _ in VOLS:
            s = sur[wall][vn]; f = s["fv"]; m = s["fits"]["M1" if f["model"] != "M3" else "M3"]
            print(f"{wall:3s} {vn:3s} {f['model']:5s} {f['dc']:8.2f} {f['x1']:6.1f} {f['x2']:6.1f}"
                  f"  {m['chi2']:6.1f}/{m['ndf']:<3d}  {s['max_deficit']:5.2f} cm at |x|={s['max_deficit_x']:3.0f}"
                  f" ({s['max_deficit_sig_there']:.1f}s) | worst {s['max_sig']:.1f}s ="
                  f" {s['max_sig_deficit']:5.2f} cm at |x|={s['max_sig_x']:3.0f}   [{f['why']}]")
    print("\nxy polygon (cushion 0):", out["polygons_cushion_0"]["xy"])
    print("\nxz polygon (cushion 0):", out["polygons_cushion_0"]["xz"])
    if "points" in out:
        p = out["points"]; e = out["ends"]
        print(f"\npoints {p['n']}: in_old {p['in_old']} in_new {p['in_new']} "
              f"gain {p['gain']} (cath {p['gain_cathode']} / anode {p['gain_anode']}) "
              f"loss {p['loss']} (cath {p['loss_cathode']} / anode {p['loss_anode']})")
        print(f"ends {e['n']}: out_old {e['out_old']} out_new {e['out_new']} "
              f"newly_out {e['newly_out']} newly_in {e['newly_in']}")
        print("  ends outside the bare envelope: box %d  sensvol %d  curved %d"
              % (e["by_probe_old"]["base"], out["envelope_base_out"], e["by_probe_new"]["base"]))
        print("  probe failures (all ends)   old:", e["by_probe_old"])
        print("  probe failures (all ends)   new:", e["by_probe_new"])
        print(f"  ends away from the readout window {e['n_nowin']}: out_old {e['out_old_nowin']} "
              f"out_new {e['out_new_nowin']} newly_in {e['newly_in_nowin']} newly_out {e['newly_out_nowin']}")
        print("  area kept vs nominal: old mean %.3f new mean %.3f | at |x|=10 old %.3f new %.3f | at |x|=300 old %.3f new %.3f"
              % (out["area"]["old_frac_mean"], out["area"]["new_frac_mean"],
                 out["area"]["old_frac_at_cathode"], out["area"]["new_frac_at_cathode"],
                 out["area"]["old_frac_at_anode"], out["area"]["new_frac_at_anode"]))
        for k, v in e["by_wall"].items():
            print(f"  {k:14s} n {v['n']:5d}  out_old {v['out_old']:5d}  out_new {v['out_new']:5d}"
                  f"  newly_out {v['newly_out']:4d}  newly_in {v['newly_in']:4d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit the WCT QLMatching SemiAnalyticalModel functional form to the PDVD v5
ANN visibility samples (work/ann_vis_v5_128nm.npz from sample_ann.py) and emit
a QLMatching-format semi-analytical-pdvd.json.

Model (exact numpy replica of match/src/SemiAnalyticalModel.cxx):
    vis = GH(d; pars(theta_bin) + s(theta)*r) * (Omega/4pi) * exp(-d/lambda) / cos
with lambda fixed to 2000 cm (the 20 m absorption length the v4 library/ANN
were generated with; Rayleigh migration is absorbed by GH, as in LArSoft).

PD groups and their status in the current C++ port:
  cathode  8 double-sided 60x60 XAs at x=0, orientation 0.  Supported EXCEPT
           the same-TPC x-sign gate must treat x=0 PDs as visible from both
           sides (fit uses both sides).
  pmt      24 8-inch PMTs (bottom wall x=-336.5 + 8 TCO-column), type 1 dome
           model, theta from the x axis.  Supported as-is.
  membrane 8 lateral 60x60 XAs on the y=+-417.6 walls.  NOT supported: the
           port fixes cosine=|dx|/d (orientation-0) which is wrong and
           divergent for y-normal PDs; the fit here uses the physically
           correct cosine=|dy|/d (what LArSoft's lateral branch does) and the
           resulting GH table is stored under an ignored "_..." key for a
           future lateral-branch port.

Outputs: semi-analytical-pdvd.json, pics/fit_*.png, fit summary on stdout.
Run in the TF venv or any python with numpy/scipy/matplotlib.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAMBDA = 2000.0          # cm, fixed (library built with 20 m AbsL)
DELTA_ANG = 10.0         # deg, GH angle bin width
NBINS = 9
PMT_RADIUS = 10.16       # cm (8-inch)
ACTIVE = dict(x=341.55, y=337.0, z_lo=0.0, z_hi=300.0)   # cm, WCT frame
CENTER_Y, CENTER_Z = 0.0, 150.0
VIS_FLOOR = 1e-5

# Omega_Dome_Model constants (SemiAnalyticalModel.cxx)
DOME_PAR0 = np.array([0., 0., 0., 0., 0., 0.597542, 1.00872, 1.46993, 2.04221])
DOME_PAR1 = np.array([0., 0., 0.19569, 0.300449, 0.555598, 0.854939, 1.39166, 2.19141, 2.57732])


def rect_solid_angle_scalar(a, b, d):
    """Rectangle_SolidAngle(a, b, d): on-axis rectangle a x b at normal distance d."""
    aa = a * 0.5 / d
    bb = b * 0.5 / d
    aux = (1. + aa * aa + bb * bb)
    return 4. * np.arccos(np.sqrt(aux / ((1. + aa * aa) * (1. + bb * bb))))


def rect_solid_angle(h, w, d1, vz, d2):
    """Off-axis rectangle solid angle, vectorized port of the 4-case C++ decomposition.
    d1: |offset along the h axis|, vz: |offset along the w axis|, d2: normal distance."""
    d2 = np.maximum(d2, 1e-9)
    A = np.abs(d1 - h * .5)
    B = np.abs(vz - w * .5)
    hi1 = d1 > h * .5
    hi2 = vz > w * .5
    r = np.empty_like(d2)
    m = hi1 & hi2
    r[m] = (rect_solid_angle_scalar(2*(A[m]+h), 2*(B[m]+w), d2[m])
            - rect_solid_angle_scalar(2*A[m], 2*(B[m]+w), d2[m])
            - rect_solid_angle_scalar(2*(A[m]+h), 2*B[m], d2[m])
            + rect_solid_angle_scalar(2*A[m], 2*B[m], d2[m])) * .25
    m = ~hi1 & ~hi2
    r[m] = (rect_solid_angle_scalar(2*(h-A[m]), 2*(w-B[m]), d2[m])
            + rect_solid_angle_scalar(2*A[m], 2*(w-B[m]), d2[m])
            + rect_solid_angle_scalar(2*(h-A[m]), 2*B[m], d2[m])
            + rect_solid_angle_scalar(2*A[m], 2*B[m], d2[m])) * .25
    m = hi1 & ~hi2
    r[m] = (rect_solid_angle_scalar(2*(A[m]+h), 2*(w-B[m]), d2[m])
            - rect_solid_angle_scalar(2*A[m], 2*(w-B[m]), d2[m])
            + rect_solid_angle_scalar(2*(A[m]+h), 2*B[m], d2[m])
            - rect_solid_angle_scalar(2*A[m], 2*B[m], d2[m])) * .25
    m = ~hi1 & hi2
    r[m] = (rect_solid_angle_scalar(2*(h-A[m]), 2*(B[m]+w), d2[m])
            - rect_solid_angle_scalar(2*(h-A[m]), 2*B[m], d2[m])
            + rect_solid_angle_scalar(2*A[m], 2*(B[m]+w), d2[m])
            - rect_solid_angle_scalar(2*A[m], 2*B[m], d2[m])) * .25
    return r


def dome_solid_angle(distance, theta_deg):
    j = np.clip((theta_deg / DELTA_ANG).astype(int), 0, NBINS - 1)
    R = np.where(distance >= 5. * PMT_RADIUS,
                 PMT_RADIUS - DOME_PAR1[j], PMT_RADIUS - DOME_PAR0[j])
    ratio_sq = np.clip((R * R) / (distance * distance), 0., 1.)
    return 2. * np.pi * (1. - np.sqrt(1. - ratio_sq))


def gaisser_hillas(x, norm, xmu, width, x0):
    diff = xmu - x0
    term = np.power(np.clip((x - x0) / diff, 1e-12, None), diff / width)
    return norm * term * np.exp((xmu - x) / width)


def load_samples(tag="v5_128nm"):
    d = np.load(os.path.join(HERE, "work", f"ann_vis_{tag}.npz"), allow_pickle=True)
    org, st, n = d["origin_cm"], d["step_cm"], d["n"]
    ax = [org[i] + st[i] * np.arange(n[i]) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    vis = d["vis"].reshape(-1, 40)
    act = ((np.abs(g[:, 0]) <= ACTIVE["x"]) & (np.abs(g[:, 1]) <= ACTIVE["y"])
           & (g[:, 2] >= ACTIVE["z_lo"]) & (g[:, 2] <= ACTIVE["z_hi"]))
    return g[act], vis[act], d


def group_features(pts, vis, meta):
    """Per (point, channel) rows for each PD group: d, theta, omega, r, vis."""
    nodes = [str(s) for s in meta["chan_node"]]
    pos = meta["chan_pos_mm"] / 10.0   # cm
    wct = meta["wct_opdet"]
    r_pt = np.hypot(pts[:, 1] - CENTER_Y, pts[:, 2] - CENTER_Z)
    groups = {}
    for gname, sel, orient in (
            ("cathode", lambda nd, ch: "Double" in nd, 0),
            ("membrane", lambda nd, ch: "XARAPUCAWindow" in nd, 1),
            ("pmt_bottom", lambda nd, ch: "pmt" in nd and pos[ch][0] < -330., None),
            ("pmt_tco", lambda nd, ch: "pmt" in nd and pos[ch][0] >= -330., None)):
        rows = []
        for ch in range(40):
            if not sel(nodes[ch], ch) or wct[ch] < 0:
                continue
            rel = pts - pos[ch]
            d = np.linalg.norm(rel, axis=1)
            if gname == "cathode":
                mask = np.ones(len(pts), bool)              # double-sided
                cos = np.abs(rel[:, 0]) / d
                omega = rect_solid_angle(60., 60., np.abs(rel[:, 1]),
                                         np.abs(rel[:, 2]), np.abs(rel[:, 0]))
            elif gname == "membrane":
                mask = (pts[:, 0] * pos[ch][0]) > 0          # same drift side
                cos = np.abs(rel[:, 1]) / d                  # y-normal (lateral branch)
                omega = rect_solid_angle(60., 60., np.abs(rel[:, 0]),
                                         np.abs(rel[:, 2]), np.abs(rel[:, 1]))
            else:
                mask = pts[:, 0] < 0                         # PMTs all at x<0
                cos = np.abs(rel[:, 0]) / d
                theta = np.degrees(np.arccos(np.clip(cos, 0, 1)))
                omega = dome_solid_angle(d, theta)
            theta = np.degrees(np.arccos(np.clip(cos, 0, 1)))
            v = vis[:, ch]
            m = mask & (v > VIS_FLOOR) & (omega > 0) & (d > 1.)
            rows.append(dict(ch=ch, d=d[m], theta=theta[m], omega=omega[m],
                             r=r_pt[m], vis=v[m], cos=cos[m]))
        groups[gname] = rows
    return groups


def fit_group(rows, p0_bins, fit_border=True):
    """Fit GH pars per 10-degree theta bin + linear-in-r Norm border correction."""
    from scipy.optimize import curve_fit
    d = np.concatenate([r["d"] for r in rows])
    th = np.concatenate([r["theta"] for r in rows])
    om = np.concatenate([r["omega"] for r in rows])
    rr = np.concatenate([r["r"] for r in rows])
    vv = np.concatenate([r["vis"] for r in rows])
    cc = np.concatenate([r["cos"] for r in rows])
    # GH target: vis * cos * 4pi / (omega * exp(-d/lambda))
    y = vv * cc * 4. * np.pi / (om * np.exp(-d / LAMBDA))
    jbin = np.clip((th / DELTA_ANG).astype(int), 0, NBINS - 1)

    pars = np.zeros((NBINS, 4))
    border = np.zeros(NBINS)
    ok = np.zeros(NBINS, bool)
    for j in range(NBINS):
        m = jbin == j
        if m.sum() < 50:
            continue
        db = (d[m] // 20).astype(int)
        ub = np.unique(db)
        xd = np.array([20. * u + 10. for u in ub])
        yd = np.array([np.median(y[m][db == u]) for u in ub])
        wd = np.array([np.sqrt((db == u).sum()) for u in ub])
        p0 = p0_bins[j]
        try:
            popt, _ = curve_fit(
                gaisser_hillas, xd, yd, p0=p0, sigma=1. / wd, maxfev=20000,
                bounds=([0., -500., 1., -2000.], [50., 2000., 5000., xd.min() - 1.]))
            pars[j] = popt
            ok[j] = True
        except Exception as e:
            print(f"    bin {j}: GH fit failed ({e}); will copy neighbor")
        if ok[j] and fit_border:
            gh = gaisser_hillas(d[m], *pars[j])
            res = y[m] - gh
            # Norm border correction: pars[0] += s1*r  =>  res ~ (gh/norm)*s1*r
            basis = gh / pars[j][0] * rr[m]
            denom = (basis ** 2).sum()
            if denom > 0:
                border[j] = (basis * res).sum() / denom
    # fill failed bins from nearest fitted neighbor
    for j in range(NBINS):
        if not ok[j]:
            k = min((jj for jj in range(NBINS) if ok[jj]),
                    key=lambda jj: abs(jj - j), default=None)
            if k is None:
                raise SystemExit("no theta bin fitted")
            pars[j], border[j] = pars[k], border[k]
    return pars, border, dict(d=d, theta=th, omega=om, r=rr, vis=vv, cos=cc, jbin=jbin)


def predict(feat, pars, border):
    p = pars[feat["jbin"]].copy()
    p[:, 0] = p[:, 0] + border[feat["jbin"]] * feat["r"]
    gh = gaisser_hillas(feat["d"], p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    gh = np.where(np.isfinite(gh) & (gh >= 0) & (gh <= 10), gh, 0.)
    return gh * (feat["omega"] / (4. * np.pi)) * np.exp(-feat["d"] / LAMBDA) / feat["cos"]


def report(gname, feat, pred):
    m = pred > 0
    rat = pred[m] / feat["vis"][m]
    print(f"  {gname}: {m.sum()} samples, pred/ANN ratio "
          f"16/50/84% = {np.percentile(rat,16):.3f}/{np.percentile(rat,50):.3f}/"
          f"{np.percentile(rat,84):.3f}")
    lines = []
    for lo in range(0, 700, 100):
        mm = m & (feat["d"] >= lo) & (feat["d"] < lo + 100)
        if mm.sum() < 20:
            continue
        r2 = pred[mm] / feat["vis"][mm]
        lines.append((lo, mm.sum(), *np.percentile(r2, [16, 50, 84])))
        print(f"    d {lo:3d}-{lo+100:3d} cm: n={mm.sum():7d} "
              f"ratio {lines[-1][2]:.3f}/{lines[-1][3]:.3f}/{lines[-1][4]:.3f}")
    return lines


def plot_group(gname, feat, pred, pars, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    y = feat["vis"] * feat["cos"] * 4 * np.pi / (feat["omega"] * np.exp(-feat["d"] / LAMBDA))
    for j in (0, 3, 6):
        m = feat["jbin"] == j
        if m.sum() < 50:
            continue
        db = (feat["d"][m] // 20).astype(int)
        ub = np.unique(db)
        xd = 20. * ub + 10.
        yd = [np.median(y[m][db == u]) for u in ub]
        l, = axes[0].plot(xd, yd, "o", ms=3, label=f"theta bin {j}")
        xs = np.linspace(max(5, xd.min()), xd.max(), 200)
        axes[0].plot(xs, gaisser_hillas(xs, *pars[j]), "-", color=l.get_color())
    axes[0].set(xlabel="distance [cm]", ylabel="GH target", title=f"{gname}: GH fit")
    axes[0].legend()
    m = pred > 0
    rat = np.clip(pred[m] / feat["vis"][m], 1e-2, 1e2)
    axes[1].hist(np.log10(rat), bins=100, range=(-1, 1))
    axes[1].set(xlabel="log10(pred/ANN)", title=f"{gname}: ratio")
    axes[2].hexbin(feat["d"][m], np.log10(rat), gridsize=60, extent=(0, 700, -1, 1),
                   bins="log")
    axes[2].set(xlabel="distance [cm]", ylabel="log10(pred/ANN)", title=f"{gname}: vs d")
    fig.tight_layout()
    fn = os.path.join(outdir, f"fit_{gname}.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print(f"    plot -> {fn}")


def main():
    pts, vis, meta = load_samples()
    print(f"active-volume sample points: {len(pts)}")
    groups = group_features(pts, vis, meta)

    pdhd = json.load(open(
        "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json"))
    gh_flat_pdhd = np.array(pdhd["VUVHits"]["GH_PARS_flat"]).T   # [9,4]
    p0_dome = np.tile([1.2, 150., 200., -100.], (NBINS, 1))

    outdir = os.path.join(HERE, "pics")
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for gname, p0 in (("cathode", gh_flat_pdhd), ("membrane", gh_flat_pdhd),
                      ("pmt_bottom", p0_dome), ("pmt_tco", p0_dome)):
        print(f"== {gname}")
        pars, border, feat = fit_group(groups[gname], p0)
        pred = predict(feat, pars, border)
        report(gname, feat, pred)
        plot_group(gname, feat, pred, pars, outdir)
        # per-channel suggested efficiency scale (ANN/pred median per channel)
        chscale = {}
        for r in groups[gname]:
            rel = None
            i0 = 0
            # recompute per-channel pred using the group's pars
            jb = np.clip((r["theta"] / DELTA_ANG).astype(int), 0, NBINS - 1)
            p = pars[jb].copy()
            p[:, 0] = p[:, 0] + border[jb] * r["r"]
            gh = gaisser_hillas(r["d"], p[:, 0], p[:, 1], p[:, 2], p[:, 3])
            gh = np.where(np.isfinite(gh) & (gh >= 0) & (gh <= 10), gh, 0.)
            pr = gh * (r["omega"] / (4 * np.pi)) * np.exp(-r["d"] / LAMBDA) / r["cos"]
            mm = pr > 0
            chscale[int(r["ch"])] = float(np.median(r["vis"][mm] / pr[mm]))
        results[gname] = dict(pars=pars, border=border, chscale=chscale)
        print(f"    per-channel ANN/pred medians: "
              + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(chscale.items())))

    # ---- emit semi-analytical-pdvd.json ------------------------------------
    nodes = [str(s) for s in meta["chan_node"]]
    pos = meta["chan_pos_mm"] / 10.0
    opdets = []
    for ch in range(40):
        if "Double" in nodes[ch]:
            typ, orient, h, w = 0, 0, 60.0, 60.0
        elif "XARAPUCAWindow" in nodes[ch]:
            typ, orient, h, w = 0, 1, 60.0, 60.0
        else:
            typ, orient, h, w = 1, 0, -1.0, -1.0
        opdets.append(dict(x=round(float(pos[ch][0]), 3), y=round(float(pos[ch][1]), 3),
                           z=round(float(pos[ch][2]), 3), h=h, w=w,
                           type=typ, orientation=orient))
    angulo = [5., 15., 25., 35., 45., 55., 65., 75., 85.]
    out = {
        "_comment": "PDVD semi-analytical optical model FITTED to the official v5 PDFastSimANN computable graph (protodune_vd_v5_128nm_tf2.6) sampled on a 10cm active-volume grid; see pdvd/photlib/fit_semi_analytical.py and pdvd/docs/pdvd-photon-model.md. lambda fixed at 2000cm (the 20m absorption length of the underlying G4 sim); Rayleigh migration absorbed in GH, as in LArSoft. OpDets ordered by v5 ANN channel == WCT flash-chain OpDet (identity; dead 24/27/28/34 kept, mask in cfg). GH_PARS_flat = cathode double-sided XAs (orientation 0). GH_PARS_dome = the 16 bottom-wall (x=-336.5) PMTs. CAVEATS: (1) cathode XAs sit at x=0 and are visible from BOTH drift sides -- the same-TPC x-sign gate must exempt them; (2) membrane lateral XAs (orientation 1) are NOT usable with the current cosine=|dx|/d port -- their correct-physics (cosine=|dy|/d) GH table is stored under _GH_PARS_membrane_lateral for a future lateral-branch port; mask channels 0-3/16-17/22-23 in semi mode until then; (3) the 8 TCO-column PMTs (ch 12-15/18-21) fit the x-axis dome model poorly, table under _GH_PARS_pmt_tco -- consider masking in semi mode.",
        "_provenance": "fit_semi_analytical.py on ann_vis_v5_128nm.npz (2026-07-09)",
        "VUVHits": {
            "FlatPDCorr": True,
            "DomePDCorr": True,
            "delta_angulo_vuv": DELTA_ANG,
            "PMT_radius": PMT_RADIUS,
            "MaxPDDistance": 1000.0,
            "GH_PARS_flat": results["cathode"]["pars"].T.tolist(),
            "GH_border_angulo_flat": angulo,
            "GH_border_flat": [results["cathode"]["border"].tolist(),
                               [0.0] * NBINS, [0.0] * NBINS],
            "GH_PARS_dome": results["pmt_bottom"]["pars"].T.tolist(),
            "GH_border_angulo_dome": angulo,
            "GH_border_dome": [results["pmt_bottom"]["border"].tolist(),
                               [0.0] * NBINS, [0.0] * NBINS],
        },
        "_GH_PARS_membrane_lateral": results["membrane"]["pars"].T.tolist(),
        "_GH_border_membrane_lateral_norm": results["membrane"]["border"].tolist(),
        "_GH_PARS_pmt_tco": results["pmt_tco"]["pars"].T.tolist(),
        "_GH_border_pmt_tco_norm": results["pmt_tco"]["border"].tolist(),
        "_per_channel_ann_over_pred": {g: results[g]["chscale"] for g in results},
        "VISHits": {},
        "Geometry": {
            "active_center_y": CENTER_Y,
            "active_size_y": 2 * ACTIVE["y"],
            "active_center_z": CENTER_Z,
            "active_size_z": ACTIVE["z_hi"] - ACTIVE["z_lo"],
            "cathode_x": 0.0,
            "vuv_absorption_length": LAMBDA,
        },
        "OpDets": opdets,
    }
    fn = os.path.join(HERE, "semi-analytical-pdvd.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {fn}")

    # round-trip through the exact key paths QLMatching.cxx reads
    t = json.load(open(fn))
    assert isinstance(t["VUVHits"], dict) and isinstance(t["Geometry"], dict)
    assert isinstance(t["OpDets"], list) and len(t["OpDets"]) == 40
    for k in ("FlatPDCorr", "DomePDCorr", "delta_angulo_vuv", "GH_PARS_flat",
              "GH_border_angulo_flat", "GH_border_flat", "GH_PARS_dome",
              "GH_border_angulo_dome", "GH_border_dome"):
        assert k in t["VUVHits"], k
    for k in ("active_center_y", "active_center_z", "cathode_x", "vuv_absorption_length"):
        assert k in t["Geometry"], k
    for od in t["OpDets"]:
        for k in ("x", "y", "z", "h", "w", "type", "orientation"):
            assert k in od, k
    assert np.array(t["VUVHits"]["GH_PARS_flat"]).shape == (4, NBINS)
    print("round-trip key check: OK")


if __name__ == "__main__":
    main()

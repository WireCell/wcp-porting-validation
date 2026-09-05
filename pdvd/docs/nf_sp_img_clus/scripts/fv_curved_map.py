#!/usr/bin/env python3
"""Doc pdvd/41 -- stage 2: the apparent side-wall position as a function of drift x,
from the cached t0-corrected points (fv_curved_load.py), for the four PDVD side walls
(y+, y-, z-, z+), both drift volumes; symmetry + factorization tests; uBooNE-style
(M1, flat + linear ramp) and smooth (M2, power law) surface fits; ready-to-use polygon
arrays; and an independent track-endpoint cross-check.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_map.py /home/xqian/tmp/doc41/points_d28dlfp.npz \
      --out /home/xqian/tmp/doc41/map --boot 200

Instrument (primary).  For a wall W and a drift bin B, the imaged points of all
flash-matched tracks within 60 cm of the nominal wall are histogrammed in their wall
distance d (0.5 cm bins, per event so the bootstrap is a reweighting).  Two shape-free
estimators of the apparent wall position are read off the cumulative C(d):
  d_eq  the equivalent missing length: a straight line fitted to C(d) over the far
        plateau d in [30, 60] and extrapolated to C = 0 -- the amount of charge missing
        near the wall expressed as a length at plateau density (an integral: comb-free
        at any bin width, doc 34; it is what a fiducial inset means for charge accounting);
  d50   the half-density point: where the 2 cm-binned density first reaches half the
        plateau (the shape check: soft edge vs sharp edge).
A third, per-TRACK number is kept alongside: the 2nd-smallest minimum wall distance
over the tracks crossing the bin (the onset of tracks; model-free, no error).
Errors: event-level bootstrap.  Control: the anode-side bins must return ~0.

Frames: x is the Bee/t0-corrected coordinate (cm); xraw the raw readout x (cm), whose
window edges are +-398.52 (late) and -+341.55 (tick 0) per side; side = anode group.
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import curve_fit

XW, YW, ZLO, ZHI = 339.91, 336.39, 0.813, 298.435        # sensvol envelope, cm
CATH = 3.0
YSEAM, ZSEAM = 168.50, 149.65
XMAX_OK = 345.0            # a track-side with any |x| beyond this has a wrong t0 (window/mismatch)
DMAX = 60.0                # wall-distance window of the sample
DLO = -5.0                 # histogram start (points beyond the nominal wall)
DB = 0.5                   # histogram bin, cm
NDB = int(round((DMAX - DLO) / DB))
PLATEAU = 30.0             # plateau fit range [PLATEAU, DMAX]
MINPTS = 3                 # points a track needs inside a drift bin to count as crossing it
RAW_LATE, RAW_EARLY, RAW_EDGE_TOL = 398.52, 341.55, 4.4   # 60 ticks x 0.5 us x 0.148 cm/us

XEDGES = np.array([3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                   190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340], float)

WALLS = ["y+", "y-", "z-", "z+"]


def wall_dist(w, y, z):
    return {"y+": YW - y, "y-": YW + y, "z-": z - ZLO, "z+": ZHI - z}[w]


def wall_other_ok(w, y, z):
    """Away from the other walls and the seams, so the sample is one wall's."""
    if w[0] == "y":
        return (z > ZLO + 15) & (z < ZHI - 15) & (np.abs(z - ZSEAM) > 3)
    return (np.abs(y) < YW - 15) & (np.abs(y) > 3) & (np.abs(np.abs(y) - YSEAM) > 3)


def other_slices(w, y, z):
    """The factorization slices: y walls split in z thirds, z walls in the y CRU quarters."""
    if w[0] == "y":
        return {"z<100": z < 100, "100<z<200": (z >= 100) & (z < 200), "z>200": z >= 200}
    return {"y<-168": y < -YSEAM, "-168<y<0": (y >= -YSEAM) & (y < 0),
            "0<y<168": (y >= 0) & (y < YSEAM), "y>168": y >= YSEAM}


# ------------------------------------------------------------------ samples
class Tracks:
    """Per-(track, drift-bin, wall) minimum wall distance, with the event id for bootstrap."""

    def __init__(self, npz, xedges=XEDGES):
        d = np.load(npz, allow_pickle=True)
        self.ev_run, self.ev_evt = d["ev_run"], d["ev_evt"]
        self.nev = len(self.ev_run)
        x, y, z, xr, side, cid, ev, ph = (d[k] for k in ("x", "y", "z", "xraw", "side", "cid", "ev", "phys"))
        self.report = {"points_total": int(len(x)), "points_sentinel": int((~ph).sum()),
                       "clusters_total": int(d["ev_nclus"].sum()), "clusters_matched": int(d["ev_nmatched"].sum())}
        m = ph
        x, y, z, xr, side, cid, ev = (a[m] for a in (x, y, z, xr, side, cid, ev))
        key = (ev.astype(np.int64) * 100000 + cid) * 2 + (side > 0)          # track = cluster x anode side
        # track-level cut: any |x| beyond the anode + 5 cm => wrong t0
        ukey, inv = np.unique(key, return_inverse=True)
        maxabs = np.zeros(len(ukey)); np.maximum.at(maxabs, inv, np.abs(x))
        bad = maxabs[inv] > XMAX_OK
        self.report.update(tracks_total=int(len(ukey)), tracks_badx=int((maxabs > XMAX_OK).sum()),
                           points_badx=int(bad.sum()), points_used=int((~bad).sum()))
        m = ~bad
        self.x, self.y, self.z, self.xr, self.side, self.key, self.ev = (a[m] for a in (x, y, z, xr, side, key, ev))
        self.xedges = xedges
        # drift-bin index, signed by volume: bins 0..nb-1 for x>0 (top), same for x<0 (bottom)
        self.nb = len(xedges) - 1
        self.ib = np.searchsorted(xedges, np.abs(self.x), side="right") - 1
        self.ib[(np.abs(self.x) < xedges[0]) | (np.abs(self.x) >= xedges[-1])] = -1
        self.vol = np.where(self.x < 0, 0, 1)      # 0 bottom, 1 top (by corrected x; side agrees except crossers)

    def dmin_table(self, wall, extra=None):
        """rows (track key, event, vol, bin, dmin, npts) for tracks near `wall`."""
        dd = wall_dist(wall, self.y, self.z)
        sel = (dd < DMAX) & (dd > -10) & wall_other_ok(wall, self.y, self.z) & (self.ib >= 0)
        if extra is not None:
            sel &= extra
        k = (self.key[sel] * 2 + self.vol[sel]) * 64 + self.ib[sel]          # (track, vol, bin)
        order = np.lexsort((dd[sel], k))
        ks, ds = k[order], dd[sel][order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        cnt = np.diff(np.r_[np.flatnonzero(first), len(ks)])
        kk = ks[first]
        return dict(key=kk // 128, vol=(kk // 64) % 2, ib=kk % 64, dmin=ds[first], npts=cnt,
                    ev=self.ev[sel][order][first])


def wall_hists(T, wall, extra=None):
    """Per (vol, bin, event) histogram of point wall-distance: array [2, nb, nev, NDB]."""
    dd = wall_dist(wall, T.y, T.z)
    sel = (dd > DLO) & (dd < DMAX) & wall_other_ok(wall, T.y, T.z) & (T.ib >= 0)
    if extra is not None:
        sel &= extra
    k = np.floor((dd[sel] - DLO) / DB).astype(np.int64)
    idx = ((T.vol[sel] * T.nb + T.ib[sel]) * T.nev + T.ev[sel]) * NDB + k
    H = np.bincount(idx, minlength=2 * T.nb * T.nev * NDB).reshape(2, T.nb, T.nev, NDB)
    return H


DGRID = DLO + DB * (np.arange(NDB) + 1)          # upper edge of each histogram bin
PLAT = DGRID >= PLATEAU


def estimators(h):
    """h: [..., NDB] histograms -> (d_eq, d50, rho) arrays over the leading axes."""
    C = np.cumsum(h, axis=-1)
    dg = DGRID[PLAT]
    y = C[..., PLAT]
    n = PLAT.sum()
    sx, sy = dg.sum(), y.sum(-1)
    sxx, sxy = (dg * dg).sum(), (y * dg).sum(-1)
    rho = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - rho * sx) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        d_eq = -b / rho
        # half-density: 2 cm bins (4 x 0.5 cm), density per cm / rho
        nb2 = NDB // 4
        dens = h[..., :nb2 * 4].reshape(h.shape[:-1] + (nb2, 4)).sum(-1) / (4 * DB)
        frac = dens / rho[..., None]
        lo = DLO + 4 * DB * np.arange(nb2)              # lower edge of each 2 cm bin
        d50 = np.full(rho.shape, np.nan)
        it = np.ndindex(*rho.shape)
        for ix in it:
            f = frac[ix]
            ok = np.flatnonzero(f >= 0.5)
            if len(ok) == 0 or not np.isfinite(rho[ix]) or rho[ix] <= 0:
                continue
            j = ok[0]
            if j == 0:
                d50[ix] = lo[0]
            else:   # centre-to-centre interpolation between 2 cm bins j-1 and j
                f0, f1 = f[j - 1], f[j]
                d50[ix] = (lo[j - 1] + 2 * DB) + 4 * DB * (0.5 - f0) / (f1 - f0)
    d_eq[~np.isfinite(d_eq) | (rho <= 0)] = np.nan
    return d_eq, d50, rho


def edge_profile(H, boot=200, rng=None):
    """From [2, nb, nev, NDB] histograms: d0 (= d_eq) +- bootstrap, d50 +- bootstrap, rho, n points."""
    rng = rng or np.random.default_rng(41)
    nev = H.shape[2]
    h = H.sum(2)
    d_eq, d50, rho = estimators(h)
    out = {"d0": d_eq, "d50": d50, "rho": rho, "n": h.sum(-1),
           "err": np.full(d_eq.shape, np.nan), "d50_err": np.full(d_eq.shape, np.nan)}
    if boot > 0:
        be, b5 = [], []
        for _ in range(boot):
            w = np.bincount(rng.integers(0, nev, nev), minlength=nev).astype(float)
            hb = np.tensordot(H, w, axes=([2], [0]))
            e, f, _ = estimators(hb)
            be.append(e); b5.append(f)
        be, b5 = np.array(be), np.array(b5)
        out["err"] = np.nanstd(be, axis=0); out["d50_err"] = np.nanstd(b5, axis=0)
        out["err"][np.isnan(be).mean(0) > 0.2] = np.nan
    out["n"] = out["n"].astype(int)
    return out


def onset_profile(tab, nb, minpts=MINPTS):
    """Per (vol, bin): the 2nd-smallest per-track minimum wall distance, and the track count."""
    good = tab["npts"] >= minpts
    d2 = np.full((2, nb), np.nan); n = np.zeros((2, nb), int)
    for v in (0, 1):
        for b in range(nb):
            m = good & (tab["vol"] == v) & (tab["ib"] == b)
            n[v, b] = m.sum()
            if n[v, b] >= 2:
                d2[v, b] = np.sort(tab["dmin"][m])[1]
    return d2, n


def chi2_compare(a, ea, b, eb):
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(ea) & np.isfinite(eb) & (ea > 0) & (eb > 0)
    if m.sum() == 0:
        return np.nan, 0, np.nan
    c2 = float((((a - b) ** 2) / (ea ** 2 + eb ** 2))[m].sum())
    return c2, int(m.sum()), float(np.average((a - b)[m], weights=1 / (ea ** 2 + eb ** 2)[m]))


# ------------------------------------------------------------------ surface models
def m1(xabs, dc, xk):                         # uBooNE form: flat, then linear ramp to the cathode
    return dc * np.clip((xk - xabs) / (xk - CATH), 0, 1)


def m2(xabs, dc, p):                          # smooth power law, 0 at the anode
    return dc * np.clip((XW - xabs) / (XW - CATH), 0, 1) ** p


def fit_models(xc, d0, err):
    m = np.isfinite(d0) & np.isfinite(err) & (err > 0)
    res = {}
    for name, f, p0, bounds in (("M1", m1, [10, 150], ([-50, CATH + 5], [80, XW])),
                                ("M2", m2, [10, 2], ([-50, 0.2], [80, 10]))):
        try:
            p, cov = curve_fit(f, xc[m], d0[m], p0=p0, sigma=err[m], absolute_sigma=True, bounds=bounds, maxfev=20000)
            c2 = float((((f(xc[m], *p) - d0[m]) / err[m]) ** 2).sum())
            res[name] = dict(params=[float(v) for v in p], perr=[float(v) for v in np.sqrt(np.diag(cov))],
                             chi2=c2, ndf=int(m.sum() - 2))
        except Exception as e:
            res[name] = dict(error=str(e))
    return res


# ------------------------------------------------------------------ track bending
BANDS = [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)]


def bend_profile(T, wall, xedges, xref=200.0, minref=15, rms_max=2.0):
    """Displacement vs loss, without survivorship.  For every long track a straight line
    d = a + b*x is fitted to its wall distance on the ANODE side (|x| >= xref) and
    extrapolated toward the cathode.  In each drift bin the track's line passes through
    (and that the track demonstrably continues past, i.e. it has points closer to the
    cathode than the bin), we record the predicted distance band, whether the track has
    >= 3 points in the bin (survival), and if so the mean residual d - d_pred (a positive
    residual = pushed inward).  Only lines that stay >= 3 cm inside the wall over the
    track's whole x range are used, so appearing in a bin is not conditional on being
    displaced.  Returns arrays [2, nb, nbands]: n_total, n_has, resid, resid_err."""
    nb = len(xedges) - 1
    xc = 0.5 * (xedges[1:] + xedges[:-1])
    dd_all = wall_dist(wall, T.y, T.z)
    okw = wall_other_ok(wall, T.y, T.z)
    order = np.argsort(T.key, kind="stable")
    keys = T.key[order]
    first = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.flatnonzero(first); stops = np.r_[starts[1:], len(keys)]
    nbd = len(BANDS)
    ntot = np.zeros((2, nb, nbd), int); nhas = np.zeros((2, nb, nbd), int)
    rs = [[[[] for _ in range(nbd)] for _ in range(nb)] for _ in range(2)]
    nref = np.zeros(2, int)
    for a0, b0 in zip(starts, stops):
        if b0 - a0 < 40:
            continue
        idx = order[a0:b0]
        x, dd, ok, ib = T.x[idx], dd_all[idx], okw[idx], T.ib[idx]
        v = 0 if np.median(x) < 0 else 1
        ax = np.abs(x)
        ref = ok & (ax >= xref)
        if ref.sum() < minref or ax[ok].min() > 150:
            continue
        A = np.column_stack([np.ones(ref.sum()), ax[ref]])
        coef, *_ = np.linalg.lstsq(A, dd[ref], rcond=None)
        pred_ref = A @ coef
        if np.sqrt(np.mean((dd[ref] - pred_ref) ** 2)) > rms_max:
            continue
        pred_all = coef[0] + coef[1] * ax[ok]
        if pred_all.min() < 3.0 or (coef[0] + coef[1] * xref) > 40.0:
            continue
        nref[v] += 1
        xmin = ax[ok].min()
        for bi in range(nb):
            if xc[bi] >= xref or xedges[bi] <= xmin:
                continue                                  # bin not demonstrably crossed
            dp = coef[0] + coef[1] * xc[bi]
            band = next((k for k, (lo, hi) in enumerate(BANDS) if lo <= dp < hi), None)
            if band is None:
                continue
            inb = ok & (ib == bi)
            ntot[v, bi, band] += 1
            if inb.sum() >= 3:
                nhas[v, bi, band] += 1
                rs[v][bi][band].append(float(dd[inb].mean() - dp))
    resid = np.full((2, nb, nbd), np.nan); rerr = np.full((2, nb, nbd), np.nan)
    agg = np.full((2, nbd, 4), np.nan)         # |x| < 100 aggregate: n_total, n_has, mean resid, se
    for v in (0, 1):
        for k in range(nbd):
            allr, nt, nh = [], 0, 0
            for bi in range(nb):
                r = np.array(rs[v][bi][k])
                if len(r) >= 3:
                    resid[v, bi, k] = r.mean(); rerr[v, bi, k] = r.std(ddof=1) / np.sqrt(len(r))
                if xc[bi] < 100:
                    allr += rs[v][bi][k]; nt += ntot[v, bi, k]; nh += nhas[v, bi, k]
            allr = np.array(allr)
            agg[v, k] = (nt, nh, allr.mean() if len(allr) else np.nan,
                         allr.std(ddof=1) / np.sqrt(len(allr)) if len(allr) >= 2 else np.nan)
    return {"n_total": ntot, "n_has": nhas, "resid": resid, "resid_err": rerr, "nref": nref,
            "bands": np.array(BANDS), "agg_x_lt_100": agg}


# ------------------------------------------------------------------ endpoint cross-check
def endpoint_table(T, minpts=40, minext=30.0):
    """PCA ends of long tracks, with their raw x, for the endpoint instrument."""
    order = np.argsort(T.key, kind="stable")
    keys = T.key[order]
    first = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.flatnonzero(first); stops = np.r_[starts[1:], len(keys)]
    rows = []
    for a, b in zip(starts, stops):
        if b - a < minpts:
            continue
        idx = order[a:b]
        P = np.column_stack([T.x[idx], T.y[idx], T.z[idx]]).astype(float)
        c = P - P.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        t = c @ axis
        if t.max() - t.min() < minext:
            continue
        for i in (int(np.argmax(t)), int(np.argmin(t))):
            xr = T.xr[idx][i]; sd = T.side[idx][i]
            late = RAW_LATE if sd < 0 else -RAW_LATE
            early = -RAW_EARLY if sd < 0 else RAW_EARLY
            atwin = (abs(xr - late) < RAW_EDGE_TOL) or (abs(xr - early) < RAW_EDGE_TOL)
            rows.append((T.key[idx][i], T.ev[idx][i], P[i, 0], P[i, 1], P[i, 2], int(atwin)))
    r = np.array(rows, float)
    return dict(key=r[:, 0].astype(np.int64), ev=r[:, 1].astype(int), x=r[:, 2], y=r[:, 3], z=r[:, 4], atwin=r[:, 5] > 0)


def at_boundary(x, y, z, tol=12.0):
    return (np.abs(x) > XW - tol) | (np.abs(x) < CATH + tol) | (np.abs(y) > YW - tol) | (z < ZLO + tol) | (z > ZHI - tol)


def endpoint_profile(E, wall, xedges, boot=200, rng=None, nev=120, through=True):
    """Mode of the end wall-distance per (vol, bin) with a bootstrap error; ends at the readout
    window are excluded; `through` additionally requires the track's OTHER end at a boundary."""
    rng = rng or np.random.default_rng(7)
    dd = wall_dist(wall, E["y"], E["z"])
    ok = (~E["atwin"]) & (dd < DMAX) & (dd > -10) & wall_other_ok(wall, E["y"], E["z"])
    if through:
        # partner end: the other row with the same key
        order = np.argsort(E["key"], kind="stable"); k = E["key"][order]
        part = np.full(len(k), -1)
        same = k[1:] == k[:-1]
        i = np.flatnonzero(same)
        part[order[i]] = order[i + 1]; part[order[i + 1]] = order[i]
        has = part >= 0
        pb = np.zeros(len(k), bool)
        pb[has] = at_boundary(E["x"][part[has]], E["y"][part[has]], E["z"][part[has]]) | E["atwin"][part[has]]
        ok &= pb
    nb = len(xedges) - 1
    ib = np.searchsorted(xedges, np.abs(E["x"]), side="right") - 1
    vol = np.where(E["x"] < 0, 0, 1)
    out = {"mode": np.full((2, nb), np.nan), "err": np.full((2, nb), np.nan), "n": np.zeros((2, nb), int)}
    hb = np.arange(-10, DMAX + 1, 1.0)

    def mode(v):
        h, _ = np.histogram(v, bins=hb)
        return hb[int(np.argmax(h))] + 0.5

    for v in (0, 1):
        for b in range(nb):
            m = ok & (vol == v) & (ib == b)
            n = int(m.sum()); out["n"][v, b] = n
            if n < 5:
                continue
            d, ev = dd[m], E["ev"][m]
            out["mode"][v, b] = mode(d)
            bs = []
            for _ in range(boot):
                w = np.bincount(rng.integers(0, nev, nev), minlength=nev)
                rep = np.repeat(d, w[ev])
                if len(rep) >= 5:
                    bs.append(mode(rep))
            out["err"][v, b] = np.std(bs) if len(bs) > 10 else np.nan
    return out


# ------------------------------------------------------------------ main
def main():
    np.seterr(all="ignore")
    import warnings; warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--xstep", type=float, default=0, help="rebin the drift axis to this step (0 = 10 cm table)")
    a = ap.parse_args()
    xedges = XEDGES if a.xstep <= 0 else np.r_[3.0, np.arange(a.xstep, 340 + a.xstep / 2, a.xstep)]
    T = Tracks(a.npz, xedges)
    nb, nev = T.nb, T.nev
    xc = 0.5 * (xedges[1:] + xedges[:-1])
    print("report", json.dumps(T.report), "events", nev, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    result = {"report": T.report, "events": int(nev), "xedges": xedges.tolist(), "xcenter": xc.tolist(),
              "walls": {}, "geometry": dict(XW=XW, YW=YW, ZLO=ZLO, ZHI=ZHI, CATH=CATH)}
    rows = []
    E = endpoint_table(T)
    print("endpoint table", len(E["x"]), "ends,", int(E["atwin"].sum()), "at the readout window", flush=True)
    result["endpoints"] = dict(n_ends=int(len(E["x"])), n_at_window=int(E["atwin"].sum()))

    for w in WALLS:
        H = wall_hists(T, w)
        prof = edge_profile(H, boot=a.boot)
        d2nd, ntr = onset_profile(T.dmin_table(w), nb)
        prof["d2nd"], prof["ntracks"] = d2nd, ntr
        ep = endpoint_profile(E, w, xedges, boot=a.boot, nev=nev, through=True)
        epa = endpoint_profile(E, w, xedges, boot=0, nev=nev, through=False)
        W = {"profile": {k: v.tolist() for k, v in prof.items()},
             "endpoint": {k: v.tolist() for k, v in ep.items()},
             "endpoint_all": {k: v.tolist() for k, v in epa.items()}}
        # symmetry: bottom vs top
        c2, n, mean = chi2_compare(prof["d0"][0], prof["err"][0], prof["d0"][1], prof["err"][1])
        W["xmirror"] = dict(chi2=c2, ndf=n, mean_diff_bot_minus_top=mean)
        c2, n, mean = chi2_compare(prof["d50"][0], prof["d50_err"][0], prof["d50"][1], prof["d50_err"][1])
        W["xmirror_d50"] = dict(chi2=c2, ndf=n, mean_diff_bot_minus_top=mean)
        # the two estimators against each other, and the endpoint instrument against d_eq
        for v in (0, 1):
            c2, n, mean = chi2_compare(prof["d0"][v], prof["err"][v], ep["mode"][v], ep["err"][v])
            W[f"endpoint_vs_density_vol{v}"] = dict(chi2=c2, ndf=n, mean_diff=mean)
            c2, n, mean = chi2_compare(prof["d0"][v], prof["err"][v], prof["d50"][v], prof["d50_err"][v])
            W[f"d50_vs_deq_vol{v}"] = dict(chi2=c2, ndf=n, mean_diff=mean)
        # factorization slices, each bootstrapped
        W["slices"] = {}
        for sname, smask in other_slices(w, T.y, T.z).items():
            sp = edge_profile(wall_hists(T, w, extra=smask), boot=max(a.boot // 4, 20))
            c2, n, mean = chi2_compare(prof["d0"].ravel(), prof["err"].ravel(), sp["d0"].ravel(), sp["err"].ravel())
            c5, n5, mean5 = chi2_compare(prof["d50"].ravel(), prof["d50_err"].ravel(), sp["d50"].ravel(), sp["d50_err"].ravel())
            W["slices"][sname] = dict(d0=sp["d0"].tolist(), n=sp["n"].tolist(), err=sp["err"].tolist(),
                                      d50=sp["d50"].tolist(), d50_err=sp["d50_err"].tolist(),
                                      chi2_vs_all=c2, ndf=n, mean_diff=mean,
                                      chi2_vs_all_d50=c5, ndf_d50=n5, mean_diff_d50=mean5)
        # folded profile (both volumes, inverse-variance), and the model fits
        with np.errstate(all="ignore"):
            wgt = 1 / prof["err"] ** 2
            wsum = np.nansum(prof["d0"] * wgt, axis=0) / np.nansum(wgt, axis=0)
            erf_ = 1 / np.sqrt(np.nansum(wgt, axis=0)); erf_[~np.isfinite(erf_)] = np.nan
            wgt5 = 1 / prof["d50_err"] ** 2
            w5 = np.nansum(prof["d50"] * wgt5, axis=0) / np.nansum(wgt5, axis=0)
            e5 = 1 / np.sqrt(np.nansum(wgt5, axis=0)); e5[~np.isfinite(e5)] = np.nan
        W["folded"] = dict(d0=wsum.tolist(), err=erf_.tolist(), d50=w5.tolist(), d50_err=e5.tolist())
        W["models_folded"] = fit_models(xc, wsum, erf_)
        W["models_folded_d50"] = fit_models(xc, w5, e5)
        W["models_bot"] = fit_models(xc, prof["d0"][0], prof["err"][0])
        W["models_top"] = fit_models(xc, prof["d0"][1], prof["err"][1])
        W["models_bot_d50"] = fit_models(xc, prof["d50"][0], prof["d50_err"][0])
        W["models_top_d50"] = fit_models(xc, prof["d50"][1], prof["d50_err"][1])
        bp = bend_profile(T, w, xedges)
        W["bend"] = {k: v.tolist() for k, v in bp.items()}
        result["walls"][w] = W
        for v in (0, 1):
            for b in range(nb):
                rows.append((w, "bot" if v == 0 else "top", xc[b], prof["n"][v, b], ntr[v, b], prof["d0"][v, b], prof["err"][v, b],
                             prof["d50"][v, b], prof["d50_err"][v, b], prof["rho"][v, b], d2nd[v, b],
                             ep["n"][v, b], ep["mode"][v, b], ep["err"][v, b], epa["n"][v, b], epa["mode"][v, b],
                             bp["n_total"][v, b, 0], bp["n_has"][v, b, 0], bp["resid"][v, b, 0], bp["resid_err"][v, b, 0]))
        print(f"wall {w}: xmirror chi2 {W['xmirror']['chi2']:.1f}/{W['xmirror']['ndf']}  "
              f"M1 {W['models_folded'].get('M1')}  M2 {W['models_folded'].get('M2')}", flush=True)

    # the surface, per drift volume, as the prototype's 6-vertex polygons (M1 on d50, 20 cm
    # bins recommended); inset = 0 where the fitted amplitude is within 2 sigma of 0.
    def m1_pair(w, v):
        M = result["walls"][w]["models_bot_d50" if v == 0 else "models_top_d50"].get("M1", {})
        if "params" not in M:
            return 0.0, XW
        dc, xk = M["params"]
        if abs(dc) < 2 * M["perr"][0] or dc < 0:
            return 0.0, XW
        return float(dc), float(xk)
    poly = {}
    for v, vn, sgn in ((0, "bottom_x_lt_0", -1), (1, "top_x_gt_0", +1)):
        P = {}
        for w in WALLS:
            dc, xk = m1_pair(w, v)
            P[w] = dict(inset_at_cathode_cm=dc, knee_abs_x_cm=xk)
        # x runs from the anode (|x| = XW) to the cathode face (|x| = CATH), signed
        ax, cx = sgn * XW, sgn * CATH
        yp, ym, zm, zp = P["y+"], P["y-"], P["z-"], P["z+"]
        xy = [(ax, -YW), (sgn * ym["knee_abs_x_cm"], -YW), (cx, -YW + ym["inset_at_cathode_cm"]),
              (cx, YW - yp["inset_at_cathode_cm"]), (sgn * yp["knee_abs_x_cm"], YW), (ax, YW)]
        xz = [(ax, ZLO), (sgn * zm["knee_abs_x_cm"], ZLO), (cx, ZLO + zm["inset_at_cathode_cm"]),
              (cx, ZHI - zp["inset_at_cathode_cm"]), (sgn * zp["knee_abs_x_cm"], ZHI), (ax, ZHI)]
        poly[vn] = dict(walls=P, boundary_xy=[[round(a, 2), round(b, 2)] for a, b in xy],
                        boundary_xz=[[round(a, 2), round(b, 2)] for a, b in xz])
    result["polygons_M1_d50"] = poly
    with open(a.out + "_polygons.json", "w") as f:
        json.dump(poly, f, indent=1)

    # transverse mirrors
    for p, q in (("y+", "y-"), ("z-", "z+")):
        A, B = result["walls"][p]["folded"], result["walls"][q]["folded"]
        c2, n, mean = chi2_compare(np.array(A["d0"]), np.array(A["err"]), np.array(B["d0"]), np.array(B["err"]))
        result[f"mirror_{p}_vs_{q}"] = dict(chi2=c2, ndf=n, mean_diff=mean)
        for v, vn in ((0, "bot"), (1, "top")):
            PA, PB = result["walls"][p]["profile"], result["walls"][q]["profile"]
            c2, n, mean = chi2_compare(np.array(PA["d50"][v]), np.array(PA["d50_err"][v]), np.array(PB["d50"][v]), np.array(PB["d50_err"][v]))
            result[f"mirror_{p}_vs_{q}_d50_{vn}"] = dict(chi2=c2, ndf=n, mean_diff=mean)

    with open(a.out + "_edges.csv", "w") as f:
        f.write("wall,vol,x_cm,n_points,n_tracks,d_eq_cm,d_eq_err_cm,d50_cm,d50_err_cm,rho_per_cm,d_2nd_cm,"
                "n_ends_through,end_mode_cm,end_mode_err_cm,n_ends_all,end_mode_all_cm,n_lines_pred0to10,n_lines_survive,line_resid_cm,line_resid_err_cm\n")
        for r in rows:
            f.write(",".join(str(v) if not isinstance(v, float) else f"{v:.3f}" for v in r) + "\n")
    with open(a.out + "_result.json", "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", a.out + "_edges.csv", a.out + "_result.json")


if __name__ == "__main__":
    main()

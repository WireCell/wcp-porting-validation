#!/usr/bin/env python3
"""doc pdvd/44 -- derive the fit's EFFECTIVE transverse smearing per plane, empirically.

The forward model (TrackFitting.cxx:7305-7312) is
    sigma_T(plane) = hypot( sqrt(2 * DT * t_drift), c_plane ) / pitch_plane
with t_drift = max(50 us, |x - x_anode| / v_drift).  This script measures the
transverse width of the MEASURED charge about its own centroid, per (block, plane,
time slice), on transversally localized ("prolonged") segments only, resolves it
in drift time and fits
    sigma_eff^2(t) = 2 * DT_eff * t + c_eff^2
per plane, plus a joint fit with one DT_eff shared by the three planes (diffusion
is a property of the argon, not of the plane).  It is a fork of
d42_transverse_moments.py (doc 42 sec 8.4) with four changes:

  * the drift time is |x_anode| - |x| over v, as the fit computes it -- NOT |x|
    (doc 42 sec 7.4 binned in |x| and so read the drift direction backwards);
  * the prolonged cut is the local wire ADVANCE per slice at the nearest trajectory
    point (doc 42 sec 8.9's definition), not the within-slice span, which is inert;
  * profiles whose window touches a dead channel are dropped (T_bad_ch);
  * the binned-variance inversion (integral vs sampling) is always applied, the
    result is resolved in drift bins with bootstrap errors over blocks, and the
    two-parameter line is fitted.  Everything is written to TSV.

Usage:
  d44_sigma_fit.py --det pdvd --out figs/44_sigma_pdvd work/*_d42fit/tracking-stm.root
  d44_sigma_fit.py --det sbnd --out figs/44_sigma_sbnd .../work-stmcamp-d42fit/*/tracking-stm.root
  --model-json <track_fitting.json>   read DT / ind_sigma_u_T / ind_sigma_v_T / col_sigma_w_T
                                      from there (default: the canonical file of --det)
  --split face,run,length,rr,foff,window,advance   the validation "occasions"
"""
import argparse, json, os, re, sys
import numpy as np
import uproot
from scipy.special import erf
from scipy.spatial import cKDTree

TOOLKIT_CFG = "/home/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment"
DET = {
    # pitch [mm], plane channel bounds (global index), x of the collection plane
    # [cm] (wires file, first W wire: PDVD protodunevd-wires-larsoft-v7-uvwfit,
    # SBND sbnd-wires-geometry-v0206), drift speed [mm/us], slice [us]
    "pdvd": dict(pitch=(7.65, 7.65, 5.10), bounds=(3808, 7616), x_anode=341.55, v=1.48073,
                 slice_us=2.0, json=TOOLKIT_CFG + "/protodunevd/pdvd_track_fitting.json"),
    "sbnd": dict(pitch=(3.00, 3.00, 3.00), bounds=(3968, 7936), x_anode=202.05, v=1.563,
                 slice_us=2.0, json=TOOLKIT_CFG + "/sbnd/sbnd_track_fitting.json"),
}
MIN_DRIFT_NS = 50e3       # m_params.min_drift_time
SIGKEYS = ("ind_sigma_u_T", "ind_sigma_v_T", "col_sigma_w_T")


def load_model(path):
    j = json.load(open(path))
    return dict(DT=float(j["DT"]), c=tuple(float(j[k]) for k in SIGKEYS),
                nsigma=float(j.get("gaus_nsigma", 4.0)), path=path)


def sigma_model_pitch(model, t_ns, plane, pitch):
    return np.hypot(np.sqrt(2.0 * model["DT"] * t_ns), model["c"][plane]) / pitch


# ----------------------------------------------------------------------------- estimator
_U = np.linspace(-0.5, 0.5, 21)          # sub-bin positions averaged over
_N = np.arange(-14, 15)                  # unit wire bins


def _binned_profiles(sig, extent, nsigma=None):
    """unit-bin integrals of a Gaussian line source of width sig (pitch units)
    spanning `extent` wires, for every sub-bin start offset in _U -> (nu, nbins).
    With nsigma, a bin farther than nsigma*sig from a sub-point gets nothing from
    that sub-point -- the fit's cal_gaus_integral acceptance window (nsigma = 4)."""
    ncs = max(2, int(extent * 20) + 2) if extent > 0 else 1
    cs = _U[:, None] + (np.linspace(0, extent, ncs) if extent > 0 else np.zeros(1))[None, :]   # (nu, ncs)
    dlt = _N[None, None, :] - cs[:, :, None]                                                    # (nu, ncs, nb)
    lo = (dlt - 0.5) / (np.sqrt(2) * sig); hi = (dlt + 0.5) / (np.sqrt(2) * sig)
    v = 0.5 * (erf(hi) - erf(lo))
    if nsigma is not None:
        v = np.where(np.abs(dlt) <= nsigma * sig, v, 0.0)
    v = v.sum(axis=1)
    tot = v.sum(axis=1, keepdims=True)
    return v / np.where(tot > 0, tot, 1.0)


def apparent_rms(sig, extent, nsigma=None):
    """rms about its own centroid of the binned profile, averaged over the sub-bin
    position -- exactly the statistic measured below (doc 42 sec 8.4)."""
    v = _binned_profiles(sig, extent, nsigma)
    mu = (v * _N).sum(axis=1)
    var = (v * (_N[None, :] - mu[:, None]) ** 2).sum(axis=1)
    return np.sqrt(var.mean())


def ring_shares(sig, extent, nsigma=None):
    """share of the binned profile in the wire nearest its centroid, the +-1 ring,
    the +-2 ring and beyond, averaged over the sub-bin position."""
    v = _binned_profiles(sig, extent, nsigma)
    mu = (v * _N).sum(axis=1)
    d = np.abs(_N[None, :] - np.round(mu)[:, None])
    return np.array([v[d == 0].sum(), v[d == 1].sum(), v[d == 2].sum(), v[d >= 3].sum()]) / len(_U)


def _bisect(f, target, lo, hi, n=36):
    for _ in range(n):
        m = 0.5 * (lo + hi)
        if f(m) < target:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi), (hi > 0.999 * 2.0)


def unfold(sig_mod, rp, rm, nsigma):
    """solve the in-slice extent from the PREDICTED rms (its sigma and truncation are
    known), then the measured sigma at that extent with no truncation.
    Returns (sig_meas [pitch], extent, ceiling_hit)."""
    ext, hitc = _bisect(lambda e: apparent_rms(sig_mod, e, nsigma), rp, 0.0, 2.0)
    sme, _ = _bisect(lambda g: apparent_rms(g, ext), rm, 0.02, 3.0)
    return sme, ext, hitc


# ----------------------------------------------------------------------------- collection
def collect(a, det, model):
    """one row per accepted (block, plane, slice) profile."""
    rows = []          # see COLS
    blocks = {}        # bid -> dict(run, length_cm, foff_W)
    P, bounds = det["pitch"], det["bounds"]
    hw = int(round(a.halfwidth))
    bid = 0
    for path in a.roots:
        try:
            f = uproot.open(path); ks = {k.split(";")[0] for k in f.keys()}
        except Exception as ex:
            print("skip", path, ex, file=sys.stderr); continue
        if "T_proj_data" not in ks:
            continue
        m_run = re.search(r"/(\d{6})_(\d+)_", path)
        run = int(m_run.group(1)) if m_run else -1
        r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "rr", "ndf", "status", "x"], library="np")
        d = f["T_proj_data"].arrays(library="np")
        bad = {}
        if "T_bad_ch" in ks:
            b = f["T_bad_ch"].arrays(["chid", "start_time", "end_time"], library="np")
            for c_, t0, t1 in zip(b["chid"], b["start_time"], b["end_time"]):
                bad.setdefault(int(c_), []).append((t0 / a.ticks_per_slice, t1 / a.ticks_per_slice))
        for i, blk in enumerate([int(c) for c in d["cluster_id"][0]]):
            m = r["ndf"] == blk
            if m.sum() < 5 or (a.status >= 0 and int(r["status"][m][0]) != a.status):
                continue
            ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
            q = np.asarray(list(d["charge"][0][i]), float)
            qp = np.asarray(list(d["charge_pred"][0][i]), float)
            pl = np.digitize(ch, bounds)
            o = np.argsort(-r["rr"][m])
            pt = r["pt"][m][o]; rr = r["rr"][m][o]; px = r["x"][m][o]
            npts = len(pt)
            it = np.gradient(pt) if npts > 2 else np.full(npts, pt[-1] - pt[0])
            # per-block fusion score on W: live charge beyond Chebyshev 2 of the trajectory
            mpW = pl == 2
            foff = np.nan
            if mpW.sum() > 0:
                tw = cKDTree(np.column_stack([np.round(r["pw"][m][o]), np.round(pt)]).astype(float))
                dW, _ = tw.query(np.column_stack([ch[mpW], ts[mpW]]).astype(float), p=np.inf)
                qW = np.where(q[mpW] > 0, q[mpW], 0)
                if qW.sum() > 0:
                    foff = 1 - qW[dW <= 2].sum() / qW.sum()
            bid += 1
            blocks[bid] = dict(run=run, length_cm=float(rr.max()), foff=float(foff), path=path, blk=blk)
            for Pi, key in enumerate(("pu", "pv", "pw")):
                mp = pl == Pi
                if mp.sum() == 0:
                    continue
                pw_ = r[key][m][o]
                iw = np.gradient(pw_) if npts > 2 else np.full(npts, pw_[-1] - pw_[0])
                adv = np.abs(iw) / np.maximum(np.abs(it), 1e-9)          # wires per slice
                cw = ch[mp].astype(float); ct = ts[mp]; Q = q[mp]; QP = qp[mp]
                cellmap = {}
                for k_ in range(len(cw)):
                    cellmap[(int(cw[k_]), int(ct[k_]))] = k_
                for s_ in np.unique(ct):
                    dt = np.abs(pt - s_)
                    near = dt <= 0.6
                    if not near.any():
                        continue
                    j0 = int(np.argmin(dt))
                    advance = float(adv[near].max())
                    wc = 0.5 * (pw_[near].min() + pw_[near].max())
                    w0 = int(np.round(wc))
                    # cells absent from the snapshot are cells with no blob there: zero
                    # measured charge (and no recorded prediction), as in doc 42
                    idx = [cellmap.get((w0 + k_, int(s_)), -1) for k_ in range(-hw, hw + 1)]
                    dead = False
                    for k_ in range(-hw, hw + 1):
                        for (t0, t1) in bad.get(w0 + k_, ()):
                            if t0 - 0.5 <= s_ <= t1 + 0.5:
                                dead = True; break
                        if dead:
                            break
                    if dead:
                        continue
                    idx = np.asarray(idx)
                    x = w0 + np.arange(-hw, hw + 1, dtype=float)
                    have = idx >= 0
                    y = np.zeros(len(idx)); yh = np.zeros(len(idx))
                    y[have] = np.where(Q[idx[have]] > 0, Q[idx[have]], 0.0); yh[have] = QP[idx[have]]
                    if y.sum() <= 0 or yh.sum() <= 0 or (y > 0).sum() < 2:
                        continue
                    mu_m = np.average(x, weights=y); mu_p = np.average(x, weights=yh)
                    vm = np.average((x - mu_m) ** 2, weights=y)
                    vp = np.average((x - mu_p) ** 2, weights=yh)
                    absx = abs(float(px[j0]))
                    t_ns = max(MIN_DRIFT_NS, (det["x_anode"] - absx) * 10.0 / det["v"] * 1e3)
                    n0 = int(np.round(mu_m)) - w0
                    dd = np.abs(np.arange(-hw, hw + 1) - n0)
                    sh = [y[dd == 0].sum(), y[dd == 1].sum(), y[dd == 2].sum(), y[dd >= 3].sum()]
                    rows.append((bid, Pi, s_, t_ns, y.sum(), vm, yh.sum(), vp, mu_m - mu_p,
                                 advance, float(rr[j0]), np.sign(px[j0]), *sh))
    cols = ("bid", "plane", "slice", "t_ns", "y", "vm", "yh", "vp", "off", "adv", "rr", "xsign",
            "r0", "r1", "r2", "r3")
    R = {c: np.array([row[k] for row in rows]) for k, c in enumerate(cols)}
    R["bid"] = R["bid"].astype(int); R["plane"] = R["plane"].astype(int)
    return R, blocks


# ----------------------------------------------------------------------------- aggregation
def bin_and_fit(R, sel, det, model, nbins, nboot, rng, label):
    """per plane: drift bins, bootstrap over blocks, unfold, line fit. Returns rows."""
    out_bins, out_fit = [], []
    joint = []      # (plane, t, sig2, err)
    for Pi in range(3):
        s = sel & (R["plane"] == Pi)
        if s.sum() < 20:
            continue
        pitch = det["pitch"][Pi]
        t = R["t_ns"][s]; y = R["y"][s]; vm = R["vm"][s]; yh = R["yh"][s]; vp = R["vp"][s]
        b = R["bid"][s]
        # equal-population (charge-weighted) drift bins
        o = np.argsort(t); cy = np.cumsum(y[o]) / y.sum()
        edges = [t[o][0]] + [t[o][np.searchsorted(cy, k / nbins)] for k in range(1, nbins)] + [t[o][-1] + 1]
        tb = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
        # per-block sums per bin: [y*vm, y, yh*vp, yh, y*t]
        ub, binv = np.unique(b, return_inverse=True)
        nb = len(ub)
        r0 = R["r0"][s]
        acc = np.zeros((nb, nbins, 6))
        for k, arr in enumerate((y * vm, y, yh * vp, yh, y * t, r0)):
            np.add.at(acc[:, :, k], (binv, tb), arr)
        tot = acc.sum(axis=0)

        def solve(tot_):
            # per bin: (t, rms_meas, rms_pred, sig_model, sig2_rms, sig2_naive, extent, ceiling, sig2_share)
            res = []
            for ib in range(nbins):
                Y, YH = tot_[ib, 1], tot_[ib, 3]
                if Y <= 0 or YH <= 0:
                    res.append((np.nan,) * 7 + (False, np.nan)); continue
                rm = np.sqrt(tot_[ib, 0] / Y); rp = np.sqrt(tot_[ib, 2] / YH)
                tm = tot_[ib, 4] / Y
                sm = sigma_model_pitch(model, tm, Pi, pitch)
                sme, ext, hitc = unfold(sm, rp, rm, model["nsigma"])
                naive2 = (sm * pitch) ** 2 + ((rm * pitch) ** 2 - (rp * pitch) ** 2)
                # second estimator: the sigma whose binned Gaussian puts the measured share
                # of charge in the wire nearest the centroid (monotone decreasing in sigma)
                share = tot_[ib, 5] / Y
                ssh, _ = _bisect(lambda g: -ring_shares(g, ext)[0], -share, 0.02, 3.0)
                res.append((tm, rm * pitch, rp * pitch, sm * pitch, (sme * pitch) ** 2, naive2, ext, hitc,
                            (ssh * pitch) ** 2))
            return res

        central = solve(tot)
        boots = np.full((nboot, nbins, 2), np.nan)
        for ib_ in range(nboot):
            pick = rng.integers(0, nb, nb)
            rs = solve(acc[pick].sum(axis=0))
            boots[ib_, :, 0] = [r_[4] for r_ in rs]
            boots[ib_, :, 1] = [r_[8] for r_ in rs]
        T = np.array([c_[0] for c_ in central])
        for est, col in (("rms", 4), ("share", 8)):
            S2 = np.array([c_[col] for c_ in central])
            # error floor: a bin fed by one or two blocks bootstraps to ~0; never let a
            # single bin pin the line (5 % of sigma^2 ~ 2.5 % of sigma)
            err = np.sqrt(np.nanstd(boots[:, :, 0 if est == "rms" else 1], axis=0) ** 2 + (0.05 * np.abs(S2)) ** 2)
            ok = np.isfinite(S2) & np.isfinite(err) & (err > 0)
            for ib in range(nbins):
                c_ = central[ib]
                if not np.isfinite(c_[0]):
                    continue
                out_bins.append(dict(label=label, est=est, plane="UVW"[Pi], tbin=ib, t_us=c_[0] / 1e3,
                                     n_prof=int(((tb == ib)).sum()), q=float(tot[ib, 1]),
                                     rms_meas_mm=c_[1], rms_pred_mm=c_[2], sig_model_mm=c_[3],
                                     sig2_eff_mm2=float(S2[ib]), sig2_err=float(err[ib]), sig2_naive_mm2=c_[5],
                                     extent=c_[6], ceiling=int(bool(c_[7])), centre_share=float(tot[ib, 5] / tot[ib, 1])))
                if ok[ib]:
                    joint.append((Pi, c_[0], S2[ib], err[ib], est))
            if ok.sum() >= 3:
                A = np.column_stack([2 * T[ok], np.ones(ok.sum())]); w = 1.0 / err[ok]
                beta, cov, chi2 = wlsq(A, S2[ok], w)
                out_fit.append(fitrow(label, est, "UVW"[Pi], beta[0], np.sqrt(cov[0, 0]), beta[1], np.sqrt(cov[1, 1]),
                                      chi2, ok.sum() - 2, model, Pi))
    # joint: one D, three c2 (per estimator)
    for est in ("rms", "share"):
        J = np.array([(p, t_, s2, e_) for (p, t_, s2, e_, es) in joint if es == est])
        if len(J) < 5:
            continue
        A = np.column_stack([2 * J[:, 1]] + [(J[:, 0] == p).astype(float) for p in range(3)])
        keep = [0] + [1 + p for p in range(3) if (J[:, 0] == p).any()]
        A = A[:, keep]
        beta, cov, chi2 = wlsq(A, J[:, 2], 1.0 / J[:, 3])
        kk = 1
        for p in range(3):
            if not (J[:, 0] == p).any():
                continue
            out_fit.append(fitrow(label, est, "UVW"[p] + "(joint)", beta[0], np.sqrt(cov[0, 0]), beta[kk],
                                  np.sqrt(max(cov[kk, kk], 0)), chi2, len(J) - len(beta), model, p))
            kk += 1
    return out_bins, out_fit


def wlsq(A, yv, w):
    Aw = A * w[:, None]; yw = yv * w
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)
    chi2 = float(((yw - Aw @ beta) ** 2).sum())
    return beta, cov, chi2


def fitrow(label, est, plane, D, dD, c2, dc2, chi2, ndf, model, Pi):
    c = np.sqrt(c2) if c2 >= 0 else -np.sqrt(-c2)
    dc = dc2 / (2 * abs(c)) if abs(c) > 1e-6 else np.nan
    return dict(label=label, est=est, plane=plane, DT_eff_cm2s=D * 1e7, DT_err=dD * 1e7, c_eff_mm=c, c_err=dc,
                c2_mm2=c2, c2_err=dc2, chi2=chi2, ndf=ndf,
                DT_model_cm2s=model["DT"] * 1e7, c_model_mm=model["c"][Pi],
                DT_json=D, c_json=abs(c))


def write_tsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as fo:
        fo.write("\t".join(keys) + "\n")
        for r_ in rows:
            fo.write("\t".join("%.6g" % v if isinstance(v, float) else str(v) for v in (r_[k] for k in keys)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--det", required=True, choices=DET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-json", default=None)
    ap.add_argument("--status", type=int, default=0)
    ap.add_argument("--ticks-per-slice", type=float, default=4.0)
    ap.add_argument("--max-advance", type=float, default=0.25,
                    help="prolonged cut: local |d wire / d slice| at the nearest trajectory point")
    ap.add_argument("--halfwidth", type=float, default=3.0, help="cells each side of the trajectory")
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--split", default="",
                    help="comma list of occasions: face,run,length,rr,foff,advance (window needs a rerun)")
    ap.add_argument("--max-foff", type=float, default=1.1)
    a = ap.parse_args()
    det = DET[a.det]
    model = load_model(a.model_json or det["json"])
    rng = np.random.default_rng(a.seed)
    R, blocks = collect(a, det, model)
    if len(R["bid"]) == 0:
        print("no profiles", file=sys.stderr); return 1
    bf = np.array([blocks[b]["foff"] for b in R["bid"]])
    base = (R["adv"] < a.max_advance)
    if a.max_foff < 1.0:
        base &= np.nan_to_num(bf, nan=1.0) <= a.max_foff
    print("  advance quantiles (wires/slice) 10/25/50/75/90%%: %s" % np.round(np.percentile(R["adv"], [10, 25, 50, 75, 90]), 3))
    print("%s: %d blocks, %d profiles, %d prolonged (adv<%.2f); model %s" % (
        a.det.upper(), len(blocks), len(base), base.sum(), a.max_advance, model["path"]))
    print("  DT %.4g cm2/s  c(U,V,W) %s mm  nsigma %g" % (model["DT"] * 1e7, model["c"], model["nsigma"]))

    bins, fits = bin_and_fit(R, base, det, model, a.nbins, a.nboot, rng, "all")
    occasions = [x for x in a.split.split(",") if x]
    for occ in occasions:
        if occ == "face":
            cats = {"x>0": R["xsign"] > 0, "x<0": R["xsign"] < 0}
        elif occ == "run":
            runs = np.array([blocks[b]["run"] for b in R["bid"]])
            cats = {"run%06d" % u: runs == u for u in np.unique(runs) if u > 0}
        elif occ == "length":
            L = np.array([blocks[b]["length_cm"] for b in R["bid"]])
            cats = {"L<100cm": L < 100, "L>=100cm": L >= 100}
        elif occ == "rr":
            cats = {"rr<10cm": R["rr"] < 10, "rr>30cm": R["rr"] > 30}
        elif occ == "foff":
            cats = {"foff<0.15": np.nan_to_num(bf, nan=1) < 0.15, "foff>=0.15": np.nan_to_num(bf, nan=1) >= 0.15}
        elif occ == "advance":
            cats = {"adv<0.10": R["adv"] < 0.10, "adv0.10-0.25": (R["adv"] >= 0.10) & (R["adv"] < 0.25),
                    "adv0.25-0.5": (R["adv"] >= 0.25) & (R["adv"] < 0.5)}
        else:
            print("unknown split", occ, file=sys.stderr); continue
        for name, msk in cats.items():
            selm = (base if occ != "advance" else np.ones_like(base)) & msk
            if occ == "advance" and a.max_foff < 1.0:
                selm &= np.nan_to_num(bf, nan=1.0) <= a.max_foff
            b_, f_ = bin_and_fit(R, selm, det, model, a.nbins, max(50, a.nboot // 4), rng, "%s:%s" % (occ, name))
            bins += b_; fits += f_
    write_tsv(a.out + "_bins.tsv", bins)
    write_tsv(a.out + "_fit.tsv", fits)

    # shape check: measured ring shares (prolonged, all t) vs the binned Gaussian at the
    # fitted sigma and the mean extent of the "all" bins
    shape = []
    for Pi in range(3):
        s = base & (R["plane"] == Pi)
        if s.sum() < 20:
            continue
        y = R["y"][s]
        meas = np.array([R[k][s].sum() for k in ("r0", "r1", "r2", "r3")]) / y.sum()
        fr = [f_ for f_ in fits if f_["label"] == "all" and f_["est"] == "rms" and f_["plane"] == "UVW"[Pi]]
        fr2 = [f_ for f_ in fits if f_["label"] == "all" and f_["est"] == "share" and f_["plane"] == "UVW"[Pi]]
        br = [b_ for b_ in bins if b_["label"] == "all" and b_["est"] == "rms" and b_["plane"] == "UVW"[Pi]]
        if not fr or not br:
            continue
        tmean = np.average([b_["t_us"] for b_ in br], weights=[b_["q"] for b_ in br]) * 1e3
        ext = np.average([b_["extent"] for b_ in br], weights=[b_["q"] for b_ in br])
        pitch = det["pitch"][Pi]
        sig_fit = np.sqrt(max(2 * fr[0]["DT_json"] * tmean + fr[0]["c2_mm2"], 1e-6)) / pitch
        sig_mod = sigma_model_pitch(model, tmean, Pi, pitch)
        mfit = ring_shares(sig_fit, ext); mmod = ring_shares(sig_mod, ext, model["nsigma"])
        sig_sh = np.sqrt(max(2 * fr2[0]["DT_json"] * tmean + fr2[0]["c2_mm2"], 1e-6)) / pitch if fr2 else np.nan
        msh = ring_shares(sig_sh, ext) if fr2 else np.full(4, np.nan)
        excess = (meas[2] + meas[3]) - (mfit[2] + mfit[3])
        shape.append(dict(plane="UVW"[Pi], sig_fit_pitch=sig_fit, sig_share_pitch=sig_sh, sig_model_pitch=sig_mod, extent=ext,
                          meas_centre=meas[0], meas_pm1=meas[1], meas_pm2=meas[2], meas_beyond=meas[3],
                          gaus_fit_centre=mfit[0], gaus_fit_pm1=mfit[1], gaus_fit_pm2=mfit[2], gaus_fit_beyond=mfit[3],
                          gaus_model_centre=mmod[0], gaus_model_pm1=mmod[1], gaus_model_pm2=mmod[2],
                          gaus_model_beyond=mmod[3],
                          gaus_share_centre=msh[0], gaus_share_pm1=msh[1], gaus_share_pm2=msh[2], gaus_share_beyond=msh[3],
                          U_stack_rms=float(np.abs(meas - mfit).sum()), U_stack_share=float(np.abs(meas - msh).sum()),
                          U_stack_model=float(np.abs(meas - mmod).sum()),
                          tail_excess_share=excess, tail_excess_over_pm1=excess / max(mfit[1], 1e-9)))
    write_tsv(a.out + "_shape.tsv", shape)

    # console summary
    print("  drift bins (label=all):  plane tbin  t[us]   n    rms_meas  rms_pred  sig_mod | sig_eff(rms) +-  sig_eff(share) +-  [mm]  extent  centre share")
    bsh = {(b_["plane"], b_["tbin"]): b_ for b_ in bins if b_["label"] == "all" and b_["est"] == "share"}
    for b_ in bins:
        if b_["label"] != "all" or b_["est"] != "rms":
            continue
        b2 = bsh[(b_["plane"], b_["tbin"])]
        print("    %s %d %8.0f %6d %8.2f %8.2f %8.2f | %6.2f +- %.2f  %6.2f +- %.2f  %.2f  %.3f%s" % (
            b_["plane"], b_["tbin"], b_["t_us"], b_["n_prof"], b_["rms_meas_mm"], b_["rms_pred_mm"],
            b_["sig_model_mm"], np.sqrt(max(b_["sig2_eff_mm2"], 0)),
            b_["sig2_err"] / (2 * np.sqrt(max(b_["sig2_eff_mm2"], 1e-6))),
            np.sqrt(max(b2["sig2_eff_mm2"], 0)), b2["sig2_err"] / (2 * np.sqrt(max(b2["sig2_eff_mm2"], 1e-6))),
            b_["extent"], b_["centre_share"], "  CEILING" if b_["ceiling"] else ""))
    print("  fits sig_eff^2 = 2 DT t + c^2:")
    print("    %-22s %-6s %-10s %14s %14s %10s" % ("label", "est", "plane", "DT_eff [cm2/s]", "c_eff [mm]", "chi2/ndf"))
    for f_ in fits:
        print("    %-22s %-6s %-10s %7.2f +- %4.2f %7.2f +- %4.2f %6.1f/%d   (model %.2f, %.3f)" % (
            f_["label"], f_["est"], f_["plane"], f_["DT_eff_cm2s"], f_["DT_err"], f_["c_eff_mm"], f_["c_err"],
            f_["chi2"], f_["ndf"], f_["DT_model_cm2s"], f_["c_model_mm"]))
    if shape:
        print("  shape (prolonged, all t): ring shares centre / +-1 / +-2 / beyond")
        for s_ in shape:
            print("    %s meas %.3f %.3f %.3f %.3f | gaus@rms %.3f %.3f %.3f %.3f | gaus@share %.3f %.3f %.3f %.3f | gaus@model %.3f %.3f %.3f %.3f | tail excess %.3f (%.0f%% of +-1) | stack U model/rms/share %.3f/%.3f/%.3f" % (
                s_["plane"], s_["meas_centre"], s_["meas_pm1"], s_["meas_pm2"], s_["meas_beyond"],
                s_["gaus_fit_centre"], s_["gaus_fit_pm1"], s_["gaus_fit_pm2"], s_["gaus_fit_beyond"],
                s_["gaus_share_centre"], s_["gaus_share_pm1"], s_["gaus_share_pm2"], s_["gaus_share_beyond"],
                s_["gaus_model_centre"], s_["gaus_model_pm1"], s_["gaus_model_pm2"], s_["gaus_model_beyond"],
                s_["tail_excess_share"], 100 * s_["tail_excess_over_pm1"],
                s_["U_stack_model"], s_["U_stack_rms"], s_["U_stack_share"]))
    print("  -> %s_{bins,fit,shape}.tsv" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

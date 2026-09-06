#!/usr/bin/env python3
"""doc pdvd/47 -- the doc-44 transverse-width estimator applied to SIMULATED frames whose
truth is known (tracks from d47_make_xtracks.py through <det>_sim/wct-sim-xtrack-sp.jsonnet).

Per (track, plane, time slice of --ticks-per-slice ticks) the +-3-wire window around the
TRUTH wire position is summed over the slice, negatives clipped (d44_sigma_fit.collect
does the same to the ctpc charge), and the profile's own-centroid rms and centre-wire
share are recorded -- exactly the statistics doc 44 measures on data -- plus the
TRUTH-centred second moment and ring shares, which data cannot have.  Six equal-charge
drift bins per plane, the sigma of each bin inverted from the binned-Gaussian model at
the track's known in-slice extent (rms-matched and share-matched, imported from
d44_sigma_fit.py so the machinery is byte-for-byte the data's), bootstrap over TRACKS,
and the per-plane + joint line sigma_eff^2 = 2 DT t + c^2.  Same TSV schema as
44_sigma_*_{bins,fit,shape}.tsv.

Frame reading: a FrameFileSink archive (frame_<tag>_<ident>.npy float32 rows x ticks in
channels_<tag>_<ident>.npy order, tickinfo = [frame t0 ns, tick ns, tbin]) -- column j is
tick tbin + j.  The splat sink writes a wildcard tag ('frame_*_0.npy'); --tag auto takes
whatever single tag the archive holds.  Windows are built in WIRE-INDEX order on the
sensitive face (the generator's convention, rebuilt here from the wires file) and mapped
to channels -- never by channel rank, which breaks on PDHD's wrapped induction planes.

x <-> tick: predicted from the truth (drift to the response plane at the Drifter's speed,
plus the field-response file's own origin/speed delay, minus the frame t0), then
SELF-CALIBRATED per track on the collection plane: the charge centroid across wires is
linear in tick (w = w0 + slope |x - x_start|), so a straight-line fit gives both the tick
of x_start and mm/tick; the residual against the prediction is written to _calib.tsv.
Drift time uses the DATA convention t = |x - x_W| / v (from the collection plane); note the
simulation diffuses only up to the response plane (9-18 cm short of the wires), so the
simulated D_T,eff is expected slightly BELOW the configured D_T in this convention.

Usage:
  d47_sim_transverse_profile.py --det pdhd --truth truth_pdhd_a1.json --frames S1-anode1-sp.tar.bz2 \\
      --tag gauss --out /home/xqian/tmp/xtrack/pdhd/ana/S1_gauss [--kernel] [--phase-split] [--no-clip]
"""
import argparse, io, json, os, re, sys, tarfile
import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import apparent_rms, ring_shares, _bisect, wlsq, fitrow, write_tsv, _binned_profiles, _N, _U  # noqa: E402
from d47_make_xtracks import load_geom, plane_geometry, WIRES, DATA  # noqa: E402


def read_frame(path, tag):
    """-> dict(frame, chans, t0_ns, tick_ns, tbin, row_of, tag)"""
    with tarfile.open(path) as t:
        names = t.getnames()
        tags = sorted({re.match(r"^frame_(.*)_(\d+)\.npy$", n).group(1) for n in names if n.startswith("frame_")})
        if tag == "auto":
            if len(tags) != 1:
                raise SystemExit("--tag auto needs one tag in %s, found %s" % (path, tags))
            tag = tags[0]
        elif tag not in tags:
            raise SystemExit("tag %s not in %s (%s)" % (tag, path, tags))
        get = lambda pre: np.load(io.BytesIO(t.extractfile(next(n for n in names if re.match(r"^%s_%s_\d+\.npy$" % (pre, re.escape(tag)), n))).read()))
        fr, ch, ti = get("frame"), get("channels"), get("tickinfo")
    return dict(frame=fr, chans=ch, t0_ns=float(ti[0]), tick_ns=float(ti[1]), tbin=int(ti[2]),
                row_of={int(c): i for i, c in enumerate(ch)}, tag=tag)


def truth_rms(sig, extent, phase=None):
    """rms of the binned profile about the TRUE centre (the Gaussian's mean), averaged
    over the sub-bin position: 1/sqrt(12) for a point source, unlike apparent_rms.
    `phase` restricts the average to segment centres inside that sub-pitch window."""
    v = _binned_profiles(sig, extent, None, phase)
    if phase is None:
        cs = _U + 0.5 * extent
    else:
        nu = v.shape[0]
        cs = phase[0] + (phase[1] - phase[0]) * (np.arange(nu) + 0.5) / nu
    var = (v * (_N[None, :] - cs[:, None]) ** 2).sum(axis=1)
    return np.sqrt(var.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=WIRES)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--frames", required=True, nargs="+")
    ap.add_argument("--tag", default="gauss", help="gauss|wiener|rawdecon|raw|auto (tag name without the anode number)")
    ap.add_argument("--wires", default=None)
    ap.add_argument("--ticks-per-slice", type=int, default=4)
    ap.add_argument("--halfwidth", type=int, default=3)
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--drop-edge-slices", type=int, default=3)
    ap.add_argument("--no-clip", action="store_true", help="use the unclipped charge for the estimator")
    ap.add_argument("--phase-split", action="store_true")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--phase-bins", type=int, default=10)
    ap.add_argument("--phase-src", default="truth", choices=("truth", "centroid"),
                    help="which phase the windows are cut on: the TRUTH (perfect knowledge) or the "
                         "profile's own charge centroid -- a charge-driven estimator with the same "
                         "kind of selection correlation the data's fitted trajectory has")
    ap.add_argument("--phase-jitter", type=float, default=0.0,
                    help="wires: bin by the TRUE phase smeared by this rms, to emulate the "
                         "resolution of the fitted trajectory phase the data must use (doc 47 sec 8)")
    ap.add_argument("--min-frac", type=float, default=0.15, help="slices below this fraction of the track's median window charge are dropped")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = json.load(open(a.truth))
    wires_path = a.wires or os.path.join(DATA, WIRES[a.det])
    anodes, faces, planes, wires, P = load_geom(wires_path)
    an = next(x for x in anodes if x["ident"] == tr["anode"])
    fidx = an["faces"][tr["face_index"]]
    pg = [plane_geometry(planes[pi], wires, P) for pi in faces[fidx]["planes"][:3]]
    for pi in range(3):
        assert abs(pg[pi]["pitch"] - tr["pitch_mm"][pi]) < 1e-6, "wires file / truth mismatch"
    hw = a.halfwidth; tps = a.ticks_per_slice
    v_mm_ns = tr["lar"]["drift_speed_mm_ns"]
    xr = tr["xregion_mm"]; x_W = tr["x_plane_mm"][2]
    frames = [read_frame(f, a.tag if a.tag == "auto" else "%s%d" % (a.tag, tr["anode"])) for f in a.frames]
    if len(frames) != 1:
        raise SystemExit("one archive per call for now")
    F = frames[0]; fr = F["frame"]; tick_ns = F["tick_ns"]; tbin = F["tbin"]; ncol = fr.shape[1]
    v_tick = v_mm_ns * tick_ns
    # the FR file's own delay from the response plane to the wires (DepoTransform bakes it in)
    fr_delay_ns = None
    cfg = json.load(open(tr["cfg"])) if os.path.exists(tr["cfg"]) else []
    for n in cfg:
        if n.get("type") == "DepoFluxSplat":
            fr_delay_ns = float(n["data"]["response_plane"]) / float(n["data"]["drift_speed"])
    if fr_delay_ns is None:
        fr_delay_ns = abs(xr["response"] - xr["anode"]) / v_mm_ns    # ~ origin / speed
    # predicted tick of a depo at x: arrival at the response plane + FR delay - frame t0
    def tick_pred(x):
        return (abs(x - xr["response"]) / v_mm_ns + fr_delay_ns - F["t0_ns"]) / tick_ns

    # ---- channel collision check, TIME-RESOLVED: all tracks share one direction, so the wire
    # offset between two tracks is constant in time; at wire k of track A, track B sits at
    # k + dk.  Their +-hw windows collide when a channel is shared (PDHD wrapped planes:
    # wires 400 or 800 apart carry the same channel).  Those k's are excluded for both.
    badk = [{} for _ in range(3)]      # plane -> track id -> set of centre wires k to skip
    trk = {t["id"]: t for t in tr["tracks"]}
    for pi in range(3):
        g = pg[pi]; N = len(g["idents"]); ch = g["channels"]
        for ta in tr["tracks"]:
            pa = ta["planes"][pi]
            for tb_ in tr["tracks"]:
                if tb_["id"] == ta["id"]:
                    continue
                pb = tb_["planes"][pi]
                dk = pb["w0"] - pa["w0"]      # constant offset (same slope)
                for k in range(max(hw, pa["k_range"][0]), min(N - hw, pa["k_range"][1] + 1)):
                    kb = int(round(k + dk))
                    if kb - hw < 0 or kb + hw >= N:
                        continue
                    if set(ch[k - hw:k + hw + 1]) & set(ch[kb - hw:kb + hw + 1]):
                        badk[pi].setdefault(ta["id"], set()).add(k)
        nbad = sum(len(v) for v in badk[pi].values())
        if nbad:
            print("plane %s: %d (track, wire) positions excluded for channel sharing across tracks (%s)" % (
                "UVW"[pi], nbad, {k: len(v) for k, v in badk[pi].items()}))
    collide = [set() for _ in range(3)]

    # ---- self-calibration on the collection plane: centroid(tick) is linear
    calib = []
    lines = {}       # track id -> (tick_start, mm_per_tick)
    for t in tr["tracks"]:
        pW = t["planes"][2]; g = pg[2]
        L = abs(t["x_end_mm"] - t["x_start_mm"])
        tk0, tk1 = tick_pred(t["x_start_mm"]), tick_pred(t["x_end_mm"])
        xs, ws, qs = [], [], []
        for s in range(int(max(tk0, tbin) // tps) + 1, int(min(tk1, tbin + ncol - 1) // tps)):
            tc = (s + 0.5) * tps
            x = t["x_start_mm"] + tr["drift_sign"] * (-1) * 0  # placeholder
            x = t["x_start_mm"] - np.sign(t["x_start_mm"] - t["x_end_mm"]) * (tc - tk0) * v_tick
            w = pW["w0"] + pW["slope_wire_per_mm"] * abs(x - t["x_start_mm"]); k = int(round(w))
            rows = [F["row_of"].get(g["channels"][j], -1) if 0 <= j < len(g["idents"]) else -1 for j in range(k - hw, k + hw + 1)]
            if rows[hw] < 0:
                continue
            j0, j1 = s * tps - tbin, (s + 1) * tps - tbin
            if j0 < 0 or j1 > ncol:
                continue
            Q = np.array([fr[r, j0:j1].sum() if r >= 0 else 0.0 for r in rows]); y = np.where(Q > 0, Q, 0)
            if y.sum() <= 0:
                continue
            xs.append(tc); ws.append(k + (np.arange(-hw, hw + 1) * y).sum() / y.sum()); qs.append(y.sum())
        xs, ws, qs = np.array(xs), np.array(ws), np.array(qs)
        good = qs > a.min_frac * np.median(qs) if len(qs) else np.zeros(0, bool)
        if good.sum() < 20:
            print("track %d: no collection signal for calibration (%d slices)" % (t["id"], good.sum())); continue
        # trim the ends, fit w = A + B * tick
        idx = np.where(good)[0][a.drop_edge_slices: -a.drop_edge_slices or None]
        B, A = np.polyfit(xs[idx], ws[idx], 1)
        slope_pred = pW["slope_wire_per_mm"] * v_tick          # wires per tick
        tick_start_fit = (pW["w0"] - A) / B
        mm_per_tick_fit = B / pW["slope_wire_per_mm"]
        lines[t["id"]] = (tick_start_fit, mm_per_tick_fit)
        q_tick = np.median(qs[good]) / tps
        calib.append(dict(track=t["id"], tick_start_pred=tk0, tick_start_fit=tick_start_fit, dtick=tick_start_fit - tk0,
                          mm_per_tick_pred=v_tick, mm_per_tick_fit=mm_per_tick_fit, ratio=mm_per_tick_fit / v_tick,
                          slope_pred=slope_pred, slope_fit=B, n_slices=int(good.sum()),
                          q_per_tick=q_tick, q_per_tick_truth=tr["charge_e_per_mm"] * v_tick))
    write_tsv(a.out + "_calib.tsv", calib)
    if not lines:
        print("no calibrated tracks", file=sys.stderr); return 1
    print("%s %s %s: %d tracks calibrated; dtick median %+.1f, mm/tick fit/pred median %.4f, q/tick %.0f (truth %.0f)" % (
        a.det, os.path.basename(a.frames[0]), F["tag"], len(calib), np.median([c["dtick"] for c in calib]),
        np.median([c["ratio"] for c in calib]), np.median([c["q_per_tick"] for c in calib]), calib[0]["q_per_tick_truth"]))

    # ---- profiles
    rows_out = []
    R = {k: [] for k in ("bid", "plane", "slice", "t_ns", "x_mm", "phase", "extent", "y", "vm", "vt", "y_raw", "vm_raw", "neg_frac",
                         "r0", "r1", "r2", "r3", "t0", "t1", "t2", "t3", "n_missing", "off")}
    prof_y = {pi: [] for pi in range(3)}
    for t in tr["tracks"]:
        if t["id"] not in lines:
            continue
        tick_start, mm_tick = lines[t["id"]]
        sgn = -np.sign(t["x_start_mm"] - t["x_end_mm"])
        L = abs(t["x_end_mm"] - t["x_start_mm"])
        tick_end = tick_start + L / mm_tick
        for p in t["planes"]:
            pi = p["plane"]; g = pg[pi]
            if t["id"] in collide[pi]:
                continue
            ext = abs(p["slope_wire_per_mm"]) * mm_tick * tps
            s_lo = int(max(tick_start, tbin) // tps) + 1 + a.drop_edge_slices
            s_hi = int(min(tick_end, tbin + ncol - 1) // tps) - a.drop_edge_slices
            for s in range(s_lo, s_hi):
                tc = (s + 0.5) * tps
                d = (tc - tick_start) * mm_tick                      # drift distance travelled since x_start
                x = t["x_start_mm"] + sgn * d
                w = p["w0"] + p["slope_wire_per_mm"] * d; k = int(round(w)); ph = w - k
                if k - hw < 0 or k + hw >= len(g["idents"]) or k in badk[pi].get(t["id"], ()):
                    continue
                rows = [F["row_of"].get(g["channels"][j], -1) for j in range(k - hw, k + hw + 1)]
                if rows[hw] < 0:
                    continue
                j0, j1 = s * tps - tbin, (s + 1) * tps - tbin
                if j0 < 0 or j1 > ncol:
                    continue
                Q = np.array([fr[r, j0:j1].sum() if r >= 0 else 0.0 for r in rows], float)
                y = Q.copy() if a.no_clip else np.where(Q > 0, Q, 0.0)
                if y.sum() <= 0 or (y > 0).sum() < 2:
                    continue
                n = np.arange(-hw, hw + 1)
                mu = (n * y).sum() / y.sum(); vm = ((n - mu) ** 2 * y).sum() / y.sum()
                n0 = int(round(mu)); dd = np.abs(n - n0)
                r = [y[dd == 0].sum(), y[dd == 1].sum(), y[dd == 2].sum(), y[dd >= 3].sum()]
                # truth-centred: the true position within the slice runs from ph - ext/2 to ph + ext/2
                vt = ((n - ph) ** 2 * y).sum() / y.sum()
                dt = np.abs(n)
                tt = [y[dt == 0].sum(), y[dt == 1].sum(), y[dt == 2].sum(), y[dt >= 3].sum()]
                yr = Q.sum(); vmr = ((n - (n * Q).sum() / yr) ** 2 * Q).sum() / yr if yr > 0 else np.nan
                t_ns = abs(x - x_W) / v_mm_ns
                for kk, vv in zip(R.keys(), (t["id"], pi, s, t_ns, x, ph, ext, y.sum(), vm, vt, yr, vmr, -Q[Q < 0].sum() / y.sum(),
                                             r[0], r[1], r[2], r[3], tt[0], tt[1], tt[2], tt[3], sum(1 for rr in rows if rr < 0),
                                             mu - ph)):
                    R[kk].append(vv)
                prof_y[pi].append((ph, y))
    R = {k: np.array(v) for k, v in R.items()}
    if len(R["bid"]) == 0:
        print("no profiles", file=sys.stderr); return 1
    write_tsv(a.out + "_rows.tsv", [dict(zip(R.keys(), vals)) for vals in zip(*[R[k].tolist() for k in R])][:200000])
    print("  %d profiles (U/V/W %d/%d/%d); extent per slice U/V/W %s wires; drift %.0f-%.0f us" % (
        len(R["bid"]), *(int((R["plane"] == pi).sum()) for pi in range(3)),
        ["%.3f" % np.median(R["extent"][R["plane"] == pi]) if (R["plane"] == pi).any() else "-" for pi in range(3)],
        R["t_ns"].min() * 1e-3, R["t_ns"].max() * 1e-3))

    model = dict(DT=tr["lar"]["DT_mm2_ns"], c=(0.0, 0.0, 0.0), nsigma=4.0, path="truth:" + os.path.basename(a.truth))
    rng = np.random.default_rng(a.seed)
    pitch = tr["pitch_mm"]

    def bin_and_fit(sel, label, phase=None):
        out_bins, out_fit, joint = [], [], []
        for Pi in range(3):
            s = sel & (R["plane"] == Pi)
            if s.sum() < 20:
                continue
            t = R["t_ns"][s]; y = R["y"][s]; vm = R["vm"][s]; vt = R["vt"][s]; r0 = R["r0"][s]; ext = R["extent"][s]; b = R["bid"][s]
            o = np.argsort(t); cy = np.cumsum(y[o]) / y.sum()
            edges = [t[o][0]] + [t[o][np.searchsorted(cy, k / a.nbins)] for k in range(1, a.nbins)] + [t[o][-1] + 1]
            tb = np.clip(np.digitize(t, edges) - 1, 0, a.nbins - 1)
            ub, binv = np.unique(b, return_inverse=True); nb = len(ub)
            acc = np.zeros((nb, a.nbins, 6))       # y*vm, y, y*t, r0, y*vt, y*ext
            np.add.at(acc, (binv, tb), np.column_stack([y * vm, y, y * t, r0, y * vt, y * ext]))

            def solve(A):
                Y = A[:, 1]; ok = Y > 0
                rm = np.sqrt(A[:, 0] / np.where(ok, Y, 1)); tm = A[:, 2] / np.where(ok, Y, 1)
                sh = A[:, 3] / np.where(ok, Y, 1); rt = np.sqrt(A[:, 4] / np.where(ok, Y, 1)); ex = A[:, 5] / np.where(ok, Y, 1)
                res = np.full((a.nbins, 3), np.nan)
                for ib in range(a.nbins):
                    if not ok[ib]:
                        continue
                    sme, _ = _bisect(lambda g: apparent_rms(g, ex[ib], None, phase), rm[ib], 0.02, 3.0)
                    ssh, _ = _bisect(lambda g: -ring_shares(g, ex[ib], None, phase)[0], -sh[ib], 0.02, 3.0)
                    str_, _ = _bisect(lambda g: truth_rms(g, ex[ib], phase), rt[ib], 0.02, 3.0)
                    res[ib] = (sme, ssh, str_)
                return res * pitch[Pi], tm, rm * pitch[Pi], sh, rt * pitch[Pi], ex, Y
            full = acc.sum(axis=0)
            sig, tm, rmm, shm, rtm, exm, Ym = solve(full)
            boots = np.full((a.nboot, a.nbins, 3), np.nan)
            for ib_ in range(a.nboot):
                pick = rng.integers(0, nb, nb)
                boots[ib_] = solve(acc[pick].sum(axis=0))[0]
            err2 = np.nanstd(boots ** 2, axis=0)
            for ie, est in enumerate(("rms", "share", "truth")):
                sig2 = sig[:, ie] ** 2; e2 = np.maximum(err2[:, ie], 0.05 * sig2)
                sig_mod = np.sqrt(2 * model["DT"] * tm) / pitch[Pi]
                for ib in range(a.nbins):
                    if not np.isfinite(sig2[ib]):
                        continue
                    out_bins.append(dict(label=label, est=est, plane="UVW"[Pi], tbin=ib, t_us=tm[ib] * 1e-3, n_prof=int((tb == ib).sum()),
                                         q=Ym[ib], rms_meas_mm=rmm[ib], rms_pred_mm=apparent_rms(sig_mod[ib], exm[ib], None, phase) * pitch[Pi],
                                         sig_model_mm=sig_mod[ib] * pitch[Pi], sig2_eff_mm2=sig2[ib], sig2_err=e2[ib],
                                         sig2_naive_mm2=(sig_mod[ib] * pitch[Pi]) ** 2 + rmm[ib] ** 2 - (apparent_rms(sig_mod[ib], exm[ib], None, phase) * pitch[Pi]) ** 2,
                                         extent=exm[ib], ceiling=0, centre_share=shm[ib], rms_truth_mm=rtm[ib]))
                good = np.isfinite(sig2)
                A = np.column_stack([2 * tm[good], np.ones(good.sum())]); w = 1.0 / e2[good]
                beta, cov, chi2 = wlsq(A, sig2[good], w)
                out_fit.append(fitrow(label, est, "UVW"[Pi], beta[0], np.sqrt(cov[0, 0]), beta[1], np.sqrt(cov[1, 1]), chi2, good.sum() - 2, model, Pi))
                joint.append((est, Pi, tm[good], sig2[good], e2[good]))
        for est in ("rms", "share", "truth"):
            js = [j for j in joint if j[0] == est]
            if len(js) < 2:
                continue
            tt = np.concatenate([j[2] for j in js]); ss = np.concatenate([j[3] for j in js]); ee = np.concatenate([j[4] for j in js])
            pl = np.concatenate([np.full(len(j[2]), j[1]) for j in js])
            A = np.column_stack([2 * tt] + [(pl == Pi).astype(float) for Pi in range(3) if any(j[1] == Pi for j in js)])
            beta, cov, chi2 = wlsq(A, ss, 1.0 / ee)
            col = 1
            for Pi in range(3):
                if not any(j[1] == Pi for j in js):
                    continue
                out_fit.append(fitrow(label, est, "UVW"[Pi] + "(joint)", beta[0], np.sqrt(cov[0, 0]), beta[col], np.sqrt(cov[col, col]),
                                      chi2, len(tt) - A.shape[1], model, Pi)); col += 1
        return out_bins, out_fit

    base = np.ones(len(R["bid"]), bool)
    bins, fits = bin_and_fit(base, "all")
    if a.phase_split:
        # the phase used for BINNING: the truth, optionally smeared to the resolution the
        # data's fitted trajectory has (doc 47 sec 8.4).  The model each window is inverted
        # against is restricted to the same window -- the uniform-phase model manufactures a
        # centre/boundary contrast out of a phase-independent sigma (sec 8.1).
        phm = R["phase"] + (R["off"] if a.phase_src == "centroid" else 0.0)
        phm = phm - np.round(phm)
        if a.phase_jitter > 0:
            phm = phm + rng.normal(0.0, a.phase_jitter, len(phm))
            phm = phm - np.round(phm)
        for name, lo, hi in (("full", -0.5, 0.5), ("q1", -0.5, -0.25), ("q2", -0.25, 0.0),
                             ("q3", 0.0, 0.25), ("q4", 0.25, 0.5), ("centre", -0.25, 0.25)):
            b_, f_ = bin_and_fit(base & (phm >= lo) & (phm < hi), "phase:" + name, (lo, hi)); bins += b_; fits += f_
        b_, f_ = bin_and_fit(base & (np.abs(phm) >= 0.25), "phase:edge", (0.25, 0.5)); bins += b_; fits += f_
        prow = []
        for Pi in range(3):
            m_ = R["plane"] == Pi
            if m_.sum() < 20:
                continue
            prow.append(dict(plane="UVW"[Pi], n=int(m_.sum()), jitter=a.phase_jitter,
                             phase_src=a.phase_src,
                             rms_off_wire=float(np.sqrt(np.average(R["off"][m_] ** 2, weights=np.maximum(R["y"][m_], 0)))),
                             med_abs_off=float(np.median(np.abs(R["off"][m_])))))
        write_tsv(a.out + "_phase.tsv", prow)
        print("  centroid - truth (wires, charge weighted): %s" % " ".join(
            "%s %.3f" % (r_["plane"], r_["rms_off_wire"]) for r_ in prow))
    write_tsv(a.out + "_bins.tsv", bins); write_tsv(a.out + "_fit.tsv", fits)

    # shape (all t): measured ring shares about the own centroid vs the binned Gaussian at the fitted sigma
    shape = []
    for Pi in range(3):
        s = R["plane"] == Pi
        if s.sum() < 20:
            continue
        y = R["y"][s]
        meas = np.array([R[k][s].sum() for k in ("r0", "r1", "r2", "r3")]) / y.sum()
        tmeas = np.array([R[k][s].sum() for k in ("t0", "t1", "t2", "t3")]) / y.sum()
        ext = np.average(R["extent"][s], weights=y); tmean = np.average(R["t_ns"][s], weights=y)
        fr_ = {f_["est"]: f_ for f_ in fits if f_["label"] == "all" and f_["plane"] == "UVW"[Pi]}
        if "rms" not in fr_:
            continue
        sig_fit = np.sqrt(max(2 * fr_["rms"]["DT_json"] * tmean + fr_["rms"]["c2_mm2"], 1e-6)) / pitch[Pi]
        sig_sh = np.sqrt(max(2 * fr_["share"]["DT_json"] * tmean + fr_["share"]["c2_mm2"], 1e-6)) / pitch[Pi]
        sig_mod = np.sqrt(2 * model["DT"] * tmean) / pitch[Pi]
        mfit = ring_shares(sig_fit, ext); msh = ring_shares(sig_sh, ext); mmod = ring_shares(sig_mod, ext, model["nsigma"])
        excess = (meas[2] + meas[3]) - (mfit[2] + mfit[3])
        shape.append(dict(plane="UVW"[Pi], sig_fit_pitch=sig_fit, sig_share_pitch=sig_sh, sig_model_pitch=sig_mod, extent=ext,
                          meas_centre=meas[0], meas_pm1=meas[1], meas_pm2=meas[2], meas_beyond=meas[3],
                          gaus_fit_centre=mfit[0], gaus_fit_pm1=mfit[1], gaus_fit_pm2=mfit[2], gaus_fit_beyond=mfit[3],
                          gaus_model_centre=mmod[0], gaus_model_pm1=mmod[1], gaus_model_pm2=mmod[2], gaus_model_beyond=mmod[3],
                          gaus_share_centre=msh[0], gaus_share_pm1=msh[1], gaus_share_pm2=msh[2], gaus_share_beyond=msh[3],
                          U_stack_rms=float(np.abs(meas - mfit).sum()), U_stack_share=float(np.abs(meas - msh).sum()),
                          U_stack_model=float(np.abs(meas - mmod).sum()),
                          tail_excess_share=excess, tail_excess_over_pm1=excess / max(mfit[1], 1e-9),
                          truth_centre=tmeas[0], truth_pm1=tmeas[1], truth_pm2=tmeas[2], truth_beyond=tmeas[3]))
    write_tsv(a.out + "_shape.tsv", shape)

    if a.kernel:
        krows, srows, ksum = [], [], []
        for Pi in range(3):
            if not prof_y[Pi]:
                continue
            ph = np.array([p_[0] for p_ in prof_y[Pi]]); Y = np.array([p_[1] for p_ in prof_y[Pi]])
            edges = np.linspace(-0.5, 0.5, a.phase_bins + 1); pb = np.clip(np.digitize(ph, edges) - 1, 0, a.phase_bins - 1)
            for ib in range(a.phase_bins):
                m = pb == ib
                if m.sum() == 0:
                    continue
                K = Y[m].sum(axis=0); K = K / K.sum()
                krows.append(dict(plane="UVW"[Pi], phase_bin=ib, phase_lo=edges[ib], phase_hi=edges[ib + 1], n_prof=int(m.sum()),
                                  **{"k_%+d" % n: K[i] for i, n in enumerate(range(-hw, hw + 1))}))
            # super-resolution stack about the true position: u = n - phase, 0.1-pitch bins
            du = 0.1; ue = np.arange(-hw - 0.5, hw + 0.5 + 1e-9, du); S = np.zeros(len(ue) - 1)
            for i, n in enumerate(range(-hw, hw + 1)):
                u = n - ph; ub = np.clip(np.digitize(u, ue) - 1, 0, len(S) - 1)
                np.add.at(S, ub, Y[:, i])
            S = S / (S.sum() * du); uc = 0.5 * (ue[1:] + ue[:-1])
            rms_sr = np.sqrt((S * uc ** 2).sum() * du)
            tmean = np.average(R["t_ns"][R["plane"] == Pi], weights=R["y"][R["plane"] == Pi])
            sig_diff = np.sqrt(2 * model["DT"] * tmean) / pitch[Pi]
            for i in range(len(S)):
                srows.append(dict(plane="UVW"[Pi], u_centre=uc[i], S=S[i]))
            ksum.append(dict(plane="UVW"[Pi], rms_sr_pitch=rms_sr, rms_sr_mm=rms_sr * pitch[Pi],
                             rms_boxcar=1 / np.sqrt(12), sig_diff_pitch=sig_diff, rms_boxcar_diff=np.hypot(1 / np.sqrt(12), sig_diff),
                             excess_over_boxcar_diff_pitch=np.sqrt(max(rms_sr ** 2 - 1 / 12 - sig_diff ** 2, 0)),
                             excess_mm=np.sqrt(max(rms_sr ** 2 - 1 / 12 - sig_diff ** 2, 0)) * pitch[Pi],
                             t_mean_us=tmean * 1e-3, n_prof=len(ph)))
        write_tsv(a.out + "_kernel.tsv", krows); write_tsv(a.out + "_kernel_sr.tsv", srows); write_tsv(a.out + "_ksum.tsv", ksum)

    # console summary
    print("  fits sig_eff^2 = 2 DT t + c^2   (model DT %.2f cm2/s, c = 0)" % (model["DT"] * 1e7))
    print("    %-12s %-6s %-10s %16s %16s %10s" % ("label", "est", "plane", "DT_eff [cm2/s]", "c_eff [mm]", "chi2/ndf"))
    for f_ in fits:
        print("    %-12s %-6s %-10s %7.2f +- %5.2f %7.3f +- %5.3f %6.1f/%d" % (f_["label"], f_["est"], f_["plane"], f_["DT_eff_cm2s"], f_["DT_err"], f_["c_eff_mm"], f_["c_err"], f_["chi2"], f_["ndf"]))
    for s_ in shape:
        print("  shape %s: meas own-centroid centre/+-1/+-2/beyond %.3f %.3f %.3f %.3f | truth-centred %.3f %.3f %.3f %.3f | gaus@share %.3f %.3f %.3f %.3f | gaus@model %.3f %.3f %.3f %.3f" % (
            s_["plane"], s_["meas_centre"], s_["meas_pm1"], s_["meas_pm2"], s_["meas_beyond"], s_["truth_centre"], s_["truth_pm1"], s_["truth_pm2"], s_["truth_beyond"],
            s_["gaus_share_centre"], s_["gaus_share_pm1"], s_["gaus_share_pm2"], s_["gaus_share_beyond"], s_["gaus_model_centre"], s_["gaus_model_pm1"], s_["gaus_model_pm2"], s_["gaus_model_beyond"]))
    print("  -> %s_{calib,rows,bins,fit,shape%s}.tsv" % (a.out, ",kernel,kernel_sr,ksum" if a.kernel else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

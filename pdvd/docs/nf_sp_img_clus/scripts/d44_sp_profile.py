#!/usr/bin/env python3
"""doc pdvd/44 sec 3 -- the SP-frame cross-check of the effective transverse width.

Doc 42 sec 8.9 measured the first-neighbour shape directly in the signal-processing
output (`run_nf_sp_evt.sh -R`, frame_gauss<anode>_<evt>.npy) but left no committed
tool.  This one reads the same frame at the fitted trajectory's (channel, slice)
positions of a tracking-stm.root, applies the doc-44 estimator (own-centroid rms,
prolonged segments only, unfolded at the extent the predicted profile implies) to
BOTH the SP waveform charge and the ctpc charge (T_proj_data) of the same profiles,
and reports the two effective widths side by side.  What the fit must describe is
the ctpc value; the SP value says how much of it is already in the waveform.

Channel scheme: T_proj_data's `channel` is the rank of the raw channel id among all
channels of that plane over all anodes (PdvdMagnifyTrackingVisitor::chan_scheme),
rebuilt here from the wires file.  Time: `time_slice` = tick / nticks_per_slice; the
frame's tick origin is found by maximising the correlation of the two charges over
a small offset scan and reported.

Usage:
  d44_sp_profile.py --root work/039252_2_d42fit/tracking-stm.root --frames /home/xqian/tmp/d44sp/039252_2 \\
                    --anode 4 [--wires protodunevd-wires-larsoft-v7-uvwfit.json.bz2] [--which-anode]
  --which-anode   only report which anode holds the fitted points of each status-0 block
"""
import argparse, bz2, glob, json, os, sys
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d44_sigma_fit import DET, MIN_DRIFT_NS, load_model, sigma_model_pitch, unfold, apparent_rms, _bisect  # noqa: E402

WIRES = "/home/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v7-uvwfit.json.bz2"


def chan_scheme(wires_path):
    st = json.load(bz2.open(wires_path))["Store"]
    wires = [w["Wire"] for w in st["wires"]]; planes = [p["Plane"] for p in st["planes"]]
    faces = [f["Face"] for f in st["faces"]]; anodes = [a["Anode"] for a in st["anodes"]]
    chans = [set(), set(), set()]
    ch2anode = {}
    for a in anodes:
        for fidx in a["faces"]:
            for p, pidx in enumerate(faces[fidx]["planes"][:3]):
                for wi in planes[pidx]["wires"]:
                    w = wires[wi]        # the wire index is assigned at load; every listed wire has one
                    chans[p].add(w["channel"]); ch2anode[w["channel"]] = a["ident"]
    rank = [sorted(c) for c in chans]
    base = [0, len(rank[0]), len(rank[0]) + len(rank[1])]
    return rank, base, ch2anode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", help="dir with frame_gauss<anode>_<evt>.npy + channels_gauss<anode>_<evt>.npy")
    ap.add_argument("--anode", type=int)
    ap.add_argument("--tag", default="gauss")
    ap.add_argument("--wires", default=WIRES)
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--model-json", default=None)
    ap.add_argument("--max-advance", type=float, default=0.25)
    ap.add_argument("--halfwidth", type=int, default=3)
    ap.add_argument("--ticks-per-slice", type=int, default=4)
    ap.add_argument("--which-anode", action="store_true")
    ap.add_argument("--tick-offset", type=int, default=None,
                    help="doc 47 sec 9: pin the frame tick origin instead of scanning for it. "
                         "The scan maximises the correlation of the frame charge with the ctpc "
                         "charge, which is 0.9 for gauss but only 0.1-0.4 for rawdecon (that "
                         "frame is not the ROI-cleaned charge) -- so a rawdecon run must be "
                         "pinned to the offset its own event's GAUSS run found.")
    ap.add_argument("--tsv", default=None, help="append per-profile rows here (for pooling over events)")
    ap.add_argument("--rings-tsv", default=None,
                    help="doc 47 sec 9: append PER-PROFILE ring shares to <path>_ctpc.tsv and "
                         "<path>_sp.tsv (columns as d47_sim_transverse_profile.py's _rows.tsv, "
                         "so d47_tail_isolation.py --rows reads either).  The ctpc window is "
                         "sparse (a cell with no blob is an exact zero) and the SP window is "
                         "dense, so the two files answer whether imaging gates the >=2-wire tail. "
                         "Additive: --tsv output is unchanged.")
    a = ap.parse_args()
    det = DET[a.det]; model = load_model(a.model_json or det["json"])
    rank, base, ch2anode = chan_scheme(a.wires)
    f = uproot.open(a.root)
    r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "rr", "ndf", "status", "x"], library="np")
    d = f["T_proj_data"].arrays(library="np")
    blocks = [int(c) for c in d["cluster_id"][0]]
    if a.which_anode:
        for i, blk in enumerate(blocks):
            m = r["ndf"] == blk
            if m.sum() < 5 or int(r["status"][m][0]) != 0:
                continue
            raw = [rank[2][int(w) - base[2]] for w in r["pw"][m] if 0 <= int(w) - base[2] < len(rank[2])]
            an = [ch2anode[c] for c in raw]
            u, n = np.unique(an, return_counts=True)
            print("block %d: %d points, anodes %s" % (blk, m.sum(), dict(zip(u.tolist(), n.tolist()))))
        return 0
    fr = sorted(glob.glob(os.path.join(a.frames, "frame_%s%d_*.npy" % (a.tag, a.anode))))
    chf = sorted(glob.glob(os.path.join(a.frames, "channels_%s%d_*.npy" % (a.tag, a.anode))))
    if not fr or not chf:
        print("no frame for anode", a.anode, "in", a.frames, file=sys.stderr); return 1
    frame = np.load(fr[0]); chans = np.load(chf[0])
    row_of = {int(c): i for i, c in enumerate(chans)}
    nt = frame.shape[1]
    hw = a.halfwidth
    # ---- collect profiles: for each (block, plane, slice) the ctpc window and the raw channels
    prof = []   # (plane, slice, t_ns, adv, y_ctpc[7], yh[7], rows[7] or None)
    for i, blk in enumerate(blocks):
        m = r["ndf"] == blk
        if m.sum() < 5 or int(r["status"][m][0]) != 0:
            continue
        ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
        ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
        q = np.asarray(list(d["charge"][0][i]), float); qp = np.asarray(list(d["charge_pred"][0][i]), float)
        pl = np.digitize(ch, det["bounds"])
        o = np.argsort(-r["rr"][m]); pt = r["pt"][m][o]; px = r["x"][m][o]
        it = np.gradient(pt)
        for Pi, key in enumerate(("pu", "pv", "pw")):
            pw_ = r[key][m][o]; iw = np.gradient(pw_)
            adv = np.abs(iw) / np.maximum(np.abs(it), 1e-9)
            mp = pl == Pi
            cellmap = {(int(c), int(t)): k for k, (c, t) in enumerate(zip(ch[mp], ts[mp]))}
            Q = q[mp]; QP = qp[mp]
            for s_ in np.unique(ts[mp]):
                dt = np.abs(pt - s_); near = dt <= 0.6
                if not near.any():
                    continue
                j0 = int(np.argmin(dt))
                wc = 0.5 * (pw_[near].min() + pw_[near].max()); w0 = int(np.round(wc))
                idx = np.array([cellmap.get((w0 + k, int(s_)), -1) for k in range(-hw, hw + 1)])
                y = np.zeros(2 * hw + 1); yh = np.zeros(2 * hw + 1); have = idx >= 0
                y[have] = np.where(Q[idx[have]] > 0, Q[idx[have]], 0); yh[have] = QP[idx[have]]
                gidx = [w0 + k - base[Pi] for k in range(-hw, hw + 1)]
                if min(gidx) < 0 or max(gidx) >= len(rank[Pi]):
                    continue
                raw = [rank[Pi][g] for g in gidx]
                if any(c not in row_of for c in raw):
                    continue                     # window not in this anode's frame
                rows = np.array([row_of[c] for c in raw])
                absx = abs(float(px[j0]))
                t_ns = max(MIN_DRIFT_NS, (det["x_anode"] - absx) * 10.0 / det["v"] * 1e3)
                prof.append((Pi, int(s_), t_ns, float(adv[near].max()), y, yh, rows))
    if not prof:
        print("no profiles in anode", a.anode); return 1
    print("%s: %d profiles in anode %d frame (%d rows x %d ticks)" % (a.root, len(prof), a.anode, *frame.shape))

    # ---- tick origin: scan offsets, maximise correlation of ctpc charge vs frame charge
    def frame_sum(rows, s_, off, raw=False):
        t0 = s_ * a.ticks_per_slice + off
        if t0 < 0 or t0 + a.ticks_per_slice > nt:
            return None
        v = frame[rows, t0:t0 + a.ticks_per_slice].sum(axis=1)
        return v if raw else np.where(v > 0, v, 0.0)
    best = None
    if a.tick_offset is not None:
        best = (a.tick_offset, float("nan"))
    for off in range(-12, 13) if a.tick_offset is None else ():
        xs, ys = [], []
        for (Pi, s_, t_ns, adv, y, yh, rows) in prof[:: max(1, len(prof) // 400)]:
            v = frame_sum(rows, s_, off)
            if v is None:
                continue
            xs.append(y); ys.append(v)
        if not xs:
            continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        c = np.corrcoef(X, Y)[0, 1] if X.std() > 0 and Y.std() > 0 else -1
        if best is None or c > best[1]:
            best = (off, c)
    off, corr = best
    print("  tick offset %d (corr %.3f); slice = %d ticks" % (off, corr, a.ticks_per_slice))

    # ---- estimator per plane, prolonged only
    n = np.arange(-hw, hw + 1, dtype=float)
    out = []
    rings = {"ctpc": [], "sp": [], "spabs": []}
    for Pi in range(3):
        acc = dict(vm=0.0, wm=0.0, vs=0.0, ws=0.0, vp=0.0, wp=0.0, t=0.0, k=0,
                   rm=np.zeros(4), rs=np.zeros(4))
        for (P_, s_, t_ns, adv, y, yh, rows) in prof:
            if P_ != Pi or adv >= a.max_advance:
                continue
            v = frame_sum(rows, s_, off)
            if v is None or y.sum() <= 0 or yh.sum() <= 0 or v.sum() <= 0 or (y > 0).sum() < 2:
                continue
            for key, arr in (("m", y), ("s", v), ("p", yh)):
                mu = np.average(n, weights=arr); var = np.average((n - mu) ** 2, weights=arr)
                acc["v" + key] += arr.sum() * var; acc["w" + key] += arr.sum()
                if key in ("m", "s"):
                    dd = np.abs(n - np.round(mu))
                    acc["r" + key] += [arr[dd == 0].sum(), arr[dd == 1].sum(), arr[dd == 2].sum(), arr[dd >= 3].sum()]
            acc["t"] += y.sum() * t_ns; acc["k"] += 1
            if a.rings_tsv:
                # neg_frac, as d47_sim_transverse_profile.py defines it: the magnitude of the
                # clipped-away negative charge over the surviving positive sum.  Doc 47 sec 9.5
                # compares the data's rawdecon shares with the simulation's, and both are read
                # under this clipping -- if one frame swings more negative than the other, the
                # shares are not comparable, so the number has to be reported next to them.
                vr = frame_sum(rows, s_, off, raw=True)
                nf = float(-vr[vr < 0].sum() / v.sum()) if v.sum() > 0 else np.nan
                for key, arr, ng in (("ctpc", y, 0.0), ("sp", v, nf), ("spabs", np.abs(vr), nf)):
                    dd = np.abs(n - np.round(np.average(n, weights=arr)))
                    rings[key].append((0, Pi, s_, t_ns, arr.sum(), adv,
                                       *[arr[dd == k_].sum() for k_ in (0, 1, 2)], arr[dd >= 3].sum(), ng))
            if a.tsv:
                out.append((a.root, Pi, s_, t_ns, y.sum(), np.average((n - np.average(n, weights=y)) ** 2, weights=y),
                            v.sum(), np.average((n - np.average(n, weights=v)) ** 2, weights=v),
                            yh.sum(), np.average((n - np.average(n, weights=yh)) ** 2, weights=yh)))
        if acc["k"] < 10:
            continue
        pitch = det["pitch"][Pi]
        rm = np.sqrt(acc["vm"] / acc["wm"]); rs = np.sqrt(acc["vs"] / acc["ws"]); rp = np.sqrt(acc["vp"] / acc["wp"])
        tm = acc["t"] / acc["wm"]
        sm = sigma_model_pitch(model, tm, Pi, pitch)
        s_ctpc, ext, hitc = unfold(sm, rp, rm, model["nsigma"])
        # measured sigma of the SP profile at the same extent
        s_sp, _ = _bisect(lambda g: apparent_rms(g, ext), rs, 0.02, 3.0)
        rmm = acc["rm"] / acc["rm"].sum(); rss = acc["rs"] / acc["rs"].sum()
        print("  %s  n=%4d  <t>=%5.0f us  rms ctpc %.2f  SP %.2f  pred %.2f mm | sigma_eff ctpc %.2f  SP %.2f  model %.2f mm (extent %.2f%s)" % (
            "UVW"[Pi], acc["k"], tm / 1e3, rm * pitch, rs * pitch, rp * pitch,
            s_ctpc * pitch, s_sp * pitch, sm * pitch, ext, " CEILING" if hitc else ""))
        print("        rings centre/+-1/+-2/beyond  ctpc %.3f %.3f %.3f %.3f   SP %.3f %.3f %.3f %.3f" % (*rmm, *rss))
    if a.rings_tsv:
        for key, rws in rings.items():
            if not rws:
                continue
            path = a.rings_tsv + "_" + key + ".tsv"
            new_ = not os.path.exists(path)
            with open(path, "a") as fo:
                if new_:
                    fo.write("bid\tplane\tslice\tt_ns\ty\tadv\tr0\tr1\tr2\tr3\tneg_frac\n")
                for row in rws:
                    fo.write("\t".join("%.6g" % v for v in row) + "\n")
            print("  -> %s (%d profiles)" % (path, len(rws)))
    if a.tsv and out:
        new = not os.path.exists(a.tsv)
        with open(a.tsv, "a") as fo:
            if new:
                fo.write("root\tplane\tslice\tt_ns\ty\tvm\tysp\tvsp\tyh\tvp\n")
            for row in out:
                fo.write("\t".join(str(v) if isinstance(v, (str, int)) else "%.6g" % v for v in row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

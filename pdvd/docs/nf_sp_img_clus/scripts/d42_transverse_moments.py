#!/usr/bin/env python3
"""doc pdvd/42 sec 8 -- is the measured charge WIDER than the model at each point,
or the SAME width but DISPLACED?  The two have different causes and different fixes.

  smearing too narrow  -> at every point the measured transverse profile is wider
                          than the predicted one, symmetric about the trajectory.
                          The remedy is the smearing model.
  trajectory error     -> at every point the two profiles have the SAME width but
                          the measured one is displaced by delta; only the ENSEMBLE
                          profile (about the trajectory) is wider.  The remedy is
                          the trajectory, and widening a smearing constant would
                          just paper over it.

Per (block, plane, time slice) this computes, over the cells of that slice:
  w_meas, w_pred  charge-weighted wire centroids of measured and predicted charge
  var about OWN centroid, for each -- the per-point width, immune to displacement
  offset = w_meas - w_pred  -- the per-point displacement
and reports the charge-weighted aggregates plus the autocorrelation of the offset
along the track (a real trajectory wander is correlated point to point; centroid
noise is not).

Usage:  d42_transverse_moments.py --det pdvd work/*_d42fit/tracking-stm.root
"""
import argparse, sys
import numpy as np
import uproot

PITCH = {"pdvd": (7.65, 7.65, 5.10), "sbnd": (3.0, 3.0, 3.0)}
BOUNDS = {"pdvd": (3808, 7616), "sbnd": (3968, 7936)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--det", required=True, choices=PITCH)
    ap.add_argument("--status", type=int, default=0)
    ap.add_argument("--max-span", type=float, default=1.0,
                    help="only use slices where the trajectory spans fewer than this many wires")
    ap.add_argument("--halfwidth", type=float, default=3.0, help="cells each side of the trajectory")
    a = ap.parse_args()
    P, bounds = PITCH[a.det], BOUNDS[a.det]
    S = {p: dict(vm=0.0, vp=0.0, wm=0.0, wp=0.0, off=[], q=0.0, qtot=0.0) for p in "UVW"}
    corr = {p: [] for p in "UVW"}
    for path in a.roots:
        try:
            f = uproot.open(path); ks = {k.split(";")[0] for k in f.keys()}
        except Exception as ex:
            print("skip", path, ex, file=sys.stderr); continue
        if "T_proj_data" not in ks:
            continue
        r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "rr", "ndf", "status"], library="np")
        d = f["T_proj_data"].arrays(library="np")
        for i, blk in enumerate([int(c) for c in d["cluster_id"][0]]):
            m = r["ndf"] == blk
            if m.sum() < 5 or (a.status >= 0 and int(r["status"][m][0]) != a.status):
                continue
            ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
            q = np.asarray(list(d["charge"][0][i]), float)
            qp = np.asarray(list(d["charge_pred"][0][i]), float)
            pl = np.digitize(ch, bounds)
            o = np.argsort(-r["rr"][m]); pt = r["pt"][m][o]
            for Pi, key in enumerate(("pu", "pv", "pw")):
                p_ = "UVW"[Pi]
                mp = pl == Pi
                if mp.sum() == 0:
                    continue
                pw_ = r[key][m][o]
                cw = ch[mp].astype(float); ct = ts[mp]; Q = q[mp]; QP = qp[mp]
                S[p_]["qtot"] += Q[Q > 0].sum()
                seq = []
                for s_ in np.unique(ct):
                    near = np.abs(pt - s_) <= 0.6
                    if not near.any():
                        continue
                    wlo, whi = pw_[near].min(), pw_[near].max()
                    if whi - wlo > a.max_span:
                        continue
                    wc = 0.5 * (wlo + whi)
                    sel = (ct == s_) & (np.abs(cw - wc) <= a.halfwidth)
                    y = Q[sel]; yh = QP[sel]; x = cw[sel]
                    y = np.where(y > 0, y, 0.0)
                    if y.sum() <= 0 or yh.sum() <= 0 or sel.sum() < 2:
                        continue
                    mu_m = np.average(x, weights=y); mu_p = np.average(x, weights=yh)
                    S[p_]["vm"] += y.sum() * np.average((x - mu_m) ** 2, weights=y)
                    S[p_]["vp"] += yh.sum() * np.average((x - mu_p) ** 2, weights=yh)
                    S[p_]["wm"] += y.sum(); S[p_]["wp"] += yh.sum()
                    S[p_]["off"].append((mu_m - mu_p, y.sum()))
                    S[p_]["q"] += y.sum()
                    seq.append(mu_m - mu_p)
                if len(seq) > 12:
                    v = np.asarray(seq) - np.mean(seq)
                    den = np.dot(v, v)
                    if den > 0:
                        corr[p_].append([np.dot(v[:-k], v[k:]) / den for k in (1, 2, 5, 10)])
    print("%s: transverse width about each profile's OWN centroid, and the displacement" % a.det.upper())
    print("  (span cut %s wires, +-%.1f cells; charge-weighted)" % (a.max_span, a.halfwidth))
    print("  %-6s %12s %12s %12s %12s %12s" % ("plane", "rms_meas", "rms_pred", "quad.diff",
                                               "rms offset", "|offset| med"))
    for p in "UVW":
        s = S[p]
        if s["wm"] <= 0:
            continue
        rm = np.sqrt(s["vm"] / s["wm"]) * P["UVW".index(p)]
        rp = np.sqrt(s["vp"] / s["wp"]) * P["UVW".index(p)]
        off = np.array([o for o, _ in s["off"]]); ow = np.array([w for _, w in s["off"]])
        ro = np.sqrt(np.average(off ** 2, weights=ow)) * P["UVW".index(p)]
        print("  %-6s %12.3f %12.3f %12.3f %12.3f %12.3f    [%.0f %% of the plane's charge used]"
              % (p, rm, rp, np.sqrt(max(rm ** 2 - rp ** 2, 0)), ro,
                 np.median(np.abs(off)) * P["UVW".index(p)], 100 * s["q"] / max(s["qtot"], 1)))
    print("  autocorrelation of the per-slice displacement along the track (lag 1 / 2 / 5 / 10):")
    for p in "UVW":
        if corr[p]:
            c = np.mean(np.array(corr[p]), axis=0)
            print("    %s  %+.3f  %+.3f  %+.3f  %+.3f   (n=%d tracks)" % (p, c[0], c[1], c[2], c[3], len(corr[p])))


if __name__ == "__main__":
    sys.exit(main())

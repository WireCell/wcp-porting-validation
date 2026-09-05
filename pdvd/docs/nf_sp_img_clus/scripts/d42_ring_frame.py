#!/usr/bin/env python3
"""doc pdvd/42 sec 7.4 -- reference-frame cross-check on the wire-ring table.

d42_shape_diag.py measures each cell's distance to the trajectory from its
NEAREST FITTED POINT.  For a prolonged segment that point is chosen mostly by
time, so its wire coordinate can sit up to ~0.8 wires from where the track
actually crosses that slice -- and PDVD is 83 % prolonged against SBND's 15 %
(sec 7.5), so the bias is not symmetric between the two detectors.

This script repeats the centre / first-neighbour ratios in a frame that cannot
have that bias: the trajectory is densified by linear interpolation to a
<= 0.2-cell step, and each cell's transverse distance is the minimum over the
densified points that share its time slice -- i.e. distance to the trajectory
AT THE CELL'S OWN TIME, not to a discrete fitted point.

Usage:
  d42_ring_frame.py --det pdvd work/*_d42fit/tracking-stm.root
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
    ap.add_argument("--step", type=float, default=0.2, help="densification step [cells]")
    ap.add_argument("--max-foff", type=float, default=1.1,
                    help="keep only blocks whose W-plane charge beyond 1.5 pitches is below this")
    ap.add_argument("--tsv", help="also write the measured/predicted ring shares here")
    a = ap.parse_args()
    P, bounds = PITCH[a.det], BOUNDS[a.det]
    acc = {p: {k: np.zeros(3) for k in ("y", "h")} for p in "UVW"}
    nb = 0
    for path in a.roots:
        try:
            f = uproot.open(path)
            ks = {k.split(";")[0] for k in f.keys()}
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
            ch0 = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            if a.max_foff < 1.0:
                pl0 = np.digitize(ch0, bounds)
                q0 = np.asarray(list(d["charge"][0][i]), float)
                mW = pl0 == 2
                if mW.sum() == 0:
                    continue
                o0 = np.argsort(-r["rr"][m])
                pwW = r["pw"][m][o0]; ptW = r["pt"][m][o0]
                dd = np.min(np.abs(ch0[mW][:, None] - pwW[None, :]), axis=1)
                lv0 = q0[mW] > 0
                if q0[mW][lv0].sum() <= 0 or \
                   q0[mW][lv0 & (dd > 1.5)].sum() / q0[mW][lv0].sum() > a.max_foff:
                    continue
            nb += 1
            ch = np.asarray(list(d["channel"][0][i]), dtype=np.int64)
            ts = np.asarray(list(d["time_slice"][0][i]), dtype=np.int64)
            q = np.asarray(list(d["charge"][0][i]), float)
            qp = np.asarray(list(d["charge_pred"][0][i]), float)
            pl = np.digitize(ch, bounds); live = q > 0
            o = np.argsort(-r["rr"][m]); pt = r["pt"][m][o]
            for Pi, key in enumerate(("pu", "pv", "pw")):
                mp = pl == Pi
                if mp.sum() == 0:
                    continue
                pw_ = r[key][m][o]
                seg = []
                for j in range(len(pw_) - 1):
                    n = int(max(abs(pw_[j + 1] - pw_[j]), abs(pt[j + 1] - pt[j])) / a.step) + 2
                    seg.append(np.column_stack([np.linspace(pw_[j], pw_[j + 1], n),
                                                np.linspace(pt[j], pt[j + 1], n)]))
                dense = np.vstack(seg) if seg else np.column_stack([pw_, pt])
                bt = np.round(dense[:, 1]).astype(np.int64)
                order = np.argsort(bt); bt = bt[order]; bw = dense[order, 0]
                uniq, start = np.unique(bt, return_index=True)
                end = np.append(start[1:], len(bt))
                idx = {int(u): (s, e) for u, s, e in zip(uniq, start, end)}
                cw = ch[mp]; ct = ts[mp]; Q = q[mp]; QP = qp[mp]; lv = live[mp]
                for s_ in np.unique(ct):
                    if int(s_) not in idx:
                        continue
                    lo, hi = idx[int(s_)]
                    W = bw[lo:hi]
                    sc = ct == s_
                    ring = np.digitize(np.abs(cw[sc][:, None] - W[None, :]).min(1), [0.5, 1.5])
                    for k in range(3):
                        kk = ring == k
                        acc["UVW"[Pi]]["y"][k] += Q[sc][kk & lv[sc]].sum()
                        acc["UVW"[Pi]]["h"][k] += QP[sc][kk].sum()
    print("%s: %d status-%d blocks; predicted/measured by wire ring (share of measured)"
          % (a.det, nb, a.status))
    rows = []
    for p in "UVW":
        y = acc[p]["y"]; h = acc[p]["h"]
        print("  %s  centre %.2f (%.0f %%)   first neighbour %.2f (%.0f %%)   beyond %.2f (%.0f %%)"
              % (p, h[0] / max(y[0], 1), 100 * y[0] / y.sum(), h[1] / max(y[1], 1),
                 100 * y[1] / y.sum(), h[2] / max(y[2], 1), 100 * y[2] / y.sum()))
        # the shape statistic used by d42_wire_filter_toy.py: first neighbour as a
        # fraction of (centre + first neighbour).  Insensitive to how much charge
        # a fused cluster puts beyond 1.5 pitches, which is why it, not the raw
        # share, is what the toy is compared against.
        rows.append((p, y[0], y[1], h[0], h[1],
                     y[1] / max(y[0] + y[1], 1e-9), h[1] / max(h[0] + h[1], 1e-9)))
    print("  shape r = 1st/(centre+1st):  " + "   ".join(
        "%s meas %.3f pred %.3f" % (r[0], r[5], r[6]) for r in rows))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("det\tplane\tmeas_centre\tmeas_first\tpred_centre\tpred_first\tr_meas\tr_pred\n")
            for r in rows:
                fh.write("%s\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%.5f\t%.5f\n" % ((a.det,) + r))
        print("  -> %s" % a.tsv)


if __name__ == "__main__":
    sys.exit(main())

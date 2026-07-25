#!/usr/bin/env python3
"""True vs fitted dQ/dx for one STM fit block, from a MC track_com_*.root.

Reads the MC-mode output of wire-cell-sbnd-magnify-tracking-convert (-f1),
where `true_dQ` is the SimEnergyDeposit charge of the paired true particle
accumulated onto each fitted point (see dump_truth_sed.C) and `rec_dQ` is the
fitted charge.

The two dQ/dx are formed with DIFFERENT denominators, and that is the point:

  * fitted dQ/dx = rec_dQ / rec_dx -- rec_dx is the fitted cell's path length.
  * true   dQ/dx = true_dQ / true_dx -- true_dx is the length of TRUE track
    whose deposits landed in that cell.  Dividing the truth by rec_dx instead
    (what the GUI did before the true_dx branch existed, and what the
    prototype's wire-cell-track-com leaves the consumer to do) reports the
    assignment cell's length fluctuation as a physics fluctuation: the cell is
    a 3-D Voronoi region of a fitted path that zig-zags at the 0.6 cm scale, so
    it holds 0.2-5.3 cm of true track while rec_dx stays near 0.62 cm.  On
    evt18 block 150 that inflated the truth scatter from rms/median 0.09 to
    0.55 and made the truth beyond the fit's end read as a 396 ke/cm Bragg
    peak on a 50 ke/cm MIP.

Three caveats this script enforces rather than hides:

  * true_dx == 0 means no truth point chose that fitted point; the true dQ/dx
    is undefined and the point is dropped from statistics and plot.
  * a cell with true_dx >> rec_dx averages the truth over more track than the
    fitted point represents, which flattens real structure (a Bragg peak on a
    stopping track).  --max-dx-ratio cuts on it; the summary always reports the
    distribution.
  * com_dis is the fitted-to-truth pairing distance.  If its median is not of
    order a cm, the two are not in the same frame and no dQ/dx number below
    means anything; the script prints it first and refuses to quote a ratio
    above --max-com-dis.

Files from a converter without the true_dx branch fall back to rec_dx and to
--edge 1, which is the pre-true_dx behaviour.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 stmfit_mc_compare.py -f showcase-stmfit-mc-evt18/track_com_18.root \
      -b 150 -o showcase-stmfit-mc-evt18/dqdx_mc_evt18_blk150.png
"""
import argparse

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def arr(v):
    return np.asarray(list(v), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="track_com_*.root (MC mode)")
    ap.add_argument("-b", "--block", type=int, required=True,
                    help="block id = cluster_id*10 + pass")
    ap.add_argument("-o", "--output", help="output PNG")
    ap.add_argument("--edge", type=int, default=None,
                    help="fitted points dropped at each end; default 0 when the "
                         "file has true_dx (the pile-up it guarded against is "
                         "cured by the correct denominator) and 1 when it does not")
    ap.add_argument("--max-dx-ratio", type=float, default=None,
                    help="drop points whose true_dx exceeds this multiple of "
                         "rec_dx (the truth is averaged over too much track "
                         "there); default keep all")
    ap.add_argument("--max-com-dis", type=float, default=5.0,
                    help="median pairing distance above which nothing is quoted")
    ap.add_argument("--half-drift", type=float, default=200.0,
                    help="SBND anode |x| in cm; drift distance is this minus |x|")
    ap.add_argument("--lifetime-ms", type=float, default=6.0,
                    help="electron lifetime for the reference attenuation only "
                         "(the runner scripts' LIFETIME)")
    ap.add_argument("--drift-speed", type=float, default=1.563,
                    help="mm/us, the runner scripts' DRIFTSPEED (reference only)")
    args = ap.parse_args()

    d = uproot.open(args.file)["T_rec"].arrays(library="np")
    cid = arr(d["rec_cluster_id"][0]).astype(int)
    if args.block not in cid:
        raise SystemExit(f"block {args.block} not in {args.file} (have {list(cid)})")
    k = int(np.where(cid == args.block)[0][0])

    x = arr(d["rec_x"][0][k])
    L = arr(d["rec_L"][0][k])
    dx = arr(d["rec_dx"][0][k])
    rec = arr(d["rec_dQ"][0][k])
    tru = arr(d["true_dQ"][0][k])
    com = arr(d["com_dis"][0][k])
    n = len(L)

    has_true_dx = "true_dx" in d
    if has_true_dx:
        tdx = arr(d["true_dx"][0][k])
    else:
        tdx = dx.copy()
        print("NOTE: no true_dx branch -- the truth is divided by rec_dx, which "
              "mixes the assignment cell length into the true dQ/dx")
    if args.edge is None:
        args.edge = 0 if has_true_dx else 1

    med_com = float(np.median(com))
    print(f"{args.file} block {args.block}: {n} fitted points, "
          f"path {L.max():.1f} cm, drift x [{x.min():.1f}, {x.max():.1f}] cm")
    print(f"pairing distance com_dis: median {med_com:.2f} cm, "
          f"p90 {np.percentile(com, 90):.2f} cm, max {com.max():.2f} cm")
    if med_com > args.max_com_dis:
        raise SystemExit(f"median pairing distance {med_com:.2f} cm > "
                         f"{args.max_com_dis} cm: the fit does not follow this "
                         f"true particle, so no dQ/dx comparison is meaningful")

    core = np.zeros(n, dtype=bool)
    core[args.edge:n - args.edge if args.edge else n] = True
    good = core & (dx > 0) & (tdx > 0)
    if args.max_dx_ratio is not None:
        good &= tdx <= args.max_dx_ratio * dx

    rq = rec / dx / 1e3          # ke/cm
    tq = tru / np.where(tdx > 0, tdx, np.nan) / 1e3

    print(f"true path assigned: {tdx.sum():.2f} cm over {n} points "
          f"(sum rec_dx {dx.sum():.2f} cm); "
          f"{(tdx <= 0).sum()} points got no truth")
    okd = tdx > 0
    print("true_dx / rec_dx: median {:.2f}, p90 {:.2f}, max {:.2f}"
          .format(*np.percentile((tdx / dx)[okd], [50, 90, 100])))
    if args.edge:
        print(f"edge points dropped: {args.edge} at each end")
    if args.max_dx_ratio is not None:
        print(f"points dropped by --max-dx-ratio {args.max_dx_ratio}: "
              f"{int((core & okd & (tdx > args.max_dx_ratio * dx)).sum())}")
    print(f"fitted dQ/dx: median {np.median(rq[good]):.1f} ke/cm, "
          f"rms/median {rq[good].std()/np.median(rq[good]):.3f}, "
          f"{(rec[good] < 0).sum()} of {good.sum()} points negative")
    print(f"true   dQ/dx: median {np.median(tq[good]):.1f} ke/cm, "
          f"rms/median {tq[good].std()/np.median(tq[good]):.3f}, "
          f"p5-p95 {np.percentile(tq[good],5):.1f}-{np.percentile(tq[good],95):.1f}")
    # Length-normalised, because the two denominators are different lengths:
    # this is the ratio of mean dQ/dx, not of collected charge.
    print(f"integral over the core: rec {rec[good].sum():.4e} e- / {dx[good].sum():.2f} cm, "
          f"true {tru[good].sum():.4e} e- / {tdx[good].sum():.2f} cm, "
          f"mean-dQ/dx ratio "
          f"{(rec[good].sum()/dx[good].sum())/(tru[good].sum()/tdx[good].sum()):.3f}")

    # profile along the path, and against |x| (drift distance) for attenuation
    print("\n L [cm]      n   fitted [ke/cm]  true [ke/cm]   rec/true   <x> [cm]")
    edges = np.linspace(0, L.max(), 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = good & (L >= lo) & (L < hi)
        if m.sum() < 3:
            continue
        print(f" {lo:6.1f}-{hi:6.1f} {m.sum():4d}   {np.median(rq[m]):8.1f}      "
              f"{np.median(tq[m]):8.1f}     "
              f"{(rec[m].sum()/dx[m].sum())/(tru[m].sum()/tdx[m].sum()):7.3f}   "
              f"{np.mean(x[m]):7.1f}")

    # Against drift distance, which is what an uncorrected electron lifetime
    # would show up in: true dQ is the charge AT THE DEPOSIT, so fitted/true is
    # the whole deposit-to-fitted-charge factor.  |x| is measured from the
    # cathode at x = 0, the anodes sit at +-`--half-drift`.
    drift = args.half_drift - np.abs(x)
    print(f"\n drift [cm]   n   fitted/true (integral)  median point ratio")
    cent, val = [], []
    for lo in np.arange(0, args.half_drift, 40.0):
        m = good & (tru > 0) & (drift >= lo) & (drift < lo + 40)
        if m.sum() < 4:
            continue
        ratio = (rec[m].sum() / dx[m].sum()) / (tru[m].sum() / tdx[m].sum())
        cent.append(np.mean(drift[m]))
        val.append(ratio)
        print(f" {lo:4.0f}-{lo+40:4.0f}  {m.sum():4d}      {ratio:8.3f}            "
              f"{np.median((rec[m]/dx[m]) / (tru[m]/tdx[m])):8.3f}")
    if len(cent) > 2:
        slope = np.polyfit(cent, val, 1)[0]
        span = float(np.max(drift[good]) - np.min(drift[good]))
        print(f"binned-ratio slope {slope:+.5f} per cm -> {slope*span:+.3f} over the "
              f"{span:.0f} cm of drift sampled")
        # what an uncorrected lifetime would look like, for comparison
        t_us = drift[good] / (args.drift_speed / 10.0)     # mm/us -> cm/us
        att = np.exp(-t_us / (args.lifetime_ms * 1e3))
        print(f"for reference, uncorrected tau={args.lifetime_ms} ms at "
              f"v={args.drift_speed} mm/us would give exp(-t/tau) = "
              f"{att.min():.3f} at {drift[good].max():.0f} cm vs {att.max():.3f} at "
              f"{drift[good].min():.0f} cm (far/near {att.min()/att.max():.3f})")

    if not args.output:
        return

    # With true_dx the point-level truth is already smooth (it is the truth's own
    # local dQ/dx, length-weighted over the cell); the running median is kept so
    # the fitted curve, which is genuinely noisy, stays readable next to it.
    def running(vals, w=15):
        out = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            lo, hi = max(0, i - w // 2), min(len(vals), i + w // 2 + 1)
            out[i] = np.median(vals[lo:hi])
        return out

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(L[good], tq[good], "o", ms=2.5, color="tab:blue", alpha=.3)
    ax[0].plot(L[good], rq[good], "o", ms=2.5, color="tab:red", alpha=.3)
    ax[0].plot(L[good], running(tq[good]), "-", lw=2, color="tab:blue",
               label=("true (SimEnergyDeposit)/true_dx" if has_true_dx else
                      "true (SimEnergyDeposit)/rec_dx") + ", running median")
    ax[0].plot(L[good], running(rq[good]), "-", lw=2, color="tab:red",
               label="fitted, running median")
    edge_m = (~good) & np.isfinite(tq)
    if edge_m.any():
        ax[0].plot(L[edge_m], tq[edge_m], "x", ms=7, color="grey",
                   label="excluded points")
        ax[0].plot(L[edge_m], rq[edge_m], "x", ms=7, color="grey")
    ax[0].axhline(50, ls="--", lw=1, color="k", label="50 ke/cm MIP")
    ax[0].set_ylabel("dQ/dx [ke/cm]")
    ax[0].set_ylim(0, max(120, np.percentile(rq[good], 99) * 1.2))
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{args.file.split('/')[-1]} block {args.block}: "
                    f"fitted vs true dQ/dx  (pairing median {med_com:.2f} cm)")
    ax[0].grid(alpha=.3)

    r = np.full(n, np.nan)
    m = good & (tru > 0)
    r[m] = (rec[m] / dx[m]) / (tru[m] / tdx[m])
    ax[1].plot(L[m], r[m], "o", ms=2.5, color="tab:green", alpha=.3)
    ax[1].plot(L[m], running(r[m]), "-", lw=2, color="tab:green",
               label="running median")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].axhline(1, ls="--", lw=1, color="k")
    ax[1].set_ylim(0, 2)
    ax[1].set_xlabel("L along the fitted path [cm]")
    ax[1].set_ylabel("fitted / true")
    ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=110)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

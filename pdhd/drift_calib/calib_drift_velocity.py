#!/usr/bin/env python
"""PDHD drift-velocity calibration from anode->cathode crossing tracks.

Idea
----
The reconstruction maps slice time to drift-x with a fixed
``drift_speed = 1.6 mm/us`` (x = xorig + dirx*(t + time_offset)*drift_speed).
A clean anode->cathode crosser drifts a *known* physical distance, so its
reconstructed drift x-span obeys

    S  ==  drift_speed * dt_drift  ==  D_U * (v_reco / v_true)

where ``D_U`` is the U(first-induction)-plane -> cathode-surface distance (the
recon anchors at the W collection plane, but both the anode-end and cathode-end
recon positions carry the same v_reco/v_true factor, so their separation maps to
the physical U->cathode distance; xorig cancels in max-min).  Hence

    v_true = v_reco * D_U / S .

We measure ``S`` from the 100+ imaging events, using both drift sides
(group02 = APAs 0+2 = TPC0/2 ; group13 = APAs 1+3 = TPC1/3).

The MABC clustering JSON ``x`` IS the drift coordinate (cm): cathode ~ 0, anode
at -352 (group02) / +352 (group13).

Cautions (per request): over-clustering inflates a cluster's span, gaps deflate
it.  We therefore (a) require one end to sit at the cathode and reject clusters
that shoot far past it (CPA / cross-volume merges), and (b) estimate ``S`` from
the *upper edge / mode* of the span distribution, where genuine full crossers
pile up at the kinematic maximum -- neither the over-merge tail (above S) nor the
short/broken tracks (below S) drive the estimate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
# Drift speed the INPUT mabc data was reconstructed with (NOT necessarily the
# current config).  The original calibration ran on data made at 1.6 and yielded
# ~1.55; the work/ dumps have since been re-clustered at 1.565, so to re-derive
# from the current data pass `--v-reco 1.565` (closure / self-consistency check).
V_RECO_DEFAULT = 1.6  # mm/us

# Geometry (cm).  Plane x-positions read from the protodunehd-wires-larsoft-v1
# store (NOT the apa_cpa centerline): the three WIRE planes sit one pitch
# (~4.9 mm) apart, U/V/W at |x| = 352.22 / 352.71 / 353.20.  time2drift anchors
# at the W collection plane.  Cathode drift-facing surface |x| = 0.5*cpa_thick.
CPA_THICK = 0.3175         # 1/8"
X_U = 352.22               # U (first-induction) plane, store
X_W = 353.20               # W (collection) plane, store -- xorig anchor
X_CATH_SURF = 0.5 * CPA_THICK                  # 0.159
D_U = X_U - X_CATH_SURF                         # 352.06 cm  (primary reference)
D_W = X_W - X_CATH_SURF                         # 353.04 cm  (W-plane systematic; U-W~0.98cm)


def load_groups(zip_path):
    """Return {grp: (x, cluster_id, run, evt)} from one mabc-all-apa.zip."""
    members = {0: "0-clustering-group02.json", 1: "0-clustering-group13.json"}
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for grp, fn in members.items():
            if fn not in names:
                continue
            d = json.loads(z.read(names[fn]))
            out[grp] = (np.asarray(d["x"], dtype=np.float64),
                        np.asarray(d["cluster_id"], dtype=np.int64),
                        d.get("runNo"), d.get("eventNo"))
    return out


def event_clusters(zip_path, min_pts):
    """Yield per-cluster geometry for one event (both drift groups)."""
    rows = []
    for grp, (x, cid, run, evt) in load_groups(zip_path).items():
        for c in np.unique(cid):
            xs = x[cid == c]
            if xs.size < min_pts:
                continue
            xlo, xhi = float(xs.min()), float(xs.max())
            if grp == 0:                      # anode at negative x, cathode at 0
                anode_reach = -xlo
                cath_coord = xhi              # cathode-end coordinate (~0 if it reaches cathode)
            else:                             # anode at positive x
                anode_reach = xhi
                cath_coord = xlo
            rows.append(dict(grp=grp, run=run, evt=evt, cl=int(c), npts=int(xs.size),
                             xlo=xlo, xhi=xhi, span=xhi - xlo,
                             anode_reach=anode_reach, cath_coord=cath_coord))
    return rows


def is_full_crosser(r, anode_reach_min, cath_win):
    """A genuine anode->cathode crosser: the anode end sits at the kinematic edge
    (anode_reach >= anode_reach_min) AND the cathode end lands within `cath_win`
    cm of the cathode (x=0) on EITHER side -- a small drift-overshoot when
    v_reco != v_true, in either direction.  A SYMMETRIC window (not the earlier
    side-flipped one) so the selection is unbiased whether v_reco is above or
    below v_true (e.g. the closure run on data re-clustered near v_true).  This
    rejects near-anode fragments (|cath_coord| >> 0 on the anode side) and CPA /
    cross-volume over-merges (|cath_coord| >> 0 past the cathode)."""
    return r["anode_reach"] >= anode_reach_min and abs(r["cath_coord"]) <= cath_win


def span_pileup(sp, lo=340.0, hi=385.0, binw=3.0):
    """Robust full-drift span = the PILE-UP (mode) of the span distribution.

    All genuine full crossers share the same true drift extent, so their
    reconstructed spans pile at one value; over-merged / large-overshoot tracks
    add a HIGH tail that drags the *median* (so the median is NOT used here -- it
    is tail-sensitive and shifts with cluster composition between reprocessings,
    while the pile-up is stable).  Peak of a lightly-smoothed histogram.
    """
    sp = np.asarray(sp)
    edges = np.arange(lo, hi + binw, binw)
    h, _ = np.histogram(sp, bins=edges)
    if h.sum() == 0:
        return float(np.median(sp))
    hs = np.convolve(h, np.ones(3) / 3.0, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[int(np.argmax(hs))])


def report(tag, full, v_reco):
    """Print pile-up-based velocity for a set of full crossers."""
    if not full:
        print(f"[{tag}] no full crossers")
        return None
    sp = np.array([r["span"] for r in full])
    ov = np.array([(r["cath_coord"] if r["grp"] == 0 else -r["cath_coord"])
                   for r in full])           # signed cathode overshoot past x=0
    S = span_pileup(sp)                       # robust full-drift span (pile-up)
    v_span_DU = v_reco * D_U / S
    v_span_DW = v_reco * D_W / S
    print(f"[{tag}] N={len(full)}  span pile-up={S:.1f} cm (median={np.median(sp):.1f}, "
          f"tail-inflated)  cath-overshoot median={np.median(ov):.1f} cm")
    print(f"      v_true = {v_span_DU:.4f} mm/us   (pile-up / D_U={D_U:.1f}, U-plane ref)")
    print(f"             = {v_span_DW:.4f} mm/us   (pile-up / D_W={D_W:.1f}, W systematic)")
    return dict(sp=sp, S=S, v_span_DU=v_span_DU, v_span_DW=v_span_DW)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work",
                    default="/home/xqian/work/scratch_wcgpu1/toolkit-dev/"
                            "wcp-porting-img/pdhd/work",
                    help="dir holding <run>_<evt>/mabc-all-apa.zip")
    ap.add_argument("--min-pts", type=int, default=50,
                    help="skip clusters with fewer stepped points")
    ap.add_argument("--anode-reach", type=float, default=340.0,
                    help="anode-most |x| (cm) must be >= this (full-drift reach)")
    ap.add_argument("--cath-win", type=float, default=25.0,
                    help="cathode end must be within this many cm of x=0 (either side)")
    ap.add_argument("--v-reco", type=float, default=V_RECO_DEFAULT,
                    help="drift_speed the INPUT data was reconstructed with "
                         "(1.6 for the original calibration; 1.565 for the "
                         "re-clustered work/ dumps -> closure check)")
    ap.add_argument("--plot", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "drift_velocity_calib.png"))
    args = ap.parse_args()
    v_reco = args.v_reco

    zips = sorted(glob.glob(os.path.join(args.work, "*", "mabc-all-apa.zip")))
    if not zips:
        sys.exit(f"no mabc-all-apa.zip under {args.work}")
    print(f"[geom] D_U(U->cathode)={D_U:.3f} cm   D_W(W->cathode)={D_W:.3f} cm   "
          f"v_reco={v_reco} mm/us")
    print(f"[data] {len(zips)} events under {args.work}\n")

    all_rows = []
    n_evt_ok = 0
    for zp in zips:
        try:
            rows = event_clusters(zp, args.min_pts)
        except Exception as e:                     # noqa: BLE001
            print(f"[warn] {zp}: {e}", file=sys.stderr)
            continue
        all_rows.extend(rows)
        n_evt_ok += 1
    print(f"[data] parsed {n_evt_ok}/{len(zips)} events, "
          f"{len(all_rows)} clusters (>= {args.min_pts} pts)\n")

    full = [r for r in all_rows
            if is_full_crosser(r, args.anode_reach, args.cath_win)]
    print(f"[select] {len(full)} full A-C crossers "
          f"(anode_reach>={args.anode_reach} cm, |cathode-end|<={args.cath_win} cm)\n")

    g0 = [r for r in full if r["grp"] == 0]
    g1 = [r for r in full if r["grp"] == 1]
    report("TPC0/2 (group02)", g0, v_reco)
    report("TPC1/3 (group13)", g1, v_reco)
    res = report("ALL", full, v_reco)

    if res is not None:
        print(f"\n==> calibrated drift velocity v_true = {res['v_span_DU']:.4f} mm/us "
              f"(U-plane pile-up); W systematic {res['v_span_DW']:.4f}.")
        print(f"==> input data reconstructed at {v_reco} mm/us  ->  "
              f"~{100*(v_reco/res['v_span_DU']-1):.1f}% off.")
        _make_plot(args.plot, np.array([r["span"] for r in g0]),
                   np.array([r["span"] for r in g1]), res["S"], v_reco)


def _make_plot(path, g0, g1, S, v_reco):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # noqa: BLE001
        print(f"[plot] skipped ({e})", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    allspan = np.concatenate([g0, g1]) if g0.size or g1.size else np.array([])
    edges = np.arange(330, 392, 3.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(allspan, bins=edges, histtype="stepfilled", alpha=0.15,
            label=f"all (N={allspan.size})", color="k")
    ax.hist(g0, bins=edges, histtype="step", label=f"TPC0/2 (N={g0.size})", color="C0")
    ax.hist(g1, bins=edges, histtype="step", label=f"TPC1/3 (N={g1.size})", color="C1")
    ax.axvline(D_U, ls="--", color="g", label=f"D_U={D_U:.1f} (U->cath)")
    ax.axvline(D_W, ls=":", color="m", label=f"D_W={D_W:.1f} (W->cath)")
    ax.axvline(S, ls="-", color="r", lw=1.5, label=f"S(pile-up)={S:.1f}")
    ax.set_xlabel("full-crosser drift x-span [cm]")
    ax.set_ylabel("clusters")
    ax.set_title(f"PDHD A-C crosser x-span (data @ {v_reco} mm/us)  ->  "
                 f"v_true={v_reco*D_U/S:.4f} mm/us")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()

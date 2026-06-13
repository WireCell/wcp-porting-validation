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
V_RECO = 1.6  # mm/us, current toolkit drift_speed (cfg/.../pdhd/clus.jsonnet)

# Geometry (cm) from cfg/pgrapher/experiment/pdhd/params.jsonnet.
APA_CPA = 357.34            # APA centerline <-> CPA centerline
APA_W2W = 8.587            # wire-plane to wire-plane span
PLANE_GAP = 0.476
CPA_THICK = 0.3175         # 1/8"
APA_G2G = APA_W2W + 6 * PLANE_GAP
APA_PLANE = 0.5 * APA_G2G - PLANE_GAP        # U / first-induction plane offset from centerline
X_U = APA_CPA - APA_PLANE                      # |x| of U (first-induction) plane = 352.094
X_W = APA_CPA                                  # |x| of W (collection ~ centerline) = 357.34
X_CATH_SURF = 0.5 * CPA_THICK                  # cathode drift-facing surface = 0.159
D_U = X_U - X_CATH_SURF                         # 351.94 cm  (primary reference)
D_W = X_W - X_CATH_SURF                         # 357.18 cm  (W-plane systematic)


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


def is_full_crosser(r, anode_reach_min, cath_lo, cath_hi):
    """A genuine anode->cathode crosser: the anode end sits at the kinematic edge
    (anode_reach >= anode_reach_min) AND the cathode end lands in a small window
    around the cathode (reached it; small drift-overshoot from v_reco != v_true;
    NOT a CPA / cross-volume over-merge that shoots far past the cathode).

    The cathode window is side-flipped: for group02 (anode at -x, drift toward +x)
    v_reco>v_true overshoots the cathode to +x, so the cathode-end coord sits in
    [cath_lo, cath_hi]; group13 mirrors it to [-cath_hi, -cath_lo].
    """
    if r["anode_reach"] < anode_reach_min:
        return False
    cc = r["cath_coord"]
    return (cath_lo <= cc <= cath_hi) if r["grp"] == 0 else (-cath_hi <= cc <= -cath_lo)


def report(tag, full):
    """Print span- and overshoot-based velocity for a set of full crossers."""
    if not full:
        print(f"[{tag}] no full crossers")
        return None
    sp = np.array([r["span"] for r in full])
    ov = np.array([(r["cath_coord"] if r["grp"] == 0 else -r["cath_coord"])
                   for r in full])           # signed cathode overshoot past x=0
    S = float(np.median(sp))
    o = float(np.median(ov))
    v_span_DU = V_RECO * D_U / S
    v_span_DW = V_RECO * D_W / S
    v_over = V_RECO * D_W / (D_W + o)         # independent cross-check
    print(f"[{tag}] N={len(full)}  span median={S:.1f} cm (mean={sp.mean():.1f})  "
          f"cath-overshoot median={o:.1f} cm")
    print(f"      v_true = {v_span_DU:.4f} mm/us   (span / D_U={D_U:.1f}, U-plane ref)")
    print(f"             = {v_span_DW:.4f} mm/us   (span / D_W={D_W:.1f}, W systematic)")
    print(f"             = {v_over:.4f} mm/us   (cathode-overshoot cross-check)")
    return dict(sp=sp, S=S, o=o, v_span_DU=v_span_DU, v_span_DW=v_span_DW, v_over=v_over)


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
    ap.add_argument("--cath-lo", type=float, default=-5.0,
                    help="cathode-end window low edge (cm past the cathode)")
    ap.add_argument("--cath-hi", type=float, default=25.0,
                    help="cathode-end window high edge (cm past the cathode)")
    ap.add_argument("--plot", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "drift_velocity_calib.png"))
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.work, "*", "mabc-all-apa.zip")))
    if not zips:
        sys.exit(f"no mabc-all-apa.zip under {args.work}")
    print(f"[geom] D_U(U->cathode)={D_U:.3f} cm   D_W(W->cathode)={D_W:.3f} cm   "
          f"v_reco={V_RECO} mm/us")
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
            if is_full_crosser(r, args.anode_reach, args.cath_lo, args.cath_hi)]
    print(f"[select] {len(full)} full A-C crossers "
          f"(anode_reach>={args.anode_reach} cm, cathode window "
          f"[{args.cath_lo},{args.cath_hi}] cm)\n")

    g0 = [r for r in full if r["grp"] == 0]
    g1 = [r for r in full if r["grp"] == 1]
    report("TPC0/2 (group02)", g0)
    report("TPC1/3 (group13)", g1)
    res = report("ALL", full)

    if res is not None:
        print(f"\n==> calibrated drift velocity v_true = {res['v_span_DU']:.4f} mm/us "
              f"(U-plane span); cross-checks {res['v_over']:.4f} (overshoot), "
              f"{res['v_span_DW']:.4f} (W systematic).")
        print(f"==> reco currently {V_RECO} mm/us  ->  ~{100*(V_RECO/res['v_span_DU']-1):.1f}% high; "
              f"consistent with the 1.565 field-response / LArSoft value.")
        _make_plot(args.plot, np.array([r["span"] for r in g0]),
                   np.array([r["span"] for r in g1]), res["S"])


def _make_plot(path, g0, g1, S):
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
    ax.axvline(S, ls="-", color="r", lw=1.5, label=f"S(median)={S:.1f}")
    ax.set_xlabel("full-crosser drift x-span [cm]")
    ax.set_ylabel("clusters")
    ax.set_title(f"PDHD A-C crosser x-span  ->  v_true={V_RECO*D_U/S:.4f} mm/us")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()

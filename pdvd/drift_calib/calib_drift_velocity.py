#!/usr/bin/env python
"""PDVD drift-velocity calibration from anode->cathode crossing tracks.

Same idea as the PDHD study (pdhd/drift_calib): a clean anode->cathode crosser
drifts a *known* physical distance, so its reconstructed drift x-span obeys

    S  ==  drift_speed * dt_drift  ==  D * (v_reco / v_true)   =>   v_true = v_reco * D / S

`x` in the MABC clustering JSON IS the drift coordinate (cm); cathode ~ 0, anode
at ∓341.6.  Both drift volumes are used: group0123 (anodes 0-3, anode at -x) and
group4567 (anodes 4-7, anode at +x).

Geometry (cm), from cfg/pgrapher/experiment/protodunevd/params.jsonnet + the
protodunevd-wires-larsoft-v5 store:

  * The U/V/W readout layers are at |x| = 341.51 / 341.53 / 341.55 -- only 0.2 mm
    apart in x, because the VD CRP is a PCB-strip anode (U/V/Z on stacked PCB
    layers), not spaced wire planes.  (PDHD's wire planes are ~4.9 mm = one pitch
    apart, U-W ~0.98 cm.)  time2drift anchors at the W collection layer (xorig,
    ident 2) = 341.55 cm; the anode reference is unambiguous (U-vs-W moves D by
    0.04 cm).  A shield grid (apa_plane, ~5.7 cm toward the drift) is a field grid
    the charge passes THROUGH, not a sensing plane.
  * Thick membrane cathode: cpa_thick = 50.8 mm, centered at x=0, so the
    drift-facing cathode SURFACE is at |x| = 2.54 cm.
  * D = |xorig| - |cathode surface| = 341.55 - 2.54 = 339.01 cm.

PDVD caution (per request): induction-plane signal-processing failures leave GAPS
in prolonged (drift-aligned) tracks.  An *interior* gap does not change a cluster's
x-span (max-min), but an *end* gap shortens the reach and a mid-track break can
split a crosser into two clusters (neither spanning the full drift).  We therefore
(a) select full crossers by requiring BOTH ends extreme (a split crosser is simply
dropped, not mis-measured), and (b) report the largest interior x-gap of the
selected crossers so any gap bias is visible rather than silent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

# Drift speed the INPUT mabc data was reconstructed with (NOT necessarily the
# current config).  The original calibration ran on data made at 1.6 and yielded
# ~1.57.  The work/ dumps have since been re-clustered at 1.57, so to re-derive
# from the current data pass `--v-reco 1.57`; the full-crosser pile-up then sits at
# D (~339 cm) and v_true comes back ~1.57 -- the closure / self-consistency check.
V_RECO_DEFAULT = 1.6  # mm/us

# Geometry (cm) -- params.jsonnet + protodunevd-wires-larsoft-v5 store.
X_W = 341.55              # W collection plane (xorig in time2drift; = apa_cpa)
CPA_THICK = 5.08          # thick membrane cathode (50.8 mm)
X_CATH_SURF = 0.5 * CPA_THICK   # cathode drift-facing surface |x| = 2.54
D = X_W - X_CATH_SURF     # 339.01 cm  (collection -> cathode surface)


def load_groups(zip_path):
    """Return {grp: (x, cluster_id, run, evt)} from one mabc-all-apa.zip.
    grp 0 = group0123 (anode at -x), grp 1 = group4567 (anode at +x)."""
    members = {0: "0-clustering-group0123.json", 1: "0-clustering-group4567.json"}
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


def max_gap(xs):
    """Largest interior gap along the drift coordinate of a cluster's points."""
    if xs.size < 2:
        return 0.0
    s = np.sort(xs)
    return float(np.max(np.diff(s)))


def event_clusters(zip_path, min_pts):
    rows = []
    for grp, (x, cid, run, evt) in load_groups(zip_path).items():
        for c in np.unique(cid):
            xs = x[cid == c]
            if xs.size < min_pts:
                continue
            xlo, xhi = float(xs.min()), float(xs.max())
            if grp == 0:                      # anode at -x, cathode at 0
                anode_reach = -xlo
                cath_coord = xhi
            else:                             # anode at +x
                anode_reach = xhi
                cath_coord = xlo
            rows.append(dict(grp=grp, run=run, evt=evt, cl=int(c), npts=int(xs.size),
                             xlo=xlo, xhi=xhi, span=xhi - xlo,
                             anode_reach=anode_reach, cath_coord=cath_coord,
                             max_gap=max_gap(xs)))
    return rows


def is_full_crosser(r, anode_reach_min, cath_lo, cath_hi):
    """Anode end at the kinematic edge AND cathode end in a small window around
    the cathode (reached it, small drift-overshoot; not a CPA/cross-volume merge).
    The window is side-flipped between the two drift volumes."""
    if r["anode_reach"] < anode_reach_min:
        return False
    cc = r["cath_coord"]
    return (cath_lo <= cc <= cath_hi) if r["grp"] == 0 else (-cath_hi <= cc <= -cath_lo)


def span_pileup(sp, lo=330.0, hi=375.0, binw=3.0):
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
    if not full:
        print(f"[{tag}] no full crossers")
        return None
    sp = np.array([r["span"] for r in full])
    ar = np.array([r["anode_reach"] for r in full])
    ov = np.array([(r["cath_coord"] if r["grp"] == 0 else -r["cath_coord"])
                   for r in full])           # signed cathode overshoot past x=0
    gaps = np.array([r["max_gap"] for r in full])
    S = span_pileup(sp)                       # robust full-drift span (pile-up)
    v_span = v_reco * D / S
    print(f"[{tag}] N={len(full)}  span pile-up={S:.1f} cm (median={np.median(sp):.1f}, "
          f"tail-inflated)  anode-reach median={np.median(ar):.1f} cm  "
          f"cath-overshoot median={np.median(ov):.1f} cm  "
          f"max interior-gap median={np.median(gaps):.1f} cm")
    print(f"      v_true = {v_span:.4f} mm/us   (pile-up span / D={D:.1f})")
    return dict(sp=sp, S=S, v_span=v_span, gaps=gaps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work",
                    default="/home/xqian/work/scratch_wcgpu1/toolkit-dev/"
                            "wcp-porting-img/pdvd/work")
    ap.add_argument("--min-pts", type=int, default=50)
    ap.add_argument("--anode-reach", type=float, default=330.0)
    ap.add_argument("--cath-lo", type=float, default=-5.0)
    ap.add_argument("--cath-hi", type=float, default=25.0)
    ap.add_argument("--v-reco", type=float, default=V_RECO_DEFAULT,
                    help="drift_speed the INPUT data was reconstructed with "
                         "(1.6 for the original calibration; 1.57 for the "
                         "re-clustered work/ dumps -> closure check)")
    ap.add_argument("--plot", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "drift_velocity_calib.png"))
    args = ap.parse_args()
    v_reco = args.v_reco

    zips = sorted(glob.glob(os.path.join(args.work, "*", "mabc-all-apa.zip")))
    if not zips:
        sys.exit(f"no mabc-all-apa.zip under {args.work}")
    print(f"[geom] D(collection->cathode surface)={D:.3f} cm   xorig={X_W} cm   "
          f"v_reco={v_reco} mm/us")
    print(f"[data] {len(zips)} events under {args.work}\n")

    all_rows = []
    n_ok = 0
    for zp in zips:
        try:
            all_rows.extend(event_clusters(zp, args.min_pts))
            n_ok += 1
        except Exception as e:                     # noqa: BLE001
            print(f"[warn] {zp}: {e}", file=sys.stderr)
    print(f"[data] parsed {n_ok}/{len(zips)} events, {len(all_rows)} clusters "
          f"(>= {args.min_pts} pts)\n")

    full = [r for r in all_rows
            if is_full_crosser(r, args.anode_reach, args.cath_lo, args.cath_hi)]
    print(f"[select] {len(full)} full A-C crossers "
          f"(anode_reach>={args.anode_reach} cm, cathode window "
          f"[{args.cath_lo},{args.cath_hi}] cm)\n")

    g0 = [r for r in full if r["grp"] == 0]
    g1 = [r for r in full if r["grp"] == 1]
    report("group0123 (anodes 0-3)", g0, v_reco)
    report("group4567 (anodes 4-7)", g1, v_reco)
    res = report("ALL", full, v_reco)

    if res is not None:
        print(f"\n==> calibrated drift velocity v_true = {res['v_span']:.4f} mm/us (span / D).")
        print(f"==> input data reconstructed at {v_reco} mm/us  ->  "
              f"~{100*(v_reco/res['v_span']-1):.1f}% off.")
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
    allspan = np.concatenate([g0, g1]) if g0.size or g1.size else np.array([])
    edges = np.arange(320, 380, 2.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(allspan, bins=edges, histtype="stepfilled", alpha=0.15,
            label=f"all (N={allspan.size})", color="k")
    ax.hist(g0, bins=edges, histtype="step", label=f"group0123 (N={g0.size})", color="C0")
    ax.hist(g1, bins=edges, histtype="step", label=f"group4567 (N={g1.size})", color="C1")
    ax.axvline(D, ls="--", color="g", label=f"D={D:.1f} (collection->cath)")
    ax.axvline(S, ls="-", color="r", lw=1.5, label=f"S(pile-up)={S:.1f}")
    ax.set_xlabel("full-crosser drift x-span [cm]")
    ax.set_ylabel("clusters")
    ax.set_title(f"PDVD A-C crosser x-span (data @ {v_reco} mm/us)  ->  "
                 f"v_true={v_reco*D/S:.4f} mm/us")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()

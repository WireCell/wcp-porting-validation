#!/usr/bin/env python3
"""QtoL refit on geometric cathode-crosser anchors, saturation-fixed dumps.

Repro:
    cd pdvd/ql_light_calib && python3 fit_qtol_crossers.py            # _satrep dumps
    python3 fit_qtol_crossers.py --tag ""                             # veto-ON dumps (bias check)

Anchors: strict geometric cathode-crosser (bot, top, flash) triplets — the
harvest of docs/qlmatch/pd_mapping_audit.py (duplicated here with the dump
tag parameterized): both halves' cathode ends within 3 cm of the cathode at
the flash time (per-crate clocks time/time1), end (y,z) within 20 cm, flash
>= 1500 PE dominating its +-30 us neighbourhood.  These are external
geometric ground truth: no dependence on the LASSO/auto selection, so the
fit_qtol_gold.py x350 trap (never fit from auto-selected bundles) is
respected.

Estimator: QtoL_new = QtoL_dump * median over anchors of
Sum_keep(meas PE) / Sum_keep(pred PE at QtoL_dump), keeping channels that
are active, not auto-masked, and not saturation-flagged in that flash (the
dump 'sat' array, present in knob-on dumps; docs/qlmatch/
pdvd-saturation-recovery.md).  Excluding railed channels from BOTH sums is
essential: their measured PE is clipped (x2-10 low) and, with
use_saturation_flag on, their pred is not fitted against anyway.

The pred_pe stored in the dump is the summed prediction of the bundles
linking this flash to each crosser half (bot + top mains).
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
U_CATH = 336.91          # cm, drift coord of the cathode active edge
RUNS = ("039252", "039253", "039349")
CATH = set(range(4, 12))
MEM = {0, 1, 2, 3, 12, 13, 18, 19}


def dump_paths(tag):
    """Index-dir calib dumps for the three runs, with a work-dir tag.

    tag='satrep' -> work/<run>_<idx>_satrep/; tag='' -> the veto-ON
    canonical dirs (039252 uses the pull2c2 variant like pd_mapping_audit,
    since the plain 039252 dumps predate the containment retune)."""
    paths = []
    for run in RUNS:
        for d in sorted(glob.glob(os.path.join(PDVD, "work/%s_*" % run))):
            base = os.path.basename(d)
            tail = base.split("_", 1)[1]
            if tag:
                if not tail.endswith("_" + tag):
                    continue
                idx = tail[: -len("_" + tag)]
                if not idx.isdigit():
                    continue
            else:
                if run == "039252":
                    if not (tail.endswith("_pull2c2") and tail[: -len("_pull2c2")].isdigit()):
                        continue
                elif not tail.isdigit():
                    continue
            paths += sorted(glob.glob(os.path.join(d, "calib-evt*.json")))
    return paths


def cluster_geom(c, apa, geom):
    """u_end(t) = umax0 - v*t with umax0 = max_i s*(x_i - anode_x)."""
    g = geom[str(apa)]
    s = 1.0 if apa == 0 else -1.0
    x = np.asarray(c["x"])
    u0 = s * (x - g["anode_x"])
    umax0 = u0.max()
    sel = u0 >= umax0 - 3.0
    y = np.median(np.asarray(c["y"])[sel])
    z = np.median(np.asarray(c["z"])[sel])
    return umax0, y, z, umax0 - u0.min()


def harvest(paths, lib=None):
    """Strict crosser triplets with measured/pred vectors + keep-masks.

    lib: optional ablib_gold.GridLib -- RECOMPUTE each anchor's prediction
    with that photon library instead of taking the dump's pred_pe (the
    Python replica of the QLMatching vis loop, validated exact in
    docs/pdvd-qlmatching.md).  Lets the 128 nm-vs-175 nm model question be
    re-examined on saturation-fixed dumps without a QL rerun."""
    import ablib_gold
    out = []
    for path in paths:
        d = json.load(open(path))
        v = d["drift_speed"]
        geom = d["geometry"]
        qtol = d["quality_params"]["QtoL"]
        keep_static = np.array([od["active"] and not od["auto_masked"]
                                for od in d["opdets"]], bool)
        pred = {}
        for b in d["bundles"]:
            if lib is not None:
                pred[(b["flash_gid"], b["main_cluster"])] = \
                    ablib_gold.predict(d, b, lib, keep_static)
            else:
                pred[(b["flash_gid"], b["main_cluster"])] = b["pred_pe"]
        bots, tops = [], []
        for c in d["clusters"]:
            if c["npoints"] < 400:
                continue
            apa = 0 if c["uid"] < 4000000 else 4
            umax0, y, z, span = cluster_geom(c, apa, geom)
            if span < 40.0:
                continue
            (bots if apa == 0 else tops).append((c["uid"], umax0, y, z))
        flashes = d["flashes"]
        for f in flashes:
            if f["total_PE"] < 1500:
                continue
            if any(g is not f and abs(g["time"] - f["time"]) < 30
                   and g["total_PE"] > 0.25 * f["total_PE"] for g in flashes):
                continue
            for ub, umb, yb, zb in bots:
                if abs(umb - v * f["time"] - U_CATH) > 3.0:
                    continue
                for ut, umt, yt, zt in tops:
                    if abs(umt - v * f["time1"] - U_CATH) > 3.0:
                        continue
                    if np.hypot(yb - yt, zb - zt) > 20.0:
                        continue
                    pb = pred.get((f["gid"], ub))
                    pt = pred.get((f["gid"], ut))
                    if pb is None and pt is None:
                        continue    # neither half got a bundle: no prediction
                    pr = (np.asarray(pb if pb is not None else [0.0] * 40, float)
                          + np.asarray(pt if pt is not None else [0.0] * 40, float))
                    sat = np.asarray(f.get("sat", [0] * 40), bool)
                    # per-flash readout coverage (knob-on dumps; 1.0 when the
                    # key is absent): self-trigger channels with no snippet
                    # over the flash window measured NOTHING -- exclude them
                    # from both sums instead of scoring fake zeros
                    # (docs/qlmatch/pdvd-lightpattern-sp-investigation.md).
                    cov = np.asarray(f.get("cov", [1.0] * 40), float)
                    out.append(dict(
                        run=os.path.basename(path).split("calib")[0] or path,
                        path=path, evt=d["charge_ident"], gid=f["gid"],
                        meas=np.asarray(f["pe"], float), pred=pr,
                        keep=keep_static & ~sat & (cov >= 1.0),
                        sat_n=int(sat.sum()),
                        cov_n=int((cov < 1.0).sum()),
                        total=f["total_PE"], qtol=qtol,
                        half_bundles=(pb is not None) + (pt is not None)))
    return out


def med(r):
    return (np.median(r), np.percentile(r, 16), np.percentile(r, 84), len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="satrep",
                    help="work-dir tag ('' = veto-ON canonical dumps)")
    ap.add_argument("--both-halves", action="store_true",
                    help="require bundles for BOTH crosser halves (default: >=1)")
    ap.add_argument("--relib", choices=["128", "175"], default=None,
                    help="recompute predictions with this photon library "
                         "(Python vis-loop replica) instead of the dump pred_pe")
    args = ap.parse_args()

    lib = None
    if args.relib:
        import ablib_gold
        lib = ablib_gold.GridLib(os.path.join(
            HERE, "..", "photlib", f"pdvd-photlib-vis-v5-{args.relib}nm.json"))
        print(f"predictions recomputed with the {args.relib} nm library")

    paths = dump_paths(args.tag)
    print(f"dumps: {len(paths)} (tag={args.tag!r})")
    anchors = harvest(paths, lib=lib)
    if args.both_halves:
        anchors = [a for a in anchors if a["half_bundles"] == 2]
    print(f"strict crosser anchors: {len(anchors)} "
          f"({sum(1 for a in anchors if a['half_bundles'] == 2)} with both halves bundled, "
          f"{sum(1 for a in anchors if a['sat_n'])} with >=1 saturated channel, "
          f"{sum(1 for a in anchors if a.get('cov_n'))} with >=1 uncovered channel)")

    qtols = {a["qtol"] for a in anchors}
    if len(qtols) > 1:
        print(f"WARNING: mixed dump QtoL values {qtols}")
    qtol0 = anchors[0]["qtol"] if anchors else float("nan")

    def ratio(a, chans=None):
        k = a["keep"].copy()
        if chans is not None:
            m = np.zeros(40, bool)
            m[list(chans)] = True
            k &= m
        sm, sp = a["meas"][k].sum(), a["pred"][k].sum()
        return sm / sp if sp > 0 else None

    rows = [(a, ratio(a)) for a in anchors]
    rows = [(a, r) for a, r in rows if r is not None and r > 0]
    r_all = np.array([r for _, r in rows])
    m, lo, hi, n = med(r_all)
    print(f"\nGLOBAL: median Sum(meas)/Sum(pred) = {m:.3f}  [{lo:.3f}, {hi:.3f}]  n={n}")
    print(f"  => QtoL = {qtol0} x {m:.3f} = {qtol0 * m:.4f}   "
          f"[{qtol0 * lo:.4f}, {qtol0 * hi:.4f}]")

    print("\nper run:")
    for run in RUNS:
        r = np.array([x for a, x in rows if f"/{run}_" in a["path"]])
        if len(r):
            mm, ll, hh, nn = med(r)
            print(f"  {run}: {mm:.3f} [{ll:.3f}, {hh:.3f}] n={nn} "
                  f"-> QtoL {qtol0 * mm:.4f}")

    print("\nper PD group (channel-restricted sums):")
    for name, chans in (("cathode XA", CATH), ("membrane XA", MEM),
                        ("PMTs", set(range(40)) - CATH - MEM)):
        r = np.array([x for x in (ratio(a, chans) for a, _ in rows)
                      if x is not None and x > 0])
        if len(r):
            mm, ll, hh, nn = med(r)
            print(f"  {name:12s}: {mm:.3f} [{ll:.3f}, {hh:.3f}] n={nn}")

    print("\nratio vs flash total PE (flat = no saturation bias left):")
    edges = [1500, 3000, 6000, 12000, 1e9]
    for a0, a1 in zip(edges[:-1], edges[1:]):
        r = np.array([x for a, x in rows if a0 <= a["total"] < a1])
        if len(r):
            mm, ll, hh, nn = med(r)
            print(f"  [{a0:>6.0f}, {a1:>6.0f}): {mm:.3f} [{ll:.3f}, {hh:.3f}] n={nn}")

    print("\nratio for anchors WITH vs WITHOUT saturated channels:")
    for lbl, sel in (("no railed ch", lambda a: a["sat_n"] == 0),
                     (">=1 railed ch", lambda a: a["sat_n"] > 0)):
        r = np.array([x for a, x in rows if sel(a)])
        if len(r):
            mm, ll, hh, nn = med(r)
            print(f"  {lbl:14s}: {mm:.3f} [{ll:.3f}, {hh:.3f}] n={nn}")

    # Per-(anchor, channel) cathode ratios binned by predicted PE: a FLAT
    # profile = one global cathode gain error; a RISING profile toward dim
    # (far) channels = a visibility-shape (distance) error in the library.
    print("\ncathode per-channel meas/pred vs predicted PE (unrailed channels only):")
    pcr = []
    for a, _ in rows:
        for ch in CATH:
            if not a["keep"][ch]:
                continue
            p, m_ = a["pred"][ch], a["meas"][ch]
            if p > 0.5 and m_ > 0:
                pcr.append((p, m_ / p))
    pcr = np.array(pcr)
    if len(pcr):
        edges = np.percentile(pcr[:, 0], [0, 20, 40, 60, 80, 100])
        for a0, a1 in zip(edges[:-1], edges[1:]):
            s = pcr[(pcr[:, 0] >= a0) & (pcr[:, 0] < a1 + 1e-9), 1]
            if len(s):
                mm, ll, hh, nn = med(s)
                print(f"  pred in [{a0:>8.1f}, {a1:>8.1f}): ratio {mm:.2f} "
                      f"[{ll:.2f}, {hh:.2f}] n={nn}")
    # same for the ganged-pair-validated membrane XAs (reference group)
    print("membrane per-channel meas/pred vs predicted PE:")
    pcr = []
    for a, _ in rows:
        for ch in MEM:
            if not a["keep"][ch]:
                continue
            p, m_ = a["pred"][ch], a["meas"][ch]
            if p > 0.5 and m_ > 0:
                pcr.append((p, m_ / p))
    pcr = np.array(pcr)
    if len(pcr):
        edges = np.percentile(pcr[:, 0], [0, 20, 40, 60, 80, 100])
        for a0, a1 in zip(edges[:-1], edges[1:]):
            s = pcr[(pcr[:, 0] >= a0) & (pcr[:, 0] < a1 + 1e-9), 1]
            if len(s):
                mm, ll, hh, nn = med(s)
                print(f"  pred in [{a0:>8.1f}, {a1:>8.1f}): ratio {mm:.2f} "
                      f"[{ll:.2f}, {hh:.2f}] n={nn}")


if __name__ == "__main__":
    main()

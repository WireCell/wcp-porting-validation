#!/usr/bin/env python3
"""Print L1SP-tagged ROI clusters from a calibration-dump NPZ in a
per-cluster summary table.

Each cluster groups adjacent-channel, time-overlapping tagged ROIs into one
row.  Feature columns (gmax, fill, fwhm_f, asym, efrac) come from the
max-gmax seed ROI in the cluster.  The ``triggered`` column lists the union
of trigger arms that fired across all seed ROIs, ordered as in decide_trigger().

Thresholds are selected automatically per anode:
  * Bottom anodes (n < 4): sp.jsonnet overrides for PDVD-tuned gate
  * Top anodes (n >= 4): C++ header defaults (no override yet)

Usage
-----
  python extract_l1sp_clusters.py --calib-dir <dir> [--anode N [N ...]]
                                   [--run R] [--event E]
                                   [--wire-schema FILE]
  # e.g.:
  python extract_l1sp_clusters.py \\
      --calib-dir /home/xqian/tmp/pdvd_l1sp_calib_039324_0/calib \\
      --run 39324 --event 0
"""

import argparse
import bz2
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


# ── Trigger-gate thresholds ──────────────────────────────────────────────────
# Source: sigproc/inc/WireCellSigProc/L1SPFilterPD.h (C++ header defaults)
#         cfg/pgrapher/experiment/protodunevd/sp.jsonnet  (PDVD overrides)

_CPP_DEFAULTS = dict(
    gmax_min         = 1500.0,
    min_length       = 30,
    asym_strong      = 0.65,
    asym_mod         = 0.40,
    asym_loose       = 0.30,
    len_long_mod     = 100,
    len_long_loose   = 200,
    len_fill_shape   = 50,
    fill_shape_fill  = 0.38,
    fill_shape_fwhm  = 0.30,
    # very_long arm OFF by default (INT_MAX threshold)
)

# sp.jsonnet overrides for PDVD bottom anodes (n < 4)
_BOTTOM_OVERRIDES = dict(
    len_long_mod    = 180,
    len_fill_shape  = 90,
    fill_shape_fill = 0.30,
    fill_shape_fwhm = 0.25,
)

def thresholds_for(anode_ident):
    thr = dict(_CPP_DEFAULTS)
    if anode_ident < 4:
        thr.update(_BOTTOM_OVERRIDES)
    return thr


# ── Wire-schema helpers ──────────────────────────────────────────────────────

PLANE_NAMES = {0: "U", 1: "V", 2: "W"}

def build_ch_plane_map(wire_schema_path, anode_ident):
    """Return dict {channel_int: plane_name_str} for the given anode."""
    with bz2.open(wire_schema_path) as f:
        ws = json.load(f)
    store = ws["Store"]
    anode = next(a["Anode"] for a in store["anodes"]
                 if a["Anode"]["ident"] == anode_ident)
    ch_plane = {}
    for fi in anode["faces"]:
        face = store["faces"][fi]["Face"]
        for pi_local, pi in enumerate(face["planes"]):
            plane = store["planes"][pi]["Plane"]
            for wi in plane["wires"]:
                wire = store["wires"][wi]["Wire"]
                ch_plane[wire["channel"]] = PLANE_NAMES[pi_local]
    return ch_plane


def find_wire_schema(hint=None):
    """Locate the PDVD wire schema, trying hint → WIRECELL_PATH → fallback."""
    candidates = []
    if hint:
        candidates.append(hint)
    for d in os.environ.get("WIRECELL_PATH", "").split(":"):
        candidates.append(os.path.join(d, "protodunevd-wires-larsoft-v3.json.bz2"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    sys.exit("Cannot find protodunevd-wires-larsoft-v3.json.bz2. "
             "Pass --wire-schema explicitly.")


# ── Trigger-arm reconstruction ───────────────────────────────────────────────
# Uses the per-ROI core sub-window values recorded in the NPZ, which are what
# decide_trigger() evaluates (not the full-ROI raw_asym_wide / nbin_fit).

ARM_ORDER = ["asym_strong", "L_long", "L_loose", "fill_shape", "very_long"]

def _arm_label(thr, cl, ca, cf, cfw, gm):
    """Return list of arm labels that fire for one ROI."""
    if gm < thr["gmax_min"] or cl < thr["min_length"]:
        return ["??_pregates"]
    arms = []
    if ca >= thr["asym_strong"]:
        arms.append("asym_strong")
    if cl >= thr["len_long_mod"] and ca >= thr["asym_mod"]:
        arms.append("L_long")
    if cl >= thr["len_long_loose"] and ca >= thr["asym_loose"]:
        arms.append("L_loose")
    if (cl >= thr["len_fill_shape"] and cf <= thr["fill_shape_fill"]
            and cfw <= thr["fill_shape_fwhm"] and ca >= thr["asym_mod"]):
        arms.append("fill_shape")
    return arms if arms else ["??_subwin"]


# ── Cluster builder ──────────────────────────────────────────────────────────

# Adjacency gap tolerance matching L1SPFilterPD default l1_adj_gap_max
ADJ_GAP_MAX = 5

def _build_clusters(channel, roi_start, roi_end):
    n = len(channel)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if abs(channel[i] - channel[j]) > 1:
                continue
            if (max(roi_start[i], roi_start[j])
                    <= min(roi_end[i], roi_end[j]) + ADJ_GAP_MAX):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return clusters


# ── Per-NPZ processing ───────────────────────────────────────────────────────

def process_npz(npz_path, run, event, ch_plane, thr):
    d = np.load(npz_path)
    anode_ident = int(re.search(r"apa(\d+)_", os.path.basename(npz_path)).group(1))

    mask = d["flag_l1_adj"].astype(int) != 0
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        return []

    channel   = d["channel"][idx].astype(int)
    roi_start = d["roi_start"][idx].astype(int)
    roi_end   = d["roi_end"][idx].astype(int)
    nbin_fit  = d["nbin_fit"][idx].astype(int)
    gmax      = d["gmax"][idx].astype(float)
    gauss_fill= d["gauss_fill"][idx].astype(float)
    gauss_fwhm= d["gauss_fwhm_frac"][idx].astype(float)
    raw_asym  = d["raw_asym_wide"][idx].astype(float)
    core_asym = d["core_raw_asym_wide"][idx].astype(float)
    core_len  = d["core_length"][idx].astype(int)
    core_fill = d["core_fill"][idx].astype(float)
    core_fwhm = d["core_fwhm_frac"][idx].astype(float)
    efrac     = d["roi_energy_frac"][idx].astype(float)
    flag_l1   = d["flag_l1"][idx].astype(int)

    n = len(channel)
    arms_per_roi = []
    for i in range(n):
        if flag_l1[i] == 0:
            arms_per_roi.append(["BFS_adj"])
        else:
            arms_per_roi.append(_arm_label(thr,
                                           int(core_len[i]), abs(float(core_asym[i])),
                                           float(core_fill[i]), float(core_fwhm[i]),
                                           float(gmax[i])))

    clusters = _build_clusters(channel, roi_start, roi_end)
    rows = []
    for root, members in sorted(clusters.items(),
                                 key=lambda kv: min(channel[i] for i in kv[1])):
        chs  = channel[members]
        t_lo = int(min(roi_start[members]))
        t_hi = int(max(roi_end[members]))
        nch  = len(set(chs))
        lmax = int(max(nbin_fit[members]))
        plane = ch_plane.get(int(chs[0]), "?")

        seeds = [i for i in members if flag_l1[i] != 0]
        rep = (seeds[int(np.argmax(gmax[seeds]))]
               if seeds else members[int(np.argmax(gmax[members]))])

        seed_arms = set()
        for i in members:
            if flag_l1[i] != 0:
                seed_arms.update(arms_per_roi[i])
        ordered = [a for a in ARM_ORDER if a in seed_arms]
        others  = sorted(seed_arms - set(ARM_ORDER))
        trig    = "+".join(ordered + others) if seed_arms else "BFS_adj"

        rows.append({
            "run":       run,
            "event":     event,
            "anode":     anode_ident,
            "plane":     plane,
            "ch_lo":     int(min(chs)),
            "ch_hi":     int(max(chs)),
            "t_lo":      t_lo,
            "t_hi":      t_hi,
            "nch":       nch,
            "len_max":   lmax,
            "gmax":      int(gmax[rep]),
            "fill":      round(float(gauss_fill[rep]), 2),
            "fwhm_f":    round(float(gauss_fwhm[rep]), 2),
            "asym":      round(float(raw_asym[rep]),   3),
            "efrac":     round(float(efrac[rep]),       2),
            "triggered": trig,
        })
    return rows


# ── Formatting ───────────────────────────────────────────────────────────────

_HDR = (f"{'Run':<7}  {'Event/Anode':<12}  {'plane':<5}  "
        f"{'ch_lo':>6}  {'ch_hi':>6}  {'t_lo':>6}  {'t_hi':>6}  "
        f"{'nch':>4}  {'len_max':>7}  {'gmax':>7}  "
        f"{'fill':>5}  {'fwhm_f':>6}  {'asym':>6}  {'efrac':>5}  triggered")

def print_table(rows):
    print(_HDR)
    print("-" * len(_HDR))
    for r in rows:
        ea = f"{r['event']}/{r['anode']}"
        print(f"{r['run']:<7}  {ea:<12}  {r['plane']:<5}  "
              f"{r['ch_lo']:>6}  {r['ch_hi']:>6}  {r['t_lo']:>6}  {r['t_hi']:>6}  "
              f"{r['nch']:>4}  {r['len_max']:>7}  {r['gmax']:>7}  "
              f"{r['fill']:>5.2f}  {r['fwhm_f']:>6.2f}  {r['asym']:>6.3f}  "
              f"{r['efrac']:>5.2f}  {r['triggered']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True,
                    help="Directory containing apa<N>_*.npz dump files")
    ap.add_argument("--anode", type=int, nargs="+", metavar="N",
                    help="Anode ident(s) to process (default: all found)")
    ap.add_argument("--run",   type=int, default=0, help="Run number for display")
    ap.add_argument("--event", type=int, default=0, help="Event number for display")
    ap.add_argument("--wire-schema", metavar="FILE",
                    help="Path to protodunevd-wires-larsoft-v3.json.bz2")
    args = ap.parse_args()

    schema_path = find_wire_schema(args.wire_schema)

    npz_files = sorted(glob.glob(os.path.join(args.calib_dir, "apa*.npz")))
    if not npz_files:
        sys.exit(f"No apa*.npz files found in {args.calib_dir}")

    if args.anode is not None:
        keep = {f"apa{n}_" for n in args.anode}
        npz_files = [f for f in npz_files
                     if any(os.path.basename(f).startswith(k) for k in keep)]

    all_rows = []
    for npz_path in npz_files:
        m = re.search(r"apa(\d+)_", os.path.basename(npz_path))
        if not m:
            continue
        anode_ident = int(m.group(1))
        thr     = thresholds_for(anode_ident)
        ch_plane = build_ch_plane_map(schema_path, anode_ident)
        rows    = process_npz(npz_path, args.run, args.event, ch_plane, thr)
        all_rows.extend(rows)

    if not all_rows:
        print("No tagged ROIs found.")
        return

    print_table(all_rows)
    total_rois = sum(
        int((np.load(f)["flag_l1_adj"].astype(int) != 0).sum())
        for f in npz_files
        if re.search(r"apa(\d+)_", os.path.basename(f))
    )
    print(f"\n{len(all_rows)} clusters, {total_rois} tagged ROIs total")


if __name__ == "__main__":
    main()

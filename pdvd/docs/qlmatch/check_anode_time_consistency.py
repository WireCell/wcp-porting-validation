#!/usr/bin/env python3
"""Empirical anode-position / time-chain consistency checks for PDVD run 039252.

Companion to pdvd-anode-time-consistency.md (same directory).  Uses the
hand/finder-validated "candles" (17 cathode-crosser pairs + 108 anode boundary
tracks, tags `crossers`/`boundary`) in the 18 events of run 039252 as probes:

  A. anode-end drift-coordinate u of boundary tracks at the (per-side
     corrected) flash T0 -- measures any systematic offset between the
     imaging x anchor (W collection plane) and the FV anode edge (grid plane,
     u=0), separately for bottom (BDE) and top (TDE) volumes;
  A'. UNBIASED anode-edge scan: PCA ends of ALL auto-selected bundles
     (span>=30cm) within +-12 cm of the anode face, no boundary gates --
     check A's boundary sample is circular (the finder picked the min-|u|
     flash per cluster), so A' is the measurement to trust for the edge;
  B. cathode-end residual u_cathode - u  -- does the cathode edge sit at the
     configured +-3.0 cm surface;  B' = the unbiased scan at the cathode face;
  C. full-drift closure for anode->cathode tracks: PCA u-span vs u_cathode;
  D. crosser meeting points: each half's cathode-end TRUE x and the pair
     midpoint -- bottom-vs-top time-base consistency.

KEY correction this script applies (and find_boundary.py does NOT): the calib
dump's f["time"] folds the input-0 = BOTTOM/BDE trigger offset for BOTH
volumes (QLMatching.cxx dump loop runs input 0 only under shared_flash).  For
top-volume (apa 4) geometry the flash time must be re-based:
    t_top = f["time"] + (off_top - off_bot)
with offsets from the dump's reference field `trigger_offsets_us`.  Run
039252: delta = off_top - off_bot = +1.2..+29.4 us  (0.2..4.6 cm of drift).

Geometry helpers (load/pca_extremes/end_faces) duplicated from
ql_display/find_boundary.py to keep this self-contained.

Usage (from this directory):
  python3 check_anode_time_consistency.py            # all 18 events
  python3 check_anode_time_consistency.py --raw-top  # skip the top re-basing
"""

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "..", "..", "work")
DEC_B = os.path.join(HERE, "..", "..", "ql_display", "decisions-boundary")
DEC_X = os.path.join(HERE, "..", "..", "ql_display", "decisions-crossers")

FACE_NAME = {0: "anode", 1: "cathode", 2: "-y(bot)", 3: "+y(top)",
             4: "-z(ups)", 5: "+z(dwn)"}


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    d["flash_by_gid"] = {f["gid"]: f for f in d["flashes"]}
    d["cluster_by_uid"] = {c["uid"]: c for c in d["clusters"]}
    d["geometry"] = {int(k): g for k, g in d["geometry"].items()}
    return d


def pca_extremes(P):
    """Max/min PCA-projection endpoints (python proxy for get_extreme_wcps)."""
    C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    proj = C @ vt[0]
    return P[proj.argmax()], P[proj.argmin()]


def end_faces(p_hi, p_lo, g, t, drift):
    """Nearest face + SIGNED distance for each end, at flash time t.
    Mirrors QLMatching nearest_face: signed distances, argmin over 6 faces."""
    xo = g["sign_offset"] * t * drift
    out = []
    for p in (p_hi, p_lo):
        u = g["s"] * (p[0] + xo - g["anode_x"])
        dd = [u, g["u_cathode"] - u,
              p[1] - g["y_lo"], g["y_hi"] - p[1],
              p[2] - g["z_lo"], g["z_hi"] - p[2]]
        f = int(np.argmin(dd))
        out.append((f, dd[f], u))
    return out


def read_decisions(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                if r["verdict"] in ("keep", "add"):
                    out.append(r)
    return out


def stats(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return "n=0"
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return "n=%3d  median %+6.2f  MAD %5.2f  mean %+6.2f  rms %5.2f  [%+.2f,%+.2f]" % (
        len(a), med, mad, a.mean(), a.std(), a.min(), a.max())


def hist_line(a, lo=-12, hi=12, w=1.0):
    """One-line ASCII histogram + peak-bin center and median of the core
    (peak +- 3 cm) -- robust edge locator for a contaminated sample."""
    a = np.asarray(a, float)
    a = a[(a >= lo) & (a < hi)]
    if len(a) < 5:
        return "n<5 in window"
    nb = int((hi - lo) / w)
    h, edges = np.histogram(a, bins=nb, range=(lo, hi))
    peak = 0.5 * (edges[h.argmax()] + edges[h.argmax() + 1])
    core = a[np.abs(a - peak) <= 3.0]
    bar = "".join(" .:-=+*#%@"[min(9, int(10 * c / h.max()))] for c in h)
    return "n=%3d peak %+5.1f core-median %+5.2f (n=%d)  |%s| %d..%d cm" % (
        len(a), peak, np.median(core), len(core), bar, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-top", action="store_true",
                    help="do NOT re-base top-volume flash times (reproduces the "
                         "BDE-folded dump-time convention used by find_boundary.py)")
    ap.add_argument("--tag", default="",
                    help="work-dir tag: read work/039252_<N>_<TAG>/ instead of "
                         "the untagged work/039252_<N>/ (e.g. --tag anodefix "
                         "for the 2026-07-12 U-plane FV reprocessing)")
    args = ap.parse_args()

    # strict dir match so tagged reprocessings (e.g. _anodefix) and the
    # original production dirs are never mixed in one scan
    pat = re.compile(r"^039252_\d+%s$" %
                     (re.escape("_" + args.tag) if args.tag else ""))
    cals = {}
    for f in sorted(glob.glob(os.path.join(WORK, "039252_*", "calib-evt*.json"))):
        if not pat.match(os.path.basename(os.path.dirname(f))):
            continue
        base = os.path.basename(f)
        if base.startswith("calib-"):
            cals[base[len("calib-"):-len(".json")]] = f

    # accumulators (all cm)
    anode_u = {"bot": [], "top": []}          # boundary anode-end u, corrected t
    anode_u_raw = {"bot": [], "top": []}      # same, dump (BDE) time for both
    cath_res = {"bot": [], "top": []}         # u_cathode - u_far (boundary a->c)
    span_res = []                             # (u_hi-u_lo) - u_cathode  (a->c)
    xr_bot, xr_top, x_mid, x_gap = [], [], [], []   # crosser cathode ends, TRUE x
    edge_anode = {"bot": [], "top": []}       # A': all-auto PCA-end u near anode
    edge_cath = {"bot": [], "top": []}        # B': all-auto u_cathode - u
    per_event = []

    for ev, cal in sorted(cals.items()):
        d = load(cal)
        drift = d["drift_speed"]
        offs = d.get("trigger_offsets_us") or [0.0, 0.0]
        delta = offs[1] - offs[0]              # top rebase, ADD to dump time
        per_event.append((ev, offs[0], offs[1], delta, delta * drift))

        def tcorr(t_dump, apa):
            if apa == 4 and not args.raw_top:
                return t_dump + delta
            return t_dump

        def ends_of(uid, gid):
            # tolerate uids/gids absent from a tagged reprocessing (the
            # decisions files reference the ORIGINAL clustering's idents)
            c = d["cluster_by_uid"].get(uid)
            if c is None or gid not in d["flash_by_gid"]:
                return None
            P = np.column_stack([np.asarray(c["x"], float),
                                 np.asarray(c["y"], float),
                                 np.asarray(c["z"], float)])
            if len(P) < 3:
                return None
            g = d["geometry"][c["apa"]]
            t = tcorr(d["flash_by_gid"][gid]["time"], c["apa"])
            t_raw = d["flash_by_gid"][gid]["time"]
            p_hi, p_lo = pca_extremes(P)
            return (c, g, end_faces(p_hi, p_lo, g, t, drift),
                    end_faces(p_hi, p_lo, g, t_raw, drift), (p_hi, p_lo), t)

        # --- boundary tracks (checks A, B, C) ---
        for r in read_decisions(os.path.join(DEC_B, "decisions-%s.jsonl" % ev)):
            got = ends_of(r["main_cluster_uid"], r["flash_gid"])
            if got is None:
                continue
            c, g, faces, faces_raw, _, _ = got
            side = "bot" if c["apa"] == 0 else "top"
            # anode end = the end whose u is closest to 0 (same rule as finder)
            ka = min((0, 1), key=lambda k: abs(faces[k][2]))
            anode_u[side].append(faces[ka][2])
            anode_u_raw[side].append(faces_raw[ka][2])
            ko = 1 - ka
            fo, dist_o, u_o = faces[ko]
            if fo == 1:                       # anode->cathode track
                cath_res[side].append(g["u_cathode"] - u_o)
                span_res.append((faces[ka][2], u_o, g["u_cathode"], side, ev,
                                 r["main_cluster_uid"]))

        # --- unbiased edge scan (checks A'/B'): ALL auto-selected bundles,
        # no two_boundary gate, no +-3 cm truncation.  Wrong-flash matches
        # (~40% of autos per hand-scan agreement) smear into a flat
        # background; the true active edge appears as a peak.
        seen = set()
        for b in d["bundles"]:
            if not b.get("auto_selected"):
                continue
            uid = b["main_cluster"]
            key = (uid, b["flash_gid"])
            if uid == 3999999 or uid not in d["cluster_by_uid"] or key in seen:
                continue
            seen.add(key)
            got = ends_of(uid, b["flash_gid"])
            if got is None:
                continue
            c, g, faces, _, _, _ = got
            P_span = np.linalg.norm  # noqa: F841 (span gate below)
            cc = d["cluster_by_uid"][uid]
            span = float(np.linalg.norm(
                np.array([max(cc["x"]) - min(cc["x"]),
                          max(cc["y"]) - min(cc["y"]),
                          max(cc["z"]) - min(cc["z"])])))
            if span < 30.0:
                continue
            side = "bot" if c["apa"] == 0 else "top"
            for k in (0, 1):
                u = faces[k][2]
                if abs(u) <= 12.0:
                    edge_anode[side].append(u)
                if abs(g["u_cathode"] - u) <= 12.0:
                    edge_cath[side].append(g["u_cathode"] - u)

        # --- crosser pairs (check D) ---
        byflash = {}
        for r in read_decisions(os.path.join(DEC_X, "decisions-%s.jsonl" % ev)):
            byflash.setdefault(r["flash_gid"], []).append(r["main_cluster_uid"])
        for gid, uids in byflash.items():
            bots = [u for u in uids if u < 4000000]
            tops = [u for u in uids if u >= 4000000]
            if len(bots) != 1 or len(tops) != 1:
                continue
            pair = {}
            for uid in (bots[0], tops[0]):
                got = ends_of(uid, gid)
                if got is None:
                    break
                c, g, faces, _, (p_hi, p_lo), t = got
                # cathode end = end with larger u
                k = max((0, 1), key=lambda k: faces[k][2])
                p = (p_hi, p_lo)[k]
                # TRUE x of the cathode end (imaging x drift-corrected to T0)
                x_true = p[0] + g["sign_offset"] * t * drift
                pair["bot" if c["apa"] == 0 else "top"] = x_true
            if len(pair) == 2:
                xr_bot.append(pair["bot"])
                xr_top.append(pair["top"])
                x_mid.append(0.5 * (pair["bot"] + pair["top"]))
                x_gap.append(pair["top"] - pair["bot"])

    mode = "RAW dump time both volumes" if args.raw_top else \
           "top re-based by (off_top - off_bot)"
    print("# per-event trigger offsets (us):  bot  top  delta  delta*v[cm]")
    for ev, ob, ot, dl, dx in per_event:
        print("  %s  %9.3f %9.3f  %7.3f  %5.2f" % (ev, ob, ot, dl, dx))
    print("\n# mode: %s\n" % mode)

    print("== A. boundary-track anode-end u (cm; 0 = grid-plane anode edge, "
          "+ = into volume) ==")
    for side in ("bot", "top"):
        print("  %s corrected: %s" % (side, stats(anode_u[side])))
        print("  %s dump-time: %s" % (side, stats(anode_u_raw[side])))
    print("  all corrected: %s" % stats(anode_u["bot"] + anode_u["top"]))

    print("\n== A'. UNBIASED edge scan: all auto bundles span>=30cm, PCA-end u "
          "within +-12 cm of the anode face (0 = grid plane) ==")
    for side in ("bot", "top"):
        print("  %s: %s" % (side, hist_line(edge_anode[side])))

    print("\n== B. anode->cathode far-end residual u_cathode - u (cm; "
          "+ = ends short of cathode) ==")
    for side in ("bot", "top"):
        print("  %s: %s" % (side, stats(cath_res[side])))
        core = [r for r in cath_res[side] if abs(r) <= 10]
        print("     |res|<=10 (drops cathode-connect merged tails): %s"
              % stats(core))

    print("\n== B'. UNBIASED edge scan: all auto bundles, u_cathode - u "
          "within +-12 cm of the cathode face ==")
    for side in ("bot", "top"):
        print("  %s: %s" % (side, hist_line(edge_cath[side])))

    print("\n== C. anode->cathode full-drift closure: (u_hi - u_lo) - "
          "u_cathode (cm) ==")
    closure = [(uo - ua) - uc for ua, uo, uc, _, _, _ in span_res]
    print("  %s" % stats(closure))
    for ua, uo, uc, side, ev, uid in sorted(span_res, key=lambda r: r[4]):
        print("    %s c%-8d %s  u_anode %+6.2f  u_cath_end %7.2f  "
              "span-u_cathode %+6.2f" % (ev, uid, side, ua, uo, (uo - ua) - uc))

    print("\n== D. crosser pairs: cathode-end TRUE x (cm; surfaces at -3/+3) ==")
    print("  bottom-half end x: %s" % stats(xr_bot))
    print("  top-half    end x: %s" % stats(xr_top))
    print("  pair midpoint    : %s" % stats(x_mid))
    print("  pair gap top-bot : %s   (6.0 = cathode thickness)" % stats(x_gap))


if __name__ == "__main__":
    main()

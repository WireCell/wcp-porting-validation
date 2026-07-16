#!/usr/bin/env python3
"""Empirical anode-position / time-chain consistency checks for PDHD run 29107.

PDHD counterpart of check_anode_time_consistency.py (PDVD, same directory);
companion to 03_pdhd-anode-time-consistency.md.  Uses the 30 events of run 29107
(pdhd/work/029107_{0..29}/calib-evt*.json) plus the 4-event hand-scan ground
truth (pdhd/work/ql_labels/labels-evt{983,991,999,1007}.json):

  A. GT anode/cathode ends: PCA ends of the hand-scan ACCEPTED matches at
     their flash T0 -- where do human-validated track ends sit relative to
     the FV anode edge (u=0 = xregions anode = U induction plane) and the
     cathode surface (u_cathode = 351.936)?  Unlike the PDVD check A, this
     sample was selected on match quality, not on |anode_u|, so it is not
     circular -- but it is only 4 events.
  A'. UNBIASED anode-edge scan: PCA ends of ALL auto-selected bundles
     (span>=30cm) within +-12 cm of the anode face, all 30 events, per side;
  B'. the same scan at the cathode face (u_cathode - u);
  C. full-drift closure for two_boundary anode->cathode bundles:
     (u_far - u_anode) vs u_cathode;
  D. crosser pairs: dump flashes sharing a coincidence `group` with
     auto-matched clusters on BOTH drift sides -- each half's cathode-end
     TRUE x and the pair midpoint (cathode surfaces at -0.159/+0.159 cm).

KEY difference from PDVD: PDHD's four APAs share ONE DAQ clock, and the
calib dump folds the single per-event trigger offset (~249.8 us, from the
opflash metadata offset_us) into every f["time"], setting the dump-level
trigger_offset to 0.  There is NO per-side re-basing to do -- the script
asserts trigger_offset == 0 and uses dump times directly.  The per-event
offsets are still tabulated (read from the opflash archive metadata).

Geometry (from the dump `geometry` block, = QLMatching compute_geometry):
  side 0 (-x, APA0+2 face 0): anode_x=-352.0945, cathode_x=-0.1587, s=+1
  side 1 (+x, APA1+3 face 1): anode_x=+352.0945, cathode_x=+0.1587, s=-1
  u = s*(x - anode_x); u_cathode = 351.936.  The imaging anchor (collection
  wire plane) sits at u = -1.11 (side 0) / -0.91 (side 1).

Usage (from this directory):
  python3 scripts/check_anode_time_consistency_pdhd.py
"""

import glob
import json
import os
import re
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> qlmatch doc dir
PDHD_WORK = os.path.join(HERE, "..", "..", "..", "pdhd", "work")
LABELS = os.path.join(PDHD_WORK, "ql_labels")

FACE_NAME = {0: "anode", 1: "cathode", 2: "-y", 3: "+y", 4: "-z", 5: "+z"}


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    d["flash_by_gid"] = {f["gid"]: f for f in d["flashes"]}
    d["cluster_by_uid"] = {c["uid"]: c for c in d["clusters"]}
    d["geometry"] = {int(k): g for k, g in d["geometry"].items()}
    return d


def opflash_offset_us(workdir):
    """Per-event readout-vs-trigger offset from the opflash archive metadata
    (the same value run_clus_evt.sh passed as trigger_offset_us)."""
    tar = os.path.join(workdir, "opflash_pdhd-wct.tar.gz")
    if not os.path.isfile(tar):
        return None
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if m.name.endswith("_metadata.json"):
                md = json.loads(tf.extractfile(m).read())
                if "offset_us" in md:
                    return float(md["offset_us"])
    return None


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
    return "n=%4d peak %+5.1f core-median %+5.2f (n=%d)  |%s| %d..%d cm" % (
        len(a), peak, np.median(core), len(core), bar, lo, hi)


def main():
    # canonical per-event dirs only (skip _orig/_perm/_qlt/verify variants)
    cals = {}
    offs = {}
    for wd in sorted(glob.glob(os.path.join(PDHD_WORK, "029107_*"))):
        if not re.match(r"^029107_\d+$", os.path.basename(wd)):
            continue
        for f in glob.glob(os.path.join(wd, "calib-evt*.json")):
            base = os.path.basename(f)
            if "-group" in base:
                continue
            ev = base[len("calib-"):-len(".json")]
            cals[ev] = f
            offs[ev] = opflash_offset_us(wd)

    labels = {}
    for f in glob.glob(os.path.join(LABELS, "labels-evt*.json")):
        lab = json.load(open(f))
        labels[lab["event"]] = lab

    # accumulators (all cm)
    gt_anode_u = {0: [], 1: []}     # A: GT-match ends nearest the anode face
    gt_cath_res = {0: [], 1: []}    # A: GT-match u_cathode - u at cathode face
    edge_anode = {0: [], 1: []}     # A': all-auto PCA-end u near anode
    edge_cath = {0: [], 1: []}      # B': all-auto u_cathode - u near cathode
    span_res = []                   # C: two_boundary anode->cathode closure
    xr = {0: [], 1: []}             # D: crosser cathode-end TRUE x per side
    x_mid, x_gap = [], []
    beam_res = []                   # E: beam-flash time residual vs offset_us
    per_event = []

    for ev, cal in sorted(cals.items(), key=lambda kv: int(kv[0][3:])):
        d = load(cal)
        drift = d["drift_speed"]
        assert abs(d.get("trigger_offset", 0.0)) < 1e-9, \
            "dump trigger_offset != 0 -- f[time] no longer trigger-folded?"
        per_event.append((ev, offs.get(ev)))

        # --- E. beam-flash closure: on a beam trigger the beam flash fires AT
        # the trigger, i.e. raw time ~0, dumped time ~offset_us.  Residual =
        # brightest >1000 PE flash within +-5 us of the trigger, minus offset.
        off = offs.get(ev)
        if off is not None:
            best = None
            for fl in d["flashes"]:
                if fl["apa"] != 0 or fl["total_PE"] < 1000:
                    continue
                if abs(fl["time"] - off) < 5 and \
                        (best is None or fl["total_PE"] > best[1]):
                    best = (fl["time"], fl["total_PE"])
            if best:
                beam_res.append((ev, best[0] - off, best[1]))

        def ends_of(uid, gid):
            c = d["cluster_by_uid"].get(uid)
            if c is None or len(c["x"]) < 3:
                return None
            P = np.column_stack([np.asarray(c["x"], float),
                                 np.asarray(c["y"], float),
                                 np.asarray(c["z"], float)])
            g = d["geometry"][c["apa"]]
            t = d["flash_by_gid"][gid]["time"]
            p_hi, p_lo = pca_extremes(P)
            return c, g, end_faces(p_hi, p_lo, g, t, drift), (p_hi, p_lo), t

        # index bundles: by (flash_gid, main ident) for GT lookup, and the
        # deduped auto list for the scans
        by_gid_ident = {}
        autos = []
        seen = set()
        for b in d["bundles"]:
            by_gid_ident.setdefault(
                (b["flash_gid"], b["main_cluster"] % 1000000), b)
            if not b.get("auto_selected"):
                continue
            key = (b["main_cluster"], b["flash_gid"])
            if key in seen or b["main_cluster"] not in d["cluster_by_uid"]:
                continue
            seen.add(key)
            autos.append(b)

        # --- A. hand-scan GT matches (4 events) ---
        lab = labels.get(ev)
        if lab:
            for m in lab["matches"]:
                for ident in m["cluster_idents"]:
                    b = by_gid_ident.get((m["flash_gid"], ident))
                    if b is None:
                        continue
                    got = ends_of(b["main_cluster"], b["flash_gid"])
                    if got is None:
                        continue
                    c, g, faces, _, _ = got
                    for f, dist, u in faces:
                        if abs(u) <= 12.0:
                            gt_anode_u[c["apa"]].append(u)
                        if abs(g["u_cathode"] - u) <= 12.0:
                            gt_cath_res[c["apa"]].append(g["u_cathode"] - u)

        # --- A'/B'. unbiased edge scans: all auto bundles, span>=30cm ---
        for b in autos:
            got = ends_of(b["main_cluster"], b["flash_gid"])
            if got is None:
                continue
            c, g, faces, _, _ = got
            span = float(np.linalg.norm(
                [max(c["x"]) - min(c["x"]),
                 max(c["y"]) - min(c["y"]),
                 max(c["z"]) - min(c["z"])]))
            if span < 30.0:
                continue
            for f, dist, u in faces:
                if abs(u) <= 12.0:
                    edge_anode[c["apa"]].append(u)
                if abs(g["u_cathode"] - u) <= 12.0:
                    edge_cath[c["apa"]].append(g["u_cathode"] - u)

            # --- C. two_boundary anode->cathode closure ---
            if b.get("two_boundary"):
                us = sorted(fc[2] for fc in faces)
                if abs(us[0]) <= 6.0 and abs(g["u_cathode"] - us[1]) <= 6.0:
                    span_res.append((us[0], us[1], g["u_cathode"],
                                     c["apa"], ev, b["main_cluster"]))

        # --- D. crosser pairs via the flash coincidence groups ---
        bygroup = {}
        for b in autos:
            fl = d["flash_by_gid"][b["flash_gid"]]
            grp = fl.get("group", -1)
            if grp is None or grp < 0:
                continue
            c = d["cluster_by_uid"][b["main_cluster"]]
            bygroup.setdefault(grp, {}).setdefault(c["apa"], set()).add(
                (b["main_cluster"], b["flash_gid"]))
        for grp, sides in bygroup.items():
            if set(sides) != {0, 1} or any(len(v) != 1 for v in sides.values()):
                continue
            pair = {}
            for side, v in sides.items():
                uid, gid = next(iter(v))
                got = ends_of(uid, gid)
                if got is None:
                    break
                c, g, faces, (p_hi, p_lo), t = got
                k = max((0, 1), key=lambda k: faces[k][2])   # cathode end
                p = (p_hi, p_lo)[k]
                pair[side] = p[0] + g["sign_offset"] * t * drift  # TRUE x
            # crosser filter: both halves must actually reach the cathode
            # (within 15 cm) -- the coincidence groups also pair unrelated
            # same-time flashes whose clusters end mid-drift.
            if len(pair) == 2 and abs(pair[0]) < 15.0 and abs(pair[1]) < 15.0:
                xr[0].append(pair[0])
                xr[1].append(pair[1])
                x_mid.append(0.5 * (pair[0] + pair[1]))
                x_gap.append(pair[1] - pair[0])

    print("# per-event trigger offset (us, opflash metadata offset_us; the")
    print("# dump folds it into f['time'], dump trigger_offset field = 0):")
    for ev, off in per_event:
        print("  %-8s  %s" % (ev, "%.3f" % off if off is not None else "n/a"))
    vals = [o for _, o in per_event if o is not None]
    if vals:
        print("  spread: %s" % stats(vals))

    print("\n== A. hand-scan GT match ends (4 events; u in cm, 0 = FV anode"
          " edge = U plane; collection anchor at u=-1.11/-0.91) ==")
    for side in (0, 1):
        nm = "-x side" if side == 0 else "+x side"
        print("  %s anode-end u        : %s" % (nm, stats(gt_anode_u[side])))
        print("  %s u_cathode - u (cat): %s" % (nm, stats(gt_cath_res[side])))

    print("\n== A'. UNBIASED edge scan: all auto bundles span>=30cm, PCA-end"
          " u within +-12 cm of the anode face (30 events) ==")
    for side in (0, 1):
        nm = "-x" if side == 0 else "+x"
        print("  %s: %s" % (nm, hist_line(edge_anode[side])))

    print("\n== B'. UNBIASED edge scan: u_cathode - u within +-12 cm of the"
          " cathode face (+ = ends short of cathode) ==")
    for side in (0, 1):
        nm = "-x" if side == 0 else "+x"
        print("  %s: %s" % (nm, hist_line(edge_cath[side])))

    print("\n== C. two_boundary anode->cathode closure: (u_cath_end -"
          " u_anode_end) - u_cathode (cm) ==")
    closure = [(uo - ua) - uc for ua, uo, uc, _, _, _ in span_res]
    print("  %s" % stats(closure))
    for ua, uo, uc, side, ev, uid in sorted(span_res, key=lambda r: r[4]):
        print("    %-8s u%-8d side%d  u_anode %+6.2f  u_cath_end %7.2f  "
              "closure %+6.2f" % (ev, uid, side, ua, uo, (uo - ua) - uc))

    print("\n== D. crosser pairs (coincidence-grouped flashes, one auto match"
          " per side): cathode-end TRUE x (cm; surfaces at -0.16/+0.16) ==")
    print("  -x half end x : %s" % stats(xr[0]))
    print("  +x half end x : %s" % stats(xr[1]))
    print("  pair midpoint : %s" % stats(x_mid))
    print("  pair gap +x−-x: %s   (0.32 = cathode thickness)" % stats(x_gap))

    print("\n== E. beam-flash closure: brightest >1000 PE flash within +-5 us"
          " of the trigger, dump-time - offset_us (us) ==")
    print("  %s" % stats([r for _, r, _ in beam_res]))
    for ev, r, pe in beam_res:
        print("    %-8s %+7.3f us  %8.0f PE" % (ev, r, pe))


if __name__ == "__main__":
    main()

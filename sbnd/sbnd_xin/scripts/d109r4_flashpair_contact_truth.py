#!/usr/bin/env python3
"""sbnd_xin/docs/109 rev 4: would nu_bundle_flash_group merge the right pairs?

The local 3067-event sample has no MC truth (doc 107: truth lives only in the
production Bee zips), so the cathode-contact rule is graded here on the
colleague's production directory, where every event has both a Bee zip with
truth (data/0/0-mc.json, interaction nodes id 9000000+) and the clusters' 3-D
points (data/0/0-clustering-global.json, x/y/z plus the per-point matched
flash time, which is how a bundle = every cluster matched to one flash is
recovered; the json's cluster ids are stage-A ids and do NOT match T_cluster).
For every event whose two T_tagger rows share a flash group (different TPC,
|dt| < flash_pair_dt_us) the rule is EMULATED offline: the exact closest points
of the two bundles' point sets (scipy cKDTree), contact iff d < gap and -- only
when xcut > 0 -- both points within xcut of the cathode plane, the same
predicate as PR::bundle_contact.  (The toolkit takes the first cluster pair in
contact using Cluster::get_closest_points, a sampled search; exact is the
reference it approximates.)  Each event is then graded against truth:

  (a true interaction = a truth node with Edep >= 20 MeV in the TPC; the rockbox
   production has several per spill, most in the dirt)
  merged     + 1 true interaction whose primary crosses the cathode  -> RIGHT (one event, reunited)
  merged     + 2 true interactions                                   -> WRONG (a second neutrino lost)
  kept apart + 2 true interactions                                   -> RIGHT (both rows kept)
  kept apart + 1 true interaction                                    -> a split left split (today's file)

Usage:
  scripts/d109r4_flashpair_contact_truth.py <run_dir> [--gap 20] [--xcut 0] [--x 0] [--jobs 24]
     --xcut 0 (default) = no cathode-window requirement, contact is distance alone
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot
from scipy.spatial import cKDTree

DT_US = 0.05
EDEP_MIN_MEV = 20.0   # a true interaction "in the event" = visible energy in the TPC above this


def pair_event(path):
    """(tag, rows, gid_clusters) for a same-flash-group two-row event, else None."""
    try:
        f = uproot.open(path)
        names = {k.split(";")[0] for k in f.keys()}
        if "T_tagger" not in names or f["T_tagger"].num_entries != 2:
            return None
        tg = f["T_tagger"].arrays(["matched_flash_gid", "cluster_id", "nu_x", "nu_y", "nu_z",
                                   "act_cluster_id", "act_is_selected", "act_length_cm"], library="np")
        kn = f["T_kine"].arrays(["kine_reco_Enu"], library="np")
        cl = f["T_cluster"].arrays(["cluster_id", "flash_id", "flash_time_us", "is_main", "is_associated",
                                    "length_cm", "tgm", "stm"], library="np")
        gidt = {int(g): float(t) for g, t in zip(cl["flash_id"], cl["flash_time_us"]) if g >= 0}
        gid = [int(tg["matched_flash_gid"][i]) for i in range(2)]
        if gid[0] // 1000000 == gid[1] // 1000000:
            return None
        if not abs(gidt.get(gid[0], float("nan")) - gidt.get(gid[1], float("nan"))) < DT_US:
            return None
        rows = []
        for i in range(2):
            sel = [(int(c), float(L)) for c, s, L in zip(tg["act_cluster_id"][i], tg["act_is_selected"][i],
                                                         tg["act_length_cm"][i]) if s == 1][0]
            rows.append(dict(gid=gid[i], tpc=gid[i] // 1000000, sel=sel[0], L=sel[1],
                             cid=int(tg["cluster_id"][i]),
                             v=(float(tg["nu_x"][i]), float(tg["nu_y"][i]), float(tg["nu_z"][i])),
                             enu=float(kn["kine_reco_Enu"][i]),
                             placeholder=(float(kn["kine_reco_Enu"][i]) == 0.0
                                          and (float(tg["nu_x"][i]), float(tg["nu_y"][i]), float(tg["nu_z"][i])) == (0., 0., 0.))))
        gc = collections.defaultdict(list)   # gid -> [(cluster_id, role, length, tgm, stm)]
        for i in range(len(cl["cluster_id"])):
            g = int(cl["flash_id"][i])
            if g in gid:
                role = "main" if cl["is_main"][i] else ("assoc" if cl["is_associated"][i] else "other")
                gc[g].append((int(cl["cluster_id"][i]), role, float(cl["length_cm"][i]), int(cl["tgm"][i]), int(cl["stm"][i])))
        tag = os.path.basename(path).replace("tracking-pr_", "").replace(".root", "")
        return tag, rows, dict(gc), {g: gidt[g] for g in gid}
    except Exception:  # noqa: BLE001
        return None


def bee(run_dir, tag):
    z = os.path.join(run_dir, "bee", "bee_%s.zip" % tag)
    zz = zipfile.ZipFile(z)
    mc = json.loads(zz.read([n for n in zz.namelist() if n.endswith("-mc.json")][0]))
    cg = json.loads(zz.read([n for n in zz.namelist() if n.endswith("clustering-global.json")][0]))
    truth = []
    for e in mc:
        if not isinstance(e, dict) or e.get("id", 0) >= 19999999:
            continue
        v = e.get("data", {}).get("start")
        if not v:
            continue
        # The rockbox production has several true interactions per spill, most
        # of them in the dirt: an interaction counts only if it deposited
        # visible energy in the TPC ("Edep <n> MeV" in the node text).
        m = re.search(r"Edep\s+([-\d.]+)\s*MeV", e.get("text", ""))
        edep = float(m.group(1)) if m else float("nan")
        if not (edep >= EDEP_MIN_MEV):
            continue
        cross = []
        for c in e.get("children", []):
            cs = c.get("data", {}).get("start")
            ce = c.get("data", {}).get("end")
            if cs and ce and cs[0] * ce[0] < 0:
                cross.append(c.get("text", "").split()[0])
        truth.append((e.get("text", ""), tuple(v), cross))
    pts = np.array([cg["x"], cg["y"], cg["z"]], dtype=float).T
    ot = np.array(cg["opflash_time"], dtype=float)
    return truth, pts, ot


def bundle_points(pts, ot, t_us, tol=0.0015):
    """The Bee points whose per-point opflash_time is this bundle's flash time.

    The Bee clustering json is the stage-A clustering (pre-unmerge ids), so its
    cluster ids do NOT match T_cluster's; its per-point opflash_time does match
    T_cluster.flash_time_us to the printed precision, so a BUNDLE (every cluster
    matched to one flash) is recovered exactly.
    """
    return pts[np.abs(ot - t_us) < tol]


def contact(A, B, cx, xcut, gap):
    """Exact closest points between two point sets: (d, xa, xb, contact?).

    contact iff d < gap and, when xcut > 0, both points within xcut of cx.
    """
    if len(A) == 0 or len(B) == 0:
        return float("nan"), float("nan"), float("nan"), False
    small, big, swapped = (A, B, False) if len(A) <= len(B) else (B, A, True)
    d, j = cKDTree(big).query(small, k=1)
    i = int(np.argmin(d))
    p_small, p_big = small[i], big[int(j[i])]
    xa, xb = (p_small[0], p_big[0]) if not swapped else (p_big[0], p_small[0])
    dd = float(d[i])
    ok = dd < gap and (xcut <= 0 or (abs(xa - cx) < xcut and abs(xb - cx) < xcut))
    return dd, float(xa), float(xb), ok


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--xcut", type=float, default=0.0)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run_dir, "tracking-pr", "tracking-pr_*.root")))
    pairs = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(pair_event, files, chunksize=64):
            if r:
                pairs.append(r)
    print("scanned %d files; same-flash-group two-row events: %d  (x=%.1f xcut=%.1f gap=%.1f cm)"
          % (len(files), len(pairs), a.x, a.xcut, a.gap))

    grade = collections.Counter()
    table = []
    for tag, rows, gc, tof in sorted(pairs):
        truth, pts, ot = bee(a.run_dir, tag)
        ga, gb = rows[0]["gid"], rows[1]["gid"]
        A = bundle_points(pts, ot, tof[ga])
        B = bundle_points(pts, ot, tof[gb])
        d, xa, xb, ok = contact(A, B, a.x, a.xcut, a.gap)
        hit = (d, xa, xb, len(A), len(B)) if ok else None
        best = (d, xa, xb, len(A), len(B))
        merged = hit is not None
        shown = best
        n_int = len(truth)
        n_cross = sum(1 for _, _, c in truth if c)
        dv = [min((dist(r["v"], tv) for _, tv, _ in truth), default=float("nan")) for r in rows]
        both_real = not rows[0]["placeholder"] and not rows[1]["placeholder"]
        if merged and n_int == 1:
            verdict = "RIGHT: one interaction reunited" if n_cross else "merged, one interaction, no truth crossing"
        elif merged and n_int >= 2:
            verdict = "!! WRONG: merged two true interactions"
        elif not merged and n_int >= 2:
            verdict = "RIGHT: two interactions kept apart"
        else:
            verdict = "kept apart, one interaction (split stays split)"
        grade[verdict] += 1
        grade["both rows reconstructed" if both_real else "one row is a placeholder"] += 1
        print("\n  %s: %s  [%d true interaction(s), %d with a cathode-crossing primary]"
              % (tag, "MERGED" if merged else "KEPT APART", n_int, n_cross))
        for tx, tv, cross in truth:
            print("     truth %-62s vtx (%.1f, %.1f, %.1f) crossing: %s" % (tx[:62], *tv, ",".join(cross) or "-"))
        for i, r in enumerate(rows):
            print("     row%d gid %8d tpc%d sel %3d L %6.1f cm Enu %7.1f d_truth %6.1f cm%s"
                  % (i, r["gid"], r["tpc"], r["sel"], r["L"], r["enu"], dv[i],
                     "  (placeholder)" if r["placeholder"] else ""))
        d, xa, xb, nA, nB = shown
        print("     closest approach of the two bundles (%d / %d Bee points): d %.1f cm at x %.1f / %.1f cm"
              % (nA, nB, d, xa, xb))
        print("     -> %s" % verdict)
        table.append((tag, n_int, n_cross, both_real, d, xa, xb, merged))
    print("\n== grades")
    for k in sorted(grade):
        print("  %-55s %d" % (k, grade[k]))
    print("\n== closest approach by truth class (d cm, x_a / x_b cm), ascending d")
    for tag, n_int, n_cross, both_real, d, xa, xb, merged in sorted(table, key=lambda r: r[4]):
        print("  %-16s d %6.1f  x %6.1f / %6.1f  true_int %d cross %d %s %s"
              % (tag, d, xa, xb, n_int, n_cross, "both-real" if both_real else "one-stub", "MERGED" if merged else "apart"))


if __name__ == "__main__":
    main()

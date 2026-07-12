#!/usr/bin/env python3
"""Find anode-involving "boundary tracks" in a PDVD calib dump.

A boundary track is one physical cosmic reconstructed as a SINGLE cluster that,
at the true flash T0, enters through one detector face and exits through a
different one -- with (at least) one end at the ANODE plane.  The anode end sits
at drift coordinate u = 0 (readout time), so it pins the event T0 by itself; the
other end "can be" the cathode (u = u_cathode, a full anode->cathode crossing
track, the tightest calibration candle) or a transverse face (top/bottom/
upstream/downstream).  These complement the cathode-crosser pairs (find_crossers.py):
a crosser constrains BOTH drift volumes at the cathode; a boundary track
constrains ONE volume from the anode.

This finder does NOT re-derive the geometry test -- it gates on the matcher's own
per-bundle diagnostic `flag_two_boundary` (QLMatching::compute_two_boundary_flag:
both main-PCA-axis ends within two_boundary_margin=3cm of a face, at two DIFFERENT
faces, >=1 of them an x-face).  It then recomputes the two end faces (max/min PCA
projection points; drift coord u = s*(x + sign_offset*t*drift - anode_x)) only to
(a) require one end at the anode (face 0) with a clean touch |u|<=--anode-margin,
and (b) report the other end's face.  Per cluster it keeps the flash whose anode
end is closest to u=0 (the best T0), so the one-cluster-many-flashes degeneracy
collapses to one selection.  See docs/ql-cathode-crosser-recipe.md (companion).

Output: a human-readable table (stdout) and, with --emit, a boundary-only
decisions JSONL for make_labels.py (the boundary bundle is keep/add, every other
auto bundle is reject).

Usage:
  python3 find_boundary.py work/039252_0/calib-evt298567.json
  python3 find_boundary.py work/039252_0/calib-evt298567.json \
      --emit decisions-boundary/decisions-evt298567.jsonl
"""

import argparse
import json
import os

import numpy as np

FACE_NAME = {0: "anode", 1: "cathode", 2: "-y(bot)", 3: "+y(top)",
             4: "-z(ups)", 5: "+z(dwn)"}


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    for g, f in enumerate(sorted(d["flashes"], key=lambda f: f["time"]), start=1):
        f["group"] = g
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--min-len", type=float, default=20.0,
                    help="min 3-D span of the cluster (cm)")
    ap.add_argument("--anode-margin", type=float, default=3.0,
                    help="max |u| of the anode end from the anode plane (cm)")
    ap.add_argument("--cathode-only", action="store_true",
                    help="keep only anode->cathode (full-drift) tracks")
    ap.add_argument("--emit", metavar="JSONL",
                    help="write a boundary-only decisions JSONL for make_labels.py")
    args = ap.parse_args()

    d = load(args.calib)
    event = os.path.basename(args.calib)[len("calib-"):-len(".json")]
    drift = d["drift_speed"]

    # Per cluster, keep the two_boundary bundle whose anode end is closest to u=0.
    best = {}   # uid -> record
    extremes = {}
    for j, b in enumerate(d["bundles"]):
        if not b.get("two_boundary"):
            continue
        uid = b["main_cluster"]
        if uid == 3999999 or uid not in d["cluster_by_uid"]:
            continue
        c = d["cluster_by_uid"][uid]
        if uid not in extremes:
            P = np.column_stack([np.asarray(c["x"], float),
                                 np.asarray(c["y"], float),
                                 np.asarray(c["z"], float)])
            if len(P) < 3:
                extremes[uid] = None
            else:
                span = float(np.linalg.norm(P.max(0) - P.min(0)))
                extremes[uid] = (pca_extremes(P), span)
        if extremes[uid] is None:
            continue
        (p_hi, p_lo), span = extremes[uid]
        if span < args.min_len:
            continue
        g = d["geometry"][c["apa"]]
        t = d["flash_by_gid"][b["flash_gid"]]["time"]
        faces = end_faces(p_hi, p_lo, g, t, drift)
        # which end is the anode end (face 0)?  if neither, not anode-involving.
        anode_ends = [k for k in (0, 1) if faces[k][0] == 0]
        if not anode_ends:
            continue
        ka = min(anode_ends, key=lambda k: abs(faces[k][2]))   # cleanest anode end
        anode_u = faces[ka][2]
        if abs(anode_u) > args.anode_margin:
            continue                     # anode end not a clean touch
        ko = 1 - ka
        other_face, other_dist, _ = faces[ko]
        if args.cathode_only and other_face != 1:
            continue
        rec = dict(uid=uid, apa=c["apa"], ident=c["ident"], bundle_idx=j,
                   flash_gid=b["flash_gid"], t_us=t,
                   pe=d["flash_by_gid"][b["flash_gid"]]["total_PE"],
                   span=span, anode_u=anode_u,
                   other_face=other_face, other_dist=other_dist,
                   auto=bool(b["auto_selected"]))
        if uid not in best or abs(anode_u) < abs(best[uid]["anode_u"]):
            best[uid] = rec

    found = sorted(best.values(),
                   key=lambda r: (r["other_face"] != 1, abs(r["anode_u"])))
    n_ac = sum(1 for r in found if r["other_face"] == 1)

    print("# %s\n# gate: flag_two_boundary; anode end |u|<=%.1f cm, span>=%.0f cm%s\n"
          % (args.calib, args.anode_margin, args.min_len,
             "; anode->cathode only" if args.cathode_only else ""))
    print("== BOUNDARY TRACKS (%d: %d anode->cathode, %d anode->transverse) =="
          % (len(found), n_ac, len(found) - n_ac))
    for r in found:
        print("gid%4d t=%9.2f pe=%7.0f c%-8d apa%d span %5.0f  anode_u %5.1f  "
              "other %-8s (%.1f cm)  %s"
              % (r["flash_gid"], r["t_us"], r["pe"], r["uid"], r["apa"],
                 r["span"], r["anode_u"], FACE_NAME[r["other_face"]],
                 r["other_dist"], "AUTO" if r["auto"] else ""))

    if args.emit:
        os.makedirs(os.path.dirname(args.emit) or ".", exist_ok=True)
        keep_idx = {r["bundle_idx"]: r for r in found}
        lines = []
        for j, r in sorted(keep_idx.items()):
            b = d["bundles"][j]
            other = FACE_NAME[r["other_face"]]
            lines.append({
                "event": event,
                "group": d["flash_by_gid"][b["flash_gid"]]["group"],
                "flash_gid": b["flash_gid"], "flash_time_us": r["t_us"],
                "apa": b["apa"], "main_cluster_uid": b["main_cluster"],
                "main_cluster_ident": r["ident"], "bundle_idx": j,
                "verdict": "keep" if b["auto_selected"] else "add",
                "auto_selected": bool(b["auto_selected"]),
                "confidence": "high",
                "reason": ("anode->%s boundary track: anode end at u=%.1f cm, "
                           "other end at %s (%.1f cm), span %.0f cm; flash %d "
                           "(%.0f PE) best-T0 (anode end closest to u=0)."
                           % (other, r["anode_u"], other, r["other_dist"],
                              r["span"], r["flash_gid"], r["pe"])),
            })
        for j, b in enumerate(d["bundles"]):
            if not b["auto_selected"] or j in keep_idx:
                continue
            uid = b["main_cluster"]
            lines.append({
                "event": event,
                "group": d["flash_by_gid"][b["flash_gid"]]["group"],
                "flash_gid": b["flash_gid"],
                "flash_time_us": d["flash_by_gid"][b["flash_gid"]]["time"],
                "apa": b["apa"], "main_cluster_uid": uid,
                "main_cluster_ident": d["cluster_by_uid"].get(uid, {}).get("ident"),
                "bundle_idx": j, "verdict": "reject",
                "auto_selected": True, "confidence": "high",
                "reason": "not an anode boundary track (boundary-only tag).",
            })
        with open(args.emit, "w") as fh:
            for r in lines:
                fh.write(json.dumps(r) + "\n")
        nk = sum(1 for r in lines if r["verdict"] in ("keep", "add"))
        print("\nwrote %s: %d boundary tracks, %d rejects"
              % (args.emit, nk, len(lines) - nk))


if __name__ == "__main__":
    main()

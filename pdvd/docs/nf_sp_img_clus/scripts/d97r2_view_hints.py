#!/usr/bin/env python3
"""doc pdvd/97 round 2 -- per-pick viewing hints for the Bee set (read-only).

    python3 d97r2_view_hints.py > ../../scan/d97r2/view_hints.txt      # also writes scan/d97r2/view_hints.tsv

For every rank-1 pick of scan/d97r2/picks.tsv, from production d103vflip:
  muon direction u   = unit(stop - centroid of the chain's role-1 points with 3 <= rr <= 15 cm) (T_stm_michel_pts)
  Michel direction v = unit(centroid of the role-3 points (the Michel object; for the dots class the admitted dot
                       pieces) - stop); none for a bare stop
  turn angle         = angle(u, v): 0 = straight on, 180 = straight back.  Printed beside the chain's michel_kink_deg,
                       which is measured on the PR segments, as a cross-check only.
  Michel reach       = max distance of a role-3 point from the stop
  best view          = the Bee hotkey that looks along the axis whose perpendicular plane shows the turn largest,
                       |u_p x v_p| (bare: the plane that shows the most of u): key x = Front (YZ), y = Top (XZ),
                       z = Side (XY) -- the deployed Bee's hotkey table
  box of interest    = stop +- 30 cm on each axis (Bee: Box of Interest -> Box Mode, x/y/z min/max)
  volume             = top (x > 0) or bottom (x < 0) drift volume; the cathode is at x = 0
  PF near the stop   = e- / gamma nodes of the candidate's mc subtree with start or end within 50 cm of the stop
"""
import csv, json, math, os, zipfile

import numpy as np
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ARM = "d103vflip"
TSV = IMG + "/pdvd/docs/scan/d97r2/picks.tsv"
OUT = IMG + "/pdvd/docs/scan/d97r2/view_hints.tsv"
KEY = {0: "x (Front, YZ)", 1: "y (Top, XZ)", 2: "z (Side, XY)"}
BOX = 30.0


def pf_near(node, cl, stop, inside=False, out=None):
    out = [] if out is None else out
    here = inside or (isinstance(node.get("id"), int) and node["id"] // 1000 == cl)
    if here:
        t = node["text"].split()
        d = node.get("data") or {}
        ends = [p for p in (d.get("start"), d.get("end")) if p]
        if t[0] in ("e-", "gamma") and ends:
            r = min(math.dist(p, stop) for p in ends)
            if r <= 50:
                out.append(f"{t[0]} {t[1]} MeV @{r:.1f} cm")
    for c in node.get("children") or []:
        pf_near(c, cl, stop, here, out)
    return out


def main():
    rows = [r for r in csv.DictReader((l for l in open(TSV) if not l.startswith("#")), delimiter="\t")
            if r["rank"] == "1"]
    cols = ["bee_event", "cls", "key", "volume", "stop", "michel_start", "turn_deg", "chain_kink_deg", "michel_reach_cm",
            "best_view_key", "box_x", "box_y", "box_z", "pf_near_stop"]
    out = []
    for i, r in enumerate(rows):
        ev, cl = r["key"].rsplit("/", 1); cl = int(cl)
        d = f"{IMG}/pdvd/work/{ev}_{ARM}"
        p = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "x", "y", "z"],
                                                                           library="np")
        m = p["cluster_id"] == cl
        X = np.c_[p["x"], p["y"], p["z"]]
        stop = np.array([float(v) for v in r["stop"].split(",")])
        body = m & (p["role"] == 1) & (p["rr"] >= 3) & (p["rr"] <= 15)
        u = stop - X[body].mean(axis=0)
        u /= np.linalg.norm(u)
        mich = m & (p["role"] == 3)
        if r["cls"] != "bare" and mich.any():
            v = X[mich].mean(axis=0) - stop
            reach = float(np.linalg.norm(X[mich] - stop, axis=1).max())
            v /= np.linalg.norm(v)
            turn = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))
            score = [np.linalg.norm(np.cross(np.delete(u, a), np.delete(v, a))) for a in range(3)]
        else:
            reach, turn = float("nan"), float("nan")
            score = [np.linalg.norm(np.delete(u, a)) for a in range(3)]
        best = int(np.argmax(score))
        mc = json.loads(zipfile.ZipFile(f"{d}/mabc-pr.zip").read("data/0/0-mc.json"))
        near = pf_near(mc[0], cl, tuple(stop))
        row = dict(bee_event=i, cls=r["cls"], key=r["key"], volume="top" if stop[0] > 0 else "bottom",
                   stop=r["stop"], michel_start=r["mstart"] if r["cls"] not in ("dots", "bare") else "-",
                   turn_deg=f"{turn:.0f}", chain_kink_deg=f"{float(r['kink']):.0f}", michel_reach_cm=f"{reach:.1f}",
                   best_view_key=KEY[best],
                   box_x=f"{stop[0] - BOX:.0f}..{stop[0] + BOX:.0f}", box_y=f"{stop[1] - BOX:.0f}..{stop[1] + BOX:.0f}",
                   box_z=f"{stop[2] - BOX:.0f}..{stop[2] + BOX:.0f}", pf_near_stop="; ".join(near) or "none")
        out.append(row)
        print("\t".join(str(row[c]) for c in cols))
    with open(OUT, "w") as f:
        f.write(f"# doc pdvd/97 round 2 -- d97r2_view_hints.py on {ARM} (definitions in the script header)\n")
        f.write("\t".join(cols) + "\n")
        for row in out:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

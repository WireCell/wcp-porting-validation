#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/108 -- charge near multi-prong junctions per arm.

Junctions are defined ONCE on the reference file (main-cluster vertices with
>= 3 distinct sub_cluster_ids within 1.5 cm); for every arm the nearest
flag_vertex row to each reference junction is reported (|d| cm) together with
the number of T_rec_charge points and their summed q within R = 1, 2, 3 cm of
the REFERENCE junction position.  This is the quantity the DL vertex net sees
(cloud charge at the vertex), read out identically for WCP and WCT arms.

Usage: pr108_junction_charge.py --ref ref.root --arm LABEL=file.root [--arm ...] [--undo-u07 LABELS]
"""
import argparse
import numpy as np
import uproot
from pr108_fit_point_compare import load, junctions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="wcp")
    ap.add_argument("--arm", action="append", default=[], help="LABEL=file[:kind]")
    ap.add_argument("--undo-u07", action="store_true", help="undo the uBooNE U/0.7 on wcp arms")
    ap.add_argument("--tag", default=""); ap.add_argument("--radii", default="1,2,3")
    a = ap.parse_args()
    R = [float(x) for x in a.radii.split(",")]
    ref = load(a.ref, a.ref_kind, a.undo_u07)
    J = junctions(ref)
    arms = []
    for spec in a.arm:
        lab, rest = spec.split("=", 1)
        f, kind = (rest.split(":") + ["wct"])[:2]
        arms.append((lab, load(f, kind, a.undo_u07)))
    print(f"[{a.tag}] {len(J)} reference junctions; arms: " + ", ".join(f"{l}({len(d['X'])} pts, sum q {d['q'].sum():.3g})" for l, d in arms))
    for k, j in enumerate(J):
        print(f"  J{k} at ({j[0]:.2f},{j[1]:.2f},{j[2]:.2f})")
        for lab, d in arms:
            # T_rec_charge repeats a vertex row once per incident segment (same
            # position, same q) on both sides: count each position once.
            keep = np.ones(len(d["X"]), bool)
            seen = set()
            for k in range(len(d["X"])):
                key = tuple(np.round(d["X"][k], 4))
                if key in seen: keep[k] = False
                seen.add(key)
            d = {kk: (vv[keep] if isinstance(vv, np.ndarray) else vv) for kk, vv in d.items()}
            dd = np.linalg.norm(d["X"] - j, axis=1)
            vd = dd[d["fv"]]
            vtx = f"nearest vtx {vd.min():.2f} cm" if len(vd) else "no vtx"
            cells = " | ".join(f"R{r:g}: n={int((dd <= r).sum()):3d} q={d['q'][dd <= r].sum():9.0f}" for r in R)
            print(f"     {lab:12s} {vtx:22s} {cells}")


if __name__ == "__main__":
    main()

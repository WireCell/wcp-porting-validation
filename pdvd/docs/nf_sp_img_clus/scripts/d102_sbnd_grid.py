#!/usr/bin/env python3
"""doc pdvd/102 round 2 sec C -- does SBND's PR retile leave the same ISO grid?  Read-only.

SBND runs the same plain 'stepped' retile sampler (cfg/pgrapher/experiment/sbnd/clus.jsonnet:200-206, used by
improve_cluster_2 at :2044-2048), with a 3 mm wire pitch instead of PDHD's 4.79 mm / PDVD's 5.1 mm (W).

Per event dir (sbnd_xin work-<sample>-<TAG>/pr_evt*), per cluster:
  retiled cloud   calib-pr-evt*.json 'steiner' points (x, y, z) -- the Steiner stage's retiled cloud
  image           Bee clustering layer (mabc-pr.zip data/0/0-clustering-global.json)
In-slice nearest-neighbour spacing: points grouped by x to 0.01 cm, NN distance in (y, z) within a slice.
Reports the median, the share at 3 W pitches (+-0.05 cm) and below 1.2 pitch, for the cloud and the image,
and the same restricted to near-isochronous clusters (principal axis > 75 deg from drift).

The same measurement on PDHD / PDVD arms (--pd det:TAG[:max_events]) gives the like-for-like comparison: the
doc 102 census (sec 5.4) read the Bee steiner_graph layer, which SBND's zips do not carry.

Usage: d102_sbnd_grid.py [--tags d101snew] [--samples mcp1k,mcp2k] [--pd pdhd:d101hnew:30 --pd pdvd:d101vnew:30]
"""
import argparse, glob, json, os, zipfile
import numpy as np

SX = "/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
PITCH = 0.3  # cm, SBND U/V/W


def inslice_nn(P):
    xs = np.round(P[:, 0], 2); out = []
    for u in np.unique(xs):
        Q = P[xs == u][:, 1:]
        if len(Q) > 1:
            D = np.linalg.norm(Q[:, None] - Q[None], axis=2); D[D == 0] = np.inf
            out += list(D.min(1))
    return np.array(out)


def angle_to_drift(P):
    if len(P) < 10:
        return np.nan
    C = P - P.mean(0)
    w, v = np.linalg.eigh(C.T @ C)
    d = v[:, -1]
    return float(np.degrees(np.arccos(min(1.0, abs(d[0])))))


def summary(name, nn):
    if not len(nn):
        return f"{name}: no points"
    return (f"{name}: n {len(nn)} median {np.median(nn):.3f} cm | at 3 pitches {np.mean(np.abs(nn - 3 * PITCH) < 0.05):.3f} "
            f"| below 1.2 pitch {np.mean(nn < 1.2 * PITCH):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="d101snew")
    ap.add_argument("--samples", default="mcp1k,mcp2k")
    ap.add_argument("--pd", action="append", default=[], help="det:TAG[:max_events] for a PDHD/PDVD arm")
    a = ap.parse_args()
    global PITCH
    jobs = [("sbnd", tag, 0.3, [d for s in a.samples.split(",") for d in sorted(glob.glob(f"{SX}/work-{s}-{tag}/pr_evt*"))])
            for tag in a.tags.split(",") if tag]
    for spec in a.pd:
        det, tag, *mx = spec.split(":")
        dirs = sorted(glob.glob(f"/home/xqian/toolkit-dev/wcp-porting-img/{det}/work/*_{tag}"))
        if mx:
            dirs = dirs[:int(mx[0])]
        jobs.append((det, tag, {"pdhd": 0.4792, "pdvd": 0.51}[det], dirs))
    for det, tag, pitch, dirs in jobs:
        PITCH = pitch
        acc = {"cloud": [], "image": [], "cloud_iso": [], "image_iso": []}
        nev = ncl = niso = 0
        for s in ["-"]:
            for d in dirs:
                cal = glob.glob(f"{d}/calib-pr-evt*.json")
                if not cal or not os.path.exists(f"{d}/mabc-pr.zip"):
                    continue
                nev += 1
                c = json.load(open(cal[0]))
                img = json.loads(zipfile.ZipFile(f"{d}/mabc-pr.zip").read("data/0/0-clustering-global.json"))
                I = np.c_[img["x"], img["y"], img["z"]]; ic = np.array(img["cluster_id"])
                st = {}
                for e in c.get("steiner", []):
                    st.setdefault(e["cluster_id"], []).append(np.c_[e["x"], e["y"], e["z"]])
                for cl, parts in st.items():
                    S = np.concatenate(parts)
                    if len(S) < 20:
                        continue
                    ncl += 1
                    nn_s = inslice_nn(S); nn_i = inslice_nn(I[ic == cl]) if np.any(ic == cl) else np.array([])
                    acc["cloud"].append(nn_s); acc["image"].append(nn_i)
                    if angle_to_drift(S) > 75:
                        niso += 1; acc["cloud_iso"].append(nn_s); acc["image_iso"].append(nn_i)
        cat = {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}
        print(f"[{det} {tag}] events {nev}, clusters with a Steiner cloud {ncl} (near-ISO > 75 deg: {niso}); pitch {PITCH} cm")
        for k in ("cloud", "image", "cloud_iso", "image_iso"):
            print("   " + summary(k, cat[k]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/97 round 2 -- close-up of the stop region for each pick, in the three projections, from the production zip.

    python3 d97r2_stop_closeup.py [--rank 1] [--half 15] [--out figs/97r2_picks_stop_closeup.png]

Read-only.  Reads scan/d97r2/picks.tsv (written by d97r2_video_picks.py) and each pick's d103vflip mabc-pr.zip.
Grey = clustering-global charge (every cluster), coloured dots = track_fit rows by cluster id (the candidate first),
red x = the chain's stop, magenta + = the chain's Michel start (Michel classes).  This is the view a Bee viewer gets
after zooming onto the stop; it is the check behind amendment A1 (the class's situation must be visible at the stop).
"""
import argparse, csv, json, os, zipfile

import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ARM = "d103vflip"
TSV = IMG + "/pdvd/docs/scan/d97r2/picks.tsv"


def layer(z, name):
    d = json.loads(z.read(f"data/0/0-{name}-global.json"))
    return np.c_[d["x"], d["y"], d["z"]], np.asarray(d["cluster_id"]), np.asarray(d["q"], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--half", type=float, default=15.0)
    ap.add_argument("--out", default=IMG + "/pdvd/docs/nf_sp_img_clus/figs/97r2_picks_stop_closeup.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader((l for l in open(TSV) if not l.startswith("#")), delimiter="\t")
            if int(r["rank"]) == a.rank]
    views = [(2, 0), (1, 0), (2, 1)]
    names = "xyz"
    fig, axs = plt.subplots(len(rows), 3, figsize=(12, 4.0 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        ev, cl = r["key"].rsplit("/", 1); cl = int(cl)
        z = zipfile.ZipFile(f"{IMG}/pdvd/work/{ev}_{ARM}/mabc-pr.zip")
        img, tf = layer(z, "clustering"), layer(z, "track_fit")
        stop = np.array([float(v) for v in r["stop"].split(",")])
        ms = np.array([float(v) for v in r["mstart"].split(",")])
        H = a.half
        bi = np.all(np.abs(img[0] - stop) < H, axis=1)
        bt = np.all(np.abs(tf[0] - stop) < H, axis=1)
        cls_ids = [cl] + sorted(set(tf[1][bt].tolist()) - {cl})
        for j, (h, v) in enumerate(views):
            ax = axs[i][j]
            ax.scatter(img[0][bi, h], img[0][bi, v], s=3, c="0.75", lw=0)
            for k, c in enumerate(cls_ids):
                m = bt & (tf[1] == c)
                ax.scatter(tf[0][m, h], tf[0][m, v], s=7, lw=0, color=f"C{k}",
                           label=f"track_fit cl {c}" + (" (candidate)" if c == cl else ""))
            ax.plot(stop[h], stop[v], "rx", ms=10, mew=2)
            if r["cls"] not in ("dots", "bare"):
                ax.plot(ms[h], ms[v], "m+", ms=10, mew=2)
            ax.set_xlim(stop[h] - H, stop[h] + H); ax.set_ylim(stop[v] - H, stop[v] + H)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{names[h]} [cm]", fontsize=8); ax.set_ylabel(f"{names[v]} [cm]", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_title(f"{r['cls']} #{a.rank}  {r['key']}  (Michel ke_best {float(r['ke_best']):.1f} MeV, "
                             f"kink {float(r['kink']):.0f} deg, len {float(r['mlen']):.1f} cm)", fontsize=8, loc="left")
                ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"doc pdvd/97 r2 ({ARM}): stop region +-{a.half:.0f} cm; grey = clustering charge, dots = track_fit by "
                 f"cluster, red x = stop, magenta + = Michel start", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=70)
    print("wrote", a.out, len(rows), "picks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Second pass over census pairs with a T0: split past-face material into
CONTIGUOUS penetration (drift-connected to the track body, steps <= 2.5 cm --
the evt298567 late-charge signature) vs DETACHED clumps (gap to body > 2.5 cm:
merged-in junk candidates or severely-late charge; need waveform vetting).

Reads cathode_tail_pairs.tsv, writes cathode_tail_anatomy.tsv (one row per
pair-side with any past-face material or pen_raw > 1).
"""
import json
import glob
import csv
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cathode_tail_census as C

PAIRS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cathode_tail_pairs.tsv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cathode_tail_anatomy.tsv")
STEP = 2.5   # cm, drift-connectivity step

# actual raw-ADC readout length in SP tick-index space (0.5 us), per run/side.
# BDE (bottom, anodes 0-3) is SHORTER than TDE; SP pads bottom frames to the
# top length, leaving a no-ADC padding zone where imaging can still put blobs.
NTICKS = {("039252", "T"): 10000, ("039252", "B"): 9766,
          ("039253", "T"): 10000, ("039253", "B"): 9766,
          ("039349", "T"): 6400,  ("039349", "B"): 6250}
WX_OFF = 1.6      # cm, W plane sits ~1.6 cm beyond anode_x (validated on data)
TICK_US = 0.5


def tick_of(pen, face, sgn, g, t_fl, ds_v):
    """SP tick index of a point at penetration pen past the face."""
    X = face - sgn * pen
    x_raw = X - g["sign_offset"] * t_fl * ds_v
    u = g["s"] * (x_raw - g["anode_x"])
    return (u + WX_OFF) / ds_v / TICK_US


def connected_pen(X, tube, q, face, sgn):
    """Deepest penetration drift-connected to the body, walking from inside.

    depth d = sgn*(face - X): negative inside the volume, positive past face.
    Returns (pen_contig, n_contig, q_contig, detached list of
    (pen, npts, qsum, ymed, zmed) clumps beyond the first >STEP gap).
    """
    d = sgn * (face - X[tube])
    o = np.argsort(d)                      # inside -> past
    ds = d[o]
    # start the outward walk at the face-most INSIDE point (mid-track breaks
    # from dead regions must not terminate the chain)
    inside = np.where(ds <= 0.5)[0]
    j0 = inside[-1] if len(inside) else 0
    pen_contig = ds[j0]
    i0 = j0 + 1
    for i in range(j0 + 1, len(ds)):
        if ds[i] - ds[i - 1] > STEP:
            i0 = i
            break
        pen_contig = ds[i]
        i0 = i + 1
    past_mask_sorted = ds > 0
    contig = (np.arange(len(ds)) < i0) & past_mask_sorted
    det = (np.arange(len(ds)) >= i0) & past_mask_sorted
    return pen_contig, contig, det, o, ds


def process(job):
    run, idx, pair_rows = job
    dump = glob.glob(f"{C.BASE}/{run}_{idx}_keep/calib-evt*.json")[0]
    d = json.load(open(dump))
    gt, gb = d["geometry"]["4"], d["geometry"]["0"]
    ds_v = d["drift_speed"]
    cby = {c["uid"]: c for c in d["clusters"]}
    fby = {f["gid"]: f for f in d["flashes"]}
    cache = {}
    out = []
    for r in pair_rows:
        gid = int(r["gid"])
        if gid < 0:
            continue
        f = fby[gid]
        for side, uidk, g, sgn, tkey in (("T", "uid_t", gt, 1, "time1"),
                                          ("B", "uid_b", gb, -1, "time")):
            uid = int(r[uidk])
            if uid not in cache:
                cache[uid] = C.cluster_info(cby[uid], gt if uid >= 4000000 else gb)
            ci = cache[uid]
            X = ci["x_raw"] + g["sign_offset"] * f[tkey] * ds_v
            face = g["cathode_x"]
            tube = ci["tube"]
            pen_c, contig, det, o, dsort = connected_pen(X, tube, ci["q"], face, sgn)
            idxs = np.where(tube)[0][o]
            n_c, n_d = int(contig.sum()), int(det.sum())
            t_fl = f[tkey]
            nticks = NTICKS[(run, side)]
            tick_deep = tick_of(float(dsort.max()), face, sgn, g, t_fl, ds_v)
            tick_contig = tick_of(float(pen_c), face, sgn, g, t_fl, ds_v)
            tick_face = tick_of(0.0, face, sgn, g, t_fl, ds_v)
            row = dict(run=run, idx=idx, evt=r["evt"], side=side, uid=uid,
                       gid=gid, t0src=r["t0src"],
                       pen_raw=round(float(dsort.max()), 2),
                       pen_contig=round(float(pen_c), 2),
                       n_contig=n_c, n_det=n_d,
                       tick_face=round(tick_face, 0),
                       tick_contig=round(tick_contig, 0),
                       tick_deep=round(tick_deep, 0), nticks=nticks,
                       pad_contig=round(tick_contig - nticks, 0),
                       pad_deep=round(tick_deep - nticks, 0))
            if n_d:
                di = idxs[det]
                dyj = np.median(ci["y"][di]) - float(r["end_y"])
                dzj = np.median(ci["z"][di]) - float(r["end_z"])
                row.update(det_pen=round(float(dsort[det].max()), 2),
                           det_gap=round(float(dsort[det].min() - pen_c), 2),
                           det_djunc=round(float(np.hypot(dyj, dzj)), 1),
                           det_q=round(float(ci["q"][di].sum() / ci["q_tot"]), 4))
            else:
                row.update(det_pen=0.0, det_gap=0.0, det_djunc=0.0, det_q=0.0)
            if row["pen_raw"] > 1.0 or n_c or n_d:
                out.append(row)
    return out


def main():
    rows = list(csv.DictReader(open(PAIRS), delimiter="\t"))
    grp = defaultdict(list)
    for r in rows:
        if r["t0src"] in ("pin", "cosel", "geo"):
            grp[(r["run"], r["idx"])].append(r)
    jobs = [(run, idx, rs) for (run, idx), rs in sorted(grp.items())]
    cols = None
    n = 0
    with Pool(6) as pool, open(OUT, "w") as fo:
        for res in pool.imap_unordered(process, jobs):
            for r in res:
                if cols is None:
                    cols = list(r.keys())
                    fo.write("\t".join(cols) + "\n")
                fo.write("\t".join(str(r[c]) for c in cols) + "\n")
                n += 1
    print(f"wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    main()

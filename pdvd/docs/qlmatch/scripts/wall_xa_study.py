#!/usr/bin/env python3
"""Wall (membrane) X-ARAPUCA behavior from matched Q/L pairs, 120 events.

The wall XAs {0,1,3,12,18,19} (+ masked 2 dim, 13 no-WLS) were excluded from
the Q/L fit at the cathode-XA operating point (doc 18).  This study uses the
canonical `_keep` dumps -- the last round where the wall XAs still RECEIVE
photon-library predictions -- to ask whether they are usable at all:

  per matched flash (auto_selected bundles, cathode-XA ruler quality gate):
    pred[ch]  = sum of pred_pe over the flash's selected bundles
    exp[ch]   = pred[ch] * R_cath   (cathode-normalized expectation, removes
                per-flash QtoL/attenuation scale so channel response is isolated)
    meas[ch]  = flash pe[ch];  cov/sat from the flash arrays

One TSV row per (gated flash, channel) for ALL 40 opdets (family labels in the
analysis).  Flash-level context repeated per row: R_cath, gold (xtpc_pin),
charge barycenter (y, z, physical X), top-side charge fraction.
"""
import json
import os
import glob
import re
import numpy as np
from multiprocessing import Pool

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
OUT = os.environ.get("WALLXA_OUT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "wall_xa_flash_channel.tsv"))

CATH = list(range(4, 12))
MIN_GOOD_CATH = 4     # >=4 unsaturated, covered, predicted cathode channels
MIN_PRED_CATH = 50.0  # PE, ruler denominator floor
R_LO, R_HI = 0.5, 2.0  # cathode ruler acceptance (well-matched flash proxy)


def flash_rows(d, run, idx):
    evt = d["charge_ident"]
    ds_v = d["drift_speed"]
    gt, gb = d["geometry"]["4"], d["geometry"]["0"]
    cby = {c["uid"]: c for c in d["clusters"]}
    sel = {}
    for b in d["bundles"]:
        if b.get("auto_selected"):
            sel.setdefault(b["flash_gid"], []).append(b)
    out = []
    for f in d["flashes"]:
        gid = f["gid"]
        bl = sel.get(gid)
        if not bl:
            continue
        pe = np.array(f["pe"], float)
        cov = np.array(f["cov"], float)
        sat = np.array(f["sat"], float)
        pred = np.zeros(40)
        trunc = False
        gold = False
        qx = qy = qz = qtot = qtop = 0.0
        for b in bl:
            pred += np.array(b["pred_pe"], float)
            trunc = trunc or bool(b.get("window_truncated"))
            gold = gold or bool(b.get("xtpc_pin"))
            for uid in [b["main_cluster"]] + list(b.get("other_clusters", [])):
                c = cby.get(uid)
                if c is None:
                    continue
                top = uid >= 4000000
                g = gt if top else gb
                t_fl = f["time1"] if top else f["time"]
                x = np.array(c["x"], float) + g["sign_offset"] * t_fl * ds_v
                y = np.array(c["y"], float)
                z = np.array(c["z"], float)
                q = np.clip(np.array(c["q"], float), 0, None)
                qs = q.sum()
                if qs <= 0:
                    continue
                qx += (q * x).sum()
                qy += (q * y).sum()
                qz += (q * z).sum()
                qtot += qs
                if top:
                    qtop += qs
        if trunc or qtot <= 0:
            continue
        # cathode ruler
        good = [c for c in CATH if cov[c] == 1 and sat[c] == 0 and pred[c] > 0]
        if len(good) < MIN_GOOD_CATH:
            continue
        ps = pred[good].sum()
        if ps < MIN_PRED_CATH:
            continue
        r_cath = pe[good].sum() / ps
        if not (R_LO < r_cath < R_HI):
            continue
        bx, by, bz = qx / qtot, qy / qtot, qz / qtot
        for ch in range(40):
            out.append(dict(
                run=run, idx=idx, evt=evt, gid=gid,
                t=round(float(f["time"]), 2), ch=ch,
                pred=round(float(pred[ch]), 3),
                meas=round(float(pe[ch]), 3),
                cov=int(cov[ch]), sat=int(sat[ch]),
                r_cath=round(float(r_cath), 4),
                n_good_cath=len(good),
                predsum_cath=round(float(ps), 1),
                total_pe=round(float(f["total_PE"]), 1),
                gold=int(gold), nbund=len(bl),
                bary_x=round(bx, 1), bary_y=round(by, 1), bary_z=round(bz, 1),
                qfrac_top=round(qtop / qtot, 3)))
    return out


def process(job):
    run, idx = job
    dumps = glob.glob(f"{BASE}/{run}_{idx}_keep/calib-evt*.json")
    if not dumps:
        return []
    return flash_rows(json.load(open(dumps[0])), run, idx)


def main():
    jobs = []
    for p in sorted(glob.glob(f"{BASE}/*_keep")):
        m = re.match(r"(\d{6})_(\d+)_keep$", os.path.basename(p))
        if m:
            jobs.append((m.group(1), m.group(2)))
    print(f"{len(jobs)} numeric-index keep dumps")
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

#!/usr/bin/env python3
"""Population-scale ophit join for the wall-XA study (120 events).

For every quality-gated matched flash and wall-XA channel with exp>=EXP_JOIN:
look up the membrane ophits (production opflash archive, ophits tensor) whose
pulse START time (col6) falls in [t_flash-1, t_flash+6] us and record where
their PE went (this flash / another flash / unassigned).

ophits tensor columns (validated on evt49746):
  0 opch, 1 hit time ns (peak-side), 2 width ns, 3 area, 4 peak amplitude
  (scaled, 100 = 1 PE/tick), 5 PE, 6 start time ns, 7 assigned flash row
  (-1 = none), 8 sat flag.

Writes wall_xa_ophit_join.tsv: one row per (flash, wall channel) with
booked PE (flash arrays), available PE (start-time-matched ophits), and the
booking destination breakdown.
"""
import csv
import os
import glob
import io
import json
import tarfile
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

SP = os.environ.get("WALLXA_DIR", os.path.dirname(os.path.abspath(__file__)))
BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
OPCH = {0: [2010, 2011], 1: [2030], 2: [2020, 2021], 3: [2040, 2041],
        12: [2050, 2051], 13: [2070, 2071], 18: [2060, 2061], 19: [2080, 2081]}
LIVE = (0, 1, 3, 12, 18, 19)
EXP_JOIN = 10.0
T_LO, T_HI = -1.0, 6.0   # us window on ophit START relative to flash time

rows = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
f = lambda r, k: float(r[k])
byev = defaultdict(list)
for r in rows:
    if int(r["ch"]) in LIVE:
        e = f(r, "pred") * f(r, "r_cath")
        if e >= EXP_JOIN:
            r["exp"] = e
            byev[(r["run"], r["idx"], r["evt"])].append(r)


def process(job):
    (run, idx, evt), rs = job
    try:
        arch = f"{BASE}/{run}_light{evt}_keep/opflash_pdvd-wct.tar.gz"
        with tarfile.open(arch) as tf:
            names = tf.getnames()
            def get(nm):
                return np.load(io.BytesIO(tf.extractfile(nm).read()))
            meta = {}
            for nm in names:
                if nm.endswith("_metadata.json") and "tensorset" not in nm:
                    j = json.load(io.BytesIO(tf.extractfile(nm).read()))
                    meta[j.get("name")] = nm.replace("_metadata.json", "_array.npy")
            oph = get(meta["ophits"])
            ofl = get(meta["opflash"])
    except Exception as ex:
        return [("ERR", run, evt, str(ex))]
    d = json.load(open(glob.glob(f"{BASE}/{run}_{idx}_keep/calib-evt*.json")[0]))
    off0 = d["trigger_offsets_us"][0]
    fby = {fl["gid"]: fl for fl in d["flashes"]}
    out = []
    for r in rs:
        gid = int(r["gid"])
        fl = fby[gid]
        t_ns = (fl["time"] - off0) * 1e3
        # this flash's tensor row (times agree exactly by construction)
        cand = np.where(np.abs(ofl[:, 0] - t_ns) < 100)[0]
        myrow = int(cand[0]) if len(cand) else -999
        ch = int(r["ch"])
        m = np.isin(oph[:, 0], OPCH[ch]) & \
            (oph[:, 6] >= t_ns + T_LO * 1e3) & (oph[:, 6] <= t_ns + T_HI * 1e3)
        hits = oph[m]
        avail = float(hits[:, 5].sum())
        pe_here = float(hits[hits[:, 7] == myrow, 5].sum()) if myrow >= 0 else 0.0
        pe_other = float(hits[(hits[:, 7] >= 0) & (hits[:, 7] != myrow), 5].sum())
        pe_none = float(hits[hits[:, 7] < 0, 5].sum())
        pk = float(hits[:, 4].max()) if len(hits) else 0.0
        wmax = float(hits[:, 2].max()) if len(hits) else 0.0
        dt1 = float((hits[np.argmax(hits[:, 5]), 1] - t_ns) / 1e3) if len(hits) else np.nan
        out.append(dict(run=run, idx=idx, evt=evt, gid=gid, ch=ch,
                        exp=round(r["exp"], 2), meas=r["meas"], cov=r["cov"],
                        gold=r["gold"], r_cath=r["r_cath"],
                        nhit=len(hits), avail=round(avail, 2),
                        pe_here=round(pe_here, 2), pe_other=round(pe_other, 2),
                        pe_none=round(pe_none, 2), peak=round(pk, 1),
                        wmax_ns=round(wmax, 0), dt_main=round(dt1, 2) if dt1 == dt1 else ""))
    return out


def main():
    jobs = sorted(byev.items())
    print(f"{len(jobs)} events, {sum(len(v) for v in byev.values())} (flash,ch) cases")
    cols, n = None, 0
    with Pool(6) as pool, open(f"{SP}/wall_xa_ophit_join.tsv", "w") as fo:
        for res in pool.imap_unordered(process, jobs):
            for r in res:
                if isinstance(r, tuple):
                    print(r)
                    continue
                if cols is None:
                    cols = list(r.keys())
                    fo.write("\t".join(cols) + "\n")
                fo.write("\t".join(str(r[c]) for c in cols) + "\n")
                n += 1
    print(f"wrote {n} rows")


if __name__ == "__main__":
    main()

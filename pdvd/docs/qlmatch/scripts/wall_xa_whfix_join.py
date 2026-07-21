#!/usr/bin/env python3
"""Join the wide-hit-fixed (_whfix, wide_hit_mode='start' on mem+pmt) opflash
archives back to the doc-25 matched-flash sample.

For every gated matched flash in wall_xa_flash_channel.tsv (times from the
canonical _keep dumps), find the _whfix flash nearest in time (light-frame
base; the matched flashes are cathode-bright so their times are stable) and
record the new per-channel PE / coverage / saturation for all 40 opdets.

Writes wall_xa_whfix_join.tsv keyed like the pairs TSV (run, idx, gid, ch).
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
DT_MAX_US = 0.6

rows = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
f = lambda r, k: float(r[k])
byev = defaultdict(dict)
for r in rows:
    byev[(r["run"], r["idx"], r["evt"])].setdefault(int(r["gid"]), r)


def process(job):
    (run, idx, evt), by_gid = job
    try:
        arch = f"{BASE}/{run}_light{evt}_whfix/opflash_pdvd-wct.tar.gz"
        with tarfile.open(arch) as tf:
            T = {}
            for nm in tf.getnames():
                if nm.endswith("_metadata.json") and "tensorset" not in nm:
                    j = json.load(io.BytesIO(tf.extractfile(nm).read()))
                    T[j.get("name")] = np.load(io.BytesIO(
                        tf.extractfile(nm.replace("_metadata.json", "_array.npy")).read()))
        ofl = T["opflash"]
        cov = T.get("flash_cov")
        sat = T.get("flash_sat")
    except Exception as ex:
        return [("ERR", run, evt, str(ex))]
    d = json.load(open(glob.glob(f"{BASE}/{run}_{idx}_keep/calib-evt*.json")[0]))
    off0 = d["trigger_offsets_us"][0]
    fby = {fl["gid"]: fl for fl in d["flashes"]}
    out = []
    for gid, r in sorted(by_gid.items()):
        t_ns = (fby[gid]["time"] - off0) * 1e3
        k = int(np.argmin(np.abs(ofl[:, 0] - t_ns)))
        dt = (ofl[k, 0] - t_ns) / 1e3
        if abs(dt) > DT_MAX_US:
            out.append(dict(run=run, idx=idx, evt=evt, gid=gid, matched=0,
                            dt=round(dt, 3), ch=-1, meas=0, cov=0, sat=0))
            continue
        for ch in range(40):
            out.append(dict(run=run, idx=idx, evt=evt, gid=gid, matched=1,
                            dt=round(dt, 3), ch=ch,
                            meas=round(float(ofl[k, 1 + ch]), 3),
                            cov=round(float(cov[k, ch]), 3) if cov is not None else 1,
                            sat=int(sat[k, ch]) if sat is not None else 0))
    return out


def main():
    jobs = sorted(byev.items())
    print(f"{len(jobs)} events")
    cols, n, nerr = None, 0, 0
    with Pool(6) as pool, open(f"{SP}/wall_xa_whfix_join.tsv", "w") as fo:
        for res in pool.imap_unordered(process, jobs):
            for r in res:
                if isinstance(r, tuple):
                    print(r)
                    nerr += 1
                    continue
                if cols is None:
                    cols = list(r.keys())
                    fo.write("\t".join(cols) + "\n")
                fo.write("\t".join(str(r[c]) for c in cols) + "\n")
                n += 1
    print(f"wrote {n} rows ({nerr} event errors)")


if __name__ == "__main__":
    main()

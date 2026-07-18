#!/usr/bin/env python3
"""Doc 25 §8: wall-XA measured-PE gain calibration from the §7 whfix sample.

Per live wall channel: scale = 1/median(meas_whfix / exp) over responding
cases (meas > 0.5 PE) with exp = pred x R_cath >= 10.  Also the residual
scatter after per-channel calibration, which sets the wall pe_err family.
"""
import csv
import json
import os
import numpy as np

SP = os.environ.get("WALLXA_DIR", os.path.dirname(os.path.abspath(__file__)))
LIVE = [0, 1, 3, 12, 18, 19]

pairs = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
new = {(r["run"], r["idx"], r["gid"], r["ch"]): r
       for r in csv.DictReader(open(f"{SP}/wall_xa_whfix_join.tsv"), delimiter="\t")
       if r["matched"] == "1"}
f = lambda r, k: float(r[k])

scale = [1.0] * 40
res = []
print("ch  n_resp  med(meas/exp)  scale   p16   p84")
for ch in LIVE:
    R = []
    for r in pairs:
        if int(r["ch"]) != ch:
            continue
        exp = f(r, "pred") * f(r, "r_cath")
        if exp < 10:
            continue
        k = (r["run"], r["idx"], r["gid"], r["ch"])
        if k not in new:
            continue
        m = f(new[k], "meas")
        if m > 0.5:
            R.append(m / exp)
    R = np.array(R)
    med = np.median(R)
    scale[ch] = round(1 / med, 3)
    res += list(R / med)
    print(f"{ch:3d} {len(R):6d} {med:12.3f} {scale[ch]:6.3f} "
          f"{np.percentile(R, 16):6.2f} {np.percentile(R, 84):6.2f}")
res = np.array(res)
print(f"\nresidual after per-ch calibration: p16={np.percentile(res, 16):.2f} "
      f"p84={np.percentile(res, 84):.2f}")
print("\nPDVD_QL_MEASURED_PE_SCALE=" + json.dumps(scale))

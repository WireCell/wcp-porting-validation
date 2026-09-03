#!/usr/bin/env python3
"""doc pdvd/28: reproduce calc_charge_wcp's per-point RMS charge (the
quantity find_peak_point_indices thresholds against terminal_charge_threshold)
directly from a pctree tensor dump, for a real PDVD event.

This reads the SAME per-point arrays Facade::Cluster::calc_charge_wcp reads
(ucharge_val / vcharge_val / wcharge_val from the flat "live" 3d point cloud)
and reproduces its RMS-of-nonzero-planes formula
(clus/src/Facade_Cluster.cxx:1031-1112, disable_dead_mix_cell=false branch).

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
      work/039252_2_stm1/pctree-evt298595.tar.gz
"""
import io
import json
import sys
import tarfile

import numpy as np


def load_point_cloud_charges(tar_path):
    t = tarfile.open(tar_path)
    names = set(t.getnames())
    # locate the flat "live" 3d point cloud's u/v/w charge_val arrays by
    # scanning tensor metadata for their datapath (index is dataset-specific).
    idx = {}
    for n in names:
        if not n.endswith("_metadata.json"):
            continue
        meta = json.loads(t.extractfile(n).read())
        dp = meta.get("datapath", "")
        if dp.endswith("pointclouds/namedpcs/3d/arrays/ucharge_val"):
            idx["u"] = n.replace("_metadata.json", "_array.npy")
        elif dp.endswith("pointclouds/namedpcs/3d/arrays/vcharge_val"):
            idx["v"] = n.replace("_metadata.json", "_array.npy")
        elif dp.endswith("pointclouds/namedpcs/3d/arrays/wcharge_val"):
            idx["w"] = n.replace("_metadata.json", "_array.npy")
    out = {}
    for plane, arrname in idx.items():
        buf = t.extractfile(arrname).read()
        out[plane] = np.load(io.BytesIO(buf)).astype(float).ravel()
    return out


def calc_charge_wcp_rms(qu, qv, qw):
    """Vectorized reproduction of Cluster::calc_charge_wcp(..., disable_dead_mix_cell=false)."""
    flag_u = qu == 0
    flag_v = qv == 0
    flag_w = qw == 0
    charge = np.zeros_like(qu)
    ncharge = np.zeros_like(qu, dtype=int)
    for q, flag in ((qu, flag_u), (qv, flag_v), (qw, flag_w)):
        nz = q != 0
        charge = charge + np.where(nz, q * q, 0.0)
        ncharge = ncharge + nz.astype(int)
    rms = np.where(ncharge > 1, np.sqrt(np.divide(charge, ncharge, where=ncharge > 0)), 0.0)
    # quality: charge_cut applied per-plane raises flag_{u,v,w} too, but since
    # the reported "charge" here is threshold-independent, quality is left to
    # the caller (it depends on charge_cut, unlike this RMS value).
    return rms


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "work/039252_2_stm1/pctree-evt298595.tar.gz"
    charges = load_point_cloud_charges(path)
    qu, qv, qw = charges["u"], charges["v"], charges["w"]
    rms = calc_charge_wcp_rms(qu, qv, qw)
    n = len(rms)
    print(f"{path}: n_points={n}")
    for name, a in (("U", qu), ("V", qv), ("W", qw), ("RMS (calc_charge_wcp)", rms)):
        nz = a[a != 0]
        print(f"  {name:24s} median={np.median(a):9.1f}  median(nz)={np.median(nz) if len(nz) else float('nan'):9.1f}  "
              f"mean={np.mean(a):9.1f}  frac>4000={np.mean(a>4000):.3f}  frac>500={np.mean(a>500):.3f}  frac==0={np.mean(a==0):.3f}")
    all3_above = (qu > 4000) & (qv > 4000) & (qw > 4000)
    all3_above500 = (qu > 500) & (qv > 500) & (qw > 500)
    print(f"  fraction with ALL THREE planes > 4000e: {np.mean(all3_above):.3f}")
    print(f"  fraction with ALL THREE planes >  500e: {np.mean(all3_above500):.3f}")


if __name__ == "__main__":
    main()

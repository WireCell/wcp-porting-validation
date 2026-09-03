#!/usr/bin/env python3
"""doc pdvd/28: reproduce calc_charge_wcp's per-point RMS charge (the
quantity find_peak_point_indices thresholds against terminal_charge_threshold)
directly from a pctree tensor dump, for a real PDVD or SBND event, and
quantify the (u,v)-wire-crossing ambiguity behind it.

This reads the SAME per-point arrays Facade::Cluster::calc_charge_wcp reads
(ucharge_val / vcharge_val / wcharge_val from the flat "live" 3d point cloud)
and reproduces its RMS-of-nonzero-planes formula
(clus/src/Facade_Cluster.cxx:1031-1112, disable_dead_mix_cell=false branch).
It also groups points by (uwire_index, vwire_index): the "stepped" sampler
(clus/src/BlobSampler.cxx:703-916) can place several 3D points, differing
only in their W position, at the same (U,V) wire crossing when a blob's
W-strip is wide (a coarse-pitch / imaging-ambiguity effect); within each such
group only the max-W-charge point is the "peak" (true) candidate, the rest
are "losing" candidates that usually fail the three-plane AND gate.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd   # or .../sbnd/sbnd_xin
  python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
      work/039252_2_stm1/pctree-evt298595.tar.gz
"""
import io
import json
import sys
import tarfile
from collections import defaultdict

import numpy as np


def _load_arrays(tar_path, suffixes):
    """Return {key: np.array} for each (key, datapath-suffix) pair found."""
    t = tarfile.open(tar_path)
    names = set(t.getnames())
    idx = {}
    for n in names:
        if not n.endswith("_metadata.json"):
            continue
        meta = json.loads(t.extractfile(n).read())
        dp = meta.get("datapath", "")
        for key, suffix in suffixes.items():
            if dp.endswith(suffix):
                idx[key] = n.replace("_metadata.json", "_array.npy")
    out = {}
    for key, arrname in idx.items():
        buf = t.extractfile(arrname).read()
        out[key] = np.load(io.BytesIO(buf)).ravel()
    return out


def load_point_cloud_charges(tar_path):
    arrs = _load_arrays(tar_path, {
        "u": "pointclouds/namedpcs/3d/arrays/ucharge_val",
        "v": "pointclouds/namedpcs/3d/arrays/vcharge_val",
        "w": "pointclouds/namedpcs/3d/arrays/wcharge_val",
    })
    return {k: v.astype(float) for k, v in arrs.items()}


def load_wire_indices(tar_path):
    arrs = _load_arrays(tar_path, {
        "u": "pointclouds/namedpcs/3d/arrays/uwire_index",
        "v": "pointclouds/namedpcs/3d/arrays/vwire_index",
    })
    return {k: v.astype(int) for k, v in arrs.items()}


def ambiguity_report(uw, vw, qw, and_gate):
    """(u,v)-crossing multiplicity, and AND-gate pass rate split by whether a
    point is the max-W-charge ("peak") or a "losing" candidate in its group."""
    n = len(uw)
    groups = defaultdict(list)
    for i in range(n):
        groups[(uw[i], vw[i])].append(i)
    sizes = np.array([len(g) for g in groups.values()])
    amb_mask = np.zeros(n, bool)
    peak_mask = np.zeros(n, bool)
    for g in groups.values():
        if len(g) > 1:
            amb_mask[g] = True
            peak_idx = g[int(np.argmax(qw[g]))]
            peak_mask[peak_idx] = True
        else:
            peak_mask[g[0]] = True
    losing_mask = amb_mask & ~peak_mask
    print(f"  distinct (u,v) wire-crossings: {len(groups)}  median candidates/crossing: {np.median(sizes):.0f}  "
          f"fraction of crossings with >1 candidate: {np.mean(sizes > 1):.3f}")
    print(f"  fraction of ALL points that are a losing (non-peak) candidate: {losing_mask.mean():.3f}")
    print(f"  AND-gate(4000) pass rate: peak candidates={and_gate(4000)[peak_mask].mean():.3f}  "
          f"losing candidates={and_gate(4000)[losing_mask].mean() if losing_mask.any() else float('nan'):.3f}")


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

    wires = load_wire_indices(path)
    if "u" in wires and "v" in wires:
        def and_gate(cut):
            return (qu > cut) & (qv > cut) & (qw > cut)
        ambiguity_report(wires["u"], wires["v"], qw, and_gate)
    else:
        print("  (uwire_index/vwire_index not in this dump -- skipping ambiguity report)")


if __name__ == "__main__":
    main()

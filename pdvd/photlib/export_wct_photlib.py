#!/usr/bin/env python3
"""Export the sampled v5 ANN visibility grids (work/ann_vis_v5_*.npz from
sample_ann.py) into the WCT PhotonLibraryModel format:

  pdvd-photlib-vis-v5-<tag>.npy    float32 C-order [nx, ny, nz, 40]
  pdvd-photlib-vis-v5-<tag>.json   meta: origin_cm/step_cm/n/nchan/vis_npy
                                   + provenance + "selfcheck" block (random
                                   points with python-side trilinear values,
                                   verified by the env-gated doctest in
                                   match/test/doctest_photonlibrarymodel.cxx)

Channel order: v5 ANN channel == WCT flash-chain OpDet (identity, verified in
sample_ann.py; dead channels 24/27/28/34 carry model visibilities and are
masked in the QLMatching cfg).

Also installs copies into wire-cell-data/pdvd/photodet/ if that directory
exists (the cfg default path 'pdvd/photodet/...' resolves through
WIRECELL_PATH).
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WCDATA = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/pdvd/photodet"
NSELF = 32


def trilinear(vis, origin, step, n, pts):
    out = np.empty((len(pts), vis.shape[-1]))
    for k, p in enumerate(pts):
        t = np.clip((p - origin) / step, 0, np.array(n) - 1)
        i0 = np.minimum(t.astype(int), np.array(n) - 2)
        f = t - i0
        v = 0.0
        for c in range(8):
            d = [(c >> (2 - a)) & 1 for a in range(3)]
            w = np.prod([f[a] if d[a] else 1 - f[a] for a in range(3)])
            v = v + w * vis[i0[0] + d[0], i0[1] + d[1], i0[2] + d[2]]
        out[k] = v
    return out


def export(tag):
    src = os.path.join(HERE, "work", f"ann_vis_v5_{tag}.npz")
    d = np.load(src, allow_pickle=True)
    vis = np.ascontiguousarray(d["vis"].astype(np.float32))
    origin, step, n = d["origin_cm"].astype(float), d["step_cm"].astype(float), d["n"]

    npy_name = f"pdvd-photlib-vis-v5-{tag}.npy"
    np.save(os.path.join(HERE, npy_name), vis)

    rng = np.random.default_rng(20260709)
    lo = origin + 0.25 * step * np.array(n)
    hi = origin + 0.75 * step * np.array(n)
    pts = rng.uniform(lo, hi, size=(NSELF, 3))
    exp = trilinear(vis, origin, step, n, pts)
    chs = rng.integers(0, vis.shape[-1], NSELF)

    meta = {
        "_comment": "PDVD photon-visibility library for WCT QLMatching (light_model: 'library').  Sampled from the official PDFastSimANN computable graph %s on a %s cm grid over the active volume; channel == WCT flash-chain OpDet.  See pdvd/photlib/ and pdvd/docs/pdvd-photon-model.md." % (str(d["model"]), step.tolist()),
        "origin_cm": origin.tolist(),
        "step_cm": step.tolist(),
        "n": [int(x) for x in n],
        "nchan": int(vis.shape[-1]),
        "vis_npy": npy_name,
        "selfcheck": [
            {"p_cm": pts[k].round(4).tolist(), "ch": int(chs[k]),
             "vis": float(exp[k, chs[k]])} for k in range(NSELF)
        ],
    }
    meta_name = f"pdvd-photlib-vis-v5-{tag}.json"
    with open(os.path.join(HERE, meta_name), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {npy_name} ({os.path.getsize(os.path.join(HERE, npy_name))/1e6:.1f} MB) + {meta_name}")

    if os.path.isdir(os.path.dirname(WCDATA)) or os.path.isdir("/nfs/data/1/xqian/toolkit-dev/wire-cell-data"):
        os.makedirs(WCDATA, exist_ok=True)
        shutil.copy(os.path.join(HERE, npy_name), WCDATA)
        shutil.copy(os.path.join(HERE, meta_name), WCDATA)
        print(f"installed into {WCDATA}")


if __name__ == "__main__":
    for tag in ("128nm", "175nm"):
        export(tag)

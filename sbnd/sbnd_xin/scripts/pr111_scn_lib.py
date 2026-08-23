#!/usr/bin/env python3
"""doc pr/111 -- offline re-run of the frozen SCN vertex net on a harvested cloud.

dl_vtx_harvest (pr/79 sec 10) stores vec_xyzq verbatim in the scoreboard, and the
header states offline voxelization from those floats reproduces the live network
input bit-exactly.  This module does exactly that, using the SHIPPED
pyutil/python/SCN_Vertex.py so there is no re-implementation to drift.

Validated in pr111_scn_validate.py against the live voxels[] top-5.
"""
import json, os
import numpy as np

WEIGHTS = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth'
RES = 0.5   # SCN_Vertex.py default; the C++ never passes resolution


def load_cloud(arm, evt):
    """the exact live SCN input: (x, y, z, q) float32 arrays, in build order."""
    p = os.path.join(arm, f'pr_evt{evt}', f'calib-pr-evt{evt}.json')
    b = json.load(open(p))['vertex_scoreboard']
    c = b.get('hv_cloud')
    if not c:
        raise KeyError(f'{p}: no hv_cloud (dl_vtx_harvest was off in this arm)')
    return (np.asarray(c['x'], dtype=np.float32), np.asarray(c['y'], dtype=np.float32),
            np.asarray(c['z'], dtype=np.float32), np.asarray(c['q'], dtype=np.float32)), b


def infer(cloud, top_k=5, weights=WEIGHTS):
    """-> list of (x, y, z, dl_score), rank-ordered, in cm.  Calls the shipped module."""
    from SCN_Vertex import SCN_Vertex
    x, y, z, q = cloud
    out = SCN_Vertex(weights, x.tobytes(), y.tobytes(), z.tobytes(), q.tobytes(),
                     'float32', int(top_k))
    # the shipped function returns PACKED float32 BYTES (the C++ side unpacks the
    # same way in pyutil/src/SCN_Vertex.cxx), not a python sequence.
    a = np.frombuffer(out, dtype=np.float32).astype(np.float64).reshape(-1, 4)
    return [tuple(r) for r in a]


def n_voxels(cloud, resolution=RES):
    x, y, z, _ = cloud
    c = np.stack((x, y, z), axis=1)
    c = ((c - c.min(axis=0)) / resolution).astype(np.int64)
    return len(np.unique(c, axis=0))


def field(cloud, weights=WEIGHTS):
    """the FULL per-voxel score map: (N,4) array of x, y, z, score."""
    k = n_voxels(cloud)
    return np.asarray(infer(cloud, top_k=k, weights=weights), dtype=np.float64)

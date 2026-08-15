'''
Voxelization -- voxelize() is a VERBATIM copy of toolkit
pyutil/python/SCN_Vertex.py:29-52 (apply-pointcloud HEAD, 2026-08-14).
Keep the float32 arithmetic: min-subtraction happens on the float32 cloud
before /resolution and int64 truncation, and a float64 rewrite can put
boundary points in a different voxel than production.

Helpers below reproduce the surrounding inference conventions
(SCN_Vertex.py:66-108): per-axis min offset, voxel->cm centre
idx*res + offset + res/2, top-K by descending score.

Truth target: Gaussian falloff exp(-(d/sigma)^2/2) around the hand-scan
vertex, the same shape as the original uBooNE training
(uboone-dl-vtx util/loader.py dist2prob).  Divergence from the original,
stated: we evaluate the Gaussian at voxel centres after voxelization instead
of per input point before it (difference is at most half a voxel diagonal,
and it makes the target independent of point multiplicity inside a voxel).
'''
import numpy as np

RESOLUTION = 0.5  # cm -- matches the weight file name res0.5; never passed
                  # from C++, always the SCN_Vertex.py default.


def voxelize(x, y, resolution=0.5):
    if len(x.shape) != 2:
        raise Exception('x should have 2 dims')

    x = x - x.min(axis=0)
    x = (x / resolution).astype(np.int64)

    # Vectorized: find unique voxels and accumulate features
    # For feature dim 0: average; for dims 1+: max (matches original logic)
    unique_coords, inverse = np.unique(x, axis=0, return_inverse=True)
    n_voxels = len(unique_coords)
    n_feat = y.shape[1]
    ft_out = np.zeros((n_voxels, n_feat), dtype=y.dtype)

    # Average first feature
    np.add.at(ft_out[:, 0], inverse, y[:, 0])
    counts = np.bincount(inverse, minlength=n_voxels).astype(y.dtype)
    ft_out[:, 0] /= counts

    # Max remaining features
    for i in range(1, n_feat):
        np.maximum.at(ft_out[:, i], inverse, y[:, i])

    return unique_coords, ft_out


def voxelize_event(xyz, q, resolution=RESOLUTION):
    """Production-identical voxelization of one event.

    xyz: (N,3) float32 cm; q: (N,) float32 already-transformed charge
    feature (dQ*dQdx_scale + dQdx_offset).
    Returns (coords int64 (V,3), ft float32 (V,1), offset float32 (3,)).
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    ft = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, 1)
    offset = xyz.min(axis=0)
    coords, ft_out = voxelize(xyz, ft, resolution=resolution)
    return coords, ft_out, offset


def voxel_center_cm(coords, offset, resolution=RESOLUTION):
    """Voxel index -> cm centre, exactly SCN_Vertex.py:89-91/100-101."""
    return coords.astype(np.float32) * np.float32(resolution) \
        + (offset + 0.5 * resolution).astype(np.float32)


def gaussian_truth(coords, offset, truth_xyz, sigma=1.0, resolution=RESOLUTION):
    """Per-voxel training target in [0,1]: exp(-(d/sigma)^2/2), d = distance
    (cm) from the voxel centre to the truth vertex."""
    centers = voxel_center_cm(coords, offset, resolution)
    d = np.linalg.norm(centers - np.asarray(truth_xyz, dtype=np.float32), axis=1)
    return np.exp(-0.5 * (d / float(sigma)) ** 2).astype(np.float32)


def top_k_voxels(pred, coords, offset, k=5, resolution=RESOLUTION):
    """Top-K (x,y,z,score) in cm, descending score -- SCN_Vertex.py:96-105."""
    k = min(k, len(pred))
    idx = np.argpartition(pred, -k)[-k:]
    idx = idx[np.argsort(pred[idx])[::-1]]
    out = np.empty((k, 4), dtype=np.float32)
    out[:, :3] = voxel_center_cm(coords[idx], offset, resolution)
    out[:, 3] = pred[idx]
    return out

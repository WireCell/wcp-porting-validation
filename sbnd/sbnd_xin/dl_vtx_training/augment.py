'''
doc pr/77 -- augmentation menu.  Everything operates on the raw cm cloud
(and the truth vertex) BEFORE voxelization; the voxelizer's per-axis
min-subtraction renormalizes translations/reflections automatically.

- Reflections X->-X, Y->-Y, both (owner's request): detector symmetries for
  the net input (topology + charge only).  NO Z flip: beam-direction
  asymmetry is real physics the net may legitimately use.
- Sub-voxel jitter: translate cloud+truth by a random fraction of the 0.5 cm
  pitch per axis.  A pure translation is a no-op after min-subtraction
  EXCEPT for the voxel-boundary phase, so this yields genuinely different
  voxelizations of the same event for free, fresh every epoch.
- Charge jitter: global gain factor N(1,5%) x per-point N(1,2%) on the RAW dQ
  scale.  Note q_feature = dQ*0.1-1000 is affine, so the jitter is applied to
  (q_feature - offset), not q_feature itself.
- Point dropout (default off): drop a random fraction of points.

Rejected on purpose: rotations (wire-plane geometry is not rotation
symmetric), Z flip (above), event mixup (unphysical).
'''
import numpy as np

FLIPS = ((1, 1), (-1, 1), (1, -1), (-1, -1))  # (sx, sy) applied to (x, y)


def apply_flip(xyz, truth_xyz, flip):
    sx, sy = flip
    s = np.array([sx, sy, 1], dtype=np.float32)
    return xyz * s, truth_xyz * s


def subvoxel_jitter(xyz, truth_xyz, rng, resolution=0.5):
    shift = rng.uniform(0.0, resolution, size=3).astype(np.float32)
    return xyz + shift, truth_xyz + shift


def charge_jitter(q, rng, dQdx_offset=-1000.0, global_sigma=0.05, point_sigma=0.02):
    raw = q - np.float32(dQdx_offset)  # back to dQ*scale (>= 0 up to fit noise)
    g = rng.normal(1.0, global_sigma)
    p = rng.normal(1.0, point_sigma, size=raw.shape)
    return (raw * np.float32(g) * p.astype(np.float32)) + np.float32(dQdx_offset)


def point_dropout(xyz, q, rng, max_frac=0.10):
    n = len(q)
    keep = rng.random(n) >= rng.uniform(0.0, max_frac)
    if keep.sum() < 2:  # never empty the cloud
        return xyz, q
    return xyz[keep], q[keep]


def augmented_views(n_events, use_flips=True):
    """Expand an event index range into (event_idx, flip) sample pairs."""
    flips = FLIPS if use_flips else FLIPS[:1]
    return [(i, f) for i in range(n_events) for f in flips]

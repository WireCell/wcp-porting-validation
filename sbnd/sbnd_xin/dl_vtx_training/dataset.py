'''
doc pr/77 -- dataset over a build_dataset.py snapshot.

One sample = one (event, reflection) view.  Reflections are enumerated
(deterministic x4); sub-voxel jitter / charge jitter / dropout are drawn
fresh per __getitem__ from a per-worker rng, so every epoch sees new
voxel-phase variants.  Voxelization + Gaussian truth happen here, after
augmentation, exactly as production would voxelize.

Batching stays per-event (batch=1), matching the original uBooNE training.
'''
import csv
import os
import numpy as np

from augment import (FLIPS, apply_flip, subvoxel_jitter, charge_jitter,
                     point_dropout)
from scn_vtx.voxelize import voxelize_event, gaussian_truth, RESOLUTION


class VtxSamples:
    """Not a torch.utils.data.Dataset on purpose: samples are variable-size
    sparse tensors consumed one at a time; a plain indexable object keeps the
    loop transparent."""

    def __init__(self, snapshot_dir, event_rows, sigma=1.0, use_flips=True,
                 jitter=True, q_jitter=True, dropout=False, train=True,
                 seed=0, hard_negative=0.0):
        self.dir = snapshot_dir
        self.rows = list(event_rows)
        self.sigma = float(sigma)
        self.flips = FLIPS if (use_flips and train) else FLIPS[:1]
        self.jitter = jitter and train
        self.q_jitter = q_jitter and train
        self.dropout = dropout and train
        # S8b: negative Gaussian at the production pick on corrective events
        # (>1 cm from truth); only applied to TRAINING targets.
        self.hard_negative = float(hard_negative) if train else 0.0
        self.rng = np.random.default_rng(seed)
        self.index = [(i, f) for i in range(len(self.rows)) for f in self.flips]

    def __len__(self):
        return len(self.index)

    def event_of(self, k):
        return self.rows[self.index[k][0]]

    def __getitem__(self, k):
        i, flip = self.index[k]
        row = self.rows[i]
        with np.load(os.path.join(self.dir, row['npz'])) as f:
            xyz = f['xyz'].astype(np.float32)
            q = f['q'].astype(np.float32)
            truth = f['truth_xyz'].astype(np.float32)
            prod = f['prod_xyz'].astype(np.float32) if 'prod_xyz' in f \
                else np.full(3, np.nan, np.float32)
            dQdx_offset = float(f.get('dQdx_offset', -1000.0)) \
                if 'dQdx_offset' in f else -1000.0

        # transform truth and the production pick identically with the cloud
        tp = np.stack([truth, prod])
        xyz, tp = apply_flip(xyz, tp, flip)
        if self.dropout:
            xyz, q = point_dropout(xyz, q, self.rng)
        if self.jitter:
            xyz, tp = subvoxel_jitter(xyz, tp, self.rng)
        if self.q_jitter:
            q = charge_jitter(q, self.rng, dQdx_offset=dQdx_offset)
        truth, prod = tp[0], tp[1]

        neg = None
        if self.hard_negative > 0 and np.isfinite(prod).all() \
                and np.linalg.norm(prod - truth) > 1.0:
            neg = prod
        coords, ft, offset = voxelize_event(xyz, q)
        target = gaussian_truth(coords, offset, truth, sigma=self.sigma,
                                neg_xyz=neg, neg_lambda=self.hard_negative)
        return coords, ft, target, dict(row=row, flip=flip, offset=offset,
                                        truth=truth)


    def consistency_views(self, k):
        """S8c consistency regularization: two independently sub-voxel-
        jittered voxelizations of the same (event, flip) view, each with the
        per-POINT voxel row index (every original point maps to exactly one
        voxel in each view), so the two score fields can be compared
        point-by-point without any cross-lattice interpolation.  Needs no
        label -- the target never enters."""
        i, flip = self.index[k]
        row = self.rows[i]
        with np.load(os.path.join(self.dir, row['npz'])) as f:
            xyz = f['xyz'].astype(np.float32)
            q = f['q'].astype(np.float32)
            dQdx_offset = float(f.get('dQdx_offset', -1000.0)) \
                if 'dQdx_offset' in f else -1000.0
        dummy = np.zeros(3, dtype=np.float32)
        xyz, _ = apply_flip(xyz, dummy, flip)
        out = []
        for _ in range(2):
            xyzj, _ = subvoxel_jitter(xyz, dummy, self.rng)
            qj = charge_jitter(q, self.rng, dQdx_offset=dQdx_offset) \
                if self.q_jitter else q
            coords, ft, _ = voxelize_event(xyzj, qj)
            # replicate voxelize()'s float32 arithmetic exactly
            idx = ((xyzj - xyzj.min(axis=0)) / RESOLUTION).astype(np.int64)
            lut = {tuple(c): m for m, c in enumerate(coords)}
            pt = np.fromiter((lut[tuple(t)] for t in idx), dtype=np.int64,
                             count=len(idx))
            out.append((coords, ft, pt))
        return out


def load_manifest(snapshot_dir, drop_lockbox=False):
    path = os.path.join(snapshot_dir, 'manifest.tsv')
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    for r in rows:
        r['evt'] = int(r['evt'])
        r['corrective'] = int(r['corrective'])
        # round-2 columns; absent from round-1 snapshots (practice66)
        r['lockbox'] = int(r.get('lockbox') or 0)
        r.setdefault('sample', '')
    if drop_lockbox:
        rows = [r for r in rows if not r['lockbox']]
    return rows


def kfold_split(rows, kfold, seed=0):
    """Stratified by (sample, corrective) so every fold sees each class of
    each sample.  Yields (fold_idx, train_rows, val_rows)."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for r in rows:
        by_class.setdefault((r.get('sample', ''), r['corrective']), []).append(r)
    folds = [[] for _ in range(kfold)]
    for key in sorted(by_class):
        cls_rows = by_class[key]
        order = rng.permutation(len(cls_rows))
        for j, idx in enumerate(order):
            folds[j % kfold].append(cls_rows[idx])
    for i in range(kfold):
        val = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        yield i, train, val

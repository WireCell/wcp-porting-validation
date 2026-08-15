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
from scn_vtx.voxelize import voxelize_event, gaussian_truth


class VtxSamples:
    """Not a torch.utils.data.Dataset on purpose: samples are variable-size
    sparse tensors consumed one at a time; a plain indexable object keeps the
    loop transparent."""

    def __init__(self, snapshot_dir, event_rows, sigma=1.0, use_flips=True,
                 jitter=True, q_jitter=True, dropout=False, train=True,
                 seed=0):
        self.dir = snapshot_dir
        self.rows = list(event_rows)
        self.sigma = float(sigma)
        self.flips = FLIPS if (use_flips and train) else FLIPS[:1]
        self.jitter = jitter and train
        self.q_jitter = q_jitter and train
        self.dropout = dropout and train
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
            dQdx_offset = float(f.get('dQdx_offset', -1000.0)) \
                if 'dQdx_offset' in f else -1000.0

        xyz, truth = apply_flip(xyz, truth, flip)
        if self.dropout:
            xyz, q = point_dropout(xyz, q, self.rng)
        if self.jitter:
            xyz, truth = subvoxel_jitter(xyz, truth, self.rng)
        if self.q_jitter:
            q = charge_jitter(q, self.rng, dQdx_offset=dQdx_offset)

        coords, ft, offset = voxelize_event(xyz, q)
        target = gaussian_truth(coords, offset, truth, sigma=self.sigma)
        return coords, ft, target, dict(row=row, flip=flip, offset=offset,
                                        truth=truth)


def load_manifest(snapshot_dir):
    path = os.path.join(snapshot_dir, 'manifest.tsv')
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    for r in rows:
        r['evt'] = int(r['evt'])
        r['corrective'] = int(r['corrective'])
    return rows


def kfold_split(rows, kfold, seed=0):
    """Stratified by `corrective` so every fold sees both classes.
    Yields (fold_idx, train_rows, val_rows)."""
    rng = np.random.default_rng(seed)
    by_class = {0: [], 1: []}
    for r in rows:
        by_class[r['corrective']].append(r)
    folds = [[] for _ in range(kfold)]
    for cls_rows in by_class.values():
        order = rng.permutation(len(cls_rows))
        for j, idx in enumerate(order):
            folds[j % kfold].append(cls_rows[idx])
    for i in range(kfold):
        val = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        yield i, train, val

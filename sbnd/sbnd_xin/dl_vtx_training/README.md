# dl_vtx_training — SBND DL neutrino-vertex fine-tuning (doc pr/77)

Standalone PyTorch pipeline to fine-tune the SCN `DeepVtx` vertex model on
hand-scan labels (pr/75 scan panel), matching the production inference path
bit-for-bit where it matters.  **Python-only: no toolkit C++/config change.**

## Environment

Runs with the direnv python on wcgpu1 (torch 2.5.1+cu121, `sparseconvnet`
from `toolkit-dev/SparseConvNet`, 2x RTX 4090).  Everything also works on
CPU (`--device cpu`); production inference is CPU.

## Data flow

```
vertex_labels/<tag>/labels-evt<ID>.json      (hand scan; rank-1 pick = truth)
        +                                     [read-only, M13]
work-*/pr_evt<ID>/calib-pr-evt<ID>.json      (PrDisplayDump; vertices[].fit +
        |                                     segments[].points + scoreboard)
        v
build_dataset.py  ->  data/<name>/evt<ID>.npz + manifest.tsv   (frozen snapshot)
        v
train.py          ->  runs/<name>/fold<k>/CP<E>.pth + log.tsv  (fine-tune)
        v
evaluate.py       ->  baseline-vs-tuned metrics (out-of-fold)
```

The net input cloud is rebuilt per `NeutrinoVertexFinder.cxx:4147-4179`:
every PR-graph vertex fit point + every segment interior fit point, cm,
`q = dQ*dQdx_scale + dQdx_offset` (scale/offset read from the calib `meta`).
`parity_check.py` validates this premise: on practice66 the rebuilt cloud
reproduces production's recorded top-1 voxel exactly on 39/66 events
(median deviation = one 0.5 cm voxel); the tail is the post-DL-refit
approximation, largest exactly on the corrective events.

## One-command examples

```bash
cd sbnd_xin/dl_vtx_training

# 0. sanity: production parity of the rebuilt input cloud
python3 parity_check.py --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0

# 1. freeze a training snapshot (refuses to overwrite an existing one)
python3 build_dataset.py --name practice66 \
    --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0

# 2. fine-tune, 6-fold CV, x4 reflections + sub-voxel jitter + charge jitter
python3 train.py --data data/practice66 --name ft0 --kfold 6 --epochs 30

# 3. out-of-fold evaluation vs the uBooNE baseline
python3 evaluate.py --data data/practice66 --run runs/ft0 --tsv runs/ft0/eval.tsv

# 4. selection stage: closure + operating-point grid (no wire-cell needed)
python3 rerank_replay.py --closure
python3 rerank_replay.py --grid --tsv runs/rerank-grid.tsv

# 5. Tier B: the net's charge-input distribution
python3 qfeature_check.py --png runs/qfeature.png
```

## Conventions inherited from the original uBooNE t48k campaign

(github.com/HaiwangYu/uboone-dl-vtx; the production weight file
`t48k-m16-l5-lr5d-res0.5-CP24.pth` is checkpoint 24 of it.)

- truth: Gaussian falloff `exp(-(d/sigma)^2/2)` around the true vertex
  (`--sigma`, default 1.0 cm);
- loss: `MSELoss(pred[:,1]-pred[:,0], truth)` — the same sigmoid-difference
  score used at inference;
- optimizer: Adam, `lr = lr0*exp(-lrd*epoch)`; fine-tune default
  `lr0=1e-6` (10x below the original), `lrd=0.05`;
- per-event batches, checkpoint per epoch.

Fine-tune additions: `--freeze head` (default) trains only the linear head +
final BatchNormReLU + last UNet decoder block (~7k of 7.2M params);
`--freeze linear` / `--freeze none` for probe / full training.

## Augmentation (`augment.py`)

- reflections X->-X, Y->-Y, both (x4, deterministic);
- sub-voxel jitter: shift cloud+truth by a random fraction of the 0.5 cm
  pitch — a translation is a no-op after the voxelizer's min-subtraction
  *except* for the boundary phase, so this is a free new voxelization every
  epoch;
- charge jitter (gain-systematic robustness), optional point dropout.
- No Z flip (beam-direction asymmetry is physics), no rotations (wire-plane
  geometry).

## Deployment of a fine-tuned checkpoint

`runs/<name>/fold<k>/CP<E>.pth` is a squeezed-3D state_dict that the
**untouched** in-tree `SCN_Vertex._load_model` loads bit-identically
(round-trip proven in doc pr/77).  To deploy: copy under a NEW name into
`wire-cell-data/` (never overwrite the uBooNE file) and point the
`dl_weights` TLA at it — config-only, A/B-able, owner-gated.

## Guard rails

- `vertex_labels/` is read-only here (M13); snapshots freeze label mtimes in
  `manifest.tsv`; `build_dataset.py` refuses to overwrite a snapshot.
- `data/` and `runs/` outputs are not committed.
- The 51 confirming labels are a do-no-harm bar: `evaluate.py` flags any
  event where the tuned net is >1 cm worse than baseline (`guard_fail`).
- With only 15 corrective labels in practice66, practice runs are pipeline
  shakedowns, not physics results; the serious round waits for the mcp1k
  scan (`vtxscan-prod0813-mcp1k`, live).

# prod-2026-09-20 — the PDHD/PDVD trajectory becomes the SBND production operating point

## Why this generation exists

> "Let's flip as the SBND production run ... the decision comes from better track trajectory as
> well as better vertex accuracy."
> "Here flip means to go against your earlier recommendation, but I feel we see enough benefits."
> -- the owner, 2026-09-20

Doc `sbnd_xin/docs/118_sbnd-trajectory-flip-and-pr-profiling-plan.md`; toolkit commit `675fd266`.  This flip OVERRIDES doc 116 sec 15.4 ("No -- not on this evidence") and doc 117
sec 11's frozen HOLD; doc 118 sec 1 records the override as an override and states what remains
unconfirmed.  Previous generation `ref/prod-2026-09-17b` is kept and still describes the pre-flip
operating point.

## Gate evidence

`prod_cfg_gate.py --ref ref/prod-2026-09-17b` was **PASS 21/21 at unmodified HEAD** before the
first edit, so the drift below has no inherited component.  Against this generation the tree is
**PASS 24/24**.

## The drift from prod-2026-09-17b -- 3 of 21 artifacts, all SBND

`bare_prjob.json`, `prod_prjob.json`, `sbnd_pr.json`.  uBooNE, PDHD and PDVD are byte-identical.
Key by key on the SBND PR job ([11]/[12] = the two retile BlobSamplers, [13] = ImproveCluster_2:pr,
[14] = CreateSteinerGraph:pr, [20] = CreateSteinerGraph:prrefresh):

```
  REMOVED [11].data.strategy[0] = 'stepped'
  ADDED   [11].data.strategy[0].disable_mix_dead_cell = False
  ADDED   [11].data.strategy[0].name = 'charge_stepped'
  CHANGED [11].name : 'live-apa0-0' -> 'live-cs-apa0-0'
  REMOVED [12].data.strategy[0] = 'stepped'
  ADDED   [12].data.strategy[0].disable_mix_dead_cell = False
  ADDED   [12].data.strategy[0].name = 'charge_stepped'
  CHANGED [12].name : 'live-apa1-0' -> 'live-cs-apa1-0'
  CHANGED [13].data.samplers[0].name : 'BlobSampler:live-apa0-0' -> 'BlobSampler:live-cs-apa0-0'
  CHANGED [13].data.samplers[1].name : 'BlobSampler:live-apa1-0' -> 'BlobSampler:live-cs-apa1-0'
  ADDED   [14].data.base_weight_blank_alpha = 0.5
  ADDED   [14].data.base_weight_scope = 'tree+path'
  ADDED   [14].data.terminal_blank_plane_mode = 'prefer3'
  ADDED   [20].data.base_weight_blank_alpha = 0.5
  ADDED   [20].data.base_weight_scope = 'tree+path'
  ADDED   [20].data.terminal_blank_plane_mode = 'prefer3'
```

## What the keys do

* `charge_stepped` on the RETILE samplers only -- the Steiner cloud every PR stage reads; the
  clustering job's own 3-D point cloud is untouched.  The prototype's retile rule; PDHD/PDVD
  production since docs pdvd/108 / 103.
* `terminal_blank_plane_mode='prefer3'` -- blank-plane admission for the Steiner terminal
  candidates (doc pdvd/114), on BOTH steiner passes.
* `base_weight_blank_alpha=0.5` / `base_weight_scope='tree+path'` -- charge-aware pricing of the
  Steiner base graph before the Voronoi step (doc pdvd/115), on both passes.
* NOT VISIBLE HERE, and that is the point: `fit_weight_pow 1.5` and `assoc_cont_center 1` were
  added to `cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`, which is read at RUNTIME and
  moved ZERO of the 21 artifacts.  This generation therefore carries THREE NEW ARTIFACTS,
  22-24: `sbnd_track_fitting.json`, `pdhd_track_fitting.json`, `pdvd_track_fitting.json`, hashed
  as bytes by `scripts/cfg/compile_consumers.sh` step (f).  A flip of that family can no longer
  pass this gate silently.

## Reproduce

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-20      # PASS 24/24
```

`prod_prjob.json` is committed with `git add -f` (`*.json` is gitignored).

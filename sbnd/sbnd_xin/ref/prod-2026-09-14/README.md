# ref/prod-2026-09-14 — SBND production operating point

**Why this generation exists.** Doc 109 turned the self-describing `tracking-pr.root` ON
in the SBND production job (owner's word, 2026-09-14; toolkit `c203b400`). The previous
generation, `ref/prod-2026-09-08`, no longer matched production **even at the unmodified
toolkit**: four artifacts had drifted through the 2026-09-10 master merge. The owner
asked (2026-09-14) for a new generation and for the stale one to be removed.

Toolkit commit: `c203b4002c27e74da489989dc8fd32d16c87125b`. Gate:
`scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14` → **PASS, 21/21 artifacts**
(cfg tree = `git archive c203b400 cfg`).

`prod_prjob.json` is committed with `git add -f` (`*.json` is gitignored). The 09-08
generation's copy was never tracked, which is why its drift below had to be recovered by
recompiling instead of read from the reference.

## The drift from `prod-2026-09-08` — 5 of 21 artifacts

```
DRIFT : prod.standalone, prod_prjob.json, sbnd_clus.json, sbnd_ql.json, sbnd_simcheck.json
```

**The 16 that did not move:** every pdhd and pdvd job, `uboone.json`, `prod.wcls`,
`sbnd_img.json`, `sbnd_pr.json` and `bare_prjob.json`. The last two compile pipelines
without the tagger, tracking-visitor or tagger-output stages (`sbnd_pr` stops at
`stm_magnify`; `bare_prjob` uses the default pipeline), so doc 109's keys have no node to
land on.

**How the drift was named.** `ref/prod-2026-09-08` stores only hashes for four of these.
1. The 21 artifacts were compiled from `git archive eacacafe cfg`, the commit 09-08 was
   cut at. They reproduce 09-08 **21/21**, so every drift comes from `eacacafe..c203b400`.
2. That compile was diffed key by key against `c203b400`.
3. Each key was attributed with `git log -S/-G`.

### 1. `prod_prjob.json` — doc 109, on the owner's word (2026-09-14)

| node | key | value |
|---|---|---|
| `[21]` TaggerCheckNeutrino | `nu_provenance` | `true` |
| `[24]` SbndPrMagnifyTrackingVisitor | `nu_provenance`, `fix_cluster_flags` | `true`, `true` |
| `[24]` SbndPrMagnifyTrackingVisitor | `provenance` | `{bdt_weights_dir: 'uboone/weights', numu_xgboost_xml: 'uboone/weights/numu_scalars_scores_0923.xml', nue_xgboost_xml: 'uboone/weights/XGB_nue_seed2_0923.xml', dl_weights: '', trackfitting_config: 'pgrapher/experiment/sbnd/sbnd_track_fitting.json'}` |
| `[25]` UbooneTaggerOutputVisitor | `nu_provenance` | `true` |

- **Commits:** `b207b6d8` (C++ and jsonnet knobs, default OFF) and `c203b400` (the three
  `wct-pr-perevt.jsonnet` TLAs `root_nu_record`, `root_cluster_flags`, `root_provenance`
  default true).
- **Nine ADDED keys, 0 removed, 0 changed.**
- **Effect:** reconstruction outputs are byte-identical; only `tracking-pr.root` gains
  content (doc 109 sec 4).
- **`dl_weights: ''`** is the compile harness's value (`compile_prjob_cfg.sh` passes
  `-A dl_weights=`). A production run records the weights it was given.

### 2. `prod.standalone`, `sbnd_clus.json`, `sbnd_ql.json` — inherited, not a round of ours

Each gains exactly one key on its `MultiAlgBlobClustering:clus_all_apa` node:

```
ADDED [..].data.bee_points_sets[1].opflash_time = true
```

- **What it is:** a per-point `opflash_time` column (the cluster's matched flash time) in
  the second Bee point set.
- **Origin:** commit `b31a0db0` (2026-08-06, "clus: add optional per-point opflash_time
  column to Bee point sets").
- **How it arrived:** `apply-pointcloud` picked it up through the master merge `98140fee`
  (2026-09-10).
- **Effect:** a Bee-output column. Clustering, matching and the pctree are not what it
  configures.
- **Not owner-validated:** no sbnd_xin round validated or recorded it. It is recorded here
  as inherited.

### 3. `sbnd_simcheck.json` — inherited SP retune, not a round of ours

On both `OmnibusSigProc` nodes (`apa0sigproc0` `[37]`, `apa1sigproc1` `[48]`):

```
ADDED   roi_mad_rms = true
ADDED   r_break_roi_loop_planes = [2, 2, 0]
CHANGED troi_col_th_factor : 5 -> 3
CHANGED troi_ind_th_factor : 3 -> 1.8
```

- **Origins:**
  - `b8086bd6` (HaiwangYu, 2026-08-13, "sigproc+sbnd: stop deleting long collection-plane
    signals (ai-helper #10)") adds `roi_mad_rms` and `r_break_roi_loop_planes`.
  - `06a02ccb` (2026-08-13, "cfg/sbnd: merge sp.jsonnet with the sbndcode fork") moves
    the two thresholds.
  - Both arrived through `98140fee` (2026-09-10).
- **Scope:** this is SBND signal processing from raw ADC, i.e. the sim-check job.
  `sbnd_img.json`, which starts from SP frames, did not move. So these keys change
  imaging/clustering inputs only for a chain that runs SP itself.
- **Not owner-validated:** no sbnd_xin round validated or recorded them.

## Samples at this operating point

- **Per-event runs:** doc 109's arms (`work-*-d109on2`, `d109dlon2`, `d109prod`) ran the
  PR job at this `prod_prjob.json`.
- **Group-mode runs:** doc 109 rev 2's arms (`work-*-d109grp`, `d109dlgrp`) did the same.
- **Not at this operating point:** the full-population production samples
  `work-*-d102m` / `d102mpr` (products `prod0908`). They are at `eacacafe` = the removed
  `prod-2026-09-08`.

## The removed generation `prod-2026-09-08`

It was the generation record of `work-*-d102m` / `d102mpr` and `products/prod0908/`.

- **Where it survives:** in git at wcp `70a49e8c`:
  `git show 70a49e8c:sbnd/sbnd_xin/ref/prod-2026-09-08/README.md`.
- **How to reproduce it:** its operating point is `git archive eacacafe cfg`, which passes
  its `consumers.sha256` 21/21 (re-checked 2026-09-14).

## Reproduce

```bash
cd wcp-porting-img/sbnd/sbnd_xin
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14                      # PASS 21/21 at c203b400
# the drift table above
git -C ../../../toolkit archive eacacafe cfg | tar -x -C <scratch>/cfg-eacacafe
git -C ../../../toolkit archive c203b400 cfg | tar -x -C <scratch>/cfg-head
git -C .. show 70a49e8c:sbnd/sbnd_xin/ref/prod-2026-09-08/consumers.sha256 > <scratch>/ref0908/consumers.sha256
scripts/cfg/prod_cfg_gate.py --ref <scratch>/ref0908 --cfg <scratch>/cfg-eacacafe/cfg --keep <scratch>/comp-eacacafe   # PASS 21/21
scripts/cfg/compile_consumers.sh <scratch>/cfg-head/cfg <scratch>/comp-head
python3 docs/109_logs/r2/drift_keys.py <scratch>/comp-eacacafe <scratch>/comp-head \
        prod.standalone sbnd_clus.json sbnd_ql.json sbnd_simcheck.json prod_prjob.json
```

The `gate308-*.txt` event lists are carried forward unchanged from `prod-2026-09-08`
(itself from `prod-2026-09-04`).

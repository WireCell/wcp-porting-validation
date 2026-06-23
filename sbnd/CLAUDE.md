- base knowledge: /exp/sbnd/app/users/yuhw/claude-utilities
  - especially how to run things in sl7: /exp/sbnd/app/users/yuhw/claude-utilities/wct-in-sl7.md
- in sl7, use a setup sript to setup ups products, LD_LIBRARY_PATH, WIRECELL_PATH etc.
- local WCT: /exp/sbnd/app/users/yuhw/wire-cell-toolkit (branch `apply-pointcloud`; do NOT commit/push on my own — keep edits local for review)
- local larwirecell: /exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/srcs/larwirecell (MRB tree; the ONLY larwirecell tree to use — do NOT use /exp/sbnd/app/users/yuhw/larwirecell)

## Running in SL7

Everything runs inside the apptainer; non-interactive shells need `.bashrc` first:
```
/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec \
  -B /cvmfs,/exp,/nashome,/pnfs /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest bash -c '
  source /nashome/y/yuhw/.bashrc
  source <setup script>
  cd /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd
  <lar / wire-cell ...> '
```
- `setup-local-opt.sh` — legacy `opt` install; sbndcode cfg wins (use for sim and the OLD matching chain).
- `setup-ap.sh` — AP matching/imaging env: setup-local-opt.sh + prepend toolkit cfg (so toolkit img/clus/qlmatching/simparams win, Xin's env) + sbnd_xin + wire-cell-data/sbnd/photodet. Use ONLY for the AP chain, not sim.

## Imaging+clustering+QL-matching chains

- OLD (obsolete): `obsolete/wcls-img-clus-matching.{fcl,jsonnet}` — larwirecell `wclsQLMatching`, sbndcode `img.jsonnet`
  (`img_config` e.g. `active2view+masked2view` / `active3view+masked1view`). Run with setup-local-opt.sh.
  Output: per-node `mabc-*.zip` + `data-sep/`; upload via `./bee-upload.sh` (merges -> combined.zip).
  Superseded by the XIN-faithful chain below.
- XIN-faithful: `wcls-img-clus-matching-xin.{fcl,jsonnet}` — artROOT input but toolkit `img` multi-3view+full_deghost,
  toolkit `clus` per_apa, joint `QLMatching` (FlashTensorToOpticalPCs + WireCellMatch), single shared `mabc.zip`.
  Run with setup-ap.sh; upload via `BROWSER=echo bash sbnd_xin/upload-to-bee.sh mabc.zip`.
  Downstream pattern-rec toggle: `enable_downstream_pr` (top of toolkit `pgrapher/experiment/sbnd/clus.jsonnet`):
  true = full patrec (tagger/steiner/vertices/mc); false = matching-only. Bulk runs: use false — full patrec has
  data-dependent crashes on some events (no main_cluster; missing steiner_pc). Do NOT modify anything in sbnd_xin.

## Where details live

- `cm-2606/STATUS-xin-chain.md` — full status: chains, toggle, all local WCT edits, known issues, BEE uploads.
- Quick CPU/mem profiling: `/exp/sbnd/app/users/yuhw/activity_logger/top.sh <pattern>` (run concurrently); plot example in `cm-2606/activity/`.
- w-gap study (SP rebaseline, DNNROI truncation, charge bias): `standalone-sample/w-gap/W-GAP-STUDY.md`.

# Run wcls-img-clus-matching-xin.fcl with the local builds (SL7, SBND)

How to run the XIN-faithful imaging + clustering + Q/L-matching (+ truth
labeling) chain using the locally-built wire-cell-toolkit and larwirecell
from `/exp/sbnd/app/users/yuhw/opt` (build them first:
`docs/0-build-wct-larwirecell-sl7-sbnd.md`).  SBND-specific throughout.

## What "local builds" means at run time

`sbnd/setup-ap.sh` (which sources `setup-local-opt.sh`) wires everything:

- **WCT libs**: `opt/lib` prepended to `LD_LIBRARY_PATH`/`CET_PLUGIN_PATH`
  (plugins `WireCellClus`, `WireCellImg`, `WireCellMatch`, ...).
- **larwirecell libs**: `opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib`
  prepended (plugins `WireCellLarsoft`, `WireCellQLMatch`, `WireCellAIML`).
  Remember: these only update via the hand-copy step of doc 0.
- **WCT cfg (jsonnet)**: `setup-ap.sh` PREPENDS the toolkit SOURCE
  `wire-cell-toolkit/cfg` to `WIRECELL_PATH`, so source-tree edits of
  `pgrapher/experiment/sbnd/*.jsonnet` take effect with NO rebuild.
  `wcp-porting-img/sbnd` is also on `WIRECELL_PATH` (the entry jsonnet +
  fcl live there), plus `sbnd_xin`, `wire-cell-data`, photodet files.
- The fcl's `plugins:` list names the libs, `inputers:` the
  IArtEventVisitor components (`wclsCookedFrameSource:sigs`,
  `wclsOpFlashSource:tpc0/1`, `wclsTensorSetLabeler:clus_all_apa`).

Quick check the run really loads the local libs: `.so` mtimes should match
your last build/copy —
`ls -la /exp/sbnd/app/users/yuhw/opt/lib/libWireCellClus.so
        /exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/libWireCellAIML.so`

## MC run (reality=sim; truth labeler ON automatically)

Run from an output directory (e.g. `sbnd/TensorSetLabeler/`) — all outputs
land in cwd:

```bash
/exp/sbnd/app/users/yuhw/claude-utilities/in-gpvm-sl7.sh bash -c '
source /nashome/y/yuhw/.bashrc >/dev/null 2>&1
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-ap.sh >/dev/null 2>&1
export FHICL_FILE_PATH=/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd:$FHICL_FILE_PATH
cd /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/TensorSetLabeler
rm -f mabc.zip trash-all-apa.tar.gz
lar --nskip 0 -n 1 -c wcls-img-clus-matching-xin.fcl \
    -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/mc_paths-10files.lst \
    --no-output > run.log 2>&1
echo LAR_EXIT=$?
'
```

- `-n 1` smoke ≈ 1–2 min once the /pnfs file is staged (first read of a
  dCache file can be slow); 10 events ≈ 6 min.
- Local single-file sample (no dCache): `-s standalone-sample/2025f-mc.root`.
- Outputs: `mabc.zip` (shared Bee zip: `img`, `clustering`, `op`, `tgm`,
  `truth_trackid_labeled`, `truth_unlabeled`, `truth_depo_sce`, `mc` sets)
  and `trash-all-apa.tar.gz` (the labeled tensor output — actually written
  because `truth_labeler` disables the sink's `dump_mode`).
- Clean transient art droppings after a run (`tf-default.root`, `debug.csv`,
  `cputime.db`, `memory.db`, `errors.log`, `messages.log`) — per CLAUDE.md.

## Data run (reality=data; no truth)

```bash
# same env as above, then:
lar -n 100 -c wcls-img-clus-matching-xin-data.fcl \
    -S /exp/sbnd/app/users/yuhw/2025-fall-prod-sample/data_paths-v10_14_02_02-10files.lst \
    --no-output
```

`wcls-img-clus-matching-xin-data.fcl` #includes the sim fcl and overrides
`reality="data"`, the `sptpc2d` product tags, and drops the
`wclsTensorSetLabeler` inputer.  It is found via the `FHICL_FILE_PATH`
prepend above.

## Toggles

| toggle | where | meaning |
|---|---|---|
| `reality` | fcl `params` (`"sim"`/`"data"`) | pos_offset (data-only), truth labeler (sim-only) |
| `use_sce` | top of `wcls-img-clus-matching-xin.jsonnet` | pipeline scope: SCE true space `x_sce` (default) vs reco `x_t0cor` |
| `truth_labeler` | entry jsonnet, `= reality=='sim'` | insert wclsTensorSetLabeler after the all-APA MABC |
| `enable_downstream_pr` | toolkit `pgrapher/experiment/sbnd/clus.jsonnet` | full pattern-rec tail; keep FALSE for bulk (data-dependent crashes) |
| labeler knobs | sbnd `clus.jsonnet` labeler pnode | `sce_correction`, `pf_ke_min`, `pf_nu_only`, `truth_tracks_nu_only`, `n_sample_truth_depo_sce`, DL/DT/SP smearing, ... |

## Log sanity greps

```bash
grep "SCECorrection probe"    run.log   # reco->true probes, |d|~0.8-1cm cathode / 0.1cm anode
grep "SCE TrueFwd probe"      run.log   # labeler true->reco probes (opposite direction)
grep "aiml.*call="            run.log   # TensorSetLabeler: "labeled N/M blobs, K truth tracks"
grep "TGM] visit"             run.log   # TaggerCheckTGM tag counts
```

Expected on corsika+GENIE MC: 60–97% of blobs labeled per event, truth
tracks = the beam-nu primaries (0–20).

## Validation + BEE

```bash
# truth-labeling sanity (bee coherence + tensor content), host python3 ok:
python3 TensorSetLabeler/check_truth_trackid.py mabc.zip trash-all-apa.tar.gz

# TGM box plots:
python3 tgm-validation/summarize_tgm.py mabc.zip -o tgm-validation/tgm_points.npz
python3 tgm-validation/analyze_tgm.py tgm-validation/tgm_points.npz -o views.png

# upload (prints the event-list URL):
BROWSER=echo bash sbnd_xin/upload-to-bee.sh TensorSetLabeler/mabc.zip
```

## Jsonnet compile check without running lar

```bash
# inside SL7 + setup-ap.sh; ALL wcls extVars must be faked:
wcsonnet -V input_mask_tags=x -V output_mask_tags=x -V recobwire_tags=x \
         -V summary_tags=x -V trace_tags=x -V opflash0_input_label=x \
         -V opflash1_input_label=x -V reality=sim \
         wcls-img-clus-matching-xin.jsonnet > /tmp/compiled.json
```

Then inspect nodes/edges with python (e.g. confirm
`MultiAlgBlobClustering -> wclsTensorSetLabeler -> TensorFileSink`).

## Gotchas

- Always `export FHICL_FILE_PATH=<sbnd dir>:$FHICL_FILE_PATH` — the fcls
  live in `wcp-porting-img/sbnd`, not in any installed fcl path.
- Do NOT use `setup-local-opt.sh` alone for this chain (wrong cfg layering
  — sbndcode's `img.jsonnet` shadows the toolkit one); `setup-ap.sh` only.
- `mabc.zip` is opened once per job: remove/move it before each run, and
  never run two jobs in the same cwd concurrently.
- If a code change "has no effect": rebuild+install (WCT) or
  rebuild+hand-copy (larwirecell) per doc 0 — the run loads `opt`, not the
  build trees; check the `.so` mtimes and `LAR_EXIT` of the build step.
- jsonnet edits under `wire-cell-toolkit/cfg` need NO rebuild (source cfg
  is prepended); edits to `wcls-img-clus-matching-xin.{jsonnet,fcl}` are
  picked up directly from `wcp-porting-img/sbnd`.

## More context

- `docs/0-build-wct-larwirecell-sl7-sbnd.md` — the build/install recipe.
- `docs/claude-session-20260623-20260707.md` — TGM-era run notes.
- `docs/depo-blob-smearing.md` + larwirecell
  `aiml/docs/TensorSetLabeler-notes.md` — truth-labeling physics/pitfalls.
- `TensorSetLabeler/README.md` — labeler validation area + results.

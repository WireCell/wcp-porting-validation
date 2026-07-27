# 64 — Sync the SBND working config into the in-tree `cfg/pgrapher/experiment/sbnd/`

**Status: DONE 2026-07-27.** Every live SBND reconstruction job and every
pattern-recognition data file now lives in-tree; `sbnd_xin` re-exports them.
All compiled configs byte-identical, end-to-end run identical.
LArSoft (`wcls-*`) reconcile **deferred** — see §6.

## Repro block

```bash
# 1. compiled-config gate (the whole gate for this change: no binary changed)
#    compile_all.sh compiles the five live jobs with the EXACT TLA sets their
#    runners pass at the production operating point.
cd /home/xqian/tmp/sbnd-cfg-sync
./compile_all.sh before                                              # pre-change tree
./compile_all.sh after  $WCT/cfg/pgrapher/experiment/sbnd             # promoted in-tree copies
./compile_all.sh afterx                                               # sbnd_xin re-export shims
for f in before/*.json; do diff -q $f after/$(basename $f); diff -q $f afterx/$(basename $f); done

# 2. compiled-config proof of the new defaults (NO TLAs at all)
wcsonnet -o /tmp/pr.json $WCT/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
python3 -c "import json;[print(c['type'],{k:v for k,v in c['data'].items() if 'guard' in k or 'chord' in k}) for c in json.load(open('/tmp/pr.json')) if c.get('type','').startswith('TaggerCheck')]"

# 3. end-to-end same-code A/B against the validated round-5 arm
cd $SBND_DIR
STM_EVENTS="62613 62303 321371 402330 353487 72586" NJOBS=6 \
  ./stm_campaign/run_round.sh d64smoke
for e in 62613 62303 321371 402330 353487 72586; do
  diff work-stmcamp-d64smoke/nusel_evt$e/nusel-evt$e.tsv \
       work-stmcamp-r5fullc/nusel_evt$e/nusel-evt$e.tsv; done

# 4. unit tests
./build/clus/wcdoctest-clus       # 49 cases / 565 assertions, rc=0
```

## 1. What was actually out of sync

The *modules* were already canonical: `sbnd_xin/clus.jsonnet` and
`cathode_fiducial.jsonnet` were one-line re-exports, `qlmatching.jsonnet` a thin
wrapper, `img.jsonnet` imported in place. Two real gaps remained.

**(a) No runnable job lived in-tree.** `cfg/pgrapher/experiment/sbnd/` held no
`wct-*` reconstruction entry point at all (only `wct-sim-check.jsonnet` and the
`wcls-*` LArSoft jobs). The imaging, clustering, matching and PR graphs existed
only under `sbnd_xin/`, and the PR job imported two files from outside the
toolkit entirely: `../particle_dataset.jsonnet` (relative — resolved only from
the real, non-symlinked working dir) and `sbnd_track_fitting.json`.

**(b) The production operating point was encoded in shell flags, not config.**
`run_full1k_nusel.sh` runs with

```
-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc
```

while canonical `pr()` still defaulted those knobs to their pre-adoption values.
Anyone reading the in-tree config saw a TGM configuration nobody runs. The
doc-63 STM guards were the exception — they had already been adopted as module
defaults in `bd4bf0af`; this change brings the TGM/beam-window/LM knobs in line
with that precedent.

## 2. Defaults reconciliation

Owner decision 2026-07-27: production values become the **module** defaults.

`cfg/pgrapher/experiment/sbnd/clus.jsonnet` — `clus_pr()` and `pr()`:

| knob | was | now (production) | runner source |
|---|---|---|---|
| `tgm_neutrino_candidate` | `false` | `true` | `NUCAND:-1` |
| `tgm_chord_charge` | `false` | `true` | `-chord` |
| `tgm_component_extremes` | `false` | `true` | `-chord` (sets both) |
| `tgm_chord_mode` | `'chord'` | `'path'` | `CHORD_MODE:-path` |
| `tgm_component_rescue` | `false` | `true` | `-rescue` |
| `tgm_rescue_chord` | `false` | `true` | `-rescue-chord` |
| `tgm_main_pair` | `false` | `true` | `-main-pair-real` |
| `tgm_main_pair_mode` | `'path'` | `'real'` | `-main-pair-real` |
| `tgm_fv_zmax_margin` | `3` | `5` | `-fvz 5` |
| `tgm_fv_zmax_margin_interior` | `0` | `3` | `-fvzi 3` |
| `tgm_fv_x_margin` | `2` | `2.5` | `-fvx 2.5` |
| `tgm_fv_y_margin` | `2.5` | `3` | `-fvy 3` |
| `beam_window` | `[0,0]` | `[0.2, 2.2] us` | `BEAM_WINDOW` |

`cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` — `matching()` /
`matching_joint()`:

| knob | was | now | note |
|---|---|---|---|
| `lm` | `false` | `true` | LM tagger; production `-lm` |
| `main_flag` | *(absent)* | `true` | new arg → `flag_matched_mains`; production `MAINFLAG:-1` |
| `cathode_diag` | *(absent)* | `''` | new arg, diagnostic, off |
| `auto_mask` | *(absent)* | `null` | new tri-state arg, see §4 |
| `beam_pref`, `beam_pref_weight`, `beam_pref_rescue` | *(absent)* | `null` | new tri-state args for scan overrides |

The same values are also the TLA defaults of the promoted jobs (the TLAs
override the module, so leaving them at legacy values would have masked the
flip — the compiled-config proof caught exactly that).

**Unchanged, already production:** `mip_dqdx=56000`, `beam_window_only=true`,
`unmerge_bundle_mode='real'`, `stm_consistent_fv` and all seven doc-63 STM
guards (`stm_accept_guards`, `stm_proton_muon_guard`, `stm_cathode_guard`,
`stm_anode_dist_fix`, `stm_second_track_guard`, `stm_deficit_guard`,
`stm_vertex_kink_guard`) — every one of them `true`.

**Line held:** physics/tagger knobs take production values; pure
output/diagnostic knobs stay off (`save_stm_fit`, `save_tensors`, `calib_dump`,
`trace_bee`, `cathode_diag`, `dl_weights`, `tensor_outname`). Production
additionally passes `-stm-fit`, which appends `stm_magnify` to the pipeline and
writes the Magnify-tracking ROOT dump; the in-tree default leaves both off.

The PR job's `pipeline_names` default is now the production tagger chain
(`switch_scope, unmerge_bundle, unmerge_assoc, steiner, fiducialutils,
tagger_check_tgm, tagger_check_stm, tagger_check_fc`) instead of `[]`, so a bare
`wire-cell -c` run exercises the taggers. `[]` still means the pass-through
round-trip identity gate, which is what `run_pr_evt.sh` passes explicitly.

## 3. File inventory

New in `cfg/pgrapher/experiment/sbnd/`:

| file | note |
|---|---|
| `wct-img-all.jsonnet` | imaging; `full_deghost=true` TLA default (module default stays `false`) |
| `wct-clustering.jsonnet` | clustering only |
| `wct-clus-matching-perevt.jsonnet` | clustering + Q/L matching |
| `wct-pr-perevt.jsonnet` | PR / TGM+STM+FC taggers; `particle_dataset` import repathed to WIRECELL_PATH |
| `wct-reco1-dump.jsonnet` | reco1 art → interchange archives |
| `particle_dataset.jsonnet` | dQ/dx + range tables, provenance header intact (E = 0.5 kV/cm, retained 0.85 scale) |
| `sbnd_track_fitting.json` | TrackFitting parameters |

Modified in-tree: `clus.jsonnet` (defaults, §2), `qlmatching.jsonnet` (defaults
+ four promoted overlay args).

In `sbnd_xin/`: the five `wct-*.jsonnet` jobs and `qlmatching.jsonnet` became
one-line re-exports; `sbnd_track_fitting.json` became a symlink to the in-tree
copy. **No runner logic changed** — the runners keep passing explicit TLAs,
which is what makes the gate a pure compiled-config identity. Only two stale
comments were refreshed (`run_nusel_evt.sh`'s `pwd -P` rationale,
`run_ql_evt.sh`'s `-auto-mask` help, §4).

## 4. Two warts recorded rather than "fixed"

**`auto_mask` was never toggleable.** `match_data` sets `auto_mask: true`
unconditionally, so the old wrapper's `automask_on()` and the `-auto-mask` flag
only ever *re-asserted* a value that was already on — passing false never
disabled it. The new canonical arg is a real tri-state (`null` = inherit,
`false` = emit `auto_mask: false` = actually off), but the per-event job maps its
own `auto_mask=false` TLA to `null` so the runner contract is byte-identical.
Same shape for `beam_pref`. The help text now says so.

**`trackfitting_config` is not `WIRECELL_PATH`-resolved.** `TaggerCheckSTM`
loads it with a plain `std::ifstream` (`clus/src/TaggerCheckSTM.cxx`
`load_trackfitting_config`), so the in-tree `sbnd_track_fitting.json` can only be
reached by absolute path, and the job's `''` default silently falls back to the
uBooNE-derived built-in parameters. Documented at the top of the job.
**Follow-up (not done):** wrap the filename in `Persist::resolve()` in
`TaggerCheckSTM` and `TaggerCheckNeutrino` — byte-identical for absolute paths
(`resolve` returns those unchanged) and it would let the job default to the
canonical file by name. That is a C++ production-path change and needs its own
gate + doctest round, so it was deliberately kept out of this config sync.

## 5. Verification

- **Compiled-config identity.** Five live jobs × the TLA sets of their
  *production* runners (`run_nusel_evt.sh`, `run_ql_evt.sh`, `run_clus_evt.sh`,
  `run_img_evt.sh`, `run_reco1_dump.sh`), compiled from the pre-change tree, from
  the promoted in-tree copies, and from the `sbnd_xin` re-export shims:
  **10/10 byte-identical**. Because no binary changed and the compiled JSON is
  wire-cell's complete input, this is a stronger statement than a run comparison.
- **Second consumers of the same jobs had to be pinned.** A job is only
  byte-identical for a caller that passes the flipped knobs explicitly. Two
  callers do not, and both were found by enumerating every invocation rather than
  trusting the production path:
  - `run_pr_evt.sh` (per-event PR debug/A-B runner) passes only a subset of the
    PR job's TLAs — no `tgm_*` at all — so its `-tgm` / `-nu` / `-dnn` demos
    would have switched to merge-aware TGM plus the wider FV margins. Its
    `beam_window_us` *is* explicit (`[0,0]`), so the beam window stayed inert.
    The twelve `tgm_*` knobs are now pinned to the pre-adoption values in the
    runner, with a comment saying to delete the block to follow production.
    Verified identical against pristine HEAD.
  - `wct-clus-matching-standalone.jsonnet`, below.
  **Open for the owner:** both pins freeze a *debug* path at legacy while
  production moved. Deleting either block makes that path track production — a
  one-line change, but an unvalidated behavior change, so it was not taken.
- **Legacy job unchanged.** `wct-clus-matching-standalone.jsonnet` (the legacy
  all-events variant, still driven by `run_clust_QL_evt.sh`) imports the
  wrapper by bare name and would have inherited the new `lm`/`main_flag`
  defaults. It is pinned to `lm=false, main_flag=false` and verified identical
  against a baseline compiled from **pristine git HEAD of both repos**
  (`git archive HEAD | tar -x` into a scratch tree, `WIRECELL_PATH` pointed at
  it). `ql_dump_scalar.jsonnet` is self-declared throwaway with no runner and
  was left alone.
- **Compiled-config proof of the new defaults.** With no TLAs at all, the PR job
  compiles to `check_neutrino_candidate/require_chord_charge/component_extremes/
  component_rescue/rescue_chord_check/main_component_pairs = true`,
  `chord_charge_mode='path'`, `main_component_mode='real'`,
  `fv_tolerance=[-25,-25,-30,-30,-50,-30]`, `beam_window_low/high = 200/2200`,
  `beam_window_only=true`, all seven STM guards `true`, `mip_dqdx=56000`; the
  matching job to `lm_tagger=true`, `flag_matched_mains=true`.
- **End-to-end same-code A/B.** Six decisive events (62613, 62303, 321371,
  402330, 353487, 72586) through the full PR tail into a fresh root
  `work-stmcamp-d64smoke` with the production flag set: 6/6 rc=0, and every
  `nusel-evt<ID>.tsv` **identical** to the validated round-5 arm
  `work-stmcamp-r5fullc`. Scored against the doc-62 owner baseline: 4 correct,
  0 still-wrong, 0 regressed, no collateral flips.
- `./build/clus/wcdoctest-clus`: 49 cases / 565 assertions, rc=0.

## 6. Excluded and deferred

- **All `wcls-*` LArSoft work — deferred (owner, 2026-07-27).**
  `wcp-porting-img/sbnd/wcls-img-clus-matching-xin.jsonnet` calls
  `clus(..., run_labeler=...)`, and canonical `clus()` on `apply-pointcloud` has
  no such parameter — it would not compile. `run_labeler` / `truth_labeler` /
  `use_sce` exist only on **`origin/tgm`** (HaiwangYu: `4b5408de`, `fc725327`,
  `040d32bf`); that branch has 8 sbnd-cfg commits we lack, we have 42 it lacks.
  Reconciling is a cross-branch merge of another developer's line, not a copy.
  Second finding: the `img_config` strings used there
  (`'active2view+masked2view'`, `'active3view+masked1view'`, set by the fcls) no
  longer match any named branch of the current `img.jsonnet`
  (`single|active|masked|multi-2view|multi-3view`) and fall through to its
  generic `else`, i.e. plain single slicing on the active fork — not the 2-view
  recovering slicing the names imply. Both need HaiwangYu's input.
- `wct-sp-to-magnify.jsonnet` — magnify viewing, not requested; and its local
  `magnify-sinks.jsonnet` has a different signature
  (`function(tools, outputfile_prefix, runinfo=null)`) from the canonical file
  that five in-tree jobs use, so promoting it under that name would break them.
- `ql_dump_scalar.jsonnet` — self-declared throwaway, zero runner references.
- `wct-clus-matching-standalone.jsonnet` — legacy `extVar` variant superseded by
  the per-event job (`docs/3_scripts.md`); kept working and pinned, not promoted.
- `sbndcode/.../cfg/pgrapher/experiment/sbnd/` — the downstream consumer already
  differs in `img.jsonnet` / `params.jsonnet` / `sp.jsonnet`; syncing that is a
  separate upstream axis (cf. `3ac96548`).

## 7. Commits

- toolkit `cfg/sbnd: promote the standalone reco jobs + PR materials in-tree`
- toolkit `cfg/sbnd: PR + matching defaults to the SBND production operating point`
- wcp `sbnd_xin: re-export the canonical in-tree reco jobs (doc 64)`

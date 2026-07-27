# 64 — Sync the SBND working config into the in-tree `cfg/pgrapher/experiment/sbnd/`

**Status: DONE 2026-07-27.** Every live SBND reconstruction job and every
pattern-recognition data file now lives in-tree; `sbnd_xin` re-exports them.
All compiled configs byte-identical, end-to-end run identical.
The TrackFitting parameter file is now WIRECELL_PATH-resolved (§4a), so the
in-tree config is self-sufficient. `lifetime` is documented as inert (§4).
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

# 5. sec 4a: the WIRECELL_PATH resolution of the TrackFitting JSON.
#    absolute path (what the runners pass) must be unchanged:
NJOBS=24 ./stm_campaign/run_round.sh d64tf        # all 72 baseline events
#    ... then every nusel TSV must equal work-stmcamp-r5fullc's.
#    relative name must give the same answer as the absolute one: run one event
#    twice with trackfitting_config=<abs> and =pgrapher/experiment/sbnd/
#    sbnd_track_fitting.json, then compare member content hashes:
python3 ../../abtest/hash_archive.py <dirA>/mabc-pr.zip <dirB>/mabc-pr.zip
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
copy. The *production* runners keep passing explicit TLAs unchanged, which is
what makes the gate a pure compiled-config identity; the only runner logic change
is the twelve `tgm_*` pins added to `run_pr_evt.sh` (§5). Two stale comments were
refreshed (`run_nusel_evt.sh`'s `pwd -P` rationale, `run_ql_evt.sh`'s
`-auto-mask` help, §4).

## 4. Warts: one fixed, one recorded

**`auto_mask` was never toggleable.** `match_data` sets `auto_mask: true`
unconditionally, so the old wrapper's `automask_on()` and the `-auto-mask` flag
only ever *re-asserted* a value that was already on — passing false never
disabled it. The new canonical arg is a real tri-state (`null` = inherit,
`false` = emit `auto_mask: false` = actually off), but the per-event job maps its
own `auto_mask=false` TLA to `null` so the runner contract is byte-identical.
Same shape for `beam_pref`. The help text now says so.

**`trackfitting_config` is now `WIRECELL_PATH`-resolved (§4a).** It was not, and
that was the one remaining hole in "the in-tree config is self-sufficient" — see
§4a for the fix.

**`lifetime` was an inherited placeholder; now 35 ms, matching the simulation.**
The PR and Q/L jobs take a `lifetime` TLA and override `params.lar.lifetime` with
it, but **no reconstruction component reads it**: `lifetime` occurs **zero** times
in the compiled imaging, clustering, Q/L and PR configs. It only feeds the sim
`Drifter`'s charge attenuation in the `wcls-sim-*` jobs. So **no electron-lifetime
/ charge-attenuation correction is applied anywhere** in imaging, clustering,
matching or PR.

It used to say 6.0 ms, which arrived as part of the triple
`DL=6.2 DT=9.8 lifetime=6 driftSpeed=1.565` in HaiwangYu's first standalone Q/L
test (`wcp-porting-img 655bd6a`, 2026-05-24) and was copied forward into
`wct-clus-matching-perevt.jsonnet` (`ba805c6`) and the PR job.
`sbnd_track_fitting.json`'s own `_comment_diffusion` calls that same set "the
earlier 6.2/9.8 placeholders inherited from the Q/L chain and uBooNE". The DL/DT
half was later corrected to the SBND physical values (`9f498089`); `lifetime`
never was, because nothing consumes it. It was never an SBND measurement, and it
disagreed with SBND's own `simparams.jsonnet` (35 ms) and with `run_clus_evt.sh`,
which already passed 35.

**Changed to 35 ms (owner, 2026-07-27)** so the reco chain and the simulation
state the same number: the `lifetime` TLA default in both jobs, and the `LIFETIME`
variable in `run_nusel_evt.sh`, `run_ql_evt.sh`, `run_pr_evt.sh` and
`run_clust_QL_evt.sh` (the runners pass it explicitly, so leaving them at 6 would
have made the config default cosmetic). `run_clus_evt.sh` and
`wct-clustering.jsonnet` were already 35.

Gates: compiling the PR and Q/L jobs with `lifetime=6` vs `lifetime=35` gives
**byte-identical JSON** — the direct proof of inertness — and the five jobs
compiled with the *new* runner TLA sets are still byte-identical to the original
pre-change baseline. Runtime: 6 decisive events through the full PR tail with
`LIFETIME=35`, `nusel` TSVs identical to `work-stmcamp-r5fullc` (6/6).

This is a documentation/consistency fix, **not** a lifetime correction. Adding a
real one would be a physics change: 35 ms is a *simulation* value, data needs a
measured lifetime, and the whole dQ/dx chain (including the retained 0.85 scale
factor in `particle_dataset.jsonnet`) would have to be revisited with it.

## 4a. `trackfitting_config` resolved through WIRECELL_PATH

`TaggerCheckSTM` and `TaggerCheckNeutrino` loaded the TrackFitting parameter JSON
with a plain `std::ifstream`, so an in-tree copy was reachable only by absolute
path and the `''` default silently fell back to the uBooNE-hard-coded C++
presets — wrong for SBND, and invisible.

Fix (owner-requested, 2026-07-27): `load_trackfitting_config()` in both taggers
now resolves the filename with `Persist::resolve()` first:

```cpp
const std::string resolved = Persist::resolve(config_file);
std::ifstream file(resolved.empty() ? config_file : resolved);
```

`resolve()` returns an absolute path unchanged, so every existing caller is
unaffected; a relative name is looked up on `WIRECELL_PATH`; and when nothing is
found the diagnostic still names what was asked for (now with "not found on
WIRECELL_PATH either"). `TaggerCheckSTM.cxx` gained the `WireCellUtil/Persist.h`
include (`TaggerCheckNeutrino.cxx` already had it).

With that, the defaults could stop lying: `clus_pr()`/`pr()`'s
`trackfitting_config_file` and the PR job's `trackfitting_config` TLA now default
to `pgrapher/experiment/sbnd/sbnd_track_fitting.json`. A bare
`wire-cell -c pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` therefore runs with
the SBND parameters; `''` still selects the uBooNE presets.

Gates for this change (all after `wcbuild` + freshness proof — `libWireCellClus.so`
mtime newer than both edited sources):

- `wcdoctest-clus`: 49 cases / 565 assertions, rc=0.
- Compiled-config identity: the same 10/10 byte-identical result as §5 — the
  production runners pass an absolute path, so nothing moved.
- **Runtime, absolute path:** the full 72-event doc-62 baseline round
  (`stm_campaign/run_round.sh d64tf`, NJOBS=24, production flag set) — 72/72
  rc=0 and **72/72 `nusel-evt<ID>.tsv` identical** to the validated round-5 arm
  `work-stmcamp-r5fullc`. So `Persist::resolve()` is a genuine no-op for the
  absolute paths production uses.
- **Runtime, relative path:** event 62613 run twice through the PR job, once with
  the absolute path and once with
  `trackfitting_config=pgrapher/experiment/sbnd/sbnd_track_fitting.json`.
  `mabc-pr.zip` **member-content hashes identical** (`hash_archive.py`, M2), and
  all 27 tagger config/verdict log lines identical — the only textual difference
  was one spdlog line torn at a different column, the known non-atomic-write
  artifact. No "Cannot open config file" in either log.

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
- `./build/clus/wcdoctest-clus`: 49 cases / 565 assertions, rc=0 (re-run after
  the §4a C++ change, with the M1 freshness proof).
- The §4a `Persist::resolve()` change adds its own gates: see §4a.

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

- toolkit `cfg/sbnd: promote the standalone reco chain in-tree + production
  defaults (doc 64)` — b2dab726.  (Planned as two commits, shipped as one: the
  promotion alone would have left `wct-clus-matching-perevt.jsonnet`
  non-compiling at the intermediate commit, since it passes `cathode_diag`, which
  only exists after the `qlmatching.jsonnet` change.)
- wcp `sbnd_xin: re-export the canonical in-tree reco jobs (doc 64)` — 99cc352.
- toolkit `clus,cfg/sbnd: resolve the TrackFitting config through WIRECELL_PATH`
  — §4a.
- wcp `sbnd_xin: doc 64 sec 4/4a` — the lifetime finding + the resolve gates.

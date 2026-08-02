# 68 — One source for the SBND production operating point: the in-tree config

**Status: DONE 2026-08-01.** The SBND production operating point now exists in
exactly one place — the TLA defaults of
`cfg/pgrapher/experiment/sbnd/wct-{pr,clus-matching}-perevt.jsonnet`. The runner
scripts hold **no** default value: they pass what is per-event plus explicit
overrides, so a bare invocation *is* production. `run_full1k_nusel.sh`'s
twelve-flag `NUF` string collapses to `-stm-fit`; the production PR invocation
drops from 45 TLAs to 13.

**No physics value changed.** This is relocation and de-duplication only. Four
intended differences — the `trackfitting_config` path string, bare-run
semantics, the `run_pr_evt.sh` pin deletion, and one deliberately surviving
duplicate (`BEAM_WINDOW_EFF`) — are each listed in §4 with their gate.
Everything else is byte-identical.

Predecessor: [64](64_cfg-sync.md), which promoted the jobs in-tree and adopted
the production values as the *module* defaults but deliberately left the
runners passing every TLA explicitly ("what makes the gate a pure
compiled-config identity").

## Repro block

```bash
D=/home/xqian/tmp/sbnd-cfg-organize          # scratch harness (not in the repo)
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 0. pristine baselines of both repos
mkdir -p $D/pristine/{toolkit,wcp}
git -C /nfs/data/1/xqian/toolkit-dev/toolkit archive HEAD cfg | tar -x -C $D/pristine/toolkit
git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img archive HEAD sbnd/sbnd_xin | tar -x -C $D/pristine/wcp

# 1. Gates A+B -- capture what each runner ACTUALLY passes, via a wire-cell
#    shim that records argv, then compile that argv with wcsonnet.
#    $D/bin/wire-cell = printf '%s\n' "$@" > "$WCARGS_OUT"; exit 0
$D/capture.sh $D/before $D/cases_before.txt      # pristine scripts + pristine cfg
$D/capture.sh $D/after  $D/cases_after.txt       # new scripts + new cfg
for f in $D/before/*.json; do cmp -s $f $D/after/$(basename $f) \
  && echo "IDENTICAL $(basename $f)" || echo "DIFFER $(basename $f)"; done
python3 $D/keydiff.py $D/before/<label>.json $D/after/<label>.json   # key-level diff

# 2. Gate B, second consumers
cd $SX
./compile_sbnd_prod.sh $D/pristine/toolkit/cfg $D/before/prod        # wcls + legacy standalone
./compile_sbnd_prod.sh $WCT/toolkit/cfg        $D/after/prod
$D/pristine/wcp/sbnd/sbnd_xin/compile_prjob_cfg.sh $D/pristine/toolkit/cfg $D/before/prjob.json
./compile_prjob_cfg.sh $WCT/toolkit/cfg $D/after/prjob.json

# 3. Gate C -- output identity
#    (a) the save_assoc flip must not move the Bee zip of a non-saving caller
for a in 0 1; do WR=$D/gc/assoc$a; rm -rf $WR; mkdir -p $WR
  ln -sfn $SX/work-mcp1000/evt284349 $WR/evt284349
  ( cd $SX && SBND_INPUT_DIR=$SX/input_files_reco1/staged-mcp2025c-1000evt/e0 \
      SBND_WORK_ROOT=$WR SBND_SAVE_ASSOC=$a ./run_ql_evt.sh data 1 ); done
python3 ../../abtest/hash_archive.py $D/gc/assoc{0,1}/ql_evt284349/mabc-all-apa.zip

# 4. Gate D -- end-to-end, 6 events, vs the current production baseline
ENTRIES="0 1 2 3 4 5" TAG=d68smoke ./run_full1k_nusel.sh 6 6
for e in 284349 284657 285185 285999 286021 286065; do
  cmp work-mcp1kall-isog1k/nusel_evt$e/nusel-evt$e.tsv \
      work-mcp1kall-d68smoke/nusel_evt$e/nusel-evt$e.tsv
  python3 ../../abtest/hash_archive.py \
      work-mcp1kall-isog1k/nusel_evt$e/mabc-pr.zip \
      work-mcp1kall-d68smoke/nusel_evt$e/mabc-pr.zip; done
```

No C++ was touched, so there is **no** rebuild, no `wcdoctest`, and no
abtest/qlport gate in this change. No file under `cfg/pgrapher/common/` was
touched either, so PDHD/PDVD/uBooNE cannot move.

## 1. What was still duplicated after doc 64

Doc 64 put the production values into the job configs but left every runner
passing them anyway, so the config copy was decorative — the runner copy won.
Two costs, both live:

**(a) The runner defaults were the *pre-adoption* configuration.** With no
flags, `run_nusel_evt.sh` ran chord/rescue/main-pair OFF, `lm` OFF, and
`fvx/fvy = 2 / 2.5` where the config says `2.5 / 3` — transposed, not merely
stale. Only the twelve-flag `NUF` string inside `run_full1k_nusel.sh` reached
production, so `./run_nusel_evt.sh data 1` ran a setup nobody uses and no
document describes.

**(b) A latent inconsistency in the config's own defaults.** The PR job's
default `pipeline_names` already listed `unmerge_assoc`, but the Q/L job's
`save_assoc` defaulted `false`, so the arrays that visitor reads did not exist.
A bare in-tree run had a pipeline stage that emitted a per-cluster WARNING and
silently did nothing. Production only escaped it because
`run_full1k_nusel.sh` exported `SBND_SAVE_ASSOC=1` by hand.

The measurement that licensed the whole change: compiling `wct-pr-perevt.jsonnet`
with the full production TLA set vs. with only event-specific paths gives
**byte-identical JSON** (252198 chars both). Every `tgm_*`, `stm_*`,
`mip_dqdx`, `beam_window*`, `unmerge_bundle_mode` and LAr TLA was already exactly
the config default.

## 2. Owner decisions (2026-08-01)

| # | question | decision |
|---|---|---|
| 1 | Do diagnostic outputs join the config default? | **No** — hold the doc-64 line. `save_stm_fit`, `save_tensors`, `calib_dump`, `trace_bee` stay OFF in cfg; `NUF` keeps `-stm-fit`. |
| 2 | Does a bare runner invocation mean production? | **Yes.** Runners hold no default values; unset ⇒ pass nothing ⇒ inherit cfg. |
| 3 | Which Q/L cfg defaults flip? | `joint`, `save_rcid`, `save_assoc` → `true` (all three). |
| 4 | `run_pr_evt.sh`'s twelve legacy `tgm_*` pins? | **Delete** — that debug path now tracks production. |

## 3. What changed

### 3a. Config (toolkit, `experiment/sbnd/` only)

`wct-clus-matching-perevt.jsonnet`, three TLA default flips:

| TLA | was | now | why |
|---|---|---|---|
| `joint` | `false` | `true` | the joint multi-APA QLMatching node is the production graph; `run_ql_evt.sh` always passed `joint=true`. Escape: `--per-apa` / `SBND_JOINT=0`. |
| `save_rcid` | `false` | `true` | the PR job's default `unmerge_bundle` runs in `"real"` mode, which reads exactly these arrays. Escape: `-no-save-rcid`. |
| `save_assoc` | `false` | `true` | closes §1(b): the default PR pipeline's `unmerge_assoc` needs them. Escape: `-no-save-assoc`. |

### 3b. Runners (wcp-porting-img, `sbnd/sbnd_xin/`)

The rule: **a runner holds no default value for anything cfg defaults.** Every
`FVX_MARGIN=2`, `${SBND_TGM_CHORD:-0}`, `DL=4.0`, `PIPELINE=…` line is deleted,
not re-pointed. The surviving shape is the tri-state already used by
`SBND_CATHODE_RESCUE` / `SBND_SEP_VVETO` / `SBND_NU_ISO_GUARD`:

```bash
knob_bool() { case "$2" in 1) KNOB_TLA+=(--tla-code "$1=true") ;;
                           0) KNOB_TLA+=(--tla-code "$1=false") ;; esac; }
```

Empty ⇒ no TLA ⇒ the config default stands. Every `-no-*` flag is an
explicit-false override; numeric knobs (`-fvz`, `-fvx`, `-mip`, `-bw`) pass a
TLA only when given.

| file | change |
|---|---|
| `run_nusel_evt.sh` | 32 TLAs deleted (LAr set, `bwonly`, `nucand`, chord/rescue/main-pair, 4 FV margins, `mip_dqdx`, 8 STM guards + d66 cuts, `unmerge_bundle_mode`, the `dl_weights=` pin, the `TFJSON` default). `pipeline_names` is now named **only** when a flag changes the default list. **45 → 13 TLAs** at production. |
| `run_ql_evt.sh` | 15 TLAs deleted (LAr set, `semimodel_file`, `joint`, `pmt_nl`, `main_flag`, `lm`, `save_rcid`, `save_assoc`, `trace_bee`, `auto_mask`, `beam_pref*`, `rcid_global`, `realign`). `anode_indices` is passed only under `-a`. **21 → 8 TLAs** at production. |
| `run_pr_evt.sh` | the twelve `tgm_*` pins deleted (decision 4) + the LAr set and `trackfitting_config`. **25 → 9 TLAs.** Also now honours `SBND_WORK_ROOT` (§6). |
| `run_pr_chain_batch.sh` | the same 28-TLA production block deleted; `PIPELINE` stays explicit (this chain adds the neutrino taggers + BDT scorers on top of the default list). |
| `run_full1k_nusel.sh` | `NUF` → `-stm-fit`; `export SBND_SAVE_ASSOC=1` deleted. |
| `run_perf54_nusel.sh` | same `NUF` collapse. |
| `compile_prjob_cfg.sh` | compiles the production PR chain nearly bare, matching the new invocation. |

Not touched, deliberately: `wct-clus-matching-standalone.jsonnet` and its
`lm=false, main_flag=false` pins (doc 64 §6 legacy job — verified unmoved,
§4 Gate B); the `wcls-*` LArSoft configs (doc 64 §6, blocked on HaiwangYu —
also verified unmoved); `tmp_run_pr_chain_*.sh` and `run_pr_geom_arm*.sh`
(single-doc A/B arm recipes that pin on purpose); anything under `work-*/`.

## 4. Intended differences, each with its gate

Everything else is byte-identical. These four are the whole delta:

**(i) `trackfitting_config` path string.** The runners passed the absolute
`$SX/sbnd_track_fitting.json` (a symlink); the config default is the
WIRECELL_PATH-resolved `pgrapher/experiment/sbnd/sbnd_track_fitting.json`. The
compiled JSON therefore differs in exactly one string:

```
< "trackfitting_config_file": "…/sbnd_xin/sbnd_track_fitting.json"
> "trackfitting_config_file": "pgrapher/experiment/sbnd/sbnd_track_fitting.json"
```

Same file (`cmp` clean — the symlink resolves to the in-tree copy) and
`Persist::resolve()` finds it either way (doc 64 §4a). **Gate C(b)**: 6 events
through the PR job, `mabc-pr.zip` member-content hashes **6/6 identical**, and
`grep -c 'Cannot open config file|not found on WIRECELL_PATH'` = 0 on all six
logs. `SBND_TRACKFIT_JSON` still overrides it for the doc-66 diffusion A/B.

**(ii) Bare/`-no-*` runner invocations now inherit production.** Intended, and
the point of decision 2. The compiled diff for `nusel_bare` is exactly: FV
tolerances `-20/-25/-30 → -25/-30/-50`, `interior_fv_tolerance` appearing,
`require_chord_charge`/`component_extremes`/`component_rescue`/
`rescue_chord_check`/`main_component_pairs = true`, `main_component_mode=real`,
`chord_charge_mode=path`, and `unmerge_assoc` entering the pipeline. For the
Q/L job: `lm_tagger=true` and the three `save_assoc*` keys. **No physics knob
moved to a value that is not the documented production value.**

**(iii) `run_pr_evt.sh` tracks production.** Its `-tgm`/`-nu`/`-dnn` demos now
show merge-aware TGM plus the doc-39 FV margins instead of the pre-adoption
configuration. The full diff is the 25 keys listed by
`keydiff.py before/pr_tgm.json after/pr_tgm.json` — the twelve deleted pins and
the FV tolerances they implied, plus (i). Nothing else.

**(iv) One production value survives in a runner, on purpose.**
`run_nusel_evt.sh` keeps `BEAM_WINDOW_EFF="${BEAM_WINDOW:-0.2,2.2}"`. It feeds
`nusel_extract.py --beam-window`, and that script is a post-processor, not a
wire-cell node — it cannot read the job's `beam_window_us`. So the headline
"runners hold no default value" has exactly this one exception, and it is
commented as such in the script. **Keep it in step with `beam_window_us` in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`.** The wire-cell side is
still clean: the `beam_window_us` TLA is emitted only under `-bw`.

**Consequence to be aware of:** every historical A/B arm recipe in docs 26-66
that relied on the *old* runner defaults must now name its flags explicitly to
reproduce. Those docs are not retro-edited; the `-no-*` flags exist precisely so
any old arm remains expressible.

## 5. Verification

**Gate A — production identity.** Each production runner's *actual* argv,
captured through a `wire-cell` shim (so the gate tests what the script passes,
not a transcription of it) and compiled with `wcsonnet`, pristine `git HEAD`
of both repos vs. the working tree:

| caller | result |
|---|---|
| `run_ql_evt.sh` production (`-save-pctree -calib`, was `-save-pctree -lm -calib -save-rcid`) | **byte-identical** (54463 chars both) |

The Q/L row is an *equivalence*, not a tautology: the before-arm named `lm` and
`save_rcid` explicitly, the after-arm names neither and reaches the same values
through the §3a defaults. Identical output is what proves the flip and the strip
cancel exactly.

| `run_nusel_evt.sh` production (`-stm-fit`, was the 12-flag `NUF`) | identical except (i) — `diff` is one line |
| `compile_prjob_cfg.sh` (PR chain + neutrino taggers + BDTs) | identical except (i), 2 keys |

**Gate B — per-caller sweep.** Every invocation whose *default* moves, plus the
second consumers doc 64 had to retrofit:

| caller | result |
|---|---|
| `wcls-img-clus.jsonnet` (LArSoft production imaging+clustering) | **IDENTICAL**, and structurally insulated: zero references to `joint`/`save_rcid`/`save_assoc` |
| `wct-clus-matching-standalone.jsonnet` (legacy pinned job) | **IDENTICAL** — but read the insulation note below before trusting that number |
| `sbnd/wct-clus.jsonnet` | insulated: zero references to the flipped job or its three TLAs |
| `sbnd/wcls-img-clus-matching-xin.jsonnet` | **does not compile on this branch** (`RUNTIME ERROR: Field does not exist: pc_transforms`) — pre-existing, exactly as doc 64 §6 recorded; HaiwangYu's `origin/tgm` file, not caused by doc 68 |

**Insulation note on the standalone job.** `compile_sbnd_prod.sh` compiles it
with `--ext-code joint=false`, so on its own that IDENTICAL would only prove
"a caller that pins `joint` is unaffected by the `joint` default flip" — true by
construction and worth nothing. The result is real for a different reason:
`wct-clus-matching-standalone.jsonnet` has **zero** references to
`wct-clus-matching-perevt.jsonnet` (it is a separate job with its own
signature), its `joint` is its own required `std.extVar('joint')`, its `lm` and
`main_flag` are its own literal `false` pins into `qlmatching.jsonnet`, and it
never passes `save_rcid`/`save_assoc` at all — those stay at the `clus.jsonnet`
module defaults, which doc 68 did **not** touch (`clus_all_apa(...,
save_real_cluster_id=false, ..., save_assoc_cluster_id=false, ...)` is
unchanged). Only the *job's* TLA defaults moved, never the module's.
| `ql_bare`, `ql_nomainflag`, `ql_isooff`, `ql_tracebee` | differ by `lm_tagger` + 3 `save_assoc*` keys only |
| `ql_perapa` (`--per-apa`) | same, and still builds `matching0`/`matching1` — the per-APA graph survives the `joint` flip |
| `nusel_bare`, `nusel_nobwonly`, `nusel_nostmguards`, `nusel_nounmerge` | differ by (ii) only |
| `-no-bwonly` / `-no-stm-guards` overrides | still key-suppressed OFF, identical before and after |
| `-no-unmerge` | pipeline correctly loses **both** un-merge visitors |
| `run_pr_evt.sh -tgm` | (iii) |

**Gate C — output identity.**
- `save_assoc` flip, bare Q/L run on evt 284349: `mabc-all-apa.zip`
  member-content hash `3668925345b68ff9234a230d0fab1f5c4f8674fa648829742cb570afaffcab1a`
  with the arrays off **and** on. So the plain `true` default was shipped rather
  than the conditional `save_tensors != ''` fallback — the flip changes only the
  pctree tarball, for callers that save one.
- `trackfitting_config`: see (i).

**Gate D — end-to-end.** 6 events (entries 0-5) through the full chain with the
**new** minimal flag set, fresh root `work-mcp1kall-d68smoke`: 6/6 rc=0, and
every `nusel-evt<ID>.tsv` **byte-identical** to the current production arm
`work-mcp1kall-isog1k` (7/18/12/15/10/9 rows). Every event reports
`assoc=<mains>/<parts>` non-zero (4/18, 12/54, 10/39, 7/22, 7/12, 7/23) — the
isolated-grouping un-merge engages from the config default alone, with no
`SBND_SAVE_ASSOC` in the environment. That is §1(b) closed, measured.

## 6. `run_pr_evt.sh` now honours `SBND_WORK_ROOT`

Found while building the §5 harness and fixed on request: `run_pr_evt.sh` was
the only runner in this directory that hard-coded its work root as
`$SBND_DIR/work` (four places: `QLDIR`, `PRDIR`, the `mkdir -p`, and the batch
log path) instead of honouring `SBND_WORK_ROOT`. That is exactly why it could
not be driven by the capture harness and had to be verified by a direct
`wcsonnet` comparison instead.

The fix is a four-line substitution to `$SBND_WORK_ROOT`. It is **provably
inert when the variable is unset**, which is how the script has always been
run: `_runlib.sh` (sourced at line 22, after `SBND_DIR` is set at line 18) does

```bash
SBND_WORK_ROOT="${SBND_WORK_ROOT:-$SBND_DIR/work}"
```

so with nothing exported the two expressions resolve to the same absolute path
(verified: both `…/sbnd_xin/work`). With the variable set, the script now lands
in the requested root like `run_ql_evt.sh` / `run_nusel_evt.sh` do.

Gate: the §5 capture harness can now drive it, which is the point.
`run_pr_evt.sh` under `SBND_WORK_ROOT=$D/wr/<label>` runs to a clean argv
capture for all four demo modes — `pr_bare` (empty round-trip pipeline),
`-tgm`, `-stm`, `-nu -no-dnn` — 4/4 rc=0, each compiling without error. Its
invocation is down to **9 TLAs**: `input`, `anode_indices`, `output_dir`,
`run`/`subrun`/`event`, `reality`, `pipeline_names`, `save_tensors`,
`dl_weights`, `beam_window_us`, `save_stm_fit`. The twelve `tgm_*` pins are
gone and the operating point comes from the config, as §4(iii) intends.

## 7. Commits

- toolkit `cfg/sbnd: adopt the production Q/L operating point as job defaults (doc 68)`
- wcp `sbnd_xin: inherit the SBND operating point from cfg (doc 68)`

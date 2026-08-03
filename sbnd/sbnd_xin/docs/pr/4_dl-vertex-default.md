# 4 — The SCN (DL) neutrino vertex becomes the SBND default

Status: ADOPTED 2026-07-30 (owner call on one event, §2).  Config-only change —
no C++ touched.  Default SBND job compiled JSON byte-identical (§4).  Evidence
base is **one event**; expansion is §6.

Companion docs: `pr/3_pr-skip-cosmic-and-outputs.md` (the PR outputs this runs
inside), `pr/2_uboone-chain-gap-analysis-and-validation-plan.md` (gap **G3**,
SCN retraining, which this does **not** close).

## 0. Repro block

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX/work-nuecc48-prsmoke2

# geometric vertex (the pre-adoption arm; explicit dl_weights=''):
./run_pr3_evt.sh 172230                       # -> nupr_evt172230/

# DL vertex, explicit weights + libpython preload:
./run_pr3_evt_dl.sh 172230                    # -> nupr_evt172230_dl/
DL_SUFFIX=_dl_rep ./run_pr3_evt_dl.sh 172230  # determinism repeat

# DL vertex INHERITED from the new config default (no dl_weights TLA at all):
#   same command as run_pr3_evt.sh with the `--tla-str "dl_weights="` line
#   deleted and LD_PRELOAD set -> nupr_evt172230_defaultdl/

# what to check in every DL run:
grep -c "DL vertex failed" <log>        # expect 0
grep "After improve vertex" <log>
python3 ../../abtest/hash_archive.py <dir>/mabc-pr.zip
```

## 1. What changed

Four places, all defaults, no C++:

| file | before | after |
|---|---|---|
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` `clus_pr(...)` | `dl_weights=''` | `dl_weights='uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth'` |
| same, `pr(...)` wrapper | `dl_weights=''` | same path |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` TLA | `dl_weights = ''` | same path |
| `sbnd_xin/run_pr_evt.sh` | `DL_WEIGHTS=""` | default path, `-no-dnn` / `SBND_DL_VTX=0` for the geometric arm |

Plus, at the SBND `cm.tagger_check_neutrino` call site, the four DL re-rank
sub-knobs are now **pinned explicitly** — `dl_vtx_rerank=true`,
`dl_vtx_top_k=5`, `dl_vtx_min_accept_score=4.0`, `dl_vtx_score_scale=1000.0`.
They were inert while `dl_weights` was `''` and went live with this adoption, so
SBND records the operating point it was validated at instead of inheriting it
from `cfg/pgrapher/common/clus.jsonnet`, where a future edit would move SBND
silently.  The values equal the current common defaults, so the compiled JSON is
unchanged by the pinning (§4).

## 2. Why — the evidence

nueCC48 evt **18253/1/172230**.  The geometric (topological) vertex finder put
the neutrino vertex at the **far end of a proton track**, not at the interaction
point.  With the DL vertex enabled:

| | x (cm) | y (cm) | z (cm) |
|---|---|---|---|
| geometric | −46.08 | −84.15 | 22.92 |
| DL (uBooNE weights) | −54.71 | −87.47 | 19.91 |

moved **9.73 cm**.  Owner hand-scanned both Bee sets and judged the DL vertex
correct:

- geometric: <https://www.phy.bnl.gov/twister/bee/set/0bfe4744-30bf-44cd-945d-b8e13470aaa7/event/list/>
- DL: <https://www.phy.bnl.gov/twister/bee/set/649c6f93-dc93-41aa-a732-d2195d892e31/event/list/>

Mechanism: `determine_overall_main_vertex_DL` (`clus/src/NeutrinoVertexFinder.cxx:3217`)
runs only when `dl_weights` is non-empty; otherwise the traditional
`determine_overall_main_vertex` (`:3659`) decides and `improve_vertex` refines.
The topological chain has no charge-density prior strong enough to separate a
short proton's stopping end from the interaction point — the same failure the
DL vertex was introduced to fix in uBooNE, where it is the production default.

**The weights are uBooNE-trained.**  This is defensible because the SCN input is
translation-invariant: `SCN_Vertex.py`'s `voxelize()` subtracts the point-cloud
minimum, so only relative geometry and the charge scale reach the net.  It is
*not* a claim that the net is tuned for SBND — `pr/2` gap **G3 (SCN retraining)
stays open** and this adoption does not close it.

## 3. The silent-fallback trap (read this before any batch run)

The first DL attempt returned **rc=0 with the geometric vertex**.  The only
evidence was one line:

```
W [clus.NeutrinoPattern] determine_overall_main_vertex_DL: DL vertex failed:
  SCN_Vertex: import failed for module: SCN_Vertex: .../_ctypes...so: undefined symbol: PyTuple_Type
```

The embedded interpreter needs libpython loaded with global symbol visibility.
This job loads no ROOT (the uBooNE qlport job pulls libpython in through
WireCellRoot), so the preload must be explicit:

```bash
export LD_PRELOAD=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
```

`run_pr_evt.sh` and `run_pr3_evt_dl.sh` set it; `run_pr_evt.sh` additionally
greps its own log afterwards and prints a loud warning if the fallback fired.
Both are gated on `tagger_check_neutrino` actually being in the pipeline, so the
`-stm` / `-tgm` / bare `-p` arms keep the exact process environment (no
`LD_PRELOAD`) they had before this flip — they are A/B comparison arms.
**Any new batch driver must do the same** — `grep -c "DL vertex failed"` must be
0 across the whole manifest.  On a 45- or 503-event expansion a single WARN line
per event is invisible in practice, and the result is a run that looks
successful but carries different physics.

Recommendation left to the owner (not done here, it is a behavior change outside
this ask): make `TaggerCheckNeutrino` **throw** instead of falling back when
`dl_weights` is non-empty and the import fails.  Silent degradation to a
different vertex is worse than a crash.

Known remaining landmine: `run_nusel_evt.sh:592` passes `--tla-str "dl_weights="`
explicitly.  It is **inert today** (that driver's `pipeline_names` has no
`tagger_check_neutrino`), so it was left untouched, but the day the production
driver picks up the neutrino stage it would silently get the geometric vertex.

## 4. Verification

**Compiled-config proof, both directions** (HEAD worktree vs working tree,
`wcsonnet`, same TLAs):

- **Default job** (production `pipeline_names`, no `tagger_check_neutrino`):
  `cmp` rc=0, 251119 bytes both sides — **byte-identical**.  The
  `tagger_check_neutrino` entry lives in `cm_by_name` and is dropped from the
  compiled JSON when not named, so the flip cannot reach any current SBND
  production or gate output.
- **PR job** (13-stage `pipeline_names`): exactly **one** line differs —
  `"dl_weights" : ""` → `"dl_weights" : "uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth"`.
  The four `dl_vtx_*` keys are byte-identical, confirming the pinning is a no-op
  on output.

**Determinism** — three runs, `hash_archive.py` on `mabc-pr.zip` members:

| run | member-content hash |
|---|---|
| `nupr_evt172230_dl` (explicit weights) | `5b4e8158c9a19c6f…` |
| `nupr_evt172230_dl_rep` (repeat) | `5b4e8158c9a19c6f…` |
| `nupr_evt172230_defaultdl` (**no** `dl_weights` TLA — inherits the new default) | `5b4e8158c9a19c6f…` |
| `nupr_evt172230` (geometric) | `c5bfe4bfaa69a854…` |

Three things at once: the DL run repeats bit-identically (**N=1 event, N=2
runs** — CLAUDE.md M4's "not bit-stable" is *untested here, not disproven*); the
new default reproduces the explicitly-configured DL run exactly, so the config
flip does what it says; and the DL and geometric arms genuinely differ.

**M4 still stands**: identity gates keep passing `dl_weights=''` explicitly
(`qlport/scripts/run_one.sh:55`, `abtest/compile_all_cfg.sh:45`,
`sbnd_xin/scripts/perf/profile_pr65.sh:24`).  The DL vertex is never a gate arm.

**Cost**: `TaggerCheckNeutrino timing: overall main vertex` goes
**0.105 ms → ~980 ms** per event (990.0 / 975.9 ms in the two DL runs, CPU
torch, `torch.set_num_threads(1)`).  ~1 s/event: negligible for the 45-event
nu-candidate expansion, ~8 min added for a 503-event Track B pass.

**Coverage of callers** (`grep -rn dl_weights --include=*.sh` and
`grep -rn "clus_maker\.pr(" --include=*.jsonnet` across the toolkit `cfg/` and
`wcp-porting-img/{sbnd,qlport,pdhd,pdvd}`): the four sites in §1, the three
gate/profile scripts above, and `run_pr3_evt.sh`, which keeps its explicit
`dl_weights=` so it still reproduces `pr/3` §6 verbatim as the geometric arm.
One out-of-tree caller exists — `sbnd/wcls-img-clus-matching-xin.jsonnet:158`
calls `clus_maker.pr(...)` and so inherits the new default — but its
`pipeline_names` stops at `tagger_check_fc`, so the `tagger_check_neutrino`
entry is dropped from its compiled JSON exactly as in the default job: unaffected.
`sbnd_xin/wct-pr-perevt.jsonnet` is a one-line re-export of the in-tree module,
so runners pick the new default up automatically.  No production or gate driver
is affected.

### 4b. Interaction with `nu_skip_cosmic` (pr/3) — checked, no regression

The two paths differ in a way that matters: `determine_overall_main_vertex`
takes `map_cluster_main_vertices` and `main_cluster` **by value**, so its
internal `check_switch_main_cluster` cannot escape; `determine_overall_main_vertex_DL`
(`NeutrinoVertexFinder.cxx:3217`) takes both **by reference** and can re-select
the main cluster (`:3560` "rerank selected cluster="). So enabling DL enables a
caller-visible main-cluster switch that the arm `nu_skip_cosmic` was validated
against could not perform — the question is whether the rerank can reinstate a
main the skip gate just rejected.

It cannot, structurally: the DL candidate set is built from the PR graph
(`ordered_nodes(graph)`), and the graph covers only `main_cluster` plus
`other_clusters`, which `TaggerCheckNeutrino.cxx:207-213` fills exclusively from
clusters carrying `Flags::associated_cluster` and the selected main's
`matched_flash_gid`.  A skipped cosmic-tagged **main** never enters either.

Confirmed on the event that validated the skip — evt **444187** rerun under the
new default:

```
TaggerCheckNeutrino: in-window cluster 6 (t0 1.096 us, L 210.6 cm) cosmic-tagged (TGM=true STM=false lm_flag=0); skipping (nu_skip_cosmic)
TaggerCheckNeutrino: selected main cluster 19 (t0 1.573 us, L 170.5 cm, 6 associated)
```

and `mabc-pr.zip` is **bit-identical** to the geometric arm
(`0aeaf41308373c17…` both) — for this event the DL vertex agrees with the
traditional one, so the flip is a no-op here.  A useful second data point: DL
does not perturb every event.

## 5. Scope of this adoption — say it plainly

Adopted on the strength of **one event**, hand-scanned by the owner, with
**uBooNE-trained weights**.  This is not "the DL vertex is validated on SBND".
What would make it that:

- the 45 in-beam nu-candidate nueCC48 events (`pr/3` §7) run both arms, vertex
  displacement distribution + hand-scan verdicts;
- a check that the DL vertex never *degrades* a case the geometric chain got
  right (the failure mode that matters for a default);
- `pr/2` gap **G3**: retrain SCN on SBND, which also removes the uBooNE-geometry
  caveat from every downstream number (kinematics, BDT inputs).

Until then, every PR result carries the caveat that its vertex comes from a net
trained on another detector.

## 6. Next

1. Both-arms run over the 45 nu-candidates; tabulate displacement and
   disagreement cases (this is also `pr/3` §7's expansion item).
2. Decide the throw-vs-fallback question in §3.
3. G3 retraining.

# doc pr/25 — cathode re-join direction-agreement fallback (ClusteringProtectBundle) + TGM demoted-main veto (TaggerCheckTGM)

Also documents a third finding from the same Bee hand-scan round (321107),
investigated alongside these two; see the status table below.

| evt | question | status |
|---|---|---|
| **489327** | cathode crosser broken in two; the pr/23 re-join knob did not fire | **root cause proven, fix implemented + validated on a small group** (§1). **Committed** (toolkit `75c703da`, wcp-porting-img `c8d4e32`); SBND default left OFF pending owner go-ahead. |
| **320029** | another cluster (30) looks like a TGM but was not tagged | root cause proven, fix implemented + small-group measured (§2): a structural TGM/demoted-main veto interaction, not a tuning question. The measurement surfaced a second, pre-existing `check_tgm` gap — see §2. **SBND DEFAULT ON** (owner 2026-08-02, impact judged small); the `check_tgm` gap remains an open, separate item. |
| **321107** | main track is a muon, tagged as an electron | mechanism fully traced AND directly instrumented (§3): `is_shower_topology`, not `is_shower_trajectory`, made the call; isochrony is population-level irrelevant; the one remaining candidate threshold shows no natural separation across its 21-event blast radius and at least one segment flips verdict between two evaluations of the same fit. **No safe SBND-scoped fix found this round** — three candidate knobs were measured directly and rejected. Diagnostic instrument (`WCT_SHOWER_TOPO_DEBUG`) **committed, default off**. |

Repro base for all three: `sbnd_xin/work-vfmcp1k-prodon` (protect_bundle ON,
the SBND production default since `f813e312`) and `work-vfmcp1k-prodoff`,
572-event valfast subset, HEAD `f813e312` on 2026-08-02.

---

## 1. Event 489327 — SHIPPED (validated, not yet committed)

### Symptom

A cathode-crossing track is broken into two pieces by the PR-stage
`ClusteringProtectBundle` (doc pr/23) split, and the pr/23 cathode re-join
pass — designed for exactly this — refuses to reunite them. `nu_sel_len_cm`
395.2 → 349.0 cm, `nue_score` **−4.30 → +2.52** (a sign flip), `numu_score`
3.76 → 2.20.

### Root cause

The two halves arrive at the PR job as **one** cluster (confirmed on
`work-vfmcp1k-prodoff`: ident 19, 727 blobs, 5354 pts, x −145.0 → +11.8 cm,
straddling the cathode). `ClusteringProtectBundle`'s relaxed-graph split then
separates them into a retained cluster (x → −0.3 cm) and a 380-pt, 33.2-cm
fragment (x +1.8 → +11.8 cm) — `wct_pr_evt489327.log`:

```
<ClusteringProtectBundle:pr> cluster 19 (main): 727 blobs -> retained 537 + 4 fragment(s) holding 190
<ClusteringProtectBundle:pr> split 1 bundle cluster(s) into 4 extra cluster(s) (0 cathode re-join(s), 1 convicted main(s) skipped, graph 'relaxed')
```

`cathode_rejoin()` (`clus/src/ClusteringProtectBundle.cxx:231-283` pre-fix)
tests the closest 3D pair between the two components against four cuts (SBND
operating point: `cathode_x=0`, `rejoin_xcut=5cm`, `rejoin_dyz=4cm`,
`rejoin_dis=8cm`). Recomputed on the pair (19, 87):

| cut | value | verdict |
|---|---|---|
| `dis < 8cm` | 6.83 cm | PASS |
| `\|x1\| < 5cm` | 0.27 cm | PASS |
| `\|x2\| < 5cm` | 1.83 cm | PASS |
| `dyz < 4cm` | **6.50 cm** | **FAIL — the only one** |

The track is steep: local direction is **74–75°** from the drift axis on
both sides. Crossing the ~2.1 cm x-gap at that angle necessarily moves
2.1·tan(74°) ≈ 7.3 cm transversely — dyz is real track travel, not
misalignment. The halves are in fact collinear to **1.5°** (local PCA,
perpendicular offset 1.9 cm).

### Why it hid

`cathode_rejoin_dyz` is a frame-aligned bound on transverse offset — correct
for the ~1.1 cm median offset of a near-drift-parallel crosser (doc pr/12,
pr/14), and wrong by construction as a track tips toward the cathode plane.
No prior cathode round exercised a steep crosser through this exact code
path (distinct from the pr/23 §4.2 residual, evt 287654, which dies on
`dis`/`xcut`, not `dyz` — same underlying cause, different cut; this fix does
**not** rescue 287654).

### Population scan

`pr25_rejoin_census.py scan/direct` (committed, `sbnd_xin` commit `e370ded`),
572-event valfast prod arm: 191 events have a protect_bundle split, 610
(retained, fragment) pairs total. **0 pairs pass all four cuts as shipped; 6
fail ONLY dyz:**

| evt | pts | dis | dyz | dir-dir angle | perp | θ to drift |
|---|---|---|---|---|---|---|
| 399118 | 660 | 5.80 | 5.27 | 3.6° | 1.80 | 82.5° |
| **489327** | 380 | 6.83 | 6.50 | 1.5° | 1.90 | 75.0° |
| 289832 | 307 | 7.14 | 6.46 | 7.1° | 2.68 | 64.0° |
| 488139 | 268 | 5.78 | 5.10 | 3.0° | 1.95 | 70.4° |
| 492913 | 201 | 5.73 | 5.54 | 4.3° | 1.50 | 84.6° |
| 279445 (**junk**) | 12 | 6.17 | 6.17 | **89.2°** | **6.11** | — |

Raising `dyz` alone cannot separate them — the junk pair sits at dyz 6.17,
*inside* the good range 5.10–6.50. The direction test separates cleanly.

### Fix

`clus/src/ClusteringProtectBundle.cxx`: four new knobs, all default OFF/
prototype-faithful —

| knob | unit | default | meaning |
|---|---|---|---|
| `cathode_rejoin_perp` | internal | `0` (OFF) | max perpendicular tip offset from the other component's local direction line |
| `cathode_rejoin_angle` | degrees | `20.0` | max direction–direction angle |
| `cathode_rejoin_dir_radius` | internal | `15cm` | local-PCA radius about each tip |
| `cathode_rejoin_dir_npts` | int | `20` | min points in that radius, per side |

When `dyz >= cathode_rejoin_dyz` and `cathode_rejoin_perp > 0`, a new
`direction_agrees()` helper computes a local PCA direction at each tip
(`get_closest_wcpoints_radius` + the same 3×3-covariance/
`Eigen::SelfAdjointEigenSolver` pattern as `Facade_Cluster::get_pca()`,
restricted to the radius), requires `>= dir_npts` points per side, folds the
angle between the two directions (`collinear_deg`), and requires both a
collinearity bound and a perpendicular-offset bound. Uses the SAME
per-component `Simple3DPointCloud` clouds `cathode_rejoin()` already built —
deliberately not `Cluster::vhough_transform()`, which at re-join time would
mix in the other component's points (both are still blobs of the same
not-yet-split cluster). `dis` and `xcut` are untouched.

jsonnet: four `null`-defaulted args threaded through
`cfg/pgrapher/common/clus.jsonnet`'s `protect_bundle()` (key-suppression
idiom) and **both** SBND entry points — `clus.jsonnet`'s `clus_pr()` top-level
function *and* `wct-pr-perevt.jsonnet`'s TLA layer (the pr/23 both-files
trap: setting one leaves the other's explicit `null` overriding it back OFF).
Runner escapes added to `run_pr_evt.sh` / `run_pr_chain_batch.sh`:
`SBND_PROTECT_REJOIN_PERP` (cm) / `_ANGLE` (deg) / `_DIR_RADIUS` (cm) /
`_DIR_NPTS`, matching the existing `_XCUT`/`_DYZ`/`_DIS` idiom.

**Still OFF in the SBND production defaults** (`clus.jsonnet`'s `pr()` and
`wct-pr-perevt.jsonnet`'s TLA both pass `null` for the four new args) —
turning the operating point on is the owner's call (escalation rule 1).
Proposed operating point once approved: `perp=3cm`, `angle=20.0`,
`dir_radius=15cm`, `dir_npts=20` (margins: good pairs perp ≤2.68cm/angle
≤7.1°, junk stub perp 6.11cm/angle 89.2° — also fails `dir_npts`
independently).

### Verification (small group, per owner direction — full valfast/1000 deferred to a joint campaign)

- **Build**: `wcbuild` — clean compile + link (`WireCellClus`), ELF verified
  (an earlier link failure was the pre-existing M3 build race, unrelated to
  this change — resolved by the standard re-run). **Freshness proof, exact
  claim**: a `git stash`/`pop` cycle (see GOTCHA below) bumped
  `ClusteringProtectBundle.cxx`'s mtime without changing its content, so a
  naive mtime-newer-than-source check reads FALSE after that point. What was
  actually established: `wcbuild` run again found (via waf's content-hash
  dependency check, not mtime) that no rebuild was needed — i.e. the
  installed `libWireCellClus.so` already reflects the current source
  content — and `wcdoctest-clus` passed 49/49 against that exact binary.
  That is a stronger guarantee than mtime ordering, just not the M1 mtime
  check as literally stated in CLAUDE.md.
- **Unit tests**: `./build/clus/wcdoctest-clus` — 49/49 PASS, 0 failed.
- **Compiled-config proof**: `wcsonnet` on `wct-pr-perevt.jsonnet` with the
  production pipeline TLAs (pr/23 gotcha: default TLAs never instantiate the
  PR node). Knob-off compiled JSON is **byte-identical** to a pre-change
  checkout (`git stash`/pop pair, `diff` empty). Knob-on: new keys
  `cathode_rejoin_perp=90` / `cathode_rejoin_angle=20` appear in the
  `ClusteringProtectBundle` data block for `protect_cathode_rejoin_perp=3*cm`.
- **Small-group run**: `run_pr_chain_batch.sh work-mcp1kall-vfprodoff
  work-pr25-{off,on} data <13 events>`, same binary, fresh arm tags
  `work-pr25-off` / `work-pr25-on`.
  - **Knob-off byte-identical**: `hash_archive.py` on `mabc-pr.zip`
    (member-content hash, M2) — **13/13 PASS** against the pre-existing
    `work-vfmcp1k-prodon` arm.
  - **Knob-on, positive set (5/5 re-joined)**: 399118, 489327, 289832,
    488139, 492913 each log exactly one re-join **"via direction fallback"**
    (new tracer text), 0 → 1 each.
  - **Knob-on, negative control (1/1 stayed rejected)**: 279445 — 0 re-joins,
    unchanged.
  - **Knob-on, regression set (8/8 byte- AND score-identical)**: 169824,
    286400, 406796, 56463 (pre-existing dyz-passing re-joins, unchanged
    counts), 315497, 409634, 287654 (pre-existing non-re-joining pairs,
    unchanged) — `hash_archive.py` identical AND `pr_scores_table.py` every
    column identical.
  - **Physics effect on the 5 positive-set events** (`pr_scores_table.py`):
    `nu_sel_len_cm` increases in every case (e.g. 489327: 349.0 → 383.0 cm;
    492913: 30.5 → 49.8 cm — the previously-truncated far fragment's charge
    is restored to the candidate). `event_label` and `cosmic_flag` **do not
    change on any event** — only internal kinematics/scores move.
    489327's `nue_score` moves 2.52 → **−15.0** — this is a sentinel, not a
    physics measurement: across the full 572-event `vf-scores.tsv`, 322
    events sit at exactly −15.0 and 114 at exactly −4.300936 (the pre-split
    489327 value), i.e. the nue BDT emits one of two floor values on the
    large majority of events in *both* arms. The +2.52 → −15.0 move says the
    nue branch stopped producing a distinguishing score on this event; it
    does **not** establish that the +2.52 was an artifact of the split, and
    this doc makes no directional claim about it. `numu_score` (a
    continuously-valued column here) is the more trustworthy read: 2.20 →
    3.14, moving further from the nue/cosmic side.
  - **Runtime vs. census numbers differ, verdicts agree.** The C++ tracer's
    live `dis`/`dyz` at re-join time (e.g. 489327: gap 5.73, dyz 5.14) are
    not the same numbers as the offline `pr25_rejoin_census.py` scan (6.83 /
    6.50) — different pipeline stage, likely a different coordinate scope
    snapshot (the census reads the persisted post-hoc pctree; the tracer
    reads the live default-scope cluster mid-pipeline). Both agree on the
    verdict (dis passes, dyz fails, direction fallback fires) for all 5
    positive-set events; the absolute values should not be treated as the
    same measurement.
- **Determinism**: `setarch x86_64 -R`, 3 independent runs of the 5
  positive-set events — **3-way `hash_archive.py` identical** on every event.

### Status

**NOT bit-identical when on** — it lands on top of a `protect_bundle` default
that flipped ON the same day (`f813e312`). Knob-off path is byte-identical
(proven above); this is currently true both because the new fallback is
prototype-faithful when disabled AND because it is not yet the SBND default.

**C++ (`clus/src/ClusteringProtectBundle.cxx`) and jsonnet
(`cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/
{clus,wct-pr-perevt}.jsonnet`) changes are validated but UNCOMMITTED** —
per CLAUDE.md, commits land only when asked. `sbnd_xin/run_pr_evt.sh` and
`run_pr_chain_batch.sh` (runner knob wiring) are similarly uncommitted in
`wcp-porting-img`. Only `pr25_rejoin_census.py` (read-only analysis, step 0
of the plan) is committed so far (`e370ded`).

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild   # freshness proof: check local/lib/libWireCellClus.so mtime after
./build/clus/wcdoctest-clus   # 49/49 expected

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 pr25_rejoin_census.py scan     # reproduces the 6-pair dyz-only census
python3 pr25_rejoin_census.py direct   # reproduces the angle/perp separation

EVTS="399118 489327 289832 488139 492913 279445 169824 286400 406796 56463 315497 409634 287654"
./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25-off data $EVTS
SBND_PROTECT_REJOIN_PERP=3 SBND_PROTECT_REJOIN_ANGLE=20.0 \
SBND_PROTECT_REJOIN_DIR_RADIUS=15 SBND_PROTECT_REJOIN_DIR_NPTS=20 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25-on data $EVTS

python3 pr_scores_table.py --root work-pr25-off > /tmp/off.tsv
python3 pr_scores_table.py --root work-pr25-on  > /tmp/on.tsv   # diff the two
```

---

## 2. Event 320029 — implemented, default OFF, small-group measured

Owner-confirmed target: **cluster 30** (point (191.2, 190.8, 112.7) cm, Bee
label `30004`) — a 37.0 cm demoted-main fragment whose two ends sit on two
different active-volume boundaries (anode, 0.2 cm; top, 0.0 cm), a genuine
TGM CASE-A shape.

### Root cause

`check_tgm` DOES evaluate it (`wct_pr_evt320029.log`):
```
check_tgm: cluster 30 CASE-A pair (0,1) rejected: neither end in the pre-merge main cluster (37.0 cm chord)
```
`main_pair_rejects` (`TaggerCheckTGM.cxx:846-857`, `tgm_main_pair_mode='real'`)
reads `real_cluster_main` — "was this blob the merge's single
representative" — which is by construction all-zero on every demoted main
(a demoted main was, by definition, a main but not the representative), and
`ClusteringUnmergeBundle::carve` (`:359-372`) copies that array verbatim into
the split part with nothing re-stamping it. So the veto fires unconditionally
on every demoted main, before any geometry. This exactly predicts the
standing census (14 demoted-main convictions in 1000 events, all STM, TGM
never) — previously read as a physics fact, actually a code interaction
between P3 (`evaluate_demoted_mains`, which opens the door) and this
unrelated guard (which closes it again).

### Fix — `exempt_demoted_main_pairs`, default OFF

`clus/src/TaggerCheckTGM.cxx`: a scoped exemption in `main_pair_rejects`,
gated by a new default-OFF knob `exempt_demoted_main_pairs`, mirroring
`m_evaluate_demoted_mains` next to it — skip the veto when
`cluster.get_flag(Flags::demoted_main)`. `real_cluster_main` itself and
`ClusteringUnmergeBundle::carve` are untouched (the array is read elsewhere;
a global re-stamp would be a much bigger blast radius than this one-consumer
exemption). Threaded through `cm.tagger_check_tgm(...)`
(`cfg/pgrapher/common/clus.jsonnet`), both SBND `clus_pr()`/`pr()` levels in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, and the standalone
`wct-pr-perevt.jsonnet` TLA layer (all four files required together — the
doc pr/23 both-files trap). Compiled-config proof against the production
pipeline (`switch_scope,...,tagger_check_tgm,...,tagger_output`, not the
default empty pipeline, which never instantiates the TGM node): knob-off
byte-identical to pre-change HEAD, knob-on inserts
`"exempt_demoted_main_pairs" : true`. `wcdoctest-clus` 49/49 PASS.

### Small-group measurement (result: fix works, AND surfaces a second gap)

**Positive/target — cluster 30, evt 320029.** With the exemption on, the
`main_pair_rejects` reject line disappears and CASE-A geometry runs:
`ngrp=2` (two extreme groups), `TGM=true`. `skip_cosmic_companions` (already
SBND ON, doc pr/20 Part I P4) then correctly drops it: `companion cluster 30
(L 38.5 cm, TGM=1 STM=0) dropped from other_clusters`. Net physics:
`nu_sel_n_assoc` 5→3, `kine_reco_Enu_MeV` 766.6→639.6 (−127 MeV, cluster 30's
charge no longer credited to the neutrino), `numu_score` 3.612→3.547.
**Cluster 29** (1.0 cm) stays `TGM=false` in both arms, as expected — CASE-B,
too short to reach any accept branch. Determinism: 3-way identical
`mabc-pr.zip` content hash under `setarch x86_64 -R`, knob on.

**Comparison — the 14 demoted-main STM-only convictions** (doc pr/20 Part I
§6 table: 283595, 281595, 489327, 394796, 73004, 169356, 317939, 315849,
285467, 278684, 314507, 288639, 59003, 282899), same-binary two-arm A/B
(`work-mcp1kall-vfprodoff` QL hub). Knob-off: byte-identical to before this
change on all 15 events combined (`nusel-table.tsv` diff empty). Knob-on: **8
of the 14 events pick up at least one new `TGM=true`** on a demoted main (one
event, 283595, gets two — see below). Of the clusters that newly flip, **5
are exact duplicates** of the cluster STM already convicted in that same
event (TGM now runs first in the pipeline and gets there before STM does;
`skip_cosmic_companions` drops the same cluster's charge either way — **zero**
net score/energy change, confirmed against `pr_scores_table.py` output:
283595/cluster 23, 281595/27, 317939/32, 59003/26, 282899/13). The remaining
**6 events are byte-identical end-to-end, no TGM change at all**: 489327,
394796, 73004, 169356, 285467, 314507.

**The other 4 flips are a genuinely new finding, not duplicates**, and this is
the part that blocks a default-ON decision:

| evt | cluster | STM in OFF | npts | bbox diag | location | companion of nu-candidate bundle? | score effect |
|---|---|---|---|---|---|---|---|
| 283595 | 26 | 0 | 12 | 0.9 cm | z 500.4–501.0 (z-max wall) | no | none |
| 315849 | 18 | 0 | 12 | 0.9 cm | y −198.1..−198.5 (bottom wall) | no | none |
| 278684 | 18 | 0 | 14 | 1.6 cm | y 196.8–198.1 (top wall) | no | none |
| 288639 | 15 | 0 | 8  | 1.3 cm | y 199.1–200.0 (top wall) | **yes** | `nu_sel_n_assoc` 10→9, `kine_reco_Enu_MeV` 808.3→811.9 (+3.6 MeV) |

All four are **8-14 point debris specks sitting exactly at a detector wall**,
not tracks. `WCT_TGM_DEBUG=1` on 288639/cluster 15 confirms the mechanism:
`ngrp 2 ... mid_inside false len 5.2/5.2 cm` — `check_tgm`'s CASE-A branch
`out_vec_wcps.size() == 2 → return true` (`TaggerCheckTGM.cxx:917`) has **no
length test at all** when the two extreme-group midpoints fail to reach the
FV interior; a speck with two "ends" ~1 cm apart, both sitting on a boundary,
satisfies it trivially. This branch is pre-existing and untouched by this
fix — it applies to any cluster, demoted or not — but it was **never
reachable for a demoted main** before, because `main_pair_rejects` vetoed
every demoted-main pair unconditionally first. Lifting that veto makes this
gap newly relevant to a population (small, wall-hugging debris fragments)
that `main_component_pairs` was never validated against. Only 288639's speck
happens to sit inside the same flash bundle as a selected nu-candidate, so
only one of the four moved a score — but that is a property of these
particular 4 events, not a bound on the effect size in general.

### Default-ON is blocked on the debris-speck gap, not the original veto question

The veto exemption itself does exactly what §2's root-cause analysis
predicted (cluster 30, cleanly). What it also does — convict tiny boundary
specks that were never a design target of `check_tgm`'s CASE-A shortcut — is
a **pre-existing gap** (CLAUDE.md: found alongside, not fixed in this
change). Candidate co-requisite fix, described but **not implemented**: an
absolute length floor on the `out_vec_wcps.size()==2` branch, or requiring
`component_min_length` even when the component-restricted extreme search
yields the fallback global-8-extremes groups. Either needs its own
measurement round (does it exclude genuine short corner-clippers like the
30cm-ish tracks `component_rescue` was built to protect, doc 33/pr20?)
before it can be proposed as a change, and touches `check_tgm`'s CASE-A path
shared with uBooNE/prototype, not an SBND-only knob.

**Escalation rule 1**: changes cosmic verdicts — flipping `exempt_demoted_main_pairs`
to the SBND default needs both an owner decision on the veto question AND,
independently, a decision on the debris-speck gap above.

### SBND DEFAULT ON (owner decision, 2026-08-02)

Owner reviewed the debris-speck finding above and judged the measured impact
small enough to accept: only 1 of 4 new-speck convictions in the 15-event
small group was ever a bundle companion, and its effect was +3.6 MeV on a
~800 MeV `kine_reco_Enu_MeV` (~0.4%) — smaller than run-to-run noise-floor
movement measured elsewhere in this chain (doc pr/20 Part I: 7 differing
`kine_reco_Enu_MeV` cells on an A/A′ control, all last-digit). `wct-pr-perevt.jsonnet`'s
`tgm_exempt_demoted_main` TLA default flipped `false → true` (the SBND
single-source-of-truth entry point per doc 68; `clus.jsonnet`'s own
`clus_pr()`/`pr()` function defaults are unchanged at `false`, same pattern
as `evaluate_demoted_mains`). Verified: bare compile (no TLA override) now
emits `"exempt_demoted_main_pairs" : true`; explicit
`tgm_exempt_demoted_main=false` still suppresses the key; a bare
`run_pr_chain_batch.sh` run on evt 320029 (no env override) reproduces the
measured ON-arm behavior exactly (cluster 30 convicts, `skip_cosmic_companions`
drops it). **NOT bit-identical** — same bar as every other SBND default flip
in this doc series. The `check_tgm` CASE-A length-floor gap (debris specks)
remains open and untouched by this decision; it is a separate, not-yet-designed
fix, tracked here for whenever it gets picked up. Runner escape to restore
the pre-flip behavior: `SBND_TGM_EXEMPT_DEMOTED_MAIN=0`.

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# target + comparison set, knob off / on (same binary, fresh tags)
SBND_TGM_EXEMPT_DEMOTED_MAIN=0 ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25b-off data \
  320029 283595 281595 489327 394796 73004 169356 317939 315849 285467 278684 314507 288639 59003 282899
SBND_TGM_EXEMPT_DEMOTED_MAIN=1 ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25b-on  data \
  320029 283595 281595 489327 394796 73004 169356 317939 315849 285467 278684 314507 288639 59003 282899

python3 pr_scores_table.py --root work-pr25b-off --out /tmp/pr25b_off.tsv
python3 pr_scores_table.py --root work-pr25b-on  --out /tmp/pr25b_on.tsv   # diff the two

# mechanism check on a specific cluster (e.g. 288639/cluster 15)
WCT_TGM_DEBUG=1 SBND_TGM_EXEMPT_DEMOTED_MAIN=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25b-dbg data 288639
grep "check_tgm dbg: cluster 15" work-pr25b-dbg/pr_evt288639/wct_pr_evt288639.log

# determinism, knob on
for i in 1 2 3; do
  SBND_TGM_EXEMPT_DEMOTED_MAIN=1 setarch x86_64 -R env PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25b-det${i} data 320029
  python3 ../../abtest/hash_archive.py work-pr25b-det${i}/pr_evt320029/mabc-pr.zip
done
```

---

## GOTCHAS

- **Doc number pr/25, not pr/24.** A concurrent session in this shared tree
  had already reserved pr/24 (uncommitted `run_pr_chain_batch.sh` comment
  referencing DL/SCN neutrino-vertex weight attribution arms) — checked via
  `git status` before writing, per the concurrent-sessions convention. If a
  reader finds no `pr/24_*.md`, it belongs to that other round, not a gap
  here.
- **`run_pr_chain_batch.sh` and `cfg/pgrapher/experiment/sbnd/clus.jsonnet`
  repeatedly carried a concurrent session's edits in the same file diff**
  (`SBND_DL_WEIGHTS`/DL main-cluster swap guard, doc pr/24; then a second
  round adding `protect_skip_iso_xext/_frac/_min_len`, also doc pr/24 —
  isochronous-cluster split veto). §1's knob wiring landed clean via a
  per-hunk `git diff | split | git apply --cached` (never `git add -A` on a
  file another session is mid-edit on); §2's `SBND_TGM_EXEMPT_DEMOTED_MAIN`
  wiring in `run_pr_chain_batch.sh` used the same technique. `sbnd/clus.jsonnet`
  changed on disk mid-edit (a `Read`-before-`Edit` staleness warning) — the
  concurrent session's iso-veto hunk landed in the SAME function-arg block my
  §2 edit was touching; reread before editing further, verified the two
  additions don't overlap line-for-line before applying.
- **Build/install raced with the concurrent session's own edit to
  `ClusteringProtectBundle.cxx`** (M3-shaped, but the cause was a live edit,
  not a link race): `./wcb install` failed
  `'m_skip_iso_xext' was not declared in this scope` even though the member
  WAS declared in the file moments later — the other session's save landed
  mid-compile. Re-running build+install once (already the M3 rule) picked up
  the completed edit and succeeded.
- **`git stash` on a shared tree sweeps in every other session's uncommitted
  changes, not just yours.** Used once here to get a pre-change compiled-config
  baseline; `stash pop` restored cleanly (verified: the 3 pre-existing stash
  entries were untouched, the concurrent session's 4-file diff was intact
  after the pop) but it bumped `ClusteringProtectBundle.cxx`'s mtime with no
  content change, which read as a stale freshness proof until traced (see
  Verification). Prefer `git show HEAD:path > /home/xqian/tmp/baseline` per
  file for a future baseline compare — it never touches the working tree.
- **First compiled-config proof for §2 was a vacuous pass**: `pipeline_names=[]`
  (the default TLA) never instantiates `TaggerCheckTGM`, so a knob-off vs
  knob-on diff came back empty for the WRONG reason (no node = nothing to
  differ) rather than the right one (key-suppressed when off). Re-ran with
  the actual production `pipeline_names` list
  (`switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,
  tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,
  steiner_refresh,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,
  tracking_visitor,tagger_output` — from `run_pr_chain_batch.sh`'s `PIPELINE`
  default) and got a real byte-identical-off / key-present-on result. Same
  root cause as the standing "doc pr/23 gotcha" above; a second, independent
  near-miss of it in the same session is worth its own line.
- **§3's proxy-vs-direct-instrument trap actually fired, not just risked.**
  A 3-D nearest-fit-point proxy (no code changed, pure Python re-derivation
  from the pctree) predicted `is_shower_topology` should NOT have fired on
  321107 (0.72 cm computed spread < the code's 0.8 cm branch threshold); the
  real code fired it. Every number that proxy produced (the "0.37 narrow
  miss", the "associated-cloud contamination" theory) was wrong and had to be
  withdrawn once a real env-gated instrument was added and run. The general
  lesson from §2 ("get the direct instrument, don't infer from an adjacent
  debug line") generalizes further: a from-scratch Python re-derivation of a
  C++ algorithm is not a "direct enough" instrument either — a 2-D
  voronoi-with-ghost-removal association cannot be approximated by a 3-D
  nearest-neighbor proxy, no matter how carefully the geometry is checked by
  eye. Add the log line and run the actual binary.
- **A discriminant that is unstable across two evaluations of the same input
  is not a "close" measurement, it is a broken one for tuning purposes.**
  §3.6: `TaggerCheckNeutrino` runs the topology test twice per event
  (`:545`, `:673`, before/after `improve_vertex`); one segment (evt 277276)
  flips verdict between the two passes on a ~1cm change in the fitted
  length. First draft of this section over-read the raw per-event guard
  table as "3/21 events flip" — two of those three were actually two
  *different* long segments landing in the same event (one demoted, one
  surviving), not the same segment moving. Caught before push: the instrument
  logs `segment->id()`, which reads `-1` here (`Segment::set_id` runs later,
  `NeutrinoPatternBase.cxx:1858`), so grouping is only by event, and a
  same-segment claim needs the length continuity to back it up, not just
  co-occurrence in one event's guard table. n=1 is still sufficient to make
  the point; it is not evidence for a rate.

## 3. Event 321107 — investigated with a direct instrument; no safe fix found

Main segment (`real_cluster_id` 13000): 415 fit points, 248.72 cm, PCA angle
to drift-x **88.55°** — confirmed isochronous. `pdg=11`, `flag_shower=1`
across all 415 points, bit-identical on `prodon`/`prodoff`.

**Correction to an earlier pass of this section**: the previous write-up's
§3.4 ("associated-cloud contamination") and its `0.37` narrow-miss estimate
were built on a 3-D nearest-fit-point proxy, not the code's actual 2-D
voronoi + ghost-removed association. That proxy predicted the flag *should
not* fire (0.72 cm spread < the 0.8 cm branch threshold) on the one event
that matters, and it did fire — the proxy does not reproduce the mechanism.
This round replaces every proxy-derived number with a direct measurement,
via a new instrument (§3.4), and reaches a materially different conclusion.

### 3.1 Why an electron, not a muon — the owner's guess is ruled out twice

`NeutrinoTrackShowerSep.cxx:52-59`: `segment_is_shower_topology` runs first
and short-circuits `segment_is_shower_trajectory`. Separately,
`segment_is_shower_trajectory` (`PRSegmentFunctions.cxx:983-1078`) has a hard
`length > 50cm → return false` guard at `:988` — this 249 cm track could
never reach it regardless of order.

**The isochronous projection handling the owner remembered is real** —
`PRSegmentFunctions.cxx:1050-1067`, the `angle_diff <= 10°` branch of
`is_shower_trajectory` — it just never ran here: wrong function (topology
fired first), and length-gated out of the right one anyway.

Once topology fires, `NeutrinoTrackShowerSep.cxx:117-135` hard-assigns
`pdg=11`/`score=100` as literals — `segment_determine_dir_track`, the real
muon/proton/pion PID call, is never attempted on this path.

### 3.2 Isochrony is not the population-level driver

`is_shower_topology`'s only angle-to-drift special case
(`PRSegmentFunctions.cxx:2570-2578`) is `angle_deg < 7.5°` — drift-*parallel*,
the opposite regime from 321107. At 88.55° the general branch always runs,
using RMS spread along `dir_3`, which at this angle is ≈ the drift axis
(`dir_3 · x̂ ≈ 0.9997`).

Census (`pr25_shower_topo_census.py population --arm work-vfmcp1k-prodon
--nu-only`), 324 nu-candidate main segments >50 cm across 444/572 `prodon`
events (the other 128 have no `T_rec_charge` tree at all — no tracking
output, not a census artifact), binned by angle to drift:

| angle to drift | n | shower-flagged | rate |
|---|---|---|---|
| 0–30° | 16 | 1 | 6.2% |
| 30–50° | 58 | 3 | 5.2% |
| 50–65° | 72 | 1 | 1.4% |
| 65–75° | 70 | 7 | 10.0% |
| 75–85° | 74 | 4 | 5.4% |
| 85–90° | 34 | 3 | 8.8% |

Flat, within counting noise. **"Isochronous tracks get called showers" is
not a population-level effect** — 321107 is an individual case, not an
instance of an angle-dependent failure, and the owner's original framing of
the question (mirrored in the previous pass of this doc) should be retired.

### 3.3 The blast radius of any fix is small and exactly known

`PRSegmentFunctions.cxx:2765-2772`'s long-track guard only runs on segments
already flagged `kShowerTopology` with geometric length > 50 cm and can only
demote them — so the population any change can possibly touch is exactly
*that* set. Census over all clusters (not just nu-candidate mains) in the
same 444 events: 1850 segments written to `T_rec_charge`, 841 shower-flagged,
of which **21 exceed 50 cm, in 21 distinct events**:

```
286353 321107 53427 292643 281527 277276 387850 316729 62281 278420 404684
57903  65289  350935 280972 315167 286681 278684 347085 287621 400504
```

This is the small group used below — the owner's own scoping request, not a
new event selection.

### 3.4 Direct instrument, not a proxy

`segment_is_shower_topology` had **zero active log lines** before this round
(one relevant `std::cout` existed, commented out, at `:2660`). Added an
env-gated diagnostic (`WCT_SHOWER_TOPO_DEBUG=1`, default off, same idiom as
`WCT_TGM_DEBUG`) emitting, per evaluation: the per-bucket `dir_3` RMS
distribution (count over the 0.4 cm cut, median, p90 — not just the max),
association coverage (`total_effective_length / geometric length`), which of
the 5 disjunction branches fired, and the long-track guard's
`total_length1/2` fractions. `clus/src/PRSegmentFunctions.cxx`.

Target (321107, cluster 13, seg 13000, L = 248.7 cm), measured directly:

```
assoc_npts 8136  nbuckets 389  n_over0.4cm 134  rms_p50 0.36cm  rms_p90 0.50cm
max_spread 1.24cm  lsl 80.4cm  tel 233.7cm  lsl/tel 0.344  tel/L 0.940  branch 2
guard: total_length1 64.2cm (0.258·L)  total_length2 63.6cm (0.256·L)  demoted false
```

This directly refutes two of the three candidate mechanisms considered:

- **Not sparse association** (candidate A, "the coverage denominator is
  small"): `tel/L = 0.940` — 94% of the track's length has ≥2 associated
  points. This is a well-covered segment, not a starved one.
- **Not a single outlier bucket** (candidate B, "one bucket carries
  `max_spread`"): 134 of 389 populated buckets (34%) exceed the 0.4 cm cut,
  and the *median* bucket RMS (0.36 cm) already sits just under that cut. The
  spread is broadly distributed across a third of the track, not concentrated
  in one or two buckets that a robuster statistic would filter out.

What remains is **candidate C**, the long-track guard's `0.25` fraction: the
target genuinely sits close to it (0.258 / 0.256 vs 0.25) — a real
near-miss, this time measured directly rather than inferred from a proxy.

### 3.5 The 0.25 threshold has no natural break across the affected population

Re-running the same instrument over all 21 events plus ~9 muon controls that
correctly stayed tracks (`pr25_shower_topo_census.py guard --arm
work-pr25s3-dbg21`): 41 evaluations of an L>50cm shower-flagged segment, of
which the *existing* guard already correctly demotes 6 (they are not part of
the population below). The controls contribute **zero** guard-evaluated
segments — the guard cannot touch a segment that never gets flagged in the
first place, so "the controls are untouched" does not by itself validate any
candidate, exactly as anticipated before running this.

The 35 that survive, by `max(total_length1, total_length2) / L`:

```
0.570 0.398 0.375 0.369 0.361 0.357 0.353 0.353 0.338 0.304 0.304 0.300 0.300
0.300 0.293 0.292 0.285 0.282 0.278 0.278 0.276 0.276 0.274 0.273 0.273 0.270
0.270 0.268 0.268 0.265 0.265 0.259 0.258 0.258 0.256 0.250
```

**Continuous from 0.250 to 0.570, no gap.** 321107 sits at the low end
(0.258), tied with several others. **The 21 is a blast-radius population,
not an error population** — the census identifies every segment a guard
change could touch, not which ones are wrongly classified; some are almost
certainly genuine showers. There is no truth label on this sample (M11/
no-truth caveat, as in every prior round) to separate them, so there is no
principled place to draw a raised threshold: any value that catches 321107
also catches an a-priori-unknown number of the other 20 (genuine or not),
and a value that catches only 321107 does not exist (277276 sits at the
identical 0.250).

### 3.6 The discriminant is not stable at the margin

`TaggerCheckNeutrino.cxx` runs `clustering_points` + `separate_track_shower`
**twice** per event — once before `improve_vertex`, once after
(`:545` and `:673`). The instrument's per-segment identity field reads `-1`
at this point in the pipeline (`Segment::id()` is not stamped until later,
`NeutrinoPatternBase.cxx:1858`), so grouping is by event, and most events in
the table contain more than one L>50cm shower-flagged segment — a demoted
one and a surviving one in the same event is unremarkable on its own and
does **not** by itself show a same-segment flip. Filtering to the case where
the length is continuous enough across the two passes to be plausibly the
same segment leaves **one clean instance, event 277276**: the fit shifts by
~1 cm between passes and the verdict flips:

| evt | pass-1 | pass-2 | verdict |
|---|---|---|---|
| 277276 | 140.4 cm (f=0.244, demoted) | 139.3 cm (f=0.250, survives) | flips |

(65289 and 278684 also show "demoted" and "survives" rows in the same event,
but at length pairs 150.3/69.5 cm and 109.6/59.2-59.3 cm — too far apart to
be the same segment continuing between passes, more likely two distinct
long segments in the same event, one genuinely a shower and one not. Listed
here only so the raw counts in §3.5 are traceable; they are **not** claimed
as flips.)

This one instance is enough to make the point, even at n=1: the
classification the guard produces for a fixed physical track is not
necessarily stable across two evaluations of the *same* input a few
centimeters apart. It does not by itself carry the rejection of candidate C
— that rests on §3.5's continuous, gapless 0.250–0.570 distribution across a
population with no truth labels — but it reinforces why tuning the `0.25`
constant would not resolve the underlying issue: at least for this one
segment, the constant sits inside a region the algorithm's own two passes
do not agree on. Per CLAUDE.md §5's "a physics number looks wrong: report,
don't tune", and per the selection rule fixed in this round's plan before
any of the above was measured: **no candidate cleanly separates the target
from its own population, so no threshold change is proposed.**

### 3.7 Prototype comparison

No divergence — `prototype_base/pid/src/ProtoSegment.cxx:319-471` (topology)
and `:543-611` (trajectory) match the toolkit line-for-line on every
threshold and the same `<7.5°` drift-parallel-only special case. This is not
a porting bug; the behavior is original to the algorithm design, and the
same-input verdict instability of §3.6 is therefore also original, not a
toolkit regression.

### 3.8 Conclusion — no fix this round; the instrument is the deliverable

Three candidate default-OFF knobs were designed in advance of measurement
(coverage floor, per-bucket-count spread statistic, raised long-track
fraction) with an explicit selection rule committed before running the
instrument. All three fail that rule: A and B are directly refuted on the
target itself (§3.4); C shows no separation and, worse, sits on a
demonstrably unstable discriminant (§3.5–3.6). Per the pre-committed rule,
**no fix is shipped.**

What *is* shipped: the `WCT_SHOWER_TOPO_DEBUG` instrument itself
(`clus/src/PRSegmentFunctions.cxx`, default off, zero behavior change either
way — it only adds log lines) and `pr25_shower_topo_census.py`, both durable
assets for the next pass at this problem.

**If pursued further**: `segment_is_shower_topology` already computes a
per-bucket `dQ/dx` (`vec_dQ_dx`, `PRSegmentFunctions.cxx:2616`) that is
currently **dead code** — read only for its `.size()`, never its values
(`:2681`, `:2722`). A shower's dQ/dx profile (cascade multiplication) differs
physically from a muon's (~constant, MIP-like) in a way none of the three
candidates above used. Using it would be a genuine algorithm addition, not a
threshold tune, and needs its own measurement round — not proposed here.

### Verification performed

- `./wcb build --notests -p && ./wcb install --notests -p`; freshness proof
  (`local/lib/libWireCellClus.so` newer than the edit).
- `./build/clus/wcdoctest-clus`: 49/49 pass, 565/565 assertions, 0 failed.
- The instrument is a pure log addition behind `std::getenv`, read once into
  a function-local `static const bool` — no control-flow change on any path,
  so no compiled-config proof or byte-identical gate is needed (there is no
  new jsonnet key; nothing is threaded to config).
- Knob-on smoke: ran on the full 21-event population + 9 controls
  (`work-pr25s3-dbg21`, `PR_JOBS=6`, all 30 events `rc=0`); the debug lines
  are exactly what §3.4–3.6 quote.

No cfg change, no default flip, no A/B gate — there is no behavior-affecting
change in this section, by design (§3.8).

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# population census: angle-independence + exact blast-radius population
python3 pr25_shower_topo_census.py population --arm work-vfmcp1k-prodon --nu-only
python3 pr25_shower_topo_census.py population --arm work-vfmcp1k-prodon   # all clusters, the 21-event set

# direct instrument on the target
WCT_SHOWER_TOPO_DEBUG=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25s3-dbg1 data 321107
grep "shower_topo dbg" work-pr25s3-dbg1/pr_evt321107/wct_pr_evt321107.log

# direct instrument on the full 21-event population + muon controls
WCT_SHOWER_TOPO_DEBUG=1 PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25s3-dbg21 data \
  286353 321107 53427 292643 281527 277276 387850 316729 62281 278420 404684 \
  57903 65289 350935 280972 315167 286681 278684 347085 287621 400504 \
  402880 65295 168614 48367 394642 291570 283771 319611 67746
python3 pr25_shower_topo_census.py guard --arm work-pr25s3-dbg21
```

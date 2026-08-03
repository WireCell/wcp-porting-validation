# doc pr/25 — cathode re-join direction-agreement fallback (ClusteringProtectBundle) + TGM demoted-main veto (TaggerCheckTGM)

Also documents a third finding from the same Bee hand-scan round (321107),
investigated alongside these two; see the status table below.

| evt | question | status |
|---|---|---|
| **489327** | cathode crosser broken in two; the pr/23 re-join knob did not fire | **root cause proven, fix implemented + validated on a small group** (§1). **Committed** (toolkit `75c703da`, wcp-porting-img `c8d4e32`); SBND default left OFF pending owner go-ahead. |
| **320029** | another cluster (30) looks like a TGM but was not tagged | root cause proven and **fix implemented + small-group measured** (§2): a structural TGM/demoted-main veto interaction, not a tuning question. The measurement surfaced a second, pre-existing `check_tgm` gap — see §2. **Default-ON blocked on that gap, not on the original veto question.** |
| **321107** | main track is a muon, tagged as an electron | mechanism fully traced: `is_shower_topology`, not `is_shower_trajectory`, made the call (§3). **No fix proposed.** |

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

## 3. Event 321107 — explained, no fix proposed

Main segment (`real_cluster_id` 13000): 415 fit points, 248.72 cm, PCA angle
to drift-x **88.55°** — confirmed isochronous. `pdg=11`, `flag_shower=1`
across all 415 points, bit-identical on `prodon`/`prodoff`.

### Why an electron, not a muon

`NeutrinoTrackShowerSep.cxx:52-59`: `segment_is_shower_topology` runs first
and short-circuits `segment_is_shower_trajectory`. Separately,
`segment_is_shower_trajectory` (`PRSegmentFunctions.cxx:983-1078`) has a hard
`length > 50cm → return false` guard at `:988` — this 249 cm track could
never reach it regardless of order.

**The isochronous projection handling the owner remembered is real** —
`PRSegmentFunctions.cxx:1050-1067`, the `angle_diff <= 10°` branch of
`is_shower_trajectory` — it just never ran here: wrong function (topology
fired first), and length-gated out of the right one anyway.

`is_shower_topology`'s only angle-to-drift special case
(`PRSegmentFunctions.cxx:2570-2578`) is `angle_deg < 7.5°` — drift-*parallel*,
the opposite regime. At 88.55° the general branch always runs, using RMS
spread along `dir_3`, which at this angle is ≈ the drift axis (`dir_3 · x̂ ≈
0.9997`) — arguably the *right* projection for an isochronous track, not a
gap: a same-cut recompute using `dir_2` (the in-plane transverse, where the
isochronous ambiguity actually lives) fires far more aggressively
(length-fraction 0.99 vs 0.41 measured — indicative 3D-radius proxy, not the
code's exact 2D-projection/ghost-removed association).

Best working theory: associated-cloud contamination, not the angle handling
— `n_frag=5`, median point-to-trajectory distance 5.4 cm (90th pct 14.8 cm).
The only long-track guard (`:2766-2773`, `length1 < 0.25·L`) misses narrowly
(≈0.37 measured vs the 0.25 threshold). Once topology fires,
`NeutrinoTrackShowerSep.cxx:117-135` hard-assigns pdg 11/score 100 — no PID
call is ever attempted.

Prototype comparison: no divergence (`prototype_base/pid/src/
ProtoSegment.cxx:319-471`/`:543-611` match line-for-line). Not a porting
bug — original to the algorithm design.

**No fix proposed.** If pursued: the `0.25` veto fraction is shared with
uboone/the prototype (a real algorithm change, not an SBND-only knob); the
associated-cloud contamination needs its own root-cause pass first.

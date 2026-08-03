# doc pr/25 — cathode re-join direction-agreement fallback (ClusteringProtectBundle) + TGM demoted-main veto (TaggerCheckTGM)

Also documents a third finding from the same Bee hand-scan round (321107),
investigated alongside these two; see the status table below.

| evt | question | status |
|---|---|---|
| **489327** | cathode crosser broken in two; the pr/23 re-join knob did not fire | **root cause proven, fix implemented + validated on a small group** (§1). **Committed + pushed** (toolkit `75c703da`, wcp-porting-img `c8d4e32`). **SBND DEFAULT ON** (owner 2026-08-03) — `protect_cathode_rejoin_perp = 3 * wc.cm` with `angle=20.0`, `dir_radius=15 * wc.cm`, `dir_npts=20`. |
| **320029** | another cluster (30) looks like a TGM but was not tagged | root cause proven, fix implemented + small-group measured (§2): a structural TGM/demoted-main veto interaction, not a tuning question. The measurement surfaced a second, pre-existing `check_tgm` gap — see §2. **SBND DEFAULT ON** (owner 2026-08-02, impact judged small); the `check_tgm` gap remains an open, separate item. |
| **321107** | main track is a muon, tagged as an electron | **mechanism fully explained AND fix SHIPPED, default OFF** (§3, three rounds). `is_shower_topology` made the call, and its only measurement axis satisfies `dir_3·x̂ = sin θ` — at 88.55° (`dir3x = 0.9994`) it *is* the drift axis, so the wide halo (rms 8.34 cm in `dir_2` vs 0.40 cm in `dir_3`) **never entered the decision**: the flag fired on drift quantization noise (0.313 cm lattice vs a 0.4 cm cut). **The 2-D-projective-narrowness direction would not have changed this verdict.** 321107 is not exceptional — **86 of 91** long firings across 429 events sit in the same noise floor. Two spread-statistic fixes were pre-committed and rejected by their own stop rules; **the owner's 2026-08-03 hand-scan of all 10 nu-main cases returned 10/10 tracks, 0 showers**, which both supplied the missing truth and refuted the round-2 candidate (evt 400504 would have survived it and is a track). Fix = a length rule reusing the existing demote-only guard: **`shower_topo_demote_len`, C++ default 0 = OFF = byte-identical**, threaded to all four call sites. At 50 cm: 23 long segments flip shower→track across 22 events, **0 the other way**; **17/572 events (3.0%)** move `numu_score`; **321107 `pdg 11→13`, `numu_score −0.783 → +0.317`**. Score moves are mixed — **286353 drops 2.023 → −1.139**, flagged for owner judgement, not tuned away. **SBND DEFAULT ON at 50 cm** (owner 2026-08-03), verified by a bare production run (evt 321107 → `pdg 13`). |

### SBND production default status — all three, verified in-tree

Read straight from `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (the
single source of truth for the SBND operating point, doc 68), not from
memory. **All three are committed and pushed; only §2 is ON.**

| § | evt | knob | value in `wct-pr-perevt.jsonnet` | SBND default | commit |
|---|---|---|---|---|---|
| 1 | 489327 | `protect_cathode_rejoin_perp` | `3 * wc.cm` | **ON** (owner 2026-08-03) | `75c703da` + flip below |
| 2 | 320029 | `tgm_exempt_demoted_main` | `true` | **ON** (owner 2026-08-02) | `08d6aef4` + flip `b04b1762` |
| 3 | 321107 | `shower_topo_demote_len` | `50` | **ON** (owner 2026-08-03) | `5beb248a` + flip below |

**All three are now the SBND production default** (owner, 2026-08-03).
§1's companion knobs go on with it: `protect_cathode_rejoin_angle = 20.0`,
`_dir_radius = 15 * wc.cm`, `_dir_npts = 20` — these three equal the C++
defaults exactly (`ClusteringProtectBundle.cxx:261-263`), so they are inert
keys; only `perp` (0 → 3 cm) changes behavior.

**Bare-run proof, both flips together** (they had only ever been measured
separately — `run_pr_chain_batch.sh` with **no** env override,
`work-pr25s3r2-prodflip`, both events `rc=0`):

- §3 — evt 321107 main cluster: **`pdg 11 → 13`, mean `flag_shower`
  1.00 → 0.00** vs the pre-flip arm.
- §1 — evt 489327: `cluster 19: cathode re-join comp 0+4 (gap 6.97 cm,
  dyz 6.50 cm, x -0.49/2.05 cm, **via direction fallback**)`, and the split
  summary now reads `1 cathode re-join(s)` where it read `0`.

**NOT bit-identical** — this is a deliberate production-behavior change, the
same bar as every other SBND default flip. Legacy is restorable per knob with
`--tla-code protect_cathode_rejoin_perp=null` /
`--tla-code shower_topo_demote_len=null` (verified: both keys drop out of the
compiled config). Note `-A` is an ext-var and does **not** override a
top-level function arg — use `--tla-code`. With `perp` nulled the compiled
config differs from pre-flip only by §1's three inert companion keys.

Caveat carried forward, not tuned away: §3.10's `numu_score` moves are mixed,
and **evt 286353 drops 2.023 → −1.139** on an event the owner scanned as a
track. The owner accepted the flip with that on record; it is the first thing
to look at if the next valfast census surprises.

Repro base for all three: `sbnd_xin/work-vfmcp1k-prodon` (protect_bundle ON,
the SBND production default since `f813e312`) and `work-vfmcp1k-prodoff`,
572-event valfast subset, HEAD `f813e312` on 2026-08-02.

---

## 1. Event 489327 — COMMITTED + PUSHED, **SBND DEFAULT ON**

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

**COMMITTED AND PUSHED** — toolkit `75c703da` (C++
`clus/src/ClusteringProtectBundle.cxx` + jsonnet
`cfg/pgrapher/common/clus.jsonnet`,
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`);
wcp-porting-img `c8d4e32` (doc) and `e370ded` (`pr25_rejoin_census.py`),
plus the `run_pr_evt.sh` / `run_pr_chain_batch.sh` runner escapes.

**SBND DEFAULT ON (owner decision, 2026-08-03).** `wct-pr-perevt.jsonnet`
now ships `protect_cathode_rejoin_perp = 3 * wc.cm`,
`protect_cathode_rejoin_angle = 20.0`,
`protect_cathode_rejoin_dir_radius = 15 * wc.cm`,
`protect_cathode_rejoin_dir_npts = 20` — the operating point this section
measured. Per doc 68 the value lives in `wct-pr-perevt.jsonnet` only;
`clus.jsonnet`'s `clus_pr()`/`pr()` function defaults stay `null` (the pr/23
both-files trap is about an explicit `null` overriding a value set here, not
the reverse). Verified by a bare `run_pr_chain_batch.sh` run with no env
override: evt 489327 logs `1 cathode re-join(s)` **via direction fallback**.

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

## 2. Event 320029 — COMMITTED + PUSHED, **SBND DEFAULT ON**

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
- **A discriminant's threshold can sit inside the reconstruction's own
  quantization, and then it measures nothing.** §3.3: reconstructed x is
  quantized on a **0.313 cm** time-slice lattice, and
  `segment_is_shower_topology`'s "large spread" cut is **0.4 cm** on an axis
  that for an isochronous segment *is* the drift axis. The per-bucket RMS
  median is 0.35 cm for essentially every long segment in the manifest — a
  constant of the detector, not a property of any track. Before tuning any
  spread threshold, check it against the granularity of the coordinate it is
  measured in.
- **`dir_3` is not "the transverse direction" — it is the drift-containing
  one.** `dir_2 = x̂ × dir_1` has no x-component by construction, so
  `dir_3 = dir_1 × dir_2` satisfies `dir_3·x̂ = sin(angle-to-drift)`. The
  function measures spread *only* along `dir_3`. For an isochronous track
  that means the visible halo (which lives in `dir_2`, in the wire plane) is
  invisible to the test, and conversely the test is reading the one axis where
  imaging is *unambiguous*. Anyone reasoning about this function from an event
  display is looking at the wrong axis — check `dir3x` in the instrument first.
- **A gap found in a blast-radius population is not evidence of a gap.**
  §3.5: `rms_p90` showed a clean 0.60→1.17 cm separation on the 21-event
  round-1 arm, with n=3 on the high side — and that population was selected
  *because* it contained long shower-flagged segments. On the full 572-event
  manifest the short-segment population fills the gap (27.6% below 0.7 cm),
  the statistic turns out to be monotonic in segment length, and the gap's
  position moves with the quantile (0.47/0.64/1.08 at p75/p90/p95). Two
  rounds in a row the selected population produced a separation that the
  general population did not have. **Write the stop rule before the wide run,
  not after seeing it.**
- **`hash_archive.py` prints `<hash>  <count>  <path>` — compare field 1.**
  Diffing the whole line reports 100% DIFF between two arms because the path
  differs, which reads exactly like a failed byte-identical gate. This fired
  in §3 round 3 (40/40 false diffs, actually 40/40 identical).
- **The owner's hand-scan is what made the cut derivable.** Two rounds of
  increasingly clever spread statistics both failed their own pre-committed
  stop rules; ten scanned events settled it in one message, and *also*
  refuted round 2's candidate (evt 400504 would have survived the quantile cut
  and is a track). When a discriminant has no truth to calibrate against, the
  cheapest path is often to go get ten labels, not a better statistic.
- **Test the fallback, don't assume it.** The "sustained, not one bucket"
  argument sounds obviously right and is measurably wrong here: by absolute
  contiguous wide-run length the noise-driven long segments score 6–12 cm
  while genuine short showers median 3.0 cm, because scattered noise runs grow
  with segment length. A physically-worded criterion can still invert.

## 3. Event 321107 — mechanism fully explained; no safe fix found (two rounds)

Main segment (`real_cluster_id` 13000): 415 fit points, 248.72 cm, PCA angle
to drift-x **88.55°** — isochronous. `pdg=11`, `flag_shower=1` across all 415
points, bit-identical on `prodon`/`prodoff`.

**Round 2 (this pass)** was opened by the owner's reading of two event
displays: *the halo is an isochronous imaging artifact; the track is narrow in
the projective view; can the function be improved?* The mechanism is now
completely traced, and it is **not** what either the owner or round 1 thought.
The headline result is counterintuitive and is the most useful thing in this
section:

> **The function never saw the halo.** The wide cloud that makes the segment
> look like a shower by eye is *orthogonal to the only axis the test measures*
> and contributes nothing to the decision. The flag fired on drift-direction
> **quantization noise**.

A second, systematic result follows: 321107 is **not an outlier**. Across 429
events, 86 of 91 long shower-topology firings sit inside the same noise floor,
and 321107 ranks 77th of 91. This is a property of the discriminant at long
lengths, not a one-event accident — which is also why no threshold fixes it.

**Two rounds, two corrections to earlier passes of this section** — both
recorded in §3.6, both instances of the same trap: reasoning about the
algorithm from a re-derivation instead of measuring it in place.

### 3.1 Why an electron, not a muon — the owner's original guess, ruled out twice

`NeutrinoTrackShowerSep.cxx:52-59`: `segment_is_shower_topology` runs first
and short-circuits `segment_is_shower_trajectory`. Separately,
`segment_is_shower_trajectory` (`PRSegmentFunctions.cxx:983-1078`) has a hard
`length > 50cm → return false` guard at `:988` — this 249 cm track could never
reach it regardless of order.

**The isochronous projection handling the owner remembered is real** —
`PRSegmentFunctions.cxx:1050-1067`, the `angle_diff <= 10°` branch of
`is_shower_trajectory`, which deliberately projects lengths onto `dir_2` to
drop the drift-ambiguous axis — it just never ran here: wrong function
(topology fired first), and length-gated out of the right one anyway.

Once topology fires, `NeutrinoTrackShowerSep.cxx:117-135` hard-assigns
`pdg=11`/`score=100` as literals — `segment_determine_dir_track`, the real
muon/proton/pion PID call, is never attempted on this path.

### 3.2 The measured axis collapses onto the drift axis — so the halo is invisible

`PRSegmentFunctions.cxx:2574-2584` builds the local frame as

```cpp
dir_2 = drift_dir_abs.cross(dir_1).norm();   // no x-component: lies in the wire plane
dir_3 = dir_1.cross(dir_2);                  // => dir_3 . xhat = sin(angle-to-drift)
```

and the decision reads **only `std::get<2>`** — the RMS along `dir_3`.
Components 0 and 1 are computed at `:2606-2613` and never read again.

For an isochronous segment `sin θ → 1`, so `dir_3` *is* the drift axis. The
instrument confirms it directly on this segment: **`dir3x = 0.9994`**.

The halo lies in the orthogonal direction. Measured on the event's own 3-D
point cloud (all 8153 points of cluster 13, in the code's own frame):

| axis | rms |
|---|---|
| `dir_2` (in wire plane — where the halo lives) | **8.34 cm** |
| `dir_3` (drift — the *only* axis the test uses) | **0.40 cm** |

a ratio of **21**. And the halo does not leak into the measured axis at all —
rms(`dir_3`) as a function of transverse distance:

| \|dir_2\| band | 0–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–99 cm |
|---|---|---|---|---|---|---|
| n points | 1026 | 846 | 2021 | 2428 | 1654 | 178 |
| rms(`dir_3`) | 0.414 | 0.423 | 0.386 | 0.383 | 0.395 | 0.375 cm |

**Flat, from the core out to 24 cm.** Points 20 cm off the trajectory
contribute exactly as much drift-direction spread as points on it.

**Consequence — and this is the answer to the owner's question:** testing
narrowness in the 2-D projected views *would not have changed this verdict*.
The 3-D halo never entered the decision, so removing it cannot remove the
flag. The good news is that the true cause is simpler.

*(Provenance: the two tables above are computed offline from the event's own
Bee point cloud by replicating the code's frame construction — they
characterise the data, not the algorithm's verdict. The load-bearing claim
they support, that the measured axis is the drift axis, is independently
confirmed in-code by `dir3x = 0.9994`.)*

### 3.3 What actually fired: the 0.4 cm cut sits inside the drift lattice

The reconstructed x of this cluster takes **28 distinct values over 8.44 cm,
spaced 0.313 cm** — the time-slice thickness. The per-bucket RMS along a
drift-aligned axis is therefore a lattice artifact of order 0.3–0.5 cm, and
the code's "large spread" threshold is **0.4 cm**.

Direct instrument output for the target (`WCT_SHOWER_TOPO_DEBUG=1`):

```
seg -1 L 248.7cm assoc_npts 8133 nbuckets 389 n_over0.4cm 134 n_over0.7cm 4
       n_over0.8cm 3 n_over1.0cm 2 rms_p50 0.36cm rms_p75 0.43cm rms_p90 0.50cm
       rms_p95 0.55cm max_spread 1.24cm maxcont 9.6cm lsl 80.4cm tel 233.7cm
       lsl/tel 0.344 tel/L 0.940 dir3x 0.9994 branch 2
seg -1 guard branch 2 L 248.7cm total_length1 64.2cm(0.258)
       total_length2 63.6cm(0.256) demoted false final_shower true
```

Branch 2 is `max_spread > 0.8 && lsl/tel > 0.3 && tel >= 15cm`. Both live
terms are artifacts:

- **`lsl/tel = 0.344` vs a 0.3 threshold** — a coin flip. The bucket-RMS
  median is 0.36 cm against a 0.4 cm cut, so the count of "wide" buckets is
  set by lattice noise. Across all 91 long firings in the 429-event run the
  median bucket RMS is **0.35 cm** (range 0.33–0.61) — a universal constant of
  the reconstruction, not a property of any segment.
- **`max_spread = 1.24 cm` comes from 3 buckets out of 390** — indices
  131/132/135, adjacent, spanning ~2 cm of a 249 cm track, each collecting
  points from 5–6 different time slices. p90 of the same distribution is
  0.50 cm.

The existing long-track guard (`:2816-2825`, the designed backstop for exactly
this) missed by 0.008: `0.258` and `0.256` against its `0.25` threshold.

### 3.4 Population: 321107 is typical, not exceptional

Instrument run over the full 572-event valfast manifest (`work-pr25s3r2-dbgall`,
all `rc=0`; 429 events produce at least one evaluation; 4327 distinct segment
evaluations, 464 fired the disjunction — 91 with `L>50cm`, 373 with `L≤50cm`).

`rms_p90` of the 91 long firings, sorted:

```
0.42 ... 0.64   (86 values, a tight lump on the noise floor)
1.18  1.67  2.46  2.46  10.86    (5 genuinely wide segments)
```

321107 sits at 0.50 — **rank 77 of 91**. So the long shower-topology category
is, as a population, noise-driven: **86 of 91 (94.5%)** carry no spread
evidence above the quantization floor.

Physics-relevant blast radius (segments on the **nu-candidate main cluster**,
where a verdict change moves a physics answer): **10 of 263** long segments
across 10 events, i.e. ~2.3% of events — not the 91 raw firings.

### 3.5 A fix was designed, pre-committed, measured — and all three forms fail

Round 2's hypothesis: strengthen the existing demote-only `L>50cm` guard by
replacing the single-bucket `max_spread` extremum with a robust statistic
(the q-quantile of the bucket RMS, `q=1.0` = legacy = byte-identical), so the
"wide" evidence must be *sustained* rather than carried by 3 buckets in 390.
On the 21-event round-1 arm this looked compelling: a clean gap at
`rms_p90` 0.60 → 1.17 cm.

**The selection rule was committed before the wider run** (the high side of
that gap was n=3, from a population selected for containing long
shower-flagged segments — the same shape as round 1's own n=3→n=1
correction). The rule: adopt only if short (`L≤50cm`) firings — the population
where nobody disputes real showers live — cluster at `p90 ≳ 1 cm`; stop if
they scatter through 0.5–1.5 cm; and require the separation to survive at
p75, p90 *and* p95.

**All three conditions fail.**

**(a) The short population scatters exactly as the stop-condition describes.**
`rms_p90` of the 373 short firings: p10 **0.53**, p25 **0.67**, median
**0.80**, p75 **1.07**, p90 **1.44**. And **103 of 373 (27.6%)** sit below
0.7 cm — indistinguishable from the long lump. **20.6% sit below 0.64**, the
long population's own gap. If `p90 ≈ 0.5` meant "not a shower", then a fifth
of all short shower firings are also not showers — a claim there is no truth
on this sample to support.

**(b) The statistic is length-dependent, so the cut selects "long", not
"not-a-shower".** In matched bucket-count bands the median `p90` falls
monotonically — and so does the *scale-free* fraction of wide buckets, so this
is not merely the `p90 → max` small-sample effect:

| nbuckets | 0–20 | 20–40 | 40–80 | 80–160 | >160 |
|---|---|---|---|---|---|
| median `rms_p90` | 0.85 | 0.71 | 0.61 | 0.47 | 0.45 cm |
| median `n_over0.8/nbuckets` | 0.125 | 0.081 | 0.056 | 0.020 | 0.012 |

**(c) q-robustness fails.** The gap's position moves with the quantile —
0.47 (p75), 0.64 (p90), 1.08 (p95) — and so does the demotion count: **89, 86,
80 of 91**. A constant that must be chosen to place a cut is a tuned constant,
whatever statistic it wears.

**The two fallback forms fail too**, both tested rather than assumed:

- *Fraction of buckets over the branch cut* (`n_over0.8/nbuckets`, the literal
  "sustained, not one bucket"): the long values run continuously to 0.161; the
  short median is **0.103**, sitting inside the long range. Complete overlap.
- *Contiguity* (`maxcont`, the longest contiguous wide run — a quantity the
  function already computes and discards): in absolute cm the long noise-floor
  segments score **6–12 cm** while genuine short showers have median **3.0 cm**
  and p90 6.0 cm. **The argument inverts** — by this measure the noise-driven
  long segments look *more* shower-like than real showers, because scattered
  noise runs grow with segment length.

Per the pre-committed rule, **no fix is shipped.** Tuning any of these to
convict 321107 would be exactly the move CLAUDE.md §5.7 forbids.

### 3.6 Corrections to earlier passes of this section

**Round 1 → round 2, candidate B ("robustify `max_spread`") was rejected on
the wrong threshold.** Round 1 dismissed it because the spread was spread
"across 134/389 buckets". That is `n_over0.4`. Branch 2's `max_spread`
condition is `> 0.8`, where the count is **3/389** — it *is* a lone-outlier
condition. Round 1 read the right number against the wrong cut and drew the
opposite conclusion. Round 2 re-tested it properly (with `n_over0.7/0.8/1.0`
added to the instrument) and it still fails — but for the well-established
reason in §3.5, not the one round 1 gave.

**Round 0 → round 1**: the original write-up's "associated-cloud
contamination" and its `0.37` narrow-miss came from a 3-D nearest-fit-point
proxy that predicted the flag *should not* fire on the one event that did.
Withdrawn and replaced by the in-code instrument.

Both are the same failure: a re-derivation of the algorithm is not a
measurement of it. The instrument exists so this does not recur.

### 3.7 Prototype comparison — faithful port, no divergence

`prototype_base/pid/src/ProtoSegment.cxx:319-541` matches
`segment_is_shower_topology` line-for-line: identical frame construction
(same cross-product order, same `<7.5°` drift-*parallel* special case), the
same five disjunction clauses with all 12 constants, and the same
`>50cm / 0.25` long-track guard. WCP also computes and discards
`vec_dQ_dx`'s values, RMS components 0 and 1, and `max_cont_length` — the last
surviving there only in a commented-out print (`:538`). The behaviour
described above is therefore **original to the algorithm design, not a porting
bug** (M15 satisfied), and any change is a genuine algorithm addition
requiring its own default-OFF knob.

Two observations, reported not fixed (§5 tie-breaker): the in-tree
`max_cont_length` is only updated when a wide run *ends*, so a run reaching the
last bucket is never recorded; and `NeutrinoVertexFinder.cxx:2287/2327/2397`
calls this function independently of `NeutrinoTrackShowerSep.cxx:53`, so any
future knob must be threaded to **both** or the same segment is classified two
ways within one event.

### 3.8 Conclusion and the open question for the owner

**Shipped this round**: the extended `WCT_SHOWER_TOPO_DEBUG` instrument
(quantiles p50/p75/p90/p95, bucket counts at every branch threshold 0.4/0.7/
0.8/1.0, `maxcont`, and `dir3x = |dir_3·x̂|`), default off, log-only, zero
behavior change; and `pr25_spread_census.py`.

**Not shipped**: any behavior change. Two rounds have now failed to find a
threshold that separates these segments, and §3.4 explains why — there is
nothing to separate. The evidence that fires the flag on 86 of 91 long
segments is the same quantization floor in every one of them.

**The real question is therefore not a threshold, and it is the owner's to
answer**: *should a segment of this length be eligible for `kShowerTopology`
at all?* An EM shower in liquid argon has X₀ ≈ 14 cm; a 249 cm shower does not
exist, and the flag's own backstop already encodes that intuition with a
`>50cm` scope. Making that scope decisive rather than conditional is a
one-line, physically-motivated change — but it is an unconditional behavior
change to a shared production path, so it needs (a) the owner's decision and
(b) hand-scan truth on the **10 nu-main cases**, which is a tractable
scan and would make the cut derivable instead of tuned. Recommended as the
next round; not taken unilaterally (escalation rules 1 and 7).

A second untouched lever remains: `vec_dQ_dx` (`:2616`) is dead code, and a
shower's dQ/dx profile differs physically from a muon's in a way none of the
spread statistics capture. Also a genuine algorithm addition, also needing its
own measurement round.

### 3.9 The 10 nu-main cases — what the owner would actually be deciding

These are the segments where a verdict change moves a physics answer. All are
on the selected nu-candidate main cluster, all `L>50cm`, all currently carrying
`flag_shower`. Scores are from `work-pr25s3r2-dbgall` (`pr_scores_table.py`);
`pdg` is of the shower-flagged segment(s) on that cluster.

| evt | seg L (cm) | angle to drift | numu_score | kine_reco_Enu (MeV) | pdg now | cosmict |
|---|---|---|---|---|---|---|
| 286353 | 271.3 | 71.3° | 2.023 | 624.9 | 11 | 0 |
| **321107** | **248.7** | **88.5°** | **−0.783** | **550.1** | **11** | **1** |
| 284013 | 149.6 | 48.9° | 0.880 | 394.1 | 11 | 0 |
| 277276 | 140.4 | 76.6° | 1.291 | 1566.3 | 11, 2212 | 0 |
| 57903 | 69.9 | 87.5° | 1.070 | 391.2 | 11 | 0 |
| 280972 | 68.5 | 71.2° | 3.409 | 3088.2 | 11, 211 | 0 |
| 315167 | 62.9 | 64.0° | 2.069 | 1354.5 | 11 | 0 |
| 278684 | 59.2 | 38.9° | −0.485 | 680.0 | 11 | 1 |
| 287621 | 54.0 | 75.7° | 1.785 | 716.0 | 13 | 0 |
| 400504 | 50.4 | 84.2° | 1.190 | 508.5 | 11 | 0 |

Notes for a hand scan: **321107 is the only one with a negative `numu_score`
*and* a nearly-perpendicular angle** — the combination the owner flagged by
eye. `nue_score` is the −4.300936 / −15.0 sentinel on 10/10 of these, so it
carries no information here (see GOTCHAS). 287621 already reads `pdg=13` on
its flagged segment, so it is a partial case, not a clean muon-as-electron.
A scan of these 10 is what would turn §3.8's question from an investigation
into a decision.

### 3.10 FIX SHIPPED (default OFF) — `shower_topo_demote_len`

The owner scanned the §3.9 Bee set on 2026-08-03 and returned a verdict:
**"none of these are actually shower, they all look like tracks" — 10/10.**
That is the truth §3.8 asked for, and it makes the cut derivable rather than
tuned. It also **refutes round 2's own candidate**: evt 400504 sits in the
"wide" class (`rms_p90` 0.90 > 0.8), so the quantile fix of §3.5 would have
*kept* it flagged — and it is a track. The discriminant was not merely
unsupported; on the one scanned case where it would have changed the answer,
it was wrong.

**The fix is a length rule, not a spread rule**, and it reuses the existing
demote-only guard rather than adding a statistic:

```cpp
// PRSegmentFunctions.cxx, inside `if (flag_shower_topology)`
if (demote_len > 0 && tmp_total_length > demote_len) {
    flag_dir = 0; flag_shower_topology = false;
}
```

Knob `shower_topo_demote_len` (cm, **C++ default 0 = off = byte-identical**),
threaded `TaggerCheckNeutrino` → `PatternAlgorithms` → **all four**
`segment_is_shower_topology` call sites (`NeutrinoTrackShowerSep.cxx:53` and
`NeutrinoVertexFinder.cxx:2287/2327/2397`), so one segment cannot be
classified two ways within an event. Runner escape:
`SBND_SHOWER_TOPO_DEMOTE_LEN`. A demoted long segment then fails the
trajectory test (`>50cm`) and lands in the Track branch with full PID (§1f).

**SBND DEFAULT ON at 50 cm (owner decision, 2026-08-03)** —
`wct-pr-perevt.jsonnet` ships `shower_topo_demote_len = 50`. Verified by a
bare `run_pr_chain_batch.sh` run with no env override: evt 321107's main
cluster reads `pdg 13`, mean `flag_shower` 0.00 (pre-flip: `pdg 11`, 1.00).

**Threshold.** The owner accepted 9/10 for this round, so the SBND operating
point is **50 cm**. Evt 400504 is the tenth: it measures **49.1 cm** by
`segment_track_length(seg,0)` — the length the guard uses — while the census
script's fit-point path length reads 50.4 cm. ~45 cm would cover all ten.
The two length measures are now documented in `pr25_shower_topo_census.py`;
when a decision depends on the length, read the instrument's `L`.

#### Measured effect (`work-pr25s3r2-dbgall` OFF vs `work-pr25s3r2-on50b` ON, 572 events, all `rc=0`)

| | OFF (legacy) | ON (50 cm) |
|---|---|---|
| guard evaluations, `L>50cm` | 91 | 94 |
| …still `kShowerTopology` after the guard | **24** | **0** |
| …demoted | 67 | 94 |

- **23 long segments flip shower → track across 22 events; 0 flip the other
  way** — demote-only confirmed empirically, not just by inspection.
- **17 of 572 events (3.0%) move `numu_score` at all.** Contained.
- The target: **321107 `pdg 11 → 13`, `flag_shower 1.00 → 0.00`,
  `numu_score −0.783 → +0.317`** — it crosses zero in the direction a muon
  candidate should.

Per-event, the ten owner-scanned cases (9 change; 400504 is the 49.1 cm one):

| evt | pdg OFF → ON | flag_shower | numu_score OFF → ON | kine_Enu OFF → ON |
|---|---|---|---|---|
| **321107** | **11 → 13** | 1.00 → 0.00 | **−0.783 → +0.317** | 550 → 680 |
| 286353 | 11 → 11,13,211 | 0.99 → 0.05 | 2.023 → **−1.139** | 625 → 908 |
| 284013 | 11 → 11,13 | 1.00 → 0.05 | 0.880 → 2.495 | 394 → 520 |
| 277276 | 11,2212 → +13,211 | 0.90 → 0.09 | 1.291 → 2.437 | 1566 → 801 |
| 57903 | 11,13 → 211,2212 | 0.66 → 0.00 | 1.070 → 0.456 | 391 → 632 |
| 280972 | 11,13,211 (same) | 0.29 → 0.19 | 3.409 → 3.353 | 3088 → 3071 |
| 315167 | 11,211,2212 (same) | 0.42 → 0.22 | 2.069 → 2.596 | 1355 → 1355 |
| 278684 | 11 → 11,13 | 0.42 → 0.10 | −0.485 → −0.850 | 680 → 504 |
| 287621 | 13,2212 (same) | 0.40 → 0.00 | 1.785 → 2.931 | 716 → 728 |
| 400504 | unchanged | 0.48 | 1.190 (unchanged) | 508 (unchanged) |

**Reported honestly: the score moves are mixed, not uniformly good.** Five go
up, three down, one flat. **286353 moves 2.023 → −1.139**, crossing zero
*downward* on an event the owner called a track — the largest single move in
the set and the opposite of the intended direction. There is no truth on
`numu` selection for mcp1k, so this is a flag for the owner's judgement, not
something to tune away (§5.7). It is the one result that argues for scanning
the knob-on outputs before any default flip.

### Verification performed

- `./wcb build --notests -p && ./wcb install --notests -p`, both `rc=0`;
  freshness proof — `local/lib/libWireCellClus.so` 20:39:29 newer than
  `clus/src/PRSegmentFunctions.cxx` 20:38:59 (M1).
- `./build/clus/wcdoctest-clus`: 49/49 cases, 565/565 assertions, 0 failed.
- The instrument is a pure log addition behind `std::getenv`, read once into a
  function-local `static const bool`; the one value it records inside the hot
  loop (`dbg_dir3x`) is itself guarded by that flag. No control-flow change on
  any path, no new jsonnet key, nothing threaded to config — so no
  compiled-config proof and no byte-identical gate apply (there is no
  behavior-affecting change in this section, by design).
- Population run: full 572-event valfast manifest, `PR_JOBS=6`, **all 572
  `rc=0`**; 429 events yielded at least one `segment_is_shower_topology`
  evaluation (the rest have no segment reaching it with a non-empty
  `associate_points` cloud).
- **Cross-arm stability of §3.4's load-bearing numbers.** The round-1 arm
  (`work-pr25s3-dbg21`) was run with the `data` reality TLA and round 2 with
  `sim`, so the 30 events common to both were compared directly. The
  population picture is the same — 31 vs 28 long firings, median `rms_p90`
  0.48 vs 0.47, 28 vs 25 below 0.7 cm, and the *same* three wide segments at
  1.17/1.18, 1.69/1.67, 10.86/10.86. What differs: 5 of 30 events gain or lose
  one **marginal** long firing, and segment lengths shift by ~1 cm
  (e.g. 201.4 → 202.9 cm), moving bucket counts slightly. This does not
  threaten the conclusion — it *corroborates* it: the segments that flip are
  exactly the near-threshold ones §3.3 says are decided by noise. It does mean
  individual counts should always be quoted with their arm, as done here.

**Round 3 (the fix) verification:**

- `./wcb build --notests -p && ./wcb install --notests -p`, both `rc=0`;
  freshness proof — `libWireCellClus.so` 21:21:22 newer than the edits
  (21:19:05 / 21:19:59) (M1).
- `./build/clus/wcdoctest-clus`: 49/49 cases, 565/565 assertions.
- **Compiled-config proof with the PRODUCTION `pipeline_names`** (the empty
  default never instantiates the tagger and would pass vacuously — the §2
  near-miss): `TaggerCheckNeutrino` present (6 nodes); key **absent** with the
  knob off, `"shower_topo_demote_len" : 50` with it on; and the compiled JSON
  with the knob off is **byte-identical to the pre-change config**.
- **Byte-identical output gate PASS**: 20 events, `mabc-pr.zip` +
  `pctree-pr-evt*.tar.gz`, **40/40 archive members identical, 0 differ**, new
  binary knob-off (`work-pr25s3r2-gateoff`) vs the pre-change binary
  (`work-pr25s3r2-dbgall`), via `abtest/hash_archive.py` member content (M2).
  *Trap hit and corrected*: the first pass compared `hash_archive.py`'s whole
  output line, which ends in the file path — 40/40 false DIFFs. Compare field 1.
- **Knob engagement proven on a real run, not just in `wcsonnet`**: a segment
  at `L 202.9cm` with guard fractions 0.296/0.287 — both *above* the legacy
  0.25 — reads `demoted false / final_shower true` OFF and
  `demoted true / final_shower false` ON.
- **Determinism**: 3 runs of 321107 knob-on under the runner's
  `setarch x86_64 -R`, all `rc=0` and byte-identical (`mabc-pr.zip`
  `6228d5f0…`, `pctree` `28bf7e8d…`) (M4). The `pctree` hash equals the
  knob-off arm's — the knob changes PID/flags, not the point cloud.
- Knob-on population: `work-pr25s3r2-on50b`, full 572 events, `PR_JOBS=24`
  (owner direction; 23–24 concurrent `wire-cell`, loadavg 27–44 of 64 cores,
  well inside M5), **all 572 `rc=0`**.
- **Not bit-identical when ON**, by design. SBND default ships OFF; the flip
  is the owner's call (escalation rule 1).

*(`work-pr25s3r2-on50` is an abandoned partial 6-job arm, killed mid-write when
the run was relaunched at 24 jobs; superseded by `-on50b` and unused. Left in
place rather than deleted, per M13.)*

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# round 2: instrument over the full valfast manifest, then the decision tables
WCT_SHOWER_TOPO_DEBUG=1 PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25s3r2-dbgall sim
python3 pr25_spread_census.py --arm work-pr25s3r2-dbgall

# physics-relevant blast radius (nu-candidate main clusters only)
python3 pr25_shower_topo_census.py population --arm work-pr25s3r2-dbgall --nu-only

# the target's own instrument lines
grep "shower_topo dbg" work-pr25s3r2-dbgall/pr_evt321107/wct_pr_evt321107.log

# round 1 arms (21-event blast radius + controls), kept for provenance
python3 pr25_shower_topo_census.py guard --arm work-pr25s3-dbg21

# ---- round 3: the fix ----
# knob-off byte-identical gate (new binary vs the pre-change arm)
WCT_SHOWER_TOPO_DEBUG=1 PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25s3r2-gateoff sim \
  321107 286353 284013 277276 57903 280972 315167 278684 287621 400504 \
  48367 53427 62281 65295 68648 73004 166650 167200 172656 173234
# compare FIELD 1 of hash_archive.py -- the rest of the line is the file path
python3 ../../abtest/hash_archive.py <arm>/pr_evt<ID>/mabc-pr.zip | awk '{print $1}'

# knob-on population (572 events, 24 jobs)
WCT_SHOWER_TOPO_DEBUG=1 PR_JOBS=24 SBND_SHOWER_TOPO_DEMOTE_LEN=50 \
  ./run_pr_chain_batch.sh work-mcp1kall-vfprodoff work-pr25s3r2-on50b sim
python3 pr_scores_table.py --root work-pr25s3r2-on50b --out /tmp/scores_on.tsv

# compiled-config proof -- MUST pass the production pipeline_names, or the
# tagger node is never built and the diff is vacuously empty
wcsonnet --tla-str input=x --tla-code 'anode_indices=[0,1]' --tla-str output_dir=/tmp \
  --tla-code run=18255 --tla-code subrun=1 --tla-code event=321107 --tla-str reality=sim \
  --tla-code "pipeline_names=['switch_scope',...,'tagger_output']" --tla-str save_tensors=y \
  [-A shower_topo_demote_len=50] cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
```

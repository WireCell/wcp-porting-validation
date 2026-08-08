# doc pr/50 — 172230-class near-vertex robustness: partition-stage defer + main-vertex kink snap

## Symptom

Owner hand-scan of the pr/49 production flip (`fit_blob_coverage = 0`,
toolkit `758e2e4d`) found near-vertex regressions in six events:

- 18255-172230, 18259-131357, 18255-268067, 18269-342199, 18255-360535
  ("regressing near vertex")
- 18255-469665 ("ISO case near vertex", same broad class)

Signature (172230 screenshots, 2026-08-08): the old result turns crisply at
the vertex with healthy dQ/dx; the new result rounds/cuts the corner — the
fitted trajectory "jumps over the vertex" — with low fitted dQ/dx.

Owner framing: the origin is that the pattern recognition is not robust —
the observation is triggered by a small distortion on the fitted trajectory
far away.  The fix must make the PR robust.

## Root cause (measured per event)

The pr/49 deweighting is per-cell correct, but `find_proto_vertex`'s
recursive break partition is **globally sensitive to any fit perturbation**:
`break_segments` refits after every accepted break and re-reads `fits()`,
first-match-wins kink accepts, LIFO recursion, a cross-iteration direction
memo, and a shared break-index memo mean a sub-mm fit change anywhere can
flip one break and re-derive the whole partition.  The proto-vertex at the
TRUE image kink then no longer exists in the graph; `determine_main_vertex`
(and the DL rerank) can only choose surviving candidates; `improve_vertex`
(MyFCN, 0.43 cm soft prior on the old position, 1.5 cm inner protect
radius) is structurally local and cannot recover.

Per-event attribution (off48d / on48c round-2 / on48d round-3 arms +
extended-sentinel diagnostics: nearest-pattern-vertex distance and degree at
fire time, claimant id:3D-distance:npoints appended to the deweight
sentinel, toolkit this round):

| evt | vertex shift (on48d vs off48d) | deweight geography | mechanism |
|---|---|---|---|
| 172230 | 2.696 cm | 200 firings, all 85-90 cm away (claimants 12-44 pts at 3-22 cm) | far-perturbation → partition reshuffle (34→33 segs round 1), true-kink candidate lost |
| 131357 | 1.83 cm | 688 firings 22-61 cm away (dominant claimant 5395 pts at 67-85 cm) | same class; partition 30→26 segs |
| 342199 | 1.18 cm (round 3 ONLY; round 2 identical to off) | 245 firings, 29% < 5 cm from vertex (near claimants 38-194 pts at 2-11 cm re-admitted by the round-3 far-gate drop) | near-vertex deweighting during round-1 PR |
| 360535 | 0 (vertex unchanged) | 952 firings ≥ 49.7 cm | far-perturbation → structure churn only (37→39 segs, new 1.7/1.2 cm stubs) |
| 469665 | 98.5 cm walk (already in round 2) | 47 firings, 96% at 1.5-4.7 cm from the ON vertex, claimant 1755 pts at 34-38 cm (3D-far, projecting onto near-vertex cells) | ISO-case; near-vertex deweighting by far claimant — no claimant-distance gate can exempt it |
| 268067 | 0 — off48d/on48d bee-identical, vertex-identical; only tsv diff is the known stmfit log-tearing artifact | 37 firings at 6.7-8.8 cm, zero downstream effect | EXCLUDED: owner confirmed the scan used an older (pre-pr/49) baseline; separate item |

**No input-level gate separates harmful from helpful deweights** (all
measured): claimant size fails (131357's harmful claimant 5395 pts/67-85 cm
has the same character as 57441's beneficial 3947 pts/156-165 cm); claimant
distance fails (469665 needs near-vertex protection from a 3D-far
claimant, which is exactly the 57441 ghost topology); pattern-vertex
proximity fails (105/169 of 57441's beneficial firings are within 5 cm of a
break vertex); own-coverage-at-looser-tolerance fails (161/169 of 57441's
beneficial deweights are also tiling-edge).  Every harmful effect flows
through one door: the perturbation entering `find_proto_vertex`.  The fixes
are therefore outcome-level, not input-level.

## Why it hid

The pr/49 census metrics could not see this failure mode: fit-vs-image
point distances barely move (172230 cid 5: mean 0.352→0.354 on 7638 image
points; the trajectory stays ON the dense image — it takes the wrong
BRANCH).  It is a topology error, visible to a human scan and to a
kink-vs-vertex displacement check, neither of which was in the gates.
All movers were "sentinel-gated" — the sentinel proves the knob caused the
diff, not that the diff is an improvement.

## Fix — two independent default-OFF knobs

### 1. `fit_blob_coverage_defer` (TaggerCheckNeutrino, bool, default false)

Suspend the deweighting while `find_proto_vertex` forms the **main
cluster's** break partition; restore it for every later fitting stage
(clustering_points onward, main-vertex determination, improve_vertex, the
final trajectory + dQ/dx).  The partition then forms on legacy fits — the
172230-class reshuffle is impossible **by construction** — while the ghost
protection keeps acting where it was validated (the final trajectory).

Main cluster ONLY, measured reason: a non-main cluster's final trajectory
is essentially its find_proto_vertex fit (no later full refit), so a global
defer un-fixes the pr/49 other-cluster ghosts (57441 cid 20 measured
1.12 → 1.23 cm under a global defer — same value as under main-only,
because 57441's cid 20 IS its main cluster; see the known cost below).

Implementation: `cov_defer_suspend()/cov_defer_restore()` lambdas around
the `:880` main-cluster `find_proto_vertex` call in TaggerCheckNeutrino's
visit(); local fitters spawned inside (`inherit_from`) copy the suspended
value.  No-ops unless `fit_blob_coverage >= 0` AND the defer knob is on.

**Known cost (the one trade)**: 57441's main-cluster projection-ghost
detour — the pr/49 flagship — reverts to ~legacy (fit-vs-image
1.12 → 1.23 cm vs the pr/49-fixed 0.45 cm): its detour is baked into the
partition (a break vertex lands in the detour; endpoints are deweight-exempt
by design), so the restored late-stage deweighting cannot pull it out.
Every OTHER pr/49 tagged fit-vs-image effect on the 48-sample is preserved
EXACTLY (all tagged rows byte-equal between the defer arm and the full-on
arm — they are other-cluster effects; e.g. 342199 cid 20 1.57 → 0.99 held).

### 2. `vertex_kink_snap` (PatternAlgorithms pass, bool, default false)

A main-vertex kink-consistency pass in the previously-empty pipeline window
after `determine_overall_main_vertex[_DL]` and before the final
`improve_vertex`:

- Scan the **wcpt paths** (steiner/image-anchored; TrackFitting never
  rewrites wcpts, so they are immune to exactly the fit distortion that
  causes the loss) of the segments incident to the final main vertex, for
  ALL interior turns ≥ `vks_angle` (25°) within `vks_radius` (5 cm), via
  `path_scan_vertex_kink` (wide-baseline PCA turn, chord fallback for the
  short-stub side; the strongest turn is deliberately NOT auto-selected —
  in 172230 a secondary wiggle at 4.9 cm out-turns the true corner at
  2.4 cm by 3°).
- Guards per candidate: never fire on a kProtectedBreak vertex (pr/48 TEB
  junctions; DL rerank protects them too); vertex degree ≥ 2;
  **pass-through** — the stub V→K anti-parallel within `vks_collinear`
  (30°; 172230 measures 23.5° — near-corner arms curve) of another
  incident arm; **strength margin** — turn ≥ bendV + `vks_margin` (10°);
  **fit-miss** — the fitted trajectory must MISS the image corner by ≥
  `vks_fit_miss` (0.35 cm): a genuinely modeled kink has fit points on it
  (172230's rounding fit misses its corner candidates by 0.45-0.77 cm and
  the guard declines the one point the fit does visit at 0.21 cm).  An
  optional Bragg-hot veto exists but defaults OFF — it MISFIRES on the
  failure class itself (with the vertex misplaced onto the muon, the arm
  near V reads the proton's charge, 1.91× MIP in 172230; local dQ/dx
  cannot tell a real hot junction from misassigned charge).
- Selection among survivors: smallest bendV (best pass-through evidence),
  then larger turn, then smaller arc.  One snap per event, no recursion.
- Action: `break_segment` at the corner wcpt, `vtx_new->cluster(&cluster)`
  (break_segment does not set it — the break_two_end_dqdx gotcha),
  `kProtectedBreak`, re-seat `main_vertex`, one
  `do_multi_tracking(true,true,true,m_fit_exclusion,false,&cluster)` so
  `improve_vertex` polishes a corner-anchored trajectory.

Toolkit-only; no prototype counterpart (verified: prototype `search_kink`
is only used inside `break_segments`; the nearest prototype machinery,
`examine_structure_final_2/_3`, merges at 2.0/2.5 cm — cannot reach
2.7 cm; precedent for toolkit-only vertex logic: doc pr/44 long-muon vote).

**Measured scope**: the snap fixes the jump-over class — 172230's final
vertex lands 2.6 mm from the correct position (from 2.70 cm off), with
`improve_vertex` doing the last centimeter from the snapped corner.  It
correctly does NOT fire on: 131357 (arms smooth; the old vertex is not on
any incident image path — the partition re-routed the geometry), 342199
(wide-V topology, bendV 109-131°), 360535 (vertex not displaced), 469665
(12 cm walk, outside the design envelope), 57441 (no candidate).

### What fixes what (the owner's decision table)

| evt | snap only | defer only | both |
|---|---|---|---|
| 172230 | vertex to 2.6 mm | exact revert | exact revert (snap correctly silent) |
| 131357 | no change | exact revert | exact revert |
| 342199 | no change | 2.2 mm from revert | snap fires on the restored graph and moves the vertex 1.2 cm from the old position — interference |
| 360535 | no change | exact revert | exact revert |
| 469665 | no change | exact revert | exact revert |
| 57441 (pr/49 flagship) | intact (0.45 cm) | **reverts to 1.23 cm** | reverts to 1.23 cm |
| other pr/49 ghost gains (48-sample tagged rows) | intact | intact (byte-equal to full-on) | intact |

## Verification

- Compiled-config proofs: bare compile (both knobs default) byte-identical
  to pre-pr/50 HEAD with the production pipeline TLAs; each knob-on compile
  differs in exactly its key(s).
- `wcdoctest-clus` 1234/1234 (new: doctest_vertex_kink_snap.cxx synthetic
  path cases; knob-default pins incl. vks_* and the defer bool).
- Off-gates (both knobs off, new binary): work-pr50-off48a vs
  work-pr49-on48d — 0/48 archive movers (member-hash), nusel-events 0/48,
  nusel-table 0/48; work-pr50-off50a vs work-pr49-on50d — 0/50, nusel 0/50
  both granularities.  Gate outputs: /home/xqian/tmp/pr50/gate_off{48a,50a}.txt.
- Defer census: work-pr50-defer48a vs work-pr49-on48d — 31/48 archive
  movers (the main-cluster reversion), nusel-events 0/48, nusel-table 0/48;
  every tagged fit-vs-image row identical to the full-on arm.
  work-pr50-defer50b vs work-pr49-on50d — 17/50 archive movers, all
  deweight-sentinel-gated (the one no-announce event, 52613, is
  "no main cluster selected" so the defer wrap is never reached);
  nusel-events 0/50, nusel-table 0/50.  ghost_case_exam fit-vs-image over
  the movers: 3 improved (52085 0.80→0.53, 56243 0.62→0.56,
  57661 2.37→2.30 cm), 3 worse (56463 1.09→1.42, 57441 0.45→1.23 —
  the flagship cost, matching the probe exactly — 59247 0.89→1.17 cm),
  the rest sub-mm jitter.
- Snap census: work-pr50-snap48a vs work-pr49-on48d — 3/48 movers
  (163543, 172230, 422851); work-pr50-snap50a vs work-pr49-on50d — 1/50
  (55539).  The SNAP sentinel fired in exactly those four events
  sample-wide and nowhere else; nusel 0/48 + 0/50 both granularities.
  The four firings:
  - 163543: cluster 14, turn 38.7°, arc 1.48 cm, bendV 22.9°
  - 172230: cluster 5, old=(-55.54,-87.92,21.97) new=(-54.29,-87.58,19.57),
    turn 43.4°, arc 3.37 cm, bendV 9.3° (the target recovery)
  - 422851: cluster 1, turn 49.7°, arc 1.91 cm, bendV 10.6°
  - 55539: cluster 18, turn 33.4°, arc 4.46 cm, bendV 20.7°
- VOID label: work-pr50-defer50a — accidentally launched over the full
  1000-event mcp sample and then corrupted mid-run by an edit to the
  running batch script (unexpected-EOF at line 874).  Superseded by
  work-pr50-defer50b; not cited anywhere; owner may delete.

## Bee hand-scan sets

Six uploads, identical event order within each sample (Bee indexes by
directory order; the `.index.txt` maps are kept next to this doc).  Built
with `scripts/bee/make_pr_bee.py` from the arms above.

48-sample (7 events, order: 172230 131357 342199 360535 469665 163543
422851 — the five targets, then the two other snap movers):

- before (pr/49 production, work-pr49-on48d):
  https://www.phy.bnl.gov/twister/bee/set/a4431c68-d8a0-4ce8-ae5e-9c6e22f76edf/event/list/
- defer (work-pr50-defer48a):
  https://www.phy.bnl.gov/twister/bee/set/d543f67b-d8df-43e4-903e-c07afded9a18/event/list/
- snap (work-pr50-snap48a):
  https://www.phy.bnl.gov/twister/bee/set/8f7da40b-c240-4424-895b-adf423d410d9/event/list/

50-sample (18 events, order: 49951 50303 51051 51513 52085 54351 55539
55627 56243 56463 57441 57661 58321 59025 59093 59179 59247 59685 — the 17
defer movers + snap mover 55539; 57441 is Bee index 11):

- before (work-pr49-on50d):
  https://www.phy.bnl.gov/twister/bee/set/e85cf49d-a888-477c-8f47-74b0aac0decd/event/list/
- defer (work-pr50-defer50b):
  https://www.phy.bnl.gov/twister/bee/set/4b01921b-3982-46c0-a4f3-b0af0c390e71/event/list/
- snap (work-pr50-snap50a):
  https://www.phy.bnl.gov/twister/bee/set/a40bb0a7-a527-4763-b7ae-37a66a1bc978/event/list/

Scan guidance: in the 48 set, defer should restore all five targets to the
pre-pr/49 near-vertex behavior, snap should fix only 172230 (plus the two
new corner-anchored vertices 163543/422851 to judge on their merits).  In
the 50 set the question is whether the 17 defer reversions — especially
57441 (index 11, the pr/49 flagship, fit-vs-image 0.45→1.23 cm) — are an
acceptable price.

## Repro

```
# toolkit 36463a81 (knobs + tests + cfg), wcp-porting-img: this doc's commit
cd wcp-porting-img/sbnd/sbnd_xin
# off-gates (both knobs default off -> byte-identical to pr/49 production)
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr50-off48a data
PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1k-cb0805  work-pr50-off50a data <50-evt list>
# defer arm
SBND_FIT_BLOB_COVERAGE_DEFER=true PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr50-defer48a data
# snap arm
SBND_VERTEX_KINK_SNAP=true PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr50-snap48a data
# per-event probes and censuses
python3 scripts/analysis/pr49/on_compare.py work-pr49-on48d work-pr50-defer48a
python3 scripts/analysis/pr49/ghost_case_exam.py <base> <new>
```

## Operating point

Both knobs C++ default OFF; SBND cfg TLAs default false ⇒ bare production
run remains byte-identical to the pr/49 record (off-gates above).  The flip
choice (defer / snap / both) is the owner's, informed by the decision table
and the Bee hand-scan links.

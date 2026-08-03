# doc pr/25 — cathode re-join direction-agreement fallback (ClusteringProtectBundle)

Also documents the other two findings from the same Bee hand-scan round
(320029, 321107) since they were investigated together; only this one has a
shipped fix — see the status table below.

| evt | question | status |
|---|---|---|
| **489327** | cathode crosser broken in two; the pr/23 re-join knob did not fire | **root cause proven, fix designed + validated on a 13-event small group** (§1). C++ + jsonnet **uncommitted**, pending owner go-ahead. |
| **320029** | another cluster (30) looks like a TGM but was not tagged | root cause proven: a structural TGM/demoted-main veto interaction, not a tuning question (§2). **Fix designed, not built** — needs its own small-group measurement round. |
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

## 2. Event 320029 — designed, not built

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

### Fix sketch (not implemented)

A scoped exemption in `main_pair_rejects`, gated by a new default-OFF knob
(`exempt_demoted_main_pairs`), mirroring `m_evaluate_demoted_mains` next to
it — skip the veto when `cluster.get_flag(Flags::demoted_main)`. Do **not**
touch `real_cluster_main` itself (read elsewhere; global re-stamp is a much
bigger blast radius than a scoped exemption).

**Cluster 30's outcome once the veto is lifted is not predicted here** —
downstream CASE-A (`TaggerCheckTGM.cxx:875-947`) forks on
`out_vec_wcps.size()`, not yet measured for this cluster: 2 groups tags
unconditionally (no length test); more groups applies a `0.45 * length_limit`
floor a 37 cm chord would likely fail. This is exactly the kind of thing the
small-group validation run would measure, not something to assert in advance.

### Next round's small group (not yet run)

Positive/target: cluster 30 (expect it fires or is absorbed by the length
floor — TBD) and cluster 29 (1.0 cm, expect it stays untagged regardless).
Comparison: the 14 demoted-main STM-only convictions from the standing
pr/20 Part I mcp1k census — re-run with the veto lifted, check whether TGM
newly convicts and whether P4 (`skip_cosmic_companions`, already ON)
correctly excludes the charge.

**Escalation rule 1**: changes cosmic verdicts — owner sign-off needed before
implementation, same bar as flipping the §1 operating point.

---

## GOTCHAS

- **Doc number pr/25, not pr/24.** A concurrent session in this shared tree
  had already reserved pr/24 (uncommitted `run_pr_chain_batch.sh` comment
  referencing DL/SCN neutrino-vertex weight attribution arms) — checked via
  `git status` before writing, per the concurrent-sessions convention. If a
  reader finds no `pr/24_*.md`, it belongs to that other round, not a gap
  here.
- **`run_pr_evt.sh` and `run_pr_chain_batch.sh` carry a concurrent session's
  edits in the same file diff** (`SBND_DL_WEIGHTS` passthrough + a DL
  main-cluster swap guard, unrelated to this doc). This doc's knob wiring
  (`SBND_PROTECT_REJOIN_PERP/_ANGLE/_DIR_RADIUS/_DIR_NPTS`) is additive and
  does not conflict, but committing either file right now would also commit
  the other session's uncommitted work — left uncommitted for that reason,
  on top of "commit only when asked."
- **`git stash` on a shared tree sweeps in every other session's uncommitted
  changes, not just yours.** Used once here to get a pre-change compiled-config
  baseline; `stash pop` restored cleanly (verified: the 3 pre-existing stash
  entries were untouched, the concurrent session's 4-file diff was intact
  after the pop) but it bumped `ClusteringProtectBundle.cxx`'s mtime with no
  content change, which read as a stale freshness proof until traced (see
  Verification). Prefer `git show HEAD:path > /home/xqian/tmp/baseline` per
  file for a future baseline compare — it never touches the working tree.

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

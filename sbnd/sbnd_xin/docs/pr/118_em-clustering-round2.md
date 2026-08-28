# doc pr/118 — EM clustering round 2: P2 body-distance admission + charge-continuity merge

**Status:** COMPLETE.  Probes shipped byte-neutral; both knob families coded
DEFAULT OFF; thresholds pinned by the §4 measurement;
`shower_merge_relax_continuity` **SBND PRODUCTION ON 2026-08-28** (owner
pre-authorization, §6 validation all green); `shower_ex1_conn3_body_dis`
ships OFF/not selected (measured zero yield).  Bee A/B pair for the three
movers uploaded for owner review (§6).

Owner directives for this round (2026-08-28): *instrument printouts first* to
understand the residual failures at the decision sites, then design the fix —
pr/91 P2 distance-to-charge admission and a charge-continuity discriminator
for the sub-5 cm stubs — with *novelty allowed* ("no need to limit by the
existing algorithm in the code now").  Validation bar same as pr/117;
production flip pre-authorized if validation passes; Bee A/B links to the
owner for any doubtful event.

## 0. Repro

```bash
# probes + knobs (toolkit, apply-pointcloud):
cd /home/xqian/toolkit-dev && direnv exec . bash -c \
  'cd toolkit && ./wcb build --notests -p && ./wcb install --notests -p'
direnv exec . toolkit/build/clus/wcdoctest-clus     # 2434/2434

# arms (98 events, 4 samples; ~5 min each at PR_JOBS=24):
#   probe0: probe binary, no env      -> byte-identity gate vs pr/117 flipchk
#   dbg:    Phase-B binary, WCT_SHOWER_{CONTENT,ABSORB,MERGE}_DEBUG=1, knobs off
#           -> Phase-B off-gate + the measurement run
/home/xqian/tmp/pr118_arms.sh probe0 0
/home/xqian/tmp/pr118_arms.sh dbg 1

# gates:
python3 scripts/pr85_hash_gate.py work-pr117r1-flipchk-<s> work-pr118r1-<arm>-<s>
cmp work-pr117r1-flipchk-<s>/pr_evt<E>/nusel-evt<E>.tsv work-pr118r1-<arm>-<s>/...

# measurement census (sec 4):
./em_display/prep_pr117.py --tag 118dbg work-pr118r1-dbg-{mcp1k,mcp2k,ncpi0,nuecc48}
./scripts/pr118_probe_census.py --prepdir em_display/emprep-118dbg \
  --pairs-tsv docs/pr/pr118-cont-pairs.tsv work-pr118r1-dbg-{mcp1k,mcp2k,ncpi0,nuecc48}
```

## 1. The residual this round targets (measured in pr/117)

- **Class A — wrong-owner, cone-unreachable.** 21 of the 25 `pass4_angle`
  marks sitting in a neighbour fragment shower fail EVERY start-anchored cone
  against the scanned shower's own axis (evt409634: six segments 55–65°
  off-axis at 75–116 cm; evt415278: six at 20–29° beyond 130 cm).  Doc 117 §5:
  the recoverable route is wholesale fragment merge, or pr/91 P2's
  distance-to-charge admission for the conn-3 case.
- **Class B — sub-5 cm stubs.** The 41 "orphan" marks are all already members
  of their own ~1-segment showers (median 1.7 cm, 40/41 pdg-11) at body
  distance median 19.1 cm (IQR 9.7–29.7).  Gap-only merging measured
  net-negative: winner and loser gap distributions overlap completely at
  2.0–5.9 cm (doc 117 §7) — no gap threshold separates a detached cascade
  piece from a separate blip.  The residual under-clustered showers are
  purity 1.000 / completeness 0.50–0.80: pure loss of reach.

## 2. Phase A — byte-neutral instrumentation (shipped)

Both probes print under `WCT_SHOWER_MERGE_DEBUG` only; the legacy decisions
are untouched (probe-arm hash gates in §3).

### 2a. `SHOWER_MERGE tag=ex_shower1_p2dis` — the P2 dual distance

At the `examine_shower_1` conn-3 gate (`NeutrinoShowerClustering.cxx`, the
`conn_type1 > 2 && min_dis > 3cm` door; pr/91 §4 "G-C"), every candidate
reaching the gate prints BOTH measurements — `start_dis` (legacy: distance to
the parent's start segment) and `body_dis` (min over ALL the parent's ordered
member segments; includes the start segment, so `body_dis <= start_dis`
always) — plus the would-be downstream `angle`/`angle1` and three verdicts:
`legacy_gate`, `body_gate`, `angles_pass`.  This predicts the P2 knob's admit
set AND its angle-gate survival (pr/91's evt394532 lesson: admission is not a
merge guarantee) before the knob fires anywhere.

### 2b. `SHOWER_MERGE tag=cont_probe` — the charge-continuity pair census

A new file-local pass at the top of `merge_shower_fragments` (production
loops untouched) enumerates every EM–EM shower pair (fragment strictly
shorter; 45 cm knn prefilter; `gap_exact <= 30 cm` — covers the stub body-
distance IQR upper quartile), with NO min-len floor and NO γγ cut so offline
analysis can apply any candidate predicate.  Per pair:

- `gap` (production recipe: `shower_get_closest_dis` min over fragment
  members) and `gap_exact` — an exact deterministic argmin over the
  fragment's member fit points against the absorber's fit cloud
  (`shower_get_closest_dis` is a 4-query seeded ping-pong approximation:
  fine for the frozen legacy gates, wrong for walk endpoints);
- the continuity walk between the exact endpoints: 0.6 cm steps,
  `dv->contained_by` → `transform->backward` (fragment cluster's t0) →
  `Grouping::is_good_point(0.2 cm, 0, 0)` — the recipe forked from
  `NeutrinoGraphAudit.cxx` / `TaggerCheckTGM.cxx` `straight_steiner_chain`,
  WITHOUT the chain-builder's `n_bad > 1` early exit (the probe wants the
  full distribution): `nstep/ngood/nbad/badrun/cont_frac`;
- line charge `qmed/qfrac` (`get_ave_3d_charge` per in-TPC sample);
- junction dQ/dx: `dqdx_frag` (the fragment segment holding the endpoint),
  `dqdx_abs` (the absorber member nearest the junction);
- `angle_fold` (local-pivot 30 cm axis-folded; −1 when unmeasurable — the
  sub-5 cm stub case), `len1/len2`, `conn1/conn2`, `mv1/mv2` (the γγ guard
  inputs), `t0_frag/t0_abs` (cross-cluster t0 sanity — measured: zero
  mismatches over 2466 pairs);
- (second measurement pass, dbg2 arm) the absorber-AXIS geometry:
  `ax15_ang/ax100_ang/ax_d` at the fragment start and
  `jx15_ang/jx100_ang/jx_d` at the junction — angle to the absorber's 30 cm
  and 100 cm start directions, added after the first census measured the
  fold and the charge walk non-separating (§4b).

No pass-4 probe: the Class A marks are a shower-PAIR question and the 30 cm
pair window enumerates exactly those pairs; pass 4 also runs before
`collect_charge_maps`, so a charge probe there would read empty maps.

## 3. Gate ledger

| gate | arms | verdict |
|---|---|---|
| probe binary, no env | `work-pr117r1-flipchk-*` vs `work-pr118r1-probe0-*` | **PASS 196/196** (28+34+38+96 archives byte-identical); `nusel-evt*.tsv` 98/98 identical |
| compiled config, knobs off | git-HEAD jsonnet vs pr/118 jsonnet, full tagger `pipeline_names` TLA + `dl_weights=` | **byte-identical** (diff empty) |
| compiled config, knobs on | 5 TLAs set | all 5 keys present (`shower_ex1_conn3_body_dis`, `shower_merge_relax_continuity`, `_cont_frac`, `_cont_gap`, `_cont_bad_run`) |
| Phase-B binary, probes on, knobs off | `work-pr117r1-flipchk-*` vs `work-pr118r1-dbg-*` | **PASS 196/196**; nusel 98/98 identical |
| axis-probe binary, probes on, knobs off | `work-pr117r1-flipchk-*` vs `work-pr118r1-dbg2-*` | **PASS 196/196** |
| final binary (tiered knob), no env | `work-pr117r1-flipchk-*` vs `work-pr118r1-off1-*` | **PASS 196/196**; nusel 98/98 identical |
| flip-equivalence | `work-pr118r1-onT-*` vs `work-pr118r1-flipchk-*` (post-flip cfg, no env) | **PASS 196/196**; nusel 98/98 identical |
| compiled config, revised keys | off diff empty; `shower_merge_relax_continuity` present when on | **PASS** both ways |
| `wcdoctest-clus` | probe / Phase-B / tiered builds | 2424, 2434, 2442 — all green |

The probe0 arm compiled its configs before the pr/118 jsonnet edit landed
(cfg mtimes 14:14 vs edit 14:15:25), so it is a clean pre-pr/118-config arm;
the compiled-config proof separately covers the jsonnet change.

## 4. Measurement (pins the Phase-B thresholds)

Census: `pr118_probe_census.py` on the dbg/dbg2 arms — 98 events, 234
`ex_shower1_p2dis` candidates, **2466 EM–EM pairs** within 30 cm, truth from
the emscan-0827 marks (MERGE = the scanner wants the fragment's segments in
the absorber, n=45; DISTINCT = scanned-and-unwanted or a no-marks event,
n=1834; UNKNOWN = absorber not a scanned shower, n=587).  Every candidate
predicate below is evaluated the way the knob would actually fire: γγ hard
guard, per-fragment argmin, and net-new relative to the production
merge_relax.

### 4a. P2 is measured near-dead

Of 234 candidates reaching the conn-3 gate over all 98 events, the
whole-body measurement admits exactly **1** that the start-segment
measurement rejects (evt388), and it fails the downstream
`angle<15 && angle1<15` test.  **Predicted yield of
`shower_ex1_conn3_body_dis`: zero merges.**  The pr/91 §4 exemplar
(evt174752, 4.914 → 1.704 cm) was the exception, not the class; the conn-3
3 cm gate is NOT what blocks the residual.  The knob ships (strictly
admissive, coded, byte-identical off) but is **not a flip candidate**.

### 4b. Charge continuity along the connector is measured dead

- `is_good_point(0.2 cm)` continuity: near ZERO even for close truth-merge
  pairs (cont_frac median 0.04 on MERGE; the 3-plane 2 mm test fails between
  blobs).  Every grid point admits ~3 true vs ~32–36 false.
- Charge presence (`get_ave_3d_charge > 0`, the qfrac feature) SATURATES in
  these dense events: truth-merge and truth-distinct stubs both sit at
  qfrac ≈ 1 (best stub operating point: 6 true / ~90 false).
- Physics reading: detached EM fragments are connected by **photon
  propagation, which deposits nothing** — line-charge continuity is the
  wrong observable for exactly the class it was proposed for.

### 4c. The absorber-axis cone is the discriminator that works

The scan labels themselves say why (`marks_detail`): 50 % of wanted marks
sit within 15° of the scanned shower's axis, 69 % within 25°, at distances
up to 186 cm — while false neighbours spread isotropically (a 7.5° cone is
~0.4 % of solid angle).  dbg2 re-measured every pair with the absorber-axis
angle (min over the 30 cm / 100 cm start directions, at the junction point).
Grid knee: axis<7.5°, d<120 cm, gap≤10 cm → 7 true / 14 false.  The
14 false decompose cleanly: touching-but-misfolded pairs (fold 45–58°) and
dim or off-window stubs — which yields the **two-tier operating point**:

| tier | predicate | admits |
|---|---|---|
| T1 touching aligned (any length) | gap_exact ≤ 1 cm ∧ axis < 7.5° ∧ fold < 30° | evt423981: 12038 → 12095 (65 cm wanted fragment, gap 0.07, axis 2.2°, fold 25.2°) |
| T2 bright aligned stub (len < 5 cm) | gap_exact ≤ 8 cm ∧ axis < 7.5° ∧ d < 120 cm ∧ qfrac = 1 ∧ qmed > 5000 | evt281485: 78057 → 88090; evt469665: 57015 and 63030 → 15003 |

**4 true / 0 false / 0 unknown over all 2466 pairs** (γγ-guarded, argmin,
net-new).  Overfit caution is stated plainly: five thresholds tuned on
45 positives; the F1 harness (§6) and owner review remain the actual gate.

Measured out of reach this round (documented, not knob-served):
- evt54332's stub (its connector genuinely carries no charge, qfrac 0.31 at
  d=100 cm — unbridgeable by any local feature);
- evt122660's two long fragments (axis 4.2° but fold 48–90° —
  indistinguishable from evt389538's false 64069 pair without more context);
- the wrong-owner flank marks of evt409634/evt415278 (fail axis and fold
  both — they are why the scanned shower missed them in the first place).

### 4d. Connected components: 16 γγ violations

The CC dry run over the pairwise relation at the loosest useful operating
point produces 16 γγ-guard violations (e.g. evt30504: five main-vertex
conn≤2 showers in one component).  The formulation stays measured-only, as
planned.

## 5. Phase B — knobs (DEFAULT OFF; coded, thresholds from §4)

### 5a. `shower_ex1_conn3_body_dis` (bool, false) — pr/91 P2

Measure-first idiom at the conn-3 gate: `body_dis` computed under
`knob || pr91_merge_dbg()`, printed under the probe, APPLIED only under the
knob (`min_dis = body_dis`).  Gate text and the 3 cm threshold unchanged;
strictly admissive (the min includes the start segment).  The substituted
value also feeds the downstream `min_dis < 28 cm` term — intentional and
coherent (strictly admissive there too).  Runner env
`SBND_SHOWER_EX1_CONN3_BODY_DIS`.

### 5b. `shower_merge_relax_continuity` family — the two-tier axis+charge path

The knob's first draft gated on `is_good_point` continuity; §4b measured
that dead, so the shipped form is the §4c two-tier predicate (measure →
design, per the round's brief).  Knobs (C++ defaults = the measured
operating point): `shower_merge_relax_continuity` (bool false) +
`_cont_frac` (1.0) `_cont_gap` (8 cm) `_cont_qmed` (5000) `_cont_axis`
(7.5°) `_cont_dmax` (120 cm) `_cont_t1_gap` (1 cm) `_cont_t1_fold` (30°);
requires `shower_merge_relax` on (the pass itself).  Inside
`merge_shower_fragments`, all pr/117 semantics retained (EM–EM,
fragment-first argmin, no chains, hard γγ guard):

- **T1** (any length): `gap_exact ≤ cont_t1_gap` ∧ absorber-axis angle
  `< cont_axis` ∧ local fold `< cont_t1_fold` — a touching, axis-aligned
  continuation whose fold is too loose for the legacy 15°.
- **T2** (fragments below the `min_len` floor only, which have no
  measurable direction): `gap_exact ≤ cont_gap` ∧ axis `< cont_axis` ∧
  junction within `cont_dmax` of the absorber start ∧ the connector walk
  charged on every sample (`qfrac ≥ cont_frac`) with median line charge
  `> cont_qmed`.  A touching pair has no connector samples and is T1's
  case, never T2's.
- ranking by the admitting metric (legacy `gap` / tier `gap_exact`; the
  smaller when both); verdicts `MERGE_CONT` / `cont_fail` and a `tier=`
  field added to the `tag=merge_relax` probe (plus `gap_exact=`).
- knob off ⇒ no new computation ⇒ byte-identical (§3 gates).

Runner envs `SBND_SHOWER_MERGE_RELAX_CONTINUITY{,_CONT_FRAC,_CONT_GAP,_CONT_QMED,_CONT_AXIS,_CONT_DMAX,_CONT_T1_GAP,_CONT_T1_FOLD}`.

Knob seats: `NeutrinoPatternBase.h` member block (+rationale),
`TaggerCheckNeutrino.h` mirror (cm), `TaggerCheckNeutrino.cxx`
configure/default_configuration/pattern_algos (cm→internal at the push only),
`doctest_clus_knob_defaults.cxx` pins,
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` args + key-suppression
(numerics null), `run_pr_chain_batch.sh` env→TLA block.

### 5c. Connected-components formulation — measured only (plan B3)

Transitivity breaks the γγ guard (two main-vertex conn-1/2 showers can join
through a shared intermediate), so the CC formulation is NOT shipped; the
§4 CC dry run over the probed pair relation counts guard violations and
over-merges, and the formulation is promoted to a knob in a later round only
if it beats the pairwise path on the residual events.

## 6. Validation and the flip

ON arm `work-pr118r1-onT-*`: `SBND_SHOWER_MERGE_RELAX_CONTINUITY=1`, all
numerics at C++ defaults (= the measured operating point).

- **Firing set = the prediction, exactly.**  4 `MERGE_CONT` lines across 98
  events — evt423981 tier 1 (12038 → 12095, gap_exact 0.07 cm), evt281485
  tier 2 (78057 → 88090), evt469665 tier 2 (57015, 63030 → 15003).  Nothing
  else fires.
- **Scores** (`em117_score.py --baseline pr117-onK12c --compare
  pr118-onT`): evt423981 shower 12095 **0.750 → 1.000 (+0.250)**; evt281485
  shower 88090 0.718 → 0.753 (+0.035); evt469665 shower 15003 0.899 → 0.911
  (+0.012).  **No negative deltas.**  Under-clustered bucket residual
  events recovered: 3 of the 13 in §1.
- **Hold-flat**: membership diffstat vs emprep-117onK12c — changed events
  **3 of 98** (the three targets; 8 segment slots), 95 unchanged, **zero
  control/good-event churn** (pr/117 shipped with 10 changed events).
- **nusel**: byte-identical on **all 98 events** including the three merged
  ones — selection and tracks untouched.
- **Flip** (owner pre-authorization 2026-08-28, "turn them on if validation
  pass"): `shower_merge_relax_continuity = true` in the SBND `tcn_knobs`;
  `shower_ex1_conn3_body_dis` ships OFF/not selected (§4a, zero yield).
  Flip-equivalence: `work-pr118r1-flipchk-*` (post-flip config, no env)
  hash-gated against the onT arm.
- **Owner review material** (the round's one honest doubt is stated in §4c:
  five thresholds tuned on 45 positives with no independent event set — the
  Bee scan is the out-of-harness check): A/B pair `bee/pr118r1/`,
  OFF `82391d8d-a90c-4596-ad7b-467f3235df52` /
  ON `c8ea5ce3-0f26-40c0-be94-3b44d0ed7132`, idx 0 = 423981, 1 = 281485,
  2 = 469665.

## 7. What stays open

- **evt54332's stub** (76038): its connector genuinely carries no charge
  (qfrac 0.31, d = 100 cm) — no local pairwise feature can bridge it; would
  need cascade-level context (e.g. the multi-fragment trunk pattern).
- **evt122660's long fragments** (47050, 53070): axis 4.2° but fold 48–90°,
  feature-identical to evt389538's false 64069 pair — the pairwise feature
  set measured this round cannot separate them; parked.
- **The wrong-owner flank marks** (evt409634, evt415278): fail axis and
  fold both; still the hardest class.  pr/91 P2 is now RETIRED as the route
  (§4a); recovery needs something that models cascade development, not
  admission geometry.
- **Connected components**: measured 16 γγ violations; stays a
  measurement, not a knob.
- π⁰ pairing/reporting — unchanged, deferred by owner instruction.

## 8. Files

| file | role |
|---|---|
| `clus/src/NeutrinoShowerClustering.cxx` (toolkit) | probes 2a/2b + knob paths 5a/5b |
| `clus/inc/WireCellClus/NeutrinoPatternBase.h`, `TaggerCheckNeutrino.{h,cxx}`, `clus/test/doctest_clus_knob_defaults.cxx` | knob seats + pins |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | jsonnet args + key-suppression |
| `run_pr_chain_batch.sh` | pr/118 env→TLA block |
| `scripts/pr118_probe_census.py` | §4 census (P2, continuity separation, grids, CC dry run) |
| `docs/pr/pr118-cont-pairs.tsv` | the joined 2466-pair feature/truth table (dbg2 arm) |
| `docs/pr/pr118-onT-score.tsv` | knob-ON harness scores (33 marked showers) |
| `em_display/emprep-118{dbg,dbg2,onT}/` + manifests | arm sidecars |
| `bee/pr118r1/` | owner A/B index + prid-maps + set URLs (zips untracked) |

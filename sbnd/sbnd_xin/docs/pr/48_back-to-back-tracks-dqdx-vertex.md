# doc pr/48 (2026-08-07) — back-to-back tracks with no angular kink

**Status: analysis only. No C++ or jsonnet shipped this round** (owner's
explicit instruction). A temporary, fully-reverted break-time diagnostic was
built and run (§4); the toolkit tree ends this doc byte-identical to
`20098cbf` (`git diff` empty, `wcdoctest-clus` 106/106, freshness proof
re-verified after revert).

## Repro block

```bash
# Phase 0 -- fresh HEAD run of the 5 events with the calib-pr dump
wcbuild
cd wcp-porting-img/sbnd/sbnd_xin
PR_EXTRA_STAGES=pr_display PR_JOBS=5 bash run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr48-case5 data 51513 56211 57903 59335 57485

# Phase 1 -- break-time diagnostic (WCT_PR48_DIAG, reverted before commit)
# see sec 4 for the exact instrumentation; example invocation:
WCT_PR48_DIAG=1 PR_JOBS=1 bash run_pr_chain_batch.sh \
    work-mcp1k-cb0805 /home/xqian/tmp/pr48/diag-59335 data 59335

# Phase 2 -- off-cathode census over the 445 dumps already in work-mcp1k-cb0805
python3 scripts/analysis/pr48/backtoback_census.py \
    > scripts/analysis/pr48/census_output.tsv
```

All numbers below are read from `work-mcp1k-cb0805` (QL) / the fresh
`work-pr48-case5` label (PR, re-run at HEAD `20098cbf`) unless noted. The
main-cluster neutrino vertex is byte-identical between `work-mcp1k-cb0805`
and `work-pr47f-on1k` for all five events, so the vertex conclusions in this
doc are HEAD-valid even though the original calib dumps predate today's
commit.

## 1. The five events

Owner's message (run 18255): 51513, 56211, 57903, 59335 "vertex @
(-172.5, 47.0, 78.8)", and 57485 ("middle of the muon... not exactly
back-to-back? ... Nu vertex around (-9.8, -108.1, 50.3), cluster 16006").

The common symptom: **the tracks are back-to-back — visible in the dQ/dx
profile — but there is no angular kink, so no graph vertex gets created, so
there is no neutrino-vertex candidate.** For the first three events there is
also no third prong to help (simple topology); for the 4th and 5th the owner
suspected the kink detector was simply missing a real kink.

| evt | main segment(s) | L (cm) | reco nu vertex (arc) | owner anchor / truth | dist to nearest anchor |
|---|---|---|---|---|---|
| 51513 | seg 16000, pdg 13 | 73.9 | arc 73.9 (far end, deg 1) | none supplied (Bragg profile suggests arc ≈ 3–7) | — |
| 56211 | seg 11000, pdg 2212 | 21.9 | arc 0 (deg 1) | none supplied (Bragg minimum ≈ arc 19–20) | — |
| 57903 | seg 14002 (+14010, 1.9cm stub), pdg 13 | 103.8 | arc 0 (deg 2) | (-16.8,-57.1,265.1) | **0.05 cm**, arc 49.8/103.8 |
| 59335 | seg 11005 (+11001, 0.89cm stub), pdg 211 | 67.8 | arc 67.8 (far end, deg 1) | (-172.5,47.0,78.8) | **0.03 cm**, arc 66.6/67.8 |
| 57485 | seg 16006 (+16004/16005, ≤3.6cm stubs), pdg 2212 | 87.3 | arc 87.3 (far end, deg 2) | (-9.8,-108.1,50.3) | **0.02 cm**, arc 37.8/87.3 |

For 57903, 59335 and 57485 the owner-supplied coordinate sits on the fitted
track to within 0.05 cm — these are not vague guesses, they land almost
exactly on a fit point, confirming the truth vertex really is mid-segment.

## 2. The unifying signature: no graph vertex exists at the true position

In all five events the neutrino vertex lies **mid-segment on a single
unbroken fitted track**, and the only graph vertices in the cluster sit at
the track's two termini (plus, in three of the five, a short stub absorbed
near one end). This is not a case of "the vertex was placed somewhere else
nearby" — it is a case of "no vertex exists within tens of centimeters."

57903's cluster 14 illustrates this most starkly: it has exactly three
vertices — the arc-0 main vertex (degree 2, with a 1.9 cm stub), the arc-103.8
far terminus (degree 1), and the stub tip — and **nothing within 50 cm of the
true vertex at arc 49.8**.

The common detector across all five is dQ/dx-only, angle-independent: **dQ/dx
rises at both ends of the single segment and dips in the middle.** A single
particle has at most one Bragg (stopping) end; two rising ends imply a
junction, located at the dip.

```
evt 56211 (arc, dQ/dx in MIP, 0.6cm steps from arc 0):
  7.57 6.12 5.51 4.95 4.80 4.22 3.46 4.37 4.09 3.91 ... 3.42 3.14 2.94 ...
     [Bragg peak, stopping]                              [dip, ~2.9 MIP]
  ... 2.85 2.81 2.79 2.91 2.95 2.37 2.57 2.45 2.94 4.57 6.32
                                                    [second Bragg peak, rising to 6.32]

evt 57903 (arc, dQ/dx in MIP, 0.6cm steps around the truth vertex at arc 49.8):
  arc 45.0:2.01  45.6:2.09 ... 47.4:2.55 48.0:2.11  48.6:0.81  49.2:0.82  49.8:1.15(=truth)
  50.4:1.42  51.0:1.50 ... 54.6:2.36
  first 8cm median 3.60 MIP, dip at truth 0.81-0.82 MIP, last 8cm median 2.25 MIP
```

## 3. Why the chain cannot see them today

**No code path anywhere in the PR chain breaks a segment on dQ/dx alone.**
`segment_search_kink` (`clus/src/PRSegmentFunctions.cxx:270`) has five accept
paths, evaluated in order once the sticky `flag_check` latch fires
(`:369/:377/:388`):

```cpp
// A0 (pr/47, cathode-crossing indices only): wide-baseline PCA turn angle
if (cathode_wide_kink_angle > 0 && wide_accepts.count(i)) { save_i = i; break; }   // :396
// pr/20 cathode veto sits here, between A0 and C1-C4                              // :408
// C1
if (para_angles[i] > 10 && refl_angles[i] > 30 && sum_angles  > 15)                // :449
// C2
else if (para_angles[i] > 7.5 && refl_angles[i] > 45 && sum_angles1 > 25)          // :452
// C3
else if (para_angles[i] > 15 && refl_angles[i] > 27 && sum_angles  > 12.5)         // :455
// C4 -- the only dQ/dx-assisted path, and only as a TIGHTENER on a lower angle floor
else if (para_angles[i] > 15 && refl_angles[i] > 22 && sum_angles > 19
      && max_dQ_dx > dQ_dx_threshold*1.5 && ave_dQ_dx > dQ_dx_threshold)           // :459-460
```

`refl_angles[i]` is a **local** statistic: the minimum reflex angle over 6
half-window offsets of 1–7 cm either side of index `i`
(`PRSegmentFunctions.cxx:301-367`). It is designed to suppress noise on an
ordinary track, and it does — but it also suppresses a genuine kink that is
spread over more than ~7 cm, exactly the geometry a back-to-back junction with
some multiple-scattering rounding produces.

The prototype has no dQ/dx-driven break either. Its one real dQ/dx **step**
finder, `NeutrinoID_ssm_tagger.h:428-474` (ported at
`clus/src/NeutrinoTaggerSSM.cxx:636-700`, `|Δ dQ/dx| > 0.7` MIP over the
first/last 4 points), produces BDT scalars and a direction flip — it never
creates a vertex or splits a segment.

## 4. Diagnosis of 59335 and 57485 (break-time instrumentation)

Both events were diagnosed with a temporary, env-gated (`WCT_PR48_DIAG`)
instrumentation added to `segment_search_kink`, `break_segments`,
`examine_structure_2`, `examine_structure_3`, `examine_vertices`, and
`find_proto_vertex` (checkpoint vertex dumps after every sub-stage) —
following the `WCT_PR47_DIAG` idiom from doc pr/47 §8: `fprintf(stderr, ...)`
gated on `std::getenv("WCT_PR48_DIAG")`, no behavior change when unset,
**fully reverted with `git checkout --` before this doc was committed**
(verified: `git diff` empty, freshness proof re-run, `wcdoctest-clus` 106/106).

### 59335 — the kink IS found, correctly, almost exactly at the truth vertex

This event's diagnosis is a genuine surprise and reframes it from the earlier
working hypothesis ("proton stub absorbed at the tip, not really back-to-back")
to something more specific and more fixable.

`segment_search_kink`'s scan over the 67.8 cm track (internal units mm; arc
positions divide by 10 for cm) finds, on the **second** break-time pass
(after an unrelated 5 cm stub near the start was already split off):

```
PR48DIAG cand i=111 x=-1721.99 refl=26.6 para=43.4 sum=29.6 sum1=29.6
         maxdqdx=24969 avedqdx=11991 thr=4800 c1=0 c2=0 c3=0 c4=1
PR48DIAG accept save_i=111 p=(-1721.99,473.70,794.54) flag_switch=0 flag_search=1
```

Index 111 is arc 66.6 cm — **0.28 cm from the owner-supplied truth vertex**
(-172.5, 47.0, 78.8). C4 fires correctly: this is the genuine junction.

But **C4 sets `flag_search=true`**, and `segment_search_kink`'s return logic
(`PRSegmentFunctions.cxx:552-556`) makes `flag_search=true` return
`flag_continue=true` **unconditionally**, bypassing the `local_dQdx` check
that otherwise gates the walk (`:563-568`). `proto_extend_point`
(`NeutrinoPatternBase.cxx:1638`) then walks forward from the kink point and
overshoots ~1.1 cm further, landing the break only **0.47 cm** from the far
terminus:

```
PR48DIAG bs_extend kink=(-1721.99,473.70,794.54) break_pt=(-1727.46,464.82,789.70)
         break_idx=8 flag_continue=1 dist_start_v=65.22 dist_end_v=0.47
PR48DIAG bs_break break_wcp=(-1727.46,...) min_dis_cm=0.467 angle=94.0
         kink_angle_at_break=18.9 end_is_terminus=1 use_replace=0
```

`use_replace` is false (`kink_angle_at_break=18.9° < 30°` required,
`NeutrinoPatternBase.cxx:1814-1817`), so `break_segment_into_two` fires and
**a real, valid split is created**: a 0.47 cm stub segment between the new
vertex and the far terminus. Confirmed live — both `examine_structure_2`
(`n_bad=2`, fails its own `<=1` gate, no merge) and `examine_structure_3`
(`angle_10cm=71.3°`, fails the `<18°` collinearity gate, no merge) leave the
split intact.

**The break is then silently erased by `examine_vertices_4`
(`clus/src/NeutrinoStructureExaminer.cxx:1812`):**

```cpp
if (direct_length < 2.0 * units::cm ||
    (tmp_dir.magnitude() < 3.5 * units::cm && std::fabs(angle - 90.0) < 10)) {
    ... // absorb: merge the short-end vertex into the long-end vertex
```

The 0.47 cm stub is far under the unconditional 2 cm floor. `examine_vertices`
is called from `find_proto_vertex` (`NeutrinoPatternBase.cxx:2460`), and the
checkpoint dumps show the vertex at (-1727.46, 464.82, 789.70) present through
`post_examine_structure_3`, then **gone** after `examine_vertices_4` (fired=1
on its first pass), leaving the single unbroken track with the vertex back at
the far terminus.

**Named line: `NeutrinoStructureExaminer.cxx:1812`** (`examine_vertices_4`'s
unconditional `direct_length < 2.0*units::cm` stub-absorption floor), acting
on a stub created by the walk overshoot in `proto_extend_point`
(`NeutrinoPatternBase.cxx:1638`, triggered by `flag_search=true`'s
unconditional `flag_continue=true` at `PRSegmentFunctions.cxx:552-556`).
**The kink detector did its job here.** The bug is downstream: a correctly
identified junction gets walked past its own evidence and then swept up by a
length-floor cleanup rule that has no exception for "this stub was born from
a high-confidence accept."

### 57485 — the kink is never found; a near-miss on the same threshold as pr/47's 52085

The live scan over the full 87.3 cm track never accepts anywhere near the
truth junction. At the closest candidate, index 63 (x = -97.34 mm = **-9.73
cm**, essentially exact match to the owner's -9.8 cm):

```
PR48DIAG cand i=63 x=-97.34 refl=23.2 para=19.3 sum=17.0 sum1=17.0
         maxdqdx=8221 avedqdx=5743 thr=4800 c1=0 c2=0 c3=0 c4=0
```

`para_angles[63]=19.3 > 15` ✓, `refl_angles[63]=23.2 > 22` ✓, but
`sum_angles=17.0` **fails `> 19` by 2.0** — the exact same conjunct, on the
exact same criterion (C4's `sum_angles > 19` floor), that doc pr/47 documented
missing by 0.21 at event 52085's cathode junction. This is not a coincidence:
`sum_angles` is a windowed RMS of `refl_angles` over 5 neighboring indices
restricted to `para_angles > 10`, and it structurally under-reads a kink whose
angular signature is spread across several cm rather than concentrated at one
index — precisely what a genuine but not razor-sharp back-to-back junction
looks like.

The **only** accept on the entire segment fires far from the truth vertex, at
the far terminus (index 135, x = -43.9 cm, refl=38.3° — C1), producing the
small stub segments (16004/16005) actually seen in the final reconstruction —
not the true vertex.

**Named line: `PRSegmentFunctions.cxx:459`** (`sum_angles > 19` in the C4
conjunct). No downstream mechanism needs to be blamed here — the kink is
simply never accepted in the first place.

## 5. Census: how far can angle alone reach, and what does it cost?

Script `scripts/analysis/pr48/backtoback_census.py`, run over the 445/1000
`calib-pr-evt*.json` dumps available in `work-mcp1k-cb0805` (same coverage
caveat as pr/20's `kink_probe.py` and pr/47's `cathode_junction_census.py` —
this is a **post-fit proxy measurement**, not a break-time measurement; §4's
live diagnostic is the ground truth for the two named events). 641
main-cluster segments (≥8 fit points, ≥12 cm long) qualified.

### Wide-baseline PCA turn angle (pr/47's statistic, un-gated from the cathode)

Evaluated at **every interior index** (≥6 cm from both segment ends —
pr/47's own end-artifact gotcha), both directions, at two baselines:

| baseline | length bucket | n | ≥20° | ≥25° | ≥45° | ≥50° |
|---|---|---|---|---|---|---|
| 15 cm | short (<30cm) | 224 | 31.7% | 21.9% | 8.0% | 6.7% |
| 15 cm | mid (30-60cm) | 119 | 26.9% | 14.3% | 4.2% | 3.4% |
| 15 cm | long (>60cm) | 298 | 36.6% | 24.5% | 4.0% | 2.7% |
| 35 cm | short | 224 | 31.2% | 21.9% | 8.0% | 6.7% |
| 35 cm | mid | 119 | 26.1% | 15.1% | 4.2% | 3.4% |
| 35 cm | long | 298 | 34.9% | 25.8% | 4.4% | 4.0% |

**This is decisive, and it overturns the "just un-gate pr/47's helper" idea
that motivated this measurement.** Taking the *maximum turn angle at any
single interior index* over an ordinary track — with no requirement that the
index be a stable local peak, and no dQ/dx corroboration — reads ≥45° on
**4-8% of all main-cluster segments**, and ≥25° on **14-26%**. An angle-only
accept at either the ~50° tier or the ~25-30°/long-baseline tier, un-gated
from the cathode, would misfire constantly. pr/47's own lesson (§7 of that
doc: don't rescue a design by moving the cut to fit motivating events)
applies directly — **the fix is not "pick a threshold below where these five
events measure," it is "angle alone is not a viable un-gated statistic."**
(pr/47's *cathode-gated* version was safe because the crossing-index set it
evaluates on is small and physically motivated — a handful of true crossings
per event, not every interior index of every track.)

### Angle AND the two-end dQ/dx rise, combined

Restricting to segments where the two-end dQ/dx-rise gate (median dQ/dx over
the first/last 8 cm > 1.3× the interior median, on **both** ends) also fires
collapses the tail dramatically:

| statistic | alone | AND two-end-rise |
|---|---|---|
| wide-turn(15cm) ≥ 45° | 35/641 (5.5%) | **1/641 (0.16%)** |
| wide-turn(35cm) ≥ 25° | 144/641 (22.5%) | **8/641 (1.25%)** |

The single survivor of the tightest combination (turn≥45° AND two-end-rise)
is **event 57485 itself** — the statistic is essentially unique to the true
positive in this sample. The 8 survivors of the looser combination include
57485 and 6 other candidates not otherwise investigated (52085, the pr/47
event, is also in the raw two-end-rise list, consistent — it's the same
family of junction).

**57903 does not survive either combined gate.** Its own two-end-rise ratios
are 1.99 (first 8 cm) and **1.24** (last 8 cm) — just under the 1.3 floor on
one end, a near-miss structurally like both the C4 misses documented above.
Its wide-turn(35cm) does clear 25° (32.8°, §"How far does angle reach"
earlier), but without the rise gate the combined criterion misses it.

### Two-end-rise gate alone

14/641 segments (2.2%) satisfy the two-end-rise gate on its own (both ends
>1.3× interior median dQ/dx) — a much more usable false-positive rate than
angle alone, and it independently recovers 57485
(ratio_lo=2.55, ratio_hi=1.4). Full list in `census_output.tsv`.

**It structurally misses 51513 and 56211**, the two events with no usable
angular signal at all — not because the physics isn't there, but because the
census script's fixed 8/12 cm windows don't fit inside a 21.9 cm track (56211:
the "interior" window `12 < arc < L-12` is empty for any L < 24 cm) or land on
the wrong side of a Bragg peak concentrated in the first 3-4 cm (51513: first-8cm
median dilutes a peak that's really over by arc 3.6). **This is an honest
limitation of the census proxy, not evidence the physics doesn't hold** — the
hand-computed 0.6 cm-binned profiles for both events (§2) show the two-end
signature clearly; a real implementation needs length-adaptive windows (§6).

## 6. Recommendation

Angle-only knobs (tiers originally proposed as "un-gate pr/47's helper,"
either at ~50° or at ~25-30°/long-baseline) are **not supported** by the
census in §5 — both are heavily populated on ordinary tracks. The viable
paths are:

**Recommended, smallest blast radius: angle AND dQ/dx two-end-rise,
combined.** A new accept path (NOT reusing `segment_cathode_wide_kink_accepts`
directly — that helper is cathode-scoped by design; a genuinely new, similarly
narrow helper is needed) requiring *both* a wide-baseline turn ≥ some
threshold in the 35-50° range *and* a two-end dQ/dx rise, gated to the
simple-topology case (§7 below). The combined statistic reaches 57485 cleanly
(unique in the census) and a handful of similar events (1.25% at the looser
operating point), at a population small enough to hand-scan before any
default flip — matching the owner's own staged-validation practice (see
`feedback_staged_small_group_validation.md`). **It does not reach 57903, 51513,
or 56211.**

**Necessary for 51513, 56211, and robustly for 57903: the two-end
residual-range algorithm (the genuinely new piece).** Frame it as dQ/dx vs
residual range measured *from each end*, reusing `do_track_comp`
(`PRSegmentFunctions.cxx:1667`, which already carries the muon/proton
dQ/dx-vs-residual-range stopping templates) rather than the crude
median-ratio proxy this census used — the proxy already shows the right shape
(56211's 7.57→2.37→6.32) but structurally cannot handle short segments or
peaks near a segment edge, exactly the two events that most need it. Scan the
candidate break index k maximising the joint two-arm stopping likelihood, and
validate the located index against the two truth anchors with sub-centimeter
precision (57903 arc 49.8, 57485 arc 37.8) before trusting it on 51513/56211
where no truth was supplied (only Bragg-profile estimates, §1).

**59335 needs neither of the above.** Its junction is already found and
already breaks correctly — §4's diagnosis names two independent, narrower
fixes, either of which is smaller-blast-radius than a new detector:
(a) do not force `flag_continue=true` unconditionally when `flag_search=true`
walks past a C4 accept (`PRSegmentFunctions.cxx:552-556`) — let the existing
`local_dQdx` check (`:563`) gate the walk the way it does for every non-C4
accept, so the walk stops at the genuine kink instead of overshooting to the
terminus; or (b) give `examine_vertices_4`'s 2 cm absorption floor
(`NeutrinoStructureExaminer.cxx:1812`) an exception for stubs whose creating
break passed a high-confidence angular/dQ/dx test, rather than absorbing every
short segment unconditionally. Either is independent of the back-to-back work
above and should be scoped, gated, and validated separately.

## 7. Blast-radius gate (for whichever knob ships next round)

- **Default OFF** (C++ member default 0/false; byte-identical when off,
  per the standing project convention).
- **Simple-topology gate, primary**: exactly one main-cluster segment, both
  endpoints degree-1 in the graph (not just "no attached prong > 5 cm" — the
  57903/57485 stubs show short attached stubs are common even in genuine
  back-to-back events, so a strict degree-1 test would wrongly exclude them;
  use "no non-stub prong," i.e. exclude only attachments over some length
  floor, TBD next round from a wider stub-length census).
- **Both endpoints inside the fiducial volume, primary, not secondary**: two
  stopping ends is physically impossible for a through-going or exiting
  track, so containment kills cosmics and exiters for free. The scope-aware
  FV already exists in this tree (`project_scope_aware_fv.md`).
- **A length window and a minimum arm length post-break** (both arms must be
  long enough to compute a meaningful median dQ/dx and residual-range
  profile — the census's own window-size failure on 56211 is the concrete
  lower bound to respect).
- **Survivability past `examine_structure_2`/`_3`**: §4 confirmed both
  merge-back passes correctly leave a *genuine* kink split alone (n_bad=2,
  angle_10cm=71.3° for 59335's break) — but a future straight-junction split
  (class A, 51513/56211, near-zero local angle) is exactly the geometry
  `examine_structure_2`'s angle-free straight-line test is designed to
  re-merge. Any tier-3 (dQ/dx-only) break MUST carry a protect flag past
  `examine_structure_2` and `_3`, or be placed after both run — otherwise a
  correctly-placed vertex on a straight track is silently undone the same way
  59335's genuinely-found kink was undone by `examine_vertices_4`.

## 8. Open items, not fixed this round

- `NeutrinoPatternBase.cxx:1552`: `int count = 0; while(...&& count < 2)` —
  `count` is never incremented anywhere in `break_segments`; the `count < 2`
  guard is dead code (the loop actually runs until `remaining_segments`
  drains). Noted, not fixed (out of scope for this investigation).
- 57903's cluster 14 has 303/1286 track_shower points (max 18.6 cm) not within
  5 cm of any fitted segment, forming a real unfitted branch. Measured
  distance from this branch to the true vertex: **≥7.3 cm, median 28.3 cm** —
  a separate issue; the branch does not explain 57903's missing vertex, and
  57903 remains a genuine back-to-back event in this doc's classification.
- 51513 and 56211 have no owner-supplied truth anchor; §1's arc estimates come
  from the Bragg-profile minimum only. If ground truth becomes available it
  should be checked against these estimates before the tier-3 algorithm is
  validated on these two events specifically.
- The `sum_angles > 19` near-miss pattern (52085 by 0.21, 57485 by 2.0) recurs
  often enough across two independent docs that it may be worth its own
  investigation — not proposed here, flagged for the owner's judgment call
  (constants like this are exactly the kind of "don't retune to fit motivating
  events" the project has already ruled on once, pr/47 §7).

## Files

- This doc.
- `scripts/analysis/pr48/backtoback_census.py` — the census script (§5).
- `scripts/analysis/pr48/census_output.tsv` — its raw tail-population output.
- Diagnostic instrumentation: **not committed** — added to
  `clus/src/{PRSegmentFunctions,NeutrinoPatternBase,NeutrinoStructureExaminer}.cxx`
  in the toolkit repo, run, and reverted (`git checkout --`) before this doc
  was written. `git diff` in the toolkit repo is empty; `wcdoctest-clus`
  106/106 after revert+rebuild.

Related: [[project_pr47_cathode_vertex_investigation]] (the `sum_angles > 19`
near-miss precedent, the wide-baseline PCA turn-angle statistic reused here),
[[project_demoted_main_partI]] (doc pr/20 Part I — simple-topology /
stub-length precedent, "length floor not derivable" caveat applies here too),
[[feedback_staged_small_group_validation]] (the staged-rollout practice this
doc's recommendation follows).

# doc pr/48 (2026-08-07) — back-to-back tracks with no angular kink

**Status: IMPLEMENTED + SBND DEFAULT ON, round 2 (§9) — three knob families
shipped, all five motivating events recover their neutrino vertex; off-gates
48/48 + 19/19 byte-identical, 1k footprint 69/1000 movers all classified and
examined, nusel diffs 0/1000.** §§1-8 are the round-1 analysis (2026-08-07,
analysis-only at `20098cbf`), kept verbatim; §9 is the implementation round
(same day), whose measured behavior refines §6's design in several places
(§9.3).

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

## 9. Implementation round (2026-08-07, toolkit commit 502cfd7e)

### 9.1 Repro block

```bash
# build + unit tests (1148 doctest assertions incl. the new pr48 cases)
wcbuild; ./build/clus/wcdoctest-clus

# off-gates (knobs forced 0 must be byte-identical; base arms = clean 20098cbf binary)
cd wcp-porting-img/sbnd/sbnd_xin
PR_JOBS=6 bash run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr48-base48 data
PR_JOBS=4 bash run_pr_chain_batch.sh work-ncpi0-cb0805  work-pr48-base19n data
SBND_TWO_END_BREAK=0 SBND_KINK_WALK_DQDX_STOP=0 SBND_KINK_BREAK_PROTECT=0 \
  PR_JOBS=6 bash run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr48-off48c data
SBND_TWO_END_BREAK=0 SBND_KINK_WALK_DQDX_STOP=0 SBND_KINK_BREAK_PROTECT=0 \
  PR_JOBS=5 bash run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr48-off19nc data

# knob-on arms (footprint) + the 1k
SBND_TWO_END_BREAK=1 SBND_KINK_WALK_DQDX_STOP=1 SBND_KINK_BREAK_PROTECT=1 \
  PR_JOBS=12 bash run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr48-on48c data
SBND_TWO_END_BREAK=1 SBND_KINK_WALK_DQDX_STOP=1 SBND_KINK_BREAK_PROTECT=1 \
  PR_JOBS=5 bash run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr48-on19nc data
SBND_TWO_END_BREAK=1 SBND_KINK_WALK_DQDX_STOP=1 SBND_KINK_BREAK_PROTECT=1 \
  PR_EXTRA_STAGES=pr_display PR_JOBS=32 \
  bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr48-on1kc data

# compares: archive movers + vertex + nusel diffs, then class + per-case exam
python3 scripts/analysis/pr48/on1k_compare.py  work-pr47f-on1k work-pr48-on1kc
python3 scripts/analysis/pr48/mover_census.py  work-pr47f-on1k work-pr48-on1kc
python3 scripts/analysis/pr48/teb_case_exam.py work-pr47f-on1k work-pr48-on1kc
```

Superseded labels (kept, never deleted): `work-pr48-{off48,off19n,on48,on19n,on1k}`
(pre-footprint-round binary), `-{off48b,off19nb,on48b,on19nb,on1kb}`
(`kink_dqdx_hot_ratio` round), `-{onT48,onW48,onP48}` (single-knob attribution,
killed mid-run — possibly truncated, do not gate against).  The `c` labels are
the final binary.

### 9.2 What shipped (all default OFF in C++; SBND operating point in
wct-pr-perevt.jsonnet after the gates below)

- **`two_end_break`** (+ 15 `teb_*` sub-knobs) — the two-end residual-range
  break pass, `PatternAlgorithms::break_two_end_dqdx`, run inside
  `find_proto_vertex` after `examine_vertices` (`NeutrinoPatternBase.cxx`),
  main cluster only.  Gates: exactly one segment longer than `teb_stub_max`
  (4 cm), length ≥ `teb_min_len` (10 cm), **both fitted endpoints inside the
  FV** (`FiducialUtils` via the grouping — two stopping ends is impossible
  for a through-goer); length-adaptive two-end rise (max over 2/4/8 cm end
  windows of the end median, referenced to `min(interior median, 1 MIP
  median)` — the clamp is what reaches 56211, whose two-proton interior is
  ~3x MIP); localization + acceptance per §9.3; break via `break_segment`
  at the located fit point; the new vertex gets
  `VertexFlags::kProtectedBreak` and the two arms
  `SegmentFlags::kTwoEndBreakArm`.
- **`kink_walk_dqdx_stop`** — 59335 fix (a): the C4/straightness
  (`flag_search`) accepts no longer bypass `segment_search_kink`'s
  local-dQ/dx walk gate, so `proto_extend_point` stops AT a dQ/dx-confident
  kink instead of overshooting to the terminus.
- **`kink_break_protect`** — 59335 fix (b): a break born from a C4 or A0
  accept (criterion surfaced via a new nullptr-default out-param on
  `segment_search_kink`) gets `kProtectedBreak`, so `examine_vertices_4`'s
  unconditional < 2 cm stub-absorption floor cannot erase it.
- **`kink_dqdx_hot_ratio`** (shared by both 59335 fixes, default 1.7x MIP
  median) + **terminus scoping** — the footprint-control round (§9.3 item 5).
  As designed, fix (a)'s bar was the walk gate's own 25/43 ≈ 0.58x MIP, which
  nearly every C4 accept clears, and fix (b) protected ~100 interior stubs:
  206/1000 movers on the first 1k arm.  Shipped: both fixes engage only at
  `kink_dqdx_hot_ratio` (59335's kink reads 2.5x), fix (a) only when the kink
  is within 3 cm of a fitted-track end, fix (b) only when the stub arm is
  < 2 cm (`examine_vertices_4`'s own floor — protection exactly where the
  erasure happens, nowhere else).  59335 is unchanged at every narrowing.
- **Shared infra**: `VertexFlags::kProtectedBreak` — checked (skip-merge)
  in `examine_structure_2`, `examine_structure_3`, `examine_vertices_1/2/4`,
  `examine_structure_final_1/2`; every check is a no-op when the bit is
  unset.  `examine_structure_final_1p/3` need no check (they relocate the
  main vertex ONTO the surviving position).

### 9.3 What the five events taught (design refinements vs §6)

The §6 design survived contact with the events only after four measured
corrections — each one an iteration on the real data, none visible to the
round-1 proxies:

1. **Stopping-template scores cannot localize.** `do_track_comp` windows
   anchor at the vector end, so with a fixed window the per-arm score is
   CONSTANT in the break index for all mid-track candidates, and with
   full-arm windows it is still nearly flat for same-species junctions (any
   prefix of a Bragg is a Bragg).  A blind argmin-J scan mislocated 57903 by
   15 cm and 51513 by 32 cm in early iterations.  The junction's measured
   markers localize instead: the **raw local dQ/dx minimum** (candidates,
   deepest-3-point-mean first) for route R1, the **wide-baseline PCA turn
   maximum** for route R2 (fallback only — the turn max sits at endpoint
   bend artifacts on 51513/57903).  The templates remain the ACCEPTANCE
   test at the located index (both arms must beat `eval_ks_ratio` + score
   caps).
2. **A dip floor separates junctions from dead regions.** 57903's deepest
   dip (0.50x MIP mean-3, arc 64.8) is instrumental; its genuine junction
   dip reads 0.83x.  Both particles sit at maximal residual range at a
   junction, so a physical junction dip stays near MIP — `teb_dip_floor`
   (0.6x MIP median) vetoes missing-charge dips.  Measured on all four
   events the deepest above-floor raw-minimum IS the junction.
3. **Short arms need a direction-flag waiver.** 56211's ~2-3 cm far arm
   (2.8x MIP proton stub) fails `eval_ks_ratio` for lack of samples, not
   physics: arms < 6 cm are held to the score cap and the absolute end
   floor but not the strict stopping-vs-flat flag.
4. **Creating the vertex is necessary, not sufficient — three separate
   mechanisms then had to let it WIN:**
   (a) `break_segment` does not set the new vertex's cluster; a
   null-cluster vertex is invisible to `determine_main_vertex`'s candidate
   loops.  The driver sets it.
   (b) The arms' weak KS direction is a coin flip; landing "into the
   junction" excludes wrong vertices inconsistently.  `determine_direction`
   reconstructs each arm's outward direction from `kTwoEndBreakArm` + the
   protected endpoint and lets it stand over a WEAK recompute (a dirsign
   stamp cannot carry this — `shower_topo_reset` zeroes dirsign on every
   segment in `separate_track_shower`).
   (c) **The SCN-DL vertex rerank was the final mover on 3 of 4 events**:
   the traditional chain, with (a)+(b) in place, selects the junction on
   ALL FOUR back-to-back events — and `determine_overall_main_vertex_DL`
   then switched 51513/56211/57485 to a DL vertex snapped onto a Bragg tip.
   The mid-track back-to-back junction is precisely the topology the
   image-based DL vertex cannot distinguish (that blind spot is why this
   doc exists), so the DL rerank now never moves the main vertex OFF a
   `kProtectedBreak` vertex (`NeutrinoVertexFinder.cxx`; a DL choice that
   AGREES still passes; inert unless `two_end_break`).
5. **The 59335 fixes needed terminus scoping to hold the footprint.**
   Footprint evolution on the 1k, measured at each narrowing: 206/1000
   (as-designed bars) → 119/1000 (`kink_dqdx_hot_ratio` 1.7x MIP) →
   **69/1000** (terminus scoping, §9.2).  The teb pass itself contributes a
   designed population (37 breaks); the excess was all fix-(a)/(b) refit
   ripple on interior kinks that 59335 never needed.  nusel diffs were
   0/1000 at every operating point.

### 9.4 The five events, before → after (knobs on; tuned in
/home/xqian/tmp/pr48/case5on* scratch arms, re-verified at the final binary
in /home/xqian/tmp/pr48/case5onA and again on work-pr48-on1kc)

| evt | baseline nu vertex | pr/48 nu vertex | anchor | dist |
|---|---|---|---|---|
| 51513 | far terminus (99.2,74.6,113.5) | **junction (45.3,46.6,138.4)**, break at arc 7.0 | Bragg est. arc 3-7 | at the estimate |
| 56211 | arc-0 Bragg tip (92.7,-38.1,96.3) | **junction (88.5,-32.3,77.7)**, break at arc 19.1 | Bragg est. arc 19.5-20.4 | at the estimate |
| 57903 | far end, 47.8 cm off | **(-16.5,-57.7,265.5)** | truth (-16.8,-57.1,265.1) | **0.81 cm** |
| 59335 | terminus, 1.17 cm off | **(-172.7,46.8,79.0)** | truth (-172.5,47.0,78.8) | **0.37 cm** |
| 57485 | far end, 47.1 cm off | **(-10.0,-105.6,46.4)** | owner guess (-9.8,-108.1,50.3) | **4.61 cm** |

57485 note: the break lands at the algorithm's junction estimate (dip at
arc 45.0, wide turn 62.2 deg at arc ~46 — the two independent markers
agree), and the final vertex fit pulls it to 4.6 cm from the owner's
eyeballed anchor.  The owner coordinate was itself flagged a guess ("not
exactly back-to-back?"); the species step and both markers put the junction
at arc 45-46 rather than 37.8.  Flagged for the owner's hand-scan verdict.

59335 knob matrix (measured): fix (a) alone → 0.37 cm; fix (b) alone →
0.12 cm (the ~1 cm proton stub survives as its own protected segment); both
→ 0.37 cm.  Both ship ON.

### 9.5 Gates

- **Unit**: `wcdoctest-clus` 1148/1148 assertions (114 cases), incl. new
  `doctest_two_end_break.cxx` (synthetic two-Bragg scan: mid-track junction
  fires R1; near-far-end junction located; kinked weak-rise junction fires
  R2 at the bend; single-Bragg / flat-MIP / short-segment never fire) and
  wide-turn + knob-default cases.
- **Compiled config**: knobs-off `wct-pr-perevt.jsonnet` compile (full
  runner pipeline TLA) byte-identical to pre-change HEAD; knobs-on compile
  carries all three keys in the `TaggerCheckNeutrino` node.
- **Off-gate (byte-identical, final binary)**: `work-pr48-base48` (clean
  `20098cbf` binary, production env) vs `work-pr48-off48c` (final binary,
  three knobs forced 0): **48/48 mabc-pr.zip member-hash identical
  (`hash_archive.py`), nusel tsvs byte-identical.**  Same on ncpi0:
  `work-pr48-base19n` vs `work-pr48-off19nc`: **19/19 + nusel identical.**
  (Earlier-binary off-gates `-off48`/`-off48b`/`-off19n`/`-off19nb` also
  passed 48/48 + 19/19.)  Harness note: finishing individual events with a
  second runner invocation overwrites the batch-merged `nusel-*.tsv` with
  only its own events; regenerate with the runner's own
  `nusel_extract.py --merge` call over all per-event tsvs before comparing.
- **Footprint (knobs on, final binary)**:
  - nueCC48 `work-pr48-on48c` vs base48: **8/48 movers, all F2-WALK refit
    ripple, nusel identical** (owner-scanned set — no verdict flips).
  - ncpi0 `work-pr48-on19nc` vs base19n: **4/19 movers (1 TEB-BREAK,
    1 F3-PROTECT, 2 F2-WALK), nusel identical** (owner-scanned set).
  - 1k `work-pr48-on1kc` vs the production arm `work-pr47f-on1k`:
    **69/1000 movers — 37 TEB-BREAK, 1 F3-PROTECT, 31 F2-WALK;
    nusel-events + nusel-table diffs 0/1000.**

### 9.6 Movers, individually examined

Full raw evidence: `scripts/analysis/pr48/mover_census.py` (class per mover,
break diagnostics from the arm's log) and
`scripts/analysis/pr48/teb_case_exam.py` (per-break main-vertex before/after
vs the break point).  Classes: **TEB-BREAK** = the two-end break fired
(designed population), **F3-PROTECT** = a protected kink-break vertex
(59335 fix b), **F2-WALK** = neither log line — refit ripple from the walk
stopping at a Bragg-hot terminus kink (59335 fix a).  WCT log lines can tear
mid-write, so both scripts match on the line head only.

**Owner-scanned movers (nueCC48 + ncpi0-19 + first-50): no nusel verdict
flips anywhere.**  The 12 nueCC48/ncpi0 movers: 8 + 2 F2-WALK ripple, one
ncpi0 TEB break (105946: genuine 44.9 cm + 3.4 cm two-Bragg split, rises
4.7/5.3x MIP — selection kept the event's real nu vertex, the unselected
degree-2 break vertex re-merged into a pre-existing vertex 2.9 cm away, and
the near-region vertex set is identical to baseline: the designed decline
path), one ncpi0 F3 protect (285567, vertex survives in the final dump).
First-50 movers (8): the four motivating events (§9.4), 59335 (main vertex
refined 0.82 cm), 50831 (TEB fired on a genuine 9.7/7.5 cm two-Bragg split,
rises 5.0/4.4x MIP; selection declined, main vertex unmoved), 54341 (F3
protect on a non-main cluster, main vertex unmoved), 58717 (TEB break,
main vertex refined 2.1 cm onto the junction).

**The 31 new (unscanned) TEB cases, each examined** (break diagnostics +
main-vertex displacement; `teb_case_exam.py` output archived in the doc's
review round):

- **20 relocate the main vertex onto the junction** — the designed win.
  Displacements 2.4-176 cm; the final vertex lands 0.3-5.5 cm from the
  break point (final `do_multi_tracking` refit shifts it, same as 57485's
  2.7 cm).  The two large jumps were spot-checked on the dQ/dx profile:
  349461 (176 cm onto the junction of a 192 cm track + 4.6 cm stub, both
  ends Bragg-rising) and 491483 (60 cm; junction dip 0.86x cluster median).
- **10 declined-and-healed** — the break fired on a genuine two-end-rise
  topology, `determine_main_vertex` kept its original choice, and the
  unselected degree-2 break vertex was re-absorbed downstream; nusel and
  the near-region structure match baseline.  This is the designed decline
  path (the protect flag defends the vertex from *erasure while it can
  still win selection*; an unselected junction vertex is not load-bearing).
- **1 flagged for owner hand-scan — 321173**: the break is genuine
  (junction dip 0.76x cluster median, termini 2.0x and 3.3x — a real
  back-to-back split of a 151 cm track into 126/26 cm arms), but selection
  declined the junction AND flipped the main vertex from the 2.0x terminus
  to the 3.3x terminus (149 cm apart; the label stays nu-candidate in both
  arms).  Neither baseline nor pr/48 puts this vertex at the junction, so
  this is not a regression — but it is the one mover where the new choice
  is not obviously at-least-as-good, and the 1k reco1 file is
  truth-stripped (no `sim::` branches), so no truth adjudication was
  possible.  Left for the owner's scan.

Population sanity: 37 TEB breaks / 1000 events (3.7%) sits between the §5
census's raw two-end-rise gate (2.2% of segments at 1.3x with no acceptance
tier) and its looser turn-assisted tail (up to 1.25% + the rise-only
population) — the designed order of magnitude, not a blowout.

### 9.7 Files touched (toolkit)

- `clus/inc/WireCellClus/{PRVertex,PRSegment}.h` — `kProtectedBreak`,
  `kTwoEndBreakArm` flag bits.
- `clus/inc/WireCellClus/PRSegmentFunctions.h`, `clus/src/PRSegmentFunctions.cxx`
  — `TwoEndBreakOptions/Result`, `segment_two_end_break_scan`,
  `segment_wide_turn_angle`, `segment_search_kink` walk-stop knob +
  accept-criterion out-param.
- `clus/inc/WireCellClus/NeutrinoPatternBase.h`, `clus/src/NeutrinoPatternBase.cxx`
  — knob members, `break_two_end_dqdx` driver, `find_proto_vertex`
  particle_data plumbing, F3 protect at the C4/A0 break site.
- `clus/src/NeutrinoStructureExaminer.cxx` — protect-flag checks.
- `clus/src/NeutrinoTrackShowerSep.cxx` — teb-arm outward-direction restore.
- `clus/src/NeutrinoVertexFinder.cxx` — DL-rerank protected-vertex guard.
- `clus/inc/WireCellClus/TaggerCheckNeutrino.h`, `clus/src/TaggerCheckNeutrino.cxx`
  — component knobs + threading.
- `cfg/pgrapher/common/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — 4-layer knob
  threading, key-suppressed; SBND operating point.
- `clus/test/{doctest_two_end_break.cxx,doctest_prsegment.cxx,doctest_clus_knob_defaults.cxx}`.

## Files

- This doc.
- `scripts/analysis/pr48/backtoback_census.py` — the census script (§5).
- `scripts/analysis/pr48/census_output.tsv` — its raw tail-population output.
- `scripts/analysis/pr48/on1k_compare.py` — archive movers + per-mover vertex
  diff + nusel diffs (§9.5).
- `scripts/analysis/pr48/mover_census.py` — mover classification (§9.6).
- `scripts/analysis/pr48/teb_case_exam.py` — per-break main-vertex
  examination (§9.6).
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

# doc pr/67 — why the fitted trajectory does not cover the image (four owner cases, isochronous)

**Status: DIAGNOSIS ONLY, per the owner's explicit scope ("For this round, we want
to investigate and understand, no need to implement the fix yet"). No production
behavior is changed.** Two new config knobs ship: one is a log-only probe
(`traj_cover_probe`, inert by construction), the other is a diagnostic
counterfactual (`pr_find_other_rounds`, inert only at its default 0 — it changes
reconstruction output when set, and is not proposed for production).

> **ROUND 2 (below, §9) SUPERSEDES THE HEADLINE AND §3/§4/§6 OF ROUND 1.**
> Round 1 stopped at the first-segment endpoint and the branch-round budget.
> Round 2 instruments the *inside* of `find_other_segments` (P5) and every
> `remove_segment` call (P6), and finds that in **all four** events a branch
> candidate **is** proposed at the owner's charge and is killed by one of two
> filters — an isolated-residual floor counting 25 Steiner terminals, or the
> step-8 "faked" cut. Read §9 first; §3–§6 are kept as the record of how the
> conclusion was reached and of the measurements that still stand.

**Headline (round 1): the four cases are not one problem.** Three of them
(18264-137238, 18259-42280, 18345-21073) are *not* trajectory-stage failures at
all — the owner's charge is associated, within 1 cm, to a segment the chain
classified as an **EM shower**, which by design receives only a short trunk fit.
The fourth (18255-58717) is a genuine track-coverage defect, and it has a clean,
specific cause: two independent gates, each individually reasonable, both exclude
this object, so nothing ever tries to push its trajectory to the end of the track.

**Both of the owner's stated hypotheses were tested and both came back negative
for these four events.** The endpoint algorithm is not what stops the trajectory
in three of the four; the trajectory is not being removed after the fact in any of
them; and the branch-search round budget — the owner's own suggestion for
137238 — is provably not the constraint: **tripling it from 2 to 6 rounds leaves
the W-plane coverage of all four owner clusters bit-for-bit identical.**

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild > /home/xqian/tmp/pr67_build.log 2>&1; echo rc=$?
ls -la local/lib/libWireCellClus.so          # freshness proof (M1)
./build/clus/wcdoctest-clus                  # 176/176, 1854 assertions

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# baseline (pre-probe binary, HEAD c955ca52)
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67-base48 data 137238 42280
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr67-base19 data 21073
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr67-base1k data 58717

# probe-OFF gate arms (post-probe binary, knobs off)   -> work-pr67-g48b/g19b/g1kb
# probe-ON arms                                        -> SBND_TRAJ_COVER_PROBE=1, work-pr67-p48/p19/p1k
# 6-round counterfactual                               -> + SBND_PR_FIND_OTHER_ROUNDS=6, work-pr67-r6a/r6b/r6c

# the W-plane coverage instrument (the owner's own criterion, as a number)
python3 scripts/analysis/pr67/pr67_wcover.py work-pr67-base48/pr_evt137238 \
    --owner-point -122.0 22.5 423.2 --ql-dir work-nuecc48-cb0805/ql_evt137238
```

## 1. Reading the owner's coordinates correctly (do this first, always)

The owner reads coordinates off the Bee **`img-global`** layer, which doc pr/13
established is the only raw layer. `clustering-global`, `shower_track-global` and
`track_fit-global` are T0-corrected and carry a **per-cluster** drift-x offset.
Measured this round, at the owner's own points:

| event | local dx (img → clustering) |
|---|---|
| 137238 | +0.45 cm |
| 42280 | −0.92 cm |
| 21073 | −0.02 cm |
| 58717 | −0.73 cm |

Small here — but in that same 58717 event a cosmic-tagged cluster carries
**dx = +99.5 cm**. Comparing the two frames directly is therefore not merely
imprecise, it can land you in a different object entirely. `pr67_wcover.py`
does the mapping and prints the offset it used.

**Bee index ↔ event, confirmed rather than assumed.** Each owner point was located
in each candidate index's own `img-global` (hits at 0.05–0.67 cm), which pins the
mapping independently of any index file: set `9e2a1a1e` = nuecc48-prod0811
(`event/15` = 137238, `event/4` = 42280), `13900b8c` = ncpi0-prod0811
(`event/1` = 21073), `f8203fcd` = mcp1k50-prod0811 (**`event/37`** = 58717, the
owner's corrected link; `bee_idx 38` is a different event, 59003).

The owner's first 58717 coordinate (152.5, 93.1, 24.8) landed in cluster 27, which
has no PR segments at all; the corrected point (195.7, 76.4, 42.1) lands in
cluster 21, which does. All 58717 numbers below use the corrected point.

## 2. The instrument

`scripts/analysis/pr67/pr67_wcover.py` turns the owner's detection criterion — "I
cannot have track trajectory covering the W plane channels" — into a number. It
projects the cluster's image points and its fitted polylines (densely resampled,
per segment, so a sparse straight fit does not fake a hole) into the event's own
per-(apa,face) W wire index via `oc53_probe.Loader.wind()` (doc pr/53's fit against
the event's real `ctpc_a<A>f0pW` arrays), and reports W channels — and
(channel, drift-x bin) cells — that carry image charge but no fit projection.

**APA assignment, stated because the channel numbers depend on it.** The
per-(apa,face) channel fit is selected by `apa = 0 if x < 0 else 1` on the
T0-corrected x. That rule is **not universally exact**: measured on these events'
own blobs (`wpid >> 4`, `WirePlaneId.cxx:37`), the two APAs overlap in
x ∈ [−33.5, +33.5] cm, and the sign rule agrees with the blobs' own `wpid` on only
82–99 % of all points. It is nevertheless correct for all four target clusters
here, verified by range exclusion rather than assumed: apa0's blobs span
x ≤ +33.5 and apa1's span x ≥ −33.5, and each target cluster's full x range
(137238: [−133.9, −110.1]; 42280: [12.1, 28.1]; 21073: [−42.4, −24.9];
58717: [182.9, 198.6]) lies inside exactly one of them. **No target cluster
straddles x = 0**, so image points and fit points at the same place always receive
the same APA and the uncovered counts are internally consistent regardless.
A cluster that did straddle it would need per-blob `wpid` instead — noted in the
script.

Baseline numbers, production settings:

| event | cluster | W channels with charge | **uncovered** | (chan, x-bin) cells | **uncovered** | longest uncovered W run |
|---|---|---|---|---|---|---|
| 137238 | 143 | 131 | **0 (0.0%)** | 521 | 71 (13.6%) | — |
| 42280 | 8 | 362 | **9 (2.5%)** | 2262 | 624 (27.6%) | ch 294–302 (9), 90 img pts |
| 21073 | 60 | 108 | **19 (17.6%)** | 790 | 247 (31.3%) | ch 1222–1238 (17), 222 img pts |
| 58717 | 21 | 34 | **1 (2.9%)** | 132 | 22 (16.7%) | ch 130 (1), 13 img pts |

21073 is the visually worst case and the numbers agree: a 17-channel contiguous
block of collection-plane charge with no trajectory over it.

## 3. Per-case findings

Geometry of each owner cluster (own PCA, 2 %-trimmed drift extent):

| event | cluster | main? | n img pts | axial L | trimmed xext | principal axis vs drift |
|---|---|---|---|---|---|---|
| 137238 | 143 | associated (main is 7) | 1438 | 45.4 cm | 20.9 cm | 63.3° |
| 42280 | 8 | **main** | 14671 | 112.0 cm | 10.0 cm | 85.1° |
| 21073 | 60 | associated (main is 11) | 4062 | 37.9 cm | 12.5 cm | 78.6° |
| 58717 | 21 | **main** | 236 | 19.0 cm | 15.0 cm | 39.7° |

That "main?" column matters and was nearly missed: in 137238 and 21073 the owner's
charge is in an **associated** cluster, not the main one. Any diagnostic or fix
scoped to the main cluster cannot reach either case. (This round's first
counterfactual was scoped that way and produced a false negative; it was widened
and re-run — see §4.)

### 3.1 — 18264-137238 @ (−122.0, 22.5, 423.2): a shower, and the chain says so

> **Superseded by §9.3.1.** A branch candidate *was* proposed at this charge
> (bbox 1.48 cm away) in both rounds; it was routed away from the owner's arm
> and then dropped as an isolated residual. The `q = 15000` observation below
> stands as a fact about the final state, but it is not the reason no
> trajectory is there.

**Symptom.** Owner: "why the fitted track trajectory is missing this piece (likely
a track). Is that limited by not sufficient round of doing the branch searching?"
Nearest fit point is 4.43 cm away and is an **interior** point of segment 143042,
not an endpoint.

**Root cause.** The charge at that point *is* associated, at **0.08 cm**, to
segment 143061 — and every associated point within 6 cm of the owner's point
carries `q = 15000`. That is the shower marker, verified at the writer rather than
taken from a docstring: `clus/src/MultiAlgBlobClustering.cxx:880` is literally
`const double charge = is_shower ? 15000.0 : 0.0;`. Segment 143061 holds **1436
associated points and 19 fitted points**. The trajectory stops because the object
stopped being treated as a track, not because an endpoint fell short or a search
ran out of rounds.

*Read the 1436 : 19 ratio with doc pr/55's correction in mind*: a shower member's
`associate_points` are filed under the **shower's start-segment id**
(`MultiAlgBlobClustering.cxx:858`), so 143061's 1436 is an aggregate over the whole
shower, not one segment's own cloud, and the ratio overstates any single segment's
coverage. It is quoted here only as the scale of "associated but not fitted"; the
classification verdict rests on the `q` value, which is per-point and unaffected.

**Supporting negatives.** The endpoint story does not fit: the owner's point is
13.3 cm from the nearest first-segment endpoint, i.e. mid-cluster, not at a tip.
The branch-budget story does not fit either: round 1 added 4 segments, **round 2
added 0** — the search had already converged inside the production budget. And
`pr67_wcover.py` reports **0 %** of this cluster's W channels fully uncovered:
some fit crosses every channel that carries charge. What is missing is coverage in
drift-x within those channels (13.6 % of cells), which is the signature of a
trajectory passing through a wide object, not stopping short of one.

**Why it hid.** `iso_endpoint` was rejected here by the **`xext_frac`** gate
(0.409 > 0.35) — this cluster's drift extent is 41 % of its length — and until this
round only the *aspect* rejection logged, so the cluster looked as if the branch
had never been consulted. (An earlier reading of this event's log attributed it to
the aspect gate; that line belongs to a different cluster, 7. Anchoring on the
owner's actual cluster is what corrected it.)

**Proposed next step (not a fix here).** The open question is upstream of the
trajectory: **should 143061 have been classified as a shower?** That is a
track/shower-separation question, not an endpoint or fitting one, and it deserves
its own round with its own evidence (dQ/dx profile, PCA linearity per doc pr/63's
discriminator). No knob in the trajectory code will change this case.

### 3.2 — 18259-42280 @ (12.1, −13.6, 89.0): the pr/24 residual, and it is a shower boundary

> **Superseded by §9.3.2 and §9.4.** The 3.4 cm is *not* mostly lateral
> centring: 3.02 cm of it is axial, and the cause is the ±3 cm band's missing
> axial penalty (§9.4). A 4.21 cm branch covering the owner's charge to within
> 0.70 cm was fitted and discarded (§9.3.2).

**Symptom.** Nearest fit point is 3.83 cm away and **is** the endpoint of segment
8026 — a genuine axial undershoot at a tip.

**Root cause.** This is the known, already-documented residual of doc pr/24
rounds 4–6, unchanged: `iso_endpoint` fires correctly
(`L=114.6 cm, xext=10.0 cm, aspect=0.300`) and picks A = (14.94, −12.71, 91.73),
which sits **3.4 cm** from the owner's point. That 3.4 cm is the deliberate
lateral-centering offset `find_iso_first_segment_endpoints` exists to pay: among
points within a 3 cm band of the axial extreme it takes the one nearest the
principal axis, precisely so the pick is not a sheet-edge corner (doc pr/24 §17.3).
`v3_extension_guard` fires once, correctly, as designed.

Beyond that endpoint the charge is claimed by segment **8050**, `q = 15000`
(shower), with 3302 associated points and 19 fitted — same picture as 137238.

**Why it hid.** Nothing new hid; doc pr/24 §18.4 predicted this residual and §18.6
recorded it as open. This round confirms it is stable at HEAD and adds the reason
the region past the tip has no trajectory: it is shower-classified.

### 3.3 — 18345-21073 @ (−30.9, 25.5, 369.7): one polyline cannot cover a 2-D sheet

**Symptom.** The worst of the four by the owner's own criterion: 19 of 108 W
channels (17.6 %) carry charge with no trajectory over them, in a 17-channel
contiguous block holding 222 image points. Owner: "it is an iso case, the final
track trajectory does not cover the entire W region signal, this may come from the
initial track segment not ideal?"

**Root cause — and it is not the initial segment.** `iso_endpoint` **fired**
correctly on this cluster (`L=45.9 cm, xext=12.5 cm, aspect=0.634, n=3950`,
`walk=0.00/0.00`), giving endpoints A = (−39.32, 40.51, 394.42) and
B = (−30.25, 19.81, 372.37). B is 6.3 cm from the owner's point and sits at the
corner of the uncovered block — i.e. the endpoint reached the object's edge; it did
not stop short along the axis.

The governing fact is the geometry: **aspect 0.634**. This object is nearly as wide
as it is long — a genuine filled 2-D sheet, not a track. A single fitted polyline
cannot cover a sheet's width no matter where its endpoints are, and the coverage
gap is transverse, not axial. The charge at the owner's point is associated at
0.92 cm to segment **60055**, again `q = 15000` (shower), 1831 associated points
against 39 fitted.

**Why it hid.** The symptom looks exactly like an endpoint failure on a Bee
plane view, and this cluster genuinely *is* isochronous (78.6° from drift), so
"iso endpoint problem" is the natural reading. The `iso endpoint: fired` line with
`walk=0.00/0.00` is what rules it out.

### 3.4 — 18255-58717 @ (195.7, 76.4, 42.1): the one real track-coverage defect

> **Amended by §9.3.4.** Both gates below are real and the P2 finding stands,
> but the residual here is **92 % transverse** (−1.22 cm axial, +2.97 cm
> perpendicular), so an axial extension recovers nothing: **0 of 236** image
> points lie beyond the endpoint along the segment direction. F1 (§6) is
> retracted on that measurement.

**Symptom.** Nearest fit point 2.58 cm from the owner's point; the trajectory
stops before the end of the charge. Owner: "in the main cluster where a couple
tracks were fitted out, but missing something, presumably due to isochronous."

**Root cause — two gates, stacked, neither aware of the other.** Unlike the other
three, the charge here is associated (0.37 cm) to segment **21002** with
**`q = 0` — a TRACK**. So this one is a real defect, and it has two causes that
compound:

1. **`iso_endpoint` never runs.** New probe line:
   `pr67 iso endpoint: rejected by min_length (value=21.357 cut=40.000 L=21.4 cm n=236)`.
   The cluster is 21.4 cm long; the gate requires 40 cm. It falls back to the legacy
   wire-footprint boundary metric plus local-PCA refinement, which places the first
   segment's endpoint at (196.68, 77.45, 39.08) — **3.2 cm short** of the owner's
   point. Before this round, that rejection produced *no log line at all*: only the
   aspect gate logged, so this cluster was indistinguishable from one the isochronous
   branch had never been offered.

2. **The recovery stage that exists to fix exactly this is a no-op here.**
   `examine_vertices_3` calls `get_local_extension` to push an endpoint further
   out. That function opens with
   (`NeutrinoStructureExaminer.cxx:2426`):

   ```cpp
   if (std::fabs(angle - 90.0) < 7.5) {   // angle = local Hough dir vs drift
       return wcp;                        // unchanged
   }
   ```

   A direction perpendicular to drift **is** the isochronous case. New probe line,
   at precisely the endpoint from (1):

   ```
   pr67 get_local_extension: NO-OP, drift angle 89.5 deg is within 7.5 deg of
   perpendicular (isochronous) at (196.68,77.45,39.08) cluster 21
   ```

   0.5° from perpendicular. The one stage that could have extended this endpoint
   declined, structurally, because the track is isochronous *at that endpoint*.

   Note the subtlety: this cluster's **global** principal axis is only 39.7° from
   drift — it is not an isochronous object overall. It is isochronous *locally, at
   the tip*, which is what `get_local_extension`'s Hough estimate sees. A
   cluster-level isochronous gate would not catch this; the two scales disagree.

**Why it hid.** Both gates are individually defensible — `min_length` keeps the
iso branch off short objects where its PCA is unreliable, and the 7.5° band avoids
a meaningless direction estimate. Neither logs when it declines, and no stage
downstream asks "is the end of this track actually covered?". §5 records that no
such check exists anywhere in the fitter.

**Prototype status (M15).** The 7.5° early return is faithful to the prototype
(`PR3DCluster_path.h:288-316`, same band). This is a prototype limitation, not a
port error — the same shape as doc pr/24 round 5's `v3_extension_guard` finding.

## 4. The owner's two hypotheses, tested

> **Scope correction, §9.5.** Hypothesis (b) is tested here only for point-level
> *trimming* (P3) — that result is unchanged, and round 2's segment-level probe
> (P6) is also negative. What neither watches is a branch rejected as a
> *candidate* before `add_segment`, which is what actually happens in all four
> events (§9.2).

**Hypothesis (a): the endpoint-finding algorithm is not robust for the isochronous
case.** *Partly true, but not the cause in three of four cases.* It fired
correctly on 42280 and 21073 (with `walk=0.00`, i.e. the axial-extreme search never
had to retreat). It was **gated out** on 137238 (`xext_frac`) and 58717
(`min_length`). Only on 58717 does that gating actually cost coverage.

**Hypothesis (b): the trajectory was produced and then removed.** *Not supported.*
The new P3 probe reports every amputation by `examine_end_ps_vec`, the chain's
primary end-trimmer. Near the owner's points the trims are:

| event | trims in event | largest end move within 12 cm of owner point |
|---|---|---|
| 137238 | 18 | 0.73 cm |
| 42280 | 193 | 0.58 cm |
| 21073 | 107 | 0.57 cm |
| 58717 | 8 | 2.52 cm |

Sub-centimetre in three cases and 2.5 cm in the fourth — too small to account for
3–6 cm shortfalls. Nothing large was fitted and then deleted. (The P3 line also
prints a segment id, but it comes back `-1` on every call, so these are
*per-call* amputations located by geometry, not attributed to a named segment.
Fixing the id would need a rebuild the analysis did not require.)

**The owner's 137238 sub-hypothesis: "not sufficient rounds of branch searching".**
*Conclusively disconfirmed.* `find_proto_vertex`'s `nrounds_find_other_tracks` is
hardcoded at all three `TaggerCheckNeutrino` call sites (2 for the main cluster,
2 for associated, 1 for the third pass) with no config surface. The new
`pr_find_other_rounds` knob overrides it. Per-round census at production settings
versus 6 rounds:

| event | cluster | round 1 | round 2 | rounds 3–6 (counterfactual) |
|---|---|---|---|---|
| 137238 | 143 | +4 | **0** | 0, 0, 0, 0 |
| 42280 | 8 | +26 | +6 | 0, 0, 0, 0 |
| 21073 | 60 | +7 | +3 | 0, 0, 0, 0 |
| 58717 | 21 | 0 | 0 | 0, 0, 0, 0 |

Two clusters were still adding segments in round 2, which looked like a live lead —
but round 3 adds nothing in every case, and **`pr67_wcover.py` on the 6-round arms
returns numbers identical to production for all four clusters**: same segment
lists, same fit-point counts, same uncovered channel and cell counts. The budget is
not the constraint.

Honest caveat: the 6-round `mabc-pr.zip` archives *do* differ from production for
3 of 4 events even though no segments are added, because `find_other_segments` also
refits in place. The difference is elsewhere in those events, not in the owner's
clusters — which is why the per-cluster coverage comparison, not the archive hash,
is the measurement that answers the question. `pr_find_other_rounds` is therefore a
behavior-changing diagnostic and is **not** proposed for production.

## 5. Surfaced, not chased

Recorded because they bear on this symptom class; **not** investigated or changed
this round, per scope.

* **No stage anywhere measures trajectory-vs-W-charge coverage.** Coverage is used
  only to *penalize* cells (`fit_blob_coverage`'s foreign-ghost deweight, doc pr/49),
  never to *pull* a trajectory toward uncovered collection-plane charge. There is no
  "extend until the W charge is exhausted" term in the fitter. The owner's detection
  criterion has no counterpart inside the code.
* **`count_live_channels_between` is half-open where the prototype is inclusive.**
  Toolkit `Facade_Cluster.cxx:3961` loops `wire_index < wire_max`; prototype
  `PR3DCluster_path.h:525-532` loops `temp_index <= temp_max_u`. The bounds here are
  two *points'* wire indices, not a blob's half-open `[min, max)` range, so M7's
  "never fix `<` to `<=`" does not obviously apply — but the divergence is **not**
  in `porting_dictionary.md`. Per M15 both readings are recorded and neither is
  picked: either the toolkit is one wire short on every boundary-metric pair (a
  uniform offset on a score whose entire non-`|dx|` dynamic range is these counts —
  which is all of it in the isochronous limit), or the half-open form is a
  deliberate unrecorded convention. Owner's call.
* Two further `get_two_boundary_wcps` divergences reported during exploration and
  not verified this round: a per-(apa,face) pure-3D-distance override
  (`Facade_Cluster.cxx:3858-3884`) that can re-introduce corner picking on
  multi-face objects, and an all-uninitialized return of `{(0,0,0),(0,0,0)}` when no
  blob clears the 1500-charge cut (the prototype's `!flag_init` fallback appears
  unported). Flagged for a future round; verify before acting.

## 6. Proposed fixes — knob shape and validating gate, none implemented

> **Superseded by §9.6 and §10** (§10 is the staged plan).
> **F1 below is retracted** (§9.3.4). F2 stands. F3's premise is weakened by
> §9.2.

Only 58717 has a defect this round can propose a fix for. The other three need
different rounds in different subsystems.

**F1 — `get_local_extension` isochronous fallback (targets 58717 cause 2).**
When the local Hough direction lands in the ±7.5° perpendicular band, fall back to
a direction the isochronous case can actually supply — the segment's own
end-to-end direction, or a per-plane projected extension — instead of returning the
vertex unchanged. *Knob shape:* `v3_iso_extension_fallback` (bool, C++ default
false) plus an angle-band parameter, threaded exactly like `v3_extension_guard`
(same file, same call site, same `TaggerCheckNeutrino` plumbing); composes with
rather than replaces the existing retraction guard. *Validating gate:* probe-off
byte-identity on nueCC48 + NCpi0 + mcp1k manifests via `hash_archive.py` on
`mabc-pr.zip`, then a knob-on census in which every mover is attributable to a new
sentinel line ("0 unclaimed"), plus `pr67_wcover.py` before/after on 58717.
*Risk to state up front:* `examine_vertices_3` touches only the main cluster's two
initial termini, so the blast radius is small — but it is shared production code and
the prototype has the same gap (M15), so it must ship default-OFF like
`v3_extension_guard` did.

**F2 — `iso_endpoint_min_length` for short isochronous tracks (targets 58717
cause 1).** The knob already exists (`iso_endpoint_min_length`, currently the C++
default 40 cm); 58717's cluster is 21.4 cm. *This is a tuning change, not new code*,
so the "fix" is a scan, not an implementation: sweep 40 → 30 → 20 cm and measure
both the recovered coverage and the cost. *Validating gate:* the existing
`iso_endpoint` gate discipline from doc pr/24 §15 — the round-3 mid-track
break-point regression detector (`pr24_iso_probe.py --junctions`) is the specific
thing to re-run, because lowering the length gate admits shorter, noisier objects
into the branch, which is exactly the failure round 3 had to fix. Not obviously
safe; do not lower it without that scan.

**F3 — the three shower cases.** No trajectory-side fix applies. The question is
whether segments 143061 / 8050 / 60055 should have been classified as showers at
all. That is a track/shower-separation round; it should start from the dQ/dx and
PCA-linearity discriminators rather than from the trajectory code.

## 7. Gates run this round

| check | result |
|---|---|
| freshness proof (M1) | `libWireCellClus.so` 06:52:20 > last source edit 06:51:49 |
| `./build/clus/wcdoctest-clus` | **176 test cases / 1854 assertions, all pass** |
| probe-OFF vs pre-change baseline, `mabc-pr.zip` member-content hash (M2) | **PASS 4/4** — `work-pr67-g48b/g19b/g1kb` vs `work-pr67-base48/base19/base1k` |
| probe-ON vs probe-OFF, same hash | **PASS 4/4** — the probe is log-only, confirmed empirically not just by inspection |
| compiled-config proof, knobs off | **byte-identical** to the pre-change tree (`cmp` on `wcsonnet` output, git-stash round-trip); neither new key present |
| compiled-config proof, knobs on | both keys appear with correct values (`traj_cover_probe: true`, `pr_find_other_rounds: 6`) |
| baseline provenance | zero commits between the baseline binary and the probe binary (`git log c955ca52..HEAD` empty); only pr/67 files modified — so the gate is a genuine probe-inertness claim |

**Gates deliberately NOT run, and why.** No `abtest` PDHD/PDVD gate. The claim
that earns this is the empirical one: **probe-ON vs probe-OFF is hash-identical on
all four events**, and every new jsonnet parameter is key-suppressed when off
(compiled-config `cmp`, above), so the probes are inert by construction and
measurement rather than by argument. A reachability argument would be weaker than
it first looks — `TrackFitting::examine_end_ps_vec`, which carries P3, is also
reached from the STM fitter path, not only from `TaggerCheckNeutrino` — which is
exactly why the inertness claim is grounded in the hash gate instead. No
48/19/1000-event manifest sweep: the
owner scoped this round to four events, and the probe-off byte-identity claim is
carried by the compiled-config proof plus the 4-event hash gate. A production flip
of either knob would need the full manifest gate first — neither is proposed.

## 8. What this round leaves behind

* This doc, and `scripts/analysis/pr67/pr67_wcover.py` (the owner's W-plane
  criterion as a reusable number).
* Toolkit: `traj_cover_probe` (log-only; P1 iso-gate rejection reasons and
  first-segment seed provenance, P2 `get_local_extension` perpendicular no-op,
  P3 `examine_end_ps_vec` end amputations, P4 per-round branch-search census) and
  `pr_find_other_rounds` (diagnostic counterfactual). Both default OFF.
* Arms: `work-pr67-{base,g,p,r6}*`. The `g*`/`base*` pairs are the gate record;
  `p*` back every probe line quoted here; `r6*` back §4's counterfactual.
* Open for the owner to direct: F1 vs F2 for 58717; whether the three
  shower classifications in §3.1/3.2/3.3 are correct (a separate round); the
  `count_live_channels_between` divergence in §5 (M15, both readings recorded).

---

# 9. Round 2 — inside `find_other_segments`: the branch **is** found, then filtered out

Round 2 was prompted by three owner questions on the round-1 write-up:

1. *137238* — "why was this track segment missed, so that the track trajectory was
   not fitted? I understand the W channels are fully covered."
2. *42280* — "why is the initial end point not at the edge?"
3. *58717* — "do you have some idea on how to fix it?"

Round 1 could not answer (1): its per-round census (P4) reported only the segment
count before and after each `find_other_segments` call, which cannot distinguish
"no candidate existed there" from "a candidate was scored and rejected". Round 2
adds that missing resolution and the answer is the same in all four events.

## 9.0 Repro (round 2)

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild > /home/xqian/tmp/pr67b/build2.log 2>&1; echo rc=$?
./build/clus/wcdoctest-clus                       # 176/176, 1854 assertions

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# probe-ON  -> work-pr67-r48 / r19 / r1k     (SBND_TRAJ_COVER_PROBE=1)
# probe-OFF -> work-pr67-s48 / s19 / s1k     (gate arms, new binary, knobs off)
PR_JOBS=2 SBND_TRAJ_COVER_PROBE=1 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67-r48 data 137238 42280
PR_JOBS=1 SBND_TRAJ_COVER_PROBE=1 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr67-r19 data 21073
PR_JOBS=1 SBND_TRAJ_COVER_PROBE=1 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr67-r1k data 58717

grep 'pr67 fos' work-pr67-r48/pr_evt137238/wct_pr_evt137238.log | grep cluster=143
grep 'pr54 isolated-residual drop' work-pr67-r48/pr_evt42280/wct_pr_evt42280.log
```

## 9.1 What round 2 adds

Both are extensions of the existing `traj_cover_probe` knob (still default OFF,
still log-only, gate PASS in §9.7).

* **P5 — `find_other_segments` component census** (`NeutrinoOtherSegments.cxx`).
  One line per connected component of untagged Steiner terminals, at each of the
  three decision points that can kill it — the step-8 quality cut, the step-9
  selection/routing, and the step-9 **re-evaluation** (a component that survived
  step 8 is re-scored *in 2D only* against the segment just added). Each line
  carries `npts`, `len`, `nnf` (`number_not_faked`), `max_dis` per plane, the
  routed endpoints `A`/`B`, and the component's **bounding box** — the bbox is
  what lets a hand-scanned coordinate be matched to a component at all.
* **P6 — `remove_segment` sentinel** (`PRGraph.cxx`). Every call logs the
  segment's fit-point count and both fitted endpoints. This tests the owner's
  hypothesis (b) at the **segment** level, which round 1's P3 (point-level end
  trimming) structurally could not.

## 9.2 The common mechanism

In **all four** events a branch candidate is proposed at the owner's charge. None
survives. Distances below are from the owner's point (mapped to the clustering
frame, §1) to the component's bounding box, and — for the three that were routed
and fitted — to the fitted branch's own `v1→v2` chord.

| event | cluster | component at the owner's charge | bbox dist | fate |
|---|---|---|---|---|
| 137238 | 143 | 10 terminals, routed 4.70 cm, fitted 7.75 cm, nnf 6→5 | 1.48 cm | routed away from the owner's arm (§9.3.1), then **isolated-residual drop** |
| 42280 | 8 | 10 terminals, routed 4.38 cm, fitted 4.21 cm, nnf 9 | 0.00 cm | **isolated-residual drop** |
| 21073 | 60 | 14 terminals, routed 4.34 cm, fitted 6.15 cm, nnf 2→1 | 0.65 cm | **isolated-residual drop** |
| 58717 | 21 | 2 terminals, 2.93 cm, nnf 0 | 0.47 cm | **step-8 `nnf0_short`** — never routed, never fitted |

The three isolated-residual drops are identical in shape, and each fires twice
(once per branch-search round — which is the mechanical reason the round budget
cannot help, independently of round 1's counterfactual):

```
pr54 isolated-residual drop: cluster 143 n_points=10 length=7.75 cm dir_mag=4.70 cm
     v1=(-122.6,24.1,418.7) v2=(-122.3,26.7,414.8) cm
pr54 isolated-residual drop: cluster 8   n_points=10 length=4.21 cm dir_mag=4.38 cm
     v1=(14.3,-10.4,89.8)   v2=(12.1,-13.9,88.4) cm
pr54 isolated-residual drop: cluster 60  n_points=14 length=6.15 cm dir_mag=4.34 cm
     v1=(-31.5,24.1,369.5)  v2=(-32.8,26.1,365.9) cm
```

**Three gates stack, and every one of them is sized above these objects.**

1. **Neither endpoint attaches to the existing graph.** `find_vertex_other_segment`
   → `check_end_point` tries three widening passes, topping out at
   `vtx_cut1 = 1.5 cm`, `vtx_cut2 = 3.0 cm` (6.0 cm for a degree-1 vertex), plus
   a `< 90°` direction test; the segment-snap path uses `sg_cut1 = 2.0 cm`. The
   branch ends are **1.8 – 8.8 cm** from the nearest fitted point of the same
   cluster. So `v1_existed == v2_existed == false` and the code enters the
   isolated-segment path.
2. **The isochronous rescue on that path never runs.** `modify_vertex_isochronous`
   / `modify_segment_isochronous` — machinery that exists for exactly this
   topology — is behind
   `dir_mag > 10 cm || (dir_mag > 8 cm && track_length > 13 cm)`.
   The three branches have `dir_mag` **4.70 / 4.38 / 4.34 cm**. Not one of them
   is large enough to be offered to it.
3. **pr/54's keep-isolated escape hatch rejects them.** SBND production has
   `other_seg_keep_isolated = true` with both thresholds `null`, i.e. the C++
   defaults `min_points = 25`, `min_length = 3.0 cm`. The lengths (7.75 / 4.21 /
   6.15 cm) all pass; **the point counts (10 / 10 / 14) all fail.** Note that
   `number_points` here counts **Steiner terminals of the component**, not fitted
   points — on a 4–8 cm branch that is an implicit length floor far above the
   advertised 3 cm `min_length`, which is why the floor does not bind the way
   pr/54's text reads.

All three branches are near-perpendicular to drift (86.3° / 59.7° / 72.5°), i.e.
this is the isochronous regime the owner identified.

## 9.3 Per case

### 9.3.1 — 137238: the component is found, but one route covers only one arm

The component at the owner's charge (P5 `group=6`, round 1; `group=2`, round 2 —
same bbox both times) spans
`[-122.94,-122.01] x [22.15,26.65] x [414.82,421.72]`, whose nearest face is
**1.48 cm** from the owner's point. So the search did see this charge.

But it is routed away from it. Step 9 runs a single `do_rough_path(special_A,
special_B)`, where `special_A` is the component's **boundary connection point**
(the terminal carrying an MST edge across the tagged/untagged frontier) and
`special_B` is simply the **farthest point from `special_A`**. Here
`A = (-122.63,24.05,418.72)` sits in the *middle* of a nearly straight ~8 cm
object: `B = (-122.32,26.65,414.82)` is 4.70 cm away on one side, and the
owner's four Steiner terminals (z ≈ 420.8–421.7, y ≈ 22.1–22.8) are ~3.3 cm away
on the other — **170° apart as seen from `A`**. One polyline from `A` to `B`
traverses one arm. Measured: the owner's point projects to `t = -0.98` on the
`v1→v2` chord, i.e. **off the end**, closest approach **4.78 cm**; the fitted
branch's own near end lands 3.57 cm short of it.

Then the branch dies anyway (§9.2). So 137238 has **two** independent reasons to
produce nothing there, and fixing only the isolated-residual floor would leave
the owner's charge ~4.8 cm from the nearest trajectory. This is the same *class*
as round 1's "one polyline cannot cover a 2-D sheet" (§3.3), relocated from the
first-segment endpoint to the component-routing step — and it is consistent with
round 1's measurement that **0 %** of this cluster's W channels are fully
uncovered.

### 9.3.2 — 42280: the branch does cover the owner's charge

`t = 0.79` on the chord — **interior** — closest approach **0.70 cm**. This is
the cleanest case in the set: a 10-terminal, 4.21 cm branch with **nnf = 9 of
10** (i.e. 9 of its points are un-shadowed on ≥2 planes, so it is well supported
in 2D, not a projection artefact) is fitted and then discarded because 10 < 25.

### 9.3.3 — 21073: covered, but weakly supported

`t = 0.07` — interior — closest approach **1.52 cm**. Same drop, but the quality
is much lower: **nnf = 2 of 14, falling to 1 of 14** on re-evaluation. Whatever
floor is chosen in §9.6, this one is the hard case: it is genuinely hard to tell
from a 2-D shadow of the neighbouring trajectory.

### 9.3.4 — 58717: the residual is transverse, not axial

Two findings, and the second **retracts round 1's proposed fix F1**.

*The branch never gets routed.* The component 0.47 cm from the owner's point has
**2 terminals, 2.93 cm, nnf = 0** — it dies at the step-8 quality cut
(`nnf0_short`), one stage earlier than the other three. Two terminals with no
un-shadowed points is below any defensible floor; there is no branch-level fix
here.

*The trajectory does not stop short — it runs alongside.* Decompose the
first-segment endpoint `E = (196.68,77.45,39.08)` against the owner's point
`(196.43,76.4,42.1)`:

| component | value |
|---|---|
| along the segment direction | **−1.22 cm** (the owner's point is *behind* E) |
| perpendicular to it | **+2.97 cm** |
| total | 3.21 cm |

The residual is **92 % transverse**. Across the whole of cluster 21, **no image
point is more than 2.93 cm from a fitted trajectory** (median 0.57 cm, 90th
percentile 1.92 cm) — the fit threads one side of a ~5 cm-wide tip blob (112
image points within 6 cm of the owner's point, bbox 5.4 × 5.6 × 3.1 cm).

Round 1's **F1** (`v3_iso_extension_fallback`: when the local Hough direction
lands in the ±7.5° perpendicular band, extend along the segment's own end-to-end
direction instead of returning the vertex unchanged) was tested here by direct
computation and **would recover nothing**. The segment direction is well
conditioned — 44.1° from drift, versus the 89.5° the Hough estimate returns, so
the fallback is well posed exactly where the guard fires — but the max-projection
search it feeds finds the endpoint is *already* the extreme: **0 of 236 image
points lie beyond E** along that direction (furthest is −0.22 cm). **Any
axial-extension fix is the wrong shape for this event.** The P2 finding itself
(`get_local_extension` is a structural no-op in the isochronous case,
`NeutrinoStructureExaminer.cxx:2426`, prototype-faithful) stands as a real
limitation — it is simply not what costs coverage here.

## 9.4 The 42280 endpoint question, answered

Owner: *"why is the initial end point not at the edge?"*

Not lateral centring, and not the Steiner snap. Measured against cluster 8's own
principal axis (85.1° from drift, 14671 image points, axial span 112.0 cm):

| quantity | value |
|---|---|
| axial coordinate of the image extreme | `s = 61.73` |
| axial coordinate of the owner's point | `s = 61.03` |
| axial coordinate of the chosen endpoint `A` | `s = 58.01` |
| **`A` is inside the extreme by** | **3.73 cm** |
| image points beyond `A` axially | **93 of 14671** |
| `A`'s own distance from the axis (`d_perp`) | **0.28 cm** |
| distance from `A` to the Steiner terminal it snapped to | **0.005 cm** |
| Steiner terminals available beyond `A` | 10, reaching the full `s = 61.73` |
| inward walk of the spike guard (probe `walk=`) | **0.00 / 0.00 cm** |

So: the terminals reach the tip, the snap costs nothing, the spike guard does not
walk in, and `A` is essentially *on* the axis — lateral centring did exactly its
job. The 3.73 cm is the **±3 cm axial band itself**. `pick_end_point` finds the
true axial extreme and then replaces it with the *laterally most central
qualified point anywhere within ±3 cm axially of it*. On a sheet the most central
point in that band is typically ~3 cm inside the tip, and **the band's selection
carries no axial penalty term at all** — a point 3 cm inside and 0.1 cm off-axis
beats a point at the tip and 0.4 cm off-axis. Reproduced directly: taking the
image extreme and applying the band rule by hand lands at `s = 58.97`,
`d_perp = 0.75`, i.e. 2.8 of the 3.7 cm.

This is a real trade, not a bug: doc pr/24 §17.3 introduced the band specifically
to stop the endpoint landing on a sheet-*corner*, and the probe line records the
branch still gaining **11.3 / 12.2 cm** over what round 2 of pr/24 would have
picked. What it does not do is stop 3 cm short of costing anything.

The charge past `A` was then recovered anyway — as the 4.21 cm branch of §9.3.2,
0.70 cm from the owner's point — and dropped. **The owner is looking at a piece
that the chain both found and fitted, twice, and threw away both times.**

## 9.5 P6 came back negative — hypothesis (b) still stands

Round 1 reported that nothing was fitted and then removed, on the basis of P3
(`examine_end_ps_vec`, point-level end trimming). That statement was about
*trimming* and it is unchanged: sub-centimetre in three events, 2.5 cm in the
fourth.

P6 now tests the stronger claim — whether a whole fitted segment was deleted —
and it is **also negative**. Across the four events (44 / 30 / 31 / 2
`remove_segment` calls), every removal within 4 cm of an owner point belongs to
the ordinary first-segment break/refit cycle: each shares its far endpoint with
the others while the fit-point count steps down (137238: 82 → 92 → 90 → … → 26),
which is `break_segments` splitting one long trajectory, not a branch being
deleted. **No completed segment is deleted at the owner's charge in any of the
four.**

What round 2 changes is therefore a matter of *scope*, not a reversal: the branch
is rejected as a **candidate**, before `add_segment` is ever called, which is a
stage neither P3 nor P6 watches and which round 1 had no instrument for.

## 9.6 Fix candidates — what each one actually recovers

> **Carried forward into §10**, which turns these into a staged plan with knob
> shapes, gates, an order and abandon conditions. §10.2 adds a fourth candidate
> (the endpoint band's missing axial penalty) that this table does not have.

Still **no implementation**, per scope. What is new is that each candidate can now
be scored against all four events instead of argued.

| candidate | 137238 | 42280 | 21073 | 58717 |
|---|---|---|---|---|
| **G1** widen the isochronous-snap gate (`dir_mag > 10` / `>8 & len>13` → ~4 cm) | reaches it (4.70) | reaches it (4.38) | reaches it (4.34) | no — never routed |
| **G2** lower `other_seg_keep_isolated_min_points` (25 → ~10) | insufficient alone | **yes** | yes, but admits nnf 1/14 | no |
| **G3** gate keep-isolated on `nnf` instead of raw `number_points` | borderline (5–6/10) | **yes** (9/10) | **no** (1–2/14) | no |
| **G4** route every arm when `special_A` is interior to its component | **the missing half** | n/a | n/a | no |

**G1 is the strongest lead.** The isochronous attachment machinery already exists
and is written for precisely this topology; the only reason it never runs on any
of these three is a size gate set at roughly twice their length. Attaching a
branch beats keeping it isolated — an attached branch becomes part of the graph
and is refit jointly, whereas the keep-isolated path (G2/G3) adds a disconnected
piece. **Untested**: whether `modify_*_isochronous` actually succeeds on a 4.5 cm
branch is unknown, and a lower gate also exposes it to genuinely fake 2-D
coincidences. It needs its own knob, its own census and its own gate.

**G2 alone is blunt.** It recovers 42280 cleanly, but on 21073 it would admit a
branch supported by 1–2 un-shadowed points out of 14, and on 137238 it recovers a
branch whose route misses the owner's charge anyway.

**G3 is the discriminator the code already computes and then ignores** —
`other_seg_keep_isolated_ok` sees only `component_points` and `track_length`,
never `number_not_faked`, even though `number_not_faked` is exactly "how many of
these points are *not* explained as a 2-D shadow of an existing trajectory". It
separates 42280 (9/10) from 21073 (1/14) correctly. It is not sufficient by
itself for 137238.

**G4 targets 137238 only** and is the one thing that would put a trajectory on
that charge. `special_B` is defined as the farthest point from `special_A`; when
`special_A` is interior to its component, the far side is never routed.

**58717 needs none of these.** §9.3.4's decomposition says the residual is 92 %
transverse — the trajectory is displaced ~3 cm sideways within the isochronous
ambiguity rather than stopping short, and round 1's F1 is retracted. Round 1's
**F2** (lowering `iso_endpoint_min_length` from 40 cm so the 21.4 cm cluster can
use the isochronous endpoint branch at all) is untouched by round 2 and remains
the only live idea for it, with the same warning: it needs doc pr/24 §15's
`pr24_iso_probe.py --junctions` regression detector first.

**Not a fix, and worth stating: none of this is upstream of the shower
classification in a causal sense.** `find_other_segments` runs inside
`find_proto_vertex`; the track/shower decision happens much later. Round 1's
observation that three of the four owner points carry `q = 15000` is a fact about
the final state, and the ordering is what is established here — not that the
dropped branch caused the classification.

## 9.7 Gates (round 2)

| check | result |
|---|---|
| freshness proof (M1) | `libWireCellClus.so` 07:20:54 > last source edit 07:20:04 |
| `./build/clus/wcdoctest-clus` | **176 test cases / 1854 assertions, all pass** |
| probe-OFF (new binary) vs round-1 baseline, `mabc-pr.zip` member hash (M2) | **PASS 4/4** — `work-pr67-s48/s19/s1k` vs `work-pr67-base48/base19/base1k` |
| probe-ON vs probe-OFF, same hash | **PASS 4/4** — `work-pr67-r48/r19/r1k` vs `work-pr67-s48/s19/s1k`; P5 and P6 are log-only, confirmed by measurement |

Same deliberate omissions as §7, for the same reason: probe-ON and probe-OFF are
hash-identical on all four events, so inertness rests on measurement rather than
on a reachability argument.

## 9.8 Caveats carried by round 2

* **P6's flag is a process-global file-static.** `remove_segment` is a free
  function with call sites in six files and no access to
  `PatternAlgorithms::m_traj_cover_probe`, so `PR::set_traj_cover_probe()`
  mirrors the knob into a `static bool` in `PRGraph.cxx`, set from
  `TaggerCheckNeutrino::configure`. With per-face `TaggerCheckNeutrino`
  instances the last `configure` wins. Harmless here — every instance got the
  same value — but a configuration with one face on and another off would not
  behave as written.
* **`seg->id()` returns −1 throughout the PR graph**, so both P3 and P6 locate
  their events geometrically rather than by segment id. Every attribution in §9
  is by coordinate.
* **P5's component bbox is over Steiner terminals**, which are sparser than the
  image points; the bbox is a lower bound on the component's true extent.

---

# 10. Proposed fix plan

**Nothing here is implemented.** This section exists so the owner can authorise
stages individually. Every stage below is a default-OFF knob whose off-state is
byte-identical, and every stage names the gate that would validate it and the
condition under which it should be abandoned.

## 10.1 The bar every stage must clear

Unchanged from §4 of the operating manual, restated because it drives the
ordering:

1. Knob-off byte-identical (`hash_archive.py` member hash) on the standard SBND
   manifests — nueCC48, NCpi0-19, mcp1k-117 — plus the compiled-config proof.
2. Knob-on census in which **every** mover is attributable to a new sentinel
   line ("0 unclaimed"), the bar pr/65 set.
3. `pr67_wcover.py` before/after on the four owner clusters — the owner's own
   detection criterion, as a number.
4. `nusel` stability on nueCC48 (48/48) and doctests for any pure predicate.

## 10.2 Recovery matrix — what each stage actually fixes

Every cell is measured, not estimated; the measurements are in §9 and §10.7.

| stage | 137238 | 42280 | 21073 | 58717 |
|---|---|---|---|---|
| **S1** keep-isolated on evidence (`nnf`) | partial — branch kept, but it routes the wrong arm (§9.3.1) | **yes** (nnf 9) | **no**, and correctly (nnf 2→1) | no — never routed |
| **S2** let the isochronous snap see short branches | attaches (4.70 cm) | attaches (4.38) | attaches (4.34) | no — never routed |
| **S3** route both arms of a component | **the missing half** | n/a | n/a | no |
| **S4** axial penalty in the endpoint band | n/a — iso branch gated out by `xext_frac` | **yes, on its own** | **no** — measured, pick unchanged for every weight tested | no |
| — | | | | **nothing proposed** (§10.8) |

Two overlaps are worth seeing before choosing: **S4 alone resolves 42280** (it
moves the endpoint to within 0.55 cm of the owner's point, after which the branch
of §9.3.2 is no longer needed and would be tagged as covered), and **S2 preempts
S1** on the same branches — a branch that the isochronous snap attaches never
reaches the keep-isolated test at all.

## 10.3 S1 — keep an isolated residual on evidence, not on point count

*Targets:* 42280 fully, 137238 partially. *Site:*
`other_seg_keep_isolated_ok` (`PRSegmentFunctions.cxx:32`, declared
`PRSegmentFunctions.h`) and its one call site
`NeutrinoOtherSegments.cxx:808`.

The predicate today is `component_points >= 25 && track_length >= 3.0 cm`.
`number_not_faked` — the count of the component's points that are *not* explained
as a 2-D shadow of an existing trajectory — is computed a few lines above, sits
in the same `temp_segments[…]` struct the call site already reads, and is
**ignored**. It is exactly the discriminator this decision needs. At the moment
of the keep-isolated test the four values are:

| event | `number_points` | `number_not_faked` | today | with an `nnf` clause |
|---|---|---|---|---|
| 42280 | 10 | **9** | dropped | kept |
| 137238 | 10 | **5** | dropped | kept |
| 21073 | 10→14 | **2 → 1** | dropped | still dropped |

*Knob:* `other_seg_keep_isolated_min_nnf`, int, **C++ default 0 = clause
disabled**. The predicate becomes a strict widening — an OR arm, never a new
rejection:

```cpp
return (component_points >= min_points && track_length >= min_length)
    || (min_nnf > 0 && number_not_faked >= min_nnf && track_length >= min_length);
```

Note the explicit `min_nnf > 0` test: with a bare `nnf >= min_nnf` a default of 0
would accept everything, so the disabled state has to be checked, not implied.

*Scan:* `min_nnf` over 3–8; 4 is the natural starting point (margin 1 below
137238's 5, margin 2 above 21073's 2). Do not tune on four events — the scan is
for the census, not for these.

*Gate:* the §10.1 four, plus **new revert-proven doctest cases** in the existing
`clus/test/doctest_other_seg_keep_isolated.cxx` — this is a pure function, so it
is the cheapest test in the whole plan and there is no excuse for shipping it
without one.

*Risk:* low, and this is the reason to do it first. It rides a path that is
**already in SBND production** (`other_seg_keep_isolated = true`); it only lowers
the admission bar. A kept residual is added as a *disconnected* piece, so it
cannot corrupt vertex topology the way an attachment can.

*Abandon if:* the knob-on census shows the kept residuals are predominantly
short 2-D coincidences (which `nnf` is supposed to prevent), or nueCC48 `nusel`
moves.

## 10.4 S2 — let the isochronous snap see short branches

*Targets:* all three routed branches. *Site:* `NeutrinoOtherSegments.cxx:721`.

```cpp
if (dir_mag > 10 * units::cm ||
    (dir_mag > 8 * units::cm && segment_track_length(new_seg) > 13 * units::cm)) {
```

Everything inside this block — `modify_vertex_isochronous`,
`modify_segment_isochronous` — exists to connect a branch that is displaced from
its parent *because* the topology is isochronous. It is the correct machinery for
all three of our cases and **it never runs on any of them**: their `dir_mag` is
4.70 / 4.38 / 4.34 cm against a gate at 8–10 cm.

*Knob:* `iso_snap_min_dir_mag`, double cm, **C++ default 10.0 = legacy**
(replacing the literal in the first clause only; the widening tiers at `>18 cm`
and `>36 cm` are untouched).

*Why this is safer than it looks, and what must be checked first:* the vertex-snap
path applies its own isochronous test — the connecting vector must be within 15°
of perpendicular-to-drift — so lowering the *size* gate does not remove the
*isochronous* requirement. **Verify before implementing** whether
`modify_segment_isochronous` carries an equivalent angle test; if it does not,
S2 needs its own angle guard rather than relying on the caller.

*Gate:* the §10.1 four, plus a sentinel on every successful snap below the legacy
gate (cluster, `dir_mag`, which of the four snap paths fired, both endpoints), so
the census can attribute movers.

*Risk:* medium-high and genuinely unknown. Whether a 4.5 cm branch can be snapped
stably has never been measured, and attaching changes graph topology — vertices,
degrees, and therefore downstream vertex selection. This is the stage most likely
to produce movers that are hard to attribute.

*Abandon if:* snaps below the legacy gate succeed but produce vertices that
`main_vertex` then prefers, i.e. if the fix moves neutrino vertices rather than
just adding coverage.

## 10.5 S3 — route both arms when the boundary point is interior

*Targets:* 137238 only, and it is the only thing that puts a trajectory on that
charge. *Site:* `NeutrinoOtherSegments.cxx` step 8/9.

`special_A` is the component's boundary connection point; `special_B` is defined
as merely *the farthest point from `special_A`*. When `special_A` is interior to
its component, one `do_rough_path(A,B)` traverses one arm and the other is never
routed. In 137238 the two arms are **170° apart** as seen from `A` and the
owner's charge is on the unrouted one (§9.3.1).

*Knob:* `other_seg_route_both_arms` (bool, default false) +
`other_seg_arm_min_length` (double cm, default 3.0). When on: after `special_B`
is chosen, find `special_C` = the farthest point from `A` among points with
`(p − A)·(B − A) < 0`; if `|A − C| >= arm_min_length`, emit a second candidate
`A→C` through the same step-9 machinery.

*Shape note:* prefer routing twice within one step-9 iteration over splitting the
component in step 7 — the component is the unit for `ncounts`, `sep_clusters`,
`map_connection` and the re-evaluation loop, and splitting it perturbs all four.

*Gate:* the §10.1 four. Determinism needs explicit attention: the far-arm search
must have a deterministic tie-break exactly like `pick_end_point`'s.

*Risk:* medium. It roughly doubles the candidate count in branched components,
and every candidate costs a `do_single_tracking` plus a `do_multi_tracking`, so
this stage has a real CPU cost that must be measured (`timecmd.py`, per §4 of the
manual).

*Do not start this before S1/S2 report*, because if the branch is going to be
dropped anyway, routing its second arm changes nothing.

## 10.6 S4 — give the endpoint band an axial penalty

*Targets:* 42280 on its own; this is the direct answer to the owner's "why is the
initial end point not at the edge?". *Site:* `pick_end_point` inside
`find_iso_first_segment_endpoints` (`NeutrinoPatternBase.cxx`, the
"laterally most central point of the end band" loop).

Today the band picks `argmin(d_perp)` over all qualified points within ±3 cm
axially of the extreme, with **no axial term** — so a point 3 cm inside and
0.1 cm off-axis beats a point at the tip 0.4 cm off-axis.

*Knob:* `iso_endpoint_band_axial_weight`, double, **C++ default 0.0 = today,
byte-identical**. The band pick becomes
`argmin(d_perp + w · (s_extreme − s_k))` (signed so the penalty is on axial
*retreat*).

*Measured response* (§10.7, cluster 8, all image points):

| `w` | picked `s` | `d_perp` | gain | distance to the owner's point |
|---|---|---|---|---|
| 0.0 (today) | 58.97 | 0.75 | — | 2.50 cm |
| 0.2 – 0.5 | 60.97 | 1.08 | +2.01 cm | **0.82 cm** |
| 0.8 – 3.0 | 61.42 | 1.41 | +2.45 cm | **0.55 cm** |

The response **saturates**: even at `w = 3` the pick never leaves 1.41 cm of the
axis, though the band contains points out to 3.56 cm. That flat plateau is the
safety argument — the knob is not sitting on a cliff. Propose `w = 0.5` as the
scan centre, range 0.2–1.0.

On 21073 the same scan leaves the pick **completely unchanged at every weight**,
because the owner's charge there is 6.11 cm *off* the axis — confirming from a
second direction that 21073's gap is transverse, not axial.

*Gate:* the §10.1 four **plus, mandatorily, doc pr/24 §15's
`pr24_iso_probe.py --junctions` regression detector.** This stage changes the
first segment of every cluster the iso branch accepts, which is the widest blast
radius in the plan. The specific failure to watch for is pr/24 round 3's: a
mis-placed endpoint leaving a stub that `find_other_segments` then claims,
producing a spurious near-0° junction mid-track.

*Risk argument in its favour:* round 3's failure came from endpoints moving
**inward**; this moves them **outward**, which is the direction round 3 wanted,
and the round-2 probe already records the branch gaining 11.3/12.2 cm over the
pre-round-3 behaviour. Outward motion also preserves the property round 3 relied
on — that the endpoint "can never move inward, which is what makes 'cannot leave a
stub for find_other_segments' true by construction".

*Caveat on the numbers above:* they were computed on all image points of the
cluster, not on the charge-qualified subset the C++ uses, and on one cluster.
They establish the shape of the response, not the production value. The pr/24
§15 38-cluster set is what must set `w`.

## 10.7 Recommended order, and why

**S1 → S4 → S2 → S3**, with a stop-and-report after each.

**S1 first** not because it recovers the most, but because it is the experiment
that de-risks everything after it. S1 is the narrowest change in the plan
(strictly additive, on a path already live in SBND production, with a pure
predicate that can be unit-tested), and it answers the question every later stage
depends on: *is putting a trajectory on these small isochronous branches
desirable at all?* If the S1 census shows it is not — spurious segments, `nusel`
movers — then S2 and S3 are dead too, because they all end in the same place, and
the plan stops having cost almost nothing.

**S4 second**, because it independently resolves the one case with the clearest
evidence (42280), it directly answers the owner's Q2, and its risk is
characterised (flat plateau, safe direction of motion) — but it needs the pr/24
junction detector, which is real work to stand up.

**S2 third.** Best physics if it holds — an attached branch beats a disconnected
one — but the largest unknown in the plan and the one most likely to move
vertices.

**S3 last, and conditionally.** Only if 137238's class still matters after S1 and
S2, since it costs CPU and touches the component bookkeeping.

## 10.8 What is deliberately NOT proposed

* **No change to `get_local_extension`'s 7.5° perpendicular band**
  (`NeutrinoStructureExaminer.cxx:2426`). It is prototype-faithful (M15) and
  §9.3.4 measured that extending along the axis there recovers nothing.
  Round 1's **F1 stays retracted.**
* **No fix for 58717.** The residual is 92 % transverse: the trajectory is
  displaced ~3 cm sideways within the isochronous ambiguity, and the only
  component near the charge has 2 terminals with `nnf = 0` — i.e. in 2-D it is
  genuinely indistinguishable from the trajectory already there. Addressing it
  would need a *fitter-level* term that pulls a trajectory transversally toward
  uncovered collection-plane charge, which is a far larger change than this
  round's evidence supports. Recorded as the open item it is.
* **No lowering of `iso_endpoint_min_length`** (round 1's F2). It is orthogonal
  to everything above and still untested; keep it as a separate item with its own
  scan, not folded into this plan.
* **No change to the "faked" thresholds** (`search_range = 1.5 cm`,
  `scaling_2d = 0.8`). These govern the tagging of every point in every cluster
  in every event; nothing measured here justifies touching a parameter with that
  reach.
* **No coverage-driven re-offer pass** (a post-`find_proto_vertex` audit that
  finds image regions far from any trajectory and re-proposes them). It is the
  most general answer to §5's observation that nothing anywhere measures
  trajectory-vs-charge coverage, and it is the wrong first move: it would be a
  new stage in the chain rather than a threshold on an existing one.

---

# 11. Round 3 — S2 implemented, validated, SBND PRODUCTION ON

**Status: `iso_snap_min_dir_mag` = 4.0 cm is SBND PRODUCTION ON — owner flip
2026-08-12** ("based on my scan, the overall is positive"), after the Bee
before/after hand-scan of the three targets *and* all eight collateral
neutrino-vertex movers of §11.7. The C++ knob default itself stays 10.0
(legacy); the flip is cfg-only, in `wct-pr-perevt.jsonnet`. See §11.12 for the
flip proof and the legacy escape.

It fixes all three in-scope events and passes every mechanical gate — off-gate
byte-identical 0/117, 0 unclaimed movers, 0 `nusel` selection flips. **The
accepted cost is in §11.7**: it also moves the reconstructed neutrino vertex on
30 of 117 events, 9 of them by more than 10 cm. That was S2's own §10.4 abandon
condition, and it was resolved the way this tree resolves such questions — by
the owner's hand-scan, not by a threshold. Read §11.7 before treating any future
vertex mover in this region as a regression.

## 11.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild
env -u LD_LIBRARY_PATH ./build/clus/wcdoctest-clus

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)

# baselines at HEAD 60bad894, captured BEFORE any edit
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67f-base48 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr67f-base19 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr67f-base50 data $M50
# off arm (knob at its 10 cm default) and on arm (4.0), same three roots
SBND_ISO_SNAP_MIN_DIR_MAG=4.0 PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67f-on48 data

python3 scripts/analysis/pr49/on_compare.py work-pr67f-base48 work-pr67f-off48   # off-gate
python3 scripts/analysis/pr67/pr67_wcover.py work-pr67f-on48/pr_evt137238 \
    --owner-point -122.0 22.5 423.2 --ql-dir work-nuecc48-cb0805/ql_evt137238
```
Census and vertex scripts: `/home/xqian/tmp/pr67f/{pr67_census,nuvtx,wcover_sweep}.py`.

## 11.1 Why S2 alone, and why first

§10 recommended S1 → S4 → S2 → S3, chosen to de-risk. Both legs of that
argument are spent:

* **The stated prerequisite is closed** (§11.2), so S2 is no longer the stage
  with the largest unknown.
* **21073 is reachable only through S2.** §10.2 measured it: S1 correctly drops
  that component (`nnf` 2→1) and S4 leaves its endpoint pick unchanged at every
  weight tested. "Fix these three" therefore *makes S2 mandatory*, and §10's
  order puts the mandatory stage third.

Owner selected S2-only. S1 and S4 remain documented fallbacks that between them
fix 2 of 3 (42280 twice over, 137238 partially) and never reach 21073.

## 11.2 §10.4's open prerequisite, closed by reading the code

§10.4 required verifying "whether `modify_segment_isochronous` carries an
equivalent angle test; if it does not, S2 needs its own angle guard". It does,
and the two paths are **complementary, not redundant**:

| path | distance guard | isochronous angle guard |
|---|---|---|
| `modify_vertex_isochronous` | caller, `< 6 cm` (`:740`, `:744`) | **caller only** — `\|angle(d1)−90°\| < 15°` at `:741`/`:745`; the callee has none |
| `modify_segment_isochronous` | caller `< 6 cm` (`:762`, `:765`) | **callee only** — `:1429`, parameterised `angle_cut`, default 15° |

So lowering the *size* gate never relaxes the *isochronous* requirement, and S2
needed no new angle guard. That is also why the widening tiers can pass
`angle_cut` 8 and 5: those tighten the callee's own test.

## 11.3 The knob, and the operating point

`iso_snap_min_dir_mag`, double **cm**, C++ default **10.0** = legacy, replacing
the literal in the **first clause only** of `NeutrinoOtherSegments.cxx:721`. The
second clause (`> 8 cm` && track length `> 13 cm`) and the `> 18 cm` / `> 36 cm`
widening tiers are deliberately untouched — deriving them from the knob would
put `0.8 × 100 = 80.00000000000001` on a candidate sitting exactly at 8 cm.

Sweep on the three events (`work-pr67f-p{off,8,6,5,4}{a,b}`), reporting
`owner point → nearest fit point` and uncovered W channels:

| knob | 137238 | 42280 | 21073 |
|---|---|---|---|
| off (10.0) | 4.39 cm, 0/131 | 3.83 cm, 9/362 (2.5%) | 4.81 cm, 19/108 (17.6%) |
| 8.0 | 4.39, 0/131 | 3.83, 9/362 | 5.63, 18/108 |
| 6.0 | 4.39, 0/131 | 3.83, 9/362 | **1.21, 0/108 (0.0%)** |
| 5.0 | 4.39, 0/131 | 3.83, 9/362 | **1.21, 0/108** |
| **4.0** | **0.64, 0/131** | **0.26, 1/362 (0.3%)** | **1.38, 1/108 (0.9%)** |

The three decisive branches sit at `dir_mag` **4.70 / 4.38 / 4.34 cm**, so the
gate must be below 4.34 to admit all three; 4.0 is the round value with margin.
Nothing here is tuned to these events beyond that.

## 11.4 The three events, fixed

At 4.0, all three clear the round-3 pass criterion:

| event | owner→fit | uncovered W | fit points | sentinel path |
|---|---|---|---|---|
| 18264-137238 | 4.39 → **0.64 cm** | 0/131 → 0/131 | 515 → 537 | `vertex_v1`, dir_mag 4.70 |
| 18259-42280 | 3.83 → **0.26 cm** | 9 → 1 (2.5%→0.3%) | 955 → 1101 | `vertex_v1`, dir_mag 4.38 |
| 18345-21073 | 4.81 → **1.38 cm** | 19 → 1 (17.6%→0.9%) | 760 → 828 | `segment_v1`, dir_mag 4.34 |

No segment came back with empty `fits()` in any of the three, so every recovered
branch actually reaches the Bee `track_fit` layer rather than sitting in the
graph contributing nothing.

**137238 is gated on the owner-point distance, not on coverage.** Its W coverage
was already 0/131 uncovered at baseline — its defect was never a missing W
channel but a trajectory 4.4 cm away in 3-D. Reading the percentage there would
have scored a no-op as a pass.

**S3 was never needed.** §10.5 proposed routing both arms of 137238's component
because `special_A` was interior and one route covered one arm. With S2 on, the
event recovers directly (4.39 → 0.64 cm), so the second-arm work — and the
candidate-doubling CPU cost it carried — is dropped. The `pr_find_other_rounds`
counterfactual planned as its cheaper substitute was not needed either.

## 11.5 Gates

| gate | result |
|---|---|
| off-gate, `hash_archive.py` member content, vs pre-edit HEAD baselines | **0/48, 0/19, 0/50** |
| off-gate `nusel` | 0/48, 0/19, 0/50 |
| freshness (M1) | `libWireCellClus.so` 11:52:04 > last source edit 11:51:12 |
| `wcdoctest-clus` | **180 cases / 1881 assertions**, 0 failed |
| compiled-config (M6) | key **absent** when null, `"iso_snap_min_dir_mag" : 4` when set |
| detectors touched | sbnd only (+ a null-defaulted param in `common/clus.jsonnet`) |

Baselines were captured at HEAD **before** any edit — there was no HEAD-era 19
or 50 arm, and capturing them afterwards would have voided the gate.

`pr24_iso_probe.py --junctions` was **not** run: it detects endpoint-motion
regressions and is inert for S2, which does not touch `pick_end_point`. It would
be mandatory for S4.

## 11.6 The census — 0 unclaimed

The sentinel `pr67 iso-snap below-legacy` fires on a successful snap the legacy
gate would have refused, so set *S* is exactly the knob's footprint.

| sample | movers | *S* | unclaimed | `nusel` flips |
|---|---|---|---|---|
| nueCC48 | 36/48 | 36 | **0** | 0/48 |
| NCπ⁰-19 | 14/19 | 14 | **0** | 0/19 |
| PR data 50 | 2/50 | 2 | **0** | 0/50 |

Movers equal *S* exactly in all three samples — no event changed where the code
path did not execute, and no event executed the path without changing. That is
the pr/65 bar met on the nose.

*Attribution caveat:* the sentinel's own condition is `dir_mag <= 10 cm`, which
also catches the **legacy second clause** (confirmed in the off arm, 42280,
`dir_mag=9.78 seg_len=14.88`). Rather than re-cut the proven-identical binary,
`pr67_census.py` re-classifies from the logged `dir_mag`/`seg_len`. The logged
length is post-snap, so the 8–10 cm window is a proxy; every target branch sits
at 4.3–4.7 cm, far below 8, so they are unambiguous.

Snap-path split, 138 newly admitted snaps over the 117: **segment 102, vertex 36**.

Conditioning — the §10.4/round-3 hazard was that `test_p` divides by `dir.x()`,
guarded only against exact zero, and `dir.x()` is small *by construction* for an
isochronously-displaced branch:

* vertex path, 39 logged snaps: `|vtx_new − test_p|` median **0.34 cm**, max
  2.51 cm — the kNN lands essentially on the projection, so the projection is
  being respected, not overridden. Only **1 of 39** has `|dir.x| < 0.05`.
* segment path: `|dir1.x|` median 0.32, min 0.0097 (one near-degenerate case).

So the *arithmetic* hazard largely did not materialise. What did is §11.7, and
it is a different problem.

## 11.7 The accepted cost — the neutrino vertex moves

Measured from `T_tagger`'s `nu_x/nu_y/nu_z`, the authoritative reconstructed
neutrino vertex (an earlier pass using the mode of the PF-root `start` values
was wrong — it read 18255-56982 as a 135 cm relocation when the true figure is
1.28 cm; that event changed *which* candidate is main, not its position):

| | knob 4.0 | knob 6.0 |
|---|---|---|
| events whose ν vertex moved | 30/117 | 19/117 |
| moved > 10 cm | **9** | **4** |
| moved 1–10 cm | 7 | 4 |
| moved ≤ 1 cm | 14 | 11 |
| largest move | **82.4 cm** (271851) | 82.2 cm (271851) |

**Target-event effects and collateral are different phenomena and must not be
averaged together.**

*Targets.* 137238's vertex does **not** move. 21073's moves 0.74 cm. 42280's
moves **62.7 cm** — from (20.6, 4.4, 150.2) to (14.4, −10.1, 89.5), i.e. **onto
the owner's own reported uncovered charge at (12.1, −13.6, 89.0)**. That is the
knob doing exactly what it was built to do, and the event's interpretation
changes with it: `e⁻ 165/305/922/1009 MeV + proton 5 MeV` becomes `e⁻ 1805 MeV +
5 γ + proton 71 MeV`. Whether that is a 62 cm improvement or a 62 cm regression
is the question in §11.11.

*Collateral.* Eight further events move > 10 cm with no owner complaint behind
them: 271851 (82.4), 180801 (45.9), 10550 (43.4), 521075 (31.5), 30504 (23.0),
111412 (22.4), 350186 (20.9), 46363 (11.6).

Raising the gate does not buy safety. At 6.0 the footprint shrinks (36/117
movers, 4 over 10 cm) but 271851 still moves 82 cm, 180801 still 45.9, and
111412 moves **further** at 6.0 (49.2 cm) than at 4.0 (22.4 cm) — different snap
sets produce different topologies, so the response is not monotonic. And 6.0
fixes only 21073. There is no operating point that fixes the three events and
leaves neutrino vertices alone; the movement is intrinsic to letting this
attachment machinery see short branches, not an artifact of the threshold.

Nor is it confined to the vertex path: 271851 (82 cm) and 180801 (45.9 cm) have
**zero** vertex-path snaps. Splitting a parent segment changes downstream
`main_vertex` selection just as effectively as relocating a vertex, so gating
the two paths separately would not have avoided this.

**Two bars, and the stricter one was invoked — then overruled on evidence.**
The owner's stated stop condition was a `nusel` **selection** flip, which did
**not** happen (0/117). The bar that held the flip back was §10.4's own
vertex-stability abandon condition, restated in the round-3 plan. The owner
hand-scanned the Bee sets of §11.11 — the three targets plus every collateral
mover — and judged the overall effect positive, so the flip proceeded. That is
the intended resolution path for a physics judgement, not a bypass of the gate.

Deliberately **not** done: no `min |dir.x()|` cut and no maximum-displacement
guard were added to suppress the movers. That would be a second behaviour change
hidden under one flag, and it would be tuning a parameter until the physics
number looked acceptable (manual §5.7).

## 11.8 The decisive measurement this round does NOT have

Whether the 9 collateral moves are regressions is not decidable from movement
magnitude. The number that would settle it is **distance to the true neutrino
vertex, off vs on**, for 42280 and each collateral mover — nueCC48 and NCπ⁰ are
MC, and prior rounds quote truth distances routinely (pr/47's 0.65 cm, pr/50's
2.6 mm). The source reco1 art files carrying `sim::MCTrack` truth are not in
this tree (`input_files_reco1/extracted-*` holds extracted frames only, and the
dump logs do not record the source path), so it was not measured here. It is the
first thing to do if the owner wants this flipped.

## 11.9 What shipped

Toolkit, all default OFF:

* `clus/src/NeutrinoOtherSegments.cxx` — the knob at `:721`; `snap_path`
  attribution; the `pr67 iso-snap below-legacy` sentinel; conditioning DEBUG
  lines in both `modify_*_isochronous` (they fire for legacy snaps too, which is
  what gave §11.6 its comparison distribution).
* `clus/inc/WireCellClus/{NeutrinoPatternBase,TaggerCheckNeutrino}.h`,
  `clus/src/TaggerCheckNeutrino.cxx` — the four plumbing hops. The component
  member is bare cm and is scaled once at the copy; declaring it
  `{10.0*units::cm}` there *and* scaling would give 100 cm and silently kill
  every isochronous snap in the OFF arm, visible only to the off-gate.
* `cfg/pgrapher/common/clus.jsonnet` + sbnd `clus.jsonnet` / `wct-pr-perevt.jsonnet`
  — null-suppressed key, 6 sites.
* `clus/test/doctest_clus_knob_defaults.cxx` — pins the 10.0 default, which is
  what makes the double-scaling trap a test failure rather than a silent gate.
* `run_pr_chain_batch.sh` — `SBND_ISO_SNAP_MIN_DIR_MAG` passthrough (wcp repo).

## 11.10 Open

* The truth-distance measurement of §11.8 — blocking for any flip.
* S1 (`other_seg_keep_isolated_min_nnf`) and S4
  (`iso_endpoint_band_axial_weight`) remain unbuilt (§10.3, §10.6). If S2 is
  abandoned they are the fallback, fixing 42280 and partially 137238, never
  21073. S1's cost is understated in §10.3: its 117-event `nnf` census is not
  free, because the `pr54 isolated-residual drop` line does not log `nnf`.
* 58717 remains out of scope (§10.8): 92% transverse residual.

## 11.11 Bee — before / after

Both arms are the **same binary**; the only difference is
`SBND_ISO_SNAP_MIN_DIR_MAG=4.0`. `before` = knob off (= production today).

| set | link |
|---|---|
| nueCC48 **before** | https://www.phy.bnl.gov/twister/bee/set/6309ef1b-091a-45e2-a107-2ab0fe508b14/event/list/ |
| nueCC48 **after** | https://www.phy.bnl.gov/twister/bee/set/6dddaf41-f9db-4304-9522-3e2643648668/event/list/ |
| NCπ⁰ **before** | https://www.phy.bnl.gov/twister/bee/set/e73ef5c7-e393-4e98-93d2-faada3f58bea/event/list/ |
| NCπ⁰ **after** | https://www.phy.bnl.gov/twister/bee/set/74f5e27a-b0c4-4374-bdcc-37890d0b0d4b/event/list/ |

`bee_idx` is the same in the before and after set of a pair, so
`.../set/<uuid>/event/<idx>` compares like for like.

| idx | event | set | what to look at |
|---|---|---|---|
| 0 | 18264-137238 | nueCC48 | **target** — trajectory reaches the charge, ν vertex unmoved |
| 1 | 18259-42280 | nueCC48 | **target** — ν vertex moves 62.7 cm onto the reported charge |
| 2 | 18255-271851 | nueCC48 | collateral, ν vertex 82.4 cm (largest) |
| 3 | 18253-10550 | nueCC48 | collateral, 43.4 cm |
| 4 | 18255-30504 | nueCC48 | collateral, 23.0 cm |
| 5 | 18255-111412 | nueCC48 | collateral, 22.4 cm (49.2 cm at knob 6.0) |
| 6 | 18255-350186 | nueCC48 | collateral, 20.9 cm |
| 7 | 18255-46363 | nueCC48 | collateral, 11.6 cm |
| 0 | 18345-21073 | NCπ⁰ | **target** — 17.6% → 0.9% uncovered W, ν vertex 0.74 cm |
| 1 | 18255-180801 | NCπ⁰ | collateral, 45.9 cm |
| 2 | 18255-521075 | NCπ⁰ | collateral, 31.5 cm |

The three targets are idx 0/1 (nueCC48) and idx 0 (NCπ⁰). The other eight are
the events §11.7 cannot call: they are the reason the knob is not flipped.

## 11.12 The flip — SBND production ON at 4.0 cm

**Owner flip 2026-08-12**, on the §11.11 hand-scan. One line of cfg, in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` only:

```jsonnet
iso_snap_min_dir_mag = 4.0,          // was null (= C++ default 10.0 = legacy)
```

`sbnd/clus.jsonnet` keeps `null` at all four of its sites, matching how
`es3_stub_guard`, `shower_absorb_unreachable_main` and
`other_seg_keep_isolated` are flipped in this tree: the PR chain compiles
`wct-pr-perevt.jsonnet` (via the thin `sbnd_xin/wct-pr-perevt.jsonnet` wrapper
that imports it through `WIRECELL_PATH`), and that is the only file production
reads. **The C++ default is untouched at 10.0** — any non-SBND consumer, and any
job that does not go through this cfg, still gets legacy behaviour.

### Flip proof

Bare production (no env override) must now BE the validated on-arm, and the
documented escape must restore the pre-flip bare exactly. Both measured on the
full 117 with `hash_archive.py` member content:

| claim | arms | result |
|---|---|---|
| bare flipped production == validated on-arm | `on{48,19,50}` vs `flip{48,19,50}` | **0/48, 0/19, 0/50** |
| legacy escape `-A iso_snap_min_dir_mag=10.0` == pre-flip bare | `off{48,19,50}` vs `esc{48,19,50}` | **0/48, 0/19, 0/50** |

```bash
# bare production, post-flip
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67f-flip48 data
# legacy A/B escape
SBND_ISO_SNAP_MIN_DIR_MAG=10.0 PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr67f-esc48 data
python3 scripts/analysis/pr49/on_compare.py work-pr67f-on48  work-pr67f-flip48   # 0/48
python3 scripts/analysis/pr49/on_compare.py work-pr67f-off48 work-pr67f-esc48    # 0/48
```

Compiled-config proof: bare yields `"iso_snap_min_dir_mag" : 4`;
`-A iso_snap_min_dir_mag=10.0` yields `10`.

So the A/B escape hatch is real and byte-exact in both directions — a future
round can recover the pre-pr/67 production output without reverting code.

### 18255-58717 after the flip — unchanged, and why that is correct

Asked for separately. 58717's `mabc-pr.zip` is **byte-identical** before and
after the flip (member hash `c40c7852…` both sides), **zero** `pr67 iso-snap
below-legacy` sentinels fired in it, and its numbers are static: owner point →
nearest fit **2.59 cm**, uncovered W **1/34 (2.9%)**.

That is §10.8 holding, not a failure: 58717's residual is **92% transverse**,
and the only component near the charge has 2 terminals with `nnf = 0` — in 2-D
it is genuinely indistinguishable from the trajectory already there, so there is
no branch candidate for the isochronous snap to attach. It needs a
*fitter-level* term pulling a trajectory transversally toward uncovered
collection-plane charge, which remains out of scope.

Bee (post-flip production):
https://www.phy.bnl.gov/twister/bee/set/a05bebe6-8e6f-4375-9ec5-b0f24919cef4/event/list/

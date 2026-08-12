# doc pr/67 — why the fitted trajectory does not cover the image (four owner cases, isochronous)

**Status: DIAGNOSIS ONLY, per the owner's explicit scope ("For this round, we want
to investigate and understand, no need to implement the fix yet"). No production
behavior is changed.** Two new config knobs ship: one is a log-only probe
(`traj_cover_probe`, inert by construction), the other is a diagnostic
counterfactual (`pr_find_other_rounds`, inert only at its default 0 — it changes
reconstruction output when set, and is not proposed for production).

**Headline: the four cases are not one problem.** Three of them
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

**Symptom.** Owner: "why the fitted track trajectory is missing this piece (likely
a track). Is that limited by not sufficient round of doing the branch searching?"
Nearest fit point is 4.43 cm away and is an **interior** point of segment 143042,
not an endpoint.

**Root cause.** The charge at that point *is* associated, at **0.08 cm**, to
segment 143061 — and every associated point within 6 cm of the owner's point
carries `q = 15000`, the Bee convention for **"painted shower"** (`q = 0` is
"painted track"). Segment 143061 holds **1436 associated points and 19 fitted
points**. That ratio is not a defect; it is what a shower classification means in
this chain. The trajectory stops because the object stopped being treated as a
track, not because an endpoint fell short or a search ran out of rounds.

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
3–6 cm shortfalls. Nothing large was fitted and then deleted.

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

**Gates deliberately NOT run, and why.** No `abtest` PDHD/PDVD gate: the C++ edits
are confined to `clus` code reached only through `TaggerCheckNeutrino`, which is
SBND-only in this tree, and all four new/edited jsonnet parameters are
key-suppressed when off (proven above). No 48/19/1000-event manifest sweep: the
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

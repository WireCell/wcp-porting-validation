# doc pr/98 — fit_exclusion: the -1.0 sentinel port bug, the §3.1 unit-question discharge, and the nueCC48 re-evaluation

Round start 2026-08-20.  Owner directive: `fit_exclusion` (prototype
`flag_exclusion`, pr/30 F1) is an important prototype feature — understand it,
fix any bug on that path, and re-evaluate on the nueCC sample.  Output change
is expected (the fit is low-level; the DL vertex and dQ/dx sit downstream);
the validation criterion is *is the fit visibly better near the vertex*
(Bee + pr_display 2D wire-plane panels), largest movers first.  nueCC48 only.

## 0. Repro

```bash
# toolkit @ 2ecfc630, wcp-porting-img @ (this commit)
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild > /home/xqian/tmp/pr98_build.log 2>&1; echo rc=$?     # build+install
./wcb --target=wcdoctest-clus && ./build/clus/wcdoctest-clus  # 215/215
# freshness proof: local/lib/libWireCellClus.so mtime > last source edit

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# knob-off gate arm (byte-identical claim):
PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr98-off-nuecc48 data
python3 scripts/pr85_hash_gate.py work-nuecc48-prod0819 work-pr98-off-nuecc48
# knob-on measurement arm:
SBND_FIT_EXCLUSION=true PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr98-fx-nuecc48 data
```

## 1. What the feature is (prototype ground truth)

`flag_exclusion` turns on per-plane 2-D charge arbitration between
overlapping segments of the SAME cluster, applied to the association sets
that feed `multi_trajectory_fit` and the downstream dQ/dx fit.  In
`form_map_multi_segments` (prototype `PR3DCluster_multi_track_fitting.h:820-821`),
for interior trajectory points only, after `form_point_association` and
before `examine_point_association`:

```cpp
if (flag_exclusion)
  update_association(temp_2dut, temp_2dvt, temp_2dwt, sg, segments);
```

`update_association` (prototype `:970-1097`) re-projects each claimed
(wire, time-slice) cell back to a (drift-x, transverse) point and keeps it iff

```
min_dis_track < min_dis1_track   // closer to THIS segment's fitted trajectory
|| min_dis_track < 0.3*units::cm // or within the unconditional 3 mm floor
```

where the distances are nanoflann 1-NN queries against each segment's
`pcloud_fit` (fitted trajectory; seeded from the raw path, so never null).
Without exclusion, two segments crossing the same wire region both claim the
same charge and a bright neighbour drags the other segment's fit — the
classic near-vertex multi-track failure, and the mechanism pr/96 §2.1
measured on 18255-70084 (uncovered vertex prong FIXED by
`SBND_FIT_EXCLUSION=true`: uncovered charge 19.1%→3.4%).

Prototype call census: 28 of 30 live `do_multi_tracking` sites pass `true`;
the only two `false` are both in `break_segments`
(`NeutrinoID_proto_vertex.h:722`, `:751` — "fit dQ/dx here, do not exclude
others": freshly-split collinear children must not arbitrate each other away
at the junction).  The toolkit's hard-false `break_segments` sites and the
single-segment local-graph site already match; the 34 knobbed sites are
gated by `m_fit_exclusion` (C++ default false).

Design properties worth keeping in mind when reading movers:

- **Intra-cluster only.**  Prototype filters competitors by
  `get_cluster_id()` (`:739-748`); the toolkit gets the same restriction for
  free because every knobbed call site passes a cluster filter and
  `get_segment_edges()` returns the per-cluster edge list when
  `m_cluster_filter` is set (`TrackFitting.cxx` `build_cluster_edges` /
  `get_segment_edges`).  Verified this round; a comment now records the
  invariant in `update_association`.
- **Interior points only; vertex points never excluded** (both sides).
- **Iterative, not self-consistent**: arbitration runs before each fit round
  against the PREVIOUS round's fitted clouds; a freshly added segment
  competes via its "main" (Steiner path) cloud — same semantics as the
  prototype's ctor-seeded `pcloud_fit`.
- **Admission side effects**: at `find_other_segments`
  (`NeutrinoOtherSegments.cxx:654/:843/:880`) the excluded fit feeds the
  `0.78·length / 10 cm / 1.6·MIP` acceptance triple, so the knob can add or
  remove segments, not just move fit points.
- **Toolkit-only third pass**: the pre-dQ/dx `form_map_graph` rebuild
  (pr/28 T4, `TrackFitting.cxx` `do_multi_tracking` dQ/dx stage) has no
  prototype counterpart and also receives `flag_exclusion`.  The prototype's
  dQ/dx consumes the round-2 exclusion-filtered associations (its
  `reset_fit_prop` is a resize), so passing the flag there is the
  parity-consistent choice; the observable is the "pre-dQ/dx form_map_graph
  dropped N trajectory points" debug line.

## 2. Bug: the -1.0 sentinel poisons the arbitration (FIXED)

**Symptom.**  Never directly observed — the path was dead (pr/30 F1: the
argument was dropped at every site).  When pr/30 §12 forced it on, 47/48
nueCC events changed, 7 nue candidates lost vs 3 gained, selected vertex
moved up to 97 cm; the round was declined without a mechanism-level read.

**Root cause.**  `DynamicPointCloud::get_closest_2d_point_info`
(`clus/src/DynamicPointCloud.cxx:372-374`) returns a **-1.0 sentinel** when
the queried (plane, face, apa) kd-tree is empty — which is the normal state
for a competitor segment lying entirely on the other drift face (the kd2d
accessor lazily creates an empty tree, `:78-81`, and `NFKDVec::knn` on an
empty index returns no results).  `segment_get_closest_2d_distances`
(`clus/src/PRSegmentFunctions.cxx:207-239`) guards only a *wholly empty*
cloud (→ 1e9); a cloud with points on the other face passes the sentinel
through.  `TrackFitting::update_association` consumed it unguarded:

- competitor answers -1.0 ⇒ `min_dis1_track = -1.0` ⇒
  `min_dis_track < min_dis1_track` is false for every cell ⇒ **the whole
  association set beyond the 3 mm floor is stripped** for that fit point;
- the fitting segment itself answers -1.0 ⇒ wins every comparison ⇒ every
  cell kept unconditionally.

The prototype cannot make this mistake: `ToyPointCloud::get_closest_2d_dis`
returns **1e9** on an empty tree (`data/src/ToyPointCloud.cxx:415-437`) —
"no measurement", which loses all comparisons and keeps the keep-rule
semantics intact.

**Why it hid.**  Reachable only with `fit_exclusion=true`, which no
production config sets; pr/30's one forced run measured the aggregate
regression, not the mechanism.  In uBooNE (single face) the prototype never
has cross-face competitors, so the port had no behavioural reference for
this corner.

**Fix** (`clus/src/TrackFitting.cxx`, `update_association`, all three plane
loops; toolkit commit 2ecfc630): map negative distances to "no measurement"
— competitor `temp_dis < 0` ⇒ skip; own `min_dis_track < 0` ⇒ 1e9.  Also
removed the dead `cluster`/`transform` locals at the top of the function
(never read; the prototype's `update_association` takes no transform either)
— their `m_pcts` dereference was the only obstacle to a fixture-level
doctest.  A one-line `friend struct TrackFittingTestHarness` test seam was
added to `TrackFitting.h` so the doctest can inject synthetic
`wpid_offsets`/`wpid_slopes` without a detector-volumes fixture.

**Verification.**
- New doctest `clus/test/doctest_update_association.cxx` (3 cases):
  cross-face competitor must not strip; own-segment sentinel must not
  always-keep; the untouched keep rule (closer competitor strips, 3 mm floor
  keeps, no-competitor keeps).  **Fails pre-fix exactly as designed** (guards
  sed-removed, rebuilt: cases 1-2 FAIL, case 3 passes; restored: 3/3 pass).
- Full `wcdoctest-clus`: 215/215 pass.
- Knob-off byte-identity: §4 gate.

## 3. The pr/30 §3.1 unit question: DISCHARGED, composition is exact

pr/30 §3.1 recorded (and §12.10 blocked the whole item on): prototype
`y = (wire - offset_u) * pitch_u` (`PR3DCluster_multi_track_fitting.h:1013`)
vs toolkit `raw_y = (coord.wire - offset_u) / slope_yu` with
`slope_yu = -sin(angle_u)/pitch_u` — "the two feed different
closest-2-D-distance helpers, each of which projects again", never composed.
The composition, written out once and for all:

- Toolkit query: `raw_y = (wire - offset_u)/slope_yu`, test point
  `(raw_x, raw_y, 0)` (`TrackFitting.cxx` U loop).
- `DynamicPointCloud::get_closest_2d_point_info` projects the query as
  `projected_y = cos(a_u)·z − sin(a_u)·y = −sin(a_u)·raw_y
  = −sin(a_u)·(wire−offset_u)·pitch_u/(−sin(a_u)) = (wire − offset_u)·pitch_u`
  (`DynamicPointCloud.cxx:365`) — **exactly the prototype's transverse
  length coordinate**.
- The kd-tree stores the same convention for the cloud points:
  `cos(angle)·z − sin(angle)·y` (`DynamicPointCloud.cxx:540`, matching the
  prototype's `ToyPointCloud::AddPoints`, `ToyPointCloud.cxx:282`).
- W plane: toolkit uses `raw_z = (wire − offset_w)/slope_zw` with the
  identity projection (`cos 0·z − sin 0·y = z`) — same physical quantity;
  the prototype's positive-sine `slope_yw` anomaly is dead code in its
  `update_association` (the six slope locals there are computed and never
  used).
- Drift: `raw_x = (time − offset_t)/slope_x` is the exact inverse of the
  forward tick map on both sides.  Offsets differ by the documented
  half-pitch bin-convention shift (pr/7 §), consistently applied on both the
  forward and inverse paths.

So both implementations query a plain Euclidean 2-D metric in
(drift-x, transverse-length), in the same units, with the same 0.3 cm floor.
**pr/30 §12.10's branch resolves to "the compositions agree"**: §12.4's
measurement was of real physics at that operating point — *plus* the §2
sentinel bug wherever a cross-face competitor existed, which the branch
language did not anticipate.  This also discharges pr/37 §9 row 9 and the
carry-forwards in pr/31 §7 / pr/32.

## 4. Knob-off gate: PASS

Arm `work-pr98-off-nuecc48` (post-fix binary, no `SBND_FIT_EXCLUSION`,
`PR_EXTRA_STAGES=pr_display`) vs production baseline `work-nuecc48-prod0819`:

```
$ python3 scripts/pr85_hash_gate.py work-nuecc48-prod0819 work-pr98-off-nuecc48
# events in A: 48  compared archives: 96  missing/unpaired events: 0
PASS all 96 archives byte-identical        (rc=0)
```

`nusel-events.tsv` and `nusel-table.tsv` also diff-identical.  So the §2 fix,
the dead-local removal and the test seam are byte-neutral with the knob off,
on the same binary epoch as production (which also re-confirms pr/97's UAF
fix left the baseline untouched).

The §5 perf helper landed after this gate, so the FINAL binary was re-gated:
`work-pr98-off2-nuecc48` vs `work-nuecc48-prod0819` — **PASS all 96 archives
byte-identical** (rc=0).

## 5. Knob-on re-measurement (nueCC48)

Arm `work-pr98-fx-nuecc48` (fixed binary, `SBND_FIT_EXCLUSION=true`,
runtime proof: `PR30AUDIT ... knobs[fit_exclusion=true ...]`) vs
`work-pr98-off-nuecc48`, via `pr83r3_scores_ab.py` (TSV in the round's
scratch; 47 of 48 events build a PR graph — 116962 does at this operating
point, 172230/444187/388/10550/90055 rows carry partial columns).

| metric | pr/30 §12.4 (buggy exclusion) | pr/98 (fixed exclusion) |
|---|---|---|
| events changed | 47/48 | 44/44 numu-evaluable moved; labels 0 flips |
| nue sign flips (+→− / −→+) | 7 / 3 | **5 / 2** (of 17 nue-evaluable both arms) |
| kine_reco_Enu \|Δ\| | median 35.5 MeV, max 429 | median 103 MeV, max 1810 |
| selected-vertex shift | median 0.70 cm, max 97 cm | 17 vtx-evaluable: median 5.9 cm, max **213.5 cm** |
| event_label flips | 0 | 0 |

nue lost: 69314, 163543, 235435, 423981, 38856.  nue gained: 122660, 389538.
The moves are LARGER than pr/30's, not smaller — expected: the §2 fix makes
exclusion actually arbitrate (pre-fix, a cross-face competitor collapsed the
association set to the 3 mm core instead).  pr/30's row measured the bug as
much as the feature; these are the first numbers for the feature itself.

**Coverage census** (`pr96_uncover_census.py`, main clusters, both arms):
uncovered-charge fraction moves NET WORSE on paper — median +0.40 pp,
9 clusters improve / 22 worsen, track-like uncovered groups 6→8 clusters.
Read with care: with exclusion OFF a fit dragged onto a neighbour's charge
"covers" it while being wrong; with exclusion ON honest fits can leave
genuinely unclaimed charge (shower rind) uncovered.  The census direction is
therefore ambiguous BY CONSTRUCTION for this knob — §6's visual review is the
decisive test (and the two most-"worsened" movers, 235435 and 122660, are
visually equal-or-better).

**Third-pass point drops** (the pr/28 T4 toolkit-only pre-dQ/dx rebuild):
"pre-dQ/dx form_map_graph dropped" fires 191 times / 546 trajectory points ON
vs 60 lines OFF.  A real toolkit-only side effect of exclusion (the prototype
never re-derives associations at final positions and so never drops these
points); candidate refinement if the owner wants it measured: pass
`flag_exclusion=false` to the third `form_map_graph` only.  Not changed this
round.

**Runtime** (the feature's inherent cost; the perf fix below is already in):
knob-off median 6.7 s/evt.  Exclusion ON with the initial implementation:
median 14.5 s, tails 4–6.7×, heaviest event 256587 63 s → 757 s (12×).  Found
and fixed a 3× inefficiency — `update_association` queried all three plane
kd-trees per arbitration via `segment_get_closest_2d_distances` where each
plane loop consumes one component; the prototype queries a single plane
(`ProtoSegment::get_closest_2d_dis(x,y,plane)`).  Replaced with the
file-local single-plane helper `exclusion_closest_2d_dis` (same cloud
fallback + sentinel semantics).  Byte-identical proof: reruns of 54095,
196649 (`work-pr98-fx2-smoke`) and 256587 (`work-pr98-fx3-smoke`) all hash
PASS vs `work-pr98-fx-nuecc48`.  Timing: 54095 146→57 s (2.3× over off),
196649 73→31 s (2.9×), 256587 757→228 s (3.6×).  Residual exclusion cost at
the operating point: roughly 2–3.6× on segment-rich events, ~1.5–2× typical.

## 6. Fit-quality review (largest movers first)

Bee A/B (48 events, 12 largest movers at idx 0–11):
OFF https://www.phy.bnl.gov/twister/bee/set/da8df549-376a-4eb7-8be3-a271fee42112/event/list/
ON  https://www.phy.bnl.gov/twister/bee/set/6f30f958-3cce-473b-9a9e-4f79e30b082d/event/list/
Annotated index: `bee/pr98/pr98.index.txt`.  Static near-vertex 2-D panels
(new `scripts/pr98_fit_panels.py`, X-Y/X-Z/Y-Z ±20 cm around each arm's main
vertex, both arms stacked) were read for the 12 movers; per-event verdicts in
the index.  Summary of the AI read (owner scan pending):

- **Visually better ON (3)**: 235435 (vertex moves from a bare track end
  into the multi-prong activity, 4–5 prongs fitted where OFF left the blob
  nearly unfitted — despite nue 4.30→−4.24), 46363 (vertex into the
  multi-prong hub, was 45 cm away at a track tip), 116962 (fit covers the
  shower column below the vertex that OFF left bare).
- **Comparable (7)**: 38856, 163543, 423981, 69314 (nue sign flips with
  near-identical fits — score sensitivity, not fit degradation; 69314 gains
  one straight-chord artifact), 122660, 111412, 271851 (vertex slides along
  the same trunk; shower coverage trades jumbled-many-fits for one clean
  trajectory).
- **Degraded ON (1)**: 52672 — the ON vertex lands in visibly empty space
  between charge blobs (31 cm move).  nue stays negative in both arms.
- 389538 is a pure DL-vertex re-selection (213 cm; both candidate sites are
  clean two-track V's with good fits).

Net visual verdict: exclusion-ON fits are equal or better near the vertex in
11/12 top movers.  The nue-score losses land almost entirely on events whose
fits look unchanged or improved — consistent with pr/30 §12.8's reading that
every downstream cut was tuned against the non-excluding fit, and NOT with
broken fits.

## 7. Owner decision (presented, not taken)

The port is now believed faithful (§1 parity list, §2 fix, §3 unit discharge)
and the feature does on SBND what it does in the prototype: keeps each
segment's fit on its own charge.  What the measurement says:

1. **Fit quality (the round's criterion): equal or better in 11/12 top
   movers**; one adverse vertex placement (52672, not nue-selected either
   way).
2. **Selection at today's operating point: net −3 nue candidates on nueCC48**
   (5 lost / 2 gained), with the losses mostly on visually unchanged-or-better
   fits.  Flipping production ON without retuning the downstream cuts would
   trade fit fidelity for nueCC selection at this sample size; no truth
   labels, so this is a sign-count, not an efficiency measurement.
3. Open options, in increasing effort: (a) keep OFF, close the port-fidelity
   item as measured; (b) flip ON and retune the affected cuts on a larger
   manifest (valfast/1000-scale) — the prototype-parity choice; (c) first
   measure the third-pass refinement (§5) and/or the pr/96 narrower 3-site
   variant, both cheap arms from this round's binary.

No flip is performed in this round.

## 8. Artifacts and labels

- toolkit commit 2ecfc630: sentinel fix + single-plane perf helper + dead
  locals removed + `friend struct TrackFittingTestHarness` seam +
  `clus/test/doctest_update_association.cxx` (3 cases, fails pre-fix).
- wcp commit (this commit): this doc, `scripts/pr98_fit_panels.py`, Bee sidecars
  (`bee/pr98/pr98.index.txt`, `*.prid-map.txt`, `*.url`), work-tags entries.
- Arms: `work-pr98-off-nuecc48` (gate arm, first post-fix binary),
  `work-pr98-off2-nuecc48` (final-binary gate arm), `work-pr98-fx-nuecc48`
  (the measurement), `work-pr98-fx2-smoke` (54095+196649 identity/perf),
  `work-pr98-fx3-smoke` (256587 identity/perf).  Q/L root
  `work-nuecc48-ql0819`; baseline `work-nuecc48-prod0819`.
- Panels: `/home/xqian/tmp/pr98_panels/pr98_evt*.png` (scratch; regenerate
  with the Repro block + `scripts/pr98_fit_panels.py`).

# pr/9 — evt 172230: the endpoint-dilution root cause and the real electron converter

Status: INVESTIGATION COMPLETE + **F1 IMPLEMENTED** (2026-07-30, toolkit
d3b5972a, sec. 10; investigation instrumentation reverted with bit-identity
re-proven).  Corrects the converter identification in pr/8 §11.  F2 and the
sec. 5 fidelity divergences still await owner decision.

Follow-on to pr/5 (initial anatomy), pr/7 (persistence divergence), pr/8
(proton_dir_vote + MIP config, toolkit 3d71e111).

## Repro block

```
# ON-arm run (vote fires; final output still electron):
cd sbnd_xin/work-nuecc48-prsmoke2
PROUT=<fresh-dir> ./run_pr3_evt.sh 172230        # binary at toolkit 3d71e111

# dQ/dx profile + offline do_track_pid reproduction + endpoint sweep:
python3 sbnd_xin/scripts/repro_track_pid_evt172230.py
#   reads nupr_evt172230_mipvote_on/tracking-pr.root (T_rec_charge, rcid 5030)
#   templates from cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet

# Instrumented runs (log-only; all reverted afterwards):
#   /home/xqian/tmp/nupr_evt172230_topoinstr{,2,3,4}   (scratch, not records)
# Post-revert identity: PROUT=/home/xqian/tmp/nupr_evt172230_postrevert2
#   hash_archive mabc-pr.zip = 65a64151... == recorded mipvote_on arm.
```

## 1. The two owner questions

1. The stopping end's dQ/dx is artificially distorted (here: low) because the
   endpoint is not well defined — shouldn't the PID comparison be robust to
   that, making this track's direction NOT weak?
2. Why does `is_shower_topology` capture this clean proton as an EM shower —
   is something not properly run?

Short answers: (1) **yes — quantified below; removing one endpoint sample
flips the whole decision to strong-direction proton**; (2) **it doesn't — the
stored shower-topology flag is false at every stage; the electron conversion
comes from `examine_all_showers`' shower-dominated-cluster rule (a faithful
prototype port), and pr/8 §11's attribution to the improve_vertex rescue was
wrong.**

## 2. The measured profile: a textbook Bragg rise with a collapsed tip

`T_rec_charge` (tracking-pr.root, `real_cluster_id==5030`, 17 fit points,
0.6 cm spacing, ordered from the true vertex at (−54.7,−87.5,19.9) toward the
stop at (−46.1,−84.2,22.9)):

| rr [cm] | 0 | 0.6 | 1.2 | 1.8 | 2.4 | 3.0 | 3.6 | 4.2 | 4.8 | 5.4 | 6.0 | 6.6 | 7.2 | 7.8 | 8.4 | 9.1 | 9.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dQ/dx [e/cm ×1000] | 98 | 128 | 127 | 127 | 127 | 132 | 139 | 141 | 150 | 155 | 156 | 158 | 161 | 168 | **193** | **190** | **121** |

A monotone Bragg rise 98k → 193k over 9.1 cm — then the **last fit point, at
the stopping tip, collapses to 120,539 e/cm (−37% from the peak)**.  The tip
point's dx is normal (0.64 cm), so this is charge dilution at an ill-defined
endpoint (deposit ends mid-bin; fit point may sit slightly beyond the true
stop), exactly the owner's diagnosis.  NOTE (owner): the endpoint artifact can
also go the other way — charge from beyond the fit end piled into the last
bin gives an artificially HIGH tip — so any fix must treat the endpoint as
unreliable regardless of its value, not "drop if low".

## 3. Quantified: the tip point alone causes the abstention

### 3.1 Why it breaks the comparison

`do_track_comp` (PRSegmentFunctions.cxx:1291, prototype ProtoSegment.cxx:1120)
evaluates the templates at residual range `end_L − L(i)` with
`end_L = L.back() + 0.15cm − offset_length` and `offset_length = 0` on the
normal path.  The tip sample is therefore matched against the template at
rr ≈ 0.15 cm — where the proton template is at its MAXIMUM (261,588 e/cm,
SBND tables) — while the measured value is 120,539.  The largest single-point
template-data discrepancy in the whole track is thus manufactured at the
endpoint, in both the KS shape term and the muon-vs-flat direction gate.

### 3.2 Offline reproduction and endpoint sweep

Reproduced `do_track_pid` bit-for-bit in Python (kslike_compare CDF max-dist,
eval_ks_ratio, mu/flat/p/e templates from `particle_dataset.jsonnet` with
linear interp + boundary clamp, MIP flat = 56,000 e/cm as in the SBND ON
config).  Validation: offset 0 reproduces the production abstention and the
vote's numbers (s_p fwd 0.140 offline vs 0.135 in-job; ks/ratio differences
are rounding in the dumped q/nq).

| variant | fwd dirbit | bwd dirbit | decision | s_p(fwd) | s_p(bwd) |
|---|---|---|---|---|---|
| offset 0 (current)      | 0 | 0 | **ABSTAIN** | 0.140 | 0.204 |
| offset 0.5 cm           | 1 | 0 | dir=+1 **proton** | 0.126 | 0.185 |
| offset 1.0 cm           | 1 | 0 | dir=+1 proton | 0.156 | 0.180 |
| offset 2.0 cm           | 0 | 0 | ABSTAIN | 0.245 | 0.163 |
| offset 3.0 cm           | 0 | 0 | ABSTAIN | 0.300 | 0.197 |
| **trim 1 sample** (no template shift) | 1 | 0 | dir=+1 **proton** | **0.077** | 0.147 |
| trim 2 samples          | 1 | 0 | dir=+1 proton | 0.060 | 0.112 |

Removing ONE endpoint sample flips the event to a strong, correctly-directed
proton through the NORMAL prototype decision path — no vote needed, score
0.077 < 0.09 (below even the improve_vertex rescue threshold) and < 0.13
(not dir-weak).  Over-trimming (≥2 cm offset) destroys the Bragg contrast:
the sweet spot is ~1 endpoint sample / ≤1 cm.

### 3.3 Prototype precedent for endpoint distrust

The mechanism already exists in the prototype signature:
`do_track_pid(L, dQ_dx, compare_range, offset_length, flag_force)`.
- The isolated-track forced path (`start_n==1 && end_n==1 && npoints>=15`,
  ProtoSegment.cxx:1555-1556) passes **offset_length = 1 cm**, which both
  shifts the template zero and DROPS the last fit point
  (end_L − L.back() = 0.15 − 1.0 < 0).  The prototype author already
  distrusted the endpoint on that path.
- The normal vertex-attached path (:1559-1560) passes 0 — tip kept.
- Commented-out lines :1561-1562 show 3 cm was tried for the normal path and
  left disabled.

## 4. Question 2: `is_shower_topology` never fires — the converter is `examine_all_showers`

Instrumented one ON run per step (log-only, reverted after; §7):

1. Every `segment_is_shower_topology` call logged (stored-flag-before,
   verdict, spread numbers) + the improve_vertex rescue gate logged:
   at the rescue evaluation for seg 5030 the segment was **already pdg=11,
   still carrying the vote's score 0.135296** (the rescue would have set 100),
   `stored_flag=0`, and `ndaughter=25 ≠ 1` — so the rescue's topology test
   was **short-circuited and never ran**.  The rescue is NOT the converter;
   pr/8 §11 stands corrected.
2. All 7 literal `set_pdg(11)` sites instrumented: none fired for this
   segment.
3. Object-level hooks (2212→11 in `ParticleInfo::set_pdg` + segment pinfo
   replacement) with backtraces: ONE hit —
   `PINFO-REPLACE 2212->11 nfits=17 score=0.135296` from
   **`PatternAlgorithms::examine_all_showers` ← `shower_determining_in_main_cluster` ← visit**,
   i.e. in the FIRST PR round, before improve_vertex and before shower
   clustering.

### The mechanism (toolkit NeutrinoTrackShowerSep.cxx:1549-1873 ↔ prototype NeutrinoID_track_shower.h:1007-1246 — line-faithful port)

`examine_all_showers` is a whole-cluster classifier:

- Census: a segment is a "good track" only if `dirsign != 0 && !dir_weak`.
  Our proton, with the vote, has dirsign=+1 but score 0.135 > 0.13 ⇒
  **dir-weak ⇒ counted as a mere "track"**.
- The nueCC main cluster is EM-dominated (shower length ≫ track length,
  n_good_tracks == 0), so `flag_change_showers` fires and the conversion loop
  (:1851-1872; prototype :1238-1245) **replaces the ParticleInfo of EVERY
  non-shower segment with an electron** — particle_score untouched, which is
  why 0.135 survived to the rescue printout.
- From then on the segment's pdg==11 satisfies every downstream `is_shower`
  predicate (`kShowerTrajectory || kShowerTopology || pdg==11`): it is seeded
  as a shower start segment in shower clustering, `Shower::update_particle_type`
  re-affirms electron, etc.  The "showerness" never came from topology.

The prototype would do the SAME to its own median-catch proton (type 2212,
score sentinel 100, flag_dir=0 ⇒ dir-weak ⇒ census "track" ⇒ converted).
This is not a porting bug — it is the design: **in a shower-dominated
cluster, only a strong-direction track survives as a track.**  The causal
frontier is therefore entirely upstream: the endpoint dilution is what makes
this proton fail the "good track" bar (0.135 vs 0.13); with the endpoint
handled (§3.2: 0.077–0.126) it becomes a good track and the wholesale
conversion is disqualified at the census (n_good_tracks=1) — subject to the
:1594 single-good-track demotion checks (back-to-back-with-shower angle
tests), which need an empirical run to evaluate on this event.

### Corrected causal chain for evt 172230 (vote ON)

1. Endpoint tip collapse ⇒ muon-vs-flat abstains both ways ⇒ legacy path
   abstains; the pr/8 vote correctly recovers pdg 2212, dir=+1, score 0.135.
2. 0.135 > 0.13 ⇒ dir-weak ⇒ not a "good track".
3. `examine_all_showers` (first PR round): shower-dominated cluster + no good
   track ⇒ wholesale electron conversion catches the proton (score kept).
4. pdg==11 makes it a shower everywhere downstream; the improve_vertex rescue
   (score<0.09) never applies (ndaughter=25≠1, and by then it is already 11).
5. Geometric vertexing, unconstrained by a weak direction, again picks the
   Bragg end; DL arm picks the true vertex but 3-4 happen regardless.

## 5. Real-but-non-causal fidelity divergences found on the way (record)

At the improve_vertex rescue site (NeutrinoVertexFinder.cxx:2334 ↔ prototype
NeutrinoID_improve_vertex.h:311):

- D1 — the toolkit **recomputes** `segment_is_shower_topology(sg,...)` in the
  gate; the prototype reads the **stored flag** (`get_flag_shower_topology()`),
  which is only recomputed for `type==0` segments earlier in improve_vertex
  (:233/:261 — the toolkit ports those two gates correctly at :2224/:2264).
  For a typed segment whose stored flag is false, the toolkit can enter the
  rescue where the prototype cannot — and the recompute has side effects.
- D2 — because the recompute runs inside the `if`, the subsequent
  `dir_save = sg->dirsign()` captures the topology function's dirsign
  side-effect (spread-asymmetry value, often 0), not the segment's real
  direction as in the prototype.
- D3 — the toolkit port of `segment_is_shower_topology` only SETS
  kShowerTopology on a true verdict (PRSegmentFunctions.cxx:2691); the
  prototype resets the member at entry (`flag_shower_topology = tmp_val`,
  ProtoSegment.cxx:320), so a false verdict CLEARS a previously-true flag.
  The toolkit flag is sticky-true.

None of these fired for evt 172230 (`ndaughter=25` short-circuits the gate),
but they can matter elsewhere; candidates for a follow-up fidelity fix
alongside pr/7 §5.

## 6. Proposed fixes (owner decision)

- F1 (systematic, recommended): make the dQ/dx PID comparison
  endpoint-robust — a `TrackPidOptions` knob (default OFF ⇒ byte-identical)
  that excludes the endpoint sample(s) of the STOPPING end from the KS/ratio
  comparison, value-agnostically (robust to both low AND high endpoint
  artifacts).  Two candidate shapes, to be chosen in calibration:
  trim-N-samples (no template shift; strongest here: s_p 0.077) vs
  offset_length=1cm (prototype's own forced-path precedent).  Applies to both
  orientations symmetrically (the "endpoint" is whichever end the orientation
  under test treats as the stop).
- F2 (optional guard, further designed divergence): in `examine_all_showers`'
  conversion loop, spare segments with a stored 2212 and a good score — i.e.
  protect a scored proton from the wholesale electron conversion.  Likely
  unnecessary if F1 lands (the proton becomes a good track), but closes the
  class of events where even a strong proton is demoted by the :1594
  back-to-back checks.
  **IMPLEMENTED, doc pr/40 F2 (2026-08-06)** — shipped as
  `shower_reclass_dqdx_guard`, SBND production default ON, generalized to a
  dQ/dx-decisiveness test (not just a stored 2212) and applied at
  `examine_all_showers` AND its sibling `improve_maps_shower_in_track_out`'s
  two reclassify loops.  F1 landing was NOT sufficient on its own for two of
  doc pr/40's nine owner-reported cases (423981, 433451): both already had a
  confidently stored proton, and were converted anyway because their
  `is_dir_weak` score sat just above the 0.13 threshold — exactly the
  "even a strong proton is demoted" class this note anticipated.
- The pr/8 three-threshold calibration (0.25 vote / 0.13 dir-weak / 0.09
  rescue) should be done jointly WITH F1, since F1 shifts the score
  distribution of exactly the affected population.

## 7. Instrumentation hygiene / verification

- 4 instrumented builds, each log-only; runs to /home/xqian/tmp scratch dirs
  (never into record dirs).
- Instrumented ON-run archive members == recorded ON arm:
  `hash_archive mabc-pr.zip = 65a64151...` (identical member set/hashes).
- Full revert: `git checkout` of all 6 touched files; rebuilt (wcbuild rc=0);
  `strings libWireCellClus.so` contains none of the instr markers;
  wcdoctest-clus 565/565; post-revert rerun
  (`/home/xqian/tmp/nupr_evt172230_postrevert2`) == `65a64151...`.

## 8. Run ledger (all scratch, /home/xqian/tmp)

`nupr_evt172230_topoinstr` (topology+rescue instr),
`_topoinstr2` (+5 set_pdg(11) sites), `_topoinstr3` (+update_particle_type),
`_topoinstr4` (+object-level 2212→11 hooks w/ backtrace),
`_postrevert2` (identity re-proof).

## 10. F1 IMPLEMENTED (owner go-ahead 2026-07-30, same session)

Owner design constraint: DYNAMIC — trim 0 or 1 sample, never more; no trim
when the endpoint is fine.  Realized as a **fallback retry**, not an
unconditional cut (and not an explicit tip-anomaly test, which cannot
distinguish an artificially-high tip from a genuine Bragg maximum):

- Primary attempt unchanged (tip included).  Only when BOTH orientations
  abstain (the same entry condition as the pr/8 vote, no flag_force), retry
  once with exactly 1 sample excluded at each orientation's hypothesized
  stopping end — template anchor (end_L) unchanged, value-agnostic.  A
  decided retry returns direction+type+score through the same mu/p/e
  competition; an abstaining retry falls through to the proton_dir_vote
  (which still votes on the UNTRIMMED results — unchanged pr/8 semantics).
- Order: legacy -> endpoint-trim retry -> proton_dir_vote -> abstain.

Code: `do_track_comp(..., int skip_stop_samples=0)` drops the trailing
(smallest-residual) samples after windowing (PRSegmentFunctions.cxx); retry
block in `segment_do_track_pid` before the vote; knob threaded
`TrackPidOptions.endpoint_trim_retry` <- `PatternAlgorithms::m_endpoint_trim_retry`
<- TaggerCheckNeutrino `endpoint_trim_retry` <- jsonnet key-suppressed arg
(`cfg/pgrapher/common/clus.jsonnet`), SBND `clus.jsonnet` clus_pr/pr args
DEFAULT TRUE.  C++ default false.

### Verification (all PASS)

- wcdoctest-clus 565/565; freshness proof done.
- Production configs: `abtest/compile_all_cfg.sh` before (worktree @34f0abd8,
  pre-F1) vs after — **0-line diff** across the full manifest.
- Compiled-config proof: `"endpoint_trim_retry" : true` present in the
  13-stage PR-job JSON; absent from the uBooNE config (arg not passed).
- SBND off-gate: pre-F1 cfg (worktree) + new binary ->
  `65a64151...` == the committed pr/8 ON reference. Bit-identical.
  (Scratch runs /home/xqian/tmp/nupr_evt172230_f1off2.)
- uBooNE off-gate PASS: 35-event sweep `sweep/f1off_ub`, all 35 ZIP member
  hashes identical to BOTH `dirweakon_ub` and `mipvoteoff_ub` (rollup md5
  d13e7c6b), all rc=0.  (ZIPS is the only discriminating gate — ab_check
  gate 2 is A/A-noisy per the 7902615c round.)

### Knob-on result, evt 172230 (geometric arm `/home/xqian/tmp/nupr_evt172230_f1on`, DL arm `nupr_evt172230_f1on_dl`)

Large improvement, one residual:

- **Main vertex moved off the Bragg tip** to (−55.7, −87.3, 22.0) — ~2.3 cm
  from the true vertex (−54.7, −87.5, 19.9), vs ~10 cm wrong before.
- **Geometric and DL arms now agree byte-identically** (both mabc-pr.zip
  `987fbf56...`): the strong direction lets the geometric vertexing find the
  same vertex the DL arm favored.  Pre-F1 the arms disagreed.
- Seg 5030 is **no longer an electron**: strong direction, pointing correctly
  into its Bragg stop; the shower-dominated wholesale conversion no longer
  catches it (n_good_tracks != 0).  PF: e− 1649 MeV + track + small pi+ stub.
- RESIDUAL: the track is labeled **mu− 51 MeV, not proton**.  Cause
  (offline-verified on the refit 20-point profile): the fit now bridges the
  ~2 cm vertex gap with ~4 near-empty samples (dQ/dx ≈ 12k e/cm ≈ 0.2 MIP)
  prepended to the Bragg profile; the PRIMARY attempt then passes forward on
  the muon-vs-flat gate with s_mu 0.258 < s_p 0.411 (the empty prefix
  deflates the measured sum, inflating the proton ratio).  The retry never
  enters (no abstention).  This is a vertex-gap / empty-sample robustness
  question (cf. the doc-50 STM gap-jumping observation), NOT an endpoint
  question — flagged for the pr/8 sec 6 joint calibration round (candidate:
  exclude near-empty samples from the comparison, or tighten the vertex fit).

## 11. Post-F1 vertex-displacement autopsy: DL is right, `search_for_vertex_activities` drags it (2026-07-30, same session)

Owner questions: (a) is the DL vertex actually running properly post-F1?
(b) which algorithm creates the extra small track at the vertex that makes
the refit drag?

### Repro block

```
# post-F1 DL arm, trace-level logging (bit-identical output: mabc 987fbf56)
PROUT=/home/xqian/tmp/nupr_evt172230_f1dl_trace \
  <scratchpad>/run_pr3_evt_dl_trace.sh 172230
# pre-F1 DL arm: worktree cfg at 7902615c (knob key absent = legacy path)
git -C toolkit worktree add /home/xqian/tmp/wt-pref1 7902615c
PROUT=/home/xqian/tmp/nupr_evt172230_pref1dl_trace \
  <scratchpad>/run_pr3_evt_pref1_dl_trace.sh 172230
```

(The DL/vertexing trace lines are SPDLOG_LOGGER_TRACE — invisible at the
usual `-L debug`; that is why earlier greps found nothing.)

### (a) DL runs properly and selects the TRUE vertex

Post-F1 trace: `determine_overall_main_vertex_DL: rerank mode, K=5`,
regime=confident, voxel 0 → cluster 5 pos (−54.7,−87.5,19.9) — the true
vertex — snap 0.32 cm, "switching to DL vertex", scn_inference=1003.6 ms.
The trace run's mabc-pr.zip hashes 987fbf56 == the recorded post-F1 arms.

### (b) The stub creator is `search_for_vertex_activities`, and the drag is a create→refit→prune ordering effect

Chain in the final `improve_vertex(flag_search_vertex_activity=true,
flag_final_vertex=true)` (TaggerCheckNeutrino.cxx:473):

1. `fit_vertex` nsegs=2: (−54.92,−87.67,19.72) → (−54.61,−87.67,19.72),
   0.313 cm.  Fine.
2. `search_for_vertex_activities` (NeutrinoVertexFinder.cxx:56, prototype
   NeutrinoID_improve_vertex.h:1039) at the main vertex, range 1.5 cm,
   finds a steiner-terminal at (−55.23,−87.67,19.72) — **0.62 cm away
   purely along −x (drift), y and z identical**: drift-blurred charge of
   the vertex blob, not a real third prong.  Round-1 accept (>0.6 cm 3D
   from both fit clouds, sum-2D >1.2 cm, angle x charge = 3.5e6).  Creates
   stub vertex+segment; main vertex nsegs 2→3, queued for refit.
3. `do_multi_tracking` with the stub, then the 3-segment refit
   (NeutrinoVertexFinder.cxx:2198): vertex dragged 2.4 cm to
   (−55.86,−87.67,22.12), mostly +z along the proton toward its Bragg tip.
4. `eliminate_short_vertex_activities` runs only AFTER the refit
   (toolkit :2213 == prototype :225 — prototype-faithful ordering).  Its
   cuts are 0.36–0.5 cm, but the stub now spans to the displaced vertex
   and re-tracking stretched it to 2.42 cm → survives (examine_direction:
   nfits=5, dqdx_ratio=0.285, assigned pdg 13 dir-weak).  **Self-locking:
   the displacement the stub causes makes the stub un-prunable.**  (The
   pruner does work: one of two other activity stubs was removed, 57→56.)

CORRECTION to sec 10: the "small pi+ stub" in the PF listing is this
2.42 cm near-empty (0.3 MIP) vertex-activity artifact, not a pion.

### Why pre-F1 (DL arm) did NOT create the stub

Pre-F1 trace: same DL selection (voxel 0 dl=0.9944, same position, snap
0.70 cm, switch) — but the final `improve_vertex` enters and exits in 3 ms
with ZERO fit/search lines, main vertex left untouched at the snap
(−54.92,−87.67,19.72).  Cause: the all-shower guard `flag_skip_two_legs`
(NeutrinoVertexFinder.cxx:1956-1968 == prototype
NeutrinoID_improve_vertex.h:48-60 "if all showers, no need to fit vertex
with only two legs").  Pre-F1, examine_all_showers had wholesale-converted
EVERY cluster-5 segment to electron (sec 4), so ntracks==0 →
flag_skip_two_legs=true → both the fit loop (:2015) and the activity-search
loop (:2149) skip the 2-leg main vertex entirely.  Post-F1 the proton is a
track again → guard off → the machinery engages and invents the stub.

**Net: F1's PID fix re-armed the vertex-activity machinery that the
all-shower mislabeling had been accidentally disarming.**  The full causal
chain of the sec-10 residual is closed: drift-blur stub → 3-seg refit drag
(2.4 cm along the proton) → near-empty bridge samples → mu-vs-proton flip.

### Candidate fixes (owner decision; all prototype divergences ⇒ default-OFF knobs)

- F3a: gate stub creation on fitted charge quality or veto drift-parallel
  candidates (Δy,Δz ≈ 0) — kills exactly this artifact class.
- F3b: run the short-stub elimination (or a creation-length <0.5 cm check)
  BEFORE the 3-segment refit, so a prunable stub cannot first move the vertex.
- F3c: exclude sub-cm stubs from `fit_vertex`'s segment set.

Run ledger addition: `/home/xqian/tmp/nupr_evt172230_f1dl_trace` (post-F1 DL
trace, mabc 987fbf56), `/home/xqian/tmp/nupr_evt172230_pref1dl_trace`
(pre-F1 DL trace, worktree wt-pref1 @ 7902615c).

## 12. F3c IMPLEMENTED: fit_vertex short-segment exclusion (owner go-ahead 2026-07-30, same session)

Owner: "the very short track ... is not good for it to be included in the
vertex fitting" -- explicit OK to diverge from the prototype.  Knob
`fit_vertex_min_seg_length` (cm; C++ default 0 = legacy include-all,
byte-identical; SBND default 1.0 cm).

### Repro block

```
# knob-on DL arm (trace):
PROUT=/home/xqian/tmp/nupr_evt172230_f3con_dl3 <scratchpad>/run_pr3_evt_dl_trace.sh 172230
# knob-on geometric arm: same runner with dl_weights=
# off-arm gate: worktree cfg @ d3b5972a (key absent) + new binary
git -C toolkit worktree add /home/xqian/tmp/wt-f3c d3b5972a
PROUT=/home/xqian/tmp/nupr_evt172230_f3coff2 <scratchpad>/run_pr3_evt_f3coff.sh 172230
# uBooNE sweep: qlport/scripts/sweep_5384.sh f3coff_ub 6
```

### Two design iterations (both verified by trace, evt 172230)

1. **Fit-length filter FAILED**: measuring segments by fit-point span missed
   the stub -- its FIT cloud spreads past the cut before any refit
   (wcpt_len=0.63 cm but fit_len=1.15 cm).  Switched to WCPT-path length
   (the graph-topology extent).
2. **MyFCN exclusion alone FAILED**: with the stub excluded from the
   AddSegment accumulation the refit still produced the IDENTICAL 2.4 cm
   drag.  Mechanism (MyFCN.cxx FitVertex): each segment contributes only
   TRANSVERSE PCA constraints (row 0 = track direction is zeroed), so the
   position along the proton axis comes from the other legs' lines -- and
   those lines are built from fit clouds re-generated by do_multi_tracking
   WITH the stub in the graph.  The stub corrupts the fit through charge
   competition in the re-tracking, not through its own PCA term.
   Final semantics: legs are segments with wcpt-path length >= cut; >=3
   surviving legs => fit on the survivors; <=2 => SKIP the fit (the two-leg
   position was already fit by the plain pass; a stub-triggered refit on
   re-tracked clouds is exactly the drag).  Excluded segments stay in the
   graph / particle flow.

### Implementation

- `PatternAlgorithms::m_fit_vertex_min_seg_length` (NeutrinoPatternBase.h),
  filter + skip in `fit_vertex` (NeutrinoVertexFinder.cxx:1888-1920 region),
  threaded via TaggerCheckNeutrino (`fit_vertex_min_seg_length`, cm) and
  cfg key-suppression (common/clus.jsonnet); SBND clus_pr + pr() default 1.0.
- Trace lines: per-candidate `wcpt_len/fit_len` (only at >2-seg vertices),
  `excluding N short segment(s)`, `skipping vertex fit: only N ...`.

### evt 172230 knob-on result (DL arm, nupr_evt172230_f3con_dl3)

- refit: `skipping vertex fit: only 2 segment(s) >= 1.00 cm (of 3)` -- the
  vertex STAYS at (-54.61, -87.67, 19.72), 0.32 cm from truth (DL snap kept).
- The 9.59 cm segment: `dirsign=1 pdg=2212 dir_weak=false` -- **proton,
  strong direction into its Bragg stop**.  PF: `e- 1651 MeV + proton 112 MeV
  + mu- 2 MeV` (truth proton ~113 MeV).  The sec-10 mu-/proton residual is
  RESOLVED: no vertex displacement => no near-empty bridge samples => the
  F1 retry decides proton through the normal path.
- The 1.15 cm vertex-activity stub remains in PF (2 MeV mu-, dqdx_ratio
  0.857 at the true vertex -- possibly genuine vertex activity there).

### Gates

- Compiled-config: production sweep (abtest/compile_all_cfg.sh) before ==
  after, zero diffs; SBND PR job diff = ONLY `fit_vertex_min_seg_length: 1`.
- doctest-clus 565/565; freshness proof done (lib 14:47:56 > edits).
- SBND off-arm (worktree cfg @ d3b5972a + final binary, dl_weights=""):
  mabc member-hash md5 2358f441 == F1-ON geometric reference
  (nupr_evt172230_f1on).  Byte-identical.
- uBooNE off-gate: sweep/f3coff_ub2, 35/35 events rc=0, all 35 ZIP member
  hashes IDENTICAL to the f1off_ub reference (diff of hashes.txt empty).
  (First attempt sweep/f3coff_ub lost 13 events to the M3 "file too short"
  race -- a concurrent 14:50 rebuild of build/ while the sweep loaded
  plugins; its 22 completed events already matched the reference.  Rebuilt
  build/==local/lib deterministically, reran full under the fresh label.)
- geometric-vs-DL knob-on arms: BYTE-IDENTICAL (both mabc 633f1ed7,
  nupr_evt172230_f3con_geo vs _f3con_dl3), same skip line, same final
  vertex -- the geometric vertexing again finds the DL answer on its own.

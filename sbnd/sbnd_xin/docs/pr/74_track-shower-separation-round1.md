# doc pr/74 — track/shower separation: four owner cases, diagnosed (round 1, no code change)

Status: **UNDERSTANDING ROUND — no C++ and no jsonnet is changed.** Owner's
instruction was to understand the four cases, compare against the prototype,
and propose fixes; the fix is next round. Every number below is reproduced by
a command in the Repro block. Owner's Bee set for the four events:
https://www.phy.bnl.gov/twister/bee/set/c402d3ac-7e75-4fd7-8cc2-8b7ad2329f84/event/list/
(bee_idx 0 = 90055, 3 = 469665, 4 = 142421, 5 = 53361).

**Headline.** All four cases are **legacy / prototype-faithful** behaviour, not
a cost of our SBND-local guards. Six of the SBND-ON shower knobs were ablated
one at a time and all together on the two "shower mis-ID'd as track" events
and the outcome did not change (§5). The one thing our own knobs do own is
*cosmetic but consequential*: pr/40 round 6's `michel_stem_muon_rescue` paints
90055's orphaned EM trunk `mu-` instead of `proton`, which is what turns a
nueCC signal topology into a muon at the neutrino vertex.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin   # symlink into wcp-porting-img/sbnd/sbnd_xin

# --- (0) READ-ONLY: paint verdict + PF tree, straight out of the archived Bee zips
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on48 90055 469665
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on19 142421 \
        --at 96.1,-74.5,232.3 --at 109.2,-77.5,219.5
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on50 53361

# --- (1) baseline with the segment dump + the two PDG/topology tracers
export WCT_PID_WRITE_DEBUG=2 WCT_SHOWER_TOPO_DEBUG=1 PR_EXTRA_STAGES=pr_display PR_JOBS=4
./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-base48 data 90055 469665
./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr74-base19 data 142421
./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr74-base50 data 53361
#   PDG-write trace  -> work-pr74-base*/pr_evt<ID>/stdout.log      (stderr)
#   topology trace   -> work-pr74-base*/pr_evt<ID>/wct_pr_evt<ID>.log
#   segment dump     -> work-pr74-base*/pr_evt<ID>/calib-pr-evt<ID>.json
#   baseline == production proof (member content hashes, M2):
for e in 90055 469665; do python3 ../../abtest/hash_archive.py \
    work-pr51r7-on48/pr_evt$e/mabc-pr.zip work-pr74-base48/pr_evt$e/mabc-pr.zip; done

# --- (2) ablation arms (attribution; every env pass-through already exists)
SBND_MICHEL_STEM_MUON_RESCUE=0 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-noF14-48   data 90055 469665
SBND_SHOWER_TOPO_RESET=0       ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-noreset-48 data 90055 469665
SBND_SHOWER_TOPO_DQDX_GUARD=0  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-nodqdx48   data 90055
SBND_SHOWER_TOPO_DQDX_GUARD=0 SBND_SHOWER_TOPO_RESET=0 \
        ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-nodqdx-noreset48 data 90055
SBND_MICHEL_STEM_MUON_RESCUE=0 SBND_SHOWER_TOPO_RESET=0 SBND_SHOWER_TOPO_DEMOTE_LEN=0 \
SBND_SHOWER_TOPO_DQDX_GUARD=0 SBND_SHOWER_TRAJ_STRAIGHT_GUARD=0 SBND_SHOWER_RECLASS_DQDX_GUARD=0 \
SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=0 \
        ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74-proto48 data 90055 469665
# trace-level variants used for the stage-3 -> examine_direction window (§3.1):
SBND_WCT_LOGLEVEL=trace ...    work-pr74-trace48 / work-pr74-nodqdx-trace48

# --- (4) PF-structure prevalence over the 117-event production manifest
python3 scripts/analysis/pr74/pr74_census.py \
        work-pr51r7-on48 work-pr51r7-on19 work-pr51r7-on50 \
        --trans-cm 3.0 --tsv docs/pr/74_pf_shape_census.tsv
```

Arm labels used below: `work-pr74-base{48,19,50}` (production baseline),
`work-pr74-{noF14-48, noreset-48, nodqdx48, nodqdx-noreset48, proto48}`
(ablations), `work-pr74-{trace48, nodqdx-trace48}` (trace-level).
Gate: `work-pr74-base*` `mabc-pr.zip` member hashes are **identical** to
`work-pr51r7-on{48,19,50}` on all four events, so `PR_EXTRA_STAGES=pr_display`
perturbs nothing and the diagnosis is of today's production.

## How to read the evidence without re-running anything

Two decoders make the whole diagnosis available from archived Bee zips:

- `0-shower_track-global.json`: `q == 15000` ⇒ painted **shower**, `q == 0` ⇒
  painted **track** (`MultiAlgBlobClustering.cxx:876-881`); `real_cluster_id`
  is the owning shower's start-segment encoded id.
- `0-mc.json` **is** the particle-flow tree (jstree nodes; `id` = the
  segment's `cluster_id*1000 + graph_index`, same encoding as
  `calib-pr-evt<ID>.json`'s `id`, `PrDisplayDump.cxx:414`).

**Caveat that governs every reading of the paint.** The paint takes shower
*membership* first (`seg_to_shower`) and falls back to the per-segment
`kShowerTrajectory` / `kShowerTopology` / `pdg==11` flags only for segments in
no shower. So "painted track" does **not** mean `kShowerTopology` is clear and
"painted shower" does **not** mean a shower test fired — 53361 is the proof (a
119 cm object painted shower that no shower test could have produced above
50 cm). Paint is *what the owner sees*; mechanism only comes from the tracers.

## Symptom (owner report) and what the data actually shows

| # | event | owner | measured |
|---|---|---|---|
| 1 | 90055 | EM trunk mis-ID'd as muon, main trunk missing | the 2020 MeV EM shower hangs off `mu- 60 MeV` seg **11045** (14.38 cm, 2.75× MIP), which is **not in any shower** (`shower_id = -1`) |
| 2 | 142421 | (96.1,−74.5,232.3) ID'd as track, "not in the particle flow" | that point is main-cluster seg **7013** (41.92 cm, 2.32× MIP, pdg 2212), painted track, **absent from the PF tree entirely** — 266 MeV invisible. (109.2,−77.5,219.5) *is* painted shower and *is* in PF, under `e- 612 MeV` 108106) |
| 3 | 53361 | clearly a muon, ID'd as electron | the whole event is one PF root `e- 405 MeV`: seg **27004**, 113.91 cm at **1.02× MIP** (a textbook MIP), particle_score 0.07 |
| 4 | 469665 | after the proton, everything should be one EM shower | `proton 78 MeV` (15004) → **`mu- 93 MeV` seg 15003** (27.59 cm, 0.95× MIP, score 0.21) → then the EM shower, which lives in **five other clusters** (58/68/66/33/63) as three root-level gammas |

Median dQ/dx (MIP = 56000 e/cm, the production `mip_dqdx`), from the calib dump:

| evt | seg | L cm | pdg | score | median dQ/dx | note |
|---|---|---|---|---|---|---|
| 90055 | 11045 | 14.38 | 13 | 0.15 | **2.75×** | the orphaned EM trunk |
| 90055 | 11044 | 10.23 | 11 | 0.26 | **2.88×** | the shower's own start segment — same charge regime, opposite verdict |
| 90055 | 11005 | 19.01 | 11 | 100 | 0.75× | ordinary shower body |
| 469665 | 15003 | 27.59 | 13 | 0.21 | 0.95× | mis-ID'd EM stem |
| 142421 | 7013 | 41.92 | 2212 | 0.27 | 2.32× | PF-invisible |
| 53361 | 27004 | 113.91 | 11 | 0.07 | **1.02×** | a MIP called an electron |
| 53361 | 27003 | 12.07 | 11 | 100 | 0.86× | the 12 cm segment that started it |

## Root cause, case by case

### 3.1 — 90055: the EM trunk never joins the shower, then gets painted `mu-`

Shower formation is a BFS from the main vertex
(`NeutrinoShowerClustering.cxx:90-140`): it walks outward and starts a shower
at the **first segment that is shower-like**, where shower-like means
`kShowerTrajectory || kShowerTopology || |pdg| == 11`. Segment 11045 satisfies
none of the three, so the BFS walks *past* it and starts the shower at 11044.
The trunk is structurally excluded — there is no later step that pulls a
vertex-attached stem into a shower it already skipped.

Why 11045 is not shower-like:

- `segment_is_shower_trajectory` cannot fire: the trunk is straight
  (`lsl/tel = 0.956`), and the test needs `length_ratio < 0.95`.
- `segment_is_shower_topology` **does** fire at stage 3 — the tracer prints
  `guard branch 1 L 14.4cm ... demoted false final_shower true`. But that log
  line is emitted *before* the pr/40 F3 `shower_topo_dqdx_guard` block
  (`PRSegmentFunctions.cxx:4007` prints, `:4026` vetoes), so it overstates the
  answer. The direct A/B settles it: at `determine_direction`, one millisecond
  later, production prints `Track ... pdg=2212` while the
  `SBND_SHOWER_TOPO_DQDX_GUARD=0` arm prints **`S_topo ... pdg=11`**. At
  2.75× MIP the guard reads the trunk as "decisively proton- or muon-like".
- **But turning the guard off does not fix the case.** In
  `work-pr74-nodqdx48` the segment is `S_topo pdg=11` at stage 3 and by
  `examine_direction` it is `is_shower=false` with `pdg=2212` again. Adding
  `SBND_SHOWER_TOPO_RESET=0` (`work-pr74-nodqdx-noreset48`) does not change
  that either. So a **second, legacy demotion** acts between
  `determine_direction` and `examine_direction`, and it is the one that
  actually decides the case.

**Open item (round 2's first step).** That second demotion is not attributable
with the no-code tracers available. Every `unset_flags(kShowerTopology)` site
was enumerated and each is excluded by the PDG trace (they all write a pdg
that the trace would have shown, and gidx=45 shows only
`0→2212 :2653`, `2212→2212 :2653`, `2212→13 NeutrinoPatternBase.cxx:331`).
Two live explanations remain: (a) a flag-clear on a path that writes no pdg,
or (b) `improve_vertex` **re-creating** the segment after stage 3, so the
final object never sees a shower test at all — consistent with there being
exactly **one** topology evaluation at L=14.4 cm in the whole event and
exactly two `:2653` PID writes. Distinguishing (a) from (b) needs one
env-gated sentinel on segment creation + flag transitions. That is a code
change, hence next round.

Finally, `override_michel_stem_muon` (`NeutrinoPatternBase.cxx:274-331`,
pr/40 round 6 F14, **SBND ON**) sees a straight long main-vertex segment typed
2212 whose far end has a shower-like sibling — the textbook "stopped muon →
Michel electron" topology — and relabels it `mu-`. On a nueCC event the
"Michel electron" *is the EM shower*, so the rescue fires on signal. With
`SBND_MICHEL_STEM_MUON_RESCUE=0` the trunk reads `proton 147 MeV` instead.
This does not restore the trunk to the shower, but it is the difference
between a proton and a **muon at the neutrino vertex** in a nueCC selection.

### 3.2 — 142421: a 41.9 cm, 266 MeV segment in a disconnected graph component

Seg 7013 runs between vertices 7013 and 7014. No other main-cluster segment
touches either: 7006 and 7011 and 7014 all hang off the main vertex 7009,
7010 hangs off 7012. So 7013 sits in a **disconnected component of the main
cluster's PR graph**, and the PF tree — which is built by walking out from the
main vertex — cannot reach it. The toolkit already knows this: pr/65 rung 3
prints, verbatim,

```
pr65 pf-orphan-audit: unclaimed seg=7013 cluster=7 pdg=2212 ke_mev=266.20 nfits=71 dirsign=1
pr65 pf-orphan-audit: 1 unclaimed segment(s), no PF node fabricated (pf_orphan_audit_only)
```

so the audit is working exactly as designed and, by design, fabricates
nothing. **This is new information relative to pr/65**, whose round-3 census
reported *0 unclaimed segments anywhere* — that census was nueCC48 only; the
NC-π⁰ sample has at least one. The owner's second point
(109.2,−77.5,219.5) is a different object: it belongs to shower 108106 and is
correctly painted and correctly present in PF as `e- 612 MeV`.

The track verdict on 7013 is separate and shallower: pdg 2212 written once by
`segment_determine_dir_track` (`PRSegmentFunctions.cxx:2653`) with
particle_score **0.27**, never revisited.

### 3.3 — 53361: an electron label cascades 114 cm down a MIP track

Main cluster 27 has exactly three segments. The PDG trace is a clean chain:

```
gidx=3 pdg 0 -> 13  at PRSegmentFunctions.cxx:2653     # 12.07 cm track PID
gidx=3 pdg 13 -> 11 at PRSegmentFunctions.cxx:2728     # shower-TRAJECTORY test fires
gidx=1 pdg 2212 -> 11 at NeutrinoVertexFinder.cxx:1397 # 0.76 cm, flag_shower_in branch
gidx=4 pdg 13 -> 11  at NeutrinoVertexFinder.cxx:1405  # 113.91 cm, flag_shower_in branch
```

`examine_direction` sets `flag_shower_in` as soon as *any* incoming segment at
a vertex is shower-flagged, and then relabels downstream segments electron
under three unconditional branches: `dirsign == 0`, `length < 2 cm`, or
`|pdg| == 13 || pdg == 0`. Segment 27004 — 113.91 cm at 1.02× MIP, i.e. as
MIP-like as a track can be — is relabelled by the third branch. There is
**no length ceiling, no dQ/dx veto, and no charge-consistency test anywhere in
the cascade.**

**Prototype comparison: this is faithful.** `NeutrinoID_track_shower.h:1998`,
`:2001`, `:2004` are the same three branches with the same conditions
(`get_flag_dir()==0`, `length<2.0*units::cm`,
`fabs(get_particle_type())==13 || ==0`), and the prototype has no guard
either. So case 3 is **not a porting defect** — it is a real algorithmic gap
that WCP shares, and any fix is a WCT improvement, not a correction.

### 3.4 — 469665: two independent failures

1. Seg 15003 (27.59 cm, 0.95× MIP) is typed `mu-` by the initial track PID
   (`PRSegmentFunctions.cxx:2653`) with particle_score **0.21** and never
   revisited by anything. Both shower tests decline it: the topology tracer
   reports `branch 0` (the 5-branch spread test found nothing), and the
   trajectory test sees a straight segment. A 27 cm MIP-like stem is genuinely
   ambiguous between "muon" and "EM trunk before the shower opens up" — but a
   0.21 score is the classifier saying it does not know, and nothing downstream
   treats that as a reason to defer.
2. The EM shower is not in the main cluster **at all**: it lives in clusters
   58, 68, 66, 33 and 63 and enters PF as three root-level `gamma` nodes
   (160 / 48 / 58 MeV) plus a 322 MeV `e-` under the muon trunk. That is a
   clustering / shower-connection failure upstream of track-shower separation
   and outside this doc's scope; it is the reason the owner sees fragments.

## Why it hid

- The `shower_topo dbg` line prints `final_shower` **before** the pr/40 F3
  dQ/dx guard can veto it, so the log reads "shower" for a segment that ends
  up a track. Anyone reading that log alone concludes the topology test is
  fine.
- The Bee paint uses shower *membership* first, so a mis-ID'd segment absorbed
  into a shower looks right and a correctly-flagged segment left out of one
  looks wrong. Neither is a statement about the classifier.
- Case 3's cascade is prototype-faithful, so no porting audit would flag it,
  and no knob exists to ablate it.
- Case 2's segment is *audited* (pr/65 rung 3 prints it) but the audit is
  deliberately non-fabricating, and the round-3 census that reported "0
  unclaimed anywhere" ran on nueCC48 only.

## PF-structure prevalence on the 117-event production manifest

From `0-mc.json` over `work-pr51r7-on{48,19,50}` (113 events have an archived
`mabc-pr.zip`). These count **PF-tree structure**, which is a genuine
structural signal; they are *not* mechanism footprints — those only come from
an A/B arm. TSV: `docs/pr/74_pf_shape_census.tsv`.

| shape | definition | events | % |
|---|---|---|---|
| A | `mu`/`pi` trunk off the PF root carrying an EM descendant ≥ 50 MeV (the 90055 shape) | **11** | 9.7 % |
| B | `e-` PF node ≥ 50 cm whose painted points have transverse RMS < 3.0 cm — a pencil, i.e. a track (the 53361 shape) | **19** | 16.8 % |
| C | ≥ 3 root-level `gamma` nodes — shower fragmentation (the 469665 shape) | **15** | 13.3 % |
| D | painted object ≥ 200 points with no PF node of that id (the 142421 shape) | **2** | 1.8 % |

Each owner case is picked up by its own shape: 90055 → A
(`11045: mu->e-:2020MeV`), 53361 → B (`27001: 119cm / rt 2.76cm`),
142421 → B + C + D (`7013: 2975pts`), 469665 → C.

Calibration for B's transverse cut: 53361's muon-called-electron is 2.76 cm;
real EM showers in the same sample are 6.5 cm (90055 seg 11044, 142421 seg
108106). A 1.0 cm cut finds only 2 events and **misses 53361**; 3.0 cm
includes it at 19/113. The threshold is fitted to include the known case and
is therefore an upper bound on the class, not a measurement of it.

Shape A's other ten events are worth an owner eye: 138009 (`mu->e- 1184MeV`),
389538 (`mu->gamma 997MeV`), 268067 (`pi+>e- 761MeV`), 285567
(`mu->e- 408MeV`), 71372, 69314, 342199, 52085, 54175, 55715.

## Ablation table (what our own knobs do and do not own)

Segment 11045 of 90055 across every arm:

| arm | knobs off | pdg | score | flag_shower | shower_id |
|---|---|---|---|---|---|
| `work-pr74-base48` | — (production) | **13** | 0.15 | false | −1 |
| `work-pr74-noF14-48` | `michel_stem_muon_rescue` | **2212** | 0.15 | false | −1 |
| `work-pr74-noreset-48` | `shower_topo_reset` | 13 | 0.15 | false | −1 |
| `work-pr74-nodqdx48` | `shower_topo_dqdx_guard` | 13 | 0.15 | false | −1 |
| `work-pr74-nodqdx-noreset48` | both | 13 | 0.15 | false | −1 |
| `work-pr74-proto48` | all six | **2212** | 0.15 | false | −1 |

469665's seg 15003 is `mu- / 0.21 / false / −1` in **every** arm including
`proto48`. Conclusion: the SBND-ON guards change 90055's *label* (mu ↔ proton)
and nothing else; they do not cause, and cannot fix, either case.

`shower_topo_demote_len = 50` deserves a separate note even though it explains
none of the four cases. Because `separate_track_shower`
(`NeutrinoTrackShowerSep.cxx:231`) only runs the trajectory test when the
topology test left the flag clear, and the trajectory test has the prototype's
own `length > 50 cm ⇒ false` early return (`ProtoSegment.cxx:547`,
`PRSegmentFunctions.cxx:1737`), **under SBND production no segment longer than
50 cm can be labelled a shower by `separate_track_shower` at all.** Both doors
shut at exactly 50 cm, measured by the same `segment_track_length(seg, 0)`.
The prototype has no such rule. Worth deciding on deliberately; not the cause
here.

## Prototype comparison

| mechanism | toolkit | prototype | verdict |
|---|---|---|---|
| `is_shower_trajectory` (>50 cm early return, 5-section wiggliness) | `PRSegmentFunctions.cxx:1711` | `ProtoSegment.cxx:543` | **faithful** |
| `is_shower_topology` (5-branch spread, `<0.25·L` guard) | `PRSegmentFunctions.cxx:3612` | `ProtoSegment.cxx:319` | **faithful** |
| `separate_track_shower` (topology first, trajectory only if clear) | `NeutrinoTrackShowerSep.cxx:210` | `NeutrinoID_track_shower.h:1` | **faithful** |
| `examine_direction` `flag_shower_in` cascade (3 relabel branches) | `NeutrinoVertexFinder.cxx:1389-1409` | `NeutrinoID_track_shower.h:1998-2007` | **faithful** (case 3's gap is shared) |
| `improve_vertex` topology re-examination, 0.09 / 0.06 score thresholds | `NeutrinoVertexFinder.cxx:2944-2982` | `NeutrinoID_improve_vertex.h:310-338` | **faithful** (toolkit recomputes `segment_is_shower_topology` where the prototype reads the cached flag — pr/32 §10.3) |
| shower-formation BFS from the main vertex | `NeutrinoShowerClustering.cxx:90` | `NeutrinoID` shower clustering | **faithful** |
| `shower_topo_demote_len`, `shower_topo_dqdx_guard`, `shower_traj_straight_guard`, `shower_reclass_dqdx_guard`, `shower_connect_main_vertex_straight_guard`, `michel_stem_muon_rescue` | pr/25, pr/40 | — | **SBND-local inventions, no counterpart** |

`clus/docs/porting/porting_dictionary.md` was checked: none of the divergences
above is listed as intentional, and none of the "faithful" rows is a candidate
for a parity fix — they are correct ports of algorithms that have a real gap.

## Proposals

All default-OFF, all preserving today's byte-identical path. Ranked by
(evidence strength × footprint × distance from a validated flip).

**P1 — `flag_shower_in` cascade needs a charge/length veto.** *(do this first)*
Case 3, mechanism fully attributed, prototype-faithful so nothing to break in
parity terms, and the knob is a pure conjunct on an existing branch.
Proposed `shower_in_cascade_guard` (bool, default false): in
`examine_direction`'s `|pdg|==13 || pdg==0` branch
(`NeutrinoVertexFinder.cxx:1405`), refuse the electron relabel when the
segment is **long** (> `shower_in_max_len`, start 40 cm) **and** MIP-like
(median dQ/dx < `shower_in_mip_hi` × MIP, start 1.3). 53361's 27004 is
113.9 cm at 1.02× — squarely caught; the shower bodies in 90055 sit at
0.75-0.90× but are 7-20 cm, so the length conjunct spares them.
Expected footprint: bounded above by shape B, 19/113 events; the real number
is smaller because B does not require the segment to be *relabelled* by this
branch. Gate: off-arm 0/117 byte-identical; on-arm nusel 0/117 required.

**P2 — do not let a stopping-muon rescue fire on a vertex-attached EM stem.**
Case 1's label half, fully attributed and single-line. Proposed
`michel_stem_require_track_far_end` (bool, default false): in
`override_michel_stem_muon` (`NeutrinoPatternBase.cxx:311-322`), require the
shower-like sibling at the far end to be **small** (below a length/energy
ceiling) — a genuine Michel electron is a few cm and tens of MeV, whereas
90055's sibling is a 2020 MeV shower. Footprint: shape A, ≤ 11/113, and F14's
own footprint is smaller still. This does not fix the missing trunk; it stops
a nueCC signal from carrying a muon at its vertex.

**P3 — instrument, then close, the second demotion in case 1.** *(round-2
first action, precedes any knob)* One env-gated sentinel logging every
`kShowerTopology` set/clear with a call-site, plus segment creation, over the
window between `determine_direction` and `examine_direction`. This is the only
open attribution in the doc and the only thing standing between us and a real
fix for the 90055 class (shape A, ≤ 11/113 events, including 1184 MeV and
997 MeV showers). No behaviour change; the sentinel ships default-off.

**P4 — let a low-confidence track PID defer instead of deciding.** All four
mis-ID'd segments carry particle_score 0.05-0.27 — the classifier reporting
that it does not know — yet the label is treated as final everywhere
downstream (shower-formation's `|pdg|==11` test, `flag_shower_in`, the Michel
rescue). Proposed `track_pid_low_score_defer` (double, default 0): below the
threshold, mark the segment ambiguous so the *topology* answer wins where the
two disagree. This is the most principled fix and by far the widest blast
radius — it should be prototyped offline on the 117-event manifest before any
implementation, and it is explicitly **not** recommended for round 2.

**P5 — decide `shower_topo_demote_len = 50` deliberately.** It closes the only
remaining shower door above 50 cm (§ablation). It explains none of the four
cases, so this is a review item, not a fix: either accept it with the
reasoning written down, or narrow it (e.g. only when the segment is also
MIP-like) behind a new default-OFF knob.

**Not proposed:** anything for 469665's cluster fragmentation (5 clusters, 3
root gammas). That is upstream of track/shower separation and belongs to the
clustering/shower-connection family (pr/62, pr/63, pr/65), not here.

## Verification plan for round 2

Whatever ships must clear the standard bar: knob-off gate **0/117
byte-identical** on `abtest`/`hash_archive.py` member hashes across
`work-*-{48,19,50}`; compiled-config proof (`wcsonnet` diff off, key present
on); `./build/clus/wcdoctest-clus` rc=0 with a new synthetic doctest for each
new predicate; on-arm **nusel 0/117** required before any production flip; and
every mover Bee'd with an annotated index for owner adjudication.

## Status / open items

- Cases 1-4 all diagnosed; **case 1's second demotion is the one open
  attribution** (§3.1) and is P3's target.
- pr/65's "0 unclaimed segments anywhere" does not hold on NC-π⁰: 142421 seg
  7013 (266 MeV) is unclaimed and PF-invisible today (§3.2). Owner call
  whether rung 2 (`connection_type=3` fallback, deferred in pr/65) should be
  revisited.
- No code changed this round. No knob added. No production default touched.

# pr/83 — Duplicate fitted trajectories on long tracks: `break_segment` slices by an unoriented edge

Owner report (2026-08-15): mcp1k events **283040, 59899, 72586** show a "weird
multiple tracks situation" — for the long track, multiple fitted trajectories
come back and forth, overlapping.  This doc measures the pathology, attributes
it to a single root cause, ships the fix as the default-OFF knob
**`break_seg_orient`**, and validates it per the standard gates.

## 0. Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# measure the pathology on any arm (reporting tool, exit 0 always):
python3 scripts/pr83_dup_metric.py work-mcp1k-prod0813 --events 283040,59899,72586

# baseline repro at HEAD (A-side of every gate below):
PR_EXTRA_STAGES=pr_display PR_JOBS=3 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr83-base data 283040 59899 72586

# lineage probe (pre-existing WCT_DET_DEBUG=2 instrumentation, PRGraph.cxx):
WCT_DET_DEBUG=2 PR_EXTRA_STAGES=pr_display PR_JOBS=1 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr83-lineage2 data 283040       # + lineage3 for 59899 72586

# controls:
SBND_SGP_MAX_SEP=-1        ... work-pr83-sgpoff  data 283040 59899 72586   # pr/73 guard OFF
SBND_PR_FIND_OTHER_ROUNDS=1 ... work-pr83-rounds1 data 283040 59899 72586  # fos 1 round

# fix on / off (per-event gates):
PR_EXTRA_STAGES=pr_display PR_JOBS=3 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr83-off3 data 283040 59899 72586
SBND_BREAK_SEG_ORIENT=true PR_EXTRA_STAGES=pr_display PR_JOBS=3 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr83-on3  data 283040 59899 72586

# full-sample gates (PR_JOBS=32, owner-authorized):
for s in nuecc48 ncpi0 mcp1k; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=32 ./run_pr_chain_batch.sh \
      work-$s-cb0805 work-$s-pr83off data $(cat products/prod0813/events-$s-prod0813.txt)
  SBND_BREAK_SEG_ORIENT=true PR_EXTRA_STAGES=pr_display PR_JOBS=32 ./run_pr_chain_batch.sh \
      work-$s-cb0805 work-$s-pr83on  data $(cat products/prod0813/events-$s-prod0813.txt)
done
```

Toolkit commit: (filled below, §7).  All numbers below from
`work-mcp1k-prod0813` (identical in `ma10` / `ma10k20-harv2` / `pr83-base` —
the pathology is deterministic and epoch-stable).

## 1. Measured pathology

Two visible forms, one generator (§3):

| event | cluster | form | numbers |
|---|---|---|---|
| 283040 | 2 | three parallel segments | segs 2004/2005/2008, 70–76 cm each, pairwise overlap 0.80–0.99 at 1.4 cm, chord angles 1–3°; vertices 2003/2005 coincident to 5 µm |
| 59899 | 5 | out-and-back ×2 | seg 5005: 68.1 cm path between vertices **1.2 cm apart**, turnaround exactly at vtx 5000 (1.1 cm), return leg 0.31 cm median from outbound; chain 5006+5009 = second out-and-back.  ~4 traversals of a ~35 cm track |
| 72586 | 17 | out-and-back (giant) | seg 17004: **633 cm** path between vertices **2.2 cm apart**; on-charge 0–100 cm, ghost bridge 100–300 cm (3–9 cm off charge), then the real ~300 cm track traversed in return |

Physics harm — each duplicate books its energy in `kine_energy_particle`:
`kine_reco_Enu` = 965 MeV (283040: the one track counted as 210 MeV π + 201 MeV
µ + a third arm), 654 MeV (59899: 133+125+183 MeV for one ~35 cm track),
1517 MeV (72586: a 1399 MeV "muon" from the doubled track).

The smoking-gun output observable: **vertex `fit_distance`** (wcpt-to-fit
distance in the calib json).  283040 vertex 2000 = **118.0 cm**, vertex 2003 =
**67.0 cm**; 72586 = **301 cm**.  A healthy vertex is < 3 cm.

## 2. Diagnosis trail (what was ruled out)

- **Baseline repro**: HEAD (`ef54d486` + this doc's knob, off) is
  byte-identical to `work-mcp1k-prod0813` on all six archives
  (`hash_archive.py` member-content rollup; `mabc-pr.zip` +
  `pctree-pr-evt*.tar.gz` × 3 events), and same-binary reruns are bit-stable.
- **`sgp_max_sep=-1` control**: duplicates identical → the pr/73 route guard
  is exonerated.
- **`pr_find_other_rounds=1` control**: duplicates identical → round-2
  residual re-claiming is NOT the generator (the prototype's sticky
  `flag_tagged_steiner_graph` divergence — toolkit recomputes per call,
  `NeutrinoOtherSegments.cxx:56` vs `NeutrinoID_proto_vertex.h:802` — is real
  but is not what creates these duplicates; noted for the record, not fixed
  here).
- **`traj_cover_probe` rerun**: `find_other_segments` adds only ~1 cm stubs on
  these clusters; the long duplicates exist before/after it unchanged.
- **Lineage probe** (`WCT_DET_DEBUG=2`, pre-existing per-edge backtrace in
  `PRGraph.cxx::add_segment`): every graph-edge birth attributed; see §3.

## 3. Root cause

`break_segment` (`clus/src/PRSegmentFunctions.cxx:1005`) splits a segment at a
point by slicing its wcpts/fits into a front half and a back half — and hands
them to the vertices as **(boost source, boost target)**:

```cpp
auto seg1 = make_segment(graph, vtx1, vtx);   // vtx1 = boost::source
auto seg2 = make_segment(graph, vtx, vtx2);   // vtx2 = boost::target
seg1->wcpts({wcpts.begin(), itwcpts+1});      // FRONT half of the path
seg2->wcpts({itwcpts,       wcpts.end()});    // BACK half
```

Boost edges carry no orientation — the function's own WARNING comment (added
in an earlier campaign) says source/target do NOT correspond to
`wcpts.front()/back()` and that callers needing orientation must use
`find_vertices()`.  The very next lines assume the correspondence anyway.

The reversed parent edges are real and routine: the `examine_vertices` Type
I/II re-routes re-add segments with `add_segment(g, seg, v1, v2)` where the
freshly routed path runs v2→v1.  Lineage capture, 283040 (mm):

```
WCT_DETA seg idx=6 nw=178 v1=(-58.4,1052.0,1662.7) v2=(-686.7,1196.6,2651.2)
                          wf=(-686.7,1196.6,2651.2) wb=(-58.4,1052.0,1662.7)
```

v1 sits at the path's BACK (`wb`), so when `break_two_end_dqdx` later breaks
this segment at its 49.8° junction (`fit idx 90`, (-45.16,106.01,220.02) cm),
each child receives the half terminating at the *other* vertex:

- child (near-end vertex ↔ junction) carries the FAR-arm wcpts →
  `get_ordered_segment_vertices` (wcpt-based) orients it "backwards" and the
  near vertex's **fit** is written at the junction (283040 vtx 2003,
  fit_distance 67.0 cm);
- child (junction ↔ far-corner vertex) carries the NEAR-arm wcpts → the far
  vertex's fit lands at the near end (vtx 2000, fit_distance 118.0 cm).

Because `multi_trajectory_fit` pins segment endpoints to **vertex fit points**
(`TrackFitting.cxx:4246-4259`, pr/73 §4.7), the far-arm segment is then
*fitted* onto the near arm: two stacked trajectories, the far arm's charge
orphaned.  The fit-vs-wcpt divergence then cascades:
`search_for_vertex_activities` (`NeutrinoVertexFinder.cxx:320`) collects
candidates within 1.5 cm of the vertex **fit** point but builds its stub path
from the vertex **wcpt** — with the two 67 cm apart it plants a third
trajectory-length "vertex activity" bridge (lineage idx=30, 3 wcpts spanning
67.7 cm).  59899 and 72586 are the same chain
(`examine_vertices_4` reversed re-route → break → crossed children), with the
crossed pieces later merged into single out-and-back segments (5005: 68 cm
between vertices 1.2 cm apart; 17004: 633 cm between vertices 2.2 cm apart).

**Prototype parity check (M15)**: WCP cannot cross a split — it resolves
(start_v, end_v) by **wcpt-index equality tested in both orientations**
(`prototype_base/pid/src/NeutrinoID_proto_vertex.h:595-601`) before slicing.
The orientation invariant was lost in the port.  The sibling toolkit function
`break_segment_into_two` (`NeutrinoPatternBase.cxx:1853`) already uses
`find_vertices()` and is not affected.

## 4. Fix — knob `break_seg_orient` (C++ default false)

`break_segment` gains `orient_split` (default false = legacy, byte-identical).
When true it resolves the parent's (front, back) vertices with
`find_vertices(graph, seg)` — the orientation-safe helper its own comment
prescribes — *before* the edge is removed, and slices accordingly.  Threaded
as `m_break_seg_orient` on `PatternAlgorithms` and passed at all three
`break_segment` call sites (`break_two_end_dqdx`, the shower start-segment
break in `shower_clustering_with_nv_from_vertices`, and
`snap_main_vertex_to_kink`); config key `break_seg_orient` on
`TaggerCheckNeutrino` with the key-suppression idiom in
`cfg/pgrapher/common/clus.jsonnet`, TLA-threaded through
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`; runner env
`SBND_BREAK_SEG_ORIENT`; default pinned in `doctest_clus_knob_defaults.cxx`.

## 5. Validation

- `./build/clus/wcdoctest-clus`: **2064/2064 assertions pass**.
- Freshness proof: `local/lib/libWireCellClus.so` 15:41 > last edit 15:39.
- **Compiled-config proof**: knob off → `break_seg_orient` absent from the
  compiled JSON (key-suppression); `-A break_seg_orient=true` → the ONLY diff
  is the added key.
- **Knob-off byte-identity (3 events)**: `work-pr83-off3` vs `work-pr83-base`
  — 6/6 archives PASS (hash_archive rollup).
- **Knob-on, 3 events** (`work-pr83-on3`): `pr83_dup_metric.py` → **0 dup
  pairs, 0 retraces, 0 folds** (was 6 pairs + 2 out-and-backs).  Structure:

| event | quantity | OFF | ON |
|---|---|---|---|
| 283040 | cluster-2 total fitted length | 225.2 cm (3× one arm) | **131.8 cm** (= 53.3 + 68.2 two-arm topology) |
| 283040 | worst vertex fit_distance | 118.0 cm | **1.5 cm** |
| 283040 | kine_reco_Enu | 965 MeV | **431 MeV** |
| 59899 | cluster-5 total fitted length | 149.8 cm (~4 traversals) | **41.9 cm** |
| 59899 | worst vertex fit_distance | 2.9 cm | 1.2 cm |
| 59899 | kine_reco_Enu | 654 MeV | **355 MeV** |
| 72586 | cluster-17 total fitted length | 637.5 cm | **312.3 cm** (single 308 cm µ) |
| 72586 | worst vertex fit_distance | 301.0 cm | **0.8 cm** |
| 72586 | kine_reco_Enu | 1517 MeV | **827 MeV** |

- **Knob-off full three samples**: §6.1.
- **Knob-on full-sample A/B**: §6.2–6.4.

## 6. Full-sample gates (nueCC48 47 + NCpi0 19 + mcp1k 445 = 511 events)

### 6.1 Knob-off byte-identity — PASS, 1022/1022 archives

`work-{nuecc48,ncpi0,mcp1k}-pr83off` (HEAD + knob code, knob off) vs
references, `hash_archive.py` member-content rollup on `mabc-pr.zip` +
`pctree-pr-evt*.tar.gz` per event:

- 889/1022 identical to the `prod0813` arms outright;
- the remaining 133 (7 nuecc48 / 3 ncpi0 / 123 mcp1k) differ from `prod0813`
  but are **identical to the `ma10` arms** — i.e. they carry the already-
  shipped pr/79 operating-point change (`dl_vtx_min_accept_score` 4→10),
  which post-dates the prod0813 arms.  0/1022 archives are attributable to
  this change.  Gate logs: `/home/xqian/tmp/pr83-fulloff-gate.log`,
  `pr83-ma10gate.log`.

### 6.2 Pathology census — strictly down, zero new

`pr83_dup_metric.py` over every event (dup pair = overlap ≥ 0.7 @ 1.4 cm +
chord angle < 20°; retrace/fold = L/C ≥ 5 with/without return-leg overlap):

| sample | OFF: dup-pair evts / retrace evts | ON | verdict |
|---|---|---|---|
| nueCC48 | 3 / 2 | 3 / 1 | 400474 retrace FIXED |
| NCpi0 | 4 / 0 | 4 / 0 | untouched |
| mcp1k | 6 / 5 | 4 / 0 | 283040+59899 dups, 285564/353223/394796/72586 retraces-folds FIXED |

Findings 25 → 12.  **Zero new pathologies.**  Every one of the 12 persisting
findings is numerically byte-identical between arms (events untouched by the
knob) and is a short (11–25 cm) near-parallel prong pair — shower-like
topology, a different class from the reversed-break defect, left for a future
round if the owner wants it.

### 6.3 Selection / kinematics A/B — 8/511 events change (1.6 %)

`pr83_ab_compare.py` (numu/nue/cosmict scores, `kine_reco_Enu`, main vertex):

- **NCpi0: zero changes.  nueCC48: nue scores unchanged on all 47** (no
  nue-selection efficiency impact); only 400474 moves (its fixed retrace):
  numu 1.32→0.72, Enu −11 MeV, vtx 1.2 cm.
- mcp1k: 7 movers = exactly the fixed-pathology events + 66712 (same
  mechanism, sub-threshold pathology).  Full rows in the doc TSVs
  (`pr83-ab-*.tsv`).
- **Both >1 cm main-vertex movers land ON the owner's own hand labels**
  (`dl_vtx_training/data/full473/manifest.tsv`):
  - 59899: 34.90 cm from the hand vertex → **3.40 cm** (the owner's
    corrective label at (-194.61,-9.10,14.59) is where the fix now puts it);
  - 285564: 11.45 cm → **0.69 cm**.
- numu-score sign flips: 59899 (1.25→−1.53) and 353223 (1.16→−1.20), both
  events whose baseline reco was a multiply-traversed track; the post-fix
  reco is structurally correct (single track, single-counted energy), so the
  score change reflects honest inputs, not a regression mechanism.

### 6.4 Runtime / RSS — neutral

Quiet-machine 3-event pairs: 9/8/10 s (off) vs 10/9/10 s (on); maxRSS median
1500 MB both.  (Batch medians 19 vs 21 s are load-contaminated — the on batch
shared the box with the 1022-archive hash gate; p90 identical at 25 s.)  The
knob adds one `find_vertices` call per break — O(1).

## 7. Ship record

- Toolkit knob commit + separate SBND production flip commit
  (`wct-pr-perevt.jsonnet`: `break_seg_orient = true`), applied after §5+§6
  passed — flip pre-authorized by the owner for this campaign ("If the
  validation pass, turn on the knob for SBND production").
- Post-flip smoke: bare 3-event rerun byte-identical to `work-pr83-on3`.
- Hashes/labels: `work-pr83-base` (pre-change A-side), `work-pr83-off3`,
  `work-pr83-on3`, `work-*-pr83{off,on}` full arms; gate logs under
  `/home/xqian/tmp/pr83-*.log` (summaries quoted above).
- Toolkit commits (apply-pointcloud): `175e4d00` (knob, default OFF, +
  doctest + cfg plumbing), `f24639d9` (SBND production flip ON).
- wcp-porting-img: this doc + `scripts/pr83_dup_metric.py` +
  `scripts/pr83_ab_compare.py` + the `SBND_BREAK_SEG_ORIENT` runner mapping.
- Not addressed here (recorded for a future round): the 12 persisting short
  near-parallel prong pairs (§6.2, different class); the toolkit's
  non-sticky `find_other_segments` tags divergence
  (`NeutrinoOtherSegments.cxx:56` vs prototype `PR3DCluster` member —
  measured NOT to be the generator of these events' duplicates).

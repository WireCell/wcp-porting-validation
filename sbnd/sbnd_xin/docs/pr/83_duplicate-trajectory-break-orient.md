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
- Bee (8 changed events, index `83_bee.index.txt`):
  before <https://www.phy.bnl.gov/twister/bee/set/804e92bc-b4de-435c-9ebb-cbfaadf80b5e/event/list/>,
  after <https://www.phy.bnl.gov/twister/bee/set/0e31e8d3-cf18-45f6-a4fe-07748f4afbc7/event/list/>.
- Not addressed here (recorded for a future round): the 12 persisting short
  near-parallel prong pairs (§6.2, different class); the toolkit's
  non-sticky `find_other_segments` tags divergence
  (`NeutrinoOtherSegments.cxx:56` vs prototype `PR3DCluster` member —
  measured NOT to be the generator of these events' duplicates).

## 8. Round 2 (2026-08-17) — the stacked-prong class the round-1 fix does not cover

**This round's root cause is a DIFFERENT mechanism from §3's title defect.**
`break_seg_orient` (round 1, shipped SBND production ON 2026-08-15) is
unaffected and stays on. Owner report: mcp1k data event **138009** still
shows "multiple tracks" stacked on one track, and asked for a scan of the
1000-event sample for more cases. **Scope, owner-decided: investigate +
document only this round — no code, no knob, no production flip.**

### 8.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# census (free — reads existing production PR arms, no reruns):
python3 scripts/pr83r2_census.py \
    work-vfnuecc48-cbr3on work-vfmcp1k-cbr3on work-vfncpi0-cbr3on \
    --tsv docs/pr/83_r2-census.tsv

# extractor cross-validation (against the 9 events that still carry both
# calib-pr-evt*.json and mabc-pr.zip):
python3 scripts/pr83_dup_metric.py work-r1qlmc-prod0813 work-r2mc-prod0813

# controls (base + 2 mechanism probes, 8 class-A events):
NUE="138009 168596 268784 74544"; MC="349945 64409 174224 281837"
PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-vfcbr3on  work-pr83r2-base-nue      data $NUE
PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1kall-vfcbr3on work-pr83r2-base-mc      data $MC
SBND_MVGA_INTERPOSED=false    PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-vfcbr3on  work-pr83r2-nointerp-nue   data $NUE
SBND_MVGA_INTERPOSED=false    PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1kall-vfcbr3on work-pr83r2-nointerp-mc    data $MC
SBND_MVGA_SPLICE_STRAIGHTEN=0 PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-vfcbr3on  work-pr83r2-nostraight-nue data $NUE
SBND_MVGA_SPLICE_STRAIGHTEN=0 PR_JOBS=4 ./run_pr_chain_batch.sh work-mcp1kall-vfcbr3on work-pr83r2-nostraight-mc  data $MC
```

Freshness proof: `local/lib/libWireCellClus.so` timestamped 2026-08-17
12:39:42, source files (`TaggerCheckNeutrino.cxx`/`NeutrinoVertexFinder.cxx`,
the round-3 cathode-rescue commit) edited 12:39:08 — the .so postdates the
edits; `strings` on the .so finds the round-3 knobs' compiled log strings.
Reads current SBND production (toolkit `apply-pointcloud` HEAD `2d8c9e5a`).

### 8.1 Sample identity

**138009 is a nueCC48 event, not a member of the 1000-event data sample.**
The 1000-event MCP2025C data sample
(`input_files_reco1/staged-mcp2025c-1000evt`) is PR'd at 445/1000 events in
`work-vfmcp1k-cbr3on/` (the other 555 have never had the PR stage run); it is
censused separately below, alongside nueCC48 and NCpi0 — the same three-sample
manifest §6 used (511 events total).

### 8.2 Census — 511 PR'd events, current production, from `mabc-pr.zip`

No surviving PR arm at ≥200 events still carries `calib-pr-evt*.json` (both
`work-*-pr83on` from round 1 and everything since have been superseded or
retired). Worked around it: `scripts/pr83r2_census.py` extracts per-segment
fitted geometry straight from each event's `mabc-pr.zip` →
`data/0/0-track_fit-global.json` (the Bee display payload, keyed by
`real_cluster_id`), then reapplies §1's exact metric (overlap ≥ 0.7 @ 1.4 cm,
chord angle < 20°, min length 10 cm).

**Extractor validated**: on the 9 events that still carry both files
(`work-r1qlmc-prod0813` 4 + `work-r2mc-prod0813` 5), the Bee segment set is
identical to the calib `segments` list — same ids, same point counts, 0
bee-only, 0 calib-only, 0 length mismatch > 0.05 cm. Feeding the Bee-derived
segments through `pr83_dup_metric.py`'s own `analyze_event()` unchanged
reproduces 138009's 12 pairs exactly.

| arm (sample) | events PR'd | events with ≥1 dup-pair finding |
|---|---|---|
| `work-vfmcp1k-cbr3on` (1000-evt data sample) | 445 (425 with a track_fit layer — see below) | 6 |
| `work-vfnuecc48-cbr3on` (nueCC48) | 47 | 7 |
| `work-vfncpi0-cbr3on` (NCpi0) | 19 | 4 |

20/445 mcp1k events have no `track_fit-global.json` in their `mabc-pr.zip`
(19 checked: `rc=0`, no `nu-candidate` row in `nusel-evt*.tsv` — the tagger
found nothing to write a track_fit layer for; 1 checked (`391854`) does have
a `nu-candidate` row but it is a 7.1 cm 43-point stub, likely below whatever
size floor the display visitor applies). Not investigated further — a
coverage caveat on the extractor, not a defect claim.

**17 (event, cluster) findings. Crosstab against each event's
`wct_pr_evt<ID>.log` for `mvga: op3 ... carried>=2` on the SAME cluster: 8
co-locate exactly (class A), 9 do not (class B).**

Full table: `docs/pr/83_r2-census.tsv`.

- **Class A — 8, new, mvga-carry co-located.**

  | event | sample | cluster | dup pairs | segs | sum fitted len | longest member |
  |---|---|---|---|---|---|---|
  | 138009 | nueCC48 | 12 | 12 | 6 | 204 cm | 43 cm |
  | 168596 | nueCC48 | 14 | 1 | 2 | 71 cm | 40 cm |
  | 268784 | nueCC48 | 13 | 1 | 2 | 49 cm | 30 cm |
  | 74544 | nueCC48 | 12 | 1 | 2 | 66 cm | 38 cm |
  | 349945 | mcp1k | 18 | 2 | 4 | 58 cm | 18 cm |
  | 64409 | mcp1k | 8 | 2 | 4 | 68 cm | 25 cm |
  | 174224 | mcp1k | 20 | 1 | 2 | 185 cm | 168 cm |
  | 281837 | mcp1k | 63 | 1 | 2 | 38 cm | 28 cm |

- **Class B — 9, no carry, not diagnosed this round.** nueCC48 `246579`
  (clus 19), `269774` (13), `46363` (19); mcp1k `350935` (clus 11, a **251 cm**
  duplicated segment — a long-track shape, not the short prong-pair class
  below), `404684` (9, 62 cm, cathode-crossing); NCpi0 `359980`, `506114`,
  `506746`, `521075`. The four NCpi0 findings match §6.2's post-round-1 NCpi0
  count of 4 exactly — consistent with being the same "short near-parallel
  prong pair" residual class §6.2 already deferred.

Specificity: across all 511 events, **41 (event, cluster) sites** carry ≥2
prongs via mvga op3; only **8 (~20%)** show the stacked-duplicate metric. A
future fix therefore cannot simply disable every carry ≥ 2 without pricing
what the other 33 sites' pr/86 benefit costs — see §8.4.

Context only, **not attributable**: §6.2 counted 11 dup-pair events
post-round-1 vs 17 now, but that delta spans pr/79, pr/85, pr/86, pr/89, pr/90
and the doc-73 cathode-rescue rounds; §6's `work-*-pr83on` arms are retired,
so no apples-to-apples arm survives to attribute the delta to any one change.

### 8.3 Root cause — mvga op3's interposed-splice carry, and why op1 misses it

138009 cluster 12's `wct_pr_evt138009.log`:

```
mvga: op3 created-splice    cluster=12 stub_arc=15.87cm carried=2 vf_kept=0
mvga: op3 splice-straighten cluster=12 carried=2 straightened=2 reach=26.49cm
mvga: op3 stub-interposed   cluster=12 len=3.05cm vf_deg=3 carried=2 far_angle=168.9deg
mvga: op3 splice-straighten cluster=12 carried=4 straightened=3 reach=35.43cm
mvga: op3 stub-interposed   cluster=12 len=8.33cm vf_deg=5 carried=4 far_angle=167.2deg
mvga: fired cluster=12 op1=0 op2=0 op3=4 (refit done)
```

`carried=2` then `carried=4` = the six stacked prongs
(12088/12091/12092/12093/12094/12095), all fitted from main vertex 12117
`(-103.3, 162.1, 81.5)`, 12 duplicate pairs at overlap 0.73–1.00 / angle
2–15°, 204 cm of fitted length on a ~43 cm trunk. The PF tree books them as
six separate electrons (673+78+105+102+518+188 MeV) plus a 223 MeV proton →
`kine_reco_Enu = 1973.6 MeV` for what is physically one shower trunk.

**Mechanism** (`clus/src/NeutrinoGraphAudit.cxx`, doc pr/85 2026-08-15 +
pr/86 2026-08-16): op3's interposed-stub absorb (pr/85) deletes an interposed
stub and re-attaches **every** far prong directly to the anchor
(`carried_prong_execute` — this is `carried`); op3's round-2
`splice-straighten` (pr/86 §15) then re-derives each carried prong's
near-anchor stretch straight over `reach` = 26–35 cm — so N carried prongs
each acquire the *same* trunk geometry near the anchor. `op1=0` on every
class-A event: op1 (the pre-existing duplicate-corridor merge, §3-adjacent
but pr/51-era, unrelated to `break_seg_orient`) runs **before** op3 in the
same pass, and `in_scope_segments()` explicitly skips the `created` set —
"segments created by this pass's own reconnects are exempt from every op ...
no delete/recreate cycling." The spliced prongs are structurally invisible to
the merge that would otherwise have caught them. Confirmed empirically: every
one of the 8 class-A events logs `op1=0`.

### 8.4 Controls — which half of the mechanism generates the stacking

Both `carried_prong_execute` (the far-prong reattach, pr/85) and
`splice-straighten` (the near-anchor re-derivation, pr/86) run inside the
same `if (m_mvga_interposed) { ... }` block; `SBND_MVGA_SPLICE_STRAIGHTEN=0`
is the narrower env knob (skips only the straighten step in principle),
`SBND_MVGA_INTERPOSED=false` the broad one (declines the whole interposed
pathway). Both wired in `run_pr_chain_batch.sh`. All 8 class-A events, base
byte-identical to production first (`hash_archive.py` PASS 8/8 mabc+pctree
before either control is read):

| event | dup pairs base→interp-off→straighten-off | `kine_reco_Enu` MeV base→interp-off→straighten-off |
|---|---|---|
| 138009 | 12 → **0** → **0** | 1973.6 → **1441.3** → **1441.3** |
| 168596 | 1 → **0** → **0** | 3151.4 → **1465.5** → 2727.0 |
| 268784 | 1 → **0** → 3 (**worse**) | 2381.9 → **1696.8** → 2324.5 |
| 74544 | 1 → **0** → **0** | 2828.5 → **2197.4** → **2197.4** |
| 349945 | 2 → **0** → 1 | 990.1 → **501.8** → 561.1 |
| 64409 | 2 → 1 (class B, pre-existing) → 1 (same) | 1386.9 → 1095.4 → 1095.4 |
| 174224 | 1 → **0** → 1 (unchanged) | 831.2 → **257.7** → 829.9 |
| 281837 | 1 → **0** → **0** | 1275.4 → **911.6** → 1065.2 |

**`SBND_MVGA_INTERPOSED=false` is the clean, reliable control: 8/8 class-A
findings clear to 0.** The one residual (`64409`, dropping from 2 pairs / 4
segs to 1 pair / 2 segs) is pre-existing and unrelated to the carry: its log
shows op1 firing normally (`op1 dup-merge cluster=8 removed seg len=4.13cm
... overlap=1.00@14.0mm ... reconnects=0`), and the leftover pair is a short
(27/16 cm) near-parallel prong of the §6.2/§8.2-class-B shape, not the
interposed-splice pathology.

**`SBND_MVGA_SPLICE_STRAIGHTEN=0` is unreliable**: it clears 4/8 (matching
the interposed-off numbers exactly on 138009/74544, but only partially on
168596/281837 — cleared as a *dup-pair metric* but with a much smaller Enu
recovery, meaning the geometry still overlaps below the 1.4 cm / 0.7 metric
threshold even though it no longer counts as a "duplicate pair"), leaves 2
untouched or nearly so (`174224` unchanged, `349945` only reduced 2→1), and
makes one **worse** (`268784`: 1 pair → 3 pairs — disabling only the
straighten step while still doing the carry apparently produces a *less*
straight, *more* dispersed set of near-anchor stretches that newly overlap
each other). Every class-A event's `kine_reco_Enu` under `interp-off` drops
substantially (966–1656 MeV), which is the double-count being removed rather
than inferred.

**Conclusion: the carry itself is the generator, not the straightening.** A
fix should decline the interposed carry above some prong count, not merely
skip straightening a carry that still happens.

`numu_score`/`nue_score` also move under `interp-off` (e.g. 138009:
0.109 → −0.129; 174224: 3.04 → −0.36) — expected: these are genuinely
different reconstructions once the double-counted energy is gone, the same
pattern §6.3 saw for round 1's fix ("the score change reflects honest
inputs, not a regression mechanism"). No adjudication against hand labels
was done this round — investigation only, no knob to gate on those labels
yet.

### 8.5 Fix design (not implemented this round)

**`mvga_carry_max`** (int, default `0` = unlimited = legacy): decline the
interposed absorb/splice when it would carry more than this many far prongs.
`1` keeps the stub as the shared trunk — the physically correct topology
(main → stub → far vertex → N prongs) — and removes the double count at
source, consistent with §8.4's finding that the carry (not the straighten)
is the generator. Cost: of the 41 carry-sites measured full-sample (§8.2),
33 do not show the stacking metric; capping carry at 1 would decline all of
them too, giving back some of pr/86's measured benefit (Class-B 90→48,
orphans 118→82, +4 nue recoveries per the pr/86 SBND-flip commit message) —
that trade needs its own A/B before shipping, not assumed here.

Touch list for whoever implements it: `clus/src/NeutrinoGraphAudit.cxx` (the
`m_mvga_interposed` block — `carried_prong_execute` / the `carried` counter),
`clus/inc/WireCellClus/NeutrinoPatternBase.h` (member, alongside
`m_mvga_splice_straighten`), `clus/src/TaggerCheckNeutrino.cxx` (`get()` +
`default_configuration()` round-trip + `pattern_algos.` thread),
`cfg/pgrapher/common/clus.jsonnet` (arg + key-suppression idiom),
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet` (TLA thread, left
OFF), `clus/test/doctest_clus_knob_defaults.cxx` (pin the default), and a
`SBND_MVGA_CARRY_MAX` mapping in `run_pr_chain_batch.sh`. Gates it would owe
(CLAUDE.md §4): knob-off byte-identical over the 511-event manifest
(`hash_archive.py` rollup), knob-on census 8 → 0 with zero new findings via
`scripts/pr83r2_census.py`, a `numu_score`/`nue_score`/`kine_reco_Enu`/vertex
A/B with >1 cm movers adjudicated against
`dl_vtx_training/data/full473/manifest.tsv`, and
`./build/clus/wcdoctest-clus`.

### 8.6 Class B — characterised, not diagnosed

Nine findings with no mvga carry co-located. Four (NCpi0) match §6.2's
post-round-1 residual count exactly and are almost certainly the same short
(11–25 cm) near-parallel prong-pair class already deferred there. Three more
(nueCC48 `246579`/`269774`/`46363`) are the same shape (24–37 cm sum length,
2-seg findings). Two do not fit that pattern and are flagged for a future
round: `350935` (mcp1k, clus 11) is a **251 cm** duplicated segment on what
its short x-range (−20.6 … −15.5 cm) suggests is a single long track, not a
near-vertex prong pair; `404684` (mcp1k, clus 9, 62 cm) is cathode-crossing
(x range −10.0 … +12.5 cm), the one class-B finding where the doc-73
cathode-rescue-round-3 knobs (`esva_ignore_empty_2d` etc., SBND production ON
as of `2d8c9e5a`) are in scope and un-investigated here. No code, no root
cause claimed — per CLAUDE.md §5's tie-breaker, named for the owner rather
than fixed in this round.

### 8.7 Bee evidence

`docs/pr/83_bee-r2.index.txt` (does **not** overwrite round 1's
`83_bee.index.txt`, M13):

- Class A, production vs `SBND_MVGA_INTERPOSED=false` (the control, not a
  shipped fix): before
  <https://www.phy.bnl.gov/twister/bee/set/79f60a2e-d5b7-4461-9ada-81208403762b/event/list/>,
  after
  <https://www.phy.bnl.gov/twister/bee/set/47292f54-6ec7-4784-a5c0-eac3b8a0a9f3/event/list/>
  (idx 0–7 = 138009/168596/268784/74544/349945/64409/174224/281837).
- Class B exemplars (production only):
  <https://www.phy.bnl.gov/twister/bee/set/608d6105-22f6-4ed1-b379-3b7021f15053/event/list/>
  (idx 0–2 = 350935/404684/246579).

Note: the first upload of all three sets was broken — only `bee_idx 0`
rendered, the rest were empty. Cause: Bee's per-event member names are
`data/<idx>/<idx>-<file>` (the numeric filename prefix must match the
directory index), and the combiner script used to build these multi-event
zips only renumbered the directory, leaving every member's filename prefixed
`0-` regardless of idx — so idx 1–7 pointed Bee at files with the wrong
names. Fixed and re-uploaded; the links above are the corrected sets.

### 8.8 Records

- `scripts/pr83r2_census.py` — the validated `mabc-pr.zip` extractor + mvga
  carry-site crosstab (this round's tool; supersedes nothing — used alongside
  `pr83_dup_metric.py`, not instead of it, wherever `calib-pr-evt*.json`
  still exists).
- `docs/pr/83_r2-census.tsv` — all 17 findings, member segment ids included.
- `docs/pr/83_bee-r2.index.txt` — links above.
- Work arms (fresh labels, nothing written under an existing one, M13):
  `work-pr83r2-{base,nointerp,nostraight}-{nue,mc}` (8 events × 3 arms × 2
  samples). Not retired by this round; not protected either — ordinary
  scratch, safe to clean up in a future retire pass.
- Toolkit commit this round: two new opt-in TRACE diagnostics (§9.1), no
  behavior change. `wcp-porting-img`: this doc section + the four files
  above.

## 9. Round 2 continued (2026-08-17 pm) — class B root cause and fix design

Owner asked to dig into §8.6's 9 undiagnosed class-B findings and propose a
fix (design only, not implemented). **Result (updated after a follow-up
round on the 2 initially-unresolved events, §9.5): three distinct,
well-evidenced mechanisms account for 8 of the 9; the 9th (`506746`) was
owner-reviewed and is not a defect.**

### 9.1 New diagnostics (committed, opt-in, byte-identical when off)

Additions to `clus/src/NeutrinoGraphAudit.cxx` / `TaggerCheckNeutrino.cxx`,
all proven byte-identical to production on every event they touched
(`hash_archive.py` PASS on every rerun below, before and after each of 5
builds this round):

- **Four new TRACE lines inside op1** (the existing duplicate-corridor
  merge): `op1 scope` + `op1 scope-member` (which segments were even
  `in_scope_segments()` this pass, and the main-vertex point `mvga_radius`
  is centered on), `op1 angle` (the computed chord angle whenever the
  overlap gate passed — previously silent), `op1 find-vertices-failed` and
  `op1 reconnect-infeasible` (the two remaining silent `continue` sites).
  Gated at `SPDLOG_LOGGER_TRACE` — invisible at production's `debug` level,
  visible with `SBND_WCT_LOGLEVEL=trace`, no code path changed.
- **`dup_stage_census()`** (new, `TaggerCheckNeutrino.cxx`, gated on a new
  env var `WCT_DUP_STAGE_DEBUG`, independent of the existing `WCT_DET_DEBUG`
  checksum tool): at 7 pipeline checkpoints (`main:find_proto_vertex`,
  `main:determine_main_vertex`, `overall_main_vertex`,
  `snap_main_vertex_to_kink`, `main_vertex_graph_audit`,
  `improve_vertex`/`examine_direction`, `shower_clustering_with_nv` — the
  first two added in a follow-up round, §9.5, once the first five proved
  insufficient to locate 350935's generator) scans **every** pair of
  segments on `main_cluster` — unscoped, unlike op1 — with op1's own metric
  (path-point fraction within 1.4 cm, chord angle), logging any pair
  clearing overlap 0.5 (looser than production's 0.8, to see a pair
  *approaching* duplication) plus an unconditional segment-count line per
  checkpoint. Answers "does this duplicate already exist, and is it in
  op1's scope, at each named stage" without per-segment coordinate matching.
- **Coordinate-matching against `WCT_DET_DEBUG=2`'s pre-existing lineage
  backtraces** (§9.5): the tool is unchanged, but note for future use —
  its `v1`/`v2`/`wf`/`wb` fields are **millimeters**, and the raw fprintf
  output lands in `stdout.log`, not the spdlog `wct_pr_evt<ID>.log` (round 1's
  doc used mm-scale examples throughout but did not call the unit out
  explicitly; this round lost time to a cm/mm mismatch before finding it).

Repro:
```bash
SBND_WCT_LOGLEVEL=trace WCT_DUP_STAGE_DEBUG=1 PR_JOBS=1 ./run_pr_chain_batch.sh \
    <ql_root> <out> data <event>
```
Freshness/gates: `wcbuild` five times this round (one per instrumentation
addition), each followed by `./build/clus/wcdoctest-clus` (2105/2105 PASS
every time) and a hash-identity rerun of the affected event(s) before reading
any new TRACE line.

### 9.2 Mechanism A — op1's `mvga_radius` scope excludes the duplicate (6/9)

`246579, 269774, 46363, 359980, 506114, 521075` — **including all four NCpi0
findings.** In every one of these, op1's own `op1 scope-member` listing (the
segments actually inside `mvga_radius` = 15 cm of the main vertex) does
**not** contain the two segments my census flags as a duplicate pair — by
length, and for 246579 (the only case with 3+ in-scope segments, so worth
checking beyond "only 1 candidate exists") independently confirmed with
`dup_stage_census`: the exact pair (12.8/24.1 cm, **overlap 0.86, angle
2.1°** — a textbook duplicate, well clear of even the pre-pr/86 0.7
threshold) is present **from the very first checkpoint** (`overall_main_vertex`,
before `main_vertex_graph_audit` even runs) and **unchanged** through every
later checkpoint — op1 simply never sees it. Distance from the main vertex
to the nearer endpoint of each member: 19.8 cm and 21.4 cm — **5–7 cm outside
the 15 cm radius**, not a large excursion.

For 359980/506114/521075/46363, op1's scope contains only **1** segment
each (`n_in_scope=1`) — a pair evaluation is structurally impossible; the
duplicate lives entirely outside the audited window. 269774's scope has 2
segments (24.75/7.64 cm, overlap 0.29 — correctly declined, a real non-dup
pair) but the actual duplicate (14.1/16.3 cm) is a *different* pair,
likewise outside scope.

Exact overlap/angle for all 6 (final Bee geometry, `pr83r2_census.py`
metric): 246579 0.86/2.1°, 269774 1.00/5.6°, 46363 0.70/17.2°, 359980
0.82/7.3°, 506114 0.81/18.2°, 521075 0.96/8.5°. Five of six clear even
production's raised 0.8 threshold comfortably — **scope, not threshold, is
what excludes them.**

### 9.3 Mechanism B — op1's threshold, raised by pr/86, is stricter than the census's (1/9)

`404684` only. Op1 evaluates this *exact* pair — confirmed by matching
lengths (18.13/61.52/15.78 cm in scope; 15.78/61.52 cm the flagged pair) —
computes **overlap 0.74**, and declines because production sets
`mvga_dup_frac = 0.8` (`wct-pr-perevt.jsonnet`, pr/86 §10.6 P4: "close the
marginal-overlap gap", raised from the C++ default 0.7 for an unrelated
purpose). The census script (and doc 83 round 1's `pr83_dup_metric.py`, its
comment claiming "same constants as NeutrinoGraphAudit op1" is now **stale**
post-pr/86) used 0.7. `dup_stage_census` confirms the geometry is stable —
final overlap 0.741 matches op1's mid-pipeline 0.74 almost exactly, so this
is not a timing artifact: **op1 saw this pair and declined it purely on the
now-stricter threshold.**

### 9.4 `506746`: owner-reviewed, not a defect

Bee (production only, `docs/pr/83_bee-r2.index.txt`):
<https://www.phy.bnl.gov/twister/bee/set/4e9e6863-8bf0-4943-8dc3-3d9c22513561/event/list/>
(idx 1). Owner looked at both remaining candidates and confirmed 506746 is
fine — the 15.9/11.5 cm pair is not a real duplication. No further
investigation.

### 9.5 Mechanism C — `350935`: the ORIGINAL main cluster is abandoned mid-pipeline, and never audited again

`350935` **is a real defect**, now fully root-caused: it is not scope, not
threshold, but a *third, distinct* mechanism — `main_cluster` gets silently
re-pointed away from cluster 11 (the one the bundle/nusel layer selected and
what Bee ultimately reports) to a different, unrelated cluster 17, and
**every single downstream duplicate-corridor audit runs on 17, never again on
11.** Owner-flagged coordinate `(-16.3, 35.2, 292.2)` sits (within ~4 cm)
exactly on the vertex where the two duplicate segments meet.

**Trail (`WCT_DET_DEBUG=2` lineage + two new `dup_stage_census` checkpoints
at `main:find_proto_vertex` and `main:determine_main_vertex`, both proven
byte-identical to production, `wcdoctest-clus` 2105/2105 throughout):**

- `TaggerCheckNeutrino: selected main cluster 11` (bundle/nusel selection,
  before `TaggerCheckNeutrino::visit()` even starts pattern recognition —
  this is the cluster id Bee ultimately reports, fixed at this point).
- `find_proto_vertex` on cluster 11 → `init_first_segment` creates the seed
  of the eventual 251.4 cm long segment; `find_other_segments` (called
  immediately after, same `find_proto_vertex` call) creates the 13.4 cm stub
  from the identical vertex `(-16.51, 37.39, 294.82)` cm. `dup_stage_census`
  at this checkpoint already reads **overlap 1.00, angle 4.0°, len
  13.4/250.7 cm** — the defect is born in the *very first* pattern-recognition
  step, before main-vertex determination, before any audit could possibly
  run.
- `determine_main_vertex` on cluster 11 (still the same duplicate, now
  13.4/251.4 cm — its final, Bee-reported lengths) picks the far end of the
  251 cm muon as cluster 11's own vertex candidate, `compare_main_vertices:
  selected vertex (-19.01,-99.01,57.97) score=-0.625` — a **negative**
  score, i.e. a weak/unconvincing determination.
- `determine_overall_main_vertex` (the ordinary, non-DL fallback — separate
  from and *not* gated by pr/89's `dl_vtx_swap_guard`/
  `main_vertex_swap_apply`, which only cover the DL path) then evaluates
  other candidate clusters, runs a *fresh* `find_proto_vertex` on a
  genuinely different, unrelated cluster (physically ~800 cm away in x, near
  what becomes the event's real interaction region), and
  `check_switch_main_cluster` calls `swap_main_cluster(new=17, old=11, ...)`
  (log: `pr59 assoc-census: swap_main_cluster 11 -> 17`,
  `NeutrinoPatternBase.cxx:3440`). **From this point on `main_cluster` is
  17** — confirmed directly: `main_vertex_graph_audit`, `improve_vertex`,
  `shower_clustering_with_nv` all log `cluster=17` for the rest of the event
  (§9.4's old table, now explained rather than mysterious).
- Cluster 11 goes into `other_clusters` and is never revisited by any
  duplicate-corridor pass. Its two segments' final lengths (251.39 cm,
  13.35 cm) match their state immediately after that first
  `determine_main_vertex` call almost exactly (250.7→251.4 cm across the two
  checkpoints, both well before the swap) — **nothing after the swap ever
  touches cluster 11 again**, yet it is what Bee (and `nusel-evt350935.tsv`'s
  `main_id=11`) reports, because the bundle-level cluster identity assigned
  before `TaggerCheckNeutrino::visit()` started is never updated to reflect
  the swap.

This is a **structural gap, not a threshold or scope tuning issue**: no
knob controls whether an *abandoned* main cluster gets a duplicate-corridor
pass before being set aside — op1/mvga's entire design assumes it always
runs on the cluster that ends up being reported, which `swap_main_cluster`
silently breaks. §9.2/§9.3's op1-only knobs (`mvga_op1_radius`,
`mvga_op1_dup_frac`) would not have caught this even if fully unscoped and
threshold-relaxed, because op1 never runs on cluster 11 at all after the
swap — this is orthogonal to Mechanisms A and B, not a variant of them.

Bee (production only, `docs/pr/83_bee-r2.index.txt`):
<https://www.phy.bnl.gov/twister/bee/set/4e9e6863-8bf0-4943-8dc3-3d9c22513561/event/list/>
(idx 0 = 350935 mcp1k — the duplication is the two segments meeting at
`(-15.5, 37.5, 295.7)` cm, i.e. the owner-flagged coordinate).

### 9.6 Fix design (not implemented this round)

Three independent, narrowly-targeted knobs, one per confirmed mechanism —
deliberately **not** a single "raise everything" change, since op1's current
scope and threshold were each set for reasons unrelated to this defect
(mvga_radius bounds an expensive near-vertex audit; 0.8 closed a different
marginal-overlap gap in pr/86) and loosening either broadly risks
re-litigating that tuning:

- **`mvga_op1_radius`** (default `0` = use `mvga_radius`, legacy): a
  separate scope radius for op1 only, decoupled from op2/op3's near-vertex
  focus (op1 is a global graph-correctness check — "no two segments should
  double-book the same charge" — not inherently a *near-vertex* one, unlike
  op2/op3 which are specifically about vertex activities). §9.2's measured
  excursion (5–7 cm past 15 cm) suggests 25 cm would already catch 246579;
  the fully general fix is `mvga_op1_radius = -1` meaning **unscoped** (scan
  the whole main cluster, as `dup_stage_census` already does for free) —
  simplest to reason about, and the 511-event manifest's op1 cost is already
  small (§8's runtime numbers were RSS/wall neutral with op1 active).
- **`mvga_op1_dup_frac`** (default `0` = use `mvga_dup_frac`, legacy): a
  separate overlap threshold for op1's own merge decision, distinct from
  `mvga_dup_frac`'s other consumer (op3's satellite-anchor absorb gate,
  `mvga_sat_dup_frac`'s fallback). Set to the pre-pr/86 0.7 to recover
  404684 without touching op3's pr/86-tuned behavior at all.
- **`swap_orphan_dup_audit`** (default `false`, Mechanism C, §9.5): inside
  `swap_main_cluster` (`clus/src/NeutrinoPatternBase.cxx:3440`, the single
  choke point for all 3 call sites in `NeutrinoVertexFinder.cxx` — the
  DL-path one already covered by `dl_vtx_swap_guard`/`main_vertex_swap_apply`
  plus the two non-DL ones neither of those touches), when true, run op1's
  duplicate-corridor merge **unscoped** (no vertex to center a radius on —
  the whole point is this cluster is being abandoned) on `old_main_cluster`
  before it is set aside into `other_clusters`. One-shot, cheap (op1 is
  already proven RSS/wall neutral in §8), and directly closes the gap: an
  abandoned cluster that is *still* what the bundle/nusel layer reports
  gets exactly the one duplicate-corridor pass it would otherwise never
  receive. Does not address *why* the bundle-selected cluster (11) and the
  pattern-recognition-refined cluster (17) can diverge in the first place —
  that is a bigger question (should Bee/nusel's reported cluster id track
  the swap?) intentionally left out of scope for a single knob.

The two op1 knobs are additive and should compose: with both set, 6/9
(§9.2) + 1/9 (§9.3) = **7 of 9 class-B findings should merge cleanly**.
`swap_orphan_dup_audit` is independent of them (§9.5 confirmed op1 never
runs on cluster 11 at all, scoped or not) and targets 350935 specifically;
between the three, **8 of 9 class-B findings have a proposed fix** — only
the owner-reviewed non-issue (§9.4, 506746) needs nothing.

Touch list: `clus/src/NeutrinoGraphAudit.cxx` (op1's `in_scope_segments()`
radius check and the `frac < m_mvga_dup_frac` gate — two one-line changes to
consult the new members when set), `clus/src/NeutrinoPatternBase.cxx`
(`swap_main_cluster` — the new unscoped op1 call on `old_main_cluster`),
`clus/inc/WireCellClus/NeutrinoPatternBase.h` (three new members alongside
`m_mvga_radius`/`m_mvga_dup_frac`), `clus/src/TaggerCheckNeutrino.cxx`
(`get()` + `default_configuration()` + `pattern_algos.` thread),
`cfg/pgrapher/common/clus.jsonnet` (key-suppression idiom),
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet` (TLA thread,
left OFF), `clus/test/doctest_clus_knob_defaults.cxx`. Gates it would owe:
knob-off byte-identical over the 511-event manifest; knob-on census
6/9+1/9+1/9 → 0 (via `pr83r2_census.py`, cross-checked against
`dup_stage_census` TRACE — the swap-orphan case needs a fresh `WCT_DET_DEBUG`
or `pr59 assoc-census` confirmation that op1 actually ran on the pre-swap
cluster); a `numu_score`/`nue_score`/`kine_reco_Enu`/vertex A/B on the 8
recovered events (NCpi0's untouched-in-round-1 status makes it the most
informative arm to watch here — all 4 of its class-B findings are Mechanism
A); `./build/clus/wcdoctest-clus`.

## 10. Round 3 (2026-08-17 evening) — the fixes, implemented, tuned and flipped

Owner instruction: implement the §9.6 knobs **and** the class-A fix (§8.5),
"make sure it does not regress much on the pr/86 issues"; validate on nueCC +
NCpi0 + mcp1k(1000) + mcp2k(2000) PR'd events (mcp2k regenerated — no PR
product survived the 2026-08-17 retire round); iterate at small scale first;
flip SBND ON if validation passes; Bee before/after links; unsure movers to
the owner for scanning.

### 10.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# knob-off byte-identity (1022/1022 over the 511-event manifest):
python3 scripts/pr83r3_hash_gate.py work-pr83r3-gateoff-mc    work-vfmcp1k-cbr3on
python3 scripts/pr83r3_hash_gate.py work-pr83r3-gateoff-nue   work-vfnuecc48-cbr3on
python3 scripts/pr83r3_hash_gate.py work-pr83r3-gateoff-ncpi0 work-vfncpi0-cbr3on
# census A/B (baseline arms carry calib dumps: PR_EXTRA_STAGES=pr_display):
python3 scripts/pr83r2_census.py work-pr83r3-gateoff-{mc,nue,ncpi0} --tsv /dev/null
python3 scripts/pr83r2_census.py work-pr83r3-gateon2-{mc,nue,ncpi0} --tsv /dev/null
# scores/movers A/B:
python3 scripts/pr83r3_scores_ab.py work-pr83r3-gateoff-mc work-pr83r3-gateon2-mc
# pr/86 regression metric (per arm):
PR86_DUMP_ARMS=work-pr83r3-gateoff-mc:work-pr83r3-gateoff-nue:work-pr83r3-gateoff-ncpi0 \
  python3 pr86_orphan_census.py
# knob-on operating point (runner envs; == the flipped production values):
#   SBND_MVGA_OP1_RADIUS=-1 SBND_MVGA_OP1_DUP_FRAC=0.7 \
#   SBND_MVGA_OP1_POST=true SBND_SWAP_ORPHAN_DUP_AUDIT=true
# mcp2k event list: work-cbr3-census-on ql_evt ids minus the 1000 mcp1k ids
# (work-pr40r7cen-mcp1k) == exactly 2000; arms work-pr83r3-m2koff / -m2kon.
```

### 10.1 What shipped (five knobs, all C++ default OFF/legacy)

| knob | default | SBND production | mechanism |
|---|---|---|---|
| `mvga_op1_radius` | 0 = use `mvga_radius` | **-1 (unscoped)** | §9.2 Mech A |
| `mvga_op1_dup_frac` | 0 = use `mvga_dup_frac` | **0.7** (≥10 cm pairs only, §10.3) | §9.3 Mech B |
| `mvga_op1_post` | false | **true** | §8 class A |
| `swap_orphan_dup_audit` | false | **true** | §9.5 Mech C + §10.4 |
| `mvga_carry_max` | 0 = unlimited | OFF (not needed) | §8.5 fallback |

Touch list as predicted in §9.6 plus: `clus/src/NeutrinoPatternBase.cxx`
(`swap_main_cluster` gains optional `Graph*/TrackFitting*/dv` params, all 4
call sites threaded), `clus/src/TaggerCheckNeutrino.cxx` (the §10.4
pre-shower sweep), `clus/src/NeutrinoGraphAudit.cxx` (op1 knob consults,
`op1-post` pass, `orphan_dup_audit` fork), `run_pr_chain_batch.sh`
(five `SBND_*` env mappings).  `wcdoctest-clus` green at every build.

### 10.2 Class A: op1-post must run AFTER the refit, and iterate

The §9.6-style "post-op3 dup pass" (before op4) cleared **0 of 7**
carry-stacked events: measured at TRACE (74544/138009), the carried prongs
read overlap 0.30–0.50 *pre-refit* — it is the op4 refit that routes them
onto the shared charge ridge (≥0.7, the census geometry).  Moved after the
refit it cleared 5/8; the last three (138009/74544/268784) re-emerged at
0.71–0.81 after op1-post's *own* refit — each merge round's refit pulls the
remaining prongs further onto the consolidated ridge.  Final form: iterate
merge-round → refit to a **fixed point** (kEditCap-bounded); 138009 took two
rounds (5 merges + 1).  Class A cleared 8/8 with `mvga_carry_max` left OFF —
the owner's no-pr/86-regression constraint satisfied *by construction* (no
carry is ever declined) and *by measurement* (§10.6).

### 10.3 Mech B's threshold needed a length gate — the 390842 lesson

The flat 0.7 op1 threshold produced one catastrophic mover in the mcp1k A/B:
**390842**, where a **1.91 cm** rider at the main vertex merges at overlap
0.75 (production 0.8 declines it; in scope even at 15 cm) and the kine tree
then loses the entire 1.03 GeV muon chain — `kine_reco_Enu` 1147.8 → **9.8
MeV**, numu_score 4.30 → 0.80 (the muon vanishes from `T_kine` entirely, not
even flagged excluded).  Bisect: with only the threshold reverted the event
is byte-restored, so the other three knobs have zero footprint there.  Two
more short-rider merges at 0.75 flipped scores the same way (285567 op1
1.98 cm, 268067 op1 1.66 cm).  A main-vertex-incidence guard (pr/86 P4's
split) did NOT catch these — the riders' endpoints are near, not on, the
main vertex.  What separates every adverse merge from the one wanted Mech-B
merge (404684: 15.8/61.5 cm at 0.74) is **length**: the relaxed threshold now
applies only when both members are ≥ 10 cm — the census's own min-length for
a duplicate finding, i.e. exactly the class the knob exists to recover.
Sub-10 cm riders stay at `mvga_dup_frac` (0.8).  With the gate: 390842
byte-restored, 404684 still fixed, all 16 targets still clear.

### 10.4 Mech C generalises: 359980 is a LOSING-CANDIDATE orphan (no swap)

§9.2 misattributed 359980 to Mechanism A.  With the knobs on, its duplicate
(cluster **75**) survived — because the main cluster is **21** and no
`swap_main_cluster` ever fired: cluster 75 went through `find_proto_vertex`
as a candidate during `determine_overall_main_vertex`, lost the contest, and
kept its segments in the final output with no audit — §9.5's gap without the
swap.  `swap_orphan_dup_audit` therefore grew a second layer: one unscoped
`orphan_dup_audit` over every **non-main cluster**, run right before
`shower_clustering_with_nv` consumes them (so shower maps never hold a
removed segment), sorted by cluster id.  This also covers 350935 (still
audited at the swap site first) and merged 506746's 15.9/11.5 cm pair at
overlap 0.95 — an event the owner reviewed as NOT a defect, flagged in
§10.7's scan list for that reason.

### 10.5 Gates

- **Knob-off byte-identity**: 1022/1022 archives (445+47+19 events × mabc +
  pctree) vs `work-vf{mcp1k,nuecc48,ncpi0}-cbr3on` — labels
  `work-pr83r3-gateoff-{mc,nue,ncpi0}`, PASS.  nueCC's 48th ql event
  (116962) has no production reference (the vf arm holds 47) — run, rc=0,
  reported, not comparable.  Final-binary spot checks after each of the two
  guard rebuilds: 6/6 and 6/6.  `PR_EXTRA_STAGES=pr_display` proven
  output-inert first (506114: mabc+pctree MATCH production with dumps on).
- **Compiled config**: knob-off wcsonnet output byte-identical (full PR
  pipeline incl. tagger_check_neutrino); knob-on shows exactly the five keys.
- **`wcdoctest-clus`**: 2115/2115 at the final binary (5 new knob-default
  pins).
- **Census, 511 events**: 17 findings (8 class A + 9 class B) → **0**, zero
  new findings.  mcp1k 6→0 (445 evts), nueCC 7→0 (48), NCpi0 4→0 (19).
  506746 (owner-cleared, §9.4) merges by design (§10.4) with **zero score
  footprint** (numu/Enu byte-level unchanged — the merge touches only the
  non-main display cluster).
- **mcp2k 2000×2**: baseline `work-pr83r3-m2koff` 2000/2000 rc=0 (46
  first-pass failures were this session's own mid-batch `wcb install` races
  — "failed to load plugin: WireCellClus" — rerun clean); production
  provenance anchored by a 20/20 archive-hash match against the surviving
  `work-cbr3-census-pr-on` on 10 shared mcp2k events, plus 6/6 + 6/6
  mixed-binary spot checks across the two mid-batch rebuilds.  Baseline
  census: **872 PR'd events, 14 findings (6 class A / 8 class B)** —
  `docs/pr/83_r3-census-m2koff.tsv`.  Knob-on A/B: §10.6a.
- **pr/86 regression**: orphan census off-vs-on — NCpi0 identical, nueCC
  identical, mcp1k **improved** by 2 pooled CUT orphans (66→64, histogram
  otherwise identical).  The class-A fix declines no carry (op1-post merges
  after the fact), so the constraint holds by construction and by
  measurement.

### 10.6 Numbers (A/B over the final arms, 511-event manifest)

| sample | events | findings off→on | movers (numu>0.05 / nue>0.05 / Enu>5 MeV / vtx>1 cm) |
|---|---|---|---|
| mcp1k (`gateoff-mc` vs `gateon2-mc`) | 445 | 6 → 0 | 16 (one 1.7 cm vtx move: 172090, unlabeled, scan list) |
| nueCC48 (`-nue`) | 48 | 7 → 0 | 18 (no downward nue-selection flip; 46363 flips UP — a recovered target) |
| NCpi0 (`-ncpi0`) | 19 | 4 → 0 | 7 |

Mover TSVs: `docs/pr/83_r3-movers-{mcp1k,nuecc48,ncpi0}.tsv`.  The recovered
targets dominate the large Enu movers (74544 −502, 268784 −396, 64409 −291,
404684 −164, 174224 −213 MeV — the double count leaving); secondary movers
are small refit drifts.  390842 and 394642 — the flat-0.7 adverse cases —
do not appear at all under the §10.3 length gate (byte-restored).

### 10.6a mcp2k A/B

Arms `work-pr83r3-m2koff` / `work-pr83r3-m2kon`, 2000/2000 events each
(rc=0 all; ql_root `work-cbr3-census-on`).  Census
(`docs/pr/83_r3-census-m2ko{ff,n}.tsv`): 872 PR'd events, **14 findings
(6 class A / 8 class B) → 0, zero new findings**.  Scores A/B
(`docs/pr/83_r3-movers-mcp2k.tsv`): 2000 joined, 41 movers.  13/14 census
events move (the double count leaving); 323301 clears with zero score
footprint (display-only, like 506746).  28 movers are sub-census
duplicates (below the 10 cm census min-length or the 0.7 census overlap
gate) newly merged by the knobs.

Attribution of the 8 largest (fresh debug rerun `work-pr83r3-diag8`,
byte-identical to the m2kon arm — hash gate 16/16): every merge is a
genuine duplicate corridor at overlap 0.75–1.00 @14 mm; the dominant
mechanism is `op1-post` catching post-refit duplicates that production's
scoped 0.8-gated op1 misses.  Notables (→ §10.7 scan list): 159569
(Enu 1697→593 from an overlap-1.00 vertex-region pair + refit), 396222
(class A; 15.3 cm + 30.3 cm stacked segments leave, Enu 4117→5639, numu
0.69→−0.87), 289508 (op1+op1-post+orphan-dup all fire, nue −15→−4.3),
179912 (numu 0.23→2.38 UP — recovered selection), 497311 (numu
−0.07→0.92, crosses 0), 58919 (2.15 cm rider @0.75 via op1-post,
Enu −203; op1-post is deliberately un-length-gated, §10.3, so this class
is expected — flagged for scan).

### 10.7 Owner-scan list (movers I cannot adjudicate)

Bee set: 12 events, before (knob-off) vs after (knob-on = production
post-flip), index `docs/pr/83_bee-r3.index.txt` (sets 5/6).  In Bee-index
order: 285567 (nue −4.2 → +4.3 from a 2.5 cm overlap-1.00 op1-post
merge), 268067 (nue 2.6 → 1.6, Enu −115), 168596 (nue 4.3 → 1.8, Enu
+236), 46363 (nue −2.3 → +0.8, Enu +250 — recovered Mech-A target),
172090 (1.7 cm vertex move, unlabeled), 506746 (owner-cleared 15.9/11.5
pair now merged at 0.95 by the orphan sweep — zero score footprint,
display-only), then the mcp2k six: 159569, 58919, 289508, 179912,
396222, 497311 (§10.6a).

### 10.8 Records

- Knob-off gate: 1022/1022 member-content identical
  (`work-pr83r3-gateoff-{mc,nue,ncpi0}` vs `work-vf{mcp1k,nuecc48,ncpi0}-cbr3on`);
  mcp2k provenance spot-checks 6/6+6/6+20/20 vs `work-cbr3-census-pr-on`.
- Knob-on arms: `work-pr83r3-gateon2-{mc,nue,ncpi0}`,
  `work-pr83r3-m2ko{ff,n}`; debug attribution `work-pr83r3-diag8`.
- Flip verification: compiled-config diff = exactly the four keys
  (`mvga_op1_radius=-1, mvga_op1_dup_frac=0.7, mvga_op1_post=true,
  swap_orphan_dup_audit=true`; `mvga_carry_max` absent); bare-config
  smoke `work-pr83r3-smoke-{nue,mc,ncpi0}` (138009 46363 349945 350935
  359980 506746) hash gate **12/12** vs the gateon2 arms — bare run ==
  production knob-on.
- `wcdoctest-clus` 2115/2115.
- Bee sets (index `docs/pr/83_bee-r3.index.txt`):
  - 511 recovered (17 evts) before: https://www.phy.bnl.gov/twister/bee/set/b8edf683-5cb8-4d1e-a041-0148e8a591a2/event/list/
  - 511 recovered after: https://www.phy.bnl.gov/twister/bee/set/fc044d24-d936-4bb0-b2ac-5ce37169a547/event/list/
  - mcp2k recovered (14 evts) before: https://www.phy.bnl.gov/twister/bee/set/61d5e03c-f106-4132-9913-955b44966fdd/event/list/
  - mcp2k recovered after: https://www.phy.bnl.gov/twister/bee/set/b6219330-3572-486f-b6cb-82f1899c5009/event/list/
  - owner-scan (12 evts) before: https://www.phy.bnl.gov/twister/bee/set/357e34e0-42eb-4e7e-9916-6b44ee5b2833/event/list/
  - owner-scan after: https://www.phy.bnl.gov/twister/bee/set/2e912475-9ad4-4523-b176-71ec7d876e59/event/list/
- Commits: toolkit `apply-pointcloud` (code + cfg flip + doctest pins),
  wcp-porting-img `main` (runner env mappings, census/mover TSVs, this
  doc, Bee index) — hashes recorded in the index/commit log.

## 11. Round 4 — projective duplicates at the shower stem (owner report 2026-08-17)

### 11.0 Repro

```
# census (metric calibrated in 11.2; vertices via pr_scores_table.py):
python3 scripts/pr83r4_projdup_census.py work-pr83r3-gateon2-nue \
    work-pr83r3-gateon2-ncpi0 work-pr83r3-gateon2-mc work-pr83r3-m2kon \
    --tsv docs/pr/83_r4-census-off.tsv
# knob-on probe (debug logs):
SBND_MVGA_PROJ_DUP_FRAC=0.7 PR_JOBS=3 bash run_pr_chain_batch.sh \
    work-nuecc48-vfcbr3on work-pr83r4-probe3 data 138009 168596 74544
```

### 11.1 Symptom (owner report)

A clear 1-track-1-shower topology (exemplar 138009) ends up with TWO
tracks inside the EM shower connecting to the neutrino vertex in the
final particle flow.  They overlap in the projective (wire) views and
differ inside the plane of ambiguity; one has very low dQ/dx.  This
threatens the e/gamma separation (stem dQ/dx is the discriminant), and
on 74544 one of the split tracks was identified as a proton instead of
an EM shower.  Also reported: 168596.

### 11.2 Root cause + why rounds 1-3 cannot see it

The track fitter split a single projective charge corridor across two 3D
trajectory interpretations.  The 2D measurements cannot distinguish
them; the fit starves one interpretation of charge.  Measured on the
round-3 knob-on arms (`pr83r4_projdup_census.py` numbers):

| evt | pair | 3D overlap @1.4cm | per-view overlap | stem dQ/dx (8cm) | ratio |
|---|---|---|---|---|---|
| 138009 | 12094/12095 | 0.58 | 1.00/0.82/0.61 | 8613 / 2384 | 0.28 |
| 168596 | 14168/14172 | 0.14 | 0.98/0.96/0.16 | 5416 / 439 | 0.08 |
| 74544  | 12105/12107 | 0.46 | 1.00/1.00/0.91 | 1010 / 3935 | 0.26 |

Every round-1-3 metric is a 3D corridor overlap (op1/op1-post/orphan
gates at 0.7-0.8): these pairs read 0.14-0.58 and never fire.  The
divergence lives in ONE view (168596: W and V track within <=1 cm over
30 cm while U walks out to 6 cm) — the classic wire-plane ambiguity.
Discrimination against a genuine collinear two-prong vertex: the
projective ghost is charge-starved (ratios 0.08-0.28), a real prong
carries MIP-level charge on both members (every non-target vertex pair
in the three events fails BOTH the angle gate, 70-90 deg, and the view
overlap, <=0.16).

### 11.3 Fix — `mvga_op1-proj` (knobs `mvga_proj_dup_frac`, `mvga_proj_dqdx_ratio`)

New pass in `main_vertex_graph_audit` (NeutrinoGraphAudit.cxx), after
op1-post, in its own fixed-point merge->refit loop (kEditCap-bounded).
Candidates are ONLY segment pairs incident on the main vertex (the
shower-stem beginning; the 390842-class near-vertex riders are not
incident and cannot enter).  Gates, in order: op1's chord-angle gate
(`mvga_dup_angle`, 20 deg) -> same (apa,face) -> per-view 2D overlap in
the three wire views (coord `(x, cos(a)z - sin(a)y)`, angles from
`grouping->wire_angles`, tol = `mvga_dup_tol` 1.4 cm), 2nd-best view
>= `mvga_proj_dup_frac` -> stem dQ/dx ratio (first 8 cm of fitted path
from the main vertex) < `mvga_proj_dqdx_ratio` (default 0.4).  Merge
recipe is op1's own: keep the higher-integrated-charge member,
pre-verified reconnects, cleanup, refit.  `mvga_proj_dup_frac = 0`
(C++ default) skips the pass entirely => byte-identical; the ratio knob
is inert while the pass is off.

### 11.4 Tuning: the ratio gate at mvga time

The C++ gate acts at mvga time, where the starvation is milder than in
the final geometry: 138009/74544 read ratio 0.47/0.48 there (final
0.28/0.26) and declined at the 0.4 default; 168596 read 0.33 and fired.
SBND ships `mvga_proj_dqdx_ratio = 0.55` -- margin over the measured
0.47/0.48, far below MIP-parity two-prongs, and safe because the
geometry gates alone admit ZERO false pairs: with the ratio gate fully
open the census finds only the 6 target pairs across all 559
PR'd-with-vertex events of the four arms.

### 11.5 Validation (all gates PASS)

- Knob-off byte-gate (new binary, knobs off, vs the r3 production
  state): **1024/1024** (`work-pr83r4-gateoff-{mc,nue,ncpi0}` vs
  `work-pr83r3-gateon2-*`, 890+96+38) + mcp2k spot 12/12
  (`work-pr83r4-m2kspot` vs `work-pr83r3-m2kon`).
- Knob-on arms (frac=0.7 ratio=0.55, 32 jobs): `work-pr83r4-gateon-*`
  (445+48+19, 0 failed), `work-pr83r4-m2kon` (2000, 0 failed).
- Projective census: **4 -> 0** (511: 138009/168596/74544 + 278046) and
  **2 -> 0** (mcp2k: 169658/284206), zero new
  (`docs/pr/83_r4-census-{on511,m2kon}.tsv`).
- Round-3 census on every knob-on arm: **stays 0** (no regression).
- pr/86 orphan census: **byte-identical** (0/1/2/3-orphan histogram
  471/24/11/3; VIA 24 / CUT 51 / BENIGN 6 both arms).
- Scores A/B: 511 sample = **exactly the 4 census events move, zero
  collateral** (nue 3, mc 1, ncpi0 0; `docs/pr/83_r4-movers-*.tsv`).
  mcp2k: **4 movers / 2000** -- the 2 census targets plus 71642 and
  178931, both attributed by debug rerun (`work-pr83r4-diag2`, hash gate
  4/4 vs the arm) to genuine projective riders (9.1 / 7.8 cm at view
  overlap 1.00/1.00, ratio 0.52) whose Bee endpoints sit just outside
  the census's 1.5 cm vertex net; score moves are small (Enu +-40 MeV).
- PID outcome: 168596's split-track **proton (180 MeV) is gone**, shower
  consolidates to one 1929 MeV electron (T_kine); 138009's two stem
  electrons (564+1133) -> one 1172 MeV; totals via the movers table.
- `wcdoctest-clus` 2118/2118.  Owner Bee scan of the 4-event 511 set:
  APPROVED (2026-08-18).

### 11.6 Records

- Flip: `mvga_proj_dup_frac = 0.7`, `mvga_proj_dqdx_ratio = 0.55` in
  `wct-pr-perevt.jsonnet`; compiled-config diff = exactly the two keys;
  bare-config smoke (138009 168596 74544 / 278046 / 169658 284206)
  hash gate **12/12** vs the knob-on arms -- bare run == production.
- Bee sets + owner-scan status: `docs/pr/83_bee-r4.index.txt`.
- Census/mover TSVs: `docs/pr/83_r4-census-{on511,m2kon}.tsv`,
  `docs/pr/83_r4-movers-{nue,mc,ncpi0,m2k}.tsv`.

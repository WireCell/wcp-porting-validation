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
  <https://www.phy.bnl.gov/twister/bee/set/0bb602a3-6c51-4792-ae8a-d0740730a631/event/list/>,
  after
  <https://www.phy.bnl.gov/twister/bee/set/9aff3fea-24ac-4815-a6d5-ccb35931e6a9/event/list/>
  (idx 0–7 = 138009/168596/268784/74544/349945/64409/174224/281837).
- Class B exemplars (production only):
  <https://www.phy.bnl.gov/twister/bee/set/8f3f427f-ee93-4c9a-8fb8-a1c02a148919/event/list/>
  (idx 0–2 = 350935/404684/246579).

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
- No toolkit commit this round (no code changed). `wcp-porting-img`: this
  doc section + the four files above.

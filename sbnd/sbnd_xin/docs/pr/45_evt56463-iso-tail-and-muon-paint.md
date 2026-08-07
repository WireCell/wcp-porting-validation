# doc pr/45 — evt 18255-56463 follow-ups: isochronous tail never fitted (empty-2D-tree sentinel) + 411 cm muon painted as shower

STATUS: SHIPPED.  Knobs `other_seg_empty_2d_guard` +
`pseudo_shower_track_paint`, C++ defaults FALSE, **both SBND PRODUCTION ON**
(wct-pr-perevt.jsonnet TLAs, flipped same round per the owner's standing
"flip if clean" policy — §7).

Owner request (2026-08-07, after the pr/43 round 2 ship): two residual issues
on 18255-56463 —
1. "this track (x,y,z)=(152.4,183.4,230.3) cluster 14006 does not cover the
   entire image near the end … the end of the track is an ISOchronous case."
2. "(x,y,z)=(-102.5,-49.7,391.4), cluster 14007 shown as red in
   shower_track-global though this is a muon in the final particle flow …
   I guess … this shower flag is still not changed."

Both issues are PRE-EXISTING: every Bee geometry layer of 56463 is
byte-identical between the pre-round baseline (`work-pr43r2-head-mcp1k`) and
the shipped round (`work-pr43r2-onall`) — the pr/43 round-2 knobs changed one
mc.json PID string on this event and nothing else.

## Repro

```bash
cd sbnd_xin
# baseline (production knobs, pr/45 knobs off = C++ default)
./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr45-<label> data 56463
# issue-1 fix on
SBND_OTHER_SEG_EMPTY_2D_GUARD=1 ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out> data 56463
# issue-2 fix on
SBND_PSEUDO_SHOWER_TRACK_PAINT=1 ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out> data 56463
```

Layer forensics: `mabc-pr.zip` → `data/0/0-<layer>-global.json`;
`shower_track` carries the flag in `q` (15000 = shower/red, 0 = track) and
`real_cluster_id = cluster*1000 + seg` (collapsed to the shower's start
segment for shower members).  Gate arms in §6.

## 1. The event, precisely

PR graph of image cluster 14 (main cluster): segs 5 + 7 form the 411.3 cm
cathode-crossing muon (x −121.5 → +130.8 cm, PF node "mu- 903 MeV", both segs
inside ONE long-muon pseudo-Shower whose cached `particle_type` = 13); seg 6
(display id 14006) is the 60.16 cm vertex track, fit ending at
(153.09, 185.13, 234.92) — the owner's quoted end.

- Issue 1 numbers: seg 6's associated cloud extends ~30 cm beyond the fit
  end — z 236.9 → 266.8 with x spread only 1.9 cm (mean x 153.4): an
  isochronous stretch ~3° from drift-perpendicular.  690 of 1322 associated
  points are >5 cm from the fit (max 35.1 cm); 946 raw image points sit out
  there (bee bundle `img-global`).
- Issue 2 numbers: the red points at (−102.5,−49.7,391.4) belong to seg 7 but
  wear id 14005 — the shower-collapse relabel.  All 3454 points of the muon
  pseudo-shower carry q=15000 (red) while the PF tree shows "mu- 903 MeV".
  Seg 14007 has ZERO points of its own in shower_track-global — that id is
  only visible in track_fit-global (owner read it there).

## 2. Issue 1 — Symptom / Root cause / Why it hid / Fix / Verification

**Symptom.** The fitted track (and hence the PF particle, the kine energy and
every display) stops at the onset of the isochronous stretch; the last ~30 cm
of image is inside the segment's *associated* cloud but outside its *fitted*
trajectory.

**Root cause — toolkit-only, NOT the isochronous geometry itself.**
The residual-coverage stage `find_other_segments`
(clus/src/NeutrinoOtherSegments.cxx) tags a steiner point "explained" when
all three per-plane 2D distances are < `scaling_2d*search_range` = 1.2 cm (or
dead-channel-covered).  The per-segment 2D distances come from
`DynamicPointCloud::get_closest_2d_point_info`, which returns the sentinel
**−1.0 when the per-(plane,face,apa) 2D kd-tree is empty**
(DynamicPointCloud.cxx, empty-knn branch) — i.e. whenever a segment has zero
fit points on the queried face.  −1 < 1.2 cm, so ONE segment living entirely
in the far TPC "covers" all three planes of every query on the near face.
Cluster 14 has exactly that: seg 7 is entirely at x<0.  Measured with a
temporary per-point diagnostic: **all 194 tail-box steiner points tagged,
u = v = w = −1.0, 3D distance 3.9–28.9 cm** — no residual component can ever
form on the whole face.

**Why it hid.** The prototype cannot express this state: uBooNE is
single-face, so `ProtoSegment::get_closest_2d_dis` always has points and the
port of the tagging logic (faithful otherwise, prototype
NeutrinoID_proto_vertex.h:797-1300, same thresholds `search_range=1.5cm`,
`scaling_2d=0.8`) silently acquired a new failure mode on multi-TPC
detectors.  It needs a cathode-crossing (or multi-APA) cluster AND a genuine
residual on one face — and the failure is invisible: fewer segments, nothing
crashes.

**Fix — knob `other_seg_empty_2d_guard` (C++ default false).**  In
`find_other_segments`' three 2D-comparison sites (tagging loop, component
`number_not_faked` census, re-evaluation loop), a negative 2D distance is
treated as "no projection information" (1e9) instead of "distance zero".
No thresholds changed; the −1 sentinel simply stops counting as coverage.
NOT a prototype-parity fix — a toolkit-only bug guard (porting_dictionary
entry added).

**Verification (knob on, evt 56463, arm `work-pr45-onA`).**
- Tagging: tail box 194 tagged → **1 tagged**; a 32-point component forms,
  A=(153.0,188.0,237.7) B=(153.9,199.9,266.8), `number_not_faked=27`,
  length 31.5 cm — passes the quality cuts; `do_single_tracking` returns a
  healthy 56-point fit (no single-fit collapse).
- Final output: new segment reaches **(153.7, 199.8, 266.7)** — the image
  end.  PF tree: `14006 pi+ 174 MeV` gains child `14007 mu- 119 MeV` (the
  tail's ~30 cm now contributes energy that was previously missing
  entirely).  nusel row for the event unchanged.
- Structural side effects on this event, reported as-is: the guard also
  un-hides a second small residual near (45,69,281) on the muon body, so
  seg 5 splits into two segments (PF muon node still "mu- 903 MeV"); the two
  gamma chains re-hang under the pi+ branch (graph re-parenting).  Display
  ids renumber (graph indices).

## 3. Issue 2 — Symptom / Root cause / Why it hid / Fix / Verification

**Symptom.** The whole 411 cm muon renders red (shower) in
shower_track-global; the PF tree simultaneously calls the same object
"mu- 903 MeV".

**Root cause.** Not a stale segment flag — segs 5/7 carry NO shower flags
(`print_segs_info` shows `Track`; the owner's hypothesized flag-inconsistency
class is real but is not what fires here).  The Bee writer
(MultiAlgBlobClustering.cxx, shower_track mode) decides paint by shower
MEMBERSHIP first: `is_shower = (member) || flags || pdg==11` → q=15000.
The PF tree (`make_shower_leaf`) reads `Shower::get_particle_type()` — the
cached 13.  Two different fields; every muon-typed long-muon pseudo-shower
(seeded from `segments_in_long_muon`, NeutrinoShowerClustering.cxx:113-133)
is painted inconsistently with its own PF verdict.  pr/44's
`shower_long_muon_keep_type` does not help (membership still wins), and the
pr/43-round-2 `pid_flag_reconcile` dissolve deliberately exempts cached-±13
multi-segment showers.

**Why it hid.** The membership-first rule is correct for real EM showers
(absorbed segments never get flags/pdg updated — that is by design); the
muon-typed pseudo-shower is the one class where membership and PF verdict
diverge.  Bee only shows the color, not the reason.

**Fix — knob `pseudo_shower_track_paint` (C++ default false).**  In the Bee
shower_track writer AND the parallel `PrDisplayDump::dump_track_shower`
(same decision code by design): when a segment's shower has cached
`|particle_type| == 13`, paint q=0 (track), overriding all disjuncts — the
PF verdict is the same cached field mc.json displays.  Scope is exactly the
long-muon pseudo-shower class; the id collapse to the start segment is kept
(one PF particle, one id — matches the "mu- 903 MeV" node).  Display-only by
construction: the code touches only the local `charge`/`is_shower` used for
the layer arrays.

**Verification (knob on, evt 56463, arm `work-pr45-onB`).**  The 3454-point
muon cloud flips q 15000 → 0; the 5-point cluster-111 fragment (a separate,
genuinely EM-typed shower, id 111022) correctly stays red; mc.json roots
byte-identical to baseline (`mu- 903 / gamma 33 / pi+ 177`); nusel unchanged.
Combined arm `work-pr45-onAB`: both fixes compose (tail covered AND muon
painted blue).

## 4. Knob threading

- C++: `NeutrinoPatternBase.h` `m_other_seg_empty_2d_guard{false}` (docstring
  with the measurement), forwarded via TaggerCheckNeutrino
  (configure/echo/pattern_algos ×3 sites); `MultiAlgBlobClustering.h`
  BeePointsConfig `pseudo_shower_track_paint{false}` + `PrDisplayDump.h`
  `m_pseudo_shower_track_paint{false}`.
- jsonnet: `common/clus.jsonnet` tagger signature + key-suppression;
  `sbnd/clus.jsonnet` 4 tagger sites + shower_track layer entry + pr_display
  data (both key-suppressed); `wct-pr-perevt.jsonnet` TLAs.
- Runner: tri-states `SBND_OTHER_SEG_EMPTY_2D_GUARD`,
  `SBND_PSEUDO_SHOWER_TRACK_PAINT` (unset = cfg default, 1/0 = force).
- Doctest pin: `other_seg_empty_2d_guard` false in
  doctest_clus_knob_defaults.cxx (the paint knob is a per-layer MABC config
  key, not a tagger key — no pin site).

## 5. Compiled-config + unit tests

- G5: bare compile byte-identical to clean HEAD (0 diff lines); knob-on TLAs
  add exactly 2 keys, in the right components (guard next to
  `oov_prototype_parity` in TaggerCheckNeutrino data; paint next to
  `particle_ids` in the shower_track layer).  Artifacts
  /home/xqian/tmp/pr45/cfg_{off,cleanhead,on}.json.
- G4: `./build/clus/wcdoctest-clus` 1061/1061 (twice: with and without the
  temporary diagnostic).

## 6. Gates and censuses

Arms (all on the final binary, ql_roots `work-nuecc48-cb0805` /
`work-ncpi0-cb0805` / `work-mcp1k-cb0805`; mcp1k subset = every 5th event,
200 events, list /home/xqian/tmp/pr45/mcp1k_sub.txt):

| arm | knobs | purpose |
|---|---|---|
| work-pr45-off48    | none (bare = production)         | G1 vs work-pr43r2-onall-48 |
| work-pr45-onA48    | guard on                          | G3 census |
| work-pr45-onB48    | paint on                          | G3 census |
| work-pr45-off19n   | none                              | G1 vs work-pr43r2-onall19n |
| work-pr45-m200on   | none                              | mcp1k baseline |
| work-pr45-m200onA  | guard on                          | G3 census (population) |
| work-pr45-m200off  | pr/43-r2 knobs forced 0           | pr/43-r2 mcp1k closure (§8) |

RESULTS:

- **G1 nueCC48: PASS 48/48 events, 96/96 archives** byte-identical
  (work-pr45-off48 vs work-pr43r2-onall-48, hash_archive.py member-content
  fields 1-2).  **G1 ncpi0-19: PASS 19/19, 38/38** (work-pr45-off19n vs
  work-pr43r2-onall19n).
- **G3 guard**: nueCC48 **0/48 moved — archive-level null** (every mabc
  member hash identical, not just mc.json).  mcp1k-200 **3/200 moved,
  nusel 0-diff, all three attributed**: each mover's cluster is
  cathode-crossing (the only topology where the −1 poisoning is active)
  with new/re-formed segments —
  - 276836 (cl 11, x −90.8→+169.7, 3→4 segs): mu- 529 → mu- 480 + pi+ 10,
    proton 17 dropped, proton 289→290 (recovered residual splits the arm).
  - 404684 (cl 9, x −10.2→+12.6, 2→3 segs): mu- 202 → proton 147 → proton
    164/334 chain (recovered residual restructures the vertex).
  - 407280 (cl 16, x −99.9→+11.5, 6 segs re-formed): mu- 837→832, proton
    111→109, e- 6 re-parents (graph re-formed, kine ±5 MeV).
- **G3 paint**: **0 PF movers anywhere** (mc.json/tracking/nusel untouched);
  shower_track q flips on **2/48 nueCC48 events** (137238, 400474 — the
  events with muon-typed pseudo-showers) and on 56463.  nusel 0-diff.
  (No mcp1k paint arm: the knob touches only the local `is_shower`/`charge`
  used for the layer arrays — the nueCC48 archive comparison plus the
  56463/137238 flip-verifies bound it.)

## 7. Flip decision

**BOTH FLIPPED SBND ON** (wct-pr-perevt.jsonnet TLAs `= true`) — the
standing policy is satisfied: zero nusel verdict flips on every arm and every
mover attributed (§6).  Flip-verify (bare post-flip run == gated-on arm,
full-archive member hash):

- 56463 vs work-pr45-onAB: MATCH `e6b4cc55a9fa5b8354be2cbe8586c0d2`
- 137238 vs work-pr45-onB48: MATCH `c5808fbe22a4d4371bdbe724a4149dd1`

Owner veto: `other_seg_empty_2d_guard=false` restores the pre-fix face
poisoning (and the three mcp1k movers' old shapes);
`pseudo_shower_track_paint=false` restores the red muon paint.

## 8. pr/43 round 2 mcp1k closure

The round-2 doc's "Scope and what is NOT claimed" left the mcp1k population
un-censused.  `work-pr45-m200off` (three round-2 knobs forced 0) vs
`work-pr45-m200on` (bare = production) closes it: **3/200 moved, nusel
0-diff, all three the designed muon→pion relabel, per-knob attributed by
single-knob-off isolation arms** (work-pr45-iso{K1,K2,K3}-<evt>):

| evt | production | knobs-off | sole knob |
|---|---|---|---|
| 292005 | `18025 pi+ 2 MeV` | `mu- 2 MeV` | K2 single_muon_long_muon_claim |
| 395610 | `28005 pi+ 189 MeV` | `mu- 177 MeV` | K2 single_muon_long_muon_claim |
| 66118  | `9003 pi+ 205 MeV` | `mu- 193 MeV` | K1 single_muon_proton_chain_veto |

No unattributed movement, no verdict flips — the round-2 flip stands on the
population the cases came from.

(The earlier attempt `work-pr43r2-m200off` is a PARTIAL INVALID arm — its
event list was empty so only 24 events launched, and the batch died at a
session interrupt with 6 events killed mid-write.  Left in place, not
reused.)

## 9. Bee evidence

`bee/pr45/pr45-{before,after}.zip` (idx 0 = 56463): before =
`work-pr43r2-onall` (pre-pr/45 production), after = `work-pr45-flipver`
(post-flip production).  Expected differences: the track now reaches
z≈266.7 (track_fit) and the 411 cm muon renders blue in shower_track.
Upload (owner-run): `./upload-to-bee.sh bee/pr45/pr45-before.zip` then
`.../pr45-after.zip`.

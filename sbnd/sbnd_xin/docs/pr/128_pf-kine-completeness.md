# doc pr/128 — PF/kine completeness: reconstructed objects that reach no output

**Status: SHIPPED — 4 knobs SBND PRODUCTION ON 2026-08-29** (toolkit `1ed308de`, `e6fb0ebd`, `b0e91449` + the cfg flip).

Owner, 2026-08-29, opening this round:

> "Let's start the 1+2 as one PF/kine-completeness round"

and setting its success metric:

> "the key here is that we do not want to lose energies (or double count
> energies) for the neutrino candidate. Note, we do not want to count energy
> for overclustering activitys, far away), but if we are losing energies,
> that would not be good."

So this round is scored on **candidate energy accounting**, not on how the
Bee tree looks. Three outcomes are distinguished throughout:

| outcome | verdict |
|---|---|
| candidate charge reconstructed but absent from `kine_reco_Enu` | **lost — fix** |
| the same charge counted twice (once in a shower, once as a track) | **double count — must not happen** |
| far-away / over-clustered activity added to the candidate | **must not be counted** |

Follow-on to [doc pr/127](127_sccc-regression-137238.md), which fixed one
instance of the first class by hand (SBND 18255-137238) and whose §5.2
blind-spot census is the measurement this round generalises.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# §1 the two structural blind spots, over both production manifests
./scripts/pr127_blindspot_census.py 'work-pr125r1-flipS98-*' 'work-pr125r1-flipS141-*' \
    --tsv docs/pr/pr127-blindspot-census.tsv

# §2 this round's characterisation (PID score, conn-4 producers, distances)
./scripts/pr128_completeness_census.py 'work-pr125r1-flipS98-*' 'work-pr125r1-flipS141-*' \
    --tsv docs/pr/pr128-completeness-census.tsv
```

Baseline for every number below: `work-pr125r1-flipS{98,141}-*`, i.e. today's
SBND production point (pr/125 K1–K5 on, pr/127 `sccc_max_gap=10` on),
239 unique events over the 98-set and 141-set manifests.

## 1. What is missing, and how much of it is energy

`fill_bee_pf_tree` (`MultiAlgBlobClustering.cxx`) and `fill_kine_tree`
(`NeutrinoKinematics.cxx`) each drop reconstructed objects through two
structural gates. Counting what falls through:

**A — cross-cluster unclaimed track segments.** Every PF orphan pool *and*
the `pr65 pf-orphan-audit` line is `same_cluster`-gated (`:1852`, `:2328`,
`:2398`; the kine twin `kine_count_orphan_tracks` at
`NeutrinoKinematics.cxx:552`). A track segment in another cluster that no
shower and no BFS claimed therefore reaches no output and is not even
counted in the audit.

- **29 objects in 18 of 239 events**; 21 of them within 10 cm of something
  the PF tree does show; 19 of those are ≥10 cm long, 8 are ≥50 cm, and many
  sit at gap **0.00 cm** — touching the displayed candidate.
- Worst rows: 72786 (µ 143.5 / µ 114.3 / µ 108.3 / π 64.7 cm), 55740
  (µ 123.1 cm, gap 0.00), 399118 (p 108.8 cm, gap 0.00), 393505 (µ 131.4 /
  63.4 cm), 318769 (µ 39.8 + p 21.1 cm, both gap 0.00).
- **Control**: the same test *inside* the main cluster, where the orphan
  machinery does apply, finds **1 object** in 1 event. The loss is the
  cross-cluster gate specifically, not a general weakness of the pools.

**B — conn-4 showers.** A shower whose `start_connection_type` is 4 is
skipped by PF (`:1636`, plus `conn4_skip_segs` at `:1345-1351` which also
removes its segments from the BFS, the child recursion, the root loop and
all three orphan pools) and by kine (`vtx_type > 3` continue,
`NeutrinoKinematics.cxx:481`).

- **514 showers in 179 of 239 events, 3514 MeV.**

## 2. Sorting B against the owner's rule: far-away activity must NOT be counted

conn-4 is assigned in exactly three places, and reading them decides which
part of that 3514 MeV is loss and which is the cut working as designed:

| producer | rule | meaning |
|---|---|---|
| `shower_clustering_in_other_clusters` (`NeutrinoShowerClustering.cxx:3733`) | `min_dis > 80 cm` ⇒ 4 | a whole *other* cluster ≥80 cm from the candidate |
| `shower_conn3_unreachable` (pr/74, `:3858`) | `min_dis > 80 cm` ⇒ 4 | a **main-cluster** segment unreachable from the main vertex |
| `pass4_prune` / `pass4_prune2` re-seed (pr/123 `:6435`, pr/124 `:6645`) | `root_sv_dis > 80 cm` ⇒ 4 | a component **shed from a shower** and re-seeded |

(`flag_save` is `true` at the only call site, so that second `= 4` branch at
`:3736` is dead.)

Measuring the distance from each conn-4 shower's own cluster to the main
cluster's point cloud:

| gap to main cluster | showers | MeV | of which >20 MeV |
|---|---|---|---|
| <5 cm | 1 | 44 | 44 |
| 5–20 | 1 | 26 | 26 |
| 20–50 | 19 | 218 | 170 |
| 50–80 | 43 | 581 | 465 |
| 80–150 | 93 | 632 | 362 |
| 150+ | 354 | 1602 | 993 |

**So the bulk of the 3514 MeV is exactly what the owner says must not be
counted**: **490 of 514 showers — 2815 of 3514 MeV — sit ≥50 cm from the
candidate**, 354 of them beyond 150 cm. Those are cosmics and over-clustered
activity, and the 80 cm cut is doing its job. This round does not touch them.

What is left after that subtraction is small and sharp:

**B1 — conn-4 material that is in the MAIN cluster: 3 showers, 2 events,
411 MeV.**

| event | shower | E | length | pdg | producer (from the run log) |
|---|---|---|---|---|---|
| 105074 | 23005 | 215.1 MeV | 82.9 cm | 13 | `pr74 conn3_unreachable: promote gidx=5 conn=4 anchor_dis 119.5cm` |
| 105074 | 23004 | 162.0 MeV | 58.2 cm | 13 | `pr74 conn3_unreachable: promote gidx=4 conn=4 anchor_dis 119.0cm` |
| 179048 | 17003 | 34.2 MeV | 9.9 cm | 11 | `pr123 pass4_prune: shower_id=8 sheds 2 detached seg(s), conn=4` |

This is a self-inflicted class, and it is the clearest "lost energy" in the
census. `shower_conn3_unreachable` exists *to rescue* main-cluster material
that the main-vertex BFS cannot reach (doc pr/74); it wraps that material in
a shower and then, when the anchor is >80 cm away, labels it conn-4 — which
is the label that means "show nowhere and count nowhere". The rescue
cancels itself. Same for a pruned component re-seeded at conn-4: it was
*removed* from its parent shower (whose kinematics are then recomputed), so
its charge is counted in no object at all.

Note what B1 is not: 105074's two objects are **pdg 13**, 58 and 83 cm, in
the neutrino candidate's own cluster. That is track material, not far-away
over-clustering.

**B2 — cross-cluster conn-4 within 20 cm of the main cluster: 2 showers.**
318769 (43.9 MeV at 3.6 cm, `pr123 pass4_prune`) and 396222 (26.3 MeV at
14.8 cm, `pr123 pass4_prune` + `pr124 pass4_prune2`). Neither comes from the
80 cm distance cut — the re-seed rule measures the shed component against
*its parent shower's start vertex*, which can be far even when the material
touches the candidate. Same mechanism as B1, so the same fix reaches them if
the predicate is written on **proximity to the candidate** rather than on
cluster identity.

Whole B target list, from the census (`Q3`), producers read from each arm's
own run log:

| event | shower | pdg | E | length | gap to main cluster | producer |
|---|---|---|---|---|---|---|
| 105074 | 23005 | 13 | 215.1 MeV | 82.9 cm | 0.0 (main) | `pr74_conn3_unreachable` |
| 105074 | 23004 | 13 | 162.0 MeV | 58.2 cm | 0.0 (main) | `pr74_conn3_unreachable` |
| 318769 | 113062 | 11 | 43.9 MeV | 5.9 cm | 3.6 cm | `pr123_pass4_prune` |
| 179048 | 17003 | 11 | 34.2 MeV | 9.9 cm | 0.0 (main) | `pr123_pass4_prune` |
| 396222 | 90166 | 13 | 26.3 MeV | 8.4 cm | 14.8 cm | `pr123_pass4_prune`, `pr124_pass4_prune2` |

**481.5 MeV in 4 events** — every megaelectronvolt of it produced by a knob
of ours that then labelled its own output invisible.

## 3. Why the existing predicate cannot be reused (measured)

The obvious implementation of A is to widen the existing orphan pools, whose
admission test is `segment_orphan_confident_track` = confident non-electron
PID **and** length > `min_len` **and** straight-long. Its first term is
`segment_confident_nonelectron_pid`, which ends `return seg->particle_score()
< 1.0`.

Pulling `particle_score` for the 19 near-and-long objects of class A:

| particle_score | count |
|---|---|
| **100.000** | **10** |
| 0.06–1.00 | 9 |

**10 of 19 would be rejected on the score term alone**, every one of them at
score exactly 100 — including 55740's
123.1 cm muon, 72786's 114.3 and 108.3 cm muons, and 393505's 63.4 cm muon,
all at gap 0.00 cm. Score 100 is the trajectory branch's unconditional
sentinel stamp (the same defect that mislabelled 137238's muon stem in doc
pr/127, and the pr/124 §B.1 finding). `pf_orphan_guard_freed` already says
so in its own comment at `MultiAlgBlobClustering.cxx:2444`: *"cross-cluster
and score-100-sentinel PID, so both the pr/93 confident-track class and the
main-cluster audit scope miss it"* — which is why pr/123 used a **flag**
predicate instead.

So this round needs its own predicate: non-electron pdg, length floor,
straight-long, **no score term**. Written as a new function
(fork-by-duplication, M10 / §2 Code) — `segment_confident_nonelectron_pid`
has live consumers and must stay byte-for-byte.

## 3.1 Proximity alone is NOT a usable predicate — the round's main finding

The first implementation admitted a cross-cluster track on *proximity to the
emitted candidate* alone (gap ≤ 5 cm, length > 30 cm, non-electron,
straight-long). Smoke-tested on the two class representatives:

| event | Enu before | Enu after | Δ | verdict |
|---|---|---|---|---|
| 105074 (class B) | 1188.7 | 1565.8 | **+377.1** | clean — exactly the two kept showers (162.0 + 215.1), `add_energy` unchanged at 105.7, no other particle touched |
| 72786 (class A) | 701.6 | **1852.9** | **+1151.3** | **WRONG — this is the failure mode the owner named** |

72786 gained a 143.5 cm muon (344.0 MeV), a 108.3 cm muon (268.9 MeV) and a
64.7 cm pion (187.5 MeV), plus 350.8 MeV of rest-mass/binding terms. Looking
at the event's cluster anatomy explains it:

| cluster | segments | total length | extent | closest approach to the ν vertex |
|---|---|---|---|---|
| **17 (main)** | 2 | **100.5 cm** | 90.5 cm | 0.0 cm |
| 9 | 6 | 342.7 cm | **148.4 cm** | **35.3 cm** |
| 45 | 2 | 112.8 cm | **101.6 cm** | **77.2 cm** |

The neutrino candidate is 100 cm of track. Clusters 9 and 45 are large
independent structures sitting 35 and 77 cm off the vertex — **cosmics**.
They read gap 0.00 cm only because they brush the *far end* of the
candidate's own 94 cm muon. Admitting them counts over-clustered activity as
neutrino energy and would have more than doubled Enu on this event.

**Fix: require a continuation, not a touch.** The discriminator is the one
doc pr/127's sccc fix already uses for exactly this question — a continuation
joins **end to end** and runs **straight on**. `segment_continuation_geometry`
(`PRSegmentFunctions.cxx`) returns, per candidate against its nearest
reference segment: `gap`, `cand_end_dis` (path distance from the touch point
to the candidate's own nearer end), `ref_end_dis` (the same on the reference),
and `angle_deg` (kink from collinear, 0 = perfectly straight continuation).
A cosmic crossing a displayed track touches its *middle*, so `ref_end_dis` is
large; a cosmic running alongside fails the kink. Knob defaults: end
tolerance 10 cm, kink 30°.

This is why class A ships only if the census below shows the continuation
terms separate the 137238 class from the 72786 class. Class B is unaffected —
its material is the main cluster's own, at gap 0.07 cm, and its recovery is
exact.

## 3.2 The census: does the continuation geometry separate the two classes?

`WCT_PFNEAR_DEBUG=1` prints one line per candidate **from the code path the
knob uses**, so the census and the knob cannot use different definitions
(the pr/127 `WCT_SCCC_DEBUG` pattern). The tape arms
`work-pr128r1-dbg{98,141}-*` also serve as the knob-off gate, which proves
the tape is byte-neutral: **478/478 archives byte-identical**, 0 missing
(98-set 196, 141-set 282).

Class A: 13 candidates in 8 events reach the geometry test. Sorted by kink —

| event | seg | pdg | len | KE | gap | cand_end | ref_end | **kink** | d(ν vtx) |
|---|---|---|---|---|---|---|---|---|---|
| 392901 | 127032 | 13 | 38.5 | 118.5 | 0.00 | 0.00 | 0.00 | **4.8** | 102.9 |
| 55740 | 22014 | 13 | 123.1 | 300.6 | 0.00 | 0.00 | 0.00 | **13.0** | 61.8 |
| 94392 | 45030 | 13 | 46.8 | 137.2 | 0.00 | 0.00 | 0.00 | **13.3** | 44.5 |
| 171572 | 10008 | 13 | 125.1 | 304.8 | 0.00 | 0.00 | 0.00 | 38.4 | 20.4 |
| 393505 | 15004 | 13 | 63.4 | 174.8 | 0.00 | 0.00 | 0.00 | 41.0 | 68.9 |
| 318769 | 19001 | 13 | 39.8 | 121.5 | 0.00 | 0.00 | 0.00 | 42.0 | 114.3 |
| **72786** | 9006 | 13 | 114.3 | 281.7 | 8.44 | **47.07** | 0.00 | 43.5 | 35.3 |
| 399118 | 16017 | 2212 | 108.8 | 481.0 | 0.00 | 0.00 | 0.00 | 47.3 | 4.9 |
| 393505 | 15005 | 13 | 131.4 | 318.4 | 28.74 | 0.00 | 0.00 | 58.8 | 77.1 |
| **72786** | 9008 | 211 | 64.7 | 187.5 | 0.00 | 0.00 | 0.00 | 89.5 | 38.3 |
| **72786** | 9004 | 13 | 143.5 | 344.0 | 2.37 | **52.20** | 0.00 | 93.7 | 46.6 |
| **72786** | 45038 | 13 | 108.3 | 268.9 | 0.00 | 0.00 | 0.00 | 108.3 | 78.4 |
| 393505 | 15013 | 13 | 108.5 | 268.7 | 0.00 | 0.00 | 0.00 | 137.3 | 74.4 |

Operating points:

| (end_tol, kink, gap) | candidates | events | Σ KE | worst single event |
|---|---|---|---|---|
| proximity only | 11 | 8 | 2707.5 | 72786 **+800.5 MeV** |
| 20 cm / 45° / 5 cm | 6 | 6 | 1157.3 | 171572 +304.8 |
| **10 cm / 30° / 5 cm (default)** | **3** | **3** | **556.3** | 55740 +300.6 |
| 5 cm / 20° / 5 cm | 3 | 3 | 556.3 | 55740 +300.6 |
| 5 cm / 10° / 5 cm | 1 | 1 | 118.5 | 392901 +118.5 |

(Footnote: this census ran on the pre-guard binary. 94392 is in the table at
kink 13.3° but is guarded out of the shipped pool — its segment is already
emitted by `pf_orphan_guard_freed` — so the **shipped class-A footprint is 2
events / 419.1 MeV**, not 3 / 556.3, and the effective margin runs from
55740's 13.0° to 393505's 41.0°.)

**The separation is real.** All four 72786 cosmics are rejected at the
default — two on `cand_end` (47.1 and 52.2 cm: the displayed track brushes
the candidate's *middle*, the signature of a crossing) and two on kink (89.5°
and 108.3°). What survives is 3 candidates in 3 events, every one of them
end-to-end at gap 0.00 with a kink under 14°: 392901 (118.5 MeV), 55740
(300.6 MeV), 94392 (137.2 MeV). There is a 25° margin between the last
accepted candidate (13.3°) and the first rejected one (38.4°), so the
operating point is not perched on an edge.

Two candidates rejected at the default deserve a note rather than a knob
change: **171572** (kink 38.4°) is already emitted by `pf_orphan_guard_freed`
— see the double-emission guard below — and **399118**'s 108.8 cm proton sits
4.9 cm from the ν vertex but kinks 47.3° away from what it touches, so it is
not a continuation of it.

**Double-emission guard (found here, not by a test).** `pf_orphan_guard_freed`
emits its nodes *without* inserting into `used_segs`, so its segments were
still visible to this pool — 171572's 125.1 cm muon is in the census at gap
0.00. At a laxer operating point the knob would have drawn it twice. The PF
pool now skips `kPass4GuardFreed` segments while that knob is on. The kine
side was already safe: `kine_count_guard_freed` inserts into `used_segments`
and runs first.

Class B at the 20 cm default: **6 showers in 4 events, 484.1 MeV.**

| event | node | pdg | KE | length | gap to main cluster |
|---|---|---|---|---|---|
| 105074 | 23005 | 13 | 215.1 | 82.9 cm | 0.07 cm |
| 105074 | 23004 | 13 | 162.0 | 58.2 cm | 0.08 cm |
| 318769 | 113062 | 11 | 44.0 | 5.9 cm | 2.59 cm |
| 179048 | 17003 | 11 | 34.2 | 9.9 cm | 0.04 cm |
| 396222 | 90166 | 13 | 26.3 | 8.4 cm | 13.74 cm |
| 396222 | 79155 | 11 | 2.5 | 0.7 cm | 10.13 cm |

The next shower out is at 20–50 cm; the 490 far ones (≥50 cm, 2814 MeV) stay
skipped, as the owner requires.

**Read that gap column carefully — for three of the six rows it does no
work.** `pf_conn4_near_gap` measures the shower's closest approach to the
*main cluster's* point cloud, so for a shower whose material **is** in the
main cluster the number is ~0 by construction. 105074 reads 0.07/0.08 cm
because it is in cluster 23, not because it is near anything. The operative
admission rule for that class is **main-cluster membership**, and the number
that lets the reader judge it is the producer's own:

| event | KE | admission actually turns on | producer's distance |
|---|---|---|---|
| 105074 ×2 | 215.1 + 162.0 | main-cluster membership (gap vacuous) | `anchor_dis` **119.5 / 119.0 cm** from the nearest main-vertex-reachable vertex |
| 179048 | 34.2 | main-cluster membership (gap vacuous) | shed component 213.2 cm from its parent shower's start vertex |
| 318769 | 44.0 | **gap 2.59 cm** (binding) | — |
| 396222 ×2 | 26.3 + 2.5 | **gap 13.7 / 10.1 cm** (binding) | — |

So of the 484 MeV class-B recovery, **411.3 MeV is main-cluster, where the
gap test is vacuous**, and 72.8 MeV is cross-cluster, where it binds. The
411 MeV is 34% of the whole round's recovery and its largest single-event
move, and the judgement it asks for is: *is main-cluster material 119 cm from
any reachable vertex part of the candidate's energy?* `shower_conn3_unreachable`
already answered "yes, it is the candidate's" when it built the shower; conn-4
then answered "no" for the outputs. This round makes the two answers agree.
That is the call to check, alongside 392901's 102.9 cm in class A.

## 4. Known-divergence check (M15)

The `same_cluster` filter in the orphan pools carries
`// prototype NeutrinoID.cxx:1488`, so relaxing it is a divergence from the
prototype and needs the M15 check before it is touched.
`clus/docs/porting/porting_dictionary.md` records the main-cluster filter's
*consequences* (the "non-main cluster, so it produces no `mc.json` delta"
note at the `shower_proton_daughter_pion_dissolve` entry) but does **not**
list it among the intentional-divergence entries. The owner has adjudicated
this exact class twice — 315167 (pr/93 r4, "missing from the final particle
flow???") and 137238 (pr/127) — and `pf_orphan_guard_freed` +
`kine_count_guard_freed` already ship **cross-cluster** with owner approval
(2026-08-28). M15 clears.

## 5. Knobs (all DEFAULT OFF)

Two pairs, deliberately split so the display can be approved without the
energy: the PF knob changes only the picture, the kine twin moves
`kine_reco_Enu`.

| knob | side | class | effect |
|---|---|---|---|
| `pf_orphan_near_cross_cluster` (+ `pf_orphan_near_gap_cm`, `pf_orphan_near_min_len_cm`) | PF | A | admit an unclaimed cross-cluster track that comes within *gap* of the emitted candidate |
| `kine_count_near_cross_cluster` (+ same params) | kine | A | count the same segments in `kine_reco_Enu` |
| `pf_main_cluster_conn4_visible` | PF | B1/B2 | a conn-4 shower whose material is the candidate's own is no longer skipped |
| `kine_count_main_cluster_conn4` | kine | B1/B2 | count those showers in `kine_reco_Enu` |

**Proximity reference set.** "The emitted candidate" is defined once, in C++,
as `used_segs \ conn4_skip_segs` — every segment the PF tree actually draws
(shower members of non-skipped showers, plus every BFS-claimed track). The
census script uses the same definition via the byte-neutral
`WCT_PFNEAR_DEBUG` tape emitted from the same code path, so "census predicted
M, knob fired N" cannot drift apart (the pr/127 `WCT_SCCC_DEBUG` pattern).

**Double-count argument, to be verified not assumed.** For class A the
admitted segment is by construction not a member of any shower and not
BFS-claimed, and it lives in a different cluster, hence on different blobs
than any counted shower's charge integration. For B1/B2 the re-seeded or
promoted component was *shed* from its parent, whose kinematics were
recomputed afterwards. Both arguments predict Δ`kine_reco_Enu` ≈ the added
object's own KE and no change to any host shower's energy; §7 checks that
per fired event rather than asserting it.

## 6. Implementation

| seat | file |
|---|---|
| predicate | `PRSegmentFunctions.{h,cxx}` — `segment_near_candidate_track` |
| PF class A | `MultiAlgBlobClustering.cxx`, new pool after the pr/123 guard-freed pool |
| PF class B | `MultiAlgBlobClustering.cxx` — `conn4_keep_showers`, consulted at the `conn4_skip_segs` build, the parentage loop and the shower skip |
| kine class A | `NeutrinoKinematics.cxx`, new pass after `kine_count_guard_freed` |
| kine class B | `NeutrinoKinematics.cxx` — `conn4_keep_showers`, consulted at the `vtx_type > 3` skip |
| config | `TaggerCheckNeutrino.{h,cxx}` (read + `default_configuration` echo + `pattern_algos` push), `NeutrinoPatternBase.h` |
| jsonnet | `sbnd/clus.jsonnet` (builder args + key-suppressed injection), `sbnd/wct-pr-perevt.jsonnet` (top-level args + pass-through + kine injection) |
| runner | `run_pr_chain_batch.sh` — `SBND_PF_ORPHAN_NEAR_*`, `SBND_PF_CONN4_NEAR*`, `SBND_KINE_NEAR_*`, `SBND_KINE_CONN4_NEAR*` |
| tests | `doctest_clus_knob_defaults.cxx` — both kine knobs pinned OFF |

Three details worth recording:

1. **The reference set is defined once, in C++.** "Touches the candidate"
   means: within *gap* of `used_segs \ conn4_skip_segs` on the PF side, and of
   the counted showers' segments ∪ `used_segments` on the kine side. Raw
   `used_segs` would have let a candidate qualify by touching an *invisible*
   conn-4 cosmic — the exact failure the round is about. The
   `WCT_PFNEAR_DEBUG` tape is emitted from that same code path, so the census
   and the knob cannot use different definitions.
2. **conn-4 keeps are decided with the producers' own metric.**
   `Facade::Cluster::get_closest_dis` against the main cluster — the same call
   `:3733`, `:3858`, `:6435` and `:6645` use to assign conn-4 in the first
   place, so "gap" means one thing across the round.
3. **Displaced objects render as `nu → n → track`**, the pr/123 convention the
   owner set on 2026-08-28 — a track not connected to the neutrino vertex
   must not hang at root.

Determinism: both new pools collect into a `std::vector` and sort by display
id (PF) or graph edge index (kine) before emitting; no pointer-keyed container
is iterated. `seg_display_id` was hoisted above its first use in
`fill_bee_pf_tree` (pure move of a capture-less lambda).

Build: `wcbuild` rc=0; freshness proof `libWireCellClus.so` 11:58:59 vs last
source edit 11:57:40; `./build/clus/wcdoctest-clus` **235/235 passed, 2524
assertions**.

## 7. Validation

**Knob off — byte-identical.** `pr85_hash_gate.py`, arms
`work-pr128r1-dbg{98,141}-*` (which also carry the `WCT_PFNEAR_DEBUG` tape, so
this simultaneously proves the tape is byte-neutral) vs the production arms
`work-pr125r1-flipS{98,141}-*`:

| manifest | sample | compared archives | result |
|---|---|---|---|
| 98 | mcp1k / mcp2k / ncpi0 / nuecc48 | 28 / 34 / 38 / 96 | PASS |
| 141 | mcp1k / mcp2k | 104 / 178 | PASS |

**478 / 478 archives byte-identical, 0 missing.** (Compared-archive counts are
quoted deliberately — a pr/127 gate printed PASS while comparing zero.)
Compiled config identical on 75/75 events with the arm path normalised, new
keys absent; `doctest_clus_knob_defaults` 235/235.

**Knob on — footprint.** `work-pr128r1-on{98,141}-*` vs production: of the same
478 archives, exactly **six differ**, in six events — 55740, 105074, 179048,
318769, 392901, 396222. `pctree-pr` identical everywhere.

**Knob on — energy accounting** (the owner's metric; note the archive gate
cannot see this, since `kine_reco_Enu` lives in the calib dump, not in an
archive). All 239 events compared:

| event | class | Enu off → on | Δ | added particles | removed |
|---|---|---|---|---|---|
| 55740 | A | 488.5 → 894.7 | **+406.2** | mu- 300.6 (+105.7 rest mass) | none |
| 105074 | B | 1188.7 → 1565.8 | **+377.1** | mu- 215.1, mu- 162.0 | none |
| 392901 | A | 2003.9 → 2228.1 | **+224.2** | mu- 118.5 (+105.7 rest mass) | none |
| 318769 | B | 782.7 → 826.6 | +43.9 | e- 43.9 | none |
| 179048 | B | 1221.7 → 1255.9 | +34.2 | e- 34.2 | none |
| 396222 | B | 3802.2 → 3831.0 | +28.8 | mu- 26.3, e- 2.5 | none |

**+1114.4 MeV over 6 of 239 events, and nothing is removed anywhere** — the
per-event multiset diff of the kine particle list is the double-count test,
and it is clean on all six. The `+105.7` on the two class-A events is the muon
rest-mass term every counted muon receives.

**Double count actually caught.** 94392's 46.8 cm muon passed the class-A
geometry (kink 13.3°) but is *not* in the footprint: it is already emitted by
`pf_orphan_guard_freed` and already counted by `kine_count_guard_freed`.
Without the §3.2 guard the knob would have drawn and counted it twice. This is
the one place the owner's "do not double count" instruction changed the code.

**Physics checks.**

- Vertex movers (`pr90_movers.py --tags vtx105`), all six sample arms:
  **0 movers > 0.05 cm, 0 ADVERSE**, 159 labelled vertices compared.
- `nusel-evt*.tsv`: **239/239 byte-identical**.
- Main vertex identical on all six moved events.
- Reported, not tuned: `numu_score` moves on four of the six —
  105074 −4.01 → −3.55, 179048 2.857 → 2.805, 318769 0.156 → 0.120, and
  **55740 0.259 → 1.099, a 4× rise and the largest score movement in the
  round**. The direction is the expected one (a 300 MeV muon joined the
  candidate), but it is a selection-relevant number and is reported, not
  tuned. **The byte-identical `nusel` table does not cover it**: if
  `numu_score` changed on four events while every `nusel-evt*.tsv` is
  identical, then that table does not carry `numu_score`, so it proves
  nothing about a numu cut. (This is the pr/127 §3.2 caveat, pointed at the
  field that actually moved; `nue_score` did not change on any event.)
- **EM label metric: provably inert, no probe arm needed.** Neither knob
  touches shower construction — `pf_conn4_near_candidate` only feeds
  `conn4_keep_showers`/`conn4_skip_segs`, `kine_count_conn4_near` only the
  `vtx_type > 3` skip, and class A adds segments. Checked rather than
  asserted: the `showers` array of the calib dump is **identical on 239/239
  events**, so `em117_score.py`'s inputs cannot have moved and the
  qF1/q_extra residual is untouched.
- **Sentinel suite** (`pr127_sentinels.py`) on the ON arms: **10 PASS, 0 FAIL,
  4 SKIP** — no previously shipped fix regressed.

**Flip + flip-equivalence.** Four knobs set in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`; compiled-config proof
post-flip shows all four keys true with no env. `work-pr128r1-flipchk-mcp2k`
(post-flip config, **no** env, 6-event subset) vs `work-pr128r1-on141-mcp2k`:
**PASS, 12/12 archives byte-identical** (12 compared, 83 unpaired = events
outside the subset).

**Owner package.** `bee/pr128r1/` — 7 events, the six fires plus 72786 as the
cosmic control. Built and annotated; **not uploaded** (outward-facing,
CLAUDE.md §5.6). **Content-verified before any link is reported**: for all
7 indices on both sides, the md5 of the package's `data/i/i-mc.json` equals
the md5 of the source arm's `pr_evt<ev>/mabc-pr.zip` `data/0/0-mc.json` —
14/14 match, including idx 5 (396222), the one row that resolves through the
98-set root rather than the 141-set root. Idx 6's mc layer is byte-identical
between the OFF and ON packages, so the package itself carries the
"72786 unchanged" control claim.

## 8. Sentinels

Every knob here carries a distance or angle threshold tuned to measured
geometry — exactly the exposure class doc pr/127 §5.1 built the registry for.
Three entries added to `scripts/pr127_sentinels.py` in the flip commit:

| event | asserts |
|---|---|
| 105074 | a `mu-` ≥ 180 MeV in PF **and** a `pr128 pf-conn4-near: KEEP` line |
| 55740 | a `mu-` ≥ 250 MeV in PF **and** a `pr128 pf-orphan-near-cross-cluster` line |
| **72786** | **`log_absent`** that line, **and** no `mu-` ≥ 250 MeV |

The third is the one that matters most: it is a *negative* sentinel guarding
the cosmic rejection. The event's legitimate maximum `mu-` is 238.8 MeV while
its four cosmic candidates are 268.9 / 281.7 / 344.0 MeV, so the 250 MeV line
sits between the two populations. Any future loosening of the continuation
terms that re-admits them fires it. A new assertion kind, `log_absent`, was
added to the registry for this.

## 9. Open / deferred

- The class-A operating point admits a continuation of a **displayed track**,
  not necessarily of the ν vertex: 392901's muon sits 102.9 cm from the vertex.
  That is the judgement call in this round and it is idx 2 of the Bee set.
- 399118's 108.8 cm proton sits 4.9 cm from the ν vertex but kinks 47.3° from
  what it touches, so it is rejected as a non-continuation. If the owner reads
  it as a real daughter, the predicate needs a vertex-proximity arm, not a
  looser kink (which would re-admit 72786's cosmics).
- The `same_cluster` gate still hides the **audit line** as well as the pools,
  so the general cross-cluster population remains uncounted by design. Only the
  continuation class is now visible.
- 318769 appears both here and in the pr/124 worst-`q_miss` rows; whether this
  fire explains part of that residual is not measured.

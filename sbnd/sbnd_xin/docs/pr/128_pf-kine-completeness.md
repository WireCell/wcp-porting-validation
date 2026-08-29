# doc pr/128 — PF/kine completeness: reconstructed objects that reach no output

**Status: OPEN (measurement complete, implementation in progress).**

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

*(pending — knob-off byte-identical gate on both manifests, per-event
ΔEnu decomposition, movers, nusel, Bee A/B)*

## 8. Sentinels

Every knob here carries a distance or length threshold tuned to measured
geometry, i.e. exactly the exposure class doc pr/127 §5.1 built the sentinel
registry for. New entries go into `scripts/pr127_sentinels.py` in the same
commit as the flip.

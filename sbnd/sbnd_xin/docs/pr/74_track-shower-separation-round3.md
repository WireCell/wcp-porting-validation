# doc pr/74 — track/shower separation, round 3: the owner's three residual findings

Round 2 (`74_track-shower-separation-round2.md`, toolkit `96054e1e` +
`2638faa8` + `064824c1`) shipped four knobs and flipped them to SBND
production. The owner then scanned the round-2 Bee set and filed three
findings:

1. **18255-469665** — the segment at (x,y,z) = (5.8, 110.8, 381.2) is painted
   EM shower but does not show up in the particle flow. *Where is it?*
2. **18255-142421** — the gamma cluster at (89.2, −71.8, 242.0) now has its
   **direction reversed**; the start point should be the end near the
   neutrino cluster.
3. **18255-506746** — the `107 MeV electron` is really a **muon + Michel**.

Findings 1 and 2 are **defects this round fixes**, both of them introduced or
left behind by round 2's own knobs. Finding 3 is **pre-existing and
diagnosis-only** — no pr/74 knob touches it in either round.

This round also **corrects a claim round 2 made about 18255-90055** (§ 5).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)

# --- the three findings, read straight out of the round-2 production arm
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr74r2-flip48 469665 --at 5.8,110.8,381.2
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr74r2-flip19 142421 --at 89.2,-71.8,242.0
python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr74r2-flip19 506746

# --- the NEW gate that would have caught finding 1 in round 2
python3 scripts/analysis/pr74/pr74_pf_roots.py work-pr51r7-on48 work-pr74r2-flip48   # 3 GAINED
python3 scripts/analysis/pr74/pr74_pf_roots.py work-pr51r7-on48 work-pr74r3-on48     # 0 GAINED

# --- Q3 structure + dQ/dx (needs the calib dump)
PR_EXTRA_STAGES=pr_display WCT_PID_WRITE_DEBUG=2 SBND_WCT_LOGLEVEL=debug PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr74r3-q3dbg data 506746
grep -E "gidx=(48|51|56)\b" work-pr74r3-q3dbg/pr_evt506746/stdout.log

# --- off-gate (0/117) and the round-3 production arms
export SBND_SHOWER_IN_CASCADE_GUARD=0 SBND_MICHEL_STEM_MICHEL_CHECK=0 \
       SBND_SHOWER_STEM_BACKFILL=0 SBND_SHOWER_CONN3_UNREACHABLE=0
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r3-off48 data   # + 19, 50
unset SBND_SHOWER_IN_CASCADE_GUARD SBND_MICHEL_STEM_MICHEL_CHECK \
      SBND_SHOWER_STEM_BACKFILL SBND_SHOWER_CONN3_UNREACHABLE
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r3-on48 data    # + 19, 50
python3 scripts/analysis/pr49/on_compare.py   work-pr74r2-off48 work-pr74r3-off48
python3 scripts/analysis/pr51/nuvtx_census.py work-pr74r3-off48 work-pr74r3-on48
```

**pctree gate, the pr/57 trap** — `hash_archive.py` prints
`<hash>  <nmembers>  <path>`. The path differs between arms by construction,
so hashing the whole line reports 117/117 "differences". Compare **column 1
only** (`awk '{print $1}'`). This bit me once in this round.

---

## 1. Finding 1 (469665) — root cause: an absorbed stem strands its PF children

### Symptom

The owner's coordinate is seg **33008** (cluster 33, 68 points, `L_pca` 3.6 cm,
painted SHOWER — `pr74_paint_pf.py --at` puts it 0.05 cm from a painted point).
It *is* in `0-mc.json` in both arms. What changed is its **parentage**:

| | 469665 PF placement of 33008 |
|---|---|
| pre-pr/74 production | `15004 proton` → `15003 mu-` → `gamma 7 MeV` → **`33008 e-`** |
| round-2 production | **top-level root** `gamma 7 MeV` → `33008 e-`, start (26,81,300), 25.6 cm from the ν vertex |
| round 3 | `15004 proton` → `15003 e- 487 MeV` → `gamma 7 MeV` → **`33008 e-`** |

So "missing from the particle flow" is exact in the sense that matters: the
node stopped being part of the neutrino interaction and floated to the top
level of the jsTree, anchored nowhere.

### Root cause

`fill_bee_pf_tree` BFSs from the main vertex through **track-only** segments
and records `vtx_incoming_seg[V]` = the segment that reached V
(`MultiAlgBlobClustering.cxx:1204-1254`). Everything anchored at V — a
shower, a pseudo-gamma, a nested shower — hangs off that segment.

Round 2's K4 (`shower_stem_backfill`) added the walked-past stem to the
shower's **membership only**. That fixes the paint layer, but the stem stops
being a track, so the vertex it used to reach drops out of `vtx_incoming_seg`
and **every object anchored there loses its parent**.

This is the *same* pathology doc pr/74 opened with for 142421 ("painted EM
shower, missing from the particle flow"), re-created by the fix for it.

### Fix (in `PatternAlgorithms::stem_backfill`, inside the existing K4 knob)

Re-seat the shower onto the stem chain it just absorbed:

```
shower->set_start_vertex(cur, 1);        // cur = main-vertex side of the last absorbed stem
shower->set_start_segment(outer_stem);
shower->update_particle_type(...);       // majority vote: track-typed start segment 13 -> 11
shower->set_flag_kinematics(false);      // let calc_kine_2 actually recompute
```

Three things had to be right together, and each was found by measurement:

- **`set_start_vertex` + `set_start_segment`** — the same pair the BFS shower
  builder itself uses (`NeutrinoShowerClustering.cxx:240-241`), with
  `conn_type = 1` because an absorbed stem is a genuine charged connection,
  not a gap. This alone re-parents the orphans.
- **`update_particle_type` BEFORE the recompute** — `calculate_kinematics`
  copies the start segment's PDG onto the shower verbatim
  (`PRShower.cxx:1121`). Without the vote first, re-seating a 2 GeV electron
  shower onto a `mu-`-typed stem would turn the shower into a muon.
- **`set_flag_kinematics(false)`** — the one that made the other two visible.
  `calculate_shower_kinematics` **skips any shower whose kinematics flag is
  already set** (`NeutrinoEnergyReco.cxx:327`), and calc_kine_1 set it for
  every shower that existed then. So **round 2's "before the second
  kinematics pass, so absorbed charge is counted" was false**: nothing was
  recomputed. Measured — 90055 stayed 2020 MeV and 469665 stayed 322 MeV
  across the round-2 flip, and 469665's shower node kept a start point
  (30,124,356) that is 100 cm from the stem it had just absorbed.

`dirsign` is also pinned on the re-seated stem (point it away from `cur`)
because `conn_type == 1` selects the start point by `dirsign`
(`PRShower.cxx:1141-1147`) and a stem the tracker never directioned has
`dirsign() == 0`, which assigns **neither** endpoint and silently leaves
`start_point` stale. That was the second measured false start.

### Evidence

| event | round-2 production | round 3 |
|---|---|---|
| **469665** | `gamma 322 MeV` root at 26.6 cm; 33008 orphaned; shower energy stale (see § 6) | `15004 proton 78 MeV` → `15003 e- 487 MeV` from (25,64,283); 33008 nested |
| **90055** | 7 dangling roots incl. the 2020 MeV shower at 13.6 cm | `11045 e- 2061 MeV` from **(129,25,202) = the ν vertex**; 11043 + 4 pseudo-gammas + neutron all children |
| **138009** | `neutron 4 MeV` root 104.6 cm out | nested under `12090 e- 1187 MeV`, which starts at the ν vertex |

The cleanest single number: on 90055 **`kine_pio_vtx_dis` goes 13.6 → 0.0 cm**
— the shower now starts exactly at the neutrino vertex, which is what round 2
claimed and did not deliver.

---

## 2. Finding 2 (142421) — root cause: K5 anchored the shower to its own far end

### Symptom

Round-2 production emitted

```
gamma 795 MeV   start=(85,-76,249) end=(85,-76,249)     <- zero length
  e- 795 MeV    start=(85,-76,249) end=(110,-71,220)    <- runs INWARD
```

The ν vertex is (118,−70,209). The electron ran from the far end back toward
the vertex — reversed, exactly as the owner reported.

### Root cause

K5 promotes a graph-**unreachable** main-cluster component by anchoring it to
the nearest candidate vertex. But the candidate list `vertices` contains the
promoted component's **own endpoints**, which sit at distance **0** from it.
So the "nearest vertex" was always its own far end — the round-2 log line says
`anchor_dis 0.0cm` and I read that as a tight anchor rather than as the tell
it was. Because conn-3 derives `start_point` from the anchor
(`PRShower.cxx:1140`) and `end_point` from the farthest vertex, the whole
object came out backwards.

The owner's framing is the right one: *this cluster comes off the neutrino
vertex, so the correct direction for the EM shower is simply the natural
one* — outward from the vertex. The bug was that the code never consulted the
vertex; it consulted the segment itself.

### Fix

New `PR::reachable_vertices(graph, root)` (`PRGraph.cxx`, the vertex-level
complement of pr/65's `unreachable_segments`, deliberately duplicated rather
than factored out so the production BFS stays byte-for-byte). The K5 anchor
loop now skips any candidate the main vertex cannot reach:

```cpp
if (!reachable_vtxs.count(vtx)) continue;   // never anchor to your own component
```

### Evidence — round 3, `work-pr74r3-on19`

```
gamma 795 MeV   start=(118,-70,209) end=(110,-71,220)   <- 14.3 cm, from the ν vertex
  e- 795 MeV    start=(110,-71,220) end=(85,-76,249)    <- 37.8 cm, OUTWARD
```

Connection type stays **3**, and `kine_reco_Enu` is unchanged (+795.3 MeV vs
off, same as round 2) — this is a pure geometry/direction correction.

**Why conn 3 here and conn 1 for K4's re-seat (§ 1)** — the two are opposite
choices in the same round, on purpose. K4 absorbs a stem that is a *charged,
continuous* connection between the vertex and the shower, so there is no gap
to model: conn 1, no pseudo-gamma, start point taken from the stem itself. K5
promotes a component the graph cannot reach at all; after the fix the anchor
sits 14.3 cm from the shower start with nothing reconstructed in between, so
the gap is real and conn 3's pseudo-gamma is the honest representation of it.
The round-2 bug was not the conn type — it was that the "gap" was 0.0 cm
because the anchor was the object's own endpoint.

---

## 3. Finding 3 (506746) — pre-existing muon+Michel, diagnosis only

**Not caused by any pr/74 knob.** `21048 e- 107 MeV` has byte-identical
`start`/`end`/KE in the pre-pr/74 arm, the round-2 arm and the round-3 arm.
K2's only effect on this event is `21052` (`pi+ 207 MeV` → `proton 376 MeV`).

### What it actually is

Not one segment — a **three-segment chain** running from the ν vertex
(vertex 21038), read out of the `pr_display` calib dump:

| seg | length | median dQ/dx | vertices | PF/graph role |
|---|---|---|---|---|
| 21048 | 20.4 cm | **1.38× MIP** | 21037 → 21038 (ν vertex) | shower start segment |
| 21056 | 8.0 cm | **1.00× MIP** | 21034 → 21037 | shower member |
| 21051 | 9.5 cm | **0.82× MIP** | 21034 → 21040 | shower member |

37.9 cm total: ~28 cm of flat, MIP-level charge and then a short sub-MIP blob.
That is a muon-then-Michel profile, not an EM cascade — a shower would rise
and branch, not sit at 1.0× MIP for 28 cm. The owner's reading is supported by
the charge.

### Why it is mis-typed, and why K1 cannot catch it

All three were typed electron by `segment_determine_shower_direction_trajectory`
(`PRSegmentFunctions.cxx:2778`) — the **shower-trajectory** branch, i.e.
`kShowerTrajectory` set by track/shower *separation*, upstream of PID. Seg
21056 is the tell: it was first typed **muon** (pdg 13, `:2703`) and then
overwritten to electron at `:2778`.

- Different mechanism from owner case 3 (53361). K1
  (`shower_in_cascade_guard`) guards `examine_direction`'s `flag_shower_in`
  cascade in `NeutrinoVertexFinder.cxx`, not the trajectory branch.
- Different length regime. K1 requires > 40 cm; the longest segment here is
  20.4 cm. Lowering that threshold to ~20 cm would change the separation
  behaviour of the whole sample.
- Different confidence signature. Round 1 found the four original cases at
  `particle_score` 0.05–0.27 ("the classifier does not know"). These three
  carry `particle_score 100.0` — the shower-trajectory sentinel. Proposal P4
  (defer on low score) would not fire here either.

### Proposal (not implemented this round)

A `shower_traj_mip_chain_guard`: refuse `kShowerTrajectory` on a chain rooted
at the main vertex whose segments are **all** MIP-like (median in ~[0.8, 1.6]×
MIP) over a total length above a floor (~25 cm), leaving the terminal sub-MIP
blob free to be its own Michel shower. This lives in the separation stage,
which governs **every** event and every sample — it needs its own default-OFF
knob and its own 117-event census, and shipping it alongside two corrective
fixes would muddy this round's mover attribution. Owner call whether it
becomes round 4.

---

## 4. New gate: `scripts/analysis/pr74/pr74_pf_roots.py`

Round 2 had three regression metrics and **none of them could see finding 1**:

- `on_compare.py` compares archive member hashes — it flags 469665 as a
  mover, which round 2 already knew and had attributed to the intended fix.
- the round-2 orphan sweep looked for painted objects with **no node** in
  `0-mc.json` — 33008 always had a node.
- `nuvtx_census.py` watches the ν vertex and Enu — both fine.

The failure mode is a node that **survives but loses its parent**. The
discriminant is not "is it a root" (every primary is a root, legitimately, and
they all start *at* the ν vertex) but **"is it a root whose start point is
nowhere near the ν vertex"**. The ν vertex is read from `T_tagger`
`nu_{x,y,z}` row 0, the same source `nuvtx_census.py` uses — deliberately not
"the most common root start", which degenerates on a one-root tree.

Retro-check, i.e. what round 2 would have seen:

```
work-pr51r7-on48 -> work-pr74r2-flip48 : GAINED a dangling root : 3   (90055, 138009, 469665)
work-pr51r7-on48 -> work-pr74r3-on48   : GAINED a dangling root : 0
```

Added to the standard on-census.

---

## 5. Correction to round 2

Round 2 § "Per-knob smoke evidence" claimed, for K4 on 90055:

> PF: `e- 2020 MeV (11044)` directly at the vertex, no separate trunk node

**The first half is wrong.** There was no separate trunk node (true), but the
shower did **not** sit at the vertex: `11044` kept `start = (127,24,216)`,
13.6 cm out, and it lost its parent in the process. The paint-layer half of
that bullet (11045's points joining the shower, 7831 → 8176) was correct — and
that is precisely the trap: the paint layer was right and I read it as
evidence about the PF tree. Round 3 delivers the claim as originally written
(`start = (129,25,202)`, `kine_pio_vtx_dis` 13.6 → 0.0 cm).

Round 2 § "Gates" also reported *"Zero stranded orphans anywhere"*. That
statement is true **only for the metric it was measured with** (painted object
with no node at all). By the parent-loss metric the same arm had 11 stranded
objects across 3 events. The round-2 doc has been annotated in place.

---

## 6. Gates

All on the shipped round-3 binary. Freshness proof done before every A/B
(`local/lib/libWireCellClus.so` 05:54 > last source edit 05:53).

- **Off-gate PASS** — `work-pr74r3-off{48,19,50}` vs
  `work-pr74r2-off{48,19,50}`: ARCHIVE-LEVEL **0/48, 0/19, 0/50**;
  pctree member hashes **0/117**; `nusel-events` / `nusel-table` **0/117**.
  The round-3 code is byte-identical with the knobs off.
- **Doctests** — `./build/clus/wcdoctest-clus` **2006/2006 pass**, rc=0.
- **PF-root gate PASS** — pre-pr/74 production → round-3 production:
  **0 gained** on the 48 and 19 arms (round 2: 3 gained). One gained on the 50
  arm, 53361, attributed in § 7.
- **On-census**, `work-pr74r3-off*` → `work-pr74r3-on*` (total knob
  footprint): archive movers **7/117** — the same 7 as round 2 (90055,
  138009, 350186, 469665, 142421, 506746, 53361), every one attributed.
  **nusel event labels 0/117 flips.**
- **ν-vertex census**: `nu-vtx > 10 cm` **0/117**; every mover has
  `dvtx = 0.00 cm`. No vertex anywhere moves.
- **Round-2 → round-3 delta** (what this round actually changed in
  production): **4/117 archive movers** — 90055, 138009, 469665 (K4 re-seat)
  and 142421 (K5 anchor). 0/50 on the data arm. nusel 0/117.

### Energy movers (round 2 → round 3): 3, all K4, all intended

| event | Δ`kine_reco_Enu` | mechanism |
|---|---|---|
| 469665 | **+165.6 MeV** | shower KE 322 → 487 MeV — see the note below: this is the energy catching up to a membership round 2 had **already** changed |
| 90055 | **−115.3 MeV** | shower KE 2020 → 2061 MeV, and the shower start moves to the ν vertex (`kine_pio_vtx_dis` 13.6 → 0.0, `kine_pio_theta_1` 83.3° → 19.5°, `kine_pio_mass` 213 → 46 MeV), so the π⁰ hypothesis follows |
| 138009 | **−105.4 MeV** | almost entirely `kine_reco_add_energy` 219.9 → 114.3: the 104.6 cm-away blob was being counted **both** inside the shower charge and again as additional energy; re-parenting removes the double count |

**Precise statement of the 469665 / 90055 energy change** — it is not "charge
that was never counted". Round 2 *did* absorb the stems into the shower's
membership; the paint census proves it (469665 rcid 68054 went 1571 → 2536
points, 90055 rcid 11044 went 7831 → 8176). What round 2 did **not** do is
recompute the energy, because of the kinematics-flag skip in § 1. So the
round-2 production arm was shipping, for these showers, **an energy that did
not correspond to its own membership**: 322 MeV computed for 1571 points while
the shower held 2536. Round 3's 487 MeV is the energy of the membership that
was actually there. That is a worse round-2 defect than "undercounted", and it
is stated that way deliberately.

These are also the first Enu changes K4 has ever produced — round 2's K4 could
not change an energy at all.

### The recompute is confined to the re-seated shower

`set_flag_kinematics(false)` unblocks `calc_kine_2` for one shower, but
`calculate_shower_kinematics` loops over all of them and the second pass reuses
the once-collected `m_charge_*` maps. Checked directly rather than inferred:
comparing every **segment-encoded** PF node id (`cluster*1000 + gidx`; the
small integer ids are `next_id++` pseudo-particles and renumber whenever tree
shape changes) between round-2 and round-3 production on the three K4 events,

- **90055**: 12 of 13 objects byte-identical in pdg and KE (11054 proton 174,
  11043 e- 19, 146160 e- 49, 13061 e- 44, 115091 proton 84, …). Only the
  re-seated shower changes, and its node id moves 11044 → 11045 because the id
  is `seg_display_id(start_segment)`.
- **469665**: 58018 e- 160 MeV, 67069 e- 48 MeV, 66044 e- 58 MeV, 33008 e- 7
  MeV, 63031 e- 7 MeV, 15004 proton 78 MeV — all unchanged. Only 68054 →
  15003.
- **138009**: 12091 proton 222, 12092 mu- 6, 32033 e- 5, 39040 proton 4 — all
  unchanged. Only 12015 → 12090.

So a shower that was kinematics-complete and is not re-seated keeps its value
exactly; the mechanism is confined, not merely un-caught by the gate.

---

## 7. 53361 — one attributed PF root, not a K1 defect

K1 restores the muon exactly as round 2 reported
(`27001 e- 405 MeV` → `27001 e- 28 MeV` + `27004 mu- 280 MeV`). The
side-effect the new gate catches: the 71 MeV proton blob at (−18,−193,169)
was a child of the 405 MeV fake electron and is now a **root**.

Mechanism: with K1 on, the first segment out of the main vertex is a shower
(`27001`), so the track BFS never reaches `27004`; the muon is rendered
through the pr/38-R4 orphan-parentage path, and that path registers its far
vertex in `vtx_incoming_seg` at `MultiAlgBlobClustering.cxx:1587` — **after**
the shower-attachment loop at `:1350-1500` has already resolved the 71 MeV
pseudo-shower as a root. It is an **ordering limitation in the PF-tree
builder**, not something in K1's logic.

Not fixed here on purpose: reordering the builder's parent resolution changes
a shared path that runs on every event, and the advisor's blind-spot check for
this round was precisely "don't touch general parent resolution while fixing
an absorption-specific bug". Proposal: `pf_orphan_anchor_before_showers`,
default OFF, hoisting the orphan far-vertex registration above the shower
loop, with its own 117-event PF-tree diff.

Net accounting, absolute (not delta), on the round-3 production arms:
**3/113 events carry any dangling PF root** — 168596 (4) and 285567 (4) are
**pre-existing and bit-identical to the pre-pr/74 baseline**, plus 53361 (1)
described above.

---

## 8. Bee

Index with per-event notes: `docs/pr/pr74r3-bee.index.txt`. Same 7 events, same
order as the round-2 index, so the sets step side by side.

- **before** (= round-2 production, the set the owner scanned):
  <https://www.phy.bnl.gov/twister/bee/set/a8264f65-b41d-488e-bdbb-08bd71fc195f/event/list/>
- **after** (= round-3 production):
  <https://www.phy.bnl.gov/twister/bee/set/794952df-7fcc-4a60-9dfc-eaaf5e648082/event/list/>

Owner coordinates: finding 1 at bee_idx **3** (469665), finding 2 at bee_idx
**4** (142421), finding 3 reference at bee_idx **5** (506746).

## 9. Status

- Findings 1 and 2 **fixed and validated**; the four knobs stay **SBND
  production ON** (no cfg change this round — `wct-pr-perevt.jsonnet` is
  untouched, the fixes are inside the existing knob bodies).
- Finding 3 **diagnosed, not fixed** — proposal in § 3, owner call.

### Open items

- `shower_traj_mip_chain_guard` (§ 3) — the muon+Michel class.
- `pf_orphan_anchor_before_showers` (§ 7) — the 53361 PF-root ordering limit.
- 469665's fragmentation into 5 clusters — still out of scope (owner's
  round-2 decision: stem fix only).
- `stem_backfill_mip_hi` margins unchanged: absorb measured at 3.21×, stop at
  3.71×, threshold 3.5×.
- P4 (low-score defer) and P5 (`demote_len` review) still unimplemented. § 3
  shows P4 would not have caught 506746.

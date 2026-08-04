# doc pr/33 — EM shower clustering: prototype↔toolkit fidelity audit

**Why.** Sixth in the series after pr/28 (vertex fit + trajectory dQ/dx), pr/29
(Steiner graph build), pr/30 (proto-vertex + track-segment finding), pr/31
(topology / PID / direction) and pr/32 (neutrino vertex identification).  Scope
is **step 5 of the eight** in doc pr/27 §0, defined in doc pr/27 §7: assemble
`PR::Shower` objects, merge over-split ones, identify π⁰ pairs.

**Status. AUDIT ONLY. No code was changed, no patch is proposed.** Every P-row
below is CLAUDE.md §5 rule 1 (it changes production output unconditionally) and
is the owner's call.

> **§10 (2026-08-04) applies the owner's filter: fourteen → five.**  Kept:
> **F1 = P1** (wrong daughter-count callee, both sites), **F2 = P2** (whose PDG,
> five sites), **F3 = P3** (π⁰ id collision — now with two live downstream
> consumers), **F4 = P6** (`abs(pdg)==11`, **one** live site, not two), and
> **F5 = P12** (`shower_less`'s address fallback — *a different class*: a
> house-rule violation, not a prototype divergence).  **P7 is RESOLVED, not
> merely dropped** — §7 loose end 3 is closed.  §10.8 records **eight
> corrections to §3/§7/§8 of this doc**, including that every anchor below is up
> to 19 lines stale (read at `f07c0299`, re-verified at `407c5ba9`).

**Headline.** The control flow is an exact port and most of the arithmetic is
faithful.  Fourteen behaviour-changing divergences follow.  Two dominate:

* **P1 — the wrong daughter-counting function is called at both of its call
  sites.** The prototype calls `calculate_num_daughter_tracks`; the toolkit
  calls `calculate_num_daughter_showers`.  These are *different* functions in
  both trees with *inverted* counting predicates, and the two sites err in
  opposite directions.  One of the two was **introduced** by applying the
  in-tree review doc's recommendation B.1.
* **P2 — "whose PDG" is read differs at five sites, in both directions.** Where
  the prototype reads `shower->get_start_segment()->get_particle_type()` the
  toolkit reads `shower->get_particle_type()` (four sites), and at one site the
  substitution runs the other way.  These are genuinely different quantities:
  the shower's type comes from `update_particle_type`'s majority vote.

**This is the first stage in the series that already has an in-tree review
doc.**  `clus/docs/patternrecognition/shower_clustering_review.md` (477 lines)
covers exactly this file pair and lists seven items B.1–B.6 + L.1.  §3.0 below
re-verifies all seven at HEAD.  Six were applied; one (L.1) was applied at its
original site and now recurs at a new one.  **Every line anchor in that doc has
drifted** — it describes a 3310-line file; HEAD's is 3384.

**Which toolkit was read.** `git show HEAD:` snapshots taken at commit
**`f07c0299`** *before* reading, into
`/home/xqian/tmp/claude-25225/pr33/`.  A concurrent session is editing this
tree.  Note this is **two commits later** than pr/30–pr/32, which all read
`4f2e7303`; `c05bc5f7` (T_tagger `boost::edges`/`vertices`/`graph_nodes` sweep)
and `f07c0299` (index aliasing) landed in between.  §6 accounts for that.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git show HEAD:clus/src/NeutrinoShowerClustering.cxx > /home/xqian/tmp/claude-25225/pr33/NeutrinoShowerClustering.cxx
git show HEAD:clus/src/NeutrinoEnergyReco.cxx       > /home/xqian/tmp/claude-25225/pr33/NeutrinoEnergyReco.cxx
git show HEAD:clus/inc/WireCellClus/PRShower.h      > /home/xqian/tmp/claude-25225/pr33/PRShower.h
git rev-parse HEAD          # f07c0299

# prototype side (read-only)
P=prototype_base/pid/src
wc -l $P/NeutrinoID_shower_clustering.h $P/NeutrinoID_em_shower.h    # 1939 + 652

# the constant-histogram triage helper (regex fixed, see §7 loose end 6)
python3 /home/xqian/tmp/claude-25225/pr33/consts.py \
  <toolkit_file> <lo> <hi> <prototype_file> <lo> <hi>
```

No event was run.  See §9.

---

## §0 Scope

**Entry point.** `PatternAlgorithms::shower_clustering_with_nv`
(`clus/src/NeutrinoShowerClustering.cxx:3160`), called from
`TaggerCheckNeutrino.cxx:716`.  Prototype `NeutrinoID::shower_clustering_with_nv`
(`NeutrinoID_shower_clustering.h:268`), called from `NeutrinoID.cxx:246`.

### Function map (re-derived at `f07c0299`; every anchor read, none inherited)

| # | toolkit (`NeutrinoShowerClustering.cxx`) | prototype | notes |
|---|---|---|---|
| 1 | `update_shower_maps` `:30` | `shower_clustering.h:1388` | |
| 2 | `shower_clustering_with_nv_in_main_cluster` `:76` | `:1654` | |
| 3 | `shower_clustering_connecting_to_main_vertex` `:224` | `:114` | P1, P4 |
| 4 | `shower_clustering_with_nv_from_main_cluster` `:464` | `:1775` | P5 |
| 5 | `shower_clustering_with_nv_from_vertices` `:753` | `:995` | P6 |
| 6 | `examine_merge_showers` `:1214` | `:380` | P2 |
| 7 | `shower_clustering_in_other_clusters` `:1295` | `:1442` | B.4, B.6 |
| 8 | `examine_shower_1` `:1632` | `em_shower.h:337` | |
| 9 | `examine_showers` `:2082` | `em_shower.h:1` | P1 |
| 10 | `id_pi0_with_vertex` `:2439` | `:735` | P3, P9 |
| 11 | `id_pi0_without_vertex` `:2783` | `:428` | P2, P3 |
| 12 | `shower_clustering_with_nv` `:3160` | `:268` | P7 |
| — | `collect_charge_maps` `NeutrinoEnergyReco.cxx:226` | `collect_2D_charges` `energy_reco.h:1` | P7 |
| — | `calculate_shower_kinematics` `NeutrinoEnergyReco.cxx:281` | `shower_clustering.h:1407` | |
| — | `calculate_num_daughter_showers` / `_tracks` `NeutrinoPatternBase.h:423-424` | `track_shower.h:688` / `:724` | **P1** |
| — | `Shower::complete_structure_with_start_segment` `PRShower.cxx:269` | `WCShower.cxx:703` | read |
| — | `WCShower::fill_maps` | `WCShower.cxx:608` | read |

### Prototype functions in the file that are *not* stage 5

* **`shower_determing_in_main_cluster`** (`shower_clustering.h:15-113`).  Looks
  unported — no toolkit symbol of that name.  It is a 12-call driver over
  `examine_good_tracks` / `fix_maps_*` / `improve_maps_*`, i.e. **stage 3**, and
  is exactly what doc **pr/31** audited as `NeutrinoTrackShowerSep.cxx`'s entry
  points.  Not a gap.  §5.2.
* **`get_start_end_vertices`** (`:1431`).  Absorbed into `find_vertices(graph,
  sg)`; its ordering is doc **pr/30**'s subject.  §5.3.

### Not audited (stated so the doc's reach is not overstated)

* `PRShower.cxx` (1273 lines) ↔ `WCShower.cxx` (756).  This is the shower
  *data-structure* port, not stage 5.  **Exception**: `fill_maps` and
  `complete_structure_with_start_segment` were read in both trees because
  sub-call 1 rests on them.  A divergence could hide in the other ~1800 lines.
* `cal_kine_charge` internals and `cal_corr_factor` — doc pr/27 §9, energy.
  Only `collect_charge_maps`'s **placement** is in scope (P7).
* `TrackFitting::collect_2D_charge` (`TrackFitting.cxx:933`) — the delegate.
* `examine_shower_1` and `examine_showers` were read in full for control flow,
  constants and the patterns catalogued here, but not line-by-line for
  arithmetic across all 450 + 357 lines.  §7 loose end 5.

---

## §1 Trust tiers

Carried unchanged from pr/28 §3b through pr/29–pr/32.

* **Tier A — read in both trees, both directions, quoted here.**  §3's P1–P14
  and §2's match claims.
* **Tier B — cleared by the constant sweep and a targeted read, not line by
  line.**  Flagged inline where it applies.

Nothing in this doc is Tier C (asserted from a doc rather than from source),
**including the seven items re-verified in §3.0** — each was re-read at HEAD
rather than taken from the review doc's word.

---

## §2 What matches

Cataloguing this first is what makes fourteen divergences readable as behaviour
changes rather than a broken port.

**§2.1 The entry point's call order is exact.**  Eleven sub-calls in the same
sequence, including both `calculate_shower_kinematics` passes and the
`shower_clustering_in_other_clusters(..., true)` flag:

| # | prototype `:275-308` | toolkit `:3208-3294` |
|---|---|---|
| 1 | `shower_clustering_with_nv_in_main_cluster` | `:3208` |
| 2 | `shower_clustering_connecting_to_main_vertex` | `:3217` |
| 3 | `shower_clustering_with_nv_from_main_cluster` | `:3224` |
| 4 | `shower_clustering_with_nv_from_vertices` | `:3231` |
| — | *(none — see P7)* | `collect_charge_maps` `:3242` |
| 5 | `calculate_shower_kinematics` | `:3248` |
| 6 | `examine_merge_showers` | `:3253` |
| 7 | `shower_clustering_in_other_clusters(true)` | `:3261` |
| 8 | `calculate_shower_kinematics` | `:3272` |
| 9 | `examine_showers` | `:3277` |
| 10 | `id_pi0_with_vertex` | `:3285` |
| 11 | `id_pi0_without_vertex` | `:3294` |

**§2.2 `update_shower_maps` is faithful.**  Same four maps, same clear-then-fill,
and the toolkit's exclusion of the start vertex from `map_vertex_in_shower`
(`:51-56`) is **correct**: prototype `WCShower::set_start_vertex`
(`WCShower.cxx:529-532`) writes only `start_vertex`, never `map_vtx_segs`, so
`fill_maps` (`:608-615`) never sees it.  The toolkit's in-code comment saying
exactly this is right.

**§2.3 `complete_structure_with_start_segment` shares `used_segments` across
rejected showers.**  In the prototype, a shower that fails the
`shower_clustering_connecting_to_main_vertex` selection is `delete`d — but the
segments it claimed stay in the shared `used_segments`, so they still block
later candidates.  The toolkit reproduces this exactly: `used_segments` is a
by-reference `IndexedSegmentSet` and rejected showers are simply never inserted
into `showers`.  This is a subtle side effect and it is ported correctly.

**§2.4 Every operator-precedence trap is reproduced.**  The prototype leans on
`&&` binding tighter than `||` in at least six conditions; each is explicitly
parenthesised in the toolkit with the same grouping.  Checked pairwise:

| prototype | grouping | toolkit |
|---|---|---|
| `:146` PID skip | `A \|\| (B && ((C&&D) \|\| E)) \|\| (F && G)` | `:283-290` |
| `:1806` shower-dir gate | `A \|\| B \|\| (C && D)` | `:528-530` |
| `:1917` angle/distance | `(A&&B) \|\| (C&&D) \|\| (E&&F)` | `:700-702` |
| `:1526` add-segment | `(A&&B) \|\| (C&&D)` | `:1447-1448` |
| `:1543` merge | `(A&&B&&C) \|\| (D&&E&&F)` | `:1484-1485` |
| `:1012` shower-or-weak | `A \|\| B \|\| ((C\|\|D) && E)` | `:788-794` |

**§2.5 The `angle_offset` ladder is exact.**  `25 + off`, `12.5 + off*8/5`,
`5 + off*2` against 80 / 130 / 200 cm, with `off ∈ {0, 5}`.  `angle_offset` is
`double` in both trees so `off*8/5 = 8.0`, not integer-truncated.

**§2.6 The elliptical distance metric is exact.**
`(d·cosθ)²/(40 cm)² + (d·sinθ)²/(5 cm)²`.  The prototype's `θ` is in radians;
the toolkit converts to degrees for the cuts and back to radians for the metric
(`:705-706`) — algebraically identical.

**§2.7 The drift-parallel test is equivalent despite looking different.**
Prototype `fabs(dir.Angle(drift)/π*180 − 90) < 5` with `Angle ∈ [0,π]`; toolkit
takes `|cos|` *first* (`:568`), folding the angle into `[0°,90°]`.  Since
`|θ−90| = |(180−θ)−90|`, the folded and unfolded tests agree for all inputs.
Same 5° threshold, same 50 cm re-computation radius, same `offset = 5`.

**§2.8 The π⁰ mass ladder is exact.**  `mass_offset = 10 MeV`, the
`(ct1==2 && ct2==2) → 6 MeV` penalty, the acceptance window
`−25 MeV < Δ < 35 MeV`, the `|Δ| − penalty < |mass_diff| − mass_penalty`
comparison, `1000 cm` seed for `best_vtx_dis`, and the
`√(4·E₁·E₂·sin²(θ/2))` mass — all identical (prototype `:896-935`, toolkit
`:2679-2703`).  `135 MeV` and `139.57 MeV` match `mp.get_mass_pion()`.

**§2.9 The π⁰-vertex pion reclassification preserves the `cal_4mom` guard.**
Prototype `:986-987`: `if (sg->get_particle_4mom(3)>0) sg->cal_4mom();`.
Toolkit `:2772-2776`: `if (sg->particle_info()->kinetic_energy() > 0) { … }`.
This is **the guard that doc pr/31 P1 found dropped at 11 of 13 sites** — here
it is ported, and the in-code comment names the prototype line.  Worth
recording as a positive: the omission is not universal.

**§2.10 The both-short `break` and the `pdg==11` fallback are present.**
`:3041` and `:2916-2923` — see §3.0.

**§2.11 Constant sweep.**  Twelve function pairs, every numeric literal with
its units suffix, comment- and debug-print-stripped.  After resolving member
substitutions (`43e3 → m_mip_dqdx_median`), formatting (`180.` vs `180.0`,
`3.1415926` vs `M_PI`) and prototype-only debug scaffolding, **zero genuine
constant divergences in the eight pairs read in full**; the other four were
cleared by the sweep plus a targeted read of each flagged literal.

---

## §3 Divergences

### §3.0 Re-verification of the in-tree review doc at HEAD

`clus/docs/patternrecognition/shower_clustering_review.md` is the only prior
in-tree review of this stage.  Its "Status" column reads `**Fix**` for six
items, which is the *recommended action*, not a claim that it was applied.  All
seven re-read at `f07c0299`:

| id | claim | state at HEAD | evidence |
|---|---|---|---|
| B.1 | `calculate_num_daughter_showers(..., false)` should drop the `false` | **applied** | `:283` now `calculate_num_daughter_showers(graph, main_vertex, sg)` |
| B.2 | `continue` → `break` when both showers short | **applied** | `:3041` `break; // … (prototype line 614)` |
| B.3 | add `\|\| abs(pdg)==11` to the shower-flag test | **applied** | `:2916-2923` |
| B.4 | add post-merge `update_particle_type` + `calculate_kinematics` | **applied** | `:1494-1496`, comment cites prototype `:1555-1556` |
| B.5 | π⁰ direction from each shower's own start vertex | **applied** | `:2651-2658` (`sv1_pt`, `sv2_pt`) |
| B.6 | extra electron-forcing in sub-pass 1, documented not removed | **still present** | `:1453-1467` |
| L.1 | hardcoded `0.511 * units::MeV` | **half** | original site now `particle_data->get_particle_mass(11)` (`:157-158`); a *new* hardcode exists at **`:743`** |

**B.1 is the problem.** See P1 — the recommendation was based on a
misidentification of the callee, and applying it turned a site that was
semantically *correct* into one that diverges.

---

### P1 — the wrong daughter-counting function, at both call sites, erring in opposite directions

Both trees define **two** distinct functions:

| function | counts a segment iff |
|---|---|
| `calculate_num_daughter_showers(vtx, sg, flag = true)` — proto `track_shower.h:688` | `get_flag_shower() \|\| !flag` |
| `calculate_num_daughter_tracks(vtx, sg, flag = false, len_cut = 0)` — proto `:724` | `(!get_flag_shower() \|\| flag) && length > len_cut` |

They are **not** two names for one thing: with `flag = true`, `_showers` counts
*only* shower-flagged segments while `_tracks` counts *everything*.

| site | prototype call | counts | toolkit call | counts |
|---|---|---|---|---|
| `shower_clustering_connecting_to_main_vertex` | `_tracks(main_vertex, sg, true)` — `shower_clustering.h:140` | **all** segments | `_showers(graph, main_vertex, sg)` — `:283`, flag defaults `true` | **shower-flagged only** |
| `examine_showers` | `_tracks(main_vertex, sg, false).second` — `em_shower.h:17` | **track** segments only | `_showers(graph, main_vertex, sg, false).second` — `:2186` | **all** segments |

**Why it matters.**

At `:283`, `pair_result.first` gates the proton skip
`particle_type == 2212 && ((medium_dQ_dx_1 > 1.45 && pair_result.first <= 3) || …)`.
Counting only shower segments makes the count strictly smaller, so `<= 3` holds
far more often, so **more proton-PID'd segments are skipped** and fewer shower
candidates are built at the main vertex.

At `:2186`, `daughter_length` is the accumulated length fed to `examine_showers`'
cuts.  Including shower segments makes it strictly larger.

**The B.1 history matters and should be stated plainly.**  Before the review
doc's B.1 was applied the call was `_showers(graph, main_vertex, sg, false)`,
which counts iff `flag_shower || !false` — i.e. **always** — which is exactly
what the prototype's `_tracks(…, true)` does.  **The pre-fix code was
semantically identical to the prototype at this site; applying B.1 introduced
the divergence.**  The review doc reached its recommendation by reading the
prototype's call as `calculate_num_daughter_tracks` (which it quotes correctly)
and then prescribing a change to `calculate_num_daughter_showers` (a different
function) — the two were treated as one.

**This one is not an interpretive tie, and saying so is not picking a fix.**
§5 rule 4 asks for both readings when a divergence is genuinely ambiguous.  Here
it is not.  A "deliberate redesign" reading would have to name a single intent
that yields *shower-flagged only* at `:283` and *everything* at `:2186`, against
a prototype that wants *everything* and *tracks only* respectively.  No such
intent exists: the two sites diverge in opposite directions, and `:2186`'s
explicit `false` makes `_showers` count everything — neither the prototype's
semantics nor a coherent "count showers" choice.  **Read this as a port defect
at both sites.**  What remains the owner's call is *what to change them to* —
restore the prototype's `_tracks` calls, or keep the current behaviour
deliberately and record it in `porting_dictionary.md` — not *whether* they
diverge.

One asymmetry that is not a matter of taste and survives either resolution:
`_tracks` filters `length > length_cut` (zero-length segments excluded),
`_showers` does not.

---

### P2 — "whose PDG" is read: five substitutions, in both directions

`Shower::get_particle_type()` returns `data.particle_type`, written by
`set_particle_type` or by `update_particle_type`'s majority vote over all
sub-segments.  `shower->start_segment()->…->pdg()` is the start segment's own
PID.  These are different quantities whenever a shower has more than one
segment.

| prototype | reads | toolkit | reads | verdict |
|---|---|---|---|---|
| `:1716` in_main_cluster, long-muon→EM loop | **start segment**, `!= 13` | `:170` | **shower**, `abs(...) != 13` | ✗ object **and** `abs` |
| `:387`, `:394` examine_merge_showers | **start segment**, `== 13` | `:1238` | **shower**, `== 13` | ✗ object |
| `:497` id_pi0_without_vertex (main-vertex loop) | **start segment**, `== 13` | `:2892` | **shower**, `== 13` | ✗ object |
| `:511` id_pi0_without_vertex (other-vertex loop) | **start segment**, `== 13` | `:2908` | **shower**, `== 13` | ✗ object |
| `:1800` from_main_cluster, long-muon skip | **shower**, `fabs(...) == 13` | `:514` | **start segment**, `abs(...) == 13` | ✗ object, **inverted** |

Two sites *do* match — `:1875` ↔ `:609` (both start segment) and `:1411` ↔
`NeutrinoEnergyReco.cxx:305` (both shower) — which is what makes the other five
look like slips rather than a convention.

**Why it matters.**  Every one of these is a muon veto.  At `:170` the toolkit
gates the long-muon→EM conversion on the *merged* shower's majority PDG; the
prototype gates it on the start segment.  At `:514` the toolkit does the
opposite.  A shower whose start segment is a muon but whose majority is
electron (or vice versa) takes different branches in the two trees, in
different directions at different sites.

The `abs` at `:170` and `:2174` compounds it: prototype `!= 13` / `== 13` is
exact, so PDG `−13` takes the other branch there.  (The review doc noted the
`:2174` case as "low risk"; it did not note the object substitution at any
site.)

---

### P3 — `acc_segment_id` is a by-value `int` seeded at 0, so π⁰ ids collide

In the prototype `acc_segment_id` is a **`NeutrinoID` member**
(`NeutrinoID.h:1982`) — the global segment-id allocator, incremented by every
`break_segment_at_point` / new-`ProtoSegment` site across
`NeutrinoID_proto_vertex.h`, `NeutrinoID_examine_structure.h`,
`NeutrinoID_improve_vertex.h` (≈20 sites).  Both π⁰ finders draw from it:

```
shower_clustering.h:933  int pio_id = acc_segment_id; acc_segment_id ++;   // id_pi0_with_vertex
shower_clustering.h:688  int pio_id = acc_segment_id; acc_segment_id ++;   // id_pi0_without_vertex
```

so π⁰ ids are (i) monotone across the two finders and (ii) disjoint from every
segment id ever allocated.

In the toolkit, `TaggerCheckNeutrino.cxx:549` declares a **local**
`int acc_segment_id = 0;` and passes it **by value** into
`shower_clustering_with_nv` (`:716`) and, separately, into `ssm_tagger`
(`:776`).  `shower_clustering_with_nv` then passes the same by-value parameter
to both π⁰ finders:

```
:3285  id_pi0_with_vertex(acc_segment_id, …)      → :2712  int pio_id = acc_segment_id++;
:3294  id_pi0_without_vertex(acc_segment_id, …)   → :3114  int pio_id = acc_segment_id;
                                                     :3115  acc_segment_id++;
```

**Two consequences, both unconditional:**

1. **π⁰ id collision.** `id_pi0_without_vertex` receives the *same* starting
   value that `id_pi0_with_vertex` received, because the increments happened in
   a copy.  If both finders fire, both allocate `pio_id = 0`, and
   `map_shower_pio_id`, `map_pio_id_showers` and `map_pio_id_mass` are keyed on
   it.  `map_pio_id_mass[0]` is **overwritten** by the second finder;
   `map_pio_id_showers[0]` accumulates **four** showers under one π⁰ id.
2. **The allocator link is severed.** π⁰ ids start at 0 and therefore alias
   real segment ids.  In the prototype that could never happen by construction.

**Reach — the source question and the runtime question, kept apart.**

*Can both finders fire in one event?*  Answerable from source, and the answer is
**yes**.  `id_pi0_with_vertex`'s only entry gate is `if (!main_vertex) return;`
(`:2441`) — it places no constraint on the main vertex's degree.
`id_pi0_without_vertex` requires the main vertex to have ≤2 segments with one of
them in a shower (`:2803`, `:2809-2815`).  Those gates are compatible.  There
*is* a partial interlock — `:2838` returns if a shower **at the main vertex** is
already in `pi0_showers` — but it does not cover a π⁰ that `id_pi0_with_vertex`
found at a non-main vertex, which is most of what that function looks for.

*How often do both fire?*  **Not measured** — §7 loose end 2.  So consequence 1
is a reachable corruption, not a demonstrated one; when it does happen the π⁰
block of `T_kine` is silently wrong rather than merely different.

Consequence 2's blast radius depends on whether anything downstream compares a
`pio_id` against a segment id.  I did not find such a comparison inside
`clus/`, so I record this as a lost invariant rather than a confirmed bug.

---

### P4 — the prototype's `max_length` shadowing bug, silently fixed

`shower_clustering_connecting_to_main_vertex` declares `max_length` **twice**:

```
shower_clustering.h:133   WCPPID::WCShower* max_shower = 0;
shower_clustering.h:134   double max_length = 0;              // outer — intended accumulator
   for (each segment at main_vertex) {
shower_clustering.h:154       double max_length = 0;          // inner — SHADOWS the outer
        …
shower_clustering.h:201       if (max_length < length) { max_length = length; max_sg = sg1; }   // inner
        …
shower_clustering.h:222       if (shower->get_total_length() > max_length) {                     // inner!
shower_clustering.h:223         max_shower = shower;
shower_clustering.h:224         max_length = shower->get_total_length();
        }
   }
```

The `max_shower` selection at `:222` compares the shower's **total** length
against the inner `max_length`, which at that point holds the longest
**single segment** *of that same shower* — and the assignment at `:224` writes
the inner variable, which is re-initialised to 0 on the next iteration.  The
outer `max_length` declared at `:134` is **never read or written**.

So in the prototype the test at `:222` is "is this shower longer than its own
longest segment", which is true for essentially every multi-segment shower, and
`max_shower` ends up being **the last qualifying shower in pointer-map order**,
not the longest.

The toolkit (`:407-411`) uses a single `max_length` across the loop, compares
`shower_len > max_length` correctly, and adds a start-segment-id tie-break.
That is what the code obviously *meant* — and it changes which shower gets
converted to an EM shower, has its start segment forced to PDG 11, and causes
conflicting showers to be deleted.

**Both readings (§5 rule 4).**  (a) A prototype bug the toolkit fixed, in which
case the fix should be recorded in `porting_dictionary.md` alongside the three
prototype bugs the review doc already lists.  (b) The prototype's behaviour is
what tuned the uBooNE working point, in which case the toolkit is off-model
here.  I do not pick.  This one is **not** in the review doc's "Prototype Bugs
Fixed" table.

---

### P5 — a toolkit-only orphan-segment rescue block, and the surviving L.1 hardcode

`shower_clustering_with_nv_from_main_cluster` ends with a block (`:721-751`)
that has **no prototype counterpart**:

```
:721  // After the sweep, some clusters may be partially claimed: …
:728  std::map<Facade::Cluster*, ShowerPtr, ClusterPtrCmp> cluster_to_shower;
      …
:738  auto it = cluster_to_shower.find(seg1->cluster());
:740  it->second->add_segment(seg1, true);
:742  if (seg1->has_particle_info() && seg1->particle_info()) {
:743      seg1->particle_info()->set_pdg(11);
:743      seg1->particle_info()->set_mass(0.511 * units::MeV);
      }
```

The prototype's function ends at `update_shower_maps()` (`:1937`).  This block
adds every unclaimed segment of a partially-claimed cluster to whichever shower
already owns a sibling segment, **and force-sets it to electron**.  The
motivating comment is a real observation (downstream
`shower_clustering_in_other_clusters` skips whole clusters already in
`used_shower_clusters`), but the remedy is a toolkit-only behaviour.

It is also where **L.1 now lives**: `0.511 * units::MeV` instead of
`particle_data->get_particle_mass(11)`.  The review doc's L.1 anchor (`:203`)
was fixed; this site was not, and the review doc's verdict for this whole
function is "Correct.  No bugs."

`cluster_to_shower.emplace` keeps the **first** shower per cluster in
`map_segment_in_shower` order.  That map is `ShowerSegmentMap =
std::map<SegmentPtr, ShowerPtr, SegmentIndexCmp>` (`PRShower.h:223`), so the
order is graph-index-stable — deterministic, but "first by segment index" is an
arbitrary choice with no prototype basis.

---

### P6 — `get_flag_shower()`'s `abs(pdg)==11` term is dropped at two sites, and the porting dictionary documents the incomplete mapping

Prototype (`ProtoSegment.cxx:1305-1312`):

```
bool get_flag_shower(){ return flag_shower_trajectory || flag_shower_topology || get_flag_shower_dQdx(); }
bool get_flag_shower_dQdx(){ if (fabs(particle_type)==11) return true; return false; }
```

Seven toolkit sites open-code this.  Five include the PDG term
(`:102-105`, `:373-376`, `:1379-1382`, `:1867-1870`, `:2916-2923`).  **Two do
not:**

* `:788` in `shower_clustering_with_nv_from_vertices` — against prototype
  `:1012`, `seg->get_flag_shower()==1 || seg->get_particle_type()==0 || …`.
  An electron-PID'd segment with neither topology nor trajectory flag is
  included by the prototype, excluded by the toolkit.
* `:1882` in `examine_shower_1`.  The review doc argues this is safe "because
  the relevant segments have already been filtered to `dir_weak()==true`, and
  segments with `pdg==11` would already have `kShowerTrajectory` set".  The
  second half of that argument is an assumption about which flags co-occur, not
  a source fact — `set_particle_type(11)` is called at several sites in this
  very file (`:206`, `:423`, `:743`, `:1461`) **without** setting either flag.
  Three of those four writes happen *before* `examine_shower_1` runs.

**The `porting_dictionary.md` entry is itself incomplete.**  Line 222 maps
`get_flag_shower()` → `seg->flags_any(kShowerTrajectory | kShowerTopology)` with
the note "Split into two flags" — the `abs(pdg)==11` term is simply absent.  So
unlike pr/29–pr/32, this divergence class *is* documented — and the
documentation is wrong.  That is worth more than the two sites: any future port
following the dictionary reproduces the omission.

---

### P7 — `collect_charge_maps` moved from before sub-call 1 to after sub-call 4

Prototype `NeutrinoID.cxx:241` calls `collect_2D_charges()` immediately before
`shower_clustering_with_nv()` at `:246` — i.e. **before all four clustering
sub-calls**.  The toolkit calls `collect_charge_maps(track_fitter)` **inside**
the entry point at `:3242`, after sub-calls 1–4, with an in-code rationale:
"after track fitting (`shower_clustering_with_nv_from_vertices`) has populated
the underlying charge data".

**What is confirmed.**  The first consumer inside the stage is
`examine_merge_showers` (toolkit `:1280`; prototype `:412`), which is sub-call 6
in both trees — so in both trees the maps are populated before first use, and
the lazy re-collection guards at `NeutrinoEnergyReco.cxx:242` and `:293` mean a
standalone call cannot read empty maps.

**What is not.**  Whether the *contents* differ.  The two implementations are
not the same function: the prototype computes a time/channel bounding box from
`main_cluster` + `other_clusters`, queries
`ct_point_cloud->get_overlap_good_ch_charge`, then overlays dead channels via
`fill_2d_charge_dead_chs`.  The toolkit delegates wholesale to
`TrackFitting::collect_2D_charge` (`TrackFitting.cxx:933`), which was not read
here.  If that delegate reads any state mutated by sub-calls 1–4, the two trees
see different charge maps.  The in-code comment asserts it does; I did not
verify it.  §7 loose end 3.

---

### P8 — `fabs(pdg == 13)`: a prototype bug whose fix widens the branch to −13

`shower_clustering.h:1688`: `if (fabs(curr_sg->get_particle_type()==13)){`.  The
`==13` is *inside* `fabs`, so the expression is `fabs(bool)` — 1.0 or 0.0 —
truthy iff `pdg == 13` exactly.  The toolkit (`:116-118`) writes
`std::abs(pdg) == 13`, which also fires for `pdg == −13`.

Already in the review doc's "Prototype Bugs Fixed" table, and it is plainly a
bug.  Carried here because it is the same *class* as P2's `abs` substitutions
and because its reach depends on whether any segment in this stage ever carries
PDG −13 — which I did not measure.

---

### P9 — "which end" is decided by distance, not index equality, at three sites

The prototype decides whether a segment starts or ends at a vertex by **exact
`wcpt().index` equality**.  The toolkit uses **nearest-fit-point distance** at
every site:

| purpose | prototype | toolkit |
|---|---|---|
| `end_vertex` for `flag_good_track`, `connecting_to_main_vertex` | `shower_clustering.h:167-179`, `wcpt_vec().back()/front().index == vtx->wcpt().index` | `:348-363`, `\|fits().back()/front().point − vtx_pt\|`, nearer wins |
| `end_vertex` for `flag_good_track`, `examine_shower_1` | `em_shower.h:479-493`, same index equality | `:1844-1857`, same distance form |
| `flag_start` at π⁰ vertices | `shower_clustering.h:974-978`, same index equality | `:2757-2761`, `dist_front < dist_back` |

Three consequences:

1. **The prototype's `flag_start` at `:974-978` is uninitialised-on-fallthrough**
   — `bool flag_start;` with `if`/`else if` and no `else`.  Reading it is UB
   when neither endpoint index matches.  The toolkit's form is always defined.
   This is **the same finding as pr/32 P6**, recurring in a second stage.
2. **The toolkit reads `fits()`, the prototype reads `wcpts()`.**  Doc pr/28's
   central lesson was the inverse (MyFCN read the skeleton where it should have
   read `fits()`); here the trees differ in the other direction.  They agree
   when the fit is close to the skeleton and can disagree otherwise.
3. **Empty `fits()` silently disables the whole branch.**  At `:348-363` and
   `:1844-1857` the
   toolkit requires `!sg1->fits().empty() && v1 && v2`; if not, `end_vertex`
   stays null and `flag_good_track` is never evaluated for that segment.  The
   prototype has no such escape.

---

### P10 — degenerate-path guards that change behaviour, not just safety

The toolkit adds null guards the prototype lacks.  Most are pure safety, but
these four change the result on a reachable input:

| site | toolkit | prototype |
|---|---|---|
| `:41` `update_shower_maps` | skips a null `start_vtx` | `map_vertex_to_shower[nullptr].insert(shower)` — creates a real entry that later lookups can hit |
| `:205-207` in_main_cluster long-muon→EM | `set_pdg(11)` only `if (has_particle_info())` | `:1746` writes unconditionally |
| `:422-424` connecting_to_main_vertex | `set_pdg(11)` only `if (has_particle_info())` | `:238` writes unconditionally |
| `:171` in_main_cluster | `if (!shower->start_segment()) continue;` | `:1716` dereferences it |

A segment that reaches the conversion sites without `ParticleInfo` is left
unconverted by the toolkit and converted to an electron by the prototype.
Whether that state is reachable at this point in the chain was not measured.

---

### P11 — argmax tie-breaks added where the prototype has none

Five argmaxes in this file gained a tie-break the prototype does not have:

| what | toolkit | prototype |
|---|---|---|
| longest muon segment in a long-muon shower | `:190`, `<` then lower `sg1->id()` | `:1730`, plain `>`; first wins in **pointer-map order** |
| longest segment in a candidate shower | `:328`, lower `id()` | `:201`, plain `<` |
| `max_shower` in connecting_to_main_vertex | `:407-411`, lower start-segment `id()` | `:222`, plain `>` — and see P4 |
| `max_length_segment` in from_main_cluster | `:494-500`, index-ordered scan | `:1782-1787`, pointer-map scan |
| π⁰ best pair | `:2688-2703` under `shower_pair_cmp` | `:917-927` under pointer-keyed `std::map` |

Each makes the toolkit deterministic — an unambiguous improvement — **and**
selects a different winner from the prototype whenever the maximum is tied.
Cross-references pr/32 P5/P8.  Exact ties in a `double` length are rare; ties in
`get_kine_charge()` less so, since showers built from the same segment set
produce bit-equal charges.

---

### P12 — the residual pointer comparison in `shower_less`

`:2829`: after ordering by start-segment graph index, the comparator falls
through to `return a.get() < b.get(); // same-index fallback: stable within a run`.
Two showers reach it only if they share a start segment (or both have none).
The comment is accurate — it *is* stable within a run — but it is stable across
runs only if the allocator is, which ASLR does not guarantee (M4).  It orders
`map_shower_ray` and `map_shower_pair_mass_point`, i.e. π⁰ pairing.  This is
the one place in the file where determinism rests on an address.

---

### P13 — toolkit-only termination guards

`:628-644` and `:1193` add a "no progress in a pass ⇒ stop" guard to loops the
prototype runs as `while (flag_continue)`.  The rationale in the comment is
sound (`TrajectoryView::add_segment` no-ops on an invalid descriptor, so
`flag_continue` could be set forever).  Behaviour differs from the prototype
only in the case where the prototype would hang — but it also emits a
`SPDLOG_LOGGER_WARN` and *truncates* the loop, so if the guard ever fires on a
non-pathological input the sweep ends early with no prototype counterpart.
Recorded, not judged.

---

### P14 — B.6: electron-forcing in `shower_clustering_in_other_clusters` sub-pass 1

`:1453-1467` forces PDG 0, or a short (< 40 cm) direction-weak muon, to
electron before the majority vote.  The prototype has this in sub-pass 2 only.
The review doc documents it as "an intentional improvement for robustness" and
did not remove it.  Carried forward unchanged so the list of unconditional
behaviour differences is complete; the owner has effectively already ruled on
this one.

---

## §4 The SBND operating point

Unlike pr/31 (where the knobs were the story) and like pr/32, the C++ defaults
in this stage are bit-faithful to the prototype.  The divergence is **config**.

| quantity | prototype literal | C++ default | SBND value | where |
|---|---|---|---|---|
| `m_mip_dqdx_median` (substitutes `43e3/units::cm`) | `43e3` | `43000/units::cm` (`NeutrinoPatternBase.h:114`) | **48000** | `cfg/pgrapher/experiment/sbnd/clus.jsonnet:889` |
| `m_mip_dqdx` (feeds `segment_cal_4mom`, `update_particle_type`) | — | `50000/units::cm` | **56000** | `cfg/.../sbnd/wct-pr-perevt.jsonnet:115` |

`m_mip_dqdx_median` appears in this stage at `:290` (the P1 proton skip),
`:344` (`flag_good_track`'s `medium_dQ_dx_norm > 2.5`), `:527` (the shower-dir
gate's `> ×1.5`) and inside `examine_shower_1`.  SBND's 48000 vs the prototype's
43000 makes every normalised dQ/dx **~10 % smaller**, so every one of those
thresholds is effectively ~10 % harder to pass than in the prototype.

This is deliberate SBND tuning, not a port defect.  Stated so the §3 rows are
not read as the only reasons SBND differs from uBooNE.

---

## §5 Looks like a divergence and is not

1. **`update_particle_type` at `:149` has no prototype counterpart in
   `shower_clustering_with_nv_in_main_cluster` — and that is a real toolkit
   addition, not an artefact.**  The prototype calls `update_particle_type` at
   six sites (`shower_clustering.h:410, 1324, 1531, 1555`; `em_shower.h:260,
   536`), none of them in this function.  The review doc records it as an
   "intentional enhancement".  Listed here rather than as a P-row only because
   it is already documented; it *is* an unconditional behaviour difference.
2. **`shower_determing_in_main_cluster` (`shower_clustering.h:15`) is not
   missing.**  It is stage 3's driver, physically filed in the shower file.
   Doc pr/31 audited its 12 calls.  Same trap as pr/31 GOTCHA 2 and pr/32
   GOTCHA 2: **discriminate by content, not by which file a function lives in.**
3. **`get_start_end_vertices` (`:1431`) is not missing** — it is
   `find_vertices(graph, sg)`, whose ordering is doc pr/30's subject.
4. **`ClusterPtrSet` is not pointer-ordered.**  `PRShower.h:227-232` gives it a
   `ClusterPtrCmp` that compares `get_cluster_id()`.  Same for
   `ClusterVertexMap`.  All four shower maps (`ShowerVertexMap`,
   `ShowerSegmentMap`, `VertexShowerSetMap`, `ShowerIntMap`) use index
   comparators (`PRShower.h:221-225`).
5. **`used_shower_clusters` changed from `set<int>` to `set<Cluster*>`** and is
   only ever `.find()`-ed (`:1346`, `:1508`), never iterated for a decision.
6. **`examine_shower_1` is not dead code.**  Called from `examine_showers` at
   `:2433`, exactly as the prototype calls it from `em_shower.h:333`.
7. **`n_showers` is dead in the prototype** (`shower_clustering.h:198`,
   `em_shower.h:508`) — incremented, never read.  The toolkit dropped it.  Same
   shape as pr/30 §5.1's never-incremented `count` and pr/29's commented-out
   kruskal.  Do not "restore" it.
8. **The angle-fold at `:568` is not a changed test** — §2.7.
9. **The degree/radian round-trip at `:705-706` is not a changed metric** —
   §2.6.
10. **The `unordered_set<ShowerPtr> claimed` in `examine_merge_showers` is
    iterated** (`:1285`), contradicting the review doc's "no iteration-order
    dependence" — but the loop body is `showers.erase(shower)` on an
    index-ordered set, whose result is order-independent.  Benign as claimed,
    for a different reason than the one given.
11. **The prototype's `ProtoVertex* vtx;` at `:903` is uninitialised** but never
    read: `mass_diff` stays `1e9` when no pair qualifies, so the guarded block
    is skipped.  Safe by construction, not by intent.  The toolkit initialises
    it to `nullptr`.
12. **`shower->add_segment(seg1, true)` vs
    `add_segment(seg1, map_segment_vertices)`** is the maps-eliminated API
    change recorded in `porting_dictionary.md:258`, not a semantic change.
13. **The prototype's `delete shower` calls have no toolkit counterpart** —
    `ShowerPtr` is a `shared_ptr`; rejected showers are simply never inserted.
    §2.3 shows the side effect that *does* matter is preserved.

---

## §6 Determinism

**Cleared.**  All four shower maps and both shower sets carry index
comparators.  `used_shower_clusters` and `ClusterVertexMap` order by cluster id.
Every graph traversal in this file uses `ordered_edges` / `sorted_out_edges` /
`ordered_nodes`.  `map_shower_dir` and `map_shower_max_sg` are pointer-keyed
`std::map`s but are either converted to an id-sorted vector before use
(`:651-673`) or have an order-independent loop body (`:418-441`).

**Note on the read revision.**  pr/30–pr/32 read `4f2e7303`; this audit reads
`f07c0299`, which includes `c05bc5f7`'s sweep of `boost::edges` /
`boost::vertices` / `graph_nodes`.  I confirmed **zero raw
`boost::edges`/`out_edges`/`vertices` calls remain in
`NeutrinoShowerClustering.cxx`** — so the three raw sites pr/31 GOTCHA 8 left
open do not recur here.

**Not cleared.**

* **P12** — `shower_less`'s `a.get() < b.get()` fallback (`:2831`) is a live
  address comparison feeding π⁰ pairing.
* **P11** — the five added tie-breaks make the toolkit deterministic; they do
  not make it agree with the prototype.
* `std::set<ShowerPtr> del_showers` (`:270`) and
  `std::map<ShowerPtr, SegmentPtr> map_shower_max_sg` (`:271`) use the default
  `std::less<shared_ptr>`, i.e. address order.  Judged benign above; **not
  proven** — no repeat-run identity check was executed.

**Verdict: not proven.**  No `repeat_check.sh`-style N-run identity was run for
this stage.  P12 is a concrete mechanism, not a hypothetical.

---

## §7 Loose ends

1. **P1's reach is unmeasured.**  How often does the proton skip at `:283`
   actually flip?  It needs a main-vertex segment with PDG 2212 and
   `medium_dQ_dx_1 > 1.45`.  A count over the valfast manifest would size it.
2. **P3's collision rate is unmeasured.**  How often do both π⁰ finders fire in
   the same event?  One `SPDLOG` line in each finder would answer it.
3. **P7 is half-verified.**  `TrackFitting::collect_2D_charge`
   (`TrackFitting.cxx:933`) was not read.  If it depends only on blob geometry
   and channel status, the relocation is free; if it reads anything sub-calls
   1–4 mutate, the two trees see different charge maps and every downstream
   `kine_charge` differs.
4. **P10's degenerate inputs are unmeasured.**  Does a segment without
   `ParticleInfo` ever reach `:138` or `:406` at this point in the chain?
5. **`examine_shower_1` (450 lines) and `examine_showers` (357) were not read
   line-by-line for arithmetic.**  Control flow, all constants, the daughter-
   count call and the flag/PDG patterns were checked; a numeric divergence
   inside their case ladders could still hide.  This is the same gap pr/32
   GOTCHA 12 flagged for `examine_direction`, narrowed but not closed.
6. **The constant-histogram helper's old blind spot is fixed; a new one is
   known.**  `consts.py` now tokenises numbers independently of the units suffix
   and records comparison operators separately, closing pr/32 GOTCHA 8.  It
   still cannot see a constant that has been replaced by a *member* on one side
   (that is what made `43e3` show as "prototype-only" in four pairs), so every
   "only in B" hit still needs a human read.  Its comparison-operator stream is
   **not usable** for this file pair: the prototype's iterator loops
   (`it != map.end()`) inflate `!=` counts against the toolkit's range-fors.
7. **`porting_dictionary.md` has no section for this stage** — only the two
   incidental entries at lines 217-229 (`ProtoSegment`) and 246-264
   (`WCShower`), one of which (line 222, `get_flag_shower`) is **wrong**: see
   P6.  Sixth audit in a row with no stage section.

---

## §8 Summary

| # | divergence | site (toolkit @ `f07c0299`) | prototype | class |
|---|---|---|---|---|
| **P1** | wrong daughter-count function, both sites, opposite errors; B.1 introduced one | `:283`, `:2186` | `shower_clustering.h:140`, `em_shower.h:17` | logic |
| **P2** | "whose PDG" — 5 substitutions, both directions | `:170 :514 :1238 :2892 :2908` | `:1716 :1800 :387 :497 :511` | logic |
| **P3** | `acc_segment_id` by value from 0 ⇒ π⁰ id collision, allocator link severed | `TaggerCheckNeutrino.cxx:549`; `:2712`, `:3114` | `NeutrinoID.h:1982`; `:933`, `:688` | state |
| **P4** | prototype `max_length` shadowing bug silently fixed | `:407-411` | `:134` / `:154` / `:222` | logic |
| **P5** | toolkit-only orphan rescue + forced PDG 11 + L.1 hardcode | `:721-751`, `:743` | *(none)* | added |
| **P6** | `abs(pdg)==11` dropped from `get_flag_shower`; dictionary entry wrong | `:788`, `:1882`; `porting_dictionary.md:222` | `ProtoSegment.cxx:1305` | logic |
| **P7** | `collect_charge_maps` moved after sub-call 4 | `:3242` | `NeutrinoID.cxx:241` | order |
| **P8** | `fabs(pdg==13)` fixed ⇒ branch widens to −13 | `:116` | `:1688` | logic |
| **P9** | "which end": index equality → nearest-fit distance, 3 sites; empty `fits()` disables | `:348-363`, `:1844-1857`, `:2757-2761` | `sc.h:167-179`, `em.h:479-493`, `sc.h:974-978` | method |
| **P10** | 4 degenerate guards change the result | `:41 :205 :422 :171` | `:1397 :1746 :238 :1716` | degenerate |
| **P11** | 5 argmax tie-breaks added | `:190 :328 :410 :494 :2696` | plain `>`/`<`, pointer order | determinism |
| **P12** | `shower_less` pointer fallback feeds π⁰ pairing | `:2829` | n/a | determinism |
| **P13** | toolkit-only loop termination guards | `:628`, `:1193` | *(none)* | added |
| **P14** | B.6 electron-forcing in sub-pass 1 (already documented) | `:1453-1467` | *(none)* | added |

---

## §9 What is NOT claimed

* **No event was run.**  Every "this changes the output" above is an argument
  from source, not an observed diff.  No A/B gate, no manifest, no label.
* **P1's and P2's mechanisms are confirmed; their frequency is not.**  Both are
  read directly out of the two trees and are not in dispute.  How often they
  flip a decision on real SBND events is unmeasured — §7 loose ends 1 and 4.
* **P3's second consequence is a lost invariant, not a demonstrated bug.**  I
  did not find any `clus/` code comparing a `pio_id` to a segment id.  Absence
  of a found comparison is not proof there is none.
* **P7 is half-verified** — §7 loose end 3.  The in-code comment's claim that
  the charge data is only populated after track fitting was taken at face value
  for the *rationale*; the delegate was not read.
* **`examine_shower_1` and `examine_showers` were not read end to end for
  arithmetic** — §7 loose end 5.  Their contribution to §8 rests on the
  patterns checked across the whole file, not on an exhaustive read.
* **`PRShower.cxx` ↔ `WCShower.cxx` (~2000 lines) is unread** except
  `fill_maps` and `complete_structure_with_start_segment`.  §2.2 and §2.3 are
  therefore about those two methods only.
* **§6's determinism verdict is "not proven", deliberately.**  No repeat-run
  identity check was executed for this stage.
* **The in-tree review doc was re-verified, not trusted.**  §3.0's seven rows
  were each re-read at HEAD.  Two of that doc's conclusions did not survive:
  B.1's prescription rests on conflating two different functions (P1), and its
  "Correct.  No bugs." verdict for `shower_clustering_with_nv_from_main_cluster`
  misses the toolkit-only block at `:721-751` (P5).  Its line anchors are all
  stale — it describes a 3310-line file, HEAD's is 3384.
* **P4 and P6 are presented with both readings and neither is picked** (§5 rule
  4 / M15).  P4's second reading is legitimate — the prototype's *runtime*
  behaviour, bug or not, is what produced the uBooNE reference results.
  **P1 is deliberately not given a second reading**: no single design intent
  explains both of its sites, so manufacturing an interpretive tie there would
  misrepresent the evidence.  What P1 leaves to the owner is the choice of fix,
  not the question of whether it diverges.

---

## §10 Owner filter, 2026-08-04 — fourteen divergences to five

**The ask.** Skip anything that is an improvement over the prototype; keep only
what is a **bug** or **missing from the port**.  Same rule the owner applied to
doc pr/32 §10 (twelve → four).

**Re-verified at committed HEAD `407c5ba9`**, not at the `f07c0299` the audit was
written against.  **Every §3 and §8 anchor in this doc is now stale by up to
+19 lines** — `397b1517` landed on this file after the read (3384 → 3403 lines).
New anchors are given per finding below; §3's are left as written so the two
revisions stay distinguishable.  Still **no code changed** in this section.

### §10.1 The filter

| # | verdict | why |
|---|---|---|
| **P1** | **KEEP — F1** | wrong callee at both sites; a port defect, not a redesign |
| **P2** | **KEEP — F2** | five object substitutions in a muon veto, in both directions |
| **P3** | **KEEP — F3** | by-value allocator ⇒ π⁰ id collision with a **live** downstream consumer |
| P4 | drop | fixes a prototype shadowing bug — an improvement |
| P5 | drop | toolkit-only addition with a stated rationale; neither bug nor gap |
| P6 | **KEEP — F4** | but **one live site, not two** — §10.8 correction 3 |
| P7 | **drop — RESOLVED** | §7 loose end 3 closed; the relocation moves *toward* the prototype |
| P8 | drop | fixes the prototype's `fabs(pdg==13)` bug — an improvement |
| P9 | drop | id→position, the class the owner pre-cleared in pr/32 |
| P10 | drop | null-safety guards; degenerate reachability unmeasured |
| P11 | drop | determinism tie-breaks — an improvement (and see correction 7) |
| **P12** | **KEEP — F5**, different class | violates *our own* §2 determinism rule; prototype is `n/a` |
| P13 | drop | anti-hang guards — an improvement |
| P14 | drop | B.6; the owner has already ruled |

Five kept.  F1–F4 are prototype-fidelity defects.  **F5 is not** — it is a
house-rule violation with no prototype counterpart, and it is listed separately
so that distinction is not smuggled past the owner.

Five findings, **eight proposed knobs** — F1 splits two ways and F2 three ways,
because a knob that moves two sites in opposite directions cannot tell the gate
which site moved a decision.  §10.10 records the three amendments that produced
that count.

---

### §10.2 F1 = P1 — restore `calculate_num_daughter_tracks` at both sites

**Confirmed at HEAD, and the confirmation narrows the defect.**  Both toolkit
implementations (`NeutrinoTrackShowerSep.cxx:178` and `:235`) are *faithful*
ports of the prototype's two functions — both carry the full
`get_flag_shower()` including the `abs(pdg)==11` term, both use the same BFS
frontier, and `_tracks` carries the `length > length_cut` filter that `_showers`
lacks.  **So P1 is purely a wrong-callee bug.  Nothing about the callees needs
to change.**

| site | HEAD anchor | now calls | counts | prototype | counts |
|---|---|---|---|---|---|
| `shower_clustering_connecting_to_main_vertex` | **`:289`** | `_showers(graph, main_vertex, sg)` (flag defaults `true`) | shower-flagged only | `_tracks(main_vertex, sg, true)` — `shower_clustering.h:140` | **everything**, `length > 0` |
| `examine_showers` | **`:2205`** | `_showers(graph, main_vertex, sg, false).second` | **everything**, no length filter | `_tracks(main_vertex, sg, false).second` — `em_shower.h:17` | tracks only, `length > 0` |

**Solution — two knobs, not one.**  The two sites err in *opposite* directions,
so a single knob makes the A/B unable to attribute which site moved a decision:

* `daughter_count_proto_main_vertex` → `:289` becomes
  `calculate_num_daughter_tracks(graph, main_vertex, sg, /*count_shower=*/true, 0)`;
* `daughter_count_proto_examine_showers` → `:2205` becomes
  `calculate_num_daughter_tracks(graph, main_vertex, sg, /*count_shower=*/false, 0).second`.

Both C++ default `false`.  Ship a counter per site reporting how often the
restored value differs from the current one — `pair_result.first` at `:289`
gates the proton skip through `<= 3`, so a count that never crosses the
threshold is the cheap way to bound the reach that §7 loose end 1 left open.

**Not an interpretive tie.**  §3 already argued this and it survives
re-verification: no single design intent yields *shower-flagged only* at one
site and *everything* at the other against a prototype that wants *everything*
and *tracks only* respectively.  What is the owner's call is whether to restore
or to keep and document — not whether it diverges.

---

### §10.3 F2 = P2 — read the start segment's PDG where the prototype does

Five substitutions, unchanged in count after re-verification, at HEAD anchors
**`:170`, `:525`, `:1247`, `:2911`, `:2927`**.  Four read the *shower's*
majority-vote type where the prototype reads the *start segment's* PID; `:525`
runs the other way.  Every one is a muon veto.

**Solution — three knobs, by the same attribution criterion F1 uses.**  An
earlier draft gave F2 one knob while conceding that `:525` inverts; that
contradicts §10.2's own rule, so it is split.  All C++ default `false`:

| knob | sites | change |
|---|---|---|
| `shower_pdg_from_start_segment` | `:170`, `:1247`, `:2911`, `:2927` | shower type → start segment's PID |
| `shower_pdg_from_shower_type` | `:525` | start segment's PID → shower type (the inverted site) |
| `shower_pdg_exact_muon_test` | `:170`, `:2193` | drop `std::abs`, matching the prototype's exact `!= 13` |

The third knob is what classifies **`:2193`**, which §3's P2 prose mentioned (as
`:2174`) but never placed in its table and which an earlier draft of §10 left
neither kept nor dropped.  It is an `abs`-only divergence with **no** object
substitution: prototype `em_shower.h:10` `get_particle_type() != 13` is exact,
so a segment already in a shower with PDG **−13** is skipped by the prototype
and **processed** by the toolkit.  The toolkit's extra `!sg->has_particle_info()`
term at that site is not a divergence — a segment with no `ParticleInfo` has no
PDG and takes the `continue` in both trees.

`:170` appears in two knobs because both defects sit in one expression; with
`shower_pdg_from_start_segment` on and `shower_pdg_exact_muon_test` off, `:170`
reads the start segment but still tolerates −13, which is neither tree's
behaviour.  Say so in the config comment: for prototype parity at `:170` both
must be on.

**Why it is a slip and not a convention** — and this is sharper than §3 said.
There are now **four** sites where both trees read the shower type, not two:
§10.8 correction 2 adds `:2550` and `:2585`.  So of nine comparable sites the
trees agree on four and disagree on five, and the five disagreements point in
two directions.  A convention would not do that.

---

### §10.4 F3 = P3 — share one π⁰-id allocator between the two finders

**Confirmed, and the blast radius is larger than §3 recorded.**
`TaggerCheckNeutrino.cxx:602` declares `int acc_segment_id = 0` and passes it
**by value** to `shower_clustering_with_nv` (`:769`), which passes the same
by-value parameter to `id_pi0_with_vertex` (`:3304` → `:2731`
`int pio_id = acc_segment_id++`) and to `id_pi0_without_vertex` (`:3313` →
`:3133-3134`).  The second finder therefore receives the value the first
*started* with.  If both fire, both allocate `pio_id = 0`.

§3 called consequence 2 "a lost invariant" because no consumer was found.  There
are two, both live:

* `NeutrinoTaggerNuE.cxx:574`, `:912`, `:1028` — `map_pio_id_showers[pio_id]`
  feeds the nue tagger's π⁰ block (`pio_flag_pio` and the companion-shower list);
* `MultiAlgBlobClustering.cxx:1552-1557` — Bee π⁰ **grouping** is keyed on
  `map_shower_pio_id.at(sh)`, and `pi0_ke` is read from
  `map_pio_id_mass.find(pi0_id)`.

So a collision does not merely lose an invariant: it puts **four** showers into
one π⁰ group and lets the second finder's mass **overwrite** the first's, in
both the tagger and the event display.

**Solution — the collision half only, and `shower_clustering_with_nv` must stay
by value.**  Change the parameter to `int&` on **`id_pi0_with_vertex` and
`id_pi0_without_vertex` only** (`NeutrinoPatternBase.h:747-748`), and have
`shower_clustering_with_nv` bind its own by-value copy to both.  Knob
`pi0_id_shared_allocator`, C++ default `false`: when off, snapshot the copy
before `:3304` and restore it before `:3313`.

**Making `shower_clustering_with_nv` itself take `int&` would break the
knob-off guarantee**, and an earlier draft of this section proposed exactly
that.  `TaggerCheckNeutrino.cxx:602`'s local is passed **twice** — to
`shower_clustering_with_nv` at `:769` *and* to `ssm_tagger` at `:829`, whose
signature (`NeutrinoTaggerSSM.cxx:581`, and `:300` for the block builder) is
already `int&`.  `ssm_tagger` **reads it as a seed**: `int temp_acc =
acc_segment_id` (`:307`), then `temp_acc++` at `:382`, `:390`, `:393`, `:408`
feeding `fill_ssmsp_pseudo_{1,2,3}`, i.e. the pseudo-particle ids written into
`TaggerInfo` and out to the ssmsp tree, before writing back at `:414`.

Today `ssm_tagger` receives **0**, because the by-value copy meant the π⁰
increments never propagated back.  Widening `shower_clustering_with_nv` to a
reference would shift every ssmsp pseudo-particle id by the number of π⁰s found
in the event — **unconditionally, with the knob off**.  Keeping the entry
point by value confines the sharing to the two finders and makes the knob-off
path leak-free by construction rather than by a restore that has to be placed
correctly.

**The second half is not proposed.**  Seeding at 0 rather than at a global
segment-id allocator is a separate divergence, and the toolkit has no global
segment-id allocator to hook — segment ids come from the graph index.  Restoring
the prototype's "π⁰ ids are disjoint from every segment id" invariant would mean
inventing one.  Recorded as a gap; **not** bundled into F3.

---

### §10.5 F4 = P6 — add the `abs(pdg)==11` term at `:797`

**One live site, not two.**  §10.8 correction 3 shows the second site (`:1894`)
feeds `n_showers`, which is `(void)`-cast at **`:1912`** in the toolkit and read
nowhere in the prototype either — provably unobservable in both trees.

The live site is `:797` in `shower_clustering_with_nv_from_vertices`:

```cpp
bool is_shower = seg->flags_any(kShowerTrajectory) || seg->flags_any(kShowerTopology);
```

against prototype `:1011` / `:1022`, where `get_flag_shower()` also returns true
for `fabs(particle_type)==11`.  It is read **twice**, and both reads diverge:

* `:803` — an electron-PID'd segment with neither flag is added to `acc_length`
  by the prototype, skipped by the toolkit;
* `:815` — `particle_type != 11 && !is_shower` adds to `acc_length1`.  **The
  prototype excludes PDG −11 here and the toolkit does not**, because the
  prototype's exclusion runs through `fabs(...)==11` inside
  `get_flag_shower_dQdx()` while the toolkit's explicit test is `!= 11`.  §3 did
  not name this sub-case.

`acc_length` and `acc_length1` are compared at `:820`
(`acc_length >= acc_length1`), which decides whether the cluster gets a centre
point at all — so both reads feed one gate, in the same direction.

**Solution.** One line, one knob `shower_flag_pdg_electron`, C++ default
`false`: append
`|| (seg->has_particle_info() && std::abs(seg->particle_info()->pdg()) == 11)`
to `:797`.  That fixes `:803` and `:815` together.  Leave `:1894` alone and
record it as dead.

**And fix the dictionary, unconditionally.**  `porting_dictionary.md:222` maps
`get_flag_shower()` → `flags_any(kShowerTrajectory | kShowerTopology)` with the
`abs(pdg)==11` term simply absent.  That is a documentation bug, not a behaviour
change, and it is what makes this class recur: five sites in this file already
open-code the term correctly (`:105`, `:385`, `:1391`, `:1882`, `:2942`), so the
dictionary is out of step with the code as well as with the prototype.  Correct
it whether or not F4 ships.

---

### §10.6 F5 = P12 — `shower_less`'s address fallback (different class)

**`:2848`** — `return a.get() < b.get();`, reached when two showers share a
start-segment graph index or both have none.  It orders `map_shower_ray` and
`map_shower_pair_mass_point`, i.e. π⁰ pairing.

**This is not a prototype-fidelity finding.**  The prototype orders these by
pointer everywhere; the toolkit is already strictly better.  It is kept because
it violates CLAUDE.md §2's own determinism rule, which the rest of this file
obeys — §6 clears every other container.

**Reachability is not decided from source.**  Distinct showers get distinct
start segments through the `map_segment_in_shower` guard, which
`update_shower_maps` refreshes between sub-calls — but that guard is not
consulted by `examine_showers`, which re-seats a start segment on an *existing*
shower at `:2363`, and five shower-construction sites (`:1076`, `:1435`,
`:1627`, `:1688`, `:2365`) pass a **throwaway local** `used_segments` rather
than a shared one.  So the fallback is plausibly reachable and not demonstrably
so.

**Solution — the pr/32 F3 precedent.**  `Shower` already carries
`m_shower_id`, assigned from a static atomic in the constructor
(`PRShower.cxx:45`), and `IndexedShowerSet` already orders by it
(`PRShower.h:223`).  Relative order within an event is preserved regardless of
how events interleave across threads.  So the fix is one line —
`return a->get_shower_id() < b->get_shower_id();` — behind
`shower_less_id_tiebreak`, **shipped with a counter for fallback hits**.
Expected byte-identical; that expectation rests on a control-flow argument, so
the counter is what converts it into a measurement.  pr/32's P7 went the same
route and came back 0 of 2219.

---

### §10.7 Dropped, with the reason each drop is safe

* **P4** — the prototype's `max_length` shadowing bug.  Dropped as an
  improvement.  **Doc-hygiene item kept**: it is not in the review doc's
  "Prototype Bugs Fixed" table and should be added, alongside P8 which is.
* **P5** — toolkit-only orphan rescue at `:730-760`.  An addition with a stated
  rationale, neither a bug nor a gap.  Its L.1 hardcode is `0.511` against
  `ParticleDataSet.cxx:66`'s `0.5109989461` — **2×10⁻⁶ relative, cosmetic**;
  worth a cleanup commit, not a finding.
* **P7 — RESOLVED, not merely dropped.**  §7 loose end 3 is closed.
  `TrackFitting::collect_2D_charge` (`:933-990`) reads exactly two things:
  `m_charge_data` and channel geometry.  `m_charge_data` is **never `.clear()`ed
  or `.erase()`d anywhere in `TrackFitting.{cxx,h}`** — it only grows, via
  `prepare_data()` on the dirty flag.  Its one transient mutation is
  `charge_err`, raised by `update_dQ_dx_data()` (`:5373`) and restored by
  `recover_original_charge_data()` (`:5406`) — and **`charge_err` appears zero
  times in `NeutrinoEnergyReco.cxx`**, the sole consumer of the three maps
  (`kine_charge_from_maps`, `:48`).  So collecting later can only make the map
  *more complete*, which is the direction of the prototype's already-global
  `main_cluster + other_clusters` bounding-box query.  The relocation moves
  toward the prototype, not away.
* **P8** — `fabs(pdg==13)`.  An improvement; already in the review doc's table.
* **P9** — index-equality → nearest-distance.  This is the id→position class the
  owner **pre-cleared in pr/32**, where it killed that doc's P6 outright.  Same
  class, same drop.  **The empty-`fits()` sub-claim is dropped separately and
  for a different reason**: it is a degenerate-input guard (the P10 family), its
  reachability is unmeasured, and folding it under the pre-clearance would
  overstate what that clearance covered.
* **P10** — null guards.  Improvements; reachability of the degenerate inputs
  unmeasured (§7 loose end 4 stands).
* **P11** — five argmax tie-breaks.  Improvements.  See correction 7: the row
  set is not a single-revision observation.
* **P13** — anti-hang guards.  Improvements.
* **P14** — B.6.  Already ruled.

---

### §10.8 Corrections to this doc's own §3, §7 and §8

Written down because §3 was produced at a different revision and two of its
claims do not survive re-verification.

1. **All §3/§8 anchors are stale by up to +19 lines.**  `397b1517` landed after
   the read (3384 → 3403).  Function heads at HEAD: `:76 :230 :473 :762 :1223
   :1304 :1641 :2099 :2458 :2802 :3179`.
2. **§3 P2's "two sites *do* match" is incomplete — there are four.**  `:2550`
   and `:2585` in `id_pi0_with_vertex` read `shower->get_particle_type()` with
   `abs`, and so does the prototype (`shower_clustering.h:743`, `:775`,
   `fabs(...)!=13`).  They **match** and are not P2 sites.  The mismatch count
   stays five; the denominator goes from seven to nine.  Note the consequence:
   the prototype itself is inconsistent *between* its two π⁰ finders —
   `id_pi0_with_vertex` reads the shower, `id_pi0_without_vertex` reads the
   start segment — which is very likely how the toolkit's uniform reading arose.
3. **§3 P6's second site is provably unobservable.**  `:1894`'s `is_shower`
   feeds only `n_showers`, which is `(void)n_showers;` at **`:1912`**.  The
   prototype's counterpart (`em_shower.h:508`) is equally dead — its gate at
   `:531-532` reads `n_tracks` and `total_length` only.  GOTCHA 9 already said
   `n_showers` is dead in the prototype; it is dead in the toolkit too, so the
   divergence at that site cannot be observed.  **P6 is one live site.**
4. **§3 P6 missed a live sub-case.**  The `:815` `acc_length1` test admits PDG
   **−11** where the prototype excludes it — see §10.5.
5. **§3 P3 understates the reach.**  "A lost invariant, not a confirmed bug" was
   written because no consumer was found.  Two exist — `NeutrinoTaggerNuE.cxx`
   and `MultiAlgBlobClustering.cxx` — see §10.4.  §9's third bullet is superseded.
6. **§7 loose end 3 is closed** — §10.7's P7 entry.  §9's fourth bullet
   ("P7 is half-verified") is superseded.
7. **§3 P11's row set changed post-hoc.**  `397b1517` **created** the
   `examine_shower_1` max-segment tie-break now at `:1900-1907` — after this
   audit was written, with the doc pr/28 §15.8 rationale in the comment.  So
   P11's five rows are not five observations of one revision.  Does not change
   the drop.
8. **§3.0 and P1's B.1 history survive unchanged.**  Re-read at HEAD: `:289` is
   still `calculate_num_daughter_showers(graph, main_vertex, sg)`, and the
   pre-B.1 form with the explicit `false` really was semantically identical to
   the prototype's `_tracks(…, true)`.  Applying B.1 introduced the divergence.

---

### §10.9 What §10 does not claim

* **Still no event was run and no code was changed.**  Every fix above is a
  proposal with a knob name, not a patch.  No gate, no manifest, no label.
* **The five kept findings are confirmed as divergences; their *frequency* is
  not.**  F1's `<= 3` crossing rate, F2's shower-vs-start-segment disagreement
  rate, F3's both-finders-fire rate and F5's fallback-hit rate are all
  unmeasured — which is why every proposed fix above ships with a counter.
* **"Expected byte-identical" is claimed only for F5**, and only as a
  control-flow argument.  F1–F4 are expected to *change* output when on; that is
  the point of them.
* **P7's resolution is a source argument, not a measurement.**  It rests on
  three greps stated explicitly in §10.7 so they can be re-run: `charge_err` in
  `NeutrinoEnergyReco.cxx` (zero hits), `m_charge_data` clear/erase in
  `TrackFitting.{cxx,h}` (zero hits), and the consumers of `m_charge_2d_*`
  (`cal_kine_charge` only).  If any of those changes, P7 reopens.
* **P12's reachability remains undecided** — §10.6 says so explicitly rather
  than ranking it as likely or unlikely.
* **The §0 "not audited" list is unchanged.**  `PRShower.cxx` ↔ `WCShower.cxx`,
  `cal_kine_charge` internals and the line-by-line arithmetic of
  `examine_shower_1` / `examine_showers` were not opened for this filter either.
  A sixth finding could live there.

---

### §10.10 Amendments to §10.2–§10.4 (self-review pass)

Recorded rather than silently rewritten, so the reasoning is auditable.

1. **F3's knob-off guarantee was false as first written** and is now corrected in
   §10.4.  The original proposal widened `shower_clustering_with_nv` to `int&`;
   that leaks the π⁰ increments into `ssm_tagger`, which already takes `int&`
   and **seeds its pseudo-particle ids from the value** — an unconditional
   output change with the knob off.  The fix is to widen only the two π⁰
   finders.  This is the one amendment that would have bitten during
   implementation.
2. **F2 is three knobs, not one** (§10.3).  The one-knob version contradicted
   §10.2's own attribution criterion at `:525`, which the single knob inverts.
   The knob name also described only four of the five sites.
3. **`:2193` is now classified.**  §3's P2 prose mentioned it (as `:2174`) but
   left it out of the table, and §10's first draft neither kept nor dropped it.
   It is an `abs`-only divergence and is folded into F2's third knob.

Nothing in §10.1's five-way verdict changed: these are amendments to the
proposed *fixes*, not to the filter.

# doc pr/113 — post-vertex EM shower clustering, π⁰ and long muon: coverage audit + a topology-tagged event list

**Result.** Nothing is *missing* from the three post-vertex stages: every
prototype function in shower clustering, π⁰ reconstruction and long-muon
construction has a live toolkit counterpart. The one ABSENT call site
(`separate_track_shower()`, the no-arg overload) turns out to feed a consumer
chain that is **dead in both trees**, so it is a matrix row and not a defect.
What the audit did find is in the two places earlier rounds never read: **one
4-line prototype function, `WCShower::set_start_vertex`, carries two independent
toolkit divergences** (a stale `start_connection_type` on the early-return path,
and a toolkit-only `add_vertex`), neither in `porting_dictionary.md`, one of them
worked around by a shipped SBND-ON knob at the point of *use* rather than fixed
at the source. A third candidate sits at the long-muon construction site, where
the prototype's `cal_4mom()` guard is dropped. Separately, the in-tree review doc
for the never-audited `PRShower`↔`WCShower` pair is **stale**: of its 8 claims
re-checked, 7 are already fixed and the single ⚠ CRITICAL is a **false positive**.
The determinism check pr/33 and pr/91 both left open is now **closed at
container level for this pair** — pr/33's P12 residual is gone. On the sample side, all four prod0825 arms
are **data** (no MC truth exists, or can), so the delivered lists are
reconstruction-defined: **90 nueCC, 46 NCπ⁰, 79 numuCC-with-≥100 MeV-EM-shower**,
disjoint by construction. A by-product worth the owner's attention:
**`kine_pio_flag` is not a π⁰ selector** — it fires on 37/48 of the curated nueCC
arm and on 232 beam events where only 55 actually carry a reconstructed γ pair.

> **Round 2 correction (2026-08-27), §10.** The list sizes and every π⁰-pairing
> count above are the *corrected* ones. Round 1 shipped `98 / 23 / 79` and
> claimed only 7 of 1433 events carry a γ pair; a falsy-zero bug in the census
> (`pio_id or -1`, and `pio_id` starts at **0**) hid the first π⁰ group of every
> event. The true count is **70 of 1433**. The audit sections §1–§5 are
> unaffected — the bug was in the sample census only. Full accounting in §10.

**No code changed this round.** The findings below are *unfiltered candidates*
with proposed default-OFF knobs, for the owner to filter (pr/33 precedent: 14 → 5).

## Repro

```sh
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# the event list (pure ROOT/JSON read of existing prod0825 products; no WCT run)
python3 scripts/pr113_topology_census.py \
    work-nuecc48-prod0825:nuecc48 work-ncpi0-prod0825:ncpi0 \
    work-mcp1k-prod0825:mcp1k     work-mcp2k-prod0825:mcp2k \
    --out docs/pr/pr113-emshower-sample.tsv --outdir docs/pr

# the audit is read-only; anchors:
git -C .. rev-parse --short HEAD            # toolkit 8d93260d
git -C ../../wcp-porting-img rev-parse --short HEAD   # wcp bf65fbe
```

Toolkit anchor **`8d93260d`**, wcp-porting-img **`bf65fbe`**, 2026-08-26.
Every line number below was re-derived at that SHA. **None is inherited** from
pr/33 or from an in-tree review doc — pr/33's map is from `f07c0299` and is
badly stale: `NeutrinoShowerClustering.cxx` has gone 3403 → **5295** lines and
`shower_clustering_with_nv` moved `:3160` → **`:4375`** (pr/33 GOTCHA 15: a
landing commit once *created* one of that doc's findings between read and
write-up).

---

## 0. Scope

### 0.1 Why this is not a repeat of pr/33

The divergence audit of `NeutrinoShowerClustering.cxx` ↔
`NeutrinoID_shower_clustering.h` + `NeutrinoID_em_shower.h` was done in pr/33
(14 findings → owner filter → 5 kept, 8 knobs) and refined by pr/84, pr/91 r1–r3,
pr/92 and pr/93 r1–r4. Re-reading that pair regenerates pr/33. This round goes
to the three places those rounds named in writing as *not* covered:

| gap | named by |
|---|---|
| `PRShower.cxx` (1833) ↔ `WCShower.cxx` (756) never audited | pr/33 §0 "Not audited" — *"its own audit"* |
| `examine_shower_1` / `examine_showers` never read line-by-line for arithmetic | pr/33 §7 loose end 5 |
| long muon has no audit at all | no prior doc; only incidental mentions in pr/91, pr/93, pr/46 |

The owner's question — *"are we missing anything"* — is a **coverage** question,
not the divergence question pr/33 answered, so §1 leads with a present/absent
matrix.

### 0.2 The post-vertex stage sequence

Verified in `clus/src/TaggerCheckNeutrino.cxx`, matching prototype
`NeutrinoID.cxx:205-269`:

```
determine_overall_main_vertex_DL  :2293   /  determine_overall_main_vertex :2315
  snap_main_vertex_to_kink :2356  /  _to_junction :2372
  improve_vertex :2378  ·  main_vertex_graph_audit :2394  ·  stitch :2408
  clustering_points :2421  ·  reassociate_cluster_orphans :2446
  examine_direction(flag_final=true)  :2455    <-- LONG MUON
  demote_cross_cluster_straight_stems :2476
  shower_clustering_with_nv           :2502    <-- EM SHOWER + pi0
  reconcile_particle_flags            :2525
  cosmic_tagger :2590 · numu_tagger :2601 · ssm :2608 · nue :2645 · sp :2662
```

`shower_clustering_with_nv`'s own sub-stage order, re-derived at `:4375-5230`:

```
4465 _in_main_cluster   4474 _connecting_to_main_vertex   4481 _from_main_cluster
4534 _from_vertices     4545 collect_charge_maps          4551 calc_kine (1)
4556 examine_merge_showers                                4564 _in_other_clusters
4578 stem_backfill*     4588 calc_kine (2)                4593 examine_showers
4606 merge_showers_sharing_start_segment*                 4633 detach_track_stem*
~5100 hadronic retype*  5180 recompute_shower_kine_charge_final*
5189 id_pi0_with_vertex 5199 id_pi0_without_vertex
```
`*` = toolkit-only, no prototype counterpart (§1.4).

### 0.3 Long muon: in scope, with a nuance

The single construction site is `NeutrinoVertexFinder.cxx:1861-1916` inside
`examine_direction`, which runs **twice**: pre-vertex from `determine_main_vertex`
(`TaggerCheckNeutrino.cxx:2209/2244/2262`) and post-vertex at `:2455`
(`flag_final=true`). A `flag_fill_long_muon` guard (`:1861-1867`) means the
post-vertex call cannot refill a cluster that already has a chain, so in practice
the chain is usually built *before* the vertex is fixed.

**This is not a divergence.** The prototype has the identical two-call structure
(`NeutrinoID_track_shower.h:1381` inside `determine_main_vertex`, and
`NeutrinoID.cxx:225` post-vertex) and the identical guard at `:2137-2142`.

### 0.4 Not audited — stated so this doc's reach is not overstated

- **Read boundary in `PRShower.cxx`↔`WCShower.cxx`**: read line-by-line —
  `set_start_vertex`, `set_start_segment`, `add_vertex`/`add_segment`,
  `add_shower`, `fill_maps`, `complete_structure_with_start_segment`,
  `calculate_kinematics_long_muon`, `get_last_segment_vertex_long_muon`,
  `count_connected_segments`, plus every container declaration. **Not read for
  arithmetic**: `calculate_kinematics` body, `get_stem_dQ_dx`,
  `get_connected_pieces`, `update_particle_type`, `detach_track_prefix`,
  `rebuild_point_clouds`/`build_point_clouds`, `fill_point_vector`,
  `get_total_length`/`get_num_*`. A numeric divergence can still hide there.
- **The in-tree review list was re-verified for 8 of its 22 items** (§2.2); the
  remaining 14 (mostly MULTI-APA and EFFICIENCY) were not re-checked.
- `examine_shower_1`/`examine_showers` — pr/33 loose end 5 is **NOT closed by
  this round** (§2.3). It was scoped in and dropped for time; it stays open.
- `cal_kine_charge` internals (pr/27 §9); the tagger bodies (docs 74/75, pr/36)
  — only their *use* of the long-muon sets is in scope.
- Pre-vertex `separate_track_shower(cluster)` and
  `shower_determining_in_main_cluster` (doc pr/31).

---

## 1. Coverage matrix

Every verdict below is a grep of this tree at `8d93260d`, not a subagent's and
not an in-tree doc's. That matters here specifically: pr/33 GOTCHA 1 found
`shower_clustering_review.md` had conflated two different functions and GOTCHA 3
found `porting_dictionary.md:222` itself wrong. Two independent checks in this
round found the same class of error again (§1.2, §1.3).

### 1.1 EM shower clustering — all PRESENT

| prototype (`NeutrinoID_shower_clustering.h` unless noted) | toolkit (`NeutrinoShowerClustering.cxx`) | verdict |
|---|---|---|
| `update_shower_maps` `:1388` | `:233` | PRESENT |
| `shower_clustering_with_nv_in_main_cluster` `:1654` | `:547` | PRESENT |
| `shower_clustering_connecting_to_main_vertex` `:114` | `:741` | PRESENT |
| `shower_clustering_with_nv_from_main_cluster` `:1775` | `:1063` | PRESENT |
| `shower_clustering_with_nv_from_vertices` `:995` | `:1537` | PRESENT |
| `examine_merge_showers` `:380` | `:2079` | PRESENT |
| `shower_clustering_in_other_clusters` `:1442` | `:2190` | PRESENT |
| `examine_shower_1` (`em_shower.h:337`) | `:2659` | PRESENT |
| `examine_showers` (`em_shower.h:1`) | `:3187` | PRESENT |
| `id_pi0_with_vertex` `:735` | `:3592` | PRESENT |
| `id_pi0_without_vertex` `:428` | `:3959` | PRESENT |
| `shower_clustering_with_nv` `:268` | `:4375` | PRESENT |
| `get_start_end_vertices` `:1431` | absorbed into `find_vertices` | DIVERGENT-BY-DESIGN (pr/30) |

### 1.2 π⁰ — PRESENT; and one false "gap" retired

Both finders are present (above). The **mass window is a faithful port**, checked
constant by constant: centre `135 MeV`, `mass_offset = 10 MeV`, accept
`(-25, +35) MeV`, `6 MeV` penalty for type2↔type2 — toolkit
`NeutrinoShowerClustering.cxx:3853-3866`, prototype `:917-930`.

**A π⁰ *tagger* looks missing and is not.**
`clus/inc/WireCellClus/NeutrinoTaggerPi0.h` is a **0-byte** placeholder,
`clus/src/NeutrinoTaggerPi0.cxx` **does not exist**, and
`grep -rn "pi0_tagger" clus/` returns nothing — while the other five taggers each
have a `.cxx` *and* an entry point in `NeutrinoPatternBase.h`. But
`prototype_base/pid/src/NeutrinoID_pio_tagger.h` is **2 bytes** — empty in the
prototype too. ⇒ **not a gap.**

> Mentioned, not fixed (CLAUDE.md §5): `clus/docs/track_shower_separation.md:233-241`
> claims `NeutrinoTaggerPi0.cxx` exists and implements a π⁰ tagger. It does not.

### 1.3 Long muon — producer and consumers all PRESENT

| prototype | toolkit | verdict |
|---|---|---|
| formation block `track_shower.h:2137-2190` | `NeutrinoVertexFinder.cxx:1861-1916` | PRESENT (one candidate divergence — F2) |
| `find_cont_muon_segment` `:2304` | `NeutrinoVertexFinder.cxx:1339` | PRESENT (+ toolkit-only `flag_stub_bridge`, pr/46) |
| `WCShower::calculate_kinematics_long_muon` `WCShower.cxx:288` | `PRShower.cxx:1723` | PRESENT |
| `WCShower::get_last_segment_vertex_long_muon` `:241` | `PRShower.cxx:1020` | PRESENT |
| long-muon → EM erase `:1712-1765` | `NeutrinoShowerClustering.cxx:700-736` | PRESENT |
| `id_pi0_without_vertex` veto `:438` | `:3987-3988` | PRESENT |
| `_from_vertices` vertex skip `:1043` | `:1619` | PRESENT |
| cosmic / numu tagger consumers | `NeutrinoTaggerCosmic.cxx`, `NeutrinoTaggerNuMu.cxx` | PRESENT |

Two constants confirmed identical, not merely similar: the seed cut
`median_dQ/dx / 43e3 > 1.3 → skip` (toolkit uses `m_mip_dqdx_median`, whose C++
default is `43000/units::cm` and which **SBND does not override** — pr/33
GOTCHA 11's "SBND sets 48000" does **not** apply to this knob), and the accept
gate `total > 45 cm && max > 35 cm && size > 1`. `segment_track_length(seg, 0)`
(default `flag=0`, geometric sum over consecutive fit points,
`PRSegmentFunctions.cxx:1458-1470`) is the same quantity as the prototype's
`ProtoSegment::get_length()` (`ProtoSegment.cxx:706`).

### 1.4 ABSENT rows — with the liveness verdict that decides each

Only one true ABSENT call site exists, and it is **not** a defect. This is the
pr/33 GOTCHA 9 pattern (`n_showers` is dead in both trees) and it is worked
here as the calibration for the whole table.

**`separate_track_shower()` (the no-arg overload), prototype `NeutrinoID.cxx:239`.**
`clus/docs/porting/neutrino_id_function_map.md:544,571` flags it as a real
behavioural gap — *"NOT re-called in the toolkit"*. Reading the prototype body
(`NeutrinoID_track_shower.h:28-63`) shows the function-map's reading is wrong:
the no-arg overload is **not** a re-run of track/shower classification. It paints
per-point `point_flag_showers` on each cluster from the final `get_flag_shower()`
of the segment owning each point. Liveness, both trees:

- **Toolkit**: the counterpart is
  `PatternAlgorithms::transfer_info_from_segment_to_cluster`
  (`NeutrinoPatternBase.cxx:3211`, declared `NeutrinoPatternBase.h:3130`).
  Repo-wide grep: **defined, never called.** Its output `point_flag_shower` is
  read only by `Cluster::shower_flags()` → `Cluster::shower_flag(i)`
  (`Facade_Cluster.cxx:890,916`), which **has no caller either**.
- **Prototype**: `point_flag_showers` is written by that overload and otherwise
  only *re-sized* by `fill_fit_parameters()` (`NeutrinoID.cxx:1004-1009`). Read
  nowhere in `prototype_base/`.

⇒ **ABSENT call site, dead consumer chain in both trees. No observable effect.**
A row and a footnote, not a finding.

### 1.5 TOOLKIT-ONLY passes (the other direction of "missing")

No prototype counterpart; all default-OFF in C++, several SBND-ON:
`stem_backfill`, `merge_showers_sharing_start_segment`, `detach_track_prefix`,
`nv_bridge_connect`/`nv_bridge_track`, the hadronic retype,
`recompute_shower_kine_charge_final`, `reconcile_particle_flags`,
`override_muon_multi_proton_pion`, `override_michel_stem_muon`,
`single_muon_long_muon_claim`. Listed so the matrix is honest in both
directions — the toolkit has *more* than the prototype here, not less.

---

## 2. Deep reads

### 2.1 `Shower::set_start_vertex` — one 4-line function, two divergences

The prototype (`WCShower.cxx:529-532`) is four lines:

```cpp
void WCPPID::WCShower::set_start_vertex(ProtoVertex* vertex, int type){
  start_vertex = vertex;
  start_connection_type = type;
}
```

The toolkit (`clus/src/PRShower.cxx:198-210`) at `8d93260d`:

```cpp
Shower& Shower::set_start_vertex(VertexPtr vtx, int type)
{
    if (!vtx || ! vtx->descriptor_valid()) {
        m_start_vertex = nullptr;
        return *this;                       // <-- (a) type NOT updated
    }
    this->add_vertex(vtx);                  // <-- (b) no prototype counterpart
    m_start_vertex = vtx;
    data.start_connection_type = type;
    return *this;
}
```

These are **two consequences of one divergent method**, reported as one row —
splitting them would inflate the count and hide that a single short function is
the source.

**(a) Stale `start_connection_type` on the early-return path** (pr/84 §8.1,
recorded as *"not picked here"*). On an invalid vertex the toolkit nulls the
start vertex but leaves `data.start_connection_type` at its **previous** value,
pairing a null start vertex with a stale connection type. The prototype assigns
both unconditionally. Connection type decides pseudo-parent insertion, so a
failed re-seat can silently flip an electron to a gamma. **Not in
`porting_dictionary.md`** — `grep -n "set_start_vertex\|start_connection_type"`
on that file returns **zero hits**, confirming pr/84's note is still true.
**Reachability: UNDECIDED** — 16 call sites; the ones worth checking first are
`NeutrinoShowerClustering.cxx:2300/2488/2606` (`min_vertex` in
`shower_clustering_in_other_clusters`) and `:3895/:3901` (the π⁰ re-seat, which
is exactly where pr/91 found orphaned start vertices). Same posture as pr/33
GOTCHA 20: ship the repair with a hit counter before believing any rate.

**(b) The toolkit-only `add_vertex(vtx)`** (pr/91 r2/r3). The prototype's
`set_start_vertex` touches no map; the toolkit adds the vertex to the shower's
own view, marking the creation vertex *visited but never expanded*. pr/91 r3
shipped `shower_walk_visited_parity` (SBND-ON) which fixes this **at the point of
use** — the walk frontier tracks `m_walked_nodes` instead of view membership —
and explicitly deferred the wider fix at the source, along with remedies R2/R3,
as "not census-boundable".

There is a **second local compensation** for the same divergence, found this
round: `update_shower_maps` (`NeutrinoShowerClustering.cxx:253-259`) explicitly
skips `start_vtx` when filling `map_vertex_in_shower`, with the comment
*"Matches prototype: set_start_vertex() never adds to map_vtx_segs"*. That
comment describes the **prototype**, not the toolkit — the toolkit's
`set_start_vertex` **does** add. So the toolkit now carries at least two
independent local work-arounds for one un-fixed source divergence. That is the
argument for adjudicating it rather than adding a third.

> **Proposed (default-OFF, unfiltered candidate):** `shower_start_vertex_parity`
> — assign `data.start_connection_type = type` before the early return, and
> (separately gated) drop the `add_vertex`. Ship with a hit counter first.

### 2.2 The in-tree `PRShower` review doc is stale — re-verification

`clus/docs/patternrecognition/prvertex_prsegment_prshower_review.md` predates
pr/84/91/93 and, per pr/33 GOTCHA 2, its status column is a *recommended action*,
not a record of a fix. Re-checked at `8d93260d`:

| # | claim | verdict at `8d93260d` | evidence |
|---|---|---|---|
| F1 ⚠CRITICAL | `Shower::fill_maps` is a no-op ⇒ shower maps silently empty | **NEVER-TRUE** | `fill_maps()` returning `*this` is the deliberate API: it hands back the `TrajectoryView`, and `update_shower_maps` (`NeutrinoShowerClustering.cxx:249-267`) iterates `traj.nodes()`/`traj.edges()` to fill `map_vertex_in_shower` / `map_segment_in_shower`. The maps are populated. The doc read the signature and never checked the 6 call sites. |
| F2 | `std::map<Cluster*,…>` pointer-keyed | **ALREADY-FIXED** | `PRSegmentFunctions.cxx:3228` uses `ClusterIdCmp` |
| F3 | `std::map<SegmentPtr,…>` pointer-keyed | **ALREADY-FIXED** (with a residue, §2.4) | `:3242-3244`, `:3403` use `SegmentIndexCmp` |
| F5 | `clear_fit` drops per-fit `paf` | **ALREADY-FIXED** | `PRSegment.cxx:138` sets `paf = {wpid.apa(), wpid.face()}` |
| F6 | `segment_track_direct_length` n2 clamp inconsistent | **ALREADY-FIXED** | `PRSegmentFunctions.cxx:1520-1525` clamps both to `fits.size()-1` |
| F7 | `wcpts().front()` unguarded | **ALREADY-FIXED** | `PRGraph.cxx:88` — `if (seg->wcpts().empty()) return true;` |
| F8 | `PR::Vertex` inherits `HasCluster<Segment>` | **ALREADY-FIXED** | `PRVertex.h:77` — `HasCluster<Vertex>` |
| F10 | `std::set<SegmentPtr>&` params | **ALREADY-FIXED** | no `std::set<SegmentPtr>` remains in `PRShower.h` |

**All 8 re-checked claims are stale** — 7 already fixed, and the one ⚠ CRITICAL a
false positive. The remaining 14 items (MULTI-APA F12-F14, EFFICIENCY F15-F19,
MINOR F20-F22, plus F4/F9/F11 whose anchors have all moved as the file grew
1273 → 1833 lines) were **not** re-checked — see §0.4.

> Recommendation: the review doc should carry a "re-verified at `8d93260d`:
> F1/F2/F3/F5/F6/F7/F8/F10 all stale" banner so the next reader does not re-chase
> them. Not edited this round (no code changes; the owner may prefer it deleted).

### 2.3 `examine_shower_1` / `examine_showers` — NOT closed

pr/33 §7 loose end 5 (arithmetic never read across the 450+357 prototype lines,
now `:2659` and `:3187` in the toolkit) was scoped into this round and **dropped
for time**. It stays open, and it is the highest-value remaining read in this
area, because pr/91 F2 already found two real numeric problems there — a conn-3
hatch that measures to `parent->start_segment()` and so reads 4.914 cm against a
hard 3 cm gate where the real touch is 1.704 cm, and a second candidate missing
a hard 3 cm total-length cut by **1.5 mm**. Both were reported as latent gaps
*shared* with the prototype rather than as divergences; only the arithmetic read
can confirm that.

### 2.4 Determinism — closed at container level for this pair

**Scope of the claim, stated first**: what follows covers every **declared
container and typedef** of the `PRShower`/`WCShower` pair. It does **not** sweep
the method bodies listed as unread in §0.4 (`calculate_kinematics`,
`get_connected_pieces`, `update_particle_type`, `detach_track_prefix`, …) — a
*local* pointer-keyed container inside one of those would not appear in the table
below. "Determinism closed for the pair" would overstate it; the container-level
result is what is claimed.

CLAUDE.md §4's port checklist requires this and pr/33 (P12) / pr/91 §8 both left
it open. Checked at `8d93260d`:

| container | key order | verdict |
|---|---|---|
| `IndexedShowerSet` | `ShowerIndexCmp` → `get_shower_id()` | index-keyed |
| `ShowerVertexMap` | `VertexIndexCmp` | index-keyed |
| `ShowerSegmentMap` | `SegmentIndexCmp` | index-keyed |
| `VertexShowerSetMap`, `ShowerIntMap` | `VertexIndexCmp` / `ShowerIndexCmp` | index-keyed |
| `ClusterPtrSet` | `ClusterPtrCmp` → `get_cluster_id()` | index-keyed |
| `Shower::m_walked_nodes`, `touched` (`PRShower.cxx:104,1438`) | `std::set<node_descriptor>` (integers) | deterministic |

**pr/33's P12 is FIXED**: `ShowerIndexCmp` (`PRShower.h:391-401`) compares
`get_shower_id()` with **no address fallback**, and warns once via an atomic when
an id is unassigned. No pointer-ordered container remains **among the pair's
declared containers and typedefs**.

**One residue, outside the pair, benign.** `PRSegmentFunctions.cxx:3364-3365`
declares `std::map<SegmentPtr, …>` **without** a comparator (default
`std::less<shared_ptr>` = address order) and both are iterated —
`map_segment_global_indices` at `:3565`, `map_segment_points` at `:3722`. Order
does **not** affect the result: the first body only inserts into a
`std::set<int>`, and the second calls `create_segment_point_cloud(seg, …)`, which
writes to each segment's own cloud with no shared mutable state. ⇒ **CLAUDE.md
§2 house-rule violations, not determinism bugs** — the same labelling discipline
pr/33 applied to F5=P12. Worth a two-line comparator fix whenever that file is
next touched; no knob, no gate.

### 2.5 Long muon — the construction site, line by line

Toolkit `NeutrinoVertexFinder.cxx:1861-1916` vs prototype
`NeutrinoID_track_shower.h:2137-2190`. Structurally 1:1: fill-once guard, seed
dQ/dx cut, chain walk, accept gate, per-segment retype, both set insertions.
Iteration order is *more* deterministic in the toolkit (`sorted_out_edges` vs the
prototype's pointer-ordered `map_vertex_segments[temp_vertex]`) — an improvement,
not a divergence. One divergence:

**F2 — the `cal_4mom()` guard is dropped.** The prototype's documented
reclassification idiom — quoted verbatim in this tree at
`clus/src/NeutrinoTrackShowerSep.cxx:14-19` — is:

```cpp
sg->set_particle_type(13);
sg->set_particle_mass(mp.get_mass_muon());
if (sg->get_particle_4mom(3) > 0) sg->cal_4mom();   // "an energy already exists"
```

The toolkit (`:1905-1913`) instead builds a fresh `Aux::ParticleInfo`
unconditionally, from `segment_cal_4mom(acc_seg, 13, …)` — which always returns
`E = kine_energy + mass > 0` (`PRSegmentFunctions.cxx:2837-2867`, no early-out).

Footprint: the two paths agree whenever a prior energy exists (both recompute
from scratch). They differ **only** for a chain segment that reaches this site
with no prior 4-momentum — the prototype leaves it at zero, the toolkit assigns
one. That population is bounded by pr/40's persistence work: `track_pid_persist_dqdx`
and `track_pid_persist_4mom` are **both SBND-ON**
(`wct-pr-perevt.jsonnet:1348,1378`), so most segments do carry `particle_info` by
this point. **Not in `porting_dictionary.md`** (only a generic pointer at `:232`
to `pid_direction_kinematics_review.md`).

Reading both ways, per CLAUDE.md §5.4 — **I do not pick**:
- *Toolkit is right*: an energy is strictly better than a zero; `ParticleInfo` is
  replaced wholesale in this object model, so "keep the old 4-mom" has no natural
  expression, and pr/40 F4 already judged the analogous free-end gate
  "unnecessarily strict".
- *Prototype is right*: the guard is deliberate and appears at 11+ sites; a
  segment with no computed energy has no *reliable* energy, and manufacturing one
  here puts a number into the PF tree and `kine_energy_particle` that the
  prototype would have left at zero.

> **Proposed (default-OFF, unfiltered candidate):** `long_muon_4mom_guard` —
> gate the 4-momentum install on an existing `particle_info()` with
> `kinetic_energy() > 0`, keeping the type+mass store unconditional. Ship with a
> hit counter; if it comes back 0 the question is moot (the pr/32 F3 precedent
> returned 0/2219).

### 2.6 Long muon — a benign PARTIAL

`calculate_shower_kinematics` (`NeutrinoEnergyReco.cxx:304-305`) takes
`IndexedVertexSet& vertices_in_long_muon` and immediately `(void)`-casts it; only
`segments_in_long_muon` is read (`:331`). The prototype's counterpart
(`NeutrinoID_shower_clustering.h:1407-1428`) also reads only the segment set —
it dispatches to `calculate_kinematics_long_muon(segments_in_long_muon)` when
`|pdg| == 13`. ⇒ **dead parameter, not a behavioural divergence.** Threading it
through the whole call chain is cosmetic noise; mentioned, not fixed.

---

## 3. Findings table — unfiltered candidates for the owner's filter

| # | where | what | class | proposed knob (all default-OFF) |
|---|---|---|---|---|
| **F1** | `PRShower.cxx:198-210` | `set_start_vertex`: (a) stale `start_connection_type` on early return; (b) toolkit-only `add_vertex`. One method, two consequences. Undocumented. | port divergence | `shower_start_vertex_parity` (two sub-gates) |
| **F2** | `NeutrinoVertexFinder.cxx:1905-1913` | long-muon retype drops the prototype's `if (get_particle_4mom(3)>0)` guard on `cal_4mom` | port divergence | `long_muon_4mom_guard` |
| **F3** | `PRSegmentFunctions.cxx:3364-3365` | two `std::map<SegmentPtr,…>` without comparator, both iterated; bodies order-independent | house-rule (CLAUDE.md §2), not a bug | none — 2-line comparator fix when next touched |
| **F4** | `NeutrinoEnergyReco.cxx:304-305` | `vertices_in_long_muon` threaded through and `(void)`-cast; prototype also ignores it | cosmetic | none |
| **F5** | `clus/docs/track_shower_separation.md:233-241` | claims a `NeutrinoTaggerPi0.cxx` that does not exist on either side | doc bug | none — mention only |
| **F6** | `prvertex_prsegment_prshower_review.md` | 8/8 re-checked claims stale (7 fixed + 1 false positive) | doc bug | none — banner or delete, owner's call |

**Zero of these ship this round.** F1 and F2 are the only two that would change
output, and both need a hit counter before a rate can be claimed.

---

## 4. SBND operating point

The audit reads the code **as SBND runs it**, not as it ships default-OFF. Knobs
relevant to these three stages that are **ON** in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`:

| knob | line | round |
|---|---|---|
| `shower_long_muon_keep_type` | 1617 | pr/44 — prototype-parity restoration |
| `single_muon_long_muon_claim` | 1775 | pr/43 r2 — *no prototype counterpart* |
| `long_muon_stub_bridge` | 1816 | pr/46 |
| `track_pid_persist_dqdx` / `track_pid_persist_4mom` | 1348 / 1378 | pr/40 — bounds F2's footprint |
| `shower_endpoint_skip_orphan_vtx`, `shower_walk_visited_parity` | — | pr/91 r1, r3 — both work around F1(b) |

`m_mip_dqdx_median` is **not** overridden by SBND; it stays at the C++ default
`43000/units::cm`, identical to the prototype's `43e3` (§1.3).

---

## 5. Looks like a divergence and is not

1. **π⁰ tagger missing** — empty in the prototype too (§1.2).
2. **`separate_track_shower()` not re-called** — dead consumer chain in both
   trees (§1.4). The function map's description of what it does is also wrong.
3. **Long muon "built pre-vertex"** — the prototype has the same two-call
   structure and the same fill-once guard (§0.3).
4. **`vertices_in_long_muon` ignored** — the prototype ignores it too (§2.6).
5. **`acc_segments.size() > 1` excludes a single-segment muon** (pr/46 §9 class
   d) — prototype-identical at `NeutrinoID_track_shower.h:2172`. **M15 parity,
   not a bug**; pr/46 declined to touch it because dropping it sweeps every
   ordinary vertex-attached CC muon into the pseudo-shower category.
6. **`kine_pio_flag` fires far more often than π⁰ pairing** (§6.4) — the kine
   branch has no mass window and the pattern branch does, in **both** trees.

---

## 6. The event list

### 6.1 There is no MC truth, and there cannot be

All four prod0825 arms are SBND **data**. Verified directly this round:
`input_files_reco1/nc-sideband_filtered_frameshift.root` (the NCπ⁰ source) has
**1909 branches, all raw DAQ** — `artdaq::Fragments`, `sbnd::crt::*`,
`raw::OpDetWaveform`, `raw::RDTimeStamps`, `sbndaq::CRTmetric` — and **zero**
`simb::MCTruth`/`MCParticle`. Consistent with doc 83, which corrected doc 80's
"MC" label. The "nueCC48"/"NCπ⁰-19" names are upstream reco/sideband selections,
not truth labels, and `nusel-events.tsv`'s `event_label` is a per-bundle reco
verdict (`nu-candidate` / `not-tagged`), not a topology.

**Every label below is a reconstruction verdict.** Purity is unknown and
unknowable here. These lists are for exercising and eyeballing the
shower/π⁰/long-muon code — **not** for efficiency or purity measurements.

### 6.2 Two pool denominators, both reported

| arm | `pr_evt*` dirs | `calib-pr-evt*.json` | `nusel` nu-candidate |
|---|---|---|---|
| nuecc48 | 48 | 48 | 47 |
| ncpi0 | 19 | 19 | 19 |
| mcp1k | 1000 | 461 | 508 |
| mcp2k | 2000 | 905 | 993 |
| **total** | **3067** | **1433** | **1567** |

The census denominator is **1433** — events where `TaggerCheckNeutrino` actually
evaluated a main cluster. Of those, **1384 have a main vertex**; the other **49**
have a calib dump with `main_vertex == null` and empty `showers`/`segments`, and
are carried as rows with `has_main=0` rather than dropped.

The 1567 `nu-candidate` count is **not joinable** with 1433: it is a *pre-PR
bundle* verdict from `nusel_extract.py`, taken before `unmerge_bundle` runs, on a
different numbering scheme. Neither number is inherited — both are counted by the
script.

### 6.3 Definitions (a priority ladder, so the lists are disjoint)

```
1. numuCC_em   has_mu and em_max >= 100 MeV        <-- the owner's ask
2. NCpi0       (not has_mu) and has_pio_pair       and em_max >= 100
3. nueCC       (not has_mu) and (not has_pio_pair)
                              and has_e_nonpio     and em_max >= 100
```
- `em_max` = max `kine_best` over `showers[]` with `particle_id == 11`.
- `has_mu` = a pdg-13 primary (segment incident on the main vertex) **≥ 30 cm**.
- `has_pio_pair` = ≥ 2 showers carry `pio_id >= 0`, i.e. a real γ **pair**.
- `has_e_nonpio` = a primary e-rooted shower that is *not* π⁰-paired.
- `has_e` is reported as a column for continuity but is **not** part of any
  verdict — the ladder uses `has_e_nonpio`.

**Both floors are measured, not assumed.** Each was put in after a hand check of
a row the census had already emitted, and each caught a real defect:

| floor | why | effect |
|---|---|---|
| muon primary **≥ 30 cm** | `nuecc48` evt 111412 had `has_mu` set by a **2.4 cm** pdg-13 stub. Unfloored the predicate fires on 1025/1366 beam events (5th pct 3.2 cm); at 30 cm, 791. | see table below |
| EM shower **≥ 100 MeV** on *all three* reco verdicts | hand check of the delivered nueCC list found `mcp1k` 166804 (`e-` **10.2 MeV**, a 3.0 cm segment) and 167684 (**41.7 MeV**) — against a curated nueCC48 arm whose `em_max` median is **1173 MeV**, 47/48 above 100. Reco additions had median 79 MeV, 43/129 below even 50. | nueCC reco 129 → **50** |

After flooring, the reco additions are comparable to the curated sets: nueCC reco
`em_max` median 200 MeV (min 101), NCπ⁰ reco median 1397 MeV (min 336 — the floor
is a no-op there), numuCC-EM median 195 MeV (min 104).

Muon-floor sensitivity of the numuCC list:

The two curated arms enter their own list wholesale as `origin=curated`; the
unbiased beam arms contribute `origin=reco` additions; `numuCC_em` is mined from
the beam arms only. Every row carries its largest shower's 50/100/200 MeV tier so
the cut can move without regenerating the list.

| floor | beam events with a muon-like primary | ∧ `em_max ≥ 100` |
|---|---|---|
| none | 1025 | 111 |
| 10 cm | 925 | 97 |
| 20 cm | 866 | 88 |
| **30 cm** | **791** | **79** |
| 50 cm | 668 | 61 |

### 6.4 `kine_pio_flag` is not a π⁰ selector — quantified

> ⚠ **The right-hand column below is the round-1 buggy count. See §10 for the
> corrected table (7 / 8 / 55). The conclusion holds; the margin is 4:1, not
> 58:1.** Text kept rather than silently rewritten, house style.

This is the sample-side finding worth the owner's attention, and it confirms
pr/93 §3's warning with numbers:

| | `kine_pio_flag > 0` | actual γ pair (`n_pio_showers ≥ 2`) |
|---|---|---|
| curated nueCC48 (48) | **37** | ~~1~~ |
| curated NCπ⁰ (19) | 14 | ~~2~~ |
| beam mcp1k+mcp2k (1366) | **232** | ~~**4**~~ |

The flag fires on **77 % of the curated *nueCC* arm**. Mechanism (§1.2, shared by
both trees): `id_pi0_with_vertex` has a *kine* branch that fills `kine_pio_*` for
the best pair with **no mass window**, and a separate *pattern* branch that
assigns `pio_id` only inside `(-25, +35) MeV` around 135. So `kine_pio_flag`
means "a pair existed", not "a π⁰ was reconstructed". **Use `pio_id` /
`n_pio_showers`.** Not a port defect — an M15 shared design.

### 6.5 Delivered lists

> ⚠ **List sizes updated by §10.** The files on disk are the corrected ones
> (90 / 46 / 79); the counts in this subsection's prose below are round 1.

| file | n (round 1) | **n (on disk, corrected)** | composition |
|---|---|---|---|
| `docs/pr/pr113-nuecc.index.txt` | 98 | **90** | 48 curated + 42 reco |
| `docs/pr/pr113-ncpi0.index.txt` | 23 | **46** | 19 curated + 27 reco |
| `docs/pr/pr113-numucc-emshower.index.txt` | 79 | **79** | reco, beam arms only |
| `docs/pr/pr113-emshower-sample.tsv` | 1433 rows | 1433 rows | master, 32 columns |

Pairwise overlap between the three lists: **0 / 0 / 0.**

numuCC-EM shape: `em_max` median **195 MeV**, muon length median **87 cm**;
tiers 41 × [100,200) and 38 × ≥200 MeV. Overlap with the standing numu
manifests is small — **10/50** with `scripts/manifests/numu50.txt` and **6/100**
with `scripts/pr112_numu100.txt` — so this is a genuinely complementary sample,
as expected: those manifests were built for vertex/track work, not EM-shower
content.

~~NCπ⁰ reco additions are thin (4) because a true γ pair is rare in the beam pool
(4/1366)~~ **— round-1 text, wrong; the reco additions are 27 and the beam-pool
pair count is 55 (§10).** What survives: `not has_mu` is restrictive on data
(791/1366 carry a ≥30 cm muon-like primary), and the 19 curated events are still
the anchor of that list.

**If a larger π⁰ sample is wanted, the lever is `has_mu`, not the π⁰ predicate.**
Of the 55 beam events carrying a γ pair, the ladder sends exactly **27 to NCπ⁰,
27 to the numuCC rung** (`has_mu`), and drops 1 on the ≥100 MeV floor — so the
muon veto moves **half** the available π⁰ sample off the NCπ⁰ list. They are not
lost to the scan, though: **25 of those 27 are in `pr113-numucc-emshower.index.txt`
already**, which makes that list a π⁰-rich sample in its own right and not only a
numuCC one. Do **not** widen with `kine_pio_flag` — §6.4/§10 is precisely the
measurement disqualifying it; the honest π⁰ columns are `pio_id` /
`n_pio_showers`.

---

## 7. Inherited open items

Re-stated, not re-derived, so nothing silently ages out.

| owner | item | status |
|---|---|---|
| pr/33 §7 loose end 4 | P10's degenerate inputs unmeasured | open |
| pr/33 §7 loose end 5 | `examine_shower_1`/`examine_showers` arithmetic | **still open** (§2.3) |
| pr/33 §11.3 | `PRShower.cxx`↔`WCShower.cxx` uncovered | **partly closed** (§2.1, §2.2, §2.4); read boundary in §0.4 |
| pr/33 P3 second half | π⁰-id disjointness invariant; `ssm_tagger`'s permanently-zero `acc_segment_id` seed | open — no global segment-id allocator to hook |
| pr/33 P12 / pr/91 §8 | determinism on this stage | **CLOSED at container level** for the `PRShower`/`WCShower` pair — P12's address fallback is gone. Method bodies in §0.4 not swept (§2.4) |
| pr/91 §2b | standard-manifest population gate owed for `shower_endpoint_skip_orphan_vtx` (shipped SBND-ON on 24 targeted events; 1 of 20 non-hand-picked moved) | open |
| pr/91 §7 P2/P3 | conn-3 distance-to-charge; late direction-based consolidation | not implemented |
| pr/93 §3 | π⁰/hadronic class deferred as its own trace; `tro_1/2/4/5` census and the per-point track→shower dQ/dx transition scan never attempted; ~37 of pr/40's 42 straight-long e⁻ residuals remain | open |
| pr/46 §9 class (d) | single-segment long-muon exclusion — M15 parity, deliberately untouched | open by design |
| pr/101 §6 | `kine_long_muon_mode` calibrated on 5 long muons; two have an *empty* muon-segment set (nueCC48 30504/235435, NCπ⁰ 399860) | open |
| pr/92 | the stray-satellite fix is **new behaviour, not a port fix** — the prototype has the identical hole (`NeutrinoID_kine.h:209-255`) | recorded so it is not later mistaken for parity work |

**One unrelated discrepancy, mentioned not fixed** (CLAUDE.md §5):
`docs/work-tags.md` marks `work-*-prod0823` KEEP as the pre-flip reference epoch,
but no such directory exists on disk — the previous epoch's reference arm is
gone. Not regenerated (M13).

---

## 8. What is NOT claimed

- **No completeness claim for `PRShower.cxx`↔`WCShower.cxx`.** §0.4 states
  exactly which methods were read for arithmetic and which were not. A numeric
  divergence can still hide in the unread half.
- **No rate for F1 or F2.** Both are structural readings. Neither has an event
  attributed to it, and the reachability of F1(a) is explicitly UNDECIDED.
  Nothing here says how often either fires — that needs a hit counter, which is
  the first step of any follow-up, not this doc.
- **No purity or efficiency claim for the event lists.** They are
  reconstruction-defined on data; there is no truth to measure against (§6.1).
- **No byte-identicality claim, because no code changed.** No gate was run and
  none is owed this round.
- **pr/33's loose end 5 is not closed** (§2.3), and 15 of the in-tree review
  doc's 22 items were not re-checked (§2.2).
- The findings table is **unfiltered**. pr/33's precedent is 14 → 5 after the
  owner's filter; the proposed knobs are candidates, not designs.

---

## 10. Round 2 — a census bug that changed the lists (2026-08-27)

**Found while building the pr/114 display, which re-derives the same π⁰ pairing
independently and disagreed.** Round 1 undercounted π⁰-paired events by **10×**.
Everything in §1–§5 (the audit) is unaffected; §6 (the sample) is corrected here.

### Root cause

`scripts/pr113_topology_census.py` read the pairing with the file's house idiom
for "absent key ⇒ sentinel":

```python
n_pio_showers = sum(1 for sh in showers if (sh.get("pio_id", -1) or -1) >= 0)
```

`pio_id` is allocated **from 0**, and `0` is falsy in Python, so `0 or -1`
evaluates to `-1`: **the first π⁰ group of every event was reported as
unpaired.** Almost every event that has a group at all has group 0, so the
predicate saw almost nothing. The idiom is harmless on the other fields it is
used for — a zero energy or a zero score maps to zero either way — and is left
alone there. It is only ever wrong for a value whose valid domain **includes
zero**, which is exactly `pio_id`.

Fixed by a named helper, `pio_id_of()`, at the three affected sites
(`n_pio_showers`, `lead_pio`, `has_e_nonpio`).

### What the numbers really are

| | round 1 (buggy) | **round 2 (correct)** |
|---|---|---|
| events with a reco γ pair (`n_pio_showers ≥ 2`) | 7 / 1433 | **70 / 1433** |
| …by arm (ncpi0 / nuecc48 / mcp1k / mcp2k) | 2 / 1 / 0 / 4 | **8 / 7 / 23 / 32** |
| events by π⁰ group count | — | 63 have 1, 6 have 2, 1 has 3 |

### What the lists really are

| list | round 1 | **round 2** | change |
|---|---|---|---|
| `pr113-nuecc.index.txt` | 98 | **90** | −8 |
| `pr113-ncpi0.index.txt` | 23 | **46** | +23 |
| `pr113-numucc-emshower.index.txt` | 79 | **79** | **byte-identical** |

The three lists are still disjoint (0/0/0). Two checks that the correction
behaved as it should, and did:

- **numuCC-EM is byte-identical**, as predicted before running: `has_mu` and
  `em_max` never touch `pio_id`, so that arm of the ladder could not move.
- **All 8 events that left nueCC landed in NCπ⁰** — none vanished. That is
  exactly the priority ladder doing its job: once those events are correctly
  seen to carry a γ pair, rung 2 claims them before rung 3 can.
  (mcp1k 172942; mcp2k 165157, 282909, 396222, 47212, 475096, 76346, 76350.)

The remaining +15 NCπ⁰ events were in **no** list in round 1: they have a real γ
pair but no primary e-rooted shower, so the buggy predicate failed them on both
rungs.

### §6.4 restated — the conclusion survives, the margin does not

The `kine_pio_flag` table in §6.4 used the buggy column for its "actual γ pair"
side. Corrected:

| | `kine_pio_flag > 0` | actual γ pair (`n_pio_showers ≥ 2`) |
|---|---|---|
| curated nueCC48 (48) | 37 | ~~1~~ → **7** |
| curated NCπ⁰ (19) | 14 | ~~2~~ → **8** |
| beam mcp1k+mcp2k (1366) | 232 | ~~4~~ → **55** |

**`kine_pio_flag` is still not a π⁰ selector** — 232 firings against 55 real
pairs in the beam pool, and 37 of 48 on the *nueCC* arm — and the mechanism in
§1.2 (a kine branch with no mass window, a pattern branch with one) is unchanged
and was never derived from the buggy column. But the honest ratio is **4:1, not
58:1**, and §6.5's "those are already at their ceiling (4 events in the beam pool
carry a pair at all)" was wrong: the ceiling is 55.

### Lesson

The round-1 number was never sanity-checked against a second, independent path
to the same quantity. It took an unrelated round re-deriving the pairing from
`showers[].pio_id` directly to expose it. **A count that lands surprisingly low
is a hypothesis, not a finding** — "the π⁰ reconstruction almost never fires"
should have been checked against a single event's dump by hand before it was
written down. One `grep pio_id` on ncpi0 evt21073 would have shown two accepted
groups, ids **0** and 1, and the id-0 group missing from the census output.

# 31 — Topology / PID / direction: prototype↔toolkit fidelity audit

**Why.** Fourth in the series after pr/28 (vertex fit + trajectory dQ/dx),
pr/29 (Steiner graph build) and pr/30 (proto-vertex + track-segment finding).
This is step 3 of the eight in doc pr/27 §0: the stage that turns a bare
segment graph into *labelled* particles — track vs shower, which particle,
which way it points. Everything downstream reads these labels. The neutrino
vertex scorer (pr/27 §6) is a pure function of them; the energy sum (§7) is a
pure function of the 4-momenta they write.

**Status.** AUDIT ONLY. No code changed, no patch proposed, no event run. Every
"this changes the output" below is an argument from source, not a measurement.
Fifteen divergences, ranked in §8.

**Headline.** The skeleton is again a faithful port — the four entry points in
order, the fixed 12-call map-repair sequence in order, and every cut constant
paired across eighteen function pairs. What differs is *what gets written when
a label changes*. The prototype's reclassification sites touch particle type
and mass and deliberately leave the 4-momentum alone; the toolkit recomputes
it. That is **P1**, it is unconditional, and it moves reconstructed energy.
**P2** is a second unconditional one: a 300-line direction analysis the
prototype runs once, during main-vertex selection, that the toolkit *also*
runs for every topology shower one stage earlier.

**Which toolkit was read.** Every toolkit anchor is `git show HEAD:<file>` at
**`4f2e7303`** ("clus: sweep all remaining boost::out_edges to
sorted_out_edges"), snapshotted to a scratch directory before reading. pr/30
was written against a working tree another session was editing and every
anchor had to be re-derived at the end; this one is immune by construction.
`git status --short clus/` was empty at snapshot time, so HEAD and the working
tree agreed anyway — but the anchors are HEAD's.

---

## Repro

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
cd $TK && git rev-parse --short HEAD        # 4f2e7303

# snapshot the audited toolkit files (this is what the anchors index)
S=$(mktemp -d /home/xqian/tmp/pr31.XXXX)
for f in clus/src/NeutrinoTrackShowerSep.cxx clus/src/PRSegmentFunctions.cxx \
         clus/inc/WireCellClus/PRSegmentFunctions.h; do
  git show HEAD:$f > $S/$(basename $f); done

# P1 — the dropped 4-momentum guard
grep -n "get_particle_4mom(3)>0" $TK/prototype_base/pid/src/NeutrinoID_track_shower.h | wc -l   # 11
grep -c "segment_cal_4mom" $S/NeutrinoTrackShowerSep.cxx                                        # 14
grep -rn "particle_4mom(3)" $TK/clus/src/                       # 1 hit, commented out, out of scope

# P2 — the extra shower-direction call site
grep -rn "determine_shower_direction()" $TK/prototype_base/pid/src/          # 1 call site: :1532
grep -rn "segment_determine_shower_direction(" $TK/clus/src/                 # 2 call sites

# the constant histogram used for triage (§2.3)
dump(){ sed -n "$2,$3p" "$1" | sed 's://.*::' \
   | grep -o '[0-9][0-9.eE+]*\s*\*\s*units::[a-zA-Z]*\|[<>=!]=\?\s*-\?[0-9][0-9.eE+]*\|[0-9][0-9.]*e[0-9]*' \
   | tr -d ' ' | sort | uniq -c | sort -rn; }
diff <(dump $TK/prototype_base/pid/src/ProtoSegment.cxx 319 542) \
     <(dump $S/PRSegmentFunctions.cxx 2513 2906)          # is_shower_topology

# the SBND operating point (§4)
grep -n "mip_dqdx\|shower_topo_demote_len\|iso_endpoint =" \
     $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
grep -n "dir_weak_use_score=\|mip_dqdx_median=\|proton_dir_vote=\|endpoint_trim_retry=" \
     $TK/cfg/pgrapher/experiment/sbnd/clus.jsonnet | head
```

Prototype line numbers are stable (`prototype_base/` is read-only).

---

## §0 Scope

### The file fan-out

One prototype file plus one prototype class map onto two toolkit files.

| stage-3 piece | prototype | toolkit |
|---|---|---|
| `clustering_points` | `NeutrinoID.cxx:1012` → `PR3DCluster_point_clustering.h` | `NeutrinoTrackShowerSep.cxx:11` → `PRSegmentFunctions.cxx:1890` |
| `separate_track_shower` (per cluster) | `NeutrinoID_track_shower.h:1` | `NeutrinoTrackShowerSep.cxx:37` |
| `separate_track_shower` (per point) | `NeutrinoID_track_shower.h:28` | `NeutrinoPatternBase.cxx:2156-2211` (see §5.2) |
| `determine_direction` | `:66` | `:68` |
| `shower_determin(g)_in_main_cluster` | `NeutrinoID_shower_clustering.h:35-100` | `NeutrinoTrackShowerSep.cxx:1877` |
| `examine_good_tracks` | `NeutrinoID_track_shower.h:213` | `:330` |
| `fix_maps_multiple_tracks_in` | `:766` | `:443` |
| `fix_maps_shower_in_track_out` | `:800` | `:508` |
| `improve_maps_one_in` | `:102` | `:577` |
| `improve_maps_shower_in_track_out` | `:883` | `:684` |
| `improve_maps_no_dir_tracks` | `:322` | `:828` |
| `improve_maps_multiple_tracks_in` | `:836` | `:1303` |
| `judge_no_dir_tracks_close_to_showers` | `:280` | `:1392` |
| `examine_maps` | `:160` / `:167` / `:209` | `:1470` |
| `examine_all_showers` | `:1007` | `:1549` |
| `calculate_num_daughter_showers` | `:688` | `:164` |
| `calculate_num_daughter_tracks` | `:724` | `:221` |
| `find_cont_muon_segment_nue` | `:2372` | `:276` |
| `is_shower_topology` | `ProtoSegment.cxx:319` | `PRSegmentFunctions.cxx:2513` |
| `is_shower_trajectory` | `:543` | `:984` |
| `determine_shower_direction` | `:72` | `:2208` |
| `determine_dir_track` | `:1516` | `:1637` |
| `determine_dir_shower_trajectory` | `:1647` | `:1829` |
| `determine_dir_shower_topology` | `:1677` | **inlined**, `NeutrinoTrackShowerSep.cxx:121-135` |
| `do_track_pid` | `:1201` | `:1418` |
| `do_track_comp` | `:1120` | `:1307` |
| `eval_ks_ratio` | `:1189` | `:973` |
| `is_dir_weak` | `:1291` | `:1080` |
| `cal_4mom` | `:1420` | `:1604` |
| `cal_kine_dQdx` / `cal_kine_range` | `:1316` / `:1380` | `:1214` / `:1387` |
| `cal_dir_3vector` ×3 | `:1450` / `:1469` / `:1491` | `:1101` / `:1141` / `:1170` |

### Explicitly out of scope, and NOT claimed clean

The prototype file `NeutrinoID_track_shower.h` **physically contains stage 4**.
Everything from `determine_main_vertex:1249` to `find_cont_muon_segment:2304`
— `examine_main_vertex_candidate:1400`, `compare_main_vertices_all_showers:1451`,
`compare_main_vertices:1589`, `calc_conflict_maps:1725`, `examine_direction:1876`
— is doc pr/27 §6, not §5, and **was not audited**. In-scope prototype ranges
are `:1–1248` and `:2372–2441`. The toolkit counterparts live in
`NeutrinoVertexFinder.cxx`, also unaudited.

Also out of scope: `change_daughter_type` (prototype `:654`, toolkit
`NeutrinoVertexFinder.cxx:2719` — it *is* ported, and both of its external call
sites are stage-4); `WCShower` / shower clustering; the BDT taggers; the
`clustering_points_segments` 2-D association machinery (`PRSegmentFunctions.cxx:1890-2207`,
317 lines) beyond confirming it exists and is called.

---

## §1 Trust tiers

Carried from pr/28 §3b and pr/29.

**Tier A — read line by line in both trees.** `is_shower_topology`,
`do_track_comp`, `do_track_pid`, `determine_dir_track`,
`determine_dir_shower_trajectory`, `determine_dir_shower_topology`,
`separate_track_shower`, `determine_direction`, `is_dir_weak`, `cal_4mom`,
`examine_good_tracks`, `judge_no_dir_tracks_close_to_showers`,
`find_cont_muon_segment_nue`, `segment_median_dQ_dx`, and the counting
prologue of `examine_all_showers`.

**Tier B — signature, defaults, constant histogram and structure compared;
branch bodies sampled, not fully read.** `improve_maps_no_dir_tracks` (331 vs
474 lines — the largest pair), `improve_maps_shower_in_track_out`,
`improve_maps_multiple_tracks_in`, `improve_maps_one_in`, `fix_maps_*`,
`examine_maps`, `calculate_num_daughter_*`, the decision ladder of
`examine_all_showers` after its prologue, `is_shower_trajectory`,
`determine_shower_direction`, `cal_kine_*`, `cal_dir_3vector`,
`clustering_points_segments`.

Tier B findings are marked. A Tier-B "matches" means the constants, the branch
count and the control-flow shape agree — not that every line was compared.

---

## §2 What matches

### 2.1 The entry-point order

`TaggerCheckNeutrino.cxx:571-581` runs, per cluster:
`clustering_points` → `separate_track_shower` → `determine_direction` →
`shower_determining_in_main_cluster`, and repeats the same four per companion
cluster at `:605-608` and `:621-624`. That is the prototype's order.

### 2.2 The fixed 12-call map-repair sequence

`shower_determining_in_main_cluster` (toolkit `:1877-1980`) issues the same
twelve calls in the same order as the prototype's
`shower_determing_in_main_cluster` (`NeutrinoID_shower_clustering.h:35-100`,
prototype spelling), including the two repeats with different arguments:

| # | call | note |
|---|---|---|
| 1 | `examine_good_tracks` | |
| 2 | `fix_maps_multiple_tracks_in` | |
| 3 | `fix_maps_shower_in_track_out` | |
| 4 | `improve_maps_one_in` | |
| 5 | `improve_maps_shower_in_track_out` | default `flag_strong_check=true` |
| 6 | `improve_maps_no_dir_tracks` | |
| 7 | `improve_maps_shower_in_track_out(…, false)` | second call, weak check |
| 8 | `improve_maps_multiple_tracks_in` | |
| 9 | `fix_maps_shower_in_track_out` | second call |
| 10 | `judge_no_dir_tracks_close_to_showers` | |
| 11 | `examine_maps` | |
| 12 | `examine_all_showers` | |

Order matters here — each call consumes the in/out maps the previous one
repaired — and it is right.

### 2.3 The constants

An automated histogram (Repro block) over eighteen function pairs extracts
every `N*units::cm`, every bare comparison threshold and every `Ne3` literal
in each range and diffs them. Results:

* **`is_shower_topology`** — all five acceptance branches identical:
  `(>0.7 cm, >0.2·tel, tel∈(3,15) cm, (>2.7 cm ‖ >0.35·tel))`,
  `(>0.8, >0.3, tel≥15)`, `(>0.8, >8 cm, >0.18)`, `(>1.0, >0.4)`,
  `(>1.0, >5 cm, >0.23)`; the `0.4 cm` spread cut, the `7.5°` drift-alignment
  bypass, the `1.1×` forward/backward direction margin and the `50 cm /
  0.25·L` demotion guard (`ProtoSegment.cxx:529-532`) all match. The toolkit's apparent duplicates are the
  `WCT_SHOWER_TOPO_DEBUG` mirror of the same ladder (`:2684-2694`), which is
  env-gated and writes nothing.
* **`is_shower_trajectory`** — identical (`10 cm` step, `>0.75`, `<0.95/0.97`,
  `<2.03/2.06`, `>50`, `<60`, `>=0.5`).
* **`eval_ks_ratio`, `cal_4mom`** — byte-identical histograms.
* **`determine_dir_track`** — this pair was read in full rather than
  histogrammed, and every constant matches: `1.75×`, `1.2×`, `1.5×` MIP,
  `4 cm`, `1.5 cm`, `10 cm`, `score>1.0`, `score<100`, `score=200`, `<0.15`,
  `npoints>=15`, the `35/15 cm` compare ranges and the `1 cm` offset. So does
  the branch order.
* **`find_cont_muon_segment_nue`** — `12.5°`, `15°`, `6 cm`, `30 cm`, `1.3×`.
* **the twelve map-repair helpers** — every numeric threshold matches. The
  only histogram differences are idiom: the prototype's `get_flag_shower()`
  expands inline to `kShowerTrajectory ‖ kShowerTopology ‖ |pdg|==11`, and
  `map_vertex_segments[v].size()==1` becomes `boost::degree(...) <= 1`.

The one genuine constant difference found anywhere: none. Every apparent extra
resolved to an inlined predicate or a commented-out prototype line.

### 2.4 The KS machinery

`WireCell::kslike_compare` (`util/src/KSTest.cxx:216-238`) is an exact
reimplementation of ROOT's `TH1::KolmogorovTest(h, "M")` for equal-binning
unweighted histograms: normalise each vector by its own sum, walk both
cumulatively, return the maximum absolute difference. The prototype calls
`h2->KolmogorovTest(h1,"M")` (reference→data); the toolkit calls
`kslike_compare(vec_y, ref)` (data→reference). The max-distance statistic is
symmetric under that swap, so the argument order is not a divergence.

### 2.5 The default arguments

Sixteen defaulted parameters across `PRSegmentFunctions.h` were compared
against the prototype's hard-coded equivalents. All agree:
`step_size=10 cm`, `mip_dQ_dx=50000/cm`, `MIP_dQ_dx=43000/cm`,
`compare_range=35 cm`, `offset_length=0`, `flag_force=false`,
`rms_cut=0.4 cm`, `search_range=1.2 cm`, `scaling_2d=0.7`,
`cloud_name="associate_points"`, `TrackPidOptions{mip_dqdx=50000/cm,
proton_dir_vote=false, 0.25, 1.3, endpoint_trim_retry=false, start_n=1,
end_n=1}`. `demote_len` defaults to `0` (off).

---

## §3 Divergences

### P1 — the 4-momentum guard is dropped at 11 of 13 map-repair sites

**Tier A. Unconditional. Changes reconstructed energy. Escalation rule 1.**

Every prototype "reclassify this segment to an electron" site in the map-repair
family writes exactly three things:

```cpp
// NeutrinoID_track_shower.h:372-374 (identical shape at :407 :439 :462
//                                    :508 :541 :578 :637 :873 :934 :951)
sg->set_particle_type(11);
sg->set_particle_mass(mp.get_mass_electron());
if (sg->get_particle_4mom(3)>0) sg->cal_4mom();
```

The guard is the load-bearing part. In the prototype `particle_4mom[3]` is the
**energy** component (`ProtoSegment.cxx:1437-1444`, `particle_4mom[3] = kine_energy +
particle_mass`), so `get_particle_4mom(3) > 0` reads "this segment already has
a computed 4-momentum". The rule is therefore: *if the segment already had an
energy, recompute it under the new particle hypothesis; if it never had one,
leave it at zero.* Eleven of the twelve in-scope prototype `cal_4mom()` call
sites carry it. The twelfth (`:147`, in `improve_maps_one_in`) does not.

The toolkit has fourteen `segment_cal_4mom` call sites — one in the topology
branch (that is P3) and thirteen in the map-repair family. Eleven of the
thirteen carry no guard:

| toolkit site | context | guard? |
|---|---|---|
| `NeutrinoTrackShowerSep.cxx:662` | `improve_maps_one_in` | `if (sg->has_particle_info())` — **stricter** than the prototype's unguarded `:147` |
| `:768` | `improve_maps_shower_in_track_out`, out-track reclassification | none |
| `:793` | same, `!is_shower1` branch | none |
| `:807` | same, shower branch | `if (is_shower1 && has_particle_info() && energy() > 0)` — **the guard, correctly ported**, with a comment naming the prototype behaviour |
| `:938 :980 :1022 :1048 :1111 :1164 :1207 :1286` | `improve_maps_no_dir_tracks`, eight reclassification branches | none |
| `:1374` | `improve_maps_multiple_tracks_in` | none |

Bucketed by function the two lists reconcile exactly, so no site is missing on
either side:

| function | prototype `cal_4mom()` | toolkit `segment_cal_4mom` |
|---|---|---|
| `improve_maps_one_in` | 1, **unguarded** (`:147`) | 1, guarded (`:662`) |
| `improve_maps_shower_in_track_out` | 2, guarded (`:934 :951`) | 3 (`:768 :793` unguarded, `:807` guarded) — `:793`+`:807` jointly port `:951` |
| `improve_maps_no_dir_tracks` | 8, guarded (`:374 :407 :439 :462 :508 :541 :578 :637`) | 8, all unguarded |
| `improve_maps_multiple_tracks_in` | 1, guarded (`:873`) | 1, unguarded (`:1374`) |

So the porter clearly saw the guard — `:807` reproduces it and its comment
paraphrases it — and it is absent everywhere else. Corroborating: the only
surviving textual trace of `get_particle_4mom(3)` anywhere in `clus/` is a
**commented-out** line, `NeutrinoShowerClustering.cxx:2771`.

**Mechanism.** Consider a short pdg-0 track with no direction — the common
input to these repairs. In the prototype it has `particle_4mom = (0,0,0,0)`,
the guard fails, and it stays at zero after reclassification. In the toolkit,
`segment_cal_4mom(sg, 11, …)` runs unconditionally: for a segment under 4 cm
or a trajectory shower it integrates `cal_kine_dQdx`, otherwise it evaluates
the electron range function at the segment length, and writes the result into
a fresh `ParticleInfo`. The segment acquires a kinetic energy the prototype
leaves at zero.

**Reach not measured.** How many segments per event take a reclassification
branch, and what the summed energy difference is, is not established here —
that needs a run. What is source-certain is that the branch is unconditional
(no knob, no default) and that it feeds `kine_energy_particle` through
`NeutrinoKinematics.cxx:104-113`, which maps the prototype's `kenergy_best`
onto `particle_info()->kinetic_energy()`.

**Structural caveat, stated because it matters for any fix.** The toolkit
cannot simply "not call `cal_4mom`": `particle_info(pinfo)` replaces the whole
`ParticleInfo`, so declining to compute a 4-momentum means constructing one
some other way. The prototype could leave a member alone; the toolkit's data
model does not offer that without explicitly copying the old 4-momentum
forward. This is a design consequence, not an oversight — but it is a
behavioural divergence either way.


### P2 — a 300-line direction analysis runs at a toolkit-only call site

**Tier A. Unconditional. Escalation rule 1.**

The prototype calls `ProtoSegment::determine_shower_direction()` from exactly
one place in the entire tree:

```
$ grep -rn "determine_shower_direction()" prototype_base/pid/src/
ProtoSegment.cxx:72:            bool WCPPID::ProtoSegment::determine_shower_direction(){
NeutrinoID_track_shower.h:1532:   tmp_sg->determine_shower_direction();
```

`:1532` is inside `compare_main_vertices_all_showers` — **stage 4**,
main-vertex selection, and only on the all-showers path.

The toolkit calls its port from two places:

```
clus/src/NeutrinoVertexFinder.cxx:506   # inside compare_main_vertices_all_showers — the faithful one
clus/src/NeutrinoTrackShowerSep.cxx:123 # inside determine_direction — stage 3, no prototype counterpart
```

What the prototype's stage-3 topology branch actually does is
`determine_dir_shower_topology` (`ProtoSegment.cxx:1677-1710`), and after the
commented-out blocks are removed it is four lines: set `particle_type = 11`,
set `particle_mass = electron`, and print. **It does not touch `flag_dir`.**
The direction of a topology shower at the end of stage 3 in the prototype is
whatever `is_shower_topology`'s forward/backward large-spread comparison left
there (`ProtoSegment.cxx:523-527`).

The toolkit's inlined branch (`NeutrinoTrackShowerSep.cxx:121-135`) runs
`segment_determine_shower_direction` first — 305 lines of associated-point PCA,
spread profiling and endpoint comparison (`PRSegmentFunctions.cxx:2208-2512`)
— which sets `dirsign` on its own terms, overwriting what
`segment_is_shower_topology` decided.

So every `kShowerTopology` segment leaves stage 3 with a direction the
prototype would only have computed later, and only on the all-showers path.
Since `dirsign` is what §6's vertex scorer and §5d's in/out maps read, this
propagates immediately.

### P3 — the topology branch also writes a 4-momentum and a score

**Tier A. Unconditional. Same site as P2.**

Beyond P2's extra call, `NeutrinoTrackShowerSep.cxx:124-135` adds two writes
that `determine_dir_shower_topology` does not make:

* `segment_cal_4mom(seg, 11, …)` → a full `ParticleInfo` with kinetic energy.
  The prototype sets `particle_type` and `particle_mass` and leaves
  `particle_4mom` untouched. Same mechanism as P1, at a different site.
* `seg->particle_score(100.0)`. The prototype leaves `particle_score` at
  whatever the previous pass wrote. `100` is the toolkit's "PID not performed"
  sentinel, and `examine_all_showers` reads it: `if (sg->particle_score() != 100)
  tracks_score += …` (`:1586`, matching prototype `:1040`). In the prototype
  the same segment may carry a stale non-100 score into that sum.

### P4 — three reclassification sites write a rest-mass-only 4-momentum

**Tier A. Unconditional.**

Distinct from P1: at three sites the toolkit does not call `segment_cal_4mom`
at all but constructs a 4-momentum of `(mass, 0, 0, 0)` — zero kinetic energy,
zero momentum:

| toolkit site | function | prototype counterpart |
|---|---|---|
| `NeutrinoTrackShowerSep.cxx:428` | `examine_good_tracks` | `:257-261` — sets type, mass, `flag_dir=0`, `dir_weak=1`; **no** 4-momentum write |
| `:1463` | `judge_no_dir_tracks_close_to_showers` | `:313-315` — sets type and mass only |
| `:1867` | `examine_all_showers` final sweep | `:1238-1239` — sets type and mass only |

The prototype leaves the previous `particle_4mom` in place; the toolkit
overwrites it with rest mass. For a segment that never had one this is a wash
in energy but not in sign convention (the prototype's stale all-zero
4-momentum yields `E − m = −m` if anyone subtracts). For a segment that *did*
have a computed energy — reachable at `examine_all_showers`'s sweep, which
runs last — the toolkit discards it and the prototype keeps it. Opposite
directions from P1, same class of decision.

**This is an M15 case.** The prototype's behaviour here is arguably a bug (a
4-momentum computed under a superseded hypothesis), and the toolkit's is
arguably a bug (an energy thrown away). Both readings stated; not picked.

### P5 — pr/30's P5 has a confirmed order-sensitive consumer in this stage

**Tier A. Unconditional.**

pr/30 P5 recorded that `find_vertices(seg)` returns its pair ordered by vertex
**id** in the prototype (`NeutrinoID_proto_vertex.h:3227-3243`) and by
**proximity to the segment's first fit point** in the toolkit
(`PRGraph.cxx:105-141`), and explicitly did not count the callers that care.
Here is one that does.

`examine_all_showers` (`NeutrinoTrackShowerSep.cxx:1595-1605`, prototype
`:1049-1061`) computes daughter-shower counts at *both* vertices and then
applies **asymmetric** acceptance:

```
first  vertex branch:  max_angle > 165  ||  (max_angle > 150 && length_good_tracks < 3 cm
                                             && length_good_tracks < 0.1*length_showers)
second vertex branch:  max_angle > 165
```

Both sides verified: prototype `:1073` vs `:1090`, toolkit
`NeutrinoTrackShowerSep.cxx:1625` vs `:1649`. The relaxed 150° clause exists
only on the branch keyed by the daughter count at `.first` (whose angles are
measured at `.second`); the mirror branch carries only the 165° test. Swap
which physical vertex lands in `.first` and a segment at 155° flips between
reclassified and not.

The two orderings are each reproducible run to run — which is what the
determinism gates test — but they are not the same ordering.

### P6 — `find_cont_muon_segment_nue` hoists `dir3` and changes its condition

**Tier A. Unconditional.**

Prototype (`NeutrinoID_track_shower.h:2402-2408`), inside the per-neighbour
loop:

```cpp
if (length > 30*units::cm || sg_length > 30*units::cm){
  TVector3 dir3 = sg ->cal_dir_3vector(vtx->get_fit_pt(), 30*units::cm);
  TVector3 dir4 = sg2->cal_dir_3vector(vtx->get_fit_pt(), 30*units::cm);
  angle1 = (3.1415926 - dir3.Angle(dir4))/3.1415926*180.;
}
```

Toolkit (`NeutrinoTrackShowerSep.cxx:286-311`), `dir3` hoisted above the loop:

```cpp
WireCell::Vector dir3 = (sg_length > 30*units::cm)
                          ? segment_cal_dir_3vector(sg, vtx_pt, 30*units::cm)
                          : dir1;                       // <-- 15 cm direction
...
if (length > 30*units::cm || sg_length > 30*units::cm) {
    WireCell::Vector dir4 = segment_cal_dir_3vector(sg2, vtx_pt, 30*units::cm);
    angle1 = (M_PI - dir3.angle(dir4))/M_PI*180.0;      // <-- may be 15 cm vs 30 cm
}
```

The reachable case: **short reference segment, long neighbour**
(`sg_length ≤ 30 cm < length`). The prototype compares two 30 cm directions;
the toolkit compares a 15 cm direction against a 30 cm one. `angle1` differs,
and it is one of the three alternatives in the `< 12.5°` continuation test —
this function is what decides whether a muon continues through a vertex.

### P7 — the empty-window early return declares a direction

**Tier A. Unconditional.**

`do_track_comp` (`PRSegmentFunctions.cxx:1334-1336`):

```cpp
if (ncount == 0) {
    return {1.0, 1e9, 1e9, 1e9};
}
```

Element 0 is the direction metric; `segment_do_track_pid` reads it as
`flag_forward = round(result_forward.at(0))`, so **1.0 means "this orientation
passes the direction gate"**. When no sample falls in the comparison window in
either orientation, both flags are 1, the both-pass branch runs, and the
segment leaves with `flag_dir = -1`, `pdg = 13` (muon) and `particle_score =
1e9`.

The prototype has no guard: it constructs `TH1F("h1","h1",0,0,0)` and calls
`KolmogorovTest` on empty histograms. The resulting value is ROOT-dependent
and was not reproduced here, but the prototype's `eval_ks_ratio` returns
`false` on `ks1 - ks2 >= 0.0`, which is what all-zero inputs give — i.e. the
prototype's degenerate answer is *abstain*, the toolkit's is *confirm*.

The toolkit's guard is defensible engineering; the value it returns for
element 0 is the permissive one. `0.0` would abstain and match.

### P8 — the median dQ/dx is taken over a different sample set

**Tier A. Unconditional.**

`determine_dir_track` needs a median dQ/dx for its three fallback branches
(`pdg==0` recovery, and the two `length < 1.5 cm` vertex-activity branches).

Prototype (`ProtoSegment.cxx:1572-1576`) takes it over **the local `dQ_dx`
vector it just built** — exactly the trimmed range `[start_n1, end_n1]`, one
entry per point, no filtering, with `dQ/(dx/units::cm + 1e-9)` for every point
including `dx == 0`.

Toolkit (`PRSegmentFunctions.cxx:1743`) calls
`segment_median_dQ_dx(segment, start_n1, end_n1)`, which rebuilds the sample
from `segment->fits()` and **filters** (`:890`):

```cpp
if (fit.valid() && fit.dx > 0 && fit.dQ >= 0) vec_dQ_dx.push_back(...);
```

Two consequences. First, the vector length differs whenever any fit in the
range is invalid, has `dx <= 0` or has negative `dQ`, and `nth_element` at
`size()/2` therefore selects a different order statistic. Second, the toolkit
also leaves `dQ_dx[i] = 0` for `dx <= 0` in the vector it hands to the PID
(`:1685-1687`), where the prototype writes `dQ/1e-9` — a huge value. The two
trees push the same pathological point to opposite ends of the distribution.

### P9 — `judge_no_dir_tracks_close_to_showers` queries different points

**Tier A. Unconditional.**

Prototype (`:298`) iterates `sg->get_point_vec()` — the segment's path points.
Toolkit (`:1419`) iterates `sg->fits()` — the fitted trajectory — with the
prototype's accessor left in the source as a comment:

```cpp
const auto& pts = sg->fits();//wcpts();
```

Neither `fits()` nor `wcpts()` is `point_vec`. The decision — "is every point
of this direction-less track within 0.6 cm of a shower in all three planes" —
is taken over a different, and generally differently-sampled, point set.

### P10 — the null-vertex path diverges

**Tier A. Unconditional.**

Prototype `determine_direction` (`:79-81`) prints an error and **falls
through**, then evaluates `map_vertex_segments[start_v].size()` with
`start_v == nullptr` — which default-constructs an empty set and yields `0`.
So the segment is still processed, with `start_n` or `end_n` equal to 0.

Toolkit (`:84-87`) logs and `continue`s: the segment gets no direction, no
PID, no particle info at all in this pass.

The prototype's behaviour also inserts a null key into `map_vertex_segments`,
which is its own problem. Both readings; not picked.

### P11 — start/end vertex assignment is geometric, not topological

**Tier A. Unconditional.**

Prototype `determine_direction` (`:76-77`) matches **wcpt indices**:

```cpp
if ((*it)->get_wcpt().index == sg->get_wcpt_vec().front().index) start_v = *it;
if ((*it)->get_wcpt().index == sg->get_wcpt_vec().back().index)  end_v   = *it;
```

Toolkit (`:97-102`) takes `find_vertices`'s pair and swaps on 3-D distance:

```cpp
if (ray_length(Ray{start_v->wcpt().point, front_pt}) >
    ray_length(Ray{start_v->wcpt().point, back_pt})) std::swap(start_v, end_v);
```

Index equality is exact and topological; distance comparison is approximate
and geometric. They agree whenever the two vertices sit on the segment's two
ends, which is the intended invariant — the prototype's error branch exists
precisely because the invariant can fail, and where it fails the toolkit
silently picks the nearer one. `examine_good_tracks` (`:359-381`) repeats the
same substitution.

### P12 — vertex degree comes from `boost::degree`, not from a segment set

**Tier B. Unconditional.**

`start_n` / `end_n` are the prototype's `map_vertex_segments[v].size()` — the
number of *distinct segments* at a vertex — and the toolkit's
`boost::degree(vd, graph)` — the number of *incident edges*. For an undirected
boost graph a self-loop contributes **2** to the degree and **1** to a
`std::set<ProtoSegment*>`. `start_n`/`end_n` gate the `== 1` free-end tests
that appear throughout `determine_dir_track` and `determine_dir_shower_trajectory`,
so a self-loop segment is scored differently. Whether the PR graph can carry a
self-loop at this stage was not established.

### P13 — `kShowerTopology` is set but never cleared

**Tier A. Unconditional.**

Prototype `is_shower_topology` opens with `flag_shower_topology = tmp_val;` —
an **assignment**, so calling it with the default `tmp_val = false` clears a
previously-set flag, and the `50 cm / 0.25·L` guard at `:534` can also set it
back to false.

Toolkit `segment_is_shower_topology` ends with (`:2901`):

```cpp
if (flag_shower_topology) segment->set_flags(SegmentFlags::kShowerTopology);
```

— set-only. A segment that was a topology shower in an earlier pass and is not
one now keeps the flag. Related: the prototype also sets `flag_dir = 0` before
its four early returns (`:321`, `:329`), so a segment whose associated cloud is
empty is left undirected; the toolkit's four early returns (`:2518 :2522 :2526
:2529`) happen before any mutation and skip `segment->dirsign(flag_dir)`
entirely, leaving the previous direction in place.

Whether stage 3 ever runs `separate_track_shower` twice on the same segment
was not established — see §7.

### P14 — the PID persistence gate (doc pr/7), re-confirmed at HEAD

**Tier A. Unconditional. Already on record.**

`determine_dir_track` keeps `particle_type` and `particle_score` as *members*
in the prototype, written unconditionally by `do_track_pid`, and gates only
`cal_4mom()` on the direction pointing at a free end (`ProtoSegment.cxx:1637-1639`).
The toolkit keeps them as *locals* and writes them into the segment only
inside that same gate (`PRSegmentFunctions.cxx:1797-1812`), so a track whose
direction does not point at a free end **loses its PID entirely**.

This is doc `pr/7 — track-PID persistence divergence`, status DIAGNOSED, fix
proposed and not implemented. Confirmed still present at `4f2e7303`. Two
knock-on effects visible from this audit:

* `determine_dir_shower_trajectory`'s else-branch (`:1850-1861`) tests
  `segment->particle_info()->pdg() != 11` where the prototype tests the member
  `particle_type != 11`. When the persistence gate suppressed the write, the
  toolkit sees absent/stale info and zeroes the direction where the prototype
  may keep it.
* it also drives the toolkit's failure-path score of `100.0`
  (`segment_do_track_pid:1600`) versus the prototype's `0`
  (`ProtoSegment.cxx:1284-1286`), which §3-P3 shows is a live sentinel in
  `examine_all_showers`.

### P15 — three raw `boost::edges` loops remain in the audited files

**Determinism. Judged benign; see §6.**

---

## §4 Knobs already on record — and which are live in SBND

pr/30's §4 could treat "default OFF" as "dormant". That is not true for this
stage: the SBND production PR job turns most of them on. The split matters, so
it is drawn explicitly.

**Live in SBND production** (`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`
for the first three, `cfg/pgrapher/experiment/sbnd/clus.jsonnet` `pr()`
defaults for the rest — the job does not override them):

| knob | C++ default | SBND | doc | effect in this stage |
|---|---|---|---|---|
| `mip_dqdx` | 50000/cm | **56000** | 48 | the flat MIP template amplitude in `do_track_comp`. The prototype hard-codes `50e3` (`ProtoSegment.cxx:1152`). SBND compares against a template the prototype never used. |
| `mip_dqdx_median` | 43000/cm | **48000** | pr/8 | replaces the prototype's hard-coded `43e3` in `is_shower_topology`'s per-point normalisation, `determine_dir_track`'s three median branches (`ProtoSegment.cxx:1577-1580`, `:1600-1613`), `find_cont_muon_segment_nue`'s ratio, and ~8 sites in `improve_maps_no_dir_tracks`. |
| `shower_topo_demote_len` | 0 (off) | **50 cm** | pr/25 §3 | unconditional demotion of long `kShowerTopology` segments to tracks (`PRSegmentFunctions.cxx:2885-2889`). No prototype counterpart. |
| `dir_weak_use_score` | false | **true** | pr/6 | routes `seg_dir_weak()` to `segment_is_dir_weak()` — **the prototype's semantics**. SBND is closer to the prototype than the C++ default is. |
| `proton_dir_vote` (+ `score_max=0.25`, `asym_min=1.3`) | false | **true** | pr/8 | beyond the prototype: lets the proton template declare a direction where the muon-vs-flat gate abstains (`:1581-1595`). |
| `endpoint_trim_retry` | false | **true** | pr/9 §6 F1 | beyond the prototype: retries the PID once with one sample trimmed at the hypothesised stop (`:1531-1566`, `skip_stop_samples` in `do_track_comp`). |
| `iso_endpoint` | false | **true** | pr/24 | stage 2, but it changes the endpoints this stage's directions are measured from. |

For the "beyond the prototype" rows, the byte-identical-when-off guarantee is
real and was verified when each shipped — but SBND does not run them off, so
prototype parity is not what SBND executes. That is the owner's deliberate
operating point, recorded here so the divergence list is not read as "SBND
matches the prototype except for §3".

**Dormant** (default off and off in SBND): none found in this stage. Every
stage-3 knob in `NeutrinoPatternBase.h` is enabled in the SBND job.

**Also worth naming:** `WCT_SHOWER_TOPO_DEBUG` (`PRSegmentFunctions.cxx:2543`)
is an environment-gated diagnostic that writes nothing and is read once into a
`static const bool`. Not a behaviour knob.

---

## §5 Looks like a divergence and is not

Recorded so the next reader does not re-derive them.

**5.1 `vec_dQ_dx` in `is_shower_topology` is dead in both trees.** Filled at
prototype `:403` and toolkit `:2636`, and thereafter used only as
`vec_dQ_dx.size()` — a loop bound identical to `vec_rms_vals.size()`. Faithful
port of a dead computation. Do not "clean it up" on one side alone; it is the
same shape as pr/30 §5.1's never-incremented `count`.

**5.2 The no-argument `separate_track_shower()` is not missing.** The prototype
has two overloads: `:1` per cluster (ported to `NeutrinoTrackShowerSep.cxx:37`)
and `:28`, which propagates the shower flag down to the cluster's per-point
`point_flag_showers` array. The second has a toolkit counterpart —
`NeutrinoPatternBase.cxx:2156-2211` writes a `point_flag_shower` array into the
default point cloud, read back at `Facade_Cluster.cxx:896`. Different file,
same product.

**5.3 The 4-vector component order differs and it is consistent.** Prototype
`particle_4mom[3]` is the energy and `[0..2]` the momentum
(`ProtoSegment.cxx:1437-1443`); toolkit `D4Vector` puts energy at index 0
(`util/inc/WireCellUtil/D4Vector.h:103`, `e() { return m_v[0]; }`) and
`segment_cal_4mom` fills it that way (`:1626-1631`). Checked: no code in
`clus/` reads index 3 as an energy.

**5.4 `segment_cal_4mom`'s `MIP_dQdx` parameter is unused.** It is accepted at
`:1604` and never referenced in the body — the energy path goes through
`segment_cal_kine_dQdx(segment, recomb_model)` or `cal_kine_range`. So the fact
that call sites disagree about it (`:126` passes `m_mip_dqdx_median`, the
eleven map-repair sites pass `m_mip_dqdx`) has **no** effect today. It will the
moment anyone uses the parameter. Listed again in §7.

**5.5 `kslike_compare`'s argument order.** Symmetric statistic — see §2.4.

**5.6 The early `break` in `judge_no_dir_tracks_close_to_showers`.** The
toolkit breaks out of the point loop on the first failing point (`:1429`,
`:1451`) where the prototype scans all of them. `flag_change` is only ever set
`false`, so the result is identical. Pure optimisation.

**5.7 `find_cont_muon_segment_nue`'s dropped bookkeeping.** `max_angle`,
`max_ratio`, `max_ratio1`, `max_ratio1_length` and `flag_cont` exist in the
prototype (`:2375-2383`) and are consumed only by commented-out code
(`:2432-2434`). The toolkit's `return {sg1, vtx1}` is equivalent to the
prototype's `flag_cont` dance: when no candidate passes, both return a null
pair.

**5.8 `min_para_angle = 1e9`.** Both trees leave it at the sentinel when the
end vertex has degree 1, and both then fail the `< 15` test. Match.

**5.9 The threshold-run restructuring in `is_shower_topology`.** The prototype's
`if (threshold_segs.size()==0 && rms>0.4) … else if (rms>0.4) …` becomes the
toolkit's `if (rms>0.4) { if (empty) … else … }`. Identical semantics. Likewise
the prototype's `for (int i=start_n; i!=end_n; i++/--)` becomes `i < end_n` /
`i > end_n` with a bounds guard; `start_n ≤ end_n` (forward) and
`start_n ≥ end_n` (backward) hold by construction, so `!=` and the inequality
agree.

**5.10 `is_dir_weak` is a faithful port.** The prototype's four-way `||` chain
(`ProtoSegment.cxx:1295-1299`) is refactored into nested `if`s at
`PRSegmentFunctions.cxx:1088-1095` with the same four thresholds
(muon `0.07/0.15`, proton `0.13/0.27`, split at `5 cm`) and the same fall-through
to the stored flag. The doc pr/6 divergence was about *which predicate the
callers use*, not about this function.

**5.11 `examine_all_showers`'s `particle_score() != 100` test.** Present in both
(prototype `:1040`, toolkit `:1586`). It looks like a toolkit-ism because 100
is the toolkit's sentinel, but the prototype wrote the test first.

**5.12 The three prototype `examine_maps` overloads lose nothing.** `:160`
takes a vertex and `:209` takes a cluster, and **both are one-line forwarders**
to `:167`, which scans the whole cluster (`return examine_maps(
temp_vertex->get_cluster_id())`). The per-vertex signature never implied
per-vertex granularity. The toolkit's single `examine_maps(graph, cluster)` is
the faithful shape, including at the stage-4 caller
(`NeutrinoVertexFinder.cxx:1684` ↔ prototype `:2298`).

**5.13 `cal_kine_dQdx`'s missing recombination constants.** The prototype's
`23.6e`, `1e`, `0.255`, `1.0` (Modified Box) are absent from the toolkit
because the toolkit delegates to `IRecombinationModel` — doc pr/10's
`PowerBoxRecombination`, already on record and SBND-default. Not a dropped
constant.

---

## §6 Determinism

**The prototype side is clean, for the same reason as pr/30.** Every map-repair
helper iterates `map_segment_vertices` / `map_vertex_segments`, keyed by
`ProtoSegmentCompare` / `ProtoVertexCompare`, which order by `get_id()` with a
pointer tiebreak that is unreachable because ids come from monotonic counters.

**The toolkit side is clean at this rev, and this is the first audit in the
series where that is true on arrival.** `4f2e7303` contains the doc pr/28 §10
sweep (117 sites / 13 files), so the finding pr/30 recorded as P14 is closed
inside the audited tree; it is not re-derived here. `NeutrinoTrackShowerSep.cxx`
uses `ordered_edges` / `sorted_out_edges` at 28 sites, several with explicit
comments naming what the order protects (`:294-295`, `:1558-1560`).

**Three raw `boost::edges(graph)` loops remain in scope** — `:44`
(`separate_track_shower`), `:75` (`determine_direction`) and `:332`
(`examine_good_tracks`). All three were read. Judgement:

* `:44` and `:75` are per-segment and self-contained. `segment_is_shower_topology`,
  `segment_is_shower_trajectory` and the three direction functions read and write
  only the segment passed to them plus its own point clouds. No accumulator, no
  argmax, no cross-segment state. Order-independent.
* `:332` calls `calculate_num_daughter_showers` (which traverses the graph but
  only reads) and then writes only to `sg`. The inner neighbour loop at `:400`
  already uses `sorted_out_edges`. Order-independent.

**Judged, not proven.** These are source arguments; no N-run identity test was
executed for this stage.

**One ordering difference that is not non-determinism.** The prototype orders
neighbours by segment id; the toolkit by graph edge index. Both are
reproducible run to run — which is what the gates check — and they are not
guaranteed to be the same order. Where an argmax uses strict `>` (e.g.
`find_cont_muon_segment_nue:319`), ties resolve to different candidates. Same
observation as pr/30 §6.

**Pointer-keyed containers.** `NeutrinoTrackShowerSep.cxx` declares
`std::set<VertexPtr>` / `std::set<SegmentPtr>` at ten places (`:168 :169 :228
:229 :579 :580 :686 :687 :1305 :1306`). All are `used_*` membership sets —
inserted into and tested with `count()`, never iterated. `PRSegmentFunctions.cxx`
declares three pointer-keyed maps at `:1916-1918` and they carry an explicit
`SegmentIndexCmp` comparator. Two maps at `:2028-2029` are default-compared
(`std::map<SegmentPtr, …>` with no comparator) inside `clustering_points_segments`
— **not examined for iteration**, see §7.

---

## §7 Loose ends

1. **Do the two `std::map<SegmentPtr, …>` at `PRSegmentFunctions.cxx:2028-2029`
   get iterated?** They are default-compared, i.e. pointer-ordered. Inside
   `clustering_points_segments`, which is Tier B here. Worth thirty seconds
   from someone who is already in that function.
2. **Is `separate_track_shower` ever run twice on the same segment?** P13's
   never-cleared flag only bites if it is. `TaggerCheckNeutrino.cxx` calls it
   once per cluster and once per companion, which suggests no — but segments
   can move between clusters across the PR chain's structural edits.
3. **`segment_cal_4mom`'s dead `MIP_dQdx` parameter** (§5.4) — either use it or
   drop it, but not while the call sites disagree about which scale to pass.
4. **`kslike_compare` divides by zero** when a sample vector sums to zero;
   ROOT's `KolmogorovTest` returns 0 with an error. Reachable only if every
   dQ/dx in the window is exactly zero.
5. **The porting dictionary has no topology/PID/direction section**, same as
   pr/29's Steiner gap and pr/30's proto-vertex gap. Every divergence above is
   undocumented by construction, so §5 rule 4 applies to all of them: both
   readings, no silent pick.
6. **Carried from pr/30 and still open:** P1 (`flag_exclusion`), P2, P3, P4,
   P6, P7, and the `update_association` coordinate question. **From pr/29:**
   D1, D2, D3.

---

## §8 Summary

Ranked by how much output moves, most first.

| # | divergence | tier | site | class |
|---|---|---|---|---|
| P1 | 4-momentum guard dropped at 11 of 13 map-repair sites | A | `NeutrinoTrackShowerSep.cxx:768…1374` | unconditional, energy |
| P2 | `segment_determine_shower_direction` at a toolkit-only call site | A | `:123` | unconditional, direction |
| P3 | topology branch also writes 4-momentum + `score(100)` | A | `:124-135` | unconditional, energy + score |
| P4 | rest-mass-only 4-momentum at 3 reclassification sites | A | `:428 :1463 :1867` | unconditional, energy |
| P5 | `find_vertices` order feeds an asymmetric test | A | `:1595-1620` | unconditional, ordering |
| P6 | `dir3` hoisted out of the loop, condition changed | A | `:286-311` | unconditional, geometry |
| P7 | empty comparison window returns "direction confirmed" | A | `PRSegmentFunctions.cxx:1334` | unconditional, degenerate |
| P8 | median dQ/dx over a filtered sample set | A | `:1743`, `:891` | unconditional, PID |
| P9 | `fits()` queried where the prototype queries `point_vec` | A | `NeutrinoTrackShowerSep.cxx:1419` | unconditional, geometry |
| P10 | null-vertex path continues vs skips | A | `:84-87` | unconditional, degenerate |
| P11 | start/end vertex by distance, not wcpt index | A | `:97-102`, `:359-381` | unconditional, degenerate |
| P12 | `boost::degree` vs segment-set size (self-loops) | B | `:107-111` | unconditional, degenerate |
| P13 | `kShowerTopology` set-only; early returns skip `dirsign` | A | `PRSegmentFunctions.cxx:2901`, `:2518-2529` | unconditional, staleness |
| P14 | PID persistence gate — **doc pr/7, already open** | A | `:1797-1812` | unconditional, PID |
| P15 | three raw `boost::edges` loops | — | `NeutrinoTrackShowerSep.cxx:44 :75 :332` | judged benign |

Every P-row is escalation rule 1 — it changes production output
unconditionally, with no knob — so none of them is mine to fix. No
recommendation is offered, per §5 rule 4 and the missing dictionary section
(§7.5).

---

## §9 What is NOT claimed

* **No event was run.** Nothing here was measured. Every "this changes the
  output" is an argument from source.
* **Reach is not quantified for any finding.** P1 in particular: the mechanism
  is source-certain, the number of segments per event that hit it and the
  resulting energy shift are not.
* **P7's prototype behaviour is inferred, not observed.** ROOT's
  `KolmogorovTest` on zero-bin histograms was not executed; the claim that the
  prototype abstains follows from `eval_ks_ratio`'s first line on all-zero
  inputs.
* **P12's premise is unverified.** Whether the PR graph can carry a self-loop
  at this stage was not established. If it cannot, P12 is vacuous.
* **§6's three-loop verdict is judged, not proven.** No N-run identity test was
  run for this stage.
* **Tier B pairs were not read line by line** — see §1 for the list.
  `improve_maps_no_dir_tracks` (331 vs 474 lines) is the largest unread body
  and also the one containing eight of P1's eleven sites; P1's *guard* claim was
  verified at each site individually, but the surrounding branch conditions
  were not compared against the prototype's.
* **Stage 4 was not audited** and is not claimed clean — see §0.
* **No recommendation is made** on any divergence.

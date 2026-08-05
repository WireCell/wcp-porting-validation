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
Fifteen divergences, ranked in §8 — **fourteen** after §10.11 withdraws P9.

> **→ If you want the short list, read §10.** The owner applied the pr/30 filter
> on 2026-08-04 — *"skip the ones that are improvements over the prototype, only
> the ones that are bugs or missing from the port … suggest a filtered list as
> well as the solutions"*. Fifteen findings become **nine**, each re-verified at
> `6206c46b` and each with a proposed fix, knob name and gate. P9 is withdrawn
> as not a divergence at all (§5.14). §10.13 corrects §3/§8.
>
> **→ F2 is now SHIPPED, default OFF — read §11.** Knob
> `shower_topo_proto_dir`; knob-off gate PASS (48/48 pctree hashes +
> `nusel-table.tsv`/`nusel-events.tsv` identical vs `work-pr30-f2on`); knob-on
> moves `kine_reco_Enu` on 7 of 48 events and **changes no selection outcome**.
> SBND default deliberately NOT flipped (§11.7).

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

> **WITHDRAWN by §10.11 (2026-08-04). This is not a divergence.** The premise
> below — "neither `fits()` nor `wcpts()` is `point_vec`" — is wrong:
> `ProtoSegment.h:16` declares
> `std::vector<WCP::Point>& get_point_vec(){return fit_pt_vec;}`, so
> `get_point_vec()` **is** the fitted points and `sg->fits()` is its faithful
> counterpart. Restated in §5.14; §8's P9 row is struck. The paragraph is kept
> so the mistake is legible.

**Tier A. Unconditional.**

Prototype (`:298`) iterates `sg->get_point_vec()` — the segment's path points.
Toolkit (`:1419`, `:1418` at HEAD) iterates `sg->fits()` — the fitted
trajectory — with the prototype's accessor left in the source as a comment:

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

**5.14 `judge_no_dir_tracks_close_to_showers` queries the *same* points — P9
withdrawn** *(added by §10.11, 2026-08-04)*. The prototype's accessor is a
one-line alias for the fitted points:

```cpp
// prototype_base/pid/inc/WCPPID/ProtoSegment.h:16
std::vector<WCP::Point >& get_point_vec(){return fit_pt_vec;};
```

`fit_pt_vec` is the same array `cal_4mom`, `get_length`, `get_medium_dQ_dx` and
`is_shower_topology` all read, and the toolkit's `sg->fits()` is its port. So
`const auto& pts = sg->fits();//wcpts();` at `NeutrinoTrackShowerSep.cxx:1418`
is the porter **choosing correctly and leaving the rejected alternative in a
comment** — the opposite of what §3's P9 read into it. Do not re-derive this;
`get_point_vec` is named for what it returns in the prototype's *other*
classes, not for `wcpt_vec`.

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
2. ~~**Is `separate_track_shower` ever run twice on the same segment?**~~
   **CLOSED — yes, and the question was aimed one level too high** (§10.4). What
   matters is `segment_is_shower_topology`, which has **four** call sites:
   `NeutrinoTrackShowerSep.cxx:54` (stage 3) and `NeutrinoVertexFinder.cxx:2311
   :2357 :2430` (stage 4). Re-entry on the same segment is the normal path, so
   P13's never-cleared flag is live, not latent.
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
| ~~P9~~ | ~~`fits()` queried where the prototype queries `point_vec`~~ — **WITHDRAWN**, `get_point_vec()` *is* `fit_pt_vec` (§5.14, §10.11) | — | `:1418` | not a divergence |
| P10 | null-vertex path continues vs skips | A | `:84-87` | unconditional, degenerate |
| P11 | start/end vertex by distance, not wcpt index | A | `:97-102`, `:359-381` | unconditional, degenerate |
| P12 | `boost::degree` vs segment-set size (self-loops) | B | `:107-111` | unconditional, degenerate |
| P13 | `kShowerTopology` set-only; early returns skip `dirsign` | A | `PRSegmentFunctions.cxx:2901`, `:2518-2529` | unconditional, staleness |
| P14 | PID persistence gate — **doc pr/7, already open** | A | `:1797-1812` | unconditional, PID |
| P15 | three raw `boost::edges` loops | — | `NeutrinoTrackShowerSep.cxx:44 :75 :332` | judged benign |

Every P-row is escalation rule 1 — it changes production output
unconditionally, with no knob — so none of them is mine to fix. ~~No
recommendation is offered, per §5 rule 4 and the missing dictionary section
(§7.5).~~

> **Superseded, 2026-08-04.** That sentence was right for an audit-only
> document. The owner's filter request (§10) authorizes the picks and the
> proposed fixes, so **§10 does offer a recommendation** — nine survivors, each
> with a code change, a default-OFF knob name, a gate and a bundling note, in
> §10.12's suggested order. Escalation rule 1 is not weakened: nothing here is
> applied, and each item still ships as a knob with a byte-identical gate. §7
> loose end 5 (no topology/PID/direction section in the porting dictionary) is
> unchanged, and §10's nine items are what that section should say.

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
* **P12's premise is *half* verified** *(updated 2026-08-04, §10.10)*. The BGL
  semantics were settled by execution against the PR graph's exact selectors:
  a self-loop **is** accepted and `boost::degree` counts it **2**, and a
  parallel edge is **rejected** with the second segment silently aliased onto
  the first's edge. What remains unverified is *reachability* — whether any of
  the 63 `PR::add_segment` call sites ever produces either. If neither is
  reachable, P12 is vacuous; §10.10 names the one-count check that decides it.
* **§6's three-loop verdict is judged, not proven.** No N-run identity test was
  run for this stage.
* **Tier B pairs were not read line by line** — see §1 for the list.
  `improve_maps_no_dir_tracks` (331 vs 474 lines) is the largest unread body
  and also the one containing eight of P1's eleven sites; P1's *guard* claim was
  verified at each site individually, but the surrounding branch conditions
  were not compared against the prototype's.
* **Stage 4 was not audited** and is not claimed clean — see §0.
* ~~**No recommendation is made** on any divergence.~~ **Superseded by §10**
  (2026-08-04): nine of the fifteen now carry a proposed fix, a knob name and a
  gate. Still not claimed there: that any fix was **implemented**, **built**,
  **run** or **measured** — §10 is source-level reasoning and one 20-line BGL
  test program, nothing more. No event was run for it either.

---

## §10 Owner filter, 2026-08-04 — the nine that are bugs or gaps

**What was asked.** The owner read §3/§8 and applied the same filter used on
doc pr/30:

> *"skip the ones that are improvements over the previous prototype, and only
> focus on the ones that are bugs or missing from the port … feel free to look
> at the prototype code and use the logic and your understanding to suggest a
> filtered list as well as the solutions."*

So this section does two things §3 deliberately did not: it **picks** where §3
recorded "both readings, not picked" (§5 rule 4 / M15 — the owner asking for a
filtered list is what authorizes the pick, and both readings are kept
underneath), and it **proposes a fix** for each survivor. §8's closing sentence
("No recommendation is offered") is superseded here and corrected in place.

**Method.** Every claim below was re-derived against toolkit **`6206c46b`** —
§3 was written at `4f2e7303` — and against `prototype_base/`. Toolkit files
were read from `git show HEAD:<file>`, not the working tree, because another
session had `PRGraph.cxx` / `PRGraph.h` / `NeutrinoPatternBase.h` modified at
the time. Two §3 claims did not survive re-verification and are corrected in
§10.13. One premise was settled by **executing** a test rather than reading
(§10.11 F9).

**No code is changed by this section.** No gate is owed and none was run.

### §10.0 The discriminator, and the two dispositions carried in from pr/30

"Improvement vs bug" is not decided by whether the toolkit does *more* or
*less* — on P2, P3 and P7 that test gives the wrong answer in both directions.
The discriminator is the one pr/30 §10.2 established: **evidence of intent vs
evidence of accident.**

* *Evidence of intent* — a comment stating the reason, a config knob, a doc
  reference, a stated deviation.
* *Evidence of accident* — the prototype's guard reproduced at 1 of 12 sites
  and nowhere else, the prototype's accessor left in the source as a comment, a
  variable hoisted out of a loop with a silently changed fallback, a defensive
  early return whose returned *value* was never chosen.

Every survivor below carries at least one accident marker. Every drop carries
an intent marker, or is not a divergence at all.

**Where the owner's positional clearance reaches.** pr/30 §10's clarification —
*"in the toolkit we are missing id information for some data, so we have to use
positions to do it — this is not a problem"* — disposes of **P11** cleanly
(start/end vertex by 3-D distance instead of `wcpt().index` equality). It does
**not** reach two things, and this is the most reusable line in this section:

1. **P5**, by pr/30 F4's own narrowing. There, `.first` is not naming a
   position; it is selecting which of two *asymmetric* acceptance tests a
   segment gets. A positional substitution is fine where `.first` means "an
   end"; it is not fine where `.first` means "the branch with the relaxed
   150° clause".
2. **P12**, which substitutes a *semantic* quantity (count of incident graph
   edges) for a different one (count of distinct segments at a vertex). Nothing
   positional about it.

### §10.1 The filtered list — nine items

| # | from | what it is | why it survives the filter | severity |
|---|---|---|---|---|
| **F1** | P1 + P3(4-mom) + P4 | the toolkit **rewrites `particle_4mom` at 15 reclassification sites** where the prototype writes type and mass and leaves the 4-momentum alone | one root cause, three shapes, and the guard is reproduced at exactly **one** site *with a comment paraphrasing it* — the signature of a lost line, not a design decision | **highest** |
| **F2** | P2 | a 305-line direction analysis runs at a stage-3 call site the prototype does not have, overwriting the direction `is_shower_topology` set | a **different function's body** was substituted for the one being ported — unconditional, undeclared, and it reaches production through `dirsign`. **Not** claimed to be the wrong physics: the prototype's own version says `// hack for now` (§10.3, corrected 2026-08-04). **Knob SHIPPED — §11** | **high** |
| **F3** | P13 | `kShowerTopology` is set and never cleared, and four early returns skip `dirsign()` | **now proven live**: `segment_is_shower_topology` has 4 call sites, 1 in stage 3 and 3 in stage 4, so re-entry on the same segment is normal (closes §7 loose end 2) | **high** |
| **F4** | P8 | `segment_median_dQ_dx` **filters** its sample where the prototype's structural counterpart `get_medium_dQ_dx` does not — and in `determine_dir_track` the median and the KS sample are now computed from **different sets** | the filter is defensible; the toolkit being inconsistent *with itself* is not, and it is invisible at the call site | medium-high |
| **F5** | P6 | `dir3` hoisted out of the loop with a fallback that silently turns a 30 cm direction comparison into a 15 cm one | textbook hoist-with-changed-semantics; the correct hoist is unconditional and is still a hoist | medium |
| **F6** | P7 (value half) | the empty-comparison-window early return yields **"direction confirmed"** | the guard is right; the value `1.0` was never chosen — `0.0` abstains and matches the prototype | medium |
| **F7** | P5 | the asymmetric 150°/165° acceptance in `examine_all_showers` is keyed on `find_vertices().first` | pr/30 F4's sibling — the one place the positional clearance provably does not reach (§10.0) | medium |
| **F8** | P14 | the PID persistence gate | **already on record** as doc pr/7, DIAGNOSED, fix proposed and unimplemented; re-confirmed at `6206c46b`. No new solution offered here | medium, open |
| **F9** | P12, widened | the PR graph **cannot represent** either degenerate topology the prototype's maps can: a self-loop counts **2** in `boost::degree` (verified by execution) and a second segment on an existing vertex pair is **silently aliased onto the first edge** (verified by execution) | not positional, not an improvement; but reachability is still unproven, so this is conditional | low, conditional |

Six things are dropped and two more are dropped by half — §10.11.

### §10.2 F1 — the `particle_4mom` rewrite family (P1 + P3's 4-mom half + P4)

§3 reported these as three findings with, in P4's case, *opposite* polarity to
P1. Re-reading both trees at HEAD, they are **one defect with three shapes**,
and seeing that is what makes a single fix possible.

**The root cause is the data model, and §3 named it without following it
through.** The prototype's reclassification idiom mutates two members:

```cpp
// NeutrinoID_track_shower.h:372-374 — and identically at :407 :439 :462
//                                      :508 :541 :578 :637 :873 :934 :951
sg->set_particle_type(11);
sg->set_particle_mass(mp.get_mass_electron());
if (sg->get_particle_4mom(3)>0) sg->cal_4mom();
```

`particle_4mom` is a third, *independent* member. The guard reads "recompute
the energy only if this segment already had one" — `particle_4mom[3] =
kine_energy + particle_mass` (`ProtoSegment.cxx:1437-1444`), so `>0` means
"previously computed". A segment that never had an energy keeps zero.

The toolkit has no independent members. `particle_type`, `particle_mass` and
the 4-momentum all live inside one `Aux::ParticleInfo`, and
`seg->particle_info(pinfo)` **replaces the whole struct**. So every
"set the particle type" in the prototype becomes "construct a new
`ParticleInfo`" in the toolkit, which *forces a decision about the 4-momentum
that the prototype never has to make*. Fifteen sites made that decision three
different ways, and only one of them made it the prototype's way.

**The three shapes, all re-derived at `6206c46b`:**

| shape | sites (HEAD anchors) | what the toolkit writes | what the prototype writes |
|---|---|---|---|
| **A — unguarded recompute** | `NeutrinoTrackShowerSep.cxx:767 :792 :937 :979 :1021 :1047 :1110 :1163 :1206 :1285 :1373` (11) | `segment_cal_4mom(...)` unconditionally | `cal_4mom()` **only if** `get_particle_4mom(3)>0` |
| **B — topology-branch recompute** | `:126` (1) | `segment_cal_4mom(...)` + a full `ParticleInfo` | `determine_dir_shower_topology` writes type and mass only — **no** 4-momentum |
| **C — rest-mass overwrite** | `:423-427`, `:1458-1462`, `:1861-1865` (3) | `D4Vector(mass, 0, 0, 0)` — zero kinetic energy | `:257-261`, `:313-315`, `:1238-1239` — type and mass only, 4-momentum untouched |

**Two sites get it right, and that is the accident marker.** `:661`
(`improve_maps_one_in`) guards on `has_particle_info()` — *stricter* than the
prototype's unguarded `:147`, so this one site diverges in the safe direction.
`:806` reproduces the prototype's guard exactly **and comments it**:

```cpp
// Prototype calls cal_4mom() for ALL segments here (including showers) if energy>0.
if (is_shower1 && sg1->has_particle_info() && sg1->particle_info()->energy() > 0) {
```

A porter who wrote that comment understood the guard. Its absence at the other
eleven is not a design decision. Corroborating: the only surviving textual
trace of `get_particle_4mom(3)` anywhere in `clus/` is a **commented-out** line
at `NeutrinoShowerClustering.cxx:2771`.

**Bucket reconciliation (unchanged from §3, re-checked at HEAD).** 14
`segment_cal_4mom` sites = 1 topology (shape B) + 13 map-repair; prototype 12
in-scope `cal_4mom()` sites; per-function counts match exactly, so no site is
missing on either side.

**Both readings, and the pick.** §3 logged shape C as M15 — the prototype's
stale 4-momentum under a superseded hypothesis is arguably its own bug, and the
toolkit's discard of a computed energy is arguably the toolkit's. That framing
dissolves once A, B and C are read together: at all fifteen sites the prototype
**does not write the 4-momentum**, and the toolkit writes one — sometimes a
recomputed energy (A, B), sometimes a zero (C). The single rule that reproduces
the prototype everywhere is *preserve the existing 4-momentum; recompute only
where the prototype's guard passes*. The pick is therefore that the toolkit is
wrong at all fifteen sites, in one way, not in two opposing ways.

**Solution.** One knob, three per-shape behaviours. The fix is **not** a deleted
line — declining to compute a 4-momentum still means constructing a
`ParticleInfo`, so the old 4-momentum has to be carried forward explicitly.

```cpp
// NeutrinoPatternBase.h, member + get() in configure(), default preserves today's behaviour
bool m_reclass_preserve_4mom{false};

// helper, one place, used at all 15 sites
static WireCell::D4Vector<double>
reclass_4mom(SegmentPtr sg, int pdg, ..., bool preserve, bool proto_recomputes)
{
    if (!preserve) return segment_cal_4mom(sg, pdg, ...);            // today
    const bool had = sg->has_particle_info() && sg->particle_info()->energy() > 0;
    if (proto_recomputes && had) return segment_cal_4mom(sg, pdg, ...);  // shape A guard passes
    if (had) return sg->particle_info()->four_momentum();            // carry forward
    return WireCell::D4Vector<double>(0, 0, 0, 0);                   // never had one
}
```

* shape **A** (11 sites): `proto_recomputes = true`.
* shape **B** (`:126`): `proto_recomputes = false` — the prototype never
  recomputes here.
* shape **C** (3 sites): `proto_recomputes = false`, and the `(m,0,0,0)`
  literal is replaced by the helper's return.

Note the zero-return case changes the sign convention §3 flagged: a segment
that never had a 4-momentum ends with `E = 0`, so `E − m = −m` for anyone who
subtracts. That is exactly the prototype's state, and any consumer that breaks
on it was reading a toolkit-only invariant.

* **Knob**: `reclass_preserve_4mom`, C++ default `false`, key-suppressed in
  jsonnet so the compiled config is byte-identical when off.
* **Gate**: knob-off byte-identical on `abtest/events.txt`; knob-on smoke on
  the `work-vfnuecc48-0804` arm — `kine_reco_Enu` is the column that must move,
  and it is already tabulated per event by `pr_scores_table.py`, so the
  before/after is a diff of two TSVs, not a new harness.
* **Bundling**: F1 is self-contained. It must be validated **alone** — it is
  the only survivor that moves reconstructed energy directly, and bundling it
  with F2/F3 (which move direction, which moves energy indirectly) would make
  the nueCC48 delta unattributable, exactly the coupling trap doc pr/28 §16.5
  hit with pr/28-vs-pr/29.

### §10.3 F2 — the stage-3 shower-direction call (P2)

Re-verified at HEAD. The prototype calls `ProtoSegment::determine_shower_direction()`
from **one** place in the entire tree, `NeutrinoID_track_shower.h:1532`, inside
`compare_main_vertices_all_showers` — **stage 4**, and only on the all-showers
path. The toolkit calls its port from two: `NeutrinoVertexFinder.cxx:506` (the
faithful one) and `NeutrinoTrackShowerSep.cxx:123` (stage 3, no counterpart).

What the prototype's stage-3 topology branch does, with the commented-out
blocks removed, is the whole of `determine_dir_shower_topology`
(`ProtoSegment.cxx:1677-1710`):

```cpp
double length = get_length();
// hack for now                          <-- the prototype author's own words
particle_type = 11;
TPCParams& mp = Singleton<TPCParams>::Instance();
particle_mass = mp.get_mass_electron();
// (flag_dir block commented out; centre-of-associated-cloud block commented out)
```

It takes `start_n` and `end_n` and **uses neither** — both survive only inside
the commented-out block. It does not touch `flag_dir`. The direction of a
topology shower leaving stage 3 in the prototype is whatever
`is_shower_topology`'s forward/backward large-spread comparison left there
(`ProtoSegment.cxx:523-527`).

The toolkit's inlined branch runs `segment_determine_shower_direction` first —
305 lines of associated-point PCA, spread profiling and endpoint comparison
(`PRSegmentFunctions.cxx:2208-2512`) — which sets `dirsign` on its own terms,
overwriting `segment_is_shower_topology`'s answer for **every** topology shower.
And `dirsign` is what the in/out maps of this stage and the vertex scorer of
stage 4 read.

**Why it survives the filter — corrected 2026-08-04, and narrower than first
written.** The original text here read *"the commented-out `flag_dir` block in
the prototype shows the prototype author considered and rejected setting a
direction here"*, and called F2 a bug on that basis. That over-reads the
evidence, and the quote above omitted the line that shows why: the function
opens with **`// hack for now`**. A self-declared hack with both of its
direction blocks commented out reads at least as naturally as *tried and
unfinished* as it does *considered and rejected*, and the toolkit's 305-line
PCA may well be the better physics. Nobody has measured which is better, so
neither claim is available.

What is certain, and is what puts F2 on this list:

* the toolkit substituted a **different function's body** for the one being
  ported — `determine_dir_shower_topology` has a counterpart, and what got
  inlined at `NeutrinoTrackShowerSep.cxx:121-135` is not it;
* the substitution is **unconditional and undeclared** — no comment, no knob,
  no doc entry on the toolkit side;
* it **reaches production output** through `dirsign`, which this stage's in/out
  maps and stage 4's vertex scorer both read.

So the defensible verdict is *an unconditional, undeclared divergence with real
reach, direction unknown* — which justifies a knob and an A/B, not a claim that
the toolkit is wrong. That is exactly what §11 implements.

**Solution.** Restore the prototype's branch under a knob:

```cpp
// NeutrinoTrackShowerSep.cxx:121-135
} else if (seg->flags_any(SegmentFlags::kShowerTopology)) {
    if (!m_shower_topo_proto_dir) {
        segment_determine_shower_direction(seg, ...);   // today
    }
    ... // particle info: see F1 shape B
}
```

* **Knob**: `shower_topo_proto_dir`, default `false` (= today). When true, the
  stage-3 call is skipped and the topology shower keeps the direction
  `segment_is_shower_topology` set.
* **Gate**: knob-off byte-identical; knob-on on nueCC48. The visible column is
  the per-segment `dirsign` in the PR display dump (doc pr/26 stage 1 already
  emits it) plus `kine_reco_Enu` / `nue_score` downstream.
* **Bundling**: F2 and F3 both change what direction a topology shower leaves
  stage 3 with. Validate separately, then jointly — per
  [[project_staged_small_group_validation]], a small event group each before
  any 572-event census.
* **Cheap pre-check worth doing first**: count, on one event, how many segments
  carry `kShowerTopology` at this point and how many have their `dirsign`
  changed by the call. If it is a handful, F2 is cheap to settle; if it is most
  of them, F2 is the largest single behaviour item in this document after F1.

### §10.4 F3 — `kShowerTopology` is never cleared (P13), now proven live

§3 raised this and could not say whether it bites, leaving §7 loose end 2 open
("is `separate_track_shower` ever run twice on the same segment?"). **It does,
and the question was aimed one level too high.** What matters is not
`separate_track_shower` but `segment_is_shower_topology`, and at HEAD it has
four call sites in two different stages:

```
clus/src/NeutrinoTrackShowerSep.cxx:54     separate_track_shower        (stage 3)
clus/src/NeutrinoVertexFinder.cxx:2311                                  (stage 4)
clus/src/NeutrinoVertexFinder.cxx:2357     gated on !sg->particle_info()  (stage 4)
clus/src/NeutrinoVertexFinder.cxx:2430                                  (stage 4)
```

Stage 4 runs after stage 3 on the same segments, so re-entry is not a corner
case — it is the normal path for any segment reaching those sites. **§7 loose
end 2 is closed: yes.**

The prototype's first two statements are assignments:

```cpp
// ProtoSegment.cxx:319-321
bool WCPPID::ProtoSegment::is_shower_topology(bool tmp_val){
  flag_shower_topology = tmp_val;      // <-- CLEARS on every entry (all callers pass false)
  flag_dir = 0;                        // <-- CLEARS the direction, before any early return
```

The toolkit's are a local and a set-only flag write:

```cpp
// PRSegmentFunctions.cxx:2513-2530, :2901
bool flag_shower_topology = tmp_val;                     // a LOCAL, not the flag
if (fits.empty()) return false;                          // 4 early returns,
if (!dpcloud_fit) return false;                          //   all BEFORE any mutation,
if (!dpcloud_assoc) return false;                        //   so dirsign() is skipped
if (assoc_npts == 0) return false;
...
if (flag_shower_topology) segment->set_flags(SegmentFlags::kShowerTopology);  // set-only
segment->dirsign(flag_dir);
```

Two distinct losses, both confirmed at HEAD:

1. **Stale flag.** A segment that was a topology shower on an earlier pass and
   is not one now keeps `kShowerTopology`. Since the flag is what routes
   `determine_direction` into the topology branch (and F2's extra call), a
   stale flag also drags F2's behaviour onto a segment that no longer qualifies.
2. **Stale direction.** A segment whose associated point cloud is empty leaves
   the prototype undirected (`flag_dir = 0`, set before the return) and leaves
   the toolkit with whatever direction it had, because the four early returns
   are all before `segment->dirsign(flag_dir)`.

**Solution.** Two lines, one knob:

```cpp
bool segment_is_shower_topology(SegmentPtr segment, bool tmp_val, ...) {
    int flag_dir = 0;
    bool flag_shower_topology = tmp_val;
    if (m_shower_topo_reset) {                       // knob, default false
        if (!tmp_val) segment->unset_flags(SegmentFlags::kShowerTopology);
        segment->dirsign(0);                         // before the early returns
    }
    ...
```

then leave the tail as it is (the `if (flag_shower_topology) set_flags(...)`
re-sets it when the answer is still yes).

* **Knob**: `shower_topo_reset`, default `false`.
* **The API already exists** — `Flagged::unset_flags(FlagsType)` clears exactly
  the named bits and keeps the rest (`util/inc/WireCellUtil/Flagged.h:69`; do
  **not** use `clear_flags()`, which zeroes every flag on the segment). So this
  really is a two-line change.
* **Gate**: knob-off byte-identical; knob-on on nueCC48, watching the count of
  `kShowerTopology` segments at stage 4 (it can only go down) and `dirsign`.
* **Bundling**: interacts with F2 (see §10.3). Also interacts with
  `shower_topo_demote_len` (SBND = 50 cm), which sets `flag_shower_topology =
  false` inside the guard at `:2885-2889` — with the flag never cleared, a
  segment demoted on one pass can be re-promoted by a stale flag on the next.
  That is worth one explicit check when this lands.

### §10.5 F4 — the median dQ/dx sample set (P8)

Re-verified, and the prototype side is sharper than §3 recorded. The toolkit's
`segment_median_dQ_dx` is the structural port of the prototype's
`get_medium_dQ_dx(int n1, int n2)`, and the two differ by a filter:

```cpp
// prototype ProtoSegment.cxx:689-699 — no filter, every index in range
for (int i = n1 ; i<=n2; i++)
  vec_dQ_dx.push_back(dQ_vec.at(i)/(dx_vec.at(i)+1e-9));

// toolkit PRSegmentFunctions.cxx:888-893 — filtered
for (int i = n1; i <= n2 && i < (int)fits.size(); i++) {
    auto& fit = fits[i];
    if (fit.valid() && fit.dx > 0 && fit.dQ >= 0)
        vec_dQ_dx.push_back(fit.dQ / (fit.dx + 1e-9));
}
```

The filter changes the vector length, so `nth_element` at `size()/2` selects a
**different order statistic**, at all three toolkit call sites (`:1021` inside
the trajectory-shower path, `:1743` inside `determine_dir_track`, and the
`find_cont_muon_segment_nue` ratio, which is the port of the prototype's
`get_medium_dQ_dx()` no-arg overload → `get_medium_dQ_dx(0, fit_pt_vec.size())`).

**The part that survives the filter is not "the median differs" — it is that
the toolkit is inconsistent with itself.** In `determine_dir_track` the
prototype takes the median over *the very vector it hands to the PID*:

```cpp
// ProtoSegment.cxx:1573-1576
std::vector<double> vec_dQ_dx = dQ_dx;      // <-- the same dQ_dx passed to do_track_pid
std::nth_element(...);
```

The toolkit instead calls the shared helper, which rebuilds a *different*
sample from `fits()`. And the two disagree about the pathological point in
opposite directions: the toolkit's KS vector is initialised
`std::vector<double> dQ_dx(npoints, 0)` and left at **0** wherever
`fits[i].dx <= 0` (`:1683-1687`), while the median simply **drops** those
points. So one bad fit is simultaneously "a zero" to the PID and "not there" to
the median — a state neither tree intends.

**Why the filter itself is not the bug.** Excluding `!valid()` fits is
defensible, and the prototype's `dQ/1e-9` for `dx == 0` is a huge sentinel that
drags the median toward the proton branch. Judged an improvement, kept.

**Solution**, smallest first:

1. **The self-consistency fix (recommended).** In `determine_dir_track`, take
   the median over the local `dQ_dx` vector the PID receives, as the prototype
   does — one line, no new helper:
   ```cpp
   std::vector<double> tmp = dQ_dx;   // instead of segment_median_dQ_dx(segment, start_n1, end_n1)
   std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/2, tmp.end());
   double medium_dQ_dx = *std::next(tmp.begin(), tmp.size()/2);
   ```
   Note the unit consequence and check it: the local `dQ_dx` is in
   charge/internal-length while `segment_median_dQ_dx` returns the same, and
   `MIP_dQdx` is `m_mip_dqdx_median` (internal) — so the three comparisons
   `>1.75×`, `<1.2×`, `<1.5×` keep their scale. Confirm this before landing;
   the sec-7.8 SSM `get_scores` bug was exactly a factor-of-`units::cm` here.
2. **Full prototype parity (not recommended without discussion)** additionally
   drops the filter from `segment_median_dQ_dx` and writes `dQ/(dx+1e-9)`
   unconditionally into the KS vector. That restores the prototype bit for bit
   including its huge-value sentinel, which is an M15 call the owner should
   make, not me.

* **Knob**: `dir_track_median_local`, default `false`, covering (1) only.
* **Gate**: knob-off byte-identical; knob-on on nueCC48. The observable is the
  `pdg==0` recovery branch — count segments whose recovered `pdg_code` changes.
* **Bundling**: independent of F1/F2/F3; affects PID, not directly energy.

### §10.6 F5 — the hoisted `dir3` (P6)

Re-verified line by line at HEAD; unchanged from §3 and unambiguous.

```cpp
// prototype NeutrinoID_track_shower.h:2402-2408 — inside the per-neighbour loop
if (length > 30*units::cm || sg_length > 30*units::cm){
  TVector3 dir3 = sg ->cal_dir_3vector(vtx->get_fit_pt(), 30*units::cm);
  TVector3 dir4 = sg2->cal_dir_3vector(vtx->get_fit_pt(), 30*units::cm);
  angle1 = (3.1415926 - dir3.Angle(dir4))/3.1415926*180.;
}

// toolkit NeutrinoTrackShowerSep.cxx:286-311 — hoisted above the loop
WireCell::Vector dir3 = (sg_length > 30 * units::cm)
                            ? segment_cal_dir_3vector(sg, vtx_pt, 30 * units::cm)
                            : dir1;                       // <-- dir1 is the 15 cm direction
```

The hoist is correct in itself — `dir3` does not depend on the loop variable.
The `? :` is not: the reachable case **short reference segment, long neighbour**
(`sg_length ≤ 30 cm < length`) makes the toolkit compare a 15 cm direction
against a 30 cm one where the prototype compares two 30 cm directions.
`angle1` is one of the three alternatives in the `< 12.5°` test that decides
whether a muon continues through a vertex, so this changes muon continuation.

**Why it is a bug.** A hoist that changes a value is a translation slip, not a
design choice; there is no comment, and the prototype's intent (compare like
with like at 30 cm) is unambiguous.

**Solution.** Keep the hoist, drop the conditional — one line, still an
optimisation, no knob strictly required but one is owed because it is
unconditional:

```cpp
WireCell::Vector dir3 = m_cont_muon_dir3_30cm
    ? segment_cal_dir_3vector(sg, vtx_pt, 30 * units::cm)      // prototype
    : ((sg_length > 30*units::cm) ? segment_cal_dir_3vector(sg, vtx_pt, 30*units::cm) : dir1);
```

* **Knob**: `cont_muon_dir3_30cm`, default `false`.
* **Gate**: knob-off byte-identical; knob-on on nueCC48 — the observable is the
  count of `find_cont_muon_segment_nue` non-null returns.
* **Bundling**: independent. **Cheapest survivor to settle; do it first.**

### §10.7 F6 — the empty comparison window (P7, value half only)

```cpp
// PRSegmentFunctions.cxx:1333-1336
// If no points fall inside the comparison window, return "no direction signal" defaults.
if (ncount == 0) {
    return {1.0, 1e9, 1e9, 1e9};
}
```

The comment says *"no direction signal"*; element 0 says the opposite.
`segment_do_track_pid` reads it as `flag_forward = round(result_forward.at(0))`,
so `1.0` means **this orientation passed the direction gate**. With no samples
in either orientation both flags are 1, the both-pass branch runs, and the
segment leaves with `flag_dir = -1`, `pdg = 13` (muon) and `particle_score =
1e9`. A segment about which nothing is known becomes a directed muon.

**The guard is an improvement and is kept.** The prototype constructs
`TH1F("h1","h1",0,0,0)` and calls `KolmogorovTest` on zero-bin histograms; not
crashing is better. What was never chosen is the returned value — the same
literal is copy-pasted at the adjacent missing-dEdx-function return (`:1343`),
which is the marker that it is a filler, not a decision.

**Solution.** One character of meaning:

```cpp
if (ncount == 0) {
    return {m_track_comp_empty_abstain ? 0.0 : 1.0, 1e9, 1e9, 1e9};
}
```

`0.0` rounds to `flag_forward = 0` — abstain — which is what
`eval_ks_ratio` returns for all-zero inputs (`ks1 - ks2 >= 0.0` → `false`),
i.e. the prototype's degenerate answer.

* **Knob**: `track_comp_empty_abstain`, default `false`. Apply to the
  missing-dEdx return at `:1343` as well, or state why not.
* **Gate**: knob-off byte-identical; knob-on on nueCC48.
* **Caveat carried from §9**: the prototype's ROOT behaviour on zero-bin
  histograms was **inferred, not executed**. If this item is going to be
  actioned rather than filed, run the four-line ROOT snippet first — it is
  cheaper than the A/B it would otherwise justify.
* **Bundling**: independent, but it and F4 both land in the PID path; expect
  their nueCC48 deltas to overlap on the same short segments.

### §10.8 F7 — the asymmetric acceptance keyed on `find_vertices` order (P5)

Both sides re-read at HEAD. The asymmetry is real and identical in both trees:

```
first  branch (num_s1, angles at .second):  max_angle > 165 || (max_angle > 150
                                              && length_good_tracks < 3 cm
                                              && length_good_tracks < 0.1*length_showers)
second branch (num_s2, angles at .first):   max_angle > 165
```

prototype `NeutrinoID_track_shower.h:1073` vs `:1090`; toolkit
`NeutrinoTrackShowerSep.cxx:1623` vs `:1647`. So the *tests* are ported
faithfully; what differs is **which physical vertex lands in `.first`** —
prototype by vertex id (`NeutrinoID_proto_vertex.h:3227-3243`), toolkit by
proximity to the segment's first fit point (`PRGraph.cxx:105-141`, re-read at
HEAD). A segment at 155° flips between reclassified and not depending purely on
that.

This is pr/30 F4 in a different file, and the two docs must agree: the
positional clearance covers `.first` where it means "an end of the segment"; it
does not cover `.first` where it selects a branch. **Do not fix this one in
isolation** — pr/30 F4 named two callers, this is a third, and a change to
`find_vertices`' ordering hits all of them at once.

**Solution**, in order of preference:

1. **Site-local, lowest risk.** Order the pair by a stable non-positional key
   at this call site only, before the two branches:
   ```cpp
   auto pair_vertices = find_vertices(graph, good_track);
   if (m_examine_showers_vertex_by_index && pair_vertices.first && pair_vertices.second
       && pair_vertices.first->get_graph_index() > pair_vertices.second->get_graph_index())
       std::swap(pair_vertices.first, pair_vertices.second);
   ```
   **Read the key honestly.** `PRVertex` has no `id()`; the only stable
   identity it carries is `get_graph_index()` (`PRVertex.h:92`), the monotonic
   `num_node_indices` counter assigned in `PR::add_vertex`. That is the *shape*
   of the prototype's `get_id()` — a creation-order counter — but the two trees
   create vertices in different orders, so this restores **a** deterministic
   topological convention, not provably *the prototype's* one. Which of the two
   physical vertices gets the relaxed 150° clause would still be a toolkit
   choice; it would just stop depending on which end of the segment happens to
   be nearer its first fit point. Say that in the commit message rather than
   claiming parity.
2. **Global**, i.e. change `find_vertices` to order by id — this is pr/30 F4's
   decision, not this document's, and it is the one that must not be taken here
   unilaterally.

* **Knob**: `examine_showers_vertex_by_index`, default `false`, covering (1).
* **Gate**: knob-off byte-identical; knob-on on nueCC48. The observable is
  `n_good_tracks` going to 0 on the relaxed branch — a handful of events at
  most, so this wants the 572-event valfast manifest to say anything about
  population reach.
* **Bundling**: with pr/30 F4. Whoever takes either should take both.

### §10.9 F8 — the PID persistence gate (P14)

Re-confirmed present at `6206c46b`. **No new solution is offered here**: this
is doc `pr/7 — track-PID persistence divergence`, status DIAGNOSED, with a fix
already written up and not implemented. What this audit adds is two knock-on
effects to include in that fix's validation, both from §3-P14:

* `determine_dir_shower_trajectory`'s else-branch tests
  `segment->particle_info()->pdg() != 11` where the prototype tests the member
  `particle_type != 11` — when the gate suppressed the write, the toolkit sees
  absent or stale info and zeroes a direction the prototype may keep;
* the failure-path score of `100.0` (`segment_do_track_pid:1600`) versus the
  prototype's `0` (`ProtoSegment.cxx:1284-1286`), which is a **live sentinel**
  in `examine_all_showers` (`if (sg->particle_score() != 100) tracks_score += …`,
  toolkit `:1584` ↔ prototype `:1040`).

The second is worth stating plainly because it cuts the other way from F1's
family: here the toolkit's `100` is arguably the *correct* sentinel and the
prototype's `0` lets a stale score into a sum. Recorded in doc pr/7, not picked
here.

### §10.10 F9 — the graph cannot represent two degenerate topologies (P12, widened)

§3 filed P12 with an unverified premise ("whether the PR graph can carry a
self-loop was not established") and only the self-loop half. Both halves were
settled **by execution** — a 20-line program against the exact
`adjacency_list` selectors the PR graph uses
(`PRGraphType.h:91-98`: `boost::setS` out-edge list, `boost::setS` vertex list,
`boost::undirectedS`):

```
$ g++ -std=c++17 -I local/include selfloop.cxx -o selfloop && ./selfloop
ok1=1 ok2=1 ok3=0
degree(a)=2 degree(b)=1 num_edges=2
out_edges(a) count=2
e1.idx=0 e3.idx=0 same_edge=1
```

Two facts, both now certain:

1. **A self-loop is accepted and counts 2.** `boost::degree` on the self-looped
   vertex returns 2 where the prototype's `map_vertex_segments[v].size()` — a
   `std::set<ProtoSegment*>` into which the same segment is inserted twice —
   returns 1. `start_n`/`end_n` gate the `== 1` free-end tests throughout
   `determine_dir_track` and `determine_dir_shower_trajectory`, so such a
   segment is scored differently in the two trees.
2. **A parallel segment is silently aliased, and this was not in §3 at all.**
   `boost::add_edge` on an existing vertex pair returns `added = false` and the
   *existing* descriptor. `PR::add_segment` then executes
   `seg->set_descriptor(desc)` (before the `if (added)`) and
   `g[desc].segment = seg`, so the second segment takes over the first
   segment's edge: the first is orphaned from graph iteration while still
   holding a descriptor it believes is valid. The prototype's
   `map_vertex_segments` holds **both** segments. So the toolkit cannot
   represent a 2-cycle between two vertices; the prototype can.

**What is still unproven is reachability.** `PR::add_segment` has no
`vtx1 != vtx2` guard and no duplicate-pair guard, so both constructions are
*permitted*; whether any of the 63 `add_segment` call sites ever produces one
was not established, and no event was run. If neither is reachable, F9 is
vacuous and should be closed as such — that is a legitimate outcome and is why
this sits at the bottom of the list.

**Solution — measure before fixing.**

1. Add a `WCT_DET_DEBUG`-style counted warning inside `PR::add_segment` for
   both cases (`vtx1 == vtx2`, and `added == false` with a *different* segment
   already on the edge). Zero cost when off, and it is diagnostic only, so it
   needs no knob and no gate.
2. Run it over the nueCC48 arm. If both counts are 0 across 48 events, close
   F9 as vacuous and record the count. If not, the fix follows the count: a
   distinct-neighbour count in place of `boost::degree` at the `start_n`/`end_n`
   sites for the self-loop case, and a real decision (reject? relax to
   `multisetS`?) for the parallel case — the latter is a graph-type change and
   is escalation rule 1 territory of its own.

* **Note the existing comment at `PRGraph.cxx:87-91`** ("Inherit the existing
  edge's graph index so `m_graph_index` is not left at `SIZE_MAX`") — the
  aliasing branch is *known* and was hardened for a different reason (doc pr/28,
  the `SIZE_MAX` class). Nobody appears to have asked what happens to the
  segment that was already there.

### §10.11 What was dropped, and why — one line each

* **P9 — `fits()` vs `get_point_vec()`: not a divergence at all, and this is
  the one §3 claim that must be withdrawn.** The prototype's accessor is
  `std::vector<WCP::Point>& get_point_vec(){return fit_pt_vec;}`
  (`ProtoSegment.h:16`) — it returns the **fitted** points, not the path points.
  The toolkit's `sg->fits()` is therefore the faithful counterpart, and the
  `//wcpts()` left in the source is the porter *rejecting* the wrong one. Moved
  to §5 as 5.14; removed from §3 and §8.
* **P3, score half** — `seg->particle_score(100.0)` at the topology branch
  writes the toolkit's "PID not performed" sentinel where the prototype may
  carry a stale non-100 score into `examine_all_showers`' `tracks_score` sum
  (`:1584` ↔ `:1040`). That is the toolkit **fixing** a prototype defect.
  Improvement, dropped. (The 4-momentum half of P3 is F1 shape B.)
* **P7, guard half** — not crashing on zero-bin `TH1F` is an improvement;
  only the returned value survives, as F6.
* **P8, filter half** — excluding `!valid()` / `dx<=0` / `dQ<0` fits is
  defensible and drops the prototype's `dQ/1e-9` sentinel; only the
  self-inconsistency survives, as F4.
* **P10 — the null-vertex path.** The prototype prints an error and falls
  through, then evaluates `map_vertex_segments[nullptr]`, which
  default-constructs an empty set (and **inserts a null key into the map** as a
  side effect) and yields 0. The toolkit logs and `continue`s. Neither is
  attractive, but the prototype's is not a model worth restoring, and
  `PRGraph.cxx:117-122` documents a null vertex as an anticipated state.
  Improvement, dropped. Residual, stated not fixed: the toolkit's segment
  leaves with no direction and no PID where the prototype's leaves with a
  garbage-input direction.
* **P11 — start/end vertex by 3-D distance instead of `wcpt().index`
  equality.** Exactly the owner's clearance in §10.0. Dropped.
* **P15 — three raw `boost::edges` loops** (`NeutrinoTrackShowerSep.cxx:44 :75
  :332`). §6 read all three and found no accumulator, argmax or cross-segment
  state. Judged benign, dropped; the judgement is still "judged, not proven"
  per §9.

### §10.12 Suggested order of work

Not a schedule — an ordering by (certainty × reach) ÷ cost, so that whoever
picks this up starts where the argument is strongest and the diff smallest.

> **Status update 2026-08-04:** **F2 was taken first, at the owner's
> direction, and is SHIPPED default-OFF — see §11.** The ordering below is
> unchanged for the remaining eight.
>
> **Status update 2026-08-04 (later): the round is DONE — see §12.**
> F5/F6/F3/F1/F4 SHIPPED and SBND DEFAULT ON; F7 shipped dormant; F9
> measured (self-loop vacuous, edge-aliasing live and open); F2 recommended
> OFF (§12.10).

| order | item | why here |
|---|---|---|
| 1 | **F5** (P6) | one line, prototype intent unambiguous, no interaction with anything else |
| 2 | **F6** (P7) | one value, but run the ROOT zero-bin snippet first (§10.7) |
| 3 | **F3** (P13) | two lines *if* `SegmentFlags` already has a clear; proven live; also de-risks F2 |
| 4 | **F1** (P1+P3a+P4) | the largest reach and the only one that moves energy directly — but 15 sites and a helper, so it wants the earlier three out of the way first |
| 5 | **F4** (P8) | one line, but check the `units::cm` scale before landing |
| 6 | **F2** (P2) | do the segment count in §10.3 first; the count decides whether this is cheap or the biggest item here |
| 7 | **F9** (P12) | instrument and measure; may close as vacuous |
| 8 | **F7** (P5) | belongs with pr/30 F4, not on its own |
| — | **F8** (P14) | already doc pr/7's; nothing new to do here |

Every one of these is escalation rule 1 — unconditional production change — so
each ships as a default-OFF knob with a knob-off byte-identical gate, exactly
as §4's existing stage-3 knobs did. Note what §4 already says about that: **every
stage-3 knob on record is ON in the SBND job**, so "default OFF" here means
"off until the owner flips it in `wct-pr-perevt.jsonnet`", not "dormant".

### §10.13 Corrections to earlier sections

1. **P9 is withdrawn as a divergence** (§10.11). §3's P9 and §8's P9 row are
   annotated in place and the finding is restated in §5.14. The divergence
   count in §8 is therefore **14, not 15**.
2. **§3's `NeutrinoTrackShowerSep.cxx` anchors have drifted 1–2 lines at
   HEAD.** `c05bc5f7` ("clus: make T_tagger deterministic — sweep
   boost::edges/vertices/graph_nodes", doc pr/28 §11) is the only commit
   touching this file between `4f2e7303` and `6206c46b`: +22/−24. The drift is
   not uniform, so the re-derived anchors are listed rather than a rule.
   Verified at HEAD:

   | §3 anchor | HEAD | what is there |
   |---|---|---|
   | `:662` | `:661` | `improve_maps_one_in`, guarded (guard itself at `:659`) |
   | `:768 :793 :807` | `:767 :792 :806` | `improve_maps_shower_in_track_out` |
   | `:938 :980 :1022 :1048 :1111 :1164 :1207 :1286` | `:937 :979 :1021 :1047 :1110 :1163 :1206 :1285` | `improve_maps_no_dir_tracks`, eight branches |
   | `:1374` | `:1373` | `improve_maps_multiple_tracks_in` |
   | `:428 :1463 :1867` | `:427 :1462 :1865` | P4/F1-shape-C rest-mass writes |
   | `:1419` | `:1418` | P9's `sg->fits();//wcpts();` |
   | `:1595-1605` | `:1593-1603` | P5/F7 `examine_all_showers` |
   | `:1625` / `:1649` | `:1623` / `:1647` | the 150°/165° pair |
   | `:1586` | `:1584` | the `particle_score() != 100` sentinel |
   | `:123-135` | `:123-135` | unchanged — the topology branch |

   `PRSegmentFunctions.cxx` was untouched between the two revisions, so every
   §3 anchor in that file stands as written.
3. **§7 loose end 2 is closed** — `segment_is_shower_topology` has four call
   sites across two stages, so re-entry on the same segment is the normal path
   (§10.4).
4. **§8's closing sentence is superseded.** "No recommendation is offered, per
   §5 rule 4 and the missing dictionary section" was correct for an audit-only
   document; the owner's filter request authorizes the picks and the solutions
   in this section. §7 loose end 5 (the porting dictionary has no
   topology/PID/direction section) is *unchanged* and is now the natural
   follow-up: the nine items above are what that section should say.

---

## §11 F2 implemented — `shower_topo_proto_dir`, default OFF *(2026-08-04)*

**What was asked.** Owner, after §10.3's corrected verdict: *"Can you work on a
fix for P2 then first."* So this section ships the knob §10.3 argued for, and
nothing else on the list. F1 and F3 are untouched.

**What it is.** A default-OFF config knob that skips the stage-3
`segment_determine_shower_direction` call, leaving a `kShowerTopology` segment
with the direction `segment_is_shower_topology` set — the prototype's state.
OFF reproduces today's path byte for byte; ON is the arm that measures the
question §10.3 could only argue about.

**It is a knob, not a verdict.** §10.3 stands: nobody has shown which direction
estimate is better, and the prototype's own `determine_dir_shower_topology`
says `// hack for now`. What this section establishes is that the question is
now *measurable* and what the answer costs — 7 of 48 events move their
reconstructed neutrino energy, and none of them change selection outcome.

### §11.1 Repro

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
cd $TK && git rev-parse --short HEAD                 # 1e169602 (this change)
                                                     # f8f2150a (the parent)
wcbuild                                              # then the M1 freshness proof

# compiled-config proof, both ways
PIPELINE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,\
tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,\
tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output"
PIPE="pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]"
wcsonnet --tla-code "$PIPE" cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  | grep -c shower_topo_proto_dir                    # 0  -- key absent when off
wcsonnet --tla-code "$PIPE" --tla-code shower_topo_proto_dir=true \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  | grep -c shower_topo_proto_dir                    # 1  -- key present when on

./build/clus/wcdoctest-clus                          # 91/91, rc=0

# the two arms (48 nueCC events, ids from work-pr30-f2on)
cd $TK/sbnd_xin
IDS=$(ls -d work-pr30-f2on/pr_evt* | sed 's/.*pr_evt//' | sort -n | tr '\n' ' ')
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31-f2off48 data $IDS
SBND_SHOWER_TOPO_PROTO_DIR=1 \
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31-f2on48  data $IDS

# the per-segment mechanism on evt 388
for a in off on; do
  [ $a = on ] && E=1 || E=
  SBND_SHOWER_TOPO_PROTO_DIR=$E PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31-disp-$a data 388
done
```

### §11.2 The change

Five files, all additive; nothing existing is edited except to add one `if`.

| file | change |
|---|---|
| `clus/inc/WireCellClus/NeutrinoPatternBase.h` | `bool m_shower_topo_proto_dir{false}` on `PatternAlgorithms`, with the mutation audit and the F3 residual in the comment |
| `clus/inc/WireCellClus/TaggerCheckNeutrino.h` | the same member on the component |
| `clus/src/TaggerCheckNeutrino.cxx` | `configure()` / `default_configuration()` / the copy into `pattern_algos` — the three sites every stage-3 knob uses |
| `clus/src/NeutrinoTrackShowerSep.cxx` | the call site, wrapped |
| `cfg/pgrapher/common/clus.jsonnet` | `tagger_check_neutrino(..., shower_topo_proto_dir=false)` + key suppression |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | threaded through both `pr()` definitions |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | TLA `shower_topo_proto_dir = false`, threaded — **the default is not flipped** |
| `sbnd_xin/run_pr_chain_batch.sh` | `SBND_SHOWER_TOPO_PROTO_DIR=1` emits the TLA; empty emits nothing |

The whole behavioural diff is:

```cpp
} else if (seg->flags_any(SegmentFlags::kShowerTopology)) {
    if (!m_shower_topo_proto_dir) {
        segment_determine_shower_direction(seg, particle_data, recomb_model,
                                           "associate_points", m_mip_dqdx_median,
                                           0.4*units::cm, m_mip_dqdx);
    }
    ...   // the particle-info block is UNCHANGED -- that is F1 shape B, not this
}
```

**Why one `if` is the whole fix, and how that was checked rather than assumed.**
`segment_determine_shower_direction` writes exactly one thing on the segment,
twice — `segment->dirsign(0)` at entry (`PRSegmentFunctions.cxx:2209`) and
`segment->dirsign(flag_dir)` at the end (`:2509`). Every other name it touches
inside its 305 lines is a local. Its return value is discarded at this call
site. Verified by enumerating every `segment->`/`seg->` write in `:2208-2512`;
the two `dirsign` calls are the complete list. So suppressing the call
suppresses precisely the direction overwrite and cannot leak anything else.

**What is deliberately NOT in this change:** the 4-momentum and
`particle_score(100)` writes in the same block. Those are F1 shape B and the
dropped half of P3 respectively (§10.2, §10.11). Bundling them would make the
arm below unattributable — the same coupling trap doc pr/28 §16.5 hit.

### §11.3 Gate — knob OFF is byte-identical

Baseline is **`work-pr30-f2on`**, the doc pr/30 §12.10 arm at parent
`f8f2150a`. It is the right baseline and `work-vfnuecc48-0804` is not: the
latter predates `oov_prototype_parity` becoming the SBND default, so comparing
against it would show doc pr/30's flip, not this change.

| gate | result |
|---|---|
| compiled config, knob off, vs `f8f2150a`'s `cfg/` | **byte-identical**, md5 `0c24101b93143a06e8e326298af821de` both sides |
| compiled config, knob on | key present, +35 bytes — nothing else moves |
| `nusel-table.tsv`, `work-pr31-f2off48` vs `work-pr30-f2on` | **identical** |
| `nusel-events.tsv`, same pair | **identical** |
| `pctree-pr-evt<ID>.tar.gz` member-content hashes, 48 events | **48/48 identical** (`abtest/hash_archive.py`, M2 — never `cmp` on the tarball) |
| `./build/clus/wcdoctest-clus` | 91 cases / 963 assertions, **0 failed**, rc=0 |
| M1 freshness | `local/lib/libWireCellClus.so` 14:17:10 > last source edit 14:15:45 |

**Labels for re-checking later:** `work-pr31-f2off48` (knob off),
`work-pr31-f2on48` (knob on), `work-pr30-f2on` (baseline),
`work-pr31-disp-off` / `work-pr31-disp-on` (evt 388 with the PR display dump).

### §11.4 What the knob does when ON — 48 nueCC events

`work-pr31-f2on48` vs `work-pr31-f2off48`, same binary, same input hub
`work-nuecc48-0804`, same 48 event ids.

**Selection outcome does not move at all.**

| quantity | changed / 48 |
|---|---|
| `event_label` (46 nu-candidate / 2 cosmic-tagged, both arms) | **0** |
| `nu_evaluated`, `n_bundle`, `n_inbeam_bundle` | **0** |
| selected bundle `t0` / `len` / `n_assoc` | **0** |
| every cosmic flag (`cosmic_flag`, `cosmict_flag`, `cosmict_score`, `cosmict_10_score`) | **0** |
| `nusel-table.tsv` and `nusel-events.tsv` (22 per-bundle columns) | **identical** |
| `pctree-pr-evt<ID>.tar.gz` member hashes | **0 of 48 differ** |

That last pair matters and is the same structural fact doc pr/28 §16.3b
established: the whole delta is created **after** `tagger_check_neutrino` has
picked its main cluster, so nothing upstream of that — including the point-cloud
products — can see it.

**What does move** is the reconstruction on the already-selected bundle:

| quantity | moved / 48 | median abs | extremes |
|---|---|---|---|
| `kine_reco_Enu` | **7** | 4.0 MeV | **−754.4 MeV** (evt 360535), **+204.3 MeV** (evt 74544) |
| `numu_score` | **8** | 0.76 | −1.455 (evt 74544), +0.932 (evt 360535) |
| `nue_score` | **1** | 0.60 | −0.600 (evt 268067) |
| neutrino vertex position | **1** | — | 0.78 cm (evt 256587) |

`nue_score > 0` is **40 in both arms**; the 32 events sitting exactly at the
`+4.301` saturation cap stay there and the two `br_filled` sentinels at `−15`
stay too (doc pr/28 §16.5 — this sample structurally cannot resolve improvement
above the cap, so read only the downward moves as information).

**The two large energy moves are the finding worth a scan.** −754 MeV on evt
360535 and +204 MeV on evt 74544 are not noise: the run-to-run floor on this
manifest is **zero** over 17 columns (doc pr/28 §16.4). Whether either is an
improvement is a hand-scan question, not a table question, and this document
does not answer it.

**No new processing problems.** 48/48 rc=0, zero E-level log lines, zero
`DL vertex failed`, no new WARN family. Cost is nil: wall 404.5 → 406.9 s
(**+0.6 %**), core 427.6 → 429.1 s (**+0.3 %**), mean peak RSS 1.525 → 1.525 GB
(**+0.0 %**) — unsurprising, since the knob *removes* a 305-line PCA on some
segments and the arithmetic is dominated by everything else.

### §11.5 The mechanism, on evt 388

`work-pr31-disp-off` vs `work-pr31-disp-on`, both with `PR_EXTRA_STAGES=pr_display`
so the per-segment state is dumped (doc pr/26 stage 1).

* 88 segments in each arm, 42 flagged shower.
* **3 segments flip `dirsign`**: 59084 (`+1 → −1`), 92137 (`+1 → −1`),
  92139 (`−1 → +1`). All three are shower-flagged, as expected — the knob only
  gates the `kShowerTopology` branch.
* `particle_id`, `shower_id` and `cluster_id` are unchanged on every segment
  that exists in both arms; the shower list (12) and vertex list (133) keep
  their lengths; the main vertex is identical to the last decimal.
* **49 of 88 segment ids shift by one** (e.g. `27073 → 27072`, `53078 → 53077`).
  `id = cluster_id*1000 + graph_index`, so this is graph-index drift: one fewer
  segment is created somewhere earlier in the sequence. That is the propagation
  path — a direction flip in stage 3 changes a later structural edit — and it is
  worth naming because it means "3 dirsigns moved" **understates** the reach.
* Downstream on this event: `kine_reco_Enu` 2811.06 → 2809.29 MeV,
  `numu_score` −0.728 → −1.532, `nue_score` unchanged at the cap, and
  `nusel-evt388.tsv` identical.

### §11.6 Residual — what ON does *not* give you

**Knob ON is prototype parity except in one corner, and that corner is F3.**
Today the entry-side `segment->dirsign(0)` inside
`segment_determine_shower_direction` masks a hole: the toolkit's
`segment_is_shower_topology` skips `dirsign` entirely on its four early returns
(`PRSegmentFunctions.cxx:2518-2529`), where the prototype clears `flag_dir` at
`is_shower_topology`'s entry (`ProtoSegment.cxx:321`, *before* its early
returns). With the knob ON that mask is gone.

It bites only for a segment carrying a **stale** `kShowerTopology` flag — i.e.
only in combination with **F3 (P13)**, which is the item that closes it. Not
fixed here on purpose: mixing the two would make neither measurable. Anyone
flipping `shower_topo_proto_dir` on before F3 lands should know this is open.

Also not addressed, and unchanged from §10.3: **which direction estimate is
better.** The numbers above say what changes and what it costs. They do not say
which arm is right. That needs the two Bee sets and a scan, on the events in
§11.4's table.

### §11.7 SBND default: unchanged, deliberately

`wct-pr-perevt.jsonnet` carries `shower_topo_proto_dir = false`. Unlike doc
pr/30's `oov_prototype_parity` — flipped ON the same day because it was
provably a port bug *and* provably free on this manifest — this one is neither:
it is not established as a bug (§10.3), and it is not free (7 events move their
energy, two of them by hundreds of MeV). Flipping it is a §5 rule 1 decision
and it is the owner's, not this document's.

### §11.8 Bee sets for the scan — OFF vs ON on the eight movers

Owner asked for the pair, to decide on-vs-off by scanning rather than by table.

| arm | knob | Bee |
|---|---|---|
| **OFF** — today's toolkit (the stage-3 PCA call runs) | `shower_topo_proto_dir=false` | https://www.phy.bnl.gov/twister/bee/set/908c3ce5-d089-4b00-90ab-68d66701af32/event/list/ |
| **ON** — the prototype's behaviour (call skipped) | `shower_topo_proto_dir=true` | https://www.phy.bnl.gov/twister/bee/set/2a421626-567a-4054-bc39-6bc95df09fe9/event/list/ |

**The two sets carry the same 8 events in the same Bee-index order**, so index
*n* is the same event in both tabs — verified, `pr31_f2off.index.txt` and
`pr31_f2on.index.txt` are identical. Only the eight movers are packaged; the
other 40 events of the manifest are bit-identical between the arms and would be
noise in a scan.

| Bee idx | event | `kine_reco_Enu` OFF → ON (MeV) | Δ | `numu_score` OFF → ON | `nue_score` | vertex |
|---|---|---|---|---|---|---|
| 0 | **360535** | 2213.4 → 1459.0 | **−754.4** | −1.353 → −0.421 | — | — |
| 1 | **74544** | 2082.1 → 2286.3 | **+204.3** | −0.034 → −1.489 | — | — |
| 2 | **256587** | 3835.9 → 3845.9 | +10.0 | −1.687 → −1.808 | — | 0.78 cm |
| 3 | **268067** | 1036.0 → 1040.1 | +4.0 | +0.800 → +0.638 | +2.525 → +1.926 | — |
| 4 | **46363** | 1850.8 → 1854.3 | +3.5 | −1.462 → −0.695 | — | — |
| 5 | **388** | 2811.1 → 2809.3 | −1.8 | −0.728 → −1.532 | — | — |
| 6 | **269774** | 2602.5 → 2603.4 | +0.9 | −0.940 → −1.702 | — | — |
| 7 | **137238** | 1030.5 → 1030.5 | +0.0 | +1.270 → +1.329 | — | — |

**Idx 0 and 1 are the whole question**; idx 2-7 move by ≤ 10 MeV and are
included only so the scan sees the full population of what changed.

Two things to hold in mind while scanning, both from §11.4:

* **No selection outcome differs on any of these eight.** `event_label`,
  `nu_evaluated`, the selected bundle and every cosmic flag are identical, and
  so are both `nusel-*.tsv` files. What is being judged is the *reconstruction
  on an already-selected bundle* — the particle flow and the energy — not
  whether the event is kept.
* **`numu_score` moves in both directions and does not track the energy.** On
  idx 0 the energy falls 754 MeV while `numu_score` rises; on idx 1 the energy
  rises 204 MeV while `numu_score` falls. The BDT weights are uBooNE-trained
  and uncalibrated on SBND (doc pr/2 gap G1), so `numu_score` is not the
  tiebreak — the particle flow in the display is.

Assets: `sbnd_xin/bee/pr31-f2/pr31_f2{off,on}.{zip,index.txt,prid-map.txt,url}`
(not git-tracked, by the existing convention). Arms `work-pr31-f2off48` /
`work-pr31-f2on48`.

**Still OFF pending the scan** (§11.7), and F3 should land before any flip
(§11.6).  *(2026-08-04, later: F3 landed and is SBND ON; the residual is
closed by measurement, and the formal recommendation is OFF — §12.10.)*

---

## §12 The §10.12 round implemented — five fixes SBND DEFAULT ON, F9 measured, F2 recommended OFF *(2026-08-04)*

**What was asked.** The owner: *"implement the fixes as suggested by §10.12
… use the 48 nueCC events as guidance and validations … we want to keep
improvements and bug fixes for the production runs [i.e. turned on] … and
provide a recommendation on whether we should turn on this knob for F2."*

**What shipped** (toolkit + this repo, one commit each):

| item | knob | C++ default | SBND production | result on nueCC48 |
|---|---|---|---|---|
| F5 (P6) | `cont_muon_dir3_30cm` | false | **ON** | null — bit-identical |
| F6 (P7) | `track_comp_empty_abstain` | false | **ON** | null — bit-identical |
| F3 (P13) | `shower_topo_reset` | false | **ON** | null — bit-identical |
| F1 (P1+P3a+P4) | `reclass_preserve_4mom` | false | **ON** | null — bit-identical (after §12.5 rework; **fires on 47/48 events**) |
| F4 (P8) | `dir_track_median_local` | false | **ON** | null — bit-identical |
| F7 (P5) | `examine_showers_vertex_by_index` | false | **OFF** (dormant, pending pr/30 F4) | null — bit-identical |
| F9 (P12) | — (counters only) | — | — | self-loop **0**; **edge-alias 35 on 23/48 events** — NOT vacuous, §12.6 |
| F8 (P14) | — | — | — | unchanged; still doc pr/7's |
| F2 (P2) | `shower_topo_proto_dir` (pre-existing, §11) | false | **OFF — recommended OFF, §12.10** | the §11.4 movers, unchanged by F3 |

"Null" everywhere below means: 48/48 `pctree-pr` member-content hashes
identical, `nusel-table.tsv` + `nusel-events.tsv` byte-identical, **every**
`tracking-pr.root` `T_tagger`/`T_kine` leaf identical, and (checked for F1)
every `mabc-pr.zip` member identical.

### §12.1 Repro

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
cd $TK && git rev-parse --short HEAD        # this change; parent 407c5ba9
wcbuild                                     # M1 freshness proof after
./build/clus/wcdoctest-clus                 # 95 cases / 984 assertions, rc=0

# compiled-config proofs (pipeline TLA required -- wcsonnet without it builds
# no TaggerCheckNeutrino node).  Knobs-off config at the parent == md5
# 3b6d0374b72f5d549daef754488e6d12 == this change with the six keys absent;
# after the §12.8 flips the five keys compile to true and F7/F2 stay absent.
PIPELINE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,\
tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,\
tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output"
PIPE="pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]"
wcsonnet --tla-code "$PIPE" cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  | grep -o '"cont_muon_dir3_30cm" : [a-z]*'

# arms (owner authorized 24 CPUs: PR_JOBS=12, two arms at a time)
cd $TK/sbnd_xin
IDS=$(ls -d work-pr32r2-allon48/pr_evt* | sed 's/.*pr_evt//' | sort -n | tr '\n' ' ')
PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-off48 data $IDS
SBND_CONT_MUON_DIR3_30CM=1      PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f5on48  data $IDS
SBND_TRACK_COMP_EMPTY_ABSTAIN=1 PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f6on48  data $IDS
SBND_SHOWER_TOPO_RESET=1        PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f3on48  data $IDS
SBND_DIR_TRACK_MEDIAN_LOCAL=1   PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f4on48  data $IDS
SBND_RECLASS_PRESERVE_4MOM=1    PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f1onb48 data $IDS
SBND_EXAMINE_SHOWERS_VTX_BY_INDEX=1 PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f7on48 data $IDS
SBND_SHOWER_TOPO_PROTO_DIR=1    PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f2on48  data $IDS
SBND_SHOWER_TOPO_PROTO_DIR=1 SBND_SHOWER_TOPO_RESET=1 \
  PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-f2f3on48 data $IDS
SBND_CONT_MUON_DIR3_30CM=1 SBND_TRACK_COMP_EMPTY_ABSTAIN=1 SBND_SHOWER_TOPO_RESET=1 \
SBND_RECLASS_PRESERVE_4MOM=1 SBND_DIR_TRACK_MEDIAN_LOCAL=1 \
  PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-allonb48 data $IDS
# after the cfg flips: bare run must equal the env-forced all-on arm
PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-0804 work-pr31r2-prod48 data $IDS

# comparisons
python3 pr32_cmp.py work-pr32r2-allon48 work-pr31r2-off48      # the gate
python3 pr32_cmp.py work-pr31r2-off48   work-pr31r2-<arm>      # per knob
python3 pr32_cmp.py work-pr31r2-allonb48 work-pr31r2-prod48    # bare == prod
# T_tagger/T_kine leaf-level and mabc-pr.zip member comparisons: ad hoc uproot/
# zipfile sweeps (this round; scripts /home/xqian/tmp/pr31r2/{arm_delta,root_leaf_cmp}.py,
# not committed -- pr32_cmp.py's docstring line 3 promises the leaf compare but
# does not implement it, a gap worth closing next round)
```

The crashed first F1 arm is preserved as the record of §12.5:
`work-pr31r2-f1on48` (1/48 rc=0) and `work-pr31r2-allon48` (same build,
1/48).  `work-pr31r2-off48b` re-proves the gate after the §12.5 rework
(48/48 identical to `work-pr31r2-off48`), which is what keeps the
pre-rework per-knob arms valid.

### §12.2 The knobs and their threading

Every knob follows the `shower_topo_proto_dir` pattern (§11.2): member +
rationale in `NeutrinoPatternBase.h`, member in `TaggerCheckNeutrino.h`,
three sites in `TaggerCheckNeutrino.cxx`, arg + key-suppression in
`cfg/pgrapher/common/clus.jsonnet`, defaults + threading in BOTH `pr()`
definitions of `sbnd/clus.jsonnet`, TLA + thread in `wct-pr-perevt.jsonnet`,
and a runner env var — the pr/31 knobs use the pr/32 **tri-state loop**
(unset = cfg default, 1 = force on, 0 = force off), which is what lets the
gate arm force production knobs off without editing cfg/.

Free-function transport, where the knob leaves `PatternAlgorithms`:

* **F6, F4** ride `TrackPidOptions` (new fields `track_comp_empty_abstain`,
  `dir_track_median_local`), filled by `track_pid_options()`.  F6 is passed
  down to `do_track_comp` as a trailing arg at its 4 call sites (all inside
  `segment_do_track_pid`).  F4's interior reach: the trajectory path
  (`:1904`) already forwards `pid_opts`; the short-segment interior call
  inside `segment_determine_shower_direction` did NOT forward, so that
  function grew a trailing `median_local` and passes a locally-constructed
  default `TrackPidOptions` carrying only this field — forwarding a caller's
  full option set there would have unconditionally switched
  `proton_dir_vote` et al. at that site.
* **F3** is a trailing `reset` param on `segment_is_shower_topology`,
  threaded at its 4 call sites (NTSS:54, NVF improve_vertex ×3).
* **F5, F7, F1** live entirely in `PatternAlgorithms` methods — no threading.

### §12.3 Gate — everything off is byte-identical (and the flips change nothing)

| comparison | result |
|---|---|
| `work-pr31r2-off48` vs `work-pr32r2-allon48` (production at `407c5ba9`) | 48/48 hashes, both TSVs, all PR30/PR32 audit counters, **all 16 physics score columns** identical |
| compiled config, knobs off | md5 `3b6d0374b72f5d549daef754488e6d12` == parent |
| `work-pr31r2-off48b` vs `work-pr31r2-off48` (§12.5 rework identity) | 48/48 identical |
| `work-pr31r2-prod48` (bare, after flips) vs `work-pr31r2-allonb48` | 48/48 identical — **bare run == production** (doc 68 invariant) |
| `./build/clus/wcdoctest-clus` | 95 cases / 984 assertions, PASS |

Wall/RSS: like-for-like pairs (arms run concurrently under the same load)
differ by ≲ 1% core-time with identical 1.47 GiB peak RSS; cross-session
differences up to ~9% are load noise from the 24-job schedule, not the knobs.

### §12.4 Per-knob ON arms — all five fixes are NULL on nueCC48

> **CORRECTED 2026-08-05 — F3 is not null. See §12.12.** The heading and the
> "every persisted product" claim below are wrong; the enumeration that
> follows them is not. Everything this section actually opened was identical.

Every per-knob arm (`f5on`, `f6on`, `f3on`, `f4on`, `f1onb`, `f7on`, and the
joint `allonb`) is **bit-identical to the off arm** on every persisted
product: pctree hashes 48/48, both nusel TSVs, every `T_tagger`/`T_kine`
leaf.  For F1 the `mabc-pr.zip` members were additionally checked: 0/48
differ.

Null does **not** mean the knobs are inert:

* **F1 is proven engaged** — its first implementation crashed 47/48 events
  at exactly the preserve-path states (§12.5), so the path fires on nearly
  every event; the rewritten 4-momenta are simply never consumed by any
  persisted output on this manifest.  The 4-momentum written at the fifteen
  reclassification sites is, today, **write-only state** downstream of the
  guards themselves.
* **F5, F6, F4, F3, F7 have no engagement counter** (nothing crashes, so no
  side channel).  For these, "null" bounds the reach at zero *observable*
  firings on 48 events; the divergent input states (empty comparison window,
  filtered-out fit sample, re-tested stale flag, short-reference/long-
  neighbour pair, order-flipped vertex pair) are evidently rare.  A larger
  manifest (572-event valfast) is where any of them would first show.

The flips are therefore **safe by measurement** (they change no production
number today) and justified by intent: each closes a latent divergence class
that would otherwise surface as an unexplained A/B diff in some future
sample, attributed to whatever change happened to be under test that day.

### §12.5 F1 crashed on first contact — the toolkit-only invariant is real, and load-bearing

* **Symptom.** The first F1-ON arm (`work-pr31r2-f1on48`) died on 47/48
  events: `ParticleInfo: total energy cannot be less than rest mass`
  (`aux/src/ParticleInfo.cxx` `validate_inputs`), thrown from
  `TaggerCheckNeutrino::visit`.
* **Root cause.** §10.2's helper sketch returned the preserved 4-momentum
  *into the validating constructor*.  The two states the prototype holds as
  a matter of course — all-zero (never computed) and an old on-shell
  4-momentum carried across a mass change — are exactly the two states
  `validate_inputs` forbids (`E < m`; energy-momentum relation).  §10.2
  predicted the first ("any consumer choking on it was reading a
  toolkit-only invariant") but did not follow through to *where* that
  invariant is enforced: at construction, unconditionally.
* **Why it hid.** The knob-off path never constructs those states, so the
  gate, the doctests and every other arm were clean; only the ON arm could
  reveal it — which is what per-knob arms are for.
* **Fix.** `reclass_pinfo()` (NeutrinoTrackShowerSep.cxx) constructs a legal
  rest-mass placeholder and then writes the carried value through
  `set_four_momentum()` — the class's own **non-validating** setter — so KE
  lands at `E − m` (i.e. `−m` for never-computed), which is precisely what a
  prototype consumer subtracting mass sees.  The shared `aux` validator is
  untouched (it even carries a commented-out "Allow zero 4-momentum as a
  placeholder" block for exactly this state — left as the owner's call).
  The 12 recompute sites collapsed to one-line `reclass_pinfo(...)` calls;
  the 3 shape-C sites keep their legacy rest-mass expression verbatim on an
  explicit else-branch.
* **Verification.** `work-pr31r2-off48b` == `work-pr31r2-off48` 48/48 (the
  rework's off-path is byte-identical), `work-pr31r2-f1onb48` completes
  48/48 rc=0 and is null per §12.4.

### §12.6 F9 measured — self-loop closes vacuous, edge-aliasing does NOT

Two `PortAuditCounters` (`selfloop_segment`, `edge_aliased`) with bounded
DEBUG logs in `PR::add_segment`, emitted per event on the new `PR31AUDIT`
log line (same contract as PR30/PR32AUDIT).  On the off arm:

* **`selfloop_segment` = 0 across all 48 events.**  The self-loop half of
  F9 (§10.10 fact 1: `boost::degree` counts a self-loop twice) is **closed
  as vacuous**, with the count on record.
* **`edge_aliased` = 35 total on 23/48 events** (15 events ×1, 5 ×2, 1 ×3,
  1 ×7).  The parallel-segment half is **REACHABLE AND LIVE**: a second
  segment on an existing vertex pair silently takes over the first
  segment's edge, orphaning it from graph iteration while it still holds a
  descriptor it believes valid.  Every logged case so far has
  `old nwcpts == new nwcpts`, consistent with a *successor object for the
  same physical segment* (re-tracking) rather than a genuine 2-cycle — in
  which case the overwrite is the intended semantics and the orphaned
  object is garbage anyway — but that reading is inferred from the wcpt
  counts, not proven by identity.  Per §10.10 the fix decision (reject?
  `multisetS`?) is a graph-type change and is NOT taken here: **reported,
  counted, left open.**  Next step when picked up: log the old/new segment
  identities at a handful of sites and classify successor-vs-parallel.

### §12.7 F2 + F3 jointly — F3 changes nothing under F2

`work-pr31r2-f2f3on48` (F2+F3) vs `work-pr31r2-f2on48` (F2 alone, both at
this HEAD): **48/48 identical on every product and leaf**.  And F2-alone at
this HEAD reproduces §11.4's mover table to the last decimal (8 movers,
−754.4 MeV on evt 360535, +204.3 on evt 74544, zero selection changes) — so
the pr/32 flips did not change F2's phenomenology, and §11.6's residual
(stale-flag segments keeping a direction once F2 removes the entry-side
`dirsign(0)`) does not fire on this manifest.  The residual is now closed
*by measurement in production*, since F3 is ON.

### §12.8 The SBND operating point after this round

`wct-pr-perevt.jsonnet`: `cont_muon_dir3_30cm`, `track_comp_empty_abstain`,
`shower_topo_reset`, `reclass_preserve_4mom`, `dir_track_median_local` all
**true** (owner's standing instruction: production keeps bug fixes and
improvements); `examine_showers_vertex_by_index` and `shower_topo_proto_dir`
**false**.  Bare-run == production re-proven (§12.3 last row).  The runner's
tri-state env vars are the per-arm escape hatch.

### §12.9 F7 and F8 — deliberately not moved

* **F7** is implemented (site-local graph-index ordering ahead of the
  asymmetric 165°/150° branches) but stays OFF: pr/30 F4 owns the
  find_vertices-ordering decision and all three known order-sensitive
  callers should move together.  Its ON arm is null on nueCC48 anyway.
* **F8** remains doc pr/7's — re-stated, nothing new here.

### §12.10 The F2 recommendation: keep it OFF

The owner asked for a recommendation on `shower_topo_proto_dir`.
**Recommendation: OFF — keep the toolkit's PCA direction.**  The grounds,
in order of weight:

1. **The owner's own filter for this round** was "keep improvements and bug
   fixes".  F2-ON does not restore a prototype *decision* — the prototype's
   `determine_dir_shower_topology` is a self-declared `// hack for now` with
   both direction blocks commented out (§10.3).  The toolkit's 305-line PCA
   is a real algorithm with declared inputs; on the discriminator of this
   whole audit (intent vs accident) the *call site* was an accident but the
   *algorithm* is the better-evidenced physics.  Turning F2 ON would remove
   measured behaviour in favour of an explicitly unfinished one.
2. **The residual that made F2 risky is gone.** §11.6's blocker — stale
   directions surviving the skipped entry-side `dirsign(0)` — is closed by
   F3 being production-ON, and measured not to fire regardless (§12.7).
   So OFF is not "blocked", it is a free choice on the physics.
3. **What ON would buy is unarbitratable here.** nueCC48 is real data — no
   MC truth exists (the sample is the 2025-fall Lynn candidate set; the
   only truth-bearing events in this validation universe are the 23 MC
   events, 10 of them nu-evaluated).  The two big movers (evt 360535
   −754 MeV, evt 74544 +204 MeV) change no selection outcome, and
   `numu_score` moves in both directions without tracking the energy
   (uncalibrated uBooNE weights, doc pr/2 gap G1).  The §11.8 Bee pair
   remains the owner's instrument for overturning this recommendation by
   hand-scan — the particle flow on those two events is the whole question.

If the scan does favour ON: flip `shower_topo_proto_dir` in
`wct-pr-perevt.jsonnet` only, and re-run the §11.5 display pair first —
with F3 now ON the mechanism plot changes meaning (the entry-side reset is
active in both arms).

### §12.11 Status of the §10.12 order

| item | status after this round |
|---|---|
| F5, F6, F3, F1, F4 | **SHIPPED, SBND DEFAULT ON** |
| F2 | shipped §11; **recommended OFF (§12.10)**, owner's scan pending |
| F7 | shipped dormant; waits on pr/30 F4 |
| F8 | doc pr/7, unchanged |
| F9 | self-loop closed vacuous; **edge-aliasing open** (§12.6) |

The porting dictionary still has no topology/PID/direction section (§7
loose end 5) — unchanged, and the §10 items plus this section are its raw
material.

---

### §12.12 Correction — F3 is **not** null (doc pr/37 §2.3, 2026-08-05)

**Symptom.** §12.4's heading says all five fixes are null on nueCC48 and its
first sentence claims bit-identity "on every persisted product". Both are
wrong for **F3 `shower_topo_reset`**, which is an SBND production default.

**Root cause.** The sentence generalizes past what the gate opened. §12.3 is
honest about the coverage — *"every `tracking-pr.root` `T_tagger`/`T_kine`
leaf"* — but `tracking-pr.root` carries **seven** trees:
`T_bad_ch Trun T_proj_data T_rec_charge T_proj T_tagger T_kine`. The movement
landed in a third one.

**Measurement** (doc pr/37 §2, `pr36_cmp.py`, no re-run — the arms already
existed): `work-pr31r2-off48b` vs `work-pr31r2-f3on48` is **47/48** on trees,
the one event being **evt 52672**, `T_rec_charge`, branch **`flag_shower`** —
**23 of 501 entries flip 1 → 0**, contiguous at index 125–147. `mabc` 48/48,
`pctree` 48/48, nusel TSVs 48/48, `T_tagger`/`T_kine` untouched, so §12.4's
enumerated channels were all correctly reported.

**Not a binary confound.** `work-pr31r2-off48` vs `off48b` — two same-config
re-runs — is 48/48 on every tree and every branch, so the movement is F3's and
not the §12.5 rework's build.

**What survives.** The flip decision. No score, no verdict, no Bee member, no
pctree entry and no nusel row moves; the §12.4 bullet's reasoning about the
other four knobs is untouched. What changes is that F3 now has a **measured
firing** on this manifest rather than a bounded-at-zero reach, and the
"a larger manifest is where any of them would first show" sentence no longer
applies to it.

**Why it hid.** The comparator, not the physics — the same instrument gap doc
pr/37 §2.1 traces across four rounds. `valfast/vf_tree_compare_all.py` (added
2026-08-05) is the fix: all seven trees, exact, and it reproduces this exact
line as its known-different calibration case.

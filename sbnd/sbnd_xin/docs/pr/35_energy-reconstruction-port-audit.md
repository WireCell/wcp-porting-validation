# doc pr/35 — Energy reconstruction: prototype ↔ toolkit fidelity audit

**Why.** Eighth in the port-audit series (pr/28 vertex fit + trajectory dQ/dx,
pr/29 Steiner graph build, pr/30 proto-vertex + segment finding, pr/31
topology/PID/direction, pr/32 neutrino vertex ID, pr/33 EM shower clustering,
pr/34 particle flow). This one covers **step 7 of the eight** in doc pr/27 §0,
defined by doc pr/27 §9: charge → energy per segment and per shower, and the
`KineInfo` summary tree.

**Status. AUDIT ONLY. No code was changed. No knob was added. No event was
run.** Every finding below is the owner's call. The owner's standing
instruction from the pr/29 round ("Please do not change any code yet") governs.

**Headline.** The `fill_kine_tree` skeleton is a faithful translation — the
two-pass walk, the `flag_reduce` rest-mass bookkeeping, the remaining-shower
loop and its type-3 `included` encoding, and the whole π⁰ block all match. So
does every piece of arithmetic in the charge→energy conversion: the
min/med/max plane selection, the asymmetry switch, the dE/dx clamp algebra,
the first/last-point `dx` shortening. What does **not** match sits in three
places:

1. **`cal_corr_factor` is a stub that returns 1.0** (`NeutrinoEnergyReco.cxx:14`).
   The prototype's applies a per-point position correction that is **live in
   production** (`flag_calib_corr` defaults to 1). Every toolkit `kine_charge`
   is uncorrected.
2. **`Shower::calculate_kinematics` accumulates `vec_dQ`/`vec_dx` over an
   unordered set whose hash is derived from pointers** (`PRShower.cxx:1134`).
   That is a determinism regression the toolkit introduced — the same file
   uses the deterministic helper 75 lines later.
3. **The shower PDG is read from a cached field, not from the live start
   segment** (four sites). The cache's refresh path covers most, but not all,
   of the places a start segment's PDG changes — P1 states exactly which.

**Severity note.** Unlike pr/34, this stage is **not** display-only. Its
product, `KineInfo`, is a BDT input: `kine_reco_Enu` is variable 69 of the numu
XGBoost feature vector (`root/src/UbooneNumuBDTScorer.cxx:234`, `:507`) and is
registered in the nue reader too (`root/src/UbooneNueBDTScorer.cxx:332`,
`:1660`). Anything that moves `kine_reco_Enu` moves both scores.

**Provenance — good news this round.** pr/34 GOTCHA 1 established that
`prototype_base/pid` is not pristine upstream (+5833/−989 over 26 files vs the
merge-base `a5fc0b9`). For **this** stage the exposure is nil:
`NeutrinoID_energy_reco.h` and `NeutrinoID_kine.h` do not appear in that diff
at all, and the only change to `ProtoSegment.cxx` inside the energy functions
is one commented-out `std::cout` in `cal_kine_dQdx` and one in `cal_4mom`. The
citations below are therefore against upstream code. `WCShower.cxx` is likewise
absent from the diff. This does **not** relieve pr/28–pr/33 of the re-check
recorded as pr/34 §7.7.

---

## Repro

Read-only. Toolkit read at **`23bd6783`** (working tree clean of tracked
modifications at the time of reading); snapshots in
`/home/xqian/tmp/claude-25225/pr35/`.

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit

# --- toolkit side -----------------------------------------------------
git rev-parse HEAD                                  # 23bd6783
sed -n '14,35p'   clus/src/NeutrinoEnergyReco.cxx   # cal_corr_factor (the stub)
sed -n '48,189p'  clus/src/NeutrinoEnergyReco.cxx   # kine_charge_from_maps
sed -n '247,278p' clus/src/NeutrinoEnergyReco.cxx   # cal_kine_charge(SegmentPtr)
sed -n '281,348p' clus/src/NeutrinoEnergyReco.cxx   # calculate_shower_kinematics
sed -n '18,21p'   clus/src/NeutrinoKinematics.cxx   # init_tagger_info
sed -n '43,327p'  clus/src/NeutrinoKinematics.cxx   # fill_kine_tree
sed -n '1214,1305p' clus/src/PRSegmentFunctions.cxx # segment_cal_kine_dQdx, cal_kine_dQdx
sed -n '1387,1415p' clus/src/PRSegmentFunctions.cxx # cal_kine_range
sed -n '1604,1635p' clus/src/PRSegmentFunctions.cxx # segment_cal_4mom
sed -n '930,1160p'  clus/src/PRShower.cxx           # Shower::calculate_kinematics
sed -n '1186,1260p' clus/src/PRShower.cxx           # calculate_kinematics_long_muon

# P3 -- the unordered accumulation, and the helper it should have used
sed -n '1134p'   clus/src/PRShower.cxx              # for (auto edesc : this->edges())
sed -n '1209p'   clus/src/PRShower.cxx              # for (... : ordered_edges(...))
sed -n '113p'    clus/inc/WireCellClus/PRTrajectoryView.h   # edges() -> edge_unordered_set
sed -n '123,143p' clus/inc/WireCellClus/PRGraphType.h       # EdgeDescriptorHash (hashes node descriptors)
sed -n '91,93p'  clus/inc/WireCellClus/PRGraphType.h        # setS vertices, setS edges => void* descriptors
sed -n '175,181p' clus/inc/WireCellClus/PRGraphType.h       # the in-tree WARNING about this order

# P1 -- the four cached-PDG reads, and where the cache is (not) refreshed
grep -n 'get_particle_type()' clus/src/NeutrinoKinematics.cxx        # :109 :119 :274 :286
sed -n '939p;1050p'  clus/src/PRShower.cxx          # the only two data.particle_type writes
sed -n '303p'    clus/src/NeutrinoEnergyReco.cxx    # the flag_kinematics latch
grep -n 'set_pdg(\|set_particle_type(' clus/src/NeutrinoShowerClustering.cxx

# P5 -- SCE
sed -n '266p'    clus/src/TaggerCheckNeutrino.cxx   # clus_geom_helper defaults ""
grep -n 'clus_geom_helper' cfg/pgrapher/experiment/sbnd/clus.jsonnet  # (no hit)

# --- prototype side ---------------------------------------------------
cd prototype_base/pid
git merge-base port origin/master                   # a5fc0b9
git diff a5fc0b9..HEAD --stat -- src/NeutrinoID_energy_reco.h src/NeutrinoID_kine.h src/WCShower.cxx
                                                    # (empty -- this stage is pristine)
git diff a5fc0b9..HEAD -- src/ProtoSegment.cxx | grep -c '^[+-][^+-]'   # 20, all comments + particle_score
sed -n '1,41p'    src/NeutrinoID_energy_reco.h      # collect_2D_charges
sed -n '44,252p'  src/NeutrinoID_energy_reco.h      # cal_kine_charge(WCShower*)
sed -n '255,272p' src/NeutrinoID_energy_reco.h      # cal_corr_factor (the real one)
sed -n '275,455p' src/NeutrinoID_energy_reco.h      # cal_kine_charge(ProtoSegment*)
sed -n '1,283p'   src/NeutrinoID_kine.h             # fill_kine_tree
sed -n '1407,1426p' src/NeutrinoID_shower_clustering.h   # calculate_shower_kinematics
sed -n '1316,1450p' src/ProtoSegment.cxx            # cal_kine_dQdx x2, cal_kine_range x3, cal_4mom
sed -n '288,526p' src/WCShower.cxx                  # calculate_kinematics{,_long_muon}
sed -n '2224,3417p' src/NeutrinoID.cxx              # init_tagger_info (1193 lines)
sed -n '190,198p' apps/wire-cell-prod-nue.cxx       # flag_calib_corr==1 -> init_corr_files()
sed -n '32p'      apps/wire-cell-prod-nue.cxx       # int flag_calib_corr = 1;
sed -n '571,600p' ../data/src/TPCParams.cxx         # init_corr_files sets flag_corr = true

# --- the mechanical tagger-default comparison (section 2.1) ------------
python3 ../../scripts/analysis/pr35/cmp_tagger_defaults.py     # from docs/pr/
#   proto assigned : 1023 / toolkit members: 1024 / VALUE MISMATCH: 0

# --- P2: is the prototype's position correction actually on? ----------
grep -n 'wire-cell-prod' prototype_base/run_5384.pl            # no -q flag passed
sed -n '54p'  prototype_base/pid/apps/wire-cell-prod-nue.cxx   # case 'q'
sed -n '32p;196p' prototype_base/pid/apps/wire-cell-prod-nue-port.cxx
sed -n '153p' prototype_base/data/inc/WCPData/TPCParams.h      # default calib paths
ls -la prototype_base/input_data_files/calib_[uvw]_corr.txt    # all three present

# --- P13: who reads KineInfo? -----------------------------------------
grep -n 'get_kine_info()' clus/src/*.cxx root/src/*.cxx        # 5 hits, no tagger
```

---

## §0 Scope

### Function map

| # | prototype | toolkit | read |
|---|---|---|---|
| 1 | `NeutrinoID::cal_corr_factor` `energy_reco.h:255` | `PatternAlgorithms::cal_corr_factor` `NeutrinoEnergyReco.cxx:14` | full |
| 2 | `NeutrinoID::collect_2D_charges` `energy_reco.h:1` | `collect_charge_maps` `NeutrinoEnergyReco.cxx:226` → `TrackFitting::collect_2D_charge` | full |
| 3 | `NeutrinoID::cal_kine_charge(WCShower*)` `energy_reco.h:44` | `cal_kine_charge(ShowerPtr, …)` `:194` / `:236` + `kine_charge_from_maps` `:48` | full |
| 4 | `NeutrinoID::cal_kine_charge(ProtoSegment*)` `energy_reco.h:275` | `cal_kine_charge(SegmentPtr, …)` `:247` | full |
| 5 | `NeutrinoID::calculate_shower_kinematics` `shower_clustering.h:1407` | `calculate_shower_kinematics` `NeutrinoEnergyReco.cxx:281` | full |
| 6 | `NeutrinoID::init_tagger_info` `NeutrinoID.cxx:2224` | `init_tagger_info` `NeutrinoKinematics.cxx:18` | mechanical (§2.1) |
| 7 | `NeutrinoID::fill_kine_tree` `kine.h:1` | `fill_kine_tree` `NeutrinoKinematics.cxx:43` | full |
| 8 | `ProtoSegment::cal_kine_dQdx()` `ProtoSegment.cxx:1341` | `segment_cal_kine_dQdx` `PRSegmentFunctions.cxx:1214` | full |
| 9 | `ProtoSegment::cal_kine_dQdx(vQ,vx)` `:1316` | `cal_kine_dQdx(vQ,vx,recomb)` `:1274` | full |
| 10 | `ProtoSegment::cal_kine_range()` ×3 `:1380 :1393 :1408` | `cal_kine_range(L,pdg,pdata)` `:1387` | full |
| 11 | `ProtoSegment::cal_4mom` `:1420` | `segment_cal_4mom` `:1604` | full |
| 12 | `WCShower::calculate_kinematics` `WCShower.cxx:339` | `Shower::calculate_kinematics` `PRShower.cxx:930` | full |
| 13 | `WCShower::calculate_kinematics_long_muon` `WCShower.cxx:288` | `Shower::calculate_kinematics_long_muon` `PRShower.cxx:1186` | full |

### Explicitly not audited

- **The recombination model itself.** `IRecombinationModel::dE` replaces the
  prototype's hard-coded Box inversion. That substitution is doc **pr/10**'s
  deliberate work (`project_energy_calibration_round`,
  `project_kine_energy_config_knobs`) — §5.5 records why it is not a P-number
  here, per M15 / CLAUDE.md §5 rule 4.
- **`TrackFitting::collect_2D_charge`'s upstream** — where `m_charge_data`
  comes from. That is pr/28's territory.
- **`Shower::calculate_kinematics`'s `start_point` / `init_dir` branches** were
  read and their branch structure compared, but the geometric helpers they
  call (`shower_get_closest_point`, `shower_cal_dir_3vector`) were not opened.
  Flagged in §9.
- **The BDT feature vectors.** `kine_reco_Enu` is shown to be a consumer; the
  scorers themselves are pr/27 §10's stage.
- **The rest of `PRShower.cxx`** — still largely unread (pr/33 GOTCHA 13).

---

## §1 Trust tiers

Carried from pr/28 §3b through pr/34.

- **Tier A — read line by line, both sides, this session.** `cal_corr_factor`,
  `kine_charge_from_maps` vs both prototype `cal_kine_charge` bodies,
  `calculate_shower_kinematics`, `fill_kine_tree`, the four `ProtoSegment`
  energy primitives, `WCShower::calculate_kinematics{,_long_muon}` vs their
  toolkit counterparts, `init_tagger_info` (mechanically, §2.1). Findings
  P1–P14 are all Tier A.
- **Tier B — structure compared, bodies not opened.** the geometric helpers
  named above; `TrackFitting::collect_2D_charge` beyond its first loop (read
  far enough to establish §5.6); `ParticleDataSet::get_range_function`'s curve
  data vs the prototype's `TGraph`s.

---

## §2 What matches

### §2.1 `init_tagger_info` — 1023 assignments, zero value mismatches

The toolkit replaced a 1193-line assignment list
(`NeutrinoID.cxx:2224-3417`) with `ti = TaggerInfo{}` and a struct full of
default-member-initializers. The in-code comment
(`NeutrinoKinematics.cxx:14-17`) asserts equivalence; nobody had checked it.

`/home/xqian/tmp/claude-25225/pr35/cmp_tagger.py` extracts every
`tagger_info.<name> = <value>;` from the prototype's function body and every
default-member-initializer from `struct TaggerInfo`
(`NeutrinoTaggerInfo.h:68-1400`), normalises `true`/`false` to `1`/`0` and
numeric literals, and diffs:

```
proto assigned : 1023
toolkit members: 1024
=== VALUE MISMATCH (0) ===
=== in prototype init, ABSENT from toolkit struct (3) ===
  shw_sp_br3_7_shower_main_length   proto=0
  br3_7_shower_main_length          proto=0
  numu_cc_3_acc_track_length        proto=0
=== in toolkit struct, NOT initialised by prototype init (4) ===
  br3_7_main_length                 toolkit=0
  match_isFC                        toolkit=0
  numu_cc_3_track_length            toolkit=0
  shw_sp_br3_7_main_length          toolkit=0
```

The three/four asymmetries are two **renames**
(`{shw_sp_,}br3_7_shower_main_length` → `{shw_sp_,}br3_7_main_length`,
`numu_cc_3_acc_track_length` → `numu_cc_3_track_length`) plus **`match_isFC`**,
which is toolkit-only (written at `TaggerCheckNeutrino.cxx:853`) and value-
initialises to 0. All four carry the same default either way. **The
replacement is exact.** This is a positive result and the script is cheap to
re-run after any `TaggerInfo` edit.

### §2.2 `ParticleInfo`'s kinetic energy equals the prototype's `kenergy_best`

pr/34 §7.6 left `ParticleInfo::update_kinematics` as an open question. It is
answered here for the path this stage depends on. `segment_cal_4mom`
(`PRSegmentFunctions.cxx:1604-1634`) is a line-for-line translation of
`ProtoSegment::cal_4mom` (`:1420-1447`) — same `length < 4 cm` /
`kShowerTrajectory` / else ladder, same `E = KE + m`, same
`p = sqrt(E² − m²)`, same direction vector. The prototype then stores
`kenergy_best = kine_energy` directly. The toolkit instead constructs

```cpp
auto pinfo = std::make_shared<Aux::ParticleInfo>(pdg_code, mass, name, four_momentum);
```

and `ParticleInfo`'s 4-momentum constructor (`aux/src/ParticleInfo.cxx:45-60`, the assignment at `:59`)
computes `m_kinetic_energy = m_four_momentum.e() - m_mass`. Since
`four_momentum[0]` was built as `kine_energy + particle_mass`, the round trip
is exact and `particle_info()->kinetic_energy()` **is** the prototype's
`kenergy_best`. `fill_kine_tree`'s `kine_best` for tracks
(`NeutrinoKinematics.cxx:140`) is therefore the right quantity.

Caveat, not a finding here: this holds only where `segment_cal_4mom` actually
ran. pr/31 P1 records the `cal_4mom` guard being dropped at 11 of 13 sites;
where it never ran, `particle_info()` is absent and the toolkit's
`push_segment_kine` publishes `kine_best = 0` (`:135-141`). That is pr/31's
finding, not a new one.

### §2.3 The charge → energy arithmetic

Every step of `kine_charge_from_maps` (`NeutrinoEnergyReco.cxx:48-189`)
reproduces both prototype `cal_kine_charge` bodies:

- **min / med / max plane selection** (`:150-162` vs prototype `:203-225` and
  `:407-429`), including the degenerate `min_index == max_index` fallback to
  `(0,1,2)`.
- **the asymmetry switch** (`:181-186` vs `:236-242` / `:440-446`): if
  `max_asy > 0.04` the three-plane weighted mean is replaced by the
  (median, minimum) pair. Same threshold, same weights `{0.25, 0.25, 1.0}`,
  same pair.
- **the conversion** `overall / recom / fudge * 23.6 / 1e6 * MeV` (`:188` vs
  `:248` / `:450`), with the literals lifted into `KineChargeOptions` whose
  defaults are exactly the prototype's.
- **the recombination/fudge ladder** — shower → `(0.5, 0.8)`, else
  `|pdg| == 2212` → `recom = 0.35` with the fudge deliberately left at 0.95,
  else `(0.7, 0.95)` — at all three sites (`:203-209`, `:254-260`, `:311-317`)
  vs prototype `:85-90`, `:313-318`.
- **the `0.6 cm` 2-D association cut** and the pcloud1 → pcloud2 fallback
  ladder, including the `pcloud1 == 0 && pcloud2 == 0 → return 0` and the
  cross-aliasing of a missing cloud (`:213-215` vs `:103-105` / `:327-329`).
- **the `associate_points` / `fit` pair** as pcloud1 / pcloud2.

### §2.4 The dE/dx clamp algebra

This looks like a divergence and is not. The prototype clamps the **rate**:

```cpp
if (dEdx < 0) dEdx = 0;
if (dEdx > 50*units::MeV/units::cm) dEdx = 50*units::MeV/units::cm;
kine_energy += dEdx * vec_dx.at(i);
```

The toolkit clamps the **energy** (`PRSegmentFunctions.cxx:1264-1268`):

```cpp
if (dE < 0) dE = 0;
if (dE > 50 * units::MeV / units::cm * dX) dE = 50 * units::MeV / units::cm * dX;
kine_energy += dE;
```

`clamp(r, 0, C) · dx ≡ clamp(r·dx, 0, C·dx)` for `dx > 0`, and `dx > 0` is
enforced (`:1228`, `:1253`). Identical.

Similarly the runaway filter: the prototype zeroes `dQdx` when
`dQdx/43e3 > 1000`, and `(exp(0 · β/…) − 1)/(β/…) = 0`; the toolkit zeroes
`dQ` (`:1256`) and any recombination model maps `dQ = 0 → dE = 0`. Same
outcome, and the toolkit is careful to use the **true** fitted `dx` for the
ratio while accumulating over the possibly-shortened `dX` — the comment at
`:1235-1238` explains exactly why, and it is right.

### §2.5 The first/last-point `dx` shortening

Prototype `ProtoSegment::cal_kine_dQdx()` (`:1358-1373`) shortens the path at
`i == 0` and `i == size()-1` when the fitted `dx` exceeds `1.5 ×` the distance
to the neighbouring fit point. Toolkit `:1240-1252` does the same, with the
same `1.5` factor and the same neighbour. The two-argument overloads on both
sides omit the shortening (`ProtoSegment.cxx:1316-1339` vs
`PRSegmentFunctions.cxx:1274-1305`) — also matching, and it matters: the
shower path goes through the two-argument form, the single-segment path
through the shortening one, on both sides.

### §2.6 `cal_kine_range`'s particle table

`{11 → electron, 13 → muon, 211 → pion, 321 → kaon, 2212 → proton}` on
`abs(pdg)`, evaluated at `L/cm`, result in MeV. Identical
(`PRSegmentFunctions.cxx:1391-1413` vs `ProtoSegment.cxx:1408-1418`). The
muon fallback is a divergence — see P12.

### §2.7 `fill_kine_tree`'s skeleton

Match, in order:

- SCE-corrected vertex first (modulo P5), then `kine_reco_Enu`/
  `kine_reco_add_energy` zeroed, `ave_binding_energy = 8.6 MeV`.
- `used_vertices` / `used_segments` pre-seeded from **every** shower via
  `fill_sets(…, /*flag_exclude_start_segment=*/false)` — the `false` matters
  and is right on both sides (`NeutrinoKinematics.cxx:89` vs `kine.h:26`).
- `map_sg_shower` keyed on the shower's **start segment**.
- first pass over the main vertex's segments, shower branch vs track branch.
- BFS with `segments_to_be_examined` → `temp_segments`, `used_vertices` guard
  at the top, `used_segments` guard in the track branch.
- `flag_reduce` — set when a neighbour shares the previous segment's PDG or
  the 211↔13 pair, applied **once per visited vertex** after the neighbour
  loop, subtracting either the binding energy (proton) or the rest mass
  (non-electron). Same condition, same placement (`:214-217`, `:245-253` vs
  `kine.h:121-124`, `:194-200`).
- the per-particle `kine_energy_info` ladder: `2` if `|best − charge| <
  0.001·best`, else `1` if `|best − range| < 0.001·best`, else `0` — at all
  four sites.
- the rest-mass rule: proton → `+8.6 MeV`, else non-electron → `+mass`.
- the remaining-shower loop, its `vtx_type > 3` skip, its
  `included = (vtx_type != 3 ? 1 : vtx_type)` encoding, and its
  proton-with-length > 5 cm binding-energy add.
- `kine_reco_Enu = Σ kine_energy_particle + kine_reco_add_energy`.
- the entire π⁰ block, including the `/π × 180` angle conversions and the
  `/cm`, `/MeV` scalings.

### §2.8 `calculate_kinematics`'s branch structure

`nseg == 1` / (`nsegments == nconnected_segs`) / else, with the energy triple
`(range, dQdx, best)` computed differently in each — and in the multi-track
branch `range = 0`, `best = 0`, `dQdx` from the concatenated vectors. Match
(`PRShower.cxx:931-1159` vs `WCShower.cxx:339-526`). The
`start_connection_type == 1 ? (len < 4 cm ? dQdx : range) : (flag_shower ? 0 :
(len < 4 cm ? dQdx : range))` ladder is reproduced at both the single-segment
and multi-segment single-track sites, and the `len > 8 cm` switch on
`init_dir` in the multi-segment branch is there too (`:1096-1100` vs
`:421-425`).

### §2.9 `calculate_shower_kinematics`'s dispatch

`|particle_type| != 13 → calculate_kinematics`, else
`calculate_kinematics_long_muon`, then `cal_kine_charge`, then
`set_kine_charge` + `set_flag_kinematics(true)`, all under a
`!get_flag_kinematics()` guard. Match (`NeutrinoEnergyReco.cxx:302-346` vs
`shower_clustering.h:1407-1425`). The toolkit inlines
`kine_charge_from_maps` rather than calling `cal_kine_charge(shower)` — a
refactor, not a behaviour change, since the inlined call passes the same
arguments the wrapper would.

### §2.10 `ChargeMap` is ordered

`using ChargeMap = std::map<TrackFitting::CoordReadout,
TrackFitting::ChargeMeasurement>` (`NeutrinoPatternBase.h:22`), with
`CoordReadout::operator<` at `TrackFitting.h:321`. The per-plane float
accumulation at `NeutrinoEnergyReco.cxx:130` therefore runs in a stable order,
matching the prototype's `std::map<std::pair<int,int>, …>`. `collect_2D_charge`
builds these ordered maps out of an `unordered_map` (`TrackFitting.cxx:5xx`
region, `m_charge_data`), so the unordered container never leaks into a sum.
**This is the one place the obvious determinism worry does not apply** — P3 is
a different container.

---

## §3 Divergences

### P1 — the shower PDG is read from a cache whose refresh path is incomplete

**class: port defect. severity: medium.**

`fill_kine_tree` needs "what particle is this shower". The prototype asks the
**start segment**, live, every time:

```cpp
// NeutrinoID_kine.h:53, :175, :224
ktree.kine_particle_type.push_back(shower->get_start_segment()->get_particle_type());
// :67, :187 -- the rest-mass test
if (shower->get_start_segment()->get_particle_type()!=11)
  ktree.kine_reco_add_energy += shower->get_start_segment()->get_particle_mass()/units::MeV;
// :242 -- the remaining-shower proton test
}else if (shower->get_start_segment()->get_particle_type()==2212 && ...
```

The toolkit asks the **shower**, at all four sites:

```cpp
// NeutrinoKinematics.cxx:109
ktree.kine_particle_type.push_back(shower->get_particle_type());
// :119
if (shower->get_particle_type() != 11) { ... start_sg->particle_info()->mass() ... }
// :274
ktree.kine_particle_type.push_back(shower->get_particle_type());
// :286
if (shower->get_particle_type() == 2212) { ... }
```

and the translation note at `:31` states the substitution as if it were an
identity:

```
//   shower->get_start_segment()->get_particle_type() -> shower->get_particle_type()
```

It is not one. `Shower::get_particle_type()` returns `data.particle_type`
(`PRShower.h:139`), a **cached** field initialised to 0 (`PRShower.cxx:48`) and
written in exactly three places:

| site | context | refreshes the cache? |
|---|---|---|
| `NeutrinoShowerClustering.cxx:118` | long-muon tagging, sub-call 1 | yes, once |
| `PRShower.cxx:939` | `calculate_kinematics`, single-segment branch | yes |
| `PRShower.cxx:1050` | `calculate_kinematics`, multi-segment branch | yes |

**The refresh discipline mostly holds — that is the honest finding.** Three
sites rewrite a start segment's PDG after the last
`calculate_shower_kinematics` (`NeutrinoShowerClustering.cxx:3272`), and two of
them immediately call `calculate_kinematics` directly, which bypasses the
`flag_kinematics` latch (the latch lives in `calculate_shower_kinematics`
`:303`, not in `calculate_kinematics`) and so does refresh the cache:

| site | sub-call | writes | followed by |
|---|---|---|---|
| `NeutrinoShowerClustering.cxx:1921` | 10 (`examine_shower_1`) | `set_pdg(11)` | `calculate_kinematics` at `:1934` — **refreshed** |
| `NeutrinoShowerClustering.cxx:2372` | 10 (`examine_showers`) | `set_pdg(11)` | `calculate_kinematics` at `:2374` — **refreshed** |
| `NeutrinoShowerClustering.cxx:2768` | 11 (`id_pi0_with_vertex`) | `set_pdg(211)` | nothing — **not refreshed** |

So the divergence is not "stale by construction". What remains is three
narrower exposures:

1. **`:2768`.** The π⁰-vertex reclassification walks `map_vertex_segments[pi0_vtx]`
   and turns an *incoming* segment with `|pdg| == 13` or `pdg == 0` into a
   pion, with no kinematics recompute. A π⁰ shower's own start segment is
   outgoing and already `11`, so it is excluded by the `is_incoming` guard
   (`:2764`) — but whether some *other* shower's start segment can be an
   incoming segment at a π⁰ vertex was **not established**. If it can, the
   toolkit publishes `kine_particle_type = 11` and skips the `!= 11` rest-mass
   add where the prototype publishes `211` and adds 139.6 MeV.
2. **The `has_particle_info()` gate.** Both cache writes sit inside
   `if (m_start_segment->has_particle_info())` (`PRShower.cxx:938`, `:1049`).
   A shower whose start segment had no `ParticleInfo` when its kinematics were
   computed keeps `data.particle_type == 0` even after the segment gains one.
   `kine_particle_type` is then published as **0**, and the `!= 11` test fires
   and adds the segment's live `mass()` where the prototype would consult the
   live PDG.
3. **The long-muon path never writes the cache at all** — see P8. Such a
   shower's `data.particle_type` comes only from
   `NeutrinoShowerClustering.cxx:118`, taken at sub-call 1.

**What is not claimed.** No concrete event is demonstrated. This is the same
class as pr/33 P2 ("whose PDG", five sites in the shower-clustering stage), and
it is four more sites in the stage whose output feeds the BDT — but here the
refresh discipline currently covers two of the three post-latch rewrites, and
the remaining exposure is conditional. §7.1 gives the one-line check that
would settle it.

### P2 — `cal_corr_factor` is a stub that returns 1.0

**class: port gap (unimplemented). severity: high.**

```cpp
// NeutrinoEnergyReco.cxx:14-35
double PatternAlgorithms::cal_corr_factor(WireCell::Point& pt, TrackFitting& track_fitter, IDetectorVolumes::pointer dv){
    double corr_factor = 1.0;
    // So far this is an empty class that needs to be filled with actual logic ...
    ...
    (void)apa; (void)face; (void)plane; (void)grouping;
    return corr_factor;
}
```

The comment is honest, and the function is called for **every** charge hit
that associates to a point, on both the shower and segment paths
(`:221`, `:276`, `:299`). The prototype's version
(`NeutrinoID_energy_reco.h:255-272`) does two distinct things:

```cpp
double factor = 1;
double central_U = offset_u + (slope_yu * p.y + slope_zu * p.z);
if (central_U >=296 && central_U <=327 || ... || central_U >=536 && central_U <=671)
  factor = factor/0.7;                                     // (a)
if (mp.get_flag_corr()){
  factor *= mp.get_corr_factor(p, offset_u, ..., slope_zw);  // (b)
}
return factor;
```

**(a)** is seven hard-coded uBooNE U-plane wire ranges given a `1/0.7 = 1.43×`
boost. That is a detector-specific dead/low-response-region correction and
**not** porting it to SBND is correct — reproducing it would be exactly the
M15 trap. Note it is applied *unconditionally*, outside the `flag_corr` guard.

**(b)** is the general position correction:
`gu->Eval(u) * gv->Eval(v) * gw->Eval(w)` over three `TGraph`s loaded from
calibration files (`data/src/TPCParams.cxx:571-592`). It is **live in the
prototype's production**, established four ways rather than one:

1. `wire-cell-prod-nue.cxx:32` declares `int flag_calib_corr = 1;` and
   `:192-193` reads `if (flag_calib_corr==1) mp.init_corr_files();`.
   `init_corr_files` ends with `flag_corr = true;` (`TPCParams.cxx:592`).
2. The flag is overridable — `case 'q':` at `:53-54` — so the default alone proves
   nothing. **The production invocations do not pass `-q`.** `run_5384.pl:34`
   and `:41` invoke
   `wire-cell-prod-nue-port … $filename 0 -d0 -o1 -gfind_other_segments`.
   Only `-d`, `-o` and `-g` appear.
3. Those scripts call the **`-port`** variant, not the one cited above. It
   carries the identical default and guard
   (`wire-cell-prod-nue-port.cxx:32`, `:196`).
4. `init_corr_files()` is called with no arguments, so the paths are the
   header defaults (`data/inc/WCPData/TPCParams.h:153`:
   `input_data_files/calib_{u,v,w}_corr.txt`, 2401/2401/3457 points). **All
   three files exist in this checkout** (30 KB / 30 KB / 45 KB, Oct 2024), so
   the `TGraph`s are populated and `Eval` returns real factors rather than a
   degenerate 1.

So every prototype `kine_charge` carries a per-point, per-wire calibration
factor; every toolkit `kine_charge` carries 1.0.

**Consequences.** `kine_charge` is the charge-based energy estimate. It is
(i) `shower->get_kine_charge()`, hence the `kine_best` fallback when
`kenergy_best == 0` (`PRShower.h:152-153`) — that is *every* shower with
`start_connection_type != 1` that is flagged shower-like, i.e. most real EM
showers; (ii) the `kine_energy_info == 2` ("charge") classifier at all four
`fill_kine_tree` sites; (iii) an input to the shower-merging decisions in
pr/33's stage. A multiplicative bias on it moves reconstructed shower energy
directly.

**What this audit does not claim**: that SBND needs correction (b) at all, or
what its magnitude would be. The uBooNE `TGraph`s are not SBND's. The finding
is that the prototype applies a calibration here and the toolkit applies
none, with no knob and no record — §7.2.

---

### P3 — `calculate_kinematics` accumulates over a pointer-hashed unordered set

**class: determinism regression, toolkit-only. severity: high (M4).**

```cpp
// PRShower.cxx:1131-1143
double total_length = 0;
std::vector<double> vec_dQ, vec_dx;
for (auto edesc : this->edges()) {
    SegmentPtr seg = view[edesc].segment;
    if (!seg) continue;
    total_length += segment_track_length(seg);
    for (const auto& fit : seg->fits()) { vec_dQ.push_back(fit.dQ); vec_dx.push_back(fit.dx); }
}
data.kenergy_dQdx = cal_kine_dQdx(vec_dQ, vec_dx, recomb_model);
```

`TrajectoryView::edges()` returns `const edge_unordered_set&`
(`PRTrajectoryView.h:113`), i.e.
`std::unordered_set<edge_descriptor, EdgeDescriptorHash, EdgeDescriptorEqual>`
(`PRGraphType.h:170`). `EdgeDescriptorHash` (`:123-143`) hashes
`boost::source(ed,g)` and `boost::target(ed,g)` — and the graph is declared
with `boost::setS` for **both** vertices and edges (`:91-93`), so a
`node_descriptor` is a `void*`. The bucket layout, and therefore the
iteration order, is a function of heap addresses.

The toolkit says so itself, in the same header (`:175-181`):

> WARNING: this order is based on pointer values and therefore VARIES BETWEEN
> RUNS of an identical program. … For anything that accumulates, pushes into
> an output vector, breaks on first match, or takes a tie-broken min/max, use
> ordered_nodes().

This loop does all three of the first: it accumulates `total_length`, and it
pushes into `vec_dQ`/`vec_dx` whose order then determines the summation order
inside `cal_kine_dQdx`. Both are float sums; both are order-dependent in the
low bits.

**It propagates**: `kenergy_dQdx` → `kenergy_best` (whenever the
`< 4 cm` branch is taken) → `kine_energy_particle` → `kine_reco_Enu` → BDT
variable 69. `total_length` → `kenergy_range` → the same chain.

**The fix already exists and is used 75 lines away.**
`calculate_kinematics_long_muon` iterates `ordered_edges(*this, m_full_graph)`
(`:1209`), with a comment explaining the determinism intent, and the
end-point searches in `calculate_kinematics` use `ordered_nodes` (`:1014`,
`:1121`) for exactly this reason. Only the dQ/dx collection was left raw.

**Reconciling with the in-tree determinism results.** Two prior efforts
looked for exactly this class of bug and are worth not tripping over:

- **doc 60 §7** (`project_sbnd_pr_chain_aslr_nondeterminism`) remeasured doc
  49 §4a's ±7-STM-tag ASLR claim and found the chain deterministic with and
  without `setarch x86_64 -R`, over 431 + 20 + 60 + 60 event pairs. **That
  measurement's scope is `switch_scope → steiner → fiducialutils → TGM → STM
  → FC`** — the cosmic-tagger side. `shower_clustering_with_nv`,
  `calculate_kinematics` and `fill_kine_tree` are downstream of it and were
  not covered. The null result therefore does not contradict P3; it does not
  reach it.
- **`c05bc5f7`** (doc pr/28 §11) swept `boost::edges` / `vertices` /
  `graph_nodes` for `T_tagger` determinism and closed ten unstable branches in
  `NeutrinoTaggerNuE.cxx` and `NeutrinoTaggerSinglePhoton.cxx`. Its own commit
  message records that the residual was *not* a float accumulation — that
  sweep was looking for output-vector **ordering**, in the tagger files.
  `PRShower.cxx:1134` is a float accumulation in a different file and was
  outside its scope.

**Still not measured**: whether the drift is large enough to move a
`kine_energy_info` classification or a BDT score. The M4 protocol applied to
*this* stage's products — `kine_energy_particle`, `kine_reco_Enu` — would
settle it. §7.3.

The prototype has the mirror-image problem — it iterates
`map_seg_vtxs` (a `std::map<ProtoSegment*, …>`) at `WCShower.cxx:448` and
`:515` — so neither side is deterministic here. The point is that the toolkit
has the tool, uses it elsewhere in the same function, and missed this loop.

---

### P4 — the segment path re-collects the entire 2-D charge map on every call

**class: efficiency. severity: medium.**

The prototype collects once, into members, via `collect_2D_charges()`
(`NeutrinoID_energy_reco.h:1-41`), and both `cal_kine_charge` overloads read
`charge_2d_u/v/w` directly. The toolkit's shower overload does the same
(`NeutrinoEnergyReco.cxx:242`: `if (m_charge_2d_u.empty()) collect_charge_maps(...)`).
The **segment** overload does not:

```cpp
// NeutrinoEnergyReco.cxx:262-264
ChargeMap charge_2d_u, charge_2d_v, charge_2d_w;
WireMap   map_apa_ch_plane_wires;
track_fitter.collect_2D_charge(charge_2d_u, charge_2d_v, charge_2d_w, map_apa_ch_plane_wires);
```

— fresh locals, every call. `collect_2D_charge` walks the whole event's
`m_charge_data` and does a `get_wires_for_channel` geometry lookup per unique
channel. `fill_kine_tree` calls `cal_kine_charge(seg, …)` once per **track
segment** (`NeutrinoKinematics.cxx:142`, reached from `:187` and `:225`), so
the cost is `O(n_track_segments × n_charge_hits)` where the prototype pays
`O(n_charge_hits)` once.

This is CLAUDE.md §4's explicit "efficiency: no per-point allocations in hot
loops that the prototype hoisted" criterion.

**The provenance question, checked.** The shower path reads maps cached at
shower-clustering time; the segment path collects at `fill_kine_tree` time. If
`TrackFitting`'s charge data changed in between, showers and tracks in one
event would get energies from different inputs. It does change —
`update_dQ_dx_data` (`TrackFitting.cxx:5360`, the write at `:5388`) writes
`measurement.charge_err = m_params.share_charge_err` for shared wires and
saves the originals to `m_orig_charge_data` (`:5387`), restored by
`recover_original_charge_data` (`:5393-5398`). But the mutated field is
`charge_err`, and `kine_charge_from_maps` reads only `charge_data.charge`
(`:130`). **So the two collections agree on everything this stage uses.** The
finding is cost, not correctness.

---

### P5 — the neutrino vertex is not SCE-corrected on SBND

**class: port gap / config. severity: medium.**

Prototype (`NeutrinoID_kine.h:3-9`):

```cpp
Point nu_vtx = main_vertex->get_fit_pt();
Point corr_nu_vtx = mp.func_pos_SCE_correction(nu_vtx);
ktree.kine_nu_x_corr = corr_nu_vtx.x/units::cm;   // y, z likewise
```

— unconditional; `mp.init_Pos_Efield_SCE_correction()` is called
unconditionally at `wire-cell-prod-nue.cxx:197`.

Toolkit (`NeutrinoKinematics.cxx:61-75`):

```cpp
if (geom_helper) { ... geom_helper->get_corrected_point(nu_vtx, IClusGeomHelper::SCE, apa, face) ... }
else {
    // TODO: SCE correction requires a valid geom_helper; using raw vertex position for now.
    ktree.kine_nu_x_corr = static_cast<float>(nu_vtx.x() / units::cm);   // y, z likewise
}
```

`m_geom_helper` comes from the `clus_geom_helper` configuration key, which
defaults to the empty string (`TaggerCheckNeutrino.cxx:266`, "empty = no SCE
vertex correction") and which the SBND config never sets — `grep
clus_geom_helper cfg/pgrapher/experiment/sbnd/clus.jsonnet` returns nothing.
The call site passes it with a comment saying so
(`TaggerCheckNeutrino.cxx:864`: "nullptr when clus_geom_helper is not
configured").

So on SBND `kine_nu_{x,y,z}_corr` are **the raw fitted vertex**, silently
carrying the name `_corr`. The fields are published in `T_kine` and dumped by
`PrDisplayDump.cxx`. Whether SBND wants an SCE correction at this point at all
is a physics question the owner should answer; the finding is that the field
name asserts a correction that is not applied and nothing at runtime says so.

---

### P6 — the BFS shower branch gained a double-count guard

**class: prototype bug not reproduced. severity: medium.**

Toolkit (`NeutrinoKinematics.cxx:231-239`):

```cpp
else {
    const ShowerPtr& shower = it2->second;
    if (!used_showers.count(shower)) {
        push_shower_kine(shower);
        ktree.kine_energy_included.push_back(1);
        used_showers.insert(shower);
    }
}
```

Prototype (`NeutrinoID_kine.h:165-190`) has **no** membership test: it pushes
the shower's energy into `kine_energy_particle`, adds its rest mass to
`kine_reco_add_energy`, and inserts into `used_showers` — every time it is
reached. Reaching one shower twice therefore double-counts its energy in
`kine_reco_Enu`.

**Precondition.** A shower's start segment is pre-seeded into `used_segments`
by `fill_sets(…, false)`, so the BFS never *traverses* it; the shower is
reached by arriving at a vertex that the start segment touches. Being reached
twice means both of the start segment's endpoints get visited by separate
track chains — i.e. a cycle in the track graph, or the shower's start segment
touching two distinct visited vertices. Frequency unmeasured.

The toolkit's behaviour is the defensible one. It is recorded here because it
means toolkit and prototype `kine_reco_Enu` **cannot** be expected to agree on
such events, and anyone diffing the two should not chase it.

---

### P7 — the published per-particle arrays are ordered differently

**class: determinism improvement (toolkit), output-order divergence. severity: medium.**

The prototype's first pass iterates `map_vertex_segments[main_vertex]`, a
`std::set<ProtoSegment*>` (`NeutrinoID_kine.h:38`), and its BFS neighbour loop
iterates `map_vertex_segments[curr_vtx]` (`:117`) — both pointer-ordered. The
toolkit uses `sorted_out_edges(…)` at both (`NeutrinoKinematics.cxx:171`,
`:205`), i.e. ordered by `EdgeBundle::index`.

`kine_energy_particle`, `kine_particle_type`, `kine_energy_info` and
`kine_energy_included` are **parallel arrays published in traversal order**, so
their element order differs from the prototype's — and the prototype's differs
from itself run to run. `kine_reco_Enu` is a float sum over
`kine_energy_particle` (`:301-304`), so the prototype's total is
order-dependent in the low bits and the toolkit's is not.

The toolkit is strictly better and reproducing the prototype's order would be
an M4 regression. Recorded so that an array-by-array comparison against
prototype output is not mistaken for a defect. Contrast P3, where the toolkit
did *not* apply the same discipline.

---

### P8 — `calculate_kinematics_long_muon` never writes `data.particle_type`

**class: port defect. severity: low.**

Prototype (`WCShower.cxx:289`) opens with

```cpp
particle_type = start_segment->get_particle_type();
flag_shower = false;
```

Toolkit (`PRShower.cxx:1194-1198`) takes a **local**:

```cpp
int particle_type = abs(m_start_segment->particle_info()->pdg());
unset_flags(ShowerFlags::kKinematics);
unset_flags(ShowerFlags::kShower);   // matches the prototype's flag_shower = false
```

`data.particle_type` is left at whatever it was. The in-code comment at
`:1187-1192` argues this is safe because the function is only reached when
`shower->get_particle_type() == 13`, which requires the earlier
`set_particle_type` at `NeutrinoShowerClustering.cxx:118`. That holds for
`+13`; for a start segment with pdg `−13` the prototype stores `−13` and the
toolkit stores whatever `:118` put there (which is `curr_sg->particle_info()->pdg()`,
so also `−13` — consistent). The residual exposure is the general one from
P1: the field is a cache and this path does not refresh it. The local `abs()`
is correct for the range lookup (`:1230`), which uses `abs` anyway.

---

### P9 — long-muon `start_point` when the direction sign is zero

**class: port defect. severity: low.**

Prototype (`WCShower.cxx:308-312`):

```cpp
if (start_segment->get_flag_dir()==1){
  start_point = start_segment->get_point_vec().front();
}else if (start_segment->get_flag_dir()==-1){
  start_point = start_segment->get_point_vec().back();
}
```

— `flag_dir == 0` leaves `start_point` **unchanged** (at its previous value,
the default-constructed origin for a fresh shower).

Toolkit (`PRShower.cxx:1241-1246`):

```cpp
int dirsign_val = m_start_segment->dirsign();
if (dirsign_val == 1) { data.start_point = fits.front().point; }
else                  { data.start_point = fits.back().point; }
```

— `dirsign == 0` takes `back()`.

`start_point` then seeds the farthest-vertex search (`:1250-1258`) and hence
`end_point`, and is the shower's start in the display. Note the toolkit's
choice is arguably better than inheriting the origin; the finding is that they
differ and neither is documented. Both branches of `calculate_kinematics`
reproduce the prototype's two-way test faithfully (`:978-984`, `:1071-1075`) —
it is only the long-muon path that collapses it.

---

### P10 — `segment_cal_kine_dQdx` skips fit points the prototype includes

**class: guard added, precondition unmeasured. severity: low.**

```cpp
// PRSegmentFunctions.cxx:1228
if (!fits[i].valid() || fits[i].dx <= 0) continue;
```

`Fit::valid()` is `index >= 0` (`PRCommon.h:145-148`). The prototype has no
such filter; it divides by `dx_vec.at(i) + 1e-9` and lets the runaway filter
handle degenerate points.

For `dx == 0` the two agree: the prototype gets `dQdx ≈ dQ·1e9`, which trips
`dQdx/43e3 > 1000`, zeroes it, and contributes 0. For `index < 0` with a
non-zero `dx` and a real `dQ`, the prototype contributes energy and the
toolkit does not. Whether such fits exist — whether a `Fit` can carry a valid
`dQ`/`dx` pair with `index < 0` — was not established. §7.4.

---

### P11 — the one-fit-point segment: prototype throws, toolkit guards

**class: prototype bug not reproduced. severity: low.**

Prototype `cal_kine_dQdx()` (`ProtoSegment.cxx:1358-1361`):

```cpp
if (i==0){
  double dis = sqrt(pow(fit_pt_vec.at(1).x - fit_pt_vec.at(0).x,2) + ...);
```

For a segment with a single fit point, `i == 0` is true, the `i+1 ==
size()` branch is never reached, and `fit_pt_vec.at(1)` throws
`std::out_of_range`. The toolkit guards both branches with `fits.size() > 1`
(`:1240`, `:1246`) and falls through to the unshortened `dX`.

Related to `project_trackfitting_single_point_abort` (toolkit `2a821fd2`),
which fixed a different single-point abort in `TrackFitting`. Recorded so the
divergence is not "corrected" back.

---

### P12 — `cal_kine_range`'s missing-curve behaviour

**class: prototype bug not reproduced. severity: low.**

The prototype has three overloads. Two of them
(`ProtoSegment.cxx:1393`, `:1408`) initialise `TGraph *g_range =
mp.get_muon_r2ke();` before the ladder, so an unrecognised PDG falls back to
the muon curve. The third, `cal_kine_range(double L)` (`:1380`), does **not**:

```cpp
TGraph *g_range = 0;
if (fabs(particle_type)==11) g_range = mp.get_electron_r2ke();
else if ... (no default)
double kine_energy = g_range->Eval(L/units::cm) * units::MeV;
```

— a null dereference for any `particle_type` outside
`{11, 13, 211, 321, 2212}`, including the very common `0`. That overload is
the one used by `WCShower::calculate_kinematics` for multi-segment showers
(`:454`) and by `calculate_kinematics_long_muon` (`:300`).

The toolkit's single `cal_kine_range(L, pdg, particle_data)` has an explicit
muon fallback and a null return (`PRSegmentFunctions.cxx:1407-1412`), so it
never dereferences null and never crashes. Where the prototype would have
crashed, the toolkit silently returns a muon-curve energy — a behaviour
difference only on inputs that would have aborted the prototype.

---

### P13 — `fill_kine_tree` moved from before the taggers to after them

**class: ordering divergence. severity: low.**

Prototype, in the `NeutrinoID` constructor (`NeutrinoID.cxx:251-255`):

```cpp
if (flag_tagger){
  fill_kine_tree(kine_info);
  bool flag_cosmic = cosmic_tagger();
  ...
```

— `kine_info` is filled **first**, then every tagger runs.

Toolkit, in `TaggerCheckNeutrino::visit()`: `init_tagger_info` at `:741`,
taggers at `:750`–`:844`, `match_isFC` at `:853`, and `fill_kine_tree` at `:861`. The `KineInfo` local
is not even declared until `:859`.

**Consequence, bounded.** The obvious worry — that the BDT loses
`kine_reco_Enu` — does not materialise: the prototype's BDT readers bind
`&kine_info.kine_reco_Enu` (`NeutrinoID_nue_bdts.h:29`,
`NeutrinoID_numu_bdts.h:80`) but evaluate later, and in the toolkit the
scorers are separate pipeline stages that read `tf->get_kine_info()`
(`root/src/UbooneNueBDTScorer.cxx:614`, `:1660`;
`root/src/UbooneNumuBDTScorer.cxx:267`, `:507`) after
`TaggerCheckNeutrino` has stored it. Both get a filled value.

The real exposure is forward-looking: **any toolkit tagger that reads
`KineInfo` would read zeros**, because it does not exist yet when the taggers
run. No current one does — `grep -n 'get_kine_info()' clus/src/*.cxx
root/src/*.cxx` returns five hits and none is a tagger:
`PrDisplayDump.cxx:452`, `:624`, `UbooneNumuBDTScorer.cxx:267`,
`UbooneNueBDTScorer.cxx:614`, `UbooneTaggerOutputVisitor.cxx:55`. Symmetrically,
a grep of the prototype found no tagger reading `kine_info.*` other than the two
BDT bindings. Recorded as a constraint on future ports, not a present defect.

---

### P14 — the `kine_best == 0` fallback was folded into the accessor

**class: cosmetic, with a forward hazard. severity: low.**

Prototype `WCShower::get_kine_best()` (`WCShower.h:50`) is a bare getter, and
`fill_kine_tree` applies the fallback at each of its three shower sites:

```cpp
float kine_best = shower->get_kine_best();
if (kine_best ==0 ) kine_best = shower->get_kine_charge();
```

Toolkit `Shower::get_kine_best()` (`PRShower.h:152-153`) folds it in:

```cpp
double get_kine_best(){
    if (data.kenergy_best != 0) return data.kenergy_best; else return data.kenergy_charge; };
```

so `fill_kine_tree` reads it bare (`NeutrinoKinematics.cxx:104`, `:269`).
**Equivalent at these sites.**

The hazard is that the fallback is now global. The prototype's other consumers
use the guarded idiom explicitly — e.g. `NeutrinoID_nue_functions.h:95-100`:

```cpp
if (shower->get_kine_best() != 0){ E_shower = shower->get_kine_best(); }
else                             { E_shower = shower->get_kine_charge(); }
```

which, checked, is also equivalent under the folded accessor (both arms yield
`kenergy_charge` when `kenergy_best == 0`). But `kenergy_best = 0` is set
*deliberately* for shower-like objects with `start_connection_type != 1`
(`WCShower.cxx:359`, toolkit `PRShower.cxx:960`) — it is a sentinel meaning
"no reliable best estimate". Any consumer that tests `get_kine_best() == 0` to
detect that condition is now unreachable in the toolkit. Whether such a
consumer exists among the ported taggers was not surveyed — §7.7.

---

## §4 The SBND operating point

This stage is one of the few whose calibration constants the SBND config
actually moves (doc pr/10).

| knob | C++ default (uBooNE) | SBND value | source |
|---|---|---|---|
| `kine_recom_factor` | 0.70 | **0.87** | `clus.jsonnet:1037` — 0.70 × 1.249 (track) |
| `kine_shower_recom_factor` | 0.50 | **0.58** | `:1039` — 0.50 × 1.169 (shower) |
| `kine_proton_recom_factor` | 0.35 | **0.51** | `:1040` — 0.35 × 1.453 (proton) |
| `kine_fudge_factor` | 0.95 | `null` (unchanged) | `:1036` |
| `kine_shower_fudge_factor` | 0.80 | `null` (unchanged) | `:1038` |
| `kine_plane_weights` | {0.25, 0.25, 1.0} | `null` (unchanged) | `:1041` |
| `kine_plane_asym_switch` | 0.04 | `null` (unchanged) | — |
| `kine_w_value` | 23.6 eV | `null` (unchanged) | `:1043` |
| `clus_geom_helper` | `""` | unset ⇒ `""` | P5 |

The comment block at `clus.jsonnet:1028-1035` is explicit that the fudge
factors, the plane weights, the asymmetry switch and the W-value "still have
NO SBND value" — they remain uBooNE literals. The three recombination factors
are the table-integrated ratio transfer from doc pr/10.

**Net for this audit**: unlike pr/33 §4, the SBND configuration does not widen
any divergence found here — but it does mean the toolkit's absolute energy
scale already differs from the prototype's by the recombination ratios *by
design*, so a numerical prototype↔toolkit energy comparison must divide those
out before it means anything. P2's missing position correction sits **on top**
of that, uncontrolled.

---

## §5 Looks like a divergence and is not

**§5.1 `min_asy` is dead in the prototype.** It is computed at
`NeutrinoID_energy_reco.h:228` and `:432` and used only inside a
commented-out branch (`:237-241`, `:441-445`). The toolkit computes only
`max_asy` (`:170-172`). Not a drop of live logic. Same shape as pr/30 §5.1
and pr/33 GOTCHA 9.

**§5.2 `kine_dQdx` is computed and discarded.** `fill_kine_tree` assigns
`float kine_dQdx = ...` at five prototype sites (`kine.h:49`, `:78`, `:140`,
`:171`, `:220`) and never reads it — the `kine_energy_info` ladder tests only `charge`
and `range`, with `dQdx` as the `else`. The toolkit simply does not compute
it. Correct. Note the prototype's `curr_sg->cal_kine_dQdx()` calls at `:78`,
`:140` are pure, so dropping them has no side effect.

**§5.3 The zero-denominator guards.** The toolkit guards `max_asy` behind
`sums[med] + sums[max] > 0` (`:171`) and both weighted means behind positive
weight sums (`:179`, `:183`). The prototype divides unguarded. With all-zero
charge the prototype gets `nan`, and `nan > 0.04` is **false**, so it takes
the three-plane branch and returns 0 — exactly what the toolkit's `max_asy = 0`
produces. With the default weights the weight guards are always true. No
behaviour change; the comments at `:174-177` say so and are right.

**§5.4 The dead `if (pcloud1 != 0)` inside the prototype's shower loop.**
`NeutrinoID_energy_reco.h:111`, `:140`, `:170` re-test a pointer that the
aliasing at `:103-105` has already made non-null. The toolkit's `if (pcloud1)`
at `:94` is the same dead test. Neither matters.

**§5.5 The recombination model substitution is pr/10's, not a finding here.**
The prototype hard-codes the Box inversion with `α = 1.0`, `β = 0.255`,
`W = 23.6e-6`, `ρ = 1.38`, `E = 0.273` (`ProtoSegment.cxx:1318-1319`,
`:1328`). The toolkit calls `recomb_model->dE(dQ, dx)`. Per CLAUDE.md §5
rule 4 / M15, a divergence a doc already owns is not re-litigated: see
`project_energy_calibration_round` (PowerBoxRecombination + the muon curve)
and `project_kine_energy_config_knobs`. **Not** re-derived in this round.

**§5.6 The `charge_err` mutation is invisible here.** See P4's second half —
`update_dQ_dx_data` mutates `charge_err`, `kine_charge_from_maps` reads only
`.charge`.

**§5.7 The remaining-shower loop does not add a rest mass.** The toolkit's
loop (`NeutrinoKinematics.cxx:263-296`) duplicates `push_shower_kine`'s body
instead of calling it, and in doing so omits the non-electron rest-mass add
that `push_shower_kine` performs at `:119-125`. That is **correct**: the
prototype's corresponding loop has the add commented out
(`NeutrinoID_kine.h:240-241`) and keeps only the proton-with-length case. The
duplication mirrors the prototype's own.

**§5.8 `used_vertices.insert(main_vertex)` after the first pass.** Both sides
insert the main vertex *after* the first-pass loop, not before
(`NeutrinoKinematics.cxx:190` vs `kine.h:102`). Since the first pass does not
consult `used_vertices`, the placement is inert — but it is the same on both
sides, so it is not a divergence.

**§5.9 `nsegments == nconnected_segs`.** The toolkit's
`this->edges().size()` vs `count_connected_segments(m_start_segment)`
(`PRShower.cxx:932`, `:1046`) is the prototype's `map_seg_vtxs.size()` vs
`get_connected_pieces(start_segment).first.size()` (`WCShower.cxx:399-404`).
Same predicate: "is the whole shower one connected chain from the start
segment".

**§5.10 The `kine_energy_included` type-3 encoding.** Prototype
`if (pair_vertex.second != 3) push(1); else push(pair_vertex.second);`
(`kine.h:233-237`) ≡ toolkit `push(vtx_type != 3 ? 1 : vtx_type)` (`:283`).
Both push the literal `3` for a type-3 start vertex.

**§5.11 `flag_reduce`'s null-guard.** The toolkit wraps the reduce block in
`if (flag_reduce && prev_sg->particle_info())` (`:245`). Without
`particle_info()` the prototype's `get_particle_type()` returns 0 and
`get_particle_mass()` returns 0, so it subtracts 0 — same net effect as
skipping. Equivalent.

**§5.12 `init_tagger_info`'s three renamed fields.** See §2.1: the names
differ, the defaults do not, and the prototype's own struct is the one that
carries `_shower_`/`_acc_` in the name.

---

## §6 Determinism

| site | prototype | toolkit | verdict |
|---|---|---|---|
| `fill_kine_tree` first pass + BFS neighbours | `std::set<ProtoSegment*>` — pointer order | `sorted_out_edges` — index order | **toolkit better** (P7) |
| `fill_kine_tree` remaining showers | `std::vector<WCShower*>` insertion order | `IndexedShowerSet` — index order | toolkit better |
| `calculate_kinematics` dQ/dx collection | `std::map<ProtoSegment*,…>` — pointer order | `this->edges()` — **pointer-hashed unordered_set** | **both broken** (P3) |
| `calculate_kinematics` end-point search | `std::map<ProtoVertex*,…>`, strict `>` | `ordered_nodes` | toolkit better |
| `calculate_kinematics_long_muon` | `std::map<ProtoSegment*,…>` | `ordered_edges` + index-keyed vertex map | toolkit better |
| `calculate_shower_kinematics` loop | `std::vector<WCShower*>` | `IndexedShowerSet` | equivalent |
| `kine_charge_from_maps` charge sums | `std::map<pair<int,int>,…>` | `std::map<CoordReadout,…>` | equivalent (§2.10) |

**The single actionable row is P3.** Everywhere else the toolkit either
already applies the ordered helpers or inherits an order that is stable by
construction. `used_showers`/`used_segments`/`used_vertices` are all
`Indexed*Set` in the toolkit (`PRShower.h:221`, `PRSegment.h`, `PRVertex.h`),
so no pointer-keyed container is *iterated* in `fill_kine_tree`.

Note also that the toolkit's determinism win at P7 makes the published
`kine_*` arrays reproducible run-to-run, which the prototype's are not — a
prerequisite for any A/B gate on this stage.

---

## §7 Loose ends

1. **P1 frequency — the whole finding turns on this.** Count
   `shower->get_particle_type() != shower->start_segment()->particle_info()->pdg()`
   at `NeutrinoKinematics.cxx:104` over the 572-event valfast manifest. Zero
   hits demotes P1 to a §5 entry; a non-zero count gives the exposure and, for
   the `11` vs `211` cases, bounds the `kine_reco_Enu` error at 139.6 MeV
   each. Log the pair, not just the count — case (2) of P1 (cache stuck at 0)
   and case (1) (π⁰ reclassification) need distinguishing.
2. **P2 scope.** Does SBND need a position-dependent charge correction here at
   all? If yes, is there an existing SBND calibration product it could read?
   The stub's signature already receives `IDetectorVolumes` and the grouping,
   so the plumbing exists.
3. **P3 magnitude.** Two `setarch x86_64 -R` runs of the same event, diffing
   `kine_energy_particle` and `kine_reco_Enu`, would show whether the drift is
   ULP-level or classification-changing. This is cheap and should probably be
   done before anything else in this list.
4. **P10 precondition.** Can a `Fit` carry a real `dQ`/`dx` with `index < 0`?
   If not, P10 collapses to a §5 entry.
5. **`Shower::calculate_kinematics`'s geometric helpers.**
   `shower_get_closest_point` and `shower_cal_dir_3vector` were not opened;
   the toolkit's `start_point` fallback ladder (`:1004-1009`, `:1080-1090`) has
   no prototype counterpart and the two copies of it in the same function are
   themselves inconsistent — one tests `sgcp_pt == (0,0,0)` as a sentinel and
   the other explicitly refuses to (`:1085-1086`, citing "B16.1 in review").
   That inconsistency is worth its own look.
6. **`PRShower.cxx` remains largely unread** (pr/33 GOTCHA 13). This round
   added `calculate_kinematics`, `calculate_kinematics_long_muon` and
   `get_kine_best`; the other ~1000 lines are still unaudited.
7. **P14's hazard.** Survey the ported taggers for a `get_kine_best() == 0`
   test that was meant to detect the "no reliable estimate" sentinel.
8. **`ProtoSegment.cxx`'s one non-comment change.** The prototype checkout adds
   `particle_score = 0;` at `do_track_pid`'s failure exit
   (`ProtoSegment.cxx:1286` in the current file). That is pr/31's area, not
   this stage's — flagged so the next round does not treat it as upstream.

---

## §8 Summary

| # | finding | class | sev | prototype | toolkit |
|---|---|---|---|---|---|
| P1 | shower PDG read from a cache whose refresh path is incomplete (4 sites) | port defect | med | `kine.h:53 :67 :175 :187 :224 :242` | `NeutrinoKinematics.cxx:109 :119 :274 :286` |
| P2 | `cal_corr_factor` returns 1.0; prototype's correction is live in production | port gap | high | `energy_reco.h:255-272`; `prod-nue.cxx:32 :192` | `NeutrinoEnergyReco.cxx:14-35` |
| P3 | dQ/dx accumulated over a pointer-hashed unordered set | determinism regression | high | — | `PRShower.cxx:1134` (cf. `:1209`) |
| P4 | segment path re-collects the whole 2-D charge map per call | efficiency | med | `energy_reco.h:1-41` (once) | `NeutrinoEnergyReco.cxx:264` |
| P5 | neutrino vertex not SCE-corrected on SBND | port gap / config | med | `kine.h:6` | `NeutrinoKinematics.cxx:70-75`; cfg `:266` |
| P6 | shower double-count guard added in the BFS | proto bug not reproduced | med | `kine.h:165-190` | `NeutrinoKinematics.cxx:234` |
| P7 | published per-particle arrays ordered by edge index, not pointer | determinism improvement | med | `kine.h:38 :117` | `NeutrinoKinematics.cxx:171 :205` |
| P8 | long-muon path never writes `data.particle_type` | port defect | low | `WCShower.cxx:289` | `PRShower.cxx:1194` |
| P9 | long-muon `start_point` takes `back()` when `dirsign()==0` | port defect | low | `WCShower.cxx:308-312` | `PRShower.cxx:1241-1246` |
| P10 | `!valid() \|\| dx<=0` fit points skipped | guard, unmeasured | low | `ProtoSegment.cxx:1345` | `PRSegmentFunctions.cxx:1228` |
| P11 | one-fit-point segment throws in the prototype | proto bug not reproduced | low | `ProtoSegment.cxx:1359` | `PRSegmentFunctions.cxx:1240 :1246` |
| P12 | `cal_kine_range(double L)` null `TGraph` in the prototype | proto bug not reproduced | low | `ProtoSegment.cxx:1380-1390` | `PRSegmentFunctions.cxx:1407-1412` |
| P13 | `fill_kine_tree` moved after the taggers | ordering | low | `NeutrinoID.cxx:251-255` | `TaggerCheckNeutrino.cxx:861` |
| P14 | `kine_best == 0` fallback folded into the accessor | cosmetic + hazard | low | `WCShower.h:50` + per-site | `PRShower.h:152-153` |

Six are port defects or gaps (P1, P2, P4, P5, P8, P9); three are the toolkit
declining to reproduce a prototype bug (P6, P11, P12); one is a determinism
regression the toolkit introduced (P3) and one a determinism improvement it
made (P7); the rest are ordering or cosmetic. **P2 and P3 are the two that do
not depend on an unverified precondition** — the stub returns 1.0 on every
call, and the unordered container is unordered on every call.

---

## §9 What is NOT claimed

- **No event was run.** Every statement is from reading source. No log, no
  output artifact, no A/B.
- **No frequency is measured** for any finding. P1 in particular is stated
  with its precondition unverified: two of the three post-latch PDG rewrites
  were traced and found to refresh the cache, and the third's reachability was
  not established. Its 139.6 MeV figure is a per-occurrence bound conditional
  on that reachability, not an event-averaged bias.
- **P2's magnitude is not estimated.** The prototype's correction graphs are
  uBooNE data files; what an SBND equivalent would do to the energy scale is
  unknown and deliberately not guessed.
- **P3 is a hazard argument, not a measurement.** The container is
  pointer-hashed and the sums are floats; whether the resulting drift is
  observable was not tested.
- **P6, P11, P12 are divergences, not defects.** In each the toolkit's
  behaviour is the better one. They are listed so a future "restore prototype
  parity" pass does not undo them.
- **The recombination model is not re-derived** — §5.5 defers to doc pr/10.
- **`Shower::calculate_kinematics`'s geometry** (`start_point`, `init_dir`,
  `end_point`) was compared at branch level; the helper functions it calls
  were not opened (§7.5). A divergence inside those would not have been seen.
- **`ParticleDataSet`'s range curves** were not compared against the
  prototype's `TGraph`s. §2.6 establishes that the *dispatch* matches; whether
  the underlying curves are the same data is Tier B.
- **`TaggerInfo`'s 1023 defaults were compared mechanically, not read.** The
  script normalises `true`/`false` and numeric literals; a semantic difference
  hidden behind an identical literal would not show up.
- **The BDT feature vectors were not audited.** §2's severity note establishes
  only that `kine_reco_Enu` is consumed, and by which lines.

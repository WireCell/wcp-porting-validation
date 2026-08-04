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

> **§10 (added later) — OWNER FILTER: 14 → 4 findings.** Re-verified at toolkit
> **`407c5ba9`**, eleven commits after the `23bd6783` this audit was written
> against. **The anchors WERE stale this round** and §10.9 re-derives all of
> them. **P3 — one of the two headline findings — is already FIXED at HEAD** by
> `026a7501`, which landed after the read; **P10 and P14 are RESOLVED** by
> §10.7. Six are dropped as improvements over the prototype. The four survivors
> are **not four knobs**: one physics knob (F1), one provably output-identical
> perf change (F3), and two scoping questions that are the owner's to answer
> (F2, F4). §10.6 shows this stage needs a **different gate artifact** from
> every earlier round in the series.

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

---

## §10 Owner filter — 14 → 4

Same round the owner asked for on pr/30 (14 → 4), pr/31 (15 → 9), pr/32
(12 → 4), pr/33 (14 → 5) and pr/34 (14 → 5): **drop the divergences where the
toolkit improves on the prototype; keep only bugs and things missing from the
port**, and give each survivor a concrete fix.

### Re-verification basis, and the anchor result

Re-verified against committed toolkit **`407c5ba9`** — eleven commits after the
`23bd6783` §0 was written against. **Read at `git show HEAD:<file>`, never from
the working tree**: twelve tracked files are dirty from a concurrent session in
this checkout, three of them this stage's (`PRSegmentFunctions.cxx`,
`TaggerCheckNeutrino.cxx`, `sbnd/clus.jsonnet`). Nothing in §10 is read from an
uncommitted edit.

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git rev-parse HEAD                                    # 407c5ba9
git merge-base --is-ancestor 23bd6783 HEAD && echo linear
git diff --stat 23bd6783..HEAD -- clus/src/NeutrinoEnergyReco.cxx \
    clus/src/NeutrinoKinematics.cxx clus/src/PRSegmentFunctions.cxx \
    clus/src/PRShower.cxx clus/inc/WireCellClus/PRShower.h \
    clus/src/NeutrinoShowerClustering.cxx clus/src/TaggerCheckNeutrino.cxx \
    cfg/pgrapher/experiment/sbnd/clus.jsonnet
```

**The anchors were stale.** Unlike pr/34, where the re-derivation came back
clean, seven of this stage's files moved between the read and HEAD:

| file | 23bd6783 → 407c5ba9 | effect on §3 |
|---|---|---|
| `NeutrinoEnergyReco.cxx` | unchanged, 348 lines | P2/P4 anchors **exact** |
| `NeutrinoKinematics.cxx` | unchanged, 327 lines | P1/P5/P6/P7 anchors **exact** |
| `PRShower.cxx` | +56/−29, 1247 → 1303 | **P3 FIXED**; P1/P8/P9 anchors shift |
| `PRSegmentFunctions.cxx` | +25 | P10/P12 anchors shift +25 |
| `NeutrinoShowerClustering.cxx` | +31 | P1's rewrite table shifts |
| `TaggerCheckNeutrino.cxx` | +97 | P5/P13 anchors shift |
| `PRShower.h` | +13 | `get_kine_best` unmoved at `:152` |

§10.9 gives the re-derived table. The lesson is now three rounds old (pr/32's
were 4 commits stale within a day; pr/33's up to +19 lines with a finding
*created* underneath; pr/34's clean): **the answer varies, so the check is not
optional.**

---

### §10.1 The filter

| # | finding | verdict | why |
|---|---|---|---|
| P1 | shower PDG read from an incompletely-refreshed cache | **KEEP — F1** | port defect; the toolkit reads a cache where the prototype reads live |
| P2 | `cal_corr_factor` is a stub returning 1.0 | **KEEP — F2** | port gap; prototype's correction is live in production. **Different class** — §10.3 |
| P3 | dQ/dx accumulated over a pointer-hashed unordered set | **RESOLVED** | **already fixed at HEAD** by `026a7501` — §10.7a |
| P4 | segment path re-collects the whole 2-D charge map per call | **KEEP — F3** | the prototype hoisted, the toolkit did not. **Perf class** — §10.4 |
| P5 | neutrino vertex not SCE-corrected on SBND | **KEEP — F4** | port gap; the field is named `_corr` and is not corrected |
| P6 | BFS shower double-count guard added | drop | toolkit better — the prototype double-counts on a cycle |
| P7 | per-particle arrays index-ordered, not pointer-ordered | drop | toolkit better; reproducing the prototype's order would be an M4 regression |
| P8 | long-muon path never writes `data.particle_type` | **folded into F1** | its whole residual *is* P1's staleness — §10.2 |
| P9 | long-muon `start_point` takes `back()` when `dirsign()==0` | drop | toolkit better, and **decisively so** — §10.8c |
| P10 | `!valid()` fit points skipped | **RESOLVED** | the guard is dead by construction — §10.7b |
| P11 | one-fit-point segment throws in the prototype | drop | prototype bug not reproduced |
| P12 | `cal_kine_range(double L)` null `TGraph` in the prototype | drop | prototype bug not reproduced |
| P13 | `fill_kine_tree` moved after the taggers | drop | no present defect; §3 already bounds it. Forward constraint only |
| P14 | `kine_best == 0` fallback folded into the accessor | **RESOLVED** | survey done, hazard not realised — §10.7c |

**Four findings from five P-numbers** (F1 = P1 + P8). Three resolved, six
dropped. The count is an output, not a target — pr/33 and pr/34 both landed on
five and this one does not.

**They are not four knobs.** This is the round's most useful structural result,
and it differs from every previous round:

| | finding | shape of the fix | done-bar |
|---|---|---|---|
| **F1** | P1 + P8 | one default-OFF knob, `kine_shower_pdg_live` | byte-identical off + visible on (§10.6) |
| **F2** | P2 | **owner scoping question**, not a knob | — |
| **F3** | P4 | unconditional, provably output-identical | perf gate (wall + RSS) |
| **F4** | P5 | **owner scoping question** (config already exists) | — |

Presenting F2 and F3 as peers of F1 would smuggle two category changes past the
owner — the pr/33 GOTCHA 20 trap. They are labelled instead.

---

### §10.2 F1 = P1 + P8 — `kine_shower_pdg_live`

**Why they merge.** §3 lists them separately, but P8's residual is stated in its
own text: the `−13` case comes out consistent, and *"the residual exposure is
the general one from P1: the field is a cache and this path does not refresh
it."* That is not a second defect; it is the same defect seen from the writer's
side instead of the reader's. One finding.

**Why the fix is on the reader, not the cache.** There are two candidate edits
and only one is correctly scoped:

- **(a) refresh the cache** in `calculate_kinematics_long_muon` — i.e. add
  `data.particle_type = …` beside the local at `PRShower.cxx:1224`. This
  changes what **every** consumer of `Shower::get_particle_type()` sees, and
  they are not all in this stage: `NeutrinoEnergyReco.cxx:305` dispatches on
  it, and `MultiAlgBlobClustering.cxx:1522`/`:1539` feed it to `keep_node` in
  the Bee particle-flow tree (doc pr/34's stage). A knob placed here leaks
  out of the stage it is gated for.
- **(b) read the live start segment** at the four `fill_kine_tree` sites
  (`NeutrinoKinematics.cxx:109`, `:119`, `:274`, `:286`), which is exactly
  what the prototype does (`kine.h:53 :67 :175 :187 :224 :242`).

**(b) is the fix.** It is confined to this stage's four reads, it is what the
translation note at `NeutrinoKinematics.cxx:31` claims is already happening,
and — the property that matters — **it is correct independently of when the
cache is refreshed.** That last point is not a convenience: §10.9 shows §3's
`:3272` citation for "the last `calculate_shower_kinematics`" is stale, and at
HEAD there are **two** such calls (`NeutrinoShowerClustering.cxx:3267` and
`:3291`), so the refresh *schedule* P1's exposure analysis rests on needs a
re-read. Fix (b) does not depend on that re-read; fix (a) would.

This is the pr/33 GOTCHA 22 lesson applied prospectively rather than after the
fact: **before widening a write, grep every other reader of the same field.**

**Proposed knob.** `BeeKineConfig`-style member on the component that owns
`fill_kine_tree`, C++ default `false`:

```cpp
// false = today's cached read.  true = the prototype's live start-segment read.
bool kine_shower_pdg_live{false};
```

and at each of the four sites:

```cpp
const int pdg = (m_cfg.kine_shower_pdg_live && shower->start_segment()
                 && shower->start_segment()->has_particle_info())
                ? shower->start_segment()->particle_info()->pdg()
                : shower->get_particle_type();
```

Note the `has_particle_info()` fallback keeps the knob-on path defined where
the prototype's `get_particle_type()` would return 0 anyway (§5.11's argument,
same shape).

**Ship it with a counter.** The pr/32 F3 precedent (which came back 0/2219, and
was the only reason that finding could be closed) applies exactly: count
`cached != live` at `:109` and log the **pair**, not the count — §7.1 already
explains why the `11`-vs-`211` cases and the cache-stuck-at-0 cases must be
distinguishable. Zero hits over the 572-event valfast manifest demotes F1 to a
§5 entry without anyone having to run an A/B.

**Bound, unchanged.** Per `11` → `211` occurrence, `kine_reco_Enu` moves by the
pion rest mass, 139.6 MeV. That is a per-occurrence bound, not an
event-averaged bias.

---

### §10.3 F2 = P2 — `cal_corr_factor`: a gap, and a question, not a knob

**This survivor is a different class and must be read as one.** It is kept
because the prototype applies a calibration here and the toolkit applies none —
that is squarely "missing from the port". But it has **no default-OFF knob that
reproduces the prototype**, and proposing one would be the M15 trap this
document already names in P2's own text:

- part **(a)**, the seven hard-coded uBooNE U-plane wire ranges given a
  `1/0.7` boost, is detector-specific and **must not** be ported to SBND;
- part **(b)**, `gu·gv·gw`, is a general position correction whose three
  `TGraph`s are **uBooNE calibration data** (`calib_{u,v,w}_corr.txt`).
  Reproducing uBooNE's graphs on SBND would be worse than applying nothing.

So the deliverable is not code. It is the escalation-rule-1 question, with the
facts assembled so the owner can answer it in one pass:

1. **Does SBND want a position-dependent charge correction at this point?**
   The prototype's is applied per charge hit, inside the plane sums, before the
   recombination division — so it is not equivalent to a global scale factor
   and cannot be absorbed into the doc pr/10 recombination retune (§4).
2. **If yes, is there an SBND calibration product it could read?** Unknown to
   this audit and deliberately not guessed.

**If and only if the answer to 1 is yes**, the implementable shape is a config
key holding a calibration-file path, absent by default:

```jsonnet
[if kine_corr_files != null then 'kine_corr_files']: kine_corr_files,
// C++ default: empty => cal_corr_factor returns 1.0, exactly today's behaviour.
// Key omitted when unset => byte-identical pre-change compiled config.
```

The plumbing already exists — the stub's signature receives
`IDetectorVolumes::pointer` and reaches the grouping — so this is a small
change *once the data exists*. **Say plainly that it delivers nothing until it
does.** Shipping the empty-default key alone would be a knob that can only ever
be off, which is worse than no knob: it reads in the config as though the
correction were available.

**What is still not claimed:** the magnitude. Every toolkit `kine_charge`
carries 1.0 where every prototype one carries a per-point factor; how much that
is worth on SBND is unknown.

---

### §10.4 F3 = P4 — a perf change with a different done-bar

Kept because the toolkit is *worse* than the prototype here, not better: the
prototype hoists the 2-D charge collection into members via
`collect_2D_charges()` once, and the toolkit's **shower** overload does the
same (`NeutrinoEnergyReco.cxx:242`), but the **segment** overload builds fresh
locals on every call (`:262-264`). `fill_kine_tree` calls it once per track
segment, so the toolkit pays `O(n_track_segments × n_charge_hits)` where the
prototype pays `O(n_charge_hits)`.

**This is not a physics knob, and it must not be gated like one.** §5.6
establishes that the two collections agree on everything this stage reads:
`update_dQ_dx_data` mutates only `charge_err`, and `kine_charge_from_maps`
reads only `.charge` (`:130`). So caching the maps the way the shower path
already does is **output-identical by construction** — the fix is unconditional
and needs no knob:

```cpp
// NeutrinoEnergyReco.cxx:262 — mirror the shower overload at :242
if (m_charge_2d_u.empty()) collect_charge_maps(track_fitter);
```

Its done-bar is CLAUDE.md §4's **perf** block, not the behaviour-change block:
byte-identical member hashes **plus** before/after wall and peak RSS from
`timecmd.py` on a named manifest, with the labels quoted. A pure byte-identical
PASS is necessary but not sufficient — it would not show the change did
anything.

One caveat to carry into that measurement: the two overloads collect at
*different times* (shower at shower-clustering time, segment at
`fill_kine_tree` time). §5.6's argument is what makes them interchangeable;
if a future change makes `kine_charge_from_maps` read any field other than
`.charge`, that argument dies and this becomes a correctness change.

---

### §10.5 F4 = P5 — SCE: also a question, and the config already exists

Kept as a gap: the prototype applies `func_pos_SCE_correction` to the neutrino
vertex **unconditionally** (`kine.h:3-9`, with
`init_Pos_Efield_SCE_correction()` called unconditionally at
`wire-cell-prod-nue.cxx:197`); the toolkit applies it only when a
`clus_geom_helper` is configured, and SBND never configures one.

Unlike F1 this needs **no new knob** — `clus_geom_helper` is already the knob,
defaulting to `""` at `TaggerCheckNeutrino.cxx:295` (re-derived; §3 cites
`:266`). So the finding reduces to two owner questions plus one thing that is
worth doing either way:

1. **Should SBND set `clus_geom_helper`?** That is a physics call about whether
   an SCE correction belongs at this point on SBND at all — escalation rule 1,
   not this audit's to make.
2. **If not, the name still lies.** `kine_nu_{x,y,z}_corr` are published into
   `T_kine` and into the PR display dump carrying `_corr` in the name while
   holding the raw fitted vertex, and **nothing at runtime says so**. Renaming
   a published tree branch is a downstream break and is not proposed. What is
   proposed is one line that costs nothing:

```cpp
// NeutrinoKinematics.cxx:70 — the else branch already has the TODO comment
SPDLOG_LOGGER_WARN(log, "fill_kine_tree: no geom_helper — kine_nu_*_corr are "
                        "the RAW fitted vertex despite the _corr name");
```

A once-per-job warning turns a silent naming lie into a visible one. It changes
no output value, so it is byte-identical on the artifact gate by inspection.

---

### §10.6 The gate — and why it is not the earlier rounds'

**This is the round's other structural result.** pr/34 §10.7 established that
the series' standard `pctree-pr-evt*.tar.gz` member-hash gate would PASS
**vacuously** for a display-only stage. This stage is the *opposite* case, and
the standard gate fails it for the opposite reason: `KineInfo` is not in the
pctree at all. Neither prior answer transfers.

`KineInfo` reaches exactly two artifacts, both verified at HEAD:

| artifact | writer | carries |
|---|---|---|
| `T_kine` in the tracking ROOT file | `root/src/UbooneTaggerOutputVisitor.cxx:1089-1095` (SBND reuses it as-is — `sbnd/clus.jsonnet:1655`) | `kine_reco_Enu` + the parallel arrays |
| `calib-pr-evt<ID>.json` | `clus/src/PrDisplayDump.cxx:455-470` (`dump_kine`) | the same, verbatim, already in MeV/cm |

**Use the JSON.** `dump_kine` emits `kine_reco_Enu`, `kine_reco_add_energy`,
`kine_nu_{x,y,z}_corr` and all four parallel arrays
(`kine_energy_particle`, `kine_energy_info`, `kine_particle_type`,
`kine_energy_included`) into a plain JSON file — **directly diffable, no
archive, no embedded timestamps, so M2 does not apply and `hash_archive.py` is
not needed.** The ROOT alternative is worse on every axis: ROOT files are not
byte-comparable (compression settings and timestamps), so `T_kine` would need
a tree-dump comparator that does not exist in `abtest/` today.

The dump is a diagnostic stage and is off by default; enable it the way doc
pr/26 and pr/28 did:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <base-tag> <new-tag> data <evts>
# gate:  diff  work/<a>/evt*/calib-pr-evt*.json  work/<b>/evt*/calib-pr-evt*.json
```

**Two-sided, as in pr/34 §10.7.** Knob-**off** identical satisfies the
byte-identical bar. Knob-**on** must *differ* in the `kine_*` keys — for F1
that is the first empirical evidence that the cache and the live PDG ever
disagree, which today rests only on a source argument. If knob-on is also
identical, F1's counter (§10.2) is the cheaper way to learn the same thing and
should be run first.

**One thing the JSON gate cannot see**: the BDT scores. `kine_reco_Enu` is numu
XGBoost variable 69 (`UbooneNumuBDTScorer.cxx:507`) and is in the nue reader
(`UbooneNueBDTScorer.cxx:332`, `:1660`). A `kine_reco_Enu` move that the JSON
shows is a *score* move that the JSON does not show. Any F1 flip needs the
score comparison too; scoping that is pr/27 §10's stage, not this one's.

---

### §10.7 Resolved outright — three, not merely dropped

**(a) P3 is already FIXED at HEAD.** Commit **`026a7501`** ("clus: order the
shower edge walk by graph index — the PR display dump is now run-to-run
identical", doc pr/28 §15) landed *after* the `23bd6783` read and converted
precisely the site P3 names. `PRShower.cxx:1134` is now `:1164`:

```cpp
// ordered_edges: this is the site that was actually caught.  Both
// the `total_length +=` and the push order of vec_dQ/vec_dx are
// observable -- cal_kine_dQdx() is a plain `kine_energy += dE` over
// the vector (PRSegmentFunctions.cxx:1281), so the summation order
// sets the last bit.  showers[1]/kine_dQdx moved 1314.124434586102
// -> ...103 (rel 6.9e-16) between two `setarch -R` runs of the same
// binary on SBND evt 388 (doc pr/28 sec 15).
for (auto edesc : ordered_edges(*this, m_full_graph)) {
```

Two things follow. First, **P3 was real** — the comment records the measurement
§7.3 asked for (two `setarch x86_64 -R` runs, same binary, SBND evt 388), and
the drift is ULP-level (rel 6.9e-16), so it was *not* classification-changing on
that event. §7.3 is closed. Second, the fix is **not** a spot fix: the same
commit converted four more accumulating walks in this file, and the
`this->edges()` occurrences remaining at HEAD are all benign —

```
:662   deliberate, with a comment: unconditional integer count, no order observable
:671   .size()
:746   .size()
:954   .size()
:1196  inside a comment
:1205  inside a commented-out debug line
```

— so "P3 is fixed" is a checked claim about the file, not about one line.

**Keep §6's row, though, with its meaning corrected.** The prototype still
iterates `map_seg_vtxs`, a `std::map<ProtoSegment*,…>`, at `WCShower.cxx:448`
and `:515`. The row moves from *both broken* to *toolkit fixed, prototype still
pointer-ordered* — which means this site is **still not bit-comparable
prototype↔toolkit**, and nobody should expect the two trees' `kenergy_dQdx` to
agree to the last bit.

**(b) P10's guard is dead by construction.** §7.4 asked whether a `Fit` can
carry a real `dQ`/`dx` with `index < 0`, and noted correctly that
`Fit::reset()` (`PRCommon.h`) clears `index` while **leaving `dQ`/`dx`
untouched** — which makes the state constructible, not merely hypothetical. It
is nonetheless unreachable at the point `segment_cal_kine_dQdx` runs, and the
proof is `form_map_graph`:

- every fit it pushes into `saved_fits` gets an index on **all three** paths —
  middle points `TrackFitting.cxx:3238`, first point `:3247`/`:3252`, last
  point `:3260`/`:3265`;
- middle points whose three plane quantities sum to zero are **not pushed at
  all** (`:3220`) — they are dropped, which is exactly what the drop counter at
  `:8350-8360` reports;
- `segment->set_fit_associate_vec(std::move(saved_fits), …)` (`:3275`) then
  **replaces** the segment's fit vector wholesale.

And both `reset_fit_prop()` sites are followed by `form_map_graph` inside the
same function (`:3125` → the segment loop immediately below; `:8335` →
`:8350`), over the **same** `get_segment_edges()` set that was just reset — so
reset-set and refit-set are identical by construction, not by coincidence. The
in-tree comment at `:8338` states this invariant explicitly ("`PR::Fit::reset()`
clears index outright and they must be rebuilt here").

⇒ `!fits[i].valid()` at `PRSegmentFunctions.cxx:1253` (re-derived; §3 cites
`:1228`) is **never true**, and the surviving `dx <= 0` clause is already shown
equivalent to the prototype by §2.4. **P10 moves to §5.** Same shape as pr/32's
P7 — dead code — but proven by construction rather than measured.

**(c) P14's hazard is not realised — the survey §7.7 asked for is done.** The
question was whether any consumer tests `get_kine_best() == 0` to detect the
"no reliable estimate" sentinel — `data.kenergy_best = 0` for a shower-flagged
object with `start_connection_type != 1`, at **`PRShower.cxx:983`** (single-
segment branch) and **`:1188`** (multi-segment), against the prototype's
`WCShower.cxx:359`; §3 cites `:960` for the first, which is stale. The folded
accessor (`PRShower.h:152-153`) makes such a test unreachable. Surveyed both
trees:

- **Toolkit** — `grep -rn 'get_kine_best()' clus/src clus/inc root/src`. Every
  site that tests it uses the charge-fallback idiom
  (`!= 0 ? get_kine_best() : get_kine_charge()`), which the folded accessor
  reproduces exactly: ~30 sites in `NeutrinoTaggerNuE.cxx`, three in
  `NeutrinoTaggerSinglePhoton.cxx`, `NeutrinoTaggerSSM.cxx:210`, and both
  if/else forms in `NeutrinoShowerClustering.cxx:1672-1676` and `:2052-2064`
  (both **do** have the `else` arm). **Zero** sites use the value for anything
  else.
- **Prototype** — the guarded idiom is the universal one: twelve
  `Eshower = get_kine_best(); … else … get_kine_charge();` pairs in
  `NeutrinoID_nue_tagger.h` (`:338 :560 :1014 :1676 :1830 :2291 :2496 :2765
  :2917 :3162 :3461 :3861`), and the same in
  `NeutrinoID_cosmic_tagger.h:93-98`, `:288-293`, `:362-368`, `:393-397`.
  Nothing reads the sentinel.

⇒ the toolkit's bare `shower->get_kine_best()` calls — including the ones
`NeutrinoTaggerCosmic.cxx:83-84` asserts are "correct and sufficient
everywhere" — are equivalent to the prototype's guarded pairs. **That assertion
is now checked rather than asserted.** P14 moves to §5, and §7.7 is closed.

---

### §10.8 Dropped, with the reason

**(a) P6, P11, P12 — the toolkit declines to reproduce a prototype bug.**
Unchanged from §3, and §9 already says they are listed so a future
"restore parity" pass does not undo them. P6 in particular means toolkit and
prototype `kine_reco_Enu` **cannot** be expected to agree on an event with a
cycle in the track graph — a diff there is not a defect.

**(b) P7 — the toolkit is deterministic where the prototype is not.**
Reproducing `std::set<ProtoSegment*>` iteration order would be a deliberate M4
regression. Note the standing consequence: the prototype's `kine_reco_Enu` is a
float sum in pointer order and is therefore **not reproducible run-to-run**,
while the toolkit's is. With P3 now fixed too (§10.7a), this stage's output is
run-to-run stable on the toolkit side — which is the prerequisite for the
§10.6 gate to mean anything.

**(c) P9 — dropped, and the evidence is stronger than §3 had.** §3 called the
toolkit's choice "arguably better" and declined to rank them. It is not
arguable. `WCShower`'s constructor (`prototype_base/pid/src/WCShower.cxx:7-30`)
explicitly zeroes `start_point`:

```cpp
start_point.x = 0;  start_point.y = 0;  start_point.z = 0;
```

and `calculate_kinematics_long_muon` (`WCShower.cxx:288`) runs once per shower
under the `flag_kinematics` latch — so it always sees a **fresh** shower. When
`flag_dir == 0` the prototype therefore leaves `start_point` at the literal
origin `(0,0,0)`, and the farthest-muon-vertex search at `:328` then measures
distance from the origin, picking an essentially arbitrary vertex and setting
`end_point` from it. The toolkit's `back()` (`PRShower.cxx:1275`) is a real
point on the start segment.

`dirsign() == 0` is plainly reachable — it is the member default
(`PRSegment.h:158`, `int m_dirsign{0}`) and there are thirteen explicit
`dirsign(0)` writes across `NeutrinoTrackShowerSep.cxx`,
`PRSegmentFunctions.cxx` and `NeutrinoStructureExaminer.cxx`. So this is a
**reachable prototype bug the toolkit does not reproduce**, which is the P11/P12
class, not an undocumented coin-flip. Dropped on the same grounds.

**(d) P13 — no present defect.** §3 already establishes the bound: the toolkit's
BDT scorers are separate pipeline stages reading `tf->get_kine_info()` after
`TaggerCheckNeutrino` has stored it, so nothing is starved. The exposure is
forward-looking only — a *future* tagger reading `KineInfo` would read zeros —
and `grep -n 'get_kine_info()'` returns five hits, none a tagger. That is a
constraint to remember when the next tagger lands, not a change to make now.

---

### §10.9 Re-derived anchors (`23bd6783` → `407c5ba9`)

§3's citations, corrected. Files not listed are unchanged.

| what | §3 says | at `407c5ba9` |
|---|---|---|
| P1 cache write, single-segment branch | `PRShower.cxx:939` | **`:962`** |
| P1 cache write, multi-segment branch | `PRShower.cxx:1050` | **`:1073`** |
| P1 rewrite, `examine_shower_1` | `NeutrinoShowerClustering.cxx:1921` → recompute `:1934` | **`:1938` → `:1951`** |
| P1 rewrite, `examine_showers` | `:2372` → recompute `:2374` | **`:2391` → `:2393`** |
| P1 rewrite, `id_pi0_with_vertex` | `:2768`, guard `:2764` | **`:2787`, guard `:2783`** |
| "the last `calculate_shower_kinematics`" | `:3272` (one call) | **`:3267` *and* `:3291` — TWO calls** |
| P3 the unordered walk | `PRShower.cxx:1134` | **`:1164`, now `ordered_edges`** |
| the deterministic helper it should have used | `PRShower.cxx:1209` | **`:1239`** |
| P14 the `kenergy_best = 0` sentinel | `PRShower.cxx:960` | **`:983`** (nseg==1) *and* **`:1188`** (multi) |
| P8 the long-muon local | `PRShower.cxx:1194` | **`:1224`** |
| P9 the `dirsign` branch | `PRShower.cxx:1241-1246` | **`:1271-1276`** |
| P10 the `!valid()` guard | `PRSegmentFunctions.cxx:1228` | **`:1253`** |
| `segment_cal_kine_dQdx` | `PRSegmentFunctions.cxx:1214` | **`:1239`** |
| §2.4 the dE clamp | `:1264-1268` | **`:1289-1293`** |
| `cal_kine_dQdx(vQ,vx,recomb)` | `:1274` | **`:1299`** |
| P12 `cal_kine_range` | `:1387` (fn), `:1407-1412` | **`:1412` (fn), `:1432-1437`** |
| §2.2 `segment_cal_4mom` | `:1604-1634` | **`:1629-1660`** |
| P5 `clus_geom_helper` default | `TaggerCheckNeutrino.cxx:266` | **`:295`** |
| P5 the call site's comment | `:864` | **`:917`** |
| P13 `init_tagger_info` | `:741` | **`:794`** |
| P13 `match_isFC` | `:853` | **`:906`** |
| P13 `fill_kine_tree` | `:861` | **`:914`** |

**Exact, needing no correction**: everything in `NeutrinoEnergyReco.cxx`
(P2 `:14-35`, the three call sites `:221 :276 :299`, `collect_charge_maps`
`:226`, the shower cache test `:242`, P4's fresh locals `:262-264`, the latch
`:303`) and everything in `NeutrinoKinematics.cxx` (P1 `:109 :119 :274 :286`,
the translation note `:31`, P5 `:61-75`, P6 `:231-239`, P7 `:171 :205`,
`kine_reco_Enu` `:301-305`) — both files are byte-identical across the eleven
commits. `PRShower.h:152-153` (`get_kine_best`) is also unmoved.

**The `:3272` correction is the one that matters**, and it is why F1's fix is
scoped to the reader (§10.2): P1's exposure analysis is a claim about the order
of PDG rewrites relative to the kinematics latch, and that ordering now has two
call sites where §3 saw one. **Anyone re-opening P1's *frequency* question must
re-read the ordering first.** F1's proposed fix is unaffected, because reading
the live PDG is correct whatever the refresh schedule is.

---

### §10.10 What §10 does not claim

- **No code was changed, no event was run, no gate was executed.** §10 is a
  filter over §3 plus the re-verification in §10.7 and §10.9. The toolkit
  working tree was read at `git show HEAD:` and not written to.
- **P3's fix was verified by reading `026a7501` and the file at HEAD, not by
  running the two `setarch` jobs.** The ULP measurement quoted in §10.7a is the
  in-tree comment's, not this round's.
- **No frequency is measured for F1**, and §10.9 shows its ordering premise
  needs a re-read. The 139.6 MeV figure remains a per-occurrence bound.
- **F2's magnitude is still not estimated**, and §10.3 deliberately declines to
  propose a knob for it.
- **F3 is argued output-identical, not measured so.** The argument is §5.6's
  and it is only as strong as the claim that `kine_charge_from_maps` reads
  `.charge` and nothing else.
- **§10.7c's survey covers `get_kine_best()`. It does not cover
  `get_kine_charge()` or `get_kine_range()`**, whose consumers were not
  enumerated.
- **The `pid` submodule re-check is still owed.** pr/34 §7.7 records
  `prototype_base/pid` as +5833/−989 over 26 files against the merge-base
  `a5fc0b9`. This stage's own files are clear of that diff (§0), but pr/28–pr/33
  have not been re-checked and this round does not do it.

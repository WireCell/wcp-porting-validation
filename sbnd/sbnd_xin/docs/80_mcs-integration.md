# Doc 80 — Multiple Coulomb Scattering (MCS) muon momentum: integration plan

Owner ask (2026-08-24): *"I would like to integrate my colleague's MCS code into
toolkit … This should be in a separate package `./mcs`?  If we gave a muon
segment, or a long muon, it can then use that to run the MCS code to estimate
the momentum.  We should figure out a way to store the momentum into the final
tree.  We need to do some validation, use stopping muon, as well as the range
based energy estimation to compare with MCS and also the dQ/dx → dE/dx energy
estimation.  For this round, I am not ready to do the implementation yet, but
try to plan it with detailed steps."*

**Status.**  PLANNING ONLY.  **No C++ is written and no toolkit file is touched
by this doc.**  Everything below is the design and the staged work plan, with
the facts it rests on verified against the tree at toolkit `436915f2`
(`apply-pointcloud`).  Section 11 lists the decisions the owner still has to
make before round 0 starts.

## Repro

```bash
S=<scratch>   # this round: /home/xqian/tmp/claude-25225/.../603257f1-*/scratchpad

# 1. The upstream sources (PINNED -- ubreco's `develop` moves, a bare URL is not
#    a reproducible citation)
cd $S
git clone https://github.com/uboone/Multiple_Coulomb_Scattering.git mcs_standalone
git -C mcs_standalone checkout 6aa0b9c576e22b954adf4472594a422929896c5d   # 2026-06-12
U=https://raw.githubusercontent.com/uboone/ubreco/8a5731fda2577b007e7dd054863d49cc261c8f83
curl -sSL -O $U/ubreco/WcpPortedReco/ProducePort/WCMCSTrajectory.h
curl -sSL -O $U/ubreco/WcpPortedReco/ProducePort/WireCellMCS_module.cc
curl -sSL -O $U/ubreco/WcpPortedReco/ProducePort/job/wirecellmcs.fcl
curl -sSL -O $U/ubreco/WcpPortedReco/ProducePort/job/run_wirecellmcs.fcl

# 2. The reference number this whole plan's round-1 gate rests on
#    (NOT YET RUN -- see sec 5, round 0 step b)
cd $S/mcs_standalone && make > make.log 2>&1; echo rc=$?
./mcs_example > mcs_example.log 2>&1; echo rc=$?
grep emu_MCS mcs_example.log      # expect ~0.699 (true 0.735 GeV)

# 3. Toolkit facts cited below
cd /nfs/data/1/xqian/toolkit-dev/toolkit
sed -n '119,131p' clus/inc/WireCellClus/PRCommon.h            # PR::Fit
sed -n '25,67p'   clus/inc/WireCellClus/NeutrinoTaggerInfo.h  # PR::KineInfo
sed -n '1682p'    clus/src/TaggerCheckNeutrino.cxx            # segments_in_long_muon
sed -n '2704p'    clus/src/TaggerCheckNeutrino.cxx            # fill_kine_tree call
sed -n '94,102p'  waft/wcb.py                                 # package discovery
sed -n '120,124p' util/inc/WireCellUtil/Interpolate.h         # irrterp CLAMPS
```

## 1. Why

The toolkit estimates particle energy three ways today — range
(`cal_kine_range`), dQ/dx→dE/dx calorimetry (`segment_cal_kine_dQdx`), and
charge (`cal_kine_charge`) — recorded per particle in `PR::KineInfo` with a
`kine_energy_info` code of 0/1/2.

MCS is orthogonal to all three: it reads only the *shape* of the trajectory.
That makes it work on **exiting** muons, where range fails outright, and makes
it insensitive to the recombination and electron-lifetime modelling that dQ/dx
depends on.  Adding it gives SBND a momentum handle on non-contained muons and
makes the three existing estimators cross-checkable against an independent one.

The code is already written and validated in MicroBooNE (arXiv:2605.03048).  So
this is **not** algorithm development.  The work is four separate problems:

  (a) finding a package home that does not violate WCT's dependency policy;
  (b) a faithful ROOT-free re-implementation;
  (c) wiring SBND's PR data model into an API written against uBooNE's
      whole-event `T_rec_charge_blob` cloud;
  (d) deciding whether a MicroBooNE angular-resolution tune is defensible on
      SBND at all.

(d) is a physics question and it is the one that can invalidate the whole
exercise.  Section 9 puts a cheap measurement in front of it.

## 2. What the upstream code is

Two shipping forms of the same algorithm:

| source | pinned | role |
|---|---|---|
| `uboone/Multiple_Coulomb_Scattering` `src/mcs.{h,cxx}` (259 + 602 lines) | `6aa0b9c` | standalone, self-contained, **this is what we port** |
| `ubreco` `WireCellMCS_module.cc` + `WCMCSTrajectory.h` (519 + 1212 lines) | `8a5731f` | the art wrapping; same algorithm inlined.  Read for the *input* convention only |
| `ubreco` `job/wirecellmcs.fcl` | `8a5731f` | the tune constants, already externalized |

License is **MIT** (`Copyright (c) 2026 MicroBooNE Collaboration`) — compatible
with WCT's.  The port carries the copyright notice plus per-function
`// upstream (mcs.cxx lines N-M)` citations, following the citation convention
already used for `prototype_base/` ports.

### 2.1 API

```cpp
MCS mcs;
mcs.run(vtx_start /*{x,y,z} cm*/, vtx_end, points /*vector<vector<double>> cm*/);
// -> members: mu_tracklen   [cm]
//             emu_tracklen  [GeV]  range-based, from MCS's OWN CSDA table
//             emu_MCS       [GeV]
//             ambiguity_MCS [1 = most ambiguous]
```

Clean and small.  Three inputs, four outputs, no framework coupling.

### 2.2 The four stages

1. **`trim_trajectory` (mcs.cxx:296)** — the input cloud is the **whole event**.
   Build a directed graph over all points (edges only along `end-start`, ≤20
   nearest by the score `d + d²`), run a Dijkstra-flavoured relaxation from the
   point nearest the muon start, then walk `prior` pointers back from the point
   nearest the muon end.  *This is the stage that isolates the muon from delta
   rays and crossing tracks* — it is not a preprocessing convenience, it is load
   bearing.
2. **`form_segs` (:365)** — global PCA, flip the principal axis along the muon
   direction, sort points by their projection on it, slice into ~14 cm segments
   (`fitSegPCA`), PCA each segment.
3. **Angles (`setSegAngles` :143)** — per adjacent segment pair, the 3-D
   scattering angle and its two projections `theta_xz` (`angleProjB`) and
   `theta_yz` (`angleProjC`).
4. **`estimate_energy` (:569)** — a double-Gaussian negative-log-likelihood over
   all segment angles, with the Highland term `sigmaH` modulated by a
   quartic-decay factor and a MicroBooNE *resolution* term added in quadrature;
   minimised in KE over `[0, 4 GeV]`.  `ambiguity_MCS` is the largest likelihood
   ratio between the global minimum and two side minima.

`estimate_energy` additionally depends on `vx` — the x-component of each
segment's fit direction — through a 5-way slicing
(`vx_edges = {0, 0.1, 0.2, 0.35, 0.75, 1}`) with energy-dependent blending
between slices.  **This is drift-geometry specific** and is the sharpest edge of
the tune-transfer question in sec 9.

## 3. Where it lands

`toolkit/CLAUDE.md` forbids adding a new external dependency to a plugin, and
forbids plugin→plugin links.  Three facts decide the layout:

- `clus/` has **zero** ROOT includes.  Only `root/` links ROOT
  (`root/wscript_build`: `use='WireCellAux WireCellClus ROOTSYS'`).
- The MCS core uses ROOT for three things only: `TMatrixDEigen` (PCA),
  `TGraph::Eval` (a 20-point table), `TF1::GetMinimumX` (1-D minimisation).
  All three are replaceable.
- `quickhull/` is the precedent for a **non-plugin helper library** inside the
  tree: `bld.smplpkg('WCPQuickhull', use='ZLib')`, consumed by `clus` via
  `clus/wscript_build`'s `uses`.

So:

```
mcs/                              NEW.  Plain numeric library, NOT a plugin.
  wscript_build                   bld.smplpkg('WireCellMcs', use='WireCellUtil')
  CMakeLists.txt                  wct_package(WireCellMcs USE WireCellUtil)
  inc/WireCellMcs/MuonMCS.h       ROOT-free port of mcs.h
  src/MuonMCS.cxx                 ROOT-free port of mcs.cxx
  test/doctest_mcs_reference.cxx  the 0.699 GeV golden test
  test/data/mcs_reference.json    fixture converted from simulated_event.root
  docs/mcs.org

clus/  wscript_build              uses += ' WireCellMcs'
       src/MuonMCSDriver.cxx      NEW free static fn: select muon, harvest
                                  points, call MuonMCS, fill KineInfo
       src/TaggerCheckNeutrino.cxx  + knob-gated call site (sec 6)
root/  src/UbooneTaggerOutputVisitor.cxx  + 5 T_kine branches, knob-gated
cfg/pgrapher/experiment/sbnd/clus.jsonnet  the knobs, key-suppressed
```

Nothing depends on a plugin and no new external dependency is introduced.

> **`mcs/` is LINKED into `clus`, not dlopened.**  Do **not** add
> `"WireCellMcs"` to any jsonnet `plugins:` list.  That list is for plugins
> loaded by `PluginManager` (`util/src/PluginManager.cxx:30-66`); adding `mcs`
> to it would make it a plugin and re-create exactly the plugin→plugin edge this
> layout exists to avoid.

### 3.1 Build mechanics that fail silently if got wrong

Verified in `waft/`:

- waf discovers packages by globbing `*/wscript_build` (`waft/wcb.py:94-102`) at
  **configure** time, not build time.  `./wcb configure` must be re-run once
  after `mcs/wscript_build` appears.
- The directory name must be the library name minus `WireCell`, **lowercased**.
  Both the rpath (`waft/smplpkgs.py:380-395`, `sd = one[8:].lower()`) and the
  per-package doctest binary name `build/mcs/wcdoctest-mcs`
  (`waft/smplpkgs.py:560-595`, which uses `testdir.parent.name`) derive from it.
  A mismatch gives a wrong rpath and a wrongly-named test binary, not a build
  error.
- Headers must live at `inc/WireCellMcs/*.h`.  Only that glob is installed;
  headers elsewhere under `inc/` compile but never install.
- `build/mcs/wcdoctest-mcs` appears automatically from `mcs/test/doctest_*.cxx`
  with no registration anywhere.
- CMake keeps an **explicit** list.  Add `mcs/CMakeLists.txt` and append `mcs`
  to `WCT_PACKAGES` at `CMakeLists.txt:191`, or `cmake/test/parity.sh` fails.
  (`flash/` is the standing precedent for a waf-only local package — it has a
  `wscript_build` and no `CMakeLists.txt`.  See sec 11 Q1.)

## 4. Mapping the uBooNE inputs onto the SBND PR model

| ubreco input | toolkit equivalent |
|---|---|
| highest-energy PDG-13 `simb::MCParticle` from `wirecellPF` (`WireCellMCS_module.cc:197-207`) | see `muon_source`, sec 4.2 |
| `portedWCSpacePointsTrecchargeblob` — the whole-event `T_rec_charge_blob` cloud | union of `Segment::fits()[i].point` over all PR segments **+** vertex `Vertex::fit().point`.  This is exactly what `MultiAlgBlobClustering.cxx:~980-1050` assembles for the Bee `track_fit` layer and what `SbndPrMagnifyTrackingVisitor.cxx:480` writes as `T_rec_charge` |

**No new producer is needed** — the driver harvests the cloud straight off the
PR graph.

### 4.1 Units trap

`PR::Fit::point` and `Fit::dx` are in WCT **internal** length units (mm = 1,
cm = 10 — see the warning at `PRSegmentFunctions.cxx:2904-2915`).  Every MCS
constant is in **cm**: `seg_length = 14`, `rho = 1.396 g/cm³`, the CSDA table,
the `d + d²` edge score, the `2*seg_length` guard.

The driver converts on the boundary — `point.x()/units::cm` going in,
`units::MeV` coming out — and `MuonMCS` itself stays in plain cm/MeV.  Keeping
the library in upstream's units is deliberate: it stays a faithful,
independently testable copy, and there is exactly one place to get the
conversion wrong instead of forty.

### 4.2 `muon_source` — three modes, all implemented

| mode | rule | rationale |
|---|---|---|
| `pf_muon` **(default)** | walk the PR graph; take the segment with `abs(particle_info()->pdg()) == 13` and the largest `particle_info()->kinetic_energy()`; start/end = that segment's first/last `Fit::point` | reproduces `WireCellMCS_module.cc:197-207` |
| `long_muon` | the `segments_in_long_muon` chain built by `examine_direction` (`clus/src/NeutrinoVertexFinder.cxx:1859-1919`); start/end = the two extreme vertices of the chain | the owner's *"a long muon"*; spans segments, closest to a real muon track |
| `longest_segment` | longest `PR::Segment` by `segment_track_length()` above `muon_min_length_cm`, no PID required | PID-independent control arm; catches muons the PF chain mislabels |

All three ship.  One becomes the SBND default after the round-4 comparison.

**Two verified facts constrain the design**, and together they are why the
driver is a *call site* rather than a new visitor (sec 6):

- **`pf_muon` cannot be defined on `KineInfo`.**  Its per-particle arrays carry
  no provenance — no segment id, no shower id
  (`clus/inc/WireCellClus/NeutrinoTaggerInfo.h:25-67`) — and MCS needs the
  actual `Fit::point` list, not a scalar KE.  So the rule is applied on the PR
  graph instead.  `Aux::ParticleInfo`
  (`aux/inc/WireCellAux/ParticleInfo.h:20`) exposes `pdg()` and
  `kinetic_energy()` directly off `Segment::particle_info()`
  (`clus/inc/WireCellClus/PRSegment.h:88`), stamped by `segment_cal_4mom`, so
  the selection needs nothing from `fill_kine_tree`.
- **`segments_in_long_muon` is not persisted.**  It is a function-local
  `IndexedSegmentSet` declared at `clus/src/TaggerCheckNeutrino.cxx:1682`,
  inside `TaggerCheckNeutrino::visit()` (which begins at `:1048`), and is only
  ever passed by reference into `PatternAlgorithms` calls.  A separate visitor
  running later **could not see it** — only an approximate reconstruction from
  the durable side effects (`ParticleInfo(13,…)` stamped, `kShowerTrajectory` /
  `kShowerTopology` cleared), which would not be exact.

## 5. Round 0 — golden reference (scratch only, nothing enters the toolkit)

**Step 0a — give the sources a durable home.**  They currently exist only in a
session scratchpad, which does not survive the session, and round 0 runs later.
Re-clone into `sbnd_xin/mcs_upstream/` at the **pinned SHAs** in the Repro
block.

**Step 0b — verify the premise.**  Build upstream verbatim
(`make && ./mcs_example`) and confirm the README's number: true 0.735 GeV →
**0.699 GeV**.  *This is unverified today and the entire round-1 acceptance gate
rests on it.*  Explicit branch: **if it does not reproduce in this ROOT
environment, stop.**  The fixture strategy needs rework before any porting
begins, and the discrepancy itself is the first thing to report.

**Step 0c — dump every intermediate** the ROOT-free port must match:

- `trajectory_points_final` after `trim_trajectory` (count + the point list)
- per segment: `segs_distance`, `segs_angle_projB` (θ_xz), `segs_angle_projC`
  (θ_yz), `vx = segs_aAxes[i][0]`, the segment COM and axis
- the likelihood curve `lnlikelihood_track(KE)` sampled on a fixed KE grid
- the three `GetMinimumX` results **and** the three likelihood values behind
  `ambiguity_MCS`

Write them as `mcs/test/data/mcs_reference.json`.  The ROOT file itself is not
committed — a plain-JSON fixture keeps `mcs/` ROOT-free and needs no data
download.  This fixture is the round-1 acceptance gate and becomes
`wcdoctest-mcs`.

**Step 0d — SBND-shaped input too.**  Run the same dump against a handful of
SBND muon clouds harvested offline (a small JSON extraction from an existing
`work-*-prod0823/pr_evt*/` dump), so the port is gated on SBND-shaped input and
not only on one uBooNE event.

## 6. Round 1 — the ROOT-free `mcs/` package

Port `mcs.{h,cxx}` into `mcs/inc/WireCellMcs/MuonMCS.h` + `mcs/src/MuonMCS.cxx`,
changing **only** what ROOT forces.

### 6.1 The three substitutions

| upstream | replacement | risk | tolerance vs round 0 |
|---|---|---|---|
| `TMatrixDEigen` in `fitPCA` (:169) | `Eigen::SelfAdjointEigenSolver<Matrix3d>` on the covariance.  Symmetric ⇒ real eigenvalues.  Eigen returns them **ascending** where ROOT does not sort — order descending explicitly, and fix each eigenvector's sign by a stable rule (largest-\|component\| positive) *before* the muon-direction flip | low | eigenvector components to ~1e-12 |
| `TGraph::Eval` on the 20-point CSDA table (`setUKEfromRR` :437) | a small **local** piecewise-linear interpolator | low, *if* extrapolation is matched | exact to 1e-12 on the fixture grid |
| `TF1::GetMinimumX` (:585-589) | Brent 1-D minimisation reproducing ROOT's recipe: `npx = 100` uniform grid pre-scan to bracket, then Brent to `1e-10` relative | **the real risk** | see 6.2 |

> **The interpolator must linearly EXTRAPOLATE outside its range, not clamp.**
> That is what `TGraph::Eval` does, and `estimate_energy` scans KE from ~0 —
> far below the table's 10 MeV floor — so the out-of-range branch is on the hot
> path, not an edge case.
> `util`'s `WireCell::irrterp` (`util/inc/WireCellUtil/Interpolate.h:73`) is
> **NOT usable**: verified at `Interpolate.h:120-124` it *clamps*
> (`return points.begin()->second` / `points.rbegin()->second`).  Same for
> `Aux::LinterpFunction`.  Write ~20 lines in `MuonMCS.cxx` instead.

### 6.2 Two separate tolerances

`emu_MCS` sits at a broad likelihood minimum and should agree with round 0 to
**< 0.5 %**.

`ambiguity_MCS` is a ratio of likelihoods evaluated at **three separately
located minima**, so a different minimiser relocates all three.  It is expected
to move more.  Acceptance: **< 5 %**, and because the round-0 fixture records
the three minimum locations, a failure can be attributed to *which* of the three
moved rather than just "the number changed".

If `ambiguity_MCS` cannot be held to 5 %, the fallback is to define it on a
fixed KE grid instead of on minimiser output — a deliberate, documented
divergence from upstream, not a silent one.

### 6.3 Determinism

This tree's recurring failure mode, and upstream walks straight into it.
`mcs.h` is built on raw `Point*`: `std::vector<Point*> edges`, a `Comparator`
ordering `Point*` by `score`, and **unstable** `std::sort` in both
`sort_edges()` and `ComparePCAProjection::sort_points`.  Ties then propagate
into index-based segment slicing in `get_seg`/`fitSegPCA`.

Required:

- every `Point` carries its input index as a stable `id` (upstream already does
  this) and **every** comparator breaks ties on `id`;
- `ComparePCAProjection::sort_points` sorts `vector<vector<double>>` with no id
  at all — carry the index alongside the point and tie-break on it, or use
  `std::stable_sort`;
- the Dijkstra `Comparator` compares a double `score` — tie-break on `id`;
- verify by N-run identity under `setarch x86_64 -R` (M4) **plus** a
  shuffled-input test: permuting the input point order must not change
  `emu_MCS`.

### 6.4 Upstream defects to fix in the port

Each documented inline, each gated by the round-0 fixture so the fix is *proven*
not to move any number:

- `trim_trajectory` (:320-326) allocates a `Point` for `i == startpoint_index`
  that is never pushed and never deleted — **a leak per event**.
- `:307-308` computes `sqrt(norm(diff(...)))` where `norm` already takes the
  square root.  Monotonic, so the arg-min is unaffected — **keep the behaviour**,
  note it.  (M7-style: not every divergence is a bug to fix.)
- `mcs.h:121` comments "only keep 4 closest edges" while `nedges_max = 20`.
- `mcs.h:22-49` defines free functions in a header with **no `inline`** — an ODR
  hazard the moment two TUs include it.  Ours will.
- `estimate_energy` reallocates its parameter vector per call; hoist.

**Acceptance for round 1:** `build/mcs/wcdoctest-mcs` passes, reproducing
0.699 GeV and every round-0 intermediate within tolerance.  Nothing outside
`mcs/` has changed, so **no A/B gate is needed yet**.

## 7. Round 2 — the `clus/` driver, default OFF

**Not a new visitor — a knob-gated call site inside `TaggerCheckNeutrino`.**

The two facts in sec 4.2 force this, and it is the better design anyway:
`segments_in_long_muon` (`:1682`) and `fill_kine_tree`'s result (`:2704`) are
both live in the same scope of `TaggerCheckNeutrino::visit()`, so a call placed
immediately after `fill_kine_tree` sees everything all three `muon_source` modes
need — with nothing persisted and no ordering question to answer.

It also removes a byte-identicality hazard a separate visitor would create:
adding `'mcs_visitor'` to `pipeline_names` **changes the compiled JSON even when
the component no-ops**, so the knob-off gate would be testing a config that had
already moved.  With a call site, knob-off is a single early `return` and the
compiled config differs only by the suppressed key.

Following the project's component idiom, `TaggerCheckNeutrino.cxx` gains only
the guarded call; the work lives in a free `static` function in a new
`clus/src/MuonMCSDriver.cxx`:

```cpp
if (m_mcs_enable && final_main_vertex) {
    mcs_fill_kine(kine_info, *pr_graph, segments_in_long_muon,
                  m_mcs_cfg);          // free static fn, MuonMCSDriver.cxx
}
```

which then

1. selects the muon per `muon_source`;
2. harvests the point cloud (all `Segment::fits()` points + vertex fits) and
   converts internal length → cm;
3. calls `Mcs::MuonMCS::run`;
4. fills `kine_info`'s new MCS fields — which reach
   `TrackFitting::set_kine_info` through the **existing** unconditional store
   below the call site, so no new storage path is added.

`TaggerCheckNeutrino.cxx` is a production file with many live consumers.  This
is a purely additive guarded call, **not** a refactor: no existing code is moved
or extracted (M10).

### 7.1 Knobs, all defaulting to the legacy no-op

| knob | default | effect |
|---|---|---|
| `mcs_enable` | `false` | the whole driver is skipped |
| `muon_source` | `"pf_muon"` | selection rule (sec 4.2) |
| `muon_min_length_cm` | `40` | below this, skip.  Upstream's own guards are `2*seg_length = 28 cm` and `npoints ≥ 20` |
| `mcs_seg_length_cm` | `14` | upstream default |
| `mcs_point_source` | `"event"` | `"event"` = whole-event cloud + `trim_trajectory` (ubreco parity); `"segment"` = the muon's own segment fits only |
| `mcs_max_points` | `20000` | perf guard, see 7.2 |

### 7.2 Performance — the largest engineering risk

`trim_trajectory` calls `add_edge` **O(N²)** times and *each* call does a full
`sort_edges()` of up to 20 entries; the Dijkstra loop then `std::sort`s the
**entire** point vector on every iteration, i.e. **O(N² log N)**.

On a uBooNE single-muon cloud (N ~ 10³) this is fine.  An SBND whole-event
`T_rec_charge`-equivalent cloud is 10⁴–10⁵ points, where N² log N is 10⁹–10¹¹
operations — almost certainly prohibitive at production rates.

*Behaviour-preserving* mitigations (same graph, same path, same answer —
provable against the round-0 fixture):

- replace the per-`add_edge` `sort_edges()` with a bounded max-heap of the 20
  best edges: identical edge set, much smaller constant;
- replace the sort-the-whole-vector Dijkstra with a real `std::priority_queue`
  (tie-broken on `id`): O(E log V) instead of O(N² log N);
- restrict the O(N²) neighbour search with a **KD-tree** — `util` already ships
  `KDTree.h` / `NFKDVec.h` / `nanoflann.hpp`, so this is a `use='WireCellUtil'`
  facility, **not a new dependency**.  A radius query at the 20th-nearest
  distance returns the identical edge set.

*Not behaviour-preserving* (must be knobs, must be justified by numbers):

- `mcs_point_source = "segment"` — feeding only the muon's own segment fits.
  This is *better* input, but it is not what the uBooNE resolution terms were
  tuned against, so it stays non-default until sec 9 says otherwise.
- `mcs_max_points` down-sampling.

**Sequencing:** land the behaviour-preserving rewrites **first**, gated on the
round-0 fixture, and *measure* before adding any approximation.  If the
behaviour-preserving set alone brings a worst-case SBND event under ~1 s, the
two approximation knobs stay off and the question closes.

**Acceptance for round 2:** `mcs_enable = false` byte-identical (sec 10);
`mcs_enable = true` completes on ≥1 event with the MCS energy visible in the
log; `./build/clus/wcdoctest-clus` and `./build/mcs/wcdoctest-mcs` pass; timing
on the slowest event quoted.

## 8. Round 3 — the output, knob-gated

`PR::KineInfo` (`clus/inc/WireCellClus/NeutrinoTaggerInfo.h:25-67`) gains

```cpp
float kine_mcs_energy{-1};        // MCS KE [MeV], -1 = not computed
float kine_mcs_ambiguity{-1};     // 1 = most ambiguous, -1 = n/a
float kine_mcs_tracklen{-1};      // MCS's own trimmed path length [cm]
float kine_mcs_range_energy{-1};  // MCS's OWN CSDA range KE [MeV]
int   kine_mcs_segment_id{-1};    // join key: cluster_id*1000 + graph_index
```

### 8.1 Scalars plus a join key — NOT parallel arrays

Forced by the same provenance gap that moved muon selection onto the PR graph
(sec 4.2).  `KineInfo`'s per-particle arrays carry no segment or shower id, so a
driver running *after* `fill_kine_tree` cannot know which row its selected muon
became — a "parallel array" would be aligned only by hope.  And MCS is
inherently **one number for one muon**, not one per particle, so the array shape
was wrong anyway.

`kine_mcs_segment_id` uses the `cluster_id*1000 + graph_index` convention
already used by `PrDisplayDump` (`:459-515`) and the Bee PF tree, so round 4
joins MCS to the corresponding range and dQ/dx numbers **on the segment id**
rather than on row position.  `KineInfo` is already per-bundle, so one scalar
set per bundle is the natural granularity.

*If a later round wants an MCS number on every muon-typed row*, the fill has to
move inside `fill_kine_tree`'s `push_segment_kine` lambda
(`clus/src/NeutrinoKinematics.cxx:~193`), which is where the rows are built and
the segment is still in hand.  That is a second production file and a separate
decision; it is not needed for the owner's stated validation.

### 8.2 Not a new `kine_energy_info` code either

`kine_energy_info` is 0/1/2 = dQdx/range/charge and records *which single
estimator produced `kine_energy_particle[i]`*.  Adding `3 = MCS` would either
silently change which estimator wins for muons — a physics change to
`kine_reco_Enu` — or add a code that never fires.  Neither is wanted.

### 8.3 `kine_mcs_range_energy` is deliberately not called "range"

Upstream's `emu_tracklen` is a range-based energy computed from MCS's **own**
20-point PDG CSDA table over the **trimmed** path.  It is *not* `cal_kine_range`,
which uses the toolkit's `MuonRange` `LinterpFunction` over
`segment_track_length()`.  The two will differ.

Keeping it is worthwhile — it is a same-trajectory control that separates
path-length differences from table differences in round 4 — but it must carry a
name that can never be mistaken for the toolkit's range estimate.

Related: MCS's own CSDA table duplicates the toolkit's `MuonRange` table in
`cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet`.  **Keep MCS's own** —
it is the table the likelihood was tuned against, and substituting the
toolkit's would silently move `emu_MCS`.  Record the numeric discrepancy
between the two (a one-line comparison) when round 4 runs.

### 8.4 Energy, not momentum

The owner asked to "estimate the momentum"; what gets stored is **kinetic energy
in MeV**, matching `kine_energy_particle`'s units and the rest of the tree.
Momentum is one line away for any consumer (`p = sqrt(E² − m²)`, and
`Aux::ParticleInfo` already exposes `momentum_magnitude()`), so storing KE loses
nothing and keeps the schema consistent.  Flagged here so the choice is not read
as an oversight.

### 8.5 Byte-identicality wrinkle

**Adding a `T_kine` branch changes the output schema even with MCS off.**  This
is the one place the "purely additive so it can't break the gate" reasoning
fails.  So the *booking* is gated too:
`root/src/UbooneTaggerOutputVisitor.cxx` books the five branches only when its
own `mcs_output` knob is on — the same shape as the existing `nu_per_bundle`
gating at `:1146` ff.  `cfg/pgrapher/experiment/sbnd/clus.jsonnet` passes both
keys with the key-suppression idiom:

```jsonnet
// ONE source of truth: both keys derive from mcs_enable, so the computation
// gate and the branch gate can never disagree.
[if mcs_enable then 'mcs_enable']: true,   // -> TaggerCheckNeutrino (clus)
[if mcs_enable then 'mcs_output']: true,   // -> UbooneTaggerOutputVisitor (root)
// C++ default false on both.  Keys omitted when off => byte-identical
// pre-MCS config.
```

Deriving both from one `mcs_enable` argument is deliberate: independent knobs
would allow `mcs_output` on with `mcs_enable` off (branches full of `-1`) or the
reverse (computed and discarded).  Neither is useful and both are confusing.

Mirror the same gate in `clus/src/PrDisplayDump.cxx` `dump_kine()` (`:619`) so
the Bee/JSON display gains the fields only when the knob is on.

## 9. Round 4 — validation (the owner's ask)

Compare MCS against range and against dQ/dx→dE/dx, using stopping muons as the
truth proxy.

### 9.1 The reconciliation the study has to make explicit

Range is trustworthy **only** for fully contained stopping muons — and those are
exactly the muons where MCS is least needed.  MCS's value is on **exiting**
muons, where no independent truth exists.  Pretending one sample answers both
questions is how this kind of study goes wrong, so it runs in two parts.

**Part A — calibration on stopping muons.**
Selection: `Facade::Flags::STM` set (`clus/src/TaggerCheckSTM.cxx`), fully
contained, `muon_source` picking a muon of length in a band (say 50–250 cm, so
range is well-measured *and* there are ≥ 3 MCS segments).  For each such muon
compute all three:

- `E_range = cal_kine_range(L, 13, particle_data)`
  (`clus/src/PRSegmentFunctions.cxx:2620`)
- `E_dQdx = segment_cal_kine_dQdx(seg, recomb)` (`:2436`)
- `E_MCS`

Metrics: the fractional-residual distributions `(E_MCS − E_range)/E_range` and
`(E_dQdx − E_range)/E_range` — **bias** (median) and **resolution** (half-IQR /
MAD) — binned in muon length, in `|vx|` (the 5 slices the tune uses), and in
`ambiguity_MCS`.  Plus the direct `E_MCS` vs `E_dQdx` scatter, since those two
are independent of each other.

*Success is a stated number, not "looks reasonable":* MCS bias within **±5 %**
of range and resolution better than **~15 %** in the 100–250 cm band would match
the published MicroBooNE performance and make the tune transfer defensible.
Anything worse is a **retune** signal, not a bug signal — that distinction has
to be drawn before the numbers arrive, not after.

**Part B — MC truth where available.**  On the MC samples the tree already uses,
compare `E_MCS` to the true muon energy directly, for both contained and exiting
muons.  This is the only handle on the exiting population.

### 9.2 Sample

Reuse existing production output rather than reprocessing.  The current baseline
is `prod0823`: `sbnd_xin/work-mcp1k-prod0823/`, `work-mcp2k-prod0823/`,
`work-nuecc48-prod0823/`, `work-ncpi0-prod0823/` (3067 events, all `rc=0`).
Part A draws stopping muons from the `mcp1k`/`mcp2k` MC samples; Part B uses the
same, since they carry MC truth.

Because MCS is purely additive and default-OFF, round 4 needs **one** extra
knob-ON arm rather than a full A/B pair — the OFF arm is the existing production
output.

### 9.3 Tune-transfer sanity check — run FIRST, and cheaply

This can kill or bless the whole transfer before any tuning effort is spent.
All the `res_sigma*` / `par_*` constants are MicroBooNE angular-**resolution**
terms added in quadrature to Highland, and the `|vx|` slicing is
drift-geometry-specific.  Three measurements:

1. **Point spacing.**  Upstream's `get_dist_score` comment pins the scoring
   exponents to uBooNE's ~0.6 cm WCP trajectory spacing.  Measure the actual
   spacing between consecutive `Segment::fits()` points on SBND muons
   (histogram + median).  If SBND is materially different, the `d + d²` score
   and `nedges_max = 20` need re-derivation — a round of its own.
2. **Angular resolution.**  Estimate SBND's directly: on high-energy (≳ 2 GeV,
   near-straight) muons the measured per-segment angle distribution is dominated
   by *resolution* rather than scattering, so its width is a direct read of
   `res_sigma`.  Compare to the uBooNE values.
3. **`|vx|` occupancy.**  Check SBND's `|vx|` distribution is comparable across
   the 5 slices and that no slice is starved.

**Fallback if the transfer fails:** the constants are already fhicl-driven
upstream (`wirecellmcs.fcl`), so they become jsonnet knobs in the style of
`cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet`, with the uBooNE values
as defaults.  Retuning is then a **config** round, not a code round.  This is
the main reason to keep the constants externalized from day one rather than
hard-coding them into `MuonMCS.cxx`.

## 10. Gates

Per `CLAUDE.md` §1/§4, every "no behavior change" claim ships with a gate label:

- **Knob-off byte-identical**, rounds 2 and 3: `/ab-verify` on the standard SBND
  PR manifest, comparing archive **member content hashes** via
  `abtest/hash_archive.py` — never `md5sum`/`cmp` on the tarball (M2).  Report
  the snapshot labels and hash-file paths.
- **Compiled-config proof**: grep the compiled JSON for **both** keys in **both**
  states — `wcsonnet … | grep -E 'mcs_enable|mcs_output'` must be **empty** with
  the knob off and show **both** keys with it on (M6).  Checking only one key,
  or only the ON state, is the doc-77-r2 lesson: a compiled-cfg diff never
  exercises the OFF knobs, so probe each explicitly.
- **Freshness proof** before every A/B: `local/lib/libWireCellClus.so` and
  `libWireCellMcs.so` mtimes newer than the last source edit (M1).
- **Unit tests per touched package**: `./build/mcs/wcdoctest-mcs`,
  `./build/clus/wcdoctest-clus`, `./build/root/wcdoctest-root` — not the
  aggregate `build/wcdoctest`.
- **Determinism**: N-run identity under `setarch x86_64 -R` (M4), plus the
  shuffled-input invariance test (sec 6.3).
- **Exit-code discipline**: `cmd > log 2>&1; echo rc=$?` throughout (M14).

## 11. Open questions for the owner

1. **CMake parity** — add `mcs/CMakeLists.txt` + append to `WCT_PACKAGES`
   (keeps `cmake/test/parity.sh` green), or follow `flash/`'s waf-only
   precedent?
2. **The call site inside `TaggerCheckNeutrino` — confirm the placement.**  The
   design puts MCS immediately after `fill_kine_tree` (`:2704`) rather than in a
   new visitor, because that is the only place all three `muon_source` modes can
   be served (sec 7).  This means touching a heavily-shared production file,
   which the owner may want to review before implementation starts.  The
   alternative — persisting `segments_in_long_muon` so a standalone visitor can
   run later — is a *larger* and more invasive change, not a smaller one.
3. **Per-bundle or per-event?**  `T_kine` fills one row per flash bundle under
   `nu_per_bundle`.  Does MCS run per bundle (one muon per neutrino candidate)
   or once per event on the main candidate?
4. **Cosmic muons.**  The richest MCS validation sample in SBND is cosmics, not
   neutrino candidates — but the PR chain's `KineInfo` is neutrino-centric.
   Does round 4 need a separate cosmic-muon path, or is the in-beam sample
   enough?
5. **Multi-APA / cathode crossers.**  SBND muons crossing the cathode have a
   known position distortion (doc 72/73 family).  Exclude them from Part A, or
   treat them?
6. **Is `mcs_point_source = "segment"` worth pursuing at all**, given it
   invalidates the tune the algorithm ships with?

## 12. Summary of the staging

| round | what | gate |
|---|---|---|
| 0 | scratch ROOT build of upstream; reproduce 0.699 GeV; dump intermediates to a JSON fixture | the number reproduces, or stop |
| 1 | ROOT-free `mcs/` package | `wcdoctest-mcs` vs the fixture; `emu_MCS` < 0.5 %, `ambiguity_MCS` < 5 % |
| 2 | `clus/` driver behind `mcs_enable`, default OFF; perf work | byte-identical OFF; knob-ON smoke + timing |
| 3 | 5 knob-gated `T_kine` branches | byte-identical OFF *including* the schema |
| 4 | validation: stopping muons vs range vs dQ/dx; tune-transfer check | bias ±5 %, resolution ~15 % — or a retune round |

No code ships before round 0's number reproduces, and nothing goes to SBND
production before round 4 says the MicroBooNE tune transfers.

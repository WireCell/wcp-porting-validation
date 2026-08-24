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
  `wscript_build` and no `CMakeLists.txt`.  See sec 11(b).)

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
| `pf_muon` **(default)** | walk the PR graph; take the segment with `abs(particle_info()->pdg()) == 13` and the largest `particle_info()->kinetic_energy()` | reproduces `WireCellMCS_module.cc:197-207` |
| `long_muon` | the `segments_in_long_muon` chain built by `examine_direction` (`clus/src/NeutrinoVertexFinder.cxx:1859-1919`) | the owner's *"a long muon"*; spans segments, closest to a real muon track |
| `longest_segment` | longest `PR::Segment` by `segment_track_length()` above `muon_min_length_cm`, no PID required | PID-independent control arm; catches muons the PF chain mislabels |

All three ship.  One becomes the SBND default after the round-4 comparison.

**`pf_muon` matching (OWNER-DECIDED).**  Match `abs(pdg) == 13`, rank by
`particle_info()->kinetic_energy()`, and **emit a WARN when a muon-typed segment
exists but none is selected.**

This diverges from ubreco, which tests `PdgCode() == 13` *exactly* and ranks by
*total* energy — deliberately, to remove a silent-failure mode: on an exact
match a stamped `−13` would select nothing and MCS would report `−1` with no
error at all.  The ranking change is cosmetic (`E = KE + m` is monotone in `KE`
at fixed mass, so for muons the order is identical).  In practice WCT's
long-muon path stamps `13` (`NeutrinoVertexFinder.cxx:1913`), so the two rules
should agree on every real event — the WARN is there to prove it rather than
assume it.

**Endpoints (OWNER-DECIDED): the endpoint vertices' `fit().point`** — for
`long_muon`, the two extreme vertices of the chain.  Not the segment's
`fits().front()/back()`.

This is also the *more faithful* choice, which is worth stating because it is
not obvious.  ubreco's endpoints were the PF particle's `Position()` /
`EndPosition()`, which were **likewise not members** of the
`T_rec_charge_blob` cloud handed to MCS.  So upstream's structural situation is
exactly reproduced: `trim_trajectory`'s nearest-point search resolves each
endpoint to a neighbouring cloud point, and the true endpoint is then
re-inserted at `mcs.cxx:353-354`.  Using `fits().front()/back()` would have made
the endpoints cloud members and quietly changed that behaviour.

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
| `TF1::GetMinimumX` (:585-589) | Brent 1-D minimisation reproducing ROOT's recipe: `npx = 100` uniform grid pre-scan to bracket, then Brent with `epsilon = 1e-10` as both abs and rel tolerance, `maxiter = 100` | **the real risk** | see 6.2 |

> **Licence note.**  Write the minimiser **from the published algorithm
> description, not from ROOT's source.**  ROOT is LGPL-2.1; WCT is LGPL-3+
> (`LICENSE`).  Upstream `mcs.cxx` is MIT and vendors freely; ROOT's
> `BrentMethods.cxx` does not.  Boost.Math's `brent_find_minima` is already
> available but uses a bits-of-precision criterion and will **not** reproduce
> ROOT's iterates — do not substitute it.  See sec 11(c).

> **The interpolator must linearly EXTRAPOLATE outside its range, not clamp.**
> That is what `TGraph::Eval` does, and `estimate_energy` scans KE from ~0 —
> far below the table's 10 MeV floor — so the out-of-range branch is on the hot
> path, not an edge case.
> `util`'s `WireCell::irrterp` (`util/inc/WireCellUtil/Interpolate.h:73`) is
> **NOT usable**: verified at `Interpolate.h:120-124` it *clamps*
> (`return points.begin()->second` / `points.rbegin()->second`).  Same for
> `Aux::LinterpFunction`.  Write ~20 lines in `MuonMCS.cxx` instead.

### 6.2 Acceptance — gate the mechanism, not the output number

The governing principle, and the most useful sentence in this section:

> **Match the expression form, not just the mathematics.**  A last-bit
> difference in the covariance matrix moves an eigenvector at 1e-15, flips one
> point across a segment boundary, and changes that segment's angle by O(mrad).
> That is the mechanism by which 1 ULP becomes 100 MeV.

Concretely: `fitPCA` accumulates the covariance with the triple loop
`covData[3*i+j] += meanPoints[k][i]*meanPoints[k][j]`, then `/= track.N`.
**Keep that loop verbatim** — do not write `X.transpose()*X`, whose blocked and
vectorised accumulation order differs.  Same for `MCSHelper::norm`
(`sqrt(Σ pow(v[i],2))`, in that order).

**The gates, in dependency order.**  Each one is only meaningful if the one
above it passed:

| # | quantity | acceptance |
|---|---|---|
| 1 | `Interp1D` vs the round-0 side probes (incl. below- and above-range) | **bitwise `==`** |
| 2 | number of segments, and point count **per segment** | **identical** |
| 3 | the pre-scan bracket triple, for **each** of the 3 `GetMinimumX` calls | **identical** |
| 4 | `vx[k]` and the resulting `ivx[k]` bin | `< 1e-12`, **identical bin** |
| 5 | `θ_xz[k]`, `θ_yz[k]` | `< 1e-9` rad |
| 6 | `keguess`, `keguess_lower`, `keguess_higher` (each separately) | `< 1e-3` MeV |
| 7 | `emu_MCS` | `< 0.5 %` (implied by 6) |

**Gate #2 and #3 are FAILs, not tolerances.**  Gate 2 is what a 1e-15
eigenvector wobble breaks — if it fails you have a segment-boundary flip; chase
it, do not loosen it.  Gate 3 matters because the objective is **multimodal**
(that is the entire reason `ambiguity_MCS` exists): with `npx = 100` over
`[0, 4000]` the grid spacing is 40 MeV, so a different pre-scan lands in a
**different basin** — an O(100 MeV) error.  Once the bracket matches, both
implementations converge to the same true minimum.

**Do NOT put a tolerance on `ambiguity_MCS` itself.**  It is
`max(exp(lnl − lnl_lo), exp(lnl − lnl_hi))` — an *exponential* of a likelihood
difference — so a basin change moves it by orders of magnitude while `emu_MCS`
is untouched.  A loose tolerance on the ratio hides exactly the failure it is
meant to catch.  Gate the three minimum *locations* independently (#6); then
`ambiguity_MCS` is a deterministic function of already-gated quantities and can
be asserted at 1e-9 relative as a derived check.

### 6.3 Determinism

**Correcting the obvious first diagnosis, because it selects the wrong test.**
`mcs.h` is built on raw `Point*`, which looks like this tree's classic
address-dependence bug (M4).  It is not.  `Comparator::operator()` reads
`exhausted` and `score`; `compare_edges` reads the distance;
`ComparePCAProjection`'s lambda reads `dot(a,axis)`.  **None of them
dereferences a pointer *value*.**

So `mcs.cxx` is order-**deterministic** but order-**sensitive**: the same input
sequence gives the same answer every run, and ties are broken by position in the
input vector.

*Therefore an N-run repeat-identity test passes even with every tie-break
missing.*  It is the wrong test.  **The right test is a shuffle test**: permute
the input point vector by K fixed permutations and require byte-identical
output.  Run it *alongside* `setarch x86_64 -R` (M4), not instead of it.

The exposure is real because **unstable** `std::sort` is used in both
`sort_edges()` and `ComparePCAProjection::sort_points`, and ties propagate into
index-based segment slicing in `get_seg`/`fitSegPCA`.

Required, in order of leverage:

1. **`ComparePCAProjection::sort_points` — highest leverage.**  It sorts
   `vector<vector<double>>` with no id at all, and `get_seg` slices a *prefix of
   that order*, so a tie flip changes segment membership → different PCA →
   different angle → different energy.  Exact ties are plausible: 0.6 cm regular
   spacing with a segment axis near a coordinate axis gives exactly-equal
   projections.  → sort key `(dot(p,axis), p.index)`.
2. **`add_edge`'s 20-nearest retention.**  Mirror-symmetric neighbours on a
   regular grid give exactly-equal `d + d²`.  → `compare_edges` key
   `(dist, p->id)`.
3. **`Comparator`.**  Add `id` as secondary key among non-exhausted points.  Two
   payoffs: permutation-independence, and it makes a lazy binary heap *provably*
   equivalent to the full re-sort — which is what licenses the 7.2 perf rewrite.
4. **`Track::remove_seg`'s exact-float coordinate match** → match by index
   (bug 6.4/#4-6 below).
5. **Input-assembly audit (clus side).**  `Segment::fits()` is a `std::vector`
   and `segments_in_long_muon` is index-ordered, so `muon_segments` is
   deterministic by construction.  `whole_event` is the only mode needing an
   audit for pointer-keyed iteration (§2 Code: never iterate `std::set<T*>`).

Verification: `doctest_mcs_shuffle` with 20 fixed permutations of the round-0
cloud, then the same on ≥3 real SBND clouds; **plus** `setarch x86_64 -R` × 5 in
the round 3/4 gates.  Round 0 also runs one pre-shuffled pass (sec 5), so we
learn *which* ties actually fire on real data before writing the port.

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

**Further defects found on a full read** (numbered for the round-2 checklist).
Several are *silent* — they publish a wrong number rather than failing:

7. **`nsegs == 1` publishes a garbage energy with no flag — the nastiest.**
   `seg_vec.push_back(Track())` runs *before* `fitSegPCA`
   (`form_segs`, mcs.cxx:406-412), so `if (!can_fit) break;` leaves an empty
   `Track` behind and `segs.size()` is **one longer** than
   `segs_distance`/`segs_angle_*`.  The guard at `:111` tests `segs.size() < 2`,
   so a single *real* segment passes it.  Then `estimate_energy` packs
   `par[0] = 1`, and `lnlikelihood_track`'s loop `for (i=2; i<nsegs+1; i++)`
   **never executes** → the objective is identically 0 → Brent minimises a
   constant → `keguess` is whatever the grid scan returns, `ambiguity = 1`, and
   `emu_MCS` is published as if it were real.
   *Fix:* require `segs_distance.size() >= 2` (≥1 real angle), else return −1.
8. **`|vx| >= 1` falls into `ivx = 0`.**  `mcs.cxx:520`'s five indicator terms
   cover `[0,1)` only, so `vx_abs == 1.0` (segment exactly along drift) — or
   `1.0 + ε` from rounding — makes every term false and selects `ivx = 0`, the
   *most-perpendicular* bin, whose `res_sigma1_yz = 0.0449` is 4× the correct
   bin's.  More reachable on SBND than uBooNE.  *Fix:* clamp `ivx = 4` for
   `vx_abs >= 0.75`.
9. **NaN singularity for drift-parallel segments.**  `mcs.cxx:151-154`:
   `vecy_plane = cross(aAxis_prior, x̂)` is the zero vector when
   `aAxis_prior ∥ x̂`; normalising gives NaN → `-log(NaN)`.  Also `:526,537`,
   where `probability` can underflow to 0 → `-log(0) = +inf`.  Interacts with
   #8: a drift-parallel track hits both.
10. **`get_angle` can return NaN** from `acos(>1)` (`:139`) — clamp to [−1,1].
11. **`cleanUp()` leaks both `TGraph`s and then bricks the object** (`:54-58`):
    sets the pointers to `nullptr` *without* `delete`, and any later `run()`
    dereferences null (`:85`).  In WCT the tables are compile-time constants —
    make them `static const` and delete `cleanUp()` entirely.
12. **`emu_tracklen` is published on `bad_path` events.**  `rr_path` is computed
    *before* the early return (`:83-95`), and on `bad_path`
    `trajectory_points_final` holds only the two vertices, so `rr_path` is the
    straight-line chord — published as a track length.  *Fix:* set both to −1.
13. **`uEnergy[]` is mislabelled.**  `:441` calls it "Total particle energy in
    MeV", but a 10 MeV *total* energy is impossible for a 105.658 MeV muon and
    `:87`/`:126` add `Mmu` to it.  It is **kinetic** energy (cross-checked
    against PDG: KE 10 MeV ↔ CSDA 0.9833 g/cm²).  Fix the comment.
14. **`fitPCA` is wrong for non-unit weights** (`:187-196`): weights enter the
    covariance *squared*, and the normalisation divides by `track.N` rather than
    `total_weight`.  **Latent only** — nothing ever passes a weight ≠ 1.
    *Note it; do NOT "fix" it* — that would break the golden (M15).
15. **`Track::remove_seg` can erase the wrong points.**  `mcs.h:164-186`: the
    segment is chosen as a prefix along `segFitVector[0]`, then `remove_seg`
    erases `seg.N` **contiguous** entries from `track`'s `aAxis` order.  When the
    two axes differ — the entire point of the refit — the segment's points are
    not contiguous there.  Related: `first_index == -1` gives UB
    (`weights[-1]`, `erase(begin()-1, …)`), and identity is matched by
    **exact float coordinate equality**, which mis-matches duplicate points.
    *Fix all three by tracking indices.*  Behaviour-changing → own knob + gate.
16. **`-1` as a sentinel for a signed angle** (`:417`) — safe only because
    `lnlikelihood_track` starts at `i = 2`.  Fragile; use NaN or an explicit
    count.
17. **Dead code to drop from the port:** `axes[1]`/`axes[2]` (`:381-382`) and
    `bAxis_prior`/`cAxis_prior` (`:147-148`) are assigned and never used;
    `beta()`, `gamma()`, `increment_energy()`, `decrement_dist()` are never
    called; **`setUKEfromEX` is declared (`mcs.h:224`) and never defined** — a
    link error if anything ever references it.
18. **A stale duplicate tune lives in the art module.**
    `WireCellMCS_module.cc:388-421` carries a commented-out earlier
    `lnlikelihood_theta_yz` with different `emu_edges` and a
    `probability = pvx[ivx]; //debugging` override.  **The standalone
    `mcs.cxx` is the source of truth** — port its live version (sigmoid
    blending, `emu_edges = {600,950,1300}`).

Bugs 7–12 and 15 change behaviour relative to upstream.  Ship each behind its
own `McsOptions` bool so its effect is separately gate-able.

**Default: FIXED (OWNER-DECIDED).**  The guards are **on** by default, with a
round-1 gate demonstrating that each fires *only* where upstream produced NaN,
inf, or the `nsegs == 1` garbage.  Flipping the bools off reproduces upstream
exactly, so the round-0 golden reference stays reachable and the divergence
stays auditable.

The rationale is worth recording, because it inverts this tree's usual default:
normally an unproven change ships OFF.  Here the "unchanged" path is the one
that publishes a **known-garbage number as if it were a real measurement** —
`emu_MCS` from a minimised constant, or a `res_sigma` 4× too wide from the
`|vx| ≥ 1` bin fallthrough.  A documented divergence from upstream is strictly
safer than that, so the burden of proof flips.

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
| `mcs_endpoints` | `"vertex"` | endpoint vertices' `fit().point` (sec 4.2) |
| `mcs_beam_window_only` | `true` | run only on bundles in coincidence with the beam spill — a **correctness** requirement, sec 7.4 |
| `mcs_point_source` | `"muon_segments"` | `"muon_segments"` = the selected muon's own `Segment::fits()` points (N ~ 10²–10³); `"whole_event"` = ubreco-literal whole-event cloud (N ~ 5e4, **validation only**) |
| `mcs_max_points` | `20000` | perf guard, see 7.2 |

> **`seg_length` is deliberately absent from the knob table.**  14 cm is one
> radiation length of liquid argon: X0(LAr)/rho = 19.55 / 1.396 = **14.004 cm**.
> That is *why* the bare Highland term `sigmaH = 13.6(T+M)/(T(T+2M))` carries no
> `sqrt(l/X0)(1 + 0.038 ln(l/X0))` factor — the factor is exactly 1 at l = X0.
> Changing `seg_length` silently invalidates the prefactor **and** every `par_*`
> quartic fitted on top of it.  It is a structural constant, not a tunable — and
> it is a property of **argon**, not of MicroBooNE, so it transfers to SBND
> unchanged.

### 7.2 Performance — the largest engineering risk

`trim_trajectory` calls `add_edge` **O(N²)** times and *each* call does a full
`sort_edges()` of up to 20 entries; the Dijkstra loop then `std::sort`s the
**entire** point vector on every iteration, i.e. **O(N² log N)**.

**Measured, not estimated.**  SBND nueCC evt 168596:

| cloud | source | N | O(N²) `add_edge` calls |
|---|---|---|---|
| whole-event live | `mabc-pr.zip: 0-clustering-global.json` | **50 087** | 2.5e9 → **~10 min/muon/event** |
| PR fit points (= `T_rec_charge`) | `0-track_fit-global.json` | **867** | 7.5e5 → **< 10 ms** |

Four orders of magnitude.  Each `add_edge` call additionally passes
`std::vector<double>` **by value** through `diff`/`norm`, so it costs 2–3 heap
allocations even for the ~50 % rejected by the `in_dir` test; the Dijkstra loop
adds ~4e10 comparisons on top.  Whole-event is prohibitive — confirmed, not
suspected.

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

*Not behaviour-preserving* (knob + its own gate):

- k-d tree for the 20-nearest-in-direction retention.  An *exactly* equivalent
  version exists (expanding-k kNN until 20 in-direction candidates are found,
  since `d + d²` is monotone in `d`); a fixed-k approximation is not.  Build the
  exact one or none.
- `mcs_max_points` down-sampling.

### 7.3 Why `muon_segments` is the default

`trim_trajectory` exists **only** because ubreco had no muon isolation: it had
to hand MCS the whole `T_rec_charge_blob` cloud and let a shortest-path hack
find the muon inside it.  WCT's PR graph already solved that problem, properly.
Re-solving it with a Dijkstra over 50 k blob points, to rediscover a
`PR::Segment` we are already holding, is not ubreco *fidelity* — it is ubreco
*scar tissue*.

Supporting evidence: **the shipped reference cloud is itself a 456-point,
0.600 cm-spaced fitted trajectory**, i.e. upstream's own demonstration input
already has the `muon_segments` shape.  `trim_trajectory` is close to an
identity map on it.

`trim_trajectory` still *runs* in `muon_segments` mode, so the code path and the
`bad_path` semantics stay identical to upstream — it just runs on N ~ 10²–10³.

**Consequence: with `muon_segments` as the default, rounds 1–4 need no
performance work at all.**  Stated explicitly so nobody funds a k-d-tree round
for a mode that may never run in production.  `whole_event` is retained for the
sec 9.4 parity check, run under `mcs_max_points` with a WARN.

**Acceptance for round 2:** `mcs_enable = false` byte-identical (sec 10);
`mcs_enable = true` completes on ≥1 event with the MCS energy visible in the
log; `./build/clus/wcdoctest-clus` and `./build/mcs/wcdoctest-mcs` pass; timing
on the slowest event quoted.

### 7.4 Granularity: per bundle, beam-window only — and WHY (OWNER-DECIDED)

MCS runs **once per bundle**, restricted to bundles **in coincidence with the
beam spill**.

The restriction is a **correctness requirement, not a cost bound.**  Outside the
spill the full readout window is not guaranteed to cover the activity, so the
track can be **truncated** — and a truncated track silently corrupts *both*
estimators that round 4 compares:

- `mu_tracklen` is short ⇒ the range-based energy is wrong, so the truth proxy
  is wrong;
- the trajectory ends where the readout ends rather than where the muon does
  ⇒ the terminal segments are spurious and their scattering angles are
  meaningless.

Neither failure announces itself: both produce a plausible-looking number.  So
this gate must be applied at the *driver*, not left to a downstream cut, and it
is `true` by default.

**Corollary for validation (sec 9): cosmic muons ARE usable — if they are
beam-spill-coincident.**  The selecting criterion is *spill coincidence*, not
*neutrino-candidate-ness*.  A cosmic crossing during the spill has a full
readout window and is perfectly good calibration material — and cosmics are by
far the richest source of long, often-stopping muons in SBND.  Out-of-spill
bundles are skipped regardless of what tagged them.

This also matches the existing beam-window tagger gate (doc 56), so the
machinery to express it already exists.

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

`clus/src/PrDisplayDump.cxx` `dump_kine()` needs **no change and no gate** —
see sec 10.2: it enumerates fields explicitly, so new `KineInfo` members are
inert for `calib-pr-evt<N>.json`.

## 9. Round 4 — validation (the owner's ask)

Compare MCS against range and against dQ/dx→dE/dx, using stopping muons as the
truth proxy.

### 9.1 The reconciliation the study has to make explicit

Range is trustworthy **only** for fully contained stopping muons — and those are
exactly the muons where MCS is least needed.  MCS's value is on **exiting**
muons, where no independent truth exists.  Pretending one sample answers both
questions is how this kind of study goes wrong, so it runs in two parts.

**Part A — calibration on stopping muons.**
Selection (OWNER-DECIDED):

- **in coincidence with the beam spill** — the governing criterion, *not*
  neutrino-candidate-ness.  Out-of-spill bundles may lack the full readout
  window, which truncates the track and corrupts range and MCS together
  (sec 7.4).  **Cosmic-tagged bundles that are spill-coincident are included**,
  and are expected to supply most of the calibration statistics: cosmics are
  SBND's richest source of long, often-stopping muons.
- `Facade::Flags::STM` set (`clus/src/TaggerCheckSTM.cxx`), fully contained
- muon length in a band (say 50–250 cm, so range is well-measured *and* there
  are ≥ 3 MCS segments)
- **cathode crossers excluded** — see the note below

> **Cathode crossers: excluded from Part A; recommendation, not yet
> owner-confirmed.**  They are the longest tracks available, which is tempting,
> but they carry a known position distortion (doc 72/73 family) that feeds
> *directly* into the per-segment scattering angles.  Since the pull-test width
> is the single number deciding whether the uBooNE tune transfers, contaminating
> it risks attributing a reconstruction artifact to MCS.  Revisit them as their
> own study once the tune is validated.

For each selected muon compute all three:

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

**Part B — exiting muons, with a free hard inequality.**  No absolute reference
exists for exiting muons, but **range on the *visible* track is a strict lower
bound on the true energy**.  So `E_MCS > E_range(visible length)` **must hold
event by event** — violations are a real failure mode (basin collapse, `ivx`
misassignment, NaN) and the check costs nothing.  Report the distribution of
`E_MCS / E_range,visible` vs visible length; it should rise systematically with
the exiting fraction.  On MC, additionally compare `E_MCS` to the true muon
energy directly for both populations.

**Part C — validate `ambiguity_MCS` itself**, or it ships with no acceptance
criterion at all.  On Part A's sample, plot median
`|(E_MCS − E_range)/E_range|` in bins of `ambiguity_MCS`.  *Success:* monotone
rise, with the top ambiguity decile at ≥2× the residual of the bottom decile.
**If flat, the score is noise — say so, and do not publish it as a quality
flag.**

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

#### What transfers for free — measured, no work needed

1. **`seg_length` = X0(LAr) = 14.004 cm** (sec 7.1) — argon physics, not uBooNE
   geometry.
2. **Point spacing: ANSWERED, no measurement required.**  Upstream's
   `get_dist_score` comment pins the exponents to uBooNE's ~0.6 cm WCP
   trajectory spacing.  SBND's fitted-trajectory density **already equals it**:
   `clus/src/TrackFitting.cxx:8856` sets `low_dis_limit = 0.6*units::cm`, and
   the shipped reference cloud measures **0.600 cm median NN spacing** over 456
   points.  ~23 points per 14 cm segment on both detectors.
3. **The `|vx|` slicing survives SBND's two-TPC geometry** because the code uses
   `std::abs(vx)` (`mcs.cxx:519`).  Both detectors drift along x.  (Still worth
   confirming no slice is starved — one histogram.)

#### What does NOT transfer for free — and it is worse than it looks

The `res_sigma*` terms add in quadrature to Highland.  Their relative weight:

| KE (MeV) | `sigmaH` (rad) | res share of θ_xz variance | res share of θ_yz variance (ivx=0) |
|---|---|---|---|
| 200 | 0.0505 | 1.3 % | 44 % |
| 500 | 0.0232 | 5.9 % | 79 % |
| 1000 | 0.0124 | 18 % | 93 % |
| 1500 | 0.00851 | 32 % | 97 % |
| 2000 | 0.00648 | 44 % | 98 % |

(`res_sigma1_xz = 0.005776`, `res_sigma1_yz[0] = 0.0449`.)

**The θ_yz channel is resolution-dominated above ~230 MeV**, and half the
likelihood comes from it.  Reading `setSegAngles` (`mcs.cxx:151-154`) explains
why: `vecy_plane = â × x̂`, so `theta_xz` measures scattering **in the plane
containing the drift axis** (measured by drift time — precise) and `theta_yz`
measures scattering **in the wire plane** (imprecise).  That 8× resolution gap
*is* the geometry-specific part.  SBND's wire layout is close to uBooNE's, but
drift length (200 vs 256 cm) and field (0.5 vs 0.273 kV/cm) — hence diffusion —
differ.

#### The measurement: a pull test.  One plot, tests the whole tune at once

On contained stopping muons (`Flags::STM`), using range as the truth proxy:

1. per segment k, `T_k = cal_kine_range(residual length from segment midpoint to
   track end, 13, particle_data)`;
2. `σ_pred,xz(T_k)` from `pred_theta_xz_pars(T_k)`, likewise yz for the
   segment's `ivx`;
3. histogram the **pulls** `θ_xz,k / σ_pred,xz` and `θ_yz,k / σ_pred,yz`.

If the uBooNE tune transfers, the pull core width is **1.00**.  That single
number exercises Highland + the quartic modifiers + the resolution terms + the
`ivx` slicing simultaneously.

Then slice by predicted `T` (200/400/800/1500 MeV) and by `ivx`.  The slicing
says *which* term is wrong, because their T-dependences differ: a wrong
resolution term gives a pull width that **grows with T** (resolution is
T-independent while `sigmaH` falls as 1/p), a wrong Highland modifier gives one
that is flat-in-T or quartic-shaped.

**Complementary intercept fit** (if the pull test fails): per `ivx` bin, fit
measured θ RMS² vs `sigmaH(T)²`.  Slope tests the Highland modifier; the
**intercept is `res_sigma²` directly**.  That yields SBND's numbers without
touching the 60+ tune parameters.

#### Fallback ladder — this is what makes "the transfer failed" fundable

- pull width within ~20–30 % of 1 → **ship as-is**, capping quoted MCS validity
  at ~1.5 GeV;
- 1.3–2× → **refit twelve numbers** (`res_sigma{1,2}_xz` + `res_sigma{1,2}_yz[0..4]`)
  via the intercept fit, shipped as an SBND tune variant behind a knob with the
  uBooNE tune retained as default so the round-1 golden gate never breaks;
- \> 2×, or strongly T-dependent → **do not tune to make it look right** (§5.7).
  Report it as a *trajectory-fitting-quality* finding, not an MCS finding: a 2×
  angular-resolution deficit is a statement about `TrackFitting`.

#### Validated window — state it up front

MCS needs ≥2 real segments (≥28 cm) to produce anything and ≥5 (~70 cm ≈
200 MeV by range) to be meaningful; the resolution crossover caps the top.
**~200 MeV – 1.5 GeV.**  Publishing that range prevents someone scanning the
tails and reporting an out-of-scope failure.

**Fallback if the transfer fails:** the constants are already fhicl-driven
upstream (`wirecellmcs.fcl`), so they become jsonnet knobs in the style of
`cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet`, with the uBooNE values
as defaults.  Retuning is then a **config** round, not a code round.  This is
the main reason to keep the constants externalized from day one rather than
hard-coding them into `MuonMCS.cxx`.

## 10. Gates

Per `CLAUDE.md` §1/§4, every "no behavior change" claim ships with a gate label:

### 10.1 BLOCKING — the existing PR gate does not cover the artifact we change

`sbnd_xin/scripts/pr85_hash_gate.py::archives_of()` (`:34-40`) compares **only**
`mabc-pr.zip` and `pctree-pr-evt<N>.tar.gz`.  It never opens
`tracking-pr.root`.  Verified:

```python
for name in ("mabc-pr.zip", "pctree-pr-evt%d.tar.gz" % evt):
```

| artifact | written by | affected by this change? | gated today |
|---|---|---|---|
| `clusters-apa-*.tar.gz` | imaging | no | `abtest/ab_compare.sh` ✅ |
| `mabc-pr.zip` | MABC | no | `pr85_hash_gate.py` ✅ |
| `pctree-pr-evt<N>.tar.gz` | pctree writer | no | `pr85_hash_gate.py` ✅ |
| `calib-pr-evt<N>.json` | `PrDisplayDump` | no (see 10.2) | **NOTHING** ⚠ |
| `tracking-pr.root` → `T_kine` | `UbooneTaggerOutputVisitor` | **YES** | **NOTHING** ⚠ |

**A "byte-identical PASS" from `pr85_hash_gate.py` would be vacuous for exactly
the artifact round 3 touches.**  This must be fixed before round 3's gate means
anything.

**New gate helper, `sbnd_xin/scripts/mcs_root_gate.py`** (uproot):

1. tree list identical (`Trun, T_bad_ch, T_proj, T_proj_data, T_rec_charge,
   T_tagger, T_kine`);
2. per tree: branch **name list** identical, `num_entries` identical;
3. per branch: array-level `np.array_equal`, with an explicit NaN-equal path;
4. `--expect-new kine_mcs_energy,…` mode: knob-**on** must differ **only** by
   the named branches, every pre-existing branch bit-identical.

### 10.2 `PrDisplayDump` is inert — do not touch it in round 3

`dump_kine()` enumerates fields **explicitly**
(`clus/src/PrDisplayDump.cxx:626-652`: `out["kine_reco_Enu"] = …` etc.), so new
`KineInfo` members do **not** appear in `calib-pr-evt<N>.json`.  Adding them
there is a separate, later, knob-gated change with its own hash check.  (This
supersedes an earlier draft of this doc that called for mirroring the gate into
`dump_kine()` in round 3 — unnecessary, and it would have created an ungated
diff.)

### 10.3 The gates

- **Knob-off byte-identical**, rounds 2 and 3: `/ab-verify` on the standard SBND
  PR manifest, comparing archive **member content hashes** via
  `abtest/hash_archive.py` — never `md5sum`/`cmp` on the tarball (M2) — **plus
  `mcs_root_gate.py` for `tracking-pr.root`** (10.1).  Report the snapshot
  labels and hash-file paths.
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

## 11. Decisions (owner, 2026-08-24) and what remains open

Eight decisions taken; three items remain.

| # | question | decision |
|---|---|---|
| 1 | `mcs_point_source` default | **`muon_segments`** — sec 7.3.  `whole_event` retained as validation-only |
| 2 | driver wiring | **knob-gated call site in `TaggerCheckNeutrino::visit()`** after `fill_kine_tree` (:2704) — sec 7.  Not a new visitor |
| 3 | granularity | **per bundle, beam-window only** — and the restriction is a *correctness* requirement, sec 7.4 |
| 4 | bug-guard defaults | **FIXED by default**, each behind its own `McsOptions` bool, difference gated — sec 6.4 |
| 5 | muon endpoints | **endpoint vertices' `fit().point`** (chain extremes for `long_muon`) — sec 4.2.  Also the more ubreco-faithful choice |
| 6 | `pf_muon` matching | **`abs(pdg)==13`, rank by KE, WARN if none selected** — sec 4.2 |
| 7 | validation populations | **beam-spill-coincident bundles, cosmic-tagged included** — sec 9.1.  Out-of-spill skipped |
| 8 | `mcs_root_gate.py` location | **`sbnd_xin/scripts/`** (SBND-only), beside `pr85_hash_gate.py` |

### Still open

**a. Cathode crossers in Part A.**  This doc *recommends excluding* them from
the calibration sample (sec 9.1) — their known position distortion feeds
straight into the scattering angles, and the pull-test width is the number that
decides the tune transfer.  Not yet owner-confirmed.

**b. CMake parity** — *proposed: add it.*  Ship `mcs/CMakeLists.txt`
(`wct_package(WireCellMcs USE WireCellUtil)`) **and** append `mcs` to
`WCT_PACKAGES` at `CMakeLists.txt:191`.  `cmake/test/parity.sh` treats the
installed-library list as a hard check, so a waf-only package makes it fail.
`flash/` is waf-only today, but that reads as pre-existing debt rather than a
precedent worth extending.  Cost is two lines.  Override if you would rather
keep `mcs/` out of the CMake build until it has proven itself.

**c. ROOT licence posture** — *proposed: re-implement, and there is really only
one lawful option.*  Write `MinimStep`/`MinimBrent` from the published algorithm
description (sec 6.1), not from ROOT's source.  ROOT is **LGPL-2.1**; WCT is
**LGPL-3+**.  Absent an "or later" grant, LGPL-2.1 source cannot be combined
into an LGPL-3+ work, so vendoring `BrentMethods.cxx` is not available to us
regardless of preference.  Upstream `mcs.cxx` is MIT and vendors freely.
Boost.Math's `brent_find_minima` is already a configured dependency but uses a
bits-of-precision stopping criterion and will **not** reproduce ROOT's iterates
— it cannot substitute (sec 6.2 gate #3).

## 12. Summary of the staging

| round | what | gate |
|---|---|---|
| 0 | scratch ROOT build of upstream; reproduce 0.699 GeV; dump intermediates + side probes to a JSON fixture; one shuffled pass | the number reproduces, **or stop** |
| 1 | ROOT-free `mcs/` package (`Interp1D`, `Minimize1D`, then the physics) | `wcdoctest-mcs` vs the fixture, gates #1–#7 of sec 6.2; shuffle test; `setarch -R` ×5 |
| 2 | `clus/` call site in `TaggerCheckNeutrino` behind `mcs_enable`, default OFF; `muon_segments`, vertex endpoints, per-bundle beam-window-only | byte-identical OFF (archives **+ `mcs_root_gate.py`**); knob-ON smoke + timing.  **No perf work needed** (sec 7.3) |
| 3 | 5 knob-gated `T_kine` branches | byte-identical OFF *including the schema*; `--expect-new` diff shows only the new branches |
| 4 | validation on beam-spill-coincident bundles (cosmics included): pull test, then Parts A/B/C | pull core width ≈ 1; bias ±5 %, resolution ~15 % — or the sec 9.3 fallback ladder |

Two hard stops:

- **No code ships before round 0's 0.699 GeV reproduces.**
- **`sbnd_xin/scripts/mcs_root_gate.py` must exist before round 3's gate means
  anything** — the existing `pr85_hash_gate.py` does not open
  `tracking-pr.root`, so a PASS from it would be vacuous for the one artifact
  this change touches (sec 10.1).

Nothing goes to SBND production before round 4 says the MicroBooNE tune
transfers — and if it does not, sec 9.3's ladder says refit twelve resolution
constants, **not** tune until the plot looks right (§5.7).

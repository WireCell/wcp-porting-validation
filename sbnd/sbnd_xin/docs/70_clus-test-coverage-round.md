# doc 70 — clus/ test coverage round: what is worth a test, and what is not

**Status:** SHIPPED, toolkit `065cfc89` (unpushed). Test-only; no production
file touched, so **no A/B gate is implicated**.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild
./build/clus/wcdoctest-clus   > /home/xqian/tmp/clus_full.log 2>&1; echo rc=$?
./build/gen/wcdoctest-gen     > /home/xqian/tmp/gen_full.log  2>&1; echo rc=$?

# a single case (note -tc, NOT -sf -- see sec. 5)
./build/clus/wcdoctest-clus -tc="pr11 dpc self-append via cloud identity is a no-op"

# the revert-proof harness
<scratchpad>/revert_proof.sh      # guards 1,3,5 (whole-file reverts)
<scratchpad>/revert_proof4.sh     # guard 4 (surgical)
```

## 1. Why this round exists

Upstream master `e53af02c` added a policy requiring that new code ship with
tests and that bug fixes be reproduced by a failing test first. `apply-pointcloud`
is 67 commits / 56 files / +7746−543 ahead of master and added exactly **one**
test file (`gen/test/doctest_powerbox_recombination.cxx`). Before opening a PR
we need a defensible answer to "where are the tests".

The answer is **not** blanket coverage. Most of the branch is heuristic
pattern-recognition logic, where a unit test pins an arbitrary event through an
arbitrary code path and breaks on every legitimate retune — worse than no test.
The answer is a deliberate split saying which verification method covers which
class of change, plus tests for the two things the A/B gates structurally
cannot see.

### Correction to the record

A previous note claimed "43 changed `clus/` sources, **0** clus tests". That was
wrong. `clus/test/` already held **12** doctests (49 cases / 565 assertions, all
passing) which the branch did not extend, and the branch added 1 test in `gen/`.

## 2. The verification split (the deliverable for the PR body)

| Class of change | Verification | Status |
|---|---|---|
| Heuristic event-level logic — taggers, PID policy, merge/split decisions (~20 files) | **byte-identical A/B gates on real events** | DONE: SBND PR chain 141/144 with **all 48 `mabc-pr.zip` + all 48 `pctree-pr-*.tar.gz` identical**; `fixedinput_gate.sh` PDHD/PDVD **178/178**; 16/16 configs compile, 11/11 PDHD+PDVD identical |
| Knob legacy defaults (~50 knobs) | doctest — `doctest_clus_knob_defaults.cxx` | **NEW** |
| Crash/UB guards (doc pr/11 audit) | doctest — `doctest_pr11_guards.cxx` | **NEW** |
| Pure numeric kernels | doctest — `gen/test/doctest_powerbox_recombination.cxx` | pre-existing on branch |
| SBND production operating point | compiled-config proof (`wcsonnet` + `abtest/cmp_cfg.sh`) | DONE, doc 68 |

For the heuristic row the gate is not a weaker substitute for a unit test — it
is a **stronger** check. It compares whole reconstruction outputs on real
events at member-content granularity, which no hand-written fixture approaches.

The PR should say plainly that these are **regression tests pinning fixes
already validated by byte-identical gates**, not test-first development. That is
true and sufficient; implying test-first was followed is not.

## 3. `doctest_pr11_guards.cxx` — 10 cases, all revert-proven

Regression pins for `clus/docs/audits/pr11-latent-pattern-audit.md`. Every case
reaches its guard through public API with no data fixture.

**Revert-proof**: revert the guard to `origin/master`, rebuild, confirm the case
fails; restore, confirm it passes.

| Guard | Production file | Without the guard |
|---|---|---|
| DPC self-append, **cloud** identity (`this == &other`) | `DynamicPointCloud.cxx` | `std::bad_alloc`; `npoints` 8 → 16, then 17 |
| DPC self-append, **batch** identity (`&m_pts == &points`) | `DynamicPointCloud.cxx` | hang, process killed (SIGTERM) |
| Cluster all-points-excluded extremes fallback | `Facade_Cluster.cxx` | throws `No valid points available for get_main_axis_points` / `..._two_extreme_points` |
| `Shower::add_segment` `clone_dpc` | `PRShower.cxx` | shower DPC **is** the segment's object (same pointer) |
| `inside_dead_region` sentinel `apa`/`face` | `FiducialUtils.cxx` | **SIGSEGV** |
| `Dataset::size()` vs `size_major()` | — | **no revert applies**, see below |

The `size()`/`size_major()` case pins a `PointCloud::Dataset` **API trap** —
`size()` returns `m_store.size()`, the number of *arrays*, so a point-less
steiner cloud with three zero-length coordinate arrays is not `size()==0` —
rather than a call site. Reverting `MyFCN.cxx` / `TrackFitting.cxx` cannot
change it, so it is labelled honestly as a documentation test.

Guard 4 needed a **surgical** revert: `git checkout origin/master -- PRShower.cxx
PRShower.h` does not build, because `PRShower.h`'s 2-line signature change
(`get_stem_dQ_dx`, `update_particle_type`) is consumed by
`NeutrinoTaggerNuE.cxx` and `NeutrinoTaggerSinglePhoton.cxx`. The proof instead
restores just the pre-fix aliasing line in `add_segment`'s fit-seeding branch.
Note this trips the pointer-identity assertion only — the `!= seg_dpc` test
still short-circuits the merge, so idempotence survives the partial revert.

## 4. `doctest_clus_knob_defaults.cxx` — 11 cases, 178 assertions

Pins the legacy default of every knob the port added. The point: a silently
flipped default would make every "no behavior change" claim false **while every
A/B gate still passed**, because the gate compares two runs of the same
already-wrong default. A failure here is a feature — it forces the CLAUDE.md §5
stop-and-ask.

Covered: `TaggerCheckNeutrino` (~40 keys — booleans all OFF, uBooNE literals,
and the vector knobs `kine_plane_weights` / `ssm_target_dir` /
`ssm_absorber_dir` / `muon_dqdx_curve` pinned for length **and** content),
`CreateSteinerGraph` (`replace` defaults **true** — the one inverted case, which
a "make all new knobs default false" sweep would break), `ClusteringUnmergeBundle`,
`ClusteringProtectBundle`, `TaggerCheck{FC,STM,TGM}`, plus the `TrackPidOptions`
and `KineChargeOptions` in-class initializers (these never pass through
`default_configuration()`, so no factory round-trip can see them) and the
`muon_dqdx_cut` vs `muon_dqdx_cut_cm` bit-identity claim.

**Proven non-vacuous**: flipping `m_proton_dir_vote` to `true` fails the suite
(`CHECK( true == false )`).

**The instance name is load-bearing.** `defaults_of()` looks up a privately
named instance (`"knobdefaults_probe"`), not the default-named one.
`Factory::lookup` caches one instance per `(type, name)` for the whole process,
and `clus/test/data/uboone-mabc_config.json` instantiates **`CreateSteinerGraph`
and `TaggerCheckNeutrino` with no `"name"`** — the two most important components
here — which `doctest_pattern_recognition.cxx`'s `configure_components()` then
calls `configure()` on. Sharing that instance would have made this file read
back configured values in whatever order doctest registered the cases, i.e. it
would have silently stopped testing defaults while still passing.

**Not covered: the cm ↔ `units` crossing.** The two headers state the same knobs
in different frames — `TaggerCheckNeutrino.h` has `m_mip_dqdx{50000.0}` and
`m_iso_endpoint_min_length{40}`, while `NeutrinoPatternBase.h` has
`{50000/units::cm}` (= 5000) and `{40 * units::cm}` (= 400). This file pins both
*endpoints* correctly in their own frame, but **not the scaling `configure()`
applies when threading tagger → `PatternAlgorithms`** — which is precisely the
family commit `1628328e` ("x10 dQ/dx unit divergences") fixed. Reading a green
run as "the unit family is protected" would be wrong; only its endpoints are.

The file header states in as many words that it pins the **C++ default only** —
SBND flips many of these ON in `sbnd/wct-pr-perevt.jsonnet` (doc 68), and that
operating point is gated by the compiled-config proof, not by doctest. Without
that sentence a green suite reads as "production is on the legacy path".

## 5. GOTCHAS

- **`-sf` is doctest's SOURCE-FILE filter; the test-case-name filter is `-tc`.**
  The first revert-proof run used `-sf="*dpc self-append*"`, matched nothing, ran
  **zero** cases, exited 0, and reported a bogus PASS for all five guards. Any
  harness branching on a doctest exit code must **also** assert the executed case
  count is non-zero. This is the same family as M14 (judging through a pipe):
  the exit code was real, it just answered a different question.
- **Factory lookup fails inside `wcdoctest-clus` without `PluginManager`.** The
  clus lib is linked, but the `WIRECELL_FACTORY` registrars live in translation
  units nothing in the test references, so the linker drops them and
  `Factory::lookup` throws `No factory for class`. Call
  `PluginManager::instance().add("WireCellClus")` first; it resolves to the
  already-mapped `build/clus/` copy, so there is no stale-install exposure. This
  also explains why `doctest_pattern_recognition.cxx`'s `configure_components()`
  wraps every lookup in a try/catch that logs "Skipping" — those lookups are
  silently failing today.
- **A revert-proof in the primary tree is safe iff `wcb install` is never run.**
  `wcdoctest-*` links `build/` libs while `wire-cell` loads `local/lib` (M1), so
  skipping install means no other session's *runtime* can observe a reverted
  guard. Restore from an EXIT trap. A fresh git worktree was the first plan but
  needs a full 889-file rebuild: ccache is installed but **not** wired in
  (`CXX = /usr/bin/g++`), so nothing is reused.
- **Cluster fixtures need one blob per point.** `get_two_extreme_points()`
  smooths each extreme with `calc_ave_pos(p, 5 cm)`, which averages **blob
  centers, not points**. A single blob holding every point collapses both
  extremes onto one centroid, and the test then passes or fails for the wrong
  reason. Each blob needs a `"3d"` PC (the default scope is
  `{"3d", {"x","y","z"}}`) *and* the one-row `"scalar"` PC that
  `Blob::fill_cache` requires — otherwise it raises
  `scalar PC is not scalar but size 0`.
- `Cluster::set_excluded_points()` exists and is documented in the header as a
  test hook; use it rather than driving `connect_graph.cxx`.
- **`Factory::lookup` caches one instance per `(type, name)`** — a defaults test
  that asks for the default-named instance is reading whatever any other test in
  the same binary configured. Always use a private probe name. See §4.
- **`revert_proof.sh`'s guard-4 block is known to `BUILD_FAILED`** by design;
  the working proof for that guard is `revert_proof4.sh`. The script says so
  inline, but expect the failure when running it end to end.
- The `conflicting types for '_mm_prefetch'` marker on
  `doctest_clus_knob_defaults.cxx:29` is **clangd's** index, not the compiler:
  `g++` builds clean (`build rc=0`) and there is **no CI** in this repo
  (`.github/workflows/` does not exist), so nothing compiles it with clang.

## 6. Mentioned, not fixed (pre-existing; §5 tie-breaker)

- **`ClusteringExamineBundles` implements `configure()` but not
  `default_configuration()`**, so none of its knobs round-trip — not
  `save_bundle_main_provenance` (ours) and not upstream's own `use_ctpc`,
  `graph_name`, `use_flash_t0`, `flags_from_longest`. Its defaults are invisible
  to a config dump. The branch is compliant with "round-trip new keys *when the
  component has one*"; the test case documents the gap instead of asserting a
  key that cannot exist.
- **`FiducialUtils::FiducialUtils()` is declared but never defined** anywhere in
  the tree. Construct from `StaticData{}`.

## 7. Result

| Suite | Before | After |
|---|---|---|
| `build/clus/wcdoctest-clus` | 49 cases / 565 assertions, 1 skipped | **70 cases / 782 assertions**, 0 failed, 1 skipped |
| `build/gen/wcdoctest-gen` | 8 cases / 131 assertions | unchanged 8 / 131, 0 failed |
| `build/wcdoctest` (aggregate) | — | **465 cases / 173164 assertions, 0 failed**, 1 skipped |

The aggregate row matters for the PR: this tree's manual prescribes the
per-package binaries, but upstream's `wct-testing` skill prescribes
`build/wcdoctest`, which is what a reviewer will run. Both are green. Build is
waf (`wcbuild`); there is **no CI** in the repo, so no cmake path was exercised
— if a reviewer configures with cmake, that is the one unverified axis.

`git status` shows no modified tracked file — only the two additions — so no
reconstruction output can have moved.

## 8. Not covered (deferred by owner, 2026-08-03)

A third file of pure numeric kernels was scoped and deferred:
`do_track_comp(..., skip_stop_samples)` 0-vs-1 (needs the `ParticleDataSet`
fixture plus the two-pass configure, so it would skip where
`clus/test/data/uboone-mabc_config.json` is absent — a local check, not a
portable gate). The `muon_dqdx_cut` half of that file was folded into
`doctest_clus_knob_defaults.cxx` instead.

Explicitly **not** testable without a production refactor that CLAUDE.md §2
(fork by duplication, M10) forbids: `collinear_deg` (file-local, 3 deliberate
copies), `alloc_ident` (2 copies), `in_beam`, `cathode_band_closest`, and
`ChanScheme::global` (`private:` in `SbndPrMagnifyTrackingVisitor.h`).

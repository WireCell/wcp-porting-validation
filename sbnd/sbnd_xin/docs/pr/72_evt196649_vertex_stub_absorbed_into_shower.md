# doc pr/72 — evt 18255-196649: near-vertex stub track absorbed into the electron shower

(renumbered from a draft "pr/67" — that number landed on a different,
concurrently-pushed investigation in the shared doc set; no relation.)

## Status: TOPOLOGY FIXED + SBND PRODUCTION ON (owner flip 2026-08-12, toolkit
fc9d1fcb); near-vertex GEOMETRY chain-audited round 3, `es3sg_vertex_fit`
implemented and measured NEGATIVE, ships default OFF, not recommended.
`es3_stub_guard` ships with C++ default OFF; the SBND cfg entry point
(`wct-pr-perevt.jsonnet`) flips it ON, cfg-only, per the owner's review of
the Bee before/after pair below. Round 1 traced the cause; round 2 censused
it across 117 events and designed/validated the geometry/topology cut; the
flip landed the same session after the owner reviewed the result. Round 3
(same session, owner follow-up: "the trajectory near the vertex is still a
bit bended") traced the chain's vertex-fitting machinery, implemented the
most targeted fix at the owner's request, and found it measurably worsens
(not improves) the near-vertex deflection — see `## Round 3` below.

## Repro block

```
cd wcp-porting-img/sbnd/sbnd_xin
PR_JOBS=1 SBND_WCT_LOGLEVEL=trace ./run_pr_chain_batch.sh work-nuecc48-cb0805 <out> data 196649
grep "examine_structure_3\|examine_structure:" <out>/pr_evt196649/wct_pr_evt196649.log
```
Owner report: run 18255 evt 196649, Bee set
`9e2a1a1e-b637-4be5-99b7-f49bf8c04c57` event 21 — "Missing the small track ided
near vertex, lumped into the electron."

## Method note: isolated worktree, not the shared tree

A concurrent session was actively mid-edit on `clus/` + `cfg/` (adding a
`traj_cover_probe` feature touching the exact same file,
`NeutrinoStructureExaminer.cxx`, plus `NeutrinoPatternBase.{h,cxx}`,
`TaggerCheckNeutrino.{h,cxx}`, `TrackFitting.{h,cxx}`) at the moment this trace
was needed. Per M9, I did not touch their files or rebuild/run against the
shared `local/lib`. Instead: `git worktree add` pinned at my last clean commit
(`c955ca52`), configured with its own install prefix (reusing the shared
third-party deps — boost/root/jsonnet/spdlog — read-only via
`--boost-libs`/`--with-jsonnet-libs`/`PKG_CONFIG_PATH`), built fully isolated
(~3 min cold build, 64 cores), and run with `PATH`/`LD_LIBRARY_PATH` pointed at
that isolated install. Zero contact with the shared tree in either direction.

## The event, restated

- Neutrino candidate `main_id=11` (nu-candidate, contained, in-beam).
- Reco vertex (`kine_nu_{x,y,z}_corr`) = (-53.72, -21.51, 193.68) cm.
- Every raw charge point within 20cm of the vertex belongs to one PF segment,
  `real_cluster_id 11052`, 35.5cm, 68 fit points — the dominant electron
  (1611/1617 MeV of reco Enu). No separate segment exists near the vertex in
  the final output.

## What actually happened (trace-confirmed)

**`examine_structure_3`** (`clus/src/NeutrinoStructureExaminer.cxx:389-503`,
called unconditionally — no knob — from `NeutrinoPatternBase.cxx:2596`,
gated only on `is_main_cluster`, run once per cluster immediately after
`find_other_segments`'s round loop finishes and BEFORE `determine_main_vertex`
or the DL-vertex rerank ever run) merged the genuine near-vertex stub into the
shower trunk. Trace line (coordinates added via a temporary, reverted probe
edit for this round — not shipped):

```
doc pr67 vtxprobe examine_structure_3: Cluster: 11 Merge two segments into
one according to angle 16.449694918303653° (10cm) and 20.993370162424846°
(3cm) at vtx=(-50.57,-20.89,197.89) vtx1=(-43.60,-20.81,227.32)
vtx2=(-53.61,-21.93,193.27)
```

`vtx2 = (-53.61,-21.93,193.27)` is the true main vertex (0.5cm from the final
`kine_nu_corr` position — the residual is pre- vs post- later refinement).
`vtx = (-50.57,-20.89,197.89)` is the shared junction that gets deleted.
`vtx1 = (-43.60,-20.81,227.32)` is the far end of the long piece.

So the two segments merged were:
- **the stub**: `vtx → vtx2`, **5.63 cm** long, running from the junction
  straight to the true vertex — this is the "small track" the owner sees.
- **the trunk**: `vtx1 → vtx`, 30.24 cm long — the rest of the shower.

`examine_structure_3`'s merge criterion (`NeutrinoStructureExaminer.cxx:429-449`):
a degree-2 junction vertex qualifies if the bulk direction agreement over a
**10cm window** is within 18° of collinear (gate), and — only if that passes —
the direction agreement over a **3cm window** is within 27° of collinear
(commit). Here: 16.45° (10cm) and 20.99° (3cm) — both comfortably inside the
lenient thresholds, so the merge fires and replaces the two segments with one
new straight-line segment from `vtx1` directly to `vtx2`, discarding the
5.6cm stub's own identity entirely (`remove_segment(sg1); remove_segment(sg2);
remove_vertex(vtx); add_segment(new straight segment, vtx1, vtx2);`).

Confirmed this is a genuine, one-time event for this cluster: `grep
examine_structure_3\|examine_structure_2\|examine_structure_final` for
cluster 11 across the whole trace log shows exactly one `examine_structure_3`
hit (this one) and one unrelated `examine_structure_final_1` hit 55cm away at
a different vertex. Downstream, `improve_vertex: cluster 11 fitting vertex
(-53.61, -21.93, 193.27) nsegs=1` (main-vertex-finding stage) confirms the
vertex has exactly one segment attached by the time it is selected as main —
consistent with the merge having already erased the stub upstream.

## What this corrects from the earlier (pre-trace) draft

The original hypothesis in this investigation named `examine_structure_final_1p`
(from doc pr/64 round 8's `assoc_clear_on_merge` site) as the likely
mechanism, based on analytically replicating `segment_search_kink`'s formula
against the *final*, fully-merged/re-fit trajectory and finding a sharp kink
1.29cm from the vertex. That mechanism did **not** fire near this vertex at
all (confirmed: zero `examine_structure_final_1p` hits for cluster 11
anywhere in the trace; the one `examine_structure_final_1` hit is 55cm away).
The real mechanism is `examine_structure_3` — an earlier, separate,
more lenient (18°/27° vs. `_final_1p`'s implicit ≤5° "collinear" bar) function
that runs at a different pipeline stage (right after `find_other_segments`,
before main-vertex determination) and was not on the original candidate list
built from static code reading. The 1.29cm kink found analytically was real
in the final geometry, but it is not diagnostic of *this* merge — it's a
downstream artifact of the same absorbed-stub geometry, viewed through the
final re-fit rather than the original wcpt-level segments.

**Lesson for next time (recorded, not any process change)**: for this class
of bug, static/analytical replication of a scoring formula against the final
output is a reasonable way to *notice* something is off, but is not a
substitute for a trace confirmation before naming a specific function as the
culprit — exactly why the owner asked for the trace before any fix
discussion.

## Files touched this round

- This doc only (renumbered from a draft "67" to "72"). No shipped code
  changes. The coordinate-probe edit to
  `clus/src/NeutrinoStructureExaminer.cxx:449` existed only in the isolated
  worktree used to produce the trace above and was never applied to the
  shared tree.

## Round 2 — census, cut design, 117-event validation, `es3_stub_guard` SHIPPED

### Repro block

```bash
# isolated worktree, pinned at clean efef4535 (M9: a concurrent session was
# mid-edit on this exact file at round-2 start; it committed before round 2
# finished, so the diff below was applied straight onto the shared tree —
# see "Method note" below)
git worktree add /path/to/wt-pr72 efef4535
cd /path/to/wt-pr72
./wcb configure --prefix=$PWD/local --with-cuda=/usr/local/cuda-12.5 \
  --with-cuda-lib=/usr/local/cuda-12.5/lib64 \
  --with-libtorch=/nfs/data/1/xqian/toolkit-dev/libtorch-shim --with-root=yes \
  --boost-mt --boost-libs=/nfs/data/1/xqian/toolkit-dev/local/lib \
  --boost-include=/nfs/data/1/xqian/toolkit-dev/local/include \
  --with-jsonnet-libs=gojsonnet --build-debug="-O2 -ggdb3" \
  --with-spdlog-active-level=trace
./wcb build -j64 --notests -p && ./wcb install --notests -p

# census run (log-only, env-gated) over the 117-event blast radius
cd wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)
WCT_ES3_MERGE_CENSUS=1 PR_JOBS=16 ./run_pr_chain_batch_isolated72.sh work-nuecc48-cb0805 work-pr72-cen48 data
WCT_ES3_MERGE_CENSUS=1 PR_JOBS=16 ./run_pr_chain_batch_isolated72.sh work-ncpi0-cb0805   work-pr72-cen19 data
WCT_ES3_MERGE_CENSUS=1 PR_JOBS=16 ./run_pr_chain_batch_isolated72.sh work-mcp1k-cb0805   work-pr72-cen50 data $M50
python3 scripts/analysis/pr72/es3_census.py --sample nuecc48=work-pr72-cen48 \
  --sample ncpi0=work-pr72-cen19 --sample mcp1k50=work-pr72-cen50 -o /home/xqian/tmp/pr72-analysis/es3_census
python3 scripts/analysis/pr72/es3_analysis.py   # offline: V5, V5b, population, grid scan, residual list

# off/on validation over the same 117 events
PR_JOBS=32 ./run_pr_chain_batch_isolated72.sh work-nuecc48-cb0805 work-pr72-base48 data   # pristine (stash the knob first)
PR_JOBS=32 ./run_pr_chain_batch_isolated72.sh work-nuecc48-cb0805 work-pr72-off48 data    # knob code, off
SBND_ES3_STUB_GUARD=true PR_JOBS=16 ./run_pr_chain_batch_isolated72.sh work-nuecc48-cb0805 work-pr72-on48 data
python3 scripts/analysis/pr49/on_compare.py work-pr72-off48 work-pr72-on48
# (same triplet for ncpi0/19 and mcp1k50/50)
```

### Method note — the concurrent-session pin, and two new worktree gotchas

At round-2 start the shared tree was mid-edit by a concurrent session (doc
pr/67 round 2, the P5/P6 census work — coincidentally the same filename,
`NeutrinoStructureExaminer.cxx`, unrelated content). Per M9, the entire round
ran in a fresh `git worktree` pinned at the last commit both trees agreed
on (`efef4535`), never touching the shared `clus/`, `cfg/`, or `local/lib`
while that session was live. By the time round 2's implementation was done,
the concurrent session had committed and the shared tree's HEAD was, again,
exactly `efef4535` with a clean `git status` — so the worktree's full diff
(11 files) applied onto the shared tree with `git apply` cleanly, no manual
reconciliation needed.

Two gotchas surfaced this round, on top of doc pr/67 round 1's already-fixed
runner-script PATH issue:

1. **Interactive-shell `LD_LIBRARY_PATH` mirage.** Running the worktree's own
   `./build/clus/wcdoctest-clus` *directly* from this session's shell (not
   through the isolated runner script) hung for 12+ minutes with RSS climbing
   ~7 MB/s — looked exactly like an infinite loop introduced by the new code.
   Root cause: this shell's `LD_LIBRARY_PATH` (from `toolkit-dev`'s direnv)
   lists the **shared tree's** `build/clus` before the worktree's own path, so
   the dynamic loader resolved `libWireCellClus.so` to the shared tree's
   *stale, mid-edit* library — an ABI mismatch against the worktree's own
   modified headers (new struct members on `PatternAlgorithms` and
   `TaggerCheckNeutrino`), which is classic undefined behavior. Confirmed via
   `ldd` (resolved to the shared tree's path) and by reproducing: the SAME
   binary, run with `env -u LD_LIBRARY_PATH` so the binary's own embedded
   `RUNPATH` takes over, passed 180/180 in seconds. **Rule for next time: any
   *manual*, ad hoc invocation of a worktree binary from an interactive shell
   under this repo's direnv needs `env -u LD_LIBRARY_PATH` (or an explicit
   override) — the isolated runner script already does this for batch runs,
   but a one-off `./wcdoctest-clus` or `ldd` in the main session shell does
   not.**
2. **Waf link-order quirk for a brand-new symbol, both in the worktree AND
   the shared tree.** `./wcb build` (and even `./wcb install` without
   `--notests`) intermittently fails to link `wcdoctest-clus` with
   `undefined reference to es3_stub_suppress(...)` even though `nm -D` on the
   freshly-built `libWireCellClus.so` shows the symbol present — this
   reproduced in the shared tree too (not just the worktree-reusing-shared-
   deps setup round 1 suspected), so it looks like a general waf/ld quirk on
   this system with brand-new symbols rather than a worktree-specific
   artifact. `./wcb install --notests -p` always installs the *library*
   correctly regardless (freshness-proofed via `nm -D` + mtime each time);
   only the *test executable* link is flaky. Workaround used throughout:
   extract the exact failing link command from `./wcb build -v` and re-run it
   with local `-L<pkg>` flags moved first (script:
   `pr72_relink_test.py`, scratch dir) — or simply retry `./wcb build`, which
   sometimes succeeds outright on a later attempt (it did, unprompted, on the
   shared tree). Neither the worktree's nor the shared tree's *library*
   install was ever affected — this is purely a test-binary link nuisance.

### Census design

Log-only, env-gated (`WCT_ES3_MERGE_CENSUS`) instrumentation added to
`examine_structure_3` (`clus/src/NeutrinoStructureExaminer.cxx`): three tagged
blocks (`ES3PB`, `ES3CENSUS`, `ES3MERGE`) emit a full geometry/topology/charge
feature vector for every degree-2 junction the function evaluates — merged and
declined alike — via `SPDLOG_LOGGER_DEBUG`, duplicating (not calling)
`segment_cal_dir_3vector`'s and `segment_median_dQ_dx`'s arithmetic so point
counts and raw centroid distances are visible, not just a bare angle. No-op
when the env var is unset: confirmed by member-content hash identity
(`hash_archive.py`) of `mabc-pr.zip` + `pctree-pr-evt196649.tar.gz` with the
census on vs. off.

Parser: `wcp-porting-img/sbnd/sbnd_xin/scripts/analysis/pr72/es3_census.py`.
Offline analysis (deg2 check, independence check, population report, grid
scan, residual list): `.../scripts/analysis/pr72/es3_analysis.py`.

**Population** (117 events: 48 nueCC + 19 NC π⁰ + 50 PR data):

| sample | events w/ census rows | junctions (rows, all sweeps) | merges | pb_skips |
|---|---|---|---|---|
| nuecc48 | 46 | 349 | 20 | 1 |
| ncpi0   | 17 | 104 | 6  | 1 |
| mcp1k50 | 30 | 62  | 10 | 0 |

- 515 total census rows; 3 dropped by the offline parser as **known-corrupted**
  (a rare, ~0.6% log-writer interleaving race truncates one very long
  `ES3CENSUS` line mid-write when another thread's log line lands in the
  middle — confirmed by raw log inspection, not a bug in the census logic
  itself; the corresponding `ES3MERGE` line for the same junction, a separate
  shorter write, was intact in all 3 cases). Excluded from all counts below.
- **Terminal-sweep-only** junctions (the ES3 sweep loop re-scans and
  re-emits a declined junction once per sweep; only the last, no-merge sweep
  enumerates every surviving junction once): 438 of 515.
- **Degeneracy population** (`nfit < 2` on either arm — would divide
  `len_long/len_short` by zero or near-zero without the guard's floor):
  29/515.
- Merges-per-event distribution (30 events with ≥1 merge): mean 1.2, median 1,
  max 2.
- Total actual merges with an intact census row: 35.

### V5 — confirms round 1's `deg(vtx2) == 1` inference

```
evt196649 clus=11 vtx=(-50.57,-20.89,197.89) len_short=6.28 len_long=33.19
deg_short=1 deg_long=5 ang10=16.450 ang3=20.993 predmerge=1
```
`deg_short == 1`: the stub's far end is genuinely a free terminus (a real
candidate vertex), not a mid-chain junction. Round 1's inference from a
downstream `improve_vertex … nsegs=1` line is directly confirmed. (`len_short`
here is the raw wcpt-level census figure at ES3 time, 6.28cm — close to but
not identical to round 1's 5.63cm, which was measured on the final re-fit
trajectory; the recovered on-arm segment in production measures 5.67cm, see
below.)

### V5b — is `ang3 > ang10` a real signal or a lever-arm artifact?

Among the 181 junctions with `len_short < 8cm && len_long/len_short > 3`,
`ang3 > ang10` in 93/181 = **51.4%** — close to a coin flip, not the ~90% that
would mean the ratio term is mechanically implied by the length asymmetry
alone. **Conclusion: the `ang3 > ang_ratio·ang10` term carries real,
independent separating power and stays in the predicate** — it is not
decorative.

### Grid scan and the chosen operating point

480 grid points evaluated over `(stub_max, len_ratio, ang3_min, ang_ratio,
require_terminal)`; 384 keep 196649 suppressed. Scored against
`n_events_touched` (first-order: exact for the first suppressed decision per
cluster, approximate afterwards since a suppressed sweep lets the scan
continue). **Chosen point** (first row on the sorted-by-`n_events_touched`
grid, arbitrary tie-break among equals since every tied row scores 1 event):

| stub_max | len_ratio | ang3_min | ang_ratio | require_terminal | n_suppress | n_events_touched | keeps 196649 |
|---|---|---|---|---|---|---|---|
| 7 cm | 2.0 | 15° | 1.0 | true | 1 | 1 | yes |

At the census level this operating point suppresses **exactly one** junction
in the entire 117-event, 512-row (post-corruption-drop) population: 196649's
own target. **Residual (misses) list — junctions the census flags as
suspicious (`merge=1 && ang3>15 && deg_far==1`) that this operating point
still lets through: 0 rows** (`es3_residual.tsv`, empty). This is why only one
Bee set exists below — there is nothing in the second population to show.

These chosen numeric values (7cm / 2.0) differ from the placeholders used
while first implementing the knob (8cm / 3.0, chosen by eye from the single
196649 measurement before the census existed) — **the C++ member defaults
were updated to the fitted operating point** (`stub_max{7*units::cm}`,
`len_ratio{2.0}` in `PRSegmentFunctions.h`/`NeutrinoPatternBase.h`/
`TaggerCheckNeutrino.h`; `ang3_min=15°`, `ang_ratio=1.0`,
`require_terminal=true` were already correct), since the fit is
SBND-specific and there is no other detector using this knob to conflict —
per CLAUDE.md's "member holds the safe default, knob overrides" idiom, the
cfg TLAs (`es3sg_stub_max` etc.) stay `null`/suppressed and simply inherit
this default; no SBND-cfg-level override is needed or shipped.

### The predicate (shipped)

`WireCell::Clus::PR::es3_stub_suppress` — pure free function,
`clus/inc/WireCellClus/PRSegmentFunctions.h:654-680`, implementation
`clus/src/PRSegmentFunctions.cxx:4040-4051`:

```
suppress the merge when ALL hold:
  nfit_short >= 2  AND  nfit_long >= 2                  <- degeneracy guard
  len_short  > 0.5 cm                                    <- degeneracy guard
  len_short                < 7 cm    (stub_max)
  len_long / len_short     > 2.0     (len_ratio)
  ang3                     > 15 deg  (ang3_min)
  ang3                     > 1.0 * ang10                 (ang_ratio)
  deg(far end of short arm) == 1                         (require_terminal)
```
Short/long assigned by **comparing `len1`,`len2`**, not by graph edge order.
Guarded early-out in `examine_structure_3`,
`clus/src/NeutrinoStructureExaminer.cxx:663` — the first statement inside
`if (angle_3cm < 27)`, above the existing merge TRACE; `continue`s to the next
node when `m_es3_stub_guard && es3_stub_suppress(...)`. Lines above and below
are textually untouched, so the off path (`m_es3_stub_guard` defaults
`false`) is byte-identical by construction (M10-style guard, not an edit to
the angle arithmetic).

Knob `es3_stub_guard` (bool, C++ default `false`) plumbed through
`clus/inc/WireCellClus/{NeutrinoPatternBase,TaggerCheckNeutrino}.h`,
`clus/src/TaggerCheckNeutrino.cxx` (`configure`/`default_configuration`/copy
into `PatternAlgorithms`), `cfg/pgrapher/common/clus.jsonnet` (key-suppression
idiom), `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (4 sites: `clus_pr(...)`
+ `pr(...)` defaults and their two forwarding call sites),
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (top-level TLA + forward
to `clus_maker.pr(...)`), and `wcp-porting-img/sbnd/sbnd_xin/run_pr_chain_batch.sh`
(`SBND_ES3_STUB_GUARD` → bare `--tla-code` pass-through, plus 5 sub-parameter
overrides, matching `SBND_ASSOC_CLEAR_ON_MERGE`'s contract).

**Tests** (`clus/test/doctest_prsegment.cxx`, 4 new `TEST_CASE`s): the 196649
shape suppresses with shipped defaults; a non-terminal short arm (deg 2, the
real near-miss pattern seen at evt235435/423981 in the census) is NOT
suppressed, and toggling `require_terminal=false` proves the switch acts in
both directions; degenerate arms (`nfit<2` on either side, or below the 0.5cm
floor) never suppress; all 4 numeric thresholds tested at strict-inequality
boundaries (7.01cm fails / 6.99cm passes, etc.). Plus
`clus/test/doctest_clus_knob_defaults.cxx`: 6 `CHECK_KNOB_*` assertions
pinning the shipped defaults (`false`/`7.0`/`2.0`/`15.0`/`1.0`/`true`).
`./build/clus/wcdoctest-clus`: **180/180 passed, 1880/1880 assertions
passed** (shared tree, post-apply).

### Off-gate — byte-identical, 0/117

`abtest/hash_archive.py` member-content hashes of `mabc-pr.zip` +
`pctree-pr-evt<ID>.tar.gz`, pristine binary (`work-pr72-base{48,19,50}`) vs.
knob-code binary with `es3_stub_guard` at its off default
(`work-pr72-off{48,19,50}`):

```
work-pr72-base48 vs work-pr72-off48: 0/48 differ
work-pr72-base19 vs work-pr72-off19: 0/19 differ
work-pr72-base50 vs work-pr72-off50: 0/50 differ
TOTAL: 0/117
```

**Compiled-config proof** (M6, `wcsonnet` on the bare `wct-pr-perevt.jsonnet`
with the production `pipeline_names` TLA set —
`scripts/cfg/compile_prjob_cfg.sh`): `es3_stub_guard` absent from the
compiled JSON with the knob off; `"es3_stub_guard" : true` present with
`--tla-code es3_stub_guard=true`; numeric sub-keys stay absent (null,
key-suppressed) in both cases since no override was passed, correctly
inheriting the C++ default.

### On-arm result — 1/117 movers, zero selection flips

`scripts/analysis/pr49/on_compare.py` on `work-pr72-off*` vs.
`work-pr72-on*` (`SBND_ES3_STUB_GUARD=true`, shipped defaults, no numeric
overrides):

| sample | archive-level movers | nusel-events.tsv diffs | nusel-table.tsv diffs |
|---|---|---|---|
| nuecc48 (48) | **1** (evt 196649) | 0/48 | 0/48 |
| ncpi0 (19)   | 0 | 0/19 | 0/19 |
| mcp1k50 (50) | 0 | 0/50 | 0/50 |
| **TOTAL**    | **1/117** | **0/117** | **0/117** |

This matches the census-level grid-scan prediction (`n_suppress=1,
n_events_touched=1`) exactly — the on-arm result is not merely within the
strict ≤5/117 bar, it is at the minimum possible nonzero value.

**196649 mover detail** (`mabc-pr.zip`, cluster 11 only differs; all other
clusters and all other events byte-identical):
- `track_fit-global.json`: cluster 11 point count 675 → 678 (+3); a new
  `real_cluster_id` (11028) carries a **5.67cm**, 10-fit-point segment running
  from `(-50.5735, -20.895, 197.895)` — the exact junction vertex the census
  independently measured (`vtx=(-50.57,-20.89,197.89)`) — to
  `(-53.5009, -21.4557, 193.681)`. This is the recovered near-vertex stub
  (round 1's "5.63cm" figure, measured on the final re-fit; the census's
  6.28cm at wcpt level; production's 5.67cm on the final tracking fit — three
  consistent measurements of the same recovered object at three different
  pipeline stages).
- `vertices-global.json`: net +2 vertices (85→87) at/near the recovered
  junction, including the exact `(-50.57,-20.89,197.9)` point.
- `0-mc.json` (the PR particle-flow jsTree, Bee's per-particle summary — **not
  MC truth** despite the filename): one lumped `e- 1611 MeV` object off →
  split into `e- 1610 MeV` (the shower, ~unchanged) plus a small additional
  PF object on. (This file shows Bee-display-level endpoint labels, which are
  not literally the segment's own start/end fit points — the `track_fit`
  arc-length figures above are the authoritative geometry.)
- **Bundle-level tagger/selection verdict is byte-identical**: `nusel-table.tsv`
  row for the main-vertex bundle (id 11, `npts_main=10830`, `flash_pe=43396.9`,
  `tgm=0 stm=0 fc=1 stmfit=contained`) and `nusel-events.tsv` (`nu-candidate`)
  match off vs. on exactly. This is the evidence for the "reco vertex does not
  degrade" acceptance criterion: FC (fiducial containment) and STM
  (through-going/stopping-muon) are both vertex-position-sensitive boundary
  checks, and neither moved — if the main vertex had shifted by more than a
  fraction of a cm, at least one of these boundary-sensitive flags would be a
  plausible casualty. Combined with the recovered segment sitting exactly at
  the same graph-topological point the pre-fix code already had as `vtx2`
  (just deleted, not relocated), this satisfies the round's vertex-quality
  bar without needing to re-derive an explicit truth-vertex distance.

### Accept/stop decision: **ACCEPT**

Against the strict bar (≤5/117 move, zero selection flips, no other event's
vertex moves >2cm): **1/117 moved, 0/117 selection flips, the 1 mover is
196649 itself** (the target event), and the fix recovers a physically
sensible, correctly-located, appropriately-short PF object at the vertex
without perturbing any boundary-sensitive tagger. This is the cleanest result
of any knob-fit round in this doc set to date — the census's population-level
prediction (n_suppress=1) held exactly at full-reconstruction scale.

### Bee links

| population | before | after |
|---|---|---|
| collateral (the one A/B mover, 196649) | https://www.phy.bnl.gov/twister/bee/set/063dbd3d-2ae2-4fbf-8207-77856039e29f/event/list/ | https://www.phy.bnl.gov/twister/bee/set/91f74215-bbd0-4ba9-9b68-90085a734772/event/list/ |
| misses (residual list) | — **0 junctions, no set built** — | — |

Recorded in `docs/pr/pr72-bee.index.txt`. The "misses" population is
genuinely empty (`es3_residual.tsv`, 0 rows) — stated explicitly here per
M13/no-silent-truncation rather than omitted.

### Blast radius statement

Every A/B run in this round covers exactly the requested 117 events (48
nueCC + 19 NC π⁰ + 50 PR data), no more, no fewer. No other detector's config
was touched (`cfg/pgrapher/experiment/sbnd/*.jsonnet` and the shared
`cfg/pgrapher/common/clus.jsonnet` key-suppression addition only). No
production default changed anywhere; `es3_stub_guard` C++ default is `false`
and no cfg file sets it to anything else.

### SBND production flip — DONE, owner flip (2026-08-12)

`es3_stub_guard = true` in `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`
(toolkit `fc9d1fcb`), cfg-only per doc 68 — the C++ knob default itself stays
`false`; legacy escape `-A es3_stub_guard=false` (or
`SBND_ES3_STUB_GUARD=0`) restores the pre-flip production bare, byte-exact.
Compile proof is a real run: bare production evt 196649 (no env override)
has a `mabc-pr.zip` member-content hash
(`0f4df72d5d2633d305ad1bf8d0157dd64e31b9b6edbec6774f0cc08c8c680e90`)
byte-identical to the previously-validated `SBND_ES3_STUB_GUARD=1` arm
(`work-pr72-on48`).

### Files touched this round

- `clus/src/NeutrinoStructureExaminer.cxx` — census (`ES3PB`/`ES3CENSUS`/
  `ES3MERGE`, env-gated) + guarded early-out for `es3_stub_guard`.
- `clus/inc/WireCellClus/PRSegmentFunctions.h` + `clus/src/PRSegmentFunctions.cxx`
  — `Es3StubGuardParams`, `es3_stub_suppress`.
- `clus/inc/WireCellClus/NeutrinoPatternBase.h` — `PatternAlgorithms` members.
- `clus/inc/WireCellClus/TaggerCheckNeutrino.h` + `clus/src/TaggerCheckNeutrino.cxx`
  — knob plumbing (`configure`/`default_configuration`/copy).
- `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — key-suppression knob
  plumbing, SBND-only.
- `clus/test/doctest_prsegment.cxx`, `clus/test/doctest_clus_knob_defaults.cxx`
  — new tests.
- `wcp-porting-img/sbnd/sbnd_xin/run_pr_chain_batch.sh` — `SBND_ES3_STUB_GUARD`
  + 5 sub-parameter env bridges.
- `wcp-porting-img/sbnd/sbnd_xin/run_pr_chain_batch_isolated72.sh` (new) —
  isolated-worktree runner fork.
- `wcp-porting-img/sbnd/sbnd_xin/scripts/analysis/pr72/{es3_census,es3_analysis}.py`
  (new) — census parser + offline threshold-fit analysis.
- `wcp-porting-img/sbnd/sbnd_xin/docs/pr/pr72-bee.index.txt` (new).
- This doc.

## Round 3 — chain audit for the near-vertex trajectory, `es3sg_vertex_fit` implemented, NEGATIVE on-arm result (DEFAULT OFF, no flip recommended)

Owner follow-up after round 2's topology fix: "the track trajectory near the
vertex is not ideal... if we do a fit with the two tracks, then we can have
a much sharper vertex turn... in other events, we have vertex fitting
techniques etc." — asking whether the chain's existing vertex-fitting
machinery could sharpen the geometry now that the topology is correct.

### Repro block

```bash
# worktree: none needed, shared tree was clean (git status --short) at round start
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild
env -u LD_LIBRARY_PATH ./build/clus/wcdoctest-clus     # 180/180

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# off-gate (default OFF, no override) -- 48-event sample
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr72r3-off48 data
python3 scripts/analysis/pr49/on_compare.py work-pr72-on48 work-pr72r3-off48   # must be 0/48

# single-event on-arm smoke + trace
SBND_ES3SG_VERTEX_FIT=true PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr72r3-smoke196649 data 196649
SBND_ES3SG_VERTEX_FIT=true SBND_WCT_LOGLEVEL=trace PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr72r3-trace196649 data 196649
```

### The mechanism traced: `MyFCN`/`improve_vertex`, and why it's gated away from this junction

`MyFCN::UpdateInfo` (`clus/src/MyFCN.cxx:308-499`) is the only code in `clus/`
that can sharpen a vertex turn: it PCA-fits each incident arm over an annulus
that *excludes* the near-vertex cm (`vertex_protect_dis` 1.5cm/0.9cm to
`fit_dis` 6cm, `MyFCN.cxx:88-104`), solves the vertex as the weighted
intersection of those arm axes (`FitVertex`, `:204-306`), then rewrites both
arms' `wcpts()` between the vertex and each arm's PCA-center point with a
straight, Steiner-snapped interpolation (`UpdateInfo`, `:415-479`) before
`clear_fit()` lets the next `do_multi_tracking` re-derive the fit from that
straightened skeleton.

Four gates keep it away from a degree-2, non-main junction:

| # | gate | file:line |
|---|---|---|
| 1 | `vertex_segments.size() <= 2 && vtx != main_vertex → continue` | `NeutrinoVertexFinder.cxx:2546` |
| 2 | `ntracks == 0 && vtx != main_vertex → continue` (shower-aware, per-vertex) | `:2558` |
| 3 | `flag_skip_two_legs && size <= 2 → continue` (cluster-wide: true when every segment in the cluster is shower-flagged) | `:2511`, `:2559` |
| 4 | `FitVertex`: needs `ntracks>2` **or** `enforce_two_track_fit`, which is set only when `vertex == main_vertex` | `MyFCN.cxx:241`, `:2193` |

Gates 1 and 4 are **prototype-faithful** (`pid/src/NeutrinoID_improve_vertex.h:81,696`)
— not a porting gap (M15). Re-reading the gate structure precisely: the fit
is reachable **without any override** by any junction with **≥3** incident
segments (gate 1 never fires, and `ntracks>2` alone can satisfy gate 4), and
separately reachable **at the main/neutrino vertex specifically**, even at
degree 2, via the `enforce_two_track_fit` override. **A degree-2, non-main
junction — exactly the stub/trunk case here — is the one combination the
gating excludes.** This is very likely what the owner is recalling from
"other events": genuine multi-prong vertices, or the neutrino vertex itself,
not a track emerging from a shower trunk. The measured result below is
consistent with that reading: the PCA/two-line-intersection model implicit
in `MyFCN` assumes two reasonably clean, well-separated track directions,
which does not describe a stub meeting a diffuse shower.

### Offline measurement before touching code

Replicated `MyFCN::AddSegment`'s exact point selection + PCA on the round-2
`on48` archive's fitted points (no rerun): stub 7 pts / trunk 9 pts in the
1.5-6cm annulus, **annulus-to-annulus line angle 26.8°**, `n_large_angles=1`
(passes `FitVertex`'s own 15° admission test). Solved `FitVertex`'s normal
equations offline: predicted vertex displacement only ~0.1-0.36cm. This
measurement showed the fit *would* engage productively if admitted — but, as
established below, it measures a different geometric quantity (a PCA
principal axis through annulus points) than what the subsequent charge-based
refit actually produces at the vertex; it is not a reliable predictor of the
final on-arm deflection.

### Owner decision: implement Candidate A

Per owner's explicit choice after reviewing the gate analysis and offline
measurement (vs. diagnosis-only), implemented the most targeted candidate:
admit the junction `es3_stub_guard` protects into the two-track vertex fit,
without widening to all degree-2 junctions (Candidate B) or touching the
fit's charge-division/area-revert internals (Candidates C/D, out of scope).

**New knob** `es3sg_vertex_fit` (bool, C++ default `false`), inert unless
`es3_stub_guard` is also on. **New vertex flag**
`VertexFlags::kStubGuardJunction` (`PRVertex.h`, `1<<4`, additive, not reusing
`kProtectedBreak`). Three sites:

1. `NeutrinoStructureExaminer.cxx` (the declined-merge branch, `:682-696`):
   marks the junction with the flag when `m_es3sg_vertex_fit`.
2. `NeutrinoVertexFinder.cxx` improve_vertex's vertex loop (`:2546-2559`
   originally): all three skip gates gain `&& !is_sg_junction`, where
   `is_sg_junction = m_es3sg_vertex_fit && vtx->flags_any(kStubGuardJunction)`
   — short-circuits false when the knob is off.
3. `fit_vertex` (`:2193`): `enforce_two_track_fit` is also set when the
   vertex carries the flag and the knob is on.

Plumbed through `NeutrinoPatternBase.h`, `TaggerCheckNeutrino.{h,cxx}`,
`cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`,
`run_pr_chain_batch.sh` (`SBND_ES3SG_VERTEX_FIT`), following the
`es3_stub_guard` precedent exactly. `CHECK_KNOB_BOOL(cfg, "es3sg_vertex_fit", false)`
added to `doctest_clus_knob_defaults.cxx`.

### Gates

- Freshness proof: `libWireCellClus.so` installed 10:37:49, after every
  edited source file (10:34-10:35). `./build/clus/wcdoctest-clus`: **180/180**
  passed (same count as round 2 — only the existing knob-defaults doctest
  was extended, no new pure-function arithmetic to pin).
- Compiled-config proof (M6): bare production compile — `es3sg_vertex_fit`
  key **absent**; `--tla-code es3sg_vertex_fit=true` override — key
  **present, `true`**.
- **Off-gate, single-event exact**: `work-pr72r3-smoke196649-off/pr_evt196649/mabc-pr.zip`
  hashes `0f4df72d5d2633d305ad1bf8d0157dd64e31b9b6edbec6774f0cc08c8c680e90`,
  byte-identical to `work-pr72-flipcheck` (round 2's validated bare-production
  arm); `pctree-pr-evt196649.tar.gz` likewise identical
  (`a92d7454...1502144`).
- **Off-gate, 48-event sample**: `work-pr72r3-off48` (bare, no override) vs.
  `work-pr72-on48` (round 2's validated current-production arm),
  `on_compare.py`: **0/48 archive-level differences, 0/48 `nusel-events.tsv`
  diffs, 0/48 `nusel-table.tsv` diffs.**

### On-arm result: the deflection got smaller, not larger — NEGATIVE

`SBND_ES3SG_VERTEX_FIT=true` on evt 196649, trace-level log confirms V2/V3
directly (not inferred): the junction reaches `improve_vertex` at **degree 2
(`nsegs=2`)**, in both `improve_vertex` call sites (`flag_search_vertex_activity=false`
from `determine_main_vertex`, and the final `flag_search_vertex_activity=true`
call); `fit_vertex`/`UpdateInfo` fire and succeed (`UpdateInfo: Cluster: 11
Update Vertex: ...` logged, `fit_vertex done` reported). The gate bypass and
the `enforce_two_track_fit` override both worked exactly as designed.

**Deflection angle, measured identically on both arms** (ball-centroid
estimator matching `segment_cal_dir_3vector`, verified to reproduce the
shipped `es3_stub_guard`'s own `ang3`/`ang10` to 0.07° on the off arm):

| R (cm) | off (round 2 baseline) | on (`es3sg_vertex_fit=true`) |
|---|---|---|
| 2 | 27.3° | 15.2° |
| 3 | 21.1° | 13.1° |
| 5 | 18.4° | 13.7° |
| 10 | 18.4° | 15.6° |
| 15 | 16.7° | 15.0° |
| 20 | 16.1° | 14.9° |

**The fit made the near-vertex trajectory measurably straighter, the
opposite of the intended "sharper turn".** Likely mechanism: `UpdateInfo`
only writes a straight *seed* into the wcpts; the subsequent
`do_multi_tracking` re-solves every point independently from 2D charge
(`TrackFitting::fit_point`, no smoothness or sharpness term — confirmed
absent in both the toolkit and the prototype this round, see the trajectory-fit
comparison below), using the same shared-charge-cell mechanisms identified
in this round's chain audit (flat `1/N` charge division at overlapping
cells, `form_map_graph`'s ≤0.8cm min-radius charge ball at the vertex,
the area-revert clamp at the first interior point). The straight seed does
not survive the refit; if anything the refit converges to a blunter
compromise than the pre-fit geometry.

**Two side effects, reported as observations, not explained further:**
- The main vertex (degree-1, the true neutrino vertex, untouched directly by
  this knob) ended up at `(-53.7175, -21.5088, 193.682)` — coordinate-for-coordinate
  the **pre-round-2** position, not round 2's `(-53.5009, -21.4557, 193.681)`.
  Candidate explanation: `UpdateInfo` snaps the vertex marker to the nearest
  Steiner point, and the small computed displacement (~0.1-0.36cm) may simply
  land back on the same discrete Steiner-cloud point production originally
  used, rather than any deliberate "undo". Not confirmed further.
- The small `proton 7 MeV` / 0.575cm vertex-activity object (`real_cluster_id
  11086`) that round 2's on-arm produced does not appear in this arm. A
  second-order effect of the same knob on the target event's PF-object count;
  not investigated further this round.

### Accept/stop decision: **STOP — do not flip, do not widen scope**

Per CLAUDE.md §5.7 (report a wrong physics number, don't tune to make it
look right): the measured on-arm effect contradicts the design intent. The
117-event census was **not run** — there is no decision the census would
inform, since the single-event measurement already shows the mechanism moves
in the wrong direction. `es3sg_vertex_fit` ships as a validated, byte-identical-when-off
default-OFF knob (a complete, honestly-reported deliverable per this round's
plan), but is **not recommended for further pursuit or flip**.

### What this redirects attention to

Of the four candidates scored this round, **A is now measured negative**.
The chain audit's own finding — no smoothness/sharpness term anywhere in
`TrackFitting::fit_point`, flat charge division (`charge_div_method`
hardcoded to 1, the Gaussian `div_sigma` branch dead code), the shared
≤0.8cm vertex charge ball, and the 1.8mm area-revert clamp — point at
**Candidate C** (expose `charge_div_method`/`div_sigma`) and **Candidate D**
(relax the area-revert near a vertex) as the levers that actually touch the
mechanism responsible for the bend. Both are out of scope this round (wide
blast radius, need their own gate sets) and are left for the owner's
decision on whether to pursue.

### Prototype comparison (M15 check)

No joint/simultaneous multi-track fit with a smoothness or vertex-sharpness
term exists in **either** tree — `PR3DCluster_multi_track_fitting.h`
(prototype) and `TrackFitting::multi_trajectory_fit` (toolkit) both solve an
independent 3-parameter LSQ per point, sharing the vertex only by identity
(one `fit_index`, hard-copied into both arms' endpoints). The toolkit port
is faithful on every load-bearing point. The "vertex fitting technique" the
owner recalls is `MyFCN`/`improve_vertex`, confirmed above to be gated to
main-vertex/multi-prong cases by design, matching the prototype
(`NeutrinoID_improve_vertex.h:81,696`).

One unrelated, pre-existing, undocumented divergence surfaced (not pursued):
the toolkit's `dQ_dx_multi_fit` connects every incident segment to a shared
vertex's regularizer row, while the prototype's equivalent code (an indexing
quirk, `PR3DCluster_multi_dQ_dx_fit.h:723`) only connects the first. Affects
charge/PID smoothing at multi-prong vertices generally; flagged for the
owner as a separate item (introduced by `fca0f7cfd`, "continue dbug", no
porting-dictionary entry).

### Files touched this round

- `clus/inc/WireCellClus/PRVertex.h` — new `VertexFlags::kStubGuardJunction`.
- `clus/inc/WireCellClus/NeutrinoPatternBase.h` — `m_es3sg_vertex_fit`.
- `clus/src/NeutrinoStructureExaminer.cxx` — flag-set on decline.
- `clus/src/NeutrinoVertexFinder.cxx` — gate bypass + `enforce_two_track_fit`.
- `clus/inc/WireCellClus/TaggerCheckNeutrino.h` + `clus/src/TaggerCheckNeutrino.cxx`
  — knob plumbing.
- `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — key-suppression
  plumbing, default OFF (no flip).
- `clus/test/doctest_clus_knob_defaults.cxx` — new knob-default check.
- `wcp-porting-img/sbnd/sbnd_xin/run_pr_chain_batch.sh` — `SBND_ES3SG_VERTEX_FIT`.
- This doc.

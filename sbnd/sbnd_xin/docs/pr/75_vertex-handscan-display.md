# doc pr/75 — neutrino-vertex hand scan: the `vertex_scoreboard` dump and the `pr_display` scan panel

Doc pr/52 §5 named two prerequisites for improving the SBND neutrino vertex
without retraining the DL model: a **vertex scoreboard** (§5.1) that records
what the two selectors actually compared, and **hand-scan labels carrying a
3-D true-vertex position** rather than a per-event correct/incorrect verdict
(§5.4). This round builds both.

Status: **tooling shipped, no labels collected yet, no operating point moved.**
Nothing in this round changes reconstruction: the C++ side is a default-OFF
recording knob and the rest is a display.

Toolkit: `vertex_scoreboard` knob + `PrDisplayDump` emitter + jsonnet + doctest.
wcp-porting-img: the `pr_display` scan panel, the driver pass-through, this doc.

> **The toolkit commit is on a BRANCH, not on `apply-pointcloud`.** `04b6e47d`,
> parented on `40651cb2`, pushed as **`origin/pr75-vertex-scoreboard`**. It was
> not landed on `apply-pointcloud` because a concurrent session held
> uncommitted work in *every* file this round touches
> (`NeutrinoPatternBase.h`, `TaggerCheckNeutrino.{h,cxx}`,
> `NeutrinoVertexFinder.cxx`, all three jsonnet files,
> `doctest_clus_knob_defaults.cxx`), and git stages whole files — committing
> from the shared tree would have swept that unfinished pr/74 work in. The
> branch is a clean fast-forward from `40651cb2`; land it with
> `git merge --ff-only pr75-vertex-scoreboard` once the tree is free.
> wcp-porting-img `dac67b2` is on `main` and pushed normally.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- produce scannable dumps (48 nueCC events).  PR_EXTRA_STAGES=pr_display
#     alone is enough: the driver turns SBND_VERTEX_SCOREBOARD on with it.
PR_EXTRA_STAGES=pr_display PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-prdisp-vtx48 data

# --- serve the hand scan on port 5017
./pr_display/serve_pr_display.sh 5017 --scan-tag vtxscan1 \
  "work-prdisp-vtx48/pr_evt*/calib-pr-evt*.json"
# from a laptop:  ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
#                 http://localhost:5017/pr_display_viewer

# --- the dump-vs-log cross-check of §4 (needs TRACE)
SBND_WCT_LOGLEVEL=trace PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-prdisp-trace1 data 469665
```

Everything below was measured against toolkit `40651cb2` in an isolated
worktree (§6), on `work-nuecc48-cb0805`.

---

## 1. Why a new knob when `pr_display` already exists

The owner's instruction was to reuse the knob already attached to
`pr_display`. That is not sufficient on its own, for a mechanical reason:
`PrDisplayDump` is an ensemble visitor that runs **after**
`TaggerCheckNeutrino`, and every number the scan needs — the DL top-K voxel
scores, the seven rerank composite terms, `compare_main_vertices`' additive
score per candidate — is a **local variable** inside
`determine_overall_main_vertex_DL` / `compare_main_vertices`. By the time the
dumper looks, they are gone.

So the recording has to happen inside the tagger, which is a different
component with its own config: knob `vertex_scoreboard` on
`TaggerCheckNeutrino`, C++ default `false`. This is exactly the knob doc
pr/52 §5.1 specified, on exactly that component.

**The owner still types only what they said they would.** The driver defaults
`SBND_VERTEX_SCOREBOARD=true` whenever `PR_EXTRA_STAGES` names `pr_display`,
so `PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh ...` — the command doc
pr/26 already documents — produces a scored dump. An explicit
`SBND_VERTEX_SCOREBOARD=false` reproduces a pre-pr/75 dump.

The board is stashed on `TrackFitting` beside `set_tagger_info`
(`TaggerCheckNeutrino.cxx`), the same channel `dump_tagger` already reads.
That site is deliberately **after** pr/50's `snap_main_vertex_to_kink` and the
final `improve_vertex`, so `final_vertex_id` is the vertex the display draws;
the dump carries it so a disagreement with `main_vertex` would be visible
rather than silent (measured: they agree to 1e-3 cm).

## 2. What the dump contains

`calib-pr-evt<ID>.json` gains a `vertex_scoreboard` object.

**An empty object means the knob was off — "no scoreboard was taken", never
"no candidates existed".** Same discipline as `vertices[].main_candidate`
(doc pr/26).

| field | meaning | source |
|---|---|---|
| `route` | which of doc pr/52 §1's routes chose the vertex | set at every exit of `determine_overall_main_vertex_DL` |
| `weights_missing` | `dl_weights` was configured but `Persist::resolve` failed | `TaggerCheckNeutrino::configure` |
| `dl_ran` / `dl_rerank` / `dl_accepted` | inference attempted / rerank branch / DL choice adopted | |
| `dl_best_score`, `dl_min_accept_score`, `dl_score_scale`, `dl_top_k` | the operating point in force **this event** | |
| `voxels[]` | the raw top-K net output: `rank, x, y, z, dl_score` | the only numbers from the model itself |
| `rows[]` | one per PR-graph vertex the selectors touched | |
| `final_vertex_id`, `final_x/y/z` | the vertex carrying `kNeutrinoVertex` at stash time | |

Each `rows[]` entry joins to `vertices[].id` on `vertex_id` (the same
`cluster_id*1000 + graph_index` encoding the Bee particle-flow tree uses) and
carries `trad_scored / trad_score / trad_winner`, then `dl_snapped /
voxel_rank / dl_score / snap_dis / host_length`, the seven terms `s_dl,
s_snap, s_fwd_z, s_clen, s_isol, s_main, s_fv`, `total`, and `dl_winner`.

Three fields exist to stop a zero being misread:

- **`dl_snapped == false` means the DL had *no opinion*** on that vertex (no
  voxel snapped to it). It is not a DL score of zero.
- **`trad_scored == false`** for clusters that took the all-showers branch:
  `compare_main_vertices_all_showers` is not additive and produces no
  comparable number, and for vertices that never entered a candidate list.
- **`skipped_by_swap_guard == true`** means `dl_vtx_swap_guard` removed the
  candidate *before* scoring, so its all-zero terms are an omission, not a
  measurement.

**Determinism.** `rows[]` is emitted sorted by `vertex_id`. It is collected
from `map_vertex_num` and `snap_map`, both keyed on `VertexPtr` — i.e. address
ordered — so an unsorted dump would reshuffle run to run (CLAUDE.md
determinism rule). The recording loop iterates the `scored` vector, not those
maps, and reads them with `.find()`: an `operator[]` read on an unscored
candidate would **insert** into a container the legacy path then walks, which
is the one way this "pure recording" change could have perturbed output.

**Route 1 vs route 3.** A failed `Persist::resolve` leaves `m_dl_weights`
empty, so at the call site "weights path not found" is indistinguishable from
"DL never configured". Both emit `route = "dl-not-run"`; `weights_missing`
is the field that separates them. This closes the doc pr/52 §5.5 ops item
(route 3 is not caught by `grep "DL vertex failed"`).

## 3. The scan panel

`pr_display_viewer.py` gains a full-width row under the projections, and
`--scan-tag <name>`.

- **Candidate table.** One row per PR-graph vertex, joined to the scoreboard:
  `pick, vtx id, clus, x, y, z, deg, main, cand, DL score, snap cm, rerank,
  trad, d(main) cm`. Sort by rerank total (default), DL score, trad score,
  distance to main, or cluster+id.
- **Filter.** Default is **main cluster + DL**: every vertex on the main
  cluster plus every DL-snapped vertex wherever it sits. Measured on these
  dumps, "candidates" is *not* a scannable set — an event has 63–162
  PR-graph vertices and **50–124** of them are main-vertex candidates, while
  the default gives 4–36. It cannot hide the failure class where the main
  *cluster* is wrong, because every vertex the DL pointed at is listed
  regardless of cluster. Two wider filters are one click away.
- **Click a row** → the centre moves there, zoom engages, and all nine panels
  reframe (reusing `apply_ranges()`); the row is ringed in amber in the three
  projections.
- **Picks are ranked.** `add pick` appends; the first is the 1st choice, later
  ones are alternates that could not be ruled out. They are drawn as green
  diamonds labelled with their rank.
- **Manual position** (`x/y/z` + `add manual pick`), with `from centre` and a
  `tap fills coords` toggle — with it on, a tap in a projection writes the two
  coordinates that panel shows, so two taps in two panels pin a full 3-D
  point.
- **Confidence**: `certain / likely / unclear`.

**A manual pick auto-sets `not_a_candidate`.** That is doc pr/52's Tier D: the
true vertex was never in the candidate set, so no vertex-*selection* tuning
can fix the event and it must be **excluded** from an acceptance fit rather
than fitted against. It is a pr/51 graph-robustness case.

**Labels**: `sbnd_xin/vertex_labels/<scan-tag>/labels-evt<ID>.json`, one file
per event, written tmp+rename so a record is never half-written. The candidate
scores are copied **into** the label, so a tuning fit joins one file per event
and never re-reads the dump.

```json
{"event": "evt10550", "runNo": 18255, "eventNo": 10550,
 "scan_tag": "vtxscan1", "confidence": "likely", "not_a_candidate": true,
 "route": "dl-rerank-accept", "scoreboard_present": true,
 "main_vertex": {"x": ..., "y": ..., "z": ..., "cluster_id": 48},
 "picks": [{"rank": 1, "kind": "candidate", "vertex_id": 48052,
            "x": 60.78, "y": 55.00, "z": 165.95, "dl_score": 0.005273,
            "rerank_total": 5.9771, "trad_score": null, "is_main": true,
            "dl_winner": true, "dis_to_main": 0.0},
           {"rank": 3, "kind": "manual", "vertex_id": null,
            "x": 11.50, "y": -22.25, "z": 333.75, "dis_to_main": 191.18}]}
```

**M13 is enforced, not just documented.** `--scan-tag` passed explicitly is
consent to write into that set. Without it the viewer uses `scan1` but
**refuses to write** if that directory already holds labels, and says so.

### A 2-D framing fix that came with it

`apply_ranges()` derived each wire-plane window from fitted segment points
within ±half-width and fell back to the panel's *full extent* when fewer than
two were found — so clicking an isolated micro-stub candidate (the pr/51
class) gave a useless 2-D view precisely where it was most wanted. It now
grows the search box (×1, 2, 4, 8) until two points are found. The three 3-D
projections were always correct.

## 4. Verification

| # | gate | result |
|---|---|---|
| 1 | freshness proof (M1) | libs 06:55:52/06:55:54 newer than last source edit 06:54:08 |
| 2 | `wcdoctest-clus` | **204/204 passed**, 2039 assertions, 0 failed |
| 3 | compiled-config proof, knob **off** | **byte-identical** to a pristine `git archive 40651cb2` tree (261398 B both) |
| 3 | compiled-config proof, knob **on** | `"vertex_scoreboard" : true` present; 0 occurrences off, 1 on |
| 4 | off-gate, 48 events | **48/48 byte-identical**, 0/48 nusel diffs |
| 5 | on-gate, 48 events | **48/48 byte-identical**; 1 nusel cell, log-parsing artifact (below) |
| 6 | dump vs the code's own TRACE | **PASS** |
| 7 | UI round-trip | **PASS** |

**Gate 4 — off-gate.** Arms `work-pr75-gateA` (pristine `40651cb2`, its own
worktree build) vs `work-pr75-gateB` (this round's code, knob at its C++
default), 48 nueCC events, production pipeline, **DL on**.
`abtest/hash_archive.py` on `mabc-pr.zip`: **48/48 identical member-content
hashes, 0 differ**, and 0/48 `nusel-evt<ID>.tsv` differences. So the code with
the knob off is byte-identical to the tree without the code.

**Gate 5 — on-gate.** `work-pr75-gateB` (knob off) vs `work-prdisp-vtx48`
(knob **on** + the `pr_display` stage appended), same 48 events:
**48/48 identical `mabc-pr.zip` hashes**. One `nusel` cell differed, on evt
52672: `stmfit` read `torn` on the knob-off arm and `contained` on the other.

That is **not** a reconstruction difference, and the diagnosis is shown rather
than asserted. `nusel_extract.py` derives `stmfit` by *parsing the log text*,
and `torn` is its own sentinel for a line it could not read (`:399`). Line 176
of the knob-off arm's log is spliced mid-word by another thread's record:

```
check_stm_conditions: cluster 9 no STM fit: [07:12:33.216] I [ clus ] <CreateSteinerGraph:prrefresh> ...
```

where the reason text `fully contained (Mid Point A)` should be — the known
log-tearing behaviour (`project_wct_log_line_tearing`). Re-running evt 52672
serially, one event at a time, on all three configurations:

| arm | `stmfit` | `mabc-pr.zip` hash |
|---|---|---|
| knob off, run 1 | `torn` | `526cf7a3c7f16749…` |
| knob off, run 2 | `torn` | `526cf7a3c7f16749…` |
| knob on + `pr_display` | `contained` | `526cf7a3c7f16749…` |

Identical reconstruction all three times; every other `nusel` field (tgm, stm,
fc, lm, label, point counts, lengths) matches. The tearing is *reproducible
per pipeline* — appending a stage shifts the thread interleaving — so it is a
property of the log writer, not of the knob.

**This qualifies doc pr/26 §6 slightly**: a `pr_display` arm's *archives* hash
identically to a plain arm's, which is what that claim is about, but a derived
TSV built by log-scraping can still differ in the `stmfit` column. Read
`stmfit == torn` as "the log line was unreadable", never as a tagger verdict.

**Gate 6 — the dump is checked against the code, not against itself.** On evt
469665 with `SBND_WCT_LOGLEVEL=trace`, every `DL rerank cand [voxel N] ... |
TOTAL=` line was parsed out of the log and compared term by term with the
JSON: voxels {0,1,2,4}, max |trace − json| = **0.0004** on all seven terms and
the total — the log's own 3-decimal print precision. `route` and
`dl_best_score` (12.7370) match the `rerank selected` line.

**Gate 7 — UI round-trip**, driving the viewer's own callbacks headlessly:
row click sets the centre and engages zoom; two ranked candidate picks plus a
manual pick save with the right schema; stepping to another event leaves 0
picks (no leakage) and stepping back restores all three picks, the confidence,
and a clean (not "unsaved") state. The M13 guard was tested by pre-seeding the
default tag: `write_allowed=False`, no file written, explanatory message
shown.

### DL run-to-run stability — a correction to M4's scope

CLAUDE.md **M4** says the DL/SCN vertex "is not bit-stable" and should be kept
out of gates with `-A dl_weights=` empty. That is why the gates above were run
at the production operating point (DL **on**) instead, and the evidence for
doing so is the gates themselves, not a side probe: **gate 4 is 48/48
identical member-content hashes across two independently built binaries with
DL on**, and gate 5 another 48/48. Forty-eight events agreeing bit-for-bit is
what establishes stability here.

The pre-check that made it worth trying was smaller and is reported as such:
evt 469665 run twice with DL on gave hash `e9b32c92e910...` both times.

Neither repeals M4. Whatever instability it was written for did not reproduce
on this chain and this box, and the safe reading is that it was
environment-dependent — so keep the `dl_weights=''` habit for gates unless you
have re-measured, as here.

## 5. What is NOT established

- **No labels have been collected.** The scan set is empty; nothing in doc
  pr/52 §5.2's failure split (a/b/c/d) can be populated yet.
- **No operating point moved.** `dl_vtx_min_accept_score` (4.0),
  `dl_vtx_score_scale` (1000) and the seven `W_*` constexpr weights are
  untouched. This round only makes them observable.
- **The `W_*` weights are still constexpr.** Re-fitting them needs the
  separate knob round doc pr/52 §5.2c describes; the scoreboard records their
  *effect* per candidate, which is enough to fit offline but not to A/B.
- **Tier B is untouched** — the `dQdx_scale`/`dQdx_offset` net-input
  calibration check (doc pr/52 §4 Tier B, §5.3) is independent of this round.
- **47 of 48 events are scannable.** evt 116962 produces no dump: its only
  in-window cluster was cosmic-tagged (TGM), so `TaggerCheckNeutrino` skipped
  the event and left no PR graph or `TrackFitting` in the grouping
  (`PrDisplayDump` warns `no TrackFitting in grouping 'live'`). That is
  correct behaviour — there is no neutrino candidate to scan — not a defect.

## 6. Two build/ops findings worth keeping

**A. The worktree ROOT trap — a new variant of M1.** The isolated worktree
was configured with `--with-root=yes`, copying the shared tree's own configure
line. `waft/rootsys.py` turns `--with-root=X` into a search path `X/bin`, so
`yes` means "look in ./yes/bin", and without `ROOTSYS` exported (direnv sets
it in an interactive shell, not in a scripted one) waf silently reported
`root-config: not found` and **dropped the entire `root` package**:

```
Removing package "root" due to lack of external dependency "HAVE_ROOTSYS"
```

The build then succeeded, `libWireCellRoot.so` was never built in the
worktree, and the run loaded the **shared** tree's copy — which was compiled
against the old `TrackFitting`. This round adds a member to that class, so its
layout shifted, and `SbndPrMagnifyTrackingVisitor::write_proj_data` read
`fc.clusters` at the wrong offset and segfaulted (rc=139) on **4 of 5**
events.

It looked exactly like a real upstream regression, and it survived the obvious
control (the same crash with the knob off and no `pr_display`, because the
control still used the ABI-shifted `libWireCellClus.so`). It was also blamed
on the wrong commit — running at `064824c1` reproduced it identically, which
seemed to exonerate the tip, when in fact both arms shared the fault.
*Rule: after configuring a worktree, check the "Configured for submodules:"
line against the shared tree's before trusting any run; pass `--with-root=$ROOTSYS`
with `ROOTSYS` exported, never `--with-root=yes`.* With `root` built, 5/5 and
then 48/48 events pass.

**B. Boolean env pass-throughs need `false`, not `0`.**
`SBND_VERTEX_SCOREBOARD=0` becomes `--tla-code vertex_scoreboard=0` and
jsonnet rejects it: `RUNTIME ERROR: Unexpected type number, expected boolean`.
Use `=false`. This is the existing convention for every boolean pass-through
in the driver, not something this round introduced.

## 7. Where this goes next

The panel exists so doc pr/52 §5.2 can be executed: scan the 47 events, then
split the failures by `route` and by whether the true vertex was a candidate
at all (`not_a_candidate`). (a) DL right but rejected → retune
`dl_vtx_min_accept_score` / `dl_vtx_score_scale`; (b) DL wrong and accepted →
raise the threshold or re-fit `W_*`; (c) both selectors wrong with the true
candidate present → Tier C; (d) `not_a_candidate` → pr/51 graph audit, and
excluded from the fit. The picks also carry the 3-D positions doc pr/52 §2.2.1
names as the missing input for an in-tree DeepVtx fine-tune.

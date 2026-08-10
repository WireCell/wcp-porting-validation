# doc pr/57 — overclustering-separation hand-scan display

Status: **IMPLEMENTED, served on port 5018 over all 117 events (48 nueCC +
19 NC-pi0 + 50 PR-data).** A default-OFF, env-gated JSONL dump inside
`connect_graph_relaxed_strict.cxx` feeds a new Bokeh viewer
(`sbnd_xin/overclustering_display/`) so the S6 2D-connectivity removals from
doc pr/56 round 3 can be judged by eye and labelled good/OK/bad with a cause,
building the statistics the next round of S6 tuning needs. No algorithm
change: `two_d_connectivity_bad`, every shipped constant, and the S6 verdict
path are untouched (dump-OFF byte-identical gate PASS 0/117, §4).
Toolkit `e47b1486` (committed, not pushed); wcp-porting-img `5a529e0`
(pushed).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# produce the scan dumps (fresh labels, per M13):
PR_JOBS=6 SBND_PROTECT_GRAPH=relaxed_strict_img_2d \
WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r4b-scan48 data <48 evts>
  ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r4b-scan19 data <19 evts>
  ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr56r4b-scan50 data <50 evts>

# verify the dump does not lie (§5):
python3 scripts/analysis/pr57/oc56_dump_check.py \
  work-pr56r4b-scan48 work-pr56r4b-scan19 work-pr56r4b-scan50

# serve:
./overclustering_display/serve_overclustering_scan.sh 5018 \
  work-pr56r4b-scan48/pr_evt*/oc56scan-evt*.jsonl \
  work-pr56r4b-scan19/pr_evt*/oc56scan-evt*.jsonl \
  work-pr56r4b-scan50/pr_evt*/oc56scan-evt*.jsonl
# ssh -L 5018:localhost:5018 wcgpu1.phy.bnl.gov, then open
# http://localhost:5018/overclustering_scan_viewer

# dump-OFF byte-identical gate (§4), fresh arms vs round-3's own baselines:
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr56r4b-off48 data <48 evts>
PR_JOBS=6 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr56r4b-off19 data <19 evts>
PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr56r4b-off50 data <50 evts>
python3 scripts/analysis/pr49/on_compare.py work-pr56r4b-off48 work-pr56r3-off48
python3 scripts/analysis/pr49/on_compare.py work-pr56r4b-off19 work-pr56r3-off19
python3 scripts/analysis/pr49/on_compare.py work-pr56r4b-off50 work-pr56r3-off50

# Bee set, all 117 events in one upload:
python3 scripts/bee/make_pr_bee.py \
  -q work-nuecc48-cb0805 -q work-ncpi0-cb0805 -q work-mcp1k-cb0805 \
  -p work-pr56r4b-scan48 -p work-pr56r4b-scan19 -p work-pr56r4b-scan50 \
  -o bee/pr57/pr57-117.zip --allow-unevaluated <117 evts, 48+19+50>
BROWSER=echo ./upload-to-bee.sh bee/pr57/pr57-117.zip
```

Full event lists: `bee/pr57/pr57-117.index.txt` (bee index -> event id, the
same order as `-q`/`-p` above -- indices 0-47 nueCC, 48-66 NC-pi0, 67-116
PR-data).

## 1. Why this exists

doc pr/56 round 3 shipped S6 (the per-plane 2D wind/tick fired-pixel
connectivity check, `relaxed_strict_img_2d`, still default OFF everywhere in
production) with a repaired BFS and a 1 cm distance floor. That round's own
honest finding was left unresolved: at the shipped operating point, across
the 48 nueCC + 19 NC-pi0 + 50 PR-data sample, S6 evaluates 1875 candidate
edges and would remove **1433 of them (76%)** -- one event alone (256587)
has 169 removals -- and most comparable clusters come out *more* fragmented
than production if S6 were ever defaulted on.

Whether any given removal is right cannot be settled by that aggregate
number. Each one is either:

- a **real gap** -- the two components genuinely don't touch at the
  wire/tick level, and S6 is correctly refusing to bridge them, or
- an **artifact** -- most often induction-plane signal inefficiency (U/V
  planes lose real hits that W usually keeps), where S6's "no fired pixel
  here" reads as a break when the true track is continuous.

Telling those apart requires looking at the actual charge pixels the BFS
saw. This display makes that look fast and records the verdict.

## 2. Why a new dump, not a Python reconstruction from existing artifacts

Nothing already written carries what the judgement needs. The census log
(`OC56CENSUS-2D edge`, doc pr/56 round 3) has endpoints, gap verdicts and the
`dw x ds` matrix, but no pixels. `pctree-pr-evt<ID>.tar.gz` is written
*after* the cluster is (or isn't) split, so the two candidate components no
longer exist as such by the time it's read, it carries no `is_wire_dead`
information, and reconstructing the fired-pixel predicate independently in
Python risks a second implementation that can silently disagree with the one
that actually decided the kill. `pr_display/` already set the pattern for
this exact situation (`PrDisplayDump.cxx` + a Bokeh viewer that reads only
that dump) -- this follows it.

## 3. The dump (`clus/src/connect_graph_relaxed_strict.cxx`)

Entirely inside the existing `two_d_gap_kill` lambda and one new anonymous-
namespace singleton (`Oc56DumpWriter`). Gated on `WCT_OC56_SCAN_DUMP` (a file
path); unset (the default) means the singleton's constructor never opens a
file and every dump-only branch is skipped -- `two_d_connectivity_bad`, every
shipped constant (`s6_dw_*`, `s6_ds_*`, `s6_reach`, `s6_dis_floor`,
`s6_dis_cap`), and the verdict path are byte-for-byte what round 3 shipped.

One compact JSON object per line, appended under a mutex. Two record types:

- **`component`** -- one per (graph_call, comp), written the first time an
  edge in this cluster-graph call references it (deduped locally, so an
  edge referencing the same component many times over doesn't re-dump it):
  `{"type":"component","graph_call":N,"comp":j,"cluster_id":C,"npts":n,"points":[[x,y,z]...],"truncated":bool}`,
  points in cm, capped at 20000 with a stride if the component is bigger.
- **`edge`** -- one per S6-evaluated edge (every edge, not only kills -- the
  442 survivors are how false negatives get labelled too):
  `{"type":"edge","graph_call":N,"j":j,"k":k,"blk":"closest|dir1|dir2","dis":cm,"apa":A,"face":F,"p1":[...],"p2":[...],"slice_step":s,"gap":[u,v,w],"excuse":[u,v],"budget":[u,v,w],"matrix":["16 chars"x3],"killed":bool,"planes":[{...}]}`.
  Each `planes[]` entry (only for planes with seed data on both sides, same
  convention the census matrix already uses) carries the window
  `[wlo,whi,slo,shi]`, `native_step`, the `fired` cell list, `dead` ranges,
  and both components' own `seeds_a`/`seeds_b`.

**Correlation key.** `graph_call` is a process-wide atomic counter, not
`j`/`k` (cluster-local indices that collide across clusters in the same
event -- exactly the failure mode doc pr/56 round 3 sec 8.3 hit and fixed
for the census log by collapsing to one self-contained line). A component
record and the edge records that reference it stay correctly paired no
matter how writes from concurrently-processed clusters interleave in the
shared file, because the key that ties them together can never collide.

**Lattice: native step, not `slice_step`.** The real BFS steps by
`slice_step` multiples *starting from each seed's own slice value* -- two
seeds on different residues mod `slice_step` query different, interleaved
lattices, so the set of cells the BFS can actually touch is a union across
residues, not one lattice anchored anywhere. The dump instead enumerates
**every integer slice** in the window (`native_step=1`), a strict superset
of anything any BFS invocation inside that window could touch, up to a
300000-cell-per-plane cap (window wires x window slices); past the cap it
falls back to stepping by `slice_step` and records that in `native_step` so
the checker (§5) knows a native dump was not attempted for that plane. In
every event scanned so far (§4), the cap was never hit.

## 4. Verification: the dump is inert

**Freshness proof**: `local/lib/libWireCellClus.so` (2026-08-10 05:46:31)
postdates the last source edit (05:46:04) -- M1.

**`wcdoctest-clus`: 145/145**, both before and after the native-step
redesign (round 2 of this implementation).

**Kill-set identity vs. doc pr/56 round 3's own shipped arms.** Parsed both
sides' `OC56CENSUS-2D edge` lines (regex-based, tolerant of the known
WCT log-tearing issue -- concurrent cluster processing occasionally
interleaves an unrelated INFO line mid-write into a DEBUG line, dropping
that one line from the regex parse on both sides symmetrically; see doc
pr/53 sec "known WCT log-tearing issue" -- not something this change
introduces) and compared
`(dis, gap_u, gap_v, gap_w, excuse_u, excuse_v, killed)` per `(evt, blk, j,
k)`:

| sample | edges parsed both sides | only-one-side | value mismatches |
|---|---|---|---|
| 48 nueCC | 1472 / 1472 | 0 | 0 |
| 19 NC-pi0 | 325 / 325 | 0 | 0 |
| 50 PR-data | 57 / 57 | 0 | 0 |

Every parseable verdict is byte-identical to round 3's own shipped arms --
the dump changes nothing about what S6 decides.

**Dump-OFF byte-identical gate.** Bare reruns (no env override at all) of
all three samples, compared via `hash_archive.py` member-content hashes
(never raw `cmp`) plus `nusel-events.tsv`/`nusel-table.tsv`, against round
3's own already-gated OFF baselines:

| sample | archive-level | nusel-events.tsv | nusel-table.tsv |
|---|---|---|---|
| 48 nueCC | PASS 0/48 | PASS 0/48 | PASS 0/48 |
| 19 NC-pi0 | PASS 0/19 | PASS 0/19 | PASS 0/19 |
| 50 PR-data | PASS 0/50 | PASS 0/50 | PASS 0/50 |

**Dump size**: 74 MB total JSONL across all 117 events (1875 edges, up to a
few hundred kB per event). Timing: single-run wall-clock on the two named
target events came out *faster* with the dump on than the bare rerun (13s
vs 17s on 269774, 20s vs 25s on 71372) under concurrent batch load -- that is
system-load noise, not a real speedup, and is reported as such rather than
claimed as evidence of anything (round 3's own timing discipline: single-run
numbers, not a profile).

## 5. Verification: the dump does not lie about the verdict

The whole point of this display is that a human trusts what it draws. If the
dumped fired/dead cells don't actually reproduce S6's own gap verdict, a
label collected against the display is worthless -- an operator could judge
"real gap" while looking at pixels that never determined the removal.

`scripts/analysis/pr57/oc56_dump_check.py` replays the exact same bounded
Chebyshev-box BFS `s6_planes_connected()` runs (same neighbor-step
construction, same window clipping, same `cell_budget=20000` circuit
breaker), entirely from each edge's own dumped `fired`/`dead`/`seeds_a`/
`seeds_b`, at **every** logged `(dw, ds)` in `1..4 x 1..4` -- not just the
shipped `(1,1)` operating point -- and asserts the replayed connectivity bit
equals the logged matrix bit, for every `native_step=1` plane record.

| sample | edges | native-step=1 plane records | matrix cells replayed | mismatches |
|---|---|---|---|---|
| 48 nueCC | 1480 | 4440 | 71040 | 0 |
| 19 NC-pi0 + 50 PR-data | 395 | 1185 | 18960 | 0 |
| **total** | **1875** | **5625** | **90000** | **0** |

Zero windows hit the 300000-cell cap in this sample (§3), so every plane
record was checked exhaustively -- no plane's evidence is asserted-but-not-
verified in the current 117-event dump.

## 6. The viewer (`overclustering_display/`)

Full layout and controls: [`overclustering_display/README.md`](../../overclustering_display/README.md).
In one sentence: three 3-D projections (component A blue, B red, event
context grey, the candidate edge as a dashed line), three U/V/W time-vs-wire
panels drawn on the fired/dead cells the BFS actually queried (green =
fired, amber = dead, open circles = seed footprints, black outline = search
window), an edge list defaulting to "removed only" sorted by a Python-side
long-track-break score (both pieces long by PCA, axes within 20 degrees,
facing endpoints close -- computed from the dumped points, so revising the
metric never costs a rerun), and a good/OK/bad + cause + comment label saved
per edge.

**Label key is geometric**, not `graph_call`/`j`/`k`: `(event, blk, p1, p2
rounded to 0.01 cm)`. `graph_call` is a per-process atomic counter and
`j`/`k` are cluster-local indices -- neither is stable across a rerun, so
keying on them would silently orphan every label the moment the dump is
regenerated. Saved to `overclustering_labels/<tag>/labels-evt<ID>.json`
(`*.json` is repo-gitignored, matching every other per-event dump in this
tree -- these are local scan records, not something this doc's commit
carries). Save is an upsert; it never touches any other saved entry (M13).

Verified end-to-end (not just import-clean): loaded a real dump, selected an
edge, populated all nine panels, saved a label, reloaded it from disk, and
confirmed the round-trip. The live server process on port 5018 returned
HTTP 200 with no server-side traceback on session creation.

## 7. Bee set

One upload, all 117 events, built via `scripts/bee/make_pr_bee.py` with
repeated `-q`/`-p` (each root searched independently per event, so three
differently-rooted samples merge into one command) and
`--allow-unevaluated` (the 50-event PR-data sample is not restricted to
nu-selected events, unlike `make_pr_bee.py`'s usual nueCC-only caller):

**https://www.phy.bnl.gov/twister/bee/set/25d75e54-32d3-4407-9bed-46a698da6f4f/event/list/**

Index -> event map: `bee/pr57/pr57-117.index.txt` (indices 0-47 = 48 nueCC,
48-66 = 19 NC-pi0, 67-116 = 50 PR-data, in the same order as the Repro
block's event lists).

## 8. Files touched

- `clus/src/connect_graph_relaxed_strict.cxx` -- the dump only (toolkit,
  committed, not pushed pending the owner's push decision, same as round
  3's own toolkit commit).
- `run_pr_evt.sh` -- `PR_OC56_SCAN_DUMP` hook (mirrors `PR_EXTRA_STAGES`);
  unused by the batch runner below but kept for the single-event path.
- `run_pr_chain_batch.sh` -- the same hook inside `process_event`'s
  subshell (this is the actual production batch path used to build every
  arm above).
- `overclustering_display/{overclustering_scan_viewer.py,
  serve_overclustering_scan.sh, README.md}` (new).
- `scripts/analysis/pr57/oc56_dump_check.py` (new) -- BFS-replay
  verification (§5) + `long_track_break_score()`, imported by the viewer.
- `bee/pr57/{pr57-117.index.txt,pr57-117.prid-map.txt}` (the `.zip` itself
  is repo-gitignored, matching every other Bee upload artifact in this
  tree).
- This doc.

## 9. Open items

- **No labels collected yet.** This round built and verified the
  instrument; the scan itself (the actual good/OK/bad judgements that were
  the point of doc pr/56 round 3's open question) is the owner's next step
  with the live server on port 5018.
- **Context points are "everything else this event's dump happens to
  carry," not "the whole event."** Only components that participated in
  *some* S6 evaluation get dumped at all -- a cluster that never came near
  another component in 2D never appears, even as grey context. This is
  usually enough to judge "is this one long track" locally, but is not a
  full-event view; the whole-event Bee link (§7) is the fallback when the
  grey context isn't enough.
- **300000-cell native-dump cap** (§3) was never hit in this 117-event
  sample; if a future sample does hit it, that plane's near-miss badge and
  fired-cell picture become non-exhaustive and the viewer flags it in the
  panel title, but `oc56_dump_check.py` currently just skips such plane
  records rather than checking them at the fallback stride -- worth
  extending if the cap ever actually bites.

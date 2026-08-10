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

**Round 2 (sec 10)**: the remaining 395 PR events of the 1000-event data
sample are processed (`work-pr57r2-scan395`), the first 50 are served on port
**5019**, and a machine first-pass label set calibrated on the owner's 575
round-1 labels is written to `overclustering_labels/claude-scan50/`. Sections
1-9 below are round 1 and are left as the record of that round.
**Round 3 (sec 11)**: the owner rescanned those 50 events and corrected 27 of
62 pairs -- my labels agreed on only 76.6 % of the good/bad pairs, with bad
recall 9/20. Root cause was the "W gap => good" rule, measured on 245
shower-rich nueCC/NC-pi0 pairs and applied to cosmic-dominated PR-data;
evt174224 c0 1-2, flagged in sec 10.6 as the case that would decide it, came
back **bad**. The classifier now carries a thin-and-collinear qualifier on that
rule and a promoted dead-W branch: 95.7 % (93.5 % CV) over all 602 labelled
pairs, bad recall 47/52.

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

---

## 10. Round 2 -- extension scan over the rest of the 1000-event data sample

Status: **395 remaining PR events processed, first 50 served on port 5019 with
a machine first-pass label set (`overclustering_labels/claude-scan50/`, 66
pairs -> 55 good / 11 bad).** No algorithm change, no toolkit change: this
round is arms, scripts and labels only.

### 10.1 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 1. the remaining 395 PR events of the 1000-event data sample.
#    NOTE the graph flavor: it must match work-pr58-scan50 (doc pr/58's
#    floor_w_override arm) or the new labels are not comparable with round 1's.
EV=$(tail -n +51 valfast/events-mcp1k-cb0805.txt)
PR_JOBS=32 SBND_PROTECT_GRAPH=relaxed_strict_img_2d_wfloor \
WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr57r2-scan395 data $EV \
  > work-pr57r2-scan395.driver.log 2>&1

# 2. the dump still does not lie (sec 5, on the new arm)
python3 scripts/analysis/pr57/oc56_dump_check.py work-pr57r2-scan395

# 3. calibrate the classifier on the owner's own labels, then label
python3 scripts/analysis/pr57/oc56_autoscan.py calibrate \
  --arm work-pr58-scan48 --arm work-pr58-scan19 --arm work-pr58-scan50
python3 scripts/analysis/pr57/oc56_autoscan.py label \
  --arm work-pr57r2-scan395 --tag claude-scan50 --first 50
python3 scripts/analysis/pr57/oc56_autoscan.py verify \
  --arm work-pr57r2-scan395 --tag claude-scan50

# 4. render pair panels for off-browser inspection
python3 scripts/analysis/pr57/oc56_render_pair.py \
  --arm work-pr57r2-scan395 --first 50 --out /home/xqian/tmp/oc57r2_panels

# 5. serve. Port is strictly $1 and --tag must follow it immediately.
./overclustering_display/serve_overclustering_scan.sh 5019 --tag claude-scan50 \
  $(awk -F'\t' 'NR>1{print $3}' docs/pr/pr57r2-scan50.index.txt)
# ssh -L 5019:localhost:5019 wcgpu1.phy.bnl.gov
# http://localhost:5019/overclustering_scan_viewer
```

Event list: `docs/pr/pr57r2-scan50.index.txt` (the 50 served events);
per-pair feature table: `docs/pr/pr57r2-scan50-pairs.tsv`.

### 10.2 How many events actually have PR results

Of the 1000 data events, **445 have real PR output** -- the discriminator is
the `TaggerCheckNeutrino: selected main cluster` log line, pinned in
`valfast/events-mcp1k-cb0805.txt`. All 1000 have `rc=0`, `pctree-pr-*.tar.gz`,
`mabc-pr.zip` and a `nusel` row, so **none of those is a "has PR" marker**;
`calib-pr-evt*.json` matches the 445 only on arms run with
`PR_EXTRA_STAGES=pr_display`.

Round 1's 50 data events are the first 50 of those 445; this round processed
the other **395**. Results: 395/395 `rc=0`, 1.4 GB, ~3 min wall at
`PR_JOBS=32` (loadavg peaked at 23 on 64 cores). 249 events wrote an
`oc56scan` dump and **184 are non-empty** -- an event only gets one when the
relaxed-strict graph is actually built and reaches S6.

`oc56_dump_check.py` on the new arm: **885 edges, 42480 matrix cells
replayed, 0 mismatches, PASS**.

### 10.3 What the owner's 575 labels actually say

Joined per-edge to the round-1 dumps, 575/575 with zero misses
(OK 458, good 71, bad 46; 536 pairs, of which 91 are good/bad):

| feature | bad | good | OK |
|---|---|---|---|
| `Lmin` = shorter component PCA length | med **36.7 cm** | med 3.5 | med 3.4 |
| `npmin` = smaller component npts | med **468** | med 31 | med 31 |
| W-plane gap present | **2 / 46** | **68 / 71** | 202 / 458 |
| local density (pts within 15 cm of the edge midpoint) | med 373 | med 816 | med **2033** |
| `d_vtx` (edge midpoint -> reco nu vertex) | med 20.7 cm | med 50.4 | med 60.2 |
| angle between component PCA axes | 47 deg | 44 deg | 44 deg |

Three findings worth keeping:

1. **The three prose rules are measurable.** "long track cut in the middle" =
   large `Lmin` **and** large `npmin` with no W gap; "busy EM shower" = high
   local density with short components; the axis-axis angle is useless
   (44-47 deg in all three classes -- do not build a rule on it).
2. **`d_vtx` alone does NOT mean good.** At pair level `d_vtx < 3 cm` is bad 9
   / good 5, and among the pairs where the long-track-break rule fires, **all
   8 labelled ones with `d_vtx < 3 cm` are bad**, at angles from 3 to 86 deg.
   The owner does not want two substantial prongs cut apart at the vertex
   either. So rule 2 must be evaluated *before* rule 3; a naive "near the
   vertex => good" override costs 8 of 32 bad recall (measured).
3. **A W-plane gap is very nearly an absolute "good" signal**: 253 pairs gap
   in W, and exactly **one** of them is labelled bad (evt137238). This is
   round 1's "W is the robust plane" stated as a number.

### 10.4 The classifier

`scripts/analysis/pr57/oc56_autoscan.py`. Verdicts are decided per component
**pair** `(event, graph_call, j, k)` and written onto every edge of that pair,
per the owner's round-1 point 2 (what matters is cluster-level separation, not
the individual edge); `verify` asserts no pair carries mixed verdicts.

```
R2   bad   Lmin > 6 cm and npmin >= 50 and no W gap        (long-track break)
R2d  bad   >=3 dead W wires spanning the seeds, no W gap,
           Lmax > 40 cm, dis < 3 cm                        (owner's case (a))
R3   good  W gap                                           (robust plane sees a hole)
R1   OK    local density > 2000 and Lmin <= 6 cm           (busy EM shower)
R4   good  everything else                                 (commit, don't hedge)
```

Measured against the owner's 575 labels (2026-08-10, 79 events):

- good/bad agreement **89/91 = 97.8 %** in-sample, **88/91 = 96.7 %** under
  leave-one-**event**-out cross-validation (refit per fold, so the honest
  number is the CV one);
- **good recall 59/59 = 100 %**, **bad recall 30/32 = 93.8 %**;
- on OK-labelled pairs: 290 good / 67 bad / 88 OK. Calling an owner-OK pair
  bad is sanctioned by round 1 ("if we judge some of these to be bad, it is
  OK"), so it is only a weak tie-breaker in the fit, never traded against a
  good/bad hit.

Those figures pool three samples, while the deliverable is 50 pure PR-data
events. Split by population, with the **shipped** operating point (not refit):

| sample | good/bad pairs | agreement | good recall | bad recall |
|---|---|---|---|---|
| mcp1k PR-data (matches the deliverable) | 17 | 16/17 = 94.1 % | 6/6 | 10/11 |
| nueCC + NC-pi0 | 74 | 73/74 = 98.6 % | 53/53 | 20/21 |

The operating point generalizes -- refitting on mcp1k alone gives different
thresholds (L=4, N=200, D=1200) but the *same* 16/17, so nothing is being
carried over from the neutrino-MC samples that the data sample contradicts.
Two caveats stay attached to the 94.1 %: n=17 is small, and the mcp1k sample
contains only **8** W-gap pairs (5 good, 3 OK, 0 bad) -- so R3's headline
evidence, 1 bad in 253 W-gap pairs, comes almost entirely from nueCC and
NC-pi0. R3 is the rule most in need of confirmation on PR-data, and
evt174224 c0 1-2 (sec 10.6) is the case that would confirm or break it.

The two misses are evt137238 (the single W-gap bad) and evt58717 (a 1.7 cm,
10-point stub). Four candidate rules to catch them were tried and **all are
refuted by the owner's own labels** -- recorded so they are not re-invented:

| candidate | effect |
|---|---|
| `d_vtx < 2 cm => good` | bad recall 30 -> 24 |
| kink at the vertex (`d_vtx < 3` and angle > 30) `=> good` | bad recall 30 -> 24 |
| both long+populated `=> bad` even with a W gap (recovers evt137238) | good recall 59 -> 54 |
| "soft" W gap (closes at larger `ds`) + long `=> bad` | good recall 59 -> 55, bad recall unchanged (24 such pairs: 4 good, 20 OK, **0 bad**) |

### 10.5 Visual audit

The classifier's own weak class is `bad`, so every pair it called bad was
inspected. `oc56_render_pair.py` draws the same content as the Bokeh page --
three 3-D projections with the two components, the grey context, the edge and
the reco neutrino vertex, plus the U/V/W wire-vs-slice panels built from the
dump's **own** fired/dead/seed cells -- as a PNG per pair.

**16 of the 66 pairs were looked at: all 11 predicted bad, plus the 5
highest-risk goods (large `Lmin` with a W gap, i.e. the evt137238 class).** The
other 50 pairs carry machine labels only. That is a partial audit and is not presented as
anything more.

Outcome: **0 verdict overrides.** Every disagreement between the panel and the
classifier was turned into a candidate rule and tested against the owner's
labels first -- and the labels refuted it every time (the table in §10.4 is
exactly that list). Concretely: evt61579 and evt66272 look like two prongs
meeting at the neutrino vertex, and evt170814 looks like a track broken across
a soft W gap; the owner's labels say all three classes are bad, bad and good
respectively, which is what the classifier already produced.

One threshold *was* moved by the audit: the dead-W branch floor `Ld` from 50 to
40 cm. The calibration set is indifferent (89/91 for every `Ld` in [20, 60]),
so this is the one number the owner's labels do not fix. At 50 it misses
evt167112 -- a 47.6 cm track continuing past a 6-wire dead-W band with V never
closing, which is the owner's case (a) drawn in full; at 30 it would also flip
evt169356, a 2.8 cm stub at a track end that should stay good.

### 10.6 The label set

`overclustering_labels/claude-scan50/labels-evt<ID>.json` -- 41 events, 66
pairs, 73 edge labels, in the viewer's exact schema and geometric key, so the
port-5019 page shows them and the owner can overwrite any of them in place.

- **55 good / 11 bad / 0 OK.** No pair in these 50 data events met the busy-EM-shower
  test (density > 2000 with short components) -- the round-1 OK class came
  overwhelmingly from the nueCC and NC-pi0 samples, not from PR-data events.
  Zero is the expected outcome rather than a defect: the density floor sits at
  the *median* of the calibration OK class, so the rule is deliberately
  conservative by construction, and a cosmic-dominated data sample has few
  busy showers to begin with.
- Every `comment` carries the rule that fired, a `conf=high|low` flag and all
  the numeric features, so any label can be audited or reversed without
  rerunning anything. 27 good and 1 bad are `conf=low` (the R4 residual and
  R2d branches); the other 38 are `conf=high`.
- The owner's own labels in `overclustering_labels/` root were **not touched**:
  the directory listing plus mtimes hash identically before and after
  (`1387f2c72942f439cec1c312ac80aba3`).

The 11 bad pairs are in `docs/pr/pr57r2-scan50-pairs.tsv`. **Start with
evt174224 c0 1-2**, which is *not* in that list: 84 cm and 175 cm components,
2164 and 1295 points, axes 4 deg apart, drawn as one straight 260 cm muon in all
three projections -- but W genuinely gaps (matrix all zeros), so R3 calls it
good. It is the closest analogue in this sample to evt137238, the one W-gap pair
the owner called bad. If that one is bad, the "W gap => good" rule needs a
qualifier and §10.4's third candidate comes back into play.

### 10.7 Experience notes

- **`PR_JOBS`, not `SBND_MAX_JOBS`.** `run_pr_chain_batch.sh:908` overwrites
  `BATCH_MAX` with `${PR_JOBS:-6}` immediately after `batch_init` read
  `SBND_MAX_JOBS`. `run_pr_evt.sh` is the opposite. Docs in this tree that say
  `SBND_MAX_JOBS=6 ./run_pr_chain_batch.sh` got 6 by coincidence.
- **`serve_overclustering_scan.sh` has no option parser.** Port is `$1`,
  `--tag` must be tokens 2-3, everything after is dump paths. Passing a glob
  first silently binds the server to a garbage port; and the script's built-in
  default glob still points at `work-pr56r4*-scan*`, which no longer exists --
  it then serves an empty viewer rather than failing.
- **Label keys are lossy by design.** The geometric key rounds to 0.01 cm while
  the entry's `p1`/`p2` are full precision, which is what lets labels survive a
  rerun. Any join back to a dump must use the rounded key.
- **The tab-close auto-save does not work** (round-1 defect, still present and
  deliberately not fixed here because the owner's port-5018 scan is live and
  bokeh re-executes the app module per session):
  `overclustering_scan_viewer.py:705` calls `flush_pending` from an
  `on_session_destroyed` lambda, but Bokeh clears the module globals first, so
  it raises `NameError` -- visible in `overclustering_display/serve5018.log`.
  **Click *Save event labels*, or switch events, before closing the tab.**
  Switching events flushes correctly (`load_event` -> line 384).

### 10.8 Open items

- 134 of the 184 events with dumps in `work-pr57r2-scan395` are not served or
  labelled. The arm is on disk and `--first`/`--events-file` will extend the
  scan without reprocessing.
- The classifier is calibrated on 91 good/bad pairs from 79 events. Its bad
  recall of ~94 % (CV 96.7 % overall) is what any downstream use of
  `claude-scan50` must be discounted by -- these are machine first-pass labels
  with a partial visual audit, not a hand scan.
- The owner's label set was still growing while this was calibrated; rerunning
  `calibrate` after the port-5018 scan finishes will move these numbers.

---

## 11. Round 3 -- the owner corrected the round-2 labels; what that taught

Status: **the owner rescanned the 50 events on 5019 and changed 27 of my 62
pairs.** My delivered labels agreed on **36/66 pairs (54.5 %)**, and on the 47
pairs that ended up good or bad, **36/47 = 76.6 %** with **bad recall 9/20** --
against the 97.8 % / 96.7 % CV the round-1 calibration had predicted. The
corrections are now truth; this section is the post-mortem and the fix.
`overclustering_labels/claude-scan50/` holds the owner's corrected verdicts
(my machine `comment` strings are preserved underneath, which is what made the
diff possible) and is **never rewritten by the tooling**.

### 11.1 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/pr57/oc56_autoscan.py calibrate \
  --arm work-pr58-scan48 --arm work-pr58-scan19 --arm work-pr58-scan50 \
  --arm work-pr57r2-scan395 \
  --labels overclustering_labels --labels overclustering_labels/claude-scan50
```

### 11.2 The shape of the errors

Edge-level, mine -> owner's (73 edges):

| my label | -> good | -> bad | -> OK |
|---|---|---|---|
| **good** (58) | 27 | **14** | 17 |
| **bad** (15) | **0** | 13 | 2 |

The asymmetry is the whole story. **Nothing I called bad came back good** --
when the classifier committed to bad it was never wrong in the dangerous
direction. Everything went wrong on the `good` side: I under-called bad by more
than a factor of two. For a scan whose purpose is finding bad separations, that
is the expensive failure.

By the rule that fired:

| rule | n | agreed | changed |
|---|---|---|---|
| R2 long-track break, no W gap | 14 | 12 | 2 -> OK |
| R2d dead-W band | 1 | 1 | -- |
| **R3 W-plane gap => good** | **31** | **16** | **10 -> bad**, 5 -> OK |
| R4 residual, committed good | 27 | 11 | 4 -> bad, 12 -> OK |

The `conf=high|low` flag I attached to every label was nearly worthless as a
triage aid: `conf=high` agreed 28/45 (62 %), `conf=low` 12/28 (43 %). It
tracked which rule fired, not whether the answer was right.

### 11.3 Root cause: R3 was measured on the wrong population

"A W-plane gap is very nearly an absolute good signal" (§10.3) was true of the
sample it was measured on and false of the sample it was applied to:

| sample | W-gap pairs | labelled bad |
|---|---|---|
| nueCC | 186 | 1 |
| NC-pi0 | 59 | 0 |
| PR-data round 1 | 8 | 0 |
| **PR-data round 2** | **28** | **7 (25 %)** |

§10.4 already flagged this ("R3 is the rule most in need of confirmation on
PR-data") and §10.6 named **evt174224 c0 1-2** as the case that would decide
it. That case came back **bad**, and so did five more W-gap pairs. The flag was
right; the classifier shipped with the rule anyway because the calibration
number said 96.7 %.

**The transferable lesson: leave-one-event-out CV inside one sample does not
estimate transfer to a different sample.** Every fold of that CV still contained
245 nueCC/NC-pi0 W-gap pairs, so every fold agreed R3 was safe. The honest
uncertainty was the 8-pair PR-data subsample, not the 91-pair pooled one. When a
rule's evidence is concentrated in one population, the per-population count is
the error bar -- and `calibrate` now prints a per-arm breakdown for exactly
this reason.

Physically, the sample difference is real, not a labelling artifact: nueCC and
NC-pi0 events are shower-rich, where a W gap between two substantial components
usually separates a shower from a track and is correct; PR-data events are
cosmic-dominated, where the same signature is usually a muon cut in two.

### 11.4 The fix

Three changes, each traceable to specific corrected pairs:

```
R2   bad   Lmin > 6 cm and npmin >= 50 and no W gap          (unchanged)
R2d  bad   >=3 dead W wires spanning the seeds, npmin >= 20,
           dis < 3 cm                                        (PROMOTED)
R2w  bad   W gap AND Lmin > 6 cm AND npmin >= 50
           AND Tmax < 2 cm AND axis-axis angle < 25 deg      (NEW)
R3   good  W gap                                             (now last-resort)
R1   OK    local density > 2000 and Lmin <= 6 cm             (unchanged)
R4   good  everything else                                   (unchanged)
```

- **R2d promoted**: the length floor `Ld` is deleted (evt167684 is a **7.4 cm**
  pair the owner called bad) and it now fires even when W itself gaps
  (evt60669, `wdeadX=4`). A dead-W band spanning the seeds is the signal by
  itself -- the owner's bad case (a) -- and does not need a long track to
  corroborate it.
- **R2w is the qualifier R3 needed**: a W gap only survives as "good" if the
  pair is *not* a thin, collinear pair of substantial components. `Tmax < 2 cm`
  is the track/shower discriminator (both pieces thin => one track), and
  `angle < 25 deg` is collinearity. This is what recovers evt174224 (Tmax 1.4,
  4 deg), evt172656 (1.7, 18 deg) and evt60017 (0.6, 1 deg) while leaving the
  fat or kinked nueCC/NC-pi0 W-gap goods (Tmax 12.7, 11.6, 6.7, 5.1 ...)
  untouched. It costs exactly one good: **evt122660 13-16** (nueCC), which is
  thin (`Tmax 1.7`), collinear (8 deg) and substantial (`npmin 170`) and so
  looks like a track pair by every feature R2w uses, but the owner called it
  good -- so "leaves the nueCC/NC-pi0 W-gap goods untouched" holds for the fat
  and kinked ones, not for that pair. It is also the branch that still does
  *not* recover evt137238 (`Tmax 4.8`, 48 deg), the round-1 W-gap bad: fat and
  kinked, so R2w declines by construction. R2w is the least-supported branch in
  the whole rule set -- 8 W-gap bads exist in total, and one counter-example
  already -- and is tagged `conf=low` accordingly.

The grid search over the extended family arrives at these thresholds
independently; they were not hand-set after the fact.

### 11.5 Measured, on 602 labelled pairs from all four arms

648 labels, 120 files (575 round-1 + 73 corrected round-2), 138 good/bad pairs:

| | agreement | good recall | bad recall |
|---|---|---|---|
| round-2 rule (what shipped) | 125/138 = 90.6 % | 86/86 | 39/52 = 75 % |
| **round-3 rule** | **132/138 = 95.7 %** | 85/86 = 98.8 % | **47/52 = 90.4 %** |
| round-3, leave-one-event-out CV | 129/138 = 93.5 % | | |

Per arm with the fitted parameters (not refit per arm), which is the check
§10.4 should have led with:

| arm | agreement | bad recall | good recall |
|---|---|---|---|
| `work-pr58-scan48` (nueCC) | 49/51 | 14/15 | 35/36 |
| `work-pr58-scan19` (NC-pi0) | 23/23 | 6/6 | 17/17 |
| `work-pr58-scan50` (PR-data r1) | 16/17 | 10/11 | 6/6 |
| `work-pr57r2-scan395` (PR-data r2) | 44/47 | 17/20 | 27/27 |

On the 50 corrected events alone: good/bad agreement **76.6 % -> 93.6 %**, bad
recall **9/20 -> 17/20**, good recall 27/27 both ways. That 93.6 % is in-sample
(these corrections are what the rule was fitted on) -- the honest figure is the
93.5 % CV.

**All six residual errors**, so the 47/52 and 85/86 above reconcile -- three
are on the round-2 fifty, three are elsewhere:

| sample | pair | truth | got | why |
|---|---|---|---|---|
| PR-data r2 | evt73004 1-2 | bad | good | tiny piece (npmin 21) off an 82 cm track |
| PR-data r2 | evt169356 6-7 | bad | good | tiny pair (npmin 12, 3.1/3.7 cm) |
| PR-data r2 | evt170814 0-1 | bad | good | `Tmax = 3.0`, just above R2w's thin cut |
| PR-data r1 | evt58717 0-1 | bad | good | 1.7 cm / 10-point stub, kinked (74 deg) |
| nueCC | evt137238 0-1 | bad | good | the original W-gap bad; `Tmax 4.8`, 48 deg -- fat and kinked, so R2w correctly declines |
| nueCC | evt122660 13-16 | **good** | **bad** | the one good R2w costs (below) |

The two tiny-piece misses sit below any `npmin` floor that would not also flood
the OK class. evt170814 would be recovered by widening R2w's thin cut to
3.5 cm, but that costs two nueCC goods (evt69314 at `Tmax 2.4`, evt269774 at
3.1), so it stays at 2.0.

### 11.6 What is still not handled: the OK class

19 of the 66 pairs are OK and the classifier produced **none** of them: 17 of my
goods became OK. All-pair agreement is therefore 66.7 % even with the round-3
rule, against 93.6 % on the good/bad subset. The density-based busy-shower test
does not describe what OK means on PR-data -- the corrected OK pairs run from
density 23 to 1203, i.e. sparse *and* dense. Since these labels are filtered out
before analysis, this costs nothing today, but "0 OK" is not a claim that no OK
cases exist; it is the classifier declining to model them.

### 11.7 Scanning-experience notes

- **The comment field is the audit trail that made this round possible.** The
  viewer preserves `comment` when only the verdict button changes, so every
  corrected label still carries `auto <verdict> conf=... [rule] <features>`.
  The diff of predicted-vs-corrected, per rule and per feature, fell out for
  free. Keep writing machine provenance into `comment`.
- **Flag the rule you distrust, name the event that would falsify it, and put
  it where it gets read.** §10.6's call-out of evt174224 is what turned a wrong
  answer into a one-scan diagnosis instead of a silent error.
- **Report per-population, never pooled.** A single agreement number over three
  samples hid a rule that worked on two of them and failed on the third.
- **Commit direction matters more than commit rate.** 0 of 15 bads came back
  good. Given the owner examines both classes, a classifier that under-calls
  bad is worse than one that over-calls it; the round-3 rule trades 1 good for
  8 bads and that is the right direction for this task.

## 12. Round 4 -- pair-level connectivity: what the scan was really labelling

**Repro block** (all numbers in this section):

```bash
cd sbnd/sbnd_xin
# 1. the four re-run arms (toolkit built with the round-4 dump record)
for a in "48 work-nuecc48-cb0805" "19 work-ncpi0-cb0805" \
         "50 work-mcp1k-cb0805" "395 work-mcp1k-cb0805"; do
  set -- $a
  PR_JOBS=32 SBND_PROTECT_GRAPH=relaxed_strict_img_2d_wfloor \
  WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \
    ./run_pr_chain_batch.sh $2 work-pr57r4-scan$1 data $(cat <event list>) \
    > work-pr57r4-scan$1.driver.log 2>&1
done
# 2. gates
python3 scripts/analysis/pr57/oc56_conn.py --selftest
python3 scripts/analysis/pr57/oc56_dump_check.py --conn-only work-pr57r4-scan{48,19,50,395}
# 3. the displays
./overclustering_display/serve_overclustering_scan.sh 5018 <96 dumps>
./overclustering_display/serve_overclustering_scan.sh 5019 --tag claude-scan50 <50 dumps>
```

### 12.1 The problem: `killed` is not the question the scan asks

The display showed one **candidate edge** at a time and one bit about it,
`killed`.  The hand scan judges something else: should these two **components**
be apart.  Those come apart in two ways, both common:

1. A component pair is offered **up to three independent candidates** --
   `closest`, `dir1`, `dir2` (`connect_graph_relaxed_strict.cxx`, the
   `two_d_gap_kill` call sites).  Killing one says nothing about the others.
2. Even with every (j,k) candidate killed, j and k can stay in one piece
   **through a third component**, j -- m -- k.

So a row reading `killed=yes` can leave the two clusters fully joined, and a
"bad, this track was cut" verdict then describes a cut that never happened.

### 12.2 The new record

`connect_graph_relaxed_strict.cxx` now writes a third JSONL record type, once
per graph call, after the emit loop -- the earliest point at which the answer
exists, and the reason this is a separate record rather than a field on the
(much earlier) per-candidate records:

```json
{"type":"connectivity","graph_call":N,"cluster_id":C,"ncomp":12,"nfinal":4,
 "final":[0,0,1,0,...],
 "edges":[{"j":1,"k":5,"src":"mst","dis":1.10,"p1":[..],"p2":[..],"dup":false}, ...]}
```

- `final[c]` -- the final connected-component label of starting component `c`,
  from `connected_components()` on the graph itself.
- `edges[]` -- every component edge the call emitted, accumulated independently
  at the three `add_edge` sites, with endpoints in the same frame as the dumped
  component point clouds.  `dup` marks an emit whose vertex pair already
  carried an edge.

The two are computed independently, so `final[j]==final[k]` **must** hold
exactly when j reaches k over `edges[]`.  That is gate 4 below, not a
tautology.

Everything is inside `if (oc56_dump_on)`; with `WCT_OC56_SCAN_DUMP` unset not
one instruction of it runs.  The only edits outside those guards are three
`if (!boost::edge(...).second)` rewritten as `const bool dup = ...; if (!dup)`,
which is the same call, once, with its result named.

`scripts/analysis/pr57/oc56_conn.py` turns a record plus a pair into
`separated / direct / path / siblings`, and is imported by the viewer,
`oc56_autoscan.py` and `oc56_dump_check.py` so all three agree by construction.
It has a `--selftest` (direct, via-2-hops, separated, and a pre-round-4 dump
degrading to unknown) -- it is the one piece whose silent bug would corrupt
every comparison built on it.

### 12.3 The display

New **`pair`** column next to `killed`: `SEP` (really separated) / `dir` (this
pair kept a direct edge anyway) / `via N` (joined through N hops) / `?`
(pre-round-4 dump -- never silently read as "connected").  Under the
projections, a line naming the surviving edge or the full hop chain, plus this
pair's other candidates and their own `killed` verdicts.  The surviving edges
are drawn **green** in the three 3-D views.  A `separated pairs only` toggle
(default OFF) narrows the table to the true separations.

One caveat on reading it: the chain shown for `via N` is the **shortest**
route.  A pair joined by several independent routes shows only one of them, so
the green line is *a* thing holding the two components together, not
necessarily *the* thing -- killing it need not separate anything.  `SEP` is
unaffected: it comes from `final[]`, i.e. the graph's own component labels, not
from the route.

Every event that carries a label is in the served list on both ports (79 of the
96 on 5018, 41 of the 50 on 5019) -- checked explicitly after the restart, since
"the labels re-join a dump" and "the labelled event is actually served" are
different claims.  The 96 (rather than 117) is pre-existing: 21 of the round-1
events produced no S6 candidate at all and were never in the 5018 argv.

### 12.4 The headline: a third of the `bad` labels were not separations

648 labels, all joined to the re-run dumps with **0 orphans**:

| set | verdict | n | really separated | killed but still joined | never killed at all |
|---|---|---|---|---|---|
| owner 5018 | **bad**  | 46  | **31 (67.4%)** | 7  | 8 |
| owner 5018 | good | 71  | 70 (98.6%) | 1  | 0 |
| owner 5018 | OK   | 458 | 283 (61.8%) | 112 | 63 |
| claude-scan50 (owner-corrected) | **bad** | 27 | **25 (92.6%)** | 2 | 0 |
| claude-scan50 | good | 27 | 26 (96.3%) | 0 | 1 |
| claude-scan50 | OK | 19 | 15 (78.9%) | 0 | 4 |

**15 of 46 `bad` labels on the 5018 set (33%) sit on a pair the code never
separated** -- and 8 of those 15 are rows where nothing was removed at all
(below the 1 cm floor, surfaced by the `killed_ignore_floor` filter, so S6
*would* have killed them but did not).  `good` is almost untouched: 70/71 are
real separations.  The error is one-sided and concentrated exactly where round
3 already found the calibration was weakest.

Concrete cases:

```
evt142421 closest killed=True  1.62cm -> STILL CONNECTED via 6 -> 9 -> 8   [6-9 mst 31.76cm, 9-8 mst 0.92cm]
evt142421 dir2    killed=True  4.88cm -> STILL CONNECTED, direct mst edge dis=0.92 cm
evt256587 closest killed=True  1.77cm -> STILL CONNECTED via 22 -> 25 -> 32 [22-25 mst 0.92cm, 25-32 mst 0.95cm]
```

Any rule fitted on `bad` without this filter was fitted on a population a third
of which was mislabelled by construction, not by judgement.  The round-3
thresholds should be refit on `bad AND SEP` before they are trusted further --
**that refit is not done here**; this round delivers the information, not a new
operating point.

### 12.5 The blind spot the display still has

An edge record exists only for candidates that reached `two_d_gap_kill` (S1-S5
let them through, same APA/face, at-or-above the 1 cm floor or W-override).  So
a pair can end up separated with **no row in the table at all**.  Measured over
524 graph calls / 15971 component pairs in the four arms:

- 12587 pairs ended separated; **11418 had no dumped candidate edge**.
- Of those 11418, 7489 involve a component whose point cloud is not in the dump
  either (it never took part in any dumped candidate), so their separation
  distance is **not measurable from the dump** -- stated, not guessed.
- Of the 3929 that are measurable: 0 below 1 cm, **96 at 1-3 cm, 142 at
  3-5 cm**, 714 at 5-15 cm, 2977 beyond 15 cm.

So within the viewer's own default reach (`dis <= 5 cm`) there are **238
separations the scan can never show**, against 2983 dumped candidate edges.
The bulk of the 11418 is components that were never plausibly connected --
the raw 90.7% figure is not a 90.7% blind spot and should not be quoted as one.

### 12.6 Gates

| gate | result |
|---|---|
| 1. runs | 512/512 `rc=0` across the four arms |
| 2. **label join** (edge-key sets old vs new, per event) | **0 differences over 2983 edge keys**; all 648 labels re-join, 0 orphans |
| 3. physics outputs (`hash_archive.py` member content) | **1024/1024 archives identical** (`mabc-pr.zip` + `pctree-pr`), 512/512 `nusel` identical |
| 4. record self-consistency (`final[]` vs BFS over `edges[]`) | **0 disagreements** over 524 graph calls / 15971 pairs |
| 5. blind-spot census | §12.5 |
| 6. unit tests | `wcdoctest-clus` 1593/1593; `oc56_conn.py --selftest` PASS |
| 7. owner labels untouched | file list + content hash identical before/after restart, both label dirs |
| 8. displays | 5018 and 5019 HTTP 200, no exceptions in the serve logs |

**Reproducibility control, run before the 512.**  SCN/DL vertex is the SBND
default and is not bit-stable (M4), and the standard `-A dl_weights=` gate
recipe is unavailable here because the arms must correspond to the owner's
labels.  So gate 3 alone would have been a joint test of the change, the tree
state and run-to-run noise.  Two events (59929, 60017) were therefore run
**twice with the new binary** into `work-pr57r4-ctlA` / `ctlB`: new-vs-new and
new-vs-old both identical on all three output kinds.  Only then was gate 3
interpretable -- and it passes.

One trap on the way: comparing `hash_archive.py` output verbatim reports DIFF
for identical archives, because the output line ends with the file path.  Hash
**field 1 only** (`awk '{print $1, $2}'`).  Same shape as round 2's `ls`
path-prefix false alarm.

### 12.7 Arms, displays, files

| arm | ql_root | events | replaces |
|---|---|---|---|
| `work-pr57r4-scan48` | `work-nuecc48-cb0805` | 48 | `work-pr58-scan48` |
| `work-pr57r4-scan19` | `work-ncpi0-cb0805` | 19 | `work-pr58-scan19` |
| `work-pr57r4-scan50` | `work-mcp1k-cb0805` | 50 | `work-pr58-scan50` |
| `work-pr57r4-scan395` | `work-mcp1k-cb0805` | 395 | `work-pr57r2-scan395` |
| `work-pr57r4-ctlA` / `ctlB` | `work-mcp1k-cb0805` | 2 each | reproducibility control |

Old arms are untouched and remain valid for anything already cited against
them.  **5018** serves the same 96 events as before (47 + 18 + 31), no `--tag`;
**5019** the same 50, `--tag claude-scan50`.  The serve argument lists were
built by string-substituting the arm name into the argv of the running
processes (`/proc/<pid>/cmdline`), never by re-globbing -- 5018's list is not
"every event in the arms" and a glob would have silently changed the set.

Files: `clus/src/connect_graph_relaxed_strict.cxx` (toolkit);
`scripts/analysis/pr57/oc56_conn.py` (new), `oc56_dump_check.py`
(`--conn-only`), `oc56_autoscan.py` (`pair` feature/column),
`overclustering_display/overclustering_scan_viewer.py`,
`overclustering_display/README.md`.

### 12.8 Experience notes

- **Check what the label is attached to before fitting anything on it.** Three
  rounds of rule-fitting ran on a `bad` population a third of which was not a
  separation.  Nothing about the display made that visible; the information
  simply was not dumped.
- **When two things must agree, compute them independently and check.**
  `final[]` from the graph vs `edges[]` from the emit sites cost ~15 lines and
  turns "the display's reachability logic is probably right" into a measured 0
  over 15971 pairs.
- **Hash the hash, not the line.** `hash_archive.py` prints the path; including
  it manufactures a DIFF on identical files. Second time this class of bug has
  cost time on this doc (round 2's `ls` prefix).
- **`A && B & C` backgrounds `A && B` and runs `C` in the original cwd.** The
  second `serve` invocation silently started in the wrong directory. Use
  absolute paths in backgrounded compound commands.
- **A raw census ratio can be true and still misleading.** "90.7% of
  separations have no dumped edge" reads like a catastrophic blind spot; the
  distance breakdown turns it into 238 cases within the display's own reach.
  Qualify before quoting.

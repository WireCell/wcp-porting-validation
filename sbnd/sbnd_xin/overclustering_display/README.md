# SBND overclustering-separation hand-scan display

A Bokeh event display for judging, by eye, whether each edge S6 (the
default-OFF `relaxed_strict_img_2d` 2D wind/tick connectivity check, doc
pr/56) would REMOVE from a cluster's candidate graph is a real gap or an
induction-plane signal inefficiency -- and for saving that judgement as a
label, so the labels accumulate into the statistics the next round of S6
tuning needs.

Full write-up: [`../docs/pr/57_overclustering-separation-scan.md`](../docs/pr/57_overclustering-separation-scan.md).

## Why this exists

doc pr/56 round 3 shipped a repaired S6 with a 1 cm distance floor. The
honest finding of that round: at the shipped operating point, S6 removes
1433 of 1875 evaluated edges across the 48 nueCC + 19 NC-pi0 + 50 PR-data
sample, and most comparable clusters come out MORE fragmented than
production. Whether that is right cannot be settled by aggregate counts --
each removal is either a real gap (S6 is correctly cutting a spurious
2D-projection bridge) or an artifact of induction-plane signal inefficiency
(S6 is wrongly breaking a real track). Only a human looking at the actual
charge pixels can tell which, and this display exists to make that judgement
fast, informed, and recorded.

## 1. Produce the dump

The viewer reads a per-event JSONL dump written by an env-gated block inside
`connect_graph_relaxed_strict.cxx` (`Oc56DumpWriter`, `WCT_OC56_SCAN_DUMP`):

```bash
PR_JOBS=6 SBND_PROTECT_GRAPH=relaxed_strict_img_2d \
WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \
  ./run_pr_chain_batch.sh <ql_root> <out_root> data <evt ...>
# -> <out_root>/pr_evt<ID>/oc56scan-evt<ID>.jsonl
```

`PR_OC56_SCAN_DUMP` and `WCT_OC56_SCAN_DUMP` are both empty by default, so
every other run of the PR chain is byte-identical whether or not this display
exists (doc pr/57 sec 4: dump-OFF gate PASS 0/117). `WCT_RELAXED_EDGE_CENSUS`
is not strictly required for the dump itself, but without it the near-miss
`dw x ds` matrix badges in the edge list are empty -- always set both.

**Before trusting any label collected against a dump, verify it does not lie**:

```bash
python3 scripts/analysis/pr57/oc56_dump_check.py <out_root>
```

replays the same bounded BFS `connect_graph_relaxed_strict.cxx` runs, entirely
from the dumped fired/dead cells, at every logged `(dw, ds)` operating point,
and asserts the result matches the logged connectivity matrix. See doc pr/57
sec 5 for what this actually checks and why it matters.

## 2. Serve

```bash
./overclustering_display/serve_overclustering_scan.sh 5018 \
  work-pr56r4b-scan48/pr_evt*/oc56scan-evt*.jsonl \
  work-pr56r4b-scan19/pr_evt*/oc56scan-evt*.jsonl \
  work-pr56r4b-scan50/pr_evt*/oc56scan-evt*.jsonl
```

With no explicit globs, the script defaults to
`../work-pr56r4*-scan*/pr_evt*/oc56scan-evt*.jsonl`. Pass `--tag NAME` (before
the globs) to namespace saved labels into `overclustering_labels/NAME/` --
useful for keeping two people's scans of the same sample apart.

From a laptop:

```bash
ssh -L 5018:localhost:5018 wcgpu1.phy.bnl.gov
# then open http://localhost:5018/overclustering_scan_viewer
```

Port 5018 is the next free one after img 5013 / pd 5014 / ql_scan 5015 /
wf_scan 5016 / pr_display 5017.

## Display layout

**event selector** -- every event that has a dump file (all 117 in the
current sample). `filter: removed only` (default ON) shows just the edges S6
would kill; toggle off to also label survivors (false negatives matter too).
`sort by` defaults to the long-track-break score.

**Row 1 -- X-Y, Y-Z, X-Z.** Component A (blue) and component B (red) that the
selected edge candidates for a merge, every other point this event's dump
happens to carry in grey (context only -- not necessarily the whole event;
see the doc for what's actually in scope), the edge itself as a black dashed
line with `x` markers at its two closest-approach points. `zoom to edge`
frames a box (at least ±30 cm) around the edge midpoint; `whole event` resets
to the full detector volume.

**Row 2 -- U-T, V-T, W-T**, for the edge's own APA/face. Green cells are real
hits (what the BFS could step on); amber bands are dead-channel ranges;
open circles are the two components' own seed footprints (same colours as
row 1); the black outline is the BFS search window. This is the panel that
actually answers "real gap vs. induction inefficiency" -- if the space
between components A and B is amber (dead) rather than blank, the "gap" is a
channel outage, not missing physics.

**Edge list** -- one row per S6-evaluated edge in the current event:
which planes gapped, distance, verdict, a **near-miss badge** read straight
off the logged `dw x ds` connectivity matrix (e.g. `U closes @ dw=2` means
widening U's wire-adjacency by one step would have rescued this edge -- free
information, and it turns a bare yes/no label into tuning data), the
**long-track-break score** (`scripts/analysis/pr57/oc56_dump_check.py`:
`long_track_break_score()` -- both pieces long by PCA length, axes nearly
collinear, facing endpoints close; a SORT KEY, not a verdict), and the
current label if one has been saved.

**Labels** -- pick **good** (removal correct, real gap), **OK** (removal
defensible but marginal / ambiguous), or **bad** (removal WRONG, a real track
got broken), a **cause** (induction inefficiency, dead channel, prolonged
shower signal, genuine separation, other), and an optional comment, then
**Save label**. Saved to
`overclustering_labels/<tag>/labels-evt<ID>.json`, keyed on
**geometry** -- `(event, blk, p1, p2 rounded to 0.01 cm)` -- not on
`graph_call`/`j`/`k`: those are not reproducible across reruns (`graph_call`
is a per-process atomic counter; `j`/`k` are cluster-local indices that
collide across clusters in the same event, exactly the key doc pr/56 round 3
had to stop using for the census log for this same reason). Save is an
upsert -- it never touches any other saved label in the file.

## Conventions

Same as `pr_display/`: positions in cm; wire/time panels use the fitted
window's own wire-index and time-slice units (no fractional-wire subtlety
here -- this display draws raw fired/dead cells, not a fitted trajectory);
the fired-cell lattice is at **native granularity** (`native_step=1`, i.e.
every integer slice, not strided by the edge's own `slice_step`) unless the
per-plane window would exceed a size cap, in which case the panel title says
so and that plane's near-miss/label evidence should be treated as
non-exhaustive.

## Not here yet

Any change to S6 itself -- this is read-only viewing plus labeling. Batch
label-vs-tuning aggregation (turning saved labels into an updated operating
point) is future work once enough labels exist.

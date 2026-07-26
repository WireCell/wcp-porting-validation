# 58 — The nusel scan viewer's bundle table showed the previous event's bundles

Viewer-only bug fix (`nusel_display/nusel_scan_viewer.py`). No C++, no jsonnet,
no reconstruction output is involved, so there is no A/B gate here; the bar is a
headless before/after on the transitions that failed.

## Repro block

Serve a **scratch** scan tag — case 3 of the test clicks a scan label, which
autosaves, and a live hand-scan tag must never be written into (M13):

```bash
cd wcp-porting-img/sbnd/sbnd_xin
SB=$PWD
PREV=""; for t in d55ton d52ron d49son; do for s in mcp10 mcp1000 mcp1000b; do
    PREV="$PREV --prev $SB/work-$s-$t:$t"; done; done
setsid nohup ./nusel_display/serve_nusel_scan.sh 5099 --tag tblrefresh \
    --charge-src pr $PREV \
    $SB/work-mcp10-d56bw $SB/work-mcp1000-d56bw $SB/work-mcp1000b-d56bw \
    > /home/xqian/tmp/serve5099.log 2>&1 < /dev/null &

python3 nusel_display/test_table_refresh.py \
    http://localhost:5099/nusel_scan_viewer d56bw      # exit 0 = 4/4 pass
```

The test drives a real headless chromium (playwright, browser under
`~/.cache/ms-playwright`) and asserts on the SlickGrid's own DOM cells.

The production viewer is served the same way on 5011 with `--tag d56bw`
(doc 56 §Repro). **Absolute** work roots — bokeh resolves relative paths against
the server's cwd and silently serves an empty page (doc 56 GOTCHA 1).

## Symptom

Reported by the owner while hand-scanning the doc-56 beam-window scan on 5011:
stepping `evt287517 → evt287759 → evt287825`, the bundle table kept listing
evt287517's bundles. The charge projections and every other panel followed the
event correctly — only the table was stale. evt287759 has **no** in-beam bundle
at all, so in in-beam-only mode it should have listed nothing.

Confirmed headless against the unmodified viewer (`test_table_refresh.py`, before
arm served from `git show HEAD:...` so the shared tree was never touched):

```
--- 287517 in-beam-only: 1 row(s)
    0 | 9 | 2.011 | 14912 | Y | 13 | 13 | 2171
--- 287759 in-beam-only (expect NO rows): 1 row(s)
    0 | 9 | 2.011 | 14912 | Y | 13 | 13 | 2171     <-- 287517's bundle
--- 287825 in-beam-only: 1 row(s)
    0 | 9 | 2.011 | 14912 | Y | 13 | 13 | 2171     <-- still 287517's bundle
FAIL C1: 287759 still shows 287517's row(s)
FAIL C1: 287825 still shows 287517's row(s)
```

## Root cause

Bokeh 3.9's `DataTable` **never listens to its source's data**. From
`bokeh/server/static/js/bokeh-tables.js`, `DataTableView.connect_signals`:

```js
// changes to the source trigger the callback below via
// compute_indices hooks in cds view
this.connect(this.model.view.change, () => this.updateGrid());
```

The only repaint channel is the `CDSView`'s change signal. `CDSViewView`
recomputes on `source.properties.data.change`:

```js
compute_indices() {
    const size = source.get_length() ?? 1;
    const indices = types_1.Indices.all_set(size);
    indices.intersect(this.model.filter.compute_indices(source));
    this.model.indices = indices;          // <-- guarded assignment
}
```

and that assignment goes through the ordinary change detection
(`HasProps._setv`: `if (check_eq === false || prop.is_unset ||
!is_equal(prop.get_value(), value))`). An all-rows-set `Indices` of the **same
length** compares equal, so no signal is emitted and `updateGrid()` — which is
what re-reads the source (`this.data.init(...)` + `grid.invalidate()`) — never
runs. The client's `ColumnDataSource` holds the new rows; the grid keeps drawing
the old ones.

Two things made this bite here rather than stay latent:

1. `rebuild_table()` emitted `dict(cols)`, and `cols` is a `defaultdict(list)`,
   so **no visible rows produced an empty dict** `{}`. On the client, an empty
   data dict makes `get_length()` return `null`, which `compute_indices` reads
   as `?? 1` — size 1. So the 0-row state is indistinguishable from a 1-row
   state, and in in-beam-only mode (1 in-beam bundle per event, doc 56) *every*
   transition compared equal: 1 → 1(fallback) → 1.
2. Equal visible row counts happen constantly in all-bundle mode too
   (`286065 → 286197` is 9 → 9).

The client-side probe pins it exactly — the data arrived, the view did not move,
the DOM stayed behind:

```
--- 287759 in-beam-only (0 rows expected)
    client CDS: len=None ncols=0            <-- the {} push
    CDSView: indices={'size': 1, 'count': 1}
    DOM rows=1 first=['0', '9', '2.011', '14912']
--- 287825 in-beam-only (1 row expected)
    client CDS: len=1 t_us=['1.410']        <-- correct new data on the client
    CDSView: indices={'size': 1, 'count': 1}   <-- unchanged => no repaint
    DOM rows=1 first=['0', '9', '2.011', '14912']
```

No JS exception, no server traceback — the update was simply never requested.

## Why it hid

- **The previous "fix" could not work.** A comment dated 2026-07-23 claimed the
  same-row-count case was handled by setting `.data` twice to force a row-count
  change. `Document` coalesces that: `ModelChangedEvent.combine()` overwrites
  `self.new` when model+attr match, so the client only ever received the final
  value — one event, exactly as if the workaround were absent. It was inert from
  the day it was written, and the event pair it cited (`286065→286197`) is a 9→9
  transition, i.e. precisely the case it was supposed to prove.
- **All-bundle navigation looks fine anyway.** Walking all 30 events in
  all-bundle mode passes even before the fix (case 4, 30/30): the surrounding
  Divs change size, Bokeh reflows the layout, and `DataTableView._after_layout`
  → `resizeCanvas()` → SlickGrid `render()` happens to re-read the provider.
  That incidental path is what masked the defect. It does not fire for the
  single-row in-beam-only table, which is the mode the doc-56 gate made the
  natural way to scan.
- **Everything else on the page was right**, so the table looked like a data
  problem (a stale TSV, a caching `Event`) rather than a rendering one.

## Fix

`nusel_display/nusel_scan_viewer.py`, +44/−8:

- `TABLE_FIELDS` = every column the table writes (`table_cols` fields + the
  `sel_bg` tint), and `rebuild_table()` now emits `{k: cols.get(k, []) for k in
  TABLE_FIELDS}` — **all columns always, empty lists when there is nothing to
  show**. `get_length()` is then 0, not `null`, so the empty state is a genuine
  0-row state.
- `table.view = CDSView(filter=ALL_FILTERS[0])` with two interchangeable
  `AllIndices()` instances, and `touch_table_view()` flips between them at the
  end of every `rebuild_table()`. Changing the view's filter changes the
  `CDSView` model, which always emits `view.change` → `updateGrid()`,
  independent of whether the indices happen to compare equal. Two fixed
  instances rather than a fresh filter per call: `rebuild_table()` also runs on
  every label click, so a new model per call would accumulate all scan long.
- Order matters and is commented: `.data` first, `touch_table_view()` second.
  Both land in one patch message and are applied in order, so `updateGrid()`
  must run after the new rows are in place.
- The inert two-assignment trick is gone, with the coalescing explained in place
  so it does not come back.

It also puts a second path on solid ground. `rebuild_table()` runs on every
label / comment / confirm action, always with the row count unchanged, so the
`scan`, `✎`, `prev ✓` cells and the green row tint were repainted only by the
same incidental layout reflow — they were never *requested* to update. That
happened to work (case 3 passes in the before arm too) and now does not depend on
luck. Case 3 keeps watching it.

## Verification

Same test, same command, both arms (`d56bw` manifest, 30 events):

| case | what | before | after |
|------|------|--------|-------|
| 1 | in-beam-only `287517 → 287759 → 287825` (1→0→1 rows) | **FAIL** ×2 | PASS |
| 2 | all bundles `286065 → 286197` (9→9 rows) | pass (incidental reflow) | PASS |
| 3 | scan-label click repaints the focused row | pass¹ | PASS |
| 4 | all 30 events, row count + row 0's t(us) vs the nusel TSV | 30/30 | 30/30 |

¹ case 3 asserts the row's cells changed, which the incidental reflow also
satisfies; the `scan` cell content check (`'' → 'TGM'`) was added with the fix.

After-arm case 1, on the production 5011 instance:

```
--- 287517 in-beam-only: 1 row(s)   0 | 9 | 2.011  | 14912 | Y | 13 | 13 | 2171
--- 287759 in-beam-only:  0 row(s)  <no rows>
--- 287825 in-beam-only: 1 row(s)   0 | 5 | 1.410  | 10057 | Y |  5 |  5 | 2199
```

and the probe confirms the mechanism now moves: `indices` 12 → 1 → 0 → 1 with
the filter alternating `p1816 / p1817`, `ncols=16` throughout (never 0).

Nothing else changed: no verdict, no TSV, no `work/` product was touched — the
30-event walk reproduces every table from the same TSVs the run produced.

## Commits

The fix, the test and this doc were **swept into `3c831f2`**
("sbnd_xin/docs: 57 -- drop the A/B arm-location line from section 5a-bis"),
which is a different session's doc-57 edit: two sessions share this repository's
`.git` **and its index**, so their `git commit` picked up files this session had
just staged, and it was pushed before that could be noticed.  The content is
exactly what was intended and was verified byte-identical on `origin/main`; only
the commit message is misleading.  It was NOT repaired -- `3c831f2` is public
history (see doc 56 GOTCHA 4, which records the same trap in the other
direction).

Rule for this repository: stage explicit paths, check `git diff --cached --stat`
immediately before committing, and expect the index to move under you.

## Notes / leftovers

- The before arm was served from `git show HEAD:...` into a scratch directory,
  never by checking out over the shared working tree (a second session works in
  this repo).
- The test run left two scratch label tags behind,
  `work-*-d56bw/nusel_labels/{tblrefresh,tblrefresh_before}/`, holding
  autosaved state for 3 events plus one synthetic `TGM` label from case 3. They
  are inert (the viewer only reads the tag it is served) and were deliberately
  **not** deleted — nothing under `nusel_labels/` is removed without the owner
  saying so. Remove them whenever convenient; the live `d56bw` tag is untouched.
- `test_table_refresh.py` hard-codes the three event IDs of cases 1–3 (they are
  the ones that failed) and so expects the 30-event MCP2025C manifest; case 4
  adapts to whatever roots the server was given.

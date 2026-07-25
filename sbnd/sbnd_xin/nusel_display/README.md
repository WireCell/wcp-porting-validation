# SBND nusel hand-scan viewer

A Bokeh event display for checking/scanning the PR-chain tagger verdicts
(TGM / STM / FC) bundle by bundle, using the **same inputs Bee shows** so a
Bee tab can be kept open next to it:

* charge points = the `clustering-global` layer of `ql_evt<ID>/mabc-all-apa.zip`
  (what `merge_mabc_bee.py` uploads to Bee: T0-corrected coordinates,
  per-point `cluster_id`)
* light = the `op` layer of the same zip (`op_pes` measured / `op_pes_pred`
  predicted, per flash)
* bundle structure (main + companion clusters, flash gid/time/group) = the
  post-QL pctree via `nusel_extract.parse_pctree` — the same code that builds
  `nusel-table.tsv`, so ids and flash groups agree with the table by
  construction.

## 1. Produce the inputs

```bash
SBND_INPUT_DIR=... SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
```

leaves per event: `nusel_evt<ID>/nusel-evt<ID>.tsv` (scan rows + verdicts),
`ql_evt<ID>/mabc-all-apa.zip` (Bee layers), `ql_evt<ID>/pctree-evt<ID>.tar.gz`.

## 2. Serve

```bash
./serve_nusel_scan.sh 5010 --tag mcp10 ../work-mcp10
```

From a laptop:

```bash
ssh -L 5010:localhost:5010 <host>
# then open http://localhost:5010/nusel_scan_viewer
```

## 3. Scan workflow

1. **prev/next evt** buttons step events; within an event, click a bundle row
   (or use the **< / > bundle** buttons — one row per physical flash group,
   the same stepping as Bee's `> <`).
2. The three projections (X-Y, Y-Z, X-Z) show the whole event in gray, the
   focused bundle's **main cluster in red** and its companion (same-flash)
   clusters in blue; switch `point color` to `charge` for a viridis charge
   view.  The solid red box is the SBND detector (`sbnd_pr_fv`, with the
   cathode line at x=0); the dashed green box is the **effective fiducial
   volume** the taggers test (margins −2/−2.5/−3 cm in x/y/z).
3. **Merge components.** QLMatching's `examine_bundles` flash merge can fuse
   several physically separate clusters into one cluster object, so the main
   cluster can appear as several disconnected pieces.  The dominant
   (most-points) component is drawn as **red circles**, the grafted fragments
   as **orange squares**; the metrics panel lists every component with its
   point count and length, and hovering a point reports its `cluster` and
   `merge comp.` (= `real_cluster_id`).  This is a *provenance* distinction,
   not a scope one: the taggers run **once on the whole merged cluster**, so
   the orange pieces are inside the object that produced the TGM/STM/FC
   verdict.  Companion clusters (blue) are genuinely different clusters that
   merely matched the same flash — they are tagged separately, and only
   main-flagged clusters get a table row.  The table's `len(cm)` is the
   dominant component's length (the `n_frag` convention of
   `nusel_extract.parse_qlbee`), not the merged extent.
4. The light row shows the flash's measured vs predicted pattern on both TPC
   sides (own-side flash + its 80 ns cross-APA partner), as PMT z-y bubble
   panels plus a per-channel overlay.
5. **STM dQ/dx panel** (needs `run_nusel_evt.sh -stm-fit` + `uproot`): the
   fitted dQ/dx vs residual range of the focused bundle's **main cluster**,
   over the muon stopping expectation (green, the exact `eval_stm` reference
   table) and the flat MIP line (grey dashes, the `mip_dqdx` knob).  Forward
   pass blue / backward red, TPC0 circles / TPC1 triangles; the same
   trajectory is drawn on the projections as black crosses.  The title and
   the summary next to it name the bundle being shown (`grp / t / main`) —
   if that does not match the row you clicked, the panel is stale, which is
   a bug.  A main cluster with no fit gets the reason the tagger stopped
   (`contained`, `midkink`, `nexits`, `nosteiner`, `shortfit`, `tgm`, …),
   which is also the table's `stm fit` column.
6. Two modes: **ALL bundles** vs **IN-BEAM only** (0.2–2.2 µs window, the
   `in_beam` column of the table).
7. Tag the focused bundle with the **TGM / STM / FC / LM** buttons
   (multi-select; LM = light mismatch), add a per-bundle comment and/or a
   per-event comment.  Everything autosaves on each change.
8. **Save labels** writes the actionable JSON.

## 3b. Re-scans after a fix (`--prev`)

When the chain is re-run after a code change, serve the NEW work root under a
NEW tag and name the previous run(s) as read-only baselines:

```bash
./serve_nusel_scan.sh 5010 --tag mcp10-merge \
    --prev ../work-mcp10:mcp10 --prev ../work-mcp10-chord:mcp10-chord \
    ../work-mcp10-merge2
```

Each `--prev ROOT[:TAG]` is an older work root plus the label tag scanned
there (repeatable; priority = given order; old dirs are never written).
Bundles are matched across runs by flash APA + time (±0.5 µs; cluster idents
and flash gids may relabel between runs), then:

* **Comments/labels survive.**  On first load of an event under the current
  tag, the previous scan labels, comments, and event comment are copied in
  (per field, the first `--prev` that has one wins).  Carried rows show a
  small `◦` in the scan column and a "carry-over" note in the metrics panel;
  editing them (or pressing **✓ re-scan OK**) adopts them as your own.
  A carried annotation you clear stays cleared on reload.
* **Changes are flagged.**  The `prev` table column shows the baseline
  verdicts of the first `--prev` covering the event: `=` unchanged,
  `t/s/f→` changed, `new` for a bundle with no baseline match.  A row whose
  tgm/stm/fc changed is tinted **amber** until re-scanned — any label or
  comment edit, or the **✓ re-scan OK** button, clears it (and appends ✓).
  The event header counts `N changed vs <tag>, M pending re-scan`, so the
  events needing attention are visible without stepping through rows.
* The metrics panel shows the baseline verdicts and your previous scan
  labels/comment with their source tag, so old and new can be compared in
  place while going back and forth between fixes.

## Saved label schema

`<work_root>/nusel_labels/<tag>/nusel-labels-evt<ID>.json`:

```jsonc
{
  "event": "evt<ID>", "run": ..., "subrun": ..., "event_no": ...,
  "source": "nusel-evt<ID>.tsv", "tag": "<tag>",
  "event_comment": "...",
  "bundles": [
    { "main_id": 11, "flash_gid": 4, "flash_grp": 7, "flash_apa": 0,
      "flash_time_us": 1.555, "flash_pe_grp": 12772.8, "in_beam": 1,
      "clusters": [11, 7],
      "auto": { "tgm": 1, "stm": 0, "fc": 0, "label": "TGM" },
      "scan_labels": ["TGM"], "comment": "",
      // present only when serving with --prev:
      "prev": { "tag": "mcp10", "matched": true, "tgm": 1, "stm": 0, "fc": 0,
                "label": "TGM", "changed": false,
                "scan_labels": ["TGM"], "comment": "" },
      "carried_over": false, "rescan_confirmed": true }
  ]
}
```

Every table row is saved (annotated or not), so a downstream comparison of
`scan_labels` vs `auto.label` covers the full bundle population.  The
in-progress state additionally autosaves to
`.scan_state-evt<ID>.json` in the same dir, so a browser reload or server
restart loses nothing.

Labels live under `<work_root>/nusel_labels/` — a **sibling** of the
per-event dirs, so `run_nusel_evt.sh` re-runs (which wipe `nusel_evt<ID>/`)
cannot delete scan results.  Use a fresh `--tag` for a fresh campaign; never
re-use a tag for a different sample.

## Notes

* PMT positions come from
  `wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json` (`OpDets`, list
  index = op channel, x<0 = TPC0).  Override with `SBND_OPDET_JSON`; without
  it the light panels fall back to channel-index-only overlays.
* The predicted pattern is `op_pes_pred` from the QL job: the sum of the
  matched clusters' predictions on that flash.  Unmatched (e.g. `no-bundle`)
  flashes have no prediction — the predicted panel says so.
* `nusel_extract.py` is imported for the pctree/table parsing — the viewer
  adds no second implementation of either.

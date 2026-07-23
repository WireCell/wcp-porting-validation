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
5. Two modes: **ALL bundles** vs **IN-BEAM only** (0.2–2.2 µs window, the
   `in_beam` column of the table).
6. Tag the focused bundle with the **TGM / STM / FC / LM** buttons
   (multi-select; LM = light mismatch), add a per-bundle comment and/or a
   per-event comment.  Everything autosaves on each change.
7. **Save labels** writes the actionable JSON.

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
      "scan_labels": ["TGM"], "comment": "" }
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

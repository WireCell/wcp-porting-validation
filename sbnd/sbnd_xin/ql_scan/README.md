# SBND Q/L matching hand-scan viewer

A Bokeh event display for reviewing and correcting the SBND Q/L (charge–light)
flash↔cluster matching by hand, and saving the result as labels for tuning the
QLMatching metrics / PE-error model.

## 1. Produce the calib dumps

The viewer reads the per-event calibration JSON written by QLMatching's `-calib`
mode (one file per event, both TPCs):

```bash
./run_ql_evt.sh <mc|data> [-N n] <idx|all> -calib
# -> work/ql_evt<ID>/calib-evt<ID>.json
```

The dump is written at the end of matching, so each bundle already carries the
matcher's final decision in its `auto_selected` flag.

## 2. Serve

```bash
# MC on 5008, data on 5009 (separate servers so their labels stay apart)
./serve_ql_scan.sh 5008 --tag mc   work/ql_evt*/calib-evt*.json
./serve_ql_scan.sh 5009 --tag data work/ql_evt*/calib-evt*.json
```

From a workstation:

```bash
ssh -L 5008:localhost:5008 -L 5009:localhost:5009 <host>
# then open http://localhost:5008/ql_scan_viewer
```

## 3. Review-and-correct workflow

Instead of building a match from scratch:

1. **Load auto-match** — seeds the selection from QLMatching's own result
   (`auto_selected` bundles). The scan then becomes a review of the matcher.
2. Examine and correct — tick/untick bundles. In the bundle table an `auto` "Y"
   column plus the green selected-row tint make disagreements visible: `Y` +
   non-green = the matcher's pick you removed; green + blank-auto = your addition.
   The selection summary shows the running diff (`+added / −removed`).
3. **Save labels** — `work/ql_labels/<tag>/labels-evt<ID>.json`.

`Clear selections` empties the scan; `Load auto-match` re-seeds it from the matcher.

## Display layout

- Controls are on two rows: event/group navigation on top, action buttons below.
- **Filter selected bundles** defaults ON — once a cluster is matched to a flash it is
  locked (🔒) from being selected for another. Toggle it off to override.
- In the bundle, compare, and cluster tables the **apa / flash / cluster** columns are
  kept adjacent for quick reading.
- Pick a cluster in the **clusters** roster, then click **Compare cluster's flashes**
  to list that cluster's candidate flashes in the second table (focusing a bundle row
  still works too; the most recent of the two wins).

## Saved label schema

```jsonc
{
  "event": "evt<ID>",
  "source": "calib-evt<ID>.json",
  "matches":        [ /* your final picks */ ],
  "rejected_auto":  [ /* auto-matches you removed */ ]
}
```

Each entry in both lists has the same shape (flash_gid/flash_id/flash_time_us, apa,
group, cluster_idents, op_pes, op_pe_err, pred_pes, metrics, flags). The hand-vs-
matcher diff is fully recoverable: within `matches`, `flags.auto_selected` is `true`
for kept auto-matches and `false` for hand-added ones; `rejected_auto` holds the
removed auto-matches.

> Note: `scripts/analysis/ql/ql_pe_error.py` / `scripts/analysis/ql/ql_prefilter_tune.py` currently read `matches` only.
> `rejected_auto` is recorded for future tuning use; consuming it is a separate task.

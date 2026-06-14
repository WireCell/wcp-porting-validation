# PDHD Q/L matching hand-scan viewer

A Bokeh event display for reviewing and correcting the PDHD Q/L (charge–light)
flash↔cluster matching by hand, and saving the result as labels for tuning the
QLMatching metrics / PE-error model. Faithful port of `sbnd_xin/ql_scan/`; see
`../docs/ql-scan-display.md` for the full schema and design notes.

> **Status:** machinery only. The PDHD processing chain (light reco + matching) is
> still being tuned, so this is not yet ready for production hand-scanning.

## 1. Produce the calib dumps

The viewer reads the per-event calibration JSON written by QLMatching when the
clustering chain is run with `-calib`. PDHD wires **one QLMatching node per drift
side**, so each event yields up to **two** per-group files (group02 = APAs 0+2,
drift −x → side 0; group13 = APAs 1+3, drift +x → side 1):

```bash
./run_clus_evt.sh -calib <run> <evt|all>
# -> work/<run6>_<evt>/calib-evt<ID>-group02.json
#    work/<run6>_<evt>/calib-evt<ID>-group13.json
```

`-calib` implies Q/L matching; the matched `mabc-*.zip` output is byte-identical with
or without it. The dump is written at the end of matching, so each bundle already
carries the matcher's final decision in its `auto_selected` flag.

The central cathode is opaque to VUV, so the two drift sides are independent; today
usually only one side carries light and only one file is produced. The viewer **merges
the 1–2 group files of an event** into one two-panel (side 0 / side 1) view and simply
leaves the unlit side blank.

## 2. Serve

```bash
# default port 5015 (img_plot owns 5013, pd_plot 5014)
./serve_ql_scan.sh 5015 --tag data work/*/calib-evt*-*.json
```

From a workstation:

```bash
ssh -L 5015:localhost:5015 <host>
# then open http://localhost:5015/ql_scan_viewer
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
- In the bundle, compare, and cluster tables the **side / flash / cluster** columns are
  kept adjacent for quick reading.
- Pick a cluster in the **clusters** roster, then click **Compare cluster's flashes**
  to list that cluster's candidate flashes **in the current coincidence group** in the
  second table (focusing a bundle row still works too; the most recent of the two wins).
  The list is restricted to the selected group, so flashes from neighbouring groups
  (e.g. a coincident flash 1–2 µs away in the adjacent group) are not shown.

## Saved label schema

```jsonc
{
  "event": "evt<ID>",
  "source": "calib-evt<ID>-group02.json",
  "matches":        [ /* your final picks */ ],
  "rejected_auto":  [ /* auto-matches you removed */ ]
}
```

Each entry in both lists has the same shape (flash_gid/flash_id/flash_time_us, apa
[= drift side], group, cluster_idents, op_pes, op_pe_err, pred_pes, metrics, flags).
The hand-vs-matcher diff is fully recoverable: within `matches`, `flags.auto_selected`
is `true` for kept auto-matches and `false` for hand-added ones; `rejected_auto` holds
the removed auto-matches.

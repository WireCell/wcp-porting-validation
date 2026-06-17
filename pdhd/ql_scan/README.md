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

The central cathode is opaque to VUV, so **no single flash lights both drift volumes** —
every flash is one-sided. A cathode-crossing cosmic therefore appears as **two
same-time one-sided flashes** (one per side). The viewer **merges the two group files**
and re-pairs these into **cross-side coincidence groups**: an S0-lit and an S1-lit flash
within `COINC_WIN` (1 µs, set from the run-29107 cross-side Δt distribution; one constant
at the top of `ql_scan_viewer.py`) share one group, shown across both panels; flashes
with no opposite-side partner stay single-sided (one panel lit, the other blank). On
run 29107 this is ~11–13 two-sided groups/event. The pairing is by each flash's **lit
side** (from its PE), not its file side — the per-side matcher runs against one global
flash list, so a flash is referenced by clusters on both sides and file-side would
mis-pair every flash with its own copy.

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
- **Hide clusters ≤5cm** defaults ON — short fragments (bbox-diagonal length ≤ 5 cm,
  `MIN_CLUS_LEN` in `ql_scan_viewer.py`) are dropped from the clusters roster and the
  bundle table, since an event has many tiny ones that swamp the display. A coincidence
  group whose clusters are *all* this short is also skipped from the group navigation /
  dropdown (an all-short group has nothing worth scanning). Toggle off to show everything.
- In the bundle, compare, and cluster tables the **side / flash / cluster** columns are
  kept adjacent for quick reading.
- Pick a cluster in the **clusters** roster, then click **Compare cluster's flashes**
  to list **all** of that cluster's candidate flashes (across coincidence groups) in the
  second table (focusing a bundle row still works too; the most recent of the two wins).
  The calib dump only emits TPC-contained bundles, so this list is already the
  physically-feasible candidate set — the cluster lands inside the drift box at each
  listed flash's T0. Clicking a row jumps the whole view (focus + group) to that flash,
  so a candidate in another group is reachable. (A cathode-hugging cluster can be
  compatible with many flash times, since a later T0 just shifts it toward the anode.)

## Light panels (measured / predicted, per side)

Each side's panel draws the **full physical PMT array**: live OpDets as grey circles
(filled/colour-scaled by PE for the group's flash), and **masked OpDets as faint red ×**
(static dead + per-event `auto_mask`). The red × matter on side 0 (−x): channels 120–159
are the DAPHNE full-stream PDs covering the z<250 half — masked when the light reco lacks
the full stream, live (and so charge-scaled) when it has it. So a side-0 panel that looks
half-empty is the mask, not missing data. Panel y/z ranges are pinned to the two-APA box.

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

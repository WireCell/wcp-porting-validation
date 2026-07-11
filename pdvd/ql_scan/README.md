# PDVD Q/L matching hand-scan viewer

A Bokeh event display for reviewing and correcting the PDVD Q/L (charge–light)
flash↔cluster matching by hand, and saving the result as labels for tuning the
QLMatching metrics / PE-error model. Duplicated from `pdhd/ql_scan/` (which in
turn ports `sbnd_xin/ql_scan/`); see `../../pdhd/docs/ql-scan-display.md` for
the base schema and design notes, and `../docs/pdvd-qlmatching.md` for the
PDVD matching chain.

> **Status:** machinery only. The light-vs-charge trigger offset is not yet
> determined (see `../docs/pdvd-ql-pending.md`), so absolute drift-corrected
> positions — and therefore the matches themselves — are not yet trustworthy
> for production hand-scanning.

## 1. Produce the calib dumps

The viewer reads the per-event calibration JSON written by QLMatching when the
clustering chain is run with `-calib`. PDVD wires **one joint QLMatching node**
over both drift volumes (`shared_flash`), so each event yields **one** file:

```bash
./run_clus_evt.sh -calib <run> <idx|all>
# -> work/<run6>_<idx>/calib-evt<ID>.json
```

`-calib` implies Q/L matching; the matched `mabc-*.zip` output is byte-identical
with or without it. Each bundle carries the matcher's final decision in
`auto_selected` and its drift volume in `apa` (0 = bottom, 4 = top).

Unlike PDHD, PDVD has a **single all-PD flash list**: the cathode X-ARAPUCAs see
both drift volumes, so there is no cross-side coincidence pairing — one scan
group per flash, referenced by bundles from either or both volumes, shown across
both panels simultaneously.

## 2. Serve

```bash
# default port 5016 (img_plot 5013, pd_plot 5014, pdhd ql_scan 5015)
./serve_ql_scan.sh 5016 --tag data work/*/calib-evt*.json
```

From a workstation:

```bash
ssh -L 5016:localhost:5016 <host>
# then open http://localhost:5016/ql_scan_viewer
```

## 3. Review-and-correct workflow

Same as PDHD:

1. **Load auto-match** — seeds the selection from QLMatching's `auto_selected`
   bundles; the scan becomes a review of the matcher.
2. Examine and correct — tick/untick bundles; the `auto` column plus the green
   selected-row tint make disagreements visible, and the selection summary shows
   the running diff (`+added / −removed`).
3. **Save labels** — `work/ql_labels/<tag>/labels-evt<ID>.json`.

## PDVD-specific display notes

- Charge-projection panels (top of the page) always draw **both** drift
  volumes' boxes together; a cathode-crossing cosmic is two bundles (one per
  volume) on the SAME flash — tick both.
- The **light panels** (2-D maps + 1-D histograms, bottom of the page) are
  grouped by **PD type**, not drift volume — the flash is one all-PD set (the
  cathode X-ARAPUCAs are double-sided and belong to both volumes at once, so a
  bottom/top split either dupes or under-predicts them):
  - **X-ARAPUCA** panel — wall/membrane XAs (ch 0–3, 12,13, 18,19; |x| > 10)
    and the 8 cathode XAs (ch 4–11, x≈0), separated by a dashed line
    (1-D) / marker shape (2-D: circle = wall, square = cathode).
  - **PMT** panel — z-wall PMTs (ch 14–17, 20–23) and the 16 bottom/floor PMTs
    (ch 24–39, behind the bottom anode), same wall/inner split. All PMTs live
    in the bottom drift volume.
  - Predicted light = the element-wise sum of **all** selected bundles on the
    group flash across **both** volumes (not just one volume's clusters), so a
    cathode-crosser's or shared cathode-XA's predicted PE is counted whole.
  - Panel y/z ranges are the PD-position envelope ∪ both drift boxes, since the
    membrane/bottom PDs sit outside the TPC box. Masked OpDets (static
    `ch_mask` + per-event `auto_mask`) are faint red ×/+ (circle-x = wall,
    square-x = cathode/bottom).
  - The wall/cathode (or wall/bottom) z-ranges overlap spatially in the 2-D
    maps (e.g. z-wall vs bottom PMTs), so the marker shape — not position — is
    what tells them apart there; same-position x-stacked PDs (e.g. membrane XA
    behind a z-wall PMT) still overlap on the scatter, use hover for the PE.
- Bundles with the new `flag_at_cathode` (cluster end within the cathode
  cushion) are tinted **light orange** with an `atCATH` tag in the flags column
  — the PDVD analogue of attention-worthy proximity flags (89% of light is
  collected at the cathode, so these dominate the PE budget).
- The metrics table adds an `at_cathode` row; `close_to_PMT` here means the
  y-wall XA cushion or the bottom-anode PMT window (`vd_surface_flags`), not
  the PDHD anode meaning.
- Drift-corrected x uses the dump's `trigger_offset`; with the offset still
  uncalibrated (diagnostic dumps force 0) the absolute positions are wrong by
  sign·T·v — scan geometry qualitatively, not quantitatively, until the offset
  lands.

## Saved label schema

Same as PDHD (`matches` / `rejected_auto`, each entry with flash_gid/flash_id/
flash_time_us, apa [= drift volume 0/4], cluster_idents, op_pes, op_pe_err,
pred_pes, metrics, flags) with `at_cathode` and `two_boundary` added to flags.

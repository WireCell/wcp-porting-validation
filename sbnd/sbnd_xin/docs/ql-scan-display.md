# Q/L matching hand-scan event display

An interactive tool to examine SBND charge–light (Q/L) matching one event at a
time, override the matcher's auto-selection by hand, and save the human-picked
flash↔cluster matches as labels for later parameter tuning.

It has two parts:

1. a **`-calib` dump** in `QLMatching` that writes, per event, the full candidate
   bundle universe (both TPCs) to a JSON file;
2. a **Bokeh server viewer** (`ql_scan/`) that renders the bundles, lets you pick
   the correct matches under the physical selection rules, and writes a labels JSON.

The viewer mirrors `pdvd/sp_plot/serve_filter_tune_viewer.sh` (Bokeh server + SSH
port-forward); UX ideas (PMT circles sized by √PE, status-bar inspection) are
borrowed from `wire-cell-bee3`.

---

## 1. Produce the calib dumps

The dump is emitted by the per-event chain (`run_ql_evt.sh`, joint mode by default,
so one file holds both TPCs). Add `-calib`:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./run_img_evt.sh mc all          # per-event imaging (prerequisite)
./run_ql_evt.sh  mc all -calib   # Q/L matching + dump
# -> work/ql_evt<ID>/calib-evt<ID>.json   (one per event, both TPCs)
```

The `-calib` flag is **off by default**; with it off the matched `mabc-all-apa.zip`
is byte-for-byte identical (the dump method is never called). The dump is
observation-only — it reads the matcher's finished state and never perturbs the
matching (verified: matched zip identical with and without `-calib`).

### Calib JSON schema

All spatial quantities are in **cm**, flash time in **µs**, drift speed in **cm/µs**.
Cluster idents are per-APA, so a globally-unique **cluster uid = apa·10⁶ + ident**
disambiguates the two TPCs (same stride as the flash gid).

| key | meaning |
|-----|---------|
| `nchan` | optical-channel count (312) |
| `drift_speed` | cm/µs; a bundle's T0 x-shift is `dx = sign_offset · flash_time_us · drift_speed` |
| `quality_params` | the active χ²/`flag_high_consistent` thresholds + `PE_err` model (`highconsist_ks_max`, `highconsist_min_ndf`, `pe_ndf_knee`, `strength_cutoff`, `pe_err_{floor,frac,knee}`, …) |
| `geometry[apa]` | fixed detector box per TPC: `anode_x`, `cathode_x`, `u_cathode`, `s`, `sign_offset`, `y_lo/y_hi`, `z_lo/z_hi` |
| `opdets[]` | per channel `{ch, x, y, z, type, apa, active}` (active = used by the matcher's per-TPC mask) |
| `flashes[]` | per flash `{gid, id, apa, time, total_PE, group, pe[nchan], pe_err[nchan]}`; `group` = ±80 ns coincidence id across both TPCs (replicates `MultiAlgBlobClustering::store_flash_groups`) |
| `clusters[]` | per cluster `{uid, ident, apa, npoints, x[], y[], z[], q[]}` (raw, un-shifted; the viewer applies `dx`) |
| `bundles[]` | one per candidate (flash, cluster-group) pair — see below |

Each `bundles[]` entry (the **TPC-contained** candidates — `run.all_bundles` filtered
to those whose cluster stays inside the box after the T0 x-shift; uncontained bundles
are skipped, since they carry zeroed metrics and are never auto-selected):

```
apa, flash_gid, flash_id, main_cluster (uid), other_clusters[] (uids),
ks_dis, chi2, ndf, strength, total_pred_light, total_PE,
contained, consistent, potential_bad_match, close_to_PMT, at_x_boundary,
spec_end, window_truncated, auto_selected, pred_pe[nchan]
```

- `auto_selected` = the bundle is in the matcher's **final** internal selection
  (`flash_bundles_map` after the fit). This is the matcher's ground-truth decision,
  and may differ slightly from the post-MABC Bee `-op.json` display (the downstream
  re-clustering can drop weak matches). The hand-scan deliberately surfaces these —
  e.g. a flash matched to a cluster that, at its T0, predicts almost no light is a
  prime candidate to deselect.
- `contained` = the bundle passed the TPC-containment gate. Always `true` in the dump
  now (uncontained bundles are filtered out at dump time); the field is kept for
  schema stability.

---

## 2. Launch the viewer

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./ql_scan/serve_ql_scan.sh 5008                       # default: scan work/ql_evt*/calib-evt*.json
./ql_scan/serve_ql_scan.sh 5008 work/ql_evt2/calib-evt2.json   # explicit file(s)
```

From a laptop, forward the port and open the app:

```bash
ssh -L 5008:localhost:5008 user@wcgpu1
# browser: http://localhost:5008/ql_scan_viewer
```

### Layout
- **Event** selector + prev/next, and a **coincidence-group** selector + prev/next.
  The hand-scan is done **one ±80 ns group at a time** (the navigation unit), so the
  busy full-event bundle list is broken into the coincident TPC0/TPC1 units the eye
  actually compares. Each group label shows its TPC0/TPC1 flash times. The group
  selector lists only groups that still have a bundle to scan; a group emptied by a
  selection made elsewhere is **skipped** by prev/next (see rules).
- **Bundle table** (left) with the **selection summary** (right): the table shows only
  the **current group's** bundles, minus any whose cluster is already claimed by a
  selection (see rules) — `state` (SELECTED / avail), `auto`, apa, flash, time, group,
  cluster, ks, chi²/ndf, strength, measured/predicted PE, flags. Click a row to
  **focus** it. The summary beside it lists the current picks (flash → clusters, all
  groups).
- **Metrics panel**: the focused bundle's full metrics next to the quality thresholds,
  and its flags.
- **Compare-cluster table**: the **Compare cluster's flashes** button lists, in a second
  table, every candidate flash the *focused* cluster could match (all groups, sorted by
  flash time) with the same ks/chi²/strength/PE columns, so the right flash for that
  cluster can be read off side by side. Once open it **follows the focus** — clicking a
  different bundle re-lists that cluster's flashes (no need to re-press the button). Note
  two clusters can share the same flashes, so the left columns may look identical while
  ks/chi²/strength/predicted PE differ. Click a row to **jump** the whole view (focus +
  group) to that candidate flash; the light/histogram/projection panels follow.
- **Light patterns** — a 2×2 grid: **measured + predicted for TPC0** and **measured +
  predicted for TPC1**. Positions are fixed, so the ranges are pinned to the detector
  box (no zoom). The two **measured** panels are anchored to the group's per-TPC flash
  (a stable reference that does not flicker as rows are clicked); the two **predicted**
  panels sum the clusters selected on that TPC's flash (or preview the focused bundle
  when nothing is selected yet). Circle radius ∝ √PE, independent per-panel scale, with
  a labelled PE colour bar.
- **1-D comparison** — below the grid, per TPC an **overlay** of measured (bars) vs
  predicted (line) PE over that TPC's active PMT channels, and a **pred/meas ratio**
  (reference line at 1; channels with zero measured PE are dropped). These read off the
  same measured/predicted vectors as the 2-D panels, giving the magnitude *and* pattern
  mismatch at a glance.
- **Charge projections**: X-Y, Y-Z, X-Z of the focused bundle's cluster(s), shifted
  by the bundle's T0 `dx`, inside the **fixed** detector box (both TPC boxes drawn);
  the currently-selected matches are shown faintly for context.

### Selection rules (enforced live)
- **±80 ns coincidence as the navigation unit** — you scan one coincidence group at a
  time; within a group every bundle is coincident by construction, and the four light
  panels show that group's TPC0 and TPC1 flash light together. Change the group with
  the selector / prev-next.
- **One flash per cluster** — selecting a bundle for a cluster removes that cluster's
  other candidate bundles from the table, **across all groups** (and replaces any
  previous pick for that cluster). This is how you whittle the candidates down.
- **Many clusters per flash** — selecting several bundles that share a flash sums
  their predicted light (the measured pattern is unchanged).
- **Empty groups are skipped** — once a cluster is picked, its rival bundles vanish
  from every group; a group left with no bundle drops out of the group navigation
  (prev/next), though the group you are currently viewing always stays selectable.
- **Compare a cluster's flashes** — the compare table (above) is the read-across that
  complements the per-group view: it pins one cluster and shows all the flashes it
  could match, so you pick the better flash, then select it as usual.

### Save
**Save labels** writes `work/ql_evt<ID>/labels-evt<ID>.json`: one entry per selected
match with `flash_gid`, `flash_time_us`, `apa`, coincidence `group`,
`cluster_idents[]`, and — for downstream tuning — the per-channel `op_pes`,
`op_pe_err`, `pred_pes`, plus the metrics and flags. Self-contained: a later tuner
reads the labels alone.

---

## Notes / scope
- SBND only (both TPCs, ±80 ns groups, semi-analytical-sbnd geometry).
- Per-flash semantics: coincident TPC0/TPC1 flashes stay **separate** matched flashes
  linked as a pair; the coincidence is only the scan-navigation unit, never a merge
  (predicted light sums only within one flash/TPC). This produces a ground-truth
  coincident-pairing label that informs the deferred joint algorithm
  (`match/docs/joint-qlmatching-design.md`).
- The labels are the deliverable; fitting updated χ²/threshold/`PE_err` parameters
  from them is downstream manual work, not part of this tool.

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

Data works identically — swap `mc` for `data`:

```bash
./run_img_evt.sh data all
./run_ql_evt.sh  data all -calib
```

mc and data event ids are disjoint, so both sets' dumps coexist under `work/ql_evt<ID>/`
in distinct per-event subdirs. Because the default serve glob picks up *every*
`work/ql_evt*/calib-evt*.json`, scan a single dataset by passing its explicit paths (and a
distinct port for a second, concurrent display) — see §2.

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
# Tag each display so its saved results land in work/ql_labels/<tag>/ (mc vs data,
# kept apart — both modes otherwise share one work/). The mc/data ids do not collide
# (mc 2,9,11,…; data 686,1258,…) but --tag keeps the two displays' labels separate.
./ql_scan/serve_ql_scan.sh 5008 --tag mc   work/ql_evt{2,9,11,12,14,18,31,35,41,42}/calib-evt*.json
./ql_scan/serve_ql_scan.sh 5009 --tag data work/ql_evt{686,1258,1302,1346,1698,1720,1808,1852,2028,2050}/calib-evt*.json
```

From a laptop, forward the port and open the app:

```bash
ssh -L 5008:localhost:5008 user@wcgpu1   # mc   (use -L 5009:localhost:5009 for data)
# browser: http://localhost:5008/ql_scan_viewer
```

### Layout
- **Page order**: the three **charge-projection views** are at the **top** of the page
  (so they are visible first on a small monitor), with a one-line caption directly
  beneath them echoing the current **Event / coincidence group / load summary**.
  Everything else — controls, status, the bundle table + selection boxes + metrics +
  cluster roster, the compare table, and the light/histogram panels — follows **below**
  the projections, so the top of the screen shows the plots and the bottom is where you
  operate.
- **Event** selector + prev/next, and a **coincidence-group** selector + prev/next.
  The hand-scan is done **one ±80 ns group at a time** (the navigation unit), so the
  busy full-event bundle list is broken into the coincident TPC0/TPC1 units the eye
  actually compares. Each group label shows its TPC0/TPC1 flash times. The group
  selector lists only groups that still have a bundle to scan; a group emptied by a
  selection made elsewhere is **skipped** by prev/next (see rules).
- **Bundle table** (left) with the **selection checkboxes + summary** (right): the table
  shows **all** of the current group's bundles, one per row, numbered in column `#` —
  `auto`, apa, flash, time, group, cluster, ks, chi²/ndf, strength, measured/predicted PE,
  flags. Click a row to **focus** it for inspection (drives the light/projection panels) —
  the focused row highlights **blue**. Rows of bundles already **selected** are tinted
  **green** persistently (distinct from the blue click-focus), so the moment you land on a
  group you can see which of its bundles you have already picked.
  Selecting is done with the **`select matches` checkbox list** beside the table: one box
  per row (same `#` order, label `#: T<apa> fl<gid> c<clus> ks.. pr..`). **Tick a box to
  select** that bundle — its predicted light joins the per-flash sum; tick **several**
  (e.g. two clusters on one TPC) to combine them; untick to remove. A 🔒 in a label marks
  a bundle the **filter** forbids selecting (see rules). The **selection summary** below
  the boxes lists the current picks (flash → clusters, all groups).
- **Metrics panel**: the focused bundle's full metrics next to the quality thresholds,
  and its flags. It sits **to the right of the table** (not below it).
- **Cluster roster** (rightmost in the table row): every cluster in the event (both
  TPCs) with its ident, TPC, `npts`, `len(cm)` (3-D bounding-box diagonal), and a green
  **✓ → flash** when it has been matched. Shows at a glance which clusters are still
  unassigned. Read-only; sorted by TPC then ident.
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
  by the bundle's T0 `dx`, inside the **fixed** detector box (both TPC boxes drawn).
  **All selected** tracks (within *and* outside the current group) are drawn **gray** for
  context; the selected tracks **in the current group** are overlaid **green**; and the
  focused (clicked) bundle is drawn **blue** on top.

### Selection rules (enforced live)
- **±80 ns coincidence as the navigation unit** — you scan one coincidence group at a
  time; within a group every bundle is coincident by construction, and the four light
  panels show that group's TPC0 and TPC1 flash light together. Change the group with
  the selector / prev-next.
- **Many clusters per flash** — tick several bundles that share a flash and their
  predicted light **sums** (the measured pattern is unchanged). The natural cap is **one
  flash per cluster, not one bundle per TPC**: you may ✓ two (or more) bundles on the same
  TPC as long as they are different clusters; if they share that TPC's flash their
  predicted patterns add, which is the whole point of the multi-cluster case.
- **Filter selected bundles** (toggle button) — by default any bundle can be ticked
  (free exploration, including, deliberately, the same cluster against two flashes). Turn
  the filter **ON** to enforce *one flash per cluster*: every bundle whose cluster is
  already matched by another selection is **locked** (🔒) — it stays visible for context
  but its checkbox is refused. Turn it back OFF to edit freely. Rivals are no longer
  hidden; the filter only locks them.
- **Compare a cluster's flashes** — the compare table (above) is the read-across that
  complements the per-group view: it pins one cluster and shows all the flashes it
  could match, so you pick the better flash, then select it as usual.

### Selection persistence
Your picks are **autosaved** to `work/ql_labels/<tag>/.scan_state-evt<ID>.json` on every
select/deselect/clear and restored when the event loads — so they survive a page
reload, a server restart, and switching between events (each event keeps its own set).
Keys are `(flash_gid, main_cluster)`, stable across re-dumps. This is the live working
state; the **Save labels** button below still writes the formal deliverable.

Both files live in `work/ql_labels/<tag>/` — a sibling of the per-event `work/ql_evt<ID>/`
workspace, deliberately **outside** it so that re-running `run_ql_evt.sh` (which does
`rm -rf work/ql_evt<ID>/` before each event) cannot delete your saved scan results. The
`<tag>` (from `--tag`, e.g. `mc` / `data`) keeps the two displays' results in separate
subdirs; with no `--tag` they go straight into `work/ql_labels/`.

### Save
**Save labels** writes `work/ql_labels/<tag>/labels-evt<ID>.json`: one entry per selected
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

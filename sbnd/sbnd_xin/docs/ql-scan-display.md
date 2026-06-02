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

Each `bundles[]` entry (the full candidate universe, i.e. `run.all_bundles`):

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
- `contained` = the bundle passed the TPC-containment gate and has a predicted-light
  vector (uncontained bundles carry an empty `pred_pe`).

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
- **Event** selector + prev/next.
- **Bundle table**: every candidate bundle for the event — `state`
  (SELECTED / avail / blocked), `auto`, apa, flash, time, group, cluster, ks,
  chi²/ndf, strength, measured/predicted PE, flags. Click a row to **focus** it.
- **Metrics panel**: the focused bundle's full metrics next to the quality
  thresholds, and its flags. **Selection summary**: the current picks.
- **Light patterns**: measured (left) vs predicted (right) PMT patterns for the
  focused flash, drawn in the y–z plane (circle radius ∝ √PE, shared colour scale);
  faint outlines mark every active PMT of both TPCs. The predicted pattern is the
  **sum** over the clusters currently selected for that flash.
- **Charge projections**: X-Y, Y-Z, X-Z of the focused bundle's cluster(s), shifted
  by the bundle's T0 `dx`, inside the **fixed** detector box (both TPC boxes drawn);
  the currently-selected matches are shown faintly for context.

### Selection rules (enforced live)
- **One flash per cluster** — selecting a bundle for a cluster drops that cluster's
  other candidate bundles (and replaces any previous pick for it).
- **Many clusters per flash** — selecting several bundles that share a flash sums
  their predicted light (the measured pattern is unchanged).
- **±80 ns coincidence as an availability filter** — once a flash is picked on one
  TPC, only opposite-TPC bundles whose flash shares the same coincidence `group`
  stay selectable (the rest show `coinc`).

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
  linked as a pair; the coincidence is only an availability filter. This produces a
  ground-truth coincident-pairing label that informs the deferred joint algorithm
  (`match/docs/joint-qlmatching-design.md`).
- The labels are the deliverable; fitting updated χ²/threshold/`PE_err` parameters
  from them is downstream manual work, not part of this tool.

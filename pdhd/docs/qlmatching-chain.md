# PDHD charge–light (Q/L) matching — chain & status

How the SBND-derived `QLMatching` component is wired and configured for
ProtoDUNE-HD, what geometry/side/cathode/anode flags it reads, what light-model
knobs still need tuning, how the one-side-missing-light situation is handled, and
the remaining gotchas.

Companion docs: [`photon-detector-chain.md`](photon-detector-chain.md) (light reco
producing the opflash), [`pdhd-light-raw-data.md`](pdhd-light-raw-data.md),
[`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md) (channel↔OpDet,
dead channels), [`ql-scan-display.md`](ql-scan-display.md) (hand-scan viewer),
[`clustering-algorithm.md`](clustering-algorithm.md) (the 4-stage clustering QL sits
inside).

Code: `match/src/QLMatching.cxx` + `match/inc/WireCellMatch/QLMatching.h` (toolkit).
Config: `cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet`,
`.../pdhd/clus.jsonnet`, `pdhd/wct-clustering.jsonnet` (this repo).

> **One-line status.** The chain runs end-to-end and produces matched flashes/T0s.
> The geometry is correct (combined two-APA box, fixed below). The remaining work is
> **light calibration**: the absolute charge→PE normalization (`QtoL`×`VUVEfficiency`)
> and the per-PMT light uncertainty are still placeholders/SBND values and must be
> tuned against PDHD data/MC before the matched flashes are physics-grade.

---

## 1. The chain

PDHD images **two drift volumes** separated by an opaque central cathode at `x = 0`.
Each volume contains **two APAs offset along Z (beam)**:

| group   | APAs | drift side | imaging face | z coverage |
|---------|------|-----------|--------------|------------|
| group02 | 0, 2 | `x < 0`   | `tpc_face=0` | APA0 z[0,230] + APA2 z[232,463] cm |
| group13 | 1, 3 | `x > 0`   | `tpc_face=1` | APA1 z[0,230] + APA3 z[232,463] cm |

Q/L matching is inserted **between clustering stage 3 (per-drift-group) and stage 4
(all-TPC)** — `pdhd/wct-clustering.jsonnet`, function `ql_chain(gd)` under the
`do_qlmatch` switch (default off → historical no-matching chain). Per drift-side
group:

```
opflash_pdhd-wct.tar.gz ─▶ TensorFileSource ─┐
                                              ├▶ FlashTensorToOpticalPCs ─▶ QLMatching ─▶ all-TPC
   stage-3 group cluster tree ────────────────┘   (nchan=160, 2→1 fan-in)    (per group)
```

- **`TensorFileSource`** (`opflash_source`) reads the per-event opflash archive
  produced by the light chain.
- **`FlashTensorToOpticalPCs`** (`flash_attach`, `nchan=160`, `correct_flash_time:false`)
  expands the opflash matrix into the canonical flash / light / flashlight point
  clouds on the cluster root. Port 0 = cluster tree, port 1 = opflash matrix.
- **`QLMatching`** (`matching`) reads those flash PCs, predicts light from the
  cluster charge, fits flash↔cluster, and writes a per-cluster matched-flash scalar
  (`Cluster::get_flash()`), T0, and `matched_flash_gid` — propagated to every cluster
  of a group so the T0 survives the all-TPC re-merge (see
  [group-aware matching](#group-aware-matching)).

`group02 → clus_all_tpc` port 0, `group13 → port 1`. The two groups are matched
**independently**, each in its own `ApaRun` (`QLMatching.cxx:478`).

<a name="group-aware-matching"></a>
**Group-aware (main + associated), then recomposed.** Stage-3 per-group clustering
tags blobs with an `isolated/perblob` array (−1 = main, ≥0 = sub-clusters).
`QLMatching` splits each group into a main + associated clusters
(`decompose_cluster_groups`, `QLMatching.cxx:694`), predicts light over the
**whole** group, and anchors the bundle on the main. The matched T0 is written to
the main and copied to its associated clusters.

> **The split is matching-INTERNAL and is undone before output.** After the fit,
> `recompose_cluster_groups` (`QLMatching.cxx`, called in `operator()` after the
> per-APA fit loop) merges each group's associated sub-clusters back into their
> main via `Grouping::merge` — the inverse of the `decompose` `separate`. So
> `QLMatching` emits a **T0/flash-annotated copy of its stage-3 input tree**: it
> assigns T0 and **does not change the clustering**. The matched cluster then
> carries the T0, and `x_t0cor` (materialized at the first all-TPC `switch_scope`)
> drives the drift correction.
>
> **Why this is required.** If the split is left in the output, the downstream
> all-TPC clustering inherits it. PDHD stage 4 is only `switch_scope` +
> `cathode_connect` (`use_flash_t0=false`) — a pure cross-cathode merge with no
> flash-T0 merge pass to re-absorb a within-volume split — so a single stage-3
> cluster came out **separated** whenever the matcher ran (run 27305 evt 150: one
> track split across clusters 1299/760/0/155). SBND's stage-4 `clus_all_apa`
> *does* run flash-T0-gated merge passes (`extend/regular/.../examine_bundles`,
> `use_flash_t0=true`) that re-absorb it, so SBND never showed the symptom. The
> recompose makes the matcher grouping-preserving for **both** detectors;
> verified on PDHD evt 150 (the QL-on cluster partition is now byte-identical to
> the no-QL chain — `cluster_id` arrays match point-for-point, only the T0-driven
> `x` shift differs) and on the SBND 10-event data sample (A/B with the recompose
> off vs on: the final point cloud and the cluster partition are **identical**
> for every event — only the arbitrary per-event `cluster_id` labels and the
> within-cluster point order differ, since stage 4 now relabels a merged rather
> than a pre-split input).

---

## 2. Geometry: side / cathode / anode flags (point 1)

All per-TPC geometry is computed in `QLMatching::compute_geometry()`
(`QLMatching.cxx:773`) from the `DetectorVolumes` service — nothing is hard-coded to
SBND.

**Representative anode → side.** Each `ApaRun` is keyed by a representative anode
(`anodes[0]`: APA0 for group02, APA1 for group13). Its ident (0 or 1) sets:

- `run.sign_offset = (tpc==0) ? −1 : +1` — the drift-direction sign
  (`QLMatching.cxx:776`).
- the **per-side OpDet mask**: an OpDet at `x < cathode_x` belongs to TPC 0, else
  TPC 1 (`QLMatching.cxx:831`). `cathode_x = 0` comes from the semi-analytical model
  JSON (`Geometry.cathode_x`), matching the PDHD central cathode.

**`tpc_face`** (`qlmatching.jsonnet`, `0` for group02 / `1` for group13) selects which
anode face defines the sensitive volume via `inner_bounds(wpid)`. PDHD's `+x` side
images on face 1, the `−x` side on face 0.

**Drift coordinate.** From the active box: `anode_x`, `cathode_x`, `s = sign(cathode−anode)`,
`u_cathode = s·(cathode_x − anode_x) > 0`. A cluster point maps to the per-TPC drift
coordinate `u = s·(x − anode_x)` (0 at anode, `u_cathode` at cathode).

**Flash → x.** A flash time becomes a drift offset:

```
flash_x_offset = sign_offset · (flash_time + trigger_offset) · drift_speed
```

(`QLMatching.cxx:955`; `drift_speed = 1.565 mm/µs`, see
[`clustering-algorithm.md`](clustering-algorithm.md) drift-velocity calibration).
Each cluster point is shifted by `flash_x_offset` before the light prediction and the
**PE-inclusion gate** (`QLMatching.cxx:1024`):

```
u  in [anode_ext1, u_cathode + cathode_ext1]      (drift)
y  in [y_lo, y_hi],  z in [z_lo, z_hi]             (transverse, from the active box)
```

### Combined two-APA box (point 5) — fixed

`compute_geometry` now builds the active box as the **union of the sensitive boxes of
every APA in the group** (`m_grouping_anodes`), not just the representative anode.

> **Why this matters.** PDHD's two same-side APAs are offset in **Z**, not X (APA0
> z[0,230] cm, APA2 z[232,463] cm). `inner_bounds()` returns one
> `IAnodeFace::sensitive()` box, so the old single-anode query covered only the
> upstream half; the transverse gate at `QLMatching.cxx:1026` then **dropped all
> charge in the downstream APA** (APA2/APA3) from the predicted-light sum — half of
> each drift volume produced no predicted light, so a cluster in that half could not
> be matched (and a track spanning both APAs under-predicted).

The fix (`QLMatching.cxx:778`) unions `inner_bounds` over the group's anodes for the
Y/Z extent. X is identical across same-side APAs, so the drift math (`anode_x`,
`u_cathode`, `s`) and the per-side OpDet mask are unchanged. Verified in the debug log
(run 27980 evt 12):

```
<matching_group13> anode 1 group-bbox (2 apa) x[0.16,352.09] y[7.61,606.00] z[0.24,462.30] cm
<matching_group02> anode 0 group-bbox (2 apa) x[-352.09,-0.16] y[7.61,606.00] z[0.23,462.30] cm
```

`z[0.24, 462.30] cm` is the full two-APA span (was ~`z[0.24, 230] cm`).

**SBND-safe (no toggle).** Only PDHD sets `grouping_anodes`; with an empty list the
union reduces to the single representative anode → **byte-identical** for SBND and
every other existing config.

### Readout-window truncation flag (`wtrunc`) — fixed

`compute_endpoint_flags` (`QLMatching.cxx:2249`) marks a bundle `window_truncated`
(the `wtrunc` label in `ql_scan`) when its earliest/latest time slice sits within
`window_edge_ticks` of the readout window `[0, readout_window_ticks]`. The slice
indices are raw post-resample ticks (`slice_index = islice->start()/tick`,
`tick = 0.5 µs`).

> **The bug.** `readout_window_ticks` defaulted to the C++ value **3427** — an SBND
> number (`daq.nticks`). PDHD's post-resample SP frame is **5999** ticks (raw data
> **5859** @ 512 ns → 5999 @ 500 ns; the full drift reaches only ~**4498** ticks, so
> the readout window is ~1500 ticks longer than the drift). 3427 sits *below* the
> cathode, so every normal cluster with charge past tick 3427 was falsely flagged
> `wtrunc`.

**Fix — read the window from the data, not a literal.** PDHD's `run_clus_evt.sh`
reads the real frame length (`shape[1]`) from the per-event SP frame archive
(`protodunehd-sp-dnnroi-frames-anode*.tar.bz2`, still in each work dir) and passes it
through `wct-clustering.jsonnet` → `qlmatching.jsonnet` → `QLMatching` via
`readout_window_ticks` (`-S readout_window_ticks=5999`). The value auto-tracks the
real per-run window; the compiled config and the `calib-evt*.json` `readout_nticks`
field both show **5999**.

> *Why not stamp it into the cluster tree?* The clustering chain reads the imaging
> tarballs through `ClusterFileSource`, which rebuilds each slice's frame as a stub
> (`SimpleFrame(ident, start)` — no traces), so the frame length is **not** present
> in the tree at `PointTreeBuilding`/`QLMatching` time. The SP frame (which *does*
> carry it) is read by the run script instead.

**SBND-safe.** `readout_window_ticks` is only set by PDHD's run script; SBND/PDVD keep
their own `qlmatching.jsonnet` value (SBND `3428 ≈ daq.nticks`) untouched →
**bit-identical**.

---

## 3. Light model — what to tune next (point 2)

Predicted PE per channel (`QLMatching.cxx:1043`), summed over every charge point of
the group:

```
pred[idet] += q · QtoL · (dir_vis · VUVEfficiency[idet] + ref_vis · VISEfficiency[idet])
```

`dir_vis`/`ref_vis` come from the `SemiAnalyticalModel` (Gaisser–Hillas VUV
parametrization), loaded from `wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json`.

### 3a. Normalization (charge → PE) — **placeholders, calibrate first**

| knob | PDHD value | role | status |
|------|-----------|------|--------|
| `QtoL` | `1.0` | global charge→light scale | **placeholder** |
| `VUVEfficiency[]` | uniform `0.03` (×160) | per-OpDet VUV detection eff | **placeholder** |
| `VISEfficiency[]` | all `0.0` | per-OpDet reflected (VIS) eff | intentional — no reflected light |
| `doReflectedLight` | `false` | compute VIS term | intentional (PDHD has no cathode WLS foils) |
| GH params, `vuv_absorption_length`, `MaxPDDistance` | in `semi-analytical-pdhd.json` | angular/distance corrections | from duneopdet; not a jsonnet knob |

`QtoL` and `VUVEfficiency` are **degenerate** (only their product sets the absolute
PE scale). Calibrate the product against PDHD data/MC — a single global scale to first
order, then per-channel `VUVEfficiency` if needed. SBND uses a non-uniform
`VUVEfficiency` (`{0, 0.01752, 0.0392}`); PDHD's uniform `0.03` is a stand-in.

### 3b. Light uncertainty (the χ² error model) — **SBND/MicroBooNE values, re-tune**

Per-PMT error feeding χ² (`TimingTPCBundle.cxx:210`): `denom = pe + perr²`,
`χ² += (pred − pe)² / denom`.

| knob | default | role |
|------|---------|------|
| `pe_err_on_pred` | `false` | `false` → use measured opflash `get_PE_err`; `true` → `perr = (pred<knee? floor : frac·pred)` |
| `pe_err_floor` | `0.3` | min error (pred mode) |
| `pe_err_frac` | `0.3` | fractional error (~30 %) of predicted PE |
| `pe_err_knee` | `1.0` | floor↔fractional transition |
| `chi2_relax` | `false` | near-PMT / one-dead-PMT χ² relaxations |
| `chi2_pmt_excess` | `350.0` | **MicroBooNE PE threshold — likely too high for PDHD light yield; re-tune** |
| `chi2_pmt_ratio` / `chi2_pmt_inflate` | `1.3` / `0.5` | excess-relaxation ratio / denom inflation |
| `lasso_flag_weight` | `false` | down-weight LASSO penalty for boundary/truncated bundles |
| `pmt_nonlinearity` (+ `pmt_nl_knee/beta/gamma`) | `false` | per-PMT saturation map (study-grade, off) |

None of these are set in `pdhd/qlmatching.jsonnet` today (all C++ defaults). For
tuning: decide the error source (`pe_err_on_pred`), set the PDHD relative light error
(`pe_err_frac`, the ~30 % core in SBND), and re-scale `chi2_pmt_excess` to PDHD's PE
scale if `chi2_relax` is enabled.

---

## 4. One side with no light (point 3)

In most runs only the `+x` volume (group13) is instrumented; the `−x` side (group02)
reads ~0 PE (run 27980 is the exception — both sides lit). This is handled cleanly:

- **Independent per-group matching.** Each group is a separate `ApaRun`
  (`QLMatching.cxx:478`), so a dark side **cannot pollute** the live side's fit.
- **Same-side visibility.** The photon model returns 0 visibility across the cathode
  (a `−x` cluster predicts 0 PE on `+x` OpDets and vice-versa), and the per-side OpDet
  mask zeroes the other side's channels. A dark-side cluster gets pred≈0, meas≈0.
- **No flash time cut.** `flash_mintime / flash_maxtime` are set to ±1 s
  (`QLMatching.cxx:682`), wider than any PDHD readout, so **every** flash in the
  event reaches matching — the readout-clipping C++ default (±1.5 ms) and even a
  full-readout window are both bypassed. Bit-identical on the run-27305 sample (0
  of 707 flashes ever fell outside the full-readout window); the change only
  guards against a flash landing outside it in some future event/run.
- **Dark flashes dropped.** `flash_minPE = 50` discards near-zero flashes, so dark-side
  clusters simply get **no flash / no T0** (rather than a spurious match).
- **Dead-channel masking** (see [`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md)):
  - static **`ch_mask`** = `[3, 86, 87, 97, 107, 116, 117] + 120..159` — noisy + LArSoft
    dead + the DAPHNE full-stream channels (always 0 PE, never in the opflash);
  - **`auto_mask: true`** (`pe_low=10`, `pe_bright=50`, `neighbors=4`, `min_contrast=1`,
    `min_flash=3`) — per-event drop of a channel that never fires while its live
    neighbours do, catching the **run-dependent** `x<0` dead channels absent from the
    static list. Verified: group13 `nopdet=79` (healthy side, no false masks),
    group02 `nopdet=28` (run 27980, after static+auto mask).

`active_opdet_types = [0]` selects the flat X-ARAPUCAs (PDHD has no SBND-style PMTs,
whose type is `1`).

---

## 5. Other things to watch (point 4)

- **`trigger_offset` absolute sign.** PDHD bakes no readout-vs-trigger offset into the
  charge x; flash times are trigger-relative. `trigger_offset` (≈250 µs = 40 cm, read
  from the opflash archive `offset_us` metadata, plumbed
  `run_clus_evt.sh → wct-clustering.jsonnet → qlmatching.jsonnet → QLMatching` **and**
  the downstream `T0Correction`) folds it into `flash_x_offset`. **Relative**
  consistency (both consumers same sign, 40 cm magnitude) is verified; the **absolute**
  sign (does `+offset` move x toward truth) is still pending physics validation on a
  known in-time track. Default `0` ⇒ bit-identical.
- **`drift_speed = 1.565 mm/µs`** (`params.lar.drift_speed`), the data-calibrated value
  — see [`clustering-algorithm.md`](clustering-algorithm.md).
- **MC vs data:** `data: true` (auto-set `false` when `params.reality == 'sim'`) gates
  the MC-only saturation channel mask.
- **Off by default:** cross-TPC cathode-crossing matching (`xtpc_flag`, needs both
  sides lit) and `require_containment` are both off; the cathode 3-D fiducial is empty
  (flat-cathode window).
- **Semi-analytical JSON lives in `wire-cell-data`**, not this repo
  (`pdhd/photodet/semi-analytical-pdhd.json`); it carries `cathode_x`, the GH tables
  and absorption length.
- **Absolute PE scale is undefined** until `QtoL`×`VUVEfficiency` are calibrated (§3a)
  — matched-flash *identity* is usable now; matched-flash *PE agreement* is not yet.

---

## 6. Optical "op" Bee instance (light + Q/L on the event display)

To **see** the matching in the Bee event display — flashes overlaid on the charge
clusters, with measured vs predicted light — the all-TPC `MultiAlgBlobClustering`
can dump an optical `op` instance alongside the usual `imaging` / `clustering` /
`channel-deadarea` ones.

**Toggle (default OFF → bit-identical).** `save_opflash` is plumbed
`run_clus_evt.sh -op` → `wct-clustering.jsonnet` (TLA `save_opflash`) →
`clus.jsonnet` (`all_tpc(..., save_opflash)`) → the `clus_all_tpc` MABC node. It
only does anything with `do_qlmatch` on (it reads QLMatching's `opflash` root PC),
so `-op` implies `-q`; with it off the clustering output is unchanged.

**What it dumps** (`MultiAlgBlobClustering::fill_bee_flashes`, written at the same
*pre-pipeline* point as the `img` charge dump, where the per-APA matched clusters'
1:1 flash association and cluster ids are still intact). One JSON
`data/<idx>/<idx>-op.json` per event, with per-flash arrays ordered by ascending
flash time:

| field | meaning |
|-------|---------|
| `op_t` | flash time (µs) |
| `op_pes` | **measured** PE per OpDet channel (the light) |
| `op_pes_pred` | **Q/L predicted** PE per channel, element-wise summed over the matched clusters (the matching) |
| `op_cluster_ids` | matched cluster id(s) — same enumeration as the charge dump, so a flash links to its physical cluster |
| `op_peTotal` | total measured PE of the flash |
| `apa` | `gid / 1000000` (the flash's APA/side) |

A matched flash carries its cluster id(s) + predicted light; an unmatched flash
emits empty `op_cluster_ids` / `op_pes_pred`. Only matched clusters with total
predicted light ≥ 100 PE contribute a prediction (same cut as the legacy
`dump_light`).

**Build a combined link.** `run_bee_combined_evt.sh` already copies every
`data/0/0-*.json` out of `mabc-all-apa.zip`, so once the events are clustered with
`-op` the `0-op.json` is folded into the upload automatically — no builder change
needed. Select the **op** instance in Bee to step through flashes and compare
`op_pes` vs `op_pes_pred`.

---

## Run it

```bash
# from pdhd/ (this repo)
./run_clus_evt.sh -q  <run> <evt>     # -q / PDHD_QLMATCH=1 enables Q/L matching
./run_clus_evt.sh -calib <run> <evt>  # + dump candidate bundles for ql_scan viewer
./run_clus_evt.sh -op <run> <evt>     # + dump the optical "op" Bee instance (§6); implies -q
```

Output `mabc-all-apa.zip` carries the matched flash/T0 per cluster (and the `op`
instance when `-op`). The [`ql_scan`](ql-scan-display.md) viewer merges an event's
group02 + group13 dumps into one two-side view; for the Bee event display use `-op`
+ `run_bee_combined_evt.sh` (§6).

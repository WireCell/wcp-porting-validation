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

> **One-line status.** The chain runs end-to-end and produces matched flashes/T0s, the
> geometry is correct (combined two-APA box), and the **full SBND-derived feature suite is
> now enabled and tuned** against the run-29107 4-event hand scan (light model + per-channel
> gain + the `highconsist_ladder` / `empty_rescue` / `chi2_pmt_excess` features — see §"Parity
> with the SBND Q/L chain"). The light **uncertainty** model is calibrated (`pe_err`); the
> remaining open item is the **absolute** charge→PE normalization (`QtoL`×`VUVEfficiency`),
> which sets matched-flash PE *agreement* — matched-flash *identity/T0* is usable now.

---

## 1. The chain

PDHD images **two drift volumes** separated by an opaque central cathode at `x = 0`.
Each volume contains **two APAs offset along Z (beam)**:

| group   | APAs | drift side | imaging face | z coverage |
|---------|------|-----------|--------------|------------|
| group02 | 0, 2 | `x < 0`   | `tpc_face=0` | APA0 z[0,230] + APA2 z[232,463] cm |
| group13 | 1, 3 | `x > 0`   | `tpc_face=1` | APA1 z[0,230] + APA3 z[232,463] cm |

Q/L matching is inserted **between clustering stage 3 (per-drift-group) and stage 4
(all-TPC)** — `pdhd/wct-clustering.jsonnet` under the `do_qlmatch` switch (default off →
historical no-matching chain). Both drift sides enter **ONE joint QLMatching node**
(`matching_joint`, `nin=2`, like SBND) so the cross-cathode consistency pass can see both:

```
opflash (group02) ─▶ TensorFileSource ─▶ FlashTensorToOpticalPCs ─┐ (port 0, side 0)
   stage-3 group02 cluster tree ──────────────────────────────────┘
                                                                    ├▶ QLMatching ─▶ all-TPC
opflash (group13) ─▶ TensorFileSource ─▶ FlashTensorToOpticalPCs ─┐ (port 1, side 1)  (premerged)
   stage-3 group13 cluster tree ──────────────────────────────────┘   (nchan=160, joint)
```

- **`TensorFileSource`** (`opflash_source`) reads the per-event opflash archive.
- **`FlashTensorToOpticalPCs`** (`flash_attach`, `nchan=160`, `correct_flash_time:false`)
  expands the opflash matrix into the canonical flash / light / flashlight PCs on the
  cluster root. Port 0 = cluster tree, port 1 = opflash matrix; one per drift side.
- **`QLMatching`** (`matching_joint`) takes both sides on input ports 0 (side 0) and 1
  (side 1). Each side is still matched **independently** in its own `ApaRun` — `anodes`,
  `grouping_anodes` and `tpc_faces` are **per input**, so each run keeps its own drift
  geometry, two-APA box and imaging face. The node then runs the cross-cathode (xTPC)
  pass over both sides (below), emits **one** merged tree (the all-TPC stage is
  `premerged` → no PointTreeMerging), and writes the per-cluster matched-flash scalar
  (`Cluster::get_flash()`), T0, `matched_flash_gid`, and `xtpc_consistent`.

Earlier PDHD wired **two** separate per-side nodes (one input each); the central cathode
is opaque so the two volumes are physically independent and matched separately. They were
merged into one node solely so the cross-cathode pass has both halves co-resident — a
cathode-crosser is imaged as **two clusters, one per volume**, which two separate nodes
never see together (`xtpc_flag` was therefore inert). The per-side fits are unchanged by
the merge: with `xtpc_flag` off, the joint node's side-0 output is **byte-identical** to
the old separate `group02` dump (run 29107 evt 983, 1098/1098 bundles).

<a name="group-aware-matching"></a>
**Group-aware (main + associated), then recomposed.** Stage-3 per-group clustering
tags blobs with an `isolated/perblob` array (−1 = main, ≥0 = sub-clusters).
`QLMatching` splits each group into a main + associated clusters
(`decompose_cluster_groups`, `QLMatching.cxx:750`), predicts light over the
**whole** group, and anchors the bundle on the main. The matched T0 is written to
the main and copied to its associated clusters.

> **PDHD: the −1 main tag is no longer produced (2026-06-14).** `examine_bundles()`
> — the only stage-3 pass that writes a `−1` main into `isolated/perblob` — is now
> **commented out** of the PDHD per-drift-group pipeline (`cfg/.../pdhd/clus.jsonnet`;
> see `clustering-algorithm.md` §Stage 3). The `perblob` then comes from `isolated()`,
> which carries **no −1**, so `decompose_cluster_groups` treats every cluster as a
> single component and does **no** main/associated split: each cluster is matched and
> flagged **whole**. Motivation: the boundary flags (`compute_endpoint_flags`) walk the
> **main** sub-cluster only, so when `examine_bundles` split a crossing-cosmic track
> into a short main spine + an associated continuation, a track whose anode end lived
> in the *associated* piece was never flagged `at_x_boundary`/`close_to_PMT` even though
> the full track reached the anode (run 27305 evt 150 cluster 386: main spine stopped
> ~63 cm short). With the whole track as the main, the flag fires correctly. This
> changes **flags only** — at stage 3 `examine_bundles` (run `use_flash_t0=false`)
> rewrites `perblob` but never cluster membership, and the predicted light was already
> summed over the whole group either way — so matching is unchanged (verified evt 150:
> 163→163 clusters point-for-point, 34→34 Q/L matches with identical flash/χ²; only
> cluster 386 flips its boundary flags to `True`). The main/associated machinery below
> still applies to any detector that keeps `examine_bundles` (e.g. SBND).

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
(`QLMatching.cxx:857`) from the `DetectorVolumes` service — nothing is hard-coded to
SBND.

**Representative anode → side.** Each `ApaRun` is keyed by a representative anode
(`anodes[0]`: APA0 for group02, APA1 for group13). Its ident (0 or 1) sets:

- `run.sign_offset = (tpc==0) ? −1 : +1` — the drift-direction sign
  (`QLMatching.cxx:860`).
- the **per-side OpDet mask**: an OpDet at `x < cathode_x` belongs to TPC 0, else
  TPC 1 (`QLMatching.cxx:936`). `cathode_x = 0` comes from the semi-analytical model
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

(`QLMatching.cxx:1081`; `drift_speed = 1.576 mm/µs`, see
[`clustering-algorithm.md`](clustering-algorithm.md) drift-velocity calibration).
Each cluster point is shifted by `flash_x_offset` before the light prediction and the
**PE-inclusion gate** (`QLMatching.cxx:1150`):

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
> upstream half; the transverse gate at `QLMatching.cxx:1151` then **dropped all
> charge in the downstream APA** (APA2/APA3) from the predicted-light sum — half of
> each drift volume produced no predicted light, so a cluster in that half could not
> be matched (and a track spanning both APAs under-predicted).

The fix (`QLMatching.cxx:881`) unions `inner_bounds` over the group's anodes for the
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

`compute_endpoint_flags` (`QLMatching.cxx:2426`) marks a bundle `window_truncated`
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

### Bundle prefilters: containment + over-prediction (enabled for PDHD)

PDHD now enables the two geometry/light bundle prefilters in `qlmatching.jsonnet`
(both C++-default **OFF**, so any config that does not set them is bit-identical):

```jsonnet
require_containment: true,
reject_overpred: true,  overpred_total_ratio: 3.0,  overpred_maxch_ratio: 10.0,
```

- **`require_containment`** (`QLMatching.cxx:1124`) drops a (flash, cluster) bundle
  whose cluster is not contained in the TPC drift box once the flash-T0 x-offset is
  applied. The box is the **two-APA union** from `compute_geometry` (the same box that
  feeds the `at_x_boundary`/`two_boundary` flag walk — see *Combined two-APA box*
  above), so on each ∓x drift side it spans both APAs (z ≈ 0…4.6 m). The cathode-side
  window edge **`cathode_ext1` is 1.5 cm** (C++ default 1.2 cm) and the flag-only lower
  edge **`cathode_ext2` is −3.0 cm** (C++ default −2.0 cm). The genuine cathode-crossers
  scatter ±~1.75 cm around the cathode (the ±1.5–2 cm t0/velocity/SCE drift residual), an
  irreducible spread; `drift_speed = 1.576` is chosen so the most-overshooting crosser
  sits *just inside* the cathode (run 29107 evt 983 clus 62 +0.84 cm), which lets `ext1`
  revert to ~the C++ default (+0.3 cm noise cushion) instead of the 2.5 cm a centered
  1.580 required, and pushes the undershoot residual onto the benign `ext2` flag window
  (clus 96 −2.61 cm → ext2 −3.0). `ext1` is the biting lever (containment + PE inclusion);
  `ext2` only flags more near-cathode ends `at_x_boundary` and cannot drop a bundle. See
  the drift-velocity / cathode-window calibration in `clustering-algorithm.md`.
  - **Robust-endpoint trim before the box test** (`compute_endpoint_flags`,
    `robust_endpoint_trim: true`). Before containment is tested, the cluster's two drift
    endpoints (shallowest `first_u` at the anode, deepest `last_u` at the cathode) are
    snapped inward past thin **outer straggle** so a few stray points cannot veto an
    otherwise-contained track. The original judge is **point count**: trim the outer
    slices if their points `≤ max(robust_endpoint_frac·N, robust_endpoint_count)`
    (`0.01·N`, floor 15). This misses **overclustered satellites** — detached low-charge
    material wrongly attached to a cluster — that carry *many* points yet negligible
    charge: run 29107 evt 991 cluster ident-31 (`uid 1000031`, +x side) has **91 points
    past the anode tolerance** (≫ the 15-pt floor, so the count judge keeps them and
    `require_containment` vetoes the bundle), but those 91 are **~0.1 % of cluster charge**
    and **~50× lower per-point charge** than the real track body, which ends *just inside*
    the anode. Two charge knobs (both C++-default `0` ⇒ disabled ⇒ byte-identical) add a
    charge-aware path, **OR**'d with the point-count judge so the original rescue is never
    weakened:
    - **`robust_endpoint_charge_frac`** (PDHD **0.005**): also trim the outer straggle when
      its charge `≤ frac · Q_cluster`. A dense genuine track-end exceeds this and is left to
      fail the box test.
    - **`robust_endpoint_charge_abs`** (PDHD **1500**): a **size-independent** per-point
      charge-**density** ceiling that gates *only* the charge-fraction path — trim the outer
      material only if it is charge-*sparse* (`q_out/pts_out < 1500`, diffuse overclustered
      satellites ≈ 150 q/pt here), **never** a charge-*dense* real track tip running into the
      boundary (≈ 2500–8800 q/pt). Because it is per-point density, not an aggregate fraction,
      it does not scale with cluster size or tail length — it cleanly separates a satellite
      from a genuine boundary-piercing tail regardless of how large the cluster is.
    - **`robust_endpoint_gap`** (PDHD **3.0 cm**) + **`robust_endpoint_gap_charge_frac`**
      (PDHD **0.01**): a **density-blind detachment-gap** path (anode end only) for
      **moderately-dense overclustered junk** that the charge-density gate above cannot
      catch. Run 29107 evt 1007 cluster `uid 126` (a real ~106 cm track, GT-matched to flash
      gid 72 @ 850 µs) carries 7 gap-separated junk groups marching from `u ≈ −111 cm` up to
      the body at **~2000 q/pt** — ~10× the diffuse satellites the `charge_abs` gate was tuned
      on, and *overlapping* the genuine track-end band (~2500 q/pt), so raising `charge_abs`
      would amputate real anode-crossers. The robust discriminator here is **detachment, not
      density**: junk sits in groups separated from the contiguous body by large drift gaps,
      whereas a real anode-piercing tip is continuous (~0.08 cm tick spacing). Trim the
      sub-anode material iff it contains a drift gap **> `robust_endpoint_gap`** AND carries
      **< `robust_endpoint_gap_charge_frac · Q_cluster`**. This path is also OR'd with the
      others, so no existing rescue is weakened; C++ default `0` ⇒ disabled ⇒ byte-identical.

    Validated bundle-level on 10 events of run 29107 (the reliable metric while the chain is
    tuned): **0 bundles lost** (the point-count path is preserved byte-identically), every
    contiguous real-track-tail case correctly **left to fail** (so the density gate adds no
    forced matches), and only genuine low-density satellites rescued — including the evt 991
    target `uid 1000031` (165 q/pt). Passing containment only makes a bundle *eligible*; the
    χ²/KS fit still decides selection. The **gap path** was A/B'd (gap=0 vs gap=3) across all
    30 events of run 29107: **28 bundles enabled, 0 removed**, exactly **1 new correct match**
    (the evt 1007 `uid 126 ↔ gid 72` rescue), 2 benign t0-flips of ~0-PE noise clusters, and
    **zero fake matches / zero GT regression**; cluster `uid→npoints` identical across passes.
- **`reject_overpred`** (`QLMatching.cxx:1208`) drops a bundle, before the χ² fit, when
  its predicted light hugely exceeds the measured light over the masked PMT set —
  `Σpred/Σmeas > overpred_total_ratio` **or** `pred/meas` at the brightest predicted
  PMT `> overpred_maxch_ratio`. Boundary/window-truncated bundles (including every
  `at_x_boundary` two-boundary light anchor) are **exempt**.

**The ceilings are now GT-tightened to (3.0, 10.0)** (from the earlier deliberately-loose
provisional 5.0 / 25.0). The run-29107 **evt-983 hand scan** provides PDHD ground truth:
of its 31 labeled matches, **24 are boundary/window-truncated (overpred-exempt)** and only
**7 are subject** to this cut; their worst values are `R_total` = 1.90 and `R_max` = 5.04
(`mc 1000020`, a genuine match that over-predicts one channel 5×). The optical-model retune
(`vuv_eff` 0.0145→0.01254 + the APA0 measured scale) also tightened genuine over-prediction,
so the loose ceilings are no longer needed. **(3.0, 10.0)** keeps every GT match with
~1.6× / 2.0× margin while culling the egregious tail (auto-selected winners run out to
`R_max` ≈ 18). `R_max` is held ≥ 8 because `mc 1000020` legitimately over-predicts one
channel 5× and this is single-event GT. **Verified on evt 983: all 31 GT matches survive;
auto-selected 92→94** (pre-LASSO pruning let two cleaner matches win). SBND keeps its own
(2.9 / 4.3, 10 GT events). (`ql_light_calib/containment_overpred_check.py` reproduces the
run-wide distributions.)

**Both cuts are PRE-LASSO** — they prune candidates and trigger a global re-solve, so
their effect is *not* the per-bundle cull count. Reprocessing all 30 events of run 29107
(no-cut → cuts on):

| metric (matched flashes, 30 evts) | before (no cut) | after (containment + overpred) |
|---|---:|---:|
| matched flashes | 1399 | 1428 |
| median best-KS over matched flashes | 0.511 | **0.407** |
| strict `ks<0.2` two-boundary anchors | 3 | **14** |
| paired ΔKS on the 799 still-matched flashes | — | **median 0.000** (41% better, 20% worse) |

The cuts **swap ~44 % of the (noise-dominated, ~47 flashes/event) matched set** while
holding the count flat: the ~600 dropped matches had median best-KS **0.87** (70 % with
KS > 0.5 — junk noise-flash matches), the ~630 that replace them have median KS **0.43**,
the matches that stay are unchanged in the median, and the strict anchor set grows ~5×.
Of the ~34 good (`ks<0.2`) before-matches that go unmatched, **31 are LASSO reshuffles**
(the cluster was not culled — it lost the flash to re-competition) and only **3 are
reject_overpred culls**, all with `R_max` 38–177 (genuine pathological single-channel
over-prediction, well past the p99 = 21 model-roughness range). The change is *not*
attributed per-cut (both reshuffle LASSO); the offline check only establishes that
containment culls **0 winners directly**.

> **Note on the light calibration.** `vuv_eff = 0.0145`, λ = 300 cm
> (`ql-light-normalization-study.md`) were tuned on the **no-cut** dumps. These
> prefilters change the matching that future calib dumps reflect; the anchor set grew
> (3 → 14), so the calibration is not invalidated, but it was derived pre-filter.
> A later **hand-scan label retune** (run 29107 evt 983) moved `vuv_eff` to 0.01254
> and added a per-channel `measured_pe_scale` + `pe_err_frac` retune — see **§3c**.
> The **4-event** retune (evts 983/991/999/1007) supersedes it: `vuv_eff` **0.01281**,
> APA0 `measured_pe_scale` **1.14** (was 1.57, over-scaled on the single event), λ kept
> **300** (KS degenerate; N90+integral pin 300). A subsequent **per-channel (per-PD) gain
> calibration** then replaces the uniform APA0 1.14 with a **block × type** array (FBK/HPK
> SPE-template types) + 12 outlier overrides — the APA0 1.14 splits into FBK **1.20** /
> HPK **1.00** — and keeps **`pe_err_frac` 0.40** (the high-PE method-of-moments value;
> an interim 0.60 that calibrated chi2/ndf→1 was reverted — chi2/ndf≈1 is not required and
> the larger frac is a looser, mismatch-tolerant matching error). See
> `ql-light-normalization-study.md` §"4-event hand-scan label retune" and §"Per-channel
> (per-PD) gain calibration".

### Cross-side mismatched-candidate filter (enabled for PDHD)

```jsonnet
cross_side_filter: true,
```

A contained bundle can still pair a cluster with a flash lit on the **opposite** drift
side (`require_containment` is a geometry-only box test; the cluster fits the box at that
flash's T0 even though the flash lit the other volume). With the opaque cathode this is
non-physical — a `+x` cluster predicts ~0 PE on the `−x` OpDets the flash measured, and
vice-versa — so the bundle clutters the candidate pool and the hand-scan with light it
can never match. The **one** physical exception is a genuine **cathode-crosser**: at that
flash's T0 the cluster's drift-corrected end sits at the cathode (`at_x_boundary`), so its
far half *can* be the source of the lit side's light (its own-side flash may be missing or
below threshold — note the `−x` full-stream gaps). **`cross_side_filter`** (`QLMatching.cxx`,
`cross_side_mismatch_drop`) therefore **keeps same-side bundles and cross-side
cathode-crossers, and drops every other cross-side bundle.** Brightness is *not* the test
(a bright crosser is as valid as a dim one; a mid-drift cluster merely contained at some
opposite flash's T0 is a coincidence). The flash's lit side is read from where its measured
PE sits (OpDet x vs the cathode plane — the same rule the merged-root opflash PC uses).

- The test is applied at **bundle creation** (`build_bundles`, removed from the pre-LASSO
  pool) *and* in **`dump_calib`**, so the fit candidate set and the hand-scan candidate
  tables show the identical universe. C++ default **OFF** → bit-identical for SBND/ICARUS.
- Matching-neutral in practice: cross-side bundles never won the LASSO (no light overlap
  on the measured side), so dropping them changes no winner; they only cluttered the pool.
  Run 29107 evt 983: of 4911 contained candidates the **2467 same-side and 115 cross-side
  cathode-crossers are kept**, the **2329 non-crosser cross-side are dropped**. The kept
  crossers stay as hand-scan candidates — there is **no auto-T0**, the human assigns their
  flash (cross-side light cannot adjudicate the T0). (`check_dim_crossside.py`,
  `verify_filter.py`.)
- A worked case: run 29107 evt 983 Bee cluster 22 (= dump ident **70**, side 0) is a
  cathode-crosser whose `−x` half made no flash; at the −1807.1 µs S1 flash's T0 it
  drift-corrects to the cathode (`at_x_boundary`), so its cross-side bundle is **kept** and
  it appears as a candidate in that S1 group (an earlier `total_pred_light < 10` PE rule
  wrongly dropped it as "bright"). The same cluster's mid-drift non-crosser candidates at
  other flash T0s are dropped.
- The viewer collapses the duplicate "phantom" flash copies so a kept cross-side crosser
  bundle is reachable in **both** the navigation table and the per-cluster Compare table —
  see [`ql-scan-display.md`](ql-scan-display.md). The cluster-**length** cut still gates
  the navigation table, so short fragments stay hidden there.

### Cross-cathode (xTPC) cathode-crossing pairing (enabled for PDHD)

With both drift sides in the joint node, the cross-cathode pass (`cull_cross_tpc`,
`xtpc_flag:true`) pairs a cathode-crosser's two halves across the central cathode. It
gathers the cathode-relevant candidates of each side (`at_x_boundary` cathode-end or
`window_truncated`), and for every **coincident** (Δt < `flash_group_window`) pair of a
**side-0** and a **side-1** candidate tests geometric consistency: **scenario 1** —
closest approach < `xtpc_dmax` (5 cm, both cathode ends present, tight/self-vetoing; sets
`xtpc_scenario1`); **scenario 2** — a truncated half whose connecting vector is collinear
(< `xtpc_angle_max` = 20°) with both Hough directions and within `xtpc_dmax2` (300 cm). A
passing pair stamps `xtpc_consistent` on **both** halves.

The pairing is on **drift side** (low-x volume = 0 / high-x volume = 1, from the run's
anode-vs-cathode position), **not** the raw APA ident — so the same code is bit-identical
for SBND (its two APAs already are the two sides) and works for PDHD (side 0 = APAs 0,2;
side 1 = APAs 1,3). The C++ knobs `tpc_faces` (per-input face list) and a per-input
`grouping_anodes` make the joint multi-face node possible; both default to the single-node
behaviour so every other config stays bit-identical (SBND joint output verified
byte-identical before/after).

> **This is NOT observation-only for PDHD.** `xtpc_flag:true` also activates the
> **xtpc-priority cull** in `cull_inconsistent` (`QLMatching.cxx:1313`): a cluster with a
> scenario-1 xTPC bundle keeps that bundle and drops its rivals, which re-steers matching
> and ripples through the per-side LASSO. On the **current** config the combined joint dump
> for run 29107 evt 983 carries **20 `xtpc_consistent` bundles forming 6 cross-cathode
> crosser pairs** (5 scenario-1 + 1 scenario-2), e.g. side-0 cluster 70 ↔ the side-1 flash
> at −1807 µs (scenario 1); all are `contained` (none spill past the cathode), and
> `cull_cross_tpc` is confirmed firing (349 coincident cross-side pairs).
> *(An earlier pre-fit-advantage measurement — against the 30-label scan and before
> `lasso_flag_weight`/`chi2_relax`/the overpred tighten — found the cull alone reassigned
> 53/1098 side-0 bundles and raised `auto_selected` agreement 3 → 5. The current combined
> auto-selection gain from the boundary/crosser fit-advantage is the +4 (18 → 22) in the
> `lasso_flag_weight` row above.)*
>
> **The no-flash-on-one-side crosser** (the user's case) is paired and flagged, but its
> cross-side bundle stays `auto_selected=false`: a side-0 cluster predicts ≈0 light on the
> side-1 OpDets (opaque cathode), so it can never win the light fit — xTPC *annotates* the
> pair rather than forcing an impossible light-match. The hand-scan viewer surfaces it
> (navigable, with the flag) so a human can confirm it.

### Parity with the SBND Q/L chain — now complete

PDHD now runs the **full SBND-derived Q/L suite**: containment + over-prediction +
cross-side filter + cross-cathode xTPC, the flag fit-advantage levers (`lasso_flag_weight`
+ `chi2_relax`), the per-channel `measured_pe_scale` + low-PE error model, and — as of the
run-29107 4-event hand-scan tuning (June 2026) — the three previously-deferred features:
`highconsist_ladder`, `empty_rescue`, and the PDHD-rescaled `chi2_pmt_excess`. Every knob
below is now **enabled and tuned** against the hand scan (each remains C++-default-OFF, so
SBND/PDVD/ICARUS stay byte-identical). The cumulative effect on the 4 events (vs the
ladder-off frac=0.4 baseline): **GT-accept 53 → 72** (xTPC 12 → 14, non-xTPC 41 → 58),
**human-rejected re-selections 219 → 165**, **0 baseline xTPC winners lost**.

| SBND knob | what it does | PDHD status / why deferred |
|---|---|---|
| `empty_rescue` (+`rescue_metric_max`) | a flash emptied by LASSO adopts its best light-quality orphan cluster | **DONE** — enabled, `rescue_metric_max` **0.20** (exponent/boundary_weight 0.8/0.8 = SBND). Unlike SBND (misses timing-degenerate, ~1/5 recoverable) PDHD strands clean GT on LASSO-emptied flashes: **+14 GT-accept (44→58), 0 steals, 0 GT lost, 0 xTPC lost**, reject-reselected −6. See §3e |
| `highconsist_ladder` (`flag_high_consistent`) | multi-branch KS/χ² quality ladder (clean / good / two-boundary / miss) gating the pre-LASSO `cull_inconsistent` purity cull | **DONE** — enabled. **KS ceilings = SBND** (KS is the purity lever: on the 4-event hand scan ks<0.10 is 88% pure, 57 GT vs 8 junk); **chi2/ndf ceilings RAISED 6/4/8 → 35/35/35** (PDHD's rougher light model runs a clean-KS GT match to chi2/ndf~32 vs SBND ~1–2; the SBND ceilings amputated clean-KS GT). 4-event GT (vs ladder-off, frac=0.4): **+5 net GT-accept (53→58), xTPC 12→14 (0 baseline xTPC lost), human-rejected re-selected −48 (219→171)**; cost = 4 displaced clean GT (§3d) |
| `lasso_flag_weight` | down-weights L1 for boundary/truncated bundles (incl. xTPC crossers) so they survive the strength cutoff | **DONE** — `lasso_boundary_weight` **0.2** (= SBND), confirmed optimal by a 4-event sweep {0.1, 0.2, 0.3}: **0.1 FAILS the xTPC hard gate** (over-protection admits boundary noise that displaces a crosser, 14→13); **0.3** costs 2 confirmed-GT boundary matches for only −4 reject; **0.2** maximizes GT-accept (xTPC 14 + non-xTPC 58) with xTPC intact |
| `chi2_relax` | widens χ² denom for measured excess at near-PD channels + drops a dead-PD worst channel | **DONE** — enabled; `chi2_pmt_excess` re-scaled **350 → 100 PE** for PDHD's lower ARAPUCA yield (genuine near-PD excess: median ~22 PE, p90 ~165, saturation tail ~10k). It IS active (softens ~27 close_to_PMT bundle χ²/event) but **selection-neutral** (the KS-led ladder with loose c2n=35 ceilings absorbs it) — live-but-benign, like SBND. `ratio`/`inflate` kept 1.3/0.5 |
| `pe_err_on_pred` + retuned `pe_err_floor/frac/knee` | χ² error from predicted (not measured) PE, SBND-tuned magnitudes | **DONE** — `pe_err_on_pred` enabled (PDHD `true`) + low-PE inflation; `pe_err_frac` `0.40` (§3c) |

**Round-by-round 4-event progression** (run 29107 evts 983/991/999/1007; `validate_chain.py`).
Primary FOM is the **hand-scan GT delta** (GT-accept + reject-reselected) — chosen because it is
**independent of the ladder's own KS-cull mechanism**, unlike the global winner-KS shift (the
ladder mechanically culls high-KS bundles, so "more low-KS winners" would be circular):

> **Production smoke.** The final config (now the `run_clus_evt.sh` default, not a branch
> experiment) was run on the pathological **evt 1015** (~900 flashes, the resource cliff):
> clean exit, **233 s wall, 1.06 GB peak RSS** — the per-empty-flash `empty_rescue` and
> per-cluster ladder cull add negligible overhead.
>
> **In-sample caveat.** These 4 events are the *only* labeled PDHD hand scans, so every moved
> threshold (`hc_*_c2`=35, `rescue_metric_max`=0.20, the lasso choice) was both **tuned and
> validated on the same events** — there is no held-out set, so **+19 GT-accept is an in-sample
> number, not a generalization claim**; out-of-sample behavior is unverified (revisit as more
> events are scanned). Likewise the −54 "reject re-selected" is **purity on the reviewed set
> only**: it tracks the 228 originally-reviewed matches, while the 70 new off-scan winners are
> *un*reviewed and some are surely junk (rule 2 — small clusters were not exhaustively scanned).

| round | change | GT-accept (xTPC + non-xTPC) | xTPC (HARD) | human-rejected re-selected |
|---|---|---:|---:|---:|
| 0 | `pe_err_frac` 0.60 → 0.40 (baseline) | 53 (12+41) | 12/12 kept | 219 |
| 1 | `highconsist_ladder` (KS=SBND, c2n→35) | 58 (14+44) | **14** | 171 |
| 2 | `chi2_pmt_excess` 350 → 100 | 58 (14+44) | 14 | 171 (neutral) |
| 3 | `empty_rescue` (`rescue_metric_max` 0.20) | **72** (14+58) | 14 | 165 |
| 4 | `lasso_boundary_weight` sweep → keep 0.2 | 72 (14+58) | 14 | 165 |

Net (full chain vs frac=0.4 ladder-off baseline): **GT-accept 53 → 72** (+19; ~9 recovered by
the ladder net of 4 displacements, +14 by `empty_rescue`), **xTPC 12 → 14 with 0 baseline
xTPC winners lost**, **human-rejected re-selections 219 → 165** (−54). The 70 off-scan new
winners (rule 2: small clusters were not exhaustively hand-scanned) are neutral.

**Final shipped selection vs the hand scan (absolute, not a baseline delta).** Of the **125**
hand-scan matches across the 4 events, **16 are xTPC annotation-only crossers** (no flash on one
side → not light-selectable by design; they stay annotated), leaving **109 auto-selectable**.
How the shipped config selects them (cluster matched to the *same* flash = reproduced; to a
*different* flash = reassigned, better/worse by the final model's KS for the two flash options):

| hand-scan match outcome | count | of 109 |
|---|---:|---:|
| **reproduced** (same cluster↔flash) | **82** | 75% |
| reassigned to a **better**-KS flash (code improved on the pick) | 15 | 14% |
| reassigned to a **worse**-KS flash | 10 | 9% |
| genuinely **dropped** (cluster unmatched) | 2 | 2% |

So **good (reproduced or improved) = 97/109 (89%)**; changed-for-worse = 10; dropped = 2 (the
documented displacement cost — see §3d). "Better/worse" is by light-agreement KS, not a
physics-truth claim.

**Additional matches** (final winners on clusters the human did *not* hand-scan): **268** across
the 4 events. These are **not** all good — the matcher assigns a best-effort cluster to every
flash above threshold, including noise flashes: only **~38 are plausibly real** (22 at ks<0.10
clean + 16 at ks 0.10–0.20), 74 are marginal, and **156 are the noise-flash tail** (ks>0.5).
So beyond the hand scan the chain adds ~38 credible new matches per the 4 events; the rest is the
expected noise-flash assignment continuum (rule 2 — these were never reviewed).

> **Known remaining SBND/PDHD difference — `bundle_mask_ks` (deliberate, not an oversight).**
> SBND sets `bundle_mask_ks: true` (apply `ch_mask` + the saturation mask to the **KS** shape
> metric, not just χ²/LASSO); PDHD leaves it at the C++ default **`false`**, so PDHD's KS is
> computed over all 160 channels including its 7 dead PDs (3/86/87/97/107/116/117) and any
> unfired full-stream PDs. This **touches the KS lever the ladder is now keyed on** — masking
> those channels would *lower* KS for matches near a dead PD. It is left off **on purpose**:
> the ladder's KS ceilings and the `rescue_metric_max` were tuned on the **unmasked** KS in
> these dumps, so enabling it shifts every KS and would require re-checking `hc_*_ks` and the
> c2n=35 ceilings. Candidate refinement for a future round (with re-tuning), not a silent flip.

---

## 3. Light model — what to tune next (point 2)

Predicted PE per channel (`QLMatching.cxx:1169`), summed over every charge point of
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
| `VUVEfficiency[]` | uniform `0.01281` (×160, `= vuv_eff`) | per-OpDet VUV detection eff | **4-event label retune (§3c)** — 0.01254 evt-983, 0.03 placeholder |
| `VISEfficiency[]` | all `0.0` | per-OpDet reflected (VIS) eff | intentional — no reflected light |
| `doReflectedLight` | `false` | compute VIS term | intentional (PDHD has no cathode WLS foils) |
| GH params, `vuv_absorption_length`, `MaxPDDistance` | in `semi-analytical-pdhd.json` | angular/distance corrections | from duneopdet; not a jsonnet knob |

`QtoL` and `VUVEfficiency` are **degenerate** (only their product sets the absolute
PE scale). Calibrate the product against PDHD data/MC — a single global scale to first
order, then per-channel `VUVEfficiency` if needed. SBND uses a non-uniform
`VUVEfficiency` (`{0, 0.01752, 0.0392}`); PDHD's `VUVEfficiency` is uniform `0.01281`
(`= vuv_eff`, the evt-983 label retune — §3c; was a `0.03` stand-in).

### 3b. Light uncertainty (the χ² error model) — **SBND/MicroBooNE values, re-tune**

Per-PMT error feeding χ² (`TimingTPCBundle.cxx:117`): `denom = pe + perr²`,
`χ² += (pred − pe)² / denom`.

| knob | default | role |
|------|---------|------|
| `pe_err_on_pred` | `false` (**PDHD `true`**) | `false` → use measured opflash `get_PE_err`; `true` → predicted-based `perr` |
| `pe_err_floor` | `0.3` | min error (pred mode) |
| `pe_err_frac` | `0.3` (**PDHD `0.40`**) | high-pred fractional error of predicted PE (0.40 = high-PE method-of-moments on the hand-scan matches; an interim 0.60 calibrating chi2/ndf→1 was reverted — chi2/ndf≈1 not required, 0.60 over-tolerated mismatch) |
| `pe_err_knee` | `1.0` | floor↔fractional transition (unused when low-PE inflation on) |
| `pe_err_lowpe_frac` / `pe_err_lowpe_knee` | `-1`(off) / `4.0` (**PDHD `1.55` / `5.5`**) | low-PE detection-inefficiency error inflation: `rel = frac + (lowpe_frac−frac)·exp(−pred/knee)`, `perr = √((rel·pred)²+floor²)`. See `ql-lowpe-efficiency-study.md` |
| `chi2_relax` | `false` | near-PMT / one-dead-PMT χ² relaxations |
| `chi2_pmt_excess` | `350.0` (**PDHD `100.0`**) | near-PD measured-excess PE threshold; re-scaled to PDHD ARAPUCA yield (active but selection-neutral) |
| `chi2_pmt_ratio` / `chi2_pmt_inflate` | `1.3` / `0.5` | excess-relaxation ratio / denom inflation |
| `lasso_flag_weight` | `false` | down-weight LASSO penalty for boundary/truncated bundles |
| `pmt_nonlinearity` (+ `pmt_nl_knee/beta/gamma`) | `false` | per-PMT saturation map (study-grade, off) |

Of these, **`pe_err_frac`** (→ `0.40`, high-PE method-of-moments; **§3c**),
**`chi2_relax`** (→ `true`) and **`lasso_flag_weight`** (→ `true`, `lasso_boundary_weight`
0.2) are now set in `pdhd/qlmatching.jsonnet`; the rest are C++ defaults. `chi2_relax`'s
excess-widening term (`chi2_pmt_excess` 350 PE) is SBND-PMT-scaled and largely inert at
PDHD ARAPUCA PE levels, so its active effect here is the dead-PD worst-channel drop — the
excess thresholds still want a PDHD re-scale. For further tuning: decide the error source
(`pe_err_on_pred`) and re-scale `chi2_pmt_excess`.

### 3c. Per-channel measured-PE gain correction + evt-983 label retune (enabled for PDHD)

> **Superseded values.** This section documents the *first* (evt-983) retune for
> provenance. The shipping values are the **4-event** retune (`vuv_eff` 0.01281, λ 300)
> and the **per-channel (per-PD) gain calibration** that replaces the uniform APA0 1.57
> with a block × type `measured_pe_scale` array (APA0 → FBK 1.20 / HPK 1.00) + 12 outlier
> overrides, and keeps `pe_err_frac` **0.40**. See
> `ql-light-normalization-study.md` §"4-event …" and §"Per-channel (per-PD) gain
> calibration"; scripts `ql_light_calib/fit_labels_multi.py` + `fit_perchannel_scale.py`.

A hand scan of run 29107 evt 983 (31 labeled flash↔cluster matches; viewer
`ql_scan/`) drove a first label-driven retune of the common optical model, fit from the
labels' per-channel `op_pes`/`pred_pes` on the **sizable (`total_PE>3000`) + low-ks
(`ks_dis<0.2`)** subset (n=12 matches). Script: `ql_light_calib/fit_labels.py`.

```jsonnet
// FIRST (evt-983) values — see the Superseded note above for what ships now.
local vuv_eff = 0.01254,                          // was 0.0145
measured_pe_scale: std.makeArray(160, function(i) if i >= 120 then 1.57 else 1.0),
pe_err_frac: 0.44,                                // was C++ default 0.3
```

- **`measured_pe_scale`** — a NEW per-channel multiplicative gain correction on the
  **measured** flash PE (C++ knob, length `nchan`; empty/unset = identity =
  byte-identical for SBND/ICARUS). Applied in `Opflash::init` **before** `PE_err` is
  synthesized, so `(PE, PE_err, total, fired)` all stay self-consistent with the
  corrected measurement — and it flows to **everything** the matcher emits (χ² fit,
  calib dump, *and* the persisted Bee opflash PC), because a genuinely gain-biased
  measurement is wrong everywhere, not just in the fit. The **−x full-data-stream half
  ("APA0", optical ch 120-159)** under-reports PE; it is scaled **×1.57** to match the
  recalibrated prediction. (1.57 = g·median(pred_old/meas|APA0) = 0.865·1.814, where g
  is the vuv_eff factor below. The brighter ~2.3 tail the scan first showed is high-ks
  **saturation**, excluded by the low-ks cut.)
- **`vuv_eff` 0.0145 → 0.01254** — the **+x side-1 anchor** (the well-calibrated side)
  over-predicted by ~16% (meas/pred 0.865), so the common light yield drops by that
  factor: `0.0145 × 0.865`. `QtoL` stays 1 (degenerate with `vuv_eff`).
- **`pe_err_frac` 0.3 → 0.44** — intrinsic per-PMT scatter, method-of-moments fit
  `E[(pred−meas)²]=meas+a·pred²` (a=frac²) on the **corrected** model over the
  calibrated channels (side 1 + APA0). APA2's systematic over-prediction is **excluded**
  from this fit (it would wrongly inflate the global error).

**Caveats.** (1) One event, n=12 matches / a few points per block — these are *seeds*,
to be cross-checked against the run-wide auto-anchor fit (`fit.py`, n=37) which gave the
higher `vuv_eff` 0.0145; the label value is higher-purity but lower-statistics.
(2) **APA2 (ch80-119)** shows the **same ~1.7 elevation** as APA0 but, per the
APA0-only scope, gets no measured scale — so the common model leaves it ~1.4× over-
predicted, a known residual. (3) The brightness-dependence (2.3 on bright APA0/APA2) is
the **saturation/nonlinearity** signature; a per-channel `pmt_nonlinearity` round is the
proper next step, deferred here. (4) `frac=0.44` is a non-robust moment estimate on one
event; refine with more hand-scanned events.

### 3d. High-consistency ladder — run-29107 4-event tuning (enabled for PDHD)

`highconsist_ladder` is now ON. The ladder (`TimingTPCBundle::examine_bundle`) sets
`flag_high_consistent`, which gates the pre-LASSO `cull_inconsistent` purity cull: a
cluster with a consistent bundle (ladder-flagged or xtpc) drops its non-consistent rivals
before the fit (an xtpc scenario-1 bundle takes absolute priority). It is **KS-led** — a
true match has low KS, a wrong one high KS; chi2/ndf only fences the tail.

> **Two PDHD-specific calibrations vs the SBND seeds** (`ql_light_calib/validate_chain.py`,
> evts 983/991/999/1007):
> 1. **KS ceilings = SBND** (`hc_clean_ks` 0.06, `hc_good_ks` 0.09→**0.10**, `hc_tb_ks`
>    0.10, `hc_miss_ks` 0.08). KS is the purity lever: on the 4-event scan **ks<0.10 is 88%
>    pure** (57 GT vs 8 junk), and the few low-KS junk sit at chi2/ndf ≥ 40.
> 2. **chi2/ndf ceilings RAISED to PDHD scale** (`hc_clean_c2`/`hc_good_c2`/`hc_tb_c2`
>    6/4/8 → **35/35/35**; `hc_miss_c2` 60 kept). PDHD's semi-analytical light model is
>    rougher than SBND's, so a *clean-KS* GT match runs chi2/ndf up to ~32 (vs SBND ~1–2).
>    At the SBND ceilings six clean-KS GT matches (e.g. evt991 clu11 ks=0.033 c2n=6.4,
>    evt1007 clu87 ks=0.051 c2n=29) fell non-consistent and were culled when their cluster
>    had another ladder-passing bundle. c2n<35 keeps every clean-KS GT (max ~32) and still
>    fences the low-KS junk (c2n≥40). High-KS boundary matches (missing charge, ks 0.15–0.40)
>    can't be separated from junk by KS, so the ladder leaves them to the LASSO + boundary
>    down-weight (cull case-3 keep-all) rather than flagging them consistent.

**Result (4 events, vs ladder-off frac=0.4 baseline).** Primary FOM is the
**hand-scan-independent GT delta** (the global winner-KS shift is *not* independent — the
ladder mechanically culls high-KS bundles, so reporting "more low-KS winners" would be
circular):

| metric (hand scan) | baseline | ladder on |
|---|---:|---:|
| GT-accept winners (xTPC + non-xTPC) | 53 | **58** (+5) |
| of which xTPC (HARD — must not regress) | 12 | **14** (0 baseline xTPC lost) |
| human-rejected auto-matches re-selected | 219 | **171** (−48) |

**Honest cost — 4 displaced clean GT matches** (gains ~9 GT recovered, nets +5):
- evt1007 clu116 (ks=0.078) → clu120 (ks=0.071): lost to a **better**-KS off-scan cluster —
  an improvement, not a regression.
- evt999 clu130 (ks=0.108): missed `hc_good_ks`=0.10 by 0.008, culled, reassigned to a wrong
  flash. A KS-gate-boundary casualty; **not** worth moving the ceiling to 0.11 (that fits one
  event and admits the [0.10,0.11) junk band).
- evt999 clu171 (ks=0.252): a marginal GT pick (the hand scan is non-exhaustive; some small
  picks are sub-optimal).
- evt991 clu4 (ks=0.026) → **unmatched** (clu56 ks=0.099 won its flash): the one genuine
  better→worse displacement. clu4/clu56 are distinct ~6644/2401-pt tracks 92 cm apart (not a
  clustering split), both ladder-consistent, so this is a **LASSO selection** between two
  consistent clusters — no ladder ceiling touches it. Documented as a known displacement.

### 3e. Empty-flash light rescue — run-29107 4-event tuning (enabled for PDHD)

`empty_rescue` is now ON. After the LASSO, a flash left empty (no winner above the
strength cutoff) adopts its best light-quality candidate from the pre-fit snapshot if
`metric = ks·(chi2/ndf)^rescue_exponent` (× `rescue_boundary_weight` per `at_x_boundary`
then `close_to_PMT`) `< rescue_metric_max`; one-flash-per-cluster is enforced by
reassignment only when the empty flash is a strictly better light match.

> **PDHD differs from SBND.** SBND found its hand-scan misses **timing/drift-degenerate**
> (the correct cluster fit a wrong flash as well as the right one), so only ~1/5 were
> light-recoverable and it shipped a conservative `rescue_metric_max` 0.5. PDHD instead
> strands **clean** GT matches on LASSO-emptied flashes: on the 4-event scan **16 empty
> flashes have a GT-accept best candidate at metric < 0.16**, with a clear gap before the
> `ks=1.0` cross-side no-flash crossers (metric > 0.8 — those are xTPC-annotation cases that
> must NOT be light-rescued). The lowest non-GT candidate is at 0.057 (off-scan, neutral) and
> **no human-rejected match** appears at low metric. **`rescue_metric_max` = 0.20** captures
> the clean recoveries, excludes the crossers, and admits only neutral off-scan adoptions.

**Result (4 events, vs the ladder+chi2 state).** **+14 GT-accept (44→58)**, **0 reassignment
steals**, **0 GT winners lost**, **0 xTPC lost**, reject-reselected −6 (171→165); total
auto-selected winners 385→395. Not pushed higher than 0.20 (the 3 remaining marginal GT
recoveries sit at metric 0.26–0.38, into off-scan/steal-risk territory — diminishing returns).

---

## 4. One side with no light (point 3)

In most runs only the `+x` volume (group13) is instrumented; the `−x` side (group02)
reads ~0 PE (run 27980 is the exception — both sides lit). This is handled cleanly:

- **Independent per-group matching.** Each group is a separate `ApaRun`
  (`QLMatching.cxx:486`), so a dark side **cannot pollute** the live side's fit.
- **Same-side visibility.** The photon model returns 0 visibility across the cathode
  (a `−x` cluster predicts 0 PE on `+x` OpDets and vice-versa), and the per-side OpDet
  mask zeroes the other side's channels. A dark-side cluster gets pred≈0, meas≈0.
- **No flash time cut.** `flash_mintime / flash_maxtime` are set to ±1 s
  (`QLMatching.cxx:726`), wider than any PDHD readout, so **every** flash in the
  event reaches matching — the readout-clipping C++ default (±1.5 ms) and even a
  full-readout window are both bypassed. Bit-identical on the run-27305 sample (0
  of 707 flashes ever fell outside the full-readout window); the change only
  guards against a flash landing outside it in some future event/run.
- **Dark flashes dropped.** `flash_minPE = 50` discards near-zero flashes, so dark-side
  clusters simply get **no flash / no T0** (rather than a spurious match).
- **Dead-channel masking** (see [`pds-opchannel-opdet-mapping.md`](pds-opchannel-opdet-mapping.md)):
  - static **`ch_mask`** = `[3, 86, 87, 97, 107, 116, 117]` — noisy + LArSoft dead only.
    Channels **120..159** (the DAPHNE full-stream PDs = the entire −x / side-0 z<250 half)
    are **no longer statically masked**: the full-160 re-extraction (all-PD light reco)
    now decodes them with real light (run 29107: ~38/40 fire, ~70 k PE/evt), so masking
    them threw away half of side 0. Runs whose light reco lacks the full stream leave
    120..159 at 0 PE; those are caught per-event by `auto_mask`. **Effect on 29107:**
    side 0 went from 34 → ~72 live OpDets/evt (incl. the y=154 row, previously all dead).
  - **`auto_mask: true`** (`pe_low=10`, `pe_bright=50`, `neighbors=4`, `min_contrast=1`,
    `min_flash=3`) — per-event drop of a channel that never fires while its live
    neighbours do, catching the **run-dependent** `x<0` dead channels absent from the
    static list, and the 2/40 genuinely-dead full-stream channels on 29107.
    *Caveat:* on a run with the whole 120..159 block dark, `auto_mask` has no live
    neighbour to contrast against, so re-mask of that block is not guaranteed — verify
    per run.

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
- **`drift_speed = 1.576 mm/µs`** (`params.lar.drift_speed`, `params.jsonnet:122`), the
  data-calibrated value — see [`clustering-algorithm.md`](clustering-algorithm.md).
- **MC vs data:** `data: true` (auto-set `false` when `params.reality == 'sim'`) gates
  the MC-only saturation channel mask.
- **Now ON for PDHD (see §2):** `require_containment`, `cross_side_filter`,
  `reject_overpred` (3.0/10.0), the cross-cathode `xtpc_flag` matching (consumed by the
  `cull_inconsistent` xtpc-priority cull, not observation-only), and the flag
  fit-advantage levers `lasso_flag_weight` + `chi2_relax`. The cathode 3-D fiducial is
  still empty (flat-cathode window).
- **Hand-scan coincidence view.** The `ql_scan` viewer re-pairs the two same-time
  one-sided flashes of a cathode-crosser into one cross-side coincidence group (PDHD's
  opaque cathode means no flash lights both volumes); see `pdhd/ql_scan/README.md`.
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

**Toggle (PDHD default ON; C++ default OFF → bit-identical).** `save_opflash` is
plumbed `run_clus_evt.sh` → `wct-clustering.jsonnet` (TLA `save_opflash`) →
`clus.jsonnet` (`all_tpc(..., save_opflash)`) → the `clus_all_tpc` MABC node. It
only does anything with `do_qlmatch` on (it reads QLMatching's `opflash` root PC),
so the dump implies `-q`; with `do_qlmatch` off it is a no-op and the clustering
output is unchanged. **`run_clus_evt.sh` now dumps `op` by default** (so the Bee
combined link carries the flash overlay without extra flags); pass `-noop` (or
`PDHD_OPDUMP=0`) to suppress it. The `wct-clustering.jsonnet` TLA itself still
defaults `save_opflash=false`, so every non-PDHD config stays bit-identical.

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
`data/0/0-*.json` out of `mabc-all-apa.zip`, so once the events are clustered (op is
dumped by default) the `0-op.json` is folded into the upload automatically — no
builder change needed. The builder takes **every** layer (img-global, clustering-global,
op, dead area) straight from the mabc zip — there is no separate `wirecell-img bee-blobs`
pass. Select the **op** instance in Bee to step through flashes and compare `op_pes` vs
`op_pes_pred`.

**Bee charge layers — `img-global` + `clustering-global`.** The all-TPC MABC's
`bee_points_sets` (in `clus.jsonnet`) emit two global charge layers: `img-global` (the
raw imaged charge, dumped from the live grouping *before* the all-TPC clustering
pipeline — the C++ `"img"`-named pre-pipeline hook, now `algorithm: "img"`,
`individual:false`, no `apa_groups`, matching SBND) and `clustering-global` (the
full-detector clustered result, post-pipeline). The pre-pipeline hook can host only one
set, so emitting `img-global` here **replaces** the former per-drift-side stage-3
`clustering-group02/13` dump. A combined bee thus carries `img-global` + `clustering-global`
+ `op` + `channel-deadarea-group02/13`.

---

## Run it

```bash
# from pdhd/ (this repo)
./run_clus_evt.sh -q  <run> <evt>     # -q / PDHD_QLMATCH=1 enables Q/L matching
./run_clus_evt.sh -calib <run> <evt>  # + dump candidate bundles for ql_scan viewer
./run_clus_evt.sh -noop <run> <evt>   # suppress the optical "op" Bee instance (on by default)
```

Output `mabc-all-apa.zip` carries the matched flash/T0 per cluster and the `op`
instance (dumped by default; suppress with `-noop`). The [`ql_scan`](ql-scan-display.md)
viewer merges an event's group02 + group13 dumps into one two-side view; for the Bee
event display just run `run_bee_combined_evt.sh` (§6) on the clustered events.

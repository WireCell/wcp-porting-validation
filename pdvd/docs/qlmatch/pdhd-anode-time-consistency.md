# PDHD anode-position & time-chain consistency examination

**Date**: 2026-07-12.  **Toolkit**: branch `apply-pointcloud` @ `e587f357`.
**Data**: run 29107, 30 events (`pdhd/work/029107_{0..29}/calib-evt*.json`,
the all-PD light + joint-QLMatching production of 2026-06/07), plus the
4-event hand-scan ground truth
(`pdhd/work/ql_labels/labels-evt{983,991,999,1007}.json`).

PDHD counterpart of `pdvd-anode-time-consistency.md` (same directory), asking
the same two questions on the detector whose Q/L matching **is** tuned and
hand-scan-validated:

1. Is the **anode plane position** used by the imaging reconstruction
   (drift-time → x) consistent with the anode used by the fiducial-volume (FV)
   checks in clustering and QLMatching?
2. Are the **charge T0 (tick 0), flash time, and trigger offset** applied
   consistently wherever points are drift-corrected?

The purpose is comparative: PDHD serves as the *reference* case for the
anode/FV/time framework that is still being tuned on PDVD, where an apparent
mismatch between the reconstructed anode position and the FV edge is under
investigation.  No code is changed by this examination.

## 0. Repro

```bash
cd pdvd/docs/qlmatch
python3 check_anode_time_consistency_pdhd.py     # all checks (A, A', B', C, D, E)
```

Inputs: `pdhd/work/029107_{0..29}/calib-evt*.json` (the combined joint-node
dumps; the `-group{02,13}` per-side dumps are skipped), the opflash archive
metadata in the same dirs (per-event `offset_us`), and the hand-scan labels
above.  GDML numbers extracted from
`dunecore/dunecore/Geometry/gdml/protodunehd_v6_refactored.gdml` (the file the
toolkit `params.jsonnet` header cites).

## 1. Executive summary

**Q1 (anode position): CONSISTENT — and, unlike PDVD, the FV anode edge is a
physical surface.**  Imaging anchors x at the **W collection wire plane**
(−353.202 / +353.002 cm per side, from the wires file = GDML exactly); the FV
checks put the anode edge at the **first-induction (U) wire plane**
(±352.0945 cm, `apa_plane = 0.5·apa_g2g − plane_gap`, "pick it to be at the
first induction wires").  These are ~1 cm apart and connected by the same SP
arrival-time convention as in PDVD (§2.1).  The decisive difference from
PDVD: GDML-verified (§2.2), the PDHD U plane is **real** and the active LAr
boundary sits 0.075 mm above it — the FV edge *is* the physical active-volume
edge to sub-mm.  Empirically the track-end pile-up lands at
**u = −0.35 cm on both sides** (§5.2): after T0 correction an anode-crossing
track starts at the U plane / active edge to ~3 mm.  **There is no PDVD-style
near-anode reconstruction gap.**

**Q2 (times): CONSISTENT, with none of PDVD's per-side structure.**  All four
APAs share **one DAQ clock**: a single per-event trigger offset
(`offset_us` ≈ 249.9 µs, event spread ±0.2 µs ≈ 0.03 cm — vs PDVD's per-crate
±15 µs floats and up to 29 µs inter-crate skew) is threaded identically into
QLMatching and the clustering T0 correction with provably equal signs (§3.3).
The absolute charge↔light↔trigger base closes on the beam flash to
**−0.69 µs (0.11 cm)** median (§5.5).  The calib/Bee dump folds the offset
into every displayed flash time and zeroes the dump-level `trigger_offset`
(§3.4) — with one clock this is exact for both volumes, so the PDVD "BDE-folded
top-volume" trap (its §3.4, fixed via `time1`/`op_t1`) structurally cannot
occur on PDHD; the new keys are absent by construction and PDHD output is
byte-identical (gated 2026-07-12).

**bee3 display** (§7): consistent — boxes span the physical cathode surface
to the *per-side as-built* collection planes (matching the wires file to the
0.1 mm), drift speed/signs match the toolkit, and the base-class single
`op_t` is the correct per-TPC time for a one-clock detector.

**Quantified residuals** (§5): unbiased anode-edge pile-up **−0.35 cm both
sides**; cathode-edge pile-up **−0.46 cm (−x) / +0.52 cm (+x)** — an
antisymmetric-in-u = *common-mode ≈ +0.5 cm x* shift; 41 crosser-pair
midpoints at **+0.69 cm (MAD 0.63)** confirm the same small global x offset
and sub-cm side-to-side coherence.  Everything is at or below the documented
±2 cm degenerate t0/velocity/SCE band (`project_cathode_crossing_offset`,
the `drift_speed = 1.576` tuning).  Contrast PDVD: 3–9 cm anode-edge
structure and a 5–7 cm missing-charge gap.

## 2. The anode planes — position map

PDHD is a horizontal-drift detector: cathode at x ≈ 0, APA centerlines at
x ≈ ±357.3 cm, two drift volumes (side 0 = −x, read by APA0+2 face 0; side
1 = +x, read by APA1+3 face 1).  Each APA is a *real* multi-plane wire
stack — grid (G), U, V, W(collection) — so unlike the PDVD CRP every plane
below has a physical counterpart.

| x (side 0 / side 1) | plane | who uses it | provenance |
|---|---|---|---|
| **−353.202 / +353.002 cm** | W collection wire plane (drift-facing face) | imaging x anchor: `BlobSampler::time2drift` — `x = plane_x(2) + dirx·(t + time_offset)·v` (`clus/src/BlobSampler.cxx:167-181`), `time_offset = 0` (`pdhd/clus.jsonnet:10-26`) | wires file `protodunehd-wires-larsoft-v1.json.bz2`; **= GDML exactly** (§2.2). Note the as-built ±0.1 cm left/right asymmetry (cathode center at LArSoft x = −0.1) |
| **±352.0945 cm** | U first-induction plane, `apa_plane = 0.5·apa_g2g − plane_gap = 52.455 mm` from the APA centerline | the FV anode edge everywhere: QLMatching `run.anode_x` (via `inner_bounds` → `sensitive()` = the xregions `anode`↔`cathode` box); the calib-dump `geometry.anode_x`; the u = 0 face of the containment / `two_boundary` walk (`QLMatching.cxx:1405-1406, 3680`) | `params.jsonnet:30-45` — deliberately "at the first induction wires" (the grid-plane choice `0.5·apa_g2g` is present but commented out at `:44`).  As-built U planes: −352.219 / +352.019 (wires file), i.e. the symmetric params value is 1.2 mm inside (−x) / 0.75 mm outside (+x) the real plane |
| **±343.046 cm** | response plane: `res_plane = 0.5·apa_w2w + 100 mm = 142.935 mm` from the centerline | where the field response starts; **not** an x anchor — the SP intrinsic shift adds it back (§2.3) | `params.jsonnet:51-52`; both FR files carry `origin = 100.0 mm` (§2.3) |

Reference planes on the same axis: cathode surface at
`cpa_plane = apa_cpa − 0.5·cpa_thick` → **x = ±0.15875 cm** relative to the
detector center (`params.jsonnet:31,56`; `cpa_thick = 3.175 mm` = 1/8″).
QLMatching drift coordinate `u = s·(x − anode_x)` (s = +1 side 0, −1 side 1):
the U plane is u = 0, the collection plane is **u = −1.108 (−x) /
−0.908 (+x)**, and the cathode surface is **u_cathode = +351.936 cm** (all
from the calib-dump `geometry` block, which the checks below consume).

### 2.1 Why collection-anchor imaging + U-plane FV is consistent

Identical mechanism to PDVD (its §2.1), smaller numbers:

1. `OmnibusSigProc` deconvolves against the field response whose t = 0 is
   the charge crossing the **response plane** (FR `origin` = 100 mm before
   the collection wires), then rotates the output **later** by
   `⌊(ctoffset + fr.origin/fr.speed)/period⌋` ticks
   (`sigproc/src/OmnibusSigProc.cxx:937,1321`) — so SP output time ≈ charge
   **arrival at the wire plane** (excess quantified in §2.3).
2. `BlobSampler::time2drift` anchors x at the **collection wires** (the
   per-face wires-file value, hence the per-side asymmetry) and extrapolates
   back along the drift at `v = 1.576 mm/µs` (`params.jsonnet:122`,
   data-calibrated; must match — and does — `pdhd/clus.jsonnet:37` and the
   QLMatching `drift_speed`).
3. A deposit at the U plane at the reference time drifts 9.52 mm (nominal)
   to the wires (~6 µs); the extrapolation places it back at u = 0.  The FV
   anode edge at the U plane and the imaging anchor at the collection plane
   describe the same surface through one consistent time convention.

`dirx` is derived, not hand-set: `dirx = (response_x > anode_x) ? +1 : −1`
(`gen/src/AnodePlane.cxx:186`) → +1 for side 0, −1 for side 1, matching the
xregions layout (`params.jsonnet:68-94`).

### 2.2 GDML verification — the U-plane FV edge is physical

From `protodunehd_v6_refactored.gdml` (the file `params.jsonnet:29` cites):

- Inside `volTPCInner` the three strip planes sit at local x
  **−176.2265 (Z), −175.7353 (V), −175.2440 (U)** — real planes **4.9125 mm
  apart** (the params `plane_gap = 4.76 mm` is 0.15 mm/gap off; net 0.3 mm
  on the U-plane position).
- `volTPCActiveInner` (x-width 351.9468, centered at local +0.7369) spans
  local x [−175.2365, +176.7103]: the active LAr starts **0.075 mm above the
  U plane** and runs to the far box edge.  **The FV anode edge = the GDML
  active-volume boundary**, to sub-mm.  (Contrast PDVD, where the active
  volume reaches to 0.5 mm below the strip planes and the FV "grid plane"
  5.7 cm below them has no physical counterpart.)
- Global placement: `volTPCInner` at x = −212.9754 / +140.7754 (the +x copy
  Y-rotated), cathode volume centered at GDML x = −36.1 → in LArSoft world
  coordinates (cathode center at **x = −0.1**, not 0) the collection planes
  land at **−353.202 / +353.002** — the wires-file values *exactly*, and the
  bee3 box edges (§7).
- Cathode: GDML panel thickness 0.3302 cm vs params 0.3175 cm; as-built
  surfaces −0.265/+0.065 vs the toolkit's symmetric ±0.159 — mm-level.

So the toolkit's symmetric `apa_cpa = 357.34` idealizes an as-built geometry
that is offset by −0.1 cm and slightly wider (centerlines −357.50/+357.30);
every FV/anchor discrepancy this produces is ≤ 2 mm.  Worth knowing, nothing
to fix at the current ±2 cm drift-residual level.

### 2.3 Predicted reconstruction bias from the SP time-shift bookkeeping

Production configuration numbers:

- FR files `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` (APA0) and
  `dune-garfield-1d565.json.bz2` (APA1-3): both `origin = 100.0 mm`,
  `speed = 1.565 mm/µs` → `intrinsic = 100/1.565 = 63.898 µs` (identical
  shift on all four APAs).
- `ctoffset = 1.0 µs`, `ftoffset = 0` (`pdhd/sp.jsonnet:106-107`).
- Applied shift: `⌊(1.0 + 63.898)/0.5⌋ = 129 ticks = 64.5 µs` (the 512→500 ns
  Resampler precedes SP on all APAs; integer-tick floor loses 0.40 µs).
- Shift required for an exact collection anchor at `v = 1.576 mm/µs`:
  `100 mm / 1.576 = 63.452 µs`.

Excess = **+1.05 µs → +0.165 cm** reconstructed deeper into the volume
(cathode-ward, antisymmetric in x).  An order of magnitude smaller than
PDVD's +1.03 cm — the FR speed (1.565) is only 0.7% off the calibrated 1.576
(PDVD: 2.4%), and `ctoffset` is 1 µs, not 4.  At the current residual level
this is invisible.

Decomposed the same way as PDVD (velocity-mismatch piece ≈0.70 mm, `ctoffset`
piece ≈1.58 mm, tick-floor loss ≈0.63 mm) with the full side-by-side
comparison table and the simulation-cancellation argument for each piece:
`pdvd-anode-time-consistency.md` §2.3.1.

### 2.4 So where does an anode-crossing track start after T0 correction?

| layer | u (FV-anode frame) | |x| side 0 / side 1 (cm) |
|---|---|---|
| **physical** (GDML active edge = U plane) | −0.12 / +0.08 | 352.21 / 352.02 |
| **bookkeeping** (collection anchor + SP excess §2.3) | ≈ +0.05 | ≈ 352.05 |
| **measured** (unbiased edge scans, §5.2) | **−0.35 both sides** | 352.44 / 352.44 |

All three layers agree within ~4 mm.  After T0 correction an anode-crossing
track starts at **|x| ≈ 352.1–352.4**, i.e. at the FV edge = active edge,
with the last ~1 cm up to the collection wires being outside the active LAr
(behind the U plane) rather than "missing".  This is the PDHD answer to the
PDVD question "where does a track at the anode start": at the FV anode edge,
*because that edge is the physical active boundary* — whereas on PDVD the
tracks also stop near the FV edge but ~5–7 cm of genuine active drift before
the CRP face carries no reconstructed points (missing charge inside the
18.1 cm FR domain).  PDHD's FR domain is 10 cm and lies almost entirely
*outside* the active volume (response plane at u = −9.1 cm, active LAr ends
at u ≈ −0.1), so the failure mode that produces the PDVD gap — charge
deposited inside the response region, deconvolved with an invalid response —
has essentially no active volume to act on here.  That, not a better
algorithm, is why PDHD shows no gap; it is also why the same machinery can
be trusted on PDVD once the convention difference is accounted for.

## 3. Time relations — the clock chain

### 3.1 The chain, formula by formula

1. **Charge tick 0** = readout-window start, **one clock for all four APAs**
   (single DAQ; the BDE 512→500 ns resample is the first NF step and
   preserves the window start).  The window opens a nominal 250 µs before
   the trigger.  Raw imaging x is offset-free: `BlobSampler`
   `time_offset = 0` — PDHD has no per-event T0, so unmatched activity stays
   at its readout-time position (`pdhd/clus.jsonnet:10-26`).
2. **Trigger offset** = the measured trigger-vs-window-start distance:
   `offset_us = (tc − rd_timestamp)·16 ns` from the light ROOT `trigoff`
   tree, selecting the candidate nearest the nominal 250
   (`run_light_allpd_evt.sh:67-82`), stamped into the opflash archive
   metadata.  Run 29107: 249.73–250.19 µs over the 30 events (median 249.86,
   MAD 0.09 — a ±0.2 µs = ±0.03 cm quantity; the full per-event table is in
   the script output).  Semantics: **ADD to a flash time** to express it on
   the charge clock.  `run_clus_evt.sh:170-215` reads it from the archive
   and passes ONE scalar `trigger_offset_us` to both the matching and the
   clustering (`wct-clustering.jsonnet:68-71,117-118`).
3. **Flash time** in the opflash tensor = PE-weighted hit time relative to
   the **trigger**, ns, no offset applied (the offset rides along as
   metadata).  `correct_flash_time: false` (`pdhd/qlmatching.jsonnet:125`),
   so `Flash::get_time()` is the raw trigger-relative time.  (The light
   readout extends ~2.3 ms before the trigger, so raw flash times run to
   ~−2300 µs; flashes earlier than −250 µs precede the charge window and
   simply never match in-window charge.)
4. **QLMatching** drift-corrects per side:
   `flash_x_offset = sign_offset·(flash_time + trigger_offset_for(input_idx))·v`
   (`match/src/QLMatching.cxx:1293`), `sign_offset = −1` side 0 / `+1` side 1
   (`:1043`).  With no per-input `trigger_offsets` array configured,
   `trigger_offset_for` returns the same scalar for both inputs — the
   per-side machinery exists (PDVD uses it) but collapses to one number
   here.  Containment: `u ∈ [anode_ext1, u_cathode + cathode_ext1]`
   (`:1405-1406`).
5. **Clustering T0 correction**: matched clusters get
   `cluster_t0 = flash->get_time()` (raw, `apply_matched_t0s`), then
   `x_t0cor = x_raw − face_dirx·(cluster_t0 + trigger_offset)·v`
   (`clus/src/PCTransforms.cxx:146-148`), the same scalar delivered through
   the DV metadata (`pdhd/clus.jsonnet:81`).  Unmatched clusters carry
   `t0 = −1e12` and are dropped from T0-corrected outputs.
6. **cathode_connect** (all-TPC stage) additionally gates cross-cathode
   merges on flash-time coincidence: `use_flash_t0 = true`,
   `flash_t0_window = 1 µs` (`pdhd/clus.jsonnet:616-620`) — a *consumer* of
   the chain above, meaningful only because both sides are on one clock.

### 3.2 What each x means

- **raw x** (imaging, calib-dump cluster arrays): position *if the ionization
  occurred at charge tick 0*.  Unlike PDVD, the two sides share the clock,
  so raw x IS comparable across the cathode — but it is still not physical
  until a T0 is chosen.
- **T0-corrected x** (`x_t0cor` / QLMatching's `x + flash_x_offset`):
  physical position under the matched-flash hypothesis; what makes the 41
  crosser halves meet at the cathode (§5.4).

### 3.3 Sign-consistency proof (QLMatching vs clustering)

| volume | `sign_offset` (QL `:1043`) | `face_dirx` (`AnodePlane.cxx:186`) | QL shift `+sign_offset·(t+off)·v` | clus shift `−face_dirx·(t+off)·v` |
|---|---|---|---|---|
| side 0 (−x, APA0+2 f0) | −1 | +1 | −(t+off)·v | −(t+off)·v ✓ |
| side 1 (+x, APA1+3 f1) | +1 | −1 | +(t+off)·v | +(t+off)·v ✓ |

Same cancellation as PDVD; both stages draw `t`, `off`, `v` from the same
sources.  The `shared_flash` guard `sign_offset == −s` (`QLMatching.cxx:1131`)
holds here too (s = +1/−1).

### 3.4 Dump convention — offset folded, and why the PDVD trap cannot occur

The calib dump writes `f["time"] = flash_time + trigger_offset_for(input)`
(`QLMatching.cxx:2755`) and sets the top-level `trigger_offset` to **0**
("already in f[time]", `:2642`); the Bee `op_t` is built the same way.  On
PDVD the analogous folding used the *bottom* crate's offset for both volumes
and skewed top-volume drift arithmetic by up to 4.6 cm (its §3.4, fixed via
per-side `time1`/`op_t1`, toolkit `e587f357` + bee3 `40c3629`).  On PDHD the
two inputs share one offset, so the folded time is exact for both volumes:
the fix's gate `m_shared_flash && !m_trigger_offsets.empty()` is false, no
`time1`/`op_t1` keys are emitted, and the outputs are byte-identical to
pre-fix production (gate: full QL rerun of 29107 evt 0, all 15 `mabc-*.zip`
member-identical, `work/029107_0_qlt`).  Scripts consuming PDHD dumps can
use `f["time"]` directly — `check_anode_time_consistency_pdhd.py` asserts
`trigger_offset == 0` and does no re-basing.

## 4. Fiducial-volume definitions side by side

| | QLMatching FV (per drift side) | clustering `dvm` (per drift side) |
|---|---|---|
| source | `inner_bounds()` = `sensitive()` box unioned over the side's two APAs (`grouping_anodes`) | hand-written numbers in `pdhd/clus.jsonnet:59-104` |
| anode x | ±352.0945 cm (xregions `anode` = U plane = active edge) | **±357.985 cm** — ≈ 4.8 cm *beyond* the drift-facing collection planes (`clus/docs/clustering-separate-fv.md:19`): at the anode the clustering x-FV is deliberately inert (no reconstructable point reaches it; raw x tops out at the collection plane + the §2.3 excess) |
| cathode x | ±0.15875 cm (xregions `cathode` = CPA surface) | **±2.54 cm** — a 1-inch inset, 2.38 cm short of the surface |
| y, z | wires-file extents (y 7.61–606.0, z 0.23–462.30, dump `geometry`) | active ± 15 cm insets (y 22.61–591.0, z 15.23–447.30) |
| cushions / margins | `anode_ext1 = −2`, `anode_ext2 = +4` (C++ defaults — the −2 cm absorbs ends reconstructing between the U plane and the collection wires, exactly the §5.2 −0.35 cm population), `cathode_ext1 = +1.5`, `cathode_ext2 = −3.0` (PDHD-tuned, `qlmatching.jsonnet:285-286`), `two_boundary_margin` default | `FV_x_margin = 2`, `FV_y_margin = 2.5`, `FV_z_margin = 3` (cm) |
| consumers | containment gate, PE-inclusion box, `at_x_boundary`/`two_boundary`/xTPC flags | `select_scope_fv`, `examine_x_boundary`, `FiducialUtils` |

Unlike PDVD (where QL and clus agree on the same FV x numbers), the PDHD
clustering `dvm` x-window is *not* the QL FV: it is wider on both ends
(anode edge past the wires, cathode edge 1 inch out).  This is a
consistency-by-inertness rather than by-equality: the clustering x-cuts
cannot clip real anode activity, and the cathode inset only affects
FV-selection algorithms (separate/neutrino), not the matching.  The two
detectors' *matching* FVs are built the same way (xregions via
`sensitive()`); the difference is in the hand-written clustering block.

## 5. Empirical results (run 29107)

All numbers from `check_anode_time_consistency_pdhd.py` (Repro §0), drift
coordinate u in cm (0 = FV anode edge = U plane, +351.936 = cathode
surface).  No time re-basing anywhere (single clock, §3.4).

### 5.1 Samples and their biases

- **Hand-scan GT (check A)**: the accepted matches of the 4 labeled events
  (evts 983/991/999/1007).  Selected on *match quality*, not on boundary
  distance — so, unlike the PDVD boundary-track check A, not circular in u —
  but small (13/8 anode ends, 7/12 cathode ends per side).
- **Unbiased edge scans (A′/B′)**: PCA ends of ALL auto-selected bundles
  (span ≥ 30 cm, dedup per (cluster, flash)) within ±12 cm of a face, 30
  events.  Wrong-flash autos smear into a background; the edge is the peak.
  This is the measurement to trust.
- **Crosser pairs (D)**: dump flashes sharing a coincidence `group` with
  exactly one auto-matched cluster per side, both halves' cathode ends
  within 15 cm of the cathode (the proximity cut removes coincidence-grouped
  non-crossers, 143 → 41 pairs).
- PCA extremes slightly underestimate ends on curved tracks; `two_boundary`
  full candles are rare in this sample (n = 2, check C).

### 5.2 Anode edge (checks A, A′)

Unbiased pile-up of track-end u at the anode (1 cm bins):

| side | ends in ±12 cm | peak | core median (±3 cm of peak) |
|---|---|---|---|
| −x | 142 | **−0.5** | **−0.35** (n = 94) |
| +x | 136 | **−0.5** | **−0.35** (n = 94) |

Hand-scan GT medians agree: −0.22 (−x, n = 13) / −0.39 (+x, n = 8).

Reading: track ends reconstruct **at the FV anode edge, symmetrically, to
~3 mm** — between the as-built U plane (u = −0.12/+0.08) and slightly toward
the wires.  The −2 cm `anode_ext1` cushion covers the whole population.
Compare PDVD's +3.2 (bottom) / −0.1 (top) with a 5–7 cm missing-charge gap
to the physical edge: the PDHD result is what "consistent" looks like when
the FV edge is the physical active boundary and the FR domain sits outside
the active volume (§2.4).

### 5.3 Cathode edge (checks B′, A-GT) and the common-mode x offset

Unbiased pile-up of `u_cathode − u` (positive = short of the surface):

| side | ends in ±12 cm | peak | core median |
|---|---|---|---|
| −x | 258 | −0.5 | **−0.46** (n = 170) — 4.6 mm *past* the surface |
| +x | 237 | +0.5 | **+0.52** (n = 170) — 5.2 mm *short* of it |

Antisymmetric in u = **common-mode in x**: both sides' cathode ends sit at
x ≈ +0.3 to +0.7 cm.  A per-side drift/timing error would move the two sides
oppositely in x; this pattern is instead a small global x displacement of
the matched charge relative to the assumed cathode at 0 — direction-
consistent with the as-built cathode center at −0.1 (§2.2) plus the
irreducible ±2 cm t0/velocity/SCE band that the `drift_speed = 1.576` tuning
already navigates (the tuning deliberately puts the worst crosser *just
inside* the cathode; `clus.jsonnet:28-36`).  The §2.3 SP excess
(+0.165 cm, antisymmetric in x) has the right sign for the −x side and the
wrong sign for +x — it is not the explanation and is subdominant.
GT cathode ends (small n) scatter around the same values.

### 5.4 Crosser pairs — side-to-side coherence (check D)

41 pairs; cathode-end **true x** per half (surfaces at ∓0.159):

- −x half end x: median **+0.44** (MAD 1.06)
- +x half end x: median **+0.97** (MAD 0.81)
- pair midpoint: median **+0.69** (MAD 0.63)
- pair gap (+x − −x): median **+0.59** vs 0.32 = cathode thickness

The halves meet: the median gap exceeds the physical cathode by < 3 mm
(each half stops ~1 mm short of its surface), and the midpoint MAD of 0.63 cm
bounds any relative side-to-side time-base error at the ~0.4 µs level.  The
+0.69 cm midpoint restates the §5.3 common-mode shift.  Compare PDVD, where
the halves stop 2–3 cm short and pairs meet 10–22 cm apart
(`xtpc_dmax = 25 cm`); PDHD's `xtpc_dmax = 5 cm` (`qlmatching.jsonnet:545`)
is consistent with this much tighter registration.

### 5.5 Beam-flash closure (check E) — absolute time base

On a beam trigger the beam flash fires at the trigger, so its dumped time
must equal `offset_us`.  10 of 30 events carry a > 1000 PE flash within
±5 µs of the trigger; residual `t_dump − offset_us`: **median −0.69 µs
(MAD 0.05)** ≈ −0.11 cm of drift.  The clean six cluster within 0.09 µs of
each other; the three outliers (evts 1015/1023/1039, −3.0..+1.4 µs) are the
brightest/saturated events (evt 1015 is the documented DAPHNE-saturation
event).  This is the PDHD analog of PDVD's −0.9 µs BDE beam closure — and
here it validates the *only* clock, hence the whole detector.

### 5.6 Full candles (check C)

Only 2 `two_boundary` anode→cathode candles pass the gates; closure
(u-span vs u_cathode) = +0.77 / +2.35 cm.  Consistent with §5.2+§5.3 but too
few to sharpen anything; the 28-crosser calibration set used for the
`vuv_eff` tuning lives in the hand-scan labels, not re-derived here.

## 6. Verdicts, and what this means for PDVD

**Q1 — anode positions: CONSISTENT.**  Imaging (per-side as-built collection
anchors) and the matching FV (U plane = GDML active edge) describe the same
geometry through the same SP arrival-time convention; the measured track-end
edge sits on the FV edge at the 3–4 mm level, symmetrically (§5.2).  The
clustering `dvm` x-window intentionally over-covers (inert at the anode,
1-inch inset at the cathode) rather than duplicating the QL numbers (§4).

**Q2 — times: CONSISTENT.**  One DAQ clock, one measured trigger offset
(±0.2 µs across events), identical signed application in QLMatching and
T0Correction (§3.3), absolute base pinned by the beam flash to −0.7 µs
(§5.5), side-to-side coherence pinned by 41 crosser midpoints to MAD 0.63 cm
(§5.4).  The dump-time folding is exact for both volumes (§3.4).

**Residual findings** (all sub-cm, none blocking):

1. **Common-mode +0.5–0.7 cm x shift at the cathode** (§5.3/§5.4): matched
   charge meets ~0.7 cm to +x of the assumed x = 0 cathode.  Part is real
   geometry (as-built cathode center −0.1, which the symmetric params
   ignore); the rest sits inside the documented ±2 cm degenerate band.  If
   sub-5-mm registration is ever needed: put the as-built −0.1 cm into
   params and revisit `drift_speed`/`ctoffset` together (§2.3 predicts
   +0.165 cm of it).
2. **params geometry idealization** (§2.2): symmetric `apa_cpa = 357.34` vs
   as-built ∓357.50/+357.30; `plane_gap` 4.76 vs 4.9125 mm; cathode
   thickness/position mm-level off.  All effects ≤ 2 mm on FV edges; note
   kept for provenance.
3. **Beam-closure outliers on saturated events** (§5.5): the flash-time
   estimate moves by 1–3 µs when the PDs saturate (evt 1015 class) —
   matches the known saturation story, no time-chain implication.

**The comparative point (why this exam was run).**  Both detectors run the
*same* anchor/FV/time machinery, and on PDHD every link closes at the
millimeter-to-sub-centimeter level.  The PDVD anode "mismatch" therefore is
not a defect of the shared reconstruction chain; it decomposes into the two
PDVD-specific items its own exam isolated: (a) the FV anode edge there is a
DocDB-203 convention 5.7 cm inside the physical CRP face (PDHD's is the
physical active edge), and (b) PDVD loses the last ~5–7 cm of drift before
the CRP to induction-ROI failure inside its 18.1 cm FR domain (PDHD's 10 cm
FR domain lies outside the active LAr, so the mechanism has nothing to act
on).  Fixing/quantifying (b) — the simulation-closure and magnify follow-ups
in the PDVD doc §6 — is where the PDVD tuning effort should go; the time
chain itself needs no repair on either detector.

## 7. The bee3 event display — PDHD anode/cathode audit

`wire-cell-bee3/events/static/js/bee/physics/experiment.js`, class
`ProtoDUNEHD` (`:907-1054`):

- **TPC boxes** (`:919-928`): x ∈ [−353.202, −0.159] and [+0.159, +353.002] —
  physical cathode surface → *per-side as-built* collection planes, matching
  the wires file / GDML to 0.1 mm (the class comment records the earlier
  1-inch-inset mistake and its fix).  Same "physical box" convention as the
  PDVD audit (§7.1 there): box anode face = collection plane (353.0/353.2),
  toolkit FV edge = U plane (352.09) — an intentional ~1 cm difference, and
  after T0 correction tracks end ~0.35 cm inside the FV edge, i.e. **~0.6 cm
  short of the box face** — visually negligible, unlike PDVD's expected
  5–7 cm gap.
- **Drift arithmetic**: `driftVelocity = 0.1576` = toolkit 1.576 mm/µs ✓;
  base-class `driftDir(i) = ((i%2)−0.5)·−2` → +1 for boxes 0/2 (−x), −1 for
  1/3 (+x) = WCT `face_dirx` ✓; both display frames are the duals of
  `x_t0cor` as on PDVD.
- **Flash time**: the base-class `flashTimeForTPC` returns the single `op_t`
  for every TPC — correct here (one clock; the dumped `op_t` is
  trigger-folded for both sides, §3.4).  The PDVD `op_t1` override does not
  engage (no `op_t1` in PDHD dumps, by construction).
- **`detectorFrameCorrection`** (`:1004-1044`): PDHD-specific
  cluster→TPC-side association by T0-corrected containment voting (handles
  the opaque-cathode "light side ≠ charge side" crossers) — a display-side
  consumer of the same time chain, sign-consistent with §3.
- **Photon detectors**: 160 X-ARAPUCAs drawn on a representative 10×4 grid
  per 40-channel block at the box faces (`:955-984`) — display-only.

**Verdict**: consistent; no PDHD analog of the PDVD op-time defect exists.

## References

- `pdvd/docs/qlmatch/pdvd-anode-time-consistency.md` — the PDVD counterpart
  this doc parallels; mechanism write-ups referenced throughout.
- `pdhd/docs/clustering-algorithm.md` — drift-speed calibration history and
  the FV table; `clus/docs/clustering-separate-fv.md` — dvm FV provenance.
- `pdhd/docs/qlmatching-chain.md`, `match/docs/qlmatching-code.md` —
  QLMatching knob reference; `pdhd/docs/ql-light-normalization-study.md` —
  the hand-scan calibration this doc's GT labels come from.
- `dunecore/dunecore/Geometry/gdml/protodunehd_v6_refactored.gdml` — GDML
  source for §2.2.

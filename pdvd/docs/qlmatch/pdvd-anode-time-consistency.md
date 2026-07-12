# PDVD anode-position & time-chain consistency examination

**Date**: 2026-07-12.  **Toolkit**: branch `apply-pointcloud` @ `1bc0b025`.
**Data**: run 039252, 18 events (`pdvd/work/039252_{0..17}/calib-evt*.json`,
Xe/175nm + geometry-fix + xTPC reprocessing of 2026-07-11).
**Probes**: the "candle" tracks — 43 cathode-crosser pairs (17 render-validated
+ 26 finder candidates, tag `crossers`) and 108 anode boundary tracks (tag
`boundary`).

Two questions are examined:

1. Is the **anode plane position** used by the imaging reconstruction
   (drift-time → x) consistent with the anode used by the fiducial-volume (FV)
   checks in clustering and QLMatching?
2. Are the **charge T0 (tick 0), flash time, and trigger offsets** applied
   consistently wherever points are drift-corrected?

No code is changed by this examination.

## 0. Repro

```bash
cd pdvd/docs/qlmatch
python3 check_anode_time_consistency.py            # all checks, per-side-corrected times
python3 check_anode_time_consistency.py --raw-top  # reproduce the dump-time (BDE-folded) view
```

Inputs: `../../work/039252_*/calib-evt*.json` (calib dumps carry raw imaging x
per cluster, flash times, per-APA geometry, and the reference
`trigger_offsets_us`), plus the candle selections
`../../ql_display/decisions-{boundary,crossers}/decisions-evt*.jsonl`.

## 1. Executive summary

**Q1 (anode position): consistent by construction, with one convention to keep
in mind.** Imaging anchors x at the **W collection wire plane (±341.55 cm)**;
the FV checks (QLMatching and clustering `dvm`) put the anode edge at the
**"grid" plane (±335.835 cm)**. These are *different planes 5.715 cm apart*,
but not an inconsistency: signal-processing shifts the deconvolved waveform so
that output time ≈ charge arrival at the wires, which is exactly the convention
under which the collection-plane anchor is correct; a deposit at the grid plane
then reconstructs at the grid plane, i.e. at the FV edge (§2). The caveat: the
57.15 mm grid-plane setback is a **ProtoDUNE-SP wire-stack constant (DocDB
203)** with **no physical counterpart in PDVD** — GDML-verified in §2.2: the
CRP's U/V/Z planes are 0.2 mm apart and the active LAr reaches to 0.5 mm below
them (physical active edge |x| = 341.50), so the physical volume extends
~5.7 cm beyond the FV anode edge. After T0 correction an anode-crossing track
*actually* starts at **|x| ≈ 335.9 (top) / 332.6 (bottom)** — near the FV
convention only by coincidence; the ~5–8 cm shortfall vs the physical edge is
near-anode reconstruction loss (§2.4, §5.2).

**Q2 (times): consistently applied in the reconstruction; one asymmetric
convention in the outputs.** The per-crate trigger offsets (BDE/bottom vs
TDE/top charge windows float independently, Δ = 1.2–29.4 µs = 0.2–4.6 cm of
drift in these 18 events) are threaded per drift side into both QLMatching and
the clustering T0 correction, and the two stages apply the *identical* signed
drift shift (sign proof in §3.3). The bottom time base is independently
validated by the beam-flash closure (residual −0.9 µs). **The one genuine
asymmetry**: under `shared_flash` the calib/Bee dumps fold the *bottom* (input
0) offset into the single displayed flash time for both volumes — top-volume
drift arithmetic done from dump times is off by Δ·v (up to 4.6 cm here) unless
re-based; `find_boundary.py` inherited this (§3.4, §5.2). The matching itself
is unaffected (it uses per-side offsets internally).

**bee3 event display** (§7): geometrically consistent — its TPC boxes use the
*physical* convention (cathode ±3.0 → collection ±341.55), drift speed and
per-volume drift signs match the toolkit exactly (both display frames are
duals of `x_t0cor`) — with one inherited defect: it applies the BDE-folded op
time to the top volume too (up to 4.6 cm skew this run, §7.3).

**Quantified residuals** (details §5): after per-side time re-basing, anode-end
pile-ups sit at u ≈ **+3.2 cm (bottom)** / **−0.1 cm (top)** relative to the
grid-plane FV edge; cathode ends sit at **+0.4 cm (bottom)** / **+1.6 cm
(top)** short of the ±3.0 cm cathode surface; crosser-pair midpoints are
centered on the cathode (median −0.3 cm). A ~+1.0 cm cathode-ward
reconstruction bias is *predicted* from the SP time-shift bookkeeping (§2.3).
The ~3.3 cm bottom-vs-top difference at the anode edge is the main open item
(§6).

## 2. The three "anode planes" — position map

Three distinct x values all get called "the anode"; each has a different
consumer. Everything below is for the bottom volume (anodes 0–3); the top
(anodes 4–7) mirrors with the opposite sign.

| x (bottom) | plane | who uses it | provenance |
|---|---|---|---|
| **−341.55 cm** | W collection wire plane | imaging x anchor: `BlobSampler::time2drift` — `x = plane_x(2) + dirx·(t + time_offset)·v` with `plane_x(2)` = first collection-wire center x (`clus/src/BlobSampler.cxx:167-181`); also clus `dvm` *overall* FV box (±3415.5 mm, `clus.jsonnet:72`) | wires file `protodunevd-wires-larsoft-v5.json.bz2` (measured: U/V/W first-wire x = −3415.1/−3415.3/−3415.5 mm — the three planes are 0.2 mm apart, real PCB-CRP geometry) |
| **−335.835 cm** | "grid" plane, `apa_plane = 0.5·apa_g2g = 57.15 mm` | the FV anode edge everywhere: QLMatching `run.anode_x` (via `inner_bounds` → `IAnodeFace::sensitive()`, whose x span is exactly the xregions `anode`↔`cathode` interval, `gen/src/AnodePlane.cxx:282-317`, `aux/src/DetectorVolumes.cxx:187-192`); clus per-drift `dvm` `FV_xmin = −3358.35 mm` (`clus.jsonnet:96`); the calib-dump `geometry.anode_x`; the u=0 face of `compute_two_boundary_flag` | `params.jsonnet:38-40` — `apa_g2g = 114.3 mm` is copied from the ProtoDUNE-SP DocDB 203 wire stack (see `pdvd-tpc-geometry-fiducial.md` §3); **it does not describe the PDVD CRP**, whose planes are 0.2 mm apart (above) |
| **−319.164 cm** | response plane: `centerline + res_plane` with `res_plane = 0.5·apa_w2w + response_plane(181 mm) = 223.86 mm` from the collection centerline | where the field-response calculation starts; **not** an x anchor — see §2.3 | `params.jsonnet:42-48`, `response_plane` synced to `protodunevd_FR_imbalance3p_260501.json.bz2` (`origin = 181.0 mm`) |

Reference planes on the same axis: cathode surface at −3.0 cm
(`cpa_plane = apa_cpa − 0.5·cpa_thick = 341.55 − 3.0 = 338.55 cm`,
`params.jsonnet:35,52`); QLMatching drift coordinate
`u = s·(x − anode_x)` so the grid plane is u = 0, the collection plane is
**u = −5.715 cm**, and the cathode surface is **u_cathode = +332.835 cm**.

### 2.1 Why collection-anchor imaging + grid-plane FV is consistent

The two stages use different planes *for different jobs*, and the time
bookkeeping connects them:

1. `OmnibusSigProc` deconvolves by the field response, whose t = 0 is the
   charge crossing the **response plane** (FR `origin` = 181 mm before the
   wires). It then rotates the output **later** by
   `time_shift = (ctoffset + intrinsic)/period` where
   `intrinsic = fr.origin/fr.speed` (`sigproc/src/OmnibusSigProc.cxx:937,1321`).
   Net effect: SP output time ≈ charge **arrival at the wire plane** (plus the
   small excess quantified in §2.3).
2. `BlobSampler::time2drift` therefore anchors x at the **collection wires**
   and extrapolates back along the drift:
   `x = x_collection + dirx·(t + time_offset)·v` with `time_offset = 0` for
   PDVD data (no per-event T0; `clus.jsonnet:9-16`) and `v = 1.568 mm/µs`
   (`params.jsonnet:119`).
3. A deposit at the grid plane at the reference time drifts 57.15 mm to the
   wires (~36 µs); the extrapolation places it back at −335.835 cm. So the FV
   anode edge at the grid plane and the imaging anchor at the collection plane
   describe **the same physical surface seen through a consistent time
   convention**. There is no double-counting and no missing 5.7 cm.

The `dirx` signs are derived, not hand-set: `dirx = (response_x > anode_x) ?
+1 : −1` (`gen/src/AnodePlane.cxx:186`) → +1 bottom, −1 top, matching the
xregions layout (`params.jsonnet:54-97`).

### 2.2 The "grid plane" does not exist physically — GDML verification

Because the 5.7 cm collection↔"grid" separation looks suspicious for a PCB
CRP, it was checked directly against the LArSoft geometry
(`dunecore/dunecore/Geometry/gdml/protodunevd_v4_refactored_nowires.gdml`):

- Inside `volTPC0` (`CRM` box, x = 338.56 cm) the three strip planes sit at
  local x = **169.23 (U), 169.25 (V), 169.27 (Z/collection)** — 0.2 mm apart,
  0.2 mm thick. **There is no plane of any kind 5.7 cm below them.**
- `volTPCActive` (`CRMActive`, x = 338.5 cm) is centered at local x = −0.03,
  so the active LAr extends to local x = +169.22 — **0.5 mm below the U
  plane**. The active volume reaches the strip planes.
- Global placement: `volTPC0` at x = −192.28 (rotated 180° about Y) puts the
  collection plane at −361.55; the cathode volume is centered at GDML x = −20
  (mesh surfaces at ±2.975 about it) → in the cathode-centered reco frame the
  collection plane is at **−341.55** ✓ (= `apa_cpa`, = the wires-file W x)
  and the physical active edge at **−341.50**.

So: **341.55 cm is real** (GDML + wires file agree); **335.835 cm is a pure
toolkit convention** — `centerline ∓ apa_plane` with
`apa_plane = 0.5·apa_g2g = 57.15 mm`, where `apa_g2g = 114.3 mm` is the
ProtoDUNE-SP DocDB-203 *wire-stack* spacing copied into the PDVD `det` block
(the same provenance as the wrong `cpa_thick = 50.8 mm` fixed on 2026-07-08).
In PDSP an APA really has a grid plane ~57 mm before the collection plane; in
the PDVD CRP nothing is there. The number survives only as the xregions
`anode` value, i.e. as **the FV/sensitive-boundary convention**, sitting
5.7 cm inside the physical active edge. Consequences:

- Real anode-touching track ends can legitimately reconstruct at **u < 0**
  (between the FV edge and the collection plane). QLMatching absorbs 2 cm of
  this with `anode_ext1 = −2 cm` (containment window starts below u = 0,
  `QLMatching.cxx:1351-1352`), and the `two_boundary` nearest-face test uses
  *signed* distances so ends beyond the face still register as "at the anode".
- Empirically (§5.2) the pile-ups sit at u ≈ +3.2 / −0.1 cm, i.e. at or inside
  the FV edge, so with the current cushions **no real activity is clipped**.
  But any future tightening of `anode_ext1` toward 0, or FV logic that
  *rejects* u < 0 points, would start eating genuine anode tracks — keep the
  grid-plane provenance in mind.

### 2.3 Predicted reconstruction bias from the SP time-shift bookkeeping

Numbers for the production configuration:

- FR file `protodunevd_FR_imbalance3p_260501.json.bz2`: `origin = 181.0 mm`,
  `speed = 1.53 mm/µs` → `intrinsic = 181/1.53 = 118.30 µs`.
- `ctoffset = 4 µs`, `ftoffset = 0` (`sp.jsonnet:116-118`, same for both
  crates; the BDE/TDE electronics differ only in the per-side
  `elecresponse`, which the deconvolution divides out).
- Applied shift: `⌊(4 + 118.30)/0.5⌋ = 244 ticks = 122.0 µs` (integer-tick
  floor loses 0.30 µs).
- Shift required for the collection-plane anchor to be exact at
  `v = 1.568 mm/µs`: `181 mm / 1.568 = 115.43 µs`.

Excess = **+6.57 µs → +1.03 cm** reconstructed *deeper into the volume*
(toward the cathode; +u in both volumes, an antisymmetric x shift). Two
contributions: the FR drift speed (1.53) being 2.4% slower than the calibrated
reconstruction speed (1.568) → +4.5 mm; and the `ctoffset = 4 µs` choice →
+6.3 mm. This is a *common-mode* prediction for both volumes; the measured
anode-edge positions (§5.2) are of this order but not symmetric, so §6 keeps
it as an open calibration item rather than a confirmed explanation. (Note the
drift-velocity calibration from crosser spans is insensitive to any such
additive offset — offsets cancel in spans.)

### 2.4 So where does an anode-crossing track start after T0 correction?

Three different answers, and the distinction matters:

| layer | anode-end |x| (cm) | u (grid-plane frame) |
|---|---|---|
| **physical** (GDML active edge) | 341.50 | −5.66 |
| **bookkeeping** (SP shift + collection anchor + T0 correction, §2.1/§2.3: a deposit at the active edge reconstructs 1.0 cm deeper) | ≈ 340.5 | ≈ −4.6 |
| **measured** (unbiased candle edge scans, §5.2) | **≈ 335.9 (top)** / **≈ 332.6 (bottom)** | −0.1 / +3.2 |

So in the current reconstruction an anode-crossing track, after T0
correction, starts **near |x| ≈ 335.9 in the top volume and ≈ 332.6 in the
bottom volume** — i.e. close to the 335.835 FV convention, but **not because
335.835 is a physical plane**. The bookkeeping says the track *should*
reconstruct out to ≈ 340.5. The gap is **missing charge, not mis-timed
charge**, demonstrated point-by-point on the T0-pinned full-drift crosser
evt298609 gid172 (c102 bottom + c4000003 top):

- Its **cathode ends land at u = 332.97 / 332.78 vs u_cathode = 332.835** —
  the T0 is verified to ~1 mm, so no time-chain error can be invoked.
- At that same verified T0, the **anode-end point density is flat (~uniform
  per cm) up to u = +2.4 (bottom) / +1.9 (top) and then simply stops** — no
  straggler points below, no pile-up of displaced charge. Each half is
  missing the last **~6.5–7.0 cm** of drift before the physical edge at
  u = −4.6.
- Not a wall exit: both anode ends sit > 15 cm from every y/z boundary and
  the extrapolated CRP crossing stays inside (bottom end (y,z) ≈ (135, 173),
  top ≈ (−8, 17); the tracks are drift-steep, du/d(yz) ≈ 3, as crossers must
  be).

The missing region is the inner ~⅓ of the **18.1 cm field-response domain**
(FR `origin` = 181 mm). Charge deposited there did not drift in from the
response plane, so the deconvolution model is wrong for it: the U/V induction
responses in particular lose their leading lobe and phase, ROI finding fails
on the induction views, and the 3-view tiling then drops the charge entirely
(collection alone cannot form blobs). This mechanism predicts W-plane signal
with absent U/V ROIs at those ticks — directly checkable (below). The rough
coincidence of the measured edge with the DocDB-203 grid-plane convention is
just that — a coincidence. (The verified pair loses ~7 cm in *both* volumes,
suggesting the §5.2 bottom-vs-top pile-up difference is at least partly
sample scatter.)

**Decisive follow-ups** (not done here): (i) simulation closure — MC depos at
known x near the CRP through the same NF/SP/imaging chain; if the ~5–7 cm gap
reproduces, it is inherent to the FR/ROI model, and its size becomes a
calibratable constant; (ii) magnify inspection of the U/V/W traces at a
validated anode-crossing end (e.g. evt298609 c102, channels around
(y,z) ≈ (135, 173), last ~45 µs of the track) to confirm the
induction-ROI-failure mechanism.

Practical implications: (a) the FV edge at 335.835 happens to track the
*reconstructable* volume better than the physical volume, which is why the
boundary-track finder works at all with a ±3 cm margin; (b) any efficiency or
length computed against the *physical* active volume must budget ~5–8 cm of
invisible track at the anode; (c) if near-anode reconstruction ever improves
(e.g. a short-drift response model), track ends would move to u < 0 and the
FV/two_boundary margins would need revisiting.

## 3. Time relations — the full clock chain

### 3.1 The chain, formula by formula

1. **Charge tick 0** = per-crate readout-window start. Bottom (BDE) and top
   (TDE) crates frame independently: window starts float ±15 µs event-to-event
   and up to ~32 µs apart *between* crates (own 64-sample frame boundaries).
   The BDE 512 ns → 500 ns resample is the first SP step and preserves the
   window start. Raw imaging x is therefore **per-crate-clocked and
   offset-free**: `BlobSampler` `time_offset = 0` — PDVD has no per-event T0,
   so unmatched activity is left at its readout-time position
   (`clus.jsonnet:9-16`).
2. **Light chain t0** = `min over all records of [round(ts·62.5) − (64 if
   nsamp ≤ 1024 else 0)] ticks × 0.016 µs` (`run_light_evt.sh:86-95`,
   replicating `PDVDOpWaveformSource`; the −64-tick term handles snippet
   records).
3. **Per-crate trigger offsets**, measured at light-processing time and
   stamped through `OpFlashFinder.metadata_extra`
   (`flash.jsonnet:169-198`):
   `offset_bot = wrap(light_t0 − charge_bde)`,
   `offset_top = wrap(light_t0 − charge_tde)` (`run_light_evt.sh:100-101`;
   wrap = 40-bit DTS rollover). Both ≈ −2.5 ms for run 039252; semantics:
   **ADD to flash time** to express the flash on the charge clock.
   `run_clus_evt.sh:172-218` reads them from the opflash archive metadata
   (plus an optional per-run residual applied to *both* sides, normally 0) and
   passes `trigger_offsets = [bot, top]` to the matching and
   `trigger_offset(_top)` to clustering (`wct-clustering.jsonnet:83-88,133-135`).
4. **Flash time** in the opflash tensor = PE-weighted hit time in the light
   frame, ns, **no offset applied** (`flash/src/OpFlashFinder.cxx:157-168,667`;
   the offsets ride along only as metadata). PDVD applies no CAF-style
   correction (`correct_flash_time: false`), so `Flash::get_time()` is the raw
   light-frame time (`aux/src/FlashTensorToOpticalPCs.cxx:110`).
5. **QLMatching** drift-corrects cluster points per drift side:
   `flash_x_offset = sign_offset·(flash_time + trigger_offset_for(input_idx))·v`
   (`match/src/QLMatching.cxx:1239`), with `sign_offset = −1` bottom / `+1`
   top (`:988-989`) and `trigger_offset_for` = the per-input `[bot, top]`
   array entry (`QLMatching.h:185-189`). The same form is used by the xTPC
   pairing (`:2846,3143`), containment (`:1351-1352`), and the
   `two_boundary` face test (`:3680-3694`).
6. **Clustering T0 correction** (`x_t0cor`, the coordinates the Bee
   "clustering" dumps and `common_corr_coords` consumers see): matched
   clusters get `cluster_t0 = flash->get_time()` — the **raw** flash time, no
   offset (`QLMatching::apply_matched_t0s`, `:2450,2461`); the per-volume
   trigger offset is then added *alongside* it by `T0Correction`:
   `x_t0cor = x_raw − face_dirx·(cluster_t0 + trigger_offset(apa,face))·v`
   (`clus/src/PCTransforms.cxx:78-88,146-148`; offsets delivered per
   anode/face through the DV metadata, bottom `clus.jsonnet:94`, top
   `clus.jsonnet:113-114`). Unmatched clusters carry `t0 = −1e12` and are
   dropped from T0-corrected outputs, never mis-corrected.

### 3.2 What each x means

- **raw x** (imaging, calib-dump cluster arrays): position *if the ionization
  occurred at charge tick 0 of its own crate*. Not a physical position until a
  T0 is chosen. Bottom and top raw x are on **different clocks** (Δ up to
  ~4.6 cm here) — never compare raw x across the cathode.
- **T0-corrected x** (`x_t0cor`, or QLMatching's internal `x + flash_x_offset`):
  physical position under the matched flash hypothesis. Bottom and top are on
  the **same clock** after the per-side offsets — this is what makes
  cathode-crosser halves meet (§5.4).

### 3.3 Sign-consistency proof (QLMatching vs clustering)

The two stages express the identical correction with opposite-sign
conventions that cancel:

| volume | `sign_offset` (QL, `:988`) | `face_dirx` (clus, `DetectorVolumes.cxx:166-169`) | QL shift `+sign_offset·(t+off)·v` | clus shift `−face_dirx·(t+off)·v` |
|---|---|---|---|---|
| bottom (0–3) | −1 | +1 | −(t+off)·v | −(t+off)·v ✓ |
| top (4–7) | +1 | −1 | +(t+off)·v | +(t+off)·v ✓ |

Both draw `t`, `off`, and `v` from the same sources (flash PC, runner-provided
offsets, `params.lar.drift_speed`), so matched points land at the same
corrected x in both stages. Verified by code reading; the
`PCTransforms.cxx:78-88` comment block documents the convention explicitly
(PDVD/PDHD supply `trigger_offset` via DV metadata because BlobSampler bakes
nothing in; SBND instead bakes it into `cluster_t0`).

### 3.4 The one asymmetric convention: dumps fold the BDE offset only

Under `shared_flash` the calib dump's flash loop runs for input 0 only
(`QLMatching.cxx:2687`), so the single dumped time is
`f["time"] = flash_time + trigger_offset_for(input 0)` — **bottom/BDE-folded
for both volumes** (`:2693`; likewise the Bee opflash display time `:2538`;
the dump sets `top["trigger_offset"] = 0` and emits the per-side values as
reference `trigger_offsets_us`, `:2580,2584-2587`).

Any *top-volume* drift arithmetic done from dump times must therefore re-base:

```
t_top = f["time"] + (off_top − off_bot)
```

Per-event offsets for run 039252 (µs, from `trigger_offsets_us`; Δ·v in cm at
v = 0.1568 cm/µs):

| event | off_bot | off_top | Δ = top−bot | Δ·v (cm) |
|---|---|---|---|---|
| 298567 | −2517.33 | −2497.18 | +20.14 | 3.16 |
| 298581 | −2507.76 | −2478.66 | +29.10 | 4.56 |
| 298595 | −2506.85 | −2478.26 | +28.59 | 4.48 |
| 298609 | −2513.81 | −2512.61 | +1.20 | 0.19 |
| 298623 | −2514.93 | −2496.58 | +18.35 | 2.88 |
| 298637 | −2509.01 | −2484.00 | +25.01 | 3.92 |
| 298651 | −2515.34 | −2507.74 | +7.60 | 1.19 |
| 298665 | −2511.63 | −2504.03 | +7.60 | 1.19 |
| 298679 | −2512.99 | −2483.63 | +29.36 | 4.60 |
| 298693 | −2504.10 | −2494.45 | +9.65 | 1.51 |
| 298707 | −2515.46 | −2509.39 | +6.06 | 0.95 |
| 298721 | −2530.13 | −2507.17 | +22.96 | 3.60 |
| 298735 | −2541.46 | −2525.66 | +15.79 | 2.48 |
| 298749 | −2510.69 | −2482.35 | +28.34 | 4.44 |
| 298763 | −2514.27 | −2489.01 | +25.26 | 3.96 |
| 298777 | −2515.14 | −2492.69 | +22.45 | 3.52 |
| 298791 | −2506.02 | −2496.11 | +9.90 | 1.55 |
| 298805 | −2500.24 | −2486.75 | +13.49 | 2.11 |

Δ is always positive in this run (TDE window opens after BDE). The **matcher
itself is unaffected** — it uses `trigger_offset_for(input_idx)` internally —
this is purely an output/display convention. But it propagates to consumers:
the ql_scan viewer's top-panel drift positions, and `ql_display/find_boundary.py`,
which computed top-volume anode distances from dump times (impact quantified
in §5.2: its top-track u values shift by −Δ·v, median −1.9 cm, when re-based).

### 3.5 Independent validation of the absolute time base

The bottom (BDE) charge↔light base was validated externally by the beam-flash
closure: for CTB beam triggers the flash must appear at `tc_us − charge_bde_us`
on the folded axis; 99/120 events of the 039252/039253/039349 batch show the
~2000 PE flash there with **median residual −0.9 µs** (≈ −0.14 cm; 1 µs flash
binning) — see `pdvd/docs/pdvd-ql-pending.md` §1. There is no equivalent
external anchor for the TDE side; the crosser midpoints (§5.4) are the current
bottom↔top consistency check.

## 4. Fiducial-volume definitions side by side

Both FV consumers anchor the anode edge at the **grid plane** and the cathode
edge at the **±3.0 cm cathode surface** — mutually consistent, and consistent
with the calib-dump geometry block (`anode_x = ±335.83`, `cathode_x = ±3.0`,
`u_cathode = 332.83 cm`, written from the same `compute_geometry` values,
`QLMatching.cxx:2668-2681`).

| | QLMatching FV (per drift side) | clustering `dvm` (per drift side) |
|---|---|---|
| source | `inner_bounds()` = `sensitive()` box unioned over the side's anodes **+ face 1** (the 2026-07-11 `tpc_extra_faces` Y-truncation fix, toolkit 565ccd62) | hand-written numbers in `clus.jsonnet:89-122` |
| anode x | ±335.835 cm (xregions `anode`) | ±3358.35 mm ✓ same |
| cathode x | ±3.0 cm (xregions `cathode`) | ±30.0 mm ✓ same |
| y, z | wires-file bounding box ± ½ pitch (y ±336.4 cm post-fix) | active ±3364.0 mm with 15 cm y/z insets |
| cushions / margins | `anode_ext1 = −2`, `anode_ext2 = +4`, `cathode_ext1 = +1.2`, `cathode_ext2 = −12` (PDVD override; all else C++ defaults), `y/z_cushion = 0`, `two_boundary_margin = 3` (cm) | `FV_x_margin = 2`, `FV_y_margin = 2.5`, `FV_z_margin = 3` (cm) |
| consumers | containment gate, PE-inclusion box, boundary/xTPC flags | `select_scope_fv` → `clustering_separate` / `neutrino`, `FiducialUtils::inside_fiducial_volume` |

One internal quirk worth knowing: the `dvm` **overall** box uses
`FV_x = ±3415.5 mm` (collection plane) while the per-drift boxes use the grid
plane — the overall box deliberately spans the whole detector including the
5.7 cm convention gap. Not a bug, but the two x conventions coexist inside one
`dvm` block.

## 5. Empirical results (run 039252 candles)

All numbers from `check_anode_time_consistency.py` (Repro §0), drift
coordinate u in cm (0 = grid-plane FV anode edge, +332.835 = cathode surface),
**top-volume flash times re-based per §3.4** unless stated.

### 5.1 Samples and their biases (read this first)

- **108 boundary tracks** (tag `boundary`): gated by the C++ `two_boundary`
  flag *and* by `|anode_u| ≤ 3 cm` at the dump time, and the finder keeps per
  cluster the flash minimizing |anode_u|. The last rule makes the sample's
  anode-u distribution **circular** (it measures the selection, not the
  detector): bottom median u = 0.00 by construction. Use it only for
  *relative* statements (e.g. the re-basing shift).
- **43 crosser pairs** (tag `crossers`): 17 render-validated + 26 finder
  candidates (not yet hand-confirmed) — some contamination expected.
- **Unbiased edge scans (A′/B′)**: all auto-selected bundles with 3-D span
  ≥ 30 cm, PCA-end u within ±12 cm of a face, no boundary gates. Wrong-flash
  auto matches (~40% per the hand-scan agreement rate) smear into a broad
  background; the true edge appears as the peak. This is the measurement to
  trust for *where ends pile up*.
- PCA extreme points slightly underestimate true track ends on curved tracks;
  clusters merged across the cathode by `cathode_connect` contribute far ends
  20–37 cm "beyond" the cathode (they contain the partner half) — filtered
  where noted.

### 5.2 Anode edge (checks A, A′)

Unbiased pile-up of track-end u at the anode (peak / core median, 1 cm bins):

| volume | ends in ±12 cm | peak | core median (±3 cm of peak) |
|---|---|---|---|
| bottom | 57 | **+3.5** | **+3.16** (n = 32) |
| top | 88 | **−0.5** | **−0.07** (n = 46) |

Both pile-ups are at or inside the FV edge — the FV clips no real anode
activity (bottom sits 3 cm inside; top straddles the edge, covered by
`anode_ext1 = −2 cm` and the signed-distance boundary test). Relative to the
*physical* CRP face (u ≈ −5.7), the ends reconstruct 5.6 cm (top) / 8.9 cm
(bottom) cathode-ward — the predicted +1.0 cm SP bias (§2.3) has the right
sign for part of this; near-anode charge loss below imaging threshold and the
FR model's validity inside the response region plausibly contribute the rest.

The **bottom-vs-top difference of ~3.3 cm (≈ 21 µs)** is the notable feature.
It is *not* produced by the dump-time convention (the scan uses per-side
re-based times, which is also what the matcher itself uses). Arguments against
a genuine ~21 µs per-crate time-base error: the beam-flash closure pins the
BDE base to −0.9 µs (§3.5), and crosser midpoints (§5.4) show no comparable
bottom↔top disagreement at the cathode. Remaining candidates: BDE vs TDE
electronics-response model residuals in the deconvolution (a group-delay error
maps directly onto x), genuinely different near-CRP charge reconstruction
between crate types, or sample composition (57 vs 88 ends, different track
angles). Open item — §6.

Effect of the §3.4 re-basing on the 108 boundary tracks (relative statement,
biased sample): top-track anode u moves from median +0.39 (dump time, how they
were selected) to **−1.48** (correct per-side time), range down to −5.5 —
i.e. `find_boundary.py`'s top-volume distances were ~2 cm too "clean" on
median, up to 4.6 cm in high-Δ events. Bottom tracks are unaffected (dump time
*is* the correct bottom time). The 108-track *selection* survives (the C++
`two_boundary` gate used correct times), but the printed `anode_u` numbers for
top tracks should be read with §3.4 in mind.

### 5.3 Cathode edge (checks B, B′)

Unbiased pile-up of `u_cathode − u` (positive = ends short of the cathode
surface):

| volume | ends in ±12 cm | peak | core median |
|---|---|---|---|
| bottom | 269 | −0.5 | **+0.41** (n = 96) |
| top | 394 | +1.5 | **+1.56** (n = 178) |

The bottom cathode edge lands within ~4 mm of the configured ±3.0 cm surface —
the `cpa_thick = 6.0 cm` correction of 2026-07-08 is validated by data. Top
ends stop ~1.6 cm short; combined with §5.2 this means the top volume's
reconstructed tracks are ~1.5 cm short *at the cathode end* while bottom's are
~3.2 cm short *at the anode end* — a pattern more like per-crate span/edge
effects than any single time offset (which would shift both ends of a volume
together, not shrink spans).

The 13 full anode→cathode candles (check C) are too few and
selection-shaped to sharpen this (median closure +3.3 cm, but 5 of 13 are
cathode-connect merged clusters whose far end is the partner half; the clean
subset scatters ±3 cm).

### 5.4 Crosser pairs — bottom↔top clock consistency (check D)

For each pair, the cathode-end **true x** of each half (per-side corrected
times), surfaces at ∓3.0 cm:

- bottom-half end x: median **−6.11** (MAD 2.7) → 3.1 cm short of −3.0
- top-half end x: median **+4.75** (MAD 4.2) → 1.8 cm short of +3.0
- pair midpoint: median **−0.27** (MAD 2.3) — centered on the cathode
- pair gap (top − bot): median **+9.25** (MAD 8.8) vs 6.0 = cathode thickness

The midpoint being consistent with 0 at the few-mm level is the key result: a
relative bottom↔top time-base error of ~21 µs (the §5.2 asymmetry read as a
clock error) would displace the midpoint by ~1.6 cm — mildly disfavored, not
excluded (robust SE ≈ 0.4 cm, but the 26 unvalidated finder pairs contaminate
the tails, means/rms are outlier-dominated). Each half stopping 2–3 cm short
of the surface matches the known "pairs meet 10–22 cm apart" observation that
motivated `xtpc_dmax = 25 cm`.

## 6. Verdicts and open items

**Q1 — anode positions: CONSISTENT.** Imaging (collection-plane anchor +
SP arrival-time convention) and the FV checks (grid-plane edge) describe the
same geometry through a coherent time convention (§2.1); QLMatching, the
clustering `dvm`, and the calib-dump geometry all share the identical
grid-plane/cathode-surface numbers (§4). Caveats to carry: the grid plane is a
DocDB-203 convention 5.7 cm inside the physical CRP face (§2.2), and a
predicted +1.0 cm cathode-ward SP bookkeeping bias (§2.3).

**Q2 — times: CONSISTENTLY APPLIED**, with one output-side asymmetry.
Per-crate offsets are measured, threaded per drift side, and applied with
provably identical signs in QLMatching and clustering (§3.1–3.3); the BDE base
is externally validated to ~1 µs (§3.5); crosser midpoints confirm bottom↔top
coherence at the cathode (§5.4). The calib/Bee dump folds the BDE offset into
the single displayed flash time for both volumes (§3.4) — correct for matching,
but up to 4.6 cm misleading for top-volume drift arithmetic done downstream of
the dumps.

**Open items** (findings, deliberately not "fixed" here):

1. **Near-anode reconstruction gap ~5–7 cm** (§2.4): with T0 verified at the
   millimeter level (crosser cathode ends), the last ~6.5–7 cm of drift
   before the physical CRP face carry no reconstructed points, in both
   volumes. Leading hypothesis: induction-view ROI failure inside the 18.1 cm
   field-response domain → 3-view tiling drops the charge. Decisive tests:
   simulation closure and magnify U/V/W inspection at a validated end
   (§2.4). The residual bottom-vs-top pile-up difference in the unbiased
   scans (bot +3.2 vs top −0.1, §5.2) is at least partly sample scatter — the
   verified pair is symmetric (+2.4/+1.9) — but per-crate response residuals
   are not excluded; fold into the same study.
2. **SP time-shift excess +6.6 µs ≈ +1.0 cm** (§2.3): the `ctoffset = 4 µs`
   choice and the FR-file speed (1.53) vs reconstruction speed (1.568) both
   push reconstruction cathode-ward. If sub-cm absolute-x fidelity is ever
   needed, recompute `ctoffset` against the calibrated drift speed (a
   config-only, knob-gated change).
3. **Dump/display top-volume time skew** (§3.4): consumers keep inheriting
   this (ql_scan viewer, find_boundary.py, and bee3's box/charge shifts §7.3).
   Per-side times in the calib and op dumps (+ a per-TPC `t` in bee3) would
   remove the trap; until then, every dump consumer doing top-volume drift
   math must re-base by `Δ = trigger_offsets_us[1] − trigger_offsets_us[0]`.
4. **Top cathode ends 1.6 cm short** (§5.3): consistent with the crosser-gap
   observation; fold into the same follow-up as item 1.

## 7. The bee3 event display — PDVD anode/cathode audit

`wire-cell-bee3/events/static/js/bee/physics/experiment.js`, class
`ProtoDUNEVD` (bee3 commit a8d216e "update the PDVD geometry in Bee"). bee3 is
a *fourth* consumer of the anode/cathode geometry, and it deliberately uses
the **physical** convention, not the FV convention:

### 7.1 TPC boxes — physical volume, collection-plane edge

Eight boxes, one per WCT anode 0–7: x ∈ **[−341.55, −3.0]** (bottom four) and
**[+3.0, +341.55]** (top four) — cathode drift-facing surface → **collection
wire plane**; y ∈ ±336.4 (matches the post-`tpc_extra_faces`-fix QL bounds);
z split at the per-drift-side gap midpoints. Against the GDML (§2.2: active
LAr 3.06 → 341.50) the box edges are correct to ≤ 6 mm. So:

- bee3 box anode edge = **341.55** (physical), toolkit FV anode edge =
  **335.835** (grid-plane convention) — an *intentional* 5.7 cm difference.
  The box shows the volume; the FV gates the matching.
- Expected visual consequence (not a bee3 misalignment): a T0-corrected
  anode-crossing track ends **~5–7 cm short of the box's anode face** — the
  §2.4 near-anode reconstruction gap. Its cathode end, by contrast, should
  touch the box's cathode face (verified, §5.3).

### 7.2 Drift arithmetic — signs and speed consistent with the toolkit

- `driftVelocity = 0.1568 cm/µs` = toolkit `drift_speed` 1.568 mm/µs ✓.
- `driftDir(i) = i < 4 ? +1 : −1` (PDVD override; the base class's
  alternating-index formula would be wrong for the grouped 8-anode layout) =
  WCT `face_dirx` ✓.
- Two display frames, exact duals of the toolkit's §3.3 convention:
  - *reco frame* (`op.js buildGroup`): charge stays at raw x; the **boxes and
    PDs** are shifted by `+driftV·t·driftDir(iTPC)` to where the detector
    lies on the raw-x axis at flash time t.
  - *detector frame* (`sst.js drawDetectorFrame`): boxes fixed; the **charge**
    is shifted by `gx − driftV·t·driftDir(tpc)` — literally the toolkit's
    `x_t0cor` formula. The TPC for the shift comes from the matched flash's
    `apa` (position-free), with box-containment `tpcOf()` as fallback.
- `drawSpaceChargeBoundary` is uboone-only (early return) — inert for PDVD.

### 7.3 The one inherited inconsistency — op time is BDE-folded

bee3's `t` is `op_t` from the Bee op dump, which the toolkit writes as
`flash->get_time() + trigger_offset_for(input 0)` (§3.4,
`QLMatching.cxx:2538`) — the **bottom/BDE offset for both volumes**. bee3
applies this single time to top-volume boxes/charge too, so its top-volume
drift alignment carries the per-event Δ = off_top − off_bot skew: **up to
4.6 cm (29 µs) in run 039252** (§3.4 table). Bottom-volume alignment is
exact. This is the same trap as the calib dump (open item 3): the clean fix
is per-side times in the op dump, consumed by a per-TPC `t` in bee3; until
then, expect top-volume charge/box misalignment at the few-cm level on
high-Δ events.

### 7.4 Photon detectors (display-only)

40 channels in WCT flash-chain order from
`cfg/.../protodunevd/pdvd-opdet-geom.json` (data-derived, raw_waveform
x/y/z): cathode X-ARAPUCAs at x = 0 (the physical cathode center plane; the
GDML mesh surfaces are at ±2.975), membrane XAs on the walls, and the bottom
PMT grid at x = −336.474 — i.e. drawn ~5.1 cm above the bottom collection
plane, inside the drift volume. These positions are as-recorded in the
detector data and are cosmetic (light-panel drawing and `opTPC()` box
assignment); four dead PMTs (24/27/28/34) hold mirrored placeholder
positions. No reconstruction quantity depends on them.

**Verdict**: bee3 is geometrically consistent with the toolkit — physical
boxes, matched drift speed, sign-correct dual of `x_t0cor` — with exactly one
inherited defect, the §3.4 BDE-folded time applied to the top volume.

## References

- `pdvd/docs/pdvd-tpc-geometry-fiducial.md` — three-source geometry
  reconciliation, cathode-thickness provenance (§3), FV-definition survey.
- `pdvd/docs/pdvd-ql-pending.md` §1 — trigger-offset resolution record and
  beam-flash closure.
- `pdvd/ql_display/docs/ql-cathode-crosser-recipe.md` — candle definitions,
  finder recipes, tag inventory (`crossers`, `boundary`, `candles`).
- `match/docs/qlmatching-code.md` — QLMatching knob reference.

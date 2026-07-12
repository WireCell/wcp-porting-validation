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

> **Update 2026-07-12**: following the PDHD companion exam
> (`pdhd-anode-time-consistency.md`), the FV anode convention was corrected
> to the U strip plane (toolkit `b8f7f3d6`) and the 18 events reprocessed —
> results, and a new selection-side finding, in **§8**.  Sections 1–7 below
> describe the pre-correction (grid-plane) production.

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
duals of `x_t0cor`). Its one inherited defect — applying the BDE-folded op
time to the top volume (up to 4.6 cm skew this run) — is **fixed** as of
2026-07-12: the toolkit now emits per-side times (`op_t1`) and bee3 consumes
them per TPC (§3.4/§7.3, toolkit `e587f357` + bee3 `40c3629`).

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

### 2.3.1 Which piece is data-only, and the PDHD comparison

Splitting the two contributions further, and asking which survives in
simulation:

- **Velocity-mismatch piece (+4.5 mm): cancels exactly in simulation.**
  `detsim` convolves truth charge with this same FR file, and
  `OmnibusSigProc` deconvolves with it, both using the FR's own internal
  `speed = 1.53` (`m_intrinsic_time_offset = fr.origin/fr.speed`,
  `sigproc/src/OmnibusSigProc.cxx:937`). Forward and backward share the
  identical kernel, so whatever time/position mapping it implies is applied
  and then exactly undone — MC never sees the independently-calibrated
  1.568 mm/µs value at all. The 1.53-vs-1.568 discrepancy is a
  model-vs-real-detector mismatch; it only exists when the "real detector"
  side of the comparison is actual data.
- **`ctoffset` piece (+6.3 mm): does not cancel in simulation either.**
  `ctoffset` only appears on the deconvolution side
  (`OmnibusSigProc.cxx:80,1321,1357`) — confirmed by direct search, there is
  no matching step anywhere in the forward chain (`gen/src/Drifter.cxx`,
  `gen/src/DepoTransform.cxx`, `gen/src/DepoFluxSplat.cxx`). Unlike
  `intrinsic` above, which is a *formula* (`fr.origin/fr.speed`) tied to the
  same field-response object the forward convolution also uses — hence
  self-cancelling — `ctoffset` is a free-standing, hand-tuned scalar with no
  analytic tie to the FR at all (its own comment reads `//consistent with
  FR: protodunevd_FR_imbalance3p_260501.json.bz2` — picked *alongside* a
  specific FR file, not derived from it; a superseded value is commented out
  next to it, `sp.jsonnet:117-118`). With nothing on the forward side to
  remove it, `ctoffset`'s value shows up as a real offset between truth and
  SP output in MC exactly as in data. This is corroborated directly, not
  just argued from code structure: DNN-ROI-SP measured a real, nonzero
  truth-vs-reco tick lag in simulation (7 ticks pre-fix on PDVD, 0 on the
  PDHD control) by cross-correlating truth against actual SP output (§2.3.2)
  — hard evidence this piece does not cancel on its own and needed an
  explicit, empirically-measured correction.
- The §2.4 "simulation closure" follow-up below (inject truth deposits at
  known x through the identical NF/SP/imaging chain) remains the direct test
  of the *position-space* consequence for imaging. The *tick-space*
  consequence is no longer hypothetical, per the point above — what that
  closure test would still add is a clean decomposition of how much of the
  measured MC lag is `ctoffset` alone vs. smearing/electronics-response
  differences between the simplified truth splat and the full SP chain.

**PDHD comparison** (`pdhd-anode-time-consistency.md` §2.3: `speed = 1.565`
vs calibrated `drift_speed = 1.576`, `fr.origin = 100 mm`, `ctoffset = 1.0 µs`):

| | PDVD | PDHD | ratio |
|---|---|---|---|
| FR speed vs calibrated | 1.53 vs 1.568 (2.4% off) | 1.565 vs 1.576 (0.7% off) | 3.4× |
| `fr.origin` | 181 mm | 100 mm | 1.8× |
| **velocity-mismatch piece** | **≈4.5 mm** | **≈0.70 mm** | **≈6.4×** |
| `ctoffset` | 4 µs | 1.0 µs | 4× |
| **ctoffset piece** | **≈6.3 mm** | **≈1.58 mm** | **≈4×** |
| tick-floor loss | ≈0.47 mm | ≈0.63 mm | — |
| **net predicted excess** | **+1.03 cm** | **+0.165 cm** | **≈6.2×** |

Both pieces scale as expected: PDVD's larger `fr.origin` (thicker FR domain)
amplifies both contributions on top of PDVD's larger velocity mismatch and
larger `ctoffset`; the velocity-mismatch ratio (≈6.4×) tracks
`%mismatch × fr.origin` (3.4×1.8 ≈ 6.2) and the `ctoffset` ratio tracks the
`ctoffset` ratio itself (4×) almost exactly, since the two detectors'
calibrated speeds are within 0.5% of each other. Practically, PDHD's total
(+0.165 cm) sits well inside its own measured cross-check agreement of
~4 mm (PDHD doc §2.4) — invisible at PDHD's residual level — whereas PDVD's
(+1.03 cm) is large enough to be a real candidate contributor to the
measured, still-unexplained asymmetric anode-edge gap (§5.2).

### 2.3.2 How the DNN-ROI-SP training avoided this

The PDVD DNN-ROI-SP training pipeline (`DNN_ROI_SP/` — truth-labeled ROI
masks generated from simulation, per `DNN_ROI_SP/docs/truth_labeling_algorithm.md`)
never has to reason about either piece above, for two independent reasons:

1. **It lives entirely in tick space, never tick→x.** Truth ROI masks are
   built by `DepoFluxSplat` (`gen/src/DepoFluxSplat.cxx`) directly from
   drifted depos on the `(channel, tick)` grid, and training/validation
   compare truth vs. reco waveforms on that same grid
   (`DNN_ROI_SP/scripts/utils/h5_utils.py`'s `get_masks`,
   `DNN_ROI_SP/scripts/measure_truth_reco_offset.py`). The §2.3.1
   velocity-mismatch bias only exists at `BlobSampler::time2drift`'s tick→x
   conversion, which uses the calibrated 1.568 mm/µs on data — a stage
   downstream of, and never reached by, DNN-ROI-SP.
2. **The truth/reco tick alignment is measured empirically, not derived
   from a formula**, so it doesn't matter whether the derivation would have
   included `ctoffset` or not. `DepoFluxSplat.reference_time = -3500 ns`
   (`DNN_ROI_SP/simulation/stageB/wct-depo-sim-deposplat.jsonnet:125`,
   subtracted from the truth frame's t0 at `DepoFluxSplat.cxx:410`) is a
   +7-tick shift determined by cross-correlating truth against reco
   waveforms (`measure_truth_reco_offset.py`): PDHD (control) = 0 ticks,
   PDVD = 7 ticks pre-fix, **0 ticks post-fix on both drift halves**
   (`DNN_ROI_SP/docs/pdvd_truth_corrections.md` §2). That single measured
   number absorbs the *entire* net truth-vs-reco lag — response-plane
   placement, FR shape, `ctoffset`, smearing — whatever it is, by
   construction. (It is a small residual on top of a separate ~246-tick
   tickinfo auto-alignment, and is not the same 8-tick quantity as
   `ctoffset` itself.)

Consistency check: `ctoffset = 4 µs` is inherited unmodified — never
overridden — by the training-data generation, sim-inference, and
data-inference jsonnets alike (`wct-depo-sim-deposplat.jsonnet:74-86`,
`DNN_ROI_SP/simulation/stageB_pdvd/wct-sim-nf-sp-dnnroi-pdvd.jsonnet:102-107`,
`wcp-porting-img/pdvd/wct-nf-sp-dnnroi.jsonnet:121-126` all pull
`protodunevd/sp.jsonnet:118` with no override), so the tick placement the
model learned in training transfers to inference on both simulation and
data.

**Reconciling §2.3.1's two pieces with what DNN-ROI-SP actually needed to
correct**: the velocity-mismatch piece (provably zero in MC) is specific to
`BlobSampler::time2drift` converting reco tick to absolute x *on data* — a
stage DNN-ROI-SP never reaches, so it simply isn't part of this story. The
`ctoffset` piece is different: it does **not** cancel in MC either
(§2.3.1, revised), and DNN-ROI-SP's own +7-tick empirical correction is
direct evidence of exactly that — a real, MC-internal tick-space offset it
had to measure and remove, not something simulation's self-consistency
spared it from. DNN-ROI-SP works correctly despite this not because the
offset is absent in MC, but because it was *measured* empirically (point 2
above) rather than assumed away. Practical note for future re-validation:
the V-plane SP fix
(`vplane_low_freq_pole.md`) shifted reco +200 ns and its own doc flags that
downstream tick-alignment calibration may need re-checking — exactly why
the empirical cross-correlation approach (re-measure the lag, don't assume
a formula) is the robust choice here, rather than deriving the shift from
`ctoffset` and `fr.speed` by hand.

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

**FIXED 2026-07-12** (toolkit `e587f357` + bee3 `40c3629`): when `shared_flash`
+ per-input `trigger_offsets` are configured (PDVD only), the dumps now carry
the top-clock time explicitly — per-flash `time1` (µs) in the calib dump, a
`time1` column in the opflash PC, and a parallel `op_t1` array in the Bee
op JSON — and bee3 consumes `op_t1` for TPC 4–7 via a new per-TPC
`flashTimeForTPC` hook (base class unchanged). Keys are absent when
`trigger_offsets` is empty, so PDHD/SBND outputs are byte-identical (gate:
PDHD 29107 evt0 full QL rerun, all 15 `mabc-*.zip` member-identical to
production, `work/029107_0_qlt`; PDVD 039252 evt0 calib+op identical after
stripping the new keys, `time1 − time == Δ` on all 197 flashes,
`work/039252_0_qlt`). Dumps produced before this fix still need the manual
re-basing above; the ql_scan viewer and `find_boundary.py` still read
`f["time"]` and can adopt `time1` as a follow-up.

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
3. **Dump/display top-volume time skew** (§3.4): **RESOLVED 2026-07-12** —
   toolkit `e587f357` emits per-side times (calib `time1`, opflash-PC
   `time1`, Bee `op_t1`; keys gated on per-input `trigger_offsets` ⇒
   PDHD/SBND byte-identical, gates in §3.4) and bee3 `40c3629` consumes
   `op_t1` for the top volume. Remaining follow-ups: ql_scan viewer and
   `find_boundary.py` still read `f["time"]` (should adopt `time1`), and
   dumps generated before the fix need the manual Δ re-basing.
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
exact.

**FIXED 2026-07-12** (bee3 `40c3629`, consuming toolkit `e587f357`'s new
`op_t1`): `Experiment.flashTimeForTPC(op, iTPC)` (base = `op_t`, so
PDHD/SBND behavior unchanged) is overridden by `ProtoDUNEVD` to return
`op_t1[currentFlash]` for TPC 4–7 when the op dump carries it; both display
frames (op.js box/PD shift, sst.js detector-frame charge shift) now use the
per-TPC time. Older op dumps without `op_t1` fall back to the previous
behavior — regenerate an event's mabc zips to get the aligned display.

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

## 8. 2026-07-12 — FV anode moved to the U plane (PDHD convention) and first reprocessing

### 8.1 The change

Toolkit `b8f7f3d6` (owner-directed, follows the PDHD companion exam):

- `params.jsonnet`: xregions `anode` moved from the DocDB-203 grid plane
  (`apa_plane = 0.5·apa_g2g = 57.15 mm` → ±335.835 cm, no physical
  counterpart, §2.2) to the **first-induction (U) strip plane**
  (`apa_plane = 2 × 0.2 mm` → **±341.51 cm** ≈ the GDML active edge 341.50).
- `clus.jsonnet` `dvm` per-drift `FV_x` moved to match (∓3358.35 →
  **∓3415.1 mm**).

Everything derived follows: QLMatching `anode_x = ±341.51`,
`u_cathode = 338.51` (was 332.835); u = 0 is now the physical CRP active
edge.  **NOT byte-identical** — containment, boundary flags, and hence
matching change.  PDHD/SBND untouched.

### 8.2 Reprocessing

```bash
cd pdvd && PDVD_MAX_JOBS=6 ./run_clus_evt.sh -s anodefix -calib 039252 all
cd docs/qlmatch && python3 check_anode_time_consistency.py --tag anodefix
```

18/18 events OK into fresh `work/039252_{0..17}_anodefix/` (cluster tarballs
symlinked from the production dirs; imaging NOT re-run — the anode xregion
does not enter tiling/sampling).  Dumps verified to carry the new geometry
block.  The check script gained `--tag` (strict dir match, so the original
Repro never mixes tagged and untagged dirs) and tolerates decisions-file
uids absent from a reprocessing.

### 8.3 Results — geometry right, selection now biased at the anode

In the new frame (u = 0 = U plane; old u = new u − 5.675):

- **Validated boundary tracks (check A, same cluster+flash pairs as
  production, 53 bot + 55 top survive):** anode-end u = **+5.68 (bot) /
  +4.19 (top)**.  This is the §2.4 near-anode missing-charge gap, now
  expressed against the physical edge: real reconstructable ends stop
  4–6 cm short of u = 0.  Expected, and the honest picture.
- **Unbiased auto-bundle scan (A′): peak +0.5 on both sides** (core-median
  +0.22 bot / +0.90 top).  This is **not** charge reaching the edge — a
  decomposition against the production selections proves it:

  | subset (span ≥ 30 cm ends within ±12 cm) | bot | top |
  |---|---|---|
  | selections **unchanged** vs production (89% of all autos) | peak +3.5, median +3.49 (n = 21) | peak +3.5, median +3.99 (n = 31) |
  | selections **changed** (242 clusters swapped flash, ~10%) | **peak +0.5, median +0.14 (n = 47)** | **peak +0.5, median +0.59 (n = 58)** |

  The pile-up at the new edge is composed almost entirely of *newly swapped*
  flash choices whose T0 places the track end inside the anode flag window.
- **Cathode cross-check (B′): bottom regressed +0.41 → +5.23** (top +1.58,
  unchanged).  The swapped bottom matches are mis-T0'd wholesale by ~5 cm —
  their cathode ends now also fall 5 cm short, where production had them on
  the surface to 4 mm.  (Crosser midpoints, check D, are decisions-pinned
  and unchanged at −0.27 — the underlying physics/time chain is untouched.)

### 8.4 Interpretation and recommendation

The anode boundary window `[anode_ext1, anode_ext2] = [−2, +4] cm` about
u = 0 grants match advantages (at_x_boundary: LASSO down-weight, overpred
exemption, relaxed-chi2 ladder branch).  Under the old convention u = 0 sat,
by coincidence, at the *reconstructable-charge* edge, so this window
captured genuine truncated ends.  With u = 0 now at the *physical* edge —
which reconstructed charge never reaches (the 4–6 cm FR-domain gap) — the
window covers a band where true ends essentially cannot be, and the
advantage instead attracts flashes ~30 µs off that slide some other end
into the window.  Net: the FV geometry is now correct, but the
boundary-flag *cushions* still encode the old coincidence.

**Recommended next step (not implemented here — owner's call):** retune the
PDVD anode cushions to the measured reconstructable band, e.g.
`anode_ext2: +4 → ~+10 cm` (so genuine truncated ends at +4..+9 keep their
boundary flags and the true flash regains its advantage) and consider
raising `anode_ext1` toward ~+1 cm (ends cannot physically reconstruct at
u < 0 by more than the §5.2 scatter, so the sub-zero window only ever
admits wrong-flash ends).  Then revalidate against the run-039252 hand-scan
labels (auto/manual agreement rate) before adopting.  Alternatively, once
the near-anode charge loss itself is fixed (open item 1), the cushions
revert to PDHD-like values.

### 8.5 Self-consistency of the current result's own boundary flags (no external truth)

During QLMatch tuning the correct T0 is unknown, so the §8.3 decomposition
against production selections (and the hand/finder-validated pairs) cannot
arbitrate which selection is right.  This check therefore uses **only the
current auto result**: for auto-selected bundles that QLMatching itself
flagged (`two_boundary`, `QLMatching.cxx:3722`; anode-window
`at_x_boundary`, `:3639`), where do the anode-facing PCA ends sit relative
to u = 0 at each bundle's *own selected flash* T0?

Repro:

```
cd pdvd/docs/qlmatch
python3 check_flagged_boundary.py --tag anodefix   # current U-plane result
python3 check_flagged_boundary.py                  # production baseline
```

(Per-side flash times: apa 0 = `f["time"]`, apa 4 = `f["time1"]`.  PCA ends
are an SVD proxy for `get_extreme_wcps` — see the script docstring.)

**Anode-facing-end u (cm), current `anodefix` result** (18 events, 2371
auto bundles, 123 two_boundary, 1853 at_x_boundary of which 175 anode-only):

| group | circularity | bot | top |
|---|---|---|---|
| two_boundary | forced: 3 cm flag margin | n=25, med **−0.29**, MAD 1.7 | n=12, med **+0.16**, MAD 2.2 |
| at_x_boundary anode-only | forced: [−2,+4] window | n=100, med +1.14, rms 18.2 | n=122, med +2.90, rms 24.1 |
| all auto (no flag gate) | none | n=160, med **+2.94** | n=202, med **+6.61** |

So at face value: **yes, the flagged ends are consistent with the new anode
(u ≈ 0 within ~±0.3 cm median)** — but rows 1–2 are consistent *by
construction*: the flags are only granted when an end falls in the window
at that flash's T0, so the flagged population self-selects flashes that put
an end at u ≈ 0.  The unforced bulk (row 3) piles several cm inside, as
§8.3 found.

**The non-circular test — span closure.**  For the 13 two_boundary bundles
whose two faces are anode+cathode, the u-span is T0-independent (a drift
shift moves both ends equally), so `span ≈ u_cathode = 338.51` is a
physical closure no flash choice can fake:

- **Only 2/13 close physically** (closures −2.0, −3.6 cm): evt 298777
  uid 181 and evt 298749 uid 3, both bottom.  These keep the *same flash*
  as production (gids 130, 105) and their anode ends genuinely reach
  **+1.7 / +0.7 cm** of the physical U-plane edge — evidence the 4–6 cm
  FR-domain gap of §8.3 is a *median*, with a leading tail that does reach
  the physical edge.
- **11/13 are impossible** — spans exceed the full drift by +7 to +91 cm
  (pileup-merged clusters / wrong flash).  They carry the flag because
  `compute_two_boundary_flag`'s nearest-face test uses **signed** distance
  ≤ margin, so an end arbitrarily far *outside* the anode (u = −89 cm)
  still counts as "at the edge", and nothing vetoes span > drift.
  Pre-existing diagnostic-flag limitation, reported not fixed.
- Production baseline had **6/13 physical**.  The 4 that disappeared are
  precisely the honest full-drift tracks whose anode ends sit at
  **+6.6..+8.1 cm** in the new frame (old-frame +1.0..+2.4) — the FR gap
  pushed them outside the 3 cm two_boundary margin, so under the new
  convention the flag keeps mostly junk plus the rare tracks that truly
  reach the edge.

**Conclusion:** the current selected boundary-flag clusters agree with the
anode position, but that agreement is what the flag *enforces*, not
evidence the selections are right; the T0-free span closure shows most of
the flagged full-drift candidates are unphysical.  This is the §8.4
mechanism seen from inside the current result alone, and it reinforces the
cushion retune: with `anode_ext2 → ~+10 cm` (and the same widening applied
to the diagnostic margin, or an |distance| + span-sanity fix to
`compute_two_boundary_flag`), the genuine +4..+8 cm truncated ends would
re-enter the flagged population and the u ≈ 0 junk would lose its
monopoly on the advantages.

### 8.6 The cathode side is the velocity meter — v = 1.568 provenance revisited

The anode-side residuals (§8.3/§8.5) are essentially **velocity-blind**: the
imaging x anchor is at the collection plane, so a velocity error δv/v scales
an end's distance *from that plane* — ~0.05 cm at an anode end sitting 5 cm
away, but **D·δv/v ≈ 3.4 cm per 1 % at the cathode end**.  Conversely the
cathode-side residuals are exactly where a velocity error would show.  So
the three observables separate cleanly:

| observable | measures | current value (anodefix, decisions-pinned) |
|---|---|---|
| beam-flash closure | absolute (BDE) time base | −0.9 µs (§3.5) |
| anode-end u of validated tracks | near-anode charge loss g | +2..+8 cm (§8.3/§8.5) |
| cathode-end residual | c_loss − D·(δv/v) — **degenerate combination** | core median **+1.0 (bot, n=4) / +0.3 (top, n=4)** cm short (check B); crosser pair gap median **+9.25** vs 6.0 cm = ~1.6 cm short per half (check D, n=43) |

Repro: `python3 check_anode_time_consistency.py --tag anodefix`, checks
B/C/D.

**Span closure decomposes the deficit onto the anode end.**  The validated
anode→cathode tracks' T0-free u-spans (check C, dropping the over-merged
+15..+32 tail) run **−1.7 to −9.2 cm short of D** (median −7.7).  Per track,
`(anode-end u) + (cathode shortfall) = span deficit` gives cathode
shortfalls of **−2.2..+2.8 cm** while anode ends carry +2..+8 — the missing
span lives at the anode, and the cathode ends sit on the surface to ~±2 cm
*at v = 1.568*.

**This breaks the stated provenance of v = 1.568.**  The calibration
(`pdvd/drift_calib/calib_drift_velocity.py`) assumed genuine full crossers
span the whole `D = 338.55` (collection → cathode surface) and solved
`v = v_reco·D/S` from the span pile-up.  We now know genuine crossers span
`D − g − c` = 329..337 cm — the assumption is wrong by the anode gap.  The
closure plot (`drift_velocity_calib.png`, N=51 @1.57 → 1.561) is in fact
double-bumped, with its pile-up (340.5) *above* the genuine validated-span
population (329..337): the estimator keyed at least partly on the
over-merged/overshooting population that happens to sit near D.  So the
span method does not anchor v to better than ~1–2 %.

(Recorded tension, stated for completeness: the 2026-07-08 per-TPC
recheck — `plot_tpc_split.py`, 142-event sample — found both TPCs' span
histograms peaking at 338–341 even in its "tight" cathode window.  But that
window, `cath_coord ∈ [−5,+8]` about x = 0, still admits up to ~11 cm of
overshoot past the −3 cm cathode *surface*, so mildly mis-T0'd/merged
tracks populate the peak; the decisions-pinned validated crossers here
(hand-scanned flash, n = 8 after dropping the merged tail) are the
higher-purity sample and pile 2–9 cm lower.  n = 8 is small — re-running
the validated-span measurement on the full 142-event sample with hand-scan
labels would settle it.)

**What actually supports v = 1.568 now** is the cathode-side agreement
above: if the true cathode-side reconstruction loss is 0..2 cm (SCE +
imaging threshold), then D·(δv/v) = c_loss − c′ ≈ −1..+2 cm, i.e.
**v correct to ~±0.5 % (±0.008 mm/µs)** — but c_loss is not independently
measured, and a coherent (c_loss, δv) trade-off along that line cannot be
excluded from this data.  Note the earlier calibrations self-consistently
land in this band (1.568 config, 1.561 closure re-run ≈ −0.45 %).

**Independent handles that would break the (c_loss, v) degeneracy** (open,
none implemented):
1. Expectation at the PDVD field: Walkowiak-style v(E, T) at the nominal
   drift field — a ±0.5 % band prediction, no reco input.
2. Sim closure: reconstruct simulated full crossers (known v_sim = 1.473
   convention) and measure the *reconstruction-induced* cathode shortfall
   c_loss and anode gap g directly; then data cathode residuals convert to
   a v measurement.
3. Angle dependence: charge-loss gaps should depend on track angle to the
   drift axis; a velocity error would not.  The validated-track span
   deficits already spread −1.7..−9.2, hinting angle/topology dependence
   (loss-like, not velocity-like).

### 8.7 The anode gap is at the imaging level, and angle-independent

Question posed (2026-07-12): with the time offsets verified end-to-end, the
FV anode at the physical U plane, and imaging anchored so that zero drift
time reconstructs at the anode — do anode-crossing tracks in data actually
reconstruct to the anode?  **No.**  Repro:

```
cd pdvd/docs/qlmatch
python3 check_anode_gap_imglevel.py
```

Method: for every validated boundary/crosser track (decisions-pinned flash),
take the cluster's anode-most end u; fit the local track direction from the
last 150 cm; extend the line toward u = 0 and search the event's ENTIRE
imaging point cloud (`0-img-global.json`, all clusters, q > 0; verified to
share the calib dump's raw-x frame point-by-point, Δx = 0.00) for points
beyond the cluster end within a 4 cm tube.  Only tracks whose extrapolated
entry point at u = 0 lies ≥ 10 cm interior in y/z count — those *must*
physically cross the CRP (a cosmic cannot begin mid-volume).

Results (76 interior-entry tracks with end u in [0, 12] cm):

- Cluster anode ends stop at **median +4.7 cm** (bottom **+5.7**, top
  **+4.2**), and for the overwhelming majority the imaging cloud contains
  **no points beyond the cluster end at all** — the gap is NOT clustering
  fragmentation, NOT flash selection, NOT the FV: **the points are absent
  at the NF/SP/imaging level**.  (The few apparent line hits at
  u = −400..−10 are unrelated tracks crossing the extrapolated line; a
  track's T0 applied to another track's points is meaningless.)
- The gap is **angle-independent**: medians +4.2 / +5.3 / +4.5 / +4.9 cm
  across dip bins 0–25 / 25–45 / 45–60 / 60–90°, corr(dip, gap) = +0.20.
  This argues against a track-topology mechanism (prolonged-ROI failures
  would scale with drift alignment) and for a *uniform position-dependent*
  loss over the last ~5 cm before the CRP.  (It also weakens §8.6 handle 3
  as stated: the span-deficit spread is not obviously angular.)
- Crosser halves cannot test this directly — their matched clusters are
  fragments (anode-side ends anywhere at u = +8..+300); the line probe is
  what restores the test.

**Per-side numbers with stopping-muon discrimination**
(`check_anode_gap_perside.py`; a bottom-volume anode end is an *exit* point,
so a ranging-out muon could in principle fake a gap there — the Bragg check
(track-end dQ over 0–15 cm vs mid-track 30–120 cm) rules this out):

| volume | n | gap median | MAD | Bragg ratio median | stopping-like (>1.5) | MIP-only gap |
|---|---|---|---|---|---|---|
| bottom (interior exit) | 37 | **+5.6** | 1.2 | 0.98 | 4/29 | **+5.5** (n=25) |
| top (interior entry) | 37 | **+4.2** | 1.2 | 1.01 | 5/33 | **+4.1** (n=28) |

Both pile-ups are sharp (MAD ~1.2 cm; bottom histogram peaks in the 4–6 cm
bins, top in 3–5) and the ends are charge-flat (MIP-like) — a stopping
population would be broad and Bragg-rising, so the bottom gap is as real as
the top one.  The **~1.4 cm bottom−top difference** is a per-side (BDE vs
TDE electronics / SP chain / FR usage) effect and matches the §5.2
asymmetry direction.

Candidate mechanisms for the ~5 cm loss (open; PDHD with the same toolkit
machinery shows −0.35 cm, so the cause is PDVD-specific — FR model, CRP
strip response, DNNROI model, or noise):

1. **FR-domain truncation**: charge born within the 18.1 cm response region
   induces a truncated version of the assumed full-path kernel; the
   deconvolved amplitude drops as the path shortens, falling below
   ROI/imaging thresholds in the last few cm.  Angle-independence is
   consistent (the loss depends on birth distance, not direction).
2. DNNROI training coverage of very-early-drift (near-CRP) signals.
3. A real detector-field effect near the perforated CRP (charge funneling /
   collection loss within the last few cm).

**Decisive next test — sim closure**: push simulated CRP-crossing tracks
(truth known) through the identical NF/SP/DNNROI/imaging chain and measure
the reconstructed gap.  Sim reproduces +4..+6 cm → inherent to the FR/SP
model; no gap in sim → data-specific (noise/DNNROI).  Complementary
waveform-level check: for one validated track, inspect the SP (and pre-ROI
decon) frames at the strips/ticks where the missing 5 cm should sit — is
there sub-threshold charge?

### 8.8 Chain trace to the raw waveforms — the "uniform 4–6 cm gap" is REVISED

Per owner direction (2026-07-12, before any simulation): check imaging vs
deconvolution (gauss) vs raw signal (W plane, post-NF `frame_raw0`) for
individual cases.  Repro:

```
cd pdvd/docs/qlmatch
python3 check_chain_consistency.py     # 5 cases; writes chain_<evt>_<uid>.png
```

Method: map (y, z, x_raw) → (W channel, tick) via the wires store
(`img_plot/geom.py`; per-side tick = (x_raw − plane_x)·dirx/v/0.5 µs);
validate on each cluster's actual points (gauss ridge found at 100 % of
mapped points, tick offset ≤ ~10); then probe the extrapolated corridor
beyond the imaging end down to u = −3 with a windowed ridge search, in both
`gauss` and `raw0`.  The anode-crossing point in time is the (per-side
folded) flash time itself.

**Case results:**

| case | topology | imaging end u | gauss/raw along corridor | verdict |
|---|---|---|---|---|
| 298749 uid 3 (GOLD full-drift) | drift-aligned | +0.70 | signal **through u = 0** (raw pk 468 at +0.2, tail to −0.8) | track reaches the anode; chain records it |
| 298777 uid 181 (GOLD full-drift) | steep | +1.66 | signal to +1.2, noise below | ~1 cm true gap |
| 298693 uid 4000003 | **V-topology** (two legs meet at min-u apex) | +4.46 | raw band ends AT the apex; nothing below in gauss OR raw | apex is a scatter/kink mid-volume — this track never reaches the CRP; no charge is missing |
| 298791 uid 259 | drift-aligned, weak | +3.89 | patchy gauss+raw charge continues to u ≈ −1 | **imaging dropped ~5 cm of real SP charge** (3-view/threshold/DNNROI) |
| 298609 uid 38 | — | +3.95 | end is a satellite blob 60 cm (in z) from the track body | cluster end ≠ track end (clustering artifact); case unusable |

**What this revises:**

1. **Raw–decon–imaging are mutually consistent at track ends** in 4/5 cases
   — where imaging stops, the deconvolved charge and the raw waveform stop
   too.  There is no generic several-cm data loss inside the SP→imaging
   chain at the anode.
2. **Where a track verifiably crosses the CRP** (the two gold full-drift
   closures), charge is recorded and reconstructed to **+1.2 / −0.8 cm** of
   the U plane.  The true near-anode loss is **~0–2 cm**, PDHD-like — not
   4–6 cm.
3. **The +4–6 cm pile-ups of §8.3/§8.7 are therefore reinterpreted**: the
   `boundary` sample inherits the finder's ±3 cm window around the OLD
   grid-plane edge (= +2.7..+8.7 cm in the new frame), and that band is
   populated by tracks whose minimum-u point genuinely lies there —
   V-topology scatter apexes (case 298693: straight-line entry-extrapolation
   is invalid across a kink, so the "guaranteed anode crossing" argument of
   §8.7 fails for them), side-wall entries, and fragments.  §8.7's
   imaging-level probe result ("no img points beyond cluster ends") stands,
   but it means the cluster ends are real signal ends — not that charge is
   lost.
4. **One real chain loss found, distinct and topology-specific**: for the
   drift-aligned weak case, imaging fails to convert ~5 cm of existing SP
   charge near the anode (candidate: induction-view ROI/coincidence or
   thresholds for near-vertical tracks).  Actionable separately.
5. Knock-ons: §8.4/§8.5's cushion-retune recommendation (`anode_ext2 → +10`)
   loses its main premise — the physical [−2, +4] window at the U plane is
   defensible since genuine crossers DO reconstruct near u = 0.  (The §8.3
   bottom cathode-edge regression remains evidence that many of the
   *specific* swapped selections are mis-T0'd; auto-match purity vs the
   hand labels still needs its own assessment.)  §8.6's "validated span
   deficit −7.7 median" sample likewise contains non-crossers; the clean
   full-drift deficits are the gold pair's −2.0/−3.6 cm (≈ 1–2 cm per end),
   which *strengthens* v = 1.568 (cathode residual ≈ 0 needs no big
   c_loss).
6. The sim closure is still worthwhile but re-targeted: confirm sim tracks
   reconstruct to the CRP, and try to reproduce the drift-aligned imaging
   loss (case 298791-type).

## References

- `pdvd/docs/pdvd-tpc-geometry-fiducial.md` — three-source geometry
  reconciliation, cathode-thickness provenance (§3), FV-definition survey.
- `pdvd/docs/pdvd-ql-pending.md` §1 — trigger-offset resolution record and
  beam-flash closure.
- `pdvd/ql_display/docs/ql-cathode-crosser-recipe.md` — candle definitions,
  finder recipes, tag inventory (`crossers`, `boundary`, `candles`).
- `match/docs/qlmatching-code.md` — QLMatching knob reference.

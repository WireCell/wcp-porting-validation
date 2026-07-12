# PDVD photon models: what exists, what we built, what QLMatching runs

Answers, in one place: is there a table-based (voxel) photon library from
official LArSoft, in addition to the two deep-learning (Ar/Xe) models? What did
the semi-analytical-model calibration produce? Which of these are actually
wired into `QLMatching` today?

**TL;DR** — there are **four** artifacts, not two:

| # | model | mechanism | source | integrated into WCT Q/L? | role today |
|---|---|---|---|---|---|
| 1 | `PDFastSimANN` — **Argon / 128 nm** | TensorFlow MLP ("computable graph") | official dunesw sim | via #4 (resampled) | available, not default |
| 2 | `PDFastSimANN` — **Xenon / 175 nm** | same MLP, Xe-doped LAr | official dunesw sim | via #4 (resampled) | **default** for PDVD |
| 3 | `PDFastSimPVS` **voxel library** (LArSoft table) | precomputed 25³ voxel table, one entry per (voxel, OpChannel) | official dunesw sim (disabled fallback) | **no** | cross-check of #1 only |
| 4 | **`PhotonLibraryModel`** (WCT) | trilinear lookup on a 10 cm grid, values = **#1 or #2 resampled** | our own sampling script | **yes** | PDVD production backend (`light_model:'library'`) |
| 5 | **`SemiAnalyticalModel`** (WCT) | closed-form solid-angle + Gaisser–Hillas, coefficients fit to #1/#2 | our own fit | yes, as fallback | `light_model:'semi'` (PDVD fallback; PDHD/SBND default) |

So: **yes**, a table/voxel-based photon library from the original LArSoft chain
does exist for PDVD (row 3) — but it is not what we integrated. What we
integrated is a resampling of the **deep-learning** model (rows 1/2) onto our
own grid (row 4), plus a from-scratch calibration of the semi-analytical form
against that same deep-learning model (row 5). The rest of this doc explains
each row and exactly what `QLMatching` runs.

Everything below is reproducible from `pdvd/photlib/` in this repo; the
underlying investigation doc is `pdvd/docs/pdvd-photon-model.md` (2026-07-09).

---

## 1. What the official PDVD simulation uses

dunesw `v10_05_00d00` (models unchanged in later releases):

| chain | optical config | model |
|---|---|---|
| `protodunevd_g4_stage2.fcl` (stock, nominal) | `protodune_vd_pdfastsim_ann_ar` → `protodune_vd_v4_pdfastsim_ann_ar` | **PDFastSimANN**: TFLoaderMLP, `PhotonPropagation/ComputableGraph/protodune_vd_v4_128nm_tf2.6` (Ar); `..._175nm` variant (Xe 10 ppm) |
| `protodunevd_refactored_g4_stage2.fcl` (our DNN-ROI training chain) | `#PDFastSim` commented out | no light at all — stops at `IonAndScint` |
| `protodunevd_refactored_g4_stage2_pureAr.fcl` | `protodunevd_v4_Ar_photonvisibilityservice` | **PDFastSimPVS** voxel library (row 3 below); Xe variant alongside |

Definitions: `duneopdet/v10_05_00d00/fcl/PDFastSim_dune.fcl:236-259`,
`photpropservices_dune.fcl:534-556`. Files live on StashCache/cvmfs
(`/cvmfs/dune.osgstorage.org/.../PhotonPropagation/{ComputableGraph,LibraryData}/`),
which also carries the **v5** graphs `protodune_vd_v5_{128,175}nm_tf2.6` used
below. For contrast, PDHD's stock chain runs `PDFastSimPAR` (the
semi-analytical model) in the active volume plus a PVS library for the
external buffer — PDVD's nominal chain does **not** use the semi-analytical
model at all; that's a WCT-side calibration, not an official-sim choice.

### Row 1/2 — the two deep-learning ("Ar" / "Xe") models

`PDFastSimANN`: a TensorFlow MLP that maps a 3D position directly to 40
per-OpDet visibilities (no voxelization inside the model itself). Two trained
graphs exist, differing in the assumed LAr scintillation wavelength/doping:

- **Argon, 128 nm** — pure LAr scintillation.
- **Xenon, 175 nm** — Xe-doped LAr (10 ppm), which shifts VUV scintillation to
  175 nm and changes which channels are sensitive (see §5, "Ar-blind"
  channels).

Both come in **v4** and **v5** geometry variants. This matters: cross-matching
the v4/v5 GDML `volOpDetSensitive_*` positions against the official channel map
and our own raw-data opdet-position table showed **v4's Arapuca layout is
y-mirrored relative to the as-built detector**, while **raw-data positions
match v5 exactly** (max Δ = 0.0 mm over all 36 live channels). Any model used
against real data must therefore be the **v5** ANN, not v4. (`extract_photlib.py`,
`gdml_opdets.py`, `sample_ann.py` in `pdvd/photlib/`.)

### Row 3 — the LArSoft voxel table (does exist, not integrated)

Yes — a classic, precomputed photon-library table exists for PDVD, in the same
`PhotonVisibilityService` format LArSoft has used for years:

- ROOT TTree `PhotonLibraryData(Voxel, OpChannel, Visibility)`, **40
  OpChannels**.
- 25×25×25 voxels over the cryostat bounding box (790×854.8×854.8 cm, centered
  at world (20, 0, 149.65) cm) → **31.6 / 34.2 / 34.2 cm voxels** — coarse.
- File: `libext_protodunevd_v4_{Ar,Xe}_Baseline_v09_69_00d00_5e7_25x25x25_landau_20231216.root`
  (5×10⁷ photons/voxel Geant4 MC), on StashCache. **v4 geometry only.**
- In dunesw it is a **disabled fallback** (`PDFastSimPVS`), used only by the
  pure-Ar training variant of the g4 stage-2 fcl, not the nominal chain.

We extracted and used this table for exactly one purpose: **validating the ANN**.
Ar↔Xe libraries correlate at 0.944 in log-visibility. **The voxel table itself
was never wired into QLMatching** — it is v4-only (mirrored geometry, wrong for
data) and far coarser than what we built from the ANN instead.

#### Row 3a — which doping does the voxel library match: Ar/128 nm or Xe/175 nm?

Both the voxel library and the ANN exist in **both** dopings for v4, so this is
answerable directly as a 2×2 consistency check, not just the single Ar-lib
vs 128nm-ANN number quoted above. `pdvd/photlib/compare_lib_ann.py` evaluates
all four v4 ANN/library combinations at the library's own 15,625 voxel centers
(600k (voxel,channel) entries per pair on the shared support where both sides
are nonzero):

| library | vs 128 nm ANN | vs 175 nm ANN |
|---|---|---|
| **Ar** | corr **0.991**, ratio 16/50/84% = 0.78/0.99/**1.17**, log-ratio RMS **0.33** | corr 0.929, ratio 16/50/84% = 1.28/3.30/9.38, log-ratio RMS 1.59 |
| **Xe** | corr 0.934, ratio 16/50/84% = 0.10/0.30/0.77, log-ratio RMS 1.65 | corr **0.986**, ratio 16/50/84% = 0.84/0.99/**1.16**, log-ratio RMS **0.27** |

**Answer: each voxel library matches its own-doping ANN, cleanly, and matches
the other doping's ANN poorly.** The Ar library is closer to the 128 nm ANN
(higher correlation, median ratio ≈1, ~3× smaller log-ratio scatter); the Xe
library is closer to the 175 nm ANN. The off-diagonal pairs are not a subtle
effect — the Xe-lib/128nm-ANN median ratio is 0.30 (the ANN systematically
predicts ~3× less visibility than the "wrong" library), and the Ar-lib/175nm-ANN
median ratio is 3.30 (ANN predicts ~3× more). A per-entry discriminator (which
ANN flavor is closer in |log-ratio|, on the support common to all four pairs)
confirms this isn't a tail artifact: **95.8%** of Ar-library entries sit closer
to the 128 nm ANN, and **94.1%** of Xe-library entries sit closer to the 175 nm
ANN. This is a strong internal-consistency result: two independently-built
official artifacts (a Geant4 photon-transport voxel table and a trained neural
net) agree with each other at the ~15-20% level (16-84% band) when compared
doping-to-doping, and disagree by factors of 3-9× when cross-compared — so
"which ANN is the library consistent with" has an unambiguous answer, not a
close call.

**One caveat, found rather than assumed**: the per-channel breakdown (native
channel index, `pdvd/photlib/pics/lib_ann_perchannel.png`) shows the diagonal
ratio (Ar/128, Xe/175) sitting flat at ≈1.0 across *all 40 channels*, including
the three official "Ar-blind" channels (16, 29, 39 — `eff_Ar=0`, `eff_Xe>0` in
`pdvd-photlib-chanmap.json`). The raw visibility tables do **not** hard-zero
those channels under Ar — they carry ordinary nonzero geometric visibility in
both the Ar library and the 128 nm ANN. This means the `eff_Ar=0` figure is a
**downstream wavelength-dependent detection-efficiency factor** (WLS/coating
response), applied on top of the visibility (same split as `VUVEfficiency` in
§6 of `match/docs/semi-analytical-model.md`), not something baked into the
transport-only visibility numbers compared here. So the Ar-blind channels are
*not* usable as an extra doping discriminator at this (visibility-only) level
— the global corr/ratio/RMS table above is the actual evidence.

Plots (`pdvd/photlib/pics/`):
- `lib_vs_ann_scatter.png` — 2×2 log-log hexbin grid, one panel per (library,
  ANN) pair; the two diagonal panels (Ar-vs-128nm, Xe-vs-175nm) hug `y=x`
  tightly, the off-diagonal panels show a clear, systematic offset/curvature.
- `lib_vs_ann_ratio.png` — log10(ANN/library) histograms per library; each
  library's own-doping curve is a narrow peak centered at 0, the other-doping
  curve is a wide, offset hump.
- `lib_ann_perchannel.png` — median ANN/library ratio vs. native channel index;
  the two diagonal curves sit flat at 1 across all 40 channels while the
  off-diagonal curves run 2-6× away, with no distinguishing feature at the
  Ar-blind channels (dashed lines) beyond ordinary channel-to-channel scatter.

Reproduce: `/home/xqian/tmp/tfvenv/bin/python pdvd/photlib/compare_lib_ann.py`
(reuses `work/photlib_vis_{Ar,Xe}.npy` and `work/ann_v4_at_voxels.npy` from the
existing pipeline; only samples the v4 175 nm ANN fresh, caching it to
`work/ann_v4_175nm_at_voxels.npy`). Full numbers: `work/lib_ann_compare.json`.

---

## 2. What we built for WCT: two backends, both derived from the ANN

`QLMatching` has a `light_model` knob (`"semi"` default, `"library"` opt-in;
`match/inc/WireCellMatch/QLMatching.h:637`) selecting between two visibility
backends. **Both PDVD backends are ultimately calibrated against the
deep-learning model** — WCT does not run the ANN or the voxel table live.

### 2a. `PhotonLibraryModel` — the gridded library backend (row 4)

`match/src/PhotonLibraryModel.{h,cxx}` (commit `f52c4a47`): loads a JSON meta
file (`origin_cm`, `step_cm`, `n`, `nchan`) plus a float32 `.npy` array of shape
`[nx,ny,nz,nchan]`, and returns per-OpDet visibilities at an arbitrary 3D point
by **trilinear interpolation**, clamping at the grid boundary. It has no
knowledge of "ANN" or "voxel table" — it is a generic gridded-lookup engine.

What feeds it, for PDVD: `pdvd/photlib/sample_ann.py` samples the **v5 ANN**
(both wavelength variants) on a **10 cm grid** over the active volume
(71×69×33 nodes × 40 channels, ≈26 MB per file), written by
`export_wct_photlib.py` to:

```
wire-cell-data/pdvd/photodet/pdvd-photlib-vis-v5-128nm.{json,npy}   # Argon
wire-cell-data/pdvd/photodet/pdvd-photlib-vis-v5-175nm.{json,npy}   # Xenon
```

Each meta JSON carries 32 random self-check points with python-computed
trilinear values, verified in C++ to ≤1e-6
(`match/test/doctest_photonlibrarymodel.cxx`, `PHOTLIB_META=...` env var).

In library mode, `QLMatching` (`QLMatching.cxx:1357`) takes the grid value as
**total photon arrival** — the reflected term is forced to 0, and there is
**no same-TPC x-sign gate** (unlike the semi path): the grid already encodes the
detector's own optical shadowing, including the cathode's double-sided
Arapucas (visibility symmetric across x=0) and cathode opacity for
membrane/PMT channels. The `semimodel_file` is still loaded in this mode
purely to supply the per-OpDet type/position table used by masks.

### 2b. `SemiAnalyticalModel` — the calibrated fallback (row 5)

`match/src/SemiAnalyticalModel.cxx` is a dependency-free port of LArSoft's
`larsim/PhotonPropagation/SemiAnalyticalModel` — closed-form solid-angle
(rectangle for Arapucas, dome for PMTs) times an exponential VUV absorption
factor, corrected by a small fitted Gaisser–Hillas table indexed by emission
angle, plus a reflected-light branch off the cathode. See
`match/docs/semi-analytical-model.md` for the full mechanism (no ML; direct
analytic geometry + small fitted correction tables baked into a JSON).

For PDVD, `pdvd/photlib/fit_semi_analytical.py` fit those correction tables
**against the v5 128 nm ANN samples** (not against data, not against the voxel
table) — i.e. row 5 is a compact closed-form re-expression of row 1, produced
by us, not an official LArSoft PDVD product. Fit quality, pred/ANN ratio
16/50/84%:

| PD group | ratio 16/50/84% | usable in current C++ port? |
|---|---|---|
| cathode XAs (8) | 0.88 / 0.99 / 1.13 | yes |
| bottom PMTs (16) | 0.75 / 1.01 / 1.46 | yes |
| TCO PMTs (8) | 0.68 / 1.03 / 1.80 | poorly — table kept, suggest masking |
| membrane XAs (8) | 0.82 / 0.97 / 1.32 | **no** — port's `cosine=|Δx|/d` (orientation-0) is wrong/divergent for these y-normal walls; a correct table exists but the lateral-cosine branch isn't ported |

Output: `wire-cell-data/pdvd/photodet/semi-analytical-pdvd.json`, in the same
schema `QLMatching` already loads for SBND/PDHD.

**Verdict from that fit**: a PDVD semi-analytical model is *possible* but
second-best — it needs two more C++ ports (both-sides cathode gate for the
double-sided XAs, lateral-cosine branch for membrane walls) and still carries
30–80% point-level spread for PMTs, versus the library's direct representation
of the official model. This is why the library backend, not the semi fit, is
PDVD's production choice.

---

## 3. What `QLMatching` actually runs today, per detector

- **PDVD** (`cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`):
  `light_model: 'library'`, `photon_library_file:
  'pdvd/photodet/pdvd-photlib-vis-v5-175nm.json'` — the **Xenon/175 nm**
  gridded library (commit `0adb15fa`, 2026-07-11). `semimodel_file` still
  points at `semi-analytical-pdvd.json` (used only for the OpDet table in
  library mode). The **Argon/128 nm** grid file exists and is drop-in
  swappable (`photon_library_file` → the 128nm path) but is not the default.
- **PDHD / SBND**: neither config sets `light_model`, so both run the C++
  default `'semi'` — SBND's own LArSoft-fit semi-analytical model
  (`sbnd/photodet/semi-analytical-sbnd.json`), unrelated to PDVD's ANN-derived
  fit. PDHD's nominal *simulation* actually uses the semi-analytical model
  (`PDFastSimPAR`), so the WCT default lines up with what PDHD's own sim ran.

### Why Xe/175 nm, not Ar/128 nm, for PDVD data

The Ar-vs-Xe choice was decided from data, not from the library files
themselves: several "Ar-blind" channels (13, 29, 32, 39 — dark under the
Ar/128 nm efficiency map) respond in real flashes at the same level as their
live peers; the 175 nm library wins a gold-pair shape A/B 56/80; a mixture
scan (α·128nm + (1−α)·175nm) is monotonic with no mixed optimum. Channels
13/29/39 were unmasked when the library switched to Xe; channel 32 stays
masked pending an official Xe efficiency (data show it live too — flagged in
the cfg comments). Detail: `pdvd-questions-dune.md` §3,
`pdvd/ql_light_calib/ablib_gold.py`.

### Geometry correctness note (affects both backends' inputs)

The QLMatching active-volume box that gates which charge points get summed
into the light prediction had a bug independent of which photon model is
selected: `compute_geometry` used only one anode face per drift side; PDVD's
two CRP faces split Y into disjoint halves (unlike PDHD/SBND, where both faces
span the full Y). Fixed in commit `565ccd62` via a PDVD-only `tpc_extra_faces`
knob (empty/no-op elsewhere, so PDHD/SBND stay byte-identical). This changed
predicted-PE values substantially (990/3795 bundles >5% in one event) and
moved the QtoL calibration (0.082 → 0.070); it did not change which photon
model is used.

---

## 4. Answering the original questions directly

- **"Are there Xe and Ar deep-learning libraries?"** Yes — `PDFastSimANN`
  official Ar/128nm and Xe/175nm graphs (§1). Data favor Xe; that's the PDVD
  default (§3).
- **"Is there also a table-based library from original LArSoft?"** Yes — the
  disabled `PDFastSimPVS` voxel library, v4-only, 25³ voxels (§1, row 3). It
  was used only to cross-check the ANN and is not wired into `QLMatching`.
  Each doping's library is quantitatively consistent with the matching ANN
  doping and inconsistent with the other one (corr 0.99 vs 0.93, ratio≈1 vs
  0.3-3.3×, 94-96% of entries closer to their own doping — §1 row 3a) — the two
  independent official artifacts agree with each other, which is a useful
  cross-check of the whole Ar/Xe modeling but does not by itself say anything
  about which doping matches *data* (that call is made in §3 from data tests).
- **"What about the semi-analytical calibration?"** Done, exists as the
  configured fallback (`semi-analytical-pdvd.json`), fit against the ANN, but
  has known gaps (membrane XAs, PMT spread) that keep it second-best to the
  gridded library (§2b).
- **"Which models are integrated into the QLMatching chain so far?"** Only the
  **gridded-library backend** (`PhotonLibraryModel`, resampled from the v5 ANN)
  and the **semi-analytical fallback** (calibrated to the same ANN) are wired
  into WCT. Production PDVD uses library/Xe-175nm; semi is the fallback;
  library/Ar-128nm is available but not selected; the LArSoft voxel table
  itself was never wired in.

---

## 5. Implementation audit: channel mapping and visibility correctness in QLMatching

Question: does `QLMatching`'s gridded-library backend correctly line up
**channel index** with **physical OpDet position**, and is the **visibility**
semantics (what a raw library value means, and how it becomes predicted PE)
sound? Checked directly against the code (`match/src/QLMatching.cxx`,
`match/inc/WireCellMatch/PhotonLibraryModel.h`) and against the actual data
files, not just the design docs.

### 5.1 Is there a channel-mapping bug in the shipped PDVD config? — No, by four independent cross-checks

| # | check | result |
|---|---|---|
| 1 | ANN npz self-map: `wct_opdet[k]` in `work/ann_vis_v5_175nm.npz` vs identity | **36/36 live channels** `wct_opdet[k]==k`; dead = exactly {24,27,28,34} |
| 2 | `semi-analytical-pdvd.json`'s `OpDets[k]` position vs the ANN's `chan_pos_mm[k]` | **40/40 channels** agree to <1 mm (exact after cm/mm conversion) |
| 3 | cfg `VUVEfficiency[k]` (`qlmatching.jsonnet`) vs official `eff_Xe` (`pdvd-photlib-chanmap.json`, keyed by `wct_opdet`) | **40/40 channels** agree exactly |
| 4 | `OpDets[k].type` (XA=0/PMT=1) vs ANN node name vs chanmap `pd_type` | **40/40 channels** agree |

`export_wct_photlib.py` copies the sampled ANN array to the shipped `.npy`
**verbatim** — it never even reads the npz's `wct_opdet`/`chan_node`/
`chan_pos_mm` fields, so there is no reindexing step to get wrong. The reason
check #1 comes out identity is that `sample_ann.py`'s v5 channel-to-GDML
assignment (bijective `linear_sum_assignment` against the raw-data opdet
table) already produces that order at generation time. **Conclusion: no
channel-mapping bug exists in the currently shipped PDVD library/config.**

### 5.2 But the code has no way to *catch* one — a structural fragility worth flagging

Reading `QLMatching.cxx`'s indexing directly: the library channel axis,
`m_opdets`, `VUVEfficiency`/`VISEfficiency`, the opflash measured-PE vector,
and `ch_mask`/`opdet_mask` are all addressed by **one shared, unchecked 0..39
index**:

- The prediction loop uses a single loop variable for everything:
  ```cpp
  // QLMatching.cxx:1373-1381
  for (std::size_t idet = 0; idet < nopdets; ++idet) {
      if (flash_opdet_mask.at(idet) == 0) continue;
      const auto dir_vis = direct_visibilities.at(idet);   // library channel idet
      const auto ref_vis = reflected_visibilities.at(idet);
      const auto dir_eff = m_VUVEfficiency.at(idet);        // config array idet
      const auto ref_eff = m_VISEfficiency.at(idet);
      pred_flash.at(idet) += qw * m_QtoL * dir_vis * dir_eff + qw * m_QtoL * ref_vis * ref_eff;
  }
  ```
- The fit step pairs measured and predicted PE the same way:
  ```cpp
  // QLMatching.cxx:1634-1647 (paraphrased indices)
  const int opdet_idx = run.opdet_idx_v.at(j);
  const double pe      = flash->get_PE(opdet_idx);       // opflash channel opdet_idx
  const double pred_pe = pred_flash.at(opdet_idx);        // same idx into pred_flash
  ```
- `m_opdets` itself is loaded in pure JSON-array order with **no id field
  read** (`QLMatching.cxx:414-424`) — position `i` in the `OpDets` array of
  `semimodel_file` *is* "OpDet `i`", by convention only.
- **The only cross-check anywhere is a scalar count**, `QLMatching.cxx:438-441`:
  ```cpp
  if (m_lib_model->nchan() != m_opdets.size())
      raise<ValueError>("QLMatching: photon library nchan %d != nopdets %d", ...);
  ```
  This catches a *length* mismatch. It cannot catch an *order* mismatch (e.g.
  two channels transposed, or a future re-export using a different channel
  convention) — the file would load, the count would match, and QLMatching
  would silently mispredict with no error and no log line.
- This isn't a trivial oversight to fix, either: `PhotonLibraryModel`'s file
  format (`PhotonLibraryModel.h:8-12`: meta = `origin_cm`/`step_cm`/`n`/
  `nchan`/`vis_npy`; the `.npy` itself is a bare `[nx,ny,nz,nchan]` float
  grid) carries **no per-channel position or id at all**. There is nothing in
  the shipped file for a loader to cross-check `m_opdets` against, even in
  principle — the export script would need to start embedding one.

**Two smaller, pre-existing findings** (noted, not fixed, per the usual
"report don't fix in this pass" rule):
- `VUVEfficiency`/`VISEfficiency` get **no config-time length check**
  (`QLMatching.cxx:368-374`), unlike `measured_pe_scale`, which explicitly
  validates `size == nchan` and raises (`QLMatching.cxx:301-306`). A
  wrong-length efficiency array either throws later at `.at(idet)` (loud
  failure) or, if merely long enough, silently mispairs values (quiet
  failure) — the two length-checked and unchecked config arrays sit right
  next to each other in the same file with different rigor.
- `ch_mask` entries index `run.opdet_mask` via a raw `operator[]`
  (`QLMatching.cxx:843`), not `.at()` — an out-of-range value in a config's
  `ch_mask` array is an out-of-bounds write, not a caught error.

**Bottom line**: today's PDVD channel mapping is correct, verified four
independent ways above — but it is correct *by convention across
independently-authored files*, not because the code enforces it. If this
becomes a maintained/re-derived pipeline (new ANN version, regenerated
`pdvd-opch-map.json`, a manually-edited efficiency array), nothing would flag
a silent transposition. If it's ever worth hardening: embed per-channel
position/id in the exported library meta JSON and cross-check it against
`m_opdets` at load time, and give `VUVEfficiency`/`VISEfficiency` the same
length check `measured_pe_scale` already has. Not implemented here — this
section is diagnosis only.

### 5.3 The "visibility" semantics question — resolved, and it's a consistent design

The doping-consistency check in §1 row 3a left an unexplained caveat: the
official "Ar-blind" channels (16/29/39 — `eff_Ar=0`, `eff_Xe>0`) showed
*ordinary*, unremarkable ratios in the raw ANN/library comparison instead of
a hard zero under Ar. That now has a clean explanation worth stating
explicitly: **the raw library/ANN value is pure optical/geometric transport
visibility**; the wavelength/coating-dependent detection response is a
*separate* multiplicative factor, applied downstream by
`VUVEfficiency`/`VISEfficiency` in the same accumulation line quoted above
(`QLMatching.cxx:1379-1380`). That is architecturally identical to the
semi-analytical model's own visibility × efficiency split (`match/docs/
semi-analytical-model.md` §6). So this isn't a bug — it's confirmation that
the library and semi backends compose visibility and detection efficiency the
same way, resolving the earlier open caveat rather than leaving it dangling.

One more visibility-handling point re-verified, not just re-asserted: in
library mode `doReflectedLight` is forced to 0 and there is no same-TPC
x-sign gate (`QLMatching.cxx:1357-1362`), because the ANN grid already
encodes total photon arrival including the detector's own shadowing
(double-sided cathode XAs, cathode opacity for membrane/PMT channels). No
counter-evidence to that design choice turned up in this audit.

---

## Cross-references

- `pdvd/docs/pdvd-photon-model.md` — the original investigation (2026-07-09):
  full geometry cross-matching, ANN/voxel-library consistency numbers,
  reproduction commands.
- `../pdvd-qlmatching.md` — QLMatching chain design and knob rationale
  (offset calibration, QtoL history, PE-scale studies).
- `../pdvd-questions-dune.md` §3 — the Ar/Xe data-verdict tests in full.
- toolkit `match/docs/semi-analytical-model.md` — mechanism of
  `SemiAnalyticalModel` in general (SBND-oriented, but the math is shared).
- toolkit `match/docs/qlmatching-code.md` §2a — full C++ knob reference
  including `light_model`/`photon_library_file`.
- Commits: `f52c4a47` (gridded library backend), `e06ea900` (PDVD
  vertical-drift knobs), `8a765d5e` (per-crate trigger offsets), `0adb15fa`
  (Xe/175nm default switch), `565ccd62` (active-volume Y-truncation fix).

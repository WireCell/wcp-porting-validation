# PDVD doc 33 — the fiducial volume, stage by stage, against SBND

**Status (2026-09-04).** Summary and review only: **no code, no config, no arm,
no A/B gate is owed and none is claimed.** Everything below is read out of the
source, out of the compiled configs, and out of products already on disk (the
120-event PR arm `d28dlfp`, and an SBND `ncpi0` PR arm). It answers the owner's
request for a side-by-side map of the fiducial volume through Clustering, Q/L
matching, the cosmic taggers and PR, and it reviews the FV half of doc 32 (§6,
§7 R3–R6) against measurement.

Two owner steers are recorded here and honoured throughout:

- **Do not shrink the fiducial volume to hide the near-CRP dQ/dx behaviour.**
  §8 measures what that region actually looks like and §9 R1 is written the way
  the owner asked: understand it first, do not cut it away.
- **The gaps matter.** SBND has exactly one structural gap (the CPA). PDVD has
  the cathode *and* the seams between neighbouring CRPs. §5 inventories them
  geometrically and measures how wide each one is to the reconstruction, and §9
  R2 proposes how a fiducial volume should carry them.

Doc 32 is being extended by another session as this is written, so nothing here
was added to it; this file stands alone and cross-references it.

---

## 0. Repro

```bash
WCPI=/home/xqian/toolkit-dev/wcp-porting-img
cd $WCPI/pdvd

# (1) the stage census: STM outcomes, march truncation, margin arithmetic, holes
python3 docs/nf_sp_img_clus/scripts/fv_stage_census.py work/*_d28dlfp

# (2) the gap and drift profiles, PDVD against SBND
python3 docs/nf_sp_img_clus/scripts/fv_gap_profile.py \
    $(ls -d work/*_d28dlfp | head -40 | sed 's/^/PDVD:/') \
    $(ls -d ../../toolkit/sbnd_xin/work-ncpi0-doc25_r2post5/pr_evt* | head -40 | sed 's/^/SBND:/')

# (3) the geometry itself -- every PR log carries all 16 (PDVD) / 2 (SBND)
#     sensitive boxes, in mm, straight from AnodePlane
grep sensvol work/039252_5_d28dlfp/wct_pr_039252_5.log
grep sensvol ../../toolkit/sbnd_xin/work-ncpi0-doc25_r2post5/pr_evt105946/wct_pr_evt105946.log
```

Arms read (read-only; nothing under `work/`, `sweep/` or `abtest/snap/` was
written): `pdvd/work/*_d28dlfp` (120 events, the post-DL-flip production arm of
doc 28 §27), `pdvd/work/039252_2_d32r3base` and `039349_14_d31r6e2e` (the doc-32
events), `sbnd_xin/work-ncpi0-doc25_r2post5/pr_evt*` (19 events with a Bee zip).

---

## 1. The one-page answer

There is no single "fiducial volume" in either detector. There are **five**
distinct notions, built from three different primitives, and they are consulted
at different stages:

| # | volume | primitive | who reads it |
|---|---|---|---|
| V1 | the **sensitive union** | `DetectorVolumes::contained()`, the union of the per-(apa,face) `IAnodeFace::sensitive()` boxes | `FiducialUtils` (so: STM internals, TGM dead/SP walks, vertex FV bonus, segment-endpoint rejection), and `TrackFitting`'s end-trim via `contained_by()` |
| V2 | the **clustering FV metadata** | `FV_*` + `FV_*_margin` keys on the `DetectorVolumes` blocks | `clustering_separate`, `clustering_examine_x_boundary`, `clustering_neutrino`, `SimpleClusGeomHelper` |
| V3 | the **Q/L active volume** | union of `m_dv->inner_bounds()` over the drift side's anodes + signed cushions | `QLMatching` (PE inclusion, containment, boundary flags) |
| V4 | the **structure-exclusion fiducial** | `CompositeFiducial` OR of boxes | SBND only: the CPA lattice, consumed by `QLMatching` |
| V5 | the **tagger box** | one `BoxFiducial` spanning both drift volumes + `fv_tolerance` | TGM, FC, STM containment, and — with the three `*_consistent_fv` flags on — the neutrino/cosmic/nue/single-photon containment |

**The headline comparison.** The two detectors agree on the *design* at every
stage: same primitives, same routing flags, same margins — with exactly two
exceptions, and both are PDVD-side:

1. **`tgm_fv_x_margin` is 30 cm on PDVD and 2.5 cm on SBND.** Every other tagger
   margin is identical (`y` 3, `z_max` 5, `z_max_interior` 3, `z_min` 3). This
   is the single largest numerical difference in the whole map, and §7 measures
   what it costs.
2. **PDVD's V1 has four gap families where SBND's has one**, and no stage
   except V1 itself knows about any of them (§5, §6).

A third difference is a stage the comparison makes visible: PDVD's V2 is inset
**15 cm** in y and z where SBND's is inset **1 cm** (§3.2). That one is
deliberate on PDVD and probably a defect on SBND.

---

## 2. V1 — the sensitive union, i.e. the geometry both detectors start from

Every PR log prints the boxes at `AnodePlane` construction. PDVD, 16 faces
(mm, `[(xmin ymin zmin) --> (xmax ymax zmax)]`):

| face | x | y | z |
|---|---|---|---|
| anode0 f0 | −3399.1 … −30 | −1684.9 … −6.1 | 8.128 … 1495.2 |
| anode0 f1 | −3399.1 … −30 | −3363.9 … −1685.1 | 8.128 … 1495.2 |
| anode1 f0/f1 | −3399.1 … −30 | (same two y tiles) | 1497.8 … 2984.35 |
| anode2, anode3 | −3399.1 … −30 | +6.1 … +1684.9 and +1685.1 … +3363.9 | as anode0 / anode1 |
| anode4 … anode7 | +30 … +3399.1 | the same four y tiles | 8.128 … 1496 and 1497 … 2984.35 |

SBND, 2 faces:

| face | x | y | z |
|---|---|---|---|
| apa0 f0 | −2014.5 … −4.5 | −1999.65 … +1999.65 | 0 … 5010 |
| apa1 f1 | +4.5 … +2014.5 | −1999.65 … +1999.65 | 0 … 5010 |

So the union's outer envelope is PDVD |x| ≤ 339.91, |y| ≤ 336.39,
z ∈ [0.813, 298.435] cm; SBND |x| ≤ 201.45, |y| ≤ 199.965, z ∈ [0, 501.0] cm.

**SBND's union is two boxes with one gap between them. PDVD's is sixteen boxes
with four gap families.** That is the structural difference the rest of this
document keeps running into.

---

## 3. Stage by stage

### 3.1 The primitives that exist

| component | shape | PDVD | SBND |
|---|---|---|---|
| `BoxFiducial` | one axis-aligned box | `pdvd_pr_fv` | `sbnd_pr_fv` |
| `DetectorVolumes` (as `IFiducial`) | the exact sensitive union, gaps included | what `MakeFiducialUtils` receives | same |
| `CompositeFiducial` | and/or/nand/nor of children | **none** | the CPA exclusion (26 boxes OR'd) |
| `EnvFiducial` | one bbox over all faces' `sensitive()` | instantiated by no config | instantiated by no config |
| `PolyFiducial` | stack of polygonal slabs (the `ToyFiducial` port) | used by no config | used by no config |

### 3.2 Clustering (V2) — the `FV_*` metadata

| | PDVD (`protodunevd/clus.jsonnet:98-151`) | SBND (`sbnd/clus.jsonnet:126-165`) |
|---|---|---|
| `overall` FV_x | ±341.55 cm — the **CRP centreline**; the sensitive volume starts 16.4 mm inward at the shield plane (±339.91), so the window reaches 1.64 cm into the CRP structure itself | ±201.05 cm — W plane (±202.05) inset 1 cm, i.e. 0.4 cm **inside** the sensitive edge |
| `overall` FV_y | ±321.4 cm = active ±336.4 **inset 15 cm** | ±199.312 cm = wires bbox inset 1 cm |
| `overall` FV_z | [15.05, 284.25] cm = active **inset 15 cm** | [0.85, 500.15] cm = wires bbox inset 1 cm |
| margins | x 2, y 2.5, z 3 cm | x 2, y 2.5, z 3 cm (identical) |
| per-face blocks | **FV_x only** (bottom [−339.91, −3.0], top [3.0, 339.91]); y/z inherited from `overall` | per-TPC FV_x ([−201.05, −2.5] / [2.5, 201.05]); y/z explicitly copied from `overall` |
| knows about CRP tiling? | **no** — every one of the 16 faces carries the same y/z window | n/a (one tile) |

Two things worth stating:

- The 15 cm y/z inset is **deliberate and correct** on PDVD. It restores the
  prototype's "on the surface = within ~15 cm of a wall" operating point that
  every `JudgeSeparateDec_2` threshold was tuned at; with the exact active
  envelope as the FV the decision is unsatisfiable
  (`clus/docs/clustering-separate-fv.md`). SBND's 1 cm inset is that same
  blindness left in place — consistent with doc 96's finding that SBND's dec2
  never fires on in-time clusters. **SBND's business, flagged not fixed here.**
- PDVD's `overall` FV_x is the CRP centreline, not the LAr boundary:
  341.55 − 339.91 = 1.64 cm = the `apa_plane` = 16.4 mm shield-plane offset the
  geometry is built with. So the window's last 1.64 cm at each end lies inside
  the CRP electrode stack rather than in liquid argon — the two numbers describe
  different surfaces, which is the substance of R7; nothing here says charge is
  reconstructed there. Because the per-face x-windows disagree (bottom vs top),
  `select_scope_fv` falls back to `overall` for an all-anode scope, so an
  all-detector clustering pass uses one x-window spanning both drifts, the 6 cm
  cathode and both shield-to-centreline offsets.
- `clustering_examine_x_boundary` raises on differing `FV_x` metadata
  (`:92-103`); `allow_mixed_faces` waives the same-*face* check, not this one.
  That is the structural reason the clustering stage cannot be given one
  whole-detector x window.

### 3.3 Q/L matching (V3, V4)

`QLMatching::compute_geometry` builds the active volume as the **union of
`m_dv->inner_bounds()` over the drift side's anodes**, then works in a per-TPC
drift coordinate u = s·(x − anode_x) with signed cushions:

| knob | meaning | PDVD | SBND |
|---|---|---|---|
| `anode_ext1` | PE-inclusion window edge below the anode | −2 cm (C++ default) | −2 cm |
| `anode_ext2` | anode flag-window outer edge | 4 cm (default) | 4 cm |
| `anode_ext1_margin` | extra slack below `anode_ext1`, for containment **and** the flag window | **2 cm** (driver; C++ default 1) | 1 cm |
| `cathode_ext1` | how far past the cathode a drift end may reach and still count contained | 1.2 cm (default) | 1.2 cm |
| `cathode_ext2` | cathode flag-window inner edge | −2 cm (explicit; a −12 cm experiment was reverted) | −2 cm |
| `y_cushion`, `z_cushion` | signed inward shift of each transverse edge | 0, 0 | 0, 0 |
| `cathode_fiducial` | CPA structure-exclusion `CompositeFiducial` | **'' (none)** | `cpa-exclusion`, 26 boxes, 0.5 cm dilation |
| face union | PDVD's two faces split y, so a second face is unioned in (`m_tpc_extra_faces`) | yes | not needed |

So the Q/L stage differs in exactly two ways: PDVD pulls the anode floor 1 cm
further (a cathode-crosser fix, `pdvd/docs/qlmatch/16_*`), and **PDVD has no
structure-exclusion fiducial at all**. SBND's exists because its CPA is not a
flat sheet — foil pads reach 0.6 cm into the drift, the steel tube lattice
2.7 cm, the knuckle joints 4.1 cm — and a cluster end inside that structure is
not a physical track end. PDVD's cathode is a 6 cm slab and its CRP seams are
listed in §5; none of them is represented anywhere in the Q/L stage.

### 3.4 The cosmic taggers (V5)

| | PDVD (`protodunevd/pr.jsonnet:1153-1163`) | SBND (`sbnd/clus.jsonnet:1948-1957`) |
|---|---|---|
| component | `BoxFiducial pdvd_pr_fv` | `BoxFiducial sbnd_pr_fv` |
| box | x ±339.91, y ±336.4, z [0.05, 299.25] cm | x ±201.05, y ±199.312, z [0.85, 500.15] cm |
| spans | both drifts + the 6 cm cathode slab | both TPCs + the 0.9 cm CPA |
| `tgm_fv_x_margin` | **30 cm** | **2.5 cm** |
| `tgm_fv_y_margin` | 3 cm | 3 cm |
| `tgm_fv_zmax_margin` / `_interior` | 5 / 3 cm | 5 / 3 cm |
| z_min margin | 3 cm | 3 cm |
| effective volume | \|x\| ≤ **309.91**, \|y\| ≤ 333.4, z ∈ [3.05, 294.25] | \|x\| ≤ 198.55, \|y\| ≤ 196.312, z ∈ [3.85, 495.15] |
| `stm_consistent_fv` | true | true |

Both are deliberately **one box across both drift volumes**, so a cathode
crosser is not read as an exiter at x = 0 — the default `fiducial=dv` cannot
serve there. The design is identical; only the x margin differs, and it differs
by 27.5 cm.

### 3.5 PR / neutrino (V5 again, plus V1 underneath)

The three routing flags are **on in production on both detectors** (verified
from the arm logs, not from the config defaults):

| flag | what it routes | PDVD | SBND |
|---|---|---|---|
| `neutrino_consistent_fv` | `match_isFC` | on | on |
| `cosmic_consistent_fv` | cosmic tagger containment | on | on |
| `nue_sp_consistent_fv` | nue / single-photon containment | on | on |
| `stm_consistent_fv` | STM's `cluster_fc_check` gate | on | on |

**What is left on V1, on both detectors**, because it has no routing at all:

- `MakeFiducialUtils` receives `fiducial=dv` (`pr.jsonnet:1098,1100`;
  `common/clus.jsonnet:1522`), so **every** `FiducialUtils` call sees the union;
- `TaggerCheckSTM` internals — `find_first_kink` (:1580, :1687), `detect_proton`,
  `eval_stm`, the endpoint tests at :3641-3642/:3668 and `check_dead_volume`
  (:3674);
- `TaggerCheckTGM`'s dead / signal-processing walks;
- `NeutrinoPatternBase.cxx:3043` — rejects a segment whose first or last fit
  point is outside;
- `NeutrinoVertexFinder` — the FV bonus that helps decide **which vertex wins**
  (:1256, :2184, :4450, :5172). No routed alternative exists at these sites;
- `TrackFitting`'s end-trim loop, through `m_dv->contained_by()` directly.

None of those calls passes a tolerance vector, so on both detectors they test
the raw union with **no margin** — a 30 cm disagreement with the tagger box on
PDVD, 2.5 cm on SBND.

---

## 4. The summary table

| stage | PDVD | SBND | same? |
|---|---|---|---|
| sensitive union (V1) | 16 boxes; \|x\| ≤ 339.91, \|y\| ≤ 336.39, z ∈ [0.813, 298.435]; **4 gap families** | 2 boxes; \|x\| ≤ 201.45, \|y\| ≤ 199.965, z ∈ [0, 501.0]; **1 gap** | structure differs |
| clustering (V2) | x ±341.55 (overhangs), y/z inset **15 cm**; margins 2 / 2.5 / 3 | x ±201.05, y/z inset 1 cm; margins 2 / 2.5 / 3 | insets differ |
| Q/L active (V3) | union of both faces' inner bounds; cushions y 0, z 0; anode slack **2 cm** | one face; cushions y 0, z 0; anode slack 1 cm | ~same |
| Q/L structure exclusion (V4) | **none** | CPA `CompositeFiducial`, 26 boxes, +0.5 cm | **PDVD missing** |
| cosmic taggers (V5) | box + margins x **30**, y 3, z 5/3 | box + margins x **2.5**, y 3, z 5/3 | x differs |
| PR routing | all four `*_consistent_fv` on | all four on | same |
| everything on `FiducialUtils` | the holed union, no margin | the holed union, no margin | same design, worse geometry on PDVD |

---

## 5. The gap inventory — and how wide each gap really is

Geometric gaps, from the `sensvol` boxes of §2:

| gap | PDVD | SBND |
|---|---|---|
| cathode | **\|x\| < 3.0 cm** (6.0 cm slab) | \|x\| < 0.45 cm (0.9 cm slab) |
| central CRP seam in y | **\|y\| < 0.61 cm** (1.22 cm), full drift and full length | — |
| CRP-to-CRP seam in y | **\|y\| = 168.49 … 168.51 cm** (0.02 cm), two of them | — |
| CRP-to-CRP seam in z | **z = 1495.2 … 1497.8 mm** bottom (0.26 cm), **1496 … 1497 mm** top (0.10 cm) | — |
| outer z start / end | 0.813 / 298.435 cm | 0 / 501.0 cm |

The geometric width is not the width that matters. Imaged-point density across
each boundary, 4.8 M points over 40 events (density relative to the far field;
`fv_gap_profile.py`):

| boundary | profile |
|---|---|
| central y seam | 0.51 at ∓0.75 cm, **0.05 / 0.01 across \|y\| < 0.5**, 0.49 at +0.75, full by ±1.2 cm |
| CRP y seam at 168.5 | shallow: **0.60 – 0.79 over roughly ±1 cm**, never empty |
| CRP z seam at 149.65 | no clean structure above the noise (0.38 – 1.9 bin to bin) |
| cathode, \|x\| inward from 3.0 | **0.22 – 0.31 inside the union's hole**, 0.49 at \|x\| ≈ 3.5, full by \|x\| ≈ 4.5 |
| SBND CPA, \|x\| inward from 0.45 | **exactly 0 inside the hole**, 0.30 at 0.95, full by ≈ 5 cm |
| SBND mid-y (control, no structure) | flat, 0.92 – 1.09 |

Three conclusions:

1. **The central y seam is real and the union has it about right** (0.61 cm
   against a measured half-width of ~0.5 cm, with partial suppression to 1 cm).
2. **The CRP y seam is real but far wider than the geometry says** — 0.02 cm of
   geometric gap, ~±1 cm of reduced efficiency. A fiducial volume built from
   `sensitive()` alone cannot see it.
3. **The cathode hole excludes real charge on PDVD and does not on SBND.**
   PDVD reconstructs 20–30 % of nominal density *inside* `|x| < 3` — 21 582
   imaged points and 85 fitted `track_fit` points over the 120-event arm — every
   one of which `DetectorVolumes::contained()` calls "outside the detector".
   SBND's 0.45 cm hole contains literally zero points. The SBND control shows
   the instrument is not manufacturing the dip.

---

## 6. What the FV work is worth, measured

All numbers from `fv_stage_census.py` over the 120-event `d28dlfp` arm
(5222 clusters longer than 10 cm, 10 444 PCA ends).

### 6.1 Why STM declines to fit

| reason (distinct clusters, summed over 120 events) | n |
|---|---|
| STM evaluated | 5543 |
| no STM fit recorded | 3360 |
| — of which **fully contained (Mid Point A)** | **2942 (87.6 %)** |
| — no `steiner_pc` | 165 |
| — round-1 forward fit ≤ 3 points | 137 |
| — single exit with a mid-track kink (Mid Point B) | 84 |
| — ≥ 3 candidate exits (Mid Point C) | 32 |
| already TGM | 1592 |
| STM = 1 | 663 |

Containment is not one reason among several; it is **seven in eight** of the
no-fit population. That is the strongest argument in this document for taking
the FV seriously — and note it survives regardless of which volume is used,
because this gate is already on the tagger box (`stm_consistent_fv`).

### 6.2 The union's holes, where they actually bite

Point containment: **not at all**. Over the arm, 0 of 832 391 `stm_fit` points
and 120 of 121 267 `track_fit` points (0.1 %) lie in a hole.

Marching: **a little**. `check_dead_volume` and `check_signal_processing` walk
outward on `while (inside_fiducial_volume(temp_p))`, so a march that would cross
the cathode or the y seam stops there:

| | |
|---|---|
| ends whose outward march is truncated by a hole | 2329 of 10 444 (22.3 %) — cathode 1531, y seam 798 |
| how much earlier it stops | p50 229 cm, p90 378 cm |
| truncated **inside the ≤ 5-step early-return window**, where a verdict can change | **387 (3.7 %)**, 119 of them on fully-contained clusters |

Both walks return "not an exit" as soon as `cut_value` (4 / 5) live steps
accumulate, so only the 3.7 % can move anything — and the bias has a direction:
truncation means fewer points, which means "not an exit", which means **more
"fully contained"**.

### 6.3 The 30 cm x margin

| | |
|---|---|
| ends inside the FV at 2.5 cm and outside at 30 cm | 672 of 10 444 (6.4 %) |
| clusters with **both** ends outside the FV at 30 cm | 1045 |
| — of those, still two-ended at 2.5 cm | 751 |
| — of those, **becoming single-exit (stopping-muon candidates) at 2.5 cm** | **232** |

against 663 `STM = 1` tags in the same arm. The direction is worth being
explicit about, because it inverts the intuition: a *larger* inset makes a
*smaller* FV, so it cannot create the fully-contained population — it converts
a stopping end near a CRP into a second exit. Confirmed in code: the primary
exit test is `!inside_fv(p1)` per extreme point (`Clustering_Util.cxx:146`), a
two-exit cluster is arbitrated by `inside_fv` on both boundary points (:225-245),
and `check_stm_conditions` only fits when exactly one exit survives
(`TaggerCheckSTM.cxx:3510`). **A muon that stops within 30 cm of a CRP reads as
through-going.** That is the doc-25 Michel population.

*Model and its limits*: clusters are reduced to their two PCA-extreme points and
the PCA axis; `cluster_fc_check` uses steiner extreme groups and a Hough
direction. These are population estimates, not a replay of the tagger.

---

## 7. The near-CRP question — measured, not acted on

The 30 cm margin exists because the fitted dQ/dx was seen to rise ~50 % over the
last ~30 cm before either CRP (doc 25 M3). The owner's instruction is that
shrinking the FV is not the answer and that the effect needs to be understood
first. A first measurement, on imaged-point charge over 40 events:

| drift distance inside the anode wall | PDVD median q (× bulk) | SBND (× bulk) |
|---|---|---|
| 0 – 2 cm | 0.89 | 0.98 |
| 2 – 5 cm | 1.31 | 1.04 |
| 5 – 10 cm | 1.33 | 1.04 |
| 10 – 20 cm | 1.37 | 1.01 |
| 20 – 30 cm | 1.37 | 0.96 |
| 30 – 50 cm | 1.28 | 1.02 |
| 50 – 100 cm | 1.13 | 1.02 |
| 100 – 150 cm | 1.10 | 1.00 |
| 150 – 200 cm | 1.00 | 1.01 |
| 200 – 300 cm | 1.00 | — |
| 300 – 400 cm | 0.93 | — |

**The PDVD excess is not a near-CRP feature. It is a smooth, monotone gradient
over the whole 340 cm of drift** — an exponential fit over drift > 10 cm gives a
factor 0.67–0.73 across the full drift, i.e. τ of order 6 ms (the fit moves
between ~5.7 and ~7.3 ms with the binning, so read the shape, not the value —
it is nowhere near tight enough to be a lifetime measurement). There is no step, edge or knee at 30 cm; 30 cm
is simply where a smooth curve first looks noticeably high. SBND's profile is
flat to within 4 %, so the shape is not an artefact of the instrument. The one
genuinely local feature is the **last 2 cm**, where PDVD drops to 0.89.

**Caveats, and they are not small.** Bee `q` is per-point blob charge, not the
fitted dQ/dx the original observation was made on; the PDVD arm is data and the
SBND arm is MC, so only PDVD carries a real electron lifetime; and no angular or
track-population control is applied. This is suggestive, not a result. The
instrument that would settle it is the one doc 25 already has: fitted dQ/dx vs
residual range on stopping muons, binned in drift distance — if the rise
survives a lifetime correction it is instrumental, and if it does not, the
correct fix is a lifetime/purity correction and the fiducial volume should never
have been involved.

---

## 8. Review of doc 32 §6 — what holds and what does not

**Holds.** §6.1: both detectors' tagger FV already spans the whole detector, so
the design goal "the entire detector is the basis of the FV" is met for the
endpoint tests. §6.2(a): `FiducialUtils` really does receive `fiducial=dv`, and
the consumer list is accurate. §6.2(b): the ±341.55 cm overhang is real.
§6.2(c): `clustering_examine_x_boundary` really does raise on differing `FV_x`.
§7 R5 is real: `Clustering_Util.cxx:301-319` computes `bp1_r2/bp2_r2` and never
writes them into `result.boundary_first/second` (only :260-263 does, from round
1), so when round 1 finds no exit the fit is anchored on a pair round 2 never
validated.

**Four corrections.**

1. **§6.2(a)'s worked example cannot occur.** It shows a −30 cm tolerance
   probing into the cathode hole. `Clustering_Util.cxx:116` routes `fc_check` to
   the configured box whenever `fiducial` is set, and PDVD has
   `stm_consistent_fv=true`; the only tolerance vectors that reach
   `FiducialUtils` are the −1.5 cm ones at `NeutrinoTaggerCosmic.cxx:566` and
   `NeutrinoTaggerNuE.cxx:2464`, and PDVD has `cosmic_consistent_fv` and
   `nue_sp_consistent_fv` on, so those sites use the box too. **Every surviving
   `FiducialUtils` call on PDVD passes no tolerance at all**, and the hole is
   therefore its bare geometric width, not width + 30 cm.
2. **The holes have no measured effect on point containment** (§6.2 above):
   0 of 832 391 `stm_fit` points. R4 cannot be justified by "the union is
   holed"; §6.2's march truncation is the mechanism that survives.
3. **§5's outcome table double-counts.** All 35 "fully contained" clusters are
   inside the 41 "evaluated but no pass recorded" — the log emits both lines for
   the same cluster. The correct reading strengthens the case: containment is
   87.6 % of the no-fit population arm-wide, not one of two comparable buckets.
4. **"35 fully contained in a cosmic event" does not survive a component
   check.** Of the 143 fully-contained clusters with PCA extent > 300 cm in the
   arm, **zero are a single connected component** (`component_extreme_wcps`); the
   extent is a multi-fragment bounding-box artefact. The suspicion was
   reasonable, the number does not support it, and the open question is now the
   per-*component* one.

**A consequence for the ranking.** With the holes measured, routing
`MakeFiducialUtils` to `pdvd_pr_fv` (doc 32 R4) is a **PDVD-config-only change
with a small, measured effect** — not the widest-blast-radius item it was ranked
as. What it must not be done without is §9 R4a.

---

## 9. Recommendations

Nothing below is implemented. Each names the knob it needs, the gate it owes and
the measurement that grades it.

**R1 — `tgm_fv_x_margin`: understand the near-CRP behaviour, do not trade the
volume for it.** Per the owner's steer, this is *not* a proposal to lower the
margin today. The order is: (a) measure fitted dQ/dx vs residual range in bins
of drift distance on the doc-25 stopping-muon sample and test whether the excess
is the global lifetime gradient §7 measured; (b) if it is, correct it as a
lifetime/purity correction and the margin question answers itself; (c) only then
revisit 30 cm, knowing what it was protecting against. The cost of leaving it is
now quantified (§6.3: 232 clusters/arm that would be stopping-muon candidates at
SBND's margin), so the decision can be made on numbers instead of instinct.
Knob exists (`tgm_fv_x_margin`); a change would owe a 120-event arm plus a
hand-scan of the STM/TGM/FC flips, since the margin is shared across all three
verdicts by design (`docs/27_fc-tgm-consistent-fv.md`).

**R2 — give PDVD a structure-exclusion fiducial, the way SBND has one for its
CPA.** This is the direct answer to the owner's note about the gaps. SBND's
`cathode_fiducial.jsonnet` returns a `CompositeFiducial` OR of boxes with a
per-axis cushion, referenced by `tn`; the primitive is generic and already in
the toolkit. PDVD's equivalent would carry: the 6 cm cathode slab, the central
y seam, the two CRP y seams and the CRP z seam — each dilated to the width §5
*measured* rather than the width the geometry declares (the y seam at 168.5 cm
is the case in point: 0.02 cm geometric, ~±1 cm real). First consumer should be
Q/L matching, mirroring SBND (`cathode_fiducial=` is already a plumbed argument
that defaults to ''), so the change is config-only and PDVD-local. A second
consumer — a tagger test "this end sits in a structural gap, so it is not a
stopping point" — is the useful one for the STM programme but needs a C++ site
and its own round.

What this recommendation does *not* carry is a measured value. §5 measures how
wide each seam is; nothing here shows that excluding those bands improves any
downstream verdict. So R2 is "the primitive exists, here is the width to build
it at, and the first consumer is config-only" — not a fix waiting to be applied.
Its own round would grade it on the Q/L side first (crosser admission, boundary
flags), where a wrong answer is cheap to see.

**R3 — measure the seams the same way for SBND before generalising.** SBND's
CPA profile (§5) shows the same qualitative shape as PDVD's cathode with a much
smaller hole. If a PDVD gap fiducial is built, the same script grades whether
SBND's 0.5 cm cushion is the right dilation there too. Read-only, no arm.

**R4 — the two-step for `FiducialUtils`.**
**R4a (C++, prerequisite, default-OFF knob):** in `check_dead_volume` and
`check_signal_processing`, score a step whose `contained_by()` gives no
(apa,face) as **transparent** rather than `num_points_dead++`. Today a step in a
structural gap counts as a dead-channel step; swap the volume without this and
every cathode step starts counting as dead. `FiducialUtils` is bound by SBND and
PDHD too, so this owes gates on all three.
**R4b (config, PDVD-local):** then route `MakeFiducialUtils` to `pdvd_pr_fv`
(`fiducial=dv` → `wc.tn(pdvd_pr_fv)` at `pr.jsonnet:1098,1100`). Use the box,
not `EnvFiducial`: same numbers, one fewer component, and it makes
`FiducialUtils` exactly the volume the taggers already use. Measured effect
(§6.2): the 3.7 % of marches, plus ~0.8 cm in z. Owes a compiled-config proof
and a 120-event arm; not byte-identical when on.

**R5 — make the vertex FV bonus consistent.** `NeutrinoVertexFinder`'s
`in_fv` / `W_FV` sites (:1256, :2184, :4450, :5172) score against the un-inset
union while every other tagger test uses box + margins — a 30 cm disagreement in
x on PDVD. Since the uBooNE DL vertex became the production main-vertex result
(doc 28 §27), which vertex wins matters more than when doc 32 was written. Knob,
plus a count of vertices whose bonus changes.

**R6 — `cluster_fc_check`'s round-2 write-back** (doc 32 R5, verified in §8).
Cheap, self-contained, and doc 32 §19.1 already puts it on the critical path for
any `good_point_pitch_frac` above ~0.4. Check the prototype's intent first.

**R7 — `dvm.overall` FV_x = ±341.55 cm** (§3.2). The CRP centreline where every
other FV in the chain uses the shield plane (±339.91), feeding
`select_scope_fv`'s all-anode fallback. Small, and possibly deliberate — but the
two surfaces are 1.64 cm apart and only one of them bounds the argon; fix or
write down which was meant.

**Not recommended.** Doc 32 R3 (giving the end-trim walk the tagger volume) as a
standalone item: with `good_point_pitch_frac` at its production value the pitch
floor dominates that loop, and which of the two tests pops each point has not
been separated. Measure first — the probe arm already exists.

---

## 10. What this document does not establish

- Every population number in §6 uses the PCA-extreme model of §6.3, not the
  tagger's own extreme groups.
- §7 is imaged-point charge, not fitted dQ/dx, and compares a data arm against a
  MC arm. It is a reason to measure properly, not a result.
- The gap profiles in §5 are density profiles over all clusters; no attempt is
  made to control for track angle, and a seam that suppresses efficiency for
  tracks *along* it will look different from one crossed at 90°.
- The per-*component* fully-contained question (§8 correction 4) is open: no
  single connected ≥ 100 cm component has been shown either to be, or not to be,
  wrongly skipped.
- Nothing here has been run through a gate, because nothing here changes
  anything.

## 11. Related

`32_stm-trajectory-end-coverage-and-fiducial-volume.md` (§6 is the audit this
reviews), `07_pdvd-tpc-geometry-fiducial.md` (the Q/L-side geometry),
`28_pdvd-pr-perf-round1.md` §27 (the DL vertex flip that raises R5),
`25_pdvd-stopping-muon-michel-chain.md` (M3, the origin of the 30 cm margin),
`clus/docs/clustering-separate-fv.md` (the 15 cm inset),
`sbnd_xin/docs/49_stm-containment-fv-inconsistency.md` and `docs/27_fc-tgm-consistent-fv.md`
(the SBND precedents), `cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet`
(the structure-exclusion primitive R2 would copy).

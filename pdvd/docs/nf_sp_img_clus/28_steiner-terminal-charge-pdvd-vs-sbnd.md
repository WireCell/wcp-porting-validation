# The Steiner terminal-charge floor: PDVD (500 e) vs SBND (4000 e default)

Answers, with code citations and fresh measurements on real PDVD **and**
SBND data: (1) what the "charge" is and how it is calculated, (2) a
PDVD/SBND side-by-side of every input that feeds that number, (3) whether
PDVD shares SBND's `rebin=4`, and (4) where the factor-of-8 gap between the
two floors (500 vs 4000 e) actually comes from — including a direct test of
whether it is a three-plane wire/strip-crossing geometric mismatch, and
whether a newer PDVD wire-geometry file (`v7-uvwfit`) fixes it. Follows up
on doc pdvd/25 §13.4 item 8 (`terminal_charge_threshold` knob, M3), which
measured the PDVD symptom but did not decompose the cause; this doc does
the decomposition.

**TL;DR.** The floor is a per-3D-point, per-view charge cut used to pick
Steiner-tree terminal candidates. It is **not** primarily a pitch, rebin, or
absolute-calibration effect — PDVD's coarser pitch and identical rebin,
taken alone, predict **more** charge per point than SBND, the opposite of the
observed problem, and each individual plane's charge distribution turns out
to be a comparable scale on both detectors. **Confirmed on real data from
both detectors (§4.3, updated)**: the actual driver is a genuine **wire-
crossing geometric mismatch**, exactly the mechanism the owner asked to check
directly. PDVD's `stepped` sampler places a **median of 4 candidate 3D
points per (U,V) wire crossing** (84% of crossings ambiguous), because its
coarser pitch leaves the third view under-constrained; SBND places a
**median of 1** (only 21% ambiguous). It is specifically the resulting
"losing" (non-peak) candidates that fail `calc_charge_wcp`'s three-plane AND
gate at a roughly 2× lower rate than the "peak" candidate at the same
crossing, on **both** detectors — PDVD just manufactures far more of them.
A fresh wire-geometry file (`protodunevd-wires-larsoft-v7-uvwfit.json.bz2`)
was also tested end-to-end (re-imaged and re-sampled, §7): it does **not**
reduce the crossing ambiguity (if anything, slightly more of it), but it
does modestly improve per-point charge accuracy, netting a real but partial
gain (AND-gate pass rate 17.4%→20.2% at 4000 e).

---

## 1. What "charge" is, and the algorithm that thresholds it

Three stages, upstream to downstream:

### 1a. Per-point, per-plane charge (`BlobSampler`)

Steiner terminal-finding runs on the cluster's 3D point cloud, built by
`BlobSampler`'s **`stepped`** strategy (`clus/src/BlobSampler.cxx:703-916`).
For each accepted blob, `stepped` walks a sub-grid of wire-crossing
candidates (step size `max(min_step_size=3, blob_width/12)` wires in each of
the two most/least-covered views) and places a 3D point at each crossing
that also lands inside the third view's strip (`BlobSampler.cxx:840-869`).
This is a **geometric combinatorics** step: for an ambiguous (coarse-pitch or
wide) blob it can place many points, not one point per blob.

For every accepted point, and **independently per plane**, the point is
projected onto that plane's pitch coordinate, snapped to the nearest wire
(`pimpos->closest`), and the **channel's already-deconvolved, calibrated
activity value for that time slice** is read off directly — no splitting or
re-normalization by point count (`BlobSampler.cxx:315-370`):
```cpp
charge_val[ipt] = act.value();       // -> stored as u/v/w "charge_val"
charge_unc[ipt] = act.uncertainty(); // -> stored as u/v/w "charge_unc"
```
These are written into the point cloud as `ucharge_val` / `vcharge_val` /
`wcharge_val` (and `*_unc`), read back by
`Facade::Cluster::charge_value()` / `charge_uncertainty()`
(`clus/src/Facade_Cluster.cxx:953-1029`).

### 1b. Per-point RMS charge and "quality" (`calc_charge_wcp`)

`Cluster::calc_charge_wcp(point_index, charge_cut, disable_dead_mix_cell)`
(`clus/src/Facade_Cluster.cxx:1031-1112`), called with
`disable_dead_mix_cell=false` from the Steiner path
(`clus/src/SteinerGrapher.cxx:435`, `NeutrinoSteinerGapGraph.cxx:170`):

```cpp
flag_u = (charge_u > charge_cut) || (charge_u == 0);   // same for v, w
// charge = RMS over planes with NONZERO charge (>1 required, else 0)
charge = sqrt( sum(charge_p^2 for p with charge_p != 0) / n_nonzero )
quality = flag_u && flag_v && flag_w
```
i.e. a plane "passes" if it clears the cut **or reports no signal at all**
(dead/no-hit planes don't count against you); the returned **charge is the
RMS of the plane values that are nonzero**, and `quality` requires all three
planes to individually pass. A plane with *some* charge below the cut fails
the whole point.

### 1c. Terminal selection (`find_peak_point_indices`)

`Steiner::Grapher::find_peak_point_indices` (`clus/src/SteinerGrapher.cxx:418-520`,
ported from WCP `PR3DCluster_steiner.h:711-780`, doc reference
`clus/docs/patternrecognition/steiner_graph_review.md`):

```cpp
const double charge_threshold = m_config.terminal_charge_threshold;   // 4000 default
for (point_idx : all_indices) {
    auto [charge_quality, charge] = cluster.calc_charge_wcp(point_idx, charge_threshold, false);
    if (charge > charge_threshold && charge_quality)
        candidates_set.insert({charge, point_idx});    // gated TWICE by the same cut
}
// then: local-maximum ("peak") selection over the graph, highest charge first
```
A point must clear `terminal_charge_threshold` **and** have all three planes
individually pass the same cut (§1b) to even enter the candidate pool; peaks
are then picked from that pool by local-maximum suppression over the cluster
graph. `terminal_charge_threshold` is a `CreateSteinerGraph` config field
(`clus/src/SteinerGrapher.h:85-97`, wired in
`clus/src/CreateSteinerGraph.cxx:64-104`), **C++ default 4000.0** — pinned by
`clus/test/doctest_steiner_terminal_charge_defaults.cxx`. This is described
in-code as "the WCP prototype constant, tuned to uBooNE's 3 mm pitch"
(`SteinerGrapher.h:85-88`) — i.e. **4000 is not an SBND-specific number**,
it is the historical uBooNE value that SBND (and every other detector)
inherits unmodified.

**Distinct from the imaging activity threshold.** This is a different gate
than the `nthreshold × RMS` cut in `MaskSlice` that decides whether a
(channel, tick) is *activity* at all before blobs are even formed (doc
pdvd/25 nf_sp_img_clus `20_imaging-threshold.md`). That one is
detector-adaptive (self-scales to each channel's live noise RMS) and both
detectors already share it identically. `terminal_charge_threshold` is
downstream of imaging and clustering entirely, is a **fixed absolute
electron count**, and gates Steiner-graph terminal selection specifically.

---

## 2. Side-by-side: every config input, verified from source

| quantity | PDVD | SBND | source |
|---|---|---|---|
| `terminal_charge_threshold` (Steiner floor) | **500 e** (`steiner_terminal_charge=500`) | **4000 e** (C++ default, unmodified) | `wcp-porting-img/pdvd/wct-pr-perevt.jsonnet:643`; `cfg/pgrapher/experiment/protodunevd/pr.jsonnet:73` (knob, `null`=off by default); `clus/src/SteinerGrapher.h:97` |
| wire pitch U / V / W | **7.65 / 7.65 / 5.10 mm** | **3.00 / 3.00 / 3.00 mm** | `wirecell-util wires-info` on `protodunevd-wires-larsoft-v6.json.bz2` and `sbnd-wires-larsoft-v1.json.bz2` (measured directly, this session) |
| pitch ratio (PDVD/SBND) | U/V **2.55×**, W **1.70×** | — | derived |
| imaging rebin | **4** (`img.jsonnet`), **6** long-ROI (`sp.jsonnet: lroi_rebin`) | **4**, **6** | `cfg/pgrapher/experiment/protodunevd/img.jsonnet:33,72`; `.../sp.jsonnet:137`; `cfg/pgrapher/experiment/sbnd/img.jsonnet:33,99`; `.../sp.jsonnet:61` — **identical**, see §3 |
| raw DAQ tick | 0.5 µs (shared base, neither overrides) | 0.5 µs | `cfg/pgrapher/common/params.jsonnet:69` |
| imaging tick-slice physical width (rebin×tick×v_drift) | 2 µs × 1.48073 mm/µs = **2.96 mm** (data-calibrated production value; config literal default is 1.568 mm/µs, unused — see below) | 2 µs × 1.6 mm/µs = **3.2 mm** (common-base nominal 500 V/cm; SBND overrides nothing) | `cfg/pgrapher/common/params.jsonnet:29`; `cfg/pgrapher/experiment/protodunevd/params.jsonnet:131`; production value from `[[project_pdvd_doc25_stopping_muon_michel]]` memory / `run_clus_evt.sh:793-794` |
| front-end electronics gain | **7.8 mV/fC** (bottom drifter, `elec.gain`, the nominal used for the majority of channels) | **14.0 mV/fC** | `cfg/pgrapher/experiment/protodunevd/params.jsonnet:166-168`; `cfg/pgrapher/experiment/sbnd/params.jsonnet:119` |
| pulser/theoretical ADC-per-1000e calibration | not stated in `params.jsonnet` (PDVD's own charge calibration is explicitly flagged incomplete, §4.2) | 41.649 ADC·tick/1ke (pulser), 36.6475 (theory, 14 mV/fC) | `cfg/pgrapher/experiment/sbnd/params.jsonnet:120-121` |
| MIP `mip_dqdx` reference table (post-recombination, e/cm) | **~55000** (E=0.44 kV/cm, scaled from SBND's rule) | ~56000 (plateau × 1.0246 rule PDVD copies) | `cfg/pgrapher/experiment/protodunevd/particle_dataset.jsonnet:1-40` |
| `mip_dqdx_median` (e/cm) | **~47000** | ~48000 | same file, same comment block |
| coherent-NF ADC-domain thresholds (`decon_limit`, `adc_limit`, ...) | detector-specific, `gain_scale`-anchored to **7.8 mV/fC** | detector-specific, no `gain_scale` (anchored at 14 mV/fC design gain) | `pdvd/docs/nf_sp_img_clus/11_coherent_nf_params_comparison.md` |

The two *physics-scale* rows (MIP dQ/dx, per cm) are within ~2% of each
other — PDVD is not depositing or reading out less charge per unit track
length than SBND. The pitch row is the one that should predict **more**
charge per PDVD wire crossing, not less (§4.1).

---

## 3. Rebin — yes, identical

Both `cfg/pgrapher/experiment/protodunevd/img.jsonnet` and
`cfg/pgrapher/experiment/sbnd/img.jsonnet` set `rebin: 4` (imaging tiling
tick-binning) and `nrebin: 4` (a second, related knob in the same file,
comment: *"this number should be consistent with the waveform_map choice"*),
and both `sp.jsonnet` files set `lroi_rebin: 6` (long-ROI ticks). None of
PDVD's three `img*.jsonnet` variants (`img.jsonnet`, `img_bkup.jsonnet`,
`img_bkup_ori.jsonnet`, `img_standalone.jsonnet`) differs from SBND on this
value. **Rebin is ruled out as a contributor to the 500-vs-4000 gap.**

---

## 4. Where the gap comes from

### 4.1 Pitch predicts the wrong sign

A track crossing a wire roughly perpendicular to it deposits charge on that
wire in proportion to the *physical length* subtended by one pitch cell,
i.e. roughly `dQ/dx × pitch`. Using the §2 numbers: PDVD W-plane,
`55000 e/cm × 0.51 cm ≈ 28000 e`; SBND (any plane, 3 mm pitch),
`56000 e/cm × 0.30 cm ≈ 16800 e`. Both are far above either 500 or 4000 — so
**a single-wire, well-aligned crossing is not the failure mode on either
detector**, and PDVD's coarser pitch should, if anything, make the "easy"
crossings even easier. This directly contradicts a pitch-driven
explanation and matches the intuition that "PDVD should have more charge
than SBND" for the well-behaved case.

### 4.2 PDVD's absolute charge calibration is explicitly incomplete (in-repo, admitted)

`protodunevd/particle_dataset.jsonnet` (comment, lines ~33-39) states, about
its own dQ/dx tables:

> "The undocumented 0.85 scale factor is deliberately RETAINED (it is not a
> physics term and is degenerate with the **missing gain / electron-lifetime
> calibration**); PDVD must not silently inherit it once a PDVD charge
> calibration exists."

This is a direct, source-level acknowledgment that PDVD does not yet have an
independently-verified absolute electron-charge calibration; the physics
tables (§2) work by matching a template shape, not by an independently
verified absolute scale. A residual gain or electron-lifetime mis-calibration
of even 20-30% would be enough to push a meaningful population of points from
just-above to just-below any fixed absolute-electron cut (4000 or 500),
without changing the qualitative track physics at all. **This is the
single most concrete, source-confirmed candidate for a genuine
detector-to-detector scale offset** — as distinct from the geometric effect
in §4.3, which does not require any calibration error to operate.

### 4.3 The AND-of-three-planes gate, and the wire-crossing ambiguity behind it — confirmed on both detectors

`calc_charge_wcp`'s `quality` (§1b) requires **all three views to
individually clear the cut** (or read exactly zero). This alone predicts
that a 3D point whose local track direction runs close to parallel to one
view's wires can fail even though the track's total dQ/dx is normal — but
that mechanism, by itself, does not explain a PDVD/SBND difference (both
detectors have tracks at all angles). The actual PDVD-specific driver is
*upstream* of the charge cut, in **which 3D points `stepped` sampling (§1a)
generates in the first place**.

`stepped` places a 3D point for every valid crossing of a sub-grid in the
two widest-covered views, filtered by whether that crossing also lands
inside the *third* ("mid") view's strip (`BlobSampler.cxx:817-869`,
tolerance `0.03` pitch units). When the mid view's strip is wide — which
happens more often the coarser its pitch, because a fixed physical
uncertainty maps to fewer, wider wire cells — **multiple mid-view wire
positions all satisfy the tolerance test for the same (U,V) crossing**,
so `stepped` emits several 3D points that share the same U and V wire but
differ only in their mid-view (typically W) position. Each such point's
charge is read independently (§1a): the true track charge is spread across
that competing set (or concentrated on one), so most of the group's points
see a fraction of it — the classic **imaging-tomography wire-crossing
ambiguity** the owner asked to check, now specifically caught at the
Steiner-sampling stage rather than at blob formation.

**Measured, this session, on real data from both detectors** (script in the
repro block, extended to report this):

| | PDVD (run 39252 evt 298595, `work/039252_2_stm1`, production v6 geometry) | SBND (`sbnd_xin/work-dbg25a-d97off/ql_evt16`, real event 16) |
|---|---|---|
| sampled 3D points | 160930 | 23930 |
| per-plane median charge (incl. zero) | U 4711 / V 3792 / W 3895 e | U 4168 / V 4132 / W 4873 e |
| AND-gate(4000 e) pass rate, all points | **17.4%** | **20.7%** |
| AND-gate(500 e) pass rate, all points | **42.7%** | **74.3%** |
| distinct (U,V) wire-crossings among sampled points | 31587 | 18712 |
| **median candidate points per (U,V) crossing** | **4** | **1** |
| fraction of (U,V) crossings with >1 candidate | **83.6%** | **21.0%** |
| fraction of ALL points that are a "losing" (non-peak) candidate | **80.4%** | **21.8%** |
| AND-gate(4000e): peak candidate | 30.5% | 23.2% |
| AND-gate(4000e): losing candidate | 14.2% | 11.9% |

Two things fall out cleanly:

1. **The per-plane charge scale and even the raw AND-gate pass rate at
   4000 e are comparable between detectors** (17.4% PDVD vs 20.7% SBND) —
   confirming §4.4's suspicion that SBND has a very similar three-plane-AND
   shortfall at the historical 4000 e floor, just never audited the way
   PDVD's STM effort audited itself. This is *not* a PDVD-specific charge
   deficiency.
2. **The mechanism is sharply different and is a geometric mismatch, not a
   charge-scale one.** SBND's `stepped` sampler almost always resolves a
   (U,V) crossing to a *single* W candidate (median 1, only 21% ambiguous).
   PDVD's resolves it to a **median of 4** candidates (84% ambiguous) —
   because PDVD's coarser pitch leaves substantially more W-wire positions
   consistent with any given (U,V) crossing within the same relative
   tolerance. On *both* detectors, a "losing" (non-peak) candidate at an
   ambiguous crossing passes the AND gate at roughly half the rate of the
   "peak" candidate (PDVD 14.2% vs 30.5%; SBND 11.9% vs 23.2% — the losing/
   peak ratio is ~0.46-0.51 on both) — so the *per-candidate* physics of
   "being a losing wire-crossing candidate" is the same mechanism in both
   detectors. PDVD simply manufactures roughly **4× more** such losing
   candidates as a fraction of all sampled points (80% vs 22%), because its
   coarser pitch structurally under-constrains the third view far more
   often. This directly confirms the owner's hypothesis: **it is a
   three-plane wire/strip-crossing geometric mismatch**, not a bulk
   calibration offset.

Doc pdvd/25's cited "W-plane median ~1400 e" (§13.4 item 8) was measured
over the *target blobs of the specific STM main clusters that were already
failing to get a Steiner skeleton* — a harder, edge-of-track-biased
subsample than this whole-event measurement. The two are consistent: the
shortfall is **concentrated** in particular tracks/regions (track ends,
near-boundary geometry, long tracks — §13.4 item 8's own finding), and the
mechanism that turns "each plane's charge is mostly fine" into "most points
fail the joint AND gate" is exactly this wire-crossing ambiguity, worse in
the harder subsample doc pdvd/25 measured directly.

### 4.4 Gain/lifetime calibration (§4.2) is not ruled out, but is no longer the leading explanation

§4.2's calibration gap is real and still unresolved in the repo, but §4.3's
head-to-head measurement now accounts for the bulk of the qualitative
PDVD/SBND difference through a purely geometric mechanism that requires no
calibration error at all. A calibration offset would shift *both* the peak
and losing populations' charge together; what is actually observed is a
population-mix effect (far more losing candidates), which calibration alone
does not produce. §4.2 remains worth fixing on its own merits (it affects
the absolute MeV scale of every dQ/dx-based measurement), just not as the
primary lever on this specific 500-vs-4000 gap.

---

## 5. Answering the four questions directly

1. **Algorithm**: `BlobSampler`'s `stepped` strategy places 3D points at
   wire-crossing sub-grid intersections and tags each with the raw
   deconvolved-and-calibrated per-channel activity value on each of the
   three planes (no splitting between points); `calc_charge_wcp` reduces
   those three per-plane values to one RMS-over-nonzero-planes number and a
   three-plane AND quality flag; `find_peak_point_indices` requires both the
   RMS value and the quality flag to clear `terminal_charge_threshold`
   before a point can even be a terminal candidate, then runs local-maximum
   suppression over the surviving candidates.
2. **Side-by-side**: §2 table — pitch 2.55×/1.70× coarser on PDVD, rebin
   identical, gain 7.8 vs 14.0 mV/fC, MIP dQ/dx scale within ~2%.
3. **Rebin**: yes, PDVD uses the identical `rebin: 4` / `lroi_rebin: 6` as
   SBND (§3) — ruled out.
4. **Where the gap comes from**: not pitch (predicts the wrong sign, §4.1);
   not rebin (identical, §3); not primarily calibration (§4.4 — a real,
   separate issue, but doesn't explain the population-mix effect). **It is a
   three-plane wire-crossing geometric mismatch** (§4.3, confirmed on real
   PDVD *and* SBND data): PDVD's coarser pitch leaves the third view under-
   constrained far more often, so its `stepped` sampler emits ~4× as many
   "losing" (non-peak, charge-starved) candidate points per (U,V) wire
   crossing as SBND's (median 4 vs 1 candidates/crossing; 84% vs 21% of
   crossings ambiguous), and those losing candidates fail the AND gate at
   about half the peak candidate's rate — the same per-candidate physics on
   both detectors, just far more common on PDVD.

## 6. A new wire geometry (`v7-uvwfit`) was also tested (§7)

The recommended next step from the previous version of this doc (an SBND-
side measurement) is now done, above. A newer PDVD wire-geometry file,
`protodunevd-wires-larsoft-v7-uvwfit.json.bz2`, arrived mid-investigation
and was tested against the same mechanism — see §7.

## 7. Does a corrected wire geometry (`v7-uvwfit`) help?

Tested end-to-end on the same event (re-imaged **and** re-sampled with the
new geometry — an imaging-only or sampling-only swap is geometrically
inconsistent and was tried first, then discarded; see the repro block).
Result:

| | v6 (production) | v7-uvwfit |
|---|---|---|
| sampled 3D points | 160930 | 196745 |
| AND-gate(4000 e) pass rate | 17.4% | **20.2%** |
| AND-gate(500 e) pass rate | 42.7% | **48.2%** |
| median candidates/(U,V)-crossing | 4 | 5 |
| fraction of crossings ambiguous | 83.6% | 86.5% |
| fraction of points that are losing candidates | 80.4% | 83.7% |
| AND-gate(4000e): peak / losing candidate | 30.5% / 14.2% | 33.0% / 17.7% |

**v7-uvwfit gives a real but partial improvement, through a different
channel than expected.** It does **not** reduce the wire-crossing ambiguity
that §4.3 identifies as the dominant mechanism — if anything the ambiguity
is very slightly worse (more sampled points, marginally higher ambiguous
fraction), consistent with §4.3's diagnosis that the ambiguity is a
*structural* consequence of PDVD's pitch, not a wire-position calibration
artifact. What v7-uvwfit *does* do is make each individual charge lookup
more accurate — both peak and losing candidates pass the AND gate at a
visibly higher rate (peak 30.5%→33.0%, losing 14.2%→17.7%) — most likely
because more accurate wire positions mean `pimpos->closest`'s wire snap
(§1a) lands on the true peak wire more often instead of an adjacent one.
Net effect: a genuine ~3 percentage-point (4000 e) to ~5.5 percentage-point
(500 e) gain, worthwhile but not a fix on its own — even with v7-uvwfit,
PDVD's AND-gate pass rate (20.2%) merely reaches parity with SBND's already-
marginal 4000 e number (20.7%, §4.3), not SBND's much healthier 500 e number
(74.3%). The lower `terminal_charge_threshold` PDVD already runs at (500 e)
is still doing most of the work; v7-uvwfit is a worthwhile independent
improvement to combine with it, not a substitute.

## Repro block

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
    work/039252_2_stm1/pctree-evt298595.tar.gz          # PDVD v6 (production)
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
    /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-dbg25a-d97off/ql_evt16/pctree-evt16.tar.gz  # SBND
```
Wire pitch (verified independently of any doc citation):
```
wirecell-util wires-info /nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v6.json.bz2
wirecell-util wires-info /nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd-wires-larsoft-v1.json.bz2
```
v7-uvwfit test (§7) — re-image AND re-sample with the new geometry (not a
partial swap: see §7). `rerun_with_wires_geometry.sh` makes a scratch copy
of `toolkit/cfg`, patches only that copy's `params.jsonnet` `wires:` line,
and runs imaging + clustering against it; no tracked file is touched.
Verified in this session to reproduce the §7 table bit-for-bit (196745
points, AND-gate(4000e)=0.202) when run standalone under a second tag:
```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./docs/nf_sp_img_clus/scripts/rerun_with_wires_geometry.sh \
    protodunevd-wires-larsoft-v7-uvwfit.json.bz2 v7img 039252 2
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
    work/039252_2_v7img/pctree-evt298595.tar.gz
```
(requires `work/039252_2_keep/protodune-sp-dnnroi-frames-anode{0..7}.tar.bz2`,
already staged from the standard production run of this event.)

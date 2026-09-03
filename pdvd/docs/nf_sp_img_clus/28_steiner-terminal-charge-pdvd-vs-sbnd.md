# The Steiner terminal-charge floor: PDVD (500 e) vs SBND (4000 e default)

Answers, with code citations and a fresh measurement on real PDVD data:
(1) what the "charge" is and how it is calculated, (2) a PDVD/SBND
side-by-side of every input that feeds that number, (3) whether PDVD shares
SBND's `rebin=4`, and (4) where the factor-of-8 gap between the two floors
(500 vs 4000 e) actually comes from. Follows up on doc pdvd/25 §13.4 item 8
(`terminal_charge_threshold` knob, M3), which measured the PDVD symptom but
did not decompose the cause; this doc does the decomposition.

**TL;DR.** The floor is a per-3D-point, per-view charge cut used to pick
Steiner-tree terminal candidates. It is **not** primarily a pitch or rebin
effect — PDVD's coarser pitch and identical rebin, taken alone, predict
**more** charge per point than SBND, the opposite of the observed problem.
The two verified drivers are (a) `calc_charge_wcp`'s **AND-of-three-planes**
gate, which a coarse-pitch, wide-angle-wire geometry fails more often even
when each view's charge is individually fine, and (b) PDVD's own
in-repo-documented **incomplete absolute-charge calibration** (gain +
electron-lifetime), explicitly flagged as unresolved in
`protodunevd/particle_dataset.jsonnet`. A true PDVD-vs-SBND side-by-side of
the *reconstructed* per-point charge distribution (not just config) is not
done here — SBND's side of that measurement is the recommended next step
(§6).

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

### 4.3 The AND-of-three-planes gate is the mechanism demonstrated on real data

`calc_charge_wcp`'s `quality` (§1b) requires **all three views to
individually clear the cut** (or read exactly zero). A 3D point whose local
track direction runs close to parallel to one view's wires deposits little
charge on that plane specifically, even though the same point's charge on
the other two views, and the track's total dQ/dx, are completely normal.
`stepped` sampling (§1a) is combinatorial over wire crossings across all
three views at once, so it actively generates candidate points at exactly
these unfavorable projections, more of them the coarser (or more strip-width
ambiguous) the pitch. This mechanism needs no calibration error and would
operate on both detectors — the open question is only how much *more*
often PDVD's coarser pitch + stereo-angle geometry produces marginal points
this way than SBND's 3 mm pitch does.

**Measured, this session**, on the M3 STM full-chain event actually behind
the doc pdvd/25 census (run 39252, evt 298595, `work/039252_2_stm1/pctree-evt298595.tar.gz`,
160930 sampled 3D points across the whole live cluster forest — reproducible
via the script in §6):

| | U | V | W | RMS (`calc_charge_wcp`) |
|---|---|---|---|---|
| median (incl. zero-charge points) | 4711 e | 3792 e | 3895 e | 10562 e |
| median (nonzero points only) | 6989 e | 6250 e | 4608 e | 11153 e |
| fraction > 4000 e | 53.1% | 49.3% | 49.5% | 76.6% |
| fraction > 500 e | 75.3% | 71.9% | 75.0% | 93.3% |
| fraction with zero charge | 14.7% | 15.5% | 5.3% | 4.7% |

but the **joint** (AND) requirement:

| cut | fraction with ALL THREE planes passing |
|---|---|
| 4000 e | **17.4%** |
| 500 e | **42.7%** |

Each individual plane's median (~3800-4700 e, whole-event) already sits
close to or above 4000 e — this is a much less dire picture than doc
pdvd/25's cited "W-plane median ~1400 e", because that number was measured
over the *target blobs of the specific STM main clusters that were already
failing to get a Steiner skeleton* (a hard, edge-of-track-biased subsample),
whereas this measurement is whole-event, unconditional. The two are
consistent, not contradictory: they show the shortfall is **concentrated**
in particular tracks/regions (consistent with §13.4 item 8's own finding
that it correlates with track ends, near-boundary geometry, and long
tracks), not a uniform detector-wide charge deficit — and the mechanism
that turns "each plane is mostly fine" into "only 17% pass all three
simultaneously" is exactly the AND-gate of §1b/§4.3, not a bulk charge
scale problem. Raising the floor from 4000 to 500 e recovers this gate from
17.4% to 42.7% passing, matching the qualitative shape (not exact value —
different population) of doc pdvd/25's 4000→500 census table.

### 4.4 What this doc does *not* establish

No SBND-side measurement of the same quantity (real per-point
`calc_charge_wcp` distribution, and the same AND-gate fraction) was made in
this session — there was no SBND `pctree-evt*.tar.gz` on disk to reuse. The
claim "SBND doesn't need a lower floor" rests only on the fact that no one
has overridden `terminal_charge_threshold` for SBND, which is weaker
evidence than a direct measurement: it is also consistent with "SBND has the
same AND-gate shortfall but nobody has run the equivalent §13.4-item-8
census to notice." §6 lists this as the recommended next step, since it is
the only way to convert §4.3's mechanism from "demonstrated on PDVD, plausible
in general" to "quantitatively responsible for the 500-vs-4000 gap."

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
   not rebin (identical, §3); the two live candidates are (a) PDVD's
   in-repo-acknowledged incomplete absolute gain/lifetime calibration
   (§4.2, unquantified) and (b) the three-plane AND gate in `calc_charge_wcp`
   interacting with coarse-pitch, stepped-sampling geometry, which is
   demonstrated mechanistically on real PDVD data in §4.3 (53→17% percent
   points when going from "any one plane passes" to "all three pass"
   simultaneously at 4000 e). Disentangling how much of the remaining
   gap is (a) vs (b) needs the SBND-side measurement in §6.

## 6. Recommended next step (not done here)

Run the §4.3 script (below) against an SBND full-chain event's
`pctree-evt*.tar.gz` and report the same table. If SBND's AND-gate fraction
at 4000 e is comparably low (~15-20%), the gap is dominantly a calibration
issue (§4.2) that SBND simply hasn't needed to fix because SBND's STM/PR
efficiency isn't being measured against it the way PDVD's is; if SBND's
AND-gate fraction is much higher (~60-80%+), the geometric AND-gate
mechanism (§4.3) is the dominant, pitch-driven, non-calibration cause.

## Repro block

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_charge_census.py \
    work/039252_2_stm1/pctree-evt298595.tar.gz
```
Wire pitch (verified independently of any doc citation):
```
wirecell-util wires-info /nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v6.json.bz2
wirecell-util wires-info /nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd-wires-larsoft-v1.json.bz2
```

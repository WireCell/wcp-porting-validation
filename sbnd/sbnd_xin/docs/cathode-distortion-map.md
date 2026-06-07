# SBND cathode-plane field-cage distortion map

## Why

SBND is suspected to have a field-cage distortion near the cathode plane,
concentrated in one top corner of the detector. After Q/L matching + the
per-cluster T0 correction every matched cosmic is reconstructed at its true
drift position, so the reconstructed geometry near the central cathode
(x = `cathode_x` ≈ ±0.45 cm) can be used to **map** the distortion across the
(Y, Z) plane (Y vertical, Z beam) and localise the problem.

This extends [`cathode-crossing-diagnostic.md`](cathode-crossing-diagnostic.md),
which measured the *aggregate* TPC0/TPC1 cathode offset (data ≈ 1.4 cm vs
MC ≈ 0.5 cm transverse), into a **spatially resolved** field over the cathode plane.

## Data

- **Source:** per-event calib dump `work/ql_evt<ID>/calib-evt<ID>.json`
  (`QLMatching::dump_calib`, `match/src/QLMatching.cxx:1626`), produced with the
  existing observation-only flags:
  ```
  SBND_DATA_DIR=input-3files-lan-reco2/{1,2,3} ./run_ql_evt.sh data all -calib -cathode-diag
  ./run_ql_evt.sh data all -calib -cathode-diag      # 10 extra (input-1file) data events
  ./run_ql_evt.sh mc   all -calib -cathode-diag      # 10 MC reference events
  ```
  These flags default OFF and leave the matched output byte-identical, so nothing
  in the reco path changes.
- **Sample:** 160 data events (150 lan-reco2 + 10) and **10 MC** as a
  low-distortion reference (MC's cathode offset is ≈ ⅓ of data).
- **T0-corrected position** of a cluster in its `auto_selected` bundle:
  `x_corr = x_raw + sign_offset · flash_time_us · drift_speed_cm_per_us`
  (y, z unchanged). The cathode-most point of a cathode-reaching track should
  land at `cathode_x`.
- The clean **xTPC** signal comes from the vetted `QLCATHODE` log lines
  (`dump_cathode_diag`, `QLMatching.cxx:1854`): for each cross-cathode track the
  two TPC halves must meet at the cathode, so the perpendicular (transverse)
  component of the connecting vector — `perp = conn − (conn·d̂)d̂` with
  `d̂ = unit(dir0 + dir1)` — is the artifact-immune distortion (the drift x part
  is degenerate with T0/velocity).

Analysis script: [`../cathode_distortion.py`](../cathode_distortion.py)
(parallel JSON load + plotting; `python3 cathode_distortion.py -j 16`).

## Method — six (Y, Z) views

| plot | what it isolates |
|---|---|
| `cathode_coverage_yz.png`            | near-cathode slab occupancy of **all** matched-track points (depletion ⇒ distortion / dead region) |
| `cathode_xresidual_yz.png`           | mean `dx = x_corr,end − cathode_x` (drift; **T0/velocity-degenerate** for single tracks — read vs MC) |
| `cathode_transverse_residual_yz.png` | per-track cathode-end **off-axis** transverse residual vs the full-track PCA axis (distortion **curvature**); data only — MC too sparse to bin |
| `cathode_xtpc_perp_yz.png`           | **xTPC** transverse offset scatter + (Δz, Δy) quiver — the clean, T0-immune signal |
| `cathode_profiles.png`               | transverse residual vs Y and vs Z (xTPC absolute, per-track curvature), data vs MC |
| `cathode_furthest_points_yz.png`     | cathode-most point of long tracks, coloured by `dx` |

The transverse (Y, Z) components are artifact-immune; the drift (x) component is
degenerate with T0/drift-velocity for single-TPC tracks and is shown only
relative to MC.

## Results

**The distortion is real, grows downstream, and peaks in the top-right corner.**

- **xTPC transverse offset:** data median **1.17 cm** vs MC **0.34 cm**
  (≈ 3.4× excess) — reproduces the aggregate 1.4 / 0.5 cm of the crossing
  diagnostic via an independent path. (`cathode_xtpc_perp_yz.png`)
- **Z dependence:** the transverse offset rises from ≈ 1.0 cm at low Z to
  ≈ 2.5 cm at Z ≳ 450 cm, while MC stays ≈ 0.3–1.3 cm across Z.
  (`cathode_profiles.png`, right)
- **Hotspot (auto-found, clean xTPC signal):** largest |perp| ≈ **3.0 cm** at
  **(Z ≈ 450, Y ≈ 150) cm — high-Z, high-Y = the top-right corner**, exactly the
  suspected field-cage region.
- **Per-track curvature** (off-axis residual) is data > MC in aggregate
  (median 0.89 vs 0.52 cm) but spatially noisy; it corroborates the excess
  without localising as cleanly as the xTPC signal.
- **Occupancy** of all near-cathode matched-track points (≈ 320k data points)
  shows the cosmic coverage; no large clean acceptance hole stands out at this
  statistics, so the signal is a *displacement* field, not a dead region.

![xTPC transverse offset](../pics/cathode_xtpc_perp_yz.png)
![Profiles vs Y and Z](../pics/cathode_profiles.png)
![Per-track off-axis residual (data)](../pics/cathode_transverse_residual_yz.png)
![Near-cathode occupancy](../pics/cathode_coverage_yz.png)
![Cathode-end drift residual](../pics/cathode_xresidual_yz.png)
![Furthest points](../pics/cathode_furthest_points_yz.png)

## Caveats

- **MC reference is only 10 events** → too sparse to bin in 2D (the off-axis map
  is data-only; MC contrast is carried by the 1-D profiles and the aggregate
  medians). MC is a *low*-, not zero-, distortion baseline.
- The corner-resolved signal rests on **244 xTPC crosser pairs** (data); a single
  (Y, Z) corner bin holds only a handful, so the hotspot location is indicative,
  not yet a precise measurement.
- Single-track `dx` is **degenerate** with T0 / drift-velocity; only the
  transverse (Y, Z) components are clean. The per-track off-axis residual
  measures distortion **curvature** (a gradient), not the absolute offset.

## Next steps

- **More events — the binding constraint.** The xTPC-crosser sample (~1.6/event)
  limits corner-by-corner statistics; ~10× more data (and especially more MC)
  would turn the indicative hotspot into a quantitative (Y, Z) distortion map.
- Once the corner is pinned, correlate (Z ≈ 450, Y ≈ 150) with the GDML
  field-cage geometry (`sbnd_geometry/sbnd_v02_06.gdml`) to identify the physical
  panel/feature responsible.
- If the displacement field is reproducible, consider a **(Y, Z)-binned transverse
  correction** feeding the existing `pos_offset` / `T0Correction` path (currently a
  single rigid offset, `pos_offset` in `cfg/.../sbnd/clus.jsonnet`) — design /
  measure only, kept toggleable + default-OFF per project convention.
- Cross-check against an independent SCE / space-charge expectation if a model
  becomes available; a true space-charge field would also bow tracks in x, which
  the drift-degeneracy currently hides.

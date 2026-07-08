# SimEnergyDeposit → Blob truth matching: SCE + smearing (future reference)

How `wclsTensorSetLabeler` (larwirecell `aiml/`) associates
`ionandscint:priorSCE` SimEnergyDeposits with the WCT clustering blobs, and
why each correction exists.  Status 2026-07-08: on corsika+GENIE MC the
remaining unlabeled blobs are dominated by real ghosts (especially
isochronous tracks) — the association itself is no longer the limit.

## The problem

Blobs are reconstructed from drifted, SCE-distorted, SP-filtered charge in
the raw (non-t0-corrected) coordinates.  `ionandscint:priorSCE` depos are
point-like samples at TRUE positions and times.  Matching them naively
(depo center inside the blob's slice/wire bounds) loses real blobs three
ways, each fixed in turn:

## 1. Drift/time mapping (raw coordinates)

BlobSampler defines raw x as `x = x_W + dirx*(t_sig + time_offset)*v` with
SBND `time_offset = -205 us` (= `sim.tick0_time`), `v = 1.563 mm/us`,
`tick = 0.5 us`.  Inverting for a depo at `(x_true, t_dep)`:

```
x_app  = x_true + dirx*v*t_dep          # apparent (drifted) position
t_sig  = (x_app - x_W)*dirx/v - time_offset
itick  = floor(t_sig / tick)            # compare to blob slice_index_[min,max)
```

Wire-in-plane indices come from the face RayGrid (`pitch_index`, layers
2/3/4 = u/v/w) — the same coordinates that defined the blob strips.

## 2. SCE: true → reco positions (evt1 blobs labeled 58% → 67%)

The SBND dualmap file
(`sbnd_data/v01_42_00/SCEoffsets/SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root`)
contains BOTH directions: `TrueBkwd_*` (reco→true, used by SCECorrection with
`sign:+1`) and `TrueFwd_*` (true→reco).  A second `SCEFieldTH3` instance
(`sbnd_dualmap_fwd`, `th3_name_* = TrueFwd_Displacement_*`, `sign: 1`) shifts
each depo true→reco before association ("postSCE" on the fly).  Displacement
is up to ~1.4 cm (|d| ≈ 0.8–1.0 cm near the cathode, ~0.1 cm near the
anodes) vs the 3 mm wire pitch — several wires.

## 3. Diffusion smearing (evt1 blobs labeled 67% → 82%)

Two smearings turn each (point-like) depo into a Gaussian ball, added in
quadrature per depo:

### 3a. Drift diffusion (detsim physics)

`sigma = sqrt(2*D*t_drift)` with `t_drift` from the depo's actual drift
distance and DL/DT from the WCT detsim configuration
(`sbndcode/WireCell/wcsimsp_sbnd.fcl`):

- `DL = 4.0 cm^2/s` → longitudinal, maps to the time direction
  (`sigma_t_diff = sigma_L / v`), ≈ 1.0 mm at full 1.3 ms drift;
- `DT = 8.8 cm^2/s` → transverse, maps to the wire-pitch directions,
  ≈ 1.5 mm at full drift.

### 3b. Signal-processing filter smearing

The SP deconvolution filters smear the charge further.  Following
`dunereco/DUNEWireCell/docs/smear-dnn-campaign.md`, with SBND filter values
from `sbndcode/WireCell/cfg/pgrapher/experiment/sbnd/sp-filters.jsonnet`:

- time: `sigma_t = 1/(2*pi*f)` with `f = Gaus_wide sigma = 0.10 MHz`
  → **1.59 us = 3.18 ticks**;
- wire: `sigma_w = 1/(2*sqrt(pi)*k)` [pitch units] with `k = 1.05`
  (`Wire_ind`, U/V) and `k = 3.60` (`Wire_col`, W)
  → **0.269 pitch (U,V)**, **0.078 pitch (W)**.

### Acceptance

A blob accepts a depo when its center is within
`slop + nsigma*sigma_total` of the blob bounds in every dimension
(tick + 3 wire views); `nsigma = 3` (as BlobDepoFill), `wire_slop = 1`,
`tick_slop = 2` remain as binning-granularity floors.  Blob trackid = argmax
of accumulated NumElectrons per track.

## Config (wclsTensorSetLabeler, defaults = the values above)

`DL`, `DT`, `sp_smear_time`, `sp_smear_wire_ind`, `sp_smear_wire_col`,
`nsigma`, `wire_slop`, `tick_slop`, `sce_field`/`sce_correction`,
`drift_speed`/`time_offset`/`tick` (MUST match the BlobSampler),
`n_sample_truth_depo_sce` (Bee: Gaussian samples per depo ball, default 1).

## Validation (event 32/10/6, corsika+GENIE)

| stage | blobs labeled | points labeled |
|---|---|---|
| point depos + slop | 2347/4031 (58%) | 65% |
| + SCE TrueFwd      | 2690/4031 (67%) | 77% |
| + diffusion        | 3287/4031 (82%) | 84% |

NN trackid coherence 100% throughout.  Final BEE set (with the
`truth_depo_sce` drifted+smeared depo cloud):
https://www.phy.bnl.gov/twister/bee/set/01ff0ce1-dd15-47f4-8d5f-929c3632b7e7/event/list/
Remaining unlabeled blobs: real ghosts, mostly isochronous tracks.

Tooling and run recipes: `sbnd/TensorSetLabeler/README.md`.

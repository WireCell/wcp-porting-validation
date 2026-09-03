# PDVD production defaults: E = 450 V/cm, DL/DT, 20 ms lifetime, v7-uvwfit wires

Owner decision (2026-09-03): adopt E = 450 V/cm (0.45 kV/cm) as PDVD's
production drift field, with the DL/DT transport coefficients and dQ/dx
reference tables recomputed at that field, the electron lifetime recorded at
20 ms, and the `v7-uvwfit` wire geometry (colleague-supplied refit) made the
PDVD default. This doc records what changed, the derivation of each number,
and a smoke-test validation on real data.

## 1. What changed and where

| # | Item | File | Before | After |
|---|------|------|--------|-------|
| 1 | Drift field (recombination) | `cfg/pgrapher/experiment/protodunevd/pr.jsonnet`, `pdvd_box_recomb.data.Efield` | 0.44 kV/cm | **0.45 kV/cm** |
| 2 | Longitudinal diffusion `DL` | `cfg/pgrapher/experiment/protodunevd/params.jsonnet`, `lar.DL` (new) / `pdvd_track_fitting.json`, `DL` | unset (inherited 7.2 cm²/s) / 4.12e-07 | **4.1307 cm²/s** / 4.1307e-07 |
| 3 | Transverse diffusion `DT` | same, `lar.DT` (new) / `pdvd_track_fitting.json`, `DT` | unset (inherited 12.0 cm²/s) / 7.82e-07 | **7.9135 cm²/s** / 7.9135e-07 |
| 4 | Electron lifetime | `cfg/pgrapher/experiment/protodunevd/params.jsonnet`, `lar.lifetime` (new) | unset (inherited 8 ms) | **20 ms** |
| 5 | Wire geometry default | `cfg/pgrapher/experiment/protodunevd/params.jsonnet`, `files.wires` | `protodunevd-wires-larsoft-v6.json.bz2` | **`protodunevd-wires-larsoft-v7-uvwfit.json.bz2`** |
| 6 | Track-fitting json | `cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` | DL/DT at 0.44/1.48073-derived | DL/DT at 0.45 (item 2/3), comments updated |
| 7 | PID dQ/dx curve | `cfg/pgrapher/experiment/protodunevd/particle_dataset.jsonnet` | tables generated at 0.44 kV/cm | **tables regenerated at 0.45 kV/cm** |

Items 2-4 (`lar.DL`/`lar.DT`/`lar.lifetime`) are new overrides in PDVD's
`params.jsonnet`. Before this change PDVD did not override any of the three
and silently inherited the generic base defaults (`cfg/pgrapher/common/
params.jsonnet`: `DL=7.2`, `DT=12.0` cm²/s, `lifetime=8` ms) — values that
have no connection to PDVD's actual field and were not consumed by anything
in the PDVD production chain either. They are now recorded as the physical
source of truth (§3).

## 2. Where E = 450 V/cm and T = 87.3 K come from, and why DL/DT follow

The DL/DT values were **not** picked by hand — they come straight out of
`pdvd/stm/pdvd_transport.py`, the same BNL LAr-properties (lar.bnl.gov/
properties) mobility/diffusion parameterisation used for the previous
(0.44 kV/cm) round (doc pdvd/25 §7a/§8):

```
mu(E,T)    = (a0 + a1 E + a2 E^1.5 + a3 E^2.5) / (1 + (a1/a0) E + a4 E^2 + a5 E^3) * (T/89 K)^-3/2   [cm^2/V/s]
eps_L(E,T) = (b0 + b1 E + b2 E^2) / (1 + (b1/b0) E + b3 E^2) * (T/87 K)                                [eV]
DL         = mu * eps_L
DT         = DL / (1 + (E/mu) dmu/dE)
```

Repro:
```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 -c "
import sys; sys.path.insert(0, 'stm')
from pdvd_transport import diffusion
print(diffusion(0.45, 87.3))"
# -> (4.1307..., 7.9135...)  matches the adopted DL/DT exactly
```

This uses **T = 87.3 K**, the site's default temperature — a different choice
than the previous (0.44 kV/cm) round, which used T = 87.68 K (dunecore
`protodunevd_detproperties`) because that round *derived* E from the
data-calibrated drift velocity and needed a specific T to invert. This round
instead **sets E = 450 V/cm directly** (a round production-target number, not
a measurement), so T is now simply the parameterisation's default input
rather than a value tied to a specific velocity measurement. At T = 87.68 K
instead the same field gives DL = 4.1217, DT = 7.8963 cm²/s — a <0.3 %
difference, well inside the existing T-band systematic already noted in
`pdvd_track_fitting.json` (±0.5 % over 87.3-88.0 K).

## 3. A flagged inconsistency: drift speed was not part of this change

`lar.drift_speed` (params.jsonnet, 1.568 mm/us) and the production Q/L drift
speed (`run_clus_evt.sh`, 1.48073 mm/us) were **not changed** by this
request. Inverting either through the same BNL parameterisation does not
land exactly on 0.45 kV/cm:

```
v_bnl(E=0.45 kV/cm, T=87.3 K) = 1.509 mm/us     (vs production 1.48073, params.jsonnet 1.568)
```

So, as of this change, PDVD's recombination model and DL/DT (now at
450 V/cm) are **not** self-consistent with the drift speed actually used to
place hits in x and to compute `add_sigma_L` in the track-fitting smearing
(`pdvd_track_fitting.json`'s `add_sigma_L = 1.9639 mm` is unchanged, still
derived from 1.48073 mm/us). This doc flags it rather than silently
reconciling it (CLAUDE.md escalation rule 7: report a physics number that
looks wrong, don't tune it to look right) — drift speed was explicitly out of
scope for this request and changing it is a separate decision with its own
consequences (x-position calibration, `add_sigma_L`, everything downstream of
`v = 1.48073`).

## 4. The PID dQ/dx curve (`particle_dataset.jsonnet`)

Regenerated with the repo's own recipe, unchanged except the field argument:

```
cd /nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel
root -l -b -q 'convert_field.C(0.45, "stopping_ave_dQ_dx_pdvd045.root", true)'
cd ../docs
python3 emit_jsonnet_dedx.py ../pion_travel/stopping_ave_dQ_dx_pdvd045.root
```

Modified-Box parameters are unchanged (`alpha=0.93, beta=0.212, rho=1.38,
W_ion=23.6e-6`, undocumented `fudge=0.85` retained per the existing file
header) — only `Efield` moves, entering through `beta' = beta/(rho*E)`.

Consequences (0.45 vs the previous 0.44 kV/cm):

| Quantity | 0.44 kV/cm | 0.45 kV/cm | Δ |
|---|---|---|---|
| Muon plateau (rr = 59.5 cm) | 53798.3 e/cm | 53965.5 e/cm | +0.31% |
| Muon Bragg bin (rr = 0.5 cm) | 158255 e/cm | 159982 e/cm | +1.1% |
| MIP (dE/dx = 2.1 MeV/cm) | 52481 e/cm | 52635.2 e/cm | +0.29% |

The previous (0.44 kV/cm) ROOT file is left untouched at
`energy_loss/pion_travel/stopping_ave_dQ_dx_pdvd.root`; the new one is a
separate file, `stopping_ave_dQ_dx_pdvd045.root`, in the same directory (the
`energy_loss` repo has its own remote, `lastgeorge/energy_loss`, is not one
of the two repos this project's CLAUDE.md governs, and is not touched by this
change's commits — only used here as a computation tool).

**`mip_dqdx`/`mip_dqdx_median` are unchanged, and this is not new staleness.**
The existing header comment already documented a scaling rule (mip_dqdx =
plateau × 1.0246, rounded to the nearest 1000) that, at the previous 0.44
kV/cm plateau, gives 55000 — but `pr.jsonnet`'s actual defaults are still the
raw SBND values `mip_dqdx=56000` / `mip_dqdx_median=48000`, i.e. that
correction was never applied. Applying the same rule to the new plateau
(53965.5 × 1.0246 = 55291, still rounds to 55000) doesn't change the
answer, so there's nothing new to reconcile from *this* table refresh, but
the pre-existing 56000-vs-55000 gap remains open and is out of scope here.

## 5. Wire geometry: v6 → v7-uvwfit

`v7-uvwfit` (`protodunevd-wires-larsoft-v7-uvwfit.json.bz2`) is a colleague-
supplied refit of the U/V/W wire geometry, superseding v6 as the PDVD
production default. It was already characterized in doc pdvd/28 §7: on the
039252 evt 2 Michel/STM sample, re-imaging and re-sampling consistently under
v7-uvwfit **partially improves** the Steiner terminal-charge AND-gate pass
rate (17.4%→20.2% at the 4000e floor, 42.7%→48.2% at 500e) by making
individual per-point charge lookups more accurate — it does **not** reduce
the underlying three-plane wire-crossing ambiguity that doc pdvd/28
identified as the dominant effect. See doc pdvd/28 §7 for the full
before/after and the discarded (geometrically inconsistent) first attempt.

## 6. Validation: smoke test on real data

Full imaging → clustering → PR chain re-run on run 039252, event 2 (the same
Michel/STM event used throughout doc pdvd/25 and doc pdvd/28), entirely
against the new production defaults (v7-uvwfit geometry, E=0.45 kV/cm
recombination, new DL/DT, new dQ/dx tables), under a fresh tag
`work/039252_2_v450/`:

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
# imaging: 8 anodes, DNN-ROI SP frames reused from 039252_2_keep (SP is
# upstream of these changes and unaffected by them)
PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -s v450 -calib -save-pctree 039252 2
./run_pr_evt.sh -s v450 -stm-fit 039252 2
```

Results:
- Imaging: all 8 anodes compiled and ran successfully (rc=0 each) under the
  new v7-uvwfit default.
- Clustering: completed, `pctree-evt298595.tar.gz` (20.2 MB), wall 51 s, peak
  RSS 1.30 GB.
- PR chain (`-stm-fit`, mode `-nu`): completed, wall 60 s, peak RSS 2.28 GB;
  `mabc-pr.zip` 1.7 MB, `calib-pr-evt298595.json` 10.0 MB,
  `tracking-{stm,pr}.root` both written non-empty. 92 STM/TGM verdict lines;
  the same 6 clusters as prior rounds (ident 39/40/55/86/87/90) tag
  `STM=1 TGM=0` and survive `ClusteringProtectBundle`'s open-for-unconvicted
  path unsplit -- consistent with the event's known STM population (doc
  pdvd/25 / doc pdvd/28), i.e. the tagging behavior did not regress.
- Confirmed the new tables actually reach the output: `calib-pr-evt298595
  .json`'s `dqdx_ref` block (`source: "ParticleDataSet"`) carries
  `muon[0]=159982.0`, `muon[-1]=53965.5`, `electron[0]=51555.3` -- an exact
  match to the regenerated 0.45 kV/cm tables (§4), confirming the whole chain
  (imaging with v7-uvwfit -> clustering -> TaggerCheckSTM/Neutrino -> PR
  display dump) is reading the new defaults end-to-end, not a stale cached
  config.

This is NOT a byte-identical gate — every one of the 7 items above is an
intentional production-default change, not a knob (CLAUDE.md §5 rule 1); the
smoke run's purpose is only to confirm the new defaults run end-to-end
without crashing and produce a physically sane result, per the "knob-on smoke
run" bar in CLAUDE.md §4. NOT byte-identical to any prior PDVD production
run; needs revalidation against the owner's stopping-muon/Michel sample
before wider use.

## Repro

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 -c "
import sys; sys.path.insert(0, 'stm')
from pdvd_transport import diffusion
print(diffusion(0.45, 87.3))"           # DL/DT derivation, sec 2

cd ../../energy_loss/pion_travel
root -l -b -q 'convert_field.C(0.45, "stopping_ave_dQ_dx_pdvd045.root", true)'
cd ../docs && python3 emit_jsonnet_dedx.py ../pion_travel/stopping_ave_dQ_dx_pdvd045.root

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -s v450 -calib -save-pctree 039252 2
./run_pr_evt.sh -s v450 -stm-fit 039252 2
```

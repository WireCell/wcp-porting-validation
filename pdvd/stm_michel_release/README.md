# STM + Michel release — ProtoDUNE-VD data (twin of the PDHD release)

**What this is.** One self-contained package per event with everything the Wire-Cell
stopping-muon (STM) + Michel-electron tagger reconstructed on ProtoDUNE-VD data, prepared
for the charge + light Michel energy work and for the stopping-muon studies (range,
dQ/dx → dE/dx, multiple Coulomb scattering).  The PDHD twin lives in
`pdhd/stm_michel_release/` with the same layout, the same scripts and the same documents
(the documents describe both detectors; PDVD-specific points are marked).

| what you want | where it is | document |
|---|---|---|
| STM verdict, muon energies (range / dQ/dx / MCS), Michel energies, stop and Michel positions | `T_stm` in the per-event ROOT file | [ROOTFILE.md](ROOTFILE.md) §2 |
| the muon trajectory with dQ/dx per point and the residual range | `T_stm_pts`, `T_stm_fit` | ROOTFILE.md §3, §4 |
| the 2-D charge measurements (wire, time, charge after signal processing) of the muon and of the Michel | `T_stm_2d` (whole cluster, fitted), `T_michel_2d` (Michel / gamma / muon-footprint cells with the fit predictions) | ROOTFILE.md §5, §6 |
| the matched light flash: time, PE per photon detector, the OpHits | `T_flash`, `T_ophit` | [LIGHT.md](LIGHT.md) |
| the optical waveforms, raw and deconvolved, in a window from the matched flash | `T_opwf` | LIGHT.md §3 |
| the Bee 3-D event display | `bee_<det>_<run6>_<evtid>.zip` next to the ROOT file | [BEE.md](BEE.md) |
| the dQ/dx expectation vs residual range (muon, electron, ...) | `dqdx_ref.json` | ROOTFILE.md §8 |

## Layout

```
README.md  ROOTFILE.md  LIGHT.md  BEE.md      this documentation
index.csv                                     one line per event: run, idx, evtid, n_candidates, n_stm, n_michel, files
dqdx_ref.json                                 dQ/dx expectation tables (e/cm vs residual range cm)
events/<run6>_<evtid>/stm_michel_<det>_<run6>_<evtid>.root
events/<run6>_<evtid>/bee_<det>_<run6>_<evtid>.zip
scripts/stm_release.py                        reader helpers (numpy + uproot only)
scripts/list_candidates.py                    table of every candidate with energies and the matched flash
scripts/plot_dqdx_vs_rr.py                    dQ/dx vs residual range against the expectation (all muons, or one track)
scripts/plot_michel_energy.py                 Michel charge energy spectra + the 2-D-cell re-derivation check
scripts/plot_muon_energy.py                   muon KE by range vs MCS vs dQ/dx
scripts/plot_flash_waveforms.py               matched flash: PE pattern and raw/decon waveforms of the brightest PDs
scripts/plot_event_2d.py                      the 2-D charge picture of one candidate in the three planes
scripts/build_release.py                      the builder (owner tool; reproduces this directory from the production outputs)
figs/                                         the figures the scripts make on this release (see below)
```

`<run6>` is the zero-padded run number, `<evtid>` the DAQ / art event number.  The index
column `idx` is the position of the event in the run's processing list (the name the work
directories carry).

## Quick start

```sh
cd <this directory>
python3 scripts/list_candidates.py                           # every candidate, one line each
python3 scripts/list_candidates.py --michel-only --csv michels.csv
python3 scripts/plot_dqdx_vs_rr.py                           # -> figs/dqdx_vs_rr.png
python3 scripts/plot_dqdx_vs_rr.py --event 039252_298567 --cluster 75
python3 scripts/plot_michel_energy.py                        # -> figs/michel_energy.png
python3 scripts/plot_muon_energy.py                          # -> figs/muon_energy.png
python3 scripts/plot_flash_waveforms.py --event 039252_298567 --cluster 75
python3 scripts/plot_event_2d.py --event 039252_298567 --cluster 75 --zoom 40
```

Or in your own code:

```python
import sys; sys.path.insert(0, "scripts")
import stm_release as sr
for path in sr.event_files("."):
    ev = sr.open_event(path)
    c = sr.candidates(ev, ["cluster_id", "is_stm", "michel_found", "muon_ke_range", "michel_ke_q2d_region", "flash_id"])
    for i in range(len(c["cluster_id"])):
        if c["is_stm"][i] and c["michel_found"][i]:
            pts = sr.points(ev, cluster_id=int(c["cluster_id"][i]), role=1)   # muon dQ/dx vs rr
            fl = sr.flash(ev, int(c["flash_id"][i]))                            # per-PD PE of the matched flash
            wf = sr.waveforms(ev, flash_id=int(c["flash_id"][i]), opdet=int(fl["pe"].argmax()))  # PDVD: several DAPHNE channels per OpDet
```

Requirements: python3 with `numpy`, `uproot` (5.x) and `matplotlib` for the plots.  The files
are plain ROOT TTrees, so ROOT / PyROOT read them too (`TTree::Draw("michel_ke_q2d_region",
"is_stm==1 && michel_found==1")`).  Nothing here needs the Wire-Cell toolkit.

## What a "candidate" is

The chain runs on every cluster the cosmic taggers flagged as a possible stopping muon.  For
each such cluster the tagger refits the track from its entry point, walks it to the Bragg
stop, tests the dQ/dx profile (Bragg contrast, shape against a muon template, plateau at the
MIP value, PID against proton / electron, continuation, fiducial stop, ...) and searches for a
Michel electron and its gammas at the stop.  Each candidate is one row of `T_stm`:

* `is_stm == 1` — the candidate passed every test (`reject_bits == 0`).  **Use this as the
  stopping-muon selection.**  `reject_bits` names what failed otherwise (decoded by
  `stm_release.reject_names`).
* `michel_found == 1` — a Michel object was found at the stop (`michel_conn_type` 1 attached,
  2 bridged, 3 unfitted dots).  A muon with no Michel is still a good stopper (mu- capture).
* The muon energies `muon_ke_range` (range), `muon_ke_dqdx` (calorimetric), `muon_ke_mcs`
  (multiple scattering), all MeV; the Michel energies `michel_ke_q2d_region` (**the production
  estimator**: 2-D charge within 10 cm of the stop minus the muon's fitted charge, through the
  recombination model), `michel_ke_best` (the association estimate), and the others of §2.

Hand-scan results for these events (PDVD purity 0.97 / efficiency 0.88 for `is_stm`; the
STM+Michel selection purity 0.96 / efficiency 0.83) are in the owner's docs (pdhd/docs/26,
pdvd/docs/116); the release carries the chain's output, not the hand labels.

## Provenance (what produced these files)

* Detector data: ProtoDUNE-VD runs 039252 (18 events), 039253 (18) and 039349 (84); signal
  processing with the Wire-Cell DNN-ROI chain, 3-D imaging, clustering with charge-light
  matching, the stopping-muon + Michel tagger (`CheckSTM_Michel`) — all Wire-Cell toolkit
  (branch `apply-pointcloud`); the arm tag and the toolkit commit of every event are in
  `T_event` (`arm`, `toolkit_commit`, `pr_dir`, `light_dir`, `frames_dir`).
* Light: all 40 OpDets (cathode and membrane X-ARAPUCAs, PMTs; 51 DAPHNE channels)
  reconstructed in one job (Wiener-inspired deconvolution, OpHit finding, OpFlash finding
  over 1 µs bins); the flashes in `T_flash` are exactly the archive the charge-light matching
  consumed; the waveforms in `T_opwf` come from a re-run of that job with a frame dump
  switched on, whose flash output was verified byte-identical to production on every event
  (120/120 PDVD, 61/61 PDHD).
* Every number in the ROOT file is copied from the production products.  The few derived
  convenience columns are listed in ROOTFILE.md §9.

## Caveats

* dE/dx is not stored per point.  Per-point values are dQ/dx in electrons per cm (after
  signal processing, no recombination or lifetime correction applied).  The energies go
  through the reconstruction's recombination model (Modified Box at the detector field);
  the expectation table `dqdx_ref.json` uses the same model.
* PDVD's two drift volumes read different charge scales (plateau at 1.04 of the expectation
  for x < 0 and 0.93 for x > 0 in the owner's studies).  `T_stm_fit.x` (and `T_stm_2d.apa`,
  CRPs 0–3 bottom / 4–7 top) tell which volume a candidate is in.  Light and charge times
  are per drift volume on PDVD (LIGHT.md §1).
* Raw optical waveforms are pedestal-not-subtracted ADC; where a self-triggered PD had no
  snippet in the window the raw array is exactly 0 (`n_raw_nonzero` counts the real samples).
  See LIGHT.md.
* The Bee zip cannot be opened offline; it is uploaded to a Bee server (BEE.md).

Built 2026-09-22 (owner doc pdhd/docs/30).

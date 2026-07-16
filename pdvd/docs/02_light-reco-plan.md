# PDVD Light Reconstruction — Plan

Status: **C++ chain DONE up to OpFlash** (2026-07-08): milestones 1–3 as python
studies (`03_pdvd-spe-template.md`, `04_pdvd-light-filter.md`, `05_pdvd-flash-dt.md`) and
the WCT chain assembled + validated in **`06_pdvd-light-chain.md`**
(PDVDOpWaveformSource, wiener-inspired OpDecon, 3-branch OpHit, all-PD
OpFlashFinder with DAPHNE→OpDet ganging; all 120 events of runs
039252/039253/039349 processed). Remaining: light↔charge `offset_us`, cathode
PE anchor, Q/L matching. The rest of this doc records the original input-data
survey, the detector layout, the signal-processing plan, the SPE-template
strategy, the OpHit/OpFlash (flash-formation) brainstorm, and the phased
implementation roadmap. The reference implementation is the PDHD light chain
(`toolkit/flash/` + `cfg/pgrapher/experiment/pdhd/flash.jsonnet` +
`pdhd/wct-light-*.jsonnet`; see `pdhd/docs/pdhd-light-raw-data.md`,
`pdhd/docs/pdhd-fullstream-light-reco.md`, `toolkit/flash/docs/design.md`).

## 1. Input data survey

Raw light waveforms from jjo, symlinked at `pdvd/input_data_light` →
`/nfs/data/1/jjo/data/PDVD/`. Extracted from the same raw HDF5 lineage as the
charge processing (Xuyang's data directory), so light and charge events match.

Five ROOT files (~150–195 MB each), each with a companion `.log` listing the
source branch (`raw::OpDetWaveforms_pdvddaphne_daq_pdvdpdsraw`) and per-event
waveform counts:

```
np02vd_raw_run039252_1176_df-s03-d3_dw_0_..._rawwf.root   18 events
np02vd_raw_run039253_0138_df-s03-d0_dw_0_..._rawwf.root
np02vd_raw_run039349_0035_df-s03-d3_dw_0_..._rawwf.root   28 events (19409–19949)
np02vd_raw_run039349_0041_df-s05-d3_dw_0_..._rawwf.root
np02vd_raw_run039349_0100_df-s03-d0_dw_0_..._rawwf.root
```

Charge data (`pdvd/input_data`) covers runs 039252/039253/039324/039349/040475/041189,
so **light exists for 039252, 039253, 039349 only**. Charge `evt_N` folders are
**index-named**, not event-numbered (same gotcha as PDHD): light↔charge matching
must join on the raw `event` number. Run 039349 has three light files from
different `df-s*-d*` streams — check whether their event lists are disjoint or
overlapping before batch processing.

### TTree schema

Unlike PDHD (datawriter ROOT → `decoana` TH1D layout), the PDVD files are
already flattened: a single TTree `raw_waveform`, one entry per waveform:

| branch | type | notes |
|---|---|---|
| run, subrun, event | int32 | raw event numbers |
| opchannel | int32 | OfflineChannel, block-coded: 10xx=Cathode XA, 20xx=Membrane XA, 30xx=PMT |
| opdet | int32 | OpDet index 0–39 |
| nsamp | int32 | 1024 (self-trigger) or 468800/468864 (full-stream) |
| x, y, z | double | OpDet center (cm, larsoft geometry) |
| timestamp | double | waveform start time (~5.74e9 ticks scale) |
| adc | vector&lt;int16&gt; | waveform |

### Framing: self-trigger vs full-stream (measured, run 039252 file)

| population | framing | count/evt |
|---|---|---|
| 8 cathode XAs (OpDet 4–11, 16 DAPHNE ch 10x0/10x1) | full-stream, 468800 or 468864 samples (7.5 ms @ 16 ns), 1 record/ch/event | 16 |
| 8 membrane XAs (2 DAPHNE ch each, 20xx) | self-trigger, 1024 samples | O(100–2500)/ch |
| 20 live PMTs (1 ch each, 30xx) | self-trigger, 1024 samples | O(100–1400)/ch |

**Both full-stream lengths (468800 and 468864) occur — the decon FFT length
handling must accommodate two sizes** (or pad/truncate to one).

### Measured waveform characteristics

- **Polarity: positive-going** in these files (PDHD was negative). Note the
  larsoft VD digitizer fcl comments that real data is negative — jjo's
  extraction has evidently already flipped it. Verify polarity per file;
  configure `input_polarity=+1`.
- Pedestals: XA ~2740–2830 ADC; PMT ~7500–8400 ADC (per-channel, from head samples).
- Self-trigger pulse peak at sample ~76 (data pretrigger; sim uses 128).
- **Membrane XA**: slow pulse (average FWHM ~66 ticks ≈ 1 µs incl. scintillation
  late light); trigger-pulse amplitude spectrum shows a **clean 1-PE peak at
  ~40–50 ADC** with a 2-PE shoulder near ~90 → data-driven SPE extraction is feasible.
- **PMT**: very fast pulse (FWHM 4–6 ticks ≈ 64–96 ns); amplitude spectrum
  starts at ~100 ADC with **no PE quantization visible** (multi-PE trigger
  threshold and/or gain smearing) → PMT SPE calibration needs a different route (§4).
- DAPHNE is 14-bit (rail 16384): the PDHD saturation machinery
  (`detect_saturation`/`veto_saturation`/`saturation_pad`) applies as-is.

### Channel status (matches colleague's scan)

- Dead PMTs **OpDet 24, 27, 28, 34** — absent from the data entirely.
- **OpDet 14** (ch 3010) low-response — very few self-triggers.
- 51 of 56 DAPHNE offline channels present. All 16 XAs good.
- larsoft `channelstatus_pdvd.fcl` has `BadChannels: []` — no official static
  list; treat the above as run-level knowledge (mask via config, like PDHD's
  dead-channel mask).

## 2. Detector layout

40 OpDets per `dunecore/ChannelMap/PDVD_PDS_Mapping_v09162025.json`
(**use this version**; the older v04152025 numbers membrane/PMT OpDets
differently — the data follows v09162025):

- **Cathode X-ARAPUCA**: OpDet 4–11, `pd_type=Cathode`, PTP WLS, 2 DAPHNE ch each.
- **Membrane X-ARAPUCA**: OpDet 0–3, 12, 13, 18, 19, `pd_type=Membrane`, 2 DAPHNE ch each.
- **PMT**: remaining 24 (14–17, 20–39), 1 ch each — reused ProtoDUNE-DP
  Hamamatsu R5912-02MOD, WLS split: 6 TPB-coated + 18 PEN-foil (mapping `wls` field).

Positions from the data (x = vertical drift coordinate, cathode at x = 0, top
CRPs at x ≈ +340, cryostat floor at x = −336.5):

- Cathode XAs: x = 0, spread over the cathode plane (y ∈ [−213, +290], z ∈ [41, 259]).
- Membrane XAs: on the y = ±417.6 walls; **4 view the top volume**
  (x = +305.6, +229.0) and **4 view the bottom volume** (x = −201.1, −277.7).
- PMTs: all x &lt; 0 — side banks at x = −205.9/−281.7 and floor bank at x = −336.5
  (some outside the active z range: z up to 456 and down to −156).

**The cathode XAs are double-sided (confirmed).** `protodunevd_v5_ggd.gdml`
places `volOpDetSensitive_XARAPUCADoubleWindow` for exactly the 8
`pos_cathode_*_xarapuca_*` positions, while membrane XAs use the single-window
`volOpDetSensitive_XARAPUCAWindow`. This also follows from coverage logic: the
top volume is otherwise viewed by only 4 wall XAs. Consequences:

- Top volume light: 4 top membrane XAs + cathode top faces.
- Bottom volume light: 4 bottom membrane XAs + 24 PMTs + cathode bottom faces.
- A cathode XA's PE is a **sum over both faces** — the flash finder cannot
  attribute it to a volume by itself; only a light model (Q-L matching) can.

bee3 already draws a PDVD `op` instance (40 ch, hand-derived from the same
mapping, flagged "not validated" in `wire-cell-bee3/docs/protodune_geometry.md`).
Our `pdvd-opdet-geom.json` (from the ROOT tree x/y/z = larsoft geometry) will
double as the validation reference for bee3.

## 3. Signal-processing plan

The toolkit `flash/` components (`OpDecon`, `OpRoi`, `OpHitFinder`,
`OpHitMerge`, `OpFlashFinder`) are detector-agnostic — **no C++ algorithm
changes expected**; possibly small knobs if gaps appear (e.g. two full-stream
lengths). Per PDHD experience and per user directive, keep **two separate SP
chains** merged at the hit level:

```
                       ┌─ self-trigger (1024): membrane XA + PMT
raw_waveform TTree ────┤     OpDecon(samples=1024) → OpHitFinder ─┐
 (PDVDOpWaveformSource)│                                          ├─ OpHitMerge → OpFlashFinder(nchan=40) → opflash tensor
                       └─ full-stream (~468k): cathode XA         │
                             OpDecon(fixed_snr) → OpRoi → OpHitFinder ─┘
```

New pieces needed:

1. **`PDVDOpWaveformSource`** (in `root/`, fork-by-duplication from
   `PDHDOpWaveformSource` but much simpler — flat TTree instead of TH1D
   directories). Selects run/event; routes by `nsamp`; positive polarity;
   keeps OfflineChannel granularity (2 DAPHNE ch per XA) through ophit and
   maps to OpDet at flash stage via an opch-map, exactly like PDHD's 4:1 ganging.
2. **cfg data** under `cfg/pgrapher/experiment/protodunevd/`:
   - `pdvd-opdet-geom.json` — 40 OpDet positions (from mapping/data).
   - `pdvd-opch-map.json` — OfflineChannel → OpDet (56 → 40).
   - `pdvd-spe-templates.json` — per-population templates + channel→template map (§4).
3. **`flash.jsonnet` for protodunevd** — builders mirroring PDHD's with PDVD
   values: `nchan=40`, snippet chain (samples=1024, pre_trigger≈76), full-stream
   chain (`fixed_snr`, `OpRoi` cleaning, `robust_baseline`/`fixed_ped_sigma`),
   saturation knobs, and possibly **separate ophit thresholds for XA vs PMT**
   (very different pulse shapes and noise).
4. **Job configs + drivers** in `pdvd/`: `wct-light-reco.jsonnet` (self-trig),
   `wct-light-fullstream-reco.jsonnet` (cathode), `wct-light-allpd-reco.jsonnet`
   (merged, the production one), `run_light_*.sh` — duplicated from `pdhd/`.

Full-stream specifics to revisit against PDHD's solutions
(`pdhd-fullstream-light-reco.md`): fixed-SNR Wiener (no self-trigger bias),
`OpRoi` HPF+hysteresis ROI cleaning with the ≥4.8 µs post-peak late-light
window, in-ROI pedestal handling (`fixed_ped_sigma`), and 7.5 ms record length
(PDHD was 5.5 ms — memory/FFT cost slightly higher, two lengths to handle).

Output: the standard opflash tensor set (`opflash [nflash, 1+40]`,
`flash_summary`, `ophits`) → `FlashTensorToOpticalPCs{nchan:40}` → future
QLMatching, unchanged schema.

## 4. SPE template strategy (key open item — plan is try-and-compare)

> **RESULT (2026-07-08): see `03_pdvd-spe-template.md`** — data-driven per-channel templates
> win (0.4–2.8 % held-out decon-tail residual vs 18–41 % for the PDHD NP04 templates);
> all 51 live channels demonstrated in `docs/pics/pd_ch*.png`; cathode PE scale provisional.

**larsoft has no measured PDVD SPE calibration.** The VD digitizer fcl
(`opticaldetectormodules_dune.fcl`, `protodunevd_opdigi`) uses:

- XA (Cathode+Membrane): `SPE_NP02_estimate_without_pretrigger.dat` — explicitly
  "FIX ME! Temporary simple PE" (analytic: 10 ADC amplitude, 13 ns rise, 386 ns
  decay; 1025 samples).
- PMT: `PhotonPropagation/PMTResponse/protoDUNEDP_waveform_20180927.txt` — the
  **measured ProtoDUNE-DP PMT response** (same physical PMTs).
- `protodunevd_ophit` runs on raw waveforms (no decon) with a single
  `SPEArea: 410` for all channels — clearly not final for data.

### Candidates per population

**X-ARAPUCA (cathode + membrane; SiPM + DAPHNE, same electronics as PDHD):**

| candidate | pros | cons |
|---|---|---|
| (a) PDHD `SPE_NP04_FBK_2024` / `HPK_2024` (our current PDHD defaults) | measured, same DAPHNE readout/sampling, tail behavior already studied (`pdhd-spe-template-tuning.md`) | PDVD SiPM vendor per module unknown (FBK vs HPK — determine, or try both); NP04 gain/shaping may differ |
| (b) `SPE_NP02_estimate` (larsoft official) | "official" VD placeholder, right rise/decay ballpark | analytic, admittedly temporary; amplitude arbitrary |
| (c) **data-driven from PDVD itself** | ground truth; membrane XAs show a clean 1-PE self-trigger peak (~40–50 ADC) to average; cathode from isolated small pulses in full-stream quiet regions | needs careful isolation cuts; cathode statistics per event limited |

**PMT:**

| candidate | pros | cons |
|---|---|---|
| (a) `protoDUNEDP_waveform_20180927.txt` | measured on these very PMTs | 2018, DP-era digitization — needs resampling to 16 ns and gain check |
| (b) data-driven small-pulse average (late-light tails of snippets) | current gains | no visible 1-PE peak in trigger spectrum; selection is delicate |
| (c) skip deconvolution for PMTs: direct pulse finding + area/SPEArea calibration (the larsoft route) | PMT pulse is only 4–6 ticks wide — deconvolution at 16 ns buys little; simplest robust start | needs per-channel SPEArea gains anyway; less uniform chain |

### Try-and-compare protocol (expect iterations before the final choice)

1. Build `pdvd-spe-templates.json` with **all** candidate templates and a
   switchable channel→template map (the PDHD json format already supports
   multiple templates + `template_index`).
2. Run OpDecon per candidate on the same events; compare:
   - decon residual flatness / post-pulse over- or under-shoot (the PDHD tail
     criterion that motivated the tuned FBK template);
   - **PE quantization** of ophit areas on low-light channels (1-PE ophits
     should land at area ≈ SPEArea, sharp peak = good template+gain);
   - snippet-chain internal closure and full-stream late-light behavior
     (cathode has no snippets, so validate via decon closure there);
   - hit-splitting behavior on pile-up.
3. Per-channel **gain (SPEArea) calibration table** regardless of template
   choice — the 1-PE peak position per channel (membrane XAs directly; PMTs
   from late-light fits; cathode from full-stream isolated pulses). This is
   what normalizes PE across the three populations, which matters for flash
   composition and Q-L matching.
4. Document the comparison in a dedicated md (like `pdhd-spe-template-tuning.md`)
   and freeze defaults; keep alternates selectable via `spe_file`.

## 5. OpHit / OpFlash plan — flash formation

> **RESULT (2026-07-08): see `05_pdvd-flash-dt.md`** — all four PD groups coincide at the
> tick level (68 % cores ≤ ±0.6 µs, per-type offsets ≤ 64 ns); keep `bin_width = 1000 ns`
> (PDHD value) and the single all-PD flash. Filter settings: see `04_pdvd-light-filter.md`.

Constraints particular to PDVD: three PD populations, two optically-coupled
drift volumes (double-sided cathode XAs see both), very different per-PD
response (XA slow/PMT fast, different gains), and the flash output feeds
Q-L matching where **predicted-vs-measured PE per PD** is what matters.

Options brainstormed:

1. **One flash across all 40 OpDets per time coincidence** (PDHD all-PD chain
   with `group_by_side=false`). A physical interaction anywhere lights PDs in
   both volumes within the ~µs binning anyway (scintillation is prompt and the
   cathode XAs integrate both faces), so a single time-clustered flash is the
   natural object. Q-L matching then compares the full 40-PD PE pattern against
   the light-model prediction — the *pattern*, not the flash partition, carries
   the volume information. **Recommended starting point.**
2. **Per-volume flashes** (top: 4 top membrane XAs [+cathode]; bottom: PMTs +
   4 bottom membrane XAs [+cathode]) — the analog of PDHD `group_by_side`. But
   PDHD's sides are optically isolated by an opaque cathode; PDVD's volumes are
   not — the shared double-sided cathode XAs would be double-counted or need an
   arbitrary split. Only worth revisiting if single-flash matching proves
   ambiguous for top-vs-bottom activity; implement later as a togglable
   `group_by_volume` knob (default OFF, per the toggleable-behavior convention).
3. **larsoft `OpFlashFinderVerticalDrift`-style plane-aware clustering**
   (cathode/membrane distance criteria). Noted for reference, but it is
   explicitly "upper volume only" and does not handle the PMTs at all — our
   option 1 already covers both volumes and all three populations.

Supporting decisions:

- Keep a **per-population PE breakdown** (cathode/membrane/PMT sums) in the
  flash diagnostics — cheap and invaluable for hand-scans and for judging
  whether option 2 is ever needed.
- Retune `OpFlashFinder` quality knobs for 40 PDs: PDHD's `min_fired_pds=5`,
  `min_total_pe=20`, `min_fired_pe=1.0` were tuned for 160 PDs; PDVD has far
  fewer PDs per volume (start lower, e.g. 2–3 fired PDs, and scan).
- **Cross-population timing alignment check** before flash formation: decon
  should align XA and PMT hit times, but PMT transit time and the very
  different pulse shapes need a verification pass (e.g. coincident cosmic
  hits, XA-vs-PMT peak-time residuals).
- Q-L matching consequence (later, separate task): the toolkit
  `SemiAnalyticalModel` currently supports flat anode/cathode XAs and dome
  PMTs but not lateral (membrane) PDs, and the only photon library around is
  the geometrically wrong ProtoDUNE-SP v7 one. The flash schema (per-PD PE)
  already supports whatever model comes; light-model work does not block
  reconstruction.

## 6. Timing / trigger reference

> **RESULT (2026-07-08)**: the `timestamp` branch is in **microseconds** on one common
> clock (16 ns quantization); the full-stream record starts at its timestamp and the
> snippet timestamps are trigger times on the same clock, spanning the same 7.5 ms window
> (`05_pdvd-flash-dt.md`). Remaining open: absolute light↔charge offset (`offset_us` analog).

- All WCT light times should be trigger-relative ns, like PDHD (`offset_us`
  convention in `OpFlashFinder`).
- The TTree `timestamp` branch (~5.74e9 tick scale) is the DTS-like absolute
  clock; the full-stream records (one per event, fixed window) give a clean
  per-event anchor: snippet times relative to the full-stream record start
  reproduce the in-window time.
- The PDVD light↔charge offset (PDHD's −250 µs analog) is **not yet
  established** for these extractions — needs a dedicated study once ophits
  exist (e.g. cosmic charge-track x/t0 vs flash times, or metadata from the
  raw HDF5 trigger records). `pdvd/docs/01_photon-detector-chain.md` §timing
  carries the −250 µs expectation from the fcl chain.

## 7. Implementation roadmap

| phase | content | verify |
|---|---|---|
| 0 | Data prep: per-event indexing of the 5 light files; light↔charge event matching table (raw event numbers; 039349 3-file overlap check) | matching table spot-checked vs .log files |
| 1 | `PDVDOpWaveformSource` + `pdvd-opdet-geom.json` + `pdvd-opch-map.json` | round-trip: source → frame dump reproduces TTree waveforms |
| 2 | Self-trigger chain (`flash.jsonnet` protodunevd + `wct-light-reco.jsonnet`): OpDecon+OpHitFinder on membrane XA + PMT with initial templates (PDHD FBK/HPK for XA, DP waveform for PMT) | decon/ophit sanity on a hand-picked event; PE spectra |
| 3 | Full-stream chain for cathode XAs (fixed_snr + OpRoi + robust baseline; two record lengths) | ROI coverage plots, late-light window check (PDHD §7 criteria) |
| 4 | SPE try-and-compare (§4) + per-channel gain table; freeze defaults; write `pdvd-spe-template-comparison.md` | quantization/closure criteria of §4 |
| 5 | All-PD merged chain (OpHitMerge → OpFlashFinder nchan=40, single flash, quality-knob scan) | flash multiplicity/PE audit vs charge activity per event (PDHD run27980-status §6 style) |
| 6 | Timing study: trigger reference, light↔charge offset | cosmic crossing-track t0 consistency |
| 7 | Bee/ql_scan display + Q-L matching (needs VD light model: lateral PDs, double-sided cathode, Xe) | — (separate task) |

Validation note: unlike PDHD there is **no LArSoft reference reconstruction**
for PDVD (no ophit/opflash ever run officially), so validation is internal
(closure, PE spectra, flash-vs-charge sanity) plus cross-checks with the
colleague's raw-waveform scan (dead/low channel list already reproduced).

## 8. Open questions

1. **SiPM vendor (FBK vs HPK) per PDVD XA module** — determines which PDHD
   template is the better prior; ask PDS colleagues / check XA production docs
   (PDE papers: arXiv:2502.05042, 2511.12328). Meanwhile: try both.
2. Polarity of jjo's extraction confirmed positive on run 039252 — verify the
   other four files.
3. Run 039349's three light files: disjoint event streams or duplicates?
4. PMT gain calibration route (no 1-PE peak in trigger spectrum): late-light
   fits vs DP-era gain values.
5. Trigger/readout-window relation between the 7.5 ms light full-stream and
   the charge readout — needed for the offset_us convention.

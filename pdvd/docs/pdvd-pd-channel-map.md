# PDVD photon-detector channel map: location, type, efficiency, PE share

Reference table for the 40 WCT flash-chain OpDet channels (== toolkit
calib-dump `pe` array order == QLMatching channel order), assembled while
investigating why the bottom/z-wall PMTs contribute so little flash PE
compared to the X-ARAPUCAs (run 39252, evt 298567 and 120-event survey
across 039252/039253/039349). Source data: channel roles + `VUVEfficiency`
from `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`; positions
from `cfg/pgrapher/experiment/protodunevd/pdvd-opdet-geom.json` (data-derived,
from the `raw_waveform` TTree x/y/z branches, run 039252; matches the v5
as-built GDML). Companion docs: `pdvd-questions-dune.md` (open DUNE asks,
incl. the uncoated-PMT and gain-calibration items below), `pdvd-qlmatching.md`,
`pdvd-photon-model.md`.

## Full channel table

Position in cm, detector frame (cathode at x=0, bottom anode wire plane at
x=-341.55, top anode at x=+341.55; same frame as the TPC box definitions in
`wire-cell-bee3/events/static/js/bee/physics/experiment.js` `ProtoDUNEVD`).

| ch | type | coating | eff (VUVEfficiency, Xe/175nm) | x | y | z | status |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | membrane XA (top vol) | PTP | 0.03 | 305.60 | 417.61 | 149.65 | live |
| 1 | membrane XA (top vol) | PTP | 0.03 | 305.60 | -417.61 | 149.65 | live |
| 2 | membrane XA (top vol) | PTP | 0.03 | 229.00 | 417.61 | 149.65 | live |
| 3 | membrane XA (top vol) | PTP | 0.03 | 229.00 | -417.61 | 149.65 | live |
| 4 | cathode XA | PTP | 0.03 | 0.00 | 123.85 | 258.52 | live |
| 5 | cathode XA | PTP | 0.03 | 0.00 | -213.15 | 258.52 | live |
| 6 | cathode XA | PTP | 0.03 | 0.00 | 290.35 | 187.28 | live |
| 7 | cathode XA | PTP | 0.03 | 0.00 | -46.65 | 187.28 | live |
| 8 | cathode XA | PTP | 0.03 | 0.00 | 42.60 | 112.03 | live |
| 9 | cathode XA | PTP | 0.03 | 0.00 | -213.15 | 112.03 | live |
| 10 | cathode XA | PTP | 0.03 | 0.00 | 209.10 | 40.77 | live |
| 11 | cathode XA | PTP | 0.03 | 0.00 | -127.90 | 40.77 | live |
| 12 | membrane XA (bottom vol) | PTP | 0.03 | -201.10 | 417.61 | 149.65 | live |
| **13** | **membrane XA (bottom vol)** | **none (no WLS)** | 0.03 | -201.10 | -417.61 | 149.65 | live (Xe-only; see anomalies) |
| 14 | z-wall PMT | TPB | 0.12 | -205.90 | 221.00 | 408.99 | live |
| 15 | z-wall PMT | PEN | 0.036 | -205.90 | -221.00 | 408.99 | live |
| 16 | z-wall PMT | TPB | 0.12 | -205.90 | 256.00 | -96.12 | live (dim anomaly, see below) |
| 17 | z-wall PMT | TPB | 0.12 | -205.90 | -221.00 | -109.69 | live |
| 18 | membrane XA (bottom vol) | PTP | 0.03 | -277.70 | 417.61 | 149.65 | live |
| 19 | membrane XA (bottom vol) | PTP | 0.03 | -277.70 | -417.61 | 149.65 | live |
| 20 | z-wall PMT | TPB | 0.12 | -281.70 | 221.00 | 408.99 | live |
| 21 | z-wall PMT | PEN | 0.036 | -281.70 | -221.00 | 408.99 | live |
| 22 | z-wall PMT | TPB | 0.12 | -281.70 | 256.00 | -96.12 | live |
| 23 | z-wall PMT | TPB | 0.12 | -281.70 | -221.00 | -109.69 | live |
| 24 | bottom PMT | PEN | 0.036 | -336.47 | 170.00 | 455.65 | **dead (no DAPHNE readout)** |
| 25 | bottom PMT | PEN | 0.036 | -336.47 | 0.00 | 455.65 | live |
| 26 | bottom PMT | PEN | 0.036 | -336.47 | -170.00 | 455.65 | live |
| 27 | bottom PMT | PEN | 0.036 | -336.47 | 0.00 | 353.65 | **dead (no DAPHNE readout)** |
| 28 | bottom PMT | PEN | 0.036 | -336.47 | 170.00 | 353.65 | **dead (no DAPHNE readout)** |
| 29 | bottom PMT | PEN+Q | 0.036 | -336.47 | -170.00 | 353.65 | live (Xe-only; see anomalies) |
| 30 | bottom PMT | PEN | 0.036 | -336.47 | 405.30 | 217.75 | live |
| 31 | bottom PMT | PEN | 0.036 | -336.47 | -405.30 | 217.75 | live |
| **32** | **bottom PMT** | **none (uncoated)** | **0.0** | -336.47 | 405.30 | 149.65 | **masked (official eff=0; see anomalies)** |
| **33** | **bottom PMT** | **PEN** | 0.036 | -336.47 | -405.30 | 149.65 | live (dim anomaly, see below) |
| 34 | bottom PMT | PEN | 0.036 | -336.47 | 170.00 | -54.35 | **dead (no DAPHNE readout)** |
| 35 | bottom PMT | PEN | 0.036 | -336.47 | 0.00 | -54.35 | live |
| 36 | bottom PMT | PEN | 0.036 | -336.47 | -170.00 | -54.35 | live |
| 37 | bottom PMT | PEN | 0.036 | -336.47 | 170.00 | -156.35 | live |
| 38 | bottom PMT | PEN | 0.036 | -336.47 | 0.00 | -156.35 | live |
| 39 | bottom PMT | PEN+Q | 0.036 | -336.47 | -170.00 | -156.35 | live (Xe-only; see anomalies) |

Dead channels 24/27/28/34 have no `raw_waveform` entry; their positions
above are cosmetic estimates (mirrored from the live neighbour in the same
grid row), not surveyed.

## PE-share by type (measured, 120 events across runs 039252/039253/039349)

| type | event-integrated share of total measured PE |
|---|---:|
| cathode XA (8 ch) | 87.6 - 90.2% |
| membrane XA (8 ch) | 8.9 - 10.0% |
| z-wall PMT (8 ch) | 0.5 - 1.8% |
| bottom PMT (16 ch) | 0.4 - 1.4% |

Per-channel quantum/WLS efficiency is **not** the driver of this gap: TPB
PMTs (eff 0.12) and PEN PMTs (eff 0.036) are quoted equal or *better* than
the X-ARAPUCAs (eff 0.03). The gap is dominated by light-collection
geometry — X-ARAPUCAs are large-area WLS light-trap panels close to where
beam-track light is generated (cathode plane, y-walls), while the PMTs are
small photocathode disks that are also geometrically remote (z-wall PMTs
sit beyond the active/instrumented z range; bottom PMTs sit ~340 cm away
behind the far anode).

## Known anomalies (open / unresolved)

- **ch 32 (uncoated bottom PMT)**: official `PDVD_PDS_Mapping_v04152025`
  gives `eff_Ar = eff_Xe = 0` (physically sensible — no WLS coating to
  convert VUV scintillation light) and it is masked out of the QL fit.
  **But data show it responding at 0.55-1.85x its PEN-coated neighbours'
  level on big flashes** — i.e. clearly not blind. Open DUNE question
  (`pdvd-questions-dune.md` §3); unmask if/when an official nonzero 175 nm
  efficiency appears for it.
- **ch 13 (membrane XA, no WLS)** and **ch 29/39 (PEN+Q bottom PMTs)**:
  officially Ar-blind at 128 nm (masked under the old Ar model), but
  respond at peer level under the Xe/175 nm model — this comparison is
  part of what confirmed the Xe/175nm library switch (`pdvd-questions-dune.md`
  §3); now unmasked/live.
- **ch 16 and ch 33**: both normally-coated (TPB and PEN respectively) but
  measure ~50x dimmer than their same-type, same-coating peers. Not a
  coating/efficiency-model issue — flagged as a likely hardware/gain fault,
  handled by the per-event dynamic `auto_mask` rather than a static
  correction. Notably ch33 sits in the same bottom-PMT grid row (z=149.65,
  outer y) as ch32, so that one row has two unrelated problem channels
  side by side.
- The relative PMT-vs-XA (and even cathode-vs-membrane XA) normalization
  is not independently calibrated: a data-driven per-channel gain fit found
  the group-to-group meas/pred ratio "measures the library as much as the
  gain" and flips substantially (0.62 to 2.8) between the Ar/128nm and
  Xe/175nm libraries (`pdvd-questions-dune.md` §2). Absolute cross-type
  scale still carries real uncertainty.

## Related

- `wire-cell-bee3` `ProtoDUNEVD` (`experiment.js`) now renders all 40
  channels at these real surveyed positions (previously a synthetic
  placeholder grid lumped z-wall PMTs in with the bottom PMTs at a single
  fake plane) — local, uncommitted change as of this writing, needs a live
  render check + BNL redeploy before it's visible where users actually look.
- [[project_pdvd_qlmatching]], [[project_bee3_protodune_geometry]] (toolkit
  memory).

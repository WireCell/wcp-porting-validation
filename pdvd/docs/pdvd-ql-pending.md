# PDVD Q/L matching — pending work after the trigger offset is determined

Status 2026-07. The PDVD Q/L machinery is **built and wired end-to-end**
(joint shared-flash QLMatching, cfg, driver, calib dumps, hand-scan viewer —
see `pdvd-qlmatching.md`), but **production tuning is blocked on one number**:
the light-vs-charge trigger offset. This file records exactly what is known,
what is needed from outside, and the ordered checklist to run once it lands.

## 1. The blocker: light↔charge trigger offset

Every drift-position correction in the matcher is
`x_true = x_raw + sign_offset · (t_flash + T) · v`. With `T` unknown, absolute
positions (and hence containment cuts, boundary flags, and match quality) are
wrong by `sign · T · v` — at 1 ms of offset that is ~1.6 m, i.e. everything.

### What is established (do not re-derive)

- **Charge side is internally coherent**: bottom-vs-top `t_mid` of gold
  two-volume through-going muons agree within ±30 µs.
- **Light side is internally coherent**: all 16 cathode full-stream records
  share one start per event (spread 0.00 µs, `ql_light_calib/dump_light_t0.py`);
  the light chain's t=0 is that earliest record start.
- **The statistical (A/C-crosser) route saturates**: with ~150 flashes ≥150 PE
  per 5–7.5 ms window (~50 µs mean gap) and ~±20 µs anchor resolution, even a
  TRUE constant offset yields only 2–3 σ on this sample — which is what we see.
  A per-run constant is therefore **neither confirmed nor excluded**. Beware
  the circular check: "N/M events have a flash within ±30 µs of T" is ~60%
  satisfied by accident at this density.
- Best current candidate: run 039252, 7 gold two-volume anchors → **T ≈ −2855 µs
  at 3.2 σ** over event-mixed accidentals (suggestive only). Runs 039253/039349
  have 2 gold anchors each — no power.
- One demonstrated anomaly: 039349 evt 19549 (6 crossers) sits ~4 ms from every
  run-level candidate — per-event window-placement shifts are possible.
- **PDHD precedent**: its offset (249.808 µs, constant) was never fitted — the
  raw extraction carried a per-event `trigoff/trigger_offset` tree from DAQ
  timestamps (`pdhd/run_light_evt.sh:76-86`); the constancy was *measured*.

### What is needed (external ask — full question list: `pdvd-questions-dune.md`)

Per-event DAQ timestamps on the charge side. Either of:

1. **Charge readout-window start timestamp** per event, on the DTS clock
   (currently the charge frame extraction writes `tickinfo time = 0`), or
2. a PDHD-style **trigger record** (`trigoff` tree) added to the PDVD `rawwf`
   extraction (light files at `/nfs/data/1/jjo/data/PDVD/` carry only the
   `raw_waveform` tree today).

The light side needs nothing: `ql_light_calib/dump_light_t0.py` already dumps
the per-event light t0 (absolute µs, 16 ns DTS clock) from the `timestamp`
branch.

### Once the timestamps exist

1. `offset_us(event) = charge_window_start_us − light_t0_us` — pure arithmetic.
2. Plumb it: per-event values belong in the **opflash metadata `offset_us`**
   (regenerate the light archives, or patch metadata); any residual per-run
   constant goes in `pdvd/data/ql_trigger_offset.txt` (already read by
   `run_clus_evt.sh`; `PDVD_TRIGGER_OFFSET_US` env overrides for tests).
3. Cross-check with physics: rerun `ql_light_calib/fit_trigger_offset.py` on
   the diagnostic dumps — matched full-drift crossers must now land with edges
   at anode/cathode (residual ≈ 0, well inside ±20 µs · v ≈ 3 mm…3 cm).
4. Answer the constancy question for the record (per-run constant vs per-event
   jitter) and note it here and in `pdvd-qlmatching.md`.

## 2. Post-offset checklist (in order)

These are Phases 6–7 of the integration plan, currently parked:

1. **Flip diagnostics off**: `require_containment=true`, production
   `flash_minPE` (≈25), drop `PDVD_QL_DIAG`.
2. **Renormalize QtoL** (now 1.0 placeholder): median pred/meas PE over
   good-KS (`ks_dis` small) auto-selected bundles from fresh `-calib` dumps of
   a few events; set in `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`.
3. **Enable `reject_overpred`** (loose) only after QtoL is sane.
4. **Retune the chi2/PE-error knobs** (`chi2_pmt_excess`, later
   `pe_err_on_pred`/`pe_err_lowpe_*`) — PDHD values are placeholders; PDVD's
   89%-cathode PE topology differs.
5. **Validate the dead-PD self-check** (`auto_mask`, `auto_mask_same_type`):
   drop known-dead ch24 from the static `ch_mask` on one event — auto-mask must
   catch it; then `grep QLAUTOMASK` over the batch for false positives.
6. **Batch**: all 120 events of 039252/039253/039349 with `-calib -op`
   (039324 stays charge-only until its light raw is staged). Watch wall/RSS of
   the joint LASSO.
7. **Hand scan** with `pdvd/ql_scan` (ready; port 5016) → labels → metric/
   PE-error tuning, PDHD-style.

## 3. Deferred by design (independent of the offset)

- **`flag_at_cathode` is inert**: set on bundles + exported in dumps
  (`at_cathode`), but no behavioral consumer yet. Design its tuning (cathode
  PDs see 89% of light — candidate for its own relax/weight path) after the
  first hand-scan round.
- **Rescues are off**: `empty_rescue`/`cluster_rescue` are not
  shared-flash-aware (matcher warns + skips under `shared_flash`). Port them
  if the batch shows stranded big clusters.
- **TCO z-wall PMTs (ch 14–17, 20–23) excluded from `pd_walls`** (~1% of PE);
  one-line cfg addition if hand scan shows z-wall-hugging mismatches.
- **Semi-analytical light model** (`PDVD_LIGHT_MODEL=semi`): membrane XAs are
  appended to `ch_mask` there (port cosine wrong for y-normal PDs); library
  model is the default and the tuned path.
- **Run-dependent PD masks**: like PDHD, static `ch_mask` can go stale — read
  effective dead channels from the opflash PE content per run when new runs
  are added.
- **DAPHNE↔module pairing check** on Arapuca data still owed
  (`pdvd-photon-model.md`).
- **039324**: no raw light file staged; charge-only reprocess done; run QL
  when light lands.

## 4. Pointers

| What | Where |
|---|---|
| Questions for Jay / DUNE (T0, cathode SPE, Xe/Ar library, …) | `pdvd/docs/pdvd-questions-dune.md` |
| Offset fit (statistical cross-check) | `pdvd/ql_light_calib/fit_trigger_offset.py` |
| Light t0 dump (per event, DTS µs) | `pdvd/ql_light_calib/dump_light_t0.py` |
| Per-run offset table (empty on purpose) | `pdvd/data/ql_trigger_offset.txt` |
| Driver offset plumbing | `pdvd/run_clus_evt.sh` (TRIGGER_OFFSET_US block) |
| Matching chain + knobs | `pdvd/docs/pdvd-qlmatching.md` |
| Hand-scan viewer | `pdvd/ql_scan/` (README there) |
| Diagnostic dumps used for the fit | `pdvd/work/0392*_*/calib-evt*.json`, `PDVD_QL_DIAG=1` |

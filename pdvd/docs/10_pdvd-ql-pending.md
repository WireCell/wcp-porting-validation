# PDVD Q/L matching — pending work after the trigger offset is determined

Status 2026-07-10. The PDVD Q/L machinery is **built and wired end-to-end**
(joint shared-flash QLMatching, cfg, driver, calib dumps, hand-scan viewer —
see `09_pdvd-qlmatching.md`), and **the trigger-offset blocker is RESOLVED**
(§1 below records how). The remaining work is the §2 tuning checklist.

## 1. RESOLVED: light↔charge trigger offset (2026-07-10)

The external ask was answered: jjo reprocessed the light extraction and the
`*_rawwf.root` files now carry a PDHD-style **`trigoff/trigger_offset` tree**
with per-event DAQ timestamps (tc_us, charge_tde_us, charge_bde_us, fs_t0_us,
light_t0_us, …), plus a per-event text table
(`pdvd/data/jjo_triglight_offsets.txt`, summary `jjo_triglight_summary.txt`).

**Measured answer to the constancy question (§1 step 4 of the old plan):**

- The **light full-stream window is trigger-locked to ±0.3 µs** (trigger −
  fs_start = 5000.4 / 5000.6 / 2500.5 µs for 039252/039253/039349 — per-run
  DAQ constants). The old "is the light window at a fixed offset?" question:
  yes, it was the **charge** window that floats.
- The **charge windows float per event (±15 µs, σ 9–12 µs)** and **per
  CRATE**: the TDE (top CRPs) and BDE (bottom CRPs) windows each open on
  their own 64-sample frame boundary, up to **32 µs apart** (≈5 cm of
  drift). So the offset is per event AND per drift volume — a per-run
  constant can never beat ±15 µs; the old statistical fit stalled exactly
  as expected. (The 039349 evt-19549 "4 ms outlier" was an artifact of the
  statistical method; its measured offset is dead-center of the run band.)
- Timing units caveat: the raw BDE charge is sampled at 512 ns and the TDE
  at 500 ns, but the **first step of charge signal processing resamples BDE
  512→500 ns**, so the processed charge frames (what clustering consumes)
  are uniformly **500 ns/tick with tick 0 = that crate's window start** —
  the stamped offsets refer to the window START, which resampling preserves.
  (jjo's "divide by 0.512 for BDE" recipe step applies to RAW RawDigits
  only, not to our SP frames.)

**Implementation (all in `./work`, no dependency on jjo's files downstream):**

1. `run_light_evt.sh` reads the trigoff tree from the SAME rawwf file it
   already opens, replicates the chain's tick origin exactly
   (min over ALL records of `round(ts·62.5) − 64 ticks` for self-trigger
   snippets — `PDVDOpWaveformSource`'s rule; the earliest record is often a
   snippet, hence a −1.024 µs shift vs the raw min timestamp), and stamps

       offset_bot_us = chain_t0 − charge_bde_us   (bottom volume, BDE)
       offset_top_us = chain_t0 − charge_tde_us   (top volume, TDE)

   (negative, ≈ −2.1..−2.5 ms; ADD to a flash time to land on that crate's
   charge axis) into the opflash archive metadata. 40-bit DTS rollover
   (17 592.19 s) is wrapped. All 120 events regenerated and verified
   consistent with jjo's independent table (120/120; crate skew and anchor
   agree to float precision).
2. `run_clus_evt.sh` reads `offset_bot_us`/`offset_top_us` (legacy archives
   fall back to scalar `offset_us`), adds the per-run residual from
   `data/ql_trigger_offset.txt` (still empty), and passes both to
   `wct-clustering.jsonnet` → QLMatching `trigger_offsets=[bot, top]` and
   clus.jsonnet `trigger_offset(_top)` (per-volume x_t0cor).
   `PDVD_QL_DIAG=1` still forces 0 (offset-hunt mode); **`PDVD_QL_DIAG=2`**
   keeps the measured offsets with the loosened diagnostic knobs (closure).
3. Byte-identity: PDHD 29107 evt0 (15/15 archives), SBND data evt686, PDVD
   no-QL, and PDVD QL `trigger_offsets=[0,0]` vs scalar — all identical.
   Toolkit commit `8a765d5e`.

The subsections below are kept for the record of HOW the blocker was
diagnosed (the statistical route and its limits).

### The old blocker, for the record: light↔charge trigger offset

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

### What was needed (external ask — DELIVERED 2026-07-10, see top of §1)

Per-event DAQ timestamps on the charge side. Either of:

1. **Charge readout-window start timestamp** per event, on the DTS clock
   (currently the charge frame extraction writes `tickinfo time = 0`), or
2. a PDHD-style **trigger record** (`trigoff` tree) added to the PDVD `rawwf`
   extraction (light files at `/nfs/data/1/jjo/data/PDVD/` carry only the
   `raw_waveform` tree today).

The light side needs nothing: `ql_light_calib/dump_light_t0.py` already dumps
the per-event light t0 (absolute µs, 16 ns DTS clock) from the `timestamp`
branch.

### Once the timestamps exist (all done 2026-07-10, per-crate variant)

1. ✓ `offset_{bot,top}_us(event) = charge_{bde,tde}_us − chain_t0_us` (sign
   flipped into "add to flash time" convention when stamped).
2. ✓ Plumbed into the opflash metadata (`offset_bot_us`/`offset_top_us`;
   the scalar `offset_us` stays 0/legacy); the per-run residual table
   `pdvd/data/ql_trigger_offset.txt` remains, empty.
3. ✓ Physics cross-check — but NOT via the crosser fit: at ~50 µs flash
   spacing the crosser-anchor scan is accidental-dominated even after the
   offsets (its peak survives event-mixing → meaningless, as its docstring
   always warned). The decisive closure is the **beam-trigger flash test**
   (`ql_light_calib/check_trigger_flash.py`): these are CTB beam-triggered
   runs, so each event must show a flash at the trigger position
   `tc_us − charge_bde_us` on the folded axis. Result: **99/120 events have
   a ~2000 PE flash within ±5 µs of it, residual median −0.9 µs** (inside
   the 1 µs OpFlash binning; ~4%/event accidental probability). The mapping
   is validated end-to-end at the sub-µs level. Gold two-volume pairs are
   consistent within their ±20 µs anchor resolution (6 clean pairs — too
   few to resolve the 15 µs mean crate skew, which rests on the DAQ
   timestamps like everything else).
4. ✓ Constancy answered: light window trigger-locked per run; charge windows
   float per event and per crate (top of §1).

## 2. Post-offset checklist — Phase 6 DONE 2026-07-10, Phase 7 batch running

1. ✓ **Diagnostics off**: production = `require_containment=true`,
   `flash_minPE=25` (the cfg defaults; production mode is simply running
   without `PDVD_QL_DIAG`).  Containment is inert in practice (all bundles of
   the smoke event contained at the measured offsets).
2. ✓ **QtoL renormalized — but NOT by the recipe above.**  The planned
   "median over good-KS auto-selected bundles" gave QtoL≈40 and is **WRONG by
   ~350x**: with a mis-scaled QtoL the LASSO is amplitude-inert (bundle
   strengths sit at their ~1.0 initial value; the per-flash background column
   absorbs the fit), so the auto-selection it feeds on is dominated by
   accidentals whose flash is ~40x brighter than the prediction.  The correct
   anchor is external ground truth: the **beam-trigger gold pairs** (flash at
   `tc_us − charge_bde_us`, cluster = largest-pred bundle on it; 80 pairs) give
   **QtoL = 0.11** ([16,84]=[0.03,0.25]) — the library+official-eff model
   OVER-predicts ~10x, flat across PD groups (so the efficiencies are not
   double-counted).  `ql_light_calib/fit_qtol_gold.py`.  With QtoL=0.11 the
   smoke event goes from 15 junk matches to 90 LASSO-driven matches, 19/27
   big clusters.
3. **`reject_overpred` stays OFF** (revised decision): the gold-pair scatter
   is still ~x3 around the scale and the per-channel PE calibration is raw;
   data-tuned ceilings need hand-scan GT (SBND 2.9/4.3, PDHD 3/10 for
   reference).
4. ✓ **chi2/PE-error retuned on the gold pairs**: `pe_err_on_pred=true` with
   floor/frac/lowpe_frac/lowpe_knee = 2.0/0.60/2.0/10.0 (wider than PDHD's
   0.3/0.40/1.55/5.5 because the per-channel PE scale is uncalibrated — the
   gold per-channel meas/pred spans x13 WITHIN the cathode-XA group alone).
   Gold median chi2/ndf 9.7, 71% under the hc ladder ceiling.  NOTE: gold-pair
   KS median is 0.45 with 0% under the ladder KS ceilings — on PDVD round 1
   the KS ladder cannot pass true matches; selection is LASSO-amplitude-driven.
5. ✓ **Dead-PD self-check validated**: with ch24 dropped from the static mask
   QLAUTOMASK catches it (after retuning `auto_mask_pe_bright` 20→10 and
   `min_contrast` 2→3 — bottom-PMT neighbour medians almost never reached 20;
   emulation over the 120 dumps: ch24 caught in 91/120 events, healthy
   channels <6%).  Side finding: **ch16 and ch33 are genuinely dim** (event-max
   PE median ~5.6, ~50x below peers; auto-masked when quiet — safe direction);
   added to the per-run bad-channel question in `11_pdvd-questions-dune.md`.
6. **Batch (RUNNING)**: all 120 events of 039252/039253/039349 with
   `-calib -op` production knobs (039324 stays charge-only until its light
   raw is staged).  Then `grep QLAUTOMASK` over the batch logs.
7. **Hand scan** with `pdvd/ql_scan` (ready; port 5016) → labels → then the
   deferred items: per-channel `measured_pe_scale` refit (beam-gold fit exists
   but is single-topology), reject_overpred ceilings, chi2/KS ceilings, QtoL
   refinement.

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
  (`08_pdvd-photon-model.md`).
- **039324**: no raw light file staged; charge-only reprocess done; run QL
  when light lands.

## 4. Pointers

| What | Where |
|---|---|
| Questions for Jay / DUNE (T0 ANSWERED; cathode SPE, Xe/Ar library, …) | `pdvd/docs/11_pdvd-questions-dune.md` |
| Offset closure check (statistical cross-check) | `pdvd/ql_light_calib/fit_trigger_offset.py` |
| Chain light-t0 dump (per event, DTS µs) | `pdvd/ql_light_calib/dump_light_t0.py` |
| Per-run RESIDUAL offset table (empty; per-event values live in the opflash metadata) | `pdvd/data/ql_trigger_offset.txt` |
| jjo's per-event timestamp table + summary | `pdvd/data/jjo_triglight_offsets.txt`, `jjo_triglight_summary.txt` |
| Driver offset plumbing (per-crate) | `pdvd/run_clus_evt.sh` (TRIGGER_OFFSET_{BOT,TOP}_US block) |
| Light-pass offset stamping | `pdvd/run_light_evt.sh` (trigoff block) |
| Matching chain + knobs | `pdvd/docs/09_pdvd-qlmatching.md` |
| Hand-scan viewer | `pdvd/ql_scan/` (README there) |
| Diagnostic dumps used for the fit | `pdvd/work/0392*_*/calib-evt*.json`, `PDVD_QL_DIAG=1` |

# PDVD charge–light (Q/L) matching

Status 2026-07-10: **built, wired, byte-identity verified; trigger offset
RESOLVED** (per-event per-crate DAQ timestamps, `pdvd-ql-pending.md` §1) —
remaining work is the tuning checklist (`pdvd-ql-pending.md` §2). Companion
docs: `pdvd-light-chain.md` (raw→OpFlash), `pdvd-photon-model.md` (visibility
library), and (toolkit repo) `match/docs/qlmatching-code.md` §2a for the C++
knob reference.

## 1. Why PDVD needed matcher changes

PDHD/SBND have per-drift-side flash lists and per-side QLMatching nodes. PDVD
does not: the 8 cathode X-ARAPUCAs (x≈0, double-sided) see BOTH drift volumes
and the light chain produces **one all-PD flash list** (40 channels). Three
consequences, implemented as default-OFF C++ knobs (commit `e06ea900`, library
backend `f52c4a47`; PDHD/SBND byte-identical — verified on PDHD 29107 evt0 all
15 mabc archives, SBND data evt686, PDVD no-QL abtest events):

- **`shared_flash`** — one joint LASSO per flash over candidate bundles from
  BOTH drift volumes (a per-side fit would let each side independently absorb
  the full PE, biasing cathode-crossers). The two sides enter ONE
  `QLMatching` node (`matching_joint`, nin=2).
- **`opdet_all_volumes`** — keep all 40 PDs on both drift-side runs (the
  legacy per-TPC cathode-x split would halve every flash).
- **`vd_surface_flags`** — `flag_close_to_PMT` re-targeted at PD-bearing
  surfaces: bottom anode → bottom PMTs (ch 24-39), ±y walls → that wall's
  membrane XAs; the chi2 relaxation applies **only to the triggering
  surface's channels** (`TimingTPCBundle::relax_channels`). The cathode end
  sets the new inert `flag_at_cathode` (calib-dump key `at_cathode`;
  behavioral treatment designed after the first hand scan — cathode PDs
  collect ~89% of the light).
- Plus **`auto_mask_same_type`** (dead-PD self-check with a same-type
  neighbour pool — XA vs PMT efficiencies differ ~4×) and
  **`light_model:'library'`** (v5 PDFastSimANN visibility sampled on a 10 cm
  grid; the fitted semi-analytical JSON is the fallback, with membrane XAs
  masked there).

All PDVD knob values and per-channel roles/efficiencies live, with rationale,
in `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet` (toolkit repo).

## 2. Graph and driver

`pdvd/wct-clustering.jsonnet` TLAs: `do_qlmatch`, `opflash_input` (full path
to the light chain's `work/<RUN6>_light<EVENTNO>/opflash_pdvd-wct.tar.gz`),
`calib`, `save_opflash`, `trigger_offset_us`, `readout_window_ticks`,
`light_model`, plus diagnostic loosening knobs. With `do_qlmatch` the QL
branch sits between clustering stage 3 (per-drift-group) and stage 4
(premerged all-TPC): per side an `opflash_source` (both read the SAME
archive) → `flash_attach` (2→1 with that side's group cluster tree) →
the joint `QLMatching` → stage 4 with `use_flash_t0` + `x_t0cor`.

Driver `pdvd/run_clus_evt.sh`:

```bash
./run_clus_evt.sh [-noq|-calib|-op] <run> <idx|all>   # PDVD_QLMATCH=0 to disable
```

- Charge work dirs are INDEX-named (`work/039252_0`) but light dirs are keyed
  by ART EVENT NUMBER — the driver bridges via the event number parsed from
  the cluster tarball, and hard-errors if the opflash metadata `event`
  disagrees. Missing light archive → automatic fallback to no-QL (e.g. run
  039324 until its raw light is staged).
- Trigger offsets are PER CRATE and PER EVENT (2026-07-10):
  `TRIGGER_OFFSET_{BOT,TOP}_US` = opflash metadata `offset_bot_us` (BDE,
  bottom volume) / `offset_top_us` (TDE, top volume — the two crates open
  their windows up to ~32 µs apart), stamped by `run_light_evt.sh` from the
  rawwf `trigoff/trigger_offset` tree, + the per-run residual from
  `data/ql_trigger_offset.txt` (empty). They feed QLMatching
  `trigger_offsets=[bot, top]` and clus.jsonnet's per-volume x_t0cor
  (`trigger_offset`/`trigger_offset_top`). Legacy archives fall back to the
  scalar metadata `offset_us`. `PDVD_TRIGGER_OFFSET_US` overrides both;
  `PDVD_QL_DIAG=1` forces 0 **and** loosens `require_containment=false`,
  `flash_minPE=100` (offset-hunt mode); `PDVD_QL_DIAG=2` keeps the measured
  offsets with the loosened knobs (closure validation).
- `READOUT_NTICKS` read from the SP frame (10000 ticks × 0.5 µs = 5 ms in
  039252/3; the charge window is per-run DAQ config — 039349 is 3.2 ms;
  fallback 10000) for the window-truncation flag. Note the raw BDE 512 ns
  sampling is resampled to 500 ns as the first SP step, so processed frames
  are uniformly 500 ns/tick with tick 0 = that crate's window start.

## 3. Outputs

- Matched `mabc-*.zip` (T0-corrected clusters; byte-identical with/without
  `-calib`), optional Bee `op.json` (`-op`, single all-PD flash display).
- `-calib` → `work/<RUN6>_<idx>/calib-evt<ID>.json`: single per-event dump
  (no per-side files), `geometry` keyed '0'/'4' (bottom/top volume), one
  shared `flashes` array, bundles tagged `apa` 0/4 with the new `at_cathode`
  flag. Hand-scan with `pdvd/ql_scan` (port 5016; see its README).

## 4. Round-1 knob summary (what's ON/OFF and why)

ON: shared_flash, opdet_all_volumes, vd_surface_flags (cushion 10 cm),
auto_mask(+same_type, 5/20/3/2/3), sparse_lasso, lasso_flag_weight (0.2),
bundle_mask_ks, chi2_relax (pmt_excess 100 — placeholder), highconsist_ladder
(PDHD KS ceilings, loose c2n), require_containment (production only),
flash_minPE 25, light_model 'library'.

OFF (deliberate): reject_overpred (until QtoL renormalized — QtoL is 1.0
placeholder), empty_rescue/cluster_rescue (not shared-flash-aware), all
cross_side/xtpc machinery (single flash — the joint fit handles crossers
natively), robust_endpoint_trim, pe_err_on_pred/lowpe, pmt_nonlinearity.
Static `ch_mask` = [13,24,27,28,29,32,34,39] (data-dead + Ar-blind).

## 5. Caveats

- **Trigger offset RESOLVED 2026-07-10** (per-event per-crate values from the
  rawwf trigoff tree; mechanism and constancy verdict: `pdvd-ql-pending.md`
  §1). The calib-dump/Bee flash `time` folds the input-0 (BDE) offset; a
  top-volume anchor read from the dump display therefore carries the ~17–32 µs
  crate skew (the matching geometry itself uses the correct per-side value).
- **QtoL / absolute PE scale uncalibrated** — KS leads round 1; chi2-based
  branches are loosened accordingly.
- z-wall PMTs not in `pd_walls` (~1% of PE); rescues off; `flag_at_cathode`
  inert — all revisited after the first hand scan (`pdvd-ql-pending.md` §2-3).

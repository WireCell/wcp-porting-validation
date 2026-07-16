# PDVD charge–light (Q/L) matching

Status 2026-07-10: **built, wired, byte-identity verified; trigger offset
RESOLVED** (per-event per-crate DAQ timestamps, `10_pdvd-ql-pending.md` §1) —
remaining work is the tuning checklist (`10_pdvd-ql-pending.md` §2). Companion
docs: `06_pdvd-light-chain.md` (raw→OpFlash), `08_pdvd-photon-model.md` (visibility
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
  The `op` dump's `op_cluster_ids` use the pre-pipeline cluster enumeration,
  which is exactly the `img-global` instance of `mabc-all-apa.zip` (dumped at
  the same point; 2026-07 change — was `clustering-group0123/4567`).  In Bee,
  check a flash↔cluster pair against `img-global`, not `clustering-global`
  (post-pipeline, re-enumerated) and not a `bee-blobs` imaging instance (own
  numbering).  See `docs/01_pdvd.md` "Bee upload / Path C".
- `-calib` → `work/<RUN6>_<idx>/calib-evt<ID>.json`: single per-event dump
  (no per-side files), `geometry` keyed '0'/'4' (bottom/top volume), one
  shared `flashes` array, bundles tagged `apa` 0/4 with the new `at_cathode`
  flag. Hand-scan with `pdvd/ql_scan` (port 5016; see its README).

## 4. Round-1 knob summary (what's ON/OFF and why)

ON: shared_flash, opdet_all_volumes, vd_surface_flags (cushion 10 cm),
auto_mask(+same_type, 5/10/3/3/3 — pe_bright/min_contrast retuned 2026-07-10,
see §4a), sparse_lasso, lasso_flag_weight (0.2), bundle_mask_ks, chi2_relax
(pmt_excess 100 — placeholder), highconsist_ladder (PDHD KS ceilings, loose
c2n — NOTE: true PDVD matches do not pass the KS ceilings yet, see §4a),
require_containment (production only), flash_minPE 25, light_model 'library',
**QtoL 0.094** (current geometry; 0.11 was the pre-Y-truncation value —
see the §4a 2026-07-13 note) + pred-based pe_err (floor/frac/lowpe_frac/knee =
2.0/0.60/2.0/10.0) — both calibrated on the beam-flash gold pairs (§4a).

OFF (deliberate): reject_overpred (gold scatter still ~x3 and per-channel PE
scale uncalibrated; enable with hand-scan-tuned ceilings),
empty_rescue/cluster_rescue (not shared-flash-aware), all cross_side/xtpc
machinery (single flash — the joint fit handles crossers natively),
robust_endpoint_trim, pmt_nonlinearity, measured_pe_scale (beam-gold fit
exists but single-topology — refit on hand-scan GT).
Static `ch_mask` = [13,24,27,28,29,32,34,39] (data-dead + Ar-blind).

## 4a. Knob finalization from the beam-flash gold pairs (2026-07-10)

Ground truth without a hand scan: in the CTB beam runs the beam flash's
folded time is KNOWN per event (`tc_us − charge_bde_us`; the anchor
`check_trigger_flash.py` validated to −0.9 µs) and its charge partner is the
largest-predicted-light bundle on that flash → 80 true (cluster, flash) pairs
from the 120 QtoL=1 DIAG=2 dumps.  `ql_light_calib/fit_qtol_gold.py`.

- **QtoL = 0.11** ([16,84] = [0.03,0.25]): the library + official-eff model
  OVER-predicts ~10x, flat across PD groups (cathXA 0.13 / botPMT 0.16 /
  zPMT 0.12 / top memXA 0.10 ⇒ efficiencies not double-counted; one global
  normalization).  **TRAP (cost us a smoke-run round): fitting QtoL from
  auto-selected "good-KS" bundles gives 40 — wrong by ~350x.**  With a
  mis-scaled QtoL the LASSO is amplitude-inert (bundle strengths sit at their
  initial ~1.0; the per-flash background column, whose entries equal the
  measured vector, absorbs the fit), so auto-selection is accidental-dominated
  (flashes ~40x brighter than pred).  Same lesson as the crosser-scan offset
  fit: only anchor calibrations on external ground truth.
- **Regime change at the correct scale**: with QtoL=0.11 the LASSO amplitudes
  become physical and selection is amplitude-driven — smoke event 039252_0
  went from 15 junk matches (KS-accidentals) to 90 matches / 61 flashes,
  19/27 big (>1000 pt) clusters.
- **Gold-pair KS median is 0.45 (0% under the 0.06–0.10 ladder ceilings)**:
  the KS ladder passes ~no true matches on PDVD round 1.  Cause is at least
  partly the raw per-channel PE calibration — the gold per-channel meas/pred
  spans **x13 within the cathode-XA group alone** (ch10 0.68 vs ch7 0.05; the
  known "no 1-PE peak" cathode SPE question, §2 of 11_pdvd-questions-dune.md).
  A per-channel `measured_pe_scale` fitted on half the gold sample improves
  the other half's KS only 0.43→0.33 median, so shape mismatch is NOT just
  channel gains (bright-beam-shower topology; possible saturation) — the
  per-channel correction is documented but NOT baked in.
- **pe_err retune on gold**: measured-based errors give chi2/ndf ~10³–10⁴ on
  true matches (catastrophic "predicted light, measured ~0" channels);
  pred-based 2.0/0.60/2.0/10.0 gives median 9.7 with 71% under the c2n=35
  ceiling.  Wider than PDHD's 0.3/0.40/1.55/5.5 deliberately (uncalibrated
  per-channel scale).
- **auto_mask retune + validation**: with ch24 dropped from the static mask,
  QLAUTOMASK catches it only after pe_bright 20→10 + min_contrast 2→3
  (bottom-PMT neighbour medians almost never reach 20; emulation over 120
  dumps: ch24 caught 91/120, healthy channels <6%).  Live-but-dim ch16/ch33
  (event-max median ~5.6 PE, ~50x below peers) get per-event masked when
  quiet — safe direction, flagged to DUNE (questions doc §4.2).

**Update 2026-07-13 — QtoL 0.11 → 0.094 (Argon, current geometry).** The
0.11 above was fit pre-Y-truncation. The 565ccd62 active-volume Y-truncation
fix raised predicted light (it did the same to the Xe value, 0.082→0.070).
The `jjo` beam-flash table needed to re-run `fit_qtol_gold.py` has since been
removed (raw files also changed schema — `triglight`, no `charge_bde_us`), so
this is an **estimate, not a fresh 80-pair refit**: 0.094 = 0.11·(0.070/0.082)
= 0.0939, corroborated by the current-geometry library pred-ratio S175/S128 on
beam-like bundles (brightest-flash-per-event median 1.29 → 0.070·1.29 = 0.090).
A proper 128 nm gold-pair refit is a follow-up once the trigger table is
restored. This is the value in the toolkit default after the 2026-07-13 revert
to Argon (see `11_pdvd-questions-dune.md` §3); run 039252 reprocessed at v=0.1523
under it (tag `_ar1523`).

**Update 2026-07-14 — crosser-anchor refit on saturation-fixed dumps
(`docs/qlmatch/12_pdvd-qtol-recalibration.md`).** The DAPHNE saturation veto had
been hiding a ×5 cathode bias in every previous calibration (railed = nearest
channels contributed exactly 0 to Σmeas). On the 120-event `_satrep`
reprocess with 192 geometric crosser anchors: under the current Ar/128 nm
default a single QtoL does not exist (per-type spread ×30, distance-dependent
per-channel slope); under the 175 nm library the membrane-XA reference group
is consistent with **QtoL 0.094 as-is** (flat, ratio ≈ 1), with the residual
work being per-type efficiency corrections (cathode ×~7, PMT ×~0.2 vs
membrane). No defaults changed — model choice + efficiency retune are the
owner's call; see the doc's §3.

## 5. Caveats

- **Trigger offset RESOLVED 2026-07-10** (per-event per-crate values from the
  rawwf trigoff tree; mechanism and constancy verdict: `10_pdvd-ql-pending.md`
  §1). The calib-dump/Bee flash `time` folds the input-0 (BDE) offset; a
  top-volume anchor read from the dump display therefore carries the ~17–32 µs
  crate skew (the matching geometry itself uses the correct per-side value).
- **QtoL calibrated (0.094, §4a; est. — table removed) but the per-channel PE scale is not** — the
  LASSO amplitude leads round-1 selection (NOT KS: true matches fail the KS
  ladder ceilings, §4a); chi2-based branches are loosened accordingly.
- z-wall PMTs not in `pd_walls` (~1% of PE); rescues off; `flag_at_cathode`
  inert — all revisited after the first hand scan (`10_pdvd-ql-pending.md` §2-3).

## 6. Cathode-crosser standard candles + xTPC enable (2026-07-11)

Cathode-crossing pairs (one cluster per drift volume meeting at x~0 with
aligned axes) pin the T0 to a single flash and are the primary light-model
probes. 17 pairs were identified in evts 298567/298581/298595 (4 owner
hand picks + 13 found) — finder, five-criterion recipe and per-pair numbers
in `ql_display/docs/ql-cathode-crosser-recipe.md`; viewer tag `crossers`.

On the toolkit side the SBND/PDHD xTPC machinery is now ENABLED for PDVD
(`cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`): it works under
`shared_flash` because pairing is by flash-TIME coincidence (trivially
satisfied) and sides split by anode_x vs cathode. PDVD-specific values:
`xtpc_dmax: 25 cm` (NOT the PDHD/SBND 5 cm — each PDVD volume's active edge
sits ~3 cm from x=0 and the top/bottom crate skew adds a few cm: the true
pairs meet at 10-22 cm) and `cathode_ext2: -12 cm` (widened at-cathode
window so clean untruncated cathode-touching halves acquire
`at_x_boundary` and enter the xtpc candidate pool; side effect: they also
get the lasso_flag_weight down-weight, which keeps them alive through the
LASSO — desirable for crossers). `xtpc_joint_pin: true` binds each
direction-confirmed pair to ONE flash, exempt from the strength prune.

### 6a. xTPC validation on run 039252 (18 events, 2026-07-11)

Repro: archive `work/039252_*/{calib-evt*.json,mabc-all-apa.zip}` as
`prextpc-*`, then `PDVD_MAX_JOBS=6 ./run_clus_evt.sh -calib 039252 all`
(18/18 ok) with toolkit 0017de8e.

- The 17 validated candle pairs: auto-matched together 4/17 -> **17/17**;
  on the exact hand-scan flash 1/17 -> **9/17**; the other 8 are pinned one
  neighboring flash away (3.5-43 us, 6/8 later). Cause: the joint pin picks
  the flash by min ks-sum among the scenario-1-confirmed flashes and uses
  geometry only as tie-break — with PDVD's single flash stream the
  confirmed flashes differ in TIME, so a marginal ks preference can move
  the pinned T0 (worst case evt298581 c183+c4000360: pinned at the
  d=19.8 cm flash over the d=3.8 cm one). Follow-up candidate: a
  min-d/ks-tie-break pin flash choice knob (toolkit qlmatching-code.md
  sec 2a). The `crossers` viewer tag keeps the geometric picks as ground
  truth.
- Per event: xtpc_consistent on 18-140 bundles, scenario-1 18-84, pins
  6-28 (pin count = 2x pinned pairs). Total auto matches 2457 -> 2391;
  ~100-130 auto entries per event differ from the prextpc dumps (the sc1
  priority in cull_inconsistent + the cathode_ext2 boundary down-weight
  reshuffle low-confidence matches; PDVD auto agreement was 58.7% at the
  geomfix scan, so churn in that population is expected — re-scan to
  re-grade).

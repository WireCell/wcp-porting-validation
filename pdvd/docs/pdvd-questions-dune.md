# PDVD — open questions for Jay / DUNE (PDS + DAQ)

Collected 2026-07 from the Q/L-matching integration. Each item says what we
need, why, and what we will do with the answer. Companion docs:
`pdvd-ql-pending.md` (blocked-work checklist), `pdvd-qlmatching.md`,
`pdvd-spe-template.md`, `pdvd-photon-model.md`, `pdvd-light-chain.md`.

## 1. Trigger T0: per-event light↔charge time-base offset — **ANSWERED 2026-07-10**

**Resolution (jjo):** reprocessed rawwf files
(`/nfs/data/1/jjo/tmp/pdvd_flash_validation/out/rawwf_trigoff/`) carry a
`trigoff/trigger_offset` tree (per-event tc_us, charge_tde_us, charge_bde_us,
fs_t0_us, light_t0_us, offsets) extracted straight from the raw HDF5 DAQ
records; per-event table archived at `pdvd/data/jjo_triglight_offsets.txt`.
Now consumed by `run_light_evt.sh` → opflash metadata → Q/L matching
(`pdvd-ql-pending.md` §1). Sub-answers, for the record:

**How it was dealt with on our side (full record `pdvd-ql-pending.md` §1):**

1. `run_light_evt.sh` reads the trigoff tree from the same rawwf file it
   already opens, replicates the light chain's tick origin exactly
   (`PDVDOpWaveformSource` rule: min over all records of
   `round(ts·62.5) − 64` ticks for self-trigger snippets), and stamps
   **per-crate** offsets into the opflash archive metadata:
   `offset_bot_us = chain_t0 − charge_bde_us`,
   `offset_top_us = chain_t0 − charge_tde_us` (≈ −2.1..−2.5 ms; ADD to a
   flash time to land on that crate's charge axis). All 120 events
   regenerated; 120/120 consistent with jjo's independent table.
2. `run_clus_evt.sh` passes both to QLMatching `trigger_offsets=[bot, top]`
   and `clus.jsonnet` per-volume `trigger_offset(_top)` for x_t0cor
   (toolkit commit `8a765d5e`; byte-identical for PDHD/SBND and for
   `[0,0]` vs legacy scalar).
3. **Closure (decisive):** beam-trigger flash test
   (`ql_light_calib/check_trigger_flash.py`) — on CTB beam runs a ~2000 PE
   flash must sit at `tc_us − charge_bde_us` on the folded axis; 99/120
   events show it within ±5 µs, **residual median −0.9 µs** (the flash
   time bin is 1 µs). The light↔charge time base is calibrated at the
   ~1 µs level, i.e. ~0.16 cm of drift.

- **Window placement:** the light full-stream start is trigger-locked
  (±0.3 µs, per-run constant: −5.0 ms in 039252/3, −2.5 ms in 039349); the
  CHARGE windows float ±15 µs event-to-event and PER CRATE (TDE vs BDE up to
  32 µs apart, each on its own 64-sample frame boundary). The "4 ms outlier"
  event was an artifact of the statistical method.
- **Record lengths:** per-run DAQ configuration. Light 7.501 ms (039252/3)
  vs 5.301 ms (039349); charge 5.000 ms (039252/3) vs **3.200 ms (039349)**
  — the charge window is NOT always 5 ms. Raw sampling: TDE 500 ns, BDE
  512 ns (our SP resamples BDE 512→500 ns as its first step, so processed
  frames are uniformly 500 ns; window STARTS are what the offsets refer to).
- **Beam trigger:** yes — the TriggerCandidates are CTB beam triggers
  (types 15/20/22 = kCTBBeamChkvHL/HLx/HxLx, occasional 29 =
  kCTBOffSpillSnapshot); `tc_us` is in the table (16 ns DTS clock).
- **Gotchas (upstream-worthy):** `raw::RDTimeStamp` units are inconsistent in
  the PDVD decoder (TDE ns-since-epoch, BDE 16-ns DTS ticks); DAPHNE
  timestamps are 40-bit-masked (rollover ~4.9 h); the stock dunesw
  `OpFlashFinderVerticalDrift` produces corrupted single-PD flashes on PDVD
  (jjo has a validated patched module) — irrelevant to our WCT chain but any
  future comparison against art `recob::OpFlash` must use the patched finder.

The original ask is kept below for the record.

### (original ask) Trigger T0: per-event light↔charge time-base offset (was: THE blocker)

**Ask:** per-event DAQ timestamps that relate the charge readout window to
the light readout window. Concretely, either of:

- the **charge readout-window start timestamp** per event on the DTS clock
  (the charge frame extraction currently writes `tickinfo time = 0`), or
- a PDHD-style **trigger record** added to the PDVD `rawwf` light extraction
  (PDHD's files carried a `trigoff/trigger_offset` tree → `offset_us`;
  the PDVD rawwf files carry only `raw_waveform`).

**Why:** every drift-position correction is
`x_true = x_raw + sign·(t_flash + T)·v`; `T` is unknown. Our light t=0 is
the earliest light record start (recoverable from the rawwf `timestamp`
branch — already dumped by `ql_light_calib/dump_light_t0.py`, absolute µs,
16 ns DTS clock, internally coherent per event). The statistical A/C-crosser
calibration cannot decide it: at ~150 flashes per 5–7.5 ms window even a
true constant saturates at 2–3σ (best candidate, run 039252 gold
through-goers: T ≈ −2855 µs at 3.2σ — suggestive only).

**Sub-questions:**
- Is the light readout window opened at a **fixed offset from the trigger**
  (i.e. is T a per-run constant), or does its placement vary event to event?
  (We see one event, 039349 evt 19549, ~4 ms from every run-level candidate.)
- What sets the light record length — 039252/039253 records are 7.5 ms
  (468800/468864 samples) but 039349 is ~5.3 ms (331264/331328)? Same
  policy for the charge 5 ms window?
- These are beam runs (8 GeV / 0.5 GeV/c): is there a **beam-trigger time**
  record we could use as a common anchor?

**With the answer:** `offset_us(event) = charge_window_start − light_t0`,
plumbed into the opflash metadata (`offset_us`) / `data/ql_trigger_offset.txt`
— then containment cuts, boundary flags, and production Q/L matching unblock
(`pdvd-ql-pending.md` §2).

## 2. Cathode X-ARAPUCA SPE normalization (absolute PE scale)

**Ask:** how should the ADC-per-PE gain of the 16 cathode full-stream
channels be set? Specifically:

- Is there an **official DAPHNE gain calibration** (or SPE fit results) for
  the cathode modules for these runs?
- Are there **dedicated calibration runs** (LED/laser, low-light) where the
  cathode channels show a resolved 1-PE peak?
- Can you confirm the cathode signal path (power/readout over fiber on the
  HV cathode) has a **different front-end gain** than the membrane modules
  — i.e. that transferring the membrane XA 1-PE scale is NOT safe?

**Why:** we measured the cathode pulse **shape** precisely (full-stream
statistics, 2.8% held-out residual) but the **scale** needs a resolved
single-PE feature, and on the cathode it is buried: continuous pile-up from
both drift volumes fills the valley, and the 5σ ≈ 10 ADC seed threshold sits
at/above the likely 1-PE amplitude, so the amplitude spectrum is a
threshold-limited continuum (apparent mode 16–28 ADC is a turn-on artifact).
More data cannot create the feature. The cathode shape (FWHM ≈ 0.2 µs) is
also much faster than the membrane's (0.35–0.9 µs) — same sensor, different
electronics — so the membrane scale is only an interim default (transferred
by 1-PE **area**, not amplitude). Details: `pdvd-spe-template.md` §5.

**With the answer:** fix the 16 cathode templates' absolute scale → the
flash PE scale (cathode carries ~89% of collected light) → QtoL and the
chi2/PE-error model in Q/L matching stop being placeholders.

**Interim data-driven attempt (2026-07-11, `ql_light_calib/fit_channel_scale.py`)**
— we tried to calibrate the scale from our own data while the official gains
are pending, using 657 vetted charge→light pairs (80 selection-free
beam-flash gold pairs across all three runs + 577 hand-scan-vetted matched
flashes from the 10 scanned 039252 events; per-channel prediction recomputed
in Python, validated exact vs the C++ matcher):

- **Relative in-group cathode gains ARE measurable**: median meas/pred per
  cathode channel, normalized to the group, spans only ×3.2 (128 nm library)
  / ×2.5 (175 nm) — the "×13 spread" seen earlier on gold pairs alone was
  mostly topology-driven model error, not gain. ch 7 is the dimmest (~0.5×
  group), ch 5/9 the hottest (~1.4–1.5×).
- **A static per-channel `measured_pe_scale` is NOT deployable yet**:
  fitting on even events and applying to odd leaves the held-out KS
  unchanged at best (0.363→0.363 at 128 nm, 0.351→0.351 at 175 nm with a
  conservative n≥100 fit) and degrades it when all channels are fitted —
  the per-channel scatter is dominated by position/topology-dependent
  visibility-model error, not by fixed gains. The knob stays empty.
- **The absolute transfer is still blocked on this question**: the membrane
  anchor is internally inconsistent (top vs bottom membrane XA group
  medians differ ×1.6, and per-channel gold-only vs all-sample medians
  disagree strongly), and the cathode/membrane group ratio flips from 0.62
  (128 nm library) to 2.8 (175 nm) — i.e. it measures the library as much
  as the gain. An external ADC-per-PE number for the cathode DAPHNE path is
  still what's needed.

## 3. Photon library: Argon (128 nm) or Xenon (175 nm)?

**Ask:** were runs **039252 / 039253 / 039349** (2025-08/09) pure argon or
Xe-doped? And which optical model does DUNE bless for them —
`protodune_vd_v5_128nm_tf2.6` (Ar) or `..._175nm` (Xe 10 ppm)?

**Why:** we currently match with the v5 **128 nm (Ar)** ANN sampled on a
10 cm grid. The choice changes more than the visibility map:

- **Channel liveness**: ch 13 (membrane XA without PTP), 29/39 (PEN+Q PMTs)
  and 32 (uncoated) are **Ar-blind at 128 nm** and sit in our static
  `ch_mask`; at 175 nm several of them become live and must be unmasked.
- **Per-OpDet efficiencies**: we use the official
  `PDVD_PDS_Mapping_v04152025` values (XA 0.03, TPB PMT 0.12, PEN 0.036) —
  are these 128 nm numbers, and what are the blessed 175 nm equivalents?
- The Xe grid is already sampled alongside (`pdvd-photlib-vis-v5-175nm`),
  so switching is a one-line cfg change once we know.

**Related check:** if partially doped or transitioning, is a mixed model
needed (prompt Ar + shifted Xe component)?

**Data-side answer (2026-07-11): the data strongly favor Xe-doped / 175 nm.**
Still please confirm officially (doping level, run-by-run history, and the
blessed 175 nm per-OpDet efficiencies). Evidence, all three runs identical
(`ql_light_calib/check_arblind_channels.py` + `ablib_gold.py`):

1. **The Ar-blind channels are live in data.** In the raw opflash archives
   (no QL mask) ch 13/29/32/39 respond to big (≥1 kPE) flashes at the SAME
   level as their live same-type peers (e.g. ch 32, the uncoated PMT: mean
   1.0 PE/big-flash vs 0.6–1.3 for its PEN neighbours; ch 13 fires >5 PE in
   20% of big flashes — the same fraction as the live membrane XAs). The
   truly-absent controls (24/27/28/34) are identically zero. At 128 nm with
   eff_Ar = 0 this light shouldn't exist.
2. **Quantitative closure under the 175 nm grid:** recomputing the gold-pair
   predictions with the 175 nm library and type-nominal efficiencies, the
   Ar-blind channels' meas/pred medians land at 0.55–1.85× their live
   peers' — i.e. inside the general per-channel scatter. Under pure Ar
   (reflected-light-only) they should sit orders of magnitude low.
3. **Shape A/B on the 80 gold pairs**: KS median 0.429 (175 nm) vs 0.447
   (128 nm), 175 nm better in 56/80 paired; on the 657 vetted hand-scan
   flashes 0.351 vs 0.363 — even though that sample was *selected by* the
   128 nm matcher. Within-group per-channel ratio spreads also shrink under
   175 nm (cathode ×13.3→×11.5 gold-only; memXA-top σ(log10) 0.48→0.36).
4. **No mixed model needed at current precision**: the shape-only KS of
   α·128 + (1−α)·175 is monotonic in α — pure 175 nm is the minimum.

Caveats: the A/B used the 128 nm efficiencies for both (no blessed 175 nm
set yet), and the per-channel PE scale is uncalibrated (§2) — this limits
the KS discrimination; the Ar-blind liveness (items 1–2) is the robust part.
**Planned action:** switch `photon_library_file` to
`pdvd-photlib-vis-v5-175nm` and unmask 13/29/39 (32 pending its true 175 nm
efficiency) once the official efficiencies arrive; QtoL and the §2 scale
work then redo under the new library.

## 4. Other items (smaller, please confirm)

1. **DAPHNE↔module pairing of the Arapucas** — jjo's DAPHNE→module
   assignment vs the v5 geometry mirror-pairs is still unconfirmed from
   data (open item, `pdvd-photon-model.md` §2: v4 geometry had the Arapucas
   y-MIRRORED vs as-built; data agrees with **v5** exactly). Is there an
   authoritative channel↔module map for these runs?
2. **Dead channels** — we observe ch 24, 27, 28, 34 absent from the DAPHNE
   readout and OpDet14 (ch3010 PMT) with no usable gain. Expected? Is there
   a per-run official bad-channel list we should consume instead of a
   static mask?  *Data point (2026-07-10, 120 events of 039252/039253/039349):*
   **ch16 and ch33 are alive but ~50x dimmer than their peers** (per-event
   max PE median ≈ 5.6 vs 25–265 for neighbouring PMTs of the same type;
   below 5 PE in ~45% of events).  Known low-gain/low-HV channels, or a
   miscalibrated SPE scale on our side?  *Additional data point (2026-07-11,
   657 vetted charge→light pairs):* **ch2 (membrane XA, top, y=+417.6)
   measures ~10x below prediction** (meas/pred median 0.12 vs 0.7–3.3 for
   the other top membrane XAs, same sign under both libraries) — another
   low-gain candidate.  In the meantime our per-event dead/dim self-check
   (`auto_mask` 10/3, same-type neighbours) is ON in production and
   per-event masks ch16/33-class channels when they are quiet.
3. **Run 039324 light data** — the raw light file was never staged
   (`/nfs/data/1/jjo/data/PDVD/` has only 039252/039253/039349); can it be
   provided so 039324 gets Q/L matching too?
4. **Charge-side timestamps in future extractions** — independent of Q1's
   one-off answer, could the standard charge extraction start filling the
   frame `tickinfo` time (DTS) so the offset is self-contained per event?

# PDVD — open questions for Jay / DUNE (PDS + DAQ)

Collected 2026-07 from the Q/L-matching integration. Each item says what we
need, why, and what we will do with the answer. Companion docs:
`pdvd-ql-pending.md` (blocked-work checklist), `pdvd-qlmatching.md`,
`pdvd-spe-template.md`, `pdvd-photon-model.md`, `pdvd-light-chain.md`.

## 1. Trigger T0: per-event light↔charge time-base offset (THE blocker)

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

## 4. Other items (smaller, please confirm)

1. **DAPHNE↔module pairing of the Arapucas** — jjo's DAPHNE→module
   assignment vs the v5 geometry mirror-pairs is still unconfirmed from
   data (open item, `pdvd-photon-model.md` §2: v4 geometry had the Arapucas
   y-MIRRORED vs as-built; data agrees with **v5** exactly). Is there an
   authoritative channel↔module map for these runs?
2. **Dead channels** — we observe ch 24, 27, 28, 34 absent from the DAPHNE
   readout and OpDet14 (ch3010 PMT) with no usable gain. Expected? Is there
   a per-run official bad-channel list we should consume instead of a
   static mask?
3. **Run 039324 light data** — the raw light file was never staged
   (`/nfs/data/1/jjo/data/PDVD/` has only 039252/039253/039349); can it be
   provided so 039324 gets Q/L matching too?
4. **Charge-side timestamps in future extractions** — independent of Q1's
   one-off answer, could the standard charge extraction start filling the
   frame `tickinfo` time (DTS) so the offset is self-contained per event?

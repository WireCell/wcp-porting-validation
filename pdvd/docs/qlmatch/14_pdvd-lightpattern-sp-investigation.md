# Why some flashes show measured light ≫ predicted (evt 298567 flash gid 57)

**Repro:**
```
cd pdvd/docs/qlmatch && python3 scripts/lightpattern_sp_investigation.py   # tables + figs 1-5
cd pdvd && python3 docs/qlmatch/scripts/saturation_signature.py            # §"Saturation" + fig 6
# inputs: work/039252_0_arcal/calib-evt298567.json (+ the 17 sibling arcal dumps),
# input_data_light_rawwf/np02vd_raw_run039252_*_rawwf.root,
# toolkit cfg pdvd-spe-templates.json / pdvd-opch-map.json,
# photlib/pdvd-photlib-vis-v5-128nm.json
```

## Symptom

On the `arcal` candle round (port 5019), flash gid 57 (t = −270.6 µs, grp 58,
matched to the cathode-crosser pair top:56 / bot:134) shows several cathode
X-Arapucas with measured PE far above the displayed prediction (od4: 12764
measured vs 0 predicted) while other X-Arapucas agree well.  Similar
mismatches appear on other cathode crossers.  Since Q-L matching *is* the
comparison of these patterns, this had to be understood.

## What the waveforms say: the light is real and the deconvolution is faithful

Raw-waveform examination (fig 1) shows one genuine, simultaneous, very bright
flash on **all** cathode channels: a single prompt peak at the flash time with
a ~8 µs tail, 30k PE total, four opdets railed at the DAPHNE 14-bit ceiling.

![fig 1 — flash-57 cathode raw waveforms with the decon overlaid, railed (od4) and unrailed (od10)](pics/lightpattern_fig1_raw_decon_cathode.png)

Decon PE tracks the raw area on every cathode channel:

| od | sat | dump PE | raw-area PE | dump/raw |
|---|---|---|---|---|
| 4 | 1 | 12764 | 9061 | 1.41 (repair recovers clipped area) |
| 5 | 0 | 940 | 1160 | 0.81 |
| 6 | 1 | 3773 | 3955 | 0.95 |
| 7 | 1 | 2462 | 2189 | 1.12 |
| 8 | 1 | 4738 | 5571 | 0.85 |
| 9 | 0 | 1139 | 1183 | 0.96 |
| 10 | 0 | 2683 | 2880 | 0.93 |
| 11 | 0 | 1052 | 1124 | 0.94 |

And the **unmasked** prediction of the selected pair (vis-loop replica ×
adopted per-type eff scale factors) matches measured on every cathode
channel, *including the railed ones* (fig 5):

| od | meas | pred shown in viewer | pred unmasked | meas/unmasked |
|---|---|---|---|---|
| 4* | 12764 | **0** | 15570 | 0.82 |
| 6* | 3773 | **0** | 4825 | 0.78 |
| 7* | 2462 | **0** | 4887 | 0.50 (ch1051 template, below) |
| 8* | 4738 | **0** | 6728 | 0.70 |
| 5/9/10/11/12 | — | = unmasked | — | 0.77–1.25 |

![fig 5 — flash 57 measured vs masked vs unmasked prediction; the masked prediction is what the viewer showed](pics/lightpattern_fig5_pattern.png)

So there is **no per-flash signal-processing failure here** — the mismatch
decomposes into three real, independent defects:

## Saturation: what it looks like, and how the deconvolution handles it

Because the four railed OpDets above are exactly the ones the display got
wrong, this section spells out the saturation signature and the current
processing. Numbers/figure: `scripts/saturation_signature.py` (fig 6),
run-039252 evt298567. Depth statistics and the estimator comparison are
**not** repeated here — they are `11_pdvd-saturation-recovery.md` §2.

**Scope warning.** §§(a)-(c) below are measured on the **16 cathode
channels**. They do **not** generalize to the membrane XA / PMT snippets,
where the membrane snippets show a different and currently unhandled
failure — see §"The zero-run".

![fig 6 — (a) the positive flat-top clip at 16383; (b) the AC-coupling undershoot after the pulse — the real "signal goes negative"; (c) the decon plateau, positive throughout the rail](pics/lightpattern_fig6_saturation_signature.png)

The answers in one line each, **on the cathode**: **(a)** saturation is a
positive flat-top at the upper 14-bit rail; **(b)** the slow negative
excursion after a bright pulse is real, but it is AC-coupling undershoot, not
the clip; **(c)** the deconvolution never goes negative inside the rail — it
produces a positive plateau, and that plateau is the actual pathology.

**On the membrane snippets there is a fourth, different answer** — the raw
ADC jumps to a flat 0 (= −pedestal once subtracted) at the pulse peak, unflagged
by `detect_saturation`. That is the sharp "goes negative and stays flat"
signature, and it is an open issue: §"The zero-run".

### The signature is a positive flat-top — there is no negative clip

PDVD DAPHNE runs **positive raw polarity** (`input_polarity: 1`,
`flash.jsonnet:93`), so a saturating pulse pins at the **upper** 14-bit
ceiling **16383**. The signature is a run of consecutive samples at exactly
16383 with a flat top (fig 6a: ch1010, 160 ticks = 2.6 µs at the rail).
This is precisely what `OpDecon` tests for — `q[i] >= m_saturation_adc`
(`OpDecon.cxx:494`, `saturation_adc` default 16383).

**On the cathode**, the lower rail is never reached: across all 16 cathode
channels of evt298567 the minimum ADC is 1592–5326 against pedestals of
~1800–5300, and the count of samples at ADC ≤ 0 is **0**. On the cathode,
saturation never drives the signal negative.

**The membrane XA snippets are a different story — see
§"The zero-run" below.** There the ADC does reach the bottom of its range, and
the pedestal-subtracted waveform shows exactly the "goes negative and stays
flat" signature. That case is *not* covered by the paragraph above, and it is
*not* caught by `detect_saturation`.

Note that the rail test is applied to the **raw** trace (`trace->charge()`,
`OpDecon.cxx:489`) *before* `input_polarity` is used — polarity only enters
later, inside `deconvolve()` (`OpDecon.cxx:371`). So detection is
polarity-independent: it always asks "did the ADC hit the ceiling", which is
the physically right question for a DAPHNE 14-bit rail regardless of how the
pulse is oriented afterwards.

### Where the signal DOES go negative: the AC-coupling undershoot

The colleague's recollection is real, but it is **not** the rail — it is the
slow undershoot that follows any bright pulse on an AC-coupled front end
(fig 6b). Measured against a **local** pre-pulse baseline (median of the quiet
band 250–50 ticks before the rail; the stream-head pedestal must not be used
here, it confounds true undershoot with baseline-reference bias), the deepest
rail of each channel is followed by:

| window after rail exit | median (ADC vs local baseline) | range | channels below 0 |
|---|---|---|---|
| +2 000 ticks (32 µs) | +3.0 | [−26, +134] | 4/9 |
| +8 000 ticks (128 µs) | **−51.0** | [−105, −33] | **9/9** |
| +30 000 ticks (480 µs) | **−96.0** | [−175, −78] | **9/9** |

(9 = the channels whose pre-band is genuinely quiet, std < 15 ADC; on the other
7 another pulse sits in the band so the local baseline is meaningless — the
script prints and excludes them.)

So: the baseline sags ~50–100 ADC below its pre-pulse level and stays there for
**hundreds of µs** — a fraction of a percent of the ~13 600-ADC pulse height,
but ~100× longer than the pulse.

**It is not saturation-specific.** Repeating the measurement on 33 bright but
**unrailed** pulses (amplitude 4295–12476 ADC, isolated, rail neighbourhoods
excluded — §D of the script): **33/33 undershoot**, median −25 ADC at +8k
ticks. Crucially the undershoot/amplitude ratio matches the railed pulses:

| pulse class | n | median undershoot/amplitude @ +8k |
|---|---|---|
| bright, unrailed | 33 | **0.31%** |
| railed (clipped, amp ≥ 13600) | 9 | **0.37%** |

Same ratio ⇒ this is **linear AC coupling** — the front end differentiates
every pulse and pays back its area slowly — not a saturation-recovery
artifact of the amplifier. Clipping neither causes nor worsens it. Two
consequences matter downstream:
- Later flashes in the same stream sit on a depressed baseline.
- Averaging pulses without undershoot repair biases an SPE template's **area**
  low — and if the undershoot dominates, negative. That is exactly root cause
  3's ch2020 (area −1595): a *harvest* artifact, not a per-event decon
  behavior.

### The zero-run: membrane snippets that jump to a flat "negative" — OPEN

**Status: found 2026-07-16. Mechanism still unconfirmed; a default-OFF fix
exists (`overflow_to_rail`, below) but is NOT enabled in production.** Found by
looking for the signature the owner recalled ("positive signal, then suddenly
negative and flat there") — the cathode-only scan above had missed it because
it only looks at 10xx channels.

On **membrane XA snippets** the raw ADC drops to **exactly 0** and stays pinned
there for tens to hundreds of consecutive samples. In the pedestal-subtracted
waveform the chain actually deconvolves, that is a flat plateau at
**−pedestal** (≈ −2500 ADC) — i.e. precisely "suddenly negative, then flat".
No sample is ever `< 0`: the raw is unsigned and floored at 0.

evt298567, pins of ≥5 consecutive samples at ADC ≤ 1 (§E of the script):

| ch | snippets | pinned samples | longest run | max ADC | flagged by `detect_saturation`? |
|---|---|---|---|---|---|
| 2010 | 1 | 32 | 32 | 16280 | **NO** |
| 2011 | 1 | 34 | 34 | 16352 | **NO** |
| 2021 | 1 | 201 | 176 | 15266 | **NO** |
| 2040 | 1 | 105 | 105 | 16299 | **NO** |
| 2041 | 1 | 77 | 77 | 16368 | **NO** |

449 pinned samples on 5 channels, **none flagged**. All five are membrane; the
PMT channels carry only isolated single samples at 0 (no run ≥5) — not, on this
evidence, the same phenomenon.

The zero-run sits exactly where the pulse **peak** should be, and is bracketed
by near-ceiling values. Full resolution, ch2011 entry 1030 (pedestal 2488):

| t | 118 | 119 | 120 | 121 | 122…155 | 156 | 157 | 158 | … |
|---|---|---|---|---|---|---|---|---|---|
| ADC | 2612 | 4096 | 8376 | 14029 | **0** (34 samples) | 16352 | 16286 | 16213 | decays |

The pulse rises steeply (≈+5000/sample), the ADC reads 0 for 34 samples, then
reappears at 16352 — just **below** the 16383 ceiling — and decays normally at
≈−70/sample. The bracketing values say the true signal was *above* the ceiling
throughout the zero-run: this reads as an **overflow reported as 0 rather than
clamped at 16383**. ch2021 entry 337 shows the same, plus a second flat run
pinned at exactly **1** (25 samples) during the post-pulse undershoot — an
analog signal does not sit at 1 ADC for 25 samples, so both are digital pins.

**Why this matters — `detect_saturation` misses all of it.** The detector tests
`q[i] >= 16383` (`OpDecon.cxx:494`). These snippets never reach 16383 — their
maxima are *below* it (ch2011 16352, ch2040 16299, ch2021 15266) — so **not one
is flagged**, even though the membrane/PMT branches do run
`detect_saturation=true` (`wct-light-reco.jsonnet:106-118`). The chain
therefore deconvolves a waveform with a −2500-ADC notch at the pulse peak as if
it were valid data. A positive-threshold hit finder is very unlikely to
recover a sensible PE from that.

**Does it explain the Phase-4 membrane/PMT puzzle? On this event, no.** It was
an attractive hypothesis — unflagged saturation on exactly the PD types whose
calibration never closed, and `12_pdvd-qtol-recalibration.md` §5 reports
membrane/PMT median meas/pred = 0.00 with 54–66% zeros **in covered flashes**,
i.e. measured ≈ 0 where the prediction is bright, which is what an unhandled
peak-overflow would produce (coverage cannot explain those: coverage says the
channel *was* reading out). But the measured scale does not support it as the
dominant cause: only **5 of 1471** membrane snippets in evt298567 overflow
(**0.34%**), and knob-on changes **3 of 415** flashes. That is orders of
magnitude short of a 54–66% zero rate. Overflow is real and worth fixing, but
on this evidence it is not the Phase-4 explanation. Prevalence across the 120
events remains unmeasured; treat the link as **not supported** rather than
open.

**Open question for the PDS/DAPHNE experts (do not guess):** why does the
self-trigger path report overflow as 0 while the cathode full stream clamps
hard at 16383? The two populations demonstrably behave differently in the same
event. The knob below is justified by the *bracketing evidence* alone (the true
signal was above the ceiling ⇒ treat it as ≥ rail and flag it), which is
conservative and mechanism-independent — but **enabling it in production is
gated on this question being answered.**

### The fix: `overflow_to_rail` (toolkit, default OFF)

The owner's proposal — detect them, move 0 to the maximal saturation value,
then let the existing chain run — is implemented as `OpDecon.overflow_to_rail`.
It composes with everything already built: once a run reads `saturation_adc`,
`detect_saturation` flags it, `saturation_repair` bridges it, `flag_saturation`
marks the hit, `OpFlashFinder` emits it in `flash_sat`, and `QLMatching`
excludes the channel from chi2/KS/LASSO. No new downstream code.

**Algorithm** (`OpDecon::unclip_overflow`, run on a copy *before* the rail scan):

```
for each maximal run [i,j) with adc[t] <= overflow_adc (default 1):
    if (j - i) < overflow_min_samples        (default 5)    -> skip
    if i == 0 or j == n                                     -> skip  (no neighbour pair)
    if i < pre_trigger - pedestal_buffer                    -> skip  (pedestal window)
    if adc[i-1] >= overflow_min_neighbor                    (default 8000)
       and adc[j] >= overflow_min_neighbor:
           adc[i..j) = saturation_adc                       (16383)
```

**Why both immediate neighbours.** The floor pin has *two* causes and only one
is overflow:

| run | bracketing (before, after) | truth | action |
|---|---|---|---|
| ch2011 [122,156) | 14029, 16352 | signal above ceiling | remap |
| ch2021 [122,298) | 12109, 15266 | signal above ceiling | remap |
| ch2021 [301,326) | **345, 3595** | deep undershoot pinned at the floor | **leave** |

Requiring *both* sides to reach 8000 separates them with a 35× margin (lowest
overflow 12109 vs 345). This matters because a false positive is the dangerous
direction: the saturation flag protects the *fit*, but the measured PE still
enters flash totals and the display, so mis-remapping an undershoot would
fabricate a rail-height pulse in a flash total. Immediate neighbours only —
widening to 3 samples lets a one-sample spike beside the undershoot run
masquerade as an overflow exit (12156 vs 12109, margin gone).

Repair accuracy is deliberately *not* the goal: a 176-sample overflow is far
past the depth ≈3 where `11_pdvd-saturation-recovery.md` §2 shows no estimator
is reliable. The flag is the point; the bridged value only improves the flash
total over an un-repaired −2500 notch.

**Limitations:** the threshold is tuned on 6 runs in one event; a run touching
a snippet edge, or reaching into the 20-sample pedestal window, is
conservatively left alone; floor-clipped *undershoot* is a separate phenomenon
this does not address (it cannot be recovered by mapping to the rail).

**Gates (all PASS, toolkit `flash`):**
- Compiled config knob-off byte-identical to pre-change (`wct-light-reco`,
  production op point); knob-on adds exactly the three `overflow_to_rail` keys.
- PDVD light knob-off member-hash identical to the pre-change binary:
  `work/039252_light298567_ovfoff` == `_spcov` (`39cf8227…`, 11 members).
- PDHD all-PD light knob-off identical: `work/029107_allpd1015_ovfgate` ==
  `_p2covgate` (`e8731138…`, 7 members).
- **False-positive guard:** knob-*on* on an event with no overflow runs
  (evt298609) is byte-identical to knob-off (`36b544b6…`) — the knob is inert
  when there is nothing to remap.
- `wcdoctest-flash` 31/31.

**Knob-on smoke (evt298567, `work/039252_light298567_ovfon2`):** membrane goes
**0 → 5** flagged tick-runs ("5 floor-pinned overflow runs remapped"); cathode
unchanged at 111 flagged / 0 remapped; PMT 0/0. 6 qualifying runs exist and
**5** are remapped — the ch2021 undershoot run is correctly rejected. Downstream
the effect reaches `flash_sat`: membrane **OpDet 2** gains a flag (cathode
OpDets 4–11 counts identical), 412 of 415 flashes are bit-unchanged, and 3
flashes near t ≈ 4.9 ms re-form. Toolkit and runner defaults are **OFF**
(`PDVD_OVERFLOW_TO_RAIL=1` to enable) pending the mechanism question above.

### Does the deconvolution go negative inside the rail? No (cathode)

A clipped flat-top deconvolves to a sustained **positive plateau** — checked on
four cathode channels (fig 6c), never a single negative sample inside the
railed run:

| ch | rail len | decon inside rail: min | max | mean | n(<0) |
|---|---|---|---|---|---|
| 1010 | 160 | +7.4 | +40.2 | +15.5 | 0/160 |
| 1050 | 178 | +6.1 | +30.5 | +10.7 | 0/178 |
| 1021 | 205 | +10.9 | +46.7 | +16.2 | 0/205 |
| 1040 | 117 | +11.3 | +46.9 | +17.7 | 0/117 |

(units: 100 = 1 PE/tick.) The plateau *is* the pathology: the WI filter
G = conj(H)·F/(|H|²+eps) is an inverse filter, so a flat top — which no SPE
shape can produce — deconvolves into a wide, flat, positive excess. That is
what makes `OpHitFinder` over-integrate it into one ~16 µs hit and fragment it
into spurious wide hits, and why the mask exists at all (`OpDecon.h:90-99`).
The decon *does* dip slightly negative **between** pulses (min ≈ −0.1 to −0.3,
i.e. −0.003 PE/tick) — that is baseline wander/undershoot leak, not the rail.

### How the current chain deals with it

Two distinct mechanisms, easily confused:

**1. The negative baseline (undershoot) is `OpRoi`'s job, not the decon's.**
`OpDecon`'s own baseline handling is crude by design: one pedestal from the
first `pre_trigger − pedestal_buffer` = **20 samples** (`OpDecon.cxx:362-365`)
— for the cathode that is 20 samples at the head of a **468 864-sample**
record, so any later sag is *not* tracked. The removal happens downstream in
`OpRoi` (cathode only, `wct-light-reco.jsonnet:98`):
- a high-pass `H(f) = 1 − exp(−(f/τ)²)`, τ = 0.05 MHz ≈ 1/20 µs — scintillation
  (<20 µs) passes, the ~500 µs undershoot does not (`OpRoi.h` step 1);
- median subtraction + MAD rms, ringing channels zeroed (steps 2–3);
- ROI hysteresis (step 4);
- **step 5**: per-ROI, subtract the line through the ROI endpoints, so every
  ROI starts and ends at exactly zero — a local detrend that removes whatever
  undershoot pedestal the ROI is sitting on.

Membrane XA / PMT have **no** `OpRoi` — they are 1024-tick (16.4 µs) snippets
deconvolved and hit-found directly. Over 16.4 µs the slow undershoot is
essentially constant, so the per-snippet 20-sample pedestal absorbs it. Their
failure mode is the template harvest (2020), not the per-event decon.

**2. The rail itself: detect → (repair) → flag, never silently.** All knobs
default OFF (byte-identical); the PDVD runners turn them on:

| stage | knob | what it does |
|---|---|---|
| `OpDecon` | `detect_saturation` | flags each run of ≥`saturation_min_samples` samples ≥16383 as a `"saturation"` ChannelMaskMap range `[tbin+lo, tbin+hi)`, padded by `saturation_pad` (`OpDecon.cxx:482-509`). Per **run**, not per trace — a full stream is not vetoed wholesale on one stray sample. |
| `OpDecon` | `saturation_repair` | before deconvolving, fills each railed run with the two-sided exponential bridge `min(rising extrapolation, falling back-extrapolation)`, clamped ≥ the measured samples, τ fit from the channel's SPE template (`repair_runs`, `OpDecon.cxx:320-354`). Deconvolves a **copy**; the mask is still emitted (repair AND flag). |
| `OpHitFinder` | `veto_saturation` | drops hits overlapping a flagged range — the old behavior, now **off**. |
| `OpHitFinder` | `flag_saturation` | keeps the hit, appends a 10th column = rail-overlap flag. |
| `OpFlashFinder` | (data-driven) | 10-col input ⇒ emits the per-flash per-OpDet `flash_sat` tensor. |
| `QLMatching` | `use_saturation_flag` | flagged channels leave that flash's opdet mask ⇒ excluded from chi2/KS/LASSO, while the **measured PE stays** in the flash and the dump. |

The ruling operating point (`_spcov`, PDVD production default since
2026-07-15) is **keep-and-mark + repair**: `detect_saturation=true`,
`veto_saturation=false`, `flag_saturation=true`, `saturation_repair=true`,
`use_saturation_flag=true`. Rationale in `11_pdvd-saturation-recovery.md` §3:
repair is a bounded second-order improvement (+2% at clip depth 1.4, +9% at 2,
+23% at 4), the median railed bright flash has a depth ≈5.6 rail where no
waveform estimator is reliable — so the *physics* fix is flag propagation, not
a better estimator. Root cause 1 below is the display half of that: the
prediction must be shown unmasked even though the fit excludes the channel.

## Root cause 1 — saturation mask zeroes the *displayed prediction*

With `use_saturation_flag` on, QLMatching removes rail-flagged channels from
the per-flash opdet mask (QLMatching.cxx:1316-17); the bundle prediction is
accumulated under that mask (:1445), and the calib dump writes this *masked*
`pred_pe` (:2878, :2892-94) while the measured `pe` is written unmasked
(:2803-05).  The ql_scan viewer never reads `flash["sat"]`, so it displays
meas 12764 vs pred 0.  **37% of auto-selected bundles** in the candle round
sit on sat-flagged flashes — this artifact is everywhere.

Per owner direction: the full predicted light must always be shown; the
saturation flag should only exclude channels from chi2/KS/LASSO.

## Root cause 2 — self-trigger coverage blindness (membrane XA + PMTs)

> **SUPERSEDED 2026-07-16 — see §12.**  The *observations* below stand (the
> snippet structure, the duty cycle, the exact covered ⇔ meas>0
> correspondence).  The *inference* — that the zeros are "fake" and must leave
> the fit — does not: the brightness scan bins by flash total PE rather than
> per-channel PE, DAPHNE has no dead time, and the self-trigger threshold is
> ~1 PE, so a zero is a real upper limit.  The zeros are kept in the fit and
> merely labelled as of §12.  Read this section as the coverage *measurement*,
> not as its interpretation.

Cathode 10xx channels are full 468800-sample (7.5 ms) streams.  Membrane
20xx / PMT 30xx channels are **self-triggered 1024-sample (16.4 µs)
snippets** — each channel is live only ~5–30% of the readout (fig 4).  When
no snippet overlaps a flash window, the chain silently scores measured = 0.

![fig 4 — membrane/PMT self-trigger snippet livetime vs the flash-57 window; covered ⇔ measured > 0](pics/lightpattern_fig4_coverage_timeline.png)

Flash-57 correspondence is exact (T3): every membrane/PMT opdet with a
covering snippet has nonzero dump PE; every one without reads 0.0 — e.g. od1
(ch2030) reads 0 against a 28-PE prediction, od3 (ch2040/41) 0 against 37,
while covered od12 (ch2050) reads 438 vs 370 predicted.

Brightness scan over the 18-event round (T4): P(meas>0) for membrane opdets
is 0.27/0.32/0.32 for flashes of 1–3k/3–10k/>10k PE — flat in *flash* PE.
(**§12: this does not imply duty cycle** — flash PE is a cathode-dominated
total; binned by per-*channel* predicted PE, coverage runs 0.58 → 0.98.)  ~70% of membrane/PMT
entries entering chi2/KS — and the 2026-07-14 per-type efficiency fits — are
fake zeros.  The adopted membrane ×1.655 and PMT ×0.352 scale factors are
therefore contaminated and must be refit once coverage is handled.

Across the 42 strict-crosser anchors (T5), membrane opdets 0/1/2/12/19 have
median meas/pred = 0.00 — entirely explained by coverage.

## Root cause 3 — broken SPE templates distort per-channel PE scales

In OpDecon's wiener-inspired branch the net PE normalization is
1/template-area (DC gain 1/H(0), OpDecon.cxx:391-93); OpHitFinder's
scale/spe_area (100/100) cancel.  A wrong template *area* is a wrong
measured-PE scale.  Outlier scan of `pdvd-spe-templates.json` (T6):

| channel | area (ADC·ticks) | peers | opdet | effect |
|---|---|---|---|---|
| 1051 | 8557 | cathode ~500–1300 | 7 | od7 PE ÷~2 (fig 2: same pulse, 1066 PE via ch1050's template vs 75 via ch1051's) |
| 2020 | **−1595** | membrane ~1200–2400 | 2 | decon DC gain < 0: pulse-window PE −40%, total area window-dependent/wrong-sign — PE meaningless (fig 3); its own 24 self-triggers are noise (sick channel) |
| 2011 | 3132 | sibling 2010 = 1221 | 0 | od0 half low ×2.6 |
| 3010 | 15935 | PMT ~900–2500 | 14 | PE ÷~10 |
| 3020 | 8479 | PMT ~900–2500 | 15 | PE ÷~5 |

These match the anchor table: od7 low with a *tight* band (0.77 [0.63,
1.03]), od14 pinned at ~0, od15 at 0.64.

![fig 2 — ch1050/ch1051 see the same physical pulse; the ×7 template-area error makes their decon PE differ by ×7](pics/lightpattern_fig2_od7_template_bug.png)

![fig 3 — ch2020's negative-area SPE template inverts the deconvolution, making its PE meaningless](pics/lightpattern_fig3_ch2020_inversion.png)

The ch2020 negative area is the harvest-side face of the undershoot measured
in §"Saturation" above: averaging pulses whose AC-coupling undershoot was not
repaired drags the template area down, and here past zero.

## Why it hid

- The saturation **veto** used to zero the measured PE on exactly these
  bright channels — meas 0 vs pred 0 looked "consistent".  The keep-and-mark
  chain (2026-07-14) surfaced the real measured values, and the masked
  prediction became visible as a huge mismatch.
- The per-type factor fits aggregate Σmeas/Σpred at flash level, where
  covered channels dominate the numerator — the fake zeros depressed the
  ratio smoothly instead of breaking it.
- Template outliers hit single sub-channels of ganged pairs; the healthy
  sibling kept the opdet nonzero, hiding the scale error inside the pair sum.

## Fix (phased; each phase gated and committed separately)

1. **Dump/viewer**: QLMatching dumps the *unmasked* prediction (`pred_pe`)
   while chi2/KS/LASSO keep the masked one; ql_scan viewer marks sat-flagged
   channels and excludes them from the ratio panel (toolkit + wcp commits).
   **DONE — toolkit af0a0284** (knob-off dumps proven identical field-by-field
   old-vs-new code, 198 flashes / 5171 bundles, setarch -R, reference tags
   `work/039252_0_p1{on,off}{old,new}`; knob-on: fit metrics identical,
   pred_pe gains the full prediction on exactly the 2303 sat-flagged
   entries — flash 57 ods 4/6/7/8 now dump 15570/4825/4887/6728) **+ wcp
   viewer commit** (orange rings/triangles for sat channels, ratio panel
   excludes them; session-tested on port 5021 scratch).  NOTE for A/B
   bookkeeping: owner commit 3c30cf58 (static ch_mask += {2,16,17,33})
   landed between the `arcal` round and these gates — dumps produced before
   it carry small nonzero pred on ch 2/17/33; not a code effect.
2. **Coverage chain**: OpHitFinder `emit_coverage` → coverage tensor →
   OpHitMerge passthrough → OpFlashFinder `flash_cov` companion tensor →
   optical PCs → `Opflash::get_cov` → QLMatching `use_coverage_flag` /
   `coverage_min` masking + dump `cov` array; viewer "no data" marker.  All
   default-OFF, byte-identical off; PDVD runners turn them on.
   **DONE — toolkit commit (flash/aux/clus/match) + wcp runner/viewer
   commit.**  Gates: compiled-config knob-off byte-identical for
   `wct-light-reco` and `wct-clustering` (keys present on);  PDVD light
   knob-off hash PASS vs the pre-change `_satrep` archive
   (`work/039252_light298567_p2off`); PDVD QL knob-off dump byte-identical
   vs the Phase-1 binary (`039252_0_p2qloff` == `039252_0_p1onnew`); PDHD
   all-PD light hash PASS (`029107_allpd1015_p2covgate` vs reference);
   wcdoctest flash/match/aux/clus.  Knob-on smoke (`_p2cov` /
   `039252_0_p2qlcov`): `flash_cov` for gid57 reproduces the raw snippet
   coverage channel-by-channel (od1/od3/od12 = 0, min-over-subchannel
   rule); the true crosser bundle (57, top:56) chi2 drops **459.6 → 91.8**
   (ndf 18→13, KS 0.120→0.088) once the fake zeros leave the fit; every
   flash carries some uncovered channel; 54/108 auto-selections move
   (intended knob-on physics change → Phase-4 refit).  Coverage semantic:
   an OpDet counts covered only when ALL its ganged DAPHNE sub-channels
   cover the window (od12's half-covered pair is masked); dead-RO OpDets
   24/27/28/34 have no mapped raw channel ⇒ cov 0 (already static-masked).
3. **SPE templates v2**: validated re-harvest (`pd_plot/spe_v2.py`);
   `pdvd-spe-templates-v2.json` behind the `flash.jsonnet` `spe_file`
   selector defaulting to v1; runner env `PDVD_SPE_V2` (default ON) selects
   v2.  **DONE — toolkit commit (v2 json + OpDecon DC≤0 warn) + wcp commit
   (builder, `spe_v2` TLA, runner env).**  Two-tier validation:
   - *shape* (area>0, area/amp within 2.5× the population median) flags
     ch2020 (−21.7), ch3010 (310), ch3020 (84);
   - *ganged-pair amplitude* (harvest 1-PE mode ratio vs the coincident
     bright-pulse gain ratio, n=44–2205 pulses/pair, only the HIGH-side
     member is the latch) flags ch1051 (mode 82 vs sibling-implied 28.9)
     and **ch2011** (mode 64 vs implied 39.5 — the ×2.6 area split of od0
     is a mode latch, not a real gain difference: the pair's bright pulses
     match at ratio 0.912).
   Repairs = population-average shape (valid channels only) × validated
   1-PE amplitude (sibling transfer for 1051/2011; own mode for
   2020/3010/3020).  Areas: 1051 8557→1042, 2011 3132→1510, 2020
   −1595→1836, 3010 15935→612, 3020 8479→1092; the other 46 channels are
   bit-for-bit == v1.  Gates: compiled-config knob-off byte-identical
   (HEAD-compiled vs new, knob-on flips exactly the three OpDecon
   `spe_file` keys); light knob-off hash PASS
   (`work/039252_light298567_p3off` == `_p2off`, member hash e3729ae8);
   wcdoctest-flash 31/31; the new OpDecon warn fires on v1 (template 18 =
   ch2020, all three branches) and is silent on v2.  Knob-on smoke
   (`_p3v2`): bright crosser flash od7 2462→6546 (×2.66; meas/unmasked-pred
   0.50→1.34, type factors to be refit in Phase 4), od0 51.9→73.2 (×1.41),
   every other opdet identical; flash census 405→415 — all movers are
   ~10–16 PE threshold-hoppers rescaled across the flash minPE cut, plus
   one fake 260-PE flash dominated by od2=225 (ch2020 negative-template
   artifact) that correctly disappears.
4. **Reprocess + refit**: 120-event reprocess at the new operating point,
   coverage-aware refit of the per-type efficiency factors (owner gate before
   adoption), fresh candle round on port 5019.  **DONE (factors deliberately
   NOT changed — owner gate open)** — light `_spcov` 120/120 + QL `spcov`
   120/120; fit_qtol_crossers.py now drops `cov<1` channels from both sums.
   Cathode closes at unity (per-group 1.006, global 1.078, od7 healed to
   1.14); membrane/PMT show a bimodal residual no scalar factor can absorb
   (5/6 membrane opdets: median meas/pred = 0.00 with 54–66% zeros in
   covered flashes vs ×4–10 overshoots at dim predictions — the 128 nm
   wall-channel shape problem, plus self-trigger selection bias at dim
   pred).  Recommendation to owner: keep the f7c66ab8 factors; details +
   per-opdet census in `12_pdvd-qtol-recalibration.md` §5.  Candle round
   `candles-spcov` (222 keep / 203 add) now serves port 5019.
   **Owner ruling 2026-07-15 (gate CLOSED):** factors kept as-is; the
   `candles-spcov` round accepted; the spcov operating point (keep-and-mark
   + repair + coverage chain + QL sat/cov masking + SPE templates v2)
   promoted to PDVD production default — the runner env defaults are
   ratified, `_spcov` supersedes `_satrep` as the canonical dump record.
   Toolkit defaults stay OFF (byte-identical).

## 12 — Root cause 2 revisited: the silence is a measurement (2026-07-16)

> Repro (read-only):
> ```
> cd wcp-porting-img/pdvd
> python3 docs/qlmatch/scripts/analyze_coverage_deadtime.py            # 18 evts
> python3 docs/qlmatch/scripts/analyze_coverage_deadtime.py 298567     # one evt
> ```

Owner question: *"when there is no data (due to self trigger threshold), the
PE is treated to be zero in the flash. I believe this is a correct treatment?
… it is OK to label things in the qlport as no data, but we should not
exclude them from the fit."*

**The owner is right, and Root cause 2 above is wrong on its central claim.**

### The one question that decides it

An uncovered self-trigger channel is in one of two worlds:

- **World A — armed and silent.** Powered, listening, light arrived below
  threshold, never fired.  The silence *is* a measurement: "the light here was
  < threshold."  PE ≈ 0 is correct, and it constrains the fit.
- **World B — not listening.** Busy reading out an earlier trigger / buffer
  full / not recording.  The silence says nothing about the light, and scoring
  0 invents a datum.

Excluding is right only in World B.  §"Root cause 2" asserted World B on the
strength of one scan: `P(meas>0)` for membrane opdets is 0.27/0.32/0.32 for
flashes of 1–3k/3–10k/>10k PE — *flat in brightness*, therefore duty cycle.

### That scan bins on the wrong variable

It bins by **flash total PE**, which the cathode XAs dominate.  What decides
whether *this* channel self-triggers is the light **on this channel**.  Binned
by per-channel predicted PE over the same 18 events (auto-selected bundles):

| pred PE on channel | 0–1 | 1–5 | 5–20 | 20–100 | 100–500 | >500 |
|---|---|---|---|---|---|---|
| P(cov=1), Arapucas | 0.576 | 0.771 | 0.766 | 0.817 | 0.868 | **0.984** |
| P(cov=1), PMTs | 0.252 | 0.601 | 0.682 | 0.550 | — | — |

Not flat.  It survives controlling for flash brightness: within the 1–3k
flash band alone, Arapuca coverage climbs 0.617 → 0.911 from pred <1 PE to
pred >100 PE.  Coverage is **downstream of the light**.

### Three measurements that close it

`analyze_coverage_deadtime.py` rebuilds each snippet's live interval from the
rawwf `timestamp`/`nsamp` in the chain's own tick base
(`PDVDOpWaveformSource.cxx:123-124`) and reproduces `OpFlashFinder`'s coverage
(`:711-763`) — **validation gate: 266071/266071 (100%)** (flash, opdet) cells
match the chain's own `flash_cov`.  Then:

1. **There is no dead time.**  Minimum gap between consecutive snippets on a
   channel = **0.336 µs** (1%ile 0.768 µs; 1.6% of gaps < 1 µs).  A channel
   re-arms essentially the instant a snippet ends — it is never busy when a
   flash arrives.  **World B has no mechanism.**
2. **Uncovered channels are quieter, not busier.**  Gap since their last
   snippet: median **93.8 µs** vs **54.2 µs** for covered ones, and 5.4% vs
   0.7% have seen nothing for >1 ms.  The opposite of the busy prediction.
3. **The threshold is ~1 PE.**  Over 43520 self-trigger snippets the total
   deconvolved PE inside the snippet has 25%ile **0.96 PE**, median **3.29
   PE**; 59% of triggers carry <5 PE.  A channel that did not fire really did
   see ~nothing.

The ~12% duty cycle (median 55 snippets × 16.384 µs / 7500 µs) is real, but it
is an **effect** of there being no light most of the time, not an independent
blindness.  Root cause 2 read "channel is rarely live" as "channel is randomly
deaf"; those are different statements.

### Consequences

- Dropping an uncovered channel discards the upper limit "< ~1 PE", so a
  prediction may over-shoot a silent channel for free — a one-directional bias
  toward over-prediction, the same direction as this doc's opening symptom.
- **969** uncovered Arapuca entries predict >20 PE (465 predict >100).  Under
  World A these are the model contradicting the hardware, not missing data.
- The Phase-2 chi2 win (**459.6 → 91.8**, ndf 18→13) is the warning, not the
  result: 5 channels carried ~370 of that chi2 because the prediction was
  wrong on them, and masking made the wrongness free.
- `coverage_min` = 1.0 with the min-over-ganged-sub-channels rule masked
  **551 entries that had measured light** (median 31 PE, max 3608) — data
  deleted because a ganged partner was quiet or the window overhung a snippet
  edge.  Indefensible under either world; fixed for free by keeping them.
- **The Phase-4 per-type refit is predicated on the falsified premise.**  Its
  premise is that membrane/PMT zeros are fake; they are real.  The f7c66ab8
  factors (kept by the 2026-07-15 ruling) were fit with those zeros IN, i.e.
  under the now-correct treatment.  A refit under coverage masking would be
  the biased one.

### Fix — the `coverage_mask_fit` op point (2026-07-16)

Labelling and masking were one knob; they are now two.  `use_coverage_flag`
keeps tracking coverage (dump `cov` array → ql_scan `nodata` label);
`coverage_mask_fit` decides whether an uncovered channel *also* leaves the fit.

| layer | knob | default | PDVD production |
|---|---|---|---|
| `QLMatching` | `coverage_mask_fit` | **true** (legacy drop ⇒ byte-identical) | **false** — keep at measured 0 |
| `QLMatching` | `pe_err_nodata` | −1 (disabled) | unset — not needed, see below |
| runner | `PDVD_QL_COV_MASK_FIT` | — | **0** (= keep) |

**The error is already right for PDVD.**  A kept no-data channel must carry
the threshold band, not a tight zero: at the C++ `pe_err_floor` default 0.3 a
5 PE over-prediction would cost chi2 ≈ 278 — over-punishing the true match
exactly as the pre-coverage path did.  PDVD's `qlmatching.jsonnet` already
sets `pe_err_floor: 2.0` with `pe_err_knee` at the C++ default 1.0, so a
measured 0 carries **±2 PE**, ≈2× the measured ~1 PE threshold.  `pe_err_nodata`
exists for detectors with a tighter floor; PDVD leaves it unset.

Gates: compiled config with the knob at default is **byte-identical** to HEAD
(`wct-clustering`, sat+cov on; diff of the compiled JSON is empty), key
present when on; `wcdoctest-match` 23/23; freshness proof
`local/lib/libWireCellMatch.so` 08:00 > `QLMatching.cxx` 07:58.  Knob-on smoke
(`work/039252_0_keepcov`, light `_spcov`, evt 298567): sentinel fires, and vs
`_am2` the **median ndf of auto-selected bundles rises 8 → 10** (the no-data
channels are back), contained bundles unchanged (5215 — containment is
geometry, not coverage), median chi2/ndf 5.02 → 5.59 (expected: the fit is now
held to more data), 23 clusters change flash, 3 lost, 0 gained; the §11 crosser
top:22 stays on the bright 274.4 µs flash.

Also collapsed three exact-duplicate `coverage_min` guard lines (differing only
in indentation, from the Phase-2 patch) while rewriting them — provable no-ops.

### Open items

- **`_am2` / `_spcov` dumps and the `candles-am2` round on port 5019 carry the
  masking** — they predate this ruling.  Needs a reprocess (owner is holding
  for other parts).
- **Partial coverage** (`0 < cov < 1`, 2.46% of cells) is now kept at face
  value, but such a channel saw only part of the flash window, so its measured
  PE is biased **low**.  Not addressed; the population is small.  A coverage-
  weighted error would be the principled treatment.
- **`pe_err_nodata` has no consumer today** (PDVD's `pe_err_floor` already
  supplies the band).  Retained as the explicit home for the threshold band so
  a future `pe_err_floor` retune cannot silently over-punish no-data channels.
- **The ~1 PE threshold is measured from snippet decon PE, not from the DAPHNE
  trigger configuration.**  Worth confirming against the run's actual threshold
  registers.
- The 6893 uncovered channel-flash entries with **no snippet anywhere in the
  event** (3.9% of uncovered) may be genuinely dead rather than quiet — the
  only real World-B population, identifiable per channel per event.

## Verification

- This doc's script regenerates every table/figure from the existing dumps
  (read-only).  Figures:
  - `lightpattern_fig1_raw_decon_cathode.png` — raw vs decon, railed +
    unrailed cathode channels.
  - `lightpattern_fig2_od7_template_bug.png` — ch1050/ch1051 same light,
    ×7 template-area bug.
  - `lightpattern_fig3_ch2020_inversion.png` — negative-area template
    corrupting the decon PE normalization.
  - `lightpattern_fig4_coverage_timeline.png` — snippet livetime vs the
    flash window; covered ⇔ measured>0.
  - `lightpattern_fig5_pattern.png` — measured vs masked vs unmasked
    prediction for flash 57.
  - `lightpattern_fig6_saturation_signature.png`
    (`scripts/saturation_signature.py`) — the saturation signature: (a) the
    positive flat-top clip at 16383, (b) the AC-coupling undershoot that
    follows it (the real "signal goes negative"), (c) the decon plateau,
    positive throughout the rail.
- Smoke case for all fixes: evt 298567 flash gid 57 (od4/6/7/8 sat-marked
  with full pred shown; od1/od3 coverage-masked; od7 ×2.66 after template
  v2).
- Status: knobs-off paths must stay **byte-identical** (hash_archive gates
  per detector); the knob-on operating point is **NOT bit-identical** by
  design and triggers the Phase-4 revalidation + factor refit.

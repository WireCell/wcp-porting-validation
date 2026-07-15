# PDVD run 39252 — photon-detector functionality audit (bottom/wall PMTs; top-wall X-Arapucas)

## Repro

```
cd pdvd/docs/qlmatch
python3 pd_functionality_audit.py
# reads work/{039252,039253,039349}_<idx>_satrep/calib-evt*.json
# writes pdfunc_perchannel.tsv, pdfunc_liveness.png,
#        pdfunc_wallxa_topbottom.png, pdfunc_crossrun.png
```
Inputs are our own pipeline products (saturation keep-and-mark `_satrep` calib
dumps). Cross-check of the measured `pe` against the independent pre-matching
raw opflash tensor done for evt 298567 (identical nonzero-channel set, identical
dead set, same brightest channel) — see [Method](#method).

## Question

A colleague reports that during **run 39252**:
1. the **PMT system (bottom PMTs and/or z-wall PMTs) may not be functional**, and
2. among the **wall (membrane) X-Arapucas the *top*-sided ones may be dead while
   the *bottom*-sided ones are OK to use**.

This note tests both claims against the data — flash-level measured light and
the Q/L-matching predicted-vs-measured light. **No configuration was changed**;
this is a data observation.

## TL;DR verdict

| Claim | Verdict |
|---|---|
| **Top-wall X-Arapucas non-functional, bottom OK** | **Not an outage — but the colleague's per-channel intuition has a real kernel.** The two brightest wall channels of *either* side are on the top (ch1=2311, ch3=2060 med-max PE — a dead detector reads ~0), so the top wall is clearly functional and the two sides respond identically as groups (meas/pred 1.01 vs 1.01). *However*, the two weakest wall-XA channels **are both on the top** (ch0 peerR 0.36, ch2 peerR 0.12); the bottom has no weak channel (excluding Ar-blind ch13). These two are equally weak in runs 39253 & 39349 — long-standing low-gain channels, not a 39252 top-side failure. |
| **PMT system non-functional** | **Live PMTs respond where light is expected (functional); a fixed dead/dim set matches the MC dead-set (long-known); the *absolute* yield is ~1/3 of the library and cannot be resolved here.** The live PMTs' per-channel meas/pred span ≈0.05–1.1 — an order of magnitude above the dead/dim channels (≈0.01) and with no single scale fitting them all — i.e. they produce signal roughly in proportion to the library across many track geometries, not mere noise. 4 bottom PMTs (24/27/28/34) are hard-dead (matches the MC dead-set exactly) and 3 more (16/17/33) are ~10–20× dim. What this data **cannot** decide: whether the PMTs' ~3× sub-library absolute yield (the ×0.352 factor) is genuine PMT degradation or library over-prediction — the cross-type normalization is untrusted. Everything (dead set, dim set, yield) is identical in 39253/39349, so **nothing here is specific to run 39252.** |

## Method

Channel map = WCT OpDet index 0–39 (the order of the dump `pe`/`pred_pe`
arrays). Every channel is reported with its (x,y,z) so the top/bottom labelling
is verifiable. Cathode is at x=0; top drift volume is +x, bottom is −x.

- **Cathode X-Arapuca** (type 0): ch 4–11, x≈0 — reference-bright, not under test.
- **Wall X-Arapuca TOP** (+x): ch 0,1,2,3 at x=+229/+306 cm.
- **Wall X-Arapuca BOTTOM** (−x): ch 12,13,18,19 at x=−201/−278 cm (ch13 Ar-blind).
- **z-wall PMT**: ch 14–17, 20–23 (bottom volume, x≈−206/−282).
- **bottom PMT**: ch 24–39 at x=−336.5 (behind the bottom anode).

**Stream A — raw liveness (primary, model-independent).** The measured `pe`
array is raw: hard-dead channels 24/27/28/34 read **exactly 0**, while dim-but-
alive channels read small-nonzero. Per channel, over all flashes of all events
per run: fraction of flashes with pe>0, and the median-over-events of the per-
event max PE. A **peer ratio** = channel brightness ÷ median of its same-type,
same-position peers → dead ⇒ 0, faulty/dim ⇒ ≪1, healthy ⇒ ~1. Verified against
the pre-matching raw opflash tensor (evt 298567): identical nonzero-channel set,
identical zero dead set, both brightest = cathode ch9.

**Stream B — matched predicted-vs-measured (supporting).** Sum(meas)/Sum(pred)
per channel over auto-selected matched flashes.

> **Circularity guard.** The dump `pred_pe` already folds in the fitted per-type
> data scale factors (cathode ×10.116, membrane ×1.655, PMT ×0.352 at QtoL
> 0.094, `qlmatching.jsonnet`), and run 39252 is *inside* that fit sample — so a
> bare meas/pred ≈ 1 is **not** evidence of health. We therefore read only
> (i) the **top-vs-bottom wall-XA comparison**, where both sides carry the *same*
> ×1.655 so the factor cancels and any asymmetry is contamination-free, and
> (ii) **per-channel relative** response. Stream B *absolute* normalization is
> aggregation- and library-dependent (the naive cathode sum-ratio here is 5.2,
> whereas the clean cathode-crosser closure in `../../ql_light_calib/fit_qtol_crossers.py`
> is 1.01) — so absolutes below are **not** a calibration and are not used for the
> PMT verdict. The PMT cross-type normalization is independently untrusted
> (`../pdvd-questions-dune.md` §2).

## Results — wall X-Arapucas (top vs bottom)

Run 39252, per channel (`peerR` = brightness vs same-side peers; meas/pred
carries ×1.655 on both sides, so it cancels in a top↔bottom comparison):

| side | ch | x (cm) | frac>0 | med-max PE | peerR | meas/pred | class |
|---|---|---|---|---|---|---|---|
| TOP | 0 | +305.6 | 0.21 | 747 | 0.36 | 0.83 | ok |
| TOP | 1 | +305.6 | 0.11 | 2311 | 3.09 | 0.87 | ok |
| TOP | **2** | +229.0 | 0.16 | **238** | **0.12** | 0.24 | **DIM** |
| TOP | 3 | +229.0 | 0.24 | 2060 | 2.76 | 1.95 | ok |
| BOT | 12 | −201.1 | 0.17 | 1275 | 1.08 | 0.72 | ok |
| BOT | 13 | −201.1 | 0.17 | 72 | 0.06 | — | Ar-blind (masked) |
| BOT | 18 | −277.7 | 0.20 | 1184 | 0.93 | 0.94 | ok |
| BOT | 19 | −277.7 | 0.15 | 1861 | 1.57 | 1.57 | ok |

**Group meas/pred (factor cancels): TOP 1.01 vs BOTTOM 1.01 — identical.**
(See `pdfunc_wallxa_topbottom.png`.)

Reading:
- **Both sides are functional.** The two brightest wall channels overall are
  ch1 and ch3 — *on the top side* (peerR ~3, med-max PE ~2000–2300); a
  non-functional detector reads ~0. Bottom has three healthy channels
  (12, 18, 19) plus the Ar-blind ch13. Flash-participation (frac>0) is the same
  ~0.11–0.24 on both sides, and the group meas/pred are equal to 1 %.
- **The colleague's per-channel intuition has a real kernel, though.** The two
  *weakest* wall-XA channels are **both on the top side**: ch2 (peerR 0.12,
  meas/pred 0.24 — a ~8× low-gain channel, matching the pre-existing note in
  `../pdvd-pd-channel-map.md` "ch2 measures ~10× below prediction") and ch0
  (peerR 0.36, modestly low). The bottom side has no weak channel (excluding the
  Ar-blind ch13). So a glance at the top wall *does* show its two dimmest members
  — which is likely what prompted the concern.
- **But it is not a run-39252 effect and not an outage.** ch0/ch2 are equally
  weak in all three runs (ch2 peerR = 0.12 / 0.23 / 0.10 across 39252/39253/39349;
  ch0 similarly), i.e. permanent low-gain channels; ch1 and ch3 remain bright in
  every run.

**⇒ The "top-wall X-Arapucas non-functional, bottom OK" hypothesis is not an
outage.** There is no top/bottom group asymmetry. What is real is that the two
long-standing weak wall channels (**ch0, ch2**) happen to sit on the top — a
per-channel gain issue present in all runs, not a top-side failure.

## Results — PMTs (z-wall + bottom)

Run 39252 (meas/pred carries ×0.352; per the guard, used only for relative /
liveness reading, not the PMT verdict):

**Hard-dead (zero readout, all three runs):** ch **24, 27, 28, 34** (all bottom
PMTs). Matches the documented dead set and the MC dead-set exactly
(`../pdvd-light-chain.md`).

**Dim (~1/10–1/20 of same-type peers, all three runs):** z-wall ch **16**
(peerR 0.05/0.05/0.05), ch **17** (0.18/0.14/0.30); bottom ch **33**
(0.08/0.10/0.13). Consistent hardware/gain faults, not run-specific.

**Alive and responding in proportion to the library (every run):** the remaining
z-wall PMTs (14, 15, 20, 21, 22, 23) and bottom PMTs (25, 26, 30, 31, 35, 36,
37, 38) plus the Ar-blind-but-responding 29/32/39.

Reading — strongest evidence first:

- **The live PMTs respond where light is expected, not just "somewhere".** The
  per-type ×0.352 is a *single* number — it does not fit per-channel. Yet the
  live PMTs land at per-channel meas/pred ≈0.05–1.1 (e.g. ch15 0.91, ch21 1.14,
  ch25 0.61, ch35 0.60, ch38 0.82), an **order of magnitude above** the dead/dim
  channels (ch16 0.007, ch17 0.02, ch33 0.015), with **no single scale fitting
  them all**. So across many different track geometries the live PMTs produce
  signal roughly in proportion to the library's per-channel expectation — a
  genuine response test, stronger than "registers PE and tracks its peers"
  (a uniformly-miscalibrated-but-live system would also do that).
- **The dead set is anchored, not guessed.** ch 24/27/28/34 read exactly 0 and
  match the **MC dead-set** exactly (`../pdvd-light-chain.md`) — a long-known
  hardware state, not a 39252 surprise. ch16/17/33 are ~10–20× dim in every run.
- **The one thing this data cannot resolve: absolute PMT yield.** Even with
  ×0.352 already folded in, the PMT group meas/pred is 0.22 (z-wall) / 0.42
  (bottom) and PMTs carry **<2 % of total PE**. Whether that ~3× sub-library
  absolute level is real PMT degradation *or* the library over-predicting the PMTs
  cannot be separated here — the cross-type (PMT-vs-XA) normalization is
  independently untrusted (`../pdvd-questions-dune.md` §2), and remote small
  photocathodes ~340 cm from the cosmic light source are geometrically starved.
  This is the honest limit of the reconstructed light: it can say the live PMTs
  *respond proportionally*, not that their *absolute efficiency* is nominal.
- **Nothing about the PMT state is specific to run 39252** — the dead set, the
  dim set, the proportional response, and the sub-2 % PE share are the same in
  39253/39349 (see below).

(See `pdfunc_liveness.png` for per-channel brightness and `pdfunc_crossrun.png`
for the per-run liveness grid.)

## Cross-run baseline (is any anomaly specific to 39252?)

For every candidate channel the liveness pattern is **stable across
39252 / 39253 / 39349** (see `pdfunc_crossrun.png` and `pdfunc_perchannel.tsv`):
the same 4 dead PMTs, the same dim channels (16/17/33, and top-XA ch2), the same
healthy channels. Run 39349 shows slightly higher frac-nonzero across many PMTs
(brighter/more events), but the same channels are alive/dead/dim. **No
run-39252-specific outage is visible in either the PMTs or the wall X-Arapucas.**

Caveat on scope — **stability is not health.** All three available runs are
adjacent and share the same PD state, so the cross-run grid can only say the
39252 state is *not anomalous relative to its neighbours* — it cannot
discriminate "fine everywhere" from "degraded everywhere." For the dead
channels, the reference that actually anchors "long-known" is the **MC dead-set
match** (above), not the cross-run grid. If the colleague's information comes
from DAQ/HV logs or an earlier run outside this set, that is a comparison the
reconstructed light here cannot make.

## Conclusion

- **Top-wall X-Arapucas:** functional — its two members are the brightest wall
  channels of either side, no top/bottom group asymmetry (meas/pred 1.01 vs 1.01).
  The real per-channel kernel behind the concern: the two long-standing *weak*
  wall channels (**ch0, ch2**, ~3–8× low) both sit on the top; they are equally
  weak in all three runs, so a per-channel gain issue, not a top-side outage.
- **PMTs (bottom + z-wall):** the live channels **respond in proportion to the
  library** (per-channel meas/pred an order of magnitude above the dead/dim set,
  not fittable by one scale) — so they are producing physics signal, not noise.
  A fixed dead set (24/27/28/34, matching MC) and dim set (16/17/33) is
  long-known. The **absolute** PMT yield is ~1/3 of the library and this dataset
  **cannot** separate genuine degradation from library over-prediction (untrusted
  cross-type normalization + geometrically starved remote photocathodes).
- **Overall:** the data does not show "PMTs / top-wall XAs became non-functional
  in run 39252." It shows live PMTs and both wall-XA sides responding, a fixed
  set of dead/dim channels (already handled by the static mask
  `[13,24,27,28,29,32,34,39]` + per-event `auto_mask`), and an unresolved
  absolute PMT-yield question — all identical in the two adjacent runs.

## Config implications (note only — no change made)

The already-deployed masking (`ch_mask_base = [13,24,27,28,29,32,34,39]` plus the
dynamic same-type `auto_mask`) already removes the hard-dead PMTs and Ar-blind
channels. ch2 (dim top-XA), ch16/17 (dim z-wall PMT) and ch33 (dim bottom PMT)
are caught per-event by the dynamic auto-mask rather than statically. If the
owner wants ch2/16/17/33 permanently excluded, that would be a one-line addition
to `ch_mask_base` in `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet` —
but it changes reconstruction output and must go in as a default-OFF knob with a
byte-identical gate, so it is **not** done here. This note is a data observation
only.

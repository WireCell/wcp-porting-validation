# PDHD W-plane (collection) SP signal break — diagnosis and fix

**Symptom** (reported 2026-06-10, `pics/Picture1.png` / `pics/Picture2.png`):
strong, prolonged (~1000+ tick) W-plane track signals survive noise filtering
(black, post-NF raw) but the SP output (red, `gauss`) is fragmented into small
islands with most of the charge gone.

| event | work idx | APA | channel |
|---|---|---|---|
| run 027409 evt 40920 | `work/027409_6` | APA3 | 9543 (W) |
| run 027409 evt 40924 | `work/027409_7` | APA1 | 4532 (W) |

(The report quoted "8532" for the first event; the plot title and the APA3 W
channel range [9280, 10240) identify the channel as 9543 — 8532 decodes to a
V-plane channel and shows an unrelated, much milder induction-path topology.)

A prolonged signal in time on a collection channel = a track segment running
nearly parallel to the drift direction. Both events are exactly that.

---

## 1. Collection-plane SP path (APA1–3; standard `[U,V,W]` slot order)

Unlike induction, the collection plane has a single, simple ROI chain
(`OmnibusSigProc.cxx`, plane==2 branch at :1909–1921 and the refinement loop
at :2005–2055):

```
decon_2D_init                 (Wire_col wire filter)
decon_2D_tightROI             (Wiener_tight_W; NO low-frequency filter)
find_ROI_by_decon_itself      threshold = troi_col_th_factor * rms + 1 = 5*rms+1
                              (ROI_formation.cxx:455; rms from cal_RMS :364)
CleanUpROIs                   -> "cleanup_roi" debug tag
BreakROIs x r_break_roi_loop  -> "break_roi_1st/2nd"
CheckROIs / CleanUpROIs        (CheckROIs: NO-OP on plane 2 — loose ROIs only)
ShrinkROIs                     (NO-OP on plane 2 — loose ROIs only,
                                ROI_refinement.cxx:1849-1870)
CleanUpCollectionROIs
ExtendROIs                    -> "extend_roi" (final mask)
decon_2D_hits / decon_2D_charge + apply_roi -> wiener / gauss
```

So on the collection plane only **ROI finding**, **BreakROI** and the
**CleanUps** can erode coverage.

## 2. Diagnosis: per-stage ROI dump

Tooling added for this study (both committed):

* `run_nf_sp_dnnroi_evt.sh --roi-debug` — bypasses DNN-ROI/L1SP (their
  FrameMergers' mergemaps drop auxiliary tags) while forcing the same
  OmnibusSigProc override as the production DNN chain
  (`use_roi_debug_mode` + `use_multi_plane_protection`, so the SP-internal
  processing is identical; W output is unaffected by the bypassed stages,
  which only touch U/V) and sinks every per-stage ROI tag into the frame
  archive (`sp_roi_debug_sink` TLA in `wct-nf-sp-dnnroi.jsonnet`).
* `sp_w_roi_diag.py` — per-channel, per-stage coverage/segment report + plot.

```
./run_nf_sp_dnnroi_evt.sh --roi-debug -a 3 -O _roidbg 027409 6
./sp_w_roi_diag.py work/027409_6_roidbg 3 9543 -b work/027409_6 -w 1700:3900
```

Result (pre-fix), window coverage of the signal region:

| stage | ch 9543 (evt 40920) | ch 4532 (evt 40924) |
|---|---|---|
| tight decon waveform | (continuous signal) | (continuous signal) |
| `cleanup_roi` | **3.8 %, 6 islands** | **6.3 %, 8 islands** |
| `break_roi_1st` | 3.8 % (unchanged) | 6.3 % (unchanged) |
| `break_roi_2nd` | 3.8 % (unchanged) | 6.3 % (unchanged) |
| `shrink_roi` | 3.8 % (unchanged) | 6.3 % (unchanged) |
| `extend_roi` (final) | 28.8 % | 21.3 % |
| `gauss` | 22.6 %, 19 fragments | 16.7 %, 10 fragments |

Two immediate conclusions:

1. **The signal is killed at the initial tight-ROI finding** — coverage is
   already 3.8 %/6.3 % at `cleanup_roi`. BreakROI never fires on these
   channels (the islands are single-peak).
2. The partial recovery between `shrink_roi` and `extend_roi` is the
   **MP2/MP3 multi-plane protection** adding back coherent chunks (e.g.
   [3347,3823] on ch 9543) — protection working as designed, but it cannot
   rebuild the whole signal.

## 3. Root cause: `cal_RMS` percentile estimate breaks at high occupancy

`ROI_formation::cal_RMS` (ROI_formation.cxx:364) estimates the noise as the
(16,50,84)-percentile spread, then takes a second moment over samples within
5×that. A strong positive signal occupying **more than ~16 % of the
waveform** puts the 84th percentile *inside the signal*, inflating the first
estimate so much that the 5× cut no longer excludes the signal, and the
second moment inflates with it. Measured on the tight decon waveform:

| channel | cal_RMS (production) | cal_RMS (signal-free region) | inflation | threshold 5·rms+1 | signal median | signal window above threshold |
|---|---|---|---|---|---|---|
| 9543 | **1594** | 170 | 9.4× | 7973 | 3950 | 4 % |
| 4532 | **1769** | 186 | 9.5× | 8846 | 4378 | 4 % |

The ROI threshold lands at ~2× the signal's own median — only the tallest
dE/dx peaks poke above it, giving exactly the observed islands. With a clean
RMS the threshold (~850–930) sits well below the signal body (75–85 % of the
window passes) and the ROI is continuous.

This also explains why the failure is rare: it needs a strong track to
stay on one W channel for ≳1000 ticks (≳16 % of the 6000-tick readout),
i.e. a track closely aligned with drift.

## 4. The fix (two parts, both default-ON for PDHD and PDVD)

### Part 1 — `roi_mad_rms` (toolkit C++, OmnibusSigProc/ROI_formation)

New OmnibusSigProc knob `roi_mad_rms` (C++ default **false** = bit-identical
legacy). When true, `cal_RMS`'s first-pass estimate becomes the **median
absolute deviation** (×1.4826 for Gaussian-equivalent sigma), robust up to
50 % signal occupancy; the second-moment pass is unchanged. Applies to all
planes (induction tight/loose thresholds use the same `cal_RMS`, so they gain
the same robustness; on quiet channels MAD and the percentile spread agree to
within finite-sample noise).

Replaying the exact ROI-finding loop on the captured tight decon waveforms
with the MAD RMS:

| channel | rms | threshold | signal-window ROI coverage |
|---|---|---|---|
| 9543 | 205 | 1026 | 3.8 % → **48 %**, core ROI [2588,3593] contiguous |
| 4532 | 225 | 1127 | 6.3 % → **51 %**, core ROI [4393,5341] contiguous |

### Part 2 — `w_col_break_roi_tune` (cfg only)

With Part 1 in place, the long multi-peak W ROI now *survives to refinement*
— where `BreakROI` would do to it exactly what the APA0 study
(`sp-apa0-plane2.md` §7) documented: find the dE/dx spikes as "peaks",
declare the track continuum between them "valleys", subtract a
valley-to-valley **linear baseline that is real charge**, and re-fragment
the signal. On induction planes that baseline subtraction is justified (the
tight/loose LF filters distort the baseline inside long ROIs); the collection
plane is deconvolved **without any LF filter**, so its baseline needs no such
fix and the subtraction only removes signal.

New `make_sigproc` arg `w_col_break_roi_tune` (default **true**) in
`cfg/pgrapher/experiment/pdhd/sp.jsonnet` and
`cfg/pgrapher/experiment/protodunevd/sp.jsonnet`: emits
`r_break_roi_loop_planes: [2, 2, 0]` — BreakROI disabled on the collection
slot only, U/V keep the production 2 loops. Set `false` ⇒ key omitted ⇒
scalar `r_break_roi_loop` applies ⇒ byte-identical config. On PDHD it is
gated to APA1–3 (`anode.data.ident != 0`); APA0's swapped slot order is
already handled by `apa0_w_roi_tune` (which disables BreakROI on its W,
slot 1). PDVD uses standard `[U,V,W]` order on all 8 anodes; being a
vertical-drift detector, tracks along drift (near-vertical cosmics) are
especially common there, so the same failure class applies.

### Knob summary

| knob | where | default | effect |
|---|---|---|---|
| `roi_mad_rms` | OmnibusSigProc → ROI_formation | C++ false; **PDHD/PDVD cfg true** | MAD-based `cal_RMS`, robust ROI thresholds |
| `w_col_break_roi_tune` | pdhd/protodunevd `sp.jsonnet` | **true** | `r_break_roi_loop_planes=[2,2,0]` (collection BreakROI off) |

## 5. Verification

* Pre-fix stage table and RMS numbers: §2–§3 (from `--roi-debug` runs on the
  production binary/config, whose W gauss segments byte-match the production
  archives in `work/027409_6` / `_7`).
* **Post-fix per-stage rerun** (`--roi-debug`, fix ON), signal-window
  coverage:

  | stage | ch 9543 | ch 4532 |
  |---|---|---|
  | `cleanup_roi` | 3.8 % → **50.3 %**, core [2584,3624] contiguous | 6.3 % → **53.5 %**, core [4370,5377] contiguous |
  | `gauss` | 22.6 % → **55.8 %**, charge sum 1.41M → **4.39M** | 16.7 % → **57.8 %**, charge sum 1.63M → **5.07M** |

  (`break_roi_*` tags are absent for W post-fix — BreakROI disabled, the
  loop never runs on slot 2.  The prolonged signal body is one continuous
  gauss chunk: [2564,3659] / [4349,5431].)
* **Toggle-off byte-identicality**: full-chain rerun with the new binary and
  `--w-fix off` reproduces the pre-fix production archives **byte-identically**
  (every npy member equal: raw/wiener/gauss, all planes, both events).
* Config level: knobs-off configs compile byte-identical to pre-change
  (wcsonnet diff on the full PDHD chain and PDVD `wct-nf-sp.jsonnet`);
  knobs-on diffs are exactly the two new keys.
* Production `work/027409_6` / `_7` frames + magnify regenerated with the fix
  ON (pre-fix outputs preserved in `bak-pre-wfix/`).

## 6. Third root cause (upstream): `rebase_waveform` baseline over-subtraction

**Symptom** (reported 2026-06-16, `pics/Screenshot 2026-06-16 at 10.32.03 PM.png`):
run 029107 evt 983, **APA3 ch 9348 (W)** — same prolonged-W class, but seen on
the **DNN-ROI** output (where W is always the gauss passthrough,
`dnnroi_pp.jsonnet`: `dnnspNw = standard SP gauss`). The §3/§4 fixes were
already ON, yet the deconvolved signal kept only the bright head (ticks 0–500)
and the two sharp out-of-track spikes, dropping the prolonged tail entirely.

This is **upstream of §1–§4**: the kill happens in `OmnibusSigProc::load_data`,
*before* deconvolution and ROI, in `rebase_waveform` (`OmnibusSigProc.cxx`).
Rebasing is on for all planes by default (`m_rebase_planes{0,1,2}`, since
`0c1d9f72`, Sep 2025) and subtracts a per-channel linear baseline tied to
front/back `rebase_nbins=200`-tick window anchors.

### Mechanism

The `sigmask` anchor (default since `97f9d233`, Jun 2026) masks window samples
with `|x − p50| > rebase_nsigma·σ`, `σ` from the 16/50/84 percentiles. ch9348's
prolonged bright signal sits **in the front window and occupies ~25 % of the
readout**, so the 84th percentile is pulled *into the signal* (p84≈214 vs
p50≈−0.1, p16≈−8.7). The symmetric `σ = √((p84−p50)²+(p50−p16)²)/2)` is then
inflated to ~151 → cut `4σ≈606 ADC`, wide enough that **197/200 front-window
signal samples pass the mask**. The front anchor is biased high (≈365) while the
clean back anchor ≈−2.4, so the subtracted baseline tilts down (−371 @t0 →
−276 @t1500) and drives the prolonged tail **negative before decon** — ROI can
never see it. (Same trigger condition as §3: a track on one W channel for
≳16 % of the readout. `cal_RMS` and `rebase` are two independent percentile-σ
estimators that both break at the same high-occupancy regime; §3 fixed the
post-decon one, this fixes the pre-decon one.)

Confirmed by disabling W rebase (`rebase_planes:[0,1]`) on a one-off rerun:
ch9348 gauss charge **0.91M → 6.49M (7.1×)**, tail t1000–1500 **0 → 985k**.

### The fix — robust `sigmask` σ (toolkit C++, replaces the old formula)

`rebase_waveform`'s `RB_SIGMASK` σ is changed from the symmetric RMS of the two
half-spreads to the **cleaner (smaller) half-spread**:

```
sigma = min(p84 − p50, p50 − p16);   // was sqrt(((p84-p50)^2 + (p50-p16)^2)/2)
```

One-sided signal inflates only *its own* half-spread, so `min()` stays a
noise-only scale (here 8.6 not 151 → cut≈34), the front-window signal is masked,
the window-widen loop finds no clean samples and falls back to the row median
(≈0) → no tilt → tail preserved. This **replaces** the buggy symmetric σ (it is
not a toggle): the symmetric form is biased low-or-high by any one-sided pulse
and the robust form is never worse. `rebase_method=median` is unchanged;
`mean` remains removed. Limitation: this does **not** help symmetric/bipolar
high-occupancy windows, where both half-spreads inflate.

### Verification

* ch9348 with the new σ reproduces the rebase-OFF result: gauss **6.47M** vs
  6.49M, `max|Δ|=17 ADC` — the over-subtraction is gone.
* Do-no-harm: over all 2560 anode3 channels, only the prolonged-track cluster
  (ch9346–9349) changes substantially (correct recovery); induction U/V are
  untouched apart from ≤~2 % on 9 channels (the slightly tighter, better σ).
* The `sigmask` string and default are unchanged — `sigmask` *is* the robust
  method now.

## 7. Related

* `sp-apa0-plane2.md` §7 — the sibling APA0 W tune (induction-path
  refinement erosion; different mechanism, same "prolonged W signal" theme).
* `sp.md` — general PDHD SP chain reference.

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
* Post-fix: re-run of the `--roi-debug` diagnosis and of the full DNN chain +
  magnify for both events. *(Pending at the time of writing: the input
  frames under `input_data_14_old_coh_grouping` →
  `/nfs/data/1/xning/wirecell-working/data/` became unreadable mid-study
  (`drwxrwxr--`, group 27658) — re-run as soon as access returns. The MAD
  replay in §4 used the captured tight-decon waveforms and the exact C++
  ROI-finding logic.)*
* Toggle-off path: `roi_mad_rms=false` takes the untouched legacy branch in
  `cal_RMS`; `w_col_break_roi_tune=false` omits the per-plane key entirely.
  Both compile to configs byte-identical to pre-fix production.

## 6. Related

* `sp-apa0-plane2.md` §7 — the sibling APA0 W tune (induction-path
  refinement erosion; different mechanism, same "prolonged W signal" theme).
* `sp.md` — general PDHD SP chain reference.

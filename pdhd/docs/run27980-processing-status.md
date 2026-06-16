# PDHD run 27980 — toolkit processing status (charge + light, +x and −x)

What is reconstructed for run 27980 with the toolkit, where the −x optical side
stands, and the two deliverables produced from it: a 10-event Bee link and the
ql_scan hand-scan dumps.

> Companion docs: `pdhd-light-raw-data.md` (§7 = −x SPE coverage + full-stream
> readout), `pdhd-light-flash-run-comparison.md` (cross-run LArSoft flash stats),
> `qlmatching-chain.md`, `which-pd-side-lights.md`.

---

## 1. Charge — complete

NF/SP (DNN-ROI), imaging, and 4-stage clustering are done for **all 31 events**
under `pdhd/work/027980_<idx>/`: `protodunehd-sp-dnnroi-frames-anode{0..3}.tar.bz2`,
`clusters-apa-apa{0..3}-ms-active.tar.gz`, and `mabc-*.zip`. Nothing to re-run.

## 2. Light — +x and the −x **snippet** PDs (data-driven), not full-stream

The toolkit light chain is
`PDHDOpWaveformSource → OpDecon → OpHitFinder → OpFlashFinder`. The source reads
**every** `decoana/run_27980_evt_<art>/ch<N>` self-triggered snippet — it is
**not** +x-only. It reconstructs whatever snippets `decoana` carries, on **both**
walls. The two −x channel regions behave differently:

| −x region | readout mode | toolkit status |
|---|---|---|
| **80–119** (self-triggered PDs) | 1024-tick **snippets** | **reconstructed** wherever `decoana` has the snippets (event-sparse, see §3) |
| **120–159** (full-stream PDs) | continuous **full-stream** (343 808 samples / event) | **not reconstructed** — `decoana` has zero channels ≥120; the raw stream sits undecoded in `rawdump/raw_waveform`. Deferred: `pdhd-light-raw-data.md` §7.4 |

So the answer to "are the two −x regions both self-trigger or full-stream?" is:
**80–119 = self-trigger (already reconstructed); 120–159 = full-stream (deferred,
needs a new software self-trigger/windowing decoder).** The deliverables below
use +x and the −x snippet side; full-stream 120–159 is a separate go/no-go task.

## 3. Per-event −x snippet coverage

`decoana` carries the −x snippet PDs in only **11 of 31 events** (3 rich). Note this is
a **`decoana` coverage gap, not a property of the −x data**: the −x self-stream PDs
actually self-trigger and reconstruct in **all 31 events** in the raw stream
(`rawdump/raw_waveform`) and the LArSoft OpHit reco — see
`pdhd-pd-activity-per-event.md`, where −x is the *most uniform* side of the detector.
`decoana` simply did not deconvolve/store those −x snippets. Per-event −x snippet
channel count and the reconstructed +x / −x total PE (from the QL calib dumps,
`group02` = −x side):

| work idx | art evt | −x snippet ch | +x PE | −x PE |
|---|---|---|---|---|
| 2  | 24  | 2  | 28084 | 400   |
| 9  | 80  | 1  | 33815 | 130   |
| **12** | **104** | **32** | 0 | **33719** |
| **13** | **112** | **34** | 0 | **9403** |
| **14** | **120** | **26** | 2155 | **15983** |
| 18 | 152 | 2  | 16646 | 967   |
| 20 | 168 | 6  | 10460 | 4153  |
| 21 | 176 | 3  | 14186 | 3987  |
| 25 | 208 | 1  | — | — |
| 27 | 224 | 2  | — | — |
| 30 | 256 | 1  | — | — |

art 104 and 112 show 0 +x **not** because they are −x-dominated, but because the **+x
self-trigger readout dropped out** for those events — a DAQ artifact. Confirmed in
`pdhd-pd-activity-per-event.md` §5: the raw `rawdump/raw_waveform` has **zero** +x
channels in evt 104/112 (the +x dropout spans the contiguous block 104/112/120/128),
while the trigger metadata is identical to normal events. That same +x readout dropout
is why `decoana` has 0 +x channels there — one root cause. The remaining 20 events are
+x-only **in `decoana`** even though the raw stream and OpHit reco carry the −x
self-stream for them too (the `decoana` coverage gap above), **not** because the −x PDs
were dark. `work_idx → art`: idx 0–28 = art 8…232 (step 8), idx 29 = art 248
(gap at 240), idx 30 = art 256.

## 4. Deliverable A — 10-event Bee link

A curated 10-event set spanning both walls — strong +x (idx 0/1/2/9/18), the 3
−x-rich events (12/13/14), and mixed (14/20/21):

**https://www.phy.bnl.gov/twister/bee/set/d8ad88e6-80f2-46bb-88cf-21fefb4d0428/event/list/**

Per event the link carries `imaging-group02/13`, `clustering-group02/13`,
`clustering-global`, `channel-deadarea-group02/13`, and the optical **`op`**
instance (measured + Q/L-predicted PE). Regenerate with:

```
cd pdhd
./run_clus_evt.sh -calib -op 27980 0   # … and 1 2 9 12 13 14 18 20 21
./run_bee_combined_evt.sh -e 0,1,2,9,12,13,14,18,20,21 27980
```

The new `-e idx,idx,...` flag on `run_bee_combined_evt.sh` restricts the link to a
subset; with no `-e` the script uploads every discovered event as before.

## 5. Deliverable B — ql_scan results

Q/L matching with `-calib` writes, per drift side, the hand-scan calibration dumps
`work/027980_<idx>/calib-evt<art>-group{02,13}.json` (opdets, flashes, clusters,
and the candidate **bundles** = Q/L matches). They exist for the 10 curated events
and carry the −x snippet PE (e.g. art 104 → 33719 PE on ch 80–119). Serve the
viewer over the 10-event set with:

```
cd pdhd
./ql_scan/serve_ql_scan.sh 5015 --tag data \
    work/027980_{0,1,2,9,12,13,14,18,20,21}/calib-evt*-group*.json
# from a laptop:  ssh -L 5015:localhost:5015 user@wcgpu1
#                 → http://localhost:5015/ql_scan_viewer
```

---

## 6. Flash multiplicity & PE audit — "too many flashes" and why

Run with the **all-PD** chain (`work/027980_allpd<idx>/opflash_pdhd-allpd-wct.tar.gz`,
31 events; this reconstructs **both** walls including the −x full-stream 120–159, so
it supersedes the "deferred" note in §2). Per flash we read the `opflash` matrix
(col 0 = time, cols 1–160 = PE per opdet) and `group_by_side` puts every flash on
one wall: **+x = ch 0–79, −x = ch 80–159**. "nPD" = opdets with **≥ 0.5 PE**
(matches `refine_fired_pe`). Script: `/home/xqian/tmp/analyze_27980_flashes.py`
(+ `analyze_27980_neg_split.py`).

### What cuts exist today

The toolkit `OpFlashFinder` (larana port, `flash/src/OpFlashFinder.cxx`) applies
**only** `flash_threshold = 3.5 PE` (`OpFlashFinder.h:39`) to total flash PE, plus
the already-on `flash_refine` 8 µs satellite merge and `remove_late_light`. There is
**no minimum-channel (multiplicity / nPD) cut at all.** The MicroBooNE prototype
(`prototype_base/2dtoy/src/ToyLightReco.cxx:765`) by contrast forms a flash only on
`pe >= 6 && mult >= 3` — total PE ≥ 6 **and** ≥ 3 fired channels (a per-channel
"fired" = content > 1.5 PE, `prototype_base/data/src/Opflash.cxx:77`). The missing
multiplicity cut is the headline gap.

### Per-side counts (31 events, after `flash_refine`)

| wall | flashes | /event | median PE | median nPD | single-PD | nPD≤2 | PE≤5 |
|---|---|---|---|---|---|---|---|
| **+x** (0–79, all snippet) | 6 263 | **202** | 12.4 | 2 | 42.0 % | 51.5 % | 18 % |
| **−x** (80–159, snippet+full-stream) | 18 013 | **581** | 5.4 | 2 | 48.9 % | 77.6 % | 44 % |

### Diagnosis (the three hypotheses)

1. **PE threshold too low — YES.** 3.5 PE is barely above noise; on −x **44 %** of
   flashes are ≤ 5 PE, on +x 18 %.
2. **nPD as low as 1–2 — YES, dominant.** With *no* multiplicity cut, **42 % (+x) /
   49 % (−x)** of flashes fire a **single** PD; over half are ≤ 2 PD. This is the
   biggest single driver of the flood.
3. **Nearby-in-time not merged — NO (minor).** `flash_refine` (8 µs) already ran;
   the residual consecutive-flash time gaps are median **10.9 µs (+x) / 6.8 µs (−x)**,
   with only ~1 % (+x) / ~2 % (−x) under 0.5 µs. The flood is genuinely many
   distinct low-nPD/low-PE flashes, not un-merged duplicates.

### Why −x is 3× +x: it is a full-stream characteristic, not a threshold problem

Splitting the −x flashes by which sub-range carries their PE:

| −x flash lights… | flashes | % of −x |
|---|---|---|
| **only** ch 120–159 (full-stream) | 13 327 | **74.0 %** |
| only ch 80–119 (snippet) | 1 493 | 8.3 % |
| both | 3 193 | 17.7 % |

**83.9 %** of −x flashes are PE-dominated by the full-stream sub-range, and **89.5 %**
of all single-PD −x flashes live in 120–159. Full-stream-dominated flashes are dimmer
(median 5.1 PE vs 9.4) and lower-nPD (median 1 vs 2) than the −x snippet ones. So the
−x flood is overwhelmingly a **full-stream (OpRoi) reconstruction characteristic** —
a uniform PE/nPD cut will tame it, but the root cause sits in the full-stream path and
may warrant OpRoi/OpHit-level tuning too (the snippet −x sub-range behaves like +x).

### Proposed cut: nPD > 4 AND PE > 30

| wall | before | after cut | /event after |
|---|---|---|---|
| **+x** | 6 263 (202/evt) | **1 282** (20.5 %) | **41** |
| **−x** | 18 013 (581/evt) | **1 518** (8.4 %) | **49** |

The bright real flashes are untouched — e.g. event 160 +x keeps its 1411 / 3002 /
14352 / 5378 / 2631 PE flashes; the cut removes the 1-PD, few-PE tail. Tradeoff: a
genuinely localized bright flash concentrated on ≤ 4 PDs (e.g. a single-PD 54 PE
afterpulse-like hit) is also dropped — the same philosophy as the prototype's
`mult ≥ 3`.

### Recommendation (not yet implemented)

Add the cut as a **togglable knob** on `OpFlashFinder` (e.g. `min_fired_pds`,
`min_total_pe`, with the 0.5 PE fired threshold reusing `refine_fired_pe`),
**default OFF so existing configs stay byte-identical**, and enable it for PDHD via
`flash.jsonnet` — per the project's toggleable-behavior convention. Numbers above
are the expected reduction at `min_fired_pds = 4` (nPD > 4) and `min_total_pe = 30`.

---

## Appendix — provenance

| item | source |
|---|---|
| raw light ROOT | `…/data/hd/run027980/np04hd_raw_run027980_0000_…_final.root` |
| −x snippet vs full-stream | `decoana` channels (≤119 only) vs `rawdump/raw_waveform` (120–159 = 343 808-sample streams) |
| per-event +x/−x PE | `work/027980_<idx>/calib-evt<art>-group02.json` (`group02` = −x side) |
| Bee link | `run_clus_evt.sh -calib -op` + `run_bee_combined_evt.sh -e …` |
| ql_scan dumps | `run_clus_evt.sh -calib` → `calib-evt<art>-group{02,13}.json` |

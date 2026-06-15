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

The −x snippet PDs self-trigger only in **11 of 31 events**; 3 are rich. Per-event
−x snippet channel count and the reconstructed +x / −x total PE (from the QL calib
dumps, `group02` = −x side):

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

art 104 and 112 are genuinely −x-**dominated** (their `decoana` has 0 +x channels),
not artifacts. The remaining 20 events are +x-only in `decoana` (the −x PDs did not
self-trigger). `work_idx → art`: idx 0–28 = art 8…232 (step 8), idx 29 = art 248
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

## Appendix — provenance

| item | source |
|---|---|
| raw light ROOT | `…/data/hd/run027980/np04hd_raw_run027980_0000_…_final.root` |
| −x snippet vs full-stream | `decoana` channels (≤119 only) vs `rawdump/raw_waveform` (120–159 = 343 808-sample streams) |
| per-event +x/−x PE | `work/027980_<idx>/calib-evt<art>-group02.json` (`group02` = −x side) |
| Bee link | `run_clus_evt.sh -calib -op` + `run_bee_combined_evt.sh -e …` |
| ql_scan dumps | `run_clus_evt.sh -calib` → `calib-evt<art>-group{02,13}.json` |

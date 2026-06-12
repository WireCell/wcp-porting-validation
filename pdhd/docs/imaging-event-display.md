# PDHD imaging event display

`pdhd/img_plot/` — an interactive Bokeh viewer for PDHD imaging results,
ported from the mature `pdvd/img_plot` (same three linked views and feature
set), served on port **5013**.  This doc records the PDHD specifics; the
generic provenance story (stepped sampling, zero-charge blobs, dead-wire
splits) is documented in detail in `pdvd/docs/imaging-event-display.md` and
applies unchanged.

## Served event and data provenance

**Run 027409 event index 0** (art event number 40896; 14 mV/fC FE gain,
pre-flip coherent-noise grouping — the run scripts auto-derive both from the
`input_data_14_old_coh_grouping` root).

The full chain was regenerated on 2026-06-09 with the then-current toolkit
build (`apply-pointcloud`, includes the now-default APA0 W-plane ROI tune),
starting from the orig frames — nothing reused from older productions:

```bash
cd pdhd
./run_nf_sp_dnnroi_evt.sh 027409 0        # NF+SP+DNN-ROI from orig frames (GPU)
./run_img_evt.sh -d on 027409 0           # imaging from the DNN-ROI SP frames
./run_clus_evt.sh 027409 0                # MABC clustering -> mabc-all-apa.zip
./run_sp_to_magnify_evt.sh -d on 027409 0 # magnify-...-apa{0..3}-dnnroi.root
cd img_plot
./preprocess_event.py                     # -> cache/evt0.npz (+ .json sidecar)
./serve_img_viewer.sh 5013
```

All artifacts live in `work/027409_0/`.

Note `run_sp_to_magnify_evt.sh` needed a fix this round: its `find_evtdir`
still looked only under `input_data/`, predating the reorganization into
`input_data_<gain>_<old|new>_coh_grouping/` roots; it now scans all
`input_data*` roots like `_runlib.sh` does.

## What the viewer shows (identical feature set to PDVD)

1. **2D blob view** (Z–Y) per time slice: fired U/V/W wires as ±half-pitch
   cells, dead channels filled grey, blob outlines (solid black = charged,
   **dashed magenta = kept zero-charge ghosts**, hideable with their points),
   the slice's stepped sampling points in orange, per-plane fired-channel
   ranges, tap-a-wire channel selection.
2. **3-D point projections** X-Y / Z-Y / X-Z with window filters and
   current-slice highlight.
3. **Waveforms** from the per-APA Magnify ROOT (lazy): 1-D per channel with
   auto-ranged ADC axis, 2-D U/V/W-vs-T with slice tick band, `T_bad<N>` dead
   channels overlaid (grey actual waveform in 1-D, grey bars in 2-D).
   Frames available in the generated ROOTs: `gauss` (= DNN-ROI output),
   `wiener`, `raw` (post-NF), `orig`, `threshold`.
4. **Position lock**: type X/Y/Z (cm, MABC/Bee convention) — auto-detects
   APA/face/slice and drives all views.

## PDHD specifics

* **4 APAs, drift-side groups {0,2} (x<0) and {1,3} (x>0)**.  The
  pre-clustering `name:"img"` Bee hook in `pdhd/clus.jsonnet` dumps the
  stepped img-stage cloud into `mabc-all-apa.zip` as
  `0-clustering-group02.json` / `0-clustering-group13.json` (the analog of
  PDVD's group0123/4567).  `0-clustering-global.json` in the same zip is the
  POST-clustering cloud — not used for the overlay.
* **Only the cathode-facing face of each APA images blobs** in this data:
  APA 0/2 face 0 (W-plane x = −353.20 cm), APA 1/3 face 1 (+353.00 cm).  The
  wall-facing faces (−361.79 / +361.59 cm) produced no blobs.  The
  preprocessing resolves `xorig`/`xsign` per (anode,face) empirically and
  found xsign **+1 for APA 0/2, −1 for APA 1/3**, residuals ≤ 0.32 cm.
* **Wrapped U/V wires**: 1148 wires share 800 channels per face/plane
  (channel↔wire-in-plane is non-monotonic and non-unique).  All geometry is
  per-(anode,face,plane) WIP lists, so nothing in the viewer needed special
  handling; the wire hover shows both the WIP and the (repeating) channel.
  Pitches: U/V 4.926 mm, W 4.792 mm (PDVD: 8.833 / 5.100).
* **Slice geometry is identical to PDVD**: 4-tick slices × 500 ns × 1.6 mm/µs
  = 0.32 cm point spacing, `time_offset = −250 µs` — so the
  `|pts_x − x_start| < 0.16 cm` point↔slice matching carries over unchanged.
* **Zero-charge (ghost) blobs exist here too**: 2972 / 22575 blobs (13.2 %)
  have `val == 0` (PDVD: ~19.5 %) — the deliberately *kept* charge-solver
  zeros (`POTENTIAL_GOOD` time-neighbor rule), distinct from the removed
  ghosts.  ~77 % of them contain stepped points.  Drawn dashed magenta.
* **No center_fallback knob**: PDHD's `clus.jsonnet` runs plain
  `strategy: ["stepped"]`; the toolkit `center_fallback` option is not
  threaded into the PDHD configs (PDVD threads it as
  `PDVD_STEPPED_CENTER_FALLBACK`, default off there too).
* **Magnify naming**: `magnify-run<RUN_PADDED>-evt<EVT>-apa{N}[-dnnroi].root`
  (PDVD says `anode{N}`).  Histograms `h{u,v,w}_<frame><N>`, dead list
  `T_bad<N>`, geometry `T_geo<N>` — same schema as PDVD.

## Validation (run 027409 evt 0)

| check | result |
|---|---|
| Gate 1 points-in-polygon (4000 sampled) | **1.000 PASS** (PDVD reference: 0.983) |
| Gate 2 band channels within Magnify range | **PASS** |
| xconv sign residuals (4 active faces) | ≤ 0.32 cm |
| scripted slice check (15 random charged slices, 3 APAs) | 15/15 have in-polygon points |
| server | HTTP 200 on 5013, clean log; PDVD on 5012 untouched |

Blob counts: APA0 7630, APA1 7847, APA2 2660, APA3 4438 (22575 total);
102363 stepped points (38332 group02 + 64031 group13).

## Gotchas

* The sidecar `event` is the **art event number** (40896), while the work dir
  and scripts use the **event index** (`evt 0`) — same convention as the rest
  of the pdhd scripts.
* The Magnify `gauss` frame in a `-dnnroi` ROOT is the DNN-ROI output, not
  plain Gaussian SP — switch to `raw` for the post-NF waveform.
* Both Bokeh viewers (PDVD 5012, PDHD 5013) share the
  `.direnv/python-3.11.9` bokeh; remember the SSH tunnel uses the matching
  port (`ssh -L 5013:localhost:5013 user@wcgpu1`).

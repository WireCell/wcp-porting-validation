# PDVD imaging event display

An interactive, browser-based event display for inspecting PDVD **imaging**
results: it puts the imaging blobs, their fired wires, the stepped sampling
points, the 3-D point cloud, and the underlying detector waveforms side by side
and keeps them linked, so a single time slice can be followed from a blob's
wires through to the raw signal on those wires.

The tool lives in [`pdvd/img_plot/`](../img_plot) and is served with a Bokeh
server on `wcgpu1`, viewed in a local browser over an SSH tunnel (the same
mechanism as the SP filter-tuning viewer in `pdvd/sp_plot/`).

## Input files (per view) — check these when switching runs/events

All paths are for run 39324, event index 0; substitute your run/event.  The work
directory below is whatever you pass as `--clusters-dir` (default
`/home/xqian/work/scratch_wcgpu1/toolkit-dev/toolkit/pdvd/work/039324_0`).

| view / quantity | input file | what is read |
|---|---|---|
| **wire geometry** (all views) | `wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2` | wire centers, ±half-pitch cells, channels, W-plane x (`xorig`) |
| **blob outlines + fired wires** (view 1) | `<work>/clusters-apa-anode{0..7}-ms-active.tar.gz` | per-blob wire ranges, slice, `corners` polygon (imaging) |
| **stepped sampling points** (orange in view 1, dots in view 2) | `<work>/mabc-all-apa.zip` → `0-clustering-group0123.json` (anodes 0-3), `0-clustering-group4567.json` (anodes 4-7) | the toolkit **stepped** (threshold-0) points, captured **pre-clustering / pre-T0** by MABC's `name:"img"` Bee hook |
| **1-D / 2-D waveforms** (view 3) | `<work>/magnify-run<run>-evt<evt>-anode{0..7}-dnnroi.root` | TH2 `h{u,v,w}_<frame><N>` (x = global channel, y = tick) |
| **dead channels** (view 3 overlay) | same Magnify ROOT | `T_bad<N>` TTree (`chid`, `plane`) — the authoritative dead-channel list |

`preprocess_event.py` consolidates the first three rows into one compact
per-event artifact (`cache/evt{idx}.npz` + `.json` sidecar) so the viewer loads
instantly; the Magnify ROOTs (last two rows) are read lazily on demand.

> **Why the sampling points come from `mabc-all-apa.zip`, not `bee-blobs`.**
> The `wirecell-img bee-blobs` tool only offers `center`/`uniform` sampling.  The
> "stepped" sampler (threshold 0) exists only inside the toolkit clustering
> (`BlobSampler` `strategy:["stepped"]`).  MABC writes the stepped point cloud
> **before** the clustering pipeline runs, under the `name:"img"` Bee hook, into
> the per-drift-side instances confusingly named `0-clustering-group0123/4567.json`
> (the `clustering-global.json` in the same zip is the *post*-clustering cloud).
> PDHD/PDVD have **no T0 correction** (`x == x_t0cor`), so these img-stage points
> are the true imaging-frame stepped positions.

## The three linked views

### 1. 2-D blob view (transverse Z-Y, at one time slice)

For the selected anode/face and time slice it draws, in the transverse plane:

* every **fired wire** as a ±half-pitch cell (filled band + center line), colored
  by plane (U red, V green, W blue);
* the **blob outline**, taken directly from the imaging `corners` polygon, on top;
* the **stepped sampling points** of those blobs, overlaid in orange.

The view auto-zooms to the displayed blob ±20 cm.  **◀ Prev / Next ▶** (or the
`slice idx` spinner) step through slices.  *Hovering* a wire shows its plane, wire
index and channel; *tapping* a wire selects its channel for the waveform views.

### 2. 3-D point projections (X-Y / Z-Y / X-Z, X = drift)

The stepped points shown as three orthogonal projections.  Six X/Y/Z spinners +
**Apply window** restrict the displayed region (points are filtered, not just
highlighted); **Reset window** returns to the full data bounds and releases the
position lock.  When the slice changes in view 1, the points of the displayed
blobs are highlighted in red across all three projections.

### 3. Waveforms (for the tapped channels)

* a **1-D overlay** of ADC-vs-tick for each selected channel (legend
  click-to-hide);
* three **2-D U/V/W-vs-T** images over the selected channels' neighborhood, with
  the current slice's tick window shaded.

A `Magnify frame` selector chooses the stage (`gauss/wiener/raw/orig`, plus
`rawdecon/decon` in the `-R` mode).  For a **DNN-ROI** ROOT the `gauss` tag (the
default) is the DNN-ROI output; switch to `raw` for the post-NF waveform.
Channels can also be added manually (plane + number + **Add**).

**Dead-channel overlay.** Channels listed in the Magnify `T_bad<anode>` TTree that
fall in the displayed channel window are drawn in **grey** with their *actual*
waveform.  A truly dead channel reads flat; a channel labelled dead that still
carries a real pulse stands out — that is the intended check for **mis-labelled
channels**.

### Position lock — one X/Y/Z point drives every view

Type an **X/Y/Z position** (cm, in the MABC point convention — i.e. read straight
off the `mabc-*` Bee display) and a **± pad** (default 20 cm), then **Center
window on pos**.  This:

* **auto-selects the anode, face and slice** the point falls in (the blob whose
  drift-x window contains posX and whose Y-Z polygon contains posY/posZ) — no
  manual anode/face selection needed;
* filters the **3-D projections** to position ± pad;
* zooms the **2-D blob view** to the Y/Z window (X centered on posX via the slice
  jump);
* sets the **1-D waveform** tick axis to the time window posX ± pad maps to;
* sets the **2-D U/V/W-vs-T** panels to the Y-Z-local channels over that tick
  window.

**pos ← current slice** fills the boxes from the displayed blobs; **Reset window**
releases the lock.

## Coordinate frame

Everything is stored and shown in the **MABC point convention (cm)**, the same
frame the stepped points are written in, so points and blob outlines share one
plane.  The drift **x** is computed from the blob slice time with the toolkit
`BlobSampler::time2drift` formula (`clus/src/BlobSampler.cxx`):

```
x = xorig + xsign · (t + time_offset) · drift_speed
```

* `time_offset = −250 µs`, `drift_speed = 1.6 mm/µs`  (from `pdvd/clus.jsonnet`);
* `xorig` = the collection-plane (W) wire-center x of that (anode,face)
  (`±341.55 cm`);
* `xsign` = `anodeface->dirx()` — resolved per (anode,face) by best agreement with
  the points: **+1 for anodes 0-3, −1 for anodes 4-7**;
* geometry y,z are mm/10 (`units.cm == 10`); blob `start`/`span` are in ns;
  tick = `start / 500 ns`.

The per-(anode,face) `{xorig_cm, xsign}` are written to the `.json` sidecar
(`xconv`) so the viewer can invert x → tick for the waveform window.

> No T0 correction is applied for PDHD/PDVD, so this single convention serves both
> the pre-clustering points and the blob outlines.

## Built-in correctness gates

`preprocess_event.py` self-validates and prints a verdict:

* **Gate 1 — points-in-polygon**: stepped points (matched to a slice by drift-x
  window) must fall inside a blob's Y-Z polygon — validates the time2drift frame
  and the `faceid`→(anode,face)+WIP geometry.  On run 39324 evt 0 the inside
  fraction is **0.983** (worst per-face sign residual 1.9 cm).
* **Gate 2 — channel numbering**: every wire-band channel must lie within the
  Magnify ROOT channel range for its anode (skipped if no ROOT template is given).

## Running it

```bash
cd pdvd

# 1. imaging -> clusters-apa-anode{N}-ms-{active,masked}.tar.gz
./run_img_evt.sh 39324 0

# 2. clustering -> mabc-all-apa.zip (carries the stepped img-stage points)
./run_clus_evt.sh 39324 0

# 3. Magnify ROOTs for the waveform view (DNN-ROI SP result)
./run_sp_to_magnify_evt.sh -d 39324 0   # -> work/039324_0/magnify-...-anode{0..7}-dnnroi.root

cd img_plot
./preprocess_event.py                    # build cache/evt0.npz (+ sidecar, runs gates)
./serve_img_viewer.sh 5012               # serve (pick a free port; 5005-5011 often taken)

# from your laptop:
ssh -L 5012:localhost:5012 user@wcgpu1
#   open http://localhost:5012/img_viewer
```

For another event, point `--clusters-dir` (holds the cluster tarballs **and**
`mabc-all-apa.zip`) and `--out` at it, and pass the matching Magnify template as
the 3rd arg to `serve_img_viewer.sh` (keep the `{anode}` placeholder).  See
[`pdvd/img_plot/README.md`](../img_plot/README.md) for the full option list.

## Status

All three views are verified end-to-end on run 39324 evt 0:

* **Gate 1** (points-in-polygon) = **0.983** with the stepped img-stage points in
  the time2drift frame; all 16 (anode,face) sign residuals < 2 cm.
* **Gate 2** (channel numbering) = **PASS** against the per-anode DNN-ROI Magnify
  ROOTs.
* Position lock auto-detects anode/face/slice, tick inversion round-trips exactly,
  and the `T_bad` dead-channel overlay loads (100 dead channels on anode 0).

Note: the waveform view needs the per-anode Magnify ROOTs present; if one is
missing the panels stay empty and a red status line names the file (views 1-2 work
regardless).

## Source files

| file | role |
|---|---|
| `pdvd/img_plot/geom.py` | wire store loader; `faceid`→(anode,face); WIP→(y,z) cm; W-plane x (`xorig`); ±half-pitch band quads; point-in-polygon |
| `pdvd/img_plot/preprocess_event.py` | build the per-event `.npz` + sidecar (stepped points + time2drift x); run Gates 1 & 2 |
| `pdvd/img_plot/img_viewer.py` | the Bokeh app (three linked views) |
| `pdvd/img_plot/serve_img_viewer.sh` | launcher + SSH-tunnel notes |
| `pdvd/img_plot/README.md` | usage reference |

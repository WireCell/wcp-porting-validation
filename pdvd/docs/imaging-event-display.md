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

## Provenance — what the blobs and points actually are

Both inputs were traced through the configs; **the blobs and the sampling points
come from the same data, at the same pipeline stage**, so they are matched at
source and no re-run is needed to align them.

**Blobs** (`clusters-apa-anode{N}-ms-active.tar.gz`, written by the `pdvd/img.jsonnet`
"multi" fork): per-anode **imaging output *after* imaging deghosting** — i.e. option
(b) of {raw imaging / post-imaging-deghosting / post-clustering-deghosting}.  The
pipeline that produces them is:

1. `multi_active_slicing_tiling` — **four** tiling passes merged into one blob set:
   3-view UVW plus the three 2-view combinations UV/VW/UW (third plane masked),
   span 4 ticks (`pdvd/img.jsonnet:164`);
2. "full" uboone solving **with deghosting** (`pdvd/img.jsonnet:202`):
   `BlobClustering → ProjectionDeghosting → {BlobGrouping, ChargeSolving ×2,
   LocalGeomClustering} → InSliceDeghosting` (3 rounds) `→ GlobalGeomClustering`;
3. `ClusterFileSink` dumps the result.

Crucially, **deghosting/solving zeroes ghost blobs but does not remove them from
the file**: 19 % of all blobs in run 39324 evt 0 have `val == 0`.  This is what
made "slice 289" (anode 3 face 1) look wrong — its 3 blobs all genuinely belong
to the same slice id 949 and are **1 real blob (val = 61678) + 2 zero-charge
ghosts**.  The viewer now draws zero-charge blobs as dashed grey outlines (with a
checkbox to hide them), so the real content is one blob, as expected.  Multiple
*charged* blobs in one slice are also legitimate: at slice id 952 the 3 blobs
(val 4845/45510/13935) sit on **adjacent single V wires** (V bounds 206-207 /
207-208 / 208-209) — one isochronous track segment crossing three V wires, not
duplicates.

**Sampling points** (`0-clustering-group0123/4567.json` inside `mabc-all-apa.zip`):
MABC's pre-clustering `name:"img"` hook samples the live grouping built from **the
very same `clusters-apa-anode{N}-ms-active.tar.gz` files**
(`pdvd/wct-clustering.jsonnet:37`) with the stepped (threshold-0) `BlobSampler` —
the same post-imaging-deghosting blobs, **before** any all-APA clustering or
clustering-deghosting.

**Point ↔ slice matching.** The stepped sampler emits *all* of a slice's points at
exactly `x = time2drift(slice start)` — one x value per slice, 0.32 cm apart.  The
viewer therefore matches points to the displayed slice by
`|pts_x − x_start| < 0.16 cm` (half the spacing).  An earlier version used the
window test `x_lo ≤ pts_x ≤ x_hi`, which dropped boundary points on ~1e-14 float
noise — that was the "3 blobs, zero points" mismatch at slice 292 (the points were
there all along; with the fixed rule that slice shows its 14 in-polygon points).
Validation over all 13210 charged slices: every slice gets its points except (a)
slices whose blobs are all zero-charge ghosts and (b) 2.6 % of charged slices (349/13210)
whose (small) blobs the stepped sampler genuinely emitted **no** points for at any
x (e.g. a3f1 slice id 1376) — a sampler-side property, visible in Bee too, not a
viewer mismatch.

## The three linked views

### 1. 2-D blob view (transverse Z-Y, at one time slice)

For the selected anode/face and time slice it draws, in the transverse plane:

* every **fired wire** as a ±half-pitch cell (filled band + center line), colored
  by plane (U red, V green, W blue);
* the **blob outline**, taken directly from the imaging `corners` polygon, on top —
  **solid black** for charged blobs, **dashed grey** for zero-charge ghosts (see
  Provenance above); hovering an outline shows the blob's solved charge, and the
  **hide zero-charge blobs** checkbox removes the ghosts (and their wires/points);
* the **stepped sampling points** of those blobs, overlaid in orange.

The header line gives the slice's x and tick windows and the blob count with the
zero-charge tally; below it a **fired-wire channels** line lists the slice's
channel range per plane (colored U/V/W) so the channels can be read off without
tapping any wire.

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
  click-to-hide); the vertical (ADC) axis **auto-ranges to the signals inside the
  displayed tick window**, so large pulses are never clipped and small ones fill
  the panel;
* three **2-D U/V/W-vs-T** images over the selected channels' neighborhood, with
  the current slice's tick window shaded and every **dead channel in the window
  drawn as a full-height grey vertical bar** (hover gives its number).

A `Magnify frame` selector chooses the stage (`gauss/wiener/raw/orig`, plus
`rawdecon/decon` in the `-R` mode).  For a **DNN-ROI** ROOT the `gauss` tag (the
default) is the DNN-ROI output; switch to `raw` for the post-NF waveform.
Channels can also be added manually (plane + number + **Add**).

**Dead-channel overlay.** Channels listed in the Magnify `T_bad<anode>` TTree that
fall in the displayed channel window are drawn in **grey** with their *actual*
waveform in the 1-D panel, and as **grey vertical bars** in the 2-D ch-vs-T
panels.  A truly dead channel reads flat; a channel labelled dead that still
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
* Point↔slice matching by slice-start x verified on the user-reported cases:
  a3f1 display-slice 289 = 1 charged blob + 2 zero-charge ghosts (3 points in the
  charged blob); display-slice 292 = 3 charged blobs with 14 in-polygon points
  (previously shown as 0 due to the float-edge window bug).

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

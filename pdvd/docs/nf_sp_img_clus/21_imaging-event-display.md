# PDVD imaging event display

An interactive, browser-based event display for inspecting PDVD **imaging**
results: it puts the imaging blobs, their fired wires, the stepped sampling
points, the 3-D point cloud, and the underlying detector waveforms side by side
and keeps them linked, so a single time slice can be followed from a blob's
wires through to the raw signal on those wires.

The tool lives in [`pdvd/img_plot/`](../../img_plot) and is served with a Bokeh
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

**Zero-charge blobs: origin, and why they have sampling points.**  The blobs
that charge solving and deghosting *reject* really are **removed from the graph**
(this matches the familiar Bee experience of point clouds shrinking after each
deghosting stage): `ProjectionDeghosting` deletes its tagged ghosts outright
(`ProjectionDeghosting.cxx:461`), `InSliceDeghosting` rounds 1 and 2 drop every
`TO_BE_REMOVED` blob (`InSliceDeghosting.cxx:791,857`), and round 3 keeps **only**
blobs tagged `POTENTIAL_GOOD` (`:871-874`).  The `val == 0` blobs that remain in
the file (19.5 % in run 39324 evt 0) are a *different*, deliberately-retained
population: the charge-solver LASSO assigned them exactly **zero** charge (and
nothing prunes them — PDVD uses the default blob threshold −1,
`ChargeSolving.h:66`), but `blob_quality_ident` (`InSliceDeghosting.cxx:284-300`)
tags a blob `POTENTIAL_GOOD` either when its own charge > 300 **or when a
front/back time-neighbor blob has charge > 300** — so a solver-zeroed blob
sitting next to a charged blob on the same track survives the final round.
Verified empirically: of 400 randomly sampled zero-charge blobs, 90 % have a
charged (> 300) overlapping blob within ±2 slices.

These kept zero-charge blobs are then **sampled like any other blob** —
`PointTreeBuilding::sample_live` (`PointTreeBuilding.cxx:206-225`) has no charge
cut — so their stepped points appear in the Bee dumps and join the clustering
(at a3f1 display-slice 298 the ghost points carry the same cluster id as the
real track).  That is what "stray" sampling points outside all *black* boxes
are: points of zero-charge blobs.  Decision (2026-06): **keep the sampling as
is** (the round-3 retention is a designed safety net for marginal track
segments) and make the display self-explanatory instead — zero-charge blobs are
drawn as **dashed magenta outlines** (charged blobs stay solid black), and the
hide-zero-charge checkbox removes the ghosts *and* the points contained only in
them.

This is also what made "slice 289" (anode 3 face 1) look wrong — its 3 blobs all
genuinely belong to the same slice id 949 and are **1 real blob (val = 61678) +
2 zero-charge ghosts**.  Multiple *charged* blobs in one slice are also
legitimate: at slice id 952 the 3 blobs (val 4845/45510/13935) sit on **adjacent
single V wires** (V bounds 206-207 / 207-208 / 208-209) — one isochronous track
segment crossing three V wires, not duplicates.

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
**Tiny blobs the stepped sampler leaves point-less — and the `center_fallback`
fix.**  2.6 % of charged slices (349/13210) used to show blobs with **zero**
sampling points (e.g. a3f1 display-slice 298, blobs with val 91 and 23).  Root
cause (`clus/src/BlobSampler.cxx`, "stepped"): for a 1×1×1-wire blob the only
candidate is the crossing of the min- and max-plane *wire centers*, and that
point must fall inside the third plane's strip window ± 0.03 pitch — measured on
the slice-298 blobs, the U×V wire crossings land 1.29 / 0.71 pitches outside the
W wire window, so every candidate is rejected and the blob gets no point.  The
stepped sampler now has a **`center_fallback`** option (default **off**, so
production output is bit-identical — verified byte-for-byte on this event): when
the stepped grid yields nothing, the blob emits one point at its corner-average
center.  A rerun with the toggle on
(`PDVD_STEPPED_CENTER_FALLBACK=true ./run_clus_evt.sh 39324 0`) gives **all
13210 charged slices** points — but the extra points also feed that rerun's
clustering (cluster ids reshuffle), and tiny *ghost* blobs gain centers too.
**Decision (2026-06): the served evt-0 artifacts are generated with the toggle
OFF** (the default; verified byte-identical to the pre-fallback state), so the
2.6 % point-less tiny blobs are accepted as a known display limitation; flip
the env var on only for studies that need every blob represented.

The viewer matches points to a displayed blob by *in-polygon OR within 0.5 cm of
the polygon edge*: a sliver blob's fallback center can land just outside the
drawn outline (the sampler re-derives ray-grid corners that differ slightly from
the file's corners for degenerate slivers; observed up to 0.42 cm).

**Why adjacent blobs are split instead of merged — dead wires.**  Imaging *does*
merge contiguous fired wires into one strip before forming blobs.  But a **dead
channel carries no activity**, so in the tiling passes where its plane is active
the strip breaks in two, and the 2-view pass that **masks** that plane re-covers
the dead region as its own blob.  Worked example, a3f1 display-slice 292 (V
channel **4707 is dead** in `T_bad3`): the 3 charged blobs are V-wire 206-only /
208-only (the split halves, from active-V passes) and 207-only with full U/W
spans (the dead-region recovery blob from the UW pass).  Same at display-slice
301 with dead U channel 3794 (wip 40): U 37-39 / 41-42 split + U 40 recovery
blob.  This is the designed dead-region behaviour of
`multi_active_slicing_tiling`, not a bug; the viewer now fills dead-channel wire
bands **grey** (hover shows `status: DEAD`) so these splits are self-explanatory.

**Why the last two points of a row sit closer together.**  The stepped sampler
walks each strip every `max(3, width/12)` wires from the strip start **and always
adds the last wire** (`BlobSampler.cxx`): a W strip [66,77) samples wires
{66,69,72,75} ∪ {76}, so the final two points are 1 pitch (0.51 cm) apart while
the rest are 3 pitches (1.53 cm) apart.  Intentional — it guarantees the blob
edge is sampled.

## The three linked views

### 1. 2-D blob view (transverse Z-Y, at one time slice)

For the selected anode/face and time slice it draws, in the transverse plane:

* every **fired wire** as a ±half-pitch cell (filled band + center line), colored
  by plane (U red, V green, W blue); a wire shared by several blobs is **drawn
  once** (overlapping fills used to stack alpha and look darker); **dead
  channels** (Magnify `T_bad`) are filled **grey** — hover shows `status: DEAD`;
* the **blob outline**, taken directly from the imaging `corners` polygon, on top —
  **solid black** for charged blobs, **dashed magenta** for zero-charge ghosts (see
  Provenance above; sampling points inside a magenta box belong to a kept
  zero-charge blob, not to any charged blob); hovering an outline shows the blob's
  solved charge, and the **hide zero-charge blobs** checkbox removes the ghosts
  (and their wires/points);
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

* `time_offset = −250 µs`, `drift_speed = 1.57 mm/µs`  (calibrated, was 1.6; from `pdvd/clus.jsonnet`);
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

# 2. clustering -> mabc-all-apa.zip (carries the stepped img-stage points).
#    The served artifacts use the default (center_fallback OFF).  Flip
#    PDVD_STEPPED_CENTER_FALLBACK=true only for studies that need every tiny
#    blob represented (it also feeds that rerun's clustering).
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
[`pdvd/img_plot/README.md`](../../img_plot/README.md) for the full option list.

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
* The served artifacts use `center_fallback` **OFF** (default; byte-identical to
  the pre-fallback zip, verified): 349/13210 charged slices contain tiny blobs
  with no stepped point — a known, accepted limitation (rerun with
  `PDVD_STEPPED_CENTER_FALLBACK=true` when needed).
* Wire dedupe verified at display-slice 292: 62 band rows → 24 unique wires;
  the dead V channel 4707 band is grey-marked (and U 3794 at slice 301).
* Zero-charge blobs draw as **dashed magenta**; scripted check at a3f1
  display-slice 298: 14 points shown with ghosts visible (4 contained only in
  magenta ghost boxes — the previously "stray" yellow points), 10 with
  hide-zero-charge on (ghost-only points filtered with the ghosts).

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

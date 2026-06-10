# PDHD imaging event display

An interactive Bokeh viewer to examine PDHD **imaging** results — blobs, their
fired wires, the sampling points, the 3-D points, and the underlying waveforms —
served on `wcgpu1` and viewed in a local browser over an SSH tunnel.

This is a port of `pdvd/img_plot` (same three views and features); see the
PDVD→PDHD deltas at the end.

## Quick start

```bash
cd pdhd/img_plot

# 1. build the per-event artifact (geometry + blob bounds + Bee points)
./preprocess_event.py                       # -> cache/evt0.npz (+ .json sidecar)

# 2. serve it
./serve_img_viewer.sh 5013                  # default event = cache/evt0.npz

# 3. from your laptop
ssh -L 5013:localhost:5013 user@wcgpu1
#   then open  http://localhost:5013/img_viewer
```

(The PDVD viewer conventionally runs on 5012; PDHD uses **5013**.)

## Input files (per view) — check these when switching runs/events

`<work>` = the dir passed as `--clusters-dir` (default
`/home/xqian/work/scratch_wcgpu1/toolkit-dev/toolkit/pdhd/work/027409_0`).

| view / quantity | input file | read by |
|---|---|---|
| wire geometry (all views) | `wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2` | `geom.py` |
| blob outlines + fired wires (view 1) | `<work>/clusters-apa-apa{0..3}-ms-active.tar.gz` | `wirecell.img.tap.load` |
| stepped sampling points (view 1 orange, view 2 dots) | `<work>/mabc-all-apa.zip` → `0-clustering-group02.json` / `0-clustering-group13.json` | `preprocess_event.py` (zip + JSON) |
| 1-D / 2-D waveforms (view 3) | `<work>/magnify-run<run>-evt<evt>-apa{0..3}-dnnroi.root` (TH2 `h{u,v,w}_<frame><N>`) | `img_viewer.py` (uproot, lazy) |
| dead channels (view 3 overlay) | same Magnify ROOT, `T_bad<N>` TTree | `img_viewer.py` (uproot, lazy) |

Use the **`-ms-active`** cluster files (real 3-view U∩V∩W blobs), not `-ms-masked`.
The sampling points are the toolkit **stepped** (threshold-0) cloud captured by
MABC *before* clustering (`name:"img"` hook in `pdhd/clus.jsonnet`, grouped by
drift side: group02 = APAs 0+2, group13 = APAs 1+3) — `bee-blobs` only does
`center`/`uniform`, so it is **not** used.  PDHD has no T0, so these img-stage
points are the true imaging-frame positions.  (The `clustering-global.json` in
the same zip is the *post*-clustering cloud — don't use it for the overlay.)

### Provenance (what stage the blobs/points are)

* **Blobs** = per-APA imaging **after imaging deghosting**: merged tiling
  passes (3-view + 2-view) → uboone-style solving with ProjectionDeghosting +
  InSliceDeghosting ×3 + GlobalGeomClustering → `ClusterFileSink`
  (`pdhd/img.jsonnet`).  Rejected ghosts are genuinely **removed** from the
  file; blobs with `val == 0` are a different, deliberately *kept* population
  (charge-solver zeros whose time-neighbors are charged — `POTENTIAL_GOOD` in
  InSliceDeghosting round 3).  They are sampled like any blob, so their points
  appear in Bee and join clustering.  The viewer draws them **dashed magenta**,
  with a checkbox to hide them (which also hides their points).
* **Points** = stepped sampling of **the same tar files**, read by MABC
  pre-clustering (`pdhd/wct-clustering.jsonnet`) — so blobs and points are
  matched at source.
* A slice's points all sit at exactly `x = time2drift(slice start)` (one x per
  slice, 0.32 cm apart — 4 ticks × 500 ns × 1.6 mm/µs, same as PDVD); the
  viewer matches points to the displayed slice by
  `|pts_x − blob_x_start| < 0.16 cm`, then keeps points in-polygon **or within
  0.5 cm of a displayed blob's edge** (sliver-blob tolerance).
* Tiny (≈1-wire) blobs can get **no** stepped points (the single wire-crossing
  candidate falls outside the third plane's strip window).  The toolkit
  `BlobSampler` has a `center_fallback` option (default off), but unlike PDVD
  it is **not threaded** into the PDHD configs — PDHD always runs plain
  `stepped`.
* **Adjacent blobs split at dead wires by design**: a dead channel breaks the
  fired-wire strip in the active-plane tiling passes, and a masked-plane pass
  re-covers the dead region as its own blob.  Dead-channel wire bands are drawn
  grey in the blob view to make this visible.

See `pdhd/docs/imaging-event-display.md` for the full notes, and
`pdvd/docs/imaging-event-display.md` § Provenance for the original trace (the
zero-charge / deghosting story is identical).

## The three views

1. **2D blob view** (transverse Z–Y, cm) at one time slice. Each fired U/V/W wire
   is drawn as a ±half-pitch cell (U red, V green, W blue) with its center line —
   a wire shared by several blobs is drawn once, and **dead channels are filled
   grey** (hover shows `status: DEAD`); the
   blob outline (from the imaging `corners`) sits on top — **solid black** for
   charged blobs, **dashed magenta** for zero-charge ghosts (hover shows the
   charge; the **hide zero-charge blobs** checkbox removes them and their
   points); the slice's stepped
   sampling points are overlaid in orange. The header reports the blob count with
   the zero-charge tally, and a **fired-wire channels** line gives the slice's
   per-plane channel ranges directly (no tapping needed). The view **auto-zooms
   to the displayed blob ±20 cm**. **◀ Prev / Next ▶** (or the `slice idx`
   spinner) step the slice.
   *Hover* a wire for its plane/**wire index/channel**; *tap* a wire to select its
   channel for the waveform views (wire→channel conversion is automatic — note
   PDHD wrapped wires: several wire segments share one channel, the hover shows
   the per-face wire index AND the channel).
2. **3-D point projections** X-Y / Z-Y / X-Z (X = drift). The six X/Y/Z spinners +
   **Apply window** restrict the shown region (points are filtered, not just
   highlighted); **Reset window** returns to the data bounds and releases the
   position lock. The current slice's points are highlighted in red.
3. **Waveforms** for the tapped channels: 1-D ADC-vs-tick overlay (legend
   click-to-hide; the **ADC axis auto-ranges to the signals in the displayed tick
   window**) plus 2-D U/V/W-vs-T images, with the current slice's tick window
   shaded. Pick the Magnify frame (`gauss/wiener/raw/orig`, default `gauss`; plus
   `rawdecon/decon` in `-R` mode). Add a channel manually (plane + number + **Add**).
   **Dead channels** from the Magnify `T_bad<anode>` TTree that fall in the shown
   channel window are drawn in **grey with their actual waveform** in the 1-D
   panel and as **grey vertical bars** in the 2-D panels (hover gives the
   channel) — a "dead" channel that still shows a real pulse flags a **mis-label**.

### Position lock — drive every view from one X/Y/Z point

Type an **X/Y/Z position** (cm, as read off the `mabc-*` Bee display) and a **± pad**
(default 20), then click **Center window on pos** (or **pos ← current slice**). This:

* **auto-selects the anode/face/slice** the point falls in (no manual selection);
* filters the **3-D projections** to position ± pad;
* zooms the **2-D blob view** to the Y/Z window (X centered on posX via the slice jump);
* sets the **1-D waveform** tick axis to the time window posX ± pad maps to;
* sets the **2-D U/V/W-vs-T** panels to the Y-Z-local wires over that tick window.

**Reset window** releases the lock.

## The preprocessing step

`preprocess_event.py` consolidates the static sources (geometry + blob bounds +
stepped points) into one `.npz` so the viewer starts instantly; waveforms and dead
channels stay lazy-loaded from the Magnify ROOT at display time.

### Coordinate frame (important)

Everything is stored and displayed in the **MABC point convention (cm)**, the same
frame the stepped points live in, so points and blob outlines share one plane. The
drift **x** is from the blob slice time via the toolkit `BlobSampler::time2drift`:

```
x = xorig + xsign · (t + time_offset) · drift_speed
```

* `time_offset = −250 µs`, `drift_speed = 1.6 mm/µs` (`pdhd/clus.jsonnet`);
* `xorig` = W-plane wire-center x of that (anode,face): the cathode-facing
  (active) faces are at **−353.20 cm** (APAs 0/2, face 0) and **+353.00 cm**
  (APAs 1/3, face 1); the wall-facing faces at −361.79 / +361.59 cm;
* `xsign` = `anodeface->dirx()`, resolved empirically per (anode,face):
  **+1 for APAs 0/2, −1 for 1/3**;
* geometry y,z = mm/10 (`units.cm == 10`); blob `start`/`span` in ns; tick = `start/500`.

The per-(anode,face) `{xorig_cm, xsign}` go to the `.json` sidecar (`xconv`); the
viewer inverts x → tick from them. No T0 is applied for PDHD.

### Built-in correctness gates

`preprocess_event.py` self-checks and prints a verdict:

* **Gate 1 — points-in-polygon**: stepped points (matched to a slice by drift-x
  window) must fall inside a blob's Y-Z polygon. This validates the time2drift frame
  and the `faceid`→(anode,face) + WIP→position geometry.
* **Gate 2 — channel numbering**: every wire-band channel must lie within the
  Magnify ROOT channel range for its APA. Pass a template to enable it:
  ```bash
  ./preprocess_event.py --magnify-template \
      /.../027409_0/magnify-run027409-evt0-apa{anode}-dnnroi.root
  ```
  It is **skipped** (with a warning) when the ROOTs are absent.

## Waveform view requires Magnify ROOTs

The 1-D / 2-D waveform panels read per-APA Magnify ROOTs
(`h{u,v,w}_<frame><anode>`, x = global channel, y = tick). If the ROOT for the
current APA is missing, the panels stay empty and a red status line names the
missing file — views 1 and 2 work regardless.

Generate them with `run_sp_to_magnify_evt.sh`. For the **DNN-ROI** SP result
(the `gauss` tag carries the DNN-ROI output), use `-d on`:

```bash
cd pdhd
./run_sp_to_magnify_evt.sh -d on 027409 0  # -> work/027409_0/magnify-run027409-evt0-apa{0..3}-dnnroi.root
```

`serve_img_viewer.sh` **defaults** to that DNN-ROI template; override it with the
3rd argument for a different ROOT set (use the `{anode}` placeholder):

```bash
./serve_img_viewer.sh 5013 cache/evt0.npz '/.../magnify-run027409-evt0-apa{anode}.root'
```

In the viewer the `Magnify frame` selector defaults to `gauss`; for a DNN-ROI ROOT
that is the DNN-ROI output. Switch to `raw` to see the post-NF waveform.

## Files

| file | role |
|---|---|
| `geom.py` | load wire store; `faceid`→(anode,face); WIP→(y,z) cm; ±half-pitch band quads; point-in-polygon |
| `preprocess_event.py` | build `cache/evt{idx}.npz` + `.json` sidecar; run Gates 1 & 2 |
| `img_viewer.py` | the Bokeh app (three linked views) |
| `serve_img_viewer.sh` | launcher (port, npz, magnify template); SSH-tunnel notes |
| `cache/` | generated artifacts (not committed) |

## Other events

`--clusters-dir` must hold both the cluster tarballs **and** `mabc-all-apa.zip`
(override the zip with `--mabc-zip` if elsewhere):

```bash
./preprocess_event.py \
  --clusters-dir /.../work/<RUN>_<evt> \
  --out cache/evt<idx>.npz
./serve_img_viewer.sh 5013 cache/evt<idx>.npz \
  '/.../work/<RUN>_<evt>/magnify-run<RUN>-evt<evt>-apa{anode}-dnnroi.root'
```

## PDVD→PDHD deltas (for maintainers)

| item | PDVD | PDHD |
|---|---|---|
| anodes / drift groups | 8; {0–3} vs {4–7} | 4; **{0,2} vs {1,3}** |
| group point files | `0-clustering-group0123/4567.json` | `0-clustering-group02/group13.json` |
| cluster tars | `clusters-apa-anode{N}-…` | `clusters-apa-apa{N}-…` |
| magnify files | `…-anode{N}[-dnnroi].root` | `…-apa{N}[-dnnroi].root` |
| wires | protodunevd v5 | protodunehd v1 (**wrapped U/V**: 1148 wires / 800 channels per face) |
| W-plane x (active faces) | ±341.55 cm | −353.20 / +353.00 cm |
| pitches U/V, W | 8.833, 5.100 mm | 4.926, 4.792 mm |
| port | 5012 | 5013 |
| center_fallback knob | threaded (`PDVD_STEPPED_CENTER_FALLBACK`) | not threaded (plain `stepped`) |

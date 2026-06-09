# PDVD imaging event display

An interactive, browser-based event display for inspecting PDVD **imaging**
results: it puts the imaging blobs, their fired wires, the sampling points, the
3-D point cloud, and the underlying detector waveforms side by side and keeps
them linked, so a single time slice can be followed from a blob's wires through
to the raw signal on those wires.

The tool lives in [`pdvd/img_plot/`](../img_plot) and is served with a Bokeh
server on `wcgpu1`, viewed in a local browser over an SSH tunnel (the same
mechanism as the SP filter-tuning viewer in `pdvd/sp_plot/`).

## Why

The information needed to understand an imaging result is scattered across four
products that were never viewable together:

* the **wire geometry** (where each wire sits, its pitch cell, its channel),
* the **imaging cluster files** (which wires each blob fired, at which slice),
* the **Bee 3-D points** (the sampled charge cloud),
* the **Magnify waveforms** (the actual signal on each channel).

This display consolidates the first three into one compact per-event artifact
(so it loads instantly) and lazily reads the waveforms on demand.

## The three linked views

### 1. 2-D blob view (transverse Z–Y, at one time slice)

For the selected anode/face and time slice it draws, in the transverse plane:

* every **fired wire** as a ±half-pitch cell — a filled band bounded by the two
  wire boundaries, with the **wire center line** through it — colored by plane
  (U red, V green, W blue);
* the **blob outline**, taken directly from the imaging `corners` polygon, on top;
* the **sampling points** of those blobs, overlaid in orange.

**◀ Prev / Next ▶** (or the `slice idx` spinner) step through the slices.
*Hovering* a wire shows its plane and channel; *tapping* a wire selects that
channel for the waveform views below — this is how you go from "this wire fired"
to "show me its signal."

### 2. 3-D point projections (X-Y / Z-Y / X-Z, X = drift)

The Bee 3-D points shown as three orthogonal projections. Six X/Y/Z spinners plus
**Apply window** restrict the displayed region; **Reset window** returns to the
full data bounds. When the time slice changes in view 1, the points belonging to
the displayed blobs are **highlighted in red** across all three projections, so
the slice you are inspecting is located within the whole event.

### 3. Waveforms (for the tapped channels)

* a **1-D overlay** of ADC-vs-tick for each selected channel (legend
  click-to-hide),
* three **2-D U/V/W-vs-T** images over the selected channels' neighborhood, with
  the current slice's tick window shaded.

A `Magnify frame` selector chooses which deconvolution stage to show
(`gauss/wiener/raw/orig`, plus `rawdecon/decon` when the ROOT was produced in the
`-R` special mode). Channels can also be added manually (plane + number + **Add**).

## Data flow

```
protodunevd-wires-larsoft-v5.json.bz2 ─┐
clusters-apa-anode{N}-ms-active.tar.gz ─┼─►  preprocess_event.py  ─►  cache/evt{idx}.npz (+ .json)
{idx}-imaging-group{0123,4567}.json   ─┘                                      │
                                                                             ▼
magnify-run…-anode{N}.root  ───────────────────────────────►  img_viewer.py  (Bokeh server)
        (read lazily at display time)                                ssh -L  →  browser
```

`preprocess_event.py` reads the blob per-plane wire ranges, slice, charge and
`corners` polygon from the **`-ms-active`** cluster files (real 3-view U∩V∩W
blobs), turns each fired wire into a ±half-pitch cell via the wire geometry, and
bundles them with the Bee points into one `.npz`. Waveforms are *not* baked in —
they stay in the Magnify ROOTs and are read per channel on demand.

## Coordinate frame

Everything is stored and shown in the **Bee frame (cm)**, reproducing exactly what
`wirecell-img bee-blobs` wrote into the Bee JSON, so blob wires (from geometry)
and Bee points (read as-is) share one plane:

* geometry / `corners` are in mm → ÷10 for (y, z) cm (`units.cm == 10`);
* the drift **x** is undrifted from the blob's slice time with the per-drift-side
  constants from `build_v4_bee_evt0to4.sh`:
  * anodes 0–3 (bottom CRP): `speed = −1.56 mm/µs`, `x0 = −341.5 cm`, `t0 = 0`,
  * anodes 4–7 (top CRP): `speed = +1.56 mm/µs`, `x0 = +341.5 cm`, `t0 = 0`;
* blob `start`/`span` are in ns; tick = `start / 500 ns`.

## Built-in correctness gates

`preprocess_event.py` self-validates and prints a verdict:

* **Gate 1 — points-in-polygon**: Bee points (matched to a slice by drift-x
  window) must fall inside a blob's Y-Z polygon. This is the single best check
  that the mm→cm/undrift transform and the `faceid`→(anode,face)+WIP→position
  geometry are right. On run 39324 evt 0 the inside fraction is **0.999**.
* **Gate 2 — channel numbering**: every wire-band channel must lie within the
  Magnify ROOT channel range for its anode (skipped if no ROOT template is given).

The same point-in-polygon membership test drives which sampling points are shown
for a slice, so the blob view and the projection highlight always agree.

## Running it

```bash
cd pdvd/img_plot
./preprocess_event.py                 # build cache/evt0.npz (+ sidecar, runs gates)
./serve_img_viewer.sh 5012            # serve (pick a free port; 5005–5011 are often taken)

# from your laptop:
ssh -L 5012:localhost:5012 user@wcgpu1
#   open http://localhost:5012/img_viewer
```

For another event, point `--clusters-dir` / `--bee-dir` / `--bee-idx` at it and
pass the matching Magnify template (with `{anode}`) as the 3rd argument to the
serve script. See [`pdvd/img_plot/README.md`](../img_plot/README.md) for the full
option list.

## Status / limitations

* Views 1 and 2 are verified end-to-end (Gate 1 at 0.999; geometry, slice
  stepping, projections, window filter and tap→channel all exercised).
* View 3 (waveforms) is wired and degrades gracefully, but its rendering is
  **unexercised** until a Magnify ROOT is generated for the event (none exists for
  run 39324 evt 0 yet). Produce one with `run_sp_to_magnify_evt.sh` and pass its
  path template to enable the waveform panels and Gate 2.
* Per-slice sampling points are matched to blobs by drift-x window + polygon
  containment; a negligible fraction of edge points (~0.1 %, the Gate-1 residual)
  may not be attributed to a blob.

## Source files

| file | role |
|---|---|
| `pdvd/img_plot/geom.py` | wire store loader; `faceid`→(anode,face); WIP→(y,z) cm; ±half-pitch band quads; point-in-polygon |
| `pdvd/img_plot/preprocess_event.py` | build the per-event `.npz` + sidecar; run Gates 1 & 2 |
| `pdvd/img_plot/img_viewer.py` | the Bokeh app (three linked views) |
| `pdvd/img_plot/serve_img_viewer.sh` | launcher + SSH-tunnel notes |
| `pdvd/img_plot/README.md` | usage reference |

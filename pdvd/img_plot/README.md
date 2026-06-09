# PDVD imaging event display

An interactive Bokeh viewer to examine PDVD **imaging** results — blobs, their
fired wires, the sampling points, the 3-D points, and the underlying waveforms —
served on `wcgpu1` and viewed in a local browser over an SSH tunnel.

## Quick start

```bash
cd pdvd/img_plot

# 1. build the per-event artifact (geometry + blob bounds + Bee points)
./preprocess_event.py                       # -> cache/evt0.npz (+ .json sidecar)

# 2. serve it
./serve_img_viewer.sh 5012                  # default event = cache/evt0.npz

# 3. from your laptop
ssh -L 5012:localhost:5012 user@wcgpu1
#   then open  http://localhost:5012/img_viewer
```

Pick a free port (5005–5011 are often taken by other viewers; 5012+ is usually free).

## The three views

1. **2D blob view** (transverse Z–Y, cm) at one time slice. Each fired U/V/W wire
   is drawn as a ±half-pitch cell (U red, V green, W blue); the blob outline (from
   the imaging `corners`) sits on top; the slice's Bee sampling points are overlaid
   in orange. **◀ Prev / Next ▶** (or the `slice idx` spinner) step the slice.
   *Hover* a wire to see its plane+channel; *tap* a wire to select that channel for
   the waveform views.
2. **3-D point projections** X-Y / Z-Y / X-Z (X = drift). The six X/Y/Z spinners +
   **Apply window** restrict the shown region; **Reset window** returns to the data
   bounds. The current slice's points are highlighted in red (drift-x window).
3. **Waveforms** for the tapped channels: 1-D ADC-vs-tick overlay (legend
   click-to-hide) plus 2-D U/V/W-vs-T images over the selected channels'
   neighborhood, with the current slice's tick window shaded. Pick the Magnify
   frame (`gauss/wiener/raw/orig`, default `gauss`; plus `rawdecon/decon` when the
   ROOT was produced in the `-R` special mode). You can also add a channel
   manually (plane + number + **Add**).

## Inputs and the preprocessing step

`preprocess_event.py` consolidates three static sources into one `.npz` so the
viewer starts instantly (waveforms stay lazy-loaded from Magnify ROOT at display
time):

| source | what it provides | how it is read |
|---|---|---|
| `protodunevd-wires-larsoft-v5.json.bz2` | wire centers, ±half-pitch cells, channels | `wirecell.util.wires.persist` (see `geom.py`) |
| `clusters-apa-anode{N}-ms-active.tar.gz` | blob per-plane wire ranges, slice, charge, `corners` polygon | `wirecell.img.tap.load` |
| `{idx}-imaging-group{0123,4567}.json` | Bee 3-D sampling points (as-is) | plain JSON |

Use the **`-ms-active`** cluster files (real 3-view U∩V∩W blobs), not `-ms-masked`.

### Coordinate frame (important)

Everything is stored and displayed in the **Bee frame (cm)**, reproducing exactly
what `wirecell-img bee-blobs` wrote into the Bee JSON:

* geometry / `corners` are in mm → divide by 10 for (y, z) cm (`units.cm == 10`);
* the drift **x** is undrifted from the blob's slice time with the per-drift-side
  constants from `build_v4_bee_evt0to4.sh`:
  * anodes 0–3 (bottom CRP): `speed = −1.56 mm/µs`, `x0 = −341.5 cm`, `t0 = 0`
  * anodes 4–7 (top CRP): `speed = +1.56 mm/µs`, `x0 = +341.5 cm`, `t0 = 0`
* blob `start`/`span` are in ns; tick = `start / 500 ns`.

This is why the blob bands (from wire geometry) and the Bee sampling points share
the same plane and can be overlaid without re-sampling.

### Built-in correctness gates

`preprocess_event.py` self-checks and prints a verdict:

* **Gate 1 — points-in-polygon**: Bee points (matched to a slice by drift-x window)
  must fall inside a blob's Y-Z polygon. This validates the mm→cm/undrift transform
  and the `faceid`→(anode,face) + WIP→position geometry. On run 39324 evt 0 the
  inside fraction is **0.999**.
* **Gate 2 — channel numbering**: every wire-band channel must lie within the
  Magnify ROOT channel range for its anode. Pass a template to enable it:
  ```bash
  ./preprocess_event.py --magnify-template \
      /.../039324_0/magnify-run039324-evt0-anode{anode}.root
  ```
  It is **skipped** (with a warning) when the ROOTs are absent.

## Waveform view requires Magnify ROOTs

The 1-D / 2-D waveform panels read per-anode Magnify ROOTs
(`h{u,v,w}_<frame><anode>`, x = global channel, y = tick). If the ROOT for the
current anode is missing, the panels stay empty and a red status line names the
missing file — views 1 and 2 work regardless. Generate the ROOTs with the
Magnify-PDVD / `run_sp_to_magnify_evt.sh` pipeline, then pass the template as the
3rd argument to `serve_img_viewer.sh`.

## Files

| file | role |
|---|---|
| `geom.py` | load wire store; `faceid`→(anode,face); WIP→(y,z) cm; ±half-pitch band quads; point-in-polygon |
| `preprocess_event.py` | build `cache/evt{idx}.npz` + `.json` sidecar; run Gates 1 & 2 |
| `img_viewer.py` | the Bokeh app (three linked views) |
| `serve_img_viewer.sh` | launcher (port, npz, magnify template); SSH-tunnel notes |
| `cache/` | generated artifacts (not committed) |

## Other events

```bash
./preprocess_event.py \
  --clusters-dir /.../work/<RUN>_<evt> \
  --bee-dir /.../pdvd/data/<idx> --bee-idx <idx> \
  --out cache/evt<idx>.npz
./serve_img_viewer.sh 5012 cache/evt<idx>.npz '<magnify-template-with-{anode}>'
```

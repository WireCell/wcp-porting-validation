# PDHD light event display (`pd_plot`)

Interactive Bokeh viewer for the PDHD photon-detector (flash) reconstruction,
the light-side counterpart of `pdhd/img_plot`. It reads the **WCT-native**
optical products written per event by the `flash/` chain and lets you browse
flashes, their OpHits, the light geometry, and per-channel waveforms.

## What it shows

Four linked panels (top to bottom):

1. **Flashes** — a table of every flash in the event (flash id, time [µs], total
   PE, nhits, y/z centroid [cm]) plus a time-vs-total-PE scatter. Select a table
   row, or tap the scatter, to pick a flash; this drives panels 2 and 3.
2. **Light geometry** — the 160 OpDet positions in three projections (Z-Y, X-Y,
   X-Z) with the PDHD TPC boxes drawn. For the selected flash the lit OpDets are
   coloured (Viridis) and sized (∝√PE) by their PE; the flash y/z centroid is
   marked with a red ✕ in the Z-Y view. Hover an OpDet for its index and PE.
3. **OpHits** — a 1-D bar histogram of PE per OpChannel for the selected flash's
   OpHits, plus a table (channel, peak [µs], PE, amplitude, width). Tap a bar or
   a table row to pick one OpHit; this drives panel 4.
4. **Waveform** — the selected OpHit channel's **raw ADC** (left axis, grey) and
   **deconvolved** (right axis, red) waveform snippet overlaid, with the hit
   peak (dashed) and start (faint) times marked. The deconvolved peak sits at the
   hit time inside the snippet.

## Data it reads

Per event directory `pdhd/work/<RUN6>_<EVT>/`:

- `opflash_pdhd-wct.tar.gz` — opflash tensor set (`flash/docs/design.md` §3.4):
  - tensor 0 `[nflash, 161]`: col 0 = flash time (ns, trigger-relative), cols
    1..160 = PE per OpDet;
  - tensor 1 `[nflash, 8]`: flash_id, total_pe, y/z center [mm], y/z width [mm],
    abs_dts, nhits;
  - tensor 2 `[nhit, 9]`: channel, peak_ns, width_ns, area, amplitude, pe,
    start_ns, flash_id, fast_to_total.
- `light-frames-wct.tar.bz2` — `frame_raw`/`frame_decon` (dense per-channel
  waveforms, zeros between the snippets), `channels_*`, `tickinfo_*`
  (`[frame_time_ns, tick_ns, tbin0]`).
- `cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json` — OpDet positions [mm],
  converted to cm for the geometry panel.

The event dropdown is built by scanning the work directory for `<RUN6>_<EVT>`
subdirectories that contain `opflash_pdhd-wct.tar.gz`, so any reconstructed event
appears automatically.

Only the WCT-native reconstruction is shown (not the LArSoft reference), and the
waveform panel overlays raw vs decon for one channel.

## Running

On the compute node (`wcgpu1`):

```bash
cd pdhd/pd_plot
./serve_pd_viewer.sh                 # port 5014, workdir pdhd/work
# ./serve_pd_viewer.sh 5014 <workdir> <opdet-geom.json>   # all optional
```

From a remote laptop, forward the port then open the app:

```bash
ssh -L 5014:localhost:5014 user@wcgpu1
# then browse to:
http://localhost:5014/pd_viewer
```

(Port 5014 by default; `img_plot` uses 5013, `pdvd` 5012.)

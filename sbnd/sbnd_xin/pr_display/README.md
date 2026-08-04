# SBND pattern-recognition (PR) event display

A Bokeh event display for validating and improving the PR code: what the
neutrino-PR chain reconstructed, in three 3-D projections plus the six
Magnify-style 2-D views, all from one self-contained JSON per event.

Full write-up, including two defects found while building it:
[`../docs/pr/26_pr-event-display.md`](../docs/pr/26_pr-event-display.md).

## 1. Produce the calib dump

The viewer reads the per-event JSON written by the PR chain's `pr_display`
stage (`PrDisplayDump`, default OFF -- it only exists when named):

```bash
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out_root> data <evt ...>
# -> <out_root>/pr_evt<ID>/calib-pr-evt<ID>.json
```

`PR_EXTRA_STAGES` is empty by default, so the driver's normal behaviour and
every existing output are unchanged. The stage is read-only: an arm run with
it hashes identically to one run without (doc pr/26 sec 6).

Example, the event this display was built on:

```bash
PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-prdisp-388 data 388
```

## 2. Serve

```bash
./pr_display/serve_pr_display.sh 5017 work-prdisp-388/pr_evt*/calib-pr-evt*.json
```

From a laptop:

```bash
ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
# then open http://localhost:5017/pr_display_viewer
```

Port 5017 is the next free one after img 5013 / pd 5014 / ql_scan 5015 /
wf_scan 5016.

## Display layout

**Row 1 -- three charge projections** X-Y, Y-Z, X-Z, with the SBND active
volume and the cathode plane drawn in red.

**Row 2 -- six panels, two columns**: TPC 0 | TPC 1 x (T vs U, T vs V, T vs W).
Each shows the fitted 2-D charge as a heat map (colour = measured charge, 0 to
the 99th percentile -- a handful of saturated cells would otherwise flatten
every track) with the best-fit trajectory drawn over it in the segment's
colour, and the dead-channel bands shaded. This is the Magnify-tracking view
of the neutrino interaction.

## Layers (each a toggle)

| layer | what | source in the JSON |
|---|---|---|
| **track fit** | PR-graph segments as polylines, one colour per segment | `segments[].points[]` |
| **shower pts** | associated 3-D points flagged shower | `track_shower` where `flag_shower` |
| **track pts** | associated 3-D points flagged track | `track_shower` where not |
| **steiner** | the Steiner skeleton of every cluster | `steiner[].x/y/z` |
| **terminals** | only the terminal subset of that skeleton | `steiner[].flag_terminal` |
| **vertices** | PR-graph vertices; the neutrino vertex is a large star | `vertices`, `main_vertex` |
| **dead (2-D)** | dead-channel bands, 2-D panels only | `dead` |

`steiner` is off by default -- 6 k points per event drawn under everything else
is noise until you go looking for it. `terminals` is a separate toggle so the
subset can be seen without the skeleton.

## Zoom and centring

`zoom` reframes **all nine panels** to ±*half-width* around a centre; the
half-width box defaults to 30 cm. The centre starts at the identified neutrino
vertex, and `centre on vertex` returns to it. Type any (x, y, z) in cm to look
somewhere else.

The 2-D panels have no wire geometry to project through (deliberately -- the
viewer loads nothing but the JSON), so their window is derived from the fitted
points inside the same 3-D sphere. If no fitted point is near the centre a
panel falls back to its full extent rather than showing an empty box.

## Conventions

Everything follows doc pr/7:

- positions in **cm**;
- `pu/pv/pw` are **fractional per-APA wire indices** -- integer = wire *centre*,
  not a wire edge. They are kept fractional; truncating them puts the drawn
  track a systematic half channel off the charge;
- time is a **slice** index, `pt / nticks_per_slice` (4 on SBND). The dump
  carries dead regions in both ticks (`t0/t1`) and slices (`s0/s1`); the viewer
  uses slices, which is what the cells are keyed on;
- charges are **raw** (`dQ` in electrons). Unlike the Bee layers, the affine
  `dQdx_scale`/`dQdx_offset` transform is *not* pre-applied; both constants are
  recorded in `meta` so the Bee colouring can be reproduced.

Alignment is checked numerically, not by eye: the charge-weighted perpendicular
offset of measured cells from the drawn polyline is ≤ 0.07 index units in all
six panels of evt 18255/388 (doc pr/26 sec 4).

## Known caveat

`proj[].charge_pred` -- the *predicted* per-cell charge -- is **not reproducible
run to run** on cells claimed by more than one cluster (6-10 % of cells). The
cause is upstream, in `TrackFitting::assemble_fitted_charge_2d`, and is not
introduced by this display; everything else in the dump is byte-identical
across runs. See doc pr/26 sec 5.2. Do not read the per-cell
measured-vs-predicted comparison as a stable number until that is fixed.

## Not in stage 1

Hand-scan label saving, batch pre-rendering, and any change to the PR
algorithms themselves. This stage is read-only viewing plus the dump that
feeds it.

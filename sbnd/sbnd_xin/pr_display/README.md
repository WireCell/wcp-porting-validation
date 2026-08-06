# SBND pattern-recognition (PR) event display

A Bokeh event display for validating and improving the PR code: what the
neutrino-PR chain reconstructed, in three 3-D projections, a **dQ/dx panel**
for checking PID by eye, and (behind `--wire-planes`, hidden by default) the
six Magnify-style 2-D views, with the reconstructed **particle flow** and the
**event features** that decide selection.

Full write-up, including two defects found while building it:
[`../docs/pr/26_pr-event-display.md`](../docs/pr/26_pr-event-display.md). The
dQ/dx panel: [`../docs/pr/42_dqdx-panel.md`](../docs/pr/42_dqdx-panel.md).

Inputs, per event: `calib-pr-evt<ID>.json` (everything drawn) and, beside it,
`mabc-pr.zip` — read **only** for its `data/*/*-mc.json` member, the
particle-flow tree. That tree is the canonical one (`fill_bee_pf_tree` writes
it, Bee shows it), so the display reads it rather than deriving a second answer
that could disagree. No zip ⇒ the particle-flow panel is empty and everything
else still works.

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
./pr_display/serve_pr_display.sh 5017 work-prdisp-cosscan2/pr_evt388/calib-pr-evt388.json
```

Pass the path **explicitly** when more than one `work-prdisp-*` arm exists: the
script's default glob is `../work-prdisp-*/pr_evt*/calib-pr-evt*.json`, and
every arm holding the same event resolves to the same label `evt388`, so all but
one are silently shadowed. Four arms hold evt 388 today.

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

**Row 2 -- particle flow and event features.** See the two sections below.

**Row 3 -- dQ/dx panel.** See the section below; full write-up doc pr/42.

**Row 4, hidden by default (`--wire-planes` to show) -- six panels, two
columns**: TPC 0 | TPC 1 x (T vs U, T vs V, T vs W). Each shows the fitted
2-D charge as a heat map (colour = measured charge, 0 to the 99th percentile
-- a handful of saturated cells would otherwise flatten every track) with the
best-fit trajectory drawn over it in the segment's colour, and the
dead-channel bands shaded. This is the Magnify-tracking view of the neutrino
interaction. Construction and data-filling are unchanged whether the row is
shown or not -- `--wire-planes` only decides whether it's part of the served
layout.

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

## Particle flow

One row per node of the Bee particle-flow tree, indented by depth. **Click a row
and that particle is traced in amber in all nine panels** -- the three
projections and each of the six 2-D views.

| kind | the row's id resolves to | what a click highlights |
|---|---|---|
| **shower** | a `shower_id` group in `segments[]` | every segment of that shower |
| **track** | one segment id | that segment |
| **gamma** | nothing (a pseudo-node the PF builder inserts before an indirectly-connected shower; its id comes from that builder's own counter) | its children's segments, recursively |

The join is exact, not heuristic: both producers encode a node/segment as
`cluster_id*1000 + segment index`, and `PrDisplayDump` puts the shower's node id
on **every** segment of that shower as `shower_id`. Without that field a shower
click would highlight only its start segment -- the 789 MeV shower in evt
18255/388 spans **29**.

`nseg` is what the click actually lights up, so a gamma pseudo-node reports its
children's count rather than 0. A shower with `start_connection_type == 4` is
dropped by the PF builder, so it has a row in `showers[]` and no PF node.

`clear highlight` removes the trace.

## dQ/dx panel

Click a particle-flow row (above) and its measured dQ/dx (`points[].dQ /
points[].dx`, e/cm) is plotted against a reference. Which end matters
depends on particle kind, so the mode auto-picks on every click and can be
overridden:

| mode | for | x axis | reference overlay |
|---|---|---|---|
| **End** | tracks (auto for a track-kind PF row) | residual range from the dumped `rr`, cm | muon/proton (solid) + pion/kaon (dashed) dQ/dx-vs-rr curves, plus the flat-MIP line |
| **Start** | showers (auto for a shower-kind PF row) | distance from the shower's own start point, recomputed from `points[].x/y/z` | horizontal lines at 1x and 2x the MIP scale (e⁻ vs. converted-γ), plus the tagger's own `stem_dqdx` samples as diamonds |

A shower's PF node id **is** its start segment's id (same encoding
`showers[].id` / `segments[].shower_id` use), so clicking a shower plots that
trunk by default; the **segment** dropdown lists the rest of the shower's
segments (start segment first) for stepping through the others.

The caption line under the plot gives the plotted segment's `particle_id`,
`particle_score`, `dirsign`, `dir_weak`, fitted `length`, point count, and the
reference-curve provenance (0.5 kV/cm Modified-Box recombination, the
retained 0.85 scale factor -- **not** a calibrated absolute charge).
Full write-up, including the unit-convention trap this panel had to get
right and a plan-stage mistake caught before implementation:
[`../docs/pr/42_dqdx-panel.md`](../docs/pr/42_dqdx-panel.md).

## Event features

Beside the particle flow:

- **selection** -- `nue_score`, `numu_score`, the **cosmic** verdict, `isFC`.
- **energy** -- reconstructed neutrino energy and the added rest-mass/binding
  term, plus the pi0 mass when one was identified.
- **topology** -- segment / shower / vertex counts, the neutrino vertex, and
  `fit_distance` = `|fit().point - wcpt().point|`: the gap between the vertex's
  fitted position and its current Steiner seed point.

  > **Do not read this as "how far the 3-D vertex fit moved the vertex", and do
  > not read 0 as "the fit did not run"** -- an earlier version of this file said
  > both, and both are wrong (corrected 2026-08-03, doc pr/28). The trajectory
  > fit moves `fit().point` for every vertex the vertex fit did *not* fix, and
  > `MyFCN::UpdateInfo` re-snaps `wcpt()` for the ones it *did*, so the number is
  > nonzero either way: 127 of 127 vertices on evt 388, degree-1 track ends
  > included. **Nothing in the dump says whether the vertex fit ran** -- that is
  > `PR::Fit::flag_fix`, which no artifact carries (doc pr/27 §14). Today the
  > only source is the trace log (`improve_vertex: ... fit_vertex done, vertex
  > moved ... cm`).
- **energy per particle** -- the `kine_*` arrays, with ✓ marking the entries
  actually summed into reco Enu.
- **BDT sub-scores** -- the 19 that are actually computed, behind a toggle.

## The cosmic answer

The field that says whether the cosmic tagger fired is **`cosmict_flag`**: the
OR of its ten tests (`clus/src/NeutrinoTaggerCosmic.cxx`). The panel shows the
verdict as one chip and the ten tests as a table -- **which one fired**, and
**whether each one ran at all**.

That second column is not decoration. Tests 2-8 only evaluate when their
topology precondition is met, so a `0` means either "ran and did not fire" or
"never ran", and on a neutrino-selected sample almost every row reads 0. Rows
that never ran are greyed. Evt 388 and 172230 show tests 2, 4 and 8 running and
not firing while the rest never run -- without the column those events are
indistinguishable from an inactive tagger.

Two fields that look like the cosmic answer and are not:

| field | why not |
|---|---|
| `cosmic_flag` | a **BDT input feature**, exactly `!cosmict_flag_9` -- its polarity is the reverse of its name, it covers one of the ten tests, and its default is 1, so a 1 is ambiguous between "never tested" and "rescued". Shown only in the table, via test 9 |
| `cosmict_score` | **never computed**, in the toolkit or the prototype. A legacy slot of the uBooNE ntuple schema belonging to the TMVA BDT path, which has no caller. Not dumped, not shown |

The same sweep removed **22** never-assigned fields in total (5 numu sub-scores,
15 nue sub-scores, `cosmict_score`, `photon_flag`) -- a displayed `0.00` that no
code ever wrote reads as a physics answer. Full account: doc pr/26 §8.

`photon_flag` was the one entry in that list that is a **port gap** rather than
a legacy slot -- the single-photon tagger ran and its verdict was discarded.
That is fixed (doc pr/26 §9) by `sp_photon_flag`, **now the SBND default**
(`SBND_SP_PHOTON_FLAG=0` restores the gap). So `photon_flag` is computed again
and is no longer a dead field -- it is simply **not on the panel yet**, deferred
to the next display round rather than dead. The other 21 stay dead.

> **The BDT scores are UNCALIBRATED on SBND.** The config books the
> uBooNE-trained weight XMLs (doc pr/2 gap G1); the numbers carry availability
> and relative ranking only. The panel prints this in red and the dump carries
> the same string in `tagger.weights` so it cannot be shown without it.

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

## Not here yet

Hand-scan label saving, batch pre-rendering, and any change to the PR
algorithms themselves. This is read-only viewing plus the dump that feeds it.

The particle-flow table's doc-58 `DataTable` refresh fix (view-filter flip after
every `.data` assignment) **is now verified across event steps** -- seven events
give seven distinct row-sets, counts 3 to 13, and stepping back reproduces the
first exactly (doc pr/26 §8.4). It was carried as unverified through stage 2.

Still unexercised: **no event tried so far has any cosmic test fire**, so the
cosmic table's FIRED rendering has not been seen on real data. All 14 events
from the nueCC-selected sample read `cosmict_flag = 0`, which is the expected
direction for that sample -- check it against a known cosmic before relying on
the decomposition in a scan.

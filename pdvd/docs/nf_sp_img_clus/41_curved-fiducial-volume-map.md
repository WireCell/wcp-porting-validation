# PDVD doc 41 — the apparent side-wall position as a function of drift: can a curved (space-charge) fiducial surface be mapped from the Q/L-matched cosmics?

**Status (2026-09-04).** Measurement + write-up only. No toolkit code, no config,
no arm, no `work/` directory written. The inputs are the 120-event `d28dlfp`
cosmic-data arm already on disk (the arm docs 33 and 35 measured on; the imaged
points and their flash t0 are invariant across every full-120 arm). Nothing in
this document is wired into any job; §8 says what would be.

Owner question: MicroBooNE's fiducial volume is smaller on the cathode side
(space charge). PDVD has Q/L-matched events, hence t0-corrected cosmics. Is
there enough data to map a *curved* fiducial surface, smooth, factorized Y-vs-X
and Z-vs-X the way the prototype does it?

**Short answer.** **Yes.** The 120-event arm is enough to map the surface, and the surface is
real, large, and *not* the flat 15 cm inset in use. Measured as the half-density
point of the imaged charge next to each wall (§3), in 20 cm bins of drift with
event-bootstrap errors (fig. 1, table §5):

- **The z walls carry a cathode-side inset that decays to zero over the last
  150–200 cm of drift**: at the cathode z+ is **17.7 ± 1.3 cm** (bottom volume)
  and **11.0 ± 0.6 cm** (top), z− is **11.2 ± 1.4 cm** (bottom) and 3–10 cm
  (top, not well described by a single ramp). The knee is at |x| ≈ 165–205 cm.
- **The y walls are asymmetric**: y− shows **13.9 ± 1.1 cm** at the cathode in
  the bottom volume only (knee 190 ± 8 cm) and ≤ 2.5 cm in the top; y+ shows
  **no inset anywhere** (0.2 ± 0.7 cm bottom, 2.8 ± 1.2 cm top).
- **The anode-side control passes**: for |x| ≥ 250 cm every wall in both
  volumes reads the nominal `sensvol` wall to within 1.1 cm (§4).
- **It is a displacement, not a loss of charge**: near-wall tracks extrapolated
  from their anode-side straight line are still imaged near the cathode
  (survival 100 % where the test has tracks) but sit **2–7 cm further from the
  z walls** than the line predicts (§4.3).
- **The factorization the prototype assumes holds** at the present precision:
  re-measuring each y-wall profile in z-slabs and each z-wall profile in the
  four CRU quarters of y gives χ²/ndf of 13–43/34 against the all-slab profile,
  mean differences ≤ 0.6 cm (§6). Two corner quarters at the cathode read
  ~10 cm lower than the rest at ~2σ; more events would settle them.
- **The mirror symmetries do NOT hold** (§6): bottom ≠ top on z+ (χ² 151/34)
  and y− (70/34); y+ ≠ y− in the bottom volume (110/34). So the surface must be
  kept per (wall, drift volume): eight one-dimensional profiles, not two.

Both a MicroBooNE-style flat-plus-linear-ramp (M1) and a kink-free power law
(M2) fit every profile with χ²/ndf ≈ 0.5–1.7 (§7); M1 is the recommendation
because it is what `PolyFiducial` consumes without approximation. The
resulting polygons, one X-Y and one X-Z per drift volume, are in §7.2 and
`figs/41_polygons_M1_d50.json`. Nothing is wired; §8 says what the flat inset
gets wrong on both sides and what to do next.

---

## 0. Repro

```bash
WCPI=/home/xqian/toolkit-dev/wcp-porting-img
cd $WCPI/pdvd
S=docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/doc41

# (1) cache the t0-corrected points + raw readout x + per-cluster t0 (~10 min at 16 jobs)
python3 $S/fv_curved_load.py work/*_d28dlfp --out /home/xqian/tmp/doc41/points_d28dlfp.npz --jobs 16

# (2) the edge profiles, tests, model fits, polygons (10 cm drift bins, 200 event bootstraps)
python3 $S/fv_curved_map.py /home/xqian/tmp/doc41/points_d28dlfp.npz --out /home/xqian/tmp/doc41/map --boot 200
#     the coarse 20 cm version used for the tables
python3 $S/fv_curved_map.py /home/xqian/tmp/doc41/points_d28dlfp.npz --out /home/xqian/tmp/doc41/map20 --boot 200 --xstep 20

# (3) figures
python3 $S/fv_curved_plots.py /home/xqian/tmp/doc41/map_result.json \
    --npz /home/xqian/tmp/doc41/points_d28dlfp.npz --figdir docs/nf_sp_img_clus/figs --prefix 41
```

The per-bin table this document quotes is committed as
`docs/nf_sp_img_clus/figs/41_edges.csv` (the `map20` run), so every number below
can be re-derived without rerunning.

---

## 1. What MicroBooNE does, precisely

The prototype's `WCPPID::ToyFiducial`
(`prototype_base/wire-cell/pid/src/ToyFiducial.cxx:60-135`) is not a box. In
each of the X-Y and X-Z planes it is a 6-vertex polygon, **flat near the anode
and ramping linearly inward toward the cathode**, and a point is inside the
fiducial volume iff it is inside *both* polygons
(`inside_fiducial_volume`, `:1756-1818`: `c1 && c2`). That AND of two
2-D tests is the factorization: the y-boundary depends on x only, the
z-boundary depends on x only. The data-mode space-charge vertices (cm; anode at
x = 0, cathode at 256):

| face | flat until x = | value at the anode | value at the cathode |
|---|---|---|---|
| top (y) | 100 | +116 | +102 |
| bottom (y) | 80 | −116 | −99 |
| upstream (z) | 120 | 0 | 11 |
| downstream (z) | 120 | 1037 | 1026 |

So MicroBooNE's inset at the cathode is 14-17 cm in y and 11 cm in z, reached
by a linear ramp over the last 136-176 cm of a 256 cm drift, after a
`boundary_dis_cut` of 3 cm everywhere. A finer variant
(`ToyFiducial.cxx:139-208`) bins the X-Y polygon in ten 1 m slabs of z and the
X-Z polygon in ten 24 cm slabs of y — the same factorization with a weak
corner dependence.

WCT already carries the vehicle: `aux/src/PolyFiducial.cxx` ("the ToyFiducial
port": stacked polygonal slabs, pnpoly per slab) AND-ed through
`CompositeFiducial{logic:'and'}`. No config in the tree instantiates it (doc 33
§3.1). PDVD's space-charge allowance today is the **flat 15 cm y/z inset,
uniform in x** (`protodunevd/clus.jsonnet:100-105`, tagger margins closed on it
in doc 35), which doc 35 §6 calls "adopted, not measured".

---

## 2. Inputs, frames, and the cuts

Per event dir `work/<run6>_<idx>_d28dlfp/`:

- `mabc-pr.zip: data/0/0-clustering-global.json` — every imaged 3-D point of
  the event (~65k-140k), `x` already t0-corrected, cm. Clusters without a flash
  carry the sentinel x = ±1.48e8 cm.
- `tracking-pr.root: T_cluster` — `cluster_id` (1:1 with the Bee layer, verified
  on every event: the sentinel-x cluster set equals the `cluster_t0_us ≤ −1e5`
  set), the flash `cluster_t0_us`.
- `pctree-evt*.tar.gz: 3d/{x, x_t0cor, y, z, wpid}` — the RAW readout x and the
  anode id of every point. The Bee layer of this arm is a PR-stage
  **re-partition in a different order** (29254 of 65532 points on 039252/0), so
  the raw x is attached by a k-d tree match on (x_t0cor, y, z) at 0.01 cm, with
  a fallback for clusters the PR stage re-timed (same (y, z), |Δx| < half a
  drift step; 558 points on 039252/4, one cluster, all 0.117 cm off), gated by
  requiring `xraw − x` constant within every (cluster, anode side).

Geometry (the 16 `sensvol:` lines, cm): cathode slab |x| < 3.0; anode planes
|x| = 339.91; y ±336.39 with seams at |y| = 0.61 and 168.5; z 0.813-298.435 with
a seam at 149.65. Bottom drift volume = anodes 0-3 = x < 0; top = anodes 4-7 =
x > 0.

The raw readout frame per side runs from the anode plane (tick 0, raw
x = ∓341.55) to the late window edge (tick 10000, raw x = ±398.52). A track
whose flash t0 puts its far end past the window is cut there; in t0-corrected
coordinates that cut lands anywhere (doc 25 M5). It removes points along x and
does not move a track in y or z, so the side-wall instrument is immune; the
endpoint instrument masks ends within 60 ticks (4.4 cm) of either raw edge.

Cuts, in order, with what they removed:

| step | what | removed | left |
|---|---|---|---|
| 0 | imaged points, 120 events | – | 9,597,695 points in 11,241 clusters |
| 1 | no flash t0 (sentinel x; `cluster_t0_us ≤ −1e5` — the two sets are identical on every event) | 289,079 points (3.0 %), 2,804 clusters, all small | 8,437 clusters = 9,097 tracks (a cathode crosser is two tracks, one per anode side) |
| 2 | wrong t0: any point of the track more than 5 cm past an anode plane (\|x\| > 345) | 489 tracks (5.4 %) but 1,477,479 points (15.9 %) — they are the long ones, median 800–1800 points vs 210 | **7,831,137 points** |
| 3 | per wall: within 60 cm of that wall, ≥ 15 cm from the other walls, > 3 cm from a seam | (selection, not a cut) | ~10–30 k points and 80–400 tracks per wall per 20 cm bin |

The endpoint instrument (§3) sees 9,228 ends of long tracks, of which
1,254 (13.6 %) sit at the readout-window edge and are masked.

The `|x| > 305` cut of the `stm/` scripts is deliberately **not** used: it would
delete the anode-side control region, and its premise was retracted by doc 33
§7.

---

## 3. The instrument

For a wall W ∈ {y+, y−, z−, z+} and a drift bin B (10 cm in |x| from 3 to 340,
each volume separately), every flash-matched **track** (a cluster on one anode
side) with ≥ 3 points in B contributes one number: the minimum distance of
those points from the nominal wall, `d_min`. Points within 15 cm of any other
wall or 3 cm of a seam are excluded so the sample is one wall's. Cosmic tracks
are uniform in the transverse coordinates, so above the apparent edge `d_min`
is uniform and its cumulative count C(d) is a ramp starting at the edge. C(d) on
a 0.5 cm grid is fitted with an erf-smoothed ramp,

    C(d) = λ [ u Φ(u/σ) + σ φ(u/σ) ],   u = d − d0,

whose onset d0 is the apparent inset of the wall in that drift bin. A
cumulative is comb-free at any bin width (doc 34), and counting tracks rather
than points removes the ~33-point-per-track weight a vertical track carries in
a 10 cm x-bin. Errors are an event-level bootstrap (200 resamples). The
second-smallest `d_min` is kept alongside as a model-free onset.

Two controls are built in. **The anode-side bins** (|x| > 250) are far from the
cathode and must return d0 ≈ 0 to within the point-pitch floor (0.29 cm in y,
0.48 in z, doc 34) — if they do not, the instrument is broken, not the
detector. **The track-endpoint instrument** is independent of the ramp model:
PCA ends of tracks with ≥ 40 points and ≥ 30 cm extent whose *other* end is at
a boundary (within 12 cm of the nominal box, or at the readout window) are
through-going by construction, so the mode of their wall distance in a drift
bin is the apparent wall.

Factorization is tested, not assumed: the y-wall profile is re-measured in
three slabs of z and the z-wall profile in the four CRU quarters of y, each
compared to the all-slab profile by χ².

---

## 4. Instrument validation

### 4.1 The anode-side control

Inverse-variance mean of d50 over the five bins with |x| ≥ 250 cm (the control),
and over the three bins with |x| < 60 cm (the effect), per wall and volume:

| wall | bottom, |x| ≥ 250 | top, |x| ≥ 250 | bottom, |x| < 60 | top, |x| < 60 |
|---|---|---|---|---|
| y+ | 0.34 ± 0.53 | 0.47 ± 0.33 | **0.1 ± 0.6** | **2.5 ± 0.9** |
| y- | 0.07 ± 0.64 | 0.85 ± 0.21 | **7.7 ± 1.7** | **1.5 ± 0.7** |
| z- | -0.04 ± 0.21 | 0.88 ± 0.09 | **8.2 ± 1.3** | **6.4 ± 2.2** |
| z+ | -0.28 ± 0.31 | -0.22 ± 0.15 | **15.5 ± 1.3** | **9.5 ± 0.5** |

Every control value is within 1.1 cm of zero — the pitch floor is 0.29 cm in y
and 0.48 cm in z, so the top volume's z− and y− walls carry a real ~1 cm offset
from the `sensvol` envelope (the top CRP's tiles start 0.1 cm later in z than
the bottom's, `35_*.md` §2, and a 1 cm effect is the size of a half-density
point's bin) that a fiducial would inherit harmlessly. The cathode-side values
are between 2.5 σ (z−, top) and 18 σ (z+, top) above their controls on the z
walls and on y−/bottom (4.2 σ); y+/bottom sits at its control and y+/top 2.2 σ
above it.

### 4.2 The track-endpoint instrument agrees

The mode of the wall distance of through-going track ends (a different
observable, no density model, 20 cm bins with ≥ 5 ends) against d50:

| wall | bottom χ²/ndf (mean end − d50) | top χ²/ndf (mean end − d50) |
|---|---|---|
| y+ | 1.4/16 (+0.4 cm) | 0.6/17 (+0.0 cm) |
| y- | 0.9/16 (+0.6 cm) | 3.5/13 (-1.1 cm) |
| z- | 12.9/17 (+0.2 cm) | 13.0/17 (+0.5 cm) |
| z+ | 5.2/17 (-0.1 cm) | 9.0/12 (+0.5 cm) |

χ²/ndf below one everywhere and mean offsets within 1.1 cm: the two instruments
see the same wall. (The end mode is a coarse statistic on the sparse
near-cathode samples, 10–40 ends per bin, hence its large per-bin errors.)

### 4.3 Displacement, not loss

For every long track (≥ 40 points) a straight line `d = a + b·|x|` is fitted to
its wall distance on the anode side (|x| ≥ 200 cm, ≥ 15 points, RMS < 2 cm)
and extrapolated toward the cathode; only lines that stay ≥ 3 cm inside the
wall over the track's whole range are used, so a track's presence in a bin is
not conditional on its having been displaced. Aggregated over the bins with
|x| < 100 cm, per band of *predicted* distance from the wall — survivors /
tracks whose line crosses the bin, and the mean residual (measured − predicted,
positive = pushed inward):

| wall | volume | reference tracks | line predicts 0–10 cm: survive, residual | 10–20 cm: survive, residual | 20–40 cm: survive, residual |
|---|---|---|---|---|---|
| y+ | bottom | 4 | 2/2, -2.6 ± 1.6 | 8/8, +1.6 ± 1.9 | 3/3, +27.8 ± 1.6 |
| y+ | top | 3 | – | 4/4, -1.2 ± 0.7 | 4/4, +3.8 ± 3.2 |
| y- | bottom | 4 | – | – | 2/2, -2.4 ± 0.2 |
| y- | top | 9 | – | 5/5, -5.2 ± 1.3 | 6/6, +1.7 ± 0.2 |
| z- | bottom | 13 | – | 1/1, -3.6 | 15/19, -2.2 ± 1.5 |
| z- | top | 16 | 4/4, +4.5 ± 1.3 | 5/5, +2.2 ± 1.1 | 16/16, +2.9 ± 0.8 |
| z+ | bottom | 15 | – | 6/6, +5.8 ± 0.9 | 14/14, +2.4 ± 1.8 |
| z+ | top | 21 | 6/6, +7.4 ± 1.0 | 18/18, +5.8 ± 1.4 | 3/3, -3.5 ± 0.2 |

Where the test has tracks it is unambiguous: **survival is 100 %** (the z−
bottom 15/19 is the one exception, 4 tracks lost in the 20–40 cm band) and the
tracks predicted to pass within 20 cm of a z wall are imaged **2–7 cm further
inside** than their own straight line says (z+: +5.8 ± 0.9, +7.4 ± 1.0 and
+5.8 ± 1.4; z−/top: +4.5 ± 1.3 and +2.2 ± 1.1). The imaged charge is moved inward,
not removed — the space-charge signature, and the reason the apparent wall
moves. The y walls have too few qualifying tracks (3–9) to say; the y+/bottom
20–40 cm entry is three entries of one aberrant track.

The amplitude here (2–7 cm) is smaller than d50 at the cathode (11–18 cm)
because the followed tracks are those predicted 0–20 cm from the wall averaged
over |x| < 100 cm, and the displacement field falls off both with distance from
the wall and with distance from the cathode (fig. 3, `figs/41_bending.png`).

---

## 5. The apparent wall inset versus drift

Half-density point d50 (cm inside the nominal `sensvol` wall) per 20 cm bin of
|x|, bottom (x < 0) and top (x > 0) drift volumes, with the event-bootstrap
error, the sample size, and the through-going-end mode where ≥ 5 ends exist.
The full table with the equivalent-length estimator, the per-track onset and
the endpoint columns is `figs/41_edges.csv`; fig. 1 is `figs/41_edge_vs_x.png`
and the occupancy maps with d50 overlaid are `figs/41_occupancy.png`.

**Wall y+** (y = +336.39):

| |x| bin centre (cm) | bottom d50 ± err | bottom n_pts / n_trk | top d50 ± err | top n_pts / n_trk | ends (bot / top) mode |
|---|---|---|---|---|---|
| 12 | **-0.5 ± 0.9** | 9269 / 109 | **1.9 ± 2.7** | 12835 / 115 | 14.5±12.9 / 4.5±15.3 |
| 30 | **1.0 ± 1.0** | 15476 / 106 | **2.4 ± 1.2** | 15332 / 123 | 0.5±14.8 / 4.5±7.6 |
| 50 | **-0.0 ± 1.3** | 10790 / 91 | **3.0 ± 1.8** | 14221 / 107 | 0.5±14.6 / 2.5±13.5 |
| 70 | **0.4 ± 1.2** | 10563 / 93 | **0.9 ± 0.7** | 11594 / 100 | 0.5±3.0 / 1.5±7.8 |
| 90 | **-0.1 ± 1.2** | 10818 / 91 | **0.5 ± 1.9** | 15296 / 122 | 0.5±7.1 / 0.5±11.6 |
| 110 | **0.2 ± 1.0** | 10175 / 88 | **0.6 ± 0.8** | 17560 / 119 | 0.5±16.1 / 0.5±9.2 |
| 130 | **-0.8 ± 0.5** | 9542 / 82 | **-0.7 ± 0.5** | 17635 / 123 | 0.5±6.9 / 0.5±3.9 |
| 150 | **0.1 ± 0.5** | 9902 / 92 | **0.1 ± 1.1** | 18784 / 117 | -0.5±13.6 / 0.5±5.4 |
| 170 | **0.1 ± 0.9** | 11528 / 82 | **0.6 ± 1.6** | 18953 / 121 | 0.5±13.6 / 0.5±2.4 |
| 190 | **0.2 ± 0.8** | 13652 / 93 | **-0.6 ± 0.6** | 17037 / 131 | 0.5±8.7 / 0.5±2.1 |
| 210 | **-0.5 ± 0.7** | 13789 / 98 | **0.3 ± 0.6** | 26852 / 143 | 0.5±12.4 / 0.5±6.4 |
| 230 | **-0.1 ± 0.9** | 13417 / 92 | **0.1 ± 1.3** | 21998 / 158 | -0.5±6.1 / 0.5±4.8 |
| 250 | **0.1 ± 0.6** | 12496 / 103 | **1.2 ± 1.0** | 31753 / 155 | 0.5±2.5 / 0.5±9.8 |
| 270 | **0.9 ± 4.9** | 11428 / 97 | **0.8 ± 0.7** | 33021 / 158 | 0.5±2.9 / 0.5±0.8 |
| 290 | **1.1 ± 2.7** | 8454 / 74 | **0.5 ± 0.6** | 33515 / 152 | 0.5±11.3 / 0.5±1.8 |
| 310 | **1.1 ± 1.4** | 5460 / 51 | **0.2 ± 0.6** | 20531 / 136 | – / 0.5±3.1 |
| 330 | **0.8 ± 2.8** | 4780 / 59 | **-0.6 ± 1.2** | 14051 / 120 | 0.5±20.9 / 0.5±11.8 |

**Wall y−** (y = −336.39):

| |x| bin centre (cm) | bottom d50 ± err | bottom n_pts / n_trk | top d50 ± err | top n_pts / n_trk | ends (bot / top) mode |
|---|---|---|---|---|---|
| 12 | **6.2 ± 4.1** | 9499 / 100 | **1.6 ± 1.2** | 17833 / 93 | 10.5±21.5 / 0.5±17.0 |
| 30 | **6.5 ± 3.1** | 11903 / 95 | **2.4 ± 1.4** | 18110 / 95 | 11.5±8.7 / 7.5±12.3 |
| 50 | **8.8 ± 2.3** | 11985 / 99 | **0.8 ± 1.2** | 13814 / 97 | 8.5±19.6 / 1.5±6.5 |
| 70 | **9.8 ± 0.8** | 10768 / 97 | **1.3 ± 1.7** | 13734 / 102 | 11.5±8.6 / 3.5±9.4 |
| 90 | **8.2 ± 2.3** | 10425 / 97 | **2.1 ± 3.5** | 16271 / 102 | 10.5±15.0 / 3.5±10.0 |
| 110 | **7.2 ± 3.8** | 9288 / 84 | **8.3 ± 3.8** | 17788 / 98 | 8.5±1.4 / 0.5±10.3 |
| 130 | **6.6 ± 2.1** | 11859 / 104 | **2.7 ± 4.7** | 21368 / 105 | 4.5±4.1 / 0.5±10.0 |
| 150 | **4.2 ± 1.5** | 14029 / 102 | **1.2 ± 3.4** | 22859 / 127 | 0.5±15.8 / 0.5±3.7 |
| 170 | **0.9 ± 0.9** | 14287 / 98 | **0.9 ± 1.6** | 20465 / 135 | 0.5±14.1 / 0.5±5.1 |
| 190 | **-0.3 ± 0.7** | 10536 / 97 | **2.1 ± 3.7** | 29501 / 134 | 0.5±3.4 / 0.5±0.8 |
| 210 | **0.5 ± 1.5** | 10911 / 95 | **0.7 ± 0.6** | 43584 / 158 | 0.5±11.2 / 0.5±0.0 |
| 230 | **0.3 ± 1.2** | 14234 / 97 | **1.7 ± 1.1** | 36314 / 155 | 0.5±18.3 / 0.5±2.5 |
| 250 | **0.2 ± 3.2** | 17610 / 98 | **1.2 ± 0.5** | 41417 / 168 | – / 0.5±0.0 |
| 270 | **-0.1 ± 2.3** | 14682 / 103 | **1.4 ± 1.2** | 46114 / 159 | 0.5±9.8 / 0.5±1.1 |
| 290 | **0.2 ± 0.8** | 10789 / 83 | **0.7 ± 0.4** | 39256 / 147 | 0.5±9.6 / 0.5±0.0 |
| 310 | **-0.3 ± 1.9** | 9362 / 77 | **0.1 ± 0.4** | 26648 / 132 | 0.5±10.1 / 0.5±0.0 |
| 330 | **-0.9 ± 3.1** | 8409 / 76 | **1.6 ± 0.5** | 43554 / 124 | 1.5±12.6 / 0.5±0.7 |

**Wall z−** (z = 0.813):

| |x| bin centre (cm) | bottom d50 ± err | bottom n_pts / n_trk | top d50 ± err | top n_pts / n_trk | ends (bot / top) mode |
|---|---|---|---|---|---|
| 12 | **12.2 ± 4.0** | 27437 / 279 | **1.7 ± 3.4** | 40611 / 252 | 40.5±13.1 / 8.5±14.4 |
| 30 | **12.6 ± 3.2** | 38057 / 239 | **9.1 ± 4.3** | 56103 / 253 | 0.5±10.8 / 1.5±9.0 |
| 50 | **6.6 ± 1.5** | 37173 / 241 | **10.4 ± 3.8** | 81446 / 247 | 8.5±8.7 / 2.5±5.6 |
| 70 | **8.1 ± 2.0** | 30889 / 222 | **3.8 ± 3.1** | 39196 / 250 | 40.5±15.6 / 2.5±4.5 |
| 90 | **8.8 ± 2.7** | 27249 / 205 | **6.9 ± 5.7** | 54018 / 297 | 8.5±12.3 / 1.5±10.8 |
| 110 | **3.7 ± 2.7** | 28090 / 202 | **2.6 ± 0.3** | 71107 / 322 | 7.5±2.5 / 0.5±19.3 |
| 130 | **6.3 ± 2.0** | 26896 / 198 | **2.5 ± 1.4** | 66044 / 350 | 1.5±13.0 / 0.5±1.0 |
| 150 | **2.3 ± 1.4** | 27510 / 211 | **2.5 ± 1.5** | 63250 / 341 | 0.5±6.4 / 1.5±5.2 |
| 170 | **1.4 ± 0.8** | 28324 / 223 | **1.7 ± 2.5** | 69215 / 375 | 0.5±0.3 / 1.5±0.3 |
| 190 | **1.4 ± 1.4** | 27291 / 221 | **1.5 ± 2.4** | 83294 / 370 | 0.5±3.6 / 1.5±1.0 |
| 210 | **1.4 ± 2.2** | 32132 / 221 | **2.0 ± 4.8** | 123003 / 405 | 0.5±8.5 / 1.5±0.3 |
| 230 | **1.0 ± 3.0** | 29404 / 220 | **0.9 ± 0.3** | 83922 / 388 | 0.5±1.1 / 1.5±0.3 |
| 250 | **0.1 ± 0.6** | 30601 / 219 | **0.8 ± 0.1** | 107462 / 386 | 0.5±0.7 / 1.5±0.3 |
| 270 | **0.1 ± 0.3** | 29073 / 202 | **1.1 ± 0.2** | 108876 / 381 | 0.5±7.5 / 1.5±0.3 |
| 290 | **-0.1 ± 0.5** | 25916 / 180 | **1.0 ± 0.2** | 86818 / 344 | 0.5±1.5 / 1.5±0.4 |
| 310 | **-0.1 ± 0.9** | 23733 / 155 | **0.7 ± 0.2** | 78670 / 328 | 0.5±0.2 / 1.5±2.6 |
| 330 | **-0.5 ± 0.5** | 15486 / 157 | **1.0 ± 0.3** | 82149 / 293 | 0.5±10.7 / 1.5±0.5 |

**Wall z+** (z = 298.435):

| |x| bin centre (cm) | bottom d50 ± err | bottom n_pts / n_trk | top d50 ± err | top n_pts / n_trk | ends (bot / top) mode |
|---|---|---|---|---|---|
| 12 | **18.4 ± 3.5** | 29450 / 276 | **11.3 ± 0.7** | 31057 / 286 | 37.5±13.4 / 17.5±18.2 |
| 30 | **16.7 ± 2.2** | 29437 / 253 | **9.6 ± 1.2** | 42290 / 279 | 19.5±8.6 / 30.5±15.0 |
| 50 | **14.1 ± 1.7** | 28747 / 225 | **6.6 ± 0.9** | 37592 / 264 | 17.5±9.2 / 1.5±8.1 |
| 70 | **9.0 ± 1.6** | 28327 / 210 | **4.8 ± 1.4** | 32426 / 271 | 16.5±9.6 / 6.5±5.5 |
| 90 | **9.8 ± 1.3** | 25193 / 200 | **3.8 ± 1.1** | 41882 / 261 | 7.5±10.9 / 5.5±7.4 |
| 110 | **12.1 ± 3.2** | 24077 / 204 | **3.8 ± 0.7** | 39559 / 269 | 6.5±5.0 / 4.5±6.2 |
| 130 | **9.6 ± 1.9** | 24571 / 209 | **2.7 ± 0.7** | 47384 / 267 | 0.5±9.2 / 4.5±2.7 |
| 150 | **4.8 ± 1.5** | 27554 / 226 | **1.3 ± 0.8** | 42879 / 318 | 0.5±9.4 / 0.5±0.3 |
| 170 | **2.2 ± 0.8** | 28405 / 228 | **0.3 ± 0.9** | 44953 / 320 | 0.5±6.8 / 0.5±0.0 |
| 190 | **4.2 ± 1.9** | 30436 / 225 | **0.1 ± 0.5** | 52793 / 334 | 0.5±8.8 / 0.5±0.0 |
| 210 | **-0.1 ± 1.7** | 31619 / 250 | **0.4 ± 0.4** | 66712 / 344 | 0.5±4.7 / 0.5±0.0 |
| 230 | **0.1 ± 2.7** | 29480 / 237 | **0.1 ± 0.5** | 49583 / 351 | 0.5±1.7 / 0.5±0.0 |
| 250 | **-0.5 ± 0.5** | 31144 / 228 | **-0.1 ± 0.4** | 78155 / 375 | -0.5±4.4 / 0.5±0.0 |
| 270 | **0.8 ± 1.2** | 30018 / 225 | **-0.1 ± 0.3** | 87956 / 353 | 0.5±10.1 / 0.5±2.1 |
| 290 | **-0.2 ± 1.0** | 25876 / 215 | **-0.1 ± 0.3** | 64461 / 337 | -0.5±1.5 / 0.5±0.1 |
| 310 | **-0.4 ± 0.6** | 23497 / 179 | **-0.5 ± 0.3** | 45069 / 294 | 0.5±2.5 / 0.5±1.3 |
| 330 | **-0.3 ± 0.6** | 19954 / 175 | **-0.0 ± 0.6** | 39088 / 261 | -0.5±4.7 / 0.5±4.3 |

Reading the tables: on z+ the inset is 18 → 17 → 14 → 9 cm over the first four
bottom bins and 11 → 10 → 7 → 5 over the first four top bins, reaching the
control level at |x| ≈ 170–210 cm; on z− the bottom volume behaves the same
way at 12 → 13 → 7 → 8 cm while the top holds ~10 cm at 30–50 cm but only 2 cm
at the cathode face and 2–3 cm out to 150 cm; y−/bottom is 6 → 6 → 9 → 10 cm
and gone by 170 cm; y−/top and y+ in both volumes never leave 0–3 cm.

The equivalent-length estimator (`d_eq_cm` in the CSV) agrees with d50 on the
z walls within its errors but those errors are 3–18 cm — extrapolating the
plateau line 30 cm back to zero amplifies the density gradient — so it is not
used further. The per-track onset (`d_2nd_cm`) is ~0 in every bin including
the cathode ones: a handful of tracks always reach the nominal wall, which is
why a minimum-based edge would have missed this effect entirely.

---

## 6. Symmetries and factorization

### 6.1 Mirror symmetries — tested on the 10 cm profiles (34 bins each)

| test | statistic | verdict |
|---|---|---|
| bottom vs top, wall y+ | χ² 21 / 34 | consistent |
| bottom vs top, wall y- | χ² 70 / 34 | **not symmetric** |
| bottom vs top, wall z- | χ² 45 / 34 | consistent |
| bottom vs top, wall z+ | χ² 151 / 34 | **not symmetric** |
| y+ vs y−, bottom | χ² 110 / 34 | **not symmetric** |
| y+ vs y−, top | χ² 20 / 34 | consistent |
| z− vs z+, bottom | χ² 56 / 34 | **not symmetric** |
| z− vs z+, top | χ² 88 / 34 | **not symmetric** |

The pattern is coherent: **the bottom drift volume (x < 0) is more distorted
than the top on every wall that shows an effect, and the two y walls differ**
(y− yes, y+ no, in the bottom). The instrument is symmetric by construction
(one code path with sign flips; the anode-side controls agree between volumes
and walls to ≤ 1 cm), so the asymmetry is in the data. It is what a
detector-specific field distortion would do — the cathode frame, field-cage
and HV-feed geometry are not mirror-symmetric, and the liquid flow is not —
and it means no fold across x or across y is allowed. The surface is eight
one-dimensional profiles.

### 6.2 Factorization — tested on the 20 cm profiles (17 bins × 2 volumes)

The prototype's `c1 && c2` assumes the y-wall inset depends on x only and the
z-wall inset on x only. Re-measuring each y-wall profile in three z-slabs and
each z-wall profile in the four CRU quarters of y (`figs/41_factorization.png`):

| wall | slice of the other coordinate | χ²/ndf vs the all-slice profile (both volumes) | mean difference (cm) |
|---|---|---|---|
| y+ | z<100 | 12.8/34 | +0.42 |
| y+ | 100<z<200 | 12.8/34 | +0.22 |
| y+ | z>200 | 16.8/34 | -0.03 |
| y- | z<100 | 19.7/34 | -0.27 |
| y- | 100<z<200 | 18.3/34 | +0.29 |
| y- | z>200 | 13.1/34 | +0.20 |
| z- | y<-168 | 21.1/34 | +0.15 |
| z- | -168<y<0 | 15.7/34 | -0.09 |
| z- | 0<y<168 | 25.5/34 | -0.10 |
| z- | y>168 | 27.7/34 | +0.47 |
| z+ | y<-168 | 42.8/34 | +0.44 |
| z+ | -168<y<0 | 18.3/34 | +0.00 |
| z+ | 0<y<168 | 25.8/34 | +0.28 |
| z+ | y>168 | 32.8/34 | +0.56 |

Every slice is consistent with the all-slab profile at χ²/ndf ≤ 1.3 with mean
differences ≤ 0.6 cm, so **the factorized form is justified at this
precision**. Two things to carry forward rather than hide: at the cathode the
z− bottom profile in the y > 168 quarter reads ~4–5 cm where the other three
quarters read 12–16, and the z+ bottom profile in the y < −168 quarter reads
~5–9 cm where the others read 18–22 (fig. 2, bottom rows) — 2σ-level hints
that the corners are less distorted than the wall centres, exactly the weak
corner dependence MicroBooNE encodes with its ten z-binned X-Y polygons. The
per-slice errors at the cathode are 3–5 cm; four times the events would make
this a measurement.

---

## 7. A smooth surface

### 7.1 Two parametrizations, per wall and per drift volume (20 cm profiles)

M1 is MicroBooNE's shape — inset 0 for |x| ≥ x_knee, linear to Δc at the
cathode face |x| = 3 — and M2 is a kink-free power law Δc·((339.91 − |x|)/336.91)^p,
zero at the anode by construction. Both fitted to d50 with its bootstrap
errors:

| wall | volume | M1: inset at cathode Δc (cm) | M1: knee |x| (cm) | M1 χ²/ndf | M2: Δc (cm) | M2: power p | M2 χ²/ndf |
|---|---|---|---|---|---|---|---|
| y+ | bottom | 0.1 ± 0.7 | 130 ± 410 | 5.0/15 | -0.1 ± 1.5 | 10.00 ± 409.20 | 5.1/15 |
| y+ | top | 2.8 ± 1.2 | 127 ± 41 | 8.2/15 | 4.3 ± 2.6 | 7.60 ± 4.58 | 9.3/15 |
| y- | bottom | 13.9 ± 1.1 | 190 ± 8 | 10.6/15 | 16.2 ± 2.2 | 3.05 ± 0.47 | 23.6/15 |
| y- | top | 2.4 ± 0.6 | 340 ± 36 | 19.6/15 | 1.3 ± 0.5 | 0.20 ± 0.21 | 13.0/15 |
| z- | bottom | 11.2 ± 1.4 | 201 ± 15 | 7.5/15 | 13.1 ± 2.1 | 2.87 ± 0.52 | 7.8/15 |
| z- | top | 3.8 ± 0.4 | 340 ± 14 | 25.1/15 | 3.0 ± 0.5 | 0.78 ± 0.14 | 23.6/15 |
| z+ | bottom | 17.7 ± 1.2 | 205 ± 10 | 12.8/15 | 21.3 ± 2.0 | 2.87 ± 0.33 | 16.6/15 |
| z+ | top | 11.0 ± 0.6 | 165 ± 8 | 11.4/15 | 12.4 ± 0.8 | 3.71 ± 0.35 | 8.5/15 |

Both forms describe every profile (χ²/ndf 0.3–1.7); where the fitted amplitude
is under 2σ from zero (y+ bottom) the wall is flat. M1 is the recommendation:
it is what `PolyFiducial` represents exactly, and MicroBooNE's own data-mode
numbers are the same shape (knee 80–120 cm of a 256 cm drift, Δc 11–17 cm;
here knee 165–205 cm of a 340 cm drift, Δc 11–18 cm — the same fraction of the
drift, a comparable amplitude). The top z− wall is the one profile M1 handles
poorly (χ² 25/15): its ~10 cm bump at 30–50 cm is not a ramp to the cathode
face; M2 does no better. A two-knee polygon would, and `PolyFiducial` accepts
any vertex list.

### 7.2 The surface as the prototype's polygons

`boundary_dis_cut` = 0 here (apply the tagger margins on top, as today).
Vertices run anode → knee → cathode face on the negative wall, then cathode
face → knee → anode on the positive wall, exactly the prototype's order
(`ToyFiducial.cxx:117-135`). Units cm, WCT frame.

Bottom drift volume (x < 0, anodes 0–3):

```
boundary_xy (x, y): (-339.9, -336.4), (-190.0, -336.4), (-3.0, -322.4), (-3.0, 336.4), (-339.9, 336.4), (-339.9, 336.4)
boundary_xz (x, z): (-339.9, 0.8), (-200.9, 0.8), (-3.0, 12.0), (-3.0, 280.8), (-205.1, 298.4), (-339.9, 298.4)
```

Top drift volume (x > 0, anodes 4–7):

```
boundary_xy (x, y): (339.9, -336.4), (339.9, -336.4), (3.0, -333.9), (3.0, 333.6), (126.8, 336.4), (339.9, 336.4)
boundary_xz (x, z): (339.9, 0.8), (339.9, 0.8), (3.0, 4.6), (3.0, 287.4), (164.6, 298.4), (339.9, 298.4)
```

(`figs/41_polygons_M1_d50.json` carries the same numbers with the per-wall
M1 parameters.) In WCT this is two `PolyFiducial` slabs per volume — one with
`axis: 2` whose corners are these (x, y) pairs, one with `axis: 1` whose
corners are the (z, x) pairs — AND-ed through `CompositeFiducial{logic: 'and'}`
(doc 33 §3.1 / the `aux/test/doctest_fiducials.cxx` example). The AND must be
the composite's: slabs inside one `PolyFiducial` OR together.

---

## 8. What this means for the flat 15 cm inset, and what to do next

### 8.1 The flat 15 cm inset is wrong in both directions

PDVD's clustering FV and, since doc 35, the tagger margins carry a uniform
15 cm allowance in y and z at every x. Against the measurement:

- **Too little where it matters most**: at the cathode face the z+ wall in the
  bottom volume is inset 17.7 ± 1.3 cm; 15 cm leaves ~3 cm of apparently-inside
  points that are really wall-crossing charge. Within |x| < 40 cm this holds on
  z+ (bottom) only; every other wall is ≤ 15 cm everywhere.
- **Far too much everywhere else**: for |x| > 200 cm — 40 % of each drift — and
  on y+ at every x, the measured inset is 0–3 cm. The flat allowance removes
  14 % of the transverse area along the whole drift (2·15 of 672.8 cm in y,
  2·15 of 297.6 cm in z); the M1 surface removes ~12 % at the cathode face,
  falling linearly to zero by |x| ≈ 200 cm (~6 % in the top volume) — roughly
  2–3.5 % averaged over the drift, a factor 4–7 less volume than today.

One caveat on *which* fiducial. Doc 33 §3.2 records that the 15 cm clustering
inset also sets the operating point of `JudgeSeparateDec_2` ("on the surface =
within ~15 cm of a wall"). That is a tuning, not a geometry statement; a
curved surface should go first into the containment / tagger volumes (V5 and
the `FiducialUtils` walks), and replace the clustering `FV_*` metadata only
with the doc-96/97 separation cases re-checked.

### 8.2 Caveats

- **No MC control exists.** `pdvd_sim/` has no Q/L output; the only control is
  the anode-side bins, which pass. A no-space-charge simulation through the
  same chain would show whether the 1 cm top-volume offsets are geometry.
- **Data-only, one running period** (runs 39252/39253/39349). Space charge
  varies with purity, flow and HV history; the map should be re-measured per
  period before it is trusted as a constant.
- **The wrong-t0 cut removes 16 % of the points.** They are long tracks; if
  their t0 is wrong by a flash mis-match their x is wrong and their inclusion
  would smear the profile in x, not shift it — but 489 tracks is a large
  population to lose and a Q/L-quality question in its own right.
- **Corner dependence** at the ~10 cm, 2σ level in two CRU quarters (§6.2).
- **The drift direction is not mapped.** 16,753 points sit more than 3 cm past
  the cathode face after t0 correction (4,032 beyond 20 cm): the longitudinal
  counterpart of this effect, and the cause of the qlmatch/24 ±2 cm band.

### 8.3 Recommended next step

Wire the eight M1 profiles as a default-OFF `PolyFiducial` + `CompositeFiducial`
in `cfg/pgrapher/experiment/protodunevd/` (a `curved_fiducial.jsonnet` sibling
of `crp_gap_fiducial.jsonnet`, imported by nothing until a knob selects it for
the tagger `BoxFiducial`'s role), byte-identical off, and A/B it on the same
120-event arm: the numbers that must move are the fully-contained / TGM / STM
counts of doc 35 §5.2, and the direction is predictable — fewer false "fully
contained" verdicts near the cathode z walls, more on the anode side. In
parallel, image and Q/L-match the ~200 further PDVD data events already on
disk under `input_data/run0393*` and `run04*`: it halves every error here and
settles the corner quarters, at which point the ten-slab MicroBooNE form is
measurable too.

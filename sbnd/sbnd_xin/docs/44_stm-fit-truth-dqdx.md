# 44. True dQ/dx for the STM fit: `true_dx`, not `rec_dx`

Owner report: *"the truth information has some quite fluctuation, I feel that
the treatment of the truth is not optimized"* — the blue truth curve of doc 42
§7 (`showcase-stmfit-mc-evt18/dqdx_mc_evt18_blk150.png`) scatters from 0 to
396 ke/cm on a track whose true dQ/dx is a flat 50 ke/cm MIP, i.e. the
*reference* was noisier than the fit it exists to validate.

This is a **bug in the truth denominator**, not truth statistics.  Fixed; no
reconstruction code is touched.

> **Extended by doc 46** (`46_stm-fit-deltarays-and-gui.md`): the truth this doc
> smooths was also *incomplete* — it grouped deposits by `origTrackID`, which
> drops every delta ray (22.6 % of the charge).  Doc 46 adds them as a separate
> `true_dQ_sec` branch and explains why summing them into `true_dQ` would undo
> this doc's result (rms/median 0.087 → 0.681, mean fitted/true 1.036 → 0.852).
> Every number below is unchanged and was re-verified array-identical after that
> change.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
D=showcase-stmfit-mc-evt18

# 1. truth dump -- now also writes the per-deposit path length as branch `dx`
root -l -b -q "dump_truth_sed.C(\"input_files/input-10evt-mc/2025f-mc.root\",228,18,
                                \"work-mcsim-stmon/nusel_evt18/tracking-stm.root\",
                                150, 5.0, \"$D/truth-evt18-blk150.root\")"

# 2. converter -- now also writes T_rec branch `true_dx`
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-mcsim-stmon/nusel_evt18/tracking-stm.root -tT_rec_charge \
  -a$D/truth-evt18-blk150.root -nT -o$D/track_com_18.root -f1

# 3. numbers + plot
python3 scripts/analysis/stm/stmfit_mc_compare.py -f $D/track_com_18.root -b 150 \
    -o $D/dqdx_mc_evt18_blk150.png

# 4. GUI (headless recipe of doc 43; block 150 is cluster index 1)
cd /nfs/data/1/xqian/toolkit-dev/Magnify-tracking-SBND/scripts
A=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/$D
xvfb-run -a -s "-screen 0 1920x1080x24" root -l -q loadClasses.C \
  "/home/xqian/tmp/drive2.C(\"$A/track_com_18.root\",\"$A/magnify_mc_evt18_blk150.png\",1)"
# drive2.C = drive.C of doc 43 plus gc->SetCurrentCluster(clus);
#            gc->data->DrawNewCluster(); before the SaveAs.
# magnify_dqdx_{recdx,truedx}.png are the top-left pad of that canvas, cropped,
# rendered from the pre-fix and post-fix track_com_18.root respectively.

# generality check quoted in §5 (different event, different muon):
root -l -b -q 'dump_truth_sed.C("input_files/input-10evt-mc/2025f-mc.root",228,2,
        "work-mcsim-stmon/nusel_evt2/tracking-stm.root",110,5.0,"/home/xqian/tmp/gen/truth-evt2-blk110.root")'
wire-cell-sbnd-magnify-tracking-convert -bwork-mcsim-stmon/nusel_evt2/tracking-stm.root \
  -tT_rec_charge -a/home/xqian/tmp/gen/truth-evt2-blk110.root -nT -o/home/xqian/tmp/gen/tc_2.root -f1
python3 scripts/analysis/stm/stmfit_mc_compare.py -f /home/xqian/tmp/gen/tc_2.root -b 110
```

Products **regenerated in place** in `showcase-stmfit-mc-evt18/` (owner
decision, 2026-07-25): one showcase directory, so opening
`track_com_18.root` in the GUI shows the corrected truth.  `truth-evt18-blk150.root`
(now carries the `dx` branch), `track_com_18.root` (now carries `true_dx`),
`dqdx_mc_evt18_blk150.png` and `magnify_mc_evt18_blk150.png` are overwritten;
the pre-fix numbers survive only as text, in doc 42 §7 and in the tables below.
`magnify_dqdx_recdx.png` / `magnify_dqdx_truedx.png` are the before/after
renders of the GUI dQ/dx pad, kept in the same directory as the visual
evidence.  `upload_mc18.zip` is untouched and still valid: `scripts/bee/make_stmfit_bee.py`
reads only the work dirs, never the truth or `track_com` file, so the Bee set
**d11b087c… / 1b654c7c…** does not depend on this fix.

## Symptom

evt 18 block 150, the through-going 220 cm muon of doc 42 §7.  Per fitted
point, true dQ/dx as published there vs. as computed here:

| | median [ke/cm] | rms/median | p5 – p95 | max |
|---|---|---|---|---|
| true, `true_dQ/rec_dx` (doc 42 §7) | 49.54 | **0.551** | 16.6 – 80.3 | 396.5 |
| true, `true_dQ/true_dx` (this doc) | 49.40 | **0.087** | 43.6 – 57.2 | 67.4 |
| fitted, `rec_dQ/rec_dx` (unchanged) | 49.59 | 0.230 | 39.0 – 73.0 | 112.9 |

The median is unchanged — this removes an artifact, it does not smooth away
physics.  The truth is now 2.6× *quieter* than the fit, which is the right
ordering: G4 charge over 0.6 cm has a Landau width of some 10 %, and the fit
is a 2-D-to-3-D charge solve on ~4 wires.

Side effects of the same fix, all visible in the GUI pad
(`magnify_dqdx_recdx.png` before, `magnify_dqdx_truedx.png` after):

- The **fake Bragg peak** is gone.  The truth beyond the fit's end piled onto
  the endpoints: 186.5 and 396.5 ke/cm, now 51.4 and 48.8.  The 5 cm `-R` cut
  in `scripts/root/dump_truth_sed.C` and `--edge 1` in `scripts/analysis/stm/stmfit_mc_compare.py` existed only
  to hide this; `--edge` now defaults to 0 on files that carry `true_dx`.
  The `-R` cut is **still wanted**, for a different reason: dropping it does
  not bring the fake peak back, but it puts the whole far part of the muon
  into the end cells, i.e. it maximises the `true_dx ≫ rec_dx` case discussed
  below.  It bounds the end-cell length; it no longer hides an artifact.
- Points to which no truth was assigned no longer read as **0 ke/cm**.  There
  are 4 of them in this block; in evt 2 block 110 there are 181 of 378, and
  their zeros dragged the published-style truth median from 50.1 down to 28.3.

## Root cause

`wire-cell-sbnd-magnify-tracking-convert -f1` assigns every truth deposit to
the **nearest fitted point** and sums its charge into `true_dQ` there.  Every
consumer then divided that by `rec_dx`.

`rec_dx` is the path length of the **fitted** cell (median 0.62 cm here, 5–95 %
spread 0.52–0.77).  The assignment cell is something else entirely: a 3-D
Voronoi region of the fitted point cloud.  Measured on this block, the length
of *true* track falling inside one such cell runs **0.2 – 5.3 cm**:

| | value |
|---|---|
| truth deposits per fitted point (should be ~21) | p5 8, median 21, p95 35, max 177 |
| `true_dx / rec_dx` | median 1.00, p90 1.37, max 8.12 |
| Σ `true_dx` | 223.18 cm = the true path, vs Σ `rec_dx` 218.81 cm |

So `true_dQ/rec_dx` is a cell-length fluctuation of up to a factor 8 reported
as a dQ/dx fluctuation.  The fix is to divide by the length that actually
produced the numerator.

## Why it hid

1. **The prototype does the same thing and gets away with it.**
   `prototype_base/mcs/apps/wire-cell-track-com.cxx` (lines 261-267) is the
   ancestor of this accumulation loop and also stores only `dQ_tru`, leaving
   the consumer to divide by `nq`/`dx`.  It gets away with it because its
   truth is *pre-resampled*: `MCSSim.cxx:79-150` (`MCSTrackSim`, used by both
   `wire-cell-mcs-sim` and `wire-cell-le-sim`) walks the particle in fixed
   `step_size` increments (default 0.1 cm) and stores the charge of that whole
   step, so its truth points are coarse and comparable to the fitted spacing.
   `scripts/root/dump_truth_sed.C` feeds the same machine raw G4 `SimEnergyDeposit`
   micro-steps at 0.03 cm — 21 per fitted point — which is what makes the
   cell-occupancy variance visible.  `wire-cell-display-convert.cxx`, the
   third writer of that tree format, only repackages an existing truth tree.
   *There is no better prototype treatment to port; the prototype avoided the
   question rather than answering it.*  Resampling our truth to the prototype's
   format would not fix it either: coarser truth points make empty cells more
   common, not fewer, and the denominator would still be wrong.
2. **The showcase track is flat.** On a MIP the artifact is symmetric noise
   about the right median, so every *aggregate* in doc 42 §7 (median 49.5 vs
   49.5, integral ratio 1.043, the 10-bin profile) was already correct.  Only
   the per-point scatter and the endpoints were wrong, and the plot's running
   median hid even that.
3. **The fitted path zig-zags at the cell scale.** Fitted point-to-point kink
   angle: median 4.3°, p90 17°, p99 47°, max 129°.  That is why the cells are
   so uneven — and also why projecting the truth onto the fitted polyline's
   arc length (the obvious alternative fix, tried first) buys nothing: it
   changed rms/median from 0.551 to 0.550, because the wiggle is in the
   polyline itself.  *Pre-existing fitter observation, reported not changed.*

## Fix

Purely additive; `true_dQ` is bit-for-bit unchanged.

1. **`sbnd_xin/scripts/root/dump_truth_sed.C`** — new truth-tree branch `dx`, the G4 step
   length `|endPos − startPos|` per deposit.  Taken from the endpoints rather
   than from consecutive-point spacing so it stays exact when the deposit list
   is not one ordered chain.  The summary line now also prints the true path
   length and mean dQ/dx (223.18 cm, 49.83 ke/cm here).
2. **`root/apps/wire-cell-sbnd-magnify-tracking-convert.cxx`** — reads that
   optional `dx`, accumulates it into a new `T_rec` branch **`true_dx`**
   alongside `true_dQ`, republishes it in `T_true`, and logs the assignment
   (`truth path assigned: … cm over N fitted points … n points got no truth`).
   When the truth tree has no `dx` (the prototype's `mcs-tracks.root`) the
   lengths are reconstructed from consecutive-point spacing by the midpoint
   rule — valid there because `MCSTrackSim` emits one ordered chain.  On this
   block the two agree to 0.05 % in the sum (223.18 vs 223.30 cm) and to three
   decimals in every per-point statistic.
   The **uBooNE** converter is untouched (M10).
3. **`Magnify-tracking-SBND/event/Data.{h,cc}`** — `DrawDQDX` uses `true_dx`
   when the branch exists, else falls back to `rec_dx` with a warning.  Points
   with `true_dx == 0` are omitted from the graph rather than plotted as zero
   (a zero draws a spike to the axis — same class of defect as the doc-43
   cursor bug), and a block with no paired truth at all draws nothing instead
   of earning a `TGraphPainter: illegal number of points (0)` on every redraw.
4. **`sbnd_xin/scripts/analysis/stm/stmfit_mc_compare.py`** — same preference and same skip;
   `--edge` defaults to 0 with `true_dx` and 1 without; per-bin and integral
   ratios are now length-normalised (ratio of mean dQ/dx, since the two
   denominators are different lengths); new `--max-dx-ratio` cut and a
   `true_dx/rec_dx` distribution line in the summary.

**Multi-block files get more robust, not less.** `pcloud1` holds the fitted
points of *every* block, so two nearly-coincident blocks — the forward and
backward passes of one cluster, `cid*10+pass`, a path doc 43 notes is still
untested — split the truth arbitrarily between them.  Under `/rec_dx` each
would then read roughly half the true dQ/dx.  Under `/true_dx` the numerator
and denominator halve together and the quotient survives.

**`true_dx` is exported, not internal, on purpose.** A cell with
`true_dx ≫ rec_dx` averages the truth over more track than the fitted point
represents.  That is harmless on this MIP, but on a **stopping** muon — the
whole point of STM — a 5 cm end cell would average the Bragg peak away.
`--max-dx-ratio` lets a future stopping-track study cut on it.  If that turns
out to be too blunt, the follow-up is to bin the truth in a fixed ±`rec_dx`/2
window along the *truth's* own arc length instead of using the Voronoi cell;
not built now, since no block in this sample needs it.

## Verification

Gates (this is an analysis converter, not the reconstruction chain — no
jsonnet knob and no `abtest`/`qlport` gate consumes it; confirmed by grep,
the gates use the uBooNE app and the visitor's `T_rec_charge`):

- **Freshness (M1):** `local/bin/wire-cell-sbnd-magnify-tracking-convert`
  2026-07-25 07:08:49.645 vs source 07:08:31.481 (the shipped binary; an
  earlier 07:00 pair built the same logic before a comment-only edit).
- **The shipped binary reproduces the committed products:** re-running it on
  `$D/truth-evt18-blk150.root` gives a `T_rec` array-identical to
  `showcase-stmfit-mc-evt18/track_com_18.root` on every branch.
- **All pre-existing output unchanged.** Re-ran the converter on
  `work-mcsim-stmon/nusel_evt18/tracking-stm.root` with the *old* truth file
  (no `dx` branch, exercising the fallback) and compared every branch of
  `T_rec` against the committed `showcase-stmfit-mc-evt18/track_com_18.root`:
  **31 of 31 branches array-identical, 0 differences**; the only new branch is
  `true_dx`.  `true_dQ` is also array-identical when the *new* truth file is
  used, i.e. adding the `dx` branch does not perturb the pairing.
- **Legacy truth files still work:** the no-`dx` run logs *"truth tree has no
  dx branch: path lengths taken from consecutive-point spacing"* and produces
  Σ`true_dx` = 223.304 cm against 223.182 cm from the explicit branch.
- **Both GUI paths render** (headless, rc=0): new file →
  *"true dQ/dx: 4 of 346 points have no truth assigned and are not drawn"*;
  the pre-fix file → *"no true_dx branch: … falls back to true_dQ/rec_dx"*.
  Block 80, whose truth file describes a different track, draws no truth curve
  and no ROOT error.
- **`scripts/root/dump_truth_sed.C`'s existing branches unchanged:** `x` and `Q` of the
  regenerated `truth-evt18-blk150.root` are array-identical to the doc-42 §7
  record.
- **Both `scripts/analysis/stm/stmfit_mc_compare.py` paths exercised.**  On the pre-fix
  `showcase-stmfit-mc-evt18/track_com_18.root` it prints the fallback NOTE,
  defaults `--edge` to 1, and reproduces doc 42 §7 exactly (true median
  49.5 ke/cm, rms/median 0.376 over the 344 core points, mean-dQ/dx ratio
  **1.043** — the published value).  The two lines that would only restate
  `rec_dx` (`true path assigned`, `true_dx / rec_dx`) are suppressed in that
  mode, since a file without the branch cannot say how much truth is missing.
  `--max-dx-ratio 2` on the fixed file drops the 8 cells wider than twice
  `rec_dx` and tightens the truth p5–p95 from 43.6–57.2 to 43.5–57.1.
- No `wcdoctest` applies (no library changed); `./wcb build` + `install` rc=0.

### Re-measured numbers for doc 42 §7

Same block, same fit, truth denominator fixed.  `--edge 0` now, so all 346
points are in (doc 42 §7 dropped the first and last).

| L [cm] | n | fitted [ke/cm] | true [ke/cm] | fitted/true | ⟨x⟩ [cm] |
|---|---|---|---|---|---|
| 0 – 22 | 37 | 48.9 | 49.4 | 1.062 | −16.4 |
| 22 – 44 | 34 | 48.9 | 48.9 | 0.970 | 3.5 |
| 44 – 66 | 35 | 51.7 | 48.9 | 1.032 | 23.4 |
| 66 – 88 | 34 | 55.8 | 50.5 | 1.100 | 43.1 |
| 88 – 110 | 35 | 49.6 | 48.2 | 1.039 | 63.5 |
| 110 – 132 | 34 | 52.8 | 49.2 | 1.184 | 83.4 |
| 132 – 154 | 33 | 50.2 | 49.5 | 1.010 | 103.0 |
| 154 – 176 | 33 | 48.8 | 50.2 | 1.075 | 123.0 |
| 176 – 198 | 33 | 46.6 | 50.3 | 0.972 | 142.1 |
| 198 – 220 | 33 | 44.5 | 49.6 | 0.925 | 161.8 |

The true column's spread across bins collapses from 5.8 ke/cm (47.5 – 53.3 in
doc 42 §7) to **2.3 ke/cm** (48.2 – 50.5): the bin-to-bin wobble the old
column showed was cell-occupancy noise, not the muon.  Everything doc 42 §7
concluded survives:

- fitted median **49.6** vs true **49.4** ke/cm; mean-dQ/dx ratio over the
  block **1.036** (was 1.043 on 344 points).
- **No drift trend.** 40 cm bins: 0.878 / 0.986 / 1.104 / 1.092 / 1.022,
  slope +0.00097 per cm → +0.165 over the 171 cm sampled, against the
  0.833 far/near an uncorrected τ = 6 ms would give.  Conclusion unchanged
  (attenuation corrected upstream, or the sim lifetime is effectively
  infinite; one track cannot separate the two).

## 5. Generality

evt 2 block 110 (a different event, µ+, `origTrackID` 20000140), an
independent check that the improvement is not tuned to one track:

| | median [ke/cm] | rms/median | p5 – p95 |
|---|---|---|---|
| true, `/rec_dx` | 28.33 | 1.346 | 0.0 – 63.8 |
| true, `/true_dx` | **50.13** | **0.093** | 44.8 – 59.3 |
| fitted | 44.56 | 0.309 | 25.0 – 72.0 |

Here the old form was **biased**, not merely noisy: 181 of the 378 fitted
points got no truth (the fit runs 235 cm while only 133.6 cm of the muon lies
within 5 cm of it — coverage 48.9 %, pairing p90 33 cm), and their zeros put
the truth median at 28.3.  `true_dx == 0` now excludes them and the median
lands on the MIP.

Two things this exposes, both **findings, not fixed here**:

- That block's fit **wanders off its own muon** for half its length — same
  family as doc 42 §7's finding that the accepted-STM blocks follow no true
  particle.  Its fitted/true of 0.901 is therefore not a calibration number.
- evt 9 block 110 pairs at median **11.3 cm** and elects a 9-deposit shower
  electron; `scripts/analysis/stm/stmfit_mc_compare.py` refuses to quote (exit 1) and
  `scripts/root/dump_truth_sed.C` writes 0 truth points.  The guards work; the block is
  unusable for truth comparison.  Of the 10-event MC sample only 4 events have
  an STM fit at all (evt 2 blk 110, evt 9 blk 110, evt 18 blk 80 + 150,
  evt 42 blk 60), and only evt 18 blk 150 and evt 2 blk 110 pair well enough
  to quote — the sample is too small for a calibration statement, which is
  the phase-4 item doc 40 still lists.

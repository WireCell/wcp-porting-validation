# 46. Delta rays in the truth, and two Magnify display defects

Owner report on evt18 cluster index 1 / cluster id 15 (block 150), the
through-going muon of docs 42 §7 and 44:

1. *"The distance in difference to MC does not seem to be correct in the
   Magnify-tracking code."*
2. *"It seems that this track was crossing cathode, the channels are separated,
   there is a line connecting the two parts in the 2D projection view, which is
   somehow weird."*
3. *"For the reco dQ/dx, sometimes we can see a bump, say distance from start at
   37 cm. But the truth is relatively flat. Sometimes the reason there is a high
   local dQ/dx is due to a delta ray... I wonder why the truth information does
   not have it, is it related to maybe it only captures the muon not the other
   particle."*

All three confirmed. (1) is a display bug, the value was always right. (2) is a
real artifact of the SBND two-TPC channel axis. (3) is exactly right — the
dumper elected one `origTrackID` and every delta ray carries its own, so 22.6 %
of the charge was invisible. **But the specific L≈37 cm bump is not a delta
ray** — see §3.3.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
D=showcase-stmfit-mc-evt18
root -l -b -q "scripts/root/dump_truth_sed.C(\"input_files/input-10evt-mc/2025f-mc.root\",228,18,
                                \"work-mcsim-stmon/nusel_evt18/tracking-stm.root\",
                                150, 5.0, \"$D/truth-evt18-blk150.root\")"
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-mcsim-stmon/nusel_evt18/tracking-stm.root -tT_rec_charge \
  -a$D/truth-evt18-blk150.root -nT -o$D/track_com_18.root -f1
python3 scripts/analysis/stm/stmfit_mc_compare.py -f $D/track_com_18.root -b 150 \
    -o $D/dqdx_mc_evt18_blk150.png
# GUI (headless recipe of doc 43/44; block 150 is cluster index 1):
A=$PWD/$D
cd /nfs/data/1/xqian/toolkit-dev/Magnify-tracking-SBND/scripts
xvfb-run -a -s "-screen 0 1920x1080x24" root -l -q loadClasses.C \
  "/home/xqian/tmp/drive2.C(\"$A/track_com_18.root\",\"$A/magnify_mc_evt18_blk150.png\",1)"
# add `scripts/root/dump_truth_sed.C(..., 5.0, "out.root", false)` for the pre-doc-46
# primary-only truth.
```

Products regenerated in place in `showcase-stmfit-mc-evt18/` again (same owner
decision as doc 44).  New evidence crops: `magnify_dqdx_deltaray.png`,
`magnify_proj_cathode_fixed.png`.

## 1. "Difference to MC": the distance was drawn on the angle's axis

**Symptom.** The red "distance" curve appears only past L ≈ 160 cm and never
exceeds 0.45, on a track whose fitted-to-truth distance is 1.78 cm median.

**Root cause — display, not value.** `com_dis` is correct: the converter's and
`scripts/root/dump_truth_sed.C`'s own printouts agree, median **1.78 cm**, p90 3.45, max
3.94.  `Data::DrawMCCompare` builds the pad frame from `com_dtheta`, which is in
**radians**, and hard-codes `GetYaxis()->SetRangeUser(0, 0.6)` — a uBooNE-era
range.  `com_dis`, in **centimetres**, is then drawn `LPsame` on that frame.
Only the 23.7 % of points with `com_dis < 0.6` land inside the window; the other
**76 % is clipped off the top and invisible**, and the visible remnant past
L ≈ 160 cm is just where the offset happens to fall below 0.6 cm.

**Why it hid.** In uBooNE the truth and the reco sit in the same frame, so
`com_dis` is sub-cm and fits under 0.6 by accident.  On SBND the truth is the
`priorSCE` instance while the reco is at the space-charge-distorted position, so
`com_dis` is a genuine 1–4 cm — the units clash only became visible here.

**Fix.** Frame from the angle (auto-ranged, floor 0.6), y-title `angle [rad]`;
draw the distance scaled onto that frame and give it its own red `TGaxis` on the
right in cm — which is what the commented-out `TGaxis` block in that function
always intended.  Legend now says "distance (right axis)".

**What it reveals.** The distance is now readable and physical: ~3.3 cm at the
start, a step at L ≈ 28 cm (the cathode crossing), then a monotonic fall to
~0.2 cm by the far anode.  That is a drift-dependent offset, consistent with
doc 42 §7's blind x-shift scan preferring +3 cm, and it was invisible before.
Not corrected here.

## 2. The cathode-crossing connector in the 2D projections

**Symptom.** A long almost-horizontal line across the channel axis joining the
track's two halves, in all six projection pads.

**Root cause.** The projection pads plot **channel**, and SBND concatenates the
two TPCs along it (per plane: TPC0 then TPC1 — U `[0,1984)`+`[1984,3968)`,
V and W likewise; `Data.cc` 3968/3968/3340).  At the cathode two consecutive
fitted points are adjacent in space but thousands of channels apart, while their
time slice barely moves.  On this block that is point 47 → 48:
**u 673 → 3755 at t 551 → 550**, `x` −0.7 → +0.9 cm.  `DrawSubclusters` draws one
`TGraph` per sub-cluster with the `"L"` option, so ROOT joins them.  The jump is
real; the connector is not — there is no track there, the two halves are on
different wires.  Impossible in single-TPC uBooNE, so the inherited code never
had to consider it.

**Fix.** Break the projection polyline wherever the drift coordinate changes
sign — this codebase's definition of the TPC split (TPC0 `x<0`, TPC1 `x>0`), so
no per-plane channel constants are needed.  One `TGraph` per
(sub-cluster, TPC run), with a new `seg_sub[]` mapping each graph back to its
sub-cluster so colours, marker styles and the legend still follow the
sub-cluster rather than the graph index (the sentinel-entry-at-index-0
convention is preserved).  The GUI now logs the break:
`sub-cluster 150: projection polyline broken at point 47 (cathode crossing,
u 673.429 -> 3755.47)`.

The **3D pad is deliberately left alone**: `x` is continuous across the cathode
there, so one `TPolyLine3D` per sub-cluster is correct.  `DrawAllClusters` needs
no fix either — it draws with `"Psame"`, markers only, so it never had a
connector.

## 3. Delta rays: the truth really did only capture the muon

### 3.1 Root cause

`sim::SimEnergyDeposit` carries **two** ids, and LArSoft puts the parentage in
the sign of one of them:

| member | meaning (from the file's own StreamerInfo v20) |
|---|---|
| `origTrackID` | "complementary simulation track id, kept **true to G4** even for shower secondaries" — a delta ray carries its **own** id |
| `trackID` | `> 0`: that particle stepped. `< 0`: a **secondary of `|trackID|`** stepped (the daughters largeant did not keep as separate MCParticles — cf. the `simb::MCParticles_largeantdropped` product) |

`scripts/root/dump_truth_sed.C` grouped by `origTrackID`, so it kept only the muon's own
steps.  Measured within 10 cm of this block: 1703 of 9346 deposits, carrying
**18.3 % of the charge**, belong to 157 other ids — and 1693 of those 1703 carry
`trackID = −20000167`, i.e. they are secondaries of exactly this muon.  Their
pdg is **±11 without exception**: delta rays and their sub-showers, each a
localised 1–3 cm blob 0.3–2.4 cm off the track, carrying 0.1–0.27 Me⁻.

So the rule is exact and needs no MCParticle tree (which cannot be read here
anyway — the emulated-class read of `simb::MCParticle` segfaults on
`ftrajectory`): **a deposit belongs to particle P iff `|trackID| == P`.**  A
geometric alternative was prototyped first (transitive closure on birth points,
0.3 cm, 89 particles, 98.3 % of the charge) and discarded once `trackID` was
found — the sign convention is exact and free.

### 3.2 Fix: two charge channels, one denominator

Merging the delta rays into `true_dQ` would have been wrong in the other
direction, and the numbers say so loudly. Same block, all 342 usable points:

| true dQ/dx | median | rms/median | p5 – p95 | max | mean fitted/true |
|---|---|---|---|---|---|
| parent only (doc 44) | 49.40 | **0.087** | 43.6 – 57.2 | 67.4 | **1.036** |
| parent + delta rays, summed | 51.18 | **0.681** | 43.7 – 119.9 | **373.4** | **0.852** |

The truth puts a delta ray's charge on the **one** fitted point nearest it; the
reco spreads the same charge over several wires and several points, and collects
only part of it.  Summing therefore trades doc 44's artifact for a new one.

So the two are kept apart:

- `scripts/root/dump_truth_sed.C` groups by `|trackID|`, keeps the whole family within `-R`,
  and tags each row `sec = 0/1`.  **Secondary rows get `dx = 0`**: dQ/dx is
  charge per unit length *of the parent track*, and a delta ray's own path is
  not parent path, so it must contribute charge without contributing length.
- The converter accumulates `true_dQ` (primary) and the new **`true_dQ_sec`**
  (secondary) over the one denominator `true_dx`.  Restricted true dQ/dx is
  `true_dQ/true_dx`; all charge available near the trajectory is
  `(true_dQ+true_dQ_sec)/true_dx`.
- The GUI draws the second curve dashed magenta over the red restricted one; the
  compare script plots it and quotes both.

One further correction this forced: the **pairing** reference (`com_dis`,
`com_dtheta`, `true_x/y/z`, `stat_beg_dis`, `stat_end_dis`) is built from
**primary rows only**.  With delta rays in that cloud, a fitted point 1.8 cm
from the muon reported 0.3 cm from a delta ray and understated the trajectory
error — `com_dis`'s median fell without the fit getting any better.  Excluding
them restores it exactly.

### 3.3 What this says about the owner's bump — and what it does not

New numbers for the block:

| | value |
|---|---|
| secondary charge | 2.4026e6 e⁻ = **21.6 %** of the primary charge, on 89 of 342 points (26 %) |
| mean true dQ/dx | **49.83** ke/cm restricted vs **60.59** ke/cm with secondaries |
| mean fitted | 51.63 ke/cm ⇒ the fit recovers **17 %** of the delta-ray charge |
| coverage (elected particle owns the nearest deposit) | 88 % → **100 %** of fitted points |
| charge purity within 5 cm | 81.6 % → **100 %** |

**Association, yes:** of the 35 points with fitted dQ/dx above 1.3× the median,
**74 % have delta-ray charge nearby, against a 26 % baseline.**  The correlation
between the fitted excess over the restricted truth and the delta-ray dQ/dx is
+0.362 point-to-point, peaking at **+0.419** when the delta-ray charge is
smeared over ±1 fitted point (±0.6 cm) — consistent with the reco spreading it.

**Causation for L ≈ 37 cm, no.**  The bump is points 60–61 (L = 37.0 and
37.6 cm, fitted 68.4 and 64.6 ke/cm against a 45 ke/cm restricted truth), and
those two points have **no secondary charge at all out to 10 cm** — every
deposit near them is a pure muon step. The nearest delta ray is point 58
(L = 35.7 cm, 37.7 ke), and point 58's own fitted excess is **−3.3 ke/cm**, i.e.
no excess where the truth excess is.  The L≈37 bump is therefore a **reco-side
feature** (the charge solve), not a delta ray, and the truth is correctly flat
there.  Note also that the fit is ~3.8 cm off the muon in that stretch, against
the 1.78 cm median.  **Reported, not fixed.**

## Verification

- **Freshness (M1):** `local/bin/wire-cell-sbnd-magnify-tracking-convert`
  built 07:43:30 / re-linked after the pairing-cloud fix, source 07:42:30.
  Installed by copying the single binary rather than `wcb install`, because a
  **concurrent session** (`claude-1109`, the `-unmerge`/doc-45 work) had
  uncommitted `clus/` changes in this tree and a full install would have pushed
  its WIP into `local/lib`.  That session's `wcb install` also `rm -f`'d
  `libWireCellClus.so` mid-link and broke one build here; waited for it rather
  than racing.
- **Everything doc 44 established is preserved exactly.** Re-running the
  converter on the new family truth file gives a `T_rec` in which **all 32
  pre-existing branches are array-identical** to the committed doc-44 file —
  including `true_dQ`, `true_dx` **and** `com_dis` — with `true_dQ_sec` the only
  addition.  The primary charge within 5 cm is bit-identical (1.1120e7 e⁻) and
  the true path is unchanged (223.18 cm), because secondaries add no length.
- **The election does not move.** Grouping by `|trackID|` instead of
  `origTrackID` changes who votes, so this was checked explicitly: evt18 blk150
  elects 20000167 either way (coverage 88 % → 99 %), evt2 blk110 elects 20000140
  either way (49 % → 55 %), and the primary-only charge is identical in both.
- **Legacy paths still work:** a truth file without `sec` is treated as
  all-primary (the pre-doc-46 meaning of `true_dQ`); without `true_dQ_sec` the
  GUI and script draw and quote only the restricted curve;
  `with_secondaries = false` reproduces the old dumper behaviour.
- **GUI renders, rc=0**, logging all three fixes: the polyline break at point
  47, `secondaries (delta rays) add 2402.64 ke`, and `4 of 346 points have no
  truth assigned and are not drawn`.  No `TGraphPainter` errors.
- No `wcdoctest` applies (no library changed); `./wcb build` rc=0.

## Follow-ups (not done)

- The L≈37 cm bump is unexplained on the reco side, in a stretch where the fit
  is 3.8 cm off the muon.
- `-R 5 cm` now drops 11.2 % of the family charge (was 10.6 % of the muon's).
  With `true_dx` the endpoint artifact is gone, so a wider `R` is safe; it still
  bounds the end-cell length (doc 44).
- The 17 % delta-ray recovery fraction is a one-track number and a candidate
  calibration handle — it is the difference between the restricted truth and
  what the charge solve actually collects. It needs the larger sample doc 40's
  phase 4 still lists (only 4 of the 10 MC events have an STM fit).
- doc 42 §7.2's "charge purity 81.58 %" line now has a name: it was the delta-ray
  family, and the number is 100 % once they are counted in.

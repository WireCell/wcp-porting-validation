# doc pdhd/24 — testing the wire-crossing origin of the periodic dQ/dx wave

Answers the open item carried by `pdhd/docs/20_stm-michel-dqdx-readability.md` §8
("the periodic-wave/wire-crossing item stays open and separate") and
`pdhd/docs/03_check-stm-michel-pdhd.md` §9 item 2 (wire-parallel tracks with
dQ/dx oscillating 20–90 ke/cm) / item 5 ("the oscillations are trajectory, not
noise").

**Read-only.** No C++ changed, no jsonnet changed, no arm re-run, no label or
record touched. Nothing here is a knob and nothing here licenses one.

## Repro

Every number and figure below. The inputs are the scan payloads under
`pdhd/stm_michel_scan/prep-pdhd/` — **not tracked** (`*.json` is gitignored),
built by `prep_stm_michel_scan.py` for arm `d53h` and read here strictly in
place, never written. `D24_PREP` and `D24_FIGS` override the two paths; the
bare commands below assume the defaults.

```
cd wcp-porting-img/pdhd/docs/scripts
S=../../stm_michel_scan/prep-pdhd            # arm d53h, 303 payloads, unmodified
python3 d24_calib.py                                # what pu/pv/pw/pt are   (sec 1)
python3 d24_survey.py    $S ../figs/24_survey.tsv   # per-item geometry
python3 d24_arclen.py                               # L is arc length; wave size (sec 5)
python3 d24_fold.py      $S ../figs/24_fold.txt     # the sub-cell fold, all planes (sec 2)
python3 d24_fold2.py                                # disjoint-strata disentangling (sec 2)
python3 d24_amplitude.py                            # pooled variance removed      (sec 3a)
python3 d24_upper.py                                # per-item free-phase bound    (sec 3a)
python3 d24_closure.py                              # is the peak at one cell?     (sec 3b)
python3 d24_ratio.py                                # lambda / cm-per-cell         (sec 3c)
python3 d24_acfstrat.py                             # the same, as raw ACF         (sec 3c)
python3 d24_coherent.py                             # how much of it is a wave     (sec 4)
python3 d24_wobble.py                               # trajectory + segment cands   (sec 5, 6)
python3 d24_figs.py                                 # 24_fig1_*, 24_fig2_*
python3 d24_look.py 028084_8 120     ../figs/24_look_028084_8_c120.png
python3 d24_zoom.py 028084_8 120 20 80 ../figs/24_zoom_028084_8_c120.png
```

Outputs land in `pdhd/docs/figs/` as `24_*.txt`, `24_*.tsv`, `24_*.png` and are
committed with the doc. Runtime: `d24_upper.py` and `d24_closure.py` are the
slow ones (minutes); the rest are seconds. `numpy`, `scipy`, `matplotlib` only.

## 0. The one-line answer

**The wire-crossing mechanism is real and is measured here for the first time —
and it is not the origin of the wave.** dQ/dx is biased by where a trajectory
point sits inside a 2-D cell (high mid-cell, low at the boundary) on *both* the
collection-wire axis and the drift time-slice axis, each visible only on the
tracks that resolve that cell. But it carries ≤ 2 % of the dQ/dx residual
variance, the wave's dominant period is not one cell, and — decisively — the
wave's correlation length λ is **not proportional to cell size**: λ/cm-per-cell
falls from 2.12 to 0.14 across the population, where a geometric scale holds it
at 1.

## 1. What was measured, and on what

The 303 `smprep-*.json` payloads of the PDHD STM/Michel scan (arm `d53h`, the
`smx18`/`smx22` sample; 61 events over runs 028084 and 029107), of which 297
carry ≥ 30 chain points. Each carries,
per chain point: `x,y,z`, `q` = dQ/dx in e/cm, path length `L`, residual range
`rr`, and the point's coordinate on each readout axis — `pu`, `pv`, `pw`, `pt`.

**Calibration of those four coordinates, measured not assumed** (`d24_calib.py`,
two independent estimators):

| coord | single-axis regression on `028084_8/120` | pooled gradient magnitude | discontinuous steps | verdict |
|---|---|---|---|---|
| `pt` | **x alone**, 0.3152 cm/unit, \|r\| = 1.000000 | 0.3220 cm | 0.017 % | **drift time slice, established** |
| `pw` | **z alone**, 0.4793 cm/unit, \|r\| = 0.999992 | 0.4827 cm | 0.009 % | **collection wire, established** |
| `pv` | y, 0.7642 cm/unit, \|r\| = 0.999935 | 0.2104 cm | **2.40 %** | **not established** |
| `pu` | y, 0.8394 cm/unit, \|r\| = 0.999716 | 0.2132 cm | **1.23 %** | **not established** |

`pw` is the collection-wire index (wires along y) and `pt` the **imaging time
slice** (`ticks_per_slice` = 4, the cell's drift dimension); both come out the
same from two estimators that share no assumption.

`pu` and `pv` do **not**. A straight chain makes x, y, z collinear, so the
single-axis number for a coordinate that depends on y *and* z is an artifact —
and the pooled estimator fails on them too, because PDHD's induction planes are
**wrapped**: 1.2 % and 2.4 % of their steps are discontinuities, against 0.01–0.02 %
for `pw`/`pt`, and that tail drives the estimate to about half the true pitch.
An earlier draft of this table read the artifact as "0.4670 cm = the PDHD
induction pitch"; it is withdrawn. **The consequence is stated where it bites,
in §2.**

**Analysis window.** Plateau only (`rr > 30` cm — the Bragg rise is not a wave),
the longest contiguous stretch with no readout-unit jump (`|Δp| > 5` on any
plane, which is how an APA/cathode crossing shows up), ≥ 60 points.
**227 of those 297 chains qualify**, median stretch 145 cm / 243 points.

**Residual.** `q` divided by the muon Bragg reference (`dqdx_ref_pdhd.json`) at
that `rr`, then a quadratic in `L` removed, then normalised to unit variance per
item. Every number below is on that residual.

**Null, everywhere.** A per-item **circular shift** of the residual. It keeps
each item's autocorrelation exactly and breaks only its alignment with the
phase. This matters: a permutation null (which destroys the serial correlation)
made the first pass look more significant than it is, and the residual here is
84 % correlated (§4) — exactly the red-noise case a permutation null gets wrong.

**Independent check that the sample is the doc-20 sample:** median cm per W wire
here is **1.44**, against doc 20's 1.43 over the last 20 cm.

## 2. The wire-crossing effect IS there — on both cell axes

A track that runs along the collection-wire direction (y) is slow in z (few W
wires) *and* slow in x (few time slices), so the two "resolvable" strata could
be each other in disguise. They are not: the strata overlap on **3 items**, and
the cross-controls are dead.

| stratum | folded on | items | A1 [σ] | z vs circular-shift null | p |
|---|---|---|---|---|---|
| W cell resolved, slice not | **`pw`** | 27 | **0.1063** | **+5.68** | 0.0010 |
| W cell resolved, slice not | `pt` (control) | 27 | 0.0334 | +1.35 | 0.095 |
| slice resolved, W not | **`pt`** | 26 | **0.1026** | **+4.00** | 0.0010 |
| slice resolved, W not | `pw` (control) | 26 | 0.0050 | −1.47 | 0.957 |
| all 227 | `L`/0.4792 (meaningless phase) | 227 | 0.0052 | — | 0.154 |

The **disjoint-strata rows are the result**; the finding is "each axis folds
where the track resolves it", not "one axis folds and the other does not". The
pooled-over-227 numbers (`pt` z = +6.11, `pw` z = −0.14) are *not* evidence for
that asymmetry and are deliberately kept out of the table: pooled `pt` is
carried by its own 74 resolvable items, and pooled `pw` is null only because
most tracks cross W wires too fast to resolve at 0.6 cm point spacing. Quoting
them side by side would say something this data does not.

`pu` and `pv` fold on nothing in any stratum (p ≥ 0.22 throughout). **That is
the weakest evidence in this doc and is not claimed as a null result**: their
cells are usually crossed too fast to resolve at 0.6 cm point spacing *and*
`frac(pu)`, `frac(pv)` are not a clean cell phase at all, because the wrapped
induction planes make those coordinates discontinuous at the percent level (§1).
The two readings cannot be separated here. The `pw` and `pt` results do not
depend on them.

**Shape** (`figs/24_fig1_subcell_fold.png`): a single hump. dQ/dx reads **high near the
cell centre (phase ≈ 0.35–0.5) and low at the cell boundary**, amplitude ≈ ±0.1
of the item's own residual σ. The natural reading — charge straddling two cells
is under-assigned by the fit — is consistent with it, but is not tested here.

## 3. It is not the wave — three independent ways

**(a) Size.** Variance of the residual removed by a 12-bin cell-phase model,
against the same circular-shift null:

| | removed | null | **excess** |
|---|---|---|---|
| W wire, pooled phase | 0.0069 | 0.0017 | **0.0052** (p = 0.0033) |
| time slice, pooled phase | 0.0057 | 0.0021 | **0.0036** (p = 0.0066) |
| W wire, **free phase per item** (upper bound) | 0.0949 | 0.0785 | **0.0164** (p = 0.025) |
| time slice, free phase per item | 0.0723 | 0.0779 | −0.0056 (p = 0.79) |

The per-item free-phase fit is the generous bound: even letting every item
choose its own 12-bin cell profile, the lattice accounts for **≤ 1.6 %** of the
residual variance on the collection axis and **nothing beyond the pooled 0.36 %**
on the drift axis (the negative excess says the slice fold really does share one
phase across items, which is itself a small result).

**(b) Period.** If the wave were the fold, the Lomb-Scargle peak would sit at one
cell. It does not:

| stratum | median peak period | share in [0.8, 1.25] cells | null | p |
|---|---|---|---|---|
| W resolved (22 items ≥ 3 cells crossed) | 1.83 cells | 0.18 | 0.127 ± 0.068 | 0.31 |
| slice resolved (18 items) | 2.11 cells | 0.11 | 0.110 ± 0.069 | 0.64 |

The **period-2 (alternating-cell) mode** — the least-constrained mode of a
degenerate lattice inverse, and the reading both medians above hint at — was
*not* separately tested. It would not change the conclusion: §3a's variance
budget bounds **any** lattice mode, period-1 or period-2, at ≤ 1.6 % of the
residual, because every such mode lives in the same residual that budget is
computed on. A positive period-2 result would sharpen the description of a small
effect, not promote it to the wave's origin.

**(c) Scaling — the decisive one** (`figs/24_fig2_correlation_length.png`). A lattice
origin makes the wave's correlation length λ proportional to cm-per-cell, i.e.
**λ / cm-per-cell constant** (= 1 if the scale is one cell). The ratio is the
statistic, not the two ranges separately:

| cm per W wire (q25–q75) | items | λ | **λ / cm-per-cell** |
|---|---|---|---|
| 0.68 (0.60–0.71) | 30 | 1.45 cm | **2.12** |
| 1.12 (0.96–1.31) | 87 | 1.74 cm | **1.55** |
| 2.13 (1.78–2.64) | 51 | 1.68 cm | **0.79** |
| 4.68 (3.66–5.49) | 38 | 1.74 cm | **0.37** |
| 18.05 (10.0–30.4) | 21 | 2.54 cm | **0.14** |

The drift-slice stratification is the same, 2.25 → 0.24 over the same five
buckets. **λ is not proportional to cell size — in the tail it is
anti-proportional.** The conclusion does not rest on the sparse top bucket
(21 tracks with a 3× internal spread): the [3, 8) cm bucket alone, 38 tracks with
q25–q75 = 3.66–5.49, already reads 0.37 against the 1.0 a geometric scale
requires. **The wave is not on the lattice clock.**

## 4. What the wave actually is, as far as this measures it

* **It is coherent, not white.** Pooled ACF(1) = **0.836**: only ~16 % of the
  residual is point-to-point noise. The scanner is not looking at digitisation
  hash.
* **Its scale is ~3 trajectory points.** ACF falls to 1/e at **1.74 cm = 2.9
  points**, and that number is nearly the same on every geometric stratum (§3c) —
  i.e. it is a **fitter scale in trajectory-point units**, not a detector scale.
* **The 30–50 cm period of doc 03 §9 item 2 is not the population's scale.** The
  only long-range structure in the pooled ACF is a shallow negative lobe at
  12–18 cm (ACF −0.03 to −0.06). Doc 03's 30–50 cm was read off two specific
  wire-parallel tracks; it does not generalise, in the same way doc 20 §7.2 found
  the trough/dead-point reading did not generalise.

## 5. Mechanisms excluded, with the number that excludes them

* **A path-length (dL) artifact.** `L` is true arc length: the 3-D step between
  consecutive points divided by ΔL is **1.0004** median, q01 0.985, q99 1.015
  over **96 606 steps of all 303 chains**. The wave it would have to make is the
  plateau dQ/dx spread, fractional RMS **0.375** per chain (q25 0.279, q75 0.540,
  233 chains): a ±1.5 % tail is two orders of magnitude short.
* **PR segment boundaries / a stitching scale.** Correlation of the residual with
  distance to the nearest PR segment vertex: r = **+0.019** (p = 0.044 two-sided).
  Non-zero, negligible.
* **The lattice as the driver** — §3, three ways.

## 6. The one non-lattice correlate that is there

**Trajectory wobble.** Signed against the local ±3 cm chord, the residual
correlates with the fit's transverse excursion at

> **r = −0.1394**, circular-shift null +0.0008 ± 0.0087 → **16 σ**, p = 0.002

i.e. **where the fitted trajectory bulges off its own local chord, dQ/dx reads
low.** That is r² = 1.9 % of the residual variance (2.3 % of the coherent part) —
larger than the lattice term, still small, and it is a *correlation*, not a
demonstrated mechanism. It is however the same direction as doc 03 §9 item 5
("the oscillations of item 2 are trajectory, not noise"), now with a number
attached and measured on 227 chains rather than 2.

**This does not separate cause from co-symptom.** Transverse excursion from the
local chord is largest exactly where the fit is least constrained, which is also
where charge assignment is worst — so r = −0.139 is equally consistent with
"wobble drives dQ/dx low" and with "wobble and dQ/dx error are two symptoms of
the same locally unconstrained fit". This data cannot tell them apart.

## 7. What is NOT concluded

* **Not** that the wave's origin is known. The lattice is excluded as the
  driver; wobble and the segment scale are measured and small. Roughly 96 % of
  the residual variance is unattributed, and part of it is certainly physical
  (Landau fluctuation and delta rays seen through the fit's ~1.7 cm resolution).
  This work did **not** run a simulation to separate physical fluctuation from
  fit noise, which is what would settle it.
* **Not** a statement about the Bragg region. Everything is on `rr > 30` cm.
  The STM verdicts are made at the stop, where none of this was measured.
* **Not** a defect claim about the charge fit. A sub-cell bias of ±0.1 σ is
  reported as measured; whether `TrackFitting::dQ_dx_fit`'s regulariser should
  do something about it is not addressed and no code was read for it.
* **Not** transferable to PDVD or SBND. PDVD payloads exist
  (`prep-pdvd`, 570) and the same scripts run on them; that was not done.
* **Not** a licence to move any threshold. Nothing here bears on
  `plateau_mip_lo/hi`, `contrast`, or any STM reject bit.
* **Not** anything about the induction planes. `pu` and `pv` are uncalibrated
  here and discontinuous at the percent level (§1); their null fold is reported,
  not relied on. Whether the U and V cells carry the same sub-cell bias as W is
  **open**, and answering it needs an unwrapped induction coordinate, which this
  payload does not carry.

## 8. Files

| what | where |
|---|---|
| scripts (15) | `pdhd/docs/scripts/d24_*.py` |
| numeric outputs (every table above) | `pdhd/docs/figs/24_*.txt`, `24_survey.tsv` |
| figures | `pdhd/docs/figs/24_fig1_subcell_fold.png`, `24_fig2_correlation_length.png`, `24_fold_profiles.png`, `24_look_028084_8_c120.png`, `24_zoom_028084_8_c120.png` |
| source payloads (read in place, never written; untracked, `*.json` ignored) | `pdhd/stm_michel_scan/prep-pdhd/` (arm `d53h`, 303 files, built by `prep_stm_michel_scan.py`) |
| the open item this answers | `pdhd/docs/20_stm-michel-dqdx-readability.md` §8; `pdhd/docs/03_check-stm-michel-pdhd.md` §9 items 2 and 5 |

# One STM fit end to end — SBND evt 286241, cluster 8 (doc 42)

Companion to docs 40/41.  Doc 41 says what the `save_stm_fit` knob writes;
this doc walks **one** fit through all three displays and compares its fitted
dQ/dx with the expectation the tagger itself uses.  It also introduces the
SBND converter app that replaces the uBooNE one.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# 0. the fit itself (already produced; op point of doc 39 + -stm-fit):
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit"
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-stmon ./run_nusel_evt.sh data 8 $F   # idx 8 = evt 286241
# (run the driver with no idx to print the idx -> EVT_ID map for the sample)

# 1. Magnify-tracking file (NEW SBND app, not the uBooNE one):
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-mcp10-stmon/nusel_evt286241/tracking-stm.root -tT_rec_charge \
  -oshowcase-stmfit-286241/track_com_286241.root -f2
cd Magnify-tracking-SBND && ./magnify.sh ../showcase-stmfit-286241/track_com_286241.root
# (needs $DISPLAY; headless recipe + the bug fixes that made this work: doc 43)

# 2. Bee upload zip (QL layers + the stm_fit dQ/dx layer), then upload:
python3 make_stmfit_bee.py -w work-mcp10-stmon -o showcase-stmfit-286241/upload_286241.zip 286241
./upload-to-bee.sh showcase-stmfit-286241/upload_286241.zip     # prints the Bee URL

# 3. numbers + plot of this doc:
python3 stmfit_showcase.py -r work-mcp10-stmon -e 286241 -b 80 \
  -o showcase-stmfit-286241/dqdx_286241_blk80.png
```

## 0. Sample caveat — read this before interpreting any dQ/dx number

The 30-event knob-on round, this event included, is **MCP2025C reco1 DATA**
(run 18255; `run_nusel_evt.sh data …`, `_runlib.sh:25` "Mode still drives
reality (sim vs data)"; doc 23 names the sample).  The input art file carries
`artdaq::Fragments` and **zero** `simb::`/`sim::` products.  Docs 40/41 record
"MC-first sample" as the owner's answer to plan question Q3 — that was the
decision, not what was run, and this doc corrects the record.

Consequences:

- Everything below is **fitted vs. expectation**, never fitted vs. truth.
  It mixes the fitter with the recombination model, the calibration scale,
  the electron lifetime, and the field response.  A normalization offset
  here cannot be attributed to the trajectory or dQ/dx fit.
- **Fitted vs. truth — the comparison that actually validates the fitter —
  is not this event.**  It is **§7**, added later, on the 10-event MC sample
  `input_files/input-10evt-mc/` and with the SimEnergyDeposit dumper
  (`dump_truth_sed.C`) this section originally listed as missing.  Take any
  statement about how well the fitter reproduces charge from §7; §2–§5 below
  are a data event and can only compare against a model.

## 1. New app: `wire-cell-sbnd-magnify-tracking-convert`

`root/apps/wire-cell-sbnd-magnify-tracking-convert.cxx` (toolkit), a
fork-by-duplication (M10) of the uBooNE app, which keeps its qlport consumers
and stays byte-for-byte untouched.  Three differences, all SBND-driven:

1. **No hard-coded uBooNE x transform.**  The uBooNE app rewrites truth x as
   `(x+0.6)/1.098*1.1009999-0.1101` (its lines 148/183) to undo a
   uBooNE-specific simulation-vs-imaging drift-speed mismatch and a Y/U plane
   offset.  Here truth x passes through unchanged; `-s<scale> -c<offset>`
   apply `x -> x*scale + offset` if SBND ever needs one.  **This is what
   unblocks MC mode (`-f1`) for SBND**, which doc 41 listed as blocked.
2. Carries the STM per-point extras into `T_rec` as nested vectors:
   `rec_rr`, `stm_pass`, `stm_status`.
3. Clones `T_stm_pass` / `T_stm_eval` into the output, so the Magnify file
   explains its own tracks without going back to the dump.  (It also avoids
   the duplicate tree cycles the uBooNE app leaves behind — cosmetic.)

Verification:

- **Data mode is a faithful fork**: on this event, every shared branch
  (`rec_x/y/z/dQ/dx/L/u/v/w/t`, `reduced_chi2`) is `array_equal` to the
  uBooNE app's output; the SBND file adds only `rec_rr`/`stm_pass`/`stm_status`.
- **MC mode pass-through proven**: with a synthetic truth tree spanning
  x = 60 → 80 cm, the SBND app writes `T_true` x = 60.0000 → 80.0000; the
  uBooNE app writes 60.6555 → 80.7101.  That ~0.7 cm shift is exactly the
  uBooNE constant that made `-f1` unusable for SBND.
- `wcdoctest-clus` unaffected (no library change); the app is a new binary,
  no existing output changes.

## 2. The event

`work-mcp10-stmon/nusel_evt286241/`, block id 80 = **cluster 8, pass 0
(forward)**, `status = 0` → **accepted STM** (stopping-muon tag).

| quantity | value |
|---|---|
| fitted points | 251 |
| track length (fit path) | 161.7 cm |
| TPC | TPC1 only (x ∈ [57.6, 82.8] cm) |
| median reduced_chi2 | 1.91 |
| kink_num / npoints | 251 / 251 (no kink → whole path is the "exit" section) |
| exit_L / exit_dqdx | 162.3 cm / 62.1 ke/cm |

Residual range `rr` is measured **from the candidate stopping end**: the STM
pass runs from the exit point (`first_wcp`, where the track leaves the
detector) toward the far end (`last_wcp`, the stopping candidate), and the
dump stores `rr = L_tot − L_i`, so `rr = 0` is the stopping end.

## 3. The three displays

1. **Bee** — <https://www.phy.bnl.gov/twister/bee/set/d11b087c-f8a3-4781-96ac-dc3a08f82a87/event/list/>
   (uploaded 2026-07-24 with owner approval), from
   `showcase-stmfit-286241/upload_286241.zip`, six layers:
   `img-global`, `clustering-global`, two `channel-deadarea-*`, `op`, and
   **`stm_fit-global`**, the fitted trajectory with `q = dQ*0.1 − 1000`
   (the uBooNE `track_fit` convention, so Bee's `q` color ramp reads as
   dQ/dx).  Built by `make_stmfit_bee.py`, which merges the QL tree
   (`ql_evt*/mabc-all-apa.zip`) with the PR tree's STM layer
   (`nusel_evt*/mabc-pr.zip`); the two `clustering-global` layers were
   checked to be the same frame and point count (25786) before merging.
   Upload with `./upload-to-bee.sh` — the URL it prints is the link.
   In Bee, select the `stm_fit` layer and colour by `q`: the Bragg end is
   the bright end.  wire-cell-bee3 itself is untouched (owner directive).
2. **Magnify-tracking** — `showcase-stmfit-286241/track_com_286241.root`,
   1 track block / 251 points, opened with `Magnify-tracking-SBND`
   (SBND geometry: U/V/W 3968/3968/3340 channels, 857 time slices).  Pads
   4–6 show the fit points on the measured channel×time maps per plane,
   pads 7–9 the `(pred−meas)/meas` residuals, pad 1 the dQ/dx vs L curve.
   Screenshot: `showcase-stmfit-286241/magnify_286241_blk80.png`.
   **Correction (doc 43):** as first written this display did not work at
   all — the GUI could not compile here, and the converter dropped two
   branches the dQ/dx panel indexes unconditionally, so it aborted on this
   file.  Doc 43 has the eight fixes; the file above was regenerated after
   them (from `work-stmbadch`, proven identical to the `work-mcp10-stmon`
   record except for the repaired `T_bad_ch`).
3. **nusel viewer** — `nusel_display/` on :5010 with the `-stmon` roots
   reads the same `tracking-stm.root` and draws the dQ/dx-vs-rr panel plus
   the trajectory crosses on the three projections (doc 41 §Consumers 3).

## 4. Fitted dQ/dx vs expectation for this fit

`stmfit_showcase.py` compares each fitted point with the **muon expectation
the tagger itself uses** (`nusel_display/stm_ref_dqdx.json`, extracted from
the compiled config's `MuonDeDx` `LinterpFunction`, i.e. the exact curve
`eval_stm` tests against; tabulated 0.5 → 59.5 cm) and with the flat
50 ke/cm MIP line.  Plot: `showcase-stmfit-286241/dqdx_286241_blk80.png`.

| rr bin [cm] | n | fitted [ke/cm] | expected [ke/cm] | ratio |
|---|---|---|---|---|
| 0 – 2 | 3 | 151.5 | 115.8 | 1.31 |
| 2 – 5 | 6 | 95.6 | 77.4 | 1.24 |
| 5 – 10 | 8 | 74.7 | 65.7 | 1.14 |
| 10 – 15 | 8 | 74.3 | 59.3 | 1.25 |
| 15 – 20 | 8 | 67.3 | 55.8 | 1.21 |
| 20 – 30 | 16 | 56.5 | 52.9 | 1.07 |
| 30 – 40 | 15 | 57.5 | 50.8 | 1.13 |
| 40 – 60 | 31 | 54.9 | 49.7 * | 1.10 * |
| 60 – 100 | 62 | 57.1 | — (no table) | — |
| > 100 | 94 | 52.9 | — (no table) | — |

The `MuonDeDx` table is tabulated on **rr = 0.5 – 59.5 cm** only (60 entries,
step 1 cm).  Rows beyond it have no expectation to compare against —
interpolation would silently clamp to the last table value and manufacture a
ratio, so those cells are left empty; the 40–60 row (*) is partly beyond the
domain.  Past ~60 cm the flat 50 ke/cm MIP line is the only reference, and
the plateau ratio below uses it.

Reading:

- **Shape is right.**  The fit reproduces the Bragg rise monotonically from a
  ~53 ke/cm plateau to 151 ke/cm in the last 2 cm — a factor ~2.9, against
  the expectation's ~2.4 over the same span.  This is the qualitative
  statement the trajectory + dQ/dx fit had to pass, and it passes here.
- **Normalization is high by ~10–25 %** across the whole rr range the muon
  table covers (0–60 cm), with no trend that would single out the stopping
  end; beyond the table, the plateau sits 8 % above the flat 50 ke/cm line.
  Since this is uncalibrated
  data (no lifetime or gain calibration applied downstream of the fit), an
  offset of this size is expected and is **not** evidence about the fitter.
- The excess is not concentrated at the stopping end, so it does not look
  like a `dx` artifact of the last few points.

## 5. Population context (why one event is not enough)

From the same 30-event round (`stmon_stats.py`), for the 11 accepted-STM
tracks with ≥ 60 points and ≥ 40 cm of range:

- **Only this one shows a Bragg rise.**  Ratio of median dQ/dx at rr < 5 cm
  to the rr > 40 cm plateau: 1.85 here; 0.90–1.09 for five others (flat, no
  rise); < 0.8 for the rest, including negative values.  Accepted-STM is
  supposed to mean "stopping muon", so either the acceptance is admitting
  non-stoppers, or the end being scored is not the physical stopping end in
  those events.  **Open — hand-scan item, not tuned.**
- **12.8 % of all 18561 fitted points have negative dQ/dx**
  (status 0: 9.8 %, status 2: 11.7 %, status 3: 13.6 %, status 4: 20.9 %).
  Regularized least-squares fits can undershoot below zero, but two accepted
  tracks are more than half negative (`evt287825` blk 20: 56 % of points, and
  the profile is negative over 60 cm of the track).  **Open finding** — it
  says the dQ/dx solve is unstable on some topologies, and any KS test run
  on such a profile is meaningless.

Both belong to phase-4 check 4 (stopping-particle shape) in doc 40 and are
the first things the fitted-vs-truth comparison should settle.

## 6. Status

- New app + this doc: committed (see git log for hashes).
- Products under `showcase-stmfit-286241/`: the Magnify ROOT and the PNG are
  committed (`git add -f`, both under 90 kB).  `upload_286241.zip` (378 kB)
  is deliberately **not** committed — the Bee set is hosted and the zip
  regenerates from `make_stmfit_bee.py` in one command.
- Bee set uploaded to the public BNL server with owner approval:
  `d11b087c-f8a3-4781-96ac-dc3a08f82a87` (link in §3).  It contains
  reconstruction points only — no raw waveforms.
- Open items this doc raises, both unfixed and both hand-scan gated: the
  missing Bragg rise on 10 of 11 accepted-STM tracks, and the 12.8 %
  negative-dQ/dx fraction (§5).  Doc 43 §D adds a third of the same family:
  rejected-pass blocks have almost no fitted 2D charge under their own
  track (8–39 % coverage) while accepted ones are at 100 %.

---

# 7. MC: fitted vs. **true** dQ/dx — run 228 event 18, block 150

Everything above compares a fit with a *model*.  This section compares it with
the *truth of the same event*, which is the only test that separates the
trajectory + dQ/dx fit from the recombination model, the calibration scale and
the electron lifetime.  It is a different event, in a different (simulated)
sample, and it is the section to cite for fitter performance.

## 7.0 Repro

> **Superseded in part by doc 44** (`44_stm-fit-truth-dqdx.md`).  The true
> dQ/dx quoted in this section divides `true_dQ` by `rec_dx`, which is the
> length of the *fitted* cell, not of the true track inside it.  Everything
> aggregate here survives (median true 49.5 ke/cm, integral ratio 1.043, the
> no-drift-trend conclusion; doc 44 reproduces 1.043 exactly by running the
> compare script on this section's own `track_com_18.root`).  What is
> superseded is the **per-point truth scatter**, the **endpoint values**, the
> **true column of the §7.3 table**, and the rationale of note 3 below.
>
> The products in `showcase-stmfit-mc-evt18/` were **regenerated in place** on
> 2026-07-25 (owner decision) and now carry the fix — so the ROOT files and
> PNGs referenced below no longer reproduce this section's per-point numbers.
> This text is the only remaining record of them.  `upload_mc18.zip` and the
> Bee set are unaffected (they never read the truth file).


```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# 1. chain on the MC sample, doc-39 op point + the fit dump, fresh work root:
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit"
SBND_WORK_ROOT=$PWD/work-mcsim-stmon SBND_MAX_JOBS=6 ./run_img_evt.sh   mc all
SBND_WORK_ROOT=$PWD/work-mcsim-stmon SBND_MAX_JOBS=4 ./run_nusel_evt.sh mc all $F

# 2. truth: SimEnergyDeposit -> the converter's (N, x, y, z, Q) tree
root -l -b -q 'dump_truth_sed.C("input_files/input-10evt-mc/2025f-mc.root",228,18,
                                "work-mcsim-stmon/nusel_evt18/tracking-stm.root",
                                150, 5.0,
                                "showcase-stmfit-mc-evt18/truth-evt18-blk150.root")'

# 3. Magnify-tracking file in MC mode (-f1 pairs fit points with truth):
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-mcsim-stmon/nusel_evt18/tracking-stm.root -tT_rec_charge \
  -ashowcase-stmfit-mc-evt18/truth-evt18-blk150.root -nT \
  -oshowcase-stmfit-mc-evt18/track_com_18.root -f1
cd Magnify-tracking-SBND && ./magnify.sh ../showcase-stmfit-mc-evt18/track_com_18.root
# (headless recipe: doc 43 Repro; block 150 is cluster index 1 in the GUI)

# 4. numbers + plot of this section, and the Bee zip:
python3 stmfit_mc_compare.py -f showcase-stmfit-mc-evt18/track_com_18.root -b 150 \
    -o showcase-stmfit-mc-evt18/dqdx_mc_evt18_blk150.png
python3 make_stmfit_bee.py -w work-mcsim-stmon -o showcase-stmfit-mc-evt18/upload_mc18.zip 18
./upload-to-bee.sh showcase-stmfit-mc-evt18/upload_mc18.zip
```

## 7.1 Sample and truth extraction

`input_files/input-10evt-mc/2025f-mc.root` — the SBND 2025-fall production
sample (GENIE ν + CORSIKA cosmics through G4 + detsim + reco1), run 228,
13 art events of which the 10 with DNN-SP frames (2, 9, 11, 12, 14, 18, 31,
35, 41, 42) are the standalone chain's `mc` mode.  Unlike the MCP2025C reco1
data of §0 it carries `sim::SimEnergyDeposit`.

`dump_truth_sed.C` (new) turns those deposits into the `(N, x, y, z, Q)` tree
the converter's `-f1` reads.  Three decisions in it matter for reading any
number below:

1. **Bare-ROOT, no LArSoft.**  The product branch
   `sim::SimEnergyDeposits_ionandscint_priorSCE_G4.obj` is unsplit, but the
   file carries the StreamerInfo for `sim::SimEnergyDeposit` v20, so
   `TTree::Draw` reads it through an emulated class.  As in
   `SBNDReco1Reader.h`, only per-branch reads are safe on art files.
   (Trap met and fixed: `SetEstimate(-1)` sizes the buffer to the tree's
   *entry* count — 13 — and `GetVal()` then returns uninitialized memory past
   the 13th deposit, which looked exactly like a truth/reco frame mismatch.)
2. **The particle is elected by coverage, not by charge**: each fitted point
   votes for the particle owning the deposit nearest to it.  Electing by
   total charge instead picks a dense blob — on the first block tried, a 1 cm
   vertex proton outvoted an 85 cm muon.
3. **A 5 cm cut around the fitted track is applied to the dumped truth.**
   The converter accumulates *every* truth point onto its nearest fitted
   point with **no distance cut**, so anything outside the fit's extent piles
   onto an endpoint and reads as a fake Bragg peak.
   *(Doc 44: the fake peak came from dividing by `rec_dx` and is gone with
   `true_dx` — those endpoints read 51.4 and 48.8 ke/cm, not 186.5 and 396.5.
   The cut is still wanted, but to bound the end-cell length, not to hide the
   artifact.)*  `Q` is `numElectrons`:
   post-recombination ionization **at the deposit**, before drift.

## 7.2 The block

`work-mcsim-stmon/nusel_evt18/`, block 150 = **cluster 15, pass 0**,
`status = 3` (STM evaluated it and *rejected* it).

| quantity | value |
|---|---|
| fitted points | 346 |
| track length (fit path) | 219.5 cm |
| drift x covered | −26.2 → 172.0 cm (**crosses the cathode**) |
| true particle | pdg 13, `origTrackID` 20000167 |
| true path (full, uncut) | (−27.3, 203.7, 398.2) → (201.3, 126.0, 335.0) cm |
| exit_L / exit_dqdx (STM) | 218.8 cm / 51.4 ke/cm |

The muon enters through the top (y = 203.7) and leaves through the x = +201
anode, i.e. **through-going, not stopping** — and STM rejecting it on a
51.4 ke/cm exit dQ/dx is the tagger behaving correctly.  A through-goer is
also the better first truth test: the true dQ/dx is a flat MIP over 220 cm,
so any normalization or drift-dependent error in the fit has nowhere to hide.

**Pairing quality first** (nothing below means anything without it):

| | value |
|---|---|
| `com_dis` (fitted point → nearest truth point) | median **1.78 cm**, p90 3.45, max 3.94 |
| fitted points whose nearest deposit is this muon | 289 / 346 = **83.5 %** |
| charge within 5 cm of the fit belonging to it | 81.6 % |
| muon charge dropped by the 5 cm cut | 10.6 % (the part beyond the fit's end) |

A blind scan of a constant x shift prefers +3 cm (median 1.22 cm there vs.
1.78 at zero) — small, one-sided, and consistent with the truth being
`priorSCE` while the reco sits at the space-charge-distorted position.  It is
**not** corrected here (`-s`/`-c` left at identity).

## 7.3 Result

Plot: `showcase-stmfit-mc-evt18/dqdx_mc_evt18_blk150.png`; first and last
fitted point excluded everywhere below (§7.1 note 3 — the truth beyond the
fit lands on them: 111 ke and 258 ke against a 31 ke median).

*(Doc 44 re-measures this table with the correct denominator: the fitted and
ratio columns barely move, but the true column's spread across bins collapses
from 5.8 ke/cm here to 2.3 ke/cm, and no point needs excluding.  The wobble
this column shows is cell-occupancy noise, not the muon.)*

| L [cm] | n | fitted [ke/cm] | true [ke/cm] | fitted/true | ⟨x⟩ [cm] |
|---|---|---|---|---|---|
| 0 – 22 | 36 | 49.1 | 49.7 | 1.050 | −16.1 |
| 22 – 44 | 34 | 48.9 | 53.3 | 0.938 | 3.5 |
| 44 – 66 | 35 | 51.7 | 47.5 | 1.024 | 23.4 |
| 66 – 88 | 36 | 54.8 | 50.3 | 1.102 | 43.0 |
| 88 – 110 | 35 | 49.6 | 48.8 | 1.042 | 63.5 |
| 110 – 132 | 34 | 52.8 | 48.8 | 1.175 | 83.4 |
| 132 – 154 | 34 | 50.1 | 48.4 | 1.027 | 102.9 |
| 154 – 176 | 33 | 48.8 | 50.0 | 1.094 | 123.0 |
| 176 – 198 | 34 | 46.4 | 50.2 | 1.038 | 142.1 |
| 198 – 220 | 33 | 44.5 | 49.1 | 0.931 | 161.8 |

- **Median fitted dQ/dx 49.5 ke/cm vs. median true 49.5 ke/cm**; integrated
  over the 344 core points, fitted/true = **1.043**.  No negative dQ/dx point
  anywhere in this block.
- **No drift trend — and that is a statement about attenuation.**  The truth
  `numElectrons` is the charge *at the deposit*, before any drift, so
  fitted/true is exactly the factor the chain applies between deposit and
  fitted charge.  This track samples nearly the whole drift: it starts 26 cm
  from the cathode in TPC0, crosses it, and ends 28 cm from the TPC1 anode,
  so the drift distance (200 − |x|) runs 199 → 29 cm.

  | drift distance [cm] | n | fitted/true (integral) | median point ratio |
  |---|---|---|---|
  | 0 – 40 | 20 | 0.895 | 0.873 |
  | 40 – 80 | 67 | 1.009 | 0.966 |
  | 80 – 120 | 68 | 1.111 | 1.032 |
  | 120 – 160 | 68 | 1.080 | 1.061 |
  | 160 – 200 | 117 | 0.998 | 0.999 |

  The ratio is flat to ~±10 % and **non-monotonic** — it peaks in the middle
  of the drift, not at either end.  A straight line through the binned
  integral ratios has slope +0.00065 per cm, i.e. +0.11 across the 171 cm of
  drift sampled; that slope is bin-edge dependent (+0.02 with 40 cm bins
  shifted by 20 cm) and its sign is *opposite* to attenuation, so read it as
  "no trend at this precision", not as a measured slope.  (Theil–Sen on the
  340 individual points agrees: +0.00063 per cm.  An OLS slope on the point
  ratios is not usable — their distribution has mean 1.26 against median
  0.99.)

  An **uncorrected** electron lifetime would have gone the other way and been
  larger than the scatter: with the runner's `LIFETIME=6` ms and
  `DRIFTSPEED=1.563` mm/µs, exp(−t/τ) is 0.809 at 199 cm of drift against
  0.970 at 29 cm, so fitted/true would **fall by ~17 %** from the near-anode
  end to the near-cathode end.  It does not (the far end is the 0.998 bin,
  the near end the 0.895 one — same size, wrong sign, and not monotonic in
  between).  So either the attenuation is corrected
  upstream of the fitted charge or the simulation carried an effectively
  infinite lifetime — **this track does not distinguish the two**, and the
  distinction matters before quoting an absolute charge scale on data.
- Point-by-point the truth is noisy (0–116 ke/cm) while the fit is smooth.
  That is the pairing, not physics: each fitted point owns a Voronoi cell of
  the truth cloud and the cells hold unequal numbers of G4 steps.  The
  comparison is only meaningful in running medians / bins, which is what the
  plot and the table show.
- **What this does and does not establish.**  On this track the trajectory is
  right to ~2 cm and the charge scale is right to a few per cent.  It is one
  through-going muon in one event: it says nothing yet about the Bragg shape
  (§5's open item), and no stopping muon in this MC sample survives §7.4.

## 7.4 The other four fitted blocks in this sample — and a finding

The 10 MC events yield only five STM fit blocks.  Running the same
coverage/pairing test on all of them:

| event | block | status | npts | nearest-deposit median | elected particle | verdict |
|---|---|---|---|---|---|---|
| 18 | 150 | 3 rejected | 346 | **1.78 cm** | μ (84 % coverage) | §7.3 |
| 2 | 110 | 3 rejected | 378 | 1.17 cm (p90 33 cm) | μ⁺ (49 %) | follows a real muon, then strays |
| 9 | 110 | 4 rejected | 515 | 11.3 cm | e⁻, 9 deposits (20 %) | no true particle |
| 18 | 80 | **0 accepted** | 142 | 6.5 cm | Ar nucleus, 1 deposit (40 %) | no true particle |
| 42 | 60 | **0 accepted** | 20 | 59.2 cm | e⁻, 1 deposit | see below |

Both **accepted** (i.e. tagged stopping-muon) blocks fail the truth test,
while the two best-paired blocks are ones STM rejected:

- **evt 42 block 60** is a real muon at the wrong place: a constant shift of
  **−80 cm in x** brings the median pairing distance to 0.42 cm.  80 cm is
  ~0.5 ms of drift, so this is a flash-time / t0 error upstream of the fit,
  not a fit error.  (No other block prefers a large shift: evt 2 +0 cm,
  evt 18/150 +3 cm, evt 18/80 −2 cm, evt 9 +5 cm.)
- **evt 18 block 80** is worse and is not a t0 effect — no shift in ±400 cm
  gets it below 6 cm.  It is 85 cm long, **42 % of its 142 points have
  negative fitted dQ/dx**, and its STM `exit_dqdx` is **−6.9 ke/cm** — a
  negative charge per unit length, accepted as a stopping muon.

This is the same family as §5 (missing Bragg rise on accepted-STM tracks;
12.8 % negative dQ/dx) and doc 43 §D (rejected-pass blocks with 8–39 %
charge coverage), now with truth attached: **on this sample the STM
acceptance is not selecting tracks that a true stopping particle produced.**
Nothing was tuned in response — the sample is 10 events and the next step is
a larger MC round, not a parameter change.

## 7.5 Displays

1. **Bee** — <https://www.phy.bnl.gov/twister/bee/set/1b654c7c-9e41-4d07-88e5-dc9e65f6cae8/event/list/>
   (uploaded 2026-07-24), from `showcase-stmfit-mc-evt18/upload_mc18.zip`,
   the same six layers as §3 including `stm_fit-global` with the fitted
   dQ/dx as `q = dQ*0.1 − 1000`.  **Reco only** — no truth layer was added
   (the truth comparison lives in the Magnify file and in §7.3).
2. **Magnify-tracking** — `showcase-stmfit-mc-evt18/track_com_18.root`,
   opened in **MC mode** (it has `T_true`, so the GUI takes its `!isData`
   branch): pad 1 draws the fitted dQ/dx in black and the **true dQ/dx in
   red** on the same axes, pad 2 the fitted-vs-true angle and distance.
   Screenshot: `showcase-stmfit-mc-evt18/magnify_mc_evt18_blk150.png`
   (block 150 = cluster index 1; index 0 is block 80, whose truth curve is
   flat zero by construction — the dumped truth is the block-150 muon).
   This is the first time the GUI's MC branch has run on a real file.
   **Display defect seen there, not fixed (unrelated to this work):** on the
   projection pads a cathode-crossing track draws a spurious horizontal line
   across the whole channel axis.  SBND concatenates the two TPCs onto one
   channel axis, so consecutive fitted points either side of the cathode sit
   ~3000 channels apart and the `"LP"` polyline joins them.  Cannot happen in
   single-TPC uBooNE, which is where the drawing code comes from.

## 7.6 Status

- New: `dump_truth_sed.C`, `stmfit_mc_compare.py`, this section.  No C++ or
  jsonnet change — the converter's `-f1` path and the GUI's MC branch already
  existed (doc 42 §1, doc 43 §E); this is their first use on real truth, so
  **no reconstruction output moves and no A/B gate applies**.
- `work-mcsim-stmon/` is a fresh work root; no existing tree was touched.
- Committed under `showcase-stmfit-mc-evt18/`: `track_com_18.root` (347 kB),
  `truth-evt18-blk150.root` (210 kB) and the two PNGs.  `upload_mc18.zip`
  (428 kB) is not committed — the Bee set is hosted and it regenerates in one
  command, matching §6.
- Open, unfixed, and now truth-backed: §7.4's accepted-STM blocks, the
  evt 42 −80 cm t0 error, and the cathode-crosser display line of §7.5.

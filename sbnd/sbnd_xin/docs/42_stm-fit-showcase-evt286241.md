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
  has not been done**, because the sample has no truth.  It needs an MC
  sample and the truth pairing that the new SBND converter (§1) now permits.
  Locally, `input_files/2025f-mc-resim.root` has the needed products
  (`sim::SimEnergyDeposits`, `sim::SimChannels`, `simb::MCParticles`,
  `sim::MCTracks`) but holds **1 event**; `input_files/2025f-mc.root` is a
  dangling symlink into `/exp`.  Still missing: a dumper turning
  SimEnergyDeposit into the converter's `(N, x, y, z, Q)` truth tree.

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
   (`b78b255`, SBND geometry: U/V/W 3968/3968/3340 channels, 857 time
   slices).  Pads 4–6 show the fit points on the measured channel×time
   maps per plane, pads 7–9 the `(pred−meas)/meas` residuals, pad 1 the
   dQ/dx vs L curve.
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
  negative-dQ/dx fraction (§5).

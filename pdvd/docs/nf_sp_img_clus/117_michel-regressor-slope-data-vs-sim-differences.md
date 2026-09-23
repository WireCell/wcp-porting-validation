# doc pdvd/117: why the DUNE-VD drift regressor reads slope 0.42 on ProtoDUNE Michels, and how to find out

**Status (2026-09-23, round 5).** The Michel-domain test ran (section 10).
- **The muon-removal truncation is the main carrier.** Applied to the model's own simulated electrons at each data
  Michel's removed fraction, truncation plus exact-zero background compresses the simulation slope 0.912 → 0.58-0.64
  (Δ −0.27 to −0.33, bracketing the two ends). The pre-registered "mostly intercept" prior is refuted.
- **It matches the data's own M against Z gap** (0.66 → 0.42, −0.24).
- **With the wire filter** (+0.105, S4), about 0.05-0.12 of the 0.49 gap remains. That is within the D_L bound (S6)
  plus the untested noise and fluctuation.
- **Next:** train with data-like truncation augmentation (section 10.5).

**Status (2026-09-23, round 4).** Study S6 ran (section 9).
- **D_L on PDVD data** is 4.9 [3.3, 6.8] cm²/s after calibration, consistent with the params value 4.13, with a
  subset spread of about ±2.
- **Top CRP**, with its own top-CRP calibration: 3.4 [2.2, 5.8]; tpw < 1 gives 2.7 [1.9, 4.9]. The training
  simulation's equivalent at the data's drift speed is 3.14.
- **D_L can account for at most about 0.1-0.15 of the remaining ~0.39 of slope** (the top CRP's lower bound). It is
  not the main cause. The rest is in the Michel image or in charge-shape effects that do not grow with drift.

**Status (2026-09-23, round 3).** Study S4 ran (section 8). The real re-SP with the training `Wire_col` raises the data
slope 0.421 → 0.525 (Δ +0.105 [0.060, 0.151]), as S1 predicted before it ran. About 0.39 of the gap to the
simulation's 0.91 remains, with a time-axis or Michel-domain cause.

**Status (2026-09-23, round 2).** Study S1 ran; see section 6. Section 7 corrects sections 2.3 and 2.4.
- **The transverse-diffusion budget is moot.** The model barely reads the channel axis at the D_T gap's size.
- **The wire-filter difference is worth about +0.11 of slope,** not zero as section 2.1 assumed.
- **The "time width does not grow on data" reading is not established:** 1σ on 56 Michels.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
python3 d117_wire_kernel.py      # the Wire_col kernels (section 2.1) and the slope budget (section 2.3)
```

Everything else is quoted from committed records:

| doc | what is taken from it |
|---|---|
| doc pdvd/98 | §6 and §6c: the data readouts and the matched-sim expectation; §7: the configuration comparison; §11: the top/bottom split and the gain in/out test |
| doc pdvd/44 | l. 147-151: joint and per-plane D_T,eff; l. 218-221: per-face values |
| doc pdvd/47 | l. 425-435: c data against sim, and D_T,eff returned by SP; §12.2: the data-only excess |
| doc 07 | l. 64: anodes 4-7 = top CRP, at x > 0 |

**Scope.** No code, config or arm is changed, and no model is retrained. This doc is a synthesis of docs pdvd/98, 44
and 47 and pdhd/02, plus one kernel calculation. Nothing here is a new measurement; section 4 lists the measurements
that would decide the question.

## 1. The question

The DUNE-VD single-particle diffusion regressor (`DNN_ROI_SP` m3-200k-w) was applied to the hand-scanned ProtoDUNE
STM+Michel sample, with the muon removed (doc pdvd/98).

It correlates with the Q-L drift on both detectors:
- PDVD latest: rho +0.60, p 1e-4;
- PDHD: rho +0.56.

But the response is compressed:
- **data**: OLS slope 0.42 on PDVD (0.45 on PDHD), intercept about +50 cm;
- **model's own matched simulation**: 0.91 [0.86, 0.96], with the same drifts, energies and saturation floor.

The model's floor is inside that simulation expectation, so it is not the cause (doc 98 §6c item 3). The scatter of mu
about the line is near expectation on PDVD (40 against 29 cm), so the images **compress the response rather than add
noise**.

A colleague's slide summarises the causes as "mismatch in the SP chain between PDVD data and DUNE-VD simulation" and
"mismatch between data and simulation". The owner named three concrete candidates:
1. diffusion in data differs from MC;
2. field response differences, including position-dependent response: the track trajectory and dQ/dx fit needed an
   added constant width;
3. different SP software filters.

## 2. Reading the three candidates

### 2.0 Constant width moves the intercept; drift-growing width moves the slope

This is a first-order expectation, not a result.

Suppose the model effectively reads drift from the image width, inverting σ² = c² + k·x:
- wire axis: k = 2 D_T / v;
- time axis: k = 2 D_L / v³.

Then a data/training mismatch in a drift-independent width c (filter kernel, deconvolution residual, response)
shifts the prediction by Δc² / k_train at every drift. That is an **intercept** change.

Only two things change the **slope**:
- a mismatch in k;
- an effect whose size grows with drift.

The observed pair, slope 0.42 with intercept +50 cm, therefore needs a slope-type cause. Constant-width terms can
account for the intercept.

The CNN is not a linear σ² inverter: it saturates, and it reads absolute amplitude by design (doc 98 §1). This frame
therefore has to be tested before it is relied on; study S1 in section 4 does that.

### 2.1 SP software filters (candidate 3)

Only one SP setting differs between the training chain and the data chain: the collection **wire** filter `Wire_col`.
The time filters (`Gaus_wide`, `Wiener_*_W`) are identical (doc 98 §7).

The table gives the real-space kernel. It is the inverse DFT of the array `HfFilter::filter_waveform` builds, with
`use_negative_freqs` defaulting to true (`HfFilter.h:28`) and frequency in Nyquist units.

| chain | `Wire_col` sigma | h0 | h±1 | rms |
|---|---|---|---|---|
| training (`dune-vd/sp-filters.jsonnet:114`) | 3.0/√π | 0.945 | 0.033 | 0.19 wire = **0.95 mm** |
| data PDVD and PDHD | 10/√π | 0.995 | 0.003 | 0.06 wire = **0.29 mm** |

The training images are smoothed across wires by an extra ~0.9 mm (in quadrature), constant in drift.
- **Under section 2.0 this is an intercept term.** The data image is sharper, so the model should read it as less
  diffused: a negative intercept offset of order 0.9² mm² / k_train. With k_train = 2 × 0.7 × 8.8 cm²/s / v_train = 0.0077 mm²/cm
  that is about −100 cm under the linear frame. It is a large term, not a small one, which is why S1(i) and S4 are worth doing.
- **It is not a slope term.**

Doc 98 §9 item 1 named "re-SP the data with Wire_col 3.0" as the leading test of the compressed **slope**. It is
re-rated here as a test of the **intercept**; its prediction is written down in S4.

### 2.2 Field response and the constant width in the track fit (candidate 2)

**The field response file is the same.** The PDVD data use `protodunevd_FR_imbalance3p_260501`, which is also the
training simulation's file (doc 98 §7).

**The position dependence within a pitch averages out.** Doc 47 withdrew the "sub-pitch phase" mechanism: corrected,
simulated edge/centre is 0.95-1.03 on every plane (doc 47 §12.4).

**The constant width the track fit needed is real, and on collection part of it is data-only.** The share-matched
constant c, in mm, U / V / W:

| | PDVD | PDHD |
|---|---|---|
| data | 2.30 / 2.30 / 1.18 | 3.21 / 2.72 / 1.46 |
| sim | 1.29 / 1.27 / 0.54 | 2.69 / 2.79 / 0.39 |

(doc 47 l. 425-426.)

The data-only excess on the plane this model reads (W) is 1.05 mm on PDVD and 1.4 mm on PDHD.

**A response mismatch cannot make it on collection.** Two accepted response files that differ 1.7-1.9× at ±1 wire put
+0.7 / +1.7 mm on U / V and nothing on W (doc 47 §12.2). The candidates doc 47 leaves open are physical charge sharing
and capacitive cross-talk.

**It is constant in drift.** Under section 2.0 it is another **intercept** term, in the direction opposite to the
filter: a wider data image reads farther.

The filter's ~0.9 mm and this ~1.05 mm are the same order and partly cancel on W. That is an order-of-magnitude remark
only: it adds a kernel rms and a share-matched sigma in quadrature, and doc 44 shows those estimators differ by ~20 %
on this peaked profile.

### 2.3 Diffusion (candidate 1)

**Wire axis (D_T).** The data value is the drift-resolved width in doc 44, which is the same measurement that produced
the track-fit constant:

| population | D_T,eff (cm²/s) | source |
|---|---|---|
| PDVD W plane alone | 4.83 ± 0.6 | doc 44 l. 150, per-plane |
| top CRP (x > 0, anodes 4-7), joint over planes | 6.14 ± 0.64 | doc 44 l. 220 |
| bottom CRP (x < 0), joint over planes | 3.16 ± 0.74 | doc 44 l. 221 |

**Most of the data's low D_T,eff is an SP effect, not diffusion.** On simulated tracks, PDVD's SP chain returns 70 %
of the configured D_T. The width it hands out grows more slowly than √t, because the ROI trims the tails harder at
larger σ (doc 47 l. 432-435). The data return 62 % of the configured 7.91, so data/sim is about 0.89.

**The training chain's own returned D_T,eff has never been measured.** It ran D_T 8.8 through different filters and
noise. The data/training ratio of k (`d117_wire_kernel.py`) is therefore conditional on an assumption:

| training SP returns | PDVD W | top CRP | bottom CRP |
|---|---|---|---|
| 70 % of 8.8 (as PDVD SP does) | 0.85 | 1.08 | 0.56 |
| 100 % of 8.8 | 0.60 | 0.76 | 0.39 |

**Time axis (D_L).** Data D_L has **never been measured** on ProtoDUNE here. The params value (4.13 at 0.45 kV/cm,
v 1.48) against the training value (4.0 at v 1.606) gives k_data / k_train = 1.32. It pushes the slope **up**, which is
the wrong direction (doc 98 §7).

**The volume split points the same way, but weakly.** Doc 98 §11 splits the common in-range objects by volume:
- top: slope 0.45, n = 29;
- bottom: slope −0.13, n = 8.

The bottom CRP is also the face with the low D_T,eff. With n = 8 the bottom slope is noise, so this doc does not lean
on it.

### 2.4 Summary table

| candidate | intercept or slope (section 2.0) | evidence | verdict |
|---|---|---|---|
| SP wire filter `Wire_col` 3 vs 10 | intercept (data sharper, reads nearer) | kernel 0.95 vs 0.29 mm rms | real difference; cannot by itself explain the slope |
| FR / position-dependent FR | intercept | same FR file; sub-pitch phase withdrawn; FR mismatch does not reach W | not a slope cause on W |
| data-only collection excess (the fit's constant width) | intercept (data wider, reads farther) | +1.05 mm PDVD W, 9σ statistical | real, mechanism open (sharing or cross-talk); partly cancels the filter term |
| D_T (wire axis) | **slope** | data W 4.83; training returned value unmeasured | can supply 0.6-0.85 of response on the whole sample; ~1 on the top CRP if training returns 70 % |
| D_L, v (time axis) | slope, but upward | params only, 1.32 | wrong sign; unmeasured on data |
| top gain scale ×1.125 | small | in/out on the same crops, 0.05 of slope (doc 98 §11) | measured, minor |

## 3. The honest negative, and the likelier candidates

The top CRP holds 29 of the 37 in-range Michels common to both arms. There, the measured D_T and the params-level D_L
predict a response of about 1 or above; the observed value is 0.45.
- **None of the three named candidates, in its simple form, explains the top slope.**
- Diffusion explains part of it only if the training chain returns nearly all its configured D_T, and that is unmeasured
  (S3).

What can compress a slope is an effect that **grows with drift**. The following are hypotheses, each with the record
that motivates it:

- **a. ROI thresholding on the diffused tails.** Doc 47 shows the returned D_T,eff falls short *because* the ROI trims
  more at larger σ. The data's signal-to-threshold is 1.5× worse than simulation (doc 47 §12.2), so the trimming bites
  harder on the faint, far Michels. Their images look less diffused than they are.
  - Doc 47's "the ROI is not it" was about the constant c, not about k.
  - The training chain has its own noise model and thresholds, tuned on different noise.
- **b. Noise and charge fluctuation.** The training simulation has no charge fluctuation. Its noise is
  `pdvd-top-noise-spectra-v3`, not the real noise. Structure that does not scale with drift reads as "sharp". Its
  relative weight grows as the diffused peak drops, which is at far drift.
- **c. Electron lifetime.** Training is at infinite lifetime; PDVD is at 20 ms, about 10 % loss at 300 cm. The model
  reads absolute amplitude (`log1p(x)/5`, no per-sample scaling), so a drift-dependent amplitude loss is a
  drift-dependent input change. Its sign on mu has to be measured, not assumed.
- **d. The Michel domain shift.** The data Michel has its start truncated by the muon mask and exact zeros around it.
  The training electrons were isolated, with noise residue.
  - On the **same labels**, the muon-left-in crop reads slope 0.66 and the Michel-only crop 0.42 (doc 98 §6).
  - So the Michel crop carries most of the compression, and flash-t0 label noise is not its main source.
  - The mask width (Zw) and the zeroing choice (S) move the slope by < 0.1 (doc 98 §7).
  - What remains untested is "zeros where training had noise residue".
- **e. The energy dependence.** Michels ≥ 20 MeV respond at about twice the slope of the softer ones (0.46 against 0.23
  on PDVD latest, doc 98 §6b). That fits a–c, all of which hurt low-amplitude images most.

## 4. Proposed studies (future sessions), cheapest and most discriminating first

Every study reads the same numbers:
- slope, intercept and rho on doc 98's in-range tier-B set (PDVD `latest_sw99`, n = 56; PDHD n = 28);
- the same on the matched-simulation draws.

Each new run gets a new tag under `scan/d117/`; doc 98's records are not written into.

**S1. In/out tests on the existing crops.** No reprocessing, `d98_rescale_check.py` style. Each test is applied to both
the data crops and the model's own test-split crops.
- **(i) Constant wire kernel.** Convolve the training test crops with (H10/H3), and separately with a 1.05 mm Gaussian
  (the data excess).
  - Prediction under section 2.0: the intercept moves and the slope stays.
  - If the slope moves, the frame fails and the constant-width terms are back in play as slope causes.
- **(ii) Drift-proportional wire smear.** Add one to the data crops with σ² = (k_train − k_data)·x, to restore k.
  This says how much slope the D_T gap is worth to the network.
- **(iii) Lifetime.** Multiply the data crops by exp(+t/20 ms) (undo it), and the sim crops by exp(−t/τ).

Caveat: the crops are post-ROI, so (i) and (ii) cannot restore tails the ROI removed. That limit is what S2 is for.

**S2. One-knob-at-a-time ladder in the model's own simulation** (`dunevd_singlep` stage A/B, new work dirs, the model
fixed). Start from the training configuration and move toward data one step at a time. Score each rung with the
matched-draw readout, and see where 0.91 falls toward 0.42.
1. `Wire_col` 10/√π;
2. D_L / D_T / v at 0.45 kV/cm: 4.13 / 7.91 / 1.48;
3. PDVD data noise spectra, at the data's signal-to-threshold;
4. 20 ms lifetime;
5. charge (recombination) fluctuation on;
6. exact-zero background outside a Michel-like mask, with a truncated start (muon-removal emulation).

**S3. Measure the training chain's own k and c.** Run doc 47's xtrack method (straight tracks at tan θ = 0.30,
`DepoFluxSplat sparse=true`, the `d44_sigma_fit.py` estimator) on the **training** SP configuration, W plane. This turns
the conditional diffusion ratio of section 2.3 into a number, and it can be done without the network.

**S4. Re-SP the data with `Wire_col` 3.0/√π.** This is doc 98 §9.1, re-rated: a new arm, with no production change.
- Written down in advance: the intercept moves (the data image widens, so it reads farther) and the slope stays within
  its interval.
- A slope change would falsify section 2.0.

**S5. Charge sharing or cross-talk** (doc 47 §12.5). Re-run `run_nf_sp_evt.sh -R` keeping the `raw` tag on the four PDVD
events. Read the ±1 collection neighbour's post-NF waveform under a large centre signal:
- unipolar means charge sharing, which is physical and belongs in the simulation;
- derivative-shaped means cross-talk.

This settles what to add to S2 for the constant width.

**S6. Measure D_L on data.** It is the only transport constant used here that was never measured. Use the time-axis
width against drift on anode-cathode crossers (known t0), with the same share/rms caveats as doc 44.

**S7. Fine-tune or recalibrate on data** (doc 98 §9.3). Only do this if S1-S2 show a pure domain shift. Use the 56 PDVD
in-range Michels with a held-out third.

**What would settle the question.**
- If S2 rungs 3-5 (noise, lifetime, fluctuation) carry the slope drop and S1(i) leaves the slope alone, the answer to
  the slide's two bullets is: "SP-chain mismatch = the intercept; data/simulation mismatch in noise and fluctuation =
  the slope".
- If S3 shows the training chain returns about 100 % of its D_T, diffusion (section 2.3) carries a large part of the
  slope instead.

## 5. What this doc does not claim

- It does not claim a cause. Section 2 is budget arithmetic under a linear-inversion assumption that S1 tests.
- The quadrature comparison of the filter kernel with the fit constant mixes estimators (a kernel rms against a
  share-matched sigma). It is an order-of-magnitude statement only.
- The per-face D_T,eff values are joint over the three planes; only the all-face value is W-only.
- The bottom CRP's regressor slope (n = 8) is not interpreted.
- PDHD also differs in pitch (4.792 against 5.100 mm) and D_L (6.2); nothing here is PDHD-specific.

## 6. Round 2: study S1 on the restored latest-arm crops

### 6.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
./d117_run_sp.sh <61 events>          # production-default SP -> work/<evt>_d117sp (setarch -R, 6 jobs); list = events of
                                      # scan/d117/latest/candidates.tsv; logs in /home/xqian/tmp/d117/sp_logs
python3 d117_cell_identity.py <61 events>        # -> scan/d117/latest/cell_identity.txt
S=../../scan
python3 d117_crops.py --det pdvd --run ../d117/latest --pdvd-arm p98vonq \
    --pdvd-record $S/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json -j 8   # scan/d117/latest/, crops /home/xqian/tmp/d117/latest/
CUDA_VISIBLE_DEVICES=1 python3 d98_predict.py --run ../d117/latest --det pdvd  # scan/d117/latest/scores.tsv
CUDA_VISIBLE_DEVICES=1 python3 d117_s1.py --run latest > $S/d117/latest/s1_paired.txt
# round-1 check: python3 d98_michel_crops.py --det pdvd --run ../d117/r1regen -j 8; d98_predict.py --run ../d117/r1regen --det pdvd
```

### 6.1 Restoring doc 98's inputs

The crops in `/home/xqian/tmp/d98/` had been swept. The SP frames of the latest arm (`p98von`) and of today's
production (`pvdimg`) had been removed by the disk-cleanup rounds.

- **Round 1's frames still exist for 24 of the 120 events** (`d27fresh`). Doc 98 §0 says the July frames were
  retired, which is wrong for those 24 events. Its 22 Michels rebuild and re-score identically: max |Δ| 0 on mu_Z and
  drift against `scan/d98/scores.tsv` (`scan/d117/r1regen/`).
- **The latest arm's frames were regenerated.**
  - Production-default SP reproduces `p98von` (doc pdvd/100 F2). It was run for the 61 events that hold doc 98's
    latest-arm candidates, into `work/<evt>_d117sp`. That is 21 GB, not protected from the next cleanup round.
  - **Check:** every frame-measured cell of `p98vonq`'s `T_stm_michel_2d` (roles 1/3/4, `flag == 1`) re-sums
    exactly from the new frames, on **61/61 events** (`scan/d117/latest/cell_identity.txt`).
  - Two kinds of row are not frame sums and are excluded:
    - role-0 region rows, whose `time` steps tick by tick;
    - `flag == 0` rows: induction-plane cells on channels whose whole frame row is empty.
    Doc 98 §2.1's "charge == frame sum" therefore holds for `flag == 1` only.
  - `039349_27` exited rc=2 after all eight frame sinks had closed (a teardown fault). Its frames pass the check.
- **Doc 98's latest arm is restored exactly.** 80/80 crops, 56 in range; max |Δ| 0 on mu_Z, mu_M, drift_tick,
  frac_lost and n_pix against `scan/d98/latest_sw99/scores.tsv`.

### 6.2 What S1 does

`d117_s1.py` re-scores two image sets under each transform, using the published bf16 CUDA function:
- **data:** the 56 in-range Michel-only (Z) crops;
- **simulation:** the 3,399 test-split electrons that neighbour a data Michel (±15 cm, ±5 MeV).

**Paired design.** The 1,000 matched simulation draws and the 1,000 data bootstrap resamples are fixed once and
reused for every transform. Each Δ is transform minus none, on the same draws, quoted as median [16, 84].

**Closure.** Against the published predictions, data 0.05 cm and simulation 0.005 cm.

Baseline: data slope 0.421; simulation 0.912 [0.857, 0.963].

### 6.3 Results (`scan/d117/latest/s1_paired.txt`)

| transform | what it emulates | sim Δslope | sim Δintercept (cm) | data Δslope | data Δmu (cm) |
|---|---|---|---|---|---|
| gt1.0\|sp, gt2.0\|sp | a constant tick blur (1, 2 ticks), support kept | +0.085, +0.04 | +28, +136 | +0.11, +0.21 | +18, +75 |
| filt3/10\|sp | training / data `Wire_col` ratio (S4 without ROI), support kept | **+0.121 [0.089, 0.157]** | +13 | **+0.112 [0.071, 0.154]** | +19 |
| gw0.21 | a constant 1.05 mm channel blur (the data-only W excess) | +0.009 | +2 | +0.011 | +2 |
| gw0.5\|sp | 2.5 mm channel blur, support kept | +0.40 | +66 | +0.27 | +91 |
| dT2, dT4, dT8 | drift-proportional transverse diffusion, +2 / 4 / 8 cm²/s | 0.000, +0.080, +0.015 | ≈ 0 | * | * |
| dL2\|sp, dL4\|sp, dL8\|sp | drift-proportional longitudinal diffusion | +0.22, +0.37, +0.48 | small | +0.24, +0.40, +0.57 * | * |
| sup_t1 / sup_t3 / sup_c1 (1e-3 e) | the non-zero footprint grown by 1-3 ticks or 1 channel, charge untouched | 0.000 | 0 | 0.000 | 0 |
| sup_t2 (50 e) | a 2-tick ring at 50 electrons | +0.008 | +3 | −0.07 | −6 |
| floor1e-3 | 1e-3 e on all 262k pixels | −0.14 | +113 | +0.08 | +85 |
| tau20 | a 20 ms lifetime applied to sim, undone on data | +0.037 | −4 | −0.028 | −2 |
| zbkg | sim: exact zeros outside the dilated main component | −0.040 [−0.085, 0.002] | +10 | – | – |

`*` The data blur is built from the drift label. Its slope change is injected by construction, so only its size
against the simulation's is read.

What these say:

1. **The model reads charge shape, not the non-zero footprint.**
   - Growing the footprint at 1e-3 electrons moves nothing.
   - Blurs that keep the original support still move the model.
   - The `floor1e-3` effect comes from a pedestal integrated over the whole crop, not from the edge. It says the
     network is fragile to a whole-image pedestal; no data effect is known to supply one.
   - Hypothesis (a) of section 3, ROI trimming, can therefore act only through the charge shape it leaves, not through
     the footprint.
2. **Section 2.0's frame holds only approximately.**
   - Constant widths move mostly the intercept.
   - Constant widths also raise the slope by +0.04 to +0.12 at plausible sizes, because the network is not a linear
     inverter.
   - Drift-proportional time width moves the slope strongly, by about +0.1 per cm²/s of D_L.
3. **The channel axis is nearly dead at the relevant size.**
   - Channel blurs of ≤ 0.2-0.3 wire do nothing: gw0.21 and dT2-dT8 reach 0.28 wire at 200 cm.
   - A 0.5-wire channel blur moves the intercept by +270 cm: a threshold response.
   - So the transverse-diffusion difference of section 2.3 is not a lever.
4. **The wire-filter difference is worth +0.11 of slope.** The emulated re-SP with `Wire_col` 3.0 takes the data
   slope 0.42 → 0.53 and mu up about 20 cm. It is the right direction and a fifth of the 0.49 gap.
   - This is S4's prediction.
   - The emulation cannot include the ROI stage; the real re-SP arm is the test.
5. **The data images are not insensitive.** Per unit of injected longitudinal diffusion, the data respond as the
   simulation does (+0.24 against +0.22 at 2 cm²/s).
6. **Lifetime and background are small.**
   - The lifetime effect is under 0.04, and on data its sign is wrong for a cause.
   - The exact-zero background costs the simulation −0.04 [−0.085, 0.002].

**Model-free time width (section C of the file) is not a lead yet.**

| sample | Δw2 per 100 cm |
|---|---|
| doc 98's estimator, all 79 data crops | +0.90 [0.31, 1.55] (doc 98's number, reproduced) |
| doc 98's estimator, the 56 in-range data crops | −0.22 [−1.20, +0.86] |
| matched simulation | +1.07 [0.30, 1.70] |
| naive expectation, data parameters | +1.02 |

- Doc 98's positive slope comes from the Michels below 80 cm.
- In range, data and simulation differ by about 1σ.
- The Zw mask gives the same numbers, so the mask width is not what sets them.
- A second variant (5 % of the total charge per channel, section B) gives −0.70 [−1.81, +0.42] against +2.15 [0.98,
  3.13]: about 2σ.
- Neither variant is a measurement at n = 56. S6, D_L from long tracks, is the one with the statistics.

## 7. Corrections to sections 2-4 after S1

- **§2.3 and §2.4, D_T.** The k_data/k_train ratios of 0.85 / 0.60 are not a slope budget for this model. The
  network does not respond to transverse broadening of that size (6.3, item 3). The D_T row of the §2.4 table should
  read "channel axis ≈ dead at this size; not a lever".
- **§2.1, the wire filter.** It is not a pure intercept term. Emulated, it raises the slope by +0.11 [0.07, 0.15] and mu
  by about 20 cm. It is the one measured lever of the three named candidates.
- **§2.2, the data-only collection excess.** A 1.05 mm channel blur moves nothing (gw0.21). It is not a lever either
  way.
- **§3.** The "top CRP predicts about 1" argument used the D_T ratio. With the channel axis dead, the top/bottom D_T
  difference cannot explain a top/bottom difference either. The remaining gap, 0.42 + 0.11 (filter) against 0.91,
  about 0.38, needs a time-axis or charge-shape cause:
  - D_L on data (never measured; S6);
  - SP time-domain effects that grow with drift (hypothesis a, through the shape);
  - the Michel domain shift (hypothesis d, the Z against M gap).
- **§4, the next studies, reordered.**
  1. **S4, the real re-SP arm with `Wire_col` 3.0.** It checks the +0.11 with the ROI stage included.
  2. **S6, D_L on data.**
  3. **S2, the simulation ladder.** Its most informative rungs are now the time-axis ones: D_L/v at 0.45 kV/cm, the
     data noise and S/N, and charge fluctuation.
  4. **A Michel-domain test.** Apply the data's muon-removal geometry (truncated start, zeroed overlap) to simulated
     electrons, and see whether that alone compresses the slope.

## 8. Round 3: study S4, the real re-SP with the training wire filter

### 8.0 Repro

```bash
# toolkit 377119ee: protodunevd sp.jsonnet / wct-nf-sp-dnnroi.jsonnet TLA wire_col_sigma_x (default null)
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
ARM=d117w3sp EXTRA="--wire-col-x 3.0" ./d117_run_sp.sh <61 events>            # work/<evt>_d117w3sp
S=../../scan
SP_ARM=d117w3sp python3 d117_crops.py --det pdvd --run ../d117/w3 --pdvd-arm p98vonq \
    --pdvd-record $S/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json -j 8   # scan/d117/w3/
CUDA_VISIBLE_DEVICES=1 python3 d98_predict.py --run ../d117/w3 --det pdvd
CUDA_VISIBLE_DEVICES=1 python3 d117_s4.py > $S/d117/w3/s4.txt
```

**The knob.** It is new, and default OFF.
- `make_sigproc(wire_col_sigma_x=null)` in the toolkit's `protodunevd/sp.jsonnet`, threaded through the in-tree
  `wct-nf-sp-dnnroi.jsonnet`. The runner's `--wire-col-x X` sets it.
- When set, the collection slot of `Wire_filters` names a filter pair (`Wire_colx_b/_t`, σ = X/√π) that is registered
  only then.
- **Compiled-config proof** (`/home/xqian/tmp/d117/cfg/`):
  - off: byte-identical to the pre-edit job, both bare (md5 `966e5b3e`) and at a runner TLA set (`32040cb5`);
  - on (X = 3.0): the two new `HfFilter`s with σ 1.6926, and `Wire_filters` changed on all eight `OmnibusSigProc`s.
    Nothing else differs.
- No C++ changed and no production default moved.

**The arm.** `work/<evt>_d117w3sp`: 61 events, all rc 0, 8/8 frames, libraries unchanged during the run.
- **The Michel masks and drift labels are p98vonq's on both arms** (max |Δ label| 0.1 cm), so only the SP pixels differ.
- **Side effect on the induction planes.** They are not bit-identical (sum 0.1-0.2 % on 039252_0), because DNN-ROI
  and L1SP downstream see the changed collection plane. The crops use W only.
- **On W** (039252_0): charge −0.7 to −1.0 %, non-zero pixels +7 to +10 %.

### 8.1 Result (`scan/d117/w3/s4.txt`; 56 in-range Michels, paired on 2000 shared bootstrap resamples)

| | production SP | re-SP, `Wire_col` 3.0 | Δ (paired) | S1 emulation (predicted before this ran) |
|---|---|---|---|---|
| slope, Z, all | 0.421 | **0.525** | **+0.105 [0.060, 0.151]** | +0.112 [0.071, 0.154] |
| intercept | 53.3 | 60.2 | +6.7 [−1.2, 14.3] | +3.8 |
| median Δmu | | | +23.9 cm | +19.3 cm |
| rho | +0.60 | +0.54 | | +0.57 |
| slope, Z, top (n 41) | 0.407 | 0.529 | +0.121 [0.064, 0.181] | +0.138 |
| slope, Z, bottom (n 15) | 0.486 | 0.515 | +0.030 [−0.017, 0.088] | +0.042 |
| slope, M (muon left in) | 0.662 | 0.785 | +0.122 [0.090, 0.160] | – |

- **The prediction holds.** The real re-SP moves the slope by the emulated +0.11 and mu by about 20 cm. Per Michel the
  two agree at r = 0.89 (median |Δ| 4.3 cm).
  - **Pixels.** The real re-SP and the emulation each differ from production by 3.6 % of the crop charge (L1 /
    charge, 52 crops cut at the same origin), and from each other by 0.6 %.
  - **The ROI stage adds little** beyond the filter itself.
- **It moves the slope more than the intercept.** Doc 117 §2.1 predicted the opposite, and §7 already corrected it
  from S1.
  - rho does not improve (0.60 → 0.54, within noise). The response grows, but so does the scatter about it.
- **About 0.39 of the gap remains.** 0.525 against the simulation's 0.91.
  - The wire filter is a real, identified part of the SP mismatch, about a fifth of the gap.
  - It is not the main part.
- **The collection wire filter is a production SP setting.** Changing it for data would be a reconstruction change for
  every consumer, and nothing here argues for that. For the regressor it is simpler to retrain with the data's
  `Wire_col` 10, or to re-SP only the regressor's input. Both are the owner's call.

### 8.2 Where this leaves the list

The two named SP-side candidates are now measured:
- the wire filter, +0.11 (S1 emulation, S4 real);
- the constant collection excess, ≈ 0 (S1 gw0.21).

What remains is on the time axis or in the Michel image itself. Next, in order:
1. **S6, D_L on data** from long tracks. It is the one unmeasured transport constant, and the model is most sensitive
   to it: about +0.1 of slope per cm²/s.
2. **The Michel-domain test.** Apply the data's muon-removal geometry (truncated start, zeroed overlap cells) to
   simulated electrons.
3. **The S2 time-axis rungs:** D_L/v at 0.45 kV/cm, data noise and S/N, charge fluctuation.

**Disk.** `work/*_d117w3sp` takes about 21 GB. It falls under the `d117*` prefix in `pdvd/scripts/retire/PROTECTED.txt`.

## 9. Round 4: study S6, D_L on PDVD data

### 9.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
S=../../scan/d117/s6
python3 d117_s6_data.py -j 8 > $S/s6_data.txt     # data: 61 events, d117sp frames + p98vonq fits -> s6_data_points_d117sp.tsv.gz
# simulation: isochronous tracks, anode 0 face 0 (the tracks/truth JSONs are copied into $S)
python3 d117_s6_tracks.py --cfg <compiled wct-sim-xtrack-sp.jsonnet, lar_drift 1.48073> --anode 0 --face 0 --outdir /home/xqian/tmp/d117/s6sim
./d117_run_s6sim.sh                                # DL0/DL4/DL8 x seeds 1,2 + DL4 noise-off (+ DL4q3: --charge -1500), libpin copy of local/lib
python3 d117_s6_sim.py --arms DL0_s1,DL0_s2,DL4_s1,DL4_s2,DL8_s1,DL8_s2,DL4_nn,DL4q3_s1,DL4q3_s2 > $S/s6_sim.txt
# top CRP (its own electronics response): the same with --anode 4 --face 0 tracks, anode_index=4 arms in .../s6sim/top
python3 d117_s6_sim.py --dir /home/xqian/tmp/d117/s6sim/top --anode 4 --arms DL0_s1,DL0_s2,DL4_s1,DL4_s2,DL8_s1,DL8_s2 > $S/s6_sim_top.txt
# NOTE: /home/xqian/tmp/d117/libpin and the simulation frames in /home/xqian/tmp/d117/s6sim are swept scratch;
#       the tracks/truth JSONs (both volumes) and every summary are committed under $S
# the data splits: s6_data_splits.txt (summarize() on subsets of the points file)
```

### 9.1 The estimator (identical on data and simulation)

- **Sample.** Collection-plane pulses on fitted trajectory points. Every fitted cluster in p98vonq carries a flash t0;
  drift = 341.55 − |x| cm, from the t0-corrected fit.
- **Selection.** Only points where the track crosses the W wires at under 3 ticks per wire.
  - PDVD drifts vertically, so cosmics are mostly steep in x. Under 1 tick per wire leaves 15 points per event.
  - The track's own extent, a box of tpw ticks, is subtracted from each variance as tpw²/12.
- **The pulse.** The contiguous ROI run holding the peak, isolated: nothing above 10 % of the peak within ±25 ticks.
- **Two variances.**
  - over the whole ROI run (`vc`);
  - within ±6 ticks of the peak (`vc6`), which is less tail-sensitive but compresses the diffusion signal.
- **Line.** σ_t² = c + k·t through the medians of 8 equal-population bins of drift time. The error is a per-event
  bootstrap. D_L,eff = k·v²·(0.5 µs)²/2.
- **Data yield.** 61 events, 124k fit points: 10,705 measured, 8,910 isolated. The median drift time is 497 µs
  (74 cm); q10 / q90 are 179 / 1360 µs.

### 9.2 Calibration on simulation (`s6_sim.txt`)

Setup:
- `pdvd_sim/wct-sim-xtrack-sp.jsonnet`, sim → noise → NF → production SP, anode 0 face 0, v = 1.48073 mm/µs;
- 24 isochronous tracks, 12 drifts from 25 to 320 cm, each flat and tilted to 1.5 ticks per wire;
- 552 pulses per arm, all isolated.

| true D_L (cm²/s) | `vc` D_L,eff | `vc6` D_L,eff | c (ticks²) |
|---|---|---|---|
| 1e-4 | 0.32 [0.20, 0.53] | 0.15 | 7.27 |
| 4.1307 | 4.16 [3.89, 4.48] | 2.10 | 7.16 |
| 8.2614 | 7.59 [7.30, 8.01] | 3.58 | 7.10 |
| 4.1307, no noise | 4.13 [4.02, 4.26] (tpw < 1) | 2.21 | 7.08 |
| 4.1307, 3× charge (every peak ≥ 8000 e) | 4.03 [3.84, 4.21] | 2.16 | 7.01 |

**The response is linear.**
- `vc`: D_L,eff = 0.39 + 0.880·D_L.
- `vc6`: D_L,eff = 0.23 + 0.415·D_L.
- Noise, pulse amplitude and the box correction do not bias it: the tpw < 1 and tilted halves agree.

**Top CRP** (anode 4 face 0, its own electronics response, `s6_sim_top.txt`):

| true D_L (cm²/s) | `vc` D_L,eff | `vc6` D_L,eff | c (ticks²) |
|---|---|---|---|
| 1e-4 | −0.19 | 0.05 | 7.16 |
| 4.1307 | 3.80 | 2.00 | 6.86 |
| 8.2614 | 7.37 | 3.64 | 6.84 |

- `vc`: D_L,eff = −0.12 + 0.915·D_L.
- `vc6`: 0.10 + 0.435·D_L.
- Every top-CRP number below uses this calibration, and every bottom-CRP and all-sample number the one above.

### 9.3 Data (`s6_data.txt`, `s6_data_splits.txt`), calibrated

| subset | isolated | `vc` → D_L | `vc6` → D_L |
|---|---|---|---|
| **all** | 8,799 | **4.9 [3.3, 6.8]** | **4.9 [3.5, 6.9]** |
| top CRP (anodes 4-7), top calibration | 7,559 | 3.4 [2.2, 5.8] | 3.9 [2.2, 6.9] |
| top CRP, tpw < 1 (least box-dependent) | 4,460 | 2.7 [1.9, 4.9] | 3.1 [1.8, 6.2] |
| bottom CRP | 1,240 | 4.5 [1.8, 8.4] | 2.5 [0.0, 5.5] |
| drift ≥ 50 cm | 5,847 | 6.8 [4.2, 8.5] | 6.4 [4.9, 8.2] |

Uncalibrated, other subsets range from 1.3 to 10: peak ≥ 8000 e reads 1.5-1.7, bottom tpw < 1 reads 10.

**The honest reading.**
- **Central value.** The full sample gives D_L = 4.9 cm²/s on both estimators, consistent with the params value 4.13.
  Doc pdvd/25 §13.1's 4.12 is an adopted model value, not a measurement.
- **Systematic.** The spread across reasonable subsets is about ±2 cm²/s, larger than the bootstrap interval.
- **The binned medians are not monotonic.** There is a bump at 600-900 µs, and the lever arm is short. The
  simulation, with uniform drifts and isolated tracks, has neither.
- **The tpw dependence is data-only.** On the top CRP, tpw < 1 reads 2.4 (uncalibrated) and tpw 2-3 reads 4.8, while in
  simulation the two halves agree. tpw does not correlate with drift in the data (top r = −0.045), so this is a
  data-side systematic of the box correction on real tracks, not a lever-arm artefact. tpw < 1 is quoted as the
  cleanest number.
- **The data's worse S/N** (1.5×, doc 47) can only trim diffused tails harder than the noise-on simulation does. If
  anything the data's physical D_L is underestimated, which makes the bound of 9.4 conservative.
- **The low high-amplitude subset is not an estimator effect** (the 3× charge arm reads 4.03). What it is, a data
  population effect (δ-rays, dense topology) or chance, is not resolved here.
- **This is the first D_L measurement on ProtoDUNE-VD in this tree.** It is a consistency check, not a precision
  number.

**The constant term, matched volumes** (`vc`):
- top: data 6.73 against top simulation 6.84-7.16 (0.1-0.4 ticks² narrower in data);
- bottom: data 8.06 against bottom simulation 7.10-7.27 (about 0.8 wider in data).
Both are small next to the constant-blur sizes S1 found the regressor reacts to (1-2 ticks, i.e. 1-4 ticks²). They
are not interpreted further.

### 9.4 What it says about the regressor

The regressor reads the time-width growth per cm of drift, k ∝ D_L / v³.
- **Training simulation:** D_L 4.0 at 1.60563 mm/µs. At the data's 1.48073 mm/µs the same growth is D_L = 3.14 cm²/s.
- **Data:** D_L 4.9 overall. On the top CRP, where 29 of the 37 common Michels sit, it is 3.4 [2.2, 5.8], and
  2.7 [1.9, 4.9] for tpw < 1. The central values are at the training-equivalent, or above it overall.
- S1 measured the regressor's response to added D_L: about +0.1 of slope per cm²/s, the same on data and simulation.
- **The bound.** The top CRP's 68 % lower bounds (2.2, and 1.9 for tpw < 1) sit up to 1.2 cm²/s below the equivalent.
  So D_L can account for at most about 0.1-0.15 of slope, and its central value accounts for about 0.

**D_L is not the main cause of the remaining ~0.39** (0.525 after S4, against 0.91).

**S6 and S1 together point at the Michel crop.**
- On muon tracks (S6), the data's time width grows with drift at the expected rate.
- On the in-range Michel crops (S1, section 6.3), it does not visibly grow: −0.22 [−1.20, +0.86] against the
  simulation's +1.07 [0.30, 1.70] ticks² per 100 cm (about 1σ).
- The drift is imprinted on the charge. What differs is how the Michel crop presents it.

What is left is not a transport or SP-filter effect:
- the Michel domain shift (hypothesis d: the truncated, muon-subtracted image with exact zeros around it; the muon-left-in
  crops read 0.66-0.79);
- charge-shape effects that do not grow with drift: noise and charge fluctuation, hypotheses (a)-(b).

**Next** (in order; item 1 ran in round 5, section 10):
1. **The Michel-domain test.** Apply the data's muon-removal geometry to simulated electrons (truncate the start, zero
   the overlap cells, exact-zero background) and re-score. This is the cheapest test of the largest remaining candidate.
2. **The S2 noise and fluctuation rungs.**
3. **If the domain shift carries it,** a fine-tune on data Michels (S7), or training on muon-subtracted simulated
   Michels.

## 10. Round 5: the Michel-domain test

### 10.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
S=../../scan/d117/dom
python3 d117_dom_masks.py -j 8 > $S/mask_closure.txt           # each data Michel's keep/muon masks in crop coordinates
                                                                # -> /home/xqian/tmp/d117/dom/masks_pdvd.npz (swept scratch)
CUDA_VISIBLE_DEVICES=1 python3 d117_dom.py > $S/dom.txt         # the arms on S1's sim draws + the data split
```

### 10.1 Pre-registration (written before the arms ran)

**The question.** A data Michel crop is the Michel with the muon removed:
- the start is truncated by the muon mask (±1 channel, ±4 ticks around the muon cells);
- everything outside the Michel's keep band is exact zero.

The training electrons are whole. On the same labels, the muon-left-in crop reads 0.66 and the Michel-only crop 0.42
(doc 98 §6). Does removal geometry alone, applied to the model's own electrons, compress their slope 0.91 toward the
data's?

**The prior: mostly an intercept.**
- Doc 98 §7: variant S, which restores the Michel's share on the overlap cells, moves the slope by < 0.1 against Z.
- The removed fraction barely correlates with drift. In-range, ρ(drift, `frac_lost`) is −0.12 and
  ρ(drift, `frac_overlap`) is −0.10. A removal that does not grow with drift should mostly move the intercept, though
  S1 showed this network is not a linear inverter.

**The criterion.** Read on the full emulation at the data's own removed fraction (zbkg + truncation, zbkg + the
literal mask):

| Δslope | reading |
|---|---|
| \|Δ\| < 0.1 | removal geometry is not the carrier; the gap passes to the S2 noise and fluctuation rungs (hypothesis b) |
| ≤ −0.2 | removal geometry is the carrier |
| between | partial |

**Scope.**
- This test covers removal geometry only. Noise and charge fluctuation are not tested.
- The store holds electrons and photons only, with no muons, so the muon-left-in (M) gap cannot be emulated.

### 10.2 Inputs and gates

- **Data masks** (`d117_dom_masks.py`). The worker runs doc 98's `build_one()` unchanged on the latest arm, wrapping
  `dilate()` so that the keep and muon masks it builds are captured. Both are cut at the crop's own centre.
  - **Gate:** `M·(keep & ~muon)` equals the stored Z bit for bit on **80/80** crops (`dom/mask_closure.txt`).
- **Simulation.** S1's 3,399 test-split electrons, with the same seed and the same 1,000 matched draws. Every arm is
  regressed on the data labels.
  - **Closure:** `none` reads 0.912 (S1: 0.912), max |mu − published| 0.005 cm. `zbkg` reads −0.040 [−0.085, +0.002]
    (S1: the same).
- **Pairing.** A draw standing in for data Michel *i* gets Michel *i*'s mask or removed fraction: 6,881 (Michel,
  neighbour) pairs.
- **Removed fraction (in range).**
  - `frac_lost` q10/50/90 = 0.075 / 0.241 / 0.571. It is a lower bound: the Michel's own cells only.
  - `frac_overlap` = 0.143 / 0.324 / 0.626. It is an upper bound, because it counts muon charge too.
- **Which end is the start?** The stored simulation crops carry no start point. The meta holds the centroid and the
  angles; the start would need the truth depos re-rasterised. So each truncation cuts from one end of the
  charge-weighted principal axis, and the two ends (a/b) are quoted as a **bracket**.

### 10.3 Results (`scan/d117/dom/dom.txt`)

Simulation slope, and its change against `none` on the same draws, median [16, 84]. "Removed" is the fraction of crop
charge the arm took away.

| arm | removed | sim slope | Δslope | Δintercept (cm) |
|---|---|---|---|---|
| none | 0 | 0.912 | – | – |
| zbkg (exact zeros off the main component) | 0.155 | 0.869 | −0.040 [−0.085, +0.002] | +10 |
| mask_lit (Michel *i*'s literal muon mask) | 0.064 | 0.832 | −0.079 [−0.130, −0.031] | +6 |
| trunc 0.2, a / b | 0.195 | 0.756 / 0.709 | −0.155 / −0.199 | +9 / +10 |
| trunc 0.4, a / b | 0.395 | 0.689 / 0.633 | −0.223 / −0.276 | +14 / +18 |
| trunc 0.6, a / b | 0.595 | 0.589 / 0.550 | −0.325 / −0.363 | +23 / +25 |
| trunc at Michel *i*'s `frac_lost`, a / b | 0.279 | 0.716 / 0.666 | −0.193 [−0.273, −0.119] / −0.246 [−0.322, −0.174] | +13 / +16 |
| trunc at Michel *i*'s `frac_overlap`, a / b | 0.351 | 0.682 / 0.633 | −0.228 / −0.276 | +16 / +19 |
| **zbkg + trunc `frac_lost`, a / b** | 0.39 | **0.643 / 0.576** | **−0.268 [−0.366, −0.168] / −0.329 [−0.433, −0.237]** | +25 / +29 |
| zbkg + mask_lit | 0.214 | 0.792 | −0.118 [−0.189, −0.054] | +16 |

**Data-side split** (no re-scoring; in-range slope of mu_Z, split at the median removed fraction):

| split | low half | high half | Δ (high − low) |
|---|---|---|---|
| `frac_lost` | 0.434 | 0.394 | −0.027 [−0.259, +0.224] |
| `frac_overlap` | 0.444 | 0.396 | −0.055 [−0.254, +0.181] |

At n = 28 per half this split is uninformative. The simulation dose response predicts a difference of about −0.1
between the halves, well inside that interval.

### 10.4 Reading

1. **The pre-registered criterion is met: removal geometry is a carrier.**
   - The full emulation at the data's own lower-bound fraction compresses the simulation slope by −0.27 to −0.33
     (the two ends), to 0.58-0.64.
   - Truncation alone, at the data's `frac_lost`, gives −0.19 to −0.25.
   - The prior ("mostly an intercept", from S against Z and ρ(drift, f) ≈ −0.1) is **wrong**. A drift-independent
     removal compresses the slope, and moves the intercept only by +10 to +30 cm.
   - This is another instance of S1's finding that the network is not a linear inverter: a truncated image reads as a
     nearer, less-diffused one.
2. **It is the start-truncation that does it, not the zeros.**
   - Exact zeros alone give −0.04.
   - The literal data mask removes only 6 % of the simulated electron's charge, because its orientation is random
     relative to the electron, and gives −0.08.
   - The response grows with the fraction removed: −0.16/−0.20 at 20 %, −0.22/−0.28 at 40 %, −0.33/−0.36 at 60 %.
3. **It matches the data's own M against Z gap.** On the same labels, the data crop with the muon left in reads 0.66,
   and the muon-removed crop 0.42: a gap of −0.24. The emulated truncation is −0.19 to −0.25 at the data's fraction.
   The mechanism now has a quantitative match on both sides.
4. **The budget now closes within its uncertainties.**
   - Emulated simulation 0.58-0.64, against data after the wire filter 0.525 (S4). What remains is about 0.05-0.12.
   - D_L can account for up to 0.1-0.15 of slope (S6), and noise and charge fluctuation are untested. Either fits in
     that remainder.
   - The terms are not strictly additive: the network is nonlinear, and each was measured on its own.

| cause | slope | source |
|---|---|---|
| wire filter (training `Wire_col` 3 against data 10) | +0.105 | S4 |
| **muon-removal truncation + exact-zero background** | **−0.27 to −0.33** (simulation emulation) | this round |
| D_L | ≤ 0.1-0.15 | S6 |
| D_T, collection excess, lifetime, footprint | ≈ 0 | S1 |
| remainder (noise, charge fluctuation, non-additivity) | ~0.05-0.12 | – |

**What this does not show.**
- **The direction.** Which end the data truncation removes is known: the start, where the muon stopped. The emulation
  brackets both ends because the start is not stored. The bracket is narrow (≈ 0.05), so the conclusion does not
  depend on it.
- **The dose.** `frac_lost` counts only the Michel's own cells under the muon mask. The true loss, including Michel
  charge the clustering assigned to the muon, lies between it and `frac_overlap`, and both give the same reading.
- **Noise and charge fluctuation.** They remain untested (S2 rungs 3 and 5).

### 10.5 Where this leaves the list

The slope gap is now mostly accounted for. Most of it is the **Michel domain** (muon-removal truncation), with the wire
filter second. It is not transport physics: D_L and D_T on data are consistent with the simulation.

For using the regressor on data, this points away from correcting the data and toward **training on what the data
looks like**:
1. **Retrain or fine-tune on truncated electrons.** Apply the data-like start truncation (the `frac_lost`/`frac_overlap`
   distribution) plus exact-zero background as training augmentation, with the data's `Wire_col` 10. This is cheaper
   and more principled than S7's fine-tune on 56 data Michels, which remains the check.
2. **Recover the start end in simulation** (re-rasterise the truth depos: first depo = start). It turns the bracket into
   one number. It is only worth doing if the augmentation needs it.
3. **The S2 noise and fluctuation rungs** for the remaining ~0.1, at low priority.

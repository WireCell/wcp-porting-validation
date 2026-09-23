# doc pdvd/117: why the DUNE-VD drift regressor reads slope 0.42 on ProtoDUNE Michels, and how to find out

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

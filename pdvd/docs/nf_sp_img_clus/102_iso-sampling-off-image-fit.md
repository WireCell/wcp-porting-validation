# 102 — Off-image fit stretches, a "broken" trajectory and the isochronous sampling grid (PDHD 028084_21; PDHD + PDVD)

**Status (2026-09-15).**

* **Three defects, three different layers (sec 4).**
  * On the owner's event, the fit leaving the charge at spots 1–2 comes from the coarse retile cloud.
  * Off-image chords in data at large come from long Steiner-graph edges under the STM rough path. They are
    **not** a fit or sampler effect: 93–95 % of chords have no retiled point under them (sec 5.3).
  * The ISO grid is the retile sampler's 3-wire step.
* **Port divergence (sec 6, M15).**
  * The toolkit PR retile samples with plain `stepped`.
  * The prototype retile used `calc_sampling_points`, whose toolkit port is `charge_stepped` (every wire when
    N_max·N_min ≤ 2500, extra wires above 4000).
  * Owner ruling: a port bug.
  * The knob `retile_sampler_strategy` is built default-OFF (toolkit `1b62d1da`): compiled PDHD/PDVD configs are
    byte-identical, and a new doctest pins its wrapped-plane charge on real PDHD geometry.
* **On the owner's event, `charge_stepped` fixes spots 1 and 2 (sec 3).**
  * Largest distance from the image ridge: 2.06 → 0.87 cm and 2.51 → 0.77 cm.
  * V on-charge 0.79 / 0.82 → 1.00.
  * No Bee zero-charge rows left.
  * Spot 2's PR segments 5 → 1.
* **It removes the grid.** PDHD 3-wire share 0.594 → 0.102; in-slice spacing 1.48 → 0.80 cm (PDVD 1.77 → 0.88 cm).
* **Pre-registered simulation verdict: no setting qualifies (sec 7).**
  * All three settings improve the ISO trajectory on both detectors: median Δ transverse p90 −0.49 to −1.36 mm,
    11–14 of 16 tracks improve.
  * All three fail **only** the no-new-segment clause c4.
  * Of the 8 flagged track-settings with a stub, 7 are ≤ 10-point stubs at the track start beside an improved
    main.
  * PDVD k11 is split at real off-line charge (on the image, 6.5 cm from the muon line).
* **Data (reported, not gating; sec 8).**
  * Resources pass: median wall +38 % PDHD / −2 % PDVD; median peak RSS +1.8 % / +2.2 %.
  * **PDHD Michel purity drops 0.919 → 0.839** (FP 5 → 10). STM efficiency is unchanged on PDHD (0.605) and
    −0.007 on PDVD.
* **No fit knob built (sec 9).** The dominant data mechanism is graph bridging, which a fit knob cannot reach.
* **NOT flipped.** No setting qualifies under the frozen rule. The c4 reading and the PDHD Michel purity drop are
  for the owner (sec 10).
* **Round 2 (sec 13): the owner asked to flip; the flip's own pre-registered gate FAILED, so it is still not
  flipped.**
  * Production runs the fit knobs OFF, so the flip configuration is knobs off + `charge_stepped`. It was measured
    under the frozen rule `figs/102_pred2.txt`.
  * **Passed:** resources, and the compile proof (the flipped default builds the measured config byte for byte).
  * **Failed: the STM/Michel tags.** PDHD STM efficiency 0.776 → 0.592. PDVD STM efficiency 0.877 → 0.754, Michel
    purity 0.923 → 0.792.
  * The sampler costs the tagger what doc 101's fit knobs cost it, in count and in kind. The tagger was tuned on
    the old trajectory.
  * **Fit knobs** are still effective under `charge_stepped`, in simulation and on data.
  * **SBND** has the same hard-coded `stepped` retile (no knob), with a cloud ~1.5× sparser than its image.

**Owner request (2026-09-15).** *"Hi, the track fitting has improved. Before we retune the tagger, from my scan
of the updated bee link, I see a few other issues worth improving. 1. for these points: (x, y, z) = (326.1,
457.4, 314.4) cluster = 128 or (x, y, z) = (-67.1, 569.9, 398.9) cluster = 55 or (x, y, z) = (-78.0, 571.8,
418.7)cluster = 55, the fitte track trajectory deviate from the 3D image somehow. I am not sure why, but clearly
when these happened, the fitted dQ/dx would not be correct. Also the deviated track trajectory can lead to
challenges in pattern recognition. 2. (x, y, z) = (-146.7, 401.6, 92.5) cluster = 26 track trajectory broken
around the track trajectory ??? 3. When the track is isochrnous (ISO), there is a large ambiguities, and I try to
not sample the points for every wire crossing. The motivation is to save memory. For PDHD and PDVD, the wire
pitch is large, so it is possible this strategy of sampling is not the best, because this kind of sampling lead
to a grid sturcture for the track trajecotory seed, and then inherit to the fitted track trajectory. Can you help
to investigate these? I suggest you to make some plots first so that you can look at them and then start the
investgation and design the improvement. Please create a new md file, commit and push."*

**Owner decisions (2026-09-15).**
* **Scope:** diagnose, then test levers. The sampler goes in as a config arm, plus at most one default-OFF fit
  knob, and only if one lever is clearly identified. No production flip without the owner's go.
* **Sampler:** plain `stepped` in the PR retile is a **port bug**. Measure `charge_stepped` and prepare a flip.
* **Verdict:** simulation truth decides by a pre-registered rule. The fixed-denominator STM/Michel grades are
  reported only.

The Bee set is doc 101's f841d51c: PDHD 028084_21, event 0 = fit knobs off (`d101smoff`), event 1 = on
(`d101smkf`). **Baseline everywhere in this doc = fit knobs ON** (`fit_weight_pow` 1.5 + `assoc_cont_center` 1).
As in doc 101, the knob pair rides on two fit JSONs:
* simulation arms use `figs/101_tf_sim_{det}_kf.json`, whose diffusion and transverse widths match the simulation;
* data arms use `figs/101_tf_prod_{det}_kf.json`, the detector's production fit JSON.

In every comparison the arm and its baseline share one JSON.

---

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs; W=$HOME/tmp/d102
PIN=$W/libpin_d102   # private copy of local/lib + bin at toolkit 419f70f3; libWireCellClus md5 091e142b9481 (== doc 101 libpin_k)
CS="-S retile_sampler_strategy='charge_stepped'"
# sec 1-2 -- figures on the doc 101 arms (PNGs are gitignored; each run also writes the committed <out>_spots.tsv)
cd $D
python3 $S/d102_spot_figs.py --det pdhd --event 028084_21 --arms d101smoff,d101smkf --uv-rank-verified --out figs/102_pdhd
python3 $S/d102_spot_figs.py --det pdvd --event 039349_64 --arms d101vnew,d101vkf --spot pdvd_iso_on:61:309.6,-219.6,171.8 --out figs/102_pdvd_on
python3 $S/d102_spot_figs.py --det pdvd --event 039253_8 --arms d101vnew,d101vkf --spot pdvd_iso_off:57:92.8,-63.5,242.2 --out figs/102_pdvd_off
# sec 3 -- counterfactual on the same three events: knob-on fit (d102dbg / d102vdbg, == d101smkf / d101vkf) vs + charge_stepped
bash $S/d102_counterfactual.sh
python3 $S/d102_spot_figs.py --det pdhd --event 028084_21 --arms d102dbg,d102cs --uv-rank-verified --out figs/102_cf_pdhd
python3 $S/d102_spot_figs.py --det pdvd --event 039349_64 --arms d102vdbg,d102vcs --spot pdvd_iso_on:61:309.6,-219.6,171.8 --out figs/102_cf_pdvd_on
python3 $S/d102_spot_figs.py --det pdvd --event 039253_8 --arms d102vdbg,d102vcs --spot pdvd_iso_off:57:92.8,-63.5,242.2 --out figs/102_cf_pdvd_off
# sec 6 -- compiled-config proofs (before the jsonnet edit: pre; after: post = default, on1 = charge_stepped) and the doctest
bash $S/d102_compile_pr.sh pdhd post; bash $S/d102_compile_pr.sh pdhd on1 $CS                 # same for pdvd; cmp pre_* post_*
cd $IMG/../toolkit && ./build/clus/wcdoctest-clus -tc="pdvd doc102*"
# sec 7 -- simulation: truth, chain, frozen rule, arms, grading
python3 $S/d101_make_muons.py --det pdvd --cfg $HOME/tmp/d101/sim/cfg_probe/pdvd_a1.json --anode 1 --face 0 --run 900103 --tag d102 \
    --theta 65,75,85,89 --phi 0,30,60,90 --repeat "16:0,17:10" --out $W/sim/truth_pdvd.json
python3 $S/d101_make_muons.py --det pdhd --cfg $HOME/tmp/d101/sim/cfg_probe/pdhd_a1.json --anode 1 --face 1 --run 900104 --tag d102 \
    --theta 65,75,85,89 --phi 0,30,60,90 --repeat "16:0,17:10" --out $W/sim/truth_pdhd.json
JOBS=4 bash $S/d102_sim_batch.sh                  # d102_sim_chain.sh, k 0-17, both detectors
cat $W/pred.sha256                                # 795adc97... frozen 2026-09-15T05:59:07, first arm 05:59:08
for d in pdvd pdhd; do DET=$d PIN=$PIN JOBS=2 bash $S/d102_knob_arms.sh
  for a in d102b d102brep d102cs1 d102cs2 d102cs3; do python3 $S/d102_sim_fit_eval.py --det $d --arm $a --out $W/eval/eval_${d}_$a.tsv; done
  for a in d101kf d102lcs1 d102lcs2 d102lcs3; do python3 $S/d101_sim_fit_eval.py --det $d --arm $a --out $W/eval/eval_${d}_$a.tsv; done; done
python3 $S/d102_repeat_check.py                   # d102brep == d102b, every tree and branch, 18/18 per detector
python3 $S/d102_knob_report.py --dir $W/eval --out $W/knob_report.tsv
python3 $S/d102_c4_decompose.py
# sec 5, 8 -- data arms (61 PDHD / 120 PDVD events, pctrees of the doc 101 arms), resources, census, chords, grades
ARM=d102hcs    DET=pdhd SRC=d101hkf JOBS=3 PIN=$PIN PR_TLA="-A trackfitting_config=$F/101_tf_prod_pdhd_kf.json $CS" bash $S/d102_run_arms.sh
ARM=d102vcsall DET=pdvd SRC=d101vkf JOBS=3 PIN=$PIN PR_TLA="-A trackfitting_config=$F/101_tf_prod_pdvd_kf.json $CS" bash $S/d102_run_arms.sh
python3 $S/d102_resource_report.py pdhd:d101hkf:d102hcs pdvd:d101vkf:d102vcsall
python3 $S/d102_census.py --out $W/census_hcs pdhd:d101hkf pdhd:d102hcs
python3 $S/d102_census.py --out $W/census_vcs_common --exclude 039252_5,039349_14,039349_78 pdvd:d101vkf pdvd:d102vcsall
python3 $S/d102_chord_attrib.py pdhd:d101hkf pdhd:d102hcs pdvd:d101vkf pdvd:d102vcsall
python3 $S/d101_stm_grade_pdhd.py d101hnew d101hkf d102hcs
M=$IMG/pdhd/stm_michel_scan; export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
python3 $M/prep_stm_michel_scan.py --det pdvd --arm d102vcsall --pin-tranche $HOME/tmp/d101/sheet_d101vkf/pdvd_stm_michel_scan_sheet.tsv \
    --outdir $W/prep_d102vcsall2 --sheetdir $W/sheet_d102vcsall2
python3 $S/d101_stm_grade_pdvd.py d101vnew=$HOME/tmp/d101/prep_d101vnew d101vkf=$HOME/tmp/d101/prep_d101vkf d102vcsall=$W/prep_d102vcsall2
```

`--pin-tranche` is needed because the prep refuses to re-draw a scan tranche once labels exist. The tranche
chooses only the scan sheet; payloads are written for every candidate (581 for both d101vkf and d102vcsall), so
the fixed-denominator grade is unaffected.

---

## 1. The four spots

All four clicks resolve to image points of the named cluster; Bee coordinates are the zip JSON frame. All four
are near-isochronous. The STM fit of each cluster is a single pass (pass 0), so there is no multi-pass overlay.

| spot | click | cluster | angle to drift | what the knob-on fit does there |
|---|---|---|---|---|
| 1 | (326.1, 457.4, 314.4) | 128 | 71.5° | STM fit leaves the V charge band: `reg_flag_v` on 30 % of rows, V on-charge 0.79, largest distance from ridge 2.06 cm, 8 rows at Bee q ≤ 0 |
| 2 | (−67.1, 569.9, 398.9) | 55 | 64.7° | same, V on-charge 0.82, largest distance 2.51 cm; PR cuts the stretch into 5 segments with 14 zero-display rows |
| 3 | (−78.0, 571.8, 418.7) | 55 | 70.0° | knob-off: the fit ran into a crossing branch, dQ/dx median 12.7 ke/cm over ~20 cm, 23 rows at q ≤ 0. Knob-on: the fit stays on the main track 6.3 cm from the click, on charge in all three planes |
| 4 | (−146.7, 401.6, 92.5) | 26 | 85.7° | see sec 2 |

Figures (regenerate with sec 0; knob-off top row, knob-on bottom row):
* `figs/102_pdhd_spot{1..4}_3d.png`: local 3-D in the track frame, with the image, the retiled Steiner cloud and
  terminals, the STM fit, PR segments with vertices, and the click;
* `figs/102_pdhd_spot{1..4}_2d.png`: wire × slice per plane, with measured charge, dead channels and the fit
  projections;
* `figs/102_pdhd_spot{1..4}_profile.png`: along-track dQ/dx, ridge distance, `reg_flag` and lattice lock;
* `figs/102_pdhd_grid_spots.png`: in-slice spacing of image vs Steiner cloud.

Numbers: `figs/102_pdhd_spots.tsv`.

**Reading the plots.**
* **"On charge" (`stm_onq_p*`)** means the fitted point's own cell (round wire, round slice) holds measured charge
  of this cluster in `T_proj_data`. That test is 98.5 % on regularisation-clean W rows.
* **The 2-D projection** of Steiner points uses a per-point local affine map (kNN = 12) fitted on the cluster's
  own fit rows. A global map does not close on wrapped U/V planes.
* **Bee q ≤ 0 is a display value.** Bee `stm_fit` / `track_fit` q equals `T_rec_charge` q = dQ·0.1 − 1000, so
  q ≤ 0 means dQ < 10 ke. `track_fit` clamps it to 0 (MultiAlgBlobClustering.cxx:1180-1181); `stm_fit` does not
  (:3130).

## 2. What "broken" is at spot 4 (cl26)

* **Geometry.** The track is 85.7° from drift. pv and pt are constant over 5 cm, so V constrains nothing.
* **Wrap edge.** U runs over wire index 9–22 at the wrapped-plane edge, and the U projection of the fit crosses
  the 0 ↔ 800 wrap.
* **The click is on another branch.** It sits on a branch displaced about 3 cm in drift. The STM fit stays on
  charge but runs 3.5 cm from the clicked image.
* **The PR fit is fragmented.** 8 segments meet in the ±12 cm window (9 knob-off), and 70 of its rows there
  display at zero (dQ < 10 ke).

The "broken" appearance is that fragmentation plus the zero-display rows. It is not missing rows: every PR row in
the window is present in `T_rec_charge`.

Under `charge_stepped` the cluster **loses its PR fit and drops out of the STM/Michel candidates altogether**. So
does cl128: 8 candidates → 8, but a different set, with `michel_found` 1 → 2. That is candidate churn, not a
repair (sec 8.4).

## 3. Counterfactual on the same event: `charge_stepped`

Arms `d102dbg` (knob-on fit with log-only debug envs) and `d102cs` (the same plus
`retile_sampler_strategy='charge_stepped'`) share the pctree and the pin.

The debug envs are inert: `d102dbg` equals `d101smkf` in every ROOT tree, the calib json and the zip member hash.
The same holds for `d102vdbg` vs `d101vkf`.

Figures: `figs/102_cf_pdhd_*`. Numbers: `figs/102_cf_pdhd_spots.tsv`.

| spot | largest distance from ridge (cm) | `reg_flag_v` share | STM on-charge U / V / W | Bee q ≤ 0 rows | STM dQ/dx median (ke/cm) | PR segments / zero rows in window |
|---|---|---|---|---|---|---|
| 1 cl128 | 2.06 → **0.87** | 0.30 → **0** | 0.97/0.79/0.97 → 0.97/**1.00**/1.00 | 8 → **0** | 52.9 → 54.7 | no PR fit in either |
| 2 cl55 | 2.51 → **0.77** | 0.30 → **0** | 1.00/0.82/0.86 → 1.00/**1.00**/**1.00** | 3 → **0** | 46.9 → **56.4** | 5 / 14 → **1 / 0** |
| 3 cl55 | 1.50 → 1.26 | 0 → 0 | 1.00 → 1.00 | 0 → 0 | 45.6 → 52.0 | 5 / 8 → 4 / 28 |
| 4 cl26 | 0.98 → 1.09 | 0 → 0 | 1.00/1.00/0.92 → 0.97/1.00/0.95 | 3 → 0 | 29.9 → 41.0 | 8 / 70 → **no PR fit** |

At spot 2 (`figs/102_cf_pdhd_spot2_3d.png`):
* **Knob-on:** the STM fit bulges ~2 cm toward the click and PR splits the stretch at three vertices.
* **`charge_stepped`:** the fit runs straight along the track and PR fits one segment through it.

The in-slice spacing of the Steiner cloud goes from 1.48 cm (3 W pitches) to a 0.49 cm peak (one pitch); the image
stays at 0.99 / 1.48 cm (`figs/102_cf_pdhd_grid_spots.png`).

**PDVD examples (not improved).** Examples chosen from the census as the highest `reg_flag` stretches ≥ 80°.
* **039349_64 cl61 (84°, on the image):** V on-charge 0.95 → 0.87, largest distance 2.97 → 2.65 cm.
* **039253_8 cl57 (87–89°):**
  * Both STM passes run a **60 cm straight chord up to 33 cm off the image**, with V/W on-charge 0.09 / 0.00.
  * `charge_stepped` changes the displayed charge (q ≤ 0 rows 37 → 14) but not the chord (V/W on-charge
    0.09 / 0.00).
  * The STM graph there is connected (ncomp = 1, 9844 Steiner points), and no Steiner point lies within 2 cm of
    the chord's interior: it is one long graph edge (sec 5.3).

## 4. Mechanisms

| defect | layer | mechanism | evidence | what acts on it |
|---|---|---|---|---|
| fit leaves V charge, dQ/dx collapses (spots 1, 2) | retile cloud → seed → fit | 3-wire stepped cloud; the seed sits off V, the fit is regularised there (`reg_flag_v`, TrackFitting.cxx:7842-7930) | sec 3: `charge_stepped` removes it | retile sampler |
| off-image chords (PDVD cl57; census) | STM rough path | Dijkstra on `steiner_graph` (TaggerCheckSTM.cxx:1162-1269) crosses an uncapped MST bridge (connect_graph.cxx, connect_graph_ctpc.cxx); `organize_orig_path` fills it with a straight line | sec 5.3: 93–95 % of chords have no retiled point under them; sampler leaves them | graph bridging (not built, sec 9) |
| route into a crossing branch (spot 3, knob-off) | trajectory fit | q² weighting pulled the fit into the branch | knob-on fit stays on the main track | doc 101 fit knobs |
| "broken" (spot 4) | PR multi-track at ISO + wrap edge | fragmentation into 8 short segments, zero-display rows | sec 2 | none tested fixes it (`charge_stepped` removes the cluster's PR fit) |
| ISO grid (issue 3) | retile sampler | `stepped` = max(3, N/12) wires + last (BlobSampler.cxx:868-921) | census: 3-wire share 0.594 (PDHD) | retile sampler |

The sampler also drives in-blob graph edges through its `max/min_wire_interval` aux
(connect_graph_closely.cxx:80-134, :255-359). A sampler change therefore moves graph connectivity as well as
density, and this doc does not separate the two.

## 5. Census (whole data arms; knob-on baseline vs `charge_stepped`)

`d102_census.py` over STM-fit rows (runs ≥ 20 rows, local direction from rows i ± 3), by angle to drift. Columns:
* `reg`: any `reg_flag_{u,v,w}`;
* `lock`: ≥ 2 of pu/pv/pw on an integer or half-integer;
* `onq`: exact-cell on-charge per plane;
* `offimg`: > 3 cm from every image point of the own cluster.

### 5.1 PDHD (61 events, no event skipped)

| angle | reg base → cs | lock | onq U | onq V | onq W | offimg |
|---|---|---|---|---|---|---|
| 0–30 | 0.327 → 0.291 | 0.051 → 0.073 | 0.815 → 0.838 | 0.799 → 0.811 | 0.856 → 0.857 | 0.115 → 0.097 |
| 30–50 | 0.166 → 0.153 | 0.029 → 0.048 | 0.917 → 0.915 | 0.916 → 0.915 | 0.923 → 0.925 | 0.034 → 0.037 |
| 50–65 | 0.142 → 0.119 | 0.025 → 0.039 | 0.948 → 0.951 | 0.932 → 0.933 | 0.938 → 0.946 | 0.018 → 0.020 |
| 65–75 | 0.132 → **0.110** | 0.021 → 0.029 | 0.942 → 0.947 | 0.930 → 0.934 | 0.940 → 0.946 | 0.027 → 0.024 |
| 75–85 | 0.161 → **0.143** | 0.023 → 0.025 | 0.924 → 0.931 | 0.912 → 0.922 | 0.911 → 0.921 | 0.048 → 0.043 |
| 85–90 | 0.263 → 0.259 | 0.031 → 0.030 | 0.862 → 0.851 | 0.854 → 0.856 | 0.829 → 0.826 | 0.092 → 0.099 |

### 5.2 PDVD (117 events common to both arms)

039252_5 has no `steiner_graph` layer in either arm; 039349_14 and 039349_78 lack it in the baseline only. All
three are excluded from both arms.

| angle | reg base → cs | lock | onq U | onq V | onq W | offimg |
|---|---|---|---|---|---|---|
| 0–30 | 0.186 → 0.169 | 0.014 → 0.017 | 0.765 → 0.769 | 0.698 → 0.699 | 0.964 → 0.967 | 0.020 → 0.020 |
| 30–50 | 0.170 → 0.150 | 0.009 → 0.009 | 0.792 → 0.797 | 0.754 → 0.760 | 0.972 → 0.978 | 0.009 → 0.008 |
| 50–65 | 0.192 → 0.159 | 0.010 → 0.008 | 0.821 → 0.829 | 0.800 → 0.805 | 0.961 → 0.970 | 0.012 → 0.011 |
| 65–75 | 0.253 → **0.210** | 0.013 → 0.011 | 0.865 → 0.882 | 0.828 → 0.848 | 0.932 → 0.947 | 0.031 → 0.029 |
| 75–85 | 0.337 → **0.289** | 0.021 → 0.016 | 0.844 → 0.859 | 0.819 → 0.836 | 0.867 → 0.870 | 0.083 → 0.087 |
| 85–90 | 0.389 → 0.393 | 0.024 → 0.016 | 0.820 → 0.811 | 0.788 → 0.790 | 0.808 → 0.795 | 0.136 → 0.150 |

**Reading.**
* `charge_stepped` lowers the `reg_flag` share by 10–20 % at every angle up to 85°.
* It does **not** help the most isochronous bin (85–90°), where `offimg` rises slightly on both detectors.
* The doc 101 fit knobs barely move any of these columns: across knob-off and knob-on the chord counts are
  287–289 (PDHD) and 307–308 (PDVD).

### 5.3 Chords and the Steiner graph

A chord is ≥ 10 consecutive off-image rows.

| arm | chords | total length | > 20 cm | longest |
|---|---|---|---|---|
| PDHD base / cs (61 events) | 287 / 278 | 9022 / 8632 cm | 130 / 124 | 223 / 225 cm |
| PDVD base / cs (117 events) | 308 / 297 | 6830 / 6592 cm | 95 / 94 | 187 / 185 cm |

`d102_chord_attrib.py` looks at the first 30 events per arm. A chord counts as having "no Steiner point under it"
when the median distance from its rows to the cluster's retiled cloud (calib `steiner`) is > 3 cm.

| arm | no Steiner point under the chord |
|---|---|
| PDHD base / cs | 117/123 = 0.951 / 119/125 = 0.952 |
| PDVD base / cs | 114/123 = 0.927 / 106/114 = 0.930 (039252_5 skipped in both) |

The retiled cloud does not extend under nearly all chords. The rough path jumps a long graph edge and the fit draws
a straight line across it. A denser sampler cannot reach that, and the census shows it does not.

### 5.4 The grid

In-slice nearest-neighbour spacing of the Bee `steiner_graph` cloud:

| arm | points | median | at 3 W pitches (± 0.15 cm) | below 1.2 W pitch |
|---|---|---|---|---|
| PDHD base → cs | 1 968 606 → 2 860 377 | 1.478 → **0.800** cm | **0.594 → 0.102** | 0.064 → 0.459 |
| PDVD base → cs | 524 030 → 795 381 | 1.767 → **0.883** cm | 0.063 → 0.103 | 0.045 → 0.177 |

PDVD's grid is not at 3 W pitches (its step lands elsewhere), but its spacing halves in the same way.

## 6. The sampler port divergence

**Both readings (M15).**
* **Toolkit.** PDHD/PDVD/SBND imaging and the PR retile all use `stepped`
  (`pdhd/pr.jsonnet` / `protodunevd/pr.jsonnet` via `bs_live_face`). The ISO sparsity is by design, to save memory
  (the owner's item 3).
* **Prototype.** The retile samples with `calc_sampling_points(..., disable_mix_dead_cell=false)`
  (prototype `ImprovePR3DCluster.cxx:59`, `PR3DCluster_steiner.h:16`), which takes every wire of small blobs. The
  toolkit port of that rule, `ChargeStepped`, is used only by uBooNE (`qlport/uboone-mabc.jsonnet`).
* **Not recorded anywhere.** Neither `improve_pr3dcluster_review.md` §2.4 nor `porting_dictionary.md` records the
  divergence.

**Owner ruling (2026-09-15): port bug; measure the prototype rule and prepare a flip.**

Doc 101's `ks` arm (`half_pitch` + `min_step` 1) was **not** that rule: with min_step 1 the step is still N/12 for
N ≥ 24. Its PDHD θ = 80° failure is evidence against that knob pair only.

### 6.1 Fidelity: `ChargeStepped` (BlobSampler.cxx) vs prototype `CalcPoints.cxx`

| rule | prototype | toolkit | same? |
|---|---|---|---|
| mandatory ("stepped") wires | step max(3, N/12), front, every step, back (:75-92) | max(`min_step_size` 3, N/12) from the first, plus the last (:1155-1197) | yes |
| all wires when N_max·N_min ≤ 2500 | :129-132 | `max_wire_product_threshold` 2500 (:1200-1217) | yes |
| charge thresholds | 4000 on max / min / other (:94-96) | 4000 ×3 (:1062-1064) | yes |
| a zero-charge wire counts as dead and is kept unless `disable_mix_dead_cell` | :142, :154, :193-199 | :1291, :1313, :1346-1357 | yes |
| bad plane → its threshold 0 | `mcell->get_bad_planes()`, always (:97-106) | uncertainty > `dead_threshold` (1e10), **only when `disable_mix_dead_cell` is true** (:1163-1176); the retile runs false, so never | **differs** |
| mid-view window | wire distance in [front − ½ pitch − 0.1 mm, back + ½ pitch + 0.1 mm] (:67-73, :161) | `pitch_relative` in (first − 0.03, second + 0.03), pitch units (:1331-1335) | same edges if the ray-grid index is the wire edge (not verified here); tolerance 0.1 mm vs 0.03 pitch |
| third-plane wire whose charge is tested | always `bounds_wires.second`; the "pick the closest" distances are computed and unused (:174-183) | `round(pitch_relative)` (:1338) | **differs** (the prototype's own comment says closest) |
| x of the point | slice start (:162) | time-binning edge; one bin at slice start by default | yes |
| charge lookup | `mcell->Get_Wire_Charge` | slice activity; a wrapped continuation resolved by channel ident under `wrapped_channel_charge` (:1488-1571) | toolkit fix, pinned below |

The two **differs** rows are surfaced, not changed. Neither is exercised by the arms of sec 7–8 in a way this doc
can isolate.

### 6.2 The knob

* **toolkit `cfg/pgrapher/experiment/{pdhd,protodunevd}/clus.jsonnet`:** `bs_live_face` / `live_sampler` gain
  `strategy_name` (default `'stepped'`), `wire_product` and `charge_threshold`.
  * With `'charge_stepped'` they emit `{name: "charge_stepped", disable_mix_dead_cell: false}`, as the prototype
    retile does.
  * Keys are suppressed when null.
  * An invalid name aborts at compile.
* **toolkit `{pdhd,protodunevd}/pr.jsonnet`:** new args `retile_sampler_strategy` / `_wire_product` /
  `_charge_threshold` (all null). They reach **only the retile samplers** (`improve_cluster_2`); imaging is
  untouched.
* **wcp `pdhd/`, `pdvd/wct-pr-perevt.jsonnet`:** the same three TLAs, passed through.

**Compiled-config proofs** (`$W/cfg/`, md5 prefix):
* PR job before vs after the edit: byte-identical. PDHD 434208b1d840 (8 BlobSampler `["stepped"]`), PDVD
  920ecb4312ab (16 samplers).
* With `charge_stepped` (`on1`: PDHD 87a86589c767, PDVD 211a49a48229) only the retile BlobSampler strategy blocks
  differ.
* Clustering job, HEAD cfg vs edited cfg: byte-identical (PDHD 2b97cd1a2254, PDVD 912d9e232045).
* SBND and uBooNE configs only mention these files in comments; they import neither.

No C++ library changed, so there is no runtime byte-identity gate to run. Every arm ran on `libpin_d102` (Clus
md5 091e142b9481).

### 6.3 Wrapped-plane doctest

`clus/test/doctest_blob_sampler_wrapped_channel.cxx`, TEST_CASE *"pdvd doc102: charge_stepped resolves wrapped
continuations' charge exactly as stepped"*.

**Setup.**
* Builds PDHD apa1 through the factory, exactly as the compiled clustering config does.
* Tiles a 4 × 4 cm blob at (y, z) = (286, 115) cm. Its U and V strips are wrapped continuations (13 / 12 orphan
  wires) and W is not.
* Puts 2000 on U/V (live, below the 4000 threshold) and 5000 on W.

**Results.**
* With the fix, both strategies read U/V = 2000 and W = 5000 on every point.
* Negative control, `wrapped_channel_charge=false`: both strategies read 0 on a wrapped plane.
* `charge_stepped` samples **84 points instead of 12** under the negative control: the 0 counts as "dead", and
  `disable_mix_dead_cell=false` keeps it.
* `stepped` samples 12 either way.

`wcdoctest-clus` 413/413 test cases.

## 7. Simulation arms and the pre-registered verdict

**Sample.**
* **ISO stratum:** 16 single muons per detector (100 cm, 5000 e/mm), θ ∈ {65, 75, 85, 89}° × φ ∈ {0, 30, 60, 90}°,
  runs 900103 (PDVD) / 900104 (PDHD), plus two re-simulated repeats (k16 = k0, k17 = k10).
* **Low stratum:** the 12 doc-101 tracks with θ ≤ 60° (runs 900101/900102) against doc 101's `d101kf`.
* **Chain:** full sim → NF/SP → imaging → clustering → PR (`d102_sim_chain.sh`), fit knobs on
  (`figs/101_tf_sim_{det}_kf.json`).

**Noise floor.** `d102brep` reproduces `d102b` on all 18 events per detector: `tracking-pr.root` and
`tracking-stm.root` are identical in every tree and branch (`d102_repeat_check.py`).

**Rule** (`$W/pred.txt`, sha256 795adc97…, frozen 2026-09-15T05:59:07, first arm 05:59:08). A setting passes on a
detector when:
* **ISO:** median Δ `res_t_p90` < 0 **and** median Δ `dev_p90` < 0 (c1); ≥ 10/16 tracks improve (c2);
  median Δ `dip_truth` ≤ +0.01 (c3); no track gains a PR segment (c4).
* **Low stratum:** median Δ `res_t_p90` ≤ +0.05 mm and Δ dip ≤ +0.01.
* **Data resources:** median RSS ≤ +25 %, wall ≤ +50 %.

A setting passing both detectors is proposed for the flip.

**Settings.**
* `cs1`: prototype defaults (2500 / 4000);
* `cs2`: thresholds 2000;
* `cs3`: wire product 10000.

### 7.1 Verdict

| setting | det | Δ res_t p90 med (base) | improve | Δ dev p90 med | Δ dip | c1 c2 c3 | c4: segments gained | low stratum Δ res_t p90 | pass |
|---|---|---|---|---|---|---|---|---|---|
| cs1 | PDVD | −0.565 mm (3.72) | 12/16 | −0.414 | −0.006 | ✓ ✓ ✓ | k5, k10, k11 | −0.019 ✓ | no |
| cs1 | PDHD | −1.139 mm (4.00) | 13/16 | −0.496 | −0.015 | ✓ ✓ ✓ | k6, k13 | −0.181 ✓ | no |
| cs2 | PDVD | −0.492 | 11/16 | −0.472 | −0.003 | ✓ ✓ ✓ | k5, k11 | −0.012 ✓ | no |
| cs2 | PDHD | −0.570 | 11/16 | −0.873 | −0.023 | ✓ ✓ ✓ | k6 | −0.031 ✓ | no |
| cs3 | PDVD | −0.979 | 13/16 | −0.511 | −0.006 | ✓ ✓ ✓ | k5, k10, k11 | −0.019 ✓ | no |
| cs3 | PDHD | **−1.363** | **14/16** | −0.764 | −0.015 | ✓ ✓ ✓ | k6, k13 | −0.181 ✓ | no |

**By the frozen rule no setting qualifies, and no flip is proposed.**

Ranking on the trajectory: cs3 > cs1 > cs2 on both detectors. PDHD's θ = 89° tracks k14/k15 are gross failures in
the baseline (res_t median 160 / 201 mm, doubled points); cs3 improves their p90 by 19 / 17 mm.

Repeats k16/k17: Δ p90 +0.02 / −0.41 (PDVD) and +0.02 / −1.11 (PDHD). These equal their originals k0/k10, so the
seed noise is small against the effects.

### 7.2 What c4 caught (`d102_c4_decompose.py`; the verdict above stands)

c4 was written for doc 101's PDHD k3 mode: a spurious segment **and** the main pulled off the track.

| track | settings | base → arm segments (points, pid) | new segment: along truth / to image / from muon line | Δ res_t p90 | class |
|---|---|---|---|---|---|
| PDVD k5 (75°) | cs1, cs3 | (182, 13) → (164, 13) + (9, 11) | 2.5–5.1 cm / 0.54 / 0.75 cm | −2.64 mm | stub at the track start, main improved |
| PDVD k5 | cs2 | (182, 13) → (174, 13) + (10, 11) | 0.8–4.6 cm / 0.52 / 0.75 cm | +0.28 mm | stub at the start, main slightly worse |
| PDVD k10 (85°) | cs1, cs3 | (178, 13) → (169, 13) + (6, 11) | 2.0–4.1 cm / 0.56 / 0.42 cm | −1.32 mm | stub at the start, main improved |
| PDHD k6 (75°) | cs1, cs2, cs3 | (174, 13) → (175, 13) + (6, 11) | −0.2–1.5 cm / 0.59 / 1.14 cm | −1.55 / −0.54 mm | stub at the start, main improved |
| PDHD k13 (89°) | cs1, cs3 | (181, 13) → (178, 13) + (4, 211) | 0.5–2.0 cm / 0.66 / 0.32 cm | −4.85 mm | stub at the start, main improved |
| **PDVD k11 (85°)** | cs1, cs3 (cs2 alike) | (177, 13) → (128, 13) + (50, 13) + (17, 13) | 68.2–72.7 cm / **0.61** / **6.5 (max 9.1) cm**, Bee q median −773 | **+27.33 mm** (cs2 +12.31) | main split at a side branch |

**Reading.**
* **The stubs.** In 7 of the 8 stub cases the new segment is a 4–10-point stub within 5 cm of the track start,
  beside a main that got better. That is not doc 101's failure mode.
* **PDVD k11.** Its two muon halves stay on the truth line (median 0.25 / 0.30 cm). The third segment lies on the
  image, 6.5 cm from the muon line, at ~8 % of the main's dQ.
  * The baseline image holds 206 points 3–9 cm off the truth line in that same stretch, 46 % of the charge there.
  * So it is real deposited charge off the muon, most likely a delta ray: the truth json lists only the muon.
  * The baseline fit ignored it; `charge_stepped` resolves it as a branch.
  * The +27 mm is the eval scoring that branch against the muon line.

That reading is inferred from the image, not from truth particles. Whether c4 should count these cases is the
owner's call (sec 10). It is not re-graded here.

## 8. Data arms (reported, not gating)

`d102hcs` (PDHD 61) and `d102vcsall` (PDVD 120) are the production `-nu -stm-fit` chain with the knob-on fit and
`charge_stepped` (cs1). They read the pctrees of `d101hkf` / `d101vkf`, on the same pin. There are no "Failed to
set parameter" lines.

PDVD 039252_5 lacks the CheckSTM_Michel line in both this arm and the baseline, so that is pre-existing.

### 8.1 Resources (the pre-registered clause)

| det | events | wall ratio median (p90) | peak RSS ratio median (max) | Steiner stage ratio | load control ("loaded live") | Steiner / control |
|---|---|---|---|---|---|---|
| PDHD | 61 | **1.38** (3.88) ✓ | **1.018** (1.495) ✓ | 1.87 | 1.01 | 1.84 |
| PDVD | 120 | **0.98** (2.76) ✓ | **1.022** (1.723) ✓ | 1.24 | 0.98 | 1.28 |

**The wall ratio is load-contaminated.** The baselines ran on another night. During its first 30 minutes this arm
ran beside the simulation campaign, and a 29-event early read gave ×3.0 with "loaded live" at ×1.7.

**The largest RSS ratios are small events.** PDHD 1.05 → 1.57 GB; PDVD 0.47 → 0.81 GB. The largest absolute peak
is 3.29 GB, against 2.92 GB in the baseline.

**The memory the owner's item 3 wanted to save.** The Bee Steiner cloud grows by +45 % (PDHD) and +52 % (PDVD)
points. The Steiner stage takes ×1.8 / ×1.3 its time, and TaggerCheckSTM and CheckSTM_Michel absorb the rest.

### 8.2 Owner's spots

See sec 3: spots 1–2 fixed, spot 3 unchanged in kind, spot 4's cluster loses its PR fit.

### 8.3 Census

See sec 5.

### 8.4 STM / Michel tags on the hand-scan records (fixed denominator)

| det | arm | STM purity / efficiency | Michel purity / efficiency | absent hand stoppers / others |
|---|---|---|---|---|
| PDHD (257 scored) | knobs off `d101hnew` | 0.974 / 0.776 | 0.971 / 0.791 | 0 / 0 |
| PDHD | knobs on `d101hkf` (baseline) | 0.947 / 0.605 | 0.919 / 0.663 | 22 / 26 |
| PDHD | + `charge_stepped` `d102hcs` | 0.937 / 0.605 | **0.839 / 0.605** | 21 / 37 |
| PDVD (546 judged) | knobs off `d101vnew` | 0.968 / 0.877 | 0.923 / 0.872 | 0 / 0 |
| PDVD | knobs on `d101vkf` (baseline) | 0.927 / 0.739 | 0.830 / 0.713 | 35 / 62 |
| PDVD | + `charge_stepped` `d102vcsall` | 0.922 / 0.732 | 0.837 / 0.689 | 35 / 86 |

**PDHD Michel is the regression:** FP 5 → 10, TP 57 → 52. The candidate set churns on both detectors (absent
others +11 PDHD, +24 PDVD). These 10 FPs are not adjudicated here.

## 9. Fit lever: none built

The plan allowed at most one default-OFF fit knob, and only if one mechanism dominates the off-image points. One
does: the long Steiner-graph edge under the STM rough path (sec 5.3). But it is not a fit mechanism.

**Candidates, ranked by what the census says they could reach:**
1. **Price or cap the bridges the STM rough path may cross.** Options: a gap-priced graph (the
   `steiner_graph_gap` flavour is reachable only from NeutrinoPatternBase.cxx:122-127, not from TaggerCheckSTM),
   or a bridge-length cap in connect_graph / connect_graph_ctpc.
   * **Reach:** 93–95 % of chords.
   * **Blocker:** the doc-102 simulation has no gaps, so this rule cannot grade it. It needs a new
     pre-registered rule (data chord reduction + simulation no-regression).
2. **`charge_stepped` retile (sec 7–8).**
   * **Reach:** spots 1–2, the grid, `reg_flag` at 65–85°.
   * **Cost:** the c4 stubs, the PDHD Michel purity drop, candidate churn, and the Steiner stage ×1.3–1.8.
3. **Revert-to-seed suppression (TrackFitting.cxx:6497-6531, :5995-6050) or a charge re-fit instead of a deletion
   in `traj_final_fill_charge_test`.**
   * **Reach:** points already on a bad seed; no census evidence that these dominate.

## 10. Flip proposal and what the owner needs to decide

**No flip is proposed:** no `charge_stepped` setting passes the frozen rule. If the owner wants to pursue it anyway,
the open questions are:
1. **c4.** Do ≤ 10-point stubs at the track start beside an improved main count as a failure? Does a branch
   resolved at real off-line charge (PDVD k11)? If not, a revised rule would need a fresh sample; this one has
   been looked at.
2. **The PDHD Michel purity drop** 0.919 → 0.839 (10 FPs) needs adjudication before any flip. Doc 101's lesson
   (one mislabel faked a cost) applies both ways.
3. **Spot 4 / cl26.** Losing the PR fit is not a repair of the "broken" trajectory.

If a flip is ever made, the prepared change is one jsonnet default:
* `retile_sampler_strategy` null → `'charge_stepped'` in `pdhd/pr.jsonnet` and `protodunevd/pr.jsonnet`;
* compiled diff = the retile BlobSampler strategy blocks only (`on1_*` in `$W/cfg/`);
* SBND shares the divergence and is untouched.

## 11. Not concluded

* **The PDVD ISO on-image example 039349_64 cl61 is not improved**, and the 85–90° bin is not improved on either
  detector.
* **Separation not done.** Sampling density and the sampler-driven graph connectivity (sec 4) were not separated.
* **Mid-view window.** The table in sec 6.1 compares the two windows by reading only; no measurement.
* **The k11 side charge** is attributed from the image. The truth json lists only the muon, so delta-ray content was
  not checked against Geant4 truth.
* **The wall p90** (×2.8–3.9) was not profiled, and part of it is load.
* **Census coverage.** The first census over the doc 101 arms (knob-off and knob-on) skipped 4 PDVD events for a
  missing `steiner_graph` layer (039252_11, 039252_5, 039349_14, 039349_78). Sec 5.2 uses a common 117-event set
  instead.
* **Not produced.** No Bee upload.

## 12. Files

* **Doc:** `pdvd/docs/nf_sp_img_clus/102_iso-sampling-off-image-fit.md`
* **Scripts (`scripts/`):**
  * `d102_spot_figs.py`, `d102_census.py`, `d102_chord_attrib.py`, `d102_c4_decompose.py`,
    `d102_resource_report.py`, `d102_knob_report.py`, `d102_repeat_check.py`;
  * `d102_sim_fit_eval.py`, `d102_sim_chain.sh`, `d102_sim_pr_arm.sh` and `d102_run_arms.sh` (forks of their
    d101 / d16 counterparts);
  * `d102_knob_arms.sh`, `d102_sim_batch.sh`, `d102_counterfactual.sh`, `d102_compile_pr.sh`.
* **Figures:** `figs/102_{pdhd,pdvd_on,pdvd_off,cf_pdhd,cf_pdvd_on,cf_pdvd_off}_spots.tsv` are committed. The PNGs
  are regenerated by sec 0 (gitignored, as in doc 101).
* **Toolkit:** `cfg/pgrapher/experiment/{pdhd,protodunevd}/{clus,pr}.jsonnet` and
  `clus/test/doctest_blob_sampler_wrapped_channel.cxx`.
* **wcp:** `pdhd/wct-pr-perevt.jsonnet`, `pdvd/wct-pr-perevt.jsonnet`.
* **Work tags:**
  * data `028084_21_d102dbg` / `_d102cs`, `039253_8` and `039349_64` `_d102vdbg` / `_d102vcs`, `*_d102hcs`,
    `*_d102vcsall`;
  * simulation `90010[34]_k_d102{,b,brep,cs1,cs2,cs3}` and `90010[12]_k_d102lcs{1,2,3}`.

---

## 13. Round 2 (2026-09-15): the owner's go to flip, the flip's own configuration, and the fit knobs

**Owner request.** *"You should clearly flip the charge_stepped knob for PDHD and PDVD running. Is this also an
issue with the SBND? Aftr the charge_stepped knob on, I wonder if the previous knob on the track trajectory
fitting is still effective or not? Can you explain to me?"*

### 13.0 Repro (round 2)

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs; W=$HOME/tmp/d102
PIN=$W/libpin_d102; CS="-S retile_sampler_strategy='charge_stepped'"
sha256sum $F/102_pred2.txt          # 90a73668...  frozen 2026-09-15T07:50:21; every round-2 arm started 07:50:22
# data: the production configuration (fit knobs off = no trackfitting_config) + charge_stepped
ARM=d102hocs DET=pdhd SRC=d101hnew JOBS=3 PIN=$PIN PR_TLA="$CS" bash $S/d102_run_arms.sh
ARM=d102vocs DET=pdvd SRC=d101vnew JOBS=3 PIN=$PIN PR_TLA="$CS" bash $S/d102_run_arms.sh
# simulation: the fit-knob-OFF cells
for d in pdvd pdhd; do DET=$d PIN=$PIN JOBS=2 ARMS="d102o d102ocs d102locs" bash $S/d102_knob_arms.sh
  for a in d102o d102ocs; do python3 $S/d102_sim_fit_eval.py --det $d --arm $a --out $W/eval/eval_${d}_$a.tsv; done
  for a in d102locs d101koff; do python3 $S/d101_sim_fit_eval.py --det $d --arm $a --out $W/eval/eval_${d}_$a.tsv; done; done
python3 $S/d102_interaction_report.py --dir $W/eval --out $W/interaction_sim.tsv
python3 $S/d102_c4_decompose.py --base d102o --arms ocs
# gate
CFG_ROOT=$W/cfg_flip/cfg bash $S/d102_compile_pr.sh pdhd flipdef; CFG_ROOT=$W/cfg_flip/cfg bash $S/d102_compile_pr.sh pdhd flipst -S "retile_sampler_strategy='stepped'"   # and pdvd; cmp with on1_* / post_*
python3 $S/d102_resource_report.py pdhd:d101hnew:d102hocs pdvd:d101vnew:d102vocs
python3 $S/d101_stm_grade_pdhd.py d101hnew d102hocs d101hkf d102hcs
python3 $S/d102_michel_fp_list.py d101hnew d102hocs
M=$IMG/pdhd/stm_michel_scan; export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
python3 $M/prep_stm_michel_scan.py --det pdvd --arm d102vocs --pin-tranche $HOME/tmp/d101/sheet_d101vkf/pdvd_stm_michel_scan_sheet.tsv --outdir $W/prep_d102vocs --sheetdir $W/sheet_d102vocs
python3 $S/d101_stm_grade_pdvd.py d101vnew=$HOME/tmp/d101/prep_d101vnew d102vocs=$W/prep_d102vocs d101vkf=$HOME/tmp/d101/prep_d101vkf d102vcsall=$W/prep_d102vcsall2
python3 $S/d102_pdvd_moves.py d101vnew=$HOME/tmp/d101/prep_d101vnew d102vocs=$W/prep_d102vocs    # and d101vkf=... for the fit knobs
# reported: zig-zag census of the four data cells, reg/chord census, grid, SBND, the owner's event
python3 $S/d101_census.py --out $W/census101_2x2 --arm phd_stm_off_st:pdhd:d101hnew:stm ...   # 16 --arm specs: {pdhd,pdvd} x {stm,pr} x 4 cells
python3 $S/d102_census.py --out $W/census_r2_hd pdhd:d101hnew pdhd:d102hocs
python3 $S/d102_census.py --out $W/census_r2_vd_common --exclude 039252_5,039252_11,039349_14,039349_78 pdvd:d101vnew pdvd:d102vocs
python3 $S/d102_sbnd_grid.py --pd pdhd:d101hnew:30 --pd pdhd:d102hcs:30 --pd pdvd:d101vnew:30 --pd pdvd:d102vcsall:30
cd $D && python3 $S/d102_spot_figs.py --det pdhd --event 028084_21 --arms d101hnew,d102hocs --uv-rank-verified --out figs/102r2_prod_pdhd
```

### 13.1 Why the flip was measured before it was made

Production runs the doc 101 fit knobs **off**. Every `charge_stepped` data arm of round 1 ran them **on**, so the
configuration a flip creates had never been measured.

`figs/102_pred2.txt` was frozen before any round-2 arm. It declares the flip's own gate (sec A), the 2×2
interaction reading (sec B) and the SBND measurement (sec C).

The setting is cs1: prototype defaults, the only one measured on data.

### 13.2 The flip gate — FAILED (not flipped)

| clause | PDHD | PDVD |
|---|---|---|
| g1 every event that completes in the baseline completes | **PASS** (61 / 61) | **FAIL**: 039349_78 loses its only candidate (sec 13.2.3); 039252_11 is the reverse (0 in baseline, 2 in the arm) |
| g2 resources (median wall ≤ ×1.50, RSS ≤ ×1.25) | **PASS**: wall ×1.00 (p90 1.25), RSS ×1.009 (max 1.44); Steiner stage ×1.58, load control ×0.92 | **PASS**: wall ×0.76 (p90 1.00), RSS ×0.993 (max 1.65); Steiner ×1.06, control ×0.87 |
| g3 tags no worse than round 1's change (each limit is a round-1 delta measured on the knobs-on pair `d101hkf → d102hcs`; here it is applied to the production pair `d101hnew → d102hocs`, which `pred2.txt` does not spell out) | **FAIL**: STM efficiency −0.184 (limit 0.010). Michel purity 0.971 → 0.933 = −0.038 is inside its limit of 0.080; the verdict does not depend on this cell | **FAIL**: STM efficiency −0.123 (limit 0.017); Michel efficiency −0.177 (limit 0.034); Michel purity −0.131 (not bounded by g3) |
| g4 compile proof (scratch cfg tree) | **PASS**: flipped default ≡ `on1` 87a86589c767; flipped + `'stepped'` ≡ pre-flip 434208b1d840; 8 retile BlobSamplers differ, nothing else | **PASS**: ≡ 211a49a48229 / ≡ 920ecb4312ab; 16 BlobSamplers |

The pre-registered consequence is **stop and report before pushing**. The flip is not committed.

It is ready to apply: `figs/102r2_flip.patch`, one line per detector in `pr.jsonnet`, `null → 'charge_stepped'`.
From the toolkit root, `git apply --check` passes on it at 1b62d1da.
Explicit `'stepped'` stays the byte-identical escape hatch.

#### 13.2.1 The tags on all four cells (fixed denominator)

| det | cell | STM purity / efficiency | Michel purity / efficiency |
|---|---|---|---|
| PDHD (257) | knobs off, `stepped` (production) `d101hnew` | 0.974 / **0.776** | 0.971 / 0.791 |
| PDHD | knobs on, `stepped` `d101hkf` | 0.947 / 0.605 | 0.919 / 0.663 |
| PDHD | **knobs off, `charge_stepped` (the flip) `d102hocs`** | 0.946 / **0.592** | 0.933 / 0.651 |
| PDHD | knobs on, `charge_stepped` `d102hcs` | 0.937 / 0.605 | 0.839 / 0.605 |
| PDVD (546) | knobs off, `stepped` (production) `d101vnew` | 0.968 / **0.877** | **0.923 / 0.872** |
| PDVD | knobs on, `stepped` `d101vkf` | 0.927 / 0.739 | 0.830 / 0.713 |
| PDVD | **knobs off, `charge_stepped` (the flip) `d102vocs`** | 0.924 / **0.754** | **0.792 / 0.695** |
| PDVD | knobs on, `charge_stepped` `d102vcsall` | 0.922 / 0.732 | 0.837 / 0.689 |

#### 13.2.2 The sampler costs the tagger what the fit knobs cost it

Hand positives the production tagger finds and the changed trajectory loses: a dropped candidate (absent from
`T_stm_michel`) vs a changed verdict on a candidate still present.
* PDHD split: `d102_michel_fp_list.py`-style join.
* PDVD split: `d102_pdvd_moves.py`.

| change from production | PDHD STM lost (dropped / verdict), gained | PDVD STM lost (dropped / verdict), gained, new FP | PDVD Michel lost (dropped / verdict), new FP |
|---|---|---|---|
| sampler: → knobs off + `charge_stepped` | 32 (17 / 15), 5 | 47 (23 / 24), 13, 14 | 36 (14 / 22), 23 |
| fit knobs: → knobs on + `stepped` (doc 101) | 33 (18 / 15), 8 | 49 (26 / 23), 11, 15 | 33 (18 / 15), 17 |

The two levers act on different layers but cost the tagger the same, in count and in kind.
* **CheckSTM_Michel and TaggerCheckSTM were tuned on the production trajectory.** Any change to where the fitted
  points sit moves the kink / stop decisions (sec 13.2.3) and which clusters reach candidacy.
* **They do not stack.** Adding `charge_stepped` on top of the knobs costs PDHD STM efficiency nothing (0.605 →
  0.605).
* **Lists for adjudication:** `figs/102r2_michel_moves_pdhd.tsv` (PDHD: 3 new FP, 18 lost TP, 1 gone FP, 6 new TP)
  and `figs/102r2_michel_moves_pdvd.txt` (PDVD: the 23 new Michel FPs).

#### 13.2.3 PDVD 039349_78, the g1 failure

It is not a crash: the run finalizes in 9 s.

**Main-cluster stage: unchanged.** ClusteringFlagMatchedMains (43 mains), ClusteringUnmergeBundle (cluster 22:
649 + 3 associated) and TaggerCheckTGM are identical in both logs.

**Where it diverges: cluster 22's STM fit.**
* **Production:** pass 0 has status 7, kink at point 280 of 280 (the track end), exit 176.5 cm. The readout-edge
  and cathode guards run, and the result is `STM=1`.
* **The flip:** pass 0 has status 2, **kink at point 7**, exit 4.5 cm, left 158.1 cm. Pass 1 has status 3, and the
  result is `STM=0`. CheckSTM_Michel: "no STM-tagged main cluster; nothing to reconstruct".

The denser retile moves the seed. The refitted trajectory shows a kink 7 points from its start, and the tagger
takes that as the stop.

### 13.3 Are the fit knobs still effective under `charge_stepped`? Yes

**Mechanism.** The two knobs act inside `trajectory_fit` / `form_point_association` (doc 101 sec 2.3):
* `fit_weight_pow` 1.5 softens the (q/err)² position weight that pulls a point onto the brightest wire;
* `assoc_cont_center` 1 centres the association window on the continuous wire coordinate instead of the rounded
  wire.

Together they fight **wire-centre snapping**: the staircase zig-zag.

`charge_stepped` acts one layer earlier, on the retiled cloud. That changes the Steiner graph, the rough-path seed
and the cells a point can reach. It fights the **seed sitting off the charge** (spots 1–2) and the 3-wire grid.

Neither layer can do the other's job, so the expectation is that both still bite, and the 2×2 confirms it.

**Simulation** (`d102_interaction_report.py`; paired medians over tracks, mm). The knob effect is on − off; the
sampler effect is `charge_stepped` − `stepped`, knobs off.

| det, stratum | knob effect on res_t p90: @`stepped` → @`charge_stepped` | knob effect on chord dev p90: @`stepped` → @`charge_stepped` | sampler effect, knobs off: res_t p90 / dev p90 | pred2 reading |
|---|---|---|---|---|
| PDVD ISO (16) | −0.046 → −0.152 | −0.150 → −0.248 | −0.376 / −0.520 | still effective |
| PDVD θ ≤ 60° (12) | −0.025 → −0.185 | −0.385 → −0.228 | −0.029 / −0.140 | still effective |
| PDHD ISO (16) | −0.361 → −0.385 | −0.003 → −0.322 | −0.752 / −0.256 | still effective |
| PDHD θ ≤ 60° (12) | −0.257 → −0.272 | −0.335 → −0.387 | +0.003 / −0.004 | still effective |

(The report's cs/stepped ratios of 97 for PDHD ISO dev and 7.5 for PDVD low res_t come from near-zero denominators
and are not quoted.)

**Data, STM fit** (`d101_census.py`; zig-zag = chord-deviation p90, cm; dQ/dx robust CV; dip share; W lattice snap,
where 1.0 = none):

| det | knobs off, `stepped` | knobs on, `stepped` | knobs off, `charge_stepped` | knobs on, `charge_stepped` |
|---|---|---|---|---|
| PDHD | 0.564 / 0.247 / 4.2 % / 1.23 | 0.542 / 0.242 / 4.1 % / 1.13 | 0.451 / 0.218 / 3.0 % / 1.24 | **0.430 / 0.210 / 3.0 % / 1.15** |
| PDVD | 0.478 / 0.212 / 3.6 % / 1.55 | 0.445 / 0.209 / 3.4 % / 1.44 | 0.330 / 0.197 / 2.1 % / 1.59 | **0.311 / 0.194 / 2.1 % / 1.46** |

**Reading.**
* **On data the sampler does most of the work.** It removes 0.11–0.15 cm of zig-zag p90 and ~1.3 points of dips;
  the knobs remove 0.02–0.03 cm.
* **Only the knobs reduce lattice snap**, and they still do under `charge_stepped` (PDVD W 1.59 → 1.46, PDHD
  1.24 → 1.15). The sampler leaves snap unchanged or slightly worse.
* **The best trajectory is both on** (last column), on both detectors.

**On the owner's event** (knobs off + `charge_stepped` vs production; `figs/102r2_prod_pdhd_spots.tsv`). The
sampler alone repairs spots 1–3:
* spot 1: largest distance from ridge 1.78 → 0.84 cm, V on-charge 0.80 → 1.00, zero rows 9 → 0;
* spot 2: 2.34 → 0.77 cm, V/W on-charge 0.76 / 0.80 → 1.00, PR segments 3 → 1;
* **spot 3: dQ/dx median 12.7 → 49.5 ke/cm, zero rows 23 → 0**, the fit back on all three planes. This was the spot
  round 1 credited to the fit knobs.

Spot 4 (cl26) again gets no PR fit; see 13.5.

**Segments, knobs-off pair** (`d102_c4_decompose.py --base d102o --arms ocs`):
* PDVD: k9, a 9-point stub at the track start beside an improved main; k11, the same off-line side branch as round
  1.
* PDHD: k6, a 4-point stub at the start; k8, a 7-point stub at the track **end**, main +0.15 mm.

### 13.4 SBND

**Same divergence in code.** `sbnd/clus.jsonnet:200-206` hard-codes `strategy: ['stepped']` in `bs_live_face`, and
the Steiner retile (`improve_cluster_2`, :2044-2048) uses it. The SBND PR job exposes **no** argument for it. Nothing
was changed for SBND.

**Measured like-for-like** (`d102_sbnd_grid.py`): in-slice nearest-neighbour spacing of the calib-dump retiled
cloud vs the image, SBND `d101snew` (16 events) against the first 30 events of the PDHD/PDVD arms.

| det (pitch) | cloud median | image median | cloud / image | cloud in pitches | near-ISO cloud / image | with `charge_stepped` |
|---|---|---|---|---|---|---|
| SBND (0.30 cm) | 1.039 cm | 0.693 cm | ×1.50 | 3.5 | ×1.38 | — (no knob) |
| PDHD (0.479 cm) | 1.478 cm | 0.985 cm | ×1.50 | 3.1 | ×1.35 | 0.800 cm (×0.81) |
| PDVD (0.51 cm W) | 1.767 cm | 1.062 cm | ×1.66 | 3.5 | ×1.39 | 0.883 cm (×0.83) |

**Answer.**
* **Yes, the same issue exists in kind.** SBND's retiled cloud is as much sparser than its image as PDHD's, in
  pitches and in ratio.
* **It is milder in effect.** The absolute spacing is 1 cm rather than 1.5–1.8 cm, and doc 101 sec 1 measured
  SBND's fit symptoms at a fraction of PDHD's: PR chord dev p90 0.31 vs 0.56 cm, dips 1.2 % vs 4.3 %, snap ≈ 1.0.
* **Indicative only.** SBND has 83 Steiner-cloud clusters against ~3000 per 30 PDHD events.

Measuring `charge_stepped` on SBND would need a new default-OFF knob in `sbnd/clus.jsonnet`, plus the same arms.
That is not done here.

### 13.5 Correction to sec 2 / sec 3: cl26

The round-1 statement "cl26 loses its PR fit" is **real, not a cluster-id renumbering**. Matched by image points,
cl26 (16 137 points) and cl128 are the same clusters, with the same ids, in all four cells. In both
`charge_stepped` arms cl26 and cl128 have zero `track_fit` rows.

What the round-1 text missed: in the same arms a **separate** cluster, **cl27** (1185 points, 100 cm from cl26),
gains the PR fit and becomes an STM candidate. It has none under `stepped`.

So on this event the retile changes **which cluster the PR stage fits**, most likely the main-cluster choice
inside a bundle. That is candidate churn upstream of the fit, not a fit that fails. Bundle membership was not
checked.

### 13.6 Census, knobs-off pair (reported)

| | PDHD `d101hnew` → `d102hocs` (61) | PDVD `d101vnew` → `d102vocs` (116 common) |
|---|---|---|
| `reg_flag` share 65–75° / 75–85° / 85–90° | 0.137 / 0.163 / 0.269 → **0.113 / 0.147** / 0.259 | 0.258 / 0.335 / 0.389 → **0.212 / 0.289** / 0.384 |
| off-image share 85–90° | 0.102 → 0.099 | 0.142 → 0.156 |
| chords (≥ 10 off-image rows): count, total length | 289, 9477 cm → 288, 8785 cm | 297, 6482 cm → 298, 6802 cm |
| Steiner in-slice spacing, 3-wire share | 1.478 cm, 0.594 → 0.800 cm, 0.102 | 1.767 cm, 0.064 → 0.883 cm, 0.104 |

Same picture as round 1:
* `reg_flag` falls by 12–18 % up to 85°;
* the most isochronous bin does not improve;
* chords stay, because they are Steiner-graph bridges (sec 5.3). But the direction differs by detector:
  * PDHD total chord length falls 7 % (9477 → 8785 cm);
  * **PDVD's rises 5 % (6482 → 6802 cm)**, with its off-image share at 85–90° up 0.142 → 0.156.

  On PDVD the knobs-off sampler slightly worsens the near-isochronous off-image stretches.

### 13.7 What a flip would need

**The obstacle is the tagger, not the sampler.**
* Both trajectory improvements, the sampler and the fit knobs, cost the STM/Michel chain the same. They cost it
  through the same two channels: candidates that stop reaching CheckSTM_Michel, and kink/stop verdicts that move
  (039349_78).
* Neither lever makes the trajectory worse. On data they are the two best zig-zag cells, and together the best.

The owner's opening message already said "before we retune the tagger". The consistent path is:
1. **Adopt the best trajectory as the tagger-retune baseline:** knobs on + `charge_stepped` (`d102hcs` /
   `d102vcsall`). Both samples are on disk with fixed-denominator grades.
2. **Adjudicate the new false positives** in `figs/102r2_michel_moves_{pdhd,pdvd}.*` (most PDHD verdicts are agent
   labels, not owner labels), so that the retune does not chase mislabels.
3. **Retune** TaggerCheckSTM's kink/stop reading and CheckSTM_Michel's admission on that baseline.
4. **Flip the sampler (`figs/102r2_flip.patch`), the fit knobs and the retuned tagger together**, graded on the
   same records.

Flipping the sampler alone now would cost production 12–18 points of STM efficiency on both detectors until the
retune lands.

### 13.8 Files (round 2)

* **Scripts:**
  * new: `d102_interaction_report.py`, `d102_sbnd_grid.py`, `d102_michel_fp_list.py`, `d102_pdvd_moves.py`;
  * extended: `d102_knob_arms.sh` (arms `d102o` / `d102ocs` / `d102locs`), `d102_compile_pr.sh` (`CFG_ROOT`),
    `d102_c4_decompose.py` (`--base`), `d102_census.py` (`--exclude`).
* **Figures and records:** `figs/102_pred.txt` and `figs/102_pred2.txt` (the two frozen rules), `figs/102r2_flip.patch`,
  `figs/102r2_prod_pdhd_spots.tsv`, `figs/102r2_michel_moves_pdhd.tsv`, `figs/102r2_michel_moves_pdvd.txt`.
* **Work tags:** data `*_d102hocs`, `*_d102vocs`; simulation `90010[34]_k_d102{o,ocs}`, `90010[12]_k_d102locs`.

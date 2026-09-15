# 101 — Sampling points, Steiner terminals and the trajectory / dQ/dx fit at coarse wire pitch (PDHD, PDVD vs SBND)

**Status (2026-09-14).**
* **Finding:** the coarse-pitch zig-zag and the low dQ/dx off the track are made in the trajectory
  fit, not in the Steiner terminals, and half-pitch sampling is not the fix (sec 5).
* **Knobs:** default-OFF knobs are in toolkit `419f70f3`.  `fit_weight_pow` 1.5 +
  `assoc_cont_center` 1 passes the pre-registered trajectory rule on simulated muons of both
  detectors (sec 4.4) and smooths the data trajectories (sec 6.4).
* **But it regresses the STM/Michel tags on both detectors' hand-scan records** (sec 6.5; fixed
  denominator, knobs off → on): STM efficiency PDHD 0.776 → 0.605, PDVD 0.877 → 0.739; Michel purity
  PDVD 0.923 → 0.830.
* **Not recommended.**  Flipping it would first need CheckSTM_Michel retuned against the softer fit,
  and in production `-nu` the fit has no consumer except the two STM taggers (sec 7).  The knobs stay
  as study instruments.
* **Gates:** byte-identical when off on every gate (sec 6.2).
* **NOT flipped.**

**Owner request (2026-09-14).** *"Compare the sampling points and Steiner Graph/Steiner Terminal
construction for PDVD, PDHD, and sbnd_xin … due to the different wire pitch and wire angle … the
Steiner Terminal construction in the PDHD and PDVD is not ideal. For example, see
https://www.phy.bnl.gov/twister/bee/set/c75d14ee-07ae-4ebb-bcaf-1cb2b8451a55/event/0/, the track
trajectory is zig-zaged, and the dQ/dx have a couple symptoms: 1. wiggled dQ/dx high and low
oscillation behavior 2. some region when the track trajectory deviates a bit from the real track
trajectory …, the dQ/dx is very low … shall we consider to sample not at one wire pitch, but at say
half wire pitch for PDVD and PDHD? Is there a better way to define the Steiner Terminals, so that
they are closer to the true charge locations? … use the toy code … run the simulation … commit and
push."*  Owner decisions the same day: study **plus a default-OFF C++ knob prototype** for a lever
that clearly wins on truth (no production flip); a **Python toy anchored by the full WCT chain** on
simulated single muons.

The Bee example is **PDHD 028084_21, cluster 55** of production arm `h28prod` (doc pdvd/97 §2.2).

---

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; S=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# sec 1 -- read-only census of the production fits (PDHD h28prod, PDVD p96vprod, SBND d146sv25)
python3 $S/d101_census.py --out $HOME/tmp/d101/census
# sec 3 -- toy calibration (base only) and the lever scan
python3 $S/d101_toy_scan.py --out $HOME/tmp/d101/cal_f30_n800 --variants base --nseed 1 --nproc 4 --fluct 0.30 --noise 800 --thr 2400
python3 $S/d101_toy_scan.py --out $HOME/tmp/d101/scan_f20_n800 --nseed 3 --nproc 12 --fluct 0.20 --noise 800 --thr 2400
python3 $S/d101_toy_report.py $HOME/tmp/d101/scan_f20_n800
python3 $S/d101_toy_scan.py --out $HOME/tmp/d101/scan3_f20_n800 --nseed 3 --nproc 12 --fluct 0.20 --noise 800 --thr 2400 \
    --variants base,wpow15_cc,wpow15_cc_hs1,hs1_steiner,wpow15_cc_hs1_steiner   # sec 3.6
# sec 4.1 -- truth, simulation -> NF/SP -> imaging -> clustering (pin = pre-round libraries)
F=$IMG/pdvd/docs/nf_sp_img_clus/figs
python3 $S/d101_make_muons.py --det pdvd --cfg $HOME/tmp/d101/sim/cfg_probe/pdvd_a1.json --anode 1 --face 0 --run 900101 --out $HOME/tmp/d101/sim/truth_pdvd.json
python3 $S/d101_make_muons.py --det pdhd --cfg $HOME/tmp/d101/sim/cfg_probe/pdhd_a1.json --anode 1 --face 1 --run 900102 --out $HOME/tmp/d101/sim/truth_pdhd.json
#   (copies committed as $F/101_truth_{pdvd,pdhd}.json; the cfg_probe JSONs are the wcsonnet-compiled sim drivers)
for k in $(seq 0 17); do DET=pdvd K=$k PIN=$HOME/tmp/d101/libpin bash $S/d101_sim_chain.sh; done   # and DET=pdhd
python3 $S/d101_make_sim_tf_json.py         # figs/101_tf_sim_{pdvd,pdhd}.json
# sec 4.2 -- baseline PR arm, its repeat, grading
for d in pdvd pdhd; do for a in d101b d101brep; do
  ARM=$a DET=$d PIN=$HOME/tmp/d101/libpin_new JOBS=4 bash $S/d101_sim_pr_arm.sh
  python3 $S/d101_sim_fit_eval.py --det $d --arm $a --out $HOME/tmp/d101/eval_${a}_$d.tsv; done; done
# sec 4.4 -- knob arms (pin = this round's build) and the pre-registered report
for d in pdvd pdhd; do DET=$d PIN=$HOME/tmp/d101/libpin_k JOBS=4 bash $S/d101_knob_arms.sh
  for a in d101koff d101kf d101ks d101kc; do python3 $S/d101_sim_fit_eval.py --det $d --arm $a --out $HOME/tmp/d101/eval_${a}_$d.tsv; done; done
python3 $S/d101_knob_report.py --dir $HOME/tmp/d101 --out $HOME/tmp/d101/knob_report.tsv
# sec 6.2 -- knob-off gates: data PR arms on both pins, SBND, clustering (see d101_gates.sh header), then
ARM=d101vold DET=pdvd SRC=d16vnu PIN=$HOME/tmp/d101/libpin   JOBS=4 bash $S/d53_run_arms.sh   # d101vnew: libpin_k
ARM=d101hold DET=pdhd SRC=d16hnu PIN=$HOME/tmp/d101/libpin   JOBS=3 bash $S/d53_run_arms.sh   # d101hnew: libpin_k
TAG=d101sold PIN=$HOME/tmp/d101/libpin JOBS=3 bash $S/d101_sbnd_arm.sh                       # d101snew: libpin_k
GATES=1,2,3,4,5,6 bash $S/d101_gates.sh > $HOME/tmp/d101/gates.log 2>&1
# sec 6.3 -- knob-ON smoke on 028084_21 and the data knob-ON census arms
python3 $S/d101_smoke_cl55.py
ARM=d101vkf DET=pdvd SRC=d16vnu PIN=$HOME/tmp/d101/libpin_k JOBS=4 PR_TLA="-A trackfitting_config=$F/101_tf_prod_pdvd_kf.json" bash $S/d53_run_arms.sh
python3 $S/d101_census.py --out $HOME/tmp/d101/census_kf2 --arm pdvd_stm_off:pdvd:d101vnew:stm --arm pdvd_stm_kf:pdvd:d101vkf:stm \
    --arm pdvd_pr_off:pdvd:d101vnew:pr --arm pdvd_pr_kf:pdvd:d101vkf:pr ...   # same four for pdhd d101hnew / d101hkf
# sec 6.5 -- STM/Michel tags against the hand-scan records (fixed denominator)
python3 $S/d101_stm_grade_pdhd.py p82bhoff d101hnew d101hkf > $HOME/tmp/d101/grade_pdhd_fixed.log
M=$IMG/pdhd/stm_michel_scan; W=$HOME/tmp/d101
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
for t in d101vnew d101vkf; do python3 $M/prep_stm_michel_scan.py --det pdvd --arm $t --outdir $W/prep_$t --sheetdir $W/sheet_$t
  python3 $M/census_score.py --prep $W/prep_$t --arm $t --json $W/score_$t.json > $W/score_$t.log; done
python3 $S/d101_stm_grade_pdvd.py d101vnew=$W/prep_d101vnew d101vkf=$W/prep_d101vkf > $W/grade_pdvd_fixed.log
```

---

## 1. What the production fits look like on the three detectors (read-only census)

Instrument `d101_census.py` over every accepted fit run (≥ 40 points, ≥ 20 cm, 3 cm trimmed at each
end), `T_rec_charge` of `tracking-stm.root` (TaggerCheckSTM single-track fit) and `tracking-pr.root`
(PR multi-track fit):

| arm | runs | points | zig-zag: chord dev med / p90 (cm) | dQ/dx robust CV | dips < 0.5× local median | ACF(1) | corr(dev, dQ/dx) | lattice snap U / V / W / (t) |
|---|---|---|---|---|---|---|---|---|
| PDHD stm (h28prod) | 343 | 92 781 | 0.196 / 0.564 | 0.247 | 4.22 % | 0.80 | −0.07 | 1.09 / 1.07 / **1.23** / 1.10 |
| PDHD pr | 649 | 91 772 | 0.236 / 0.563 | 0.323 | 4.30 % | 0.76 | −0.08 | 1.03 / 1.03 / 1.16 / 1.08 |
| PDVD stm (p96vprod) | 681 | 149 759 | 0.155 / 0.478 | 0.212 | 3.58 % | 0.59 | −0.11 | **1.39** / 1.19 / **1.55** / 1.10 |
| PDVD pr | 1090 | 148 054 | 0.151 / 0.375 | 0.206 | 3.01 % | 0.54 | −0.10 | 1.23 / 1.12 / 1.37 / 1.01 |
| SBND pr (mcp2k d146sv25) | 74 | 11 703 | 0.109 / 0.314 | 0.176 | 1.23 % | 0.42 | −0.06 | 1.02 / 1.01 / 1.03 / 0.99 |
| SBND pr (mcp1k d146sv25) | 32 | 4 808 | 0.129 / 0.323 | 0.187 | 1.16 % | 0.50 | −0.15 | 1.02 / 1.00 / 1.15 / 1.01 |

**Lattice snap index** = share of fitted points whose projected wire coordinate `p*` lies within
±0.15 wire of an integer, over the 0.30 a uniform phase gives.  Integer `p*` **is the wire centre**:
`TrackFitting.cxx:829-857` builds `pu = (y2d − center)/pitch − 0.5`, and the ctpc stores wire `wind` at
`pitch·(wind+0.5)+center` (`PointTreeBuilding.cxx:313`).  `pt` (drift slice) is the control.

Read in one line: **fitted points pile up on wire centres in proportion to the pitch** (PDVD W 1.55,
U 1.39; PDHD W 1.23; SBND ≈ 1.0), and the dQ/dx dips are 2.5–3.5× more frequent on the ProtoDUNEs.
The Bee example itself (028084_21 cl55, pass 0): 267 points, chord deviation median 0.20 / p90 0.69 /
max 2.30 cm, dQ/dx CV 0.84, corr(deviation, dQ/dx) = **−0.32** — the owner's symptom 2 as a number.

Caveat carried to every later section: SBND is MC, the ProtoDUNEs are data.  The census alone cannot
separate "pitch" from "data".  That is what the toy (sec 3) and the simulation (sec 4) are for.

---

## 2. The chain, and where a coarse pitch enters

(Every citation is toolkit `apply-pointcloud` at `67937f45`.)

### 2.1 Sampling points
`BlobSampler::Stepped` (`clus/src/BlobSampler.cxx:782-977`) — used by imaging AND by the PR retile
(`ImproveCluster_2`, same `BlobSampler:live-<a>-<f>` instance):
* steps in **wire units**: `max(3, width/12)` wires in the fewest- and most-wire views, plus the last wire;
* each point is the **wire-centre crossing** of a (min-view, max-view) wire pair (`offset 0.5`, `:882`),
  kept if it falls inside the third view's strip (±0.03);
* its drift coordinate is the **slice start**, not the slice centre (`bins.edge(0)`, `:457-487`).

So every sampled point sits on the (U,V,W)×slice lattice: at PDVD's 7.65 mm U/V pitch a blob of a
MIP track holds 1–2 wires per view and the candidate positions are ≥ 3.8 mm apart.

### 2.2 Steiner terminals and graph
* candidates: `calc_charge_wcp` = rms of the point's three per-plane cell charges, all planes above the
  threshold (`Facade_Cluster.cxx:1031-1112`); 500 e on PDVD/PDHD, 4000 e on SBND;
* one local maximum per **blob** (`SteinerGrapher.cxx:769-815`; doc pdvd/31 r6-r8: terminal density =
  candidate-bearing blob density ≈ one per slice), thinned at 0.5 cm (P3b, doc pdvd/37), plus extremes;
* Voronoi-union Steiner graph with edge weight × (0.8 + 0.4·avg Q0/(Q+Q0)) (`:1224-1406`);
* the segment's seed path = Dijkstra over **`steiner_pc` points, which are retiled lattice points**.

### 2.3 The fit
`TrackFitting::do_single_tracking` (`TrackFitting.cxx:10077-10563`), passes 1.2 cm then 0.6 cm:
* `form_map` → `form_point_association` (`:2940-3340`): the cells a trajectory point may use.  The
  point's wire is `point2wind` = `std::round` of the continuous coordinate (`Facade_Util.cxx:550-557`),
  the admitted range is `j = round(cur − half) … round(cur + half)` around that **integer** wire, with
  `half = sqrt(range)/pitch` from `calculate_ranges_simplified` and a distance cut
  `min(step·0.9, 1.2 cm)` = **0.54 cm on the 0.6 cm pass — 0.71 wire on PDVD U/V, 1.16 on PDHD, 1.8 on SBND**;
* `trajectory_fit` (`:5397-6060`): per point, a least-squares fit whose wire rows are
  `(q/err)·fac·(wire − proj(p))` (`:5656`) — i.e. the fitted pitch coordinate is a weighted centroid of
  **integer wire indices**, weight `(q/err)²`.  The measured `charge_err` is nearly flat in q (sec 2.4),
  so the weight goes as **q²**: the centroid is pulled onto the dominant wire;
* `dQ_dx_fit` (`:8501-9286`): Gaussian response per point (diffusion ⊕ effective width), overlap
  weights, second-difference regularisation.

### 2.4 Charge error is flat, so the position weight is q²
`T_proj_data` cells of fitted clusters (`charge_err` vs `charge`):

| q band (e) | PDHD err median | PDVD err median | SBND err median |
|---|---|---|---|
| 2–5 k | 1600 | 1105 | 1155 |
| 5–10 k | 1614 | 1204 | 1190 |
| 10–20 k | 1585 | 1237 | 1389 |
| > 20 k | 1842 | 1637 | 1772 |

### 2.5 The hypothesis this implies (to be tested, sec 3-4)
When the transverse charge spread is small compared with the pitch (PDVD: σ ≈ 2–3 mm vs 7.65 mm), a
track between two wires puts most charge on one wire; the association window is centred on the
nearest wire and the q²-weighted centroid lands near that wire's centre.  Fitted points then hop from
wire to wire as the track crosses the lattice: a staircase (the zig-zag), an inflated path length `dx`
and a response model evaluated at the wrong transverse position (dQ/dx low where the path leaves the
charge).  Nothing in this mechanism is specific to Steiner terminals — the seed only decides where the
first window is centred — which is exactly what the toy's oracle decomposition must check.

---

## 3. The toy

### 3.1 What it is
`d101_toy.py` runs the whole PR-stage chain in Python on the **real wire lattices** (pitch, wire
direction and wire-0 origin read from the production wires files): truth muon → per-(plane, wire,
4-tick slice) charge with diffusion ⊕ the simulation's effective transverse width (doc pdvd/47 S1)
and flat noise → per-slice blob tiling → an exact port of `Stepped` → base graph → `calc_charge_wcp`
per-blob peaks + 0.5 cm thinning + extremes → Voronoi-union Steiner graph with the charge-weighted
edges → Dijkstra seed → `do_single_tracking` (organize 1.2 cm → form_map/association/examine →
trajectory_fit method 1 → skip/area smoothing → organize 0.6 cm → method 2 → final fill charge test
→ `dQ_dx_fit` with response integrals, compact-matrix overlap weights and the regulariser).  Every
step cites its toolkit lines in the module header, and so does every **stated simplification**
(single face, no dead or wrapped channels, no PR retile re-run, a proxy base graph, Steiner P2/P3
skipped, no end trim).  Each lever the study scans is one field of `Opts`; `Opts()` is production.

Tracks: 100 cm straight muons on a grid of angle to the drift axis θ ∈ {80, 60, 40, 20}° × azimuth
from the collection-wire direction φ ∈ {0, 30, 60, 90}° × 3 seeds, each with a random sub-pitch
placement, i.e. 48 tracks per detector.  Metrics are taken 3 cm inside each end, against truth:
transverse residual in the plane perpendicular to both the track and the drift axis (`res_t`, mm),
the same ±3 cm chord deviation the data census uses (`dev`, mm), the lattice snap index per plane,
dQ/dx over the true value (robust CV, dips, ACF(1)).

### 3.2 Two fidelity checks before believing any lever
1. **The toy reproduces the data's lattice snapping with nothing tuned.**  Base scan (noise 300 e,
   no fluctuation, 16 tracks/detector):

   | | snap U / V / W, toy | snap U / V / W, data (sec 1, stm) | toy res_t med / p90 (mm) |
   |---|---|---|---|
   | PDVD | 1.24 / **1.55** / 1.34 | 1.39 / 1.19 / **1.55** | 0.57 / 1.20 |
   | PDHD | 1.13 / 1.11 / 1.20 | 1.09 / 1.07 / 1.23 | 0.22 / 0.71 |
   | SBND | 0.99 / 1.05 / 1.06 | 1.02 / 1.01 / 1.03 | 0.09 / 0.23 |

   The ordering by pitch, the magnitude on PDHD and SBND, and "snapping on PDVD, none on SBND" all
   come out; which PDVD plane is worst differs (toy V, data W) — the toy has no wire-angle-dependent
   SP, so plane-by-plane is not claimed.
2. **The dQ/dx spread calibrates on SBND and then predicts PDVD.**  With a Landau-like Moyal factor
   per 3 mm of track as the only free parameter:

   | fluct scale (noise 800 e / thr 2400 e) | SBND CV | PDVD CV | PDHD CV | dips SBND / PDVD / PDHD |
   |---|---|---|---|---|
   | 0.10 | 0.072 | 0.091 | 0.082 | 0 / 0 / 0 |
   | 0.20 | 0.124 | 0.149 | 0.135 | 0 / 0 / 0 |
   | **0.30** | **0.174** (data 0.176–0.187) | **0.208** (data **0.206–0.212**) | 0.188 (data 0.247–0.323) | 0 / 0.7 % / 0.3 % |

   Fixing the one parameter on SBND reproduces PDVD's dQ/dx spread to 0.003.  PDHD's data are wider
   than the toy by 0.06–0.13 and **all three detectors' data carry more dips** (1.2 / 3.0–3.6 /
   4.2 %) than the clean-track toy (0 / 0.7 / 0.3 %).  The dips are therefore NOT this lattice
   mechanism on a clean track; PDHD's excess matches the drift-volume charge-completeness deficit of
   doc pdvd/50 §4.2 in kind, which is left as a lead, not a claim.  Likewise the data's chord-deviation
   p90 (3.1–5.6 mm) is 3–6× the toy's on every detector including SBND — something the toy does not
   contain (sec 4 asks the full chain).

### 3.3 Which stage owns the error: the oracle
Replacing the Steiner seed by the TRUE track (the fit then runs unchanged) removes only a small part
of the residual (paired over the same 48 tracks per detector, variant − production):

| | production res_t p90 | truth seed: Δres_t p90 | Δdev p90 |
|---|---|---|---|
| PDVD | 1.16 mm | −0.07 (41/48 tracks improve) | −0.08 |
| PDHD | 0.61 mm | −0.05 | −0.03 |
| SBND | 0.25 mm | −0.02 | +0.04 |

**Even a perfect set of Steiner terminals would move the fitted trajectory by ≤ 6 %.**  The
zig-zag the toy reproduces is made in the fit, where sec 2.3 located it.

### 3.4 Round-1 lever scan (paired Δ = variant − production, median over 48 tracks; mm)

| lever | PDVD Δres_t med / p90 / Δdev p90 | PDHD | SBND |
|---|---|---|---|
| A1 sampling min step 3 → 1 wire | −0.001 / −0.015 / −0.088 | −0.020 / −0.111 / −0.299 | −0.002 / −0.005 / −0.024 |
| A2 **half-pitch sampling** (owner) | −0.010 / −0.040 / −0.135 | −0.022 / −0.092 / −0.232 | −0.006 / −0.008 / −0.016 |
| A1+A2 | −0.031 / −0.050 / −0.161 | −0.025 / **−0.145** / **−0.271** | −0.006 / −0.022 / −0.023 |
| A3 slice-centre x | +0.001 / +0.044 / +0.136 | +0.008 / +0.046 / +0.112 | +0.006 / +0.019 / +0.086 |
| B1 terminals moved to 3-view charge centroid | −0.019 / −0.032 / −0.022 | −0.013 / −0.016 / −0.027 | −0.005 / −0.008 / +0.009 |
| B2 thinning at 1 pitch / none | ≈ 0 | ≈ 0 | 0 |
| C1 association floor 0.5–1.0 pitch | ≈ 0 (p1.0: dev −0.043) | 0 | 0 |
| C2 window centred on the continuous coordinate | −0.007 / −0.016 / −0.028 | −0.006 / −0.017 / −0.039 | −0.004 / −0.002 / +0.004 |
| C3 **position weight (q/err)¹ instead of (q/err)²** | **−0.190 / −0.316 / −0.225** | +0.046 / +0.068 / +0.077 | +0.033 / +0.072 / +0.065 |
| C4 div_sigma floor = 1 pitch | −0.008 / −0.034 / −0.038 | 0 | 0 |

Snap index moves with res_t (C3 on PDVD: −0.15 / −0.12 / −0.11 on U / V / W).  **No lever moves the
toy's dQ/dx CV by more than 0.007 or its ACF(1) by more than 0.016** — in the clean-track toy the
dQ/dx fit is robust to the trajectory changes these levers make.

Reading: (i) the owner's half-pitch sampling helps on every detector and most on PDHD (−24 % p90,
−27 % chord p90), never hurts; (ii) the largest single lever on PDVD is the fit's position weight,
but as a global change it degrades PDHD and SBND, so it cannot ship as-is; (iii) terminal
re-definition alone is small, as the oracle predicts.  Round 2 tests the combinations that could win
on all three (sec 3.5).

### 3.5 Round-2 combinations
Same 48 tracks per detector, same settings (`scan2_f20_n800`); paired Δ vs production, mm; the
count is tracks whose chord p90 improved:

| combination | PDVD Δres_t med / p90 / Δdev p90 | PDHD | SBND |
|---|---|---|---|
| weight (q/err)^1.5 | −0.100 / −0.216 / −0.162 | +0.007 / +0.003 / +0.002 | +0.007 / +0.020 / +0.010 |
| (q/err)^1 + centred window | −0.205 / −0.375 / −0.239 | +0.033 / +0.047 / +0.079 | +0.031 / +0.073 / +0.063 |
| (q/err)^1.5 + centred window | −0.105 / −0.222 / −0.200 | −0.014 / −0.027 / −0.033 | +0.007 / +0.023 / +0.008 |
| half-pitch + step 1 + centred window | −0.040 / −0.098 / −0.215 (46/48) | −0.033 / −0.139 / −0.303 (42/48) | −0.009 / −0.022 / −0.037 (34/48) |
| (q/err)^1 + half-pitch + step 1 | −0.213 / −0.565 / −0.523 | −0.007 / −0.064 / −0.101 | +0.014 / +0.029 / +0.030 |
| (q/err)^1 + centred + half-pitch + step 1 | **−0.249 / −0.601 / −0.582** (47/48) | −0.016 / −0.104 / −0.199 | +0.011 / +0.031 / +0.028 |
| **(q/err)^1.5 + centred + half-pitch + step 1** | **−0.173 / −0.356 / −0.440 (47/48)** | **−0.035 / −0.146 / −0.322 (46/48)** | **−0.012 / −0.026 / −0.020 (31/48)** |
| truth seed + (q/err)^1 + centred | −0.276 / −0.568 / −0.405 | −0.024 / −0.035 / +0.055 | +0.009 / +0.015 / +0.050 |

Only **(q/err)^1.5 + a window centred on the continuous coordinate + half-pitch / step-1 sampling**
improves all three detectors: PDVD transverse p90 −31 %, chord p90 −34 %; PDHD −24 % / −32 %;
SBND −10 % / −4 %.  The linear weight is PDVD's best but still costs SBND.  None of the rows moves the
toy's dQ/dx CV by more than 0.006.

**Scope caveat, checked in sec 3.6:** in the toy the sampling levers change both the cloud the fit
associates against and the Steiner cloud.  In the C++ chain the fit's association reads the
clustering job's points; a PR-job sampler knob changes only the retiled Steiner cloud.

### 3.6 Sampling levers scoped to the Steiner stage only
`Opts.sampler_scope = "steiner"`: the Steiner stage (terminals, graph, seed path) uses the lever's
sampling, while the fit associates against production-sampled points, as a knob on the PR job's
retile samplers would act (`scan3_f20_n800`, same 48 tracks, paired Δ, mm, 90 % bootstrap CI of the
median in brackets where it touches zero):

| variant | PDVD Δres_t med / p90 / Δdev p90 | PDHD | SBND |
|---|---|---|---|
| half-pitch + step 1, both scopes (sec 3.4 A1+A2) | −0.031 / −0.050 / −0.161 | −0.025 / −0.145 / −0.271 | −0.006 / −0.022 / −0.023 |
| half-pitch + step 1, **Steiner only** | −0.024 / −0.037 / −0.102 | −0.021 / −0.121 / −0.192 | −0.005 / −0.018 / −0.014 [−0.030, +0.017] |
| (q/err)^1.5 + centred window (fit only) | −0.105 / −0.222 / −0.200 | −0.014 / −0.027 / −0.033 [−0.092, +0.013] | **+0.007 / +0.023 / +0.008** (p90 CI [+0.003, +0.030]) |
| (q/err)^1.5 + centred + half-pitch + step 1, both scopes | −0.173 / −0.356 / −0.440 (47/48) | −0.035 / −0.146 / −0.322 (46/48) | −0.012 / −0.026 / −0.020 (31/48) |
| (q/err)^1.5 + centred + half-pitch + step 1, **Steiner only** | −0.153 / **−0.358** / −0.351 (44/48) | −0.034 / −0.117 / −0.214 (44/48) | −0.008 / −0.027 / −0.002 (every CI touches 0) |

Read: the buildable form of the round-2 winner, with the sampling knob on the retile samplers only,
keeps **all of PDVD's transverse-p90 gain and 80 % of its chord gain, and 80 % / 66 % on PDHD**, and
is neutral on SBND.  The fit-only pair costs SBND a small but CI-significant +0.023 mm, which the
sampling part offsets.  The SBND numbers are small in absolute terms; with 31–33 of 48 tracks
improving, the honest statement is "does not degrade SBND", not "improves SBND".  None of the rows
moves the toy's dQ/dx CV by more than 0.005.

## 4. The full-chain simulation anchor

### 4.0 Pre-registered selection rule for the knob arms (written 2026-09-14 before any knob-ON arm ran)
The plan's rule asked for the toy to improve "the dQ/dx low-dip fraction" as well.  The clean-track
toy has almost no dips (0 / 0.7 / 0.3 %, sec 3.2), so that clause cannot be judged in the toy.  It is
replaced by a no-regression clause on the full chain, below.

* **Arms** (18 simulated muons per detector, the same pctree as the baseline `d101b`):
  `d101kf` = fit knobs only (`fit_weight_pow` 1.5, `assoc_cont_center` 1, in the fit JSON);
  `d101ks` = retile sampler only (`retile_sampler_half_pitch` true, `retile_sampler_min_step` 1);
  `d101kc` = both.
* **Noise floor.** The repeat arm `d101brep` (same pin, same inputs) reproduces `d101b` exactly,
  every `T_rec_charge` value on 18/18 events per detector.  A paired Δ between arms is therefore
  exact, not a draw.
* **Verdict stratum:** the 12 grid tracks with θ ≤ 60°.  The four θ = 80° (near-isochronous) tracks are
  reported separately and are **not** in the verdict.  Their clusters already sit 2.1 (PDVD) / 3.1
  (PDHD) cm p90 from truth before any fit.  Over all 16 tracks, the correlation between cluster
  residual p90 and fit transverse p90 is +0.75 / +0.64, and it vanishes within θ ≤ 60° (−0.09 / −0.05).
  The θ = 80° error is inherited from imaging, not made by the fit.
  **This stratum was chosen from the baseline alone** (`d101b` and the simulation agent's
  `residual_{pdvd,pdhd}_d101.tsv`), before any knob arm existed.  The rule was frozen as
  `/home/xqian/tmp/d101/pred.txt`, sha256 `0c67b8ae60bff9da…79da`, at 2026-09-14T21:13:44-07:00.
  The first knob arm started at 21:25:23 (`knob_arms_pdvd.log`).  The exclusion covers error
  *inherited* from imaging; a knob that adds error on those tracks is still reported against it
  (sec 4.4).
* **A knob arm PASSES on a detector** when, over those 12 tracks:
  1. the median paired Δ`res_t_p90` < 0 **and** the median paired Δ`dev_p90` < 0;
  2. at least 8 of 12 tracks improve `res_t_p90`;
  3. dQ/dx does not regress: median Δ`dqdx_cv` ≤ +0.005, median Δ|`dqdx_med` − 1| ≤ +0.01, and median
     Δ`dip_truth` ≤ +0.01.
* **Consequence.** The knob set of an arm that passes on **both** PDVD and PDHD is committed default
  OFF, with no production flip.  If several arms pass, all are reported and the simplest passing arm is
  named.  If none passes, **no C++ knob is committed** and the doc reports the ranking.
* **Caveat carried to every sec 4 number.** The arms run a NON-production PR variant, needed because
  the simulation has no light: `-nu-legacy`, `flag_mains_unmatched`, `nu_per_bundle=false`, no STM-only
  bundle gating.  PDVD production runs `-nu` (STM+Michel) with per-bundle gating.  These are
  fit-quality numbers on the same fitter, not production-chain physics numbers.

### 4.1 Sample and chain
* **Truth** (`d101_make_muons.py` → `figs/101_truth_{pdvd,pdhd}.json`).
  * 18 straight 100 cm muons per detector, 5000 e/mm: the 16 grid tracks of sec 3 (θ × φ, k = 0–15)
    plus two simulation repeats, k16 = k0 and k17 = k5 with a new simulation seed.
  * PDVD: bottom CRP anode 1, face 0, wires `v7-uvwfit`.  PDHD: APA 1, face 1, wires `larsoft-v1`
    (away from APA 0's simulation plane-swap defect, doc pdvd/47 sec 10.6).
  * Placement: mid-drift, at least 15 cm inside the face in y/z, at least 20 cm from anode and cathode.
* **Simulation → clustering** (`d101_sim_chain.sh`, pin `libpin` = the pre-change libraries).
  * Simulation: `<det>_sim/wct-sim-xtrack-sp.jsonnet` with noise, fluctuation and production NF+SP.
    Drift constants are read from the **compiled** simulation config: DL/DT 4.13/7.91 (PDVD) and
    4.12/8.20 (PDHD) cm²/s, drift speed 1.568/1.576 mm/µs, lifetime 10 s.
  * Then `run_img_evt.sh -d off` and `run_clus_evt.sh -noq -save-pctree -save-assoc`, with the
    simulation's drift speed and frame time offset.  Without that offset every cluster sat 35–40 cm
    off in x.
  * The PR TLA `time_offset` is −226.598 µs (PDVD) and −251.234 µs (PDHD).
* **PR** (`d101_sim_pr_arm.sh`, pin `libpin_new`):
  * the light-less settings of sec 4.0;
  * a **simulation-matched fit JSON** (`d101_make_sim_tf_json.py` → `figs/101_tf_sim_<det>.json`):
    the production JSON with the simulation's DL/DT, `add_sigma_L`, and doc pdvd/47's simulation
    effective transverse widths.  Without it, the truth residual would measure data-tuned model
    mismatch;
  * each arm writes fresh `work/<RUN6>_<k>_<ARM>/` directories that symlink the same `_d101` pctree.
* **Grading** (`d101_sim_fit_eval.py`): the toy's metrics, computed on `T_rec_charge` of
  `tracking-pr.root` against truth.

### 4.2 Baseline `d101b`, and the owner's symptom 2 against truth
Medians over the 16 grid tracks:

| | res_t med / p90 (mm) | x bias (mm) | chord dev p90 (mm) | snap U / V / W | dQ/dx med / CV | dips < 0.6 truth | ACF(1) | corr(\|residual\|, dQ/dx) |
|---|---|---|---|---|---|---|---|---|
| PDVD | 0.71 / 1.32 | +1.19 | 2.03 | 1.16 / 1.06 / 1.10 | 0.997 / 0.061 | 0 | 0.41 | −0.26 |
| PDHD | 0.94 / 1.77 | +1.03 | 2.89 | 1.03 / 0.98 / 1.02 | 0.930 / 0.081 | 3.3 % | 0.60 | −0.39 |

* **Symptom 2 is reproduced with the true track known.**
  * The correlation between distance from truth and dQ/dx is negative on 14/16 PDVD and 16/16 PDHD
    grid tracks.
  * It is strongest exactly where the trajectory leaves the charge:
    * PDVD k6 (θ 60°, φ 60°): p90 3.7 mm, 17.6 % of points below 0.6 × truth, r = −0.65;
    * PDHD k13 / k14 (θ 20°): p90 6.7 / 4.2 mm, dips 19.6 / 13.8 %, r = −0.63 / −0.86.
  * A trajectory point off the charge collects less of it in `dQ_dx_fit`'s response integral: the low
    dQ/dx is a consequence of the misplacement, not a charge deficit.
* **The x bias grows with θ.**  Medians over 4 tracks at 20 / 40 / 60 / 80°: PDVD +0.26 / +0.88 /
  +1.59 / +2.13 mm, PDHD +0.21 / +0.80 / +1.51 / +1.84 mm.
  Half a 4-tick slice is 1.57 mm, and the sampler places x at the slice start (sec 2.1).  Recorded,
  not pursued: the toy's slice-centre lever A3 made the toy worse.
* **Snap caveat.** The simulated fits snap less than the data (PDVD W 1.10 vs data 1.37–1.55).  The
  simulation therefore under-represents the data's lattice locking, so a lattice lever's effect
  measured here is, if anything, an under-estimate for data.
* **Noise floors — two different ones.**
  * Re-running PR on the same pctree (`d101brep`) is bit-identical, so knob-vs-baseline deltas on
    the same input are exact.
  * A new *simulation* seed moves one event's p90 by up to 0.56 mm: PDVD k0/k16 1.21/1.20,
    k5/k17 0.78/1.34; PDHD k0/k16 0.68/0.77, k5/k17 2.40/2.05.  A single event's delta below about
    0.5 mm therefore does not generalise, which is why the verdict (sec 4.0) is a 12-track median
    plus a count.

### 4.3 Toy vs full chain, per angle (res_t med / p90, chord dev p90; mm; medians over 4 tracks)

| θ | PDVD C++ | PDVD toy | PDHD C++ | PDHD toy |
|---|---|---|---|---|
| 80° | 2.06 / 5.08, 4.25 | 1.13 / 2.41, 3.20 | 1.87 / 4.72, 4.01 | 1.90 / 5.25, 6.60 |
| 60° | 0.47 / 0.81, 1.65 | 0.24 / 0.61, 1.05 | 1.03 / 2.19, 2.70 | 0.23 / 0.73, 1.25 |
| 40° | 0.51 / 1.08, 1.41 | 0.51 / 1.03, 1.04 | 0.48 / 1.30, 1.73 | 0.20 / 0.50, 0.83 |
| 20° | 0.93 / 1.63, 2.28 | 0.57 / 1.26, 1.91 | 0.83 / 2.67, 3.55 | 0.22 / 0.53, 0.83 |

**The toy is NOT anchored on residual magnitude.**
* Against the plan's ±30 % criterion it holds only for PDVD 40° and PDHD 80°.  The full chain is
  1.3–2× worse on the other PDVD angles and 2.6–5× worse on PDHD at 20–60°.
* What still anchors: the snap pattern (sec 3.2), PDVD's dQ/dx spread (sec 3.2), and the sign and
  location of symptom 2.
* One missing piece is identified: at θ = 80° the clustered charge itself sits 2.1 (PDVD) / 3.1 (PDHD)
  cm p90 from truth.  Over all 16 tracks, the fit residual tracks that (r = +0.75 / +0.64), but not
  within θ ≤ 60° (−0.09 / −0.05).  The PDHD excess at 20–60° is therefore not explained by the cluster
  cloud's spread; it is something in the full fit/retile/imaging path the toy does not model (it has no
  retile, no dead or wrapped channels, a single face and ideal SP).
* **Consequence:** the toy's lever ranking is a hypothesis generator, not the arbiter.  Which knobs
  qualify is decided on the full chain, by the pre-registered rule of sec 4.0.

### 4.4 The knob arms, read by the pre-registered rule
`d101_knob_report.py` → `knob_report.txt`.  Paired Δ = arm − `d101b` on the same pctree (exact).
The fit JSON keys of `kf` / `kc` were accepted: no "Failed to set parameter" in any arm log.

**Verdict stratum, θ ≤ 60° (12 tracks).**  Baselines: PDVD res_t p90 1.23, chord p90 1.73 mm;
PDHD 1.51 / 2.21 mm.

| arm | PDVD Δres_t p90 (improved) / Δchord p90 / Δres_t med / ΔCV / Δ\|med−1\| / Δdips | PDVD | PDHD Δ… same columns | PDHD |
|---|---|---|---|---|
| `kf` fit knobs | −0.025 (9/12) / −0.385 / −0.020 / −0.008 / +0.005 / 0 | **PASS** | −0.257 (10/12) / −0.335 / −0.114 / −0.006 / −0.006 / 0 | **PASS** |
| `ks` retile sampler | +0.006 (5/12) / −0.172 / −0.002 / 0.000 / +0.001 / 0 | fail (1, 2) | +0.007 (5/12) / +0.016 / +0.007 / 0.000 / −0.003 / 0 | fail (1, 2) |
| `kc` both | −0.169 (11/12) / −0.423 / −0.053 / −0.015 / +0.004 / 0 | **PASS** | −0.145 (8/12) / −0.208 / −0.088 / −0.003 / −0.005 / +0.003 | **PASS** |

**Reported, not in the verdict.**

| arm | θ = 80°, PDVD Δres_t p90 / Δchord p90 | θ = 80°, PDHD | sim repeats k16/k17, PDVD / PDHD Δres_t p90 |
|---|---|---|---|
| `kf` | −0.16 (3/4) / −0.28 | −0.31 (4/4) / −0.46 | −0.28 / −0.27 (2/2 each) |
| `ks` | −2.49 (4/4) / −1.59 | **+2.19** (2/4) / +0.04 | −0.26 / −0.17 |
| `kc` | −2.69 (4/4) / −1.92 | **+0.91** (2/4) / −0.62 | −0.43 / −0.49 |

**Per event** (the medians hide the structure):
* **The fit knobs fix the worst symptom-2 tracks and break none.**
  * PDHD k14 (θ 20°): transverse p90 4.23 → 1.34 mm, chord p90 5.82 → 1.66 mm, dips below
    0.6 × truth 14 % → 2 %, corr(\|residual\|, dQ/dx) −0.86 → −0.14.
  * PDHD k13: p90 6.66 → 4.60 mm.  PDHD k2 (θ 80°): 11.38 → 9.79 mm.
  * The largest increase `kf` causes on any θ = 80° track is +0.02 mm p90.
* **The retile sampler is double-edged exactly where it acts, on near-isochronous tracks.**
  * PDVD k1 / k3: p90 −3.85 / −3.54 mm; the finer Steiner cloud follows the track across the 7.65 mm
    U/V lattice.
  * PDHD k3 (θ 80°, φ 90°): p90 6.54 → **26.90 mm**.  The finer cloud hands the PR a spurious 5-point
    second segment (sub-cluster 1001) and pulls the main segment off (res_t median 2.57 → 6.13 mm).
    PDHD k2: +4.40 mm.
  * On PDVD k6 (θ 60°), the worst θ ≤ 60° track, it raises the dip fraction by 13 (`ks`) / 16 (`kc`)
    points.
  * On θ ≤ 60°, where the Steiner seed already sits on the charge, it does nothing measurable.

**Outcome of the rule.**
* `kf` and `kc` qualify; `ks` does not.  By the rule, the knob set of every qualifying arm is
  committed default OFF, so all three knobs ship, and none is flipped.
* **The set named by the simulation is `kf`** (`fit_weight_pow` 1.5 + `assoc_cont_center` 1).
  **Superseded on data: it regresses the hand-scanned STM/Michel tags (sec 6.5), so it is not
  recommended.**  The simulation verdict stands as a trajectory result.  Why `kf` over `kc`:
  * it is the simpler arm: two fit-JSON keys, no jsonnet, no sampler;
  * it beats `kc` on PDHD's verdict stratum (−0.257 vs −0.145 mm);
  * it improves θ = 80° on both detectors.
* `kc`'s extra PDVD gain (−0.169 vs −0.025 mm) arrives with `ks`'s PDHD θ = 80° regression and PDVD
  k6's dips.  The exclusion of θ = 80° was for error inherited from imaging, not for error a knob adds,
  so that regression counts against `kc`.
* The sampler knobs stay in the tree as a study instrument.

## 5. Answers to the owner's questions

**Q1 — Is the Steiner terminal construction the reason the trajectory zig-zags on PDVD/PDHD?**
**No.**
* Terminals are about one per slice on all three detectors (doc pdvd/31, sec 2.2).  Replacing the whole
  Steiner seed by the TRUE track moves the toy's fitted trajectory by at most 6 % (sec 3.3).
* The zig-zag is made in the fit.
  * `trajectory_fit` places a point at a (q/err)²-weighted centroid of integer wire indices, and with a
    flat charge error that is q² (sec 2.4).
  * The association window is centred on the rounded wire.
  * On a pitch larger than the charge's transverse spread, the brightest wire wins, and fitted points
    pile onto wire centres in proportion to the pitch: snap index PDVD W 1.55, PDHD W 1.23, SBND
    ≈ 1.0 (sec 1).
* Wire angle enters through the same mechanism (PDHD's ±35.7° U/V changes how fast a track crosses
  the lattice); it is not a separate defect.

**Q2 — Should sampling be at half a wire pitch?**  **No, as measured.**
* It can only act in the PR job's retile samplers.  Changing the clustering job's sampler would change
  every pctree and every downstream product.
* In the toy with that scope it helps a little: PDVD −0.04, PDHD −0.12 mm p90 (sec 3.6).
* In the full chain it is null on θ ≤ 60° (+0.006 / +0.007 mm, 5/12 tracks each) and double-edged on
  near-isochronous tracks: PDVD −2.5 mm, PDHD +2.2 mm with a spurious extra segment (sec 4.4).
* It fails the pre-registered rule on both detectors.  It is built, default OFF
  (`retile_sampler_half_pitch`, `retile_sampler_min_step`), and **not recommended**.

**Q3 — Is there a better terminal definition, closer to the true charge?**
* Moving terminals to the three-view charge centroid (toy lever B1) changes transverse p90 by −0.03 /
  −0.02 / −0.01 mm, as the oracle predicts; not built.
* What brings the trajectory onto the true charge is in the fit: a softer position weight
  (`fit_weight_pow` 1.5, weight |q/err·…|^1.5) plus the window centred on the continuous wire
  coordinate (`assoc_cont_center` 1).
* Full chain, θ ≤ 60°: chord p90 −0.39 mm on PDVD (9/12 tracks) and −0.34 on PDHD (10/12);
  transverse p90 −0.03 / −0.26 mm.
* **But that fit lever is not free.**  On data, the same two keys cost the STM/Michel tags on both
  detectors' hand-scan records (sec 6.5).  The lever is real, but no safe operating point exists
  without retuning the tagger that was tuned on the q² fit.  **Not recommended as-is.**

**Q4 — the two dQ/dx symptoms.**
* **Symptom 2** (dQ/dx very low where the trajectory leaves the track) is reproduced with truth: the
  correlation is negative on 30 of 32 simulated tracks (sec 4.2).  It is a consequence of the
  misplacement, not a charge deficit, and the fit knobs remove its worst case (PDHD k14: r −0.86 →
  −0.14, dips 14 % → 2 %).
* **Symptom 1** (high/low oscillation) moves only a little under any lever (dQ/dx CV −0.006 to −0.015).
  The data's CV (PDHD 0.25–0.32, PDVD 0.21) is 3–4× the simulation's (0.06–0.08) on the same fitter,
  so most of the data's oscillation is not this lattice mechanism on a clean track.  That is left open
  (sec 7).

Data effect on the owner's Bee event (PDHD 028084_21 cl55) and on the 120 + 61 data events: sec 6.3.

## 6. Knob prototype and gates

**Status: all knobs default OFF, byte-identical when off (gates below).  No production file sets
them.  NOT flipped.**

### 6.1 The knobs
| knob | where | default (= legacy) | on |
|---|---|---|---|
| `fit_weight_pow` | `TrackFitting::Parameters` (`TrackFitting.h`), fit JSON key; applied at the six position-LSQ row sites of `fit_point` and `trajectory_fit` | 2 (the line is skipped) | row scale s → \|s\|^(pow/2), weight \|s\|^pow; simulation-named 1.5 |
| `assoc_cont_center` | same; `form_point_association`, both branches | 0 | the per-plane wire window is centred on `Grouping::convert_3Dpoint_wire_cont` (new; `Facade::point2wind_cont`) instead of the rounded wire; simulation-named 1 |
| `half_pitch` | `BlobSampler` `stepped` strategy | false | adds the half-pitch crossings of the min/max views, with the legacy range and mid-view tests |
| `retile_sampler_half_pitch`, `retile_sampler_min_step` | TLAs of `pdvd/pdhd wct-pr-perevt.jsonnet` → `pr.jsonnet` → `clus.jsonnet live_sampler` | false / null (keys omitted) | only the PR job's retile samplers (the Steiner cloud) |
| `flag_unmatched` (sim plumbing) | `ClusteringFlagMatchedMains`; TLA `flag_mains_unmatched` | false | a cluster without a Q/L match can be a main — light-less SIMULATION only |

**Simulation-named set, NOT recommended (sec 6.5):** `"fit_weight_pow": 1.5, "assoc_cont_center": 1`
added to the fit JSON, as in `figs/101_tf_prod_{pdvd,pdhd}_kf.json`.  Nothing else changes.  On data
it regresses the STM/Michel tags.

### 6.2 Proofs and gates
* **Doctests.**
  * `wcdoctest-clus` passes 412/412.
  * New `doctest_trackfitting_lattice_knobs.cxx` (3 cases, 34 assertions) pins:
    * both defaults and the set/get round trip;
    * `std::round(point2wind_cont) == point2wind` on 50 000 random points over the three detectors'
      pitches and angles.
  * `doctest_flag_matched_mains_defaults` checks the `flag_unmatched` default.
* **Freshness.**
  * `local/lib/libWireCellClus.so` 21:19:48 (md5 `091e142b…`) is newer than the last source edit
    (21:14:05).
  * The pins used: `libpin` `64c91641…` (pre-round), `libpin_new` `7264bc9d…` (flag_unmatched
    only), `libpin_k` `091e142b…` (this round), unchanged through every arm.
* **Compiled configs, knobs off, identical.**
  * `wct-pr-perevt` PDVD md5 `1cc5ad6f…`, PDHD `1d2dfcb3…`, identical to before the round.
  * `wct-clustering` PDVD/PDHD compiled against a HEAD export of `cfg/` vs the edited tree:
    identical.
  * The HEAD path is proven live by a control: the edited `wct-pr-perevt` fails against it on
    `flag_mains_unmatched`.
  * **Knobs on:** exactly `"half_pitch": true, "min_step_size": 1` on the 16 (PDVD) / 8 (PDHD)
    retile samplers.  SBND's `clus.jsonnet` only mentions the ProtoDUNE files in a comment.
* **Runtime byte gates**, `d101_gates.sh` → `/home/xqian/tmp/d101/gates*.log`.  Compared per event:
  every TTree, zip member, tarball member (`hash_archive.py`), JSON and TSV.

| # | pair (new pin vs old pin, same config) | events | verdict |
|---|---|---|---|
| 1 | sim PR `d101koff` vs `d101b`, PDVD / PDHD (flag_unmatched ON path) | 18 / 18 (72 / 54 files) | **BYTE-IDENTICAL** |
| 2 | PDVD data PR `d101vnew` vs `d101vold` (production `-nu`, d16vnu) | 120 (479 files) | **BYTE-IDENTICAL** |
| 3 | PDHD data PR `d101hnew` vs `d101hold` (production `-nu`, d16hnu) | 61 (244 files) | **BYTE-IDENTICAL** |
| 4 | SBND PR `d101snew` vs `d101sold` (pr146 manifest on d102m, geometric vertex) | 16 (80 files) | **BYTE-IDENTICAL** |
| 5 | clustering, PDHD 027409_0 `d101cnew2` vs `d101cold2`; PDVD sim 900101_0 `d101cnew3` vs `d101cold3` | 1 + 1 (16 + 6 files) | **BYTE-IDENTICAL** |
| 6 | control: sim `d101kf` vs `d101koff` (PDVD), `d101ks` vs `d101koff` (PDHD) — must differ | 18 + 18 | NOT IDENTICAL (as required) |

### 6.3 Knob-ON smoke on the owner's Bee event, PDHD 028084_21 cluster 55
`d101_smoke_cl55.py`.  Tags `d101smoff` / `d101smkf` read the `h28prod` pctree on `libpin_k`; the only
difference is the two fit keys.

| fit | knobs | points / length | chord dev med / p90 (cm) | dQ/dx CV | dips < 0.5× local median | ACF(1) | corr(dev, dQ/dx) | snap U / V / W |
|---|---|---|---|---|---|---|---|---|
| STM pass 0 (the Bee trajectory) | off | 257 / 168.1 cm | 0.206 / 0.734 | 0.434 | 3.9 % | 0.83 | −0.05 | 1.05 / 1.03 / 0.86 |
| | **on** | 234 / 156.9 cm | **0.175 / 0.566** | **0.291** | **1.3 %** | 0.71 | −0.16 | 1.08 / 1.10 / 1.00 |
| PR, 40 cm segment | off / on | 58 / 56 | 0.152 / 0.425 → 0.165 / 0.440 | 0.418 → 0.308 | 0 → 0 | 0.69 → 0.58 | | |
| PR, 78 cm segment | off / on | 122 / 118 | 0.139 / 0.469 → 0.154 / 0.380 | 0.217 → 0.254 | 0.8 % → 0 | 0.55 → 0.58 | | |

The Bee trajectory's chord p90 drops 23 %, its dQ/dx CV 33 % and its dips 3 → 1 %.
* Caveat: the knob-on STM fit is 11 cm shorter (257 → 234 points), so the two rows do not cover
  exactly the same span.
* Caveat: today's knob-off numbers are not sec 1's `h28prod` numbers (CV 0.84, 267 points).  PR
  production has moved since `h28prod` ran; the smoke compares like with like.
* The PR segments move less and in both directions.
* This is one cluster.  The population statement is sec 4.4 on simulation and sec 6.4 on data.

### 6.4 Knob ON on the data sets (production `-nu`, same pctrees, same pin)
`d101vkf` / `d101hkf` (production fit JSON plus the two keys) vs `d101vnew` / `d101hnew`, graded with
the sec 1 ruler (`d101_census.py --arm …` → `census_kf2/`).  The knob-off rows reproduce sec 1's
production census exactly (e.g. PDVD STM 681 runs, 149 759 points), so today's production equals
`p96vprod` / `h28prod` for these fits.

| fit, detector | knobs | runs | chord dev med / p90 (cm) | dQ/dx CV | dips | ACF(1) | snap U / V / W |
|---|---|---|---|---|---|---|---|
| STM, PDVD | off | 681 | 0.155 / 0.478 | 0.212 | 3.58 % | 0.59 | 1.39 / 1.19 / 1.55 |
| | **on** | 678 | **0.127 / 0.445** | 0.209 | 3.43 % | 0.58 | **1.25 / 1.14 / 1.44** |
| PR, PDVD | off | 1090 | 0.151 / 0.375 | 0.206 | 3.01 % | 0.54 | 1.23 / 1.12 / 1.37 |
| | **on** | 1064 | **0.127 / 0.336** | 0.198 | 2.75 % | 0.53 | 1.16 / 1.08 / 1.33 |
| STM, PDHD | off | 343 | 0.196 / 0.564 | 0.247 | 4.22 % | 0.80 | 1.09 / 1.07 / 1.23 |
| | **on** | 331 | **0.182 / 0.542** | 0.242 | 4.13 % | 0.79 | 1.07 / 1.05 / **1.13** |
| PR, PDHD | off | 649 | 0.236 / 0.563 | 0.323 | 4.30 % | 0.76 | 1.03 / 1.03 / 1.16 |
| | **on** | 640 | **0.228 / 0.544** | 0.312 | 4.40 % | 0.75 | 1.02 / 1.03 / 1.11 |

* **Trajectory quality improves on data, in the same direction as in simulation.**
  * The zig-zag median drops 18 % on PDVD (STM and PR) and 4–7 % on PDHD.
  * Wire-centre snapping drops on every plane that had it (PDVD U −0.14, W −0.11; PDHD W −0.10).
  * dQ/dx CV drops by 0.003–0.011; dips move by ±0.3 points.
  * Symptom 1 on data is therefore barely touched (sec 7).
* **The tag set moves.**  CheckSTM_Michel shares the fit JSON, so the knob changes the STM/Michel
  output on most events:

  | detector | events changed | candidates | pass every check | with a Michel |
  |---|---|---|---|---|
  | PDVD (120 events) | 102 | 596 → 589 | 265 → 266 | 170 → 176 |
  | PDHD (61 events) | 53 | 341 → 320 | 129 → **120** | 133 → **118** |

  * Three PDVD events lose their only STM candidate: 039252_5, 039349_14 (which passed every
    check) and 039349_78 (which had a Michel).  One event gains two: 039252_11.
  * PDHD loses 9 passes and 15 Michels.
  * Whether these changes are right or wrong is a question for the hand-scan record (sec 6.5), not
    for the counts.

### 6.5 The tag change graded against the owner's hand-scan records — **a regression**
**PDHD.**  Record smx22, the 303-item population of the smx18 key, truth precedence owner_review >
owner_smx1 > agent.
* `pdhd/docs/scan/h21/d21_grade.py` skips a hand-scanned item whose cluster is not a candidate in the
  arm.  An arm that loses candidates therefore loses denominator.
* `d101_stm_grade_pdhd.py`, a fork of that grader, keeps those items: an absent hand stopper is a miss.

| arm | scored | TP | FP | FN | TN | purity | efficiency | hand stoppers no longer candidates |
|---|---|---|---|---|---|---|---|---|
| `p82bhoff` (instrument check, must be 61/0/86/110) | 257 | 61 | 0 | 86 | 110 | 1.000 | 0.415 | — |
| `d101hnew` (knobs off, today's production) | 257 | 114 | 3 | 33 | 107 | 0.974 | 0.776 | 0 |
| `d101hkf` (fit knobs on) | 257 | 89 | 5 | 58 | 105 | **0.947** | **0.605** | **22** (+ 26 non-stoppers) |

* The unforked grader gives 0.947 / 0.712 for `d101hkf`: it scores 209 items and hides the 22 lost
  stoppers.
* Michel side, graded on hand stoppers with `d21_michel_census.py`'s truth: purity 0.971 → 0.919,
  efficiency 0.791 → **0.663** (57/86).  11 hand Michels are no longer candidates.  Unforked, that
  census scores only 75 and gives 0.760.

**PDVD.**  `census_score.py` with the smx1a–smx9 record.  The knob-off arm reproduces doc pdvd/96's
production numbers exactly (is_stm 242 / 8 / 34, efficiency 0.877; Michel 143 / 12 / 21), so the
instrument is valid.
* `census_score.py` scores only records with a payload in the arm, the same denominator trap: 546
  knobs off, 460 knobs on.
* `d101_stm_grade_pdvd.py` fixes the population at the 546 judged items of the knob-off arm.  A
  record absent from the knob-on arm is scored chain-negative.

| arm | is_stm TP / FP / FN / TN | purity | efficiency | Michel TP / FP / FN / TN | purity | efficiency | hand positives no longer candidates (STM / Michel) |
|---|---|---|---|---|---|---|---|
| `d101vnew` (knobs off) | 242 / 8 / 34 / 262 | 0.968 | 0.877 | 143 / 12 / 21 / 370 | 0.923 | 0.872 | 0 / 0 |
| `d101vkf` (fit knobs on) | 204 / 16 / 72 / 254 | **0.927** | **0.739** | 117 / 24 / 47 / 358 | **0.830** | **0.713** | **35 / 21** (+ 62 non-stoppers) |

* The unforked scorer on `d101vkf` gives is_stm 0.920 / 0.845 and Michel 0.831 / 0.814 over 460
  items.
* On its unenriched tranche 2 alone, Michel purity is 0.921 → 0.798.
* 11 knob-on candidates carry a record key with no knob-off payload; they are unscored.
* Other `census_score.py` indicators move the same way:
  * scan-tagged Michel segments given role 3: 83 % → 67 %;
  * **38 (16 %) "lost (no fitted segment there)"**, a class absent knobs off;
  * the stop residual against the scan pins: median 2.23 → 3.11 cm (25 / 21 pins).

**What moves: the candidate set, not the fitted length.**  Join by (event, cluster id), which is stable
downstream of clustering; the pctrees are the same (`idset_compare.txt`).

| detector | events with an identical candidate-id set | candidate ids lost (is_stm = 1 knobs off) | ids gained (is_stm = 1 knobs on) | kept ids flipping is_stm 0→1 / 1→0 |
|---|---|---|---|---|
| PDHD (61 events) | 10 | 70 (22) | 49 (14) | 15 / 16 |
| PDVD (116 events readable in both arms) | 19 | 102 (30) | 96 (32) | 29 / 29 |

* No lost cluster id is fitted anywhere in the knob-on `tracking-pr.root`.  No gained id is fitted
  anywhere in the knob-off file.
* So the knob changes which clusters the taggers carry to a fit, and it flips verdicts on the clusters
  they keep.
* It does not shorten fits uniformly.  Per event, the total fitted length knob-on / knob-off (census
  runs, `census_kf2/`) has a median of 0.99 on both detectors, but p10 0.57 (PDHD) / 0.73 (PDVD) and
  p90 1.35 / 1.40.  In total PDHD loses 5 % and PDVD gains 1–2 %.
* Which tagger decision drops a cluster is not chased (sec 7).

## 7. What is not concluded
* **No production flip, and no recommendation to flip.**
  * `fit_weight_pow` 1.5 + `assoc_cont_center` 1 improves trajectories on 24 simulated tracks, on
    the owner's cluster and in the data census.  It regresses the hand-scanned STM/Michel tags on both
    detectors (sec 6.5), and the hand-scan records outweigh the 24 tracks.
  * CheckSTM_Michel and TaggerCheckSTM (its cuts, end-point trims and profile tests) were tuned on the
    q² fit.  A flip would need that tagger retuned against the softer fit, and then re-graded on the
    same records.  Not attempted.
  * **No scope avoids the tagger in production.**  In production `-nu` the fit JSON's only consumers
    are TaggerCheckSTM and CheckSTM_Michel.  The compiled PR pipeline is SwitchScope, FlagMatchedMains,
    CreateSteinerGraph, FiducialUtils, TGM, STM, FC, ProtectBundle, CreateSteinerGraph refresh,
    CheckSTM_Michel, Magnify, DisplayDump.  The PR fits in `tracking-pr.root` are those taggers' fits.
  * A split-config arm tested this directly.  `d101vkfpr` set `trackfitting_config` = the knob JSON and
    `stm_trackfitting_config` = the production JSON.  Its compiled config names the knob JSON nowhere;
    both consumers read `pdvd_track_fitting.json`.  It was stopped after 8 events.  All 8 equal
    `d101vnew` in every tree of `tracking-pr.root` (`vkfpr_vs_vnew.txt`), so the arm was inert.  The
    other 112 `work/*_d101vkfpr` dirs are staging only and not an arm.
  * **The mechanism of the loss is not chased.**  The lost candidates are not fitted anywhere in the
    knob-on file, and the fitted length is not uniformly shorter (sec 6.5).
* **The toy is not anchored on residual magnitude** (sec 4.3).  Its ranking picked the right fit
  levers but over-predicted the sampling lever.
* **The PDHD full-chain excess at θ 20–60°** (2.6–5× the toy) is unexplained, as is the +1–2.7 mm
  drift bias growing with θ (sec 4.2).
* **The data's dQ/dx oscillation (symptom 1)** is 3–4× the simulation's on the same fitter; its cause
  is not this lattice mechanism on a clean track.  Leads: charge completeness (doc pdvd/50) and
  real-data SP.
* **The sim arms are a non-production PR variant** (`-nu-legacy`, `flag_mains_unmatched`, no
  per-bundle gating; sec 4.0).  The data smoke and the data census run production `-nu`.
* **SBND knob-ON is not measured**, only gated off.  The toy says `fit_weight_pow` alone costs SBND a
  little (sec 3.6).  No SBND config sets the keys.
* **The retile sampler's PDHD θ = 80° failure mode** (a spurious short segment) was seen on 2 tracks
  and not chased; the knob is not recommended.

## 8. Files
* **Scripts** (`scripts/`):
  * `d101_census.py` (sec 1; `--arm` for other arms);
  * `d101_toy.py`, `d101_toy_scan.py`, `d101_toy_report.py` (sec 3);
  * `d101_make_muons.py`, `d101_sim_chain.sh`, `d101_residual.py` (simulation, sec 4.1);
  * `d101_make_sim_tf_json.py`, `d101_sim_pr_arm.sh`, `d101_sim_fit_eval.py` (sec 4.1–4.2);
  * `d101_knob_arms.sh`, `d101_knob_report.py` (sec 4.4);
  * `d101_gates.sh`, `d101_sbnd_arm.sh` (sec 6.2);
  * `d101_smoke_cl55.py` (sec 6.3);
  * `d101_stm_grade_pdhd.py`, `d101_stm_grade_pdvd.py` (sec 6.5): fixed-denominator forks of
    `pdhd/docs/scan/h21/d21_grade.py` + `d21_michel_census.py` and of `census_score.py` sec A
    (all untouched).
* **Figures and inputs** (`figs/`):
  * `101_truth_{pdvd,pdhd}.json`;
  * `101_tf_sim_{pdvd,pdhd}.json` (simulation-matched fit JSON);
  * `101_tf_sim_{det}_kf.json`, `101_tf_prod_{det}_kf.json` (the same plus the two knob keys).
* **Work tags** (never overwritten; M13):
  * simulation: `900101_*` / `900102_*` with `_d101` (the chain), `_d101b`, `_d101brep`, `_d101koff`,
    `_d101kf`, `_d101ks`, `_d101kc`;
  * pilots `_d101p0` … `_d101p3` (the record of the no-light failures, sec 4.1);
  * data gates `_d101vold` / `_d101vnew` / `_d101hold` / `_d101hnew`;
  * data knob-on `_d101vkf` / `_d101hkf`;
  * smoke `028084_21_d101smoff` / `_d101smkf`;
  * SBND `work-{mcp1k,mcp2k}-d101sold` / `-d101snew`;
  * clustering gates `027409_0_d101cold2` / `_d101cnew2`, `900101_0_d101cold3` / `_d101cnew3`.
* **Staging-only tags that never ran wire-cell** (symlinks, and one stray compiled config): PDHD
  `027409_0_d101cold` / `_d101cnew` (a log-name typo), PDVD `900101_0_d101cold` / `_d101cnew` and
  `_d101cold2` / `_d101cnew2` (the replay lacked `img-provenance.txt`, which `run_clus_evt.sh`'s wires
  guard reads).  They are not arms.
* **Stopped partial tag:** PDVD `*_d101vkfpr` (sec 7).  8 events ran, identical to `d101vnew`; 112
  dirs are staging only.  It is not an arm.
* **Logs and tables:** `/home/xqian/tmp/d101/` (`pred.txt`, `knob_report.txt`, `gates*.log`,
  `eval_*.tsv`, `smoke_cl55.txt`, `census_kf2/`, `grade_pdhd*.log`, `michel_pdhd.log`,
  `grade_pdvd_fixed.log`, `score_d101v{new,kf}.{log,json}`, `prep_*`/`sheet_*`, `idset_compare.txt`).

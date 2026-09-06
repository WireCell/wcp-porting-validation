# 02 — The PDHD track fit's effective transverse smearing: re-derived on the production arm, graded, and shipped

**Status.** `cfg/pgrapher/experiment/pdhd/pdhd_track_fitting.json` carries the data-derived
constants since 2026-09-05 (owner decision): `DT = 6.129e-7` (6.13 ± 2.11 cm²/s),
`ind_sigma_u_T = 3.210`, `ind_sigma_v_T = 2.717`, `col_sigma_w_T = 1.460` mm, replacing the
closed-form seeds `8.2017e-7 / 1.05375 / 1.75625 / 0.05407`. **NOT bit-identical, by
design**: the file is read at runtime, so the compiled jsonnet is unchanged and a config
hash cannot see this flip. Gate record: `pdhd/stm/gates/d02_eff_sigma_gate.txt`.

This executes item 2 of `stm-tagger-chain.md` §10, the PDHD analogue of doc pdvd/44 §7.1.
The *origin* of the constant — why the SP output is this wide on every detector — is the
subject of doc pdvd/47 (`pdvd/docs/nf_sp_img_clus/47_transverse-smearing-in-simulation.md`),
which reproduces the induction constant in simulation; read that before touching these
numbers again. Its §8 (second round) withdraws the sub-pitch-phase mechanism of its §3.4 and
adds a phase-window mode to the estimator — the legacy path is byte-identical on this arm too
(5 `stmwc` events, `_bins/_fit/_shape.tsv` unchanged), so the constants below are unaffected.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd; S=docs/scripts; F=docs/figs; A=/home/xqian/tmp/pdhd02
# pin (a peer's wcbuild swaps local/lib mid-campaign): libWireCellClus.so e3304cb9b362cd680ee3182452751a96, toolkit 76f47614
mkdir -p /home/xqian/tmp/pdhd02_libpin && cp -a /home/xqian/toolkit-dev/local/lib/. /home/xqian/tmp/pdhd02_libpin/
export LD_LIBRARY_PATH=/home/xqian/tmp/pdhd02_libpin
# --- A. derive on the production arm stmwc (both wrapped-plane knobs on, doc pdhd/01), primary cut 0.436 + controls
python3 $S/d44_sigma_fit.py --det pdhd --max-advance 0.436 --nboot 200 --split run,length,rr,foff,advance --out $F/d02_sigma work/029107_*_stmwc/tracking-stm.root
for adv in 0.25 0.30 0.60; do python3 $S/d44_sigma_fit.py --det pdhd --max-advance $adv --nboot 120 --out $A/ctl/adv$adv work/029107_*_stmwc/tracking-stm.root; done
for hw in 2 4; do python3 $S/d44_sigma_fit.py --det pdhd --max-advance 0.436 --halfwidth $hw --nboot 120 --out $A/ctl/hw$hw work/029107_*_stmwc/tracking-stm.root; done
python3 $S/pdhd_sigma_plots.py --bins $F/d02_sigma_bins.tsv --fit $F/d02_sigma_fit.tsv --out $F/d02_sigma
# --- B. JSON copies (canonical untouched until step E)
C=/home/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/pdhd/pdhd_track_fitting.json
python3 $S/d02_make_tf_json.py --fit $F/d02_sigma_fit.tsv --est share --src $C --out stm/pdhd_track_fitting_d02.json      # share-matched joint line
python3 $S/d02_make_tf_json.py --fit $F/d02_sigma_fit.tsv --est rms   --src $C --out stm/pdhd_track_fitting_d02b.json     # rms-matched
python3 $S/d02_make_tf_json.py --bins $F/d02_sigma_bins.tsv --est share --fix-dt 8.2017 --src $C --out stm/pdhd_track_fitting_d02fix.json   # DT physical, c only
# --- C. arms (30 events of 029107, pctree symlinked from the stm0 dirs; completion = 30 pr_resource markers, never the log)
PDHD_MAX_JOBS=8 PDHD_KEEP_CFG=1 ./run_pr_evt.sh -s d02ref -stm -stm-fit 029107 all                      # canonical JSON, no TLA, this pin
python3 $S/d02_hash_gate.py stmwc d02ref                                                                    # GATE 1: PASS 30/30
for t in d02sig:d02 d02sigb:d02b d02fix:d02fix; do ARM=${t%%:*} PIN=/home/xqian/tmp/pdhd02_libpin TFJSON=stm/pdhd_track_fitting_${t#*:}.json JOBS=8 $S/run_d02_arms.sh; done
# --- D. grading (the doc-42/44 scripts unchanged)
for t in d02ref d02sig d02sigb d02fix; do
  python3 $S/d42_proj2d_resid.py --det pdhd --out $A/ana/resid_$t work/029107_*_$t/tracking-stm.root
  python3 $S/d42_shape_diag.py   --det pdhd --out $A/ana/diag_$t  work/029107_*_$t/tracking-stm.root
  python3 $S/d42_dqdx_rr.py --det pdhd --ref stm/pdhd_ref_dqdx.json --ref-key MuonDeDx --out $A/ana/dqdx_$t work/029107_*_$t/tracking-stm.root
  python3 $S/d42_ring_frame.py --det pdhd work/029107_*_$t/tracking-stm.root > $A/ana/rings_$t.txt; done
python3 $S/d44_compare.py --ref d02ref=$A/ana/resid_d02ref,$A/ana/diag_d02ref,$A/ana/dqdx_d02ref --arm d02sig=... --arm d02sigb=... --arm d02fix=... --out $F/d02
python3 $S/d44_foff_shells.py d02ref=$A/ana/resid_d02ref_blocks.tsv d02sig=$A/ana/resid_d02sig_blocks.tsv d02fix=$A/ana/resid_d02fix_blocks.tsv
python3 $S/d44_sigma_fit.py --det pdhd --max-advance 0.436 --model-json stm/pdhd_track_fitting_d02.json --out $A/d02_sigma_after work/029107_*_d02sig/tracking-stm.root
# --- E. flip the canonical file (values from d02_sigma_fit.tsv, label all, est share, joint rows), then GATE 2
ARM=d02prod PIN=/home/xqian/tmp/pdhd02_libpin EVENTS="0 6" $S/run_d02_arms.sh                             # flipped file, no TLA
for e in 0 6; do python3 ../abtest/hash_archive.py work/029107_${e}_{d02sig,d02prod}/mabc-pr.zip; python3 ../qlport/scripts/hash_root_trees.py work/029107_${e}_{d02sig,d02prod}/tracking-stm.root; python3 $S/d42_proj2d_selfcheck.py --det pdhd work/029107_${e}_d02prod/tracking-stm.root; done
```

Committed products: `figs/d02_sigma_{bins,fit,shape}.tsv`, `figs/d02_sigma_{fit,shape}.png`,
`figs/d02_compare.tsv`, `figs/d02_compare_{2d,dqdx}.png`, `figs/d02_verdicts.tsv`,
`stm/pdhd_track_fitting_d02{,b,fix}.json`, `stm/gates/d02_eff_sigma_gate.txt`, the scripts
`d02_make_tf_json.py`, `d02_hash_gate.py`, `run_d02_arms.sh`, and the PDHD copies
`d44_compare.py`, `d42_proj2d_selfcheck.py`. The `d42_*` analysis products live under
`/home/xqian/tmp/pdhd02/ana/` (regenerated, not committed).

## 1. The wire-filter statement, corrected

The old `_comment_transverse_smearing` quoted PDHD's induction wire filter as "true Gaussian
sigma 0.376 pitch = 1.756 mm". That number is the closed form `1/(√π · 0.75) = 0.752` pitch
**times the V-plane factor 0.5** — i.e. the V seed restated, not the kernel. Inverse-DFT of the
exact array `HfFilter::filter_waveform` builds (`d42_wire_filter_toy.py kernel_of`, N = 3200,
rms over |x| ≤ 12 wires):

| filter | X | closed form [pitch] | **true rms [pitch]** | [mm] | h0 / h±1 / h±2 | pass @ Nyquist |
|---|---|---|---|---|---|---|
| PDHD `Wire_ind` | 0.75 | 0.752 | **0.729** | **3.40** | 0.521 / 0.227 / 0.010 | 0.061 |
| PDHD `Wire_col` | 10.0 | 0.056 | 0.007 | 0.034 | 0.995 / 0.003 / −0.001 | 0.984 |
| PDVD `Wire_ind` | 5.0 | 0.113 | 0.028 | 0.213 | 0.979 / 0.012 / −0.003 | 0.939 |
| PDVD `Wire_col` | 10.0 | 0.056 | 0.007 | 0.036 | 0.995 / 0.003 / −0.001 | 0.984 |
| SBND `Wire_ind` | 1.05 | 0.537 | 0.468 | 1.405 | 0.675 / 0.174 / −0.016 | 0.241 |
| SBND `Wire_col` | 3.60 | 0.157 | 0.053 | 0.159 | 0.961 / 0.023 / −0.006 | 0.886 |

PDHD's induction filter is the widest in the tree (the ordering claim was right); its
kernel is 3.40 mm, not 1.76. Keep this table beside any future reading of `sp-filters.jsonnet`
(feedback: a frequency-domain sigma is not a width).

## 2. Derivation on the production arm

Arm `stmwc` = run 029107, events 0–29, both wrapped-plane knobs on (the PDHD PR production
config since doc pdhd/01), 178 accepted STM passes, 57 376 profiles, 18 267 prolonged
(|Δwire/Δslice| < 0.436, the geometric equivalent of PDVD's 0.25). Estimator = doc pdvd/44
§2.1 unchanged (`pdhd/docs/scripts/d44_sigma_fit.py`, positive control in
`stm-tagger-chain.md` §8.1).

### 2.1 Joint fits (`figs/d02_sigma_fit.tsv`, label `all`)

| estimator | D_T,eff [cm²/s] | c_U [mm] | c_V [mm] | c_W [mm] | χ²/ndf |
|---|---|---|---|---|---|
| rms-matched | 1.29 ± 2.99 | 3.697 ± 0.112 | 3.400 ± 0.132 | 3.070 ± 0.206 | 15.5/14 |
| **share-matched (shipped)** | **6.13 ± 2.11** | **3.210 ± 0.088** | **2.717 ± 0.128** | **1.460 ± 0.171** | 11.3/14 |
| per-plane share (free D) | U 2.4 ± 3.2 / V 6.8 ± 4.0 / W 10.9 ± 3.9 | 3.317 | 2.687 | 1.171 | 0.1 / 0.2 / 8.1 (4) |
| seeds (pre-flip) | 8.2017 | 1.054 | 1.756 | 0.054 | — |
| `stm0` (§8.3 of the chain doc, retiler off) | 3.68 ± 3.03 | 3.23 | 2.55 | 2.82 | 8.6/14 |
| `stmw` (retiler on, sampler off) | 4.03 ± 2.48 | 3.20 | 2.60 | 2.64 | 6.1/14 |

U and V are where the pre-fix arms had them (±0.1 mm). **W halves** (2.82 → 1.46 mm) on the
production arm: with both knobs on, the retiled Steiner input has W charge where it used to
read zero (doc pdhd/01 §4), the fitted trajectories move, and the W profile the fit is compared
to is narrower. W's own-centroid profile is not Gaussian on PDHD (§2.3), so its constant is the
least stable number here.

### 2.2 Stability (`/home/xqian/tmp/pdhd02/ctl/*_fit.tsv`, share-matched joint)

| occasion | D_T,eff | c_U | c_V | c_W |
|---|---|---|---|---|
| primary (adv < 0.436, ±3) | 6.13 ± 2.11 | 3.210 | 2.717 | 1.460 |
| adv < 0.25 | 6.38 ± 2.77 | 3.242 | 2.800 | 0.903 |
| adv < 0.30 | 7.33 ± 2.55 | 3.172 | 2.732 | 1.019 |
| adv < 0.60 | 5.60 ± 2.06 | 3.174 | 2.706 | 1.575 |
| window ±2 | 7.07 ± 1.54 | 3.015 | 2.623 | 1.169 |
| window ±4 | 5.55 ± 2.75 | 3.285 | 2.772 | 1.558 |
| adv < 0.10 | 3.63 ± 3.84 | 3.305 | 3.156 | 0.473 |
| adv 0.10–0.25 | 6.72 ± 2.31 | 3.193 | 2.689 | 1.478 |
| adv 0.25–0.50 | 4.39 ± 2.15 | 3.195 | 2.649 | 1.755 |
| L < 100 cm | 0.3 ± 4.6 | 2.951 | 2.801 | 1.985 |
| L ≥ 100 cm | 7.72 ± 2.18 | 3.167 | 2.678 | 1.325 |
| rr < 10 cm | 4.84 ± 4.17 | 2.989 | 2.830 | 1.399 |
| rr > 30 cm | 8.11 ± 2.46 | 3.132 | 2.653 | 1.276 |

U within ±4 % and V within ±5 % of the primary across every cut; **W spans 0.5–2.0 mm** with
the advance cut and window. D_T,eff is consistent with the physical 8.20 and with zero at
~1.5 σ throughout: the drift axis carries little information once c ≳ 3 mm on U/V.

### 2.3 Shape (`figs/d02_sigma_shape.tsv`, all t, stacked ring shares about the own centroid)

| plane | measured centre / ±1 / ±2 / beyond | Gaussian at the seeds | Gaussian at the shipped σ (share) |
|---|---|---|---|
| U | 0.436 / 0.466 / 0.083 / 0.015 | 0.595 / 0.399 / 0.006 / 0 | 0.436 / 0.483 / 0.078 / 0.003 |
| V | 0.481 / 0.439 / 0.068 / 0.012 | 0.554 / 0.428 / 0.018 / 0 | 0.481 / 0.469 / 0.050 / 0.001 |
| W | 0.622 / 0.303 / 0.058 / 0.017 | 0.753 / 0.247 / 0 / 0 | 0.627 / 0.367 / 0.006 / 0 |

The seeds put too much in the centre wire on every plane; the shipped σ reproduces the centre
and ±1 shares on U/V to ≤ 0.03, and on W the centre only — the W profile has a 7.5 % tail
beyond ±1 that no single Gaussian carries (the rms-matched W σ, 3.07 mm, is a tail-driven
number and describes the core worse; `d02sigb` confirms it, §5).

## 3. The D_T choice

PDVD shipped the share-matched joint line (D_T,eff + three c's, `869a554c`). On PDHD the fitted
slope is 6.13 ± 2.11 cm²/s, consistent with the physical 8.20. With D fixed at 8.2017 and c
fitted alone per plane (`stm/pdhd_track_fitting_d02fix.json`): c = 3.148 / 2.629 / 1.341 mm —
0.06–0.12 mm below the joint values, and the two lines differ by at most ±0.14 mm in σ anywhere
in 200–2070 µs. The graded metrics cannot separate them (§5: `d02sig` and `d02fix` agree on every
row within noise). The joint line ships, as on PDVD (owner decision 2026-09-05): the fit
consumes only the effective line, the joint fit is its least-biased estimate, and the JSON
generator, re-derivation trigger and comment structure stay identical to PDVD's.

## 4. Gates

- **Gate 1 — same epoch.** `d02ref` (canonical JSON, no TLA, pin `e3304cb9`) vs `stmwc` (pin
  `pdhdstm_libpin`, run before the three default-OFF toolkit knobs of 2026-09-05 landed):
  `mabc-pr.zip` member hashes PASS 30/30; `tracking-stm.root` trees SAME on events 0 and 6;
  `T_proj_data` content sha identical (`b6646423fd08561f`, `10d1b8c333b6761a`). The new pin and
  the newer knobs are inert on this chain.
- **Compiled proof.** `.wct-pr_<arm>.json` carries `trackfitting_config` once for every TLA arm;
  the `T_proj_data` sha differs from `d02ref` on `d02sig` / `d02sigb` / `d02fix`
  (`73768f725d515704` / `f2ad5cb04dd3f594` / `2eb02e06cb31d25d`): the runtime file was read.
- **Gate 2 — the flip reproduces the graded arm.** `d02prod` (flipped canonical file, no TLA,
  same pin) vs `d02sig` on events 0 and 6: zip members identical (`3a2ac3a0…`, `27c10835…`),
  trees SAME, `T_proj_data` sha identical; pin md5 unchanged end to end.

## 5. Re-validation (`figs/d02_compare.tsv`, 30 events, medians over status-0 blocks)

| plane | metric | d02ref (seeds) | **d02sig (shipped)** | d02sigb (rms) | d02fix (D fixed) |
|---|---|---|---|---|---|
| U | ring-share gap r_meas − r_pred, r = 1st/(centre+1st) | 0.073 | **0.030** | — | — |
| V | 〃 | 0.047 | **0.018** | — | — |
| W | 〃 | 0.059 | **0.032** | — | — |
| U | uncov_foot | 0.000 [0, 0.006] | 0.000 [0, 0] | 0.000 | 0.000 |
| U | chi2N_foot | 12.73 | 12.60 | 13.97 | 12.48 |
| V | chi2N_foot | 13.02 | 11.66 | 13.13 | 11.98 |
| W | chi2N_foot | 35.57 | 32.40 | 52.84 | 33.19 |
| U | pull_rms | 2.63 | 2.61 | 2.82 | 2.67 |
| V | pull_rms | 2.80 | 2.70 | 2.99 | 2.74 |
| W | pull_rms | 4.38 | 4.67 | 6.52 | 4.63 |
| U | B_foot | −0.157 | −0.182 | −0.220 | −0.186 |
| V | B_foot | −0.172 | −0.195 | −0.234 | −0.197 |
| W | B_foot | −0.264 | −0.248 | −0.348 | −0.259 |
| U | U_foot | 0.533 | 0.520 | 0.556 | 0.521 |
| V | U_foot | 0.571 | 0.595 | 0.594 | 0.583 |
| W | U_foot | 0.432 | 0.447 | 0.577 | 0.448 |
| U/V/W | f_off_near (shells) | 0.059 / 0.054 / 0.005 | 0.057 / 0.054 / 0.004 | — | 0.061 / 0.055 / 0.004 |
| all status 0 | dQ/dx k_pop (n) | 0.971 (172) | 0.958 (165) | 0.906 (174) | 0.964 (173) |
| doc-55 tier (3 tracks) | k_pop | 1.084 | 1.078 | 0.987 (1) | 1.068 |
| STM status 0 | n (flips in / out vs ref) | 178 | 171 (32 / 42) | 180 (47 / 50) | 179 (36 / 38) |

Residual re-derivation on `d02sig` with its own model (`/home/xqian/tmp/pdhd02/d02_sigma_after_fit.tsv`):
σ_eff = 3.390 / 2.980 / 1.808 mm against the model's 3.210 / 2.717 / 1.460 — a quadrature
deficit of **1.09 / 1.22 / 1.07 mm**, χ²/ndf 6.8/14 (was 11.3). The shipped σ reproduces the
stacked U/V shape (§2.3) yet the drift-resolved re-derivation still wants ~1 mm more: the same
self-consistency residual doc 44 §5 found on PDVD (0.69/0.65/0.35 mm there), larger here.

**Verdict against the pre-registered bar.** Ring-share gap ≤ 0.03 on U/V and 0.032 on W: met.
`uncov_foot` → 0: met. U/V χ²/N and pulls better: met (−1/−10 % and −1/−4 %). W: χ²/N −9 % but
pull rms +7 %: not met on W. Residual deficit < 1 mm: not met (1.1–1.2 mm). B_foot moves by
−0.03 on U/V (a normalization deficit, doc 44 §5's reading: the wider kernel spreads the
prediction so the footprint's predicted sum falls further below the measured one; it is not a
width effect). dQ/dx k_pop −1.3 % (within the 30-event noise; the doc-55 tier has 3 tracks and
is reported, not graded). The rms-matched arm is worse on every W row, as predicted by the
shape check. **A weaker gain than PDVD's flip, honestly a mixed one on W**; the owner's decision
to flip stands because the shape — the quantity these keys parametrise — is now right, and
because the alternative (seeds derived from a withdrawn closed form) had no physical standing.
The STM verdict churn (74 of 178 status-0 blocks) is ungraded by hand, as on PDVD.

## 6. What this hands on

1. **The origin of the constant is now understood in simulation (doc pdvd/47):** the
   induction constant is the SP wire filter's real-space kernel where that is wide (PDHD,
   SBND) plus a pitch-scaled term from the 2-D deconvolution against the impact-averaged
   field response; the collection constant on PDHD (1.46 mm) and PDVD (1.18 mm) is *not*
   reproduced by the simulation and is the open data-only item. Re-derive here if
   `sp-filters.jsonnet` changes. **Doc 47 §9** shows the missing width on PDHD is charge at
   ≥ 2 wires from the track that survives every isolation and straightness cut (the excess is
   unchanged on a selection keeping 2.5 % of U's charge and 7.4 % of W's), that it is what
   item 3 below is describing, and that the difference from simulation is made in ROI
   formation rather than in the field response or in front-end cross-talk. Note for anyone
   re-running that study here: PDHD's block `foff` runs 0.71–0.96, so doc 47 §8.6's
   `--max-foff 0.15` selects nothing on this detector — use the `oth4` tag.
   **Doc 47 §10 corrects that reading**: the ≥2-wire tail is under-produced by the simulation
   on all three detectors including SBND, so it is not the ProtoDUNE-specific effect; what is
   ProtoDUNE-specific is a widening of the profile core, worth 1.76 mm on U and 1.41 mm on W
   over the simulated floor (5.2 σ and 4.0 σ; PDHD V is consistent with the floor). A
   deliberate field-response mismatch reaches the induction number but produces **nothing** on
   collection, so W's 1.41 mm is still open. §10.6 splits this measurement by APA — PDHD
   deconvolves APA0 against an MCMC refit to PDHD data and APA1–3 against the generic DUNE
   response, and **the refit does not reduce the excess** (2.01 mm on APA0's U against
   1.93/1.91/2.41 on APA1–3). Only U is quotable there: the PDHD simulation does not apply
   `plane2layer`, so a simulated APA0 event has the collection response on the V readout plane
   while SP deconvolves V as induction (doc 47 §10.8, upstream report).
2. The ~1 mm self-consistency residual (§5) and the −0.03 B_foot shift are the same
   normalization question doc 44 §7 handed on for PDVD; next test is the fit's cell weights
   (`rel_uncer_ind` 0.075).
3. W: a non-Gaussian profile with a 7.5 % tail beyond ±1; a single σ cannot carry it. The
   unused `flag = 1` branch of `cal_gaus_integral` (bipolar induction shape) is the wrong
   tool for a collection plane; a two-component kernel would be the right one if W ever
   becomes the limiting plane.
4. TaggerCheckNeutrino reads the same file; its track_fit layer is ungraded under these
   constants (PDHD runs `-stm` by default, so nothing in production consumes it yet).

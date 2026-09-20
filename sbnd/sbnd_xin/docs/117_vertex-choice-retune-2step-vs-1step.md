# 117 — The vertex-choice re-tune: the 2-step chain against the 1-step chain, on both trajectories

**Status: MEASUREMENT, no SBND production change.** No toolkit C++ and no jsonnet is changed by this
round; every cell is reached through existing default-OFF TLAs plus the runtime TrackFitting JSON.
SBND production stays at `ref/prod-2026-09-17b`.

Owner ask (2026-09-20): *"Let's do the vertex-choice retune round next to see if we can boost the
performance? After that, if things are good, we may profile to improve the running efficiency and
time. … Note, even if there are no clear improvements, since the track trajectory is better, I
wonder whether it is better to do the change? Note, for vertex choice re-tune, please also evaluate
the 2-step chain vs. 1-step chain (exclude fit) etc."*

This is doc 116 sec 10.3's recommended next round. Doc 116 established that the PDHD/PDVD
trajectory changes no selection metric beyond its own churn while two thirds of the vertex movers
are **CHOICE** — the right vertex is still on the candidate list and the chooser takes another — so
the chooser is where a gain would have to come from. Doc 116 sec 16 then showed, with truth, that
the trajectory *does* place the vertex significantly more precisely at 1–2 cm on νe-like events
(599 → 669 of 1 626 within 1 cm, p 0.0003), a gain the 5 cm working point is saturated against.

**Reading of "2-step vs 1-step (exclude fit)"**, stated in the pre-registration before anything ran
(`docs/117_figs/117_pred.txt`, sha `1a7570c4…`): the **2-step chain** is what SBND production runs
today — the *dual chain* of doc pr/112 sec 11, in which a second, exclusion-free PR pass proposes
the neutrino vertex and the production (exclusion-ON) pass decides by snapping the proposal to its
nearest own candidate and accepting it within 2 cm (`dl_vtx_dual_chain=true`,
`dual_chain_mode='snap'`, `dual_chain_transfer_max=2.0`; owner flip 2026-08-23). The **1-step chain
(exclude fit)** is one pass with the second dropped and the surviving pass made the exclusion-free
one (`dl_vtx_dual_chain=false`, `fit_exclusion=false`) — doc pr/112's `nofitx` arm, which scored
best of five strategies on the data hand-scan metric (812 of 1 011, against the dual chain's 805 and
the single exclusion-ON chain's 777). Both halves are separate knobs, so the round runs the
factorial rather than the single contrast.

**Answer.** **Keep the 2-step chain.** Dropping the second pass costs νeCC efficiency — 40.9 → 39.2 %
at the > 7 cut (−65/+39, p 0.014) and 48.6 → 46.5 % at > 4 (−76/+44, p 0.004), the first DEGRADED
verdicts of this two-round arc — and the vertex within 1 / 2 / 3 cm falls with it. The one-step
*exclusion-free* chain (`fit_exclusion=false`, doc pr/112's `nofitx`) is **not** a substitute: on the
current trajectory it merely restores what the second pass was providing (`c1x` ≈ `c2` on every
metric) while trading trajectory quality (more charge claimed twice: `qneg` +0.9 pt, holes +0.35 per
10 m, both p < 1e−25 on `nuecc`), and on the **new** trajectory it is worse than keeping the second
pass (vertex < 5 cm 69.7 → 68.5 %, −38/+19, p 0.016). What the second pass costs is now measured
exactly: **23 % of the PR job** on νe-like events (`TaggerCheckNeutrino` 9.40 → 6.28 s of a 12.90 s
job), so it is an optimisation target, not a deletion target.

**And the trajectory question is no longer neutral.** With the 2-step chain kept, the PDHD/PDVD
trajectory places the vertex significantly better: within 1 cm 36.8 → 41.1 % of 1 626 νe
interactions (−145/+215, p 0.00027), within 2 cm +2.8 pt (p 0.023), the median distance 0.960 →
0.840 cm (p 0.00026), and — on the selected signal's own chosen vertex — 0.850 → 0.780 cm
(308 closer / 223 further, p 0.00026). Every selection metric stays not separable, and the cost is
+33 % of the PR job. By this round's frozen table that is still HOLD (no *primary* metric improves,
and it is not cheaper); but the case for the change has moved from "buys nothing" (doc 116) to "buys
a significant, mechanistically-explained, dose-responsive vertex-precision gain", and sec 10.2 says
what the confirmation round needs.

## 0. Repro

From `wcp-porting-img/sbnd/sbnd_xin`, toolkit `apply-pointcloud` at `0a2807f4`, the doc-115 pin
`~/tmp/d115-libsnap` (`libWireCellClus.so` md5 `71ebd5aeb3868cbc0803f2fa2654b46b` == `local/lib`),
stage A = the doc-115 arms `work-r3{cv,nue,off}-d115` unchanged, and the two 2-step corners
(`products/d115/`, `products/d116/*-tfull`) reused as they are.

```bash
SX=$PWD; D=docs/117_figs
# 0. pre-flight: tripwire, pre-registration, compiled-config proof of the four cells
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b; echo rc=$?        # PASS 21/21
sha256sum -c $D/117_pred.sha256 $D/117_pred_b.sha256
bash scripts/d117/cfg_proof.sh > $D/117_cfg_proof.txt                             # sec 2
# 1. what the 2-step chain did on the arms that already exist -- no new compute (sec 3)
python3 scripts/d117/dual_chain_census.py > $D/117_dual_census.txt
# 2. stage B for the four new cells, N lock-sharing workers (same launcher design as doc 116)
for c in c1x t1x c1e t1e; do for s in cv nuecc off; do JOBS=6 scripts/d117/stageB_cell.sh $c $s; done; done
# 3. the doc-115 analysis on every new arm
for c in c1x t1x c1e t1e; do for s in cv nuecc off; do JOBS=8 scripts/d117/analyze_cell.sh $c $s; done; done
# 4. the grid, paired against both corners (sec 5-7)
python3 d117_compare.py --arm c2=products/d115/{s} --arm t2=products/d116/{s}-tfull \
  --arm c1e=products/d117/{s}-c1e --arm c1x=products/d117/{s}-c1x \
  --arm t1e=products/d117/{s}-t1e --arm t1x=products/d117/{s}-t1x \
  --pair c1e:c2 --pair c1x:c2 --pair t2:c2 --pair t1e:c2 --pair t1x:c2 \
  --pair t1e:t2 --pair t1x:t2 --pair c1x:c1e --pair t1x:t1e --out $D/117_compare
# 5. the closure clauses on every cell (fit_exclusion=false reverts a FIT flip -- 117_pred_b.txt P3)
for a in c1e c1x t1e t1x; do for s in cv nue; do
  python3 scripts/d117/traj_eval.py extract --arm-dir work-r3$s-d117$a --label $s-$a; done; done
# 6. cost (doc 116 sec 14's instrument)
python3 scripts/d116/perf_cost.py cv nuecc off > $D/117_cost.txt
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b; echo rc=$?        # PASS 21/21 after
```

## 1. The grid

Trajectory (rows) × vertex chain (columns). **Two corners already exist and are not re-run**: the
doc-115 baseline arms are `c2`, and doc 116's `tfull` arms are `t2`.

| | 2-step (dual, snapD2) | 1-step, exclusion ON | 1-step, exclusion-free |
|---|---|---|---|
| current trajectory | `c2` = doc-115 baseline | `c1e` | `c1x` |
| PDHD/PDVD (`tfull`) | `t2` = doc-116 `tfull` | `t1e` | `t1x` |

- `c1e` — `dl_vtx_dual_chain=false`. The documented production revert, one key.
- `c1x` — `+ fit_exclusion=false`. doc pr/112's `nofitx`.
- `t1e`, `t1x` — the same two on top of `tfull`'s four trajectory keys plus the runtime fit JSON
  (`docs/117_figs/tla/t1{e,x}.tfjson` → `SBND_TRACKFIT_JSON`; the fit keys never reach the compiled
  jsonnet, doc pr/150's trap).

Samples, stage A, pin and output set are docs 115/116's exactly: `cv` 2 017, `nuecc` 2 001, `off`
1 000; `PR_EXTRA_STAGES=pr_display`; the arms differ from the baseline only in the TLA set, which
lands in `Trun.op_config_sha256`.

**Noise floor.** Doc 116 sec 4 measured it on this pin, box and inputs: the production configuration
re-run reproduced the baseline tables row for row, every exchange −0/+0. This round adopts floor = 0
(so the rule is |Δ| > 0.5 pt **and** p < 0.05) and verifies it by re-running one `nuecc` sub-root of
`t1x` into a throwaway out_root and comparing row for row.

## 2. The cells compile to what they claim

`scripts/d117/cfg_proof.sh`, the runner's own `_optla` form, node-diffed against the no-TLA compile
(`docs/117_figs/117_cfg_proof.txt`). The null compile's sha is `4ee005c935dac7df` — **identical to
doc 116's**, so the compile environment has not moved.

| cell | sha | node diff against the production compile |
|---|---|---|
| `c1e` | `6b729e097dab7328` | **one node, one key**: `TaggerCheckNeutrino:pr` loses `dl_vtx_dual_chain` |
| `c1x` | `254528c52d52d137` | the same node loses `dl_vtx_dual_chain` **and** `fit_exclusion` |
| `t1e` | `dbd0a6352c56f9c3` | `tfull`'s six changed nodes (sampler swap on both APAs, the three Steiner keys on `CreateSteinerGraph:pr` and `:prrefresh`, `trackfitting_config_file` on `TaggerCheckNeutrino:pr` and `TaggerCheckSTM:pr`) **minus** `dl_vtx_dual_chain` |
| `t1x` | `4ddd562e6564543e` | the same, also minus `fit_exclusion` |

`dual_chain_mode`, `dual_chain_transfer` and `dual_chain_transfer_max` keep their production values
and are inert when the pass does not run — the revert is the single documented key, and the diff
shows nothing else moved.

## 3. What the 2-step chain actually does — measured on the arms that already ran

`scripts/d117/dual_chain_census.py` → `docs/117_figs/117_dual_census.txt`. No new compute: every
event of `c2` and `t2` carries the second pass's own record in its calib dump
(`vertex_scoreboard.dual_chain`: `agree`, `transferred`, the snap distance `d`, and `off_ms`, the
second pass's wall time), and `rows[]` carries every candidate vertex with the composite rerank
score `total` that the production pass would have decided on alone.

| | `c2` cv | `c2` nuecc | `t2` cv | `t2` nuecc |
|---|---:|---:|---:|---:|
| events with a scoreboard | 955 | 1 870 | 952 | 1 869 |
| second pass ran | 929 | 1 867 | 931 | 1 865 |
| chains agreed | 897 | 1 753 | 901 | 1 762 |
| transfer accepted (≤ 2 cm) | 445 | 669 | 452 | 611 |
| … and it **moved the pick** | 346 | 614 | 349 | 572 |
| snap distance, median / p90 (cm) | 0.325 / 1.31 | 0.497 / 2.07 | 0.272 / 1.27 | 0.413 / 1.80 |
| second-pass cost, median (ms) | 1 799 | 2 895 | 1 573 | 3 249 |
| second-pass cost, arm total (core-h) | 0.58 | 2.58 | 0.53 | 3.16 |

**The transfers are right far more often than they are wrong.** On the events where the accepted
snap moved the pick away from the composite winner, the transferred vertex is closer to a true
in-FV vertex than the composite winner would have been:

| arm / sample | closer | further | same | exact two-sided sign p |
|---|---:|---:|---:|---:|
| `c2` cv | **159** | 58 | 1 | 4.6e−12 |
| `c2` nuecc | **318** | 164 | 0 | 2.1e−12 |
| `t2` cv | **172** | 51 | 1 | 1.6e−16 |
| `t2` nuecc | **294** | 157 | 0 | 1.1e−10 |

(128 / 132 / 125 / 121 of those events have no true in-FV interaction to score against and are
excluded.) This is a **proxy**: it does not replay the acceptance logic the no-dual arm would use
(the `min_accept` floor, the distance gate, the traditional fallback), which is exactly what the
`c1e` / `t1e` arms measure by running it. It was the prediction frozen in `117_pred.txt` before it
was computed, and its confirmation flipped the round's live hypothesis — recorded, before any cell
was analysed, in `117_pred_b.txt` (sha `e1f9b1ba…`).

Against doc 116 sec 14's TICK totals (7.17 core-h for `c2` nuecc, 9.52 for `t2` nuecc) the second
pass is **36 % / 33 % of the whole PR compute** on the νe-rich sample, and 32 % / 28 % on `cv`
(0.58 of 1.83, 0.53 of 1.87). That is the price the 1-step cells stop paying.


## 4. What ran

Twelve new arm-samples, every one complete, `rc ≠ 0` = **0**, one `op_config_sha256` per cell and
reality (the `cv`/`nuecc` arms share a hash because they share `reality=sim`; the beam-off arms
differ because they are `reality=data`):

| cell | cv | nuecc | off | `op_config_sha256` (sim / data) |
|---|---:|---:|---:|---|
| `c1e` | 2 017 | 2 001 | 1 000 | `7a49fed2e77b0472` / `2d90f9e8a2093fa2` |
| `c1x` | 2 017 | 2 001 | 1 000 | `808a9dbc19c2d442` / `7a6fb2ef6b272176` |
| `t1e` | 2 017 | 2 001 | 1 000 | `89193b4342c08142` / `cc87a5b92338f502` |
| `t1x` | 2 017 | 2 001 | 1 000 | `43a34296ddb37bd5` / `3c0d652ac99e11e8` |

Eight to fourteen lock-sharing workers at `PR_JOBS=6`. The tripwire
(`prod_cfg_gate.py --ref ref/prod-2026-09-17b`) is **PASS 21/21 before and after**.

**The floor is verified, not assumed.** `117_pred.txt` promised a determinism re-run: one `nuecc`
sub-root of `t1x` was re-run into a throwaway arm (`t1xrep`, a byte copy of `t1x.tla`). Same
`op_config_sha256` `43a34296ddb37bd5`, all 10 events' `nusel-evt*.tsv` identical row for row, and
the DL vertex scoreboard — `route`, `final_x/y/z`, `dl_best_score` — identical on 10 of 10. CLAUDE.md
M4's "the DL vertex is not bit-stable" again did not bite on this pin and box, so every Δ below sits
on a floor of 0.

**One defect of the harness, found and fixed mid-round.** `scripts/d117/analyze_cell.sh`, inherited
from the doc-115/116 drivers, had **no completeness guard**, and the beam-off table for `c1e` was
first built from a partially written arm (504 of 1 000 events). It was caught by reading the log,
the table was rebuilt on the full arm, every other products table was checked, and the script now
refuses an arm whose `pr_evt` count or `rc=0` count does not match stage A (`FORCE=1` overrides).
The guard immediately earned itself by refusing the still-running `t1e` nuecc arm.

**Ops note.** `loadavg` is not the licence meter here: it counts I/O wait, and this chain is
I/O-bound. At 14 workers `loadavg` read 31–46 while sustained CPU was 29–36 cores of 64; the fleet
was cut to 8 and reniced to +10 when a 80 s average did cross 33 cores. Sustained `user+sys` over a
minute is the number to watch.

## 5. The grid

`docs/117_figs/117_compare.{txt,tsv}` (`d117_compare.py`, which imports doc 116's selection logic,
`Arm` and exact sign test unmodified, so both rounds score with the same code). Every cell is paired
against the shipping configuration `c2` **and** against its own row corner.

### 5.1 The primary metrics — two DEGRADED verdicts, no IMPROVED

| id | metric | cell vs base | base | cell | Δ (pt) | −lost/+gain | p | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| M3 | νeCC eff > 7 | `c1e` vs `c2` | 40.9 % | 39.2 % | −1.72 | −65/+39 | 0.014 | **DEGRADED** |
| M5 | νeCC eff > 4 | `c1e` vs `c2` | 48.6 % | 46.5 % | −2.12 | −76/+44 | 0.004 | **DEGRADED** |
| M8 | vertex < 5 cm, nuecc | `t1x` vs `t2` | 69.7 % | 68.5 % | −1.17 | −38/+19 | 0.016 | **DEGRADED** |
| M3 | νeCC eff > 7 | `t1e` vs `t2` | 41.7 % | 40.5 % | −1.19 | −62/+44 | 0.098 | not separable |
| M5 | νeCC eff > 4 | `t1e` vs `t2` | 50.1 % | 49.1 % | −0.99 | −65/+50 | 0.191 | not separable |
| M8 | vertex < 5 cm, nuecc | `c1e` vs `c2` | 69.0 % | 67.4 % | −1.60 | −97/+71 | 0.053 | not separable |
| — | every other primary, every cell | — | — | — | ≤ ±1.3 | — | ≥ 0.1 | not separable |

The three DEGRADED verdicts all point the same way: **taking the second pass away costs, and the
exclusion-free single pass does not pay it back on the better trajectory.** No cell improves a
primary metric against `c2`.

### 5.2 The vertex at 1 cm — the whole grid in one table

`nuecc`, 1 626 true interactions in the FV, nearest candidate vertex, paired, floor 0:

| | 2-step (production) | 1-step, exclusion ON | 1-step, exclusion-free |
|---|---:|---:|---:|
| current trajectory | `c2` **599** (36.8 %) | `c1e` 573 (−1.60 pt, **p 0.010 DEGRADED**) | `c1x` 601 (+0.12, p 0.94) |
| PDHD/PDVD trajectory | `t2` **669** (+4.31 vs `c2`, **p 0.00027 IMPROVED**) | `t1e` 654 (+3.38 vs `c2`, **p 0.005 IMPROVED**; −0.92 vs `t2`, p 0.15) | `t1x` 637 (−1.97 vs `t2`, **p 0.017 DEGRADED**) |

and the same shape at 2 and 3 cm (`c1e` −2.15 p 0.007 and −1.97 p 0.018 against `c2`; `t1x` −1.60
p 0.013 and −1.17 p 0.032 against `t2`). Reading the rows and columns:

- **the trajectory is the largest single effect** — +70 interactions at 1 cm, and it survives the
  chain change (`t1e` is still +3.38 pt over `c2`, p 0.005);
- **the second pass is worth ~26 interactions at 1 cm on the current trajectory** (`c1e` → `c2`) and
  ~15 on the new one (`t1e` → `t2`, not separable);
- **exclusion-free is not a drop-in replacement for the second pass**: it recovers the loss on the
  current trajectory (`c1e` → `c1x` +1.72 pt, p 0.093) and *costs* on the new one (`t1e` → `t1x`
  −1.05 pt, p 0.30). Neither half of that interaction is separable on its own; it is reported as the
  suggestion it is, and sec 10.2 says how to settle it.

### 5.3 What the chains do NOT change: the vertex of the events that survive

The same comparison on the **selected candidate's own vertex** (`t_dist_cm`, signal selected in both
arms — the metric M7/M8 cannot see, because they score the *nearest* candidate and are therefore
nearly blind to which one the chooser takes):

| comparison | sample | n | median (cm) | closer/further | p |
|---|---|---:|---|---:|---:|
| `t2` vs `c2` | nuecc > 4 | 562 | 0.850 → **0.780** | 308/223 | **0.00026** |
| `t2` vs `c2` | nuecc > 7 | 450 | 0.800 → **0.740** | 245/179 | **0.0016** |
| `t2` vs `c2` | cv νμ | 348 | 0.830 → **0.760** | 185/141 | **0.017** |
| `c1e` vs `c2` | nuecc > 7 | 553 | 0.800 → 0.820 | 3/6 | 0.51 |
| `t1e` vs `t2` | nuecc > 7 | 568 | 0.750 → 0.750 | 5/4 | 1.00 |

**The trajectory moves the shipped vertex of selected signal closer to truth; the chain does not
move it at all.** What the chain changes is *which events survive the selection*: `c1e` loses 65 νeCC
interactions at the > 7 cut and gains 39, and among the events selected in both arms the vertex is
the same to within 9 events of 658. The second pass is not improving the vertices you keep — it is
keeping events whose vertex would otherwise land far enough off that the νe BDT drops them.

## 6. Trajectory closure: which knob touches the fit

`scripts/d117/traj_eval.py` (doc pr/150's instrument, reused as a module), the pre-registered P3 of
`117_pred_b.txt`. Two results, each replicated on both trajectories:

**The dual chain does not touch the trajectory.** `c2` → `c1e` on cv: every clause flat, largest
|p| = 0.14, and only ~45 of 890 common events differ on any clause at all. That is the control that
lets the grid be read as a factorial — the second pass is a vertex-choice mechanism and nothing else.

**`fit_exclusion=false` changes the fit, in both directions.** `t2` → `t1x` on `nuecc` (1 800+
events with a main-cluster fit):

| clause | `t2` | `t1x` | better/worse | p | same on `c2` → `c1x`? |
|---|---:|---:|---:|---:|---|
| R2D_W, rows > 1 wire off | 0.0145 | 0.0137 | 303/177 | **9.6e−9 better** | same sign, n.s. on cv |
| P4 image coverage | 0.6767 | 0.6819 | 763/494 | **3.3e−14 better** | yes (0.0008) |
| med_d_W in-cell residual | 0.2459 | 0.2455 | 665/620 | 0.22 | yes (0.012 better) |
| P3 holes per 10 m | 2.355 | 2.706 | 545/947 | **1.5e−25 worse** | yes (0.014) |
| qneg, rows with q < 0 | 7.55 % | 8.45 % | 411/890 | **5.4e−41 worse** | yes (0.046) |
| P1, P2, uncov, wig1 | — | — | — | ≥ 0.13 | yes |

This is what removing the exclusion should do: the fit may claim charge shared between prongs, so it
follows the image more closely (R2D_W, P4) and the same charge is counted twice, so far more rows go
negative and more Bee-visible holes appear (qneg, P3). It is a real trade, not a free lunch, and it
is why "the exclusion-free chain scored best on the data hand scan" (pr/112 sec 12.2) could not be
adopted on that evidence alone.

## 7. Cost

`scripts/d117/cost.py` → `docs/117_figs/117_cost.txt`, doc 116 sec 14's instruments (the job's own
TICK ladder; `getrusage(CHILDREN)` for memory). The d117 arms ran under a varying worker count, so
the totals carry doc 116's ±10 % environment term — but the signal is stage-localised, and the
second pass's own cost is recorded per event in the `c2`/`t2` arms as `dual_chain.off_ms`, which
needs no control at all.

| arm | `nuecc` TICK mean | vs `c2` | `TaggerCheckNeutrino` mean | `cv` TICK mean | vs `c2` | RSS max (nuecc) |
|---|---:|---:|---:|---:|---:|---:|
| `c2` (production) | 12.90 s | 1.000 | 9.40 s | 3.26 s | 1.000 | 1.51 GiB |
| `c1e` | 9.90 s | **0.77** | 6.28 s | 2.62 s | 0.81 | 1.52 |
| `c1x` | 9.28 s | **0.72** | 5.44 s | 2.58 s | 0.79 | 1.51 |
| `t2` | 17.13 s | 1.33 | 11.51 s | 3.34 s | 1.02 | 2.21 |
| `t1e` | 12.27 s | 0.95 | 6.80 s | 2.91 s | 0.89 | 2.21 |
| `t1x` | 11.83 s | 0.92 | 6.10 s | 2.87 s | 0.88 | 2.20 |

The second pass is **~3 s of the 9.4 s tagger visit ≈ 23 % of the whole PR job** on νe-like events
(the census's own `off_ms` says 2.58 core-h of `c2` nuecc's 7.17, i.e. 36 % — the two instruments
bracket it because `off_ms` is measured under the doc-115 load and the TICK difference under this
round's). On `cv` it is 20 %. Memory is unchanged by the chain in every cell; the νe tail at
2.2 GiB belongs to the trajectory (doc 116 sec 14), not to the vertex chain.

So the arithmetic the owner's profiling round faces: **23 % of the PR job buys 1.7–2.1 pt of νeCC
efficiency.** Deleting it is not a trade worth making; making it cheaper is (sec 10.3).

## 8. Beam-off

1 000 Run-1 off-beam gates per cell, paired by gate; the pre-registration counts a change only from
5 gates, and doc 116 sec 15.2 showed the baseline's gate count is flat at 5 over the whole cut
region, so these are directions, not results.

| cell | νμ-selected gates vs `c2` | exchange | p |
|---|---:|---:|---:|
| `t2` | 5 → 4 | −1/+0 | 1.00 |
| `c1e` | 5 → 4 | −1/+0 | 1.00 |
| `c1x` | 5 → 7 | −0/+2 | 0.50 |
| `t1e` | 5 → 6 | −1/+2 | 1.00 |
| `t1x` | 5 → 6 | −1/+2 | 1.00 |

The two exclusion-free cells sit one to three gates above their exclusion-ON partners (`c1x` vs
`c1e` −0/+3), which is the direction a fit that may claim shared charge should push a cosmic-only
sample. No νe cut admits a single gate in any cell (0/1 000 everywhere).

## 9. The route census

`scripts/d117/dual_chain_census.py` also prints the decision route of every event. In `c2`/`nuecc`,
1 661 of 1 870 scoreboards read `dl-dual-snap-accept` and 162 `dl-dual-snap-reject-accept`: the
second pass is on the critical path of **97 %** of the νe events that have a vertex at all, and the
snap is accepted on 669 of them, moving the pick on 614. That is the population the DEGRADED
verdicts of sec 5.1 come from.

The same census on the 1-step cells is the runtime proof that the knob did what the compiled config
said: **"second pass ran: 0 (0.0 %)"** in `c1e`, `c1x`, `t1e` and `t1x`, and the routes move wholesale
from `dl-dual-snap-accept` (1 661 of 1 870 in `c2`) to `dl-rerank-accept` (1 495 in `c1e`, 1 516 in
`t1x`) — the composite rerank deciding alone, which is exactly the path the second pass used to
override.

## 10. The owner's three questions

### 10.1 "2-step chain vs 1-step chain (exclude fit)" — keep the 2-step chain

Measured with truth on 5 018 events per cell: dropping the second pass costs 1.7–2.1 pt of νeCC
efficiency (p 0.014 / 0.004) and 1.6–2.2 pt of vertex accuracy at 1–3 cm (p 0.007–0.018) on the
current trajectory, and the same sign, not separable, on the new one. Substituting the exclusion-free
fit for it is not a fix: it restores the vertex on the current trajectory but trades fit quality
(sec 6), and on the new trajectory it is the worst cell of the grid (M8 −1.17 pt, p 0.016). doc
pr/112's data hand-scan ranking (`nofitx` 812 > `snapD2` 805 > single chain 777) is **not**
reproduced with truth; the part of it that is reproduced is that the single exclusion-ON chain is
the worst of the three on the current trajectory.

### 10.2 "Even if there are no clear improvements, since the trajectory is better, should we change?"

The premise has moved. Doc 116 could say only "no selection change, and the trajectory is better by
its own closure clauses". This round adds a **downstream** measurement: the trajectory places the
vertex significantly closer to truth — +4.31 pt within 1 cm (p 0.00027), +2.8 pt within 2 cm, median
0.960 → 0.840 cm, and 0.850 → 0.780 cm on the *selected* signal's own vertex (p 0.00026) — with no
primary metric degraded and the effect dose-responsive in the knobs (doc 116 sec 16.3).

Against that: it costs +33 % of the PR job on νe-like events and a 2.2 GiB memory tail (doc 116
sec 14), it re-decides 10–13 % of the selected signal's membership, and the gain lives in a metric
this round pre-registered as **secondary**. The frozen table in `117_pred.txt` therefore says
**HOLD** — a package is adopted only if a *primary* metric improves or it is cheaper, and neither
holds.

**Recommendation.** Do not flip yet, and do not let that read as "the evidence is unchanged" — it
is not. Run the confirmation round, which is small because everything for it exists:

1. pre-register the vertex-within-1-cm metric (and the median distance) as **primary**, with the
   5 cm M7/M8 kept as the saturated control;
2. run `t2` against `c2` on an **independent** νe sample (the doc-115 stage A is reusable; a fresh
   intrinsic-νe file set is the only new input), because the present evidence is one sample with
   p 0.0003 and one (`cv`) with the same sign at p 0.078;
3. flip only if it replicates, and then flip the trajectory **with** the 2-step chain, never without.

That is a two-command re-run once the sample exists, and it converts a secondary signal into a
primary verdict instead of arguing about the rule after the fact.

### 10.3 "Then profile to improve the running efficiency"

The target is now identified and sized. The second pass is 23 % of the PR job on νe events and it is
earning its keep, so the profiling round should make it **cheaper**, not absent. Three leads, in
order of how much the measurements here support them:

1. **Run it only where it can matter.** The snap is accepted on 669 of 1 867 νe events and moves the
   pick on 614; on the other ~1 200 the second pass is computed and discarded. A cheap predicate
   (chain agreement is already recorded as `dual_chain.agree`, and `off_ms` is per event) could gate
   the pass. Requires a new default-OFF knob and its own byte-identical gate.
2. **Reuse instead of recompute.** The OFF pass duplicates the whole sequence on its own graph and
   its own `TrackFitting` (`run_dual_chain_off_pass`); what the production pass needs from it is one
   vertex position. Whether the earlier stages can be shared is a code question this round does not
   answer.
3. **The trajectory's own cost** (+33 %) sits in `CreateSteinerGraph` (doc 116 sec 14: ×2.9 / ×3.4
   from `charge_stepped`), which is a separate optimisation target with a separate owner decision.

## 11. Verdicts against the frozen rules

| frozen statement | outcome |
|---|---|
| `117_pred.txt` P-set: primary M1–M8, floor 0, rule \|Δ\| > 0.5 pt **and** p < 0.05 | applied unchanged; 3 DEGRADED, 0 IMPROVED among primaries |
| `117_pred.txt`: determinism re-run of one `t1x` sub-root | done, identical on both instruments (sec 4) |
| `117_pred.txt` ADOPT table | no cell qualifies: none has a primary IMPROVED, and the only cheaper cells (`c1e`, `c1x`, `t1e`, `t1x`) either degrade a primary or trade closure |
| `117_pred_b.txt` P1: "`c1e`/`t1e` expected WORSE on the vertex" | **confirmed** on the current trajectory (p 0.010–0.018 at 1–3 cm), same sign and not separable on the new one |
| `117_pred_b.txt` P2: "does the exclusion-free single pass recover it?" | on the current trajectory yes (`c1x` ≈ `c2`); on the new one no (`t1x` is DEGRADED vs `t2`) |
| `117_pred_b.txt` P3: closure graded on all four cells | done, sec 6 — the trade is real and replicated |
| `117_pred_b.txt` P4: cost as a stated outcome | done, sec 7 |
| the zero-cost proxy (`117_dual_census.txt`) predicted the transfers are net-favourable | confirmed by the arms: removing them degrades νeCC efficiency and the vertex |

## 12. Open items

1. The confirmation round of sec 10.2 — an independent νe sample with the 1 cm metric pre-registered
   as primary. Nothing else in this arc can turn the trajectory question from HOLD into a flip.
2. The interaction of sec 5.2 (exclusion-free helps the old trajectory, hurts the new one) is
   suggestive at p 0.09 / 0.30 and would need its own arm pair to establish.
3. The doc-115 and doc-116 open items carry: the off-beam gate count `f`, the νeCC purity needing a
   full CV production, the `dvm()` CPA face applied to MC.
4. `analyze_cell.sh`'s completeness guard exists only in `scripts/d117/`; `scripts/d115` and
   `scripts/d116` still have none. They are records of what ran and are not edited here (M13), but a
   future round should fork the guarded version.

## 13. Files

`scripts/d117/{stageB_cell.sh,analyze_cell.sh,cfg_proof.sh,traj_eval.py,dual_chain_census.py,cost.py}`,
`d117_compare.py`; `docs/117_figs/tla/{c1e,c1x,t1e,t1x,t1xrep}.tla` + `t1{e,x}.tfjson`,
`117_pred.txt` + `.sha256`, `117_pred_b.txt` + `.sha256`, `117_cfg_proof.txt`, `117_dual_census.txt`,
`117_compare.{txt,tsv}`, `117_cost.txt`, `traj/*.tsv`; `docs/117_{sel,sel_edep100,vtx,scan,time,off}/`;
`products/d117/<sample>-<cell>/`. Nothing under `docs/115_*`, `docs/116_*`, `products/d115`,
`products/d116`, `scripts/d115`, `scripts/d116` (except the two doc-116 sec-16 scripts this round's
sibling commit added), or any toolkit file is modified by this round.

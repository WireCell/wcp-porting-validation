# 116 — The PDHD/PDVD trajectory, fit keys and `charge_stepped` on SBND, scored with truth against the doc-115 baseline

**Status: MEASUREMENT, no SBND production change.** No toolkit C++ and no jsonnet is changed by
this round; every study cell is reached through existing default-OFF TLAs plus the runtime
TrackFitting JSON. SBND production stays at `ref/prod-2026-09-17b`.

Owner question (2026-09-20): docs pr/149 and pr/150 integrated what PDHD and PDVD production now
run — (a) the `charge_stepped` PR retile sampler, (b) the priced Steiner seed
(`terminal_blank_plane_mode='prefer3'`, `base_weight_blank_alpha=0.5`,
`base_weight_scope='tree+path'`), (c) the two lattice fit keys `fit_weight_pow 1.5` /
`assoc_cont_center 1` — and judged them on SBND **data** with hand labels only: the trajectory fit
closed better on every clause, but the neutrino PR re-decided (vertex ≤ 3 cm 677 → 620 of 823,
symmetric churn), the retune ladder failed, nothing was flipped. Doc 115 now gives truth on the
round-3 MC. Re-score the same cells; say whether the changes improve the **final** results; where
they degrade, find the origin and try to improve it.

**Answer.** On 2017 `mc-cv` + 2001 `mc-nuecc` events, the full PDHD/PDVD configuration (`tfull`)
changes **no primary metric beyond its own churn**: νμCC efficiency 69.5 → 70.0 %, νeCC efficiency
40.9 → 41.7 % (> 7) and 48.6 → 50.1 % (> 4), purities within ±0.7 pt, vertex-within-5 cm −0.8 /
+0.7 pt — every delta "not separable" under the pre-registered rule, because each one sits on an
exchange of 40 (νμCC) to 180 (νeCC) signal interactions lost and about as many gained. The noise
floor is **zero**: a re-run of the production configuration reproduces the baseline's candidate and
truth tables row for row. The trajectory itself is better on MC exactly as on data (W-plane rows
> 1 wire off 0.81 → 0.54 % on `cv`, 2.13 → 1.44 % on `nuecc`; off-charge rows −30 / −33 %;
uncovered charge −0.8 / −2.0 pt). With truth, the churn has a name: two thirds of the vertex movers
in **both** directions are **CHOICE** — a main-candidate vertex within 5 cm of the true one exists
in the arm that missed, and the chooser (the DL dual-snap re-rank, on the same route in both arms)
took another — and on the score-stage movers the whole tagger block re-evaluates (νe score moves by
a median of 7 points either way). Nothing is DEGRADED by the rule, so no retune rung is triggered;
the improvement question is answered in sec 10. **Recommendation: do not flip on this evidence.**
The trajectory earns nothing at the selection level until the vertex-choice stage is re-tuned
against it — and this round delivers the truth-adjudicated mover set that retune needs
(`116_figs/116_movers_*.tsv`), which doc pr/150 could only get from a blind hand scan.

## 0. Repro

From `wcp-porting-img/sbnd/sbnd_xin`, toolkit `apply-pointcloud` at `0a2807f4`, the doc-115 pin
`~/tmp/d115-libsnap` (`libWireCellClus.so` md5 `71ebd5aeb3868cbc0803f2fa2654b46b` == `local/lib`),
stage A = the doc-115 arms `work-r3{cv,nue,off}-d115` unchanged.

```bash
SX=$PWD; D=docs/116_figs
# 0. pre-flight: tripwire, pre-registration, compiled-config proof of every cell (standalone half)
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b; echo rc=$?        # PASS 21/21
sha256sum -c $D/116_pred.sha256                                                      # frozen before the first arm
bash scripts/d116/cfg_proof.sh > $D/116_cfg_proof.txt                               # sec 2
# 1. stage B per cell, N lock-sharing workers per cell (sub-roots are 6-21 events; one loop idles the box)
for c in s0rep tfull csp3bw cs p3bw; do for w in 1 2 3 4 5; do
  nohup bash -c "for s in cv nuecc off; do JOBS=6 $SX/scripts/d116/stageB_cell.sh $c \$s; done" \
    > ~/tmp/d116/cell-$c-w$w.log 2>&1 &
done; done                                                                           # tfull exports SBND_TRACKFIT_JSON itself (tla/tfull.tfjson)
# 2. the doc-115 analysis on every arm (truth_base.tsv reused; truth.tsv regenerated per arm)
for c in s0rep tfull csp3bw cs p3bw; do for s in cv nuecc off; do JOBS=8 scripts/d116/analyze_cell.sh $c $s; done; done
bash scripts/d116/cfg_proof.sh > $D/116_cfg_proof.txt                               # sec 2, now with the arms' own .d109-opcfg.json
# 3. the paired comparison, the movers, the trajectory closure
python3 d116_compare.py --cells s0rep cs p3bw csp3bw tfull --out $D/116_compare     # secs 4, 5, 8, 9
for s in cv nuecc; do for c in tfull csp3bw cs p3bw; do python3 d116_movers.py --sample $s --cell $c; done; done   # sec 7
python3 scripts/d116/traj_eval.py extract --arm-dir work-r3cv-d115pr  --label cv-baseline    # sec 6 (and nuecc, and each cell)
python3 scripts/d116/traj_eval.py compare --a cv-baseline --b cv-tfull
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b; echo rc=$?        # PASS 21/21 after
```

Tables: `products/d116/<sample>-<cell>/`. Figures: `docs/116_sel/`, `docs/116_sel_edep100/`,
`docs/116_vtx/`, `docs/116_scan/`, `docs/116_time/`, `docs/116_off/`, `docs/116_figs/`.

## 1. Samples, baseline, definitions

The three doc-115 samples and their baseline arms, unchanged:

| key | events | stage A (shared) | baseline stage B | baseline numbers (doc 115) |
|---|---:|---|---|---|
| `cv` | 2 017 | `work-r3cv-d115/f000..f153` | `work-r3cv-d115pr` | νμCC eff 387/557 = 69.5 %, purity 387/448 = 86.4 %, vertex < 5 cm 539/783 = 68.8 % |
| `nuecc` | 2 001 | `work-r3nue-d115/f000..f224` | `work-r3nue-d115pr` | νeCC eff 618/1511 = 40.9 % (> 7), 734/1511 = 48.6 % (> 4); purity 97.8 / 94.0 %; vertex 1122/1626 = 69.0 % |
| `off` | 1 000 | `work-r3off-d115` | `work-r3off-d115pr` | 5/1000 gates pass νμ > 0.9 in FV; 0/1000 pass either νe cut |

Definitions are doc 107 sec 5.5's, applied by the doc-115 scripts **unmodified**: FV
`5 < |x| < 190, |y| < 190, 10 < z < 450` cm; match = reco vertex within 5 cm of the true vertex;
selection = score cut AND reco vertex in FV; sign-blind flavour; every in-FV interaction is its own
row. Every knob here is a **PR-job** knob, so stage A (imaging, clustering, Q/L, the pctree) is
shared by construction and only stage B is re-run — with the **same binary** as the baseline arms.
The only thing that differs between a cell arm and the baseline arm is the TLA set, and that set
is hashed into every event's `Trun.op_config_sha256` (sec 2).

## 2. The cells and their proofs

`docs/116_figs/tla/`, byte-copies of doc pr/150's:

| cell | TLAs (`PR_EXTRA_TLA` file, one `--tla-code` per line) | + | = |
|---|---|---|---|
| `s0rep` | none | | production re-run: the noise floor |
| `cs` | `retile_sampler_strategy='charge_stepped'` | | (a) |
| `p3bw` | `steiner_blank_plane_mode='prefer3'`, `steiner_base_weight_blank_alpha=0.5`, `steiner_base_weight_scope='tree+path'` | | (b) |
| `csp3bw` | cs + p3bw | | (a)+(b), the doc-116 trajectory |
| `tfull` | csp3bw | `SBND_TRACKFIT_JSON=docs/pr/149_figs/149_tf_sbnd_kf.json` (`tla/tfull.tfjson`) | (a)+(b)+(c) = **PDHD/PDVD production on SBND** |

Two facts behind the last row. The fit keys are **not jsonnet**: they live in the TrackFitting JSON
read at runtime (`run_pr_chain_batch.sh:134-136` → `--tla-str trackfitting_config=…`), so
`tfull.tla` alone reproduces `csp3bw` byte for byte (doc pr/150's trap; the launcher exports the
JSON from the sibling `.tfjson` file so it cannot be forgotten). And `tfull` **is** the whole
PDHD/PDVD point on SBND: SBND already runs `stm_proton_muon_guard=true`, has no Michel stage, and
carries no other key that the doc-116 flip touched.

**Compiled-config proof, `docs/116_figs/116_cfg_proof.txt`.** (a) Standalone, in the runner's own
`_optla` form (reality `sim`, production pipeline + `pr_display`): `s0rep` is **IDENTICAL** to the
no-TLA compile (`4ee005c935dac7df`); `cs` changes only the two `BlobSampler:live-*` nodes (→
`live-cs-*`) and `ImproveCluster_2:pr.samplers`; `p3bw` changes only the three keys on
`CreateSteinerGraph:pr` and `:prrefresh`; `csp3bw` is the union; `tfull` adds
`trackfitting_config_file` on `TaggerCheckNeutrino:pr` **and** `TaggerCheckSTM:pr` (both consumers
of the fit JSON) plus the provenance string. Exactly doc pr/150's `150_cell_cfg.txt`, plus the
JSON path. (b) On the arms: the runner's own `.d109-opcfg.json` — the file it hashes into
`Trun.op_config_sha256` — node-diffed against the baseline arm's on every sub-root:

| arm | sub-roots | distinct `op_config_sha256` | node diff vs baseline |
|---|---:|---|---|
| `s0rep` cv / nuecc | 154 / 225 | `399badee163e0f4d` (== baseline) | none |
| `tfull` cv / nuecc | 154 / 225 | `e7bcc88b0ba426a0` | the (a)+(b)+(c) set above, nothing else |
| `s0rep` off | 1 | `adfa7d4e41a7dda9` (== baseline off; `reality=data`) | none |
| `tfull` off | 1 | `393be7a1ed4b5d50` | the (a)+(b)+(c) set, nothing else |
| `cs` cv / nuecc / off | 154 / 225 / 1 | `c5f5312fbb87b55d` / same / `fbd239a0e6c8d427` | the sampler swap + `ImproveCluster_2:pr.samplers`, nothing else |
| `csp3bw` cv / nuecc / off | 154 / 225 / 1 | `6089ecad115d3197` / same / `972231b088a80b66` | sampler swap + the three Steiner keys, nothing else |
| `p3bw` cv / nuecc / off | 154 / 225 / 1 | `872186ed00979a25` / same / `9d13714a93fcbbf6` | the three Steiner keys on both `CreateSteinerGraph` nodes, nothing else |

One hash per (cell, reality) across every sub-root and every event (`products/d116/*/summary.txt`
counts `op_config_sha256` per event: exactly one value per arm).

Tripwire: `prod_cfg_gate.py --ref ref/prod-2026-09-17b` **PASS 21/21** before the campaign
(`~/tmp/d116-tripwire-before.txt`) and after (sec 12).

**Pre-registration.** `docs/116_figs/116_pred.txt`, sha256
`22e42c09d5728bf5b82b3010f930f8aea04bf1d7cf9b72d2d3ddb508d35d69ff`, written 07:06 before the first
arm (07:06:39): the ten primary metrics M1–M10, the noise floor, and the verdict rule —
IMPROVED / DEGRADED iff |Δ| > 2 × floor (> 0.5 pt when the floor is 0) **and** the exact two-sided
sign test on the paired exchange gives p < 0.05; otherwise NOT SEPARABLE, never "unchanged".

## 3. What ran

`scripts/d116/stageB_cell.sh` is a fork by duplication of `scripts/d115/stageB.sh` with the cell's
TLA file, the `.tfjson` hook, a `.d116_cell` marker (an existing out_root of another cell is
refused — M13), an atomic per-sub-root claim so several workers can share a cell (the sub-roots are
6–21 events each; one sequential loop left the box at load 15), and the pin md5 printed at start and
end. Three to eight workers per cell at `PR_JOBS=6`, two cells at a time, load 25–40 on 64 cores
(the beam-off root is one lock, so it runs on one worker).

| arm | events | rc ≠ 0 | wall (5 workers) | `op_config_sha256` | pin md5 start = end |
|---|---:|---:|---:|---|---|
| `s0rep` cv | 2017/2017 | 0 | 1 970 s | `399badee…` × 2017 | `71ebd5ae…` |
| `s0rep` nuecc | 2001/2001 | 0 | 2 690 s | `399badee…` × 2001 | `71ebd5ae…` |
| `tfull` cv | 2017/2017 | 0 | 2 030 s | `e7bcc88b…` × 2017 | `71ebd5ae…` |
| `tfull` nuecc | 2001/2001 | 0 | 3 160 s | `e7bcc88b…` × 2001 | `71ebd5ae…` |
| `s0rep` off | 1000/1000 | 0 | 1 814 s (one root, one worker) | `adfa7d4e…` × 1000 | `71ebd5ae…` |
| `tfull` off | 1000/1000 | 0 | 3 317 s (one root, one worker) | `393be7a1…` × 1000 | `71ebd5ae…` |
| `cs` cv / nuecc / off | 2017 / 2001 / 1000 | 0 / 0 / 0 | 2 538 / 3 545 / 2 217 s | `c5f5312f…` / same / `fbd239a0…` | `71ebd5ae…` |
| `csp3bw` cv / nuecc / off | 2017 / 2001 / 1000 | 0 / 0 / 0 | 2 529 / 3 199 / 2 377 s | `6089ecad…` / same / `972231b0…` | `71ebd5ae…` |
| `p3bw` cv / nuecc / off | 2017 / 2001 / 1000 | 0 / 0 / 0 | 2 011 / 2 096 / 1 317 s | `872186ed…` / same / `9d13714a…` | `71ebd5ae…` |

Two sub-roots (`csp3bw` nuecc `f007`, `cs` nuecc `f008`) were re-run in full after their worker was
stopped to bring the load back under the 32-CPU licence; the re-run is the same launcher on the one
sub-root, and the arm-level hash and `rc` checks above cover it.

Completeness is judged off the products (every stage-A `ql_evt` has a `pr_evt` with `rc=0`), never
off a runner verdict. `Trun.toolkit_git` = `0a2807f4` on every event of every arm. The flash-time
association (doc 115 V11) holds on every MC arm: `tfull` nuecc median offset −0.0005 µs from the
doc-108 reference, 99.1 % within 0.1 µs (`docs/116_time/`).

## 4. The noise floor is zero

`s0rep` — the production configuration re-run on the same pctrees with the same binary — reproduces
the doc-115 baseline **row for row**: `products/d116/cv-s0rep/candidates.tsv` (968 rows) and
`nuecc-s0rep/candidates.tsv` (1 925 rows), and both `truth.tsv`, differ from `products/d115/` in
**0** rows, and every metric M1–M8 and S1–S5 has exchange −0/+0. CLAUDE.md M4 says the DL/SCN vertex is not bit-stable; on this pin, this box and
these inputs it was, to the last candidate. Consequence for the rule: the floor is 0, so a delta
needs |Δ| > 0.5 pt **and** p < 0.05 to be called.

## 5. The headline: `tfull` against the baseline, paired per interaction

`docs/116_figs/116_compare.txt` (`d116_compare.py`). "−lost/+gain" is the number of signal
interactions selected in the baseline only / in the cell only; for a purity, the events where the
cell has more / fewer selected background candidates; for the vertex, the interactions matched in one
arm only. p is the exact two-sided sign test on that exchange.

| id | metric | baseline | `tfull` | Δ (pt) | −lost/+gain | p | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| M1 | νμCC efficiency, cv (numu > 0.9) | 387/557 = 69.5 % | 390/557 = 70.0 % | +0.54 | −39/+42 | 0.82 | not separable |
| M2 | νμCC in-sample purity, cv | 387/448 = 86.4 % | 390/450 = 86.7 % | +0.28 | −29/+30 | 1.00 | not separable |
| M3 | νeCC efficiency, nuecc (nue > 7) | 618/1511 = 40.9 % | 630/1511 = 41.7 % | +0.79 | −168/+180 | 0.56 | not separable |
| M4 | νeCC in-sample purity, > 7 | 618/632 = 97.8 % | 630/649 = 97.1 % | −0.71 | −14/+9 | 0.41 | not separable |
| M5 | νeCC efficiency, nuecc (nue > 4) | 734/1511 = 48.6 % | 757/1511 = 50.1 % | +1.52 | −172/+195 | 0.25 | not separable |
| M6 | νeCC in-sample purity, > 4 | 734/781 = 94.0 % | 757/803 = 94.3 % | +0.29 | −31/+32 | 1.00 | not separable |
| M7 | vertex < 5 cm, all true ν in FV, cv | 539/783 = 68.8 % | 533/783 = 68.1 % | −0.77 | −42/+36 | 0.57 | not separable |
| M8 | vertex < 5 cm, all true ν in FV, nuecc | 1122/1626 = 69.0 % | 1133/1626 = 69.7 % | +0.68 | −147/+158 | 0.57 | not separable |

Secondary (reported, not graded): νμCC efficiency on the nuecc arm 47/90 → 47/90 (−6/+6); νeCC
on the cv arm 4/5 → 4/5; vertex < 5 cm for true νμCC in cv 463 → 457 of 557 (−30/+24, p 0.50), for
NC in cv 71 → 71 of 221 (−12/+12), for νeCC in nuecc 1067 → 1077 of 1511 (−141/+151, p 0.60).

**Reading.** Not one primary metric moves by more than its exchange. The exchanges are large: on
the νeCC selection 11–13 % of the signal swaps membership (168 lost + 180 gained on 618 selected),
on the νμCC selection 10 % (39 + 42 on 387), on the νe vertex 13 % of the matched set (147 + 158
on 1122). This is doc pr/150 sec 8 on data — nueCC 36 → 42, numuCC 789 → 791, every net change
inside its exchange — now with truth deciding who is signal, and the same verdict. Every
"improvement" in the table (+0.5 to +1.5 pt) is one to two sigma of its own exchange; every
"degradation" (−0.7 / −0.8 pt) likewise. The doc-115 vertex tables by class and Edep bin
(`docs/116_vtx/d115_vtx_{cv,nuecc}-tfull.txt`) show the same picture inside every bin, the largest
bin move being 100 ≤ Edep < 300 MeV on nuecc, 34 → 30 of 67.

## 6. The trajectory itself is better on MC, as it was on data

`scripts/d116/traj_eval.py` reuses doc pr/150's instrument (`pr150_traj_eval.py`, imported, not
copied; keyed here by run/subrun/event because MC event numbers repeat across files) on the
main-cluster bundle of every event with a calib dump: the PR fit rows against the measured 2-D
cells (`T_proj_data`) and the 3-D image (the Bee clustering layer). Row-weighted means over the
events with a fit in both arms; better/worse counts at |d| > 0.005; exact sign test.

| clause (lower is better unless noted) | cv baseline | cv `tfull` | better / worse | p | nuecc baseline | nuecc `tfull` | better / worse | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2D_W — live rows > 1 wire off the W-cells on their slice | 0.0081 | **0.0054** | 184 / 74 | 6e−12 | 0.0213 | **0.0144** | 923 / 212 | 4e−106 |
| R2D_U | 0.0110 | **0.0076** | 217 / 95 | 4e−12 | 0.0299 | **0.0197** | 1044 / 206 | 4e−135 |
| R2D_V | 0.0142 | **0.0096** | 256 / 92 | 5e−19 | 0.0377 | **0.0257** | 1184 / 207 | 2e−166 |
| P2 — rows off-charge in ≥ 1 live plane | 0.0196 | **0.0138** | 307 / 98 | 3e−26 | 0.0439 | **0.0296** | 1239 / 190 | 9e−189 |
| P1 — rows > 1 cm from the image ridge | 0.0323 | **0.0285** | 267 / 183 | 9e−5 | 0.0952 | **0.0897** | 878 / 633 | 3e−10 |
| P4 — image charge within 1.5 cm of the fit (higher is better) | 0.8199 | **0.8222** | 254 / 180 | 4e−4 | 0.6691 | **0.6771** | 953 / 604 | 8e−19 |
| uncov — measured charge the fit predicts < 10 % of | 0.2171 | **0.2092** | 361 / 205 | 6e−11 | 0.2897 | **0.2693** | 1093 / 500 | 6e−51 |
| qneg — rows with q < 0 | 0.0343 | **0.0306** | 354 / 190 | 2e−12 | 0.0821 | **0.0754** | 956 / 562 | 4e−24 |
| wig1 — 3-row wiggle > 1 cm | 0.1032 | 0.1053 | 272 / 320 | 0.05 | 0.2400 | 0.2408 | 741 / 799 | 0.15 |
| med_d_W — median residual, wires | 0.2383 | 0.2426 | 271 / 465 | 8e−13 | 0.2408 | 0.2459 | 528 / 969 | 2e−30 |

Events with a fit: cv 925 → 928 (906 common), nuecc 1861 → 1858 (1845 common). The fit closes better
on every plane, is off-charge less often, covers more of the image charge and predicts more of the
measured charge, on both samples and more strongly on the shower-rich νe sample. The two clauses
that do not improve are the same two as on data: the 3-row wiggle is flat and the median
in-cell residual grows by 0.004–0.005 wire (the rows that were already inside a cell sit
marginally less centrally). Doc pr/150 sec 3's reading transfers to MC unchanged: **a better
trajectory that the downstream PR re-decides on.**

## 7. Where the churn comes from — the movers, adjudicated by truth

`d116_movers.py` (`docs/116_figs/116_movers_<sample>-tfull.{txt,tsv}`) takes every interaction
the two arms disagree on and reads the arm that missed it. For a vertex mover — true ν in FV whose
nearest candidate vertex is within 5 cm in one arm only — the calib dump of the missing arm
(`main_vertex`, the `vertices` list with its `main_candidate` flags, `vertex_scoreboard`) gives
doc pr/150's classes without a scan: **CHOICE** (a main-candidate vertex within 5 cm of the truth
exists; the chooser took another), **STRUCTURE-near** (a vertex within 5 cm exists in the
segmentation but is not a main candidate), **STRUCTURE** (no vertex of the segmentation within
5 cm), **no candidate** (no `T_tagger` row at all).

### 7.1 Vertex movers

| sample, direction | n | CHOICE | STRUCTURE-near | STRUCTURE | no candidate | DL route same in both arms |
|---|---:|---:|---:|---:|---:|---:|
| cv, lost (matched in baseline only) | 42 | **30** | 1 | 8 | 3 | 32 (dual-snap-accept → dual-snap-accept) |
| cv, gained (matched in `tfull` only) | 36 | **22** | 3 | 10 | 1 | 25 |
| nuecc, lost | 147 | **99** | 28 | 20 | 0 | 108 |
| nuecc, gained | 158 | **102** | 25 | 30 | 1 | 109 |

The single-knob cells read the same (`116_movers_*-{cs,p3bw,csp3bw}.txt`): on nuecc CHOICE is
95 ↔ 106 (`cs`), 96 ↔ 97 (`p3bw`), 105 ↔ 89 (`csp3bw`) of 141–164 lost / 145–152 gained; on cv
35 ↔ 31, 24 ↔ 22, 37 ↔ 27 of 43–52 / 37–42.

Every class is mirrored: CHOICE 30 ↔ 22 and 99 ↔ 102, STRUCTURE 8 ↔ 10 and 20 ↔ 30. By true class
(cv: νμCC 30 lost / 24 gained, NC 12 / 12; nuecc: νeCC 141 / 151) and by Edep bin the same. The
cathode band |x| < 15 cm holds 3 / 3 and 8 / 11 of them — the doc-115 open item about the ±1.5 cm
CPA face on MC is not what moves here. Where the missing arm's nearest candidate ended up is also
mirrored (nuecc lost → 5–20 cm 57, 20–50 cm 54, > 50 cm 36; gained from 70 / 55 / 32).

Two thirds of the movers, in both directions, on both samples, are the vertex **chooser** picking
between candidates it has in both arms — 73 % of the pairs went through the DL dual-snap-accept
route in both arms, so this is the DL re-rank scoring near-equal candidates differently once the
segments under them move. The fraction is what doc pr/150 sec 4 estimated from the blind scan
("CHOICE ~ half"); with truth it is higher, and it is symmetric. STRUCTURE (the segmentation no
longer has a vertex near the truth) is the smaller, also symmetric, remainder.

### 7.2 Selection movers — the first failing stage in the other arm

| selection | direction | n | no candidate | not matched (< 5 cm) | reco vertex outside FV | score below cut | score Δ on those (median) | reco_Enu moved > 25 MeV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| νμCC > 0.9, cv | lost | 39 | 2 | 22 | 0 | 15 | −1.43 | 14/15 |
| | gained | 42 | 1 | 18 | 0 | 23 | −1.49 | 16/23 |
| νeCC > 7, nuecc | lost | 168 | 0 | 68 | 0 | **100** | **−6.7** | 77/100 |
| | gained | 180 | 0 | 74 | 0 | **106** | **−7.9** | 90/106 |
| νeCC > 4, nuecc | lost | 172 | 0 | 82 | 0 | 90 | −19.5 | 77/90 |
| | gained | 195 | 0 | 92 | 0 | 103 | −14.1 | 94/103 |

Roughly 40 % of the νe selection churn is the vertex (7.1) and 60 % is the BDT: the same interaction,
matched and in the FV in both arms, scores above 7 in one and a median 7 points below in the other.
On those score-stage movers **every** tagger sub-score in the calib `tagger` block changed —
`numu_1_score`, `br3_3`, `tro_1`, `tro_4`, `pio_2`, `sig_2`, `lol_1`, `stw_3`, `br3_5`, `br3_6` in
≥ 88 % of them — and the reconstructed neutrino energy moved by more than 25 MeV in three quarters.
The trajectory re-segments the candidate, every kinematic input of the uBooNE-trained BDTs
(doc 107 sec 6) re-evaluates, and the score crosses the cut in both directions with no preferred
sign. The `numu_1_score` family is the one that changes on all 15 + 23 νμCC score movers too.

## 8. Beam-off and the exposure-weighted purities

The 1 000 Run-1 off-beam gates re-run through the same cells (`reality=data`, the doc-115 beam-gate
census as the denominator; `docs/116_off/<cell>/d115_beamoff_rate.txt`), paired by gate:

| cell | gates with ≥ 1 candidate | … reco vertex in FV | … numu > 0.9 (SELECTED) | … nue > 7 / > 4 | vs baseline (gates −lost/+gained) |
|---|---:|---:|---:|---:|---|
| baseline (doc 115) | 87 | 25 | **5** = 0.50 % [0.32, 0.78] | 0 / 0 | — |
| `s0rep` | 87 | 25 | 5 | 0 / 0 | −0/+0 (row-identical, 0 of 89 candidate rows differ) |
| `tfull` | 85 | 22 | **4** = 0.40 % [0.24, 0.66] | 0 / 0 | −1/+0 |
| `cs` | 88 | 22 | 4 | 0 / 0 | −3/+2 |
| `csp3bw` | 87 | 23 | 4 | 0 / 0 | −2/+1 |
| `p3bw` | 86 | 17 | **2** = 0.20 % [0.10, 0.40] | 0 / 0 | −4/+1 |

Fewer gates pass the νμ selection in every knob cell, each time through a different exchange
(`cs` −3/+2, `csp3bw` −2/+1, `tfull` −1/+0, `p3bw` −4/+1); the pre-registration counts a beam-off
change only from 5 gates, and these are rates whose 68 % intervals overlap the baseline's, not
verdicts. `p3bw`'s 5 → 2 is the largest and is one cell of five on five gates; it is the direction a
cosmic-only selection should move if the priced seed makes fewer through-going tracks look like
neutrino candidates, and a later round with the off-beam gate count `f` in hand can score it.
Neither νe cut admits a cosmic gate in any cell.

Doc 115 sec 13's exposure-weighted numbers, recomputed per cell from the cell's own counts with the
same constants (`scripts/d115/normalisation.py`: `w` = 2.7558e−3, 22 325 beam gates, `f` = 1 —
still the missing input):

| cell | νμCC purity, in-sample → f = 1 (cosmic-only events at 22 325 gates) | νeCC > 7 common-exposure purity | νeCC > 4 |
|---|---|---:|---:|
| baseline | 86.4 → 69.2 % (111.6) | 62.1 % (S 1.70, cv background 1) | 39.3 % |
| `s0rep` | 86.4 → 69.2 % (111.6) | 62.1 % | 39.3 % |
| `tfull` | 86.7 → 72.3 % (89.3) | 62.3 % (S 1.74, cv background 1) | 40.0 % |
| `cs` | 87.1 → 72.5 % (89.3) | 96.5 % (S 1.67, cv background **0**) | 39.0 % (background 3) |
| `csp3bw` | 85.3 → 71.0 % (89.3) | 96.6 % (S 1.74, cv background **0**) | 33.1 % (background 4) |
| `p3bw` | 87.4 → 79.3 % (44.7) | 62.6 % (S 1.75, cv background 1) | 39.5 % (background 3) |

The 3 pt rise in the f = 1 νμCC purity is one beam-off gate scaled by 22.3; it carries the same
one-gate uncertainty and is quoted for completeness, not as a result. The νeCC common-exposure
purity at > 7 swings from 62 % to 96 % between cells because the mc-cv non-νe background is **one
event** in the baseline and zero in `cs` / `csp3bw` — doc 115 sec 13.2's point that this round does
not measure it, illustrated. At > 4 it is three or four events and moves 33–40 %.

## 9. Decomposition by knob: `cs`, `p3bw`, `csp3bw`

The same paired comparison on the single-knob cells (`docs/116_figs/116_compare.txt`), so the
question "which of (a), (b), (c) does what" has an answer per metric. Δ in percentage points,
then −lost/+gained and the sign-test p; every entry below is **not separable** by the rule.

| id | metric | baseline | `cs` (a) | `p3bw` (b) | `csp3bw` (a)+(b) | `tfull` (a)+(b)+(c) |
|---|---|---:|---:|---:|---:|---:|
| M1 | νμCC eff, cv | 69.5 % | 69.1 (−0.4; −43/+41, 0.91) | 68.6 (−0.9; −39/+34, 0.64) | 67.9 (−1.6; −45/+36, 0.37) | 70.0 (+0.5; −39/+42, 0.82) |
| M2 | νμCC purity, cv | 86.4 % | 87.1 (+0.7; −24/+28, 0.68) | 87.4 (+1.0; −27/+33, 0.52) | 85.3 (−1.1; −33/+29, 0.70) | 86.7 (+0.3; −29/+30, 1.00) |
| M3 | νeCC eff > 7 | 40.9 % | 40.1 (−0.8; −181/+169, 0.56) | 42.1 (+1.2; −165/+183, 0.36) | 41.8 (+0.9; −159/+173, 0.48) | 41.7 (+0.8; −168/+180, 0.56) |
| M4 | νeCC purity > 7 | 97.8 % | 96.5 (−1.3; −16/+8, 0.15) | 97.4 (−0.4; −14/+11, 0.69) | 96.6 (−1.2; −15/+7, 0.13) | 97.1 (−0.7; −14/+9, 0.41) |
| M5 | νeCC eff > 4 | 48.6 % | 48.6 (+0.1; −187/+188, 1.00) | 49.3 (+0.7; −178/+189, 0.60) | 49.6 (+1.0; −175/+190, 0.46) | 50.1 (+1.5; −172/+195, 0.25) |
| M6 | νeCC purity > 4 | 94.0 % | 92.5 (−1.5; −45/+32, 0.17) | 93.6 (−0.4; −36/+32, 0.72) | 92.2 (−1.7; −45/+29, **0.08**) | 94.3 (+0.3; −31/+32, 1.00) |
| M7 | vertex < 5 cm, cv | 68.8 % | 68.2 (−0.6; −47/+42, 0.67) | 68.1 (−0.8; −43/+37, 0.58) | 67.6 (−1.3; −52/+42, 0.35) | 68.1 (−0.8; −42/+36, 0.57) |
| M8 | vertex < 5 cm, nuecc | 69.0 % | 68.3 (−0.7; −164/+152, 0.54) | 69.7 (+0.7; −141/+152, 0.56) | 68.8 (−0.2; −148/+145, 0.91) | 69.7 (+0.7; −147/+158, 0.57) |

Three readings:

1. **The exchange is the same size in every cell** — 40–50 νμCC and 150–190 νeCC interactions
   each way — whichever knob is on. It is not additive: `cs` and `p3bw` each churn as much as
   both together. Any perturbation of the trajectory re-decides the same population; the three
   knobs are three ways of perturbing it.
2. **The one consistent sign is the νeCC in-sample purity**, down 0.7–1.7 pt in every knob cell,
   with `csp3bw` at > 4 the closest any entry comes to the bar (−45/+29, p 0.08). Sec 10 reads
   what that background is.
3. **The fit keys (c) are not neutral on the selection.** `csp3bw` → `tfull` moves νμCC efficiency
   −1.6 → +0.5 pt and νeCC purity > 4 −1.7 → +0.3 pt; both are inside the exchange, so this is
   the churn re-rolling, not a recovery — but it is why `tfull`, the cell that matters, sits closest
   to the baseline on every primary metric.

The trajectory-closure clauses decompose the same way on both samples (sec 6's instrument;
`traj/`, row-weighted means, every entry with sign-test p < 0.05 unless marked):

| clause | baseline | `cs` (a) | `p3bw` (b) | `csp3bw` (a)+(b) | `tfull` (a)+(b)+(c) |
|---|---:|---:|---:|---:|---:|
| R2D_W, cv | 0.81 % | 0.70 | 0.67 | 0.57 | **0.54** |
| R2D_W, nuecc | 2.13 % | 1.93 | 1.80 | 1.54 | **1.44** |
| P2 off-charge rows, nuecc | 4.39 % | 3.83 | 3.92 | 3.26 | **2.96** |
| uncovered charge, nuecc | 29.0 % | 27.6 | 28.5 | 26.9 | **26.9** |
| uncovered charge, cv | 21.7 % | 21.2 | 21.7 (n.s.) | 21.0 | **20.9** |
| med_d_W, nuecc (wires) | 0.2408 | 0.2397 | 0.2398 | **0.2381** | 0.2459 |

The two seed-side knobs each buy about a third of the W-plane gain and add; the priced Steiner seed
(b) does slightly more for the wire residual, the sampler (a) does more for the uncovered charge
(the `charge_stepped` points carry the charge the fit then predicts), and the fit keys (c) add the
last step on every clause but one: they are the only change that moves the median in-cell residual
the wrong way (nuecc 0.2381 → 0.2459 wire from `csp3bw` to `tfull`), the same trade doc pdvd/101
recorded when they were introduced.

## 10. Degradation, origin, and what "improve" would take

The pre-registered rule finds **nothing DEGRADED** on `tfull` or `csp3bw`: the negative deltas
(M4 −0.7 pt on 14 vs 9 background events, M7 −0.8 pt on 42 vs 36 movers on `tfull`; M6 −1.7 pt on
45 vs 29 on `csp3bw`) are inside their exchange with p 0.41, 0.57 and 0.08. So no retune rung is
triggered, and none was run — running one on a not-separable baseline would be tuning against
noise, which the rule was frozen to prevent.

The one sign that repeats across cells deserves its origin anyway. The νeCC in-sample purity falls
0.7–1.7 pt in every knob cell (sec 9), and `docs/116_sel*/nuecc-<cell>/d107_selection.txt` says
what the added background is: **not** a new class. At > 4 the "true vertex outside FV" background
is 13 in the baseline and 9–12 in every cell; the "no true vertex within 5 cm" background goes
34 → 37 (`tfull`), 48 (`cs`), 52 (`csp3bw`), and in every one of those the nearest true
interaction is a **νeCC at 5–50 cm** (baseline 22 + 10 + 2 at 5–20 / 20–50 / > 50 cm; `csp3bw`
28 + 19 + 5). At > 7 the same: 3 → 11 / 12 / 14, all νeCC at 5–50 cm. These are signal events the
νe BDT still accepts whose vertex the chooser put 5–50 cm off — the CHOICE mechanism of sec 7.1
seen from the purity side, where the doc-107 definition counts a misplaced-vertex signal as
background. The physics content of the selection (a νe interaction was found and tagged) is
unchanged; what moved is where the vertex was put, and that is what sec 10.2 says the chooser
re-decides.

What the round establishes is the **origin of the churn** (sec 7), and that decides what an
improvement would have to be:

1. **It is not a parameter of the trajectory.** The trajectory is strictly better on every closure
   clause (sec 6) and the churn is symmetric in every class and bin (sec 7). Doc pr/150 sec 5
   already showed on data that terminal thinning, the sampler charge cut and the DL admission
   score do not move it (the last one makes the vertex worse). Nothing in this round contradicts
   that, and nothing in it motivates spending an arm on the same ladder.
2. **It is the vertex chooser and the BDT inputs re-deciding on a re-segmented candidate.** CHOICE
   movers are, by definition, events where the right vertex is still on the candidate list; the DL
   re-rank scores it differently. The BDT movers are events whose features moved by more than the
   score margin. Neither is a knob today: the DL vertex weights (`uboone/scn_vtx/…`) and the BDT
   weights (`uboone/weights`) are uBooNE-trained (doc 107 sec 6, doc 115 sec 10), and the SBND
   operating point sits on their uncalibrated scores.
3. **What this round delivers toward that.** `116_figs/116_movers_nuecc-tfull.tsv` lists the
   99 + 102 CHOICE movers, 28 + 25 STRUCTURE-near, 20 + 30 STRUCTURE and the 100 + 106 score-stage
   movers with their true vertex, class, Edep, both arms' nearest-candidate distance, the DL route
   in both arms and the score / energy deltas. That is the validation set a re-tune of the
   vertex-choice stage (DL re-rank features, candidate scoring in `TaggerCheckNeutrino`) against
   the new trajectory needs — doc pr/150 sec 5's recommended next round — and it comes with truth,
   which the 108-mover blind scan could not give. The same file is the training-set seed for an
   SBND-tuned BDT operating point, whichever trajectory is chosen.

**Recommendation.** Not to flip on this evidence. The PDHD/PDVD trajectory is a better trajectory on
SBND MC and buys nothing at the selection level while the vertex chooser and the BDT inputs are the
uBooNE ones; flipping would exchange ~11 % of the νeCC signal membership for no net gain and
re-open every downstream sentinel (doc pr/150 sec 2.3) without a compensating number. Re-tune the
vertex-choice stage against the new trajectory first, on the mover set above, and re-score with
`d116_compare.py` — the whole chain here is a two-command re-run.

## 11. What is and is not comparable to doc pr/150

Same cells, same TLAs, same fit JSON; different binary (pr/150 pinned `f9665bea`, this round the
doc-115 pin `0a2807f4` — one cfg-only and one default-OFF-knob commit later, and the same binary as
the baseline being compared to); different stage A (doc 102's data arms vs doc 115's MC arms);
different truth (hand clicks on 823 labelled events vs GENIE on 2 409 in-FV interactions). The
readings agree on every point the two can share: the closure clauses improve by the same factors
(R2D_W 0.97 → 0.68 % on data, 0.81 → 0.54 % / 2.13 → 1.44 % here), the selections churn without a
net sign (nueCC 36 → 42 on data, 618 → 630 here), the vertex movers are CHOICE-dominated (half by
scan, two thirds by truth). The one number that does not transfer is the data vertex loss itself
(677 → 620 of 823 ≤ 3 cm): on MC, at 5 cm, the vertex is −0.8 pt on cv and +0.7 pt on nuecc,
neither separable. Doc pr/150 sec 4.2 already suspected the click-anchored 3 cm criterion of
penalising any refit (189 of 823 clicks lie within 0.05 cm of the `s0` vertex); the truth-based
5 cm criterion here does not see the asymmetry.

## 12. Open items

1. The vertex-choice retune against the new trajectory (sec 10.3) — the only route to a net gain.
2. An SBND-tuned BDT operating point: the score-stage movers show the uBooNE BDTs re-deciding on
   feature changes that are within their own resolution on SBND.
3. The doc-115 open items carry: the off-beam gate count `f` behind the exposure-weighted νμCC
   purity; the νeCC purity needs the full CV production; the ±1.5 cm data CPA face in `dvm()` is
   applied to MC (not what moves here, sec 7.1, but still unconditional).
4. `s0rep` reproduced the baseline to the row on this pin. That is a property of this pin and
   box, not a guarantee; a future round should keep its own `s0rep`.

## 13. Files

`scripts/d116/stageB_cell.sh`, `scripts/d116/analyze_cell.sh`, `scripts/d116/cfg_proof.sh`,
`scripts/d116/traj_eval.py`, `d116_compare.py`, `d116_movers.py`; `docs/116_figs/tla/`,
`116_pred.txt` + `.sha256`, `116_cfg_proof.txt`, `116_compare.{txt,tsv,md}`, `116_movers_*.{txt,tsv}`,
`traj/*.tsv`; `docs/116_{sel,sel_edep100,vtx,scan,time,off}/`; `products/d116/<sample>-<cell>/`.
Nothing under `docs/115_*`, `products/d115/`, `scripts/d115/`, the doc-115 or pr/149–150 records,
or any toolkit file is modified.

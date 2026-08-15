# doc pr/77 — DL neutrino-vertex fine-tuning infrastructure

Status: **infrastructure round, shipped** (2026-08-14); **round 2 (S8e)
executed same day** — 8a/8b/8c strategies run on 247+ labels incl. the
numu50 data anchor.  Python-only — no toolkit C++ or config change, no A/B
burden, nothing flipped.  Practice training on the finished hand-scan tags;
the serious campaign waits for the live 1000-event data scan.

Owner request: prepare the training/fine-tuning infrastructure for the DL
vertex per doc pr/52, using `sbnd_xin/dl_vtx_training/` as the software
base; practice on the scanned 47 nueCC + 19 NCpi0 events; data augmentation
(X→−X, Y→−Y reflections and more); net first, selection fine-tune second.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/dl_vtx_training
# parity of the rebuilt net input vs production scoreboard (66 events)
python3 parity_check.py --tsv runs/parity-practice66.tsv
# frozen snapshot from the two finished scan tags
python3 build_dataset.py --name practice66 --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0
# 6-fold practice fine-tune + out-of-fold evaluation
python3 train.py --data data/practice66 --name ft0 --kfold 6 --epochs 30
python3 evaluate.py --data data/practice66 --run runs/ft0 --tsv runs/ft0/eval.tsv
# selection-stage replay: closure + operating-point grid
python3 rerank_replay.py --closure
python3 rerank_replay.py --grid --tsv runs/rerank-grid-practice66.tsv
# Tier B charge-feature check
python3 qfeature_check.py --png runs/qfeature-practice66.png
```

Environment: wcgpu1 direnv python 3.11.9, torch 2.5.1+cu121,
`sparseconvnet` from `toolkit-dev/SparseConvNet`, RTX 4090 (CPU also works;
production inference is CPU).

## 1. What was built

`sbnd_xin/dl_vtx_training/` (committed; `data/` + `runs/` outputs are not):

| file | role |
|---|---|
| `scn_vtx/model.py` | `DeepVtx` **verbatim** from toolkit `pyutil/python/SCN/DeepVtx.py` + `load_weights()` replicating the in-tree dim=1 squeeze |
| `scn_vtx/voxelize.py` | `voxelize()` **verbatim** from `SCN_Vertex.py:29-52` + voxel↔cm + Gaussian truth target |
| `scn_vtx/io.py` | rebuild the net input cloud from `calib-pr-evt*.json`; load pr/75 labels (read-only, M13) |
| `build_dataset.py` | labels + calib → frozen `data/<name>/` snapshot (npz + manifest with label mtimes; refuses overwrite) |
| `augment.py` / `dataset.py` | augmentation menu + per-epoch sample views |
| `train.py` | fine-tune loop resuming the uBooNE t48k conventions |
| `evaluate.py` | out-of-fold baseline-vs-tuned metrics incl. candidate-snap + do-no-harm guard |
| `parity_check.py` | rebuilt-cloud CP24 top-5 vs recorded `vertex_scoreboard.voxels[]` |
| `rerank_replay.py` | offline replay of the 7-term rerank acceptance (pr/52 §5.2a/b grid) |
| `qfeature_check.py` | pr/52 Tier B charge-input distribution |

## 2. Ground facts the pipeline is built on

- **Inference input** (`NeutrinoVertexFinder.cxx:4147-4179`): every PR-graph
  vertex fit point (or `wcpt()` with dQ=0 if the fit is invalid) + every
  segment **interior** fit point (`i=1..n-2`), cm, float32,
  `q = dQ*dQdx_scale + dQdx_offset` (= `dQ*0.1 − 1000`), whole event.
  Voxelized at 0.5 cm after per-axis min subtraction (`SCN_Vertex.py:29-52`,
  translation-invariant); voxel→cm centre `idx*0.5 + offset + 0.25`; score
  = `sigmoid(c1) − sigmoid(c0)`.
- **The weights are checkpoint 24 of the public uBooNE campaign**
  (github.com/HaiwangYu/uboone-dl-vtx, `t48k/m16-l5-lr5d-res0.5/CP24.pth`):
  truth Gaussian `exp(−(d/σ)²/2)`, loss `MSE(pred1−pred0, truth)`, Adam
  `lr0=1e-5`, decay `lr0·exp(−0.05·epoch)`, per-event batches.  The
  fine-tune resumes these conventions (`lr0=1e-6` default).
- **Checkpoint compatibility**: the uBooNE file stores conv kernels 4-D
  `(27,1,Cin,Cout)`; the in-tree loader squeezes dim=1 on all of them
  (normal path).  Checkpoints saved here are squeezed-3-D; **round-trip
  proven**: a `train.py` checkpoint loaded through the untouched in-tree
  `SCN_Vertex._load_model` gives bit-identical scores (max |Δs| = 0.0 on
  evt388, 822 voxels).  Deployment = new file in `wire-cell-data` (never
  overwrite the uBooNE one) + the `dl_weights` TLA.
- **Labels** (pr/75): rank-1 pick = truth. `vtxscan-prod0813` 47 nueCC +
  `vtxscan-prod0813-ncpi0` 19 NCpi0 (complete);
  15/66 corrective (scanner moved the vertex; nueCC: 30504/100.8 cm,
  163543/58.6, 111412/48.6, 271851/31.9, 38856/30.1, 122660/5.0 + 4 sub-3 cm;
  NCpi0: 259542/136.7, 180801/94.8, 359980/56.5, 56982/33.3, 463565/30.2),
  51/66 confirming — also valid labels, and the do-no-harm bar.
  `confidence` is null throughout both tags (nothing to weight on);
  0 manual picks / 0 `not_a_candidate`.  `vtxscan-prod0813-mcp1k` is
  **live** (87 labels and counting) — the serious-round training set.

## 3. Parity: can we train off the calib dump? (yes, with flags)

The calib JSON is written **after** `snap_main_vertex_to_kink` +
`improve_vertex` + shower reclustering, i.e. the post-DL-refit graph, while
the net saw the pre-refit graph.  `parity_check.py` measures the damage:
CP24 on the rebuilt cloud vs the recorded `vertex_scoreboard.voxels[]`
(`runs/parity-practice66.tsv`):

- top-1 voxel deviation `d1`: p50 = **0.500 cm** (one voxel), p90 = 2.55 cm,
  max = 23.4 cm; **39/66 voxel-exact with |Δscore|<0.05**.
- the tail is concentrated on corrective events (worst: evt 56982, a 33 cm
  scanner correction) — exactly where the refit moved the graph most.

Verdict: good enough for the practice round; deviant events are visible in
the TSV and can be excluded or reweighted.  If the serious round wants
byte-exact inputs, the fix is the deferred dump knob (§9), not silent
training.

## 4. Augmentation menu (owner's request + additions)

- **Reflections** X→−X, Y→−Y, both (×4, deterministic; applied to cloud +
  truth before voxelization — min-subtraction renormalizes).  **No Z flip**:
  beam-direction topology is physics, not a symmetry.  No rotations
  (wire-plane geometry).
- **Sub-voxel jitter** (novel, free): shift cloud+truth by a random fraction
  of the 0.5 cm pitch per axis.  A translation is a no-op after
  min-subtraction *except* for the voxel-boundary phase ⇒ a genuinely new
  voxelization of the same event, fresh every epoch.
- **Charge jitter**: global N(1,5%) × per-point N(1,2%) on the raw-dQ scale
  (applied to `q − offset`, since the feature is affine).
- **Point dropout** (default off).

## 5. Practice run (pipeline shakedown, NOT a physics result)

15 corrective labels cannot support a physics conclusion; this run
demonstrates the machinery end-to-end.  Run `ft0cpu`: 6-fold CV stratified
by corrective/confirming, ×4 reflections + sub-voxel jitter + charge
jitter, `--freeze head` (linear + final BN + last UNet decoder block ≈ 7k
of 7.2M params), 30 epochs, lr0=1e-6.  (Practical note: this net is too
small for GPU to win — CPU was ~5× faster per epoch than a 4090; smoke runs
on both give identical losses at the same seed.)

Out-of-fold `evaluate.py` (`runs/ft0cpu/eval.tsv`), d_argmax in cm:

| slice | baseline (CP24) p50 / p90 / max | tuned p50 / p90 / max |
|---|---|---|
| all (66) | 0.69 / 40.8 / 139.5 | 0.61 / 61.8 / 426.9 |
| corrective (15) | 16.6 / 88.2 / 134.2 | 19.8 / **66.2** / **101.0** |
| confirming (51) | 0.58 / 2.4 / 139.5 | 0.45 / 28.7 / 426.9 |

Snap-hit (tol 1 cm): 45/66 both arms.  **Confirmation-guard failures: 4**
(tuned >1 cm worse than baseline on confirming events) — the do-no-harm bar
works and correctly rejects this checkpoint.  Reading: the corrective tail
shrinks (p90/max) but medians don't move and 4 confirmations degrade —
about what ~12 corrective training examples per fold can buy.  The
machinery (stratified folds, out-of-fold selection, guard, deployment
round-trip) is what this round ships; the physics waits for the mcp1k
labels.

## 6. Selection stage (the owner's step 2) — replayable offline

`rerank_replay.py` reproduces the 7-term composite from the scoreboard
`rows[]`:

- **Closure exact**: max |Σterms − recorded total| = 0 over 136 DL-snapped
  rows (66 events).
- **Grid over (`dl_vtx_min_accept_score`, `dl_vtx_score_scale`)**: the
  production point (4.0, 1000) scores **48/66 correct (tol 1 cm)** and *no
  grid point beats it* (`runs/rerank-grid-practice66.tsv`).  Reading: on
  this sample the acceptance threshold is not the bottleneck — the 18
  misses are net/candidate-set failures.  This sharpens the case for
  fine-tuning the net (and pr/51 graph work) over threshold tuning, and the
  same harness will re-answer the question on the mcp1k labels and on any
  fine-tuned net (it accepts new `dl_score`s).

## 7. Tier B charge-feature check (pr/52 §4)

`q = dQ*0.1 − 1000` over the 66 events (42,667 points,
`runs/qfeature-practice66.png`): p50 = 2082, p25–p75 = 867–3827,
MIP reference `mip_dqdx_median(43000) × median dx(0.60 cm) → q ≈ 1580`;
10.4 % of points negative (fit noise), 2.7 % at the dQ≈0 floor.  The
feature sits in a sane positive range with the MIP scale well inside the
bulk — no gross out-of-distribution pathology.  A quantitative uBooNE
comparison would need a uBooNE-era reference histogram (not in hand);
revisit only if fine-tuning stalls.

## 8. Label-efficient strategies — what O(100) labels can buy (2026-08-14)

Owner question after the practice run: hand-scan labels will stay at an
order of magnitude of 100 for the foreseeable future, MC truth at scale
comes later — is there anything achievable *now*?  The practice run itself
defines the regime: plain fine-tuning on ~10² labels moves tails, not
medians, and trips the do-no-harm guard.  The unifying principle for
everything below: **spend labels on validation and selection; spend other,
cheaper signals on gradients.**  Ordered by what each idea needs, not by
novelty.

### 8a. Zero new labels — achievable immediately

- **Failure taxonomy before any training.**  §6 already showed the 18
  misses are net/candidate failures, not threshold failures.  Split them
  further, per event, into (i) *net-wrong* — the truth vertex has a
  candidate in the scoreboard `rows[]` but the net's heat sits elsewhere;
  (ii) *candidate-missing* — no PR-graph candidate within snap tolerance of
  truth (no amount of net training helps; this is pr/51 graph territory);
  (iii) *snap/tolerance* — heat is right, snapping picks a wrong neighbour.
  All three are computable today from the scoreboard + labels
  (`rerank_replay.py` has the join).  This decides where the investment
  goes before a single gradient step.
- **Test-time augmentation (TTA).**  Run inference 4× under the §4
  reflections (plus a few sub-voxel shifts), map heat back to the original
  frame, average.  A cheap ensemble from one checkpoint — no training, no
  new labels; the 66 labels *evaluate* it rather than train it.  If it
  wins offline, production deployment is a config/pyutil knob round
  (default OFF, inference-path change), ~4× inference cost on one event.
- **Checkpoint ensemble / disagreement.**  The public uBooNE campaign has
  sibling checkpoints around CP24; averaging their heatmaps (or using
  their *disagreement* as an uncertainty flag for the rerank stage) is
  free modulo downloading them.  Same evaluate-don't-train logic as TTA.

### 8b. Stretching the labels we have

- **Candidate-ranking objective instead of dense regression.**  The
  deployed decision is *which scoreboard candidate wins*, a choice among
  ~2–10 discrete options — far lower-dimensional than a dense heatmap.
  Train with a margin loss (`score(truth candidate) >
  score(others) + m`) on candidate-pooled net features, or just a small
  calibration head on the frozen net.  A ~10-parameter decision problem is
  matched to a ~10²-label budget in a way 7.2M-param MSE regression is not.
- **Hard negatives from corrective events.**  The 15 corrective labels
  carry *two* facts each: where the vertex is AND where production wrongly
  put it.  Plain MSE uses only the first.  Adding a negative Gaussian at
  the production pick ("unlearn this spot") roughly doubles the gradient
  signal per corrective label, and targets exactly the mistakes the owner
  scanned to find.  Small change to `gaussian_truth()` + a manifest column
  (the production pick is already in the labels' rank-1-vs-pick delta).
- **Labels as selection currency, not gradients.**  With O(100) labels the
  statistically strongest use is *choosing between* models trained on
  other signals (MC, pseudo-labels, TTA variants) — a binomial comparison
  on 66 events resolves ~10 % differences, which is plenty for model
  selection even when it is far too noisy for gradient descent.  Concrete
  rule for the serious round: a strictly held-out label subset that never
  touches a gradient or an early-stopping decision.
- **Learned rerank weights.**  The 7 composite terms are recorded per row;
  fitting the 7 `W_*` by logistic regression / LASSO on the labels (the
  QL flag-penalty precedent) is a 7-parameter fit — perfectly sized.  §6
  says thresholds are not the practice-sample bottleneck, so this waits
  for mcp1k statistics, but the harness (`rerank_replay.py --grid` with
  `--w-*` multipliers) already exists.

### 8c. Gradients from the ~900 unscanned events

- **Confident pseudo-labels.**  The confirming rate is high (51/66 ≈ 77 %
  overall, and much higher when the rerank margin is large).  Select
  unscanned events where independent signals agree — high DL score, small
  snap distance, large margin over the runner-up candidate — and use the
  *production* vertex as a pseudo-label, weighted below hand labels.
  Risk is confirmation bias (the net re-learns its own habits); the guard
  is that pseudo-labels only ever enter training, never validation, and
  the held-out hand labels arbitrate.
- **Consistency regularization (FixMatch-style).**  On unlabeled events,
  penalize disagreement between the net's heatmaps under reflection/jitter
  views of the same event.  Needs zero labels, uses all 1000 events, and
  directly attacks a real observed failure mode (predictions that flip
  under symmetry are wrong on at least one side).
- **Active learning: steer the remaining scan.**  The scan is still
  running — the ordering of the remaining events is a free choice.
  Corrective labels are the scarce resource (15/66 ≈ 1-in-4 so far);
  ranking unscanned events by disagreement signals (TTA variance,
  checkpoint disagreement, small rerank margin, large DL-to-candidate
  distance) concentrates the owner's remaining scan effort where labels
  are most informative.  Output: a ranked event list for the scan panel,
  computable today from existing scoreboards.

### 8d. The MC path, sharpened

When large SBND MC arrives, truth vertices are free at scale — but the
right use is **pretrain on SBND MC, select on data labels**: fine-tune
CP24 on MC truth first (this fixes the uBooNE→SBND domain shift that no
amount of data labels can address at O(100)), then use the hand-scan
labels exclusively for data-vs-MC domain checks and checkpoint selection.
The infrastructure is already MC-ready: `build_dataset.py` only needs a
truth source instead of a label tag.

**Recommended order** (all pre-MC): 8a taxonomy → 8a TTA evaluation → 8c
active-learning scan ordering (while the scan is still running, so it
compounds) → 8b candidate-ranking head as the first training experiment on
the mcp1k labels.

### 8e. Round-2 execution (2026-08-14): results

Owner go-ahead for 8a+8b+8c, with two corrections: the data sample with PR
results is ~455 events (census: exactly **445**, all with scoreboards;
`products/prod0813/events-mcp1k-prod0813.txt`), and the samples differ
strongly, so a **numu50** anchor was built: the top-50 *labeled* mcp1k
events by `numu_score` from `products/prod0813/mcp1k-scores-prod0813.tsv`
(cut equivalent ≈ ≥2.43; beware `nue_score` is a −15.0 sentinel for
387/445 events — never select on numu>nue).  The mcp1k label tag is
**live and growing while this round ran** (176 at census → 181 at the
taxonomy run → 197 at the ranker run); every TSV below records its own
label count.  All results per sample throughout.

Repro block (round 2), from `dl_vtx_training/`:

```bash
python3 build_dataset.py --name numu50 --tags vtxscan-prod0813-mcp1k --numu-top 50
python3 build_dataset.py --name mix116 --tags vtxscan-prod0813 \
    vtxscan-prod0813-ncpi0 vtxscan-prod0813-mcp1k --numu-top 50 --lockbox 0.2
python3 taxonomy.py --tsv runs/taxonomy-20260814.tsv
python3 tta_eval.py --data data/mix116 --tsv runs/tta-mix116.tsv
python3 tta_eval.py --arm work-mcp1k-prod0813 --tsv runs/tta-signals-mcp1k.tsv
python3 rank_fit.py --tsv runs/rankfit-20260814.tsv
python3 scan_ranker.py --tta-tsv runs/tta-signals-mcp1k.tsv \
    --tsv runs/scan-ranking-20260814.tsv
python3 pseudo_labels.py --precision
python3 pseudo_labels.py --build pseudo0 --margin 200 --dl-best 500 --snap 99.0
python3 train.py --data data/mix116 --name ft1   --kfold 6 --epochs 30
python3 train.py --data data/mix116 --name ft1hn --kfold 6 --epochs 30 --hard-negative 0.5
python3 train.py --data data/mix116 --name ft1ps --kfold 6 --epochs 30 \
    --pseudo-data data/pseudo0 --pseudo-weight 0.25
python3 evaluate.py --data data/mix116 --run runs/<name> --tsv runs/<name>/eval.tsv
```

**8a taxonomy** (`runs/taxonomy-20260814.tsv`, 247 labels at run time,
tol 1 cm; classes mutually exclusive, sum asserted):

| sample | n | correct | cand-missing | net-wrong | sel-wrong |
|---|---|---|---|---|---|
| nueCC | 47 | 39 | 5 | 1 | 2 |
| NCpi0 | 19 | 14 | 0 | 4 | 1 |
| mcp1k (all) | 181 | 121 | 21 | 32 | 7 |
| — numu50 | 50 | 45 | 3 | 0 | 2 |
| **ALL** | **247** | **174** | **26** | **37** | **10** |

Reading: of 73 misses, **51 % are net-wrong** (training can help), **36 %
candidate-missing** (no training can help — pr/51 graph work), 14 %
selection-wrong (rerank/snap stage).  The samples really do differ:
**numu50 has zero net-wrong misses** — on beam-numu-like data the net is
not the problem; the net-wrong class concentrates in the low-`numu_score`
part of mcp1k and in NCpi0.  This reweights the program: for numu the
payoff is graph+selection work, for shower-rich samples it is the net.

**8a TTA** (`runs/tta-mix116.tsv`; x4 reflections, ensemble on the identity
lattice, identity check passed): **a wash for accuracy** — snap-hit 82/116
→ 82/116, medians 0.67→0.51 cm but tails sometimes blow up (an averaged
heatmap can hop clusters; worst 357 cm).  Argmax moved >1 cm on 29 events:
18 improved / 11 worsened (corrective slice: 8/3).  The per-event
**disagreement signal is the real product**: `tta_argmax_spread` > 5 cm
marks 36 events of which 27 (75 %) are production misses, vs 25 % below —
a 3x error concentrator, fed into the 8c ranker.  Verdict: do NOT deploy
TTA averaging; keep TTA disagreement as a flag.

**8a checkpoint ensemble — negative**: the public uboone-dl-vtx repo ships
code only (no .pth anywhere, no releases); CP24 in wire-cell-data is the
only public checkpoint.  Sibling-checkpoint ensembling/disagreement is not
buildable here; a future SBND fine-tune campaign produces its own siblings
natively (every epoch is checkpointed).

**8b candidate-ranking fit** (`runs/rankfit-20260814.tsv`, 253 labels at
run time): 163 events have ≥2 usable candidates, **84** also have the truth
reachable (the rest are single-candidate — no ranking information — or
candidate-missing).  Pairwise logistic on the 7 recorded terms + raw
dl_score + snap_dis + log1p(host_length) + trad_winner (11 params, term
sum == recorded total closure asserted).  Out-of-fold candidate choice:
**65/84 correct vs production's 61/84** (total-argmax also 61) — +6 on
mcp1k (32→38), −2 nueCC, −1 numu50.  Largest weights: s_isol +0.52,
dl_score/s_dl +0.37 each; s_snap and snap_dis ≈ 0 (the snap term buys
nothing in this fit).  Honest caveats: 65-vs-61 on n=84 is ~1σ binomial —
direction-setting, not a result; and the per-sample flips are exactly the
heterogeneity the owner flagged.  Re-fit on the completed scan before
drawing conclusions; the W_* knob round (§9) is how it would deploy.

**8c active-learning scan ranking** (`runs/scan-ranking-20260814.tsv`,
197 labeled / 248 unlabeled at run time): single "suspicion" heuristics
mostly do NOT enrich: margin x1.1, dl_best x1.3, snap_dis x1.5,
tta_spread x1.5 (top-quartile corrective rate vs 33 % base) — and
**route=dl-rerank-reject events are LESS corrective (25 %) than accepts
(34.8 %)**: the traditional fallback is usually fine; low DL confidence is
not label-value.  Fitting instead of hand-weighting (the S8 principle):
logistic P(corrective) on [log-margin, dl_best, snap_dis, route one-hots,
tta_spread] reaches **5-fold CV AUC 0.773**, top-quartile enrichment
**x2.0** (65.3 % corrective).  Dominant coefficients: log-margin −0.93,
route_reject −0.52, dl_best −0.38, snap_dis +0.17, tta_spread +0.14.
The TSV lists all 248 unlabeled events most-informative-first for the
scan panel; scanning in this order roughly doubles corrective labels per
scanned event at the top of the list.

**8c pseudo-labels** (`pseudo_labels.py --precision`, 197 labeled):
production is correct (≤1 cm) on 67.0 % of labeled events overall; the cut
`route=accept AND margin≥200 AND dl_best≥500` measures **95.6 % precision
on 45 labeled events** and passes 44 unlabeled → snapshot `data/pseudo0`
(truth := production vertex, `is_pseudo=1`, cut + measured precision frozen
in `cut.json`).  Pseudo events join only fold TRAIN sets at weight 0.25
(`train.py --pseudo-data`), never validation.

**8c consistency regularization**: implemented as a label-free
view-agreement MSE between two independently jittered voxelizations,
compared point-by-point (every input point maps to exactly one voxel in
each view — no cross-lattice interpolation).  2-epoch smoke together with
the hard-negative path: loss finite and stable, plumbing proven; full use
deferred to the serious round.

**8b training runs A/B/C** (mix116 minus 23 lockbox events = 93 events,
6-fold stratified by sample x corrective, freeze=head, lr0=1e-6, 30
epochs; out-of-fold `evaluate.py`, d_argmax in cm,
`runs/{ft1,ft1hn,ft1ps}/eval.tsv`):

| slice | baseline CP24 | ft1 plain | ft1hn hard-neg λ=0.5 | ft1ps +44 pseudo w=0.25 |
|---|---|---|---|---|
| all (93) | 0.73 / 41.1 / 173 | 0.74 / 64.8 / 476 | *identical to ft1* | 0.74 / 73.9 / 476 |
| corrective (16) | 27.2 / 81.2 / 134 | 31.2 / **63.7** / **76** | *identical* | 31.2 / 63.7 / 76 |
| confirming (77) | 0.67 / 12.6 / 173 | 0.70 / 67.8 / 476 | *identical* | 0.70 / 90.3 / 476 |
| numu50 (40) | 0.72 / 35.2 / 173 | 0.91 / 87.7 / 476 | 0.91 / 87.7 / 476 | 1.07 / 146.6 / 476 |
| guard failures | — | 11 | 11 | 14 |
| snap-hit (1 cm) | 63/93 | 58/93 | 58/93 | 58/93 |

Readings, all honest negatives at this label count:

- **The do-no-harm guard rejects all three checkpoints** (11–14 confirming
  events degraded >1 cm; snap-hit 63→58).  Same shape as round 1: the
  corrective tail shrinks (p90 81→64, max 134→76) but confirmations pay
  for it.
- **ft1hn ≡ ft1 on every out-of-fold metric although the models differ**
  (different checkpoint md5s, different best epochs): d_argmax is
  *discrete* — it only changes when the winning voxel flips — and at
  lr0=1e-6/7k-param head the hard-negative term (16 corrective events)
  never flips an out-of-fold argmax.  The hard-negative machinery is
  proven but unmeasurable at this scale/learning-rate; retry in the
  serious round with `--freeze none` or larger lr before judging it.
- **Pseudo-labels made things slightly worse** (guard 11→14, numu50 p90
  87.7→146.6): 44 pseudo events at weight 0.25 add gradient mostly where
  the net is already right — consistent with the confirmation-bias risk
  §8c named; the labels arbitrating (not the pseudo cut) is what caught
  it.
- **The numu50 slice degrades the most** under every fine-tune — exactly
  the sample where taxonomy found zero net-wrong misses.  Training a
  mixed nueCC/NCpi0/numu cocktail on ~10² labels pushes the net toward
  the shower-rich samples at the numu sample's expense; the serious round
  should either train per-sample or up-weight numu confirming events.
- **Lockbox (23 events) stays sealed**: every checkpoint was rejected on
  the working set, so no selection decision exists to confirm; the seal
  is opened the first time a candidate passes the guard.

Bottom line of round 2: with O(100) labels the *gradient*-consuming ideas
(8b/8c training arms) all fail the guard, while the *selection*-consuming
ideas deliver — the taxonomy (invest in pr/51 graph work for numu;
net work pays only on shower-rich samples), the x2.0 active-learning scan
ordering, and the 95%-precision pseudo-label cut are the shipped products.
This is the S8 principle confirmed empirically: spend labels on
validation and selection, not on gradients.

## 9. Deferred (documented, not built this round)

- **Exact-input dump knob** (toolkit C++, default-OFF): write `vec_xyzq`
  verbatim right after the assembly loop
  (`NeutrinoVertexFinder.cxx:4180`), before the `SCN_Vertex` call, one file
  per event, key-suppressed jsonnet knob like `vertex_scoreboard`.  Needs a
  knob round + gate ladder + fresh arms (the scanned prod0813 arms are
  records, M13 — never regenerate).  Build only if §3's parity is not good
  enough for the serious round.
- **W_* knob round** (pr/52 §5.2c): the seven rerank weights are constexpr;
  `rerank_replay.py` can scan multiplicative factors on the recorded terms,
  but honest retuning of the term *shapes* needs the knob round.
- **Serious training** on the mcp1k labels: snapshot the tag when the scan
  completes (`build_dataset.py --name mcp1k-<date> --tags
  vtxscan-prod0813-mcp1k`), switch `--kfold` to a held-out split, sweep
  σ ∈ {0.5, 1, 2} cm and freeze ∈ {head, none}, select on out-of-fold
  corrective improvement subject to zero guard failures, then Bee hand-scan
  of movers before any `dl_weights` flip.

## 10. Files / provenance

- Package: `sbnd_xin/dl_vtx_training/` (round 1).  Outputs under
  `data/practice66/`, `runs/` (uncommitted; `runs/parity-practice66.tsv`,
  `runs/rerank-grid-practice66.tsv`, `runs/qfeature-practice66.png`,
  `runs/ft0cpu/`; `runs/ft0/` is an aborted GPU duplicate of ft0cpu,
  ignore it).
- Round 2 (S8e): new tools `taxonomy.py`, `tta_eval.py`, `rank_fit.py`,
  `scan_ranker.py`, `pseudo_labels.py`; extensions in `build_dataset.py`
  (`--numu-top`, `--event-list`, `--lockbox`, sample/prod_xyz manifest
  columns), `dataset.py` (lockbox drop, sample-stratified folds,
  hard-negative targets, consistency views), `train.py`
  (`--hard-negative`, `--consistency`, `--pseudo-data/--pseudo-weight`,
  lockbox exclusion), `evaluate.py` (per-sample summaries, lockbox),
  `scn_vtx/io.py` (`load_scores_tsv`, `is_corrective`, `sample_of_label`),
  `scn_vtx/voxelize.py` (negative-Gaussian target).  Outputs (uncommitted):
  `data/{numu50,mix116,pseudo0}/`, `runs/taxonomy-20260814.tsv`,
  `runs/tta-mix116.tsv`, `runs/tta-signals-mcp1k.tsv` (16 empty-cloud
  events skipped), `runs/rankfit-20260814.tsv`,
  `runs/scan-ranking-20260814.tsv`, `runs/{ft1,ft1hn,ft1ps}/`,
  `runs/smoke-r2/`.
- Verbatim sources: toolkit `pyutil/python/SCN/DeepVtx.py`,
  `pyutil/python/SCN_Vertex.py` (apply-pointcloud HEAD 2026-08-14);
  input assembly `clus/src/NeutrinoVertexFinder.cxx:4147-4179`; scoreboard
  `clus/src/PrDisplayDump.cxx:643-720`.
- Prior docs: pr/52 (map + program), pr/75 (scan panel + labels), pr/4 (DL
  adoption), pr/2 G3 (retraining gap — this round makes it actionable).

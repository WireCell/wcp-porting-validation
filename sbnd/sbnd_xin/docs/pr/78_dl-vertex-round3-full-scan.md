# doc pr/78 — neutrino-vertex round 3 on the full hand scan: DL retrain, re-rank refit, pre-DL diagnosis

**Status (2026-08-15): round 3 EXECUTED on the completed mcp1k scan (407/445
labeled).  Headlines: (a) the ft2u fine-tune is the FIRST to pass both the
do-no-harm guard and the sealed lockbox (+8/378 OOF, +2/95 lockbox, 0 guard
fails) — a deployment checkpoint is staged; (b) `dl_vtx_min_accept_score`
4→10 gains +15/470 and CAPTURES THE ENTIRE learned-ranking gain (the two do
not stack, §4c); (c) the candidate-missing class is an ADMISSION gap, not
graph work — 47/50 events have a PR-graph vertex exactly at truth,
bottlenecked by `dl_vtx_top_k=5`; (d) the round-2 active-learning scan
ranking validated out of time at AUC 0.778.  Optimal-approach ranking and
owner decisions: §8.**

Continues doc pr/77 (infrastructure + rounds 1–2).  All tools live in
`sbnd_xin/dl_vtx_training/`; every number below traces to a TSV under
`dl_vtx_training/runs/` named in place.

## 0. Repro block

```bash
cd sbnd_xin/dl_vtx_training

# census + snapshot (473 labels, 20% lockbox, top-100 numu flagged)
python3 build_dataset.py --name full473 \
    --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0 vtxscan-prod0813-mcp1k \
    --numu-flag 100 --lockbox 0.2 --lockbox-seed 20260815

# S1: out-of-time validation of the round-2 active-learning ranking
#     (inline script; output frozen in runs/ranker-retrospective-20260815.txt)

# S2: taxonomy on all 473
python3 taxonomy.py --numu-manifest data/full473/manifest.tsv \
    --tsv runs/taxonomy-20260815.tsv

# S3: selection stage on full labels
python3 rank_fit.py --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0 \
    vtxscan-prod0813-mcp1k --tsv runs/rankfit-20260815.tsv
python3 rerank_replay.py --grid --tags vtxscan-prod0813 \
    vtxscan-prod0813-ncpi0 vtxscan-prod0813-mcp1k \
    --min-accepts 4 6 8 10 12 15 20 30 1e9 --scales 250 500 1000 2000 \
    --tsv runs/rerank-grid-full473-ext.tsv

# S4: pre-DL diagnosis of the candidate-missing class
python3 pre_dl_diag.py --taxonomy runs/taxonomy-20260815.tsv \
    --tsv runs/pre-dl-diag-20260815.tsv
python3 topk_replay.py --taxonomy runs/taxonomy-20260815.tsv \
    --tsv runs/topk-replay-20260815.tsv

# S5: training arms (fold-parallel; CPU and/or GPU)
FLAGS="--data data/full473 --kfold 6 --epochs 18 --bn-freeze --min-cloud 16 --clip 5.0"
for k in 0 1 2 3 4 5; do
  python3 train.py $FLAGS --name ft2u --fold $k --freeze none --lr0 1e-5 &
  python3 train.py $FLAGS --name ft2w --fold $k --freeze none --lr0 1e-5 \
      --upweight numu100:conf:3.0 &
  python3 train.py $FLAGS --name ft2  --fold $k --freeze head --lr0 1e-6 &
done
python3 merge_oof.py runs/<name>
python3 evaluate.py --data data/full473 --run runs/<name> --tsv runs/<name>/eval.tsv

# S6: round B (from the round-A reading)
#   ft2hn: $FLAGS --freeze none --lr0 1e-5 --hard-negative 0.5
#   ft2m3: $FLAGS --freeze linear --lr0 3e-6

# S7: stacked-selection 2x2 + pick-policy bound
python3 stack_sim.py --tsv runs/stack-sim-20260815.tsv
python3 topk_replay.py --taxonomy runs/taxonomy-20260815.tsv --pick-sim \
    --tsv runs/topk-picksim-20260815.tsv

# S8: ONE-TIME lockbox read (ft2u only) + deployment build
python3 lockbox_eval.py --data data/full473 --run runs/ft2u | tee runs/ft2u/lockbox.log
python3 train.py --data data/full473 --name ft2u-deploy --kfold 0 --epochs 10 \
    --freeze none --lr0 1e-5 --bn-freeze --min-cloud 16 --clip 5.0
```

## 1. Census — what the completed scan gives us

- PR results: **445** `work-mcp1k-prod0813/pr_evt*/calib-pr-evt*.json`, all
  with scoreboards.
- Labels: nueCC 47 + NCpi0 19 + **mcp1k 407** = **473** (38 PR events remain
  unlabeled).  Corrective at 1 cm: 8 + 5 + 137 = **150** (mcp1k rate 33.7%).
  The manifest's `dis>0` bookkeeping definition counts 157.
- Sample identity: `--numu-flag 100` marks the top-100 labeled mcp1k events
  by `numu_score` as sample `numu100` (round-2 caveat still applies:
  `nue_score` is a −15 sentinel — never cut numu>nue).  Corrective rate in
  the numu100 subset is 14%, vs ~40% for the shower-rich remainder —
  round 2's sample split holds at scale.
- Snapshot `data/full473`: 473 events, stratified 20% lockbox = **95 events**
  (seed 20260815, sample × corrective), train pool 378 (125 corrective).
  Frozen; `runs/` and `data/` stay uncommitted as before.

## 2. Free science: the round-2 active-learning ranking validated out of time

Round 2 froze `runs/scan-ranking-20260814.tsv` — the 248 then-unlabeled
events ranked by a fitted P(corrective) (in-sample CV AUC 0.773).  The scan
has since labeled 210 of them, so the frozen predictions can be scored
against reality (`runs/ranker-retrospective-20260815.txt`):

- **out-of-time AUC 0.778** (CV had promised 0.773 — held exactly);
- top-quartile corrective enrichment **65.4% vs 34.3% base (×1.91)**;
- scanned in ranked order, the first 50 scans would have contained 33
  corrective events vs 17 expected at random.

This is the strongest possible validation of the round-2 product: the
ordering was fixed *before* the labels existed.  The active-learning tool is
real and should be used for any future scan.

## 3. Taxonomy on 473 (`runs/taxonomy-20260815.tsv`)

Classes as in round 2: correct / candidate-missing (no scoreboard-row
candidate within tol of truth) / net-wrong (candidate exists, no recorded
top-5 voxel near truth) / selection-wrong (top-5 voxel near truth, chosen
vertex elsewhere).  tol = 1 cm (3 cm table in the TSV run log):

| sample  |   n | correct | cand-missing | net-wrong | sel-wrong |
|---------|----:|--------:|-------------:|----------:|----------:|
| nuecc   |  47 |      39 |            5 |         1 |         2 |
| ncpi0   |  19 |      14 |            0 |         4 |         1 |
| mcp1k   | 407 |     269 |           45 |        76 |        17 |
| numu100 | 100 |      86 |            7 |         4 |         3 |
| ALL     | 473 |     322 |           50 |        81 |        20 |

(The mcp1k row excludes the numu100 events: the four sample rows are
disjoint and sum to 473.)

- The trainable class (net-wrong) grew 37 → **81 events**; round 2's "numu
  has zero net-wrong" softens to 4/100 at the larger subset — still 6× below
  the shower-rich rate (76/307).  Net work still pays mostly on shower-rich
  data events.
- Selection-wrong (20) + a share of candidate-missing (§6) are addressable
  without touching the net — see §4.

## 4. Selection stage refit — the decisive phase-2 results

### 4a. Learned candidate ranking beats production (McNemar p = 0.044)

`rank_fit.py` on all 473 labels (pairwise logistic on the 7 recorded
composite terms + raw dl_score + snap_dis + log1p host length + trad_winner;
6-fold CV stratified by sample; closure vs the recorded composite exact on
all 156 eligible events; `runs/rankfit-20260815.tsv`):

| sample | n | prod recorded | composite argmax | rankfit (out-of-fold) |
|--------|---|--------------|------------------|----------------------|
| mcp1k  | 123 | 81 | 83 | **98** |
| ncpi0  | 7 | 5 | 5 | 5 |
| nuecc  | 15 | 14 | 15 | 12 |
| numu100| 11 | 10 | 10 | 10 |
| ALL    | 156 | 110 | 113 | **125** |

Round 2's ~1σ hint (65/84 vs 61/84) is now a significant result:
discordant events 32 fit-only-right vs 17 prod-only-right, McNemar exact
**p = 0.044**.  The gain is entirely on the data sample (+17 on mcp1k, −2 on
nueCC MC).  Fitted weights (standardized): s_isol +0.58,
log1p_host_length +0.39, trad_winner +0.34, s_fwd_z +0.33, s_clen +0.28,
dl_score/s_dl +0.27 each, s_main +0.24, and — consistent with round 2 —
**s_snap and snap_dis fit to ≈ 0** (+0.008 / −0.045): the snap-distance
term carries no selection information at this sample size.

The 7 composite weights are hard-coded in `NeutrinoVertexFinder.cxx`; a
default-OFF `dl_vtx_rerank_weights` knob (array of per-term multipliers)
would let this be deployed config-side.  **Proposal, not implemented** (§8).

### 4b. The (min_accept, scale) operating point moved

`rerank_replay.py --grid` on 470 truth-reachable events
(`runs/rerank-grid-full473-ext.tsv`; production = `dl_vtx_min_accept_score`
4.0, `dl_vtx_score_scale` 1000, both already config,
`TaggerCheckNeutrino.cxx:305-306`):

- production 4.0/1000: 268/470 correct (DL route taken on 404);
- **best 10.0/1000: 283/470 (+15)**, an *interior* optimum (12/15/20/30 all
  lower), DL route on 252;
- DL fully off (min_accept=∞): 236/470 — the DL stage is worth +32 even
  before retuning;
- round 1's "4.0/1000 is already optimal" was a 66-label statement; at 473
  labels the optimum is a stricter acceptance — the DL rerank was
  over-riding the traditional pick on events where the traditional pick was
  right.

Config-only change, but a production behavior change ⇒ owner-gated (§8).
Caveat: replay assumes route/candidates fixed while only the acceptance
threshold moves; a real A/B run must confirm.

### 4c. The two selection gains DO NOT stack (`stack_sim.py`)

2×2 replay — chooser (composite | rankfit-OOF) × acceptance (4.0 | 10.0),
same acceptance semantics as production, rankfit acting only on its 156
eligible events (`runs/stack-sim-20260815.tsv`):

| min_accept | chooser   | correct/473 |
|-----------:|-----------|------------:|
| 4.0 | composite | 268  (production) |
| 4.0 | rankfit   | 280  (+12) |
| 10.0 | composite | **283  (+15)** |
| 10.0 | rankfit   | 283  (+15) |

The learned ranking and the stricter acceptance fix (almost exactly) the
same events: the composite's wrong picks are concentrated on marginal
accepts that min_accept=10 re-routes to the traditional winner.  For
deployment this is decisive: **the config-only acceptance retune captures
the entire measured selection gain, and the C++ rerank-weights knob adds
nothing on top of it** — proposal 3 is demoted to "only if (1) is
declined".

## 5. DL-vertex training arms (round A)

### 5a. Two real bugs the bigger dataset exposed (both fixed in train.py)

- **Tiny-cloud NaN**: evt 171892/287431 have 4-point PR graphs (3 voxels).
  A 3-voxel cloud reaches the deepest UNet level as a single active site;
  sparseconvnet's BN then (a) NaNs its *running stats* (unbiased batch
  variance 0/0) — training looks healthy (batch stats) while **every
  eval-mode forward returns NaN** (the round-A first launch showed
  val_loss=nan, val_d50 frozen at the model-independent 35.02 = argmax
  falling back to voxel 0), and (b) with some augmentation draws NaNs the
  train-mode forward itself (first NaN traced to evt 171892's second flip
  view, step 814 of fold-0 epoch 0).  SCN's BN autograd asserts train mode
  in backward, so eval-mode-BN fine-tuning is not an option.  Fixes:
  `--min-cloud 16` (drop degenerate clouds from TRAIN folds only; they stay
  in val), `--bn-freeze` (snapshot the CP24 running stats and restore them
  after every training epoch — batch-stat training unchanged, clean stats
  for every eval and every saved checkpoint), `--clip 5.0`
  (clip_grad_norm_) for lr 1e-5 robustness.  1-epoch smoke after the fix:
  train 0.0098 / val d50 2.72 (= CP24 baseline).
- **freeze=linear at lr 1e-5 diverged** even before the tiny-cloud NaN (the
  first ft2m launch hit nan train loss at epoch 1); retried in round B at
  lower lr rather than debugged further.

### 5b. Arms

All: full473 minus lockbox (378 events), 6-fold stratified
(sample × corrective), ×4 reflections + sub-voxel/charge jitter, 18 epochs,
`--bn-freeze --min-cloud 16 --clip 5.0`, fold-parallel (ft2u on the two
RTX 4090s, 3 folds each; ft2w/ft2 on CPU).

| arm | freeze | lr0 | extra |
|-----|--------|-----|-------|
| ft2  | head (7k params) | 1e-6 | round-1/2 continuity control |
| ft2u | none (7.2M) | 1e-5 | the original t48k campaign lr |
| ft2w | none | 1e-5 | `--upweight numu100:conf:3.0` (numu do-no-harm pressure in-loss) |

### 5c. Round-A results (out-of-fold on the 378 non-lockbox events)

`evaluate.py` per arm (`runs/<arm>/eval.{log,tsv}`); guard = confirming
events whose tuned d_argmax is >1 cm worse than CP24; snap-hit = events
with d_argmax ≤ 1 cm:

| arm | guard fails | snap-hit (base 165) | movers >1cm up/down | corrective p50 (base 44.2) | nuecc p90 (base 35.9) | ncpi0 p90 (base 62.6) | numu100 p90 (base 54.9) |
|-----|------------:|--------------------:|--------------------:|------:|------:|------:|------:|
| ft2 (head, 1e-6)  | 0 | 165 (+0) | 0 / 0 | 44.2 | 35.9 | 62.6 | 54.9 |
| ft2u (none, 1e-5) | **0** | **173 (+8)** | 17 / 1 | 42.0 | 25.6 | 43.3 | 37.0 |
| ft2w (ft2u + numu-upweight ×3) | 2 | 175 (+10) | 19 / 5 | 42.0 | 25.6 | 43.3 | 37.0 |

- **ft2 is bit-inert**: every one of the 378 events identical to baseline
  to 0.01 cm.  Verified non-vacuous (7/137 checkpoint tensors differ from
  CP24, val_loss creeps down) — at lr 1e-6 the head moves but no argmax
  ever flips.  Round 2's discreteness explanation reproduced at 4× the
  data; this lr/freeze regime is dead and stays dead.
- **ft2u is the first fine-tune in three rounds to pass the do-no-harm
  guard**: +8 snap-hits, ZERO confirming events degraded, every sample's
  tail improves (nueCC p90 36→26, NCpi0 62→43, numu100 55→37).  What
  changed vs round 2: 3× the labels, lr 1e-5 with all 7.2M params free
  (argmaxes can actually flip), BN stats frozen, degenerate clouds
  excluded, gradients clipped.
- **ft2w** (numu confirming ×3 up-weight) buys +2 more snap-hits at the
  price of 2 guard failures (evt 286400: 146→187 cm on an already-lost
  event; evt 292005: 0.48→1.55 cm), and its numu100 p50 is *not* better
  than ft2u's (0.72 vs 0.64) — the upweight is neutral-to-slightly-negative
  here.  The round-2 "cocktail pushes the net off numu" effect did not
  recur at 473 labels even without the upweight.

### 5d. Round-B results and the lockbox verdict

Round B from the Round-A reading (same base flags):

| arm | recipe | guard fails | snap-hit |
|-----|--------|------------:|---------:|
| ft2m3 | freeze=linear, lr 3e-6 (diverged-probe retry) | 0 | 165 (+0) — inert |
| ft2hn | ft2u + `--hard-negative 0.5` | 2 | 168 (+3) |

- The linear probe is argmax-inert even at a stable lr — capacity, not lr,
  is what ft2/ft2m3 lack.  Only full unfreeze moves this net.
- Hard negatives are **net-negative even at the argmax-flipping lr** (+3
  with 2 guard failures vs plain ft2u's +8 with 0).  Round 2's "HN will
  bite once the lr can flip argmaxes" prediction is refuted; the idea is
  retired.

**Lockbox read (ONE-TIME, on ft2u only** — chosen on OOF metrics before the
seal was broken; `runs/ft2u/lockbox.log`): baseline snap-hit 41/95;
per-fold checkpoints +0..+2 with at most 1 guard fail in one fold; pooled
per-event median over the 6 fold checkpoints **+2 snap-hits, 0 guard
failures**.  The +2.1% lockbox gain rate matches the +2.1% OOF rate —
**ft2u passes, the first fine-tune in three rounds to clear both bars.**

Deployment build: `runs/ft2u-deploy/fold0/CP9.pth` — the ft2u recipe on all
378 non-lockbox events, no folds, 10 epochs (≈ the median fold-selected
epoch).  To A/B: copy under a NEW name into `wire-cell-data/` (never
overwrite `t48k-m16-l5-lr5d-res0.5-CP24.pth`) and point the `dl_weights`
TLA at it.  Production flip is the owner's call; expected effect from
OOF+lockbox: ~+2% absolute snap-hit on data events, no measured harm.

## 6. Pre-DL diagnosis — the candidate-missing class is an ADMISSION gap, not a graph gap

`pre_dl_diag.py` on the 50 candidate-missing events (truth vs the PR graph
itself, `runs/pre-dl-diag-20260815.tsv`):

- **47/50: a PR-graph VERTEX sits exactly at truth** (d_vtx = 0.00 — the
  owner's pick is itself a graph vertex that never became a scoreboard
  candidate); 3/50 sit on segment interiors (`on-track`); **0/50 off-graph**.
- Round 2 filed this class as "pr/51 graph territory, untrainable".  That
  was wrong for the completed scan: the graph has the vertex; the DL
  *candidate admission* dropped it.

Mechanism (`NeutrinoVertexFinder.cxx:4151-4368`): DL candidates = each of
the top-`dl_vtx_top_k` voxels snapped to the nearest graph vertex
(`cand_vertices` = ALL graph vertices, nothing structurally excluded);
production `dl_vtx_top_k = 5` (`cfg .../sbnd/clus.jsonnet:1898`).  Only 2/47
lost to snap tie-breaks; 45/47 simply had no top-5 voxel near truth.

`topk_replay.py` re-runs CP24 on the rebuilt cloud and simulates larger
top_k (`runs/topk-replay-20260815.tsv`; rebuilt-cloud caveat: pr/77 parity
is exact top-1 on 39/66, and indeed 4/50 events replay-admit at k≤5 where
production did not):

| simulated top_k | candidate-missing events admitted |
|-----------------|----------------------------------|
| 5 (production)  | 4/50 (replay disagreement) |
| 10              | 15/50 |
| 20              | **28/50** |
| 50              | 38/50 |
| never (≤50)     | 12/50 — true net misses |

Raising `dl_vtx_top_k` 5→20 is config-only and would make ~28 events
*available* to the reranker (admission is necessary, not sufficient — the
rerank must still choose them; and more candidates also give the rerank
more chances to go wrong on currently-correct events).

The realized recovery cannot be simulated offline: a pick-policy check
(`topk_replay.py --pick-sim`, `runs/topk-picksim-20260815.tsv`) shows
DL-score-alone selection reaches only 226/473 vs the composite's 322/473 —
the geometric terms carry much of the selection power, and their values for
the newly admitted candidates are not recorded anywhere.  (The dl-argmax
policy is also k-independent by construction — the max-score candidate is
always the top-1 voxel's snap — so that number is a policy baseline, not a
k-scan.)  The honest statement: **k=20 admission recovers ≤28/50; the
realized number needs one real PR rerun of the 445 events with
`dl_vtx_top_k=10/20` into a fresh work dir** — the concrete next-round
experiment (§8).

## 7. Files

New round-3 tools (committed): `pre_dl_diag.py`, `topk_replay.py` (incl.
`--pick-sim`), `stack_sim.py`, `lockbox_eval.py`, `merge_oof.py`; extended:
`train.py` (`--upweight`, `--bn-freeze`, `--min-cloud`, `--clip`, per-fold
`oof_fold<k>.tsv`), `build_dataset.py` (`--numu-flag`), `taxonomy.py`
(`--numu-manifest`), `scn_vtx/io.py` (`sample_of_label` numu_name),
`README.md`.  Uncommitted outputs: `data/full473`,
`runs/taxonomy-20260815.tsv`, `runs/rankfit-20260815.tsv`,
`runs/rerank-grid-full473{,-ext}.tsv`, `runs/pre-dl-diag-20260815.tsv`,
`runs/topk-replay-20260815.tsv`, `runs/topk-picksim-20260815.tsv`,
`runs/stack-sim-20260815.tsv`, `runs/ranker-retrospective-20260815.txt`,
`runs/ft2*/` (each: fold CPs + eval.tsv; `runs/ft2u/lockbox.log`;
`runs/ft2u-deploy/fold0/CP9.pth` = the staged deployment checkpoint).

## 8. The optimal-approach ranking (owner decisions)

Every direction measured this round, ordered by verified gain per unit of
deployment risk (473-label replays and OOF/lockbox training results;
"gain" = correctly-chosen vertices at 1 cm):

| # | direction | measured gain | change needed | caveat |
|---|-----------|--------------|---------------|--------|
| 1 | `dl_vtx_min_accept_score` 4.0 → 10.0 | **+15/470** (replay, interior optimum) | config value only | needs one real A/B rerun to confirm route stability |
| 2 | fine-tuned net (ft2u recipe) | **+8/378 OOF, +2/95 lockbox, 0 guard fails anywhere** | new weights file + `dl_weights` TLA | orthogonal to (1) — net gain is at the voxel stage, (1) at acceptance |
| 3 | `dl_vtx_top_k` 5 → 10/20 | ≤15–28/50 admission (upper bound) | config value only | realized gain unknown offline (§6); test WITH (1), which guards the added candidates |
| 4 | learned rerank weights knob | +12/473 alone but **+0 on top of (1)** (§4c) | default-OFF C++ knob | demoted: redundant with (1); revisit only if (1) declined |
| 5 | hard negatives / linear probe / head-only | 0 or negative | — | retired with evidence (§5) |

Suggested sequence: A/B of (1) alone → A/B of (1)+(2) → PR rerun with
(1)+(3) on the 445 events to measure the top_k realized recovery.  The 38
still-unlabeled PR events: the round-2 active-learning ranking (§2,
out-of-time-validated) orders them for scanning.

## 9. Candidate ideas for a future round (proposed, not built)

Claude-session discussion folded in here (2026-08-15). Motivation: every
gradient-based retrain so far (round 1 `ft0`, round 2 `ft1*`, round 3
`ft2`/`ft2u`/`ft2w`) trains a dense per-voxel MSE against the Gaussian
truth, then only *discovers* do-no-harm violations after the fact via
`evaluate.py`'s guard. Meanwhile the piece that is actually winning (§4a
`rank_fit.py`) works because it (a) has a tiny parameter count matched to
the label budget and (b) is trained directly on the deployed decision
(which candidate wins), not a proxy. The ideas below try to give the *net*
those same two properties without touching backbone weights.

### 9a. Adapter head on frozen intermediate features (not just the final score)

Keep `DeepVtx`'s pretrained backbone entirely frozen (freeze **everything**,
stricter than `--freeze head`) and add a small new module that reads an
intermediate per-voxel feature map (e.g. the last UNet decoder block's
16-channel output, before `linear`) instead of only the collapsed
`pred[:,1]-pred[:,0]` score `rank_fit.py` currently consumes. Two design
choices that matter:

- **Near-identity init**: initialize the adapter so its contribution starts
  at ~0 (zero-init final layer, standard adapter/LoRA practice). At init
  the model is then *exactly* the frozen baseline — do-no-harm safety is
  structural, not just hoped for, unlike unfreezing existing decoder
  weights (why `ft2`/`ft2u`/`ft2w` all risk confirming-event drift: they
  edit weights that also produce today's correct predictions).
- **Feed `rank_fit`, don't replace it**: rather than a new dense heatmap,
  have the adapter emit 1-4 extra per-candidate scalars (pooled from the
  richer feature map at each scoreboard candidate's voxel) and add them as
  new columns to `rank_fit.py`'s existing 11-feature logistic fit. This
  reuses the exact infrastructure that already cleared McNemar p=0.044
  rather than restarting the dense-MSE/guard cycle that has failed twice.

Cost: new forward-pass code (fork `DeepVtx`, per CLAUDE.md M10 — the
production class stays untouched), plus a hook to expose intermediate
activations that don't currently leave `sparseModel`. Not yet built.

### 9b. Train on the deployed objective, not a proxy for it

`train.py` optimizes dense per-voxel MSE against a Gaussian truth; the
thing actually deployed is a discrete choice among ~2-10 scoreboard
candidates (pr/77 §8b's framing). Training loss and deployed decision have
never matched. Concretely: replace (or add as a second loss term) a
listwise/margin loss evaluated only at the scoreboard candidate voxels —
`score(truth candidate) > score(other candidates) + m` — so gradient steps
optimize the exact quantity `evaluate.py`'s guard and `rank_fit`'s McNemar
test both measure. This is a training-target change, not an architecture
change, and composes with 9a (the adapter could be trained with this loss
instead of dense MSE) or with plain `--freeze head`.

### 9c. Do-no-harm as an in-loop regularizer, not a post-hoc guard

Every gradient round has learned about guard violations only after
`evaluate.py` runs — by then the label-scarce confirming set (51-323
labels depending on round) has already been spent finding out. The ~900
still-unscanned PR events are free signal for the *opposite* direction: on
high-confidence unlabeled events (large rerank margin, low TTA spread,
`route=accept` — the same signals `scan_ranker.py` already scores), treat
CP24's own top-1 voxel as a pseudo-anchor and penalize moving away from it
during training (same weighting discipline as `pseudo_labels.py`'s 0.25,
same confirmation-bias caveat pr/77 §8c already named). This scales
anti-drift protection from ~51-150 labeled confirming events to
effectively the full unlabeled pool, turning the guard from a
reject-after-the-fact check into gradient signal the training itself
respects. Composes with 9a/9b.

**None of 9a-9c is built.** If pursued, order by cost: 9c is a `train.py`
flag (reuses `pseudo_labels.py` machinery); 9b is a new loss function on
existing model outputs; 9a is the only one requiring new architecture code
and a `DeepVtx` fork.

## 9-EXECUTED (round C, same night): 9b built and measured, 9c empirically dead, 9a deferred

### 9c — no anchor pool left (measured null)

`pseudo_labels.py --precision` on today's labels: only **38** PR events
remain unlabeled, and **zero** of them pass any ≥95%-precision confidence
cut (every `dl_best>=500` cut has unlabeled yield 0 — the unscanned residue
is exactly the low-confidence tail, partly *because* the scan followed the
round-2 active-learning ordering).  The "~900 unscanned events" premise no
longer holds post-scan; 9c was a round-2 idea whose window has closed.  It
would apply again to any future large unlabeled pool (`data/pseudo1` holds
the empty build as the record).

### 9b — deployed-objective margin loss: built (`build_cands.py`,
`dataset.py` candidate transform, `train.py --cands/--cand-margin`)

Candidate sidecar `data/full473-cands`: 473 events, 156 margin-eligible
(exactly rank_fit's eligibility, as designed).  The truth candidate's voxel
score is required to beat every other candidate's by 0.1 (margin term added
to the dense MSE; candidates ride through the same flip/jitter transforms
as the truth).

Arm `ft2c9b` (ft2u recipe + margin weight 1.0), OOF:

| arm | guard fails | snap-hit | movers up/down | corrective p50/p90 |
|-----|------------:|---------:|---------------:|-------------------:|
| ft2u (reference) | 0 | 173 (+8) | 17 / 1 | 42.0 / 237.6 |
| ft2c9b (margin 1.0) | 4 | **176 (+11)** | 24 / 6 | **41.3 / 226.7** |
| ft2c9bw3 (margin 0.3) | 2 | 175 (+10) | 20 / 4 | 41.5 / 226.7 |

The margin loss does exactly what §9b predicted — the biggest raw gain of
all seven arms this round, concentrated on the deployed decision — but it
buys gain with guard failures at every weight tried (4 at w=1.0, 2 at
w=0.3; evt 71372 ncpi0 0.24→4.84 cm persists at both weights).  Nothing
dominates plain ft2u's 0-fail/+8 point on both axes.  Verdict: **the
deployed-objective loss is real and is the best raw-gain recipe, but on a
do-no-harm-first policy ft2u remains the deployment choice**; a weight
sweep below 0.3 (or margin loss on an adapter, 9a) is the natural
follow-up if the owner wants to trade ~2 confirming events for ~3 extra
corrective fixes.  The lockbox was NOT re-read for the round-C arms (it
had already arbitrated ft2u; a second read would turn it into a selection
set) — any round-C deployment candidate needs the owner's A/B instead.

### 9a — deferred, with a reason beyond cost

The adapter head's payoff path is "extra features for rank_fit" — but §4c
showed the rank_fit gain is fully subsumed by the `min_accept` retune on
this label set.  Until an A/B of §8 items (1)+(2) lands and the residual
selection errors are re-measured, new rank_fit features have no measurable
headroom; building the fork now would optimize a margin that may not
exist.  Revisit after the §8 sequence.

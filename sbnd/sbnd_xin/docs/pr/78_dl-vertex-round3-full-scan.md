# doc pr/78 — neutrino-vertex round 3 on the full hand scan: DL retrain, re-rank refit, pre-DL diagnosis

**Status (2026-08-15): round 3 EXECUTED on the completed mcp1k scan (407/445
labeled).  Selection stage: rank_fit now beats production at McNemar p=0.044
and the (min_accept, scale) operating point is no longer optimal — both are
concrete config/knob proposals.  Pre-DL: the candidate-missing class is NOT
graph work — 47/50 events have a PR-graph vertex exactly at truth and the
admission bottleneck is `dl_vtx_top_k=5` (raising it to 20 would admit 28/50).
Training arms: see §5.  Owner decisions requested in §8.**

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

RESULTS_PLACEHOLDER_ROUND_A

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
rerank must still choose them, which is exactly what §4a improves; and more
candidates also give the rerank more chances to go wrong on
currently-correct events).  **Proposal for an owner-gated A/B, not flipped
here** (§8).

## 7. Files

New round-3 tools (committed): `pre_dl_diag.py`, `topk_replay.py`,
`merge_oof.py`; extended: `train.py` (`--upweight`, `--bn-freeze`,
`--min-cloud`, `--clip`, per-fold `oof_fold<k>.tsv`), `build_dataset.py`
(`--numu-flag`), `taxonomy.py` (`--numu-manifest`), `scn_vtx/io.py`
(`sample_of_label` numu_name).  Uncommitted outputs: `data/full473`,
`runs/taxonomy-20260815.tsv`, `runs/rankfit-20260815.tsv`,
`runs/rerank-grid-full473{,-ext}.tsv`, `runs/pre-dl-diag-20260815.tsv`,
`runs/topk-replay-20260815.tsv`, `runs/ranker-retrospective-20260815.txt`,
`runs/ft2*/`.

## 8. Owner decisions requested

1. **`dl_vtx_min_accept_score` 4.0 → 10.0** (config-only): +15/470 on the
   full scan replay (§4b).  Needs a real A/B (behavior change).
2. **`dl_vtx_top_k` 5 → 10 or 20** (config-only): admits 15–28 of the 50
   candidate-missing events (§6).  Interacts with (1) and (3); suggest
   testing combined.
3. **Configurable rerank weights** (`dl_vtx_rerank_weights`, default-OFF
   toolkit knob): the fitted ranking is +15/156 over production at p=0.044
   (§4a).  C++ change under the usual byte-identical-off bar.
4. **Fine-tuned net deployment**: see §5 verdict.
5. The 38 still-unlabeled PR events: the round-2 ranking (§2) orders them.

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

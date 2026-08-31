# doc pr/105 — neutrino-vertex selection: strategy comparison + re-rank re-optimization

**Status: CLOSED 2026-08-21 — production UNCHANGED (owner decision); selection stack measured optimal; `fit_exclusion` cost quantified, stays ON.**

Owner ask (2026-08-21): the current selection (DL + composite re-rank) was
optimized before the pr/90–104 PR improvements; doc pr/100 §3.4 measured a
real −3.2 pp same-event regression across that span, `fit_exclusion`
(pr/98) suspected.  Systematically compare (3.1) DL only / no re-rank,
(3.2) re-rank at `min_accept=4`, (3.3) with vs without the pr/89 topology
term, (3.4) a fine-tune; attribute the regression; pick the best approach
for the vertex itself; validate live; flip ON.  Metric = the data hand-scan
labels, per sample, nueCC first.

## 0. Repro block

```bash
cd sbnd_xin   # toolkit c550541f (pr/104 flip) + this round's cfg thread; lib build/clus 08:10 (no C++ change)

# 0. cfg thread for strategy 3.1 (cfg only): dl_vtx_rerank TLA in wct-pr-perevt.jsonnet +
#    sbnd/clus.jsonnet, SBND_DL_VTX_RERANK env in run_pr_chain_batch.sh.  Compiled-config proof
#    (evt 10550 TLA set, scratch pr105cfg/): unset => md5 6baabf1a == HEAD; =false => one key differs.

# 1. labeled universe (1054 labels: harv3 x4 + mcp2k + mcp2k-auto + mcp2k-ragree), per-sample lists
#    -> /home/xqian/tmp/vtx105-evts-{nuecc48,ncpi0,mcp1k,mcp2k}.txt (47/19/407/581)

# 2. arms over the labeled universe only, sequential, PR_JOBS=32 PR_EXTRA_STAGES=pr_display
#    (/home/xqian/tmp/pr105_arms.sh; env per arm):
#    base (none) | nofitx SBND_FIT_EXCLUSION=false | dlonly SBND_DL_VTX_RERANK=false |
#    ma4 SBND_DL_VTX_MIN_ACCEPT=4 | topo3 SBND_DL_VTX_TOPO_WEIGHT=3.0 | topk10 SBND_DL_VTX_TOP_K=10 |
#    trad SBND_DL_WEIGHTS='' | pre103 SBND_MVGA_PASSTHRU=0 SBND_MVGA_INTERPOSED_FALLBACK=0
#                                    SBND_VERTEX_JUNCTION_SNAP=false SBND_VJS_OVERRIDE_KINK_SNAP=false
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<s>-ql0819 work-vtx105-<arm>-<s> data <evts>

# 3. carry the 1054 labels onto the base arm (position join, TOL 1.0) -> vtxscan-vtx105-* (7 tags)
python3 vtx_rules/carry_labels.py --write --delta-list dl_vtx_training/runs/vtx105/delta.txt \
    --tsv dl_vtx_training/runs/vtx105/carry.tsv --arms <7 pairs, see sec 1>

# 4. strategy table, lockbox EXCLUDED (tuning set)
python3 dl_vtx_training/vtx_strategy_table.py --carried-tags <vtx105 x7> --orig-tags <source x7> \
    --arms base=work-vtx105-base-{sample} ... --ipw-tsv dl_vtx_training/runs/ipw-vtx100-20260820.tsv \
    --exclude-events dl_vtx_training/runs/vtx100-20260820/lockbox.txt --events-tsv docs/pr/105_events-tune.tsv

# 5. re-rank search on the tuning set (outcome-calibrated ruler, coverage from every selection-only arm)
python3 dl_vtx_training/rerank_tune.py --arm-template '<SX>/work-vtx105-{arm}-{sample}/pr_evt{evt}/calib-pr-evt{evt}.json' \
    --arm-names base,topo3,dlonly,ma4,topk10,trad --rows-arm topo3 --tags <source x7> \
    --exclude-events dl_vtx_training/runs/vtx100-20260820/lockbox.txt --guard-samples nuecc48 --search

# 6. ONE lockbox read (--only-events lockbox.txt) of the same table; then live validation + flip (sec 6)
```

## 1. Setup

**Why now.** Doc pr/100 §3.4: on 831 identical events the selection went
669/831 (pr/89 epoch, `a681b3e1`) → 642/831 (pr/100 epoch, `8573877f`),
−3.2 pp, CIs disjoint, every sample the same direction.  ~30 PR knobs
shipped in between; `fit_exclusion` (pr/98) is the prime suspect because it
changes the vertex *fit* that every candidate position — and therefore every
DL snap and every label carry — is measured against.  The selection stack
itself (DL + 7-term composite, `min_accept` 4→10, pr/89 topo term OFF) was
tuned against a candidate graph that no longer exists.

**Label universe.** 1054 hand-scan labels from data, 755 human / 299
ai-scanner (pr/88 gate 39/40): `TAGS_HARV3` (nuecc48 42, ncpi0 19, mcp1k
388, delta 24) + `vtxscan-mcp2k` 242 + `-auto` 299 + `-ragree` 40.  Per
sample: nuecc48 47, ncpi0 19, mcp1k 407, mcp2k 581.

**Held-out.** The pr/100 lockbox (`runs/vtx100-20260820/lockbox.txt`, 219
events = 9 nuecc48 / 1 ncpi0 / 95 mcp1k / 114 mcp2k; seed 20260820; never
read before this doc).  Every number in §2–§4 excludes it; it is read once
in §5.  Caveat registered up front: with 9 nueCC events the lockbox cannot
adjudicate nueCC on its own — the nueCC verdict is read on tuning-set +
lockbox together, stated as such.

**Truth rule.** Carried label (re-anchored on `work-vtx105-base-*` by
position, TOL 1.0) where the carry succeeded, the original frozen label
otherwise; denominator = all label events, carried subset printed beside
(pr/100 §3.3: carrying is a filter, not a coordinate fix).  Tolerance 1.0 cm
primary, 1.5 cm always beside it.

**Reference rows at the pr/100 epoch** (same tooling, same lockbox
exclusion, `work-vtx100-{base,topo}-*`; 835 events, 659 carried):
base 513/835 = 61.4 % (carried 513/659 = 77.8 %), topo3 508/835 — the topo
term was already −5 on this pool at that epoch.

## 2. The strategy table — tuning set (lockbox excluded)

All arms: toolkit `c550541f` binary, SBND production config, one knob each
via TLA; 835 label events (659 carried / 176 frozen), 219 lockbox events
excluded.  Cells are `correct @1.0 cm (%) / correct @1.5 cm (%)`.  Rows:
`p100base` = the pr/100-epoch arm on disk (reference), `base` = production
now, `nofitx` = `fit_exclusion=false`, `pre103` = pr/103+104 knobs off,
`trad` = no DL (`dl_weights=''`), `dlonly` = 3.1, `ma4` = 3.2, `topo3` =
3.3 with the term, `topk10` = admission check.

| arm | ALL n=835 1.0 / 1.5 | nuecc n=38 1.0 / 1.5 | ncpi0 n=18 1.0 / 1.5 | mcp1k n=312 1.0 / 1.5 | mcp2k n=467 1.0 / 1.5 | human n=593 1.0 / 1.5 | ai-scanner n=242 1.0 / 1.5 |
|---|---|---|---|---|---|---|---|
| p100base | 513 (61.4%) / 564 (67.5%) | 23 (60.5%) / 28 (73.7%) | 11 (61.1%) / 12 (66.7%) | 205 (65.7%) / 216 (69.2%) | 274 (58.7%) / 308 (66.0%) | 338 (57.0%) / 371 (62.6%) | 175 (72.3%) / 193 (79.8%) |
| base | 518 (62.0%) / 571 (68.4%) | 23 (60.5%) / 29 (76.3%) | 11 (61.1%) / 12 (66.7%) | 210 (67.3%) / 222 (71.2%) | 274 (58.7%) / 308 (66.0%) | 342 (57.7%) / 377 (63.6%) | 176 (72.7%) / 194 (80.2%) |
| nofitx | 653 (78.2%) / 659 (78.9%) | 30 (78.9%) / 31 (81.6%) | 14 (77.8%) / 14 (77.8%) | 241 (77.2%) / 244 (78.2%) | 368 (78.8%) / 370 (79.2%) | 422 (71.2%) / 428 (72.2%) | 231 (95.5%) / 231 (95.5%) |
| pre103 | 513 (61.4%) / 564 (67.5%) | 23 (60.5%) / 28 (73.7%) | 11 (61.1%) / 12 (66.7%) | 205 (65.7%) / 216 (69.2%) | 274 (58.7%) / 308 (66.0%) | 338 (57.0%) / 371 (62.6%) | 175 (72.3%) / 193 (79.8%) |
| trad | 479 (57.4%) / 524 (62.8%) | 16 (42.1%) / 22 (57.9%) | 4 (22.2%) / 5 (27.8%) | 200 (64.1%) / 211 (67.6%) | 259 (55.5%) / 286 (61.2%) | 316 (53.3%) / 345 (58.2%) | 163 (67.4%) / 179 (74.0%) |
| dlonly | 390 (46.7%) / 437 (52.3%) | 22 (57.9%) / 28 (73.7%) | 9 (50.0%) / 10 (55.6%) | 142 (45.5%) / 154 (49.4%) | 217 (46.5%) / 245 (52.5%) | 236 (39.8%) / 269 (45.4%) | 154 (63.6%) / 168 (69.4%) |
| ma4 | 463 (55.4%) / 511 (61.2%) | 23 (60.5%) / 29 (76.3%) | 11 (61.1%) / 12 (66.7%) | 178 (57.1%) / 189 (60.6%) | 251 (53.7%) / 281 (60.2%) | 297 (50.1%) / 328 (55.3%) | 166 (68.6%) / 183 (75.6%) |
| topo3 | 513 (61.4%) / 568 (68.0%) | 23 (60.5%) / 29 (76.3%) | 11 (61.1%) / 12 (66.7%) | 207 (66.3%) / 221 (70.8%) | 272 (58.2%) / 306 (65.5%) | 338 (57.0%) / 375 (63.2%) | 175 (72.3%) / 193 (79.8%) |
| topk10 | 518 (62.0%) / 571 (68.4%) | 23 (60.5%) / 29 (76.3%) | 11 (61.1%) / 12 (66.7%) | 210 (67.3%) / 222 (71.2%) | 274 (58.7%) / 308 (66.0%) | 342 (57.7%) / 377 (63.6%) | 176 (72.7%) / 194 (80.2%) |
IPW mcp2k-arm (n=353 weighted; do not mix with raw): p100base 78.6%/81.3%  base 78.0%/81.0%  nofitx 83.1%/83.5%  pre103 78.6%/81.3%  trad 72.1%/75.1%  dlonly 55.3%/58.4%  ma4 68.4%/71.5%  topo3 77.4%/80.4%  topk10 78.0%/81.0%

### 2.1 carried subset only (truth anchored on the production arm)
| arm | ALL n=659 1.0 / 1.5 | nuecc n=31 1.0 / 1.5 | ncpi0 n=15 1.0 / 1.5 | mcp1k n=262 1.0 / 1.5 | mcp2k n=351 1.0 / 1.5 | human n=467 1.0 / 1.5 | ai-scanner n=192 1.0 / 1.5 |
|---|---|---|---|---|---|---|---|
| p100base | 510 (77.4%) / 515 (78.1%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 205 (78.2%) / 206 (78.6%) | 271 (77.2%) / 274 (78.1%) | 336 (71.9%) / 340 (72.8%) | 174 (90.6%) / 175 (91.1%) |
| base | 518 (78.6%) / 521 (79.1%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 210 (80.2%) / 211 (80.5%) | 274 (78.1%) / 275 (78.3%) | 342 (73.2%) / 345 (73.9%) | 176 (91.7%) / 176 (91.7%) |
| nofitx | 539 (81.8%) / 541 (82.1%) | 26 (83.9%) / 26 (83.9%) | 12 (80.0%) / 12 (80.0%) | 211 (80.5%) / 212 (80.9%) | 290 (82.6%) / 291 (82.9%) | 353 (75.6%) / 355 (76.0%) | 186 (96.9%) / 186 (96.9%) |
| pre103 | 510 (77.4%) / 515 (78.1%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 205 (78.2%) / 206 (78.6%) | 271 (77.2%) / 274 (78.1%) | 336 (71.9%) / 340 (72.8%) | 174 (90.6%) / 175 (91.1%) |
| trad | 473 (71.8%) / 479 (72.7%) | 16 (51.6%) / 18 (58.1%) | 4 (26.7%) / 4 (26.7%) | 199 (76.0%) / 200 (76.3%) | 254 (72.4%) / 257 (73.2%) | 313 (67.0%) / 318 (68.1%) | 160 (83.3%) / 161 (83.9%) |
| dlonly | 387 (58.7%) / 389 (59.0%) | 22 (71.0%) / 23 (74.2%) | 9 (60.0%) / 9 (60.0%) | 141 (53.8%) / 141 (53.8%) | 215 (61.3%) / 216 (61.5%) | 235 (50.3%) / 237 (50.7%) | 152 (79.2%) / 152 (79.2%) |
| ma4 | 462 (70.1%) / 464 (70.4%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 178 (67.9%) / 178 (67.9%) | 250 (71.2%) / 251 (71.5%) | 296 (63.4%) / 298 (63.8%) | 166 (86.5%) / 166 (86.5%) |
| topo3 | 512 (77.7%) / 515 (78.1%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 207 (79.0%) / 208 (79.4%) | 271 (77.2%) / 272 (77.5%) | 337 (72.2%) / 340 (72.8%) | 175 (91.1%) / 175 (91.1%) |
| topk10 | 518 (78.6%) / 521 (79.1%) | 23 (74.2%) / 24 (77.4%) | 11 (73.3%) / 11 (73.3%) | 210 (80.2%) / 211 (80.5%) | 274 (78.1%) / 275 (78.3%) | 342 (73.2%) / 345 (73.9%) | 176 (91.7%) / 176 (91.7%) |
IPW mcp2k-arm (n=346 weighted; do not mix with raw): p100base 79.9%/81.2%  base 79.9%/81.2%  nofitx 83.3%/83.5%  pre103 79.9%/81.2%  trad 73.9%/75.5%  dlonly 56.7%/57.9%  ma4 70.1%/71.4%  topo3 79.3%/80.5%  topk10 79.9%/81.2%
wrote /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/docs/pr/105_events-tune.tsv


### 2.2 Reading the table

| strategy | verdict | evidence |
|---|---|---|
| 3.1 DL only, no re-rank | **−128** (390 vs 518); nueCC 22 vs 23 | the composite re-rank is the single largest contributor on numu; never a candidate |
| 3.2 `min_accept=4` | **−55** (463); nueCC/NCpi0 unchanged, numu −32/−23 | pr/79's 4→10 flip re-confirmed on the full pool; sweep §4: 6 → −46, 8 → −23, 12 → −7, 14 → −19 |
| 3.3 topo term (w=3) | **−5** (513); nueCC/NCpi0 unchanged | same sign as pr/89's live −8; stays OFF |
| top_k 10 | **0 events move** (0/832 differ from base) | admission ≠ recovery, exactly pr/79 §4 |
| no DL at all | −39; nueCC 16 vs 23, NCpi0 4 vs 11 | DL carries the shower samples |
| pr/103+104 off | = p100base exactly (513) | the last two rounds are +5 on this pool; they are not the regression |
| `fit_exclusion=false` | **+135 raw (653); +21 conservative** | §3 — the only lever that moves nueCC (23 → 30 raw, 23 → 26 carried) |

## 3. Attribution: `fit_exclusion` is the lever, and most of its raw gap is fit-position epoch

Raw: 518 → 653 @1 cm, but 571 → 659 @1.5, 646 → 680 @3, 681 → 702 @5 cm.
Production's 1.0→1.5 cm jump (+53) against nofitx's (+6) says most of the
gap is *where the same vertex is fitted*, not *which vertex is picked*.
Per-event decomposition (base vs nofitx main-vertex separation):

| verdict flip @1 cm | sep ≤1.5 cm | 1.5–3 cm | >3 cm |
|---|---|---|---|
| nofitx fixes (153) | 44 | 46 | **63** (42 human / 21 ai) |
| nofitx breaks (18) | 1 | 2 | **15** (all human) |

- The 44 + most of the 46 are the same vertex fitted 1.0–1.6 cm away
  (median base d = 1.21 cm).  Labels are 98.6 % `candidate`-kind — clicks
  snapped to a candidate position of their *own* epoch (pre-pr/98) — so a
  human click cannot adjudicate a ~1 cm fit shift; nofitx reproduces the
  label-epoch geometry at d = 0.00 exactly.  This is also what the 172 pr/100
  "delta" events were: nofitx scores 114/173 of the frozen bucket at 1 cm
  where production scores 0 (by construction of the carry).
- The **>3 cm flips are real selection differences**: 63 vs 15 on the full
  pool, **29 vs 15 on carried labels whose truth is anchored on production's
  own geometry** (the estimate biased *against* nofitx), 23 vs 15 human-only.
- Four-way classes (carried, tuning set): candidate-missing 45 → 32,
  net-wrong 69 → 62, selection-wrong 27 → 26; transitions 20 net-wrong→correct,
  11 candidate-missing→correct vs 10 correct→net-wrong.  Changed events are
  39 reject→reject (traditional path sees different geometry) and 26
  accept→accept (DL snaps onto differently-placed candidates): a
  candidate-graph effect on both routes, not a scorer effect.
- Conservative bottom line (carried): **518 → 539 (+21); nueCC 23 → 26,
  NCpi0 11 → 12, mcp1k 210 → 211, mcp2k 274 → 290**; IPW mcp2k 79.9 → 83.3 %.

### 3.1 What `fit_exclusion` OFF would cost elsewhere (why it is an attribution, not a candidate)

`scripts/pr90_movers.py base nofitx --tags vtx105`: 89 labels move toward
the click but **49 ADVERSE** (off the click by >1 cm: nueCC 1, NCpi0 2,
mcp1k 22, mcp2k 24; `docs/pr/105_movers-nofitx-<sample>.tsv`), and the
downstream physics moves with it (e.g. nueCC 122660: vertex 5.7 → 0.3 cm
but nue score 4.30 → −15; `docs/pr/105_scores-nofitx-<sample>.tsv`).
`fit_exclusion` is the prototype's `flag_exclusion` (pr/98 §1: per-plane 2-D
charge arbitration feeding `multi_trajectory_fit` and dQ/dx; 28/30
prototype call sites pass `true`).  **Owner decision 2026-08-21: production
is not changed and `fit_exclusion` stays ON.**  The number this section
establishes is the *size* of the selection cost of that feature on data
labels (+21 conservative / +135 raw at 1 cm), so a future round can weigh a
narrower form (e.g. exclusion in the post-selection fit only) against it.

## 4. Strategy 3.4 — re-rank fine-tune (offline search, outcome-calibrated ruler)

Ruler: `rerank_tune.py` (pr/89 §13.4) with the outcome map fed by **every
selection-only arm of this round** (`--arm-names base,topo3,dlonly,ma4,topk10,trad`,
rows from `topo3`), lockbox excluded, nueCC guarded (`--guard-samples nuecc48`).
Closure at production θ: **0 mismatches** vs the live base arm (832 events,
831 with snapped candidates); production = 520/832, **0 uncovered**.

| θ (rest = production) | correct @1 cm | Δ | nueCC | uncovered |
|---|---|---|---|---|
| production (10 / 1000 / all weights 1 / topo 0) | 520 | — | 23/38 | 0 |
| `min_accept` 6 · 8 · 12 · 14 | 474 · 497 · 513 · 501 | −46 · −23 · −7 · −19 | 23 each | 0 · 0 · 23 · 44 (4-arm coverage) |
| `min_accept` 12 (6-arm coverage) | 522 | +2 | 23 | 0 |
| `scale` 750 | 522 | +2 | 23 | 0 |
| `min_accept` 12 + `scale` 750 | 524 | +4 | 23 | 0 |
| `w_isol` 1.5 · `w_snap` 0.5 (constexpr) | 520 · 520 | 0 · 0 | 23 | 0 |
| coordinate-ascent best: 12 / 750 / `w_isol` 1.5 / `w_snap` 0.5 | 526 | +6 (6 mcp2k fix, 1 mcp1k fix / 1 break) | 23 | 0 |
| topo `w` 0.5 (4-arm run) | 521 | +1 | 23 | 0 |

Same search on the `nofitx` graph (rows/outcomes from that arm alone):
production θ = 650/832, **no move improves it**.  Reading: +6/832 is inside
the noise the pr/89 live A/Bs measured (offline +12 → live −8; +3 → −36),
touches two `constexpr` weights (C++), and moves **zero** nueCC events.  Per
the pre-registered outcome (pr/100 §4 item 2): **production is the optimum
of the selection stack at this epoch — no candidate is built.**

## 5. The lockbox read (once; 219 events; `runs/vtx105/lockbox-read-20260821.md`)

Read 2026-08-21 after §2–§4 were frozen, all arms at once, no selection made
on it.  All 219 lockbox events carried (the box was drawn from carried
events), so this is the anchored-truth estimand.

| arm | ALL n=219 1.0 / 1.5 | nuecc n=9 1.0 / 1.5 | ncpi0 n=1 1.0 / 1.5 | mcp1k n=95 1.0 / 1.5 | mcp2k n=114 1.0 / 1.5 | human n=162 1.0 / 1.5 | ai-scanner n=57 1.0 / 1.5 |
|---|---|---|---|---|---|---|---|
| p100base | 166 (75.8%) / 167 (76.3%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 71 (74.7%) / 71 (74.7%) | 89 (78.1%) / 90 (78.9%) | 111 (68.5%) / 112 (69.1%) | 55 (96.5%) / 55 (96.5%) |
| base | 167 (76.3%) / 168 (76.7%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 71 (74.7%) / 71 (74.7%) | 90 (78.9%) / 91 (79.8%) | 111 (68.5%) / 112 (69.1%) | 56 (98.2%) / 56 (98.2%) |
| nofitx | 172 (78.5%) / 172 (78.5%) | 9 (100.0%) / 9 (100.0%) | 1 (100.0%) / 1 (100.0%) | 73 (76.8%) / 73 (76.8%) | 89 (78.1%) / 89 (78.1%) | 115 (71.0%) / 115 (71.0%) | 57 (100.0%) / 57 (100.0%) |
| pre103 | 166 (75.8%) / 167 (76.3%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 71 (74.7%) / 71 (74.7%) | 89 (78.1%) / 90 (78.9%) | 111 (68.5%) / 112 (69.1%) | 55 (96.5%) / 55 (96.5%) |
| trad | 154 (70.3%) / 155 (70.8%) | 5 (55.6%) / 5 (55.6%) | 1 (100.0%) / 1 (100.0%) | 68 (71.6%) / 69 (72.6%) | 80 (70.2%) / 80 (70.2%) | 107 (66.0%) / 108 (66.7%) | 47 (82.5%) / 47 (82.5%) |
| dlonly | 120 (54.8%) / 121 (55.3%) | 5 (55.6%) / 5 (55.6%) | 0 (0.0%) / 0 (0.0%) | 47 (49.5%) / 47 (49.5%) | 68 (59.6%) / 69 (60.5%) | 70 (43.2%) / 71 (43.8%) | 50 (87.7%) / 50 (87.7%) |
| ma4 | 135 (61.6%) / 136 (62.1%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 53 (55.8%) / 53 (55.8%) | 76 (66.7%) / 77 (67.5%) | 84 (51.9%) / 85 (52.5%) | 51 (89.5%) / 51 (89.5%) |
| topo3 | 162 (74.0%) / 163 (74.4%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 69 (72.6%) / 69 (72.6%) | 87 (76.3%) / 88 (77.2%) | 107 (66.0%) / 108 (66.7%) | 55 (96.5%) / 55 (96.5%) |
| topk10 | 167 (76.3%) / 168 (76.7%) | 6 (66.7%) / 6 (66.7%) | 0 (0.0%) / 0 (0.0%) | 71 (74.7%) / 71 (74.7%) | 90 (78.9%) / 91 (79.8%) | 111 (68.5%) / 112 (69.1%) | 56 (98.2%) / 56 (98.2%) |
IPW mcp2k-arm (n=114 weighted; do not mix with raw): p100base 79.9%/80.7%  base 80.6%/81.3%  nofitx 76.0%/76.0%  pre103 79.9%/80.7%  trad 70.1%/70.1%  dlonly 55.6%/56.4%  ma4 69.2%/70.0%  topo3 78.4%/79.2%  topk10 80.6%/81.3%
wrote /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/docs/pr/105_events-lockbox.tsv

Every ordering of §2 reproduces: base 167 ≥ topo3 162 > trad 154 > ma4 135
> dlonly 120; topk10 = base; pre103 = p100base; nofitx 172 (+5; nueCC 9/9
vs 6/9 — 9 events, read as a sign, not a rate).  The lockbox is now SPENT
for selection purposes; a future round seals a fresh one (`vtx_prep.sh`
seeds from the tag).

## 6. Decision and what ships

- **No production change.**  Owner 2026-08-21: "we should not change the
  production yet … we do not want to change fit_exclusion off."  The
  selection stack (DL top-5 → 7-term composite → `min_accept` 10, topo OFF)
  is the measured optimum on 835 + 219 data labels; every alternative in
  §2 is worse or inert.
- Ships: the `dl_vtx_rerank` cfg thread (default true, compiled config
  byte-identical — lets 3.1 be re-run), `TAGS_VTX105`, the lockbox flags in
  `rerank_tune.py` / `vtx_report.py`, `vtx_strategy_table.py`, the
  `vtxscan-vtx105-*` carried labels, and this doc with its TSVs.
- Not run: a full-sample live validation (nothing to validate) and a Bee
  package (no movers to review — the `nofitx` fixes/adverse lists are in
  the TSVs if a scan is wanted).

## 7. nueCC — where the 15 misses are (tuning set, 38 labels)

| evt | d_prod (cm) | class | note |
|---|---|---|---|
| 30504 | 74.8 | candidate-missing | graph: no candidate at the click |
| 10550 | 47.4 | net-wrong | DL heat elsewhere; trad 35.8 |
| 111412 | 40.5 | candidate-missing | nofitx 0.80, trad/dlonly 2.45 |
| 163543 | 37.4 | candidate-missing | pr/104 residual (wrong object) |
| 235435 | 30.8 | net-wrong | nofitx 0.18 |
| 52672 | 30.6 | candidate-missing | nofitx 3.85 |
| 38856 | 22.8 | net-wrong | |
| 268067 · 437699 · 168596 · 234638 | 1.2–2.7 | frozen | same vertex; nofitx = 0.00 (fit-position) |
| 271851 · 360535 · 423981 | 1.2–2.6 | frozen | same vertex, 1–3 cm |
| 469665 | 1.41 | selection-wrong | nofitx 0.44 |

Seven are real misses (>20 cm): 4 candidate-missing (a graph problem — no
re-rank can reach them) and 3 net-wrong (the uBooNE net's heat is on the
wrong object).  Eight are the same vertex 1.2–2.7 cm off — the
fit-position class of §3.  With `fit_exclusion` ON and the scorer at its
optimum, the nueCC levers left are (a) the candidate graph on 30504 /
111412 / 163543 / 52672 (four named events for a PR round), (b) a
post-selection position refinement for the 1–3 cm class (pr/89 §13.5,
pr/100 §4 — still the un-spent lever), (c) label volume / DL retrain
(owner-excluded).

## 8. Known limits

- Labels are reco-anchored at their own epoch (98.6 % `candidate` picks);
  a 1 cm tolerance cannot separate a fit-position change from a selection
  change — §3's >3 cm flip counts are the robust part of the fit_exclusion
  number, the +135 raw is not.
- The lockbox holds 9 nueCC / 1 NCpi0 events; nueCC verdicts rest on the
  tuning set's 38.
- The outcome-calibrated ruler had 0 uncovered decisions at every θ quoted
  (six-arm coverage); the 4-arm `min_accept` 12/14 rows carry 23/44
  uncovered and are pessimistic.
- `work-vtx100-{base,topo}-*` were untagged in `docs/work-tags.md` until
  this round (now **KEEP**).

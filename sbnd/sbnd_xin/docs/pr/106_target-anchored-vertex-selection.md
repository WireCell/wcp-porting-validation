# doc pr/106 — target-anchored re-optimization of the DL main-vertex selection

**Status: CLOSED 2026-08-21 — production UNCHANGED. On the target metric production is the best of the four strategies (tuning 510/673, lockbox 259/339); the re-rank surface is flat (scan + joint fit both ≤ +11); the one lever, stricter acceptance (`min_accept` 10→20), was live-validated (+13/1012, closure 0) but carries 10 ADVERSE 1-cm movers incl. a nueCC48 nue-selection loss → NOT flipped, owner's call. nueCC is blocked by the heat map (8 net-blind + 4 confidently-wrong of 12 misses), not by θ; ft2u does not change nueCC admission.**

Owner (2026-08-21), after doc pr/105 closed with "production optimal":
the pr/105 ruler was wrong for the question.  The hand-scan labels were
taken on `fit_exclusion`-OFF reconstructions, so a 1 cm match of the
*current* fitted vertex to the click is biased against the current
(`fit_exclusion` ON) sample — pr/105 §3 measured exactly that.  The cut, not
the selection, failed.  New metric: on the **current topology** every event
has a finite candidate-vertex set; the **target** = the candidate closest to
the click (no cut); a selection method scores by whether it **picks the
target**.  Compare (1) DL alone, (2) re-rank with the topo term, (3) re-rank
without it (= production), (4) re-tuned weights.  All offline; implement +
live-validate only the final optimum.  Two owner constraints: "if we do not
have the vertices before the DL vertex, we should dump them", and "the
off-line optimization must be consistent with the code".

## 0. Repro block

```bash
cd sbnd_xin   # toolkit c550541f binary (no C++ change this round)

# 1. harvest re-dump over the 1054-label universe (lists /home/xqian/tmp/vtx105-evts-<s>.txt,
#    47/19/407/581).  dl_vtx_harvest (pr/79 sec 10) records the EXACT live SCN input cloud
#    whose first n_vertex_rows points are every pre-DL candidate vertex.  /home/xqian/tmp/pr106_arms.sh:
SBND_DL_VTX_HARVEST=true PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-<s>-ql0819 work-vtx106-harv-base-<s> data <evts>
SBND_DL_VTX_TOPO_WEIGHT=3.0 SBND_DL_VTX_HARVEST=true ... work-vtx106-harv-topo3-<s>   # rows carry s_topo

# 2. harvest neutrality: scripts/pr85_hash_gate.py work-vtx105-base-<s> work-vtx106-harv-base-<s>
#    + calib JSON identical after dropping hv_*/harvest keys (sec 1.1)

# 3. lockbox sealed BLIND (every 3rd event of the universe sorted by (sample, evt)):
#    dl_vtx_training/runs/vtx106/{lockbox.txt (351), tuning.txt (703)}

# 4. evaluator (tuning set): closure -> tables -> sweep -> search -> ledgers
T="--carried-tags vtxscan-vtx105-{nuecc48,ncpi0,mcp1k,delta,mcp2k,mcp2k-auto,mcp2k-ragree} \
   --orig-tags vtxscan-harv3-{nuecc48,ncpi0,mcp1k,delta} vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree"
python3 dl_vtx_training/vtx_target_eval.py $T --exclude-events dl_vtx_training/runs/vtx106/lockbox.txt \
   --closure --table --sweep docs/pr/106_sweep.tsv --search \
   --events-tsv docs/pr/106_events-tune.tsv --miss-ledger docs/pr/106_miss-ledger-tune.tsv

# 4b. joint fit (sec 4.2) and checkpoint admission (sec 7.1)
python3 dl_vtx_training/rank_fit_target.py $T --exclude-events dl_vtx_training/runs/vtx106/lockbox.txt --lam 0.1 1 10 30 100 300 1000 3000
python3 dl_vtx_training/ckpt_target_admission.py --events-tsv docs/pr/106_events-{tune,lockbox}.tsv \
   --ckpt CP24=<wire-cell-data>/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth ft2u=<wire-cell-data>/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth \
   --jobs 12 --tsv docs/pr/106_ckpt-admission.tsv

# 5. ONE lockbox read (sec 5): runs/vtx106/lockbox-read-20260821.md (tables + 3 pre-declared thetas)

# 6. live validation arm (sec 6), closure + ledgers
SBND_DL_VTX_MIN_ACCEPT=20 PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<s>-ql0819 work-vtx106-ma20-<s> data <evts>
python3 dl_vtx_training/vtx_target_eval.py $T --live-arms <DEFAULT_LIVE> ma20=work-vtx106-ma20-{sample} --exclude-events ... --closure --table
python3 scripts/pr90_movers.py work-vtx105-base-<s> work-vtx106-ma20-<s> --tags vtx105 --tsv docs/pr/106_ma20/movers-<s>.tsv
python3 scripts/pr83r3_scores_ab.py work-vtx105-base-<s> work-vtx106-ma20-<s> --tsv docs/pr/106_ma20/scores-<s>.tsv
```

## 1. Setup

### 1.1 The candidate set is the pre-DL cloud, not `vertices[]`

`calib-pr-evt*.json` `vertices[]` is written after `snap_main_vertex_to_kink`
/ `improve_vertex` / shower reclustering: ids renumber and vertices
appear/disappear (nueCC48 163543: 5 of 124 pre-DL candidate ids are absent
from `vertices[]`; pr/105 saw 9/119 on 269774).  The `dl_vtx_harvest` knob
(pr/79 §10, `NeutrinoVertexFinder.cxx:4802-4824`, writer
`PrDisplayDump.cxx:796-847`) records the cloud the net actually consumed:
`hv_cloud.{x,y,z,q,vertex_ids,n_vertex_rows}` — the first `n_vertex_rows`
points are **every** candidate vertex (`cand_vertices`, `:4771-4787`) with
their pr75 ids (`cluster_id*1000 + graph_index`).  Every scoreboard row's id
is in the cloud (63/63 on 163543).  The old harvest arms were retired, so
the universe was re-dumped (step 1).

Neutrality: `pr85_hash_gate.py` PASS on every sample (nueCC48 94/94
archives, NCpi0 38/38, mcp1k 814/814, mcp2k 1162/1162); calib JSON identical after
dropping the `hv_*`/`harvest` keys (both scoreboard copies).

### 1.2 Target and metric

- truth = the carried label position where the event carried onto the
  pr/105 base arm (878/1054, TOL 1.0), else the frozen click (pr/105 §1).
- **target** = argmin over the pre-DL cloud vertex rows of |xyz − truth|;
  `d_target` recorded.  Primary universe = `d_target ≤ 3 cm` (the click is
  on a candidate); 3–10 cm = secondary band reported beside; >10 cm =
  candidate-missing (no selection method can win).
- a method's **pick** = the candidate it selects; hit iff pick == target
  (id identity on the same pre-DL cloud).  Live-arm picks resolve by
  `final_vertex_id` when it is a cloud id, else by position ≤ 1.5 cm to the
  cloud (post-refit renumbering).
- **admission**: target ∈ the snapped rows (top-5; top-10 via the topk10
  arm).  Non-admitted targets are unreachable by every re-rank variant.

### 1.3 Consistency contract (offline == code)

Offline decision space = exactly the live one: only `dl_snapped` rows; the
dumped `s_*` terms (only `s_dl = dl_score·scale` and `s_topo = w·(frac −
center)` are recomputed, both knob-linear); `accept iff best ≥ min_accept`;
the pr/48 protected veto as dumped (`route = dl-veto-protected` ⇒ any accept
resolves to the veto vertex).  A REJECT resolves to the live fallback vertex
of an arm that rejected on that event (the no-DL `trad` arm, route
`dl-not-run`, covers every event; base/topo3/ma4/dlonly/topk10 rejects agree
where present); an uncovered decision counts WRONG.  Closures (offline
decision == live route/`dl_winner`, every label event) are required to be 0
before any number is quoted — §2.

Evaluator: `dl_vtx_training/vtx_target_eval.py` (new; reuses
`vtx_strategy_table.load_truth`, does not touch `rerank_tune.py`).

## 2. Closure — offline decision == live code

All label events (tuning + lockbox, 1050 with a harvest dump; 1 empty cloud
= net never ran, 2 without a dump):

| live arm | offline rule | mismatches |
|---|---|---|
| harv-base, base | production θ | **0 / 0** |
| harv-topo3, topo3 | θ + `w_topo=3, center=0` | **0 / 0** |
| ma4 | θ, `min_accept=4` | **0** |
| topk10 | θ on the topk10 rows | **0** |
| dlonly | top voxel + 2.5 cm cut | 8 (4 tuning + 4 lockbox): the legacy branch's direction/proton sanity check and its own snap are not replayed — M1 is therefore read from the **live** dlonly arm |

Snapped-row set and `dl_score` identical between harv-base and harv-topo3
on every event (0 mismatches).  Live-pick resolution onto the pre-DL cloud
(all arms × events): exact `dl_winner` id 4964, `final_vertex_id` 3165,
position ≤ 1.5 cm 176, ≤ 3 cm unambiguous 56, unresolved 55 (1 %, counted
as misses).  Reject fallback: on all 160 base-reject events of
nueCC48/NCpi0/mcp1k the base final equals the no-DL `trad` arm final to
0.00 cm — the trad arm *is* the fallback; `hv_trad_main_vertex_id` is only
the pre-fallback state (the fallback re-picks within the main cluster on
23/160).  Reject coverage 661/673 (tuning), 336/339 (lockbox).

## 3. Target tables — tuning set (703 ids; 701 with dumps)

`d_target` (click → nearest pre-DL candidate): ≤1 cm 560, 1–3 cm 113,
3–10 cm 26, >10 cm 2.  Primary universe **673**.  Admission of the target:
top-5 **456/673 (67.8 %)**, top-10 531/673 (78.9 %).

| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| M1 DL alone (live dlonly arm) | 379/673 (56.3%) | 22/32 | 8/12 | 131/264 | 218/365 | 54.7% | 228/476 | 151/197 |
| M2 re-rank + topo (w=3) | 504/673 (74.9%) | 23/32 | 9/12 | 196/264 | 276/365 | 77.1% | 328/476 | 176/197 |
| **M3 re-rank, no topo = PRODUCTION** | **510/673 (75.8%)** | 23/32 | 9/12 | 200/264 | 278/365 | 77.9% | 334/476 | 176/197 |
| production (live base arm) | 510/673 (75.8%) | 23/32 | 9/12 | 200/264 | 278/365 | 77.9% | 334/476 | 176/197 |
| min_accept 4 (live ma4) | 447/673 (66.4%) | 23/32 | 9/12 | 163/264 | 252/365 | 68.7% | 285/476 | 162/197 |
| top_k 10 (live topk10) | 510/673 (75.8%) | 23/32 | 9/12 | 200/264 | 278/365 | 77.9% | 334/476 | 176/197 |
| no DL, traditional (live trad) | 466/673 (69.2%) | 18/32 | 5/12 | 192/264 | 251/365 | 70.2% | 310/476 | 156/197 |

(secondary band 3–10 cm, n=26: M1 10, M2 13, M3 15, ma4 11, trad 13.)

Production misses (163): **reject-to-wrong 89**, not-admitted 41 (DL
accepted a candidate while the target was outside the top-5),
wrong-accept 22, veto-to-wrong 8, uncovered 3.

Anatomy:
- of the 89 reject-to-wrong, the target was in the top-5 in only **33**
  (and DL's own top voxel sat on it in 5); 77 − 24 = 53 were not even in the
  top-10.  These are DL misses + a wrong traditional fallback.
- of the 27 wrong/veto accepts with the target admitted
  (`106_miss-ledger-tune.tsv`), the losing term is **`s_dl` in 25/27**
  (the heat map itself prefers the wrong candidate, by 190–560 score units on
  the four nueCC cases 122660/268067/360535/389538); `s_snap` 2; a
  geometric term 0.
- oracle bound over the rows the code can see: 510 + 33 + 27 = 570/673
  (84.7 %).  Everything above that needs a better heat map or a better
  fallback, not a better re-rank.

## 4. Weight re-optimization (tuning set, offline, `106_sweep.tsv`)

1-D sweeps from production θ (hits / 673; production 510):

| knob | values → hits |
|---|---|
| `w_snap` | 0→508, 0.5–1.5→510, 2–3→509 |
| `w_fwd_z` | 0…3 → 510 (inert) |
| `w_clen` | 0→512, 0.5→512, 1→510, 1.5→497, 2→481, 3→458 |
| `w_isol` | 0→509, 0.5–3→510 |
| `w_main` | 0→512, 0.5→512, 1→510, 1.5→496, 2→479, 3→452 |
| `w_fv` | 0→511, 0.5→511, 1→510, 1.5→506, 3→496 |
| `w_topo` | 0→510, 1→509, 2→509, 3→504, 5→504 (center 0.25→508, 0.5→510 at w=3) |
| `min_accept` | 0→389, 4→440, 6→453, 8→480, **10→510, 12→511, 15→513, 20→514** |
| `scale` | **500→514**, 750→511, 1000→510, 1500→477, 2000→448 |

No knob moves more than +4; the topo term is ≤ 0 everywhere; the only
monotone direction is *stricter acceptance* (`min_accept` ↑ or `scale` ↓,
both = "trust low-confidence DL accepts less, let the fallback decide").
Coordinate ascent with a nueCC guard: 516 (`w_clen=0, w_main=0, w_topo=1,
min_accept=20`); 8 seeded random restarts reach 516–521 from mutually
unrelated θ (best 521 = `w_snap 3, w_isol 3, w_fv 0.5, w_topo 5,
min_accept 12, scale 500`) — a flat, degenerate surface, +11 at most.  The
same search on the top-10 rows: 518 (admission +75 targets buys nothing:
the extra candidates never win).


### 4.2 Joint multi-dimensional fit (owner ask: "smarter than scanning") — `dl_vtx_training/rank_fit_target.py` (pr/77's `rank_fit.py`, an 11-feature RankNet on the 1 cm ruler, is left untouched)

The live scorer is linear in its knobs, so "which candidate wins" is a
pairwise-ranking problem and "accept or fall back" a threshold on the
winner.  `rank_fit_target.py` fits all nine coefficients at once by L2-regularised
pairwise logistic regression on every (target, other admitted candidate)
pair of the tuning set (456 admitted-target events → 505 pairs), shrunk
toward production θ with strength λ, then picks `min_accept` by a 1-D scan
of the fitted scorer; 5-fold CV (seed 106) reports generalisation.  Same
decision space as the code (`vtx_target_eval.decide`), so every fit is
directly replayable.

| λ | 5-fold CV hits | in-sample | fitted θ (scale, snap, fwd_z, clen, isol, main, topo, center; min_accept) |
|---|---|---|---|
| 0.1 | 453 | 453 | 2.8e5, 155, 51, 1960, 2270, 2270, 895, −1.3; 0 — unregularised: blows up, over-fits |
| 1 | 419 | 453 | 2.9e4, 25, 5.6, 188, 221, 223, 88, −1.3; 0 |
| 10 | 448 | 450 | 680, −7.5, 1.0, 17.6, 22.8, 22.2, 6.3, −1.05; 40 |
| 30 | **516** | 517 | 893, −2.0, 1.0, 7.0, 8.9, 8.8, 2.3, −1.07; **50** |
| 100 | **516** | 516 | 968, 0.07, 1.0, 2.9, 3.5, 3.4, 0.7, −1.09; **30** |
| 300 | 516 | 517 | 989, 0.69, 1.0, 1.6, 1.8, 1.8, 0.24, −1.09; 25 |
| 1000 | 515 | 516 | 997, 0.91, 1.0, 1.2, 1.25, 1.25, 0.07, −1.09; 25 |
| 3000 | 515 | 516 | ≈ production; 25 |
| production | — | 510 | 1000, 1, 1, 1, 1, 1, 0, 0; 10 |

Pairwise accuracy of production θ on the 505 pairs is only **0.596**; no
fit exceeds 0.63 — the pairs are not linearly separable in these terms
because `s_dl` dominates and is itself wrong on most of them (§3).  The fit
therefore cannot beat the scan: CV saturates at 516 (+6) for every λ ≥ 30,
and every regularised solution reaches its gain the same way — geometric
weights within ~2× of production and a **higher accept threshold (25–50)**.
(`w_fv` never moves: within an event the target and its rivals share the
FV flag, so the pair difference is 0.)  A pairwise objective also ignores
the fallback outcome, which is where the accept/reject knob earns its
+4–+8 — hence the threshold is scanned, not fit.  Conclusion unchanged:
the selection surface is flat; the one robust lever is acceptance
strictness.

## 5. Lockbox read (351 ids; 349 with dumps; primary 339) — `runs/vtx106/lockbox-read-20260821.md`

Read once, 2026-08-21 19:26 UTC, θ pre-declared: C1 = best search θ (521),
C2 = `scale=500` (514), C3 = `min_accept=15` (513).

| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| M1 DL alone (live dlonly) | 201/339 (59.3%) | 12/15 | 4/7 | 73/130 | 112/187 | 58.6% | 127/245 | 74/94 |
| M2 re-rank + topo | 254/339 (74.9%) | 12/15 | 5/7 | 100/130 | 137/187 | 76.3% | 171/245 | 83/94 |
| **M3 = PRODUCTION** (offline = live) | **259/339 (76.4%)** | 12/15 | 5/7 | 101/130 | 141/187 | 77.6% | 174/245 | 85/94 |
| min_accept 4 (live) | 229/339 (67.6%) | 12/15 | 5/7 | 90/130 | 122/187 | 66.1% | 151/245 | 78/94 |
| top_k 10 (live) | 259/339 (76.4%) | 12/15 | 5/7 | 101/130 | 141/187 | 77.6% | 174/245 | 85/94 |
| no DL (live trad) | 244/339 (72.0%) | 9/15 | 3/7 | 97/130 | 135/187 | 74.1% | 165/245 | 79/94 |
| C1 best search θ | 267/339 (78.8%) | 12/15 | 5/7 | 103/130 | 147/187 | 81.1% | 181/245 | 86/94 |
| C2 `scale=500` | 267/339 (78.8%) | 12/15 | 5/7 | 103/130 | 147/187 | 81.1% | 181/245 | 86/94 |
| C3 `min_accept=15` | 266/339 (78.5%) | 12/15 | 5/7 | 103/130 | 146/187 | 80.8% | 180/245 | 86/94 |

The lockbox reproduces the tuning ordering exactly (M3 > M2 > trad > ma4 >
M1; top-10 = M3) and the candidates collapse onto one another: the whole
gain of C1 is its stricter acceptance, not its re-weighting.  Combined
(tuning + lockbox, 1012 events): production 769, C2 781 (**+12, +1.2 pp**;
nueCC 35 → 36, NCpi0 14 → 14, mcp1k 301 → 303, mcp2k 419 → 428).

Mechanism (C2 vs production, both sets): **pure accept→reject** — 84 events
fall back to the traditional vertex, 16 fixed / 4 broken (47278, 175896,
315497, 283299 — all numu), 64 same outcome.  `min_accept=20` takes the
identical decision on 986/1012 events and is an existing TLA
(`SBND_DL_VTX_MIN_ACCEPT`), whereas `dl_vtx_score_scale` is pinned in
`clus.jsonnet` — so the live validation candidate is `min_accept 10 → 20`.

## 6. Live validation of `min_accept 10 → 20` — `work-vtx106-ma20-<s>` (env `SBND_DL_VTX_MIN_ACCEPT=20`, 1054/1054 rc=0)

**Closure: 0 mismatches** on every label event (tuning and lockbox) — the
offline prediction is the code.  Target metric, live:

| set | production | min_accept 20 | Δ | nueCC48 | NCpi0 | mcp1k | mcp2k |
|---|---|---|---|---|---|---|---|
| tuning (673) | 510 | 514 | +4 | 23→24 | 9→10 | 200→198 | 278→282 |
| lockbox (339) | 259 | 268 | +9 | 12→12 | 5→5 | 101→104 | 141→147 |
| combined (1012) | 769 (76.0 %) | 782 (77.3 %) | **+13** | 35→36 | 14→15 | 301→302 | 419→429 |

Production-safety checks against `work-vtx105-base-<s>` (`106_ma20/`):

- `pr90_movers.py --tags vtx105` (the 1 cm click metric): movers 3 / 1 / 8 /
  17 per sample; **ADVERSE 10** — nueCC48 **46363** (45→86 cm) and
  **235435** (31→37 cm, both already wrong), mcp1k **315497** (0.00→56 cm),
  **409546** (1.9→18 cm), **175016** (0.00→4.6 cm), mcp2k **173620**
  (15→102 cm), **395277** (10→68 cm), **321015** (0.00→29 cm), **47278**
  (0.00→27 cm), **283299** (0.00→24 cm); toward: nueCC48 111412 (41→2.5 cm),
  NCpi0 84229 (99→0.0), mcp1k 282899/286681/282385/278046, mcp2k 323369/
  497311/389379/407886/65266/97062.
- `pr83r3_scores_ab.py`: nueCC48 3 movers — **46363 nue 4.30→−3.44**
  (loses the nue selection), 111412 nue 0.04→**4.30** (gains it), 235435
  nue −15→−4.30; NCpi0 84229 nue 0.69→−15 (a background leaving the nue
  selection), 259542 nue −1.08→−4.20; mcp1k 13 rows, mcp2k 26 rows (numu
  score / Enu movers, several cosmict flips both ways).

Verdict: a real but marginal net gain (+13/1012 on the target metric,
+1 nueCC) bought with 10 ADVERSE movers on the 1 cm bar, among them five
clean breaks off the click (315497, 175016, 321015, 47278, 283299) and one nueCC48 event losing its nue
selection while another gains one.  Under the round rules ADVERSE is the
stop-the-line class, so **not flipped**; the knob, its arms and both ledgers
are on disk for the owner's decision.  Note the two metrics disagree on
purpose: the target metric counts a fallback that lands on a *different*
wrong candidate as neutral, the 1 cm metric counts any move off the click
as ADVERSE.

## 7. nueCC: what blocks more correct vertices, and the checkpoint question

nueCC48 (47 labeled, all with the click on a candidate): production 35
correct, 12 misses, in two modes (`106_events-*.tsv`, harvest rows):

| mode | events | evidence |
|---|---|---|
| **the net never proposes the vertex** — 8 | 10550, 38856, 52672, 111412, 235435, 30504, 163543 (target outside the top-5; only 10550/235435 inside the top-10) + 46363 (rank 3, dl 0.007) | best voxel score in the event 0.006–0.022, composite 4–26: the heat map found nothing vertex-like; production still *accepts* the weak wrong voxel in 6 of the 8 because `s_clen+s_main+s_fv` alone reach 4.5 (the pr/79 `min_accept` 4→10 flip helped here and 20 helps once more: 111412) |
| **the net confidently prefers another candidate** — 4 | 122660 (target rank 1, dl 0.81 vs winner 1.00), 268067 (0.35 vs 0.91), 360535 (0.74 vs 0.99), 389538 (0.75 vs 0.98) | gap 190–560 composite units against geometric terms of O(1); no weight in a sane range overturns it, `scale` 1500 already costs −33 on numu. The traditional path was right on 122660/389538 |

So for nueCC the re-rank is inert by construction: O(1) terms on top of an
O(1000) DL term that is either absent or confidently wrong.  The levers are
the heat map itself and a shower-aware fallback (mode 1: best dl < ~0.03 ⇒
the net abstained), not θ.  Action items added at the owner's request
(2026-08-21): §4.2 (joint fit, done) and §7.1 below.

### 7.1 Surviving fine-tuned checkpoint on the target metric

Inventory (pr/77–81): every fine-tuned `.pth` from the `runs/*/fold*/`
sweeps was deleted on 2026-08-16 except the ft2u deployment copy
`wire-cell-data/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth` (pr/78 §7,
pr/79 §3: REJECTED live −40/473 on the 1 cm ruler); hft1 (pr/79 §11b) and
hr3 (pr/81 §B) are not reconstructible without retraining; pr/81 Phase A/C
produced no weights.  The harvest clouds now make a clean offline test
possible: they are the exact live net input (pyutil `SCN_Vertex`
reproduces the recorded top-K bit-exactly), so target **admission** and the
**DL-alone pick** — the weight-independent quantities, and exactly where
nueCC is blocked — can be scored offline without the rebuilt-cloud parity
caveat of pr/77–81.  `dl_vtx_training/ckpt_target_admission.py`
(`106_ckpt-admission.tsv`, 1012 events, `--jobs 12`; CP24 top-5 ==
recorded live voxels on **1012/1012**):

| checkpoint | nueCC48 top1 / adm5 / adm10 (n=47) | NCpi0 (19) | mcp1k (394) | mcp2k (552) | ALL (1012) |
|---|---|---|---|---|---|
| CP24 (production) | **35 / 40 / 42** | 12 / 14 / 14 | 185 / 254 / 300 | 304 / 379 / 435 | 536 / 687 / 791 |
| ft2u (pr/78 deploy, rejected live on the 1 cm ruler) | **35 / 40 / 42** | 13 / 14 / 14 | 190 / 261 / 308 | 309 / 387 / 437 | 547 / 702 / 801 |

ft2u moves the numu samples slightly in the right direction (+11 top-1, +15
admission@5 — consistent with its pr/79 "−40" being mostly the 1 cm-ruler
fit-epoch artefact this round exposed) and leaves nueCC48 **exactly
unchanged** on every count: a fine-tune on numu-dominated labels does not
learn the nueCC shower vertex.  Full re-rank replay for ft2u is not possible
offline (the geometric terms exist only for candidates CP24 snapped); a live
arm would be the next step if the owner wants ft2u re-adjudicated on the
target metric.  The nueCC blocker stands: 7 of the 12 misses are events where
no checkpoint proposes the vertex at all — those need nueCC-specific training
labels (the 47 data nueCC + MC nueCC) or a shower-start-aware candidate
proposal outside the net.

## 8. Residuals / action items

1. Owner decision on `min_accept 10→20` (§6): +13/1012 target hits vs 10 ADVERSE
   1 cm movers and one nueCC48 nue-selection loss.  Cfg-only (existing TLA).
2. nueCC heat-map blindness (§7): 7 of 12 misses have no vertex-like voxel;
   needs nueCC labels in the SCN training set, or a shower-aware fallback
   (accept only if best dl ≥ ~0.03, else the traditional vertex) — the
   latter is a one-knob C++ change that this round's data can pre-score.
3. Re-adjudicate ft2u live on the target metric (§7.1) if wanted.
4. `hv_trad_main_vertex_id` is the pre-fallback state, not the fallback
   outcome (§2) — a harvest of the post-fallback vertex id would remove the
   position-resolution tier (55 unresolved picks).

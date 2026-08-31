# doc pr/100 — neutrino-vertex situation report + re-rank tuning plan

Owner asked three things: (1) summarize what the many vertex-adjacent
campaigns actually changed in the code; (2) prepare the data and evaluation
code so the existing hand-scan labels stay usable as pattern recognition
keeps moving; (3) write the situation + a plan for a *final slight tuning of
the re-rank* — explicitly not a DL retrain — for once the PR chain is
declared final. No toolkit C++ or jsonnet change is part of this doc.

## Repro block

```bash
cd sbnd_xin

# 0. freshness proof (M1) -- done this round at toolkit 8573877f
wcbuild; ./build/clus/wcdoctest-clus                # 215/215

# 1. labeled-universe event lists per sample (vtx_rules/vtx_io tags, never glob)
python3 -c "... see vtx_prep.sh step 1 ..."

# 2. base + topo arms over the labeled universe only (NOT the full samples)
PR_JOBS=16 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-<sample>-ql0819 work-vtx100-base-<sample> data <evts>
PR_JOBS=16 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_TOPO_WEIGHT=3.0 \
    ./run_pr_chain_batch.sh work-<sample>-ql0819 work-vtx100-topo-<sample> data <evts>

# 3. carry every hand-scan label onto the base arm (position join, TOL=1.0)
python3 vtx_rules/carry_labels.py --write --delta-list <path> --tsv <path> --arms \
    vtxscan-harv3-nuecc48=work-vtx100-base-nuecc48:vtxscan-vtx100-nuecc48 \
    vtxscan-harv3-ncpi0=work-vtx100-base-ncpi0:vtxscan-vtx100-ncpi0 \
    vtxscan-harv3-mcp1k=work-vtx100-base-mcp1k:vtxscan-vtx100-mcp1k \
    vtxscan-harv3-delta=work-vtx100-base-{sample}:vtxscan-vtx100-delta \
    vtxscan-mcp2k=work-vtx100-base-mcp2k:vtxscan-vtx100-mcp2k \
    vtxscan-mcp2k-auto=work-vtx100-base-mcp2k:vtxscan-vtx100-mcp2k-auto \
    vtxscan-mcp2k-ragree=work-vtx100-base-mcp2k:vtxscan-vtx100-mcp2k-ragree

# 4. IPW weights + the report card
python3 dl_vtx_training/ipw_weights.py --runs vtx_rules/runs/mcp2k-20260816 \
    --tags vtxscan-vtx100-mcp2k vtxscan-vtx100-mcp2k-auto vtxscan-vtx100-mcp2k-ragree \
    --tsv dl_vtx_training/runs/ipw-vtx100-20260820.tsv
python3 dl_vtx_training/vtx_report.py \
    --tags vtxscan-vtx100-nuecc48 vtxscan-vtx100-ncpi0 vtxscan-vtx100-mcp1k \
           vtxscan-vtx100-delta vtxscan-vtx100-mcp2k vtxscan-vtx100-mcp2k-auto \
           vtxscan-vtx100-mcp2k-ragree \
    --arm-roots nuecc=work-vtx100-base-nuecc48 ncpi0=work-vtx100-base-ncpi0 \
                mcp1k=work-vtx100-base-mcp1k mcp2k=work-vtx100-base-mcp2k \
    --ipw-tsv dl_vtx_training/runs/ipw-vtx100-20260820.tsv \
    --tsv dl_vtx_training/runs/vtx-report-vtx100-20260820.tsv

# 5. rerank closure at the new epoch (search deliberately NOT run -- see sec 4)
python3 dl_vtx_training/rerank_tune.py \
    --arm-template ".../work-vtx100-{arm}-{sample}/pr_evt{evt}/calib-pr-evt{evt}.json" \
    --ipw-tsv dl_vtx_training/runs/ipw-vtx100-20260820.tsv \
    --tags vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree \
           vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta
```

**Epoch stamp.** Toolkit `8573877f` (doc pr/99 round 3: `kine_charge_dedup`,
`kine_charge_rebuild`, `shower_hadronic_tag` + friends). Verified none of that
diff touches `NeutrinoVertexFinder.cxx` or any `dl_vtx_*` knob — it is
energy-reconstruction and shower-tagging code, orthogonal to everything in
this doc. All eight arms below (§3) ran **before** the owner flipped the
round-3 knobs ON in `wct-pr-perevt.jsonnet` — i.e. at the round-2 operating
point, one commit earlier than the SBND production cfg that exists as this
doc is written. `fit_exclusion=true` (doc pr/98) is confirmed live via the
`PR30AUDIT ... knobs[...]` log line in every arm.

**A second Claude session shared this working tree today** (`doc-pr99-round3-electron-fix`,
running the pr/99 round-3 campaign concurrently). Two coordination facts worth
recording because they generalize: (1) wire-cell `dlopen`s `build/<pkg>/`, not
`local/lib` — a concurrent `./wcb build` mid-batch relinks the `.so` a running
batch has already `dlopen`ed, and the failure mode is a clean, attributable
`invalid ELF header` / `failed to load plugin` rather than a silent wrong
answer; four events hit this and re-ran clean once the relink settled. (2)
**Both sessions independently forgot `PR_EXTRA_STAGES=pr_display` on their
first pass** — the calib-pr-evt\*.json / vertex_scoreboard dumps this whole
doc depends on. If your campaign reads `calib-pr-evt<ID>.json` at all, set it
from the very first arm; a `pr_evt<ID>/` directory with `rc=0` but no
`calib-pr-evt<ID>.json` is *this*, not a refused event, and is easy to
mistake for one until you check for the flag.

---

## 1. The vertex chain, in one paragraph

Per-cluster `determine_main_vertex` (`NeutrinoVertexFinder.cxx:3419`) always
runs. The *global* winner comes from `determine_overall_main_vertex_DL`
(`:4395`) when `dl_weights` is set (SBND: always) — DL never creates a
vertex; its top-K voxels snap to existing PR-graph vertices, and a 7-term
composite (`s_dl + s_snap + s_fwd_z + s_clen + s_isol + s_main + s_fv`)
accepts the highest scorer iff it clears `dl_vtx_min_accept_score`. Only on
failure does the traditional `determine_overall_main_vertex` (`:5050`) run.
Either winner then passes through `snap_main_vertex_to_kink` →
`improve_vertex` → `main_vertex_graph_audit` (`TaggerCheckNeutrino.cxx:2101-2124`).
Full map: doc pr/52 (architecture) and pr/27 (the scorer, line-cited).

## 2. Q1 — every modification to the vertex-selection path, by outcome

**The re-rank itself received exactly one production change, ever:**

| change | commit | live result |
|---|---|---|
| `dl_vtx_min_accept_score` 4.0 → **10.0** | `9c9d0a61` (doc pr/79) | **+36/473** live (322→358) — the only shipped number |

Everything else on the DL/re-rank path is one of:

| knob | state | evidence |
|---|---|---|
| `vertex_scoreboard` | recording-only, OFF | doc pr/75 — the diagnostic dump everything below reads |
| `dl_vtx_harvest` | recording-only, OFF | doc pr/79 §10 — exact live SCN input cloud |
| `dl_vtx_top_k` 5→20 | tried, **NULL** | doc pr/78/79 §4 — 17 more near-truth candidates admitted, rerank selects **zero** of them; admission ≠ recovery |
| `dl_vtx_topo_weight`/`_center` | merged, **OFF** | doc pr/89 — offline +12/924 did not transfer; **live A/B −8/1014** |
| `dl_vtx_swap_guard` | tried, **OFF**, closed | doc pr/89 round 5 — live A/B **−36/1014** (6 fixed / 42 regressed) |
| `dl_vtx_swap_min_len`/`_min_frac` | tried, **removed from the tree** | doc pr/24 — "also does not work well; remove" |
| uBooNE-weight fine-tune (ft2u, hft1, hr1–hr4, cand-head, router, MLP) | every variant negative, **not deployed** | docs pr/77–81, pr/89 §11 — "gradients don't pay at O(10³) labels", proven twice (rebuilt-cloud artifact once, live-distribution retrain once) |
| joint composite retune (6 weights + w_topo + center + min_accept + scale) | **closed: production is a verified local optimum** | doc pr/89 §13.4 — coordinate ascent terminates at production; 0/822 pairwise configs beat it in the fully-covered region |

**The vertex gains that did land all came from the candidate graph and the
post-selection refinement, never the re-rank scorer:** `mvfit_robust`,
`vertex_kink_snap`, `two_end_break`, `steiner_gap_penalty`, `es3_stub_guard`,
the pr/85–86–99 `mvga_*` family, `fit_exclusion` — roughly thirty default-OFF
knobs across docs pr/15, 47, 48, 50, 51, 72, 84, 85, 86, 90–99, all flipped ON
by the owner after a live gate. `main_vertex_swap_apply` stays OFF (the
traditional path's cluster-swap decision is computed and then discarded).

**Why this matters for §4 below**: the composite's constexpr weights
(`W_MAIN=2.0`, `W_CLEN=2.0/60cm`, `W_ISOL=2.0/6cm`, `W_FV=0.5`,
`W_SNAP=5cm/2.0`, `W_FWD_Z=0.25/400cm`, `NeutrinoVertexFinder.cxx:4743-4752`)
are hard-coded, not configurable. The one shipped change is arithmetic:
`s_dl = dl_score × 1000` while the geometric terms span ≈ −4.25…+4.5, so at
threshold 4.0 a near-zero-`dl_score` voxel on a long main cluster cleared it
on geometry alone (`s_main+s_clen+s_fv=4.5`); at 10.0 it needs
`dl_score ≳ 0.006`. That is the entire mechanical content of doc pr/79's
flip — and the entire deployable surface of the re-rank today is
`dl_vtx_min_accept_score`, `dl_vtx_score_scale`, `dl_vtx_top_k`,
`dl_vtx_topo_weight`/`_center`, `dl_vtx_swap_guard`, and (on the traditional
path) `vertex_z_prior_scale` (100 in SBND vs 200 C++ default — the only live
non-boolean override there, applied at double the uBooNE weight).

## 3. Q2 — the label asset, measured this round

**Durability fixed first.** `vertex_labels/` was **git-ignored since doc
pr/75** — 1537 label files, 6.2 MB, zero git history, every `source` path
already pointing into an archived arm. Committed this round
(wcp `0b8c95c`), unconditionally, before anything else in this doc.

**Current-epoch pool, via `vtx_io.load_labels()` — never by globbing** (a
glob sees 1537 files across 13 tags with the same event under up to 4 tags,
and no error):

| set | tags | count |
|---|---|---|
| current-epoch pool | `TAGS_HARV3`(4) + `TAGS_MCP2K` + `TAGS_MCP2K_AUTO` | 1014 (715 human / 299 ai-scanner) |
| + IPW filler | `vtxscan-mcp2k-ragree` | +40 (human) — disjoint, fills the one stratum that made pr/88's IPW a bound |
| **total universe** | | **1054** (755 human / 299 ai-scanner) |

`taxonomy.py`'s `ALL_TAGS` still defaults to the 3 pr/78-era tags and
`load_labels()` with no argument returns 481 (the historical `prod0813`
epoch, same events as `TAGS_HARV3`, never pool the two) — every consumer
must pass `--tags` explicitly; this is a live foot-gun, not fixed here
(fixing it would change `taxonomy.py`'s and `ab_vertex_compare.py`'s
published-numbers-reproducing default).

### 3.1 Fresh arms, fresh carry (this round's numbers)

Eight arms over the 1054-event labeled universe only (not the full 3067-event
samples): `work-vtx100-{base,topo}-{nuecc48,ncpi0,mcp1k,mcp2k}`. 2108
event-runs, **0 failures** both passes — the first pass omitted
`PR_EXTRA_STAGES=pr_display` (see the coordination note above) and was
completely redone.

`carry_labels.py --arms` (new, additive flag — defaulting to the pr/82 dict
byte-for-byte when omitted, verified) carried the whole 1054-event universe
onto the new arms, position join, TOL=1.0:

| result | n | note |
|---|---|---|
| **carried** (≤1.0 cm) | **878 / 1054 (83.3%)** | 251 bit-identical, 137 changed `vertex_id` |
| delta, held back | 172 | 132 "loose" (1–3 cm), 40 "broken" (>3 cm) — **not rescanned**, per owner decision; worst-first list committed at `dl_vtx_training/runs/vtx100-20260820/delta.txt` |
| refused (rc=0, no candidate) | 4 | `169758`, `395060` (mcp1k), `317427`, `398115` (mcp2k) |

By sample: nuecc 37/42 (88.1%), ncpi0 16/19 (84.2%), mcp1k 342/386 (88.6%),
mcp2k 467/579 (80.7%), the mixed-arm `harv3-delta` tag 16/24 (66.7%, the
worst — expected, since it is already the set of once-recovered anomalies).

**This carry rate (83.3%) is meaningfully lower than pr/82's (449/473 =
94.9%).** Between pr/89 (the labels' prior carry epoch) and today, roughly
thirty more production knobs flipped ON (§2) — the candidate graph moved
enough that 1 in 6 labels no longer sits within 1 cm of any candidate. Held
back, not discarded: 4 of pr/82's 6 "materially different" delta events
re-anchored inside 3 cm anyway, and two of those landed ~300 cm from the old
point — a "carry anything under 3 cm" shortcut would write wrong labels
silently, which is why the loose bucket stays in the delta list for a human,
not an automatic re-anchor.

### 3.2 The report card — human vs AI, raw vs IPW, both tolerances

New driver `dl_vtx_training/vtx_report.py` (thin, reuses
`taxonomy.classify()` and `ab_vertex_compare.py`'s conventions — not a new
scorer). On the 878 carried labels, current epoch:

| group | n | 1.0 cm | 1.5 cm |
|---|---|---|---|
| **ALL** | 878 | **77.3%** | 78.0% |
| nuecc | 39 | 71.8% | 74.4% |
| ncpi0 | 16 | 68.8% | 68.8% |
| mcp1k | 356 | 77.8% | 78.1% |
| mcp2k | 467 | 77.7% | 78.6% |
| human | 631 | 71.2% | 71.9% |
| ai-scanner | 247 | 93.1% | 93.5% |

IPW mcp2k-arm (doc pr/89 sec 1.2 estimand — **not** to be averaged with the
row above): **78.9% / 81.1%**, all five strata measured (0 UNDEF, because the
ragree tag fills the one that used to be a bound).

Four-way decomposition at 1.0 cm: correct 679, candidate-missing 62,
net-wrong 99, selection-wrong 38 (of 878). Per doc pr/82: only
candidate-missing is a true admission gap; the other 137 are scored-and-
ranked-below — the tuning target, not the graph.

The human/ai-scanner gap above is exactly the sampling-limit caveat from doc
pr/88, not evidence AI labels are "more correct": the auto-accept tier was
gated at 39/40 = 97.5% precision on a stratum partly selected by *agreement
with the reconstruction in the first place*.

**Fixed two pre-existing gaps while building this**, both purely additive
(no prior caller ever exercised the broken path): `carry_labels.py` silently
dropped `label_source` when writing a carried label (every carried AI-scanner
pick was reading back as `human` — the human/AI split above was wrong until
this was fixed); and `sample_of_label()`/`rerank_tune.sample_of()` both
predate the `mcp2k` tag family and misrouted it to a wrong sample bucket.

### 3.3 The load-bearing finding: carrying is a filter, not a coordinate fix — and that is enough

First cut, same denominators as pr/89's own headline (all 1051 scored events,
frozen truth vs the same events re-scored on carried truth — different
event *sets*, since "carried" drops the 172 delta + 4 refused):

| | n | correct @1.0cm | candidate-missing |
|---|---|---|---|
| frozen-coordinate, full universe | 1051 | 64.7% | 182 |
| carried subset only (§3.2) | 878 | 77.3% | 62 |

That 12.6-point jump is **not** evidence that stale coordinates were
depressing the accuracy number. Restricting *both* measurements to the exact
same 878 events (join on event id, compare `cls1.0` under frozen truth vs
carried truth) isolates the coordinate effect from set membership:

| same 878 events | correct @1.0cm | correct @1.5cm |
|---|---|---|
| frozen truth | 680 (77.4%) | 687 (78.2%) |
| carried truth | 679 (77.3%) | 685 (78.0%) |

**Statistically indistinguishable — the re-snap the carry performs (≤1 cm by
construction) essentially never flips a verdict.** The 176 non-carried events
score 0/176 under either truth definition: not carried and not
correctly-classified are close to the same fact, not two independent ones —
120 of the 176 are `candidate-missing` outright, and the rest fail because
the geometric vertex that *would* carry them is one the reranker's
scoreboard never scored (the `no-row` join case `carry_labels.py` already
names).

**So the real finding is smaller and more precise than "carrying recovers
accuracy hidden by stale coordinates": carrying is what tells you which 878
of the 1054 labels the current graph still has an answer for at all.** The
172 delta + 4 refused are not a measurement artifact to correct for — they
are the actual, quantified list of labels the current PR chain has moved
away from, and the delta list (§3.2) exists precisely so a human can decide
whether to rescan them, rather than either silently dropping them or
silently scoring them against a vertex they no longer describe.

**On comparing the two epochs' *headline* numbers (77.3%/878 vs pr/89's
79.0%/1014): don't** — different denominator, different label epoch,
different candidate graph. But see §3.4 for a same-event-set comparison that
*is* valid, and it does support a claim.

**This is still the answer to "can we still use the old hand scans": yes —
carry them by position every time pattern recognition moves (`carry_labels.py`,
TOL=1.0), read the delta list as a rescan candidate list, and never score a
frozen coordinate against a graph it was not taken on.**

### 3.4 Same-event-set check against the pr/89 epoch: worse, not flat

§3.3 ruled out comparing the two epochs' headline percentages because the
denominators differ. That does not rule out a comparison *restricted to the
identical event set* — and one is available for free: the pr/89 §11.13 live
A/B tsvs (`ab-pr89topo-<sample>-20260817.tsv`, toolkit `a681b3e1`) already
carry each event's `cls_base` — the base-arm classification against the
*same* final vertex position used everywhere else in this doc (not a
scoreboard row, so the row-coordinate bias from §11.13's own offline-replay
warning does not apply here). Joining those files' `evt` column against this
round's 878 carried events finds 831 in common (47 of the 878 are outside
the pr/89 tsvs' six-tag pool — mostly `mcp2k-ragree`, added after that
round). Scoring both epochs on exactly those 831 events, same carried truth,
tol = 1.0 cm:

| sample | n | pr/89 (`a681b3e1`) | this round (`8573877f`) | Δ |
|---|---|---|---|---|
| nuecc | 37 | 32/37 = 86.5% [72.0, 94.1] | 27/37 = 73.0% [57.0, 84.6] | −13.5 pp |
| ncpi0 | 16 | 13/16 = 81.2% [57.0, 93.4] | 11/16 = 68.8% [44.4, 85.8] | −12.5 pp |
| mcp1k | 342 | 273/342 = 79.8% [75.3, 83.7] | 269/342 = 78.7% [74.0, 82.7] | −1.2 pp |
| mcp2k | 436 | 351/436 = 80.5% [76.5, 84.0] | 335/436 = 76.8% [72.7, 80.6] | −3.7 pp |
| **ALL** | **831** | **669/831 = 80.5% [77.7, 83.1]** | **642/831 = 77.3% [74.3, 80.0]** | **−3.2 pp** |

(brackets are Wilson 95% CIs). Repro: join `dl_vtx_training/runs/vtx-report-vtx100-20260820.tsv`
(`evt`,`sample`,`cls1.0`) against `dl_vtx_training/runs/ab-pr89topo-<sample>-20260817.tsv`
(`evt`,`cls_base`) on `evt` per sample; no new script was written for this,
it is a five-line join over two files already in this doc's asset list.

**Reading it straight: worse, not flat, on the one apples-to-apples check
available.** The aggregate CIs don't overlap (77.7–83.1 vs 74.3–80.0), so the
−3.2 pp move on 831 events is not just noise. The per-sample CIs on nuecc
(n=37) and ncpi0 (n=16) are wide enough that either sample alone is
consistent with no change — don't hang the finding on those two rows — but
all four samples move the same direction, which a single noisy sample
wouldn't do.

**What this is not**: it is not an indictment of any one pr/90–99 knob.
Roughly 30 knobs shipped between these two epochs (§2), each validated at
the time against its own targeted symptom events with a live Bee A/B — not
against this full labeled corpus. This is the first time the whole pool has
been re-checked since pr/89, and the honest statement is exactly that: a
real, non-artifactual net regression on the full labeled pool, cause not
yet decomposed. Decomposing it (bisecting pr/90→99 knob-by-knob against this
same 831-event set) is follow-up work, not done here — flagged in §5.

## 4. Q3 — the plan for a final re-rank tuning round

**This is a plan, not an executed tuning round.** The PR chain is not yet
final — the owner's round-3 knob flip (`kine_charge_dedup`/`shower_hadronic_tag`)
landed literally while this doc's arms were running, in a session sharing
this tree. Running a tuning search against a chain still in motion would
immediately need re-doing.

**What is ready now, verified this round:**
- `rerank_tune.py`'s outcome-calibrated ruler and closure check both work at
  the current epoch: **0 mismatches, 0 uncovered** on 1050/1051 events with
  snapped candidates, against fresh `work-vtx100-{base,topo}-*` arms (new
  `--arm-template`/`--ipw-tsv`/`--tags` flags, all additive, all default to
  the exact pr/89 invocation when omitted).
- A fresh lockbox is sealed and committed: **219/878 events (25%), seed
  20260820 (today's date — sealed interactively before `vtx_prep.sh`'s
  tag-derived seeding existed; see the note below), at
  `dl_vtx_training/runs/vtx100-20260820/lockbox.txt` — NOT read.** The pr/89
  lockbox is spent (its rounds 4–5 selected on the full 1014); this is the
  one a future round may spend, once. `vtx_prep.sh` itself now seeds from
  `crc32(tag)` instead (so a fresh tag always draws a fresh lockbox
  automatically) — a deliberate improvement made *after* this seal, not
  what sealed it.

**The plan, once the chain is declared final:**

1. Re-run `vtx_prep.sh <tag>` with a **fresh tag** (new runbook script, this
   doc's steps 1–6 as one command; seeds its lockbox from the tag, so a new
   tag draws a new lockbox automatically). **Not yet run end-to-end as one
   script** — every step was executed and verified individually this round
   (§3.1–3.2, this section), and the script was written to match, including
   one bug fix (a ragree-tag collision with `sample_of_label`, the same class
   of bug §3.2 fixes elsewhere) and one gap fix (a missing `ipw_weights.py`
   call) found by inspection after assembling it. Run it once on a throwaway
   tag before trusting it unattended on a real round.
2. Re-run the closure check first. **"Production is still optimal → stop" is
   an explicit, legitimate, pre-registered outcome** — doc pr/89 §13.4
   already found 0/822 pairwise configs beating production at its own
   epoch, and §3.3 above shows the graph has moved materially since. The
   honest premise for a new search is that pr/89's optimum was verified
   against a candidate graph that no longer exists, not that it was wrong.
3. If closure holds but coordinate ascent finds a mover: gate identically to
   every prior round in this space — offline positive is not sufficient
   (doc pr/89 §11.13: two candidates that looked positive offline came back
   −8 and −36 live) — **any candidate needs a live A/B before it ships**,
   scored against the fresh lockbox exactly once.
4. Register before that read, not after: tolerance **1.0 cm primary, 1.5 cm
   always reported beside it** (a candidate can flip sign between them —
   doc pr/89 r5's selection-only s_topo was −2 at 1.0 and +2 at 1.5); human
   labels primary, ai-scanner secondary; raw and IPW-weighted both printed,
   never mixed.
5. The seven composite weights are `constexpr`, not configurable
   (§2). The search may explore them offline and flag it if one pays, per
   the owner's explicit instruction this round — but shipping a weight move
   needs new default-OFF C++ knobs, a separate decision this doc does not
   make.

**Out of scope, flagged not folded in.** Doc pr/89 §13.5 names two remaining
levers: post-selection position refinement (three genuine topology rescues
land at 1.0–1.5 cm, downstream of the re-rank in `snap_main_vertex_to_kink`/
`improve_vertex`, not in its weights) and label volume. The owner ruled out
DL retraining for this round; both levers are named here for a future
decision, not planned.

## 5. Known limits

- §3.4's −3.2 pp same-event regression (pr/89 epoch vs this round, 831
  events) is not decomposed by knob. Bisecting the ~30 knobs shipped between
  `a681b3e1` and `8573877f` (§2) against this same 831-event set — one knob
  flipped OFF at a time, or a handful of intermediate-commit arms — would
  attribute it; not done here, flagged for a follow-up round.
- The 172 held-back delta events are a real, quantified rescan candidate —
  worst-first at `dl_vtx_training/runs/vtx100-20260820/delta.txt` — not
  scheduled here.
- `rerank_tune.py`'s hardcoded `TAGS` still excludes `vtxscan-mcp2k-ragree`
  by default; the new `--tags` flag works around it per-invocation (used
  throughout this doc) rather than changing the default, since the default
  is what reproduces doc pr/89's exact numbers.
- No determinism floor was measured this round (none requested); every arm
  ran under the runner's own `setarch x86_64 -R` pinning.
- This doc's arms predate the owner's round-3 cfg flip by one operating
  point (§epoch stamp) — re-verify the `PR30AUDIT knobs[...]` line at the
  start of any future round rather than trusting this doc's stamp to still
  hold.

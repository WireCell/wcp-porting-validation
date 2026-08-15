# doc pr/80 — a neutrino-vertex hand-scan procedure for an AI scanner

**Status: qualified result, pre-declared bar MISSED.** The rules alone do *not*
beat the reconstruction (64.1% vs 76.2% on the locked test half). Used as a
**cross-check** against the reconstructed vertex they endorse **61.7% of events,
on which the answer is right 83.3% of the time** against a 76.2% baseline on all
scored events — but see §4.1: that 83.3% is the *reconstruction's* accuracy on
the endorsed subset, not the rules' own. The 38% they decline is enriched
**×1.48** in reconstruction errors, and that enrichment is the actual product.
The pre-declared primary target — ≥90% precision at ≥40% answer-rate — was not
met, and nothing was retuned after the test half was opened.

Written in response to the owner's request to turn their hand-scan practice into
an instruction a future AI session can follow, validated against the labels they
have already produced, with "I am not sure" recorded explicitly.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the frozen split (refuses to overwrite; --check proves it reproduces)
python3 vtx_rules/make_split.py --check

# baselines, including the doc pr/79 reproduction gate
python3 vtx_rules/baselines.py

# the polarity control -- run before any rule was fitted
python3 vtx_rules/polarity_control.py

# staged dev, then the locked test half
python3 vtx_rules/eval_rules.py --half dev --limit 20
python3 vtx_rules/eval_rules.py --half dev --limit 60
python3 vtx_rules/eval_rules.py --half dev  --out vtx_rules/runs/dev-stage3
python3 vtx_rules/eval_rules.py --half test --out vtx_rules/runs/test-final
python3 vtx_rules/eval_rules.py --half test --independent --out vtx_rules/runs/test-final-independent

# look at one event (writes a PNG an agent can open with the Read tool)
python3 vtx_rules/render_event.py \
    work-mcp1k-ma10/pr_evt287654/calib-pr-evt287654.json \
    --out /home/xqian/tmp/evt287654.png
```

Everything here is read-only with respect to `vertex_labels/`. No C++, no
jsonnet, no knob, no A/B gate: this round changes nothing in the reconstruction.

---

## 1. Data, and the definition of "correct"

**481 hand-scan labels** across four production tags — `vtxscan-prod0813` (47),
`-ncpi0` (19), `-mcp1k` (407), `-mc` (8). `uitest75` and `vtxscan1` are
single-event UI tests of an event `vtxscan-prod0813` already holds and are
excluded rather than deduplicated.

**correct = Euclidean distance ≤ 1.0 cm to the label's rank-1 pick.** This is
not a new choice: it is exactly what `dl_vtx_training/taxonomy.py` and every
pr/77–79 number uses (`--tol 1.0`; truth = `sorted(picks, key=rank)[0]`). A 3 cm
column is reported beside it. Any other definition would make these numbers
incomparable to the earlier campaigns.

Two dataset facts, stated rather than hidden:

- `labels-evt287431.json` has `main_vertex: null`, so baseline denominators are
  480 rather than 481.
- The owner's "additional 9 MC events" are the port-5017 set. **8 of the 9
  already carry labels** (`vtxscan-prod0813-mc`) and are inside the 481; only
  `evt3` (r2mc) is unlabeled. Those 8 have no `-ma10` arm, so they are evaluated
  on the dump the scan was taken on and marked `arm=scanned` in the TSVs.

### The split, frozen before any rule existed

`vtx_rules/runs/split-20260815.tsv` — **150 dev / 331 test**, stratified by
(scan tag, is-the-scanned-arm-already-correct), seed 20260815, committed in
`c3a97ca` *before* the first rule was written. `make_split.py` refuses to
overwrite it; `--check` reproduces it byte-identically.

The owner asked for staged tuning that eventually consumes every sample. That
would make the final number a training number, so the staging happens **inside
the dev half only** (20 → 60 → 150) and the test half was read once, at the end,
with the configuration already frozen.

> **`--limit N` takes the first N in (tag, run, subrun, event) order, not a
> random N.** The stage-1 twenty are therefore all `vtxscan-prod0813` and the
> stage-2 sixty spill into `-mc` / `-mcp1k`. The stages are useful for working
> on a small set while iterating, but they are tag-ordered prefixes and so
> **cannot** be read as drift detection across nested random samples; the
> stage-1 84.2% vs stage-3 89.6% is mostly a change of sample, not of rules.
> Only the dev-150 and test-331 numbers are quoted as results.

### Baseline reproduction gate — passes exactly

The deployed operating point is identified from each dump's own
`vertex_scoreboard.dl_min_accept_score`, never from a directory name: five
`work-*-ma10*` variants exist and only the field distinguishes them. The
deployed arm is `min_accept = 10.0, top_k = 5`.

**358/473 at 1 cm — doc pr/79's number on the nose.** The 473 subset is not a
choice: it is exactly the labels that have an `-ma10` arm.

| baseline | answered | correct @1 cm | @3 cm |
|---|---|---|---|
| **B0** deployed `main_vertex` | 472 | **358 (75.8%)** | 377 (79.9%) |
| B1 the scanned `min_accept=4.0` arm | 480 | 325 (67.7%) | 340 (70.8%) |
| B2 lowest-z vertex of the main cluster | 472 | 276 (58.5%) | 319 (67.6%) |
| B3 non-Bragg end of the longest main track | 314 | 187 (59.6%) | 197 (62.7%) |
| B3b lower-z end of that same track | 470 | 293 (62.3%) | 306 (65.1%) |

B3b exists because B3 embeds the Bragg polarity; if that polarity were wrong, B3
would inherit the bug and stop being an independent control.

---

## 2. The polarity control — run before any rule was fitted

The single assumption that could invert the whole track branch is "the vertex is
at the end **opposite** the dQ/dx rise". `rr` is accumulated per `dirsign`
(`PrDisplayDump.cxx:447-454`) and the dump carries no explicit direction vector,
so a Bragg test written in `rr`-space could be backwards with nothing to catch
it. `vtx_geom` therefore reads **neither `rr` nor `dirsign`**: it measures arc
length from each physical endpoint of `points[]`, which has no convention to get
wrong.

Measured on the **112 dev events where production already agrees with the owner**,
so the truth vertex is not in doubt:

- `points[0] ↔ start_vertex_id`, `points[-1] ↔ end_vertex_id`: **2572 / 2572**.
- "vertex at the end opposite the Bragg rise": **86 of 98 decisive segments
  (87.8%)**. The rule is not inverted.
- `dirsign` independently agrees with the polarity-free measurement on the same
  87.8% — a cross-check, never an input.
- **`dir_weak` is useless as a quality gate**: it is set on 1335/1377 = 96.9% of
  segments. Gating R1/R3/R5 on `not dir_weak` discards the rule rather than
  sharpening it.

---

## 3. The scan procedure

This is the instruction. It is implemented in `vtx_rules/vtx_rules.py`; the
step numbers are the `rule` strings in the output TSVs, so any verdict can be
traced back to the step that produced it.

**Step 0 — the cluster.** Work in the main cluster (`main_vertex.cluster_id`,
equivalently `segments[].is_main_cluster`). On dev the owner's pick was in the
main cluster on **56 of 58** events, so no cluster-override rule is used. The
34/155 wrong-cluster cases are real but choosing when to override needs a
precision no available signal supports — pr/79's whole finding.

**Step 1 — is there anything to read?** If the main cluster's total fitted
length is under 8 cm, or no single segment exceeds 3 cm, this is the owner's
case 8 ("only dots"): **abstain**. Fired on 8 dev / 18 test events.

**Step 2 — branch.** A track of ≥10 cm puts you in the *track* branch; otherwise
the *shower* branch.

### Track branch

**Step 3 (R1) — eliminate stopping ends.** For each candidate vertex, look at
every attached track ≥5 cm and ask which end of it carries the dQ/dx rise (mean
dQ/dx over the last 5 cm, one end vs the other, ratio ≥ 1.3 to count as
decisive). A vertex with a *stopping* end on it is not the neutrino vertex —
"everything comes out of the vertex". Drop it. A vertex whose attached tracks
give no decisive answer **survives**: unmeasured is not evidence against.

This is the strongest single discriminator measured. On dev, of the vertices
where the test is decisive, **86.5% of truth vertices have every attached track
pointing away**, against **31.9% of the other main-cluster vertices**; and
**63.8% of non-truth vertices are a stopping end**, against 8.1% of truth ones.

**Step 4 (R3) — drop Michel ends.** A vertex carrying both the Bragg end of a
≥10 cm track and a <8 cm stub is a stopped muon wearing its Michel electron.
The neutrino vertex is at the far end of that muon. Applied only when it leaves
at least one survivor.

**Step 5 (R1 again, as a preference) — prefer proved-outgoing.** Among the
survivors, prefer those where the test was *decisive and clean* over those that
merely had no opinion.

**Step 6 (R2) — upstream tie-break.** Among what is left, take the lowest z.
This is a **tie-break, not a selector**: on production's misses the owner's pick
is at lower z 72% of the time but only by a median of 5.1 cm. If the two best
candidates are within 5 cm in z, the answer is genuinely ambiguous: **abstain**.

**R5 ("a long muon connects to the neutrino vertex") is deliberately NOT a
filter.** This is the one place the data contradicted the heuristic as literally
stated: on the dev misses the truth vertex sat off *any* long track in **10 of
15** cases, so filtering on it threw the answer away. Five orderings built from
the owner's rules were compared on dev — (outgoing, low z) 40/56, (on-long,
outgoing, low z) 39/56, (outgoing, on-long, low z) 37/56, lowest-z alone 36/56,
(outgoing, most attached length, low z) 32/56 — and R5 survives as evidence
printed in the notes, not as a cut.

### Shower branch

**Step 7 (R6) — converging showers.** If two or more showers start at the same
vertex, that is the owner's NC stub topology: take it. **This step performed
badly and should be distrusted** — 1/6 correct on the test half.

**Step 8 (R7) — the biggest shower's start.** Otherwise take
`showers[]` argmax `kine_best` and its `start_vertex_id`. Fired only twice
across both halves; not meaningfully validated.

### Step 9 (R9) — cross-check against the reconstruction

The display draws the reconstructed vertex as a star, so a human scanner is
never choosing from a blank picture — they are deciding whether to accept the
star or move it. Reproduce that: compare your answer with `main_vertex`.

- **They agree** → this is the confident answer.
- **They disagree** → do **not** prefer either. Record **unclear** and put the
  event in front of a human.

This is the only confidence signal in the procedure with ground truth on both
sides, and it is what carries the result (§4).

### Not implemented

**R4, hadronic showers.** The dump carries no hadronic-shower tag, and
`segments[].particle_id` on shower segments is e/γ-oriented. Declared out of
scope rather than faked.

---

## 4. Results

Scored events exclude the 3 `not_a_candidate` Tier-D labels (doc pr/52: the
owner placed the vertex by hand because no PR-graph vertex was near it, so no
selection rule can reach them) and the Step-1 "just dots" events (the owner
states the answer does not matter there, yet they still had to click something,
so scoring them charges the engine a miss for abstaining *and* a miss for
answering differently). Both exclusions were fixed **before** the first
measurement and both counts are printed; a `FLOOR` line with nothing excluded is
printed beside every headline.

| | dev (150 labels) | **test (331 labels, locked)** |
|---|---|---|
| scored | 141 | 311 |
| production baseline on the same events | 106 (75.2%) | 237 (**76.2%**) |
| **rules alone**, precision @ answer-rate | 64.7% @ 84.4% | **64.1% @ 91.3%** |
| **rules + R9 cross-check** — reco accuracy on the endorsed subset (§4.1) | **89.6%** | **83.3%** |
| ...at answer-rate | 54.6% | 61.7% |
| ...precision @3 cm | 90.9% | 85.4% |
| reco wrong: all scored / endorsed / declined | 24.8% / 10.4% / 42.2% | 23.8% / 16.7% / **35.3%** |
| enrichment on the declined subset | ×1.70 | **×1.48** |

Per step, on the test half:

| step | n | precision |
|---|---|---|
| R1-outgoing | 141 | 83.7% |
| R1-outgoing + R2-z | 26 | 92.3% |
| R2-upstream + R2-z | 18 | 88.9% |
| R6-shower-convergence | 6 | 16.7% |
| R7-biggest-shower-start | 1 | 100% |

Why the engine declined, on the test half: R9 disagreement 92, z-ambiguity 23,
no shower start 3, no main cluster 1.

### 4.1 What the 83.3% is, and what it is not

With the R9 cross-check on, the engine answers **only** where its independent
pick coincides with `main_vertex`. The answered set is therefore *by
construction* the reco-agreement subset, and every answered row has
`dist_cm == prod_dist_cm`. So **83.3% is the reconstruction's accuracy on the
subset the rules endorse — it is not the rules' own accuracy**, and the
"+7.1 points over 76.2%" is a selection effect, not an improvement the rules
produce. The rules contribute exactly one bit per event: endorse or decline.

That bit is the deliverable, and the number that measures it is the enrichment,
not the precision. The rules' own accuracy is the `--independent` row: 64.1%.

### Against the pre-declared bar

- **Primary — ≥90% precision at ≥40% answer-rate: MISSED.** 83.3% at 61.7%.
  Dev reached 89.6%; the 6.3-point dev→test drop is the finding, not a rounding
  error, and it is what a holdout is for.
- **Secondary — overall accuracy ≥ B0: MISSED.** 51.4% vs 76.2%, because an
  abstention counts as wrong in that metric and the engine abstains on 38%.
- As pre-declared, this is written up as it came out. Nothing was retuned after
  the test half was opened; the `--independent` numbers come from the same
  single opening.

### What is nonetheless useful

1. **The endorse/decline bit is a usable triage.** Endorsed events carry a
   16.7% reconstruction-error rate against 23.8% overall; declined events carry
   35.3%. Read §4.1 first: the endorsed subset is *selected on* agreement, so
   this is triage, not correction. The rules inherit reconstruction failures;
   they do not manufacture new ones.
2. **Disagreement is a scan-prioritisation signal.** On the test half the
   reconstruction is wrong on 23.8% of scored events but **35.3%** of the ones
   the rules decline — ×1.48 enrichment, in the same family as pr/78's
   `scan_ranker` (×1.91 for corrective enrichment) but derived from physics
   rather than from a fit, and therefore not tied to the 473 labels the earlier
   fits consumed. That is directly usable against pr/79 §7e prerequisite 1
   ("more labels").
3. **Owner rule 1 is quantified**: 86.5% vs 31.9% separation, the strongest
   single feature found, and one that **no pr/77–79 formulation used** — every
   one of those ranked on the 23-column scoreboard row, which contains no dQ/dx,
   no Bragg direction, no shower start and no vertex topology.

### The internal confidence tier is inverted — do not use it

`certain` scores **81.1%** and `likely` **90.9%** on the test half (86.2% vs
100% on dev). The rule-internal tier is anti-calibrated at this sample size and
is kept in the output only so the fact stays visible. **R9 agreement is the
confidence signal**; the tier is not.

---

## 5. A worked failure — evt287654

`render_event.py` on `work-mcp1k-ma10/pr_evt287654`, one of the eight dev events
where the engine answered confidently and was wrong by 52 cm:

```
cluster 12: 5 vertices, total 75.0 cm, longest segment 56.3 cm
track branch: longest track 56.3 cm
R1: dropped 1 of 5 candidates carrying a stopping end
R1: 1 candidate(s) proved fully outgoing
R5: 1 of 1 survivors sit on a long track (evidence, not a cut)
R9: agrees with the reconstructed vertex
```

A single 56 cm track runs from (z≈497, y≈97) down to (z≈465, y≈58). Vertex
12000 sits at the high-z, high-y end; vertices 12001–12004 cluster at the low-z
end with several short stubs. R1 proved 12000 outgoing — the stubs at the low-z
end put stopping ends there — so the procedure took it, and the reconstruction
agreed, so R9 raised no flag. The owner picked the low-z cluster.

The lesson is the ordering, not the threshold: **R1 outranked R2, and on this
event R2 was right.** The dQ/dx along the long track is nearly flat (40–60 k
e/cm, cyan-green on the fixed ramp), so R1's "decisive" verdict rested on the
short stubs rather than on the muon itself. A future round should consider
weighting a Bragg verdict by the length of the track that carries it — measured
on dev before being believed, since five orderings were already compared and the
simplest won.

---

## 6. Limits

- **The rules do not replace the selector, and this round does not propose
  that.** 64% independent vs 76% deployed. Anyone reading this as "the
  heuristics beat the reconstruction" has it backwards.
- **The abstain branch is unvalidated as such.** The owner set `confidence` on
  2 of 483 labels and never once used "unclear", so there is no ground truth for
  "the scanner was unsure". Precision-on-answered and answer-rate are reported
  separately and the abstain criterion itself is not claimed to be calibrated.
  Re-scanning ~30 events with the confidence buttons set would make it testable.
- **R6 and R7 rest on 7 and 2 events.** Treat them as unmeasured.
- **R4 is not implemented.**
- Everything is measured against the owner's picks, which are the operational
  truth here, not MC truth — the dumps carry none.
- Numbers are on the deployed `-ma10` arm for 473 labels and on the scanned
  arm for the 8 `-mc` labels, marked per row in the TSVs.

## 7. Files

| file | what |
|---|---|
| `vtx_rules/vtx_rules.py` | the decision procedure of §3 |
| `vtx_rules/vtx_geom.py` | dQ/dx and Bragg-end primitives; reads neither `rr` nor `dirsign` |
| `vtx_rules/vtx_io.py` | label loading, the 1 cm convention, dump helpers |
| `vtx_rules/make_split.py` | the frozen split; refuses to overwrite |
| `vtx_rules/baselines.py` | B0–B3b and the pr/79 reproduction gate |
| `vtx_rules/polarity_control.py` | §2 |
| `vtx_rules/eval_rules.py` | scoring, per-rule tables, precision-vs-coverage |
| `vtx_rules/render_event.py` | the PNG an agent opens with the Read tool |
| `vtx_rules/runs/` | the split, and the dev/test result TSVs |

`.gitignore` carries a `test*` rule, so `vtx_rules/runs/test-final*` is committed
with `git add -f`. Re-running the Repro block rewrites those files and **git will
not show them as dirty** — compare them by hand if you need to know whether the
numbers moved.

Determinism: two consecutive runs produce byte-identical TSVs, and both match
the committed `vtx_rules/runs/dev-stage3/events.tsv`.

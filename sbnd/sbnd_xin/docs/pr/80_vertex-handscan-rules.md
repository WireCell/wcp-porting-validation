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

# --- round 2 (sec 9-10): the scan kit and the 120-event re-measurement -------

# blindness + determinism of the kit, over 25 dumps
python3 vtx_rules/scankit.py selftest

# one event's full panel set, and a zoom on one of its vertices
python3 vtx_rules/scankit.py prepare \
    --dump work-mcp1k-ma10/pr_evt65289/calib-pr-evt65289.json \
    --out /home/xqian/tmp/evt65289-kit --title evt65289
python3 vtx_rules/scankit.py zoom \
    --dump work-mcp1k-ma10/pr_evt65289/calib-pr-evt65289.json \
    --vertex 19007 --half-width 8 --out /home/xqian/tmp/evt65289-z.png

# the four arms.  Each `prepare` is deterministic; the scanning between
# prepare and score is done by fresh subagents handed vtx_rules/scan_prompt.md
# verbatim, so re-running it re-scans rather than replaying.
python3 vtx_rules/selfscan.py prepare --half dev --n 60 --kit new  --workers 4 \
    --out /home/xqian/tmp/scan2/dev-new
python3 vtx_rules/selfscan.py prepare --half dev --n 60 --kit old  --workers 4 \
    --out /home/xqian/tmp/scan2/dev-old-a          # and again -> dev-old-b
python3 vtx_rules/selfscan.py prepare --half test --n 60 --kit new --workers 4 \
    --seed 20260816 \
    --exclude-manifest vtx_rules/runs/selfscan-20260815/manifest.json \
    --out /home/xqian/tmp/scan2/test-new
python3 vtx_rules/selfscan.py score   --dir /home/xqian/tmp/scan2/<arm>
python3 vtx_rules/selfscan.py compare --a <old arm> --b <new arm>
```

The committed picks under `vtx_rules/runs/scan2-20260815/` are the record of the
runs quoted in §10; `score` and `compare` replay from them exactly.

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

## 7. Can an AI session scan by eye? — a blind self-scan, measured

§4 measures the *engine*. The owner's actual question is whether an agent can
scan new events and leave the doubtful ones for a human, so the scanner in
question is an agent looking at rendered pictures — a different thing, with no
reason to score alike.

`vtx_rules/selfscan.py` measures it. `prepare` renders **blind** PNGs: no
reconstructed vertex, no engine pick, only the dQ/dx-coloured points and the
numbered candidate vertices, plus a worksheet of candidate ids, degrees and
positions. A scanner that can see the reco star anchors on it, and labels that
only confirm the reconstruction are worse than no labels. `score` refuses to run
until `picks.json` exists, so the picks are on disk before any truth is read.

**20 events drawn from the locked test half, seed 20260815, scanned blind:**

| | |
|---|---|
| answered | 16 / 20 (80%) |
| correct @1 cm | **11 (68.8% precision)** |
| correct @3 cm | 12 (75.0%) |
| **correct by vertex id** (rescored, see F2) | **12 (75.0%)** |
| reconstruction on the same 20 | **15 (75.0%)** |

**The agent scanner is not better than the reconstruction.** At n=20 the
difference is not itself significant, but "not better" is clear.

> **Rescored after F2.** evt52085 was charged as a miss at 1.29 cm, but the
> agent named vertex **6002** and the owner's recorded truth vertex **is 6002** —
> the 1.29 cm is the arm-to-arm refit shift, not an error. By vertex id the scan
> is 12/16, not 11/16. This is the only event of the 20 that changes.

### The abstention does not separate — but the evidence for that is thin

| stated confidence | answered | correct @1 cm | correct by vertex id |
|---|---|---|---|
| certain | 3 | 2 (67%) | **3 (100%)** |
| likely | 10 | 7 (70%) | 7 (70%) |
| unclear | 3 | 2 (67%) | 2 (67%) |

The original reading of this table — "flat, therefore the hand-back workflow
cannot work" — was **overstated**, and the correction is worth stating plainly:
the single event that separates the `certain` tier from the others is the same
evt52085 that F2 shows was never wrong. On the corrected column the tiers do
order correctly (100 / 70 / 67).

What survives is much weaker: **the `certain` tier has n=3.** Three events
cannot establish calibration in either direction. The honest conclusion from §7
alone is that the abstention signal is *untested*, not that it is broken — and
that is why §10 re-runs this at n=60 with enough events per tier to decide it.

The rest of this section's reasoning is unaffected: the errors that remain are
spread through the confident picks, so on this sample the human would still have
to check most events, which is the work the
exercise was meant to save.

### Where the agent and the reconstruction disagree

|  | reco correct | reco wrong |
|---|---|---|
| agent right | 9 | **2** |
| agent wrong | 4 | 1 |
| agent abstained | 2 | 2 |

Two useful facts. The agent **never confidently endorsed a wrong reconstruction
answer** — on the 5 events the reconstruction got wrong it was right twice,
abstained twice, and missed once by 1.3 cm (correct at the 3 cm tolerance). And
it **broke 4 events the reconstruction already had right**, which is where the
whole deficit lives.

### One failure mode explains all four big misses

Miss distances: **1.3, 10.6, 44.4, 127.7, 165.2 cm**. The 1.3 cm is a near miss
on an otherwise textbook vertex. The four real misses share a single error: on a
long track with a clear Bragg peak, the agent applied "the vertex is at the end
opposite the rise" and took the far, downstream end, where the owner had picked
the upstream one. That is **exactly the R1-over-R2 ordering error the engine
made on evt287654** (§5).

Both scanners fail the same way, from the same rule ordering, which makes it a
correctable defect rather than noise: the Bragg verdict on a single long track
is being trusted more than the upstream prior deserves, and a stopping track
whose far end is deep downstream is more often a cosmic-like crosser than a
neutrino daughter. Fixing the ordering is the obvious next round, and it must be
fitted on dev and re-checked on a fresh blind draw, not on these 20.

### Verdict on the owner's question — as it stood at §7, and see §10

**Not yet.** Autonomous scanning at this bar would produce a label set that is
~30% wrong with no way to tell which 30%. What the agent scan *can* be used for
today, because it is measured and safe:

- **triage** — it never confidently endorsed a wrong reconstruction, and it
  found 2 of the 5 reconstruction errors in 20 events;
- **a second opinion on flagged events**, where the human makes the call;
- **not** for generating labels that anything is then fitted to.

> **This verdict is superseded by §10 and the reasoning behind it was wrong in
> one specific way.** "No way to tell which 30%" rested on a `certain` tier of
> three events. At n=60 that tier separates sharply, and with the §9 kit it
> covers 45% of events at 92.6% (96.3% by vertex id) — so there *is* a way to
> tell, and §7 could not see it. What survives is that the overall rate was not
> good enough; §10 shows the tooling, not the procedure, was the reason.

Repro:
```bash
python3 vtx_rules/selfscan.py prepare --half test --n 20 --out /home/xqian/tmp/selfscan1
# look at the PNGs, write picks.json, then:
python3 vtx_rules/selfscan.py score --dir /home/xqian/tmp/selfscan1
```
The picks and the scored output of the run quoted here are committed under
`vtx_rules/runs/selfscan-20260815/`.

## 9. Better eyes — the tooling round

§7's verdict was "not yet", and the owner's response was a hypothesis about
*why*: the scan was done on one flat PNG, whereas the real object is 3-D, needs
the endpoints checked, needs zooming into the vertex region, and needs multiple
views. This section tests that hypothesis rather than assuming it.

### 9.1 What the five misses actually were

Re-opening all five, four are information-presentation failures whose evidence
was already in the dump, and one is not a scan error at all:

| event | miss | diagnosis |
|---|---|---|
| evt282737 | 165.2 cm | **two** Bragg ends existed — the 175 cm muon stops at 21000, the 30 cm pion at 21001 — and the vertex 21003 is the junction both point away from. One segment was read; the connectivity was never seen. |
| evt283991 | 127.7 cm | truth 15003 joins a flat 133 cm muon to a 16 cm proton that Braggs at its far end. It is also the *lower-z* end, so rule 2 agreed — the ends could not be told apart in the picture. |
| evt399856 | 44.4 cm | dQ/dx is hot at **both** free ends and cold in the middle. Separating a monotone Bragg rise from a hot end blob needs the profile **shape**, not the end value. |
| evt174637 | 10.6 cm | 23 candidates inside a 60 cm shower blob. The discriminator was a 26 cm track attached at 9006 that stops away from it — unreadable at full-event scale. |
| evt52085 | 1.29 cm | **not a miss.** The right vertex was named; it sits 1.29 cm from the click because the label was taken on a different arm. See F2. |

### 9.2 Five findings that change what success means

**F1 — the candidate list was silently narrowed, and it cost 5.7 points of
ceiling.** The frozen plan said "every PR-graph vertex"; `selfscan.prepare`
filtered on `cluster_id != main`. Over 473 labels:

```
ceiling @1cm : main-cluster pool 92.0%   all-cluster pool 97.7%
```

evt59085 is exactly this — the §7 scan recorded "the real start looks like
z=73, which is not in this cluster's candidate list" and abstained on an
unanswerable question. One `if`, 5.7 points. Fixed.

**F2 — the residual ceiling is a cross-arm refit shift, not a free-hand click.**
The first reading of this was that the owner clicks somewhere near a vertex
rather than on one. That is **wrong**, and the correct version matters more:

```
labels with a recorded truth vertex id and a deployed arm: 470
  that id sits <=1cm from the click IN THE LABEL'S OWN ARM : 470 (100.0%)
  that id sits <=1cm from the click IN THE DEPLOYED ARM    : 460 (97.9%)
  that id is ABSENT from the deployed arm entirely         :   0 (0.0%)
  arm-to-arm shift of the SAME vertex id: median 0.000 cm, p99 1.85, max 10.72
                                          >1cm: 10 labels (2.1%)
```

The click is snapped, every time. What moves is the **vertex**. The labels were
taken on `work-*-prod0813` (`min_accept = 4.0`); production and every pr/78–80
measurement score against `work-*-ma10`. Changing `min_accept` changes which
vertex is main, and by F5 `improve_vertex` refits the trajectories *after* that
choice — so the same graph vertex id lands somewhere slightly different in the
two arms. 2.1% of labels move by more than 1 cm.

Three consequences:

- The 97.7% all-cluster ceiling of F1 is now **fully accounted for**: 2.1%
  cross-arm shift plus the handful of `not_a_candidate` manual picks. There is
  no unexplained residual.
- Scoring by distance charges a scanner for a refit it can neither see nor
  control. `selfscan.py` now reports **both** the 1 cm distance number (kept as
  the headline, so pr/78–79 comparability is not quietly broken) and a
  `by vertex id` number, which is the same question without the artifact.
- **This applies retroactively to every pr/78–80 number**, including §4's. At
  2.1% it overturns no conclusion, but a procedure quoted as if it could reach
  100% against these labels never could.

**F3 — the engine's conclusion must never be shown to a scanner.** Run without
the §3 step-9 reco cross-check on the §7 events, the engine scores **8/20**
against the unaided eye's 11 and the reconstruction's 15, and is right on
exactly **one** event where the reconstruction is wrong. Its *verdict* is worse
than the eye. Its *evidence* — connectivity plus which end is hot — is precisely
what the eye missed on three events. So the kit ships the evidence and never the
verdict.

**F4 — beating the reconstruction is not this round's target.** Eye-correct and
engine-correct are largely disjoint, but their oracle union on those 20 events
is 14/20, still under the reconstruction's 15. The goal is to stop the eye
destroying events the pipeline already gets right, and to make confidence mean
something.

**F5 — contamination is deeper than §2 assumed, and it invalidates the shower
branch.** `TaggerCheckNeutrino.cxx:1237-1469` fixes the order:
`determine_main_vertex` → `determine_overall_main_vertex_DL` → `improve_vertex`
(refits trajectories, re-runs PID) → `examine_direction(main_vertex)` →
`shower_clustering_with_nv`. So `dirsign`, `dir_weak`, `points[].rr` (which
`PrDisplayDump.cxx:447-454` reverses *according to* `dirsign`), the whole of
`showers[]` including `start` and `stem_dqdx`, `is_main`, `main_candidate` and
`is_main_cluster` are all **downstream of the vertex choice**. §3's R6/R7 read
`showers[]` and are therefore contaminated — recorded here, not fixed.

§2's decision to read neither `rr` nor `dirsign`, taken for polarity safety,
turns out to be exactly the right blindness decision as well, and `vtx_geom.py`
is reused unchanged.

### 9.3 The three-tier rule, enforced in code

| tier | fields | treatment |
|---|---|---|
| 1 safe | `meta`, `dqdx_ref`, `dead`, `steiner` xyz + `flag_terminal`, `track_shower` xyz, `vertices[].degree`, cluster ids, and arc length **recomputed by us** | shown freely |
| 2 vertex-conditioned | fitted `points[]` xyz/dQ/dx, `segments[].length`, the start/end vertex connectivity, `particle_id` | shown, and flagged as conditioned. The scan is not executable without them, and it is what the owner looks at on port 5017. |
| 3 the answer | `main_vertex`, `is_main`, `main_candidate`, `is_main_cluster`, `vertex_scoreboard`, `dirsign`, `dir_weak`, `rr`, `showers[]`, `kine`, `tagger`, and every field of `decide()` | **stripped**, not merely unrendered |

`scankit.sanitize()` builds a whitelist copy of the dump and every renderer works
only off that copy, so nothing in the kit can read `main_vertex` even by mistake.
`assert_blind()` then greps the output as a check that `sanitize` is what ran.
`scankit.py selftest` runs both over 25 dumps.

### 9.4 The kit

Each panel exists because a specific miss demanded it.

| panel | what | the miss it answers |
|---|---|---|
| P1 overview | three projections **twice** — whole detector for containment, auto-framed to the event for resolution; per-segment colours; candidates from **every** cluster | the 50 cm event drawn across a 500 cm box; F1 |
| P2 rotations | four 3-D viewpoints plus the event's own principal-axis frame | evt174637; a diagonal track is foreshortened in X-Y, Z-Y **and** X-Z at once, and rotating is the only fix |
| P3 zoom (on demand) | `scankit.py zoom --vertex <id> --half-width <cm>`; attached segments in colour, everything else grey, an arrow per segment toward increasing dQ/dx | the owner's "zoom into the vertex region" |
| P4 profiles | dQ/dx vs arc length from **both** ends, with the `dqdx_ref` stopping-muon and stopping-proton templates anchored at *each* end | evt399856, blob vs Bragg |
| P5 cone | transverse RMS vs distance from each candidate apex, computed from the point cloud | rules 6–7, which can no longer read `showers[]` (F5) |
| P6 evidence sheet | per candidate, all clusters: degree, position, and per attached segment the 5 cm end means at both ends with the far vertex named | evt282737, evt283991, evt174637 — all three are immediate from this table |

P4 draws the templates at *both* ends deliberately: the panel poses the question
("if this end were the stop, a proton would look like that") instead of
answering it. P2 and P5 are skipped on events with too little structure to
support them — a cone profile on a two-segment track is noise wearing the
costume of evidence.

### 9.5 The port-5017 viewer, patched

The owner chose to make these the default view. Four additive changes;
`on_vscan_save` and the 17-key label schema are **untouched**, and the saved
document never serialises a table row wholesale, so the new column cannot leak
into a label.

1. **"Both ends (no dirsign)"** mode on the 1-D dQ/dx panel — arc length
   recomputed from the fitted points, templates anchored at each end. The
   existing End mode is oriented by `dirsign`, so when the reconstruction has
   the direction wrong the Bragg peak is drawn at the wrong end with nothing on
   screen to catch it. The caption now also prints the polarity-free two-end
   reading **in every mode**, which is the one line that can contradict
   `dirsign` on screen.
2. **A per-segment table** — the panel showed one segment at a time through a
   dropdown, so reading a 6-prong vertex meant six clicks and six remembered
   numbers. Every segment's two-end dQ/dx now fits on one screen.
3. **Direction arrows**, a toggleable layer, on by default: one arrow per
   segment at its cooler end pointing at its hotter end. A segment whose ends
   are within 1.3× gets **no** arrow — no opinion must not be drawn as one.
4. **An `out/meas` column** in the hand-scan vertex table: of the attached
   segments with both ends measured, how many get hotter going away. Rule 1 as
   a measurement.

> **Label epoch.** From this patch onward the viewer shows the rule-1 column, so
> labels taken afterwards are **no longer independent of the rules being
> validated**. The 481 existing labels are the only pre-patch reference set that
> will ever exist. Any future validation must either use them, or state plainly
> that it is validating against post-patch labels.

## 10. Does better tooling help? — 120 blind events, measured

### 10.1 Design

§7 scanned 20 events with one scanner and one kit, which cannot separate a
tooling effect from how one scanner happened to feel that day. This round runs
**three scans over the identical 60 dev events**:

| arm | kit | purpose |
|---|---|---|
| `dev-old-a` | old (one flat PNG + main-cluster candidate list) | |
| `dev-old-b` | old, **again**, independent scanners | together with A, the **scanner-noise floor** |
| `dev-new` | the §9.4 kit | the tooling delta, read against that floor |

plus **`test-new`**, 60 events from the locked test half that are disjoint from
§7's twenty and from dev, scanned once with the new kit — the held-out number.

Every arm is scanned by four fresh subagents, 15 events each, handed
`vtx_rules/scan_prompt.md` **verbatim**. The template carries no per-event text,
so whoever launches a run cannot steer it toward events whose answers they know.

**Protocol deviation, recorded rather than buried.** One `dev-old-b` worker
cropped and upscaled `event.png` itself and left a `_zoom.py` behind, i.e. it
manufactured zoom in the arm that is defined by not having it. `dev-old-a` is
clean (checked: zero stray files). So **`old-a` vs `new` is the primary
comparison** and `old-b` vs `new` is a second, weaker one whose old arm had
partial zoom on 15 of its 60 events. They are reported separately, never pooled.

### 10.2 The scanner-noise floor — and it is low

```
paired on 60 common events:  A = dev-old-a   B = dev-old-b
  --- correct @1cm ---   A 36/60, B 37/60
    both  34    neither  21
    B only  3    A only  2      discordant 5     exact binomial p = 1.0000
  --- correct by vertex id ---   A 36/60, B 38/60
    both  34    neither  20
    B only  4    A only  2      discordant 6     exact binomial p = 0.6875
```

Two independent scanners with the same kit agree on **55 of 60** events and
their disagreements split evenly. Between-scanner variance is therefore small,
and a tooling effect of more than a handful of discordant pairs in one direction
is real rather than mood.

### 10.3 The §7 calibration finding does not survive n=60

This is the correction that matters most to the owner's actual question. §7
concluded from three `certain` events that confidence was flat. At n=60, with
the **old** kit — no new tooling at all:

| arm | certain | likely | unclear |
|---|---|---|---|
| `dev-old-a` | 14 events, **92.9%** | 29 events, 58.6% | 15 answered, 40.0% |
| `dev-old-b` | 12 events, **91.7%** | 30 events, 60.0% | 15 answered, 53.3% |

**The tiers order, and they order steeply.** A `certain` tier at ~92% over ~22%
of the sample is exactly the shape the hand-back workflow needs, and it was
already there in §7's kit — §7 simply had three events in the tier and could not
see it. The pre-declared secondary criterion (certain ≥90%, and ≥20 points above
unclear) is **met on both old-kit arms**.

Accuracy alone is not the whole criterion, though, so coverage is reported beside
it from here on: `certain` is not a fixed-size stratum, and a scanner that spends
the word freely could put half the sample there at a lower rate. A tier holding
20% of events at 92% and one holding 60% at 85% are different products.

For context on the same 60 dev events: the reconstruction gets **43/60 (71.7%)**,
the old-kit scanners 36 and 37, and the §3 engine 28/43 answered when run without
its reco cross-check. The eye beats the engine; neither beats the reconstruction.

### 10.4 The new kit, on the same 60 dev events

| | `dev-old-a` | `dev-old-b` | **`dev-new`** | reconstruction |
|---|---|---|---|---|
| answered | 58/60 | 57/60 | 59/60 | — |
| correct @1 cm | 36 | 37 | **43** | **43** |
| correct by vertex id | 36 | 38 | **43** | 43 |
| answered-precision | 62.1% | 64.9% | **72.9%** | — |
| triage enrichment | ×1.76 | ×2.16 | **×2.59** | — |

The scanner goes from clearly below the reconstruction to **level with it**,
43/60 either way. Paired, against a noise floor whose discordant split was 3:2:

```
old-a (clean) vs new :  new wins 10, old wins  3   discordant 13   p = 0.092
old-b         vs new :  new wins  8, old wins  2   discordant 10   p = 0.109
noise floor a vs b   :  B wins    3, A wins    2   discordant  5   p = 1.000
```

Both comparisons agree in direction and magnitude — the new kit wins about 3–4×
as often as it loses, where two runs of the *same* kit split evenly. As
pre-registered, no pass/fail threshold is applied: at n=60 neither two-sided
exact test reaches 0.05 (95% CI on the new-kit win fraction: [0.46, 0.95] and
[0.44, 0.97]). **The two comparisons are not pooled** — they share the `dev-new`
arm, so they are not independent, and pooling them to 18:5 (p = 0.011) would be
a fabricated significance.

**Where the +7 comes from** (old-a vs new):

```
  reco RIGHT  n=43   old 33   new 37   (+4)   events the eye used to break
  reco WRONG  n=17   old  3   new  6   (+3)   genuinely new correct answers
  correct new-kit answers whose vertex is NOT in the largest cluster: 2  (F1)
```

Both halves moved. F4 predicted the round would be about not breaking what the
pipeline already gets right; that is +4 of the +7, but the other +3 **doubles**
the eye's yield on the events the reconstruction gets wrong — which is the half
that has any value for finding reconstruction errors.

**Calibration, with coverage:**

| tier | coverage | answered | correct @1 cm | by vertex id |
|---|---|---|---|---|
| certain | 27/60 (45.0%) | 27 | 25 (**92.6%**) | 26 (**96.3%**) |
| likely | 23/60 (38.3%) | 23 | 14 (60.9%) | 13 (56.5%) |
| unclear | 10/60 (16.7%) | 9 | 4 (44.4%) | 4 (44.4%) |

The new kit roughly **doubles the coverage of the `certain` tier** (45% against
old-kit 20–23%) at the same ~92% accuracy — 96.3% by vertex id. That is the
result that matters for the owner's workflow: it is not that the scanner became
more accurate overall, it is that it became accurate *and knew when*.

### 10.5 The held-out arm — 60 test events, read once

Disjoint from §7's twenty and from the dev sixty. Scanned once with the new kit,
by four fresh subagents, and reported as it came out.

| | `test-new` | reconstruction |
|---|---|---|
| answered | 59/60 | — |
| correct @1 cm | 45 (76.3% precision) | **46 (76.7%)** |
| **correct by vertex id** | **46 (78.0%)** | 46 |
| triage enrichment | ×2.41 | — |

**Parity replicates on held-out data.** 46 against 46. Dev was 43 against 43.

**Calibration — the pre-declared secondary criterion, on the locked half:**

| tier | coverage | correct @1 cm | by vertex id |
|---|---|---|---|
| certain | 22/60 (36.7%) | 21 (**95.5%**) | 21 (**95.5%**) |
| likely | 26/60 (43.3%) | 18 (69.2%) | 18 (69.2%) |
| unclear | 11 answered (20.0% coverage) | 6 (54.5%) | 7 (63.6%) |

The bar was "certain ≥90% **and** ≥20 points above unclear". Result: **95.5% and
+41 points. PASSED**, on data never used for tuning.

### 10.6 What that actually buys — and the part that does not work

Reading the tiers as a workflow on these 60 events:

- **Auto-accept the 22 `certain` picks** at 95.5%.
- **Hand back the other 38.** Those 38 contain **12 of the 14 events the
  reconstruction gets wrong** — so reviewing 63% of the sample surfaces 86% of
  the reconstruction's errors.

That is a real and useful division of labour, and it is what the owner asked
for. But two qualifications belong next to it, not underneath it:

**The `certain` tier is almost entirely the reco-agreement subset.** 21 of the
22 certain picks agree with `main_vertex`. This is §4.1's selection effect
arriving on its own rather than by construction: the scanner is confident where
the reconstruction is also right, so the 95.5% is *not* evidence that the scanner
could stand alone at 95.5%. As a **labelling accelerator** the certain tier
works. As an **error finder** it contributes one event.

**The scanner does find reconstruction errors — just not confidently.** On the
14 events the reconstruction gets wrong it is right on **8 (57%)**, against 38 of
46 (83%) where the reconstruction is right. But only one of those 8 was flagged
`certain`-and-disagreeing (evt175808, a numuCC junction with a 9.5 cm proton
leaving it — scanner right, reconstruction wrong). The rest land in `likely` or
`unclear`, i.e. in the pile the human reviews anyway.

**And §7's "it never confidently endorsed a wrong reconstruction" does not
survive n=60.** It did so once here: one `certain` pick agreed with a
`main_vertex` that was wrong. Once in 60 is a rate, not never.

### 10.7 Verdict, replacing §7's

**Yes, with a defined split.** The owner's original request — scan new events,
hand back the doubtful ones — is supported by the measurement: 37% of events can
be auto-accepted at ~95%, and the 63% handed back carry ~86% of the
reconstruction's errors. The scan is at parity with the reconstruction overall
(46/60 vs 46/60 held out, 43/60 vs 43/60 on dev), not better.

What it is still **not** good for: generating labels that a model is then fitted
to without review, and being trusted as an independent error-finder on the
strength of its `certain` tier alone.

The named prediction registered in the plan — that the eye would keep making the
R1-over-R2 ordering error even with the connectivity sheet in front of it — is
resolved in §10.8, and the answer is no.

### 10.8 The diagnostic — the original twenty, rescanned last

Run only after §10.5 was scored and committed, because whoever launches it knows
those answers. Two fresh subagents, new kit, same twenty events as §7.

```
answered 19/20, correct 14 @1cm, 15 by vertex id     (§7 old kit: 11 / 12)
reconstruction on the same 20: 15
```

**Four of §7's five misses are fixed, and by the panel that was built for each:**

| event | §7 | now | what fixed it |
|---|---|---|---|
| evt282737 | 21000, −165 cm | **21003 ✓** | the evidence sheet — the junction both Bragg ends point away from |
| evt283991 | 15002, −128 cm | **15003 ✓** | the sheet: seg 15004 rising 90k → 305k away from it |
| evt174637 | 9014, −10.6 cm | **9006 ✓** | the P4 profile: a textbook monotone rise on the 26 cm seg 9039 |
| evt52085 | 6002 | **6002 ✓** | nothing — it was always right (F2) |
| evt399856 | 3002, −44 cm | 3002, **still wrong** | — |

So the registered prediction is **falsified**: the R1-over-R2 ordering error was
a *visibility* problem, and showing the connectivity dissolved it.

**What survives is more interesting, and it is a limit on the owner's rule 1.**
All three `certain`-but-wrong picks on this set (evt399856, evt388224, evt59085)
are events where **rule 1 fires cleanly and is wrong**. evt399856 is the clearest:
dQ/dx is hot at both free ends and cold in the middle, so both prongs
demonstrably rise *away* from the middle junction 3002 — rule 1 selects 3002
unambiguously — and the owner picked 3001, the upstream free end. The scan is not
misreading the evidence; the evidence says 3002 and the label says 3001.

That produces the one inverted calibration in this round:

| tier | coverage | by vertex id |
|---|---|---|
| certain | 11/20 (55%) | 8 (**72.7%**) |
| likely | 6/20 (30%) | 6 (100%) |
| unclear | 2 answered | 1 (50%) |

`certain` below `likely`, against 95.5% on the held-out sixty. These twenty are
not a random sample — they are §7's sample, which happened to contain several
rule-1 traps — so this is not evidence that the held-out calibration is unstable.
It is evidence of a specific, nameable hazard:

> **Better evidence for a rule that is wrong on a case makes the scanner
> confidently wrong rather than quietly wrong.** The old kit missed evt399856 at
> low confidence; the new kit misses it at `certain`, because it can now *see*
> that both prongs point away. Any future round that strengthens rule-1 evidence
> should expect the confident-error rate on this class to go up, not down.

**The class is real and unified — checked, not assumed.** Measuring the angle
between prong directions at each wrongly-picked junction:

| event | picked | truth | most back-to-back prong pair |
|---|---|---|---|
| evt399856 | 3002 | 3001 | **177°** (segs 3001, 3002) |
| evt388224 | 3011 | 3004 | **176°** (segs 3026, 3027) |
| evt59085 | 3004 | 3001 | **175°** (segs 3006, 3007) |

All three are the same failure: the scanner named a point that is *interior* to a
single scattered particle, where rule 1 is satisfied by construction because both
halves of one track cool toward the middle.

**But the obvious fix does not work, and it is worth saying so before anyone
builds it.** Base-rating "has a ≥150° prong pair" over the dev half:

```
TRUTH vertices                        n= 107   >=150deg pair:  24 (22.4%)
other deg>=2 vertices, same cluster   n= 299   >=150deg pair: 144 (48.2%)
```

A 2.2× enrichment *against* being the vertex — real signal, but **22.4% of true
vertices are also collinear**, so as a veto it would discard roughly one correct
answer in four and a half. This is pr/79 §8's lesson arriving again: an override
of a mechanism that is right far more often than wrong needs precision this does
not have. Collinearity belongs in the scan procedure as a **caution that lowers
confidence**, not as a rule that eliminates a candidate — which is exactly the
treatment §3 already gives owner rule 5.

## 11. Files

| file | what |
|---|---|
| `vtx_rules/scankit.py` | the §9.4 kit: sanitize + P1/P2/P4/P5/P6 + the `zoom` CLI + `selftest` |
| `vtx_rules/scan_prompt.md` | the fixed scanner instruction, handed verbatim to every arm |
| `vtx_rules/vtx_rules.py` | the decision procedure of §3 |
| `vtx_rules/vtx_geom.py` | dQ/dx and Bragg-end primitives; reads neither `rr` nor `dirsign` |
| `vtx_rules/vtx_io.py` | label loading, the 1 cm convention, dump helpers |
| `vtx_rules/make_split.py` | the frozen split; refuses to overwrite |
| `vtx_rules/baselines.py` | B0–B3b and the pr/79 reproduction gate |
| `vtx_rules/polarity_control.py` | §2 |
| `vtx_rules/eval_rules.py` | scoring, per-rule tables, precision-vs-coverage |
| `vtx_rules/render_event.py` | the PNG an agent opens with the Read tool; `--blind` hides the reco vertex and the engine pick |
| `vtx_rules/selfscan.py` | the blind self-scan harness of §7 and §10; all-cluster candidate pool, `--kit old\|new`, per-worker picks with mtime audit, `compare` |
| `vtx_rules/runs/` | the split, the dev/test result TSVs, and `scan2-20260815/` — the four arms of §10 |

`.gitignore` carries a `test*` rule, so `vtx_rules/runs/test-final*` is committed
with `git add -f`. Re-running the Repro block rewrites those files and **git will
not show them as dirty** — compare them by hand if you need to know whether the
numbers moved.

Determinism: two consecutive runs produce byte-identical TSVs, and both match
the committed `vtx_rules/runs/dev-stage3/events.tsv`.

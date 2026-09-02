# doc 94 — recovering neutrino events lost to the STM tagger

**Round 1 CLOSED. `stm_vertex_hadron_guard` is SBND PRODUCTION as of
2026-09-02 (owner authorized, §0.4). 3 of 4 recovered at zero measured cost.**

**Round 3 CLOSED. `stm_entry_rise_guard` is SBND PRODUCTION as of 2026-09-02
(owner authorized, §13.11; `ref/prod-2026-09-03`).** It releases **2** data
bundles of 34,827 and the owner adjudicated **both as neutrinos**, so measured
contamination is **zero**. Right on **10 of 11** hand-labelled bundles.

**The target set is 5, not 4.** `707-18-12` was adjudicated a genuine STM on
2026-09-02 and **re-adjudicated a neutrino the same day** — the owner was right
at the start of this campaign, all five events are neutrinos. Four are
recovered; `707-18-12` is not, and §14 measures why no tuning of this mechanism
can reach it.

Predecessors: doc 63 (the STM improvement campaign, closed at 3 irreducible
errors), doc 62 (`scan-d59k/stm-baseline.tsv`, the 72 bundles the owner
adjudicated by hand — the truth set used here), doc 93 (the colleague's 8-event
feedback sample).

## Headline

| | |
|---|---|
| symptom | 5 events our chain tags STM that the owner said are neutrino candidates |
| ~~after adjudication~~ | ~~707-18-12 is a genuine STM~~ — **re-adjudicated a NEUTRINO** the same day (§13.1). The target set is **5** |
| **recovered** | **4 of 5** — 966-2-22, 304-6-28, 146-60-31 (round 1) and 827-27-4 (rounds 2–3) all flip STM → **nu-candidate**. `707-18-12` is not recovered (§14) |
| cost on the **3067** data events | 1 bundle flips of 34,827 — and the owner calls it a **neutrino**, so the measured cost is **zero** |
| owner's 36 confirmed-correct STMs broken | **0** |
| ~~still not recovered~~ | **827-27-4 recovered** by `entry_rise_guard` (§12–13) |
| still not recovered | **707-18-12** — no entry rise exists on it (entry 1.04 MIP on a 0.84 MIP body); its signature is a mid-track two-prong vertex at MIP charge, which is round 4 (§14) |
| ~~round-2 cost~~ | round 2 released one of the owner's STMs and missed one of his neutrinos; **round 3 fixed both** (§13) |
| round-3 cost on the **3067** data events | 2 bundles flip of 34,827, **both owner-adjudicated neutrinos** — measured cost **zero** |
| hand-labelled bundles correct | **10 of 11** (round 2: 8) |
| ideas measured dead | travel direction (§4), `proton_muon_guard` re-tune (§5) |

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 1. instrument the 5 symptom events (save_stm_fit + TRACE)
./scripts/doc94_stmfit_instrument.sh                      # -> work-stmfb8-stmfit
python3 scripts/doc94_stmfit_probe.py work-stmfb8-stmfit 4:18 22:6 28:5 12:7 31:12 17:20

# 2. probe the descent feature over the population, cut disabled
PR_JOBS=24 ./scripts/doc94_probe_arms.sh                  # -> work-*-d94probe

# 3. the vertex-hadron guard: 8 events, then the population
PR_JOBS=4  ./scripts/doc94_arm2.sh work-stmfb8-ql work-stmfb8-hadron sim -
PR_JOBS=4  ./scripts/doc94_arm2.sh work-stmfb8-ql work-stmfb8-negctl sim - stm_hadron_len_cm=30.0
PR_JOBS=24 ./scripts/doc94_hadron_arms.sh                 # -> work-*-d94hadron

# 4. all four censuses
./scripts/doc94_measure.sh
```

Binaries pinned: `~/tmp/doc94-libsnap` (probe arms), `~/tmp/doc94b-libsnap`
(hadron arms). A peer session shares `local/lib`.

## 0. Owner adjudication — 2026-09-02

Three calls from the owner after scanning the round-1 package. All three change
the arithmetic, and one of them opens round 2.

### 0.1 `64475:23` is clearly a neutrino

The single bundle `vertex_hadron_guard` releases in 3067 data events (§9) is a
neutrino. **The guard therefore has no measured cost at all**: 3 recovered
neutrino candidates, 1 released bundle, and that bundle is also a neutrino.
The condition stated in round 1 for flipping it on for SBND is met.

### 0.2 `707-18-12` is a genuine STM — a muon coming in

> **SUPERSEDED, 2026-09-02, by the owner:** *"this event 707-18-12 is actually
> a neutrino, not STM. I was wrong reading through the truth."* The section is
> kept because rounds 1–2 were designed around it — it was the negative control
> that justified anchoring the elevated run at the boundary — and §14 records
> what changed and what the replacement justification is. **All five events the
> owner flagged at the start of this campaign are neutrinos.**

Our tag on it was **correct**. It was never a recovery target, so the round's
denominator is **4**, not 5, and the score is **3 of 4**. This also removes it
from §6: there is no "second unrecoverable event", only one.

It is now something more useful than a residual — it is a **negative control
with an owner verdict**, and the only true-STM example in this 6-event set.

### 0.3 `827-27-4` is a neutrino, and the tell is at the ENTRY end

The owner's words: *"The telling feature is actually on the rise of dQ/dx near
the exiting detector (or entry point). This is a signature of two particles,
one particle going out of the detector. I understand that you have been testing
the stopping point candidate."*

That last sentence is the diagnosis of round 1's blind spot, and it is correct:
**every predicate in the tagger looks at the STOP end.** `eval_stm_core`
searches for the Bragg peak in a window that *ends* at the kink;
`detect_proton` tests the last 20–35 cm; all five doc-63 guards read the stop
region; and doc 94's own `vertex_hadron_guard` reads prongs. Nothing in
`TaggerCheckSTM.cxx` reads the charge profile at the boundary point where the
fit *starts*.

**The physics.** At the boundary end a single muon is at its most energetic, so
its dQ/dx there must be at its *lowest*, ≈ 1 MIP. Charge well above MIP at the
boundary that then **decays to the body level** over the next 10–20 cm is two
particles sharing that stretch, one of which leaves the detector.

**Measured on the 6 events** (`scripts/doc94_entry_rise.py`, plot
`products/doc94/scan/entry_rise.png`; dQ/dx from the `save_stm_fit` trajectory,
entry = median over the first 3 cm from the boundary point, body = median over
the middle of the track):

| event | entry 0–3 cm | body | **entry/body** | decays to body by | owner verdict |
|---|---|---|---|---|---|
| **827-27-4** | 2.43 | 0.98 | **2.49** | ≈14 cm | **neutrino** |
| 304-6-28 | 3.25 | 1.51 | 2.15 | *never* — body is hot throughout | neutrino (already recovered) |
| 146-60-31 | 2.62 | 1.37 | 1.92 | ≈20 cm | neutrino (already recovered) |
| **707-18-12** | 1.04 | 0.83 | **1.26** | flat from L = 0 | **genuine STM** |
| 966-2-22 | 0.93 | 0.86 | 1.09 | flat from L = 0 | neutrino (recovered by the prong) |
| 36-77-17 | 0.91 | 0.95 | 0.96 | flat from L = 0 | control, correctly not STM |

The separation is clean where it matters: **827-27-4 at 2.49 against the true
STM at 1.26**, with the control at 0.96. 827-27-4's profile is the textbook
version of the owner's picture — 3.4–3.9 MIP at L = 0, decaying through 2.1
(3 cm), 1.8 (5 cm), 1.25 (9 cm) to a flat 1.05 by L ≈ 14 cm. Two tracks
overlapping for ~14 cm, then one leaves.

**Refinement the plot forces, and round 2 must respect it.** The feature is not
"the entry is hot" — it is "the entry is hot **and decays to the body level**".
304-6-28 has entry/body 2.15 but its body sits at 1.5–2.0 MIP for the whole
40 cm shown, so there is no decay; the ratio alone would call it for the wrong
reason. A round-2 predicate should key on the **decay** (entry charge falling to
the body level over a ~10–20 cm scale), not on a bare ratio.

**Two more cautions for round 2**, recorded now so they are not rediscovered:

1. **Anode confound.** 304-6-28 enters at the anode face (x = −201.4), where
   drift and recombination effects on dQ/dx are largest. The two clean firing
   cases do *not*: 827-27-4 enters the upstream z face and 146-60-31 the bottom
   y face, so the feature is not an anode artifact — but a population study
   must split by entry face.
2. **`ratio1` is the same fact seen from elsewhere.** §6 records 827-27-4's
   `ratio1 = 0.549`, i.e. measured charge ≈1.8× the muon template while the
   *shape* matches. That is exactly what two overlapping particles produce, and
   doc 55 already noted nothing gates on `ratio1` in the non-strong acceptance
   branch. The entry-rise feature and the `ratio1` anomaly are one phenomenon,
   which is mild independent support for both.

**Status: IMPLEMENTED in round 2 — see §12.** The predicate is
`entry_rise_guard`, and it keys on the contiguous elevated run *anchored at the
boundary*, which is a sharper statement of "decays to the body level" than the
ratio this section wrote down. It recovers 827-27-4 (shoulder 8.4 cm), leaves
707-18-12 alone (0.0 cm — its 1.8 MIP spike is at 2-4 cm, not at the boundary)
and leaves 36-77-17 alone (0.0 cm). The anode confound this section warned
about did not arise: none of the events in play enters at the anode except
304-6-28, which was already recovered in round 1.

### 0.4 The guard is flipped ON for SBND production

Owner, 2026-09-02: *"By the way, we can turn this knob on for SBND default
running with all these validation."*

`stm_vertex_hadron_guard` now defaults **true** in
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`. The C++ default
stays `false`, so no other detector moves. New pinned operating point:
**`ref/prod-2026-09-02/`** (`prod-2026-09-01c` left byte-untouched).

The drift is exactly three keys on one component, and nothing else:

```
DRIFT     : bare_prjob.json, prod_prjob.json, sbnd_pr.json
  ADDED   [17].data.vertex_hadron_guard = True
  ADDED   [17].data.guard_hadron_len_cm = 12
  ADDED   [17].data.guard_hadron_mip    = 1.5
```

`[17]` is `TaggerCheckSTM:pr`. The other **18 of 21** artifacts — uboone, six
pdhd, five pdvd, the four other sbnd jobs, `prod.standalone`, `prod.wcls` — are
byte-identical. `prod_cfg_gate.py` **PASSes against `prod-2026-09-02`**.

**The production default is the validated configuration, checked not assumed:**
the compiled `TaggerCheckSTM` data block of `work-mcp1k-d94hadron` — the arm
that produced the 3067-event A/B, which forced the knob on through
`PR_EXTRA_TLA` — and of the new reference differ in **0 of 28 keys**.

This is **not byte-identical** and is a deliberate physics change; that is why
it gets its own reference generation. Reproduce the old behaviour for an A/B
with a `PR_EXTRA_TLA` line `stm_vertex_hadron_guard=false`.

`descent_guard` remains absent from the production config and must stay that
way (§4).

## 1. The sample and the baseline

Doc 93 established that our chain agrees with the colleague's event for event.
The owner then stated that all 5 STM events are actually neutrino candidates.
**That premise is the owner's assertion and is taken as given** — this document
does not attempt to prove it. What it establishes is *which code path tagged
each event and by what margin*, and what it costs to change that path.

The validation sample is the 3067 numuCC **data** events
(`work-{mcp1k,mcp2k,ncpi0,nuecc48}-prod0901b`). Baseline STM census:

| | count |
|---|---|
| events with ≥1 in-beam STM bundle | **416** of 3067 (13.6%) |
| … STM-only, i.e. releasable | 398 |
| in-beam STM bundles | 417 of 2576 (16.2%) |
| STM tags in the nueCC48 and NCπ0 arms | **0 / 0** |

The nueCC48/NCπ0 zeros reproduce doc pr/1 and are the check that the tables
were read correctly.

## 2. Where each of the 5 was tagged

`save_stm_fit` (doc 41) records a per-pass status. All five accepted on the
**forward pass with status 0** — eval passed, and neither `check_other_tracks`
nor `detect_proton` stopped it. All five are `flag_double_end=false`, so there
is no second orientation to fall back on.

| event | cluster | body MIP | end-5cm med / peak | ks1 | ratio1 | prong | what let it through |
|---|---|---|---|---|---|---|---|
| 827-27-4  | 18 | 1.08 | 3.52 / **5.88** | 0.022 | **0.549** | none | nothing to catch — see §6 |
| 966-2-22  |  6 | 0.96 | 0.85 / 1.55 | 0.064 | 0.884 | **14.9 cm @ 2.47 MIP** | `check_other_tracks` perpendicular skip |
| 304-6-28  |  5 | 1.58 | 1.79 / 3.32 | 0.030 | 0.997 | **25.9 cm @ 2.22 MIP** | `:2909` hot-curved skip **+** `proton_muon_guard` |
| 707-18-12 |  7 | 0.85 | 1.60 / 4.32 | 0.115 | 1.113 | none | nothing to catch — see §6 |
| 146-60-31 | 12 | 1.23 | 1.85 / 3.36 | 0.019 | 1.045 | **14.8 cm @ 1.65 MIP** | `:2909` hot-curved skip **+** `proton_muon_guard` |
| **36-77-17** (control) | 20 | 0.96 | 2.00 / 2.53 | 0.047 | 1.175 | 2.9 cm @ 1.17 | **status 5** — `detect_proton` called a proton end and it stood |

The control is the sharpest single fact in this table: 36-77-17 is saved by
`detect_proton`, and 304-6-28 and 146-60-31 were saved by it too until
`proton_muon_guard` overruled the call. See §5.

**This corrects doc 93 §4**, which concluded that 36-77-17's `STM=0` came from
the core fit-based condition. It did not: pass status is **5**, so `flag_pass`
was TRUE — the eval accepted it as a stopping muon — and `detect_proton` is
what saved it. Doc 93 §4 now carries a pointer here. The correction is
load-bearing rather than cosmetic: the whole §5 argument depends on the
control being saved by `detect_proton` at `ks1 = 0.047`.

## 3. The fix that works — `vertex_hadron_guard`

`check_other_tracks` (`TaggerCheckSTM.cxx:2835`) is the last topological gate
before an STM accept. It was ported to find **a second muon**. Every
acceptance clause in it demands either high straightness (`>0.99` / `>0.975`)
or MIP-band charge; `:2909` skips a hot *curved* segment by name
(`if (medQ>1.5 && len>8 && straight<0.9) continue;`); and
`second_track_guard`'s round-4c clause is bounded to 0.4–1.5 MIP with
len > 15 cm.

A **proton from a neutrino vertex is short, heavily ionizing and curved**, so
it falls through all of it. This is a muon-hunting predicate being asked to see
a hadron.

`vertex_hadron_guard` vetoes an STM accept whose fitted main carries a prong
longer than `guard_hadron_len_cm` (12 cm) at more than `guard_hadron_mip`
(1.5 MIP) — **with no straightness requirement**, which is exactly what
`:2909` gets wrong.

**Placement, stated deliberately.** The guard sits *before* the
`|angle−90°| < 7.5°` perpendicular skip. That skip discards near-isochronous
segments where 2-D reconstruction noise lives; a segment carrying >1.5 MIP over
>12 cm is not noise, so the concern the skip exists for does not apply to it.
This is not a guess — 966-2-22's prong measures **90° to drift** and reaches
that skip, so nothing placed after it can see the prong at all. It returns
`true` like every other veto in that function, i.e. terminal with no backward
pass, which is right for a prong: the prong is a property of the cluster, not
of which end the fit called the stop.

### It fires, and only where intended

```
vertex_hadron_guard: cluster  6 rejected: 14.9 cm prong at 2.47 MIP, straightness 0.971,  90 deg to drift
vertex_hadron_guard: cluster  5 rejected: 25.9 cm prong at 2.22 MIP, straightness 0.719, 150 deg to drift
vertex_hadron_guard: cluster 12 rejected: 14.8 cm prong at 1.65 MIP, straightness 0.861, 144 deg to drift
```

| event | OFF | ON |
|---|---|---|
| 966-2-22 | STM | **nu-candidate** |
| 304-6-28 | STM | **nu-candidate** |
| 146-60-31 | STM | **nu-candidate** |
| 827-27-4, 707-18-12 | STM | STM (no prong) |
| 36-77-17 (control) | nu-candidate | nu-candidate |
| 921-29-10, 658-38-25 | TGM | TGM |

The three do not merely lose the STM tag — they become **nu-candidates**, i.e.
they now reach `TaggerCheckNeutrino` and are reconstructed as neutrinos. That
is the recovery the owner asked for.

### Causal negative control

Raising `stm_hadron_len_cm` from 12 to 30 — above all three prong lengths and
changing nothing else — gives **0 fires** and returns all 8 events to verdicts
byte-identical to `work-stmfb8-pr`. The guard is keyed on exactly the feature
claimed for it, not on some correlate. (`work-stmfb8-negctl`.)

### What it costs — measured, all 3067 data events

| | |
|---|---|
| STM=1 bundles carrying the descent probe | 401 |
| … with **any** logged prong | 47 |
| … with a prong len>12 cm **and** medQ>1.5 MIP | **1 of 401 (0.25%)** |
| the one bundle | `64475:23` — main 340.4 cm, prong 20.0 cm @ 2.38 MIP, **straightness 0.894** |

16 bundles in the whole sample carry a firing prong, but **15 of them already
have `stm=0`** — the existing clauses caught them, because their prongs are
straight (0.92–0.996, i.e. above the `straight>0.975` acceptance clause). The
guard's marginal effect is confined to the *curved* ones, which is precisely
the `:2909` blind spot it was written for. `64475:23` is the only STM=1 bundle
among them, and at straightness 0.894 it is exactly that class.

### The A/B, measured — `work-*-d94hadron` vs `work-*-prod0901b`

Not predicted from the probe: the guard was actually run over all 3067 events
and the per-bundle verdicts compared (`scripts/doc94_flip_report.py`, keyed on
`(event, main_id)` from `nusel-evt<ID>.tsv`, reporting one-arm-only bundles
separately rather than dropping them).

```
events compared            : 3067
bundles identical          : 34826
bundles FLIPPED            : 1
bundles only in OFF / ON   : 0 / 0
  evt 64475 main 23  len 340.4cm  stm:1->0 label:STM->nu-candidate
```

**One flip in 34,827 bundles**, in the direction intended, and it is the bundle
the probe-arm census predicted. Nothing gained a tag; nothing moved onto TGM;
no bundle appeared or vanished. The nueCC48 and NCπ0 arms are inside this
comparison and contribute zero flips.

### Negative control — the owner's own 72 verdicts

`scan-d59k/stm-baseline.tsv`. The join to `work-mcp1k-prod0901b` on
`(event, main_id)` was validated first, not assumed: **68 of 72 bundles agree
on both `len_main_cm` (±0.5 cm) and `flash_time_us` (±0.01 µs)**; 4 drifted
(doc pr/127's epoch-drift failure mode) and are excluded.

| owner class | in baseline | resolved | any prong | **firing prong** |
|---|---|---|---|---|
| code-STM-correct | 36 | 36 | 3 | **0** |
| code-FALSE-STM | 15 | 15 | 1 | 0 |
| code-MISSED-STM | 1 | 1 | 0 | 0 |
| code-not-STM-correct | 20 | 20 | 4 | **1** — `405432:15`, 18.9 cm @ 2.69 MIP |

Two things to read honestly here. **The guard breaks none of the owner's 36
confirmed-correct STMs.** But only 3 of those 36 carry any prong at all, so on
the specific failure mode a prong-aware guard would have — a Michel electron or
delta ray reading as a hadron — the control is *thin*, not silent: it says the
prongs that do exist on correct STMs are not hot, and 3/36 is the sample it
says it on. The one firing prong anywhere in the owner's set sits on
`405432:15`, class **code-not-STM-correct** — a bundle the tagger correctly did
*not* tag and the owner agreed. That is a point in the guard's favour, not
against it.

## 4. Measured dead — travel direction (`descent_guard`)

**The idea.** A cosmic stopping muon must have travelled *downward* to reach
its stop: muons arrive from the sky, so one that comes to rest inside the
detector entered a boundary face *above* its stopping point. The STM fit knows
both ends. Define `cos_y` = Δy/|Δ| of (stop − entry): −1 straight down, 0
horizontal, +1 straight up. Nothing else in the tagger reads direction — every
existing test reads charge or topology — so this is the "new information
source" doc 63 §9 required, and it is the PR-chain direction argument evaluated
on what the tagger already holds.

**On the 6 symptom events it looked perfect.** The five to recover measure
−0.240, +0.908, −0.167, +0.368, +0.360; the control measures **−0.770**. Three
of the five reach their stop travelling *upward* — 966-2-22 enters the bottom
face and climbs 72 cm.

**On the population it is dead.** Over all 3067 data events, 401 STM=1
bundles carry the probe:

| | |
|---|---|
| STM=1 bundles with `cos_y > −0.25` | **246 of 401 (61.3%)** |
| owner-confirmed **correct** STMs above that cut | **16 of 30 resolved** |

There is no cut that helps. To recover 827-27-4 the cut must sit below −0.240,
and at −0.25 the guard destroys **half of the owner's own confirmed-correct STM
tags**.

### Why it fails — and this is the durable part

`68956:13` is the case that explains it: entry outside the FV at the **bottom**
face, fitted "stop" 240 cm **higher**, owner class **code-STM-correct**. The
only consistent reading is that the fitted stop is a **clustering truncation,
not a Bragg stop** — the cosmic entered the top, was cut by clustering, and the
tagger fitted the surviving fragment. The owner's label means *"correctly
tagged as a cosmic"*, not *"the stop is physical"*.

So the feature is not weak — it is **undefined on a large subpopulation**. That
constrains the next round: **any geometric feature anchored on the fit's
`pts[kink]` inherits the same defect.** A second, independent way the premise
fails: `65647:12`'s entry point is *inside* the FV, because the `temp_set`-empty
branch (`:3053`) picks `candidate_exit_wcps.at(0)`, a dead-region or
anode-clipped end rather than a geometric boundary crossing.

**`descent_guard` is shipped (default OFF) as the instrument that produced this
measurement, and is NOT recommended for production.** Do not flip it on.

## 5. Measured dead — re-tuning `proton_muon_guard`

`detect_proton` had called a proton end on **both** 304-6-28 and 146-60-31, and
`proton_muon_guard` (doc 63 round 2) overruled it:

```
detect_proton: proton_muon_guard: end matches the muon hypothesis (ks1=0.035, ratio3=1.033); not a proton
detect_proton: proton_muon_guard: end matches the muon hypothesis (ks1=0.031, ratio3=0.981); not a proton
```

The guard fires below `guard_proton_ks1 = 0.040`. The control 36-77-17 survives
only because its `ks1 = 0.047` is *just above* that bar. So the obvious lever is
to lower it — and it does not exist:

- to stop firing on 0.035 and 0.031, the bar must be ≤ 0.031;
- but doc 63 round 2 added this guard to recover a single missed STM,
  **62613:17 at ks1 = 0.030**, which needs the bar > 0.030.

The admissible window is `(0.030, 0.031]` — **0.001 wide**. That is not a
threshold, it is a coincidence, and tuning into it would be exactly the
"parameter tuned until the answer looks right" this tree forbids.

The population says the same thing from the other side: over 3067 events the
guard fires **238** times with `ks1 ∈ [0.008, 0.039]`, median 0.021, and **227**
of those fires are in events carrying an STM tag. Lowering the bar to 0.031
would cancel **50 of 238** fires blind, none of them examined. The direction is
abandoned.

The vertex-hadron guard separates the *same* two events with a margin of
**13 cm and 0.7 MIP** instead of **0.001 in ks1**.

## 6. Not recovered by round 1 — 827-27-4

> Written before the owner's adjudication; §0.2 has since established that
> **707-18-12 is a genuine STM**, so only 827-27-4 belongs in this section, and
> §0.3 supplies the mechanism this section says is missing. Kept as the record
> of what round 1 could and could not see.

Neither carries a prong: `search_other_tracks` returned nothing, so
`check_other_tracks` exits at `fitted_segments.size() <= 1` and no
topology-based predicate has anything to work with.

827-27-4 in particular is worth stating plainly: its dQ/dx is a **textbook
Bragg peak** — body 1.08 MIP rising smoothly to 5.88 MIP at rr = 0, with
`ks1 = 0.022` against the muon template. It is indistinguishable from a
stopping muon by every charge- or topology-based quantity available at STM
time. The one anomaly is `ratio1 = 0.549`: the measured charge is ~1.8× the
muon template in **normalization** while the **shape** matches, and doc 55
already recorded that nothing gates on `ratio1` in the non-strong acceptance
branch. That is n = 1 and is **not** a lead pursued in this round.

**Update (§0.3):** that anomaly was the right thread and it was dropped one
step too early. The owner localized it — the excess charge sits at the **entry**
end and decays over ~14 cm, which is two particles overlapping with one
exiting. Every predicate in this file reads the *stop* end, which is why
nothing saw it.

707-18-12 has body 0.85 MIP with a 4.32 MIP end peak; `detect_proton` ran to
block C and returned false.

## 7. Verification

| gate | result |
|---|---|
| `prod_cfg_gate.py`, both knobs off | **PASS 21/21** vs `ref/prod-2026-09-01c` (run twice: after each guard) |
| `prod_cfg_gate.py`, after the production flip | **PASS 21/21** vs the new `ref/prod-2026-09-02`; the drift from `01c` is 3 keys on `[17]` and 18 of 21 artifacts unchanged |
| production default == validated arm | `TaggerCheckSTM` data block differs in **0 of 28 keys** from `work-mcp1k-d94hadron` |
| compiled-config proof | `descent_guard`/`vertex_hadron_guard` families appear when on, **no keys at all** when off |
| `./build/clus/wcdoctest-clus` | 235 → 236 cases, **all pass**; new case `TaggerCheckSTM descent_guard is off and inert` (7 assertions) |
| freshness proof | `libWireCellClus.so` newer than `TaggerCheckSTM.cxx` before every arm |
| probe arm, cut disabled, vs `prod0901b` | verdicts **identical on 3067 of 3067 events** (`doc94_identity_check.py`), and on the 8 MC events (8/8) |
| causal negative control | bar raised above the feature ⇒ **0 fires**, 8/8 verdicts back to baseline |
| owner baseline join validated | 68/72 agree on length **and** flash time before any score was quoted |

## 8. Reported, not fixed here

Surfaced while mapping the arms; unrelated to this change, so it is recorded
and nothing more.

`tracking-pr.root:T_cluster` carries `tgm/stm/fc/lm/beam_flash` branches and the
compiled config has `save_in_scope = true`, but those five branches are
**identically zero on every row measured** (11,047 rows, 155 events, four
arms), while `in_scope` / `is_associated` / `flash_time_us` on the same rows are
filled. This contradicts `ref/prod-2026-09-01c/README.md`, which advertises
per-bundle flags in that tree. Consequence for this round: every flip table
here is built from `nusel-evt<ID>.tsv`, never from `T_cluster`.

## 9. Hand scan of the one release — `64475:23`  *(resolved: neutrino)*

The guard releases exactly one bundle in 3067 data events. There is no truth
for data, so it was scanned by eye. Images:
`products/doc94/scan/scan64475{,_zoom}.png`, rendered by
`scripts/doc94_scan_plot.py`, which deliberately draws **the detector and not
the verdict** — no tag, no guard result and no feature value appears on the
image being judged.

Re-run for the scan with full diagnostics: `work-d94scan-64475`. With the
guard on, `cluster 23 → STM=0 TGM=0` and the event becomes a nu-candidate
(a calib dump appears where there was none).

**What the picture shows.** A 240 cm track from the **anode** face
(x = −201, z = 270, y = −140) to (x = −125, z = 28, y = −152) — nearly
horizontal, 3° below the horizontal over its whole length. Its last ~25–40 cm
is *not* a single track coming to rest: the charge spreads into a compact blob
with at least two directions, and the fitted dQ/dx is **ragged, not a smooth
Bragg ramp** — flat at 0.95 MIP down to rr ≈ 28 cm, then spikes of 2.2, 3.2,
2.7 and 4.1 MIP scattered through the last 25 cm. A genuine Bragg peak is
monotone over the last ~10 cm.

**What the reconstruction says once it is released**, from the calib dump the
released event now produces:

| | |
|---|---|
| `numu_score` | **+2.117** (positive = neutrino-like) |
| `nue_score` | −15.000 (the not-filled sentinel) |
| `match_isFC` | 0 |
| reco Enu | 1092.0 MeV |
| main vertex | (−123.3, −152.7, 27.8) cm on cluster 23 |
| vertex segments | 24.9 cm typed µ, plus 2.0 cm typed **p** and 1.8 cm typed π |

The corroboration worth noting: `TaggerCheckNeutrino`'s vertex finder, which
knows nothing about this guard, independently places the interaction vertex at
**exactly the point the STM fit called the stop**, and puts a proton- and a
pion-typed stub there.

> **RESOLVED 2026-09-02 (§0.1): the owner calls it clearly a neutrino.** The
> guard's measured cost is therefore zero. The scan below is kept as the record
> of what the evidence looked like before the call.

**My verdict at the time: ambiguous, leaning neutrino-like — put to the owner
rather than decided.** For it: a 20 cm
prong at 2.38 MIP, an independent vertex on the same point, proton and pion
stubs, `numu_score` +2.12. Against it: a 240 cm near-horizontal track entering
the anode face is a very cosmic-like object, and a ragged end can be a stopping
muon with delta rays rather than a hadronic vertex.

The owner called it a neutrino, so the second branch is the one that applies:
**`vertex_hadron_guard` has no measured cost at all.**

## 10. Bee — the A/B pair

| arm | set |
|---|---|
| **OFF** (production, prod0901b) | https://www.phy.bnl.gov/twister/bee/set/fba260db-6834-46fa-aa95-2a759ce29269/event/list/ |
| **ON** (`stm_vertex_hadron_guard`) | https://www.phy.bnl.gov/twister/bee/set/6bf6dbc6-57b9-4f6b-80e4-5dbe8ef412d8/event/list/ |

9 events, same order in both, content-verified after upload. Indices 0–7 still
line up 1:1 with the colleague's set `9797078d-…`; **index 8 is `64475`**, the
one data bundle the guard releases. Annotated index:
`bee/d94/d94.index.txt`. (The earlier 8-event OFF-only set `abc72dc0-…` is
superseded by this pair and its sidecars have been removed.)

Both sets carry **`stm_fit-global`** — the fitted STM trajectory with its
per-point dQ/dx, i.e. the evidence the verdict is made on. Present wherever the
STM fit ran; absent on the two TGM events, which exit before it.

### Why most events carry no PR layers in the OFF set — and why that is the point

An STM tag does not merely label a bundle, it **costs the entire PR
reconstruction**. `TaggerCheckNeutrino` refuses a bundle whose main is
cosmic-tagged:

```
[nu_per_bundle] gid 10 activity 6 (L 93.9 cm) cosmic-tagged (TGM=false STM=true lm_flag=0); not a candidate
[nu_per_bundle] gid 10: no neutrino candidate among 1 evaluated activit(ies)
no main cluster selected (15 mains, 1 in-window); skipping.
```

so no PR graph is built, and `MultiAlgBlobClustering` then leaves the point
sets empty **by design**:

```
bee points set 'track_fit':    visitor TaggerCheckNeutrino:pr produced no PR graph; leaving the set empty (require_pr_graph)
bee points set 'shower_track': ... (require_pr_graph)
bee points set 'vertices':     ... (require_pr_graph)
```

and writes no calib dump either (which is why only 1433 of 3067 prod0901b
events have one). So `track_fit-global`, `shower_track-global`,
`vertices-global` and `mc` are missing on every cosmic-tagged event — that is
not a packaging gap, it is the defect this round is about.

**The A/B makes it visible**: compare indices 2, 3, 5 and 8 across the two
sets. In OFF they carry `clustering-pr-global` + `stm_fit-global` and nothing
else; in ON the four PR layers appear, because the event is now a
nu-candidate. Indices 0 and 4 (827-27-4, 707-18-12) stay bare in both — they
are the two not recovered. Index 1, the control, has them in both.

## 11. Recommended next step

1. ~~Is `64475:23` a neutrino or a cosmic?~~ **Answered (§0.1): a neutrino**,
   and the guard is **now SBND production** (§0.4, `ref/prod-2026-09-02`).
   Nothing outstanding.
2. ~~Round 2: the entry-end rise (§0.3).~~ **Done — §12.** `entry_rise_guard`
   recovers 827-27-4; 707-18-12 and 36-77-17 both measure 0.0 cm. What is
   still open is the **owner's adjudication of the two data bundles it
   releases** (§12.10), which is what gates the production flip.
3. **Round 4: `707-18-12`.** A mid-track two-prong vertex at MIP charge —
   §14.3 states the problem and why widening `vertex_kink_guard` is not the
   answer (two of the owner's own STMs carry 27° and 42° mid-track turns).
4. **Leave `descent_guard` OFF permanently** unless someone finds a stop
   definition that is not the fit's `pts[kink]`. §4 explains why nothing
   anchored there can work.
5. ~~827-27-4 and 707-18-12 have no information to act on.~~ **Both superseded
   by §0**: 707-18-12 is a genuine STM, and 827-27-4 has a measured feature the
   tagger simply never looks at.

---

# Round 2 — the entry-end rise, implemented

**Round 2 IMPLEMENTED and validated; NOT closed.** `entry_rise_guard` recovers
827-27-4, the one event round 1 could not, and the owner-adjudicated score
goes from 3 of 4 to 4 of 4. It stays **default OFF** until the owner
adjudicates the two data bundles it releases (§12.7). Round 1 used "CLOSED" to
mean *shipped to production*; this round has not earned that word.

## 12. `entry_rise_guard` — the predicate the owner described

### 12.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# instrument + probe the 8 MC events (min_cm=1000 => pure measurement)
PR_JOBS=8  ./scripts/doc94r2_arm.sh work-stmfb8-ql work-stmfb8-r2probe sim 1000.0
python3 scripts/doc94r2_entry_census.py --arm work-stmfb8-r2probe

# probe the feature over all 3067 data events, cut disabled
PR_JOBS=24 ./scripts/doc94r2_probe_arms.sh            # -> work-*-r2probe
python3 scripts/doc94r2_identity.py r2probe d94hadron
python3 scripts/doc94r2_entry_census.py --arm work-ncpi0-r2probe \
        --arm work-nuecc48-r2probe --arm work-mcp1k-r2probe \
        --arm work-mcp2k-r2probe --baseline \
        --out products/doc94r2/entry-census.tsv

# the guard ON: 8 MC events, the causal negative control, then the population
PR_JOBS=8  ./scripts/doc94r2_arm.sh work-stmfb8-ql work-stmfb8-r2on    sim 5.0
PR_JOBS=8  ./scripts/doc94r2_arm.sh work-stmfb8-ql work-stmfb8-r2negctl sim 5.0 stm_entry_frac=5.0
PR_JOBS=24 ./scripts/doc94r2_on_arms.sh r2entry 5.0    # -> work-*-r2entry
python3 scripts/doc94r2_flip_report.py r2entry d94hadron
```

Binary pinned to `~/tmp/doc94c-libsnap` (a peer session shares `local/lib`).
**The baseline is `work-*-d94hadron`, not `work-*-prod0901b`**:
`vertex_hadron_guard` is production as of `ref/prod-2026-09-02`, so diffing
against prod0901b would re-attribute round 1's three recoveries and one
release to this guard.

### 12.2 What the guard measures

`entry_rise_reject()` in `clus/src/TaggerCheckSTM.cxx`, evaluated beside the
doc-63 family and `descent_guard` and returning through the same `nullopt`
path. On the muon segment `[0, kink_recorded]` of the accepted fit:

| quantity | definition |
|---|---|
| `body` | median dQ/dx over `L ∈ [20 cm, L_stop − 25 cm]` — the muon level, with the entry shoulder and the Bragg region excluded. Fixed geometry, not knobs |
| `thresh` | `guard_entry_frac × max(body, 1 MIP)`. Clamping the body at 1 MIP stops a charge-deficient reconstruction from lowering its own bar |
| **`shoulder`** | length of the **contiguous** run **anchored at L = 0** over which the forward 5 cm running median stays ≥ `thresh`. **The feature.** |
| `shoulder_nofirst` | the same run re-anchored at the *second* fit point — the L = 0 systematic, printed but never gating |
| `excess` | MIP-equivalent extra track length inside the run: `∫ max(0, dQ/dx − body) dL / body`. Confined to the run, so unlike a windowed integral it does not rectify noise into a signal |

Reject when `guard_entry_min_cm ≤ shoulder ≤ guard_entry_max_cm`.

Two shape requirements carry the discrimination, and each is answerable to a
labelled event — this is why the predicate is not the bare entry/body ratio
§0.3 first wrote down:

* **Anchoring at the boundary** is what rejects **707-18-12**, the one event
  of the six the owner confirmed is a *genuine STM*. Its profile does carry a
  1.8 MIP spike — but at 2–4 cm, on top of a 0.98 MIP first window, so its
  anchored run is **0.0 cm**. A hot stretch that does not reach the boundary
  is a delta ray on an entering muon, not a particle that left the detector.
  Its bare ratio is 1.25, uncomfortably close to a 1.3 ratio cut; its
  shoulder is unambiguously zero.
* **A required decay** (`shoulder ≤ max_cm`) separates "hot at the entry, then
  MIP" from "hot everywhere" — the failure mode §0.3 predicted. It is not
  hypothetical: `350099:15` in the population sits at **48.8 cm** and is
  declined for exactly that reason (§12.7).

The 5 cm running median with ≥ 3 fit points is deliberate: one truncated `dx`
on a single fit point can neither create a run nor break one.

### 12.3 The six labelled events

Probe arm `work-stmfb8-r2probe`, cut disabled, so these are measurements and
not outcomes:

| event | owner verdict | L_stop | body | ent | rise | **shoulder** | nofirst | excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **827-27-4** | **neutrino (the target)** | 108.1 | 0.99 | 2.43 | 2.46 | **8.4** | 7.8 | 10.3 |
| 304-6-28 | neutrino (round 1) | 111.9 | 1.52 | 3.25 | 2.13 | 19.4 | 18.8 | 13.9 |
| **707-18-12** | **genuine STM** | 111.2 | 0.84 | 1.04 | 1.25 | **0.0** | 0.0 | 0.2 |
| 966-2-22 | neutrino (round 1) | 94.5 | 0.85 | 0.93 | 1.10 | 0.0 | 0.0 | 0.1 |
| 36-77-17 | control, not STM | 72.3 | 0.90 | 0.91 | 1.01 | 0.0 | 0.0 | 0.0 |
| 146-60-31 | neutrino (round 1) | 56.9 | — | — | — | *declined* | | |

`146-60-31` is **declined by `guard_entry_min_len_cm = 70`**: its muon reaches
the kink after 56.9 cm, which leaves no room for the `[20, L_stop − 25]` body
window. Checked, not asserted: `work-stmfb8-r2probe/pr_evt31` emits **no**
`entry_rise:` probe line at all while its `stm_pass` record shows a status-0
accept with `npts = 88` (so the `n < 8` gate cannot be the reason) and
`kink = 88`, i.e. out of range, so the stop is the path end at 56.9 cm. It was already recovered by `vertex_hadron_guard`, so nothing is lost —
but the decline is the honest cost of the length gate and §12.6 measures it.

### 12.4 Positive control — the target flips, nothing else does

`work-stmfb8-r2on` vs `work-stmfb8-hadron` (the round-2 baseline):

```
events compared            : 8
bundles identical          : 112
bundles FLIPPED            : 1
bundles only in OFF / ON   : 0 / 0
  evt 4 main 18  len 114.4cm  stm:1->0 label:STM->nu-candidate
```

All six labelled verdicts with the guard on:

| event | verdict | |
|---|---|---|
| 827-27-4 | **nu-candidate** | **recovered — this round** |
| 966-2-22 | nu-candidate | recovered, round 1 |
| 304-6-28 | nu-candidate | recovered, round 1 |
| 146-60-31 | nu-candidate | recovered, round 1 |
| 707-18-12 | **STM** | correct — the owner adjudicated it a genuine STM |
| 36-77-17 | nu-candidate | control, unchanged |

**The owner-adjudicated score is 4 of 4.**

The guard also *fires* on 304-6-28 (19.4 cm) without changing its verdict: that
bundle was already released by `vertex_hadron_guard`, and `entry_rise` runs
first, so the release is simply re-attributed. The flip table shows it as
unchanged, which is what it is.

### 12.5 Causal negative control

Not "turn the knob off" — that only tests the boolean. `guard_entry_frac` is
raised to **5.0**, so `thresh` becomes 5 MIP and *no* elevated run can form,
which corrupts exactly the feature the guard keys on while leaving the guard
switched on, the cut at 5 cm and every other threshold in place:

```
work-stmfb8-r2negctl (stm_entry_frac=5.0)  vs  work-stmfb8-hadron
  bundles FLIPPED : 0        entry_rise_guard fires : 0
```

The two arms differ in one number and the effect vanishes completely.

### 12.6 The population — the go/no-go, all 3067 data events

The probe (`stm_entry_min_cm=1000`, above the feature's range) is inert by
construction and measured inert in fact: **3067 of 3067 events byte-identical**
to `work-*-d94hadron` on every per-bundle field.

Of the **246** STM-tagged bundles that reach the guard with a long enough muon:

| shoulder | STM=1 bundles |
|---|---:|
| **exactly 0** | **235** |
| (0, 2.5) | 6 |
| [2.5, 5.0) | 2 |
| [5.0, 7.5) | 1 |
| [15, 20) | 1 |
| [40, 60) | 1 |

**The feature is bimodal, not a continuum** — 96% of the STM population sits at
exactly zero, and the entire non-zero tail is 11 bundles in 3067 events. That
is the go/no-go this round was required to answer before choosing a cut, and it
answers it: the target at 8.4 cm is not being carved out of a smooth
distribution. The whole tail, sorted:

| shoulder | event:cid | stm | rise | nofirst | excess | L_stop | verdict |
|---:|---|:---:|---:|---:|---:|---:|---|
| 48.8 | 350099:15 | 1 | 1.82 | 47.8 | 26.2 | 143.8 | **declined** — above `max_cm`, no decay |
| 18.5 | 164466:7 | 1 | 3.29 | 17.6 | 34.3 | 93.7 | **released** |
| *8.4* | *827-27-4* | *1* | *2.46* | *7.8* | *10.3* | *108.1* | *the MC target, for scale* |
| 5.5 | 95500:15 | 1 | 1.87 | 4.9 | 4.5 | 182.3 | **released** |
| 5.0 | 352796:10 | 0 | 1.48 | 4.2 | 2.6 | 176.6 | not STM anyway |
| 4.4 | 290316:10 | 1 | 1.78 | 3.6 | 3.2 | 244.4 | below the cut |
| 2.6 | 174642:9 | 1 | 1.45 | 2.1 | 1.2 | 356.5 | below |
| 2.1 | 289840:19 | 1 | 1.43 | 1.2 | 0.6 | 89.4 | below |
| 1.8 | 397194:13 | 1 | 2.03 | 1.1 | 1.6 | 261.0 | below |
| 1.3 | 95371:9 | 1 | 1.42 | 0.5 | 0.7 | 144.4 | below |
| 1.0 | 282033:13 | 1 | 2.14 | 0.0 | 0.2 | 185.3 | below |
| 0.8 | 176731:5 | 1 | 1.40 | 0.0 | 1.1 | 81.3 | below |
| 0.8 | 355106:7 | 0 | 1.60 | 0.0 | 1.2 | 361.1 | not STM |
| 0.6 | 284145:11 | 0 | 1.33 | 0.0 | 1.9 | 238.5 | not STM |
| 0.5 | 56257:13 | 1 | 1.79 | 0.0 | 1.4 | 113.7 | below |

**This table is the argument for the shape and against the ratio.** Four
bundles carry a bare `rise` of 1.79–2.14 — as high as or higher than plenty of
what a ratio cut would want to keep — and their anchored runs are 0.5–1.8 cm.
`282033:13` at rise 2.14 has a **1.0 cm** shoulder that vanishes entirely when
re-anchored one fit point in. The ratio and the shoulder are not the same
measurement, and only the shoulder is quiet on this population.

**The L = 0 systematic, measured.** `shoulder_nofirst` tracks `shoulder` to
within ~1 cm everywhere it matters — 8.4 → 7.8 on the target, 18.5 → 17.6 and
48.8 → 47.8 on the two large population values — so the boundary-most fit
point is *not* what creates these runs. The one exception is the marginal
release `95500:15`, whose 5.5 cm falls to **4.9 cm** (below the cut) when the
first point is dropped. That is recorded as a property of that bundle in
§12.7, not as a reason to change the predicate.

**`guard_entry_min_len_cm = 70` is a choice, not a measurement.** It was fixed
before any population data existed, and the only thing that derives it is the
body window: `[20 cm, L_stop − 25 cm]` needs `L_stop > 45` to be non-empty at
all, and 70 leaves 25 cm of window. The data now says what it costs — 170 of
416 STM bundles never evaluated, including one known-good target
(146-60-31 at 56.9 cm). It is **not** changed in this round: every number here
comes from one pinned binary, and lowering the floor is a round-3 question
with a cost that can be measured rather than argued.

**Coverage — stated, not buried.** The guard evaluates 246 of the **416**
in-beam STM bundles that reach it; the rest are declined because their muon is
too short for a body estimate. The split is clean: bundles *with* a probe have
median main length 165.7 cm, bundles *without* 53.3 cm. This guard is a
long-muon predicate and says nothing about short ones.

### 12.7 What it releases in data — and the one it declines

Two STM-tagged data bundles of 246, **0.81%**.

Both arms of the scan were re-run with `save_stm_fit=true`
(`scripts/doc94r2_scan_arms.sh`), so the profile and the fitted trajectory
below are read out of the guard's own inputs, not inferred.

#### `164466:7` — the cleanest example of the signature, and I cannot call it

| | |
|---|---|
| flip | STM → **nu-candidate** |
| feature | shoulder **18.5 cm** (17.6 without the first fit point), rise 3.29, excess **34.3 cm** of extra MIP track |
| profile | a smooth monotone decay, 3.9 MIP at the boundary → 1.0 MIP by L ≈ 21 cm, then flat MIP all the way to the Bragg peak. Textbook |
| boundary point | (78.3, 199.6, 388.1) cm — the **top y face** (active volume `y ≤ 199.3`) |
| end of the run | (71.3, 196.2, 371.4) cm — **3.4 cm below the top face** |
| geometry | a real kink there: the run travels 18.5 cm nearly *along* the top face (Δy = −3.4 cm), then the track dives (Δy = −48 cm over the remaining 75 cm) |

By charge this is the strongest instance of the owner's signature in the whole
sample — better than the target. By geometry it is the weakest: the putative
vertex sits 3.4 cm under the face **cosmic muons enter through**, and a
near-horizontal, heavily-ionizing 18 cm segment lying along the top face has an
obvious competing explanation (a second cosmic that clustering merged, or a
hard delta ray at the muon's entry). **Not adjudicated — owner call**
(Bee index 1).

#### `95500:15` — marginal, and the only fragile bundle in the sample

| | |
|---|---|
| flip | STM → **nu-candidate** |
| feature | shoulder **5.5 cm**, rise 1.87, excess 4.5 cm |
| fragility | `shoulder_nofirst` = **4.9 cm** — *below the 5 cm cut*. This is the one bundle of 246 whose verdict depends on the boundary-most fit point |
| profile | not a decay: 2.5 MIP at L = 2, 2.1 at 5, then further isolated spikes at 14–21 and 42 cm on a ~1 MIP body |
| boundary point | (−19.9, 56.0, 2.5) cm — the **upstream z face**, the beam-entry side, the same face as the target |

The face is right and the shape is not. **Not adjudicated — owner call**
(Bee index 2).

#### `350099:15` — the one `guard_entry_max_cm` declines

Elevated for **48.8 cm** of a 143.8 cm muon, with the body never returning to
MIP (1.5–2.0 MIP throughout). This is exactly the "hot everywhere, no decay"
case §0.3 warned a bare ratio would misclassify, and the upper bound declines
it. Boundary at (86.6, −193.7, 500.1) cm — the downstream-z / bottom-y corner.
Unchanged in both arms; shipped in the Bee (index 3) so the owner can overrule
the bound if they disagree.

#### The controls

`290316:10` (4.4 cm, the nearest bundle below the cut), `282033:13` (rise
**2.14** — as high as the target's 2.46 — but shoulder 1.0 cm, and 0.0 cm when
re-anchored one fit point in), `56257:13` (0.5 cm). All three stay STM in both
arms. `282033:13` is the one to look at: it is the case that shows the ratio
and the shoulder are not the same measurement.

#### An observation, offered and not acted on

Of the four boundary points in play, the two the owner has already adjudicated
as neutrinos (**827-27-4** at z = 4.3 cm, and **146-60-31** at the bottom y
face) and the marginal release **95500:15** (z = 2.5 cm) sit on faces a
*downward* cosmic muon cannot have entered through; the one release I cannot
defend, **164466:7**, sits on the top face, which is precisely where it can.

That suggests the natural next refinement: the guard's premise — *at the
boundary a cosmic stopping muon is at its lowest dQ/dx, because that is where
it entered with maximum energy* — only bites when the boundary point is a face
the muon could not have entered through. It is `descent_guard`'s logic (§4)
applied to the **boundary** point instead of `pts[kink]`, which sidesteps
exactly the defect that killed it: the boundary point is a real fiducial-face
crossing that `cluster_fc_check` computed, not a fit endpoint that is often a
clustering truncation.

**It is not implemented and no number here depends on it.** It is motivated by
n = 1, which is the shape of argument this campaign has repeatedly measured
dead. If the owner calls `164466:7` a cosmic it becomes round 3 with a
population measurement of its own; if they call it a neutrino it is wrong and
should be dropped.

### 12.8 The population A/B, measured

`work-*-r2entry` (guard on at 5 cm) vs `work-*-d94hadron` (the production
point), all 3067 SBND data events, per bundle, from `nusel-evt<ID>.tsv`:

```
events compared            : 3067
bundles identical          : 34825
bundles FLIPPED            : 2
bundles only in OFF / ON   : 0 / 0
  evt 164466 main 7   len 118.7cm  stm:1->0 label:STM->nu-candidate
  evt 95500  main 15  len 177.9cm  stm:1->0 label:STM->nu-candidate
```

Both flips are in the intended direction and are exactly the two bundles the
probe predicted. **No bundle gains a tag; nothing moves to TGM; no bundle
appears in one arm only.** The guard fires three times in 3067 events; the
third (`352796:10`, 5.0 cm) was already not STM in the baseline, so its
rejection is redundant and changes nothing — which is also the check that a
`nullopt` return is not perturbing which pass runs.

### 12.9 Verification

| gate | result |
|---|---|
| `prod_cfg_gate.py`, knob off | **PASS 21/21** vs `ref/prod-2026-09-02` — the production operating point does not move |
| compiled-config proof, knob on | `entry_rise_guard`/`guard_entry_*` appear (5 keys); **no keys at all** when off |
| `./build/clus/wcdoctest-clus` | 236 → **237 cases, all pass**; new case pins OFF, `max_cm > min_cm`, `frac > 1`, and `min_len_cm > 45` (the body window cannot be empty) |
| freshness proof | `libWireCellClus.so` 04:43:43 newer than `TaggerCheckSTM.cxx` 04:42:10, before any arm |
| binary pin | `~/tmp/doc94c-libsnap`, md5 `a9370b3b…` identical at the start and the end of the campaign |
| probe arm inert, MC | 8 of 8 events identical to `work-stmfb8-hadron` |
| probe arm inert, data | **3067 of 3067** events identical to `work-*-d94hadron`, every per-bundle field |
| positive control | 827-27-4 flips STM → nu-candidate; 1 flip in 113 bundles |
| causal negative control | `guard_entry_frac` 1.3 → 5.0 (the elevated run cannot form): **0 fires, 0 flips**, every verdict back to baseline |
| population A/B | 34,825 identical, **2 flipped**, 0 one-arm-only |
| owner baseline (doc 62) | **0 of 36 correct STMs break** — but see the caveat below |

**The owner-baseline control is thin, and saying "0 of 36" without the caveat
would overstate it.** Only **6** of the 36 correct-STM bundles are even
evaluated by this guard; the other 30 are declined because their muon is
shorter than 70 cm (24 of them have a main shorter than 70 cm outright). For
those 30 the "0 break" is *guaranteed by the length gate*, not measured. The
measurement that carries the weight is the population: **246 long-muon
STM-tagged bundles, 244 untouched**, and the true STM `707-18-12` at shoulder
0.0.

### 12.10 Status and what the owner has to decide

**Shipped default OFF.** `stm_entry_rise_guard=false` in
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`; C++ defaults
`false / 1.3 / 5 / 30 / 70`, keys suppressed when off, so
`ref/prod-2026-09-02` is untouched and every other detector is untouched.

Round 1's guard was flipped on only after the owner adjudicated its single
release a neutrino. The same question is open here, twice over, and the owner
said he is happy to judge:

1. **`164466:7` — cosmic or neutrino?** (Bee index 1.) The charge is the best
   example of the signature in the sample; the geometry puts the vertex 3.4 cm
   under the top face.
2. **`95500:15` — cosmic or neutrino?** (Bee index 2.) Right face, wrong shape,
   and it is the only bundle whose verdict moves if the boundary-most fit
   point is dropped.
3. **`350099:15` — should `guard_entry_max_cm` be raised?** (Bee index 3.)
   Currently declined at 48.8 cm because the charge never comes back to a MIP
   body.

**Recommendation.** If both releases are neutrinos: flip
`stm_entry_rise_guard=true` for SBND and pin a new `ref/prod-2026-09-03`; the
trade is +1 owner-adjudicated neutrino recovered (827-27-4) and 2 more
neutrino candidates for 34,827 bundles, i.e. the same shape of trade the owner
accepted in round 1. If `164466:7` is a cosmic, do **not** raise `min_cm` to
exclude it — that is a one-event fit; take the boundary-face refinement of
§12.7 to a round 3 and measure it. If both releases are cosmics, the guard
stays off and 827-27-4 is not recoverable by charge shape alone.

### 12.11 The Bee A/B pair

| | |
|---|---|
| OFF (production, `ref/prod-2026-09-02`) | https://www.phy.bnl.gov/twister/bee/set/def734e6-1dd8-4307-b9b3-07ae59bc0524/event/list/ |
| ON (`stm_entry_rise_guard=true`, `min_cm=5.0`) | https://www.phy.bnl.gov/twister/bee/set/8264254e-1c52-4c3c-b231-ca5114892e3f/event/list/ |

Same 8 events, same order, in both sets; annotated index
`bee/d94r2/d94r2.index.txt`. Both arms carry the `stm_fit-global` layer (the
fitted STM trajectory coloured by charge) — that is the layer to look at, and
the guard reads the end of it *away* from the Bragg peak.

| idx | event | what it is |
|---|---|---|
| 0 | 827-27-4 | **RESCUED** — the round-2 target |
| 1 | 164466 | **RELEASE — owner call** |
| 2 | 95500 | **RELEASE — owner call** |
| 3 | 350099 | declined by `max_cm`; shown so the bound can be overruled |
| 4 | 290316 | control, 4.4 cm — the nearest below the cut |
| 5 | 282033 | control, rise 2.14 but shoulder 1.0 cm |
| 6 | 56257 | control, 0.5 cm |
| 7 | 707-18-12 | negative control — the owner-confirmed genuine STM |

**The set contents were verified, not assumed.** In the OFF set all eight
events carry four layers and no PR reconstruction (they are all cosmic-tagged);
in the ON set exactly indices 0, 1 and 2 gain `track_fit` / `shower_track` /
`vertices` / `mc`, and nothing else does. That is an independent check that
these are the right two sets, in the right order, containing exactly the three
intended flips.

### 12.12 Reported, not fixed

The guard's DEBUG probe does not print the **boundary point's coordinates**,
so the boundary-face question in §12.7 had to be answered by re-running the
scan set with `save_stm_fit=true` and reading the trajectory back. Adding
`entry=(x,y,z)` to that line — log-only, no verdict — would make the census
self-contained. It was deliberately *not* added in this round: every number
above comes from one pinned binary, and changing the log line would have meant
re-running all four arms to use it.

---

# Round 3 — the owner's kink, and the two errors it fixes

## 13. `guard_entry_kink_deg` — "along the track, there is a kink"

### 13.1 What the owner's hand scan said

The owner scanned the round-2 Bee pair (§12.11) and adjudicated all six data
events, and later the same day **re-adjudicated `707-18-12`** after re-reading
the truth: *"this event 707-18-12 is actually a neutrino, not STM. I was wrong
reading through the truth."* Three verdicts moved:

| Bee idx | event | round-2 verdict | **owner** | |
|---|---|---|---|---|
| 1 | `164466:7` | released | **good neutrino** | ✔ correct |
| 2 | `95500:15` | released | **STM, not a neutrino** | ✘ **false release** |
| 3 | `350099:15` | declined at `max_cm` | **neutrino** | ✘ **missed** |
| 4 | `290316:10` | kept STM | STM | ✔ |
| 5 | `282033:13` | kept STM | STM | ✔ |
| 6 | `56257:13` | kept STM | STM | ✔ |
| 7 | `707-18-12` (MC) | kept STM | **neutrino** | ✘ **missed** |

**§0.2 is superseded.** The original doc-94 headline said the target set was 4
rather than 5 because `707-18-12` was a genuine STM. It is 5 again: **all five
events the owner flagged at the start of this campaign are neutrinos**, he was
right the first time, and the doc-93 chain agreed with our tag for the wrong
reason.

And the mechanism, in the owner's words:

> *"95500 looks like a STM, not a neutrino, the near entry point looks like
> fluctuation of dQ/dx with delta ray. Note, another key to separate the other
> events from this event is along the track, there is a kink (large angle
> change). This event does not have a large angle change, which makes it more
> STM like."*

Two things follow, and both are corrections to round 2's physics rather than
re-tunings of its numbers:

1. **Charge alone cannot tell an exiting particle from a delta ray.** Round 2
   assumed the anchored elevated run *was* the second particle. `95500:15`
   shows it can be a fluctuation with a delta ray on a single straight muon.
   What distinguishes them is that two particles meeting at a vertex make the
   fitted path **turn**.
2. **`guard_entry_max_cm = 30` was built on a false premise.** §0.3 argued that
   an elevated run that "never decays" is not the signature. `350099:15` runs
   elevated for 48.8 cm and the owner calls it a neutrino, so the premise is
   wrong. The bound goes to 60 cm and the kink now carries what it was there
   for.

### 13.2 The kink, measured

Largest direction change between two **5 cm chords** meeting at a fit point,
scanned over the muon segment `[0, kink_recorded]` with the **last 15 cm
before the stop excluded** (a muon scatters hard in its final centimetres, and
range straggling is not a vertex). Window and exclusion are fixed geometry,
not knobs; only the bar `guard_entry_kink_deg` is configurable.

The window was chosen by scanning, on the owner's labelled set, for the
largest margin between the firing neutrinos and the firing STM:

| chord half-length | firing neutrinos, min | firing STM (95500) | margin |
|---|---:|---:|---:|
| **5 cm** | **30.1°** | **13.8°** | **16.2°** |
| 8 cm | 24.0° | 16.4° | 7.6° |
| 10 cm | 21.9° | 9.8° | 12.1° |
| 12 cm | 22.3° | 9.9° | 12.4° |
| 15 cm | 21.1° | 12.0° | 9.1° |

5 cm wins on both counts: the widest margin, and the neutrino side is tightly
clustered (30–37°) instead of spread. The stop-region exclusion moves exactly
one number on the whole labelled set — 95500 from 17.3° to 13.8° — and no
verdict; that is the check that it is removing end-of-range scattering and not
quietly doing the separating.

**The bar is 22°**, the midpoint of the 13.8 – 30.1 gap.

### 13.3 The full labelled set — 12 bundles, 11 with an owner verdict

`shoulder ≥ 5 cm` **AND** `shoulder ≤ 60 cm` **AND** `kink ≥ 22°`:

| event | owner | L_stop | shoulder | kink | round 3 | round 2 |
|---|---|---:|---:|---:|---|---|
| **827-27-4** | neutrino | 108.1 | 8.4 | **33.2°** | release ✔ | release ✔ |
| **164466:7** | neutrino | 93.7 | 18.5 | **37.1°** | release ✔ | release ✔ |
| **350099:15** | neutrino | 143.8 | 48.8 | **30.1°** | **release ✔** | *missed* ✘ |
| 304-6-28 | neutrino | 111.9 | 19.4 | 32.6° | release ✔ | release ✔ |
| 146-60-31 | neutrino | 56.9 | 17.1 | 33.1° | *(declined: muon 56.9 cm)* | same |
| 966-2-22 | neutrino | 94.5 | 0.0 | 50.6° | keep — not this guard's | same |
| **707-18-12** | **neutrino** | 111.2 | **0.0** | 28.1° | **keep ✘ MISS** | same |
| **95500:15** | **STM** | 182.3 | 5.5 | **13.8°** | **keep ✔** | *released* ✘ |
| 290316:10 | STM | 244.4 | 4.4 | 24.2° | keep ✔ | keep ✔ |
| 282033:13 | STM | 185.3 | 1.0 | 27.4° | keep ✔ | keep ✔ |
| 56257:13 | STM | 113.7 | 0.5 | 41.0° | keep ✔ | keep ✔ |
| 36-77-17 | control | 72.3 | 0.0 | 31.1° | keep ✔ | keep ✔ |

**Round 3 is right on 10 of the 11 owner-adjudicated bundles; round 2 was
right on 8.** The one error is `707-18-12` — a **miss**, not a false release:
the guard leaves a neutrino tagged. That is the safer direction of error (it
admits no cosmic background), but it is an error and §14 says why the mechanism
cannot reach it.

`966-2-22` and `146-60-31` are also neutrinos this guard does not release — the
first has no shoulder, the second is below the muon-length floor — but both
were recovered in round 1 by `vertex_hadron_guard`, so the **chain** is right
on 11 of 12, missing only `707-18-12`.

**Neither test works alone, and the table is the proof.** Two of the owner's
four STMs clear the 22° bar on their own — `56257:13` at 41° and `282033:13`
at 27° — as does the not-STM control `36-77-17` at 31°, so a kink-only
predicate would release them. And the shoulder alone made both of round 2's
errors. What separates is the conjunction: *extra charge anchored at the
boundary **and** a path that turns*.

### 13.4 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the labelled-set measurement, offline from the round-2 scan arms' stm_fit
python3 scripts/doc94r3_kink_scan.py            # window scan + the 12-bundle table

# 8 MC events, then the population: probe (inert) and ON, concurrently
PR_JOBS=8  ./scripts/doc94r3_arm.sh work-stmfb8-ql work-stmfb8-r3on sim 5.0
PR_JOBS=14 ./scripts/doc94r3_arms.sh r3probe 1000.0     # -> work-*-r3probe
PR_JOBS=14 ./scripts/doc94r3_arms.sh r3entry 5.0        # -> work-*-r3entry
python3 scripts/doc94r2_identity.py r3probe d94hadron
python3 scripts/doc94r3_entry_census.py --arm work-ncpi0-r3probe \
        --arm work-nuecc48-r3probe --arm work-mcp1k-r3probe \
        --arm work-mcp2k-r3probe --baseline \
        --out products/doc94r2/entry-census-r3.tsv
python3 scripts/doc94r2_flip_report.py r3entry d94hadron
```

Binary pinned to `~/tmp/doc94r3-libsnap` (round 2 used `doc94c-libsnap`).

### 13.5 The same table, from the compiled guard

The numbers above come from the offline re-implementation; these come from the
guard's own DEBUG probe in the arm logs, which is what actually decides. The
C++ scans **fit points** where the offline scan uses a 0.5 cm grid, so its
angles run ~1–3° lower. The separation survives with room to spare:

| event | owner | shoulder | **kink (compiled)** | at L |
|---|---|---:|---:|---:|
| 827-27-4 | neutrino | 8.4 | **32.2°** | 73.9 |
| 164466:7 | neutrino | 18.5 | **34.7°** | 19.9 |
| 350099:15 | neutrino | 48.8 | **28.3°** | 60.5 |
| 304-6-28 | neutrino (r1) | 19.4 | 32.3° | 70.2 |
| 146-60-31 | neutrino (r1) | *declined* — muon 56.9 cm | | |
| 966-2-22 | neutrino (r1) | 0.0 | 47.2° | 66.0 |
| **707-18-12** | **neutrino** | **0.0** | 32.0° | 66.8 |
| **95500:15** | **STM** | 5.5 | **13.8°** | 37.7 |
| 290316:10 | STM | 4.4 | 18.3° | 5.2 |
| 282033:13 | STM | 1.0 | 24.6° | 143.4 |
| 56257:13 | STM | 0.5 | 42.3° | 45.4 |
| 36-77-17 | control | 0.0 | 29.9° | 49.3 |

Lowest firing neutrino **28.3°**, the firing STM **13.8°**, bar at 22° — a
14.5° margin measured by the code that ships.

### 13.6 Why `guard_entry_max_cm` moved 30 → 60

This is a **correction, not a re-tune**. §0.3 predicted that an elevated run
which "never decays" would be a different animal, and §12.2 built `max_cm` to
exclude it. `350099:15` is that case — elevated for 48.8 cm of a 143.8 cm muon,
body 1.18 MIP, never returning to 1 MIP — and the owner adjudicated it a
**neutrino**. The premise was wrong, so the bound goes.

Two things make the move safe rather than a fit to one event:

* **It is the kink, not the length, that was doing the work.** The failure mode
  `max_cm` was protecting against — a track that is hot end to end for
  reconstruction reasons — is a track with no vertex, so it has no turn. The
  kink tests that directly instead of using length as a proxy for it.
* **Nothing else in 3067 events lives in the widened band.** The round-2 probe
  measured every evaluated bundle's shoulder: exactly one exceeds 30 cm, and it
  is 350099 itself. Raising the bound to 60 therefore admits that bundle and
  nothing else, and the shoulder histogram is empty above 60.

### 13.7 The population, with the kink measured — all 3067 data events

The round-3 probe (`stm_entry_min_cm=1000`) is inert by construction and
measured inert in fact: **3067 of 3067 events byte-identical** to
`work-*-d94hadron` on every per-bundle field, so every number below is a
measurement and not an outcome.

Of the **246** STM-tagged bundles the guard evaluates:

| | STM=1 bundles |
|---|---:|
| shoulder inside `[5, 60]` cm | **3** |
| ... and kink ≥ 22° — i.e. **released** | **2** |
| held by the kink alone | **1** |

| shoulder | event:cid | rise | kink | outcome |
|---:|---|---:|---:|---|
| 48.8 | `350099:15` | 1.82 | **28.3°** | **released** — owner: neutrino |
| 18.5 | `164466:7` | 3.29 | **34.7°** | **released** — owner: neutrino |
| 5.5 | `95500:15` | 1.87 | **13.8°** | **held by the kink** — owner: STM |

Three bundles enter the shoulder window in 3067 events and the kink splits
them exactly the way the owner did.

**The kink is not rare, and that is the point.** Over the 246 evaluated STM
bundles it runs 4.8° to 70.0° with a median of **19.5°**, and **104 of them
(42.3%) clear the 22° bar**. A kink-only predicate would release 42% of the
STM population. (The doc-62 correct STMs alone span 6.3° to 44.9°.) It is only after the shoulder has selected the handful of bundles
with boundary-anchored charge that the turn carries information, and there it
splits them cleanly.

### 13.8 The A/B, measured

`work-*-r3entry` (guard on, `min_cm=5`, `max_cm=60`, `kink≥22°`) vs
`work-*-d94hadron` (production), all 3067 SBND data events, per bundle:

```
events compared            : 3067
bundles identical          : 34825
bundles FLIPPED            : 2
bundles only in OFF / ON   : 0 / 0
  evt 350099 main 15  len 139.9cm  stm:1->0 label:STM->nu-candidate
  evt 164466 main 7   len 118.7cm  stm:1->0 label:STM->nu-candidate
```

### 13.9 Verification

| gate | result |
|---|---|
| `prod_cfg_gate.py`, knob off | **PASS 21/21** vs `ref/prod-2026-09-02` |
| compiled-config proof, knob on | 6 keys appear (`guard_entry_kink_deg` = 22 among them); none when off |
| `./build/clus/wcdoctest-clus` | **237 cases, all pass**; 2634 assertions (round 2: 2629) |
| freshness proof | `libWireCellClus.so` 06:16:51 newer than `TaggerCheckSTM.cxx` 06:15:42 |
| binary pin | `~/tmp/doc94r3-libsnap`, md5 `8b4f44b4…` |
| probe arm inert | 3067 of 3067 events identical to the production baseline |
| positive control (MC) | 827-27-4 flips STM → nu-candidate; 1 flip in 113 bundles |
| owner's labelled set | **10 of 11** adjudicated bundles correct (round 2: 8 of 11); the miss is 707-18-12, §14 |
| rebuilt binary vs pinned | after the comment-only corrections of §14, the rebuilt library reproduces `work-stmfb8-r3on` on 8 of 8 events with byte-identical probe lines (`work-stmfb8-r3chk`) — so every arm above stands |
| doc-62 baseline | 0 of the 6 evaluated correct STMs fire (30 of 36 are below the muon-length floor — the caveat of §12.9 still applies) |
| population A/B | **34,825 identical, 2 flipped, 0 one-arm-only** — and both flips are bundles the owner adjudicated NEUTRINOS, so the measured cost is **zero** |

### 13.10 Status and recommendation

**Written before the flip; §13.11 is what happened.** The guard shipped
DEFAULT OFF with C++ defaults `false / 1.3 / 5 / 60 / 70 / 22`, which is where
the C++ defaults remain — the flip is in the SBND jsonnet only, so no other
detector moves.

What changed is that the two round-2 questions are now answered, by the owner,
and the code agrees with him on both:

| | round 2 | round 3 |
|---|---|---|
| owner-adjudicated bundles correct | 8 of 11 | **10 of 11** |
| owner-adjudicated neutrinos this guard releases | 2 of 4 | **3 of 4** |
| owner-adjudicated STMs wrongly released | 1 | **0** |
| neutrinos left tagged (misses) | 2 | **1** (`707-18-12`, §14) |

**Recommendation: flip `stm_entry_rise_guard=true` for SBND and pin
`ref/prod-2026-09-03`.** The condition round 2 set for the flip — the owner
adjudicating the releases — is met, and it is met in the strongest available
form: every bundle the guard releases in 3067 data events is a bundle the owner
has personally called a neutrino, and every bundle he called an STM is held.
That is the same bar `vertex_hadron_guard` cleared in §0.4, on a labelled set
three times the size. **Accepted by the owner — §13.11.**

Two things this round does **not** claim:

* **The labelled set is 11 bundles**, and the guard is right on 10. The
  population measurement is what says the guard is quiet (2 fires in
  3067 events), not that it is correct.
* **It does not recover `707-18-12`**, and §14 shows it cannot. Every claim in
  this file that rested on `707-18-12` being an STM has been rewritten, in the
  doc and in the C++ comments — including the round-2 justification for
  anchoring the run at the boundary, which is now made on measured separation
  instead (§14.2).
* **The boundary-face refinement of §12.7 is dead as stated.** It was proposed
  because `164466:7` sits on the top face and I could not defend it. The owner
  calls `164466:7` a **good neutrino**, so a top-face veto would have thrown
  away a real neutrino. Recorded as measured-wrong, not carried forward.


### 13.11 The guard is flipped ON for SBND production

Owner, 2026-09-02, after the contamination audit of §13.12:
*"please turn on this knob for SBND production."*

`stm_entry_rise_guard` now defaults **true** in
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`. The C++ defaults
stay `false / 1.3 / 5 / 60 / 70 / 22`, so no other detector moves. New pinned
operating point: **`ref/prod-2026-09-03/`** (`prod-2026-09-02` left
byte-untouched).

The drift is six keys on one component, and nothing else:

```
DRIFT     : bare_prjob.json, prod_prjob.json, sbnd_pr.json
  ADDED   [17].data.entry_rise_guard       = True
  ADDED   [17].data.guard_entry_frac       = 1.3
  ADDED   [17].data.guard_entry_min_cm     = 5
  ADDED   [17].data.guard_entry_max_cm     = 60
  ADDED   [17].data.guard_entry_min_len_cm = 70
  ADDED   [17].data.guard_entry_kink_deg   = 22
```

`[17]` is `TaggerCheckSTM:pr`. The other **18 of 21** artifacts — uboone, six
pdhd, five pdvd, the four other SBND jobs, `prod.standalone`, `prod.wcls` — are
byte-identical. `prod_cfg_gate.py` **PASSes against `prod-2026-09-03`**.

**The production default is the validated configuration, checked not assumed:**
the compiled `TaggerCheckSTM` data block of `work-mcp1k-r3entry` — the arm that
produced the 3067-event A/B through `PR_EXTRA_TLA` — and of the new reference
differ in **0 of 34 keys**.

Not byte-identical, and a deliberate physics change; that is why it gets its
own reference generation. A/B escape: a `PR_EXTRA_TLA` line
`stm_entry_rise_guard=false`.

Both SBND STM neutrino-recovery guards are now production:
`vertex_hadron_guard` (§0.4, `prod-2026-09-02`) and `entry_rise_guard`
(`prod-2026-09-03`). Between them they release **3 bundles in 3067 data
events**, and the owner has adjudicated **all three as neutrinos**.

### 13.12 The contamination audit the flip rests on

The owner's question before authorizing: *"I want to confirm after flipping
this knob, we do not have more contaminations in STM to neutrino samples."*

Every bundle, every tag field (`in_beam / tgm / stm / fc / lm / label`),
`work-*-r3entry` vs `work-*-d94hadron`, all 3067 events:

```
bundles compared                     : 34827
bundles unchanged                    : 34825
bundles that moved                   : 2
  evt 350099 main 15  stm 1->0, label STM->nu-candidate
  evt 164466 main 7   stm 1->0, label STM->nu-candidate
bundles that GAINED a cosmic tag     : 0
bundles present in only one arm      : 0
STM-tagged bundles                   : 416 -> 414
```

| sample | events | STM off | STM on | flips |
|---|---:|---:|---:|---:|
| ncpi0 | 19 | 0 | 0 | 0 |
| nuecc48 | 48 | 0 | 0 | 0 |
| mcp1k | 1000 | 142 | 141 | 1 |
| mcp2k | 2000 | 274 | 273 | 1 |
| **total** | **3067** | **416** | **414** | **2** |

Nothing moves in the other direction, nothing gains a cosmic tag, and the two
that move are the two the owner adjudicated as neutrinos.

**Independent corroboration on `164466:7`.** Once released it gets a full PR
reconstruction and the PID is a textbook νμCC 1p: **proton 19.7 cm at 3.22 MIP
plus muon 76.0 cm at 1.30 MIP** from one vertex, `numu_score = 2.06`. That is
evidence for the owner's call which does not come from the scan.

**The honest limits**, recorded with the flip rather than after it:

1. **The 22° bar is exercised by 3 bundles, not a population.** Only 3 of the
   246 evaluated bundles reach the kink test at all (13.8°, 28.3°, 34.7°), so
   the bar sits in a gap containing no data. A future bundle at ~20–27° would
   be decided by a threshold nothing has tested. The shoulder is the selective
   cut; the kink only arbitrates the few that pass it.
2. **Bounded, not zero.** Zero cosmic admissions observed gives a 90% CL upper
   limit of 2.3 events, i.e. **< 0.075%** of events could gain a cosmic. The
   binding constraint is now statistics, not the predicate: a larger data
   sample would tighten it.
3. **"Neutrino" here means the owner's hand scan** — for data there is no
   truth — on one sample (runs 18255/18259). The 170 short-muon STM bundles
   are never evaluated, so those are guaranteed untouched rather than merely
   measured so.

Unverified and flagged: downstream, `cosmict_flag = 1` fires on `350099` (and
0 on `164466`). Doc 85 established that `cosmic_flag` is a BDT feature fill
rather than a verdict, and the semantics of `cosmict_flag` were **not**
established in this round — so this is recorded as something to check, not as
a disagreement with the owner's call.

## 14. `707-18-12` — why this mechanism cannot reach it

The owner re-adjudicated `707-18-12` a **neutrino** on 2026-09-02 after
re-reading the truth. It was, until that message, this round's principal
negative control. Two things follow and both are recorded here rather than
quietly patched.

Repro: `python3 scripts/doc94r3_anchor_test.py`.

### 14.1 The measurement: there is no entry rise to find

| | |
|---|---|
| body | **0.84 MIP** |
| entry, 0–3 cm | **1.04 MIP** |
| first 5 cm running median | **1.00 MIP** |
| threshold at `frac = 1.3` | 1.30 MIP (`1.3 × max(body, 1 MIP)`) |
| **shoulder** | **0.0 cm** |

The threshold is floored at 1 MIP precisely so a charge-deficient
reconstruction cannot lower its own bar, and `707-18-12`'s first window sits
*exactly* on that floor. For the guard to fire, `guard_entry_frac` would have
to be **≤ 1.00** — a bar at or below MIP, which fires on every track in the
detector. This is not a threshold that needs adjusting; **the feature is
absent**.

Relaxing the *anchor* does not help either. Allowing the run to begin anywhere
within 1, 2, 3, 4, 5 or 8 cm of the boundary leaves `707-18-12` at 0.0 cm at
every tolerance (and moves no other labelled bundle by a millimetre) — its one hot point (1.8 MIP at 2–4 cm) cannot sustain a 5 cm
median at any starting offset.

### 14.2 Dropping the anchor entirely is worse, and this is what now justifies it

Round 2 justified anchoring with `707-18-12` — "a hot stretch that does not
reach the boundary is a delta ray on an entering muon". That argument is void.
The replacement is a measurement, not a story:

| bundle | owner | anchored run | longest run **anywhere** |
|---|---|---:|---:|
| 827-27-4 | neutrino | **8.4** | 35.7 |
| 164466:7 | neutrino | **18.5** | 18.5 |
| 350099:15 | neutrino | **48.8** | 48.8 |
| 707-18-12 | neutrino | 0.0 | 11.9 |
| 95500:15 | STM | **5.5** | 13.5 |
| 290316:10 | STM | 4.4 | 15.3 |
| 282033:13 | STM | 1.0 | 12.4 |
| 56257:13 | STM | 0.5 | 12.9 |

Un-anchored, the four STMs land at 12.4–15.3 cm and `707-18-12` at 11.9 cm —
**the same band**, so the feature separates nothing there. Un-anchoring would
gain `707-18-12` and lose `282033:13` and `56257:13`, both of which clear the
kink bar: one neutrino gained for two cosmics admitted. The anchor stays, and
it stays for this measured reason rather than the void one.

### 14.3 What `707-18-12` actually has, and why nothing in the file sees it

Its signature is **a 32° turn at L = 66.8 cm — 44 cm *before* the stop — with
MIP charge on both sides**. That is a different topology from this guard's:
not two particles *sharing* a stretch near the boundary, but a vertex in the
middle of the fitted path with one prong in and one prong out. Nothing
overlaps, so there is no charge excess anywhere.

`vertex_kink_guard` (doc 63 round 5b) is the closest existing predicate and it
misses on all three of its conditions:

| `vertex_kink_guard` requires | `707-18-12` has |
|---|---|
| the turn within `[L_stop − 12 cm, L_stop − 2 cm]` | the turn 44.4 cm before the stop |
| turn ≥ `guard_vertex_turn` = **45°** | **32°** |
| post-turn median > `guard_vertex_mip` = **2.2 MIP** | ~**0.9 MIP** |

**This is round 4, not a re-tune.** And it is not a matter of widening
`vertex_kink_guard`'s window: a mid-track large-angle turn at MIP charge is
common on genuine stopping muons — `282033:13` (27° at L = 143) and
`56257:13` (42° at L = 45) are both owner-adjudicated STMs — so a turn alone
releases cosmics. The round-4 question is what second observable distinguishes
a two-prong neutrino vertex from a scattered muon when *neither* side carries
excess charge. Nothing in the present feature set answers it, which is exactly
the situation doc 63 §9 called "no further round without a new information
source".

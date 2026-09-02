# doc 94 — recovering neutrino events lost to the STM tagger

**Round 1. Two candidate mechanisms measured dead, one shipped default-OFF and
recommended for the owner's scan.**

Predecessors: doc 63 (the STM improvement campaign, closed at 3 irreducible
errors), doc 62 (`scan-d59k/stm-baseline.tsv`, the 72 bundles the owner
adjudicated by hand — the truth set used here), doc 93 (the colleague's 8-event
feedback sample).

## Headline

| | |
|---|---|
| symptom | 5 events our chain tags STM that the owner says are neutrino candidates |
| **recovered** | **3 of 5** — 966-2-22, 304-6-28, 146-60-31 all flip STM → **nu-candidate** |
| cost on the **3067** data events | **1 bundle flips, of 34,827** — measured A/B, zero collateral |
| owner's 36 confirmed-correct STMs broken | **0** |
| not recovered | 827-27-4, 707-18-12 — mechanism identified, no predicate available at STM time |
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

## 6. Not recovered — 827-27-4 and 707-18-12

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

707-18-12 has body 0.85 MIP with a 4.32 MIP end peak; `detect_proton` ran to
block C and returned false.

## 7. Verification

| gate | result |
|---|---|
| `prod_cfg_gate.py`, both knobs off | **PASS 21/21** vs `ref/prod-2026-09-01c` (run twice: after each guard) |
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

## 9. Hand scan of the one release — `64475:23`

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

**My verdict: ambiguous, leaning neutrino-like — and this is the one thing in
this round I am putting to the owner rather than deciding.** For it: a 20 cm
prong at 2.38 MIP, an independent vertex on the same point, proton and pion
stubs, `numu_score` +2.12. Against it: a 240 cm near-horizontal track entering
the anode face is a very cosmic-like object, and a ragged end can be a stopping
muon with delta rays rather than a hadronic vertex.

If the owner calls it a cosmic, the measured price of `vertex_hadron_guard` is
**one cosmic per 3067 events entering the selection at `numu_score` +2.12**,
against three recovered neutrino candidates. If the owner calls it a neutrino,
the guard has no measured cost at all.

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

1. **One owner call decides this round: is `64475:23` (§9) a neutrino or a
   cosmic?** It is the single bundle the guard releases in 3067 data events,
   and my own scan says ambiguous-leaning-neutrino. If it is a neutrino, the
   guard has no measured cost and should go ON for SBND. If it is a cosmic,
   the trade is one cosmic per 3067 events entering the selection at
   `numu_score` +2.12, against three recovered neutrino candidates — still a
   good trade in my judgement, but the owner's to make. The Bee set in §10
   carries the 5 symptom events plus the control with the fitted trajectory
   drawn; `products/doc94/scan/` carries the images for `64475:23`.
2. **Leave `descent_guard` OFF permanently** unless someone finds a stop
   definition that is not the fit's `pts[kink]`. §4 explains why nothing
   anchored there can work.
3. For 827-27-4 and 707-18-12 the honest answer is that STM has no information
   to act on. The next place to look is not the tagger but whether the
   *neutrino* activity in those events is being clustered into the muon at all
   — which is an upstream question, not an STM one.

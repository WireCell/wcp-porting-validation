# doc pr/88 — hand-scanning the mcp2k vertices with the round-3 kit

**Status: IN PROGRESS.** Phases 0–2 complete and recorded below; Phase 3
(the owner's port-5017 pile) and Phase 4 (the auto-accept calibration fork)
are pending. No C++, no jsonnet, no knob, no production flip — **no A/B gate
applies to any part of this round**, stated so its absence is not read as an
omission.

Doc pr/82 processed 2000 new MCP2025C **data** events into `work-mcp2k-harv3`
and deliberately stopped before its §5, the scan. This is that scan. The
product is labels for the combined DL-vertex training pool (pr/82 §6).

**Why the doc is 88 and not 87**: `pr87` is already a live arm token in this
tree — `work-{nuecc48,ncpi0,mcp1k}-pr87ion3`, referenced throughout docs 82
and 86. A doc numbered 87 would read as describing those arms.

---

## 0. Repro

All read-only except where noted.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# (a) the scannable pool: 880 dumps -> 845
python3 - <<'EOF'
import glob, json, os
d = sorted(glob.glob("work-mcp2k-harv3/pr_evt*/calib-pr-evt*.json"))
empty = [p for p in d if not (json.load(open(p)).get("vertices") or [])]
print(len(d), "dumps,", len(empty), "with an empty PR graph")
EOF

# (b) blindness + determinism of the kit (was broken -- see sec 1)
python3 vtx_rules/scankit.py selftest            # -> 25 dumps, 0 failures

# (c) the ranker re-validation gate
python3 dl_vtx_training/scan_ranker.py --arm work-mcp1k-harv3 \
    --tag vtxscan-harv3-mcp1k --tsv /home/xqian/tmp/pr88/rank-mcp1k-harv3.tsv
# apply-only to the new arm (no labels there yet)
python3 dl_vtx_training/scan_ranker.py --arm work-mcp2k-harv3 \
    --tag vtxscan-mcp2k --tsv /home/xqian/tmp/pr88/rank-mcp2k.tsv

# (d) one wave, end to end
python3 vtx_rules/selfscan.py prepare --kit new --workers 4 \
    --dumps /home/xqian/tmp/pr88/wave-0.txt --out /home/xqian/tmp/scan-mcp2k/wave0
#   ... N scanner subagents, each handed vtx_rules/scan_worker_prompt.md ...
python3 vtx_rules/b2_checkpoint.py --dir /home/xqian/tmp/scan-mcp2k/wave0
python3 vtx_rules/selfscan.py review --dir /home/xqian/tmp/scan-mcp2k/wave0
```

---

## 1. Tooling defects — four found before any event was scanned, two during

Each of these would have corrupted the round quietly rather than loudly. All
are in `vtx_rules/` / `dl_vtx_training/` analysis scripts.

### 1.1 The blindness selftest had been unrunnable since 2026-08-16

`scankit.selftest()` globbed `work-mcp1k-ma10`, which the pr/82 retirement
round archived **with the calib class dropped**. It therefore printed
`no dumps found` and returned 1 — so the check that proves `sanitize()` is
what ran, over real dumps, had been silently dead. Any "selftest green" claim
dated after that retirement is void; the last real one is 2026-08-15, quoted
in `runs/mix60-20260815/REPORT.md`.

Fixed by falling through live arms (`-harv3`) rather than naming one, so the
next retirement degrades to the next arm instead of disarming the check
again. Now: **25 dumps, 0 failures**.

### 1.2 The supersession guard was dead on the production path

`_load_picks` warns when a picks file changed after the last scoring pass —
the guard added in response to pr/80 §10.9, where `dev-new` was published at
43 and the worker's final answer was 42. It keyed on `scored.json` **alone**.
But `review` writes `review.json` and never `scored.json`, and `score` is
unavailable on a `--dumps` run. So on the one path a new-sample scan actually
uses, there was no revision warning at all. Now checks both artifacts.

### 1.3 `score` could not run on a partly-labelled sample

`score` does `L = labs[ev]` and raises `KeyError` on the first unlabelled
event. pr/80 §11 Step 4 describes exactly the workflow this breaks ("hand-scan
a subset yourself, then `score`"), and it had never been exercised. Phase 4's
calibration is that workflow. Added `--partial` (opt-in, and it **prints its
denominator** — a precision quoted over whichever events happened to be
labelled, read as the whole scan's accuracy, is the worst failure available
here) and `--tags` (a new sample's tag must stay out of `vtx_io.TAGS`, pr/82
§12.2).

Verified end-to-end on live data before use: 6 labelled + 2 unlabelled
`work-mcp1k-harv3` events; without `--partial` it refuses with a count, with
`--partial` it scores 6 of 8 and says so. The 6 synthetic picks (each the
label's own truth vertex id) score 6/6 at 0.00–0.01 cm, which independently
re-confirms the pr/80 F2 property that labels and dumps from the same arm
agree.

### 1.4 The ranker could not be applied to a new arm at all

`scan_ranker.py` is the pr/78 §2 active-learning ranker. Applying a
validated ranking to a fresh arm is its entire purpose, and a fresh arm has
no labels — but `vio.iter_labels` raises `FileNotFoundError` on a missing tag
directory. Added an apply-only path that prints what it is not measuring;
with no labels the enrichment table is skipped entirely rather than printed
as `nan`, and the ordering falls back to the rule-based `suspicion` score,
whose scales are fixed from the labelled-set distributions and need no fit.

### 1.5 Recorded, not fixed

`review` files a pick naming a vertex in **no** candidate group as
`REVIEW`/`agrees=False` with no error (`selfscan.py:432-438`). Rather than
change the bucketing mid-round, §3 counts them explicitly and the pilot bars
on zero.

### 1.6 Two viewer defects found *during* the owner's scan

Unlike 1.1–1.5 these surfaced mid-round, both reported by the owner as "the
page hangs" / "client connection was lost". They are recorded together
because they present identically in the browser and are told apart only from
the server side — and because between them they cost the owner two
interruptions in a 45-event instalment.

**(a) Bokeh session tokens expire in 300 s by default.** Every instalment
restarts the server with a new event list, which invalidates the token held
by any tab still open from the previous one. The reconnect is then *refused*:

```
bokeh.protocol.exceptions.ProtocolError: Token is expired.
```

The symptom is a **hang with no visible error**, so it reads as a broken
viewer rather than a stale tab. Fixed in `pr_display/serve_pr_display.sh`
with `--session-token-expiration ${SESSION_TOKEN_EXPIRATION:-86400}` — one
day, longer than any scanning session. Token lifetime is not a security
boundary here: the server binds localhost behind an ssh tunnel.

**(b) The ssh tunnel dies during the pauses a hand scan is made of.** Later
the same hour the owner reported the same symptom with a *different* cause,
and the server logs were clean of token errors. `last` against the bokeh log
identifies it in one step:

```
last:   xqian pts/8 ... 13:36 - 13:42  (00:05)      <- ssh session ends
bokeh:  13:38:55 WebSocket connection opened
        13:42:02 WebSocket connection closed: code=None, reason=None
```

Same second. A `code=None` close is an abrupt drop — the server never
received a close frame — and the tunnel goes with the ssh session. Bokeh's
JS does **not** auto-reconnect, so the banner persists after the tunnel is
back and only a reload clears it. Mitigation is on the client side and is now
in the serve script's header: `ssh -o ServerAliveInterval=30 -o
ServerAliveCountMax=6 -L 5017:...`. A hand scan sends no bytes while one
event is being studied, which is exactly the idle pattern a NAT or firewall
reaps.

**The diagnostic rule this round leaves behind**: when the viewer "hangs",
check the server before touching it — `curl -s -o /dev/null -w '%{http_code}'
http://localhost:5017/pr_display_viewer` plus a `bokeh.client.pull_session`
(1.4 s here) proves the app renders. If those pass, the fault is client-side
and **restarting the server is actively harmful**: it invalidates every other
open tab's token, which is failure mode (a). Both times, the fix was a
reload, and the owner's already-saved labels were untouched.

---

## 2. The pool: 845, not 2000

| | n |
|---|---:|
| `work-mcp2k-harv3/pr_evt*/` dirs, all `rc=0` | 2000 |
| …that emitted a calib dump | **880** |
| …minus dumps with an empty PR graph (`vertices==[]`, `dl_ran=false`) | −34 |
| …minus evt69314 | −1 |
| **scannable** | **845** |

The 1120 dumpless events are not failures: the chain ran clean and the main
cluster was cosmic-convicted before `PrDisplayDump` ever ran. Only events
where PR selects a main cluster emit a dump; 43.95% yield on data matches the
44.5% mcp1k prior. The user's instruction was to scan only the events with PR
results, so these are out of scope by construction.

**evt69314 is a genuine duplicate, not an id coincidence.** It carries
`run 18255 / subrun 1 / event 69314` in both `work-mcp2k-harv3` and
`work-nuecc48-harv3`, and is already labelled in `vtxscan-harv3-nuecc48` and
`vtxscan-prod0813`. The viewer names label files `labels-evt<ID>.json` with
no arm in the name, so scanning it again under a new tag would alias across
tags on any bare-id join. Excluded, and recorded here rather than silently
dropped.

That 845 is also confirmed from the other direction: no wave's `prepare`
emitted a `no-candidates.txt`, i.e. every one of the 845 still has ≥1
candidate **after** the 0.8 cm B2 merge.

---

## 3. The ranker gate: PASS at ×1.4, but degraded

pr/78 §2 validated the scan ranker out-of-time at AUC 0.778 with ×1.91
top-quartile corrective enrichment, and says it "should be used for any
future scan". It was fitted before pr/83/85/86. Pre-registered bar for reuse:
top-quartile enrichment ≥ 1.3×.

Re-validated on `work-mcp1k-harv3` + `vtxscan-harv3-mcp1k` (388 labels,
current epoch):

```
combined score   top-25% (n=97): 28.9% corrective  (base 20.6%, enrich x1.4)
fitted P (oof)   top-25% (n=97): 29.9% corrective  (base 20.6%, enrich x1.4)
5-fold CV AUC = 0.673
```

**PASS**, and the degradation from ×1.91/0.778 is itself informative: the
base corrective rate fell 34.3% → 20.6%, i.e. much of the ranker's old
headroom was reconstruction error that pr/83/85/86 removed.

The sharper signal is now `route`, not the fitted score:

```
route dl-rerank-accept   n=186 corrective 10.2%
route dl-rerank-reject   n=199 corrective 30.7%     <- x3.0 separation
```

Applied apply-only to all 880 mcp2k events for the §5 pile ordering.

---

## 4. The pilot — B2's first at-scale test

pr/80 §13.5 left one thing open: the B2 co-located-vertex merge
(`scankit.MERGE_R = 0.8 cm`) shrinks the candidate list a scanner is shown by
16–33%, and **no scanner had ever scanned from a merged picture** — §13.4's
47→47 replayed old picks through the new lookup, it did not re-scan. This
round is the first at-scale use, so 60 events answer it before 845 do.

60 events drawn at random from the 845 (seed 20260816, the draw and its event
list written to `pilot-draw.json` *before* any scanning), 4 workers × 15
events — the load every validated pr/80 arm used.

### 4.1 Mechanical checks — PASS

`vtx_rules/b2_checkpoint.py`:

```
events with picks: 60 (abstentions 5)
[1] off-list picks (BAR: zero) : 0
[2] picks resolving via an ALIAS: 0   via representative: 55
[3] candidate-list shrink      : 1255 -> 860 over 60 events (31.5%)
      events where the merge changed the list: 56 of 60
      shrink ON THOSE events: 32.0%   (doc pr/80 sec 13.4 measured 16-33%)
VERDICT: PASS -- no pick fell through the merge
```

Not one pick named a vertex outside a candidate group, and the shrink lands
at the top of the range §13.4 measured. The alias count being **0** is worth
stating precisely: it does not mean the merge is inert — it changed the list
on 56 of 60 events — it means every scanner named the group representative
that the panels draw, which is what the merge intends. Alias resolution
remains untested by this round because nothing exercised it.

### 4.2 Smoke numbers — a close replication of the held-out arm

| | pilot (mcp2k, n=60) | prior |
|---|---:|---|
| `certain` coverage | **37%** | 36.7% §10.5, 45.0% §10.4, 40% §13.2 |
| auto-accept | **35%** | 32% §13.2, 37% §10.6 |
| REVIEW FIRST | **1 / 60** | "rare, 1 in 60" §10.6 |
| scanner abstained | 5 / 60 (8%) | 0–3 / 60 |
| `certain` that agree with the reconstruction | **21 of 22** | 21 of 22 §10.6 |

Bucket census: REVIEW 33 (55%), auto-accept 21 (35%), abstained 5 (8%),
REVIEW FIRST 1 (2%).

These are smoke numbers, **reported and not thresholded**, deliberately:
pr/80 §13.2 is the reason. There the scanner claimed `certain` twice more
while getting the *same 21 correct* — coverage moved and accuracy did not
follow, so coverage alone cannot see a regression. With no truth on mcp2k at
pilot time, accuracy is unmeasurable here; §4.1's mechanical bar is the
actual gate, and the table above is corroboration.

The abstention rate (8%) is the one number above its prior range. On a data
sample with 34 events already dropped for having no PR graph at all, more
"only dots" events surviving into the pool is the expected direction.

---

## 5. Phase 3 — the owner's pile, served in instalments

The owner asked to start scanning before all 845 events were done, so the
pile is built in instalments rather than once at the end.

**Instalment 1 — 43 events, served on 5017 at 11:35 under tag
`vtxscan-mcp2k`**, built from waves 0–2 (286 events):

| tier | n |
|---|---:|
| 1 REVIEW FIRST (scanner `certain`, disagrees with reco) | 13 |
| 2 scanner abstained | 16 |
| 3 CALIBRATION (blind auto-accept draw, unmarked) | 14 |

Sized proportionally: 286/845 = 34% of the scan, so 34% of the 120-event cap
and 14 of the 40 calibration events. **This makes the calibration draw
stratified by wave rather than simple-random over the whole auto-accept
tier** — still a valid sample, and Phase 4 pools the strata, but it is a
different claim from "40 drawn at random" and is recorded as such in each
instalment's `calibration-draw.json`.

Not served, and staying unlabelled: 147 `REVIEW` + 96 auto-accept from these
waves.

### The review sheet is published after the scan, not before

`docs/pr/88-review-sheet.md` was planned as a companion giving one row per
served event with the scanner's pick, confidence and reco separation. Writing
it **before** the owner scans would defeat the calibration: the tier-3 events
are exactly the rows with `conf = certain` and `reco_sep = 0.0 cm`, so any
sheet carrying those two columns identifies the blind draw by inspection,
whatever the tier column says. The sheet is therefore written after the
owner's picks are in, where it is a record rather than a key.

This is a real residual limit, not a solved problem: the viewer draws the
reconstructed vertex as a star on every event, so the owner always sees the
reco answer. §4's measurement is "the owner confirms scanner ∧ reco", never an
independent number.

### A third class the buckets do not distinguish

evt157215 was filed `REVIEW FIRST (confident disagreement)` while its dump
carries **`main_vertex: null`** — the reconstruction found 2 vertices and
selected no main one. `review` computes `agrees` from the separation to
`main_vertex`, so a missing answer reads as a disagreement. The event belongs
in the review pile either way, so nothing is mis-served, but "confident
disagreement" overstates it: there is nothing to disagree with. Counted
separately from here on; the bucketing itself is left alone mid-round.

(The first pass of that diagnostic in `build_review_pile.py` called this an
"off-list pick" — a pick naming a vertex in no candidate group. It is not:
`b2_checkpoint.py` confirms zero off-list picks across all 286 events. The
two classes share a symptom, `reco_sep_cm = None`, and are now reported
apart.)

## 6. Phase 4 — the auto-accept gate, and what the owner's skipping taught us

### 6.1 The gate passes on the first stratum: 14 / 14

All 14 blind calibration events of instalment 1 came back labelled, and every
one agrees with the AI scanner's pick within 1 cm.

```
=== AUTO-ACCEPT CALIBRATION on mcp2k (pr/82 sec 9 gate 5) ===
calibration events labelled: 14 of 14 drawn
agree with owner @1cm: 14/14 = 100.0%   @3cm: 14/14 = 100.0%
BAR: >=90% -> auto-accept tier admitted.  VERDICT: PASS
```

**Not treated as settled.** 14/14 has a 95% Clopper-Pearson lower bound near
77%, i.e. this stratum alone cannot distinguish 95% from 80%. The remaining
26 calibration events are collected across later instalments and the strata
are pooled before the tier is admitted. And the standing limitation from §5
applies: the viewer draws the reco star, so this measures *"the owner confirms
scanner ∧ reco"*.

### 6.2 The owner skipped exactly where the scanner abstained

Of the 43 events served, 30 were labelled and 13 skipped:

| tier | labelled | skipped |
|---|---:|---:|
| 1 REVIEW FIRST | 12 | 1 |
| 2 scanner abstained | 4 | **12** |
| 3 CALIBRATION | 14 | 0 |

The owner's account: *"there are many events in there, there are only dots.
In this case, they are impossible to hand scan the true vertices, so I just
skipped them."*

That is owner rule 8 arriving as behaviour, and it is an **independent
validation of the blind scanner's abstention signal** — 12 of the 16 events
the AI declined from the pictures alone, the human also declined. pr/80 §6
listed the abstain branch as unvalidated because the owner had never once
used the "unclear" button; this is the first evidence for it, and it comes
from the owner's feet rather than their mouse.

### 6.3 The filter: `longest fitted segment < 5 cm`

The owner asked for these events to be filtered out of the training pool.
Their own behaviour defines the cut, and it is nearly a step function:

```
longest fitted segment, owner SKIPPED (n=13):
    0.3 0.3 0.3 0.3 0.4 0.5 0.6 0.8 0.9 1.0 1.3 2.2 4.4
longest fitted segment, owner LABELLED (n=30):
    2.4 3.5 7.4 19.4 24.7 30.1 34.5 45.0 ... 423.6
```

| cut | catches skipped | catches labelled |
|---|---:|---:|
| longest < 2.4 cm | 12 / 13 | 0 / 30 |
| **longest < 5.0 cm** | **13 / 13** | **2 / 30** |
| pr/80 §3 step 1 (cluster total < 8 **or** longest ≤ 3) | 13 / 13 | 3 / 30 |

`vtx_rules/scannability.py` ships the 5.0 cm operating point. It removes every
event the owner declined; the two it costs (evt102247 at 2.4 cm, evt177360 at
3.5 cm) are both events the blind scanner had *also* abstained on, so the
price is two labels nobody was confident about. pr/80's own criterion is
stricter and costs a third, evt318549 (cluster total 7.4 cm) — which the owner
labelled and the scanner called `certain`. That gap is the difference between
a rule for *"should the scanner abstain"* and one for *"should this event
carry a training label"*; this module implements the second.

Over the full sample the cut removes **76 of 845 (9.0%)**.

**Where it is and is not applied.** It filters the served pile and the
training pool. It is deliberately **not** applied inside `selfscan.prepare`:
an event with no readable vertex is still a legitimate thing to show a
scanner, and dropping events at prepare time would move every denominator in
this round's census.

Applied to instalment 2 it removed 23 events — 18 `REVIEW`, 4 abstentions,
and **1 auto-accept**. That last one is the case that justifies the filter
existing at all: an auto-accept is admitted to the training pool *without*
human review, so a dots event in that tier writes a confident label onto an
event with no readable vertex, and nothing downstream would have caught it.

### 6.4 Instalment 2 — 30 events

Built from waves 0–3 minus instalment 1's events (the builder now refuses to
re-serve: a re-served calibration event is no longer blind), with the §6.3
filter on: 6 REVIEW FIRST, 1 abstention, 5 CALIBRATION, 18 ranker-hot fill.
Served 11:59.

### 6.5 Instalment 2 returned: 30 of 30 labelled, none skipped

Against instalment 1's 13 skips. The §6.3 filter is doing exactly what the
owner asked for — no time spent on events nobody can answer.

### 6.6 The REVIEW FIRST tier is the round's strongest result

18 tier-1 events adjudicated across both instalments (scanner `certain` AND
disagreeing with the reconstruction):

| | n |
|---|---:|
| scanner right, **reconstruction WRONG** | **13** |
| reconstruction right, scanner wrong | 4 |
| both wrong | 1 |

The reconstruction is wrong on **14 of 18 (78%)** of this tier, against a
~21% base rate on this sample — **×3.7 enrichment**, above the ×2.86 pr/80
§13.2 measured on the mixed sixty. 13 genuine reconstruction errors surfaced
from 18 events of owner time.

This is the first time the AI scan has been used as an error *finder* at
scale rather than a labelling accelerator, and pr/80 §10.6's caution — "as an
error finder it contributes one event" — does not survive contact with a
tier built this way. The difference is that §10.6's certain-tier was
overwhelmingly the reco-agreement subset (21 of 22 agreed); here the tier is
*defined* by disagreement and the ranker orders within it.

### 6.7 Calibration after two strata: 18/19, and the honest interval

```
inst1  14/14
inst2   4/ 5
POOLED 18/19 = 94.7%     95% lower bound 74.0%
```

**Passes the pre-registered bar**, which was written on the measured rate
(§Phase 4: "≥ 90% → the tier enters the label pool"). The interval is
reported beside it rather than substituted for it — restating the bar as a
confidence bound after seeing the data would be moving the goalposts, and
restating it the other way would be hiding that n=19 cannot separate 95% from
80%.

Worth stating before the remaining strata are collected, so it is a
prediction and not an excuse: **with one miss already recorded, even the full
40 lands the lower bound near 83%, not 90%.** A lower bound above 90% would
need ~29 consecutive agreements from here. The decision the round will
actually face is therefore whether a 94–95% measured rate over 40 events is
enough to admit ~350 unreviewed labels — a question for the owner, not one
the arithmetic settles.

### 6.8 The §10.9 trap fired, on this round, and the Phase 0 fix caught it

Wave 5 was reviewed the moment its eighth picks file appeared on disk —
before every worker had reported completion. One worker (`w5-4`) then revised
`picks-4.json` **12 seconds later**, after going back to read six `p4-dqdx`
panels it had initially skipped:

```
picks-4.json  12:45:22      <- revised
review.json   12:45:10      <- written first
```

This is precisely pr/80 §10.9 — a picks file with a full set of substantive
entries is not necessarily a finished one — repeated by the orchestrating
session that had just written a doc section about it. The rule ("wait for the
completion notification, not the file") is easy to state and easy to skip
when eight files are sitting there looking done.

**The §1.2 fix caught it.** Re-running printed:

```
NOTE: picks-4.json changed after the last review.json was written
(2026-08-16 12:45:10). The previous numbers were computed on an earlier state
of the scan and are superseded by this run.
```

Before this round that notice was unreachable on the `--dumps` path — the
guard keyed on `scored.json`, which `review` never writes — so the stale
review would have propagated silently into instalment 5's tier census.

Damage: none. The re-run's bucket totals are identical (40 auto-accept, 73
review), and instalment 4 had been built from waves 0–4 only, so nothing
served to the owner was affected. The worker's revision was also the right
call — two of its earlier picks had already been overturned by reading `p4`,
which is the panel trap 2 exists for.

## 7. The gate closes: 39/40 = 97.5%

The full 40-event blind calibration draw came back, pooled across five
wave-stratified instalments:

```
pile1 14/14   pile2 4/5   pile3 6/6   pile4 6/6   pile5 9/9
GATE: 39/40 = 97.5%   95% CI [86.8%, 99.9%]   bar >=90%   VERDICT: PASS
```

Above the pre-registered bar and above pr/80 §10.5's 95.5% prior from the
previous reconstruction epoch — so the prior transferred, which §5 of doc
pr/82 explicitly refused to assume. The single miss is evt103391 at 5.60 cm.

§6.7 predicted the lower bound would land "near 83%"; it came in at **86.8%**,
i.e. that prediction was slightly pessimistic. It does not change the verdict,
which was always written on the measured rate.

**Consequence: the auto-accept tier is admitted to the training pool.**

### 7.1 The scanner's own calibration, measured on 174 owner labels

The owner labelled 174 events across the five instalments. Scoring the blind
AI scanner's picks against them:

| scanner tier | agrees with owner @1 cm | coverage |
|---|---:|---:|
| `certain` | **57 / 66 = 86.4%** | 38% |
| `likely` | 21 / 60 = 35.0% | 34% |
| `unclear` | 7 / 48 = 14.6% | 28% |
| **all** | **85 / 174 = 48.9%** | |

**The 48.9% must not be read as the scanner's accuracy.** This sample is
adversarially selected by construction: four of the five tiers exist
*because* the scanner was uncertain or disagreed with the reconstruction, and
the one tier where it was confident and agreeing appears only through the
40-event calibration draw. On that tier it scored 97.5%. A random-sample
number for this scan does not exist and was never measured.

What the table does establish is **calibration**, and it is steeper than
anything pr/80 measured — 86.4 / 35.0 / 14.6 against §10.4's 92.6 / 56.5 /
50.0. Two things follow:

- The `certain` tier holds up at 86.4% *even on a sample built to be hard*.
- `likely` and `unclear` would be 65% and 85% wrong as labels. The round's
  decision not to admit them — the direct cost of the ~120-event cap — is
  vindicated by measurement rather than by assumption. Those events keep
  their scanner picks in `review.json` and enter no training pool.

## 8. Phase 5 — pending

To be filled in: the tier census over all 845, the ~120-event pile served on
5017 under the fresh tag `vtxscan-mcp2k`, the blind 40-event auto-accept
calibration draw, and the pre-registered fork (≥90% admits the auto-accept
tier to the label pool; <90% does not).

### Protocol deviations, recorded as they happen

1. **The wrapper prompt gained one paragraph after the pilot.** Two pilot
   workers both defaulted to `/home/xqian/tmp/pr88/notes.md` and overwrote
   each other's lines; picks were unaffected (they are per-worker by
   construction) but notes were lost. Waves 1–7 are told to keep working
   files under a path unique to the worker. This touches bookkeeping only,
   not the scan procedure, but it does mean the pilot and the later waves did
   not receive byte-identical wrappers.
2. **The wrapper prompt is now a file.** pr/80 §11 Step 2 specifies it in
   prose and says "copy it verbatim", but the text was never saved anywhere,
   so every round re-derived it. It is now
   `vtx_rules/scan_worker_prompt.md`.

### Found by a scanner, recorded not fixed: P6 under-reads short prongs

A wave-2 worker reported that the evidence sheet's `N of M rise away` counter
— the direct implementation of owner rule 1, the strongest discriminator in
the whole procedure at 86.5% vs 31.9% — is **degenerate on segments shorter
than about 5 cm**. The two end-means are taken over 5 cm windows from each
end, so on a shorter segment both windows cover the same points, the means
come out identical by construction, and the prong is scored "flat" whatever
its actual gradient.

The consequence is not that the evidence is wrong, it is that a short prong
contributes *no* rule-1 vote while looking as though it contributed one. The
scanner caught it because several of its picks turned on gradients visible
only in the P4 profile, and on two of them the P6 summary line alone would
have pointed the other way.

Not fixed in this round, deliberately: changing the sheet mid-campaign would
mean waves scanned before and after saw different evidence, which is exactly
the confound pr/80 §10.1 built its scanner-noise floor to exclude. It is a
candidate for the next kit round — the fix is presumably a window that
shrinks with the segment, or an explicit "too short to measure" marker so the
counter's denominator stops counting prongs it cannot read. Either way it
should be base-rated first, per §13.3's rule that no panel change ships on
the strength of a few hand-picked events.

### Out of scope, and left open

- The **38 unlabelled `work-mcp1k-harv3` events** (pr/82 §1.4) — the user
  scoped this round to the 2000.
- **pr/82 §4.3 (C3): the 449 carried labels are still unvalidated.** They
  were bulk-written in one second at 08:44 on 2026-08-16; the pre-registered
  ≥95% blind re-scan of ~60 of them has not run and competes for the same
  scanner hours as this round.
- **evt54629's `TrackFitting::organize_ps_path` segfault** (pr/82 §12.7) —
  **fixed in a concurrent session** (toolkit `c3741088`, wcp `d5f8534`), not
  by this round. It does not touch anything here: this scan reads the calib
  dumps already on disk in `work-mcp2k-harv3`, produced at `771f075b`, and
  evt54629 produced no dump so it was never in the 845.
- Training round 3 itself (pr/82 §6).

# doc pr/86 — orphan segments at the neutrino vertex: the top 10 over the 1000-event sample

**Status: investigation only. No C++ or jsonnet is changed. §7's proposals are
proposals; none is implemented and none has been gated.**

The owner looked at the 2-D measurement overlays for `18255-268067`,
`18304-38856` and `18255-349945`, confirmed all three have a bad PR graph near
the neutrino vertex, and asked for the top 10 more events like them out of the
existing 1000-event sample, an analysis of what the PR issue is, and suggested
fixes.

Continues doc pr/85, whose §11.3 pre-registered exactly this round.

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# every number in §§2-6 (read-only, ~3 min over 511 calib dumps)
python3 pr86_orphan_census.py --top 10 --detail --json /home/xqian/tmp/pr86_rows.json

# the 2-D overlays of §3.  PANELS_DUMP_ARMS is REQUIRED: without it the panel
# script reads the -ma10 arm, which predates the pr/83 and pr/85 production
# flips (see §1.1).
PANELS_DUMP_ARMS="work-mcp1k-pr85ion2:work-nuecc48-pr85ion2:work-ncpi0-pr85ion2" \
  python3 pr85_panels2d.py /home/xqian/tmp/pr86-panels \
    30504:11096 67394:19054 38856:12106 463565:13006 423981:12099 \
    175896:17037 349945:18003 283595:12005 316025:16049 281214:60040

# the op3 decline evidence of §5 -- a TRACE rerun of the ten, fresh tags
export PR_EXTRA_STAGES=pr_display SBND_WCT_LOGLEVEL=trace PR_JOBS=5
./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr86-trace-mcp1k   data \
    67394 175896 349945 283595 316025 281214
./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr86-trace-nuecc48 data \
    30504 38856 423981
./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr86-trace-ncpi0   data 463565

# -a is REQUIRED: WCT logs carry invalid UTF-8, so plain grep prints nothing
grep -a -h -o "mvga: op3[^|]*" work-pr86-trace-*/pr_evt*/wct_pr_*.log
```

Arms: `work-mcp1k-pr85ion2` + `work-nuecc48-pr85ion2` + `work-ncpi0-pr85ion2`
(the doc pr/85 §10.7 flip-config arms, proven byte-identical to a bare
production run by that section's post-flip smoke) and the three
`work-pr86-trace-*` tags produced here. Installed
`local/lib/libWireCellClus.so` mtime 2026-08-15 16:44, toolkit `412b4e93`.

---

## 1. Symptom

The owner's words, from doc pr/85: *"for a neutrino vertex, in general the
connection should be simple — one vertex with multiple tracks (or shower stem)
coming out of it."*

The measured failure is the exact negation, and it has a name in this doc:

> An **orphan** is a segment whose charge comes within 3 cm of the vertex but
> which is **not incident on it** in the PR graph — at any length.

`evt67394` is the cleanest statement of it. Four real prongs — 32.8, 27.6, 17.1
and 6.6 cm — meet at one point in the U, V and W images. In the PR graph the
vertex the owner clicked has **degree 1**, and its single arm is a 3.2 cm stub.
All four prongs are orphans, every one of them 0.20 cm from the vertex.

### 1.1 A correction that reframes the earlier three

The overlays that started this round were rendered from `work-*-ma10`, which
**predates** the doc pr/83 (`break_seg_orient`) and doc pr/85
(`main_vertex_graph_audit` and friends) production flips. Re-scored on the
shipped configuration:

| event | at the click, pre → post flip | at the reco `main_vertex`, post |
|---|---|---|
| 268067 | 3 orphans → **5**, and no PR vertex within 1.52 cm of the click | **0**, degree 4 |
| 38856 | 0 → 0 | **4**, degree 2 |
| 349945 | 2 → **1** | **3**, degree 2 |

Two consequences, both load-bearing:

1. **evt268067 is no longer in the population.** pr/85's flip cleaned its reco
   vertex (degree 4, zero orphans) and moved it 1.52 cm off the click, which is
   why the click anchor now reports *more* orphans — it is anchoring on a
   neighbour. It is excluded from the top 10 and its post-flip state is
   §8.3's own item.
2. **Each of the three is defective at a different anchor.** That is not a
   detail; it decides the whole design of the search (§2.1).

---

## 2. The measurement

### 2.1 Two anchors, ranked by the worse

`pr86_orphan_census.py` scores every event at **both**:

- **`A_reco`** — the dump vertex nearest the dump's own `main_vertex`. Needs no
  hand label, so it covers the whole sample, and it is where the reconstruction
  believes the neutrino is, which is what every downstream tagger and BDT sees.
- **`A_click`** — the dump vertex nearest the owner's rank-1 pick, where a
  label exists. This is what the owner is looking at when they call the PR
  "not ideal".

Neither alone is the population: **19 events have orphans only at the reco
anchor, 2 only at the click, 48 at both.** Scoring at the click alone — what
doc pr/85 did — would have missed 19 of 69.

Events whose click has no PR vertex within 1 cm are recorded as their own class
(17 of them), never silently dropped; post-flip evt268067 is one, and dropping
it would have hidden a real change.

### 2.2 One constant deliberately removed

pr/85 counted a near-vertex segment as clutter only below `STUB = 3` cm and as a
prong only above `LONG = 10` cm, so a non-incident 8 cm segment was counted as
neither — doc pr/85 §11.1 measured 20 of 462 events lost to that band. This
census has **no length band at all**: an orphan is an orphan at any length. The
length is carried as a column and binned where it matters (§4).

### 2.3 The population — and why it is 511, not 1000

| | |
|---|---|
| calib dumps read | **511** (mcp1k 445, nueCC48 47, NCpi0 19) |
| with a reco anchor | 493 (18 where `main_vertex` is >1 cm from any dump vertex) |
| with a hand label | 473 (456 click-anchored, 17 click-unmatched) |

The other 555 of the 1000 mcp1k events **cannot** be scored, and the reason is
worth recording because it cost this round an hour of batch time. The plan
assumed they lacked a calib dump because they had been run without
`PR_EXTRA_STAGES=pr_display`; a 561-event rerun with the flag set
(`work-pr86-dumps`, stopped at 341 events) produced **zero** dumps. The flag had
been on all along: `PrDisplayDump` is configured in all 1000 reference-arm logs,
and **exactly the same 555** carry

```
W <PrDisplayDump:pr> no TrackFitting in grouping 'live' -- is this stage after tagger_check_neutrino?
```

There is no fitted neutrino cluster in those events to dump. **The searchable
population of the 1000-event sample is 445 events, and all of them are already
scored** — the sample is complete, not partial. `work-pr86-dumps` is a dead arm;
it is left in place rather than deleted (M13 habit) and should be ignored.

### 2.4 How common

Orphan count at the worse anchor, over the **494** events that have at least
one usable anchor (17 of the 511 dumps have neither):

| orphans | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| events | 425 | 37 | 22 | 5 | 3 | 2 |

**69 of 494 (14 %)** carry at least one orphan; **32 (6.5 %)** carry two or
more. 178 orphans in total.

### 2.5 The result worth reading first

Crowding at the reconstructed vertex is a near-perfect flag for the vertex being
in the wrong place. Over the 472 events that have both a reco anchor and a hand
label, binned on orphan count at that anchor:

| orphans at `A_reco` | events | median `b1` (click ↔ reco) | fraction `b1` > 1 cm |
|---|---|---|---|
| 0 | 406 | 0.00 cm | 0.30 |
| 1 | 36 | 0.00 cm | 0.31 |
| 2 | 22 | 0.28 cm | 0.36 |
| **3 or more** | **8** | **16.32 cm** | **1.00** |

**Every one of the 8 events with three or more orphans at the reconstructed
vertex has a mis-picked vertex**, against a 30 % base rate. n = 8 is small — the
point is not the precision of the number but that the class is not a cosmetic
one: this is the same charge that the vertex selector had to choose among.

(For contrast, anchor degree is a much weaker flag: degree-1 anchors are 39 %
`b1` > 5 cm, degree-4 anchors 12 %.)

---

## 3. The top 10

Excluding events owned by another round — 283040/59899/72586 (pr/83),
283713/287517/284794/65289 (pr/84), 59685/280972/360535/278785/437699/314838
(pr/85) — ranked by orphan count at the worse anchor, then by segments touching:

| # | event | root | `b1` | `A_reco` deg/touch/orph | `A_click` deg/touch/orph | orphans (cm, class) |
|---|---|---|---|---|---|---|
| 1 | **30504** | nueCC48 | 100.8 | 11096 · 2/7/**5** | 11001 · 1/1/0 | 8.8 7.1 5.0 3.0 1.7 — all BENIGN |
| 2 | **67394** | mcp1k | 0.4 | 19053 · 5/6/1 | 19054 · 1/6/**5** | 32.8 27.6 17.1 6.6 3.0 — all BENIGN |
| 3 | **38856** | nueCC48 | 30.1 | 12106 · 2/6/**4** | 12004 · 2/2/0 | 15.6 12.7 VIA · 4.5 2.6 CUT |
| 4 | **463565** | NCpi0 | 30.2 | 13006 · 1/5/**4** | 108102 · 2/2/0 | 10.5 8.0 6.7 3.3 — all BENIGN |
| 5 | **423981** | nueCC48 | 1.5 | 12099 · 2/5/**3** | 12003 · 4/5/1 | 85.0 31.2 29.9 — all VIA |
| 6 | **175896** | mcp1k | 43.7 | 17037 · 2/5/**3** | 66034 · 2/2/0 | 20.8 VIA · 5.0 CUT · 3.0 VIA |
| 7 | **349945** | mcp1k | 1.1 | 18003 · 2/5/**3** | 18011 · 3/4/1 | 11.4 VIA · 6.3 BENIGN · 4.3 VIA |
| 8 | **283595** | mcp1k | 0.6 | 12045 · 3/5/2 | 12005 · 2/5/**3** | 3.5 BENIGN · 2.4 1.8 VIA |
| 9 | **316025** | mcp1k | 2.5 | 16049 · 1/4/**3** | 16003 · 3/4/1 | 81.7 9.4 2.6 — all CUT |
| 10 | **281214** | mcp1k | 0.0 | 60040 · 4/6/2 | 60040 · 4/6/2 | 4.8 BENIGN · 1.6 CUT |

Panels are in `/home/xqian/tmp/pr86-panels/` (regenerate with §0). Five of the
ten have `b1` > 30 cm — the §2.5 correlation, seen one event at a time.

`evt281214` is the one event of the ten where the click and the reco vertex are
the *same* vertex (`b1` = 0.00) and it still carries two orphans: the defect is
not only a symptom of a mis-picked vertex.

---

## 4. What an orphan is, structurally

Walking the PR graph from each orphan's own near-end vertex to the anchor:

| | orphans | |
|---|---|---|
| **VIA** — a path exists, every hop under `mvga_stub` = 2.5 cm | 68 | connected, through clutter (pr/85's interposed stub) |
| **BENIGN** — a path exists through at least one longer segment | 55 | ordinary grand-daughter topology — *but see below* |
| **CUT** — no path at all | 55 | 37 of them (67 %) in a **different cluster** |

The dominant shape is a **chain of one to three short segments interposed
between the anchor and the real prongs**. The connecting hops in the top 10 are
1.13, 1.48, 1.94, 2.30, 2.86, 3.04, 3.24 and 4.43 cm. `evt67394`'s four prongs
hang off a vertex two hops away, through 3.04 and 3.24 cm segments;
`evt30504`'s five hang off one three and four cm away.

That makes the VIA/BENIGN label a threshold artefact rather than a fact. A
"BENIGN" path whose longest hop is 3.0 cm is visually the same clutter as a
"VIA" path whose longest hop is 2.4 cm; the census carries `path_max` alongside
the label for exactly this reason. **The honest split is on the length of the
hop that touches the anchor**, because that is the number the code actually
tests (§5):

| anchor-incident hop | orphans |
|---|---|
| under 2.5 cm (`mvga_stub`) | 71 |
| 2.5 – 5.0 cm | **50** |
| 5.0 cm or more | 2 |
| no path (CUT) | 55 |

The distribution **stops at 5 cm**. That is the single most useful number in
this doc: whatever these connecting segments are, they are not a tail — they are
a bounded population, and there is essentially nothing between 5 cm and a real
prong for a wider rule to over-reach into.

---

## 5. Why nothing fixes it — the evidence, not the argument

### 5.1 mvga op3 only ever looks at segments already attached

`NeutrinoGraphAudit.cxx:383-395` builds `incident` from the anchor's own
out-edges and skips any with `len >= m_mvga_stub`. An orphan is by definition
*not* an anchor edge, so op3 can only ever reach one **through** an incident
stub under the ceiling. That is not a tuning statement: it is what the loop
iterates.

Applying it to all 178 orphans: **71 are reachable in principle, 107 are not** —
50 because the anchor-incident hop is over the ceiling, 55 because there is no
path, 2 because the hop is over 5 cm.

### 5.2 What op3 actually did, from a TRACE rerun of the ten

10/10 `rc=0`; the trace arms reproduce the reference arm's `(deg, touch,
orphan)` triple **exactly on all ten**, so these lines explain the numbers in §3.
mvga fired in every one of the ten. The op3 lines:

| event | op3 line | why it declined |
|---|---|---|
| 38856 | `eval-interposed len=1.94cm vf_deg=3 far_angle=139.8deg` | **collinearity, by 10.2°** — `mvga_interposed_angle` is 150° |
| 423981 | `eval-interposed len=1.48cm vf_deg=5 far_angle=130.1deg` | collinearity, by 19.9° |
| 30504 | `eval anchor=sat d=1.26cm len=1.65cm nfit=4 overlap=0.75` | **`mvga_dup_frac = 0.8` rejects 0.75** |
| 67394 | `stub-interposed len=0.39cm vf_deg=5 carried=4 far_angle=163.2deg` | **fired** — and 5 orphans remain |
| 463565, 283595, 316025 | `stub-absorb ... gate=overlap` | fired; orphans remain |
| **175896, 349945, 281214** | *no op3 line at all* | see §5.3 |

Three findings, in order of how actionable they are:

1. **The gate that most often stops op3 on this population is collinearity, not
   the ceiling.** Both interposed evaluations in the ten were declined at 139.8°
   and 130.1° against a 150° requirement. A stub that turns a corner between two
   prongs is exactly the geometry `mvga_interposed_angle` was written to
   exclude — and it is also exactly what a real neutrino vertex looks like.
2. **The `mvga_dup_frac = 0.8` guard, added in pr/85 §10.6 to stop four adverse
   movers, is what declines evt30504.** That is a genuine trade, correctly made
   at the time on the evidence then available; this doc is the first evidence
   from the other side of it.
3. **op3 firing does not clear the event.** evt67394 carried four prongs onto
   its anchor at 163.2° and still shows five orphans at the click anchor,
   because the click anchor is a *different* vertex from the one op3 repaired.

### 5.3 A gap in the instrumentation

Three of the ten produce no op3 line whatsoever. op3's first three gates —
`len >= m_mvga_stub` (`:395`), far vertex invalid (`:397`), far vertex
`kProtectedBreak` (`:407`) — all `continue` **before** any logging, so a decline
there is completely invisible.

`prongs.empty()` (`:421`) is excluded by the census: in all three the far vertex
demonstrably carries the orphan prongs. `segment_track_length` (default
`flag = 0`) computes geometric length from the same fits the dump reports, so
the ceiling should not be silently disagreeing with the dump's 1.13 / 1.48 cm.
**That leaves `kProtectedBreak` on the far vertex as the leading candidate** —
the flag `snap_main_vertex_to_kink` (pr/50) and the two-end break (pr/48) set
precisely to stop later passes undoing a deliberate break.

This is stated as the leading hypothesis, **not as established**: the three
silent gates are indistinguishable from outside, which is itself the point (§7
Q1).

### 5.4 Why pr/85's sweep could not have found this

doc pr/85 §10.5 swept `mvga_stub` over 2.0 / 2.3 / 2.5 and concluded
"**2.5 adopted, 3.0 not pursued**". That conclusion was sound on its evidence
and is not contradicted here — but the sweep could not have seen the 2.5–5.0 cm
population, because **pr/85's 57-event set was selected by a census whose
`STUB` constant was 3.0 cm.** A population defined by "stubs shorter than 3 cm"
contains almost nothing above 3 cm for a ceiling sweep to test against. The
50 orphans behind a 2.5–5.0 cm hop are new evidence, not a re-litigation.

### 5.5 The other passes

- **`examine_vertices_4`** — doc pr/85 §3.3 established it is structurally
  barred at the main vertex (`v1 != main_vertex` and `degree >= 2` leave a
  terminal stub with no eligible dying vertex). Nothing in this population
  changes that; it remains unable to act at either anchor.
- **`examine_structure_3`** — pr/85 §3.1 exonerated it with its own
  `WCT_ES3_MERGE_CENSUS`: the clutter does not exist yet when ES3 looks. The
  §5.2 trace is consistent — the orphan structure is present at mvga time,
  which is after every `examine_*` pass.
- **`merge_nearby_vertices`** — 0.1 cm co-location in **wcpt** space. Every
  distance in this doc is **fit** space. Per doc pr/84 §8.2 no radius may be
  proposed from these numbers, and none is.

---

## 6. Answering the question directly

*What is the PR issue in these events?*

The vertex is under-connected while the charge around it is over-segmented.
Between the vertex and its real prongs sits a short chain — one to three
segments of 1 to 4.4 cm — so the prongs attach to a neighbour instead of to the
vertex. In a third of cases (55 CUT orphans, 37 of them cross-cluster) there is
no graph connection at all, and that is an upstream clustering decision, not a
PR-graph one.

The single pass positioned to clean this up, mvga op3, **cannot see 107 of the
178 orphans at all** because it only inspects segments already incident on the
anchor. Of the ones it can see, the gate that most often stops it is the 150°
collinearity requirement — which rejects the corner geometry a real neutrino
vertex has.

---

## 7. Proposals

All default OFF. **None is implemented and none is gated.** Each names the gate
it would need. Ordered by evidence strength, not by ease.

### Q1 — `mvga_decline_log` (instrumentation only, no behaviour change)

Emit a TRACE line at op3's three silent gates (`:395`, `:397`, `:407`) naming
which one fired. Three of ten events in this doc's own top list are
unexplained purely because these gates are mute; §5.3's conclusion is a
hypothesis where it could be a fact.

*Gate:* byte-identity is trivial (log-only at TRACE), but it must still be
proven with `scripts/pr85_hash_gate.py` on one arm.
*Supersedes nothing.* This is the cheapest item here and it is what makes the
rest measurable.

### Q2 — `mvga_interposed_angle` as a fitted number rather than an assumption

150° was chosen in pr/85 §10.1 without a fit. Both interposed declines in this
doc's ten are at 139.8° and 130.1°. A sweep over 150 / 140 / 130 against the
§3 population, with the pr/85 §10.4 mover adjudication rerun, would tell us
whether the requirement is protecting anything.

*Risk, stated plainly:* lowering it admits genuine corners — a stub that really
does turn between two different particles. The §10.4 adjudication is the
instrument for that, not an argument.
*Gate:* full-sample knob-on arm, mover adjudication against the hand labels,
nueCC48 nue-score ledger.
*Sharpens* pr/85 §7 Q1, which shipped the mechanism without fitting its angle.

### Q3 — raise the anchor-incident ceiling, bounded by the measured 5 cm edge

50 orphans sit behind an anchor-incident hop of 2.5–5.0 cm and **only 2 above
5 cm**. That bounded distribution, not a preference, is the argument for testing
a ceiling near 5 cm.

*This is not a re-proposal of what pr/85 §10.5 declined.* That sweep stopped at
2.5 on a population its own 3.0 cm `STUB` constant had pre-filtered (§5.4).
*Risk:* absorbing a 4 cm segment is a much bigger edit than absorbing a 2 cm
one, and pr/85 §10.6 showed marginal absorbs are exactly where adverse movers
live.
*Gate:* the pr/85 §10.5 sweep protocol re-run on **this** population —
`mvga_stub` at 2.5 / 3.5 / 5.0, mover adjudication, per-value stub counts.

### Q4 — revisit `mvga_dup_frac = 0.8` with a second discriminator

0.8 was set in pr/85 §10.6 to reject the `(nfit=4, overlap=0.75)` adverse-mover
class. It also declines evt30504. Rather than moving the threshold back — which
would reinstate four known adverse movers — find the discriminator that
separates the two, most plausibly the *number of prongs carried* (an absorb that
reconnects several prongs is not the same edit as one that deletes a duplicate).

*Gate:* the four pr/85 §10.6 adverse movers must stay at 0.00, and evt30504
must improve, in the same arm.

### Q5 — nothing is proposed for CUT

55 orphans, 37 of them in a different cluster from the anchor. No PR-graph edit
can attach a segment that clustering never put in the same object. This is
recorded as out of scope, exactly as doc pr/85 §7 Q5 recorded it, and it is the
larger of the two problems by count.

### Explicitly not re-proposed

- `es3sg_vertex_fit` — pr/72 R3 measured it **blunter** (27.3° → 15.2°), fully
  reverted.
- `fit_exclusion` — pr/30 P1, lost 7 nue candidates, blocked on a units question.
- `graph_endpoint_strict` — pr/30 P8, closed, "must stay OFF".
- Any `merge_nearby_vertices` radius derived from the distances in this doc —
  they are fit-space, that pass is wcpt-space (pr/84 §8.2).
- `vks_carry_prong` as it stands — pr/85 §10.4 adjudicated it NEGATIVE and it
  ships OFF. pr/85 §10.9 notes the one adverse CARRY had the weakest turn of the
  ten firings (33.6°); a turn-strength floor would be a *different* proposal and
  is not made here.

---

## 8. Surfaced, not picked

### 8.1 The 555 events with no fitted neutrino cluster

Not a defect of this round, but the sample is half what it looks like: 555 of
1000 mcp1k events reach `PrDisplayDump` with no `TrackFitting` in the live
grouping. Whether that is correct (no neutrino candidate) or a loss is not
answered here and is not visible from the calib dumps, since there are none.

### 8.2 The 18 events where `main_vertex` is more than 1 cm from any dump vertex

Reported as their own class rather than dropped. Unexplained.

### 8.3 evt268067 after the pr/85 flip

Its reco vertex is now clean (degree 4, zero orphans) and 1.52 cm from the
owner's click, so the click anchor lands on a neighbour and reports five
orphans. pr/85 §10.6 recorded this event as a **nue recovery** (−15.0 → +1.28,
cosmict cleared). Whether the 1.52 cm move is a small regression or the vertex
being placed correctly is an owner hand-scan question, not a census one.

### 8.4 pr/85 §8.1's degree-1 class is still open

165 of the 472 labelled reco anchors have degree 1, and they are 39 % `b1` > 5 cm
against 12 % for degree 4. Larger than this doc's 69 events and still its own round.

---

## 9. Verification

- Every number above comes from one run of `pr86_orphan_census.py` (§0). Its
  first output is an **exemplar gate** that reproduces the three events the
  owner named; it prints PASS before any other number is emitted.
- The trace arms reproduce the reference arm's `(deg, touch, orphan)` triple on
  **10 of 10** events, so §5.2's log lines explain §3's table.
- 10/10 `rc=0` in the trace arms (`rc.txt` per event).
- Nothing was written into any existing `work-*`, `vertex_labels/`,
  `abtest/snap/`, `sweep/` or `decisions*/`; four fresh tags were created
  (`work-pr86-dumps`, dead — see §2.3 — and the three `work-pr86-trace-*`).
- No C++ or jsonnet was changed. `git diff` for this round touches only
  `sbnd_xin/pr86_orphan_census.py`, `sbnd_xin/pr85_panels2d.py` (a
  `PANELS_DUMP_ARMS` override so panels can be drawn from a chosen arm), this
  doc, and a pointer line in doc pr/85 §11.3.

# doc pr/130 — the guard-freed pool has no predicate, and the shipped
# continuation test would undo the rescue the pool was built for

**Status: MEASURED, OPEN — one owner decision blocks it.  No knob written, no
behavior changed.  The only code added is a byte-neutral, env-gated probe.**

## Repro

```bash
# toolkit @ <this commit>, SBND production config (pr/128 flips ON)
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_EXTRA_STAGES=pr_display PR_JOBS=3
# the tape
WCT_KINE_GUARDFREED_PROBE=1 ./run_pr_chain_batch.sh \
    work-mcp2k-grp0825 work-pr130-probe2-mcp2k data 393505 171572 94392
grep -h PR130_GUARDFREED work-pr130-probe2-mcp2k/pr_evt*/*.log
# the byte-neutrality control (probe env absent)
./run_pr_chain_batch.sh work-mcp2k-grp0825 work-pr130-neutral-mcp2k data 393505 171572 94392
```

Tape → `docs/pr/pr130-guardfreed-geometry.tsv`.

## Symptom

`kine_count_guard_freed` (doc pr/123 round 2, **SBND PRODUCTION ON**) counts
energy into `kine_reco_Enu` on a single predicate — the `kPass4GuardFreed`
flag.  `NeutrinoKinematics.cxx`:

```cpp
if (!seg->flags_any(SegmentFlags::kPass4GuardFreed)) continue;
```

The block's own comment says it outright: *"The flag is the predicate."*  There
is no cluster, length, distance or continuation term.

doc pr/129 found the consequence: on SBND 18255-**393505** the pool counts
**268.70 MeV** of an object the owner ruled over-clustering — *"all three
should be reject, overclustering, so not counting in the enr eneryg"* — into
the neutrino candidate's energy.  That is the "do not count far-away activity"
half of the owner's pr/128 metric being broken by a shipped knob.

## Blast radius (verified, not recalled)

`grep "kine_count_guard_freed: COUNT"` over all 239 events of the pr/128
production arms (`work-pr128r1-on{98,141}-*`): **exactly 3 objects in 3 events,
710.66 MeV**.  The three-event run below is therefore the complete population,
not a sample.

## The measurement

Byte-neutral probe `WCT_KINE_GUARDFREED_PROBE` tapes, for every candidate the
pool is about to count, the four terms of the continuation test pr/128 already
ships (`segment_continuation_geometry`), measured against the same reference
cloud pr/128's pools use — what the tree actually counts, frozen at pool entry
so same-pool candidates cannot qualify each other.

| event | KE (MeV) | len (cm) | d_mainvtx | gap | cand_end | ref_end | **kink** | ref len / KE | ref is shower start | owner verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 94392  | 137.21 |  46.8 | 44.49 | 0.00 | 0.00 | 0.00 | **13.32°**  | 29.8 cm / 98.44 MeV | yes | unruled |
| 171572 | 304.75 | 125.1 | 20.36 | 0.00 | 0.00 | 0.00 | **38.43°**  |  3.0 cm /  5.93 MeV | no  | unruled |
| 393505 | 268.70 | 108.5 | 74.37 | 0.00 | 0.00 | 0.00 | **137.34°** |  4.2 cm /  8.11 MeV | no  | **REJECT** |

### Finding 1 — three of the four terms are vacuous here

`gap`, `cand_end_dis` and `ref_end_dis` are **0.00 cm for all three**.  Every
guard-freed candidate touches counted content end-to-end, and in all three
cases the reference segment is **in the candidate's own cluster**
(`ref_cluster == cluster`: 15/15, 10/10, 45/45).  A cluster always contains
something already counted, so the distance terms cannot come out any other way.
This is doc pr/128's own lesson — *"a gap test can be vacuous — check what your
own metric measures"* — recurring one level down.  **The kink is the only live
term.**

### Finding 2 — the kink separates, but not the way the question was posed

The question going in was "does the continuation test reject 393505 and keep
171572/94392?"  It does not.  At the shipped `kine_near_kink_deg = 30°`:

- 393505 **137.34° → REJECT** ✓ matches the owner's ruling
- 171572 **38.43° → REJECT**
- 94392  **13.32° → KEEP**

So the shipped test, reused unchanged, removes **573.45 of the 710.66 MeV**
this pool contributes and keeps only 94392's 137.21 MeV.

### Finding 3 — the sharp part: this would undo the pool's own reason to exist

The pr/123 block comment names its target explicitly: *"SBND 18255-**171572**'s 125cm
muon, ~390 MeV silently absent from `kine_reco_Enu`."*

**Same object, verified — this is not a comment-drift artifact.**  pr/123's own
exposure census (doc pr/123, "Exposure census") reads: *"exactly **2**
guard-freed-and-lost segments — 171572 **seg 10008 (125.1 cm mu)** and 393505
seg 15013 (108.5 cm mu)"*, and its commit `899e4ff6` records the validation as
*"muons return as root nodes **304.8**/268.7 MeV"*.  One segment, 125.1 cm,
304.8 MeV — identical to what this round's probe tapes (304.75 MeV, 125.1 cm).
The `~390 MeV` in the C++ comment is loose prose in that one line, contradicted
by the same commit's own numbers; the run id (18255 vs the manifest's 18259) is
likewise stale there.  Neither is fixed in this round.

Cluster 10's other EM material is accounted for too: pr/123 explicitly parked it
— *"Residual noted: 171572 seg 10005 (10 cm, pdg-11, flag_shower, score-100)
also left the fake shower and stays unowned ... not addressed here."*  So the
object pr/123 rescued and the object the continuation test rejects are the same
single segment.  The pool was
built to recover 171572.  The pr/128 continuation test says 171572 does not
qualify.  Two shipped rounds disagree about the same object, and the tape is
what surfaced it.

### Finding 4 — a second discriminator, and it agrees

The reference each candidate continues *from* separates the three more cleanly
than the kink, and it is not a distance (doc pr/129 ruled a geometric bound
out):

- **94392** continues a **29.8 cm / 98.44 MeV pdg-13 shower-start muon** at
  13.3°.  Cluster 45 holds exactly two segments, 29.8 + 46.8 cm — one physical
  76 cm muon broken in the middle.  A real continuation.
- **171572** continues a **3.0 cm / 5.93 MeV** electron crumb at 38.4°.
- **393505** continues a **4.2 cm / 8.11 MeV** electron crumb at 137.3°.
  Cluster 15 holds 4.2 + 63.4 + 131.4 + 6.1 + 108.5 cm — the ~295 cm
  non-main-cluster object pr/129 identified as a cosmic.

A "continuation of substance" rule (the reference must be a real object, not a
sub-10-MeV crumb) rejects 393505 and 171572 and keeps 94392 — **the same split
the 30° kink gives**.  Two independent discriminators agreeing on the same
partition is the strongest evidence in this round.

## The owner decision (this is what blocks a fix)

Both candidate rules keep only 94392 and drop 171572's 304.75 MeV.  171572 is
unruled and is the pool's founding target, so the choice is not mine:

1. **Accept the split** — gate the guard-freed pool on the shipped continuation
   test.  Removes 573.45 MeV across 3 events; 393505's owner-ruled cosmic goes,
   and so does 171572, reversing part of pr/123 round 2.
2. **Keep 171572** — removes only 393505's 268.70 MeV, the one object with an
   owner ruling.  Note this option is *not* symmetric with option 1.  Widening
   the kink to some value in (38.43°, 137.34°) would be a threshold justified by
   nothing but a gap between two data points, one of which is the object being
   spared — CLAUDE.md §5.7.  If 171572 reads as real, the honest conclusion is
   that **the kink is the wrong discriminator for this pool and the
   continuation-of-substance rule (Finding 4) replaces it**: require the
   reference to be a real object rather than a sub-10-MeV crumb, which rejects
   393505 on the reference it continues from, not on an angle.  That rule still
   drops 171572 as written, so a 171572-keeping form of it needs the owner's
   reasoning for keeping it, not a number picked to fit.
3. **Rule on 171572 first** — one Bee read: is a 125.1 cm muon hanging off a
   3.0 cm / 5.9 MeV crumb at 38° in a non-main cluster 20 cm from the ν vertex
   the candidate's energy, or over-clustering?  That single verdict picks
   between 1 and 2 and needs no further measurement.

Recommendation: **option 3, then implement.**  Option 2 is the conservative
default if 171572 reads as real — it touches only the object already
adjudicated, and CLAUDE.md §5.7 says do not tune a threshold to make an unruled
number look right.

Whichever is chosen, the fix is **kine-side only** — doc pr/129 consequence 2
established that PF display and Enu accounting are separate concerns and the
codebase already splits them (`pf_orphan_*` vs `kine_count_*`).  No display
path is touched.

## Verification of the probe itself

Content hashes via `abtest/hash_archive.py` on `mabc-pr.zip` +
`pctree-pr-evt<N>.tar.gz`, all 3 events:

- new binary, probe env **absent**, vs the pre-probe production arm
  `work-pr128r1-on141-mcp2k` — **6/6 SAME**
- probe **ON** vs probe **OFF**, same binary — **6/6 SAME**
- extended probe (ref identity) ON vs `work-pr128r1-on141-mcp2k` — **6/6 SAME**

`./build/clus/wcdoctest-clus`: 235/235 test cases, 2524/2524 assertions, rc=0.
Freshness proof (M1): `local/lib/libWireCellClus.so` newer than the edited
source, and `WCT_KINE_GUARDFREED_PROBE` / `PR130_GUARDFREED` present in the
installed library.

**Byte-identical: YES.**  The probe block is a separate block placed before the
production one, which stays byte-for-byte untouched (CLAUDE.md §2, M10).

## Sentinel note

Nothing in the sentinel registry covers this pool's *bound*.  The two existing
entries (171572, 393505, both "pr/123 r2") assert the pool still **fires** —
they would pass while it counts a cosmic.  Whichever option is chosen needs a
**negative** sentinel in the pr/128 style (`log_absent` on the 393505 count
line), or the fix is exposed to the pr/127 failure mode: a shipped fix dying
silently.

---

# Part 2 — sentinel coverage (doc pr/130 item 5).  DONE.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_sentinel_arm.sh 130          # 12 events, ~15 min
./scripts/pr127_sentinels.py --arms work-sent130-mcp1k work-sent130-mcp2k \
    work-pr128r1-on98-mcp1k work-pr128r1-on98-mcp2k work-pr128r1-on98-ncpi0 \
    work-pr128r1-on98-nuecc48 work-pr128r1-on141-mcp1k work-pr128r1-on141-mcp2k
# -> 25 PASS, 0 FAIL, 0 SKIP
```

## Symptom

A coverage sweep of the manifests against the registry found the **entire
doc-84 long-muon / MCS family in no manifest and no registry entry** — nine
target events across four shipped, SBND-PRODUCTION-ON rounds:

```
497311 66366 313847 281595 53793 177536 77978 172794 347890 67026
```

Plus two more holes: **315167** had a registry entry that always reported
`SKIP` because no arm held the event — which reads like a pass — and
**406125**, pr/124's headline win (qF1 0.097 -> 1.000), had no entry at all.

This is precisely the exposure doc pr/127 was written about: a shipped fix that
died silently for ten days because nothing asserted it still worked.

## Fix

1. **`docs/pr/pr130-sentinel-manifest.tsv`** — the 12 events, their sample and
   QL root.  These live in no standard manifest, so they need their own.
2. **`scripts/pr130_sentinel_arm.sh`** — produces `work-sent<TAG>-mcp{1,2}k`
   at the current production point (fresh tag per run, M13).
3. **`scripts/pr127_sentinels.py`** — registry extended **11 -> 25 entries**.
   Ten of the eleven new thresholds sit between a knobs-off value measured in
   this round and the production value; the eleventh (66366) is an outcome-only
   sentinel for the reason below.

One measurement correction worth recording: 497311's rescue is quoted in doc 84
as `mu- 508 -> 766 MeV`, but its **PF node reads 473 MeV** at the production
point.  Those are different quantities — the 766.3 is the *shower* `kine_best`
(`kine_long_muon: ... range=766.3 ... fallback=1`).  The sentinel therefore
uses `shower_max_ge 13`, not `pf_node_ge "mu-"`.  A `pf_node_ge` assertion here
would have failed on a perfectly healthy fix.

## Verification — both directions

**Production (knobs on): 25 PASS, 0 FAIL, 0 SKIP.**  315167 now executes
(`arm=work-sent130-mcp1k`, proton 613 MeV) instead of reporting SKIP.

**Negative control** — a sentinel that cannot fail is worse than no sentinel.
Five targets re-run with their own knobs off
(`SBND_LONG_MUON_CATHODE_BRIDGE_{SHORT_GAP,TRACK_PARTNER}=0`,
`_LEVER=5`, `SBND_LONG_MUON_MEMBERS_GEOMETRY=0`,
`SBND_SHOWER_PASS4_PRUNE_GAP2=0`) into `work-sent130neg-*`:
**0 PASS, 5 FAIL, 20 SKIP** — every targeted sentinel fires.  The pre-fix
values it reports reproduce doc 84's own numbers exactly (313847 547 vs doc's
547.5; 172794 533 vs 533/533; 347890 429 vs 429/429), which independently
validates both the arm and the doc.

**Two negative arms were needed, not one.**  The first
(`work-sent130neg-*`) exercised only five entries — the other six reported SKIP
because their knobs were not in the disabled set.  A second arm
(`work-sent130neg2-*`, `SBND_LONG_MUON_RANGE_FALLBACK=0`,
`SBND_LONG_MUON_STUB_BRIDGE_LEN=6`, `SBND_LONG_MUON_MEMBERS_GEOMETRY=0`,
`SBND_LONG_MUON_CATHODE_BRIDGE=0`) covered the rest: **5 FAIL, 1 PASS**.
497311's knobs-off value is 508.5 — doc 84's pre-fix number to the decimal.

### The masked knob (66366) — the one that passed

66366 **PASSED with its own knob off**, and that is not a threshold problem.
Compiled-config proof: `long_muon_stub_bridge_len` reads **7.5 in production
and 6 in the negative arm**, so the override took effect; members_geometry,
cathode_bridge and range_fallback were off in that arm too.  The event still
reads `nseg_chain=4 L_cm=300.6 range=692.2` — **identical to production**.

doc 84 r1 recorded P3 as moving 66366's chain 126.2 -> 301 cm.  At today's
production point that outcome no longer depends on the knob: something later
produces the same chain by another route.  The *result* is correct; the
*dependency* is gone.  Consequences:

- 66366 is kept as an **outcome-only** sentinel (it still catches a regression
  of the result from any cause) and its registry comment says so.
- **`long_muon_stub_bridge_len` currently has no guarding event.**  Finding one
  needs a census of where the knob still changes an outcome — not attempted
  here, and listed as open.
- Generalisation worth carrying: *a sentinel proves an outcome, not a
  mechanism.*  Only the knobs-off arm distinguishes the two, and a fix can go
  inert without anything going wrong.

**The control also found a real defect in my own first draft.**  347890's threshold
was `pf_node_ge mu- 400`; with the knob off the event still reads 429 MeV, so
that assertion **PASSED on a dead knob** and only the log line caught it.
Tightened to 460 (between the measured 429 and the post-fix 488.7) and
re-verified in both directions.  Any future entry should be checked this way —
a threshold picked from the post-fix value alone is not a sentinel.

## Still open

- **292643** is in em114c so it is not a coverage hole, but it carries no
  sentinel because there is no settled verdict to assert yet — it drifted off
  the shape the owner approved (no pi+ at all now).  It needs the owner's read
  first; a registry entry written before that would pin the drifted state.
- **`long_muon_stub_bridge_len` has no guarding event** (see "the masked knob").
- The other doc-84 r1/r2/r4 knobs are each guarded by exactly one event; a knob
  whose only target has been subsumed the way 66366 was would look identical
  until someone runs the knobs-off arm.  Worth a periodic re-run, not just a
  one-time one.
- The pr/125 K5 / pr/123 r2 entries still assert only that their pool **fires**.
  For `kine_count_guard_freed` specifically, Part 1 shows firing is not the
  property we want guarded — that entry needs a negative companion once the
  171572 ruling lands.

---

# Part 3 — 292643: the drift has a named cause (doc pr/130 item 5b)

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr127_pf_history.py 292643          # 53 arms, mtime order, changes marked
PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_STEM_BACKFILL_BACK_GUARD=0 \
  ./run_pr_chain_batch.sh work-mcp1k-grp0825 work-pr130-292643-guardoff data 292643
```

## Finding

doc pr/127 §5.3 recorded 292643 as **"drifted — owner look wanted"** with no
cause.  `pr127_pf_history.py` brackets the drift to a **63-minute window** on
2026-08-28:

| arm | time | PF tree | Enu |
|---|---|---|---|
| `work-em114c-knobsoff-mcp1k` | 08-28 **17:07** | `mu- 441 \| pi+ 91 \| pi+ 65 \| pi0 150 \| g 154 \| g 65 \| g 8 \| g 5` | **1092.4** |
| `work-pr121r1-base141-mcp1k` | 08-28 **18:10** | `e- 216 \| g 65 \| g 8 \| g 5 \| mu- 59 \| mu- 441` | **857.5** |

Exactly one thing landed in that window — toolkit `59b3fb53` + `06dfa09f`,
**08-28 17:27, doc pr/120 round 1: `stem_backfill_back_guard` SBND PRODUCTION
ON.**

**Confirmed by a one-event A/B**, not by the timeline alone:

| | PF tree | Enu |
|---|---|---|
| approved (`work-mcp1k-prod0825`, 08-25) | `mu- 441 \| pi+ 91 \| pi+ 65 \| pi0 150 \| g 154 \| g 65 \| g 8 \| g 5` | 1092.4 |
| production today (guard **ON**) | `e- 227 \| g 65 \| g 8 \| g 5 \| mu- 59 \| mu- 441` | 858.5 |
| **guard OFF** (this run) | `mu- 441 \| pi+ 91 \| pi+ 65 \| pi0 150 \| g 154 \| g 65 \| g 8 \| g 5` | **1092.4** |

Turning the knob off restores the approved shape **exactly**.  The cost of the
guard on this event is **233.9 MeV, both pi+ (91 and 65), and an entire pi0
(150 MeV + its 154 MeV gamma)** — replaced by a spurious `e- 227` head and a
`mu- 59`.

Compiled-config proof (key-suppression idiom — absent key = off): the
production arm carries `"stem_backfill_back_guard" : true`; the guard-off arm
omits the key entirely.

## Why no gate caught it

`stem_backfill_back_guard` was flipped ON for an owner-approved win on 47212
(0.??? -> 1.000, doc pr/120).  292643 **is** in em114c, and the drift **is**
visible in `work-pr121r1-base141-mcp1k` — but that arm was a *base* arm taken
at the new production point, so the changed value silently became the new
baseline and every later comparison agreed with it.  Same shape as doc pr/127's
ten-day silence: the flip moved the baseline and nothing diffed across it.

This is the "tuned-to-one-event exposure class" Part 2's registry exists for,
and it is the second confirmed instance in two rounds.

## Open — for the owner

1. Is the pr/120 win on 47212 worth a pi0 and 234 MeV on 292643?  That is a
   physics trade, not a threshold.
2. **Blast radius MEASURED** (`work-pr130r1-gd141off-*` vs `work-pr128r1-on141-*`,
   member-content hashes): **6 movers in 141 events**, 135 unchanged.  And the
   trade is *two-sided* — turning the guard off is a net **-292.1 MeV**:

   | event | Enu guard ON (prod) | Enu guard OFF | delta | PF head |
   |---|---|---|---|---|
   | 179369 | 1713.1 | 1337.1 | **-376.0** | gamma 10 -> e- 38 |
   | 283515 |  866.9 |  739.1 | **-127.8** | proton 188 -> neutron 5 |
   | 67394  | 1207.3 | 1151.0 |  -56.3 | mu- 94 -> e- 43 |
   | 286655 |  554.4 |  505.5 |  -48.9 | proton 103 -> e- 304 |
   | 347824 | 1652.7 | 1735.6 |  +82.9 | pi+ 55 -> pi+ 55 |
   | 292643 |  858.5 | 1092.4 | **+234.0** | e- 227 -> mu- 441 |

   **So the guard is not a regression to revert.**  292643 is a real casualty,
   but on four of the six movers the guard *adds* energy, and it is net positive
   overall.  The owner question is therefore narrow: is 292643's pi0 worth more
   than the other five rows?

   **Caveat that must travel with this table**: these are Enu *deltas*, not
   accuracy.  Nothing here says which direction is closer to truth — a truth-level
   comparison would, and none was run.

   Cross-link worth noting: **179369 and 283515 are both in the 141-set q_miss
   top-10** (ranks 9 and 4), and 179369 is the parked "backward cluster" item.
   The pr/120 guard is entangled with the under-clustering residual, so any
   q_miss round has to hold this knob fixed or it will mis-attribute.
3. **Bee A/B uploaded for the owner's read** (`bee/pr130r2/`, index
   `pr130r2.index.txt`, same 6 events same order in both sets):
   guard **ON** (= production) `2a259e9f-7d28-45f9-b3bc-a43e26b3f5f0`,
   guard **OFF** `1747be7e-1fbb-432b-a026-d26d254fa63d`.
   Content-verified live-vs-package 24/24, and orientation cross-checked
   semantically on idx 0 (ON carries `e- 227` and no pi+; OFF carries `pi+ 91`
   and no `e- 227`) — doc pr/124's pair was once recorded swapped AND inverted.
4. Whichever way it goes, `stem_backfill_back_guard` needs a registry entry
   with **both** an assertion for the 47212 win and one for whatever 292643
   settles at — a single-sided sentinel is what let this sit unattributed.

---

# Part 4 — owner verdicts on the guard, and why the obvious fix is dead

## Owner ruling (2026-08-29), on the `bee/pr130r2` A/B

| idx | event | verdict | meaning for the guard |
|---|---|---|---|
| 0 | 292643 | **OFF better** | the guard's decline is WRONG — the stem should be absorbed |
| 1 | 179369 | **OFF better** | decline WRONG |
| 2 | 283515 | ON better | decline right |
| 3 | 347824 | ON better | decline right |
| 4 | 67394  | ON better | decline right |
| 5 | 286655 | ON better | decline right |

Note this **reverses the framing in Part 3**: 179369 was presented as the
strongest case *for* the guard (its ON side gains a `pi0 138` + a 216 MeV
gamma, +376 MeV).  The owner rules that pi0 spurious.  Enu direction was not a
proxy for correctness, exactly as the caveat warned.

## The complete labelled set — there is no unlabelled exposure

`stem_backfill_back_guard` declines on **exactly 8 candidates in 239 events**:
the 6 above plus pr/120's own two targets, 47212 and 281567, both
scanner-condemned (the events the guard was built for).  So every firing is
labelled.  Features are from the pre-existing byte-neutral `P120_STEM` census
(`WCT_SHOWER_ABSORB_DEBUG`), which tapes every chain candidate whether or not
the guard declines:

| evt | verdict | pdg | len_cm | ratio (MIP) | ang15 | ang60 | dist_cm |
|---|---|---|---|---|---|---|---|
| 47212  | decline ok (pr/120) |   13 |  3.76 | 1.62 | 150.17 | 152.17 |  4.99 |
| 281567 | decline ok (pr/120) |  211 |  6.95 | **1.08** | 150.42 | 155.85 |  8.91 |
| 283515 | decline ok | 2212 | 26.47 | 1.57 | 148.37 | 150.83 | 41.48 |
| 347824 | decline ok | 2212 |  1.90 | 3.07 | 167.93 | 161.74 | 10.58 |
| 67394  | decline ok | 2212 |  6.24 | 3.21 | 174.35 | 176.84 | 39.16 |
| 286655 | decline ok | 2212 |  8.32 | 3.26 | 154.24 | 154.99 | 27.88 |
| 286655 | decline ok (2nd) | 2212 | 8.32 | 3.26 | 144.06 | 144.35 | 24.71 |
| **292643** | **absorb wanted** |   13 | 14.02 | **1.27** | 172.84 | 171.67 | 16.87 |
| **179369** | **absorb wanted** |  211 | 11.26 | **1.46** | 161.94 | 172.83 | 88.11 |

## MEASURED DEAD: no admission-time feature separates

Every taped feature is interleaved between the two classes:

```
pdg      absorb=[13, 211]        decline=[13, 211, 2212 x5]      NO
len_cm   absorb=[11.26, 14.02]   decline=[1.9 .. 26.47]          NO
ratio    absorb=[1.27, 1.46]     decline=[1.08, 1.57 .. 3.26]    NO
ang15    absorb=[161.94,172.84]  decline=[144.06 .. 174.35]      NO
ang60    absorb=[171.67,172.83]  decline=[144.35 .. 176.84]      NO
dist_cm  absorb=[16.87, 88.11]   decline=[4.99 .. 41.48]         NO
```

**The obvious fix — exempt MIP-like stems — looked perfect on the owner's six
and is killed by pr/120's own two targets.**  On the 141-set alone the split is
clean (absorb-wanted are pdg 13/211 at ratio 1.27/1.46; decline-ok are all pdg
2212 at 1.57-3.26).  But **47212 is pdg 13 at 1.62 MIP and 281567 is pdg 211 at
1.08 MIP**, and both are scanner-condemned over-clustering.  281567's 1.08 sits
*below* both absorb-wanted ratios.  A MIP exemption would re-break the two
events the guard exists for.

*Method note: this is the second time this round that checking a shipped fix's
ORIGINAL targets killed a hypothesis that fit the new evidence perfectly.  Fit
a separator only against the complete labelled set, never the current round's
half of it.*

Two-feature cuts are available on paper (e.g. an 8.32 < len < 26.47 band
isolates both positives) but with **2 positives and 7 negatives** any such cut
is fitted noise, not a discriminator.  Not proposed.

## What could still work (ranked, none attempted)

1. **Vertex-relative geometry — the one physically-motivated family not yet
   taped.**  Every feature above is local to the shower/stem pair; *none
   references the main vertex*.  The physics: a stem that is the shower's true
   parent should lie BETWEEN the shower and the neutrino vertex, so absorbing
   it moves the shower start TOWARD the vertex; a separate hadronic prong sits
   AT the vertex and points away.  Cheap to test — extend the existing
   `P120_STEM` tape with `d(stem, main_vertex)`, `d(shower_start, main_vertex)`
   and the sign of the change, then re-run the two probe arms.  One probe, one
   arm, and the labelled set is already complete.
2. **Accept the trade.** The guard is right on 6 of 8, including both events it
   was built for.  Its two errors cost 234.0 MeV lost (292643) and 376.0 MeV
   spuriously gained (179369) — both failures of the owner's pr/128 metric, so
   "accept" is not free.
3. **A downstream coherence test** — decide after the absorb by asking whether
   the enlarged shower is a consistent object, rather than at admission.
   Larger design; matches the recurring finding below.

**Pattern, now four rounds deep**: pr/119 (no local separator), pr/128
(proximity is not continuation), pr/129 (over-clustering is not a distance),
and now pr/130 Part 4.  Local admission-time geometry keeps failing to encode
the owner's judgement.  That is an argument for spending the next effort on
option 1 and, if it also fails, on option 3 — not on further threshold work.

---

# Part 5 — vertex-relative geometry: one feature separates, on one event

## Repro

```bash
# toolkit: P120_STEM census extended with dvtx_start / dvtx_stem / toward / vang
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_EXTRA_STAGES=pr_display PR_JOBS=4 WCT_SHOWER_ABSORB_DEBUG=1
./run_pr_chain_batch.sh work-mcp1k-grp0825 work-pr130-vtx-mcp1k data 292643 283515 67394 286655
./run_pr_chain_batch.sh work-mcp2k-grp0825 work-pr130-vtx-mcp2k data 47212 281567 179369 347824
grep -h P120_STEM work-pr130-vtx-*/pr_evt*/stdout.log
```

Byte-neutrality: extended census sits inside the existing `pr93_absorb_dbg()`
gate; **8/8 mabc-pr.zip SAME** vs the production arms.

## Measurement (complete 8-candidate firing set)

| evt | verdict | ang15 | dvtx_start | dvtx_stem | toward | vang |
|---|---|---|---|---|---|---|
| **292643** | **ABSORB wanted** | 172.84 | **46.84** | **22.64** | +24.20 | **14.83** |
| **179369** | **ABSORB wanted** | 161.94 | **88.11** | 0.00 | +88.11 | n/a |
| 283515 | decline ok | 148.37 | 44.34 | 0.00 | +44.34 | n/a |
| 67394  | decline ok | 174.35 | 43.46 | 0.00 | +43.46 | n/a |
| 286655 | decline ok | 154.24 | 27.88 | 0.00 | +27.88 | n/a |
| 286655 | decline ok | 144.06 | 26.66 | 0.00 | +26.66 | n/a |
| 347824 | decline ok | 167.93 | 11.54 | 0.00 | +11.54 | n/a |
| 281567 | decline ok (pr/120) | 150.42 |  8.91 | 0.00 |  +8.91 | n/a |
| 47212  | decline ok (pr/120) | 150.17 |  5.01 | 0.00 |  +5.01 | n/a |

### The designed features mostly collapsed

**`dvtx_stem` is 0.00 for 7 of 8** — the stem candidate sits *on the neutrino
vertex* in almost every case.  So "does the stem lie between the vertex and the
shower" is trivially true for the whole population, `toward` degenerates into
`dvtx_start`, and `vang` is undefined (zero-length vector at the vertex).  The
hypothesis as designed — parent stem *between* vertex and shower vs prong *at*
the vertex — **does not describe this population**: nearly every candidate is a
vertex prong.

### The one feature that does separate — and why I will not ship it

`dvtx_start` separates the complete labelled set: ABSORB-wanted
[46.84, 88.11] vs decline-ok [5.01 … 44.34], **margin 2.50 cm**.

It is not shippable on this evidence:

- **The separation rests on a single event.** For the 7 vertex-prong candidates
  `dvtx_start` is numerically identical to `dist_cm`, which Part 4 already
  measured as NOT separable.  The only row that moves is **292643, 16.87 ->
  46.84**, because it is the one stem that is not a vertex prong.  Remove that
  one event and the feature is Part 4's `dist_cm` again.
- **The margin is 2.50 cm on a ~45 cm scale — 5.6%.**  Compare what this
  campaign has actually shipped: doc pr/128's continuation kink had **25 deg**
  between last-accepted and first-rejected; Part 1's guard-freed kink had
  **99 deg**.  5.6% with 2 positives is fitted noise by the same standard that
  rejected Part 4's length band.

There *is* a coherent physical story available (a shower starting far from the
vertex is detached, so a backward stem is plausibly its parent; a
vertex-attached shower's backward stem is a sibling prong).  The story is worth
remembering, but a story plus one event is not a discriminator.

## Verdict: the local-geometry family is exhausted

Ten features have now been measured against the complete labelled set —
pdg, len_cm, dQ/dx ratio, ang15, ang60, dist_cm (Part 4) and dvtx_start,
dvtx_stem, toward, vang (here).  **None separates on evidence that meets this
campaign's own bar.**

**Recommendation: stop threshold work on admission-time features.**  Two
options remain, and they are qualitatively different from everything tried:

1. **Widen the labelled set before trusting `dvtx_start`.**  The guard fires
   only 8 times in 239 events, but the P120_STEM census tapes *every* chain
   candidate including accepted absorbs.  A 239-event re-run with the extended
   census would show whether the 44 -> 47 cm boundary is a real gap in the
   candidate population or an accident of nine points.  Cheap and decisive
   about whether Part 5 is worth revisiting; it does not by itself produce a fix.
2. **A downstream coherence test** (Part 4 option 3): judge the enlarged shower
   after the absorb, not the geometry before it.  This is the only remaining
   family that could encode what the owner is actually judging — and the
   pr/119 / pr/128 / pr/129 / pr/130-p4 / pr/130-p5 sequence is now five
   consecutive measurements pointing that way.

Until one of those lands, the guard should stay as shipped: right on 6 of 8
including both founding targets, with its two known errors (-234.0 MeV lost on
292643, +376.0 MeV spurious on 179369) documented here.

---

## Part 6 — where the campaign should go next (measured, 2026-08-29)

Part 5 ended with two options and a recommendation to stop admission-time
threshold work. A third option now exists, and it is better founded than
either, so it supersedes them as the recommendation.

Item 1 (`pr130-qmiss-refresh.md`, run in a peer session) refreshed the q_miss
ranking at the true production point and returned **GO** on concentration
while overturning the premise: the "75% of charge error is q_miss" figure is
**98-set only**; on the 141-set q_miss is 48.4% and q_extra is the larger
half. The 43% figure carried in earlier pr/130 notes is stale — 48.4% is
current.

Item 1b (`pr130-qextra-refresh.md`) took that finding apart. Held to the
**same affirmative standard on both sides** — only charge a scanner
explicitly marked — the 141-set splits **q_miss 43.7% / q_extra 56.3%**
(1.345e7 vs 1.731e7). The split *widens* under the stricter standard. The
whole affirmative q_extra pool is **22 segments in 10 events, top-4 = 74%**,
and **18 of the 22 sit in a cluster other than the shower's own**.

Why that beats Part 5's options 1 and 2 for this knob's own problem: the back
guard's difficulty has been a labelled set of **8** candidates with no
separator across ten measured features. The q_extra pool is the same
absorber's *outcomes*, already adjudicated, and one of its ten events
(**286655**) is one of those very 8 candidates. It replaces threshold-fitting
on eight points with a truth-anchored target list — and its top item is a
110 cm pdg-13 track absorbed into an EM shower, the same failure the pr/128
metric's third term names.

Options 1 and 2 from Part 5 are not withdrawn; they are now second and third.
The two knowingly-wrong events this doc leaves in production (292643 −234.0
MeV, 179369 +376.0 MeV spurious) are unchanged by any of this and still need
the Part 4 fix.

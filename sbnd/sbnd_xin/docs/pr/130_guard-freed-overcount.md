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

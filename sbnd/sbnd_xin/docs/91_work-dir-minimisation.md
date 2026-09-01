# doc 91 — the count-driven retire round, and the sentinel regression it found

**Status: round executed, 101 → 52 work dirs. One finding is OPEN and is the
recommended next step (§7): three shipped, owner-approved fixes no longer fire
in production.**

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the finding (§7) -- 27 PASS, 6 FAIL against the current production arm
./scripts/pr127_sentinels.py --arms 'work-*-prod0901b'; echo rc=$?

# per-arm, which is NOT the same question (§7.2)
python3 scripts/retire/sentinel_guard_20260901b.py \
        scripts/retire/state-20260901b/plan.json "$PWD"

# the round
python3 scripts/retire/verify_group_dupes_20260901b.py
python3 scripts/retire/plan_20260901b.py                  # 16 asserts, OVERALL: PASS
RETIRE_JOBS=32 python3 scripts/retire/archive_records_20260901b.py
./scripts/retire/retire_20260901b.sh A                    # DRY RUN
RETIRE_REPLAN=1 python3 scripts/retire/plan_20260901b.py
CONFIRM=yes ./scripts/retire/retire_20260901b.sh A
```

## 1. Why this round exists

Owner, verbatim: *"The sbnd_xin directory still have a lot of work* directory,
do we need all of them? Can we retire some and go minimum? I understand that
they do not take much disk space, but it is just difficult to look at them."*

Then, on being shown the classification: *"Peer is done, we can remove or
retire them. We want to keep the latest production though."*

So **the metric is directory count, not bytes** — 101 → 52. The 7.08 GiB freed
is a side effect and is not the reason for any decision here. Two consequences
that a byte-driven round would have got wrong:

- the 7 `work-87cal*` arms (21 MB total) are worth releasing, because they are
  the worst count-per-byte in the tree — the exact thing the owner complained
  about;
- `work-pr134-f086-*` (1.2 GiB) is worth keeping, because §7 makes it load-bearing.

## 2. Three protections that turned out not to hold

The 09-01 round's KEEP set was inherited rather than re-derived. Checking it
found three faults, all in the *evidence for keeping*, not in the arms:

**2.1 The citation census was circular.** It counted an arm as "cited" when a
previous retire *planner* named it — i.e. an arm was protected because it had
been protected. `scripts/retire/` is excluded from the census now. With it
included, `work-tfix388-r9` scored 15 script citations; with it excluded, 0
(it is still kept, on the separate and real ground that it is not reproducible).

**2.2 The census was name-exact, and doc 87 cites templates.** All 28 doc-87
arms scored `docs=0`, because doc 87 names them as `work-87knob-def-` with a
trailing dash. A census that returns 0 for a cited arm is indistinguishable
from one that works. Both faults are fixed; the census is template-aware and
`scripts/retire/`-blind.

**2.3 Two "hardcoded default arm" protections were mis-reads.**
`work-em114c-prodnow-*` and `work-pr117r1-onK1-mcp1k` were kept on the stated
ground that they were argparse defaults in `em_display/prep_pr121.py:15` and
`prep_pr117.py:17,28`. They are **docstring usage lines**.
`grep -n "default=" em_display/prep_pr11{7,21}.py` matches no arm name at all.

## 3. Two directories in the removal set were not arms

| dir | what it actually held | disposition |
|---|---|---|
| `work-nuecc48-prsmoke2` | 3 git-tracked runner scripts, no event data. Its one consumer, `scripts/repro_track_pid_evt172230.py:4`, already pointed at `nupr_evt172230_mipvote_on/` — a subdir that no longer exists | scripts `git mv`-d to `scripts/legacy/` |
| `work-stmcamp-d66new` | nothing but `nusel_labels/d66flip/` — **the tree's only nusel label store**, 22 tracked files. Its comparison partner `work-stmcamp-d66old` was retired rounds ago, so `d66_flip_report.py` could not run either way | store `git mv`-d to `./nusel_labels/d66flip/`; `d66_scan_score.py` repointed to `--root .` |

Both were emptied **before** the round, so ASSERT 3 saw no tracked file in the
removal set and no record was deleted. `nusel_labels/` now sits at top level
alongside `em_labels/`, `vertex_labels/` and `overclustering_labels/`, which is
where the convention already pointed.

## 4. What was released, and what it costs

49 dirs, 7.08 GiB.

| block | n | ground | stated cost |
|---|---|---|---|
| doc 90 peer arms | 9 | owner released them by name | doc 90 §6's determinism claim ("140 pi0 events, 0 differences") keeps its **five repeat arms** only as archived records. A determinism claim is exactly the kind whose value dies with one lost repeat — priced here because the release was a one-line instruction |
| doc 87 gate arms | 12 | §5 — the release condition this tree wrote for itself is met for these and only these | none; the successor-gate TSV keeps the claim re-checkable |
| doc 87 `SBND_PR_CALIB` matrix | 7 | 1 ncpi0 event each, 21 MB, re-runnable from `work-ncpi0-grp0825` | doc 87 §6.5's table stays as the record |
| r1qlmc / r2mc sim chain | 10 | released **in full** — both imaging hubs, both Q/L hubs, both PR arms, both post-flip Q/L roots, both out-roots | the two sim samples leave the tree entirely. Re-entering them means imaging from reco1. doc pr/76's numbers become text-only. Released as a chain so closure holds in both directions rather than by exempting a link |
| `work-sent130-{mcp1k,mcp2k}` | 2 | §6 — production now contains all 30 sentinel events itself | none; ASSERT 15 re-derives this every run |
| `pr117r1-onK1-*`, `em114c-prodnow-*` | 6 | §2.3 mis-read; displays superseded by the f086 manifests at the 0.86 production scale | `em117-117onK1-manifest.tsv` (98 rows) stops resolving — printed by name in ASSERT 11's STATED COST list |
| the three non-arms + `work-em114-probe3` | 3 | §3; and probe3's `prod0825` comparator was released on 08-31b, so doc pr/114 §3's proof was **already** text-only before this round touched it | none new |

`work-em114-probe3` is the cleanest instance of the pattern this round found
three times: **a protection outliving its ground**. The proof it existed to
support died when its comparator was released a round earlier, and nothing
noticed.

## 5. doc 87, split by section rather than kept or released as a block

`PROTECTED.txt` wrote the release condition on 09-01: *"these go when doc 87 has
a settled successor round that re-establishes sec 6.1 and sec 6.2 at production
scale — not merely when doc 87 is closed."* Applied section by section:

**§6.1 and §1.4 are re-established, at 3067 events instead of 482, and not by
inference.** doc 89's successor gate compared `work-*-prod0901` (toolkit
`ddce7430`, which *predates* the doc-87 knobs entirely) against
`work-*-prod0901b` (knobs present, at their defaults, post-merge) and found
every shared product identical on all 3067 events. "Knobs at their defaults
change nothing" and "the merge changes nothing" are exactly what that measured.
The evidence outlives the arms: `state-20260901/successor-gate.tsv`, one row
per event. → the 12 gate arms go.

**§6.2 is not re-established.** `prod0901b` ran *full* output; nothing in doc 89
exercises output suppression. `work-87knob-{min,sup}-*` are still the only
minimal-output arms in the tree. → 4 stay.

**§6.4 and §4.6** — group-mode is a different output layout, not a different
setting of the same one, and `prod0901b` is per-event mode. → 5 stay.

## 6. The sentinel arms: released on a measurement, not a guess

`work-*-prod0901b` was checked to contain **all 30** `pr127_sentinels.py`
events. The suite locates an event by scanning arm roots for `pr_evt<N>/`, and
30/30 resolve, so it now runs against production directly.
`work-sent130-{mcp1k,mcp2k}` existed only because production used to be an event
*subset*; it is not any more. ASSERT 15 re-derives this on every run rather than
trusting today's reading.

The **negative** controls (`work-sent130neg*`, 6 arms) stay. They are the only
on-disk proof the registry *can* fail — the one property a green suite cannot
demonstrate about itself.

## 7. OPEN FINDING — three shipped fixes no longer fire in production

Run against `work-*-prod0901b` the sentinel suite is **27 PASS, 6 FAIL**.

### 7.1 The six

| event | fix | verdict |
|---|---|---|
| 47212 | pr/120 backward-stem guard | REGRESSED — `pi+` node is back |
| 137238 | pr/93 r4 + pr/127 sccc | REGRESSED — `mu-` 88/60/58 MeV, want ≥150 |
| 173819 | pr/125 pass3_cone guard | REGRESSED — e⁻ 283 MeV, want <200 |
| 292643 | pr/130 B back-guard dvtx | REGRESSED — no `pi0` node |
| 406125 | pr/124 gap-band prune | REGRESSED — the fix never logs |
| 393505 | pr/129 pointing test | Enu 559.9 vs window [560, 572] — a **0.1 MeV** miss, i.e. drift |

For 406125 the knob is still `shower_pass4_prune_gap2 = 25` in
`wct-pr-perevt.jsonnet` and the C++ log line still exists in
`NeutrinoShowerClustering.cxx:9702`. It simply no longer fires. That is the doc
pr/127 failure mode — a shipped fix dying silently — recurring.

**Not diagnosed here** (CLAUDE.md §5.7: report, do not tune). It is the
recommended next step.

### 7.2 A methodological correction, and it changed the answer

`pr127_sentinels.py:find_arm()` returns the **first** arm in sorted glob order
that holds `pr_evt<N>/`, and evaluates the event there only. So passing several
arms at once reports the verdict of whichever arm sorts first, not the best
verdict available.

Measured: a combined run over every non-production arm reports **all six FAIL**.
Per-arm evaluation finds **five of the six passing somewhere**. A guard built on
the combined run would have concluded there was nothing to protect and released
the arms.

This corrects two readings made earlier in the same session from combined runs:
137238 and 292643 are **not** "pre-existing failures everywhere" — each passes
in exactly **one** arm on disk:

| event | arms holding it | arms where it still PASSES |
|---|---|---|
| 47212 | 5 | `pr125r1-flipK598-mcp2k`, `pr130r1-probe98-mcp2k` |
| 137238 | 14 | `pr130r1-probe98-nuecc48` — **single witness** |
| 173819 | 6 | `pr125r1-flipK5141-mcp2k`, `pr130r1-probe141-mcp2k`, `pr134-f086-mcp2k` |
| 292643 | 6 | `pr134-f086-mcp1k` — **single witness** |
| 406125 | 9 | 4 arms, 3 of them kept |
| 393505 | 6 | none — red everywhere |

So the corrected count is **five regressions with a surviving witness, one red
everywhere**.

### 7.3 Bee sets for the hand scan (owner request, 2026-09-01)

Two sets, **same event order**, so Bee index *i* is the same event in both and
they can be stepped through side by side:

| | set |
|---|---|
| **production — the FAIL side** | https://www.phy.bnl.gov/twister/bee/set/38f1f41a-86c8-4deb-95c2-2842588b54fc/event/list/ |
| **witness arms — the PASS side** | https://www.phy.bnl.gov/twister/bee/set/b614ff24-1894-4d30-8b2c-7255e902ee5c/event/list/ |

| bee idx | event | production arm | witness arm |
|---|---|---|---|
| 0 | 47212 | `work-mcp2k-prod0901b` | `work-pr125r1-flipK598-mcp2k` |
| 1 | 173819 | `work-mcp2k-prod0901b` | `work-pr134-f086-mcp2k` |
| 2 | 406125 | `work-mcp2k-prod0901b` | `work-pr134-f086-mcp2k` |
| 3 | 137238 | `work-nuecc48-prod0901b` | `work-pr130r1-probe98-nuecc48` |
| 4 | 292643 | `work-mcp1k-prod0901b` | `work-pr134-f086-mcp1k` |
| 5 | 393505 | `work-mcp2k-prod0901b` | `work-pr134-f086-mcp2k` — **no passing arm exists**, shown for context only |

Built by `scripts/bee/make_pr_bee.py` from `bee/d91/`; the `mc` layer is the PF
jsTree the sentinels assert on, so the failure is visible directly in the tree.

**CONTENT-VERIFIED against the live server**, not just the local zips — all six
events serve different PR layers between the two sets, and the differences are
the ones the sentinels name:

| event | production serves | witness serves |
|---|---|---|
| 47212 | **no `pi+` node** | `pi+ 53 MeV` |
| 137238 | `mu-` 88 / 60 / 58 | `mu-` 88 / 60 / 58 **+ 207 + 66** |
| 173819 | `e- 283 MeV` | `e-` max 18 MeV (shower re-rooted as proton) |
| 292643 | **no `pi0` node** | `pi0 140 MeV` |

**A correction to §7.1's table.** 47212 was reported as "`pi+` node back". It is
the opposite: the sentinel is `pf_contains "pi+"`, the guard-ON state has a
`pi+ 53 MeV` track (pr127_sentinels.py:161), and production now has **no `pi+`
at all** — the shower ate the backward stem, which is the guard-OFF signature
the fix was shipped to prevent. The witness arm reproduces the documented
guard-ON value exactly, 53 MeV.

### 7.3 Why the next round is tractable

406125 has a passing witness (`work-pr134-f086-mcp2k`) and a failing one
(`work-mcp2k-prod0901b`) both on disk, same event, knob ON in both. That is a
diffable pair, not an open-ended hunt.

## 8. The new guard, and proof it can fail

`scripts/retire/sentinel_guard_20260901b.py` — one implementation, two callers
(plan ASSERT 15+16 and driver interlock 8), so they cannot drift.

Interlock 7 (the successor gate) **degenerates to "not required"** this round,
because no production arm is released. A round whose newest interlock cannot
fire has no new guard at all — the 09-01 lesson, where a tmp-sweep interlock
read "clear" only because its glob matched zero arms. So interlock 8 was proven
able to fail before being trusted:

| corrupted KEEP | result |
|---|---|
| real KEEP | clean |
| drop `work-pr134-f086-*` | **FIRES** — 292643 loses its only witness |
| drop `work-pr130r1-probe-*` | **FIRES** — 137238 loses its only witness |
| drop `work-pr125r1-*` | clean (correct — its events keep other witnesses) |
| KEEP = production only | **FIRES** |

**First formulation was too weak and the control caught it.** It began as
"every failing sentinel keeps some non-production arm *holding* that event".
Dropping `work-pr134-f086-*` did not make it fire, because `vtx105-base`,
`em114c` and `pr130r1-probe` hold the same events — they are coverage, at the
pre-0.86 EM scale, where a FAIL proves nothing. Holding the event is not the
property that adjudicates a regression; **still passing** it is.

**Idempotent across the deletion boundary**, checked before the round ran: the
guard returns clean both with all 101 dirs visible and with only the 52
survivors visible. Interlock 8 re-runs at delete time and could run again after
a `RETIRE_REPLAN` cycle; a guard that refuses once its own round has succeeded
is a tripwire, not a guard. The only difference between the two runs is 406125
dropping 4 → 3 witnesses, which does not change the verdict.

**Read ASSERT 15 (coverage) as weak.** Its control showed it does not fire even
when the production arms are dropped from KEEP, because the other kept arms
cover all 30 events. It is honest but nearly unfalsifiable given this KEEP set;
the regression half is what bites.

## 9. Things the guards caught

- **ASSERT 12 caught my own shell.** It scans `ps -eo args` for removal-set
  names, and the bash command that ran the planner also carried the heredoc
  that *writes* the header naming `work-nuecc48-prsmoke2` and
  `work-stmcamp-d66new`. Fixed by running the planner in its own clean
  invocation — a test-hygiene fix, not a relaxation.
- **ASSERT 12 then held the round for ~50 minutes** on the doc-90 arms' mtimes,
  after the owner had already released them. Correct behaviour: the window
  (`FRESH_WINDOW_S = 3600`) is about *writes*, not permission, and disarming it
  because a human said "done" is how a half-written arm gets deleted. Waited.
- **ASSERT 2's PASS message was stale.** It printed "the tree's only live label
  dir inside `work-*` is `work-stmcamp-d66new/nusel_labels`, in KEEP" — a
  hardcoded string, and that dir had just been emptied and moved to the removal
  set. The assert was right (0 label dirs in the removal set, strict form); the
  explanation was wrong. Corrected, because a misleading PASS message is what
  misleads the next round.
- **ASSERT 10** required all 8 newly-broken script references to be
  acknowledged by name with their cost before the round could proceed.

## 10. Result

| | before | after |
|---|---|---|
| `work*` dirs | 101 | **52** |
| sbnd_xin | 74.4 GiB | **65.8 GiB** |
| released | — | 49 dirs / 7.08 GiB |
| record archive | — | `campaign-close-20260901b/`, 180 MB, integrity **PASS 49/49** |

Asserts: **16 PASS**. Interlocks: 8, of which 7 degenerates by design and 8 is
new and proven able to fire.

## 11. Recommended next step

**Diagnose the 406125 regression** (§7.3). It is the most tractable of the five:
one event, knob ON, log line present in the source, a passing witness arm and a
failing production arm both on disk. Whatever killed it is a firing-condition
change between the pr/134 epoch and toolkit `d52d818c`, and the same mechanism
plausibly explains 47212 and 173819.

Then re-baseline the suite: 393505's window is missed by 0.1 MeV, which is
drift, and `pr127_sentinels.py`'s own header says an energy-scale change (0.80 →
0.86, doc pr/135) moves every absolute-MeV threshold. Re-baselining *before*
diagnosing would hide the other five; do it after.

Deferred, unchanged from doc 89 §11: the `~/tmp` sweep
(`scripts/retire/sweep_tmp_20260901.sh`, 18.7 GiB) still needs an owner-run
`CONFIRM=yes`. The bokeh viewer that pinned `~/tmp/pr138_glcolor` was stopped
this round on the owner's instruction, so that directory is now releasable too.

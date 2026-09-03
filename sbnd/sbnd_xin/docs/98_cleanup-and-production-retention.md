# doc 98 — cleanup: retire the middle, keep the latest production

**Status:** planned, gated, dry-run clean; the two `CONFIRM=yes` steps are
**owed** (the permission gate declines them, as it did in doc 89 §Phase 4b).

Owner scope, verbatim (2026-09-02):

> "the sbnd_xin directory now is exploded to 189G, which needs to be clean up.
> We should retire the middle work* directory, and we only need to save the
> latest production results for Q/L matching as well as PR results."

and, separately:

> "By the way, please also clean up a bit for ~/tmp please"

So the metric this round is **bytes** (doc 91's round was dir *count*), and the
keep rule is given rather than inferred: the latest production stage A and
stage B. After doc 97's flip that is `work-*-d97fv` (Q/L at
`ref/prod-2026-09-04`, `sep_fv_point` ON) and `work-*-d97fvpr2` (its PR tail).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# A. freeze the record of the half of grp0825 that is being retired in place
python3 scripts/retire/rollup_grp0825_ql_20260902.py     # 18402 member rows

# B. carve the sentinel witness out of an arm that is about to go
python3 scripts/retire/witness_sentinels_20260902.py     # 28 events -> work-sent97-*
python3 scripts/d97_sentinels.py --arms 'work-sent97-*'  # MUST be 31 PASS, 0 FAIL

# C. plan (7 interlocks) and archive the record layer
python3 scripts/retire/plan_20260902.py                  # OVERALL: PASS
python3 scripts/retire/archive_records_20260902.py       # INTEGRITY GATE PASS 62/62

# D. dry run, then execute  <-- D2 and E2 are OWED
./scripts/retire/retire_20260902.sh                      # dry run
CONFIRM=yes ./scripts/retire/retire_20260902.sh          # (D2) 125.0 GiB

# E. the same for ~/tmp
./scripts/retire/sweep_tmp_20260902.sh                   # dry run
CONFIRM=yes ./scripts/retire/sweep_tmp_20260902.sh       # (E2) 30.0 GiB
```

## 1. What goes, and what it comes to

| | dirs | GiB |
|---|---|---|
| stage-B arms, 9 superseded PR epochs (`prod0901b`, `d94probe`, `d94hadron`, `r2probe`, `r2entry`, `r3probe`, `r3entry`, `d97onpr`, `d97off2pr`) | 36 | 92.0 |
| stage-A arms, 2 superseded Q/L epochs (`d97on`, `d97off2`) | 8 | 20.0 |
| doc 97 identity-gate probes (`d97idg`, `d97idgb`, `d97idgc`, `d97idgd`) | 13 | 4.0 |
| small dead probes (`d97chk{,2,3}`, `d94scan{,off}-64475`) | 5 | 0.03 |
| `work-*-grp0825/ql_evt*`, retired **in place** | (3067) | 9.0 |
| **sbnd_xin total** | **62** | **125.0** |
| `~/tmp`: 5 closed-round libsnaps, `doc87/lib-knob`, 3 cmake trees, `dbg25/groupbuild` | 10 | 30.0 |

sbnd_xin 189 → ~64 GiB; `~/tmp` 62 → ~32 GiB. 172 → 114 work dirs (110 kept
plus the 4 new `work-sent97-*`).

## 2. Five things measured rather than inherited

**2.1 `grp0825` is not one arm, and deleting it would have broken everything.**
10,478 symlinks across the tree resolve into `work-*-grp0825/evt<N>` — it is the
imaging substrate every Q/L arm borrows, including production's. Nothing
anywhere points at `grp0825/ql_evt<N>`, and that Q/L is two operating points
stale (doc 97 §2 measured today's knob-off run *failing* to reproduce it —
epoch drift, not nondeterminism). So the arm is **split**: imaging kept, Q/L
retired in place. Because that is a partial deletion of a `PROTECTED.txt` arm,
its 18,402-member hash rollup was frozen first
(`state-20260902/grp0825-ql-rollup.tsv`), and the `PROTECTED.txt` line was
rewritten to say **imaging-only** so the next round cannot read the old KEEP as
still covering Q/L.

**2.2 Retiring `d97off2pr` would have destroyed the only 31/31 sentinel arm.**
This changed the plan. Production `d97fvpr2` is **30 PASS / 1 FAIL** — 105074
(pr/128 class B) asserts a PF node production deliberately no longer makes, and
re-anchoring it is an open owner call I flagged last round. The only arms where
the suite is 31/31 were `work-*-d97off2pr` and `work-*-prod0901b`, and this
round releases **both**. Deleting them would have stranded the open sentinel
with nothing to adjudicate against — doc 91's interlock 8 exists for exactly
this. So the 28 distinct sentinel events were copied out into `work-sent97-*`
(165 MB) **first**, and interlock 2 refuses the round unless the suite reports
31 PASS / 0 FAIL against the witness *alone*. It does.

**2.3 A peer session is live in this tree.** `wcp-porting-img` `19d32520`
("pdvd: doc 25 §13 … gate round 7") landed at 16:54 today, and
`work-{ncpi0,nuecc48}-doc25new7` — PDVD doc 25's *SBND regression-gate* arms,
which live in sbnd_xin — was written at 16:46. `doc25*` is protected by prefix
for the life of that round, the same treatment doc 91 gave doc 90's arms.

**2.4 The `prod0901b` pin is spent, and says so itself.** Its `PROTECTED.txt`
line reads "the SBND production baseline at `ref/prod-2026-09-01b`". It was
pinned by doc 91's owner instruction *"We want to keep the latest production
though"* — and prod0901b is no longer the latest. The same sentence that
protected it releases it. Recorded so the removal is a decision, not an
inference.

**2.5 `doc87/lib-knob` released on today's measurement.**
`sweep_tmp_20260901.sh` asserted it was "an exact md5 duplicate of lib-flip" —
but **that sweep never ran** (doc 89 recorded it as "still owed"), yet all
eleven of its DROP dirs are already gone by another route, so its list is
spent. Re-measured today: 790/790 files md5-identical. `lib-flip` is kept.

## 3. The interlocks, and the one that had to be rewritten

`plan_20260902.py` — all 7 PASS:

1. production complete — `d97fv` 3067/3067, `d97fvpr2` 3067/3067, every product
   present, `rc=0` (the "runner rc=1 with failed:0" lesson: verify products, not
   the runner's own summary)
2. **sentinel witness 31/31 standalone**, before its sources go
3. no symlink in a kept dir resolves into a retiring dir — 0 would dangle
4. no live writer on any retire target
5. no `PROTECTED.txt` arm retired except the spent `prod0901b` pin
6. the grp0825 ql rollup covers 3067 of 3067 dirs
7. keep/retire disjoint; all 12 production dirs on the keep side

**Interlock 4 was wrong twice and is the interesting one.** First formulation
was "nothing touched in the last 6h", which failed on arms *this same session*
produced hours ago. Age is not liveness, and tuning the threshold until it
passed would have been the wrong move — so it was replaced by a causal test:
sample mtimes, wait 20 s, re-sample, demand zero change, plus zero open file
handles and zero matching processes. That then failed on 6 `wire-cell`
processes belonging to **another user** (`/home/jjo/…/pdhd/`) which cannot
touch sbnd_xin; the process match is now scoped to this tree, because a match
that fires on unrelated noise is unfalsifiable in practice. Proven able to
fail before being trusted: a scratch dir written during the window is detected,
and only that dir.

## 4. What the record layer preserves

`archive_records_20260902.py` → `archive/records/campaign-close-20260902/`,
**integrity gate PASS 62/62**. Heavy classes (pctree, mabc, calib, npz,
clusters, opflash, tracking) are dropped; what survives per arm is
`stdout.log`, `wct_{ql,pr}_evt<N>.log`, `.wct-cfg-evt<N>.json`,
`nusel-evt<N>.tsv` and the arm-level `nusel-events.tsv` / `nusel-table.tsv` —
the per-event verdict tables and the compiled config each arm actually ran
under. Symlinks are *recorded*, never followed, so no record tar can pull in a
copy of the imaging substrate.

## 5. The judgement call, stated as a line item

`work-*-d97on` / `work-*-d97onpr` (20.4 GiB) are released, and this is the one
call worth naming. `sep_track_recarve` is still default OFF and its Bee idx 9
and 10 (mcp2k 94392, 53793 — its two sentinel breaks) are **unadjudicated**, so
the owner may still ask for it. What survives the bytes: doc 97 §4–§5 carry the
measurements, the Bee sets are uploaded
(recarve `ce2b4924-a466-4149-8af1-274ea28c3b3c`), and the arm re-runs in ~25 min
from kept imaging via `scripts/d97_on_arms.sh`. If the owner would rather scan
first, keep `d97on`/`d97onpr` and the round frees 104.6 GiB instead of 125.0.

## 5.1 Two things the `~/tmp` sweep does NOT touch, on purpose

The sbnd_xin round archives a record layer for all 62 arms; the `~/tmp` sweep
has **no record layer**, which is fine for regenerable cmake trees and
explanatory-only libsnaps but not for evidence. So:

- `scan2`, `pr40r5`, `pr43_cleanhead_ref48`, `pr88`, `pr63_render` were
  **dropped from the sweep**. A `find` for `*.md5|*.tsv|*.json|*.log` returns
  10–74 artifact-class files in *each* (`scored.json`, `rank-*.tsv`,
  `bisect-*.log`, `track_candidates.json`) — closed-round evidence, not
  scratch. 0.33 GiB is not worth deleting a possible last copy; doc 89's own
  rule was to copy such artifacts out of `doc87/` *before* sweeping it. Left
  for a round that archives them properly.
- `dbg25/groupbuild` is swept, but a pre-step first copies its
  artifact-class files (`entry_group.tsv` and friends, <4 MB each) into
  `archive/records/campaign-close-20260902/tmp-artifacts/`. The `dbg25/` top
  level — `cfg-live-{before,after}.md5` (the doc-95 compiled-config proof),
  `dump.log`, `img-*.log`, `pr-*.log` — is not in the drop list and survives
  untouched.

## 6. Kept, with reasons

`grp0825/evt*` (imaging substrate) · `d97fv` + `d97fvpr2` (production) ·
`sent97-*` (sentinel witness) · `vtx105-base` (1930 citations incl.
`dl_vtx_training`) · `em114`, `em114c`, `pr130r1-probe141` (live `em_display`
manifests) · `pr130r1-probe98`, `pr134-f086` (sentinel witnesses, doc 91
interlock 8) · `doc25*` (live peer round) · `dbg25a/b-*` (doc 95/96 owner scan
set) · `stmfb8-*` (doc 93/94 symptom sample) · `87knob-min/sup`, `87flip`,
`87grp-*` (doc 87 §6.2/6.4; release condition still unmet) · `91neg-*`
(sentinel negative controls) · `d97prodchk` (doc 97 §9 production-default proof — it scores **zero** in a
name-exact citation census because the doc cites it as a path inside a table
cell, which is doc 91's own failure mode; `PROTECTED.txt` now says so, so the
next round cannot release it on a zero-citation reading) · `tfix388-r9`, `probe178410a` (`PROTECTED.txt`).

In `~/tmp`: `doc25gate` and `pinlib*` (live peer), `claude-*` (session
scratchpads), `d97b-libsnap` (the binary production ran under), `doc94c-libsnap`
(backs the kept `r2scan*` arms), `doc94r3b-libsnap` (backs the kept `dbg25a-*`),
`prod0901b-libsnap` (backs kept `em114*` / `stmfb8*` / `pr130r1-probe*`), and
`doc87/lib-{pre,tc,post,flip}`.

## 7. Reported, not fixed here

- The `pr/128` class B sentinel on mcp2k 105074 still asserts a PF node
  production no longer produces. This round *preserves the ability* to
  adjudicate it (`work-sent97-*`) but does not re-anchor it — that remains the
  open owner call from doc 97 §9.4.
- `sweep_tmp_20260901.sh` never ran, yet all eleven of its DROP dirs are gone.
  Something removed them outside the retire machinery; the machinery's record
  of what was released is therefore incomplete for 2026-09-01. Noted, not
  investigated.

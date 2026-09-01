# doc 89 — sbnd_xin cleanup: production rebased onto the pinned operating point, and the campaign layer retired (2026-09-01)

**Status: COMPLETE.** 150.1 GiB → **65.6 GiB**. 218 arms released, 98 kept. Production re-run at `ref/prod-2026-09-01b` over all 3067 events and
gated against the arm it replaces. The archive record layer re-encoded
gzip → zstd-19, verified member-for-member.

Owner scope, verbatim:

> *"the sbnd_xin directory is large now with a lots of work\* directory, it is
> time to do a clean up and retire some work\* directory. We have done this
> several times. We basically just need to keep the latest production results,
> note, we have been testing the results with minimal outputs. if this is the
> case, we should save the other outputs locally so that we have the full
> validation sets. Other than this we want to minimize the sbnd_xin directory.
> retire directory, archive results etc."*

and, mid-round: *"For processing, you can use 32 CPUs to speed up."*

## Repro

```bash
cd sbnd_xin
# --- Phase 0: pre-flight -------------------------------------------------
ls -la ../local/lib/libWireCell{Clus,Root,Gen}.so    # freshness vs newest source (M1)
../../toolkit/build/root/wcdoctest-root ; echo rc=$?  # 4035 assertions PASS
../../toolkit/build/clus/wcdoctest-clus ; echo rc=$?  # 2603 assertions PASS
cp -a ../../toolkit/cfg           ~/tmp/prod0901b-cfgsnap
cp -a ../../local/lib/*.so*       ~/tmp/prod0901b-libsnap/
python3 scripts/cfg/prod_cfg_gate.py --cfg ~/tmp/prod0901b-cfgsnap   # PASS 21/21
# --- Phase 1: production at the pinned point (3067 evts) -----------------
./scripts/doc89_prod0901b_arms.sh                    # ~32 min, rc=0 per sample
# --- Phase 2: the successor gate -----------------------------------------
python3 scripts/doc89_successor_gate.py --jobs 32    # 3067/3067 OK
# --- Phase 3: the retire round -------------------------------------------
#   EDIT scripts/retire/PROTECTED.txt BY HAND FIRST -- ASSERT 7 trips otherwise.
python3 scripts/retire/verify_group_dupes_20260901.py
python3 scripts/retire/plan_20260901.py              # 14 asserts, "OVERALL: PASS"
RETIRE_JOBS=32 python3 scripts/retire/archive_records_20260901.py
./scripts/retire/retire_20260901.sh A                # DRY RUN -- check dirs=/bytes=
RETIRE_REPLAN=1 python3 scripts/retire/plan_20260901.py   # re-stamp planned_at
CONFIRM=yes ./scripts/retire/retire_20260901.sh A
# --- Phase 4: beyond work* -----------------------------------------------
python3 scripts/retire/recompress_archive_20260901.py --apply --jobs 6 --min-mb 1.0
```

---

## 1. The premise was measured, and it does not hold

The request conditions the work on *"we have been testing the results with
minimal outputs. **if this is the case**, we should save the other outputs
locally"*. The condition is false, and establishing that changed the round.

Doc 87 shipped a family of output-suppression knobs earlier the same day, so
the concern is well-founded in principle. But **no production arm was ever run
with them.** Measured across every arm, not sampled:

| arm | pr_evt dirs | `mabc-pr.zip` | pctree | `tracking-pr.root` | `nusel-evt<N>.tsv` | calib |
|---|---|---|---|---|---|---|
| `work-mcp2k-prod0901` | 2000 | 2000 | 2000 | 2000 | 2000 | 905 |
| `work-mcp1k-prod0901` | 1000 | 1000 | 1000 | 1000 | 1000 | 461 |
| `work-nuecc48-prod0901` | 48 | 48 | 48 | 48 | 48 | 48 |
| `work-ncpi0-prod0901` | 19 | 19 | 19 | 19 | 19 | 19 |

Every `work-*-empre0901` arm matches. `scripts/pr142_arms.sh` sets
`PR_EXTRA_STAGES=pr_display`, which **adds** the calib dump — these are
*maximal*-output arms, not minimal ones. (The calib column is event-conditional,
not arm-conditional: `PrDisplayDump::visit` early-returns when there is no
neutrino candidate. Reading a single event dir gives the wrong answer, which is
how this nearly became a false finding.)

The only minimal-mode arms in the tree are doc 87's own
`work-87knob-{min,sup}-{ncpi0,nuecc48}` — 67 events, and they are **kept** (§4).

**So there were no "other outputs" to save: nothing had been dropped.**
`prod0901` is 30 % smaller than `prod0830` for an unrelated reason — it dropped
group-mode scaffolding (`.groups/g<N>.tar.gz`, 3.4 GB per 2k arm) while *adding*
per-event provenance (`.wct-cfg-evt<N>.json`, full `stdout.log`, `.time.meta`).
Reconciled to 0.4 %: 3596 MB of 0830-only scaffolding minus 646 MB of 0901-only
provenance = 2950 MB predicted, 2938 MB measured.

## 2. What *was* wrong, and the round it produced

`prod0901` finished at 05:00. The `save_in_scope` flip (toolkit `d52d818c`,
pinned in `ref/prod-2026-09-01b`) landed at 13:36. So the kept production set's
`tracking-pr.root` had **no `T_cluster` tree** — it was one flip behind the
operating point it was supposed to represent.

The owner chose to re-run rather than keep a stale baseline. That turned a
housekeeping round into a validation one, because `prod0901` and `prod0901b` do
not share a binary:

| lib | `pr142-libsnap` (prod0901) | doc 87 `lib-pre` | `lib-flip` | live / `prod0901b-libsnap` |
|---|---|---|---|---|
| `libWireCellClus.so` | `430ffa3e` | `8c083812` | `31b7e2ed` | `31b7e2ed` |
| `libWireCellRoot.so` | `b3347808` | `2f515d11` | `2b62da0d` | `2b62da0d` |
| `libWireCellGen.so`  | `e6568d94` | `e6568d94` | `355e4b1a` | **`5cf3f299`** |

`prod0901` ran at toolkit `ddce7430`; `prod0901b` at `d52d818c`. Between them
lie **five changes, each gated byte-identical but each only on the 308-event
manifest**: doc 77 r3 → doc 77 r4 → the master merge (§1.4, 482/482) → doc 87's
knobs at defaults (§6.1, 482/482) → the `save_in_scope` flip (§4.5).

**Phase 2 is therefore the first end-to-end check of that whole chain at full
3067-event scale.** It is the reason `prod0901` may be deleted at all.

## 3. A concurrent peer session, handled rather than assumed away

The shared toolkit tree was being edited *during* this round. This is recorded
because two of the three signals are invisible unless you look for them:

- `libWireCellGen.so` was rebuilt at 14:23, minutes before Phase 1.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet` and `wct-pr-perevt.jsonnet` were
  **modified and uncommitted** — the SBND production configs.
- `prod_cfg_gate.py` reported `DRIFT: uboone.json` on one run and `PASS 21/21`
  on the next, minutes apart, from the same `cfg` content. Cause:
  `qlport/uboone-mabc.jsonnet` was toggled between the two runs. The compiled
  `uboone.json` carried `PracticalBoxRecombination` during the first.

That peer owns **doc 88** (`qlport/scripts/sweep/doc88ub-box/`), the uBooNE half
of doc 87 §1.5's Birks/Box item — which is why this round is doc 89.

**Phase 1 was insulated rather than raced.** Both the binary
(`~/tmp/prod0901b-libsnap`, md5-verified equal to doc 87's `lib-flip` for Clus
and Root) and the cfg tree (`~/tmp/prod0901b-cfgsnap`, `PASS 21/21`) were
pinned, and `run_pr_chain_batch.sh`'s `PR_CFG_TREE` hook points the job at the
snapshot. **Every SBND artifact — `prod_prjob.json`, `sbnd_pr/clus/img/ql/simcheck.json`
— was byte-identical to the reference in *both* gate runs**; the single drift
was uBooNE's, and it is the peer's intended change, not ours.

> **A git-blame oddity, recorded so it does not confuse anyone.** This very
> document was swept into the peer's doc 90 commit `2e0f445b` at an intermediate
> state, because they staged broadly while it was still untracked — the shared
> index hazard again. Its final content is `f4561098`. `git log` on this file
> therefore shows a *doc 90* commit as its origin.
>
> The `df -h /nfs/data/1` line every prior round's driver printed was reporting
> the wrong filesystem: this tree lives on `/home/xqian` (`/dev/nvme2n1p6`), and
> `/nfs/data/1/xqian/toolkit-dev` is a symlink to it. Fixed in
> `retire_20260901.sh`.

## 4. What was kept, what was released

**98 arms kept (50.44 GiB), 218 released (86.13 GiB).** The removal set is
exactly the closed campaigns plus the two superseded production baselines;
nothing was released on age or size.

| block | arms | size | disposition |
|---|---|---|---|
| `work-*-prod0901b` **(new)** | 4 | 10.4 G | **KEEP** — production at the pinned operating point, full output |
| `work-*-grp0825` | 4 | 24.6 G | **KEEP** — the Q/L stage output, which doc 87 §2 classifies as the PR job's *only* input (`TensorFileSource`). Every prod/empre/87 arm, including `prod0901b`, was built from it |
| `work-vtx105-base-*` | 4 | 4.2 G | **KEEP** — the `pr90_movers --tags vtx105` metric epoch |
| doc 87 arms (`87scope-on`, `87tc-off`, `87mrg-post`, `87knob-def`, `87knob-min/sup`, `87flip`, `87grp-*`, `87cal*`) | 28 | 4.9 G | **KEEP** — see §5 |
| hand-scan display + probe layer (`em114*`, `pr117r1-onK1`, `pr130r1-probe*`) | 25 | 3.7 G | **KEEP** — PROTECTED; hardcoded display defaults and the pr/130 census source |
| `work-pr125r1-flipK5*` | 6 | 1.2 G | **KEEP** — the sentinel registry's worked FAILING case |
| `work-pr134-f086-*` | 4 | 1.2 G | **KEEP** — the 0.86 EM-scale production point |
| `work-sent130*` | 8 | 0.14 G | **KEEP** — the sentinel suite, incl. negative controls |
| sim samples + hubs, `work-probe178410a`, `work-tfix388-r9`, bare `work/` | ~12 | 0.4 G | **KEEP** — PROTECTED / non-reproducible / record dirs |
| pr136 / pr138 / pr139 / pr140 / pr141 / pr142 | 186 | **49.4 G** | **RELEASE** — all CLOSED docs |
| doc 77 r3 + r4 | 20 | **3.5 G** | **RELEASE** — both rounds executed and closed |
| `work-*-prod0830` | 4 | **15.1 G** | **RELEASE** — superseded twice over |
| `work-*-empre0901` | 4 | **10.2 G** | **RELEASE** — doc pr/142 COMPLETE, tables distilled to `products/empre0901/` |
| `work-*-prod0901` | 4 | **10.2 G** | **RELEASE** — on §6's measured evidence, and on nothing else |

## 5. The block the planner wanted and the evidence kept

The planner opened intending to release doc 87's four three-sample gate arms —
`work-87scope-on-*`, `work-87tc-off-*`, `work-87mrg-post-*`, `work-87knob-def-*`,
4.2 G — as "superseded by Phase 2's gate".

**That is wrong, and the reasoning is worth writing down because it is the same
shape as the 08-31 round's `prod0825` refusal.** Phase 2 compares `prod0901`
against `prod0901b`. Doc 87 §6.1 compares *knobs at their defaults* against
*the pre-knob arm*, and §6.2 compares *reduced modes* against the default arm.
These are different claims about different pairs. A gate that passes for one
says nothing about the other, and doc 87 shipped hours before this round —
those arms are the acceptance evidence for a knob that moved the production
operating point today.

`work-87knob-{min,sup}-*` are additionally the **only** minimal-output arms in
the tree; §6.2's table rests on them and nothing else can rebuild it.

4.2 G is not worth the ambiguity when 86 G is already coming out. The arms are
now in `PROTECTED.txt` with an explicit, written **release condition** — "when
doc 87 has a settled successor round that re-establishes §6.1 and §6.2 at
production scale, not merely when doc 87 is *closed*" — so the next round makes
a decision rather than an inference.

## 6. Phase 2 — the successor gate

`scripts/doc89_successor_gate.py`, one row per event in
`scripts/retire/state-20260901/successor-gate.tsv`. The claim is narrow and
cannot pass by accident: on every event, every product the two arms **share** is
identical, and the only difference is the added `T_cluster` tree.

| product | method | why that method |
|---|---|---|
| `mabc-pr.zip`, `pctree-pr-evt<N>.tar.gz` | member-content sha256 | tar/zip embed mtimes; `cmp` on the container reports a regression that does not exist (M2) |
| `tracking-pr.root` | every tree in **both** files, branch by branch, `equal_nan=True` | `T_rec_charge.reduced_chi2` carries NaNs and a naive `!=` flags every NaN row. Extra trees in B are reported, not failed — `T_cluster` is expected, anything *else* extra is a failure |
| `calib-pr-evt<N>.json` | JSON compare **excluding** `off_ms`/`on_ms`/`elapsed_ms` | `vertex_scoreboard.dual_chain.off_ms` is a wall-clock timer that makes two identical dumps read as DIFFER |
| `nusel-evt<N>.tsv` | raw bytes | a plain TSV with no embedded time |

An exception inside a worker is recorded as a FAIL, never swallowed as a skip —
the silent-`continue` trap doc pr/135 §11.2 closed for manifests, applied here.

Spot check before the full run, `ncpi0 evt105946`: `calib` 1036941 B both
sides, `mabc-pr.zip` 96454 B both, `nusel` 2285 B both, `tracking-pr.root`
322915 → 326985 B (**+4070 B, the added tree**), and `T_cluster` present only in
`prod0901b`.

## 7. Phase 4a — the archive record layer, re-encoded

`archive/records` held **16.08 GiB** in 4035 `.tar.gz`. Their members are
per-event logs and compiled configs — roughly a hundred near-identical copies of
the same text per arm. gzip's 32 KiB window cannot see across files; `zstd -19`
can. Measured on `prod0825-groupmode-20260825/other/work-pr112i-off-mcp2k.tar.gz`:

```
128 MB .gz  ->  751 MB raw  ->  8 MB .zst      (16x smaller than the gz)
round-trip: 14003 members, name/size/sha256 identical
```

This is a **re-encoding, not a deletion of scientific record (M13)**. Each
tarball is rewritten, then both sides are streamed and compared member-for-member
on `(name, size, sha256)`, and only then is the `.gz` removed. A tarball that
fails verification keeps its `.gz` and is reported. The ledger is
`scripts/retire/state-20260901/recompress.tsv`.

**Excluded, because they do not compress:**
`clean-slate-20260805/imaging-bases/work-mcp1000.tar.gz` (1.3 G) and
`work.tar.gz` (0.49 G) are already-compressed binary payloads — a 382 MB sample
went to 381 MB. `archive/records/labels/` is excluded too: it holds *verbatim*
copies, not tarballs, and is what ASSERT 2/6/6b compare against.

**A size floor of 1 MB**, added after measurement. The distribution is extreme:
3978 tarballs, median 0.64 MB, but the top 800 hold 85 % of the bytes and the
top 1600 hold 96 %. Converting the ~2400 sub-MB files costs most of the wall
time for under 5 % of the saving. They stay `.gz`; a mixed archive is fine
because both extensions decode with no special flags.

### Result

| | |
|---|---|
| tarballs re-encoded | **1672 / 1672 OK, 0 failures** |
| those tarballs | 12.87 GiB → **3.58 GiB** (3.6×, **9.29 GiB freed**) |
| `archive/records` overall | **16500 MB → 6657 MB** |
| file count | 2207 `.gz` + 1828 `.zst` = **4035**, the original count exactly |
| ledger | `state-20260901/recompress.tsv`, 1672 rows, 0 FAIL |

**The aggregate is 3.6×, not the 16× of the probe.** The probe file
(`work-pr112i-off-mcp2k`, 106 events of near-identical logs) is the best case,
not the typical one — a small tarball holding one event's logs gains only
~1.5×. The honest headline for this step is the measured **9.29 GiB freed**;
the 16× belongs only to the sentence describing that one file.

The ledger carries 1672 rows while 1828 `.tar.zst` exist. The 156 difference is
the killed first pass, whose completed conversions happened but were never
recorded — the defect that made the ledger incremental (below). 1672 + 156 =
1828, and 2207 + 1828 = 4035, so **every original tarball is accounted for**.

Nothing in either repo reads these tarballs programmatically (the only
reference is `pr126_pi0_select.py`'s error message, which names the directory),
so the extension change is safe. It is recorded anyway.

### Two defects in this script, both caught by its own verification

1. **`--long=31` made the archive unreadable by plain `zstd -d`.** The first run
   compressed with `zstd -19 --long=31`, which writes frames needing a 2 GB
   window to *decode*. Verification — which decodes — failed on every file:
   `Frame requires too much memory for decoding ... Use --long=31 or
   --memory=2048MB`. The verification did its job (every `.gz` was kept), but
   the flag was also a **footgun for any future reader** of this archive. It was
   removed rather than matched on the decode side; the measured 16× does not
   need it.
2. **The ledger was only written at the end**, so killing the first pass lost
   the record of 156 completed files. A record layer's ledger must survive the
   thing it is recording — it is now appended and flushed per file.

Cleanup after the first pass was reconciled by count, not assumed: 11 files had
both a `.gz` and an unverified `.zst` (workers killed mid-verify). The 11
unverified `.zst` were deleted and their `.gz` kept, giving 3879 + 156 = **4035**,
exactly the original count. **No archive byte was lost.**

## 8. What the guards caught that a hand-rolled `rm -rf` would not

Five refusals, each of which would otherwise have been silent damage or a
wasted round.

**1. The fork dropped the `KEEP_PREFIX` loop — 40 arms, 4.9 G, silently in the
removal set.** `plan_20260901.py` was spliced from `plan_20260831b.py` by
replacing a line range, and the range swallowed the six lines that fold
`KEEP_PREFIX` into `KEEP_WHY`. The trial planner run showed `[REMOVE] 258 dirs`
where 218 was expected, and **ASSERT 7 and ASSERT 11 both refused**: every
`work-87*`, every `work-sent130*` (including the sentinel registry's negative
controls) and all four `work-pr134-f086-*` had become removal candidates, and
two live pi0 manifests stopped resolving. This is the fork hazard doc pr/135
§11 recorded biting a third time, in a new place. *Always check the dry run's
`dirs=`/`bytes=` against the planner's own totals.*

**2. `em_labels` had drifted 298 → 540.** ASSERT 6b refused: the hand-scan label
corpus grew across the pr/136-142 rounds while the archive record copy stayed at
298 and git tracked 415 (`*.json` is gitignored, `.gitignore:2`, M9). Verified
additive before repairing — **0 files present in the archive but absent from
live, 0 whose content differs** — then copied and `git add -f`'d to 540/540/540
(commit `11a870ee`).

**3. A live process named a removal-set dir — via a substring.** ASSERT 12
refused on `work-mcp1k-prod0901` while the running job was writing
`work-mcp1k-prod0901**b**`. The interlock matches by substring, and
`prod0901` ⊂ `prod0901b`. Here it over-refused, which is the safe direction, and
it cleared when Phase 1 finished. Worth knowing before someone "fixes" it.

**4. `zstd --long=31` would have made the archive unreadable.** See §7 — caught
by the recompression's own member-for-member verification, not by inspection.

**5. The bare `work/` dir matches `d.startswith('work')`.** It holds
`work/nusel_labels/`, a hand-scan record (M13). ASSERT 2 would have caught it
via the label-archive check, but a record directory should be *named*, not
rescued by a safety net — it is now in `KEEP_WHY` and `PROTECTED.txt`.

**6. The group-duplicate verifier could not name a production arm's sample.**
`verify_group_dupes_*.py` derives an arm's sample from its name to find the Q/L
root it should be a duplicate of. It only handled the **suffix** convention
(`work-pr134-f086-mcp1k`), and every production arm uses the **middle**
convention (`work-mcp1k-prod0830`). `sample_of()` returned `None`, so
`QL_ROOT.get(None)` was not in KEEP and the verifier refused the round.

It never surfaced before because no earlier removal set contained a group-mode
*production* arm. This one does: `work-*-prod0830` carries **193
`.groups/g<N>.tar.gz`** bundles. Fixed by trying the suffix first (so no
previously-classified arm changes meaning) and then `work-<sample>-…`.

With that, the class is proven rather than assumed:

```
work-mcp1k-prod0830      63 tars   410546/410546 members byte-identical to work-mcp1k-grp0825
work-mcp2k-prod0830     125 tars   821110/821110 members byte-identical to work-mcp2k-grp0825
work-ncpi0-prod0830       2 tars     7802/7802   members byte-identical to work-ncpi0-grp0825
work-nuecc48-prod0830     3 tars    19718/19718  members byte-identical to work-nuecc48-grp0825
PASS -- 1259176/1259176 members across 193/193 archives
```

That proof is what lets `archive_records_20260901.py` **drop** the `groupin`
class instead of tarring ~5 GB of duplicated pctree data into the record layer.
Interlock 4 re-checks it by row count at deletion time.

## 9. Stated costs

Named plainly, because each is a thing that stops being re-derivable.

- **doc pr/142's before/after is no longer re-derivable from arms.** Releasing
  `work-*-empre0901` leaves the campaign comparison alive only as the distilled
  tables in `products/empre0901/*.tsv` (4 files) and the doc's own numbers. The
  doc is COMPLETE and its Proof C passed 478/478, so this is the same trade the
  08-31b round made for the pi0 A/B ends — but it is a trade, not a free action.
- **`prod0830` is gone**, so doc 85 r2 and doc 86's video-picks scripts
  (`scripts/analysis/d85r2_prod0830.py`, `d86_video_picks.py`,
  `scripts/bee/build_d86_bee.sh`) name an arm that no longer resolves. They are
  RECORD scripts — they record how a finished round was run — so they keep the
  name they actually read rather than being repointed at `prod0901b`, which
  would make them claim a provenance they did not have.
- **`prod0901` is gone.** Anything wanting the pre-`T_cluster` operating point
  must use `prod0901b` and ignore that tree; §6's gate is the evidence that this
  is safe on all 3067 events.
- **The pr/136-142 arms are gone**, so those docs' tables are no longer
  re-checkable against the dumps that produced them. Every one of those docs is
  CLOSED and carries its own gate labels.
- **~2400 archive tarballs under 1 MB stay `.gz`** (§7) — the archive is now
  mixed-format. Both extensions decode with no special flags.

## 10. Result

| | before | after |
|---|---|---|
| `sbnd_xin` | 153 725 MB (**150.1 GiB**) | 67 143 MB (**65.6 GiB**) |
| `work*` dirs | 306 | 101 |
| `archive/` | 16 500 MB | 6 848 MB |
| free on `/dev/nvme2n1p6` | 509 G | 596 G |

The `df` row is **whole-filesystem and not attributable to this round** — two
peer sessions (doc 88, doc 90) were writing the same disk throughout, adding at
least the nine `work-90pi0-*` arms and the `qlport/scripts/sweep/doc89ub-*`
trees. Only the `sbnd_xin` row and the per-step table below are this round's.

Where the space went, attributably:

| step | freed |
|---|---|
| retire round — 218 arms | **86.1 GiB** |
| archive re-encode — 1866 tarballs (1672 + 194) | **10.6 GiB** |
| new production arm `prod0901b` (added back) | −10.2 GiB |
| record layer for the 218 released arms (added back) | −0.2 GiB |

**Gates and labels, so any PASS here can be re-checked later**

| gate | result | evidence |
|---|---|---|
| `prod_cfg_gate.py --cfg ~/tmp/prod0901b-cfgsnap` | **PASS 21/21** | vs `ref/prod-2026-09-01b` |
| `wcdoctest-root` / `wcdoctest-clus` | 4035 / 2603 assertions, 0 failed | |
| Phase 1 arms | **3067/3067** events, every `rc=0`, `T_cluster` in all four | verified by count, not by the runner's word |
| successor gate | **PASS 3067/3067** | `state-20260901/successor-gate.tsv` |
| group-dupes | **PASS 1259176/1259176 members, 193/193 archives** | `state-20260901/group-dupes.tsv` |
| `plan_20260901.py` | **OVERALL: PASS**, 14 asserts | `state-20260901/plan.json` |
| `archive_records_20260901.py` | **integrity PASS 218/218**, 10.57 GiB raw → 1.51 GiB | `campaign-close-20260901/` |
| dry-run count check | planner 218/86.13 GiB **==** driver 218/86 GiB | the 08-31 catch, re-applied |
| deletion post-checks | 0 broken symlinks, 0 git-deleted tracked files, manifest 218 rows | `state-20260901/removed.tsv` |
| recompression | **1866/1866 OK, 0 failures** | `state-20260901/recompress.tsv` |

`work*` is 101 rather than 98 because a peer session created three more
`work-90pi0-r*` arms after the plan ran. That is the design: the driver iterates
the tier list, so an arm born after the plan is invisible to it.

## 11. Not done — needs the owner

**`~/tmp` sweep, 18 688 MB.** `scripts/retire/sweep_tmp_20260901.sh` is written,
vetted and its dry run is clean — all three interlocks pass now that the arms
are gone. The permission gate declined the `CONFIRM=yes` execution, exactly as
it declined the ad-hoc `rm -rf` in doc pr/135 §11.1. One command:

```bash
cd sbnd_xin && CONFIRM=yes ./scripts/retire/sweep_tmp_20260901.sh
```

It drops the ten `pin-pr13[89]*`/`pin-pr14[01]*` snapshots and
`prod0830-libsnap` — binaries whose arms this round deleted, so they now
reproduce nothing. **Interlock A refuses to run until those arms are actually
gone**, which is why it could not be run earlier.

The projected saving here was ~28 G; the measured figure is **18.7 G**, because
four of doc 87's five library snapshots turned out to be *distinct* binaries
(md5 on Clus/Root/Gen/Aux) backing arms this round keeps, not copies. Only
`lib-knob` was an exact duplicate of `lib-flip`, and it is already gone. The
smaller honest number is reported rather than the projection.

## 12. Recommended next step

**First, check whether `ref/prod-2026-09-01b` is still the production
reference.** It was pinned before doc 88 and doc 90 landed (toolkit
`50df4094`), and `prod_cfg_gate.py` with no `--ref` uses the *newest*
`ref/prod-*`. This round deliberately gated a **snapshot** of the cfg tree, not
the live one, so nothing here depends on the answer — but the next round should
re-run `prod_cfg_gate.py` against the live tree before trusting the reference.

(`scripts/cfg/fire_census.py`'s usage example was repointed to `prod0901b` in
this commit. Its arms are positional, so nothing was broken — but an example
naming a deleted arm is a trap. The doc 85/86 scripts in §9 keep their
`prod0830` names by the disposition rule: they *record* how a finished round was
run.)

After that, the natural next front is unchanged from doc pr/141: **PID, not
clustering** — ≥29 % of μ-typed objects are EM showers and ~1045 MeV of Enu is
missing. `prod0901b` is now the baseline any such round measures against, and it
is the first production arm whose `tracking-pr.root` carries the in-scope set
directly, so a PID census no longer has to go through the Bee zip to learn which
clusters were evaluated.

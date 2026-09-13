# 104 — cleanup round 2026-09-10: retire the intermediate arms, keep production and the hand scans

**Status: PARTLY EXECUTED (2026-09-10, late).** The owner ran sbnd and pdvd;
both completed. pdhd refused on an INTERLOCK A defect, now fixed (§12), and is
ready to re-run. The `~/tmp` tiers are staged.

**Original status, kept for the record: STAGED. Nothing in `work/` had been deleted.** Every step up to the
deletion has run and passed: the sentinel suite, both censuses, sixteen
interlocks, the frozen record layer, a stubbed run of the real confirm path and
a causal negative control. The three `CONFIRM=yes` commands are the owner's —
the permission gate declines them, as it has on every round since 09-01.

**One thing DID execute, because it deletes nothing:** the binary-pin dedup.
`~/tmp`'s 61 pin roots went from **59.92 GiB to 28.68 GiB**, 12553 files now
sharing an inode, every pin still complete and runnable, 0 files missing.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire
cd $D
python3 ../../../sbnd/sbnd_xin/scripts/pr127_sentinels.py --arms 'work-*-d102mpr'   # 21 PASS / 0 FAIL
python3 scan_arms_20260910.py --json=scan_arms_20260910.json    # which arms the hand scans read
python3 toks_20260910.py && python3 cit_20260910.py toks.txt cit_20260910.json
python3 plan_20260910.py                       # -> plan_20260910.out, tier1_*.txt, keep_*.txt
python3 dedup_pins_20260910.py                 # dry run; CONFIRM=yes to execute (deletes nothing)
python3 archive_records_20260910.py 1          # freeze the record layer  (DONE)
./sweep_tmp_20260910.sh 1                      # ~/tmp dry run
# the owner's steps:
CONFIRM=yes ./retire_20260910.sh 1 sbnd
CONFIRM=yes ./retire_20260910.sh 1 pdhd
CONFIRM=yes ./retire_20260910.sh 1 pdvd
CONFIRM=yes ./sweep_tmp_20260910.sh 1
```

## 1. The instruction, and why it changes the machinery

The owner's words: *"we want to keep the latest production, but for work*
directory that are intermediate results, we can retire them to save disk"*, and
then: *"We want to save the hand scan results for STM, which we still need them
for improvements. PDVD and PDHD"*.

The first sentence **changes the keep test**, and that is the whole round.
Every planner from 09-02 to 09-08 auto-kept any dir a doc cited
(`keep_dirs |= set(CITED)`). On 09-08 that rule made doc 103 §4 conclude that
pdvd and pdhd could release **nothing at all**: 454 pdvd arm families and 271
pdhd ones, every single one cited by the doc that made it. That is the right
answer to *"is this arm documented"* and the wrong answer to the question being
asked. A committed doc **is** the record of a closed A/B; the per-event bytes
beneath it are the intermediate result.

So `plan_20260910.py` **reports** the citation count and no longer obeys it. An
arm survives only if it is one of five things:

| keep category | how it is derived | not by |
|---|---|---|
| substrate | corrected symlink census, inbound > 0, transitive closure | a name that looks structural |
| latest production | the newest doc's own words (§3) | the `prod` substring |
| a hand-scan source | each scan prep set's own provenance key (§4) | a hand-written list |
| a live round | arm tokens in a record that is **not yet committed** (§5) | a typed prefix |
| an OPEN decision | doc pdvd/81 shipped OFF and handed the flip to the owner | the doc being closed |

## 2. What it comes to

| tree | keep | release | | |
|---|---|---|---|---|
| | dirs | GiB | dirs | GiB |
| pdvd `work/` | 5341 | 54.05 | 9355 | **77.31** |
| pdhd `work/` | 1323 | 42.51 | 3180 | **35.22** |
| sbnd_xin | 81 | 76.12 | 4 | 0.09 |
| **total** | | **172.68** | **12539** | **112.62** |

Plus `~/tmp`: dedup **−31.24 GiB already recovered**, and a sweep tier of 6
binary pins worth **3.41 GiB** more (§7). The sweep number is the union of
unique inodes, not `du` — after dedup those six pins *look* like 6.7 GiB and
are not.

`l1sp_wf_v9` (11 GiB, 889589 npz, no regeneration path found) is **still not
staged**; it has been put to the owner and declined twice, on 09-05 and 09-08,
and nothing this round changes that.

## 3. Latest production, from primary source

- **PDVD = `p79vprod`.** Doc pdvd/81 §6 states it outright: *"`p81vleg2`
  reproduces production (`p79vprod`) exactly"*. Read from the doc, not from the
  name.
- **PDHD = `p81hoff3`**, the knob-off half of the doc-81 gate, which is bare
  production config on the shipped binary.
- **sbnd = `work-*-d102m` + `work-*-d102mpr`** (doc 102, ref/prod-2026-09-08),
  unchanged from the 09-08 round.
- The sentinel suite was run against sbnd production **first, not last**:
  `pr127_sentinels.py --arms 'work-*-d102mpr'` reads **21 PASS / 0 FAIL / 2 OPEN
  / 7 INERT**. A cleanup round has surfaced a real silent regression this way
  before (doc 91).

## 4. The hand scans — and the arms that would have gone with them

Keeping the *labels* is not keeping a scan. A label says
`039252_7/98 is STM_MICHEL`; the only thing that turns it back into physics is
the dump the scanner was looking at, and a fresh arm at a different operating
point is never a substitute.

`scan_arms_20260910.py` reads each scan prep set's **own provenance key** and
each viewer's `ARM` binding, and resolves them against what is on disk:

| scan tag | source arm | dirs |
|---|---|---|
| pdvd `smx1`, `smx1a` | `d53v` (569 of 570 prep items) | 120 |
| pdvd `smx3` | `d68d4` (36 items) + `d67v` (13) | 120 + 120 |
| pdvd `smx4` — scanned **today**, labels written 09:19 | `d68a3` (54 items) | 120 |
| pdvd `d08pvflip0` | `d08pv30on` / `d08pv30off` | 30 + 30 |
| pdhd `smx1` | `d53h` (303 of 304 prep items) | 61 |
| pdhd `retile0` | `stm0` / `stmw` | 30 + 30 |
| pdhd `d08flip0` | `d08cap10` / `d08goff` | 30 + 30 |
| pdhd `movers0` | `d05mON` | 6 |

**Every one of those would have been released**, and neither existing guard
would have said a word:

1. a label JSON never names an arm, so INTERLOCK 9's manifest test
   (*"which arms does a live manifest resolve into"*) finds nothing; and
2. `cit_20260908.py`'s `ROOTS` list has `pdhd/stm_scan` and `pdhd/ql_scan` but
   **not `pdhd/stm_michel_scan`**, so the citation census never read the prep
   files either. `d53v`, `d53h`, `d67v`, `d68d4`, `d68a3` scored zero on both
   tests while being the entire evidentiary basis of the STM Michel scans.

Both gaps are closed: the roots are added, the derivation is positive, and
**INTERLOCK 13** refuses any round whose release intersects it. A hand-written
list of scan directories also missed `pdvd/d08_scan` and `pdhd/d05_scan`
entirely, so the script now **discovers** the scan dirs rather than enumerating
them.

## 5. Liveness is derived, not typed — and it caught a round mid-flight

`open_prefix` said `p79/p80/p81`. While this round was planning, **doc pdvd/82
was written at 21:13 and its arms at 21:26**, and a peer `claude --resume
d9d688aa` was alive. A hand-typed prefix list is stale the moment someone else
starts working.

So `live_tokens()` derives it: **an arm named by a record that is not yet
committed is live.** Untracked file → every arm token in it; modified file →
only tokens on *added* lines. That found doc 82 and pulled `p82vcs`, `p82vk10`,
`p82voff`, `p82vtp`, `p82vtp0`, `p82hoff` out of the release.

Then, between the first plan and the confirm, the same session created
`p82btp`, `p82boff`, `p82btp0` — **a sub-family that did not exist when the
derivation ran**. That is doc 100's lesson verbatim (seven families between plan
and confirm), and the fix is the same: `p82*` is now protected **by prefix** in
the planner and in both `PROTECTED.txt` files. The pdvd universe grew 14096 →
14696 during the round and all 600 new dirs went to KEEP.

### 5.1 The round read its own output and nearly halved itself

The first run of the live derivation reported **279 live tokens, 154 of them
resolving in pdvd**, and cut the release from 119.93 GiB to 11.40. The cause:
`tier1_*.txt`, `keep_*.txt` and `toks.txt` all live under `scripts/retire`, are
untracked, and name **every arm in the tree**. The planner was reading its own
tier files and concluding the whole tree was live.

This is doc 91's *"protected because protected"* and its 09-05/09-06
recurrences — `cit_*.py` carries `--exclude-dir=retire` for exactly this reason
and the new code did not. Fixed; the derivation now reports **8 tokens from 1
file**, which is doc 82 and nothing else.

## 6. The substrate moved, and the old census could not see it

`os.path.realpath()` follows the **whole** chain. A link into
`<evt>_keep/<file>` whose target is itself a link into
`/home/xqian/pdvd-frame-store` resolves *past* `keep` — and the intermediate hop
is exactly what is being priced. The first census scored `keep` at **0 inbound**
and would have released the tree's input.

Resolving the target **lexically** (readlink + normpath, root canonicalised once
for the `/nfs/data/1` ↔ `/home/xqian` alias) reproduces `PROTECTED.txt`'s
documented **968** for `keep`, and finds what has changed since 09-08:

| pdvd | inbound | | pdhd | inbound |
|---|---|---|---|---|
| `d51vclus` | **23728** | | `d51hclus` | **5132** |
| `d27fresh` | 4113 | | `d09` | 2010 |
| `d41prov` | 3368 | | `d09ctl2` | 950 |
| `d48nu7` | 1524 | | `stm0` | 876 |
| `keep` | 968 | | bare `029107_<N>` | 847 |

`d51vclus` and `d51hclus` are now the trees' largest hubs — every `p4*`/`p7*`/
`p8*` arm stages its clustering from them. They stay; the rest of their doc-50
families go. `/home/xqian/pdvd-frame-store` is the true input and is **outside
every tree this round touches**.

## 7. `~/tmp`: the dedup is the answer, again

Priced before dedup, by unique inode, because `du` over-reports a shared pin:

- **Executed:** 61 pin roots (the 09-06 glob saw 28 and missed 21 pins written
  *inside* round dirs — `p81/libpin_p81`, `d64/libpin`, …). 14177 files,
  **59.92 → 28.68 GiB**, 12553 linked, 0 skipped, 0 missing, 623 made read-only
  to close the in-place-overwrite hazard.
- **Staged:** 6 pure-`.so` pins, **3.41 GiB** union.

Fifteen candidate pins were **pulled back by the "a pin goes with its arms"
test**, and four of them look releasable by name: `d08_libpin` backs the
`d08cap10`/`d08goff`/`d08pv30*` **hand-scan** arms; `d09_libpin` backs pdhd
substrate; `d45/d41/d42/d44_libpin` back shipped-value evidence still named in
`PROTECTED.txt`; `d102m-libsnap` is the latest production binary; `d145/d146`
back `work-*-d145np` (pr/148 is **open**) and `work-*-d146sv25`. `d144_libpin*`
is kept for **ambiguity**: `work-*-d144fixprod` went in the 09-08 round, but the
sentinel suite's `work-s144pos/neg-*` layer survives and the `s144` token does
not contain `d144`, so the derivation cannot clear it. Ambiguity keeps a pin.

Four more pins — `d58_libpin`, `xtrack_libpin`, `pdhd02_libpin`,
`d97b-libsnap` — were refused by `pure_so()`: each carries 13448 non-`.so`
files, so they are **full** `local/lib` snapshots rather than the ~19-object pin
class. The round that just ended is why that distinction matters: doc pdvd/81
§5.3 records an old clus library mixed with a rebuilt root library spinning one
job for 85 minutes at 867 MB with no error line.

**Not swept, deliberately** — the four largest things in `~/tmp`:
`wt-premerge` (10.07 GiB) and `wt-merge` (7.72) are **registered git
worktrees**, both written 08:49 today; the correct verb is `git worktree
remove` and that is the owner's call on a merge that landed today (`98140fee`).
`claude-25225` (4.17) is live session scratchpads. `p82` (3.67) and `mg10`
(2.63) are live rounds.

## 8. The record layer is frozen, and it is real

`archive_records_20260910.py 1` wrote **12539 manifests and 12539 `.tar.zst`,
667 MB total**, one per released dir: a SHA-256 per dropped file, the symlinks
recorded rather than followed, and the per-event logs and compiled config
carried whole. Spot-checked on both trees — this was the first time the script
had been pointed at `<run6>_<idx>_<arm>` dirs rather than sbnd sibling dirs, so
an empty manifest would have been the one unrecoverable failure. One manifest is
legitimately empty (`028084_2_d30r2p` holds only symlinks; they are in its
`.links.txt`).

The record layer lives **on disk only** — `sbnd/sbnd_xin/archive/` is in
`.git/info/exclude`, as it was for the 09-06 and 09-08 rounds. It is 667 MB
against the 112.62 GiB it stands in for.

## 9. Interlocks

All sixteen PASS on all three trees. Four found something real:

| # | what it caught |
|---|---|
| 9 | `work-pr134-f086-*` — of the 420 arms `em_display`'s manifests name, this is one of the ~10 that still **exist**, so the display resolves into it. Pulled back. |
| 13 | **new.** The hand-scan sources (§4), re-derived rather than read from the config, so a typo cannot quietly drop a scan. |
| 14 | **new.** Citation no longer keeps an arm, which removed the only thing that made an *undocumented* family visible. So the burden flips: an uncited family must map to a **committed doc by round number**. 19 pdvd and 22 pdhd families do; all are one- and two-event probes. |
| 15/16 | **new.** Nothing released may be named by an uncommitted record (§5), and 16 prints what the derivation found so a PASS is readable rather than vacuous. |

`PROTECTED.txt` prefix entries are **now honoured as prefixes**. The old matcher
did `arm == line.strip("*")`, so a line reading `d51v*` protected exactly
nothing and the 09-08 round was relying on its inline `open_prefix` to do that
job. Fixed in the safe direction. The 09-08 blocks that were explicitly
temporary (*"protect by prefix while the round moves"*) are **moved to each
file's RETIRED section** with today's date and the owner's instruction as the
ground — which is what makes this round's release possible at all, and is a
deliberate edit rather than a silent expiry.

### 9.1 The stub and the negative control

Three bugs once sat behind five clean dry runs and 33 passing interlocks
(09-06), because the dry-run and confirm paths are *different code*. So the
driver was copied with its single `rm -rf` replaced by an `echo` — the
substitution is **asserted**, not assumed, and the first attempt at it silently
failed — and the **real** `CONFIRM=yes` path was run per tree: 4 / 9355 / 3180
dirs, all present, 0 already gone. The causal negative control points the
archive path at `NOSUCH-` and the driver **refuses with rc=6 before the first
deletion**, not after.

## 10. What the owner runs

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire
CONFIRM=yes ./retire_20260910.sh 1 sbnd     #    4 dirs,   0.09 GiB
CONFIRM=yes ./retire_20260910.sh 1 pdhd     # 3180 dirs,  35.22 GiB
CONFIRM=yes ./retire_20260910.sh 1 pdvd     # 9355 dirs,  77.31 GiB
CONFIRM=yes ./sweep_tmp_20260910.sh 1       #    6 pins,   3.41 GiB
```

`REPLAN=yes` is the default and **must not be overridden**: the driver re-runs
the planner at confirm time and refuses if any tier file moved, which is what a
live peer looks like. Doc 82 was still writing at 21:38; if it is still writing,
the guard will fire and that is it working.

Interlock 4 recorded the pre-existing broken-symlink count as **0 / 0 / 0**
before the round; the post-state number only means something against that.

## 11. Next

Run the four commands above. Then: `l1sp_wf_v9` (11 GiB) is still an open
question with no regeneration path, and `wt-merge` / `wt-premerge` (17.8 GiB)
are worktrees of a merge that landed today and can be `git worktree remove`d
once the doc-82 and mg10 rounds close.

## 12. Post-execution: what ran, and the guard bug that stopped pdhd

**sbnd and pdvd executed cleanly.** Verified after the fact: **0 broken symlinks
on all three trees**, against the pre-round baseline of 0/0/0 that interlock 4
recorded. Every protected name still resolves — the hand-scan sources
(`d53v` 120, `d67v` 120, `d68d4` 120, `d68a3` 120, `d08pv30on/off` 30+30),
production (`p79vprod` 120), the open doc-81 gate arms, the live `p82*` family,
and the whole substrate chain (`keep`, `d27fresh`, `d51vclus`, `d41prov`).
pdvd went **146 GiB → 81 GiB**.

**Then `CONFIRM=yes ./retire_20260910.sh 1 pdhd` refused:**

```
   CHANGED since plan time: tier1_pdvd_20260910.txt
   REFUSING: the tier files moved between plan and confirm
```

That was **not** a peer. It was INTERLOCK A firing on the round's own completed
work. The 09-08 version scoped the comparison to the **tier** being run, and its
own comment explains why: *"comparing every tier file means tier 1 having
executed makes tier 2 refuse — the round's own first pass raising a peer
alarm."* This round has **one tier and three trees**, so the same defect came
back on the other axis: running sbnd and pdvd emptied their tier files —
correctly, there is nothing left to release there — and the pdhd invocation
compared **all** tier files, saw pdvd go 9355 → 0, and refused.

**Fixed:** the comparison is now scoped to the tier **and** the trees this
invocation will touch. An already-executed tree is not a peer. Proven by running
the real `REPLAN=yes` guard path with the deletion stubbed:

```
== INTERLOCK A: re-planning before deleting (peer-session guard)
   OK: all interlocks still PASS; the tier file of every tree being
   run (pdhd) is unchanged.
   lines 3180 | present 3180 | already gone 0 | 35.22 GiB
```

**The generalisation worth keeping:** a confirm-time guard must be scoped to
exactly what the invocation will act on. Scope it to any wider set and the
round's own earlier passes look like a concurrent writer. It has now cost two
rounds, on two different axes.

## 13. Going further: where the remaining bytes are

Measured after the sbnd and pdvd releases, against the owner's narrower rule
(*"maintain the latest production, as well as keep the hand scan results"*).

| what | GiB | needs |
|---|---|---|
| **pdhd tier 1, already staged** | **35.22** | re-run, now unblocked (§12) |
| **`~/tmp` tier 2: the two merge build trees** | **14.5** | run `./sweep_tmp_20260910.sh 2` |
| pdvd old flip-evidence + `d41prov` | ~12 | a round 2 — see below |
| `pdhd/l1sp_wf_v9` | 11 | **owner decision**, no regeneration path, declined 09-05 and 09-08 |
| sbnd `work-*-d145np` | 9.8 | **owner decision** — is doc pr/148 still open? |
| `~/tmp` tier 1: pure pins of closed rounds | 3.41 | run `./sweep_tmp_20260910.sh 1` |
| `wt-merge`/`wt-premerge` `install/` | 3.1 | after the merge arms are finished |
| `~/tmp` tier 3: nested pins of dead rounds | 0 today | self-sequencing, grows after pdhd |

**What CANNOT go, and why the trees stay sizable.** sbnd_xin's 92 GiB is
**45 GiB of `work-*-d102m`** (latest production stage A) plus 10 of `d102mpr`
(stage B), 8.1 of the frozen record layer, 4.5 of `input_files_reco1` (the art
files stage A re-images from), and **3.9 GiB of `work-vtx105-base-*`, which 878
hand-scan label files reference 1724 times** — a hand scan, kept by the owner's
own instruction. pdvd's remaining 59 GiB is **20 GiB of pure substrate**
(`d27fresh` 8.41, `keep` 6.77, `d51vclus` 3.28, `d41prov` 1.61): the corrected
census still reads 7204 inbound links into `d51vclus` and 4110 into `d27fresh`
from the arms that survive, so it is the input, not history.

**The pdvd round 2, stated but not built.** The chain is
`keep ← d27fresh ← d41prov ← {d42fit, d43*, d44*, d45prod, d48nu3, d48nu7,
d143pnew, d41prod, d38qnewprod}` — old **flip evidence** for constants shipped
weeks ago, kept today only because `PROTECTED.txt` still names them. They are
neither latest production nor a hand scan. Releasing that set would also strand
`d41prov` (1.61 GiB), except that the `d08pv30on/off` **hand-scan** arms pin 92
links into it — a materialise step, cheap. `keep`, `d27fresh` and `d51vclus`
stay regardless, because the surviving `p79*`/`p81*`/`p82*` arms borrow from
them. Estimated ~12 GiB. It needs those `PROTECTED.txt` lines retired
deliberately, the way §9 retired the 09-08 ones, so it is the owner's call
rather than something to fold in silently.

## 14. What the owner runs now

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire
CONFIRM=yes ./retire_20260910.sh 1 pdhd     # 3180 dirs, 35.22 GiB  (unblocked)
CONFIRM=yes ./sweep_tmp_20260910.sh 1       #    6 pins,  3.41 GiB
CONFIRM=yes ./sweep_tmp_20260910.sh 2       #    2 build trees, 14.5 GiB
CONFIRM=yes ./sweep_tmp_20260910.sh 3       #    self-sequencing; run it AFTER pdhd
```

## 15. Post-state, read back on 2026-09-12 (doc 105)

Nobody wrote down which of §14's commands ran, so the next round read it off
the disk rather than assume:

- **pdhd tier 1 RAN.** `tier1_pdhd_20260910.txt` lists 3180 dirs and **0 of
  them exist**. The driver's own guard would now refuse it with "already gone"
  — the shape of pointing at a previous round's list — so it must not be re-run.
- **`~/tmp` tier 1 RAN.** All six pins (`d46_libpin`, `d30_libpin`,
  `d30_libpin_post`, `d30_libpin_r3`, `d30_libpin_r3g`, `d30r2_libpin`) are gone.
- **`~/tmp` tier 2 RAN.** `wt-merge/build` and `wt-premerge/build` are gone;
  both worktrees are still registered, with `install/` in place, as intended.
- **A record was overwritten by the confirm path.** The `1 pdhd` invocation's
  INTERLOCK A re-plan (2026-09-11 05:00) re-ran `plan_20260910.py` for all
  three trees, which rewrote every plan-time output (`tier1_*`, `keep_*`,
  `prebroken_*_20260910.txt` all carry mtime 09-11 05:00). Only one changed in
  content: the committed `keep_pdvd_20260910.txt` (+120 `p82vprod` lines, arms
  created after the plan); the prebroken baselines happened to read 0 both
  times, which is luck rather than protection. The committed keep file was restored
  from HEAD; the confirm-time version is kept beside it as
  `keep_pdvd_20260910.confirm-20260911.txt`. Doc 105 fixes the driver so a
  confirm-time re-plan writes only `*.confirm.txt`.

Every command in §10 and §14 is spent. The next round is doc 105.

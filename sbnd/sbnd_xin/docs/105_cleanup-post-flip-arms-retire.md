# 105 — cleanup round 2026-09-12: the post-flip STM/Michel arms, and four guards that were quietly wrong

**Status: EXECUTED 2026-09-12 on the owner's instruction ("Can you run these for
me? Push") — §12 is the post-state.** The original status is kept below.

**Original status: STAGED. Nothing in `work/` or `~/tmp` had been deleted.** Every step
up to the deletion has run: the sentinel suite, the scan-source census, the
citation census, the planner (all interlocks PASS on all three trees), the
frozen record layer, and the real `CONFIRM=yes` path under a stub and a causal
negative control (§9). The `CONFIRM=yes` commands in §10 are the owner's —
the permission gate declined even the pin dedup, which deletes nothing.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire
cd $D
python3 ../../../sbnd/sbnd_xin/scripts/pr127_sentinels.py --arms 'work-*-d102mpr'   # 21 PASS / 0 FAIL / 2 OPEN / 7 INERT
python3 scan_arms_20260912.py --json=scan_arms_20260912.json       # hand-scan sources, incl. docs/scan records
python3 toks_20260912.py && python3 cit_20260912.py toks_20260912.txt cit_20260912.json
python3 plan_20260912.py                          # -> plan_20260912.out, tier1_*_20260912.txt, keep_*_20260912.txt
python3 dedup_pins_20260912.py                    # dry run: 31.14 GiB recoverable, deletes nothing
python3 archive_records_20260912.py 1             # freeze the record layer  (DONE)
./sweep_tmp_20260912.sh 1                         # ~/tmp dry runs
./sweep_tmp_20260912.sh 2
```

## 1. The instruction, and what did not change

The owner's words: *"it is time to clean up a bit the pdhd, pdvd, sbnd_xin and
~/tmp directories to save some disk space. We have done this multiple times,
please do it like what we did before."*

So the keep test is **doc 104's, unchanged**. A committed doc is the record of
a closed A/B; the per-event bytes beneath it are the intermediate result. An arm
survives only if it is **substrate**, **latest production**, a **hand-scan
source**, part of a **live round**, or evidence for a decision still **OPEN**.

What made a new round necessary is volume: since doc 104, PDVD ran docs 82–96
and PDHD ran docs 18–27, about ninety new arm families at ~1 GiB each, and
every one of those rounds copied a whole ~4.75 GiB binary pin into `~/tmp`.

## 2. What it comes to

Before the round: pdvd `work/` 106.06 GiB, pdhd `work/` 85.90, sbnd_xin 91.74,
`~/tmp` 133 G by `du` (111 GiB unique). `/home/xqian` free: 273 G.

| tree | keep dirs | keep GiB | release dirs | release GiB |
|---|---|---|---|---|
| pdvd `work/` | 3659 | 45.67 | 7083 | **60.14** |
| pdhd `work/` | 1632 | 48.81 | 2326 | **33.52** |
| sbnd_xin | 81 | 76.12 | 0 | 0 |
| **work total** | | | **9409** | **93.66** |

| `~/tmp` | GiB | deletes |
|---|---|---|
| step 0: pin dedup (65 roots, 61.45 → 30.31) | **31.14** | nothing |
| tier 1: ten push worktrees, every HEAD on the remote | ~5.1 | `git worktree remove` |
| tier 2: pins of rounds with no surviving arm | self-sequencing (§7.3) | pure-`.so` dirs |

sbnd releases nothing, and that is the measurement: its 76 GiB is production
(`work-*-d102m` / `d102mpr`), the frozen record layer, the Reco1 input, the
`vtx105` hand-scan arms, the sentinel suite's negative-control layer, the open
pr/148 input `d145np` (9.85 GiB, §11), and the undocumented `mg10` family.

## 3. Latest production, from primary source

- **PDVD = `p96vprod`** — doc pdvd/96 §0: *"`p96vprod`, the flipped file with
  NO TLA"*, confirmed against its measurement arm by
  `d96_confirm.py --arm p96vscope --confirm p96vprod`. One claim, two arms;
  both kept.
- **PDHD = `h26q2dprod`** — doc pdhd/26 line 121: *"PDHD production =
  `h26q2dprod` (identical to `h26conf` on every verdict)"*; line 104 grades
  `h26q2d` additively against `h26conf`. Kept: `h26q2dprod`, `h26conf`
  (and `h26q2d`, by the live prefix, §5).
- **sbnd = `work-*-d102m` + `work-*-d102mpr`**, unchanged since 09-08. The
  sentinel suite was run against it **first**: 21 PASS / 0 FAIL / 2 OPEN /
  7 INERT, identical to doc 104.
- **`p79vprod` is superseded.** It was 09-10's PDVD production and was named
  in `PROTECTED.txt` as such; fifteen PDVD flips landed after it. Its line is
  moved to the RETIRED section with the date and ground, deliberately, as doc
  104 §9 did for the 09-08 blocks. The same for `p82*` (doc pdvd/82 committed
  2026-09-10 22:16).
- **OPEN, kept:** doc pdvd/81's `p81voff3`/`p81vq2d3`/`p81vleg2` and the PDHD
  half `p81hoff3`/`p81hq2d3`/`p81hleg2` — doc pdhd/26 (`f793e557`) still
  names the two q2d knobs as set by neither detector.

## 4. The hand scans — half of the new ones have no prep set

`scan_arms_20260912.py` re-derived the sources. New since 09-10:

| scan | source arms | how it is known |
|---|---|---|
| pdvd smx5..smx9 | `p85vprod`, `p85vwh`, `p88vprod`, `p88v5fr`, `p90vprod` | prep-pdvd-smx5..9 per-item `arm`; `docs/scan/pdvd_stm_michel_smx8_key.tsv` |
| pdhd smx18..smx27, own19..own26 | `h18s` (+ `d53h`, already kept) | prep-pdhd-smx18/19 `--arm h18s`; every later record re-uses prep-pdhd-smx19 |
| pdhd smx18 double scan | `h18b`, `p82bhoff`, `p82hoff` | `docs/scan/smx18/pdhd_stm_michel_scan_key_{h18b,p82bhoff}.tsv`, provenance |
| pdhd own23 | `h23conf` | `smx23/owner_rulings_own23.json`, prep-pdhd-h23conf |
| pdhd smx25..27, own25/own26 | `h25base`, `h25k`, `h25kr`, `h25r` (+ `h25va`, `h25vb` over-emitted) | prep-pdhd-h25base; the movers put in front of the blind re-judge, `smx25/key_smx25.tsv`, `items_a*` |

**smx20..smx27 have no prep directory at all.** They re-use prep-pdhd-smx19
and add items from other arms, so a prep-only harvest sees `h18s` and misses
`h25k`/`h25kr`/`h25r` — the arms smx25 and smx26 actually put in front of the
scanners. That is doc 104 §4's blind spot, one round later. The census now also
reads `<det>/docs/scan`.

**And the first fix over-reached, measurably.** A free token scan of *all* of
`docs/scan` kept **44 pdhd families**: the `census_*`, `g_*`, `gates_*`,
`score_*`, `q*_*`, `cfg_proofs` and `preregistered_*` files tabulate every arm a
round *compared*. That is citation by another name — the keep test the owner
retired on 09-10. Restricted to scanner-facing files (keys, sheets, owner
queues and rulings, `items_a*`, provenance, verdicts, records, movers), and
then tightened twice more on evidence: a bare `flip` matched
`h22/p1_sizing_postflip.txt` and a bare `items` matched `trace_fp_items.txt`,
which between them kept six closed h21/h22 families. Final: 11 pdvd and 16 pdhd
source arms. `PROTECTED.txt` names the new ones in a 2026-09-12 block.

## 5. Liveness: a peer round, a new letter, and the round's own doc

- **A live peer.** The session that committed doc pdhd/26 at 18:24 was still
  running while this round planned — `~/tmp/h27` written 18:24, `h27cfg*` work
  dirs, its transcript written 18:31 — and it committed doc pdhd/27 during the
  round (`a8859300`, `cf998593`). `live_tokens()` cannot see `~/tmp`, and doc 26
  was already committed, so `h26*`/`h27*` (and the not-yet-started `p97*`) are
  protected **by prefix** in the planner and both `PROTECTED.txt` files, before
  the first plan run.
- **And it announced the next one mid-round.** At ~19:30 the same peer sent a
  message: a new live round, work tags `h28off`/`h28wl`/`h28prod` (pdhd) and
  `p97voff`/`p97vwl` (pdvd), pin `~/tmp/h28/libpin_h28`, uncommitted record
  `pdhd/docs/scan/d28/`, with `h26q2dprod` and `p96vprod` as its gate
  baselines. `p97*` was already a prefix; `h28*` is added to the planner and
  both `PROTECTED.txt` files, and `~/tmp/h26..h28`, `p97`, `mg10` are excluded
  **by name** from both `~/tmp` pin tools (`LIVE_TMP`) — the dedup's
  hardlink-and-chmod is exactly the hazard for a pin still being written. The
  planner was re-run after that edit and the tier files came out
  **md5-identical** (none of those arms existed yet, which is what a prefix
  protection is for).
- **The round grammar grew an `h`.** PDHD rounds since doc pdhd/18 name their
  arms `h18b`, `h22conf`, `h25kr`. Every arm-token regex in the machinery said
  `(?:p|d)`: `live_tokens()` could not see an h-round as live (fail-open, the
  dangerous half) and INTERLOCK 14 could not map `h22conf` to doc pdhd/22
  (fail-closed). Both accept `h` now.
- **The regex also needed two characters after the round letter** —
  `[a-z][a-z0-9]{1,14}` — so it could never match `d53v`, `d53h` or `d67v`,
  the smx1/smx3 scan sources. In the pin tier (§7.3) that released
  `d53/libpin` while 14 files named it. `{0,14}` now, in both places.
- **The round read its own doc.** On the second plan run `p82vprod` was kept
  as "live", and the only uncommitted record naming it was this round's edit to
  doc 104 §15. A cleanup doc names arms in order to *release* them; reading it
  as liveness is doc 91's "protected because protected" on the doc layer.
  `live_tokens()` now skips `*cleanup*`/`*retire*` files, as it already skipped
  `scripts/retire/`. After the fix: 0 live tokens, `p82vprod` released.

## 6. The confirm path overwrote a committed record

Doc 104's §14 commands had all been run by the time this round started (doc 104
§15 now says so; nobody had written it down). Reading back how they ran found a
defect in the 09-10 driver: `CONFIRM=yes ./retire_20260910.sh 1 pdhd`'s
INTERLOCK A re-ran `plan_20260910.py` **with no arguments**, and the planner
writes `tier1_*`, `keep_*` and `prebroken_*` for every tree. On 2026-09-11
05:00 that rewrote all nine plan-time files; one changed in content — the
committed `keep_pdvd_20260910.txt`, +120 `p82vprod` lines. The committed file
is restored from HEAD; the confirm-time version is kept beside it as
`keep_pdvd_20260910.confirm-20260911.txt`.

The guard was right; its side effect was an M13 overwrite. Fixed in
`retire_20260912.sh`: the re-plan runs with `PLAN_SUFFIX=confirm`, so it
writes only `*.confirm.txt` and compares `tier1_<t>_20260912.confirm.txt`
against the plan-time file, and it is passed `$TREES` — the same variable the
deletion loop iterates — so it cannot compute a tree the invocation does not
act on. `toks.json`/`toks.txt` had the same shape (an unstamped name every
round rewrites, and they are committed); they are `toks_20260912.*` now.

## 7. `~/tmp`

### 7.1 Step 0: the dedup is the answer, again

65 pin roots, 19729 files, 6540 inodes, **61.45 GiB → 30.31 GiB**. The p83..p96
rounds each copied a whole pin (cuDNN included, 544 MB per `libcudnn_cnn_infer`
copy) and none had been deduplicated. Byte-compare before linking, atomic
`os.replace`, read-only afterwards — `dedup_pins_20260910.py` unchanged but for
the stamp. It deletes nothing, and the permission gate still declined
`CONFIRM=yes`, so it is the owner's first command.

### 7.2 Tier 1: push worktrees whose commit is on the remote

Ten registered worktrees of wcp-porting-img: `~/tmp/push22, 22c, 22d, 23, 24,
25, 26` and a `pushwt` in each of three **dead** sessions' scratchpads
(transcripts last written 09-10 06:54, 09-12 06:33, 09-12 14:10; the live
sessions are this one and the peer). Every HEAD is an ancestor of the remote
`main` tip `ff8f3ee4`, read by `git ls-remote` at run time because the local
`origin/main` ref is stale; every tree is clean and idle 686–3595 min. The verb
is `git worktree remove`, never `rm -rf`, so git itself refuses a dirty tree.

### 7.3 Tier 2: pins of dead rounds, and a survivor test that could never fire

Carried from doc 104 tier 3 (value-first: keep a pin if any file naming it also
names an arm on disk). Its first dry run held **all 30 nested pins on the same
two survivors, `d16hnu` and `d16vnu`** — not a stray doc: the rounds' own
runners (`d83_arms.sh`, `d96_arms.sh`, `h25/run_arm.sh`) name them as the
substrate the arms are staged FROM. An input read as an output keeps every pin
of every round that staged from kept substrate, which is all of them. Substrate
names (read from the planner's own `substrate=[...]` lists) are now dropped
from the survivors unless they carry the pin's own round number.

Today, before the work release, it frees one pure pin, `d51/libpin` — checked
value-first: doc pdvd/51 lines 470/641 and `~/tmp/d51/run_all3.sh` tie it to
`d51gv`/`d51gh`/`d51gvleg`/`d51ghleg`, none on disk; `d51vclus` (09-07 15:19)
predates the pin (09-08 16:59). Six more are FREE but refused by the purity
filter as full `local/lib` snapshots. After the work release the `p75`..`p95`
pins lose their survivors, so run it last.

**Not swept, each an owner call:** `wt-merge`/`wt-premerge` install trees
(~3.5 GiB; `mg10`'s `env_{A,B}.sh` put them on `LD_LIBRARY_PATH` and mg10 still
has no doc), `mg10` (2.63), session scratchpads (4.66), round dirs' non-`.so`
content (prep sets and gate logs are records), and the top-level pins that back
kept arms.

## 8. The record layer

`archive_records_20260912.py 1` wrote **9409 manifests, 9409 `.links.txt` and
9409 `.tar.zst`, 563 MB total**, one per released dir (7083 pdvd, 2326 pdhd):
a SHA-256 per dropped file, symlinks recorded rather than followed, the
per-event logs and one copy of the compiled config carried whole.

**45 manifests are empty, and each was checked rather than accepted:** all 45
are `p81vq2d` dirs holding **only symlinks**, and for 45/45 the number of links
on disk equals the line count of the dir's `.links.txt`. Spot-checked the other
way on pdhd (`029107_11_p92hoff`, `028084_12_h23a`, `029107_20_h22base`):
manifest rows = files on disk, 7 = 7, with 3 non-heavy members carried in each
`.tar.zst`.

It lives on disk only — `sbnd/sbnd_xin/archive/` is in `.git/info/exclude`, as
for the 09-06..09-10 rounds.

## 9. Interlocks, the stub and the negative control

All interlocks PASS on all three trees (1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15,
16). Pre-existing broken symlinks recorded before the round: **0 / 0 / 0**.
INTERLOCK 13 names the 11 + 16 scan sources of §4; INTERLOCK 16 reads 0 live
tokens after §5's fix, with `h26*`/`h27*`/`p97*`/`mg10*` held by prefix.

The driver was copied with its single `rm` replaced by an echo — the
substitution **asserted** (no live `rm` line left), not assumed — and the
**real** `CONFIRM=yes` path was run per tree, INTERLOCK A re-plan included:

| tree | rc | INTERLOCK A | present / already gone | stub |
|---|---|---|---|---|
| pdvd | 0 | OK, tier file unchanged | 7083 / 0 | would delete 7083 |
| pdhd | 0 | OK, tier file unchanged | 2326 / 0 | would delete 2326 |
| sbnd | 0 | OK | empty tier — no-op | — |

After the three confirm re-plans the plan-time `tier1_*`, `keep_*` and
`prebroken_*_20260912.txt` files are **md5-identical** to before (§6's fix,
proven on the real path), the `*.confirm.txt` files exist beside them, 0 targets
are missing, and broken symlinks still read 0 / 0 / 0.

**Causal negative control:** the same stub with the archive path pointed at
`NOSUCH-cleanup-20260912` **refuses with rc=6 before the first deletion**, on
pdhd, after INTERLOCK A passed — plan-time files md5-identical, 2326/2326
targets present.

## 10. What the owner runs

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire
CONFIRM=yes python3 dedup_pins_20260912.py      # 31.14 GiB, deletes nothing -- first
CONFIRM=yes ./retire_20260912.sh 1 pdvd         # 7083 dirs, 60.14 GiB
CONFIRM=yes ./retire_20260912.sh 1 pdhd         # 2326 dirs, 33.52 GiB
CONFIRM=yes ./sweep_tmp_20260912.sh 1           # 10 push worktrees, ~5.1 GiB
CONFIRM=yes ./sweep_tmp_20260912.sh 2           # dead-round pins; run LAST
```

`REPLAN=yes` is the default and must not be overridden. If the peer PDHD round
creates an arm family outside `h26*`/`h27*` between plan and confirm, INTERLOCK
A refuses — that is it working. (sbnd has no tier-1 lines; its invocation is a
no-op.)

## 11. Open for the owner

| what | GiB | why it is not staged |
|---|---|---|
| `pdhd/l1sp_wf_v9` | 11 | no regeneration path found; declined 09-05, 09-08, 09-10 |
| sbnd `work-*-d145np` | 9.85 | the named input of pr/148, which is still open |
| pdvd old flip evidence (`d42fit`, `d43*`, `d44*`, `d45prod`, `d48nu3`, `d41prod`, `d38qnewprod`) | ~4.6 | still named in `PROTECTED.txt`; doc 104 §13's "round 2", needs those lines retired deliberately |
| `mg10*` (all three trees) + `wt-merge`/`wt-premerge` + `~/tmp/mg10` | ~8 | the master-merge validation round has no committed doc |
| `h26*`/`h27*`/`h28*`/`p97*` arms, `~/tmp/h28` | ~2+ | live today (§5); next round |

## 12. Post-execution: what ran, and what it left

The owner asked this session to run §10 and push. The permission gate that had
declined `CONFIRM=yes` all round allowed it once the owner asked directly.
Every command ran in §10's order, and the `work/` releases each went through
their own INTERLOCK A re-plan (`OK: all interlocks still PASS` in both
`retire_confirm_{pdvd,pdhd}_20260912.log`).

| step | result |
|---|---|
| dedup | 5663 files linked, 0 skipped; **19729 / 19729 files present, 0 missing**; pins 61.45 → 30.59 GiB on disk; free 266 → 296 G |
| `retire 1 pdvd` | 7083 / 7083 deleted, rc=0; `pdvd/work` 106 → **69 G** |
| `retire 1 pdhd` | 2326 / 2326 deleted, rc=0; `pdhd/work` 86 → **66 G**; free → **389 G** |
| `sweep_tmp 1` | 10 / 10 worktrees removed by `git worktree remove`, rc=0; each HEAD re-verified against remote main `a7b378a5` |
| `sweep_tmp 2` | 97 arm families alive after the release, 33 nested pins: 23 held value-first (e.g. `d53/libpin` by `d53h`/`d53v`, `p75`/`p80`/`p81` pins by the OPEN doc-81 arms, `p96/libpin_p96` by the `h25*` scan sources), 10 FREE; 8 of those refused by the purity filter as full `local/lib` snapshots (kept); **2 removed: `d51/libpin`, `p80/libpin_p80full`**, rc=0. `~/tmp` 133 → **99 G** (`du`); free → **394 G** |

**Nothing protected was touched.** Broken symlinks: **0 / 0 / 0**, equal to
the pre-round baseline. Every production, substrate, OPEN and scan-source name
still resolves at its full count. For pdvd: `p96vprod`/`p96vscope` 120, `keep`
240, `d27fresh`/`d51vclus` 120, `d41prov` 99, `d53v`/`d67v`/`d68a3`/`d68d4`/
`p85vprod`/`p85vwh`/`p88vprod`/`p88v5fr`/`p90vprod` 120, `d08pv30on/off` 30,
`p81voff3`/`p81vq2d3`/`p81vleg2` 120. For pdhd: `h26q2dprod`/`h26conf`/`h26q2d`
61, `d51hclus`/`d09` 61, `stm0` 30, `d53h`/`h18s`/`h18b`/`h23conf`/`h25base`/
`h25k`/`h25kr`/`h25r`/`h25va`/`h25vb`/`p82bhoff`/`p82hoff` 61, `d08cap10` 30,
`d05mON` 6, `p81hoff3`/`p81hq2d3` 61.

**The prefix protection earned itself.** The peer's round announced in §5
created **431 dirs at 19:31, after the plan**: `h28off`, `h28prod` and `h28wl`
(61 each) plus 8 `h27cfg*`/`h28cfg*` on pdhd, and `p97voff`/`p97vwl` (120 each)
on pdvd. None was in a tier file, and all 431 survive. Because `h28*`/`p97*`
were prefixes, INTERLOCK A's re-plan put them in KEEP and the tier files did
not move. Without the prefix, the new dirs would have entered the confirm-time
tier and the driver would have refused. That is the safe failure, but it would
have blocked the round rather than protecting it.

**Pushed:** `059c8e3c` was cherry-picked onto remote main `7061f57a` in a scratch
worktree and pushed as **`a7b378a5`** over https. `git ls-remote` then read
`refs/heads/main` = `a7b378a5`. Local `main` stays diverged, as in every round.

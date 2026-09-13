# 106 — cleanup round C 2026-09-13: retire the intermediate results after the h28 production flip

**Status: EXECUTED 2026-09-13 on the owner's instruction ("please run it"). §13 is
the post-state.** The staged status follows.

**Original status: STAGED. Nothing in `work/` or `~/tmp` had been deleted.** Every step
up to deletion has run:
- the planner, with every interlock PASS on all three trees;
- the frozen record layer for all 1375 released dirs;
- the `~/tmp` census, with set-relative byte accounting;
- the sbnd sentinel suite;
- all four `CONFIRM=yes` paths, each under a stub with a causal negative control (§9).

The commands in §10 are the owner's to run.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire
cd $D
python3 scan_arms_20260913.py --json=scan_arms_20260913.json     # unchanged from 20260912b
python3 toks_20260913.py && python3 cit_20260913.py toks_20260913.txt cit_20260913.json   # 4908 tokens -> 274 cited
python3 plan_20260913.py              # -> plan_20260913.out: 19.02 GiB, interlock failures NONE
python3 archive_records_20260913.py 1 # record layer: 1375/1375 manifests, 224 MB  (DONE)
python3 tmp_census_20260913.py        # -> tmp_census_20260913.out: 362 units, 45.07 GiB freed set-relative
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # 21 PASS / 0 FAIL / 2 OPEN / 7 INERT
./retire_20260913.sh 1                # dry run
./sweep_tmp_20260913.sh               # dry run
```

## 1. The instruction

The owner, 2026-09-13: *"I wonder if there are rooms to further improve? Note, we
can retire the intermediate results now. Four directories ~/tmp, sbnd_xin, pdvd,
pdhd, please. We can clean up a bit the work\* directories."*

The keep test is doc 104's and doc 105's. An arm survives only if it is one of:
- **substrate**;
- **latest production**;
- a **hand-scan source**;
- evidence for a decision still **OPEN**.

"Now" is the new part. The h26–h28/p97 peer round that rounds A and B held by
prefix is committed and pushed (doc pdhd/28 `5f53a58b`, doc 26 §3.4 `1a3c7d87`).
The session that ran it confirmed on 2026-09-13 that it writes nothing. So that
round's intermediates come under the test.

## 2. What it comes to

Before: pdvd `work/` 64 G, pdhd `work/` 65 G, sbnd_xin 81 G, `~/tmp` 91 G (by `du`).
`/home/xqian` free: 421 G.

| tree | keep dirs | keep GiB | release dirs | release GiB |
|---|---|---|---|---|
| pdvd `work/` | 1983 | 34.55 | 854 | **8.21** |
| pdhd `work/` | 1299 | 42.13 | 496 | **8.31** |
| sbnd_xin | 46 | 61.34 | 25 | **2.50** |
| **work total** | | | **1375** | **19.02** |

| `~/tmp` unit class | KEEP | FREE |
|---|---|---|
| pins (a dir holding `libWireCellClus.so`) | 29 | 37 |
| prep dirs (`<round>/prep_<arm>`) | 14 | 106 |
| idle session scratchpads | 4 | 40 |
| scratch (`~/tmp/h28` minus its pin; harness leftovers > 7 d) | 5 | 179 |
| **362 units: nominal 79.46 GiB, FREED 45.07 GiB** (3.49 GiB shared with a kept file frees nothing) | | |

Expected after both: free ≈ 421 + 19 + 45 ≈ **485 G**, and `~/tmp` ≈ 46 G.

## 3. Production, from primary source

- **PDHD moved.** Doc pdhd/28 line 12: *"PDHD's is now `h28prod` (pin `libpin_h28`, Clus md5 30113227)"*.
  - Gate F: `h28prod` == `h28wl`, bit-identical. So production = `[h28prod, h28wl]`, one claim with two arms, the same pattern as `p96vprod`/`p96vscope`.
  - This was set **before** the h28 prefix was dropped. Dropping the prefix first would have taken production out of every keep category at once.
- **`h28off` is kept.** It is the knob-off baseline on the production pin.
  - G1 (`docs/scan/d28/gate_h28off.txt`) shows `h28off` == `h26q2dprod` on every branch, point row, tree (incl. `T_stm_michel_2d`), zip member, calib json and census.
  - With it kept, the OFF side of the doc 28 flip stays checkable (doc 98). Its bit-identical twin `h26q2dprod` is therefore released, and so is `h26q2d`.
  - The session that wrote doc 28 asked for exactly this.
- **`h26conf` is released.** It is the pre-q2d production baseline, which differs from `h26q2dprod` in 49 branches (doc 28 line 68). It is superseded production, the same precedent as `p79vprod` in round A.
- **PDVD** stays `p96vprod`/`p96vscope`. Doc 28's G2 gates `p97voff` against it and keeps PDVD off the wire-lookup key. `p97voff` (== `p96vprod`) and `p97vwl` (0 verdict changes) are released.
- **sbnd** stays `work-*-d102m` / `d102mpr` (doc 102).
- **Pins.** `~/tmp/h28/libpin_h28` and `~/tmp/p96/libpin_p96` become **permanent production pins** with their own PROTECTED lines. They are no longer "live round" exclusions.

## 4. Doc 81's OPEN decision is closed

PROTECTED held `p81voff3`/`p81vq2d3`/`p81vleg2` and `p81h*` as *"an OPEN decision,
not a closed one"* (doc pdvd/81 §8–9). Both halves have since been decided:
- **PDVD.** Doc pdvd/95 answered §9 item 6 and flipped the region definition into production.
- **PDHD.** Doc pdhd/26 lines 114–115: *"Doc pdvd/81 §8a held PDHD OFF because its Michels lean more on cross-shared cells; the owner took the flip with that known"* (`8ae3bfba`).
- `pdhd/wct-pr-perevt.jsonnet` now carries `michel_q2d: true`, `michel_q2d_cells: true` and the region keys.

The six p81 arms (pdvd 3.13 + pdhd 2.68 = 5.81 GiB) are released.

## 5. Substrate, re-derived by a lexical first-hop census

The 09-10 substrate lists were carried forward unchanged into rounds A and B. This
round recounts inbound links, reading each link's target **lexically** (one hop),
as the PROTECTED census note requires:

| tree | inbound (links) | zero inbound → released |
|---|---|---|
| pdvd | d27fresh 4110, d51vclus 4080, keep 968, d41prov 689, d39r2prov 157, d28dlfp 9, d42fit 4, d143pnew 2 | d48nu7, d143pnew (+ d08cfgA, its only borrower, and d48flipcfg, which chains through d08cfgA), d28dlfp, d34base, d31r6e2e, d11vtrace (+ the 2 d42fit dirs only it held) |
| pdhd | d51hclus 2686, d09 1116, (bare) 483, d09ctl2 270, stm0 244, d09ctl 240 | d11prod, d11sepoffA, d11seponA, stmwc, d02prod |
| sbnd | work-dbg25a-ql ← work-dbg25a-d97prodchk (20); nothing else crosses a sibling dir | work-dbg25a-ql goes with its only borrower |

**Kept although zero-inbound: `d16vnu` and `d16hnu`.** No link points at them,
because every p9x/h2x arm links through to d51vclus/d51hclus. But they are the PR
runners' `SRC` tag (`d95_arms.sh:54`, `d96_arms.sh:64`, `~/tmp/h28/run_arm28.sh:9`).
Production cannot be re-run without them. An inbound count alone would have
released them; this is the doc 98 shape again.

`d41prov` and `d39r2prov` stay because the `d08pv30on`/`off` hand-scan source borrows from them.

**`stmwc` (pdhd, 0.29 GiB) needs a sentence, because it was protected text
yesterday.** Its only protection was a substrate listing on PDHD PROTECTED line 52,
dating from an earlier census that counted 62 inbound links. Today it has 0.
- **Not in the blind claim.** `stmw` stays: it is half of the stm0/stmw byte-identity claim (stm-tagger-chain §13.1) that the scan's blind rests on. `stmwc` is not named in that claim.
- **Not in the "Not committed" list.** It is also absent from that doc's list, which holds the `stmc*`, `phdump*` and `wcc*` probes.
- **So it is released.** Its record-layer manifest is in `cleanup-20260913/pdhd-tier1`.

## 6. sbnd: superseded evidence

| released | why |
|---|---|
| `work-*-d97prodchk`, `work-dbg25a-d97prodchk` | doc 97 §9's check of **prod-2026-09-04** |
| `work-ncpi0-d99r3prod{,pr}` | the prod-2026-09-05 flip arm |
| `work-*-d145prod` | pr/145's shipped arm, gated against `d144fixprod` (itself gone); production is d102m since |
| `work-d147-{tailflip,flipchk}-*` | pr/147 flip-equivalence, pr closed |
| `work-87{flip,knob-min,knob-sup,grp-*}-*` | doc 87's output-knob evidence (doc committed) |

**Kept:**
- production d102m/d102mpr;
- `vtx105-base`, `pr130r1-probe*` and `pr134-f086`, because scan and display manifests resolve into them (INTERLOCK 9);
- the sentinel layer (`s144*`, `sent97`);
- `d146sv25` (pr/146 §12 is still OPEN: `kine_sat_cont_keep_deg` is `null` in production);
- `probe178410a` and `tfix388-r9`, both non-reproducible.

The three `87grp-*` families score 0 citations. Their names carry no d/p/h round
prefix, so INTERLOCK 14 cannot map them to doc 87 by number. They pass on
`uncited_ok` (exact names), and the planner prints that they did.

## 7. `~/tmp`

**Pins, value-first.** A pin is kept if some text file names it and also names an
arm that survives round C. The name can appear as its `<top>/<leaf>` path, or as
its leaf name from inside its own round dir. Substrate names count as survivors
only if they carry the pin's own round number.

Kept on that test:
- every hand-scan source's pin (`d53`, `d64`, `d66`, `d71`, `p4`, `p65`, `p72`, `p82`–`p88`, `p92`, `d08_libpin/*`, `d61/libpin_item1_before`);
- sbnd production (`d102m-libsnap`, `d97b-libsnap`);
- `d146_libpin*` (OPEN);
- `d09_libpin/pin` and `d41_libpin/new6`, whose round-substrate survives.

Released: 37 pins.

**Rule change, stated.** Rounds A and B skipped any pin with more than 2 MiB of non-`.so` content, calling it "a full local/lib snapshot". Every modern pin is one: doc 81 says *"cp -a ../local/lib … a partial pin is not a pin"*. That filter therefore kept every dead round's pin forever. The size is now printed, not obeyed.

**Preps.** `<round>/prep_<arm>` holds the per-candidate `smprep-*.json` payloads of one arm. A prep is kept if its arm survives, or if a scan record names it; the pdvd smx3–smx9 sheets name `tmp/p90/prep_p90vprod`, `tmp/d68/prep_d68a3`, and others. Measured today, every prep a scan record names also has a surviving arm, so the second rule changes nothing yet. It exists so that a future arm release cannot take a served prep with it.

**Sessions.** A scratchpad is kept if any of these holds:
- it is this session's or the live peer's;
- its transcript was written in the last 12 h;
- a file in it was written in the last 6 h;
- a process has its cwd inside it;
- a committed doc names its path (`d9d688aa` and `1fce3b9b` are named by doc pdvd/69 and `docs/scan/smx19/cmp_review.txt`).

40 idle scratchpads are released, 1.44 GiB.

**Scratch.** `~/tmp/h28` minus `libpin_h28`: launcher logs, private git indexes and superseded readouts, ~9 MB, per the session that wrote it. Also 106 harness leftovers at the `claude-25225` root older than 7 days (0.85 GiB). The root entries `pr33`–`pr36f` stay, because the pr/33–36 port audits name them.

**Bytes are set-relative.** Round A hardlinked 5663 files across pins, so a pin's `du`
is not what removing it frees. An inode is counted only when every one of its links
lies inside the 362-unit set: 45.07 of the 79.46 nominal GiB.

**Left for the owner** (not intermediate on any rule here), largest remaining `~/tmp` dirs by unique bytes after the sweep:

| dir | GiB |
|---|---|
| `xtrack` | 2.71 |
| `doc28` (heap profiles) | 2.20 |
| `d45` | 1.76 |
| `d44sp` | 0.97 |
| `doc37` | 0.82 |
| `d38_arms2` | 0.79 |
| `d15_oldprep` | 0.71 |

## 8. Liveness during the round: a real refusal

At ~04:40 the peer session started doc pdhd/29. It was uncommitted and named
`h26q2dprod` as provenance. At that point the planner's live-token derivation held
`h26q2dprod`, so the stubbed confirm for pdhd re-planned, found `tier1_pdhd`
changed, and **refused with rc=11**. That is INTERLOCK A catching a real peer, not
a constructed one.

The peer pushed doc 29 (`f0524f75`). Its Repro block now names `h28off` as the
rebuild source. The re-plan then reproduced the frozen tier and keep files for all
three trees **byte-for-byte**, so the record freeze made before the push is still
the right one.

## 9. Records, stubs and controls

| path | positive run | causal negative control |
|---|---|---|
| work record layer | `archive_records_20260913.py 1`: 1375/1375 manifests (sbnd 25, pdvd 854, pdhd 496), 224 MB | — |
| `retire_20260913.sh`, stubbed `rm` | CONFIRM=yes on sbnd/pdvd/pdhd: INTERLOCK A re-plan OK, record gate 25/25, 854/854, 496/496, `STUBRM` = tier count, rc=0 | a record copy with **one manifest withheld** → `REFUSING: 1 of 25 targets have no manifest`, **rc=14**, no `STUBRM` |
| same, live peer | — | uncommitted doc 29 naming `h26q2dprod` → tier moved, **rc=11** (§8) |
| `sweep_tmp_20260913.sh`, stubbed `rm` | CONFIRM=yes, `REQUIRE_WORK=no`, into a scratch record dir. Re-census unchanged (362 units), which also confirms the census script and the staged tier file agree after its two edits. Record gate 362/362 fresh manifests (164 MB), the 6 FIFO/special files recorded as `SPECIAL`, `STUBRM 362 units`, rc=0 | `RETIRE_OUT=/proc/nosuch` breaks the **freeze itself**, on the final archiver → 362 × `FAILED creating`, **rc=13**, no `STUBRM` |
| same, run out of order | — | today's state, before the work release → `REFUSING: 25 sbnd work dir(s) from tier1_sbnd_20260913.txt are still on disk`, **rc=4**, no freeze started |

**New gate on the work driver.** Round B's driver only checked that the record
*directory* existed. Round B had already measured that a directory's existence
cannot tell a complete freeze from a partial one. The driver now requires a
manifest for every target (exit 14).

**The ~/tmp sweep refuses to run before the work release.** The census computes
"alive" as disk minus `tier1_*`, which is the *future* state. Without a gate, a
sweep run before, or instead of, `retire_20260913.sh` would remove the pins and
preps of arms still on disk: the value-first failure the whole test exists to stop.
Ordering in §10 is not a guard. The sweep now requires every `tier1` dir to be
gone (exit 4).

**The ~/tmp sweep has the same shape.** It removes nothing unless the freeze
exits 0 and wrote a manifest for every unit during this run (exit 13). Its archiver
creates the output directory *before* hashing, so a bad path fails in seconds
rather than after 45 GiB of reads.

**The first positive ~/tmp stub hung, and that was a real defect.** Three stale
`vgdb-pipe-*` entries at the `claude-25225` root (June valgrind debug pipes) are
**FIFOs**. `sha()` opened one to hash it, and `open()` on a FIFO waits for a
writer that will never come. The freeze sat at unit 347/362 until its workers were
killed. The sweep then refused with rc=13 and removed nothing, which is the gate
behaving as designed on an unplanned failure. The archiver now records every
non-regular file as `SPECIAL mode=…` in the links file and never opens it.

The sbnd sentinel suite on production before the round: **21 PASS / 0 FAIL / 2 OPEN / 7 INERT**. No released sbnd arm is in its default arm set.

## 10. What the owner runs

```bash
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D
CONFIRM=yes ./retire_20260913.sh 1 sbnd
CONFIRM=yes ./retire_20260913.sh 1 pdvd
CONFIRM=yes ./retire_20260913.sh 1 pdhd
python3 tmp_census_20260913.py        # re-census right before the sweep: session ages move the list
CONFIRM=yes ./sweep_tmp_20260913.sh
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # expect 21/0/2/7
```

## 11. Costs, stated

- **Doc pdhd/26 §§1–3.3 become read-only.** Once `h26q2dprod`/`h26conf` go, those sections can only be read from their committed txt (`census.txt`, `michel_energy.txt`, gate files); the Repro block no longer regenerates them. The PDHD per-track numbers can be rebuilt on `h28off`.
- **Doc pdvd/48's 09-06 flip loses its evidence arms.** `d48nu7`, `d48flipcfg` and `d08cfgA` go. Each compiled config (`.wct-pr_*.json`) is carried in the record tar, so the operating point stays readable.
- **Pins of dead rounds** are gone with their arms. A re-run of those rounds would need a rebuild from the commit its doc records.

## 12. Open for the owner

- **`pdhd/l1sp_wf_v9`, 11 G.** Still no regeneration path; untouched.
- **sbnd production imaging inputs.** The `icluster-apa*.npz` inside `work-mcp1k-d102m` (9.82 GiB) and `work-mcp2k-d102m` (19.52 GiB) are what a PR-stage re-run reads. They are inside the production arm, so this round does not touch them. If PR-only re-runs are enough, re-imaging would regenerate them.
- **Hand-scan sources, ~22 GiB.** pdvd 9.4 GiB (11 arms), pdhd 12.7 GiB (16 arms). Kept per the owner's 09-10 instruction.
- **The seven `~/tmp` scratch dirs in §7**, ~10 GiB.

## 13. Post-execution: what ran, and what it left

The owner, 2026-09-13: *"please run it"*. §10's commands ran in order; the logs are
`retire_confirm_{sbnd,pdvd,pdhd}_20260913.log`, `tmp_census_20260913.final.out`,
`sweep_confirm_20260913.log` and `archive_tmp_20260913.log`.

| step | result |
|---|---|
| `retire_20260913.sh 1 sbnd` | INTERLOCK A re-plan unchanged, record gate 25/25, deleted 25 dirs (2.49 GiB), rc=0 |
| `retire_20260913.sh 1 pdvd` | re-plan unchanged, record gate 854/854, deleted 854 dirs (8.21 GiB), rc=0 |
| `retire_20260913.sh 1 pdhd` | re-plan unchanged, record gate 496/496, deleted 496 dirs (8.31 GiB), rc=0 |
| `tmp_census_20260913.py` | 362 units, 45.07 GiB freed; the tier file is identical to the staged one |
| `sweep_tmp_20260913.sh` | work-release gate passed, re-census unchanged, freeze 362/362 (164 MB, 0 failures), record gate 362/362, removed, rc=0; empty containers `d47_libpin`, `d58_libpin` removed by `rmdir` |

**After:**

| | before | after |
|---|---|---|
| pdvd `work/` | 64 G | 56 G |
| pdhd `work/` | 65 G | 56 G |
| sbnd_xin | 81 G | 78 G |
| `~/tmp` | 91 G | 44 G |
| `/home/xqian` free | 421 G | **486 G** |

**Checks on the post-state:**
- **Broken symlinks** are 0 / 0 / 0 in pdhd, pdvd and sbnd_xin after each tree's release.
- **No released family survives.** `h26q2dprod`, `h26conf`, `h26q2d`, `p97v*`, `p81*`, `d48nu7`, `d143pnew`, `d11prod` and `stmwc` are all at 0 dirs.
- **Everything kept is whole:**
  - pdvd: `p96vprod`/`p96vscope` 120/120, d51vclus/d27fresh/keep/d16vnu 120 each, d41prov 99, d39r2prov 21, and all 11 hand-scan sources at full count;
  - pdhd: `h28prod`/`h28wl`/`h28off` 61 each, d51hclus/d16hnu 61, d09 31, stm0/stmw 30, and all 16 hand-scan sources at full count;
  - sbnd: 46 work dirs, including 8 d102m/d102mpr, 18 sentinel arms and 4 d146sv25.
- **Permanent pins** `libpin_h28`, `libpin_p96` and `pdhdstm_libpin` are intact.
- **sbnd sentinel suite** on production after the release: **21 PASS / 0 FAIL / 2 OPEN / 7 INERT**, unchanged.

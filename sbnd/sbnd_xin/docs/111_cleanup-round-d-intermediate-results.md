# 111 — cleanup round D 2026-09-16: keep production's inputs and outputs, retire the intermediate test arms

**Status: EXECUTED.** `/home/xqian` free went from **214 G** before the round to **449 G** after its
last step. About 10 G of that rise came from outside the round (§13).
- **Round E, same evening (§15): EXECUTED.** On the owner's *"clean up a bit the pdvd/work
  directory"*, the closed doc pdvd/111 and 113 study arms were released: 975 dirs, 9.80 GiB.
  - `pdvd/work` went from 115 G to 105 G, and free space to **459 G**.
  - The doc-113 session's two follow-up arms and `d111vst` stay.
- **Round F (§16): EXECUTED.** On the owner's yes, the same studies' arms were released from pdhd and
  sbnd: 511 + 4 dirs, 8.00 GiB.
  - `pdhd/work` went from 69 G to 61 G, and free space to **466 G**.
  - `d113hbase`, `d113hnone`, `d111hst` and doc 113's sbnd gate pair stay.
- **sbnd_xin and pdvd `work/`:** 2211 dirs, 138.4 GiB, released by this session on 2026-09-16.
  - Both ran behind a frozen record layer, a confirm-time re-plan and a stub run with a causal
    negative control (§9).
  - Free went from 214 G to 354 G.
- **pdhd `work/` and `~/tmp`:** released by the owner at 17:22–17:30 (§10). The session's permission
  gate had refused the pdhd deletion.
  - pdhd: 848 dirs, 7.98 GiB.
  - `~/tmp`: 9672 units, 29.63 GiB, plus 3 registered worktrees.
  - An earlier owner run at 13:53 was refused by INTERLOCK 3 while the peer's `d113h*` arms were
    running, and deleted nothing.
- **Owner follow-up, same day (§14):** two file-level releases inside kept production arms.
  - pdvd: `p98von`'s 960 SP frame archives, 39.07 GiB.
  - sbnd: the mcp1k/mcp2k `d102m` icluster npz, 12756 files and 29.34 GiB, plus their 12000 in-arm
    links.
- **After every step:**
  - broken symlinks are 0 / 0 / 0 in the three trees;
  - every planned keep dir is on disk;
  - the sbnd sentinel suite is unchanged at 21 PASS / 0 FAIL / 2 OPEN / 7 INERT.
- **One cited path was restored (§8).** The sweep released `~/tmp/d102/cfg`.
  - Doc pdvd/113 cites that dir for its G1 compiled-config proof. The doc was pushed after the
    pinned remote head.
  - Its four configs were recompiled byte-identical to the freeze's SHA-256 and put back.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D
git -C $IMG fetch https://github.com/WireCell/wcp-porting-validation.git main   # remote_head_20260916.txt = 1ef0e67f
python3 scan_arms_20260916.py --json=scan_arms_20260916.json
python3 toks_20260916.py && python3 cit_20260916.py toks_20260916.txt cit_20260916.json   # 10085 tokens -> 899 cited
python3 plan_20260916.py              # -> plan_20260916.out: FREED 95.86 + 42.55 + 7.97 GiB, interlock failures NONE
python3 archive_records_20260916.py 1 # record layer 3059/3059 manifests (sbnd 2.7 GiB, pdvd 203 MiB, pdhd 67 MiB)
python3 tmp_census_20260916.py        # -> tmp_census_20260916.out: 9632 units, 29.62 GiB freed set-relative
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # 21 PASS / 0 FAIL / 2 OPEN / 7 INERT
CONFIRM=yes ./retire_20260916.sh 1 sbnd     # EXECUTED, retire_confirm_sbnd_20260916.log
CONFIRM=yes ./retire_20260916.sh 1 pdvd     # EXECUTED, retire_confirm_pdvd_20260916.log
CONFIRM=yes ./retire_20260916.sh 1 pdhd     # EXECUTED by the owner 17:22, retire_confirm_pdhd_20260916.log, plan_20260916.confirm_pdhd.out
CENSUS_SUFFIX=sweep python3 tmp_census_20260916.py          # 17:25: 9672 units, 29.63 GiB (tmp_census_20260916.sweep.out.gz)
CENSUS_SUFFIX=sweep CONFIRM=yes ./sweep_tmp_20260916.sh     # EXECUTED 17:29, sweep_confirm_20260916.log, archive_tmp_20260916.log.gz
cat restore_d102cfg_20260916.log            # sec 8: doc pdvd/113's G1 configs recompiled, SHA-256 == freeze manifest
```

## 1. The instruction

The owner, 2026-09-16: *"it is time to clean up some disks for ./pdvd ./pdhd ./sbnd_xin and ~/tmp
directories. We have accumulated quite a bit intermediate debug files (e.g. work\*) direcotries. We
want to retire some to reduce the disk space usage. We have done this multiple times. Note, we want
to keep the input and output files for the production setting for these experiments, but want to
retire the intermediate test files. Please act on this and write the md file, commit and push."*

The keep test is the one docs 104–106 used. A work arm survives only if it is one of:
- **substrate** of something kept;
- **latest production**, including its inputs;
- a **hand-scan source**;
- evidence for a decision still **OPEN**;
- held by a **live** session.

The growth since round C (doc 106, 2026-09-13) made the round necessary:

| | after round C | 2026-09-16, before this round |
|---|---|---|
| `/home/xqian` free | 486 G | **224 G** (214 G by the time the deletions started) |
| pdvd `work/` | 56 G | 190 G |
| pdhd `work/` | 56 G | 73 G |
| sbnd_xin | 78 G | 175 G |
| `~/tmp` | 44 G | 88 G |

## 2. What it comes to

| tree | keep dirs | keep GiB | release dirs | release GiB (set-relative) | state |
|---|---|---|---|---|---|
| sbnd_xin | 54 | 61.60 | 132 | **95.86** | EXECUTED |
| pdvd `work/` | 4758 | 125.79 | 2079 | **42.55** | EXECUTED |
| pdhd `work/` | 1749 | 49.71 | 848 | **7.97** | EXECUTED (owner) |
| `~/tmp` | 835 units | — | 9632 units + 3 worktrees (9672 at the sweep, §8) | **29.62** (+1.6; 29.63 at the sweep) | EXECUTED (owner) |

"Set-relative" means an inode counts toward the freed bytes only when every one of its hard links is
inside the release. That matters in `work/` this round: `pvdimg`'s imaging archives are hard links of
`p98von`'s (doc pdvd/100 line 422). A per-dir `du` would count them twice.

## 3. Production, from primary source

**PDVD moved three times since round C.** It is now:

| layer | arm | primary source |
|---|---|---|
| SP frames (top gain 0.889) | `p98von` | doc pdvd/100 §7.5; `pvdimg` = hard links of `p98von`'s 16 imaging archives (line 422) |
| imaging | `pvdimg` | doc pdvd/100 §7.5 "imaging tag `pvdimg`" |
| clustering + Q/L, LASSO 0.1 | `q29flip` (+ gate twin `q29stm`) | doc qlmatch/29 §7: F1 "production, no overrides", `q29flip` == `q29stm`; Bee set 51c1e410 |
| pctrees the PR re-runs read | `p100flip` | doc pdvd/103 §10.5: "PDVD production since doc 100 §7.5 is `p100flip`" |
| PR, both trajectory levers | `d103vflip` (+ `d103vprod1`) | doc pdvd/103 §14.4 "APPLIED on the owner's go" (toolkit `8fc6070e`); production Bee set (doc pdvd/110 §1); `d103vprod1` = one event through the applied tree with no overrides |

`p96vprod`/`p96vscope` are superseded production. Their PROTECTED line moved to RETIRED.
- `p96vprod` survives anyway, as a hand-scan source (§4).
- `p96vscope` is released.

**PDHD moved.** Doc pdvd/108 records the flip as an explicit owner override of a D2 grade.
- **Production:** `d108hflip`, the flipped config with no TLA, run on `d102hcs`'s pctrees and gated
  equal to `d102hcs`.
- **Kept as production:**
  - `d108hflip`;
  - `d102hcs`, the graded A1 it reproduces (one claim, two arms);
  - `d109hstm`, the display-scope proof and the source of Bee set c3165b38 (doc pdvd/109).
- **The flip's OFF side:** its graded A0, `d101hnew`, which equals `h28prod` on `T_stm_michel`
  (doc pdvd/103 §10). It is kept.
- **Superseded:** `h28prod`, `h28wl`, `h28off` and `h28cfg*` (doc 106's production) are released.

**sbnd** is unchanged in reconstruction.
- Production arms: `work-*-d102m`/`d102mpr`. Doc 109 line 512: they *"stay at `eacacafe`"*.
- `work-{mcp1k,nuecc48}-d109prod` is the only arm run on the current reference `ref/prod-2026-09-14`
  (doc 109 Repro lines 61–62). It is 30 MB and kept as production.

**Production pins.** `~/tmp/d102/libpin_d102` is the pin `d108hflip` ran on (doc 108 §3: "clus md5
091e142b9481 unchanged before and after"). It is also the pin of `d103vflip` (doc 103 §10.5).
`~/tmp/d102m-libsnap` backs sbnd production.

## 4. Hand-scan sources, re-derived

`scan_arms_20260916.py` is round C's derivation, re-run. It adds every arm that the scans since
09-13 put in front of a scanner:

| tree | new sources | records |
|---|---|---|
| pdvd | `d101vkf d102vcsall d102vocs d103v0 d103v1 p100bx p100bxp p100c p96vprod p98vonq p99rwon p99wflip` | smx10, smx11, own100, own100m, own100x, own103v, own103v2, sw99, the carried records |
| pdhd | `d101hkf d102hcs d102hocs` | own103h, own103h2 |

INTERLOCK 13 passes on all three trees. Every source resolves at full count after the release (§13).

## 5. Live and OPEN

**The live peer.** Session `steiner-path-3d-deviation` (docs pdvd/111–112) was asked before planning.
It answered that doc pdvd/113 would write `d113*` arms and `~/tmp/d113`, and that it reads the
following, which it asked to hold:
- PDHD `d101hnew`;
- PDVD `d103v1` and `d101vnew`;
- `~/tmp/d103`, `~/tmp/d108*`;
- the d111 pins and `libpin_d102`.

The prefixes `d111`, `d112` and `d113` are held in every tree and in `~/tmp`. The peer started
writing arms during the round, and all of them were kept:
- pdvd: `d113vbase` (120 dirs) and `d113vstep1` (1);
- sbnd: `work-{mcp1k,mcp2k}-d113snew`.

The confirm-time `keep_*_20260916.confirm.txt` files are larger than the plan-time ones by exactly
those dirs. The tier files re-planned byte-identical, so INTERLOCK A let both trees run.

**Liveness is derived against the pushed remote head.** Local main (`30121ad8`) was nine commits
behind the remote (`1ef0e67f`), and those nine commits are exactly docs pdvd/110–112 and pr/149.
Against local main every one of them read as untracked, so all their arms would have looked live.
`live_tokens()` now reads a private index built from `remote_head_20260916.txt`. It found 0 live
tokens: the only paths that differ are this round's own retire files, which it skips.

**OPEN: `p101q`.** Doc pdvd/100 §8.7 leaves QtoL 0.0783 *"for the owner's decision"*. The recommended
next step reads `p100flip` and `p101q` dumps side by side. `p101q` is kept.

## 6. Substrate, re-derived by a lexical first-hop census

| tree | inbound links (kept hubs) | dropped |
|---|---|---|
| pdvd | pvdimg 11883, d27fresh 8317, d51vclus 4454, p98von 4097, p100flip 2190, keep 876, d41prov 92, d39r2prov 28 | `d16vnu` |
| pdhd | d51hclus 3852, (bare) 515, d09 279, d09ctl2 270, d09ctl 240, stm0 180 | `d16hnu` |

`d16vnu` and `d16hnu` have 0 inbound links. Round C kept them only as the PR runners' `SRC` tag,
because "production cannot be re-run without them". That is no longer true: production is re-run with
`SRC=d103vflip` (PDVD) and `SRC=d108hflip` (PDHD) (doc pdvd/111 Repro lines 141–142). Both PROTECTED
lines moved to RETIRED.

`d51vclus`, `d27fresh` and `keep` are no longer production's substrate. They stay because the hand-scan
sources (`d53v`, `d67v`, `d68*`, `p85*`, `p88*`, `p90vprod`, `p96vprod`, `d08pv30*`) link into them.

## 7. What is released

| tree | round (families) | dirs | GiB | why it is intermediate |
|---|---|---|---|---|
| sbnd | pr/149 rounds 1–2 (`pr149*`, 46) | 88 | 79.77 | "do not switch", knobs OFF, doc and metric TSVs pushed (`794e470c`, `a80b345d`, `1ef0e67f`) |
| sbnd | docs 109–110 gate arms (`d109*` except `d109prod`, `d110*`, 14) | 40 | 16.05 | ROOT-output and group-mode gates; production ON since `c203b400`/`67937f45` |
| sbnd | doc 101's SBND gate (`d101snew/sold`) | 4 | 0.16 | doc pdvd/101, closed |
| pdvd | doc 98/99 SP reruns (`p98voff`, `p98voffq`, `p98kchk`, `p98kq`, `p98g*`) | 261 | 15.86 | the gain-OFF side, superseded by the gain flip |
| pdvd | qlmatch/29 levers (`q29base`, `q29c1`, `q29l1..9`, `q29v6*`, `q29v7nw`) | 252 | 11.00 | 18-event lever arms; the flip is `q29flip` (kept) |
| pdvd | doc 99 window/gain arms (`p99rwprod`, `p99rgon`, `p99rgprod`, `p99w5/w6`, `p99rwnul`) | 364 | 7.53 | superseded by `p100flip`; `p99rwon`/`p99wflip` stay as scan sources |
| pdvd | doc 100 candidates (`p100bd`, `p100bg`, `p100boff`, `p100spflip`) | 361 | 4.26 | not scanned; `p100bx`/`p100bxp`/`p100c` stay as scan sources |
| pdvd | docs 101–102 sim + real cells (`d101*`, `d102*` minus scan sources/holds) | 598 | 1.81 | "not recommended" / "not flipped" |
| pdvd | `p96vscope`, `d16vnu`, `p101cfg` | 243 | 2.11 | superseded production twin; old SRC tag; compile probes |
| pdhd | `h28prod h28wl h28off h28cfg*` | 187 | 2.82 | superseded production (§3) |
| pdhd | doc pdvd/100's PDHD arms `h100a/b`, `d105hdiag` | 183 | 2.89 | closed diagnostics |
| pdhd | docs 101–102 (`d101*`, `d102*` minus `d101hnew`/scan sources) | 417 | 1.47 | as pdvd |
| pdhd | `d16hnu` | 61 | 0.89 | old SRC tag |

All interlocks passed on every tree. INTERLOCK 14 first **refused** sbnd. Four uncited `pr149*`
families could not be mapped to a doc, because the round-number grammar was `(d|p|h)<N>`. It now
also reads `pr<N>` and `q<N>`: `pr149` maps to `docs/pr/149_*.md` and `q29` to `docs/qlmatch/29_*.md`.
Both docs are committed, so the families pass on a doc, not on an excuse.

## 8. `~/tmp` (EXECUTED)

`tmp_census_20260916.py` is round C's census with five changes:

1. **Permanent pins follow production:** `d102/libpin_d102`, `d102m-libsnap`, `pdhdstm_libpin`.
   `libpin_h28` and `libpin_p96` are judged value-first now. Both still stay, because a surviving arm
   is named next to them (§12).
2. **A fifth class, round scratch.** Round C left the closed rounds' scratch "for the owner"
   (doc 106 §7: `xtrack`, `doc28`, `d45`, `d44sp`, `doc37`, `d38_arms2`, ~10 GiB). The owner's words
   now cover it.
   - A top-level dir whose name carries a round number with a committed doc (or is `xtrack`, doc
     pdvd/47) contributes each immediate child as a unit.
   - Excluded children:
     - pins and preps, which their own classes judge;
     - registered worktrees;
     - scan-facing children (`scan*`, `own*`, `shots*`, `prep*`);
     - anything named in a PROTECTED.txt;
     - anything written in the last 2 h;
     - anything whose name carries a **surviving arm** as a `[-_.]` token. This keeps production's
       run logs: `arm_p100flip`, `chain_p98von`, `d102m-A-mcp1k.log`.
   - A dir whose every child is free collapses into one unit.
3. **Production names do not make a pin survive.** Substrate names already could not; now production
   names cannot either, unless they carry the pin's own round number.
   - First run: `pr149/libpin` and `pr149r2/libpin` were "kept" only because doc pr/149 names `d102m`
     as its baseline.
4. **Re-census without overwriting the record**: `CENSUS_SUFFIX` suffixes all three outputs (§10).
5. **Registered worktrees** (`d101/wcp_wt`, `d104/wt104`, `d106/wt106`) are handed to
   `git worktree remove`, never `rm -rf`. The sweep checks, per worktree, that it is clean and that its
   HEAD is on the pushed remote. `d103/wcp_wt_r3` is held (the peer).

| class | KEEP | FREE | GiB |
|---|---|---|---|
| pins | 42 | 5 (`pr149/libpin`, `pr149r2/libpin`, `pr149r2/local/lib`, `d109-libsnap/new`, `d110-libsnap/base`) | 6.94 |
| preps | 22 | 3 (`prep_p96vscope`, `prep_d102vcsall2`, `h25r/prep_sheets`) | 0.52 |
| session scratchpads | 5 | 8 idle (largest: the round A–C cleanup session, 0.49) | 0.49 |
| harness leftovers | 5 | 27 | 0.00 |
| round scratch | 761 | 9589 | 21.66 |

The largest round-scratch releases:

| dir | GiB | what |
|---|---|---|
| `xtrack` | 2.71 | doc pdvd/47 transverse-smearing sim |
| `d102/*` | 2.36 | `sim/`, `logs/` |
| `doc28` | 2.20 | PDVD PR perf heap profiles |
| `d45` | 1.76 | PDVD doc 45 |
| `d101/*` | 1.73 | `sim/`, `bee/` |
| `d44sp` | 0.97 | PDVD doc 44 |
| `p98/*` | 0.92 | gain-OFF chain logs |
| `d38_arms2` | 0.79 | PDVD doc 38 |

**At the sweep (17:25 re-census, 17:29 sweep).**
- The re-census listed **9672** units, 29.63 GiB, against the plan's 9632. The sweep's own re-census
  was byte-identical to it (`tmp_tier_20260916.confirm.txt.gz`).
- No plan-time unit dropped out. All 40 additions had been KEEP at plan time only as "written in the
  last 2 h" or "transcript written in the last 12 h", and had aged out:
  - 30 `pr149r2/*` and 8 `pr149/*` logs and score tables (doc pr/149, closed);
  - an idle session scratchpad (`8c9d4e9c`);
  - `d102/cfg`.
- The sweep then ran:
  - every guard passed;
  - the record gate saw 9672/9672 manifests (`archive/records/cleanup-20260916/tmp`, 257 M, local);
  - rm rc=0;
  - `git worktree remove` for `d101/wcp_wt` (`face213e`), `d104/wt104` (`a27af8e9`) and
    `d106/wt106` (`282c2948`), each HEAD on the remote;
  - the empty `d106` container was removed.
- `~/tmp` went from 96 G to **64 G**.
- The permanent pins, the peer's `d111`/`d113` pins, `d103/wcp_wt_r3`, `d108` and `h28/libpin_h28`
  are all present.

**`d102/cfg` was released but is still cited, and was restored.**
- **The citation.** `d102/cfg` is where `pdvd/docs/nf_sp_img_clus/scripts/d102_compile_pr.sh` writes
  compiled PR configs. Rounds after 102 kept using it: docs 108, 109, 111 and 113.
- **Why the census released it.** Its round-scratch class maps a child to its parent dir's round
  number, which is 102, a committed doc. That mapping is lexical, not what the dir holds.
- **Why no guard caught it.**
  - At plan time it survived only on age: the peer was writing its doc 113 proofs into it.
  - Doc 113 was pushed (`f35e68f1`) after the pinned remote head `1ef0e67f`, so no committed record
    named the path.
  - Of the 40 late units, it is the only one any committed `.md`/`.sh`/`.py` at `f35e68f1` names
    (`git grep`).
- **The restore.**
  - Doc 113's G1 row points at `d113knob{off,none}_{pdhd,pdvd}.json`.
  - They were recompiled with the script's exact `wcsonnet` line and the same `libpin_d102`, and put
    back.
  - All four JSONs and their logs match the freeze manifest's SHA-256, and the knob-off md5 prefixes
    equal doc 113's quoted `a870511c9b22` / `211a49a48229` (`restore_d102cfg_20260916.log`).
  - The dir's other 70 files are older rounds' proofs, not restored. Their hashes stay in
    `d102__cfg.manifest.tsv`, and the logs and small JSONs are in `d102__cfg.tar.zst`.

## 9. Records, stubs and controls

| path | positive run | causal negative control |
|---|---|---|
| work record layer | `archive_records_20260916.py 1`: **3059/3059** manifests (SHA-256 per file, links recorded, text layer carried). Sizes: sbnd 2.7 GiB, pdvd 203 MiB, pdhd 67 MiB | — |
| `retire_20260916.sh`, `rm` stubbed on PATH | CONFIRM=yes on sbnd: INTERLOCK A re-plan unchanged, record gate 132/132, `STUBRM 133 targets` (`-rf` + 132), rc=0 (`retire_stub_sbnd_20260916.log`) | one manifest withheld (`work-mcp1k-d101snew`) → `REFUSING: 1 of 132 targets have no manifest`, **rc=14**, no `STUBRM` (`retire_negctl_sbnd_20260916.log`) |
| `sweep_tmp_20260916.sh`, run out of order | — | before the work release → `REFUSING: 132 sbnd work dir(s) … still on disk`, **rc=4** (`sweep_negctl_order_20260916.log`) |
| same, freeze broken | guards: 9632 units, none written in the last hour, no process cwd inside, permanent pins intact | `RETIRE_OUT=/proc/nosuch` → `froze 0/9632`, **rc=13**, nothing removed (`sweep_negctl_freeze_20260916.log`, `archive_tmp_negctl_20260916.log.gz`) |

## 10. What ran

Ran (2026-09-16):

| step | by | result |
|---|---|---|
| `CONFIRM=yes ./retire_20260916.sh 1 sbnd` | session | re-plan unchanged, record gate 132/132, deleted 132 dirs, rc=0 |
| `CONFIRM=yes ./retire_20260916.sh 1 pdvd` | session | re-plan unchanged, record gate 2079/2079, deleted 2079 dirs, rc=0 |
| `CONFIRM=yes ./retire_20260916.sh 1 pdhd` | — | the session's permission gate refused it five times, including after the owner's explicit go-ahead; it was handed back |
| same, 13:53 | owner | **refused, nothing deleted.** The re-plan failed `INTERLOCK 3: no live writer (0 of 122 sampled dirs moved, 16 tree-scoped wire-cell procs)`: the peer's `d113hfoot/hnone/hnopaint/hbase2` arms were running in `pdhd/work` until 14:04. They are held and not in the release, but the guard is tree-scoped by design. rc=10; the sweep behind it stopped on its order guard |
| `PLAN_SUFFIX=check1500 python3 plan_20260916.py pdhd`, 15:05 | session | a check-only re-plan (`PLAN_SUFFIX=check1500`, outputs removed) with no procs running: interlock failures NONE, tier byte-identical, 848 dirs |
| same, 17:22 | owner | INTERLOCK A re-plan unchanged (`plan_20260916.confirm_pdhd.out`), record gate 848/848, deleted, rc=0; broken symlinks 0 / 0 / 0 |
| `CENSUS_SUFFIX=sweep python3 tmp_census_20260916.py`, 17:25 | owner | 9672 units, 29.63 GiB, rc=0 (§8) |
| `CENSUS_SUFFIX=sweep CONFIRM=yes ./sweep_tmp_20260916.sh`, 17:29 | owner | re-census unchanged, record gate 9672/9672, removed, 3 worktrees removed, pins intact, rc=0; free 449 G |
| `pr127_sentinels.py --arms 'work-*-d102mpr'`, 17:30 | session | 21 PASS / 0 FAIL / 2 OPEN / 7 INERT (`sentinels_post_sweep_20260916.txt`) |

The commands as the owner ran them:

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D
CONFIRM=yes ./retire_20260916.sh 1 pdhd        # re-plan unchanged, record gate 848/848, rc=0
CENSUS_SUFFIX=sweep python3 tmp_census_20260916.py            # re-census right before the sweep: session ages move the list
CENSUS_SUFFIX=sweep CONFIRM=yes ./sweep_tmp_20260916.sh       # work-release gate, re-census gate, freeze, record gate, rm, git worktree remove
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # 21/0/2/7
```

**`CENSUS_SUFFIX` is not optional.** The census writes `tmp_tier_20260916.txt`,
`tmp_census_20260916.json` and `tmp_worktrees_20260916.txt`, which are the committed plan-time
records. A plain re-run would overwrite them in place (the doc-104 defect `PLAN_SUFFIX` fixed for the
planner). With the suffix set, the census writes `*.sweep.*` and the sweep acts on those files. A dry
run with `CENSUS_SUFFIX=nosuch` refuses with `tmp_tier_20260916.nosuch.txt missing`, rc=2, so the
sweep really reads the suffixed file.

**If INTERLOCK A refuses pdhd (rc=11), the tier moved.** Diff `tier1_pdhd_20260916.txt` against
`tier1_pdhd_20260916.confirm.txt`.
- **Not the cause:** new `d113h*` arms, because they are held by prefix.
- **The likely cause:** the peer's doc pdvd/113, before it is pushed.
  - Liveness reads against the pinned remote head `1ef0e67f` (`remote_head_20260916.txt`), so an
    unpushed doc 113 reads as untracked.
  - If it names a released pdhd arm as a baseline, `live_tokens()` marks that family live and the tier
    shrinks. `h28prod`, the pre-flip reference, is the obvious candidate; `h100a/b`, `d101hold` and
    `d105hdiag` are also in the tier.
- **Resolution:** confirm the new line comes from the peer's doc. Then either:
  - add that family to pdhd `keep_arms` and re-plan (release less); or
  - wait for the peer's push, re-fetch, write the new head into `remote_head_20260916.txt` and re-plan.

## 11. Costs, stated

- **Doc pr/149's tables can be read but not regenerated.** Its tables are the zzi sign test, jitter/bow,
  association trace, vertex tolerance and topology. The per-event calib dumps and trace logs are gone.
  What remains is the committed `149_figs` TSV/txt and the record layer's hashes.
  - §13.8's recommended next step compares the fitted drift-x profile with the image's time centroid
    on the same 150 ISO events. The fitted profile lived in the released calib dumps.
  - That step is re-runnable, not free. The stage-A inputs (`work-*-d102m`) are kept and the knobs ship
    OFF, but PR must be re-run after rebuilding from toolkit `f573cdeb`/`e73850ad`, because
    `pr149/libpin` and `pr149r2/libpin` are in the `~/tmp` tier.
- **Doc pdvd/99's OFF side and doc 29's lever arms can be read but not regenerated.**
  - `p98voff` held the gain-OFF SP frames; they regenerate from raw with the doc 99 §9 recipe.
  - Each lever arm's compiled config is in its record tar.
- **Doc pdhd/28's flip has lost its evidence arms** (`h28prod/wl/off`, released with pdhd). `d101hnew`
  still carries the pre-doc-108 production on `T_stm_michel`.
- **Compiled-config proofs of docs 102–111 in `~/tmp/d102/cfg`** are gone except doc 113's four
  (restored, §8). Each is a deterministic `wcsonnet` compile, checkable against
  `d102__cfg.manifest.tsv`.
- **Docs 109–110's gate arms** are gone. The gates' verdicts are the committed `109_logs`/`110_logs`.

## 12. Open for the owner

- ~~**`p98von`'s SP frames, 40 G**~~ **RETIRED on the owner's yes (§14).** *Original item:* the 960 `protodune-sp-dnnroi-frames-*.tar.bz2`, inside production's input chain. They are the frames production
  imaging read. `pvdimg` holds only the imaging archives, as hard links. Doc pdvd/99 §9 regenerates SP
  frames bit for bit (it retired the July frames that way). Retiring these is a file-level release
  inside a kept arm, a different unit from this machinery, so it is not staged.
- ~~**sbnd production imaging inputs**~~ **RETIRED on the owner's yes (§14).** *Original item:*
  `icluster-apa*.npz` inside `work-mcp1k-d102m` / `work-mcp2k-d102m`, 29.3 GiB (doc 106 §12).
- **Hand-scan sources**: the set grew by 15 arms this round (§4): 24 G in pdvd and 3.0 G in pdhd by
  `du`. Kept per the owner's 09-10 instruction.
- **`~/tmp/h28/libpin_h28` (1.76 GiB)** survives value-first only because a doc names `p96vprod` (a
  scan source) next to it. The association is lexical, not physical.
- **`~/tmp/d15_oldprep` (0.71 GiB)**: its children are named `prep-*`, so round scratch keeps them as
  scan-facing. The prep class does not judge them (it reads `prep_*`).
- **`pdhd/l1sp_wf_v9`, 11 G**: still no regeneration path.
- **For the next round's machinery (§8):** a unit that survives the plan only on age must be re-judged
  against the pushed remote head *at sweep time*.
  - The `~/tmp` sweep's re-census read liveness and citations against the pinned head, so doc 113,
    pushed in between, could not protect `d102/cfg`.
  - The fix: re-fetch before the re-census, or refuse a unit whose KEEP reason changed from age to FREE
    unless a fresh citation check passes.
  - `d102/cfg` also belongs in pdvd `PROTECTED.txt` if later rounds keep writing proofs there.

## 13. Post-state

### 13.1 pdhd and `~/tmp` (the owner's run, 17:22–17:30)

| | before (15:09) | after |
|---|---|---|
| pdhd `work/` | 77 G | **69 G** |
| `~/tmp` | 96 G | **64 G** |
| `/home/xqian` free | 409 G | **449 G** |

The 409 G "before" is 10 G above §14's 399 G. That rise came from outside this round: `pdvd/work` went
from 121 G to 115 G with no step of this round running.

- **Nothing released is left:** 0 of the 848 tier dirs are on disk. `h28prod`, `h28wl`, `h28off`,
  `h100a/b`, `d105hdiag`, `d16hnu`, `d101hold` and `d101cnew` are at 0 dirs.
- **All 1749 planned keep dirs are on disk.**
  - Production: `d108hflip` 61, `d102hcs` 61, `d109hstm` 5.
  - OFF side: `d101hnew` 61.
  - Substrate: `d51hclus` 61, `d09` 61, `stm0` 30.
  - Hand-scan sources at their planned counts: 61 each; `d08cap10`/`d08goff` 30; `d05mON` 6.
  - The peer's `d111h*` (377) and `d113h*` (306) are untouched.
- **Broken symlinks are 0 / 0 / 0** in pdhd, pdvd and sbnd_xin (the driver's post-state check).
- **`~/tmp` holds 285 dangling symlinks, none of them made by this round.** Each was resolved by
  target, and none points into the sweep's tier or into any round-D work tier:
  - 239 point at work arms that were already gone;
  - 40 sit inside `sbnd-cfg-organize` and `layoutcheck-166650-a`, neither in any tier, and point at
    files missing from those same dirs;
  - 6 are others: 3 inside the peer's worktree `d103/wcp_wt_r3`; 1 onto a toolkit cfg file; 2
    relative links (`../../../Woodpecker/`, `../../upload-to-bee.sh`).
- **sbnd sentinel suite: 21 PASS / 0 FAIL / 2 OPEN / 7 INERT**, identical to every earlier reading
  (`sentinels_post_sweep_20260916.txt`).

### 13.2 sbnd + pdvd (this session, 13:39–13:40)

| | before (13:0x) | after |
|---|---|---|
| sbnd_xin | 178 G | **82 G** |
| pdvd `work/` | 193 G | **149 G** |
| `/home/xqian` free | 214 G | **354 G** |

- **Broken symlinks** are 0 / 0 / 0 in pdhd, pdvd and sbnd_xin after each tree (the driver's
  post-state check).
- **No released family survives.** `p98voff`, `p98voffq`, `p99rwprod`, `p96vscope`, `d16vnu`, `q29base`,
  `d101vold` and all `pr149*` families are at 0 dirs.
- **Everything kept is whole in pdvd.**
  - Production chain: `d103vflip` 120, `d103vprod1` 1, `q29flip` 120, `q29stm` 120, `p100flip` 120,
    `pvdimg` 120, `p98von` 120.
  - OPEN `p101q` 120; peer reads `d103v0`/`d103v1`/`d101vnew` 120 each.
  - Every hand-scan source at full count: 120 each, except `d08pv30on/off` 30.
  - Substrate: `d51vclus`/`d27fresh` 120, `d41prov` 99, `d39r2prov` 21.
  - The peer's `d111v*` (729) and `d113v*` (121) are untouched.
- **sbnd:** 56 work dirs: `d102m`/`d102mpr`/`d109prod` 10, sentinel layer, `d146sv25`, the display
  manifests' arms, the peer's `d111s*` 6.
- **sbnd sentinel suite** on production after the release: **21 PASS / 0 FAIL / 2 OPEN / 7 INERT**,
  identical to before (`sentinels_pre_20260916.txt`, `sentinels_post_20260916.txt`).

## 14. Owner follow-up: the SP frames in `p98von` and the sbnd icluster npz (EXECUTED)

The owner, 2026-09-16: *"We can remove the 40 G SP frames inside p98von, and remove the sbnd's final
icluster npz files."* Both are FILE-level releases inside arms the keep test protects, which is a
different unit from §2–§10. They have their own plan, freeze and driver:
- plan and freeze: `plan_files_20260916b.py`;
- driver: `retire_files_20260916b.sh`;
- records: `archive/records/cleanup-20260916b/`, a new label (M13).

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D
python3 plan_files_20260916b.py --hash          # gates F1-F5 + SHA-256 manifests (freeze_files_20260916b.log)
./retire_files_20260916b.sh                     # dry run: record + product gates (retire_files_dry_20260916b.log)
CONFIRM=yes ./retire_files_20260916b.sh         # EXECUTED (retire_files_confirm_20260916b.log)
```

**Scope.**

| set | files | GiB | also removed |
|---|---|---|---|
| `pdvd/work/*_p98von/protodune-sp-dnnroi-frames-anode*.tar.bz2` | 960 | 39.07 | — |
| `sbnd_xin/work-{mcp1k,mcp2k}-d102m/{evt,g}<N>/icluster-apa*-{active,masked}.npz` | 12756 | 29.34 | the 12000 `ql_evt<N>/icluster-*.npz` symlinks onto them (same arms), and the 3000 `evt<N>/` dirs they emptied (`rmdir` only) |

The item the owner approved was doc 106 §12's mcp1k/mcp2k 29.3 GiB. `work-ncpi0-d102m` and
`work-nuecc48-d102m` hold another 0.7 GiB of icluster npz and are **not** touched.

**Gates.** Each gate refuses with its own exit code.

| gate | check | result |
|---|---|---|
| F1 | every target is a regular file with a single name (`st_nlink == 1`) | PASS |
| F2 | no symlink in pdvd/pdhd `work/`, sbnd_xin or `~/tmp` resolves onto a target by inode, except the listed in-arm links | PASS |
| F3 | every listed in-arm link resolves onto its own set | PASS |
| F4 | no process holds a target open | PASS |
| F5 | `pvdimg`'s 2040 files share no inode with the frames | PASS |
| record (rc 14) | a manifest row with the file's size for every target, and a frozen links file identical to the list | 960/960, 12756/12756 |
| product (rc 15) | every `p98von` and `pvdimg` event keeps 16 non-empty imaging archives; every `ql_evt<N>` of the two arms keeps a non-empty `pctree-evt<N>.tar.gz` | PASS |
| re-plan (rc 10/11) | at confirm time, F1–F5 re-run into `*.confirm.*` and the lists are unchanged | OK |
| broken symlinks (rc 16) | not allowed to rise | 0 → 0 in pdvd and sbnd_xin |

**Controls.**
- **F2 is causal.** Two symlinks were placed in `~/tmp/cleanup-20260916/`, one onto a frame archive and
  one relative link onto an npz. F2 failed naming both (`plan_files_negctl_20260916b.out`, rc=1), and
  `--hash` froze nothing. After the links were removed, F2 passed.
- **The record gate is causal.** A manifest copy with one npz row withheld gave
  `REFUSING: 1 of 12756 sbnd-d102m-icluster targets have no manifest row`, rc=14
  (`retire_files_negctl_20260916b.log`).

**Post-state.**

| | before §14 | after |
|---|---|---|
| pdvd `work/` | 149 G (plus the peer's new `d113*` arms since) | 121 G |
| sbnd_xin | 82 G | 53 G |
| `/home/xqian` free | 346 G | **399 G** |

Checks after the delete:
- 0 targets left on disk.
- `pvdimg` and `p98von` still hold 1920 imaging archives each.
- All 3000 `pctree-evt<N>.tar.gz` of the two arms are intact.
- The sbnd sentinel suite reads 21 PASS / 0 FAIL / 2 OPEN / 7 INERT (`sentinels_post_files_20260916b.txt`).
- The PROTECTED lines of `p98von` and `d102m` carry a dated note.

**What this costs, and how to get it back.**
- **PDVD imaging cannot be re-run from disk.** Production's imaging is still readable (`pvdimg`), but
  a re-run needs SP first. SP is bit-deterministic (doc pdvd/99 G2). Its ON-arm command regenerates the
  frames, and the manifest's hashes check them.
- **sbnd stage-A clustering cannot be re-run from disk for mcp1k/mcp2k.** PR-only re-runs are
  unaffected: they read the kept stage-A pctrees. Re-imaging is doc 102's stage A (`run_chain_group.sh`
  on the reco1 files). The manifest's hashes check the regenerated npz.

**Still on disk, if more is wanted:**
- the sbnd group-mode SP frames `g<N>/frames-dnn.tar.bz2` in the four `d102m` arms: 194 files, 3.9 G;
- the ncpi0/nuecc48 icluster npz: 0.7 G.


## 15. Round E: the closed doc 111 and 113 arms in `pdvd/work` (EXECUTED)

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D     # remote_head_20260916d.txt = 55da248e
python3 scan_arms_20260916d.py --json=scan_arms_20260916d.json       # scan set unchanged from round D
python3 toks_20260916d.py && python3 cit_20260916d.py toks_20260916d.txt cit_20260916d.json   # 7671 tokens -> 611 cited
python3 plan_20260916d.py pdvd        # -> plan_20260916d.out: 975 dirs, FREED 9.79 GiB, interlock failures NONE
python3 archive_records_20260916d.py 1   # 975/975 manifests, 50 M (archive_t1_20260916d.log)
PATH=<stub>:$PATH CONFIRM=yes ./retire_20260916d.sh 1 pdvd                                   # STUBRM 976 (-rf + 975), rc=0
PATH=<stub>:$PATH REC_OVERRIDE=<copy, 1 manifest withheld> REPLAN=no CONFIRM=yes ./retire_20260916d.sh 1 pdvd   # rc=14
CONFIRM=yes ./retire_20260916d.sh 1 pdvd   # EXECUTED by the owner, retire_confirm_pdvd_20260916d.log, rc=0
python3 survey_pdvd_20260916d.py; python3 patterns_pdvd_20260916d.py   # family / file-class census (survey writes /home/xqian/tmp/cleanup-20260916c/survey.json)
```

**The instruction.** The owner, 2026-09-16, after §10–§14: *"you may clean up a bit the pdvd/work
directory to remove some intermediate files?"*

**What was left to release, measured before planning.** After round D, `pdvd/work` held 92.47 GiB
(first link of every inode). Walked through the keep test, it was nearly all keeps:
- production and its input chain (§3);
- substrate (§6);
- OPEN `p101q`;
- the 23 hand-scan sources (§4);
- the arms round D held by prefix for the live peer: `d111*`, `d112*`, `d113*`.

The prefix hold was the only mass reachable without a new policy. A new session,
`steiner-graph-retile-removal`, had started minutes earlier, so it was asked before planning.

**The peer's answer**, by message:
- **Keep `d113vbase` and `d113vnone`.** Its one open follow-up is a hand scan, waiting for the owner,
  of the 29 new PDVD false positives in `none` vs base. The scan reads these arms'
  `tracking-pr.root`, `tracking-stm.root` and `mabc-pr.zip`.
- **Not needed:** the rest of `d111v*`, `d111svoff5` and `d113v{base2,foot,nopaint,step1}`.
- No new prefix is planned. The earlier holds are unchanged.
- **Costs it named:**
  - Docs 112/113's Repro and the `d112_*` scripts' `--arm` name `d111vst`.
  - `d113_verdict.py` cannot re-run levels L1/L2 once `foot`/`nopaint` are gone.

**The planner** is `plan_20260916d.py`, round D's planner with only the pdvd config changed. It was
run on pdvd only.
- The `d111`/`d113` prefix hold is lifted; `d112` stays held.
- `keep_arms` gains `d113vbase`, `d113vnone` and `d111vst`. `d111vst` stays although the peer could
  spare it, so docs 112/113 still reproduce as written (1.24 GiB).
- Liveness is read against the pushed head `55da248e`.

| released family | dirs | | released family | dirs |
|---|---|---|---|---|
| `d111voff` | 120 | | `d113vbase2` | 120 |
| `d111vsr6` | 120 | | `d113vfoot` | 120 |
| `d111vsr10` | 120 | | `d113vnopaint` | 120 |
| `d111vsr15` | 120 | | `d113vstep1` | 1 |
| `d111vtr` | 120 | | `d111voff5`, `d111svoff5` | 5 + 5 |
| `d111v{S,A0,A1,K}` | 1 each | | **total** | **975 dirs, 9.79 GiB set-relative** |

**Gates and controls.**
- **Planner interlocks:** all pass, with 0 live writers and 0 tree-scoped procs.
  - No kept symlink resolves into the release, and nothing released is PROTECTED or a record dir.
  - All 23 hand-scan sources are kept, and INTERLOCK 14 finds 0 uncited families.
  - 0 live tokens from uncommitted records.
- **Outside links:** a walk of `~/tmp` (depth 4), `pdhd/work`, `sbnd_xin` and `pdvd` found **0** symlinks
  resolving into a released dir.
- **Stub run:** INTERLOCK A re-plan unchanged, record gate 975/975, `STUBRM 976 targets`, rc=0
  (`retire_stub_pdvd_20260916d.log`).
- **Causal control:** one manifest withheld (`039349_31_d111vtr`) → `REFUSING: 1 of 975 targets have no
  manifest`, **rc=14**, no `STUBRM` (`retire_negctl_pdvd_20260916d.log`).
- **Execution:** the owner ran `CONFIRM=yes ./retire_20260916d.sh 1 pdvd` in this session's shell mode.
  - The re-plan was unchanged (`plan_20260916d.confirm.out`: interlock failures NONE).
  - Record gate 975/975, deleted, rc=0.

**Post-state.**

| | before | after |
|---|---|---|
| `pdvd/work` (`du`) | 115 G | **105 G** |
| `pdvd/work` (first link) | 92.47 GiB | 82.69 GiB |
| `/home/xqian` free | 449 G | **459 G** |

- **Release complete:** 0 of 975 targets are left, and every released family is at 0 dirs.
- **Keeps intact:** all 4384 planned keep dirs are on disk.
  - Production at full count: `d103vflip` 120, `d103vprod1` 1, `q29flip`/`q29stm`/`p100flip`/`pvdimg`/`p98von` 120.
  - `p101q`, `d103v0`, `d103v1`, `d101vnew`, `d113vbase`, `d113vnone` and `d111vst` at 120 each.
- **Broken symlinks** are 0 / 0 / 0 in pdhd, pdvd and sbnd_xin.
- **PROTECTED:** pdvd `PROTECTED.txt` now carries `d113vbase d113vnone`, `d111vst` and
  `/home/xqian/tmp/d102/cfg` (§8, §12), each with its reason.

**Costs.**
- **Doc pdvd/111's lever tables** (`sr6/sr10/sr15`, `off`, `tr`) and the single-event trace arms can be
  read from the committed figs but not regenerated without re-running.
- **Doc pdvd/113's verdict:** the L1/L2 levels (`foot`, `nopaint`) and the base2 repeat are gone.
  `figs/113_verdict.txt` and `figs/113_gate_base2_*.txt` remain the record.

**What is left in `pdvd/work`, by file class (first link, after this round).**
- **Every family is a keep.** More space means removing files inside kept arms, the §14 unit, which
  needs the owner to name each class.

| class | GiB | why it is not proposed |
|---|---|---|
| `calib-pr-evt*.json` + `calib-evt*.json` | 23.36 + 13.33 | records of past runs (M13), and the dumps hand scans were read against; production's Bee set comes from `d103vflip`'s |
| `pctree-evt*.tar.gz` | 13.48 | the inputs every PR re-run reads (production, substrate, scan sources) |
| `clusters-apa-*-ms-{active,masked}.tar.gz` | 9.69 + 0.96 | production imaging (`pvdimg` hard links `p98von`); its SP frames are already gone (§14), so nothing on disk regenerates it |
| `tracking-{stm,pr}.root`, `mabc-*.zip` | ~12 | scan-facing outputs (ROOT trees, Bee zips) |
| `wct_{clus,pr,img}_*.log` | ~5.6 | 120 committed pdvd docs and scripts name `wct_pr_*` logs (census and scan tools read them by name); gzip would break them |
| `gpu_mem_*.csv` in `keep`/`p98von` | 0.83 | the only pure run-time trace; a candidate if the owner wants it |

## 16. Round F: the same studies' arms in `pdhd/work` and sbnd_xin (EXECUTED)

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D     # remote_head_20260916e.txt = ee91071b
python3 scan_arms_20260916e.py --json=scan_arms_20260916e.json       # scan set unchanged
python3 toks_20260916e.py && python3 cit_20260916e.py toks_20260916e.txt cit_20260916e.json   # 6681 tokens -> 605 cited
python3 plan_20260916e.py pdhd sbnd   # -> plan_20260916e.out: pdhd 511 dirs 7.84 GiB, sbnd 4 dirs 0.16 GiB, interlock failures NONE
python3 archive_records_20260916e.py 1   # 515/515 manifests, 45 M (archive_t1_20260916e.log)
./retire_20260916e.sh 1 pdhd sbnd        # dry run, rc=0 (retire_dry_20260916e.log)
CONFIRM=yes ./retire_20260916e.sh 1 pdhd sbnd   # EXECUTED by the owner, retire_confirm_20260916e.log, rc=0
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # 21/0/2/7 (sentinels_post_20260916e.txt)
```

**The instruction.** The owner, answering §15's recommended next step (*"pdhd and sbnd still hold
this same round's doc 111/113 test arms … If you want those gone too"*): *"yes, please go ahead
removing those."*

**The peer's answer.** The doc-113 session, now named `pdvd-doc-97-refresh`, replied by message:
- **Keep in pdhd:**
  - `d113hbase` and `d113hnone`, which the pending hand scan of the 21 PDHD Michels lost in `none`
    reads;
  - `d111hst`, named in docs 112/113's Repro and the `d112_*` scripts' `--arm`. Doc 112's logs are
    in `~/tmp/d111/arm_d111hst`.
- **Keep in sbnd:** `work-mcp{1k,2k}-d113snew` and `work-mcp{1k,2k}-d111ssnew`, the two sides of doc
  113's G3 gate (`figs/113_gate_sbnd.txt`).
- **Not needed:** everything else.
- **Its flag:** check whether a doc 111 figure names `d111snew`/`d111sold` before releasing them.
  - They are doc 111's own SBND gate sides: round 1 compared `snew` with `sold`, round 2 `ssnew` with
    `sold`.
  - The verdicts are committed (`figs/111_gate_off_sbnd.txt`, `figs/111s_gate_sbnd.txt`), the same
    standing as docs 109–110's gate arms released in round D. They are released.
- **New prefixes:** none. Its current task (doc 97 round 2) reads production `pdvd/work/*_d103vflip`
  only.

**The planner** is `plan_20260916e.py`, round E's planner with only the pdhd and sbnd configs changed.
- The `d111`/`d113` hold is lifted in both trees; `d112` stays held.
- `keep_arms` gain the five kept arms above.
- Liveness is read against `ee91071b`.

| tree | released families | dirs | GiB (set-relative) |
|---|---|---|---|
| pdhd | `d111h{off,sr6,sr10,sr15,tr}` 61 each; `d113h{base2,foot,nopaint}` 61 each | 488 | |
| pdhd | `d111h{A0,A1,K,S,sr10tr,stsmk}` 1 each; `d111hoff5` 5; `d111shoff1` 1; `d111shoff{5,5b}` 5 each; `d113hstep1` 1 | 23 | |
| pdhd | total | **511** | **7.84** |
| sbnd | `work-mcp{1k,2k}-d111snew`, `work-mcp{1k,2k}-d111sold` | **4** | **0.16** |

`d111shoff*` were not in the 683-dir count the owner said yes to, because a `*_d111h*` glob does not
match them.
- They are doc 111 round 2's PDHD off arms, the counterparts of pdvd `d111svoff5` released in §15.
- The peer said everything outside its keep list can go.
- The plan summary shown to the owner before the delete ran names them.

**Gates and controls.**
- **Interlocks:** all pass in both trees, with 0 live writers and 0 tree-scoped procs.
  - All hand-scan sources and production are kept.
  - INTERLOCK 14 finds 0 uncited families. INTERLOCK 16 saw 2 uncommitted tokens, neither resolving
    in these trees.
- **Outside links:** a walk of `~/tmp` (depth 4), `pdhd`, `sbnd_xin` and `pdvd/work` found **0** symlinks
  resolving into a released dir.
- **Record layer:** 515/515 manifests (`archive/records/cleanup-20260916e`, 45 M). A direct check found a
  non-empty manifest for every target.
- **No stub run this round.** The session's permission classifier refused the stubbed `CONFIRM=yes`
  practice run, and it was not retried. Instead:
  - a dry run: rc=0, 511 + 4 present, 0 already gone;
  - a diff showing `retire_20260916e.sh` differs from round E's driver only in the 7 stamp lines.
    Round E's driver passed the stub run and the rc=14 withheld-manifest control (§15).
- **Execution:** the owner ran `CONFIRM=yes ./retire_20260916e.sh 1 pdhd sbnd`.
  - The re-plan was unchanged (`plan_20260916e.confirm.out`: interlock failures NONE).
  - Record gates 511/511 and 4/4, both deletes rc=0.

**Post-state.**

| | before | after |
|---|---|---|
| `pdhd/work` (`du`) | 69 G | **61 G** |
| sbnd_xin (`du`) | 53 G | 53 G |
| `/home/xqian` free | 459 G | **466 G** |

- **Release complete:** 0 of 511 and 0 of 4 targets are left. Of the `d111h*`/`d111sh*`/`d113h*`
  families only `d111hst`, `d113hbase` and `d113hnone` remain, at 61 dirs each.
- **Keeps intact:** all 1544 pdhd and 52 sbnd keep dirs are on disk.
  - Production: `d108hflip` 61, `d102hcs` 61, `d109hstm` 5.
  - `d101hnew` 61 and substrate `d51hclus` 61.
  - The four sbnd gate-pair dirs.
- **Broken symlinks** are 0 / 0 / 0.
- **sbnd sentinel suite:** 21 PASS / 0 FAIL / 2 OPEN / 7 INERT, unchanged.
- **PROTECTED:** pdhd `PROTECTED.txt` now carries `d113hbase d113hnone` and `d111hst`; sbnd's carries the
  `d113snew`/`d111ssnew` gate pair.

**Costs.**
- **Doc pdvd/111's PDHD lever and trace arms and doc pdvd/113's PDHD L1/L2 levels and base2 repeat** can
  be read from the committed figs but not regenerated without re-running.
- **Doc 111's two SBND gates** (`snew`/`ssnew` vs `sold`) cannot be re-checked.

After rounds D–F, the only `d111*`/`d113*` arms left in the work trees are the named keeps:
- pdvd: `d113vbase`, `d113vnone`, `d111vst`;
- pdhd: `d113hbase`, `d113hnone`, `d111hst`;
- sbnd: `work-mcp{1k,2k}-d113snew`, `work-mcp{1k,2k}-d111ssnew`.

`~/tmp/d111` and `~/tmp/d113` were not in scope and are untouched.

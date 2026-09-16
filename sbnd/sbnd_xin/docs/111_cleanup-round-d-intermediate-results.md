# 111 — cleanup round D 2026-09-16: keep production's inputs and outputs, retire the intermediate test arms

**Status: PARTLY EXECUTED.**
- **Done:** sbnd_xin and pdvd `work/` were released on 2026-09-16: 2211 dirs, 138.4 GiB.
  - Both ran behind a frozen record layer, a confirm-time re-plan and a stub run with a causal
    negative control (§9).
  - `/home/xqian` free went from **214 G to 354 G**.
  - Broken symlinks are still 0 / 0 / 0.
  - Every kept arm is whole.
  - The sbnd sentinel suite is unchanged at 21 PASS / 0 FAIL / 2 OPEN / 7 INERT.
- **Staged, not run:** the pdhd `work/` release (848 dirs, 7.97 GiB) and the `~/tmp` sweep
  (9632 units, 29.62 GiB, plus 3 registered worktrees).
  - The pdhd deletion was refused by the session's permission gate after the other two trees
    ran. The session did not try to get around it.
  - The sweep's own order guard refuses to run until the pdhd release is done (rc=4, §9).
  - The commands are in §10.

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
| pdhd `work/` | 1749 | 49.71 | 848 | **7.97** | staged |
| `~/tmp` | 835 units | — | 9632 units + 3 worktrees | **29.62** (+1.6) | staged |

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

## 8. `~/tmp` (staged)

`tmp_census_20260916.py` is round C's census with four changes:

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
4. **Registered worktrees** (`d101/wcp_wt`, `d104/wt104`, `d106/wt106`) are handed to
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

## 9. Records, stubs and controls

| path | positive run | causal negative control |
|---|---|---|
| work record layer | `archive_records_20260916.py 1`: **3059/3059** manifests (SHA-256 per file, links recorded, text layer carried). Sizes: sbnd 2.7 GiB, pdvd 203 MiB, pdhd 67 MiB | — |
| `retire_20260916.sh`, `rm` stubbed on PATH | CONFIRM=yes on sbnd: INTERLOCK A re-plan unchanged, record gate 132/132, `STUBRM 133 targets` (`-rf` + 132), rc=0 (`retire_stub_sbnd_20260916.log`) | one manifest withheld (`work-mcp1k-d101snew`) → `REFUSING: 1 of 132 targets have no manifest`, **rc=14**, no `STUBRM` (`retire_negctl_sbnd_20260916.log`) |
| `sweep_tmp_20260916.sh`, run out of order | — | before the work release → `REFUSING: 132 sbnd work dir(s) … still on disk`, **rc=4** (`sweep_negctl_order_20260916.log`) |
| same, freeze broken | guards: 9632 units, none written in the last hour, no process cwd inside, permanent pins intact | `RETIRE_OUT=/proc/nosuch` → `froze 0/9632`, **rc=13**, nothing removed (`sweep_negctl_freeze_20260916.log`, `archive_tmp_negctl_20260916.log.gz`) |

## 10. What ran, and what the owner runs

Ran (2026-09-16):

| step | result |
|---|---|
| `CONFIRM=yes ./retire_20260916.sh 1 sbnd` | re-plan unchanged, record gate 132/132, deleted 132 dirs, rc=0 |
| `CONFIRM=yes ./retire_20260916.sh 1 pdvd` | re-plan unchanged, record gate 2079/2079, deleted 2079 dirs, rc=0 |
| `CONFIRM=yes ./retire_20260916.sh 1 pdhd` | **not run**: the session's permission gate refused the command |

The owner runs, in this order:

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D
CONFIRM=yes ./retire_20260916.sh 1 pdhd        # expect: re-plan unchanged, record gate 848/848, rc=0
python3 tmp_census_20260916.py                 # re-census right before the sweep: session ages move the list
CONFIRM=yes ./sweep_tmp_20260916.sh            # work-release gate, re-census gate, freeze, record gate, rm, git worktree remove
(cd ../../../sbnd/sbnd_xin && python3 scripts/pr127_sentinels.py --arms 'work-*-d102mpr')   # expect 21/0/2/7
```

If INTERLOCK A refuses pdhd (rc=11), the tier moved. Diff `tier1_pdhd_20260916.txt` against
`tier1_pdhd_20260916.confirm.txt`. Held `d113h*` arms cannot cause it, because they are held by prefix.

## 11. Costs, stated

- **Doc pr/149's tables can be read but not regenerated.** Its tables are the zzi sign test, jitter/bow,
  association trace, vertex tolerance and topology. The per-event calib dumps and trace logs are gone.
  What remains is the committed `149_figs` TSV/txt and the record layer's hashes. §13.8's recommended
  next step (does the image follow the bow?) needs new arms on the image, not these.
- **Doc pdvd/99's OFF side and doc 29's lever arms can be read but not regenerated.**
  - `p98voff` held the gain-OFF SP frames; they regenerate from raw with the doc 99 §9 recipe.
  - Each lever arm's compiled config is in its record tar.
- **Doc pdhd/28's flip loses its evidence arms** (`h28prod/wl/off`), once pdhd runs. `d101hnew` still
  carries the pre-doc-108 production on `T_stm_michel`.
- **Docs 109–110's gate arms** are gone. The gates' verdicts are the committed `109_logs`/`110_logs`.

## 12. Open for the owner

- **`p98von`'s SP frames, 40 G (`du` of the 960 `protodune-sp-dnnroi-frames-*.tar.bz2`), inside production's input chain.** They are the frames production
  imaging read. `pvdimg` holds only the imaging archives, as hard links. Doc pdvd/99 §9 regenerates SP
  frames bit for bit (it retired the July frames that way). Retiring these is a file-level release
  inside a kept arm, a different unit from this machinery, so it is not staged.
- **sbnd production imaging inputs**: `icluster-apa*.npz` inside `work-mcp1k-d102m` / `work-mcp2k-d102m`,
  29.3 GiB (doc 106 §12, unchanged).
- **Hand-scan sources**: the set grew by 15 arms this round (§4): 24 G in pdvd and 3.0 G in pdhd by
  `du`. Kept per the owner's 09-10 instruction.
- **`~/tmp/h28/libpin_h28` (1.76 GiB)** survives value-first only because a doc names `p96vprod` (a
  scan source) next to it. The association is lexical, not physical.
- **`~/tmp/d15_oldprep` (0.71 GiB)**: its children are named `prep-*`, so round scratch keeps them as
  scan-facing. The prep class does not judge them (it reads `prep_*`).
- **`pdhd/l1sp_wf_v9`, 11 G**: still no regeneration path.

## 13. Post-state (sbnd + pdvd)

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

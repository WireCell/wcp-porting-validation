# 125 — cleanup round K (2026-09-25): retire the doc 123/124 ladders and the ToT campaign after two production flips

**Status (2026-09-25 14:00): round K is COMPLETE — `/home/xqian` went 120 G → 635 G free.**
- **Pass 1 (11:40), the directory release:** 3562 dirs, 369.97 GiB, rc=0 in all three trees,
  120 G → 488 G free, 0 broken symlinks against a recorded pre-count of 0. The permission
  classifier then refused a read-only spot check, so the session stopped and handed the rest over
  as the sec 6 block.
- **Pass 2 (12:10–14:10, sec 9):** the owner asked "can we reduce them further?" and then chose
  "You run it now", so this session ran the sec 6 block itself. The owner also added three
  levers: archive the cold pins, compress the kept-arm logs, and re-baseline the sentinels so
  `pr150s0` can go. Pass 2 took free space from 481 G to **635 G** (peers had written 7 G since
  pass 1); `sbnd_xin` went 257 → 147 G and `~/tmp` 118 → 73 G. One pin, `d115/libpin_d115`, turned out to be PDHD
  production's and was restored the same day (sec 9.2). Every deletion in pass 2 went through a gated
  release the owner ruled on: the sec 1 rulings, plus the `pr150s0` ruling in sec 9.3. The pins
  and the logs were not deleted but archived or compressed; every file is sha256-verified, and
  each has a restore script.

Owner instruction, verbatim (2026-09-25):

> "Hi, the /home/xqian disk is quite full, I think it is time to retire some directories in ~/tmp,
> ./pdvd ./pdhd ./sbnd_xin. We have done several rounds of this clean up. We want to keep the latest
> production, and can retire the intemediate results and work* directories. Can you investigate this
> and perform the cleanup. Can you summarize in a dedicated md file, commit and push."

Round J (doc 122) closed at 768 G free on 09-21. Four days later there were **122 G** left, so
~646 G had come back. Nearly all of it came from two campaigns that both ended in a production flip
today or yesterday:

- the SBND doc 123/124 hit-flash campaign: sbnd_xin 117 → **568 G**, of which `work-*-d123*` was
  ≈ 420 G;
- the PDVD ToT flip campaign (qlmatch 31–35) plus doc pdvd/117's SP arms: pdvd/work 75 → **190 G**.

pdhd (51 G of work) and `~/tmp` (108 G) moved little.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D
SX=$IMG/sbnd/sbnd_xin

# BEFORE anything is touched: the gates as before/after pairs (logs *_20260925.log)
python3 $SX/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0'   # 21 PASS, 0 FAIL, 2 OPEN, 7 INERT, 0 SKIP
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref $SX/ref/prod-2026-09-25b              # PASS, 28 artifacts
find $IMG/{pdhd,pdvd} $SX -xtype l | wc -l                                           # 0 / 0 / 0
( cd ~/tmp && find d102/libpin_d102 pdhdstm_libpin d123-libpin-r6 p35/libpin_prod -type f -print0 \
    | sort -z | xargs -0 md5sum ) > pins_before_20260925.md5                         # 28063 files
git -C $IMG -c credential.helper='!gh auth git-credential' \
    ls-remote https://github.com/WireCell/wcp-porting-validation.git refs/heads/main  # 07ae168f -> remote_head_20260925.txt

# fork (sec 2), then census -> plan
python3 toks_20260925.py                                          # 10462 dirs -> 10895 tokens
python3 cit_20260916d.py toks_20260925.txt cit_20260925.json      # 928 cited, 151598 hits
python3 scan_arms_20260916d.py --json=scan_arms_20260925.json     # pdvd 27, pdhd 22 (unchanged)
python3 plan_20260925.py                                          # every interlock PASS, 369.99 GiB
RETIRE_JOBS=16 python3 archive_records_20260925.py 1              # 3562/3562 arms frozen (M13)
./retire_20260925.sh 1 sbnd pdvd pdhd                             # dry run
CONFIRM=yes ./retire_20260925.sh 1 sbnd pdvd pdhd                 # EXECUTED 11:3x, rc=0
python3 plan_files_20260925.py                                    # the file level: see sec 4 (F2 FAILS until the sweep)
```

## 1. The owner's four rulings (asked with the measurements on the table)

| question | ruling |
|---|---|
| `d123hits{,pr}` — production for ~6 h today (hit flashes, no light gate), ~106 GB | **"Release all of it"** (r3nue included) |
| the old SBND production set still PROTECTED on grounds that stopped being true today (d102m, prod0923, d116tfull, d116s0rep, d115, d118flip*, closed sbnd/119 arms) | **"All but d115pr"** |
| file level inside kept arms: split icluster copies + d123base's reco1-flash `ql_evt*/` (~90 GB) | **"Both"** |
| PDVD holds: ToT Q/L blind-scan sources, `q29flip/q29stm`, the doc 111–113 peer holds | **all three released** |

## 2. What is kept — the latest production, its inputs, and live work

Production was re-derived **value-first** (`feedback_audit_value_first_not_name_adjacency`): the
shipped value was read from each flip section, and the arm carrying it was found from that.

**SBND.** Production moved twice today:
- doc 123 sec 17: `flash_source=hits`, ref/prod-2026-09-25;
- sec 19: the QLXTPC scenario-1 light gate + ceiling 2.9, ref/prod-2026-09-25b.

| kept | why |
|---|---|
| `work-<7 samples>-d123base` | **substrate.** Its `g*/` holds the **only** SP frames + imaging on our disk (d115 and d102m hold **0** `frames-dnn.tar.bz2` since round J), and every d123 Q/L arm links into `work-<s>-d123base/g*/` by absolute path (315/630/770/315 links for the mcp1k/mcp2k/r3cv/r3off lgop). `mc_base.sh` built the round-3 bases from reco1 through `products/d115/<s>/files.lst`, not from d115. |
| `d123lgop`, `d123lgoppr` (mcp1k, mcp2k, r3cv, r3off) | **the latest production output**: exactly the sec 17 + sec 19 values, by TLA, on the four samples they were measured on; pin `~/tmp/d123-libpin-r6` |
| `d123lgflip{,off,pr}` | the flipped runner with **no TLA** on the production libs (== lgop 48/48 A+B, MC 18/18) and its `SBND_XTPC_SC1_GATE=0` escape; the only production-runner output for nueCC-48 / NCpi0 |
| `d123flip{,off,pr}` | the sec 17 flip gate's record, **and** the link target of the live pdvd/119 `work-nuecc48-d119g*` (absolute links into `work-nuecc48-d123flip/g0`) |
| `pr150s0` | **the sentinel reference, no longer production.** The suite gives 21/0/2/7 on it and 13 PASS / 9 FAIL on the d123 arms. It was never re-baselined there, and several of those FAILs are seg-id log strings. `pr127_sentinels.find_arm` SKIPs silently (exit 0) when the arm is gone. |
| `d115pr` | the owner's keep of 09-21, re-affirmed today |
| `pr150csp3bw` | the B arm of pr/150's blind vertex scan (120 KEY items; re-rendering needs its calib dumps) |
| `work-nuecc48-d119*`, `work-mcp10-m66d119*` | **live**: the pdvd doc 119 round's SBND gates, created today |
| everything PROTECTED already held (vtx105, sent97, s144*, pr134-f086, pr130r1, d146sv25, d113snew, d111ssnew, tfix388, probe178410a), the two empty `mcp10-m66d34*sb` placeholders | unchanged |

**PDVD.** Production moved 2026-09-24 (doc qlmatch/35 sec 11, wcp b989f63b): ToT light +
`lasso_weight_unrailed` + `ks_sat_tol` 0.3075.

| kept | why |
|---|---|
| `q35flip` | the flipped runner with no override, == the graded cell `q35tk` 120/120; pin `~/tmp/p35/libpin_prod` |
| `q35tk q35ctl q31ctl q34tk1` | the flip's gate controls (q35tk + q35ctl are also the smx35/smx35c STM scan sources) |
| `d116vflip` | no longer production, **kept**: the record key-arm of `d35_stm_grade.py:10` and the STM+Michel release builder's source (`build_release.py:64,76`) |
| substrate, 27 derived scan sources, `p101q`, `d103v0/v1`, `d101vnew`, labels | unchanged |
| `d117*`, `d118*`, `d119*` | **live** (doc pdvd/117 lists next steps; docs 118/119 were being written today by peer `pdvd-run-39305-tpc-beam-flash`), except the closed sbnd/119 arms `d119vnu/vr3`, released by **exact name** |
| light dirs `<run>_light<evt>_<arm>` | **not in scope.** The planner grammar does not match them, so they fail closed. q35flip reads `_tot` through `.tlas`, not a symlink, so a closure walk could not see that dependency. Carried forward. |

**PDHD.** Production is unchanged (`d116hflip d116hr2 d102hcs d109hstm`). The live pdvd/119 knob-off gates `d119g*`/`d119sg*` are held.

## 3. What was released (executed, set-relative bytes)

| tree | universe | KEEP | RELEASE | freed |
|---|---|---|---|---|
| sbnd_xin | 203 dirs | 88 = 234.18 GiB | **115 dirs** | **311.99 GiB** |
| pdvd | 8314 dirs | 5191 = 115.08 GiB | **3123 dirs** | **53.19 GiB** |
| pdhd | 1945 dirs | 1621 = 31.50 GiB | **324 dirs** | **4.79 GiB** |

**sbnd:**
- `d123hits` ×7 76.87 GiB, `d123hitspr` ×7 29.32;
- the d123 ladder: `lg` 37.60, `basepr` ×7 29.26, `nr` 18.70, `lgpr` 12.66, `hitsnr` 10.13, `nosplit` 10.13, `hitsnu` 9.44, `nrpr` 6.30, `hitsnrpr` / `nosplitpr` 3.74 each, `hitsnupr` 3.34, `r6off` 0.50;
- the old production set: `d116tfull` 18.98, `d115` 15.96, `d102m` 10.69, `d116s0rep` 8.00, `prod0923{,pr}` 1.44, `d118flip{,rep}` 0.31;
- the closed sbnd/119 r3* arms (21 families, ~3.4, including round J's carried `d119flip-torn`);
- `pr150tcsp3bw` 1.47, `pr150g16new` + `d113g16off` 0.16, `d109prod` 0.03.

**pdvd:**
- `q29flip/q29stm` 8.90;
- the ToT ladder:
  - scan sources `q32ti`, `q33{ts,tm,tu,cs,cm,cu}`, `q34tk2/3`, `q34ck1-3` (~24);
  - `q31tot/tot2`, `q32tis`;
  - the `q3xctlp` pin gates (~8);
  - `q35cfg`;
- `d111vst` / `d113v*` 3.60;
- `d119vnu/vr3` 2.38.

**pdhd:** `d113h*` / `d111hst` 2.74, `d119hnu/hr3` + `d119nu` 1.83, `d34*` 0.26.

The record layer comes first (M13): `archive/records/cleanup-20260925/{sbnd,pdvd,pdhd}-tier1/`
holds a SHA-256 manifest, a links file and a `.tar.zst` of the non-heavy files for every one of the
3562 dirs, 1.3 GiB on disk. The driver's record gate read back 115/115, 3123/3123 and 324/324
before it deleted anything. The distilled tables of every released d123 arm are
`products/d123/` (99 files, **committed with this doc** — they had never been tracked). The
colleague Bee sets of doc 123 sec 18.7 (`numu3000_{LOST,GAINED}_{base,hits}.zip`) are built from arms
this round released, so they cannot be regenerated. They move from `~/tmp/d123/bee` into
`bee/d123-numu3000/` (sha256-verified copy, committed).

## 4. The file level (planned and gated; not yet run)

`plan_files_20260925.py` (fork of round J's) has two sets, both on allow-listed arms only:

| set | what | files | GiB |
|---|---|---|---|
| `sbnd-icluster-split` | `evt<N>/icluster-apa*-{active,masked}.npz` in d123base ×7 + d123lgop ×4, plus the 24068 `ql_evt<N>/icluster-*` symlinks in lgop that point at them | 56408 | 67.65 |
| `sbnd-base-qlevt` | every file + link in `work-*-d123base/**/ql_evt<N>/`: base's own reco1-flash Q/L output (pctree, Bee zips, opflash) | 48510 | 23.24 |

**What they are:**
- The split copies are the Q/L job's input handoff. `run_chain_group.sh:425-432` says: "The Q/L job above is its ONLY consumer -- the PR chain's compiled config contains no icluster/.npz reference at all".
- The campaign ran with the runner default `SBND_QL_KEEP_ICLUSTER=1`. That is the runner's default, not a production contract to drop them.
- The copies are re-made from the arm's `g*/` group imaging by the split step.

**What is never touched:** `g*/` (gate F8). It holds the only frames and imaging, and lgop links into it.

**Cost, stated:** the off-path gates of doc 123 sec 17.3 (flipoff == base) and sec 19.3
(lgflipoff == hits) can no longer be re-checked from disk without re-running Q/L.

**Gates, first run:**
- F1, F3, F4–F9 **PASS**.
- **F2 FAILS, correctly.** 648 symlinks in `~/tmp/d124/nudge/ql_{mcp1k,mcp2k}_{null,t0,t0b}/` (doc 124's t0-nudge Q/L views, built on d123base) resolve onto set-A targets. Their other inputs, the hits arms, are released anyway, and they are closed doc 124 scratch.
- The fix is the **order**, not an exception: the `~/tmp` sweep runs **before** the file level, and the confirm-time re-plan must then meet F2 with them gone.
- If the census keeps `~/tmp/d124`, F2 keeps failing and the driver refuses (rc=10). That is the intended stop.

The driver `retire_files_20260925.sh` does five things:
1. re-plans at confirm time (rc 10/11);
2. requires a manifest row with size + sha256 for every file and a row for every link (rc 14);
3. removes the files, then the links;
4. `rmdir`s only the per-event dirs this empties;
5. refuses after the fact if any tree's broken-symlink count rose (rc 16).

## 5. Defects a plain fork would have carried, and the ones the gates caught

1. **The three PROTECTED files are UNIONED across trees.** A pdvd line `d118* d119*` for the live peer would have held sbnd's closed `d118flip`/`d119ctl…`, and the pdvd/pdhd exact-name releases, because `prot_hit` prefix-matches arm tokens from any tree. Caught before the first plan, by listing every d117–d119 name in all three trees. The line now names the live pdvd arms plus the narrow prefixes `d118cfg*`, `d119beam*` and `d119s*`, and the tree-scoped hold lives in the planner's pdvd `open_prefix`.
2. **`release_exact` (new) + INTERLOCK 17.** The closed sbnd/119 arms `d119vnu/vr3/hnu/hr3` and pdhd `d119nu` (09-21) sit under the live pdvd/119 prefix. They are released by exact name, and the release is gated on each name resolving and being wholly released. It is never allowed to override production, substrate, a scan source or PROTECTED.
3. **Liveness is typed by hand again.** `live_tokens()` against the pinned remote head found **0** tokens: docs pdvd/117–119 are committed and pushed while their peer is busy writing. The ~/tmp census's `RNUM` would have mapped `~/tmp/d117`, `d118*` and `d119*` to the **closed sbnd** docs 117–119 and released the live peer's scratch: `d119inst` 10.6 G, `d119wt` 7.0 G (a toolkit worktree) and `d119wcp` (a wcp worktree at 55ed7333). `LIVE_TOP = d117 d118 d119 wcfm`, kept in both copies (census and sweep). `wcfm-*` (the live FM campaign) is named explicitly, because it was safe from RNUM only by accident.
4. **`tmp_census.SAMP` still lacked `r3cv|r3nue|r3off`.** Round I fixed the planner's copy only.
5. **`registered_worktrees()` asked only the wcp repo.** It now also asks the toolkit repo, so a unit containing `~/tmp/d119wt` can never be `rm -rf`'d into a stale registration.
6. **Permanent pins follow production:**
   - `~/tmp/d123-libpin-r6` is now the **only** binary that reproduces lgop. `local/lib` Clus, Util and Aux moved today with the pdvd/119 install, while Match is still md5 d57af13f.
   - `~/tmp/p35/libpin_prod` is the only binary that reproduces q35flip.
   - Both are in `PERMANENT` and PROTECTED, with md5 baselines taken.
   - `d102m-libsnap` leaves the permanent set and is judged value-first.
7. **The planner's sbnd `production` list named arms that were going.** They were replaced, not deleted, so INTERLOCK 11 still resolves every name.
8. **Shared PROTECTED line split before retiring.** pdvd line 56 carried `d116vflip` (kept, as the release source) together with `d116vr2`. It was split before either changed.
9. **INTERLOCK 14 fired on the first plan run.** The trigger was two empty (4 KB) placeholders `work-mcp10-m66d34{b,n}sb` that map to no doc. They were kept rather than excused by a ruling nobody gave; they free nothing.
10. `PROTECTED` sbnd line 832 named a non-existent `work-r3nue-d118fliprep`. Retired with the line.
11. **The pin dedup was never scoped to the live holds.** `dedup_pins_20260912.py` is stamp-free, so it was
    not in the fork list. Its `LIVE_TMP` held only round C's `h26 h27 h28 p97 mg10`. It discovers any dir holding
    at least 5 `.so` files as a pin root, so it would have hardlinked and `chmod -w`'d the live peer's seven
    `~/tmp/d119inst/*/lib` trees while its install waiter was running. It was forked as
    `dedup_pins_20260925.py`, with `LIVE_TMP` += `d117 d118 d119 wcfm claude-`. The report-only run shows 58 roots, none of them held.
    The lesson from sec 2's fork list: a stamp-free helper is still part of the round, and it needs the round's holds.

## 6. The command block — the rest of the round (owner runs this from bash mode: type `!` first, then paste)

**Pass 2 (sec 9): this session ran this block on the owner's go, with the measured results in sec 9.1.**

Each step has its forecast; a disagreement means stop and read the log.

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; SX=$IMG/sbnd/sbnd_xin; cd $D

# 0. the production spot checks the classifier refused (read-only)
for a in $SX/work-*-d123lgop; do echo "$a broken $(find $a -xtype l | wc -l)"; done         # expect 0 each
ls -d $IMG/pdvd/work/*_q35flip | wc -l; ls -d $IMG/pdvd/work/*_light*_tot | wc -l            # 120 / 120
ls -d $SX/work-*-d123lgoppr $SX/work-*-d123base $SX/work-*-pr150s0 $SX/work-*-d115pr | wc -l  # 4+7+4+3 = 18

# 1. ~/tmp census + sweep (BEFORE the file level -- it releases ~/tmp/d124/nudge, see sec 4)
python3 tmp_census_20260925.py > tmp_census_20260925.out 2>&1; echo rc=$?; tail -15 tmp_census_20260925.out
./sweep_tmp_20260925.sh > sweep_dry_20260925.log 2>&1; echo rc=$?; tail -8 sweep_dry_20260925.log
CONFIRM=yes ./sweep_tmp_20260925.sh > sweep_run_20260925.log 2>&1; echo rc=$?; tail -12 sweep_run_20260925.log
#    exit 11 = the unit list moved between census and confirm: CENSUS_SUFFIX=sweep re-census, then re-run
git -C $IMG worktree remove /home/xqian/tmp/wt-d123    # clean, HEAD e396d5b8 is on the remote; not a census unit (no round number)

# 2. the file level (forecast: 104918 files, 90.89 GiB + 56408 links)
python3 plan_files_20260925.py > plan_files_20260925.out 2>&1; echo rc=$?; grep -E 'PASS|FAIL' plan_files_20260925.out
#    all of F1-F9 must PASS (F2 now included); then freeze and release:
python3 plan_files_20260925.py --hash > plan_files_20260925.hash.out 2>&1; echo rc=$?; tail -4 plan_files_20260925.hash.out
./retire_files_20260925.sh > files_dry_20260925.log 2>&1; echo rc=$?; cat files_dry_20260925.log
CONFIRM=yes ./retire_files_20260925.sh > files_run_20260925.log 2>&1; echo rc=$?; cat files_run_20260925.log

# 3. the driver logs this round orphaned (planned AFTER the release, so frozen == live list)
python3 plan_orphanlogs_20260925.py > plan_orphanlogs_20260925.out 2>&1; echo rc=$?; tail -12 plan_orphanlogs_20260925.out
python3 archive_orphanlogs_20260925.py files_orphanlogs_pdvd_20260925.txt files_orphanlogs_pdhd_20260925.txt > archive_orphanlogs_20260925.log 2>&1; echo rc=$?
CONFIRM=yes ./retire_orphanlogs_20260925.sh > orphanlogs_run_20260925.log 2>&1; echo rc=$?; tail -6 orphanlogs_run_20260925.log

# 4. the non-destructive win: hardlink identical pin libraries, nothing deleted
#    FORKED as dedup_pins_20260925.py: the 09-12 original's LIVE_TMP does not hold d117/d118/d119/wcfm/claude-,
#    and would hardlink + chmod -w the live peer's seven ~/tmp/d119inst/*/lib trees mid-install.
#    Report-only run this session: 58 roots, none held; 12.32 GiB recoverable (dedup_pins_20260925.report.out).
CONFIRM=yes python3 dedup_pins_20260925.py > dedup_pins_20260925.log 2>&1; echo rc=$?; tail -5 dedup_pins_20260925.log

# 5. after-gates -- each must equal its sec 0 before-value
python3 $SX/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0' > sentinels_after_20260925.log 2>&1; tail -1 sentinels_after_20260925.log   # 21/0/2/7/0
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref $SX/ref/prod-2026-09-25b > prodcfg_after_20260925.log 2>&1; echo rc=$?                # PASS 28
for t in pdhd pdvd sbnd/sbnd_xin; do echo "$t $(find $IMG/$t -xtype l 2>/dev/null | wc -l)"; done                                  # 0 0 0
( cd ~/tmp && md5sum -c --quiet $D/pins_before_20260925.md5 ) && echo "pins intact"
df -h /home/xqian | tail -1
```

Forecast for steps 1–4:
- the file level ≈ 91 GiB;
- `~/tmp` ≈ 20–40 GiB. The census decides: a large part of `~/tmp` is lib pins and the live peer's `d119*`;
- orphan logs ≈ 1–2 GiB;
- dedup ≤ 12.3 GiB (report-only; the sweep releases some of those pins first).

That takes `/home/xqian` from 488 G to **roughly 600 G free**.

## 7. As executed

| step | result |
|---|---|
| baselines | sentinels `pr150s0` **21 PASS / 0 FAIL / 2 OPEN / 7 INERT / 0 SKIP**; `prod_cfg_gate` **PASS 28**; broken symlinks 0/0/0; pins 28063 files md5'd; toolkit HEAD `9de7fcae` (peers commit to the shared tree); remote head `07ae168f` |
| record layer | `products/d123` + `bee/d123-numu3000` committed; `archive_records_20260925.py 1` froze **3562/3562** arms |
| planner | every interlock PASS on the second run (the first run's INTERLOCK 14 fail is `plan_20260925.firstrun.out`); INTERLOCK 3: 0 sampled dirs moved, 0 tree-scoped writers |
| directory release | INTERLOCK A re-plan **unchanged** in all three trees; record gate 115/115, 3123/3123, 324/324; **sbnd 311.99, pdvd 53.19, pdhd 4.79 GiB** removed, rc=0; broken symlinks **0 / 0 / 0** |
| production links | the post-state **0 broken symlinks over all of sbnd_xin** already implies every `lgop` link into `d123base/g*/` resolves; the refused spot check (sec 6 step 0) re-checks it per arm |
| free space | **120 G → 488 G** (+368 G, against a set-relative forecast of 369.6 GiB) |
| file level, sweep, orphan logs, dedup, after-gates | not run in pass 1, because the classifier refused the next command. **Pass 2 ran all of them (sec 9.1).** |

## 8. Carried forward to round L

1. ~~Re-baseline the sentinel suite on d123~~: **done in pass 2 (sec 9.3), and `pr150s0` is released.**
   **Round L's before/after sentinel gate is therefore**
   `python3 $SX/scripts/analysis/d125/sentinels_prod.py --arms 'work-*-d123lgoppr' 'work-*-d123lgflippr'`,
   expected **13 PASS / 0 FAIL / 10 OPEN / 7 INERT / 0 SKIP**. The old `--arms 'work-*-pr150s0'` line now SKIPs
   silently with exit 0. `work-sent150-*` holds the pre-flip record.
2. **PDVD light dirs** (`<run>_light<evt>_<arm>`): ~12 GiB, of which ~4.8 are ToT-ladder light (`f3x18`, `g3x`, `q31/q32`). They need a grammar with an allow-list, because q35flip's `_tot`, the release builder's `_keep`/`_wf` and the flip evidence `_q32ti/_q35esc/_g31off` are read through config paths that a symlink closure cannot see.
3. The hit-flash `opflash_apa*.tar.gz` in the production arms (doc 123 sec 17.4 item 6; ~3 GiB per big lgop arm).
4. Round J's items that are still open:
   - the 120 shared pdhd frames (6.77 GiB);
   - the bee/mabc zips and `.wct-*.json`;
   - `restore_compress`'s sbnd family filter.
5. `pdvd/work/*_d117sp/gpu_mem_*.csv` is still being written today by a leftover monitor. It is inside a held arm.
6. **Gates that skip the `.zst` names.** `scripts/d119/lever_gate.py` and `d118/hash_gate.py` skip
   `stdout.log`, `wct_pr_evt*.log`, `.wct-cfg-evt*.json` with `$`-anchored regexes, so a `.zst` copy
   would show up on one side only. They compare the closed `r3*` d119/d116 arms, which are gone. Widen
   the regexes to `(\.zst)?$` before reusing them on a compressed arm.
7. `scripts/d113_eval{,_pr}.sh` compare against `pr150s0` (closed doc 113). Its cached metrics in
   `docs/pr/150_figs/metrics/pr150s0-*.tsv` remain; the arm does not.

## 9. Pass 2 (2026-09-25 12:10–14:10): the sec 6 block, and three more levers

The owner, 2026-09-25: *"it looks like the sbnd_xin directory is still very big, and the same to
~/tmp. I wonder if we can reduce them further?"* The session started by measuring. None of sec 6
had run: `sbnd_xin` was 257 G, `~/tmp` 118 G and free space 481 G. Pass 1 had left 488 G free; the
7 G since then are peer writes (wcfm-gnn/e1, pdvd/119), not a regression. The owner was asked
with the sizes on the table and answered:
- "You run it now (Recommended)";
- "Archive cold pins (~26 GiB)";
- "Compress kept-arm logs (~11 GiB)";
- "Re-baseline sentinels, drop pr150s0 (~10.5 GiB)";
- then, once the 8 FAILs had been traced: "OPEN as doc 118 cost, release".

### 9.0 Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; SX=$IMG/sbnd/sbnd_xin; cd $D
python3 tmp_census_20260925.py; ./sweep_tmp_20260925.sh; CONFIRM=yes ./sweep_tmp_20260925.sh      # 554 units
git -C $IMG worktree remove /home/xqian/tmp/wt-d123
python3 plan_files_20260925.py; python3 plan_files_20260925.py --hash
./retire_files_20260925.sh; CONFIRM=yes ./retire_files_20260925.sh                                # 90.89 GiB
python3 plan_orphanlogs_20260925.py                                                               # 0 logs
python3 archive_pins_20260925.py; CONFIRM=yes python3 archive_pins_20260925.py                    # 53 cold pins
RESTORE_TO=<scratch> ./restore_pins_20260925.sh p82/libpin_p82                                    # test restore
python3 compress_logs_20260925.py; CONFIRM=yes python3 compress_logs_20260925.py                  # 41983 files
python3 $SX/scripts/retire/witness_sentinels_20260925.py                                          # work-sent150-*
python3 $SX/scripts/analysis/d125/sentinels_prod.py --arms 'work-*-d123lgoppr' 'work-*-d123lgflippr'   # 13/0/10/7/0
python3 toks_20260925b.py; python3 cit_20260916d.py toks_20260925b.txt cit_20260925b.json
python3 scan_arms_20260916d.py --json=scan_arms_20260925b.json
export LIVE_REF=cdeceb5f562f709931b81008721ffc0c64ddda93     # the pushed head carrying sentinels_prod.py
python3 plan_20260925b.py sbnd; python3 archive_records_20260925b.py 1
./retire_20260925b.sh 1 sbnd; CONFIRM=yes ./retire_20260925b.sh 1 sbnd                            # pr150s0
python3 dedup_pins_20260925.py; CONFIRM=yes python3 dedup_pins_20260925.py                        # 13 roots
```

### 9.1 As executed

| step | result |
|---|---|
| sec 6 step 0, spot checks | `lgop` broken links 0/0/0/0; `q35flip` 120, `_tot` light 120; 18 kept production/reference arms |
| `~/tmp` sweep | **554 units, 15.41 GiB** set-relative; INTERLOCK A re-census unchanged; record gate 554/554; rc=0; `~/tmp` 118 → 103 G |
| `wt-d123` | clean, and its HEAD e396d5b8 is the main tree's HEAD; `git worktree remove`, rc=0 |
| file level | F1–F9 **PASS** at plan and at confirm (F2 now passes, since the sweep released the d124 nudge links). **104918 files, 90.89 GiB + 56408 links**; 22187 emptied per-event dirs rmdir'ed; broken symlinks 0 → 0 in all three trees; 488 → 588 G |
| orphan logs | **0**: every driver log's event dir is still alive |
| cold pins (9.2) | **53 pins, 33.04 GiB freed** into one 6.57 GiB archive (52 stay archived: `d115/libpin_d115` was restored, see 9.2); 941467 manifest rows, re-checked on a full extract; test restore 572/572 |
| log compression (9.4) | **41983 files, 10.49 → 1.22 GiB**; each .zst decompressed to its frozen sha256 before the original went; broken symlinks 0 → 0 |
| after-gates | `prod_cfg_gate` **PASS 28**; sentinels on `pr150s0` **21/0/2/7/0** (== before); the four permanent pins md5-intact (`pins_before_20260925.md5`) |
| sentinel re-baseline (9.3) | witness `work-sent150-*` 21/0/2/7/0, verdict-for-verdict == `pr150s0`; production **13 PASS / 0 FAIL / 10 OPEN / 7 INERT / 0 SKIP** |
| `pr150s0` release (stamp `20260925b`) | tier = exactly the 4 `pr150s0` arms, **9.78 GiB**; every interlock PASS; record gate 4/4 (`archive/records/cleanup-20260925b`); INTERLOCK A unchanged; rc=0; broken 0/0/0 |
| pin dedup | 13 roots (the live ones held), 1130 files linked, **3.28 GiB**; pins still md5-intact after |
| correction | `d115/libpin_d115` (PDHD production's pin) restored from the archive and made PERMANENT (9.2); `work-nuecc48-d123flip` logs uncompressed (9.4) |
| final-state gates (after everything) | production sentinels `sentinels_prod.py` **13/0/10/7/0**; witness `work-sent150-*` **21/0/2/7/0**; `prod_cfg_gate` **PASS 28**; broken symlinks **0/0/0** (`*_final_20260925.log`) |
| end state | **635 G free** (637 before the 2.4 GiB pin restore); `sbnd_xin` 147 G; `~/tmp` 73 G |

### 9.2 Cold pins: archived, not deleted

A pin is kept because an arm that survives *ran* on it. Rerunning that arm needs the pin;
reading the arm does not. `archive_pins_20260925.py` puts every pin the census KEEPs into
`~/tmp/pin-archive-20260925/cold-pins-20260925.tar.zst` (`zstd --long=31`), except:
- the four PERMANENT production pins;
- `d123-libpin`, the pin of the SBND substrate `d123base`;
- every live prefix (`d117 d118 d119 wcfm d125flip claude-`);
- anything mapped by a process, with a process cwd inside it, or written in the last 2 h.

That leaves 53 pins: 285435 files, 656032 symlinks, 272620 inodes.

**Correction, same day: one of the 53 was not cold.** `d115/libpin_d115` is the pin that PDHD
production `d116hflip` ran on, as did the PDVD release source `d116vflip`: clus md5 `2efa7fa09325`,
`pdvd/docs/nf_sp_img_clus/figs/116_flip_gate.txt`. Doc pdvd/116's Repro runs its whole ladder,
including `d116hr2`, on the same pin. The census cannot show this, because its "survivors" list
subtracts production arms by design, so a production arm's own pin looks cold. The pin was restored
with `restore_pins_20260925.sh d115/libpin_d115` (`restore_pins_d115_20260925.log`), and is now
PERMANENT in the census and sweep, HOT in `archive_pins_20260925.py`, and PROTECTED (pdhd). The
archive still holds a copy. **Rule for round L: before archiving a pin, check the recorded pin of
every production arm in all three trees, not the census survivor list.**

The other 52 were checked the same way. PDHD `d102hcs`/`d109hstm` ran on `d102/libpin_d102`, and
SBND and PDVD production run on the four hot pins. The two ToT flip-evidence arms `q31ctl`/`q34tk1`
ran on the archived `p100/libpin_p100b` and `p34/libpin_q34`. They are evidence, not production:
rerunning them needs a restore first. The only scripts that name an archived pin are historical campaign drivers
under `sbnd_xin/scripts/` (`pr150_arm.sh`, `d102m_stage*.sh`, `d115/stage*.sh`, `d116/stageB_cell.sh`,
`d118/stageB_flip.sh`, …). None of the production runners names one. **Restore the pin before rerunning any of those drivers**: if
`LD_LIBRARY_PATH` points at a missing directory, the loader silently falls back to `local/lib` (M1). Before any unlink, the script:
1. froze every file's sha256 into `archive/records/cleanup-20260925/pins-cold/cold-pins.manifest.tsv`;
2. ran `zstd -t` on the archive;
3. extracted the archive in full and compared all 941467 rows.

Freed bytes are set-relative, and 2.29 GiB of these inodes are shared with hot pins, so they stay on disk under the hot pins' names. Each pin leaves a
`<pin>.ARCHIVED` stub naming the restore command. The census now keeps those stubs and never
treats `pin-archive-20260925` as a unit.

**Restore.** Run `./restore_pins_20260925.sh [pin ...]`. It extracts the whole archive to staging
(~35 GiB, because pins share hardlinks), verifies the requested pins against the manifest, moves
them into place and removes the staging copy. It was tested on `p82/libpin_p82`: 572/572 rows identical.

The archive was chosen *instead of* dedup for these roots, because both acting on the same root
would double-count. Dedup then ran on the 13 roots that stay on disk.

### 9.3 The sentinel suite re-baselined on production, and `pr150s0` released

On `work-*-d123lgoppr` + `d123lgflippr` the unchanged suite reads **13 PASS / 8 FAIL**:

| event | sentinel | the assertion that fails |
|---|---|---|
| 69314 | pr/125 K5 | 22 calib showers, window [14, 20] |
| 171572 | pr/123 r2 | `pf-orphan-guard-freed` does not fire; the muon is nevertheless a PF root |
| 315167 | pr/93 r4 | the 150.7 cm proton (613 MeV on pr150s0) is not a PF node |
| 72786 | pr/128 A control | cosmics are still out; `pass4_prox_guard` declines 3, where ≥ 4 is required |
| 393505 | pr/129 | Eν 546.3 is in its window; the 267 MeV muon reads 262 |
| 497311 | doc 84 r1 | the unbroken muon is split (459 + 749 MeV); the range fallback is not reached |
| 292643 | pr/130 B | the outcome holds (no π⁰, e⁻ < 200 MeV); the dvtx suppress line does not fire |
| 179369 | pr/130 B | no 112 MeV π⁰; the dvtx suppress line does not fire; `mu- 1042` becomes `e- 1437` |

**Attribution.** These 8 are exactly the recorded FAILs of pr/150's `tfull` cell, the trajectory
that doc 118 flipped into SBND production, minus 66366, which production now passes.
`docs/pr/150_figs/150_s3_sentinels_tfull.txt` was measured before the flip, and doc 116 line 454
warned that the flip "re-opens every downstream sentinel". The same trajectory family on the old
flash source, `pr150csp3bw`, fails 7 of the same set. So the d123 hit flashes and the light gate
add none. Evidence: `125_figs/125_sentinel_evidence_pr150s0_vs_d123.txt`, made by
`125_figs/125_sentinel_evidence.py`, plus the five suite outputs in `125_figs/`.

**Owner's ruling: "OPEN as doc 118 cost, release".**
- `scripts/analysis/d125/sentinels_prod.py` is now the production suite. It runs `pr127_sentinels.py`
  unchanged (M10), with the pr/149 PF tolerance, and adds the 8 events to its KNOWN_OPEN registry
  with the reasons in the table above.
- Result: **13 PASS / 0 FAIL / 10 OPEN / 7 INERT / 0 SKIP**. A NEW FAIL still fails the run.
- The 8 stay OPEN, not green, until the vertex-choice retune that pr/150 sec 1 names re-closes them.

**The pre-flip record.** `witness_sentinels_20260925.py` copied pr150s0's 27 sentinel events into
`work-sent150-{mcp1k,mcp2k,ncpi0,nuecc48}` (164 MB, PROTECTED, disk-only like sent97). On the
witness alone the suite reads 21/0/2/7/0, verdict-for-verdict identical to pr150s0.

**The release.**
- `pr150s0`'s PROTECTED line was retired, and it left `production` in `plan_20260925b.py`. That is a one-line fork at
  a new stamp, so pass 1's committed tier/keep records were never re-written.
- **INTERLOCK 16 held it on the first pass-2 plan**, because the new, still-unpushed `sentinels_prod.py` names pr150s0.
- The fix was the planner's own: push the record first (cdeceb5f), then plan with
  `LIVE_REF=` that head. There was no exception.
- The tier was then exactly the 4 arms, 9.78 GiB. They were frozen and released, rc=0.

### 9.4 Kept-arm logs and per-event configs: compressed, not deleted

`compress_logs_20260925.py` zstd's (`-6`) the following in kept sbnd arms: `stdout.log`,
`wct_pr_evt<N>.log`, `ql.stdout`, `wct_ql.log`, `img.stdout`, `wct_img.log` and
`.wct-cfg-evt<N>.json`. mtime is kept.

**Held, never compressed:**
- calib dumps (M13);
- the live pdvd/119 SBND gates;
- the logs of every sentinel arm (`pr150s0`, `d123lgoppr`, `d123lgflippr`, `sent97`, `sent150`),
  because `pr127_sentinels.py` greps `pr_evt<N>/*.log`;
- any file that is hardlinked, is a symlink's target, is open, or was written in the last 2 h.

**The confirm-time list gate caught one real move.** The witness arms were created after the plan,
so their logs appeared at confirm. The run refused (`compress_logs_20260925.refused.log`); the
witness logs were added to the sentinel hold and the plan re-run.

**`work-nuecc48-d123flip` was restored afterwards** (`uncompress_d123flip_20260925.log`: 12 files, 0
mismatches) and is now held. The live pdvd/119 gates link into its `g0/`. Their 36 links are all
to data files (frames, icluster, opflash, rse), so none of them read a compressed file, but a live
peer's link target should not change under it at all.

**Readers that must decompress first**, with `zstdcat` or `./uncompress_logs_20260925.sh ['work-*-<arm>']`,
which checks each restored file against the frozen sha256:
- `scripts/d123/{r3_evidence,r3_ql_compare,r6_rescue_ruling}.py`;
- `scripts/d124/pr_null_cmp.py`;
- the closed pr/8x–14x census scripts.

**A defect this lever would have created, fixed in the same change.** `archive_records_*`
treats every `*.zst` as heavy, meaning hashed but not carried. That rule dates from round G's
compressed calib dumps. So a later round retiring one of these arms would have silently lost its
logs from the record. `archive_records_20260925.py`, which the next round forks, now carries
`*.log.zst`, `*.stdout.zst` and `.wct-*.json.zst` as record layer; compressed configs follow ONEPER.
The rule is inert for pass 1's freeze, because no such file existed then.

### 9.5 What is left, and why

- **`sbnd_xin` 147 G:**
  - `d123base` is 51.5 G. It is the substrate: the only SP frames on disk (10.3 G) plus the group
    imaging `g*/icluster` (40.7 G) that lgop's Q/L links resolve into.
  - `d123lgop` is 26.6 G, `d123lgoppr` 18.0 G, the record layer `archive/` 14.8 G,
    `pr150csp3bw` 9.5 G (blind vertex scan B arm), `d115pr` 5.7 G (owner keep) and
    `input_files_reco1` 4.5 G.
- **`~/tmp` 73 G:**
  - about 30 G is held for the live peers: `d119inst` 10.6, `wcfm-gnn` 9.7, `d119wt` 7.0,
    `d117` 2.5;
  - the pin archive is 6.6 G, and the hot pins 3.5 G after dedup, plus the restored `d115/libpin_d115` (2.4 G);
  - `d115/arm_*` and `d116/arm_*`, 1–1.2 G each, are the run products of surviving pdvd/pdhd scan-source arms.
- **Round L options, not taken:**
  - regenerate `d123base`'s `g*/icluster` on demand from its frames (≈ 40 G, but lgop's links and a
    re-imaging cost stand in the way);
  - `pr150csp3bw` once pr/150's blind scan is closed (9.5 G);
  - `torchinductor_xqian` (1.3 G) once no torch peer is live;
  - the peers' ~30 G when their rounds close.

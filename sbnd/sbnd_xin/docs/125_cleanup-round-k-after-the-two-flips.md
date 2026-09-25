# 125 — cleanup round K (2026-09-25): retire the doc 123/124 ladders and the ToT campaign after two production flips

**Status (2026-09-25 11:40): the DIRECTORY release is EXECUTED.** It removed 3562 dirs, 369.97 GiB,
rc=0 in all three trees. Free space on `/home/xqian` went **120 G → 488 G**, and every tree ended
with **0 broken symlinks**, against a recorded pre-count of 0.

**Not yet run: the file level (sec 4), the orphan logs, the `~/tmp` sweep, the pin dedup and the
after-gates.** Right after the release the permission classifier refused this session's next
command, a read-only spot check of the production arms, as "irreversible deletion". Per the
standing rule (`feedback_classifier_refuses_bulk_delete`) the session did not retry. It handed the
rest to the owner as the sec 6 block, and sec 7 records only what is proven to have run.

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
| file level, sweep, orphan logs, dedup, after-gates | **not run by this session**: the classifier refused the next command. Sec 6 is the block. |

## 8. Carried forward to round L

1. **Re-baseline the sentinel suite on the d123 production arms** (`work-*-d123lgoppr` +
   `d123lgflippr`). Then `pr150s0` (10.5 GiB) can go.
2. **PDVD light dirs** (`<run>_light<evt>_<arm>`): ~12 GiB, of which ~4.8 are ToT-ladder light (`f3x18`, `g3x`, `q31/q32`). They need a grammar with an allow-list, because q35flip's `_tot`, the release builder's `_keep`/`_wf` and the flip evidence `_q32ti/_q35esc/_g31off` are read through config paths that a symlink closure cannot see.
3. The hit-flash `opflash_apa*.tar.gz` in the production arms (doc 123 sec 17.4 item 6; ~3 GiB per big lgop arm).
4. Round J's items that are still open:
   - the 120 shared pdhd frames (6.77 GiB);
   - the bee/mabc zips and `.wct-*.json`;
   - `restore_compress`'s sbnd family filter.
5. `pdvd/work/*_d117sp/gpu_mem_*.csv` is still being written today by a leftover monitor. It is inside a held arm.

# 100 — Cleanup round 2026-09-04: sbnd_xin retired, pdvd planned, ~/tmp swept

**Owner scope, verbatim:** *"I would like retire some work* file in ./sbnd_xin
and ./pdvd directory. We want to keep the latest production result as well as
their input. Please also clean up a bit the ~/tmp directory."*

Asked which depth, the owner chose **honour PROTECTED.txt** for sbnd_xin,
**option A** (substrate + shipped-flip gates + the live round) for pdvd, and
**plan-now-execute-later** for pdvd because a peer session is live in it.

| tree | before | after | released |
|---|---|---|---|
| `sbnd_xin` | 84 G / 199 work dirs | **71 G / 122** | 77 dirs, 12.6 GiB — **EXECUTED** |
| `pdvd/work` | 90 G / 5444 children | unchanged | 3661 dirs, 47.2 GiB — **PLANNED, not executed** |
| `~/tmp` | 132 G | **76 G** | 30 dirs, 56.7 GiB — **EXECUTED** |

The pdvd half is doc [pdvd/29](../../pdvd/docs/29_pdvd-work-dir-retire.md).

## 0. Repro block

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 scripts/retire/plan_20260904.py                 # 8 interlocks -> OVERALL: PASS
RETIRE_JOBS=12 python3 scripts/retire/archive_records_20260904.py   # integrity 77/77
./scripts/retire/retire_20260904.sh                     # DRY RUN
CONFIRM=yes ./scripts/retire/retire_20260904.sh         # executed 2026-09-04
CONFIRM=yes ./scripts/retire/sweep_tmp_20260904.sh      # executed 2026-09-04
```

State: `scripts/retire/state-20260904/plan.json`.
Records: `archive/records/campaign-close-20260904/` (246 MB, 1.41 GiB raw → 0.24 GiB gz, integrity **77/77**).

## 1. What was kept, and why

- `work-{mcp1k,mcp2k,ncpi0,nuecc48}-d97fv` — **latest production stage A (Q/L)**, 3067/3067, `ref/prod-2026-09-04`
- `work-*-d97fvpr2` — **latest production stage B (PR)**, 3067/3067
- `work-*-grp0825` — the imaging substrate; 10478 symlinks resolve into its `evt<N>`
- `work-ncpi0-d99r3prod{,pr}` — the `ref/prod-2026-09-05` flip's own arm, the newest operating point on disk
- `work-sent97-*` — the sentinel witness, 31/31 standalone
- every `PROTECTED.txt` line, including all doc-87 arms (their written release
  condition — a successor round re-establishing §6.1 and §6.2 at production
  scale — is still not met) and the display/sentinel block the owner chose to keep

## 2. What was released

Three superseded gate **epochs** of one chain, `doc 92 → doc 99 r1 → doc 99 r2`,
each superseded by the next link, with the last link (r3) **shipped**:
`d92gate{,pr}` 2.23 G, `d99fix{,pr}` 2.23 G, `d99r2*` (7 arms) 6.82 G. Plus
doc 95's `dbg25a/b-*` 1.03 G, doc 93/94's `stmfb8-*` 0.25 G, doc 91's
`91neg-*` and the `r2scan*` pair.

`d99r2wr` and `d99r2bothpr` are the **source side** of doc 99 round 3's flip
gate (`d99r3prod == d99r2wr`, `d99r3prodpr == d99r2bothpr`) — the evidence that
licensed cutting `ref/prod-2026-09-05`. **INTERLOCK 8** was added for exactly
this and refuses the round unless their member hashes are frozen into the
archive first.

## 3. INTERLOCK 5 earned its keep before anything was deleted

The first draft of `RETIRE_LABELS` contained two arms `PROTECTED.txt` pins, and
the planner refused the round rather than reporting a clean dry run:

- `work-probe178410a` — the **only** on-disk proof that mcp2k evt 178410's
  SIGSEGV is nondeterministic (`rc=0` / 683 MB at `-j 1` vs `rc=139` / 2403 MB
  at `-j 32`). A nondeterministic crash cannot be re-captured on demand.
- `work-dbg25a-d97prodchk` — part of doc 97 §9's "production default == the
  validated arm" set.

Both moved to KEEP.

## 4. THE DEFECT THIS ROUND CAUSED — read this before writing another planner

**Symptom.** `retire_20260904.sh` completed `rc=0`, then its own post-state line
reported **20 broken symlinks** where `INTERLOCK 3` had proved "0 would dangle".
100 dangling links in total, all inside `work-dbg25a-d97prodchk` — a
**PROTECTED** arm — because `work-dbg25a-ql`, which it borrowed its imaging
input from, had just been deleted.

**Root cause.** `INTERLOCK 3` (inherited unchanged from `plan_20260902.py` and
every planner before it) resolved each link with `os.path.realpath` and then
compared against `ROOT` spelled as `/nfs/data/1/xqian/toolkit-dev/...`. But
`/nfs/data/1/xqian/toolkit-dev` **is a symlink** to `/home/xqian/toolkit-dev`,
so `realpath` always returned the `/home/xqian` spelling and

```
os.path.relpath(realpath(link), ROOT).split(os.sep)[0]  ==  '..'
```

— never an arm name, for **any** absolutely-spelled link. The arm's links use a
*third* alias again (`toolkit/sbnd_xin → ../wcp-porting-img/sbnd/sbnd_xin`).

**Why it hid.** Most links in this tree are relative (`../work-X/evt7`), and for
those the check works, so it passed on every previous round and reported
plausible non-zero dangle counts when genuinely wrong. It was also one level
deep (`os.listdir`), while these links sit at depth 1 under an arm whose real
content is a sibling directory. A check that is right for the common case and
silently vacuous for the rest is the worst shape a guard can have.

**Fix.** `realpath` **both** sides and walk at full depth
(`plan_20260904.py` interlock 3). Proven against the exact link that was missed:

```
link : /home/xqian/toolkit-dev/toolkit/sbnd_xin/work-dbg25a-ql/evt22
OLD  relpath vs ROOT     : '..'              <- never an arm name
NEW  relpath vs realpath : 'work-dbg25a-ql'  <- CAUGHT
```

**Verification.** Re-run against the current tree: `INTERLOCK 3 PASS … (0 would
dangle)` with the fixed logic. The **pdvd** planner is immune by construction —
it resolves a target by scanning path *components* for the `<run6>_<idx>_<arm>`
grammar rather than by `relpath` against a root, so no alias spelling can hide a
target; this was tested on one relative and one `/nfs`-absolute link.

**Damage, bounded and recoverable.** Lost: `work-dbg25a-ql` (196 MB) and, inside
`work-dbg25a-d97prodchk`, the 20 `evt<N>` imaging-input links and the 105 MB of
`icluster-apa*.npz` behind them. **Intact:** that arm's 20 `ql_evt<N>` dirs with
their `mabc-*.zip`, `pctree-evt<N>.tar.gz` and logs — i.e. everything doc 97 §9
actually measures. The arm is no longer *re-runnable from its own directory*;
its *measurement* stands. Regenerable end-to-end: the source
`input_files_reco1/stm_tagger_feedback/debug-25evt-reco1.root` (708 MB), the 26
`staged-dbg25/` per-event dirs and `bee/dbg25/dbg25.manifest.tsv` (29 rows) all
survive, so `dbg25_stage.sh` → `dbg25_run.sh` → `d97_dbg25_arm.sh` rebuilds it.
**Not done — owner call**, because re-creating a just-deleted label is the M13
question this file exists to raise.

**Also fixed:** the driver's keep-guard used `work-*-d97fv`, which matches
`work-dbg25a-d97fv` and refused a legitimate target mid-run. It is now anchored
on the four sample names, and tested in **both** directions — 12/12 REFUSE on
production and protected arms, 7/7 ALLOW on this round's real targets.

## 5. Two sentinel scripts, and they answer different questions

`pr127_sentinels.py` is **30 PASS / 0 FAIL** on production (doc 98 retired the
105074 case there when it was shown production emits no PF tree for that event).
`d97_sentinels.py` still carries 105074, so it reads **30/1** on production and
**31/0** on `work-sent97-*` — which is exactly the job that witness was carved
out to do. Neither is stale; do not "fix" one to match the other. Post-round,
production is unchanged at 30/0.

## 6. ~/tmp: 132 G → 76 G

`~/tmp` is overwhelmingly **pinned library snapshots** (1.2–1.9 GiB copies of
`local/lib`, taken so a campaign survives a peer's mid-round `wcbuild`) and
cmake build trees. The records — arm logs, gate outputs, `.md` appendices —
are small text and were **kept in place**. The sweep removes only whole `lib*/`
snapshot dirs and build trees; it never touches a `*.log`, `*.txt`, `*.md`,
`*.json` or `*.zip`. A libsnap is regenerable from the commit its doc records;
a gate log is not.

Applied doc 98's rule — *a pinned binary goes with its arms, never before* —
per pin: dropped `doc28/lib_*` (18), `d31{,r3,r4,r5,r7}lib`, `d92gate-libsnap`,
`d99-libsnap`, `doc94c-libsnap`, `doc94r3b-libsnap`, `d99r2-cmake2` (5.4 G
cmake), `doc37/cmbuild` (5.1 G cmake), `m1gate`. Interlock A refuses
`CONFIRM=yes` until the arms those pins back are actually gone.

**Kept:** `d39/lib_d39` (LIVE, doc pdvd/39 §0 names it), `doc25gate` +
`doc25r12/13` + `pinlib*` (live peer), `d97b-libsnap` (the SBND production
binary), `d99r2-libsnap` (backs the surviving `d99r3prod{,pr}`),
`prod0901b-libsnap` (backs the surviving `em114*`), `d31r6lib`,
`doc37/lib_{pin,base,new}` (doc 37 cites them by md5 as its symbol
present/absent witness pair, and its gate arms survive), `doc87/lib-*`.

**AGE IS NOT LIVENESS, again.** `claude-25225/…/7117f9b1-…` is 18 GiB and had
not been written since 09-03 05:49 — which reads as a dead session — but
`claude --resume 7117f9b1-…` is **running**. The sweep derives live sessions
from `ps`, never from mtime, and protects every one.

## 7. Not done

- **pdvd is planned, not executed** — see doc pdvd/29. A peer session is live in
  `pdvd/work`; the owner chose to run it after that round closes.
- `work-dbg25a-ql` regeneration (§4) — owner call.
- The archive is gzip. doc 89 measured `zstd -19` at 3.6× aggregate over gz;
  `recompress_archive_20260901.py` would apply here for ~0.6 GiB across both
  new record trees. Not run.

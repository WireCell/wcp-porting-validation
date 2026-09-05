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
| `pdvd/work` | 90 G / 5444 children | **43 G / 1850** | 3661 dirs, 47.2 GiB — **EXECUTED 2026-09-05** |
| `~/tmp` | 132 G | **66 G at sweep completion** | 38 dirs, 66.4 GiB in two passes — **EXECUTED** |

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

## 6. ~/tmp: 132 G → 66 G

**The figure moves.** `~/tmp` was 66 G when the sweep finished and read 67G a few
minutes later, with `df` drifting 1 G in the same window: three live Claude
sessions and their background tasks write here continuously. Quote it as
"66 G at sweep completion", not as a steady state, and **re-measure rather than
diffing against this number** — that is the same drift doc 89 recorded when
09-01's DROP dirs vanished with no sweep having run.

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

## 7. ~/tmp pass 2 (owner-selected)

`sweep_tmp_20260904b.sh`, 9.7 GiB: `pinlib2..pinlib7` (zero citations anywhere;
`pinlib` unsuffixed IS cited by doc pdvd/25 and stays) plus `doc27` and `doc35`,
both closed rounds. **The cost, stated:** `pinlib{2..7}` are the pins behind
`work-*-doc25new{2..7}`, which are KEPT — so this deliberately departs from
doc 98's "a pin goes with its arms". Those arms keep their measurements but can
no longer be re-run against the exact binary that made them, only re-read. The
owner made that trade; the commit each pinned is recorded in doc pdvd/25.
## 8. The archive re-encode, measured — and why the estimate was wrong

Run on both trees. **sbnd_xin: 35 tarballs, 0.22 → 0.07 GiB (3.3×, 0.16 GiB
freed). pdvd: 2046 tarballs, 0.53 → 0.23 GiB (2.3×, 0.30 GiB freed).**
Total **0.46 GiB** — against an off-the-cuff "~4–5 G" estimate given mid-round.

The estimate was wrong because it read `archive/records` as 7.16 GiB of gzip.
**4.68 GiB of it was already `.tar.zst`** — doc 89 did that pass on 2026-09-01.
Only 0.22 GiB of gz above the 1 MB floor remained. *Check what a lever has
already been pulled on before quoting its size.*

pdvd's 2.3× is also below doc 89's 3.6× aggregate and far below its 16×
single-file probe, for the reason doc 89 already recorded: the win comes from
zstd seeing across near-identical files, and these are 3661 **small per-arm**
tarballs (median ~170 KB), not one big one. The floor had to drop to 0.05 MB
for anything to qualify at all.

**Consequence both record trees now carry, and every future round inherits:
the archive is MIXED CODEC.** pdvd 2046 `.tar.zst` / 1615 `.tar.gz`; sbnd_xin
2143 / 2308. Every interlock that looks for a record — `plan` interlock 5, the
archiver's `verify()`, the driver's interlock B — was written against
`.tar.gz` alone and would have refused the pdvd round for 2046 arms whose
records are present and valid. All three now accept either codec and treat
*both* being present as an error (an interrupted re-encode), not as a choice.
Re-verified: pdvd **3661/3661** across both codecs.

## 9. Not done

- **pdvd is planned, not executed** — see doc pdvd/29. A peer session is live in
  `pdvd/work`; the owner chose to run it after that round closes. The end-to-end
  re-run also demonstrated the peer protection working: `KEEP` moved 1516 → 1522
  dirs because the live session created six new `d39` arms since the plan was
  written, and the prefix rule absorbed them with no edit.
- `work-dbg25a-ql` regeneration (§4) — **DONE, see §12.**
- The PROTECTED display/sentinel block — **settled in §10**, measured not deferred.


## 10. Round C — the PROTECTED block, measured before it was released

Owner, later on 2026-09-04: *release the PROTECTED display/sentinel block*
(~9.8 GiB by my estimate). **Measuring it first cut that to 1.2 GiB, and the
measurement overturned two of my own claims.**

### The decisive test: which arms do the live manifests still resolve into?

`em_display/*manifest*.tsv` reference **420 distinct arms; exactly 10 still
exist** — `work-pr130r1-probe98-*` (4), `work-pr130r1-probe141-*` (2),
`work-pr134-f086-*` (4). Everything else the manifests name (pr117…pr142,
prod0825, prod0901, empre0901) was retired in earlier rounds. So:

| family | manifest-resolving? | verdict |
|---|---|---|
| `pr130r1-probe98/141`, `pr134-f086` | **yes — the only 10** | **KEEP** |
| `em114`, `em114c` | no | **RELEASED, 1.2 GiB** |
| `vtx105-base` | no — but see below | **KEEP** |
| `d97prodchk` | n/a | **KEEP** — cited by pdvd docs 31 and 37 |

### `em114`/`em114c`: a protection that was true of a *different* arm

Their PROTECTED ground was "live em_display manifests". False. My first
citation grep scored them "cited" by substring-matching `em114` inside
`work-em114c-prodnowdbg-mcp1k` — a **different arm, and one already deleted**.
Exact-arm matching shows no manifest references `work-em114-*` or
`work-em114c-*` at all. Released; records in
`archive/records/campaign-close-20260904c/` (integrity 6/6).

### `vtx105-base`: I was wrong, and the interlock caught me

I listed it for release on the ground that PROTECTED.txt's
`dl_vtx_training` citation was stale. **That was wrong — I grepped `scripts/`
only, and `dl_vtx_training/` is a top-level directory** (67 MB, `runs/vtx105/`
present). INTERLOCK 5 refused the round, which forced the check that settles it:
`vertex_labels/vtxscan-vtx105-{delta,mcp1k,mcp2k,mcp2k-auto,mcp2k-ragree,ncpi0,nuecc48}`
hold **878 hand-scan label files referencing `work-*-vtx105-base` 1724 times**,
and those arms exist. Hand-scan labels are scientific record (M13) and were
scanned against *those* dumps; a different operating point is never a
substitute — the same argument that refused `prod0825` in the 08-31 round.
**4.21 GiB stays.**

### A SECOND casualty of the §4 defect, found only by asking this question

PDVD docs 28 (§364) and 31 (§403) cite
`sbnd_xin/work-dbg25a-d97off/ql_evt16/pctree-evt16.tar.gz` as the **SBND side of
a cross-detector comparison**. `dbg25a-d97off` was on this round's retire list
and is **gone**. A same-event pctree survives at
`work-dbg25a-d97prodchk/ql_evt16/` but at a *different operating point*, so it
is not a substitute for a byte-level claim. The five `ql_evt{2,12,14,31,41}`
doc 31 uses are present.

**Why this was missed:** §4's blast-radius check scanned `sbnd_xin/docs` only.
`dbg25` is an SBND set (doc 95, `wire-cell-sbnd-reco1`) that **PDVD docs borrow**,
so the consumer lived in the other tree. *Scan both repos before releasing
anything either tree can cite.*

### ~/tmp pass 3: doc 87's libsnaps, 7.5 GiB

`doc87/lib-{flip,post,pre,tc}`. **Checked for duplication first and they are not
duplicates:** `libWireCellClus.so` md5 matches across `lib-pre`/`lib-post`/
`lib-tc`, but the full 14007-file directory rollups all differ — a
dedupe-to-symlinks would have been wrong. Four distinct binaries, regenerable
from the commits doc 87 records. doc 87's **arms stay PROTECTED and untouched**;
they become re-readable rather than re-runnable, the same trade taken for
`pinlib2..7`.

### Round C totals

`sbnd_xin` 71 → **70 G**; `~/tmp` 67 → **60 G**; `/home/xqian` free
408 → **494 G** across the whole day.


## 11. Final state, all four passes

| | start | end |
|---|---|---|
| `sbnd_xin` | 84 G / 199 work dirs | **70 G / 116** |
| `pdvd` | 90 G / 5444 work children | **43 G / 1850** |
| `~/tmp` | 132 G | **60 G** |
| `/home/xqian` free | 408 G | **541 G** |

**133 GB recovered.** Production verified sound at every step: `work-*-d97fv`
and `work-*-d97fvpr2` 3067/3067, `pr127_sentinels.py` 30 PASS / 0 FAIL, all 39
active `PROTECTED.txt` arms on disk, and — after the pdvd round — **0 broken
symlinks** in a 6245-link graph.

The pdvd execution also settled the peer question empirically: between planning
and confirm the live doc-39 session created **seven new arm families**
(`d39r2base`, `d39r2prov`, `d39r2unm`, `d39r2cfg`, `d39r2cfgc`, `d39r2ctl`,
`d39r2noprov`), KEEP moved 1516 → 1583 dirs, and the `d39` **prefix** rule
absorbed all of them with no edit. A name-list would have released seven live
arms.


## 12. `work-dbg25a-ql` regenerated, and the §4 damage is undone

Owner-authorised 2026-09-05. `scripts/d100_dbg25a_img_regen.sh`, 20 events at
`D100_JOBS=32`, **30 seconds wall**. `scripts/d100_dbg25a_regen_gate.sh` is
the re-runnable proof.

**Imaging only, and no write into the protected arm.** All 100 dangling links
resolve *through* `work-dbg25a-ql/evt<N>/`:

```
work-dbg25a-d97prodchk/evt16                    -> work-dbg25a-ql/evt16
work-dbg25a-d97prodchk/ql_evt16/icluster-*.npz  -> …/d97prodchk/evt16/icluster-*.npz
```

so recreating `evt<N>/` repairs all 100 with **no repair script and no M13
question** — the links were never wrong, their target had simply been deleted.
`work-dbg25a-ql/ql_evt<N>/` was deliberately **not** rebuilt: nothing points at
it, and `d97prodchk` carries its own.

**The driver had to be forked, and this is the reason.** `scripts/dbg25_run.sh`
pins `LD_LIBRARY_PATH` to `~/tmp/doc94r3b-libsnap` — **which this round's own
~/tmp sweep deleted** (dropped on the ground "backs `work-dbg25a-*`, which
retire"). A missing directory in `LD_LIBRARY_PATH` is *silently ignored*, so
running it unchanged would have fallen back to live `local/lib` with no warning:
the M1 shape exactly. The fork names a fresh pin (`~/tmp/d100regen-libsnap`,
snapshotted before launch against a peer's mid-run `wcbuild`) and refuses if it
is absent. `dbg25_run.sh` itself stays byte-untouched — it is the doc-95 record.

### The gate: this is a restoration, not merely a regeneration

The original ran 2026-09-02 under a pin that no longer exists, so "same
operating point" had to be *demonstrated*, not assumed. Two independent checks:

| check | result |
|---|---|
| npz count + byte total vs the archived manifest | **80 files / 105 545 029 bytes — exact** |
| per-event `wct_img_evt<N>.log` content vs the archived logs | **20 / 20 identical** |
| dangling links in `work-dbg25a-d97prodchk` | **0** (was 100) |

The byte total is the *only* number the record kept for the npz — they were a
DROPPED heavy class, so no per-file hash survives. The log comparison is what
supplies content-level evidence, and the record layer kept those in full.

**What had to be normalised away, and why none of it is physics:** the
`[HH:MM:SS.mmm]` line prefix; `TICK: N ms (this: N ms)` integer counters; and
the whole `I [ timer ] Timer:` block — wall/core-sec figures which are *sorted
by duration*, so their order varies run to run as well as their values. Nothing
else is stripped.

**The gate carries its own negative control**, because normalising in the safe
direction is easy and worthless: it compares evt2's archived log against evt4's
regenerated one under the *same* normalisation and fails loudly if they compare
equal. They do not. A gate that cannot fail proves nothing.

Conclusion: **the SBND imaging stage did not move between 2026-09-02 and
2026-09-05**, and `work-dbg25a-d97prodchk` is once again self-contained and
re-runnable. The pdvd doc 28/31 casualty in §10 is a *different* arm
(`work-dbg25a-d97off`) and is **not** restored by this — it was a Q/L-stage
product at its own operating point.

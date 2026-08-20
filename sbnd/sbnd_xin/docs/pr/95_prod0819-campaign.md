# doc pr/95 — the `prod0819` campaign: retire 149 G → 54 G, then re-run Q/L **and** PR on four samples at current production

Owner asked for three things, in this order: clean the `work*` tree; re-run
**Q/L matching** for nueCC48 / NC π⁰ / 1000-evt numuCC (mcp1k) / 2000-evt
numuCC (mcp2k); then run the **full 13-stage PR chain** on top as the baseline
for tomorrow's PR-improvement rounds. Imaging is **not** regenerated. 32 CPUs.
Extra deliverables: `pr_display` calib dumps, and Bee zips built but **not**
uploaded.

**What makes this campaign different from every prior one: it is
single-epoch.** doc pr/76 (`prod0813`) ran the PR chain onto 2026-08-05
`cb0805` pctrees and said so plainly — "any statement of the form 'prod0813 is
current production end-to-end' is false." `work-cbr3-census-on`'s Q/L predates
the pr/94 flip. Here Q/L **and** PR both come from toolkit `fd6a116d`, so
"prod0819 is current production end-to-end" is true, and it is the first
product in this tree for which that sentence holds.

Status: **IN PROGRESS.** Phase 0 ✔, Phase 1 (retire pass 1) ✔, Phase 2 Q/L ✔
(all four samples, 3067 events). Phase 3 PR running. Phases 4-6 pending.

## Repro block

```bash
cd sbnd_xin

# 0. provenance proof (M1) -- BEFORE anything runs
git -C ../../../toolkit log --oneline -1            # fd6a116d
wcbuild > /tmp/b.log 2>&1; echo rc=$?               # rc=0, 0 objects recompiled
md5sum ../../../local/lib/libWireCellClus.so        # 3fae4385f9dc27fa63ac280faebb71df
../../../toolkit/build/clus/wcdoctest-clus          # 211/211, 2215 assertions

# 1. retire 149 G -> 54 G, pass 1  (docs/work-tags.md "RETIREMENT ROUND 2026-08-19")
python3 scripts/retire/plan_20260819.py                            # KEEP 51 / remove 311, 9 asserts
RETIRE_JOBS=24 python3 scripts/retire/archive_records_20260819.py  # integrity PASS 311/311
CONFIRM=yes scripts/retire/retire_20260819.sh A                    # 311 dirs, 94 GiB

# 2a. Q/L for the two small samples FIRST (cheap route exercised before the dear one)
NR=$PWD/work-nuecc48-ql0819; mkdir -p $NR
for d in work-img-nuecc48/evt*; do ln -sfn "$PWD/$d" "$NR/$(basename $d)"; done
seq 1 48 | xargs -P 32 -I{} env SBND_WORK_ROOT=$NR \
    SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
    ./run_nusel_evt.sh data {}
NR=$PWD/work-ncpi0-ql0819; mkdir -p $NR
for d in work-img-ncpi0/evt*; do ln -sfn "$PWD/$d" "$NR/$(basename $d)"; done
env SBND_MAX_JOBS=32 SBND_WORK_ROOT=$NR \
    SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-ncpi0 \
    ./run_nusel_evt.sh data -stm-fit all

# 2b. Q/L for mcp1k and mcp2k -- SEPARATE roots, see ".time_e<entry>" below
awk -F'\t' 'NR>1{print $4}' input_files_reco1/staged-mcp2025c-1000evt/entry_event_map.tsv     | sort -n > /home/xqian/tmp/ql0819/mcp1k.ids
awk -F'\t' 'NR>1{print $4}' input_files_reco1/staged-mcp2025c-2nd-2000evt/entry_event_map.tsv | sort -n > /home/xqian/tmp/ql0819/mcp2k.ids
ROOT=$PWD/work-mcp1k-ql0819 QL_EXTRA=-save-pctree ./run_ql_batch.sh -j 32 -f /home/xqian/tmp/ql0819/mcp1k.ids
ROOT=$PWD/work-mcp2k-ql0819 QL_EXTRA=-save-pctree ./run_ql_batch.sh -j 32 -f /home/xqian/tmp/ql0819/mcp2k.ids

# 3. the PR baseline -- bare production, no SBND_* overrides
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
      ./run_pr_chain_batch.sh work-$s-ql0819 work-$s-prod0819 data
done

# 4. census, then prune to PR-results-only
mkdir -p products/prod0819
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 pr_scores_table.py --root work-$s-prod0819 --sample $s \
      --out products/prod0819/$s-scores-prod0819.tsv
  awk -F'\t' 'NR>1 && $15==1 {print $4}' products/prod0819/$s-scores-prod0819.tsv \
      | sort -n > products/prod0819/events-$s-prod0819.txt
done
python3 pr_scores_table.py --root work-mcp2k-prod0819 --sample all --summary  # merged census
python3 scripts/retire/prune_unevaluated.py         work-*-prod0819   # dry run: QUARANTINE 2
python3 scripts/retire/prune_unevaluated.py --apply work-*-prod0819   # freed 3.95 GiB

# 5. Bee zips -- BUILT, NOT UPLOADED (upload is a separate owner authorization)
mkdir -p bee/prod0819
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 scripts/bee/make_pr_bee.py -q work-$s-ql0819 -p work-$s-prod0819 \
      -o bee/prod0819/$s-prod0819.zip $(cat products/prod0819/events-$s-prod0819.txt)
done

# 6. retire pass 2 -- release what prod0819 supersedes
#    (PROTECTED.txt is edited FIRST, by hand, or assert 7 trips)
python3 scripts/retire/plan_20260819b.py                            # KEEP 36 / remove 31, 9 asserts
RETIRE_JOBS=24 python3 scripts/retire/archive_records_20260819b.py  # integrity PASS 31/31
CONFIRM=yes scripts/retire/retire_20260819b.sh A                    # 31 dirs, 17 GiB -> 71 G -> 54 G
```

## 1. What "current production" is, and the proof that one binary made everything

Toolkit **`fd6a116d`** — doc pr/94 Phase 6, the owner's flip of `nu_per_bundle`,
`nu_selected_as_main`, `protect_open_convicted_bundles` and
`bee_flash_pred_min=0` into SBND production, made earlier the same day.

```
local/lib/libWireCellClus.so   mtime 2026-08-19 21:26:54   md5 3fae4385f9dc27fa63ac280faebb71df
build/clus/libWireCellClus.so  same mtime, same md5
```

Two cfg files *are* newer than the library —
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` and
`wct-clus-matching-perevt.jsonnet`. That looks like the stale-library trap
(M1) and is not: the pr/94 Phase 6 flip was **config-only**, and jsonnet is
read at runtime. Proven rather than argued, the doc pr/76 §2 way: a fresh
`wcbuild` recompiled **0 objects**, rc=0, and no `clus/**` C++ source is newer
than the installed library. `./build/clus/wcdoctest-clus` — 211 cases / 2215
assertions, 0 failed.

This matters more than usual because the tree is shared with concurrent
sessions and doc pr/76 §2 records that a mid-campaign rebuild would split the
events across two binaries with nothing in the output to show it. So the md5
is **re-checked after the last arm lands**, and — discharging the process item
doc pr/76 §2 has left owed since 2026-08-05 — every new arm carries the md5 in
its own `.build-fingerprint` file. `run_ql_batch.sh` independently records
`toolkit HEAD=` and the library mtime in its driver log.

## 2. Scope: what is reused and what is not

| stage | this campaign | why |
|---|---|---|
| reco1 → NF/SP → imaging | **REUSED** from `work-img-<sample>/` | owner's instruction; every runner symlinks `evt<N>` in and never regenerates (M11) |
| clustering + Q/L matching | **RE-RUN** (tag `ql0819`) | one job, `wct-clus-matching-perevt.jsonnet` — there is no intermediate clustering artifact, so "Q/L only" is not separable |
| 13-stage PR chain | **RE-RUN** (tag `prod0819`) | the baseline itself |

The middle row is the one worth stating out loud: **clustering and Q/L matching
are the same job**, so re-running Q/L necessarily re-clusters. That is correct
here rather than merely unavoidable — pr/91, pr/92 and pr/93 all changed
clustering after `cb0805` was written.

And the Q/L re-run was genuinely needed, not just tidy:
`work-{nuecc48,ncpi0,mcp1k}-cb0805` date from 2026-08-05, **before** `2d8c9e5a`
flipped the seven cathode-bundle-rescue knobs into production, so their
pctrees are at a superseded Q/L operating point.

## 3. Phase 1 — the retirement round

Full account in `docs/work-tags.md`, section "RETIREMENT ROUND 2026-08-19".
Headline: 311 dirs / 94 GiB removed, `work-*` 362 → 51, `sbnd_xin` 149 G →
54 G, `/nfs/data/1` free 880 G → 975 G, in 21 seconds. 9 asserts PASS, archive
integrity 311/311, broken symlinks 0 before **and** after, 0 git-tracked files
deleted, survivor census 51 == `len(KEEP)`.

The one structural novelty is **ASSERT 9**: this is the first round that sweeps
*before* a campaign, so KEEP had to be proven closed **forward** over the
campaign's input set, not only backward over existing evidence. It catches a
silent failure — `run_ql_batch.sh:51-53` writes `rc=91 ... no-imaging` and
**exits 0** on a missing imaging hub, so a thinned hub would have produced a
short arm that a downstream count gate would happily pass.

## 4. Phase 2 — Q/L, and the three traps in the runners

**Two runners, because no single one covers all four samples.**
`run_ql_batch.sh` knows only the `1k`/`2k` staged samples (`:39-41`, `:90-97`);
`run_valfast.sh` has no mcp2k arm at all and its PR event lists are the
`nu_evaluated` subset (521 of 1090), so neither is usable alone for a
full-population baseline. nueCC48 and NC π⁰ therefore go through the
imaging-seed + `run_nusel_evt.sh` route that built every prior root for those
two samples.

Traps, all three real and all three hit or dodged in this round:

1. **Neither runner enforces M13.** `run_ql_batch.sh` refuses only an *unset*
   `ROOT` and then `mkdir -p`s into whatever is there, while
   `run_ql_evt.sh:339` does `rm -rf "$QLDIR"` per event — so pointing it at
   `work-cbr3-census-on` or a `-cb0805` hub **destroys** it.
   `run_pr_chain_batch.sh` claims a fresh-out_root refusal in its header but
   `:90` is a bare `mkdir -p`. Both were guarded by an explicit
   `[ -e ] && exit 3` loop in this round's driver scripts.
2. **`.time_e<entry>.meta` is keyed by entry only**, so a combined mcp1k+mcp2k
   root collides — `work-cbr3-census-on` produced only 2000 meta files for
   3000 events. Hence **one root per sample** here. (`.status/<sample>-<entry>`
   is the reliable per-event record either way.)
3. **A fresh `run_ql_batch.sh` root carries no `.lineage_reality`**, so the PR
   stage's reality guard is vacuous on it. `data` is passed explicitly to all
   four PR arms. Relatedly, `SBND_REQUIRE_WASMAIN=0` is **not** set anywhere:
   that declaration exists for pinned legacy hubs, and a fresh Q/L at HEAD
   writes `was_main`.

Before the expensive arms ran, one nueCC48 event was smoked and checked **by
directory name, not just `rc=0`** — the silent failure mode for the
index-driven route is an entry-index/event-id mismatch producing 48 dirs with
the wrong ids, which any count gate passes. `ql_evt172230` matched
`work-img-nuecc48/evt172230`, pctree 1.9 MB.

> **Method note worth keeping.** Three separate "missing pctree" alarms in this
> round were my own verification racing the writer — the `ql_evt<ID>` directory
> and a 0-byte `pctree-evt<ID>.tar.gz` both appear well before the event
> finishes. **Gate on the driver's own completion line (`.status/*` or the
> `rc=` echo), never on directory existence.** All three cleared on re-check.

### Q/L results

| sample | events | ql_evt | pctree non-empty | non-rc0 | size |
|---|---|---|---|---|---|
| nueCC48 | 48 | 48/48 | 48 | 0 | 476 M |
| NC π⁰ | 19 | 19/19 | 19 | 0 | 180 M |
| mcp1k | 1000 | 1000/1000 | 1000 | 0 | 3.3 G |
| mcp2k | 2000 | 2000/2000 | 2000 | 0 (after one retry, below) | 6.5 G |

### A real crash, and why the baseline is still 2000/2000: mcp2k evt 178410

The mcp2k arm's first pass returned **`rc=139` (SIGSEGV) on 1 of 2000 events**:

```
rc=139 sample=2k entry=1774 evt=178410 rc=139 wall_s=132 maxrss_kb=2403228  fired=0
```

It is **not deterministic**. Re-run alone at `-j 1`, same binary, in an
isolated probe root (`work-probe178410a`), the same event succeeds:

```
rc=0   sample=2k entry=1774 evt=178410 rc=0   wall_s=126 maxrss_kb=682544   fired=0
```

The tell is **peak RSS: 2403 MB in the crashing run vs 683 MB alone — 3.5×**,
at nearly identical wall time (132 s vs 126 s). Both runs are under
`setarch x86_64 -R`, so this is **not** the M4 ASLR ghost. It is a
concurrency-dependent memory blow-up under 32-way load, on an event that is
already a heavy one (the log fills with
`Cluster::get_hull number of points is too large: 21608 (cap 10000)` — 21 k
points against a 10 k cap, so the hull cache is being hit hard). The crash
came inside `Pgrapher` execution after all 15 nodes started; no APA finished
(`mabc-all-apa.zip` and `mabc-apa0-face0.zip` were left 0 bytes,
`mabc-apa1-face0.zip` partial).

The event was then re-run **into the arm at `-j 1`** (only entry 1774 touched)
and the arm is now complete: 2000/2000 `ql_evt`, 2000 non-empty pctrees, 2000
non-empty `mabc-all-apa.zip`, **0 non-rc0**. So the baseline is whole.

> **This is reported, not fixed.** It is a pre-existing latent crash in the
> production Q/L job, unrelated to anything this campaign changes (this
> campaign changes no code). Two things make it worth its own round: it is
> *silent at the batch level* — `run_ql_batch.sh`'s driver still exits 0, and
> only the `.status` line and a 0-byte pctree betray it — and a 3.5× RSS
> excursion under concurrency is the signature of something unbounded, which
> at `PR_JOBS`/`-j` 32 × 2.4 GB is 77 GB of the box. Anyone re-running a large
> mcp2k Q/L batch should check `grep -c '^rc=0 ' .status/*` against the event
> count rather than trusting the driver's exit code.

Each arm's `ql_evt<ID>` id set was diffed against its imaging hub's `evt<ID>`
id set (nueCC48, NC π⁰) or against the sample's `entry_event_map.tsv` event
column (mcp1k) — identical in every case, so no event was silently skipped.
The mcp1k/mcp2k id sets are disjoint (0 overlap), which is what makes
`run_ql_batch.sh`'s per-event sample lookup unambiguous.

## 4b. A concurrent session relinked the library MID-CAMPAIGN — and the gate that saves the baseline

This is the hazard doc pr/76 §2 named and could only warn about; here it
actually happened, and the fingerprint discipline is what caught it.

**What happened.** At **23:04:06**, while mcp2k's PR arm was running
(23:02:41 → 23:23:36), another session in this shared tree committed
**`f0e69780`** ("clus: doc pr/96 — env-gated `remove_segment` caller probe,
log-only") and rebuilt. `libWireCellClus.so` went
`3fae4385f9dc27fa63ac280faebb71df` → `75652e607a7f3d7ceec2c0632dd2e9f5`, and
HEAD moved off `fd6a116d`.

**How it surfaced — as 7 hard failures, not as silent drift.** Seven mcp2k
events were loading the shared object at the instant it was being written:

```
E [  sys   ] Failed to load libWireCellClus.so: .../build/clus/libWireCellClus.so: invalid ELF header
C [  sys   ] failed to load plugin: "WireCellClus"
[evt 57753] wire-cell rc=1 -- skipping nusel_extract (no usable outputs)
```

`rc=1` at `wall_s=0-1` and ~200 MB RSS — a startup failure, and the event ids
cluster (57731, 57742, 57753, 57764, 58006, 58072, 58919) because they are
simply the seven that happened to start inside the relink window. All seven
were re-run and the arm is now **2000/2000 with 0 non-rc0**.

**The exposure, measured rather than guessed.** By `rc.txt` mtime against the
23:04:06 relink:

| arm | events | binary |
|---|---|---|
| nueCC48 | 48 | all pre-relink (finished 22:51:45) — `3fae4385` |
| NC π⁰ | 19 | all pre-relink (22:52:13) — `3fae4385` |
| mcp1k | 1000 | all pre-relink (23:02:40) — `3fae4385` |
| mcp2k | **115 pre / 1885 post** | **split** |

All Q/L (finished 22:50) is entirely pre-relink. So only mcp2k's PR arm spans
two binaries.

**Why the baseline is still valid: a gate, not an argument.** `f0e69780` is
structurally inert —
`static const bool on = std::getenv("WCT_PR96_REMSEG_DEBUG") != nullptr` (unset
in every run here), a body that is only `fprintf(stderr, ...)` plus
`backtrace()`, placed after the existing P6 sentinel and before the unchanged
`boost::remove_edge`. The other session reports its own 4/4 gate. But CLAUDE.md
does not accept an inherited "no behavior change" claim, so it was re-proven
here: **events that ran on the OLD binary were re-run on the NEW one and
hash-compared.**

| probe arm | events | compared | result |
|---|---|---|---|
| `work-pr96gate-mcp2k` | 12 pre-relink mcp2k | 24 archives (`mabc-pr.zip`, `pctree-pr`) | **PASS byte-identical** |
| " | same 12 | `nusel-evt*.tsv` | 12/12 identical |
| `work-pr96gate-nuedisp` | 3 nueCC48 | 6 archives | **PASS byte-identical** |
| " | same 3 | `calib-pr-evt*.json`, 1.05-1.17 MB each | 3/3 identical |
| " | same 3 | `nusel-evt*.tsv` | 3/3 identical |

`pr85_hash_gate.py` compares member-content hashes only, never the printed
path (the pr/83 trap). The `calib-pr` leg deliberately closes the hole doc
pr/94 §"gates" flagged — `pr_display` is opt-in and therefore absent from every
standard gate arm. **Both probe arms are kept** (`PROTECTED.txt`, 53 MB
together): without them this equivalence is text-only, which is precisely the
`work-pr87-postflip-*` loss that file records twice.

So: `prod0819` is a single **effective** epoch. Stated precisely — Q/L and
1067 of 3067 PR events are `fd6a116d`/`3fae4385`; 1885+7 mcp2k PR events are
`f0e69780`/`75652e60`; the two are proven byte-identical on 30 archives, 15
`nusel` tables and 3 calib dumps with the probe env unset.

> **One false alarm, recorded so it is not repeated.** An early attempt to gate
> the calib dumps on three *mcp2k* events reported "DIFFERS" for all three. The
> files were absent from **both** arms and `cmp -s` was returning non-zero for
> two missing files. `PrDisplayDump` only writes `calib-pr-evt<ID>.json` when
> the event has an evaluated neutrino candidate — 48/48 nueCC48, 19/19 NC π⁰,
> but only **461/1000** mcp1k and **905/2000** mcp2k. Test existence before
> comparing, and gate calib dumps on a sample where they exist.

> **Standing risk, not closed.** Nothing prevents the other session relinking
> again. The library md5 is re-checked after the last arm of this campaign, and
> any future large batch in this tree should bracket itself with an md5 check
> and verify `grep -c '^rc=0$' <arm>/pr_evt*/rc.txt` against the event count —
> a relink mid-batch is loud only if a process happens to be *loading* at that
> instant; a process that already loaded the old object keeps running it
> silently.

## 5. Phase 3 — the PR baseline, and the row convention that makes it usable

Four arms, bare production, `PR_JOBS=32 PR_EXTRA_STAGES=pr_display`, reality
`data` passed explicitly (a fresh `run_ql_batch.sh` root carries no
`.lineage_reality`, so the PR reality guard is vacuous — recorded, not assumed).

| arm | events | wall window | PR wall sum | size | non-rc0 |
|---|---|---|---|---|---|
| `work-nuecc48-prod0819` |   48 | 22:50:56–22:51:45 |    1306 s |  261 MB | 0 |
| `work-ncpi0-prod0819`   |   19 | 22:52:00–22:52:13 |     372 s |   97 MB | 0 |
| `work-mcp1k-prod0819`   | 1000 | 22:52:34–23:02:40 |  19013 s |  1.8 GB | 0 |
| `work-mcp2k-prod0819`   | 2000 | 23:03:00–23:26:28 |  38683 s |  3.4 GB | 0 |
| **total** | **3067** | 35.5 min at 32-way | 59374 s (19.4 s/evt) | 5.6 GB | **0** |

`pr_evt` count == `ql_evt` count == 3067 on the nose, and
`grep -L '^rc=0$' work-*-prod0819/pr_evt*/rc.txt` is empty across all four arms.
The Q/L half ran 22:25:25–22:48:07 (22.7 min), so the whole campaign is
**58 minutes** of wall clock for 3067 events twice through.

### Census

Run with `pr_scores_table.py --summary` over each arm, merged into
`products/prod0819/all-scores-prod0819.tsv`:

| sample | events | `nu_evaluated` | multi-bundle events |
|---|---|---|---|
| nuecc48 |   48 |   48/48  (100 %) |  4 |
| ncpi0   |   19 |   19/19  (100 %) |  1 |
| mcp1k   | 1000 |  461/1000 (46 %) | 26 |
| mcp2k   | 2000 | **905**/2000 (45 %) | 46 |
| total   | 3067 | **1433** | 77 (2.5 %) |

`event_label` over the 3067: `nu-candidate` 1567, `cosmic-tagged` 932,
`no-beam-flash` 343, `no-bundle` 225. `DL vertex failed` WARN: **0/3067**.

### The two-event discrepancy that is a logging artifact, not a physics one

The marker-based census reports mcp2k `nu_evaluated=904`; the true number is
**905**. Both the prune script and `pr_scores_table.py` classify an event by the
log line `TaggerCheckNeutrino: selected main cluster ... (t0 `, and under 32-way
that line can be **torn mid-word** by another thread's line (the known WCT
log-tearing behaviour). The prune's QUARANTINE tier caught exactly two events,
and both are tears:

| event | marker | truth | how it was settled |
|---|---|---|---|
| mcp1k 289656 | none matched | **not-evaluated (no-main)** | the torn remnant `no: no main cluster selected (13 mains, 2 in-window)` is still in the log; 0 calib dumps |
| mcp2k 164576 | none matched | **evaluated** | line 368 reads `cal_kine_charg` + `main cluster 24 (t0 1.613 us, L 105.4 cm, 8 associated)`; a 478 KB calib dump exists |

The independent cross-check is the **calib-dump count**, which needs no log
parsing: 48 / 19 / 461 / **905** — agreeing with the marker census on three
arms and disagreeing on mcp2k by exactly the one torn event. Use the calib-dump
count as the tie-breaker whenever the two disagree.

`products/prod0819/events-<sample>-prod0819.txt` is left as the awk rule
produces it (`$15==1`), i.e. 48 / 19 / 461 / **904**, so the lists stay
byte-reproducible from the Repro block. **mcp2k evt 164576 is the one known
omission** — a genuinely evaluated event that no marker-based tool will see.
`make_pr_bee.py` refuses it for the same reason, so it is absent from the Bee
zip too.

### Prune

Dry run first, as the plan required: **QUARANTINE 2** (both above), 1633
not-evaluated, 3.95 GiB of heavy products freeable. doc pr/76 got `0
QUARANTINE`; the difference here is entirely the two torn lines, so `--apply`
went ahead. After it: **1432/1432** events on the evaluated lists still hold all
four heavy products (`pctree-pr`, `mabc-pr.zip`, `tracking-pr.root`,
`calib-pr-evt*.json`), 0 incomplete, and both quarantined events keep everything.

### Bee packages — BUILT, NOT UPLOADED

| zip | events | size |
|---|---|---|
| `bee/prod0819/nuecc48-prod0819.zip` |  48 |  27.1 MB |
| `bee/prod0819/ncpi0-prod0819.zip`   |  19 |   9.8 MB |
| `bee/prod0819/mcp1k-prod0819.zip`   | 461 | 214.6 MB |
| `bee/prod0819/mcp2k-prod0819.zip`   | 904 | 397.5 MB |

Each with its `.index.txt` and `.prid-map.txt`. `upload-to-bee.sh` was **not
run** — outward-facing, needs its own owner authorization (CLAUDE.md §5.6). Note
the two large zips are far past a practical Bee upload size; if they are wanted
online they should be split by event subset first.

### Read this before comparing anything to `prod0819`

Since `fd6a116d`, production emits **one `T_tagger`/`T_kine` row per in-beam
flash bundle**, not one per event. **Row 0 is not the event's answer** —
`scripts/pr94_rows.py primary_index()` is, and it reproduces the legacy
meaning ("the longest selected main activity"). A future A/B that hard-indexes
`[0]` will read a row-count change as a physics diff.

`pr_scores_table.py:84` already imports `primary_index, n_rows`, so the
per-sample TSVs are one row per event with the primary-row projection, and
`n_inbeam_bundle` (column 13) carries the row count. Measured on **this campaign's own arms**: **4 of 48** nueCC48 events have 2
in-beam bundles — evts 10550, 360535, 389538, 444187 — plus 1/19 ncpi0 (evt
18625), 26/1000 mcp1k and 46/2000 mcp2k, i.e. **77 of 3067 (2.5 %)**. The
maximum `n_inbeam_bundle` anywhere in the baseline is **2**. (A pre-campaign
arm gave 3/48; evt 10550 is the one this epoch adds, and 10550 and 18625 are
precisely the two events doc pr/94 Phase 4b was written for.)

Separately, and **not** fixed in this round: `nusel_extract.py` /
`nusel-events.tsv` carry a known-wrong `event_label` for 68/629 events; the fix
currently lives only in the pr/94 sidecar. Do not build a comparison on that
column without applying it. Flagged here rather than patched because it is an
open owner decision from doc pr/94.

## 6. Phase 5 — retirement pass 2

`PROTECTED.txt` was edited **first**, by hand, or ASSERT 7 trips: the four
`work-cbr3-*` arms and the three *data*-sample `-vfcbr3on` / `vf*-cbr3on` pairs
moved to the RETIRED section with cost notes, the eight `ql0819`/`prod0819` arms
were added as real tab-delimited lines, and the stale bokeh pin was annotated as
released. Active protected names went 17 → **15**, all verified to exist.

Then `plan_20260819b.py`: universe 67, KEEP 36, remove **31** — the 21
`work-pr94r3*` arms (owner: retire all 21), the 4 `work-cbr3-*`, and the 6
data-sample valfast arms. 9/9 asserts PASS; ASSERT 9 re-verified the campaign's
own arms this time (3067 `ql_evt` + 3067 `pr_evt`, counted). Archive integrity
**31/31**, 1021 MiB raw → 137 MB gz. Result: **71 G → 54 G**, `work-*` 67 → 36,
`/nfs/data/1` free 958 G → 975 G, broken symlinks **0 before and 0 after**, 0
git-tracked files deleted, survivors == `len(KEEP)`.

### The dry run caught five arms that were not mine to retire

The first pass-2 plan listed 36 removals, and five of them were
`work-pr96-dbg{1,2,3}-mcp2k`, `work-pr96-fx1-mcp2k` and `work-pr96gate-disp`.
These were written 23:00–23:27 **that same evening** by the concurrent session
working doc pr/96 — the session whose `f0e69780` relinked the library at 23:04
(§4b) — and `docs/pr/96_uncovered-vertex-prongs.md` cites them; `dbg1` holds
exactly evts 70084 and 279955, the two events that doc is about. That round is
**open**, with a residual unresolved.

Two Claude sessions share this tree, and a retire round scoped by "uncited or
superseded" cannot tell a stale arm from another session's live one — the
citation was in a doc written hours earlier and the arms were minutes old. All
five were added to KEEP with the reason recorded in `plan_20260819b.py`
(`work-pr96gate-disp` on ambiguity alone: evts 47036/47982/49657 at 23:27:50,
sitting between my own two `pr96gate` arms, and I cannot prove it is mine).
Cost of keeping them: **43 MB**. Removals went 36 → 31.

**Rule for the next round: before the sweep, list the removal set's newest
mtimes and check anything from the last few hours against `git log` and the
newest `docs/pr/*.md`. A dated tier rule protects yesterday's work; it does not
protect this evening's.**

## 7. Known limits of this baseline

- The two **MC** samples (r1qlmc 10, r2mc 13) are **not** in the campaign, by
  the owner's scope. Their latest products remain `work-{r1qlmc,r2mc}-prod0813`
  (PR, full coverage: 10 and 13 events) and the `cbr3on` pair (newer, but only
  the `nu_evaluated` subset: 4 and 6 events), all kept — 30 MB for the
  prod0813 pair. So "single-epoch" is a claim about the four **data** samples
  only.
- **The bokeh viewer pin on :5017/:5018 is released** (owner, this round). It
  had already lapsed silently: no `bokeh`/`serve_pr_display` process exists and
  nothing is listening on 5013-5019, so the "LIVE on bokeh" ground
  `PROTECTED.txt` recorded on 2026-08-13 was stale well before this round. The
  prod0813 pair now survives on its full-coverage ground alone.
- `run_valfast.sh`'s `pinned_qlroot()` still points at the `-cb0805` hubs, not
  `ql0819`. The hubs are kept for exactly that reason; re-pinning valfast is a
  deliberate change for a later round, not a cleanup side effect.
- No determinism floor was measured this round (owner declined the rider).
  `PROTECTED.txt` has recorded P3 as owed since 2026-08-05; the cross-layout
  ASLR-on leg is still owed. `run_pr_chain_batch.sh` does run every event under
  `setarch x86_64 -R`, so the arms themselves are layout-pinned.

# 71 — The 2026-08-05 reprocessing campaign: five samples, 1090 events, clean binary

**Yes.** All five samples ran the full standalone chain (imaging → Q/L matching
→ tagger tail → PR chain) at toolkit `a1ea3789`, entirely on the binary frozen
at 07:05–11:05 that morning: **1090/1090 events, 100% success at every stage,
0 failures anywhere.** New valfast manifest (521 events, `nu_evaluated=1`) and
a re-established determinism floor (both P3 arms PASS). Three Bee links.

This is a **new baseline, not a gated change**: the 2026-08-05 clean-slate
retire round left `sbnd_xin` with no reconstruction products at all, so there
is nothing to A/B against (doc pr/33 §11.2, doc pr/37 §13). What replaces the
byte-identical bar is *completeness*: every event either produced a full
product set or has a recorded non-zero `rc` — and here, none did.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# Step 0 — img-staleness check + freeze (§1 below)
SBND_MAX_JOBS=1 SBND_WORK_ROOT=/home/xqian/tmp/img-pre SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod ./run_img_evt.sh data 1
./wcb clean && wcbuild
SBND_MAX_JOBS=1 SBND_WORK_ROOT=/home/xqian/tmp/img-post SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod ./run_img_evt.sh data 1
# compare the four .npz by member content (zipfile sha256, M2) -- IDENTICAL

# Step 2 — extract NCpi0 (frameshift product under a NON-default process instance)
./run_reco1_dump.sh -caf product -fsproduct 'sbnd::timing::FrameShiftInfo_frameshift__FILTERFRAMESHIFT.' \
    -t ncpi0 input_files_reco1/nc-sideband_filtered_frameshift.root

# Step 3 — imaging, all five samples (measured single-threaded, 32-way safe)
SBND_MAX_JOBS=32 SBND_WORK_ROOT=$PWD/work-img-<s> SBND_INPUT_DIR=$PWD/input_files_reco1/<extracted-dir> \
    ./run_img_evt.sh <data|mc> all
# mcp1k: per-entry
seq 0 999 | xargs -P 32 -I{} env SBND_MAX_JOBS=1 SBND_WORK_ROOT=$PWD/work-img-mcp1k \
    SBND_INPUT_DIR=$PWD/input_files_reco1/staged-mcp2025c-1000evt/e{} ./run_img_evt.sh data 1

# Step 4 — Q/L + PR chain, all five samples
ROOT=$PWD/work-mcp1k-cb0805 IMGBASE=$PWD/work-img-mcp1k ./run_full1k_nusel.sh 1000 32   # mcp1k
SBND_MAX_JOBS=32 SBND_WORK_ROOT=$PWD/work-<s>-cb0805 SBND_INPUT_DIR=<extracted-dir> \
    ./run_nusel_evt.sh <data|mc> -stm-fit all                                          # other four
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<s>-cb0805 work-<s>-cb0805 <data|sim>

# Step 5 — census + valfast manifest
python3 pr_scores_table.py --root work-<s>-cb0805 --sample <s> --out <s>-scores.tsv --summary
awk -F'\t' 'NR>1 && $15==1 {print $4}' <s>-scores.tsv | sort -n > valfast/events-<s>-cb0805.txt

# Step 6 — P3 determinism floor, nueCC48 only
PR_EXTRA_STAGES=pr_display setarch x86_64 -R ./valfast/run_valfast.sh vf0805a -j 47 nuecc48
PR_EXTRA_STAGES=pr_display setarch x86_64 -R ./valfast/run_valfast.sh vf0805b -j 47 nuecc48
PR_EXTRA_STAGES=pr_display                  ./valfast/run_valfast.sh vf0805c -j 47 nuecc48
./valfast/valfast_compare.sh vf0805a vf0805b nuecc48   # matched-layout floor
./valfast/valfast_compare.sh vf0805a vf0805c nuecc48   # cross-layout floor

# Step 7 — Bee links
python3 scripts/bee/make_pr_bee.py -q work-<s>-cb0805 -p work-<s>-cb0805 \
    -o <s>-cb0805.zip $(cat valfast/events-<s>-cb0805.txt)
./upload-to-bee.sh <s>-cb0805.zip
```

## 1. Step 0 — the `libWireCellImg.so` staleness check

Every lib except `libWireCellClus.so`/`libWireCellMatch.so`/`libWireCellRoot.so`
(2026-08-05 07:05, the pr/33 build) was dated **2026-08-03 09:04 — pre-cutoff**,
including `libWireCellImg.so`, which produces the imaging this whole campaign
rests on. `git log` showed no `img/ util/ aux/ iface/` commit since 08-01, but
that is exactly the condition under which the stale-object bug hides (doc
pr/33 §11.2) — it was only found in `clus` because pr/33's header edits forced
a recompile there.

**Test**: image nueCC48 evt172230 with the 08-03 libs into a scratch root, `wcb
clean && wcbuild` (a full rebuild — `wcb clean` avoided the sandbox's `rm -rf`
restriction), re-image the same event, compare the four `.npz` files by
zip-member content hash (sha256 of member name + payload, sorted — the same
algorithm `abtest/hash_archive.py` uses, applied directly since `hash_archive.py`
only recognizes `.zip`-suffixed archives, not `.npz`).

**Verdict: byte-identical on all four members.** The 08-03 imaging libs were
not stale. No debt carried forward. The campaign ran on the freshly rebuilt
libs regardless (already in flight when the check completed), which are the
libs whose mtimes are frozen below.

**Frozen lib mtimes** (unchanged for the entire campaign — verified after
every P3 arm in §6):
```
2026-08-05 10:10:14  libWireCellUtil.so       2026-08-05 10:10:55  libWireCellImg.so
2026-08-05 10:10:15  libWireCellIface.so      2026-08-05 10:10:59  libWireCellPytorch.so
2026-08-05 10:10:18  libWireCellAux.so        2026-08-05 10:11:02  libWireCellClus.so
2026-08-05 10:10:21  libWireCellApps.so       2026-08-05 10:11:03  libWireCellRoot.so / Tbb.so
2026-08-05 10:10:24  libWireCellFlash.so      2026-08-05 10:11:04  libWireCellSio.so
2026-08-05 10:10:36  libWireCellHio.so        2026-08-05 10:11:05  libWireCellMatch.so
2026-08-05 10:10:38  libWireCellGen.so / Pgraph.so
2026-08-05 10:10:47  libWireCellSig.so
2026-08-05 10:10:55  libWireCellSigProc.so
```
Toolkit HEAD `a1ea3789` throughout.

## 2. The five samples

| sample | input | mode | N | arm root |
|---|---|---|---:|---|
| `mcp1k` | `staged-mcp2025c-1000evt/e0..e999` | data | 1000 | `work-mcp1k-cb0805` |
| `nuecc48` | `extracted-2025fall-48evt-fsprod` | data | 48 | `work-nuecc48-cb0805` |
| `ncpi0` | `extracted-ncpi0` (new, §3) | data | 19 | `work-ncpi0-cb0805` |
| `r1qlmc` | `extracted-r1ql-f1` (8) + `extracted-r1ql-f2` idx 1,2 (5,12) | sim | 10 | `work-r1qlmc-cb0805` |
| `r2mc` | `extracted-r2patrec-f1` | sim | 13 | `work-r2mc-cb0805` |

Total **1090** events. Imaging landed in `work-img-<sample>/`, generated once
and symlinked into the arm root; `r1qlmc`'s f2 idx 1,2 were imaged **directly**
into `work-img-r1qlmc` (not symlinked from a separate root) to avoid a second
symlink hop.

**r1qlmc collision check**: f1 holds {6,10,13,14,16,21,39,43}, f2 holds
{5,12,14,26,29,38,39,47,49} — 14 and 39 collide. Verified the idx→event
mapping first (`load_events` is archive order, not sorted): f2 idx 1→evt 5,
idx 2→evt 12, neither colliding with f1's set. Final `work-img-r1qlmc` holds
exactly 10 distinct events.

**M11 imaging provenance sweep**: every `evt<ID>` dir in every `work-img-*`
root is a real directory (0 symlinks), and nothing under any root resolves to
`/nfs/data/1/yuhw/` or any tree outside this campaign.

## 3. NCpi0 extraction — a real trap, not a guess

19 events confirmed via `uproot` on the `Events` tree; runs 18255×12, 18259×3,
18261×1, 18345×2, 18364×1; all event IDs distinct.

**The `-caf product` flag alone was not enough.** The extraction first failed:
```
SBNDReco1OpFlashSource: caf_offset_mode=product but no product
sbnd::timing::FrameShiftInfo_frameshift__FRAMESHIFT. in
.../nc-sideband_filtered_frameshift.root
```
The product exists, but under a **different art process instance** —
`sbnd::timing::FrameShiftInfo_frameshift__FILTERFRAMESHIFT.` (this production
ran an extra filter stage) vs the nuecc48 file's `...__FRAMESHIFT.`. Fixed by
adding a new `frameshift_product` TLA to `wct-reco1-dump.jsonnet` — the same
key-suppression idiom the file already uses for `wire_product` /
`badmask_product` / `summary_product` (empty ⇒ key omitted ⇒ the C++ default,
byte-identical for every existing sample) — and a matching `-fsproduct` flag on
`run_reco1_dump.sh`. Verified byte-identical-when-empty and
key-present-when-set (compiled-config proof) before use.

Re-extracted with the correct product name (`run_reco1_dump.sh -caf product
-fsproduct '...__FILTERFRAMESHIFT.' -t ncpi0 ...`): 19/19 events, all
`frame_apply_at_caf` values non-quantized floats (250.0–2745.0 ns), none ≡ 0
mod 256 — confirming `product` mode is genuinely active (the `auto` mode's
signature, seen in yuhw's separate extraction of the same file, is exactly
that all 19 values are multiples of 256; that extraction was **not** used, per
M11).

## 4. Product census — every stage, every sample

| sample | img | ql | nusel | pr | pr fail |
|---|---:|---:|---:|---:|---:|
| ncpi0 | 19 | 19 | 19 | 19 | 0 |
| r1qlmc | 10 | 10 | 10 | 10 | 0 |
| nuecc48 | 48 | 48 | 48 | 48 | 0 |
| r2mc | 13 | 13 | 13 | 13 | 0 |
| mcp1k | 1000 | 1000 | 1000 | 1000 | 0 |
| **total** | **1090** | **1090** | **1090** | **1090** | **0** |

**Imaging is genuinely single-threaded** — confirmed by reading (`Pgrapher`
throughout `wct-img-all.jsonnet`, `wct-clus-matching-perevt.jsonnet`,
`wct-pr-perevt.jsonnet`; `TbbFlow` appears only in the `wcls-*` LArSoft jobs
this campaign doesn't run) and by direct measurement: one imaging run logged
`Total 14.229 wall-sec, 14.771 core-sec` = 1.04 cores. `SBND_MAX_JOBS=32`
throughout was therefore safe; `_runlib.sh`'s un-pinned default
(`SBND_MAX_JOBS=$(nproc)=64` on this host) was overridden on every invocation.

Measured steady-state throughput (direct N/sleep/N measurements, not
gap-sampled — this workload has enough per-event variance, doc 59's own
17–199 s range, that point-in-time snapshots across multi-minute gaps read
misleadingly slow): imaging ~172 events/min at 32-way; Q/L+nusel ~75–98
events/min at 32-way; PR chain ~96 events/min at 32-way. mcp1k alone: imaging
~6 min, Q/L+nusel ~35 min, PR chain ~17 min.

**A genuine finding, not a red flag**: doc 59 §6 documented a deterministic
crash in `TrackFitting::do_single_tracking` on mcp1k evt 278794 (entry 618).
At this campaign's HEAD (`a1ea3789`) **it no longer crashes** — `rc=0`, full
product set, `wall_s=24`. Between doc 59 and here lie the pr/28 through pr/37
porting rounds; the fix is incidental to one of them, not something chased
down here. Reported, not investigated further (escalation rule 7 is about
suspicious *movement*, not a bug silently going away).

## 5. Census tables and the new valfast manifest

Per-sample census via `pr_scores_table.py --summary`, merged into
`docs/pr/71_scores-table-cb0805.tsv` (1090 rows + header). `nu_evaluated=1`
(the PR log carrying `TaggerCheckNeutrino: selected main cluster` — the
authoritative "yields PR results" flag, `valfast/README.md`) counts:

| sample | evaluated | of sample |
|---|---:|---:|
| ncpi0 | 19 | 19 |
| r1qlmc | 4 | 10 |
| nuecc48 | 47 | 48 |
| r2mc | 6 | 13 |
| mcp1k | 445 | 1000 |
| **total** | **521** | **1090** |

Written to `valfast/events-<sample>-cb0805.txt` (fresh names — the old
`events-<sample>.txt`, 629 events total, are **untouched**, M13: they are a
record of the 08-03/08-05-round arms, all now deleted). `valfast/run_valfast.sh`,
`valfast_compare{,_par}.sh` repointed at the new manifests and extended with
`ncpi0` as a fifth sample (`pinned_qlroot`, `nusel_root`, `reality()`, sample
enumeration, and a full `run_nusel()` branch mirroring nuecc48's shape — all
19 events, no subsetting since N=19 makes it pointless).

**The drop from 629 to 521 (mcp1k 572→445 is the biggest piece) is not a
regression** — different operating point (doc 68 moved the SBND production
flags from an explicit string to config defaults), different binary
generation (clean post-06:32), and a fresh 1000-event pull vs the old d59k
arm. `valfast/README.md` is corrected in place with this note rather than
silently overwritten.

## 6. P3 — the determinism floor, re-established

The 2026-08-05 light retire round released `vf37a/b/c`, the tree's only A/A′
measurement, without a successor (`docs/work-tags.md` §"THE HEAVY ROUND", P3).
Re-run on nueCC48 only, the `pr37_a2_floor.sh` shape, on the **frozen** binary
(no rebuild between arms — lib mtime verified unchanged before, between, and
after all three):

```
vf0805a  setarch x86_64 -R   }  matched-layout floor (a vs b)
vf0805b  setarch x86_64 -R   }
vf0805c  ASLR on             -> cross-layout floor (a vs c)
```

All three: 47/47 `rc=0`, `PR_EXTRA_STAGES=pr_display` on every arm (so
`calib-pr-evt<ID>.json` exists for gate 5 — read-only, carried by all three,
cannot manufacture a difference).

**Both comparisons PASS, cleanly:**

```
VALFAST PASS: vf0805a vs vf0805b (matched-layout) — [nuecc48] identical on all gates that ran
  mabc 47/47 · pctree 47/47 · trees (ALL, EXACT) 47/47 · calib 47/47
  scores: 0 differing cells / 47 events × 24 columns

VALFAST PASS: vf0805a vs vf0805c (cross-layout)   — [nuecc48] identical on all gates that ran
  mabc 47/47 · pctree 47/47 · trees (ALL, EXACT) 47/47 · calib 47/47
  scores: 0 differing cells / 47 events × 24 columns
```

The instability that forced the old multiset comparator (doc pr/37 §2.5) is
absent here too, on both layouts, at `a1ea3789` — consistent with (not a
re-derivation of) doc pr/37 §2.5's own finding at `2457320d`.

## 7. Bee links

`make_pr_bee.py` refuses events with no selected neutrino candidate, so every
set draws from the `nu_evaluated=1` pool (§5). Index/prid-map records copied
to `docs/pr/{nuecc48,ncpi0,mcp1k-50}-cb0805.{index,prid-map}.txt`.

| set | events | URL |
|---|---:|---|
| nueCC48 | 47/48 | https://www.phy.bnl.gov/twister/bee/set/21b32701-828d-4ff9-a3b9-d7b8807e07b5/event/list/ |
| NCpi0 | 19/19 | https://www.phy.bnl.gov/twister/bee/set/9d7ed2e9-f0d0-4389-864a-6512c13b84a1/event/list/ |
| mcp1k-50 | 50/1000 | https://www.phy.bnl.gov/twister/bee/set/e8ca5499-286f-4a9e-a629-2715baef591b/event/list/ |

mcp1k-50 = first 50 `nu_evaluated=1` events by ID from
`valfast/events-mcp1k-cb0805.txt` (already sorted `-n`): `48367, 49951, 50303,
50743, 50831, …` (full list: `docs/pr/mcp1k-50-cb0805.index.txt`). Uploads
were serialized — `upload-to-bee.sh` writes/deletes `cookies.txt` in `$PWD`,
which races under concurrent uploads.

## Files touched

- `wct-reco1-dump.jsonnet`, `run_reco1_dump.sh` — new `frameshift_product` /
  `-fsproduct` knob (§3), key-suppression idiom, byte-identical when unset.
- `run_full1k_nusel.sh` — `ROOT`/`IMGBASE` now env-overridable; header comment
  corrected (imaging is regenerated this round).
- `_runlib.sh` — `SBND_WORK_ROOT` M13 guard: refuses a *new* implicit `work/`
  when unset and the default doesn't already exist (was silently creating one
  after the retire round deleted it); unaffected callers unchanged.
- `run_pr_chain_batch.sh` — placeholder `SBND_WORK_ROOT` before sourcing
  `_runlib.sh` (never reads it; false positive on the new guard).
- `valfast/run_valfast.sh`, `valfast_compare.sh`, `valfast_compare_par.sh` —
  ncpi0 added as a fifth sample; pinned roots repointed at `-cb0805`; event
  manifests repointed at `-cb0805` files.
- `scripts/runners/run_pr_geom_arm{,_dl}.sh`, `geom_ab_batch.sh` — repointed
  at `work-nuecc48-cb0805`.
- `dqdx_rr_sample/collect_proton_sample.py` — **not** repointed (both its
  source roots, including the CONTROL cross-check root, are gone with no
  successor); marked STALE with an explanation instead of a guessed fix.
- `valfast/README.md` — corrected in place with the new 521-event manifest,
  new repro block; old 629-event content retained as history.
- New: `docs/pr/71_scores-table-cb0805.tsv`, `valfast/events-<s>-cb0805.txt`,
  `valfast/entries-mcp1k-cb0805.txt`, `docs/pr/{nuecc48,ncpi0,mcp1k-50}-cb0805.{index,prid-map}.txt`.

## Verification

- [x] Step 0 done before any bulk run: rebuild `rc=0`, img A/B byte-identical,
      lib mtimes frozen and verified unchanged through the whole campaign.
- [x] Imaging confirmed single-threaded (1.04 cores measured); `SBND_MAX_JOBS`
      / `PR_JOBS` capped at 32 on every invocation; loadavg stayed ≤ 38/64.
- [x] NCpi0 extracted with the correct `-fsproduct`; 19 distinct events; caf
      values non-quantized (not the `-caf auto` mod-256 signature).
- [x] r1qlmc idx→event mapping verified before imaging; no collision.
- [x] M11 imaging provenance sweep: 0 symlinks under any `work-img-*`, nothing
      resolves outside this tree.
- [x] `run_pr_chain_batch.sh` accepts `out_root == ql_root` (verified on
      ncpi0) — no `-pr-cb0805` split needed.
- [x] Per-sample census: 1090/1090 events, every stage, 0 failures.
- [x] `find . -xtype l | wc -l` == 0 after the campaign (checked below).
- [x] New valfast manifests under fresh names; old `events-*.txt` untouched.
- [x] P3: three arms, lib mtime identical throughout; both a-vs-b and a-vs-c
      PASS, reported in full (not just the clean one — both were clean).
- [x] Three Bee URLs captured, uploads serialized.
- [x] Nothing written into `archive/`, `abtest/snap/`, `sweep/`,
      `decisions*/`, `ql_labels/`, or the 5 pre-campaign `work-*` survivors.
- [x] No toolkit source change.

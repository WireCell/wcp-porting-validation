# SBND `work-*` tag index

Repro:

```bash
cd sbnd_xin
ls -d work* | wc -l              # 27 after the 2026-08-03 round (138 before it; 23 after
                                 #   2026-08-02, 254 / 155 GiB before that, 15 after 2026-07-30)
ls archive/*/ -d                 # 3 campaign archives + records/, 79 dirs
find . -xtype l | wc -l          # 0 -- MUST stay 0, see "the symlink hazard" below.
                                 #   MEASURE IT BEFORE A ROUND, NOT ONLY AFTER: it was
                                 #   284 going into 2026-08-03 and nobody had noticed.
python3 relink_tags.py           # dry-run repair after any move

# the 2026-08-03 retirement round (see that section below):
python3 scripts/retire/plan_20260803.py       # 3 tier lists + 4 safety asserts
python3 scripts/retire/lightcheck_20260803.py # light/SP coverage proof (BLOCKING)
python3 scripts/retire/archive_records_20260803.py  # archive/records/pr23-25-era-20260803/
scripts/retire/retire_20260803.sh V,1,2       # dry run of the removal list

# the 2026-08-02 retirement round (see that section below):
scripts/retire/materialize_20260802.sh        # pr/22 exhibit chain self-contained (dry run)
python3 scripts/retire/plan_20260802.py       # removal set + footprint + dangling-link dry run
python3 scripts/retire/lightcheck_20260802.py # SP+light coverage proof / exceptions
python3 scripts/retire/archive_records_20260802.py  # archive/records/pr-era-20260802/ (additive)
scripts/retire/retire_20260802.sh 1           # dry run of the removal list

# the 2026-07-30 retirement round (see that section below):
python3 scripts/retire/inventory.py       # size / symlink-dep / citation inventory
python3 scripts/retire/plan.py            # removal set + dangling-link dry run
python3 scripts/retire/archive_records.py # write archive/records/  (additive)
scripts/retire/retire_20260730.sh 1,2     # dry run of the removal list
```

Every tagged output dir is one arm of one A/B or one hand scan, produced by
`run_ql_evt.sh` / `run_pr_evt.sh` / `run_nusel_evt.sh -t <tag>` over one of four
event manifests:

| manifest prefix | what it is |
|---|---|
| `work-mcp10-<tag>` | the 10/30-event hand-scan set (imaging from `work-mcp10`) |
| `work-mcp1000-<tag>` | 30 events drawn from the 1000-event MC set (imaging from `work-mcp1000`) |
| `work-mcp1000b-<tag>` | a second 30-event draw from the same 1000-event set |
| `work-mcp1kall-<tag>` | **all 1000** events of that set, not a draw (doc 59; imaging from `work-mcp1000`) |
| `work-mcsim-<tag>` | the MC-truth sim set (truth dQ/dx, delta rays) |
| `work-nuecc48-<tag>` | the 48-event Lynn nueCC data candidate set (`samples/lynn-nuecc-rse.csv`; input `input_files_reco1/extracted-2025fall-48evt-fsprod`; imaging in the shared `work/`) |
| `work-vf<sample>-<tag>` | a **valfast** PR out-root over the 629-event `nu_evaluated=1` subset (`valfast/README.md`; sample ∈ mcp1k/nuecc48/r1qlmc/r2mc). TRANSIENT by contract — delete after the gate report. `-full` arms also create nusel roots `work-mcp1kall-vf<tag>`, `work-nuecc48-vf<tag>`, `work-r1qlmc-vf<tag>`, `work-r2mc-vf<tag>` |

**These arms are not reproducible.** A tag names a *config*, not a build. The
binary has moved many commits since most of them were written, and the SBND PR
chain is ASLR-non-deterministic on top of that. Re-running `-t <tag>` today
produces a *different* arm, so retiring a directory permanently retires the
ability to re-check the doc table it backs — the doc's stated PASS becomes the
only surviving record. That is why the classification below is by **who names
the directory**, not by age or size.

## The symlink hazard — read before moving anything

A tag dir does **not** hold a private copy of every event. Any stage the run did
not redo is an **absolute symlink** into an earlier tag's dir, often chained two
deep:

```
work-mcp10-dq48v3/ql_evt284349
    -> /nfs/.../sbnd_xin/work-mcp10-fvxy/ql_evt284349
    -> /nfs/.../sbnd_xin/work-mcp10-mainreal/ql_evt284349
```

So the tags form a dependency graph and **moving a tag breaks every dependent**.
Archiving the doc-29..49 tags below broke 1536 links. It failed silently in the
worst way: `stm_fv_census.py` reported `0 "contained" clusters` instead of 147,
because a missing pctree is a `continue`, not an error. `relink_tags.py`
rewrites broken links to wherever the tag now lives; run it after ANY move and
confirm `find . -xtype l | wc -l` is 0.

Verification that the move was faithful: `python3 stm_fv_census.py` after the
repair reproduces doc 49 §4 line for line (147 contained / 96 outside / 65 %,
median 2.88, p90 3.54, max 3.77, walls 23/61/4/8, agree 96/96), and
`stmon_stats.py` reproduces 30 events / 36 fitted clusters / 18561 fit points.

## RETIREMENT ROUND 2026-08-03 — the pr/23..pr/25 era + valfast, 111 arms, 27 GiB

**STATUS: EXECUTED 2026-08-03 (`CONFIRM=yes ALLOW_LIVE_JOBS=yes
retire_20260803.sh V,1,2`) — 111 dirs / 27 GiB removed, refused=0.**
Post-checks: relink `repaired=0 unresolved=0`; `find . -xtype l` = **0** (it was
**284** before the round — see "the 284" below); no git-tracked deletion;
`work*` 138 → **27** dirs; `sbnd_xin` 55 GB → **28 GB**; `/nfs/data/1` free
485 → **512 GB**. Exhibit check: `gapjump_probe.py` on
`work-pr22gap-b/pr_evt386948/mabc-pr.zip` reproduces doc pr/22 §6 exactly
(634 fit pts, 50/634 = 7.9 % uncovered, 33.3 cm across 7 stretches).

State at the start: **138 top-level `work*` dirs, 48 GB**, `/nfs/data/1` at
87 % with 485 GB free — **no disk pressure**; this round was preparation for the
next campaign, not an emergency. The tree regrew from the post-08-02 23 dirs /
19.5 GiB in ~24 hours: docs pr/23, pr/24 and pr/25 plus their valfast gate arms,
the master-merge gate (doc 69) and the test round (doc 70) all ran back to back
and none were retired in flight.

Round tooling (all in `scripts/retire/`, forked from the 08-02 round; state in
`state-20260803/`, lists in `tier{V,1,2}_20260803.txt`). **Do not re-run the
08-02 scripts for a later round** — `plan_20260802.py` hardcodes that round's
survivor list and would treat everything added since (`work-r1ql-*`,
`work-r2patrec-*`, the cath01 pair, `work-mrgB-post`, the live prod0803
campaign) as a removal candidate. Fork instead.

1. **`plan_20260803.py`** — three tiers with **different dispositions** (the
   08-02 round had one), an explicit PROTECTED set, a hard error on any
   unclassified dir, and four safety asserts. All four PASSed: **0** real SP
   frames, **0** `nusel_labels`/`ql_labels`/`decisions*` dirs, **0** git-tracked
   files, and **0** of the 13 029 symlinks outside the removal set resolve into
   it. Every cross-directory edge from a candidate points *outward* into a KEEP
   hub; the big inbound counts (`work-mcp1kall-vfprodoff` 2288,
   `work-nuecc48-poc0` 192) are self-referential.
2. **`lightcheck_20260803.py`** — the blocking pre-flight. **1270/1270** real
   `opflash_apa*.tar.gz` in the removal set are byte-identical to a surviving
   copy: `missing=0 differs=0`, and 0 matches needed the cross-family fallback.
   (The 08-02 round found 8 differing out of 25 112, so a clean sweep was not
   assumed.) It also refuses outright if an exception lands in a DROP-whole
   tier-V arm, which has no archive to fall back on.
3. **`archive_records_20260803.py`** — record layer of the 80-arm ARCHIVE set
   into `archive/records/pr23-25-era-20260803/<group>/<tag>.tar.gz` +
   `.links.txt` + `.manifest.tsv`: **818.4 MiB raw → 362 MB gz**. HEAVY
   (dropped) = pctree / mabc zips / calib / npz / clusters **+ opflash** (per 2).
   Integrity gate — tar members == manifest record count — **PASS 80/80**.
4. **`retire_20260803.sh V,1,2`** — dry run by default. Guards: tier-aware
   archive-present refusal, Bokeh interlock, PROTECTED-in-tier-file refusal, and
   a **new** live-batch interlock (below).

### The 284 — a pre-existing invariant violation this round fixed

`find . -xtype l` was **284**, not 0, when the round started. All 284 were
inside three tier-V roots (`work-nuecc48-vfprodoff` 192, `work-r2mc-vfprodoff`
52, `work-r1qlmc-vfprodoff` 40) — relative links to `evt*` dirs that were never
created there. That made "0 after the round" a *sharper* post-check than
`relink_tags.py`'s own `repaired=0`: a nonzero result would have meant something
outside the plan's model broke. It came back 0.

### Two things the round changed about the tooling

- **The live-batch interlock (new).** A `wire-cell`/runner batch was writing
  into `work-nuecc48-prod0803` throughout, and 26 GB of NFS deletes alongside it
  is exactly M5. The script refuses while such a batch runs, **scoped to
  `sbnd_xin` command lines** — this box is shared and an unrelated user's
  `wire-cell` (one was running out of `/nfs/data/1/xning`) must not be able to
  block our housekeeping. `ALLOW_LIVE_JOBS=yes` overrides; used here because the
  live jobs touched only PROTECTED arms and loadavg was 1.19 on 64 cores.
- **Work in flight is auto-PROTECTED.** `work-pr116962-nocosmicveto` appeared
  *during* the round, created by a runner reading `work-nuecc48-prod0803`. So an
  unclassified dir is not necessarily a classification bug. Rule now: unclassified
  **and** touched within `RETIRE_FRESH_HOURS` (default 6) ⇒ auto-PROTECT and say
  so loudly; unclassified and stale ⇒ hard error, classify by hand.

### PROTECTED — 3 dirs, never listed, never walked

`work-nuecc48-prod0803` and `work-vfnuecc48-prod0803` (the owner's 2026-08-03
campaign, named explicitly so no glob change can pull them in), plus the
auto-protected `work-pr116962-nocosmicveto`.

### KEEP — 24 dirs, 20.2 GiB

BASE/HUB `work`, `work-mcp10`, `work-mcp1000`, `work-mcp1kall-d59k` (18.6 GiB,
never touched); the pr/22 exhibit chain `work-oc19scan-old` +
`work-pr22gap-{a,b,c,input}`; `work-mcp1kall-cath01` + `work-nuecc48-cath01`
(pr/12); `work-nuecc48-{base,nuf,prsmoke,prsmoke2}`; `work-r1ql-*` +
`work-r2patrec-*` (doc 67); the git-tracked `work-stmcamp-d66new` label stub;
and `work-mrgB-post`.

**`work-mrgB-post` is held deliberately** (168 MB) — it is the current-production
48-event baseline until `work-nuecc48-prod0803` completes, at which point
prod0803 supersedes it. Revisit next round.

### TIER V — valfast, 34 dirs, 20.19 GiB — DROPPED whole, no archive

`valfast/README.md` states the contract: *"valfast arms are transient. Record
the `valfast_compare.sh` summary (with tags) in the round doc, then DELETE the
`work-vf*-<tag>` and `work-*-vf<tag>` roots."* Every arm's compare summary is in
docs pr/23, pr/24 or pr/25. Largest: `work-mcp1kall-vfprodoff` 4.9 GB, then
eight ~1.68 GB `work-vfmcp1k-*` arms.

The old production baseline pair `work-vfmcp1k-prod{off,on}` (3.4 GB) went with
them, on the owner's 2026-08-03 call: the pr/24 `iso_endpoint` and pr/25 trio
defaults flipped ON that day, so as an A-side it was already pre-flip and stale.

**Exception — 3 arms were ARCHIVED instead of dropped**, because a citation scan
across `docs/`, `valfast/`, `*.py`, `*.sh` found **zero** references, so their
gate result existed nowhere else: `work-vfmcp1k-pr24i0` (1123 MB),
`work-vfnuecc48-pr24r3a` (71 MB), `work-vfmcp1k-pr24r3a` (49 MB).

### TIER 1 — docs pr/23–pr/25, 73 dirs, 6.27 GiB — archived, then removed

All three campaigns are CLOSED and **shipped with the SBND production default
ON** (pr/23 §9 flip 2026-08-02; pr/24 `iso_endpoint` 2026-08-03; pr/25 all three
2026-08-03).

| group | dirs | GiB | what it was |
|---|---|---|---|
| `pr25` | 22 | 4.26 | doc pr/25 cathode-rejoin / TGM-veto / shower-topo gates, incl. the two 1.66 GB `pr25s3r2-{dbgall,on50b}` arms |
| `poc48` | 6 | 1.26 | the pr/23+24 48-event PR baseline hub `work-nuecc48-poc0` + the `poc48*` protect-over-clustering arms |
| `pr23` | 24 | 0.60 | doc pr/23 protect-over-clustering pilot / cathode / v1 arms + the two `mcp1kall-pr23*` subsets |
| `pr24` | 21 | 0.16 | doc pr/24 isochronous-shower-trunk rounds a/b/c, r2 and r3 incl. the flip arms |

`work-nuecc48-poc0` was cited by three docs and was the 48-event hub, but it is
superseded by `work-nuecc48-prod0803` — a fresh run over the same manifest at
current defaults, fully self-contained (48 of its own SP frames, 100 self-links,
no outward dependency on poc0).

### TIER 2 — merge / test gate arms, 4 dirs, 0.34 GiB — archived, then removed

`work-mrgA-pre`, `work-mrgdet-r1`, `work-mrgdet-r2` (doc 69 master merge) and
`work-d70-eb` (doc 70 test round). Both gates are recorded in their docs.

### What survives, what is lost

**Survives**: every doc-quoted number (tsv, logs, `tracking-*.root` are in the
record tarballs), all 1270 light files (proven byte-identical to surviving
copies), all SP frames (none were ever in the removal set), and doc-table
re-checkability.

**Lost**: pctree/Bee-level re-analysis of these arms — the same class of loss as
the two prior rounds, and permanent, since a tag names a *config*, not a build
(see "These arms are not reproducible" above). For tier V it is total: those 31
arms leave no record layer at all, by the valfast contract.

### Totals

| | dirs | GiB |
|---|---|---|
| start | 138 | 48 (du) |
| KEEP | 24 | 20.2 |
| PROTECTED | 3 | ~1.0 and growing |
| TIER V dropped | 34 | 20.19 |
| TIER 1 archived + removed | 73 | 6.27 |
| TIER 2 archived + removed | 4 | 0.34 |
| archive added | — | +0.35 |
| **sbnd_xin after** | **27 work\*** | **28 GB total incl. `archive/` 4.2 GB** |

---

## RETIREMENT ROUND 2026-08-02 — the pr/11..pr/22 era, 231 arms, 127 GiB

**STATUS: EXECUTED 2026-08-02 (`CONFIRM=yes retire_20260802.sh 1`) — 231 dirs
/ 135 GiB removed, refused=0.** Post-checks: relink repaired=0 unresolved=0;
`find . -xtype l` = 0; no git-tracked deletions; `gapjump_probe.py` on
`work-pr22gap-b/pr_evt386948/mabc-pr.zip` reproduces doc pr/22 §6 exactly
(634 fit pts, 50/634 = 7.9 % uncovered, 33.3 cm across 7 stretches) — the
materialized exhibits are faithful.

State at the start: **254 top-level `work*` dirs, 155.1 GiB** (`du` 158.8 GB;
sbnd_xin 160 GB total), `/nfs/data/1` at 90 %. The tree regrew from the
post-2026-07-30 20 GB in ~62 hours (07-31 → 08-02): the doc pr/11–pr/22
campaigns ran roughly ten back-to-back generations of validation arms and
none were retired in flight. Eleven ~8.5 GB full-chain 1000-event arms alone
(`u17on1k(b)`, `cbron/off1k`, `vveto1k`, `isog1k`, `oc19on1k`, `cathA12*`,
`pi5cens`) hold 83 GiB. Everything from `rescue01` onward was never indexed
in this file — the tier table below is its only documentation.

Round tooling (all in `scripts/retire/`, forked from the July round;
state + tier list in `scripts/retire/state-20260802/` and
`tier1_20260802.txt`):

1. **`materialize_20260802.sh`** — made `work-oc19scan-old` self-contained so
   the doc pr/22 exhibits (`work-pr22gap-{a,b,c,input}`) survive: 10 `ql_evt*`
   symlinks into removal-set hubs replaced by `cmp`-verified copies (inner
   npz links repointed to their `readlink -f` targets in `work-mcp1000`/`work`)
   and `evt444187` repointed to its physical target `work/evt444187`.
   Executed 2026-08-02: repoint=1 copy=10 fail=0, 0 broken links after.
2. **`plan_20260802.py`** — survivors = the documented KEEP set + the pr/22
   exhibit chain; removal set = the other **231 dirs, 133.9 GiB real bytes**
   (owner decision: ALL pr-era arms retire, incl. the four pr11v3 census
   arms). Dangling-link dry run walks every symlink outside the removal set
   (12 837): **0** point into it. SP-data safety scan found 144 real
   `sp-frames.tar.bz2` in `work-nuecc48-{cathA12off,cathA12on,pi5wm}` — all
   144 verified byte-identical to the surviving `work/evt*` copies.
3. **`lightcheck_20260802.py`** — proves the round may DROP (not archive)
   `opflash_apa*.tar.gz` (new in this round's HEAVY class; the July round
   archived them): of 25 112 SP/light files in the removal set, 25 104 are
   byte-identical to surviving copies (hubs `d59k`/`nuf`/BASE); the **8**
   that differ (unique light variants in `work-cbr-det{1,2}`,
   `work-oc444187-{off,trace}`, evts 437699/444187) are force-ARCHIVED via
   `state-20260802/light_exceptions.txt`.
4. **`archive_records_20260802.py`** — record layer of all 231 arms into
   `archive/records/pr-era-20260802/<group>/<tag>.tar.gz` + `.links.txt` +
   `.manifest.tsv`: **6804 MiB raw → 2.1 GiB gz**. HEAVY (dropped) =
   pctree/mabc zips/calib/npz/clusters **+ opflash** (per 3). Integrity gate:
   tar members == manifest records, **159 251 == 159 251, 231/231 arms**.
   No `nusel_labels`/hand-scan/decision dir exists in any removal candidate
   (verified); the only labels under `work*` are in KEEP dirs and the
   `work-stmcamp-d66new` git-tracked stub.
5. **`retire_20260802.sh 1`** — dry run: **231 dirs, 135 GiB (du), refused=0**.
   Same guards as the July script (refuse-without-archive, Bokeh interlock,
   post-checks incl. `git status` for deleted tracked files).

### KEEP — 23 dirs, ~19.5 GiB

The 15 dirs of the July KEEP table, plus `work-mcp1kall-cath01` +
`work-nuecc48-cath01` (added 2026-08-01, doc pr/12), the
`work-stmcamp-d66new` label stub, and the pr/22 exhibit chain
`work-oc19scan-old` (37 MB, now self-contained) + `work-pr22gap-{a,b,c,input}`
(13 MB, cited by `docs/pr/22_track-fit-gap-jumping.md`).

### TIER 1 — 231 dirs, 133.9 GiB real (135 GiB du)

All campaigns are CLOSED (docs pr/11–pr/22 shipped/pushed). Grouped as in
`archive/records/pr-era-20260802/`; full list `scripts/retire/tier1_20260802.txt`.

| group | dirs | GiB | what it was |
|---|---|---|---|
| `pr11-census` | 20 | 7.5 | doc pr/11 1071-event population census v1→v3 + the crash-fix arms (`all73*`, `failfix`, `badallocfix`, `*-fix`) incl. the four `pr11v3` reference arms |
| `pr11-audit` | 46 | 1.6 | doc pr/11 latent-pattern audit + determinism/DL arms (`ab30*`, `audit*`, `det*`, the 21 single-event `469665`/`389538`/`52672` repeats) |
| `cath13-ccfeat` | 10 | 5.1 | docs pr/13–14 cathode-crossing Bee sets, `cath13` Q/L arms, `ccfeat300*` connector-feature census |
| `rescue01` | 4 | 0.1 | doc pr/14 cathode-bundle-rescue gate arms |
| `cbr` | 32 | 17.6 | docs pr/14+17 cathode-bundle-rescue 1k census (`cbron/off1k`) + the 16 `cbr286191*`/`cbr56463*` determinism probes |
| `vveto` | 4 | 8.7 | doc pr/15 separate() vertex_veto 1k census + `vv*rr` repeats |
| `nsc` | 6 | 0.6 | doc pr/16 nu_skip_cosmic size-guard arms (`nscon/off/base/rep`) |
| `isog-u17` | 26 | 27.8 | doc pr/17 unmerge (`u17*` incl. the two 8.5 GB 1k hubs) + isolated-grouping arms (`isog*`, `iso10550*`) |
| `nbl` | 5 | 0.5 | doc pr/18 iso-band nu guard arms (`nbl15`, `nbloff`, `nblrep`) |
| `oc19` | 13 | 9.5 | doc pr/19 isolated-absorb (`oc19*`, `oc444187-*`) + `work-oc19scan-new` (the OLD scan dir is KEEP) |
| `cathA12-b0` | 27 | 32.6 | doc pr/20 Part II: `cathA12*` A1/A2 census (three 8.5 GB 1k arms) + `b0*` cathode-kink-veto gates |
| `pi-partI` | 30 | 22.1 | doc pr/20 Part I: `pi0..pi5` demoted-main census/gates incl. the 8.5 GB `pi5cens` and four 2.7 GB `pi5cens-pr*` arms |
| `pr20x` | 5 | 0.04 | doc pr/20 Part X production-default determinism arms |
| `probes` | 3 | 0.1 | `d68smoke` (doc 68), `nuecc48-probe`, `rpg-ab31` |

**Survives removal**: every doc-quoted number (tsv, logs, `tracking-*.root`
are in the record tarballs), the 8 unique light files, and doc-table
re-checkability. **Lost**: pctree/Bee-level re-analysis of these arms (the
same class of loss as the July round; arms are not reproducible — a tag
names a config, not a build). The `pr11v3` retirement additionally idles
`pr11_br_filled_census.py`, `cathode_nu_census.py` and `stub_census.py`
until pointed at a future census arm.

### Totals (final, post-execution)

| | dirs | GiB |
|---|---|---|
| start | 254 | 155.1 real (158.8 du) |
| KEEP | 23 | ~19.5 |
| TIER 1 removed | 231 | 133.9 real (135 du) |
| archive added | — | +2.1 |
| sbnd_xin after | 23 work* | 26 total incl. archive (transient `work-vf*` A/A′ arms deleted separately per the valfast disposal rule) |

---

## RETIREMENT ROUND 2026-07-30 — 134 arms REMOVED, 35.9 GiB reclaimed

State at the start of the round: **149 top-level `work*` dirs, 54.9 GiB of real
bytes** (`du` says 56 GB; the difference is directory overhead — 208499 entries
under `work*`). `sbnd_xin` as a whole was 58.5 GB (60 GB after the archive was
written); `/nfs/data/1` was 87 % full, 481 GB free.

The round was **archive-then-remove**: the record layer of every retiring arm
was first copied into `archive/records/` (additive — nothing under `work-*` was
moved or rewritten), the removal list was reviewed, and then
`CONFIRM=yes scripts/retire/retire_20260730.sh 1,2` deleted tiers 1+2 on the
owner's approval. The script refuses to delete any arm whose `<tag>.tar.gz` is
not present in the archive, and refuses to run at all while a `bokeh serve`
viewer is up.

One arm survives as a stub: `work-stmcamp-d66new/nusel_labels/d66flip/` is
**git-tracked** (22 label json force-added past the `*.json` ignore rule), so
`git checkout` restored it in place after the removal — the directory now holds
nothing but those 152 K of hand-scan labels. No other removed arm held a tracked
file (`git status` showed 22 deletions, all of them that one dir).

**Result** (`scripts/retire/retire_20260730.sh 1,2`, `CONFIRM=yes`):
134 dirs removed, 0 refused; `relink_tags.py` dry run `repaired=0
unresolved=0`; `find . -xtype l | wc -l` = **0**; `work*` 149 dirs / 56 GB →
**15 dirs + the d66new label stub / 20 GB**, `sbnd_xin` 60 GB → **24 GB**; `/nfs/data/1` free
481 GB → **517 GB**. SP + light after the round: 6158 files / 1765 MB, intact.

### What was archived, and what removal costs

`archive/records/` (731 MB gz, 1587.9 MiB raw, 68856 files, README there) holds
per arm: a `<tag>.tar.gz` of every real file **except** `pctree-*.tar.gz`,
`mabc*.zip`, `calib-evt*.json`, `*.npz`; a `<tag>.links.txt` symlink map; a
`<tag>.manifest.tsv` of kept-vs-dropped bytes; and — verbatim as directories,
never tarred — the ten `nusel_labels/` hand-scan record dirs that live inside
removal candidates:

`work-mcp10-{d49son,d52on,d52ron,d55ton,d56bw}`,
`work-mcp1000-{d55ton,d56bw}`, `work-mcp1000b-{d55ton,d56bw}`,
`work-stmcamp-d66new` → `archive/records/labels/<tag>/nusel_labels/`.

Integrity gate: tar member count == manifest record-file count, 68856 == 68856,
all 134 arms.

**Survives removal**: doc-table re-checkability. Every number the docs quote
from these arms comes from `nusel-evt*.tsv`, the run logs or `tracking-stm.root`
— `bwgate_report.py`, `d60_ab_report.py`, `d66_flip_report.py`,
`d66_proton_sweep.py`, `p54_ab_report.py`, `mabc_step_totals.py`,
`stmon_stats.py` read only those.

**Lost on removal**, precisely — the loss depends on whether the arm re-ran the
Q/L stage or shared it:

- **91 of the 134 arms symlink their `ql_evt*` into `work-mcp1kall-d59k` or
  `work-mcp1000`** (every `stmcamp` and `d60` arm). For those, the Q/L-side
  products — `mabc-all-apa.zip`, the per-face Bee zips, the Q/L pctrees,
  `opflash_apa*.tar.gz`, `calib-evt*.json` — **survive in the kept hub**. What
  goes is only the PR side: `nusel_evt*/pctree-pr-evt*.tar.gz` and
  `nusel_evt*/mabc-pr.zip`.
- **43 arms own their `ql_evt*`** (the 10-event `mcp10`/`mcp1000`/`mcp1000b`
  `d52*`/`d53*`/`d55*` arms plus `trace51`, `m66*sb`, `d55pv` — 371 Q/L Bee
  zips). Those re-ran clustering, so their Q/L products are unique and go too.

So after removal `stm_fv_census.py`, `unmerge_crosser_audit.py`,
`stm_main_connectivity.py` and `nusel_extract.py`'s archive mode cannot be
re-run against any removed arm, and the scan viewer cannot display one.
Combined with "these arms are not reproducible" above, removal is permanent:
the doc's stated numbers, the tsv/logs/`tracking-stm.root` and the labels become
the only surviving record.

### SP and light data are untouched

The owner's one constraint. SP = `sp-frames.tar.bz2` and
`sbnd-sp-frames-anode{0,1}.tar.bz2`; light = `opflash_apa{0,1}.tar.gz`. Both
live in the BASE `evt*` / `ql_evt*` dirs (`work`, `work-mcp1000`, `work-mcp10`
— 4062 files, 1752 MB) and in the kept hub `work-mcp1kall-d59k` and the current
`work-nuecc48-*` arms. **No removal candidate holds an SP frame at all**, and
the per-arm `opflash_apa*.tar.gz` copies that some 30-event arms carry are
included in the archive tarballs, so no light product is lost either.
`input_files`, `input_files_reco1/` (1.7 GB) and `scan-d59k/` are not in scope.

### Dangling-link dry run — 0

The scar this file documents (1536 broken links, `stm_fv_census.py` silently
reporting 0 instead of 147) was checked mechanically, not argued: every symlink
under every **surviving** dir (`work`, `work-mcp1000`, `work-mcp10`,
`work-mcp1kall-d59k`, the `work-nuecc48-*` / `work-r1ql-*` / `work-r2patrec-*`
arms, plus `archive/`, `scan-*`, `nusel_display/`, `ql_scan/`, `docs/`,
`stm_campaign/`, `bee-*`, `showcase-*`) was resolved and none points into the
removal set. The removal set is closed under the dependency graph: the only
in-set arms with dependents are `work-mcp10-d55ton`, `work-mcp1000-d55ton`,
`work-mcp1000b-d55ton` (11 dependents each) and `work-mcp10-d49son` (1), and
every one of those dependents is itself in the set. After deletion, run
`python3 relink_tags.py` and confirm `find . -xtype l | wc -l` is 0 anyway —
`retire_20260730.sh` does both.

### KEEP — 15 dirs, 19.05 GiB (all that remains)

| dir | size | why |
|---|---|---|
| `work` | 2792 MB | BASE: data-sample imaging + **SP frames** + light + `ql_labels/` |
| `work-mcp1000` | 7127 MB | BASE: 1000-event MC imaging + SP frames; 14372 inbound links |
| `work-mcp10` | 96 MB | BASE: 10/30-event hand-scan set + `nusel_labels/` |
| `work-mcp1kall-d59k` | 8383 MB | HUB + LIVE: doc 59 production scan (`s59k`, labels), **18462 inbound links** — every `stmcamp`/`d60` arm's `ql_evt*` is a symlink into it. Its 2.5 GiB of `calib-evt*.json` is the largest non-BASE block deliberately not touched |
| `work-nuecc48-{base,nuf,prsmoke,prsmoke2}` | 745 MB | CURRENT: the 48-event Lynn nueCC campaign behind docs `pr/1`–`pr/10` |
| `work-r1ql-{f1,f1-nobw,f2,f2-nobw,first10}` | 173 MB | CURRENT: round-1 Q/L arms (doc 67), created 2026-07-30 |
| `work-r2patrec-{f1,f1-nobw}` | 191 MB | CURRENT: round-2 pattern-rec arms (doc 67), created 2026-07-30 |

Optional additions the owner may want to retire too, listed but **not** in the
tiers: `work-nuecc48-base` (151 MB, the partial earlier arm superseded by
`-nuf`) and `work-nuecc48-prsmoke` (8 MB, superseded by `prsmoke2`).

### TIER 1 — REMOVED 2026-07-30: 125 dirs, 25.14 GiB

Campaigns that shipped and closed. Full per-dir listing with sizes:
`scripts/retire/tier1.txt`.

| group | dirs | size | status |
|---|---|---|---|
| doc-63 STM improvement rounds — `work-stmcamp-{r0,r1,r1b,r2,r2full,r2fullb,r3full,r3fullb,r4afull,r4bfull,r5full,r5fullb,r5fullc,r5off,r5offb,r5offc,r1off,r2off,r3off,r3offb,r4aoff,r4boff,d64tf,d64lt,d64smoke}` | 25 | **20.30 GiB** | doc 63 SHIPPED + default ON, closed. Eight near-duplicate 883/1000-event `*full` arms at 2.3 GiB each are 19.5 GiB of the total |
| doc-60 TrackFitting single-point abort — `work-mcp1kall-d60{base,bw1,bw2,sr1,sr2,sfix,nr1,nr2,nfix,fix,fixchk,crash}` | 12 | 2.09 GiB | doc 60 FIXED + pushed (`2a821fd2`); determinism repeats and gate arms |
| docs 52–57 30-event arms — `work-mcp{10,1000,1000b}-{d49son,d52*,d53*,d55*,p54*,p55opt,p56off,d56bw,d57mip*,m66*,p65fin}`, `work-mcp10-{trace51,d52chk,d52trace,d53leg,m66*sb}`, `work-smoke-d55pv` | 78 | 2.67 GiB | docs 52/53/54/55/56/57/65 all closed; includes the four `nusel_labels` arms (labels archived) |
| `work-stmcamp-dbg1..9` | 9 | 0.08 GiB | ad-hoc debug probes of the doc-63 campaign |

### TIER 2 — REMOVED 2026-07-30 (owner approved): 9 dirs, 10.76 GiB

The doc-66 diffusion-revert campaign (`work-stmcamp-d66{old,new,fix,fixoff,oldtrace,newtrace,newtrace0,newtrace0b,newtrace5}`; list in
`scripts/retire/tier2.txt`). Closed and shipped like tier 1, but two reasons to
pause before deleting:

- `d66fix` / `d66fixoff` are the validation **pair** for the doc-66 §12 STM cut
  package (toolkit `c0501d7e`) which is **currently default ON**. If that
  default ever has to be re-validated at pctree level, this pair is the arm that
  did it.
- `d66new` is the shipped-revert arm and carries the largest surviving
  `nusel_labels/` (148 K, archived).

Keeping the `d66fix`/`d66fixoff` pair whole would have cost 5.27 GiB; the owner
chose to remove all nine (10.76 GiB). Their record layer — including `d66new`'s
148 K `nusel_labels/` — is in `archive/records/doc66-diffusion/` and
`archive/records/labels/work-stmcamp-d66new/`.

### TIER 3 — optional, not scripted: ~0.9 GiB

The three existing `archive/` campaign trees still hold their heavy layer:
`stm-docs40-49` 553 MB, `tgm-docs29-39` 229 MB, `aborted-d54` 132 MB of
pctree/mabc/calib/npz against 137 MB of records. Applying the same record-only
rule to them reclaims ~0.9 GiB. Not scripted here because they were already
curated once (2026-07-25) with the whole-directory convention.

### Totals

| | dirs | GiB |
|---|---|---|
| start | 149 | 54.9 |
| KEEP | 15 | 19.05 |
| TIER 1 | 125 | 25.14 |
| TIER 2 | 9 | 10.76 |
| archive added | — | +0.71 |
| **after tiers 1+2 (actual)** | **15** | **19.05** (`work*` 20 GB by `du`; sbnd_xin 24 GB incl. `input_files_reco1` 1.7 GB, `scan-d59k` 694 MB, `archive/` 1.8 GB) |

---

*Sections below describe the state at the 2026-07-25 consolidation. The BASE /
LIVE / CURRENT tables are superseded by the KEEP and TIER tables above; the
`archive/` and RETIRED tables remain current.*

## BASE — 3 dirs, 10026 MB (as of 2026-07-25; still current, see KEEP above)

**base input** — the imaging / PR products every tagged arm links into. Never delete: regenerating `work-mcp1000` alone is a 2000-event imaging run.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work` | 715 | 2795M | — |
| `work-mcp10` | 63 | 96M | — |
| `work-mcp1000` | 2000 | 7136M | — |

## LIVE — 7 dirs, 8.5 GB (as of 2026-07-25 — SUPERSEDED, retained for the "referenced by" columns)

> Currency warning: this table's "live" claims are from 2026-07-25. Every arm
> listed here except `work-mcp1kall-d59k` was **removed** on 2026-07-30 (tiers
> 1/2 above); its record layer is in `archive/records/`. Use the KEEP table for
> what exists; use this table only for the per-dir reference lists.

**wired into a running viewer** — the port-5011 `nusel_scan_viewer.py` command line names these as its current tag or as a `--prev` baseline. Deleting or moving one blanks the live scan.

The doc-56 `work-*-d56bw` arms are also live as `--prev` baselines of the doc-59
scan; they are listed under CURRENT below.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp1kall-d59k` | 2999 | 8.3G | `59_full1k-production-scan.md`, `nusel_scan_filter.py`, `run_full1k_nusel.sh`, `make_scan_bee.sh` — the port-5011 `s59k` scan (648 of its 999 tables) |
| `work-mcp1kall-d60crash` | 9 | 88K | `60_trackfitting-single-point-abort.md` — 1-event repro root for the evt 278794 abort (entry 618 only; only the pctree *tarball* is symlinked in, never the `ql_evt278794` dir, so a from-scratch Q/L rerun cannot write into the d59k record) |
| `work-mcp1kall-d60base` | 2728 | 1.2G | doc 60 §7 — pre-fix PR-only re-run of entries 0-430 + 618, used as the pinned determinism arm vs `d59k`. **Bee zips carry `runNo="0"`**: its `ql_evt*` hold only the pctree tarball, so `nusel_extract.py` could not run (rc=1 on every entry by construction, not a failure) — compare it `--archives-only` |
| `work-mcp1kall-d60nr1` / `-d60nr2` | 102 | 47M each | doc 60 §7 — two **un-pinned** (no `setarch`) repeats, entries 0-19, production config |
| `work-mcp1kall-d60sr1` / `-d60sr2` | 302 | 170M each | doc 60 §7 — two un-pinned repeats over the 60 STM-tagged entries; `d60sr1` doubles as the pre-fix arm of §6 gate 1 |
| `work-mcp1kall-d60bw1` / `-d60bw2` | 302 | 198M each | doc 60 §7 — two un-pinned repeats of the same 60 events with **pre-doc-56 `-no-bwonly`** (146 STM / 256 TGM tags), and the negative control against `d60sr1` |
| `work-mcp1kall-d60sfix` | 302 | 170M | doc 60 §6 gate 1 — **post-fix** arm, 60 STM-tagged events, byte-identical to `d60sr1` |
| `work-mcp1kall-d60nfix` | 102 | 47M | doc 60 §6 gate 2 — **post-fix** arm, entries 0-19, byte-identical to `d60nr1` |
| `work-mcp1kall-d60fixchk` | 5 | 2.0M | doc 60 §6.1 — evt 278794 with the fix in: `rc=0`, 8-bundle table, in-beam bundle tagged STM |
| `work-stmcamp-d66old` / `-d66new` | 1000 events each | — | `66_diffusion-revert-validation.md`, `55_dqdx-vs-rr-three-bundles.md` §12, `d66_flip_report.py` — the diffusion A/B: `d66old` = `DL/DT` 6.5781/13.1349, `d66new` = 4.0/8.8 (the shipped revert). **Same binary, same d59k pctrees, differing only in the runtime fit JSON** — arm identified by `SBND_TRACKFIT_JSON` and recoverable from each event log's `trackfitting_config=` line. Both 1000/1000 `rc=0`. Built with `stm_campaign/run_round.sh` + `STM_EVENTS`, so they carry the `work-stmcamp-` prefix despite being the full 1000-event manifest |
| `work-stmcamp-d66oldtrace` / `-d66newtrace` / `-d66newtrace0` / `-d66newtrace0b` / `-d66newtrace5` | 6 / 6 / 141 / 13 / 9 events | — | `66_diffusion-revert-validation.md` §12, `d66_proton_sweep.py` — TRACE-level (`SBND_WCT_LOGLEVEL=trace`) reruns for the STM cut-fixability study: the 6 scan-mistake events in both diffusion arms, every event with an accepted-STM (status-0) bundle, the torn-log redo, and the 9 proton-vetoed (status-5) events. Same arms as `d66old`/`d66new`, extra logging only — statuses verified identical. detect_proton TRACE lines must be read from the batch stderr sink `.log_<evt>.log` (the per-event file sink tears them deterministically) |
| `work-stmcamp-d66fixoff` / `-d66fix` | 1000 events each | — | `66_diffusion-revert-validation.md` §12.5 — validation arms for the doc-66 §12 STM cut package (toolkit `c0501d7e`): `d66fixoff` = package OFF (`-no-stm-d66cuts`, the byte-identical gate vs `d66new`, PASS all 1000), `d66fix` = package ON (production default; exactly the 4 target STM flips, plus 2 tsv-only `stmfit`-column diffs from log tearing — pctrees identical). Same binary and d59k inputs as `d66new` |
| `work-mcp10-d49son` | 43 | 29M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `52_isolated-grouping-fix-design.md`, `d52_ab_report.py`, `stm_main_connectivity.py`, `stm_merge_attribution.py` |
| `work-mcp10-d52ron` | 53 | 60M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |
| `work-mcp1000-d49son` | 32 | 23M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `d52_ab_report.py`, `stm_main_connectivity.py` |
| `work-mcp1000-d52ron` | 30 | 46M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |
| `work-mcp1000b-d49son` | 32 | 23M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `d52_ab_report.py` |
| `work-mcp1000b-d52ron` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |

## CURRENT — 52 dirs, 2109 MB (as of 2026-07-25 — SUPERSEDED, retained for the "referenced by" columns)

> Currency warning: every arm in this table was **removed** on 2026-07-30
> (TIER 1 above) except the `work-nuecc48-*` roots at the end, which are KEEP.
> Records in `archive/records/docs52-57-arms/`.

the campaigns still in flight: docs 52 (isolated grouping) and 53 (`real_cluster_id`), plus the `d55b`/`d55t` arms of doc 52 §13, the doc-54 perf A/B arms and the doc-56 beam-window-gate arms (`p56off` knob-off gate, `d56bw` = the new production default, served on :5011).

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-d52chk` | 12 | 4M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52off` | 52 | 60M | `d52_ab_report.py` |
| `work-mcp10-d52on` | 53 | 60M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp10-d52roff` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52rpoff` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52trace` | 2 | 2M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d53beeoff` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53beeon` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53dflt` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53leg` | 30 | 30M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53off` | 52 | 60M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53on` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53r` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d55boff` | 52 | 60M | — |
| `work-mcp10-d55bon` | 52 | 60M | — |
| `work-mcp10-d55toff` | 52 | 60M | — |
| `work-mcp10-d55ton` | 53 | 60M | — |
| `work-mcp10-p54base` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p54opt` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p55opt` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p56off` | 30 | 30M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp10-d56bw` | 30 | 25M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp10-p65fin` | 30 | 26M | `65_tgm-stm-perf-final.md`, `mabc_step_totals.py`, `profile_pr65.sh` |
| `work-mcp10-trace51` | 6 | 36M | `51_clustering-merge-attribution.md`, `stm_merge_attribution.py` |
| `work-mcp1000-d52off` | 30 | 47M | `d52_ab_report.py` |
| `work-mcp1000-d52on` | 30 | 47M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp1000-d52roff` | 30 | 47M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000-d52rpoff` | 30 | 47M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000-d53off` | 30 | 47M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp1000-d55boff` | 30 | 47M | — |
| `work-mcp1000-d55bon` | 30 | 46M | — |
| `work-mcp1000-d55toff` | 30 | 47M | — |
| `work-mcp1000-d55ton` | 30 | 46M | — |
| `work-mcp1000-p54base` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p54opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p55opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p56off` | 30 | 24M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp1000-d56bw` | 30 | 20M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp1000-p65fin` | 30 | 21M | `65_tgm-stm-perf-final.md`, `mabc_step_totals.py`, `profile_pr65.sh` |
| `work-mcp1000b-d52off` | 30 | 44M | `d52_ab_report.py` |
| `work-mcp1000b-d52on` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp1000b-d52roff` | 30 | 44M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000b-d52rpoff` | 30 | 44M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000b-d53off` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp1000b-d55boff` | 30 | 44M | — |
| `work-mcp1000b-d55bon` | 30 | 44M | — |
| `work-mcp1000b-d55toff` | 30 | 44M | — |
| `work-mcp1000b-d55ton` | 30 | 44M | — |
| `work-mcp1000b-p54base` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p54opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p55opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p56off` | 30 | 23M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp1000b-d56bw` | 30 | 19M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp1000b-p65fin` | 30 | 19M | `65_tgm-stm-perf-final.md`, `mabc_step_totals.py`, `profile_pr65.sh` |
| `work-smoke-d55pv` | 12 | 5M | — |

### Added 2026-07-29 — nueCC48 campaign roots (docs `pr/1_`, `pr/2_`)

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-nuecc48-base` | 92 | 154M | `pr/2_uboone-chain-gap-analysis-and-validation-plan.md` §5 — partial earlier arm of the 48-event Lynn nueCC candidate set (48 evt, 24 ql_evt, 20 nusel_evt) |
| `work-nuecc48-nuf` | 146 | 485M | `pr/1_beam-window-cosmic-vs-nu-division.md` (the run that created it, 48/48 rc=0), `pr/2_…` §5 — **complete** cosmic-tagger-tail arm: 48 evt (symlinks into shared `work/` imaging) + 48 ql_evt (with pctrees) + 48 nusel_evt + merged `nusel-table.tsv`/`nusel-events.tsv`; production NUF flags (unmerge_assoc in-log). 45 nu-candidate / 3 cosmic-tagged. The planned Track C PR arm symlinks from here |
| `work-nuecc48-prsmoke` | 2 | 8M | `pr/2_…` §5.3 — 2-event smoke of the PR neutrino stage (evt 172230, 444187; production tail + `tagger_check_neutrino`, rc=0, full PR Bee layers) |
| `work-nuecc48-prsmoke2` | 2 | ~5M | `pr/3_pr-skip-cosmic-and-outputs.md` §6 — single-event validation of the full PR output chain (evt 172230 nu-candidate + 444187 TGM skip-demo; 13-stage pipeline with `nu_skip_cosmic`, Bee prototype-parity knobs, BDT scorers, `tracking-pr.root` T_tagger/T_kine); runner `run_pr3_evt.sh` in the root.  Also holds the `pr/4` DL-vertex adoption arms: `nupr_evt172230_dl{,_rep}` (explicit SCN weights + determinism repeat), `nupr_evt172230_defaultdl` (inherits the new config default), `nupr_evt172230_dl_importfail` (the silent-fallback failure, kept as the exhibit); runner `run_pr3_evt_dl.sh`.  `pr/5` investigation arms: `nupr_evt172230_dl_trace2` (per-logger trace used to pin the PID/vertex chain; `_dl_trace` is the clobbered-log first attempt, kept), `nupr_evt172230_pufix` (verifies the SbndPrMagnifyTrackingVisitor fractional-channel fix, mabc bit-identical to the DL arm).  `pr/6` arms: `nupr_evt172230_dirweak{,_geo}`, `nupr_evt444187_dirweak` (dir_weak_use_score knob-on runs, all bit-identical to their pr/4 counterparts) |

### Added 2026-08-01 — cathode-crossing PR arms (doc `pr/12_`)

Both are subset re-runs of the doc pr/11 population arms at HEAD `3fe65876`, made only
so every number in `pr/12` comes from one binary (md5 `07a78447…` verified before and
after both). Selection came from `work-mcp1kall-pr11v3` / `work-nuecc48-pr11v3`, which
stay the population reference.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp1kall-cath01` | 54 | ~150M | `pr/12_cathode-crossing-neutrino-pr.md` — the 54 mcp1k candidates whose fitted trajectory and/or charge crosses the cathode, full 13-stage PR chain, 54/54 rc=0. Source of `docs/pr/12_cathode-census.tsv` and the `cath_spanned`/`cath_broken` Bee sets |
| `work-nuecc48-cath01` | 3 | ~10M | `pr/12_…` — the 3 nueCC48 cathode-crossing candidates (214469 spanned, 267597 the single genuine split, 437699 fit-stops-at-cathode), 3/3 rc=0 |

## `archive/tgm-docs29-39/` — 30 dirs, 257 MB

**TGM / LM / FV campaign.** every arm whose newest citation is doc ≤39 — the merge-aware TGM chain (docs 29-33), the LM tagger (34), interior FV and main-component-pairs (35-36), pctree provenance (38) and the FC/TGM x-y margins (39).

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-chord` | 43 | 3M | `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `32_tgm-component-rescue-fvz.md`, `35_tgm-interior-fv.md`, `nusel_display/README.md`, `run_nusel_evt.sh`, `wct-pr-perevt.jsonnet` |
| `work-mcp10-ctpcfix` | 43 | 33M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp10-fvxy` | 43 | 3M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp10-lm` | 53 | 37M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `lm_tune.py`, `nusel_display/nusel_scan_viewer.py`, `nusel_extract.py`, `qlmatching.jsonnet`, `run_ql_evt.sh`, `wct-clus-matching-perevt.jsonnet` |
| `work-mcp10-lm-offgate` | 2 | 2M | `34_lm-tagger.md` |
| `work-mcp10-lm2-offgate` | 2 | 2M | `34_lm-tagger.md` |
| `work-mcp10-mainflag` | 43 | 30M | `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md` |
| `work-mcp10-merge` | 42 | 3M | `15_overclustering-evt11-gamma.md`, `23_nusel-tgm-stm-chain.md`, `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `51_clustering-merge-attribution.md`, `52_isolated-grouping-fix-design.md`, `nusel_display/README.md`, `nusel_display/nusel_scan_viewer.py`, `nusel_display/serve_nusel_scan.sh`, `stm_main_connectivity.py` |
| `work-mcp10-merge2` | 43 | 3M | `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `nusel_display/README.md`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp10-offgate` | 42 | 3M | `29_tgm-chord-charge.md` |
| `work-mcp10-offgate2` | 42 | 3M | `29_tgm-chord-charge.md` |
| `work-mcp10-pathchord` | 33 | 3M | `31_tgm-chord-path-mode.md` |
| `work-mcp10-rcidoff` | 30 | 34M | `38_pctree-provenance-tgm-main-real.md` |
| `work-mcp10-rcoff` | 42 | 3M | `33_tgm-rescue-chord.md` |
| `work-mcp10-reschord` | 43 | 3M | `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md` |
| `work-mcp10-tgmfv` | 43 | 3M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md` |
| `work-mcp10-tgmfv-offgate` | 2 | 233K | `32_tgm-component-rescue-fvz.md` |
| `work-mcp1000-ctpcfix` | 33 | 25M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000-fvxy` | 33 | 2M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000-lm` | 33 | 28M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `38_pctree-provenance-tgm-main-real.md`, `lm_tune.py`, `nusel_display/nusel_scan_viewer.py`, `nusel_extract.py`, `qlmatching.jsonnet`, `run_ql_evt.sh`, `wct-clus-matching-perevt.jsonnet` |
| `work-mcp1000-mainflag` | 32 | 22M | `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md` |
| `work-mcp1000-pathchord` | 22 | 2M | `31_tgm-chord-path-mode.md` |
| `work-mcp1000-rcoff` | 30 | 2M | `33_tgm-rescue-chord.md` |
| `work-mcp1000-reschord` | 32 | 2M | `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `36_tgm-main-component-pairs.md` |
| `work-mcp1000-tgmfv` | 33 | 2M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md` |
| `work-mcp1000b-fvxy` | 33 | 2M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000b-fvzi-offgate` | 3 | 131K | `35_tgm-interior-fv.md` |
| `work-mcp1000b-nucdbg` | 2 | 109K | `36_tgm-main-component-pairs.md` |
| `work-mcp1000b-smoke38` | 3 | 1M | `38_pctree-provenance-tgm-main-real.md` |
| `work-mcp1000b-smoke38c` | 3 | 1M | `38_pctree-provenance-tgm-main-real.md` |

## `archive/stm-docs40-49/` — 43 dirs, 622 MB

**STM track-fit campaign.** arms whose newest citation is doc 40-49 — the STM fit dump and showcase (41-43), truth dQ/dx and delta rays (44,46), the un-merge into main+associated (45), the Bragg reference retune (47-48) and the STM containment FV fix (49). Also the three `stmon` arms, named only by `stmon_stats.py`.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-dq48` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp10-dq48base` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp10-dq48tab` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp10-dq48v3` | 45 | 28M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp10-dq49off` | 42 | 28M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp10-dq49off2` | 42 | 27M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp10-fvzi` | 43 | 3M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-lm2` | 43 | 37M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-mainpair` | 42 | 3M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-mainreal` | 42 | 37M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-stmon` | 43 | 4M | `41_stm-fit-dump.md`, `42_stm-fit-showcase-evt286241.md`, `43_magnify-tracking-sbnd-bugs.md`, `47_stm-bragg-reference-sbnd-retune.md`, `make_stmfit_bee.py`, `stmfit_showcase.py`, `stmon_stats.py` |
| `work-mcp10-unm45` | 43 | 27M | `45_unmerge-bundle-main-associated.md` |
| `work-mcp1000-dq48` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md`, `stm_fv_census.py` |
| `work-mcp1000-dq48base` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000-dq48tab` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000-dq48v3` | 35 | 22M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp1000-dq49off` | 32 | 22M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000-dq49off2` | 32 | 21M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000-fvzi` | 32 | 2M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-lm2` | 33 | 28M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-mainpair` | 32 | 2M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-mainreal` | 32 | 28M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-stmon` | 32 | 4M | `stmon_stats.py` |
| `work-mcp1000-unm45` | 30 | 21M | `45_unmerge-bundle-main-associated.md` |
| `work-mcp1000b-dq48` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md`, `stm_fv_census.py` |
| `work-mcp1000b-dq48base` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-dq48tab` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-dq48v3` | 34 | 20M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp1000b-dq49off` | 32 | 20M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000b-dq49off2` | 32 | 20M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000b-evnew` | 6 | 394K | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-fvzi` | 32 | 2M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-lm2` | 33 | 27M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-mainpair` | 32 | 2M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-mainreal` | 32 | 27M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-stmon` | 32 | 3M | `stmon_stats.py` |
| `work-mcp1000b-unm45` | 30 | 20M | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-diffusion` | 1 | 5K | `run_ql_evt.sh` |
| `work-mcsim-stmon` | 52 | 109M | `42_stm-fit-showcase-evt286241.md`, `44_stm-fit-truth-dqdx.md`, `45_unmerge-bundle-main-associated.md`, `46_stm-fit-deltarays-and-gui.md` |
| `work-mcsim-unmcomp` | 21 | 397K | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-unmoff` | 42 | 3M | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-unmon` | 42 | 3M | `45_unmerge-bundle-main-associated.md` |
| `work-stmbadch` | 2 | 344K | `42_stm-fit-showcase-evt286241.md`, `43_magnify-tracking-sbnd-bugs.md` |

## `archive/aborted-d54/` — 6 dirs, 145 MB

**SBND `d54off`/`d54on` pair — off-arm complete, on-arm empty.** What is
observable: each `d54off` holds a full run (20 `.batch_nusel_*.log`, `ql_evt*`,
`nusel_evt*`), while each `d54on` holds ten event dirs that are empty and **no
batch log at all** — so the on-arm never wrote anything. Whether it was killed,
never launched, or is queued for relaunch is not recorded here, so treat
"aborted" as a description of the output, not of intent. No doc cites either
arm; doc 52's `d54base`/`d54opt2` are `abtest/snap/` labels, a different thing.
Kept rather than deleted because the off-arm is a complete 30-event run.

**If that campaign is relaunched** with `-t d54on`, the runner will create a
fresh dir at the *top level* while its sibling sits here — move the pair back
out (and re-run `relink_tags.py`) rather than leaving the arms split.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-d54off` | 52 | 60M | — |
| `work-mcp10-d54on` | 10 | 5K | — |
| `work-mcp1000-d54off` | 29 | 41M | — |
| `work-mcp1000-d54on` | 10 | 5K | — |
| `work-mcp1000b-d54off` | 30 | 44M | — |
| `work-mcp1000b-d54on` | 10 | 5K | — |

## RETIRED 2026-07-25 — 44 dirs, 307 MB, deleted

Every entry here was verified to be named by no committed doc, no analysis
script, no running viewer, and no `abtest/`/`sweep/` gate snapshot, using a
**bare-tag** search (`dq48tab`, `d53leg`, …) across `docs/`, `nusel_display/`,
`ql_scan/`, `*.py`, `*.sh`, `*.jsonnet`. A full-directory-name search is NOT
sufficient — the docs cite tags without the manifest prefix, and that mistake
initially mislabelled `d53dflt`/`d53leg` (evidence for a gate committed the same
day) as unreferenced.

Group A is the ad-hoc probe class: 2-5 event runs made while chasing one number.
Group B is full-manifest arms that nothing names — mostly gate off-arms and
superseded variants whose partner arm survives. No link anywhere pointed into
any of them, so the deletion broke nothing. Nothing inside `work/` was touched.

| dir | group | entries | size | what it was |
|---|---|---|---|---|
| `work-d49diag-nofit` | A | 3 | 866K | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-off` | A | 3 | 928K | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-rep1` | A | 3 | 1M | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-rep2` | A | 3 | 1M | doc 49 STM-containment FV diagnostic probe |
| `work-det287517-fit1` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-fit2` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-flagtest2` | A | 3 | 4M | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-nofit3` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-nofit4` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-oldfit` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-oldnofit` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-mcp10-d49soff` | B | 42 | 27M | doc 49/50 STM-scope gate off-arm |
| `work-mcp10-d53chk` | A | 2 | 2M | doc 53 Bee-layer spot check, evt 284349 |
| `work-mcp10-dq48fit` | B | 43 | 4M | doc 48 dQ/dx fit intermediate |
| `work-mcp10-dq48v2` | B | 42 | 28M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp10-dq49` | B | 42 | 29M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp10-mp36off` | B | 42 | 3M | doc 36 main-component-pairs off-arm |
| `work-mcp10-mpoff` | B | 42 | 3M | doc 36 main-component-pairs off-arm |
| `work-mcp10-prodoff` | B | 42 | 3M | production-default off-arm |
| `work-mcp10-stmoff` | B | 42 | 3M | STM-tagger off-arm |
| `work-mcp1000-d49soff` | B | 32 | 21M | doc 49/50 STM-scope gate off-arm |
| `work-mcp1000-dq48fit` | B | 33 | 4M | doc 48 dQ/dx fit intermediate |
| `work-mcp1000-dq48v2` | B | 32 | 22M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp1000-dq49` | B | 32 | 23M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp1000-mp36off` | B | 30 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000-mpoff` | B | 32 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000-prodoff` | B | 30 | 2M | production-default off-arm |
| `work-mcp1000-stmoff` | B | 30 | 2M | STM-tagger off-arm |
| `work-mcp1000-tgmfv-dbg3` | A | 2 | 325K | doc 32 TGM FV debug |
| `work-mcp1000-tgmfv-dbg5` | A | 2 | 330K | doc 32 TGM FV debug |
| `work-mcp1000b-d49soff` | B | 32 | 20M | doc 49/50 STM-scope gate off-arm |
| `work-mcp1000b-dq48fit` | B | 32 | 3M | doc 48 dQ/dx fit intermediate |
| `work-mcp1000b-dq48v2` | B | 32 | 20M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp1000b-dq49` | B | 32 | 23M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp1000b-evbase` | A | 6 | 394K | doc 48 dQ/dx table pre-check |
| `work-mcp1000b-mp36off` | B | 30 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000b-mpoff` | B | 32 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000b-nucoffdbg` | A | 2 | 109K | doc 36 nu-candidate off-arm debug |
| `work-mcp1000b-prodoff` | B | 30 | 2M | production-default off-arm |
| `work-mcp1000b-sa1` | B | 32 | 20M | doc 49 scope-alignment trial |
| `work-mcp1000b-sa2` | B | 32 | 20M | doc 49 scope-alignment trial |
| `work-mcp1000b-stmoff` | B | 30 | 2M | STM-tagger off-arm |
| `work-mcsim-unmerge` | A | 20 | 6K | doc 45 empty stub (run produced no output) |
| `work-why285185` | A | 15 | 740K | one-off "why is evt 285185 like this" probe |

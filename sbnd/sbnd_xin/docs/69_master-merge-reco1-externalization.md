# doc 69 — master merge 2026-08-03: LLM-agent policy files, cmake, and the reco1 externalization

**Status:** merge landed (toolkit `87ada3d5`); SBND PR chain gated byte-identical;
reco1 dump path externalized and proven byte-identical.

## Repro

```bash
# 0. the standalone reco1 plugin (sibling of toolkit/, outside any tracked repo)
cd /nfs/data/1/xqian/toolkit-dev
git clone https://github.com/WireCell/wire-cell-sbnd-reco1.git
cd wire-cell-sbnd-reco1
cmake -B build -S . \
      -DCMAKE_PREFIX_PATH=/nfs/data/1/xqian/toolkit-dev/local \
      -DCMAKE_INSTALL_PREFIX=$PWD/install \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_INSTALL_RPATH="$(root-config --libdir);/nfs/data/1/xqian/toolkit-dev/local/lib"
cmake --build build -j6 && cmake --install build
(cd build && ctest --output-on-failure)          # reco1_factories PASS

# 1. the merge
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git fetch origin && git merge --no-ff origin/master     # 2 conflicts, see §3
wcbuild

# 2. gates
./build/clus/wcdoctest-clus ; ./build/sigproc/wcdoctest-sigproc
./build/util/wcdoctest-util ; ./build/aux/wcdoctest-aux

cd ../wcp-porting-img/sbnd/sbnd_xin
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-poc0 work-mrgB-post data   # post-merge
# pre-merge arm: checkout 12d65f1d, wcbuild, same command -> work-mrgA-pre
```

`abtest/hash_seq.py` was added by this round (see the `hash_archive.py` GOTCHA in
§4). The remaining one-off drivers stayed in the session scratchpad and are not
committed: `compile_prod_cfg.sh`, `reco1_ab.sh`, `premerge_arm.sh`, `gate_cmp.sh`.

## 1. What master brought

12 commits, head `b806824f`. Relevant to us:

| commit | effect here |
|---|---|
| `e53af02c` | `CLAUDE.md`, `AGENTS.md`→symlink, `.claude/skills/wct-{architecture,beads,contributing,testing}`, `.claude/hooks/policy-guard.py`; **un-ignores** `CLAUDE.md`/`AGENTS.md`/`.claude` |
| `b806824f` | cmake build, single aggregate `wcdoctest`, README.org |
| `501c5fe6` | sigproc decon_2D OOB read fix (#491) |
| `5f684887` | **deletes** the 9 SBNDReco1 files → standalone `wire-cell-sbnd-reco1` |
| `8d069234`, `49a6ec3e`, `cdca41c0` | SBND SCE wiring + `use_sce`/`reality` in `sbnd/clus.jsonnet` |
| `2edf0a88` | `Bee::Points` optional per-point `e` / `nu_idx` |

## 2. CLAUDE.md — why ours moved out of the repo

Our 16 KB operating manual lived at `toolkit/CLAUDE.md` as an **untracked but
gitignored** file. `e53af02c` un-ignores that path and adds a tracked file there.
Git treats ignored untracked files as expendable, so it overwrites them **with no
conflict, no warning and nothing stashed** — and since the file was never tracked,
git holds no copy to recover. Demonstrated on git 2.45.2 in a throwaway repo, and
then observed for real during this merge.

Two exposure paths, not one: `git merge origin/master`, **and** a plain
`git checkout master` (we keep a local `master` tracking `origin/master`), which
clobbers it just as silently after any fetch.

**Resolution.** Our manual now lives at `/nfs/data/1/xqian/toolkit-dev/CLAUDE.md`
— one level *above* the repo. Both working paths resolve to that same parent
(`/home/xqian/work/scratch_wcgpu1` → `/nfs/data/1/xqian`, both inode 62161792), so
one file covers both entry paths. It opens with an explicit precedence block:
upstream's *policies* (no new deps, no plugin↔plugin deps, human writes the
Issue/PR preamble, tests ship with changes) are in force; where the two disagree
on **build/test mechanics** (`wcbuild` vs cmake, `wcdoctest-<pkg>` vs the
aggregate) ours governs this tree.

`.claude/` is no longer blanket-ignored, so our `settings.json` and the three
local skills (`ab-verify`, `wct-knob`, `wct-run`) now show as untracked. Hide them
per-clone with `.git/info/exclude` rather than editing the now-tracked
`.gitignore`. Upstream's `policy-guard.py` hook is **dormant** — no
`.claude/settings.json` is committed to wire it up.

## 3. The two merge conflicts

**`cfg/pgrapher/experiment/sbnd/clus.jsonnet`** (ours: 848 insertions over 39
commits; theirs: the SCE refactor) — resolved as a **union**:

- `clus_all_apa` signature takes both our four args
  (`cathode_rescue_on`, `cathode_rescue_unmatched`, `adopt_nu_fragments`,
  `save_bundle_main_provenance`) and upstream's `use_sce`, `reality`.
- `all_apa` keeps our args and threads `use_sce`/`reality` through.
- Our `pr()` PR-tagger method survives (upstream's "clustering+matching only"
  note refers to what `clus.jsonnet` *wires*, not to the method).
- The tail `function(...)` block auto-merged: upstream's
  `reco = {sim:{use_sce:false,pos_offset_on:false}, data:{use_sce:false,pos_offset_on:true}}[reality]`
  replaces `local pos_offset_on = reality == 'data'`. Semantically identical for
  `reality='data'`, and both realities carry `use_sce=false`, so our chain still
  clusters in `x_t0cor`.

**`cfg/pgrapher/experiment/sbnd/wct-reco1-dump.jsonnet`** (modify/delete) —
**accepted the deletion**. See §4.

> **GOTCHA — the asymmetric delete.** Git raised a modify/delete conflict on the
> *config* (we had touched it in `9cb6c860`) but deleted the nine *C++* files
> silently, because we had never touched those. Resolving the visible conflict
> "keep ours" would have left a config naming factories that no longer exist —
> passing merge, build and `wcsonnet`, and failing only at `wire-cell -c` runtime,
> possibly weeks later at the next staging run.

## 4. reco1 externalization

Upstream moved `SBNDReco1FrameSource` / `SBNDReco1OpFlashSource` out of
`libWireCellRoot.so` into `WireCell/wire-cell-sbnd-reco1` (WCT issue #494: the
mirror ROOT classes carry the **real** LArSoft names with a trimmed layout, so
inside a `lar` process ROOT could pick the trimmed dictionary for a full-layout
product and corrupt memory). Symbol census confirms the move:
`libWireCellRoot.so` 111 → **0** `SBNDReco1` symbols; the standalone exports 112.

Our side:

- The plugin is built as a **sibling of `toolkit/`** at
  `/nfs/data/1/xqian/toolkit-dev/wire-cell-sbnd-reco1`, outside every tracked
  repo, so no `.gitignore` interaction exists at all.
- Upstream's `cfg/wct-reco1-dump.jsonnet` is forked from **before** our
  `9cb6c860` and has no MC product-name TLAs (`wire_product`,
  `badmask_product`, `summary_product`, `flash_process`) — it can only read SBND
  *data* reco1 files. Rather than diverge their clone, `sbnd_xin/wct-reco1-dump.jsonnet`
  (previously a one-line re-export of the in-tree file) now carries our full
  version, changed from it in exactly one way: plugin `WireCellRoot` →
  `WireCellSBNDReco1`. That file is now the single source of truth for our chain.
- `run_reco1_dump.sh`, `staged-mcp2025c-1000evt/{stage_all,process_all}.sh` set
  `LD_LIBRARY_PATH` / `WIRECELL_PATH` for the plugin and hard-fail with a build
  recipe if it is missing.

> **GOTCHA — RPATH.** The standalone's documented build produces a `.so` with no
> RUNPATH (waf gives the in-tree libs one). `dlopen` then fails on
> `libCore/libRIO/libTree` and `wire-cell` reports only
> `failed to load plugin: WireCellSBNDReco1` — no dlerror detail. Configure with
> `-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` plus an explicit
> `-DCMAKE_INSTALL_RPATH="$(root-config --libdir);<prefix>/lib"`. Worth sending
> upstream.

### Verification (byte-identical, member content via `hash_seq.py`)

Same 48-event data file, `-caf product`:

| artifact | old in-tree reader | new standalone | verdict |
|---|---|---|---|
| `frames-dnn.tar.bz2` | `ed39e70b…3107`, 240 members | `ed39e70b…3107`, 240 | **IDENTICAL** |
| `opflash_apa0.tar.gz` | `c1eb369f…c214f`, 144 | same | **IDENTICAL** |
| `opflash_apa1.tar.gz` | `f214c7d4…89149`, 144 | same | **IDENTICAL** |

And end-to-end **after** the merge (in-tree factories gone), the updated
`run_reco1_dump.sh` + our cfg reproduce the same
`ed39e70b…3107` — extraction `input_files_reco1/extracted-mergecheck/`.

> **GOTCHA — `hash_archive.py` is O(n²) on `.bz2`.** It sorts members then seeks;
> every seek re-decompresses from the start. On this 240-member, ~1.4 GB
> archive it ran >20 min without finishing one file. `hash_seq.py` streams once
> in archive order and sorts the digests at the end — same rollup semantics,
> seconds instead of tens of minutes. Use it for `.tar.bz2`; `hash_archive.py`
> is fine for the small `.tar.gz`/`.zip`.

## 5. SBND PR-chain gate

Compiled-config proof on the production entry point
(`sbnd/wct-pr-perevt.jsonnet`, real TLAs, full 15-stage `pipeline_names`):
pre-merge 257 933 B vs post-merge 258 267 B, **differing by exactly 13 lines**,
all upstream SCE wiring — one `SCEFieldTH3:sbnd_dualmap` node plus the per-APA
`sce_field` metadata key on two `DetectorVolumes` entries. Every flipped SBND
default survives unchanged:

| knob | pre | post |
|---|---|---|
| `shower_topo_demote_len` | 50 | 50 |
| `cathode_rejoin_perp` | 30 | 30 |
| `cathode_rejoin_angle` | 20 | 20 |
| `cathode_rejoin_dir_radius` | 150 | 150 |
| `iso_endpoint` | true | true |

Compiled-code delta reaching this chain: **none**. `OmnibusSigProc` does not
appear in the compiled PR config at all, so the decon_2D fix cannot touch it;
`Bee::Points`' new fields are gated on `m_has_extra`, set only by the new
`append()` overload, which has no caller.

Arms (48 nueCC events, `work-nuecc48-poc0`, `PR_JOBS=6`, all rc=0):
`work-mrgA-pre` (12d65f1d) vs `work-mrgB-post` (87ada3d5). The pre-merge rebuild
was confirmed faithful: its compiled production config is byte-identical to the
snapshot taken before the merge started.

**Result — 144 comparisons, 141 identical, 3 differing:**

| artifact | n | verdict |
|---|---|---|
| `mabc-pr.zip` | 48 | **all byte-identical** |
| `pctree-pr-evt<ID>.tar.gz` | 48 | **all byte-identical** |
| `nusel-evt<ID>.tsv` | 48 | 45 identical, **3 differ in one column** |

**Every reconstruction artifact is byte-identical.** The three differing TSVs
(evts 268067, 269774, 52672) differ in exactly one field, `stmfit`
(`contained` ↔ `eval`), on the nu-candidate row. `tgm`/`stm`/`fc`, `label`,
point counts and lengths are identical in all three. **GATE: PASS on physics.**

> **GOTCHA — `stmfit` is scraped from spdlog, and the merge tore the line.**
> `stmfit` is not a reconstruction output: `nusel_extract.py:parse_stm_skips`
> greps `check_stm_conditions: cluster N no STM fit: <reason>` out of the WCT
> DEBUG log. WCT writes long spdlog messages **non-atomically**, so lines tear
> and splice (already documented in `stmfit_code`'s own docstring). Post-merge
> logs are ~460 B longer (upstream's SCE component adds lines), which shifts the
> interleaving. Concretely, evt 268067 cluster 15:
>
> ```
> pre : [09:03:02.632] D [clus.NeutrinoPattern] check_stm_conditions: cluster 15 no STM fit: fully contained (Mid Point A)
> post: tions: cluster 15 no STM fit: fully contained (Mid Point A)
> ```
>
> The post line is torn mid-word, `RE_STM_SKIP`'s `check_stm_conditions:` anchor
> misses it, cluster 15 drops out of `skips`, and the column falls through to
> `eval`. It moves in **both** directions across events (269774 went
> `eval`→`contained`), which is the signature of an interleaving artifact rather
> than a systematic code change. Repeat runs r1/r2 on the post-merge binary
> reproduce the post values, so it is stable per-build, not run-to-run noise.
>
> Consequence for future gates: **never treat a `stmfit`-only diff as a physics
> regression**, and prefer `mabc-pr.zip` / `pctree-pr-*.tar.gz` member hashes as
> the gate. Hardening `RE_STM_SKIP` to match on `cluster (\d+) no STM fit:`
> without the `check_stm_conditions:` prefix would make the column robust to
> tearing — deliberately **not** done in this round (unrelated to the merge; see
> CLAUDE.md §5 tie-breaker on pre-existing bugs).

## 6. All-config gate (`compile_all_cfg.sh` + `cmp_cfg.sh`)

The 48-event gate above runs `pr()` on pre-existing pctrees, so it never
instantiates `clus_all_apa` — the very function both conflicts landed in.
Jsonnet is lazy, so an error there would be invisible to it. All 16 live job
configs were therefore compiled pre and post (`CFGROOT` points the pre run at a
`git archive 12d65f1d cfg` tree, so no second checkout is needed):

```
JOB               ELEMENTS ORDER      EDGES      NORMDIFF
pdhd_clus         222->222 same       same       0
pdhd_img/nfsp/sim*        (5 jobs)    same       0
pdvd_clus         402->402 same       same       0
pdvd_img/nfsp/sim*        (4 jobs)    same       0
sbnd_img            98->98 same       same       0
sbnd_simcheck       54->54 same       same       0
sbnd_clus         68->69!! DIFFERS    same       21
sbnd_pr           35->36!! DIFFERS    same       13
sbnd_ql          100->101!! DIFFERS   same       25
```

0 compile failures — `sbnd_clus` does instantiate `clus_all_apa`
(9 `MultiAlgBlobClustering`, 4 `PointTreeMerging`), so the merged function is
exercised and valid. **All 11 PDHD/PDVD configs are identical.** The three SBND
jobs each gain exactly **one** element, the `SCEFieldTH3:sbnd_dualmap` node, plus
its `sce_field` metadata references; `edges` and component order are unchanged
everywhere. (`cmp_cfg.sh` normalizes away master's new `_pnode` key, a sibling of
`data` that `ConfigManager` never reads.)

## 7. sigproc #491 — no gate needed for our detectors

`501c5fe6` changes `OmnibusSigProc`, which PDHD/PDVD do run. It is nonetheless a
**no-op here**, by construction rather than by measurement:

- The seven `decon_2D_*` extraction sites now derive the wire-axis offset as
  `(tm_r_data.rows() - m_nwires)/2` instead of hard-coding `m_pad_nwires`. That
  expression **equals `m_pad_nwires` whenever the FFT wire padding is still
  present**, and is 0 only after `decon_2D_init()->unpad_data()` has shrunk the
  array — which happens for the ICARUS **"twofaced"** split-induction layout.
  No SBND/PDHD/PDVD config sets `twofaced` (grep: zero hits), so every one of
  our planes keeps the padding and takes the unchanged branch.
- The `restore_baseline` hardening drops non-finite samples. Non-finite samples
  arose *from* the OOB read; with an in-bounds read there are none, and the
  commit states "no behavior change for finite data".

So PDHD/PDVD NF+SP output is expected byte-identical. This is a **reasoned**
conclusion, not an empirical one — `abtest/mergegate_{run,cmp}.sh` (4 PDHD +
2 PDVD, snapshotting nfsp/img/clus separately) is the harness to confirm it, and
has **not** been run in this round.

## 8. Not covered here

- The SCE path itself is untested: both realities ship `use_sce=false`, so
  `SCECorrection` is built but never applied in our scope.
- The `/cvmfs` SCE map (`sbnd_data/v01_42_00/SCEoffsets/…voxelTH3.root`,
  5.8 MB) must be mounted for the SBND clus/ql/pr jobs to configure — it is
  mounted here. A machine without `/cvmfs` will now fail to configure those
  three jobs where it previously succeeded.
- MC (`reality='sim'`) reco1 extraction through the standalone plugin is not
  re-verified; only the data path was A/B'd. The MC TLAs are carried in our cfg
  unchanged from `9cb6c860`.

# Porting SCE-true clustering + the truth labeler (`wclsTensorSetLabeler`) into the apply-pointcloud WCT

**Goal.** Make the 1-step LArSoft chain `wcls-img-clus-matching-xin[-data].fcl` run on the
`apply-pointcloud` WCT with (a) **SCE true-space** all-APA clustering for sim and (b) the
**`wclsTensorSetLabeler`** truth/nugraph stage — the two tgm-branch features that never
merged cleanly into apply-pointcloud. Tested on **data** (event 269774, `reality=data`) and
**MC** (1st event of `mc_paths-v10_14_02_03-100files.lst`, `reality=sim`).

All edits below are **local / uncommitted** (kept for review); only this doc is committed.

---

## 1. `clus.jsonnet` port (apply-pointcloud)

`cfg/pgrapher/experiment/sbnd/clus.jsonnet` — grafted from tgm, adapted to apc's
(193-commit-diverged) structure. apc already had `reality`+`pos_offset_on`; added:

- **`reco` local** grouping reality toggles: `sim → {use_sce=true, pos_offset_on=false}`,
  `data → {use_sce=false, pos_offset_on=true}`.
- **SCE wiring:** `sce_field` (`SCEFieldTH3`, TrueBkwd reco→true, `sign:1`) and
  `sce_field_fwd` (TrueFwd true→reco, for the labeler); `common_sce_coords =
  ['x_sce','y_sce','z_sce']`; `common_corr_coords(pos_offset_on, use_sce)`; `sce_field` in
  the `dvm` per-APA metadata + `pctransforms` `uses:[dv, sce_field]` (apc's
  `PCTransforms.cxx` already reads the `sce_field` metadata key and builds a real
  `SCECorrection`). In `clus_all_apa`, the scope step is
  `if use_sce then switch_scope('SCECorrection') else switch_scope()` and the Bee/clustering
  coords use `common_corr_coords(pos_offset_on, use_sce)`.
- **Labeler:** `use_sce`/`reality`/`run_labeler` threaded through the top-level `clus()` →
  `all_apa` → `clus_all_apa`; a `wclsTensorSetLabeler` pnode (reality, `sce_field=sce_field_fwd`,
  `sce_correction:true`, `fv_box`, truth cuts) inserted `[mabc, labeler, sink]` when
  `run_labeler`; sink `dump_mode: tensor_outname=='' && !run_labeler` (labeled pctree is the
  deliverable). `bee_sink` attached to the labeler in sim only.

The wrapper `sbnd/wcls-img-clus-matching-xin.jsonnet` already calls
`clus(reality=reality, run_labeler=true)` and relies on all of the above — no wrapper change
needed (sbnd_xin/wrapper untouched).

## 2. The `libWireCellRoot` ⟂ LArSoft dictionary clash (blocker + fix)

The 1-step chain **segfaulted** on apc — right at art-event read, before any node ran —
with `"class art::Wrapper<...> ... already in libWireCellRoot.so"` warnings. Cause: apc's
`root/` added the **SBNDReco1 bare-ROOT sources** (commits `cc3e3f87`, `6e78050d`), whose
`root/dict/LinkDef.h` bakes ROOT dictionaries for LArSoft/art product types
(`recob::Wire/OpFlash/OpHit`, `raw::ptb::sbndptb`, `sbnd::timing::*`, `art::Wrapper<...>`,
`art::EventAuxiliary`, …) into `libWireCellRoot.so`. Loaded inside a `lar` process alongside
sbndcode/canvas, those duplicate dicts corrupt ROOT's dictionary state → SIGSEGV. (tgm has no
`SBNDReco1Products.h`, so the tgm dump worked; the standalone pure-wire-cell perevt is fine —
no canvas to clash with.)

**Fix (owner-approved trade-off):** emptied the `#pragma link` block in
`root/dict/LinkDef.h`, and disabled the two sources that reference the (now missing) dict
vtables by renaming `root/src/SBNDReco1{FrameSource,OpFlashSource}.cxx` → `.cxx.disabled`.
`libWireCellRoot` then links clean and no longer clashes. **Trade-off:** the bare-ROOT
`SBNDReco1FrameSource`/`OpFlashSource` (the no-larsoft reco1-read path used by
`wct-reco1-dump`) are disabled. Proper fix for later: move that dict into a separate lib the
LArSoft chains don't load.

Also dropped the obsolete `"WireCellQLMatch"` plugin (larwirecell's old qlmatch;
`undefined symbol: Ress::solve` on apc) from `sbnd/wcls-img-clus-matching-xin.fcl` — this
chain uses the toolkit `WireCellMatch`, not larwirecell's qlmatch.

## 3. larwirecell rebuild against apc — and a stale-`.so` gotcha

The 1-step chain needs larwirecell (`wclsCookedFrameSource`, `wclsTensorSetLabeler`) rebuilt
against the current apc WCT (ABI). Recipe: `docs/0-build-…` §2/§3, but **force recompilation**
(`find $MRB_SOURCE/larwirecell -name '*.cxx' -o -name '*.h' | xargs touch`) because `make`
doesn't track the external `opt/include` WCT headers.

**Gotcha (cost real time):** `make install` aborts with `MAKE_EXIT=2` on a benign
`README.md → /usr/local` prefix error (`CMAKE_INSTALL_PREFIX`), and this abort happens
**before** the freshly-linked `libWireCellAIML.so` is staged into `$MRB_INSTALL`. So the
usual "hand-copy from `$MRB_INSTALL`" grabs a **stale** `.so`. Symptom seen here: the labeler
emitted the old Bee set `truth_depo_sce` instead of the current `sed-sce_drift_smear_readout`
+ `sed-smear_readout` (renamed in larwirecell commit `4ace1e2`), even though the fresh
`TensorSetLabeler.cxx.o` *did* contain the new names.

**Fix:** copy the freshly-linked `.so` from the **build staging dir**
`build_slf7.x86_64/larwirecell/lib/libWireCellAIML.so` (verify with
`strings … | grep sed-sce_drift_smear_readout`), not from `$MRB_INSTALL`. General rule for
this tree: after a larwirecell rebuild, verify the hand-copied `.so` contains an
expected new symbol/string before trusting it.

## 4. Tests + BEE

| chain | reality | result |
|---|---|---|
| `wcls-img-clus-matching-xin-data.fcl`, event 269774 | data | `use_sce=false`, labeler data-mode; clustering-global 15 real / 9 cluster |
| `wcls-img-clus-matching-xin.fcl`, MC 1st event | sim | `SCECorrection: ISCEField wired in` (SCE true space); labeler `labeled 2889/3044 blobs, 5 truth tracks`; Bee truth sets `mc`/`truth_trackid_labeled`/`truth_unlabeled` + **`sed-sce_drift_smear_readout`** + **`sed-smear_readout`** |

- MC (corrected, with sed-* sets): https://www.phy.bnl.gov/twister/bee/set/c65b46ab-c770-4d51-abe7-b0834d3f0372/event/list/
- Data: https://www.phy.bnl.gov/twister/bee/set/e767a2dd-66cd-4043-b43f-3d33503cd87a/event/list/

## 5. Uncommitted edits (for review)

- wire-cell-toolkit (`apply-pointcloud`): `cfg/…/sbnd/clus.jsonnet` (SCE+labeler port);
  `root/dict/LinkDef.h` (emptied SBND dict); `root/src/SBNDReco1{FrameSource,OpFlashSource}.cxx`
  → `.cxx.disabled`.
- wcp-porting-img: `sbnd/wcls-img-clus-matching-xin.fcl` (dropped `WireCellQLMatch`).
- opt: larwirecell rebuilt against apc (with the correct `libWireCellAIML.so`).

`tgm` intact; `sbnd_xin` untouched. Memory: `project_tensorsetlabeler`, `project_build_recipe`.

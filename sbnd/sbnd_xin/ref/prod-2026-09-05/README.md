# ref/prod-2026-09-05 — SBND production operating point

**What moved from `prod-2026-09-04`, and on whose word.**

Owner, 2026-09-03, after doc 99 round 2 gated both sides of the wrong-flash
defect: *"Please flip both knobs on and perform the relevant validations."*

Both doc-99 knobs are now **ON for SBND**. The C++ defaults in
`match/src/QLMatching.cxx` (`merge_flash_pcs`) and
`root/src/SbndPrMagnifyTrackingVisitor.cxx` (`flash_by_gid`) stay **false**, so
no other detector moves — see "What is deliberately NOT in this generation".

## The drift — two keys on two components, in two of the 21 artifacts

```
DRIFT : prod_prjob.json
  ADDED  SbndPrMagnifyTrackingVisitor:pr .flash_by_gid    = true
DRIFT : sbnd_ql.json
  ADDED  QLMatching:matching_joint       .merge_flash_pcs = true
```

The other **19 of 21** artifacts are byte-identical to `prod-2026-09-04`: the six
pdhd jobs, the five pdvd jobs, `uboone.json`, `prod.wcls`, `prod.standalone`,
`sbnd_clus.json`, `sbnd_img.json`, `sbnd_pr.json`, `sbnd_simcheck.json` and
`bare_prjob.json`. `prod-2026-09-04` is left byte-untouched (M13); this is a new
generation because the change is deliberate and is **not** byte-identical.

`prod.standalone` not drifting is a fact about that job, not an oversight: it
compiles with `--ext-code joint=false`, giving two single-input `matching0` /
`matching1` nodes where the merge loop never executes, so `merge_flash_pcs` is
inert there by construction.

## What the knobs do

- **`flash_by_gid`** (read side) — `tracking-pr.root:T_cluster` fills
  `flash_id`/`flash_time_us`/`flash_pe` from `Cluster::get_matched_flash()`,
  resolving `matched_flash_gid` against the merge-safe `opflash` PC, instead of
  `Cluster::get_flash()`'s per-input row index. Diagnostic columns only: no
  archive byte moves. `flash_id` now carries the **gid**, joinable to the
  `opflash` PC, not a per-input row id.
- **`merge_flash_pcs`** (write side) — the joint `QLMatching` node's multi-input
  merge carries **every** input's canonical `flash`/`light`/`flashlight`/
  `flashcov` PCs, re-basing each input's flash-row references (including the
  per-cluster `flash` scalar) onto the merged list, instead of keeping only
  input 0's and dropping the rest.

Together they take SBND's `T_cluster` flash resolution from **50.7% → 100.0%**
(20 156 matched rows, 308 events).

## Why the source change is four files and not two

The two per-event jobs are not SBND's only callers. The LArSoft chain
(`wcp-porting-img/sbnd/wcls-img-clus-matching-xin.jsonnet`) calls
`qlm.matching_joint()` and `clus_maker.pr()` **without** passing either key, and
`pr-operating-point.jsonnet` does not carry them either. So the module defaults
in `sbnd/qlmatching.jsonnet` and `sbnd/clus.jsonnet` moved as well; a TLA-only
flip would have left LArSoft on the defect while both entry points still
compiled and ran.

## What is deliberately NOT in this generation

- **The C++ defaults stay false.** `merge_flash_pcs`' other binders are pdhd and
  pdvd, which have their own `qlmatching.jsonnet` and were never gated for this;
  flipping the C++ default would ship an unvalidated archive change to both.
  `flash_by_gid`'s other binder is `PdvdPrMagnifyTrackingVisitor`, whose
  gid-uniqueness precondition has never been checked —
  `doctest_pdvd_tracking_defaults.cxx` asserts the default stays false so the
  omission is enforced rather than remembered.
- **`flashcov` is now reachable but still unexercised.** The write fix shifts
  `flashcov.flash` too, but that PC exists only when the light chain ran with
  `emit_coverage`, and no SBND input here did. The branch is uniform with the
  three that were exercised; it has no measurement behind it. If coverage is
  ever turned on, gate that path before trusting it.

## Source evidence — do not retire these

The six `work-{ncpi0,nuecc48,mcp1k}-d99r2*` arms and
`work-ncpi0-d99r3prod{,pr}` are this generation's evidence: the before/after
tables in doc 99 §10 were computed from them, and `d99r3_flip_gate.sh` re-gates
against them. A retire round that keeps the outputs but drops these arms makes
§10 un-re-runnable (the recorded lesson from a gate whose SOURCE arm was
retired).

## Re-check

```
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-05     # 21/21, exit 0
```

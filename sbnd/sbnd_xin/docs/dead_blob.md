# Dead-blob tiling & bee dumping in `sbnd_xin`

Audit of the SBND imaging chain in this directory: what the live and dead
blob tilings actually do today, where dead blobs end up in the bee output,
and one regression of a previous fix that this commit restores.

The `toolkit/sbnd_xin/` path is a symlink to
`wcp-porting-img/sbnd/sbnd_xin/`. Edits below land in the `wcp-porting-img`
repo; there is no separate `sbnd_xin` tree in the toolkit repo.

## Question

The expectation (based on the reference `wcp-porting-img/sbnd` chain and a
prior fix landed on `apply-pointcloud`):

- **Live tiling:** 3-live-plane + 2-live-plane + 1-dead-plane tiling — the
  4-branch fanpipe covering `{UVW, UV+W-dead, VW+U-dead, UW+V-dead}`.
- **Dead-blob tiling:** 2-dead-plane tiling — tiles formed from 2
  dead-channel-masked planes plus 1 geometrically-open ("dummy") plane.
- **Dead blobs into the final bee files:** yes, alongside live blobs.

## Where the tilings are defined

All slicing+tiling for SBND is built in
`cfg/pgrapher/experiment/sbnd/img.jsonnet`:

- `slicing()` (l.133–155) wraps `MaskSlices` and accepts three plane-role
  lists: `active_planes`, `masked_planes`, `dummy_planes`.
- `tiling()` (l.158–175) wraps `GridTiling` per anode face.
- `multi_active_slicing_tiling()` (l.178–188) — live fanpipe, 4 branches:
  - `active=[0,1,2], masked=[]` → 3-live
  - `active=[0,1], masked=[2]` → 2-live + 1-dead
  - `active=[1,2], masked=[0]` → 2-live + 1-dead
  - `active=[0,2], masked=[1]` → 2-live + 1-dead
- `multi_masked_2view_slicing_tiling()` (l.191–202) — dead fanpipe, 3
  branches:
  - `dummy=[2], masked=[0,1]` → 1 dummy + 2 dead-channel-masked
  - `dummy=[0], masked=[1,2]` → 1 dummy + 2 dead-channel-masked
  - `dummy=[1], masked=[0,2]` → 1 dummy + 2 dead-channel-masked
- `imgpipe()` (l.317–358) — top-level dispatcher. Important conditional at
  l.342:
  ```jsonnet
  local st = if multi_slicing == "multi-2view" || multi_slicing == "multi-3view"
              then img.multi_active_slicing_tiling(anode, ..., 4)
              else g.pipeline([
                img.slicing(anode, ..., active_planes=[0,1,2], masked_planes=[], dummy_planes=[]),
                img.tiling(anode, ...)
              ]),
  ```
  Only the two listed strings hit the multi-active fanpipe; any other
  string silently produces a single 3-view active pipeline. The masked
  fork (`multi_masked_2view_slicing_tiling`) is invoked unconditionally
  in the same `else` block (l.353).

## Plane-role semantics (`MaskSlices`)

- `active`: real ADC activity on that plane is required for a tile.
- `masked`: the dead-channel mask is used as activity; tiles can form
  across dead regions on this plane.
- `dummy`: the plane contributes geometric coverage everywhere — no signal
  constraint.

So "2-dead-plane tiling" in this codebase corresponds to:
`masked = [2 planes], dummy = [1 plane], active = []` — which is exactly
what `multi_masked_2view_slicing_tiling` produces.

## Live tiling — fixed by 32852232, regressed by eef2df7, restored here

`32852232` (Xin Qian, 2026-04-25, in toolkit) extended the conditional at
l.342 from `if multi_slicing == "multi-2view"` to
`if multi_slicing == "multi-2view" || multi_slicing == "multi-3view"`.
Before that fix, callers passing `"multi-3view"` silently fell through to
the single-3view branch. The commit message records
`evt2/APA0: 3114 → 4260` active blobs after the fix, with all four active
slicer branches firing.

`eef2df7` (HaiwangYu, 2026-04-27, in `wcp-porting-img`) then renamed the
mode string in 5 SBND configs from `'multi-3view'` to
`'active3view+masked1view'`:

- `sbnd/sbnd_xin/wct-img-all.jsonnet`
- `sbnd/standalone-sample/wcls-img-dump.jsonnet`
- `sbnd/wcls-img-clus-matching.jsonnet`
- `sbnd/wcls-img-clus.jsonnet`
- `sbnd/wcls-img-dump.jsonnet`

But `eef2df7` did **not** add a matching case in
`cfg/pgrapher/experiment/sbnd/img.jsonnet`. The new string is not one of
the two recognized by the conditional, so it falls into the single-3view
branch — re-disabling the 2-view active branches that `32852232` had
restored. Live tiling silently regressed to single-3view-only across all
five configs.

This commit restores `'multi-3view'` in
`sbnd_xin/wct-img-all.jsonnet:37`. The other four configs share the same
latent regression and are intentionally left untouched — out of scope for
this task. They should be revisited as a follow-up.

## Dead-blob tiling — already 2-dead

The masked fork is invoked unconditionally for every mode that is not
`"single"` / `"active"` / `"masked"`. Both the regressed
`'active3view+masked1view'` mode and the restored `'multi-3view'` mode
land in the same `else` block, which runs
`multi_masked_2view_slicing_tiling(..., span=500)` followed by
`BlobClustering`. The dead branch list above (`{U+V, V+W, U+W}` masked
with the third plane as dummy) is the "2-dead-plane tiling" requested.

No change needed for dead-blob tiling.

## Dead blobs in bee files — clustering stage, via `save_deadarea`

Dead blobs are persisted at **two** stages, but only the second produces
bee files:

1. **Imaging stage** — `sbnd_xin/wct-img-all.jsonnet` ports 0/1 wire the
   per-anode fanpipe outputs to two `ClusterFileSink` nodes (`format:
   'numpy'`):
   - Port 0 → `icluster-apa<N>-active.npz` (live)
   - Port 1 → `icluster-apa<N>-masked.npz` (dead)
   These are raw cluster-tree NPZ archives, not bee format. The helper
   script `wct-img-2-bee.py` converts only the active NPZ to bee for
   live-blob inspection; it does not carry dead blobs into bee.

2. **Clustering stage** — `sbnd_xin/clus.jsonnet`:
   - Two `BlobSampler`s, one per scope:
     `bs_live_face` (`strategy=['stepped']`) and
     `bs_dead_face` (`strategy=['center']`) at l.77–94.
   - `PointTreeBuilding` (l.101–112) merges them into a 2-multiplicity
     point tree with tags `['live','dead']` and scopes `{ '3d': live,
     dead: dead }`.
   - Per-face `MultiAlgBlobClustering` (l.134–164) and all-APA
     `MultiAlgBlobClustering` (l.208–246) both set
     `save_deadarea: true` and emit bee `.zip` files:
     - `mabc-<apa>-face<f>.zip` (per face)
     - `mabc-all-apa.zip` (all APAs)
   - `save_deadarea: true` is the flag that pulls the dead-area / dead
     point set into the bee output. The clustering pipeline also calls
     `cm.live_dead(dead_live_overlap_offset=2)` and
     `cm.extend(..., num_dead_try=1)`, so dead blobs additionally
     participate in cluster extension.

So the answer to "imaging or clustering?" is **clustering** — dead blobs
enter the final bee `.zip` files via `MultiAlgBlobClustering` with
`save_deadarea: true`. No change is needed in either `clus.jsonnet` or
the MABC config; both are already correct.

## Summary

| Goal | State before this commit | State after this commit |
|---|---|---|
| Live: 3-live + 2-live + 1-dead | regressed to single-3-view (mode-string mismatch from eef2df7) | restored via mode `'multi-3view'` |
| Dead: 2-dead-plane (1 dummy + 2 masked) | already in place | unchanged |
| Dead blobs → final bee files | already in place via `save_deadarea: true` (clustering stage) | unchanged |

## Verification

1. Re-run `run_img_evt.sh` (or equivalent) on `evt2` and grep the
   wire-cell log for active slicer branches — four
   `multi_active_slicing_tiling-<anode>` slicings should fire per anode
   (one for each of the four `active_planes` combos), versus only one
   under the regressed mode. Active blob count per anode should jump
   accordingly; the prior measurement was 3114 → 4260 on evt2/APA0.
2. `icluster-apa<N>-masked.npz` should be byte-identical before/after,
   since the masked fork was unchanged.
3. Open one `mabc-<apa>-face<f>.zip` and confirm both the live
   `clustering/3d` point set and the dead-area geometry are present —
   this is just a sanity check; `save_deadarea: true` has been on
   throughout.

## Follow-ups (out of scope here)

- Restore `'multi-3view'` in the four sibling SBND configs touched by
  `eef2df7` (`standalone-sample/wcls-img-dump.jsonnet`,
  `wcls-img-clus-matching.jsonnet`, `wcls-img-clus.jsonnet`,
  `wcls-img-dump.jsonnet`).
- If the `active3view+masked1view` / `active2view+masked2view` naming is
  preferred for future configs, add the corresponding aliases to the
  dispatcher in `cfg/pgrapher/experiment/sbnd/img.jsonnet:342` so the
  intent is no longer silently dropped.

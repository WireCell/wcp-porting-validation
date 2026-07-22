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
`save_deadarea: true`.

## Per-TPC dead-area JSON (wire-cell-bee3 v2 schema)

`wire-cell-bee3` commits `1967cd1` and `293539a` changed the bee server
to (1) load *every* `channel-deadarea*.json` file in an upload as a
separate mesh and (2) honor an optional per-TPC wrapper format that
places the dead-area slab on the correct anode face for multi-TPC
detectors. Without the wrapper, all polygons render on the
most-negative-X face of the union envelope — wrong for SBND TPC 1.

The v2 wrapper schema (see `wire-cell-bee3/docs/dead-area.md`):

```json
{
  "version": 2,
  "tpc": <int>,
  "polygons": [ [[y0,z0], [y1,z1], ...], ... ]
}
```

`tpc` is a zero-based index into the bee3 detector's `experiment.tpc.location`.
For SBND the mapping is `tpc.location[0]` → anode at x = −201.75 cm
(matches WCT anode ident 0) and `tpc.location[1]` → x = +201.75 cm
(matches WCT anode ident 1), so **WCT `apa` ident is the bee3 TPC index
for SBND**. This is only correct for detectors whose anodes are
single-face; multi-face anodes (e.g. PDHD) need a separate
`(apa, face) → tpc` table.

### Toolkit implementation

- `util/inc/WireCellUtil/Bee.h`, `util/src/Bee.cxx`: `Bee::Patches` gains
  an optional `int tpc=-1` constructor parameter. When `tpc >= 0`, the
  underlying `m_data` is an object `{version:2, tpc, polygons:[…]}` and
  `flush`/`clear`/`size`/`empty` operate on `m_data["polygons"]`. When
  `tpc == -1` (default), the legacy bare-array form is preserved for
  back-compat with old bee viewers.
- `clus/.../MultiAlgBlobClustering.{h,cxx}`: new config knob
  `dead_area_version` (default `1` = legacy). When `≥ 2`, each
  per-(apa, face) `Patches` is constructed with `tpc = apa`.
- `sbnd_xin/clus.jsonnet`: both `MultiAlgBlobClustering` nodes
  (per-face and all-APA) set `dead_area_version: 2`, so all SBND
  uploads from this directory carry the v2 schema.

Other detectors (PDHD, PDVD, μBooNE, …) stay on legacy bare-array until
they opt in via their own jsonnet; the legacy format still renders
correctly on bee3 (it lands on the union envelope, which is right for
single-TPC detectors).

## Hand-declared dead region (the W-defect band)

The SBND anode has a fixed defect band — the **middle 6-ch W-plane dead area**
(TPC0 W channels `4800–4805` / TPC1 mirror `10438–10443`, already chndb-bad): the
collection (W) plane has no wires, and the induction (U/V) signal is locally
distorted (tear-drops on the gap edge, empty decon in the center). The gap is
**uniform top-to-bottom** — a full vertical column at the band's z. In
`examine_bundles`' relaxed connectivity graph the imaging hole fragments a single
through-going cosmic into a main + associated sub-cluster, which group-aware
`QLMatching` materializes as two clusters (first seen on SBND data evt 1720, APA0:
one ~5240-pt track split into 2789 + 2376 near y ≈ 0, z ≈ 251 cm).

We declare the **whole vertical column** dead on all three planes, driven from one
shared per-TPC channel list in `cfg/pgrapher/experiment/sbnd/dead_regions.jsonnet`
(W only — `region(anode_ident)` = the 6 W channels + full-drift x window +
`gap: true`).

**Why key the gap off the W channels (not a wider U/V wire list).**
The grouping dead check `get_closest_dead_chs` keys on `(wind, x-drift)` with **no
z/y discrimination along a wire**. A **W wind maps to a single z** (collection wires
are vertical), so a dead W wind already describes a full-y, full-drift dead
**column** — exactly the uniform gap. U/V winds map to **diagonal** (y,z) tracks, so
declaring a U/V wire dead would pollute the entire diagonal far from the band. Hence
the region carries **W channels only**; the full-column treatment is done in C++.

**The functional fix (in `clus`, toolkit repo).**
The relaxed-graph bridge (`connect_graph_relaxed`) decides a gap is crossable via
`Grouping::is_good_point()` / `is_good_point_wc()` / `test_good_point()`, which count
a plane OK where there is live charge **or** a *dead wind*. The default-empty
`inject_dead_winds` config on `PointTreeBuilding` registers the region's wires
(here W only). Entries flagged `gap: true` additionally route their **W winds** into
a separate **dead-gap registry**, serialized to a `dead_gap_a*f*pW` point cloud and
reconstituted downstream by `Grouping::build_wire_cache`. A new
`Grouping::in_dead_gap()` projects a point to the W wind and, if it lands in that
registry (x in window), returns **all three planes dead** — so the whole vertical
column is crossable, not just the y ≈ 0 center where the old U∩V∩W triple-crossing
happened to overlap. The check is folded into the three good-point functions only
(not `get_closest_dead_chs`), so charge averaging is untouched. The gap registry is
**separate** from the general `dead_winds[W]` map on purpose: keying off the latter
would turn every chndb-bad W column into a crossable gap.

**Bee visualization.** The 6-ch W dead band is already chndb-bad, so it renders in
the Bee deadarea via the standard channel-deadarea slab (`save_deadarea` /
`dead_area_version: 2`) — no hand-declared dead blob is injected. (An earlier
`masked_channels` / 4th-branch dead-blob path existed for this and has been removed.)

The `inject_dead_winds` config and the `gap` flag default empty/false → non-SBND
configs and production are bit-identical; SBND output changes by design. Validated on
evt 1720: APA0 cluster 6 stays unified (one ~5165-pt cluster, not 2789 + 2376), and
the generalization additionally bridges off-center crossings of the same band.

## Summary

| Goal | State before this commit | State after this commit |
|---|---|---|
| Live: 3-live + 2-live + 1-dead | regressed to single-3-view (mode-string mismatch from eef2df7) | restored via mode `'multi-3view'` |
| Dead: 2-dead-plane (1 dummy + 2 masked) | already in place | unchanged |
| Dead blobs → final bee files | already in place via `save_deadarea: true` (clustering stage) | unchanged |
| Dead-area slab placement on each TPC anode | legacy bare-array JSON → both TPCs rendered on union envelope (wrong for SBND TPC 1) | v2 wrapper JSON with `tpc=apa` → correct per-anode placement on bee3 |
| Hand-declared W-defect band | dead winds (U∩V∩W) + dead blob, local to the y≈0 center patch | dead-gap registry (W-keyed), **full vertical column** dead on all planes; Bee blob removed (chndb-bad slab suffices) |

## Verification

1. Re-run `run_img_evt.sh` (or equivalent) on `evt2` and grep the
   wire-cell log for active slicer branches — four
   `multi_active_slicing_tiling-<anode>` slicings should fire per anode
   (one for each of the four `active_planes` combos), versus only one
   under the regressed mode. Active blob count per anode should jump
   accordingly; the prior measurement was 3114 → 4260 on evt2/APA0.
2. `icluster-apa<N>-masked.npz` should be byte-identical before/after,
   since the masked fork was unchanged.
3. Open one `mabc-<apa>-face<f>.zip` and inspect a
   `channel-deadarea-apa<N>-face0.json` entry — it must be a JSON object
   with `version: 2`, `tpc: <apa>`, and a non-empty `polygons` array
   (not the legacy top-level array form).
4. Upload `mabc-all-apa.zip` to bee3 and confirm both TPCs' dead-area
   slabs render on their respective anode faces (x ≈ −201.5 cm for
   apa0, x ≈ +201.5 cm for apa1) rather than both stacking on the
   negative-X envelope face.

## Follow-ups (out of scope here)

- Restore `'multi-3view'` in the four sibling SBND configs touched by
  `eef2df7` (`standalone-sample/wcls-img-dump.jsonnet`,
  `wcls-img-clus-matching.jsonnet`, `wcls-img-clus.jsonnet`,
  `wcls-img-dump.jsonnet`).
- If the `active3view+masked1view` / `active2view+masked2view` naming is
  preferred for future configs, add the corresponding aliases to the
  dispatcher in `cfg/pgrapher/experiment/sbnd/img.jsonnet:342` so the
  intent is no longer silently dropped.
- Extend `dead_area_version: 2` to multi-face-anode detectors (PDHD,
  PDVD, ICARUS). The current toolkit code passes `tpc = apa`, which is
  only correct when each anode has exactly one face. Multi-face anodes
  need a `(apa, face) → tpc` mapping driven from the bee3 detector
  geometry.

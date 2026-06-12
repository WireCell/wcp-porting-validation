# SBND Q/L matching chain (standalone) — current state

How the charge–light (Q/L) matching chain runs **today**, end to end:
input data → per-APA clustering → `QLMatching` → all-APA clustering → BEE
packaging. Written before the planned refactor of the matching code/chain. For
the matching component internals see
`toolkit/match/docs/qlmatching-code.md`; for the larsoft→WCT port and the
photon-model JSON schema see `toolkit/match/docs/qlmatching-port.md`.

Drivers (two):
- **`sbnd_xin/run_ql_evt.sh [mc|data] <idx|all>`** — *recommended.* Per-event,
  **self-contained**: matches one event at a time from the toolkit's OWN imaging
  output (`run_img_evt.sh` → `work/evt<ID>/{active,masked}.npz`) + that event's
  opflash, writing `work/ql_evt<ID>/mabc-all-apa.zip`. `all` runs events in
  parallel (`_runlib.sh` `batch_*`, `SBND_MAX_JOBS`). See §8.
- **`sbnd_xin/run_clust_QL_evt.sh [mc]`** — legacy all-10 quick-run: one
  `wire-cell` call over yuhw's bundled multi-event larsoft active npz, with the
  2-view dead area imaged in-graph from the SP frames. Single
  `work/ql_mc/mabc-all-apa.zip` (10 events). §1–§7 describe this chain.

The graphs differ in where the **charge clusters** come from: the per-event chain
uses the toolkit's own imaging (`run_img_evt.sh`), so the MC sim→Q/L chain is
self-contained and independent of yuhw's larsoft dumps; the all-10 chain uses
yuhw's active clusters. Both produce the same BEE layers (img / clustering /
2-view dead / op).

---

## 1. Dataflow

```
                 input-10evt-<mode>/                (read-only, yuhw's)
   icluster-apa0-active.npz   opflash_apa0.tar.gz
   icluster-apa1-active.npz   opflash_apa1.tar.gz
   semi-analytical-sbnd.json  (wire-cell-data/sbnd/photodet, via WIRECELL_PATH)
   sp-frames-10evt.tar.bz2    (assembled by the driver from work/evt<ID>/sp-frames.tar.bz2)

  per APA n ∈ {0,1}:
     ClusterFileSource(active) ──────────────────┐ (port 0, /live)
                                                  ├─► clus.per_apa ─┐
     FrameFileSource ─► FrameFanout ─► masked     ┘ (port 1, /dead) │
       imaging (multi_masked_2view_slicing_tiling)                  ├─► FlashTensorToOpticalPCs(n) ─► QLMatching(n)
     TensorFileSource(opflash) ───────────────────────────────────┘   (expand matrix into          (charge–light
                                                                        flash/light/flashlight       match; reads
                                                                        PCs on live root node)       flash from root,
                                                                                                     writes match scalar)
                                                                       │
                                       fan-in (per-APA charge+t0) ─► clus.all_apa (pre-tagging)
                                                                       │  (MultiAlgBlobClustering,
                                                                       │   save_opflash:true)
                                              ┌────────────────────────┴───────────────┐
                                       mabc-all-apa.zip                          TensorFileSink
                                  per event: img + clustering + dead +           (trash-all-apa)
                                  op (flash/Q-L) layers, all in one zip
                                              │
                                              └─► BEE URL  (uploaded as-is; no combine)
```

Light I/O is two graph nodes: **`Sio::TensorFileSource`** reads
`opflash_apa<n>.tar.gz` (a `[nflash, 1+nchan]` matrix), and
**`Aux::FlashTensorToOpticalPCs`** (a generic, detector-agnostic converter in `aux/`)
expands it into the canonical
`flash`/`light`/`flashlight` point clouds on the live root node of the cluster
pctree — the **same schema** the MicroBooNE `UbooneClusterSource` writes.
`QLMatching` reads the flashes from those PCs (no separate flash input port) and
**persists the match into the pctree** — the per-cluster `flash` scalar +
`cluster_t0`, plus (for the BEE op dump) a per-cluster `matched_flash_gid` /
`flashpred` and a self-contained per-root `opflash` PC — so SBND flashes
interoperate with all `clus` tooling (`Cluster::get_flash()`, `ClusterFlashDump`,
`retile`, BEE). See `toolkit/match/docs/qlmatching-code.md` §1/§1a.

**Flash-time correction (`frame_apply_at_caf`).** Newer opflash dumps carry a
per-frame scalar **`frame_apply_at_caf`** (ns) in the *tensor-set* metadata
(`opflash_tensorset_<ident>_metadata.json`). When present, `FlashTensorToOpticalPCs`
adds it to **every** flash/light time so downstream code (matching, `cluster_t0`,
drift-x) sees only the corrected time. This re-references the raw optical clock to
the CAF/trigger frame: an in-time beam flash that reads ≈ −0.7 µs raw lands in the
**0.3–1.9 µs** window after correction (validation signature — see
[`flash-coincidence.md`](flash-coincidence.md)). Controlled by the
`correct_flash_time` knob on `flash_attach` in
`cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` (default **on**):

- key **absent** in the file → no-op (offset 0), original time (the older
  10-event MC/data dumps have empty metadata, so they are byte-identical);
- key **present** + `correct_flash_time: true` → time `+= frame_apply_at_caf`;
- `correct_flash_time: false` → raw, uncorrected time (key not read).

Units are ns on both sides (the dump writes µs × 1000), so the correction is a
plain add. Code: `aux/src/FlashTensorToOpticalPCs.cxx` (offset read from
`data_ts->metadata()`, applied at the single matrix column-0 read site).

**All BEE output now comes from the all-APA clustering node.** The
`clus.all_apa` `MultiAlgBlobClustering` (`save_opflash: true`) dumps, per event,
the charge `img`/`clustering` layers, the dead-area patches **and** the optical
`op` (flash / Q-L matching) layer into a single `mabc-all-apa.zip` — `QLMatching`
no longer writes any BEE JSON itself, and there is no `data-sep/` tree and no
`merge-apa.py`/`bee-upload.sh` combine step. The graph's main tensor output goes
to a throwaway `TensorFileSink` (`trash-all-apa.tar.gz`); the deliverable is the
BEE zip.

**Flash→multi-cluster and original cluster ids in the display.** Two SBND-only
display refinements (both off by default elsewhere, so other detectors are
bit-identical):
- `op` layer — with `bee_flash_per_flash: true` the MABC emits **one row per
  flash** carrying **all** matched cluster ids in `op_cluster_ids` (the predicted
  light `op_pes_pred` is the element-wise sum over the matched clusters), instead
  of one row per `(flash, cluster)`. So a flash matched to several clusters shows
  them together (MicroBooNE-style); the viewer already reads `op_cluster_ids` as
  an array. (Cluster ids here still index `img-global`, as before.)
- `clustering-global` layer — the final flash-time merge (`examine_bundles`,
  `use_flash_t0`) collapses a flash group (main + associated clusters) into one
  cluster with one `cluster_id`. It now also stores, per blob, each member's
  **original (pre-merge) ident** in a `real_cluster_id`/`perblob` array; the Bee
  writer puts that into the per-point `real_cluster_id`, which the viewer uses for
  coloring (falling back to `cluster_id` when absent). So the far-apart members of
  a merged flash group paint with **distinct colors** while `cluster_id` stays the
  merged-group id used for the `op` association. Spatial merges (extend/regular/
  close) earlier in the all-APA stage are unaffected — they stay one object.
- `op` layer, TPC0/TPC1 flash grouping — with `flash_group_window: 80*wc.ns` on the
  all-APA MABC, flashes from the two TPC sides within ±80 ns are tagged with a
  shared `op_flash_group` id. The MABC computes this **pre-pipeline** (a pure
  function of flash times) and stashes a per-flash `group` array on the root
  `opflash` PC, so the op dump reads it and every later pipeline step can reuse it.
  (It must be pre-pipeline: the op dump runs before clustering so the 1:1
  cluster↔flash mapping and the `flashpred` predicted-PE arrays are still intact —
  both are gone after `examine_bundles` merges. `switch_scope`/post-pipeline were
  tried and fail for exactly this reason.) The viewer shows a whole group together
  (both flashes, TPC-labeled, + the union of their matched clusters); absent the
  field it falls back to one-flash-at-a-time, so older op.json still works.

**Dead area is generated in-toolkit, on the dead side — not by QLMatching.** The
active clusters feed PointTreeBuilding port 0 → the `/live` pctree (what QLMatching
matches to flashes); the in-graph masked imaging feeds port 1 → the `/dead` pctree
(what `MABC::fill_bee_patches_from_cluster` dumps as `channel-deadarea-*` patches).
QLMatching never reads the dead area. Using the toolkit's `multi_masked_2view_slicing_tiling`
(2 dead views + dummy) instead of yuhw's 1-dead-view larsoft `masked.npz` yields the
correct localized 2-view dead patches — identical to the standard `run_clus_evt.sh`
chain. The live (active) clusters and opflash are unchanged, so matching is unaffected.

The bundled npz/opflash hold **all 10 events**, so one `wire-cell` invocation
processes the whole sample (events indexed 0..9 internally). The driver assembles
`sp-frames-10evt.tar.bz2` in that same event order so PointTreeBuilding pairs each
event's `/live` and `/dead` clusters.

---

## 2. Input data

Reached via the symlink `sbnd_xin/input_files` →
`/nfs/data/1/yuhw/wcp-porting-img/sbnd/standalone-sample`, which contains
`input-10evt-mc/` (MC) and `input-10evt-data/` (data). **These are yuhw's,
read-only** — the driver never writes into them; it symlinks them into a
writable `work/` dir.

| File | Producer | Contents |
|------|----------|----------|
| `icluster-apa{0,1}-active.npz` | `wcls-img-dump.fcl` | per-APA **live** `ICluster` dump (blobs/wires/channels) for all 10 events, serialized as a numpy archive read by `ClusterFileSource` |
| `opflash_apa{0,1}.tar.gz` | `wcls-flash-dump.fcl` | per-APA optical flashes as a tensor archive (`prefix: "opflash_"`), 2-D `[nflash, 1+nchan]`; read by `Sio::TensorFileSource` and expanded into the canonical `flash`/`light`/`flashlight` PCs on the live root node by `Aux::FlashTensorToOpticalPCs` |
| `sp-frames-10evt.tar.bz2` | **driver** (this chain) | combined SP frames for the 10 events, concatenated by `run_clust_QL_evt.sh` from the per-event `work/evt<ID>/sp-frames.tar.bz2` (members are uniquely event-id–suffixed) in the active-npz event order. Read by `FrameFileSource`; the matching graph **images the dead/masked clusters in-toolkit** (`multi_masked_2view_slicing_tiling`) → the `/dead` pctree. Replaces the old larsoft `icluster-apa*-masked.npz` (which was tiled 1-dead-view, giving full-height dead strips). |
| `semi-analytical-sbnd.json` | `build-semi-analytical-data/` (one-off) | SBND photon model: `VUVHits`, `VISHits`, `Geometry`, 312 `OpDets`; loaded by `QLMatching` via `Persist::load` (found on `WIRECELL_PATH` at `wire-cell-data/sbnd/photodet/`). Schema → `qlmatching-port.md`. |

`mc` ⇒ `reality=sim` (`QLMatching data:false`); `data` ⇒ `reality=data`
(`data:true`).

> **`data` mode frames.** The in-graph dead-area imaging needs each event's
> `work/evt<ID>/sp-frames.tar.bz2`. These are now self-extracted by
> `run_img_evt.sh <mode>` (or `run_sp_to_magnify_evt.sh <mode>`) from
> `input_files/input-10evt-<mode>/frames-dnn.tar.bz2`, for both mc and data. Run
> `run_img_evt.sh data all` first to populate `work/evt<dataid>/`, then
> `run_clust_QL_evt.sh data`. (The per-event `run_ql_evt.sh` in §8 is the
> recommended driver and needs no combined `sp-frames-10evt.tar.bz2`.)

**Reference output** (mc only): `input_files/input-10evt-mc/archive-runs/wct-standalone-10ev/<n>/`
is yuhw's saved `data-sep` from his standalone run. Note this chain now uses the
in-tree `clus.jsonnet` (different clustering than yuhw's `../clus.jsonnet`), so
its output legitimately differs from that archive — see §6.

---

## 3. The driver — `run_clust_QL_evt.sh`

```
./run_clust_QL_evt.sh [mc|data] [--upload]      # default: mc, build-only
```

What it does:
1. Sets `WIRECELL_PATH = toolkit/cfg : wire-cell-data : wire-cell-data/sbnd/photodet`
   (the last entry lets `Persist::load` find `semi-analytical-sbnd.json`).
2. Makes a fresh writable `work/ql_<mode>/` and **symlinks** the 6 inputs into
   it (inputs read from CWD: `opflash_*.tar.gz` is read by name from CWD, the
   npz via `input=.`).
3. Runs `wire-cell` from that dir (outputs land there):
   ```
   wire-cell -l stderr -l "$LOG:debug" -L debug \
     -V reality=<sim|data> -V input=. -V semimodel_file=semi-analytical-sbnd.json \
     -C DL=6.2 -C DT=9.8 -C lifetime=6 \
     -c <sbnd_xin>/wct-clus-matching-standalone.jsonnet
   ```
   **`-V` = ext-str (string), `-C` = ext-code (numeric)** — the jsonnet does
   `std.extVar('DL') * wc.cm2`, so the numeric vars must be `-C` (the jsonnet's
   header comment showing `-V DL=…` is wrong and would error). DL/DT/lifetime are
   editable vars at the top of the script; defaults are the documented sim set.
   (Note: `drift_speed` is **not** passed — both the clustering and `QLMatching`
   take it from the common SBND config (`simparams.jsonnet`: 1.563 mm/us) via
   `params.lar.drift_speed`, so charge and light share one value.)
4. The run already produces `mabc-all-apa.zip` (the complete BEE zip; §5).
   **Upload is opt-in**: default is build-only; `--upload` uploads that zip
   directly via `../upload-to-bee.sh` and prints the BEE URL (no combine step).

---

## 4. The config

- **`sbnd_xin/wct-clus-matching-standalone.jsonnet`** — the standalone graph,
  self-contained under `sbnd_xin` (no `../sbnd` dependency). It imports the
  **local** `clus.jsonnet` and `qlmatching.jsonnet` (thin re-exports of the in-tree
  canonical `pgrapher/experiment/sbnd/{clus,qlmatching}.jsonnet`, resolved on
  `WIRECELL_PATH`), builds per-APA `ClusterFileSource` + `clus.per_apa` +
  `qlm.opflash_source`(`TensorFileSource`) + `qlm.flash_attach`(**`FlashTensorToOpticalPCs`**)
  + `qlm.matching`(`QLMatching`, single input, `nin=1`), fans into `clus.all_apa`, and
  declares the plugins (incl. `WireCellAux`, `WireCellSio`, `WireCellMatch`).
- The matching nodes and the SBND matching constants (`nchan`, `ch_mask`) come from the
  canonical `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` helper (factory
  `function(params)`, mirroring `clus.jsonnet`); `drift_speed` flows from
  `params.lar.drift_speed`. The standalone just calls `qlm.opflash_source/flash_attach/matching`.
- Per-APA wiring: `clus.per_apa`→`FlashTensorToOpticalPCs` port 0 (pctree) and
  `TensorFileSource`(opflash)→`FlashTensorToOpticalPCs` port 1 (light), then
  →`QLMatching`. The `Aux::FlashTensorToOpticalPCs` fan-in (`nchan: 312`) expands the
  opflash matrix into the canonical `flash`/`light`/`flashlight` PCs on the live
  root node; `QLMatching` reads them there and writes back a per-cluster `flash`
  scalar (no 2-port `QLMatching` fanin).
- **`clus.all_apa(dump=true)`** — the in-tree module is the **pre-tagging**
  chain (no neutrino-tagging downstream, no `nu_tagging` parameter), so no
  `tagger_info` handoff is required and the run does not hit the
  `TaggerCheckNeutrino` null-`main_cluster` segfault.

### Clean break vs yuhw's reference jsonnets
`QLMatching` changed from a 2-input `ITensorSetFanin` (cluster + flash ports) to
a 1-input `ITensorSetFilter` (cluster only; flash read from the live root-node
`flash` PC). Only this `sbnd_xin` chain was updated. yuhw's reference jsonnets in
`wcp-porting-img/sbnd/{standalone-sample/,}` still wire `QLMatching` with 2
ports and will need a matching update by their owner before they run again.

---

## 5. Outputs & BEE link

In `work/ql_<mode>/` the single deliverable is **`mabc-all-apa.zip`**, written
directly by the all-APA `MultiAlgBlobClustering`. Per event `<n>` it holds:
- `<n>-img-global.json` / `<n>-clustering-global.json` — charge (3-D points),
  pre- and post-clustering;
- `<n>-channel-deadarea-apa{0,1}-face0.json` — dead-area patches (v2, per-TPC);
- `<n>-op.json` — the **light / Q-L matching** layer (`op_t`, `op_pes`,
  `op_pes_pred`, `op_peTotal`, `cluster_id`; every flash, matched or not — the
  `cluster_id` indexes the `img-global` clusters). Flashes are emitted in
  ascending `op_t` (flash-time) order, so the viewer's flash next/prev steps
  low→high in time instead of the old first-seen order. See `qlmatching-code.md` §5.

This zip is already the complete event-display set — no `data-sep/`, no
`merge-apa.py`, no `bee-upload.sh` combine. To get a BEE link:
```
cd work/ql_mc
BROWSER=echo bash /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/upload-to-bee.sh mabc-all-apa.zip
```
or re-run `./run_clust_QL_evt.sh mc --upload`. In the viewer, the **`op`** layer
is the charge–light matching; `img`/`clustering` are the imaging/clustering.

---

## 6. Verification & known state (mc)

- **Matching decisions are unchanged.** The matching math + the persisted match
  (per-cluster `flash` scalar / `cluster_t0`) are byte-identical to the saved
  baseline (verified while the legacy `data-sep` dump still existed: all 40
  `{img,op}-apa{0,1}` × 10 JSONs were identical). The op layer now produced by
  the MABC equals that legacy `dump_light` op (`op_t`/`op_pes`/`op_peTotal`/
  `op_pes_pred`), with `cluster_id` now indexing `img-global` (a matched flash
  resolves to the same physical cluster). **Known degeneracy (to fix later with
  matching-algorithm tuning):** the pctree match is one-flash-per-cluster, so a
  handful of legacy per-*bundle* display rows (one cluster matched to two
  flashes, or a duplicate `(flash,cluster)` bundle) collapse to the single
  persisted last-wins match (~5 rows over 4/10 mc events).
- **Output differs from yuhw's archived reference**, by design: this chain now
  uses the in-tree canonical `clus.jsonnet` (pre-tagging) instead of yuhw's
  `../clus.jsonnet`, a different clustering graph → different clusters →
  different matches. That switch (for standalone-ness) was made deliberately and
  is independent of the light-IO refactor.

### Open items
- **Multi-flash-per-cluster degeneracy** (above): the LASSO match can assign one
  cluster to two flashes; the one-flash-per-cluster pctree keeps the last. Revisit
  with matching-algorithm tuning if those rows matter for the display.
- BEE packaging is now self-contained (the MABC writes the one zip; only
  `../upload-to-bee.sh` is shared, for the network upload). `bee-upload.sh` /
  `merge-apa.py` are no longer used by this chain.

---

## 7. Where things live

| Thing | Path |
|-------|------|
| driver | `sbnd_xin/run_clust_QL_evt.sh` |
| standalone jsonnet | `sbnd_xin/wct-clus-matching-standalone.jsonnet` |
| clustering config | `sbnd_xin/clus.jsonnet` → in-tree `pgrapher/experiment/sbnd/clus.jsonnet` |
| matching config | `sbnd_xin/qlmatching.jsonnet` → in-tree `pgrapher/experiment/sbnd/qlmatching.jsonnet` |
| inputs (read-only) | `sbnd_xin/input_files/input-10evt-{mc,data}/` |
| work dir | `sbnd_xin/work/ql_<mode>/` |
| BEE op dump | `toolkit/clus/` (`MultiAlgBlobClustering` `save_opflash`) + `toolkit/util/` (`Bee::Flashes`) |
| BEE upload (shared) | `wcp-porting-img/upload-to-bee.sh` |
| light-IO components | `toolkit/sio/` (`TensorFileSource`) + `toolkit/aux/` (`FlashTensorToOpticalPCs`) |
| matching component | `toolkit/match/` (`WireCellMatch`: `QLMatching`, `Opflash`, `TimingTPCBundle`) |
| photon model JSON | `wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json` |

---

## 8. Per-event self-contained chain (`run_ql_evt.sh`)

The recommended driver for new work. Unlike the all-10 chain (§1–§7), it processes
**one event at a time** and sources the **charge clusters from the toolkit's own
imaging** (`run_img_evt.sh`), not yuhw's larsoft dumps — so the MC sim→Q/L chain is
self-contained and parallelizable.

**Workflow** (per event): `run_img_evt.sh <mode> <idx>` → `run_ql_evt.sh <mode> <idx>`.

```
work/evt<ID>/icluster-apa{0,1}-active.npz  ┐(port 0, /live)   (toolkit imaging,
work/evt<ID>/icluster-apa{0,1}-masked.npz  ┘(port 1, /dead)    run_img_evt.sh)
        → clus.per_apa → FlashTensorToOpticalPCs → QLMatching → clus.all_apa
opflash_apa{0,1}.tar.gz (this event, split from input-10evt-<mode>/) ┘
        → work/ql_evt<ID>/mabc-all-apa.zip   (img + clustering + 2-view dead + op)
```

- Graph: `wct-clus-matching-perevt.jsonnet` — mirrors `wct-clustering.jsonnet`
  (reads active+masked from npz; **no in-graph imaging**) plus the opflash source /
  `FlashTensorToOpticalPCs` / `QLMatching` nodes. TLAs: `input`, `output_dir`,
  `anode_indices`, `run/subrun/event`, `reality`, `semimodel_file`, `DL/DT/lifetime/driftSpeed`.
- Driver: `run_ql_evt.sh [mc|data] <idx|all> [-a anode]`. Requires `work/evt<ID>/`
  imaging output (errors with a pointer to `run_img_evt.sh` if missing); splits the
  event's opflash into `work/ql_evt<ID>/`; runs the graph. `all` fans out over events
  in parallel via `_runlib.sh` `batch_*` (`SBND_MAX_JOBS`). Output is one
  `work/ql_evt<ID>/mabc-all-apa.zip` per event (upload individually, or combine).
- Verified (mc): per-event dead area is **identical** to `run_clus_evt.sh` (same
  `work/evt<ID>/masked.npz`); op layer self-consistent (0 dangling `op_cluster_ids`,
  legacy schema); all 10 events run clean in parallel.

> **Data** now runs the same way. The data SP frames are provided in
> `input_files/input-10evt-data/frames-dnn.tar.bz2`, so
> `run_img_evt.sh data <idx>` produces `work/evt<dataid>/` and
> `run_ql_evt.sh data <idx>` then matches against that event's opflash
> (`input-10evt-data/opflash_apa{0,1}.tar.gz`). The event-id list is derived from
> the data frames archive, so `data all` fans out over all 10 data events.

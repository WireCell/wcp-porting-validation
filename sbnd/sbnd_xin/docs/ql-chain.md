# SBND Q/L matching chain (standalone) — current state

How the charge–light (Q/L) matching chain runs **today**, end to end:
input data → per-APA clustering → `QLMatching` → all-APA clustering → BEE
packaging. Written before the planned refactor of the matching code/chain. For
the matching component internals see
`toolkit/match/docs/qlmatching-code.md`; for the larsoft→WCT port and the
photon-model JSON schema see `toolkit/match/docs/qlmatching-port.md`.

Driver: **`sbnd_xin/run_clust_QL_evt.sh`**.

---

## 1. Dataflow

```
                 input-10evt-<mode>/                (read-only, yuhw's)
   icluster-apa0-active.npz  icluster-apa0-masked.npz   opflash_apa0.tar.gz
   icluster-apa1-active.npz  icluster-apa1-masked.npz   opflash_apa1.tar.gz
   semi-analytical-sbnd.json (wire-cell-data/sbnd/photodet, via WIRECELL_PATH)

  per APA n ∈ {0,1}:
     ClusterFileSource(active)  ┐
                                ├─► clus.per_apa ─┐
     ClusterFileSource(masked)  ┘   (clustering)  ├─► OpflashToFlashPCs(n) ─► QLMatching(n)
     TensorFileSource(opflash) ───────────────────┘   (expand matrix into          (charge–light
                                                        flash/light/flashlight       match; reads
                                                        PCs on live root node)       flash from root,
                                                                                     writes match scalar)
                                                                       │
                                       fan-in (per-APA charge+t0) ─► clus.all_apa (pre-tagging)
                                                                       │
                                              ┌────────────────────────┼───────────────┐
                                       data-sep/<n>/            mabc-*.zip        TensorFileSink
                                   <n>-img/op-apa{0,1}.json  (clustering layers)  (trash-all-apa)
                                              └──────────── bee-upload.sh ─────────┘
                                                       merge-apa.py + union
                                                              │
                                                        combined.zip ─► BEE URL
```

Light I/O is two graph nodes: **`Sio::TensorFileSource`** reads
`opflash_apa<n>.tar.gz` (a `[nflash, 1+nchan]` matrix), and
**`Aux::OpflashToFlashPCs`** (a generic, detector-agnostic converter in `aux/`)
expands it into the canonical
`flash`/`light`/`flashlight` point clouds on the live root node of the cluster
pctree — the **same schema** the MicroBooNE `UbooneClusterSource` writes.
`QLMatching` reads the flashes from those PCs (no separate flash input port) and
writes back a per-cluster `flash` scalar, so SBND flashes interoperate with all
`clus` tooling (`Cluster::get_flash()`, `ClusterFlashDump`, `retile`, BEE). See
`toolkit/match/docs/qlmatching-code.md` §1/§1a.

Two side outputs feed BEE: **`data-sep/`** (written by `QLMatching`, the img +
op layers) and **`mabc-*.zip`** (written by the clustering nodes). The graph's
main tensor output goes to a throwaway `TensorFileSink`
(`trash-all-apa.tar.gz`) — the deliverable is the BEE JSON, not the tensor
stream.

The bundled npz/opflash hold **all 10 events**, so one `wire-cell` invocation
processes the whole sample (events indexed 0..9 internally).

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
| `icluster-apa{0,1}-masked.npz` | `wcls-img-dump.fcl` | per-APA **dead/masked**-region clusters (the dead-area blobs; deserialized via `aux/ClusterArrays::to_cluster`, the `nudge=1e-3` dead-blob fix) |
| `opflash_apa{0,1}.tar.gz` | `wcls-flash-dump.fcl` | per-APA optical flashes as a tensor archive (`prefix: "opflash_"`), 2-D `[nflash, 1+nchan]`; read by `Sio::TensorFileSource` and expanded into the canonical `flash`/`light`/`flashlight` PCs on the live root node by `Aux::OpflashToFlashPCs` |
| `semi-analytical-sbnd.json` | `build-semi-analytical-data/` (one-off) | SBND photon model: `VUVHits`, `VISHits`, `Geometry`, 312 `OpDets`; loaded by `QLMatching` via `Persist::load` (found on `WIRECELL_PATH` at `wire-cell-data/sbnd/photodet/`). Schema → `qlmatching-port.md`. |

`mc` ⇒ `reality=sim` (`QLMatching data:false`); `data` ⇒ `reality=data`
(`data:true`).

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
4. Packages via `bee-upload.sh` (§5). **Upload is opt-in**: default builds
   `combined.zip` only (a runtime stub replaces the uploader); `--upload` runs
   the real upload and prints the BEE URL.

---

## 4. The config

- **`sbnd_xin/wct-clus-matching-standalone.jsonnet`** — the standalone graph,
  self-contained under `sbnd_xin` (no `../sbnd` dependency). It imports the
  **local** `clus.jsonnet` and `qlmatching.jsonnet` (thin re-exports of the in-tree
  canonical `pgrapher/experiment/sbnd/{clus,qlmatching}.jsonnet`, resolved on
  `WIRECELL_PATH`), builds per-APA `ClusterFileSource` + `clus.per_apa` +
  `qlm.opflash_source`(`TensorFileSource`) + `qlm.flash_attach`(**`OpflashToFlashPCs`**)
  + `qlm.matching`(`QLMatching`, single input, `nin=1`), fans into `clus.all_apa`, and
  declares the plugins (incl. `WireCellAux`, `WireCellSio`, `WireCellMatch`).
- The matching nodes and the SBND matching constants (`nchan`, `ch_mask`) come from the
  canonical `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` helper (factory
  `function(params)`, mirroring `clus.jsonnet`); `drift_speed` flows from
  `params.lar.drift_speed`. The standalone just calls `qlm.opflash_source/flash_attach/matching`.
- Per-APA wiring: `clus.per_apa`→`OpflashToFlashPCs` port 0 (pctree) and
  `TensorFileSource`(opflash)→`OpflashToFlashPCs` port 1 (light), then
  →`QLMatching`. The `Aux::OpflashToFlashPCs` fan-in (`nchan: 312`) expands the
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

In `work/ql_<mode>/`:
- `data-sep/<n>/<n>-img-apa{0,1}.json` — charge (3-D points) per APA, per event.
- `data-sep/<n>/<n>-op-apa{0,1}.json` — light/matching per APA, per event (the
  **matching result**: `op_t`, `op_pes`, `op_pes_pred`, `cluster_id`; see
  `qlmatching-code.md` §5).
- `mabc-all-apa.zip` (+ `mabc-apa{0,1}-face0.zip`) — clustering layers.

Packaging (`wcp-porting-img/sbnd/bee-upload.sh`, runs on `$PWD`):
1. unzip `mabc-*.zip` into `data/<n>/`;
2. `merge-apa.py --inpath=data-sep --outpath=data --eventNo=<n>` merges the
   per-APA `img`/`op` into `data/<n>/<n>-img.json` / `-op.json`;
3. zip `data/` → `combined.zip`;
4. upload via `../upload-to-bee.sh` → prints `https://www.phy.bnl.gov/twister/bee/set/<UUID>/event/list/`.

To get a BEE link with matching from an already-built zip:
```
cd work/ql_mc
BROWSER=echo bash /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/upload-to-bee.sh combined.zip
```
or re-run `./run_clust_QL_evt.sh mc --upload`. In the viewer, the **`op`** layer
is the charge–light matching; `img`/`clustering` are the imaging/clustering.

---

## 6. Verification & known state (mc)

- **Canonical-flash consolidation is bit-identical.** Expanding the opflash
  matrix into the canonical `flash`/`light`/`flashlight` root-node PCs (via
  `Aux::OpflashToFlashPCs`) and rebuilding `Opflash` from them reproduces the
  baseline **byte-for-byte**: all 40 `data-sep` JSONs (`{img,op}-apa{0,1}` × 10
  events) are identical — the matching math is unchanged. The output tensorset
  additionally gains the canonical flash PCs + a per-cluster matched-flash
  scalar (verified: `Cluster::get_flash()` returns valid for matched clusters),
  which the `data-sep` BEE files do not depend on.
- **Output differs from yuhw's archived reference**, by design: this chain now
  uses the in-tree canonical `clus.jsonnet` (pre-tagging) instead of yuhw's
  `../clus.jsonnet`, a different clustering graph → different clusters →
  different matches. That switch (for standalone-ness) was made deliberately and
  is independent of the light-IO refactor.

### Open items
- **BEE packaging** still reaches into `../sbnd` (`bee-upload.sh`,
  `merge-apa.py`, `../upload-to-bee.sh`); to be localized later.

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
| packaging (still ../sbnd) | `wcp-porting-img/sbnd/{bee-upload.sh,merge-apa.py}`, `wcp-porting-img/upload-to-bee.sh` |
| light-IO components | `toolkit/sio/` (`TensorFileSource`) + `toolkit/aux/` (`OpflashToFlashPCs`) |
| matching component | `toolkit/match/` (`WireCellMatch`: `QLMatching`, `Opflash`, `TimingTPCBundle`) |
| photon model JSON | `wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json` |

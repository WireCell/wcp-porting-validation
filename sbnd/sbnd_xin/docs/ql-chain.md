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
     ClusterFileSource(masked)  ├─► clus.per_apa(dump=false)  ─► QLMatching(matching n)
     TensorFileSource(opflash)  ┘        (clustering)             (charge–light match)
                                                                       │
                              fan-in (per-APA charge+t0) ─► clus.all_apa(nu_tagging=false)
                                                                       │
                                              ┌────────────────────────┼───────────────┐
                                       data-sep/<n>/            mabc-*.zip        TensorFileSink
                                   <n>-img/op-apa{0,1}.json  (clustering layers)  (trash-all-apa)
                                              └──────────── bee-upload.sh ─────────┘
                                                       merge-apa.py + union
                                                              │
                                                        combined.zip ─► BEE URL
```

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
| `opflash_apa{0,1}.tar.gz` | `wcls-flash-dump.fcl` | per-APA optical flashes as a tensor archive (`prefix: "opflash_"`); read by `TensorFileSource` into the 2-D `[nflash, 1+nchan]` tensor QLMatching expects |
| `semi-analytical-sbnd.json` | `build-semi-analytical-data/` (one-off) | SBND photon model: `VUVHits`, `VISHits`, `Geometry`, 312 `OpDets`; loaded by `QLMatching` via `Persist::load` (found on `WIRECELL_PATH` at `wire-cell-data/sbnd/photodet/`). Schema → `qlmatching-port.md`. |

`mc` ⇒ `reality=sim` (`QLMatching data:false`); `data` ⇒ `reality=data`
(`data:true`).

**Reference output** (mc only): `input_files/input-10evt-mc/archive-runs/wct-standalone-10ev/<n>/`
is yuhw's saved `data-sep` from the pre-tagging standalone run — the thing to
diff against.

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
     -C DL=6.2 -C DT=9.8 -C lifetime=6 -C driftSpeed=1.565 \
     -c <sbnd_xin>/wct-clus-matching-standalone.jsonnet
   ```
   **`-V` = ext-str (string), `-C` = ext-code (numeric)** — the jsonnet does
   `std.extVar('DL') * wc.cm2`, so the numeric vars must be `-C` (the jsonnet's
   header comment showing `-V DL=…` is wrong and would error). Drift params are
   editable vars at the top of the script; defaults are the documented sim set.
   (Note: these feed the *clustering* drift model — `QLMatching` itself still
   uses a hardcoded drift speed; see `qlmatching-code.md` §7.)
4. Packages via `bee-upload.sh` (§5). **Upload is opt-in**: default builds
   `combined.zip` only (a runtime stub replaces the uploader); `--upload` runs
   the real upload and prints the BEE URL.

---

## 4. The config

- **`sbnd_xin/wct-clus-matching-standalone.jsonnet`** — a local copy of yuhw's
  `standalone-sample/wct-clus-matching-standalone.jsonnet`. Only change:
  `clus_maker.all_apa(tools.anodes, dump=true, nu_tagging=false)`. It imports
  `../clus.jsonnet` (resolves to `sbnd/clus.jsonnet` from `sbnd_xin/`), builds
  per-APA `ClusterFileSource`/`TensorFileSource` + `clus.per_apa` + `QLMatching`,
  fans into `clus.all_apa`, and declares the plugins (incl. `WireCellMatch`).
- **`sbnd/clus.jsonnet`** (shared) — `all_apa`/`clus_all_apa` gained a
  backward-compatible **`nu_tagging=true`** parameter gating the May-28 qlport
  neutrino-tagging chain (pipeline steps + bee point sets + `bee_pf`). Default
  `true` preserves existing behavior (verified: rendering the unchanged default
  path is **byte-identical** pre/post-edit). The standalone copy sets it `false`.

### Why `nu_tagging=false`
The all-APA tagging chain `tagger_flag_transfer → recover_bundle →
tagger_check_neutrino` (added to `clus.jsonnet` 2026-05-28, **after** yuhw's
reference run) needs a per-cluster `tagger_info` point cloud to raise the
`beam_flash` flag. The merged `QLMatching` does **not** write `tagger_info`, so
`ClusteringTaggerFlagTransfer` sets zero `beam_flash` → `recover_bundle`
recovers nothing → null `main_cluster` → null-deref segfault in
`TaggerCheckNeutrino`. Disabling the chain both removes the crash and faithfully
reproduces the pre-tagging reference. The real fix (write `tagger_info` in
`QLMatching` + a null-guard in the tagger) is left for the refactor.

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

- **Imaging bit-identical** to the reference (`img` point counts match exactly
  for all 10 events × both APAs) — confirms the merged dead-blob +
  `ClusterFileSource` fixes don't perturb the live points.
- **Matching bundle counts differ** slightly from the reference (e.g. evt0 op1
  17 vs 19). This is **expected**: the improved clustering on this branch
  produces different input clusters → different matches. Ruled out drift params
  (header vs sim-detsim set give identical results).

### Open items (for the refactor)
- **`data` mode crashes** in `QLMatching` itself: unmatched clusters build a
  `TimingTPCBundle(nullptr, …)` whose ctor derefs the flash
  (`QLMatching.cxx:629` / `TimingTPCBundle.cxx:50`). MC doesn't hit it.
- **Latent default-tagging crash**: `nu_tagging` defaults to `true`, so any
  other caller of `clus.jsonnet`'s `all_apa` that hits a no-recovered-bundle
  event still segfaults until the `tagger_info` handoff + null-guard land.

---

## 7. Where things live

| Thing | Path |
|-------|------|
| driver | `sbnd_xin/run_clust_QL_evt.sh` |
| standalone jsonnet (copy) | `sbnd_xin/wct-clus-matching-standalone.jsonnet` |
| shared clustering config | `sbnd/clus.jsonnet` (the `nu_tagging` toggle) |
| inputs (read-only) | `sbnd_xin/input_files/input-10evt-{mc,data}/` |
| reference (mc) | `…/input-10evt-mc/archive-runs/wct-standalone-10ev/` |
| work dir | `sbnd_xin/work/ql_<mode>/` |
| packaging | `wcp-porting-img/sbnd/{bee-upload.sh,merge-apa.py}`, `wcp-porting-img/upload-to-bee.sh` |
| matching component | `toolkit/match/` (`WireCellMatch`) |
| photon model JSON | `wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json` |

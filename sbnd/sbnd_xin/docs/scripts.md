# Script Reference (`sbnd_xin/`)

All scripts are run from `sbnd_xin/`. Each sets `WIRECELL_PATH` to include
`toolkit/cfg` and `wire-cell-data` — no manual export needed.

> For the end-to-end pipeline overview, quick start, and common conventions
> (no-arg listing, `IDX=all` parallel mode, `SBND_MAX_JOBS`) see **[sbnd.md](sbnd.md)**.

---

## Common options (every `run_*.sh`)

| Option | Effect |
|---|---|
| `-h`, `--help` | Print the script's usage, the resolved input/work-dir scheme, and the available input sets, then exit. |
| `-N <n>` | Event-sample size: use `input_files/input-<n>evt-<mode>/` (default `10`). E.g. `-N 100` runs the 100-event sample. Equivalent to exporting `SBND_SAMPLE=<n>`. Validated against the directories that actually exist. |
| `[mc\|data]` | Input set (default `mc`); also sets `reality=sim/data` where relevant. |
| *(no `<idx>`)* | List the `idx → EVT_ID` map for the chosen **sample & mode**. |

**"Which directory / which event am I running?"** — run any script with `-h`
(shows the `input_files/input-<N>evt-<mode>/` input dir and the `work/evt<EVT_ID>/`
output scheme) and with no `<idx>` (shows the index→event map). The runtime echoes
the resolved work dir and `EVT_ID` for each event as it runs.

**Samples available:** `input-10evt-{mc,data}` and `input-100evt-{mc,data}`.
The **`mc` 100-event set is malformed** upstream — its `frames-dnn.tar.bz2` lists
100 `frame_dnnsp` members but only **41 unique event ids** (duplicated frames); the
matching opflash covers exactly those 41. `load_events` dedups (first-occurrence
order), so the `mc` 100-set resolves to **41 events**. The `data` 100-set is clean
(100 events, 1:1 with opflash).

---

## Shell scripts (pipeline order)

### `_runlib.sh`

Shared helper library sourced by every `run_*.sh` script. Provides:

| Function | Description |
|---|---|
| `load_events <mode>` | Populate `SBND_EVENTS` (in archive order, **deduped**) from `$(sbnd_input_dir <mode>)/frames-dnn.tar.bz2`; keeps built-in mc list as fallback |
| `ensure_sp_frames <mode> <evt_id> <dest>` | Extract one event's 5-member SP-frame subset from the mode archive into a fresh single-event `<dest>` (re-extracts each call) |
| `sp_frames_archive <mode>` | Echo the mode's `frames-dnn.tar.bz2` path (sample-aware) |
| `sbnd_input_dir <mode>` | Echo `input_files/input-${SBND_SAMPLE}evt-<mode>` (honors `-N` / `SBND_SAMPLE`, default 10) |
| `sbnd_check_sample <mode>` | Verify the chosen sample/mode input dir exists; on failure list available sets and return 1 |
| `sbnd_list_samples` | List the `input-<N>evt-<mode>` sets present under `input_files/` |
| `sbnd_common_help` | Print the shared `-N` / `-h` / paths footer used by every script's `usage()` |
| `list_events` | Print the chosen sample line + idx→EVT_ID mappings; called on no-arg invocation |
| `lookup_evt_id <idx>` | Resolve 1-based index to event ID; error + table on bad input |
| `discover_event_indices` | Print `1 2 … N`; used in `all`-mode loops |
| `batch_init` | Initialise counters and `BATCH_PIDS` assoc array |
| `batch_wait_slot` | Block until fewer than `SBND_MAX_JOBS` (default `nproc`) jobs are running |
| `batch_drain` | Wait for all remaining background jobs |
| `batch_summary` | Print ok/failed counts; returns 0 if at least one event succeeded |

Not invoked directly.

---

### `run_sp_to_magnify_evt.sh`

**Purpose:** Convert SP frames for one event to per-anode Magnify ROOT files
(for visual validation) and per-anode `gauss<N>`-tagged frame archives (input
for `run_select_evt.sh` / Woodpecker).

```
Usage: ./run_sp_to_magnify_evt.sh [mc|data] [-N n] [-s sel_tag] <idx|all> [run] [subrun]
  (no args)  list available events (for the chosen sample & mode); -h for full help
  mode:      mc (default) | data — selects input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2
  idx:       1-based event index into the mode's event list
  all:       process all events in parallel
  run:       run number stored in ROOT Trun tree (default 0)
  subrun:    subrun number (default 0)
  -s:        use work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2 instead of work/evt<ID>/
```

**Input:** `input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2` (per-event subset
extracted to `work/evt<ID>/sp-frames.tar.bz2`). With `-s`, reads
`work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2`.

**Output** (in `work/evt<ID>[_<SEL_TAG>]/`):

| File | Description |
|---|---|
| `magnify-evt<ID>-anode{0,1}.root` | Magnify ROOT (T_bad, T_charge, …) |
| `sbnd-sp-frames-anode{0,1}.tar.bz2` | per-anode `gauss<N>`-tagged archives for Woodpecker |

**Jsonnet driven:** `wct-sp-to-magnify.jsonnet`

**Log:** `work/evt<ID>/wct_magnify_evt<ID>.log`; in `all` mode `work/.batch_magnify_evt<ID>.log`

---

### `run_select_evt.sh`

**Purpose:** Open the Woodpecker browser GUI to select a tick/channel ROI,
then merge the masked per-anode archives back into a combined `dnnsp`-tagged
archive that downstream pipeline scripts consume via `-s <sel_tag>`.

```
Usage: ./run_select_evt.sh [mc|data] [-N n] [-a anode] <idx> <sel_tag>
  (no args) list available events (for the chosen sample & mode); -h for full help
  idx:      1-based event index into the chosen sample/mode
  sel_tag:  short label for this selection (e.g. sel1, tight, track5)
  -a:       restrict to one anode (0 or 1)
```

**Requires:** `run_sp_to_magnify_evt.sh` run first (produces
`work/evt<ID>/sbnd-sp-frames-anode*.tar.bz2`).

**External commands:**
1. `woodpecker select <archive> --detector sbnd --outdir <SELDIR> --prefix sbnd-sp-frames` (per anode)
2. `python3 merge_sel_archives.py <orig> <out> <evt_id> <masked...>`

Sets `MPLBACKEND=WebAgg`; prints SSH port-forward instructions.

**Output** (in `work/evt<ID>_<SEL_TAG>/input/`):

| File | Description |
|---|---|
| `sbnd-sp-frames-anode<N>.tar.bz2` | Woodpecker-masked per-anode archive |
| `selection-anode<N>.json` | tick/channel sidecar |
| `sp-frames.tar.bz2` | combined `dnnsp`-tagged archive for all downstream `-s <sel_tag>` runs |

---

### `run_img_evt.sh`

**Purpose:** Run 3D imaging on one event, producing per-anode active and masked
cluster `.npz` files.

```
Usage: ./run_img_evt.sh [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all>
  (no args)  list available events (for the chosen sample & mode); -h for full help
  mode:      mc (default) | data — selects input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2
  all:       process all events in parallel
  -a:        restrict to one anode (0 or 1)
  -s:        use work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2
```

**Input:** `input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2` — this event's
SP-frame subset is self-extracted to `work/evt<ID>/sp-frames.tar.bz2` (fresh each
run, so imaging tracks the current frames archive). No separate
`run_sp_to_magnify_evt.sh` step is required. With `-s`, reads the
Woodpecker-masked `work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2` instead.

**Output** (in `work/evt<ID>[_<SEL_TAG>]/`):

| File | Description |
|---|---|
| `icluster-apa<N>-active.npz` | live-channel cluster arrays |
| `icluster-apa<N>-masked.npz` | dead-channel cluster arrays |

**Jsonnet driven:** `wct-img-all.jsonnet`

**TLAs forwarded:**

| TLA | Type | Value |
|---|---|---|
| `input` | str | path to self-extracted `work/evt<ID>/sp-frames.tar.bz2` |
| `anode_indices` | code | `[0,1]` or `[<N>]` with `-a` |
| `output_dir` | str | `work/evt<ID>[_<SEL_TAG>]/` |

**Log:** `work/evt<ID>/wct_img_evt<ID>[_a<N>].log`; in `all` mode `work/.batch_img_evt<ID>.log`

---

### `run_clus_evt.sh`

**Purpose:** Per-APA and all-APA blob clustering using `MultiAlgBlobClustering`.
Pre-validates `.npz` files and skips anodes with no active clusters so
`PointTreeMerging` does not stall.

```
Usage: ./run_clus_evt.sh [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all> [run] [subrun]
  (no args)      list available events (for the chosen sample & mode); -h for full help
  mode:          mc (default) | data — selects the event list; sets reality=sim/data
  all:           process all events in parallel
  run / subrun:  stored in Bee RSE metadata (default 0)
  -a:            restrict to one anode; skips all-APA stage
  -s:            use work/evt<ID>_<SEL_TAG>/ as working directory
```

The clustering graph itself is mode-agnostic (`reality` is a passthrough TLA, and
the drift/diffusion defaults below apply to both modes); `mode` only chooses the
event list and the `reality` flag.

**Input:** `work/evt<ID>[_<SEL_TAG>]/icluster-apa{0,1}-{active,masked}.npz`

**Output** (in `work/evt<ID>[_<SEL_TAG>]/`):

| File | Description |
|---|---|
| `mabc-apa<N>-face0.zip` | per-APA clustering Bee zip (includes point cloud) |
| `mabc-all-apa.zip` | all-APA combined clustering |
| `trash-*.tar.gz` | TensorFileSink dump (~29 bytes, harmless) |

**Jsonnet driven:** `wct-clustering.jsonnet`

**TLAs forwarded:**

| TLA | Type | Default | Description |
|---|---|---|---|
| `input` | str | — | directory with `icluster-apa*.npz` |
| `anode_indices` | code | `[0,1]` | anodes to process |
| `output_dir` | str | — | output directory |
| `run` / `subrun` / `event` | code | 0 / 0 / EVT_ID | Bee RSE metadata |
| `reality` | str | `'sim'` | `'sim'` or `'data'` |
| `DL` | code | 4.0 | longitudinal diffusion (cm²/s) |
| `DT` | code | 8.8 | transverse diffusion (cm²/s) |
| `lifetime` | code | 35 | electron lifetime (ms) |
| `driftSpeed` | code | 1.565 | drift speed (mm/µs) |

**Log:** `work/evt<ID>/wct_clus_evt<ID>[_a<N>].log`; in `all` mode `work/.batch_clus_evt<ID>.log`

---

### `run_ql_evt.sh`

**Purpose:** Per-event charge–light (Q/L) matching, **self-contained**. Reads the
toolkit's own imaging output + that event's opflash, runs per-APA clustering +
`QLMatching` + all-APA `MultiAlgBlobClustering`, and writes one Bee zip with the
img / clustering / 2-view dead-area / op (flash + Q/L match) layers. The recommended
matcher driver; see also `docs/ql-chain.md` §8. (The legacy all-10 single-run variant
is `run_clust_QL_evt.sh`.)

```
Usage: ./run_ql_evt.sh [mc|data] [-N n] [-a anode] <idx|all>
  [mc|data]      mode (default mc); selects input_files/input-<N>evt-<mode>/
  (no idx)       list available events for the chosen sample & mode; -h for full help
  all:           process every event in parallel (_runlib.sh batch_*, SBND_MAX_JOBS)
  -a:            restrict to one anode
  -calib:        also dump work/ql_evt<ID>/calib-evt<ID>.json for the ql_scan viewer
env:
  PMT_NL=true|false    predicted-PE non-linearity (default true; false = OFF baseline)
  CALIB_SUFFIX=.nl     insert a suffix in the calib filename (calib-evt<ID>.nl.json),
                       so an NL rerun does not clobber a linear dump
```

**Prerequisite:** `run_img_evt.sh <idx>` first (produces the per-event
`work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz`). Errors cleanly if missing.

**Input:** `work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz` (toolkit imaging) +
`input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz` (split per event by the driver;
events with no opflash in the sample are skipped with a clear message).

**Output:** `work/ql_evt<ID>/mabc-all-apa.zip` (one event; isolated from the
`run_clus_evt.sh` output in `work/evt<ID>/`).

**Jsonnet driven:** `wct-clus-matching-perevt.jsonnet`

**TLAs forwarded:** like `run_clus_evt.sh` plus `semimodel_file` (photon model);
Q/L drift/diffusion defaults `DL=6.2 DT=9.8 lifetime=6 driftSpeed=1.563`.

**Log:** `work/ql_evt<ID>/wct_ql_evt<ID>.log`; in `all` mode `work/.batch_ql_evt<ID>.log`

**Note:** both `mc` and `data` are runnable — `input-{10,100}evt-{mc,data}` all exist.
Run `run_img_evt.sh <mode> [-N n] <idx>` first to produce the per-event imaging the
matcher reads. On the 100-event sets, `data` is fully covered (100 events, 1:1 with
opflash) while `mc` resolves to 41 events (duplicated frames upstream; see Common options).

---

### `run_bee_img_evt.sh`

**Purpose:** Convert imaging `.npz` cluster files to Bee JSON (one file per
anode), package as a zip, and upload to the Bee event-display server.

```
Usage: ./run_bee_img_evt.sh [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all> [run] [subrun]
  (no args)  list available events (for the chosen sample & mode); -h for full help
  mode:      mc (default) | data — selects the event list
  all:       combine all events into one upload zip and do a single Bee upload
  -a:        restrict to one anode
  -s:        use work/evt<ID>_<SEL_TAG>/ as working directory
```

**Input:** `work/evt<ID>[_<SEL_TAG>]/icluster-apa{0,1}-active.npz` (skips
empty/22-byte files automatically).

**Single-event path** (`idx` is a number):
1. `python wct-img-2-bee.py <run> <subrun> <evt> <N>:<path> ...` → `data/0/0-apa<N>.json` + `upload.zip`
2. `mv upload.zip upload_evt<ID>[_<SEL>][_a<N>].zip`
3. `./upload-to-bee.sh <zipname>`

**`all`-mode path**:
- Invokes `wirecell-img bee-blobs` directly for each anode in parallel.
- Writes `data/<bee_idx>/<bee_idx>-apa<N>.json` (filename prefix matches
  directory index — required by Bee's `parse_pathname` to distinguish events).
- Produces `upload-batch.zip` and does a single upload.

**Output:**
- Single-event: `upload_evt<ID>[_<SEL_TAG>][_a<N>].zip` in `sbnd_xin/`
- All-mode: `upload-batch.zip` in `sbnd_xin/`

---

### `upload-to-bee.sh`

Symlink to `../../upload-to-bee.sh`. Uploads the given zip to the Bee server.
Called automatically by `run_bee_img_evt.sh`.

---

## Jsonnet entry-points

### `wct-sp-to-magnify.jsonnet`

Converts a `dnnsp`-tagged SP frame archive into per-anode Magnify ROOT files
and per-anode `gauss<N>`-tagged frame archives for Woodpecker.

**Imports:** `pgrapher/experiment/sbnd/simparams.jsonnet`,
`pgrapher/common/tools.jsonnet`, `magnify-sinks.jsonnet`

**TLAs:**

| TLA | Type | Default | Description |
|---|---|---|---|
| `input` | str | `'sp-frames.tar.bz2'` | input frame archive |
| `anode_indices` | code | `[0,1]` | anodes to process |
| `output_file_prefix` | str | `'magnify'` | prefix for `.root` outputs |
| `sp_frame_prefix` | str | `'sbnd-sp-frames'` | prefix for `.tar.bz2` outputs |
| `run` / `subrun` / `event` | code | 0 / 0 / 0 | stored in ROOT Trun tree |
| `nticks` | code | 3427 | total ticks (matches actual SP-frame readout); written to `Trun.total_time_bin` |

**Pipeline per anode:**
```
FrameFileSource(dnnsp)
  → FrameFanout (rename dnnsp→dnnsp<N> per anode)
  → ChannelSelector (5638-wide, keeps only anode N's channels)
  → tap: Retagger(dnnsp<N>→gauss<N>) → FrameFileSink(sbnd-sp-frames-anode<N>.tar.bz2)
  → Retagger(dnnsp<N>→[dnnsp<N>, threshold<N>])
  → MagnifySink(magnify-evt<ID>-anode<N>.root)
  → DumpFrames
```

---

### `wct-img-all.jsonnet`

Runs 3D imaging on both anodes, writing active and masked cluster arrays.

**Imports:** `pgrapher/experiment/sbnd/simparams.jsonnet`,
`pgrapher/experiment/sbnd/img.jsonnet`

**TLAs:**

| TLA | Type | Default | Description |
|---|---|---|---|
| `input` | str | `'sp-frames.tar.bz2'` | input frame archive |
| `anode_indices` | code | `[0,1]` | anodes to process |
| `output_dir` | str | `''` | directory for output `.npz` files |

**Pipeline per anode:**
```
FrameFileSource(dnnsp)
  → FrameFanout (rename dnnsp→gauss<N> and wiener<N>)
  → ChannelSelector(5638*N .. 5638*(N+1)-1)   ← defensive per-anode filter
  → img.per_anode(anode, 'multi-3view')
    ├─ port 0 → ClusterFileSink(icluster-apa<N>-active.npz)
    └─ port 1 → ClusterFileSink(icluster-apa<N>-masked.npz)
```

---

### `wct-clustering.jsonnet`

Runs per-APA and all-APA blob clustering using `MultiAlgBlobClustering`.

**Imports:** `pgrapher/experiment/sbnd/simparams.jsonnet` (with `lar` block
overlaid by TLAs), `clus.jsonnet`

**TLAs:** see `run_clus_evt.sh` table above.

**Pipeline:**
```
ClusterFileSource(icluster-apa<N>-active.npz)  ─┐
ClusterFileSource(icluster-apa<N>-masked.npz)  ─┤ clus.per_apa(anode<N>)
                                                  │   (PointTreeBuilding → MABC → per-APA zip)
                                                  ▼
                                          PointTreeMerging
                                                  │
                                          clus.all_apa(anodes)
                                                  │   (MABC → all-APA zip)
```

---

## Jsonnet helpers

### `clus.jsonnet`

Thin re-export of the canonical in-tree module
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, which defines the per-face,
per-APA, and all-APA clustering subgraphs.  Imported by `wct-clustering.jsonnet`.

**Exposes:** `per_face(anode, face, dump)`, `per_apa(anode, dump)`,
`all_apa(anodes, dump)`, `detector_volumes(anodes, face)`

**Imports:** `pgrapher/common/clus.jsonnet` (provides clustering algorithms:
`pointed`, `live_dead`, `extend`, `regular`, `parallel_prolong`, `close`,
`extend_loop`, `separate`, `connect1`, `switch_scope`, `neutrino`, `isolated`)

Key locals (in the canonical `cfg/pgrapher/experiment/sbnd/clus.jsonnet`):
`time_offset = -205 us`, `drift_speed = 1.563 mm/us`.
See [geometry-and-timing.md](geometry-and-timing.md).

### `magnify-sinks.jsonnet`

Builds per-anode `MagnifySink` pipeline nodes. Imported by
`wct-sp-to-magnify.jsonnet`. Returns `{ decon_pipe: [pipe_anode0, pipe_anode1] }`.
No `pgrapher/experiment/sbnd/` imports.

---

## Python helpers

### `wct-img-2-bee.py`

Called by `run_bee_img_evt.sh`. Constructs and executes `wirecell-img bee-blobs`
for each anode, then zips the output JSON files.

```
Usage: python wct-img-2-bee.py <run> <subrun> <event> <idx0>:<path0> [<idx1>:<path1> ...]
  idx:  anode index (0 = APA0 at x=-201.45 cm, 1 = APA1 at x=+201.45 cm)
  path: path to icluster-apa<N>-active.npz
```

Geometry arguments passed to `wirecell-img bee-blobs`:

| APA | `--x0` | `--speed` | `--t0` |
|---|---|---|---|
| 0 (x=-201.45 cm) | `-201.45*cm` | `-1.563*mm/us` | `200*us` |
| 1 (x=+201.45 cm) | `201.45*cm` | `+1.563*mm/us` | `205*us` |

Note `--t0 "205*us"` is the **positive** value even though `clus.jsonnet`
defines `time_offset = -205*us`. See [geometry-and-timing.md](geometry-and-timing.md).

**Output:** `data/0/0-apa<N>.json` (one per anode), then `upload.zip`. Used only
by the single-event path of `run_bee_img_evt.sh`; the `all`-mode path calls
`wirecell-img bee-blobs` directly to achieve correct per-event filename prefixes.

### `merge_sel_archives.py`

Called by `run_select_evt.sh` after Woodpecker selection. Loads the original
combined `sp-frames.tar.bz2`, overwrites rows for masked-anode channels with
the Woodpecker-masked values, and writes a new combined archive.

```
Usage: python3 merge_sel_archives.py <orig_archive> <out_archive> <evt_id> <masked1> [masked2 ...]
```

Channels for anodes that were not selected keep their original unmasked values.

### `pmt_nonlinearity_curve.py`

Standalone reproduction of the sbndcode PMT non-linearity (saturation) that maps
`NPE_true → NPE_observed/reco`. Reproduces `PMTNonLinearityTF1::NObservedPE` exactly
(TF1 `x/sqrt(1+(x/p0)^p1)`, 5-sample `PreTime+1` PE-accumulation window, per-bin scaling,
hard cap); no LArSoft needed. The non-linearity is applied **on the waveform**, so the tool
also does an explicit waveform round-trip (`roundtrip_reco_pe`: build `Σ observed·SER`, ADC
clip, integrate back ÷SPEArea) confirming `NPE_reco = Σ observed` — reco is linear, the ADC
clip never engages. Plots **envelope** (single burst, worst-case `Eval(N)`) vs **realized**
(scintillation, near-linear, physical). See `match/docs/sbnd-opdetsim-chain.md` for the trace.

```
# single-channel: NPE_true vs NPE_obs for single-burst + scintillation, plus inverse
python3 pmt_nonlinearity_curve.py                  # -> pmt_nonlin_out/{png,csv}

# all-PMT overlay (one curve per channel), real params from the conditions DB:
python3 pmt_nonlinearity_curve.py --all-pmt --params-csv perchan.csv   # -> pics/pmt_nonlinearity_allpmt.png
```

Per-channel `(PESat, Alpha)` are in the remote conditions DB (table `pds_calibration`,
tag `v3r1`); export to a CSV `opch,pesat,alpha[,range_hi]`. The real v3r1 values are
checked in as `pmt_nonlin_params_v3r1.csv` (120 PMTs: 104 with a saturation curve,
16 with `pesat=alpha=0` → nonlinearity off / linear), and `pics/pmt_nonlinearity_allpmt.png`
is generated from it — the **realized (scintillation)** per-PMT curve (the physical
expectation for real flashes; near-linear, ~8–14% attenuation by NPE≈5000), not the
worst-case envelope. With no `--params-csv` the all-PMT plot falls back to a
**clearly-labelled illustrative synthetic spread**.

`--emit-qlmatching` fits each PMT's realized curve (to NPE_true=10⁵) to a monotone
log-quadratic capped power law `observed = knee·exp(β·L+γ·L²)`, `L=ln(x/knee)` (identity
below `knee`≈700), and writes the per-OpDet `(β, γ)` arrays + knee as a jsonnet param file
for QLMatching, plus a fit-vs-MC validation plot (`pics/pmt_nonlinearity_fit.png`,
residual ≤2% over the data regime):

```
python3 pmt_nonlinearity_curve.py --emit-qlmatching --params-csv pmt_nonlin_params_v3r1.csv \
        --params-out ../../toolkit/cfg/pgrapher/experiment/sbnd/pmt_nonlinearity_params.jsonnet
```

### `ql_nonlin_compare.py`

Compares QLMatching predicted-vs-measured PE with the PMT non-linearity OFF vs ON, from two
`mabc.zip` BEE archives per sample (`run_clust_QL_evt.sh` with `PMT_NL=false` / default on).
Plots median pred/meas vs predicted-PE brightness (`pics/ql_pmt_nonlin_compare.png`). Finding:
**MC** shows a mild saturation trend the correction flattens; **data** sees more light than the
reconstructed charge explains (a charge/light effect, not PMT saturation), so the correction
does not help there. As of 2026-06-04 the correction is **ON by default for SBND** (canonical
`qlmatching.jsonnet`; `PMT_NL=false` / `pmt_nl=false` recovers the OFF baseline).

```
python3 ql_nonlin_compare.py --mc-off mc_off.zip --mc-on mc_on.zip \
                             --data-off data_off.zip --data-on data_on.zip
```

### `ql_pe_error.py`

Measures the per-PMT light-error fraction `a` from the hand-scanned matches (the matcher assumes
30% ⇒ `a=0.09`), modelling `E[(pred−meas)²] = meas + a·pred²`. Reads the per-event calib dumps
(NL-on from `work/ql_nl_study/`, NL-off from `work/ql_evt<ID>/`) and the hand-scan selections
(`work/ql_labels/<mode>/.scan_state-evt*.json`); aggregates **per flash** (pred summed over the
selected clusters on a flash, meas once), drops `window_truncated`/`close_to_PMT` flashes, keeps
PMTs (`opdet type==1`) in the flash's TPC. Writes 4-panel figures
`pics/ql_pe_error_<mode>{,_nloff,_consistent}.png` (pred-vs-meas; local `a` vs pred; Y vs pred²;
pull). Full writeup + findings in `docs/pe-error-study.md`. Default runs both modes:

```
python3 ql_pe_error.py            # data + mc
python3 ql_pe_error.py mc         # one mode
```

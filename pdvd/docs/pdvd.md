# ProtoDUNE-VD Imaging and Clustering (`pdvd/`)

## Provenance

Configs and scripts imported from
[HaiwangYu/wcp-porting-img `xn_vd_debug` branch](https://github.com/HaiwangYu/wcp-porting-img/tree/xn_vd_debug/pdvd).
Original author: Xin Ning (`xning`).

## Directory layout

```
pdvd/
├── docs/                    ← this documentation
├── data/                    ← Bee JSON staging area (written by wct-img-2-bee.py)
├── work/                    ← per-event scratch dirs (created by helper scripts)
│   └── <run>_<evt>/         ← imaging + clustering outputs for one event
├── input_data               → /nfs/data/1/xning/wirecell-working/data/
│                               symlink to Xin Ning's sample data tree
├── upload-to-bee.sh         → ../upload-to-bee.sh
│
├── wct-img-all.jsonnet      ← top-level imaging driver
│                              (imaging library lives in toolkit cfg:
│                               pgrapher/experiment/protodunevd/img.jsonnet)
├── wct-clustering.jsonnet   ← top-level clustering driver
│                              (clustering library lives in toolkit cfg:
│                               pgrapher/experiment/protodunevd/clus.jsonnet)
├── wcls-nf-sp-out.jsonnet   ← ART/LArSoft NF+SP → frame tarballs (upstream step;
│                              = cfg copy + raw-frame tap, see file header)
│
├── _runlib.sh               ← shared helper library sourced by all run_*.sh scripts
├── run_img_evt.sh           ← imaging only
├── run_clus_evt.sh          ← clustering only
├── run_nf_sp_evt.sh         ← standalone NF+SP (no LArSoft)
├── run_sp_to_magnify_evt.sh ← SP frames → per-anode Magnify ROOT files
├── run_select_evt.sh        ← interactive Woodpecker region-of-interest crop
├── run_bee_img_evt.sh       ← Bee conversion + upload, from IMAGING output (no clustering)
├── run_bee_combined_evt.sh  ← combined Bee link: img-global + clustering-global + op + dead, all from mabc-all-apa.zip
├── run_img.sh               ← original manual recipe (commented examples)
├── wct-img-2-bee.py         ← convert cluster tarballs → Bee JSON
├── wct-img-2-bee-only.py    ← single-anode debug variant
├── plot_frames.py           ← visualise SP frame archives as PNG
├── select_frames.py         ← interactive crop of frame archives (Qt UI)
│
├── old/                     ← retired configs/scripts kept for reference
│   ├── clus-new.jsonnet     (older PDHD-derived clustering variant)
│   ├── wct-raw-to-magnify.jsonnet, wct-sim-check-track.jsonnet
│   └── run_evt.pl, unzip.pl, build_*.sh (superseded by run_*.sh family)
│
└── protodune-sp-frames-anode{0..7}.tar.bz2  ← SAMPLE INPUT (see Q1 below)
```

## Input data under `input_data/`

`input_data` symlinks to `/nfs/data/1/xning/wirecell-working/data/`.
Run/event naming is **not uniform**:

| Run dir       | Event subdir(s)          | Notes                                      |
|---------------|--------------------------|--------------------------------------------|
| `run039324/`  | `evt1/`                  | Has SP frames + cluster tarballs           |
| `run040475/`  | `evt_0/` (empty), `evt_1/` | Underscore naming; `evt_0` has no data   |
| `run41189/`   | *(none — flat layout)*   | No leading zeros; data files at run root  |

Each event directory (or run root for flat layout) contains:

- `protodune-sp-frames-anode{0..7}.tar.bz2` — per-anode SP frames (imaging input)
- `protodune-sp-frames-raw-anode{0..7}.tar.bz2` — raw frames before SP (~47 MB each)
- `clusters-apa-anode{0..7}-ms-{active,masked}.tar.gz` — pre-computed imaging output
- `woodpecker_img_clus/` — sometimes a second copy of cluster tarballs from a later run

The helper scripts accept the run number with or without leading zeros and try all
candidate directory names automatically.

---

## Q1. Are the `protodune-sp-frames-anode*.tar.bz2` files here input or output?

They are **input** to the imaging step — staged sample data.

They are per-anode signal-processing (SP) output frames produced upstream by
`wcls-nf-sp-out.jsonnet` running inside LArSoft/ART.  The eight files staged in
this directory come from `input_data/run039324/evt1/` so that the imaging step can
be run standalone without LArSoft.

---

## Q2. Imaging

**Purpose:** convert per-anode SP frames → per-anode imaging cluster tarballs.

**Config:** `wct-img-all.jsonnet` (imports `pgrapher/experiment/protodunevd/img.jsonnet` from the toolkit cfg)

**Input:** `<input_prefix>-anode{0..7}.tar.bz2`

**Output:** `<output_dir>/clusters-apa-anode{N}-ms-active.tar.gz`
          + `<output_dir>/clusters-apa-anode{N}-ms-masked.tar.gz` per anode

**Command:**
```sh
wire-cell -l stdout -L debug \
  --tla-str input_prefix=protodune-sp-frames \
  --tla-code 'anode_indices=[0,1,2,3,4,5,6,7]' \
  --tla-str output_dir=. \
  -c wct-img-all.jsonnet
```

TLA reference:

| TLA | Type | Default | Meaning |
|-----|------|---------|---------|
| `input_prefix` | `--tla-str` | `protodune-sp-frames` | Path prefix before `-anodeN.tar.bz2` |
| `anode_indices` | `--tla-code` | `[0..7]` | Anodes to process |
| `output_dir` | `--tla-str` | `""` (current dir) | Directory for output cluster tarballs |

**Required `WIRECELL_PATH`** (set automatically by the helper scripts):
```sh
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}
```

---

## Q3. Clustering

> For a deep-dive on graph topology, RSE propagation, dead-channel handling, and per-APA/face selection see **[clus-workflow.md](clus-workflow.md)**.
> Per-pass scope capabilities and the T0/containment knobs: **[clustering-scope.md](clustering-scope.md)**.
> Feasibility of replacing the two-faces-per-anode description with a single SBND-style face: **[single-face-anode-feasibility.md](single-face-anode-feasibility.md)**.

**Purpose:** convert per-anode imaging cluster tarballs → multi-algorithm blob clustering → Bee zips.

**Config:** `wct-clustering.jsonnet` (imports `pgrapher/experiment/protodunevd/clus.jsonnet` from the toolkit cfg)

**Input:** `<input>/clusters-apa-anode{N}-ms-active.tar.gz`
         + `<input>/clusters-apa-anode{N}-ms-masked.tar.gz`

**Output:** `<output_dir>/mabc-anode{N}.zip`, `<output_dir>/mabc-all-apa.zip`
           (plus `trash-*.tar.gz` debug sinks)

**Command:**
```sh
wire-cell -l stdout -L debug \
  --tla-str input=. \
  --tla-code 'anode_indices=[0,1,2,3,4,5,6,7]' \
  --tla-str output_dir=. \
  -c wct-clustering.jsonnet
```

TLA reference:

| TLA | Type | Default | Meaning |
|-----|------|---------|---------|
| `input` | `--tla-str` | `"."` | Directory containing the 16 cluster tarballs |
| `anode_indices` | `--tla-code` | `[0..7]` | Anodes to process |
| `output_dir` | `--tla-str` | `""` (current dir) | Directory for Bee zips |

---

## Q4. Single config doing both imaging and clustering?

No — the two steps are separate `wire-cell` invocations.  Run
`./run_img_evt.sh` then `./run_clus_evt.sh` (the old `perl run_evt.pl`
dispatcher is retired to `old/`).

---

## Helper scripts

All scripts are run from the `pdvd/` directory.  Outputs go to
`work/<run>_<evt>/` (6-digit zero-padded run, e.g. `work/039324_1/`).

### Common conventions

Every `run_*.sh` script shares three ergonomic features provided by `_runlib.sh`:

**No-arg listing** — run any script with no arguments to list available runs:
```sh
./run_img_evt.sh
# Available runs under .../pdvd/input_data:
#   run039324     events: 0 1 2 ...
#   run040475     events: 0 1
#   run41189      (flat layout — specify event number explicitly)
```

**`EVT=all` parallel mode** — pass `all` as the event number to process every
discovered event for that run in parallel:
```sh
./run_img_evt.sh 040475 all
./run_clus_evt.sh 040475 all
```
Events are discovered from `input_data/run<N>/evt_*` subdirectories and from
existing `work/<RUN_PADDED>_<EVT>/` directories.  Jobs run concurrently up to
`$(nproc)` (override with `PDVD_MAX_JOBS=N`).  Each event's output is written
to its own log at `work/.batch_<script>_<run>_<evt>.log`.  A summary is printed
at the end showing ok / failed counts and the list of failed event ids.

**Skip-on-missing** — when running in `all` mode, an event whose required input
files are absent (e.g., no SP frames, no cluster tarballs, or a `-s sel_tag`
directory that doesn't exist) is skipped with a one-line note rather than
aborting the whole batch.  In single-event mode, the same conditions still exit
non-zero (exit code 2 for a skip, 1 for a hard failure).

**Concurrency cap** — controlled by the `PDVD_MAX_JOBS` environment variable
(default: `nproc`):
```sh
PDVD_MAX_JOBS=4 ./run_img_evt.sh 040475 all   # cap at 4 simultaneous jobs
```

### `./run_nf_sp_evt.sh [-a anode] <run> <evt|all>`

Runs standalone NF+SP signal processing (no LArSoft).  Reads
`protodune-orig-frames-anode{N}.tar.bz2` from `input_data/`, writes
`protodune-sp-frames{,-raw}-anode{N}.tar.bz2` to `work/<run>_<evt>/`.
Use `-a N` (0–7) to process a single anode; without it all 8 are processed.

### `./run_sp_to_magnify_evt.sh [-I] [-s sel_tag] <run> <evt|all> [subrun]`

Converts SP frame archives to per-anode Magnify ROOT files for waveform
inspection.  Reads from `work/<run>_<evt>/` (preferred) or `input_data/`.
Outputs `magnify-run<R>-evt<E>-anode<N>.root` per anode.  Orig frames from
`input_data/` are included when available (adds `hu/hv/hw_orig<N>` histograms).
`-I` forces reading from `input_data/` even when a work-dir copy exists.
`-s sel_tag` reads from a Woodpecker selection dir (see `run_select_evt.sh`).

### `./run_img_evt.sh [-I] [-a anode] [-s sel_tag] <run> <evt|all>`

Reads SP frames from `input_data/` event dir (or work dir if present), writes
cluster tarballs to `work/<run>_<evt>/`.  Use `-a N` (0–7) to process only
anode `N`; without it all 8 are processed.  `-I` forces reading from
`input_data/`.  `-s sel_tag` reads from a Woodpecker selection dir.

### `./run_clus_evt.sh [-a anode] [-s sel_tag] <run> <evt|all> [subrun]`

Reads cluster tarballs from `work/<run>_<evt>/` (after imaging) or falls back
to the pre-computed tarballs in `input_data/` event dir.  Writes Bee zips to
`work/<run>_<evt>/`.  Use `-a N` (0–7) to restrict to a single anode; without
it all 8 are processed.  The Art event number is parsed automatically from the
cluster-tarball filenames and embedded in the Bee output.  The optional
`[subrun]` arg sets the subRun number (default `0`).

### `./run_bee_img_evt.sh [-a anode] [-s sel_tag] <run> <evt|all> [subrun]`

Converts imaging cluster tarballs (`clusters-apa-anode{N}-ms-active.tar.gz`,
produced by `run_img_evt.sh`) directly to Bee JSON via `wirecell-img bee-blobs`
— does **not** run clustering/MABC.  Reads from `work/<run>_<evt>/` if imaging
has been run, otherwise from `input_data/` event dir.

**Single-event mode** (`EVT` is a number): produces `upload_<run>_<evt>.zip`
and prints the Bee URL.  Use `-a N` (0–7) to convert only that anode.

**`EVT=all` mode**: processes every discovered event for the run, combining all
events into a single zip (`upload-batch-run<RUN_PADDED>.zip`) and uploading
once.  Per-event Bee data is staged in `data/0/`, `data/1/`, … so each event
occupies its own Bee event-list slot.  `wct-img-2-bee.py` is bypassed for this
path (it hardcodes `data/0/`); `wirecell-img bee-blobs` is called directly with
the correct subdirectory per event.  The drift-speed / x0 geometry constants
must stay in sync between `wct-img-2-bee.py` and the shell `all`-path in
`run_bee_img_evt.sh`.

By default the eight anodes are merged by drift side into **two** Bee instances
per event — `<i>-imaging-group0123.json` (anodes 0–3, bottom drift, x0=−341.5cm)
and `<i>-imaging-group4567.json` (anodes 4–7, top drift, x0=+341.5cm) — so each
event shows two images instead of eight.  Each drift side shares geometry, so a
single `bee-blobs` call per group renders correctly.  Set `PDVD_BEE_GROUP=0` to
fall back to the legacy one-instance-per-anode output (`<i>-apa<N>.json`).
Note `wirecell-img bee-blobs` has no dead-area code path, so this imaging-only
link never carries a dead layer (the dead layer comes from clustering).

### `./run_select_evt.sh [-a anode] <run> <evt> <sel_tag>`

Opens an interactive Woodpecker browser GUI to select a tick/channel region of
interest.  Produces masked SP frame archives in
`work/<run>_<evt>_<sel_tag>/input/`.  Pass `-s <sel_tag>` to the downstream
pipeline scripts to use the selection.  This script does **not** support
`EVT=all` (GUI is interactive).

---

## Bee upload

### Path A — imaging → Bee directly (used by `run_bee_img_evt.sh`)

Reads the per-anode imaging cluster tarballs
(`clusters-apa-anode{N}-ms-active.tar.gz`) and calls
`wirecell-img bee-blobs -g protodunevd`, by default merging the four anodes of
each drift side into one instance (`data/<i>/<i>-imaging-group0123.json` and
`-imaging-group4567.json`; `PDVD_BEE_GROUP=0` restores per-anode `0-apa{N}.json`).
No clustering is performed — the Bee display shows the raw imaging blobs.
The drift speed and x-offset sign differ by TPC half:

| Anodes 0–3 (bottom drift) | Anodes 4–7 (top drift) |
|---|---|
| `--speed "-1.57*mm/us"` | `--speed "1.57*mm/us"` |
| `--x0 "-341.5*cm"` | `--x0 "341.5*cm"` |

After conversion: `zip -r upload data`, then `./upload-to-bee.sh upload.zip`.

### Path B — clustering → Bee via MABC's built-in writer (used by `run_clus_evt.sh`)

`wct-clustering.jsonnet` (with the cfg library
`pgrapher/experiment/protodunevd/clus.jsonnet`) runs MABC on the imaging
tarballs and produces `mabc-anode{N}.zip` / `mabc-all-apa.zip` via
`MultiAlgBlobClustering`'s built-in Bee writer — the Bee display here shows
**clustered** blobs, in contrast to Path A's raw imaging blobs.  To upload:
```sh
./old/unzip.pl    # expands mabc*.zip into data/ (retired helper, still works)
./zip-upload.sh   # rezips data/ → upload.zip, calls ../upload-to-bee.sh
```

The all-anode `mabc-all-apa.zip` carries (2026-07 update, PDHD/SBND parity):
`img-global` (the raw imaged charge, dumped **pre-pipeline** via the
`name:"img"` bee points set — its cluster ids are the same enumeration the
`op` dump writes into each flash's `op_cluster_ids`, so the flash↔cluster
pairing is checked against this instance), `clustering-global` (the
**all-anode** clustering, end dump, re-enumerated ids), `op` (when Q/L
matching ran), and `channel-deadarea-group0123` / `-group4567` (dead area,
`dead_apa_groups`).  The pre-pipeline hook hosts only one set, so `img-global`
replaced the former per-drift-side `clustering-group0123/4567` dump.  See
[../../clus/docs/bee_output.md](../../clus/docs/bee_output.md) for the grouping
mechanism (routing is per-anode via `wpid.apa()`, so each anode's two faces fold
into its drift-side group automatically).

### Path C — combined link (imaging + clustering + op + dead)

```sh
./run_bee_combined_evt.sh [-e evt,evt,...] <run>   # run run_clus_evt.sh <run> all first
```

Produces one `upload-combined-run<RUN_PADDED>.zip` / Bee link.  Every instance
is taken straight from each event's `mabc-all-apa.zip` (2026-07 update — no
separate `bee-blobs` imaging pass, PDHD parity):

| Instances | Stage |
|---|---|
| `img-global` | raw imaged charge (MABC `img` pre-pipeline dump) |
| `clustering-global` | after **all-anode** clustering (MABC end dump) |
| `op` | optical flashes, measured + Q/L-predicted PE (when matching ran) |
| `channel-deadarea-group0123` / `-group4567` | dead area (v2 wrapper) |

**Checking a matched pair in Bee:** step through the `op` instance flashes and
read each flash's matched cluster id(s), then look the ids up in the
`img-global` charge layer — those two are dumped at the same pre-pipeline
point and share the cluster-id enumeration.  `clustering-global` ids are
re-enumerated post-pipeline and do **not** correspond, and the former
`bee-blobs` `imaging-group0123/4567` instances carried their own unrelated
numbering (the pre-2026-07 combined link mixed these, which made the
flash↔cluster pairing look mismatched).

`-e 0,1,...` restricts the link to a subset of event indices (e.g. the ten
hand-scan events of run 039252).

### Upload mechanism

`upload-to-bee.sh` authenticates against `https://www.phy.bnl.gov/twister/bee`
and POSTs the zip.  On success it prints:
```
https://www.phy.bnl.gov/twister/bee/set/<UUID>/event/list/
```

---

## Known gotchas

- **RSE in clustering Bee output**: `run_clus_evt.sh` automatically parses
  the Art event number from the cluster-tarball filename and passes
  `run`/`subrun`/`event` as TLAs to `wct-clustering.jsonnet`, which threads
  them into all three `MultiAlgBlobClustering` instances.  The optional third
  positional arg sets subrun: `./run_clus_evt.sh 039324 1 7` → subRun=7;
  default is 0.

- **`run41189` vs `run039324` naming**: `run41189` has no leading zeros and stores
  data files directly at the run root with no `evt*/` subdir.  The helper scripts
  handle this automatically.

- **`run040475/evt_0/` is empty**: that directory exists but contains no data.
  Use `evt_1` instead.

- **`old/clus-new.jsonnet`**: 4-anode PDHD-style variant with hard-coded
  `detector: "protodunehd"` / `bee_detector: "sbnd"`.  Not used by current entry
  points; retired to `old/` for reference.

- **`wct-img-2-bee.py` clears `data/` on each run**: in single-event mode,
  running `run_bee_img_evt.sh` sequentially for multiple events would overwrite
  `data/0/` each time; each run produces a persistent `upload_<run>_<evt>.zip`.
  Use `EVT=all` mode to combine multiple events into a single zip automatically
  (it stages events in `data/0/`, `data/1/`, … without overwriting).

- **`run_clus_evt.sh -a N` shrinks the clustering topology to a single APA**:
  detector volumes, `PointTreeMerging` multiplicity, and MABC's anode list all
  become single-APA.  Cross-APA clustering steps are a no-op in this mode.
  Use this for debugging a single anode, not for production.

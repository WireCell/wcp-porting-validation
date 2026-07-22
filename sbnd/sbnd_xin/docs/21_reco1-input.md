# reco1 direct input — running the standalone chain from an SBND art file

The `input_files_reco1/` reco1 art/LArSoft ROOT files can now feed the whole
standalone chain (imaging → clustering → Q/L matching → PR) **without
LArSoft**: two new toolkit components read the art file with bare ROOT and a
converter job materializes the standard sample-dir layout, after which every
existing runner works unchanged via `SBND_INPUT_DIR`.

## Repro block

```sh
cd sbnd_xin
./run_reco1_dump.sh -t 2025fall-48evt          # one-shot, ~1 min for 48 events
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt
./run_img_evt.sh data                          # list the 48 events
./run_img_evt.sh data all                      # imaging (checkpoint npz), SBND_MAX_JOBS caps
./run_ql_evt.sh  data <idx> [-calib ...]       # iterate on Q/L from the npz checkpoint
```

`run_ql_evt.sh` re-reads `work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz`
without re-running imaging — the usual QL-debug loop applies to reco1 events
exactly as to the older samples.

## What's new (and where)

| piece | where | notes |
|---|---|---|
| `SBNDReco1FrameSource` / `SBNDReco1OpFlashSource` | toolkit `root/` | bare-ROOT art-file readers; see `toolkit/root/docs/sbnd-reco1-source.md` for internals, gotchas, verification |
| `wct-reco1-dump.jsonnet` | `sbnd_xin/` | reco1 → `frames-dnn.tar.bz2` + `opflash_apa{0,1}.tar.gz` (all events streamed in one process; `entry` TLA for one event) |
| `run_reco1_dump.sh` | `sbnd_xin/` | runner; writes `input_files_reco1/extracted-<tag>/` (refuses to overwrite an existing tag); `-caf product` for `*_frameshift.root` inputs |
| `SBND_WORK_ROOT` | `_runlib.sh` (+ img/ql runners) | work-tree root override (default `work/`); land a reprocessing campaign in a fresh tree instead of overwriting an old one |
| `merge_mabc_bee.py` | `sbnd_xin/` | re-keys N per-event `ql_evt<ID>/mabc-all-apa.zip` Bee dumps into one multi-event upload zip (`data/<i>/<i>-*.json`) |
| Bee run/subrun numbers | `run_ql_evt.sh` | reads `run`/`subrun` from the staged opflash tensor-set metadata and passes them as the `run`/`subrun` TLAs, so every Bee layer shows the full `(runNo, subRunNo, eventNo)` triplet; samples without those metadata keys (yuhw's dumps) keep `run=0 subrun=0` unchanged |

The extraction is the toolkit-only counterpart of yuhw's LArSoft dumps
(`input_files/wcls-frame-dump.*`, `wcls-flash-dump.*`); output formats are
pinned to his conventions (frame scale ×50, nticks 3427, `chanmask_bad`
triplets, opflash `[nflash, 313]` col-0 ns).  yuhw's own chain ran
`reco1 → FrameShift producer → wcls dumps`; ours ports the FrameShift
derivation into the opflash source (below), so the plain reco1 file
(`*_eventidfiltered.root`, pre-FrameShift) suffices.

## The development sample

`data_filtered_decoded_reco1-fe6033f3-…_eventidfiltered.root`: 48 real-data
**neutrino-candidate** events across 12 runs 18253..18409 (30 of them run
18255), 2025 fall production, BNBLight stream (all 48 classify as beam stream
in the FrameShift port; `fTimingType = 0` in the FRAMESHIFT product).
Contents relevant to us: `sptpc2d` post-SP DNN wires + badmasks +
wienersummary, `opflashtpc{0,1}` + `ophitpmt` light reco, PTB/TDC/DAQ-header
timing.  **No** `raw::RawDigits` (NF/SP cannot be re-run) and **no** MC truth.

`…_eventidfiltered_frameshift.root`: the same 48 events after the sbndcode
`FrameShift` producer (process `FRAMESHIFT`), adding
`sbnd::timing::FrameShiftInfo` (with the authoritative per-event
`fFrameApplyAtCaf`) and `TimingInfo`.  This is the preferred input:
extract it with `-caf product`.

## Flash-time correction (`frame_apply_at_caf`)

The extracted opflash tensor-sets carry `frame_apply_at_caf` (ns).  Two
sources, chosen by `run_reco1_dump.sh -caf`:

- **`-caf product`** (preferred, needs the `*_frameshift.root` input): the
  authoritative `FrameShiftInfo::fFrameApplyAtCaf` written by the sbndcode
  FrameShift producer.  The FRAMESHIFT product answered the previously-open
  reduction question: on all 48 events it equals **`fFrameTdcRwm`** — the
  decoded-frame → SPEC-TDC RWM (per-spill beam arrival) shift, 247–2757 ns,
  *not* 256-ns quantized.
- **`-caf auto`** (pre-FrameShift files): the ported derivation
  `frame_etrig − frame_default` (0–2560 ns, 256-ns quanta).  Now known to
  sit **43–482 ns (mean 262) below** the product value — right scale, wrong
  formula; the exact decoded-frame reference needs PMT-decoder products the
  extraction does not read, so `auto` cannot be made exact.  Keep it as the
  fallback only.

Validation on the product offsets (all 48 events are neutrino candidates):

- raw in-time flash (min |t|, both APAs): median **−0.715 µs**, matching the
  −0.71 µs signature ([10_flash-coincidence.md](10_flash-coincidence.md));
- corrected in-time flash: median **+1.28 µs**, **45/48 inside the
  +0.3–1.9 µs beam window** (vs 42/48 with `auto`); 214469/111412 sit at
  1.94/2.04 µs just outside, 131357's min-|t| flash (−0.40 µs) is likely a
  cosmic.

## Full-chain reprocessing with product offsets (2026-07-21)

Tag `extracted-2025fall-48evt-fsprod` (from the `*_frameshift.root` input,
`-caf product`), landed in a fresh work tree so the earlier campaign's
`work/` stays intact:

```sh
cd sbnd_xin
./run_reco1_dump.sh -caf product -t 2025fall-48evt-fsprod \
    input_files_reco1/*_eventidfiltered_frameshift.root
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod
export SBND_WORK_ROOT=$PWD/work-fsprod
SBND_MAX_JOBS=8 ./run_img_evt.sh data all      # 48/48 ok
SBND_MAX_JOBS=6 ./run_ql_evt.sh  data all      # 48/48 ok
EVTS=$(tar tjf $SBND_INPUT_DIR/frames-dnn.tar.bz2 \
       | sed -n 's/^frame_dnnsp_\([0-9]*\)\.npy$/\1/p' | awk '!seen[$0]++')
python3 merge_mabc_bee.py -w work-fsprod -o upload-fsprod-48evt.zip $EVTS
./upload-to-bee.sh upload-fsprod-48evt.zip
```

- Extraction `frame_apply_at_caf` == product `fFrameApplyAtCaf` on 48/48
  events (both APAs); QL logs show the exact per-event value applied
  (e.g. evt 256587: 2217 ns, evt 10550: 255 ns).
- Cross-file frame consistency: the fsprod `frames-dnn.tar.bz2` member-hashes
  to the same `46ff819f…` (240 members) as the original-file extraction —
  the FrameShift producer touches nothing but the timing products, so the
  reprocessing differs from the first campaign **only** in the flash-time
  offsets.
- **Bee (48 events, layers `clustering-global` + `img-global` + per-APA
  dead area + op) — current set, labelled `run-subrun-event`:**
  <https://www.phy.bnl.gov/twister/bee/set/0fbcecd1-23ff-4103-8a58-cd9a23551d80/event/list/>
  (Bee event index 0..47 in frames-archive order = the `run_*_evt.sh`
  idx−1; the set page lists 48 events.)  Supersedes the first upload
  `c3254159-…`, whose events carried only the event number (`0-0-<evt>`).
- Existing-chain A/B re-gate after the toolkit additions (FrameShiftInfo/
  TimingInfo mirrors + `caf_offset_mode=product`): data evt 686 joint QL
  rerun under `setarch -R` into fresh `work-abfs1/`, `mabc-all-apa.zip`
  member-hash `4b006453…` (5 members) — identical to the recorded
  baseline.  PASS.

### Bee run/subrun numbers (2026-07-21, second pass)

The first upload showed `0-0-<event>` because `run_ql_evt.sh` hardcoded
`run=0 subrun=0`.  The art run/subrun are already in each extracted opflash
tensor-set metadata (`{"run": 18253, "subrun": 1, "event": 172230, …}`), so the
runner now reads them from the staged per-event opflash and passes them as the
`run`/`subrun` TLAs; `clus.jsonnet` already forwards them into every
`MultiAlgBlobClustering` (`use_config_rse: true`), so all five Bee layers carry
the triplet.  The lookup is in the *runner*, not the cluster files — minimal and
default-inert: the older samples (`input_files/…`) have no `run`/`subrun` keys
in their opflash metadata, so they still get `run=0 subrun=0` and their Bee
output is unchanged.  No C++ change, no rebuild.

```sh
cd sbnd_xin
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod
export SBND_WORK_ROOT=$PWD/work-fsprod-rse    # evt<ID> symlinked to work-fsprod
SBND_MAX_JOBS=6 ./run_ql_evt.sh data all      # 48/48 ok; logs "[evt <ID>] rse=(run, subrun, evt)"
python3 merge_mabc_bee.py -w work-fsprod-rse -o upload-fsprod-48evt-rse.zip $EVTS
./upload-to-bee.sh upload-fsprod-48evt-rse.zip
```

Verified: 240/240 Bee layer JSONs have non-zero `runNo`, 12 distinct runs
(18253..18409); the Bee set page labels events `18253-1-172230` etc.

## Validation snapshot (2026-07-21)

- Extraction determinism: two independent 48-event extractions member-hash
  identical (`abtest/hash_archive.py`): `frames-dnn.tar.bz2` `46ff819f…`
  (240 members), `opflash_apa{0,1}.tar.gz` `554c6924…`/`307de3f6…`.
- Format identity vs `input-1file-data-v10_14_02_02`: same member scheme,
  shapes, dtypes; same first bad channel (546).
- End-to-end smoke: evt 256587 imaging (46 s) + joint QL (32 s) through the
  **unchanged** runners; QL applied `frame_apply_at_caf offset=2048 ns`,
  matched 15+11 flashes, produced `mabc-all-apa.zip`.
- Existing-chain A/B gate PASS (toolkit side): data evt 686 joint QL,
  baseline vs new `libWireCellRoot` under `setarch -R`, `mabc-all-apa.zip`
  member-hash identical (`4b006453…`).
- Full-sample batch (`SBND_MAX_JOBS=6`): imaging 48/48 ok →
  `work/evt<ID>/icluster-*.npz`; joint QL 48/48 ok →
  `work/ql_evt<ID>/mabc-all-apa.zip`.  Per-event `frame_apply_at_caf`
  applied in every run (0–2560 ns in 256 ns quanta); 22–41 in-scope matched
  clusters per event collapsing to ~10 flash-time groups.

## Caveats

- MC reco1 files need the `simtpc2d` product labels (`wire_product` etc. knobs
  on the sources; the dump jsonnet currently hardcodes data labels).
- The 48-event sample is data ⇒ use `data` mode everywhere (`reality=data`).
- Static QL PMT mask caveat applies as for any data sample
  (see memory: data PMT mask is run-dependent).

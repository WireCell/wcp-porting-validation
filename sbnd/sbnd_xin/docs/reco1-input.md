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
| `run_reco1_dump.sh` | `sbnd_xin/` | runner; writes `input_files_reco1/extracted-<tag>/` (refuses to overwrite an existing tag) |

The extraction is the toolkit-only counterpart of yuhw's LArSoft dumps
(`input_files/wcls-frame-dump.*`, `wcls-flash-dump.*`); output formats are
pinned to his conventions (frame scale ×50, nticks 3427, `chanmask_bad`
triplets, opflash `[nflash, 313]` col-0 ns).  yuhw's own chain ran
`reco1 → FrameShift producer → wcls dumps`; ours ports the FrameShift
derivation into the opflash source (below), so the plain reco1 file
(`*_eventidfiltered.root`, pre-FrameShift) suffices.

## The development sample

`data_filtered_decoded_reco1-fe6033f3-…_eventidfiltered.root`: 48 real-data
events, runs 18306 (1) / 18255 (47), 2025 fall production, BNBLight stream
(all 48 events classify as beam stream in the FrameShift port).  Contents
relevant to us: `sptpc2d` post-SP DNN wires + badmasks + wienersummary,
`opflashtpc{0,1}` + `ophitpmt` light reco, PTB/TDC/DAQ-header timing.  **No**
`raw::RawDigits` (NF/SP cannot be re-run) and **no** MC truth.

## Flash-time correction (`frame_apply_at_caf`)

The extracted opflash tensor-sets carry `frame_apply_at_caf` (ns) computed by
the ported sbndcode FrameShift derivation (`run_reco1_dump.sh` default
`-caf auto`; `-caf none` omits the key; `-caf override:<ns>` forces a value).
The reduction is `frame_etrig − frame_default` — fixed empirically, since the
sbnobj `FrameApplyAtCaf()` accessor is a local (non-public) modification:

- raw in-time flash (min |t|, both APAs): median **−0.715 µs** on 48/48
  events, matching the −0.71 µs signature of the older samples
  ([flash-coincidence.md](flash-coincidence.md));
- offsets 0–2560 ns, median 1536 ns (same scale as the lan-reco2 dumps'
  342–2110 ns);
- corrected in-time flash: median **+0.98 µs**, **42/48 inside the
  +0.3–1.9 µs beam window** (the documented validation signature; the
  remainder mirror lan-reco2's off-time tail).

The opposite sign pushes flashes out of the window.  **Pending confirmation
with the SBND timing authors / yuhw** ("does `FrameApplyAtCaf()` return
`frame_etrig − frame_default`?"); revisit `SBNDReco1OpFlashSource.cxx` if the
answer differs.

## Validation snapshot (2026-07-21)

- Extraction determinism: two independent 48-event extractions member-hash
  identical (`abtest/hash_archive.py`).
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

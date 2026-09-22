# doc pdhd/30 — the colleague-facing STM + Michel release: PDHD and PDVD, one ROOT file + Bee zip per event, with the matched flash and its raw / deconvolved waveforms

**Status (2026-09-22):** DELIVERED. `pdhd/stm_michel_release/` (61 PDHD events, runs 028084 +
029107) and `pdvd/stm_michel_release/` (120 PDVD events, runs 039252 / 039253 / 039349), each
with a per-event ROOT file (9 TTrees), the Bee zip, four documents, a reader library and six
example / validation scripts.  **No production output changed.**  The one toolkit change is
a default-OFF jsonnet knob (`frames_dir`) on the two all-PD light jobs, gated byte-identical
when off (toolkit `03f0a569`); the flashes of the rerun that dumped the waveforms reproduce
the production archives member-for-member on **181/181** events.

The owner's ask: a dedicated directory for colleagues who want to do the charge + light
energy reconstruction of Michel electrons and look at the stopping-muon side (range,
dQ/dx → dE/dx, MCS): (1) STM / Michel trajectory + dQ/dx in ROOT, (2) the 2-D charge
measurements, (3) the flash + the raw and deconvolved light waveforms, (4) the Bee zip,
(5) the muon and Michel energies; md files explaining the ROOT file, the Bee zip and the
light; python helpers; validation scripts for dQ/dx vs rr, the Michel charge energy and the
matched-flash waveforms.  Both detectors.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; T=/nfs/data/1/xqian/toolkit-dev/toolkit
export WIRECELL_PATH=$T/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data

# 1. the knob gate (compiled JSON, runners' TLA sets; artefacts in pdhd/docs/scan/d30/)
cd $T && wcsonnet -A snip_file=/x/snip.root -A fs_file=/x/fs.root -A output_dir=/x/out -S run=29107 -S event=1135 -S offset_us=249.728 \
    -o /tmp/x.json cfg/pgrapher/experiment/pdhd/wct-light-allpd-reco.jsonnet && md5sum /tmp/x.json      # 7cb3f9189005... = pre-knob
cat $I/pdhd/docs/scan/d30/cfg_knob_gate.txt

# 2. the light rerun with the frames (new suffix dirs; 61 + 120 events, cap 6)
cd $I/pdhd && PDHD_LIGHT_FRAMES=1 ./run_light_allpd_evt.sh -s _wf 29107 1135
cd $I/pdvd && PDVD_LIGHT_FRAMES=1 PDVD_FLASH_TAIL_MERGE=0 ./run_light_evt.sh -s _wf \
    -f input_data_light/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root 39252 298567
cat $I/pdhd/docs/scan/d30/light_gate.txt | grep '^##'          # hash_archive prod vs _wf + compiled-config parity, 181 events

# 3. the releases (~1 min/event on a loaded box) and the figures
cd $I/pdhd/stm_michel_release/scripts && python3 build_release.py --det pdhd     # -> ../events, ../index.csv, ../dqdx_ref.json, ../build_log.txt
cd $I/pdvd/stm_michel_release/scripts && python3 build_release.py --det pdvd
cd $I/pdhd/stm_michel_release && python3 scripts/list_candidates.py && python3 scripts/plot_dqdx_vs_rr.py && \
    python3 scripts/plot_michel_energy.py && python3 scripts/plot_muon_energy.py && \
    python3 scripts/plot_flash_waveforms.py --event 029107_1135 --cluster 30 && python3 scripts/plot_event_2d.py --event 029107_1135 --cluster 30 --zoom 40
```

## 1. What already existed, and what did not

The request read as "define a ROOT output for the STM + Michel tagger the way the neutrino PR
has one".  It turned out the tagger **already writes one**: `CheckSTM_Michel` hangs three
point clouds on each candidate cluster (`stm_michel`, `stm_michel_pts`, `stm_michel_2d`,
`clus/src/CheckSTM_Michel.cxx:2357-2596`) and `PdvdPrMagnifyTrackingVisitor::write_pc_tree`
(`root/src/PdvdPrMagnifyTrackingVisitor.cxx:293,358-365`) turns them into `T_stm_michel`
(199 branches, one row per candidate), `T_stm_michel_pts` and `T_stm_michel_2d` in the PR
job's `tracking-pr.root` — the same writer family as the neutrino PR output, and the file the
docs pdhd/14-29 and pdvd/51-116 censuses read.  The same file carries `T_rec_charge`,
`T_proj_data` and `T_cluster` (with `flash_id` = the row of the opflash tensor, verified on
evt 029107/19: rows 94 / 72 / 37 / 134 reproduce time and PE).  So on the charge side nothing
had to be defined; it had to be **joined, documented and made self-contained**.

What did not exist: the raw and deconvolved optical waveforms of the *production* light
jobs.  Both all-PD jobs wire only the opflash sink; the frames that exist on disk are side
products of older or validation jobs with different deconvolution settings (PDHD 11-13
channels of the June snippet-only job; PDVD one event from `wct-light-frames.jsonnet`
without spe_v2 / saturation repair).  Hence the knob.

Design decision (owner, 2026-09-22): **no new C++ writer.**  The release file is assembled
in Python (uproot) from the production products; the light job is a separate wire-cell
process, so a C++ join with the waveforms inside the PR job is not possible anyway, and
every production output stays byte-identical.  Waveforms: a window from the matched flash
to +10 µs (and 2 µs before), not the full frames.  Two directories, one per detector.

## 2. The toolkit knob and its gate

`cfg/pgrapher/experiment/pdhd/wct-light-allpd-reco.jsonnet` and
`cfg/pgrapher/experiment/protodunevd/wct-light-reco.jsonnet` gain the TLA `frames_dir=''`.
When set, a `FrameFanout` in front of each `OpHitFinder` feeds a `FrameFileSink` per branch
(`flash.waveform_sink`, tags raw / decon, plus `decon_roi` on the continuous-stream branches:
`OpRoi` re-tags every input trace tag and adds its own, `flash/src/OpRoi.cxx:194-201`):

| job | files | tags |
|---|---|---|
| PDHD all-PD | `<frames_dir>/light-frames-allpd-snip-wct.tar.bz2` (OpDets 0-119), `light-frames-allpd-fs-wct.tar.bz2` (120-159) | raw, decon / raw, decon, decon_roi |
| PDVD all-PD | `<frames_dir>/light-frames-{cath,mem,pmt}-wct.tar.bz2` | raw, decon, decon_roi / raw, decon / raw, decon |

The edge list is built in its original order with the decon→hit link of each branch
replaced in place: a first version that appended the conditional edges at the end compiled
to a different JSON (same graph, permuted edge list) and failed the gate.

| gate | result |
|---|---|
| knob off, PDHD runner TLA set, before vs after | **byte-identical** (md5 `7cb3f9189005`) |
| knob off, PDVD runner TLA set (veto off, flag + repair + coverage + spe_v2 + overflow + tail merge on) | **byte-identical** (`9038443a02db`) |
| knob off, PDVD bare | **byte-identical** (`c8db4806d252`) |
| knob on | 2 + 2 (PDHD) / 3 + 3 (PDVD) FrameFanout + FrameFileSink nodes, 13 / 18 edges (`pdhd/docs/scan/d30/{pdhd,pdvd}_on.json`) |

Runners (`pdhd/run_light_allpd_evt.sh`, `pdvd/run_light_evt.sh`): env `PDHD_LIGHT_FRAMES=1`
/ `PDVD_LIGHT_FRAMES=1` passes `-A frames_dir=$WORKDIR`; unset, the wcsonnet call is
unchanged.

## 3. The light rerun: settings parity and the flash gate

New suffix dirs only (M13): `pdhd/work/<run6>_allpd<evtid>_wf/` (61) and
`pdvd/work/<run6>_light<evtid>_wf/` (120), 6 jobs in parallel, every `rc=0`.

**Settings parity first.** The production archives are from July (PDHD `_allpd<evtid>`
Jul 8; PDVD `_light<evtid>_keep`, the `opflash_input=` every production `.tlas` names).  A
census of the 181 compiled production configs found ONE signature per detector, and today's
PDVD runner defaults differ from it by the flash tail merge (`PDVD_FLASH_TAIL_MERGE`, default
1 since 2026-07-22; the `_keep` set predates it) — the first PDVD smoke run therefore gave a
different flash archive.  With `PDVD_FLASH_TAIL_MERGE=0` the compiled config differs from
production only in the sink / fanout nodes and the paths (checked node by node, every
event); PDHD needed no override.

**The gate** (`pdhd/docs/scan/d30/light_gate.txt`, `abtest/hash_archive.py` on the opflash
archives, production vs `_wf`, plus the node-by-node config comparison):

| detector | events | opflash content hash identical | compiled-config parity |
|---|---|---|---|
| PDHD | 61 | **61/61** | 61/61 |
| PDVD | 120 | **120/120** | 120/120 |

So the waveforms in the release are the ones the production flashes were built from.  Cost:
PDHD 17 s/event (+ the decoana conversion), PDVD 11 s/event; disk 7.1 G (PDHD) + 6.4 G
(PDVD) of frames under `work/*_wf/` — the release copies only the windows.

Alignment of the two products was checked before building: the flash time (opflash column
0, ns) mapped onto the frame axis (`tickinfo` = [frame time, tick, tbin0]; sample i is at
`time + (tbin0 + i) * tick`) lands on the OpHit peaks of that flash within a few ticks on
both detectors (PDHD evt 1135 flashes 94 / 72 / 134; PDVD evt 298567 flashes 237 / 156).

## 4. The release

Layout (identical in both directories; the scripts are detector-parametrised copies, not
symlinks, so a directory can be copied away whole):

```
README.md ROOTFILE.md LIGHT.md BEE.md   index.csv  dqdx_ref.json  build_log.txt
events/<run6>_<evtid>/stm_michel_<det>_<run6>_<evtid>.root   9 TTrees
events/<run6>_<evtid>/bee_<det>_<run6>_<evtid>.zip           = the arm's mabc-pr.zip
scripts/stm_release.py build_release.py list_candidates.py plot_dqdx_vs_rr.py plot_michel_energy.py
        plot_muon_energy.py plot_flash_waveforms.py plot_event_2d.py relplot.py
figs/   the figures the scripts make on the release
```

The per-event ROOT file (uproot-written **TTrees** — note that uproot 5.7's plain
`file["T"] = ...` assignment writes an RNTuple, which older ROOT cannot read; the builder
uses `mktree` explicitly):

| tree | rows | content | source |
|---|---|---|---|
| `T_event` | 1 | provenance (arm, commits, dirs, wires file), offsets, counts, the window | `.tlas`, `Trun`, opflash metadata |
| `T_stm` | candidate | every `T_stm_michel` branch + `run, event, flash_id, flash_time_us, flash_total_pe, cluster_length_cm, cluster_npoints` | `T_stm_michel` ⋈ `T_cluster` ⋈ opflash |
| `T_stm_pts` | chain point | x,y,z, q (dQ/dx e/cm), L, rr, role, seg_id, q_sup + verdict columns | `T_stm_michel_pts` |
| `T_stm_fit` | PR fit point of a candidate cluster | x,y,z, q/nq + `dQ, dx, dqdx`, pu/pv/pw/pt, rr, ids | `T_rec_charge` |
| `T_stm_2d` | fitted 2-D cell of a candidate cluster | plane, chan_rank, channel, apa, face, wire, time_slice, tick, charge, charge_err, charge_pred | `T_proj_data` flattened; channel decoded with the visitor's rank scheme rebuilt from the wires file (`wirecell.util.wires.persist`; PDHD nch 3200/3200/3840, PDVD 3808/3808/4672) |
| `T_michel_2d` | cell of the Michel / region table | as written + run, event | `T_stm_michel_2d` |
| `T_flash` | every flash | time (light + charge axes), total_pe, pe[nchan], y/z centre / width, nhits, matched / STM cluster ids; PDVD sat[], cov[] | opflash tensors + `T_cluster` |
| `T_ophit` | hit of an STM-matched flash | channel, opdet, times (µs), area, amplitude, pe, fast_to_total | ophits tensor |
| `T_opwf` | (STM-matched flash, readout channel) | branch, t0_us, tick_ns, n, raw[n], decon[n], decon_roi[n], n_raw_nonzero, pe | the `_wf` frames, window [−2, +10] µs |

Every number is copied; the derived columns are listed in ROOTFILE.md §9.  The PDVD
DAPHNE-channel → OpDet map comes from `pdvd-opch-map.json`; on PDHD channel = OpDet.

**Self-checks the builder runs on every event** (`build_log.txt`):
* `T_stm.t0_us` equals the matched flash's time (`T_cluster.flash_id` → opflash row):
  max |Δ| = **0.0 µs** on every candidate of both detectors (the join is exact);
* every OpHit of an exported flash lies inside its waveform window;
* the deconvolved trace of the brightest PD of each exported flash peaks within 1 µs of
  the flash time (a WARN otherwise: PDHD 23 of 479 checks, PDVD 117 of 1218 -- in every case looked at, a brighter later pulse in the same window (the Michel's light, or a second flash within 10 µs) or a dim self-triggered PD whose snippet starts inside the window; none is an alignment error, the OpHit peaks of the flash itself sit at the flash time on every event);
* the production Michel estimator re-derived from `T_michel_2d` with the C++ rule reproduces
  `michel_q2d_region_{u,v,w}` and the cell counts (relative deviation < 1e-6): **PDHD 137/137, PDVD 204/204** candidates with a Michel object reproduce the stored per-plane sums and cell counts;
* row counts equal the source trees.

Census: 

| detector | events (no candidate) | candidates | `is_stm` | STM+Michel | STM-matched flashes | `T_opwf` rows | OpHits of those flashes outside [−2, +10] µs | file size |
|---|---|---|---|---|---|---|---|---|
| PDHD | 61 (1) | 325 | 135 | 85 | 313 | 49 014 | 46 / 15 941 (late light beyond +10 µs) | 0.3–8.1 MB, 266 MB total |
| PDVD | 120 (1) | 546 | 285 | 178 | 529 | 27 295 | 0 / 12 903 | 0.3–4.4 MB, 309 MB total |

PDVD carries 324 `T_opwf` rows (DAPHNE channel 2031, runs 039253 / 039349) that are recorded
but not in `pdvd-opch-map.json`: raw only, `opdet = -1`, empty decon.  One event per detector
has no candidate at all (PDHD 028084 idx 13, PDVD one of the 120); its file has the full
schema with empty charge trees and the complete `T_flash`.

## 5. The documents and the example scripts

* `README.md` — what it is, layout, quick start, what a candidate is, provenance, caveats.
* `ROOTFILE.md` — tree by tree, branch by branch, with the joins, the role and reject-bit
  codes, the energy definitions, the channel-rank coordinate, the derived-column list and
  an auto-generated appendix of every `T_stm` branch.
* `LIGHT.md` — the two time axes and the offsets, `T_flash` / `T_ophit` / `T_opwf`, the
  OpDet numbering and the PDVD channel map, the raw zero-fill convention, saturation, how the
  light was reconstructed.
* `BEE.md` — how to upload and read the zip, which layer is what, how to find a candidate
  (the `stm` layer's cluster ids are exactly the `T_stm` rows).

The scripts double as the worked examples of the data structure (each md quotes the call):
`list_candidates.py` (the table), `plot_dqdx_vs_rr.py` (pooled 2-D density + binned median
vs `dqdx_ref`, and the ratio with a bootstrap over tracks; or one track), `plot_michel_energy.py`
(region vs best spectra, region vs best per candidate, the T_michel_2d re-derivation
against the stored per-plane sums, the body control), `plot_muon_energy.py` (range vs MCS vs
dQ/dx), `plot_flash_waveforms.py` (PE pattern + raw / decon of the brightest PDs with the
OpHit times), `plot_event_2d.py` (the three-plane charge picture with the fit projection and
the Michel cells outlined).  Figures in `figs/` of each release: `dqdx_vs_rr.png`, `dqdx_track_<evt>_c<id>.png`, `michel_energy.png`, `muon_energy.png`, `flash_waveforms_<evt>_c<id>.png`, `event_2d_<evt>_c<id>{,_zoom}.png`, plus `candidates.txt`, `michel_energy.txt`, `muon_energy.txt`.

Read-outs on the releases (the scripts' printed lines): 

| read-out | PDHD | PDVD |
|---|---|---|
| dQ/dx vs rr, plateau (40–60 cm) median measured / expected | 1.016 (135 muons, 46 898 points) | 1.049 (285 muons, 82 399 points) |
| Michel `michel_ke_q2d_region` median / mean / above 52.8 MeV | 38.6 / 40.2 MeV / 21 of 85 | 31.8 / 31.4 MeV / 6 of 178 |
| Michel `michel_ke_best` median | 24.8 MeV | 22.4 MeV |
| body control `michel_ke_q2d_ctl` median | 9.6 MeV | 2.1 MeV |
| muon KE range median; MCS / range; dQ/dx / range | 424 MeV; 0.943 (MCS on 125/135); 1.071 | 329 MeV; 0.968 (249/285); 1.053 |

These reproduce the standing pictures of docs pdhd/26 and pdvd/50 on the production arms
(PDHD's larger Michel floor and its APA0 charge scale; PDVD's plateau above expectation in
the bottom volume); the release is the chain's output, so the numbers are the chain's.

## 6. What is NOT in the release, and caveats

* No per-point dE/dx: the chain stores dQ/dx (e/cm, after SP) and converts to energy through
  its recombination model; the expectation table uses the same model.
* `decon_roi` exists only on the continuous-stream branches (PDHD OpDets 120-159, PDVD
  cathode); the self-triggered branches have raw + decon.  Raw is zero-filled outside a
  snippet (`n_raw_nonzero`).
* The waveform window is [−2, +10] µs around the flash time by default (`--pre/--post` of the
  builder).  The first smoke build used −1 µs and showed the first OpHit of a bright flash
  peaking 0.85 µs before the flash time (the 1 µs seed bin), so the pre-margin was widened.
* No hand-scan labels: the release carries the chain's output.  The owner's purity /
  efficiency numbers (docs pdhd/26, 28; pdvd/116) are quoted in the README for context.
* Bee upload stays owner-gated (`upload-to-bee.sh`); colleagues upload the zip themselves
  through the Bee site.
* The release data (`events/`, `figs/`, `index.csv`, `dqdx_ref.json`, `build_log.txt`) are
  NOT committed (0.7 GB); the scripts and documents are.  `build_release.py` rebuilds it.

## 7. Files and commits

| what | where |
|---|---|
| the knob | toolkit `cfg/pgrapher/experiment/pdhd/wct-light-allpd-reco.jsonnet`, `protodunevd/wct-light-reco.jsonnet` (`03f0a569`) |
| knob gate artefacts | `pdhd/docs/scan/d30/cfg_knob_gate.txt`, `*_before.json`, `*_after.json`, `*_on.json` |
| light rerun gate | `pdhd/docs/scan/d30/light_gate.txt` (181 lines + 2 summaries) |
| runner knobs | `pdhd/run_light_allpd_evt.sh` (`PDHD_LIGHT_FRAMES`), `pdvd/run_light_evt.sh` (`PDVD_LIGHT_FRAMES`) |
| the frames | `pdhd/work/<run6>_allpd<evtid>_wf/`, `pdvd/work/<run6>_light<evtid>_wf/` (13.5 G, not committed) |
| the releases | `pdhd/stm_michel_release/`, `pdvd/stm_michel_release/` (scripts + docs committed) |
| this doc | `pdhd/docs/30_stm-michel-release.md` |

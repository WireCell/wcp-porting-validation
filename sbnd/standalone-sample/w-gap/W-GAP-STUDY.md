# W-gap study: SP rebaseline, DNNROI truncation, charge in-efficiency & bias

Investigation (June 2026) of charge in-efficiency and per-channel charge bias on
the SBND collection ("W") plane, and the signal-processing bugs/choices behind
them. All software lives under `sbnd/standalone-sample/w-gap/`; the WCT C++
changes are local commits on the `wire-cell-toolkit` `apply-pointcloud` branch
(not pushed).

## Sample & how to reproduce

- Input: `standalone-sample/2025f-mc.root` (13 events) and its SP-frame bundle
  `2025f-mc-sp-frames.tar.bz2`.
- Sim/SP job: `w-gap/standard_detsim_sbnd-dump.fcl` (WCT sim-drift + NF + SP +
  DNN-ROI). Re-sim quickref in `standalone-sample/README.md`.
- WIRECELL_PATH gotcha: sbndcode WCT sim fcl needs sbndcode cfg prepended over
  `opt/share/wirecell`; for the *standalone wire-cell* img/matching chain the
  toolkit `cfg` must also be prepended (NOT for lar/wcls).

## Findings

### 1. SP `rebase_waveform` was biased by in-window signal
`OmnibusSigProc::rebase_waveform` subtracted a straight line anchored on the
plain MEAN of the first/last 200 ticks. A signal pulse inside either window
(e.g. ch 5259 peak at ticks ~110–156) biased the anchor and tilted the whole
channel, inflating `cal_RMS` ~6x (→ ~1190 vs ~190 on quiet channels) and the
ROI threshold, suppressing real ROIs on W.

The NF stage (`mbOneChannelNoise` → `SignalFilter` + `RawAdapativeBaselineAlg`,
`sigproc/src/Microboone.cxx`) already does a peak-immune adaptive rebaseline, so
the SP linear rebase was redundant *and* harmful. SBND now sets
`rebase_planes: []` in `pgrapher/experiment/sbnd/sp.jsonnet`.

Two signal-safe anchor methods were added as a config option
(`rebase_method: median|sigmask`, `rebase_nsigma`, default sigmask; the biased
"mean" was removed). 10-event scan verdict: with NF active, dnnsp charge bias is
insensitive to the SP rebase choice (none +0.62% vs sigmask +0.58% median) — so
`rebase_planes: []` is fine for SBND; `sigmask` is the safe option where NF
rebaselining is absent.
- WCT commits: `70bd3015` (median/sigmask), `97f9d233` (drop mean).

### 2. DNNROIFinding silently truncated dnnsp in drift time
`pytorch/src/DNNROIFinding.cxx` derived `input_ticks` from the FIRST tagged
trace's size. Under SBND `roi=="both"` the SP output is SPARSE, so the window
collapsed to one arbitrary ROI's length (logged 19–3392 vs the full 3427) and
all charge later in drift time was dropped → "missing tracks". This looked
scheduler-dependent (TbbFlow vs Pgrapher gave different deficits) but BOTH
truncated — just by different random amounts (different noise → different first
ROI). Fix: `input_ticks` = max trace extent (`tbin + size`) over ALL `intags`
plus `decon_charge_tag`. Verify via `input_ticks=3427` in lar logs.
- WCT commit: `8e883236`.

### 3. larwirecell QLMatching link fix (incidental)
`normalize_cluster_flags` had become file-`static` in
`clus/src/MultiAlgBlobClustering.cxx`; the larwirecell `wclsQLMatching` plugin
forward-declares + links it. Re-exported in namespace `WireCell::Clus::Facade`.
No larwirecell rebuild needed. WCT commit: `a25f0225`.

## rebase-method scan

`w-gap/rebase-scan/` runs 4 SP rebase variants (mean/median/sigmask/none) on the
same events, single-threaded **Pgrapher** for deterministic (bit-identical)
noise so |A−B| isolates the method:
- `cfg-{mean,median,sigmask,none}/` each shadow only `sp.jsonnet`.
- `cfg-pgrapher/` holds sibling symlinks + a SYMLINK of
  `wcls-sim-drift-depoflux-nf-sp.jsonnet` to the sbndcode tree (managed there;
  its `app.type` ~line 605 flips Pgrapher/TbbFlow and must match the fcl `apps`;
  `sed_label` uses `std.extVar('inputTag')`).
- `run-scan.sh` (nohup + STATUS marker) builds then runs all 4 × 10 events.
- Outputs `rb_*/sp.root`; `rb_{none,sigmask}_test/` are 1-event byte-identity
  checks confirming the debug-code removal didn't change physics (|A−B|=0).

## Charge bias / in-efficiency conclusions (10 events)

Per-channel charge bias `(Q_reco − Q_truth)/Q_truth` vs simchannel truth:
- gauss median bias ~+2.1%, dnnsp ~+0.3–0.6%.
- ~84% of in-efficiency channels (bias < −0.9) sit below 1e4 e⁻ truth; gauss
  loses more of those very-low-charge channels than dnnsp (legend totals can
  disagree with the plotted [1e4,1e6] curves for this reason).
- rebase variants nearly indistinguishable for dnnsp.

## Tools (in `w-gap/`, committed in wcp-porting-img main, not pushed)

- **`compare_wires_viewer.py`** + `serve_compare_wires.sh` (Bokeh, port 5010):
  3 linked 2D panels (A, B, A−B, bipolar cmap, transposed channel×tick) +
  1D overlay (A line / B dots / always-on simchannel(A) line) + signed-relative
  diff mode + APA/plane selectors + clickable region-scoped top-|A−B| table +
  zoom-range charge integrals colored by sim-closeness. recob::Wire ×50→electrons;
  simchannel tick = tdc − 2990; A/B independent file:tag (e.g. gauss vs dnnsp).
  Restart gotcha: `pgrep -f 'bokeh serve'` then `kill <pids>` in a SEPARATE Bash
  call (a compound kill matches its own cmdline → exit 144).
- **`plot_charge_bias.py`**: file:alg pairs → per-pair 2D bias hist + ineff +
  mean-bias; `<out>_grid/` with both 1D metrics AND a per-input 2D hist for every
  tick×plane combo (`--tick-split`, default 200; `--xbins/--xbins1d`). Generated
  grids: `charge_bias_grid/` (split 200), `charge_bias_grid_1000/` (split 1000),
  `charge_bias_grid_1000_x1e4/` (split 1000, truth axis 1e4–1e6).
- **`vis_waveforms.py`**, **`vis_response.py`**: single-channel waveform overlay
  across SP stages, and the OSP overall-response (FR⊗ER decon kernel) viewer.

## Standalone imaging + Q/L matching + BEE (xin chain)

From `standalone-sample/` (see README): prepend WIRECELL_PATH
`photodet : sbnd_xin : wire-cell-toolkit/cfg`, then
1. `wire-cell ... -c ../sbnd_xin/wct-img-all.jsonnet` → `icluster-apa{0,1}-active.npz`
2. `wire-cell -V reality=sim -V input=. -V frames=<bundle> -V semimodel_file=semi-analytical-sbnd.json -C DL=6.2 -C DT=9.8 -C lifetime=6 -C joint=true -C pmt_nl=true -c ../sbnd_xin/wct-clus-matching-standalone.jsonnet` → `mabc.zip`
3. BEE upload: `BROWSER=echo bash ../sbnd_xin/upload-to-bee.sh mabc.zip`
   → `https://www.phy.bnl.gov/twister/bee/set/<UUID>/event/list/`

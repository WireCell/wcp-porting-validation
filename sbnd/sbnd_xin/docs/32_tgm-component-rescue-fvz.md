# TGM component rescue + downstream-z FV inset (doc 32)

Two opt-in refinements of the merge-aware TGM chain (docs 29/30/31), motivated
by two hand-scan misses in the MCP2025C 20-event sample:

1. **Component rescue** (`tgm_component_rescue`, C++ `component_rescue`,
   default OFF): a connected component SHORTER than `component_min_length`
   (10 cm) still donates its extreme points when it is path-connected — same
   30 cm-step charge-path rule as the path-mode chord guard — to a component
   that passed the length cut.
2. **Downstream-z FV inset** (`tgm_fv_zmax_margin`, default 3 cm =
   byte-identical legacy): parametrizes the z ≈ 500 cm face inset of the
   shared TGM/FC fiducial box (`sbnd_pr_fv_margins[4]`); the new operating
   point is 5 cm.

## Symptom

- **evt286681 bundle grp 13 (t = 1207.6 µs) main 7**: a genuine anode→top
  corner clipper (x = −201.2 → y = +198.8 at z ≈ 235 cm) arrives as 3
  fragments (gaps 4.4 / 6.0 cm).  The last 2.5 cm before the top wall — the
  only points beyond the y boundary on that end — is its own 16-pt component,
  below the 10 cm `component_min_length`, so `component_extremes` (doc 30's
  anti-speck guard, evt285185 clus 20) silently drops the exit and no CASE-A
  pair forms.  Tagged STM instead of TGM.
- **evt286065 bundle grp 6 (t = −703.3 µs) main 13**: bottom-wall entry
  (y = −199.8), far end ranges out at z = 496.13 cm — 1.02 cm INSIDE the
  effective boundary 497.15 cm (box 500.15 − 3 cm margin), 5.0 cm from the
  physical wire-bbox face at 501.15 cm.  One qualifying end only ⇒ no TGM.

## Root cause

The anti-speck guard's min-length cut cannot distinguish a merge-grafted
speck (path-DISCONNECTED, ≥ 93 cm from any other charge on this sample) from
a genuine track end fragmented behind small gaps (path-connected within
30 cm).  Path connectivity — already trusted by the chord guard — separates
the two exactly.  Independently, the 3 cm downstream-z margin leaves a ~4 cm
blind band in front of the physical z face where a real exit reads
"contained".

## Fix

- `clus/src/TaggerCheckTGM.cxx`: `component_rescue` knob.  In
  `component_extreme_wcps()`, components failing the length cut are retried:
  if any point shares a `path_components()` id with a kept component, its 8
  extremes join the union (same 5 cm proximity grouping).  Knob off ⇒ pass
  never taken, byte-identical.
- `cfg/pgrapher/common/clus.jsonnet`: `component_rescue=false` builder arg,
  key-suppressed.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `tgm_component_rescue`,
  `tgm_fv_zmax_margin` (cm, default 3) threaded into `clus_pr`;
  `sbnd_pr_fv_margins[4]` (the `contained(z - tv[4])` slot = downstream face)
  now `-tgm_fv_zmax_margin * wc.cm`.  Shared by `tagger_check_tgm` AND
  `tagger_check_fc` (docs/27 consistency: containment keeps one meaning, so
  FC also sees the tighter face).
- `sbnd_xin/wct-pr-perevt.jsonnet`: TLAs `tgm_component_rescue=false`,
  `tgm_fv_zmax_margin=3`.
- `sbnd_xin/run_nusel_evt.sh`: `-rescue` / `SBND_TGM_RESCUE=1` and
  `-fvz <cm>` / `SBND_TGM_FVZ_MARGIN=<cm>`.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# QL products + imaging symlinked per event from work-mcp10-ctpcfix / work-mcp1000-ctpcfix
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-tgmfv ./run_nusel_evt.sh data all -chord -rescue -fvz 5
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-tgmfv \
    ./run_nusel_evt.sh data 1 -chord -rescue -fvz 5
done
python3 nusel_extract.py --merge work-mcp1000-tgmfv/nusel_evt*/nusel-evt*.tsv \
  --out work-mcp1000-tgmfv/nusel-table.tsv \
  --events-out work-mcp1000-tgmfv/nusel-events.tsv
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-tgmfv \
  --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix \
  --prev ../work-mcp10-mainflag:mcp10-mainflag --prev ../work-mcp10-chord:mcp10-chord \
  ../work-mcp10-tgmfv ../work-mcp1000-tgmfv
```

## Verification

**Byte-identical, knobs off.**
- Compiled config: HEAD cfg + HEAD sbnd_xin jsonnets vs new trees, ctpcfix
  TLAs (`-chord`, path mode, nucand on) and new TLAs at defaults
  (`rescue=false, fvz=3`) — `cmp` PASS
  (scratch `tgmfv-cfg/{old,new}-off.json`).
- End-to-end: evt286065 rerun with new `libWireCellClus.so`, `-chord` only,
  into `work-mcp10-tgmfv-offgate` — `mabc-pr.zip` member hash
  `b3a1c877…` identical to `work-mcp10-ctpcfix`, TSV identical, all tagger
  verdict lines identical.
- `./build/clus/wcdoctest-clus`: 41/41 pass.

**Knobs on** (`-chord -rescue -fvz 5`, tags `work-mcp10-tgmfv` /
`work-mcp1000-tgmfv`):
- Compiled config gains `component_rescue: true` on `TaggerCheckTGM:pr` and
  `fv_tolerance[4] = -50` (mm) on both `TaggerCheckTGM:pr` and
  `TaggerCheckFC:pr`.
- evt286681 main 7: `component_rescue: cluster 7 rescued 2 of 2 short
  component(s)` → **TGM** (was STM).
- evt286065 main 13: tip z = 496.13 > new boundary 495.15 → **TGM** (was
  not-tagged).

Full 20-event verdict diff vs ctpcfix: see table below.

## Verdict flips (ctpcfix → tgmfv)

20 events, 214 tabulated bundles.  6 physics flips (below); a further 8
`-1↔0` STM/FC column changes are log-tearing parse artifacts, not physics
(`-1` = verdict line torn by interleaved TICK/MEM output; the C++ always logs
a verdict, so a `-1↔0` move can never be a real flip — verified on
evt286197's old log, STM lines for clusters 3-6 swallowed mid-stream).

| bundle | flip | knob | reading |
|---|---|---|---|
| evt286065 main 13 (361 cm, t=−703 µs) | not-tagged → **TGM** | fvz | target: bottom-wall entry, tip z=496.13 now outside 495.15 |
| evt286681 main 7 (54 cm, t=1208 µs) | STM → **TGM** | rescue | target: anode→top clipper, 16-pt top-exit fragment rescued (2/2) |
| evt287099 main 6 (231 cm, t=−728 µs) | not-tagged → **TGM** | fvz | single-component track ending in the 495–497 cm band |
| evt288727 main 6 (227 cm, t=411 µs) | not-tagged → **TGM** | rescue | 6 of 8 short fragments rescued along the track |
| evt285999 main 1 (0.9 cm, t=−757 µs) | not-tagged → **TGM** | fvz | degenerate speck at the downstream face, both "ends" now outside (same class as the pre-existing evt286065 main 6 TGM) |
| evt287517 main 16 (98 cm, t=561 µs) | **TGM → not-tagged** | fvz | REGRESSION: genuine top/downstream corner clipper. Whole chord (−180.5, 199.7, 493.4)→(−177.7, 102.0, 500.9) hugs the downstream wall inside the widened band, so the CASE-A midpoint test (`mid_inside`) flips false and the re-check chords to a far merge fragment land inside ⇒ no tag. `WCT_TGM_DEBUG=1` replays in `work-mcp1000-tgmfv-dbg{3,5}` |

Net TGM on the sample: +4 / −1.  No FC 1→0 flip from the tighter face on
these 20 events (FC column changes are all parse artifacts).

**Caveat for the 5 cm operating point**: a track that RUNS ALONG the
downstream wall inside the 3→5 cm widened band loses the CASE-A midpoint
support (evt287517 above).  If hand-scan shows this class matters, the margin
change may need to be endpoint-only (a separate tolerance for the
midpoint/`flag_check` test) rather than a uniform inset.

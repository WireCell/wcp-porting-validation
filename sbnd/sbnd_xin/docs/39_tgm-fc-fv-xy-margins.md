# TGM/FC fiducial x/y margin knobs: X 2.0 → 2.5 cm, Y 2.5 → 3.0 cm (doc 39)

Owner-requested widening of the TGM/FC fiducial-box insets in the drift-x
and vertical-y dimensions, delivered as two default-legacy knobs:

- **`tgm_fv_x_margin`** (cm; default 2 = byte-identical legacy): the |x| ~
  201.05 cm anode-face inset, both faces symmetric.  Runner
  `run_nusel_evt.sh -fvx <cm>` / env `SBND_TGM_FVX_MARGIN`.
- **`tgm_fv_y_margin`** (cm; default 2.5 = byte-identical legacy): the |y| ~
  199.312 cm top/bottom inset, both faces symmetric.  Runner `-fvy <cm>` /
  env `SBND_TGM_FVY_MARGIN`.

Both parametrize `sbnd_pr_fv_margins` (and the x/y entries of the doc-35
interior vector, which continues to differ from the endpoint vector only in
downstream-z), so — like `tgm_fv_zmax_margin` — they are shared by
`tagger_check_tgm` AND `tagger_check_fc`: "contained" keeps one meaning
across both verdicts (docs/27).  TaggerCheckSTM takes no `fv_tolerance` and
is untouched.  Pure jsonnet: no C++ change, no rebuild.

Threading: `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`clus_pr` + `pr()`),
`wct-pr-perevt.jsonnet` (TLAs), `run_nusel_evt.sh` (`-fvx`/`-fvy`).

New operating point: QL `-save-rcid`; PR
`-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3`.

## Verification

- Compiled-config proofs (wcsonnet, pre-edit baselines vs post-edit):
  - knob off at the doc-38 op-point TLAs: `cmp`-identical;
  - knob off at production-default TLAs: `cmp`-identical;
  - knob on (`-fvx 2.5 -fvy 3`): TGM+FC `fv_tolerance` become
    `[-25,-25,-30,-30,-50,-30]` mm and the TGM `interior_fv_tolerance`
    x/y entries track (`[-25,-25,-30,-30,-30,-30]`), z staying
    endpoint-only per doc 35.
- 30-event reprocess `work-mcp{10,1000,1000b}-fvxy` (QL per-event dirs
  symlinked from `*-mainreal` — pctree reused, PR-only rerun; 30/30 ok).
  Verdict diffs vs `*-mainreal`: **four flips, all not-tagged → TGM,
  all out-of-beam**; no losses, no nu-candidate changes, and the FC
  column is unchanged on all 30 events:
  - evt284657 main 7 (401.2 cm)
  - evt285185 main 18 (394.2 cm)
  - evt286021 main 7 (3.2 cm, 22 pts): an anode-wall stub at
    x 198.7–201.2 cm.  Old x inset 2.0 put the FV boundary at 199.05 so
    its inner end sat 0.36 cm INSIDE; at 2.5 both ends are outside =>
    TGM.  The out-of-time-cosmic apparent-x class the x widening
    targets, but flagged here for the hand-scan: near-wall stubs whose
    whole extent lies in the widened band are now taggable.
  - evt287517 main 12 (233.8 cm)
  - (evt287517 main 14 `stm 0→-1` in the TSV is the usual torn-log
    parse artifact — the log line's prefix is clobbered; the actual
    verdict `STM=0 TGM=0` is unchanged.)
- Viewer: `:5010` tag `mcp10-fvxy` over the three fvxy roots, `--prev`
  mainreal (×3) → mainpair → fvzi → lm2 → ctpcfix.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# fresh roots; QL per-event dirs symlinked from the mainreal roots so the
# runner reuses the -save-rcid pctree and only the PR job reruns
for suff in mcp10 mcp1000 mcp1000b; do
  mkdir -p work-$suff-fvxy
  for d in work-$suff-mainreal/ql_evt* work-$suff-mainreal/evt*; do
    ln -sn $PWD/$d work-$suff-fvxy/$(basename $d)
  done
done
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3"
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-fvxy ./run_nusel_evt.sh data all $F
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in $(seq 10 19); do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-fvxy ./run_nusel_evt.sh data 1 $F
done
for e in $(seq 20 29); do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-fvxy ./run_nusel_evt.sh data 1 $F
done
for r in work-mcp1000-fvxy work-mcp1000b-fvxy; do
  python3 nusel_extract.py --merge $r/nusel_evt*/nusel-evt*.tsv \
    --out $r/nusel-table.tsv --events-out $r/nusel-events.tsv
done
```

# TGM interior-support FV: endpoint-only downstream-z widening (doc 35)

Opt-in refinement of the doc-32 downstream-z FV inset: `check_tgm`'s CASE-A
INTERIOR-support tests (chord midpoints + waypoint re-check) get their own
tolerance vector, so the 3→5 cm endpoint widening no longer starves a
wall-hugging corner clipper of its midpoint support.  This is exactly the
remedy doc 32's caveat anticipated ("the margin change may need to be
endpoint-only (a separate tolerance for the midpoint/`flag_check` test)
rather than a uniform inset").

## Symptom

- **evt289805 bundle grp 6 (t = −686.062 µs) main 9 + companion 8**
  (owner-flagged, e20–e29 slice): a 114.2 cm top→downstream-z corner
  clipper.  Extreme ends (−123.4, **199.7**, 494.0) — outside the top face —
  and (−119.4, 85.7, **500.9**) — outside the downstream face.  The whole
  track lives in the z = 493–501 cm band; 1278/1736 points sit outside the
  effective FV.  Clearly TGM, not tagged at `-fvz 5`.
- **evt287517 main 16 (98 cm, t = 561.007 µs)**: same topology, chord
  (−180.5, 199.7, 493.4) → (−177.7, 102.0, 500.9).  Was TGM at the legacy
  3 cm margin and LOST the tag when doc 32 widened it to 5 cm (the single
  regression in doc 32's flip table).

## Root cause

The doc-32 widening insets the downstream face for EVERY `inside_fv` call in
`check_tgm`, not just endpoint qualification.  For a track running ALONG the
downstream wall inside the widened band:

- both CASE-A pair ends correctly read outside, but the chord between them
  also lies entirely beyond z = 495.15 cm ⇒ the midpoint test (`flag_check`)
  reads false ("track seems to leave the FV in the middle");
- the fallback waypoint re-check (`flag_check_again`) samples chords through
  the OTHER extreme groups, which dip at most ~2.2 cm inside the effective
  face (cluster z_min = 493.0) ⇒ some waypoint lands inside ⇒ the re-check
  vetoes the pair.  `WCT_TGM_DEBUG=1` replay (evt289805 cluster 9):

```
check_tgm dbg: cluster 9 pair (0,1) ngrp 4 pe1 (-123.4,199.7,494.0) pe2 (-119.4,85.7,500.9) mid_inside false len 114.2/114.2 cm
```

At the legacy 3 cm inset the first quarter-point (z ≈ 495.7 <
497.15) is inside ⇒ `flag_check` true ⇒ both clusters tag directly.  The
interior test, not the endpoint test, is what the widening broke.

## Fix

- `clus/src/TaggerCheckTGM.cxx`: new config `interior_fv_tolerance`
  (default empty ⇒ fall back to `fv_tolerance`, byte-identical).  Used by
  `inside_fv_interior()` at exactly three call sites: the CASE-A
  `flag_check` chord midpoints and both `flag_check_again` waypoint loops.
  Endpoint outside/inside tests, CASE-B, and the dead-volume/SP checks keep
  `fv_tolerance`.
- `cfg/pgrapher/common/clus.jsonnet`: `interior_fv_tolerance=[]` builder
  arg, key-suppressed when empty.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `tgm_fv_zmax_margin_interior`
  (cm, default 0 = OFF).  When > 0, `tagger_check_tgm` gets the margins
  vector with index 4 = −interior·cm; `tagger_check_fc` and the endpoint
  margins are untouched (docs/27 containment meaning unchanged for exit
  qualification and FC).
- `sbnd_xin/wct-pr-perevt.jsonnet`: TLA `tgm_fv_zmax_margin_interior=0`.
- `sbnd_xin/run_nusel_evt.sh`: `-fvzi <cm>` / `SBND_TGM_FVZ_INTERIOR=<cm>`.

Operating point: `-fvz 5 -fvzi 3` — endpoint qualification keeps the doc-32
5 cm inset (evt286065-class ranged-out tips still count as exits), interior
support reverts to the legacy 3 cm semantics.  Since the 5 cm box is a
strict subset of the 3 cm box, any pair whose `flag_check` passed at fvz 5
still passes at interior 3; the change can only ADD midpoint support.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# fresh roots; per-event imaging AND ql products symlinked from the lm2 roots
for pair in "work-mcp10-lm2 work-mcp10-fvzi" "work-mcp1000-lm2 work-mcp1000-fvzi" \
            "work-mcp1000b-lm2 work-mcp1000b-fvzi"; do set -- $pair
  mkdir -p $2; for d in $1/evt* $1/ql_evt*; do ln -sfn $PWD/$d $2/$(basename $d); done
done
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-fvzi \
  ./run_nusel_evt.sh data all -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-fvzi \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm
done
for e in 20 21 22 23 24 25 26 27 28 29; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-fvzi \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm
done
for r in work-mcp1000-fvzi work-mcp1000b-fvzi; do
  python3 nusel_extract.py --merge $r/nusel_evt*/nusel-evt*.tsv \
    --out $r/nusel-table.tsv --events-out $r/nusel-events.tsv
done
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-fvzi \
  --prev ../work-mcp10-lm2:mcp10-lm2 --prev ../work-mcp1000-lm2:mcp1000-lm2 \
  --prev ../work-mcp1000b-lm2:mcp1000b-lm2 --prev ../work-mcp10-lm:mcp10-lm \
  --prev ../work-mcp10-reschord:mcp10-reschord --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix \
  ../work-mcp10-fvzi ../work-mcp1000-fvzi ../work-mcp1000b-fvzi
```

## Verification

**Byte-identical, knob off.**
- Compiled config: OLD trees (HEAD cfg + HEAD `wct-pr-perevt.jsonnet`) vs
  new trees at the production TLAs (`-chord` path mode, rescue,
  rescue-chord, nucand, fvz 5, no fvzi) — `cmp` PASS (scratch
  `fvzicfg/{old,new-off}.json`); `fvzi=0` also `cmp`-identical to omitting
  the TLA.
- Compiled-config proof, knob on: `fvzi=3` adds exactly
  `interior_fv_tolerance: [-20,-20,-25,-25,-30,-30]` (mm) on
  `TaggerCheckTGM:pr` only; `TaggerCheckFC:pr` unchanged.
- End-to-end: evt289805 rerun with the new `libWireCellClus.so`, knob off,
  into `work-mcp1000b-fvzi-offgate` — `mabc-pr.zip` member digest
  `65e8373b…` identical to `work-mcp1000b-lm2`, TSV identical.
- `./build/clus/wcdoctest-clus`: 518/518 assertions pass.

**Knob on** (`-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm`, tags
`work-mcp10-fvzi` / `work-mcp1000-fvzi` / `work-mcp1000b-fvzi`): full
30-event, 329-bundle diff vs the lm2 roots —

| bundle | flip | reading |
|---|---|---|
| evt289805 main 9 (114.2 cm, t=−686.062 µs) | not-tagged → **TGM** | owner-flagged target: top→downstream corner clipper |
| evt287517 main 16 (98.0 cm, t=561.007 µs) | not-tagged → **TGM** | doc-32 fvz regression undone |

No other tgm/stm/fc/lm/label change (two `-1↔0` stm/fc moves are the usual
log-tearing parse artifacts).  Both flips are out-of-beam, so the event
labels move not-tagged → TGM with no nu-candidate impact.

## Caveats

- The interior box (3 cm inset) is a superset of the endpoint box (5 cm
  inset), so relative to plain `-fvz 5` this knob can only ADD CASE-A tags
  whose midpoint support lies in the 495.15–497.15 cm band; relative to the
  legacy `-fvz 3` the interior semantics are IDENTICAL.  A pair that tagged
  at fvz 5 through the `!flag_check_again` route tags earlier via
  `flag_check` instead — same verdict.
- Only the downstream-z entry differs between the two vectors on SBND; the
  x/y/upstream-z insets are shared.  If a future margin retune widens other
  faces, revisit whether those should be endpoint-only too.

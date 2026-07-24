# TGM pairs must touch the main charge component (doc 36)

New default-OFF knob `main_component_pairs` in `TaggerCheckTGM` (TLA
`tgm_main_pair`, runner flag `-main-pair`): a CASE-A/CASE-B pair may tag
TGM only when at least one of its two ends lies in the cluster's **main
charge component** — the `path_components()` component (30 cm-step voxel
linkage, the same rule the path-mode chord guard and `component_rescue`
use) holding the most points.  The TGM verdict then follows the bundle's
main cluster instead of whichever merged-in fragment happens to be
through-going on its own.

## Symptom

**evt289343 bundle grp 5 (t = 1.363 µs, IN BEAM) main 9**
(`work-mcp1000b-lm2` / `-fvzi`): tagged TGM — which, being in-beam,
removes the bundle from neutrino candidacy — but the 910-pt merged
cluster's main track never touches two walls:

- Main track (586 pts, ~52 cm): x [−29.6, −5.5], y up to **199.8** (top
  wall), z 264–306.  ONE wall contact only.
- Merged-in fragment (236 pts, 26.5 cm): x [17.7, 33.9] — the *other*
  TPC — y down to **−199.1** (bottom wall) and z up to **500.2**
  (downstream wall).  A corner-clipping cosmic, through-going by itself,
  ~450 cm from the main track.

## Root cause

The doc-29 chord guard deliberately allows a within-component pair to tag
("a sub-track that is itself through-going still tags on its own
charge-supported pair") — correct when the question is "does this flash
group contain a through-going cosmic", wrong when the TGM label decides
the fate of the bundle's main cluster.  The debug rerun
(`WCT_TGM_DEBUG=1`, roots `work-mcp1000b-nucdbg`/`-nucoffdbg`) shows the
winning pair is (2,3) = the fragment's own two ends,
(17.7,−181.0,500.2)–(33.6,−199.1,490.1), 26.1 cm chord vs length_limit
51.7 cm (0.45·51.7 = 23.3 passes); all four cross-component pairs were
correctly path-rejected (410–462 cm chords).

`tgm_neutrino_candidate` (ON since doc 26 — the runner default) cannot
protect: the ported `check_neutrino_candidate(cluster, pe1, pe2)` walks
the in-cluster Dijkstra path **between the tagging pair's own endpoints**
— the smooth 26 cm cosmic stub — finds nothing neutrino-like, and lets
the tag through.  The main track's topology is never consulted.  With
`-no-nucand` the verdict flips to TGM=0 only via the blunt v1
"never tag anything in-beam" branch.

## Fix

- `clus/src/TaggerCheckTGM.cxx`: `main_component_pairs` knob (default
  false).  `path_components()` is now also computed when this knob is on;
  the main component is the label with the most points (ascending-id
  iteration, strict `>`, lowest id wins ties — deterministic).  A pair
  with NEITHER end's nearest cluster point in the main component is
  rejected before the tagging branches, in both CASE-A and CASE-B; the
  guard fails OPEN (empty kd lookup ⇒ no veto).  Log line:
  `check_tgm: cluster {} CASE-x pair ({},{}) rejected: neither end in the
  main charge component ({} cm chord)`.
- Why this spares the earlier fixes:
  - a cathode crosser is ONE path component (30 cm linkage bridges the
    CPA gap, doc 31) — its pair ends both sit in the main component;
  - evt286681's rescued top-exit speck is path-connected to its own track
    ⇒ same (main) component;
  - evt288727's two touching cosmics share one path component — that
    false positive is handled by `rescue_chord_check` (doc 33), not here;
  - main↔fragment cross pairs keep one end in the main component and
    remain for the chord guard to reject.
- `cfg/pgrapher/common/clus.jsonnet`: `main_component_pairs=false`
  builder arg, key-suppressed.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `tgm_main_pair=false`
  threaded through `clus_pr` / `pr()`.
- `sbnd_xin/wct-pr-perevt.jsonnet`: TLA `tgm_main_pair=false`.
- `sbnd_xin/run_nusel_evt.sh`: `-main-pair` / `SBND_TGM_MAIN_PAIR=1`.

## Verification

- Compiled-config proof at the doc-35 operating point (`-chord` path
  mode, rescue, rescue-chord, nucand, fvz 5, fvzi 3): old trees (HEAD
  cfg + HEAD `wct-pr-perevt.jsonnet`) vs new trees, knob off —
  `cmp`-identical; knob on adds exactly one `main_component_pairs: true`
  on `TaggerCheckTGM:pr`.
- `./build/clus/wcdoctest-clus`: 518/518.
- Knob-off runtime gate (new binary, doc-35 flags, knob OFF):
  `work-mcp{10,1000,1000b}-mpoff` vs `work-mcp{10,1000,1000b}-fvzi` —
  **30/30 events byte-identical PASS** (`mabc-pr.zip` member-content
  hashes via `abtest/hash_archive.py` + per-event TSVs).
- Knob-on 30-event run (`work-mcp{10,1000,1000b}-mainpair` vs the fvzi
  roots): exactly ONE verdict change — **evt289343 main 9 TGM 1→0,
  label TGM → nu-candidate** (in-beam bundle, so removing the false TGM
  promotes it to the event's neutrino candidate).  Sentinel:
  `check_tgm: cluster 9 CASE-A pair (2,3) rejected: neither end in the
  main charge component (26.1 cm chord)`.  Doc-35's recovered TGMs
  (evt287517 m16, evt289805 m9) and every doc-29..33 target verdict are
  unchanged; the only other TSV diffs are the usual torn-log `0↔-1`
  stm/fc parse artifacts (evt287099 m9 stm, evt287517 m16 fc, evt289849
  m10 stm — flipping both directions).
- Viewer: `:5010` tag `mcp10-mainpair` over the three mainpair roots,
  `--prev` fvzi (×3) → lm2 → reschord → ctpcfix.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# fresh roots; per-event imaging AND ql products symlinked from the fvzi roots
for pair in "work-mcp10-fvzi work-mcp10-mainpair" \
            "work-mcp1000-fvzi work-mcp1000-mainpair" \
            "work-mcp1000b-fvzi work-mcp1000b-mainpair"; do set -- $pair
  mkdir -p $2; for d in $1/evt* $1/ql_evt*; do ln -sfn $PWD/$d $2/$(basename $d); done
done
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-mainpair \
  ./run_nusel_evt.sh data all -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-mainpair \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair
done
for e in 20 21 22 23 24 25 26 27 28 29; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-mainpair \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair
done
for r in work-mcp1000-mainpair work-mcp1000b-mainpair; do
  python3 nusel_extract.py --merge $r/nusel_evt*/nusel-evt*.tsv \
    --out $r/nusel-table.tsv --events-out $r/nusel-events.tsv
done
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-mainpair \
  --prev ../work-mcp10-fvzi:mcp10-fvzi --prev ../work-mcp1000-fvzi:mcp1000-fvzi \
  --prev ../work-mcp1000b-fvzi:mcp1000b-fvzi --prev ../work-mcp10-lm2:mcp10-lm2 \
  --prev ../work-mcp10-reschord:mcp10-reschord --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix \
  ../work-mcp10-mainpair ../work-mcp1000-mainpair ../work-mcp1000b-mainpair
```

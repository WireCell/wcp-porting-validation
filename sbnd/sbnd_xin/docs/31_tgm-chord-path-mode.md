# 31 — Chord-charge guard "path" mode: stop rejecting curved through-goers

**Symptom.** `work-mcp10-mainflag` (doc 30) evt285185: the bundle at flash grp
13, t = −200.337 µs, main cluster 16 (+ companion 10) — a 479.9 cm, 4196-point
track — is `not-tagged`, though its ends sit on the top wall (y = +199.6) and
the TPC1 anode (x = +201.2, T0-corrected): a textbook top→anode through-going
muon.

**Root cause.** The doc-29 chord-charge guard measures support along the
STRAIGHT segment between the two extreme points (`chord_support_radius` 6 cm,
`chord_max_gap` 30 cm).  Cluster 16 is continuous — its largest
point-to-point gap is 3.1 cm — but it bows up to 10 cm away from its own
end-to-end chord, so a 126 cm stretch of the chord has its nearest cluster
charge at 6–8.3 cm and counts as "unsupported":

```
check_tgm: cluster 16 CASE-B pair (0,2) rejected: chord 170.0 cm has an unsupported run > 30.0 cm
check_tgm: cluster 16 CASE-A pair (1,2) rejected: chord 479.9 cm has an unsupported run > 30.0 cm
```

**Why it hid.** The 6 cm radius was calibrated on the pre-mainflag population
(genuine through-goers 0.0 cm longest unsupported run vs 93–583 cm for merge
artefacts).  `flag_matched_mains` (doc 30) added ~43 never-before-evaluated
bundles — including long tracks whose ordinary curvature the straight-chord
test cannot tolerate.

**Fix.** New `TaggerCheckTGM` config `chord_charge_mode` (C++ default
`"chord"` = the doc-29 straight-chord test, so earlier runs stay
reproducible).  `"path"` asks the intended question directly: the two extreme
points must be joined by a piecewise charge path through the cluster's own
points with no jump longer than `chord_max_gap` — robust to curvature by
construction.  Implementation: single-linkage components over a voxel
downsample (cell = max_gap/6, one representative point per voxel, linking
representatives closer than max_gap); reps of adjacent occupied voxels are
≤ ~17 cm apart so a contiguous track can never split, while a real gap breaks
the path at max_gap ∓ 2 cell diagonals ≈ [13, 47] cm — both edges far from
the ~3 cm (genuine) and ≥ 93 cm (artefact) populations the knob separates.
This is NOT the pre-built "relaxed"-graph connectivity (whose ~1 cm linking
stops at the cathode, the trap doc 29 documents): the linking distance here is
the 30 cm max_gap, so cathode crossers pass.  `chord_support_radius` is unused
in path mode.

| piece | file |
|---|---|
| mode knob + validation | `clus/src/TaggerCheckTGM.cxx` (`m_chord_charge_mode`) |
| `path_components` / `path_connected` | `clus/src/TaggerCheckTGM.cxx` |
| builder arg, key-suppressed | `cfg/pgrapher/common/clus.jsonnet` (`chord_charge_mode`; key only when guard ON and mode ≠ "chord") |
| SBND pass-through | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`tgm_chord_mode`) |
| entry-point TLA | `sbnd_xin/wct-pr-perevt.jsonnet` (`tgm_chord_mode`, default `'chord'`) |
| runner | `sbnd_xin/run_nusel_evt.sh` — `-chord` now passes **path**; `SBND_TGM_CHORD_MODE=chord` restores doc-29 |

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# QL products symlinked per event from work-mcp10-mainflag / work-mcp1000-mainflag
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-pathchord ./run_nusel_evt.sh data all -chord
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do   # 5-way parallel
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-pathchord \
    ./run_nusel_evt.sh data 1 -chord
done
python3 nusel_extract.py --merge work-mcp1000-pathchord/nusel_evt*/nusel-evt*.tsv \
  --out work-mcp1000-pathchord/nusel-table.tsv \
  --events-out work-mcp1000-pathchord/nusel-events.tsv
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-pathchord \
  --prev ../work-mcp10-mainflag:mcp10-mainflag \
  --prev ../work-mcp10-merge2:mcp10-merge --prev ../work-mcp10:mcp10 \
  ../work-mcp10-pathchord ../work-mcp1000-pathchord
```

## Verification

**Compiled config.** Old (HEAD) vs new jsonnet trees, same TLAs
(`/home/xqian/tmp/pathchord-cfg/`): guard off ⇒ byte-identical even with
`tgm_chord_mode=path` passed (the runner passes it unconditionally); guard on
+ `mode=chord` ⇒ byte-identical to the doc-29 config; guard on + `mode=path`
⇒ exactly one added key, `"chord_charge_mode" : "path"`.

**Knob-off runtime gate — PASS, byte-identical.** 10-event mcp10 manifest,
knob-off PR rerun with the new binary (`/home/xqian/tmp/pathgate-mcp10-190639`,
QL inputs symlinked from `work-mcp10`) vs the `work-mcp10` baseline: 10/10
`mabc-pr.zip` identical by `abtest/hash_archive.py` member-content hash, 10/10
TSVs identical.  Freshness: `libWireCellClus.so` 19:02 > source edit.
`./build/clus/wcdoctest-clus`: 518/518 assertions.

**Knob-on effect (`work-mcp10-pathchord` + `work-mcp1000-pathchord`, vs the
`*-mainflag` chord-mode runs).**  118 + 96 rows, same row sets.  Exactly
three TGM changes, all 0→1:

| bundle | geometry (T0-corrected) | why chord mode rejected it |
|---|---|---|
| evt285185 main 16 | 479.9 cm top (y=+199.6) → TPC1 anode (x=+201.2), continuous (max point gap 3.1 cm) | bows ≤10 cm off its chord ⇒ 126 cm "unsupported" run |
| evt286241 main 2 | 205.1 cm anode (x=−201.3) → top (y=+199.2), continuous (max gap 2.5 cm) | bows ≤12.4 cm off its chord |
| evt287825 main 7 | 449 cm top (y=+199.5) → bottom (y=−199.1); the 11.8k-pt component is one 30 cm-linked object | branched/kinked: points sit up to 63 cm off the end-to-end chord |

All four doc-29 artefact drops stay dropped (285185/20, 286065/11,
286197/11, 286527/21 — their fragments are separate path components), and
every previously kept TGM keeps its tag.  The only other row diffs are four
stm/fc cells flipping between 0 and −1 — torn interleaved log lines
(`nusel_extract` reports an unparsed verdict as −1; see doc 30), visibly
spliced in the raw logs, not verdict changes.

## Open

The path mode keeps the four doc-29 artefact rejections only because their
grafted fragments are far (≥ 93 cm) from the rest of the bundle charge; a
future merge artefact closer than ~30 cm to the main track would pass.  If
one appears in a hand scan, tighten per-case rather than reviving the
straight-chord test.

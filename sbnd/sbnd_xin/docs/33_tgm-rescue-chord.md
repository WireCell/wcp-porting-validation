# TGM rescued-end straight-chord check (doc 33)

Opt-in refinement of the doc-32 `component_rescue` knob:
`tgm_rescue_chord` (C++ `rescue_chord_check` in `TaggerCheckTGM`, default
OFF): a CASE-A/CASE-B pair whose boundary end was donated by a **rescued**
sub-`component_min_length` component must ALSO pass the doc-29
**straight-chord** support test (`chord_has_charge`, 6 cm support radius /
30 cm max unsupported run), even when the chord guard runs in "path" mode.

## Symptom

**evt288727 bundle grp 6 (t = 410.9 µs) main 6** (`work-mcp1000-tgmfv` only;
baseline `work-mcp1000-ctpcfix` is correct): tagged TGM, but the merged
4505-pt cluster is TWO distinct cosmics.

- Track A (3488 pts, 230 cm): enters the DOWNSTREAM wall at
  (99, +33, 500.3), runs down/upstream, ends in the interior at
  (73, −133, 344).
- Track B (896-pt drift-parallel slab + a trail of specks — classic
  prolonged-track SP fragmentation, z-span 5 cm over 120 cm): from interior
  (42, −122, 374) down to the BOTTOM wall, where its last fragment is a
  6-pt / 0.7 cm speck at (135.4, −197.95, 384.4).

The junction angle between the two is ~86° (track A low-end direction
(0.20, −0.71, −0.68) vs slab direction (0.82, −0.57, 0.03)) — not one muon.

## Root cause

`component_rescue` adds the bottom speck's extremes (it sits 10 cm from the
kept slab, well inside the 30 cm path-connect rule), giving the composite a
second outside-FV contact.  The pair (bottom speck ↔ track A's downstream
end) then passes the **path-mode** chord guard through the L-shaped detour
speck → slab → (26.3 cm gap) → track A, every jump < 30 cm.  The doc-29
straight chord would have killed it — 88% of the straight segment between
the two ends has no cluster point within 6 cm (max miss 60 cm) — but doc 31
switched the guard to path mode for good reasons (curved genuine crossers).

The discriminator: rescue's target case — a genuine track end fragmented
behind small gaps (evt286681 cluster 7, the last 2.5 cm before the top
wall) — lies ON the straight line to its own track's other end, so the
straight-chord test passes there.  A cross-track composite reaches its
second wall only via a detour.  So the straight-chord test is re-applied
ONLY to pairs with a rescued end; pairs between two length-passing
component ends keep pure path mode (doc-31 behavior).

## Fix

- `clus/src/TaggerCheckTGM.cxx`: `rescue_chord_check` knob (default false).
  `component_extreme_wcps()` optionally reports, per extreme point, whether
  it came from the rescue pass; `check_tgm()` rejects a CASE-A/CASE-B pair
  with a rescued end whose straight chord fails `chord_has_charge`.
  Log line: `... rejected: rescued end, straight chord ... cm has an
  unsupported run > ... cm`.
- `cfg/pgrapher/common/clus.jsonnet`: `rescue_chord_check=false` builder
  arg, key-suppressed.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `tgm_rescue_chord=false`
  threaded into `clus_pr` / `pr()`.
- `sbnd_xin/wct-pr-perevt.jsonnet`: TLA `tgm_rescue_chord=false`.
- `sbnd_xin/run_nusel_evt.sh`: `-rescue-chord` / `SBND_TGM_RESCUE_CHORD=1`.

## Verification

- Compiled-config proof: knob-off compile at the doc-32 operating point
  (`-chord -rescue -fvz 5` TLAs) byte-identical to the pre-change compile;
  `rescue_chord_check` key present when on.
- `wcdoctest-clus`: 518/518.
- Knob-off runtime gate (new binary, doc-32 settings, knob OFF):
  `work-mcp10-rcoff` + `work-mcp1000-rcoff` vs `work-mcp10-tgmfv` +
  `work-mcp1000-tgmfv` — **20/20 byte-identical PASS** (`mabc-pr.zip`
  member-content hashes via `abtest/hash_archive.py` + per-event TSVs).
- Knob-on 20-event run (`work-mcp10-reschord` + `work-mcp1000-reschord` vs
  the tgmfv roots): exactly ONE verdict change — **evt288727 main 6
  TGM 1→0** (sentinel: `check_tgm: cluster 6 CASE-A pair (0,12) rejected:
  rescued end, straight chord 261.4 cm has an unsupported run > 30.0 cm`).
  Doc-32 gains preserved: evt286681 main 7 stays TGM=1 (its rescued
  wall fragment lies on its own track's chord), evt286065 main 13
  unchanged.  All other TSV diffs are the known torn-log −1 cells
  (stm/fc parse artifacts, flipping both directions).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-reschord \
  ./run_nusel_evt.sh data all -chord -rescue -rescue-chord -fvz 5
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-reschord \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5
done
python3 nusel_extract.py --merge work-mcp1000-reschord/nusel_evt*/nusel-evt*.tsv \
  --out work-mcp1000-reschord/nusel-table.tsv \
  --events-out work-mcp1000-reschord/nusel-events.tsv
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-reschord \
  --prev ../work-mcp10-tgmfv:mcp10-tgmfv \
  --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix --prev ../work-mcp10-mainflag:mcp10-mainflag \
  ../work-mcp10-reschord ../work-mcp1000-reschord
```

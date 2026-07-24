# 29. Merge-aware TGM: per-component extreme points + a chord-charge guard

## Repro

```bash
cd sbnd_xin
ROOT=$PWD/work-mcp10-merge2; mkdir -p $ROOT
for d in work-mcp10/ql_evt* work-mcp10/evt*; do ln -sfn "$PWD/$d" "$ROOT/$(basename $d)"; done
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$ROOT ./run_nusel_evt.sh data -chord all
# viewer, with the earlier scans as read-only baselines (carries their
# labels/comments into tag mcp10-merge and ambers the changed-verdict rows;
# see nusel_display/README.md section 3b)
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-merge \
  --prev ../work-mcp10:mcp10 --prev ../work-mcp10-chord:mcp10-chord \
  ../work-mcp10-merge2
# knob-OFF gate run (byte-identical control):
#   SBND_WORK_ROOT=$PWD/work-mcp10-offgate2 ./run_nusel_evt.sh data all
```

## Symptom

evt285185 (run 18255 subrun 1) bundle grp 17, t = 1012.159 us, main cluster 20,
len 173.0 cm, `n_frag=2`, was tagged **TGM** with only one end at a detector
boundary.

## Root cause

`clustering_examine_bundles(use_flash_t0=true)` (sbnd `clus.jsonnet:338`)
merges every cluster in an 80 ns flash-time group into ONE `Cluster` object.
The taggers then run on that composite. The prototype does the opposite:
`check_tgm(bundle,...)` pulls `bundle->get_main_cluster()`, one PR3DCluster
(`2dtoy/src/ToyFiducial.cxx:905-928`, `pid/src/Cosmic_tagger.h:1331`); the
bundle's other clusters never enter `get_extreme_wcps()`.

`get_extreme_wcps()` is a pure bounding-extremes scan (8 candidates: +-PCA main
axis, +-x, +-y, +-z) with NO connectivity requirement -- faithful to
`data/src/PR3DCluster.cxx:3751`, which was written for one connected cluster.
And `check_tgm`'s only guard on a pair is the mid-chord FV test, which asks
whether the STRAIGHT LINE crosses the FV, never whether charge exists on it.

For cluster 20 the two "ends" were:

| `real_cluster_id` | npts | x | y | z | extent |
|---|---:|---|---|---|---:|
| 7 | 1882 | [-201.3,-124.7] | [-136.9,13.5] | [21.1,73.7] | 173.0 cm |
| 15 | **14** | [188.5,189.4] | [199.0,199.5] | [349.7,350.3] | **1.1 cm** |

```
check_tgm dbg: cluster 20 pair (0,1) ngrp 3 pe1 (189.4,199.3,350.3)
               pe2 (-201.3,-130.8,21.2) mid_inside true len 608.2/608.2 cm
```

pe1 is the 14-point speck (other TPC, top field cage, nothing else within
40 cm); pe2 is the track's anode end. Both outside the FV
(`BoxFiducial:sbnd_pr_fv` + `fv_tolerance` -2/-2.5/-3 => |x|<=199.05,
|y|<=196.81, 3.85<=z<=497.15), chord crosses the FV, t0 out of the beam
window => immediate `return true`. 96% of that 608 cm chord is empty.

## Why the obvious fixes don't work

**"Only the main cluster / `isolated == -1`"** -- measured on this sample:
**no `isolated` component ever spans the cathode** (0 of 33 cathode-spanning
main clusters); the relaxed graph stops at |x| ~ 1 cm. Genuine crosser halves
sit 2.3 / 3.0 / 3.2 / 3.5 / 3.7 / 13.0 cm from the main component. This rule
loses every cathode crosser. (`isolated` also is not in the post-PR tree at
all -- `-save-pr-tree` shows no `perblob` PC -- so it would need a
`connected_blobs()` recompute.)

**"Union components within `link_gap`, keep the primary group"** -- simulated
at 25 cm: 13 of 30 deciding pairs drop, several of them real tracks the
relaxed graph merely fragments (286329/18 is one 217 cm track split into 10
components at gaps of 37-68 cm).

## Fix

A through-going muon deposits charge ALONG its path. Sample the chord between
the two extreme points every ~1 cm; a sample is *supported* if a cluster point
lies within `chord_support_radius`. Reject the pair if any contiguous
unsupported run exceeds `chord_max_gap`.

Measured separation on the 10-event MCP2025C reco1 sample, over all 35
deciding CASE-A chords (support radius 6 cm):

| class | n | longest unsupported run |
|---|---:|---|
| genuine through-goers, incl. every cathode crosser | 23 | **0.0 cm** |
| merge artefacts | 12 | **93 - 583 cm** |

Nothing in between. Sensitivity:

| support radius | max run, genuine | min run, artefact | |
|---:|---:|---:|---|
| 2 cm | 254.3 | 101.0 | overlap |
| 3 cm | 116.6 | 99.0 | overlap |
| 4 cm | 57.6 | 97.0 | clean, 2x |
| **6 cm** | **0.0** | **93.0** | clean |
| 15 cm | 0.0 | 66.1 | clean |

Below ~4 cm the 3d point spacing itself opens gaps on real tracks. Operating
point 6 cm / 30 cm has ~3x margin both ways.  CAVEAT: that operating point was
derived from these 10 events and checked on the same 10 events -- same-sample,
not independent validation.  What argues it generalizes is that the 0-vs-93 cm
separation is STRUCTURAL (a real track has charge continuously along it; a
merge artefact has a literal vacuum gap), not a fitted threshold, and the
sensitivity scan shows it is not knife-edge. Cathode crossers pass at 0.0 cm
because the CPA gap is a few cm, inside the support radius -- verified on
285185/15 (476.8 cm chord) and 286197/10 (451.3 cm).

The guard is a `continue`, not a `return`: it can only SUPPRESS a pair, and
the loop carries on, so a sub-track that is itself through-going still tags on
its own charge-supported pair.

### Knob

```
TaggerCheckTGM:
  require_chord_charge  bool    C++ default false   <- OFF = byte-identical
  chord_support_radius  double  C++ default 6*wc.cm
  chord_max_gap         double  C++ default 30*wc.cm
```

`clus/src/TaggerCheckTGM.cxx` `chord_has_charge()`, called at the top of the
CASE-A block (covers the `flag_check` path, the `out_vec_wcps.size()==2` early
exit and the `flag_check_again` branch) and after the CASE-B perpendicularity
gate. jsonnet: `cfg/pgrapher/common/clus.jsonnet tagger_check_tgm()` with the
key-suppression idiom, threaded through
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` as `tgm_chord_charge` and
`sbnd_xin/wct-pr-perevt.jsonnet`. Runner: `run_nusel_evt.sh -chord`
(`SBND_TGM_CHORD=1`), **default OFF**.

## Verification

- `./build/clus/wcdoctest-clus`: 41 cases / 518 assertions PASS.
- Freshness: `clus/src/TaggerCheckTGM.cxx` 17:00 < `local/lib/libWireCellClus.so` 17:01.
- **Knob-OFF byte-identical gate** (the real gate, not just the config diff):
  new binary, knob off, 10 events -> `work-mcp10-offgate/`, compared to the
  pre-change `work-mcp10/`:
  - `nusel-table.tsv` identical;
  - `mabc-pr.zip` member-content hashes (`abtest/hash_archive.py`, filename
    column stripped) identical on **10/10** events;
  - every `TaggerCheckTGM: cluster N -> TGM=` verdict line identical per event.
  Caller set checked: `grep -rl 'tagger_check_tgm\|TaggerCheckTGM' cfg/` returns
  only `cfg/pgrapher/common/clus.jsonnet` and
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet` -- SBND is the only instantiator,
  so no other experiment is affected.
- **Compiled-config proof**: `wcsonnet wct-pr-perevt.jsonnet` with
  `tgm_chord_charge=false` is `cmp`-identical to the same command on the
  stashed pre-change tree. With `=true` the compiled `TaggerCheckTGM.data`
  gains exactly `"require_chord_charge": true` (radius/gap keys stay absent,
  so the C++ 6 cm / 30 cm defaults apply).
- Knob ON, 10-event rerun: `work-mcp10-chord/nusel-table.tsv` vs
  `work-mcp10/nusel-table.tsv` -- same 76 rows, **69 byte-identical**,
  7 verdict changes, TGM count 34 -> 27:

| evt | clus | len cm | n_frag | t us | OFF | ON |
|---|---:|---:|---:|---:|---|---|
| 284657 | 25 | 287.2 | 2 | -461.639 | TGM | not-tagged |
| 284657 | 26 | 195.4 | 2 | 1070.211 | TGM | not-tagged |
| **285185** | **20** | **173.0** | **2** | **1012.159** | **TGM** | **not-tagged** |
| 286065 | 11 | 245.8 | 3 | -327.045 | TGM | not-tagged |
| 286197 | 11 | 370.9 | 4 | 485.687 | TGM | not-tagged |
| 286329 | 18 | 217.4 | 2 | 431.471 | TGM | not-tagged |
| 286527 | 21 | 177.5 | 8 | -481.236 | TGM | not-tagged |

  108 pairs rejected across the 10 events, incl. the target line
  `cluster 20 CASE-A pair (0,1) rejected: chord 608.2 cm has an unsupported
  run > 30.0 cm`.
- 7 flips, not the 12 chords that fail the test: 5 clusters re-tagged on a
  different, charge-supported pair. That includes both "two real tracks"
  cases (285999/21, 286065/3) -- the desired behavior.
- No in-beam-window verdict changed: all 7 flips are out-of-beam cosmics, so
  the `n_inbeam_bundle` / event_label columns are unchanged.

**NOT bit-identical with the knob ON, by construction.** Knob OFF is
byte-identical (compiled-config proof above).

## Open / next

- Hand-scan the 7 flipped bundles before considering the knob for production
  default.  Viewer is on :5010, tag `mcp10-chord`.
  **Priority: 284657/25 and 284657/26** -- these are the two the
  pre-implementation analysis flagged as possibly-genuine (two long tracks
  ~100-130 cm apart in one bundle, i.e. maybe ONE muon broken by clustering
  rather than two cosmics).  They carry the false-negative risk.  Note
  284657/25 flipped while the structurally similar 285999/21 stayed tagged,
  which is exactly the ambiguous regime. Labels land in
  `work-mcp10-chord/nusel_labels/mcp10-chord/`; the earlier mcp10 campaign's
  labels in `work-mcp10/nusel_labels/mcp10/` are untouched.
- `TaggerCheckSTM` and `cluster_fc_check` read the same merged object and have
  the same exposure. Not changed here.
- The underlying issue -- taggers evaluating the merged composite rather than
  `get_main_cluster()` -- is not a documented divergence.
  `clus/docs/porting/porting_dictionary.md:16-33` picks the merged
  representation deliberately and notes "This merge can be undone to get back
  to representation 1"; no consumer does. Surfaced, not decided (CLAUDE.md
  Sec.5.4).


---

# PART 2 -- the chord guard alone over-corrected

## Symptom

With `require_chord_charge` alone, 284657/25 (287.2 cm) and 284657/26
(195.4 cm) went TGM -> not-tagged.  Both are REAL through-going muons.

## Root cause: the merge also destroys the extreme-point set

`get_extreme_wcps()` scans the WHOLE cluster for 8 GLOBAL extremes (+-PCA main
axis, +-x, +-y, +-z).  On a merged cluster each slot is claimed by whichever
component reaches furthest, so the other component's own wall-exit never
becomes a candidate and its legitimate within-component pair is never formed.

Extreme groups the tagger actually saw (`WCT_TGM_DEBUG`):

```
cluster 25 grp 0/3 e0 (-119.1, 199.9, 304.7) inside_fv false   <- comp 11 TOP
cluster 25 grp 1/3 e0 ( -30.0,-165.3, 500.5) inside_fv false   <- comp 10
cluster 25 grp 2/3 e0 (  -0.6, -86.5, 462.4) inside_fv true    <- comp 10
cluster 26 grp 0/6 e0 (-194.4, 199.9, 397.7) inside_fv false   <- comp 20 TOP
cluster 26 grp 1/6 e0 (-200.1,-149.8, 381.5) inside_fv false   <- comp 16
cluster 26 grp 2/6 e0 (-195.7,  69.9, 420.1) inside_fv true    <- comp 20
cluster 26 grp 3,4,5                                           <- comp 16
```

- cluster 25: comp 11 runs top wall (y=199.9) -> downstream wall (z=500.5).
  Comp 10 ALSO reaches z=500.5 and took the global max-z slot, so comp 11's
  downstream exit is not an extreme group at all.  Its only extreme is the top.
- cluster 26: comp 20 runs top wall -> anode (x=-201.3).  Comp 16 also reaches
  x=-201.3 and took the global min-x slot, hiding comp 20's anode exit.

So the chord test was right to reject the cross-component chords -- but the
correct within-component pair did not exist to be tested.

## Fix, part 2: `component_extremes`

Run the same 8-extreme scan PER connected component
(`connected_blobs(dv, pcts, "relaxed")`) and union the results.  Components
shorter than `component_min_length` (default 10 cm) contribute nothing, so a
few-point speck cannot donate two "ends".  Cross-component pairs are still
formed -- and the chord test is what rejects them -- so a cathode crosser,
whose two halves ARE separate components, keeps its tag via its
charge-supported chord across the ~3 cm CPA gap.  **The two knobs are only
correct together**; `run_nusel_evt.sh -chord` sets both.

Recovered pairs (per-component extremes):

| evt | clus | comp | pair | chord | empty run |
|---|---:|---:|---|---:|---:|
| 284657 | 25 | 11 | (-119.1,199.9,304.7) top <-> (-16.9,16.2,500.5) downstream | 287.2 cm | **0.0 cm** |
| 284657 | 26 | 20 | (-194.4,199.9,397.7) top <-> (-201.3,4.6,397.0) anode | 195.4 cm | **0.0 cm** |

### Knob (part 2)

```
TaggerCheckTGM:
  component_extremes    bool    C++ default false   <- OFF = byte-identical
  component_min_length  double  C++ default 10*wc.cm
  component_graph       string  C++ default "relaxed"
```

jsonnet arg `tgm_component_extremes`, same key-suppression idiom.

## Final result on the 10-event sample

| | TGM count |
|---|---:|
| knob OFF (pre-change) | 34 |
| `require_chord_charge` only | 27 |
| **both knobs (`-chord`)** | **30** |

4 flips remain, all verified correct -- no single component of any of them has
two FV-exiting extremes joined by a charge-supported chord:

| evt | clus | len cm | n_frag | why not TGM |
|---|---:|---:|---:|---|
| 285185 | 20 | 173.0 | 2 | track comp exits only at the anode; 14-pt speck below the 10 cm min |
| 286065 | 11 | 245.8 | 3 | comps 245.8 / 196.1 / 43.0 cm, none with a supported both-outside pair |
| 286197 | 11 | 370.9 | 4 | comps 370.9 / 29.4 cm, ditto; two 0.9 cm specks excluded |
| 286527 | 21 | 177.5 | 8 | 177.5 cm comp has ONE outside extreme; seven 1-4 cm specks excluded |

284657/25, 284657/26 and 286329/18 are back to TGM.

## Verification (final binary, instrumentation removed)

- `./build/clus/wcdoctest-clus`: 41 cases / 518 assertions PASS.
- Freshness: source 17:23 < `local/lib/libWireCellClus.so` 17:24.
- **Knob-OFF gate** `work-mcp10-offgate2` vs pre-change `work-mcp10`:
  `nusel-table.tsv` identical; `mabc-pr.zip` member-content hashes and all TGM
  verdict lines identical **10/10 events**.
- Compiled config with both knobs off: `cmp`-identical to the stashed
  pre-change tree.  Both on: `TaggerCheckTGM.data` gains exactly
  `require_chord_charge: true` and `component_extremes: true`.
- Knob-ON `work-mcp10-merge2`: 70/76 rows identical to knob-off, 6 changed
  rows = 4 TGM flips (285185/20, 286065/11, 286197/11, 286527/21) + 2
  stm/fc 0 -> -1 (286065/10 fc, 286527/14 stm), TGM 34 -> 30.
- Viewer: `:5010`, tag `mcp10-merge`, over `work-mcp10-merge2`, with
  `--prev mcp10 + mcp10-chord` baselines: earlier scan labels/comments are
  carried into the new tag and the 6 changed-verdict rows are tinted amber
  until re-scanned (README.md section 3b).

**NOT bit-identical with the knobs ON, by construction.**

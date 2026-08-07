# doc pr/40 — track (proton/pion/muon) mis-identified as electron: three fixes, SBND PRODUCTION DEFAULT ON

## Repro block

```bash
cd sbnd_xin

# Part 0 attribution probe (temporary instrumentation, since kept as
# permanent WCT_PID_WRITE_DEBUG / WCT_PID_TRACE_DEBUG diagnostics):
PR_JOBS=9 PR_EXTRA_STAGES=pr_display WCT_SHOWER_TOPO_DEBUG=1 WCT_PID_WRITE_DEBUG=1 \
  WCT_PID_TRACE_DEBUG=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-probe9b data \
  388 74544 174637 256587 267597 269774 423981 433451 489330

# G0/G1: clean-source reference at the SAME HEAD (stash the pr/40 diff,
# rebuild, run; pop, rebuild) -- see sec 17.2, why the obvious "compare
# against a recent same-day arm" shortcut failed here.
git stash push -m "pr40-g0-clean-check" -- <14 touched toolkit files>
wcbuild
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-base48 data
git stash pop
wcbuild   # M1 freshness proof; also cp build/clus/libWireCellClus.so local/lib/
          # if a later build fails at compile (M3-adjacent link-order gotcha, sec 17.2)

# G1/G2/G3/G4: knob-off vs knob-on population, all 48 nueCC48 events
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-off48 data
SBND_TRACK_PID_PERSIST_DQDX=1 SBND_SHOWER_RECLASS_DQDX_GUARD=1 SBND_SHOWER_TOPO_DQDX_GUARD=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-on48 data
python3 ../../abtest/hash_archive.py work-pr40-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40-base48, x48

# flip verification (cfg-only change, no rebuild)
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-flip-verify data 388
python3 ../../abtest/hash_archive.py work-pr40-flip-verify/pr_evt388/mabc-pr.zip work-pr40-on48/pr_evt388/mabc-pr.zip
```

## Symptom

Owner report, 9 SBND events where a hadron/muon **track** segment shows up as
an **electron** in the Bee `track_fit` display:

| # | run-event | (x,y,z) cm | cluster.seg | owner note |
|---|---|---|---|---|
| 1 | 18255-388 | (−160.6, 31.7, 425.9) | 23.020 | |
| 2 | 18259-74544 | (131.0, −173.2, 406.6) | 12.018 | |
| 3 | 18255-174637 | (−183.2, −68.5, 307.7) | 9.050 | may be a pion or muon |
| 4 | 18306-256587 | (−91.8, −16.7, 261.7) | 11.079 | |
| 5 | 18279-267597 | (139.2, −154.5, 427.6) | 5.001 | |
| 6 | 18255-269774 | (57.4, −9.2, 117.6) | 13.034 | |
| 7 | 18255-423981 | (53.9, 142.6, 176.2) | 12.013 | |
| 8 | 18255-433451 | (−173.7, 54.0, 235.6) | 4.078 | **long track** mis-ID'd |
| 9 | 18255-489330 | (−40.0, −111.9, 162.9) | 4.018 | |

Owner: *"at some point of the chain, these tracks got mis-identified as the
electron. They may not be one single place, but multiple places. Can you
investigate this thoroughly and provide a fix?"*

All 9 are in the **nueCC48** manifest (`sbnd_xin/valfast/events-nuecc48-cb0805.txt`,
hub `work-nuecc48-cb0805`, reality `data`), so the gate manifest and the study
population are the same 48 events. Segment id decoding
(`clus/src/PrDisplayDump.cxx`, `clus/src/MultiAlgBlobClustering.cxx`):
`id = cluster_id*1000 + segment->get_graph_index()`. `flag_shower` in
`calib-pr-evt<ID>.json` is `kShowerTrajectory || kShowerTopology` **only** —
not the `abs(pdg)==11` disjunct — so `pdg==11 && flag_shower==false` proves a
PID-writing site did it, not the topology/trajectory tests.

## Reproduction (measured directly on the reported segments)

| evt | seg | flag_shower | dirsign | L (cm) | median dQ/dx ÷ 56000 (MIP) |
|---|---|---|---|---|---|
| 388 | 23020 | false | +1 | 2.1 | **2.94** |
| 74544 | 12018 | false | +1 | 11.3 | **2.75** |
| 174637 | 9050 | false | +1 | 24.6 | 1.39 |
| 256587 | 11079 | **true** | +1 | 29.2 | 1.26 |
| 267597 | 5001 | false | −1 | 5.6 | **3.16** |
| 269774 | 13034 | false | +1 | 7.7 | **4.09** |
| 423981 | 12013 | false | +1 | 5.5 | **2.42** |
| 433451 | 4078 | false | +1 | 30.7 | **1.99** |
| 489330 | 4018 | **true** | +1 | 17.6 | **2.73** |

Exactly the split the owner predicted. **Family A** (7/9, no stored shower
flag): a PID/reclassification site wrote pdg 11 directly; 6 of 9 sit above
1.75× MIP, the prototype's own proton threshold. **Family B** (2/9, flag
set): `segment_is_shower_topology` fired.

Population scale on the same 48 events: 1185 main-cluster segments, 1111 pdg
11, of which 249 carry no shower flag; 129 of those (in 42 events) sit above
1.75× MIP with L > 2 cm. The class this bug belongs to is broad, which is why
the fix is gated on the full population (G3/G4), not just the 9.

## Root cause: three distinct mechanisms, one shared root idea

### Root cause 1 — the PID persistence gate discards a computed identity (F1)

`PRSegmentFunctions.cxx`, `segment_determine_dir_track`'s final store:
```cpp
if (pdg_code != 0 && ((dirsign==1 && end_n==1) || (dirsign==-1 && start_n==1))) {
    // ... persist type, mass, AND recompute the 4-momentum ...
}
```
This gates the **type+mass persistence** on the SAME condition that should
only gate the **4-momentum recompute**: whether the segment's direction
points at a topologically free end. Doc pr/7 (2026-07-30) diagnosed this
exactly, and doc pr/31 §10.9 (F8) re-confirmed it at HEAD `6206c46b` — neither
implemented a fix; no knob existed until this round.

Consequence, measured this round: a segment whose dQ/dx genuinely says
"proton" or "muon" — either via the median-dQ/dx fallback
(`medium_dQ_dx > 1.75×MIP ⇒ 2212`, `< 1.2×MIP ⇒ 13`) with `dirsign` left at 0
by that value-agnostic recovery, or via a confident template competition
whose winning direction happens to point at a non-free end — has that
identity **computed and then thrown away**. The segment exits with
`has_particle_info() == false`, indistinguishable from "PID never ran."

### Root cause 2 — wholesale track-to-electron conversion ignores the segment's own charge (F2)

Once a segment has no particle info (root cause 1) or is merely direction-weak
or short, THREE sites in `NeutrinoTrackShowerSep.cxx` convert it to electron
**unconditionally**, with no re-check of its own dQ/dx:

- `examine_all_showers`'s `flag_change_showers` loop — every non-shower
  segment in a shower-dominated cluster (`n_good_tracks==0` and a
  length-ratio family of thresholds) becomes electron.
- `improve_maps_shower_in_track_out` — TWO reclassify loops, one for
  direction-weak/untyped "out_tracks" at a shower-fed vertex, one for
  no-direction segments at the same vertex.
- `improve_maps_no_dir_tracks` Case E — a muon-typed segment (`pdg==13`)
  surrounded by enough showers gets demoted to electron based purely on
  vertex topology (daughter-shower counts, direct-length ratios).

Two of the nine owner cases (423981, 433451) were **already correctly
persisted as proton** by the PID (root cause 1 did not apply — their
direction was confident and the store gate passed) and were converted to
electron anyway by the `out_tracks` loop above, because
`seg_dir_weak(sg)` returned true: both scores (0.1749, 0.1371) sit just above
the proton `is_dir_weak` threshold (0.13 for segments ≥5 cm). This is
independently the mechanism doc pr/9 §6 F2 named and left unimplemented:
*"spare a segment with a stored 2212 and a good score from the wholesale
electron conversion."*

### Root cause 3 — the topology shower test never consults dQ/dx (F3)

`segment_is_shower_topology` (`PRSegmentFunctions.cxx`) builds a per-point
`vec_dQ_dx` array (normalized by `MIP_dQ_dx`) — and, per doc pr/31 GOTCHA 5,
**never reads it for anything but `.size()`**. The whole shower/track call is
decided by a 5-branch geometric spread test alone; the segment's own charge
never gets a vote. Family B (256587, 489330) both fired this test at 1.26×
and 2.73× MIP respectively — a hadron-density profile the test had no way to
see.

## Why it hid

- The persistence gate (root cause 1) makes the failure **silent by
  construction**: a segment with a perfectly good computed identity simply
  has no `particle_info` at all afterward, which every downstream reader
  (correctly, by its own contract) treats the same as "PID never ran."
  Pinning this needed object-level instrumentation, not a `set_pdg(11)`
  grep — the exact gotcha doc pr/9 §7 already paid for with evt 172230:
  *"pinning converters needs OBJECT-level hooks — literal `set_pdg(11)` site
  instrumentation missed it."*
- The wholesale conversion sites (root cause 2) are individually
  **prototype-faithful** (doc pr/9 §4: *"in a shower-dominated cluster, only
  a strong-direction track survives as a track"* is deliberate prototype
  design, not a port bug) — so nothing here looked wrong in isolation; only
  the population census (129 segments across 42 events) showed the class was
  broad.
- The topology test (root cause 3) has carried a dead `vec_dQ_dx` array
  since the port; `pr/31`'s audit flagged it as dead code, not as a missing
  cross-check, because nobody had yet connected it to a track-mislabeled-
  as-shower symptom.
- The un-gated `flag_print` trace that would have shown the PID's own
  conclusion (`segment_determine_dir_track`'s `Seg ... pdg score` line,
  `PRSegmentFunctions.cxx`) was hardcoded `false` with **no config or env
  path at all** — `NeutrinoTrackShowerSep.cxx:173` had
  `// if (seg->cluster() == main_cluster) flag_print = true;` commented out.
  Nobody had ever seen this line fire.

## Fix

Three default-OFF knobs, one per root cause, all threaded through
`TaggerCheckNeutrino` → `PatternAlgorithms` (the same idiom as every other
knob on this branch) and key-suppressed in `cfg/pgrapher/common/clus.jsonnet`
so the knob-off compiled config is byte-identical.

- **F1 `track_pid_persist_dqdx`** (`TrackPidOptions::track_pid_persist_dqdx`,
  `PRSegmentFunctions.cxx segment_determine_dir_track`). When `true`, persists
  type+mass whenever `pdg_code != 0` and gates only the 4-momentum recompute
  on the existing free-end test — the shape doc pr/7 described, though not
  independently re-verified against prototype source this round (the
  `prototype_base` symlink is currently broken on this machine; the toolkit's
  free-end test is used as the stand-in for the prototype's
  `get_particle_4mom(3)>0` guard). When the free-end test fails, the
  4-momentum is a rest-mass-only stub (`E=mass, p=0`), matching the
  prototype's own zero-momentum convention for an uncomputed energy.

- **F2 `shower_reclass_dqdx_guard`** (`PatternAlgorithms`,
  `NeutrinoTrackShowerSep.cxx`, three sites: `examine_all_showers`,
  `improve_maps_shower_in_track_out` ×2, `improve_maps_no_dir_tracks` Case
  E). When `true`, spares a segment from the electron conversion at these
  sites if `segment_dqdx_spares_electron_reclass(seg, m_mip_dqdx)` — the
  segment's own median dQ/dx exceeds 1.75× MIP (proton-like) or falls below
  1.2× MIP (muon-like), the SAME thresholds `segment_determine_dir_track`'s
  own fallback already trusts. A zero/absent median never spares — "no
  evidence" is not "MIP-like evidence." Guards only the *conversion action*,
  not entry to the surrounding `if`/`else if` chain, so a spared segment
  falls through to neither the conversion nor a sibling case in the same
  mutually-exclusive branch (Case E / Case F in `improve_maps_no_dir_tracks`).

- **F3 `shower_topo_dqdx_guard`** (`PatternAlgorithms`,
  `segment_is_shower_topology`, all 4 call sites in
  `NeutrinoTrackShowerSep.cxx` + `NeutrinoVertexFinder.cxx`). When `true`,
  after the existing geometric test and length-based demotions decide
  `flag_shower_topology`, overrides it back to `false` if the segment's
  median dQ/dx is decisively proton- or muon-like (same helper as F2). Runs
  unconditionally — the existing length guards only fire above their own
  length cuts (50 cm legacy, `shower_topo_demote_len`), so this is the only
  guard reachable for Family B's two segments (29.2, 17.6 cm). Does **not**
  touch `flag_dir` or the geometric test itself.

`segment_dqdx_spares_electron_reclass` (new shared helper, `PRSegmentFunctions.h/.cxx`)
is the one piece of logic F2 and F3 both call.

**This is a designed divergence from the prototype, not a port-fidelity
restoration**, for F2 and F3 — the wholesale conversion sites and the
topology test are individually prototype-faithful (see "Why it hid" above).
A new `porting_dictionary.md` section records this explicitly (§ below). F1
is a restoration of prototype behaviour per doc pr/7's reading.

### Runner env escape hatches (`run_pr_chain_batch.sh`)

```
SBND_TRACK_PID_PERSIST_DQDX=1|0
SBND_SHOWER_RECLASS_DQDX_GUARD=1|0
SBND_SHOWER_TOPO_DQDX_GUARD=1|0
```
Same tri-state contract as every other knob on this branch (unset = cfg
default, 1 = force on, 0 = force off).

## Part 0 attribution (which site wrote pdg 11, before any fix)

Object-level instrumentation (`WCT_PID_WRITE_DEBUG`, hooked at the
`particle_info()` setter plus the 6 non-const-ref reseat sites and 12 direct
`set_pdg()` mutations that bypass it) plus an un-gated PID trace
(`WCT_PID_TRACE_DEBUG`) run on the 9 events before any fix landed:

| evt | seg | writer site | PID's own conclusion at that point |
|---|---|---|---|
| 388 | 23020 | `improve_maps_no_dir_tracks` Case E | pdg=13 (muon), score=0.20 — confident, already stored |
| 74544 | 12018 | `examine_all_showers` | pdg=2212, score=100 (abstain-fallback), dirsign=0 — computed, discarded by the persistence gate |
| 174637 | 9050 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=13, score=0.073 — confident, discarded by the persistence gate (free end failed) |
| 256587 | 11079 | `segment_is_shower_topology` direct write | (topology path; PID never ran) |
| 267597 | 5001 | `examine_all_showers` | pdg=2212, score=100, dirsign=0 — same as 74544 |
| 269774 | 13034 | `improve_maps_shower_in_track_out` (no-dir segments) | pdg=2212, score=0.26, dir=−1 — confident, discarded by the persistence gate |
| 423981 | 12013 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=2212, score=0.175 — confident, **already persisted**, overwritten anyway |
| 433451 | 4078 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=2212, score=0.137 — confident, **already persisted**, overwritten anyway |
| 489330 | 4018 | `segment_is_shower_topology` direct write | (topology path; PID never ran) |

This table pins the mechanism that **originally** wrote pdg 11. It predates
F1/F2/F3 and is not the same measurement as the G2 result table below, which
was run with all three knobs together — per-case single-knob attribution
(which of F1/F2/F3 specifically rescues which case) was not isolated this
round; the shape above makes the qualitative split clear (persistence-only:
74544, 174637, 267597, 269774; already-persisted-but-reconverted: 423981,
433451; topology-only: 256587, 489330; a muon-specific demotion: 388) without
claiming exact single-knob credit.

## Gates

**G0/G1 — knob-off byte-identical: PASS, 48/48 events, 96/96 archives.**
First attempt compared against `work-pr39-verify-nuecc48` (built the same
morning, believed to be a valid same-HEAD reference) and got a 46/96 mismatch
— alarming, since the knob-off code path should be provably inert. Two
independent single-event reproductions at current HEAD (`work-pr40-repro-a`,
`work-pr40-repro-b`) both matched `work-pr40-off48` exactly
(`28a1ca60...`), proving the current build is fully deterministic and the
divergence was NOT in this round's code. A `git stash`-based same-HEAD clean
rebuild (`work-pr40-base48`, no pr/40 diff at all) also matched
`28a1ca60...` exactly. **`work-pr39-verify-nuecc48` was simply not a valid
reference for this comparison for reasons out of this doc's scope** (its own
history is not investigated here); the correct reference is
`work-pr40-base48`, and against it: `work-pr40-off48` vs `work-pr40-base48` =
**48/48 events, 96/96 archives (`mabc-pr.zip` + `pctree-pr-evt*.tar.gz`)
byte-identical** via `hash_archive.py`.

**G2 — the 9 cases: 8/9 fixed, 1/9 correctly left alone.**

| evt | seg (on) | off pdg/fs | on pdg/fs | medMIP | verdict |
|---|---|---|---|---|---|
| 388 | 23020 | 11/false | **13**/false | 2.94 | fixed |
| 74544 | 12018 | 11/false | **2212**/false | 2.75 | fixed |
| 174637 | 9050 | 11/false | **13**/false | 1.39 | fixed |
| 256587 | 11079 | 11/true | 11/true | 1.26 | **unchanged** |
| 267597 | 5001 | 11/false | **2212**/false | 3.16 | fixed |
| 269774 | 13034 | 11/false | **2212**/false | 4.09 | fixed |
| 423981 | 12013 | 11/false | **211**/false | 2.42 | fixed |
| 433451 | 4187* | 11/false | **13**/false | 1.99 | fixed |
| 489330 | 4018 | 11/**true** | **2212**/false | 2.73 | fixed |

\* 433451's segment renumbers from `4078` to `4187` (same physical track,
confirmed by position, within 0.04 cm) — the upstream graph's segment
creation order shifts slightly once earlier segments in the same cluster
reclassify, which changes later `get_graph_index()` values. Physically the
same fix, same track.

**256587 is not a bug.** Its own median dQ/dx (1.26× MIP) sits in the
deliberately conservative gap between the muon threshold (< 1.2×) and the
proton threshold (> 1.75×) — F3 correctly declines to override the topology
test's electron call because the evidence is genuinely ambiguous, not because
the guard failed. Loosening the threshold to catch this one case would sweep
a much larger, less-certain population into the guard (see G4) without a
principled reason to move the cut; left as an open residual, not touched this
round.

**G3 — population: 3/48 events move beyond the 9, zero verdict flips.**
`nusel-table.tsv` diff, off vs on, all 48 events: 3 rows differ
(219295, 268067, 52672), **only in the `stmfit` column**
(`contained` ↔ `eval`); the `label` column (nu-candidate / not-tagged
verdict) is unchanged on every one of the 48 events. None of these 3 is among
the 9 owner-reported cases.

**G4 — census: of the 25 segments in the `pdg11 & !flag_shower & med>1.75×MIP
& L>2cm` class (off-arm, all 48 events), 9 move.** Breakdown of what they
move to: 5 → proton (2212), 3 → muon (13), 1 → pion (211). This is the
breadth the flip decision rests on — the fix reaches beyond the 9 named
events by design (it is evidence-based, not event-specific), and the
population effect measured here is narrow (9/1185 main-cluster segments) and
directionally consistent (always toward the pdg the segment's own charge
argues for).

**G5 — evt 388, reported explicitly (target segment is one of the 9, so its
PID is meant to move):**

| quantity | off | on |
|---|---|---|
| main vertex (cm) | (−163.0996, 31.5755, 426.3197) | (−163.1775, 31.4456, 426.1875) |
| kine_reco_Enu (MeV) | 2810 | 2910 |
| numu_score | −0.728 | −1.51 |
| kine_pio_flag | 0 | 0 |

The vertex moves ≈0.19 cm — small, and plausible: segment 23020 sits
essentially at the vertex (0.8 cm away) and the multi-track refit that
positions the vertex can be weighted by nearby segments' particle hypotheses.
This is a genuine, non-zero move, not the "only energy moves" prediction
written into this round's plan — noted here rather than silently revised.
`kine_reco_Enu` and `numu_score` moving is the expected, predicted
consequence of the pdg change (mass and BDT features derived from it);
`kine_pio_flag` is unchanged. (Both off/on values here differ from the
historical doc pr/28 §13.12 snapshot — that snapshot predates ~10 unconditional
commits between pr/28 and this round, each capable of moving these numbers on
its own; the relevant comparison for this gate is pr/40's own off-vs-on
delta, not a re-litigation of that drift.)

**G6 — unit tests: `wcdoctest-clus` 97/97 PASS**, 1003/1003 assertions, both
before and after the SBND flip. New cases: the three F1/F2/F3 defaults added
to "TaggerCheckNeutrino switches are all OFF"; `TrackPidOptions`'s
`track_pid_persist_dqdx` default; a dedicated
`segment_dqdx_spares_electron_reclass` case (no-evidence, proton-like,
muon-like, ambiguous-middle, degenerate-scale), revert-proven (temporarily
widened the proton threshold to 99×, confirmed the test FAILED, restored,
confirmed clean).

**G7 — compiled-config proof: PASS.** Knob-off: all three keys absent from
the compiled `wct-pr-perevt.jsonnet` JSON (checked against a byte-identical
diff to the pre-pr/40 compile). Knob-on (`--tla-code
track_pid_persist_dqdx=true` etc.): all three keys appear with `true`, and
the diff against the knob-off compile is exactly those three lines — nothing
else in the compiled config moves.

## Flip — the SBND production default (owner 2026-08-06)

Owner, in scoping this round: *"Flip to SBND ON if gates pass."* All of
G1/G2/G3 (the stated flip condition) passed:

- G1 48/48 byte-identical.
- G2 fixes 8/9 (the 9th correctly, not incorrectly, untouched).
- G3 shows zero `label` (verdict) regression.

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s three TLA defaults
flipped `false → true`. Verified with a bare single-event run
(`work-pr40-flip-verify`, evt 388, no rebuild needed — cfg-only change):
hash-matches `work-pr40-on48`'s already-gated result exactly
(`33fe37c4f05050902b197d4cdf1156273037c428ff047d3e2cfb2429408948e6`).

## Scope and what is NOT claimed

- **256587 (Family B, 1.26× MIP) is not fixed** — see G2. Open residual, not
  a designed exclusion; a future round could revisit the ambiguous-band
  threshold with a larger sample if the owner wants it narrowed.
- **Per-knob attribution was not isolated.** F1/F2/F3 were gated and flipped
  together (they compound on several of the 9 cases, see Part 0). Isolating
  each knob's individual population effect would need three more 48-event
  runs (F1-only, F2-only, F3-only); not done this round.
- **Not touched, per M15/§5 rule 4** (deliberate prototype-faithful design,
  not a port bug): `examine_all_showers`'s wholesale conversion logic itself
  (doc pr/9 §4), pr/33 P14's sub-pass-1 electron-forcing in
  `shower_clustering_in_other_clusters`, and the five existing
  `porting_dictionary.md` M15 entries (Magnify coords, TrackFitting T frame,
  `calculate_boundary_metric`'s isochronous-endpoint bypass,
  `examine_vertices_3`/`get_local_extension`, `skip_trajectory_point`'s
  unbounded revert). None of these is this round's fix; F1/F2/F3 sit
  alongside them, not in place of them.
- **Round 10's `skip_revert_iso_xext_cut`** (toolkit `a5a824cf`, doc pr/28
  §17) is unrelated: none of its 13 changed events overlaps this round's 9,
  and neither this round's knobs nor round 10's touch the same code paths.
- **The `is_dir_weak` proton/muon thresholds (0.13/0.27, 0.07/0.15)** and the
  main-vertex rescue thresholds (0.09/0.06) that decided several of the 9
  cases (423981, 433451 both cross the proton threshold by 0.02–0.05) remain
  hardcoded and uncalibrated, as recorded in doc pr/8 §6/§11 and doc pr/9 §6
  — this round did not touch them, and their proximity to several of the 9
  cases' scores is worth naming for a future calibration round.
- Two prior-round items this closes: doc pr/7 §5 (F1's persistence fix,
  implemented) and doc pr/9 §6 F2 (the "protect a scored 2212" idea,
  implemented at its sibling `improve_maps_shower_in_track_out` site as well
  as `examine_all_showers`).

## `porting_dictionary.md` entry

New section, "`NeutrinoTrackShowerSep.cxx`'s wholesale track-to-electron
conversion sites and `segment_is_shower_topology` never consult the
segment's own dQ/dx — `shower_reclass_dqdx_guard`/`shower_topo_dqdx_guard`
are designed divergences, not port corrections" — full M15-record with
prototype behaviour description (both mechanisms ARE prototype-faithful),
the measured consequence, and "do NOT make either guard unconditional" (the
ambiguous-band segments, like evt 256587, are intentionally left alone).

## Verification (how the owner re-checks)

```bash
cd sbnd_xin
python3 scripts/analysis/pr40/pr40_seg_pid.py work-pr40-off9-disp work-pr40-on48
python3 ../../abtest/hash_archive.py work-pr40-off48/pr_evt<ID>/mabc-pr.zip work-pr40-base48/pr_evt<ID>/mabc-pr.zip  # x48
./build/clus/wcdoctest-clus
grep -c track_pid_persist_dqdx <(wcsonnet --tla-str input=/dev/null --tla-code 'anode_indices=[0,1]' \
  --tla-str output_dir=/tmp --tla-code run=1 --tla-code subrun=1 --tla-code event=1 --tla-str reality=data \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet)  # 1 (now SBND default true)
```

---

# Round 2 — two follow-on defects from the pr/40 Bee display: F4 zero-energy muon (FIXED, SBND ON), F6 negative-KE stub (FIXED, SBND ON), F5 electron-fathers-proton (NOT fixed, blocked)

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical, current HEAD, against work-pr40-on48
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-off48 data
python3 ../../abtest/hash_archive.py work-pr40r2-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40-on48, x48

# G2/G4: knob-on population (F5 forced on too, for measurement only -- it is
# NOT the SBND default; see Fix below)
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-on48 data
python3 scripts/analysis/pr41/pr41_check.py work-pr40r2-off48 work-pr40r2-on48

# flip verification (cfg-only change for F4/F6, no rebuild)
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-flip-verify data 174637
python3 ../../abtest/hash_archive.py work-pr40r2-flip-verify/pr_evt174637/mabc-pr.zip work-pr40r2-on48/pr_evt174637/mabc-pr.zip

./build/clus/wcdoctest-clus
```

## Symptom

Owner, reviewing the Bee display of the just-shipped pr/40 fix
(`https://www.phy.bnl.gov/twister/bee/set/52dd2243-4e7a-4d55-8831-db12c023d0d5/event/list/`):

- *"for 174637, why the muon has a 0 energy??? this is not correct."*
- *"for 256587, the reason that it is not an electron is that in the end of
  the particle, there is a proton, which is high dQ/dx. These two require
  taking a look to get them completely correct."*
- Clarifying, on 256587 specifically: *"an electron cannot change to proton.
  So the fact that we identified a proton should change it to pion instead
  of electron. Not sure why it was labeled as electron to start with. But
  this is the logic."*

A third defect (F6, negative kinetic energy) was found incidentally while
tracing the first; owner: *"Fix it in this round too."*

## F4 — the rescued muon carries zero energy

**Root cause.** `segment_determine_dir_track`'s final store
(`PRSegmentFunctions.cxx`) gated the 4-momentum computation on the same
free-end test that pr/40's F1 already stopped gating type+mass persistence
on: when the test fails, the stored 4-momentum is a rest-mass-only stub
(`E=mass, p=0`). `D4Vector[0]` is total energy E, and
`Aux::ParticleInfo`'s constructor computes `kinetic_energy = E - mass` — so
the stub reads **exactly 0 MeV**, always, for every segment pr/40's own F1
fix newly rescues into having a stored PID. Evt 174637 seg 9050 (muon,
25.8 cm) is exactly this case.

`segment_cal_4mom` (the function that should have run instead) has **no
actual free-end dependence** — its only direction coupling,
`segment_cal_dir_3vector`, already degrades gracefully to a zero 3-vector
when `dirsign()==0`. The free-end gate on it was external and unnecessary.

**Fix — `track_pid_persist_4mom`** (`TrackPidOptions`,
`PRSegmentFunctions.cxx`). When true, calls `segment_cal_4mom` unconditionally
instead of the rest-mass stub. Owner-approved approach ("Always
segment_cal_4mom") over a narrower `dirsign==0`-only condition.

**Gates.**
- G1 (knob off): **PASS, 48/48 events, 96/96 archives byte-identical**
  (`work-pr40r2-off48` vs `work-pr40-on48`, `hash_archive.py`).
- G2a (evt 174637 seg 9050): off=`(mu-, 0 MeV)`, on=`(mu-, 86 MeV)`. **PASS.**
- G4 census (zero-MeV PF nodes, all 48 events): off=1, on=**0**. **PASS.**

**Flip — SBND production default ON.** Bare single-event run (no env
override, cfg-only change) hash-matches the gated `work-pr40r2-on48` result
for evt 174637 exactly
(`c5c850e34c8b22925acac2ac29ef60f24244eafd5f6918bf396273783dba68f4`).

## F6 — `reclass_pinfo`'s never-computed path reads a negative energy

**Root cause**, found while tracing F4. `reclass_pinfo`
(`NeutrinoTrackShowerSep.cxx`) constructs a `(mass,0,0,0)` `ParticleInfo` and
then, on its non-hadron (`!had`) path, calls
`set_four_momentum(D4Vector(0,0,0,0))` — zeroing E below the mass, so
`kinetic_energy() = 0 - mass`, a **negative** number. Reachable whenever
`reclass_preserve_4mom` (SBND default **true** since pr/40) takes this
branch with a placeholder object that was never actually assigned a computed
4-momentum.

**Fix — `reclass_never_computed_ke_floor`** (`PatternAlgorithms`,
`reclass_pinfo`). When true, the non-hadron/never-computed path leaves the
constructed `(mass,0,0,0)` 4-vector in place instead of zeroing it, so
`kinetic_energy() == 0` rather than `-mass`. All 15 `reclass_pinfo` call
sites thread the new flag.

**Gates.**
- G1: same run as F4 above, same PASS (both knobs off together).
- Population: **not exercised on the 48-event nueCC48 manifest** — a
  negative-energy PF-node census found 0/0 (off/on). This flip rests on
  direct unit-test verification of `reclass_pinfo`'s `!had`/never-computed
  path (`doctest_clus_knob_defaults.cxx`), not on population evidence; stated
  explicitly rather than implying a population gate that wasn't actually
  exercised.

**Flip — SBND production default ON.** Verified by doctest only, per above;
covered by the same G1/flip-verify byte-identical checks as F4 (both knobs
flip together in the same cfg change and the same bare-run verification).

## F5 — an electron cannot father a proton (NOT FIXED — knob stays OFF)

**Root cause.** `set_default_shower_particle_info`
(`NeutrinoPatternBase.cxx`, called from `examine_direction` at stage 4) is
the single choke point where a shower-flagged segment still missing
`particle_info` defaults to electron — mirroring the prototype's
`ProtoSegment::get_particle_type()`, which unconditionally returns 11 for
any shower segment. Neither function ever looks at the graph around the
segment. Evt 256587 seg 11079 (labelled e−, 29.2 cm, median 1.26× MIP — in
the deliberately ambiguous band between pr/40's muon/proton thresholds, so
F2/F3 correctly declined it) **starts exactly at the neutrino vertex**
(d=0.00 cm) and its **far end touches a PID'd, charge-confirmed proton**
(segment 11080, 3.7 cm, median 3.72× MIP, d=0.00 cm) — a daughter an
electron cannot physically produce.

Population census (2209 electron-labelled segments, 48 events) shaped the
rule's width:

| rule | fires |
|---|---|
| any electron segment with a >1.75× MIP *neighbour* (naive) | 348 — rejected |
| + PID'd proton specifically at the far end | 15 |
| + near end **is** the neutrino vertex (graph identity) | 6 |
| **+ proton daughter independently charge-confirmed (>1.75× MIP)** | **5** ← shipped rule |

**Fix (as designed) — `shower_proton_daughter_pion`** (`PatternAlgorithms`,
new helper `segment_has_proton_daughter` in `PRSegmentFunctions.cxx`). When
true, relabels the candidate segment **pion (211)**, not proton — the
owner's explicit correction ("the fact that we identified a proton should
change it to pion instead of electron"), reasoning that the segment itself
cannot BE the proton, only cannot be an electron given what it fathered.

**Why it does not work end-to-end.** Tracing segment 11079's `particle_info`
writes across the whole pipeline (`WCT_PID_WRITE_DEBUG`) found not one
writer but a chain of four:

| # | writer | effect |
|---|---|---|
| 1 | `NeutrinoTrackShowerSep.cxx:234` (`determine_direction`, stage 3) | `pdg 0 -> 11` — unconditional electron default, no `main_vertex` in scope yet |
| 2 | `NeutrinoTrackShowerSep.cxx:929` ×2 | no-op (`11 -> 11`) |
| 3 | `NeutrinoPatternBase.cxx` (`set_default_shower_particle_info`, this fix) | `pdg 11 -> 211` — **the fix fires correctly** |
| 4 | `Shower::update_particle_type` (`PRShower.cxx:788-801`, called from 9 sites in `NeutrinoShowerClustering.cxx`) | `pdg 211 -> 11` — **reverts it** |

Writer #4 unconditionally reasserts electron on a shower's `m_start_segment`
whenever `shower_length > track_length`, with **no** PID or topology
awareness — it runs after #3 in the same pass and silently undoes it. For
256587 specifically, this revert fires: G2b measured off/on both `pdg=11,
flag_shower=true` — **unchanged**. Population-wide, the override survives
end-to-end in **only 1/2209** electron-labelled segments (evt 342199 seg
72098) — the 1-in-2209 case where writer #4 happens not to fire on the same
shower.

**Gate G2b bar, corrected.** The original plan's pass bar required
`flag_shower` to also flip `true -> false`; F5 as designed only ever touches
`particle_info`/pdg, never the shower flags (Bee and `PrDisplayDump` both
read pdg, not `flag_shower`, so this is what the owner's report is actually
about) — the bar was loosened to `pdg == 211` only. Even under the corrected
bar this still fails for 256587, because of the writer-#4 revert above, not
because of an overly strict check.

**Left OFF, not flipped.** `shower_proton_daughter_pion` remains SBND
default false. Turning it on today would not fix the reported case and would
only unpredictably touch the 1-in-2209 segments where writer #4 doesn't
collide with it — not a safe or meaningful flip. `porting_dictionary.md` and
the knob's own docstring (`NeutrinoPatternBase.h`) both record this
end-to-end-broken state explicitly, so a future reader doesn't have to
re-derive the writer chain from scratch.

**Round 3** (not started): guard `Shower::update_particle_type` itself —
most likely threading a `main_vertex`/graph-aware flag through its signature
and its 9 `NeutrinoShowerClustering.cxx` call sites (same mechanical pattern
already used for F6's 15 `reclass_pinfo` sites) so it does not clobber a
start segment already carrying pdg 211.

## Gates summary

| # | gate | bar | verdict |
|---|---|---|---|
| G0 | freshness | `.so` mtime newer than last edit | PASS |
| G1 | knob-off byte-identical | 48/48 events, 96/96 archives vs `work-pr40-on48` | **PASS** |
| G2a | evt 174637 seg 9050 energy | 0 -> nonzero | **PASS** (86 MeV) |
| G2b | evt 256587 seg 11079 pdg | 11 -> 211 | **FAIL** (writer #4 revert, see F5) |
| G4 | census | zero-MeV PF nodes 1 -> 0; proton-daughter rule population effect | PASS (F4); 1/2209 (F5, informational only) |
| G5 | unit tests | `wcdoctest-clus` | **PASS, 98/98 test cases, 1016/1016 assertions** |
| G6 | compiled-config | keys present/absent correctly; flip verified against gated arm | **PASS** |

## Flip — SBND production default (owner 2026-08-06)

Owner: *"for these fixed bugs, we should have their knobs on for SBND as
default."* Two of three are actually fixed:

- **F4 `track_pid_persist_4mom`: SBND ON.** G1/G2a/G4 all pass.
- **F6 `reclass_never_computed_ke_floor`: SBND ON.** Doctest-verified;
  covered by the same G1 byte-identical run as F4.
- **F5 `shower_proton_daughter_pion`: left OFF.** Not actually fixed end to
  end (see above) — flipping it would not deliver the reported behavior and
  CLAUDE.md's stop-and-ask rule for changing a knob's production default
  applies only to changes that are known-correct; this one demonstrably
  isn't yet.

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s F4/F6 TLA defaults
flipped `false -> true`; F5 stays `false`. Verified with a bare single-event
run (`work-pr40r2-flip-verify`, evt 174637, no rebuild needed — cfg-only
change): hash-matches the already-gated `work-pr40r2-on48` result exactly.

## Scope and what is NOT claimed

- **256587 is still not fixed.** This round diagnosed it precisely (four
  sequential writers, the last one previously unknown) but does not close
  it. Round 3 is needed.
- **F6's flip rests on unit-test evidence, not population evidence** — the
  negative-KE precondition was not observed on the nueCC48 manifest in
  either arm. Recorded explicitly rather than implied.
- **F5's helper (`segment_has_proton_daughter`) and its doctest are
  correct and unit-tested** — the defect is entirely in a downstream writer
  this round did not touch, not in the new code.
- `PrDisplayDump.h/.cxx` changes visible in the toolkit working tree during
  this round (`sbnd_xin/docs/pr/42`, the dQ/dx display panel) are **not**
  part of this round and are excluded from this round's commit — different,
  already-documented, concurrent work.

## `porting_dictionary.md` entry

Existing pr/40 section's 256587 note ("not a bug", genuinely ambiguous
median dQ/dx) superseded in part: the *topology* is not ambiguous, only the
intra-segment charge test was the wrong instrument. New section: "
`set_default_shower_particle_info`'s electron default never consults graph
topology — `shower_proton_daughter_pion` is a designed divergence, not a
port correction" — includes the full writer-chain finding and an explicit
"do not flip this knob ON" until the round-3 guard lands.

## Verification (how the owner re-checks)

```bash
cd sbnd_xin
python3 scripts/analysis/pr41/pr41_check.py work-pr40r2-off48 work-pr40r2-on48
python3 ../../abtest/hash_archive.py work-pr40r2-off48/pr_evt<ID>/mabc-pr.zip work-pr40-on48/pr_evt<ID>/mabc-pr.zip  # x48
./build/clus/wcdoctest-clus
grep -E "track_pid_persist_4mom|reclass_never_computed_ke_floor" <(wcsonnet --tla-str input=/dev/null --tla-code 'anode_indices=[0,1]' \
  --tla-str output_dir=/tmp --tla-code run=1 --tla-code subrun=1 --tla-code event=1 --tla-str reality=data \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet)  # both true (now SBND default)
```

---

# Round 3 — closes F5: guard `Shower::update_particle_type` against the proton-daughter-pion override (gate-clean, NOT flipped)

## Repro block

```bash
cd sbnd_xin

# demonstration: evt 256587 seg 11079 survives 211 end-to-end
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  WCT_PID_WRITE_DEBUG=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-dbg256587 data 256587
grep "gidx=79 " work-pr40r3-dbg256587/pr_evt256587/stdout.log   # last write: 11 -> 211, no revert

# G1: knob-off byte-identical -- true apples-to-apples needs a git-stash clean
# reference, since F4/F6 are now SBND defaults (round 2) and work-pr40-on48
# predates them
git stash push -m "pr40r3-g1-clean-check" -- clus/inc/WireCellClus/PRShower.h clus/src/NeutrinoShowerClustering.cxx clus/src/PRShower.cxx
wcbuild
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-cleanref48 data
git stash pop
wcbuild   # M1 freshness proof
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-off48 data
python3 ../../abtest/hash_archive.py work-pr40r3-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40r3-cleanref48, x48

# G2/G4: population impact
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-on48 data
python3 scripts/analysis/pr41/pr41_check.py work-pr40r3-off48 work-pr40r3-on48
diff <(sort work-pr40r3-off48/nusel-table.tsv) <(sort work-pr40r3-on48/nusel-table.tsv)  # 0 lines

./build/clus/wcdoctest-clus
```

## What this round does

Round 2 traced F5 (`shower_proton_daughter_pion`) end-to-end: the override
fires correctly at `set_default_shower_particle_info`
(`NeutrinoPatternBase.cxx`, `pdg 11 -> 211`) but is silently reverted moments
later by `Shower::update_particle_type` (`PRShower.cxx`, called from 8 sites
in `NeutrinoShowerClustering.cxx`), which unconditionally reasserts electron
on a shower's start segment whenever `shower_length > track_length`. Owner:
*"Yes, I like the first option"* — guard the reassignment itself (skip it
when a start segment is already a proton-daughter-confirmed pion) rather
than broadening the length-majority classification's own criteria (which
would touch every shower in the pipeline with a non-electron, non-proton
PID'd segment, not just this case).

**Why the guard is the narrower fix.** `update_particle_type`'s
classification (`is_shower || is_not_proton`) only exempts a segment from
`shower_length` if it is BOTH not shower-flagged AND specifically PID'd
proton (2212) — every other pdg, including a freshly-relabelled pion (211),
still counts toward `shower_length` regardless of flags. So the broader
option (teach the classification to also recognize pion/muon as track-like)
would change the shower/track majority vote for every shower with such a
segment anywhere in the pipeline; the guard only intercepts the one
reassignment this round's fix specifically produced.

## Fix

`Shower::update_particle_type` (`PRShower.h`/`.cxx`) gains two trailing
parameters, both legacy-default (`nullptr`/`false` = byte-identical):
`VertexPtr main_vertex` and `bool protect_proton_daughter_pion`. Every one
of the 8 `NeutrinoShowerClustering.cxx` call sites already had both `graph`
(reused via the Shower's own `m_full_graph` member, not a new parameter) and
`main_vertex` in scope — no stage-3/stage-4 availability problem here, unlike
F5's own choke point in round 2. Threaded as
`update_particle_type(particle_data, recomb_model, m_mip_dqdx, main_vertex,
m_shower_proton_daughter_pion, m_mip_dqdx_median)`.

Inside the reassignment block, when the guard is active it re-derives
`segment_has_proton_daughter(m_full_graph, m_start_segment, main_vertex,
proton_daughter_mip_dqdx)` and, if it fires, skips the electron reassignment
entirely (if-guarded, not an early `return` — keeps future additions to this
function from being silently skipped for a protected shower).

**MIP-scale finding, not just an implementation detail.** The function's
existing `mip_dqdx` parameter (bound to `m_mip_dqdx` = 50000/units::cm, the
flat-template amplitude used for the reassigned electron's 4-momentum) is a
DIFFERENT scale than what F5's original check used
(`m_mip_dqdx_median` = 43000/units::cm). Reusing `mip_dqdx` for the guard's
re-check would have compared the proton daughter's dQ/dx against a 16%
higher threshold (1.75×50000 vs 1.75×43000) than `set_default_shower_
particle_info` used, meaning the guard could disagree with F5's own verdict
on borderline daughters and silently fail to protect a segment F5 legitimately
relabelled. A sixth parameter, `proton_daughter_mip_dqdx`, carries
`m_mip_dqdx_median` in explicitly so the guard's re-check uses the SAME
scale as the original decision. Do not simplify this back to reusing
`mip_dqdx` — the two scales measure different things and the match is not
coincidental.

**Guard scope is broader than the reported case, by design.**
`segment_has_proton_daughter` requires graph-identity emanation from
`main_vertex`; three of the 8 call sites (`shower_clustering_in_other_
clusters`, `examine_shower_1`, `examine_showers`) process showers in OTHER
clusters, whose own main vertex is not the `main_vertex` argument passed in
— for those, `find_vertices` simply won't match and the guard correctly
returns false. This is why the population measurement (G4 below), not just
the single-event demonstration, is the real check that the guard fires only
where intended.

## Demonstration — evt 256587 seg 11079

`WCT_PID_WRITE_DEBUG` trace, all 3 round-2/3 knobs forced on:

```
gidx=79 pdg 0 -> 11    NeutrinoTrackShowerSep.cxx:234   (stage 3)
gidx=79 pdg 11 -> 11   NeutrinoTrackShowerSep.cxx:929   (no-op) x2
gidx=79 pdg 11 -> 211  NeutrinoPatternBase.cxx:176      (F5 fires)
```

No further write to `gidx=79` for the rest of the run — the
`PRShower.cxx:801` revert that fired every time in round 2 does not fire
here. Confirmed in the final `calib-pr-evt256587.json`: `particle_id: 211`,
`flag_shower: true` (flags intentionally untouched, as designed since
round 2 — see G2b bar below).

## Gates

- **G0 freshness**: `.so` mtime newer than every touched source file,
  verified before each gate below (`wcbuild`, `ls -la`).
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives.**
  `work-pr40r3-off48` (round-3 code, current SBND defaults: F4/F6 on, F5
  off) vs `work-pr40r3-cleanref48` — a git-stash clean-HEAD (`11bbfd75`)
  reference at the SAME defaults, built specifically because `work-pr40-on48`
  (round 1's reference) predates F4/F6 and is no longer a valid "off"
  baseline for this round's diff (confirmed: `work-pr40r3-off48`'s only
  divergence from `work-pr40-on48`, evt 174637's `mabc-pr.zip`, hash-matches
  round 2's `work-pr40r2-on48` exactly — fully explained by F4 being SBND
  default now, not a round-3 regression; see "GOTCHA" in memory).
- **G2a** (evt 174637, unaffected by this round): off=on=86 MeV. **PASS**,
  confirms round 3 didn't disturb F4.
- **G2b** (evt 256587 seg 11079): off `pdg=11`, on `pdg=211`. **PASS** — the
  case this whole investigation started from is now fixed.
- **G4 census**: PF nodes at 0 MeV: 0/0 (F4/F6 unaffected). Segments moved
  `11 -> 211`: **exactly 2** (evt 256587 seg 11079 — newly fixed; evt 342199
  seg 72098 — the one case that already survived in round 2's measurement).
  Matches the round-2 population prediction (5/2209 upper bound on where the
  rule *can* fire; not every one of the 5 is a shower's start segment with
  `shower_length > track_length`, so 2 actually reaching the guarded branch
  is consistent, not a discrepancy).
- **Population regression check**: `nusel-table.tsv` off vs on, all 48
  events, sorted diff: **0 lines** — zero verdict/feature-column impact
  anywhere on the manifest beyond the 2 segments' own pdg.
- **G5 unit tests**: `wcdoctest-clus` **98/98 test cases, 1016/1016
  assertions**, both before and after the guard-structure fix (`return` ->
  `if`-guarded, see Fix above).

## Flip — SBND production default (owner: "flip it default on please")

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s `shower_proton_
daughter_pion` TLA default flipped `false -> true`. Cfg-only change, no
rebuild needed. Verified with a bare single-event run
(`work-pr40r3-flip-verify`, evt 256587, no env override): hash-matches the
already-gated `work-pr40r3-on48` result exactly
(`c2095b1fd3e71a47fa65b9a24a96d1c42837b9747532c6dc49eb5717fb75bc3b`).

All three doc pr/40 round-2/3 knobs (`track_pid_persist_4mom`,
`shower_proton_daughter_pion`, `reclass_never_computed_ke_floor`) are now
SBND production defaults, alongside round 1's original three
(`track_pid_persist_dqdx`, `shower_reclass_dqdx_guard`,
`shower_topo_dqdx_guard`).

## `porting_dictionary.md` / knob docstring updates

Both the `shower_proton_daughter_pion` entry (`porting_dictionary.md`) and
its docstring (`NeutrinoPatternBase.h`) — which round 2 explicitly marked
"KNOWN BROKEN END-TO-END... do not flip" — are updated to record that the
writer-chain is now closed, gates pass, and the knob is flip-ready pending
owner request. The new `PRShower.h` docstring on `update_particle_type`
documents the guard's own default-off, byte-identical contract.

## Scope and what is NOT claimed

- **Not flipped** — see above.
- **The guard's `main_vertex` scoping to the immediate cluster is correct
  behavior, not a limitation** — a shower in a different cluster has a
  different main vertex, and the F5 relabeling rule was never meant to
  apply there. Population measurement (G4) confirms no unintended firing.
- **This closes F5 as a mechanism**, not as an open-ended guarantee that
  every possible proton-daughter-pion case in the full detector will be
  found — the underlying rule (5/2209 in this round's population framing)
  was scoped and owner-approved in round 2; this round only makes that rule
  reach the output reliably.

---

# Round 4 — F7/F8: a pion stops being a Shower, a muon cannot father two protons

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical -- git-stash clean-HEAD reference
git stash push -m "pr40r4-base48-gate" -- cfg/pgrapher/common/clus.jsonnet \
  cfg/pgrapher/experiment/sbnd/clus.jsonnet cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  clus/docs/porting/porting_dictionary.md clus/inc/WireCellClus/NeutrinoPatternBase.h \
  clus/inc/WireCellClus/PRSegmentFunctions.h clus/inc/WireCellClus/TaggerCheckNeutrino.h \
  clus/src/NeutrinoPatternBase.cxx clus/src/NeutrinoVertexFinder.cxx clus/src/PRSegmentFunctions.cxx \
  clus/src/TaggerCheckNeutrino.cxx clus/test/doctest_clus_knob_defaults.cxx
wcbuild
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-base48 data
git stash pop
wcbuild   # M1 freshness proof; cp build/clus/libWireCellClus.so local/lib/ once (link-order trap, see G0)

# G1/G2/G3/G4: knob-off vs knob-on, all 48 nueCC48 events
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-off48 data
SBND_SHOWER_PROTON_DAUGHTER_PION_DISSOLVE=1 SBND_MUON_MULTI_PROTON_PION=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-on48 data
python3 scripts/analysis/pr40r4/pr40r4_check.py work-pr40r4-off48 work-pr40r4-on48
diff <(sort work-pr40r4-off48/nusel-table.tsv) <(sort work-pr40r4-on48/nusel-table.tsv)  # 0 lines

./build/clus/wcdoctest-clus
```

## Symptom

Owner, reviewing the round 2/3 Bee display of the `shower_proton_daughter_pion` (F5) fix:

1. **evt 18306-256587**: *"the particle flow do show the pion+, but we do not see the
   proton after it in the particle flow. Also the end point of this pion+ is at an
   isolated piece. It seems that the EM shower were modified as pion, but not on the
   individual tracks, this is a problem."*
2. **evt 18255-489330**: *"in the particle flow, there is one muon --> two protons.
   This is not physical, in this case, the muon should be changed to pion."*

## Root cause 1 — F5 changes the pdg, not the shower membership (F7)

`shower_proton_daughter_pion` (round 2/3) relabels a shower-flagged segment's `particle_id`
11 -> 211 in `set_default_shower_particle_info` (`NeutrinoPatternBase.cxx`), but never
touches `SegmentFlags::kShowerTrajectory`/`kShowerTopology`. Those flags are exactly what
`shower_clustering_with_nv_in_main_cluster` (`NeutrinoShowerClustering.cxx:116-119`) tests:
`is_shower_seg = flags_any(kShowerTrajectory) || flags_any(kShowerTopology) ||
|pdg|==11`. A relabelled-but-still-flagged segment still satisfies the first two disjuncts,
so a `Shower` object is still rooted there.

Measured on evt 256587 seg 11079 (pi+, 29.1 cm): it owns a 3-segment shower whose OTHER
members are seg 11080 (the charge-confirmed proton daughter, 3.7 cm, PID'd 2212 -- the very
evidence F5 used to relabel 11079 in the first place) and seg 81153 (0.35 cm, cluster 81, a
non-main-cluster fragment). Two consequences, exactly what the owner saw:

- `fill_bee_pf_tree` (`MultiAlgBlobClustering.cxx:1173`) pre-claims every shower-owned
  segment (`used_segs = shower_segs`) before the track BFS runs, so proton 11080 never gets
  its own particle-flow node -- it is silently swallowed into the shower.
- the pi+ Bee node's displayed `end` point is the SHOWER's end (`(-95.0, -10.1, 266.6)`, a
  0.35 cm fragment absorbed from cluster 81) rather than segment 11079's own end
  (`(-90.2, -17.2, 264.3)`) -- the "isolated piece" the owner flagged.

## Root cause 2 — no proton-multiplicity veto on a muon (F8)

Segment 4019 (mu-, 65.2 cm, evt 489330) has its far (non-neutrino-vertex) endpoint at a
vertex where TWO charge-confirmed protons attach (seg 4018, 17.6 cm; seg 4044, 10.0 cm,
both PID'd 2212 with median dQ/dx > 1.75x MIP). The prototype has no PID rule that
consults a track's neighbor multiplicity at all -- there is nothing to "correct", this is
new physics-motivated logic the owner requested directly.

Segment 4019 sits behind a degree-2 kink vertex (4007) from a second muon segment (4043,
28.4 cm, running to the neutrino vertex). **Owner decision:** relabel only 4019; 4043 stays
mu-. (Asked directly during planning: propagating pion across the kink was rejected in
favor of the narrower change.)

## Fix

**F7 `shower_proton_daughter_pion_dissolve`** (`NeutrinoPatternBase.cxx`,
`set_default_shower_particle_info`): when the F5 override fires (or, on re-entry, when a
segment is already 211 from a prior pass -- `examine_direction` runs more than once, so the
OVERRIDE test widens to `pdg()==11 || (dissolve-knob-on && pdg()==211)` to make the clear
idempotent), also `unset_flags(kShowerTrajectory)` / `unset_flags(kShowerTopology)`. With
the flags gone, `is_shower_seg` at the shower-clustering seed no longer fires for that
segment, so no `Shower` is rooted there and its neighbours stay ordinary tracks.

**F8 `muon_multi_proton_pion`** (`NeutrinoPatternBase.cxx`, new
`override_muon_multi_proton_pion`, called immediately after
`set_default_shower_particle_info` in `examine_direction` -- same per-cluster `main_vertex`,
same last-word-before-shower-clustering position): for every non-shower-flagged, PID'd-muon
segment, test the new `segment_at_multi_proton_vertex` (`PRSegmentFunctions.h/.cxx`, sibling
of F5's `segment_has_proton_daughter`) at EITHER endpoint other than `main_vertex`, with
`min_protons=2`. On a fire, relabel to pion (211) via `segment_cal_4mom`. Both knobs are
config keys threaded the same way as every other pr/40 knob (default `false`, key-suppressed
when off).

## Demonstration

```
=== F7 (evt 256587) ===
  off: seg 11079 pdg=211 flag_shower=True  shower_id=11079
  off: seg 11080 (proton) pdg=2212 flag_shower=False shower_id=11079
  on:  seg 11079 pdg=211 flag_shower=False shower_id=-1
  on:  seg 11080 (proton) pdg=2212 flag_shower=False shower_id=-1
  on mc.json: node 11079 text='pi+  105 MeV' end=[-90.17, -17.17, 264.35]
  on mc.json: children of 11079: [11080]        <- proton is now a PF child
  VERDICT: proton 11080 is a direct PF child of pion 11079: True
  VERDICT: node end is segment's OWN end (not shower fragment's): True

=== F8 (evt 489330) ===
  off: seg 4019 pdg=13
  off: seg 4043 (sibling, should stay mu-) pdg=13
  on:  seg 4019 pdg=211
  on:  seg 4043 (sibling, should stay mu-) pdg=13   <- unchanged, as decided
  on mc.json: node 4019 text='pi+  188 MeV'
  on mc.json: node 4043 text='mu-  92 MeV'
```

Both owner-reported cases fixed exactly as specified.

## Population census (48-event nueCC48, off vs on)

Exactly 5 segments move, all attributable:

| evt | seg | off pdg/flag_shower | on pdg/flag_shower | why |
|---|---|---|---|---|
| 256587 | 11079 | 211 / shower | 211 / **track** | F7 fires (owner case 1) |
| 256587 | 81153 | 13 / track | **11** / track | released from the dissolved 11079-shower; no longer inherits the shower's member-pdg vote, defaults to electron on its own (non-main cluster, no `mc.json` visibility) |
| 342199 | 72098 | 211 / shower | 211 / **track** | F7 fires (same class as case 1, not owner-reported but predicted by the round-2/3 population census of 2/2209) |
| 342199 | 72096 | 13 / track | **11** / track | same release-from-dissolved-shower effect as 81153 (non-main cluster) |
| 489330 | 4019 | 13 (mu-) | **211 (pi+)** | F8 fires (owner case 2) |

The 81153/72096 secondary moves are a direct, expected consequence of dissolving a shower:
a member segment that is itself independently shower-flagged, previously absorbed and
majority-voted to the dissolved shower's pdg, is no longer absorbed and gets its own
DEFAULT-fill pdg (11, ordinary shower default) instead. Both are on non-main clusters, so
neither produces an `mc.json`/Bee visibility change -- but G3's `nusel-table.tsv` diff
confirms they cause zero downstream verdict/feature impact either.

## Gates

- **G0 freshness**: `local/lib/libWireCellClus.so` newer than every touched source file,
  verified before each gate. Hit the documented link-order stale-.so trap once (`local/lib`
  preceded `build/clus` in the linker's `-L` search order after a prior session's
  install left a stale copy there): `cp build/clus/libWireCellClus.so local/lib/` once,
  then `wcbuild` succeeded.
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives.** `work-pr40r4-off48`
  (current code, both round-4 knobs at their false default) vs `work-pr40r4-base48`
  (git-stash clean-HEAD `9d5c6a9a` reference, same 48 events) -- `abtest/hash_archive.py`
  member-content hashes of `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz`: **0 mismatches**.
- **G2a** (evt 256587 seg 11079): proton 11080 becomes a direct PF child, node end point is
  the segment's own. **PASS.**
- **G2b** (evt 489330 seg 4019/4043): 4019 -> pi+, 4043 stays mu-. **PASS.**
- **G3 population**: `nusel-table.tsv` off vs on, 48 events, sorted diff: **0 lines** -- zero
  verdict/feature-column impact anywhere on the manifest.
- **G4 census**: exactly 5 segments move (table above), all attributed; zero unexplained
  moves.
- **G5 unit tests**: `wcdoctest-clus` **99/99 test cases, 1025/1025 assertions**.
- **G6 compiled-config**: both keys absent from the compiled JSON with the knobs at their
  function defaults (`false`); both present, `true`, with `--tla-code
  shower_proton_daughter_pion_dissolve=true --tla-code muon_multi_proton_pion=true`.
  Verified directly with `wcsonnet`.

## Flip — SBND production default

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s `shower_proton_daughter_pion_dissolve`
and `muon_multi_proton_pion` TLA defaults flipped `false -> true`. Cfg-only change, no
rebuild needed. Verified with a bare 2-event run (`work-pr40r4-flip-verify`, evts 256587 +
489330, no env override): both `mabc-pr.zip` hashes match the already-gated
`work-pr40r4-on48` result exactly --
`b6cfa864...c04c58b` (256587) and `a0980d7f...9e88b4` (489330).

Both doc pr/40 round-4 knobs are now SBND production defaults, alongside all six of rounds
1-3 (`track_pid_persist_dqdx`, `shower_reclass_dqdx_guard`, `shower_topo_dqdx_guard`,
`track_pid_persist_4mom`, `shower_proton_daughter_pion`, `reclass_never_computed_ke_floor`).

## `porting_dictionary.md` entries

Two new sections (both designed divergences, no prototype anchor -- the prototype has
neither a shower-dissolution-on-relabel mechanism nor a proton-multiplicity veto on track
PID): "Relabelling a shower segment's PDG does not make it stop being a Shower" (F7) and
"A muon segment cannot terminate in a multi-proton hadronic vertex" (F8).

## Scope and what is NOT claimed

- **No propagation across the degree-2 kink** (evt 489330 seg 4043 stays mu-) -- owner's
  explicit choice, asked and answered during planning; not derived from any population
  measurement.
- **The two secondary moves (81153, 72096) are not independently owner-reviewed** -- they
  are a mechanical, fully-explained consequence of F7 dissolving a shower, verified to have
  zero downstream verdict impact (G3), but not a case the owner looked at directly.
- **This is not a general "clean up multi-membership shower artifacts" pass** -- only the
  specific mc.json/particle-flow consequences of F5's earlier relabelling are addressed.

---

# Round 5 -- three owner cases: three segment-level fixes implemented, gate-clean when
off, but **G2 FAILS on all three and one causes a new regression** (NOT flipped, negative
result)

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical -- git-stash clean-HEAD reference
git stash push -m "pr40r5-base48-gate" -- cfg/pgrapher/common/clus.jsonnet \
  cfg/pgrapher/experiment/sbnd/clus.jsonnet cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  clus/inc/WireCellClus/NeutrinoPatternBase.h clus/inc/WireCellClus/PRSegmentFunctions.h \
  clus/inc/WireCellClus/TaggerCheckNeutrino.h clus/src/NeutrinoShowerClustering.cxx \
  clus/src/NeutrinoTrackShowerSep.cxx clus/src/PRSegmentFunctions.cxx clus/src/TaggerCheckNeutrino.cxx \
  clus/test/doctest_clus_knob_defaults.cxx
wcbuild   # hit the M1 link-order stale-.so trap here; cp build/clus/libWireCellClus.so local/lib/, re-run
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r5-base48 data
git stash pop
wcbuild   # M1 freshness proof again (hit the trap a second time on restore)
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r5-off48 data
# 48/48 events, 96/96 archives (mabc-pr.zip + pctree-pr-evt<ID>.tar.gz), 0 mismatches

# the three owner cases, all three knobs forced on
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r5-cases data 84229
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r5-cases data 54341 55715

# isolation check for the 15005 regression (F11 alone)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=0 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=0 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r5/verify-f11only data 55715
# seg 15005: pdg 211 (all-off) -> pdg 11 (F11 alone). F11 owns the regression.

./build/clus/wcdoctest-clus   # 100/100 test cases, 1035/1035 assertions
```

## Owner's three cases and the attribution (Phase 1)

| run-evt | owner's point | owner's id | reading | topology |
|---|---|---|---|---|
| 18364-84229 | (73.0, 129.2, 380.0) | seg 19038 | electron -> muon? (stopping mu + Michel) | main vtx `19042`(d2) -> seg `19039` (e-, 4.9cm) -> vtx `19043`(d3) -> seg `19038` (e-, 21.2cm, owner's point) + 3.6cm shower stub `19040` |
| 18255-54341 | (134.7, 168.6, 155.5) | seg 18007 | electron -> muon? (stopping mu + Michel) | main vtx `18002`(d2) -> seg `18005` (e-, 21.3cm) -> vtx `18004`(d3) -> `18006` (1.7cm, shower) + `18007` (1.7cm, mu-, owner's point) |
| 18255-55715 | (-38.6, -37.5, 492.2) | seg 15007 | not-electron -> muon (exiting mu, pi+ parent is wrong) | main vtx `15003` -> seg `15005` (pi+, 6.1cm) -> vtx `15004`(d4) -> seg `15007` (e-, 14.7cm, `flag_shower` 1, owner's point) + `15006` (proton, 15.4cm) + `15035` (1.1cm) |

Phase 1b's per-segment `WCT_PID_WRITE_DEBUG`/`WCT_PID_TRACE_DEBUG`/`WCT_SHOWER_TOPO_DEBUG`
trace found **three independent mechanisms**, not one shared bug (owner: "Fix all three"):

- **F9** (`84229`): F1's persist-on-dQ/dx rescue (`track_pid_persist_dqdx`,
  `PRSegmentFunctions.cxx` `segment_determine_dir_track`) fires unconditionally once
  `pdg_code != 0`, including on a segment whose own free-end direction test failed --
  this is the round-1-introduced regression that turned 19038 from muon (pre-pr/40) into
  electron. Bisected and confirmed necessary+sufficient with a single-knob-off arm.
- **F10** (`54341`): `shower_clustering_connecting_to_main_vertex`
  (`NeutrinoShowerClustering.cxx`) has three skip branches (`pdg==11`,
  `pdg==2212 && dqdx`, `pdg==211 && dqdx`) but none for a long, straight track with no
  confident PID yet -- so seg 18005 gets absorbed into the shower seeded downstream of it
  before track PID ever gets a chance to run on it.
- **F11** (`55715`): `segment_is_shower_trajectory` (`PRSegmentFunctions.cxx`) has no
  straightness exemption at all (only `segment_is_shower_topology` got one, pr/40 F3) --
  seg 15007, straight and MIP-like but under the 34cm absolute-length floor, gets the
  trajectory-door shower flag regardless.

None of the three measured segments reach the existing 34cm absolute-length threshold at
`NeutrinoVertexFinder.cxx:1432-1447`; all three fixes therefore rely on the ratio branch
(`direct_length > 0.93*length`), added as a new shared helper
`segment_is_straight_long_track` (`min_length=10cm`, `min_direct=34cm`,
`straight_ratio=0.93`). Prototype cross-check (`prototype_base/pid/`): none of the three
sites have an analogous straightness/muon guard, and there is no Michel/stopping-muon rule
anywhere in the PR chain (only in the downstream STM/cosmic taggers) -- all three are
**designed divergences**, not port-fidelity fixes (M15).

## Fix (implemented, gate-clean off, three new knobs)

- **F9** `track_pid_persist_dqdx_electron_guard` -- narrows F1's rescue condition in
  `segment_determine_dir_track`: the unconditional-persist branch is skipped specifically
  when the would-be persisted pdg is 11 and the free-end direction test itself failed
  (`pdg_code == 11 && !free_end_dir`). Everything else about F1 (`track_pid_persist_dqdx`)
  is untouched, including the separate F4 `track_pid_persist_4mom` gate.
- **F10** `shower_connect_main_vertex_straight_guard` -- adds a fourth skip branch to
  `shower_clustering_connecting_to_main_vertex`'s existing three-branch guard: `if
  (segment_is_straight_long_track(sg)) continue;`.
- **F11** `shower_traj_straight_guard` -- `segment_is_shower_trajectory` gains a fourth,
  default-`false` parameter; when true, a shower-trajectory verdict is overridden to
  `false` if `segment_is_straight_long_track(seg)` also holds. Threaded only at
  `NeutrinoTrackShowerSep.cxx`'s `separate_track_shower` call site (the one Phase 1
  identified for 55715) -- the other three call sites
  (`NeutrinoVertexFinder.cxx:93,2547-2548`, `PRSegmentFunctions.cxx:2705`) keep the 3-arg
  form and default to the legacy `false`, out of this round's scope.

Plumbing follows every prior pr/40 round exactly: `NeutrinoPatternBase.h` members ->
`TaggerCheckNeutrino.{h,cxx}` (config read, `default_configuration()` round-trip,
`pattern_algos.m_... =`) -> `cfg/pgrapher/common/clus.jsonnet` (key-suppression idiom) ->
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet` threading (4 sites each, same
shape as F7/F8) -> `run_pr_chain_batch.sh` tri-state env overrides -> doctest default-false
assertions plus a 5-check hand-built-graph test case for `segment_is_straight_long_track`.

## Demonstration -- the segment-level fix works, the Bee/mc.json outcome does not (G2)

All three arms below use `work-pr40r5-cases` (all three knobs forced on):

| case | segment pdg, intended | segment pdg, measured | mc.json PF tree, measured | verdict |
|---|---|---|---|---|
| 84229 | seg 19038 -> mu- | **mu- (13), correct** | still ONE node: `id=19039 'e- 89 MeV' end=[73.0,129.2,380.0]` -- unchanged from before the fix | **G2a FAIL** |
| 54341 | seg 18005 -> mu- (stem), 18006/18007 -> e- child(ren) | seg 18005 = **proton (2212)**, 18007 = mu- (13) | split shape achieved: `id=18005 'proton 171 MeV'` -> children `18006 'e- 19 MeV'`, `18007 'mu- 11 MeV'` | **G2b FAIL** (shape right, stem label wrong) |
| 55715 | seg 15007 -> mu-, seg 15005 unchanged (pi+) | seg 15007 = **mu- (13), correct**; seg 15005 = **e- (11), was pi+ (211)** | ONE node: `id=15005 'e- 105 MeV' start=[-48.0,-38.2,482.1] end=[-34.5,-38.2,496.8]` -- covers all of 15005+15006+15007, endpoint lands at the owner's clicked point | **G2c FAIL + regression** |

## Root cause of the G2 failures

The displayed Bee/mc.json outcome is decided at the **shower seeding/absorption
boundary**, not at the segment's own pdg -- fixing a segment's pdg upstream of that
boundary does not change what a `Shower` rooted at a neighbor renders:

- `Shower::complete_structure_with_start_segment` (`PRShower.cxx:337-408`) flood-fills the
  downstream sub-tree from a shower-seeded segment with **no per-segment shower test** --
  once any neighbor is legitimately shower-flagged (84229's true 4.9cm stem `19039`;
  55715's now-declassified-from-a-different-door `15007`... see below), the whole
  downstream chain is swallowed regardless of the other members' own pdg.
- `Shower::update_particle_type` (`PRShower.cxx:788`) then sets the **shower's start
  segment** to pdg 11 whenever non-proton member length exceeds proton member length -- a
  confident `mu-` member counts as shower length, same class of bug round 3's
  `protect_proton_daughter_pion` guard fixed for a *different* trigger (a relabelled
  pion daughter), but with no equivalent guard for a muon member.
- A second, independent seeding path, `shower_clustering_with_nv_in_main_cluster`'s
  `is_shower_seg` test (`NeutrinoShowerClustering.cxx:116-119`), is what actually reaches
  84229 and 55715 -- **F10 only gates
  `shower_clustering_connecting_to_main_vertex`**, a sibling function, so it does not
  intercept this path at all. This is why 84229's fix (a different mechanism, F9) shows no
  display change, and why 55715's absorption still happens even with F11 active.

**84229** (G2a): 19038's own pdg is fixed by F9, but it is absorbed into the shower seeded
by its neighbor 19039 (a genuine 4.9cm shower-flagged stub, correctly flagged, out of
scope for any of the three fixes) via the flood-fill above -- the display cannot change
without also touching the seeding/absorption boundary itself.

**54341** (G2b): the split shape IS achieved (F10 correctly stops the 21.3cm stem from
being absorbed into the downstream shower), but once un-shielded from the shower path, seg
18005 goes through **ordinary track PID** for the first time -- not through any of the
three round-5 guards -- and that PID concludes proton (2212) from the segment's own dQ/dx,
which runs elevated (~1.55x MIP) plausibly from Bragg-peak rise near the stopping point.
Confirmed by `WCT_PID_WRITE_DEBUG`: no logged transition through pdg 11 for this segment in
the post-fix arm at all (unlike every other cluster-18 segment, which is logged going
`0->11->11` via the normal shower-default path) -- 18005 is classified 0->2212 directly by
track PID, bypassing the shower-labeling machinery the guards target. **This exposes a
pre-existing gap in ordinary track PID** (no muon-vs-proton Bragg-peak discrimination for a
short, high-dQ/dx track end) that the shower flag was incidentally masking, not a defect
introduced by F9/F10/F11.

**55715** (G2c, regression): isolated with a single-knob arm (`F11=1`, `F9=F10=0`) against
an all-off baseline -- seg 15005 reads `pdg=211` with all three knobs off, and `pdg=11`
with **F11 alone** on. F11 clearing 15007's shower-trajectory flag removes it as a shower
seed at its own door, but `is_shower_seg` (`NeutrinoShowerClustering.cxx:116-119`) then
re-seeds the shower one segment further up, at 15005 -- which the owner's Round-5 planning
answer explicitly said must stay untouched ("Only 15007 becomes mu-"). **This is a genuine
new regression against an explicit owner decision**, not merely an unmet display goal.

## Gates

- **G0 freshness**: `local/lib/libWireCellClus.so` newer than every touched source file.
  Hit the M1 link-order stale-.so trap twice (once after the initial edits, once after the
  `git stash pop` restore) -- both times fixed with `cp build/clus/libWireCellClus.so
  local/lib/`, re-run `wcbuild`.
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives, 0 mismatches.**
  `work-pr40r5-off48` (round-5 code, all three new knobs at their `false` default) vs
  `work-pr40r5-base48` (git-stash clean-HEAD `3fa0aeb3` reference, same 48 events),
  `abtest/hash_archive.py` member-content hashes.
- **G2a/b/c**: **FAIL, all three** -- see Demonstration table above.
- **G3/G4**: not run -- gated on G2 passing first (CLAUDE.md M13/§5 discipline: don't scale
  up measurement on a fix that doesn't yet do what it's meant to).
- **G5 unit tests**: `wcdoctest-clus` **PASS, 100/100 test cases, 1035/1035 assertions**
  (round-5's new `segment_is_straight_long_track` case included).
- **G6 compiled-config**: **PASS.** All three keys absent from the compiled JSON with the
  knobs at their function defaults (`false`); all three present and `true` with
  `wcsonnet --tla-code track_pid_persist_dqdx_electron_guard=true --tla-code
  shower_connect_main_vertex_straight_guard=true --tla-code
  shower_traj_straight_guard=true`.

## Flip -- NOT flipped

All three `wct-pr-perevt.jsonnet` TLA defaults stay `false`. G1/G5/G6 are clean, so the
code is safe to land (byte-identical off, no build/test regression), but **G2 fails on
every one of the three owner-reported cases**, and F11 introduces a **new regression
against an explicit owner decision** (seg 15005 in 55715). CLAUDE.md §5 rule 5 ("an A/B
gate FAILs and the diff is not explained by your intended change: report... and stop
iterating") and rule 7 ("a physics number looks wrong: report, don't tune parameters to
make it look right") both apply -- 54341's proton-vs-muon call is exactly that kind of
number. This round is landed as infrastructure (the three knobs, the shared
`segment_is_straight_long_track` helper, the attribution) but not as a fix.

## `porting_dictionary.md` entries

Three new sections recording F9/F10/F11 as designed divergences (no prototype anchor --
confirmed against `prototype_base/pid/`), each explicitly marked **"segment-level fix
only; does not reach the Bee/mc.json display outcome without also changing the shower
seeding/absorption boundary (`PRShower.cxx:337-408`, `PRShower.cxx:788`,
`NeutrinoShowerClustering.cxx:116-119`); not flipped; G2 open."**

## Scope and what is NOT claimed

- **This round does not fix any of the three owner-reported display cases.** The
  segment-level pdg is corrected in all three (19038, 15007 confidently; 18005 changes but
  not to the expected label), but the visible Bee/mc.json outcome only changed for 54341,
  and even there not to the intended label.
- **The 15005 regression is a real, confirmed defect** against the owner's explicit
  Round-5 planning answer, not a hypothetical risk -- isolated to F11 alone via a clean
  single-knob A/B, reported here rather than patched around.
- **The 54341 "proton" outcome is not diagnosed as a bug in F9/F10/F11** -- it is ordinary
  track PID running for the first time on a previously shower-shielded segment and calling
  a plausible-but-unverified proton, likely from Bragg-peak dQ/dx rise. Whether that is the
  right call physically is an open question for the owner, not something this round tunes.
- **No F12/redesign attempted in this round.** The structural fix implied by the root-cause
  analysis (teaching the seeding/absorption boundary itself about a confident non-electron
  member, in the shape of round 4's F7 shower-dissolve) is a materially bigger change than
  three surgical guards and needs its own scoping decision from the owner before
  implementation.

# Round 6 -- the boundary-level fixes: F12 absorption guard + F14 Michel-stem rescue
(F13 measured dead), all three owner cases fixed, owner-accepted residuals on the
parent stubs

## Repro block

```bash
cd sbnd_xin

# Phase-1 traces (round-5 F11-only regression writer + 54341 proton writer)
WCT_PID_WRITE_DEBUG=1 SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace-55715-f11only data 55715
# -> "PID_WRITE_DEBUG set_pdg ... clus=15 gidx=5 pdg -> 11 at NeutrinoShowerClustering.cxx:connecting_to_main_vertex"

# full-transition writer histories (WCT_PID_WRITE_DEBUG=2, new this round)
WCT_PID_WRITE_DEBUG=2 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace2-55715-off data 55715
WCT_PID_WRITE_DEBUG=2 SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 \
  SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace2-55715-on data 55715

# the three owner cases, all six pr40 r5+r6 knobs forced on (G2 smoke arm)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 \
  SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD=1 SBND_MICHEL_STEM_MUON_RESCUE=1 \
  PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r6-cases data 84229
# (same env) ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r6-cases data 54341 55715

# post-flip bare-config verification arm (NO env overrides = the production
# defaults after this round's flip; also includes nueCC48 evt 10550 for the
# hash-match against the gated on-arm)
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r6-flipverify data 84229
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r6-flipverify data 54341 55715
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-flipverify data 10550

# G1 (knobs off byte-identical) -- see the G1 note below for why pr40r5-off48
# is a valid clean-HEAD reference this round
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-off48 data
# hash_archive.py member hashes, off48 vs work-pr40r5-off48, mabc-pr.zip + pctree

# G3 (population, flip set forced on across the 48-event manifest)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 SBND_MICHEL_STEM_MUON_RESCUE=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-on48 data

./build/clus/wcdoctest-clus
```

## Phase 1 -- the round-5 mechanism map was two-thirds wrong, traces first

Round 5 hypothesized the 55715 regression came from `shower_clustering_with_nv_in_
main_cluster`'s `is_shower_seg` re-seed and round 6's plan aimed a guard at
`examine_showers`' re-root.  Both wrong -- the Phase-1 trace (existing round-1 probe)
showed the writer is `shower_clustering_connecting_to_main_vertex`'s accept-time
force-set (probe tag `connecting_to_main_vertex`):

- **55715**: with F11 on, the straightness demotion (`NeutrinoVertexFinder.cxx:1445`)
  correctly makes 15007 `mu-`; `connecting_to_main_vertex` then selects the unshielded
  6.1 cm parent 15005 as its EM candidate (UNDER `segment_is_straight_long_track`'s
  10 cm floor, so F10's straight branch cannot save it) and force-sets it to 11.
- **54341**: 18005's 2212 is a direct `0->2212` (invisible to the +-11-filtered probe;
  neighbors 18004/18006 show the examine_direction recomputes at `NeutrinoVertexFinder
  .cxx:1807`, timing the write inside examine_direction, before the F8/F14 call site).
- **84229**: `pf_shower_vertex_barrier=true` in SBND production; a segment excluded
  from every shower view either joins the track BFS or lands in the doc pr/38 orphan
  safety net (`MultiAlgBlobClustering.cxx`) -- guaranteed its own PF node either way
  (19038 dirsign=-1, so the orphan `dirsign==0` skip does not bite).

## Fix

- **F12 `shower_absorb_track_guard`** -- per-segment exclusion inside
  `Shower::complete_structure_with_start_segment` (`PRShower.cxx`), threaded to all 7
  call sites: skip (and terminate the walk at) a confidently PID'd non-electron
  (`pdg != 0 && |pdg| != 11`) that is `segment_is_straight_long_track`; do NOT claim it
  in `used_segments`; exempt long-muon pseudo-showers (`get_particle_type()==13`).
- **F13 `shower_connect_protected_pion_guard`** -- fifth skip branch in
  `connecting_to_main_vertex` (`pdg==211 && segment_has_proton_daughter`).  **Measured
  DEAD, never to be flipped** -- see below.
- **F14 `michel_stem_muon_rescue`** -- new pass `override_michel_stem_muon`
  (`NeutrinoPatternBase.cxx`, F8's call site): a `pdg==2212`, straight-long,
  main-vertex-emanating stem with >=1 shower-like sibling at its stopping vertex
  relabels `mu-` -- the toolkit's own Michel rescue rule (`NeutrinoVertexFinder.cxx`
  "a stopped proton cannot produce a Michel electron") minus its two reach limits
  (weak-direction only; stopping vertex degree exactly 2 -- 18005 has confident
  direction and degree 4).

Debug infra: `WCT_PID_WRITE_DEBUG=2` now logs EVERY pdg transition (the round-1 probe
only logged +-11 ones, blind to 211->2212).  Physics-inert, kept.

## Demonstration (work-pr40r6-cases, all six knobs on; the post-flip bare-config
arm work-pr40r6-flipverify is identical on all three -- F13 confirmed inert)

| case | owner ask | measured mc.json | verdict |
|---|---|---|---|
| 84229 | seg 19038 -> mu-, Michel child | `19039 'pi+ 38'` -> { `19038 'mu- 77'`, `19040 'e- 5'` } | **G2a PASS** (owner accepted 19039 pi+) |
| 54341 | stem mu-, Michel child(ren) | `18005 'mu- 74'` -> { `18006 'e- 19'`, `18007 'mu- 11'` } | **G2b PASS** (exact) |
| 55715 | 15007 mu-, 15005 untouched, 15006 proton kept | `15005 'proton 84'` -> { `15006 'proton 147'`, `15007 'mu- 60'`, `15035 'mu- 3'` } | **G2c PASS** (owner accepted 15005 proton) |

**The two owner-accepted residuals, fully attributed** (WCT_PID_WRITE_DEBUG=2
histories):

- 19039 (84229): OFF `0->11` (F1's poisoned undirected electron persist,
  `PRSegmentFunctions.cxx:2019`) then `11->11` reassert (`PRShower.cxx`).  ON (F9
  removes the poison): `0->13` -- its own honest muon call -- then `13->211` at the
  single-muon selection (`NeutrinoVertexFinder.cxx:1679`; 19038 wins the muon slot).
  The 4.9 cm stub and 19038 are physically the same stopping muon split at a degree-3
  vertex; a collinear-muon-fragments rule would be a new round.
- 15005 (55715): OFF `0->2212` (own charge, `NeutrinoVertexFinder.cxx:1476`) ->
  `2212->13` (Michel rescue at :1807, TRIGGERED BY 15007's WRONG e- LABEL) ->
  `13->211` (single-muon demotion, :1679).  ON: `0->2212` and nothing else -- the
  baseline pi+ was derivative of the bug being fixed; proton is the segment's own
  call.  Matches the owner's original reading ("mislabeled pion due to an attached
  proton").

**F13 negative result**: 15005 is already 2212 at candidate-selection time, so a
`pdg==211` guard cannot fire (confirmed: F11+F13 arm still shows the merged
`e- 105 MeV` node; F11+F13+F12 fixes it).  The legacy `pdg==2212` skip band
(`>1.45x MIP, nd<=3`) does not catch it either (its ratio sits in the 1.3-1.45
window); F12's exclusion makes the candidate fail EM acceptance instead, which is
sufficient.  Kept in-tree as a documented dead knob (doc pr/36 F2 precedent),
default false, excluded from the flip.

## Gates

- **G0 freshness**: `wcbuild` x2 (round-6 code; then the WCT_PID_WRITE_DEBUG=2
  widening); `local/lib/libWireCellClus.so` == `build/clus/` both times.  The M1
  link-order trap did NOT recur this round.
- **G1 knobs-off byte-identical**: **PASS, 48/48 events, 96/96 archives, 0
  mismatches** (`work-pr40r6-off48` vs `work-pr40r5-off48`,
  `abtest/hash_archive.py` member hashes, `mabc-pr.zip` + `pctree-pr-evt*.tar.gz`).
  The r5 off arm is a valid clean-HEAD reference this round because round 5
  flipped nothing: it was itself gated 48/48 vs the git-stash clean base at
  `3fa0aeb3`, and HEAD `aab6ccce` is byte-wise that same source with the same
  false defaults -- byte-identity is transitive.  (The round-3 lesson "a prior
  arm stops being a valid reference once defaults change" does not bite here
  precisely because no default changed in round 5.)
- **G2a/b/c**: **PASS, all three** -- table above.
- **G3 population**: **PASS -- nusel-table.tsv diff is 0 lines** across the 48-event
  manifest (`work-pr40r6-off48` vs `work-pr40r6-on48`, flip set F9/F10/F11/F12/F14
  forced on).  Zero verdict flips, zero score movement at table precision.
- **G4 census** (off48 vs on48 calib JSON, 48 events): 26/48 events restructure,
  fully decomposed -- 95 shower-membership-only reassignments + 126 renumbered-pair
  + 86 length/combination knock-ons (F12's absorption changes), 19 flag-only clears
  (F11's straightness population, 11-30 cm straight segments plus 2 short-segment
  knock-ons on evt 256587), **42 pdg transitions dominated by the intended
  11->13 direction (19 recoveries)**; counter-direction 13->11 is 7, of which 6 are
  sub-cm score-100 never-PID'd stubs released to their own default-fill shower
  (round 4's documented "secondary release" class) and 1 (evt 46363 seg 86123,
  8.4 cm) is a genuine re-PID with confident score 0.30.  Long-muon pseudo-shower
  count unchanged (2 off == 2 on) -- broken-muon reassembly intact, the F12
  exemption works.
- **G5 unit tests**: `wcdoctest-clus` **PASS 100/100 cases, 1042/1042 assertions**.
- **G6 compiled-config**: **PASS** -- keys absent off (compiled JSON byte-identical
  to HEAD; note the TaggerCheckNeutrino node only exists with the runner's
  `pipeline_names` TLA -- grep the pipeline-enabled compile, not the bare one);
  present exactly once each with the TLAs on.

## Flip -- FLIPPED, all five knobs SBND PRODUCTION DEFAULT ON

Owner pre-authorized during round-6 planning ("flip all if gates pass") and
separately accepted the two parent-stub residuals (19039 pi+, 15005 proton) when
asked.  G1 96/96 + G2 3/3 + G3 0-line diff met the pre-authorization bar ->
`wct-pr-perevt.jsonnet` TLA defaults flipped: `track_pid_persist_dqdx_electron_
guard`, `shower_connect_main_vertex_straight_guard`, `shower_traj_straight_guard`
(the round-5 trio, whose flip round 5 explicitly deferred), `shower_absorb_track_
guard`, `michel_stem_muon_rescue` all `false -> true`.
`shower_connect_protected_pion_guard` (F13) stays `false` -- measured dead, never
flip.

Cfg-only flip verified with bare runs (no env overrides, `work-pr40r6-flipverify`):
nueCC48 evt 10550 hash-matches the gated on-arm exactly (`mabc-pr.zip`
`139918f0...`, `pctree` `3bf518d2...`), and all three owner-case `mc.json` trees
are byte-wise the Demonstration table's.

## Scope and not-claimed

- The single-muon-per-cluster selection (`NeutrinoVertexFinder.cxx:1679` pion
  demotion) is untouched; the two owner-accepted parent-stub labels (19039 pi+,
  15005 proton) are its output on newly-honest inputs.  A collinear-muon-fragments
  rule (19039+19038 are physically one muon) is a possible future round, not
  attempted.
- `shower_clustering_connecting_to_main_vertex`'s accept-time force-set and its
  1.3-1.45x MIP proton-skip blind spot are documented (F13's negative result), not
  changed.
- The template competition's lack of an absolute quality gate (54341's original
  proton call) is compensated by F14's topology rescue, not fixed at the source.

# Doc 113 — missing particles on the 3067 data events, found without truth: the 2-D-vs-3-D imaging losses (instrument, census, causes, a default-OFF imaging knob that is not viable as built) and, on the owner's redirect, the pattern-recognition-stage losses (3-D census, blind scan, a default-OFF PR knob `nu_adopt_touching` evaluated against production)

**Status (2026-09-19): study complete; nothing flipped; two default-OFF knobs built and gated, neither recommended.**

- **Owner's question as clarified mid-round (sec 6): particles of the matched beam bundle that the PR does not
  reconstruct.** An instrument that needs no truth and no rerun was built (the bundle's 3-D image against the points
  the PR reconstructed = fitted trajectories + shower-held points). On 1392 candidate events the unheld charge is 0 at
  the median and 13.8 % at the 90th percentile, but the blind scans show that almost all of it is the imaging's
  isochronous band around long fitted tracks (0 of 28 wide near-vertex groups real) or candidates the PR never
  reconstructed (8 stubs without a vertex). The MC study's class — a prong leaving the vertex that no trajectory or
  shower holds — **is present in data but rare: 3 events confirmed by both scanners, 7 by either, of 1392 (0.2–0.5 %),
  each a 20–60 cm prong carrying 4–6 % of the candidate's charge** (mcp2k 292810, 395870, 70222; 292788, 54475, 275539,
  93105), and the census cannot separate a prong that hugs its muon (pr/96's 279955) from the band without a scan.
  The mechanism is pr/96's (charge associated to the neighbouring segment, no admission re-offers it); the fix it
  needs (pr/96 F2, an uncovered-charge admission inside the segment predicate) was **not built** this round (sec 7b).
- **Before the redirect (sec 2–5)**: the 2-D-vs-3-D imaging census found that 3.4 % of candidate events lose an
  imaged-in-2-D, absent-in-3-D segment because one plane's SP output is empty and the tiling needs all three; a
  quiet-plane 2-view tiling knob recovers 96–97 % of it at the imaging stage but costs 4–9 × and the clustering
  stage reads its cells as dead wires — **not production-viable as built**, no full-sample arm.
- **Side result (sec 7)**: `nu_adopt_touching` (adopt flashless clusters touching the candidate) — built, gated OFF,
  evaluated on all 3067 events; out of scope after the clarification; see sec 7 for its numbers.

## 0. Repro block

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin          # (/nfs/data/1/xqian/toolkit-dev -> /home/xqian/toolkit-dev)

# 1. the instrument's reconciliation gate (two whole groups = 32 events; refuses to census below its bars)
python3 scripts/analysis/d113/d113_selftest.py --sample nuecc48 --group 0 --sample mcp1k --group 0 \
        --out docs/113_figs/113_selftest.txt                                        # VERDICT PASS

# 2. the census on all 3067 events (production stage A work-<s>-d102m + stage B work-<s>-pr150s0; no rerun)
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 scripts/analysis/d113/d113_missing2d.py extract --sample $s --jobs 8 --outdir docs/113_figs/census
done                                                                                  # ~14 s per 16-event group
python3 scripts/analysis/d113/d113_summary.py                # -> 113_census_summary.txt, 113_flags.tsv, 113_exhibits.tsv
python3 - <<'EOF'   # whole-event coverage numbers (sec 2.1)
exec(open('docs/113_figs/113_event_coverage.txt').read()) if False else None
EOF

# 3. the blind scan: panels (60 exhibits + 12 controls, class hidden) and two independent Opus scanners
python3 scripts/analysis/d113/d113_render.py --ranks 0-59 --blind
python3 scripts/analysis/d113/d113_render.py --controls 12 --blind
#   scanners: docs/113_figs/113_scan_prompt.md over missing_labels/mscan-d113-{a,b}/items.txt
#   -> missing_labels/mscan-d113-{a,b}/labels.tsv ; scored by scripts/analysis/d113/d113_scan_score.py

# 4. baseline reproducibility gate (stage A of d102m with today's binary, 2 groups) and the counterfactual arms
scripts/d113_stageA_arm.sh nuecc48 rep 0; scripts/d113_stageA_arm.sh mcp1k rep 0     # -> 113_baseline_repro.txt
JOBS=6 scripts/d113_counterfactual_arms.sh     # cf0 / pdoff / isdoff / dt0 / th0 on the 54 flagged groups (TO=img)
python3 scripts/analysis/d113/d113_counterfactual.py                                  # -> 113_counterfactual.tsv

# 5. the imaging knob (quiet-plane 2-view tiling, MaskSlices quiet_mask_window / _gap; default -1 = off): sec 5
JOBS=6 ARMS="qpg64" LIST=docs/113_figs/113_flagged_groups_mcp1k.txt scripts/d113_fix_arms.sh   # imaging-only fix arm
python3 scripts/analysis/d113/d113_counterfactual.py --arms cf0,qp2v4,qpg64 --out docs/113_figs/113_counterfactual_fix.tsv
python3 scripts/analysis/d113/d113_arm_cost.py --sample mcp1k --arms qp2v4 qpg64                # -> 113_fix_cost.txt
FROM=ql TO=ql JOBS=6 scripts/d113_stageA_arm.sh mcp1k cfqpg64 <groups>                         # clustering on it
python3 scripts/analysis/d113/d113_cluster_presence.py --sample mcp1k --arm d113cfqpg64 --arm d113cfqpg64dg --groups <groups>

# 6. the PR stage (owner redirect): 3-D census, blind scan, causes
python3 scripts/analysis/d113/d113_bundle_class.py                                             # -> 113_bundle_class.txt
python3 scripts/analysis/d113/d113_pr_uncover.py --jobs 8 && python3 scripts/analysis/d113/d113_pr_summary.py   # -> 113_pr_summary.txt
python3 scripts/analysis/d113/d113_pr_render.py --ranks 0-59 --blind; python3 scripts/analysis/d113/d113_pr_render.py --controls 12 --blind
#   scanners: docs/113_figs/113_pr_scan_prompt.md over missing_labels/prscan-d113-{c,d}/items.txt
python3 scripts/analysis/d113/d113_scan_score.py --tags c d --panels docs/113_figs/pr_panels --labels-prefix prscan-d113- \
        --classes UNHELD-TRACK UNHELD-WIDE NEARBY-COMPANION NEARBY-OTHER NEARBY-UNMATCHED CONTROL --out docs/113_figs/113_pr_scan_score.txt

# 7. the PR knob nu_adopt_touching: OFF gate, ON arm on all 3067 (stage B only, ~35 min at JOBS 8), evaluation
TAG=g16off PIN=/home/xqian/toolkit-dev/local/lib SAMPLES="mcp1k mcp2k" MANIFEST=docs/pr/149_figs/manifest_gate16 NO_DL=1 JOBS=4 bash scripts/d113_arm.sh
python3 scripts/analysis/pr149/arm_identity.py work-mcp1k-pr150g16new work-mcp1k-d113g16off --allow Trun.cfg_tree Trun.op_config_sha256 Trun.provenance Trun.toolkit_git Trun.wcp_git
TAG=adopt PIN=/home/xqian/toolkit-dev/local/lib SAMPLES="nuecc48 ncpi0 mcp1k mcp2k" TLA_FILE=docs/113_figs/tla/adopt.tla JOBS=8 bash scripts/d113_arm.sh
scripts/d113_eval_pr.sh d113adopt                                                              # -> docs/113_figs/eval/d113adopt/
```

Config proofs: `docs/113_figs/113_cfg_proof.txt` (imaging), `113_pr_cfg_proof.txt` (PR). Rule files: `113_figs/113_pred.txt` (imaging knob; never reached, see sec 5) and `113_figs/113_pred_pr.txt` (PR knob; sec 7); sha256 in `113_figs/113_pred.sha256`.

## 1. The question, and why the existing instruments could not answer it

The owner's standing concern from the 3067-event development: *a major particle, plainly visible in the 2-D
wire-plane signal, is sometimes absent from the 3-D image*, so its energy never reaches the neutrino candidate.
Data has no truth, so the instrument has to work from the 2-D measurements alone.

Every earlier instrument in this tree starts from an object that already exists in 3-D: pr/96 and pr/102
(image charge no fitted trajectory covers), pr/149 `uncov` and pr/150 R2D/P4 (`T_proj_data` cells the fit does not
predict), pr/38 / pr/146 / pr/128 (fitted objects dropped from the PF / kinematics). None of them can see a blob
that was never formed. Doc 4 / doc 6 recorded the one imaging-stage incident of this kind (the 2-view dispatcher
bug, fixed 2026-04) but no census of the phenomenon existed.

## 2. The instrument (`scripts/analysis/d113/`)

### 2.1 Levels, all on disk for every event (no rerun)

| level | what | source |
|---|---|---|
| L0 | the SP charge per (channel, 4-tick slice), bad-mask channels zeroed | `work-<s>-d102m/g<K>/frames-dnn.tar.bz2` (`frame_dnnsp_<evt>.npy` 11276 ch × 3427 ticks, `summary_dnnsp`, `chanmask_bad`) |
| L1 | the activity the tiling saw: MaskSlices' thresholding (`img/src/MaskSlice.cxx:173-215`, 3.6 × the per-channel wiener summary, the uBooNE absolute fallback where the summary is 0) | re-implemented in numpy; equals the pctree `ctpc_a<A>f0p<P>` map on every slice that holds a blob |
| L2′ | the 3-D image: the stage-A live blobs' per-plane wire ranges × slices, plus the dead-fork (500-tick masked) blobs | pctree per-blob `scalar` PC (`u/v/w_wire_index_min/max` half-open, `slice_index = 4 × slicebin`, `wpid`), live and dead trees |
| L4 | the PR candidate: the bundle's `T_proj_data` cells (union of its stage clusters, pr/150 rule) | `work-<s>-pr150s0/pr_evt<ID>/tracking-pr.root`, `calib-pr-evt<ID>.json` |

A level-3 "in scope" reading was planned and dropped: every stage-A cluster carries a t0 and a matched-flash gid in
the pctree (unmatched ones get a 1000000+ rescue gid), and the Bee `img` and `clustering` layers draw the same
points to within ~100, so the PR-stage scope exclusion of `clustering_switch_scope` is not readable from these
products. Imaged-but-not-in-the-candidate charge is one class (BUNDLE) here.

### 2.2 The reconciliation gate (`d113_selftest.py`, `113_figs/113_selftest.txt`)

Every convention was derived from the data and then checked on two whole groups (nuecc48 g0 + mcp1k g0, 32 events)
before any number was read (the doc pdvd/31 trap: cross-referenced numbers are worthless until the calibration is
exact on cells with known charge):

| test | result |
|---|---|
| T1 frame → activity: the numpy port of `MaskSlices::thresholding` reproduces the ctpc activity set on the slices that hold a blob | only-mine 111 / only-ctpc 1 / value mismatches 2 of 915 149 cells (nuecc48), 70 / 4 / 2 of 819 873 (mcp1k) — ≤ 2e-4 |
| T1′ the slices that do NOT hold a blob in any branch are absent from ctpc altogether | 1834 slices / 3948 cells / 1.6e7 e (nuecc48 g0), 1464 / 3440 / 1.9e7 e (mcp1k g0) — the "no blob in the slice" population |
| T2 channel → (apa, plane, wire): TPC0 U 0–1983, V 1984–3967, W 3968–5637, TPC1 + 5638, no wrapping | 915 039 / 915 039 and 819 807 / 819 807 ctpc rows |
| T3 blob bounds: every blob's `[wmin, wmax)` holds activity on its slices in ≥ 2 planes (half-open; identical to the imaging npz bounds where kept: 1664 / 1664 blobs of evt 10550 APA0) | 90 301 / 90 301 and 64 046 / 64 046 |
| T5 `T_proj_data` rank = base[p] + apa × nwire[p] + wip (base 0 / 3968 / 7936), `time_slice` = slicebin, charge stored rounded to 1 e | 201 312 / 201 312 and 44 782 / 44 782 ctpc-present bundle cells charge-matched |
| T6 control: the candidate's own cells are covered by the stage-A blobs | 96.8 % (nuecc48) and 98.8 % (mcp1k); the PR stage re-tiles the candidate from ctpc, so its cells can exceed the stage-A blobs by a wire |

### 2.3 Census definition (`d113_missing2d.py`, `d113_summary.py`)

Per (apa, plane) the uncovered map is L0 charge above a data-driven floor (0.5 × the median active-cell charge of
the plane in that event: U 2.7k, V 3.0k, W 3.9k e per cell) on live, non-dead-blob cells that no L2′ blob covers.
8-connected components are classed by where the charge dropped out:

- **THRESH** — below the slicing threshold (L1 absent);
- **NOBLOB-SLICE** — activity present, the slice holds no blob in any tiling branch;
- **NOBLOB-INSLICE** — activity present, the slice holds blobs elsewhere, none here;
- **BUNDLE** — imaged (L2′) but not in the PR candidate (L4).

Selection = "a missing particle near the candidate": ≥ 12 cells, ≥ 2e4 e, ≥ 6 wires or slices, not a stripe, not a
hot channel (one wire holding > 80 % of the charge, or a cell > 40 × the floor), within 25 cells of a candidate cell
in the same plane, with a same-family component in ≥ 1 other plane whose slice range overlaps ≥ 50 % (a real object
has the same drift extent in every view), and `Qfrac` = charge / the candidate's own charge in that plane ≥ 3 %.

### 2.4 Results on all 3067 events (`113_figs/113_census_summary.txt`, `113_event_coverage.txt`)

Whole-event coverage first: the 3-D image covers **99.0 %** of the above-floor 2-D charge (median uncovered 0.75 %,
p90 1.9 %, max 14 % of an event's charge). The instrument is looking at the last per cent.

Near the neutrino candidate (1392 events with a PR candidate):

| class | nuecc48 | ncpi0 | mcp1k | mcp2k | all | median Qfrac |
|---|---|---|---|---|---|---|
| THRESH | 0 / 48 | 0 / 19 | 0 / 448 | 0 / 877 | 0 / 1392 | — |
| NOBLOB-SLICE | 0 / 48 | 0 / 19 | 5 / 448 | 12 / 877 | 17 / 1392 (1.2 %) | 5.2 % |
| NOBLOB-INSLICE | 1 / 48 | 0 / 19 | 13 / 448 | 20 / 877 | 34 / 1392 (2.4 %) | 4.4 % |
| any imaging class | 1 | 0 | 18 | 28 | **47 / 1392 = 3.4 %** | 5.8 % (p90 13.6 %, max 28.5 %) |
| BUNDLE (imaged, not in the candidate) | 25 | 6 | 186 | 340 | 557 / 1392 (40 %) | 17 % |

11 events carry an imaging-class loss of ≥ 10 % of the candidate's charge in one plane, 29 of ≥ 5 %. The rate is
stable against the cuts (dmax 12–40: 3.1–3.4 %; overlap 0.3–0.7: 3.3–3.4 %; min cells 8–24: 2.7–3.6 %) and moves
only with `Qfrac` (1 %: 5.2 %, 10 %: 0.8 %). **THRESH never passes**: charge below the slicing threshold is never
large and coherent enough; the zero-summary channels (H1 of the plan: 20–40 % of channels fall back to the uBooNE
absolute thresholds) carry 0 % of the flagged cells. The BUNDLE class is the cosmic / other-cluster neighbourhood
of the candidate; it is reported, not chased here (pr/38, pr/146, pr/128 own the PF end).

Calibration on the known cases: pr/96's in-image-but-uncovered-by-the-fit events (279955, 91653, 317077, 51546,
285680, 410698, 469665, 116962) and pr/146's 94392 / 392009 are **not** flagged as imaging losses (correct: they
are imaged); pr/96's two `pr54 residual-drop` events 70084 and 91697 **are** (NOBLOB-INSLICE, 8 % and 6 % of the
candidate's charge): part of those tracks was never imaged. pr/64's headline event 314507 ("a collection-plane
break") is exhibit #1 here: the break is in the image, not only in the PR graph.

### 2.5 What the flagged objects are (`113_figs/113_quiet_plane.txt`, sec 3 panels)

106 flagged components in 47 (event, apa) sets. In 28 sets the partners lie in exactly two planes and the third is
quiet — W 12, U 9, V 7 — and the components are elongated along the drift (slice extent / wire extent median 1.7,
26 % above 3). The panels show the same thing every time: a track whose projection onto one plane is (nearly)
a single wire — a track along that plane's wire direction or along the drift — so that plane's SP output carries
no activity over the stretch (a prolonged, low-amplitude signal on one wire), while the other two planes see a
clean line. `GridTiling` refuses a slice unless every plane layer is non-empty (`img/src/GridTiling.cxx:158-160`),
and the 2-view branches fill the third plane **only from the bad-channel mask** (`MaskSlice.cxx:384-419`), so the
result is no blob at all: the mechanism M1 of the plan, in the data.

## 3. The blind scan (`113_figs/113_scan_score.txt`, panels in `113_figs/panels/`, labels `missing_labels/mscan-d113-{a,b}/`)

72 items — the 60 largest imaging-class flags (one per event × class, ranked by charge) plus 12 **controls** (candidate
events with no imaging flag, the box placed on a blob-covered part of the candidate; the right answer is NO) — were
rendered as three-plane 2-D panels (charge in grey, the 3-D image as a blue outline, dead-fork blobs hatched, the
candidate's `T_proj_data` cells in green, the box in red) with the class hidden, shuffled independently, and scanned
by two Opus agents under `113_figs/113_scan_prompt.md` ("a real ionisation object inside the box, outside every blue
outline, plausibly of this interaction?").

| class (hidden) | n | scanner A YES | scanner B YES | both YES | either YES |
|---|---|---|---|---|---|
| NOBLOB-SLICE | 17 | 17 (100 %) | 11 (65 %) | 11 (65 %) | 17 (100 %) |
| NOBLOB-INSLICE | 34 | 24 (71 %) | 14 (41 %) | 14 (41 %) | 24 (71 %) |
| CONTROL | 12 | 1 (8 %) | 0 | 0 | 1 |

The scanners agree on 53 / 72; every one of the 25 both-YES items is a coherent track segment sitting in a gap of the
blue outline with the outline resuming at both ends and a matching gap in ≥ 1 other plane (nuecc48 90055: a
105-slice dark track running straight into the candidate's end; mcp1k 314507 and mcp2k 393089: 70-wire stretches of
a long muon simply dropped mid-track; three events contribute two independent YES items each). The 17 split items
(A YES, B NO) fall into two classes, both resolved by measurement rather than by a tie-break:

- **10 "streaks"** (B: "a single-wire vertical smear present in U and V, absent in W — an induction tail"). The test
  in `113_figs/113_streak_check.txt`: fit W = a·U + b·V + c on the event's own 3-view blobs (rms 0.6–1.4 wires),
  predict the W wire of the U∩V crossing, and read the W charge there. In all 6 items with an induction partner the
  predicted W wires are **live** (not in the bad mask) and carry **0.0** of the induction charge, while the U and V
  per-slice profiles are correlated at 0.78–1.00 and **rise monotonically with drift time** (2.5 k → 20–90 k e per
  slice), the Bragg signature of a stopping particle moving along the drift. An SP tail neither rises nor agrees
  across two planes. These are real: **tracks along the drift direction, which the collection plane does not see**.
- **5 "green-covered"** (B: green candidate cells drawn along the track ⇒ already in the image). The renderer draws
  green without blue exactly where the PR stage's own re-tiling from ctpc admitted cells the stage-A image lacked;
  the census records that share per component (`fq_cov4`). These items are recovered downstream (`fq_cov4` 0.6–1.0);
  they are un-imaged but not lost.

So the honest precision is scanner A's: **100 % on NOBLOB-SLICE, 71 % on NOBLOB-INSLICE, 8 % false positives on
covered charge**; the strict both-YES figures (65 % / 41 %) are a lower bound set by one scanner's reading of the
streak class. Over the 41 items judged YES by either scanner the candidate's `T_proj_data` already holds a median
**11 %** of the un-imaged charge; 32 of 41 lose more than half of it. Across the whole flagged population the PR
re-tiling recovers a median 24 % of the flagged imaging-class charge; 69 of 106 components (34 events) keep more
than half of their charge out of the candidate — that is the energy-relevant loss this doc targets.

## 4. Which imaging step lost them — counterfactual arms (`113_figs/113_counterfactual.{txt,tsv}`)

The 54 groups holding a flagged imaging-class event were re-imaged (imaging stage only, the production reco1 dump
re-used, `scripts/d113_counterfactual_arms.sh`) with one knob changed at a time, through the new `IMG_EXTRA_TLA`
channel of `run_chain_group.sh` and the `img_knobs` object threaded into `cfg/pgrapher/experiment/sbnd/img.jsonnet`
(compiled-config proofs in `113_figs/113_cfg_proof.txt`: `{}` ⇒ byte-identical, each knob lands only on its node):

| arm | change | flagged components recovered (≥ 50 % of the charge) | mean recovered charge |
|---|---|---|---|
| cf0 | none (today's binary) — separates clustering deletions | 2 / 106 | 2.6 % |
| isdoff | InSliceDeghosting dryrun (no in-slice ghost removal, no round-3 cut) | 2 / 106 | 2.7 % |
| pdoff | ProjectionDeghosting dryrun | 1 / 106 | 5.0 % |
| dt0 | zero-summary fallback threshold 3.6 × the plane median (500 / 800 / 640 e) instead of 2351 / 3347 / 2272 | 2 / 106 | 2.6 % |
| th0 | 2.5 σ slicing threshold on every channel | 5 / 106 | 11.9 % |
| **TILING** (none of the above) | | **101 / 106 = 99 % (NOBLOB-SLICE) and 97 % (NOBLOB-INSLICE) of the charge** | |

So the deghosters are innocent, the thresholds nearly so, and clustering deleted almost nothing: the flagged charge
has **no blob because one plane's SP output is empty over the stretch** and `GridTiling` will not tile a slice
without all three planes (`GridTiling.cxx:158-160`; the 2-view branches fill the third plane only from the
bad-channel mask, `MaskSlice.cxx:384-419`). Mechanism M1 of the plan, quantified: it is the whole population.
(Hypothesis H1 — the uBooNE absolute fallback thresholds on the 20–40 % zero-summary channels — is measured dead
for this population: `dt0` recovers nothing and the flagged cells sit on channels with a summary.)

## 5. The fix: quiet-plane 2-view tiling (`MaskSlices quiet_mask_window / quiet_mask_gap`, default OFF)

**Design.** The 2-view tiling branches (`[U,V]+masked W`, `[V,W]+masked U`, `[U,W]+masked V`) fill their masked
plane only from the bad-channel mask. The knob adds a second source: a masked-plane channel is also marked masked
(`{masked_charge, masked_error}` = `{0, 1e12}`, exactly like a bad channel) in every slice where it has **no active
tick within ± `quiet_mask_window` slices** (the same thresholding rule the active planes use, re-applied to that
plane's own trace), so the two live planes can tile where the third plane is silent. `quiet_mask_gap ≥ 0`
restricts this to slices within ± gap of an active slice of the same channel — the prolonged-signal gap of a
track that plane does see at its ends — leaving channels silent everywhere live and empty. Both C++ defaults are
−1 = off; the legacy path is untouched (`img/inc/WireCellImg/MaskSlice.h`, `img/src/MaskSlice.cxx`; jsonnet
`img_knobs.ms_quiet_mask_window / ms_quiet_mask_gap`, key-suppressed, landing only on the three 2-view branches of
the active fork — `113_figs/113_cfg_proof.txt`).

**Gates.** `wcdoctest-img` 2/2 cases, 15/15 assertions. Freshness: `local/lib/libWireCellImg.so` 10:59 / 11:34
after the 09:41 / 11:32 edits. Compiled config with the knob absent byte-identical (`cmp`), `prod_cfg_gate.py`
21/21 after every edit. Runtime OFF gate (`113_figs/113_off_gate.txt`): stage A re-run with the new binary and no
knob on nuecc48 g0 + mcp1k g0 = 32 events — 128 / 128 imaging archives `numpy.array_equal`, pctrees 0 changed
datapaths, twice (after each C++ edit). The PDHD/PDVD imaging manifest could not be re-run (its SP frames are no
longer on disk after the cleanup rounds and regenerating them would write into existing `work/` dirs, M13;
`113_figs/113_pdgate.txt`) — what stands for them is the guarded code path, their unchanged compiled configs
(the 21-artifact gate) and the SBND runtime gate on the same class; the uBooNE chain has no MaskSlices node at all.

**Knob-ON smoke and recovery** (imaging-only arms on the 20 mcp1k flagged groups, `113_figs/113_counterfactual_fix.tsv`,
`113_fix_cost.txt`): the log line `quiet-plane masking: window 4 ... 1 663 069 cells` / `gap 64 ... 205 923 cells`
fires on every 2-view branch; both variants recover the flagged charge — **97 % (window 4) and 96 % (gap 64) of the
flagged components' charge is covered by blobs, 34 / 34 components above one half** (production: 2.6 %). The
mechanism is confirmed: give the tiling a way past the silent plane and the particles come back.

**Cost, at the imaging stage** (vs the cf0 baseline on the same groups):

| variant | quiet-masked cells per slicer call | surviving imaging blobs | imaging wall | peak RSS |
|---|---|---|---|---|
| window 4 (qp2v4) | 1.66 M | × 1.38 (max 1.46) | × 8.6 (104 → 882 s per 16-event group) | × 5.6 (0.96 → 5.4 GB) |
| gap 64 (qpg64) | 0.21 M | × 1.99 (max 2.18) | × 4.3 (105 → 424 s) | × 1.9 |

The extra blobs are the 2-view crossings of U/V activity that no 3-view blob owns (noise and induction-only
activity where the third plane is silent); the in-slice deghosting removes the crossings whose wires belong to
3-view blobs but has nothing to hold against these.

**Downstream: the clustering stage reads the quiet-masked cells as dead wires and deletes real clusters.** The
gap-64 imaging was carried through the production clustering / Q-L stage (`FROM=ql`) on the 20 groups and the
instrument re-run on the result (`stageA d113cfqpg64`): the per-event pctree has **0.71 ×** the production blobs
(2766 vs 4455 median), 9 vs 17 clusters, and the whole-event uncovered charge goes from 0.70 % to **63 %**; on evt
487633 six of the fifteen production clusters (1695, 4883, 929, 1079, 2268, 3377 points) are gone entirely while
the `dead_winds` rows rise from 94 to 5652. The quiet cells carry the bad-channel uncertainty (1e12), so
`PointTreeBuilding::add_dead_winds` (`dead_threshold` 1e10) turns every quiet (wire, slice) into a dead wire,
and the clustering stage then loses the clusters that sit on "dead" wires. It is **not** `clustering_deghost`:
re-running the Q/L stage on the same imaging with the deghost unable to delete anything (`dg_length_cut = 1e-6`,
a new key threaded for this test, `113_figs/113_cluster_presence.txt`) reproduces the loss exactly — on 320 events
the pctree keeps 0.71 × the blobs, 9 of 17 clusters, 41 % of the production points, and **2107 of 2852 production
clusters of ≥ 200 points are lost** either way. The loss sits between the imaging output and the clustering
graph (the blob sampling / live-dead handling of blobs whose wires are now "dead"), and it is where a second
round would have to start: give the quiet cells an uncertainty class of their own (above ChargeSolving's 1e9 and
ProjectionDeghosting's `uncer_cut`, below PointTreeBuilding's `dead_threshold` 1e10) so the imaging treats them as
unmeasured while the clustering treats them as live. **Verdict for this doc: the knob is built, gated OFF and
proven to recover the missing particles at the imaging stage, but it is not production-viable as built (4–9 ×
imaging cost, 2 × surviving imaging blobs, and a clustering-stage cluster loss), so no full-sample arm was run and
nothing is recommended for flipping.** The owner then redirected the study to the pattern-recognition stage
(sec 6 onward): the goal is the particles that are in the 3-D image and missing from the neutrino candidate, not
the imaging losses above.

## 6. The pattern-recognition stage: particles inside the matched beam bundle that the PR does not reconstruct

Owner redirect (2026-09-19, mid-round): *"only the final neutrino pattern recognition stage issues in terms of
missing particles"*, then *"missing particles from the matched beam bundle, not the unmatched clusters ... in the
MC study we sometimes miss a main particle coming out of the neutrino vertex"*. The stage-A image is taken as given;
the question is which charge of the candidate's own bundle the PR fails to reconstruct as a particle — doc 107 sec
5.8's "missing prong / EM misreco" class, pr/96's "missing vertex track".

### 6.1 Instrument (`d113_pr_uncover.py`, `d113_pr_summary.py`, `113_figs/pr_census/`)

3-D, per candidate event, from the production PR products: the bundle's image points (the PR-stage pctree
`pctree-pr-evt<ID>.tar.gz`: the stage-A 3-D cloud with the PR job's cluster ids, `x_t0cor`, cluster ids = the
`T_proj_data` union of pr/150) against the points the PR **reconstructed** = the fitted trajectory points
(`track_fit-global`) plus the points a **shower** holds (`shower_track-global` with q = 15000). Points a track merely
*associated* (q = 0) do not count as held: a track's energy is a range / dQ/dx energy of its fitted trajectory, so a
prong or a shower swept into a neighbouring track's association is lost from `kine_reco_Enu`. (A first, lenient
version counted every associated point as held and found nothing — `113_figs/113_pr_summary_asc.txt` — which is
itself a fact: the PR associates essentially every bundle point to something.) Frame check: held points sit on the
image at 0.5 cm (half-pitch lattice), 100 % within 1 cm. The PR Bee `clustering` layer cannot serve as the image
(it holds only the re-sampled held points). Per event the unheld points (> 3 cm from every held point) are grouped
(single-link 2 cm); per group: points, charge, charge / candidate, PCA extent and transverse rms, distance to the
main vertex, the share of points within 3 cm of a *track-associated* point (`f_absorbed`), and the distance of the
group's points to the nearest fitted trajectory (`dfit_p90`).

### 6.2 Results (`113_figs/113_pr_summary.txt`, `113_pr_sensitivity.txt`, `113_unheld_causes2.txt`, `113_absorbing_segments.txt`)

Unheld charge of the candidate under this definition: median 0, **p90 13.8 %**, above 10 % in 173 of 1392
candidate events, above 25 % in 84. Near the vertex (≤ 15 cm, ≥ 40 points, ≥ 3 % of the candidate's charge):
**143 events (10.3 %)** carry an unheld group — 20 track-like (transverse rms ≤ 0.8 cm) and 125 wide. Three
populations make it up, and only the last is the owner's class:

1. **Candidates the PR did not reconstruct at all** (8 events, 0.6 %): main cluster 16–52 cm, no vertex found,
   `Enu 0`, `nue_score −15`; their whole image is "unheld" (the p90 distance to any fit is 17–370 cm). A
   candidate-selection question (`nu_per_bundle` picked a stub), not a missing particle; listed in
   `113_unheld_causes.txt`.
2. **The image band of long fitted tracks** — the dominant population. 165 of the 172 wide groups and 19 of the
   22 track-like ones are ≥ 50 % *absorbed* into a track's association, and the absorbing segment is the
   candidate's long muon (46–423 cm, `113_absorbing_segments.txt`). The panels (e.g. `pr_panels/mcp2k-283833-p001`,
   `mcp2k-395079-p003`) show what these are: the 3-D image of a track that runs near-parallel to the wire planes is a
   band 10–30 cm wide in y–z at fixed x (the isochronous ambiguity of the imaging), the fitted line runs down its
   middle, and every point of the band is associated to the muon. The muon's energy is a range energy; the band is
   not a particle and not an energy loss. A halo veto (the group must reach ≥ 12 cm from every fitted trajectory
   at its 90th percentile) removes most of it: 143 → 30 events; ≥ 8 cm: 62; ≥ 20 cm: 10.
3. **Prongs and clumps near the vertex that no trajectory or shower holds** — the owner's class. Its size cannot be
   read off the census alone, because the discriminator that kills class 2 also kills a prong that hugs its muon:
   pr/96's owner-confirmed missing prong (mcp2k 279955, 298 unheld points, 20 % of the candidate's charge, 3 cm from
   the vertex, within 12 cm of the muon for its whole length) is flagged only with the halo veto **off**
   (`113_pr_sensitivity.txt`: every combination of min points 20/40 and rms 0.8/1.5 flags 143–146 events with the
   veto off and none of them keeps 279955 once the veto is ≥ 8 cm). So the class is bounded from above by the
   143-event superset and from below by the scanned subsets of sec 6.3.

The PF / kinematics stage excludes nothing on this sample (`kine_energy_excluded` = 0 on all 1435 dumps). The
`create_steiner_tree ... only 1 steiner terminal` warning fires in 90 % of candidate events, on 6205 associated
clusters — every one of them ≤ 50 points (`113_nosteiner_census.txt`): fragments, not particles.

### 6.3 Blind scans (`113_figs/113_pr_scan_score.txt`, `113_pr_scan_score_track.txt`; panels `pr_panels/`, `pr_panels_track/`; labels `missing_labels/prscan-d113-{c,d,e,f,g,h}/`)

Three-projection panels (candidate image grey, held points green, the item red, other clusters light blue, vertex
star), class hidden, two independent Opus scanners per set under `113_figs/113_pr_scan_prompt.md`.

- **Set 1 (scanners c, d)** — the first PR-stage definition (every associated point held), 60 exhibits = 8 unheld +
  52 clusters *outside* the bundle touching the candidate, + 12 controls. Not the owner's class; kept as a record:
  agreement 50/72, 22 both-YES, the NO pile = 17 through-going cosmics, halos and drift-boundary ghost clumps; the
  both-YES items were companions the taggers convicted (`skip_cosmic_companions`, by design), flashless
  (rescue-gid) clusters with prongs leaving the vertex (174422, 94942, 321235, 401252) and clusters of another
  flash. This is where the `nu_adopt_touching` side knob of sec 7 came from.
- **Set 2 (scanners e, f)** — the missing-prong definition with the 12 cm halo veto: 30 unheld-group events + 30
  nearby clusters + 12 controls (controls now on a fitted trajectory, away from its ends). **Result**
  (`113_pr_scan_score.txt`): agreement 62/72; **UNHELD-WIDE 0 both-YES of 28** (2 yes+unsure; 26 neither — both
  scanners read them as the band of the fitted track, "red on one side of the green line and grey on the other, displaced
  only in the drift projection", or as the diamond ghost lattice straddling a shallow track), UNHELD-TRACK 0 of 2;
  controls 1 / 12 (a real ~13 cm unheld stub beyond the vertex end of a green track — a genuine, small, class-3 object
  that the random control happened to land on). Every both-YES item (10) is a cluster *outside* the bundle (companions
  280466, 104172, 398225; other-flash 179369, 282909, 395837; flashless 401252, 401516, 100728) — the set-1 class the
  owner excluded. So the wide unheld population near the vertex is the imaging's isochronous band, not missing
  particles.
- **Set 3 (scanners g, h)** — the 22 track-like unheld groups with NO halo veto (the class that holds 279955,
  the class of pr/96's owner prong; 279955 itself sits in the wide class by this doc's 3-D rms and is not in the set)
  + 8 controls. **Result** (`113_pr_scan_score_track.txt`): scanner g 4 YES / 18 NO, scanner h 6 YES / 14 NO / 2
  UNSURE, agreement 24/30, **both-YES 3, either-YES 7, controls 0 / 8**. The both-YES items are the owner's class
  exactly: mcp2k 292810 (a third ~18 cm prong straight down from the vertex, the other two prongs green), 395870
  (a ~30 cm prong rising from the vertex), 70222 (a ~25 cm prong leaving the vertex below the green track); each
  4–6 % of the candidate's charge. Either-YES adds 292788 (a ~60 cm straight prong from the vertex, g), 54475 (a
  thin line parallel to and offset from the green track, h), 275539 and 93105 (h). The 21 NOs are the band of an
  already-green trajectory (collinear mid-segments and tails, same-slope stripes 2–6 cm off the line) — class 2 of
  sec 6.2 leaking through the straightness cut. So on this sample the track-like missing-prong rate at the census's
  sensitivity is **3–7 of 1392 candidates (0.2–0.5 %)**, each a prong of ~20–60 cm carrying ~4–6 % of the
  candidate's charge; the MC study's category is present in data but rare.

Caveats: control items carry a visible `p9xx` suffix (class hidden, control/exhibit split not); two earlier scanners
(a, b) were voided when the panel set was regenerated under them and correctly refused to relabel stale ids; the
first PR-stage controls (a random *associated* region) were loose — the held points thin out over the last ~10 cm of
a track — and gave a 50 % / 8 % control YES rate, which is why set 2's controls sit on a fitted trajectory.

### 6.4 Causes

For class 3 the mechanism is pr/96's: the prong's charge is associated to the neighbouring segment and no admission
predicate re-offers it (`other_seg_keep_isolated_*` acts on *isolated* residuals only; `other_seg_uncover_3d`, the
one uncovered-charge admission ever built, was 23 / 24 adverse in pr/102), and F2 — an uncovered-charge admission
disjunct inside the existing predicate — was designed in pr/96 sec 7 and never built. For class 2 the "loss" is
the imaging's isochronous band, upstream of the PR. For class 1 it is the candidate choice.

## 7. Side result: `nu_adopt_touching` (TaggerCheckNeutrino, default OFF), built on the first PR-stage reading and evaluated on all 3067 events

This knob answers scan set 1 (sec 6.3): flashless image clusters touching the candidate. The owner's clarification
("the matched beam bundle, not the unmatched clusters") makes it out of scope for the study's goal; it was already
running on the full sample, so its evaluation is recorded here and nothing is recommended.

**Design.** After the bundle's own companions are gathered (per-bundle path), every cluster that is not TGM/STM-tagged,
not the main, not already a companion, carries a **rescue gid** (no flash; `nu_adopt_touching_unmatched_only`,
default true — a cluster matched to another flash keeps its bundle), has length in [3, 100] cm (the cap is the
through-going-cosmic guard the scan found decisive) and whose closest approach to the main cluster is ≤ 3 cm
(`Cluster::get_closest_points`, the same helper `nu_bundle_flash_group` uses) is added to `other_clusters`, so it is
reconstructed and its charge reaches `kine_reco_Enu`. Log line `[nu_adopt_touching] gid G: adopted cluster C`.
Keys `nu_adopt_touching / _dis / _min_length / _max_length / _unmatched_only`, C++ defaults false / 3 / 3 / 100 /
true; jsonnet TLAs in `wct-pr-perevt.jsonnet` (keys omitted when off ⇒ byte-identical compiled config;
`prod_cfg_gate.py` 21/21 after the edit; `wcdoctest-clus` 24 683 / 24 683). Compiled-config proof of the ON arm:
`113_figs/113_pr_cfg_proof.txt`. Runtime OFF gate: stage B on the 16-event `manifest_gate16` with the new binary vs
`pr150g16new` (`arm_identity.py`): **16 / 16 events identical over 80 files and every ROOT branch**, the only allowed differences being the provenance strings (`Trun.cfg_tree / op_config_sha256 / provenance / toolkit_git / wcp_git`; `113_figs/113_pr_off_gate.txt`).

**Rule** (`113_figs/113_pred_pr.txt`, sha `e7426ea4…4320131`, 13:01, before the ON arm was read); the rule's P1
targets the flashless-cluster class, which the owner then excluded — read the numbers as a record.

**Result** (`113_figs/eval/d113adopt/{compare.log, sentinels.log, pr_summary.txt, adopt_census.txt}`; ON arm
`work-<s>-d113adopt`, 3067 events, rc 0 everywhere, wall median 0.27 × because the arm ran on a quiet machine):

- **P1 FAILS.** The knob fired on 8 events (10 clusters, length median 5.3 cm, closest approach median 0.5 cm), and
  the ON-arm census flags the same NEARBY-UNMATCHED events as production plus one (24 vs 23; the extra is 49767,
  where an adopted 5 cm stub re-drew the candidate). Of the six scan-confirmed flashless clusters only 174422 is
  adopted; the other five are outside the knob's design, not its thresholds: 321235, 401252, 351639 and 100728
  carry the **same rescue gid as the candidate** (a flashless bundle's members share one gid, so the "another
  bundle" test excludes them) and the taggers had already marked them **STM / TGM / TGM / STM**, and 94942 (gid
  1000003, closest approach 0.6 cm) is 153 cm long against the 100 cm cosmic cap (the census `ext` of 87.6 cm is a
  PCA extent, `Cluster::get_length` is the bounding extent). Removing the same-gid rule would re-admit clusters the
  taggers rejected, which is the TGM/STM decision, not an admission problem.
- **P2 partial.** `kine_reco_Enu` rose on 3 of the 8 adopted events (49767 0 → 115 MeV, 55286 0 → 36 MeV, 162143
  +0.004 MeV) and fell on none; on 174422 and the other four it did not move at all — a companion adopted at this
  point is reconstructed but its charge only reaches the energy when the PR connects it to the main vertex tree.
- **G1–G6 pass vacuously.** Vertex on the 397 + 481 + 158 + 7 labelled events identical (toward 0, away 0);
  event_label migrations 0; nu_evaluated flips 0; working points nue7 36/36, nue43 43/43, numu09 789/789; |dEnu| over
  1435 both-evaluated candidates median 0, nonzero in 3; TGM / STM / FC flips 0; π0 flag 1032/1032, mass window
  61/61; sentinels 21 PASS / 0 FAIL / 2 OPEN / 7 INERT (same as production); the 3059 events without an adoption
  have |dEnu| max 0.0 MeV — the knob is inert where it does not fire.

**Reading.** As built the knob touches 0.26 % of the events, changes the energy of 3, and cannot reach the
scan-confirmed cases because they are same-gid clusters the taggers rejected or > 100 cm. No ladder rung was run: the
rule's FAIL branch (max_length 60 cm or dis 2 cm) only narrows the knob further, and the class itself is outside the
study after the owner's clarification. Recorded; not recommended; default OFF.

## 7b. The owner's class (sec 6.2 population 3): no fix was built this round

The instrument bounds the class between the scanned subsets and the 143-event superset, and the mechanism is the one
pr/96 named; the fix it needs — an uncovered-charge admission inside the existing segment-admission predicate
(pr/96 F2) — is a PR-code change whose first measurement must be this doc's instrument (a segment that exists and does
not cover the charge is not a fix), with `other_seg_uncover_3d`'s 23/24 adverse movers as the cautionary precedent.
It was not built today: the population was only established after the redirect, and building it on an unscanned
superset would have repeated pr/102's mistake. Recommended next round: scan the 143-event superset in two halves
(track-like / wide) with the halo-band items labelled as such, size the class, then build F2 with the census as its
gate.

## 8. What this doc did not do, and defects found

- **No production flip.** Both knobs are default OFF; the imaging knob is measured non-viable (sec 5), the PR knob's
  verdict is in sec 7.
- **The imaging loss (sec 2–5) is real and unfixed**: 3.4 % of candidate events lose a plane-degenerate track
  segment or a drift-direction stub at the imaging stage because one plane's SP output is empty and the tiling
  needs all three. The next imaging round should (a) give quiet-masked cells an uncertainty class the clustering
  does not read as dead, or (b) bridge such gaps at the PR stage from the 2-D activity rather than in the imaging;
  either way the ghost load (× 1.4–2 surviving blobs) and the cost (× 4–9 imaging wall) of plain 2-view tiling are
  the numbers to beat.
- **SP defect surfaced, not addressed** (upstream of this tree): the collection plane carries no signal along
  tracks running along the drift direction or along its own wire direction (the "streak" class of sec 3, evt
  49951 / 58607 / 280709 / 56763 / 97084: the W wire predicted from U∩V is live and empty while the induction
  profiles rise like a Bragg peak); pr/64's 314507 is the same class.
- **H1 (zero-summary channels → uBooNE absolute thresholds) is dead** for the missing-particle population
  (`dt0` recovers nothing) but the 20–40 % of channels on the 2351 / 3347 / 2272 e fallback remain a
  threshold-uniformity question for another doc.
- **The 7 candidates the PR cannot reconstruct** (short mains, no vertex, `Enu 0`; sec 6.2) are a
  candidate-selection question (`nu_per_bundle` picks a 16–52 cm stub), not a missing-particle one.
- **PDHD/PDVD imaging gate not re-run** (inputs gone; `113_figs/113_pdgate.txt`); the `abtest/snap/d113pre|post`
  snapshots are vacuous and can be deleted.
- **Scan blinding caveat**: control items carry a visible `r9xx` / `p9xx` suffix (class hidden, control/exhibit split
  not); the PR-stage controls were a loose design (a random held region: the PR's held points thin out at track
  ends, so scanners read the last ~10 cm of tracks as unheld) and their 50 % / 8 % YES rates make the PR scan's
  YES figures upper bounds.
- **Runner and cfg additions kept (all default-OFF / byte-identical):** `IMG_EXTRA_TLA` and `QL_EXTRA_TLA` channels
  in `run_chain_group.sh`; `img_knobs` in `img.jsonnet` / `wct-img-all.jsonnet`; `dg_length_cut` in `clus.jsonnet` /
  `wct-clus-matching-perevt.jsonnet`; `quiet_mask_window` / `quiet_mask_gap` in `MaskSlices`; `nu_adopt_touching*`
  in `TaggerCheckNeutrino` / `wct-pr-perevt.jsonnet`.
- **Disk**: the quiet-plane arms hold ~70 G (`work-mcp1k-d113cfqp2v4`, `-cfqpg64`, `-cfqpg64dg`) and are
  releasable now; the counterfactual arms (~20 G) once the doc is accepted (`docs/work-tags.md`).

## 9. Files

- Doc: `docs/113_missing-particles-in-3d-image.md`; figures / tables / rule files under `docs/113_figs/`
  (census TSVs `pr_census/`, `pr_census_asc/`; panels `panels/`, `pr_panels/`, `pr_panels_track/`; the raw
  imaging census `census/` (96 MB, 388 per-group files) is NOT committed — `d113_missing2d.py` regenerates it from
  the stage-A products in ~30 min; proofs `113_cfg_proof.txt`,
  `113_pr_cfg_proof.txt`, `113_baseline_repro.txt`, `113_off_gate.txt`, `113_pr_off_gate.txt`, `113_pdgate.txt`;
  results `113_selftest.txt`, `113_census_summary.txt`, `113_event_coverage.txt`, `113_quiet_plane.txt`,
  `113_streak_check.txt`, `113_scan_score.txt`, `113_counterfactual.{txt,tsv}`, `113_counterfactual_fix.tsv`,
  `113_fix_cost.txt`, `113_cluster_presence.txt`, `113_bundle_class.txt`, `113_pr_summary.txt`,
  `113_pr_summary_asc.txt`, `113_pr_summary_track.txt`, `113_pr_sensitivity.txt`, `113_pr_scan_score.txt`,
  `113_pr_scan_score_track.txt`, `113_pr_flags*.tsv`, `113_pr_exhibits*.tsv`, `113_nearby_causes.txt`,
  `113_unheld_causes.txt`, `113_unheld_causes2.txt`, `113_absorbing_segments.txt`, `113_nosteiner_census.txt`,
  `113_kine_excluded.txt`, `eval/d113adopt/`; rules `113_pred.txt`, `113_pred_pr.txt`, `113_pred.sha256`;
  TLA files `tla/`).
- Scan labels (new tags, never written into an existing one): `missing_labels/mscan-d113-{a,b}/`,
  `missing_labels/prscan-d113-{a..h}/` (a, b of the PR scan are void: stale panels; c, d = scan set 1 on
  `pr_panels/`; e, f = scan set 2 (wide groups) on `pr_panels/`; g, h = scan set 3 (track-like groups) on
  `pr_panels_track/`).
- Scripts: `scripts/analysis/d113/{d113_common, d113_selftest, d113_missing2d, d113_summary, d113_render,
  d113_scan_score, d113_counterfactual, d113_arm_cost, d113_cluster_presence, d113_bundle_class, d113_pr_uncover,
  d113_pr_summary, d113_pr_render, d113_adopt_census}.py`; launchers `scripts/d113_stageA_arm.sh`,
  `scripts/d113_counterfactual_arms.sh`, `scripts/d113_fix_arms.sh`, `scripts/d113_arm.sh`, `scripts/d113_eval.sh`,
  `scripts/d113_eval_pr.sh`; runner edit `run_chain_group.sh` (IMG_EXTRA_TLA, QL_EXTRA_TLA).
- Toolkit (branch `apply-pointcloud`): `img/inc/WireCellImg/MaskSlice.h`, `img/src/MaskSlice.cxx`
  (quiet_mask_window / quiet_mask_gap), `clus/inc/WireCellClus/TaggerCheckNeutrino.h`, `clus/src/TaggerCheckNeutrino.cxx`
  (nu_adopt_touching*), `cfg/pgrapher/experiment/sbnd/{img,wct-img-all,clus,wct-clus-matching-perevt,wct-pr-perevt}.jsonnet`.
- Arms: `docs/work-tags.md` (doc 113 block).

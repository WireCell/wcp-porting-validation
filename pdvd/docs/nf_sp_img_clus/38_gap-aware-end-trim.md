# 38 — The gap-aware end trim: retuning what the ctpc metric was being graded through

**Status.** Toolkit change shipped **default OFF** (`TrackFitting` parameter
`end_trim_gap_len`, C++ default `0`). Opened by doc 36 §12 and by the owner's
instruction when flipping PDVD to the anisotropic ctpc metric: *"we can then
work on the later part step by step to improve the performance — this is the
first step."* This is the second step.

Doc 36 §10.1 established that `TrackFitting::examine_end_ps_vec` is a **tip
test**: it pops points off each end of a fitted path while the strict
three-plane good-point test fails, breaks at the first point that passes, and
never examines the interior. That is only safe while the good-point test is
wrong in the helpful direction — the isotropic test failed 82 % of points
sitting on real charge, which is what used to walk a 55 cm chord back off PDVD
039252/2 cluster 109 by accident. With the metric (PDVD production since
`605833eb` / `38245d18`) the same trajectory passes at 0.91 on charge, the tip
of a detached fragment correctly passes, and nothing is popped: the chord
survives as a 46 cm tail. The trim, not the metric, is what sets the
coverage-for-ghost exchange rate of doc 36 §10.3.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus -tc="clus end trim gap*"

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
SC=docs/nf_sp_img_clus/scripts
# five 120-event arms: the reference pin, the knob-off arm on the new pin, and
# the sweep.  Post-flip production config (metric ON, floor retired, doc-37
# terminal thinning ON), inputs symlinked from d27fresh.
PIN=<pin_d38ref> JOBS=16 ARMS='d38ref'                   $SC/run_d38_arms.sh
PIN=<pin_d38new> JOBS=16 ARMS='d38off d38g2 d38g3 d38g5' $SC/run_d38_arms.sh
# byte identity of the knob-off path
for d in work/*_d38ref; do b=${d%_d38ref}; \
  python3 ../abtest/hash_archive.py $d/mabc-pr.zip ${b}_d38off/mabc-pr.zip; done
# the grade
python3 $SC/d36_fit_twoaxis_scan.py d38off,d38g2,d38g3,d38g5 /tmp/d38_twoaxis.tsv
python3 $SC/d36_stm_verdict_flips.py d38off d38g3
python3 $SC/d38_deadarea_census.py  d38off d38g3 /tmp/d38_dead.tsv
# shared-component gate (SBND + uBooNE), knob off
OLD_ARM=d38old NEW_ARM=d38new OLD_LIB=<pin_d38ref>/libpin NEW_LIB=<pin_d38new>/libpin \
  QL_SUFFIX=d99fix ./pdvd/stm/gates/shared_gate.sh arms && ... shared_gate.sh compare
```

## 1. The rule

At a tip the pop loop has already stopped on a point that PASSES. Ask whether
that tip is **attached**. `Clus::scan_tip_island`
(`clus/inc/WireCellClus/EndTrimGap.h`) walks inward and measures three lengths:

| | |
|---|---|
| `supported_len` | the supported stretch at the tip — the candidate island |
| `gap_len` | the unsupported run immediately inward of it |
| `body_len` | the supported stretch beyond that run — the body |

and the trim drops the island when

```
detached && gap_len > end_trim_gap_len && gap_len > supported_len && body_len > supported_len
```

— i.e. **the island is the smallest of the three**. It iterates, so a path
ending in several islands loses them one at a time, and each iteration removes
at least one point, so it terminates. `body_len` stops being measured as soon
as it exceeds `supported_len` (that is all the caller asks), and a scan cap of
1000 points bounds the cost on a pathological path; hitting either the cap or
the end of the path leaves `detached` false, which is the inert answer.

## 2. Two formulations that measurement killed

Recorded because each looked right on the flagship cluster and failed on the
population — the same trap doc 36 §5 fell into.

**(a) "Is the next point inward supported?"** Fixes cluster 109, whose island is
a single point, and does **nothing** on 039349/48 cluster 53, whose island is
*two* points: the immediate neighbour passes, so the scan reports no gap. The
island is a stretch, not a point.

**(b) "The gap is longer than the island."** Fixes both (109: 1 point holding
53.7 cm; 53: 2 points holding 65.8 cm) and then **chews**: on a trajectory
whose supported runs are all short, each iteration eats one island and its gap,
the next short run becomes the tip, and the whole path is amputated. Simulated
offline on 12 events it took one 168-point fit down to 2 points and dropped
aggregate coverage by 7 points. The `body_len > supported_len` clause is the
fix, and it is also the dead-channel protection: the strict three-plane test
has no bad-plane allowance, so a dead region reads exactly like empty space,
but at a real track end the supported stretch is the track itself and outweighs
what lies beyond the gap.

Both were found by replaying the three-plane test offline on the arm's own
trajectories. That replica is trustworthy where its verdicts can be checked
against the arm's actual behaviour (cluster 109: 0.91 on-charge pass, and the
arm really does keep the chord) and is **not** trustworthy as a population
instrument — on some clusters it reports on-charge pass rates near zero where
the job's own census reports 0.70. The numbers below come from the arms.

## 3. What was built

- `clus/inc/WireCellClus/EndTrimGap.h` (new): `TipIsland` and the
  `scan_tip_island` function template, iterator- and predicate-templated so the
  rule is unit-testable — `examine_end_ps_vec` needs a Grouping, a
  DetectorVolumes and a PCTransformSet and is not reachable from a doctest.
- `clus/inc/WireCellClus/TrackFitting.h`: `Parameters::end_trim_gap_len`
  (a LENGTH in WCT internal units; 0 = off), with the physics of both
  comparison clauses in the comment.
- `clus/src/TrackFitting.cxx`: `set_parameter` / `get_parameter` entries; a
  `trim_point_supported` lambda that is the same strict three-plane test the pop
  loops apply, as a predicate on an arbitrary point; and the two pop loops
  wrapped in an outer loop that runs them exactly once and breaks when the knob
  is 0. **The pop loops are left at their original indentation on purpose** so
  the diff of this production path shows only added lines.
- `clus/test/doctest_end_trim_gap.cxx` (new): the cluster-109 shape, the
  cluster-53 shape, a real track end beyond a dead region, the chewing case, the
  no-body sentinel, the reverse-iterator (far end) case, the scan cap, polyline
  vs chord, degenerate ranges, and the knob default.

## 4. Gates — knob OFF is byte-identical

Pins: `pin_d38ref` = `b38c6ea0` clean (lib built 11:27); `pin_d38new` = the same
tree plus the knob (11:29). Freshness proved by `ls -la` on
`local/lib/libWireCellClus.so` against the last source edit in both directions.

| gate | arms | result |
|---|---|---|
| **PDVD PR, 120 events** | `d38ref` (old pin) vs `d38off` (new pin), production config | **238 / 238 identical**, 0 diff: **all 120** `mabc-pr.zip` member rollups, plus the calib JSON (minus the `vertex_scoreboard.dual_chain` timer) of the **118 of 120 events that produce one** — 039349/65 writes no dump because it has 0 STM tags (doc 25 §13.10), and 039349/76 writes none despite 1 STM tag, which this round does not explain. Both are covered by the zip comparison. |
| **SBND PR chain** | `doc25d38old` vs `doc25d38new`, nuecc48 + ncpi0 | **201 / 201 identical** (zip + calib + nusel), 0 diff |
| **uBooNE, 35 events** | `sweep/doc25d38old` vs `doc25d38new` | **35 / 35 zips content-identical; tagger 35 / 35** |
| doctests | `./build/clus/wcdoctest-clus` | **294 / 294**, 22 624 assertions |

PDHD is deliberately not in the list: `examine_end_ps_vec` lives in
`TrackFitting`, which only the PR chain runs, and PDHD has no PR chain. SBND is
the detector that binds the changed component, and it is gated.

### 4.1 Two harness traps this round hit

Recorded because both produced a confident wrong answer before being caught.

**An empty TLA value is not a no-op.** The first sweep arm ran with
`-A trackfitting_config=` — an empty string — because a `set -u` abort inside
the driver (`local cm=$1 dst="...${cm}..."` expands `${cm}` before the
assignment takes effect) left the path empty. `TaggerCheckSTM::configure` guards
with `if (!m_trackfitting_config_file.empty())`, so it skipped
`load_trackfitting_config` entirely and the arm ran on the **C++ default**
TrackFitting parameters instead of PDVD's. It completed 120/120 rc=0 with normal
logs, and one hand-checked cluster was identical to two decimals — but member
hashes differed on **111 of 111** events. The arm (`d38g2`) is abandoned under
its own tag rather than overwritten (M13); the sweep was re-run as `d38h*`. The
driver now asserts the generated JSON contains `end_trim_gap_len` and refuses
any arm whose composed TLA string ends in `<name>=`.

**A byte-identity gate against a still-running arm lies.** Run early, the PDVD
gate reported 13 differing events; all 13 had 0-byte or half-written zips and
two were not in the arm's own rc list. Gate on the runner's `rc_<tag>.txt`, not
on file existence.

## 5. The measurement — the sweep

Six 120-event arms on `pin_d38new`, all 120/120 rc = 0, post-flip production
config (metric ON, `good_point_pitch_frac` 0, doc-37 terminal thinning ON).
`d38off` is that config with the knob off; `d38h<N>` sets
`end_trim_gap_len` = N cm through a generated trackfitting JSON.

**The two axes** (`d36_fit_twoaxis_scan.py` + `d38_sweep_grade.py`; 2159
clusters with ≥ 50 charge points and a fit in some arm, 4 034 975 charge
points). *coverage* = the fraction of each cluster's own 3-D charge within 2 cm
of its fit; *> 2 cm* = the fraction of fit points that far from ANY 3-D charge
of the event. The last three columns are the harm the two axes hide, because
coverage is charge-weighted and a short cluster losing everything barely moves
it.

| L (cm) | coverage | > 2 cm | > 10 cm | STM tags | trimmed | lost > 5 cov pts | fit destroyed |
|---|---|---|---|---|---|---|---|
| **off** | 71.7 % | 12.7 % | 5.8 % | 685 | – | – | – |
| 2 | 71.1 % | 6.2 % | 2.3 % | 738 | 1213 | 219 | 10 |
| 3 | 71.2 % | 6.2 % | 2.3 % | 730 | 1076 | 206 | 10 |
| 5 | 71.3 % | 6.3 % | 2.3 % | 723 | 899 | 175 | 9 |
| 10 | 71.4 % | 6.4 % | 2.4 % | 709 | 759 | 143 | 9 |
| **20** | **71.5 %** | **7.0 %** | 2.4 % | 720 | 619 | 104 | 7 |
| **40** | **71.6 %** | **8.5 %** | 2.9 % | 720 | 406 | **46** | **1** |

**The comparison that is clean is within this table**: every arm here shares one
config epoch and one binary family, and differs only in `end_trim_gap_len`.

For reference, doc 36's arms graded 68.3 % / 8.3 % (legacy frac-0),
**70.3 % / 10.2 % / 688 STM (pre-flip production: isotropic + the 0.35 floor)**
and 71.1 % / 12.6 % (the metric). **Read that reference with two caveats.**
(i) *Different epoch.* Doc 36's arms predate doc 37's Steiner terminal thinning,
which went to PDVD production at `730bc794` between the two rounds; the d38 arms
all carry it. Holding the metric and the floor fixed, thinning accounts for
71.1 → 71.7 % coverage and 12.6 → 12.7 % unsupported, i.e. it helps coverage a
little and is neutral on support — so "metric + trim beats pre-flip production
on both axes" is a **three-change** comparison, not one. (ii) *Different
cluster set*: 2176 clusters there against 2159 here, because the PR stage
re-clusters slightly.
(iii) `dl_weights` is the **same on every side of every comparison in this doc**
(the SCN vertex is ON, the jsonnet default; no arm overrides it), verified from
the compiled config rather than from the invocation. An earlier version of this
paragraph cited doc 37 §15.3 for a pointer-order determinism defect on the TGM
path; **that claim has been retracted at its source** (doc 37 §15.5) — the pair
it rested on spanned the doc-36 §11 flip as well as the vertex. The DL/SCN
vertex is inert for both cosmic taggers, as the pipeline order always implied.
What the retraction leaves standing is a measurement of the **flip**, not of the
vertex: see doc 36 §11.1.

Within the epoch, **every threshold takes unsupported trajectory from 12.7 % to
6.2–8.5 % while coverage moves by at most 0.6 points and STM rises** — the first
time in this campaign either axis has moved without paying on the other.

Three things the sweep says:

1. **The benefit is nearly flat in L and the harm is not.** From 2 to 40 cm the
   unsupported fraction rises only 6.2 → 8.5 % while the clusters losing more
   than five coverage points fall 219 → 46 and destroyed fits 10 → 1. Physically:
   the chords worth removing are LONG, so a permissive threshold catches
   essentially all of them while sparing short dropouts inside real tracks.
   (The offline simulation of §2 preferred L = 3; the arms say the opposite,
   which is the third time this round that the replica was a good mechanism
   probe and a bad population instrument.)
2. **L is bounded above by the defects it must catch.** Cluster 109's chord is
   53.7 cm and cluster 53's 65.8 cm, so above ~50 cm the rule stops seeing the
   cases it exists for. The usable window is roughly 20–40 cm.
3. **STM moves the right way.** 685 → 709–738 tagged clusters. Trimming a
   spurious tail restores the stopping-muon signature, so the 31 tags the metric
   cost (doc 36 §4.2) are not merely recovered — the trim overshoots them.
   TGM is 0 in every arm of this manifest and does not move.

### 5.1 The two clusters this campaign was built on

| cluster | arm | fit pts | extent | coverage | > 2 cm | max dist |
|---|---|---|---|---|---|---|
| 039252/2 cl 109 (charge 205 cm) | off | 816 | 201.8 | 91 % | 41 % | 28.4 |
| | L = 20 | 562 | **151.1** | 90 % | 19 % | 18.4 |
| | L = 40 | 627 | 151.1 | 90 % | 26 % | 18.4 |
| 039349/48 cl 53 (charge 281 cm) | off | 430 | 246.5 | 84 % | 37 % | 29.0 |
| | L = 20 | 255 | **151.8** | **86 %** | **0 %** | 1.0 |
| | L = 40 | 321 | 190.1 | 83 % | 18 % | 17.2 |

The 46 cm tail is gone at every threshold ≥ 5 cm, and cluster 109 now stops at
151 cm — where the retired 0.35 floor used to stop it by accident (150.2 cm),
but at 19 % unsupported instead of 18 % **with a test that is right about real
charge** rather than one that fails 82 % of it. Cluster 53 is a clean win on
both axes at L = 20: the 95 cm ghost extension is removed entirely and coverage
*rises*.

### 5.2 What the trim costs, cluster by cluster

At L = 20 seven clusters lose their fit outright and 104 lose more than five
coverage points. They are a recognisable class: **sparse multi-blob objects** —
51 to 71 charge points strung over 80 to 101 cm, i.e. 1.1–2.0 cm of extent per
charge point, with the charge in a few tight knots (nearest-neighbour spacing
0.30 cm median) separated by long empty stretches. Their fits ran 77–81 % more
than 2 cm from any charge, at a median 6.4–7.8 cm. That is the doc 32 §2.2
over-clustering / endpoint-selection defect, not a trim failure: the trim is
refusing to invent a trajectory across it. Whether refusing is better than
inventing is a physics call for the owner, and it is the only reason to prefer
L = 40 (one destroyed fit) over L = 20 (seven).

## 6. Dead channels — measured, not designed around

A long run of failing points is also what a genuine dead-channel region looks
like: this is the strict three-plane test with no bad-plane allowance. Of the
85 902 fit points the L = 3 arm removes, **81.2 % are more than 2 cm from any
3-D charge** — it is removing ghost, not track — and **0.7 % fall inside a Bee
`channel-deadarea` polygon**. That 0.7 % is an upper bound: the dead-area layers
carry no x extent, so a dead region in one drift volume shadows the other.
No dead-channel exemption is needed. (`d38_deadarea_census.py`.)

## 7. Cost

Summed over the 120-event manifest, 16 jobs, same round:

| arm | wall (s) | peak RSS (GB) |
|---|---|---|
| `d38off` | 5364 | 210.3 |
| L = 5 | 5282 | 216.3 |
| L = 20 | 5215 | 214.1 |
| L = 40 | 5186 | 213.7 |

Not measurably slower — slightly faster, if anything, since a shorter trajectory
is cheaper downstream. The scan itself is bounded: `body_len` stops being
measured once it beats the island, and `max_scan` caps the walk at 1000 points.

## 8. The operating point is a physics choice, not a margin

Flipping the knob is worth doing: within its own epoch every threshold in the
window cuts unsupported trajectory by 4.2 to 5.7 points for at most 0.6 points
of coverage and raises STM tagging, at no measurable wall or RSS cost, and the
knob lives in
`cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` — the same file
the retired 0.35 floor lived in.

**Which threshold is not a question this doc can settle**, because 20 and 40 cm
do not differ by margin: they trade *different clusters*.

| | L = 20 cm | L = 40 cm |
|---|---|---|
| aggregate | 71.5 % / 7.0 % / STM 720 | 71.6 % / 8.5 % / STM 720 |
| 039349/48 **cluster 53** | 152 cm, **0 %** off charge, coverage 86 % | 190 cm, 18 % off charge, coverage 83 % |
| 039349/80 **cluster 67** | **fit destroyed** | 64 points, 36 cm, coverage 90 % |
| clusters losing > 5 coverage points | 104 | 46 |
| fits destroyed | 7 | 1 |

The question that decides it: **is a trajectory drawn through 80 cm of empty
space on a 51-point cluster better than no trajectory at all?** If yes, 40 cm —
it keeps those fits and gives back 1.5 points of ghost removal. If no, 20 cm —
it takes cluster 53's ghost extension to zero and finishes the defect this
campaign exists for. Both are far better than either arm shipped today.

**Do not go below 10 cm** whichever way that falls: between 20 and 2 cm the
extra ghost removal is 0.8 points and it costs 115 more damaged clusters.

Not decided here (CLAUDE.md §5.1): the flip is the owner's, and this doc ships
the knob default-OFF with the evidence.

## 9. Not done in this round

- **The over-clustering itself.** §5.2's sparse multi-blob clusters and cluster
  109's five-point fragment are the same defect at the source; the trim only
  refuses to draw through it. Endpoint selection (`get_two_boundary_wcps`) that
  rejects a boundary point across a charge gap is the next candidate.
- **Splitting a path at an interior gap.** The trim reaches only from the ends.
  039252/16 cluster 103 keeps 17 % unsupported at every threshold because its
  chords sit between two supported stretches.
- **SBND.** The knob is detector-agnostic and default-OFF; nothing here measures
  it there, and SBND is staying on the isotropic metric anyway, where doc 36
  §10.1's argument says the rule cannot be stated.
- **The PDVD clustering job** still uses the isotropic metric (doc 36 §11).

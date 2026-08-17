# 73 — Cathode bundle rescue, round 2: the residual one-sided crossers

**Status: IMPLEMENTED (toolkit `44aec999`) + VALIDATED on the signal set (§5) +
knob-off byte-identical (§4). ALL FOUR KNOBS SHIP DEFAULT OFF.** No SBND
production flip; that is the owner's call on the §6 census.

doc 72 §A found 10 events in 3000 where the in-beam bundle main is cut at the
cathode and the raw image plainly continues into the other TPC.
`ClusteringCathodeBundleRescue` (doc pr/14 + pr/17 + pr/19) already exists to fix
exactly this and is **SBND default ON**, so every one of the 10 is a case where
the existing rescue did not fire. Round 1 of this doc (2026-08-17, design only)
attributed the misses and proposed three extensions. This is round 2: the
implementation, four corrections to round 1's design, and the measurement.

**Result: 9 of the 12 signal events are fixed**, every one of them into the beam
bundle. Knob-off is byte-identical to the production arms on all 12 (member
content hash). Bee before/after in §7.

## 1. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the 12 signal events (doc 72 §A's 10 + 281165 + 167744)
printf '%s\n' 65289 65053 398115 237798 493439 78242 51128 317427 319913 167964 \
              281165 167744 > /home/xqian/tmp/cbr2_signal.txt

# Q/L ONLY -- no PR chain anywhere in this round (owner, 2026-08-17)
CATHODE_RESCUE_DEBUG=1 ROOT=$PWD/work-cbr2-off \
    ./run_ql_batch.sh -j 6 -f /home/xqian/tmp/cbr2_signal.txt
CATHODE_RESCUE_DEBUG=1 SBND_RESCUE_IN_BEAM=1 SBND_RESCUE_GEOM_FIRST=1 \
    SBND_RESCUE_PIERCE=1 SBND_RESCUE_DEST_BEAM=1 ROOT=$PWD/work-cbr2-on2 \
    ./run_ql_batch.sh -j 6 -f /home/xqian/tmp/cbr2_signal.txt

# did the missing half JOIN the in-beam main?  (PR-free; reads only the Q/L zip)
python3 scripts/analysis/cathode/cbr_join_metric.py \
    --baseline work-mcp1k-cb0805 --baseline work-mcp2k-cb0816 --test work-cbr2-on2 \
    --rows products/beam-cathode-split-mcp1k.tsv \
    --rows products/beam-cathode-split-mcp2k.tsv

# per-knob ablation (F4 on throughout -- it is a destination fix, not an admission one)
for a in f1:SBND_RESCUE_IN_BEAM f2:SBND_RESCUE_GEOM_FIRST f3:SBND_RESCUE_PIERCE; do
  env ${a#*:}=1 SBND_RESCUE_DEST_BEAM=1 ROOT=$PWD/work-cbr2-${a%%:*} \
      ./run_ql_batch.sh -j 4 -f /home/xqian/tmp/cbr2_signal.txt
done

# firing census, mcp1k 1000 events, OFF then ON from ONE binary
ROOT=$PWD/work-cbr2-c1koff ./run_ql_batch.sh -j 8 -f /home/xqian/tmp/cbr2_mcp1k.txt
SBND_RESCUE_IN_BEAM=1 SBND_RESCUE_GEOM_FIRST=1 SBND_RESCUE_PIERCE=1 \
    SBND_RESCUE_DEST_BEAM=1 ROOT=$PWD/work-cbr2-c1kon \
    ./run_ql_batch.sh -j 8 -f /home/xqian/tmp/cbr2_mcp1k.txt
```

**Binary provenance.** Every arm in this doc: toolkit HEAD `91b78d67` +
the round-2 patch, `local/lib/libWireCellClus.so` mtime `2026-08-17 08:14:46`.
Both arms of every comparison come from that one binary, so an OFF/ON delta is
attributable to these knobs and to nothing else in the tree. (A concurrent
session shares this working tree and moved HEAD `a681b3e1` → `91b78d67`
mid-round; hence the explicit record.)

## 2. What exists today

`clus/src/clustering_cathode_bundle_rescue.cxx`, run in the Q/L all-APA pipeline
after `cathode_connect` and **before** `examine_bundles`. Its own header states
the mechanism the owner described: *"an early flash opens a readout window
(~8 µs) that can ABSORB a later flash. A beam interaction whose track crosses the
central cathode scintillates in both drift volumes and should yield a
beam-coincident flash on each side; when one side's flash is absorbed, that
side's charge half gets Q/L-matched to a DIFFERENT flash."*

SBND production values (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:517-530`; the
rest are C++ defaults):

| parameter | value | role |
|---|---|---|
| `beam_window_low/high` | 0.2 / 2.2 µs | which cluster counts as `K_beam` |
| `rescue_t0_early` / `rescue_t0_late` | 8 / 13 µs | `t0(K_far) − t0(K_beam)` must lie in `[−8, +13]` |
| `require_far_out_of_beam` | **true** | `K_far` must NOT itself be in the beam window |
| `min_length_short` | 2 cm | floor on either half |
| `cathode_x_cut` | 5 cm | each tip's distance to the cathode plane |
| `drift_cut` | 8 cm | \|Δx\| between the two tips |
| `max_dis` | 25 cm | 3-D tip-to-tip distance |
| `angle_cut` | 10° | collinearity of the two half directions |
| `conn_far_cut` | 30° | angle between the **tip-to-tip vector** and a half's direction |
| `short_dir_len` / `conn_short_cut` | 25 cm / 30° | short-stub prolongation path |

## 3. Why the existing rescue does not fire — now PRINTED, not inferred

Round 1 attributed classes A and B from the Q/L products, because the `[cbrx]`
tracer lives *inside* `is_cathode_crossing_pair` and those pairs are pruned in the
**selection loop**, before that function is ever called — so "no line" was
ambiguous between "never tested" and "tested, far apart". Round 2 adds a
`[cbrsel]` print in the selection loop carrying both clusters' `t0`, `gid`,
in-beam flag and length plus the gate that pruned the pair, and adds `pierce` to
the `[cbrx]` line. Every claim below is now a printed line.

Knobs OFF, 12 events, 506 candidate pairs:

| event | pairs | prune reasons |
|---|---|---|
| 398115 | 61 | dt0 57, **in_beam_far 4** |
| 237798 | 40 | dt0 36, **in_beam_far 4** |
| 65053 | 45 | dt0 42, in_beam_far 3 |
| 319913 | 51 | dt0 45, in_beam_far 6 |
| 167964 | 45 | dt0 42, in_beam_far 3, scope 13, len 6 |
| 281165 | 46 | dt0 44, in_beam_far 2 |
| 167744 | 34 | dt0 34 |
| 493439 | 22 | dt0 20, scope 5, len 2, geometry 1 |
| 78242 | 30 | dt0 26, scope 1, geometry 3 |
| 317427 | 27 | dt0 25, scope 1, geometry 1 |
| 51128 | 69 | dt0 60, scope 3, len 3, **geometry 3** |
| 65289 | 36 | dt0 34, **geometry 2** |

and the two pairs that did reach the geometric test:

```
evt65289  [cbrx] c5<->c13 far_conn        dis=5.28 dX=5.14 tip=3.04/2.10 ttH=15.0 ttP=9.0 cc=45.2 pierce=5.46 len=160/89  -> reject
evt51128  [cbrx] c11<->c8 close_shortstub dis=2.78 dX=0.43 tip=0.53/0.10 ttH=11.6 ttP=-1.0 cc=41.0 pierce=2.83 len=4/284  -> reject
```

**Round 1's clean A/B/C split does not survive the printed evidence.** The classes
overlap: 65053, 319913 and 167964 carry `in_beam_far` prunes as well as `dt0`
ones, so they were never purely "class B". The per-knob ablation in §5 replaces
the class table.

### 3.1 The `conn_far_cut` bias (evt65289) — round 1's finding, confirmed

`conn = p1 − p2s` is the **tip-to-tip vector**, and `cc = angle(conn, dir)`. Here
`dis = 5.28` of which `dX = 5.14` — the vector is **97 % pure drift**. That is not
a track segment; it is the cathode dead gap plus the near-cathode imaging loss
(the two tips sit 3.04 and 2.10 cm short of the plane). Its angle to the track
direction is therefore ≈ `acos(|dir_x|)`, and for this event `|dir_x| = 0.748`
gives 41.6° against a measured 45.2°.

**So with an x-dominated `conn`, `conn_far_cut = 30°` is not testing collinearity
at all — it is testing `|dir_x| > cos 30° = 0.866`, i.e. it rejects any crosser
more than 30° off the drift axis.** Note `tt_pca = 9.0°` had *already* passed
`angle_cut = 10°`. 10 of the 12 candidate rows have `|dir_x| < 0.866`, median 0.71.

The printed `pierce = 5.46 cm` is the substitute measurement, and it is good.

## 4. Four corrections to round 1's design

Round 1 was a design doc; three of its four load-bearing claims did not survive
implementation. Recorded as retractions, not silently edited away.

**(a) §4.2 was wrong about the direction donor.** It claimed evt51128's short-stub
path takes its direction from the 4 cm stub. The code is
`one_is_anchor = (length_1 >= length_2)` and the tracer prints `c11<->c8 …
len=4/284`, so cluster2 (284 cm) is the anchor — the donor is already the long
half. The actual defect is different: `dis = 2.78` with `dX = 0.43` means the
baseline is 2.8 cm and almost entirely *transverse*. A 2.8 cm baseline cannot
define a direction, so `cc = 41.0°` is noise. The real transverse offset is
2.75 cm, which is fine.

**(b) Round 1's F3 trigger would have fixed only one of the two class-C events.**
It triggered on `dX/dis > conn_drift_frac ≈ 0.8`:

| event | dis | dX | dX/dis | round-1 trigger? |
|---|---|---|---|---|
| 65289 | 5.28 | 5.14 | 0.973 | yes |
| 51128 | 2.78 | 0.43 | 0.155 | **no** |

The shipped trigger is **two conditions, either sufficient**: drift-dominated
(`dX/dis > conn_drift_frac`) **or** baseline too short to define a direction
(`dis < conn_min_dis`). Round 1's companion requirement (`angle(dir1,dir2) <
angle_cut`) is also **not** carried into the short-stub branch: evt51128 has
`tt_hough = 11.6°` against `angle_cut = 10°` and no PCA fallback
(`tt_pca = −1`, the both-long branch never ran), so requiring it keeps 51128
rejected.

**(c) Round 1 put the regression risk on F3. It belongs to F2.** F2 widens the
candidate pool from the 1–2 clusters inside the `dt0` window to every matched
cluster in the event (15 in evt65289), and a wrong merge re-stamps a cosmic with
the beam T0. F3 by contrast only ever *replaces* one angle test with a sharper
transverse one on a pair that already passed every hard gate.

**(d) Round 1's contrast set is a near-vacuous regression guard.** The 162
`already-in-main` events have their two halves already in one cluster, so at
rescue time there is no *pair* for the pair test to evaluate and nothing in that
set can move whatever the geometry does. The real false-positive guard is the
**whole-sample firing census** (§6). Round 1 §6 would have let a bad F2 through on
a clean-looking contrast number.

Also recorded: the component header's safety argument — a wrongly adopted cosmic
"becomes a joined cathode crosser the TGM/STM taggers evaluate downstream" — is
**unobservable this round**, because PR is out of scope by owner direction.

## 5. What was implemented

Four knobs in `clus/src/clustering_cathode_bundle_rescue.cxx`, all C++ default
**false**, plus the `[cbrsel]` tracer. Config path is the existing one:
`SBND_RESCUE_*` → `run_ql_evt.sh` → TLA → `wct-clus-matching-perevt.jsonnet` →
`clus_all_apa` → `cm.cathode_bundle_rescue`, key-suppressed at every level.

### F1 — `rescue_allow_in_beam_far`

`K_far` may itself be in the beam window provided it is in a different flash
bundle. Both halves in-beam ⇒ the T0 hypothesis moves by ≤ (2.2−0.2) µs ×
0.1563 cm/µs = **0.31 cm**, sub-blob, so the direction rule needs no new
tie-break.

### F2 — `rescue_geom_first`

A pair the `dt0` window rejected is still tested, but every accept must
additionally pass a tightened geometry: `dis < geom_first_dis`, piercing
agreement `< pierce_cut`, and direction evidence. Purely additive — no pair the
legacy path takes can change.

Round 1's condition 4 (corroboration that the far half's flash story is
independently broken) is **dropped**: 4(a) is unmeasurable from inside the
component (a flash that matched no cluster is invisible there) and 4(b) is
satisfied by construction, since F2 only ever sees pairs the window already
rejected.

**F2 needed one thing round 1 did not anticipate — see §5.3.**

### F3 — `rescue_pierce_test`

Where `conn` is drift-dominated or the baseline is too short, substitute the
**cathode-piercing agreement**: extrapolate each tip along its own local
direction to `x = cathode_x` and compare the two `(y, z)`. A half whose direction
is unusable (nearly isochronous, or shorter than `short_dir_len`) contributes its
own tip `(y, z)` instead, which is already within `cathode_x_cut` of the plane.
Applied in the FAR branch (alongside the existing collinearity gate) and in the
SHORT-STUB branch (alone — correction (b)).

Why it is sharper, not looser: `conn_far_cut = 30°` is a tolerance that *scales*
with the tip separation (≈2.5 cm at `dis = 5`, ≈12 cm at `dis = 25`) and collapses
to a `|dir_x|` cut on a drift-dominated `conn`; `pierce_cut` is a fixed transverse
bound on an extrapolated point.

### F4 — `rescue_dest_beam_for_new`

**Acceptance is not a fix, and this was measured, not hypothesised.** With
F1+F2+F3 on and F4 off, 7 events fired and **two of them went to the wrong
bundle**:

```
evt281165  c15 (gid 1000007, t0 0.861 us, 21.6 cm) + c9 (gid 3, t0 276.652 us, 121.5 cm)
             -> gid 3  (longer-half; a=21.6 b=0.0 c=121.5 d=0.0 cm)
evt51128   c11 (gid 1000004, t0 1.670 us,  3.8 cm) + c8 (gid 2, t0   6.680 us, 283.9 cm)
             -> gid 2  (far-dominant; a=3.8 b=68.7 c=283.9 d=1.4 cm)
```

Destination t0 = 276.65 µs and 6.68 µs — both outside the beam window. The
length-based a/b/c/d rule handed the joined crosser to the **cosmic** bundle,
which is worse for PR than leaving it split. The cause is structural: this pass
runs *before* `examine_bundles`, so the beam-side donor can still be a 3.8 cm stub
the flash-collapse has not yet glued to its ~290 cm sibling.

`rescue_dest_beam_for_new` forces the beam destination for pairs admitted **only**
by an F1/F2/F3 path. Legacy pairs keep a/b/c/d byte-identical, so the pr/14 hand
scan (4 moves in, 4 out) is not re-opened. With it on, all nine destinations are
in-beam.

### 5.1 Result on the signal set

`cbr_join_metric.py`, OFF arm vs ON arm (PR-free: it q-joins `img-global` ↔
`clustering-global` in the Q/L zip and asks whether the partner's img cluster and
the main's now land in the same all-APA cluster, then cross-checks the
destination t0 against the rescue's own log):

| | OFF arm | ON arm |
|---|---|---|
| JOINED-BEAM | 0 | **8** |
| JOINED-COSMIC | 0 | 0 |
| NOT-JOINED | 10 | 2 |

(the metric covers the 10 events carried in the doc-72 products; 281165 also fires
`new-path-beam`, giving **9 of 12**.)

### 5.2 Per-knob ablation

Each knob alone, F4 on throughout:

| knob | events fixed |
|---|---|
| `rescue_allow_in_beam_far` (F1) | 398115, 237798 |
| `rescue_geom_first` (F2) | 78242, 65053, 317427, 281165 |
| `rescue_pierce_test` (F3) | 65289, 51128 |
| F2 **and** F3 together | 319913 |

319913 is the one event no single knob reaches: its pair is admitted by F2 and
then survives F2's strict gate only on F3's piercing evidence (a 9.6 cm stub gives
`tt_hough = 11.2°` against `angle_cut = 10°`, while its piercing agrees to
1.20 cm). F1 and F3 reproduce round 1's classes A and C exactly; F2's set is
smaller than round 1's class B, because two of that class turned out to be F1
events and one is not reachable at all.

### 5.3 Two things the tuning found

**`geom_first_dis` is 8 cm, not round 1's 5 cm.** evt65053's pair is
`dis = 6.68` with `pierce = 2.68` and `tt_hough = 3.5°` — a good crosser cut on the
one quantity the piercing test measures better. Printed as
`far_accept … -> reject (strict)` before the change.

**F2 needs the closest-point pair re-seated in the destination-T0 frame.** The
pair test's header argues that running `Find_Closest_Points` on unshifted
coordinates is harmless because *"a ≤ 2 cm rigid x-shift does not change which
tips face each other at the cathode"*. That holds for the legacy population — but
F2 drops the time window, and `xshift` is then `v_drift · dt0`: **−92 cm** on
evt493439, **−134 cm** on evt78242. At that size the unshifted selection picks a
tip pair in the wrong frame and the whole test evaluates the wrong two points.
evt493439 went from *no pair within `max_dis = 25 cm` at all* to
`dis = 1.94, pierce = 2.01` once seated correctly. `reseat_closest_shifted()` runs
the same alternating nearest-neighbour convergence `Find_Closest_Points` uses,
with each query point translated into the other cluster's frame, seeded from the
unshifted pair — and only for F2-admitted pairs, so nothing else can move.

This is a genuine defect that round 1 did not see, and it would have made F2's
accepts accidental rather than meaningful.

### 5.4 The three events that are still not fixed

| event | printed reason |
|---|---|
| 493439 | correctly seated now (`dis 1.94`, `pierce 2.01`) but the two halves' own directions disagree: `tt_hough 31.0`, `tt_pca 40.0` against `angle_cut 10`. Rejected as `close_fallthrough`. |
| 167964 | fails the HARD gates: far tip **7.15 cm** from the cathode (cut 5), drift **8.04 cm** (cut 8). |
| 167744 | fails the HARD gates: beam tip **5.11 cm** (cut 5), drift **13.03 cm** (cut 8). Outside doc 72's production working point to begin with. |

**493439 is an open question worth the owner's eye.** doc 72's estimator (a 20 cm
PCA at each cathode end, in raw coordinates) called this same pair collinear to
**1.2°**; the rescue's Hough-at-closest-point calls it **31°**, and the global PCA
of the 230 cm far cluster calls it 40°. One of the two estimators is wrong about
a hand-checked event. Accepting it would mean dropping collinearity entirely in
the close regime — the one protection against merging two unrelated tracks that
happen to touch near the cathode — so it is **not** done here. Resolving which
estimator is right is the natural next step, and is cheaper than loosening a cut.

## 6. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **PASS** — 210 cases, 2093 assertions, 0 failed |
| freshness (M1) | `libWireCellClus.so` 08:14:46 > source 08:14:19; new symbols present in the installed lib |
| compiled config, knobs OFF | **byte-identical** to `git archive HEAD cfg` — `cmp` on the full 54200-byte compiled JSON |
| compiled config, knobs ON | all four keys present, `pierce_cut` threads through as 70 (= 7 cm at `SBND_RESCUE_PIERCE_CUT=7`) |
| knob-off Q/L output, 12 signal events | **byte-identical** to the `work-mcp1k-cb0805` / `work-mcp2k-cb0816` production arms — member-content hash (`hash_archive`), 12/12 |
| firing census, mcp1k 1000 events, OFF vs ON, one binary | **6 new firings, 0 legacy firings lost, 992/1000 byte-identical** — §6.1 |

The knob-off hash gate is stronger than it needs to be: it shows both that the
restructuring left the legacy accept path alone **and** that the current binary
(which carries doc pr/90 from a concurrent session) reproduces the production Q/L
output exactly.

### 6.1 Firing census — mcp1k, 1000 events, OFF vs ON from one binary

This is the real false-positive bound (§4(d)); the 162-event contrast set is not.
Arms `work-cbr2-c1koff` / `work-cbr2-c1kon`.

| | events |
|---|---|
| OFF arm firing (the legacy pr/14 population) | **8** — 56463, 59003, 169824, 288952, 352365, 392200, 395148, 398690 |
| ON arm firing | **14** |
| **new** (ON only) | **6** — 65289, 65053 (the two doc-72 signal events in mcp1k) + 169758, 291064, 395060, 486907 |
| **lost** (OFF only) | **0** |
| byte-identical OFF vs ON (member-content hash) | **992 / 1000** |

The legacy population is reproduced exactly — the 8 OFF firings are the pr/14
hand-scan set — and **nothing the legacy path used to do is lost**. Six new
firings in 1000 events (0.6 %) against a prediction of ~2 (the mcp1k half of the
signal set): four events beyond the signal set, all of them in-beam-anchored
merges into the beam bundle.

```
evt169758  c7  (gid 3,       t0 1.376 us,  29.5 cm) + c13 (gid 1000003, t0   1.379 us,  70.0 cm) -> gid 3       (new-path-beam)
evt395060  c5  (gid 15,      t0 1.077 us,  39.4 cm) + c23 (gid 1000006, t0   1.078 us, 129.2 cm) -> gid 15      (new-path-beam)
evt291064  c5  (gid 5,       t0 1.090 us, 164.7 cm) + c24 (gid 1000003, t0 555.872 us, 148.7 cm) -> gid 5       (new-path-beam)
evt486907  c26 (gid 1000006, t0 0.995 us,  51.5 cm) + c12 (gid 3,       t0 989.987 us, 182.2 cm) -> gid 1000006 (new-path-beam)
```

169758 and 395060 are F1 events whose two halves are matched to flashes **3 ns
apart** — the class-A signature (two APAs' views of one scintillation) in its
purest form. 291064 and 486907 are F2 events with the far half matched +556 and
+990 µs away. All four merge into the beam bundle.

Hand-scan set (same event order in both, `clustering-global` is the layer):

* BEFORE: `https://www.phy.bnl.gov/twister/bee/set/788bbab5-5b69-45b0-a657-0c2538528ac3/event/list/`
* AFTER: `https://www.phy.bnl.gov/twister/bee/set/022f34d3-6d3f-453f-aaec-61ebe101032a/event/list/`
  (bee idx 0-3 = 169758, 291064, 395060, 486907)

### 6.2 The two events that moved without a rescue move — resolved

Two of the 1000 (292643, 390182) differ OFF vs ON while the rescue logged **no
move in either arm**. That is the escalation trigger, so it was chased rather
than waved through, and it is **the Q/L chain's own run-to-run nondeterminism**
(the known SBND effect, doc 49 / M4), not a knob effect:

* **292643** — a re-run of the arm with knobs **OFF** reproduced the *ON* arm's
  hash, not the first OFF arm's.
* **390182** — 11 knobs-OFF replicas span **both** hashes (10 × `f4b974…`,
  1 × `b82287…`), i.e. the "ON" result is reachable with every knob off. Per-knob
  replicas: F1 1/1 each hash, F2 2/2 the OFF hash, F3 2/2 the OFF hash.

Neither event contains a rescue move in either arm, so no charge was re-bundled;
the difference is downstream float-level ordering. Worth stating plainly: this
sets the floor on what a Q/L A/B of this size can resolve — **~2 events in 1000
will differ for reasons unrelated to any knob**.

## 7. Bee — before / after

Same event order in both sets, so Bee event N is the same event in each; open two
tabs and flip. Index and the per-event admitting knob:
`bee/cbr2/cbr2-beforeafter.index.txt`.

* **BEFORE** (all four knobs OFF):
  `https://www.phy.bnl.gov/twister/bee/set/5f6b0455-b7ab-4bd0-b32e-71d4021537e2/event/list/`
* **AFTER** (all four ON):
  `https://www.phy.bnl.gov/twister/bee/set/bf5b75dc-2bbf-46bd-989b-e2d304afc4cf/event/list/`

Look at `clustering-global`: in BEFORE the in-beam main stops dead at x = 0, in
AFTER it continues into the other TPC. `img-global` is the raw image and is the
same in both — the beam flash sits at t ≈ 0, so the raw image always showed one
consistent track through the cathode; the defect was only ever in which bundle
Q/L put each half.

| bee idx | event | missing cm | admitted by | dest t0 µs |
|---|---|---|---|---|
| 0 | 398115 | 325 | `rescue_in_beam_far` | 1.857 |
| 1 | 237798 | 141 | `rescue_in_beam_far` | 1.567 |
| 2 | 281165 | 94 | `rescue_geom_first` | 0.861 |
| 3 | 65289 | 88 | `rescue_pierce_test` | 1.746 |
| 4 | 78242 | 74 | `rescue_geom_first` | 1.907 |
| 5 | 65053 | 74 | `rescue_geom_first` | 0.735 |
| 6 | 51128 | 54 | `rescue_pierce_test` | 1.670 |
| 7 | 317427 | 48 | `rescue_geom_first` | 0.912 |
| 8 | 319913 | 9 | `rescue_geom_first` + `rescue_pierce_test` | 0.745 |

Every destination t0 is inside `[0.2, 2.2) µs`.

## 8. What is deliberately not done

* No SBND production flip. All four knobs ship OFF; §6.1 is the evidence the
  owner needs to decide on.
* No retune of `angle_cut`, `conn_far_cut` or the `dt0` window **in place**. They
  are validated against the pr/14 hand scans and nothing here overturns them; the
  round-2 knobs are strictly additive paths beside them.
* No change to where the rescue runs in the pipeline. Running before
  `examine_bundles` is load-bearing (doc 53) — F4 exists precisely because that
  ordering is kept.
* Nothing in `cathode_connect` itself (M10: the production connector stays
  byte-identical; the rescue already forks its geometry by duplication).
* `pierce_cut` has **no independent true-positive sample.** It is set at 8 cm,
  just above the largest piercing distance measured on a genuine signal event
  (5.46 cm, evt65289), and bounded from above by §6.1. An honest efficiency
  sample would be `cathode_connect`'s accepted crosser pairs (the `clus.jsonnet`
  pr/20 comment cites 183 of them); instrumenting that touches a production file,
  so it is raised here rather than done unasked.

## 9. Open items

1. **evt493439's 31° vs 1.2°** (§5.4) — two estimators disagree on a hand-checked
   event. Resolve before considering any loosening of the close-regime
   collinearity gate.
2. **Is class B ours to fix?** For 65053 and 319913 a perfectly good in-beam flash
   exists on the far side and Q/L still matched the cluster hundreds of µs away.
   That is arguably a `beam_pref` (doc 22) failure, and fixing it there would
   remove the need for F2 on those events. Note the ablation moved both of them
   out of round 1's "class B" reading anyway.
3. **Class A is not a light-reconstruction failure at all.** On 398115 and 237798
   both flashes exist, both are in the beam window, and Q/L matched each half to
   its own side's flash correctly — the halves were then left in different bundles.
   On evt237798 both became separately in-beam mains, so PR ran twice on half a
   track. If the two in-beam flashes are the two APAs' views of the *same*
   scintillation, grouping at flash level before Q/L matching is cleaner than
   re-joining the charge afterwards.
4. **Hand-scan the four census events** (§6.1 Bee sets) before any production
   flip. They are the whole cost side of the ledger: 4 events in 1000 that the
   knobs newly touch and that are not in the doc-72 signal set.
5. **PR-side effect is unmeasured** by design this round. The rescued halves reach
   PR as part of the beam bundle main; what the taggers then do with them needs a
   PR round.

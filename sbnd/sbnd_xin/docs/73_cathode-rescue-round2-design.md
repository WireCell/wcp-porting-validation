# 73 — Cathode bundle rescue, round 2: the residual one-sided crossers

**Status: IMPLEMENTED, C++ knobs shipped default OFF — briefly SBND PRODUCTION
ON, now TURNED BACK OFF pending a round 3.**
Toolkit `17a9929a` (code), `8464c354` (the flip), **`f3706c45` (the revert,
owner instruction 2026-08-17: "turn off the four knobs for SBND default …
we need fix these four knobs first")**.

> **Why the revert — §11.** The full PR chain on the 9 events the knobs fix
> removes the neutrino candidate entirely from **5 of them** and breaks a
> previously-good trajectory fit in a 6th. The cause is the round-2 rescue, not
> the PR chain: the join makes the in-beam main long enough to be tagged TGM/STM,
> and the cosmic veto then discards the only in-window main. Two of the five
> (398115, 237798) are legitimate purifications; 51128 and 78242 are not.
>
> The revert is gate-proven **byte-identical** to the pre-round-2 baseline (§11.9),
> and the knobs remain fully usable via
> `SBND_RESCUE_{IN_BEAM,GEOM_FIRST,PIERCE,DEST_BEAM}=1`.

**This is NOT bit-identical.** It is a behaviour change delivered as config. The
escape to the pre-round-2 baseline is
`SBND_RESCUE_{IN_BEAM,GEOM_FIRST,PIERCE,DEST_BEAM}=0`, which omits every key and
restores a byte-identical compiled config (verified by `cmp`, §9).

doc 72 §A found 10 events in 3000 where the in-beam bundle main is cut at the
cathode and the raw image plainly continues into the other TPC.
`ClusteringCathodeBundleRescue` (doc pr/14 + pr/17 + pr/19) already exists to fix
exactly this and is **SBND default ON**, so every one of the 10 is a case where
the existing rescue did not fire. Round 1 of this doc (2026-08-17, design only)
attributed the misses and proposed three extensions. This is round 2: the
implementation, four corrections to round 1's design, and the measurement.

**Result: 9 of the 12 signal events are fixed** (8 of the 10 at doc 72's
production working point), every one of them into the beam bundle, all measured
by a PR-free join metric. Knob-off is byte-identical to the production arms on all
12 (member content hash). Bee before/after in §7.

> The owner hand-scanned the §6.1 census set and found that **2 of the 4 new
> firings were false merges** — a cosmic dragged 33 cm through the cathode.
> §6.3 is the diagnosis, §5.5 the fix (`far_contain_tol`, round 1's condition 3,
> designed and then never implemented), §5.6/§6.4 the validation. Both are now
> byte-identical to production again.
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

# 281165 / 167744 live only in the LOOSENED search, so make its rows first
BCS_GAP2D=25 BCS_ANGLE=30 BCS_P_MIN_LEN=5 BCS_P_MIN_PTS=10 BCS_CATH_NEAR=15 \
BCS_P_CATH_NEAR=12 BCS_DIRX_MIN=0.2 \
python3 scripts/analysis/cathode/beam_cathode_split.py --arm work-mcp2k-cb0816 \
        --out /home/xqian/tmp/cbr2_loose2k.tsv --jobs 8

# did the missing half JOIN the in-beam main?  (PR-free; reads only the Q/L zip)
python3 scripts/analysis/cathode/cbr_join_metric.py \
    --baseline work-mcp1k-cb0805 --baseline work-mcp2k-cb0816 --test work-cbr2-on2 \
    --rows products/beam-cathode-split-mcp1k.tsv \
    --rows products/beam-cathode-split-mcp2k.tsv --rows /home/xqian/tmp/cbr2_loose2k.tsv

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

**Binary provenance.** Every arm in this doc: toolkit HEAD `91b78d67` + the
round-2 patch, built by one `wcbuild` at **08:14:46**. Both arms of every
comparison come from that one binary, so an OFF/ON delta is attributable to these
knobs and to nothing else in the tree. (A concurrent session shares this working
tree and moved HEAD `a681b3e1` → `91b78d67` mid-round; hence the explicit record.)

**Correction, and it matters for every freshness proof in this tree.** An earlier
draft of this paragraph cited `local/lib/libWireCellClus.so`, following CLAUDE.md
M1 (*"wire-cell loads plugins from `local/lib`, not `build/`"*). **That is not
what happens here.** direnv puts the `build/*` directories *ahead* of `local/lib`
in `LD_LIBRARY_PATH`, and `/proc/<pid>/maps` on a live job shows

```
/nfs/data/1/xqian/toolkit-dev/toolkit/build/clus/libWireCellClus.so
```

so the **build tree** is what is actually mapped. Two consequences:

* a freshness proof must stat `build/clus/libWireCellClus.so`; stat-ing
  `local/lib` checks a file the job may never open;
* `./wcb build` **alone** is enough to swap the library under a running campaign —
  `install` is not required to break one. M1's remedy (always `wcbuild`) is still
  right, but its stated reason is incomplete.

Nothing in this doc is invalidated: `wcbuild` writes both trees in one
invocation, so the 08:14:46 `local/lib` timestamp and the `build/clus` library the
arms actually loaded came from the same compile. The 08:14:46 `build/clus`
library no longer exists on disk — a concurrent session rebuilt over it at
09:01:50, after every arm here had finished (the last, the mcp1k census, at
~08:50). That is also what killed the §8 `pierce_cut` 10/12 cm arms.

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
| JOINED-BEAM | 0 | **9** |
| JOINED-COSMIC | 0 | 0 |
| NOT-JOINED | 12 | 3 |

All twelve are measured by the join metric, not by the rescue log. 281165 and
167744 are not in the doc-72 products (they came from the loosened working
point), so the loosened search is re-run and fed in as a third `--rows` file —
otherwise those two would have been scored on the *weaker* evidence (a beam
destination in the log) while §5 makes exactly that distinction its centrepiece.
Their rows: `281165 main 16 p_cid 8 p_ext 94.4`, `167744 main 13 p_cid 12
p_ext 99.6`.

Two ways of reporting the numerator, both honest:

* **8 of 10** at doc 72's production working point (the set doc 72 §A reports as
  10 events in 3000);
* **9 of 12** including 281165 and 167744, which are genuine but were lost to doc
  72's own search cuts rather than to the reconstruction.

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

### 5.5 The containment veto — round 1's condition 3, restored

**Status: IMPLEMENTED and VALIDATED** (pointed validation §5.6, census §6.4).
Toolkit `17a9929a`, build 09:20:47 (`build/clus/libWireCellClus.so`).

`far_stays_in_tpc()`: for a pair admitted by any round-2 path, every point of the
**far half**, shifted into the destination-T0 frame, must stay on its own side of
the cathode within `far_contain_tol`. The side is derived from the same `dirx`
`far_xshift()` already uses (`SIGN = -dirx`, so a face's volume is the side where
`(x - cathode_x) * dirx < 0`), so it introduces no new geometry assumption.

**`far_contain_tol = 1 cm`, and it is not a resolution tolerance.** Measured over
**3463 per-APA cluster parts in 300 production events**, legitimate charge never
crosses the cathode *at all* — max overshoot **0.00 cm**. x is derived from drift
time inside the cluster's own wire volume, so it is hard-clipped at the plane.
Anything past the cathode is therefore `v_drift × (T0 error)`, and the tolerance
converts directly into a bound on how wrong the destination-T0 hypothesis may be:

| `far_contain_tol` | admits a T0 error up to |
|---|---|
| 1 cm | 6.4 µs |
| 3 cm | 19 µs |

Since `rescue_geom_first` exists *precisely* to admit large-|dt0| pairs, this is
the main thing bounding that hypothesis, so tighter is strictly better here — 3 cm
would pass any false merge whose T0 error is under 19 µs. (An earlier draft used
3 cm on "it sits in the measured gap"; the owner pushed for 1 cm, and the
production distribution above says the owner is right.)

Predicted effect, from the measured post-merge overshoot of every round-2 firing
(the merged-cluster maximum, which is an upper bound on the far half's own):

| event | overshoot past the cathode | verdict at 1 cm |
|---|---|---|
| 398115, 237798, 281165, 65289, 78242, 65053, 317427, 319913 | −1.65 … +0.05 cm | keep |
| 51128 | **+0.55 cm** | keep (0.45 cm margin) |
| 169758, 395060 | −0.30, −0.02 cm | keep |
| **291064** | **+33.46 cm** | **reject** |
| **486907** | **+33.79 cm** | **reject** |

So the veto is predicted to remove exactly the two false merges and keep all nine
fixes. evt51128 carries the thinnest margin and is the one to watch when the arm
runs. Note the 11-event sample is small — the tracer prints the measured overshoot
on every rejection, so a clipped genuine rescue shows up as a printed line rather
than as a silent loss.

### 5.6 Pointed validation of the containment veto

13 events — the 9 fixes, the 2 false merges, and the 2 clean census firings —
knobs all on, arm `work-cbr2-contain`. Build 09:20:47 (`build/clus`, the file
actually mapped; see the provenance note in §1).

| check | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **PASS** — 210 cases, 2101 assertions |
| events still firing | **11** — the 9 fixes + 169758 + 395060 |
| containment rejections | **2**, both printed |
| join metric | **JOINED-BEAM 9 / 9** |
| any cluster crossing the cathode | **none** |

The veto prints what it measured, so a rejection is never silent:

```
[cbrsel] c12 far half leaves its own TPC by 33.3 cm under the destination T0 (tol 1.0) -> reject   (evt486907)
[cbrsel] c24 far half leaves its own TPC by 33.0 cm under the destination T0 (tol 1.0) -> reject   (evt291064)
```

Per-event overshoot past the cathode, before and after the veto — the two false
merges collapse from 33 cm to −0.10 cm (i.e. inside their own TPC), and **not one
of the eleven good firings moves by so much as 0.01 cm**:

| event | before | after |
|---|---|---|
| 398115 / 237798 / 65289 / 78242 / 65053 / 317427 / 319913 / 169758 / 395060 | −1.65 … −0.02 | unchanged |
| 281165 | +0.05 | +0.05 |
| 51128 | +0.55 | +0.55 |
| **291064** | **+33.46** | **−0.10** |
| **486907** | **+33.79** | **−0.10** |

The prediction in §5.5 was made before the build and is reproduced exactly,
including that evt51128 keeps the thinnest margin (0.45 cm to the 1 cm bound).

### 5.7 Why the pointed validation was enough

The owner's call, and it is right: the veto can only ever *reject* a pair on the
round-2 path, so it cannot create a firing, cannot touch the legacy path, and
cannot alter a knob-off run. Its whole possible effect on the mcp1k census is to
remove firings from the six already enumerated, and all six were run explicitly
above. §6.1's re-run is therefore a confirmation, not a discovery.

Run anyway, because the lesson of §6.3 is that reasoning about what code *can* do
is exactly what missed a physical impossibility once already.

## 6. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **PASS** — 210 cases, 2093 assertions, 0 failed |
| freshness (M1) | `libWireCellClus.so` 08:14:46 > source 08:14:19; new symbols present in the installed lib |
| compiled config, knobs OFF | **byte-identical** to `git archive HEAD cfg` — `cmp` on the full 54200-byte compiled JSON |
| compiled config, knobs ON | all four keys present, `pierce_cut` threads through as 70 (= 7 cm at `SBND_RESCUE_PIERCE_CUT=7`) |
| knob-off Q/L output, 12 signal events | **byte-identical** to the `work-mcp1k-cb0805` / `work-mcp2k-cb0816` production arms — member-content hash (`hash_archive`), 12/12 |
| firing census, mcp1k 1000 events, OFF vs ON | **4 new firings with the veto, 0 legacy firings lost, 988/988 non-firing events byte-identical** — §6.4 |
| containment veto, pointed validation | **PASS** — 9/9 fixes kept, both false merges rejected, nothing crosses the cathode — §5.6 |

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

> **291064 and 486907 are FALSE MERGES.** The owner hand-scanned this set and
> caught it in one look; see §6.3. The fix is §5.5. Do not read the four as a
> single benign population.

Hand-scan set (same event order in both, `clustering-global` is the layer):

* BEFORE: `https://www.phy.bnl.gov/twister/bee/set/788bbab5-5b69-45b0-a657-0c2538528ac3/event/list/`
* AFTER: `https://www.phy.bnl.gov/twister/bee/set/022f34d3-6d3f-453f-aaec-61ebe101032a/event/list/`
  (bee idx 0-3 = 169758, 291064, 395060, 486907)

### 6.4 Census with the containment veto — the final numbers

mcp1k 1000 events re-run with the veto on (`work-cbr2-c1kon2`, build 09:20:47):

| arm | firing events | new vs OFF |
|---|---|---|
| OFF (legacy pr/14 path) | 8 | — |
| ON, no containment veto | 14 | 65053, 65289, 169758, 395060, **291064**, **486907** |
| **ON, with the veto** | **12** | 65053, 65289, 169758, 395060 |

The veto removes **exactly** the two false merges, and **no legacy firing is
lost**. So the round's cost side over mcp1k is **4 new firings in 1000 events**,
of which two are doc-72 signal events and two (169758, 395060) are the clean
class-A pairs whose halves are matched 3 ns apart.

**The complement check — the one that would have caught §6.3.** Hash every event
OFF vs ON and require that any event *without* a rescue log line be byte-identical:

| | byte-identical | differs |
|---|---|---|
| events that fire | 8 | 4 |
| events that do NOT fire | **988** | **0** |

Zero quiet-but-changed. That assertion is worth more than its result: §6.3's bug
lived precisely in the gap between "the checks I print" and "everything else", and
a gate that only inspects the pairs it logs can never close it. (The previous
census had 2 quiet-but-changed events, both traced to the chain's run-to-run
nondeterminism — §6.2; this run happened to reproduce them.)

### 6.3 Two of the four new firings are FALSE MERGES (owner, hand scan)

**A cluster's charge sits on the wires of one TPC, so its x cannot cross the
cathode.** The owner read the §6.1 Bee set and saw exactly that: a track from one
TPC running through the cathode plane after the merge. Measured:

| event | far half | its t0 | x under its own t0 | x after the merge | past the cathode |
|---|---|---|---|---|---|
| 486907 | APA0, 1826 pts | 989.99 µs | [−201.3, −121.2] | **[−87.3, +33.3]** | **33.8 cm into TPC1** |
| 291064 | APA1, 1370 pts | 555.87 µs | [+53.7, +180.3] | **[−33.0, +93.6]** | **33.5 cm into TPC0** |

Both are F2 (`rescue_geom_first`) merges. The give-away is that each far half's
**raw** x (t0 = 0) already runs past the cathode — only possible if its true t0 is
genuinely hundreds of µs positive. They are cosmics, and forcing the beam T0 on
them is unphysical. They are not missing halves at all.

**Root cause, and it is an omission in the implementation, not in the design.**
Round 1's F2 listed four conditions; condition 3 was *"the far half's corrected x
under the beam T0 must lie inside its own TPC (all 12 candidates satisfy this; it
is a cheap veto against absurd merges)"*. §5 records dropping condition 4 and why.
**Condition 3 was never implemented and its absence was never recorded.** The
geometric test only ever inspects the two TIP points; the merge then
re-materializes the WHOLE far cluster under the destination T0, and nothing
re-checked it.

**Why every gate in §6 missed it.** The byte-identical gate, the firing census,
the per-knob ablation and the join metric all passed. Each asks *"did the output
change in the way I expected?"*. None asks *"is the output physically possible?"*.
That is the general lesson of this round: **check the object the code modifies
(the whole cluster), not the object you reasoned about (the two tips)** — and
prefer a check a wrong answer cannot survive over a check that merely looks
unsurprising.

The nine fixed events of §5.1 are all clean (§5.5 table), so the fix removes the
two false merges without touching them.

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

  Swept **downward** on the signal set, which is the direction that matters (the
  census bounds it from above):

  | `SBND_RESCUE_PIERCE_CUT` | events firing / 12 |
  |---|---|
  | 6 cm | 9 |
  | 7 cm | 9 |
  | 8 cm (default) | 9 |

  The fixed set is flat over 6–8 cm, i.e. the operating point is not perched on a
  threshold. **The 10 cm and 12 cm arms are not reported**: they were launched
  just as a concurrent session began a `wcb install`, and every job in them died
  on `failed to load plugin: WireCellRoot` (rc≠0 on 12/12 and 6/12). They are not
  evidence of anything and were not re-run, because re-running would have used a
  different binary from every other arm in this doc.

## 9. Open items

1. **evt493439's 31° vs 1.2°** (§5.4) — two estimators disagree on a hand-checked
   event. Resolve before considering any loosening of the close-regime
   collinearity gate.
2. **Is class B ours to fix?** For 65053 and 319913 a perfectly good in-beam flash
   exists on the far side and Q/L still matched the cluster hundreds of µs away.
   That is arguably a `beam_pref` (doc 22) failure, and fixing it there would
   remove the need for F2 on those events. Note the ablation moved both of them
   out of round 1's "class B" reading anyway.
3. **Class A is not a light-reconstruction failure at all, and the census makes
   the case much sharper.** On 398115 and 237798 both flashes exist, both are in
   the beam window, and Q/L matched each half to its own side's flash correctly —
   the halves were then left in different bundles. On evt237798 both became
   separately in-beam mains, so PR ran twice on half a track.

   The strongest evidence is in §6.1, not in the signal set: two of the four new
   census firings (169758, 395060) have their two halves matched to flashes
   **3 ns apart** (1.376/1.379 µs and 1.077/1.078 µs). Two flashes 3 ns apart are
   not two scintillations — they are the two APAs' views of one. Grouping them at
   flash level *before* Q/L matching would dissolve this whole class, and would be
   cleaner than re-joining the charge afterwards as F1 does.
4. **Hand-scan the four census events** (§6.1 Bee sets) before any production
   flip. They are the whole cost side of the ledger: 4 events in 1000 that the
   knobs newly touch and that are not in the doc-72 signal set.
5. **PR-side effect is unmeasured** by design this round. The rescued halves reach
   PR as part of the beam bundle main; what the taggers then do with them needs a
   PR round.


## 10. The production flip (owner decision, 2026-08-17)

All four knobs flipped to SBND production ON in
`cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` (the TLA block
that IS the SBND operating point, doc 68) and in `sbnd/clus.jsonnet`'s
`clus_all_apa` / `all_apa` defaults, so a caller that threads nothing inherits
production. `rescue_pierce_cut` stays `null` ⇒ the C++ default 8 cm, which is the
validated point (§8: flat over a 6–8 cm sweep).

**Gates on the flip itself:**

| gate | result |
|---|---|
| compiled config, bare production defaults | all four keys present and `true` |
| compiled config, escape (`…=0` on all four) | **byte-identical** to the pre-round-2 baseline (`cmp` on the full compiled JSON) |
| bare run == the validated arm | **4/4 member-content hashes match** |

The bare-run check is the one that matters for doc 68's "the operating point lives
only in cfg, so a bare run *is* production" invariant. Running with **no**
`SBND_RESCUE_*` environment at all:

| event | moves | matches |
|---|---|---|
| 65289 | 1 | the all-knobs-on arm, hash `b727216a…` |
| 51128 | 1 | the all-knobs-on arm, hash `a8781cc1…` |
| 291064 | 0 | the knobs-**off** arm, hash `cce0fcea…` (containment veto holds) |
| 486907 | 0 | the knobs-**off** arm, hash `9b42fa64…` (containment veto holds) |

So production now reproduces the validated behaviour exactly, including declining
the two false merges.

**What production gains and costs**, from §5.1 and §6.4:

* **gains** 9 of 12 one-sided crossers rejoined, every one into the beam bundle;
* **costs** 4 new firings per 1000 events on mcp1k (2 of them the doc-72 signal
  events 65289/65053; the other 2, 169758 and 395060, are class-A pairs whose
  halves are matched 3 ns apart), 0 legacy firings lost, and all 988 non-firing
  events byte-identical.

**Not carried by the flip, and still open** (§5.4, §9 items): 493439, where doc
72's estimator and the rescue's disagree (1.2° vs 31°) on the same hand-checked
event; 167964 and 167744, which fail the hard tip/drift gates. And the PR-side
effect of the rejoined halves is unmeasured — this round was Q/L-only by owner
direction, so what the taggers do with the recovered charge needs a PR round.

**That PR round is §11, and it is ADVERSE. Read it before relying on the flip.**

---

## 11. The PR round — the flip costs the neutrino in 5 of 9 events

Owner request 2026-08-17, after the flip: *"run the PR chain on them and provide
me the bee link — I want to confirm the PR state for these events."* Then, on
seeing the displays, three specific questions (65289, 78242, 51128) and the
diagnosis *"these problems were not PR problems, but with the previous fix in the
cathode crossing tracks."* **That diagnosis is correct.** Every finding below is
caused by the round-2 rescue, not by anything in the PR chain.

**These are DATA events** (owner). There is no truth, and no guarantee any of
them contains a neutrino.

### 11.1 Repro

```bash
cd .../sbnd/sbnd_xin
EVTS="398115 237798 281165 65289 78242 65053 51128 317427 319913"
# Q/L with the pctree the PR chain reads (round 2 never wrote it: QL_EXTRA is new)
ROOT=$PWD/work-cbr2-prql-off QL_EXTRA=-save-pctree \
  SBND_RESCUE_IN_BEAM=0 SBND_RESCUE_GEOM_FIRST=0 SBND_RESCUE_PIERCE=0 \
  SBND_RESCUE_DEST_BEAM=0 ./run_ql_batch.sh -j 6 $EVTS
ROOT=$PWD/work-cbr2-prql-on  QL_EXTRA=-save-pctree ./run_ql_batch.sh -j 6 $EVTS
PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-cbr2-prql-off work-cbr2-pr-off data $EVTS
PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-cbr2-prql-on  work-cbr2-pr-on  data $EVTS
```

Both arms from ONE binary: toolkit HEAD `8464c354`,
`build/clus/libWireCellClus.so` 2026-08-17 09:42:58, no clus source newer. 9/9
`rc=0` in both arms. Bee: `bee/cbr2/cbr2-pr-beforeafter.index.txt`.

Gates on the Q/L inputs, member-content hash (`hash_archive.py`), **9/9 each**:
OFF-arm `pctree-evt<ID>.tar.gz` == production `cb0805`/`cb0816`; OFF-arm mabc ==
production; ON-arm mabc == the validated `work-cbr2-contain` arm. The pctree gate
is the one that matters — it is the file `run_pr_chain_batch.sh` actually reads,
and gating only `mabc-all-apa.zip` would have checked the display instead of the
PR input. *Instrument note:* `hash_archive.py` prints
`<content-hash> <size> <path>`; hashing whole lines compares filenames and fails
9/9 for the wrong reason.

### 11.2 The headline

**5 of 9 events lose the neutrino candidate entirely** — `TaggerCheckNeutrino`
selects nothing where it previously selected a main.

| event | join: beam cm + far cm @ far t0 | merged main L | tag | PR selection OFF → ON |
|---|---|---|---|---|
| 398115 | 75.0 + 342.9 @ **0.529 µs (in beam)** | 414.7 | TGM | L 341.9 → **none** |
| 237798 | 252.6 + 164.8 @ **0.393 µs (in beam)** | 417.2 | TGM | L 252.4 → **none** |
| 317427 | 286.7 + 143.6 @ 109.1 µs | 429.1 | TGM | L 286.7 → **none** |
| 51128 | **3.8** + 283.9 @ 6.68 µs | 287.1 | TGM | L 57.7 → **none** |
| 65289 | 160.2 + 88.9 @ −2.77 µs | 248.1 | STM | L 159.2 → **none** |
| 281165 | 21.6 + 121.5 @ 276.7 µs | 138.4 | — | L 19.2 → L 138.4 |
| 78242 | 201.8 + 149.7 @ **857.2 µs** | 314.4 | — | L 166.8 → L 314.4 |
| 65053 | 46.9 + 94.2 @ 581.6 µs | — | — | L 113.9 → L 113.9 (unchanged) |
| 319913 | 55.0 + 9.6 @ 28.9 µs | 64.4 | — | L 55.0 → L 64.4 |

**Correction to §10 and to the first cut of the Bee index.** The `label` column
of `nusel-evt<ID>.tsv` (`nu-candidate` / `TGM` / `STM` / `no-bundle`) and
`TaggerCheckNeutrino`'s selection are **two different notions of "main"** — the
`run_pr_chain_batch.sh` header says so explicitly ("the T_tagger row is NOT
joinable to a nusel-evt<ID>.tsv main_id", because `unmerge_bundle` renumbers).
Reading `label` as the PR outcome mis-scores two events: 65053 shows
`nu-candidate → STM` in `nusel` while `TaggerCheckNeutrino` selects the *same*
main `6 (L 113.9 cm)` in both arms, and 281165 shows `no-bundle` while PR does
select a neutrino — its bundle simply moved from flash gid 0 (APA0) to gid
1000007 (APA1), the same ~0.86 µs scintillation seen by the other APA. Score PR
outcomes on the `selected main cluster` log line, never on `label`.

### 11.3 The single mechanism behind all five losses

Identical chain every time, and none of its links is a PR defect:

1. the rescue joins the two halves, so the in-beam main gets **longer**;
2. at 248–429 cm the object reaches through-going or stopping-muon size (SBND is
   ~200 cm of drift per TPC, so a genuine cathode crosser tops out near 400 cm);
3. `TaggerCheckTGM`/`TaggerCheckSTM` tag it cosmic — **correctly, for the object
   they are shown**;
4. `nu_skip_cosmic` skips that main, and it is the **only** in-window main;
5. `TaggerCheckNeutrino: no main cluster selected` → no neutrino at all.

The rescue's whole purpose is to make the in-beam main longer. Step 2 is
therefore not an accident of these 9 events; it is what the knobs do.

### 11.4 The owner's three questions

**Q1 — 65289: two clusters in the bundle, one is STM; why is the other
discarded? Is it because it is not identified as main? And is this a bug the
four knobs introduced, or was it always there?** *Yes to the first; and it is
**pre-existing behaviour, not a regression from the knobs**.*

The two are cluster **13** (the merged main, L 248.1 cm, `STM=1`) and cluster
**18**, a **demoted main** — a cluster that was a bundle main before the flash
merge. `evaluate_demoted_mains` (doc pr/20 Part I, P3) re-adds cluster 18 to the
*tagger* evaluation list, and it is scored there: `TGM=false`, `STM=0`,
`FC=true`. But `TaggerCheckNeutrino`'s candidate loop opens with
`if (!cluster->get_flag(Flags::main_cluster)) continue;`
(`clus/src/TaggerCheckNeutrino.cxx:808`), so a demoted main is never a
*candidate*. The log's `14 mains, 1 in-window` counts cluster 13 alone.

**The multi-main machinery the owner remembers does exist and is working.** Same
loop, lines 807–870: `nu_skip_cosmic` is applied **per main**, with the comment
saying so explicitly — *"Per-main, so a cosmic-tagged longest bundle does not
veto a clean runner-up"* — and when two in-window mains survive the vetoes it
keeps the longer and logs `in-window bundle cluster N ... not selected`. There is
also a real safety valve: `nu_skip_cosmic_bundle_min_length`, doc pr/16 design A,
set to **15 cm in SBND production** (`cfg/.../sbnd/clus.jsonnet:927`), which
spares an untagged in-window main ≥ 15 cm from the bundle veto.

None of that is touched by the four knobs. 65289 simply has **one** in-window
main, and the merge turned that one into an STM; there was no runner-up to fall
back to. The pre-existing asymmetry it exposes — the taggers can see demoted
mains (P3) but the neutrino selector cannot — predates this round and is
unchanged by it.

**But note what this means for 51128 (Q3):** the 15 cm guard is exactly the
safety valve for "a cosmic-tagged bundle-mate should not kill a real neutrino",
and it **could not fire**, because the guard only applies to clusters that are
still `main_cluster`. The merge demoted the 57.7 cm neutrino out of main status
*before* the guard was reachable. So the knobs did not break the protection —
they stepped around it. The two clusters that did reach the guard were 1.2 cm,
far under 15.

**Q2 — 78242: `track_fit-global` is missing part of the neutrino, an EM shower
and part of the long track. Owner: "I thought the rescue would ADD the activity
from the other TPC — why is something missing?"** Both are true at once: the far
half *was* added, and a stretch that used to be fit is now unfit. The charge is
not lost — the fit is. Resolved by segment (`real_cluster_id`) rather than by
bin, which is what makes it legible:

| arm | `track_fit` segments of the main cluster |
|---|---|
| OFF (cid 8) | `8007` x −102.2…−90.3 · `8005` x −90.3…−54.8 · **`8004` x −54.8…−1.4 (169 pts)** · `8006` x −54.8…−50.1 |
| ON (cid 17) | `17017` x −102.2…−90.4 · `17007` x −90.4…−54.2 · **(nothing)** · `17011` x +15.5…+16.6 · **`17012` x +16.3…+67.5 (206 pts, the added far half)** |

So the join **did** add the other TPC: segment `17012`, 1489 charge points and 206
fit points over x +1.2…+67.5, which did not exist before. What vanished is
OFF's segment `8004` — the 54 cm of the beam-side track running up to the cathode
— and it has no counterpart in the ON arm at all.

**The charge in that stretch is untouched and still labelled *track*.** In
`shower_track` the beam-side segment is essentially identical between arms
(`8005` x −90.6…−0.9, 1494 pts → `17007` x −90.6…−0.9, 1498 pts), and a
track/shower split by `q` shows 472/288/316 **track** points and **zero** shower
points in the x = −60…0 bins *in both arms*. The newly attached far half is the
opposite: 380/550/414/145 points at x > 0, **all shower, zero track**.

Net effect: the unfit region is roughly **x ∈ [−54.8, +16.3]**, about 71 cm
straddling the cathode — i.e. **the fit no longer crosses the junction it was
created to make.** It fits the far half, it fits the upstream part of the beam
half, and it drops the join. Re-running pattern recognition on the merged cluster
re-segmented it and did not reproduce `8004`.

(The EM shower's absence from `track_fit` is by design — `track_fit` holds fitted
trajectories, showers live in `shower_track`. The *long-track* hole is the
defect.) This event's far half is matched **857 µs** away.

*Correction to an earlier reading of this event:* the beam-side loss is **not**
PR "reclassifying the material as shower" — that happens only to the newly
attached far half. The beam-side points keep their track label and lose only
their trajectory segment.

**§11.8 carries the resolved diagnosis** — the join here is geometrically
excellent (a dead-straight line through the cathode), and the defect is that
**100 % of the rescued far half is labelled EM shower** while the beam half of
the same muon stays 100 % track. Read §11.8 before this table; an intermediate
answer that blamed the join was retracted there.

**Q3 — 51128: the neutrino is gone; should the fix consider clusters other than
the beam-flash-matched one?** *Yes — this is the clearest defect of the round,
and the owner's reading is exactly right.* The beam-side donor `c11` is a
**3.8 cm fragment**, not the bundle's neutrino, which is a *different* cluster
(L 57.7 cm, `selected main cluster 22` in the OFF arm). Sequence:

1. rescue joins `c11` (3.8 cm, beam) to `c8` (283.9 cm cosmic, t0 6.68 µs);
2. F4 `rescue_dest_beam_for_new` puts the 287.1 cm result in the **beam** bundle;
3. longest-cluster-wins makes it the bundle **main**, demoting the real 57.7 cm
   neutrino to associated;
4. `TaggerCheckTGM` tags the merged main → `nu_skip_cosmic` skips it, and
   `nu_skip_cosmic_bundle` then skips the two remaining in-window clusters
   (`L 1.2 cm` each) *because they share flash bundle gid 1000004 with a
   cosmic-tagged main*;
5. `no main cluster selected (23 mains, 3 in-window)`.

The real neutrino is not even in the skip log — demotion removed it from the
candidate list before the veto ran. A 3.8 cm fragment was allowed to decide the
fate of a 57.7 cm neutrino interaction.

### 11.5 Root cause, stated once

The rescue picks the beam-side donor by **cathode geometry alone**. It never asks
(a) whether that donor is the bundle's main or a minor fragment, nor (b) what
attaching a large far half does to the *rest* of the destination bundle. F4 then
forces the result into the beam bundle, where "longest = main" hands the bundle
to the newly created cosmic and the bundle-level veto discards everything else
in-window.

This is the same failure shape as the false-merge bug of §6.3, one level up:
there, the geometry inspected two tip points while the merge re-materialised a
whole cluster; here, the geometry inspects two clusters while the merge
re-organises a whole *bundle*. **Check the object the code modifies.** The
round-2 gates could not have caught it — the join metric (§5.6) scored "did the
missing half end up in the beam bundle", which is *true* in all 9 cases and is
precisely how the neutrino gets killed.

**The assertion that would have caught it, for the next round:** a Q/L-stage
change must be scored on **"does PR still select a neutrino main?"**, never on
"did the clusters merge as intended". Merging as intended is the *premise* of the
damage here, so no amount of Q/L-side checking can see it. This is the PR-stage
twin of §6.3's complement assertion (every event with no log line must be
byte-identical): both work by checking an object the change was not reasoning
about. Cost is small — the PR chain is ~20 s/event on an existing pctree, so the
whole 12-event census is minutes. Round 2 skipped it only because PR was out of
scope by owner direction, which was the right call at the time and is why this
sat undetected for a day rather than being a gate failure.

### 11.6 What is and is not a win

Not all five losses are wrong. The two events whose far half is **itself in the
beam window** — 398115 (0.529 µs) and 237798 (0.393 µs) — are genuine
cathode-crossing muons in the beam window; joining them is right, 415/417 cm is a
through-going muon, and TGM is the correct verdict. For those the pre-fix
`nu-candidate` was the artifact and the flip is a **purification**.

The rest need a hand scan, and two are already indefensible: 51128 (a 3.8 cm
donor kills a 57.7 cm neutrino) and 78242 (fit broken over 40 cm, far half
matched 857 µs away). 317427, 65289, 281165 and 65053 join far halves whose own
flashes are 109 / −2.8 / 277 / 582 µs away — `rescue_geom_first` asserts those
flash matches were all wrong, which is a strong claim with no independent
evidence behind it.

### 11.7 Recommendation (owner call — §5 rules 1 and 7)

**Consider setting the four SBND production defaults back OFF pending a round 3.**
The flip is currently ON in production and, on the only 9 events where its
behaviour has been examined end-to-end, it removes the neutrino candidate from 5
and demonstrably damages 2 more.

**The production harm rate is UNMEASURED — do not read "1%" out of this.** The
rescue fires on 12 events per 1000 (mcp1k), but that is the *firing* rate, not
the harm rate; the harm rate is 12/1000 × (fraction of firings that are harmful),
and the second factor is unknown. These 9 events are doc 72's **signal** set,
selected precisely because a beam bundle was visibly cut at the cathode — i.e.
enriched for the exact condition that drives the mechanism of §11.3 — so 5/9 is
not an estimate of anything in a random sample. The 12 census firings include 4
events never scanned end-to-end (169758, 395060, 291064, 486907). Measuring the
real rate means running the PR chain over the full census, both arms.

Escape, no rebuild needed: `SBND_RESCUE_{IN_BEAM,GEOM_FIRST,PIERCE,DEST_BEAM}=0`.

Round-3 directions, in the order the evidence supports them:

1. **Do not let a rescued join displace a bundle's existing main.** If the
   destination bundle already has a longer, in-window, non-cosmic main, the
   merged object should not take the main slot (51128, and the mechanism behind
   every loss).
2. **Require the beam-side donor to be substantial** — a `min_beam_donor_len`
   guard would decline 51128 (3.8 cm) and 281165 (21.6 cm) outright.
3. **Do not demote a longer, untagged in-window main below a rescued join.**
   SBND already runs the doc pr/16 design-A guard at 15 cm, which exists to stop
   a cosmic bundle-mate killing a real neutrino; on 51128 it never got the
   chance, because demotion removed the 57.7 cm neutrino from `main_cluster`
   before the guard could see it. Preserving main status is a Q/L-side fix in
   *this* component and is preferable to changing `TaggerCheckNeutrino` — no
   PR-chain change is needed, and §11.4 Q1 shows the selector logic is sound.
   (A separate, genuinely PR-side question — should the neutrino selector see
   demoted mains the way the taggers already do via P3? — predates this round;
   raise it with whoever owns that chain rather than folding it in here.)
4. **Bound `rescue_geom_first` by |dt0|.** Unlimited, it asserts that a flash
   match hundreds of µs away is wrong; 78242 (857 µs) and 65053 (582 µs) are the
   cases to test a cap against.
Bee for the three: `311a2d2e-1dac-4fb5-bd91-05685a6d8184`.

### 11.8 Two owner follow-ups

**"Should `TaggerCheckNeutrino` also examine the demoted main?" (65289)** —
There is a real case for it, and the two events where the neutrino is lost both
have a demoted main that looks like a candidate:

| event | demoted main | verdicts | note |
|---|---|---|---|
| 65289 | cluster 18 | `TGM=false`, `STM=0`, **`FC=true`** | in-window, contained, ~118 cm chord |
| 51128 | cluster 28 | `TGM=false`, `STM=0`, **`FC=true`** | in-window, contained |
| 51128 | cluster 27 | `TGM=false`, `STM=0`, `FC=false` | 64.5 cm chord |

`FC=true`, in the beam window, and untagged by both cosmic taggers is exactly
the profile of a neutrino candidate. The verdicts already exist — doc pr/20 P3
computes them via `evaluate_demoted_mains` — so the information is free; only the
`Flags::main_cluster` gate at `TaggerCheckNeutrino.cxx:808` keeps them out. And
the admission pattern is already established: the doc pr/16 design-A guard
admits an untagged in-window main ≥ 15 cm past the bundle veto, which cluster 27
would clear. So the asymmetry (taggers see demoted mains, the selector does not)
is arguably an oversight rather than a design.

**But it should be a separate, independently-gated change, and it should come
after the cathode fix, not as part of it.** Three reasons:
* **Blast radius.** Demoted mains exist in most events, not just the ~12 per 1000
  where the rescue fires. This changes candidate selection detector-wide and needs
  its own default-OFF knob plus a full census gate — a much larger commitment than
  the four cathode knobs.
* **It would mask the defect.** On 65289 it is still unknown whether the 248 cm
  merged object is one track. Promoting a runner-up would restore a candidate
  while leaving a possibly-wrong merge in place, and would make the round-3 A/B
  unreadable.
* **The cathode fix removes most of the need.** Round-3 direction 3 (do not demote
  a longer untagged in-window main) recovers 51128 at the source. 65289 would
  still benefit — but only if its join is correct, which is the open question.

It is also PR-chain code, so it belongs to whoever owns that chain (§11.7 item 3).

**"For 78242, are you implying it is a PR problem?" — and "so what IS the
problem?"**

**RETRACTION.** An earlier answer in this section called the 71 cm fit hole
*"evidence against the join"*. **That was wrong**, on two counts, and the
geometry disproves it.

*First, the join is excellent.* The merged main's trajectory runs dead straight
through the cathode — mean (y, z) per 10 cm slab:

| x slab | −20…−10 | −10…0 | ‖ | 0…+10 | +10…+20 |
|---|---|---|---|---|---|
| y | 16.3 | 17.8 | ‖ | 19.2 | 20.8 |
| z | 174.8 | 190.4 | ‖ | 205.4 | 222.0 |

dz/dx is ~15.5 cm per 10 cm before the cathode and ~16 cm after; dy/dx is ~1.4
on both sides. One continuous straight line from (−102, 16, 79) to
(+67, 23, 322), ~300 cm long. There is nothing wrong with this merge.

*Second, the "PR crosses fine elsewhere" comparison was too weak to carry the
conclusion.* 281165 and 319913 do cross, but only to x = +7.3 and +7.9 — far
halves of **218 and 84 points**. 78242's far half is **1489 points reaching
+67.5**. Those two never tested a substantial far half.

**The actual problem: the rescued far half is classified as an EM SHOWER.**
Track/shower labelling of the selected main, by side of the cathode:

| event | beam half (x < 0) | far half (x > 0) |
|---|---|---|
| 281165 | 2071 track, 0 shower | 218 track, 0 shower |
| 319913 | 524 track, 0 shower | 84 track, 0 shower |
| **78242** | **1681 track, 0 shower** | **0 track, 1489 shower — 100 %** |

Every point of the newly joined far half is labelled shower, while the beam half
of the *same straight muon* stays 100 % track. 78242 is the only one of the three
where this happens — and the only one whose far half is big enough to matter.

That misclassification is what propagates into the display the owner saw:

* the far half renders as shower, not as fitted track;
* the PR graph is reorganised — the vertex that sat at the cathode end of the
  track in the OFF arm, `(−1.4, 18.4, 197.1)`, **disappears**, and new vertices
  appear at `(+15.5, 24.1, 228.8)` / `(+16.3, 21.1, 223.5)`;
* no fitted segment covers x = −54 → +16, which includes the 54 cm that *was*
  fit before (OFF segment `8004`).

So "missing an EM shower as well as part of the long track" is **one** defect,
not two: the far half became a shower, and the trajectory through the junction
was lost when the object was re-fit around that labelling.

**Is it a PR problem?** The misclassification happens in PR's track/shower
separation, but it fires only on rescued far-half charge and only when that half
is large — so it is *triggered* by the round-2 join, not independent of it. The
neutrino vertex is unchanged between arms at `(−90.4, 8.7, 66.6)`, and the event
keeps its candidate, so this one is a quality defect rather than a loss.

### 11.8.1 Is the far half graph-connected across the cathode gap? — **YES. Lead disproven.**

Checked on owner request. The "isolated component" hypothesis above is **wrong**;
recorded here rather than deleted, because it was the obvious guess and the next
person will have it too.

**Evidence 1 — the connected-component count does not change.**
`component_extreme_wcps` (`clus/src/TaggerCheckTGM.cxx:544-663`) reports
`comp_pts.size()` = connected components from `connected_blobs`, and `n_used` =
those above `component_min_length` (10 cm):

| arm | main | components | above 10 cm | extreme groups |
|---|---|---|---|---|
| OFF | cid 8 (beam half only, 1680 pts) | 2 | **1** | 5 |
| ON | cid 17 (merged, 3170 pts) | 2 | **1** | 5 |

The ON cluster carries an extra **67 cm / 1489-point** far half. Were it a
separate component it would be far above the 10 cm floor and the count would
read **2 above 10 cm**. It reads 1, unchanged — the far half was absorbed into
the single large component.

**Evidence 2 — TGM's extreme pair spans both halves.** `check_tgm` on the merged
cluster forms `CASE-B pair (1,4)` with a **straight chord of 312.0 cm**. The
merged object is ~300 cm end to end and each half alone is ≤ 155 cm, so a 312 cm
chord can only run from one half's far end to the other's. Both halves are
therefore in the same component's extreme set. (That pair is then rejected for
*"unsupported run > 30 cm"* — simply because a straight 312 cm chord does not
follow a gently curving 300 cm track; it deviates ~20 cm mid-chord. TGM=false is
the right answer and this is not the defect.)

**The join geometry is good on every measure that matters:**

| quantity | value |
|---|---|
| nearest 3-D approach across the cathode | **2.16 cm** — *smallest of the three crossers* (281165: 4.09, 319913: 2.14) |
| transverse alignment of that nearest pair | Δy 0.0 cm, Δz 0.6 cm |
| **local** direction agreement, ±30 cm of the cathode | **3.8°** (beam 58.3°, far 62.1° in the x–z plane) |
| end-to-end direction difference | 10.7° — the track's own curvature over 300 cm, not a junction kink |

Note the local-vs-end-to-end distinction: judging this join on end-to-end
directions would call it a 10.7° mismatch and condemn it. Locally at the
junction the two halves agree to **3.8°**, which is what a single muon looks
like.

**So the cause of the 100 % shower labelling is still unknown.** It is not
disconnection, not the gap size, not a junction kink, and not point density
(far half ~10 points per cm of path, beam half ~11). What is established:

* the far half is **in the same cluster, the same connected component, and the
  same PR graph** (it forms segment `17012`);
* it is nonetheless labelled **100 % shower** while the beam half of the same
  muon is 100 % track;
* 281165 and 319913, whose far halves are labelled 100 % track, differ mainly in
  **size** — 218 and 84 points against 1489.

Size is the one axis that still separates the working cases from the broken one,
so the next probe is whether the track/shower separation's behaviour on a
rescued far half depends on its length — not on how it was attached.

### 11.8.2 Step-by-step through the PR chain on 78242 (owner request)

Repro — both arms, one event, per-segment PID trace un-gated:

```bash
PR_JOBS=1 PR_EXTRA_STAGES=pr_display WCT_PID_TRACE_DEBUG=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-cbr2-prql-on  work-cbr2-pr-on-dbg  data 78242
PR_JOBS=1 PR_EXTRA_STAGES=pr_display WCT_PID_TRACE_DEBUG=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-cbr2-prql-off work-cbr2-pr-off-dbg data 78242
```

#### (1) What the PR chain is fed — nothing is missing

`run_pr_chain_batch.sh` consumes `ql_evt<ID>/pctree-evt<ID>.tar.gz` (plus the
opflash tarballs for RSE). Comparing the two arms' pctrees: **220 tensors each,
identical datapath structure, zero paths present in one and not the other, zero
count differences.** What it carries:

| group | contents |
|---|---|
| live 3-D | per-blob `x, y, z, t`, `x_t0cor`, `y_cor`, `z_cor`, per-plane `u/v/w charge_val + charge_unc`, `u/v/w wire_index`, `wpid` |
| live per-APA charge maps | `ctpc_a{0,1}f0p{U,V,W}` — charge, charge_err, cident, slice_index, wind, x, y — **both APAs present** |
| live dead maps | `dead_winds_a{0,1}f0p{U,V,W}`, `dead_gap_a0f0pW` |
| cluster_scalar | `ident`, `cluster_t0`, `flash`, `matched_flash_gid`, `flag_main_cluster`, `flag_associated_cluster`, `lm_flag` |
| perblob | `real_cluster_id`, `real_cluster_main`, `real_cluster_was_main`, `assoc_cluster_id`, `assoc_cluster_main`, `isolated` |
| blob scalar | centre, charge, npoints, wire-index ranges, slice range, wpid |
| light | `flash`, `opflash`, `light`, `flashlight` |
| dead grouping | scalar, corner, cluster_scalar |

So the merged cluster reaches PR complete, with **both** APAs' charge and dead
maps. No input is lost by the rescue.

#### (2) Where the trajectory is lost — inside `tagger_check_neutrino`

Pipeline: `switch_scope → unmerge_bundle → unmerge_assoc → steiner →
fiducialutils → tagger_check_tgm → tagger_check_stm → tagger_check_fc →
protect_bundle → steiner_refresh → tagger_check_neutrino → numu_bdt → nue_bdt →
tracking_visitor → tagger_output`.

ON-arm timeline for the merged main (cluster 17):

| time | stage | what happens |
|---|---|---|
| 55.853 | steiner | cluster 17 graph = **2499 vertices, 4451 edges**; `do_rough_path` runs from `(674.6, …)` to `(−962.2, …)` mm — i.e. **+67.5 cm to −96.2 cm, across the cathode**. Graph is joined. |
| **57.772** | **`main_cluster initial PR`** (1936 ms) | segments are created and fitted. The trace shows **`Seg 132.53 cm Track … pdg 13, KE 320.6 MeV`** — the junction piece — alongside `Seg 122.02 cm` (far half) and `Seg 60.18 cm` (beam half). **All Track, all muon. No `S_traj` anywhere on the long segments.** |
| 57.891 | other_clusters PR | |
| 57.903 | deghosting | 12 ms |
| **58.858** | **`overall main vertex`** (955 ms) | ← graph rebuilt around the chosen vertex |
| **59.478** | **`improve_vertex + examine_direction`** (620 ms) | ← graph rebuilt again |
| 59.538 | `shower_clustering_with_nv` | 60 ms |
| 59.573 | Bee dump | cluster 17 dumps segments **7, 11, 12, 17 only** |

The Bee dump iterates **PR-graph edges** (`ordered_edges(*pr_graph)`,
`MultiAlgBlobClustering.cxx:846`), so only segments still in the graph appear.
Segment ids **8, 9, 10, 13, 14, 15, 16 were created and are gone** — the
132.53 cm junction segment among them.

**Contrast with OFF, which is clean:**

| arm | segments created for the main | segments surviving to the dump |
|---|---|---|
| OFF (cid 8) | ids 4, 5, 6, 7 | **all 4** — `8004`=169 fitted pts, `8005`=101, `8006`=11, `8007`=34 |
| ON (cid 17) | ids 7 … 17 | **4** — `17007`=103, `17011`=12, `17012`=206, `17017`=34 |

OFF's `8004` (169 fitted points, x −54.8 … −1.4) is exactly the stretch the ON
arm ends up missing, and OFF keeps every segment it makes.

**Not a fitting failure.** The doc pr/55 sentinels for an empty `fits()` or a
missing `associate_points` cloud fire **zero times** in either arm. Every
surviving segment has both. The junction segment was *fitted* — 132.53 cm of it
— and then **removed from the PR graph** by one of the two vertex stages between
57.772 and 59.573.

#### (3) Why the far half paints as shower

The `shower_track` layer's rule (`MultiAlgBlobClustering.cxx:866-878`):

```
is_shower = (segment ∈ seg_to_shower)          // shower membership is authoritative
         || kShowerTrajectory || kShowerTopology || |pdg| == 11
```

The PID trace settles which disjunct fires: the far-half segments are **`Track`,
`pdg 13`, KE 298 MeV** — so `kShowerTrajectory`, `kShowerTopology` and the
`pdg == 11` test are all **false**. The far half is therefore painted shower
**solely because `shower_clustering_with_nv` absorbed its segments into a shower
object.** SBND runs `pseudo_shower_track_paint = true`
(`wct-pr-perevt.jsonnet:1302`), which repaints a shower as track when the
shower's `get_particle_type()` is ±13 — that override did not fire, so the shower
is not typed as a muon.

**Summary of the causal chain for 78242**

1. The rescue joins the halves — geometry excellent (§11.8.1), inputs complete.
2. The Steiner graph spans the cathode; `main_cluster initial PR` fits the whole
   object, including a 132.5 cm junction segment, and calls every long segment a
   **muon track**.
3. `overall main vertex` / `improve_vertex` rebuild the graph and the junction
   segment does not survive → the 71 cm hole in `track_fit`.
4. `shower_clustering_with_nv` absorbs the far half into a shower object → all
   1489 far-half points paint as EM shower despite being PID'd as muon track.

Steps 3 and 4 are the two places to instrument in round 3. Both are downstream of
a join this doc now considers sound, so neither is fixed by tightening the
rescue's admission gates.

### 11.9 The revert (owner instruction, 2026-08-17)

All four SBND production defaults set back to `false` in
`cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` (the TLA block)
and `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`clus_all_apa` and `all_apa`
defaults). Toolkit `f3706c45`. **The C++ is untouched** — the knobs and the
far-half containment veto stay in `clustering_cathode_bundle_rescue.cxx`,
default OFF as originally shipped.

Gate: the knobs-off default compiles **byte-identical** to HEAD compiled with the
runner escape `SBND_RESCUE_{IN_BEAM,GEOM_FIRST,PIERCE,DEST_BEAM}=0`
(`cmp`, 54115 B) — and that escape was itself gate-proven equal to the
pre-round-2 baseline at flip time, so this restores the validated baseline
exactly. Compiled-config proof in the other direction: all four keys
(`rescue_allow_in_beam_far`, `rescue_geom_first`, `rescue_pierce_test`,
`rescue_dest_beam_for_new`) are absent from the bare config and reappear as
`true` when the runner passes the TLAs, so the knobs stay fully usable for
round 3.

```bash
# both halves of the proof
wcsonnet -A input=. -S anode_indices='[0,1]' -A output_dir=. -S run=0 -S subrun=0 \
         -S event=0 -A reality=data \
         cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet   # no rescue_* keys
# ... same with -S rescue_geom_first=true etc.                          # keys present
```
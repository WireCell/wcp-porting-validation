# 73 — Cathode bundle rescue, round 2: design for the residual one-sided crossers

**Status: design only. No code is changed by this doc.** Implementation is the next
session's work; §7 is the task list.

doc 72 §A found 10 events in 3000 where the in-beam bundle main is cut at the cathode
and the raw image plainly continues into the other TPC. `ClusteringCathodeBundleRescue`
(doc pr/14 + pr/17 + pr/19) already exists to fix exactly this and is **SBND default
ON**, so every one of the 10 is a case where the existing rescue did not fire. This
doc establishes *why*, per event, and proposes three narrow extensions.

**Doc number caveat**: next free top-level number on 2026-08-17; a concurrent session
may have claimed it.

## 1. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the candidate sample (doc 72 sec A)
python3 scripts/analysis/cathode/beam_cathode_split.py \
        --arm work-mcp1k-cb0805 --out products/beam-cathode-split-mcp1k.tsv --jobs 8
python3 scripts/analysis/cathode/beam_cathode_split.py \
        --arm work-mcp2k-cb0816 --out products/beam-cathode-split-mcp2k.tsv --jobs 8

# the loosened working point that sizes the validation pool (sec 6)
BCS_GAP2D=25 BCS_ANGLE=30 BCS_P_MIN_LEN=5 BCS_P_MIN_PTS=10 BCS_CATH_NEAR=15 \
BCS_P_CATH_NEAR=12 BCS_DIRX_MIN=0.2 \
python3 scripts/analysis/cathode/beam_cathode_split.py --arm work-mcp2k-cb0816 \
        --out /home/xqian/tmp/loose-mcp2k.tsv --jobs 8

# the per-event rescue tracer (sec 4) -- FRESH roots, never a production arm
export CATHODE_RESCUE_DEBUG=1
ROOT=$PWD/work-cbrtrace-mcp1k IMGBASE=$PWD/work-img-mcp1k ENTRIES="548 439" \
        ./run_full1k_nusel.sh 1000 2
ROOT=$PWD/work-cbrtrace-mcp2k IMGBASE=$PWD/work-img-mcp2k \
        ENTRIES="1029 1711 1801 697 1878 354 370 1067" ./run_full1k_nusel_2k.sh 2000 6
grep -a cbrx work-cbrtrace-mcp*/.log_e*.log
```

Entry↔event map: mcp1k 548→65289, 439→65053; mcp2k 1029→398115, 1711→237798,
1801→493439, 697→78242, 1878→51128, 354→317427, 370→319913, 1067→167964.

## 2. What exists today

`clus/src/clustering_cathode_bundle_rescue.cxx`, run in the Q/L all-APA pipeline
after `cathode_connect` and **before** `examine_bundles`. Its own header states the
mechanism the owner described: *"an early flash opens a readout window (~8 µs) that
can ABSORB a later flash. A beam interaction whose track crosses the central cathode
scintillates in both drift volumes and should yield a beam-coincident flash on each
side; when one side's flash is absorbed, that side's charge half gets Q/L-matched to a
DIFFERENT flash."*

SBND production values (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:517-530`; the rest
are C++ defaults):

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

Pass 2 `rescue_unmatched` (doc pr/17) adopts **flashless** clusters; pass 3
`adopt_nu_fragments` (doc pr/19) is OFF in production.

## 3. The residual sample

10 events at the doc-72 working point, 13 at a loosened one:

| | events |
|---|---|
| doc 72 §A production working point | **10** (mcp1k 65289, 65053; mcp2k 398115, 237798, 493439, 78242, 51128, 317427, 319913, 167964) |
| + loosened (`gap2d ≤ 25`, `angle ≤ 30°`, `\|dir_x\| ≥ 0.2`) | +3: **281165** (94 cm missing, gap 2.5, 1.0° — excluded only by `\|dir_x\| = 0.274 < 0.30`), **167744** (100 cm, gap 15.9, 3.0°), 403023 (2 cm — noise) |
| contrast sample: `already-in-main`, the crossers the chain joins correctly | **162** (56 mcp1k + 106 mcp2k) |

evt281165 and evt167744 are genuine and were lost to *my* search cuts, not to the
reconstruction; they belong in the signal sample. So the working signal set is **12
events**, the contrast set **162**.

## 4. Why the existing rescue does not fire — measured, per event

Two independent lines of evidence: the time-gate arithmetic (from the Q/L products)
and the rescue's own `CATHODE_RESCUE_DEBUG=1` tracer.

**First, the geometry is never the problem for the hard gates.** For all 12
candidate rows the tip-to-tip closest approach in the beam-T0 frame is

```
dis 1.5 – 6.1 cm   (cut 25)      tip1 0.08 – 2.59 cm  (cut 5)
dX  0.2 – 4.9 cm   (cut 8)       tip2 0.13 – 1.38 cm  (cut 5)
```

— every row passes all four. The blockers are elsewhere.

| class | events | blocker | evidence |
|---|---|---|---|
| **A** | 398115, 237798 | `require_far_out_of_beam = true` | both halves are matched, each to a *different in-beam flash* (398115: 0.53 and 1.86 µs; 237798: 0.39 and 1.57 µs). Line 471 skips the pair. Tracer: 0 pair tests. |
| **B** | 493439, 78242, 65053, 317427, 319913, 167964 | `dt0 ∈ [−8, +13] µs` | the partner is matched to a flash **+589, +855, +581, +108, +28, −43 µs** away. Line 470 skips the pair. Tracer: 0 pair tests. |
| **C** | 65289, 51128 | the angle tests | tracer fired and rejected, see below |

Class C, quoted verbatim from the tracer:

```
evt65289  [cbrx] c5<->c13 far_conn        dis=5.28 dX=5.14 tip=3.04/2.10 ttH=15.0 ttP=9.0 cc=45.2 len=160/89 -> reject
evt51128  [cbrx] c11<->c8 close_shortstub dis=2.78 dX=0.43 tip=0.53/0.10 ttH=11.6 ttP=-1.0 cc=41.0 len=4/284  -> reject
```

### 4.1 The `conn_far_cut` bias (evt65289)

`conn = p1 − p2s` is the **tip-to-tip vector**, and `cc = angle(conn, dir)`. Here
`dis = 5.28` of which `dX = 5.14` — the vector is **97 % pure drift**. That is not a
track segment; it is the cathode dead gap plus the near-cathode imaging loss (the two
tips sit 3.04 and 2.10 cm short of the plane). Its angle to the track direction is
therefore ≈ `acos(|dir_x|)`, and for this event `|dir_x| = 0.748` gives 41.6° against
a measured 45.2°.

**So with an x-dominated `conn`, `conn_far_cut = 30°` is not testing collinearity at
all — it is testing `|dir_x| > cos 30° = 0.866`, i.e. it rejects any crosser more than
30° off the drift axis.** Note `tt_pca = 9.0°` had *already* passed `angle_cut = 10°`:
the two halves are demonstrably collinear and the connection test overrode that.

The candidate sample is consistent with the bias being the operative one: 10 of the 12
rows have `|dir_x| < 0.866`, median 0.71.

### 4.2 The pre-collapse stub (evt51128)

`len = 4/284`: at rescue time the beam-side cathode-touching charge is a **4 cm stub**,
because the rescue deliberately runs *before* `examine_bundles`, i.e. before the
flash-collapse that would glue that stub to the rest of the beam bundle (the 297 cm
object doc 72 sees post-collapse). `min(4, 284) < short_dir_len = 25` routes the pair
into the short-stub path, whose direction estimate comes from the 4 cm stub, and
`cc = 41.0° ≥ conn_short_cut = 30°`.

Running before `examine_bundles` is load-bearing (it is what keeps the crosser one
flash-collapse member so the PR job's `unmerge_bundle` keeps it whole — doc 53), so
the ordering must not change. The direction estimate must be made robust instead.

## 5. Proposed fix

Three independent, default-OFF knobs. Each targets one measured class; none changes
the existing accept path.

### F1 — `rescue_allow_in_beam_far` (class A, 2 events)

Let `K_far` be in the beam window **provided it belongs to a different flash bundle**.
Today `require_far_out_of_beam` blocks it outright.

* Direction rule: both halves are in-beam, so the T0 hypothesis changes by
  ≤ (2.2 − 0.2) µs × 0.1563 cm/µs = **0.31 cm** — below one blob. Merge direction can
  therefore follow the existing beam-dominant/longer-half rule without a new tie-break.
* Risk: lowest of the three. It cannot move a cluster out of the beam window, and the
  x re-materialisation is sub-blob.
* Implementation: gate line 471 on the new knob instead of removing it.

### F2 — `rescue_geom_first` (class B, 6 events)

Drop the `dt0 ∈ [−8, +13] µs` window when the pair passes a **tightened** geometric
test, and require corroboration that the far half really belongs at the beam time.

The `dt0` window encodes the absorbing-window mechanism (a wrong flash 2–13 µs away).
The measured failures show the wrong flash can be anywhere — up to **855 µs** — so a
time prior cannot reach them. Replace the prior with evidence:

1. tip-to-tip `dis < dis_cut` (5 cm) and both tips within `cathode_x_cut`;
2. **cathode-piercing agreement**: extrapolate each half's local axis to `x = 0` and
   require the two `(y, z)` piercing points to agree within `pierce_cut` (propose
   5 cm). This is the discriminator doc 72 used; it is far sharper than `max_dis`,
   which admits 25 cm of transverse slop;
3. the far half's corrected `x` under the **beam** T0 must lie inside its own TPC
   (all 12 candidates satisfy this; it is a cheap veto against absurd merges);
4. **corroboration, one of**: (a) there is no in-beam flash at all on the far half's
   side — the true absorption signature, which holds for 65289, 78242, 51128, 317427;
   or (b) there *is* one and the far half's assigned flash is more than
   `geom_first_min_dt` (propose 20 µs) away, so the assignment is not a near-miss —
   which holds for 493439 (+589), 65053 (+581), 319913 (+28), 167964 (−43).

Condition 4 is what keeps this from becoming "merge anything that lines up": it
requires the far half's own flash story to be independently broken.

### F3 — `rescue_conn_from_pierce` (class C, 2 events)

Replace the tip-to-tip connection angle with the piercing-point test whenever `conn`
is drift-dominated or the direction donor is a stub:

* if `dX / dis > conn_drift_frac` (propose 0.8), **skip** `cc` entirely and require
  instead `angle(dir1, dir2) < angle_cut` plus piercing agreement within `pierce_cut`.
  `conn` in that regime is the cathode gap, not a track segment (§4.1);
* in the short-stub path, take the direction from the **long** half only and test the
  stub by piercing agreement, not by the stub's own 4 cm direction (§4.2).

This is the one with real regression risk — `conn_far_cut` is what stops two unrelated
tracks that happen to end near each other from merging. The piercing test is a strictly
sharper substitute (5 cm on an extrapolated point vs 30° on a 5 cm lever arm), but that
claim must be demonstrated on the contrast sample, not asserted.

### What is deliberately not proposed

* Retuning `angle_cut`, `conn_far_cut` or the `dt0` window in place. They are validated
  against the pr/14 hand scans (8 moves) and the doc-72 evidence does not overturn
  those; widening them globally would re-open a settled validation.
* Any change to where the rescue runs in the pipeline (§4.2).
* Anything in `cathode_connect` itself (M10: the production connector stays
  byte-identical; the rescue already forks its geometry by duplication).

## 6. Validation plan

**Signal**: the 12 events of §3. Target — each acquires the missing half, and the
in-beam main's `len_main_cm` grows to cover both sides. Re-run
`beam_cathode_split.py` on the ON arm: a fixed event moves from `MISSING` to
`already-in-main`.

**Contrast**: the 162 `already-in-main` events. Target — **zero** change. These are
the crossers the chain already joins; any of them that moves is a regression.

**Byte-identical**: all three knobs OFF ⇒ compiled config byte-identical (key
suppression) and the standard `/ab-verify` gate PASS on `abtest/events.txt`. Then a
knob-off run of the full 3000 must reproduce the current arms member-for-member.

**Whole-sample A/B**: mcp1k + mcp2k, ON vs OFF at one binary, counting events whose
`nusel-table.tsv` in-beam `main_id` / `npts_main` / `len_main_cm` change. The
prediction is 12 changed events; anything much larger means F2 or F3 is over-firing
and the extra events must be hand-scanned before the knob is considered.

**Regression guard from the pr/14 round**: the 8 hand-scan moves that validated the
original direction rule must still reproduce (memory: "any retune needs re-validation
against the same 8 moves").

## 7. Next-session task list

1. Extend the `[cbrx]` tracer to print the two clusters' `t0`, flash gid and in-beam
   flag alongside the lengths. §4 attributes classes A and B from the Q/L products
   because the tracer prints neither; and its print is guarded by `dis < max_dis`, so
   "no line" is currently ambiguous between "never tested" and "tested, far apart".
   This is the one piece of §4 that rests on inference rather than a printed line.
2. Implement F1 (smallest, lowest-risk, 2 events) and gate it alone first.
3. Implement F2, then F3.
4. Run the validation of §6 after each, not once at the end — F3 is the one expected
   to move the contrast sample.
5. Decide with the owner whether class B is better fixed here or upstream in Q/L
   matching: for 493439, 65053, 319913 and 167964 a perfectly good in-beam flash
   exists on the far side (dt ≈ 0.00 µs) and Q/L still matched the cluster to a flash
   hundreds of µs away. That is arguably a `beam_pref` (doc 22) failure, and fixing it
   there would remove the need for F2's condition 4(b).

## 8. Open question worth the owner's eye

Class A (398115, 237798) is not a light-reconstruction failure at all: both flashes
exist, both are in the beam window, and Q/L matched each half to its own side's flash
correctly. The two halves are then left in different bundles and each becomes its own
in-beam main — on evt237798 both are separately in-beam mains (5842 and 1505 points),
so PR ran twice, each time on half a track. If the two in-beam flashes are the two
APAs' views of the *same* scintillation (0.39 vs 1.57 µs apart here), the cleaner fix
may be to group them at flash level before Q/L matching, rather than to re-join the
charge afterwards.

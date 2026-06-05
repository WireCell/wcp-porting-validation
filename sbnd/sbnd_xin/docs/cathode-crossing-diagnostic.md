# Cathode-crossing TPC0/TPC1 offset diagnostic

## Why

After Q/L matching and the per-cluster T0 correction, cosmic tracks that cross the
central cathode are reconstructed as two halves — one in TPC0 (East, x<0), one in
TPC1 (West, x>0). In **data** the two halves do not line up perfectly at the cathode
(x≈0); the mismatch is much smaller in **MC**. This diagnostic measures that mismatch
*inside QLMatching*, on the actual cluster geometry the matcher sees, and decomposes
it so the cause can be narrowed among: a common **time/T0 offset**, a **drift-velocity**
error, and a **transverse (y–z) position shift**.

It supersedes an earlier offline analysis (from the `calib-evt<ID>.json` dump) that
concluded the offset was a drift-direction effect of ≈ −1.5 cm / δt ≈ −4.7 µs. That
analysis fit δt by minimizing the cathode (y,z) gap, which absorbed genuine transverse
offset into the drift term and inflated it. The precise per-track method below shows the
data excess is **dominated by a transverse y–z misalignment**, with at most a small
drift component.

## Method (three vectors)

For every pair of clusters that (a) sit on opposite TPCs, (b) share one T0 flash group
(±`flash_group_window`, 80 ns), and (c) have their **closest pair of points in the
T0-corrected frame** near the cathode, log three vectors:

- **dir0**, **dir1** — the local track direction (`Cluster::vhough_transform`, a Hough
  axis within `cathode_diag_radius`, default 15 cm) at each half's cathode-end point.
  A constant per-cluster T0 x-shift does **not** rotate a Hough axis, so dir0/dir1 give
  the *true* track direction, independent of any drift miscalibration.
- **conn** = p1 − p0 — the vector joining the two closest points (p0 on TPC0, p1 on
  TPC1, both in corrected coordinates). This is the quantity a miscalibration distorts.

The closest pair must be found in the **corrected** frame: the raw k-d trees place the
two cathode ends ~2·v·t apart in x (opposite drift offsets), so `get_closest_points`
on raw points would pair the wrong points.

### Decomposition

`conn` carries a geometric floor of ~0.9 cm from the cathode **dead gap** (TPC0 active
ends at x≈−0.45 cm, TPC1 at x≈+0.45 cm) plus point granularity — a slide **along** the
track. Remove it: with `dhat` = unit mean of (dir0, dir1),

```
along = conn · dhat                  (nuisance: cathode-gap + arc-length slide)
perp  = conn − along·dhat            (the line-to-line offset = the physics)
```

Then, by cause:

| component of `perp` | caused by | notes |
|---|---|---|
| x (drift direction) | T0/timing offset **or** drift velocity | the two are **degenerate** for cathode-crossers (both pinned at the cathode-to-anode distance); cannot be separated here |
| y, z (transverse)   | position shift (space charge / alignment) | **cannot** be produced by T0 or velocity (those move x only), nor by the cathode gap (that is in x) |

Caveat: a pure drift x-translation, projected into the plane ⊥ dhat, also has small y,z
shadows; conversely the cathode-gap leaks into the perp **x** estimate on angled tracks.
So the **transverse y–z** part is the clean, artifact-immune signal; the **drift** part
is method-floor-limited and must be read against the MC baseline, not as an absolute δt.
The per-pair printout is raw data — the global fit and the data-vs-MC contrast are done
in the aggregation script. Per single pair the split is under-determined; aggregate.

## How to run

```bash
# one event (idx into the data/mc sample) or 'all'; adds -calib if you also want the
# ql_scan JSON. Matched output (mabc-all-apa.zip) is byte-identical with/without the flag.
sbnd_xin/run_ql_evt.sh data -cathode-diag all
sbnd_xin/run_ql_evt.sh mc   -cathode-diag all

# the diagnostic logs one "QLCATHODE" line per pair into the per-event run log:
grep QLCATHODE sbnd_xin/work/ql_evt<ID>/wct_ql_evt<ID>.log

# aggregate + global fit + data-vs-MC contrast:
python3 /home/xqian/tmp/ql_cathode_offset/agg_diag.py
```

The flag is off by default. Plumbing: `QLMatching` config `cathode_diag` (non-empty =>
on) and `cathode_diag_radius`; the SBND wrapper `sbnd_xin/qlmatching.jsonnet` passes it
through `extra={}`; `wct-clus-matching-perevt.jsonnet` exposes the `cathode_diag` TLA;
`run_ql_evt.sh -cathode-diag` sets it. With the flag empty, `dump_cathode_diag` is never
called and production output is bit-identical. Code: `match/src/QLMatching.cxx`
(`dump_cathode_diag`, called after `dump_calib`).

### QLCATHODE columns

```
QLCATHODE <count> 0/<id0> 1/<id1> <t0_us> <t1_us> d=<closest-dist_cm>
  p0=(x,y,z)_cm p1=(x,y,z)_cm          # closest corrected points; x≈0 expected (else not at cathode)
  dir0=(...) dir1=(...)                 # local Hough directions (sign-fixed to conn)
  conn=(x,y,z)_cm                       # p1 - p0
  a_d0d1 a_d0c a_d1c a_cdh (deg)        # angles: dir-dir, dir-conn, conn-dhat
  along perp (cm)                       # conn split along/perp to the track
  dX dY dZ (cm)                         # raw conn components (dX has the cathode-gap floor)
```

Quality cuts in the aggregation (a clean single cathode-crosser): `a_d0d1 < 10°`
(same physical track), `|p0.x|,|p1.x| < 5 cm` (both halves at the cathode), `d < 5 cm`
(reject merged / different tracks).

## Result (all 10 data + 10 MC events of the standard SBND samples)

Clean pairs (one per cathode-crosser): **5 data** (evt 686, 1302, 1346, 1852, 2028) from
5/10 events; **8 MC** from 6/10 events. The other events have no cross-TPC cathode pair,
or only merged/multi-track candidates (notably data evt2050) that fail the angle cut.

| quantity | DATA | MC | |
|---|---|---|---|
| **transverse \|perp_yz\|** (artifact-immune) | **1.37 cm** | 0.48 cm | **~2.9×** |
| transverse residual after global-drift removal (rms) | 1.67 cm | 0.53 cm | ~3× |
| global drift translation Dx | +1.02 cm | +0.58 cm | MC = cathode-gap/method floor |
| data-excess drift (DATA−MC) | +0.44 cm | — | small, floor-limited (no clean δt) |

Per-pair transverse offsets that drive the data signal: evt686 `perp_yz`=2.56 cm,
evt1852 **2.18 cm with perp.x≈0.05** (a *pure* transverse offset — impossible for
T0/velocity), evt2028 1.29 cm. Both evt686 and evt1852 were confirmed to be single
continuous cathode-crossing tracks (not two parallel cosmics) by plotting the two halves
(`/home/xqian/tmp/ql_cathode_offset/verify_686_1852.png`): one straight line through
x=0 in x–y, with a visible ~2 cm z-jump at the cathode in evt1852.

### The effects combine — combined global fit

The offset is best modelled as a **combination**: one global relative translation
between the two TPC halves, T = (Δx, Δy, Δz), where Δx is the drift (T0/velocity,
degenerate) and (Δy, Δz) is the transverse position shift, fit simultaneously to all
clean pairs (`fit_global_T`), plus a per-track residual:

| | DATA | MC |
|---|---|---|
| combined T = (Δx, Δy, Δz) cm | **(+0.90, −0.22, +1.34)** | (+0.58, −0.01, +0.09) |
| drift \|Δx\| / transverse \|Δy,Δz\| | 0.90 / **1.36** | 0.58 / 0.09 |
| residual after combined fit (rms) | 1.10 cm | 0.52 cm |

Adding the transverse term drops the data residual 1.67→1.10 cm (while the MC residual
barely moves, 0.53→0.52, and MC transverse fits ≈0) — so the transverse component is a
genuine data signal, not extra free parameters absorbing noise. The fitted transverse is
mostly in **z** (Δz≈+1.3 cm). Data `perp.z` shows a **tentative one-sided positive trend**,
driven by the three large-offset tracks (686:+2.31, 1852:+2.16, 2028:+1.24) with the two
small pairs mixed (1302:+0.55, 1346:−0.26); MC `perp.z` scatters around 0. With n=5 this
one-sidedness is suggestive, not established — it would need more cathode-crossers to
confirm whether the transverse is a coherent TPC0-vs-TPC1 z-shift or a position-dependent
(SCE-like) distortion that merely averages positive here. (This is *not* in tension with
the earlier "Δz not constant" remark: that range included the later-cut outlier/merged
pairs, and used a different rigid-Δ definition.) So the data offset = **a small drift
(order the MC floor) + a transverse ~1.3 cm component (largely z) + a ~1.1 cm
position-dependent residual.** MC has only the drift floor.

### The X gap is physical — do NOT close it

The ~1.25 cm cathode dead-gap in x (TPC0 active ends at x≈−0.45, TPC1 at +0.45) is real and
must be preserved. Per-pair X structure (closest cathode-end points), MC as reference:

| | p0.x (TPC0) | p1.x (TPC1) | gap_x | mid_x |
|---|---|---|---|---|
| MC (reference) | −0.61 | +0.64 | **1.25 ± 0.15** | +0.02 (centered) |
| DATA | −0.20 | +1.33 | **1.53 ± 0.37** | +0.56 |

Data's x-gap **agrees with MC** (within scatter) — there is no drift gap to "close". The
combined fit's Δx (data +0.90, MC +0.58 cm) is the cathode-gap leaking through the ⊥
projection (that is why MC shows it too), **not** a real closing offset; applying it as a
per-side shift wrongly fills the physical gap. The only genuine x difference is a small
**common +0.56 cm displacement of the whole junction off the cathode** (both halves same
direction — not a t0 effect, which is opposite-sign). **Correction to apply for display =
transverse (y,z) only; leave x untouched.**

### The exact Y–Z shift applied (display correction)

The transverse part of the combined fit is the relative offset of the two halves,
T_yz = (Δy, Δz) = **(−0.22, +1.34) cm** (conn = p1 − p0, p1 on TPC1). It is closed
**symmetrically**, moving each half by half of it toward the midline (±T_yz/2), with **x
left exactly untouched** so the physical cathode gap is preserved:

| TPC | region | Δx | Δy (cm) | Δz (cm) |
|---|---|---|---|---|
| **TPC0** | East, x < 0 | **0** (untouched) | **−0.11** | **+0.67** |
| **TPC1** | West, x ≥ 0 | **0** (untouched) | **+0.11** | **−0.67** |

i.e. ±(Δy, Δz)/2 = ±(−0.11, +0.67) cm, opposite signs on the two TPCs. This is a single
rigid per-TPC translation applied to every point of that half (verified flat across all 10
data events). It is the shift used to build the transverse-only corrected Bee
(`mabc_data_transverse.zip`); the earlier `mabc_data_combinedT.zip` additionally shifted x
per-side and was **wrong** because it filled the physical gap.

## Conclusion

The data cathode-crossing offset is a **combination**, dominated by the transverse piece:

1. a **transverse (y–z) misalignment ~1–1.5 cm that T0/velocity cannot explain** — mostly
   in z, with a tentative one-sided trend (n=5; could be a TPC0-vs-TPC1 z alignment or a
   coherent space-charge shift) plus a ~1.1 cm position-dependent residual (SCE-like); and
2. a **small drift-direction (T0/velocity, degenerate) component** of order the MC/method
   floor (data excess ≈0.3–0.4 cm), with no clean T0-vs-velocity value extractable here.

MC shows essentially only the drift/method floor and no transverse offset. This refines —
does not contradict — the earlier finding: both methods always agreed on a larger
transverse offset in data; only the drift attribution moves, downward, and the transverse
is now resolved into a coherent global z-shift plus a position-dependent part.

To make progress on the drift/T0-vs-velocity piece (if pursued), break the degeneracy
with anode-anchored / through-going tracks measured vs drift distance, or fold in the
independent SBND timing / drift-velocity calibration. For the transverse piece, compare
against the SBND space-charge map.

## Artifacts
- `match/src/QLMatching.cxx` — `dump_cathode_diag` (the in-code instrument)
- `/home/xqian/tmp/ql_cathode_offset/agg_diag.py` — parse QLCATHODE lines, global fit, data-vs-MC
- `/home/xqian/tmp/ql_cathode_offset/verify_686_1852.png` — single-track confirmation of the two largest-offset data pairs

## The same geometry now drives a pre-fit cull (`flag_xtpc_consistent`)

The three-vector machinery above is reused (not just for the offset measurement) to **cull bundles
before the fit**: `QLMatching::cull_cross_tpc` (config `xtpc_flag`, SBND-on) pairs **candidate** main
clusters across TPC0×TPC1 in a coincident flash group; a pair confirmed as one cathode-crosser sets
`flag_xtpc_consistent` on both bundles and drops each marked cluster's non-consistent rivals before
the LASSO (so fewer bundles enter the fit). Unlike this diagnostic it uses the **full** clusters (so
a window-truncated half still gets a closest pair) and applies the per-TPC `dy/dz` to the
closest-point vector.

*(History: originally a post-matching observation-only confirm-stamp with a per-cluster
`xtpc_consistent` output scalar; now a pre-fit cull that changes matching, and the root-node scalar
was removed — only the `-calib` bundle field remains.)*

Two scenarios, cuts tuned on the 10 hand-scan **data** events (truth = both halves hand-scan-selected),
on the real C++ `vhough` values:

| scenario | condition | cut |
|---|---|---|
| 1 — cathode end present | closest approach `d` (T0-corrected, `dy/dz` applied) | `d < 5 cm` |
| 2 — `window_truncated` (cathode end missing, `d` large) | `conn`,`dir0`,`dir1` collinear **AND** bounded `d` | `a01,a0c,a1c < 20°` AND `d < 300 cm` |

`flag = (d < 5) OR (truncated AND all three angles < 20° AND d < 300 cm)`. The scenario-2 **distance
ceiling is new and load-bearing**: pairing *candidate* (not just matched) bundles admits far-apart
collinear truncated pairs (data false `d=473 cm`, MC false `d=406/326 cm`, angles `< 14°`) that angle
cannot reject (the MC false at `3.6°` is *more* collinear than the data true evt1302 at `2.8°`) — but
every true scenario-2 pair sits at `d ≤ 264 cm`, so `d < 300 cm` cleanly separates them.
**Flag purity 100% (data 16/16, MC 30/30).** End-to-end, the cull removes ~46 (data) / ~139 (MC)
rival bundles pre-fit and takes **MC true-match agreement 91→97 (+6), DATA flat 92**. Full design:
`match/docs/chisquare_flags_comparison.md` §16.

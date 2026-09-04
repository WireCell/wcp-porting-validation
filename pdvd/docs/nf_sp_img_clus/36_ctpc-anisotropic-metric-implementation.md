# 36 — The anisotropic ctpc metric, implemented: a two-level query behind a default-OFF knob, measured against the 0.35 floor

**Status.** Toolkit change shipped **default OFF** (knob `ctpc_aniso_metric`,
C++ default `false`, PDVD PR job TLA default `false`). Knob-off path proven
byte-identical on every detector that binds the changed component (§3).
Knob-on measured on the 120-event PDVD PR manifest against the production
`good_point_pitch_frac = 0.35` floor (§4–§7). **Verdict: the metric does to
the good-point test exactly what doc 34 predicted (pass rate 0.18 → 0.70,
the best trajectory coverage yet measured) and that is what disqualifies it
at the legacy radius: the 46 cm tail of doc 32 §19 returns on cluster 109,
manifest-wide unsupported trajectory rises 10.8 → 13.5 % and STM tags fall
by 78.** Recommendation (§8): do not flip; the tail is a pitch-axis effect
that no metric on a three-plane 2-D test can remove. **PDVD production is
unchanged by this round.**

Doc 34 is the investigation this executes; doc 32 is where the 0.35 floor
came from. Read §1 of doc 34 for the lattice measurements this rests on.

## 0. Repro

```bash
WCPI=/home/xqian/toolkit-dev/wcp-porting-img
T=/home/xqian/toolkit-dev/toolkit
SP=<scratch>                       # /home/xqian/tmp/... in this round

# --- build, tests --------------------------------------------------------
cd $T && wcbuild > build.log 2>&1; echo rc=$?          # freshness: ls -la local/lib/libWireCellClus.so
./wcb build --tests -p -k > build_tests.log 2>&1; echo rc=$?
./build/clus/wcdoctest-clus > doctest.log 2>&1; echo rc=$?
./build/clus/wcdoctest-clus -tc='ctpc aniso metric*,clus knob defaults: ctpc_aniso_metric*'

# --- compiled-config proof (PR job) ---------------------------------------
cd $WCPI/pdvd && ./scripts/stage_pr_tag.sh 039252 2 d36cfg d27fresh
PDVD_PR_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 ./run_pr_evt.sh -s d36cfg -stm-fit 039252 2   # knob off
PDVD_PR_TLA="-S ctpc_aniso_metric=true" PDVD_PR_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 \
    ./run_pr_evt.sh -s d36cfg -stm-fit 039252 2                                       # knob on
#   cmp the knob-off .wct-pr_d36cfg.json against the pre-change one; grep -c '"ctpc_aniso_metric"' the knob-on one

# --- G1: PDVD PR knob-off byte identity + the knob-on arms (120 events) ---
cd $WCPI/pdvd
PIN=$SP/pin_d36new TFOFF=$SP/tf_knoboff.json JOBS=12 OUT=$SP/d36/p120 \
    ./docs/nf_sp_img_clus/scripts/run_d36_arms.sh            # arms d36off d36on d36on035 (+ ARMS=d36p000)
PIN=$SP/pin_d36ref TFOFF=$SP/tf_knoboff.json JOBS=12 ARMS=d36refpr OUT=$SP/d36/p120ref \
    ./docs/nf_sp_img_clus/scripts/run_d36_arms.sh            # the knob-off reference on the OLD pin
#   TFOFF = git show bec1bd75:cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json (49 keys, no floor)
for d in work/*_d36off; do python3 ../abtest/hash_archive.py $d/mabc-pr.zip; done   # vs the same for *_d36refpr

# --- G2: clustering stage, PDVD + PDHD, old pin vs new pin ----------------
cd $WCPI/pdvd && PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -s d36cref 039252 2   # old pin; d36cnew under the new pin
cd $WCPI/abtest && ./run_events.sh d36cref_pdhd clus events_pdhd.txt          # old pin; d36cnew_pdhd under the new pin
./ab_compare.sh d36cref_pdhd d36cnew_pdhd; echo rc=$?

# --- G3: SBND + uBooNE (the doc-25 shared-component gate recipe) ----------
cd $WCPI/pdvd/stm/gates
OLD_ARM=d36old NEW_ARM=d36new OLD_LIB=$SP/pin_d36ref/libpin NEW_LIB=$SP/pin_d36new/libpin QL_SUFFIX=d99fix ./shared_gate.sh arms
OLD_ARM=d36old NEW_ARM=d36new ./shared_gate.sh compare

# --- grading ------------------------------------------------------------------
cd $WCPI/pdvd
python3 docs/nf_sp_img_clus/scripts/pr_arm_census_diff.py d32p035 d36on
python3 docs/nf_sp_img_clus/scripts/pr_arm_census_diff.py d32p000 d36on
python3 docs/nf_sp_img_clus/scripts/pr_arm_census_diff.py d36on d36on035
python3 docs/nf_sp_img_clus/scripts/pr_arm_support120.py d32p000 d32p035 d36on d36on035
python3 docs/nf_sp_img_clus/scripts/stm_trajectory_coverage.py \
    base:work/039252_2_d32p000 f035:work/039252_2_d32p035 aniso:work/039252_2_d36on stack:work/039252_2_d36on035 \
    --axis 109:343.1,140.1,195.3:221.4,196.8,253.7
python3 docs/nf_sp_img_clus/scripts/stm_endtrim_grade.py \
    base:work/039252_2_d32p000 f035:work/039252_2_d32p035 aniso:work/039252_2_d36on stack:work/039252_2_d36on035 \
    --axis 109:343.1,140.1,195.3:221.4,196.8,253.7 --core-extent 149.4
python3 docs/nf_sp_img_clus/scripts/ctpc_metric_census.py PDVD:work/039252_2_d36on
```

The pins: `pin_d36ref` = `local/lib` + `build/apps/{wire-cell,wcsonnet}` at
toolkit `28cd60d8` before any edit (`libWireCellClus.so` 06:02:44);
`pin_d36new` = the same after the build of this round (08:27:03). Every arm
below names the pin it ran on. A peer shares this tree, so nothing ran on
whatever `local/lib` happened to hold at the time.

## 1. What was built

### 1.1 The two-level query

Nothing about the ctpc changes: the `(x, y)` arrays, the k-d tree, the scope,
the pctree dump are untouched. Only the two functions every ctpc distance goes
through change, and only when the knob is on:

```
level 1   isotropic k-d query with the CIRCUMSCRIBING radius  r_out = r / s
level 2   keep candidates with  dx² + (s·dy)² < r²
```

with `s = min(1, drift_step / pitch)` per (apa, face, plane) and
`drift_step = nticks_live_slice · tick · drift_speed`. Level 1 must query with
the *larger* radius, not `r` — the ellipse sticks out past the circle of
radius `r` along the pitch, and querying `r` would silently lose points. Since
the ellipse lies entirely inside the circumscribing circle, no point escapes,
and the result is exact.

The existence-only form (`has_closest_point`, the good-point tests) keeps its
early termination for the common cases through two exact brackets: a hit
inside the inscribed circle (radius `r`) is inside the ellipse ⇒ accept; an
empty circumscribing circle ⇒ reject; only the in-between case enumerates.
The returned pairs of `get_closest_points` keep the **isotropic** squared
distance the tree computed (physical mm², what callers `sqrt`); only
membership is anisotropic.

**The trap the design is built around**, now pinned by a doctest: the
Euclidean-nearest lattice point can lie *outside* the ellipse while a
Euclidean-farther one lies *inside* it (drift-displaced at 2.1 mm vs
pitch-displaced at (1.2, 4.0) mm under `s = 0.387`, `r = 2 mm`). A filter on
a nearest-only query would say "no". Only the all-candidates `radius()` may be
filtered.

Code: `clus/inc/WireCellClus/CtpcAnisoMetric.h` (three inline templates:
`ctpc_yscale`, `ctpc_radius_aniso`, `ctpc_exists_within_aniso`, templated on
the tree type so a doctest can drive them on a bare `NFKDVec::Tree`);
`clus/src/Facade_Grouping.cxx` `get_closest_points` / `has_closest_point`
branch on the grouping's flag after the unchanged projection block, legacy
statement kept verbatim; `fastgeom()` fills `drift_step` and `yscale[3]`
beside the tick / drift-speed lines it already had (one more `.at()` on the
same metadata loop that fills them).

### 1.2 Where the switch lives, and why there

`Grouping::set_ctpc_aniso_metric(bool)`, a grouping-wide member, set once per
job by `MultiAlgBlobClustering::load_grouping` from its new config key
`ctpc_aniso_metric` (default `false`), copied to derived groupings by
`Grouping::from` (the retile shadow grouping). `Grouping::set_params` was
declared and never defined, so there was no existing config route into a
grouping; the anodes/detector-volumes setters in `load_grouping` are the
precedent.

It is grouping-wide on purpose. The request was that PDVD's several places
"should be made consistent", and the alternative — threading a per-call flag
through the two `good_point_pitch_frac` chains — would reach the end trim and
the 18 pattern-code sites and leave `test_good_point` at 0.6 cm in the
connectors and `get_ave_charge` at 0.3 cm in the vertex finder isotropic. Then
"within 0.6 cm" would mean two different neighbourhoods to two callers on the
same event. With the flag inside the two shared functions, all six entry
points and the two direct external callers agree by construction.

Under the scaled metric every ctpc reach is the same number of **lattice
cells** on PDVD as on SBND: 0.6 cm is 2 slices × 2 pitches on both. That is
the lattice-unit argument of doc 34 §5, and it is what "consistent" buys.

### 1.3 Knob-on evidence in the log

The first `fastgeom()` fill per (apa, face) with the knob on logs one INFO
line with `drift_step` and the three `s`; if `drift_step ≤ 0` (a
DetectorVolumes without `nticks_live_slice`) it logs a WARN naming the face,
because an ON-but-inert knob is doc 77's failure mode and must not be silent.
`MultiAlgBlobClustering::configure` logs the knob when it is set. From the
`d36on` arm, 039252/2:

```
[08:46:03.446] I [  clus  ] <MultiAlgBlobClustering:clus_pr> ctpc_aniso_metric ON: ctpc radius queries use the lattice-normalised metric (doc pdvd/36) on every grouping loaded by this node
[08:46:19.245] I [  clus  ] Grouping ctpc_aniso_metric ON: apa 0 face 0 drift_step 2.9615 mm, pitch U/V/W 7.6500/7.6500/5.1000 mm, yscale U/V/W 0.3871/0.3871/0.5807
... one line per (apa, face), all 16 identical on PDVD ...
```

The 2.9615 mm and the two scales are exactly the constants doc 34 §1 measured
from the pctree dump, now read from the DetectorVolumes metadata at run
time.

### 1.4 Plumbing

- toolkit `cfg/pgrapher/experiment/protodunevd/pr.jsonnet`: `pr(...)` arg
  `ctpc_aniso_metric=false` → the `clus_pr` MultiAlgBlobClustering `data`
  carries `[if ctpc_aniso_metric then 'ctpc_aniso_metric']: true` — key
  suppressed when off. PDVD file only.
- `pdvd/wct-pr-perevt.jsonnet`: TLA `ctpc_aniso_metric = false`, passed
  through. This is the one line the production flip changes (§8).
- Not wired: the **clustering** job (`protodunevd/clus.jsonnet`, four
  MultiAlgBlobClustering nodes) and every other detector. The C++ key exists
  for all of them; nothing sets it.
- Tests: `clus/test/doctest_ctpc_aniso_metric.cxx` (five cases: yscale
  clamp/fallback; `s = 1` reproduces `radius()`/`exists_within()` exactly on an
  SBND-shaped lattice, indices, distances and order; a PDVD-shaped lattice
  matches a brute-force ellipse over 5 radii × 81 phases, with the sweep
  required to contain both empty and non-empty answers and phases where the
  ellipse and circle disagree; the nearest-outside/farther-inside trap; empty
  tree and zero radius) and a `ctpc_aniso_metric is off` case in
  `doctest_clus_knob_defaults.cxx` pinning the MultiAlgBlobClustering default
  and the Grouping default.

## 2. What the knob touches in the PR job

Every ctpc query in the PR job goes through the two functions, so with the
knob on the following change, at these radii. PDVD `drift_step = 2.9615 mm`,
pitch 7.65 (U/V) / 5.10 (W) mm, `s = 0.387 / 0.581`:

| radius | callers (PR job) | drift semi-axis | pitch semi-axis U/V · W | pitch reach, pitches (was) |
|---|---|---|---|---|
| 0.2 cm | end trim `examine_end_ps_vec` ×4; 17 pattern-code `is_good_point` sites; `NeutrinoTaggerNuE` gap test ×3; rough-path probe | 2.00 mm | 5.17 · 3.44 | 0.675 (was 0.261 / 0.392) |
| 0.3 cm | 2 pattern-code sites; `get_ave_charge` / `get_ave_3d_charge` in the vertex finder and `pr118_connector_walk` | 3.00 | 7.75 · 5.17 | 1.013 (was 0.392 / 0.588) |
| 0.6 cm | `test_good_point` and `is_good_point` in `connect_graph_relaxed{,_strict}` (run here by `protect_bundle`, `steiner`, `steiner_refresh`); TGM `is_good_point` | 6.00 | 15.50 · 10.33 | 2.026 (was 0.784 / 1.176) |
| 2.4 cm | TGM `check_neutrino_candidate` (`low_dis_limit·2`) | 24.0 | 62.0 · 41.3 | 8.10 (was 3.14 / 4.71) |

The drift reach is unchanged in every row; the pitch reach becomes
plane-independent (`r / drift_step`), which is the whole point. The area of
the query grows by `1/s` (2.58× on U/V, 1.72× on W). For comparison the
shipped floor at 0.2 cm gives a 2.68 mm *circle* on U/V (area 22.5 mm² vs the
ellipse's 32.5 mm² and the legacy circle's 12.6 mm²) and reaches 2.68 mm in
drift.

`is_good_point_wc` is also covered but has no production caller.
`get_ave_charge` is the one consumer that uses the returned indices for a
number (a charge average) rather than a boolean; with the knob on its average
runs over the ellipse.

## 3. Gates — knob OFF is byte-identical everywhere

| gate | arms | result |
|---|---|---|
| doctests | `./build/clus/wcdoctest-clus` | **284 cases / 8722 assertions, rc=0** (the first run of the new file failed its own degenerate-sweep guard at 324 < 324 -- every phase at r >= 2 mm was non-empty; a 1 mm radius was added to the sweep so the guard can fail) |
| freshness | `libWireCellClus.so` | 08:27:03 against a last source edit of 08:26 (`Facade_Grouping.cxx`); test rebuild 08:27:49 / 08:38:41 (final) after the test fix |
| compiled config, PR job | `.wct-pr_d36cfg.json` knob off vs the pre-change compile | `cmp` identical, 280 559 bytes; knob on: the key appears exactly once, in `MultiAlgBlobClustering:clus_pr`, 60 nodes both ways, no other node differs |
| **G1** PDVD PR, 120 events | `d36off` (new pin, production config, no TLA) vs `d36refpr` (old pin, same config, same inputs) | **PASS: 120 / 120 `mabc-pr.zip` member-content rollups identical; 119 / 119 `calib-pr-evt*.json` md5 identical** (039349/78 has no calib dump in either arm -- zero STM tags, doc 25 §13.10). The first attempt compared against `d32p035` and got 120 / 120 *different*: doc 35 changed the production tagger FV at 07:49 (`02a0832f`, `89d7cef4`), after that arm ran at 05:52 -- TGM 8 -> 12 on 039349/9 is doc 35's own 1592 -> 2185. A baseline is only a baseline while the config is the one it ran with; the old-pin arm under today's config is the reference, and `d32p035` is retired as a gate reference. |
| **G2** PDVD clustering | `039252_2_d36cref` / `039349_14_d36cref` (old pin) vs `_d36cnew` (new pin); Q/L on, `PDVD_LIGHT_SUFFIX=_keep`, inputs symlinked from `d27fresh` | **PASS: 54 / 54 archives identical** by member content (27 per event: 8 anodes x {face0, face1, anode}, 2 groups, all-apa); all-apa rollups `3d8b766a8fc3b179` (298595) and `d54401550787983e` (19689) on both pins; wall 53 -> 52 s and 17 -> 13 s |
| **G2** PDHD clustering | `abtest/snap/d36cref_pdhd` vs `d36cnew_pdhd`, 4 events of `events.txt` | **`ab_compare.sh` OVERALL PASS**, 61 PASS lines, 0 FAIL, rc=0; wall 137 -> 143, 19 -> 15, 22 -> 18, 117 -> 118 s |
| **G3** SBND PR chain | `work-{nuecc48,ncpi0}-doc25d36old` vs `-doc25d36new` (67 events, `d99fix` Q/L source, DL off) | **PASS: 201 same / 0 diff / 0 missing** (per event: `mabc-pr.zip` member content, calib JSON minus the dual-chain timer, nusel TSV) -- `/home/xqian/tmp/doc25gate/sbnd_compare_d36new.txt` |
| **G3** uBooNE | `qlport/scripts/sweep/doc25d36old` vs `doc25d36new` (35 events) + `ab_check.sh` | **Bee zips 35 / 35 content-identical; tagger logs 34 / 35**, the one difference being event 5384-136-6805's four `kine_pio_*` values (`kine_pio_angle` 14.81 vs 109.51) -- the **known bistable event of doc 90** (memory: the same commit produces both states; the Bee zip is identical in both). Proven by repetition rather than by reasoning: `doc25d36old2` (old pin again) ≡ `doc25d36old` 35 / 35 tagger-identical; `doc25d36new2` (new pin again) ≡ `doc25d36new` 35 / 35; and **`doc25d36new3`, a code-identical relink of the new source** (post comment-fix build; `.text`, `.rodata`, `.data.rel.ro` byte-identical to `pin_d36new`, only the build ID differs) **≡ `doc25d36old` 35 / 35 tagger-identical** and differs from `doc25d36new` on exactly 6805. The pi0 state follows the link layout, not the code: **PASS**, with the bistable event recorded as such (M4: pattern-recognition order can be pointer-dependent). |

Two harness notes for whoever repeats this. `abtest/run_events.sh` cannot run
the two PDVD events of `events.txt` (`039349/0`, `039252/5`): their base work
dirs predate doc 27's `img-provenance.txt`, and `run_clus_evt.sh` runs under
`set -e`, so the `awk` that reads the missing provenance file exits the
runner with rc=2 **before wire-cell starts**, and the snapshot then copies the
stale August zips out of the work dir — a comparison of those two snapshots
would PASS on stale-vs-stale. The `snap/d36ref` label made at the start of
this round carries exactly those two stale entries (they were not deleted:
M13) and is superseded by `d36cref_pdhd`; the PDVD clustering gate ran on
tags staged from `d27fresh`, which has the provenance file. Second, the
PDVD light archives for these events live in `work/<run>_light<evt>_keep`, so
the clustering runner needs `PDVD_LIGHT_SUFFIX=_keep` to run Q/L matching as
production does. Third: `run_d36_arms.sh` was edited (two arm names added)
while its first invocation was still running; bash resumed at a stale byte
offset and the driver died with a syntax error *after* its last arm
(`d36on035`, 120 / 120 rc = 0, verified by count) had completed. Never edit a
running driver; every later invocation was a fresh process.

## 4. The measurement — new metric vs the 0.35 floor

Four arms, one binary (`pin_d36new`), one config epoch (today's, i.e. after
doc 35's tagger FV), 120 events, inputs symlinked from `d27fresh`:

| tag | metric | end-trim floor (`good_point_pitch_frac`) | meaning |
|---|---|---|---|
| `d36p000` | isotropic | 0 (TLA `tf_knoboff.json`) | the legacy reference, today's config |
| `d36refpr` ≡ `d36off` | isotropic | **0.35** (production JSON) | **production** |
| `d36on` | **anisotropic** | 0 | the recommendation of doc 34: metric *replaces* floor |
| `d36on035` | **anisotropic** | 0.35 | stacked |

`d32p000` / `d32p035` (doc 32 round 3b) are the same events under the
pre-doc-35 tagger FV; they are shown where the axis is the trajectory itself
(support), never for the tagger census.

### 4.1 Nothing stopped working

All four arms: 120 / 120 events, rc = 0, every `mabc-pr.zip` written.

### 4.2 Tagger census (`pr_arm_census_diff.py`)

| metric | `d36p000` | `d36refpr` (0.35) | `d36on` (aniso) | `d36on035` (stacked) |
|---|---|---|---|---|
| TGM = true | 2185 | 2185 | 2187 (+2, 31 events) | 2187 |
| **STM = 1** | 2049 | 2064 (+15) | **1971 (−78, −3.8 %; 101 events change; worst 039252/10 +21)** | 1959 (−90) |
| stm_eval (`persist_stm_fit`) | 2445 | 2485 (+40) | 2533 (+88) | 2534 |
| nu candidates | 5018 | 5034 (+16) | 5095 (+77) | 5100 |

The floor moved STM by +15 across the manifest; the metric moves it by −78,
with 101 of 120 events changing their count. That is a much larger churn than
the floor's, and in the opposite direction.

### 4.3 The SUPPORT axis over the manifest (`pr_arm_support120.py`)

Distance from each `stm_fit` point to the nearest reconstructed 3-D charge;
read as deltas — the baseline is not zero.

| tag | stm_fit points | > 2 cm from any charge | > 10 cm |
|---|---|---|---|
| `d32p000` (old FV) | 832 391 | 7.63 % | 3.26 % |
| `d32p035` (old FV) | 876 031 | 9.25 % | 4.08 % |
| `d36p000` | 842 911 | 8.82 % | 3.78 % |
| `d36refpr` (0.35, production) | 896 379 | 10.83 % | 4.77 % |
| **`d36on` (aniso)** | **953 441** | **13.51 %** | **6.16 %** |
| `d36on035` (stacked) | 956 177 | 13.61 % | 6.22 % |

The floor costs +2.0 points of unsupported trajectory for +6.3 % more
points; the metric costs **+4.7 points for +13.1 % more points**, i.e. more
than twice the floor's price per point kept. Stacking the floor on top of the
metric changes nothing measurable (+0.1 point) — §6.

### 4.4 The COVERAGE axis on the doc-32 event (`stm_trajectory_coverage.py`, 039252/2, evt 298595)

Long clusters (≥ 200 Steiner points, λ0/Σλ > 0.95, extent > 50 cm), shortfall
of the fitted trajectory against the terminal extent:

| arm | n | median shortfall, nearer end | further end | **total** | both ends covered |
|---|---|---|---|---|---|
| `d36p000` | 9 | 2.8 cm | 13.4 cm | 20.5 cm | 0 / 9 |
| `d36refpr` (0.35) | 9 | 1.1 | 12.1 | 13.3 | 1 / 9 |
| **`d36on` (aniso)** | 10 | **0.3** | **2.6** | **3.3** | **5 / 10** |
| `d36on035` | 10 | 0.3 | 2.6 | 3.3 | 5 / 10 |

On coverage alone the metric is by far the best arm ever run on this event:
median total shortfall 20.5 → 13.3 → **3.3 cm**, and half the long clusters
reach both terminal extremes where the floor managed one. This is the
prediction of doc 34 §6 coming true — and it is exactly the one-sided score
doc 32 §19 warned must never be read alone.

### 4.5 The predictor, closed

`ctpc_metric_census.py` on the `d36p000` arm's own interior trajectory points
(n = 7103) gives the all-3-plane pass rate 0.182 (iso 0.2 cm) → 0.289 (floor
0.35) → 0.645 (floor 0.50) → **0.700 (aniso 0.2 cm)** — doc 34 §6 predicted
0.175 / 0.287 / 0.651 / 0.703 on the doc-32 arm. The offline model of the
good-point test is right to a few parts per thousand. What it could not model
is what a trajectory *does* with a looser test, which is §5.

## 5. The tail verdict (cluster 109 of 298595)

Cluster 109 of evt 298595 is the 146 cm track whose fitted trajectory, under
any isotropic floor above 0.392 pitch, grew a **45 cm excursion past the B
end onto a branch** with no charge under it (doc 32 §19). Doc 34 §9 said the
anisotropic metric might or might not do the same — W's pitch reach rises
0.392 → 0.675 of a pitch under it, and whether the tail was carried by the
pitch axis or the drift axis was not established. `stm_endtrim_grade.py`
with the owner's A→B axis and the cluster's own core extent (149.4 cm):

| arm | fit points | within the cluster's own extent | **past it (a tail)** | trajectory extent |
|---|---|---|---|---|
| `d36p000` | 215 | 215, median 0.43 cm from charge, 0.0 % > 2 cm | — | t = [17.2, 144.9] cm |
| `d36refpr` (0.35) | 590 | 518, median 0.46 cm, 7.7 % > 2 cm | 72 pts, median 4.6 cm from charge, 80.6 % > 2 cm | [4.3, 153.7] |
| **`d36on` (aniso)** | 852 | 563, median 0.47 cm, 12.6 % > 2 cm | **289 pts, median 11.1 cm from charge, 91.0 % > 2 cm** | **[3.5, 195.2]** |
| `d36on035` | 852 | identical to `d36on` | identical | identical |

**The tail is back: −46.3 cm past the B end**, the same excursion the
isotropic 0.50 / 0.60 floors grew in doc 32 §19 (−46.0 cm). Keeping the drift
axis pinned at 2.0 mm did not prevent it. The mechanism runs through the
**pitch axis**: with three planes each allowed ±5.2 mm (U/V) and ±3.4 mm (W)
of pitch reach, a trajectory point 11 cm from any 3-D charge still finds
*some* 2-D charge within the ellipse on every plane — the projections of
other objects — and the end trim, whose whole purpose is to pop such points,
keeps them. That answers doc 34 §9's open question, and it is the answer
that refutes the doc-34 recommendation as stated.

Two things worth recording beside it. First, under today's tagger FV the
production 0.35 floor already carries a small tail on this cluster (72
points, 4.3 cm past the extent, median 4.6 cm from charge) that doc 32's
arm did not show; the doc-35 FV change moved this event slightly, and the
0.392 threshold was always within 0.02 of the floor's own W reach. Second,
the STM verdict on cluster 109 flips 1 → 0 in *every* arm that loosens the
trim, the floor included — the tail is not merely cosmetic.

## 6. Stacked vs subsumed

`d36on` and `d36on035` differ only in the end-trim floor (0 vs 0.35 on top
of the metric). Across the 120 events they are indistinguishable at every
level measured: STM 1971 vs 1959, stm_eval 2533 vs 2534, nu candidates 5095
vs 5100, unsupported fraction 13.51 vs 13.61 %, and on the doc-32 event the
per-cluster support, coverage and cluster-109 tail are **identical to the
point**. Under the metric the floor's 2.68 mm drift semi-axis on U/V versus
2.00 mm changes nothing a trajectory does — the pitch axis, at 5.2 mm, is
what governs. Doc 34 §8's "subsume, do not stack" is confirmed mechanically:
if the metric were ever turned on, `good_point_pitch_frac` would be dead
weight and should be retired in the same commit.

## 7. Cost

Per-event wall and peak RSS from `pr_resource_*.txt`, summed over the
manifest (16 → 12 jobs this round, so only same-round arms are comparable):

| arm | wall (s) | peak RSS (GB) |
|---|---|---|
| `d36p000` | 5096 | 214.2 |
| `d36refpr` (0.35) | 5731 | 213.9 |
| `d36on` (aniso) | 5620 | 207.5 |
| `d36on035` | 5193 | 208.1 |

The spread between `d36refpr` and `d36on035` (−9.4 %), two arms that differ
by nothing that should cost time, is the noise floor of a 12-way parallel
batch on a shared box. Within it, the two-level query is **not measurably
slower** than the isotropic one on the PR job: `d36on` sits between the two
isotropic arms, and the knob-off `d36off` and `d36refpr` (same code path,
different pins) are 5731 s both. The offline bound of doc 34 (2.31×
candidate inflation) is a per-query count, not a job cost; on this job the
early-exit brackets and the small absolute candidate counts absorb it. Peak
RSS is 3 % lower with the metric on — fewer, shorter connector graphs, not a
property of the query.

## 8. Recommendation and what a production flip would take

**Do not flip.** Leave `ctpc_aniso_metric` OFF in PDVD production and keep
`good_point_pitch_frac = 0.35`. The metric does what doc 34 said it would do
to the *test* (0.182 → 0.700 pass rate, best coverage ever measured on the
doc-32 event, 5 / 10 long clusters reaching both ends), and that is precisely
what makes it unsafe for the *trajectory*: the end trim's job is to remove
unsupported points, and a test that passes 70 % of interior points also
passes the 2-D ghosts under a 46 cm excursion. Manifest-wide it costs
+4.7 points of unsupported trajectory and −78 STM tags for the coverage it
buys.

What the round establishes for the next one:

1. **The tail is a pitch-axis effect.** Any test that reaches past ~0.39 of
   a W pitch grows it, isotropic or not. The drift axis is innocent (doc 34
   §9's question, closed). So the end trim cannot be fixed by *any* choice of
   metric on the good-point test alone; the fix is where doc 32 §2.2 put it,
   in endpoint selection — or in a support test that is genuinely 3-D (distance
   to reconstructed charge, which `stm_endtrim_grade.py` already computes
   offline) rather than three 2-D projections.
2. **The metric is not wrong, the radius is.** Under the lattice-normalised
   metric the single tuning parameter is `r / drift_step` (doc 34 §5). At
   0.2 cm it is 0.675 pitch on every plane, above the tail threshold. A
   per-call radius that keeps W below 0.39 pitch (r < 1.16 mm) is below the
   drift step's own half-width (1.48 mm) and would fail the drift axis
   instead — there is no radius that satisfies both, which is a statement
   about the *test*, not about the metric. A pitch-axis cap (an ellipse whose
   pitch semi-axis is bounded at a fraction of a pitch while the drift
   semi-axis stays `r`) is the natural next knob, and it is a one-line
   change inside `ctpc_yscale`'s caller; it was not built here because it is
   a new tuning surface and §5.7 says report, do not tune.
3. **The connectors and charge averages are untested in isolation.** The
   knob is grouping-wide, so `d36on` also changed `test_good_point` at 0.6 cm
   in `protect_bundle` / `steiner` and `get_ave_charge` at 0.3 cm in the
   vertex finder. The census shows the effect (nu candidates +77, stm_eval
   +88) but nothing here attributes it. An arm with the metric on only inside
   the end trim (a per-call switch, the route this round deliberately did
   not take) would separate the two.
4. **If the owner does flip it** — for the coverage, accepting the support
   cost — the change is one line in `pdvd/wct-pr-perevt.jsonnet`
   (`ctpc_aniso_metric = true`) and, in the same commit, `good_point_pitch_frac`
   removed from `pdvd_track_fitting.json` (§6), with the compiled-config
   proof and the `d36on` arm as its gate. The clustering job stays on the
   isotropic metric until it gets its own arm (§9).

## 9. Not done in this round

- **The clustering job.** `protodunevd/clus.jsonnet`'s four
  MultiAlgBlobClustering nodes do not carry the key. Turning the metric on
  there changes the pctree itself (every connector, `Separate_overclustering`,
  the Steiner graphs) and needs its own arm, gated at the clustering stage.
  With the PR job on and the clustering job off, the two stages use different
  metrics on the same ctpc; that is where this round leaves PDVD if the flip
  is taken, and it is a known inconsistency, not an oversight.
- **PDHD** is exposed (`s ≈ 0.67`) and unmeasured — no PR chain exists for it.
- **uBooNE** is *not* the identity under the metric: its drift step
  (2.20 mm) is finer than its pitch (3.00 mm), so `s = 0.73`. It is gated OFF
  here (G3) and must stay OFF — uBooNE is a frozen reference.
- **Families 2–4** of doc 34 §3 (the continuous 2-D projections) are
  untouched.
- **`is_good_point_wc`** is covered by the switch but has no production
  caller; nothing measures it.

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

> **CORRECTED THE SAME DAY — read §10 before §5 or §8.** Two claims above are
> wrong. (a) The STM figure is a LOG-LINE count: `pr_arm_census_diff.py` greps
> the string `STM=1`, which every verdict prints three times (once by
> `TaggerCheckSTM`, twice by `ClusteringProtectBundle`). The tagged-CLUSTER
> census is 688 → 657, **−31 of 688 (−4.5 %)**, not −78 of 2064. (b) The tail
> is **not** a pitch-axis ghost-support effect: 281 of the 289 tail points FAIL
> the ellipse test as well. The tail survives because the end trim is a
> TIP-ONLY test that used to walk the chord back by accident, on the isotropic
> test's 82 % false-fail rate on real charge. §10 has the mechanism, a
> population census that also shows the 0.35 floor doing the same thing at
> nearly the same rate, and the two-axis grade the decision was finally taken
> on. §11 records the owner's flip.

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
| **STM = 1** (log lines — see the note) | 2049 | 2064 (+15) | 1971 | 1959 |
| **STM = 1** (tagged clusters, corrected) | 683 | 688 (+5) | **657 (−31, −4.5 %)** | 653 (−35) |
| stm_eval (`persist_stm_fit`) | 2445 | 2485 (+40) | 2533 (+88) | 2534 |
| nu candidates | 5018 | 5034 (+16) | 5095 (+77) | 5100 |

**Correction (§10.1).** The `STM = 1` row as first published counts log
LINES, not clusters: `pr_arm_census_diff.py` matches the bare string, and each
verdict is echoed by two `ClusteringProtectBundle` debug lines on top of the
`TaggerCheckSTM` one. Anchoring the census on the line that carries the cluster
id (`TaggerCheckSTM: cluster N → STM=d TGM=d`) gives 683 / 688 / 657 / 653
distinct tagged clusters out of 4950 / 4950 / 4948 / 4948 evaluated, with no
cluster evaluated twice in any arm. The ratio survives the inflation, which is
why it went unnoticed; the absolute number does not. **The metric costs 31
stopping-muon tags of 688, not 78 of 2064.** The `TGM`, `stm_eval` and
`nu candidate` rows are unaffected (their patterns match one line each).

The floor moved STM by +5 across the manifest; the metric moves it by −31,
and 204 clusters lose the tag while 170 gain it — the churn is two-sided, not
a uniform loss (§10.2).

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
axis pinned at 2.0 mm did not prevent it.

> **The mechanism stated here in the first version of this doc was WRONG and is
> withdrawn.** It read: "the mechanism runs through the pitch axis — a
> trajectory point 11 cm from any 3-D charge still finds *some* 2-D charge
> within the ellipse on every plane, and the end trim keeps them." It does not.
> Replaying the three-plane test offline on all 289 tail points, **281 fail the
> ellipse** (per-plane pass 5 % U, 24 % V, 6 % W) and 0 of 289 pass under the
> isotropic or 0.35-floor tests. The metric admits no ghost support along the
> chord. The eight points that do pass sit on real charge. §10.1 has the actual
> mechanism: `examine_end_ps_vec` is a **tip-only** test — it pops from each end
> until the first point that PASSES and never examines the interior — so the
> only thing that differs between the arms is whether the far endpoint's own
> on-charge tip points pass, and under the isotropic test they falsely fail
> 82 % of the time.

Two things worth recording beside it. First, under today's tagger FV the
production 0.35 floor already carries a tail on this cluster that doc 32's arm
did not show. The grader's "past the extent" count (72 points) is an
axis-projection and over-states it — split by distance to charge, those 72 are
14 on charge, 25 within 2–5 cm and **33 chord points**, a 37 cm excursion
reaching 18.3 cm from any charge. The metric's 289 split 26 / 40 / **223**, a
159 cm chord path. The frac-0 arm has none. So on the flagship cluster
production is not the clean case: the floor already put 37 cm of trajectory
into empty space, and the metric enlarges an existing production defect rather
than creating one. Second, the STM verdict on cluster 109 flips 1 → 0 in
*every* arm that loosens the trim, the floor included — only `d36p000` keeps
the tag — so the tail is not merely cosmetic.

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

## 8. Recommendation as it stood at the end of round 1 — SUPERSEDED by §10–§11

> Kept as the record of what this round concluded on its own evidence. Item 1
> below is withdrawn (§10.1) and the recommendation itself was overturned by
> the owner on the same day (§11) after the population census of §10.2–§10.3.

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

1. ~~**The tail is a pitch-axis effect.**~~ **WITHDRAWN — see §10.1.** The
   chord fails the ellipse test at 281 / 289 points; there is no ghost support
   to blame. What survives of this item is its last clause, and it is now the
   whole story: the fix is where doc 32 §2.2 put it, in endpoint selection —
   or in a trim that is gap-aware rather than tip-only.
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

## 10. Follow-up the same day — the mechanism, the population, and the grade the decision was taken on

The owner's question after §5: *"focus on the 46 cm tail on cluster 109 and
compare 0.35 vs the two-step case, to understand why it shows up on the latter.
We are tuning something fundamental, and things built upon it may need to
retune — changing it, not changing the higher-level things, and concluding it
is not good, would not be a good approach."*  That is exactly what happened in
§5, and the answer below reverses it.

### 10.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
SC=docs/nf_sp_img_clus/scripts
# is a cluster id the same object in both arms?  (98.41 % yes; the charge point
# set is bit-identical, so a per-cluster join is legitimate)
python3 $SC/d36_cluster_id_stability.py d36off d36on 50
# per-cluster STM/TGM verdict flips, anchored on the line carrying the cluster id
python3 $SC/d36_stm_verdict_flips.py d36off d36on
# per-cluster fit-vs-charge support, every arm
python3 $SC/d36_fit_support_scan.py  d36p000,d36off,d36on  /tmp/support.tsv
# what each step ADDS to / REMOVES from a trajectory, and is it on charge?
python3 $SC/d36_fit_added_scan.py    d36off  d36on   /tmp/added_off_on.tsv
python3 $SC/d36_fit_added_scan.py    d36p000 d36off  /tmp/added_p000_off.tsv
# the two-axis grade (coverage vs support) -- sec 10.3
python3 $SC/d36_fit_twoaxis_scan.py  d36p000,d36off,d36on  /tmp/twoaxis.tsv
# the per-cluster table of sec 10.4
python3 $SC/d36_cluster_support.py   d36p000,d36off,d36on <<'IN'
039252 2 109
IN
# the hand-scan Bee sets, then pdvd/upload-to-bee.sh on each zip
python3 $SC/d36_build_bee_sets.py /home/xqian/tmp/d36bee
```

Validity of the per-cluster join: the three arms read the same `d27fresh`
pctree, so each event's Bee `clustering-global` point set is **bit-identical**
across arms (lexsorted arrays equal, 0 events differ). The PR stage does
re-cluster slightly — cluster counts move by 4–6 per event and 0.04–0.14 % of
points change owner — but 4776 of 4853 clusters with ≥ 50 points (98.41 %)
keep their id at > 90 % overlap, and all five clusters showcased below have
Jaccard 1.000 across the three arms.

### 10.1 The mechanism: the end trim is a TIP test, and it was tuned on a broken good-point test

`TrackFitting::examine_end_ps_vec` (`clus/src/TrackFitting.cxx:2279+`) pops
points from each end of the fitted path **while** `is_good_point(…, 0.2 cm, 0,
0, pitch_frac)` fails, and **breaks at the first point that passes**. It never
looks at the interior. After the pop it interpolates back toward the popped
point in 0.2 cm steps and re-inserts the first passing test point.

Cluster 109 of evt 298595 is three things in one cluster: the 146 cm main
track A→B; a real second segment beyond B (a kink to ≈ (269, 204) cm then a
short piece up to (248, 233), x 236–238); and a **five-point fragment** at
(211.83, 267.04, 287.66) — 5.7 cm from the cluster's other small blobs, 47.9 cm
from any other cluster, and ≈ 55 cm past the end of the real charge.
Endpoint selection (`get_two_boundary_wcps`, the doc 32 §2.2 defect) picks that
fragment as the far endpoint, so the raw fitted path **always** contains a
55 cm chord through empty space, in every arm.

Replaying the strict three-plane test offline on all 289 points of the
aniso arm's tail:

| test | tail points passing all three planes | per-plane pass (U / V / W) |
|---|---|---|
| isotropic 0.2 cm | 0 / 289 | — |
| isotropic + 0.35 floor | 0 / 289 | — |
| **anisotropic ellipse** | **8 / 289** | 5 % / 24 % / 6 % |

and the eight that pass are on real charge (six at the top of the second
segment, two at the fragment's tip). **The metric admits no ghost support
along the chord.** What differs between the arms is only the *tip*:

| arm | fragment tip passes? | what the pop does | result |
|---|---|---|---|
| frac 0, isotropic | no | walks back the chord **and the real second segment** | stops at B — 50 cm of real charge amputated (the doc 32 defect) |
| 0.35 floor (production) | no | walks back the chord | stops at the top of the second segment — right, by accident |
| anisotropic | **yes** (its points are 0.3–0.6 cm from charge) | pops nothing | chord kept: the 46 cm tail |

Under the isotropic test a point sitting on real charge fails 82 % of the
time; the production result on this cluster was that false-fail rate landing on
the right five points. The metric passes the tip because the tip **is** on
charge — which is what the test is for. Run structure inward from the tip under
the metric: (pass 1), (fail 90 = 54 cm), (pass 1), (fail 6), (pass 1),
(fail 62), (pass 3), (fail 25); on-charge pass rate along the trajectory 0.91.

**So the fundamental is right and the layer above it is what needs retuning.**
A gap-aware trim — after finding a passing tip, look inward and discard the tip
if the run of failing points exceeds a few cm — is writable only *with* the
metric: at 0.91 on-charge pass the failing runs on real charge are 1–6 points,
against a 90-point run across the chord. Under the isotropic test, where
failing runs are everywhere, that rule cannot be stated at all. That is doc
38's subject.

### 10.2 The population — the 0.35 floor already does this, the metric does more of it

Not one cluster: 2354 clusters carry a fit in some arm. A point is *added* by a
step if it is more than 2 cm from every point of the other arm's fit for the
same cluster.

| step | added points | > 2 cm / > 10 cm from any charge | removed points | > 2 cm |
|---|---|---|---|---|
| frac 0 → 0.35, extensions of an existing fit | 45 958 | 48 % / 23 % | 4 095 | 50 % |
| **0.35 → metric, extensions** | **49 880** | **64 % / 31 %** | 18 525 | 29 % |
| 0.35 → metric, wholly new fits (67 clusters that had none) | 12 612 | 29 % / 14 % | — | — |

Ghost extension is **already production behaviour**: per added point the 0.35
floor is at 48 % / 23 % and the metric at 64 % / 31 %. The metric is not
introducing a new failure mode; it is scaling one up — and it separately buys
67 clusters a first fit, most of it on charge.

Classifying the clusters whose fit diameter moves by more than 20 cm
(0.35 → metric): **95 gain a ghost the floor did not have, 10 lose one the
floor had, 8 are pure coverage wins**, and 26 % of all fitted clusters are
unchanged. The STM churn is two-sided in the same way: 204 clusters lose the
tag, 170 gain it (§4.2).

### 10.3 The two-axis grade — the number the decision turns on

`d36_fit_twoaxis_scan.py`, 2176 clusters with ≥ 50 charge points and a fit in
some arm. **Coverage** = the fraction of a cluster's own 3-D charge within 2 cm
of its fit (what doc 32 was trying to buy). **Support** = the fraction of fit
points more than 2 / 10 cm from any 3-D charge of the event (what it costs).

| arm | clusters fitted | own charge covered | fit points | > 2 cm off charge | > 10 cm |
|---|---|---|---|---|---|
| `d36p000` (frac 0) | 2134 | 68.3 % | 834 792 | 8.3 % | 3.5 % |
| `d36off` (production 0.35) | 2148 | 70.3 % | 886 261 | 10.2 % | 4.5 % |
| **`d36on` (metric)** | 2158 | **71.1 %** | 939 184 | **12.6 %** | 5.8 % |

The exchange rate is the finding: the 0.35 floor bought **+2.0 points of
coverage for +1.9 points of ghost** (1.05); the metric buys **+0.8 for +2.4**
(0.33). With the end trim unchanged the metric is on the wrong side of the same
trade production already makes. It is the trim, not the metric, that sets that
rate — which is why §11 flips the metric and opens doc 38 on the trim.

### 10.4 The hand-scan Bee sets

Five sets, uploaded 2026-09-04, built by `d36_build_bee_sets.py` and
content-verified server-side (3 events each, DAQ id and per-cluster fit point
counts checked against the arms they were built from). **The Bee event index is
the arm**: 0 = frac 0 isotropic, 1 = production 0.35, 2 = anisotropic metric.
`cov` = own charge within 2 cm of the fit; `off` = fit points > 2 cm from any
charge.

| set | event / cluster | UUID | what it shows |
|---|---|---|---|
| A | 039252/2 cl 109 | `fcd75df5-33ca-478f-acce-52b15c43469d` | the 46 cm tail |
| B | 039349/48 cl 53 | `969be515-8ff6-4764-969a-28f5b65216e5` | a ghost extension that *creates* an STM tag |
| C | 039252/16 cl 103 | `4d638988-0891-4396-9bd1-168bf169436e` | the largest ghost extension in the manifest |
| D | 039349/58 cl 68 | `3526d506-ecff-42ce-8248-6f183184dc2e` | production's **own** 67 cm ghost, removed by the metric |
| E | 039349/3 cl 71 | `919f6569-a8e6-4d08-a349-9e8ef495e057` | a 434 cm cluster production never fits at all |

| set | charge | arm 0 (frac 0) | arm 1 (production 0.35) | arm 2 (metric) |
|---|---|---|---|---|
| A cl 109 | 1049 pts, 205 cm | 128 cm, cov 79 %, off 0 %, **STM=1** | 150 cm, cov 90 %, off 18 % | 202 cm, cov 91 %, off 41 % |
| B cl 53 | 1042 pts, 281 cm | 143 cm, cov 78 %, off 0 % | 151 cm, cov 83 %, off 0 % | 246 cm, cov 84 %, off 38 %, **STM=1** |
| C cl 103 | 3152 pts, 408 cm | 343 cm, cov 86 %, off 10 % | 343 cm, cov 86 %, off 10 %, **STM=1** | 407 cm, cov 87 %, off 28 % |
| D cl 68 | 76 pts, 74 cm | 4.7 cm, cov 7 %, off 0 % | 67 cm, cov 33 %, off 86 %, **STM=1** | 0.4 cm, cov 11 %, off 0 % |
| E cl 71 | 2862 pts, 434 cm | no fit | no fit | 380 cm, cov 95 %, off 0 %, **STM=1** |

B is the case that stops "STM gained" from being read as "better": the metric's
95 cm extension is 38 % off charge and it is what earns the tag. D is a
production-only defect — a 67 cm fit that is 86 % in empty space on a 76-point
cluster, and production is the arm that calls it a stopping muon — but the
metric's 3-point answer is not a fit either (11 % coverage): **neither arm
reconstructs that object**, and that is a separate defect nothing here fixes.
E is an unambiguous win.

## 11. Owner decision, 2026-09-04: flip PDVD PR to the metric

> *"I would like PDVD to update to this way of doing CTPC. For SBND, I want it
> to stay with the old ways; we can then work on the later part step by step to
> improve the performance — this is the first step."*

Accepted on the §10 evidence, with the cost stated: **+0.8 points of trajectory
coverage, +2.4 points of unsupported trajectory, −31 stopping-muon tags of
688**, and 95 clusters gaining a ghost extension against 10 losing one. The
owner's framing is the reason the §8 recommendation does not survive: the
metric is the fundamental, the trim is built on top of it, and grading the
fundamental through an untuned consumer is what §5 did wrong.

**What the flip is** (one production change, three files):

| file | change | why |
|---|---|---|
| `pdvd/wct-pr-perevt.jsonnet` | `ctpc_aniso_metric = false` → `true` | the PDVD PR operating point lives in the driver, not in `pr.jsonnet` |
| `cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` | `good_point_pitch_frac` `0.35` → `0` | the metric subsumes the floor (§6); they must not stack |
| `cfg/pgrapher/experiment/protodunevd/pr.jsonnet` | comment only — the `ctpc_aniso_metric` arg's "PDVD production stays OFF" note | the arg default stays `false`; that file documents the SBND operating point |

**What does NOT change.** The C++ default of `ctpc_aniso_metric` stays
`false`, so SBND, PDHD and uBooNE are untouched *structurally*, not by
convention: a config that does not mention the key gets the legacy metric, and
none of theirs does. SBND is explicitly staying
isotropic at the owner's instruction. The PDVD **clustering** job
(`protodunevd/clus.jsonnet`, four MultiAlgBlobClustering nodes) also stays
isotropic: wiring it changes the pctree itself and would invalidate every
measurement in this doc, all of which share one `d27fresh` pctree. That is a
separate round with its own imaging baseline (§9).

**Gate for the flip.** This is not a knob-off byte-identity gate — the flip is
an intended production output change. What must be proven is *equivalence to
the measured arm*: the post-flip default config must reproduce `d36on`
member-for-member, because `d36on` ran with `-A trackfitting_config=<the
49-key file with the key ABSENT>` while the flipped production file carries an
explicit `"good_point_pitch_frac": 0`. `load_trackfitting_config`
(`TaggerCheckSTM.cxx:1023-1031`) calls `set_parameter(name, 0.0)`, which is the
same value the C++ default holds, so they should be identical — but that is a
claim about a loader, and it is cheap to prove on two events from different
runs with `hash_archive.py --members`.

**Status: the config edits are HELD, not yet made.** A concurrent session is
running three 120-event PDVD PR arms (doc 37, `d37off0` / `d37off1` /
`d37on05`) that compile `wct-pr-perevt.jsonnet` and read
`pdvd_track_fitting.json` per event; editing either mid-flight would split
their arms across two configurations and void their gates. The flip lands after
their arms finish and their commits land. Sequencing agreed between the two
sessions; the pre-flip arm recipe afterwards is
`-S ctpc_aniso_metric=false -A trackfitting_config=<the 0.35 file>`.

### 11.1 Disclosed cost the count-based census hid: the tagger SET

Added 2026-09-04 after the flip was pushed, prompted by doc 37 §15.5.
`d36_tagger_set_diff.py` on the same 120 events, holding everything but the flip
fixed (`d36off` → `d36on`: both pre-doc-37-thinning, same pin, same
`dl_weights`), compares the *set* of tagged cluster ids rather than its size:

| tagger | events whose tagged SET changes | mean symmetric difference | total tagged |
|---|---|---|---|
| **STM** | **117 / 120 (97.5 %)** | 3.23 clusters | 688 → 657 |
| **TGM** | **33 / 120 (27.5 %)** | 0.30 clusters | 2185 → 2187 |

The TGM row is the one worth pausing on: §4.2 reports the TGM count moving by
**+2** across the manifest, which reads as nothing, while **more than a quarter
of events change which clusters are tagged as through-going**. PDVD runs its
per-bundle PR on the tagged set, not on its size, so a count-flat census is not
evidence that downstream work is unchanged. Same trap as §4.2's log-line count,
in the other direction.

**Is set churn just what any change does?** No — and STM alone would not have
told us. The same instrument on the same 120 events, one change per row:

| change | STM set differs | mean | TGM set differs | TGM count |
|---|---|---|---|---|
| **this flip** (`d36off` → `d36on`; thinning off both sides, one pin) | 117/120 (97.5 %) | 3.23 | **33/120 (27.5 %)** | 2185 → 2187 |
| doc-37 terminal thinning (`d36on` → `d38off`; metric on both sides, also crosses the binary) | 95/120 (79.2 %) | 1.78 | **0/120** | 2187 → 2187 |
| doc-38 gap trim at 20 cm (`d38off` → `d38h20`; everything else fixed) | 107/120 (89.2 %) | 2.01 | **0/120** | 2187 → 2187 |

Every change of this class reshuffles the STM set on most events — 79 to 98 % —
so the STM row is not what distinguishes this flip. **TGM is.** It is the only
one of the three that moves a through-going verdict at all, and the reason is in
§2: `TaggerCheckTGM` is one of the two direct external callers of the changed
query (`TaggerCheckTGM.cxx:1209-1211`), while the thinning and the trim reach
the STM path only. That is what makes 27.5 % a property of the metric rather
than of "any config change moves sets".

The three-row framing is owed to the doc-37 session, which measured the same
contrast on 20 events and corrected its own earlier reading twice to get there;
the rows above are re-derived here on 120 events with the arms named.

This is a disclosed cost of a flip that is already in production, not a reason
to revert: the flip's coverage/support case (§10.3) is unchanged, and the
gap-aware trim of doc 38 takes STM back past its pre-flip value (685 → 720 on
its own epoch). It is recorded because 27.5 % of events changing a cosmic-tagger
verdict is not something to ship silently.

Independent corroboration: doc 37 §15.3 first attributed these same rates
(TGM 27.5 %, STM 97.5 %) to `dl_weights`, from a 20-event pair that turned out
to span this flip. That reading is retracted (doc 37 §15.5, a 20-event arm with
every factor separated). The 120-event measurement above holds `dl_weights`
fixed and reproduces the rates, which is what identifies the flip — not the
vertex — as their cause.

## 12. Doc 38 — the gap-aware end trim (opened by this round)

The retune §10.1 names, and the reason the exchange rate of §10.3 is not the
metric's fault. A default-OFF `TrackFitting` knob: after the pop loop finds the
first passing tip, walk inward; if the arc length of the run of consecutive
failing points before the next passing point exceeds the threshold, the tip is
a detached island — pop it and resume. Off (0) is byte-identical. Sweep
L ∈ {3, 5, 10} cm against post-flip production and grade with
`d36_fit_twoaxis_scan.py`: the target is coverage holding at ≈ 71 % while the
> 2 cm fraction falls below production's pre-flip 10.2 %. If coverage falls
materially, L is amputating real ends — report it, do not tune past it.

One risk to measure rather than design around: a long failing run is also what
a genuine **dead-channel region** looks like, and this test is the strict
three-plane one with no bad-plane allowance. Count, in the sweep arms, how many
popped tips sit next to a dead area (the `channel-deadarea-*` layers are in
every Bee zip). A large count means the rule needs a dead-channel exemption —
that is a finding, not a failure.


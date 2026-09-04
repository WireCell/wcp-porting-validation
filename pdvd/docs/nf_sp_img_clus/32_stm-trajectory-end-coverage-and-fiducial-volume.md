# PDVD doc 32 — the STM trajectory stops short of both track ends, and what PDVD's fiducial volume actually is

**Status.** Diagnosis complete and pinned to one mechanism (rounds 1, §§1-9);
the *reason* PDVD is affected and SBND is not is now measured and **confirmed**
(round 2, §§10-16). **No code and no config changed in either round**, so no A/B
gate is owed and none is claimed. The two arms in §2 use only knobs that are
default-OFF and log-only, and both reproduce the shipped stack's STM output
exactly (§2.1); round 2 ran no arm at all.

> **Line numbers.** `clus/src/TrackFitting.cxx` and
> `clus/inc/WireCellClus/TrackFitting.h` are **dirty in this working tree**
> (a peer's uncommitted +422/-155), so every `TrackFitting.cxx:NNNN` below is a
> working-tree line and will shift against HEAD `8bf5b1bc`. The config and
> `Facade_*` citations are on clean files. The pinned library used for both arms
> in section 2 was built from that dirty tree; section 2.1's first gate shows it
> reproduces the shipped stack's STM output on this event regardless.

**Scope.** Two questions the owner asked together, which turn out to share a
code path (§5):

1. In run 039252 / event 298595, the Bee `stm_fit-global` layer for **cluster
   109** does not span the cluster — the trajectory is missing at both ends.
   Why?
2. How is the fiducial volume (FV) defined for the whole detector, compared with
   SBND, given the goal that **the entire detector** be the basis of PDVD's FV?

The three points the owner named:

| role | (x, y, z) cm |
|---|---|
| on the fitted trajectory | (281.5, 169.2, 228.0) |
| **end A — not covered** | (343.1, 140.1, 195.3) |
| **end B — not covered** | (221.4, 196.8, 253.7) |

**The answer in one paragraph.** The STM tagger does exactly what the owner
expected: it picks the two ends and builds a trajectory between them. On cluster
109 the path handed to the fitter reached to **3.3 cm** of end A and, after the
first end-cleanup call, to **0.9 cm** of end B. `TrackFitting`'s
`examine_end_ps_vec` then amputated **15.5 cm off the A end** and **11.0 cm off
the B end**. The Steiner terminals were never the problem — they cover the whole
track with a largest gap of 3.1 cm, and 39 + 27 of them were left stranded
outside the final fit. The amputation loop pops trajectory points while either
`DetectorVolumes::contained_by()` is invalid or `is_good_point(..., 0.2 cm, 0,
0)` fails; the first of those is the same holed sensitive-volume union that §6
shows is also PDVD's *de facto* fiducial volume, which is why the two questions
are one.

---

## 0. Repro

```bash
WCPI=/home/xqian/toolkit-dev/wcp-porting-img

# ---- coverage measurement (read-only, on arms already on disk) ----
cd $WCPI/pdvd
python3 docs/nf_sp_img_clus/scripts/stm_trajectory_coverage.py \
    d31r6e2e:work/039252_2_d31r6e2e \
    d31r7t500:work/039349_14_d31r7t500 \
    --axis 109:343.1,140.1,195.3:221.4,196.8,253.7

# ---- round 2: the pitch measurement (read-only, no arm) ----
# Reads the ctpc lattice straight out of the pctree tensor dumps, so it needs
# only products already on disk.  ~4 min for PDVD, ~1 min per SBND event.
cd $WCPI/pdvd
python3 docs/nf_sp_img_clus/scripts/goodpoint_pitch_census.py \
    PDVD:work/039252_2_d31r6e2e
python3 docs/nf_sp_img_clus/scripts/goodpoint_pitch_census.py \
    $(ls -d ../sbnd/sbnd_xin/work-mcp2k-d97fvpr2/pr_evt* | head -16 | sed 's/^/SBND:/')

# ---- the two probe arms (section 2) ----
# Pin first: local/lib and build/ were rebuilt at 20:49, AFTER the d31r6e2e arm
# ran at 18:00, and the tree carries uncommitted peer edits to TrackFitting.cxx.
#   cp -a local/lib/. <pin>/libpin/ ; cp -a toolkit/build/apps/{wire-cell,wcsonnet} <pin>/binpin/
#   export LD_LIBRARY_PATH=<pin>/libpin:$LD_LIBRARY_PATH ; export PATH=<pin>/binpin:$PATH
mkdir -p work/039252_2_d32base work/039252_2_d32probe
for t in d32base d32probe; do
  ln -sfn ../039252_2_d31r6e2e/pctree-evt298595.tar.gz work/039252_2_$t/pctree-evt298595.tar.gz
  ln -sfn ../039252_2_d31r6e2e/pctree-evt298595.tlas   work/039252_2_$t/pctree-evt298595.tlas
done
./run_pr_evt.sh -s d32base -stm-fit 039252 2 > pr_d32base.log 2>&1; echo rc=$?
PDVD_PR_TLA="-S traj_cover_probe=true -A trackfitting_config=$WCPI/pdvd/stm/pdvd_track_fitting_probe.json" \
  ./run_pr_evt.sh -s d32probe -stm-fit 039252 2 > pr_d32probe.log 2>&1; echo rc=$?

# ---- read the verdict ----
grep -a "pr67 examine_end_ps_vec" work/039252_2_d32probe/wct_pr_039252_2.log | grep -a "cluster=109"
```

Arms used, all read-only except the two fresh tags this round created:

| tag | what |
|---|---|
| `work/039252_2_d31r6e2e` | the doc 31 round-6 end-to-end arm — the shipped stack |
| `work/039349_14_d31r7t500` | the doc 31 round-7 arm, second event |
| `work/039252_2_d32base` | **new** — production config on the pinned binary; the reference |
| `work/039252_2_d32probe` | **new** — same + `traj_cover_probe`; the diagnostic |

Nothing under `work/`, `abtest/snap/`, `sweep/`, `decisions*/` or `ql_labels/`
was deleted, overwritten or regenerated. Both new arms take their `pctree` as a
read-only symlink into the round-6 arm.

---

## 1. The symptom, measured

Along the owner's own end-A→end-B segment (146.41 cm straight line), over the
938 of 1049 cluster-109 points inside an 8 cm tube around it:

| quantity | value |
|---|---|
| track core extent | t = 0.0 → **149.4 cm** |
| Steiner terminals in the tube | 291, extent t = 0.0 → **148.9 cm** |
| terminal gaps along the axis | median **0.36 cm**, largest **3.14 cm** |
| `stm_fit` trajectory | 197 pts, t = **18.6 → 136.9 cm**, arc 123.9 cm, median step 0.60 cm, **one continuous piece** |
| **uncovered** | **18.6 cm at end A**, **12.0 cm at end B** |
| **terminals stranded outside the fit** | **39 at the A end, 27 at the B end** |

The PR log agrees on the fit: `persist_stm_fit: cluster 109 stmfit pass=0
status=3 kink=197 exit_L=124.6 left_L=0.0 npts=197` — one pass, `kink == npts`
so no kink was found, and 124.6 cm matches the 123.9 cm arc.

**This is not a Steiner-terminal coverage problem.** That distinction matters
because doc 31 spent eight rounds on terminal coverage: here the terminals reach
both ends, and the trajectory does not use them. Note that the shortfall alone
does not say *where* the loss happened — an endpoint never chosen and a
trajectory fitted then trimmed look identical from the outside. §2 settles it.

> **A measurement that proves nothing, recorded so it is not repeated.** The
> fit's first and last points lie 0.28 cm and 0.64 cm from a Steiner terminal.
> That is not evidence of snapping: the median along-track terminal gap here is
> 0.36 cm, so *any* point on this track is ~0.18 cm from a terminal on average.

**Which volumes the track crosses.** Matched on (y, z) against the per-face Bee
zips — the per-face zips carry per-face *uncorrected* drift x, so x cannot be
used to match — cluster 109 crosses exactly two:

| volume | points in tube | t range |
|---|---|---|
| `anode7-face0` | 468 | 0.0 → 75.4 cm |
| `anode7-face1` | 470 | 75.5 → 149.4 cm |

so **each end is the extreme of its own (apa, face) key**. Bee and WCT
coordinates coincide here (checked against cluster 86's WCT log line), so end A
at x = 343.1 cm lies *beyond* the shield plane at 339.91 cm — outside the
sensitive volume — while the fit's A end at x = 327.6 cm is 12.3 cm inside it.
Leaving the sensitive volume accounts for ~3.2 cm of the 18.6 cm, and end B has
no boundary anywhere near it.

### 1.1 It is not one bad cluster

Using doc 31's own long-straight selection unchanged (≥ 200 Steiner points, PCA
linearity λ0/Σλ > 0.95, p5–p95 first-axis extent > 50 cm), and measuring each
fit's shortfall against **that cluster's own terminal extent**:

| arm / event | long-straight clusters | of those, fitted | median min shortfall | median max | median total | both ends covered |
|---|---|---|---|---|---|---|
| 039252/2 evt 298595 | 32 | 10 | 3.1 cm | 16.2 cm | 22.1 cm | 1/10 |
| 039349/14 evt 19689 | 22 | 10 | 0.7 cm | 5.8 cm | 7.0 cm | 2/10 |
| **pooled** | 54 | **20** | **1.8 cm** | **11.4 cm** | **13.0 cm** | **3/20** |

The sign of a PCA axis is arbitrary, so "first end" and "second end" are not
comparable across clusters; every population number above is built from the
orientation-independent min / max / sum of the two shortfalls.

Two things in that table are worth stating separately:

- **A typical long straight track loses ~13 cm of trajectory across its two
  ends, and only 3 of 20 reach both of their outermost terminals.**
- **Most long straight clusters get no STM trajectory at all** — 22 of 32 in one
  event, 12 of 22 in the other — including the longest ones (terminal extents of
  666, 646, 598, 518 cm). Those are the clusters already tagged TGM or skipped as
  "fully contained"; see §5.

---

## 2. The mechanism, pinned

### 2.1 The two arms, and why a second arm was needed

`local/lib` and `toolkit/build/` were rebuilt at 20:49 — **after** the round-6
arm ran at 18:00 — and the working tree carries uncommitted peer edits to
`clus/src/TrackFitting.cxx` (+422 lines). The round-6 arm's stored outputs are
therefore not a valid inertness reference for today's binary. So both arms ran
on **one pinned snapshot** of `local/lib` plus the `wire-cell` / `wcsonnet`
executables (`libWireCellClus.so` md5 `1ac6f78fcd6bbe656de9ef55f2363aec`, and
`build/clus/libWireCellClus.so` had the same md5 at pin time):

- `d32base` — production config, `traj_cover_probe` off.
- `d32probe` — production config plus two **log-only, default-OFF** knobs: the
  jsonnet TLA `traj_cover_probe=true` and a one-key copy of the track-fitting
  JSON (`pdvd/stm/pdvd_track_fitting_probe.json`, adding
  `"traj_cover_probe": 1.0`). The canonical
  `cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` is untouched.

Three gates, all passed:

| gate | result |
|---|---|
| `d32base` vs the shipped round-6 arm: `persist_stm_fit` numbers and `TaggerCheckSTM` verdicts (timestamps stripped) | **identical** — the library rebuild did not move this event's STM output |
| **logger liveness** — any `pr67 examine_end_ps_vec` line from the `clus.TrackFitting` logger, on any cluster | **1213 lines**; the logger is alive, so an absence would have been informative |
| **inertness** — `d32probe` vs `d32base`: `persist_stm_fit` numbers and STM verdicts | **identical** — both knobs are inert, as designed |

The liveness gate matters: the track-fitting JSON is read at **runtime**, so a
byte-identical compiled jsonnet would not have caught a non-inert probe, and a
silent logger would have read exactly like "the amputation did not fire".

### 2.2 The verdict for cluster 109

Four `examine_end_ps_vec` calls touched cluster 109. The probe logs a line only
when an end moved more than 0.5 cm (or ≤ 2 points survive), so these four *are*
the amputations:

| call | npts | front moved | back moved |
|---|---|---|---|
| 1 | 170 → 136 | **15.45 cm** (340.04, 141.22, 195.49) → (327.12, 147.27, 201.44) | 78.40 cm (211.83, 267.04, 287.66) → (220.59, 196.78, 253.99) |
| 2 | 223 → 222 | 0.35 cm | 0.83 cm |
| 3 | 136 → 126 | 0.56 cm | **11.00 cm** (220.23, 197.02, 254.00) → (228.84, 190.22, 253.14) |
| 4 | 205 → 204 | 0.37 cm | 1.14 cm |

Reading those against the owner's two endpoints:

- **Call 1's incoming front, (340.04, 141.22, 195.49), is 3.26 cm from end A.**
  The path handed to the fitter *had* reached the A end. The call then removed
  15.45 cm of it; its outgoing front, (327.12, 147.27, 201.44), is 18.56 cm from
  end A and is exactly where the persisted trajectory begins.
- **Call 1's outgoing back, (220.59, 196.78, 253.99), is 0.86 cm from end B.**
  Its incoming back was 78.6 cm away at (211.83, 267.04, 287.66), off the track
  entirely — so this 78.40 cm "move" is a legitimate walk-back off a branch onto
  the real tip, not an amputation. After call 1 the path spanned essentially the
  whole track.
- **Call 3 then removed 11.00 cm from that B end**, ending 9.95 cm from end B.

So **C2 (end amputation inside the fit) is confirmed**: the trajectory that
reached both ends existed, and the fit threw it away.

**But C1 is only half exonerated, and the doc should not be read as clearing
it.** Call 1's incoming *front* was 3.26 cm from end A — that end was selected
correctly. Its incoming *back* was 78.6 cm away at (211.83, 267.04, 287.66),
off the track entirely: that is the rough path's other terminus, and it came
from the endpoint pair. **So the selected pair was one good end plus one point
on a branch**, and `examine_end_ps_vec`'s 78.40 cm walk-back is what rescued it.
Endpoint selection is exonerated at the A end and is *not* exonerated at the B
end; it simply is not what caused the missing coverage, because the amputation
undid its rescue 11.00 cm later.

C3 (`cluster_fc_check`'s round-2 boundary points never being written back,
`Clustering_Util.cxx:301-319` vs `:260-263`) is not this symptom's cause either,
but remains a real latent defect — see R5.

### 2.3 The amputation at population scale

832 amputation events were logged across 20 clusters in this event:

| moved an end by more than | events |
|---|---|
| 1 cm | 477 / 832 |
| 5 cm | 254 / 832 |
| 10 cm | 146 / 832 |
| 20 cm | 66 / 832 |
| 50 cm | 34 / 832 |
| largest single move | **489.1 cm** |

**These are per-fit-call events, not per-cluster, and they are concentrated.**
Five clusters (39, 40, 55, 86, 87) carry **735 of the 832** calls — the neutrino
stage re-fits some segments hundreds of times — so the table above describes how
often the trimmer acts, not how many clusters it damages. The per-cluster view
is the §1.1 shortfall table.

**Caveat, and it is not a small one:** as cluster 109's own call 1 shows, a large
move can be a path being correctly walked back onto the track after wandering
onto a branch. These counts are therefore an upper bound on genuine amputation,
not a measurement of it. Separating the two at population scale needs a per-call
comparison against the cluster's terminal extent and is not done here.

One correlation is clean, because the probe's own 0.5 cm threshold makes an
absence meaningful:

> **Cluster 108 is the only fitted long-straight cluster in this event for which
> `examine_end_ps_vec` never moved an end by more than 0.5 cm — and it is also
> the one whose trajectory reaches its terminal extent (0.5 cm and 0.1 cm short
> at the two ends).** Every other fitted cluster has both amputation lines and a
> shortfall.

### 2.4 What the pop loop actually tests

`TrackFitting.cxx:2305-2315`, the front loop (the back loop at `:2365-2375` is its
mirror), pops a point when **either** of two tests fails:

```cpp
auto test_wpid = m_dv->contained_by(ps_list.front());
if (test_wpid.face() != -1 && test_wpid.apa() != -1) {
    auto temp_p_raw = transform->backward(ps_list.front(), cluster_t0,
                                          test_wpid.face(), test_wpid.apa());
    if (m_grouping->is_good_point(temp_p_raw, test_wpid.apa(), test_wpid.face(),
                                  0.2*units::cm, 0, 0)) break;
}
temp_start = ps_list.front();
ps_list.pop_front();
```

then bisects back in 0.2 cm steps to re-insert the last good point (`:2318-2337`).

1. **`DetectorVolumes::contained_by()`** — the union of the 16 per-face
   sensitive boxes. This is the same volume §6 shows is PDVD's de facto
   fiducial volume, holes and all. End A at x = 343.1 cm is outside it
   (the sensitive edge is the shield plane at 339.91 cm), which is why cluster
   109's own first path point was popped before `is_good_point` was even
   reached — but that accounts for only ~3.2 of the 18.6 cm.
2. **`is_good_point(p_raw, apa, face, 0.2 cm, ch_range 0, allowed_bad 0)`**
   (`Facade_Grouping.cxx:537-554`). With `allowed_bad = 0`, **all three planes**
   must each have either a measured hit within a **0.2 cm circular radius** in
   the (drift, wire-pitch) plane (`has_closest_point`, `:687-701`, a 2-D k-d
   `exists_within(radius²)` query) or a dead channel within `ch_range = 0`.

The remaining ~12.8 cm of cluster 109's A-end amputation is inside the sensitive
volume and has imaged 3-D points, so it is test 2 that fails there.

### 2.5 Leading hypothesis for *why* PDVD amputates more — a 3 mm constant at 7.65 mm

`0.2 cm` is a physical radius in the wire-pitch direction, and it is the same
number on every detector. Against the pitches:

| detector / plane | pitch | 2 × 0.2 cm as a fraction of a pitch | consequence |
|---|---|---|---|
| SBND & uBooNE, U/V/W | 0.300 cm | **1.33** | the radius spans more than a full pitch — a projection *always* has a wire centre within 0.2 cm |
| PDVD U/V | 0.765 cm | 0.52 | **48 % of the pitch is outside the radius** |
| PDVD W | 0.510 cm | 0.78 | 22 % of the pitch is outside the radius |

So the "is there a hit near this projection" test is *guaranteed satisfiable* at
the pitch it was written for and is not at PDVD's, and `allowed_bad = 0` demands
all three planes pass simultaneously. This is the same failure shape doc 31
found twice (`terminal_charge_threshold` 4000 e tuned at 3 mm, `BlobSampler`
stepping in wire units) and doc 28 found once (crossing ambiguity), and it
predicts exactly the observed asymmetry: PDVD loses metres of trajectory where
SBND loses centimetres.

**This was a hypothesis when round 1 shipped. Round 2 measured it and it is
confirmed — see §§10-12.** The measurement named here as H1 has been done: the
per-plane `has_closest_point` match rate along a track, as a function of radius,
on both detectors. PDVD sits at the geometric ceiling on all three planes, SBND
does not, and each detector's pass rate stops rising at its own half-pitch
(§11.1).

One sentence of this section was wrong and is corrected rather than deleted,
because it is why the measurement waited a round: it said that the PR calib
dump's `proj` section carries only apa 0 in this event *and therefore* the test
could not be run offline. The first clause is true; the conclusion is not. The
**pctree tensor dump carries the complete ctpc** for all 16 (apa, face) keys,
and both detectors already had such dumps on disk. No arm was needed.

---

## 3. What this is *not*

- **Not a Steiner-terminal problem** (doc 31's subject): terminals cover both
  ends, largest gap 3.14 cm, 66 stranded outside the fit (§1).
- **Not the `iso_endpoint` branch** (doc `sbnd_xin` pr/24): that lives in
  `PatternAlgorithms::init_first_segment` and is gated to isochronous clusters
  (`m_iso_endpoint_max_xext` = 25 cm; cluster 109's drift extent is 121.7 cm).
  The STM trajectory does not go through `init_first_segment` at all.
- **Not a discontinuous fit**: the persisted trajectory is one piece, 197 points
  at a 0.60 cm median step, no gap above 2 cm.
- **Not doc 30's defect**: doc 30 (same event) was the `stm_fit` Bee layer's
  hard-coded `real_cluster_id`, a display-layer gap. This is reconstruction.
- **Not caused by the library rebuild**: `d32base` reproduces the shipped
  round-6 STM output exactly (§2.1).

---

## 4. The call chain, for reference

```
TaggerCheckSTM::visit                              TaggerCheckSTM.cxx:591
 └─ check_stm_conditions                           TaggerCheckSTM.cxx:3392
     ├─ Facade::cluster_fc_check                   Clustering_Util.cxx:75     <- endpoints AND the FV verdict
     │    └─ get_two_boundary_steiner_graph_idx    Facade_Cluster.cxx:3608    (flag_cosmic = TRUE in round 1)
     │         └─ get_two_boundary_wcps            Facade_Cluster.cxx:3685    (14 extremes per (apa,face); blob charge < 1500 e skipped)
     ├─ pick first_wcp / last_wcp                  TaggerCheckSTM.cxx:3492-3529
     └─ run_pass(start, end, is_forward)           TaggerCheckSTM.cxx:3543
          ├─ do_rough_path  (steiner dijkstra)     TaggerCheckSTM.cxx:1046
          ├─ TrackFitting::do_single_tracking      TrackFitting.cxx:9588      ROUND 1
          │    └─ organize_orig_path / organize_ps_path
          │         └─ examine_end_ps_vec          TrackFitting.cxx:2275      <-- THE AMPUTATION
          ├─ adjust_rough_path (only if a kink)    TaggerCheckSTM.cxx:1093
          ├─ do_single_tracking                    ROUND 2
          └─ begin_pass_record -> persist_stm_fit  TaggerCheckSTM.cxx:911     (the Bee stm_fit layer's source)
```

Other end-trimming sites in the same chain, found but **not** implicated here
(recorded so a later round does not re-derive them): `form_map`'s zero-charge
drop (`TrackFitting.cxx:4306`), `skip_trajectory_point`'s last-point rule
(`:6165`), and `organize_ps_path`'s final `end_point_limit == 0` call, which
re-appends the true last point only if it is ≥ 0.45 cm from the previous one
(`:2555`).

---

## 5. Why the two questions are one question

`Facade::cluster_fc_check` is **both** the endpoint source for the STM fit and
the fiducial containment test. And the amputation loop's first test is
`DetectorVolumes::contained_by()` — the same sensitive-volume union that §6
shows is what `FiducialUtils` actually receives on PDVD. So the FV appears twice
in this one failure: once deciding whether a cluster is even evaluated, and once
deciding where its trajectory is allowed to start.

The STM outcome census for this event (from the PR log) shows the first of those
biting hard:

| outcome | clusters |
|---|---|
| evaluated, no pass recorded (exited after the round-1 fit) | 41 |
| **"fully contained (Mid Point A)" — no STM fit attempted at all** | **35** |
| no `steiner_pc` | 3 |
| round-1 forward fit gave ≤ 3 points | 2 |
| single exit point with a 151° mid-track kink, arms 186.0 / 75.6 cm (Mid Point B) | 1 |

plus 32 clusters already tagged TGM and 6 with STM = 1. Thirty-five clusters
declared "fully contained" in a cosmic-ray event is the number to be suspicious
of, and it is the same suspicion `TaggerCheckSTM.cxx:70-99` already records for
SBND: on a 30-event sample, **96 of 147 clusters that STM skipped as "contained"
were exiters to `TaggerCheckFC`** — purely because the two used different
volumes.

And a trajectory that stops 12–19 cm short evaluates the stopping / through-going
test at an **interior** point, so the FV verdict is taken in the wrong place
however the FV is drawn.

---

## 6. The fiducial volume — PDVD vs SBND

### 6.1 The tagger FV already spans the whole detector, on both detectors

| | PDVD (`protodunevd/pr.jsonnet:1153-1163`) | SBND (`sbnd/clus.jsonnet:1948-1957`) |
|---|---|---|
| component | `BoxFiducial` `pdvd_pr_fv` | `BoxFiducial` `sbnd_pr_fv` |
| box | x ±339.91, y ±336.4, z [0.05, 299.25] cm | x ±201.05, y ±199.312, z [0.85, 500.15] cm |
| spans | both drift volumes + the 6 cm cathode slab | both TPCs + the 0.9 cm CPA |
| margins (`fv_tolerance`; negative = inset) | **x 30**, y 3, z_max 5, z_min 3 cm | x 2.5, y 3, z_max 5, z_min 3 cm |
| effective volume | \|x\| ≤ **309.91**, \|y\| ≤ 333.4, z ∈ [3.05, 294.25] | \|x\| ≤ 198.55, \|y\| ≤ 196.312, z ∈ [3.85, 495.15] |
| `stm_consistent_fv` | true (`pr.jsonnet:316`, `wct-pr-perevt.jsonnet:676`) | true |

Both are deliberately **one box across both drift regions**, so a cathode
crosser is not read as an "exiter" at x = 0; the default `fiducial=dv` cannot
serve there because the per-face union excludes the cathode slab. PDVD's x is
the shield plane where the sensitive volume starts; y/z are the active volume.
`stm_consistent_fv = true` on both means `TaggerCheckSTM`'s containment gate uses
the same box and margins as `tagger_check_tgm` and `tagger_check_fc`, so
"contained" has one meaning across the three verdicts.

**So the design goal "the entire detector forms the basis of the FV" is already
met for the tagger endpoint tests.** What is not met is everything in §6.2.

One number stands out. `tgm_fv_x_margin = 30` cm on PDVD against SBND's 2.5
(`wct-pr-perevt.jsonnet:1108`), chosen because the fitted dQ/dx rises ~50 % over
the last ~30 cm before either CRP — **60 of ~680 cm of drift inset away**.
Whether a margin chosen for a dQ/dx calibration artefact should be doing
fiducial-volume duty is a question this doc raises and does not answer.

### 6.2 Three FV notions on PDVD that are *not* whole-detector

**(a) What `FiducialUtils` actually receives is `fiducial=dv`.**
`pr.jsonnet:1098` and `:1100` pass `fiducial=dv` into `clustering_methods`,
which `common/clus.jsonnet:1522` forwards to `MakeFiducialUtils`. That is
`DetectorVolumes::contained()` (`aux/src/DetectorVolumes.cxx:111-113`) — the
union of the 16 per-face sensitive boxes. On PDVD that union has:

- a **full-detector \|y\| < 0.61 cm slab hole** (the CRP y-split runs the whole
  length and height of the detector),
- a **\|x\| < 3.0 cm cathode hole**,
- seams at \|y\| ≈ 168.5 cm and z ≈ 149.5 cm.

SBND's union has exactly one hole (\|x\| < 0.45 cm, the CPA). `pr.jsonnet:1087-1094`
already notes that this volume is ~3 cm more permissive than `pdvd_pr_fv +
margins` at every *outer* wall; it does not note the interior holes.

Everything hard-wired to `FiducialUtils` sees that volume — it is never routed
to a configured fiducial: `NeutrinoVertexFinder` (vertex scoring at `:1252`,
`:2156`, `:4443`, `:5168` — i.e. **which vertex wins**),
`NeutrinoPatternBase.cxx:3037` (rejects a segment whose first or last fit point
is outside), `TaggerCheckSTM` internals (`find_first_kink` `:1580`, `:1687`;
`detect_proton` `:1787`; `eval_stm` `:2848`; `check_dead_volume` `:3655`),
`TaggerCheckTGM`'s dead/SP walks (`:1005`, `:1009`, `:1025`, `:1029`), and every
step of `FiducialUtils::check_dead_volume` / `check_signal_processing`
(`FiducialUtils.cxx:136`, `:184`). **And §2.4's amputation loop.**

A worked consequence of the holes: `FiducialUtils` applies tolerance by probing
a *shifted point*, not by shrinking a box (`FiducialUtils.cxx:113-118`). With
`tol_xhi = −30 cm`, a point at x = +3.5 cm — 0.5 cm inside the cathode on the
top side — is probed at x = −26.5 cm, which lands in the \|x\| < 3 cm hole, so
it reads as **not contained** and the cathode is treated as a wall. On the single
`pdvd_pr_fv` box the same probe is contained and the cathode is correctly
interior. The holed union is *over*-strict here, not under-strict.

**(b) `dvm.overall` FV_x = ±341.55 cm** (`protodunevd/clus.jsonnet:100-101`) is
the W-plane centreline — **1.64 cm past the sensitive edge** at ±339.91 — while
FV_y and FV_z are inset 15 cm from the active volume (`:102-105`). Because the
per-face blocks disagree (bottom [−339.91, −3.0], top [3.0, 339.91] cm),
`select_scope_fv`'s `common_face_x` branch (`clustering_separate.cxx:96-127`)
falls back to `overall` for an all-anode scope, so an all-detector clustering
pass already uses one x-window spanning both drift volumes, the cathode, and
1.64 cm of PCB stack.

**(c) `clustering_examine_x_boundary` cannot span both drift sides.** It raises
`ValueError` on differing `FV_x` metadata (`clustering_examine_x_boundary.cxx:92-103`);
`allow_mixed_faces` waives only the same-*face* check, not the identical-`FV_x`
one. This is the tightest structural constraint on making the whole detector one
FV, because that stage *requires* per-drift-side `FV_x`.

### 6.3 The primitives available, and the prototype's version

| component | shape | status |
|---|---|---|
| `BoxFiducial` (`aux/src/BoxFiducial.cxx`) | one axis-aligned box | what `pdvd_pr_fv` / `sbnd_pr_fv` already are |
| `DetectorVolumes` (`aux/src/DetectorVolumes.cxx:111`) | exact per-face sensitive union, **holes included** | today's `FiducialUtils` basis on PDVD |
| `EnvFiducial` (`aux/src/EnvFiducial.cxx:21-24`) | one bbox over all faces' `sensitive()`, gaps and cathode **included** | **instantiated by no detector config**; would give PDVD x ±339.91 / y ±336.39 / z [0.813, 298.435] with zero new numbers |
| `CompositeFiducial` (`aux/src/CompositeFiducial.cxx`) | and / or / nand / nor of children | already used by SBND for the CPA structure (`sbnd/cathode_fiducial.jsonnet`, 26 boxes OR'd); PDVD has no equivalent |
| `PolyFiducial` (`aux/src/PolyFiducial.cxx:20`) | stack of polygonal slabs — the `ToyFiducial` port | **used by no detector config**; only `aux/test/doctest_fiducials.cxx` |

For contrast, the prototype's FV (`prototype_base/.../ToyFiducial`) was two
uBooNE-specific 6-corner polygons (x-y and x-z, both must hold), a set of
data-vs-MC space-charge-boundary constants, per-plane y-z boundary arrays, and a
**single scalar inset** `boundary_dis_cut` (production 3 cm) applied once in the
constructor — with *positive* tolerance meaning a *larger* volume, the opposite
of the toolkit's negative-inset convention. The space-charge boundary and the
MC-vs-data switch were **not ported**. uBooNE in this tree has no `IFiducial` and
no taggers at all; its only FV is the `FV_*` box metadata.

The **QLMatching** FV is a separate, per-drift-TPC construction with its own
cushions (`QLMatching.cxx:1413-1465`) and is documented in
`pdvd/docs/07_pdvd-tpc-geometry-fiducial.md`; it is not the volume this doc is
about, and nothing here changes it. PDVD has no `cathode_fiducial`.

---

## 7. Recommendations, ranked

Nothing below is implemented. Each names the default-OFF knob it would need and
the gate it would owe. Grading for R1–R3 is the §1.1 instrument (shortfall
against the terminal extent) **plus** what actually consumes the trajectory —
STM/TGM verdict counts and the segment/vertex census — because a longer
trajectory is not automatically a better one.

**R1 — make `is_good_point`'s search radius pitch-aware in the amputation walk.**
The single highest-value change: it is the confirmed mechanism, and §2.5 gives a
principled value (a radius of at least half a pitch makes the test satisfiable
by construction, as it already is at 3 mm). Knob: a `traj_end_good_radius`
track-fitting parameter, C++ default `0.2*units::cm` = today's behaviour, with
the option of expressing it per plane in pitch units. Evidence bar: mechanism
**confirmed**; the pitch explanation **hypothesis**. Owes: H1 first (below), then
a byte-identical gate with the knob at its default, plus a knob-on arm on the
PDVD manifest and an SBND arm — at SBND's pitch a half-pitch radius is *smaller*
than 0.2 cm, so this is **not** PDVD-only in effect.

> **Superseded by §16 R1.** H1 is done and the pitch explanation is confirmed
> (§§10-12), so the evidence bar above is met. Two things also changed: a
> per-plane radius cannot be delivered by a track-fitting parameter at all,
> because `is_good_point` takes one scalar for three planes (§14), and writing
> the knob as a **floor** — `max(radius, frac·pitch)` — makes it inert at SBND
> and uBooNE by construction, which is the useful half of the sentence here.
> It does **not** make it PDVD-only: PDHD's 0.467 cm pitch puts half a pitch
> above 0.2 cm, so PDHD changes too (§16 R1).

**R2 — relax `allowed_bad` at the trajectory ends only.** `allowed_bad = 0`
demands all three planes simultaneously; two-of-three is the convention
elsewhere in `is_good_point`'s callers (`minimal_views = 2` in
`FiducialUtils::inside_dead_region`). Cheaper and blunter than R1 and does not
need the pitch measurement. Knob: `traj_end_allowed_bad`, C++ default 0.

> **Superseded by §16 R2, which recommends against it.** Now that H1 is
> measured, R1 and R2 can be graded against each other on the same points: R2
> recovers less on PDVD (0.518 against R1's 0.715) *and* moves SBND from 0.624
> to 0.936, so it is both weaker and less containable.

**R3 — give the amputation walk the same volume the taggers use.** Test 1 of the
pop loop is `DetectorVolumes::contained_by()`, so a trajectory point in the
\|y\| < 0.61 cm slab or the cathode slab is popped for being "outside the
detector" when it is inside it. Routing this one call to the configured fiducial
(`pdvd_pr_fv`) would make it consistent with `stm_consistent_fv`. Note this is
strictly a *widening*, so it can only lengthen trajectories.

**R4 — route `MakeFiducialUtils` to a whole-detector fiducial.** The direct
answer to the owner's design goal, and the one with the widest blast radius: it
moves vertex scores, segment rejection, both taggers' dead/SP walks, and the
amputation walk at once. `EnvFiducial` over all 16 faces needs no new numbers;
`pdvd_pr_fv` is already written. Knob: a `fiducial` argument on `clustering_methods`
defaulting to `dv` so the compiled config is byte-identical when off. Owes a
compiled-config proof and a full A/B, and it is **not** byte-identical when on —
the SBND precedent at `TaggerCheckSTM.cxx:70-99` (96 of 147) is the scale to
expect. Do this after R1–R3, whose scope is small enough to grade cleanly.

**R5 — fix `cluster_fc_check`'s round-2 boundary points.** `Clustering_Util.cxx:301-319`
computes `bp1_r2/bp2_r2` and never writes them into `result.boundary_first/second`
(only `:260-263` does, from round 1). Not this symptom's cause (§2.2), but a real
latent defect: when round 1 finds no exit, the fit is anchored on a pair the
round-2 test never validated. Cheap, self-contained, and worth a knob of its own.

**R6 — reconsider `tgm_fv_x_margin = 30 cm`.** It removes 60 of ~680 cm of drift
from the fiducial volume to compensate a dQ/dx calibration artefact near the
CRPs. If the artefact is the real target, a dQ/dx correction is the right
instrument and the FV margin should come back toward SBND's 2.5 cm. Owner call;
not a code change.

### Measurements owed before R1 ships

- **H1 — the pitch hypothesis. DONE in round 2 (§§10-12): confirmed.** It
  predicted a step at radius ≈ half a pitch on PDVD and none on SBND; what the
  data show is a step at half a pitch on *each* detector, at 0.15 cm for SBND
  and 0.38 cm for PDVD (§11.1). It needed neither the `proj` section nor an
  in-process probe — the pctree dump carries the whole ctpc.
- **H2 — separate genuine amputation from branch walk-back** at population scale
  (§2.3's caveat), by comparing each call's incoming and outgoing endpoints
  against the cluster's terminal extent rather than against each other.

---

## 8. Gates, and what is not established

- **No toolkit change and no production config change this round**, so **no A/B
  gate is owed** and none is claimed. Both probe knobs are default-OFF and
  log-only, and §2.1's inertness gate demonstrates it on this event rather than
  asserting it.
- Both new arms ran on a **pinned** binary snapshot; `d32base` reproduces the
  shipped round-6 arm's STM output identically despite the 20:49 rebuild.
- Every number here is regenerated by `scripts/stm_trajectory_coverage.py` or
  quoted from a log line under a path named in §0.
- **Two events only** (039252/2 and 039349/14), and the probe ran on one of them.
  The 20-cluster population in §1.1 is small.
- §2.3's counts are an **upper bound** on amputation, not a measurement of it.
- §2.5's pitch explanation is a **hypothesis**; the mechanism (§2.2, §2.4) is
  confirmed, the reason PDVD suffers more is not.
- The `L_fit` column in the script's output can exceed the p5–p95 extent used
  for *selection*, because that window trims 10 % of the terminal span by
  construction; all shortfalls are measured against the full terminal extent.
- §6's FV map is read from source and config; the per-face sensitive-box
  geometry was recomputed from the wires file rather than read from a comment,
  and the method reproduces SBND's documented \|y\| ≤ 199.965 / z ∈ [0, 501.0]
  exactly.

---

## 9. What this pins down for the next round

1. The end-coverage defect has **one confirmed site**: `examine_end_ps_vec`
   (`TrackFitting.cxx:2275`), reached from `do_single_tracking`. Round 2 does not
   need to re-search the chain.
2. The endpoint selection (`get_two_boundary_wcps`) is **exonerated at one end
   only**: on cluster 109 it delivered end A to within 3.3 cm but put the other
   terminus 78.6 cm away on a branch, which the end walk then rescued. It is not
   this symptom's cause, and it is not cleared — a later round must still check
   it.
3. `DetectorVolumes`'s holed union is load-bearing in *two* places at once — the
   FV that `FiducialUtils` serves and the amputation walk's containment test —
   which makes R3/R4 one decision, not two.
4. The 0.2 cm / 3 mm arithmetic is the third instance of the campaign's
   pitch-blind-constant pattern (after doc 31's 4000 e terminal threshold and
   `BlobSampler`'s wire-unit stepping, and doc 28's crossing ambiguity). It is
   worth asking, once, which other constants in the fitting chain were tuned at
   3 mm.

---

# Round 2 — the pitch hypothesis, measured

**Status of round 2.** §2.5 left the pitch explanation as an explicitly
unproven hypothesis and said it could not be tested from what is on disk. The
second half of that was wrong, and the correction is what made this round
possible: the calib dump's `proj` section does carry only apa 0 in this event,
but the **pctree tensor dump carries the entire ctpc** — every
`ctpc_a<A>f<F>p<U|V|W>` dataset, for all 16 (apa, face) keys — and both
detectors already have such dumps on disk. **Still no code change, no config
change, and no new arm this round**, so again no A/B gate is owed and none is
claimed.

**Verdict: confirmed.** The 0.2 cm radius is pitch-blind, PDVD sits at the
geometric ceiling that blindness imposes, SBND does not, and the region the
trajectory loses is a region that has charge in it.

## 10. How the test can be replicated exactly, offline

`Grouping::has_closest_point` (`Facade_Grouping.cxx:687-701`) is a 2-D
existence query in the (drift, pitch) plane against the ctpc — and the ctpc is
an exact lattice, by construction:

```
PointTreeBuilding.cxx:313   y = pitch_mags[apa][face][plane] * (wind + 0.5) + proj_centers[...]
PointTreeBuilding.cxx:312   x = time2drift(face, time_offset, drift_speed, slice->start())
```

Every occupied cell therefore sits at a wire centre in the pitch coordinate and
at a slice boundary in drift. A trajectory point lands at an arbitrary phase in
that grid, so "is there a hit within 0.2 cm of this projection" is only partly a
question about charge — it is also a question about **where between two wires
the point fell**. If half the pitch exceeds the radius, there is a band of
phases that cannot match no matter how much charge is present.

`docs/nf_sp_img_clus/scripts/goodpoint_pitch_census.py` replicates the query
from a pctree dump plus the trajectories in `mabc-pr.zip` /
`calib-pr-evt*.json`. Three things are proven rather than assumed:

| gate | why it matters | PDVD | SBND |
|---|---|---|---|
| projection map refitted from the dump's own `2dp{p}_y` arrays against (y, z) | `y_proj = cos(a)·z − sin(a)·y` is linear, so a non-zero residual would mean the replica is not the code | max &#124;residual&#124; **8.19e-12 mm** | **7.28e-12 mm** |
| pitch and drift step read off the ctpc arrays, not from a geometry file | the whole argument is about these two numbers | pitch **7.650 / 7.650 / 5.100 mm**, step **2.9615 mm** | pitch **3.000 mm**, step **3.126 mm** |
| corrected→raw x offset read off the 3d point cloud (`x` − `x_t0cor` at the nearest point, mode over the trajectory) | the pctree dump is written at the **clustering** stage, so its cluster idents do **not** match the PR/Bee ids and `cluster_scalar/cluster_t0` cannot be indexed by them | purity 1.00 on cluster 109 | — |

**Interior points only.** The first and last 3 points of every polyline are
dropped. Those are exactly the points the pop loop *accepted*, so including
them would measure survivorship instead of geometry. Every other point on a
fitted trajectory was never subjected to the test, which makes it an unbiased
sample of "a smooth trajectory through real charge".

**One bias, stated before the numbers.** A fitted trajectory is fitted *to*
charge, so its lattice phase is pulled toward occupied wires. Every rate below
is therefore an **upper** bound on what an arbitrary point on the track would
score. A measured rate at or slightly under the uniform-phase ceiling is the
expected outcome; a rate far below it would mean charge absence, not geometry.

## 11. The result: PDVD is limited by the lattice, SBND by the charge

Distance from an interior trajectory point to the nearest ctpc cell, against the
uniform-phase ceiling for that plane's own (pitch × drift-step) cell — the
fraction of the cell within 0.2 cm of a lattice point when *every* neighbouring
cell is occupied, i.e. the best any event can do.

**PDVD** — 22 trajectories, 6700 interior points, `work/039252_2_d31r6e2e`
(039252/2 evt 298595):

| plane | median | p75 | p90 | pass @ 0.2 cm | lattice ceiling |
|---|---|---|---|---|---|
| U | 2.334 mm | 3.768 | 42.83 | **0.436** | 0.470 |
| V | 2.322 mm | 3.546 | 22.10 | **0.428** | 0.470 |
| W | 1.591 mm | 2.469 | 30.89 | **0.624** | 0.705 |

**SBND** — 56 trajectories, 2830 interior points, 16 events of
`sbnd_xin/work-mcp2k-d97fvpr2`:

| plane | median | p75 | p90 | pass @ 0.2 cm | lattice ceiling |
|---|---|---|---|---|---|
| U | 1.247 mm | 1.527 | 1.782 | **0.963** | 0.988 |
| V | 1.332 mm | 1.662 | 2.394 | **0.865** | 0.988 |
| W | 1.448 mm | 2.173 | 4.348 | **0.729** | 0.988 |

Read the last two columns together. **PDVD is at its ceiling on all three
planes** — 0.436 against 0.470, 0.428 against 0.470, 0.624 against 0.705. There
is no charge to add that would improve those numbers; the grid is the binding
constraint. SBND's U plane is also at its ceiling (0.963 against 0.988) but the
ceiling is 0.988, and its V and W fall well below theirs (0.865, 0.729), which
is charge absence — dead channels and the sparser charge of a neutrino event.

Same code, same constant, different binding constraint. The median PDVD U/V
distance, 2.33 mm, sits *on* the 2.0 mm threshold; SBND's, 1.25 mm, sits at
five eighths of it.

### 11.1 The knee is at half a pitch, on each detector

`is_good_point(radius, ch_range = 0, allowed_bad = 0)` — all three planes must
pass — as a function of radius:

| radius (cm) | PDVD | SBND | |
|---|---|---|---|
| 0.10 | 0.027 | 0.043 | |
| 0.15 | 0.076 | 0.321 | ← SBND half-pitch = 0.150 cm |
| **0.20 (production)** | **0.175** | **0.624** | |
| 0.25 | 0.304 | 0.694 | ← PDVD W half-pitch = 0.255 cm |
| 0.30 | 0.449 | 0.729 | |
| 0.40 | 0.715 | 0.819 | ← PDVD U/V half-pitch = 0.3825 cm |
| 0.50 | 0.738 | 0.880 | |
| 0.60 | 0.751 | 0.912 | |

PDVD climbs +0.54 between 0.15 and 0.40 cm and then flattens — +0.036 over the
whole of 0.40 → 0.60. SBND climbs +0.58 between 0.10 and 0.20 cm and then
crawls. **Each detector's rise ends where its own half-pitch is crossed.** That
is the discriminating observation of this round: a charge-density explanation
predicts no knee, and predicts any knee it did produce at the *same* radius on
both detectors, because 0.2 cm is the same 0.2 cm. A lattice explanation
predicts a knee at pitch/2, which is a different radius on each detector. The
knee is at pitch/2 on each.

The plateau heights say what geometry cannot explain: PDVD flattens at 0.75,
SBND at 0.91. That residue is genuine charge absence, and it is what
`allowed_bad` and `ch_range` exist for — not what the radius should be asked to
fix.

## 12. The region the trajectory loses has charge in it

The rates above are about the whole trajectory. This is about the part that was
amputated. For every trajectory end, walk outward along the local direction in
0.6 cm steps and keep the **contiguous run of steps that have a 3d point within
1.5 cm** — i.e. only where charge demonstrably exists. Because 3d points exist
only inside the sensitive volume, that filter also keeps the walk inside the
detector, which is what separates this measurement from the pop loop's *other*
test, `DetectorVolumes::contained_by()`.

The 1.5 cm threshold is not itself the argument, and it is wide enough (about
two U/V pitches) to admit a point that is near *some* charge without being near
a wire centre in any plane. What calibrates it is §12.1: on cluster 109 the
charge-bearing extension it selects, 19.2 and 11.4 cm, reproduces the
independently measured amputations of 15.45 and 11.00 cm from the probe arm.
The filter is validated by that agreement, not by the choice of 1.5 cm.

**PDVD** — 44 ends; charge continues a median **4.8 cm** past the fit tip
(p90 18.1 cm, max 19.8 cm):

| radius (cm) | U | V | W | all three |
|---|---|---|---|---|
| **0.20** | 0.315 | 0.345 | 0.507 | **0.064** |
| 0.30 | 0.507 | 0.578 | 0.786 | 0.266 |
| 0.40 | 0.694 | 0.795 | 0.837 | **0.522** |
| 0.60 | 0.801 | 0.884 | 0.904 | 0.677 |

**SBND** — 109 ends; charge continues a median 3.0 cm:

| radius (cm) | U | V | W | all three |
|---|---|---|---|---|
| **0.20** | 0.702 | 0.572 | 0.697 | **0.371** |
| 0.40 | 0.823 | 0.712 | 0.811 | 0.557 |

In the region where the fit gave up and the charge did not, the production test
passes **6.4 %** of the time on PDVD and **37.1 %** on SBND — a 5.8× gap, same
code, same radius, charge present on both sides. Widening PDVD's radius past its
own half-pitch takes it to 52.2 %, an **8.2× recovery**, while SBND gains only
1.5×. This is §1's stranded-terminal argument made quantitative: 39 + 27
terminals lay beyond the amputated tips, terminals mean blobs mean occupied
lattice cells, and a test that reports "nothing within 0.2 cm" there is failing
on grid phase, not on charge.

### 12.1 Cluster 109, and cluster 108 as the negative control

| cluster / end | charge continues past the tip | all three @ 0.2 cm | @ 0.4 cm | amputated (§2.2) |
|---|---|---|---|---|
| 109, A / front | **19.2 cm** | 0.094 | 0.750 | 15.45 cm |
| 109, B / back | **11.4 cm** | 0.000 | 0.474 | 11.00 cm |
| **108, both ends** | **1.8 cm** | *n = 3* | *n = 3* | (reaches its terminals) |

Cluster 109's B end is the cleanest single number in this document: charge
continues 11.4 cm past where the fit stops, the fit was cut back by 11.00 cm,
and **not one** of the 19 sampled points in that stretch passes the production
test — while nearly half of them pass at half a pitch.

Cluster 108 is the control. §2.3 identified it as the only fitted long-straight
cluster in the event whose trajectory reaches its terminal extent and the only
one with no probe lines at all. It has **1.8 cm** of charge-bearing extension at
either end — its fit stops where the charge stops. The same measurement, applied
to the cluster that did not lose its ends, finds essentially nothing there to
lose. That is the negative control this round owed, and it passes.

Its pass-rate cells are deliberately left blank: 1.8 cm is **three sampled
steps** per end, and a rate over three points is not a measurement. The
load-bearing number for the control is the extension length, and only that.
(For the record those three steps sit a median 7.6 mm from the nearest U cell,
i.e. they are not near charge in the pitch coordinate at all — consistent with
"the track ends here", not with "the test failed here".)

## 13. What round 2 does *not* establish

- **Scale.** One PDVD event (22 trajectories) against 16 SBND events (56
  trajectories). The PDVD side is one event because one event is what has a
  pctree dump on disk; §2.3's population evidence is separate and broader.
- **It does not separate the pop loop's two tests.** The charge-within-1.5-cm
  filter keeps §12's walk inside the sensitive volume, so §12 is a clean
  statement about `is_good_point` alone — but the *production* loop pops on
  `contained_by` **or** `is_good_point`, and §2.2's estimate that ~3.9 of end
  A's 18.6 cm lies outside the sensitive volume still stands and is not
  displaced by anything here.
- **Dead channels are not modelled.** `get_closest_dead_chs` can rescue a plane
  that has no live hit, so the true production pass rate is somewhat higher than
  every number above wherever dead channels are near. This does not touch the
  cross-detector comparison, which applies the identical live-only model to
  both.
- **Correlation along a track is observed but not explained.** Failure runs:
  PDVD median 3 points, p90 10, max 257; SBND median 1, p90 8, max 94. 12.1 % of
  PDVD's failure runs are ≥ 10 points (≳ 6 cm) against 9.2 % of SBND's. Rates
  alone would predict shorter runs than are observed, so failures are correlated
  — the lattice phase advances smoothly along a smooth trajectory, and with
  `allowed_bad = 0` three phases must be simultaneously in-band. The script
  reports pass rate against phase-advance rate; the trend is monotone on both
  detectors but in *opposite* directions, so no mechanism is claimed from it.
  The long runs are the operative fact; their cause is not pinned.

## 14. The same constant, in twenty other places

The user asked for the places this has been met. There are two answers: the call
sites of *this* test, and the campaign's wider pattern of pitch-blind constants
(§15).

Grouped by the `(radius, ch_range, allowed_bad)` triple actually passed —
because it is the triple, not the file, that determines whether a site is
exposed:

| triple | sites | where |
|---|---|---|
| **0.2 cm, 0, 0** | **20** | `NeutrinoStructureExaminer.cxx` :211, :373, :1107, :1678, :2014, :3252, :3621, :3632, :3658, :3676, :3692, :3810, :3821 · **`TrackFitting.cxx` :2312, :2331, :2371, :2390** · `NeutrinoOtherSegments.cxx` :1382, :1568 · `NeutrinoShowerClustering.cxx` :2944 |
| 0.3 cm, 0, 0 | 2 | `NeutrinoStructureExaminer.cxx` :907, :3847 |
| 0.6 cm, 1, 1 (the defaults) | 12 | `connect_graph_ctpc.cxx` :118, :218, :274, :699, :735, :770 · `connect_graph_relaxed.cxx` :435, :549 · `connect_graph_relaxed_strict.cxx` :2000, :2135 · `clustering_protect_overclustering.cxx` :397 · `TaggerCheckTGM.cxx` :1234 |
| 0.6 cm, 1, 0 | 4 | `connect_graph_relaxed.cxx` :444, :558 · `connect_graph_relaxed_strict.cxx` :2004, :2139 |
| configurable radius | 2 | `NeutrinoSteinerGapGraph.cxx:76` (`m_sgp_point_radius`, `TaggerCheckNeutrino.h:538`, default 0.2 cm, exposed as `sgp_point_radius` and referenced at `pr.jsonnet:975`) · `NeutrinoGraphAudit.cxx:125` (`good_radius`) |

Against the pitches, as 2 × radius in units of one pitch — the ratio that
crosses 1.0 is the tell:

| radius | SBND / uBooNE (0.300 cm) | **PDHD (0.467 / 0.480 cm)** | PDVD W (0.510 cm) | PDVD U/V (0.765 cm) |
|---|---|---|---|---|
| **0.2 cm** | **1.33** | **0.86 / 0.83** | **0.78** | **0.52** |
| 0.3 cm | 2.00 | 1.29 / 1.25 | 1.18 | 0.78 |
| 0.6 cm | 4.00 | 2.57 / 2.50 | 2.35 | 1.57 |

(Pitches measured from the wires files themselves —
`protodunehd-wires-larsoft-v1.json.bz2`,
`protodunevd-wires-larsoft-v7-uvwfit.json.bz2`,
`sbnd-wires-geometry-v0206.json.bz2` — as the perpendicular spacing of adjacent
wire centre lines, not quoted from a config.)

**PDHD is exposed too, and this document did not look for it.** At 0.86 of a
pitch the strict test is not satisfiable-by-construction there either, only less
severely than at PDVD's 0.52. Nothing here measures PDHD; the prediction is that
the same amputation happens at PDHD at roughly half the PDVD amplitude, and a
PDHD arm is owed before any fix ships (§16 R1).

**The loose family is safe everywhere**; 0.6 cm spans more than a full pitch on
every plane of both detectors, which is why the graph builders have never shown
this symptom. Only the strict 0.2 cm family — and, on U/V, the 0.3 cm pair —
crosses below 1.0 on PDVD while sitting above 1.0 at the pitch it was written
for. Twenty sites are exposed; this round measured one of them.

Two asymmetries worth recording, because they are themselves evidence:

- `get_closest_dead_chs` takes `ch_range` in **channel** units
  (`Facade_Grouping.cxx:703-717`), so the dead half of `is_good_point` is
  already pitch-relative. Only the live half is blind. A single function
  measures one thing in wires and the other in centimetres.
- `is_good_point` takes **one scalar radius for three planes**, and PDVD has two
  different pitches behind a single (apa, face). No value a caller could pass is
  correct for all three planes at once — which is why §16 puts the fix inside
  the loop rather than at the call sites.

## 15. The campaign's pitch-blind constants — five now, four before

Every one of these first presented as something else, and each was found by
writing the constant as a fraction of the pitch:

| # | constant | tuned at | on PDVD | doc |
|---|---|---|---|---|
| 1 | `terminal_charge_threshold` 4000 e | 0.300 cm pitch | needs 500 e | doc 31 / doc 28 |
| 2 | `BlobSampler::Stepped` steps in **wire** units | — | 2.55× fewer sample points for a physically similar blob | doc 31 round 8 |
| 3 | (U, V) crossing ambiguity | 21.0 % at SBND | **83.6 %** | doc 28 |
| 4 | `channels()` indexed as a wire lookup | — | silent on PDVD/PDHD, three sites | doc 31 rounds 3, 5 |
| 5 | **`is_good_point(..., 0.2 cm, 0, 0)`** | 1.33 pitches | **0.52 pitch** (PDHD 0.86) | **this doc, confirmed §11** |

The general rule this round adds: a distance constant that is compared against
detector charge is only meaningful **relative to the sampling that produced the
charge**. The ctpc is a lattice; a radius smaller than half its cell can never
be satisfied for a whole band of phases, whatever the physics. Where a constant
of this kind is introduced or reviewed, write it as a fraction of the pitch and
of the drift step, on every detector that binds the component, and put the table
in the doc.

## 16. Solutions

Ranked. None is implemented — this round changed no code and no config — and
each names the default-OFF knob it needs and the gate it would owe. R1 replaces
round 1's R1 and R2; R3–R6 are unchanged from §7 and are listed for ordering
only.

### R1 — give `is_good_point` a per-plane pitch floor (recommended)

The fix cannot go at the call sites. Twenty of them pass a scalar, and the
scalar must serve three planes with two different pitches, so no value is right.
It has to resolve inside the per-plane loop — and `fastgeom_t` **already caches
what is needed**: `std::array<double,3> pitch` alongside `angle`
(`Facade_Grouping.h:116`, filled at `Facade_Grouping.cxx:755`), so the lookup is
free.

```cpp
// Facade_Grouping.h:296 -- one new defaulted argument
bool is_good_point(const geo_point_t& point, int apa, int face,
                   double radius = 0.6*units::cm, int ch_range = 1,
                   int allowed_bad = 1, double pitch_frac = 0.0) const;

// Facade_Grouping.cxx:537
const auto& fg = fastgeom(apa, face);
for (int pind = 0; pind < 3; ++pind) {
    // pitch_frac = 0 => r == radius => every existing caller is unchanged.
    const double r = pitch_frac > 0.0
                   ? std::max(radius, pitch_frac * fg.pitch[pind])
                   : radius;
    if (has_closest_point(point, r, apa, face, pind)) { ... }
```

Then one call site opts in — `examine_end_ps_vec`'s four calls
(`TrackFitting.cxx` :2312, :2331, :2371, :2390) — reading the fraction from the
TrackFitting params struct in the same idiom as `traj_cover_probe`
(`TrackFitting.cxx:63`, `TrackFitting.h:147`), C++ default 0.

**Recommended value 0.5, and the `max()` is load-bearing.** Half a pitch is
0.3825 cm on PDVD U/V and 0.255 cm on W — the plateau in §11.1. On SBND and
uBooNE half a pitch is **0.150 cm, which is smaller than the existing 0.2 cm**,
so `max(radius, 0.5·pitch)` returns 0.2 cm unchanged there. Expressed as a
floor rather than a replacement, the knob **cannot** make a fine-pitch detector
stricter — which is the deliberate answer to the standing warning that a
pitch-expressed constant usually does exactly that.

**It is not, however, PDVD-only.** Half a pitch at PDHD is **0.233–0.240 cm,
which is larger than 0.2 cm**, so a PDHD job with the knob on *would* change:

| detector | pitch (cm) | 0.5 × pitch | knob-on effect |
|---|---|---|---|
| uBooNE, SBND | 0.300 | 0.150 | **inert** (floor below 0.2) |
| **PDHD** | 0.467 / 0.480 | **0.233 / 0.240** | **changes** — mild loosening |
| PDVD W | 0.510 | 0.255 | changes |
| PDVD U/V | 0.765 | 0.3825 | changes — the case measured here |

So the three detectors that bind this component split into "inert by
construction" (uBooNE, SBND) and "must be gated" (PDHD, PDVD). An earlier draft
of this section claimed PDVD-only and was wrong; the arithmetic above is the
correction.

Expected effect, from §11.1 and §12: interior pass rate 0.175 → 0.715; in the
amputated region 0.064 → 0.522.

**Gate owed.** Knob off ⇒ byte-identical on **every** detector that binds
`is_good_point` — PDVD, PDHD, SBND, uBooNE — because the signature change
touches a function on all of their hot paths. Knob on ⇒ a PDVD arm showing the
trajectories reaching their terminal extents; an SBND knob-on arm proving the
`max()` really is inert there rather than only arguably so; and, because §14's
table puts PDHD at 0.86 of a pitch, **a PDHD knob-on arm as a first-class part
of the grading, not an afterthought** — PDHD is the detector this document
never measured and the one where the effect is real but unquantified.
`./build/clus/wcdoctest-clus`.

### R2 — relax `allowed_bad` at the ends (measured, and rejected in favour of R1)

Round 1 listed this as a peer of R1. It is not; §11's data grades them directly.
At the current 0.2 cm radius, requiring only 2 of 3 planes gives:

| | all three | ≥ 2 of 3 | ≥ 1 of 3 |
|---|---|---|---|
| PDVD @ 0.2 cm | 0.175 | **0.518** | 0.795 |
| SBND @ 0.2 cm | 0.624 | **0.936** | 0.995 |

R2 buys PDVD less than R1 does (0.518 against 0.715) **and** moves SBND a long
way (0.624 → 0.936), so it is neither as effective nor as containable. It also
weakens the test's purpose: two planes agreeing is a real reduction in
confidence that a point is on charge, whereas R1 leaves the confidence
requirement intact and only stops asking for a precision the wire pitch cannot
deliver. Recommend **not** pursuing R2 on its own. If R1 alone proves
insufficient, R1+R2 together reach 0.829 at 0.4 cm — but that is a second round
with its own evidence, not a hedge to ship now.

### R3–R6 (unchanged from §7, re-ordered behind R1)

- **R3** — give the amputation walk the tagger's fiducial volume rather than the
  holed sensitive union. Independent of R1: it addresses the pop loop's *other*
  test, which §13 records as still unseparated.
- **R4** — route `MakeFiducialUtils` to a whole-detector fiducial (§6.2). The
  broadest change and the one that most directly answers the owner's second
  question; `EnvFiducial` needs no new constants (§6.3).
- **R5** — fix `cluster_fc_check`'s round-2 write-back (`Clustering_Util.cxx`
  :301-319 vs :260-263). Latent, not this defect.
- **R6** — reconsider `tgm_fv_x_margin = 30 cm` (§6.1). An owner call, not a
  code fix.

### The measurement that is now discharged, and the one that is not

**H1 is done** — this round is H1, and it confirms. **H2 is not**: separating
genuine amputation from the branch walk-back at population scale still needs the
probe arm's per-call front/back deltas cross-referenced against §12's
charge-bearing extension per end. It is the natural next measurement and it
needs no code change either.

# doc pr/96 — imaged charge with no fitted trajectory: two owner-reported vertex prongs

Status: **DIAGNOSIS ONLY.** Two owner events resolved to **two different
mechanisms**, one of which is **fixed by an existing, deliberately-unflipped
knob** and one of which is a **measured residual**. No fix implemented, no
production default changed. The only code edit is a log-only, env-gated probe
(`WCT_PR96_REMSEG_DEBUG`), hash-gated byte-identical when unset.

Owner report (Bee set `e1357e60` = `bee/vtxpr-review-20260818/`, 6 mcp2k events,
run 18255 subrun 1):

| slot | event | owner point (x,y,z) cm | owner note | owner "cluster" |
|---|---|---|---|---|
| 0 | 18255-**279955** | (-95.9, 136.2, 281.3) | "missing a vertex track???" | 8 |
| 2 | 18255-**70084** | (-195.0, 18.4, 123.0) | "missing a track, vertex PR weird" | 11 |

> The owner's "0-70084" is slot indexing; `vtxpr-review.prid-map.txt` puts both
> events at run 18255 subrun 1. The quoted cluster ids are **`img-global`** ids;
> they are `clustering-global` **16** and **20** at HEAD. Every number below is
> in the T0-corrected frame (`clustering-global` / `track_fit-global`), never
> `img-global` — that layer is the one raw layer and carries a per-cluster
> drift-x offset (doc pr/13, pr/67 §1).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 0. provenance (M1).  Tree clean in clus/ and cfg/ before the probe edit.
git -C ../../../toolkit log --oneline -1        # fd6a116d
wcbuild > /home/xqian/tmp/pr96/build.log 2>&1; echo rc=$?    # rc=0, 0 objects
../../../toolkit/build/clus/wcdoctest-clus     # 211/211, 2215 assertions

# 1. the diagnostic arm -- pr/67's traj_cover_probe, log-only
SBND_TRAJ_COVER_PROBE=1 PR_JOBS=2 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr96-dbg1-mcp2k data 279955 70084

# 2. add the remove_segment CALLER probe (clus/src/PRGraph.cxx), rebuild, and
#    prove the edit is byte-neutral with the env UNSET
wcbuild > /home/xqian/tmp/pr96/build2.log 2>&1; echo rc=$?
SBND_TRAJ_COVER_PROBE=1 PR_JOBS=2 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr96-dbg2-mcp2k data 279955 70084
python3 scripts/pr85_hash_gate.py work-pr96-dbg1-mcp2k work-pr96-dbg2-mcp2k; echo rc=$?
#   -> PASS all 4 archives byte-identical, rc=0

# 3. the caller trace
SBND_TRAJ_COVER_PROBE=1 WCT_PR96_REMSEG_DEBUG=1 PR_JOBS=2 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr96-dbg3-mcp2k data 279955 70084
grep -a -A9 "PR96REMSEG nfits=45 " work-pr96-dbg3-mcp2k/pr_evt279955/stdout.log | c++filt

# 4. MEASURE pr/30's P1 port-fidelity gap on the two events (NOT a flip; see sec 8)
SBND_FIT_EXCLUSION=true SBND_TRAJ_COVER_PROBE=1 PR_JOBS=2 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr96-fx1-mcp2k data 279955 70084

# 5. the metric + the population census (new script, see sec 3)
python3 scripts/pr96_uncover_census.py work-pr96-dbg1-mcp2k
python3 scripts/pr96_uncover_census.py work-cbr3-census-on            # 6-event calibration
python3 scripts/pr96_uncover_census.py work-nuecc48-prod0819 work-ncpi0-prod0819 --quiet
python3 scripts/pr96_uncover_census.py work-mcp1k-prod0819 --quiet
python3 scripts/pr96_uncover_census.py work-mcp2k-prod0819 --quiet    # 686/2000 at time of writing
```

Arms: `work-pr96-{dbg1,dbg2,dbg3,fx1}-mcp2k`. Q/L root **`work-mcp2k-ql0819`**
(doc pr/95: Q/L *and* PR both at `fd6a116d`, the tree's first single-epoch
product), deliberately not `work-cbr3-census-on`, whose Q/L predates the pr/94
flip and whose PR stopped at `tagger_check_neutrino`.

---

## 1. Symptom, measured at HEAD

`work-pr96-dbg1-mcp2k`, neutrino cluster = the cluster holding the `q=15000`
main vertex:

| | 279955 (cid 16) | 70084 (cid 20) |
|---|---|---|
| charge points / fit points | 2752 / 371 | 1422 / 208 |
| charge points > 3 cm from **any** fit point | 232 (8.4 %) | 182 (12.8 %) |
| **of the cluster's charge** | **13.9 %** | **19.1 %** |
| largest connected uncovered group | 232 pts | 179 pts |
| its PCA extent | **20.2 cm** | **9.6 cm** |
| its transverse rms | 0.43 cm | 0.45 cm |
| max distance to any fit point | 5.5 cm | 6.5 cm |
| min distance to the main vertex | 4.9 cm | 6.5 cm |
| nearest fitted segment / angle to it | 16004 / **5.6°** | 20036 / 32.4° |
| segments in the cluster | **one** (222.1 cm) | three (97.9 / 2.5 / 22.9 cm) |
| PF tree (Bee `mc` layer is the **reco** PF tree) | μ⁻ 515 MeV only | e⁻ 8 → μ⁻ 247 MeV; e⁻ 178 MeV |

**These are identical to the numbers on the owner's own (stale) Bee zip**, so the
symptom survived pr/93 r1-4 and the pr/94 four-knob flip. 70084's row also
reproduces **bit-for-bit in bare production** (`work-mcp2k-prod0819`, no probe,
no env) — 179 pts, 19.1 %, 9.62 cm — which is the independent confirmation that
`traj_cover_probe` is log-only.

In both events the uncovered charge is a **straight, dense prong at the neutrino
vertex**, carrying 14–19 % of the cluster's charge, with no trajectory on it. It
is not an imaging degeneracy: measured against the nearest fitted point, the
prong is offset ~3 cm in y and 2–3 cm in drift and <1 cm in z — ≈8–10 wires in
U/V plus ≈25 ticks — i.e. resolvable in every plane. Its charge per point is
13.3k / 13.5k against cluster means of 8.1k / 8.9k, so it is *denser* than the
average of the cluster it belongs to.

---

## 2. Root cause, evt 70084 — the candidate is found, fitted, and dropped under
## a point-count floor

The `pr54 isolated-residual drop` line (unconditional DEBUG, present in every
arm run at `-L debug`) names the seat outright:

```
pr54 isolated-residual drop: cluster 20 n_points=12 length=8.81 cm dir_mag=8.87 cm
     v1=(-195.2,17.2,119.6) v2=(-194.2,15.1,128.1) cm        [twice, once per round]
```

That candidate's endpoints sit **0.1 cm** from the measured uncovered group
(the census joins them automatically and prints
`pr54drop(cid20,npts12,8.8cm,d0.1)`). With the probe on, its whole life is
visible and it passes every quality gate:

```
pr67 fos step8: cluster=20 group=4 npts=12 len=8.87 cm nnf=10 max_dis=(4.09,2.78,4.57) cm
    A=(-195.16,17.12,120.22) B=(-194.22,14.96,128.77) bbox=[-195.16,-194.22]x[14.01,19.20]x[120.22,128.77] -> KEEP
pr67 fos step9: cluster=20 group=4 npts=12 len=8.87 cm nnf=10 ... -> SELECTED
```

`nnf=10` of 12 points are non-faked, i.e. ≥2 planes see them clear of every
existing segment — this is *not* a shadow artifact. It is then routed, fitted,
and destroyed by the point floor in `other_seg_keep_isolated_ok`
(`clus/src/NeutrinoOtherSegments.cxx:32`, call site `:834`):

* SBND production has `other_seg_keep_isolated = true`
  (`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet:2146`) with
  `_min_points = null → 25` and `_min_length = null → 3.0 cm`
  (`clus/inc/WireCellClus/TaggerCheckNeutrino.h:261-262`).
* The candidate has **12** points against the floor of **25**. `min_points`
  counts **Steiner terminals**, not image points — an implicit length floor far
  above the advertised 3 cm (pr/67 §9.2 made the same observation).
* On the else-branch (`:864-877`) both vertices are `remove_vertex`ed and the
  segment was **never** `add_segment`ed. This is a hard loss: no trajectory, no
  Bee points, nothing.

**This is the same residual pr/67 §9.2 already measured** on three of its four
owner events (10 / 10 / 14 terminals against the same floor), and whose designed
remedy **S1 `other_seg_keep_isolated_min_nnf` was specified and never built**
(pr/67 §10.3, §11.10).

### 2.1 …but the causal lever is one level up, and it is measured

`SBND_FIT_EXCLUSION=true` (`work-pr96-fx1-mcp2k`) **fixes 70084 outright**:

| | production | `fit_exclusion=true` |
|---|---|---|
| uncovered charge points | 182 (12.8 %) | **46 (3.2 %)** |
| uncovered **charge** | **19.1 %** | **3.4 %** |
| fit points in the cluster | 208 | **243** |
| track-like uncovered groups | 1 | **0** |
| segments | 20014 (97.9), 20034 (2.5), 20036 (22.9) | 20023 (99.1), **20034 (13.0)**, **20022 (9.4)**, 20035 (15.7), 20020 (5.4) |

The two new segments **20022** (9.4 cm) and **20034** (13.0 cm) meet at
(-196.2, 16.8, 121.8) — the centre of the uncovered group is (-195.1, 16.8,
123.7). The prong now has a trajectory. `oseg_iso_drop` is still 2, so the
pr/54 floor still fires; what changed is that with exclusion the *neighbouring*
segments' fits no longer absorb the prong's 2-D cells, and the prong is
recovered through the ordinary segment path instead of the isolated-residual
path.

---

## 3. Root cause, evt 279955 — the candidate is found and fitted, then deleted
## as a duplicate, because its fit lands on the muon

Here **no admission gate is at fault.** The probe shows the component KEPT and
SELECTED, twice, with a bbox that contains the owner's point:

```
pr67 fos: cluster=16 existing_segments=2 steiner_N=1522 tagged=1298 terminals=298
pr67 fos step8: cluster=16 group=0 npts=31 len=18.88 cm nnf=25 max_dis=(1.75,3.77,3.25) cm
    A=(-93.29,133.95,286.27) B=(-100.17,132.83,268.72)
    bbox=[-101.10,-92.66]x[132.05,136.38]x[268.72,286.57] -> KEEP
pr67 fos step9: cluster=16 group=0 ... -> SELECTED
pr67 find_other_segments: cluster=16 round 1/2 segments 2 -> 3 (added 1)
pr67 fos step8: cluster=16 group=0 npts=25 len=17.37 cm nnf=25 ... -> KEEP     (round 2)
pr67 find_other_segments: cluster=16 round 2/2 segments 3 -> 3 (added 0)
```

`oseg_reject=2` in this event is the pr/30 P4 accept test
(`NeutrinoOtherSegments.cxx:673-696`), which only decides whether an
**already-added** segment is queued for `break_segments`; it discards nothing.
The segment is added — and then removed twice. The `WCT_PR96_REMSEG_DEBUG`
backtrace names both callers:

```
PR96REMSEG nfits=45 front=(-93.60,134.79,286.15) back=(-99.78,131.01,266.00)
  bt[1] PatternAlgorithms::examine_vertices_4(...)
  bt[2] PatternAlgorithms::examine_vertices(...)
  bt[3] PatternAlgorithms::find_proto_vertex(...)

PR96REMSEG nfits=48 front=(-93.60,134.79,286.15) back=(-100.02,129.87,264.78)
  bt[1] PatternAlgorithms::main_vertex_graph_audit(...)
  bt[2] TaggerCheckNeutrino::visit(...)
```

and the mvga line one entry later gives the reason verbatim:

```
mvga: op1 dup-merge cluster=16 removed seg len=27.96cm sumdQ=7.21e+05
      overlap=0.90@14.0mm vs survivor len=221.58cm sumdQ=1.28e+07 reconnects=0
mvga: fired cluster=16 op1=1 op2=0 op3=0 (refit done)
```

The branch is deleted as a **duplicate trajectory** of the muon by doc pr/83
round 4's op1 projective dup-collapse
(`clus/src/NeutrinoGraphAudit.cxx:340-465`). Every gate behaves as designed:

| gate | value | branch | verdict |
|---|---|---|---|
| `path_overlap_fraction(shorter, longer, tol)` — fraction of the shorter segment's **fitted** points with any of the longer's within `tol` (`clus/src/PRSegmentFunctions.cxx:1653`, plain 3-D) | tol = `mvga_dup_tol` **1.4 cm** | **0.90** | — |
| accept threshold: `min(la,lb) ≥ 10 cm ? mvga_op1_dup_frac : mvga_dup_frac` | **0.7** (both 27.96 and 221.58 ≥ 10 cm) | 0.90 ≥ 0.7 | pass |
| near-parallel guard `mvga_dup_angle`, chord-vs-chord folded to [0,90], decline if **greater** | **20°** | chord angle **13.1°** | pass (i.e. does not save it) |
| keep the higher integrated charge | 1.28e7 vs 7.21e5 | branch loses | deleted |

### 3.1 The branch's fit collapsed onto the muon — the deletion is a symptom

The op1 verdict is *correct about the trajectory it was given*. Sampling the
straight chord between the branch's own two fitted endpoints against the muon's
371 fitted points:

```
 t     sample point            d(muon fit)   d(nearest charge)
0.00 ( -93.6, 134.8, 286.1)      3.03            0.29
0.19 ( -94.8, 134.1, 282.3)      3.01            0.80
0.48 ( -96.5, 133.0, 276.6)      2.17            1.21
0.76 ( -98.3, 131.9, 270.8)      1.78            0.96
0.95 ( -99.5, 131.2, 267.0)      0.56            0.38
straight-chord overlap within 1.4 cm of the muon: 0.11
```

**0.11 for the straight chord versus 0.90 measured for the actual fit.** The
branch's endpoints are where its charge is; its *interior* was fitted onto the
muon. And the straight chord is a good answer: it sits a mean of **2.01 cm**
(p90 2.91 cm) from the uncovered charge and spans it along [-0.9, 19.2] cm. The
object PR needs is simple and PR found its ends — the fit did not follow it.

### 3.2 `fit_exclusion` does **not** fix 279955

| | production | `fit_exclusion=true` |
|---|---|---|
| uncovered charge points | 232 (8.4 %) | 241 (8.8 %) |
| uncovered charge | 13.9 % | **15.2 %** |
| fit points | 371 | 367 |
| segments | one, 222.1 cm | one, 219.5 cm |
| op1 dup-merge | fires (27.96 cm, overlap 0.90) | **still fires** (27.34 cm, overlap 0.89) |

At 5.6° and 2–3 cm separation from a muon carrying **11× its charge**, excluding
the neighbour's claimed cells is not enough to keep the fit off it. **279955 is
a residual with a named mechanism and a measured negative**, in the same class
as pr/67's declined 58717 (residual 92 % transverse).

---

## 4. Why it hid

- **No stage anywhere measures trajectory-vs-charge coverage** (pr/67 §5). The
  three shipped instruments — `pr23_fitcover_census.py`, `gapjump_probe.py`,
  `pr94r3_gap_metric.py` — all measure the **inverse** (a fit point straying
  into a void). pr/55's Family B measures the right direction but against
  `shower_track` (the *associated* points), so charge never associated to any
  segment is structurally invisible. Nothing read `clustering-global → nearest
  track_fit-global`, whole-event, in 3-D. §5 fills that gap.
- **The counters were visible and unread.** `PR30AUDIT oseg_iso_drop=2` and the
  `pr54 isolated-residual drop` line were in 70084's log all along; nothing
  joined them to a coordinate.
- **Both losses look identical on a Bee scan** — nothing in `track_fit` — while
  being opposite failures: an admission floor (70084) versus a fit that
  converged onto its neighbour and was correctly garbage-collected (279955).
- **mcp2k had never been censused for this family.** Every gate behind the
  family's production-ON knobs ran on nueCC48-48 / NCπ⁰-19 / mcp1k
  (pr/67 §11.5-6, pr/54 §13, pr/90:469 "mcp2k excluded", pr/94:1755).
- **The near-parallel guard cannot see this case by construction.** Its comment
  says "a genuine small-opening V whose short prong hugs the long one within tol
  must NOT merge" — but the implemented discriminator is the chord angle, and a
  genuine 13° V and a corridor duplicate are both *near-parallel*. Nothing in
  op1 asks whether the loser explains charge the survivor does not.

---

## 5. The metric, and calibrating its discriminator

`scripts/pr96_uncover_census.py` (new; forked from `scripts/pr94r3_gap_metric.py`
for the zip loader only — everything past it is new). Per event, per cluster:
`clustering-global → nearest track_fit-global` in 3-D, group the >3 cm points by
2 cm single-link, and score each group on points, charge fraction, PCA extent,
transverse rms, max distance, distance to the main vertex, and the nearest
segment's local angle. It also joins the `pr54 isolated-residual drop`
endpoints, so a 70084-class event is confirmed **with no rerun at all**.

The shower confound is the whole difficulty, and the owner's own 6-event Bee set
is a ready-made labelled set. Calibrated on `work-cbr3-census-on` (the arm that
scan came from):

| slot | event | uncovered | q | groups | label | flagged at final cuts |
|---|---|---|---|---|---|---|
| 0 | 279955 | 8.4 % | 13.9 % | 1 | **positive** (owner) | ✅ yes |
| 2 | 70084 | 12.8 % | 19.1 % | 2 | **positive** (owner) | ✅ yes |
| 1 | 405707 | 0.0 % | 0.0 % | 0 | negative | ✅ no |
| 4 | 316025 | 0.6 % | 0.3 % | 1 | negative | ✅ no |
| 3 | 283713 | **30.2 %** | 26.1 % | **43** | **shower**, must not fire | ✅ no |
| 5 | 395148 | 5.2 % | 5.0 % | 8 | pr/94 §9.10 family | ✅ no |

**6/6.** The default cuts (`npts ≥ 40`, `len ≥ 5 cm`, `rms ≤ 1.5`,
`qfrac ≥ 0.03`) scored only 5/6: 283713's two largest groups (227.8 cm at
rms 1.12, 123.7 cm at rms 1.31, both ~4° from segment 19064) are **rinds running
alongside a trajectory**, and no straightness or angle cut separates them from
279955's 20.2 cm at 5.6°. What does separate them is **distance to the main
vertex**: 4.9 and 6.5 cm for the owner's prongs against 70.3 and 190.9 cm for
the rinds. `--dvtx 15` (plus `rms ≤ 0.8`) gives 6/6 and states the census's
scope honestly: *a missing prong at the neutrino vertex*, which is what the
owner reported. Broader uncovered-charge classes are still printed, just not
flagged.

---

## 6. Population — first measurement of this family on mcp2k

`prod0819` bare-production arms (doc pr/95 Phase 3, toolkit `fd6a116d`
end-to-end). No probe, no env: the census reads only `mabc-pr.zip` and the
already-unconditional log lines.

| sample | events | uncovered groups | track-like groups | **events flagged** | rate | with a `pr54` hit |
|---|---|---|---|---|---|---|
| nueCC48 + NCπ⁰ | 67 | 516 | 9 | **8** | **11.9 %** | 3 |
| mcp1k | 1000 | 1163 | 28 | **24** | **2.4 %** | 3 |
| mcp2k (686/2000 done) | 686 | 1422 | 32 | **27** | **3.9 %** | 4 |

10 of the 69 track-like groups carry a `pr54 isolated-residual drop` at the
charge (**70084 class**, and `fit_exclusion` is measured to fix that class on
70084). The other 59 do not, and 279955 is one of them — so the residual class
is the larger one, which is the single most important number here.

**Worse exhibits than either owner event exist.** Sorted by fraction of the
cluster's charge left uncovered:

| event | cid | q uncovered | extent | group centre (x,y,z) | angle | `pr54`? |
|---|---|---|---|---|---|---|
| mcp2k **91653** | 13 | **52.7 %** | 17.7 cm | (-73.1, 187.9, 259.7) | 3.6° | — |
| mcp2k **91697** | 2 | **35.8 %** | 8.8 cm | (-175.0, 43.8, 493.1) | 52.7° | **yes** |
| mcp2k **51546** | 16 | 24.8 % | 32.1 cm | (-159.0, 173.8, 435.4) | 0.7° | — |
| mcp2k **52121** | 11 | 19.4 % | 31.3 cm | (-28.8, -157.2, 72.0) | 7.2° | — |
| mcp2k 70084 | 20 | 19.1 % | 9.6 cm | (-195.1, 16.7, 123.7) | 32.4° | **yes** |
| nueCC48 **469665** | 15 | 17.7 % + 12.0 % | 17.2 + 23.4 cm | (25.6, 68.2, 295.2) | 11.7° | one of two |
| nueCC48 **116962** | 21 | 14.8 % | 23.0 cm | — | 30.9° | — |

Several flagged events are already named in earlier docs — 116962 (pr/94 r3's
one primary mover), 395148 (pr/94 §9.10), 122660 (pr/86 §14.5's named loss),
67394 (pr/86 v6's adverse case), 90055 and 71372 (pr/54 §13) — which is
consistent with one under-measured failure family rather than seven unrelated
bugs. Two events could not be scored (`73422`, `99860`: no `track_fit` layer);
that is doc pr/55's `require_pr_graph` empty-layer path, out of scope here.

279955 itself is not in the table only because `work-mcp2k-prod0819` had reached
686/2000 when this was written; it is measured in `work-pr96-dbg1-mcp2k` off the
same Q/L root and the same binary.

---

## 7. Fix design (specified, not implemented)

The framing that makes any of this defensible against the pr/83
duplicate-corridor family: **every admission and duplicate gate in this path is
a cheap proxy for "this candidate is not a parallel re-trace of a segment I
already have"** — `direct_length < 0.78·length` (curvature), `min_points ≥ 25`
(bulk), `path_overlap_fraction ≥ 0.7` at 1.4 cm plus a 20° chord angle. A
straight 10–20 cm vertex prong at a small opening angle fails all of them while
being exactly the object they were never meant to reject. What none of them
measures is **whether the candidate explains charge nothing else explains** —
and that is the property they are all standing in for, because a duplicate
covers charge that is already covered and therefore fails the direct test by
construction.

**F1 — `other_seg_keep_isolated_min_nnf`** (build pr/67's unbuilt S1). Admit an
isolated residual below the 25-terminal floor when `number_not_faked ≥ N`;
C++ default 0 = legacy. `nnf` is already computed at the call site and thrown
away: `other_seg_keep_isolated_ok` (`NeutrinoOtherSegments.cxx:32`) sees only
`component_points` and `track_length`. 70084's candidate has **nnf = 10 of 12**,
so any N in pr/67 §10.3's stated scan range (3–8, centre 4) admits it, while the
noise components pr/54 §13 sized the floor against (3, 4, 4, 10 points, and here
`nnf = 0` at both `nnf0_short` drops) stay out. pr/67 §10.3 already fixed the
predicate shape and the doctest file
(`clus/test/doctest_other_seg_keep_isolated.cxx`).
*Instrumentation gap to close with it:* the `pr54 isolated-residual drop` line
does not print `nnf`, so an ON census must join the `step8 KEEP` line by bbox —
or add that one field, log-only.

**F2 — uncovered-charge admission**, as an additional disjunct **inside** the
existing predicate, never a new pass. Admit when the component's imaged charge
farther than `D` cm from every existing fitted trajectory exceeds a floor in
points and/or charge fraction. This must be written as a threshold on an
existing decision: pr/67 §10.8 declined a "coverage-driven re-offer pass"
precisely because it "would be a new stage in the chain rather than a threshold
on an existing one". Take dQ/dx from `segment_median_dQ_dx`, never from the Bee
layer's `q` (per-blob and geometry-dependent). F2 subsumes F1 and is the general
statement; F1 is the cheap version that is already specified and already has a
target with measured `nnf`.

**F3 — an uncovered-charge veto at the mvga op1 seat** — designed and
**recommended against on this evidence.** Refusing to delete a loser whose own
charge is not covered by the survivor would keep 279955's branch, but that
branch's trajectory lies within 1.4 cm of the muon for 90 % of its length: the
event would gain a duplicate trajectory and *still* not cover the charge. It
would trade the owner's complaint for pr/83's. Do not build F3 without first
fixing the fit that produced the trajectory.

**279955's real lever is the fit, and it is out of reach this round.** Its ends
are right, its route is right, its interior is wrong. Neither available lever
reaches it: `fit_exclusion` is measured not to (§3.2), and F3 would keep a
trajectory that does not cover the charge. Recorded as the open item it is, in
the same slot pr/67 gave 58717.

---

## 8. What is **not** proposed, and why

- **Do not flip `fit_exclusion`.** It fixes 70084 (§2.1), and it is the clearest
  port-fidelity gap in the tree — pr/30 P1: the prototype passes
  `flag_exclusion` at **28 of 30** live sites, the toolkit at none. But
  **pr/30 §12.8 declined it** ("do NOT turn on at this operating point"): on 48
  nueCC events it changed 47/48, moved `numu_score` on 47/48, lost 7 nue
  candidates against 3 gained, and moved the selected vertex by up to 97 cm.
  It is also **still blocked on a specific unresolved check** — pr/30 §3.1's
  transverse-coordinate units inside `update_association`:

  ```cpp
  // prototype PR3DCluster_multi_track_fitting.h:1013
  double y = (wire - offset_u) * pitch_u;
  // toolkit  TrackFitting.cxx:2568
  double raw_y = (coord.wire - offset_u) / slope_yu;   // slope_yu = -sin(angle_u)/pitch_u
  ```

  Until this round that path was dead code, which is exactly where a unit error
  survives. **Owner decision, presented not taken** (CLAUDE.md §5.4): either
  (a) settle §3.1 first, then re-measure P1 at *today's* operating point — three
  PR rounds and the pr/94 flip have landed since §12.4 — or (b) pursue the
  narrower variant below, which inherits the same §3.1 question over a much
  smaller blast radius.
- **The narrower variant, for the record**: pass `flag_exclusion = true` at only
  the **three** `find_other_segments` `do_multi_tracking` sites
  (`NeutrinoOtherSegments.cxx:654`, `:843`, `:880`) instead of all 27 — i.e.
  exclude only when fitting a *newly proposed* segment against segments that
  already exist. That is the exact operation 70084 needs, and it cannot perturb
  the first-segment or post-break fits that pr/30 §12.4 measured. Untested;
  offered as the cheapest next measurement, not as a knob to flip.
- **No fitter-level transverse pull and no coverage-driven re-offer stage** —
  declined twice (pr/67 §10.8, §11.12). §6's population is the new argument for
  re-opening that with the owner, not a licence to route around it.
- **No change to `iso_snap_min_dir_mag`** (production 4.0): pr/67 §11.7 measured
  its response as non-monotonic with intrinsic ν-vertex movement, and neither
  event here reaches the isochronous snap path.
- **No lowering of `other_seg_keep_isolated_min_points`** from 25. pr/54 §13
  sized that floor against 3–10-point noise components and 70084's candidate has
  12 — squarely in the danger zone. F1's `nnf` discriminator is the way in, not
  the floor.
- **The pr/94-4b candidate-0 slot bug is a clean negative** for the `track_fit`
  layer at HEAD (every SBND reader threads an explicit fitter). Not a display
  artifact. Do not re-chase it.

---

## 9. Gates

This round owes no behaviour gate — nothing behavioural changed. What was
gated:

| claim | gate | result |
|---|---|---|
| the `WCT_PR96_REMSEG_DEBUG` edit is byte-neutral | `pr85_hash_gate.py work-pr96-dbg1-mcp2k work-pr96-dbg2-mcp2k` (pre-edit binary vs post-edit binary, env unset) | **PASS 4/4 archives byte-identical**, rc=0 |
| `traj_cover_probe` is log-only | 70084's census row in bare `work-mcp2k-prod0819` vs `work-pr96-dbg1-mcp2k` | identical (179 pts, 19.1 %, 9.62 cm) |
| the `SBND_FIT_EXCLUSION` hook reaches the component | `PR30AUDIT … knobs[fit_exclusion=true …]` in `work-pr96-fx1-mcp2k` | present |
| build provenance | `wcbuild` rc=0; `wcdoctest-clus` | 211/211, 2215 assertions |

Gates that F1/F2 will owe next round: knob-off byte-identical via
`pr85_hash_gate.py` on nueCC48 / NCπ⁰ / mcp1k plus a compiled-config proof, and
a knob-on census in which every mover is attributable to a sentinel (the pr/65
"0 unclaimed" bar). `scripts/pr96_uncover_census.py` is the before/after metric:
the target is the flagged-event count per sample in §6 and the per-event charge
fraction.

## 10. Open items

1. **279955** — residual. Mechanism named and both available levers measured or
   reasoned negative. Needs a fit that follows its own charge at 2–3 cm from an
   11×-brighter neighbour.
2. **pr/30 §3.1 unit question** — blocking any honest re-read of `fit_exclusion`.
   Owner decision (§8).
3. **mcp2k census is partial** (686/2000). Re-run §6's last row when pr/95
   Phase 3 finishes; the two `MISSING_LAYER` events want a separate look.
4. **New exhibits for a scan** — 91653 (52.7 %), 91697 (35.8 %), 51546, 52121,
   469665 (two prongs), 116962. Not packaged for Bee; no upload authorized.

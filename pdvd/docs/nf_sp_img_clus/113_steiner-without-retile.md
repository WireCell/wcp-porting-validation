# doc pdvd/113 — the Steiner graph without the retile: it removes doc 112's ghost terminals and fixes h1, but misses the frozen bars and costs tags; not recommended as is

**Owner, 2026-09-16** (after doc 112): *"In the current PDVD processing chain, we have implemented the retiling. This
is inheriting from MicroBooNE configuration when we have a lot more dead channels. Now, I wonder if we remove the
retiling from the PDVD and PDHD chain, whether they would fix the problem we are seeing while not introducing more
issues? Can you investigate this and give a try? We can then later deal with the CreateSteinerGraph in the future
session. For this one, we can see if we remove the retiling step, will it lead to better performance in general? Note,
we still want to have the stepped_charge sampling method."*

**Scope.**
- **C++:** one default-OFF knob, `ImproveCluster_2` `retile_mode` (toolkit `d2777286`, pushed). **Knob OFF is
  byte-identical** (sec 2). **Knob ON is NOT bit-identical.**
- **Config:** `retile_mode=null` in `pdhd/pr.jsonnet` and `protodunevd/pr.jsonnet`, plus the TLA in both
  `wct-pr-perevt.jsonnet`. Production is unchanged.
- **Not touched:** `CreateSteinerGraph` and its terminal admission, which were deferred by the owner.
- **Sampling:** `charge_stepped` is kept in every arm.

## Status: the answer

**Removing the retile fixes what doc 112 found, but that is not the whole off-image problem, and it costs tags.**
All three levels of removal **FAIL** the rule frozen before any knob-on arm ran (`figs/113_verdict.txt`). **Nothing is
flipped.**

| | PDHD | PDVD |
|---|---|---|
| terminals with one blank plane (doc 112's ghosts), production → no retile | 81,995 → **57** | 119,476 → **9,666** |
| Steiner seed length > 1 cm off the image (S1, bar −25 %) | 9.87 → 8.51 % (**−13.8 %**) | 7.82 → 6.06 % (**−22.5 %**) |
| … of which 1–5 cm off (near) | 5.87 → 4.36 % (**−25.8 %**) | 5.51 → 3.79 % (**−31.1 %**) |
| … of which > 5 cm off (far bridges) | 4.00 → 4.15 % (+3.7 %) | 2.31 → 2.27 % (−1.8 %) |
| fitted STM rows > 1 cm off (P1, bar −20 %) | 7.74 → 7.00 % (**−9.6 %**) | 5.10 → 4.71 % (**−7.6 %**) |
| owner's spot, seed max offset (P5, bar 1.2 cm) | h1 1.97 → **0.68 cm** | v3 1.64 → **1.24 cm** |
| owner's spot, fit max offset | h1 1.77 → **0.46 cm** | v3 0.91 → **1.19 cm** |
| `is_stm` purity / efficiency | 0.959 / 0.636 → 0.958 / 0.614 | **0.982** / 0.709 → **0.956** / 0.701 |
| `michel_found` purity / efficiency | 0.880 / **0.702** → 0.886 / **0.596** | **0.940** / 0.725 → **0.861** / 0.708 |
| PR wall, median per event (vs concurrent base) | **−27 %** | **−18 %** |

What this says, in order:

1. **Most ghosts are the retile's painting, and removing it removes them.**
   - With the painting off and everything else in the retile kept (`no_paint`), one-blank terminals fall 99.4 %
     (PDHD) and 90.7 % (PDVD).
   - On PDHD that answers doc 112's open question: the empty cells are painted cells, not wire-range sentinels
     (81,995 → 464 → 368 → 57 across the ladder).
   - On PDVD about 9,700–11,100 remain in every reduced level, `none` included. They are neither painted cells nor
     footprint sentinels, and this round does not identify them (sec 11).
   - h1 is fixed in every level: seed 0.68 cm, fit 0.46 cm (`figs/113_spot_h1.png`). h2's fit goes 2.55 → 0.76 cm and
     v2's fit 1.27 → 0.42 cm.
2. **It is only part of the off-image seed.**
   - The near deviations (1–5 cm), where the ghosts live, drop 24–31 % in every level on both detectors.
   - The far deviations (> 5 cm) do not move. They are the graph's own bridges across gaps, as doc 111 round 2 found,
     and are now about half (PDHD) and a third (PDVD) of what remains.
   - The fitted track moves less than the seed: rows > 1 cm drop 4–10 %.
3. **v3 misses its spot bar by 0.04 cm, and its fit gets worse.** The seed drops 1.64 → 1.24 cm but the fit rises
   0.91 → 1.19 cm. Over 3 cm of track the image splits into two bands; the seed and fit ride the upper one, while the
   ridge metric follows the charge-weighted middle (sec 7).
4. **Tags change, and on the current truth records the change is a cost.**
   - The two detectors fail differently.
   - **PDHD loses efficiency:** Michel efficiency −0.106 in `none` (21 true Michels lost, 10 gained); purity is flat
     or better.
   - **PDVD loses purity:** `is_stm` −0.025, Michel −0.079, from 9 and 20 new false positives. Efficiency is nearly
     flat. **None of those false-positive labels has been reviewed by the owner** (sec 5).
   - Doc 102 r2 saw the same pattern: a trajectory change the taggers were not tuned on.
5. **Gap jumping did not break by any measure the rule set.**
   - Truncated fits: 1.2–1.6 % (the accepted `charge_stepped` flips had 3.4–4.5 %).
   - Far deviations: unchanged.
   - Tags whose production Steiner graph touched a dead channel are not lost more often than the others.
   - The graph now makes the jumps itself: it holds 2.2× (PDHD) and 2.4× (PDVD) as many ctpc/MST bridge edges.
6. **It is faster:** median PR wall −18 to −33 %; the Steiner-stage cloud shrinks to 43–55 % of its size.

**Recommendation: do not remove the retile on this evidence.** The fix it brings is the doc-112 mechanism only; the
arm-wide seed, the fit and the tags all miss their bars. The next step is in sec 10.

---

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
T=/home/xqian/toolkit-dev/toolkit; P=/home/xqian/tmp/d113/libpin_d113; TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"

# build (toolkit d2777286), doctests, pin (clus 6d97984ebbba)
(cd $T && ./wcb build --notests -p && ./wcb install --notests -p && ./wcb build --targets=wcdoctest-clus -p \
   && ./build/clus/wcdoctest-clus)                                                 # 430 / 430
mkdir -p $P && cp -a /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so $P/

# sec 2 -- knob-off gates
bash $S/d102_compile_pr.sh pdhd d113knoboff; bash $S/d102_compile_pr.sh pdvd d113knoboff      # a870511c9b22 / 211a49a48229
bash $S/d102_compile_pr.sh pdhd d113knobnone -S "retile_mode='none'"                        # differs only in retile_mode
bash $S/d102_compile_pr.sh pdvd d113knobnone -S "retile_mode='none'"
ARM=d113hbase DET=pdhd SRC=d108hflip JOBS=6 PIN=$P LOGD=/home/xqian/tmp/d113/arm_d113hbase TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d113vbase DET=pdvd SRC=d103vflip JOBS=6 PIN=$P LOGD=/home/xqian/tmp/d113/arm_d113vbase TRACE_ENV="$TR" bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d111hst --b d113hbase > $F/113_gate_base_pdhd.txt   # 61 / 61
python3 $S/d111_identity_gate.py --det pdvd --a d111vst --b d113vbase > $F/113_gate_base_pdvd.txt   # 120 / 120
(cd $S && TAG=d113snew PIN=$P JOBS=2 bash d101_sbnd_arm.sh)
python3 $S/d111_sbnd_gate.py --old d111ssnew --new d113snew --ignore-branch Trun.toolkit_git > $F/113_gate_sbnd.txt
(cd $IMG/qlport/scripts && LD_LIBRARY_PATH=$P ./sweep_5384.sh d113ub 3 && ./ab_check.sh d113ub d111ub_s)   # 113_gate_uboone.txt
for X in h:pdhd:d108hflip:029107_16 v:pdvd:d103vflip:039349_20; do IFS=: read x det src ev <<< "$X"
  ARM=d113${x}step1 DET=$det SRC=$src JOBS=1 PIN=$P LOGD=/home/xqian/tmp/d113/arm_d113${x}step1 EVENTS=$ev \
    PR_TLA="-S retile_mode='none' -S retile_sampler_strategy='stepped'" bash $S/d111_run_arms.sh; done  # 113_gate_step_identity.txt
python3 $S/d113_grade.py --self-test > $F/113_grade_selftest.txt                                    # reproduces docs 104 / 103

# sec 3 -- the rule, frozen 2026-09-16T13:48:25 before any knob-on arm; amendment 1 at 14:01:39 (sec 3)
(cd $F && sha256sum -c 113_pred.sha256)
python3 $S/d113_compare.py --det pdhd --base d101hnew --arm d102hcs     # reference magnitudes of the accepted flips
python3 $S/d113_compare.py --det pdvd --base d103v0 --arm d103v1
python3 $S/d113_compare.py --det pdhd --base d113hbase --arm d113hbase --logd-base /home/xqian/tmp/d113/arm_d113hbase  # Gc denominators

# sec 4-9 -- the arms (base2 + three levels per detector, all four concurrently) and every clause
DET=pdhd JOBS=4 bash $S/d113_run_levels.sh; DET=pdvd JOBS=4 bash $S/d113_run_levels.sh
for x in h v; do det=$([ $x = h ] && echo pdhd || echo pdvd)
  for a in base none foot nopaint; do
    python3 $S/d113_steiner_census.py --det $det --arm d113$x$a --out $F/113_steiner_d113$x$a; done
  for l in none foot nopaint; do
    python3 $S/d111_eval.py --det $det --base d113${x}base  --arm d113$x$l --out $F/113_eval_${det}_$l.txt
    python3 $S/d111_eval.py --det $det --base d113${x}base2 --arm d113$x$l --out $F/113_evalR_${det}_$l.txt
    python3 $S/d113_compare.py --det $det --base d113${x}base --arm d113$x$l \
        --logd-base /home/xqian/tmp/d113/arm_d113${x}base --out $F/113_compare_${det}_$l.txt; done
  python3 $S/d113_grade.py --det $det --cells A0=d113${x}base,L3=d113${x}none,L2=d113${x}foot,L1=d113${x}nopaint \
      > $F/113_grade_$det.txt; done
python3 $S/d111_identity_gate.py --det pdhd --a d113hbase --b d113hbase2 > $F/113_gate_base2_pdhd.txt   # 61 / 61
python3 $S/d111_identity_gate.py --det pdvd --a d113vbase --b d113vbase2 > $F/113_gate_base2_pdvd.txt   # 120 / 120
python3 $S/d113_spot_figs.py --out $F/113_spot                          # 113_spot_<h1,h2,v1,v2,v3>.png + .tsv
python3 $S/d113_nearfar_movers.py > $F/113_nearfar_movers.txt           # near / far split, tag movers, resources
python3 $S/d113_v3_step.py > $F/113_v3_step.txt                         # the v3 bands (sec 7)
python3 $S/d113_verdict.py > $F/113_verdict.txt                         # every clause; writes 113_arms_complete.txt
```

All arms are new tags (M13). Their pctrees are symlinks to the production arms' (`d108hflip`, `d103vflip`). The dump
logs are in `/home/xqian/tmp/d113/arm_<arm>/`.

---

## 1. What the retile does, and the knob

Each STM candidate's Steiner graph is built on a copy of the cluster. `CreateSteinerGraph.cxx:271` calls the configured
retiler: `ImproveCluster_2` in both PR chains, and in `steiner_refresh` too. `ImproveCluster_2::mutate` makes that copy:

1. The original cluster's `basic_pid` Dijkstra path (`improvecluster_2.cxx:148-170`).
2. A temp retile, `ImproveCluster_1::mutate`, and its own path.
3. Per face, the activity (`get_activity_improved`, `improvecluster_1.cxx:263-455`):
   - the cluster's blob footprints, with sentinel 1e-3 where the CTPC has no charge;
   - **dead channels within 20 cm** (Step 2);
   - **all good CTPC charge within 20 cm** (Step 3).
4. **Painting** (`hack_activity_improved`, called at `improvecluster_2.cxx:256, 261`): a ±3-wire × ±3-slice disc
   around both paths, sentinel 1e-3.
5. Tiling, `sample_live` with the per-face sampler (`charge_stepped` since docs 103 / 108), and `remove_bad_blobs`.

The sentinel becomes (0, 1e12). `charge_stepped` with `disable_mix_dead_cell=false` admits a wire at charge exactly 0
(`BlobSampler.cxx:1291`; pinned by the new doctest). So painted cells become points with a zero-charge plane: doc 112's
ghosts.

**The retile is also the only place `charge_stepped` runs in the PR job.** Clustering still samples `stepped`
(doc 108). Dropping the retiler is not an option either: `CreateSteinerGraph`'s whole loop runs only when a retiler is
configured. So "remove the retile, keep `charge_stepped`" needs a copy of the cluster that is re-sampled but not
re-tiled.

**`retile_mode`** (`ImproveCluster_2`, C++ default `"full"`; the non-full modes are in `mutate_reduced`,
`improvecluster_2.cxx:376`). Each level removes one more piece of the fabrication:

| level | mode | what the Steiner copy is |
|---|---|---|
| — | `full` | today's retile (production) |
| L1 | `no_paint` | steps 1-3 and 5, **no painting** (and so no temp retile, whose only use is the second painting) |
| L2 | `footprint` | L1 and **no dead / nearby-charge extension**: the cluster's own footprints re-tiled (`get_activity_improved(..., extend=false)`) |
| L3 | `none` | **no tiling**: each original blob rebuilt from its wire bounds and re-sampled in place with the same samplers. Activity comes from the grouping's CTPC row (live), else the dead registry (0, 1e12), else absent. The loop is duplicated from `ClusteringResampleLive` (that file is untouched) |

- **The orig block runs in every mode.** Step 1 runs first, so the `basic_pid` graph it caches on the source cluster
  (read later by `TrackFitting.cxx:3104` and `PRSegmentFunctions.cxx:3415`) is the same in every arm. The A/B changes
  only the copy.
- **Census line.** Non-full modes log one `RETILEMODE` line per mutate: points and xyz sums of the source and the copy.
- **"none" still keeps zero-charge cells.** It means no new blobs. A dead wire inside an original blob is still
  written (0, 1e12) and still admitted.

## 2. Gates: knob off is byte-identical

| gate | result | record |
|---|---|---|
| G1 compiled PR config, knob unset | PDHD `a870511c9b22`, PDVD `211a49a48229`, unchanged. With `-S retile_mode='none'` the only difference is `data.retile_mode` in the one `ImproveCluster_2` node | `/home/xqian/tmp/d102/cfg/d113knob{off,none}_{pdhd,pdvd}.json` |
| G2 runtime, production config with dump | `d113hbase` == `d111hst` **61 / 61**; `d113vbase` == `d111vst` **120 / 120**; base2 reruns identical **61 / 61**, **120 / 120** | `figs/113_gate_base_*.txt`, `figs/113_gate_base2_*.txt` |
| G3 other detectors (same C++) | SBND `d113snew` == `d111ssnew` **BYTE-IDENTICAL** (16 events). uBooNE: zips **35 / 35**; tagger 34 / 35, the one difference being event 6805's `kine_pio_*_2`, a known bistable event (doc 90) that differs in 56 earlier gate logs | `figs/113_gate_sbnd.txt`, `figs/113_gate_uboone.txt` |
| G4 L3's rebuild is exact | `none` with a `stepped` sampler reproduces the clustering cloud on **every** mutate: PDHD 029107_16 **419 / 419** (171,891 points), PDVD 039349_20 **221 / 221**; point counts and xyz sums equal to 17 digits | `figs/113_gate_step_identity.txt` |
| doctest | `wcdoctest-clus` **430 / 430**; new `doctest_retile_mode.cxx` (default, typo refused, rebuilt blob = tiled blob under production `charge_stepped`, zero-charge wire admitted) | — |
| freshness | `libWireCellClus.so` 13:21:16 is newer than every edited source (last 13:20:00) | — |
| grader | `d113_grade.py --self-test` reproduces the folded grades of doc 104 (PDHD) and doc 103 round 5 (PDVD) count for count | `figs/113_grade_selftest.txt` |
| census | `d113_steiner_census.py` on `d111hst` / `d111vst` reproduces doc 111's ladder base rows (9.87 / 4.00 / 4.43 / 59.19 %; 7.82 / 2.31 / 9.71 / 49.51 %) and doc 112's terminal table | `figs/113_steiner_d113{h,v}base.txt` |

## 3. The rule

`figs/113_pred.txt`, sha256 `a1275a12…`, frozen **2026-09-16T13:48:25**, after the gates and the base census and before
any knob-on arm started. `d113_run_levels.sh` refuses to launch on a sha mismatch.

- **FIX, all required on both detectors:**
  - **S1:** seed length > 1 cm off the ridge drops ≥ 25 %.
  - **P1:** fit rows > 1 cm off drop ≥ 20 %.
  - **P5:** seed max offset ≤ 1.2 cm at h1 and at v3.
- **NO NEW ISSUES, all required:**
  - **T:** the four tag metrics each ≥ base − 0.020 on the folded truth, with the unlabelled candidates bounded both
    ways; UNDECIDED if a bound straddles.
  - **Fit quality:** P2 (off-charge rows) ≤ base; P3 (Bee holes) ≤ base; P4 (coverage) ≥ base − 1 pt; P6' (clean
    records) ≤ 1 %; P5f (spot fit max) ≤ base + 0.2 cm.
  - **Gap jumping, where the reference flips are d101hnew→d102hcs and d103v0→d103v1:**
    - Ga: truncated fits ≤ the accepted `charge_stepped` flip's share;
    - Gb: far deviations ≤ 1.10 × base;
    - Gc: tags whose production graph touched a dead channel not lost more than the others + 10 points, and not more
      than the flip's loss + 5 points;
    - Gd: clusters losing every fit ≤ the flip's count.
  - **C:** completeness. **R:** wall ≤ 1.20 × base2.
- **Predictions written into it:**
  - L3 removes the one-blank terminals;
  - part of the zero-charge planes turn into dead-plane ones;
  - far deviations do not shrink;
  - tags churn.

**Amendment 1** (`figs/113_pred_amend1.txt`, hashed into `113_pred.sha256`). It was written after three PDHD knob arms
finished and before any metric was computed.
- **What prompted it:** the runner flagged event 028084_14 (PDHD, all three levels) and 039252_13 (PDVD `none`)
  "incomplete". Both jobs exited 0 and wrote every output. The runner greps a `CheckSTM_Michel: N candidate(s)` line
  that is printed only when there are candidates, and these events lost theirs in the arm.
- **What it changed:** C now reads "wire-cell rc 0 and the three output files". The lost candidates are counted by Gd
  and T anyway. Nothing else changed.

## 4. Does it fix the problem?

### 4.1 The ghosts go; the off-image seed shrinks by less than the bar

`figs/113_nearfar_movers.txt`, `figs/113_steiner_*.txt`:

| arm | seed > 1 cm off | near 1–5 cm | far > 5 cm | Steiner cloud | terminals (> 1 cm off) | one-blank terminals | dead-plane terminals | ctpc/MST bridges |
|---|---|---|---|---|---|---|---|---|
| PDHD production | 9.87 % | 5.87 % | 4.00 % | 11.57 M | 382,126 (87,618) | 81,995 | 14,797 | 2,413 |
| PDHD L1 no_paint | 8.65 % | 4.46 % (−24.0 %) | 4.19 % | 6.33 M | 354,742 (50,602) | 464 | 16,802 | 3,778 |
| PDHD L2 footprint | 8.56 % | 4.32 % (−26.4 %) | 4.24 % | 6.02 M | 348,618 (46,628) | 368 | 13,114 | 4,911 |
| PDHD L3 none | 8.51 % | 4.36 % (−25.8 %) | 4.15 % | 6.00 M | 363,796 (47,743) | 57 | 20,158 | 5,224 |
| PDVD production | 7.82 % | 5.51 % | 2.31 % | 13.62 M | 672,331 (96,850) | 119,476 | 70,930 | 6,096 |
| PDVD L1 no_paint | 6.43 % | 4.01 % (−27.2 %) | 2.42 % | 6.39 M | 668,081 (81,589) | 11,128 | 82,022 | 9,226 |
| PDVD L2 footprint | 6.18 % | 3.78 % (−31.3 %) | 2.40 % | 5.87 M | 654,119 (71,982) | 10,147 | 65,558 | 13,039 |
| PDVD L3 none | 6.06 % | 3.79 % (−31.1 %) | 2.27 % | 5.88 M | 700,630 (71,712) | 9,666 | 86,490 | 14,709 |

- **The painting is the main source.** `no_paint` alone removes 99.4 % (PDHD) and 90.7 % (PDVD) of the one-blank
  terminals and more than half the cloud. Footprints and dead / nearby fill are still there in that level.
  - Doc 112 sec 9 left "painted cell or wire-range sentinel?" open.
  - **PDHD, answered:** footprint sentinels survive in `no_paint` and give 464 one-blank terminals; `footprint` 368;
    `none` 57. The ghosts are painted cells.
  - **PDVD, partly answered:** `no_paint` leaves 11,128, `footprint` (which also drops the footprint sentinels) still
    10,147, and `none` 9,666. The ~9,700 that survive every level are not painted cells or footprint sentinels, and
    their source is not identified this round (sec 11).
- **The prediction held for dead-plane terminals.** In `none` they rise on both detectors (PDHD 14,797 → 20,158,
  PDVD 70,930 → 86,490) as dead wires inside original blobs are sampled at charge 0. The share of them > 1 cm off
  falls (25.0 → 17.1 %, 22.3 → 16.4 %).
- **The ghosts' part shrinks as expected.** Near deviations drop 24–31 %, in every level, on both detectors, above the
  25 % bar where the ghosts live.
- **Far deviations are flat or up 4–6 %.** On the whole seed length they carry S1 below its bar: −13.8 % PDHD, −22.5 %
  PDVD.
  - Far deviations are the Steiner graph's straight bridges across gaps (doc 111 r2: 93–97 % have no on-image route).
    The retile does not make them, so removing it cannot remove them.
  - PDHD's far share is 4.0 %, 40 % of its off-image seed at production, which is why S1 moves less there.

### 4.2 The fit moves less than the seed

`figs/113_eval_*.txt`:

| clause | PDHD L3 / L2 / L1 | PDVD L3 / L2 / L1 |
|---|---|---|
| P1 fit rows > 1 cm (bar −20 %) | 7.74 → 7.00 / 7.09 / 7.24 % (−9.6 / −8.4 / −6.7 %) | 5.10 → 4.71 / 4.82 / 4.90 % (−7.6 / −5.3 / −3.7 %) |
| rows > 2 cm | 4.72 → 4.65 / 4.70 / 4.79 % | 2.91 → 2.85 / 2.93 / 2.93 % |
| P2 rows off-charge in ≥ 1 plane (≤ base) | 7.99 → 7.41 / 7.46 / 7.74 % pass | 10.57 → 10.50 pass / 10.59 FAIL / 10.64 FAIL |
| P3 Bee holes per 10 m (≤ base) | 8.47 → 8.81 / 8.91 / 8.79 FAIL | 5.90 → 6.20 / 6.27 / 6.09 FAIL |
| P4 image coverage (≥ base − 1 pt) | +0.92 / +0.71 / +0.54 pt pass | +0.73 / +0.34 / +0.25 pt pass |
| P6' clean records, rows > 1 cm (≤ 1 %) | 0.19 / 0.19 / 0.21 % pass | 0.16 / 0.17 / 0.16 % pass |
| dQ/dx median | 56.2 → 56.1 ke/cm | 57.1 → 56.9 ke/cm |

The 1–2 cm band improves by about a fifth on PDHD (3.02 → 2.35 % of rows); the > 2 cm rows barely move. Rows the Bee
client drops (q < 0) rise 6–12 %.

## 5. Does it introduce new issues? Tags

`figs/113_grade_{pdhd,pdvd}.txt`.
- **Truth:** PDHD `own103h2 > own103h > smx27 > smx28` (amendment 5 reading); PDVD
  `own103v2 > own103v > p99rwon_carried_corrected > smx11`.
- **Cells:** production + the three levels. Population 331 / 578.
- **Unlabelled candidates:** 47 / 51, bounded by NEG / POS.

| metric | PDHD base | L3 none | L2 footprint | L1 no_paint |
|---|---|---|---|---|
| `is_stm` purity | 0.959 | 0.958 (UNDECIDED) | 0.967 | 0.952 (UNDECIDED) |
| `is_stm` efficiency | 0.636 | 0.614 (−0.022, UNDECIDED) | 0.641 | 0.641 |
| `michel_found` purity | 0.880 | 0.886 | 0.873 | 0.877 |
| `michel_found` efficiency | 0.702 | **0.596 (−0.106, FAIL)** | 0.663 (−0.038, UNDECIDED) | 0.683 |
| T | | **FAIL** | UNDECIDED | UNDECIDED |

| metric | PDVD base | L3 none | L2 footprint | L1 no_paint |
|---|---|---|---|---|
| `is_stm` purity | 0.982 | **0.956 (−0.025, FAIL)** | **0.945 (FAIL)** | **0.953 (FAIL)** |
| `is_stm` efficiency | 0.709 | 0.701 | 0.693 | 0.696 |
| `michel_found` purity | 0.940 | **0.861 (−0.079, FAIL)** | **0.860 (FAIL)** | **0.874 (FAIL)** |
| `michel_found` efficiency | 0.725 | 0.708 | 0.703 (UNDECIDED) | 0.678 (UNDECIDED) |
| T | | **FAIL** | **FAIL** | **FAIL** |

The label sources of `none`'s movers (`figs/113_nearfar_movers.txt`):

- **PDHD Michel:** 21 true Michels lost against 10 gained (lost: owner_review 8, owner 2, smx27 7, agent 4). Only 2
  new false positives.
- **PDVD:** 9 new `is_stm` and 20 new Michel false positives. Their labels are the carried record (6 / 15) and agent
  labels (3 / 5), **none owner-reviewed**. One of them is 039349_20 cl80, the cluster of the owner's v1 / v2.
- **Churn is balanced:** net `is_stm` tags PDHD −2, PDVD +7. It is about as large as the accepted `charge_stepped`
  flips' churn (PDVD candidates only in one arm 85 / 89 here vs 118 / 141 there).

Doc 103 is the relevant precedent. The owner's adjudication there relabelled 29 of 32 disputed PDVD false positives,
and a D2 turned into a D1. The PDVD purity cost here rests on the same kind of labels. It is a measured cost under the
frozen rule, not yet an owner-confirmed one.

## 6. Did gap jumping break?

`figs/113_compare_*.txt`:

| clause | PDHD L3 / L2 / L1 | PDVD L3 / L2 / L1 | bar |
|---|---|---|---|
| Ga truncated fits (< 0.9 × base length) | 1.58 / 1.23 / 1.57 % | 1.41 / 1.41 / 1.41 % | ≤ 4.48 % / 3.37 % (accepted flip) |
| extended fits (> 1.1 ×) | 5.55 / 5.02 / 4.55 % | 3.37 / 3.05 / 2.23 % | reported |
| Gb far deviations | 4.15 / 4.24 / 4.19 % | 2.27 / 2.40 / 2.42 % | ≤ 4.40 % / 2.54 % |
| Gc tags lost, production graph on a dead channel vs not | 22.9 vs 26.1 / 21.7 vs 26.1 / 15.7 vs 28.3 % | 17.9 vs 32.0 / 19.5 vs 32.0 / 18.3 vs 28.0 % | dead ≤ other + 10 and ≤ flip + 5 |
| Gd clusters losing every STM fit | 16 / 17 / 11 | 2 / 2 / 3 | ≤ 19 / 9 |
| CreateSteinerGraph "no steiner_graph" WARNs | 6,433 → 6,987 / 7,004 / 6,679 | 6,465 → 6,128 / 6,135 / 5,608 | reported |

- **Every gap clause passes.** Tags on clusters whose production graph touched a dead channel (83 of 129 PDHD, 257 of
  282 PDVD) are lost *less* often than the others. The lost ones are listed in the compare files for a look.
- **The graph takes over the jumping.** Where the retile's fill used to put points into a gap, the Steiner graph now
  connects the pieces itself: ctpc/MST bridge edges go 2,413 → 5,224 (PDHD) and 6,096 → 14,709 (PDVD) in `none`.
- **Seed length changes by under 2 %:** PDHD 2,820 → 2,841 m, PDVD 4,368 → 4,287 m.
- **One caveat about the sample.** PDHD and PDVD have few dead channels, and these 61 / 120 events are not selected for
  dead regions. A detector period with more dead channels would need its own look.

## 7. The owner's spots

`figs/113_spot_<spot>.png`:
- **Layout:** one image-only frame per spot, the same in every row. Rows: production, `no_paint`, `footprint`, `none`.
  Columns: two transverse views and the ridge offset along the track.
- **Markers:** grey image; cyan Steiner cloud; triangles are terminals coloured by plane class, red for one blank
  plane; black seed; purple fit.

| spot | seed max, production → none (cm) | fit max, production → none (cm) | one-blank terminals in window (> 1 cm off) |
|---|---|---|---|
| **h1** PDHD cl108 | 1.97 → **0.68** | 1.77 → **0.46** | 5 (4) → 0 |
| h2 PDHD cl106 | 0.84 → 0.60 | 2.55 → **0.76** | 21 (14) → 0 |
| v1 PDVD cl80 | 1.03 → 1.05 | 0.53 → 0.72 | 14 (3) → 0 |
| v2 PDVD cl80 | 0.99 → 0.80 | 1.27 → **0.42** | 10 (6) → 1 (0) |
| **v3** PDVD cl25 | 1.64 → **1.24** | 0.91 → **1.19** | 14 (3) → 0 |

- **h1: fixed.**
  - At production, the red one-blank terminals pull the seed across the bend 1–2 cm off the image (row 1; the chord of
    doc 112 sec 3).
  - In every reduced level the cloud is the image's own points. The terminals are all 3-live and the seed and fit
    follow the image (ridge offset < 0.7 cm).
- **v3: the ghosts go, and a step remains** (`figs/113_v3_step.txt`, 1 cm slabs along s in the same frame).
  - The production cloud is a wide halo with 14 one-blank terminals. In every reduced level it collapses onto the
    image, and the seed's three stretches over 1.5 cm become one step.
  - **The image splits.** For s from −3 to 0 cm the image points span e2 −1.8 to +2.6, with 43–50 % of them above
    +1 cm: two bands. Its charge-weighted e2 centroid is +0.47 to +0.66, between them.
  - **The seed and fit ride the upper band.** Without the retile the seed sits at e2 +1.34 to +1.47 and the fit at
    +0.98 to +1.18. The production seed stayed near the middle (+0.07 to +0.61) and its fit below it (−0.51 to +0.35).
  - **How the ridge scores it.** Against that middle, the reduced seed's largest offset is 1.24 cm in the slab
    [−4, −3), where it crosses between the bands (slab means −0.88 before, +1.41 after). This is why v3 fails P5 (by 0.04 cm) and P5f (by 0.08 cm).
  - **Open:** whether the upper band is a second particle, a delta ray, or a displaced piece of the same track (sec 11).
- **Same at every level.** h1 and v3 look the same in all three levels. Their ghosts were painted cells, consistent
  with sec 4.1.

## 8. The ladder: which piece of the retile matters

| | PDHD L1 no_paint | PDHD L2 footprint | PDHD L3 none | PDVD L1 | PDVD L2 | PDVD L3 |
|---|---|---|---|---|---|---|
| one-blank terminals left | 0.6 % | 0.4 % | 0.1 % | 9.3 % | 8.5 % | 8.1 % |
| near deviations | −24.0 % | −26.4 % | −25.8 % | −27.2 % | −31.3 % | −31.1 % |
| P1 | −6.7 % | −8.4 % | −9.6 % | −3.7 % | −5.3 % | −7.6 % |
| T | UNDECIDED | UNDECIDED | FAIL | FAIL | FAIL | FAIL |
| wall vs base2 | 0.76 | 0.67 | 0.73 | 0.81 | 0.78 | 0.83 |

- **What the painting buys.** Painting alone accounts for most of the seed effect. Removing the dead / nearby
  extension and the tiling adds a few more points of near-deviation drop and of P1.
- **PDHD tags.** On PDHD, `no_paint` has the smallest tag change (every labelled delta within 0.02). It is UNDECIDED
  only through the unlabelled bound on `is_stm` purity.
- **PDVD tags.** On PDVD no level keeps purity.

## 9. Resources

`figs/113_nearfar_movers.txt`, `figs/113_evalR_*.txt`; each level against `base2`, which ran concurrently at the same
JOBS.

| | wall median ratio | total wall | peak RSS median ratio |
|---|---|---|---|
| PDHD none / foot / no_paint | 0.725 / 0.672 / 0.759 | 2,585 / 2,498 / 2,741 s vs 3,639 s | 0.92 / 0.94 / 0.98 |
| PDVD none / foot / no_paint | 0.825 / 0.779 / 0.813 | 3,981 / 3,714 / 3,997 s vs 5,003 s | 0.93 / 0.94 / 0.95 |

R passes everywhere.

## 10. Verdict, and what would come next

**Every level FAILs** (`figs/113_verdict.txt`):
- **L3 `none`:** PDHD S1, P1, T, P3; PDVD S1, P1, P5, T, P3, P5f.
- **L2 `footprint` and L1 `no_paint`:** PDHD S1, P1, P3 (T UNDECIDED); PDVD S1, P1, P5, T, P2, P3, P5f.

The C++ default stays `"full"` and no production config changes.

What the owner may want to decide between (each needs the owner's go):

1. **Adjudicate the tag movers of `none`** before closing the question. The two detectors need different looks:
   - **PDVD, a false-positive review:** 29 new false positives (9 `is_stm`, 20 Michel; 26 distinct clusters), none
     reviewed by the owner. Are these stoppers / Michels after all?
   - **PDHD, a lost-Michel review:** 21 true Michels no longer found. Are they real Michels the reduced Steiner graph
     misses, and why?
   - Doc 103 showed PDVD-style label sets can flip. This settles T only. S1 and P1 would still miss their bars.
2. **Target the far bridges.** Without the retile they are about half of the remaining off-image seed on PDHD
   (4.15 of 8.51 %) and a third on PDVD (2.27 of 6.06 %), and they do not depend on the retile. This is graph-level (the ctpc/MST connection and its straight edges), doc 111 sec 11.8.
3. **The deferred `CreateSteinerGraph` terminal admission (doc 112 sec 8).** Its motivating ghosts are removed by
   `no_paint` alone, so if the painting is ever dropped that fix has little left to do. With the retile kept, it is
   still the narrow fix for the h1-type chord.

**Recommended next step: option 1, a small owner scan of the `none` tag movers.** That is the only item that could
change the reading of "new issues". Only after it, decide whether the seed gain (near deviations −26 / −31 %, h1 fixed,
PR 18–27 % faster) is worth a partial fix.

## 11. Not concluded

- **Which movers are real.** Whether the new PDVD false positives and the lost PDHD Michels are real tagger changes or
  label conditioning (sec 5).
- **A second band at v3.** What the upper image band at v3 is (s −3 to 0 cm, sec 7): a second track, a delta ray, or a
  displaced piece of the same track. No 2-D look was taken this round.
- **The PDVD one-blank residue.** About 9,700 one-blank terminals remain in every reduced PDVD level (1.4 % of
  terminals in `none`), with no painted cells and no footprint sentinels. Candidates, none measured:
  - a live CTPC wire whose charge prints as 0 in the dump;
  - crossings from the ±2-wire activity margin;
  - a wire absent from the CTPC inside a blob.
- **Dead-rich data.** The effect on data with many more dead channels (sec 6 caveat).
- **Levels are not combined with other levers.** Each level is one knob against production. None was combined with a
  far-bridge or terminal-admission change.

## 12. Files

| file | what |
|---|---|
| toolkit `clus/src/improvecluster_2.cxx`, `improvecluster_1.{h,cxx}` | `retile_mode`, `mutate_reduced`, `get_activity_improved(..., extend)` (`d2777286`) |
| toolkit `clus/test/doctest_retile_mode.cxx` | 4 cases (sec 2) |
| toolkit `cfg/pgrapher/experiment/{pdhd,protodunevd}/pr.jsonnet` | `retile_mode=null`, key omitted when null |
| `pdhd/wct-pr-perevt.jsonnet`, `pdvd/wct-pr-perevt.jsonnet` | TLA `retile_mode` |
| `scripts/d113_run_levels.sh` | launch base2 + three levels concurrently; refuses on a rule sha mismatch |
| `scripts/d113_steiner_census.py` | S1 / far / spots seed / terminal classes / bridges per arm |
| `scripts/d113_compare.py` | Ga / Gc / Gd, candidate churn, lost tags split by the production graph |
| `scripts/d113_grade.py` | T on the folded truth, NEG / POS bounds; `--self-test` |
| `scripts/d113_spot_figs.py` | `figs/113_spot_<spot>.png`, `figs/113_spot.tsv` |
| `scripts/d113_nearfar_movers.py` | near / far split, tag movers by label source, resources |
| `scripts/d113_v3_step.py` | `figs/113_v3_step.txt`: the v3 image bands, seed and fit per 1 cm slab (sec 7) |
| `scripts/d113_verdict.py` | every clause, the verdict, `figs/113_arms_complete.txt` |
| `figs/113_pred.txt`, `113_pred_amend1.txt`, `113_pred.sha256` | the frozen rule and its amendment |
| `figs/113_gate_*.txt`, `113_grade_selftest.txt` | gates |
| `figs/113_steiner_*.{txt,json}`, `113_eval_*.txt`, `113_evalR_*.txt`, `113_compare_*.txt`, `113_grade_*.txt`, `113_nearfar_movers.txt`, `113_verdict.txt` | results |

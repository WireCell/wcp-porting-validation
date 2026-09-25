# 124 — Anatomy of the νμ flips on 3000 SBND data events: reco1 flashes vs hit flashes

*2026-09-25. Analysis only: no toolkit or production change. One new arm (`work-mcp2k-d123lgop{,pr}`, the
light gate of doc 123 §18 on mcp2k). Companion of doc 123 (§13.3 counts, §18.7 Bee sets).*

## Repro

```bash
cd sbnd_xin
# 1. the anatomy of every numu > 0.9 flip (data: mcp1k, mcp2k; MC: r3cv)
for s in mcp1k mcp2k r3cv; do
  python3 scripts/d124/flip_anatomy.py --a work-$s-d123basepr --b work-$s-d123hitspr \
      --pa products/d123/${s}_base --pb products/d123/${s}_hits --label base,hits --sample $s \
      --tsv docs/124_flips/124_anatomy_$s.tsv --detail docs/124_flips/124_anatomy_${s}_detail.txt
done
python3 scripts/d124/mc_flip_truth.py docs/124_flips/124_anatomy_r3cv.tsv products/d123/r3cv_base products/d123/r3cv_hits
# 2. determinism null: stage B re-run on the flip events of each arm (fresh roots, d123 pin, pr_display)
LD_LIBRARY_PATH=~/tmp/d123-libpin:~/tmp/d123-libpin/reco1/lib:$LD_LIBRARY_PATH PR_EXTRA_STAGES=pr_display PR_JOBS=8 \
  ./run_pr_chain_batch.sh work-<s>-d123<arm> ~/tmp/d124/null/<s>_<arm> data <events>     # ~/tmp/d124/chain_null.sh
python3 scripts/d124/pr_null_cmp.py work-<s>-d123<arm>pr ~/tmp/d124/null/<s>_<arm>
# 3. the flash-time nudge: reco1-arm Q/L output with the beam clusters' t0 moved to the hit-flash time
python3 scripts/d124/t0_nudge.py --ql work-<s>-d123base --pra work-<s>-d123basepr --prb work-<s>-d123hitspr \
    --out ~/tmp/d124/nudge/ql_<s>_{null,t0b} --mode {null,t0} <events>                    # then run_pr_chain_batch.sh
python3 scripts/d124/nudge_cmp.py work-<s>-d123basepr ~/tmp/d124/nudge/pr_<s>_t0b work-<s>-d123hitspr docs/124_flips/124_anatomy_<s>.tsv
# 4. blinded display + scan (5 subagents, docs/124_flips/124_scan_protocol.md), then unblind
python3 scripts/d124/flip_display.py --ql-a work-<s>-d123base --pr-a work-<s>-d123basepr --ql-b work-<s>-d123hits \
    --pr-b work-<s>-d123hitspr --map docs/124_flips/124_blind_map.tsv --outdir docs/124_figs --tag <s> <events>
python3 scripts/d124/unblind.py docs/124_flips/124_scan_verdicts.tsv docs/124_flips/124_blind_map.tsv docs/124_flips/124_anatomy_mcp{1k,2k}.tsv
#    scanner calibration: the 30 MC flips, one flip_display.py call per file sub-root (--seed 1240 --tag mcfNNN,
#    map docs/124_flips/124_blind_map_mc.tsv, outdir docs/124_figs_mc; ~/tmp/d124/mc_render.sh), 5 fresh subagents, then
python3 scripts/d124/scan_calibration.py docs/124_flips/124_scan_verdicts_mc.tsv docs/124_flips/124_blind_map_mc.tsv \
    docs/124_flips/124_anatomy_r3cv.tsv products/d123/r3cv_base products/d123/r3cv_hits work-r3cv-d123basepr
# 5. context: fragment distances, cross-TPC pairing census, data-vs-MC flash census, the light gate on mcp2k
python3 scripts/d124/fragment_distance.py docs/124_flips/124_anatomy_<s>.tsv work-<s>-d123basepr work-<s>-d123hitspr
python3 scripts/d124/beam_dt_census.py work-mcp{1k,2k}-d123{base,hits}pr
for f in work-r3cv-d123hits/f*; do python3 scripts/d123/r1_census.py $f --mc --summary ~/tmp/d124/census/$(basename $f).json; done
PIN=~/tmp/d123-libpin-r6 QLTLA=scripts/d123/tla/xtpc_lgop.txt JOBS=8 scripts/d123/hits_arm.sh work-mcp2k-d123base work-mcp2k-d123lgop data
PIN=~/tmp/d123-libpin-r6 JOBS=16 scripts/d123/stageB.sh work-mcp2k-d123lgop data; scripts/d123/pr_tables.sh work-mcp2k-d123lgoppr products/d123/mcp2k_lgop
python3 scripts/d124/doc_table.py                                                        # the §6 table
```

## 0. Answers in brief

The 3000 data events (mcp1k + mcp2k) have **46 events whose νμ > 0.9 verdict differs** between the reco1-flash
arm and the hit-flash arm (23 lost, 23 gained; doc 123 §13.3). Every one was taken apart: what the PR stage
was given in each arm, and what it produced. Both reconstructions were also given a blind visual scan. The scan was
done by five model subagents that saw only the images and a written protocol, not by a human. Calibrated on the
30 MC flips with truth (§4.1), it is right 11 times in 13 on Q/L-driven flips and has no skill on the rest.

1. **Q/L-driven flips: 17 of 46** (10 gained, 7 lost). The neutrino candidate's main cluster is matched to a
   different flash, so the candidate appears, disappears or is a different object.
   - The blind scan prefers the hit-flash reconstruction in **9 of the 10 gains** (labelled numuCC in 7,
     unclear in 2): a track on a beam-window flash that reco1 had merged into a cosmic flash or vetoed. The tenth (165060) is a
     through-going cosmic that the new flash made a candidate.
   - Of the 7 losses, the scan prefers the hit-flash arm in 3: **candidates it judges cosmic-like, or not a
     neutrino, that the hit flashes removed** (402297 labelled cosmic; 158227 and 174422 labelled unclear, the
     latter at confidence 1). **4 are real losses**: 59003 and 169824 (the cathode-crosser cull that the light
     gate of doc 123 §18 fixes), 390743 and 157215 (§2.3).
2. **"The vertex changed": 29 of 46 flips, and it is not the Q/L match of the interaction.** In these events the
   main cluster, its flash and its charge are the same in both arms. What differs is either
   - (19 events) 1–3 **tiny clusters of 7–37 points, every one more than 1.1 m (typically 2–5 m) from the
     vertex**, that join or leave the beam bundle; or
   - (10 events) **nothing in the bundle at all**: identical points, only the ~10 ns flash time and the
     cluster numbering or flash list of the rest of the event differ.

   The PR stage is **exactly reproducible run to run** (92 of 92 reruns identical, §3.1), so this is not
   randomness in the usual sense. It is **sensitivity**: decisions near a tie (muon vs pion ID of the long
   track, which end the vertex goes on, a track split into track + shower) flip on inputs that should not
   matter. A 10 ns shift of the beam clusters, 0.016 mm in x, alone reproduces the hit-flash verdict in 6 of
   24 tested events (§3.3).
3. **On MC truth the two classes behave differently** (§4). The Q/L-driven flips are 13 gained / 1 lost, with
   10 of the 13 gains true νμCC. The sensitivity flips are 7 gained / 9 lost with signal 3 / 4: a symmetric
   random walk that adds noise and no bias. On data the sensitivity flips are also symmetric (13 / 16).
4. **Do we have the Q/L gain we expected? Yes in kind, smaller in size, and compatible within about 1σ** (§5).
   - The raw 784 → 784 count hides it, for two reasons. The Q/L class on data is +10 / −7, and the scan judges
     3 of the "losses" to be non-neutrino candidates removed. And that count had no fiducial cut: with the MC's
     selection definition, data goes **655 → 661 (+6)**.
   - The MC's signal-only gain is +9 per 2017 events, which scales to about +13 on 3000 data events. The cosmic
     term is 0 in the MC overlay and +6 ± 6 from beam-off. So the expectation is +13 to +19, against +6 ± 7 observed.
   - The large beam-flash partners that make the Q/L changes occur at the same rate in data and MC (§5.3). The
     data have fewer recoveries (9 vs about 16 scaled) and more losses (5 vs 0). Two of those losses are the
     cathode crossers the light gate fixes; the rest are within the statistics of this sample.
   - With the recommended light gate the data reach **664 (+9 over reco1 flashes)**, inside 1σ of the MC-based
     expectation (§5.4).
5. **Other findings** (§5–§6):
   - The recommended light gate (doc 123 §18) recovers 59003 and 169824 on mcp1k. On mcp2k it changes no flip,
     and it adds two passes that no arm had before (273559 in the FV). It does not recover 390743 or 157215.
     With it the 3000-event count is **664** under the MC definition (655 / 661 for reco1 / hits flashes),
     and 788 without the FV cut (§5.4).
   - Data have 5× more *small* (20–100 PE) pulses beside the beam flash than the MC (§5.3). They do not drive
     the flips, but they are a data/MC difference in the light.
   - Hit flashes crowd the cross-TPC pairing window (pairs within 10 ns of the 50 ns edge: 18 → 43 per
     3000 events), though no flip traces to a pairing change.
   - The PR's dependence on far specks and on cluster numbering is an invariance defect worth a separate
     look (§7).

## 1. The 46 events and their classes

`scripts/d124/flip_anatomy.py` compares the two PR outputs of each event.

- **Cross-arm key.** The key is the point, not the cluster id. PR cluster ids are renumbered between arms,
  and one id can hold different points. A point is keyed by (y, z, charge) from the PR job's own
  clustering layer. y and z do not move with the flash time; only x = x_raw − t0 · v_drift does.
- **PR input.** A candidate's PR input is its selected main plus the clusters `T_tagger.act_in_pr` marks.
- **Flash provenance.** Every cluster's flash comes from `T_cluster`.

Classes (first match wins): **QL-main** (the selected main's points differ, Jaccard < 0.5, or one arm has no
candidate); pairing (the input differs by clusters that entered through the cross-TPC flash group); t0-merge
(the main's own points differ); **fragment** (same main, separate small clusters differ); **PR-only**
(identical input points).

| data, 3000 events | gained | lost | total |
|---|---|---|---|
| QL-main | 10 | 7 | 17 |
| fragment | 9 | 10 | 19 |
| PR-only | 4 | 6 | 10 |
| pairing / t0-merge | 0 | 0 | 0 |
| **all** | **23** | **23** | **46** |

No flip falls in the pairing or t0-merge class. Two QL-main losses involve a merge (§2.2):
- **158227:** the hit-flash timing joins the candidate with the other TPC's half of a longer track, and the
  merged object is STM-tagged.
- **402297:** the candidate's charge becomes part of a 400 cm cluster on a −673 µs flash. The detail sheets (`docs/124_flips/124_anatomy_{mcp1k,mcp2k}_detail.txt`) list, per event and arm, the
beam-window flashes of both TPCs, the PR input with the differing clusters, where those clusters are in the other
arm (their flash and time), and the BDT-input groups that changed.

## 2. The Q/L-driven flips (17)

### 2.1 Gains: the recovered beam-window flash (10)

| event | what the hit flashes changed | blind scan |
|---|---|---|
| 280884 | a 1.32 µs 16.8 k PE TPC0 flash that reco1 lacks; a contained 147 cm track that was absent from the reco1-arm PR scope (unmatched there, or placed outside the drift volume by its t0) now matches it | hits (2) |
| 281808 | reco1 matched the 186 cm track to a −2.47 µs flash; the hit finder splits out a 0.84 µs 14.2 k PE flash and the track takes it | hits (2) |
| 390644 | 150 cm track moves from a −1.99 µs flash to a new 1.44 µs 9.5 k PE flash | hits (3) |
| 68428 | 195 cm track moves from −0.76 µs to a new 2.04 µs 14.7 k PE flash | hits (3) |
| 53749 | 126 cm track + shower moves from 280 µs to a new 1.79 µs 19 k PE flash | hits (3) |
| 105338 | 123 cm track moves from a 0.19 µs flash (just outside the PR gate) to a new 1.25 µs 17.5 k PE flash | hits (2) |
| 163595 | 179 cm track, absent from the reco1-arm PR scope, on a new 1.08 µs 14.7 k PE TPC1 flash | hits (2) |
| 399963 | 195 cm track moves from −138 µs to a new 1.05 µs 10 k PE flash (reco1 had only the 1.86 µs one) | hits (2) |
| 66944 | same event, larger candidate: a 100 cm track joins the 56 cm main on the beam flash (it sat on a 355 µs flash); numu 0.03 → 1.77 | hits (1) |
| **165060** | a 401 cm cluster moves from −330 µs to the 2.05 µs beam flash; the scan calls it a **through-going vertical cosmic** | **reco1 (3)** |

These are the §13.2/§14.2 "recovered beam-window flash" class of doc 123. On data they appear as 9 plausible
neutrinos and one cosmic fake. The fake is the beam-off cost of doc 123 §14.1 made visible.

### 2.2 Losses where the scan prefers the hit-flash arm: candidates judged not a neutrino (3)

| event | reco1 arm | hit-flash arm | blind scan |
|---|---|---|---|
| 158227 | a 216 cm TPC1 main on a 1.25 µs flash; TPC0 had no beam-window flash | a 1.258 µs TPC0 flash (10.9 k PE) is restored; the two halves share a time, so the stage-A clustering joins them into one 415 cm cluster, which is **STM-tagged** | hits (2), labelled unclear: "half of one long straight track, vertex mid-track, likely a through-going muon" |
| 402297 | a 230 cm piece on the 1.43 µs flash, vertex mid-track, predicted light 32.5 k vs 18.3 k measured | the same charge is part of a 400 cm cluster at −673 µs; the beam flash is matched well (16.8 k vs 18.4 k) by other charge | hits (2); labelled cosmic |
| 174422 | a 93 cm track placed at the cathode (mu321), over-predicting its flash | the track moves to a 558 µs flash, where it reaches the anode; no candidate | hits (1), labelled unclear: "a cosmic touching the boundary" |

### 2.3 Real losses (4)

| event | mechanism | blind scan | light gate (lgop) |
|---|---|---|---|
| 59003 | the `QLXTPC coincident` cull drops the TPC0 half of a 298 cm cathode crosser (doc 123 §18.1) | reco1 (3) | **recovered whole**: numu 3.65, 819 MeV |
| 169824 | both halves of a 325 cm crosser pulled onto the restored −3.14 µs flash (doc 123 §13.2) | reco1 (3) | **recovered**: numu 5.88 |
| 390743 | the 65 cm main (muon + protons) moves from the 1.89 µs 15 k PE TPC0 flash to a 172 µs flash; the beam flash is left unmatched and the candidate becomes an 18 cm edge piece | reco1 (2) | not recovered (numu −2.17, as with hit flashes) |
| 157215 | the vertex-moved 212 cm track becomes STM-tagged in the hit-flash arm, and the bundle is rejected | reco1 (2); the scanner notes that a stopping cosmic cannot be ruled out | not recovered (no candidate) |

## 3. The "vertex changed" flips (29): sensitivity, not randomness

### 3.1 The PR stage is reproducible run to run

The 46 flip events were re-run through stage B in both arms: 92 runs into fresh roots, with the same d123 pin
and the DL vertex on. Every branch of `T_tagger`, `T_kine`, `T_bundle`, `T_cluster`, `T_flash` and `T_rec_charge`
came out identical, as did the nusel rows. There were no "DL vertex failed" lines
(`docs/124_flips/124_null_pr_rerun.txt`). M4's caveat on the DL vertex does not bite here: identical input gives
an identical answer.

### 3.2 Fragments: far specks move in and out of the bundle (19)

| | |
|---|---|
| points that differ in the PR input | 7–37 per event (1–3 clusters of 1.2–2.7 cm) |
| distance of those points from the neutrino vertex | **min 111 cm, typically 190–500 cm** (`124_fragment_distance.txt`) |
| main cluster | identical points in both arms |
| where the specks go | a different flash (for example 67868: three 1.2–1.5 cm clusters on the 1.99 µs beam flash with reco1, on a −228 µs flash with hit flashes) |

The specks have nothing to do with the interaction, yet their presence changes the PR result. Examples:
- **67868:** the long track is pi268 with reco1 flashes and mu257 with hit flashes (`numu_cc_flag` 0 → 1), numu 0.47 → 3.25, vertex identical.
- **59261:** mu235 → pi246, numu 3.85 → 0.15, vertex moved 91 cm.

The blind scanners saw exactly this: "the same track, one row adds an unrelated red speck far away". They
preferred the speck-free row, at low confidence.

### 3.3 Identical input (10), and the flash-time nudge

In 10 flips the PR input points are identical. In four of them (71642, 275385, 281985, 393538) every cluster of
the whole event is identical, and only the cluster numbering and the flash list differ. The hit finder adds 3–7
flashes per event.

**The nudge test.** `t0_nudge.py` rewrites the reco1-arm Q/L output with one array changed: each beam bundle's
`cluster_t0` is moved to the hit-flash time of its TPC. That is a 1–12 ns move, 0.002–0.019 mm in x.
Everything else stays the reco1 arm's: the clusters, ids and flash list. The PR stage rebuilds corrected x from
`cluster_t0`, so this moves the beam clusters rigidly. A repack with no change is the null, and it came out
identical in 24 of 24 events (`124_t0_nudge_repack_null.txt`).

| 24 flips tested (`124_t0_nudge.txt`) | verdict follows the hit-flash arm | verdict stays reco1 |
|---|---|---|
| PR-only (9) | **4** (285531 and 281985 reproduce the hit-flash output exactly; 71642, 293536 the verdict) | 5 |
| fragment (15) | 2 (396222, 410566) | 13 |

So:
- A 10 ns flash-time change **alone** flips 6 of 24 verdicts. That is a direct measurement of the tie-sensitivity.
- In 13 of 15 fragment events the nudge leaves the verdict alone. What flipped those is the specks, or the
  cluster renumbering and flash list that change along with them. The five identical-input events show that
  renumbering and the flash list alone can flip a verdict, so the three causes are not separated here.
- Five PR-only flips do not follow the nudge. Their only remaining differences are the cluster numbering and the
  flash list of the rest of the event. These could not be separated further, because the flash lists differ in
  length, so there is no one-to-one swap.

### 3.4 What changes inside the PR stage

For the churn flips, the `T_tagger` groups that differ are, in order of frequency:
- the `numu_cc_*` block and the cosmic-tagger `cosmict_*` block, i.e. the muon-candidate identification;
- `shw_sp_*` and `ssm_*` when a segment is re-labelled shower or track;
- the neutrino type.

In the particle lists the recurring changes are mu ↔ pi of the longest track (60933, 67868, 71642, 285531, 59261), a track split into mu + shower piece (275385), and the vertex placed at the other end of the
track (281985, 497399, 285665, 172832). The vertex moves by more than 5 cm in 12 of the 29. In 15 it stays
within 1 cm and only the labels change. **So "the vertex changed" is often the particle ID changed, and both are
the same tie-sensitivity.**

## 4. MC truth: the Q/L class carries the gain, the sensitivity class is noise

The same anatomy on the round-3 inclusive MC (`mc-cv`, 2017 events; `124_anatomy_r3cv.tsv`, `124_r3cv_truth.txt`).
Signal means reco vertex in the FV and within 5 cm of a true νμCC vertex:

| MC flips, 30 | gained: signal / bkg | lost: signal / bkg |
|---|---|---|
| QL-main (14) | **10 / 3** | 0 / 1 |
| fragment + PR-only + t0-merge (16) | 3 / 4 | 4 / 5 |

- On MC the Q/L-driven change is one-sided and mostly signal: the recovered beam-window flash.
- The sensitivity class moves signal both ways in equal measure (3 in, 4 out) and background both ways (4 in, 5 out).
- The data show the same symmetric pattern for the sensitivity class: 13 gained, 16 lost.
- The data scan of this class reads hits 17, reco1 6, same 4, unclear 2, mostly at confidence 1. That tally is
  largely built in: in 17 of the 19 fragment events the far specks sit in the reco1 arm's bundle, and the protocol
  tells the scanner to penalise far unrelated pieces. The tally says nothing about which verdict is right.
- The finding behind it is real, though. **In the flip sample, the hit flashes almost always take the far specks
  *off* the beam flash** (17 of 19). That is a one-directional Q/L effect whose effect on the verdict is
  symmetric. At the population level, doc 123 §13.1 found the beam-window flow of small clusters balanced
  (332 out, 335 in on mcp1k), so this may be specific to the flip sample.

### 4.1 How good is the blind scan? Calibrated on the 30 MC flips

The 30 MC flips were rendered with the same display and a fresh per-event shuffle (`124_blind_map_mc.tsv`,
`docs/124_figs_mc/`). Five fresh subagents scanned them under the identical protocol (`124_scan_verdicts_mc.tsv`).
The file names showed the scanners that these were MC, but nothing showed which row was which arm. The
**truth-right arm** is the one whose νμ > 0.9 verdict is correct: pass if its candidate is signal, fail otherwise.
Scores are in `124_scan_calibration_mc.txt`:

| MC flips | scan agrees with truth | disagrees | "same" / unclear |
|---|---|---|---|
| QL-main (14) | **11** | 2 | 1 |
| churn (16) | 4 | 4 | 8 |
| confidence 3 (all classes) | 5 | 1 | 2 |

- On the Q/L-driven flips the scan is right **11 times in 13 decided**. Both misses are true neutrinos outside the
  FV or off-vertex that the scanner, who cannot see the FV cut, judged as found. So the data scan's reading of
  the QL-main class (§2) is worth about 85 %.
- On the churn flips the scan has **no skill** (4 agree, 4 disagree, and half "same"). Particle-ID and vertex-end
  ties cannot be settled by eye in these displays. This is consistent with §4: that class is noise, and its
  scan verdicts on data (§6) should not be read as physics.

## 5. The expected gain on data

### 5.1 Count with one definition

| νμ > 0.9 (best candidate per event) | reco1 | hits | light gate (lgop) |
|---|---|---|---|
| data mcp1k, no FV cut (doc 123 §13) | 271 | 271 | 273 |
| data mcp1k, FV + `vertex_default==0` (the MC definition) | 231 | 231 | 233 |
| data mcp2k, no FV cut | 513 | 513 | 515 |
| data mcp2k, FV + default vertex | 424 | 430 | 431 |
| **data 3000, FV + default vertex** | **655** | **661 (+6)** | **664 (+9)** |
| data 3000, no FV cut | 784 | 784 | 788 |
| MC cv 2017, FV + default vertex | 446 | 455 (+9) | 454 |
| beam-off 1000 gates, FV + default vertex | 4 | 6 | 7 |

Flips under the MC definition:

| sample | gained | lost | of which QL-main, gained / lost | sign test |
|---|---|---|---|---|
| data 3000 | 25 | 19 | 9 / 5 | p = 0.45 |
| MC 2017 | 16 | 7 | 11 / 0 | p = 0.09 |

### 5.2 The arithmetic

- **MC signal-only gain.** In the MC truth join (§4), the FV-selected flips are 13 signal gained against 4
  signal lost, so +9. Background is 3 gained against 3 lost, so 0. Per event that scales to **+13 on 3000 data events**.
- **The cosmic term replaces the MC's background term; it is not stacked on the MC total.** The MC carries a
  cosmic overlay and shows 0. The beam-off gates show +2 per 1000 (3 gained, 1 lost), that is **+6 ± 6 on 3000**.
- **Expectation vs observation.** The expectation is +13 (MC overlay) to +19 (beam-off cosmics). Observed is
  **+6 ± 7** (σ ≈ √44 flips), so it is low by 1σ to 2σ depending on the cosmic term. That is a tension, not a
  contradiction.
- **Class by class.** QL-main gains are 9 on data against 16 expected from the MC rate (11 per 2017 events).
  QL-main losses are 5 on data against 0 on MC.
- **Scan reading of the data losses.** The scan prefers the hit-flash arm in 3 of the 7 losses (§2.2). The light
  gate recovers 2 more (§2.3).
- **Summary.** The Q/L change on data finds **9 plausible neutrinos, removes 3 candidates the scan does not take for
  neutrinos, and adds 1 cosmic, against 4 losses of which the light gate recovers 2.** That is a real gain,
  somewhat smaller than the MC's.

### 5.3 The beam flash's partners, data vs MC

Per event, `r1_census.py` on the MC hit-flash arm (all 154 per-file roots) was set against the data arms
(`124_flash_census_data_vs_mc.txt`, `124_flash_census_big_partners.txt`). The table counts beam-window reco1
flashes that have an absorbed or vetoed hit-flash partner of at least X PE:

| partner ≥ X PE | data, per event | MC cv, per event |
|---|---|---|
| 20 | **0.340** | **0.066** |
| 100 | 0.053 | 0.053 |
| 1 000 | 0.020 | 0.027 |
| 5 000 | 0.010 | 0.011 |

- **The fivefold data excess is entirely small pulses of 20–100 PE**, mostly vetoed ones before the beam flash.
  For partners of 100 PE and above, the MC matches the data. For the 1–5 k PE partners that make the Q/L
  changes, the MC has slightly more.
- So **the MC does model the large beam-window partners**. The data's smaller gain is not explained by a missing
  data/MC light feature at that scale.
- The small-pulse excess is a separate data/MC difference. Two candidate causes: the MC photon/noise model, or a
  trigger bias (data spills are light-triggered, the MC is not). A second sign of a light-model difference is
  that the MC makes 2.5× more "dropped" small flashes than data (13.8 vs 5.5 per event).

**Per flip event** (`124_flip_beam_partners.txt`):
- 8 of the 10 QL-main gains have the recovered beam light as a 9–22 k PE partner that reco1 had merged or
  vetoed. The exceptions are 66944 and 165060.
- 4 of the 7 QL-main losses have such a partner: 59003, 169824, 158227 and 157215.
- 390743, 402297 and 174422 have none. They are fit re-balances between flashes both arms had.
- The churn flips have only small partners, or none.

### 5.4 The recommended light gate on all 3000

The mcp2k arm was run exactly as mcp1k was in doc 123 §18:
- `work-mcp2k-d123lgop`, pin `~/tmp/d123-libpin-r6`, `libWireCellMatch.so` md5 d57af13f, library list identical
  at start and end;
- the compiled Q/L config against `work-mcp2k-d123hits` differs only by `xtpc_sc1_light_gate: true` and
  `xtpc_sc1_overpred_max: 2.9` (groups g0, g57, g120 checked), and `rse.json` is identical;
- stage B 2000 of 2000.

Results (`124_pr_mcp2k_hits_lgop.json`):

| mcp2k, `lgop` vs `hits` | |
|---|---|
| νμ > 0.9 | 513 → **515**: 2 gained, 0 lost |
| candidates | 891 → 893, no vertex moved > 5 cm, no νe change |
| the 26 mcp2k flips of this doc | none changes (390743 stays −2.17, 157215 stays without a candidate) |
| new passes | 273559 (289 cm, numu 3.38, 12.4 k PE flash, vertex in the FV), 476900 (121 cm, vertex at y = 193 cm, outside the FV) |

Neither new pass had a candidate in any arm. On an unblinded look (hit-flash vs light-gate display, not part of
the blind scan):
- **273559:** a straight track that starts inside at z ≈ 250 cm and exits the downstream face. Under the gate it
  takes the beam flash, 10.6 k PE predicted against 12.4 k measured; with hit flashes it was unmatched. It is
  νμCC-like.
- **476900:** enters at the top upstream corner. It is entering-track-like and outside the FV anyway.

**On all 3000 data events the light gate moves the MC-definition count 661 → 664.** The three are 59003, 169824
and 273559, and nothing is lost. Hit flashes plus the gate then stand at +9 over reco1 flashes (655). That
closes most of the gap to the MC expectation of §5.2 (+13 to +19).

## 6. The 46 events

"blind scan prefers" is the unblinded verdict: the rows were shown as A/B in a per-event random order
(`124_blind_map.tsv`, written before any figure). Five scan agents saw only the PNGs and the protocol
(`124_scan_protocol.md`). Confidence is 1 (weak) to 3 (clear). "t0 nudge" is §3.3; "—" means not tested (QL-main,
or 5 churn events classed QL-main in the first pass, when the nudge list was drawn). Figures:
`docs/124_figs/<sample>_<event>.png`, with rows in the blinded order. Bee sets are in doc 123 §18.7.

| sample | event | flip | class | νμ reco1 → hits | light gate (lgop) | blind scan prefers (conf) | scan object | t0 nudge | reco1 is row |
|---|---|---|---|---|---|---|---|---|---|
| mcp1k | 59003 | lost | QL-main | 3.20 → none | 3.65 | reco1 (3) | numuCC | — | B |
| mcp1k | 59261 | lost | fragment | 3.85 → 0.15 | 0.15 | hits (1) | unclear | stays reco1 | B |
| mcp1k | 59929 | gained | fragment | -0.41 → 1.90 | 1.90 | hits (1) | unclear | not run | A |
| mcp1k | 60933 | lost | PR-only | 3.75 → 0.83 | 0.83 | reco1 (1) | numuCC | stays reco1 | B |
| mcp1k | 62459 | lost | fragment | 3.30 → 0.77 | 0.77 | reco1 (1) | cosmic | stays reco1 | A |
| mcp1k | 65053 | lost | fragment | 2.88 → -0.31 | -0.31 | unclear (1) | numuCC | not run | B |
| mcp1k | 65999 | lost | fragment | 2.10 → -1.14 | -1.14 | same (1) | unclear | stays reco1 | A |
| mcp1k | 277298 | lost | fragment | 1.28 → 0.89 | 0.89 | hits (1) | unclear | stays reco1 | A |
| mcp1k | 280884 | gained | QL-main | none → 4.40 | 4.40 | hits (2) | numuCC | — | B |
| mcp1k | 281639 | lost | PR-only | 1.37 → 0.60 | 0.60 | same (2) | other-nu | stays reco1 | B |
| mcp1k | 281808 | gained | QL-main | none → 2.96 | 2.96 | hits (2) | numuCC | — | A |
| mcp1k | 284211 | gained | fragment | -0.57 → 2.79 | 2.79 | hits (2) | numuCC | stays reco1 | A |
| mcp1k | 285531 | gained | PR-only | 0.47 → 4.24 | 4.24 | hits (2) | numuCC | follows hits | A |
| mcp1k | 285665 | gained | fragment | 0.55 → 1.03 | 1.03 | hits (1) | unclear | stays reco1 | A |
| mcp1k | 390644 | gained | QL-main | none → 2.83 | 2.83 | hits (3) | numuCC | — | A |
| mcp1k | 68428 | gained | QL-main | none → 2.54 | 2.54 | hits (3) | numuCC | — | A |
| mcp1k | 169824 | lost | QL-main | 5.65 → none | 5.88 | reco1 (3) | numuCC | — | A |
| mcp1k | 170792 | gained | fragment | 0.50 → 0.98 | 0.98 | hits (1) | unclear | stays reco1 | A |
| mcp1k | 172832 | gained | fragment | 0.64 → 3.10 | 3.10 | reco1 (1) | numuCC | stays reco1 | B |
| mcp1k | 174422 | lost | QL-main | 1.40 → none | none | hits (1) | unclear | — | A |
| mcp2k | 53749 | gained | QL-main | none → 1.60 | 1.60 | hits (3) | unclear | — | B |
| mcp2k | 58006 | lost | fragment | 0.99 → 0.74 | 0.74 | hits (1) | unclear | stays reco1 | B |
| mcp2k | 66944 | gained | QL-main | 0.03 → 1.77 | 1.77 | hits (1) | numuCC | — | A |
| mcp2k | 67868 | gained | fragment | 0.47 → 3.25 | 3.25 | hits (2) | numuCC | stays reco1 | B |
| mcp2k | 71642 | lost | PR-only | 2.26 → 0.64 | 0.64 | reco1 (1) | unclear | follows hits | A |
| mcp2k | 77846 | gained | fragment | -0.56 → 3.18 | 3.18 | hits (2) | unclear | stays reco1 | B |
| mcp2k | 105338 | gained | QL-main | none → 1.42 | 1.42 | hits (2) | numuCC | — | A |
| mcp2k | 275385 | gained | PR-only | -0.30 → 4.22 | 4.22 | hits (2) | numuCC | stays reco1 | A |
| mcp2k | 281985 | lost | PR-only | 2.25 → 0.40 | 0.40 | reco1 (2) | numuCC | follows hits | B |
| mcp2k | 293536 | gained | PR-only | -1.07 → 0.98 | 0.98 | hits (1) | unclear | follows hits | B |
| mcp2k | 390743 | lost | QL-main | 1.79 → -2.17 | -2.17 | reco1 (2) | numuCC | — | A |
| mcp2k | 391777 | lost | PR-only | 1.15 → 0.89 | 0.89 | same (3) | unclear | stays reco1 | A |
| mcp2k | 393538 | gained | PR-only | 0.36 → 1.08 | 1.08 | hits (1) | numuCC | stays reco1 | A |
| mcp2k | 396222 | gained | fragment | 0.89 → 0.93 | 0.93 | hits (1) | other-nu | follows hits | A |
| mcp2k | 399963 | gained | QL-main | none → 2.59 | 2.59 | hits (2) | unclear | — | A |
| mcp2k | 402297 | lost | QL-main | 1.28 → none | none | hits (2) | cosmic | — | A |
| mcp2k | 410566 | lost | fragment | 1.05 → 0.78 | 0.78 | same (2) | unclear | follows hits | A |
| mcp2k | 415564 | lost | fragment | 1.88 → -2.29 | -2.29 | reco1 (1) | unclear | not run | A |
| mcp2k | 480970 | lost | fragment | 1.62 → 0.44 | 0.44 | hits (1) | unclear | stays reco1 | B |
| mcp2k | 497399 | lost | PR-only | 2.08 → -2.58 | -2.58 | hits (2) | unclear | not run | A |
| mcp2k | 157215 | lost | QL-main | 1.48 → none | none | reco1 (2) | numuCC | — | A |
| mcp2k | 158227 | lost | QL-main | 3.71 → none | none | hits (2) | unclear | — | A |
| mcp2k | 161725 | lost | fragment | 2.24 → 0.85 | 0.85 | hits (2) | numuCC | stays reco1 | A |
| mcp2k | 163595 | gained | QL-main | none → 4.00 | 4.00 | hits (2) | numuCC | — | B |
| mcp2k | 165060 | gained | QL-main | none → 1.29 | 1.29 | reco1 (3) | cosmic | — | B |
| mcp2k | 169724 | gained | fragment | 0.67 → 1.17 | 1.17 | unclear (1) | unclear | not run | B |

## 7. Follow-ups (not done here)

1. **Adopt the light gate** (doc 123 §18.6): it removes the two clearest real losses and loses nothing on 3000 data events (§5.4: 661 → 664). **Done:** flipped into production on the owner's word, 2026-09-25 (doc 123 §19, `ref/prod-2026-09-25b`).
2. **An invariance audit of the PR stage.** Its verdict depends on things that should not matter: 1–3 specks
   several metres away in the bundle, the cluster numbering, and a 10 ns flash-time shift. The test is a
   relabel/permutation null (renumber the clusters of one pctree, re-run) and a speck-removal null. A PR that
   passed both would turn these 29 flips per 3000 events into zero, a noise term of ±1 % on the selected count.
3. **The data/MC small-pulse difference** (§5.3): 20–100 PE pulses beside the beam flash occur 5× more often in data. The candidate causes are the photon/noise model or the light trigger.
4. **Cross-TPC pairing.** The hit flashes widen the TPC0–TPC1 beam-flash time difference: pairs within 10 ns of
   the 50 ns `flash_pair_dt_us` edge go from 18 to 43 per 3000 events (`124_beam_pair_dt.txt`). No flip traces
   to it, but the window may want re-tuning for hit-flash times.

## 8. Files

- **Scripts:** `scripts/d124/`
  - `flip_anatomy.py`, `mc_flip_truth.py`
  - `pr_null_cmp.py`
  - `t0_nudge.py`, `nudge_cmp.py`
  - `flip_display.py`, `unblind.py`, `scan_calibration.py`
  - `fragment_distance.py`, `beam_dt_census.py`
  - `doc_table.py`
- **Tables:** `docs/124_flips/`
  - anatomy TSVs and detail sheets per sample
  - the null, the nudge and its repack null
  - the scan protocol, verdicts, blind map and unblinded join
  - the census comparison, the pairing census, the fragment distances, the MC truth join
- **Figures:** `docs/124_figs/` (46 data PNGs) and `docs/124_figs_mc/` (30 MC PNGs), both in blinded row order.
  The data map was drawn as one seeded sequence per sample, before the per-event seeding now in `flip_display.py`.
  The committed `124_blind_map*.tsv` files are the record.
- **Arms:**
  - `work-mcp2k-d123lgop` (Q/L, pin `~/tmp/d123-libpin-r6`, knob file `scripts/d123/tla/xtpc_lgop.txt`)
  - `work-mcp2k-d123lgoppr` (PR) and `products/d123/mcp2k_lgop`
  - scratch under `~/tmp/d124/` (null and nudge roots)

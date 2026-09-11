# 86 — Where is the stopper's Michel? Doc 78 action item 6, the hand check (`smx5`)

Doc 78 §5 item 6: *"Hand-check, no code. `039349_43/66`, `039349_72/11`, `039253_0/44`, `039349_30/45`: no fitted charge near the stop; the Michel is unimaged or in a dead region — doc 96's territory."*

**Status (2026-09-11): SCANNED and SCORED; item 6 is closed and two new action items follow (§9).** The owner judged all four: **4 of 4 STM + MICHEL (attached)**, and their words land on the predicted mechanism on 4 of 4 (§7). The first two are `fit-through`: the Michel is collinear with the muon, and the fit runs from the Bragg peak into a short segment after it. The last two are `unfitted`: a clear blob at the stop that PR never made a segment of. None is unimaged, and item 6's "doc 96 territory" is withdrawn. Folding the answers into a new merged record (`…smx1a_smx3_smx4_smx5_verdicts.json`) moves no census number. §8 sizes both mechanisms over the whole record; §9 names what to build next. Nothing was built: no toolkit change, no config change, no new arm. The input is doc 85's production-equivalent prep (`p85vwh`, identical to the flipped production arm `p85vprod` on 120/120 zips and 596 candidates × 145 branches, doc 85 §8).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts; cd $IMG
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
PREP=/home/xqian/tmp/p85/prep_p85vwh          # doc 85's production-equivalent prep
# sec 2: the pre-look
python3 $X/d86_offfit.py --prep $PREP > /home/xqian/tmp/p86/offfit.txt
# sec 3-4: the scan set (refuses to rebuild an existing set; --force-questions rewrites the wording only)
python3 $X/d86_build_smx5.py --prep $PREP --outprep pdhd/stm_michel_scan/prep-pdvd-smx5 \
    --sheet pdvd/docs/scan/pdvd_stm_michel_smx5_sheet.tsv \
    --questions pdvd/docs/scan/pdvd_stm_michel_smx5_questions.json \
    --key pdvd/docs/scan/pdvd_stm_michel_smx5_key.tsv
# served on :5017 (labels -> pdvd/work/stm_michel_labels/smx5/labels.json)
(cd pdhd/stm_michel_scan && nohup ./serve_stm_michel_scan.sh 5017 --det pdvd --scan-tag smx5 \
    --manifest $IMG/pdvd/docs/scan/pdvd_stm_michel_smx5_sheet.tsv --prepdir $PWD/prep-pdvd-smx5 \
    --questions $IMG/pdvd/docs/scan/pdvd_stm_michel_smx5_questions.json > /home/xqian/tmp/p86/serve_5017.log 2>&1 &)
# sec 7: the owner's answers, scored against the key, folded into a NEW merged record
cp -p pdvd/work/stm_michel_labels/smx5/labels.json pdvd/docs/scan/pdvd_stm_michel_smx5_labels.json
python3 $X/d86_score_smx5.py --labels pdvd/docs/scan/pdvd_stm_michel_smx5_labels.json \
    --key pdvd/docs/scan/pdvd_stm_michel_smx5_key.tsv \
    --write-merged pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json   # -> /home/xqian/tmp/p86/score_smx5.txt
# the census is the same on both records (diff empty), and the frozen check still holds
for r in smx1a_smx3_smx4 smx1a_smx3_smx4_smx5; do (cd pdhd/stm_michel_scan && \
    STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_${r}_verdicts.json python3 census_score.py \
    --prep $PREP --arm p85vwh --baseline $PREP > /home/xqian/tmp/p86/census_$r.txt); done
(cd pdhd/stm_michel_scan && python3 census_score.py --check)          # "0 of 14 differ"
# sec 8: both mechanisms over the whole record (U, U2, R, A)
STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json \
    python3 $X/d86_sizing.py --prep $PREP --arm p85vwh > /home/xqian/tmp/p86/sizing_smx5rec.txt
STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json \
    python3 $X/d62_probe.py p85vwh $PREP > /home/xqian/tmp/p86/probe62_p85vwh.txt   # doc 62's own probe, re-run
```

`d86_offfit.py`, `d86_build_smx5.py` and `d86_score_smx5.py` assert the smx4 record, the one they were run against. **From this doc on, the current record is `pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json`**; set `STM_SCAN_RECORD` to it. It differs from the smx4 record only on the four rows of §7.2.

Before the server started, every existing PDVD label file was copied to `/home/xqian/tmp/p86/label_backup_20260911_094116/` with checksums (`…094116.md5`: `smx1` `0fb74040`, `smx1a` `6c03c4cc`, `smx3` `52edf4e0`, `smx4` `2a026c1e`). `smx5` is a new tag; no existing label directory is written.

## 1. Item 6's premise does not hold on three of the four

| item 6 says | the production prep says (§2) |
|---|---|
| no fitted charge near the stop | the chain's own muon fit ends at the stop on all four. What none of them has is a **second** PR segment: the nearest other segment of the cluster is 26.9, 93.7, — (none), 142.5 cm away |
| the Michel is unimaged | within 10 cm of the fit end there are 55 / 73 / 148 / 142 imaged points, **all in the muon's own cluster** (in bundle). Within 20 cm the only other cluster is `c210` on `039349_72/11`: 9 points, out of bundle, i.e. a different drift time |
| or in a dead region | one item, `039253_0/44`: a V dead band (channels 4360–4367) 1.8 channels from the fit end. `039349_43/66` has two dead V channels 1 channel off the end and sits 0.44 cm from the y = 168.5 CRU seam, with the owner's smx3 pin across it. The other two have nothing dead within 7 channels |

All four are found stoppers with a failed Michel search (doc 77's class (c)): production reads `is_stm` 1 on all four (`039253_0/44` since doc 75; `039349_43/66` and `039349_72/11` since doc 68's peak anchor) and `michel_found` 0. The Michel object is the one open quantity.

What discriminates the readings is where the imaged charge sits relative to the fit. Within 10 cm of the fit end:

- **On-fit** (`039349_43/66`, `039349_72/11`): not one point lies more than 2 cm from the fit, the same as a body stretch 30–40 cm upstream (max 1.5–1.7 cm). Nothing is imaged past the fit end. If there is a Michel, either the fit runs through it, or it is not in the image.
- **Off-fit** (`039253_0/44`, `039349_30/45`): 82 and 50 points lie more than 2 cm off the fit (1.1 M e and 0.6 M e), and on `039349_30/45` 19 points lie past the fit end. The charge is imaged in the muon's cluster, and no PR segment follows it.

So at most one item (`039253_0/44`) could be doc 96's territory. The hand check decides which of these readings is right, item by item.

## 2. The pre-look (`d86_offfit.py`, `/home/xqian/tmp/p86/offfit.txt`)

| item | own points ≤ 10 cm of the fit end, by distance off the fit (0–1 / 1–2 / 2–3 / 3–5 / > 5 cm) | past the fit end | dQ/dx along the last rows (k e/cm; plateau) | dead channels within 15 of the fit end | seam |
|---|---|---|---|---|---|
| `039349_43/66` | 41 / 14 / 0 / 0 / 0 | 0 | from the owner's pin (rr 4.8): 98 65 91 63 **19** 33 54 67 (54) | V 4978–4979, 1 ch | **y 168.5, 0.44 cm**; the pin is across it (CRU 5, the fit end CRU 4) |
| `039349_72/11` | 54 / 19 / 0 / 0 / 0 | 0 | from the owner's pin (rr 3.0): 77 69 **40 41 38 31** (60) | U 293 (7 ch), W 8171 (11 ch) | 20.3 cm |
| `039253_0/44` | 30 / 36 / 16 / 36 / 30 | 8 (2 off-fit) | last 5 cm: 87 95 102 112 121 128 131 127 68 (72) | **V 4360–4367, 1.8 ch**; U 580 (6.8 ch) | 65.2 cm |
| `039349_30/45` | 55 / 37 / 20 / 26 / 4 | 19 (15 off-fit) | last 5 cm: 85 68 35 **9 7** 21 39 (49) | none | 11.7 cm |

The method's check against a prior reading: the smx1a note on `039349_30/45` describes "a diffuse fan of about thirty image points springing off the stop"; the pre-look counts 50 points more than 2 cm off the fit within 10 cm, 15 of them past the end.

The fit end's U/V/W coordinates and T_bad_ch come from the same plane-rank scheme (`PdvdPrMagnifyTrackingVisitor` `ChanScheme`), so the dead-channel column needs no channel map.

## 3. Predictions (written before the scan; the key `pdvd_stm_michel_smx5_key.tsv`)

| item | predicted MECH | predicted DEAD | why |
|---|---|---|---|
| `039349_43/66` | `fit-through` | V? — and **seam** (addendum below) | on-fit; past the owner's pin the fit dips to 0.36 plateau 2 cm before its end, then ends on a second deposit |
| `039349_72/11` | `fit-through` | none | on-fit; a drop to 0.5–0.7 plateau over the last 2 cm, no recovery. The weakest prediction: the Michel may simply not be imaged |
| `039253_0/44` | `unfitted` | V | off-fit beside the last few cm, one PR segment in the cluster; the V dead band abuts the end. **Caveat:** in a dead V band imaging runs on two planes, and off-axis points there may be two-plane artefacts rather than a Michel |
| `039349_30/45` | `unfitted` | none | off-fit, including past the end; the fit collapses to 0.14–0.19 plateau and ends on a second rise. The verdict itself is open (smx1a, medium) |

*Addendum, still before the scan (09:45):* the display's stopping-point badge put `039349_43/66`'s fit end 0.44 cm from the y = 168.5 seam, and the owner's smx3 pin is 1.7 cm on the far side of it. The dip to 19 k may be the seam's charge loss rather than a gap before a Michel. The key (written at 09:41) is not rewritten. This addendum is its correction, and the question text gained the `seam` option (`--force-questions`; the prep, sheet and key checksums are unchanged).

## 4. What the owner judges

The display is `stm_michel_viewer.py` on :5017, tag `smx5`, 4 items in doc 78's order. Each item's panel states production's reading (`is_stm` 1, `michel_found` 0, no other segment within 20 cm) and says that the stopper call is not what is asked. On the three items the owner judged in smx3, the panel repeats their smx3 verdict, pin and gamma tags, and draws the smx3 pin as the green dot-dash line in the dQ/dx panel. **`039349_30/45` is blind:** its panel carries none of the smx1a reading.

Per item:

1. **Where is the Michel?** The first line of *notes* is `MECH: <word>; DEAD: <U/V/W/seam or none>;`, typed before the verdict click. MECH is one of:
   - `fit-through`: the Michel is the muon fit's own last few cm;
   - `unfitted`: imaged beside or past the fit end, in this cluster, with no segment;
   - `other-cluster`;
   - `dead-region`: a dead band or a seam;
   - `not-imaged`;
   - `no-michel`;
   - `unclear`.
2. **The pin**, at the muon's true end.
3. **The verdict**, which is what saves the row. STM + MICHEL needs the Michel radio (attached / detached dots / both). `039349_30/45`'s verdict is the one that is genuinely open.

**What the display shows, and what it cannot.** It shows:
- the 3-D imaged points, coloured by cluster, with the bundle filter;
- the fit;
- the particle-flow table;
- the measurement tab (measured / predicted / difference in U, V, W for the cluster's own cells, with the T_bad_ch dead bands);
- the seam distance on the badge.

It **cannot** show charge that imaging never turned into a blob of this cluster. The measurement tab is `T_proj_data`, i.e. the cluster's own cells, not the decon frames. So `not-imaged` is a suspicion the display can raise but not prove. The panel says so, and promises a decon-level view for any item answered that way. The DNN-ROI frames for all four events are in `/home/xqian/pdvd-frame-store/<evt>_keep/`. A view built from them must map channel ids to the plane-rank scheme, with the muon track lighting up along its own `pu/pt` as the positive control. It is not built until an answer asks for it.

## 5. What each answer leads to

| answer | layer | the next step it implies |
|---|---|---|
| `fit-through` | the stop: the fit runs past the true stop into the Michel | the overshoot family of doc 78 §2, i.e. docs 57/58/74/82's movers, which do not fire here. Read each item's retreat/split tests at the owner's pin, as doc 82 §2 did for the 21 |
| `unfitted` | PR: imaged charge that no segment follows | doc 62's territory (Steiner terminals / no trajectory). Size it as "own-cluster charge off the fit near a stop" over the whole record before any code |
| `dead-region` / seam | readout gap or imaging | doc 96's territory, as item 6 guessed. On a seam: does the far CRU's image carry the Michel? |
| `not-imaged` | imaging | build the decon view for that item (§4), then decide |
| `no-michel` | the record | a record correction (verdict → STM_ONLY). The chain was right |

## 6. After the scan

1. Copy `pdvd/work/stm_michel_labels/smx5/labels.json` to `pdvd/docs/scan/pdvd_stm_michel_smx5_labels.json`. The label directory is not tracked.
2. Read each row's `MECH` / `DEAD` against the key and the pin against the smx3 pin.
3. Fold any verdict change into the merged record under a new name. Never over the old one (M13).
4. Write the result into §7 here and item 6 of doc 78.

## 7. Results

### 7.1 The owner's answers against the key (`d86_score_smx5.py`, `/home/xqian/tmp/p86/score_smx5.txt`)

The owner answered in the notes and in the message that returned the scan: *"note the first two cases, the Michel seems to be aligned with the STM, there is a clear Bragg peak, and then a short one after it that could be the Michel electron. For the latter two events, there are some clear blob topology near the end of the stopping point of the muon. Those should be Michel electrons, but not identified by the PR chain at all. These should be improved."* They did not type the panel's `MECH:` line, so the MECH column is **my mapping of those words**, quoted beside it. DEAD was not answered, and it is not scored.

| item | verdict (record → smx5) | Michel | predicted MECH | owner, mapped | the owner's note |
|---|---|---|---|---|---|
| `039349_43/66` | STM_MICHEL → STM_MICHEL | attached | `fit-through` | `fit-through` ✓ | "Michel may be aligned with muon direction?" |
| `039349_72/11` | STM_MICHEL → STM_MICHEL | attached | `fit-through` | `fit-through` ✓ | "Michel is aligned with muon track, still a bit segment after dQ/dx" |
| `039253_0/44` | STM_MICHEL → STM_MICHEL | attached | `unfitted` | `unfitted` ✓ | "There is clearly some segment (not identified) at the end of STM" |
| `039349_30/45` | STM_MICHEL (smx1a, medium) → STM_MICHEL (owner) | attached | `unfitted` | `unfitted` ✓ | "There is also an reconstructed pieces near the end point, that should be the Michel." |

**4 of 4 on the mechanism, 4 of 4 STM + MICHEL.** The blind item, `039349_30/45`, is now owner-confirmed. The smx1a scanner's medium call rested on the collapse plus a fan with no separate charge blob; the owner sees the blob. On `039253_0/44` the owner saw a segment, not a two-plane artefact of the V dead band. That was §3's caveat, and it is answered.

### 7.2 The fold (`pdvd_stm_michel_smx1a_smx3_smx4_smx5_verdicts.json`, 601 rows)

The rows are smx4's `owner_row` (doc 70) with one belt: an unplaced smx5 pin never replaces a placed one. That is the viewer's own no-downgrade rule. The owner re-clicked the verdict on `039349_43/66` and `039349_72/11` without moving the pin, and their smx3 pins sit at the junction they describe ("a clear Bragg peak, and then a short one after it").

| item | what changes |
|---|---|
| `039349_43/66` | `michel_kind` both → attached; the smx3 pin (rr 4.8) kept |
| `039349_72/11` | nothing but the source; the smx3 pin (rr 3.0) kept |
| `039253_0/44` | `michel_kind` both → attached |
| `039349_30/45` | `michel_kind` none → attached, confidence medium → owner; smx1a's `pin_rr` 3.8 carried as `pin_rr` with `pin_rr_source` smx1a |

The rows also carry a new `mech` field, which is **my mapping of free text, not a typed answer**. Stopper, Michel and judged class are identical on all 601 rows, so every census number is identical: `census_score.py` on `p85vwh` gives the same output on both records (diff empty). That is `is_stm` FP 7 / FN 45 and `michel_found` FP 12 / FN 19. `--check` is 0 of 14.

## 8. The two mechanisms over the whole record (`d86_sizing.py`, `/home/xqian/tmp/p86/sizing_smx5rec.txt`)

546 judged items have a payload on `p85vwh`. The cells are the record's class × `michel_found` × `is_stm`. **19 missed Michels** are STM_MICHEL with `michel_found` 0: 9 are `is_stm` 1, 10 are `is_stm` 0.

### 8.1 Unfitted charge near the stop (items 3–4's mechanism)

**A raw count does not discriminate.** Counting own-cluster points more than 2 cm off every fit within 10 cm of the fit end, the population's halo is wider than the four items suggested. Around a body stretch 30–40 cm upstream, the maximum off-fit distance is p50 1.74, p90 4.01 and p99 4.97 cm. At ≥ 10 such points, 34 of 261 `is_stm`-0 THRU items fire.

**Subtracting each item's own body rate** (U2: the same count in three 10 cm spheres on the fit 25/35/45 cm upstream, the maximum subtracted) at excess ≥ 10:

| cell | fires |
|---|---:|
| STM_MICHEL, not found, `is_stm` 1 | **4 / 9** (`039253_0/44` 82, `039349_30/45` 50, `039253_3/61` 36, `039349_58/69` 15) |
| STM_MICHEL, not found, `is_stm` 0 | 4 / 9 |
| STM_MICHEL, found, `is_stm` 1 | 36 / 120 |
| STM_ONLY, `is_stm` 1 | 9 / 93 (`039253_15/36` 77, `039349_12/44` 41, `039349_66/78` 25, `039349_82/60` 18, …) |
| THRU, `is_stm` 1 / 0 | 0 / 6, 7 / 256 |

As a Michel source on its own, the image count buys 4 Michels for 9 STM_ONLY items. That is not a rule.

**What PR does with the blob (R).** On both owner items, PR fits a residual at the stop and then drops it as isolated (`pr54 isolated-residual drop`, DEBUG, in today's production logs):
- `039253_0/44`: 20 points and 10.6 cm, with an endpoint 3.1 cm from the stop;
- `039349_30/45`: 6 points and 8.5 cm at 1.4 cm, plus a 4-point, 5.5 cm residual at 1.7 cm.

That is doc 62 T3a's object. `stop_local_residual_cm` keeps such a residual, then doc 62 T3b (`stop_local_michel_pieces`, in production) admits it into the Michel, and the knob exists, default OFF. Doc 62 left it OFF at 20 cm, and part of that judgement no longer holds: it counted `039253_0/44` among its "five extra michel FPs" because the record then read STM_ONLY, but the owner now says STM_MICHEL twice. Re-counted on today's production:

| keep rule | items it touches | by cell |
|---|---:|---|
| doc 62's T3a: 20 cm, no size floor | 31 | 2 targets, 1 missed Michel on an `is_stm`-0 item, 11 found Michels, 3 STM_ONLY, **14 THRU** |
| 5 cm, no floor (config only) | 11 | 2 targets, 5 found Michels (incl. `039349_61/21`, doc 62's lost TP), 1 STM_ONLY, **3 THRU** (short stubs: 0.8, 1.5 and 6.8 cm with 3 points) |
| **5 cm, ≥ 5 points and ≥ 5 cm** (needs C++) | **5** | **2 targets** (`039253_0/44`, `039349_30/45`), 1 STM_ONLY (`039253_15/36`, 12 points, 8.8 cm), 2 found Michels (`039349_36/63`, `039349_37/39`), **0 THRU** |

A kept residual changes the PR graph that later stages build, and doc 62 §4.3 saw that move verdicts both ways. So the gated keep is **5 items whose graph changes**, not "safe": 2 targets, 1 record-FP, 2 items whose Michel is already found, 0 through-going. The floors are new code: the anchor keep has no size floor today (`NeutrinoOtherSegments.cxx:868`). The distance is measured from the payload's fit end here; the knob anchors on the tagger's stop, which is close but not identical, so the arm is the measurement.

### 8.2 The collinear Michel inside the fit's tail (items 1–2's mechanism)

The production verdict already locates the Bragg peak. `bragg_anchor_shift_cm` (doc 68) is the length of fit past the anchored peak: 2.80 cm on `039349_43/66`, 2.77 on `039349_72/11`, against the owner's smx3 pins at 4.8 and 3.0. The Michel search still starts from the fit end (doc 78 §2.1). The shift is bounded by the 3 cm anchor search window, so no item reads ≥ 3.

A sketch rule applies after the verdict and only to Michels, so `is_stm` cannot move by construction: `is_stm` 1, `michel_found` 0, not the geometric fallback, shift ≥ S, and the tail's median dQ/dx ≤ T × plateau.

| S, T | targets (STM_MICHEL, not found) | STM_ONLY | THRU (`is_stm`-1 FPs already) |
|---|---|---|---|
| 1.5 cm, 1.0 | 5 | 6 | 3 |
| **2.5 cm, 1.0** | **5**: `039253_3/61`, `039349_30/45`, `039349_43/66`, `039349_58/69`, `039349_72/11` | **3**: `039252_3/75`, `039349_47/68`, `039349_82/60` | **2**: `039349_14/22`, `039349_30/21` |

That is 5 gained Michels for 5 spurious ones, an added-set purity of 0.5. It would take `michel_found` from 141 / 12 / 19 to 146 / 17 / 14, and purity **falls**. **The rule does not meet the campaign bar.** A tighter charge cut will not fix it. Two of the three STM_ONLY fires show the same dip-then-rise as the owner's items (`039349_47/68`: … 1.81 0.35 0.60 1.18; `039349_82/60`: … 0.31 0.24 0.78 0.99, relative to the plateau), which is the owner's own kink discriminator again: charge shape alone does not name the particle. All three STM_ONLY calls are the smx1a scanner's at medium confidence. A collinear Michel inside the fit is exactly what a scanner reading the profile would call STM_ONLY. Whether they are missed Michels decides the rule: 8 / 2 if they are, dead if they are not.

## 9. The plan: item 6 closes, items 7 and 8 open (doc 78 §5)

- **Item 7, the unfitted Michel blob at the stop (owner: "should be improved").** *Done in doc 87:* built and OFF. At 5 cm with 5 / 5 the keep gains both owner Michels (Michel census 143 / 13 / 17) with 0 THRU, but `039349_36/63` loses its stopper call when the stop snaps onto the kept residual. Item 7b is next.
  - **Build:** a size floor on doc 62 T3a's anchor keep, default OFF: `stop_local_residual_min_points` 5 and `stop_local_residual_min_len_cm` 5. Run it with `stop_local_residual_cm` 5.
  - **Arms:**
    - OFF gate on both detectors;
    - `stop_local_residual_cm` 5 alone (config only) and 20, re-grading doc 62 on today's production and record;
    - the floored keep.
  - **Predicted:** +2 Michels (`039253_0/44`, `039349_30/45`), 0 THRU touched, `039253_15/36` a Michel FP unless re-judged, and graph changes confined to 5 items. Flip only on 0 `is_stm` change outside those five and 0 lost TP.
  - **Recommended first:** the owner named it, it rests on PR's own object, and its footprint is five items.
- **Item 8, the collinear Michel in the fit's tail.** A post-verdict "anchor tail" Michel (`is_stm` cannot move).
  - **Gate before any build:** a small blind re-judge of the rule's 10 fires (the 5 targets, the 3 STM_ONLY, the 2 THRU), asking whether a collinear Michel follows the Bragg peak.
  - **Outcome:** build if the 3 STM_ONLY come back as Michels (8 / 2); drop it if they don't.
  - The same scan can carry `039253_15/36`, item 7's one record-FP, whose 77-point blob at an "STM_ONLY" stop is the largest in the set.

## 10. Observations

1. **Unfitted charge also sits beside found Michels.** 36 of 120 found stoppers carry an excess of ≥ 10 unfitted own-cluster points near the stop (U2), so the Michel **energy** is under-counted on about a third of the found population. That is a larger lever than items 7 and 8, and it is doc 81's estimator (built, OFF). It is recorded, not scoped here.
2. **Doc 62 T3a's grading is stale.** `039253_0/44` was one of its "five extra michel FPs" and is an owner-confirmed Michel, and the production it was graded on (`d61v`) predates docs 63–85. Item 7 re-grades it.
3. **The item 6 hypothesis (unimaged, dead region) came from the absence of a second PR segment.** It was not a look at the image. On all four items the 3-D image carries the charge, in the muon's own cluster.
4. `:5017` still serves `smx5`. The labels are copied into `pdvd/docs/scan/pdvd_stm_michel_smx5_labels.json`, and the four other PDVD label files are unchanged against the pre-scan backup (`cmp`).

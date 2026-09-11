# 78 — The owner's three observations, sized on the record, and the action items they lead to

The owner's question (2026-09-10, after doc 77): *"STM end point not found properly. Michel not identified in the close-to-ISO case (missing track trajectory). Michel or photon clustering … not associated segments, how to deal with them?"* — taking doc 77 into account, what concrete set of actions follows?

**Status (2026-09-10): the sizing is done, read-only, on the production arm; the action items are §5, ordered in §6.** Nothing is built in this doc. Doc 79 (the candidate cap, doc 77 §7.1) and doc 80 (action item 1, the segment census) follow it.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p78
# the production baseline prep (doc 75's confirmation arm p75vprod, toolkit 567a7232 + cfg 50af7a70, pin 02557b8d)
python3 $X/d78_size_obs.py --prep /home/xqian/tmp/p75/prep_p75vprod > /home/xqian/tmp/p78/size_obs.txt
```

Sections A/B/C of the output are §2/§3/§4 here. The record is `pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json` (601 items, 576 judged, 568 with a payload on this arm). The chain's per-candidate output is read from the prep payloads: the record's tag keys are PR segment ids; a tag "has a row" when the chain wrote that segment into `T_stm_michel_pts` with a role (1 muon, 2 delta, 3 Michel, 4 dot, 5 capture gamma, 6 survey, 7 gamma-collect); `pf.seg` is `T_rec_charge`, the PR's own segment table for the cluster. `near_clusters` are the clusters with an image point within 60 cm of the stop, with the Q-L bundle and `is_associated` flags from `T_cluster` (doc 53).

## 1. The three observations are two mechanisms and one non-problem

| observation | population on the record | what production does |
|---|---|---|
| the stop overshoots the true end (§2) | **25** items where the scanner put the stop 1.8–11.1 cm back along the fit (`pin_rr`; 23 STM_MICHEL, 2 STM_ONLY) | the retreat / split (docs 57, 58, 74) moved **4** of them, to within 0.1–1.2 cm of the pin. On the other **21** no mover fired: 8 are `is_stm` 0, 13 have no Michel |
| the Michel is not identified (§3) | **22** STM_MICHEL items with `michel_found` 0 | **12** are the stop's (the pin_rr items above, plus one whose Michel is the muon's own last segment). **5** have a PR segment in the muon's own cluster that the chain never saw. **1** is a delta at an interior vertex. **4** have nothing fitted near the stop |
| "not associated segments" (§4) | **0** of 269 michel tags and **0** of 308 gamma tags land in an unassociated cluster | the scanner's only whole-cluster tags on unassociated clusters are "delta / other" (11) and "muon" (1). The real uncollected population is *same-bundle, associated*: 161 gamma tags on 87 items with no chain role — 54 of those items have no Michel object, and the gamma collect requires one by construction |

## 2. The stop (§A of the output)

### 2.1 What the record holds

Eleven placed pins carry a `moved_cm` (the pin moved 3–10 cm on six STM_MICHEL items, 29 cm on one THRU with a spurious Michel); the richer record is the **25 `pin_rr` entries** of smx3 / smx4 — every note reads "OVERSHOOT: the fit runs N cm past the muon's real stop", with the Bragg peak at rr ≈ pin and a tail after it. The 3 cm peak anchor (doc 68) reads the verdict past that tail; the *stop* the Michel search starts from is still the fit's end.

### 2.2 Production's movers on the 25

| | n | items |
|---|---:|---|
| retreat / split fired, and lands on the pin | 4 | `039252_2/39` (11.0 vs pin 11.1), `039349_10/58` (10.9 vs 9.7), `039252_16/32` (3.3 vs 3.3), `039349_36/63` (1.8 vs 1.8) |
| no mover fired | 21 | `is_stm` 0 on 8 (`039349_51/29`, `81/44`, `71/37`, `81/25`, `34/48`, `039252_9/101`, `039349_44/28`, `039253_2/13`), `michel_found` 0 on 13 |

### 2.3 Why the 21 survive: the movers' own tests, read offline at the scanner's pin

Production's collapse test (retreat and split alike) needs the tail past the stop to read below **0.5 × plateau**; the split additionally needs the fit to bend ≥ 15° at the row, the retreat an existing PR vertex there. At the pin of each of the 21:

| group | n | items | reading |
|---|---:|---|---|
| **tail NOT collapsed** | **14** | `039253_17/77` 1.21, `039349_18/36` 1.66, `039253_7/80` 1.08, `039349_51/29` 0.68, `039349_81/44` 0.89, `039349_52/36` 1.54, `039349_71/37` 0.56, `039349_60/40` 0.95, `039349_67/78` 1.87, `039349_44/28` 0.70, `039253_3/29` 0.81, `039253_2/13` 0.59, `039349_58/69` 0.88, `039252_15/81` 0.54 (tail median / plateau) | the fit rides **through the Michel**: after a Bragg peak of 1.5–2.9 × plateau the tail falls back to 0.54–1.87 × plateau — far below what a post-Bragg muon carries, not below the track's own plateau (the scanner's own words on `60/40`). **11 of the 14 bend by 14–33° at the pin**; the three straight ones (`17/77`, `18/36`, `52/36`: 3–6°) are `is_stm` 1 with a found Michel |
| collapsed, no vertex, bend < 15° | 4 | `039349_81/25` (12°), `039349_7/4` (10°), `039349_34/48` (9°), `039252_9/101` (13°) | the straight gap-bridge: tail 0.12–0.49, peak 1.5–2.4, no PR vertex within 1.5 cm, bend under the split's 15° |
| the split's tests pass offline, it did not fire | 2 | `039349_30/45` (tail 0.41, peak 1.60, bend 55°), `039252_16/110` (0.23, 1.80, 28°) | to be read from the log (the split acts on the chain's last segment only, `stop_split_max` 1) |
| no peak | 1 | `039253_3/61` (peak 1.29 < 1.4; tail 0.07; a vertex at 0.0 cm; bend 52°) | the retreat is blocked by `retreat_peak_frac` alone |

*Correction (doc 82 §2.1, 2026-09-10):* this table reads the movers' tests at the pin, but on **14 of the 21** production never reached them. `CheckSTM_Michel` runs neither the retreat nor the split on a chain that is already Bragg-confirmed (`:2270-2273`, `:2328-2332`) unless `michel_collinear_split` (doc 74's P3 knob) is on, and it is OFF in production; 14 of these 21 are Bragg-confirmed. The uncollapsed tail is a symptom of the same overshoot, not the gate that blocked the move. The same correction retires the `retreat_peak_frac` suggestion in item 2 below: `039253_3/61` is Bragg-confirmed, so its retreat is never tried and the peak fraction is not what stops it.

The discriminator production lacks is the **drop after the peak**, judged against the peak rather than the plateau, together with the bend. A muon cannot follow a 1.9 × plateau peak with a 0.95 × plateau tail; a Michel electron does exactly that. The named confounder is a **delta ray at the end of a through-going track** (a spike, then a return to plateau) — the 284 THRU items are the negative control, and the bend is what separates the two on this record.

### 2.4 The Michel carried as the muon

Eight record michel tags are segments the chain carries as **role 1** (the muon's own trajectory): `039253_3/61`, `039253_7/80`, `039349_44/28`, `039349_52/36`, `039349_60/40`, `039349_61/21`, `039349_64/65`, `039349_81/44`. Six of the eight are pin_rr items; doc 74's strict tail retreat reached two others of this kind and left these.

## 3. The Michel (§B)

### 3.1 The 22 missed Michels, by where the tagged segment lives

| class | n | items | mechanism |
|---|---:|---|---|
| the stop's: a pin_rr item, or the Michel is role 1 | 12 | `039252_9/101`, `039253_2/13`, `039253_3/61`, `039349_30/45`, `039349_44/28`, `039349_51/29`, `039349_58/69`, `039349_60/40`, `039349_71/37`, `039349_81/25`, `039349_81/44`; `039349_64/65` | §2. On `039253_3/61` and `039349_32/63` a piece *was* found (9.3 MeV / 7.1 cm; 4.0 MeV / 5.7 cm) and demoted by the range-energy guard (doc 62 T3c: > 5 cm from the stop, < 10 MeV) — the gap (7.3, 5.5 cm) is measured from the **overshot** stop |
| an orphan PR segment of the main cluster | 5 | `039349_64/24`, `039349_64/52`, `039349_69/56`, `039349_9/19`, `039349_32/63` | the record's michel tag is a PR-fitted segment of the muon's own cluster that has **no row at all** in the chain's output (checked in `T_stm_michel_pts`: the chain wrote only the muon and, where present, its claimed pieces) |
| a delta at an interior vertex (role 2) | 1 | `039252_8/102` | the Michel is in the chain, classified as a delta, because the stop is downstream of it |
| nothing fitted near the stop, no michel tag | 4 | `039252_2/79`, `039349_43/66`, `039349_72/11`, `039253_0/44` | `039252_2/79` is different: the split fired and an attached 8.9 MeV, 9.3 cm Michel at a **59.58°** kink exists; the moved-stop guard (doc 61 T2c) demotes it because the kink exemption (doc 72 P3b) sits at **60°**. Doc 61 counted this item as THRU; the owner's re-judge says STM_MICHEL. The other three (`43/66`, `0/44` owner-judged) have gamma tags at 27–50 cm and nothing at the stop |

### 3.2 The orphan segments

Across all judged stoppers, **42 michel tags on 32 items** are PR segments of the main cluster with no chain row: 21 touch the stop (nearest point ≤ 0.5 cm), 8 sit 3–10 cm away, 5 further, 2 at 0.5–3 cm, and 6 were never fitted by the PR at all. The 21 that touch the stop on items with a *found* Michel are Michel members the object did not absorb (energy under-counted). On the five michel_found-0 items, doc 62 T3b's three gates (dot radius 15 cm, max length 25 cm, body exclusion 5 cm) read offline:

| item | segment | length / dQ/dx median / d_stop / d_body | direction vs the muon's last 5 cm | reading |
|---|---|---|---:|---|
| `039349_64/24` | 24003 | 14.4 cm / 52 k / 2.5 / **2.4** | cos **−0.84** | a MIP-like 14 cm piece running **back along the muon** from 2.5 cm off the stop: inside the body exclusion by design |
| `039349_9/19` | 19003 | 8.0 / 41 k / 3.3 / **1.2** | −0.84 | same |
| `039349_64/52` | 52018 (+ 52016/17/19/20) | 7.8 / 26 k / 5.5 / **0.0**; four more pieces 5.8–10.8 cm at 11–13 cm | −0.75 … −0.97 | a five-piece backward shower, one piece on the body, four free at 11–13 cm |
| `039349_69/56` | 56006 | 13.6 / 13 k / 4.7 / **0.0** | −0.12 | on the body |
| `039349_32/63`, `039253_3/61` | 63006, 61007 | 8.6 and 7.8 cm, **touching the stop** | +0.14, +0.82 | attached at the stop, not taken by the walk (and each item's *other* piece is T3c-demoted from the overshot stop) |

So the "close-to-ISO, missing trajectory" case is, on this record, mostly a Michel that **goes backward along the muon body** — the one geometry the body exclusion was written to reject (a muon's own residuals look the same). Nothing in the output today lets a scanner or a script see these segments at all.

*Correction (doc 80 §5.1, 2026-09-10):* the census knob's own reading of these four segments is **rej 13, attached to the muon chain** (an endpoint vertex on the chain or an out-edge to a chain segment), not the body test: T3b admits only *disconnected* pieces and never examined them. The offline `d_body < d_stop` reading above was right about the geometry and wrong about the gate. Action item 4 below is therefore an attached-arm rule, not a body-exclusion exception; the body test does bite on the four free pieces of `039349_64/52`'s shower at 11–13 cm.

### 3.3 The record's "detached dots" Michels, as the chain reads them

Of the 68 STM_MICHEL items the scanner called "both" (attached + detached) the chain finds 58 (49 attached, 9 bridged) and misses 10 (5 of them `is_stm` 0); of the 4 "detached dots" it finds 2 (one attached, one bridged). A bridged reading is not the failure mode; the misses are the stop's and the orphans'.

## 4. "Not associated segments" (§C)

### 4.1 Where the record's tags land

| tag | n | chain row | main cluster, no row | same-bundle near cluster, no role | other bundle / unassociated |
|---|---:|---|---|---|---|
| michel | 269 | 222 (role 3: 211, role 1: 8, role 2: 2, role 4: 1) | 42 (36 PR-fitted, 6 never fitted) | 5 | **0** |
| gamma | 308 | 143 (role 4: 83, role 5: 39, role 3: 21) | 2 | 161 (+ 2 whole-cluster) | **0** |

Whole-cluster tags (the scanner's only handle on an unfitted piece): 11 "delta / other" and 1 "muon" on unassociated in-bundle clusters, 5 "delta / other" and 2 "gamma" on associated ones.

### 4.2 The clusters around a stop

| within | bundle | associated | PR-fitted | clusters | items |
|---|---|---|---|---:|---:|
| ≤ 20 cm | same | yes | yes | 125 | 88 |
| ≤ 20 cm | other | yes | no | 15 | 13 |
| ≤ 20 cm | other | **no** | no / yes | 27 / 2 | 26 / 2 |
| 20–60 cm | same | yes | yes / no | 284 / 96 | 145 / 65 |
| 20–60 cm | same | **no** | no / yes | 12 / 1 | 12 / 1 |
| 20–60 cm | other | yes | no / yes | 383 / 12 | 119 / 5 |
| 20–60 cm | other | **no** | no / yes | 212 / 25 | 139 / 25 |

The unassociated clusters near a stop are numerous (27 within 20 cm on 26 items, 212 within 60 cm on 139) and, on this record, **never the Michel and never a gamma**. They are other-flash pieces that only look adjacent (doc 53's t0 argument). For the STM / Michel chain the answer to "how to deal with them" is: leave them out, and the record is the evidence.

### 4.3 The population that is real: same-bundle gammas the collect does not take

161 gamma tags on 87 items sit in same-bundle, associated near clusters with no chain role. By P4's admission (doc 71: radius 50 cm in production, C++ 35; max length 10 cm; a 60° cone around the Michel's direction; per-gamma 20 MeV, total 60 MeV):

| distance to the stop | length ≤ 10 cm | length > 10 cm |
|---|---:|---:|
| ≤ 35 cm | **76** | 3 |
| 35–50 cm | 42 | 0 |
| > 50 cm | 38 | 2 |

Cluster length p50 2.8 cm, p90 6.3 cm — they are small. **54 of the 87 items have no Michel object**, and the gamma collect runs only for an attached or bridged Michel (`michel_conn_type` 1 or 2; `CheckSTM_Michel.cxx:2845`), so on those items nothing can be collected until the Michel is. Only 11 of the 87 items have any stop gamma today. What excludes the 76 within reach on the other 33 items — the cone, the per-gamma cap, the body test — needs a per-piece gate readout the output does not carry.

## 5. Action items

Each under the doc 56 bar (default-OFF knob, byte-identical OFF gate on both ProtoDUNEs, criteria and predictions before the arm, gains and losses by name, own doc, PDVD flip only if confirmed).

1. **Segment census knob (output only; no physics).** Every unclaimed PR segment of the main cluster gets a row in `T_stm_michel_pts` with a new role (8), its distance to the stop, distance to the muon body, and — for the ones the piece admission looked at — the gate that rejected it, the way role 6 carries `rej` / `d_stop` / `d_body` for survey companions (doc 53). Target: the 42 orphan michel tags (§3.2) and the 21 that touch a stop with a found Michel; makes items 2, 4 and 5 gradeable and gives the viewer something to click. **Must not go through the survey's `preload_clusters` path** (doc 53 measured a 20–25 % mover rate on the muon's profile branches there): a read-only pass over the cluster's PR segments after the verdict. Scan need: none; the tags exist.

2. **A peak-then-drop stop mover, default OFF.** After a Bragg peak ≥ 1.4 × plateau, a tail whose median falls to ≤ *f* × the peak (not the plateau) with a bend ≥ 15° at the drop row marks the stop; split at that row when no vertex exists (doc 58's `break_segment` primitive). Target: the 14 hot-tail items of §2.3 by name, 11 of them with the bend. Negative control: the 284 THRU items, in particular end deltas. Grading needs no new scan: the 25 `pin_rr` values are the record (pin residual before / after, as docs 57 / 58 reported). In the same arm set: `split_kink_min_deg` swept to 8–10 for the four straight bridges (the collapse test stays as the guard), `retreat_peak_frac` 1.25 for `039253_3/61`, and the log read on `039349_30/45` / `039252_16/110`. Expected coupling: `039253_3/61` and `039349_32/63` recover their Michel through T3c once the stop is right; the `is_stm` 0 items among the 21 recover if the verdict profile is re-read from the moved stop.

   *Result (doc 82): the mechanism is measured DEAD and not flipped* — on the arm the peak-relative tail costs 7 new `is_stm` false positives and 4 true positives (two owner-confirmed) for 5 recovered stoppers, because the through-going items it fires on bend 27-69 deg, straddling the signal's. `split_kink_min_deg: 10.0` from side-sweep A IS now PDVD production: 4 stops move, the census is identical on every class, mean |stop - pin| 4.76 -> 4.33 cm.

   *Correction (doc 82, 2026-09-10), on three points.* **(a)** The rule as written above is not a collapse test. The peak gate already demands peak ≥ 1.4 × plateau, so `tail ≤ 0.7 × peak` is `tail ≤ 0.98–2.1 × plateau` — a muon that simply keeps going passes it. Offline it fires on 88 of 259 through-going control items (`d82_sizing.py` §1, rows restricted to the chain's last segment). **(b)** What blocks the movers on 14 of the 21 is the Bragg gate of §2.3's correction, not the collapse test; the working rule is therefore the peak-relative tail **plus** `michel_collinear_split`, **plus** the row's own bend as the thing that carries the looser tail, at ≥ 25° rather than 15°. Gated that way, the through-going control shrinks 6.5× (259 → 40 Bragg-confirmed items) and the rule reaches 8 of the 21 with 2 control fires. **(c)** `retreat_peak_frac` 1.25 is moot for the reason in §2.3. `split_kink_min_deg` 10 remains worth an arm and reaches three of the items the Bragg-gated rule cannot (`039349_81/25`, `039349_34/48`, `039252_9/101`). Doc 82 builds (b) and reports the rest. *(Doc 83 §9.5: with the compiled `mip_dqdx` 55000 in the twin's live-row floor, rather than 50000, the offline count is two; `039252_9/101` drops out. Doc 82's arm results are C++ and unaffected.)*

3. **T2c's kink margin.** `039252_2/79`: owner-confirmed STM_MICHEL, split fired, 8.9 MeV / 9.3 cm Michel at 59.58°, demoted by `moved_stop_michel_guard` with `moved_stop_michel_kink_min` 60. Re-grade T2c on the current record (doc 61 graded it on smx1a, where this item read THRU): if it is now 1 owner TP lost for 4 THRU FPs removed, state so; a value under 59.5° recovers the item — a one-item margin, to be declared as such. No scan.

4. **Backward-Michel admission, default OFF** *(re-read after doc 80: an attached-arm rule, not a body-exclusion exception — the four segments are connected to the chain, rej 13, and the Michel classification did not take them)*. A main-cluster piece whose near end lies within ~5 cm of the stop and whose direction has cos < −0.5 to the muon's last 5 cm joins the Michel. Target: `039349_64/24`, `039349_9/19`, `039349_64/52`, `039349_69/56` (and the 21 stop-touching members on found items, as energy). Negative control: the muon's own split-off fragments and pr54 residuals — which item 1's census is what names. After item 1.

   *Result (doc 83): FLIPPED in PDVD production* as `michel_near_stop_arm_cm: 5.0`. `039349_64/24` and `039349_9/19` recover their Michel, with the tagged segment as the seed; `039349_9/19` also turns `is_stm` 1 through `topology_stop_evidence`. That is `michel_found` TP 134 → 136 and `is_stm` TP 225 → 226, with 0 new FP, 0 TP lost, and the other 594 candidates bit-identical.

   *Correction (doc 83, 2026-09-11), on three points.* **(a)** All four segments leave the chain at its **penultimate vertex**, 2.5–6.5 cm before the stop, and the chain's last segment is a short stub. The stop-arm classifier never sees them; the interior one files them as kOther. The working rule is therefore "offer production's stop-arm gate to a kOther arm at a chain vertex within 5 cm of the stop", not a new direction cut. **(b)** The `cos < −0.5` above (a turn of ≥ 120°) would keep only `039349_64/24` of the four targets. A direction threshold also does not port between windows: "the muon's last 5 cm" and the shipped gate's kink against the 15 cm incoming chain segment disagree by up to ~40° on these very segments. The shipped gate reads the kink against the incoming chain segment, and **the distance gate is what buys the purity**: 0 control fires within 7 cm, the first STM_ONLY and THRU arms at 8.5–9.9 cm. `039349_69/56` (0.29 MIP) and `039349_64/52` (a 34.6 cm shower subtree) are refused by that gate, which is not loosened for them. **(c)** "The 21 stop-touching members on found items, as energy" are stop-vertex arms that failed the gate. Absorbing them takes 19 scanner-tagged Michel pieces with 14 muon/delta/gamma pieces and 2 untagged ones, on the same items that sank doc 73. It was not built.

5. **P4 rejection census.** For the 76 same-bundle gamma tags within 35 cm and ≤ 10 cm on items *with* a Michel object, which gate excluded them (cone, per-gamma cap, body); then decide whether the collect should anchor on the stop when no Michel is found (the other 54 items). Item 1's per-piece `rej` column is the natural carrier. Doc 71's purity (0.933) is the bar.

6. **Hand-check, no code.** `039349_43/66`, `039349_72/11`, `039253_0/44`, `039349_30/45`: no fitted charge near the stop; the Michel is unimaged or in a dead region — doc 96's territory.

## 6. Order

Doc 77's own items stay in front: **the candidate cap first** (two owner-confirmed Michel stoppers, a config value, no scan — doc 79), then P1's floors. Of the items here, **1 before 2, 4 and 5** (it is what makes them gradeable); 2 is the largest physics gain (13 missed Michels and 8 missed stoppers sit behind the stop); 3 is a one-line re-grade.

## 7. Next

Doc 79: the cap. Doc 80: item 1. Doc 82: item 2 (with the corrections above; the peak-relative rule is dead, `split_kink_min_deg` 10 flipped). Items 3, 4 and 5 remain — and doc 82 §9 recommends **4 before 3**, since `039252_2/79` turns out to want a smaller stop correction than item 2 applied, and doc 80's census now makes the attached-arm population (rej 13) gradeable. Doc 83: item 4 (flipped: `michel_near_stop_arm_cm` 5). Items 3 and 5 remain; 3 is next, a re-grade with no scan.

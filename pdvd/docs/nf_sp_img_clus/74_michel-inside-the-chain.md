# 74 — P3: the Michel the fit carried inside the muon chain

This is P3 of doc 70 §4.2 / §6. The owner asked for each proposal in its own file; this is the P3 file.

**Status (2026-09-10): DONE; one of three sub-knobs is in PDVD production.** §1–§4 were written before any arm ran; the results are in §5–§9.
- **On: `retreat_tail_strict: true`** (§8). The stop retreat judges the dropped tail on the rows past the vertex.
  - `michel_found` 134 / 12 / 24 → **136 / 12 / 22**: `039252_16/32` (32009, the scanner's "real Michel") and `039349_11/19`.
  - `is_stm` is identical (225 / 7 / 51). The owner's `michel` tags in role 3 go 209 → 215.
  - 0 new false positives, 0 lost true positives, and no scan pin further away.
- **Off: `michel_collinear_split`**, doc 70's proposal as written. It fires on one production item, and that item loses its `is_stm` TP (`039349_68/65`).
- **Off: `retreat_tail_sublive`.** It adds `039253_3/61`, but makes a through-going Michel false positive (`039349_23/43`), its pre-stated test.
- **Correction to doc 70 (§2):**
  - on production the named items are not Bragg-confirmed at the chain walk's origin;
  - the cool two are a retreat case, not a split;
  - the hot two are not collapses.
- **Toolkit `54e6ab99`.**

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# the production re-measurement of sec 2 (read-only: doc 72's bare-production arm p72vprod) -> /home/xqian/tmp/p74/sizing.txt
python3 $X/d74_sizing.py --json /home/xqian/tmp/p74/pred.json
# build (toolkit clus/), the tests, the pin
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus
cp -a ../local/lib /home/xqian/tmp/p74/libpin_p74
# wave 1 (7 arms, bare production config), gates, grading
$X/d74_arms.sh; $X/d74_gates.sh 2>&1 | tee /home/xqian/tmp/p74/gates.log
```

## 1. The question

Doc 70 §4.1 found four owner-confirmed Michels **inside the muon chain**: the scanner's `michel` tag sits on a role-1 segment, and the fit runs straight through.
- `039252_16/32` (32007: 3.4 cm, 0.17 MIP)
- `039253_3/61` (61008: 2.4 cm, 0.27 MIP)
- `039349_60/40` (40002: 6.8 cm, 1.35 MIP)
- `039349_64/65` (65003: 3.8 cm, 1.47 MIP)

Doc 70 read the mechanism as the Bragg-confirmed guard: "because `bragg_confirmed(chain)` is already true on a found stopper, T1a/T1c never run". It proposed `michel_collinear_split`:
- on a Bragg-confirmed chain whose last segment continues past the peak at a collapsed dQ/dx, split it at the fall;
- re-classify the remainder.

The two hot items were its named risk.

The scan record calls every one of the four an **OVERSHOOT**: the fit runs past the muon's real stop, and the scanner moved the pin back to the junction (`pin_rr` 3.3, 2.1 and 6.1 cm; `039349_64/65` is unpinned, at low confidence). On `039252_16/32` the scanner adds that "the real Michel is the 6.8 cm arm 32009 off the same vertex".

## 2. Re-measured on today's production

This section is measured on `p72vprod`, doc 72's bare-production confirmation arm. `d74_sizing.py` holds offline twins of the three C++ tests: the chain walk's `bragg_confirmed` (geometric origin), `stm_michel_stop_retreat` and `stm_michel_stop_split`, at the PDVD production constants.
- **The payload profile is the C++ profile row for row**, dead rows included.
- **A chain vertex is written twice.** The twin rebuilds segment membership from those duplicate rows; the rebuild matches `n_chain_segs` on 537 of 568 candidates.
- **The geometric contrast** matches the payload's `contrast` wherever T7's anchor shift is 0: 214 items, median |Δ| 1e-6.
  - 34 differ by more than 1e-3. 19 of those are T7's end_L = L[peak] + 0.2 cm, with the reported shift clamped at 0.
  - The other 15 are not explained. Five of them sit within 0.1 of the 0.6 × expected cut: `039253_5/121`, `039253_8/53`, `039349_18/33`, `039349_32/56` and `039349_54/66`. For those five, the twin's Bragg-confirmed call is uncertain.

### 2.1 Doc 70's P3 as proposed is nearly dead on production

The retreat, and failing that the split, let loose on every Bragg-confirmed chain the retreat and split have not already moved fire on **one** candidate: `039349_68/65`, a split 20.4 cm from the end at 18.2°, on an STM_ONLY item that is already `is_stm` 1.

The reason is the precondition. **Neither cool named item is Bragg-confirmed at the geometric origin.** The chain walk's tail window, rr 0.5–3 cm, reads the collapsed overshoot, so the contrast fails.

Their verdict passes (`is_stm` 1) only because T7 (doc 65) re-reads the Bragg test from a peak-anchored origin (shift 2.63 cm and 2.76 cm). The chain walk keeps the geometric origin on purpose. So T1a and T1c were *eligible* on both, and failed their own tests.

42 found stoppers are in this state: not Bragg-confirmed at the geometric origin, but `is_stm` 1 through the anchor.

| group (record, production `is_stm`) | n | Bragg-confirmed (geometric) | T7 anchor shift > 0 | not confirmed but `is_stm` 1 |
|---|---:|---:|---:|---:|
| stopper, is_stm 1 | 225 | 183 | 122 | 42 |
| stopper, is_stm 0 | 51 | 23 | 28 | 0 |
| THRU, is_stm 0 | 261 | 31 | 176 | 0 |
| THRU, is_stm 1 | 7 | 4 | 5 | 3 |
| not judged | 24 | 11 | 12 | 3 |

### 2.2 What does stop the retreat: how it reads the dropped tail

On both cool items the Michel is the **whole last chain segment**, ending at a chain vertex. That makes it a retreat case (drop a segment), not a split (cut inside one). The retreat (doc 57) drops the segment when:
- the tail's median is below 0.5 × plateau, and
- a Bragg rise survives to retreat to.

Two things in how it builds that tail defeat it on a short overshoot:
1. **The tail includes the vertex row.** `stm_michel_profile` writes the junction twice (once per segment), and the doc 57 test takes every row with L ≥ the boundary. So both copies of the Bragg-peak row are in the tail. Over a 2–3 cm segment of a handful of rows they set the median.
2. **The live floor removes the collapse.** Rows below 0.15 MIP are dropped as dead cells, and a collapsed overshoot reads 0.1–0.2 MIP.

The n_drop = 1 tail, as median / plateau (rows in brackets):

| item | drop | as built (doc 57) | strictly past the vertex | + the sub-live rows | both |
|---|---:|---:|---:|---:|---:|
| `039252_16/32` | 3.35 cm | 1.17 (4) | **0.47** (2) | 0.17 (8) | 0.13 (6) |
| `039253_3/61` | 2.36 cm | 0.80 (3) | 0.35 (1: too few) | 0.35 (5) | **0.08** (3) |
| `039349_60/40` | 6.80 cm | 1.09 (13) | 1.03 (11) | 1.09 (13) | 1.03 (11) |

- **The two hot items are out of reach, and correctly so.** `039349_60/40`'s terminal segment is not a collapse (1.03 × plateau). The scanner's own note says it sits "far below what a post-Bragg muon would carry but NOT below the track's own plateau". `039349_64/65`'s 65003 is the Bragg stub `absorb_bragg_stub` took into the chain (`n_stub_absorb` 1); the scanner's confidence is low and three-way.
- **Reach over every chain the doc 57 guard admits**: not Bragg-confirmed, and not already moved in production. The four readings fire as follows beyond what the doc 57 reading already does (which, on the final chains, is nothing):
  - **strictly past the vertex:** 4 stoppers, **0 THRU**:
    - `039252_0/75` (`is_stm` 1, `michel_found` 1 bridged);
    - `039252_16/32`;
    - `039349_11/19` (`is_stm` 1);
    - `039349_51/21` (`is_stm` 0; `no_bragg`, `shape_flat`).
  - **plus the sub-live rows:** those 4, plus `039253_3/61` and **one THRU**, `039349_23/43` (a 21.1 cm drop; `is_stm` 0 on `no_bragg`, `shape_flat` and `vertex_hadron`).
  - **the payload's dead-channel lists** (does a sub-live row sit on a dead wire in any plane?) separate nothing: the same fires either way.
- **The 18 items where the retreat (7) or the split (9) already fired in production cannot be replayed offline:** their chain is already mutated. A strict tail can also re-order the two shipped mechanisms, because the split runs only when the retreat did not fire. The arm decides these, by name (§6).

### 2.3 What a retreat would expose at the new stop

- **`039252_16/32`.** The new stop is the 32008/32007 junction. Today 32009 (6.79 cm, 0.90 MIP, shower-flagged, about 50° off the body) hangs there as a role-2 delta at an interior vertex. At the stop, the classifier would read it as a Michel: a shower-flagged arm with a turn ≥ `michel_shower_min_kink_deg` 15°.
  - Production's own attached single-segment Michels (68) put a 6.79 cm, 0.90 MIP electron at **15.8 MeV**; its neighbours in length and charge carry 10.0–21.4 MeV.
  - A retreat sets `n_retreat`, so the moved-stop veto (T2c) applies, and ~50° does not reach P3b's 60° exemption. **Above 10 MeV it survives; below, T2c demotes it.**
  - 32007 itself becomes a 3.35 cm, 0.17 MIP collinear stop arm: kOther.
- **`039253_3/61`.** The new stop is the 61004/61008 junction. 61008 (2.44 cm, 0.27 MIP, collinear) becomes the stop arm, with 61007 + 61001 in its far subtree. It is below `michel_mip_lo` 0.3 and does not turn, so it is kOther. No Michel is predicted there; only the stop moves toward the pin.

## 3. Design (toolkit `clus/`)

Two families. Each is default OFF, and the OFF path is the old code.

**P3-literal, `michel_collinear_split`** (bool, default false). This is doc 70's proposal as written: the retreat and the split also run on a chain whose profile already shows the Bragg rise.
- The guards become `(!bragg_confirmed(chain) || michel_collinear_split)`.
- `bragg_confirmed` is pure, and it is evaluated under exactly the conditions it was before.

**P3-measured, the retreat's tail reading.** Two new fields of `StmMichelRetreatThresholds`, each off by default and wired from its own knob:
- `retreat_tail_strict`: the tail takes only rows strictly past the vertex it would retreat onto (L > boundary).
- `retreat_tail_sublive`: the live floor is not applied to the tail. The peak test keeps it.
  - **Hazard, stated in the header and pinned by a test:** a dead-channel stretch past a live Bragg rise then reads as a collapse. Nothing at this site knows the channel map; `FiducialUtils::check_dead_volume` is a volume test.

**Diagnostic.** `stop_move_p3_bits` is written only when any of the three knobs is on (the P1/P4 pattern):
- bit 0: the P3 tail reading changed the retreat's answer. The doc 57 reading is re-run to decide; it is a pure function.
- bit 1: the stop moved on a Bragg-confirmed chain.

A DEBUG line names each fire.

**Deliberately unchanged: the moved-stop veto (T2c) reads a P3 retreat like any other.** Doc 61 built it because moving a stop is exactly how a through-going track picks up a spurious attached arm. P3 does not argue that case away, and the arm reports every veto (§6).

**No drop-length cap** is designed in. Capping at ~10 cm would exclude `039349_23/43`, but that is tuning on the one negative control (doc 62's rule).

## 4. Pre-stated criteria and predictions (written before wave 1 launched)

Arms (bare production config, `d74_arms.sh`):
- `p74vleg` / `p74hleg`: the P72 pin (production, toolkit 299d8bc4).
- `p74voff` / `p74hoff`: the P74 pin, knobs off.
- `p74vcs`: `michel_collinear_split`.
- `p74vts`: `retreat_tail_strict`.
- `p74vtl`: `retreat_tail_strict` + `retreat_tail_sublive`.

**Gates** (all must hold before any flip):
1. OFF gate, byte-identical: `p74vleg` ↔ `p74voff` (120 events) and `p74hleg` ↔ `p74hoff` (61). Checks: zip member content, calib md5, every `T_stm_michel` branch and point row. Stale-baseline check: `p72vprod` ≡ `p74vleg`.
2. Compiled-config proof: each key appears in its ON arm only. Done before launch, `/home/xqian/tmp/p74/cfg/`.
3. **The 18 production retreat/split items, off → on, by name.** Fields: `n_retreat`, `n_split`, `n_chain_segs`, `stop_dis`, `michel_conn_type`, `michel_found`, `is_stm`. Plus every other candidate whose stop moved. **Any change not explained item by item stops the round** (CLAUDE.md §5.5).

**Predictions** (from §2, on final chains):
- **`p74vcs`:** fires on `039349_68/65` only. No change on the four named items.
- **`p74vts`:** fires on `039252_0/75`, `039252_16/32`, `039349_11/19` and `039349_51/21`, plus whatever the 18 re-order.
  - `039252_16/32`: the stop moves to the pin (3.35 cm vs pin 3.3), 32009 becomes an attached Michel at about 16 MeV, so **+1 `michel_found` TP**. If it lands under 10 MeV, T2c demotes it and the gain is 0.
  - Named risks: `039252_0/75`'s `michel_found` (today a bridged Michel); `is_stm` on the three `is_stm` 1 fires. Possible gain: `039349_51/21` `is_stm`.
- **`p74vtl`:** `p74vts`'s fires, plus `039253_3/61` (the stop moves toward the pin, no Michel predicted) and `039349_23/43` (THRU, 21.1 cm: the false-positive risk).

**Flip criteria, per sub-knob:**
- **Gain:** at least one of:
  - a `michel_found` TP or an `is_stm` TP on a judged item;
  - one more owner `michel` tag in role 3.
- **Guards:**
  - 0 new `is_stm` FP;
  - 0 `is_stm` TP lost;
  - 0 `michel_found` TP lost;
  - 0 new `michel_found` FP on judged items;
  - no scan-pinned item's stop more than 1 cm further from its pin;
  - every change among the 18 explained.
- `retreat_tail_sublive` stays OFF unless it clears the same guards; `039349_23/43` is its named test.
- A sub-knob that fails stays OFF, with the failure named (the T3a and P2 precedent).
- A flip is one edit of `pdvd/wct-pr-perevt.jsonnet` with the three compiled-config proofs, then a bare-production confirmation arm. PDHD stays OFF.

## 5. What was built (toolkit `54e6ab99`)

- **`StmMichelFunctions.{h,cxx}`**: `StmMichelRetreatThresholds::tail_strict` / `tail_sublive`, plus two lines in the retreat's tail loop. With both false the loop is the doc 57 one.
- **`CheckSTM_Michel.cxx`:**
  - the three knobs, round-tripped in `default_configuration()`;
  - the retreat/split guards as in §3;
  - `stop_move_p3_bits`, written only with a knob on;
  - one DEBUG line per fire.
- **Tests:** 5 new doctest cases, plus the 3 defaults pinned.
  - A 120-shape overshoot grid pins the OFF reading to a literal copy of doc 57's.
  - Two cases pin the strict and the sub-live overshoots.
  - One pins the dead-stretch hazard.
  - One refuses a live continuation under every reading.
  - `wcdoctest-clus` 371 / 371 (was 366).
- **Freshness:** the installed lib (13:14:03) is newer than the last source edit (13:13:16). Pin `/home/xqian/tmp/p74/libpin_p74` (md5 `566dc517`), unchanged before and after every arm. The base pin `libpin_p72` (`8aa45329`) is also unchanged.
- **`prep_stm_michel_scan.py`** carries `stop_move_p3_bits` into the payloads.

## 6. Gates

| gate | result |
|---|---|
| completeness | every PDVD arm 120 / 120, every PDHD arm 61 / 61; 0 loader deaths |
| compiled config (before launch) | each key appears in its own ON arm only (`/home/xqian/tmp/p74/cfg/`) |
| **OFF gate PDVD** `p74vleg` ↔ `p74voff` | zip member content 120 / 120; calib 119 / 119 (`039252_11` writes none on either); **578 / 578 candidates bit-identical on all 138 branches**; point rows 578 / 578 |
| **OFF gate PDHD** `p74hleg` ↔ `p74hoff` | zip 61 / 61; calib 61 / 61; **325 / 325 on 130 branches**; points 325 / 325 |
| stale baseline `p72vprod` ↔ `p74vleg` | zip 120 / 120; 578 / 578 on 138 branches |
| the 18 production retreat/split items | `p74vcs`: none changes. `p74vts`: 4 change. `p74vtl`: 4 change. All explained below. |
| prediction vs arm | `cs` 1 predicted, 1 fired, the same item. `ts`: all 4 predicted fired, plus 4 of the 18. `tl`: all 6 predicted fired, plus 3 of the 18. 0 predicted-but-silent. |
| `census_score.py --check` | 0 of 14 differ |

**The 18, item by item.** These are the chains the offline twin could not replay (§2.2).

*Under `retreat_tail_strict`:*
- **`039349_10/58` and `039349_5/54`, STM_MICHEL, both `is_stm` 1.** In production the split had fired on each. Under the strict reading the retreat fires first, and doc 58's design gives it precedence when it applies, so the split no longer runs. The retreat drops the last chain segment where the split cut inside it.
  - `039349_10/58`: stop distance 6.92 → 9.32 cm. The Michel stays attached; 8.4 → 10.9 cm, 10.7 → 23.4 MeV.
  - `039349_5/54`: the owner's `michel` tag on the swallowed segment moves from role 1 to role 3.
  - `is_stm` and `michel_found` unchanged on both.
- **`039349_20/41`, THRU.** The production retreat (4.27 cm) no longer fires: without the vertex row the tail does not qualify. The stop returns to the fit end.
  - The moved-stop veto no longer has a spurious 5.7 MeV Michel to demote, and the owner-muon-tagged 41001 leaves role 3.
  - `is_stm` 0 and `michel_found` 0, as before.
- **`039349_54/56`, MESSY.** The production retreat (2.66 cm) no longer fires. `michel_found` stays 1 (bridged) and `is_stm` stays 0. Not judged.

*Under `retreat_tail_strict + retreat_tail_sublive`:*
- `039349_10/58` and `039349_5/54`: the same as above.
- `039349_20/41`: unchanged from production. The sub-live rows keep its retreat.
- `039349_54/56`: retreats two segments (11.79 cm) and turns `is_stm` 0 → 1 (MESSY, not judged).

## 7. Result, by name

Census on the smx1a + smx3 + smx4 record, `p74voff` as the baseline (`census_score.py`, `d74_score.py`; `/home/xqian/tmp/p74/score_p74.txt`):

| arm | `is_stm` TP / FP / FN | `michel_found` TP / FP / FN | owner `michel` tags in role 3 | role-3 segments tagged muon / delta | median distance to the 25 pins (≤ 2 cm) |
|---|---|---|---|---|---|
| `p74voff` (= production) | 225 / 7 / 51 | 134 / 12 / 24 | 209 of 263 | 10 / 31 | 2.88 cm (8) |
| `p74vcs` | **224** / 7 / **52** | 134 / 12 / 24 | 209 | 10 / 31 | 2.88 cm (8) |
| **`p74vts`** | 225 / 7 / 51 | **136 / 12 / 22** | **215** | 9 / 31 | 2.67 cm (9) |
| `p74vtl` | 225 / 7 / 51 | 137 / **13** / 21 | 217 | **11 / 32** | 2.67 cm (10) |

### `retreat_tail_strict`: passes every criterion

- **`039252_16/32`**, STM_MICHEL, `michel_found` 0 → 1. The stop retreats 3.35 cm onto the 32008/32007 junction: pin distance 3.31 → 0.00 cm.
  - 32009 is now a stop arm. On the C++ line: 6.79 cm, 0.92 MIP, shower-flagged, 73.1°, kind Michel.
  - It is attached at **16.1 MeV**; the prediction was 15.8. Above 10 MeV, T2c leaves it alone (so would P3b's 60° exemption).
  - 32007 (0.17 MIP, 15.1°) is a kOther stop arm.
  - The owner's `michel` tag on 32009 goes from role 2 to role 3.
- **`039349_11/19`**, STM_MICHEL, `michel_found` 0 → 1. **Not predicted:** §2 listed this item only as an `is_stm` risk.
  - The stop retreats 2.03 cm. Three arms meet at the new stop; 19008 is kind Michel: 9.30 cm, 0.35 MIP, shower, 84.9°, 12.6 MeV attached.
  - 19007 (0.25 MIP, far_len 57.6 cm) and 19009 (collinear, 0.67 MIP) are kOther.
  - 2 of the owner's 3 `michel` tags are now in role 3. This is doc 73's P2 item (19004's subtree), reached from the right stop instead.
- **`039252_0/75`**, STM_MICHEL, retreats 2.83 cm. The named risk did not happen.
  - `michel_found` stays 1 (bridged). The object grows from 9.5 to 24.8 MeV, and 2 more owner `michel` tags are in role 3.
- **`039349_51/21`**, STM_ONLY, retreats 3.63 cm. `is_stm` stays 0; the capture-gamma rows go 5 → 7.
- **The four among the 18** are in §6.
- **Totals:**
  - `is_stm`: no mover.
  - `michel_found`: 2 gained TP, nothing lost, no new FP.
  - Tagged-bad role-3 segments: none new; one gone (`039349_20/41`'s 41001, muon).
  - Pins: two items closer (`039252_16/32`; `039349_10/58` by < 0.01 cm), none further.

### `retreat_tail_sublive`: fails its named test, stays OFF

- It adds **`039253_3/61`** (STM_MICHEL, both).
  - The stop retreats 2.36 cm to the 61004/61008 junction: pin distance 2.09 → 0.00 cm.
  - 61008 becomes the attached Michel: 2.44 cm, far subtree 14.88 cm, 22.3 MeV. All 3 of the owner's `michel` tags are in role 3.
  - **The prediction said kOther and was wrong.** On the C++ line, 61008 reads 0.45 MIP and 53.2° at the new stop; the payload's segment median read 0.27 (doc 72's lesson, again: the payload points are not the fits).
- It also retreats **`039349_23/43`** (THRU) by 21.1 cm and attaches 43025: 21.11 cm, 0.34 MIP, shower-flagged, 15.3°, which clears the 15° shower minimum.
  - That is a `michel_found` **false positive**, with the owner's muon (43025) and delta / other (43034) tags in role 3.
  - That is exactly the pre-stated risk, so the sub-knob stays OFF.

### `michel_collinear_split`: fails, stays OFF

Its one fire, **`039349_68/65`** (owner STM_ONLY), splits the Bragg-confirmed chain 18.98 cm from the tagger's stop at 18°. The shortened profile loses its verdict: `is_stm` 1 → 0, a **lost TP**. As doc 70 wrote it, P3 has no other reach on production.

## 8. The flip

`pdvd/wct-pr-perevt.jsonnet` gains one key after `stop_retreat_max: 2`, with a comment carrying the C++ default, the guarantee and the graded result:

```jsonnet
        retreat_tail_strict: true,
```

md5 `f596bedc` → `fa794f03`. It was edited with no PDVD arm in flight.

Compiled-config proofs (`/home/xqian/tmp/p74/flip/`, `flip_proofs.sh` with `-P`; every compile rc 0 and about 277 kB):
- **flip-equivalence:** the pre-flip file + `-S stm_michel_extra={retreat_tail_strict:true}` against the flipped file: 0 lines.
- **OFF path:** both files with the key forced false: 0 lines.
- **what moved:** pre vs post differ by exactly `"retreat_tail_strict": true`.

PDVD production is therefore what `p74vts` ran. Confirmation arm `p74vprod`: bare flipped production, P74 pin. *(result below)*

PDHD stays OFF: there is no PDHD hand-scan record.

## 9. Observations and next

- **The chain walk and the verdict disagree about where the peak is.**
  - 42 found stoppers are `is_stm` 1 only through T7's peak anchor, and are not Bragg-confirmed at the chain walk's geometric origin.
  - T7 moved the origin for the shape tests only, on purpose. So the graph's stop stays at the fit end, while the verdict reads a peak up to 3 cm before it.
  - `retreat_tail_strict` closes part of that gap on two items. A retreat that reads T7's anchor directly would be the general form. Not built.
- **The strict reading also undoes two production retreats** (`039349_20/41` THRU, `039349_54/56` MESSY). The doc 57 reading had fired on those tails because of the vertex row. On the THRU item that removes a spurious attached arm.
- **`039253_3/61` is reachable only by reading below the live floor.** That needs a dead-channel-aware sub-live reading. The payload's dead-channel lists did not separate anything on this sample, and the C++ site has no channel map.
- **15 of the offline twin's geometric contrasts** differ from the payload's by up to 0.68 for no identified reason (§2), five of them near the cut. They did not matter here (the prediction matched the arm on every replayable chain), but the twin should not be trusted on those five.
- **Next: P1b** (doc 70 §3.4, the anchor rise precondition), sized on a bare-production arm.

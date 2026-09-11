# STM + Michel hand scan of the PDHD sample on production — doc pdhd/18

**Status: done — a scan round. No C++ change and no production-config change,
so no A/B gate is owed; the scan arms' additivity gate is §2.2.** It
produces the PDHD record that the PDVD STM/Michel knobs have been waiting for:
every PDHD jsonnet comment that leaves a PDVD-production knob OFF gives the same
reason, *"no PDHD hand-scan record"*. Tag **`smx18`**, 317 items, verdict-blind,
scanned by five subagents driving the real `stm_michel_viewer.py` headless, on
**today's PDHD production** plus the 14 candidates its `max_candidates` = 8 cap
drops.

Owner ask, 2026-09-11: *"For PDVD, we have done a visual hand scan campaign …
509 events scanned by you with 5 subagents. Now, it is time to repeat this for
PDHD. … I have scanned 30 events myself, and for the remaining ones, I would
like you to repeat what you did for PDVD … up to 5 subagents. Hopefully, with
previous experience, this round of scan can be more efficient. You should use
the display for stm (port 5017), as well as the latest production as the
basis."* Three decisions the owner took before the round started:
**verdict-blind** scanners; a **blind calibration wave on the owner's 30** first;
and **scan the cap-64 extras too**.

**What it found, in five lines.**

1. **PDHD production's `is_stm` is pure and inefficient.** On the 303 items it
   reaches: **0 false positives** (purity 1.000) and efficiency **0.349**
   (60 of 172 hand stoppers). Counting only the owner's and the agents'
   `high` calls, efficiency is 0.592. `michel_found` on hand stoppers is
   0.813 pure and 0.629 efficient (§7).
2. **The misses are the shape tests, and half of them already hold a Michel.**
   * 87 of 112 missed stoppers are rejected only by `shape_flat` /
     `no_bragg` / `profile_sparse`.
   * 49 of them carry `michel_found = 1`: the population PDVD's P1 was built
     for.
   * APA0 is the weakest region (efficiency 0.132), because of its hardware
     charge deficit.
3. **The blind agents reproduce the owner on every call they mark `high`.**
   * Calibration on the owner's 30: 18/18 `high` verdicts agree, and 24/30
     agree on stopper-or-not.
   * A seeded blind double scan of 20: 3/3 `high`, 14/20 overall (§5, §8).
   * The confidence field is the reliability estimate.
4. **The scanners found two defects in the instrument (§6.2).**
   * The PDHD anode is at |x| ≈ 352 cm, but the display and every x distance
     say 358. 23 items were re-scanned, and 5 stoppers became `THRU`.
   * The measurement panels show only the fitted cluster's own cells.
5. **Each agent scanned an item in ~1.5 min, against ~2.7 on PDVD.** The
   scanning itself took 1 h 50 min for 310 records, against PDVD's ~5.5 h
   for 517. Both wall-clocks count scanning only: calibration, frames and
   apply are excluded on both sides. Most of the gain came from the pre-made
   frames and the numbers-first rubric (§6.1).

The next step (§11): grade PDVD's production knobs on PDHD against this record,
starting with P1, after the owner's review of queue tiers A–E.

## 0. Repro

```bash
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; H=/home/xqian/tmp/h18
C=$I/pdhd/stm_michel_scan/campaign
# 1. the two arms (pinned binary libpin_p82 = toolkit 082376c5) and the additivity gate
$I/pdhd/docs/scripts/h18_arms.sh            # h18s (display), h18b (grading of the extras)
bash $I/pdhd/docs/scripts/h18_gates.sh      # -> $H/gates/gates.log
# 2. the prep (display) and the bare-production grading keys
cd $I/pdhd/stm_michel_scan
for a in h18s p82bhoff h18b; do
  ./prep_stm_michel_scan.py --det pdhd --arm $a --outdir $H/prep_$a --sheetdir $H/sheet_$a \
      --pin-tranche ../../pdhd/docs/scan/pdhd_stm_michel_scan_sheet.tsv
done   # prep_h18s -> prep-pdhd-smx18; sheets re-keyed to the old scan_ids (sec 2.4)
# 3. blind frames, 4 processes, then the zoomed dQ/dx frame and the frame check
$C/shoot.sh $H pdhd $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv \
    $I/pdhd/stm_michel_scan/prep-pdhd-smx18 4
python3 $C/mkzoom.py $H/shots; python3 check_shots.py $H/shots
# 4. waves (each agent: AGENT_TASK.md + an items file + a private out dir)
python3 $C/nextwave.py $H $H/items_calib.txt cal --agents 5 --per 6
python3 $C/cmp_owner.py $H/backup/pdhd_smx1_labels.json $H/v_parts/cal_a*   # sec 5
python3 $C/nextwave.py $H $H/items_main.txt w1 --agents 5 --per 15            # ... w4
python3 $C/nextwave.py $H $H/items_main.txt dbl --agents 5 --double 20        # sec 8
# 5. resolve -> spec -> apply (real widgets) -> merge -> record -> census
python3 $C/resolve.py $H $H/items_all.txt
python3 $C/mkspec.py $H <items> 3 <batch>; $C/apply_parallel.sh $H pdhd <sheet> <prep> <batch> 3
python3 $C/fixup_spec.py $H       # rows that did not land -> re-apply into their own labeldir,
                                  # scan_harness.py apply ... --tag-passes 10 --settle-scale 2.0
python3 $C/merge.py $H pdhd smx18 $H/items_all.txt --write
python3 $C/mkrecord.py $H $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv \
    $H/backup/pdhd_smx1_labels.json $I/pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json \
    $I/pdhd/docs/scan/smx18/provenance.json --calib-items $H/items_calib.txt
./verify_scan_record.py --det pdhd --tag smx18 --record ../../pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json
python3 $I/pdhd/docs/scripts/d18_census.py --record $I/pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json \
    --key $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv \
    --key-extras $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_h18b.tsv --shots $H/shots
```

The label file is served for the owner on **:5023**, the app's PDHD default:

```bash
./serve_stm_michel_scan.sh 5023 --det pdhd --scan-tag smx18 \
    --manifest $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv \
    --prepdir  $I/pdhd/stm_michel_scan/prep-pdhd-smx18
```

It was checked headless: 317 items, 317 labelled. :5017 was taken at the time
by the PDVD option scan `smx5`, and a live label session is a scientific
record, so it was not touched. The app is the same one either way.

---

## 1. Where this starts

* **PDVD** has a 569-item record (doc pdvd/55: 60 by the main session, 509 by
  five subagents), later corrected by the owner's option scans (docs 68, 70).
  Docs 57–82 flipped about twenty `CheckSTM_Michel` knobs to PDVD production on
  that record.
* **PDHD** had the owner's own 30 labels (tag `smx1`, all tranche 1, on arm
  `d53h`, 2026-09-09) and nothing else, so every one of those knobs stayed OFF
  on PDHD.
* **What was different about PDHD going in.** The draw (doc pdhd/12, 303 items,
  seed 20260907, tranche 1 = 60) was already pinned. The scan display was
  already built. The frames, the rubric and the wave tooling existed only for
  PDVD, and most of the tooling lived only in a session scratchpad (doc
  pdvd/69 flagged this as an unverifiable repro).

## 2. The basis: production, the scan bag, and the cap

### 2.1 Arms

| arm | what | TLA | role |
|---|---|---|---|
| `p82bhoff` | PDHD production (bare) | none | the **grading** key for the 303 items production reaches |
| `p82hoff` | production + the scan survey bag | `survey_enable, survey_radius_cm 60, survey_max_len_cm 25, publish_other_arms` | reference for `h18s` |
| **`h18s`** | `p82hoff` + `max_candidates: 64` | the same bag + `max_candidates:64` | **what the scanner looks at** |
| **`h18b`** | `p82bhoff` + `max_candidates: 64` | `max_candidates:64` | the grading key for the **14 cap extras** |

* **Binary.** Both new arms ran on `libpin_p82`, a full `local/lib` snapshot
  (Clus md5 `5f2c3ede`, Root `46bf5105`) equal to toolkit HEAD `082376c5`,
  the binary that ran `p82hoff` / `p82bhoff`. The md5 was the same before and
  after each arm. Both arms completed 61/61 events with 0 loader deaths.
* **Why no `segment_census`.** `h18s` carries exactly `p82hoff`'s bag, without
  `segment_census`, so the gate below is like for like. It also keeps the
  object tables of the same kind as the PDVD tranche-2 scan.

**Compiled-config proof** (M6): `wcsonnet` on `pdhd/wct-pr-perevt.jsonnet`
with each TLA, pretty-printed and diffed, in `/home/xqian/tmp/h18/cfgproof/`.
Each diff is exactly one line:

```
p82hoff -> h18s :  >    "max_candidates": 64,
bare    -> h18b :  >    "max_candidates": 64,
```

### 2.2 The additivity gate (`h18_gates.sh`, doc pdvd/79's test)

| | `h18s` vs `p82hoff` | `h18b` vs `p82bhoff` |
|---|---|---|
| shared candidates bit-identical, every branch | **325 / 325** (133 branches) | **325 / 325** (130 branches) |
| shared candidates, identical point geometry and roles | 325 / 325 | 325 / 325 |
| `is_stm` flips / dropped candidates | 0 / 0 | 0 / 0 |
| new candidates | 16 | 16 |
| events whose zip / calib / trees differ | 7 (exactly the cap events) | 7 (the same seven) |

The seven events where the cap fired are `028084_23`, `028084_7`, `029107_12`,
`029107_18`, `029107_20`, `029107_26` and `029107_5`. The log is
`/home/xqian/tmp/h18/gates/gates.log`, with the branch census in `g_h18s.txt`
and `g_h18b.txt`. **On PDHD, as on PDVD, the cap only adds candidates.**

### 2.3 The item set

`prep_stm_michel_scan.py --arm h18s --pin-tranche <the d53h sheet>` gives
**317 items**: all 303 of the pinned sheet, plus 14 of the 16 new candidates.
The prep's own `has_pass` / ≥ 20 points / ≥ 10 cm filter drops the other two:
`028084_7/109` (28 cm, bits 512) and `029107_5/107` (29 cm, bits 520).
* No item dropped out and no tranche moved.
* The 303 kept items have the same `npts` and `muon_len`.
* All 30 owner items are present.
* The per-item list is `pdhd/docs/scan/smx18/itemset_diff.tsv`.

**What changed between `d53h` (the owner's arm) and `h18s`**, payload by
payload on the 303 shared items:
* The **muon fit** (x, y, z, q) is identical on all 303. Only the projected
  wire/tick coordinates (`pu/pv/pw/pt`, which the measurement panels use) moved,
  on 10 items, one of them an owner item (`028084_7/56`).
* The **verdict** block gained the diagnostic branches of docs 57–82. Its shared
  scalars differ only in the `survey` / `michel` / `delta` / `gamma` sub-blocks,
  on 16 / 5 / 5 / 3 items. `is_stm` is identical.
* The **particle flow** gained role-7 chain roles (`publish_other_arms`) on 207
  items.

So **the Bragg evidence the owner judged on `d53h` is the evidence the agents
judged on `h18s`**. A calibration disagreement cannot be an arm artifact.

### 2.4 Sheet ids and grading keys

* **`scan_id`.** The prep numbers items in (event, cluster) order, so the 14
  extras renumbered 220 items. The sheet and keys were re-keyed after the prep:
  the 303 keep their old `scan_id` (so `smx1`'s ids name the same item in
  `smx18`), and the extras are 304–317.
* **Grading keys:**
  * `pdhd_stm_michel_scan_key_p82bhoff.tsv` is the grading key for the 303.
  * `…_key_h18b.tsv` is the grading key for the extras.
  * `…_key.tsv` (`h18s`) is the scan arm's own key and is **never** used for
    grading. The survey moves 197 of 325 candidates' profile branches.
* **How far the scan arm's key is from production.** On the 317 items the
  `h18s` key agrees with the bare keys on `is_stm` everywhere. It differs on
  `michel_found` on one item (`028084_4/20`) and on `reject_bits` on 2 of 303.

## 3. The instrument

### 3.1 Harness flags, all defaulting to the old behaviour (`scan_harness.py`)

Two more, `--tag-passes` and `--settle-scale`, are apply-only and are §9's.

* **`--prepdir` / `--manifest`.** The viewer has always taken them, but the
  harness never passed them through, so it could only drive `prep-<det>`.
* **`--blind`** drops the chain's **verdict** from `context.json`:
  `ends.is_stm`, `ends.reject_names`, `ends.in_fv`, and the `flow` summary
  ("the chain's answer — particle flow"). It stamps `blind` in the file.
  * **The frames need no change.** They are canvas clips, and no canvas title
    carries a verdict.
  * **Residual leak, stated rather than hidden.** The object table's `group` /
    `chain` columns, and `seg_head`'s group counts, are the chain's *typing* of
    each object. Every 3-D frame draws that typing as per-group layers, so this
    round is **verdict-blind, not blind**.
* **`--hide-selection`** empties the amber "picked object" layer (`pfsel`: 9 px
  at alpha 0.95 in 3-D, 7 px in the measurement panels) before each frame.
  * **Why.** The object table opens with row 0 picked, and row 0 was the
    muon's stop segment on 152/152 items checked. So the band sat on exactly
    the cells `f_meas` exists to show (smoke item `029107_18/146`).
  * **Why PDVD rarely showed it.** On PDVD the same band often fell outside
    the ±150 window: `039252_0/35`, re-shot today, is byte-for-byte the frame
    of 2026-09-09.
  * **Nothing else is removed.** Smaller objects (a 2-point gamma, a 5-point
    cluster on `028084_3/91`) and every vertex still draw.

**Unchanged-path proof.** `scan_harness.py context` on `039252_0/35` and
`039252_1/36`, without the new flags, is **byte-identical** before and after
the edit (`/home/xqian/tmp/h18/blindproof/`). `selftest_webgl_loss.py`
passes on the edited harness. The pre-existing `selftest_pin_persistence` PDHD
failure (check 10 uses a PDVD fixture) is unrelated and was left alone.

### 3.2 The frames

* **Volume.** 317 items × 8 frames = 2536 renders, 196 MB, from **4
  concurrent browsers in about 45 min** (~20 s per item per browser).
* **Canvas2D, not WebGL.** The forwarded `$DISPLAY` is unreachable, so
  chromium falls back to Canvas2D (doc pdvd/69 §8.3), and there is no WebGL
  context to lose. Every process recorded `backend: canvas2d`.
* **`check_shots.py` on all 317.**
  * 0 incomplete dirs and 0 blank (md5-shared) frames.
  * `c_3d_stop.png` unique colours: min **1511**, median **4935**, 0 below
    1000.
  * On PDVD WebGL the clean processes gave min 1168 and median 3877, so PDHD's
    Canvas2D frames sit well above the 1000 threshold. This is the first
    measurement of it on PDHD.
* **`h_dqdx_zoom.png`** (`mkzoom.py`): `g_dqdx.png` Lanczos-upscaled 3×. Every
  PDVD scanner had written its own crop-and-zoom helper. Checked by eye on
  `028084_3/91`: the last 5 cm reads unambiguously at 3× and the axes survive.
  The panel still clips at s ≈ +48 cm, as on PDVD.

## 4. The rubric

`pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md`, **ported, not copied**, from
the frozen PDVD rubric. Its last section tabulates every change with its
source. The main ones:
* the PDHD detector: cathode at x ≈ 0, the APA seam at z ≈ 231, the APA0 and
  APA2 charge deficits;
* verdict-blind;
* a precedence order: topology before a flat-looking profile, from the owner's
  PDVD re-judges;
* no absolute charge anchors (doc pdvd/55 §17.3);
* `straddles the stop` withdrawn;
* capture gammas on `STM_ONLY` in any direction;
* verdict-vs-kind consistency enforced by `mkv.py` (doc pdvd/55 §17.4).

| version | sha256 | frozen | used by |
|---|---|---|---|
| v1 | `7b8406b8…` | 2026-09-11 06:03 | the calibration wave (30 records) |
| v2 | `b0c3842e…` | 2026-09-11 06:25 | wave `w1` (52 records in the final record, plus the 23 it superseded) |
| v3 | `f0f695f0…` | 2026-09-11 06:52 | waves `w2`–`w4` with the 23 re-scans (235 records in the final record), and the double scan's 20 (`smx18/double_scan.tsv`) |

A first v3 draft (`4829aa6e…`, 06:52:29) still carried two stale v2 lines.
It was corrected and re-frozen as `f0f695f0` 28 s later, before any v3 wave
started. No record carries the draft's sha. v3's changes are §6.2.

Every record carries the sha of the rubric it was judged under (`mkv.py` reads
`ROUND/RUBRIC.sha`). The frozen text never changed while a wave was running.
The one exception is a 3-minute window, §6.2c.

## 5. The blind calibration: the agents against the owner's 30

Five agents took six of the owner's items each, blind to both `smx1` and the
chain's verdict, under rubric v1. `cmp_owner.py`:

| agent confidence | verdict | stopper-or-not | `michel_kind` |
|---|---:|---:|---:|
| **high** | **18 / 18** | **18 / 18** | 15 / 17 |
| medium | 3 / 11 | 6 / 11 | 2 / 7 |
| low | 0 / 1 | 0 / 1 | 0 / 1 |
| all | 21 / 30 | 24 / 30 | 17 / 25 |

* **Tags.** Tags on object keys both sides tagged agree 50/56. The owner tagged
  mostly the gamma specks and left Michel pieces at the chain's typing.
* **Kind.** The owner's `michel_kind` is not mechanical ("attached" with four
  gamma-tagged specks on `028084_0/97`), so the kind row compares two
  conventions.
* **Framing: the owner looked, the agents did not.** The owner's labels were
  taken with the chain's answer on screen; the agents' were not. Measured
  against the chain's `is_stm` on the same items, both agree on **21** of the
  scored items (27 owner, 28 agent), so neither side leans on the chain here.
* **The confidence field is calibrated, again.** As on PDVD (24/24 `high`),
  every `high` call reproduces the owner. The disagreements all sit in
  `medium` / `low`, which is where the owner's queue sends them.

The nine verdict disagreements, and what each taught the rubric:

| item | owner | blind agent | lesson → v2 |
|---|---|---|---|
| `028084_15/102` | STM_ONLY | THRU (medium) | mid-volume end, charge ends in all planes, flat profile → **a stop** (rule 3) |
| `028084_1/39` | STM_MICHEL | UNCLEAR (low) | the same, and the backward arm from the stop vertex is the Michel |
| `028084_17/55` | STM_MICHEL | STM_ONLY (medium) | an 8 cm arm from the stop running back along the body is **the Michel** |
| `028084_26/109` | STM_MICHEL | STM_ONLY (medium) | two clumps 3.2 cm out are **the Michel**, not detached dots (the ≤ 5 cm rule) |
| `028084_27/57` | THRU | STM_MICHEL (medium) | a collinear straight-on stub is **weak topology**; near-isochronous tracks excepted from rule 3 |
| `029107_10/50` | MESSY | STM_MICHEL (medium) | a stop beside a 316 cm cluster lying on the body is **MESSY** |
| `028084_23/114` | STM_ONLY | STM_MICHEL, pin 2.7 (medium) | a 2.6 cm soft tail after the peak; **not** turned into a rule (the owner moved PDVD pins by 1.2–9.5 cm) |
| `028084_29/38` | UNCLEAR | STM_ONLY (medium) | — |
| `029107_0/56` | MESSY | UNCLEAR (medium) | — (both unscored) |

* **One pattern the rubric carried in from PDVD was wrong for the owner:**
  "flat profile ⇒ `THRU`" wherever the track ends.
* **Where the owner does call `THRU`:** all four owner `THRU` items end at a
  face, at the APA seam, or on a near-isochronous track.
* **Where the owner calls a flat-profile item a stop:** both such items end
  mid-volume, and there `f_meas` shows the charge ending in every plane.

Rubric v2 folds in the five changes the table names. The calibration records
were not re-scanned: they stay in the record as the v1 blind calls, and the
owner's `smx1` verdicts stay the grading truth for those 30.

## 6. The scan

### 6.1 Waves, and what it cost

| wave | items | rubric | agents × items | records written (local time) |
|---|---:|---|---|---|
| `cal` | 30 (the owner's) | v1 | 5 × 6 | 06:04 – 06:18 |
| `w1` | 75 | v2 | 5 × 15 | 06:26 – 06:50 |
| `w2` | 60 + 23 re-scans | v3 | 5 × 16–17 | 06:54 – 07:15 |
| `w3` | 75 | v3 | 5 × 15 | 07:21 – 07:44 |
| `w4` | 77 | v3 | 5 × 13–16 | 07:47 – 08:14 |
| `dbl` | 20 (seeded re-scan) | v3 | 5 × 4 | §8 |

**Against PDVD tranche 2, the same instrument and the same five agents** (doc
pdvd/55 §13.2, findings §0b):

| | PDVD tranche 2 | PDHD `smx18` |
|---|---|---|
| items / records | 509 / 517 | 317 / 340 (incl. 23 re-scans) |
| frames | 5 browsers, 204 re-shot after WebGL loss | 4 browsers (Canvas2D), 0 re-shot |
| frame phase, wall | ~50 min + re-shoot | ~45 min |
| rubric churn | a whole wave (60 items) discarded | a calibration wave on purpose, 23 targeted re-scans |
| per agent | ~2.7 min / item, 12-item chunks, ~32 min | **~1.5 min / item**, 15–17-item chunks, 20–27 min |
| scanning, wall (scan waves only: calibration, frames and apply excluded on both sides) | ~5.5 h for 517 records (07:45–13:18 on 2026-09-09, the frozen-rubric waves; the discarded wave 06:55–07:35 not counted) | **1 h 50 min** for 310 records (`w1`–`w4`, 06:26–08:14, re-scans included; the double scan not counted) |
| apply (real widgets) | ~4 h after the scan (3–5 procs, OOM) | pipelined behind the waves, 4–6 procs |

**What made it faster**, in order of effect:
* **The pre-made 3× dQ/dx frame.** No scanner wrote a zoom helper.
* **Blind `context.json` plus the rubric's numbers-first design.** Scanners
  used `ends` and `seams_at_stop` instead of eyeballing the box.
* **Larger chunks.** 15–17 items per agent ran in ~280–330k agent tokens; none
  ran out.
* **The apply, pipelined.** Each finished wave's batch was put onto the real
  widgets while the next wave scanned. The PDVD round ran it after everything.

### 6.2 Two corrections to the instrument, found by the scanners

**(a) The anode is at |x| ≈ 352 cm, not 358 cm (rubric v3).** Three wave-1
scanners independently found fit ends clustered 5.9–7.1 cm "inside" the anode
face. Measured on all 634 ends of the 317 items:
* no end reaches past |x| = 352.2 cm;
* 46 stops and 10 entries sit at `face.x` 5–8 cm.

The anode wire planes are at |x| = 352.09 cm (the `AnodePlane` sensitive
volumes in the job log). But `smgeom.ENVELOPE["pdhd"]` puts the x faces at
±357.985 cm, so the red dashed box and every x number in `ends` overstate the
distance to the anode by about 6 cm. v2 read such ends as mid-volume flat
stops.
* The 23 wave-1 items at risk were re-scanned under v3: the 17 anode-band
  stops and the 6 other `FLAT_STOP` calls (one item is both).
* The re-scan changed 5 verdicts, all stopper → `THRU`, and all anode-band
  items: `028084_0/43`, `028084_11/116`, `028084_14/20`, `028084_16/110`,
  `028084_17/110`.
* The other 18 (all six `FLAT_STOP` calls among them) kept their verdict.
* The v2 records are kept and listed in `smx18/duplicates.tsv`.
* **Not fixed here, and it should be:** `smgeom.ENVELOPE["pdhd"]` still draws
  the box at 358 cm on the owner's display. *(Fixed in
  [doc pdhd/19](19_stm-michel-display-fix-and-review.md) §1; this round's
  records keep the numbers they were judged under, and `smx18/provenance.json`
  says so.)*

**(b) `f_meas` shows only this cluster's own cells.** `prep_stm_michel_scan.
proj_cells` reads one `T_proj_data` row: the muon's cluster, and only the
cells its fit touched. So the "measured" panels cannot show charge that the
clustering put in another cluster: a scanner saw a 99 cm unfitted cluster
inside the window that `f_meas` did not draw.
* v2 rule 3 had used `f_meas` for "nothing continues past the end". v3 moves
  that test to the 3-D frames, and the re-scan above covered the calls that
  relied on it.
* **Why waves 1–4 can be pooled.** Rule 3 (`FLAT_STOP`) is a judgement rule
  and the largest single driver of hand-stopper calls (38 items carry the
  prefix), and its load-bearing test changed between wave 1 and wave 2. All 6
  wave-1 `FLAT_STOP` calls were re-scanned blind under v3, and **6 of 6 kept
  their verdict**: the v2→v3 change to rule 3 is measured to have moved
  nothing. The other 52 wave-1 records were judged under v2's rule 3 and
  were not re-scanned. That result is the evidence they pool with waves 2–4.
  The anode correction is the change that did move verdicts (5 of 17).
* **The viewer's own text overstates the panel** ("that is exactly where an
  unreconstructed Michel shows up", README). That needs correcting, or the
  panel needs the event's other clusters; both are outside this round.
  *(Both done in [doc pdhd/19](19_stm-michel-display-fix-and-review.md) §2.)*

**(c) One procedural slip, recorded as it happened.** While writing v3, the v3
draft sat in the rubric's repo path for about 3 minutes. At that moment one
wave-1 scanner (`w1_a4`) had one item left. The frozen v2 text was restored
before it finished, and every `w1` record carries the v2 sha. If that scanner
re-read the rubric in those minutes, one record could have seen v3 text.

## 7. What the scan says about PDHD production

`d18_census.py`, graded on **bare production**: `p82bhoff` for the 303 items,
`h18b` for the 14 extras. Truth is the owner's verdict on the 30 items the
owner labelled, and the agent record elsewhere. `MESSY` and `UNCLEAR` are
unscored.

**Verdicts, 317 items:** 101 STM_MICHEL, 81 STM_ONLY, 90 THRU, 34 UNCLEAR,
11 MESSY. Three of them are `FRAG_*`.

| | TP | FP | FN | TN | purity | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| **`is_stm`, production (303)** | 60 | **0** | 112 | 87 | **1.000** | **0.349** |
| `is_stm`, owner + agent-`high` only | 45 | 0 | 31 | 44 | 1.000 | 0.592 |
| `is_stm`, without the `FLAT_STOP` calls | 59 | 0 | 77 | 87 | 1.000 | 0.434 |
| `is_stm`, the 14 cap extras (`h18b`) | 5 | 0 | 5 | 3 | 1.000 | 0.500 |
| `michel_found` on hand stoppers, production | 61 | 14 | 36 | 61 | 0.813 | 0.629 |

Scanned against the agents alone (`--agent-only`, i.e. without the owner's 30),
the numbers barely move: `is_stm` 60 / 0 / 113, efficiency 0.347.

**`score_stm_michel_scan.py --tag smx18 --key …_key_h18b.tsv`** (agent labels
on all 317). The `h18b` key equals production on the 303, by the §2.2 gate.
* **Every stratum is fully labelled (weights 1.00),** so these are
  *population* numbers for the 61-event sample, not sheet numbers.
* **The numbers:**
  * `is_stm` 65/65 pure, 65/183 = 0.355 efficient;
  * **STM + Michel jointly 22/28 = 0.786 pure, 22/101 = 0.218 efficient**;
  * 39 hand-`THRU` items carry a chain Michel (`michel_found = 1`, `is_stm = 0`).
* **The scorer's "REVEALED" heading is a schema artifact.** The apply writes
  every row through the viewer, which stamps `revealed_before_label = True`,
  but these scanners never saw the verdict (§3.1).

**The headline: PDHD production never calls a through-goer a stopper, and misses
between 40 % and 65 % of the stoppers**, depending on how strictly the hand
calls are read.
* **The misses are the shape tests.** 87 of the 112 missed stoppers are
  rejected *only* by `shape_flat` / `no_bragg` / `profile_sparse`.
* **49 of the 112 already carry `michel_found = 1`.** That is exactly the
  population PDVD's P1 (`topology_stop_evidence`, doc pdvd/70 §10) was built
  for, and it recovered 24 owner stoppers on PDVD at 0 new FP.
* **22 carry `plateau_off_mip`**: the APA0 charge deficit, seen by the chain.

By the stop's APA (production, `is_stm` efficiency):

| | APA0 | APA1 | APA2 | APA3 |
|---|---:|---:|---:|---:|
| items | 103 | 51 | 81 | 68 |
| efficiency | **0.132** | 0.345 | 0.489 | 0.467 |
| unscored (UNCLEAR + MESSY) | 26 | 5 | 10 | 3 |

APA0 is where the hardware fault is (doc pdvd/50 r2). The chain's shape tests
fail there most, and so does the scanners' profile reading (26 unscored).

**Pins and prefixes.**
* **Pins.** 18 were moved: 2.7–26 cm, median 8.8 cm. Seven moves exceed the
  largest pin the owner endorsed on PDVD (9.5 cm), and they are tier B of the
  queue.
* **Prefix counts (agent records):**

  | prefix | items |
  |---|---:|
  | `UNDERSHOOT` | 16 |
  | `FLAT_STOP` | 38 |
  | `NO_MEASUREMENT` | 35 |
  | `ANODE` | 45 |
  | `ISOCHRONOUS` | 45 |
  | `SEAM` | 14 |
  | `FACE` | 16 |
  | `DEAD` | 17 |
  | `REVERSED` | 17 |
  | `CATHODE` | 3 |

* **Tags:** 1252 muon, 199 michel, 267 gamma, 1258 delta / other.

**An open question the rubric does not settle: the direction of travel.**
* **The count.** 23 of the 182 hand stoppers stop at the *upper* end of their
  track, so a stop there needs an upward-going particle. 11 of them rest on
  `FLAT_STOP`.
* **The owner's own precedent.** Two of the 23 are the owner's own stoppers
  (`028084_1/39`, `028084_15/102`).
* **The chain on these.** It calls none of the 23 a stopper.
* **What the scanners did.** They split. Some called such ends `THRU` on the
  direction argument; others applied rule 3 as written.
* **Status.** It is the owner's call. The 16 not already in a higher tier are
  tier C of the queue.

## 8. Reproducibility — the seeded double scan

PDVD measured its label reproducibility by accident (a discarded wave and a
wave-list race, findings §0). Here it is measured on purpose.
* **The draw.** `nextwave.py --double 20 --seed 20260911` drew 20 items. The
  pool was the 212 items whose primary record was written under **v3**
  (waves 2–4), minus the 23 re-scans, which are already double-scanned.
* **The scan.** Five fresh agents each re-scanned four items under v3, blind
  to both the chain and the first scan.
* **Where the results go.** The re-scans never enter the record;
  `resolve.py` writes them to `double_scan.tsv`.

| | agreement |
|---|---:|
| verdict | **14 / 20** |
| stopper-or-not | 15 / 20 |
| `michel_kind` | 15 / 20 |
| per-object tag (rows both tagged) | 174 / 192 (91 %) |

| both scans said | verdict agreement |
|---|---:|
| `high` | **3 / 3** |
| at least one `medium`, none `low` | 9 / 12 |
| either `low` | 2 / 5 |

* **Every one of the six disagreements has a `medium` or `low` call on at least
  one side.**
* **By kind.** Two are `UNCLEAR` / `MESSY` vs a verdict (`029107_0/103`,
  `029107_8/44`, `028084_26/38`, counting the UNCLEAR↔MESSY swap). Two turn
  on the direction question of §7 (`028084_20/39`, `028084_29/101`). One is a
  seam call (`029107_19/123`: THRU vs a stop 8 cm past the seam).
* **A lower bound on a hard sample.** The PDVD accidental double scan
  agreed on 53/57 verdicts, but it was the first sequential slice of the sheet.
  This draw is random over waves 2–4, and 16 of its 20 primaries were `medium`
  or `low` (13 / 3; 4 `high`). The confidence field is calibrated here as on PDVD: the `high` rows
  reproduce, and the `low` rows are close to coin-flips. Read the record that
  way.

## 9. The record, the label file, and how they were checked

**The chain from record to artifact**, each step gated:
* `resolve.py`: one record per item. The 23 re-scans supersede their v2
  records, the 20 double scans stay out, and it refuses a missing item.
* `mkspec.py`: disjoint batches, asserted across batches.
* `apply_parallel.sh`: the real widgets, into a private labeldir per process,
  pipelined behind the waves.
* `fixup_spec.py`: every row compared with its record on verdict, kind,
  every tag and the pin.
* `merge.py --write` into the **new** tag `smx18`.
* `verify_scan_record.py`.

**The apply's own failure mode, measured.** On a loaded machine, 4 items with
17–26-object tables did not land in four passes (the table re-sorts under the
clicks, doc pdvd/55 §17.5), and each such failure stopped its process.
* **What it cost.** 14 rows needed a second apply: 4 with short tags and
  10 never reached.
* **The first retry.** It landed 9 of the 14.
* **The fix for the rest.** `scan_harness.py` gained `--tag-passes` (default
  4) and `--settle-scale` (default 1.0), and the last 5 landed with 10 passes
  at 2× settle.
* **Final state.** `fixup_spec.py` reports 0 rows to re-apply.

**Final gates:**

| gate | result |
|---|---|
| `merge.py --check` | 317 rows, 0 missing, 0 extra, every row = its resolved record |
| `merge.py --write` | `pdhd/work/stm_michel_labels/smx18/labels.json`, sha `f63a7034…` (the stale-tab guard will refuse to overwrite an owner edit) |
| `verify_scan_record.py --det pdhd --tag smx18` | **OK**: 317 records over 317 rows, verdict / kind / every one of 2976 tags / 18 pins |
| Phase-0 shas | `smx1/labels.json`, the d53h sheet and key, the PDVD records (21 files) and the `prep-pdhd` tree: all unchanged |
| PDHD production config | `pdhd/wct-pr-perevt.jsonnet` unchanged since `1d72f5fc` |
| the scan record's provenance | every record carries `arm`, `rubric_sha`, `scanner`, `wave`; `smx18/provenance.json` names arms, TLA, pin, rubric shas, gates |

## 10. The owner's review queue

`pdhd/docs/scan/pdhd_stm_michel_smx18_owner_queue.md`
(`campaign/mkqueue.py`). 230 of 317 items are queued, **in tiers**, and each
appears once:

| tier | what | items |
|---|---|---:|
| A | blind calibration vs the owner's own label | 9 |
| B | the scanner moved the stop, or read an undershoot | 31 |
| C | a stopper at the upper end of its track (§7) | 16 |
| D | a high-confidence call against production's `is_stm` | 18 |
| E | a low-confidence call | 20 |
| F | a medium call against production's `is_stm` | 46 |
| G | UNCLEAR / MESSY, the cap extras, other medium calls (keys only) | 90 |

Tiers A–E (94 items) are the ones where a second opinion moves a number.

**Items the scanners asked the owner to look at together:**
* `029107_19/106` + `029107_19/123`: the same wires and ticks, drawn 25–30 cm
  apart in x. Possibly one muon at two t0s.
* `029107_18/44` + `029107_18/45`: two parallel tracks whose fit ends share a
  V wire.
* `028084_6/97` + `028084_8/110`: the parallel-strand question, one decision
  for both.
* `029107_21/41` + `028084_16/110`: the same anode geometry, called opposite
  ways.

## 11. What this enables, and what is left

* **Grade the PDVD production knobs on PDHD against this record.** Each of the
  ~20 knobs PDVD runs and PDHD does not now has a PDHD record to be graded on,
  the same way docs 57–82 did on PDVD. Start with P1 (`topology_stop_evidence`
  + `topology_clears_sparse`): 49 of the 112 missed stoppers already carry
  `michel_found = 1`.
  * The grading instrument, `census_score.py` / `census_lib.py`, is
    PDVD-hardcoded (`MIP_MEDIAN = 47000`, the PDVD record path). It needs a
    `--det pdhd` port first, and `d18_census.py` is the minimal PDHD version.
    **No number in this doc went through `census_lib`.** Every figure comes from
    `d18_census.py`, `score_stm_michel_scan.py`, `cmp_owner.py` or
    `resolve.py`'s `double_scan.tsv`.
* **The owner's review of tiers A–E**, above all tier C (the direction
  question) and the seven pins beyond 9.5 cm. A ruling on the direction test
  would be the first v4 rubric change.
* **Fix the display's PDHD anode**: `smgeom.ENVELOPE["pdhd"]` x to ±352.1 cm
  (§6.2a). It changes the red box and every x distance the owner reads.
* **`f_meas` and the viewer README**: say that the measured panels are this
  cluster's own cells (§6.2b).
* **The draw no longer matters for the numbers.** All 317 items are
  labelled, so every stratum weight is 1.00. The numbers above are population
  numbers for the 61 events, subject to the prep's own filter (≥ 20 profile
  points, ≥ 10 cm, `has_pass`) and to the cap-64 candidate set.

## Files

| what | where |
|---|---|
| rubric (v3 as frozen) | `pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md` (sha `f0f695f0…`); v1 and v2 texts as frozen in `smx18/rubric_v1.md` / `rubric_v2.md`; the shas in `smx18/rubric_shas.txt` |
| census, scorer, gate outputs | `smx18/census_d18.txt`, `smx18/score_h18b.txt`, `smx18/gates_h18.txt`; the double scan in `smx18/double_scan.tsv` |
| the record (317) | `pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json`; header in `smx18/provenance.json` |
| sheet, keys, item-set diff | `pdhd/docs/scan/smx18/` (`…_key.tsv` is the scan arm's; `…_key_p82bhoff.tsv` / `…_key_h18b.tsv` grade) |
| calibration table, re-scans | `smx18/calib_vs_owner.tsv`, `smx18/duplicates.tsv` |
| owner queue | `pdhd/docs/scan/pdhd_stm_michel_smx18_owner_queue.md` |
| label file (the artifact) | `pdhd/work/stm_michel_labels/smx18/labels.json` (not tracked; the record is what makes it checkable) |
| tooling | `pdhd/stm_michel_scan/campaign/` (`AGENT_TASK.md`, `mkv.py`, `nextwave.py`, `shoot.sh`, `mkzoom.py`, `resolve.py`, `mkspec.py`, `apply_parallel.sh`, `fixup_spec.py`, `merge.py`, `mkrecord.py`, `mkqueue.py`, `cmp_owner.py`) |
| harness flags | `pdhd/stm_michel_scan/scan_harness.py` (`--blind`, `--hide-selection`, `--prepdir`, `--manifest`) |
| arms, gate, census | `pdhd/docs/scripts/h18_arms.sh`, `h18_gates.sh`, `d18_census.py` |
| scratch (not tracked) | `/home/xqian/tmp/h18/` — shots (196 MB), `v_parts/`, waves, logs, gates, backups |

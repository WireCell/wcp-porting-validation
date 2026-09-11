# 19 — PDHD STM + Michel: the two display defects fixed, and the owner queue reviewed

Doc pdhd/18 scanned all 317 PDHD STM candidates with five verdict-blind agents
(tag `smx18`). Its scanners found two defects in the display, and it left a
review queue for the owner (tiers A–E, 94 items). The owner asked (2026-09-11)
for the display to be fixed, and the queue then worked **by agents** (at most
three at a time), not by hand. Grading PDVD's production knobs on PDHD is next,
in a new session.

**Status.** Display: **fixed, gated**. Review round (tag `smx19`): §4–§6.
The owner's rulings, rubric v5 and a v5 double scan (tag `smx20`, now the
current record): §8.
No C++ and no production jsonnet changed; nothing here moves reconstruction
output.

---

## 0. Repro

```bash
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; H=/home/xqian/tmp/h19
S=$I/pdhd/stm_michel_scan; C=$S/campaign
# 1. the prep with the event's other charge (display arm h18s, doc 18's pin)
cd $S && ./prep_stm_michel_scan.py --det pdhd --arm h18s --ctx-cells \
    --outdir $S/prep-pdhd-smx19 --sheetdir $H/sheet_ctx \
    --pin-tranche ../../pdhd/docs/scan/pdhd_stm_michel_scan_sheet.tsv
#    gates: flag OFF is byte-identical; flag ON minus `proj_ctx` is byte-identical
./prep_stm_michel_scan.py --det pdhd --arm h18s --limit 3 --outdir $H/prep_off \
    --sheetdir $H/sheet_off --pin-tranche ../../pdhd/docs/scan/pdhd_stm_michel_scan_sheet.tsv
# 2. frames for the 94 queued items, fixed display, blind (3 browsers)
scan_harness.py shots --det pdhd --tag shots19 --labeldir <blank> --prepdir $S/prep-pdhd-smx19 \
    --manifest $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv --blind --hide-selection \
    --items-file $H/chunk_<n>.txt --out $H/shots_p<n>
python3 $C/mkzoom.py $H/shots; python3 check_shots.py $H/shots
# 3. blind review (rubric v4), then adjudication of the class changes
#    agents: $H/AGENT_TASK.md + items_rv<w>_a<i>.txt -> $H/v_parts/rv<w>_a<i>
X=$I/pdhd/docs/scan/smx19          # tiers.json and adj_ab_map.json were private during the round
python3 $C/cmp_review.py $H $I/pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json $X/tiers.json \
    --owner /home/xqian/tmp/h18/backup/pdhd_smx1_labels.json --nonblind $X/nonblind_named.txt \
    --disagree <outside the round>/adjudicate.tsv                  # -> $X/cmp_review.txt, $X/adjudicate.tsv
python3 $C/mkadj.py $H $I/pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json \
    <outside the round>/adjudicate.tsv 3 --map <outside the round>/ab_map.json   # -> $X/adj_ab_map.json
#    agents: $H/ADJ_TASK.md + $H/adj/adj_items_a<i>.md -> $H/v_parts/adj_a<i>
# 4. resolve (the adjudication supersedes) -> apply on the real widgets -> merge into smx19
python3 $C/resolve.py $H $H/items_review.txt --supersede $X/supersede.txt
python3 $C/mkspec.py $H $H/items_review.txt 3 rev
$C/apply_parallel.sh $H pdhd $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv $S/prep-pdhd-smx19 rev 3
python3 $C/merge.py $H pdhd smx19 $H/items_review.txt --base smx18 --write
python3 $C/mkreview_record.py $H $I/pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json \
    $I/pdhd/docs/scan/pdhd_stm_michel_smx19_verdicts.json $X/provenance.json --provenance-json $H/provenance_in.json
./verify_scan_record.py --det pdhd --tag smx19 --record ../../pdhd/docs/scan/pdhd_stm_michel_smx19_verdicts.json
python3 $I/pdhd/docs/scripts/d18_census.py --record $I/pdhd/docs/scan/pdhd_stm_michel_smx19_verdicts.json \
    --key $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv \
    --key-extras $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_h18b.tsv --shots /home/xqian/tmp/h18/shots
```

The reviewed record is served for the owner on **:5017**, with the fixed display
(grey cells and the true anode). It was first served on :5024, which the owner's
laptop did not forward, so the page showed empty. It moved to :5017, the port the
owner already uses, on 2026-09-11. After the owner's look (§8), both viewers
(:5017 smx19, :5023 smx18) were stopped at the owner's request; the command
below restarts one.

```bash
./serve_stm_michel_scan.sh 5017 --det pdhd --scan-tag smx19 \
    --manifest $I/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_sheet.tsv \
    --prepdir  $I/pdhd/stm_michel_scan/prep-pdhd-smx19
```

It was checked headless by reading the item selector from the Bokeh document
model: 317 items, 317 labelled. smx18 was served on :5023, on the display it
was scanned with.

---

## 1. Defect 1 — the PDHD box and every x number sat ~6 cm off the anode

**Symptom.** Wave-1 smx18 scanners found fit ends piled up 5.9–7.1 cm "inside"
the anode face (doc 18 §6.2a). Over all 634 ends, none reaches past
|x| = 352.2 cm.

**Root cause.** `smgeom.ENVELOPE["pdhd"]` said x = ±357.985 cm and cathode =
2.54 cm. The docstring claims "the sensvol union", but both are **fiducial-cut
numbers**: `FV_xmin/FV_xmax` of `pdhd/clus.jsonnet:61-62` and the per-face
`FV_xmax` (`clus.jsonnet:84`). The job's own sensitive volume, from its log:

```
<AnodePlane:apa0> face:0 with 3 planes and sensvol: [(-3520.95 76.1 2.34345) --> (-1.5875 6060 2302.36)]
```

That is |x| from **0.159** (the CPA face) to **352.095 cm** (the first
induction plane: `params.jsonnet` `apa_cpa − (0.5·apa_g2g − plane_gap)` =
357.34 − 5.2455). y and z were already the sensvol.

**Why it hid.** Nothing in the display is wrong at a glance: a 6 cm offset on a
720 cm box is 4 px in the projections. It surfaced only as a statistical pile-up
of `face.x` values at 5–8 cm.

**Fix.** `ENVELOPE["pdhd"]` = x ±352.095, cathode 0.159, with the log line and
the derivation in the docstring. **PDVD is unchanged**: its envelope was checked
against its own sensvol lines (±339.91 x, cathode 3.0, ±336.39 y,
0.813–298.435 z) and matches on every axis.

**Verification.**
* `028084_0/43`, one of the five smx18 stoppers that the anode correction
  turned into `THRU`: its stop now reads `face.x` 1.0 cm, where it read 6.9.
* The box sits on the anode in the projection frame.
* The PDVD `context` of `039252_0/35` and `039252_1/36` is **byte-identical**
  to doc 18's proof file (`$H/pvproof/post19.json` vs
  `/home/xqian/tmp/h18/blindproof/post_pdvd_context.json`).
* The smx18 records keep the numbers they were judged under.
  `smx18/provenance.json` gains a `display_envelope` block saying so, and
  saying that a re-shoot now gives different x numbers from the ones their
  evidence quotes.

## 2. Defect 2 — `f_meas` could not show charge outside the fitted cluster

**Symptom.** A scanner saw a 99 cm unfitted cluster inside the ±150 window
that the measurement panels did not draw (doc 18 §6.2b). The README said the
panels are "exactly where an unreconstructed Michel shows up".

**Root cause.** `prep_stm_michel_scan.proj_cells` reads one `T_proj_data` row:
the cells this cluster's fit touched. `T_proj_data` holds one row per *fitted*
cluster (15 of 438 clusters on `028084_0`), and `T_proj` is empty on PDHD. So
nothing the fit did not touch could reach the panel.

**Fix — the imaging's ctpc, which already holds every live cell of the
event.** The pctree tarball each PR job writes carries the ctpc whole
(`ctpc_a<A>f<F>p<U|V|W>`, `PointTreeBuilding.cxx:293-327`), and each entry
carries two fields:
* **the channel ident**, which maps to the panel's global channel rank with
  `ChanScheme`'s own rule (`PdvdPrMagnifyTrackingVisitor.cxx:138-177`,
  re-derived from the wires file and asserted against `smgeom.NCH`);
* **the slice's start tick**, which maps to the panel's time slice as
  tick // 4.

No geometry and no drift model is involved.

`prep_stm_michel_scan.py --ctx-cells` (default **OFF**) adds `proj_ctx` to
each payload. It holds every live cell within ±200 channels and ±200 slices
of the fit's rr = 0 end that this cluster's own row does not carry.

The viewer draws those cells in the `measured` column as **grey squares**:
* on their own scale, which is the plane's p90 over the 317 payloads (18.1k /
  19.7k / 27.6k e);
* under the coloured cells;
* with a dashed box marking the coverage.

A payload without the key draws nothing, as before. The panel note says what
the grey is.

**The causal gate — the fitted cells must land on ctpc cells.** Over the 317
items, **99.88 % (U), 99.74 % (V) and 99.83 % (W)** of the fitted clusters'
own cells land on a ctpc cell of the same channel and slice. The rest are
dead-region fillers, which the ctpc does not carry. W charges agree cell for
cell (ratio 1.000, p5–p95).

**A third finding, labelled rather than changed: U/V cells in `T_proj_data`
count a wrapped channel's charge 2–3 times.** On `028084_0` the ctpc-to-row
charge ratio is 1/3 (p5–p75) and 1/2 (p95) on U and V, and exactly 1 on W.
`write_proj_data` keys a cell by global channel, but it accumulates the
fitter's per-*wire* snapshot. A wrapped channel whose charge the fitter put
on two or three wire segments is therefore summed two or three times.

Measured and predicted are both summed, so the residual is consistent. The
coloured U/V scale is simply not the channel's charge. This is a C++ output
convention, so it is **reported, not changed**. The panel note, the README
and rubric v4 say so, and the grey is always the channel's charge counted
once.

**Additivity gate.**
* Flag OFF: 13 of 13 payloads from a 3-event prep are **byte-identical** to
  `prep-pdhd-smx18`.
* Flag ON: all **317 of 317** payloads, with `proj_ctx` removed, are
  **byte-identical** to `prep-pdhd-smx18`.
* Payloads grow from 171 to 231 MB. The median item carries 9.4k context
  cells (p90 20.8k, max 57.5k).

**Seen on the first frames.**
* `028084_3/91`: another track crosses the stop region in U and W, and the
  item's own horizontal track continues to channel ~6200 (V) and ~9200 (W),
  past the fit.
* `029107_18/79` (queue tier C): the grey cells carry the track's own line
  well past the fit end in all three planes. The "stop" at the track's upper
  end is where clustering split one track.

**Tests.**
* `selftest_smx3d_browser.py`: 102 passed, 0 failed. It includes the
  nine-panel paint gate and its causal control.
* `selftest_stm_michel_scan.py`: fails only on PDVD tags `smx3` / `smx4` /
  `smx5`. Those labels "no longer sit where the sheet puts them", because
  PDVD's option scans were labelled against their own manifests, not the
  default sheet. No PDVD sheet, label or code path was touched here. The
  check reads PDVD data files only.

## 3. A production observation for the owner (not changed)

The PDHD tagger fiducial box, `pr.jsonnet` `pdhd_pr_fv_box` (and
`clus.jsonnet` `FV_x`), puts x at **±357.985 cm**, commented "where the
sensitive volume starts". The job's own sensitive volume ends at 352.095.

With production's `tgm_fv_x_margin` = 2.5 cm, the taggers' anode-side x cut
sits at |x| = 355.5 cm. Every reconstructed point lies inside 352.2, so **the
anode-side x cut never fires**. A track leaving through an anode ends 3.4 cm or
more inside what the taggers treat as the fiducial volume.

PDVD's box matches its sensvol. Changing PDHD's would change production
output, so it needs its own knob and A/B (CLAUDE.md §5 rule 1).

## 4. The review round (tag `smx19`) — design

**Items.** Doc 18's queue tiers A–E: **94 items** (A 9, B 31, C 16, D 18,
E 20). They are shot again on the fixed display and scanned blind by three
agents at a time, in two waves of 3 × ~16:
* `shots19`: 3 browsers, Canvas2D, `check_shots.py` clean;
* 94/94 complete, 0 blank frames, `c_3d_stop` colours min 1511 / median 5158;
* no `is_stm`, `reject_names` or `in_fv` in any `context.json`.

**Blindness.** The reviewers get a **flat, shuffled item list**
(seed 20260911), with no tier: tiers B–D are *defined* by the chain's
`is_stm`, so a tier label would hand it over. The tier map lives outside the
round dir. The reviewers are forbidden the previous round's frames, records,
queue and doc, and see neither the chain's verdict nor the previous scan's.

**Tier A is scored apart.** Those are the owner's own 30-item calibration
items. The owner's `smx1` label stays the grading truth there and is never
overridden, so tier A is kept out of every pooled number.

**Rubric v4** (`2bf6ff5c`, frozen 10:46 before any reviewer started, and read
from the round dir, never the repo path) changes three things:
* the display section: the true anode and cathode, the grey cells, and the
  U/V ×2–3;
* rule 3's continuation test, which now includes the grey cells;
* **rule 7, direction**, which is new. Cosmic muons travel down. When the fit
  end is the track's upper end (`dy ≥ 0.3 × chord`), a flat profile there is
  where the track *starts*, and only a clear Bragg rise makes it a stop (at
  most `medium`, `DIRECTION:`).

  The rule is written for this round, not taken from an owner ruling. It
  answers doc 18's open direction question, and **the owner can overrule
  it**. The 0.3 cut (~17°) is where geometry stops being able to tell up from
  down.

  **Every upper-end hand stopper of smx18 is in the queue:** 18 of 18 at
  `dy ≥ 0.3 × chord` (A 1, B 3, C 14).

**A leak found by the reviewers.** The rubric's v2 change table and rule 6
name 7 of the 9 tier-A items together with the owner's verdict. Two
reviewers met such an item and said so in their notes (`NOT BLIND:`). Tier A
is therefore **not owner-blind** where a reviewer recognised the key. It is
scored apart anyway (above).

**Adjudication.** Where the review and smx18 put an item in a different class
(stop / through / unscored), a third agent is given both scans as "A" and
"B" in seeded random order, without being told which is older. The adjudicator
still does not see the chain's verdict, and decides from the fixed frames
(`ADJ_TASK.md`, `mkadj.py`). The adjudication supersedes the blind review in
the resolved record. The blind call is kept in the record's `review.blind`
block.

## 5. The blind review against smx18

**Cost.** 94/94 records, six agents in two waves of three, 15–16 items each:
* 23–29 min and 305–352k agent tokens per agent;
* about 1.6–1.9 min per item, a little slower than smx18's 1.5, with v4's
  two extra reads (rule 7, the grey cells).

**What the comparison measures.** The display changed, the rubric changed
(v4) and the scanner changed, all at once. So the agreement below is **change
plus noise**, not a reproducibility number. The reproducibility baseline is
doc 18's seeded double scan: 14/20 verdicts, and 3/3 where both calls were
`high`.

`cmp_review.py` output, pooled over tiers B–E. Tier A is left out (the owner's
items). So is `029107_19/111`, the one pooled item the rubric names; it agrees
anyway:

| | verdict | stopper-or-not | `michel_kind`, same verdict |
|---|---:|---:|---:|
| **pooled B–E** (n = 84) | 51 / 84 | 49 / 69 | 43 / 44 |
| old call `high` | **29 / 29** | 29 / 29 | 28 / 29 |
| old call `medium` | 11 / 23 | 12 / 22 | 11 / 11 |
| old call `low` | 11 / 32 | 8 / 18 | 4 / 4 |
| **both calls `high`** | **18 / 18** | 18 / 18 | 17 / 18 |

| tier | n | verdict | stopper-or-not |
|---|---:|---:|---:|
| B — pin moved / undershoot | 31 | 24 / 31 | 25 / 29 |
| C — stop at the track's upper end | 16 | **0 / 16** | **0 / 16** |
| D — `high` hand call against the chain | 18 | **18 / 18** | 18 / 18 |
| E — `low` call | 20 | 10 / 20 | 7 / 7 |

**The confidence field is calibrated, a third time.** Every smx18 `high` call
survives:
* a new scanner,
* a new rubric,
* a display that now shows the anode correctly and every other cluster's
  charge.

Every change sits in `medium` and `low`, as on PDVD (24/24) and in doc 18's
calibration (18/18).

**Tier D stands.** The 18 items where a `high` hand call said "stopper" and
production's `is_stm` said no reproduce 18 of 18, kinds 17 of 18. These are
stoppers production misses. That is the population the PDVD knobs were
flipped for.

**Tier C falls, all 16.** Every hand stopper whose fit end is the upper end of
the track is now `THRU`:
* rule 7 (`DIRECTION:`) decides most of them;
* on several, the grey cells show the track carrying on past the fit end in
  all three planes (`CONTINUES:`).

`029107_18/79` (§2) is the example. Across the 94 records, `DIRECTION:` is
used 25 times and `CONTINUES:` 17 times. smx18 had neither.

**Where the pooled changes go** (old → review):

| old → review | n |
|---|---:|
| a stopper → `THRU` | 20 (10 `STM_ONLY`, 10 `STM_MICHEL`) |
| `UNCLEAR` → `THRU` | 7 |
| `UNCLEAR` → `STM_ONLY` | 2 |
| a stopper → `UNCLEAR` | 3 |
| `STM_MICHEL` → `STM_ONLY` | 1 |

**No item moved from `THRU` to a stopper.** `FLAT_STOP:` falls from 12 to 3
records, and `OVERSHOOT:` from 17 to 11.

### 5.1 Tier A — the owner's items, against the owner

The owner's `smx1` label is the truth here and is **not** changed. The review
agrees with it on 3 of 9, where the old blind calibration agreed on 0 of 9
(these are its nine disagreements by construction):

| item | owner | old blind (v1) | review (v4) |
|---|---|---|---|
| `028084_17/55` | STM_MICHEL | STM_ONLY | **STM_MICHEL** = owner |
| `028084_26/109` | STM_MICHEL | STM_ONLY | **STM_MICHEL** = owner |
| `028084_27/57` | THRU | STM_MICHEL | **THRU** = owner |
| `028084_23/114` | STM_ONLY | STM_MICHEL, pin 2.7 | STM_MICHEL, pin 2.9 |
| `028084_1/39` | STM_MICHEL | UNCLEAR | **FRAG_THRU** (high) |
| `028084_15/102` | STM_ONLY | THRU | **THRU** |
| `028084_29/38` | UNCLEAR | STM_ONLY | FRAG_THRU |
| `029107_0/56` | MESSY | UNCLEAR | THRU |
| `029107_10/50` | MESSY | STM_MICHEL | THRU |

Five of these seven keys are named in the rubric with the owner's verdict
(§4), so their agreements are not independent.

**The two to put to the owner.** `028084_1/39` and `028084_15/102` are the two
items that *justified* rule 3 (`FLAT_STOP`, v2 change #10). Both are the
**upper end of a steep track** (dy/chord 0.84 and 0.85).
* On `028084_1/39` the grey cells carry the track's own line on past the fit
  end in U, V and W at matching slices. The reviewer called it `FRAG_THRU`
  **high**.
* `028084_15/102`'s other end sits 0.4 cm from the **floor**. A cosmic
  entering there would be going up.

The owner labelled both on a display that drew neither the grey cells nor a
direction. If the owner's reading changes, rule 3 loses its evidence.
**26 of smx18's 38 `FLAT_STOP` calls were not in the queue**, so they have
not been checked against the grey cells (§7).

## 6. Adjudication, and what the record now says about production

**Adjudication.** 32 items changed class between smx18 and the review, tier A
excluded (`smx19/adjudicate.tsv`: B 6, C 16, E 10). Three agents, 10–11 items
each, saw the two scans as A/B in seeded random order:
* **they kept the review's call on 26, smx18's on 5, and neither on 1**;
* none ended as a stopper: 25 through-going, 7 unscored;
* confidence: 6 `high`, 21 `medium`, 5 `low`.

| smx18 → review → adjudicated | n |
|---|---:|
| stop → through → **through** | 20 |
| unscored → through → through | 4 |
| unscored → through → **unscored** | 3 |
| stop → unscored → unscored | 2 |
| unscored → stop → **unscored** | 2 |
| stop → unscored → through | 1 |

| tier | kept review | kept smx18 | neither |
|---|---:|---:|---:|
| B | 5 | 0 | 1 |
| C | **16** | 0 | 0 |
| E | 5 | 5 | 0 |

In tier E the adjudicators split evenly. Every smx18 call they restored is
`UNCLEAR` (low-confidence isochronous or unreadable items), never a stopper.

**Adjudicator a0's warning, stated as it was made.** In 7 of its 11 items, the
scan that had called a stopper applied no direction test, and in 2 the other
scan never mentioned the grey cells. "Much of the move to `THRU` is
mechanical from v4", and rule 7 has no owner calibration behind it. So a
packet where every contested stopper becomes `THRU` is exactly where
over-applying rule 7 would show. That is why §7 puts rule 7 first.

**Integrity.** The session scratchpad turned out to be shared with the
subagents. Two adjudicators wrote generic-named `f_meas` crops there, and one
of them briefly saw the other's panels. The timestamps show that no
adjudication record rests on another item's crop: a1 noticed and re-cropped
privately; a2 read only its own.

A path audit of all nine transcripts (six reviewers, three adjudicators)
counts **0** accesses under `/home/xqian/tmp/h18/`, `pdhd/docs/scan/` or
`pdhd/work/`, and 0 references to the private tier and A/B maps.
`smx19/objections.md` has the detail. Next round: give each agent a private
scratch dir in its brief.

**The record.** `pdhd_stm_michel_smx19_verdicts.json`: 317 records.
* **94 reviewed** (32 adjudicated), each with a `review` block that keeps the
  smx18 call and the blind call.
* **223 copied verbatim** from smx18.
* The owner's `smx1` label still decides the 30 owner items.

**The labels behind it.**
* The 94 reviewed items were put on the real widgets: 3 processes, 94 saved,
  **0 did not land**, 11 pins moved.
* `merge.py --base smx18` gated every row against its resolved record, then
  wrote the new tag `smx19`. It holds 317 rows: 94 from this round, and 223
  carried from smx18 (sha `f63a7034…`, unchanged).
* The `smx19` labels have sha `8ae742b4…`.
* `verify_scan_record.py --tag smx19` passes: "the artifact says what the
  record says, on every field the doc publishes" (12 pins, 2976 tags).

**What changes in the grading of production** (`d18_census.py`, bare
production `p82bhoff` + `h18b`; truth = owner where labelled, else agent):

| | smx18 | **smx19** |
|---|---|---|
| verdicts | 101 SM / 81 SO / 90 THRU / 34 UNCL / 11 MESSY | 88 SM / 71 SO / **115 THRU** / 32 UNCL / 11 MESSY |
| hand stoppers (all 317) | 182 | **159** |
| `is_stm`, production 303 | 60 TP / **0 FP** / 112 FN — purity 1.000, eff **0.349** | 60 TP / **1 FP** / 89 FN — purity **0.984**, eff **0.403** |
| `is_stm`, owner + agent-`high` | eff 0.592 | eff **0.662** |
| `is_stm`, without `FLAT_STOP` calls | eff 0.434 | eff 0.480 |
| `michel_found` on hand stoppers | purity 0.813, eff 0.629 | purity 0.806, eff 0.643 |
| hand stoppers at the track's upper end (dy > 0) | 23 of 182 | 4 of 159 |

In short: the review removes 23 hand stoppers the chain never took. None of
the 60 production `is_stm` successes moves. So efficiency rises and the
denominator shrinks, and tier D's 18 misses, which production still makes,
all stand.

**Production's first false positive, and why it is contingent.**
`029107_4/56` (tier E) is `is_stm = 1` with `michel_found = 1` in the chain.
* smx18 called it `UNCLEAR` low;
* the review and the adjudicator call it `THRU` medium.

The fit end is the upper end of a steep track whose other end is 5.4 cm above
the floor. The end sits in U and W dead bands, so its profile is unreadable.
It is therefore decided by exactly the two conflicts the adjudicators flagged
as unranked in v4:
* rule 7 against rule 4 (an unreadable profile);
* rule 7 against rule 1 (decay activity at an upper end — here the chain
  found a Michel).

**It is a false positive only if the owner keeps rule 7 as written.** It
heads the owner's list.

## 7. What is left, and what the reviewers found in the rubric

**For the owner, in order of how much they move:**
1. **Rule 7 (direction), and the two rule-3 exemplars** (§5.1). *Ruled by the
   owner on 2026-09-11 (§8): rule 7 stands, rule 1 outranks it, and both
   exemplars are withdrawn.* Keeping rule 7
   confirms tier C's 16 `THRU` calls. The owner's own `028084_1/39` and
   `028084_15/102` then need a second look on the fixed display. Three items
   show how far rule 7 reaches, all on the smx19 viewer:
   * **`029107_4/56`**: production's only false positive (§6). It rests on
     rule 7 at an unreadable end, where the chain found a Michel.
   * **`028084_18/17`**: the weakest direction call (dy/chord 0.41).
   * **`029107_15/26`**: an angled arm at an upper end, rule 1 against rule 7.

   The two rankings v4 leaves open are rule 7 vs rule 4 (unreadable profile)
   and rule 7 vs rule 1 (activity at an upper end). They decide these three
   items and most of the adjudication.
2. **The 26 unqueued `FLAT_STOP` calls.** They are smx18 stoppers whose stop
   rests on rule 3, and nobody has checked them against the grey cells. A
   16-item-per-agent pass over them is the cheapest way to settle the record
   before grading on it.
3. **The PDHD tagger FV box** (§3). It needs a knob and an A/B, not a scan.

**Next (a new session, as the owner asked):** grade PDVD's production knobs on
PDHD against the `smx19` record, P1 (`topology_stop_evidence`) first.
`census_score.py` / `census_lib.py` still need a `--det pdhd` port. `d18_census.py`
is the minimal one, and no number here went through `census_lib`.

**Rubric points the six reviewers raised** (logged, not folded in: v4 stayed
frozen through the round; `$H/objections.md` has each with its item key):
* **The rubric names answers.** Its change history and rule 6 name nine keys
  with the owner's verdict. Seven are tier A, and one, `029107_19/111`, is
  pooled. v5 should cite rules without keys, or move the examples out of the
  scanner's text.
* **The grey test on near-isochronous tracks.** The track's own continuation
  and unrelated charge at the same drift time are the same pixels. The same
  holds for a track parallel to the W wires, and for a continuation that
  leaves the frame's channels (across the seam or the cathode).
* **Grey is raw (channel, tick).** A track from another flash can line up with
  a stop without being near it in 3-D, so three-plane alignment is necessary,
  not sufficient.
* **Rule 7's edges.**
  * It has no minimum chord: it fired on a 10 cm chord and nominally on a
    5 cm hook.
  * Its "upper end reaches a face" premise fails when both ends are inside
    the volume.
  * It is silent on an upper end that has a clear rise *and* a grey
    continuation (the continuation won).
* **The thin horizontal line in `f_meas`.** It also appears in the predicted
  and difference columns, so it is the fit's projection on a wrapped plane,
  not charge. Rule text v3 reads it as isochronous charge.
* **Smaller points:**
  * `pin_rr` is arc length; two reviewers first wrote a straight-line
    distance;
  * the APA0/APA2 smooth wave is neither "rise" nor "flat";
  * the ~10 cm Michel/gamma and 60 cm gamma edges were hit exactly;
  * size-0 clumps inside 5 cm collide between rule 13 and the degenerate-row
    rule;
  * there is no prefix for the far end, or for the readout-window edge;
  * a Michel inside one fitted row cannot be recorded;
  * a true stop can lie off the chain, in a side branch.

## 8. The owner's rulings, rubric v5, and a v5 double scan (tag `smx20`)

```bash
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; H=/home/xqian/tmp/h20
C=$I/pdhd/stm_michel_scan/campaign; X=$I/pdhd/docs/scan
K="--key $X/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv --key-extras $X/smx18/pdhd_stm_michel_scan_key_h18b.tsv --shots /home/xqian/tmp/h19/shots"
# the census script's new owner_review precedence: its smx19 output is byte-identical before and after
python3 $I/pdhd/docs/scripts/d18_census.py --record $X/pdhd_stm_michel_smx19_verdicts.json $K   # = $X/smx20/census_smx19_baseline.txt
# the owner's rulings alone
python3 $C/mkowner_record.py $X/pdhd_stm_michel_smx19_verdicts.json $X/smx20/owner_rulings.json $H \
    $H/interim_owner_only.json $H/interim_prov.json --skip-v5
python3 $I/pdhd/docs/scripts/d18_census.py --record $H/interim_owner_only.json $K   # = $X/smx20/census_owner_rulings_only.txt
# the v5 double scan: agents rv5_a1, rv5_a2 ($H/AGENT_TASK.md, $H/items_a<i>.txt -> $H/v_parts/rv5_a<i>), then
python3 $C/mkowner_record.py $X/pdhd_stm_michel_smx19_verdicts.json $X/smx20/owner_rulings.json $H \
    $X/pdhd_stm_michel_smx20_verdicts.json $X/smx20/provenance.json --provenance-json $H/provenance_in.json
python3 $I/pdhd/docs/scripts/d18_census.py --record $X/pdhd_stm_michel_smx20_verdicts.json $K   # = $X/smx20/census_smx20.txt
# labels: the adopted items onto the real widgets, into the new tag smx20 on top of smx19
python3 $C/mkspec.py $H $H/items_adopted.txt 1 v5
$C/apply_parallel.sh $H pdhd $X/smx18/pdhd_stm_michel_scan_sheet.tsv $I/pdhd/stm_michel_scan/prep-pdhd-smx19 v5 1
python3 $C/merge.py $H pdhd smx20 $H/items_adopted.txt --base smx19 --write
cd $I/pdhd/stm_michel_scan && ./verify_scan_record.py --det pdhd --tag smx20 --record ../../pdhd/docs/scan/pdhd_stm_michel_smx20_verdicts.json
```

### 8.1 What the owner looked at, and the rulings

The owner looked at five smx19 items on the fixed display (:5017) and answered
three questions in the session chat, without clicking labels. Their words, verbatim:
*"Q1: these may not be soppers: Q2: it seems to be STM + Michel Q3: looks like
STM topology and #46 not likely STM, I guess."*

| # | item | question | before | owner now |
|---|---|---|---|---|
| 9 | `028084_1/39` | Q1: should rule 7 exist? | owner smx1 `STM_MICHEL`; smx19 `FRAG_THRU` high (grey continuation, upper end) | not a stopper |
| 35 | `028084_15/102` | Q1 | owner smx1 `STM_ONLY`; smx19 `THRU` medium (flat upper end of a steep track) | not a stopper |
| 270 | `029107_4/56` | Q2: rule 7 vs rule 4 at an unreadable upper end | smx18 `UNCLEAR`; smx19 `THRU` (production's only false positive) | `STM_MICHEL` |
| 184 | `029107_15/26` | Q3: rule 7 vs rule 1, an angled arm at an upper end | smx18 `STM_MICHEL`; smx19 `THRU` | `STM_MICHEL` |
| 46 | `028084_18/17` | Q3: the shallowest rule-7 call (dy/chord 0.41) | smx18 `STM_MICHEL`; smx19 `THRU` | not a stopper |

What the rulings say about the rubric:
* **Rule 7 stands for flat upper ends** (#9, #35, #46). Two of these are the
  owner's own smx1 stoppers. This is the first time the owner has revised an
  earlier call of their own, so the record keeps both values.
* **Rule 1 outranks rule 7** (#184, and #270, which the owner read as
  topology): a clear decay arm at an upper fit end makes it a stopper. This
  overrules v4 rule 7's second bullet ("activity at that end is not enough").
* **Rule 7 against rule 4 is not ruled.** #270 was settled on topology, so an
  upper end with no topology and no readable profile is still open. By their
  notes, four reviewed items are that case: `028084_15/28`, `029107_15/40`,
  `029107_18/59`, `029107_21/64` (all `THRU` in smx19).

**How the rulings are stored.** `smx20/owner_rulings.json` holds the words, the
question each answers, and the reading of each answer. `mkowner_record.py`
writes them to the record as an `owner_review` block and changes nothing else.
The agent verdict stays in place. `d18_census.py` now takes `owner_review`
first, then `owner_smx1`, then the agent. The owner gave no `michel_kind` for
the two new stoppers, so they count in `is_stm` and are left out of the
`michel_found` tally. They are neither scored as "no Michel" nor filled in from
the chain. `smx1/labels.json` is not written (M13). The labels stay the
agents'; the owner's call lives in the record.

**One edit the look made.** The smx19 viewer writes every change to disk,
and it has no read-only mode. During the owner's look, one smx19 label row
changed:
* **The row.** `029107_15/26` (#184) gained a placed pin at rr 3.0 cm, 1.91 cm
  from the fit end. The label stayed `THRU`.
* **When.** The file's mtime is 14:17:41, inside the owner's session.
* **Extent.** Rebuilding the smx19 labels from smx18 plus the round's three
  apply label dirs shows no other row changed.

The pin is the owner's, so it is kept (M13). It is not known whether it marks
the stop of the owner's `STM_MICHEL` call, so it is stored as
`owner_review.app_edit` and used nowhere. Two consequences:
* `verify_scan_record.py` now reports this row as its one mismatch. That holds
  on smx19, which was clean when doc 19 was committed, and on smx20, which
  carries the row over.
* A viewer served for a look should serve a copy of the tag, not the record's
  own tag.

### 8.2 Rubric v5

`pdhd_stm_michel_scan_rubric.md` v5 (sha `750751ea…`; the v4 text is kept as
`smx20/rubric_v4.md`) makes rows 26–29 of its change table:
* **Rule 7's second bullet.** At an upper end, topology decides as in rule 1.
  A clear arm or piece leaving the fit end at an angle makes it `STM_MICHEL`.
  A straight-on stub, or the track's own line continuing in grey, is not
  topology. A clear rise with no topology makes it `STM_ONLY`. In both cases
  the call is at most `medium`, with `DIRECTION:`.
* **Rule 3 loses both of its owner examples.** The `FLAT_STOP` rule was written
  from #9 and #35, and the owner now calls both not stoppers. The rule now
  rests on the argument alone and applies only at a lower end. This makes the
  26 unreviewed smx18 `FLAT_STOP` calls (§7 item 2) more pressing: every one is
  a stopper whose only support is this rule.
* **The unruled case gets a marked default.** An upper end with no topology and
  no readable profile is called as in v4 (`THRU`, `DIRECTION:`) with
  `NO_MEASUREMENT:` added, so it can be re-graded once the owner rules.
* **No item keys in the rubric.** v4 named nine keys with the owner's verdict
  (§4). v5 cites "calibration items" instead, and a regex check confirmed no
  key is left.

### 8.3 What the rulings alone change in the census

Production (303 items), bare-production key, from `census_smx19_baseline.txt`
and `census_owner_rulings_only.txt`:

| | TP | FP | FN | TN | purity | efficiency |
|---|---|---|---|---|---|---|
| smx19 | 60 | 1 | 89 | 111 | 0.984 | 0.403 |
| + the owner's five rulings | 61 | **0** | 88 | 112 | **1.000** | 0.409 |

Item by item:
* #270 moves from FP to TP; production's only false positive is gone.
* #184 moves from TN to FN, a stopper production misses (reject `plateau_off_mip`).
* #9 and #35 move from FN to TN.
* #46 does not change.

The hand-stopper count stays at 159 (two in, two out). Owner plus agent-high
efficiency goes from 0.662 to 0.676. `michel_found` loses the two withdrawn
owner stoppers (TP 54 → 53, FP 13 → 12). The two new owner stoppers have no
kind and are left out of that tally.

### 8.4 The v5 double scan

**Which items.** The smx19 round called 25 reviewed items not-a-stopper under
rule 7. Remove the owner's five and tier A `029107_10/50` (owner `MESSY`, which
stands), and 19 are left. By their notes, five describe an arm or piece
leaving the upper fit end at an angle, the rule-1 topology that v4's second
bullet overruled:
* two clear: `029107_3/23`, a 5.3 cm arm folding back from the fit end at
  about 30°, plus a detached piece; `029107_1/32`, a short hook leaving the end
  sideways;
* three borderline: `029107_24/47`, pdg-11 forks after a peak 7 cm past the fit
  end; `029107_1/36`, a backward arm from a vertex 8.75 cm back; `029107_5/93`,
  a 20 cm pdg-11 line leaving at a 40° kink.

The other 14 carry no topology claim: bare ends, straight-on stubs, or a
three-plane grey continuation. Two that look close, `028084_2/26` and
`028084_20/39`, are straight-on continuations with grey past the end. Rule 1
already excludes those, so v5 cannot change them.

**Design.** Two verdict-blind scanners each judged all five items under v5,
in independently shuffled orders (seeds 2026091101 and 2026091102):
* each had a private out dir and a private scratch dir, the lesson of the v4
  round's crop collision;
* the frames were the v4 round's blind frames, linked one by one into
  `/home/xqian/tmp/h20/shots`;
* h18 and h19 were off limits.

Each item's outcome follows its stop / thru / unscored class:
* **confirmed**: both scans agree with smx19;
* **adopted**: both scans agree on a different class, and the record is rebuilt
  from one scan;
* **split**: the scans disagree; smx19 is kept and the item goes to the owner
  queue.

**Result.** Both scanners' audits are clean: 50 and 51 tool calls, with no
forbidden path. The audit script flags a synthetic read of h19 and passes a
read through `<SHOTS>`, so its zero can be trusted.

| item | smx19 | `rv5_a1` | `rv5_a2` | outcome |
|---|---|---|---|---|
| `029107_3/23` | `THRU` | `STM_MICHEL` both | `STM_MICHEL` both | **adopted** (a1) |
| `029107_1/32` | `THRU` | `STM_MICHEL` attached | `STM_MICHEL` attached | **adopted** (a1) |
| `029107_1/36` | `THRU` | `THRU` (+`NO_MEASUREMENT:`) | `THRU` (+`NO_MEASUREMENT:`) | confirmed |
| `029107_24/47` | `THRU` | `STM_MICHEL` attached, `UNDERSHOOT:` (a peak ~7 cm into 47010, then the forks as the electron) | `THRU` (47010 is the track's own line; the fall is APA2 coverage) | **split**, owner queue |
| `029107_5/93` | `THRU` | `FRAG_THRU` (the 40° arm hands off across the cathode to 92017) | `STM_MICHEL` (the arm is the incoming muon; the chain has the muon and the electron swapped) | **split**, owner queue |

* **The two clear topology items** became stoppers on agreement.
* **The two splits** are about *where* the track stops (an undershoot, a
  muon/electron swap), not about whether an arm at the upper end counts. So
  v5's rule change is not what the scanners disagree on.
* **`029107_1/36`** was flagged `NO_MEASUREMENT:` by both scanners, so it
  joins the four rule 7 against rule 4 items in §8.1.

All five calls are `medium`, which is the rule 7 cap at an upper end.

**Census, the full smx20 record** (`census_smx20.txt`, production 303):

| | TP | FP | FN | TN | purity | efficiency |
|---|---|---|---|---|---|---|
| smx19 | 60 | 1 | 89 | 111 | 0.984 | 0.403 |
| + the owner's rulings | 61 | 0 | 88 | 112 | 1.000 | 0.409 |
| + the v5 pass (smx20) | 61 | **0** | 90 | 110 | **1.000** | 0.404 |

* **The two adopted stoppers** are both chain rejects: `shape_flat|profile_sparse`
  and `shape_flat|plateau_off_mip`. Both have `michel_found` = 1, so the
  `michel_found` TP goes from 53 to 55.
* **Hand stoppers:** 159 → 161.
* **Upper-end hand stoppers:** 6 of the 161. Four of them (`029107_1/32`,
  `029107_15/26`, `029107_3/23`, `029107_4/56`) stand on topology, the case the
  owner's ruling admits.

Against smx19, production's efficiency is unchanged: 0.403 → 0.404. Its one
false positive is gone.

**Labels.** The two adopted items were applied on the real widgets into the new
tag `smx20`, on top of smx19:
* 315 rows carried over and 2 replaced; sha `7433d5c7…`;
* the smx1, smx18 and smx19 label files keep their shas (`e928a784`,
  `f63a7034`, and `39e51f00` since the owner's pin);
* `verify_scan_record.py --tag smx20`: 317 records over 317 rows, with one
  mismatch, the owner's pin on `029107_15/26` (§8.1). The pin is kept.

### 8.5 What is left

0. **The two v5 splits**, `029107_24/47` and `029107_5/93` (`owner_queue` in
   the record): smx19's `THRU` stands until the owner looks.
1. **The 26 unreviewed `FLAT_STOP` calls.** Rule 3 now has no owner example.
   A v5 pass over them is the cheapest way to settle the record before the
   knob grading uses it.
2. **Rule 7 against rule 4**, for the four items in §8.1 plus `029107_1/36`,
   whenever the owner wants to rule.
3. **The owner's other tier A disagreements** (§5.1). The owner's smx1 label
   stands on each; they are listed only in case the owner wants a look:
   * `028084_23/114`, a kind difference only;
   * `028084_29/38`, `UNCLEAR` against `FRAG_THRU`;
   * `029107_0/56` and `029107_10/50`, `MESSY` against `THRU`.
4. **Next session:** grade PDVD's production knobs on PDHD against `smx20`.

## Files

| what | where |
|---|---|
| this doc | `pdhd/docs/19_stm-michel-display-fix-and-review.md` |
| display fixes | `pdhd/stm_michel_scan/smgeom.py` (envelope), `prep_stm_michel_scan.py` (`--ctx-cells`), `stm_michel_viewer.py` (grey cells, note), `README.md` |
| review tooling | `pdhd/stm_michel_scan/campaign/cmp_review.py`, `mkadj.py`, `mkreview_record.py`, `merge.py --base` |
| rubric v4 | `pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md` (sha `2bf6ff5c…`); v3 kept as `smx18/rubric_v3.md` |
| briefs | `pdhd/docs/scan/smx19/AGENT_TASK.md`, `ADJ_TASK.md` (as frozen for the round) |
| record | `pdhd/docs/scan/pdhd_stm_michel_smx19_verdicts.json` (317 records; the 94 reviewed carry a `review` block) and `smx19/provenance.json` |
| comparison | `pdhd/docs/scan/smx19/cmp_review.txt`, `adjudicate.tsv`, `objections.md` (every reviewer's and adjudicator's rubric point, with item keys, plus the integrity audit) |
| round maps (private during the round) | `pdhd/docs/scan/smx19/tiers.json` (key → queue tier), `adj_ab_map.json` (key → which of A/B was smx18), `supersede.txt`, `nonblind_named.txt`, `rubric_shas.txt` |
| smx18 provenance note | `pdhd/docs/scan/smx18/provenance.json` `display_envelope` |
| prep (gitignored) | `pdhd/stm_michel_scan/prep-pdhd-smx19/` (317 payloads with `proj_ctx`) |
| round scratch | `/home/xqian/tmp/h19/` (shots, `v_parts/rv*`, `v_parts/adj*`, `adj/`, logs) |
| §8: the owner's rulings | `pdhd/docs/scan/smx20/owner_rulings.json` (verbatim words, question, reading per item) |
| §8: rubric v5 | `pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md` (sha `750751ea…`); the v4 text kept as `smx20/rubric_v4.md` |
| §8: record | `pdhd/docs/scan/pdhd_stm_michel_smx20_verdicts.json` (`owner_review` on five items, `review_v5` on five) and `smx20/provenance.json` |
| §8: census | `pdhd/docs/scripts/d18_census.py` (`owner_review` precedence; kind-less owner stoppers out of the Michel tally), `smx20/census_smx19_baseline.txt`, `census_owner_rulings_only.txt`, `census_smx20.txt` |
| §8: tooling | `pdhd/stm_michel_scan/campaign/mkowner_record.py` |
| §8: round scratch | `/home/xqian/tmp/h20/` (`AGENT_TASK.md`, `RUBRIC.md`, `items_a<i>.txt`, `v_parts/rv5_a<i>`, `scratch_a<i>`, links into h19's shots) |

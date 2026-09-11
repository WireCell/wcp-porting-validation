# h21 — the v5 double scan of the 26 unqueued FLAT_STOP items (doc pdhd/19 §9), written BEFORE any result

Owner OK: 2026-09-11 ("Yes, please go ahead").

**Items.** The 26 smx18 records whose notes start `FLAT_STOP:`, which were not in doc 18's
owner queue, so smx19 and smx20 carry them verbatim. All 26 are `medium`; 21 are
`STM_ONLY` and 5 `STM_MICHEL`; none is owner-labelled. Every one is a stopper whose only
support is rule 3, which v5 narrowed. They were picked **because** they rest on that
rule.

**Design.** Two independent verdict-blind scans per item under rubric v5, sha
`750751ea…` (the same text as the h20 pass). The 26 are shuffled with seed 20260912 into
G1 (9), G2 (9) and G3 (8):
* wave 1: a1 = G1, a2 = G2, a3 = G3;
* wave 2 (fresh agents): a4 = G2, a5 = G3, a6 = G1.

The frames are blind, on the fixed display with grey cells, shot for this pass
(`check_shots` clean; canvas2d backend).

**Outcome rule** (`mkowner_record.py --stopper-split`), judged on stopper-or-not only:
* **confirmed:** both scans call it a stopper. smx20 is kept.
* **adopted:** both call it not a stopper. The record is rebuilt from one scan:
  * if the verdicts differ (THRU against UNCLEAR/MESSY), the unscored one is taken,
    because it is the conservative choice: it drops the item from scoring rather than
    asserting THRU;
  * otherwise the lower-numbered scanner.
* **split:** one scan calls it a stopper, the other does not. smx20 is kept, and the
  item joins the owner's `own19` tag on :5017.
* Kind or FRAG differences inside an agreed class never go to the owner.

**Headline, fixed now.** Compare against `smx20/census_smx20.txt` (production 303:
TP 61, FP 0, FN 90, TN 110; purity 1.000; efficiency 0.404; 161 hand stoppers). Report
the result as **"rule 3's narrowing removes N stoppers (M of them production also
calls stoppers)"**, plus the split count. Any efficiency change is reported as a
consequence of the rule on a block selected for resting on it: it measures the rule,
not production.

**Audits.** Each transcript is checked against the forbidden list, and additionally
for any `v_parts/rv5_a<N>` other than the agent's own. Wave-2 agents run after
wave-1 records exist in the same round dir.

**Doc placement.** Decided after the adoption count: doc 19 §9 if few items move,
a new doc pdhd/20 if many do.

**Builder gate (15:07, before any h21 result).** The `--stopper-split` edit leaves the
default mode unchanged. Rebuilding smx20 without the flag gives all 317 records
identical to the committed record once `owner_review.source` is masked. That field is a
path relative to the output file, so it differs only because the gate wrote elsewhere.
The five unmasked differences are the five owner-ruled items, in that field alone.
Provenance is byte-identical.

**The owner's own19 rulings land first (15:21), so the baseline moves.** Their 8 labels
are folded as `owner_review` on top of smx20 (`smx21/owner_rulings_own19.json`). Effect on
production (303), against `census_smx20.txt`:
* is_stm TP 61, FP 0, FN 90 -> 91, TN 110 -> 108; purity stays 1.000, efficiency
  0.404 -> 0.401. 029107_15/40 becomes a stopper the chain misses; 029107_18/59 becomes
  MESSY and leaves the scored set.
* michel_found TP 55 -> 56: 029107_15/26 now carries the owner's kind (attached) and
  enters the tally.
* hand stoppers 161 -> 162; upper-end stoppers 6 -> 7.
The 26-item pass is therefore reported against `ops/census_own19.txt`, and the smx21
build takes base = smx20, rulings = the own19 file, plus the h21 pass with
`--stopper-split`.

**Owner caveat, carried into the write-up:** "All of these cases, the dQ/dx fit are not
good enough. Difficult to judge." The five rule-7-vs-rule-4 items were ruled one by one
(3 THRU, 1 STM_ONLY, 1 MESSY), so no blanket rule follows, and why the profile is
unreadable at APA0/APA2 upper ends is an open reconstruction-side item.

**Audits.** rv5_a1: 88 tool calls, 0 flagged (forbidden list plus any other scanner's
v_parts/scratch dir); scratch dir empty; 9 records.
rv5_a2: 86 tool calls, 0 flagged; scratch dir holds only its own report_lines.txt; 9 records.
rv5_a3: 98 tool calls, 0 flagged; 8 records. Wave 1 complete: all 26 items scanned once.
  rv5_a3 wrote three f_meas crops in its OWN scratch dir for a 7x re-check (the h20 lesson
  holds: private scratch, no collision). That re-check moved one verdict (028084_16/106,
  THRU -> STM_ONLY).
rv5_a6: 86 tool calls, 0 flagged; 9 records (second look at G1).
rv5_a5: 77 tool calls, 0 flagged; 8 records (second look at G3).

**Result of the pass (all 26 items, two scans each, rubric v5 only).**
* **confirmed 19** — both scanners call it a stopper, so smx20 stands.
* **adopted 2** — `028084_20/20` (THRU / FRAG_THRU: the track's own line continues in grey)
  and `029107_8/136` (THRU / THRU, at the APA seam). Both leave the stopper set.
* **split 5** — `028084_16/106`, `029107_20/24`, `029107_21/32`, `029107_21/43`,
  `029107_4/72`. Every split is a stopper against UNCLEAR/THRU on an unreadable end,
  which is the owner's own difficulty, not the rule 7 question.
* Two agreed stoppers differ on michel_kind (`029107_3/99`, `029107_7/34`): resolved by
  rule, base kept, never sent to the owner.

**So the headline, as pre-registered:** rule 3's narrowing removes **2** of the 26
stoppers, not the block. The rule survives two fresh blind looks on 19 of 26, and the
remaining 5 are unsettled rather than overturned.

**Doc placement: doc 19 §9** (2 items move, so it is not its own document).
rv5_a4: 91 tool calls, 0 flagged; 9 records (second look at G2). All six audits clean.

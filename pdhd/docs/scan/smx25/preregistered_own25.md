# own25: the owner's scan of the smx25 items the agents could not settle, pre-registered

Written 2026-09-12 before the viewer is served and before the owner makes any call. Committed later
into `docs/scan/smx25/`, unchanged.

## Queue: a rule, not a pick
Every smx25 tranche item where **either agent scan was `low` confidence, or the two scans split**,
plus one anchor.

| item | group | stratum | APA0 strict | agent scans | why it is in |
|---|---|---|---|---|---|
| `029107_21/65` | decision | M | 1 | STM_ONLY medium / UNCLEAR medium | split; decides lever 1 under reading B |
| `029107_3/99` | decision | M | 1 | STM_MICHEL low / low | low |
| `028084_2/49` | decision | M | 1 | STM_MICHEL low / medium | low |
| `029107_7/85` | decision | M | 1 | STM_ONLY low / medium | low |
| `028084_29/53` | decision | N | 0 (majority only) | STM_MICHEL medium / low | low |
| `028084_18/17` | control_thru | M | 1 | STM_MICHEL medium / low | low; the owner ruled it THRU in own19 (doc 19 §8) |
| `029107_5/28` | control_thru | M | 1 | THRU low / medium | low |
| `028084_20/116` | decision | M | 1 | STM_MICHEL high / high | anchor: the agents' most confident lever-1 stopper |

Order on the sheet is shuffled with seed 2525. Group and agent calls are not on the sheet.

## Blindness: declared, not claimed
- The viewer has been un-blinded since 2026-09-08 (doc pdhd/12 §13): the chain's answer is on
  screen. Frames come from `prep-pdhd-h25base` = **production**, where every decision item is
  `is_stm` 0 and every control is `is_stm` 0. The owner therefore sees "the chain says not a stopper"
  on all 8. That bias runs **toward THRU**, i.e. against lever 1. The agents' lean ran the other way.
- The owner has read the doc 25 report, which names `029107_21/65`, `028084_18/17` and `029107_3/99`.
  So these are not blind re-judges. They are owner rulings, and the owner's call governs by the
  campaign's precedence (`owner_review` > `owner_smx1` > agent).

## How the rulings are used (fixed now)
1. The labels (tag `own25`, empty at start) go to `docs/scan/smx25/owner_rulings_own25.json` →
   `mkowner_record.py` → a new record **`smx26`** (smx25 plus `owner_review` on the ruled items).
   Every other record and label tag must stay byte-unchanged (sha list taken before serving).
2. Fold check: every `owner_review` / `owner_smx1` block in smx25 still exists in smx26, and no
   non-queue item changes.
3. Re-grade h25base / h25k / h25r / h25kr on smx26 with the same graders (`STM_SCAN_RECORD`,
   `D25_GATES`), APA0 strict headline, majority alongside, all-APA and golden fraction reported.
4. **Decision rule, the same as §6:** a lever is free iff 0 new FP in APA0 strict. With the split
   ruled, readings A and B coincide on it. An owner `UNCLEAR`/`MESSY` leaves the population as in
   earlier rounds; it is neither TP nor FP.
5. What each call does:
   - decision item → THRU: an arm TP becomes an **FP** (lever 1, and h25r too if the item is one of
     its 3);
   - `028084_18/17` or `029107_5/28` → stopper: TN becomes FN (efficiency only);
   - anchor `028084_20/116` → THRU: a strong signal that the agents' confident calls do not match the
     owner. Report it; do not re-select anything on it.
6. **Not a flip.** A flip changes production and needs the owner's explicit go (CLAUDE.md §5.1).

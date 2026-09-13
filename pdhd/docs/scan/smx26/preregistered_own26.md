# own26: the owner's scan of lever 1's remaining agent-only recoveries, pre-registered

Written 2026-09-12 before the viewer is served and before the owner makes any call. Committed later into
`docs/scan/smx26/`, unchanged. The owner chose option 3 of doc 25 §7.4: scan these first, then decide.

## Queue: a rule, not a pick
**All of lever 1's APA0-strict recoveries that no owner has ruled** (smx25 decision items with strict 1 and
no `owner_review`/`owner_smx1` in `smx26`), plus **two THRU controls**. The controls are one per stratum,
drawn with `random.Random(2626).choice` from the smx25 THRU controls that have no owner block.

| item | group | stratum | agent scans (smx25) | record truth (smx26) |
|---|---|---|---|---|
| `028084_10/46` | decision | N | STM_ONLY detached dots, medium / medium | STM_ONLY, agent |
| `028084_2/116` | decision | M | STM_MICHEL attached, medium / medium | STM_MICHEL, agent (low) |
| `028084_5/115` | decision (also `h25r`) | N | STM_ONLY detached dots, medium / medium | STM_ONLY, agent |
| `029107_1/85` | decision | M | STM_MICHEL attached, medium / high | STM_MICHEL, agent |
| `029107_12/118` | decision | N | STM_ONLY none, medium / medium | STM_ONLY, agent |
| `029107_12/95` | decision | N | STM_MICHEL attached, medium / medium | STM_MICHEL, agent |
| `029107_24/33` | decision (also `h25r`) | N | STM_ONLY detached dots, medium / medium | STM_ONLY, agent |
| `029107_28/109` | decision | N | STM_MICHEL attached, medium / medium | STM_MICHEL, agent |
| `029107_19/123` | THRU control | M | THRU / FRAG_THRU, medium / medium | THRU, agent |
| `028084_24/114` | THRU control | N | THRU / THRU, medium / medium | THRU, agent |

Control pools: M = {`028084_18/122`, `029107_19/123`}; N = {`028084_0/108`, `028084_1/28`,
`028084_24/114`, `029107_4/118`}. Sheet order is shuffled with seed 2626 (`own26_sheet.tsv`), and the
sheet carries no group, stratum or agent call.

## Blindness: declared, not claimed
As own25: the viewer shows the chain's answer, and frames come from production (`prep-pdhd-h25base`,
`--dead-points`). Production calls all 10 not a stopper, so the bias on screen runs toward THRU, against
the lever. These are owner rulings and govern by precedence.

## How the rulings are used (fixed now)
1. Labels tag `own26`, empty at the start. They go to `docs/scan/smx26/owner_rulings_own26.json` through
   `d25_own25_rulings.py --tag own26`, then `mkowner_record.py --skip-v5` on smx26, giving a new record
   **`smx27`**. Every other label tag must stay byte-unchanged (sha list taken before serving). The fold
   diff must show exactly the 10 changed and no block lost.
2. Re-grade h25base / h25k / h25r / h25kr on smx27 with the same scorer (`--name smx27`) and the committed
   graders on derived gates. APA0 strict is the headline; majority, all APAs and golden are also reported.
3. **The rule is unchanged, and lever 1 is already not free** (`029107_21/65`). This scan measures the size
   of the trade. On smx27, lever 1's strict false positives = 1 + the number of the 8 decision items the
   owner calls THRU. An owner `MESSY`/`UNCLEAR` removes the item from the population: it is neither TP nor FP.
4. **`compare_range_cm` 45 can lose its "free" verdict here.** Two of its three recoveries (`028084_5/115`,
   `029107_24/33`) are in this queue. An owner THRU on either one is a new FP for `h25r`, which the rule
   makes not free.
5. A THRU control called a stopper turns TN into FN on every arm (efficiency only). It is reported as a
   check on the owner-vs-agent agreement, and no threshold is re-selected on it.
6. **Not a flip.** After this scan the owner decides, with the trade stated from smx27. A flip needs the
   owner's explicit go. It is a production jsonnet edit plus a confirmation arm that must be bit-identical
   to the measured arm.

# doc pdvd/99 sec 6.4 -- pre-registered readout of the blind swap scan (written before any verdict)

Written 2026-09-13, before the frames were finished and before any scanner was assigned. Not changed after.

## Items (`d99_swap_scan_set.py --n-controls 45 --seed 20260913`)

- `on_only`: all 91 clusters `p98vonq` tags (is_stm 1) that `p96vprod` does not tag (not a candidate 62, candidate not
  is_stm 19, object not matched 10). Displayed on `p98vonq`.
- `prod_only` control: 45 of the 80 clusters `p96vprod` tags that `p98vonq` does not (82, minus the 2 event/cluster keys
  that also occur in `on_only`), seeded. Displayed on `p96vprod`.
- One shuffled list (seed 20260914), verdict-blind (`--blind --hide-selection`), 3 scanners, the same rubric
  (`RUBRIC.md`, sha stamped on every record). The scanner is not told that two groups or two arms exist.

## Readout

1. **Classes.** stopper = `STM_MICHEL`, `STM_ONLY`, `FRAG_STM_MICHEL`, `FRAG_STM_ONLY`; non-stopper = `THRU`, `FRAG_THRU`;
   excluded and counted = `MESSY`, `UNCLEAR`.
2. **Purity of each side** = stoppers / (stoppers + non-stoppers), with a binomial (Wilson) 68 % interval.
3. **The statistic:** purity(`on_only`) - purity(`prod_only`), bootstrap over items (10 000 resamples, seed 20260915),
   68 % and 95 % intervals. Reading: a purity loss of the swap if the 95 % interval lies entirely below 0; a gain if
   entirely above; otherwise purity-neutral within this sample, with the interval quoted.
4. **Secondary, same classes:** by volume (top / bottom); `on_only` by production-side status (not a candidate / candidate
   not is_stm / object not matched, the last as its own row); `high`-confidence calls only; MESSY+UNCLEAR rate per side.
5. **Michel:** per side, the fraction of hand stoppers that are `STM_MICHEL`; and the chain's `michel_found` on the display
   arm against the hand Michel (purity, efficiency) per side.
6. **Calibration against the existing record** (the scanners are not shown it): stopper-or-not agreement with the carried
   record (`on_only` items that have one) and with the smx record (controls), by scanner confidence. Expected to be
   imperfect for known reasons: the record was partly taken with the chain's answer on screen and under the frozen
   PDVD rubric, this round is verdict-blind under the ported v5 rules.
7. **Derived, labelled as mixing instruments:** each arm's whole-population is_stm purity with the shared (tagged on both)
   clusters taken from the record: (n_shared x p_shared + n_side x p_side) / n_arm.

No threshold, knob or rubric is changed on these numbers in this round.

# doc qlmatch/35 -- pre-registration amendment 1: the blind scan of the C4 unlabelled candidates (written 2026-09-23,
# after the C4 STOP and before any scan item was drawn, rendered or shown)

Owner, 2026-09-23, choosing route (a) of doc 35 sec 6: "Please do a blind scan by the agent, and then update md file,
aim to flip the production settings, commit and push."  `prereg.md` stays as written (sha in `prereg_sha.txt`); this
file adds to it.  Its sha is appended to `prereg_sha.txt` before the item list is drawn.

## 1. Items
- **New items:** every row of `c4_unlabelled.tsv` (43: 33 from `q35tk`, 10 from `q35ctl`) -- the candidates with no
  truth in either cell after the doc-35 carry -- each displayed on its own arm under its native key.
- **Calibration:** 10 % of the new items (at least 2), drawn with seed 35 from the `q35ctl` candidates that the record
  judges (not MESSY / UNCLEAR), displayed on `q35ctl`.  The doc 116 V2 check.
- The item list is private (key, display arm, role) and never shown to a scanner (`d35_scan_items.py`).

## 2. Procedure (doc 116 sec 5, unchanged in substance)
- Prep: `prep_stm_michel_scan.py --det pdvd --arm <arm> --ctx-cells --redraw` for `q35ctl` and `q35tk`; set:
  `d103_scan_set.py`; blind headless shots (`shoot.sh`, `--blind --hide-selection`), `mkzoom.py`, `check_shots.py`.
- Items the prep does not render (its own filters: `has_pass`, >= 20 profile points, muon >= 10 cm) are reported and
  stay unlabelled; they are NOT given a verdict any other way.
- Rubric: `pdvd/docs/nf_sp_img_clus/d99/swap_scan_rubric.md` (sha256 prefix d760e223bfa1770a), copied unchanged as the
  round's RUBRIC.md.  Agent task: `116_agent_task_pdvd.md` with only the round paths changed.
- Scanners: Opus agents, one per wave file (`nextwave.py`, seed 35, <= 9 items each), blind, writing only through
  `mkv.py` into their own out dir.  Every transcript is audited (`d35_audit.py`, a fork of `d116_audit.py` with the
  round root changed); a flagged scanner's records are discarded and its items re-waved to a new scanner.
- Record: `d103_scan_record.py --tag smx35` -> `pdvd/docs/scan/pdvd_stm_michel_smx35_verdicts.json` (new file).

## 3. Bars
- V2 (d103_scan_record.py's rule): stopper-or-not disagreement of the calibration items with the existing record
  > 25 % -> STOP (the scan is not used).
- Audit: any unresolved flag -> STOP.

## 4. The fold and the re-grade
- smx35 is folded at the LOWEST precedence (below smx116: an item already labelled anywhere keeps its label).  A
  `q35tk` item's native key is mapped into the A0 key space through the same carry map as C4 (the grade's own `T`
  re-keying); a `q35ctl` item is keyed as is.
- Re-grade: `d35_stm_grade.py` reading 1 with the unchanged C4 rule -- each of `is_stm` purity / efficiency and Michel
  purity / efficiency must be >= -0.020 vs `q35ctl` in both the NEG and POS scenarios for whatever stays unlabelled.
  PASS -> C4 PASS.  Anything else -> no flip, report.
- **Post-hoc clause fix, stated as such:** C3's PR sub-clause counts CheckSTM_Michel's zero-candidate end line ("no
  STM-tagged main cluster") as a completed PR (the clause was written for crashes; doc 35 sec 4.2).  With it C3 reads
  PASS (`c3_tails_prod_zero_ok.txt`); the literal reading stays on record.

## 5. Then the flip (prereg.md sec 0 and sec 2 F1, unchanged)
If C4 passes: re-take the F1 pre-half (`d35_flip_compiled.sh pre`) immediately before the edit; make the edit of
prereg.md sec 0; run F1 (compiled config, `_tot` light == `_q32ti`, light escape == `_g31off`, `q35flip` through
`stm/run_campaign.sh` == `q35tk` + control differs + the `ks_sat_tol` deploy tell).  Any F1 failure -> the edit is
reverted, nothing is committed as a flip, and it is reported.

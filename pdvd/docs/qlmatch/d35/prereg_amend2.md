# doc qlmatch/35 -- pre-registration amendment 2: a larger calibration of the blind scanners (written 2026-09-24,
# after the amendment-1 V2 FAIL, before any new item was drawn)

Owner, 2026-09-24, choosing option (ii) of doc 35 sec 6: "Approve a larger calibration set to be scanned."
`prereg.md` and `prereg_amend1.md` stay as written; this file adds to them. Its sha is appended to `prereg_sha.txt`
before the calibration items are drawn.

## 1. What is added
- **26 new calibration items**, so the combined calibration is 4 + 26 = 30.
- Drawn with seed 36 from the `q35ctl` STM candidates judged (not MESSY / UNCLEAR) by an **owner-derived source only**:
  grade-truth source `record` (the owner-corrected p99rwon record and own116v) or `owner_review` (own103v / own103v2).
  The AI scans smx11 and smx116 are excluded from the new draw, since agreeing with another AI scan says little.
  Keyed on the record lineage `d116vflip`.
- Round 1's 4 calibration items stay exactly as drawn, including 039253_6/124 (source smx11). None of the 4 is among
  the keys the doc 35 sec 8 defect shifted or failed to carry, so the 2/4 is not a keying artefact.
- Each item is shown on `q35ctl` under its native key, through the corrected re-keying of doc 35 sec 8, carried with
  status ok only.
- Excluded: every key of the amendment-1 item list, and the 3 record-labelled objects of `scan_posthoc_agreement.txt`.
  Those stay non-governing and are not counted.
- **One new item, blind-mixed with the calibration items: `q35ctl` 039252_8/102.** A coverage check of the corrected
  `c4k_unlabelled.tsv` (38 rows) against the smx35 record's keys and `no_payload.tsv`, by key membership only with no
  verdict read, found 36 rows scanned or unscannable. 039349_82/19 counts as scanned: both arms' keys are one object.
  039252_8/102 is unlabelled only because its record label (THRU) fails to carry (the cluster split, sec 8), so it was
  never drawn.
- What stays unlabelled after a pass: the 6 unscannable objects. Of these only 039349_77/40 is tagged (`is_stm` 1,
  Michel 1), and it is the same object in both arms, so it enters NEG symmetrically.

## 2. Procedure (amendment 1 sec 2, unchanged)
- A new round dir `/home/xqian/tmp/p35scan/round_pdvd2`, with the same prep (`q35ctl`, already made), `d103_scan_set.py`,
  blind shots, the same RUBRIC.md (sha d760e223) and the same agent task with the round path changed.
- Fresh Opus scanners (none of amendment 1's), at most 7 items each, `nextwave.py` seed 36.
- Every transcript is audited (`d35_audit.py` with round name `round_pdvd2`). A flagged scanner's records are discarded
  and its items re-waved.
- Record: `d35_scan_record.py --tag smx35c` -> `pdvd/docs/scan/pdvd_stm_michel_smx35c_verdicts.json` (calibration only).

## 3. The bar (combined)
- V2 over the combined calibration: stopper-or-not disagreements / items judged on both sides, amendment-1's 4 (2
  disagreements) plus the 26 new.
- **PASS if <= 25 %.** The first 4 cannot be dropped. With all 30 judged on both sides that allows at most 7
  disagreements in total: 2 are spent, so at most 5 may come from the new 26. If exclusions shrink the denominator,
  the ratio rule governs.
- Items the prep cannot render, or that either side calls MESSY / UNCLEAR, are outside the ratio (d103_scan_record.py's
  rule) and are reported.
- Computed by `d35_v2_combined.py` from the two record files.

## 4. If it passes
- Amendment 1 sec 4 applies unchanged: smx35 (its non-calibration rows) and the new item's smx35c row are folded at the lowest precedence into the
  corrected grade, now `d35_stm_grade.py --key-arm d116vflip --scan-record ...`, and C4 is re-read by the unchanged
  rule.
- **C4 PASS** -> F1 and the flip (amendment 1 sec 5), then commit and push.
- **C4 not PASS** -> no flip, report.

## 5. If it fails
No flip, and the smx35 labels stay unused. Report.

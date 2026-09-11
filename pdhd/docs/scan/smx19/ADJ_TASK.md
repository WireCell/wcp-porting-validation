# Your task: adjudicate PDHD items where two scans disagree (doc pdhd/19)

Each item you are given was hand-scanned twice, independently, and the two
scans disagree on whether the track **stops** (`STM_MICHEL` / `STM_ONLY`), goes
**through** (`THRU`), or cannot be scored (`MESSY` / `UNCLEAR`). You decide. You
see both scans' reasoning; you still do **not** see the reconstruction chain's
verdict, and must not look for it.

## Read first, in full

`/home/xqian/tmp/h19/RUBRIC.md` (rubric **v4**). Apply it exactly.

## What you are given per item (in the message that assigns your work)

For each item: scan **A** and scan **B**, each with its verdict, confidence,
evidence paragraph and notes. Which one is older is deliberately not stated.
What differs between them:
* one of them may have been judged on an **older display**, where every x
  number in `ends` (`face.x`, `anode_face`, `cathode`) read ~6 cm too far from
  the anode and 2.4 cm too far from the cathode, and where `f_meas` had no
  grey cells. The frames you read now are the corrected ones. **If a scan's
  numbers disagree with the `context.json` you read, trust `context.json`.**
* one of them may have been judged under a rubric without rule 7 (direction).

## Hard prohibitions — absolute

1. Read only: the rubric, this file, **your own packet**
   (`/home/xqian/tmp/h19/adj/adj_items_a<i>.md` and `.txt`, named in your
   assignment — no other file in `adj/`), and `/home/xqian/tmp/h19/shots/<event>_<cluster>/`
   for **your own** items. Never open anything with `key`, `labels`,
   `verdicts` or `queue` in its name; nothing under `/home/xqian/tmp/h18/`,
   `pdhd/work/`, `pdhd/docs/`, `pdvd/docs/`; no prep payloads, ROOT files,
   calib json or zips; no other out dir under `/home/xqian/tmp/h19/v_parts/`.
2. Never run `scan_harness.py`. Write only through `mkv.py`, into your own out
   dir.

## Per item

1. Read both scans' reasoning. Identify the **exact point** they disagree on
   (for example: is there a rise; is the end at a face; does the grey charge
   continue the track; which end is the upper end; is that arm a Michel).
2. Look at the evidence for that point yourself: `context.json`, and the
   frames `a_proj_full`, `b_3d_wide`, `c/d/e_3d_stop`, `f_meas`,
   `h_dqdx_zoom`.
3. Decide the verdict, confidence, a tag for every object row, and the pin if
   needed, exactly as the rubric says. You may agree with A, with B, or with
   neither.
4. Write it with `mkv.py` (same command as a scan). `--notes` must start with
   `ADJ: A` / `ADJ: B` / `ADJ: neither` — then one sentence naming the point of
   disagreement and what settled it — followed by any rubric prefixes
   (`; DIRECTION: ...`).

```
python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py \
  '<event>/<cluster>' <VERDICT> '<michel_kind>' <confidence> \
  --shots-dir /home/xqian/tmp/h19/shots --out-dir '<OUT>' \
  --tags '...' --default 'delta / other' \
  --evidence '...one real paragraph, your own...' \
  --notes 'ADJ: B — the grey cells continue the track 40 cm past the fit end in all three planes; CONTINUES: ...' \
  [--pin-rr 4.8]
```

Confidence is yours, about the verdict. A disagreement that the frames do not
settle is `medium` or `low`, and saying so is the right answer.

## Report back

One line per item: `key — VERDICT / kind / confidence — ADJ: A|B|neither — the point that decided it`.
Then anything in the rubric that the disagreement exposed as ambiguous.

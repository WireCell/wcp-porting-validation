# pr/130 item 4 — owner rulings on the q_miss whole-object scan

Bee set `8daa1825-f386-4ba2-9094-61577e075f9d`
(`bee/pr130r4/pr130r4-qmiss.index.txt`). One row per adjudicated object.
Mechanism, ranking and repro: `pr130-qmiss-mechanism.md`.

Fresh file for this round's decisions — no earlier decisions record is written
into (M13).

## Repro for the geometry quoted below

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 - <<'PY'
import json, math, itertools
d = json.load(open('em_display/emprep-pr130q98/emprep-evt463565.json'))
sh = d['showers']
for a, b in itertools.combinations(['13001','115088','114086','109073','72064'], 2):
    ea, eb = sh[a], sh[b]
    dot = max(-1, min(1, sum(x*y for x, y in zip(ea['dir15'], eb['dir15']))))
    print(a, b, round(math.dist(ea['start'], eb['start']), 1),
          round(math.degrees(math.acos(dot)), 1))
PY
```

## idx 0 — 463565 (ncpi0), shower 13001 — **MERGE, all four**

**Owner, 2026-08-29:** *"There should be two EM shower, but the beginning part
of the two showers are connected, so I assume it is difficult to separate them.
In this case, the later shower should be merged together with the two showers."*

**Verdict: YES on all four objects.** 115088, 114086, 109073, 72064 → 13001.
Recovers **336.4 MeV** (719.9 → 1056.2), the round's largest single item and
9.077e6 of q_miss.

The geometry matches the ruling term for term:

| pair | start separation | dir15 opening angle |
|---|---|---|
| 13001 – 115088 | **11.8 cm** | 104.0° |
| 13001 – 114086 | **9.4 cm** | 144.4° |
| 13001 – 109073 | **9.4 cm** | 156.6° |
| 115088 – 114086 | **9.5 cm** | 83.4° |
| 114086 – 109073 | 0.0 cm | 33.1° |
| 13001 – **72064** | **46.1 cm** | 140.6° |
| 114086 – 72064 | 38.2 cm | **12.4°** |

So the four starts sit inside a **12 cm ball** ~63–72 cm from the ν vertex
while their directions differ by **83–157°** — "the beginning part of the two
showers are connected" is literally true, and it is why no angular test can
separate them. 114086 and 109073 share a start *and* an end: they are two
decompositions of the same material. 72064 is the **"later shower"** — 38–46 cm
downstream and only 12.4° off 114086, i.e. a continuation of that leg, not a
third origin.

### Two consequences, and the second one is a warning

1. **The SPLIT/WHOLE class is a real defect, confirmed on the largest item.**
   This was the round's open question (`pr130-qmiss-mechanism.md` Part 6) and
   the answer on idx 0 is yes. It does not yet generalise — see the open rows
   below.

2. **The merge is licensed by inseparability, not by identity — and that puts
   it in direct tension with the pi0 work.** The owner's own words are *"there
   SHOULD be two EM shower"*. Merging gets the **energy** right (1056.2 MeV,
   nothing lost, nothing double-counted) and simultaneously makes the **two-
   gamma separation a pi0 mass needs** impossible on this event. That is
   exactly the failure `pr130-qextra-98set.md` Part 6 named — *"314838's 130.6
   MeV is the right number from the wrong pairing"* — and the PID blocker from
   `pr126-pi0-audit`. **A merge-side knob tuned on q_miss would show a clean
   win here while quietly making pi0 reconstruction worse**, and no metric in
   this round would report it. Any such knob must be gated on a pi0 metric as
   well as on q_extra.

3. **The admission feature, if one is ever built, is start proximity — not
   angle.** Every guard the pr/123→pr/130 campaign shipped is a length/PID or
   angle test. Here the angles are 83–157° apart and would *reject* the merge;
   the only feature that separates is the 9–12 cm start separation. A merge
   predicate would therefore be a genuinely new family, not a loosened guard.

## Open — awaiting owner

| idx | event | objects | q_out | status |
|---|---|---|---|---|
| 1 | 122660 | 47050, 54071, 53070 | 6.868e6 | **next up** — the other decisive one |
| 2 | 54332 | 7 objects → 16014, 122091 | 3.648e6 | open |
| 3 | 181050 | 68031, 61021 | 1.768e6 | open |
| 4 | 105946 | 53030, 54032, 39013 | 1.530e6 | open |
| 5 | 21073 | 31023 (PARTIAL 0.64) | 1.423e6 | open |
| 6 | 342199 | 25109 (PARTIAL 0.88) | 1.318e6 | open |
| 7 | 76346 | 6 objects, all WHOLE | 1.234e6 | open |
| 8 | 469665 | 66042, 62073, 68052 | 1.178e6 | open |
| 9 | 54453 | 58051 (PARTIAL 0.17), 1 NOHOLDER | 1.087e6 | open |

idx 1 matters most next: if 122660 also merges, two of two decisive events say
the class is real and a merge-side round is justified. If it does not, the
separator between idx 0 and idx 1 is the predicate, and finding it is the
deliverable rather than a knob.

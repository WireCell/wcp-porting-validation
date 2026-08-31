# doc pr/139 §4 — the scan brief  (tag `splitscan-0902-pi0`, port 5022)

```
./split_display/serve_split_display.sh 5022 --scan-tag splitscan-0902-pi0
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   http://localhost:5022/split_viewer
```

The keepalive flags are not decoration: a bare `ssh -L` is reaped by an idle
timeout during exactly the long pauses a hand scan is made of, and bokeh's JS
does not reconnect — the symptom is a silent hang, which reads as "the tool is
broken" rather than "reload me".

## A — the four π⁰s the splitter broke  *(4 objects, the ones that decide `shower_split_max_impact`)*

**Two questions each, and they are different questions.** (1) *Is the cut right?*
(2) *Should the two parts still be treated as ONE γ for the π⁰?* A correct cut
whose parts are one γ is a real case, and only answer (2) can tell us.

| event | object | what the splitter did | what it cost |
|---|---|---|---|
| **54332** | node **122091** | cut 13 seg → 5 + an 8-seg daughter (57.2 MeV) | the π⁰ **changed partner**: 122091+27025 @ 117.5 MeV → 122091+128111 @ 110.6. **The only one of the four that cost an *exact*.** |
| **165157** | node **9000** | cut 8 seg → 1 (187.9 → 94.4 MeV) | pair lost. Also a shared-membership peel: the daughter was seeded on segment 58027, which already **roots** another shower |
| **281485** | node **89095** | cut 8 seg → 4 | the daughter came out at **0.00 MeV** — its 4 segments were **also members of shower 91112**, which gained exactly what the parent lost |
| **396222** | node **9059** | cut 123 seg → 73 | the OFF "exact" was a **2879 MeV** blob landing at 135.7 MeV. Busy event you already discounted — **judge only if you want to; I have written it off** |

`b` (how far the object's own axis misses the ν vertex) is 39.7 / 13.1 / 23.7 /
29.2 cm here, against 1.7–10.9 cm for the four splits that *gained* a π⁰. A bound
at 12 cm separates all eight — but it was chosen after seeing them, and it
silences **29 of 51 fires**. That trade is the call this scan is for.

## B — the ~15 hand-marked showers the splitter touches, re-marked PER PART  *(the enabler)*

This is the one that unblocks measurement. `pr136_completeness.py`'s target comes
from the 2026-08-27/28 attribution scan, which called several of these objects
**one** shower; the split scan says three to five. **93–94 % of the measured
`q_miss` rise is that conflict**, and it runs the other way too — **28 % of the
`q_extra` gain sits on objects the split scan calls KEEP**. Until each part has
its own marks, `q_miss`/`q_extra` cannot *grade* a splitter or a re-home; they
can only be quoted.

## C — evt314838  *(1 object, a standing adjudication)*

Three of your own instruments disagree: split scan says **SPLIT2**, high
confidence; the attribution scan says the split takes purity 0.715 → **1.000**;
the hand π⁰ needs the charge that the split removes. One call settles **which
instrument leads when they conflict** — that ruling outlives this round.

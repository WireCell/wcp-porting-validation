# pr/130 item 4 — owner rulings on the q_miss whole-object scan

Bee set `8daa1825-f386-4ba2-9094-61577e075f9d`
(`bee/pr130r4/pr130r4-qmiss.index.txt`). One row per adjudicated object.
Mechanism, ranking and repro: `pr130-qmiss-mechanism.md`.

Fresh file for this round's decisions — no earlier decisions record is written
into (M13).

## Scanning it in em_display (preferred over Bee for this question)

Bee shows the objects but not *why* each segment is out of the shower;
em_display carries the probe sidecar, so it shows the absorb-tape reason per
segment and lets marks be saved. Served on **5017** with a fresh scan tag:

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# one-off: 10-event manifest + a prepdir of symlinks to the two record dirs
python3 - <<'EOF'
import csv, os
ORDER = [463565,122660,54332,181050,105946,21073,342199,76346,469665,54453]
rows = {}
for m in ('em117-pr130q98-manifest.tsv', 'em114c-pr130q141-manifest.tsv'):
    for r in csv.DictReader(open('em_display/'+m), delimiter='\t'):
        rows.setdefault(int(r['event']), r)
cols = ['sample','run','subrun','event','dump']
with open('em_display/em130q-scan10-manifest.tsv', 'w') as fh:   # plain write => LF
    fh.write("\t".join(cols)+"\n")
    for e in ORDER:
        fh.write("\t".join(str(rows[e][c]) for c in cols)+"\n")
os.makedirs('em_display/emprep-pr130scan10', exist_ok=True)
for e in ORDER:
    for src in ('emprep-pr130q98', 'emprep-pr130q141'):
        p = 'em_display/%s/emprep-evt%d.json' % (src, e)
        if os.path.exists(p):
            l = 'em_display/emprep-pr130scan10/emprep-evt%d.json' % e
            if not os.path.lexists(l): os.symlink(os.path.abspath(p), l)
            break
EOF
./em_display/serve_em_display.sh 5017 --scan-tag emscan-0829-pr130qmiss \
    --manifest $PWD/em_display/em130q-scan10-manifest.tsv \
    --prepdir  $PWD/em_display/emprep-pr130scan10
```

From a laptop (the keepalives are not optional — doc pr/88):

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    -L 5017:localhost:5017 <user>@wcgpu1.phy.bnl.gov
# http://localhost:5017/em_display_viewer
```

**Write the manifest with a plain `write`, not `csv.DictWriter`.** DictWriter
defaults to CRLF, every `dump` path then ends in `\r`, and the viewer degrades
each row to "no probe" *silently* rather than failing — the exact trap the
em_display README warns about. Caught here by `test -f` on all ten paths before
handing the port over; the check is worth repeating on any new manifest.

Scan tag `emscan-0829-pr130qmiss` is **new** — no existing label dir is written
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

## The scan is in — 2026-08-29, tag `emscan-0829-pr130qmiss`

Nine events saved plus the verbal 463565 ruling. Scored with the scan's own
labels against the same dumps and sidecars the census used:

```bash
cd em_display
./em117_score.py --tag emscan-0829-pr130qmiss \
    --manifest $PWD/em130q-scan10-manifest.tsv --prepdir $PWD/emprep-pr130scan10 \
    --cross-run --tsv ../docs/pr/pr130-qmiss-scanned-score.tsv
```

### Result 1 — every answerable question is YES

**17 of 17.** Not one candidate object was ruled correctly separate. The
SPLIT/WHOLE class is a **real defect**, and the alternative Part 6 posed —
"the label store is over-marking neighbours a physicist would leave separate" —
is **dead**.

### Result 2 — three of ten events have a WRONG ν VERTEX

| event | note | q_out at stake |
|---|---|---|
| 54332 | *"vertex wrong"* | 3.648e6 |
| 54453 | *"wrong vertex"* | 1.087e6 |
| 76346 | *"wront vertex"* | 1.234e6 |

**5.969e6 — 20.5% of the scan pool — is not a shower-clustering question at
all.** With the vertex wrong the shower assembly cannot be judged, so these 15
questions (Q8–Q14, Q22–Q27, Q31–Q32) have no answer and never will until the
vertex is fixed. Read with Part 1's 17% REROOT and Part 5c's mis-seeding, this
is the third instrument pointing at the same layer: **the binding constraint on
these events is vertex and seeding, not shower admission.**

### Result 3 — no geometric bound exists, again

Every approved merge whose candidate roots its own reconstructed object:

| span | approved range |
|---|---|
| start separation | **9.4 – 182.8 cm** |
| dir15 opening angle | **9.4 – 156.6°** |

All 17 approved across that entire range. The start-proximity hypothesis this
doc raised from 463565 alone (9–12 cm) is **falsified**: 122660 merges at
88–90 cm, 469665 at 60–183 cm. This is the same negative shape as pr/128's
proximity finding and pr/129's extent/distance finding — *"over-clustering is
not a distance"*, and now neither is under-clustering. **Do not propose a
geometric merge bound.**

### Result 4 — the census under-counted, and q_extra needs a caveat

| | old labels | this scan | delta |
|---|---|---|---|
| q_miss over the 6 scanned | 1.335e7 | **1.522e7** | **+1.871e6 (+14%)** |
| q_extra over the same 6 | 4.661e6 | **0** | −4.661e6 |

Same dumps, same sidecars — **the entire difference is labelling.** The owner
marked segments the census never flagged (53068 on 122660; 32006/38012 on
105946; 57017 on 181050; 28020/29021/47040/52045 on 21073; 26040 on 342199;
66047/68053/68059 on 469665), so the true under-clustering is *larger* than
this round measured.

**The q_extra collapse must NOT be read as "the guards removed it".** The task
given was *"I marked in the clusters that should be merged"* — marking segments
**out** was not asked for, so zero `out` marks is consistent with the framing
and is **not** evidence that nothing on these events is over-clustered. The
earlier 4.661e6 came from agent-authored labels (`emscan-0827`,
`emscan-0828-agent5`); whether it was real over-clustering or agent
over-marking is **not decided by this scan** and would need a scan that asks
the out question explicitly.

### What this leaves

- A merge-side knob is **justified** (17/17) but has **no geometric predicate**
  (Result 3), so it cannot be built the way every guard in this campaign was.
- Any such knob still carries the 463565 pi0 warning below: it fixes energy and
  can cost the two-gamma separation.
- **The larger prize is the vertex** (Result 2): 20.5% of this pool is
  unscannable, and no shower work can recover it.

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

## Part 8 — what is left to try, measured

The scan says the merges are right (17/17) and Result 3 says no geometry
selects them. So the next question is *mechanism*, not threshold. Two
measurements, both off the existing sidecars:

### The right shower is nearly always the bigger one

For every missing segment held by some other reconstructed object, comparing
the target shower with the holder (distinct (event, target, segment) triples;
main-cluster charge excluded, as everywhere else in this round):

| | segments | charge | share |
|---|---|---|---|
| target **more** energetic than the holder | **148** | 3.197e7 | **93.6%** |
| target less energetic | 10 | 2.189e6 | 6.4% |

Median energy ratio **7.8×**; 127 of 158 exceed 3×. Over 61 distinct
(event, target, holder) triples. **Small showers are holding material that
belongs to much larger ones** — e.g. 122660: a 73 MeV / 7-segment object holds
charge belonging to a 960 MeV / 39-segment shower; 84229: 31.5 MeV holding from
960.5 MeV.

### The candidate mechanism is first-come-first-served

`PRShower.cxx`'s flood-fill skips any segment already in `used_segments`
(`used_segments.find(seg) == used_segments.end()`), and that skip is **silent** —
no tape line. Whichever shower's walk arrives first claims the segment
permanently; there is no comparison and no revisit.

### …but the tape cannot yet distinguish that from unreachability

Of 108 STOLEN segments, the target shower appears in the segment's own tape
**once**. 91 segments name exactly one shower — the holder. That is consistent
with "the holder got there first", **and equally consistent with "the target's
walk never reaches that segment at all"**, because the `used_segments` skip
emits nothing. The two hypotheses are **observationally identical in the current
tape**, so the 93.6% is *not* evidence that reordering would fix anything. It is
evidence that one measurement is worth making.

### The measurement — a contention probe

Tape, from inside the flood-fill, every segment a walk **reached and skipped
because `used_segments` already held it**, with the holder's id. Env-gated,
byte-neutral with the env unset, emitted from the code path it measures so
census and mechanism cannot drift — the shape that worked for
`WCT_PFNEAR_DEBUG` (pr/128) and the guard-freed tape (pr/130). It splits the
pool cleanly:

- the target's walk **did** reach the segment → genuine contention, the outcome
  is order-dependent, and an ordering change is on the table;
- it never reaches it even when free → a **reach** failure, and ordering is
  irrelevant.

**If it comes back contention, it also supplies the predicate Result 3 says we
do not have** — *"merge when the dominant shower's walk was blocked by a
smaller holder"* is a mechanism rule, not a distance, and it is the only
candidate this round has produced that is not falsified.

### Two constraints the owner should weigh before choosing this

1. **An ordering change is not knobbable the way this campaign's knobs are.**
   Every guard shipped in pr/123→pr/130 is a local admit/decline with an
   absent-key default and a blast radius of a handful of events. Processing
   order touches **every event with a shower**, so the whole reorder sits behind
   one flag and the gate is the full 239-event manifest. This is a bigger piece
   of work than anything in the campaign so far.
2. **Determinism bites here specifically.** An energy-ordered walk needs a total
   order with a stable tiebreak on cluster/segment ids — `kine_best` ties, and
   float ordering over pointer-keyed containers is exactly M4 / §2-Determinism
   territory. Get the tiebreak in before the first gate, not after.

The 463565 π⁰ warning above is **not** retired by the scan and stays attached to
any merge-side proposal: fixing the energy can cost the two-gamma separation.

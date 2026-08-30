# pr/130 item 4 — owner rulings on the q_miss whole-object scan

**STATUS: PARKED 2026-08-29 by the owner ("we will move on for now, may come
back to this later"). Nothing is in flight; nothing is half-shipped.**

One-paragraph state of play: the owner's hand scan approved **17 of 17** merges,
so the under-clustering is a **real defect** and not a labelling convention.
Three candidate fixes were then built and measured, and **all three are dead** —
loosening the F12 walk guard (Part 3), walk ordering / `used_segments`
contention (Part 9, 0 of 78), and cross-cluster threshold / arbitration /
owned-skip (Part 10). What survives is one **structural** lead, stated in
Part 11 with the exact next measurement. Two default-OFF probes are **shipped in
the tree**, so resuming costs no rebuild of instruments.

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

## Question sheet — final state

All 32 questions in `pr130-qmiss-questions.txt` are resolved one way or the
other; none is still waiting on the owner.

| idx | event | outcome |
|---|---|---|
| 0 | 463565 | **MERGE all four** (verbal ruling, Q1–Q4) |
| 1 | 122660 | **MERGE**, Q5–Q7 |
| 3 | 181050 | **MERGE**, Q15–Q16 |
| 4 | 105946 | **MERGE**, Q17–Q19 |
| 5 | 21073 | **MERGE**, Q20 |
| 6 | 342199 | **MERGE**, Q21 |
| 8 | 469665 | **MERGE**, Q28–Q30 |
| 2 | 54332 | **not scannable** — ν vertex wrong (Q8–Q14) |
| 7 | 76346 | **not scannable** — ν vertex wrong (Q22–Q27) |
| 9 | 54453 | **not scannable** — ν vertex wrong (Q31–Q32) |

17 of 17 answerable = merge. The 15 unanswerable questions are the three
wrong-vertex events; the owner has said explicitly **not** to chase the vertex
for now.

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

## Part 9 — the contention probe answered: ordering is DEAD

Probe shipped (`WCT_SHOWER_BLOCKED_DEBUG`, `PRShower.cxx`), gate PASS, measured.

### Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_blocked_probe.sh gate    # env UNSET -> byte-identity gate
./scripts/pr130_blocked_probe.sh probe   # env SET   -> the tape
scripts/pr130_blocked_census.py > docs/pr/pr130-blocked-census.txt
```

### Validation before the result

`wcbuild` rc=0; freshness proof lib **19:40:15** > src **19:39:53** (M1);
`wcdoctest-clus` **235/235**. Byte-identity gate with the env unset, new binary
vs production `work-pr130r1-bon{,141}-*`: **PASS, all archives byte-identical**
on both arms, and the calib dumps compare SAME.

And the probe is not silently mis-matching ids — the check that had to pass
before the null result could be believed: **every target shower emits BLOCKED
lines** (4–21 each) and ADD lines in its own event. The walks ran, the join
works; they are simply never blocked on the segments they are missing.

### The result

| | segments | charge | share |
|---|---|---|---|
| **CONTENTION** — the target's walk reached it and was refused | **0** | 0 | **0.0%** |
| **UNREACHED** — the target's walk never gets there at all | **78** | 2.196e7 | **100.0%** |

**0 of 78, over 7 events and 2.196e7 of charge.** First-come-first-served on
`used_segments` is **not** the mechanism, and a processing-order or revisit
change would recover **nothing**. That is the whole value of the probe: it
retired, for the cost of one gated build, the reorder that Part 8 flagged as
needing a full 239-event gate.

### What it is instead

The walk is a flood-fill over graph connectivity, so it cannot acquire a
cluster it is not connected to — and that is exactly the situation:

| | segments | charge | share |
|---|---|---|---|
| missing segment's cluster **already held** by the shower | 1 | 8.272e5 | 3.8% |
| **cluster entirely absent** from the shower | **77** | 2.113e7 | **96.2%** |

So the whole of the owner-approved under-clustering is **cross-cluster
acquisition**: the shower has to take on whole new clusters, which no walk
change can do.

### Where that leaves the energy asymmetry — it moves one level up

Part 8's 93.6% (target more energetic than the holder, median 7.8×) is **not**
"the small shower got there first". The cross-cluster absorbers — `pass4_angle`
39 seg, `in_other_clusters_seg_cone` 28, `from_vertices` 20,
`in_other_clusters_A` 13 — reached these clusters and **assigned them to the
smaller shower**. It is still an assignment error; it just happens at those
sites, not in the walk.

**The next discrimination is the same question one level up**, and it needs the
same kind of probe at those sites: *when a cross-cluster absorber gives a
cluster to shower B, was shower A ever a candidate for it?*

- **A was a candidate and lost** → a tie-break rule ("prefer the more energetic
  shower") fixes 93.6% by charge, fires only where there is a genuine
  competition, and is therefore naturally low-blast-radius. It is **not** a
  geometric predicate, so Result 3 does not falsify it.
- **A was never a candidate** → the candidate enumeration is too narrow, and the
  fix is upstream of any predicate.

Those absorbers currently tape admissions only, so the question is not
answerable from anything on disk today.

The 463565 π⁰ warning still stands against any merge-side change.

## Part 10 — the cross-cluster probe: the site never runs for the shower that needed it

Second probe shipped (`WCT_SHOWER_XCLUS_DEBUG`, `NeutrinoShowerClustering.cxx`):
the pass-4 direct-cone **rejections** (both the cheap `angle_v2 > 30` filter and
the acceptance disjunction), the **owned-segment skip** at two sites, and every
**rival's arbitration metric**. Gate PASS with the env unset; `wcdoctest-clus`
235/235; freshness lib 19:52:43 > src 19:52:06.

### Result

| | segments | charge | share |
|---|---|---|---|
| **ABSENT** — the pair never entered the loop | **56** | 1.734e7 | **79.0%** |
| **REJECTED** — the target evaluated it, the cone refused | 21 | 4.477e6 | 20.4% |
| **OWNED** — skipped because another shower held it | 1 | 1.428e5 | 0.7% |

And the split is **per shower, not per segment**:

| target shower | XCLUS lines it emitted | its segments |
|---|---|---|
| 122660 / 9110 | **NONE** | 11 ABSENT |
| 181050 / 15006 | **NONE** | 8 ABSENT |
| 463565 / 13001 | **NONE** (event total 0 over 0 showers) | 26 ABSENT |
| 469665 / 15003 | **NONE** | 11 ABSENT |
| 21073 / 60081 | 28 | 11 REJECTED |
| 105946 / 55063+56056 | 80 | 7 REJECTED, 1 OWNED |
| 342199 / 25109 | 38 | 3 REJECTED |

**Four of seven target showers never enter the cross-cluster loop at all**, and
they carry 79% of the charge. On **463565** — the largest event and the one the
owner ruled on first — the loop emits **zero lines over zero showers**: the
cross-cluster absorber never executes in that event.

### The other 20% is not a near miss either

All 21 REJECTED fail the cheap `angle_v2 > 30` filter, and they fail it wide:
angle_v2 spans **37.9° – 124.5°** against a 30° gate, with only 3 of 21 inside
45°. Catching the bulk would mean opening the gate past 65°, which admits
everything. That is Result 3's "no geometric bound" arriving a third time, now
with the exact quantity and the exact margin.

### Three hypotheses tested, three dead

| round | hypothesis | verdict |
|---|---|---|
| Part 3 | loosen the F12 walk-add guard | dead — 69 declines / 2.852e8 to get 4 wrong, interleaved |
| Part 9 | walk ordering / `used_segments` contention | dead — **0 of 78** |
| Part 10 | cross-cluster threshold, arbitration, or owned-skip | dead — 0.7% OWNED, and the rejections miss by 2× the gate |

What is left is **structural and upstream of every predicate**: which showers
`shower_clustering_with_nv_from_vertices` iterates as absorbers in the first
place. Four of the seven showers the owner says should have grown were never
offered the chance. That is not a knob at an existing seat — it is the
enumeration itself, and it is the only lever this round has not falsified.

**Both probes are default-off and shipped**, so the next round starts with the
instrument already in the tree rather than rebuilding it.

## Part 11 — picking this up later

Everything below is what a future session needs and nothing it can re-derive
cheaply.

### What is settled and must not be re-litigated

| claim | evidence |
|---|---|
| the under-clustering is real, not a labelling artifact | owner scan **17/17 merge**, tag `emscan-0829-pr130qmiss` |
| no **geometric** merge predicate exists | approved merges span d_start **9.4–182.8 cm**, angle **9.4–156.6°** |
| the F12 walk guard cannot be loosened | 69 distinct declines / **2.852e8** to get 4 wrong; two 16.5 cm pdg-13 declines with opposite verdicts |
| walk **ordering** is not the mechanism | contention probe: **0 of 78**, 100% UNREACHED |
| it is **cross-cluster acquisition** | 96.2% of the missing charge is in clusters the shower does not hold at all |
| cross-cluster **thresholds** cannot reach it | the 21 evaluated rejections fail `angle_v2 > 30` at **37.9–124.5°** |
| cross-cluster **arbitration** is not the gap | `shower_pass4_best_owner` has been SBND-ON since 2026-08-28; only 0.7% is an owned-skip |

### The one live lead

**Which showers `shower_clustering_with_nv_from_vertices` iterates as
cross-cluster absorbers.** Four of the seven target showers — 122660/9110,
181050/15006, 463565/13001, 469665/15003 — emit **no** `SHOWER_XCLUS` lines at
all and carry **79% of the charge**; on 463565 the loop runs zero times over
zero showers. They were never offered the chance, so no predicate, ordering rule
or tie-break at any existing seat can reach them.

**Next measurement**, and it is a read not a knob: instrument the enumeration
itself — which showers reach that pass, and why those four do not. Only once
that is known does a change have a defined shape. Note it will not be a local
default-OFF guard like everything in pr/123→pr/130; expect a full 239-event gate.

### The instruments, already in the tree

| probe | env var | site |
|---|---|---|
| walk contention | `WCT_SHOWER_BLOCKED_DEBUG` | `PRShower.cxx` flood-fill, toolkit `0cccb5f5` |
| cross-cluster REJECT / OWNED / RIVAL | `WCT_SHOWER_XCLUS_DEBUG` | `NeutrinoShowerClustering.cxx`, toolkit `deca3467` |

Both default-OFF and byte-identical with the env unset (gate PASS vs
`work-pr130r1-bon{,141}-*`, archives and calib dumps).

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_blocked_probe.sh gate    # byte-identity gate, env unset
./scripts/pr130_blocked_probe.sh probe   # BLOCKED tape  -> work-pr130r1-blkon-*
./scripts/pr130_blocked_probe.sh xclus   # XCLUS tape    -> work-pr130r1-xcon-*
scripts/pr130_blocked_census.py > docs/pr/pr130-blocked-census.txt
scripts/pr130_xclus_census.py   > docs/pr/pr130-xclus-census.txt
```

### Records not to overwrite (M13)

`em_labels/emscan-0829-pr130qmiss/` (9 files — the owner's own scan, the only
one authored by him rather than an agent), `bee/pr130r4/`, and the
`work-pr130r1-{blkon,blkgate,xcon,xgate}-*` arms.

### Two things still attached to any future merge-side change

1. **The 463565 π⁰ warning** (top of this doc): the merge is licensed by
   inseparability, not identity — the owner said there *should* be two showers.
   Fixing the energy can cost the two-gamma separation. Gate on a π⁰ metric,
   not only on q_extra.
2. **The q_extra caveat**: this scan asked only what should be merged *in*, so
   the 4.661e6 → 0 collapse is **not** evidence that nothing is over-clustered.

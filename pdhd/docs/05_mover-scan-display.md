# PDHD doc 05 — a scan display for the wrapped-channel movers

**Owner request, 2026-09-06:** *"this scan is not easy with Bee, though it is very useful for me
to understand things. We should be is like stm_display, em_display. You can then show what you
want me to look at as well as the choices that you want me to choose from. Can you do that and put
the display in port 5017?"*

This doc covers the display only. The physics it serves — what the wrapped-strip channel-lookup
bug was, what flipping it changed, and the **pre-registered bar** the labels are scored against —
is [doc 04](04_stm-tagger-scan.md) §9–§11. Nothing here re-opens that bar.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/d05_scan

python3 selftest_d05_scan.py                 # rc 0; proves the blind, the alphabet, the arm
./serve_d05_scan.sh 5017                     # --tag NAME to namespace a second pass
#   from a laptop:  ssh -L 5017:localhost:5017 <user>@wcgpu1
#   then open       http://localhost:5017/d05_scan_viewer

# after labelling, score with the UNCHANGED doc 04 scorer:
cd .. && python3 docs/scripts/d04_movers_score.py \
    --sheet work/d05_scan_labels/movers0/filled_sheet.tsv \
    --key   bee-pr-run029107-d04movers.KEY.tsv
```

Port 5017 is **shared with `pdhd/stm_scan`** (img_plot owns 5013, pd_plot 5014, ql_scan 5015,
wf_scan / pdvd ql_scan 5016). Run one or the other, not both — a second server on a busy port
does not fail loudly, it just logs `port 5017 is already in use` and leaves the *old* app
answering, which is how a stale display gets scanned. Check with `ss -ltnp | grep 5017`.

Files, all forked **by duplication** from `pdhd/stm_scan/` (which is untouched and still serves
its own scan):

| file | what |
|---|---|
| `d05_scan/d05_scan_viewer.py` | the Bokeh app |
| `d05_scan/serve_d05_scan.sh` | `bokeh serve` wrapper, port + `--tag` |
| `d05_scan/selftest_d05_scan.py` | headless; §6 |
| `work/d05_scan_labels/<tag>/labels.json` | the labels, rewritten on every click |
| `work/d05_scan_labels/<tag>/filled_sheet.tsv` | the same labels as a sheet the doc 04 scorer eats |

## 1. Why a display and not the Bee set

The Bee set (`bee-pr-run029107-d04movers.zip`, doc 04 §11.1) is correct and stays as the record.
It is just a poor instrument for *this* task: you have to find the object among ~100 clusters,
hold the eight-label alphabet in your head, and type the answer into a TSV in another window. The
scan is 39 objects; the friction is the whole cost.

This app puts the object on screen with its ends marked and the eight choices as buttons. It
draws exactly the same charge Bee does, from the same `mabc-pr.zip`.

## 2. What it shows

**Three overview panels** — side (Z vs Y), top (Z vs X), end (X vs Y) — each at the **full
detector volume** with the active boundary (the `pr.jsonnet` BoxFiducial) drawn dashed. Full
volume is deliberate: an object auto-zoomed to its own extent looks contained in every
projection, which would invert THRU/STOP judgements. `Zoom to object` exists and is not the
default.

- **coloured** = the cluster, colour is its charge, rescaled per object.
- **grey** = *all* other charge in the event. With `Dense context` on (default) **every** grey
  point within the context radius (20/40/80/150 cm) is drawn and the rest of the event is thinned
  1-in-N. Both selections are purely geometric over all other charge — neither is "the clusters
  the tagger considered", which would leak the answer.
- **red circles** = the two ends along the object's principal axis (0.5 / 99.5 percentile, so a
  single stray point cannot define an end).
- **dark grey patches** = dead channels. Bee writes these as `[y, z]` polygons with no drift
  extent, so they are drawn on the **side view only**; faking them into the other two would
  invent information.

**Two close-up panels**, one per end, in a plane you choose, at ±15/25/50/100 cm. Inside that
window **nothing is thinned** — every point, cluster and context alike. This is the panel that
decides most of these labels: a stopper's end is charge that simply ends, a fragment's end has
grey charge carrying on along the same line, and 1-in-N decimation renders that continuation as a
handful of dots. It is not idle: over the 39 objects, **18 have other charge within ±25 cm of an
end** (up to 1204 points), and the other 21 have none at either end. The self-test asserts that,
so the panel cannot quietly become decoration.

**A measured line under the panels**: the straight-line extent, and for each end its distance to
the nearest active-volume face (naming the face) and to the nearest dead channel in (y, z). That
is arithmetic a scanner would otherwise do by squinting at an axis. It **decides nothing** — an
end can sit on a wall and still belong to a longer object that continues in grey — and the text
says so on screen.

## 3. The blind, which is a different argument here

`stm_scan` blinded by **byte-identity**: the two layers it drew were SHA-256-identical between its
two arms, so the pixels could not encode which arm you were looking at.

**That argument is dead here and is not used.** The whole point of the wrapped-channel fix is that
the per-point charge changes: `q` differs on **141 434 of 161 854** clustering points in event 12.

This scan is blind for a different reason: it draws **one arm**, so there is no "which arm is
this" for the pixels to encode, and the thing being tested — which *direction* each verdict moved
— is not in the charge at all. It is in `bee-pr-run029107-d04movers.KEY.tsv`, which this process
never opens.

That makes the blind an **absence**, and an absence has to be proven rather than asserted. The
self-test enumerates the zip members the viewer actually opened across all 39 objects and asserts
the set is exactly `{clustering-global, channel-deadarea-*}` — no `stm`, `stm_fit`, `stm_tagged`,
`steiner_graph`, `steiner_terminals`, though all five sit in the same archive. It also asserts the
key file is never handed to `open()`/`ZipFile` (it is *named* on screen, in the scoring command
the export button prints).

The **stratum** (`>=1000` / `200-1000` / `<200` points) is deliberately absent from the UI, as it
is in `stm_scan`. The point count is shown; calling a small object UNCLEAR has to be your
judgement, not a nudge from a label that says "small".

## 4. Which arm is drawn — and the one object where it matters

`ARM = d05mON`, the fixed (production) arm. Two reasons, and one caveat that had to be measured.

**The sheet came from that arm.** All **39/39** rows' `npts` match `d05mON` exactly. Against
`d05mOFF`, `d04bee` and `d04prb` it is 38/39 — the same single row each time.

**The taggers run on the PR job's re-partition, not the clustering job's.** Doc 04 §10.2 proved
"only `q` differs" over `mabc-all-apa.zip`, the *clustering job's* output. The cosmic taggers see
`mabc-pr.zip`, where the PR chain has re-clustered, and there the two arms do **not** agree on the
cluster list:

| event | 0 | 1 | 12 | 16 | 20 | 22 |
|---|---|---|---|---|---|---|
| clusters in `mabc-pr.zip`, ON / OFF | 98 / 98 | 86 / 99 | 126 / 96 | 104 / 103 | 131 / 150 | 103 / 92 |

So the naive reading — "idents are comparable, §10.2 proved it" — does not follow. The right
measurement is over the clusters the taggers actually evaluate, and it comes out **stronger** than
the claim it replaces:

- every ident `TaggerCheckTGM` evaluates **is** a Bee `cluster_id` (0 exceptions, both arms, all
  six events), so the PR log ident and the Bee cluster id are the same numbering;
- the **evaluated ident set is identical between the arms on all six events** (574 idents each,
  matching §10.3's "TGM evaluated 574 = 574");
- **565 of those 574 clusters have bit-identical point sets** across the arms.

The cluster-count spread above is entirely in clusters *below* the tagger's threshold, which never
enter the mover derivation.

**The 9 that differ** are `evt1/36`, `evt12/{28,35,37}`, `evt16/67`, `evt20/31`, `evt22/{31,34,56}`
— the PR chain moved points between clusters. **One of them is on the scan sheet**: `evt 20
cluster 31`, 11 432 points in the ON arm and 11 065 in the pre-fix arm. That row is judged as the
ON-arm object, which is the one the fix produces. It is the only row where the choice of arm
changes what is drawn.

Doc 04 §10.2/§10.3 has been corrected in place to say this.

## 5. Labels, and who scores them

Eight buttons, in three rows that mirror the question:

```
the cluster IS the whole object:        THRU        STOP        CONT
the cluster is only PART of it:         FRAG>THRU   FRAG>STOP   FRAG>CONT
                                        MESSY       UNCLEAR
```

`FRAG>` carries the **full object's** verdict on purpose: a fragment of a through-goer and a
fragment of a stopper are opposite physics truths, and one undivided FRAG bucket throws the second
kind away. `CONT` / `FRAG>CONT` land in neither numerator of the primary bar — that is deliberate,
not an oversight (doc 04 §11.5).

Every click writes `work/d05_scan_labels/<tag>/labels.json` atomically — a reload or a restart
loses nothing — and refreshes `filled_sheet.tsv` beside it. The committed blind sheet
`bee-pr-run029107-d04movers.sheet.tsv` is **never written to** (M13), and the self-test checks its
SHA-256 is unchanged after a full synthetic pass.

**This directory contains no scorer.** The bar was pre-registered in doc 04 §11.5 and frozen at
commit `546fcbaa`; `docs/scripts/d04_movers_score.py` applies it and is used **unchanged**. Zero
new scoring logic is a stronger statement than any promise not to re-tune. The self-test proves
the two ends fit: the eight button strings are asserted **equal as a set** to the scorer's `VALID`,
a synthetic full sheet scores rc=0, and a free-text label still makes the scorer exit 2 REFUSING
rather than fall through into a class.

## 6. Self-test

`python3 selftest_d05_scan.py`, rc 0, no server needed. It imports the app under a scratch tag and
checks:

| # | check | result |
|---|---|---|
| 1 | 39 objects, each resolving to charge whose point count **equals the sheet's `npts`** | ok |
| 2 | the blind: members opened = `{clustering-global, channel-deadarea-*}`, no tagger layer; key never opened | ok |
| 3 | the 8 buttons are exactly the scorer's `VALID` set | ok |
| 4 | a full synthetic sheet scores rc=0; a free-text label gives rc=2 | ok |
| 5 | the committed blind sheet's SHA-256 is unchanged (`1af0c7fb27a6…`) | ok |
| 6 | `ends_of` / `wall_gap` / `dead_gap` finite on all 39, including the 3-, 4- and 9-point ones | ok |
| 7 | 18 of 39 objects carry other charge within ±25 cm of an end (max 1204) | ok |

Check 1 is the one that would have caught the worst silent failure: `d04bee` and `d04prb` also
have all six events on disk, and pointing the display at either would have drawn a different
epoch's objects under the sheet's numbers.

## 7. It was used

The owner scanned all 39 objects on it in one pass on 2026-09-06 (tag `movers0`, no notes) and the
pre-registered bar **PASSES**: 24/24 through-going, 0/24 stopping. The result, including the one
adverse finding it turned up — two stopping muons that lost their STM tag — is
[doc 04 §11.7](04_stm-tagger-scan.md); the labels joined to the key are
`bee-pr-run029107-d04movers.LABELS.tsv`.

Two things the display can take credit for, and one it cannot. All nine `UNCLEAR` labels landed
below 1000 points and none at or above it, which is what §11.4's judgeability measurement
predicted — so the panels were sufficient wherever the bar depended on them. The close-ups earned
their place: `evt 16 cluster 90` came back `FRAG>THRU`, a distinction that only exists if you can
see grey charge continuing past an end. What it cannot take credit for is the verdict itself.

## 8. Not done

- Nothing further on this scan. The display stays for the next one — the natural user is doc 04
  §12 item 0, the two lost stoppers, which needs `stm_fit` / `steiner_graph` layers this app
  deliberately does not open and so would need a non-blind variant.
- No 2-D wire-plane (channel vs time) view: the ctpc is not in the Bee zip, so "the end is in a
  dead region" is answerable here only through the (y, z) dead-area polygons and the measured
  distance to them.
- No keyboard shortcuts. Bokeh 3 binds JS handlers silently to nothing in several ways
  (`feedback_bokeh3_silent_js_traps`); the buttons are server-side callbacks and are tested.

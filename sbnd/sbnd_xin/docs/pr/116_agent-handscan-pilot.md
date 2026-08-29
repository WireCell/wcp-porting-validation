# pr/116 — can the model hand-scan? a 5-event pilot on the 5018 sample

**Scope.** The owner asked whether I can do the *visual* hand scan myself — from
the 3-D view, rotated about the neutrino vertex, the way a person does it — and
to try five events first. Five events of the 141 on port 5018
(doc [pr/114](114_em-pi0-handscan-display.md) §27) are scanned and recorded, in
the display's own label format, under a **fresh tag `emscan-0828-agent5`**.

**No C++, no jsonnet, the toolkit repo is untouched, so no A/B gate is owed and
none is claimed.** Two new files under `em_display/`; `em_display_viewer.py`,
`em3d.py` and `em_geom.py` are **byte-identical** (md5 checked before and after —
`bokeh serve` re-executes the app script on every new browser session, so an edit
would have changed what the owner sees on 5017/5018 on their next tab, mid-scan,
with no restart and no warning).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# 1. a server of my own -- never point the harness at 5017 or 5018 (M13)
./em_display/serve_em_display.sh 5019 --scan-tag emscan-0828-agent5 \
    --manifest $PWD/em_display/em114c-manifest.tsv \
    --prepdir  $PWD/em_display/emprep-c

# 2. capture: PNGs + state.json per event, no judgement, never saves
python em_display/scanbot_capture.py --event evt400504 --port 5019
python em_display/scanbot_capture.py --event evt280159 --port 5019 \
       --pio-pairs "12119:95114,24015:95114,12119:24015"

# 3. record: replay the judgement and press Save
python em_display/scanbot_record.py --decisions em_display/decisions-agent5.json \
       --port 5019 --only evt400504
```

Captures land in `/home/xqian/tmp/emscan-agent5/<event>/`; labels in
`sbnd_xin/em_labels/emscan-0828-agent5/`; the figures below in `sbnd_xin/pics/`.

## 1. Yes — and here is the loop

Three phases with a **disk handoff**, not one interactive sequence. The browser
harness carries 68 hand-tuned sleeps (3500–6000 ms after every `event_select`
change); a single long loop would blow every timeout and force a re-capture on
each re-judgement.

| phase | file | what it does |
|---|---|---|
| **A capture** | `em_display/scanbot_capture.py` | drives headless chromium, writes 14–39 PNGs of the 3-D canvas + a `state.json`. **No judgement, never saves.** It does change server-side session state — event, shower and camera selection, and with `--pio-pairs` the two γ slots, which is the only way to read the mass conventions back — but on a session that is new every invocation, and nothing reaches disk. |
| **B judge** | me | read the PNGs and the tables. No browser in the loop. |
| **C record** | `em_display/scanbot_record.py` | replays a decisions JSON: select shower → mark segments → assign γ slots → verdict + confidence → note → Save. |

Two things had to be learned before any of it worked.

**The camera is written, not dragged.** A drag is 0.0075 rad/px with the
elevation clamped at the pole (`em3d.py:524-530`), so a fixed drag sequence
*ratchets*: in the first dry run every view of the second shower was top-down and
two shots came out byte-identical. Angles are now set through `cam_src` — the
same route the preset buttons take (`em3d.py:116-122`) — so every event is shot
from the same six cameras. One real drag is still performed, as the check below.

**A clip must be scrolled into view first.** `getBoundingClientRect` is
viewport-relative; the acceptance plot sits below the fold and its first
screenshot came back **silently truncated** to 200 px of its 330.

**Framing.** `frame the reco` uses the bounding sphere of *everything* including
far vertices: on evt400504 that is R 237 cm for an event 20 cm across — the whole
event was 100 px of 2200. The sweep therefore centres on the **ν vertex** and
sets the half-span from the 95th percentile of segment-point distance to it.

### 1.1 What one event costs

| event | shots | capture | PNG |
|---|---:|---:|---:|
| evt284791 (10 showers, 1 in scope) | 14 | 23 s | 2.3 MB |
| evt284200 | 19 | 30 s | 3.6 MB |
| evt400504 | 21 | 40 s | 4.5 MB |
| evt175896 | 21 | 40 s | 3.6 MB |
| evt280159 (35 showers, 5 in scope, 3 extra π⁰ pairings) | 39 | 86 s | 10.4 MB |

**Extrapolated to all 141: about 1.5 h of capture and ~700 MB of PNG.** The
judging is the expensive half, not the capture.

**Shower scope rule**, stated because on a busy event it decides what was *not*
looked at: every shower with `kine_best ≥ 20 MeV`, plus any with `pio_id ≥ 0`,
capped at the top 8 by `kine_charge`. On evt280159 that reaches **5 of 35** — the
other 30 are all under 6.2 MeV and I did not look at them.

### 1.2 The judging rule

**Image for the judgement, table for the identification.** The picture decides
whether a shower is missing the blob to its left; the segment ids to mark come
from `cand_src` (`dist`, `angle`, `tier`, `in shower`, `absorbed by`). No mark is
ever resolved from a pixel position.

The visual grammar carries it: soft-yellow 9 px halo = what the reconstruction
put in the selected shower, one colour per shower with **grey = no shower claims
it**, orange arrow = the shower's axis, blue star = the ν vertex, X = the shower
start (`em_display_viewer.py:353-375`).

One quantitative channel was added on top, because the display shows angle to the
axis but not transverse spread, and *track vs shower is a width question*: a PCA
over each shower's own segment polylines giving length, width (2×transverse RMS),
`w/l`, distance of the closest member point to the ν vertex, and the largest
internal gap. **Its known failure mode, and it bit twice:** the PCA measures the scatter of
the member *set*, not the width of a shower. On evt175896 it returned `w/l` 0.29
for an object that is two straight pieces at an angle; on evt284791 it returned a
16.4 cm width for a chain of 2-point stubs strung along the axis, and restricting
to the four members with ≥5 fitted points collapses that to 6.8 cm. **Neither
verdict below rests on it.** Note also that the shower table's own `length` and
this extent are different definitions and can disagree by 2× on exactly these
objects (evt284791: 27.7 vs 62.1 cm).

## 2. The five events

Selection was **purposive, not random** — one per origin bucket plus a busy event
and a π⁰ event, chosen so the pilot shows where I break rather than only whether
I work. A large run should go in manifest order.

| event | sample / origin | verdict (shower) | confidence | π⁰ | marks |
|---|---|---|---|---|---|
| `evt400504` | mcp1k numucc_em | **correct** (63024) | likely | reco's pair confirmed | 1 OUT |
| `evt280159` | mcp2k numucc_em | **over-clustered** (12119) | unclear | **reco pairing disputed** | — |
| `evt284791` | mcp1k numucc_em | **correct** (37016) | likely | only one γ in the event | — |
| `evt175896` | mcp1k nuecc | **vertex-bad (undecidable)** (17044) | unclear | — | 2 OUT |
| `evt284200` | mcp1k other_em | **correct** (6013) | likely | partner γ absent | — |

`evt54341` was deliberately excluded: the owner has already labelled it on 5018
and I had read that label, so it is not a clean comparison point.

### 2.1 evt400504 — good, and the mark that moves the mass

Three prongs converge on the star: the 57 cm muon `6001`, and two EM objects
([`116_400504_vertex-3prong.png`](../../pics/116_400504_vertex-3prong.png)).
Shower 63024 (153.5 MeV) is one cone, 3.5 cm wide over 63.8 cm, and the detached
`65033` at 60.6 cm / 11.9° is on-axis and belongs
([`116_400504_shw63024.png`](../../pics/116_400504_shw63024.png)). Every
non-member sits at >120° from the axis: nothing is missing.

Shower 62014 (67.7 MeV) is good apart from **`21003`** — a 0.8 cm `pdg 13`
fragment absorbed by `pass4_angle` at 27.6 cm / 26.8°, visibly off the shower
body ([`116_400504_shw62014.png`](../../pics/116_400504_shw62014.png)). Marked
OUT. The charge is negligible, but the effect is not nothing: with the mark
applied γ2 drops 67.7 → 63.9 MeV and the hand mass moves from the reco's stored
**145.9 to 141.8 MeV** on the vertex convention — from 10.9 MeV off 135 to 6.8.

### 2.2 evt280159 — the reco paired a photon with something sitting on the vertex

The busy one: 35 showers, the 173 cm muon `12118` ending on the star
([`116_280159_vertex.png`](../../pics/116_280159_vertex.png)).

**Shower 12119 (446.3 MeV, 17 segments over clusters 12/84/85/87/89) reads as
over-clustered**, and the reason is not its shape — it is contiguous (largest gap
2.6 cm) and 7.8 cm wide, which is shower-like
([`116_280159_shw12119.png`](../../pics/116_280159_shw12119.png)). It is that
**its closest member point is 0.0 cm from the ν vertex**, its seed `12119` is in
**cluster 12 — the same cluster as the 173 cm muon**, which reads `dist 0.0 /
angle 0.0` against it, and it then absorbs the vertex hadronic clusters including
`84067` (`pdg 2212`, 10.3 cm) and `87083` (`pdg 13`). That reads as a vertex blob
seeded on the muon stem. **No segment marks: I could not name which pieces with
confidence**, which is why the confidence is `unclear`.

**The π⁰ pairing looks wrong.** The reconstruction paired 12119 + 24015 (stored
134.0 MeV). But 12119 is attached to the vertex and so cannot be a converted
photon — while **shower 95114 (129.6 MeV, compact 20.6 × 3.3 cm, contiguous,
detached by 42.7 cm — the most photon-like object in the event) is left
unpaired**. Priced through the display's own formula:

| pairing | axis convention | vertex convention |
|---|---:|---:|
| 12119 + 24015 — *the reco's own* | 200.5 | — (γ on the vertex: no chord) |
| 12119 + 95114 | 444.1 | — |
| **24015 + 95114** — *both detached* | **133.9** ✓ accept window | **75.6** |

Stored as a candidate, **not as a settled answer**: the two conventions disagree
by a factor of 1.8 on my own preferred pair, which is exactly the inconsistency
pr/114 §6.1 documents in the code
([`116_280159_pio-alt-24015-95114.png`](../../pics/116_280159_pio-alt-24015-95114.png)).

### 2.3 evt284791 — good, and sparse is not the same as under-clustered

Shower 37016 (132.9 MeV, 11 segments) is detached — its closest member is 26.4 cm
from the vertex — and it *looks* alarmingly fragmented
([`116_284791_shw37016.png`](../../pics/116_284791_shw37016.png)): the members are
a chain strung along the axis from t = −38 to +24 cm, the 19-point seed nearest
the vertex and a trail of 2- and 3-point stubs running out to 88 cm. The
temptation is to call it under-clustered.

**The verdict rests on the non-members, not on the shape.** Every non-member
within 40 cm lies at **103–173°** from the axis, i.e. behind the start, and the
six of them are 0.2–7.1 MeV vertex fragments on the muon side. There is nothing
in the event to add — that is the whole argument, and it does not need a width.

The shape number is *not* usable on this object and the first draft of this entry
leaned on it wrongly: transverse RMS 8.2 cm (`w/l` 0.26) is the scatter of the
stubs, and restricted to the four members with ≥5 fitted points it is 3.4 cm
(`w/l` 0.13). **What I did not test:** whether those far stubs, out to 88 cm,
genuinely belong to this shower. `correct` here means *nothing is missing*.

### 2.4 evt175896 — a straight line through the vertex, called two 200 MeV showers

The νe CC event, and the one I could not resolve. Within 25 cm of the vertex the
charge is **one straight narrow line**: 61 fitted points, transverse RMS 1.49 cm,
PCA singular values 6.9 / 1.1 / 0.7 — with points on **both** sides (35 beyond
−3 cm, 11 beyond +3 cm). Rotating 70° off the default view shows it as one
continuous streak
([`116_175896_line-through-vertex.png`](../../pics/116_175896_line-through-vertex.png)).

The reconstruction splits that one line into two showers of **253.9 and 217.8
MeV**: 17044 and 17043 are both cluster 17, both `conn 1`, both with a member
point at **0.00 cm** from the vertex, and each reads `dist 0.0 / angle 0.0`
against the other (site `examine_showers_retarget_seed`). Each then attaches a
distant clump ([`116_175896_shw17043.png`](../../pics/116_175896_shw17043.png)).

**Two readings, and I cannot separate them:**

- **(a)** the vertex is wrong and sits in the middle of a through-going track;
- **(b)** the vertex is right, 17043 is the electron — a 250 MeV electron *does*
  start as a narrow MIP stem for about one radiation length — and 17044 is
  spurious.

Recorded as `vertex-bad (undecidable)` with confidence `unclear`, which is what
that bucket is for.

**One correction the saved record forced on me, and it is the useful part of this
entry.** I marked `66037` and `66041` (23 cm of `pdg 2212` proton track, 33 cm
out) OUT of 17044 and first wrote that they sit "at 168° behind the axis". That
was **17043's** axis. `mark_metrics` recomputes against the shower the mark is
recorded on, and against **17044's own axis they are at 9.1° and 11.5°, tier 1,
absorbed by `pass3_cone`** — the gate accepted them legitimately. The marks stay,
but the note now says what they are: **a PID statement, not a geometry one.**
Reading my own record back is what caught it.

### 2.5 evt284200 — good, and a clean answer to "why is there no π⁰ here"

`other_em` — the bucket pr/113's ladder never named. It is a clean single-photon
topology: a hadronic V at the vertex (cluster 31 — `31028`, `31029` both `pdg
2212`, plus three more, **all correctly left unowned**) and shower 6013 (183.5
MeV) detached 20.0 cm from it, 62 cm long, 6.0 cm wide, branching
([`116_284200_vertex-V.png`](../../pics/116_284200_vertex-V.png)). Every
non-member within 40 cm is at 47–172°.

The reason this event has no π⁰ group is not a pairing failure: **there is no
second photon anywhere in the reconstruction**. For `other_em` that is worth
knowing — the bucket may be mostly "π⁰ with one gamma lost", not "π⁰ mis-paired".

## 3. Verification

| # | check | result |
|---|---|---|
| 1 | app source unmodified — md5 of `em_display_viewer.py`, `em3d.py`, `em_geom.py`, `prep_em_scan.py`, `em114_categorize.py` | **all OK** before and after |
| 2 | 5017 and 5018 untouched | both pids still listening; `emscan-0827` still 97 labels; `emscan-0828-beam141` still 1; both manifests unchanged in mtime **and** size |
| 3 | the screenshot clip lands on the 3-D canvas | eyeballed on the first PNG before the other four ran; the truncated-acceptance-plot bug was caught this way |
| 4 | **the pixels move, not just the readout** | a real 100 px drag gives Δaz **0.7500 rad** against 0.0075 rad/px expected, **and** `seg3_src` *and* `cloud_src` both reproject. The JS reprojects only registered sources (`_LINE3`, `_PT_SRC`); one outside those registries would freeze at a stale camera with a perfectly correct readout |
| 5 | label round-trip in a **fresh** session | **5/5** restore exactly: verdict radio, confidence radio, every mark drawn as a halo *and* named in `marks_div`, stored π⁰ candidates, and the green "already scanned" chip. (pr/114 §12.6a is the record of this exact path once being silently broken) |
| 6 | schema conformance | verdict/confidence verbatim from `EM_VERDICTS`/`CONF`; `event`/`eventNo` agree; `pio.candidates` present |
| 7 | `selftest_em3d_browser.py` — the app itself still works | **90 checks, 0 failures** (it starts its own server on 5029, independent of mine) |
| 8 | the tag guard actually refuses | both scripts pointed at **5018** exit 1 before selecting an event |

**Both** scripts read the served scan tag off the page and refuse to run if it is
not the one named on the command line, so a mistyped `--port` cannot reach
`emscan-0827` or `emscan-0828-beam141`. Tested by pointing capture at the owner's
live 5018: `port 5018 is NOT serving tag 'emscan-0828-agent5' … refusing`, exit 1,
before any event was selected. It also asserts after the
first save that the file is on disk and that `save_note` does not say *refused* —
`write_allowed()` refuses into a populated tag with a **Div message, not an
exception**, which a scripted click would swallow in silence.

## 4. What this pilot does NOT settle

- **Whether I am right.** Five events against no ground truth measures
  feasibility, not accuracy. The number this does not produce is an agreement
  rate. If the owner scans these same five independently on 5018 — separate tag,
  no contact between the records — that comparison is clean and costs one pass.
- **"Correct" is a fast pass**, the same caveat as pr/115 §9: it means nothing
  jumped out at the six cameras and the tables listed here, not verified segment
  by segment.
- **My verdicts are scoped to clustering and silent on PID**, except where a note
  says otherwise (evt175896).
- **On evt280159 I looked at 5 showers of 35.** The over-clustering verdict is a
  claim about shower 12119, not about the event.
- **Two of five came out `unclear`**, and both are the interesting ones. That is
  the honest rate on a purposive sample chosen to be hard.

## 5. The roll-up problem, and one thing to fix before a large run

`on_save` stores **one** `em.verdict`, for the shower selected at save time
(`em_display_viewer.py:4160-4218`), while `marks_by_shower` holds many. The rule
used here: **the verdict is always the leading EM shower's**, and every other
shower's finding lives in the note plus its marks.

That produces a case `em114_categorize.py` will bucket differently from my own
verdict. On evt400504 the verdict is `correct` (shower 63024) while there is one
OUT mark on a *different* shower, 62014 — and pr/115's rule 3 reads an OUT mark on
a member as "the real remove-this statement" and would file the event under
over-clustering. Both are right about different things, and it is the same
tension pr/115 §12 records for `evt64591`.

Notes are written in the vocabulary `em114_categorize.py`'s regexes already know
(`RE_OVER`, `RE_UNDER`, `RE_VERTEX`) and prefixed `[agent]`, so a large run can be
bucketed by the same rule as the owner's 97. That script hardcodes
`em114-manifest.tsv` at `:21` and would need a `--manifest` flag — a
default-preserving addition, provable by diffing `pr115-handscan-buckets.tsv`
before and after. **Not done in this pilot**, and noted so a large run does not
invent a second taxonomy.

## 6. Files

| file | |
|---|---|
| `sbnd_xin/em_display/scanbot_capture.py` | phase A — capture, never saves |
| `sbnd_xin/em_display/scanbot_record.py` | phase C — replay a judgement, with the tag guard |
| `sbnd_xin/em_display/decisions-agent5.json` | the five judgements, as input to phase C |
| `sbnd_xin/em_labels/emscan-0828-agent5/` | the five labels, openable in the display |
| `sbnd_xin/pics/116_*.png` | the ten frames the calls above rest on |
| `sbnd_xin/docs/pr/116_agent-handscan-pilot.md` | this document |

`*.json` and `*.png` are gitignored in `wcp-porting-img`, so the labels, the
decisions file and the ten figures need **`git add -f`** (CLAUDE.md M9). Nothing
is committed.

To review: serve the tag on a spare port and open it — 5018 stays as it is.

```bash
./em_display/serve_em_display.sh 5019 --scan-tag emscan-0828-agent5 \
    --manifest $PWD/em_display/em114c-manifest.tsv --prepdir $PWD/em_display/emprep-c
# ssh -L 5019:localhost:5019 wcgpu1.phy.bnl.gov  ->  http://localhost:5019/em_display_viewer
```

---

# Part 2 — the full 141-event campaign

## Repro

    # the scan itself (5 agents x 3 waves; brief, shards, ledgers, decisions)
    ls /home/xqian/tmp/emscan-bulk/{BRIEF.md,shards,ledger,decisions}
    PY=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python3.11
    $PY /home/xqian/tmp/emscan-bulk/rollup.py            # verdict census + hidden category
    $PY /home/xqian/tmp/emscan-bulk/scope_audit.py       # charge the scope rule dropped
    $PY /home/xqian/tmp/emscan-bulk/recheck_gate.py evt98844 70063 63040 66043 60037
    $PY em_display/em114_categorize.py --tag emscan-0828-agent5 \
        --manifest em_display/em114c-manifest.tsv --use-verdict

    # the confirm-set display, with Bee links
    ./em_display/serve_em_display.sh 5027 --scan-tag emscan-0828-agent5 \
        --manifest $PWD/em_display/em116confirm-manifest.tsv \
        --prepdir $PWD/em_display/emprep-c

## Result

141 of 141 events labelled in `em_labels/emscan-0828-agent5/`; assigned =
ledgered = on disk for every agent, no gaps.

| verdict | n | | confidence | n |
|---|---|---|---|---|
| correct | 71 | | likely | 111 |
| over-clustered | 24 | | unclear | 25 |
| under-clustered | 22 | | certain | 5 |
| not an EM shower | 12 | | | |
| vertex-bad (undecidable) | 10 | | | |
| both | 2 | | | |

18 pi0 pairings stored.  Cost: ~340-475 k tokens per agent per 7-10 events.

## Scope, stated as a number rather than a caveat

`scope_audit.py` over all 136 captured events: the `kine_best >= 20 MeV` rule
skipped **1568 showers holding 4489 of 59695 MeV**, so the scan looked at
**92.5% of the sample's charge**.  The `kine_best` / `kine_charge` column
disagreement that can hide a shower from the rule bit **4 showers in 136
events**.

## Two categories the bucket table cannot show

1. **`correct` carrying real marks — 6 events** (`evt52044`, `evt71872`,
   `evt74326`, `evt168432`, `evt386442`, `evt400504`).  `on_save` stores ONE
   `em.verdict`, for the shower selected at save time, while `marks_by_shower`
   holds many, so an event whose leading shower is fine and whose *second*
   shower is broken files as good.  `rollup.py` lists them.
2. **`start/axis wrong` — 6 events** (`evt167612`, `evt173819`, `evt174771`,
   `evt321767`, `evt347824`, `evt396037`).  `EM_VERDICTS` has no string for it
   and `vertex-bad (undecidable)` means the *neutrino* vertex.  The list is
   append-only and the owner's; agents wrote the literal phrase in the note.

## Tooling changed, and why

- **`em114_categorize.py`** gained `--manifest` and `--use-verdict`, both
  default-OFF.  Proof the default path is untouched: the pr/115 command still
  produces a **byte-identical** bucket TSV matching the committed
  `docs/pr/pr115-handscan-buckets.tsv`.  `--use-verdict` was not optional in
  the end: an agent measured the note regexes against its own ten notes and
  **5 of 10 hit a pattern contradicting the verdict** (`RE_OVER` on the literal
  "over-cluster" while discussing a different shower, `RE_UNDER` on "missing"
  inside "Nothing is missing"), and `RE_GOOD` is `^...$`-anchored so a note
  prefixed `[agent-bulk]` could never match "good".
- **`scanbot_capture2.py`** — a fork (M10; the original is byte-identical at
  `cf63bf67dce3857ca4c6afe4d48e4343`, four agents were driving it).  Two fixes:
  `--vtx-span` plus an automatic close-in `vtxz-*` sweep when the event scale
  exceeds 150 cm (one long muon frames the vertex at +-400 cm; it fired on about
  a third of events), and a **drag-check probe that is not the rotation
  centre**.  The old probe read `seg3_src.data.xs[0][0]`, which on any event
  whose first segment starts at the nu vertex IS the sweep's centre: four
  wave-1 events sat at |seg_u| ~1e-6, two reported False and **two reported
  True only by crossing a 1e-6 absolute threshold on noise**.  The passes were
  the dangerous half.  No wave-1 judgement was made off stale frames -- verified
  on evt172266, where the new probe sits 366.7 px off the centroid and moves
  u 114.98 -> 46.63.

## QC — 8 events, pre-registered

See `qc/QC-RESULT.md`.  Sample drawn by `random.Random(20260828)` and written to
disk **before any bulk verdict was read**.  Raw 4 agree / 3 differ / 1 not
adjudicated; adjudicated, **5 compatible, 2 genuine splits, 0 errors found**.
Both genuine splits (`evt292524`, `evt58755`) already appear *in the agent's own
note* as the named alternative reading.  My pass was the lighter one (1-2 frames
vs 3-9), so this is not an accuracy rate.

**The number worth acting on:** both splits and part of a third event turn on the
same question -- *how far past its own body may a shower reach before the far
`pass4_angle` stubs count as over-clustering?*  Three of eight QC events hinge on
it.  That is one gate decision, not 141 judgements.

## Known bug found in a shared tool, NOT fixed here

`prep_em_scan.py:bee_index()` second pass walks `prefer` **in order** and, for a
round with a zip but no `.url`, overwrites the entry with `("", round)`.  So
listing a local-only round (`em114c`) *before* an uploaded one (`em116confirm`)
clobbers freshly minted links, and the manifest silently comes out with
`bee_url` empty on every row.  Worked around here by leaving `em114c` out of
`--bee-prefer` and restoring the two local-cloud rows by hand.  The fix is to
run the second pass in reverse `prefer` order, or to skip an event that already
carries a URL from a preferred round.

## What this campaign does not settle

- **Whether the calls are right.**  141 events, no ground truth, and the QC
  compared them against a lighter pass by the same kind of scanner.
- **`correct` is still a fast pass** (pr/115 sec 9, Part 1).  71 of them.
- **The verdict-shower rule was inconsistent across waves 1-2.**  Three agents
  invented three rules before correction 17 fixed it as *highest `kine_charge`
  with `pdg == 11`*.  Each note names the shower it chose and any higher-energy
  non-EM object it passed over, so it is recoverable but not uniform.
- **`evt179048`'s verdict is rubric-dependent** and was left as recorded --
  changing a scanner's call is the owner's, not the run lead's
  (`OWNER-DECISIONS.md`).

## Part 2 addendum — checks run after the campaign closed

**The categorizer over all 141 (`--use-verdict`) reconciles, except where it
cannot.**  rc=0, 141 of 141 bucketed.  `good` 71 and `vertex-bad` 10 match
`rollup.py` exactly.  The other three buckets are each larger than the verdict
census — over 28 vs 24, under 25 vs 22, both 3 vs 2, plus 4 "scanned, no
clustering correction" — and the excess is **exactly the 12 `not an EM shower`
events**, which carry no clustering verdict and therefore fall through to the
note regexes:

| bucket the regexes gave them | n | events |
|---|---|---|
| 2 over-clustered | 4 | evt94224, evt172266, evt390634, evt392901 |
| 1 under-clustered | 3 | evt171143, evt173819, evt398191 |
| 1+2 both | 1 | evt179611 |
| scanned, no clustering correction | 4 | evt277298, evt292005, evt293536, evt397401 |

**Say this plainly rather than leave it implied:** 8 of those 12 receive a
*clustering* bucket from a rule this campaign measured as unreliable (5 of 10
notes hit a regex contradicting their own verdict).  8.5% of the sample has no
trustworthy bucket, and what to do with a PID verdict is the owner's taxonomy
call, not something to paper over.  Everything else in the table stands.

**Bee links verified on both sides, not just for HTTP 200.**  The manifest's 28
links use contiguous indices 0-8 (mcp1k) and 0-18 (mcp2k) with no duplicate
`(set, index)` pair; every one matches the `bee_idx` in the round's own
`.index.txt`; and the two sets list exactly 9 and 19 events server-side, equal to
their index rows.  That is the check `bee_index.__doc__` warns about — a link
into the wrong epoch renders perfectly and silently answers a different question.

**Marks, for the shower-expel census.**  The tag holds **49 IN and 83 OUT marks
across 53 events (54 showers)**; 29 events carry at least one OUT.  Three events
(`evt396323`, `evt409140`, `evt499577`) also carry `?` marks, which
`scanbot_record.py` accepts but pr/115's IN/OUT rule does not define.  Listed by
`rollup.py`.

**A line-ending trap worth recording.**  The confirm manifest was hand-edited
through `csv.DictWriter`, whose default `lineterminator` is `\r\n`; the display
does `rstrip("\n")`, so the last column name becomes `scan_note\r` and that key
reads `None`.  Found by `cat -A`, rewritten as LF, and the server restarted on
the fixed file.  Harmless here (the column was empty on all 30 rows) and not
harmless in general.

**Reproducibility record** is in `docs/pr/pr116-bulk/`: the brief with its 21
corrections, the 15 shard files, the 5 ledgers, the 15 decisions JSONs, the QC
pre-registration / my calls / comparison / result, `CONFIRM-SET.json`,
`OWNER-DECISIONS.md`, `FINDING-gate-inversion.md`, the bucket TSV, and the three
analysis scripts.

**Not mine, noticed in passing:** `clus/src/NeutrinoShowerClustering.cxx` in the
toolkit tree gained 364 lines at 16:01 during this run — a `pr/119` byte-neutral
probe under `WCT_SHOWER_EXPEL_DEBUG` whose header reads *"The 29 hand-scan OUT
marks ... sit in 6 events."*  Untouched here.  Whoever is writing it should know
this tag now holds 83 OUT marks in 29 events.

**The review viewer (`em116_confirm_viewer.py`).**  The notes this campaign
wrote run 2000-5200 characters; the production `note_in` is a single-line
`TextInput` 520 px wide, so a note can only be read by arrowing along one line.
The confirm display on 5027 therefore serves a variant with an 18-row,
full-width `TextAreaInput`.  It is a **loader, not a copy**: it reads
`em_display_viewer.py`, applies three asserted substitutions and execs the
result, so the production file stays byte-identical (`a5f0e84c...`, still served
on 5018 and 5019) and any upstream fix reaches the review viewer for free.  A
substitution that stops matching raises at startup rather than serving a viewer
that silently lacks the change.  `max_length=20000` is load-bearing:
`TextAreaInput` defaults to **500 characters**, which would truncate every note
here on save -- silent data loss, not a display problem.  URL path is
`/em116_confirm_viewer`, not `/em_display_viewer`.

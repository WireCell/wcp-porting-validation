# Scan brief — EM shower / pi0 hand scan, bulk run (doc pr/116, 141-event sample)

You are one of five agents extending a verified 5-event pilot to the remaining
136 events of the SBND EM-shower sample.  The pilot is documented at
`sbnd_xin/docs/pr/116_agent-handscan-pilot.md` — **read it first**, it is short
and it is the standard your output is measured against.

Repo root for every relative path below:

    /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

Python is `/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python3.11`
(call it `$PY`; the system python has no playwright).

---

## 0. Hard rules — violating one corrupts other people's work

1. **Your port only.**  You are given one port.  Never touch **5017, 5018,
   5019** or another agent's port.  5018 carries the owner's own live scan.
2. **Never pass `--expect-tag ''`, and never change its value.**  That guard is
   the only thing between you and the owner's label sets.  If it refuses, STOP
   and report — do not work around it.
3. **The scripts are read-only to you.**  `scanbot_capture.py`,
   `scanbot_record.py`, `em_display_viewer.py`, `em3d.py`, `em_geom.py`,
   `em114_categorize.py`.  If one needs a change, report it; do not edit it.
   Five agents editing a shared script concurrently corrupts everyone's run.
4. **Never read `em_labels/emscan-0828-beam141/` or `em_labels/emscan-0827/`.**
   Those are other people's judgements.  Reading them contaminates yours —
   `evt54341` in particular has an owner label and is a deliberate blind
   comparison point.
5. **Only your assigned events.**  Never re-record one already on disk in
   `em_labels/emscan-0828-agent5/` (CLAUDE.md M13: never overwrite a record).
6. **Never invent a verdict for an event you could not see.**  Capture failed,
   nothing passed the scope rule, `kine_div` empty — log the reason, record
   nothing, and report it.  A fabricated label is worse than a gap.
7. Temp files under `/home/xqian/tmp/`, never `/tmp` (M16).
8. Do not commit, push, or run git write commands.

---

## 1. The loop, per event

### A. Capture

    cd <repo root>
    $PY em_display/scanbot_capture.py --event evtNNNNN --port <YOUR PORT> \
        --out /home/xqian/tmp/emscan-bulk/shots/evtNNNNN

Takes 25–90 s and writes PNGs plus `state.json`.  Check `rc=0` **without a
pipe** (`cmd > log 2>&1; echo rc=$?`).

Shot names you will get:

| name | what it is |
|---|---|
| `vtx-az<AZ>-el<EL>.png` | **the six-frame sweep about the neutrino vertex** — the core evidence |
| `ov-iso.png`, `ov-xz.png` | whole event, reco framing |
| `shw<NODE>-az<AZ>-el<EL>.png` | one in-scope shower, auto-framed (~9x zoom) |
| `shw<NODE>-acc.png` | that shower's acceptance plot |
| `pio-az<AZ>-el<EL>.png` | pi0 mode, only if a pi0 group exists |
| `page.png` | the whole page — tables, divs, everything |

### B. Look at the images.  **This is the job.**

Budget **6–9 images per event**, not all of them:

- **always**: 3 of the vertex sweep — `vtx-az-055-el+020`, `vtx-az+065-el+020`,
  `vtx-az-055-el+060` (two azimuths + one elevation: a 3-D shape claim needs a
  second viewpoint, a projection can hide a 90-degree kink).
- **always**: `shw<NODE>-az-055-el+20` for the leading in-scope shower.
- **when the call is under- or over-clustered**: that shower's `-acc.png`, and
  a second azimuth of it.
- **when a pi0 pair exists or you propose one**: one `pio-*.png`.
- add the 4th–6th vertex frame only when the first three leave you unsure.

Reading the visual grammar (`em_display_viewer.py:353-375`), halos widest-first:
cyan 17px = selection, green/red 13px = marks, blue/red 11px = gamma slots,
**soft yellow 9px = reco membership of the selected shower**.  One colour per
shower; **grey = no shower claims this segment**.

### C. Read the numbers — but only to *identify*, never to *judge*

Decide "this shower is missing the blob to its left" **from the picture**; take
the segment ids to mark **from `state.json`**.  Never resolve a mark from pixel
positions.

    $PY -c "
    import json,sys; d=json.load(open('/home/xqian/tmp/emscan-bulk/shots/evtNNNNN/state.json'))
    print('scope', d['scope_rule'])
    for s in d['showers']:
        print(s['node'], 'E=%.1f'%s['kine_best'], 'nseg',s['nseg'],'pdg',s['pdg'],
              'len=%.1f'%s['length'],'pio_id',s['pio_id'])
    "

`state.json` is big — always select fields like this, never `cat` it.
Useful keys: `showers[].candidates` (per-shower `sid/cid/pdg/dist/angle/tier/
metric/owner/site/mark`), `segments`, `shower_table`, `main_vertex`,
`vertices`, `pio.kine_div`, `scope_rule.skipped`, `drag_check`.

**`drag_check.segments_redrew` and `.cloud_redrew` must both be `true`.**  If
either is false the frames are stale and every judgement off them is void —
stop and report.

### D. Decide

Verdict strings, **verbatim** (a typo silently restores as unset):

    correct | over-clustered | under-clustered | both
    vertex-bad (undecidable) | not an EM shower | is an EM shower (reco PID wrong)

Confidence, **verbatim**, and **always set**: `certain` | `likely` | `unclear`.

Rubric, in this order:

- **`vertex-bad (undecidable)`** — the blue vertex star is not where the tracks
  and shower stems converge.  Judge this **first**; if it fires, stop on that
  event and record it as a vertex result, not a clustering failure.
- **`under-clustered`** — charge contiguous with the yellow member set, inside
  the shower's cone (small angle to its axis, distance within or just beyond
  the member spread), that is **grey** or owned by a neighbour.  Corroborate on
  the acceptance plot: a circle just outside a dashed pass-1 step.
- **`over-clustered`** — the yellow set spans two apexes or two directions, or
  swallows a straight minimum-ionising stem.
- **`both`** — both true of one shower.
- **`correct`** — members cover the visible cone, single apex, no contiguous
  unclaimed charge.
- **PID verdicts** — only when unambiguous.  Otherwise your verdict is a
  statement about *clustering* and silent on PID; say so in the note.
- **pi0** — where two EM showers point back to a common origin, fill the gamma
  slots and read both mass conventions out of `pio.kine_div`.  Compare against
  the reco pairing but **never merge `pio_id` with `kine_pio_*`** — they are
  different quantities that can name different pairs.

**`unclear` is an expected outcome, not a failure.**  Two of the pilot's five
were `unclear`, both of them the interesting events.  If you return a whole
shard of `certain`, something is wrong with your calibration, not with the
sample.  Equally, do not reach for `unclear` to avoid deciding: use it when you
looked and the evidence genuinely does not separate two readings, and say in
the note what the two readings are.

**A verdict is recorded for ONE shower** (the app stores `em.verdict` for the
shower selected at save time).  Pick the **leading EM shower**; every other
shower's finding goes in the note and in its marks.

### E. Write the decision and record it immediately

Append to `/home/xqian/tmp/emscan-bulk/decisions/agent<N>-wave<W>.json` (a JSON
list), one object per event:

    {"event": "evtNNNNN",
     "verdict_shower": 63024,
     "verdict": "correct",
     "confidence": "likely",
     "note": "[agent-bulk] shw 63024: good, members cover the cone. shw 62014: ...",
     "event_flags": [],
     "marks": {"62014": {"21003": "out"}},
     "pio": {"g1": 63024, "g2": 62014, "store": true}}

`marks` and `pio` are optional.  `note` **must** start with `[agent-bulk]` —
that is the provenance marker separating this run from the hand-verified pilot.
In the note, name **which showers you did not look at and why** (the scope rule
reaches the top few on a 35-shower event; without that line "over-clustered on
the busy event" reads as an event-level claim when it is a claim about 5
showers).

Then record it **before moving to the next event**, so progress is durable:

    $PY em_display/scanbot_record.py \
        --decisions /home/xqian/tmp/emscan-bulk/decisions/agent<N>-wave<W>.json \
        --port <YOUR PORT> --only evtNNNNN

Confirm `em_labels/emscan-0828-agent5/labels-evtNNNNN.json` now exists.  A
refusal surfaces as a **message on the page, not an exception**, so the script's
own post-save assertion is what you trust — read its output.

### F. Ledger — one line per event, appended immediately

    printf 'evtNNNNN\t<verdict>\t<confidence>\t<n_images_looked_at>\t<ok|skipped:reason>\n' \
        >> /home/xqian/tmp/emscan-bulk/ledger/agent<N>.tsv

Every assigned event gets a line, including skips.  This is how the run is
reconciled; an event with no line counts as never looked at.

---

## 2. When you finish

Report back, compactly:

1. the ledger contents (verdict + confidence per event),
2. the 3–5 events worth the owner's eye, one sentence each on why,
3. anything that failed, and anything you could not decide,
4. any place this brief was wrong or ambiguous.

Do **not** report a number you did not measure.  You have no ground truth; you
are producing judgements for the owner to examine, not an accuracy claim.

---

## 3. CORRECTIONS — read these, they supersede section 1 above

Found by agent 2 on wave 1 (10/10 events, no failures) and verified.

1. **`--out` takes the PARENT directory.**  The capture script appends the event
   name itself, so the section-1 example double-nests into
   `shots/evtNNNNN/evtNNNNN/`.  Use:

       --out /home/xqian/tmp/emscan-bulk/shots

2. **Shot names**: the vertex and shower frames are `…-el+20`, `…-el+60`,
   `…-el-25` (not `el+020`); only the pi0 frames use `el+020`.  Don't guess —
   `ls` the capture directory and read the names you actually got.

3. **`pio-*.png` is written even when the event has no reco pi0 group** — the
   capture auto-slots the top two showers so you have something to look at.
   **Never read that auto pair as the reconstruction's pairing.**  The reco's
   pairing is `showers[].pio_id` in `state.json`.

4. **`showers[].candidates` is a dict of COLUMNS**, not a list of rows:
   `cand["sid"][i]`, `cand["angle"][i]`, … all parallel arrays.  Slicing it
   raises `TypeError: unhashable type: 'slice'`.

5. **Long tracks destroy the vertex sweep — fix it when you see it.**  The
   sweep is framed from r95 of *every* segment point, so one 500 cm muon frames
   the vertex at ±400 cm and the vertex region is a few pixels.  After capture,
   check:

       $PY -c "import json;d=json.load(open('.../state.json'));print(d['event_half_span_cm'])"

   **If it is above ~150 cm, re-capture the sweep close in** with the fork that
   exists for exactly this (original untouched, M10):

       $PY em_display/scanbot_capture2.py --event evtNNNNN --port <YOUR PORT> \
           --vtx-span 60 --out /home/xqian/tmp/emscan-bulk/shots-vtx

   and judge the vertex off `shots-vtx/evtNNNNN/vtx-*.png`.  Verified on
   evt283713: r95 437.7 cm → the vertex neighbourhood is legible at 60 cm.
   Say in the note which framing you judged the vertex from.

6. **A PID verdict is not a clustering bucket.**  If you record
   `not an EM shower` or `is an EM shower (reco PID wrong)`, say explicitly in
   the note that the event carries **no** clustering judgement — otherwise it
   reads downstream as a silent "nothing wrong here".

7. **When two IN/OUT marks differ in strength, say so in the note.**  A mark is
   flat to the categoriser: a tier-1 segment sitting inside the accepted cloud
   and a tier-2 one 100 cm out past the pass-1 box both count as one IN.  Name
   the weak one as the one you would let the owner reverse.

8. **`drag_check.segments_redrew` from `scanbot_capture.py` is UNRELIABLE — do
   not stop on it alone.**  The probe reads `seg3_src.data.xs[0][0]`, and on any
   event whose first drawn segment starts on the main vertex that point *is* the
   sweep's rotation centre, so an azimuth change cannot move it.  Measured over
   wave 1: four events sat at |seg_u| ~1e-6 — two reported `False` (a false
   alarm) and **two reported `True` only by crossing a 1e-6 absolute threshold on
   numerical noise**, which is the dangerous half: a genuinely frozen source
   would report `True` the same way.

   **Use the fork, whose probe is the point farthest from the projected
   centroid:**

       $PY em_display/scanbot_capture2.py --event evtNNNNN --port <YOUR PORT> \
           --out /home/xqian/tmp/emscan-bulk/shots

   It writes `drag_check.probe_point`, `probe_radius_px` and `probe_degenerate`.
   Verified on evt172266, where the old probe said `False`: the new probe sits
   366.7 px off the centroid and moves u 114.98 → 46.63, unambiguously live.
   `--vtx-span` (correction 5) lives in the same fork, so waves 2 and 3 should
   use `scanbot_capture2.py` for everything.

   If `probe_degenerate` is true **and** `cloud_redrew` is false, then stop and
   report.  One of the two moving is sufficient evidence the camera JS ran.

9. **There is no verdict string for "the shower start / axis is wrong."**
   `vertex-bad (undecidable)` is about the **neutrino** vertex and does not
   cover it.  `EM_VERDICTS` is the owner's list and is not ours to extend, so:
   record the closest available string, and write the exact phrase
   **`start/axis wrong`** in the note so these events can be grepped out and
   counted.  Seen already on evt321767 and evt67394.

10. **You may correct your own label from this run.**  Correction 5 in section 0
    ("never re-record") is about other people's records — the owner's tags, and
    another agent's events.  If you decide your own confidence or verdict was
    wrong, re-run `scanbot_record.py` for that event and say so in your report.
    A knowingly-wrong label left on disk is worse than a second save.

11. **A segment can belong to TWO showers while the display names only one.**
    The "in shower" / `owner` column comes from `owner_map()`, which is
    **first owner wins**.  On evt280972 six cluster-72 segments are members of
    both 79136 (625.6 MeV) and 72076 (127.3 MeV) — their charge is inside both
    energies — and the table shows one owner.  Reading that as "the neighbour
    stole it" manufactures an under-clustering finding that is not there; agent
    4 caught itself about to record exactly that.

    **So the section-1 under-clustering rule "owned by a neighbouring shower" is
    unsafe as written.**  Before calling under-clustering on that basis, test it:
    a segment is genuinely a non-member only if excluding it reconciles the
    owner-grouped count with the count in that shower's `cmp_div`.  Charge that
    is **grey (unowned)** carries no such trap and remains the strong case.

12. **Your note will NOT be regex-bucketed — the verdict radio is what counts.**
    Agent 4 measured `em114_categorize.py`'s patterns against its own ten notes:
    **5 of 10 hit a regex that contradicts the verdict** — `RE_OVER` fires on
    the literal "over-cluster" even when the sentence is discussing a *different*
    shower or rejecting the reading; `RE_UNDER` fires on "missing" inside
    "Nothing is missing"; and the one genuine vertex verdict hit no regex at all.
    The bucketing for this scan runs with `--use-verdict`, which reads the radio.
    Set the radio carefully, and write the note for a human, not for a regex.

13. **The image budget is a ceiling, not a quota.**  3–4 frames on a
    seven-segment single-shower event is the right answer.  Record the number
    you actually looked at; never open frames to hit a number.

14. **When a clustering defect is vertex-INDEPENDENT, record the clustering
    verdict, not `vertex-bad`.**  Section 1's rubric says judge the vertex first
    and stop if it fires.  The reason that rule exists is that distances and
    angles measured from a wrong origin are unsafe — so it does **not** apply to
    a defect measured from the **shower start**.  Concretely: if a shower's own
    `cmp_div` says its members span 248 cm at 0–5°, that claim stands whatever
    the vertex is doing.  Record the clustering verdict, describe the bad vertex
    in the note, and let the marks carry the rest.  `em114_categorize.py`'s own
    `classify()` already encodes this precedence: *"A NAMED DIRECTION OUTRANKS a
    vertex complaint."*  Raised by agent 5 on evt179048, which is left as it was
    recorded for the owner to settle.

15. **Write your helper scripts to your OWN directory.**  The session scratchpad
    is **shared by all five agents**.  In wave 2 one agent's `append.py` was
    overwritten by another's copy pointing at a different decisions file, and an
    event was written into the wrong agent's JSON; two read-modify-writes in
    that window could have dropped an entry.  Use

        mkdir -p /home/xqian/tmp/emscan-bulk/work/agent<N>

    for every script and scratch file, and never a bare name in the shared
    scratchpad.  The decisions and ledger paths you were given are already
    per-agent — keep it that way, and append with a read-modify-write that
    re-reads immediately before writing.

16. **The standard membership test is a triple-check.**  Correction 11 is stated
    for a segment owned by two showers; the commoner case is a segment that is
    the **seed of its own sub-20 MeV shower**.  Both are settled the same way:
    a segment is genuinely a non-member only when `nseg`, the owner-grouped
    count, and `cmp_div`'s count all agree after excluding it.

17. **"The leading EM shower" — the definition, because waves 1–2 used two.**
    The verdict is stored for **the highest-`kine_charge` shower with
    `pdg == 11`**, i.e. the first `pdg 11` entry of `scope_rule.in_scope`, which
    is already in that order.  Not the highest-energy object overall: that is
    frequently `pdg` 13 or 2212 (a 746 MeV muon, a 309 MeV proton) and is not an
    EM shower.  Not `kine_best` either — it and `kine_charge` disagree on some
    events and the capture orders by `kine_charge`.  Name the shower you chose
    in the note, and name any higher-energy non-EM object you passed over.

18. **Do NOT compute MeV/cm from `kine_charge / shower_table.length`.**
    `kine_charge` is **blob** charge; `length` is **skeleton** length.  Where the
    skeleton under-covers the blob the ratio overstates density and reads as
    "proton-like".  Measured case: evt287794 gave 12 MeV/cm that way, while the
    point spray shows the skeleton covering ~9 cm of a ~30 cm object — really
    ~3.7 MeV/cm.  An agent nearly recorded a wrong PID flag on it.  If you want
    density, get the extent from the member points in `segments.xs3/ys3/zs3`.

19. **`event_half_span_cm` saturates at 400.0** (`min(400, 1.25*r95)`), so it
    cannot distinguish a 400 cm event from a 1000 cm one — read `r95_cm` when
    you care.

20. **The close-in vertex sweep is now AUTOMATIC.**  `scanbot_capture2.py` adds
    four `vtxz-*` frames at 60 cm half-span whenever the event scale exceeds
    150 cm — it fired on about a third of waves 1–2, so it is not an exception
    worth a manual step.  The `vtx-*` set is unchanged.  Judge the vertex off
    `vtxz-*` when they exist, and say so in the note.  `--vtx-span` still works
    if you want a different radius.

21. **The length reconciliation — how to prove an accept/reject flag.**
    Correction 16's triple-check tells you the counts disagree; it does not tell
    you *which* segment is shared.  The arithmetic does: take
    `shower_table.length` for the shower, subtract the summed candidate `length`
    of the segments whose `owner` is that shower, and match the residual against
    subsets of the non-owned candidates.  Agent 4 identified a hidden trio this
    way on evt499423 (residual 5.5177 cm matched to 0.0001 by 51031 + 47021 +
    38012), proving those were members wearing another owner, not rejects.
    A **zero** residual is equally informative: the owner grouping is complete
    and the accept/reject flags on that event are safe to reason from.
    Script: `/home/xqian/tmp/emscan-bulk/recheck_gate.py <evt> <node> [sids...]`.

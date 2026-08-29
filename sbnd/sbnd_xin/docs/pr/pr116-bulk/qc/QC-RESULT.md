# QC of the bulk scan — 8 pre-registered events

**Repro**

    PY=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python3.11
    cat qc/QC-preregistration.json      # sample + rule, written before any verdict existed
    cat qc/QC-my-calls.json             # my calls, written before reading any label
    cat qc/QC-comparison.json           # the mechanical comparison

Sample drawn by `random.Random(20260828).shuffle` over the sorted 136 bulk events,
first 8, **written to disk before a single bulk verdict was read**, so it cannot
have been picked to flatter the result.

**My pass was lighter than theirs** — 1–2 frames per event against the agents'
3–9 — so it can catch a gross mis-call and cannot adjudicate an under-clustering
finding that lives on the acceptance plot.  One event is recorded as *not
adjudicated* for exactly that reason rather than being forced into a verdict.

## Raw comparison: 4 agree, 3 differ, 1 not adjudicated

| event | my call | agent | outcome |
|---|---|---|---|
| evt282979 | over-clustered | over-clustered | agree |
| evt277298 | not an EM shower | not an EM shower | agree |
| evt409140 | correct | correct | agree |
| evt487853 | correct | correct | agree (I said `unclear`, they said `likely`) |
| evt318769 | over-clustered | **both** | compatible — see below |
| evt292524 | over-clustered | **correct** | genuine split |
| evt58755 | over-clustered | **correct** | genuine split (both `unclear`) |
| evt54453 | — | under-clustered | not adjudicated by my pass |

## Adjudicated: 5 compatible, 2 genuine splits, 0 errors found

**evt318769 is not a conflict.**  `both` = over-clustered + under-clustered, and
the over-clustered half is exactly my reading (two of four members ~100 cm away
at 171–173 deg behind the start, admitted by `in_other_clusters_seg_cone`).  The
agent additionally found an under-clustered half I never evaluated.  My call is a
subset of theirs.

**Neither genuine split is an error by either side — and this is the finding.**
In both cases the agent's own note already contains my reading, in writing, as
the named alternative:

- **evt292524** — the note says: *"8 of the 17 members lie beyond 60 cm, 7 of
  those 8 are pdg 13/211, and two are tier '-' i.e. outside the pass-1 box,
  pulled in by pass4_angle (direct) ... if the owner counts that reach as
  over-reach this event reads over-clustered."*  It also ran the membership
  triple-check (nseg 17 = owner-grouped 17 = cmp_div 17) and established that the
  only unowned segment in the whole event sits at 176.5 deg behind the start.
  Its evidence is stronger than mine; we differ on where to put the line.
- **evt58755** — the note says: *"the member set has a ~26 cm internal gap
  between 22.7 and 48.8 cm, so an over-clustered reading ... is not excluded; I
  did not resolve it."*  Both sides recorded `unclear`.

## What this actually measures

**Not an accuracy rate.**  Eight events against no ground truth, and my pass was
the lighter one.  What it shows is that on this sample there was **no case where
an agent asserted something the images contradict** — the failures available to
find were judgement splits, and both of them were already documented as splits by
the agent that made the call.

**The one number worth acting on:** both genuine splits, and part of the third
event, turn on the *same* question — **how far past its own body may a shower
reach before the far `pass4_angle` stubs count as over-clustering?**  Three of
eight QC events hinge on it.  That is one gate decision, not 141 independent
judgements, and settling it would move the `correct` / `over-clustered` split
across the whole sample.  It is the highest-leverage thing the owner can decide.

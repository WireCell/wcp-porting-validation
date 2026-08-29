# The accept/reject angle inversion — re-derived, and what survives

**Repro**

    PY=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python3.11
    $PY /home/xqian/tmp/emscan-bulk/recheck_gate.py evt91917 87053 66021 69024
    $PY /home/xqian/tmp/emscan-bulk/recheck_gate.py evt98844 70063 63040 66043 60037

## Why it needed re-deriving

Agent 4 raised the pairs in wave 2, then in wave 3 warned they might be
artefacts: `owner_map()` is first-owner-wins, so a genuine member can carry
another shower's owner and read as "rejected", fabricating an inversion.  It
asked for the pairs to be checked before anyone believed them.  They were.

## Result: the owner grouping is complete on both events

| event | shower | `shower_table.length` | owner-grouped sum | residual |
|---|---|---|---|---|
| evt91917 | 87053 | 49.2352 cm, nseg 8 | 49.2352 cm, 8 seg | **0.0000** |
| evt98844 | 70063 | 104.2213 cm, nseg 11 | 104.2213 cm, 11 seg | **-0.0000** |

Nothing is hidden, so the accept/reject flags on these two events are safe to
reason from and the pairs are real.  Agent 4's caution was right as a method and
did not overturn these two.

## What actually survives, and it is narrower than first reported

**evt98844 — a genuine same-tier inversion.**

| segment | dist | angle | tier | outcome |
|---|---|---|---|---|
| 60037 | 44.85 cm | **16.28°** | 1 | accepted, `pass4_angle (direct)` |
| 63040 | 47.08 cm | **6.29°** | 1 | rejected |
| 66043 | 25.42 cm | **2.84°** | 1 | rejected |

Same tier, near-equal distance, and the accepted segment has 2.6x the angle of a
rejected one that is 2.2 cm farther out.

**evt91917 — weaker, and should not be quoted as an equal case.**

| segment | dist | angle | tier | outcome |
|---|---|---|---|---|
| 69024 | 67.89 cm | 20.99° | 1 | accepted, `pass4_angle (direct)` |
| 66021 | 80.60 cm | 12.50° | **2** | rejected |

Different tier and 12.7 cm farther out.  If `tier` is the gate's own
distance-dependent band, a tier-1 acceptance beating a tier-2 rejection may be
by design rather than an inversion.

## The caveat that applies to both

Every rejected segment above has `owner == its own sid`: they are the **seeds of
their own sub-20 MeV showers**, not unowned charge.  So "rejected" here means the
gate preferred to leave the piece as a separate shower.  Whether a tier-1
segment at 2.84 deg / 25.4 cm should seed its own 20 MeV shower instead of
joining the 70063 cone is a question about the gate's design, and it is the
owner's to answer -- not evidence of a bug on its own.

#!/usr/bin/env python3
"""doc 91 -- the sentinel-coverage guard, shared by plan ASSERT 16 and driver
interlock 8 so the two cannot drift apart.

WHAT IT GUARDS.  Run against work-*-prod0901b the pr/127 sentinel suite is not
green.  A sentinel that fails in production but still PASSES in some arm on
disk is a REGRESSION with an adjudication asset, and this round may not delete
that asset.  One that fails everywhere has no asset to protect and must not
block the round -- otherwise a permanently red sentinel freezes every future
retire round.

WHY IT EVALUATES ONE ARM AT A TIME.  pr127_sentinels.py:find_arm() returns the
FIRST arm (sorted glob order) that contains pr_evt<N>/ and evaluates the event
there only.  Passing several arms at once therefore reports the verdict of
whichever arm sorts first, not the best verdict available -- measured: a
combined run over every non-production arm reports all six events FAIL, while
per-arm evaluation finds five of them passing somewhere.  A guard built on the
combined run would have concluded "nothing to protect" and released the arms.

TWO EARLIER READINGS THIS CORRECTS, both from combined runs:
  * 137238 and 292643 are NOT "pre-existing failures everywhere".  Each passes
    in exactly ONE arm -- work-pr130r1-probe98-nuecc48 and
    work-pr134-f086-mcp1k respectively.  Each is a single point of failure.
IDEMPOTENT ACROSS THE DELETION BOUNDARY, checked before the round ran: the
guard was evaluated with all 101 dirs visible and again with only the 52
survivors visible, and both return clean.  This matters because interlock 8
re-runs at delete time and could re-run again after a RETIRE_REPLAN cycle; a
guard that refuses once its own round has succeeded is a tripwire, not a guard.
The only difference between the two runs is 406125 dropping from 4 passing arms
to 3 -- one of its witnesses is in the removal set -- which does not change the
verdict because three survive.

  * 393505 is the only one red everywhere (Enu 559.9 vs a [560, 572] window --
    a 0.1 MeV miss, i.e. drift, not a structural loss).
"""
import os
import re
import subprocess
import sys

SENT = os.path.join('scripts', 'pr127_sentinels.py')


def _verdicts(arm):
    sp = subprocess.run([sys.executable, SENT, '--arms', arm],
                        capture_output=True, text=True, timeout=900)
    if not re.search(r'^\d+ PASS, \d+ FAIL', sp.stdout, re.M):
        raise RuntimeError(f"sentinel suite gave no summary for {arm}")
    return {int(ev): v for v, ev in re.findall(r'^(PASS|FAIL)\s+(\d+)\s', sp.stdout, re.M)}


def sentinel_events():
    evs = sorted({int(m) for m in
                  re.findall(r'^\s*\((\d{4,7}),\s*"', open(SENT).read(), re.M)})
    if not evs:
        raise RuntimeError("parsed 0 sentinel events -- the matcher is broken")
    return evs


def evaluate(all_dirs, keep, prod_suffix='-prod0901b'):
    """-> (events, homeless, regressed{ev: [passing arms]}, red_everywhere, unadjudicable)"""
    events = sentinel_events()
    homeless = [e for e in events
                if not any(os.path.isdir(os.path.join(a, 'pr_evt%d' % e)) for a in keep)]

    prod = sorted(a for a in keep if a.endswith(prod_suffix))
    prod_fail = {e for e, v in _verdicts_multi(prod).items() if v == 'FAIL'} if prod else set()

    regressed, red = {}, []
    for e in sorted(prod_fail):
        holders = [a for a in all_dirs
                   if a not in prod and os.path.isdir(os.path.join(a, 'pr_evt%d' % e))]
        passing = [a for a in holders if _verdicts(a).get(e) == 'PASS']
        (regressed.setdefault(e, passing) if passing else red.append(e))
    unadjudicable = sorted(e for e, arms in regressed.items()
                           if not any(a in keep for a in arms))
    return events, homeless, regressed, red, unadjudicable


def _verdicts_multi(arms):
    if not arms:
        return {}
    sp = subprocess.run([sys.executable, SENT, '--arms'] + arms,
                        capture_output=True, text=True, timeout=1800)
    if not re.search(r'^\d+ PASS, \d+ FAIL', sp.stdout, re.M):
        raise RuntimeError("sentinel suite gave no summary for the production arms")
    return {int(ev): v for v, ev in re.findall(r'^(PASS|FAIL)\s+(\d+)\s', sp.stdout, re.M)}


def report(all_dirs, keep, out=print):
    events, homeless, regressed, red, bad = evaluate(all_dirs, keep)
    if homeless:
        out(f"  !! {len(homeless)} sentinel event(s) would lose every arm: {homeless}")
    else:
        out(f"  OK  {len(events)} sentinel events, every one resolvable in a KEEP arm")
    for e in sorted(regressed):
        kept = [a for a in regressed[e] if a in keep]
        if kept:
            out(f"  OK  evt {e:<7d} REGRESSED -- passes in {len(regressed[e])} arm(s), "
                f"{len(kept)} kept (e.g. {kept[0]})")
        else:
            out(f"  !! evt {e} REGRESSED but every arm that passes it is being "
                f"removed: {regressed[e]}")
    for e in red:
        out(f"    note evt {e:<7d} fails in production AND in every arm on disk -- "
            f"no adjudication asset exists")
    return (not homeless) and (not bad)


if __name__ == '__main__':
    import json
    plan = json.load(open(sys.argv[1]))
    os.chdir(sys.argv[2])
    ok = report(sorted(d for d in os.listdir('.')
                       if d.startswith('work') and os.path.isdir(d)), set(plan['KEEP']))
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""Cut one event's section out of a group run's wire-cell log.

doc 76 round 2.  nusel_extract.py reads the TGM/STM/FC verdicts and the "no STM
fit" reasons from the PR job's log and expects one event per file.  A group job
writes one log for all of its events.

Two rules, in order.  (1) A line that NAMES its event ("evt<ID> ", stamped by
TaggerCheck{TGM,STM,FC} into exactly the lines nusel_extract.py parses, and only
when a group is streaming) is assigned by that name.  (2) Everything else is
assigned BY TIMESTAMP, NOT BY FILE ORDER.  The log carries lines from more than one
spdlog sink and they do NOT arrive in time order: on a measured group run the
TaggerCheckTGM line stamped 16:28:01.926 sat at file line 272, ahead of the
`loading tensor set ident=48367` line stamped 16:28:01.479 at line 318.  A
positional slice therefore hands event N a line belonging to N+-1 -- and since
cluster idents restart every event, that line lands on a COLLIDING ident and
silently rewrites a physics column.  Measured cost of getting this wrong: 140
differing rows in a 100-event nusel table.

Each event owns the half-open time interval from its own "loading tensor set
ident=" line to the next one.  Lines with no parseable timestamp (spdlog lines
torn by an interleaved write -- see the MEM sampler) inherit the assignment of
the line before them, which is the best available guess and matches what a
per-event run would have written.  The configure-phase preamble, before the
first load line, is prepended to every slice.

usage: slice_group_log.py <group.log> <event-id>   > per-event.log
"""
import re
import sys

LOAD = re.compile(r"loading tensor set ident=(\d+)")
TAG = re.compile(r"\bevt(\d+) ")
STAMP = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d+)\]")


def stamp_of(line):
    """Seconds-in-day of a line's leading timestamp, or None."""
    m = STAMP.match(line)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + frac)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, want = sys.argv[1], sys.argv[2]

    lines = open(path, "r", errors="replace").readlines()

    # Pass 1: each event's start time, in the order the events were loaded.
    starts = []          # [(t, ident)]
    first_load = None
    for i, line in enumerate(lines):
        m = LOAD.search(line)
        if not m:
            continue
        if first_load is None:
            first_load = i
        t = stamp_of(line)
        if t is not None:
            starts.append((t, m.group(1)))
    if not starts:
        sys.exit("no 'loading tensor set ident=' line in %s" % path)

    # Pass 2: assign every line to the last event whose start it is at or after.
    head = lines[:first_load]
    out = []
    cur = None           # ident of the line most recently assigned
    idx = 0
    for line in lines[first_load:]:
        tag = TAG.search(line)
        if tag:
            # The line NAMES its event: TaggerCheck{TGM,STM,FC} stamp the id
            # into exactly the lines nusel_extract.py parses, precisely so this
            # does not have to be inferred.  Authoritative, and it deliberately
            # does not move `cur` for the untagged lines around it.
            if tag.group(1) == want:
                out.append(line)
            continue
        t = stamp_of(line)
        if t is not None:
            # Advance through any events that started at or before this line.
            while idx + 1 < len(starts) and starts[idx + 1][0] <= t:
                idx += 1
            while idx > 0 and starts[idx][0] > t:
                idx -= 1
            cur = starts[idx][1]
        # else: torn line, keep `cur`.
        if cur == want:
            out.append(line)

    if not out:
        sys.exit("event %s not found in %s" % (want, path))
    sys.stdout.writelines(head)
    sys.stdout.writelines(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc 82 -- count TORN lines in a WCT log, and show where they tear.

A WCT log file is written by one independently-buffered stdio handle PER
component logger (Aux::Logger::set_name -> Log::logger(name, false) -> a fresh
basic_file_sink, each with its own FILE*).  The buffers drain at different
times, so one component's buffer is flushed into the middle of another
component's line.  The signature is a record header appearing somewhere other
than the start of a line.

Two properties this script reports make the cause unambiguous:

  * every splice offset is an exact multiple of 4096 -- a stdio buffer
    boundary, not a thread race (and the jobs run single-threaded Pgrapher);
  * the spliced-in fragment is typically SECONDS OLDER than the line it cuts,
    which only an independently-buffered second handle can produce.

usage: log_tear_scan.py <log> [<log> ...] [-v]
"""
import os, re, sys

HDR = re.compile(rb"\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]")


def secs(m):
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
            + int(m.group(3)) + int(m.group(4)) / 1000.0)


def scan(path, verbose=False):
    data = open(path, "rb").read()
    off = 0
    nlines = 0
    hits = []
    for line in data.split(b"\n"):
        nlines += 1
        # A header anywhere but position 0 means another record was spliced in.
        m = HDR.search(line, 1)
        if m:
            here = HDR.match(line)
            age = secs(here) - secs(m) if here else None
            hits.append((off + m.start(), age, line[:150]))
        off += len(line) + 1
    aligned = sum(1 for o, _, _ in hits if o % 4096 == 0)
    ages = [a for _, a, _ in hits if a is not None]
    print("%-58s %7d lines %4d torn  %d/%d on a 4096 boundary%s"
          % (os.path.basename(path), nlines, len(hits), aligned, len(hits),
             "  max staleness %.1fs" % max(ages) if ages else ""))
    if verbose:
        for o, a, l in hits:
            print("    @%-10d (mod4096=%-4d) age=%-7s %s"
                  % (o, o % 4096, "%.1fs" % a if a is not None else "?",
                     l.decode("utf8", "replace")))
    return nlines, len(hits)


def main():
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    tot_l = tot_t = 0
    for p in args:
        l, t = scan(p, verbose)
        tot_l += l
        tot_t += t
    if len(args) > 1:
        print("%-58s %7d lines %4d torn" % ("TOTAL", tot_l, tot_t))
    return 1 if tot_t else 0


if __name__ == "__main__":
    sys.exit(main())

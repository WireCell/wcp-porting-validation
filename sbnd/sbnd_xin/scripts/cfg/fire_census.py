#!/usr/bin/env python3
"""fire_census.py -- which shipped knobs actually FIRE, at population scale.

doc 77 sec 11 (round 3).  Companion instrument to scripts/pr127_sentinels.py:

    sentinel     : "does this fix still produce the right answer on ITS event?"
    fire census  : "does this fix still run ANYWHERE?"

doc pr/142 sec 5.3 showed the two questions have different answers -- 406125's
pr/124 prune stopped firing on its own event while firing on 70 others, and
shower_samevtx_track_absorb is ON in production and fires on none of 3067.

Method.  Every shipped fix on this branch logs a line tagged `prNN <name>:`
(the convention since doc pr/55).  We take the INSTRUMENTED set from the
toolkit source -- not from the logs -- and bucket each tag three ways:

    fires N of M   : the tag appears in N event logs
    ZERO           : instrumented, never fires -> the only cell that is a finding
    (uninstrumented knobs cannot appear here at all -- see the caveat below)

CAVEAT, and it is the whole point of building the set from source: a knob with
no tag is INVISIBLE to this instrument.  Silence for such a knob is a gap in
the instrument, never evidence that the knob is dead.  Coverage is reported so
the denominator is never mistaken for "all knobs".

A ZERO row is also not automatically a defect.  It is expected when the knob is
shipped OFF, when the emit site is gated by a diagnostic that is off, or when
the code path's precondition simply did not occur in the sample.  Only a ZERO
row whose knob is ON *and* whose motivating event is present in the sample is a
finding -- that adjudication is per-knob and lives in the doc, not here.

Usage:
    scripts/cfg/fire_census.py work-mcp1k-prod0901b work-mcp2k-prod0901b \
        work-ncpi0-prod0901b work-nuecc48-prod0901b [--tsv out.tsv]
    (doc 89: repointed from work-*-prod0901, which that round released.  The
     arms are positional, so this is a usage example rather than a default --
     nothing was broken, but an example naming a deleted arm is a trap.)
"""
import argparse, collections, os, re, subprocess, sys

TOOLKIT = os.environ.get("WCT_TOOLKIT", "/nfs/data/1/xqian/toolkit-dev/toolkit")
TAG_RE = re.compile(r"\bpr[0-9]{2,3} [a-z0-9_]+:")


def instrumented_tags():
    """The tag literals present in clus/ source -- the instrument's true extent."""
    out = subprocess.run(
        ["git", "-C", TOOLKIT, "grep", "-hoE", r'"[^"]*\bpr[0-9]{2,3} [a-z0-9_]+:',
         "--", "clus/src/*.cxx"],
        capture_output=True, text=True).stdout
    return sorted({m.group(0) for m in TAG_RE.finditer(out)})


def shipped_tla_count():
    """ON / OFF split of the SBND PR job signature, for the coverage line."""
    f = os.path.join(TOOLKIT, "cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet")
    on = off = 0
    try:
        lines = open(f).read().splitlines()
        s = next(i for i, l in enumerate(lines) if l.startswith("function("))
        e = next(i for i, l in enumerate(lines) if i > s and l.startswith(")"))
        for l in lines[s:e]:
            m = re.match(r"^\s+[a-z_][a-z0-9_]*\s*=\s*(.*?),?\s*(//.*)?$", l)
            if not m:
                continue
            v = m.group(1).strip().rstrip(",")
            if v in ("false", "null"):
                off += 1
            else:
                on += 1
    except (OSError, StopIteration):
        return None, None
    return on, off


def scan(roots):
    """tag -> {sample: nevents}, counting each event at most once per tag."""
    per = collections.defaultdict(collections.Counter)
    nevt = collections.Counter()
    for root in roots:
        sample = os.path.basename(root.rstrip("/"))
        for d in sorted(os.listdir(root)):
            if not d.startswith("pr_evt"):
                continue
            log = os.path.join(root, d, "wct_pr_%s.log" % d[3:])
            if not os.path.exists(log):
                continue
            nevt[sample] += 1
            seen = set()
            with open(log, errors="replace") as fh:
                for line in fh:
                    for m in TAG_RE.finditer(line):
                        seen.add(m.group(0))
            for t in seen:
                per[t][sample] += 1
    return per, nevt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    tags = instrumented_tags()
    if not tags:
        sys.exit("no instrumented tags found -- is WCT_TOOLKIT correct?")
    per, nevt = scan(a.roots)
    samples = sorted(nevt)
    total = sum(nevt.values())

    rows = sorted(((sum(per[t].values()), t) for t in tags), reverse=True)
    stray = sorted(set(per) - set(tags))

    w = open(a.tsv, "w") if a.tsv else None
    if w:
        w.write("tag\tevents_fired\tfrac\t" + "\t".join(samples) + "\n")
    print("%-42s %6s %8s   %s" % ("tag", "evts", "frac", "  ".join(samples)))
    for n, t in rows:
        cells = "  ".join("%*d" % (len(s), per[t][s]) for s in samples)
        print("%-42s %6d %7.2f%%   %s" % (t, n, 100.0 * n / total if total else 0, cells))
        if w:
            w.write("%s\t%d\t%.4f\t%s\n" % (t, n, (n / total if total else 0),
                                            "\t".join(str(per[t][s]) for s in samples)))
    if w:
        w.close()

    zero = [t for n, t in rows if n == 0]
    on, off = shipped_tla_count()
    print("\nevents scanned : %d (%s)" % (total, ", ".join("%s %d" % (s, nevt[s]) for s in samples)))
    print("instrumented   : %d tags, %d fire, %d ZERO" % (len(tags), len(tags) - len(zero), len(zero)))
    if on is not None:
        print("coverage       : %d tags against %d ON / %d OFF job TLAs -- knobs with no tag are"
              " INVISIBLE here, not dead" % (len(tags), on, off))
    if zero:
        print("\nZERO fires (adjudicate each: knob OFF? diagnostic? precondition absent?"
              " or ON-but-inert = the finding):")
        for t in zero:
            print("   ", t)
    if stray:
        print("\ntags in logs but NOT in source (tag-extraction gap -- fix the regex):")
        for t in stray:
            print("   ", t)


if __name__ == "__main__":
    main()

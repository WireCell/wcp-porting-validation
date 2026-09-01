#!/usr/bin/env python3
"""tag_coverage.py -- which SHIPPED-ON knobs the fire census can actually see.

doc 77 sec 12 (round 3), item 4.  Companion to scripts/cfg/fire_census.py:

    fire census   : "of the knobs we instrumented, which still fire?"
    tag coverage  : "which shipped-ON knobs did we never instrument at all?"

sec 11.3 could only say "35 tags against 315 ON job TLAs".  That ratio was a
remark, not a measurement -- nothing regenerated it, and nothing said WHICH
knobs were invisible.  This script names them, so the gap is a number a future
round re-derives instead of a sentence it inherits.

Why it matters, from this round's own findings:

  * `shower_samevtx_track_absorb` was ON and inert for three days before the
    fire census noticed -- it was instrumented, so the census could see it.
  * The six kind-3 knobs adjudicated in sec 12.1 have NO tag at all.  Worse,
    `dqdx_fit_keep_all_points`'s nearest emit (TrackFitting.cxx:9074) fires
    when the knob is OFF and goes silent when it is ON, so a log-grep answers
    "did it fire" exactly backwards.  Silence from an untagged knob is a gap
    in the instrument, never evidence about the knob.

Method, and its limits -- stated because an over-claimed coverage number is
worse than none:

  ON set   : job TLAs in the SBND PR job whose literal value is not `false`
             and not `null`.  This is a MECHANICAL split by value (sec 11.1's
             dagger): a numeric sub-parameter of an ON feature counts as ON,
             and a `null` tuning param under an ON parent counts as OFF.  Read
             it as "TLAs carrying a value", never as "live features".
  tag set  : `prNN <name>:` literals in clus/src (same source as fire_census).
  match    : a tag matches a knob when every token of the tag name appears in
             the knob's token set (`samevtx_absorb` matches
             `shower_samevtx_track_absorb`).  This is a HEURISTIC, so the
             reported coverage is a LOWER bound on instrumentation and the
             UNCOVERED list may contain a knob instrumented under an unrelated
             tag name.  Treat an entry as a lead to check, not a verdict.

Usage:
    scripts/cfg/tag_coverage.py [--tsv docs/77-tagcoverage-prod0901.tsv]
"""
import argparse, os, re, subprocess, sys

TOOLKIT = os.environ.get("WCT_TOOLKIT", "/nfs/data/1/xqian/toolkit-dev/toolkit")
JOB = "cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet"
TAG_RE = re.compile(r"\bpr([0-9]{2,3}) ([a-z0-9_]+):")
# tokens that carry no identity -- dropped from both sides before matching
STOP = {"do", "the", "a", "on", "off", "get", "set", "is", "has"}


def instrumented_tags():
    """({tag_name: 'prNN,...'}, n_literals) from clus/ source.

    Keyed by tag NAME because matching is name-based, but the literal count is
    returned separately: two docs can tag the same name (pr138/pr139
    `shower_split`), so 35 literals collapse to 34 names.  Reporting only the
    collapsed number would quietly under-count the instrument.
    """
    out = subprocess.run(
        ["git", "-C", TOOLKIT, "grep", "-hoE", r'"[^"]*\bpr[0-9]{2,3} [a-z0-9_]+:',
         "--", "clus/src/*.cxx"],
        capture_output=True, text=True).stdout
    lits = {(m.group(1), m.group(2)) for m in TAG_RE.finditer(out)}
    names = {}
    for pr, nm in sorted(lits):
        names.setdefault(nm, []).append("pr" + pr)
    return {k: ",".join(v) for k, v in names.items()}, len(lits)


def job_tlas():
    """[(name, literal_value)] from the job's top-level function signature."""
    lines = open(os.path.join(TOOLKIT, JOB)).read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("function("))
    e = next(i for i, l in enumerate(lines) if i > s and l.startswith(")"))
    out = []
    for l in lines[s:e]:
        m = re.match(r"^\s+([a-z_][a-z0-9_]*)\s*=\s*(.*?),?\s*(//.*)?$", l)
        if m:
            out.append((m.group(1), m.group(2).strip().rstrip(",")))
    return out


def toks(name):
    return {t for t in name.split("_") if t and t not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    a = ap.parse_args()

    tags, n_lit = instrumented_tags()
    tlas = job_tlas()
    if not tags or not tlas:
        sys.exit("empty tag or TLA set -- is WCT_TOOLKIT correct? (%s)" % TOOLKIT)

    tag_toks = {t: toks(t) for t in tags}
    rows = []
    for name, val in tlas:
        state = "OFF" if val in ("false", "null") else "ON"
        kt = toks(name)
        hits = sorted(t for t, tt in tag_toks.items() if tt and tt <= kt)
        rows.append((name, state, val, hits))

    on = [r for r in rows if r[1] == "ON"]
    on_cov = [r for r in on if r[3]]
    on_unc = [r for r in on if not r[3]]
    matched_tags = {t for r in rows for t in r[3]}

    w = open(a.tsv, "w") if a.tsv else None
    if w:
        w.write("knob\tstate\tvalue\ttags\n")
        for name, state, val, hits in rows:
            w.write("%s\t%s\t%s\t%s\n" % (name, state, val, ",".join(hits) or "-"))
        w.close()

    print("job            : %s" % JOB)
    print("TLAs           : %d (%d ON / %d OFF by literal value -- sec 11.1 dagger:"
          " a mechanical split, not a feature count)"
          % (len(rows), len(on), len(rows) - len(on)))
    print("tags in source : %d literals / %d distinct names, of which %d match a"
          " knob heuristically" % (n_lit, len(tags), len(matched_tags)))
    print("ON + tagged    : %d" % len(on_cov))
    print("ON + UNTAGGED  : %d  <- invisible to fire_census.py; silence here is"
          " a gap in the instrument, not evidence" % len(on_unc))
    print("coverage       : %.1f%% of ON TLAs (LOWER bound -- name-token"
          " heuristic, see docstring)"
          % (100.0 * len(on_cov) / len(on) if on else 0.0))
    unmatched = sorted(set(tags) - matched_tags)
    if unmatched:
        print("\ntags matching no knob name (tag named for the fix, not the knob --"
              " check before trusting an UNTAGGED row):")
        for t in unmatched:
            print("    %s %s:" % (tags[t], t))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ledger_line.py -- generate 77_knob-ledger.tsv rows for knobs being retired.

doc 77 sec 6: "Every knob that is removed under sec 4's kind-2 rule gets one
line, GENERATED -- never hand-transcribed (Trap 1, sec 3.2) -- from the current
jsonnet plus `git log` on the knob name."  Round 1 produced its 20 rows this
way; this script is that step made re-runnable.

Trap 1 is the reason: doc prose goes stale, the jsonnet does not.  So every
field is derived from the tree rather than typed:

    component            <- which clus/ source file holds the config read
    originating_doc      <- the `doc pr/NN` cited at the read site, else the
                            subject of the introducing commit
    add_commit           <- first `git log -S<knob>` touching clus/ or cfg/
    one_line_why         <- the verdict sentence in the jsonnet TLA comment,
                            which is the record CLAUDE.md treats as current
    was_doctest_pinned   <- does doctest_clus_knob_defaults.cxx CHECK it

MUST be run BEFORE the deletion commit -- it reads the knob out of the working
tree.  `remove_commit` is not knowable yet, so it is emitted as a placeholder
and filled in after the commit exists (pass --remove to do that in one step).

Usage:
    scripts/cfg/ledger_line.py --verdict INERT knob_a knob_b ...
    scripts/cfg/ledger_line.py --verdict INERT --remove <sha> knob_a ...
"""
import argparse, os, re, subprocess

TOOLKIT = os.environ.get("WCT_TOOLKIT", "/nfs/data/1/xqian/toolkit-dev/toolkit")
JOB = "cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet"
DOCTEST = "clus/test/doctest_clus_knob_defaults.cxx"


def sh(*args):
    return subprocess.run(["git", "-C", TOOLKIT] + list(args),
                          capture_output=True, text=True).stdout


def read_site(knob):
    """(component, doc) from the C++ file holding get(config, "<knob>", ...)."""
    out = sh("grep", "-n", '"%s"' % knob, "--", "clus/src")
    for line in out.splitlines():
        if "get(config," in line:
            path = line.split(":", 1)[0]
            comp = os.path.basename(path).replace(".cxx", "")
            m = re.search(r"doc (pr/\d+[^;,)]*)", line)
            return comp, (m.group(1).strip() if m else "")
    return "", ""


def add_commit(knob):
    out = sh("log", "-S", knob, "--oneline", "--reverse", "--", "clus/", "cfg/")
    lines = out.splitlines()
    return (lines[0].split()[0], lines[0].split(None, 1)[1]) if lines else ("", "")


def cfg_verdict(knob):
    """The trailing `//` comment on the knob's TLA signature line."""
    path = os.path.join(TOOLKIT, JOB)
    pat = re.compile(r"^\s+%s\s*=" % re.escape(knob))
    for line in open(path):
        if pat.match(line):
            c = line.split("//", 1)
            if len(c) > 1:
                return " ".join(c[1].split())
            return ""
    return ""


def doctest_pinned(knob):
    return "yes" if sh("grep", "-c", '"%s"' % knob, "--", DOCTEST).strip() not in ("", "0") else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("knobs", nargs="+")
    ap.add_argument("--verdict", required=True,
                    help="verdict_class: round 1's vocabulary is ADVERSE / INERT / "
                         "ROLLED_BACK / SUPERSEDED / WITHDRAWN / REVERTED")
    ap.add_argument("--remove", default="<pending>", help="remove_commit sha")
    a = ap.parse_args()

    for k in a.knobs:
        comp, doc = read_site(k)
        sha, subj = add_commit(k)
        why = cfg_verdict(k)
        if not doc:
            m = re.search(r"doc (pr/\d+)", subj)
            doc = m.group(1) if m else ""
        print("\t".join([k, "clus (%s)" % comp if comp else "clus", doc, sha,
                         a.remove, a.verdict, why, doctest_pinned(k)]))


if __name__ == "__main__":
    main()

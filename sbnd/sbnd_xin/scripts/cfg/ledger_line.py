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
    one_line_why         <- the verdict sentence, resolved in three tiers:
                            the jsonnet TLA comment first (the record CLAUDE.md
                            treats as current), then the comment block above the
                            knob's stanza filtered to sentences naming THIS knob,
                            then the originating doc.  doc 77 sec 11.2's case is
                            why tier 3 exists: a cfg comment that states only
                            "C++ default false" states a default, not a verdict,
                            and the verdict is in the doc under a campaign label
                            (K4, M1) rather than the knob's name.
    was_doctest_pinned   <- does doctest_clus_knob_defaults.cxx CHECK it

MUST be run BEFORE the deletion commit -- it reads the knob out of the working
tree.  `remove_commit` is not knowable yet, so it is emitted as a placeholder
and filled in after the commit exists (pass --remove to do that in one step).

Docs root for tier 3 comes from $WCP_DOCS (default: this repo's
sbnd_xin/docs).

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
            m = re.search(r"doc (pr/\d+[^;,()]*)", line)
            return comp, (m.group(1).strip() if m else "")
    return "", ""


def add_commit(knob):
    out = sh("log", "-S", knob, "--oneline", "--reverse", "--", "clus/", "cfg/")
    lines = out.splitlines()
    return (lines[0].split()[0], lines[0].split(None, 1)[1]) if lines else ("", "")


# Verdict markers, in the vocabulary these records actually use.  Matched
# case-insensitively: the jsonnet shouts ("STAYS OFF"), the docs do not
# ("stay OFF", "do not flip").
VERDICT_MARK = re.compile(
    r"(stays? +(default +)?off|adverse|do not flip|measured dead|measured zero|"
    r"zero yield|failed|regression|inert|no targets|moot|harmful|negative)", re.I)

DOCS = os.environ.get(
    "WCP_DOCS",
    "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/docs")


def _mentions(text, knob, label):
    """Does this sentence/line talk about THIS knob, not a block sibling?

    Three ways a record names a knob, all seen in this tree:
      - in full (`pi0_mu_shower_max_len`);
      - by a family glob (`teb_chain_topology / teb_r3_* STAY OFF`);
      - by its distinguishing suffix (`min_nnf: STAYS OFF`, `ac_veto_radius
        stays OFF`) -- the block covers a family and the prefix is shared with
        a knob that is NOT going, so a prefix match would hand back the wrong
        knob's verdict.  Suffixes must be >= 6 chars to stay distinguishing.
    Plus the campaign label (`K4`, `M1`) for tier 3, where docs rarely spell
    the knob name at all.
    """
    if knob in text:
        return True
    for g in re.findall(r"([a-z0-9_]+)_\*", text):
        if knob.startswith(g + "_"):
            return True
    parts = knob.split("_")
    for i in range(1, len(parts)):
        suf = "_".join(parts[i:])
        if len(suf) >= 6 and re.search(r"(?<![a-z0-9_])%s(?![a-z0-9_])" % re.escape(suf), text):
            return True
    return bool(label) and bool(re.search(r"\b%s\b" % re.escape(label), text))


def _pick(text, knob, label):
    sents = [x for x in re.split(r"(?<=[.;])\s+", text) if x.strip()]
    hit = [x for x in sents if VERDICT_MARK.search(x) and _mentions(x, knob, label)]
    return " ".join(hit)[:400].strip()


def cfg_verdict(knob, label=""):
    """The knob's recorded verdict, DERIVED (doc 77 sec 6, Trap 1) in 3 tiers.

    1. the trailing `//` on the TLA signature line, when it carries a verdict;
    2. else the contiguous `//` block above the knob's stanza -- reduced to the
       sentences that both carry a verdict marker AND name this knob (the
       other_seg and teb blocks each cover two knobs, one kept and one going,
       so an unfiltered block hands you the WRONG knob's verdict);
    3. else the originating doc, which is where a knob whose cfg comment states
       only a C++ default keeps its verdict (sec 11.2's case, e.g. pr/132 K4).
       Docs refer to knobs by their campaign label ("K4", "M1") as often as by
       name, so the label from the C++ read-site comment is searched too.
    """
    path = os.path.join(TOOLKIT, JOB)
    lines = open(path).read().splitlines()
    pat = re.compile(r"^\s+%s\s*=" % re.escape(knob))
    for i, line in enumerate(lines):
        if not pat.match(line):
            continue
        c = line.split("//", 1)
        if len(c) > 1 and VERDICT_MARK.search(c[1]):
            return " ".join(c[1].split())
        # tier 2: nearest comment block above, skipping sibling TLA lines
        j = i - 1
        while j >= 0 and re.match(r"^\s+[a-z_][a-z0-9_]*\s*=", lines[j]):
            j -= 1
        block = []
        while j >= 0 and lines[j].lstrip().startswith("//"):
            block.insert(0, lines[j].lstrip()[2:].strip())
            j -= 1
        got = _pick(" ".join(" ".join(block).split()), knob, label)
        if got:
            return got
        break
    return doc_verdict(knob, label)


def _doc_candidates(path):
    """Verdict-sized pieces of a doc: table rows whole, prose UNWRAPPED.

    These docs are hard-wrapped at ~78 columns, so a line-based grep returns
    half a verdict ("...stay DEFAULT OFF (K13 fired correctly as a defensive").
    Prose paragraphs are re-joined and split on sentence ends instead; table
    rows stay one candidate each, since a row IS the verdict record.
    """
    para = []
    for raw in open(path, errors="replace"):
        line = " ".join(raw.split())
        if line.startswith("|") or line.startswith("#") or not line:
            if para:
                text = " ".join(para); para = []
                for sent in re.split(r"(?<=[.;])\s+", text):
                    if sent.strip():
                        yield sent.strip()
            if line:
                yield line
        else:
            para.append(line)
    if para:
        for sent in re.split(r"(?<=[.;])\s+", " ".join(para)):
            if sent.strip():
                yield sent.strip()


def doc_verdict(knob, label):
    """Tier 3 -- the originating doc's own verdict for this knob.

    Ranked, because a doc's `**Status:**` header matches almost any query: a
    candidate naming the knob outright beats one carrying only its campaign
    label, a recommendation-table row beats prose, and among equals the FULLER
    statement wins (the terse `| K4/K5 | stay OFF |` is true but says nothing a
    reader can act on).  Candidates over 600 chars are status blobs, never
    verdicts, and are dropped before ranking.
    """
    import glob
    best, best_rank = "", None
    for f in sorted(glob.glob(os.path.join(DOCS, "pr", "*.md"))):
        for cand in _doc_candidates(f):
            if len(cand) > 600 or not VERDICT_MARK.search(cand):
                continue
            if not _mentions(cand, knob, label):
                continue
            rank = (0 if knob in cand else 1,
                    0 if cand.startswith("|") else 1,
                    -len(cand))
            if best_rank is None or rank < best_rank:
                best, best_rank = cand, rank
    return re.sub(r"\*+", "", best)[:400].strip()


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
        label = doc.split()[-1] if doc and re.match(r"^[A-Z]\d+$", doc.split()[-1]) else ""
        why = cfg_verdict(k, label)
        if not doc:
            m = re.search(r"doc (pr/\d+)", subj)
            doc = m.group(1) if m else ""
        print("\t".join([k, "clus (%s)" % comp if comp else "clus", doc, sha,
                         a.remove, a.verdict, why, doctest_pinned(k)]))


if __name__ == "__main__":
    main()

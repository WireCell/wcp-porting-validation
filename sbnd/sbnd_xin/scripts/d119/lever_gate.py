#!/usr/bin/env python3
"""doc sbnd_xin/119 rounds 1-2: is a lever's arm the same reconstruction as the control arm?

Fork by duplication (M10) of scripts/d118/hash_gate.py, which stays untouched as doc 118's record.
Three things differ, and they are the whole reason this is a separate script:

  * the arm pair is an ARGUMENT, not a module constant, so one script gates every lever;
  * each lever declares WHAT IT IS ALLOWED TO MOVE, up front, and anything else fails.  The two
    levers have genuinely different allowances and collapsing them into one "expected" list would
    let round 2's display change excuse a round-1 defect;
  * round 1 (tcmalloc) is allowed NOTHING except the stopwatch.  An allocator that changed a
    reconstruction product would be a defect in the toolkit, not a trade to accept.

Lever allowances
  tcm  the allocator only.  Same compiled config, same fit JSON, same everything.  The only
       permitted difference is the dual chain's own wall-clock stopwatch (it differs between two
       runs of ONE configuration -- doc 118 measured that on the ARM_SUFFIX=rep pair).
  pad  proj_pad_wire/proj_pad_time.  Permitted: the T_proj_data tree, the calib dump's `proj`
       block, and the provenance that records WHICH fit JSON was read (the measurement arm names
       an absolute path, the control the in-tree one, so op_config_sha256 moves with it).
       NOT permitted: T_rec_charge, any tagger tree, mabc-pr.zip, the pctree, nusel, or any
       non-`proj` key of the calib dump.  That is the doc-30 bar, restated for SBND.

Usage:
  python3 scripts/d119/lever_gate.py <tcm|pad> [cv nuecc off]
"""
import glob
import hashlib
import json
import os
import re
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(SX, "scripts", "d118"))
import root_gate as RG                                                            # noqa: E402

ARMS = {"cv": "work-r3cv-d119%s", "nuecc": "work-r3nue-d119%s", "off": "work-r3off-d119%s"}
SKIP = re.compile(r"(stdout\.log|wct_pr_evt\d+\.log|\.time\.meta|rc\.txt|\.wct-cfg-evt\d+\.json|\.d119_arm)$")

# The stopwatch: it moves between two runs of one configuration, so it is forgiven for every lever.
# Suffix, not substring, so no other scoreboard field slips through.
TIMING_SUFFIX = "vertex_scoreboard.dual_chain.off_ms"
PROV_BRANCH = {"op_config_sha256", "toolkit_git", "trackfitting_config", "wcp_git"}
PROV_KEY = ("trackfitting_config", "op_config_sha256", "toolkit_git", "wcp_git")
# The two spellings of the fit JSON in the pad arm vs the control.
PATHS = ("119_tf_sbnd_pad.json", "pgrapher/experiment/sbnd/sbnd_track_fitting.json")

# What each lever may move.  Empty tuple = nothing beyond the stopwatch.
ALLOW = {
    # THE NULL PAIR.  Same allowance as tcm, run against a second copy of the control
    # configuration.  Its job is the specificity column: it says that the stopwatch-only
    # differences tcm reports are exactly what two runs of ONE configuration produce, so tcm's
    # PASS is not the gate being blind.  (feedback: a byte gate needs a null pair -- doc 118
    # found three comparator defects this way, and lever_gate.py's allowance logic is a NEW fork
    # that had never been null-tested.)
    "ctl2": {"trees": (), "json_prefix": (), "provenance": False},
    "tcm": {"trees": (), "json_prefix": (), "provenance": False},
    "pad": {"trees": ("T_proj_data",), "json_prefix": ("proj",), "provenance": True},
}


def h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def members(path):
    """{member name: content hash}; a plain file is {'': hash}.  Archives by MEMBER CONTENT --
    a tar.gz/zip embeds mtimes and is never byte-reproducible (CLAUDE.md M2)."""
    if path.endswith((".tar.gz", ".tgz")):
        out = {}
        with tarfile.open(path, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    out[m.name] = h(tf.extractfile(m).read())
        return out
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            return {n: h(zf.read(n)) for n in sorted(zf.namelist())}
    with open(path, "rb") as fh:
        return {"": h(fh.read())}


def jflat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from jflat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from jflat(v, f"{p}[{i}]")
    else:
        yield p, o


def json_note(pa, pb, allow):
    try:
        A = dict(jflat(json.load(open(pa))))
        B = dict(jflat(json.load(open(pb))))
    except (OSError, ValueError) as e:
        return f"unreadable: {e}"
    d = [k for k in sorted(set(A) | set(B)) if A.get(k) != B.get(k)]
    if not d:
        return None

    def ok(k):
        if k.endswith(TIMING_SUFFIX):
            return True
        if allow["provenance"] and any(t in k for t in PROV_KEY):
            return True
        # "proj" as a path COMPONENT, so a key merely containing the letters cannot pass.
        parts = re.split(r"[.\[]", k)
        return any(p in allow["json_prefix"] for p in parts)

    outside = [k for k in d if not ok(k)]
    if outside:
        return f"{len(d)} of {len(A)} keys differ, {len(outside)} NOT allowed: {outside[:6]}"
    return f"EXPLAINED: {len(d)} of {len(A)} keys, all allowed ({d[:3]})"


def root_note(pa, pb, allow):
    with uproot.open(pa) as fa, uproot.open(pb) as fb:
        d = RG.brdiff(fa, fb)
    if not d:
        return None
    outside = {}
    for t, bad in d.items():
        if t in allow["trees"]:
            continue
        keep = [b for b in bad
                if not (t == "Trun" and allow["provenance"] and b in PROV_BRANCH)]
        if keep:
            outside[t] = keep
    if outside:
        return "trees differ: " + "; ".join(f"{t}{b[:6]}" for t, b in sorted(outside.items()))
    return "EXPLAINED: " + ", ".join(f"{t}({len(b)} br)" for t, b in sorted(d.items()))


def explained_path(pa, pb):
    """True when a text product differs only by which fit-JSON path it names."""
    try:
        ta, tb = open(pa, errors="replace").read(), open(pb, errors="replace").read()
    except OSError:
        return False
    for p in PATHS:
        ta, tb = ta.replace(p, "<TFJSON>"), tb.replace(p, "<TFJSON>")
    ta = re.sub(r"/[^\"\s]*<TFJSON>", "<TFJSON>", ta)
    tb = re.sub(r"/[^\"\s]*<TFJSON>", "<TFJSON>", tb)
    return ta == tb


def one_event(da, db, allow):
    diffs, n = [], 0
    for nm in sorted(set(os.listdir(da)) | set(os.listdir(db))):
        if SKIP.search(nm):
            continue
        pa, pb = os.path.join(da, nm), os.path.join(db, nm)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            diffs.append((nm, "present in only one arm"))
            continue
        n += 1
        if nm.endswith(".root"):
            note = root_note(pa, pb, allow)
        elif nm.startswith("calib-pr-evt"):
            note = json_note(pa, pb, allow)
        else:
            try:
                ma, mb = members(pa), members(pb)
            except Exception as e:                   # a corrupt archive is a real finding
                diffs.append((nm, f"unreadable: {e}"))
                continue
            if ma == mb:
                continue
            if allow["provenance"] and nm.endswith((".json", ".tsv")) and explained_path(pa, pb):
                note = "EXPLAINED: differs only by the fit-JSON path string"
            else:
                bad = [k for k in sorted(set(ma) | set(mb)) if ma.get(k) != mb.get(k)]
                note = "members differ: " + ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else "")
        if note:
            diffs.append((nm, note))
    return n, diffs


def evdirs(arm):
    """{(sub-root, pr_evt<id>): dir}.  The key MUST carry the sub-root -- event ids restart in
    every stage-A input file, and keying on the basename alone pairs f000/pr_evt25 with another
    file's event 25.  That defect fabricated a whole arm's difference on doc 118's first run."""
    out = {}
    for d in glob.glob(os.path.join(arm, "f*", "pr_evt*")):
        out[(os.path.basename(os.path.dirname(d)), os.path.basename(d))] = d
    for d in glob.glob(os.path.join(arm, "pr_evt*")):
        out[("", os.path.basename(d))] = d
    return out


def main():
    lever = sys.argv[1]
    samples = sys.argv[2:] or ["cv", "nuecc", "off"]
    allow = ALLOW[lever]
    print(f"# doc sbnd_xin/119 lever gate: '{lever}' arm vs the 'ctl' arm (production as it runs today)")
    print(f"# allowed to move: trees={allow['trees'] or '(none)'} "
          f"calib-json={allow['json_prefix'] or '(none)'} provenance={allow['provenance']}")
    print("# always forgiven: the dual chain's off_ms stopwatch (it moves between two runs of one config)")
    print("# archive members by content (M2); logs, rc.txt, .time.meta, compiled cfg are not products")
    rc = 0
    for s in samples:
        a = os.path.join(SX, ARMS[s] % "ctl")
        b = os.path.join(SX, ARMS[s] % lever)
        A, B = evdirs(a), evdirs(b)
        common = sorted(set(A) & set(B), key=lambda k: (k[0], int(re.sub(r"\D", "", k[1]) or 0)))
        print(f"\n## {s}: ctl {len(A)} events, {lever} {len(B)}, compared {len(common)}")
        if not common:
            print("   NO OVERLAP -- nothing gated")
            rc = 1
            continue
        nfile, unexplained, nexp = 0, [], 0
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda e: (e,) + one_event(A[e], B[e], allow), common))
        for e, n, diffs in res:
            nfile += n
            for nm, note in diffs:
                if note.startswith("EXPLAINED"):
                    nexp += 1
                else:
                    unexplained.append((e, nm, note))
        print(f"   files compared         : {nfile}")
        print(f"   differ, within allowance: {nexp}")
        print(f"   differ, NOT allowed    : {len(unexplained)}")
        for e, nm, note in unexplained[:40]:
            print(f"      {'/'.join(x for x in e if x)}/{nm}: {note}")
        if unexplained:
            rc = 1
    print("\nPASS -- nothing outside this lever's declared allowance moved" if rc == 0 else
          "\nFAIL -- see the differences above")
    return rc


if __name__ == "__main__":
    sys.exit(main())

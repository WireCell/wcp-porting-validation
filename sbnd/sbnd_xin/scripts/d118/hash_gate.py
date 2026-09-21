#!/usr/bin/env python3
"""doc sbnd_xin/118 gate G1: does the FLIPPED PRODUCTION DEFAULT reproduce the doc-116 `tfull` arm?

Two of the three components of this flip never enter the compiled config, so no config-hash
argument can prove it.  This compares the OUTPUTS, per event, between

    work-r3<s>-d118flip     the flipped jsonnet call sites + the in-tree fit JSON, no cell TLAs
    work-r3<s>-d116tfull    the measured arm: the same knobs delivered as TLAs + a runtime JSON

Archive members are compared by CONTENT (abtest/hash_archive.py's rule, CLAUDE.md M2): a tar.gz or
zip is never byte-reproducible, so a raw md5 of the container would report a regression that does
not exist.

The claim being gated is NOT "everything is identical".  The two arms name the TrackFitting JSON
differently -- an absolute developer path in the arm, the in-tree WIRECELL_PATH name after the flip
-- and that string is recorded in the compiled config, in Trun's operating-point hash and in the
doc-109 provenance block.  So: every difference is printed, and the gate passes only if each one
resolves to that path string.  Predicting the list up front and explaining away whatever else turned
up would be the same claim with the burden the wrong way round.

Usage: python3 scripts/d118/hash_gate.py [cv nuecc off]
"""
import hashlib, io, json, os, re, sys, tarfile, zipfile
import uproot
from concurrent.futures import ThreadPoolExecutor

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARMS = {"cv":    ("work-r3cv-d116tfull",  "work-r3cv-d118flip"),
        "nuecc": ("work-r3nue-d116tfull", "work-r3nue-d118flip"),
        "off":   ("work-r3off-d116tfull", "work-r3off-d118flip")}
# the two spellings of the fit JSON: any difference must be explained by one of them
PATHS = ("149_tf_sbnd_kf.json", "pgrapher/experiment/sbnd/sbnd_track_fitting.json")
SKIP = re.compile(r"(stdout\.log|wct_pr_evt\d+\.log|\.time\.meta|rc\.txt|\.wct-cfg-evt\d+\.json)$")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import root_gate as RG                                                            # noqa: E402

# Keys whose difference IS the thing this flip changes about provenance, or a stopwatch.  Anything
# outside this set is a physics difference and fails the gate.
PROV_BRANCH = {"op_config_sha256", "toolkit_git", "trackfitting_config", "wcp_git"}
# The ONLY non-provenance field allowed to differ: the dual chain's own wall-clock stopwatch for
# its second pass, in milliseconds.  It appears both at the top level and per candidate
# (candidates[N].vertex_scoreboard.dual_chain.off_ms), so the match is on the suffix -- and it is a
# suffix, not a substring, so no other scoreboard field can slip through.  It differs between two
# runs of ONE configuration too (scripts/d118/stageB_flip.sh ARM_SUFFIX=rep), which is why it is
# here rather than being argued about.
TIMING_SUFFIX = "vertex_scoreboard.dual_chain.off_ms"


def h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def members(path):
    """{member name: content hash} for an archive; {'': filehash} for a plain file."""
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


def explained(path, a, b):
    """True when this file's difference is the fit-JSON path string and nothing else."""
    if not path.endswith(".json"):
        return False
    try:
        ta, tb = open(a, errors="replace").read(), open(b, errors="replace").read()
    except OSError:
        return False
    for p in PATHS:
        ta, tb = ta.replace(p, "<TFJSON>"), tb.replace(p, "<TFJSON>")
    ta = re.sub(r"/[^\"\s]*<TFJSON>", "<TFJSON>", ta)
    tb = re.sub(r"/[^\"\s]*<TFJSON>", "<TFJSON>", tb)
    return ta == tb


def jflat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from jflat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from jflat(v, f"{p}[{i}]")
    else:
        yield p, o


def json_note(pa, pb):
    """None when the two calib dumps differ only in provenance / timing keys."""
    try:
        A = dict(jflat(json.load(open(pa))))
        B = dict(jflat(json.load(open(pb))))
    except (OSError, ValueError) as e:
        return f"unreadable: {e}"
    d = [k for k in sorted(set(A) | set(B)) if A.get(k) != B.get(k)]
    outside = [k for k in d
               if not k.endswith(TIMING_SUFFIX)
               and not any(t in k for t in ("trackfitting_config", "op_config_sha256",
                                            "toolkit_git", "wcp_git"))]
    if outside:
        return f"{len(d)} of {len(A)} keys differ, {len(outside)} outside provenance/timing: {outside[:6]}"
    return None if not d else f"EXPLAINED: {len(d)} of {len(A)} keys, all provenance/timing ({d[:3]})"


def root_note(pa, pb):
    """None when every tree matches but the Trun provenance branches."""
    with uproot.open(pa) as fa, uproot.open(pb) as fb:
        d = RG.brdiff(fa, fb)
    outside = {t: [b for b in bad if not (t == "Trun" and b in PROV_BRANCH)] for t, bad in d.items()}
    outside = {t: b for t, b in outside.items() if b}
    if outside:
        return "trees differ: " + "; ".join(f"{t}{b[:6]}" for t, b in sorted(outside.items()))
    return None if not d else "EXPLAINED: only Trun provenance (" + ", ".join(sorted(
        set(b for bad in d.values() for b in bad))) + ")"


def one_event(da, db):
    """(n compared, [ (file, note) ... ]) for one pr_evt dir pair."""
    diffs, n = [], 0
    names = sorted(set(os.listdir(da)) | set(os.listdir(db)))
    for nm in names:
        if SKIP.search(nm):
            continue
        pa, pb = os.path.join(da, nm), os.path.join(db, nm)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            diffs.append((nm, "present in only one arm")); continue
        n += 1
        if nm.endswith(".root"):
            note = root_note(pa, pb)
            if note:
                diffs.append((nm, note))
            continue
        if nm.startswith("calib-pr-evt"):
            note = json_note(pa, pb)
            if note:
                diffs.append((nm, note))
            continue
        try:
            ma, mb = members(pa), members(pb)
        except Exception as e:                       # a corrupt archive is a real finding
            diffs.append((nm, f"unreadable: {e}")); continue
        if ma == mb:
            continue
        if explained(pa, pa, pb):
            diffs.append((nm, "EXPLAINED: differs only by the fit-JSON path string")); continue
        bad = [k for k in sorted(set(ma) | set(mb)) if ma.get(k) != mb.get(k)]
        diffs.append((nm, "members differ: " + ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else "")))
    return n, diffs


def evdirs(arm):
    """{(sub-root, pr_evt<id>): dir}.  The KEY MUST carry the sub-root: event ids restart in every
    stage-A input file, so keying on the basename alone silently pairs f000/pr_evt25 with some
    other file's event 25 and reports a whole arm as divergent.  (It did, on the first run.)"""
    import glob
    out = {}
    for d in glob.glob(os.path.join(arm, "f*", "pr_evt*")):
        out[(os.path.basename(os.path.dirname(d)), os.path.basename(d))] = d
    for d in glob.glob(os.path.join(arm, "pr_evt*")):
        out[("", os.path.basename(d))] = d
    return out


def main(samples):
    print("# doc 118 G1 -- flipped production default vs the doc-116 `tfull` arm, per event")
    print("# archive members by content (M2); logs, rc.txt, .time.meta and the compiled cfg are not products")
    rc = 0
    for s in samples:
        ta, fa = (os.path.join(SX, x) for x in ARMS[s])
        A, B = evdirs(ta), evdirs(fa)
        common = sorted(set(A) & set(B), key=lambda k: (k[0], int(re.sub(r"\D", "", k[1]) or 0)))
        print(f"\n## {s}: tfull {len(A)} events, flip {len(B)}, compared {len(common)}")
        if not common:
            print("   NO OVERLAP -- nothing gated"); rc = 1; continue
        nfile, unexplained, explained_n = 0, [], 0
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda e: (e,) + one_event(A[e], B[e]), common))
        for e, n, diffs in res:
            nfile += n
            for nm, note in diffs:
                if note.startswith("EXPLAINED"):
                    explained_n += 1
                else:
                    unexplained.append((e, nm, note))
        print(f"   files compared            : {nfile}")
        print(f"   differ, provenance/timing only: {explained_n}")
        print(f"   differ, NOT explained     : {len(unexplained)}")
        for e, nm, note in unexplained[:40]:
            print(f"      {'/'.join(x for x in e if x)}/{nm}: {note}")
        if unexplained:
            rc = 1
    print("\nPASS -- every difference is provenance (the fit-JSON path and the hashes derived\n       from it) or a stopwatch; no physics product differs" if rc == 0 else
          "\nFAIL -- see the unexplained differences above")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["cv", "nuecc", "off"]))

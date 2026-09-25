#!/usr/bin/env python3
"""doc sbnd_xin/123 sec 17 -- the flip gate: two stage-A roots (group layout, optionally with
per-file f*/ sub-roots) compared member by member, and their stage-B roots (<root>pr) when both
exist.  Member CONTENT only -- tar/zip/npz embed mtimes (CLAUDE.md M2).  tracking-pr.root is NOT
compared (a ROOT file carries a UUID and write timestamps in its header; its physics content is
the calib-pr json + nusel tsv beside it, which are).

usage: flip_gate.py <root_a> <root_b> [--no-pr]
prints one line per product family (same/differ/missing) and the first differing member; exit 1
on any difference or missing product.
"""
import glob, hashlib, os, sys, tarfile, zipfile

GROUP = ["opflash_apa0.tar.gz", "opflash_apa1.tar.gz", "frames-dnn.tar.bz2",
         "icluster-apa0-active.npz", "icluster-apa0-masked.npz",
         "icluster-apa1-active.npz", "icluster-apa1-masked.npz"]
QL = ["pctree-evt%s.tar.gz", "mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip"]
PR = ["pctree-pr-evt%s.tar.gz", "mabc-pr.zip", "calib-pr-evt%s.json", "nusel-evt%s.tsv"]


def members(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".npz"):
        # the imaging npz is a zip too, but wire-cell's writer leaves zipfile.is_zipfile() false on
        # some files (a data-descriptor layout); numpy reads them all, so hash the ARRAYS.
        import numpy as np
        with np.load(path, allow_pickle=True) as z:
            return {k: hashlib.sha256(np.ascontiguousarray(z[k]).tobytes() + str(z[k].shape).encode()
                                      + str(z[k].dtype).encode()).hexdigest() for k in z.files}
    if path.endswith(".json"):
        # the calib-pr dump carries wall-clock timings (vertex_scoreboard.dual_chain.off_ms, the
        # only key that differed between two identical PR runs); hash the content with every
        # *_ms key dropped, so a physics difference is named and a timing is not.
        import json
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if not k.endswith("_ms")}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o
        try:
            doc = json.dumps(strip(json.load(open(path))), sort_keys=True)
        except ValueError:
            doc = open(path, "rb").read()
        return {"<json minus *_ms>": hashlib.sha256(doc.encode() if isinstance(doc, str) else doc).hexdigest()}
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    if path.endswith((".tar.gz", ".tar.bz2")):
        t = tarfile.open(path)
        return {ti.name: hashlib.sha256(t.extractfile(ti).read()).hexdigest()
                for ti in t if ti.isfile()}
    return {"<bytes>": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def label_of(name):
    return (name % "" if "%s" in name else name).split(".")[0].rstrip("-")


def compare(pairs, label):
    same = differ = missing = 0; first = None
    for a, b in pairs:
        ma, mb = members(a), members(b)
        if ma is None or mb is None:
            missing += 1; first = first or ("missing", a if ma is None else b); continue
        if ma == mb:
            same += 1; continue
        differ += 1
        if first is None:
            bad = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
            first = ("differ", "%s :: %s" % (a, bad[0] if bad else "?"))
    print("%-10s same=%d differ=%d missing=%d%s" % (label, same, differ, missing,
          ("  first %s: %s" % first) if first else ""))
    return differ + missing


def groups(root):
    out = sorted(glob.glob(os.path.join(root, "g[0-9]*")))
    out += sorted(glob.glob(os.path.join(root, "f[0-9]*", "g[0-9]*")))
    return [g for g in out if os.path.isdir(g)]


def events(root):
    evs = []
    for g in groups(root):
        if os.path.exists(os.path.join(g, "events.txt")):
            sub = os.path.dirname(g) if os.path.basename(os.path.dirname(g)).startswith("f") else root
            evs += [(sub, e.strip()) for e in open(os.path.join(g, "events.txt")) if e.strip()]
    return evs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    A, B = args[0].rstrip("/"), args[1].rstrip("/")
    bad = 0
    ga, gb = groups(A), groups(B)
    rel = lambda g, r: os.path.relpath(g, r)
    gpairs = [(g, os.path.join(B, rel(g, A))) for g in ga]
    if [rel(g, A) for g in ga] != [rel(g, B) for g in gb]:
        print("group sets differ: %s vs %s" % ([rel(g, A) for g in ga], [rel(g, B) for g in gb])); bad += 1
    for name in GROUP:
        bad += compare([(os.path.join(g, name), os.path.join(h, name)) for g, h in gpairs], name.split(".")[0])
    evs = events(A)
    if [e for _, e in evs] != [e for _, e in events(B)]:
        print("event lists differ"); bad += 1
    for name in QL:
        bad += compare([(os.path.join(s, "ql_evt" + e, name % e if "%s" in name else name),
                         os.path.join(B, rel(s, A), "ql_evt" + e, name % e if "%s" in name else name))
                        for s, e in evs], label_of(name))
    if "--no-pr" not in sys.argv and os.path.isdir(A + "pr") and os.path.isdir(B + "pr"):
        for name in PR:
            bad += compare([(os.path.join(A + "pr", rel(s, A), "pr_evt" + e, name % e if "%s" in name else name),
                             os.path.join(B + "pr", rel(s, A), "pr_evt" + e, name % e if "%s" in name else name))
                            for s, e in evs], "pr:" + label_of(name))
    print("VERDICT: %s (%d events)" % ("IDENTICAL" if bad == 0 else "DIFFER", len(evs)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

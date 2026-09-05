#!/usr/bin/env python3
"""pdvd work/ arm census (doc pdvd/29).

Universe is the ARM-SUFFIX GROUP over work/<run6>_<idx>[_<arm>], NOT work/ itself:
`work*` globs to exactly one directory here, so a planner written for sbnd_xin's
sibling-arm layout would either target nothing or target everything (the 08-31
catch #2 failure mode).  See doc pdvd/29 sec 2.

Emits, per arm: dir count, bytes, newest mtime, inbound symlink count (who
borrows FROM it), outbound substrates (who it borrows), and citations.

Citation rules carry doc 91's three recorded failure modes:
  - the retire machinery itself is NOT a consumer (protected-because-protected),
  - matching is substring-on-word-boundary, not name-exact (name-exact scored
    docs=0 for arms cited as templates),
  - a line that only mentions the tag inside a `default=`/`--src-tag` usage
    docstring is reported separately, not counted as a hard dependency.
"""
import os, re, subprocess, sys, json
from collections import defaultdict

PDVD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.path.join(PDVD, "work")
SELF_DIR = os.path.join(PDVD, "scripts", "retire")

EVT = re.compile(r"^(\d{6})_(\d+)(?:_(.*))?$")

def arm_of(name):
    m = EVT.match(name)
    if not m:
        return None
    return m.group(3) or "(bare)"

def main():
    entries = {}
    for name in os.listdir(WORK):
        p = os.path.join(WORK, name)
        if not os.path.isdir(p) or os.path.islink(p):
            continue
        a = arm_of(name)
        if a is None:
            continue
        entries[name] = a

    # ---- bytes + mtime, one du pass (dedups hardlinks within the invocation)
    du = subprocess.run(["du", "-s", "--block-size=1"] + sorted(entries),
                        cwd=WORK, capture_output=True, text=True).stdout
    size = {}
    for line in du.splitlines():
        n, d = line.split("\t", 1)
        size[d.rstrip("/")] = int(n)

    arms = defaultdict(lambda: {"dirs": 0, "bytes": 0, "mtime": 0,
                                "inbound": 0, "outbound": set(),
                                "consumers": set()})
    for name, a in entries.items():
        r = arms[a]
        r["dirs"] += 1
        r["bytes"] += size.get(name, 0)
        r["mtime"] = max(r["mtime"], os.stat(os.path.join(WORK, name)).st_mtime)

    # ---- symlink graph: resolve each link to the arm dir that OWNS the target
    for name, a in entries.items():
        d = os.path.join(WORK, name)
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                continue
            tgt = os.readlink(fp)
            # the owning dir of the target is the last path component that
            # parses as <run6>_<idx>[_<arm>]
            owner = None
            for part in os.path.normpath(
                    os.path.join(os.path.dirname(fp), tgt)).split(os.sep):
                if EVT.match(part):
                    owner = part
            if owner is None:
                continue
            oa = arm_of(owner)
            if oa is None or oa == a:
                continue
            arms[oa]["inbound"] += 1
            arms[oa]["consumers"].add(a)
            arms[a]["outbound"].add(oa)

    # ---- citations
    roots = [os.path.join(PDVD, "docs"), os.path.join(PDVD, "scripts"),
             os.path.join(PDVD, "stm"), PDVD]
    files = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [x for x in dns if x not in
                      ("work", "pics", ".git", "input_data", "photlib")]
            if os.path.abspath(dp).startswith(SELF_DIR):
                continue
            for fn in fns:
                if not fn.endswith((".md", ".py", ".sh", ".txt", ".tsv", ".json", ".jsonnet")):
                    continue
                fp = os.path.join(dp, fn)
                if fp in seen:
                    continue
                seen.add(fp)
                files.append(fp)
            if root == PDVD:
                dns[:] = []   # PDVD itself: top level only

    blobs = []
    for fp in files:
        try:
            blobs.append((fp, open(fp, encoding="utf-8", errors="replace").read()))
        except OSError:
            pass

    USAGE = re.compile(r"(default[s]?\s*[=:]|src_tag|SRC_TAG|usage:|Usage:)", re.I)
    for a in arms:
        if a == "(bare)":
            continue
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![A-Za-z0-9_])")
        hard, soft = set(), set()
        for fp, blob in blobs:
            for line in blob.splitlines():
                if pat.search(line):
                    (soft if USAGE.search(line) else hard).add(
                        os.path.relpath(fp, PDVD))
        arms[a]["cites"] = sorted(hard)
        arms[a]["cites_usage_only"] = sorted(soft - hard)

    out = {}
    for a, r in sorted(arms.items()):
        out[a] = {"dirs": r["dirs"], "bytes": r["bytes"], "mtime": r["mtime"],
                  "inbound": r["inbound"],
                  "consumers": sorted(r["consumers"]),
                  "outbound": sorted(r["outbound"]),
                  "cites": r.get("cites", []),
                  "cites_usage_only": r.get("cites_usage_only", [])}
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(f"arms={len(out)} dirs={sum(v['dirs'] for v in out.values())} "
          f"bytes={sum(v['bytes'] for v in out.values())/2**30:.2f} GiB "
          f"scanned_files={len(blobs)}")

main()

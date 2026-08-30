#!/usr/bin/env python3
"""doc pr/130 item 1b part 2 -- confirm the label store's absorber attribution
against a LIVE census run, and look for guard decisions that were overruled.

`marks_detail[shw]["marked"][seg]["absorbed_by"]` was written when the scan
was made, against the arms of that day.  Guards have shipped since.  This
re-reads the same 22 scanner-condemned segments out of a census arm run at
today's production point (scripts/pr130_qextra_census.sh) and reports, per
segment:

  * which site actually admitted it now, vs what the label recorded
  * whether any guard DECLINED it earlier in the same event

The second column is the point.  A segment that a guard declined and a later
site absorbed anyway is not an admission-threshold problem -- the right call
was already made and then lost.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/pr130_qextra_census.sh qx1
  ./scripts/pr130_qextra_attrib.py
"""
import collections
import csv
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
EMD = os.path.join(SX, "em_display")

ADJUDICATED = {318769, 415278, 283515, 179369}
ARMS = ["work-pr130-qx1-mcp1k", "work-pr130-qx1-mcp2k"]
TSV = "docs/pr/pr130-141-score-prod.tsv"
LABEL_TAG = "emscan-0828-agent5"
MANIFEST = "em114c-pr130q141-manifest.tsv"
PREPDIR = "emprep-pr130q141"


def load_scorer():
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        spec = importlib.util.spec_from_file_location("s117", "em117_score.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def census(ev):
    for a in ARMS:
        p = os.path.join(SX, a, "pr_evt%d" % ev, "stdout.log")
        if os.path.exists(p):
            return open(p, errors="replace").read().splitlines()
    return None


def main():
    s117 = load_scorer()
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        labs = s117.load_labels(LABEL_TAG)
        man = s117.load_manifest(MANIFEST)
    finally:
        os.chdir(cwd)

    outs = {}
    for ev, rec in labs.items():
        for shw, det in ((rec.get("em") or {}).get("marks_detail") or {}).items():
            outs[(ev, int(shw))] = {int(s): v for s, v in (det.get("marked") or {}).items()
                                    if v.get("kind") == "out"}

    rows = list(csv.DictReader(open(os.path.join(SX, TSV)), delimiter="\t"))
    agree = dis = 0
    laundered = []
    q_laundered = q_total = 0.0
    print("%-8s %-8s %-13s %-22s %s" % ("event", "seg", "label said", "live admit site", "declined earlier by"))
    for r in rows:
        ev = int(r["event"])
        if ev in ADJUDICATED:
            continue
        shw = int(r["shower"])
        o = outs.get((ev, shw), {})
        aff = [s for s in [int(x) for x in (r["extra"] or "").split(",") if x.strip()] if s in o]
        if not aff:
            continue
        log = census(ev)
        if log is None:
            print("  evt %d: NO CENSUS LOG -- run pr130_qextra_census.sh first" % ev)
            continue
        cwd = os.getcwd(); os.chdir(EMD)
        try:
            dump = s117.load_dump(man[ev]["dump"])
            _, seginfo, _ = s117.digest_dump(dump, s117.load_prep(ev, PREPDIR))
        finally:
            os.chdir(cwd)
        for s in aff:
            q = seginfo.get(s, {}).get("charge", 0.0)
            q_total += q
            admits, declines = [], []
            pat = re.compile(r"\bseg=%d\b" % s)
            for ln in log:
                if not pat.search(ln):
                    continue
                if "SHOWER_ABSORB DIRECT" in ln:
                    m = re.search(r"site=(\S+)", ln)
                    if m:
                        admits.append(m.group(1))
                elif "WALK_ADD" in ln:
                    m = re.search(r"site=(\S+)", ln)
                    if m:
                        admits.append("walk:" + m.group(1))
                if "decline" in ln:
                    m = re.search(r"(pr\d+ \w+|\w+_guard)[: ]", ln)
                    declines.append(m.group(1) if m else ln.strip()[:40])
            lab = (o[s].get("absorbed_by") or "own-root").split(" ")[0]
            live = ",".join(sorted(set(admits))) or "(none seen)"
            dec = ",".join(sorted(set(declines))) or "-"
            ok = lab in live or lab == "own-root"
            agree += ok
            dis += (not ok)
            if declines and admits:
                laundered.append((ev, s, dec, live, q))
                q_laundered += q
            print("%-8d %-8d %-13s %-22s %s" % (ev, s, lab[:13], live[:22], dec[:44]))

    print("\nlabel-vs-live attribution: %d agree, %d disagree" % (agree, dis))
    print("\nDECLINED BY A GUARD, THEN ADMITTED ANYWAY: %d segments, %.3e charge"
          " (%.1f%% of the affirmative pool)"
          % (len(laundered), q_laundered, 100 * q_laundered / q_total if q_total else 0))
    for ev, s, dec, live, q in sorted(laundered, key=lambda x: -x[4]):
        print("   evt %-8d seg %-8d %.3e  declined by [%s] -> admitted by [%s]"
              % (ev, s, q, dec, live))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc qlmatch/33 -- the only writer of blind-scan verdicts (fork of d32_record.py; adds the `tie:` answer).  Appends one JSON line to
<scan-root>/wave<k>/verdicts.jsonl after validating the sheet id against that wave's INDEX.md, the pick against the
sheet's candidate letters, and refusing a second verdict for the same id.

    python3 d33_record.py --wave 0 --id s12345 --pick B --conf high --reason "..."
    --pick is a letter, tie:B,C (two or more candidates that cannot be separated), none, or unsure.
"""
import argparse
import datetime
import json
import os
import re
import sys

SCAN_ROOT = os.environ.get("D33_SCAN_ROOT", "/home/xqian/tmp/p33/scan/r2d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--conf", required=True, choices=["high", "med", "low"])
    ap.add_argument("--reason", required=True)
    a = ap.parse_args()
    wd = os.path.join(SCAN_ROOT, f"wave{a.wave}")
    idx = {}
    with open(os.path.join(wd, "INDEX.md")) as fh:
        for ln in fh:
            m = re.match(r"- (s\d+)\s+candidates ([A-H]+)", ln)
            if m:
                idx[m.group(1)] = m.group(2)
    if a.id not in idx:
        sys.exit(f"REFUSED: {a.id} is not in wave {a.wave}")
    pick = a.pick.strip()
    if pick.startswith("tie:"):
        ls = [x.strip() for x in pick[4:].split(",") if x.strip()]
        if len(ls) < 2 or len(set(ls)) != len(ls) or any(len(x) != 1 or x not in idx[a.id] for x in ls):
            sys.exit(f"REFUSED: tie needs >= 2 distinct letters from {list(idx[a.id])}, e.g. tie:B,C")
        pick = "tie:" + ",".join(sorted(ls))
    elif pick not in ("none", "unsure") and (len(pick) != 1 or pick not in idx[a.id]):
        sys.exit(f"REFUSED: pick '{pick}' not one of {list(idx[a.id])} / tie:X,Y / none / unsure")
    out = os.path.join(wd, "verdicts.jsonl")
    if os.path.exists(out):
        with open(out) as fh:
            if any(json.loads(ln)["id"] == a.id for ln in fh if ln.strip()):
                sys.exit(f"REFUSED: {a.id} already has a verdict")
    rec = dict(id=a.id, wave=a.wave, pick=pick, conf=a.conf, reason=a.reason.strip(),
               at=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    done = sum(1 for _ in open(out))
    print(f"recorded {a.id} -> {pick} ({a.conf}); wave {a.wave}: {done}/{len(idx)} done")


if __name__ == "__main__":
    main()

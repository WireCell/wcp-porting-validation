#!/usr/bin/env python3
"""doc pr/127 -- PF-tree history of one event across every arm on disk.

Written to answer "when did 137238's cross-cluster muon disappear from the PF
tree?" (owner report 2026-08-29).  It reads `data/0/0-mc.json` out of each
arm's `pr_evt<ID>/mabc-pr.zip`, flattens the jsTree to a one-line signature,
and prints the arms in mtime order so a structure change shows up as the
line where the signature turns over.  Nothing is written; no arm is touched.

This is the cheap half of the sentinel check proposed in doc pr/127 sec 5:
for a shipped fix whose target event is known, the expected signature is a
constant, and a round can assert it instead of re-discovering the loss in a
hand scan.

Repro:
  ./scripts/pr127_pf_history.py 137238
  ./scripts/pr127_pf_history.py 137238 --expect 'mu-  207 MeV'   # sentinel form
  ./scripts/pr127_pf_history.py 37112 --arms 'work-pr125r1-*'
"""
import argparse
import datetime
import glob
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)


def flatten(node, depth=0, out=None):
    out.append("%s%s" % ("  " * depth, node.get("text", "?")))
    for child in node.get("children", []):
        flatten(child, depth + 1, out)
    return out


def signature(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        mc = json.loads(zf.read("data/0/0-mc.json"))
    out = []
    for root in mc:
        flatten(root, 0, out)
    return " | ".join(x.strip() for x in out)


def enu(arm_dir, event):
    p = os.path.join(arm_dir, "pr_evt%d" % event, "calib-pr-evt%d.json" % event)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["kine"]["kine_reco_Enu"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event", type=int)
    ap.add_argument("--arms", default="work-*", help="arm-root glob (default work-*)")
    ap.add_argument("--expect", help="substring the signature must contain; "
                                     "exit 1 if the NEWEST arm lacks it (sentinel mode)")
    ap.add_argument("--width", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for z in sorted(glob.glob(os.path.join(SX, args.arms, "pr_evt%d" % args.event,
                                           "mabc-pr.zip"))):
        arm = os.path.basename(os.path.dirname(os.path.dirname(z)))
        try:
            sig = signature(z)
        except Exception as exc:
            sig = "ERR %s" % exc
        rows.append((os.path.getmtime(z), arm, sig,
                     enu(os.path.dirname(os.path.dirname(z)), args.event)))
    rows.sort()
    prev = None
    for mtime, arm, sig, e in rows:
        mark = " " if sig == prev else "*"
        prev = sig
        print("%s %s  %-40s Enu=%-8s %s" %
              (mark, datetime.datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M"),
               arm, "%.1f" % e if e is not None else "-", sig[:args.width]))
    print("\n%d arm(s); '*' marks a PF-tree signature change from the arm above" % len(rows))

    if args.expect:
        if not rows:
            print("SENTINEL: no arm found for event %d" % args.event)
            return 1
        newest = rows[-1]
        ok = args.expect in newest[2]
        print("SENTINEL %s: newest arm %s %s '%s'" %
              ("PASS" if ok else "FAIL", newest[1],
               "contains" if ok else "does NOT contain", args.expect))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

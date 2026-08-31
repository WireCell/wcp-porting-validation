#!/usr/bin/env python3
"""doc pr/139 sec 8.4 criterion 4 -- hold the arm to the PRE-REGISTERED prediction.

Reads the arm's own WCT_SHOWER_SPLIT_DEBUG tape and checks, object by object,
that the shipped C++ did what docs/pr/pr140-prereg.tsv said it would.  The
prediction was committed (2e996db6) before this arm was run; this script is the
only thing entitled to say whether it held.

    python3 scripts/pr140_tape_verify.py --arm work-pr140r1-on

Tape grammar (WCT_SHOWER_SPLIT_DEBUG=1):
    cand   ... fired=<kernel decision, written BEFORE the veto> ... b_cm=… veto=…
    shared shower=<node> part=<p> nseg=<n>   -- skip_shared refused this peel
    peel   ...                               -- a peel that actually happened
"""
import argparse, collections, csv, glob, os, re, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(SX, "docs", "pr", "pr140-prereg.tsv")

CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?fired=(\d) .*?b_cm=(-?[\d.]+) veto=(\d)')
SHARED = re.compile(r'SHOWER_SPLIT shared shower=(\d+) part=(\d+)')


def read_tape(arm):
    cand, shared = {}, collections.defaultdict(list)
    logs = sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log'))
    if not logs:
        sys.exit("no tape under %s-*/pr_evt*/stdout.log -- was WCT_SHOWER_SPLIT_DEBUG set?" % arm)
    for lg in logs:
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = CAND.search(line)
            if m:
                cand[(ev, int(m.group(1)))] = dict(fired=int(m.group(2)),
                                                   b=float(m.group(3)),
                                                   veto=int(m.group(4)))
                continue
            m = SHARED.search(line)
            if m:
                shared[(ev, int(m.group(1)))].append(int(m.group(2)))
    return cand, shared, len(logs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="work-pr140r1-on")
    a = ap.parse_args()

    pre = {(int(r["event"]), int(r["node"])): r
           for r in csv.DictReader(open(PREREG), delimiter="\t")}
    cand, shared, nlog = read_tape(a.arm)

    print("=== sec 8.4 criterion 4 -- the tape vs the pre-registered prediction ===")
    print("arm %s   (%d event logs; %d candidates on the tape, %d fired, %d vetoed, "
          "%d shared-refusals)"
          % (a.arm, nlog, len(cand), sum(c["fired"] for c in cand.values()),
             sum(c["veto"] for c in cand.values()), len(shared)))
    print()

    bad, seen = [], 0
    fires, sup_shared, sup_bound = [], [], []
    print("%-8s %-7s %-9s %-9s %-11s %-11s %s"
          % ("event", "node", "verdict", "b_pre", "predicted", "actual", ""))
    for k in sorted(pre):
        r = pre[k]
        c = cand.get(k)
        if c is None:
            print("%-8d %-7d %-9s %-9s %-11s %-11s  NOT ON TAPE"
                  % (k[0], k[1], r["owner_verdict"], r["b_cm"], r["predicted"], "-"))
            bad.append((k, "absent from tape"))
            continue
        seen += 1
        if not c["fired"]:
            act = "no-fire"
        elif k in shared:
            act = "SUPPRESSED"
        elif c["veto"]:
            act = "SUPPRESSED"
        else:
            act = "fires"
        ok = (act == r["predicted"])
        if act == "fires":
            fires.append(k)
        elif act == "SUPPRESSED" and r["confirmed_cut"] == "1":
            (sup_shared if k in shared else sup_bound).append(k)
        flag = "" if ok else "  <-- MISMATCH (b_cxx=%.2f veto=%d shared=%s)" % (
            c["b"], c["veto"], k in shared)
        if not ok:
            bad.append((k, "predicted %s, got %s" % (r["predicted"], act)))
        print("%-8d %-7d %-9s %-9s %-11s %-11s%s"
              % (k[0], k[1], r["owner_verdict"], r["b_cm"], r["predicted"], act, flag))

    conf = sum(1 for r in pre.values() if r["confirmed_cut"] == "1")
    right = sum(1 for k in fires if pre[k]["confirmed_cut"] == "1")
    print()
    print("  labelled objects found on the tape : %d of %d" % (seen, len(pre)))
    print("  fires                              : %d   (predicted 15)" % len(fires))
    print("  trigger efficiency                 : %.3f  (predicted 0.737)" % (right / conf))
    print("  trigger purity                     : %.3f  (predicted 0.933)"
          % (right / len(fires) if fires else float("nan")))
    print("  confirmed cuts suppressed, TOTAL   : %d   (predicted 5)"
          % (len(sup_shared) + len(sup_bound)))
    print("     deferred by skip_shared         : %d %s  (predicted 2: 281485/89095, 350354/18092)"
          % (len(sup_shared), sorted(sup_shared)))
    print("     rejected by the b bound         : %d %s"
          % (len(sup_bound), sorted(sup_bound)))
    print()
    if bad:
        print("CRITERION 4 FAIL -- %d object(s) did not do what was predicted:" % len(bad))
        for k, why in bad:
            print("   %d/%d: %s" % (k[0], k[1], why))
        return 1
    print("CRITERION 4 PASS -- every labelled object did exactly what was pre-registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

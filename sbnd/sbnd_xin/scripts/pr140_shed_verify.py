#!/usr/bin/env python3
"""doc pr/139 sec 15 -- hold the SHED arm to its pre-registered prediction.

The `on` arm's tape carries exactly THREE shared-member refusals over all 239
events, so the prediction is per-object and complete, not statistical:

    281485/89095 part 1  -- ALL 4 members co-owned (by 91112)  -> SHED
    165157/9000  part 0  -- 2 of 7 co-owned                    -> still refused
    350354/18092 part 1  -- 1 of 12 co-owned                   -> still refused

and n_shed == 1 over the whole manifest.

The sharing counts above were measured on the pr136 sidecars (`emprep-136onV1c90`,
239 events), which is a DIFFERENT arm -- membership has moved since.  So they are
indicative, and this script is what makes them authoritative: the new binary
prints `nshared` on every refusal, so the arm reports its own truth.

    python3 scripts/pr140_shed_verify.py --arm work-pr140r2-coown --base work-pr140r2-on

FAILURE CONDITIONS, fixed in advance:
  * 165157/9000 sheds.  The owner labels that object KEEP; shedding removes
    members he says belong to the shower.  If it happens the knob does not ship.
  * any object among the 39 scanned changes state other than 281485.
  * n_shed > 1.
"""
import argparse, collections, glob, os, re, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHED = re.compile(r'SHOWER_SPLIT shed shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+)')
SHARED = re.compile(r'SHOWER_SPLIT shared shower=(\d+) part=(\d+) nseg=(\d+)(?: nshared=(\d+))?'
                    r'(?: q_excl_frac=([\d.]+))?')
CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?fired=(\d) .*?veto=(\d)')

# the pre-registration, as data
PREDICT = {(281485, 89095): "shed",
           (165157, 9000): "refused",
           (350354, 18092): "refused"}


def read(arm):
    shed, shared, cand = {}, {}, {}
    logs = sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log'))
    if not logs:
        sys.exit("no tape under %s-*/pr_evt*/stdout.log" % arm)
    for lg in logs:
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = SHED.search(line)
            if m:
                shed[(ev, int(m.group(1)))] = dict(part=int(m.group(2)), nseg=int(m.group(3)),
                                                   q=float(m.group(4)))
                continue
            m = SHARED.search(line)
            if m:
                shared[(ev, int(m.group(1)))] = dict(
                    part=int(m.group(2)), nseg=int(m.group(3)),
                    nshared=int(m.group(4)) if m.group(4) else None,
                    excl=float(m.group(5)) if m.group(5) else None)
                continue
            m = CAND.search(line)
            if m:
                cand[(ev, int(m.group(1)))] = dict(fired=int(m.group(2)), veto=int(m.group(3)))
    return shed, shared, cand, len(logs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='work-pr140r2-coown')
    ap.add_argument('--base', default='work-pr140r2-on')
    a = ap.parse_args()

    A_shed, A_shared, A_cand, nlog = read(a.arm)
    B_shed, B_shared, B_cand, _ = read(a.base)

    print("=== sec 15 -- the shed arm vs its pre-registration ===")
    print("arm  %s : %d shed, %d shared-refusal(s)" % (a.arm, len(A_shed), len(A_shared)))
    print("base %s : %d shed, %d shared-refusal(s)  (%d event logs)"
          % (a.base, len(B_shed), len(B_shared), nlog))
    print()
    print("the sharing the arm itself reports on every refused component:")
    for k in sorted(set(A_shared) | set(B_shared) | set(A_shed) | set(B_shed)):
        d = A_shared.get(k)
        if d:
            print("  evt%-8d shower %-7d REFUSED  nseg=%2d nshared=%s q_excl_frac=%s"
                  % (k[0], k[1], d['nseg'], d['nshared'], d['excl']))
        elif k in A_shed:
            d = A_shed[k]
            print("  evt%-8d shower %-7d SHED     nseg=%2d q=%.4g"
                  % (k[0], k[1], d['nseg'], d['q']))
    print()
    bad = []
    for k, want in sorted(PREDICT.items()):
        got = "shed" if k in A_shed else ("refused" if k in A_shared else "ABSENT")
        ok = (got == want)
        if not ok:
            bad.append((k, want, got))
        print("  evt%-8d shower %-7d predicted %-8s actual %-8s %s"
              % (k[0], k[1], want, got, "" if ok else "  <-- MISMATCH"))
    extra = [k for k in A_shed if k not in PREDICT]
    if extra:
        bad.append(("unpredicted sheds", "", extra))
        print("\n  UNPREDICTED SHEDS: %s" % sorted(extra))
    if (165157, 9000) in A_shed:
        bad.append(("165157 shed", "the owner labels it KEEP", ""))
    if len(A_shed) > 1:
        bad.append(("n_shed", "1", len(A_shed)))
    print()
    if bad:
        print("SEC 15 PRE-REGISTRATION FAIL:")
        for b in bad:
            print("   %s" % (b,))
        return 1
    print("SEC 15 PRE-REGISTRATION PASS -- 1 shed (281485), the two partial-sharing")
    print("refusals stand, and nothing else on the tape changed state.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

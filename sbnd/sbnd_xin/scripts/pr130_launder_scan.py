#!/usr/bin/env python3
"""doc pr/130 item 1b part 3 -- blast radius of the overruled-guard pattern.

Part 2 found, on the 10 events carrying the affirmative q_extra pool, that two
segments were DECLINED by a guard and then absorbed into the SAME shower by a
later site -- 50.7% of that pool's charge.  This sizes the pattern over all
239 events of both standard manifests, using census arms already on disk.  No
new runs.

Matching rule.  A guard's decline line does not itself name the shower, but it
is emitted inside a candidate block whose preceding SHOWER_ABSORB line carries
`shower_start_seg=`.  So the scan tracks the most recent shower context and
pairs (segment, shower).  A decline for shower A followed by an admit into
shower B is a different phenomenon and is counted separately, not folded in.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/pr130_launder_scan.py                  # default: the pr/130 probe arms
  ./scripts/pr130_launder_scan.py <arm> [<arm>...]
"""
import collections
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
DEFAULT_ARMS = ["work-pr130r1-probe98-*", "work-pr130r1-probe141-*"]

RE_CTX = re.compile(r"shower_start_seg=(\d+)")
RE_SEG = re.compile(r"\bseg=(\d+)\b")
RE_SITE = re.compile(r"site=(\S+)")
RE_GUARD = re.compile(r"(pr\d+ [a-z0-9_]+|[a-z0-9_]*_guard)\s*:")


def scan_event(path):
    """-> (declines, admits): {(seg, shower): guard}, {(seg, shower): site}"""
    declines, admits = {}, {}
    ctx = None
    for ln in open(path, errors="replace"):
        m = RE_CTX.search(ln)
        if m:
            ctx = int(m.group(1))
        seg = RE_SEG.search(ln)
        if not seg:
            continue
        s = int(seg.group(1))
        if "SHOWER_ABSORB DIRECT" in ln or "WALK_ADD" in ln:
            site = RE_SITE.search(ln)
            if site and ctx is not None:
                admits.setdefault((s, ctx), site.group(1))
        elif "decline" in ln and ctx is not None:
            # The census emits both a machine line (`... declined=1`, no guard
            # name) and the guard's own log line.  Keep the named one -- first
            # -wins would leave every pass3_cone_guard row as "?".
            g = RE_GUARD.search(ln)
            key = (s, ctx)
            if g or key not in declines:
                declines[key] = g.group(1) if g else "?"
    return declines, admits


def main():
    pats = sys.argv[1:] or DEFAULT_ARMS
    logs = []
    for p in pats:
        for arm in sorted(glob.glob(os.path.join(SX, p)) or glob.glob(p)):
            logs += sorted(glob.glob(os.path.join(arm, "pr_evt*", "stdout.log")))
    if not logs:
        sys.exit("no census logs matched: %s" % pats)

    same_shower, other_shower = [], []
    by_guard = collections.Counter()
    by_site = collections.Counter()
    nev = 0
    for lg in logs:
        nev += 1
        ev = int(re.search(r"pr_evt(\d+)", lg).group(1))
        dec, adm = scan_event(lg)
        for key, guard in dec.items():
            if key in adm:
                same_shower.append((ev, key[0], key[1], guard, adm[key]))
                by_guard[guard] += 1
                by_site[adm[key]] += 1
            else:
                seg = key[0]
                for (s2, h2), site in adm.items():
                    if s2 == seg:
                        other_shower.append((ev, seg, key[1], h2, guard, site))
                        break

    print("scanned %d event logs from %s" % (nev, ", ".join(pats)))
    print("\nDECLINED THEN ADMITTED INTO THE SAME SHOWER: %d (over %d events)"
          % (len(same_shower), len({e for e, *_ in same_shower})))
    for ev, seg, shw, guard, site in sorted(same_shower):
        print("   evt %-8d seg %-8d shower %-8d  [%s] -> [%s]" % (ev, seg, shw, guard, site))
    print("\n  by declining guard: %s" % dict(by_guard))
    print("  by admitting site : %s" % dict(by_site))
    print("\nDECLINED FOR ONE SHOWER, ADMITTED INTO ANOTHER: %d  (counted separately)"
          % len(other_shower))
    for ev, seg, h1, h2, guard, site in sorted(other_shower)[:20]:
        print("   evt %-8d seg %-8d shower %d -> %d  [%s] -> [%s]" % (ev, seg, h1, h2, guard, site))
    return 0


if __name__ == "__main__":
    sys.exit(main())

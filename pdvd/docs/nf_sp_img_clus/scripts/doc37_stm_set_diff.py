#!/usr/bin/env python3
"""doc pdvd/37 round 3 -- how often does the STM-tagged cluster SET differ
between two PDVD arms, and is that more than the noise floor?

Round 2 (sec 13.3) found the STM-tagged SET moving on 109/120 events while the
COUNT barely moved.  Those arms ran with dl_weights='' , so the repeat control
was bit-identical and 109/120 was entirely the knob.  With the DL/SCN vertex ON
-- which is what PDVD production actually runs since 2026-09-04 -- DL inference
is not bit-stable, so the same statistic is knob PLUS DL, and it means nothing
until it is compared against a REPEAT of one arm against itself.

Usage:
  doc37_stm_set_diff.py <tagA> <tagB> [<tagA'> ...]

Report per pair: events compared, events whose STM-tagged cluster-id SET differs,
mean symmetric-difference size, and the same for TGM.  Run it on (off, on) and
on (off, off-repeat); the second is the floor the first has to clear.
"""
import glob, os, re, sys

PDVD = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RE_STM = re.compile(r'TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)')
RE_TGM = re.compile(r'TaggerCheckTGM: cluster (\d+) . TGM=(true|false)')


def sets_of(d):
    """(STM-tagged ids, TGM-tagged ids) from an event's job log, or None."""
    logs = glob.glob(os.path.join(d, 'wct_pr_*.log'))
    if not logs:
        return None
    stm, tgm = set(), set()
    with open(logs[0], errors='replace') as f:
        for line in f:
            m = RE_STM.search(line)
            if m:
                if m.group(2) == '1':
                    stm.add(int(m.group(1)))
                continue
            m = RE_TGM.search(line)
            if m and m.group(2) == 'true':
                tgm.add(int(m.group(1)))
    return stm, tgm


def compare(ta, tb):
    n = 0
    stm_diff = tgm_diff = 0
    stm_sym = tgm_sym = 0
    for da in sorted(glob.glob(os.path.join(PDVD, 'work', '*_' + ta))):
        ev = os.path.basename(da)[:-(len(ta) + 1)]
        db = os.path.join(PDVD, 'work', ev + '_' + tb)
        a, b = sets_of(da), sets_of(db)
        if a is None or b is None:
            continue
        n += 1
        if a[0] != b[0]:
            stm_diff += 1
        if a[1] != b[1]:
            tgm_diff += 1
        stm_sym += len(a[0] ^ b[0])
        tgm_sym += len(a[1] ^ b[1])
    if not n:
        print('%-12s vs %-12s   no overlapping events' % (ta, tb))
        return
    print('%-12s vs %-12s   n=%3d   STM set differs %3d (%5.1f %%), mean sym-diff %.2f'
          '   |   TGM set differs %3d (%5.1f %%), mean sym-diff %.2f'
          % (ta, tb, n, stm_diff, 100.0 * stm_diff / n, stm_sym / n,
             tgm_diff, 100.0 * tgm_diff / n, tgm_sym / n))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base = sys.argv[1]
    for other in sys.argv[2:]:
        compare(base, other)

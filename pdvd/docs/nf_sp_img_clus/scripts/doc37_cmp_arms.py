#!/usr/bin/env python3
"""doc pdvd/37 R1 -- compare two PDVD PR arms event by event, on two axes.

  mabc-pr.zip      member-content hash (abtest/hash_archive.py).  NEVER the
                   file bytes: zips embed mtimes and always differ (M2).
  calib-pr-*.json  the PrDisplayDump.  This is the axis that matters here --
                   it is the only product carrying steiner[].flag_terminal, so
                   it is where a terminal-selection leak would show.

`vertex_scoreboard.dual_chain.off_ms` is a wall-clock TIMER: two byte-identical
reconstructions differ in it.  It is dropped before the JSON compare.  (On PDVD
since 2026-09-04 the dual chain is off and the whole key is null, so this is a
guard against a future flip rather than a live correction.)

A calib-pr dump ABSENT ON BOTH SIDES is agreement, not a gap: PDVD runs the
per-bundle PR only on STM-tagged bundles (nu_per_bundle_stm_only, doc 25
sec 13.10), so an event with zero STM tags writes no dump at all.  Absent on ONE
side is a real divergence and fails.  The two are counted separately -- collapsing
them would let a change that silently stops producing dumps pass as "missing".

Usage: doc37_cmp_arms.py <tagA> <tagB>
Exit 0 only when every event matched on both axes.
"""
import glob, json, os, subprocess, sys

PDVD = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
HASHER = os.path.join(PDVD, '..', 'abtest', 'hash_archive.py')


def zhash(p):
    out = subprocess.run([sys.executable, HASHER, p], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 and out.stdout.split() else None


def scrub(j):
    vs = j.get('vertex_scoreboard')
    if isinstance(vs, dict):
        dc = vs.get('dual_chain')
        if isinstance(dc, dict):
            dc.pop('off_ms', None)
    return j


def load(p):
    with open(p) as f:
        return scrub(json.load(f))


def main(ta, tb):
    zp = zf = zm = jp = jf = jm = jboth = 0
    for da in sorted(glob.glob(os.path.join(PDVD, 'work', '*_' + ta))):
        ev = os.path.basename(da)[:-(len(ta) + 1)]
        db = os.path.join(PDVD, 'work', ev + '_' + tb)

        za = os.path.join(da, 'mabc-pr.zip')
        zb = os.path.join(db, 'mabc-pr.zip')
        if os.path.exists(za) and os.path.exists(zb):
            ha, hb = zhash(za), zhash(zb)
            if ha and ha == hb:
                zp += 1
            else:
                zf += 1
                print('ZIP  DIFFER %s  %s  %s' % (ev, ha, hb))
        else:
            zm += 1

        ca = glob.glob(os.path.join(da, 'calib-pr-evt*.json'))
        cb = glob.glob(os.path.join(db, 'calib-pr-evt*.json'))
        if ca and cb:
            ja, jb = load(ca[0]), load(cb[0])
            if ja == jb:
                jp += 1
            else:
                jf += 1
                keys = sorted(set(ja) | set(jb))
                bad = [k for k in keys if ja.get(k) != jb.get(k)]
                print('JSON DIFFER %s  sections=%s' % (ev, ','.join(bad)))
        elif not ca and not cb:
            jboth += 1          # no STM tag on either side -- agreement
        else:
            jm += 1
            print('JSON ONE-SIDED %s  A=%d B=%d' % (ev, len(ca), len(cb)))

    print('mabc-pr.zip     identical=%d differ=%d missing=%d' % (zp, zf, zm))
    print('calib-pr.json   identical=%d differ=%d one-sided=%d absent-both=%d'
          % (jp, jf, jm, jboth))
    return 0 if (zf == 0 and jf == 0 and zm == 0 and jm == 0) else 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

#!/usr/bin/env python3
"""doc pdvd/40 round 3 -- member-content hash gate between two PR arms over the
120-event manifest.  Never cmp/md5 on archives (M2): mabc-pr.zip is hashed by
member content (abtest/hash_archive.py semantics), and the calib dump is
compared after stripping the wall-clock *_ms timer fields
(feedback_calib_dump_cmp_timer_field).

Usage: d40r3_hash_gate.py <base_tag> <arm_tag> [events.txt]
"""
import sys, os, glob, json, hashlib, zipfile
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W = os.path.join(PDVD, 'work')
base, arm = sys.argv[1], sys.argv[2]
ev_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(PDVD, 'stm', 'events.txt')
events = ['%06d_%s' % (int(l.split()[0]), l.split()[1]) for l in open(ev_file) if l.strip() and not l.startswith('#')]

def zip_members(p):
    out = {}
    with zipfile.ZipFile(p) as z:
        for n in sorted(z.namelist()):
            if n.endswith('/'): continue
            out[n] = hashlib.sha256(n.encode() + z.read(n)).hexdigest()
    return out

def strip_timers(o):
    if isinstance(o, dict):
        return {k: strip_timers(v) for k, v in o.items() if not k.endswith('_ms')}
    if isinstance(o, list):
        return [strip_timers(v) for v in o]
    return o

def calib_hash(d):
    fs = glob.glob(os.path.join(d, 'calib-pr-evt*.json'))
    if not fs: return 'NONE'
    j = strip_timers(json.load(open(fs[0])))
    return hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()

npass = nfail = nmiss = 0
for e in events:
    db, da = os.path.join(W, e + '_' + base), os.path.join(W, e + '_' + arm)
    zb, za = os.path.join(db, 'mabc-pr.zip'), os.path.join(da, 'mabc-pr.zip')
    if not (os.path.exists(zb) and os.path.exists(za)):
        print('%-12s MISSING' % e); nmiss += 1; continue
    mb, ma = zip_members(zb), zip_members(za)
    cb, ca = calib_hash(db), calib_hash(da)
    if mb == ma and cb == ca:
        npass += 1; continue
    nfail += 1
    diff = [n for n in sorted(set(mb) | set(ma)) if mb.get(n) != ma.get(n)]
    print('%-12s DIFF  zip members differing: %d %s  calib %s' % (e, len(diff), diff[:3], 'same' if cb == ca else 'DIFFERS'))
print('GATE %s vs %s: PASS %d  FAIL %d  MISSING %d  of %d' % (base, arm, npass, nfail, nmiss, len(events)))

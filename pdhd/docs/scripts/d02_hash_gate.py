#!/usr/bin/env python3
"""doc pdhd/02 -- member-content hash gate between two PDHD PR arms over the 30-event
run-029107 manifest.  Forked BY DUPLICATION from pdvd/docs/nf_sp_img_clus/scripts/
d40r3_hash_gate.py (untouched): the work root is pdhd/ (three levels up), the manifest
is every work/029107_<evt>_<base> dir, and the calib dump is reported as NONE in -stm
mode (PrDisplayDump is inert there).  Never cmp/md5 on archives (M2): mabc-pr.zip is
hashed by member content (abtest/hash_archive.py semantics).

Usage: d02_hash_gate.py <base_tag> <arm_tag> [run6=029107]
"""
import sys, os, glob, json, hashlib, zipfile, re
PDHD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(PDHD, 'work')
base, arm = sys.argv[1], sys.argv[2]
run6 = sys.argv[3] if len(sys.argv) > 3 else '029107'
events = sorted({re.match(r'.*/(%s_\d+)_%s$' % (run6, base), d).group(1)
                 for d in glob.glob(os.path.join(W, '%s_*_%s' % (run6, base)))},
                key=lambda e: int(e.split('_')[1]))

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

#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/143 -- pairwise gate comparer for the cross-detector arms.

Member-content hashes for zip/tar.gz (never cmp an archive -- embedded mtimes,
CLAUDE.md M2), timer-stripped JSON for calib dumps (the *_ms scoreboard fields
differ between identical runs), bytes for tsv/txt, and a per-tree content hash
over EVERY tree found in a ROOT file (not a fixed tree list, so a file whose
tree set differs is caught rather than skipped).  A file present on one side
only is a FAIL.

Usage: pr143_pair_compare.py A1 B1 [A2 B2 ...]
"""
import sys, os, re, zipfile, tarfile, hashlib, json
import numpy as np

def zip_h(p):
    with zipfile.ZipFile(p) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist()) if not n.endswith('/')}

def tar_h(p):
    out = {}
    with tarfile.open(p) as t:
        for m in t.getmembers():
            if m.isfile(): out[m.name] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return out

def strip_timers(o):
    if isinstance(o, dict): return {k: strip_timers(v) for k, v in o.items() if not k.endswith('_ms')}
    if isinstance(o, list): return [strip_timers(v) for v in o]
    return o

HRT = '/home/xqian/toolkit-dev/toolkit/qlport/scripts/hash_root_trees.py'

def root_h(p):
    """Per-tree content hashes from the project's own hasher (qlport/scripts/
    hash_root_trees.py), over EVERY tree the file actually holds -- its default
    tree list is the SBND PR triple and would silently skip tracking-stm.root's
    trees.  Unordered by default: T_rec_charge row order is run-dependent."""
    import uproot, subprocess
    with uproot.open(p) as f:
        trees = sorted(set(k.split(';')[0] for k in f.keys()))
    r = subprocess.run([sys.executable, HRT, '--per-tree', '--trees', ','.join(trees), p],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('hash_root_trees failed: %s' % r.stderr.strip()[:200])
    # output: an unindented "<filehash>  <path>" line, then one indented
    # "<hash>  <treename>" line per tree.  Keep only the indented rows: the
    # path token would otherwise make every cross-arm pair "differ".
    out = {}
    for l in r.stdout.splitlines():
        if not l.strip() or not l.startswith(' '):
            continue
        parts = l.split()
        if len(parts) < 2 or not re.fullmatch(r'[0-9a-f]{64}', parts[0]):
            continue                      # a note line, not a per-tree row
        out[' '.join(parts[1:])] = parts[0]
    if not out:
        raise RuntimeError('hash_root_trees produced no per-tree rows for %s' % p)
    return out

fails = 0
def cmp_pair(a, b):
    global fails
    if not (os.path.isfile(a) and os.path.isfile(b)):
        print('MISSING |', a, '|', b); fails += 1; return
    try:
        if a.endswith('.zip'):      ha, hb = zip_h(a), zip_h(b);  note = '%d members' % len(ha)
        elif a.endswith('.tar.gz'): ha, hb = tar_h(a), tar_h(b);  note = '%d members' % len(ha)
        elif a.endswith('.json'):   ha, hb = strip_timers(json.load(open(a))), strip_timers(json.load(open(b))); note = 'timers stripped'
        elif a.endswith('.root'):   ha, hb = root_h(a), root_h(b); note = '%d trees' % len(ha)
        else:                       ha, hb = open(a,'rb').read(), open(b,'rb').read(); note = 'bytes'
    except Exception as e:
        print('ERROR', e, '|', a); fails += 1; return
    same = ha == hb
    print(('SAME' if same else 'DIFF'), note, '|', os.path.relpath(a, '/home/xqian/toolkit-dev/toolkit'),
          '|', os.path.relpath(b, '/home/xqian/toolkit-dev/toolkit'))
    if not same and isinstance(ha, dict):
        for k in sorted(set(ha) | set(hb)):
            if ha.get(k) != hb.get(k): print('     differs:', k)
    if not same: fails += 1

for i in range(1, len(sys.argv), 2):
    cmp_pair(sys.argv[i], sys.argv[i+1])
print('RESULT', 'PASS' if fails == 0 else 'FAIL', 'pairs', (len(sys.argv)-1)//2, 'fails', fails)
sys.exit(1 if fails else 0)

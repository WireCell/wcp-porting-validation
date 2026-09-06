#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/143 -- compare two SBND PR arms produced by run_pr_chain_batch.sh.

Member-content hashes for zip/tar.gz (never cmp on an archive: embedded mtimes,
CLAUDE.md M2), timer-stripped JSON for the calib dump (the
vertex_scoreboard.dual_chain.*_ms fields make identical dumps differ), plain
bytes for the nusel TSV, and hash_root_trees.py --per-tree for tracking-pr.root.
An artefact present on one side only is a FAIL, not a skip.

Usage: compare_arms.py <armA> <armB> [--classes tsv,root,zip,tar,calib] [--jobs N] [--events-file F]
"""
import sys, os, zipfile, tarfile, hashlib, json, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor

HRT = '/home/xqian/toolkit-dev/toolkit/qlport/scripts/hash_root_trees.py'

def zip_h(p):
    with zipfile.ZipFile(p) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist()) if not n.endswith('/')}

def tar_h(p):
    out = {}
    with tarfile.open(p) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return out

def strip_timers(o):
    if isinstance(o, dict):
        return {k: strip_timers(v) for k, v in o.items() if not k.endswith('_ms')}
    if isinstance(o, list):
        return [strip_timers(v) for v in o]
    return o

def root_h(p):
    """Per-tree hashes over EVERY tree the file holds.  Two traps, both hit
    once in this round: hash_root_trees' default tree list is the SBND PR
    triple (it would skip the rest), and its first output line carries the FILE
    PATH -- keeping it makes every cross-arm pair "differ" for free."""
    import uproot, re
    with uproot.open(p) as f:
        trees = sorted(set(k.split(';')[0] for k in f.keys()))
    r = subprocess.run([sys.executable, HRT, '--per-tree', '--trees', ','.join(trees), p],
                       capture_output=True, text=True)
    out = {}
    for l in r.stdout.splitlines():
        if not l.strip() or not l.startswith(' '):
            continue
        parts = l.split()
        if len(parts) < 2 or not re.fullmatch(r'[0-9a-f]{64}', parts[0]):
            continue
        out[' '.join(parts[1:])] = parts[0]
    if not out:
        raise RuntimeError('no per-tree rows for %s' % p)
    return out

def one_event(args):
    a_dir, b_dir, evt, classes = args
    res = {}
    def pair(name, kind):
        pa, pb = os.path.join(a_dir, name), os.path.join(b_dir, name)
        ea, eb = os.path.isfile(pa), os.path.isfile(pb)
        if not ea and not eb:
            return 'ABSENT_BOTH'
        if ea != eb:
            return 'MISSING_' + ('B' if ea else 'A')
        try:
            if kind == 'zip':   same = zip_h(pa) == zip_h(pb)
            elif kind == 'tar': same = tar_h(pa) == tar_h(pb)
            elif kind == 'calib':
                same = strip_timers(json.load(open(pa))) == strip_timers(json.load(open(pb)))
            elif kind == 'root': same = root_h(pa) == root_h(pb)
            else: same = open(pa,'rb').read() == open(pb,'rb').read()
        except Exception as e:
            return 'ERROR:%s' % e
        return 'SAME' if same else 'DIFF'
    todo = [('nusel-evt%s.tsv' % evt, 'tsv'), ('tracking-pr.root', 'root'),
            ('mabc-pr.zip', 'zip'), ('pctree-pr-evt%s.tar.gz' % evt, 'tar'),
            ('calib-pr-evt%s.json' % evt, 'calib')]
    for name, kind in todo:
        if kind in classes:
            res[kind] = pair(name, kind)
    return evt, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('armA'); ap.add_argument('armB')
    ap.add_argument('--classes', default='tsv,root,zip,tar,calib')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--events-file')
    ap.add_argument('--out')
    a = ap.parse_args()
    classes = set(a.classes.split(','))
    ev = lambda root: set(d[len('pr_evt'):] for d in os.listdir(root) if d.startswith('pr_evt'))
    EA, EB = ev(a.armA), ev(a.armB)
    if a.events_file:
        want = set(open(a.events_file).read().split())
        EA &= want; EB &= want
    only_a, only_b = sorted(EA - EB), sorted(EB - EA)
    common = sorted(EA & EB, key=int)
    print('arms: %s (%d events)  vs  %s (%d events); common %d; only-A %d; only-B %d'
          % (a.armA, len(EA), a.armB, len(EB), len(common), len(only_a), len(only_b)))
    if only_a: print('  only in A:', ' '.join(only_a[:20]))
    if only_b: print('  only in B:', ' '.join(only_b[:20]))
    tasks = [(os.path.join(a.armA, 'pr_evt'+e), os.path.join(a.armB, 'pr_evt'+e), e, classes) for e in common]
    tally = {}
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for evt, res in ex.map(one_event, tasks, chunksize=8):
            rows.append((evt, res))
            for k, v in res.items():
                tally.setdefault(k, {}).setdefault(v, []).append(evt)
    print()
    for k in ('tsv', 'root', 'zip', 'tar', 'calib'):
        if k not in tally: continue
        d = tally[k]
        print('%-6s ' % k + '  '.join('%s=%d' % (s, len(v)) for s, v in sorted(d.items())))
        for s, v in sorted(d.items()):
            if s not in ('SAME', 'ABSENT_BOTH'):
                print('       %s: %s%s' % (s, ' '.join(sorted(v, key=int)[:40]), ' ...' if len(v) > 40 else ''))
    # arm-level tables
    for t in ('nusel-table.tsv', 'nusel-events.tsv'):
        pa, pb = os.path.join(a.armA, t), os.path.join(a.armB, t)
        if os.path.isfile(pa) and os.path.isfile(pb):
            same = open(pa,'rb').read() == open(pb,'rb').read()
            print('%-16s %s' % (t, 'SAME' if same else 'DIFF'))
    if a.out:
        with open(a.out, 'w') as f:
            f.write('event\t' + '\t'.join(sorted(classes)) + '\n')
            for evt, res in sorted(rows, key=lambda r: int(r[0])):
                f.write(evt + '\t' + '\t'.join(res.get(k, '-') for k in sorted(classes)) + '\n')
        print('wrote', a.out)

if __name__ == '__main__':
    main()

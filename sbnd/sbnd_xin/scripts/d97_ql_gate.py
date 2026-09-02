#!/usr/bin/env python3
"""doc 97 -- member-content gate between two Q/L (stage-A) roots.

Compares every ql_evt<ID>/ product of an arm against the same product in a
baseline root: the three Bee zips and the pctree tarball.  Never md5/cmp on the
archive bytes -- zip members carry wall-clock timestamps and tar members are
written with mtime=time(0) (CLAUDE.md M2) -- so this hashes
sha256(member_name + payload) over members sorted by name, the same rule as
abtest/hash_archive.py.

  Usage: d97_ql_gate.py <arm-suffix> [baseline-suffix] [sample ...]
      d97_ql_gate.py d97idg grp0825
      d97_ql_gate.py d97on  grp0825 mcp1k mcp2k

Reports per sample: events compared, events identical, and the first few
differing (event, product) pairs.  Exit 0 only if every compared product on
every event matches.  Read-only.
"""
import glob, hashlib, os, sys, tarfile, zipfile
from multiprocessing import Pool

ARM = sys.argv[1] if len(sys.argv) > 1 else 'd97idg'
BASE = sys.argv[2] if len(sys.argv) > 2 else 'grp0825'
SAMPLES = sys.argv[3:] or ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']


def rollup(path):
    """sha256 over (member name + payload) for every regular file member."""
    h = hashlib.sha256()
    n = 0
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith('/'):
                    continue
                h.update(name.encode()); h.update(zf.read(name)); n += 1
    else:
        with tarfile.open(path) as tf:
            infos = {m.name: m for m in tf.getmembers() if m.isfile()}
            for name in sorted(infos):
                h.update(name.encode()); h.update(tf.extractfile(infos[name]).read()); n += 1
    return h.hexdigest(), n


PRODUCTS = ('mabc-all-apa.zip', 'mabc-apa0-face0.zip', 'mabc-apa1-face0.zip',
            'pctree-evt{evt}.tar.gz')


def compare_event(job):
    """(smp, arm_event_dir, base_root) -> (all_ok, n_missing, n_compared, bad_rows)"""
    smp, d, base_root = job
    evt = os.path.basename(d)[len('ql_evt'):]
    ok, nmiss, nprod, rows = True, 0, 0, []
    for pat in PRODUCTS:
        f = pat.format(evt=evt)
        a, b = os.path.join(d, f), os.path.join(base_root, f'ql_evt{evt}', f)
        if not (os.path.exists(a) and os.path.exists(b)):
            nmiss += 1; ok = False
            rows.append((smp, evt, f, 'MISSING'))
            continue
        ha, na = rollup(a)
        hb, nb = rollup(b)
        nprod += 1
        if ha != hb:
            ok = False
            rows.append((smp, evt, f, f'{na} vs {nb} members'))
    return ok, nmiss, nprod, rows

tot_ev = tot_same = tot_miss = 0
bad = []
print(f'{"sample":<10}{"events":>7}{"same":>7}{"diff":>7}{"missing":>9}  '
      f'products compared')
for smp in SAMPLES:
    # the tree convention is work-<sample>-<tag>; this round's identity arms
    # were launched as work-<tag>-<sample>, so accept either order.
    def root_of(tag):
        for cand in (f'work-{smp}-{tag}', f'work-{tag}-{smp}'):
            if os.path.isdir(cand):
                return cand
        return None
    arm_root, base_root = root_of(ARM), root_of(BASE)
    if arm_root is None or base_root is None:
        print(f'{smp:<10}  -- missing root (arm={arm_root} base={base_root})')
        continue
    dirs = sorted(glob.glob(f'{arm_root}/ql_evt*'))
    with Pool(int(os.environ.get('D97_GATE_JOBS', 12))) as pool:
        results = pool.map(compare_event, [(smp, d, base_root) for d in dirs])
    nev = nsame = nmiss = nprod = 0
    for ok, nm, np_, rows in results:
        nev += 1; nsame += ok; nmiss += nm; nprod += np_; bad.extend(rows)
    print(f'{smp:<10}{nev:>7}{nsame:>7}{nev - nsame:>7}{nmiss:>9}  {nprod}')
    tot_ev += nev; tot_same += nsame; tot_miss += nmiss

print()
print(f'TOTAL events {tot_ev}, identical {tot_same}, differing {tot_ev - tot_same}, '
      f'missing products {tot_miss}')
for row in bad[:40]:
    print('  DIFF', *row)
if len(bad) > 40:
    print(f'  ... and {len(bad) - 40} more')
print('VERDICT:', 'PASS' if (tot_ev and tot_same == tot_ev and not bad) else 'FAIL')
sys.exit(0 if (tot_ev and tot_same == tot_ev and not bad) else 1)

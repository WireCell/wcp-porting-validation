#!/usr/bin/env python3
"""doc pdvd/119 sec 8: compare two clustering arms work/<run6>_<idx>_<tagA> vs _<tagB>.
Every archive must be member-content identical, except that with --key K the
mabc-all-apa.zip op json may differ ONLY by key K being added in arm A (then the
added array is checked: row-aligned with op_t, each row parallel to op_cluster_ids,
and the anode histogram is printed).  Without --key the arms must be identical.

Usage: d119s_arm_check.py [--key op_cluster_anodes] <run6> <tagA> <tagB> <idx> [idx ...]
"""
import collections, glob, json, os, sys, tarfile, zipfile

KDIR = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(KDIR)
args = sys.argv[1:]
key = None
if args[0] == '--key':
    key, args = args[1], args[2:]
run6, ta, tb = args[:3]
idxs = args[3:]


def members(p):
    out = {}
    if p.endswith('.zip'):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            out[n] = z.read(n)
    else:
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if m.isfile():
                    out[m.name] = t.extractfile(m).read()
    return out


ok_all = True
narch = 0
for idx in idxs:
    dA, dB = f'{PDVD}/work/{run6}_{idx}_{ta}', f'{PDVD}/work/{run6}_{idx}_{tb}'
    arch = lambda d: sorted(os.path.basename(p) for p in glob.glob(f'{d}/*.zip') + glob.glob(f'{d}/*.tar.gz')
                           if not os.path.islink(p))
    la, lb = arch(dA), arch(dB)
    ok = la == lb and len(la) > 0
    notes = [] if ok else [f'archive lists differ {sorted(set(la) ^ set(lb))[:4]}']
    for f in la:
        if f not in lb:
            continue
        narch += 1
        ma, mb = members(f'{dA}/{f}'), members(f'{dB}/{f}')
        diff = sorted(n for n in set(ma) | set(mb) if ma.get(n) != mb.get(n))
        if not diff:
            continue
        if key and f == 'mabc-all-apa.zip' and all(n.endswith('-op.json') for n in diff):
            for n in diff:
                ja, jb = json.loads(ma[n]), json.loads(mb[n])
                arr = ja.pop(key, None)
                if ja != jb or arr is None or key in jb:
                    ok = False; notes.append(f'{n}: differs beyond {key}'); continue
                shape = len(arr) == len(ja['op_t']) and all(
                    len(a) == len(c) for a, c in zip(arr, ja['op_cluster_ids']))
                h = collections.Counter(a for row in arr for a in row)
                ok &= shape
                notes.append(f'{n}: only {key} added; shape {"OK" if shape else "BAD"}; anodes {dict(sorted(h.items()))}')
        else:
            ok = False
            notes.append(f'{f}: members differ {diff[:4]}')
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {run6} idx {idx}: {len(la)} archives; " + '; '.join(notes))
print('OVERALL', 'PASS' if ok_all else 'FAIL', f'({narch} archives)')
sys.exit(0 if ok_all else 1)

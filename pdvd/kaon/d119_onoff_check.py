#!/usr/bin/env python3
"""doc pdvd/119: the label changes ONLY the Bee op file.  For each event, compare
the label-ON arm (work/<run6>_<idx>_<on>) with the label-OFF twin (same libs,
same inputs): every archive must be member-content identical except
mabc-all-apa.zip, whose only differing member may be the *-op.json, and that
op json must be identical once its op_beam key is removed.  Also prints the
labelled rows.

Usage: d119_onoff_check.py <run6> <on_tag> <off_tag> <idx> [idx ...]
"""
import glob, hashlib, json, os, sys, zipfile, tarfile

KDIR = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(KDIR)
run6, on, off = sys.argv[1:4]
idxs = sys.argv[4:]


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
for idx in idxs:
    dA = f'{PDVD}/work/{run6}_{idx}_{on}'
    dB = f'{PDVD}/work/{run6}_{idx}_{off}'
    arch = lambda d: sorted(os.path.basename(p) for p in glob.glob(f'{d}/*.zip') + glob.glob(f'{d}/*.tar.gz')
                           if not os.path.islink(p))
    la, lb = arch(dA), arch(dB)
    ok = la == lb and len(la) > 0
    notes = []
    for f in la:
        if f not in lb:
            continue
        ma, mb = members(f'{dA}/{f}'), members(f'{dB}/{f}')
        diff = sorted(n for n in set(ma) | set(mb) if ma.get(n) != mb.get(n))
        if not diff:
            continue
        if f == 'mabc-all-apa.zip' and all(n.endswith('-op.json') for n in diff):
            for n in diff:
                ja, jb = json.loads(ma[n]), json.loads(mb[n])
                lab = ja.pop('op_beam', None)
                if ja != jb or lab is None or 'op_beam' in jb:
                    ok = False
                    notes.append(f'{n}: differs beyond op_beam')
                else:
                    rows = [(i, round(ja['op_t'][i], 3), round(ja['op_peTotal'][i]), ja['op_cluster_ids'][i])
                            for i, b in enumerate(lab) if b]
                    notes.append(f'{n}: only op_beam added; labelled rows {rows}')
        else:
            ok = False
            notes.append(f'{f}: members differ {diff[:4]}')
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {run6} idx {idx}: {len(la)} archives; " + '; '.join(notes))
print('OVERALL', 'PASS' if ok_all else 'FAIL')
sys.exit(0 if ok_all else 1)

#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 1 -- the STM-dump neutrality gate's archive half.

arm_identity.py reports 'members differ' when an archive gains members.  save_stm_fit adds a Bee 'stm_fit'
layer to mabc-pr.zip and stm_fit/stm_pass/stm_eval cluster PCs to the pctree ON PURPOSE.  This checks that the
difference is ADDITIVE ONLY: every member of arm A's archive is present in arm B's with identical content, and
lists the members B adds.  The pctree is compared BY DATAPATH, never by member name: member names are tensor
indices and renumber when tensors are added (feedback: pctree_dump_offline_traps).  For each metadata tensor
the key is its datapath; its content is the metadata JSON (a pcdataset's 'arrays' / a pcnamedset's 'items'
dictionary restricted to the datapaths A has, so an ADDED point cloud does not count as a change) plus the
sha256 of its companion _array.npy.  Exit 0 iff nothing of A is missing or different in B.
Usage: members_added.py <armA> <armB>
"""
import glob, hashlib, json, os, sys
sys.path.insert(0, '/home/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive  # noqa: E402

def by_datapath(path):
    """datapath -> (metadata dict, sha256 of the companion array or None)."""
    ms = dict(hash_archive.members(path)); out = {}
    for n, payload in ms.items():
        if not n.endswith('_metadata.json'):
            continue
        md = json.loads(payload); arr = ms.get(n.replace('_metadata.json', '_array.npy'))
        out[md.get('datapath', n)] = (md, hashlib.sha256(arr).hexdigest() if arr is not None else None)   # no datapath: keep the member name
    return out

def same(a, b):
    """A's entry equals B's, with B's dataset/namedset dictionaries restricted to A's keys."""
    ma, sa = a; mb, sb = b
    if sa != sb:
        return False
    mb2 = dict(mb)
    for dk in ('arrays', 'items'):
        if dk in ma and dk in mb:
            mb2[dk] = {k: v for k, v in mb[dk].items() if k in ma[dk]}
    return ma == mb2

A, B = sys.argv[1:3]
bad = 0; added = {}
for da in sorted(glob.glob(f'{A}/pr_evt*')):
    ev = os.path.basename(da); db = f'{B}/{ev}'
    for arch in ('mabc-pr.zip', f'pctree-pr-{ev.replace("pr_evt", "evt")}.tar.gz'):
        pa, pb = f'{da}/{arch}', f'{db}/{arch}'
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'{ev} {arch}: missing on one side'); bad += 1; continue
        if 'pctree' in arch:
            ma, mb = by_datapath(pa), by_datapath(pb)
            miss = [n for n in ma if n not in mb]
            diff = [n for n in ma if n in mb and not same(ma[n], mb[n])]
        else:
            ma = {n: hashlib.sha256(p).hexdigest() for n, p in hash_archive.members(pa)}
            mb = {n: hashlib.sha256(p).hexdigest() for n, p in hash_archive.members(pb)}
            miss = [n for n in ma if n not in mb]; diff = [n for n in ma if n in mb and ma[n] != mb[n]]
        add = [n for n in mb if n not in ma]
        if miss or diff:
            print(f'{ev} {arch}: MISSING in B {miss}  DIFFER {diff}'); bad += 1
        for n in add:
            added.setdefault(arch.split('-')[0], set()).add(n.split('/')[-1] if 'pctree' in arch else n)   # pctree: the datapath leaf
        print(f'{ev} {arch}: shared {len(ma) - len(miss)} identical, B adds {len(add)}')
for k, v in added.items():
    print(f'# added member names in {k}: {sorted(v)[:12]}{" ..." if len(v) > 12 else ""}')
print('PASS additive-only' if bad == 0 else f'FAIL {bad}')
sys.exit(1 if bad else 0)

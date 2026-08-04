"""Same-binary-repeat instability census on T_tagger.

Usage: python3 ttag_cmp5.py <armA> <armB>

For each event present in both arms, compares every T_tagger branch.  A branch
that differs is classified:

  reorder  -- the two flattened value lists are permutations of each other
              (same multiset), i.e. only the ORDER of a per-candidate vector
              moved.  This is the pointer-ordered-loop signature.
  value    -- the multisets themselves differ.  Either the candidate SET moved
              or a number genuinely changed.

Prints every differing branch, not a top-N.
"""
import uproot, os, sys, numpy as np

A_, B_ = sys.argv[1], sys.argv[2]
evts = sorted(int(d.split('pr_evt')[1]) for d in os.listdir(A_) if d.startswith('pr_evt'))


def flat(x):
    a = np.asarray(x, dtype=object).ravel()
    out = []
    for v in a:
        va = np.asarray(v)
        out.extend(va.ravel().tolist() if va.size else [])
    return out


nev = 0
anydiff = []
cnt = {}          # branch -> events differing
kind = {}         # branch -> {'reorder': n, 'value': n}
for e in evts:
    fa = f'{A_}/pr_evt{e}/tracking-pr.root'
    fb = f'{B_}/pr_evt{e}/tracking-pr.root'
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    A = uproot.open(fa)
    B = uproot.open(fb)
    k = [x for x in A.keys() if 'T_tagger' in x]
    if not k:
        continue
    ta = A[k[0]].arrays(library='np')
    tb = B[k[0]].arrays(library='np')
    nev += 1
    d = []
    for br in ta:
        fa_, fb_ = flat(ta[br]), flat(tb[br])
        if fa_ == fb_:
            continue
        d.append(br)
        cnt[br] = cnt.get(br, 0) + 1
        kk = kind.setdefault(br, {'reorder': 0, 'value': 0})
        if sorted(fa_) == sorted(fb_):
            kk['reorder'] += 1
        else:
            kk['value'] += 1
    if d:
        anydiff.append(e)

print(f'{A_}  vs  {B_}')
print(f'  events compared: {nev}   events with ANY T_tagger diff: {len(anydiff)}')
print(f'  unstable branches: {len(cnt)}')
for br, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
    kk = kind[br]
    print(f'     {n:3d}  {br:40s}  reorder={kk["reorder"]:3d} value={kk["value"]:3d}')
if anydiff:
    print('  events: ' + ' '.join(str(e) for e in anydiff))

#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 1 -- the per-TLA plumbing gate when a tree ADDS TLAs.

scripts/cfg/tla_probe_gate.py (untouched, imported here) refuses two trees whose TLA surface differs,
which is exactly the case of a knob addition.  This wrapper reuses its probe machinery:
  (1) every TLA of the PRE tree is probed on both trees and must compile identically (nothing that
      existed before is re-plumbed).  A probe that makes wcsonnet ABORT on both trees is compared with
      the jsonnet source positions (file:LINE:COL) and the log timestamps masked, because the knob addition inserts lines and
      moves every later assert's position without touching its plumbing;
  (2) every TLA gained on POST is probed on POST and must CHANGE the compiled output vs POST's own
      null compile (the new knob is threaded, not declared-and-dropped).
Usage: tla_probe_added.py <cfgPRE> <cfgPOST> [--jobs N]
"""
import argparse, importlib.util, os, re, sys
from concurrent.futures import ThreadPoolExecutor
spec = importlib.util.spec_from_file_location(
    'tpg', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cfg', 'tla_probe_gate.py'))
tpg = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('\nsys.exit(main())', '\n')   # import without running its main
exec(compile(src, spec.origin, 'exec'), tpg.__dict__)

ap = argparse.ArgumentParser(); ap.add_argument('pre'); ap.add_argument('post'); ap.add_argument('--jobs', type=int, default=16)
a = ap.parse_args()
pre_names = tpg.tlas(a.pre); post_names = tpg.tlas(a.post)
lost = [n for n, _ in pre_names if n not in dict(post_names)]
gained = [(n, d) for n, d in post_names if n not in dict(pre_names)]
print(f'# PRE TLAs {len(pre_names)}  POST TLAs {len(post_names)}  lost {lost}  gained {[n for n, _ in gained]}')
if lost:
    print('FAIL: TLAs lost'); sys.exit(1)

MASK = re.compile(r':\(?\d+:\d+\)?(-\(?\d+:\d+\)?)?|\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+\]')   # source positions + log timestamps
def compile_norm(root, extra):
    """tpg.compile_one, plus the REALPATH spelling of the tree masked too (a trace line prints
    wirecell.jsonnet through the resolved path when the tree is given through a symlinked prefix)."""
    rc, out, err = tpg.compile_one(root, extra)
    rp = os.path.realpath(root)
    return (rc, out.replace(rp, '@CFG@'), err.replace(rp, '@CFG@'))

def one(item):
    name, default = item
    if name in tpg.BASE_KEYS:
        return (name, 'base', 'identical')
    extra = tpg.probe_arg(name, default)
    ra = compile_norm(a.pre, extra); rb = compile_norm(a.post, extra)
    if ra == rb:
        verdict = 'identical'
    elif ra[0] != 0 and ra[0] == rb[0] and MASK.sub('@', ra[1]) == MASK.sub('@', rb[1]) and MASK.sub('@', ra[2]) == MASK.sub('@', rb[2]):
        verdict = 'identical-abort-modulo-source-positions'
    else:
        verdict = 'DIFFER'
    return (name, 'ok' if ra[0] == 0 else f'rc{ra[0]}', verdict)
with ThreadPoolExecutor(max_workers=a.jobs) as ex:
    res = list(ex.map(one, pre_names))
bad = [r for r in res if r[2] == 'DIFFER']
mod = [r for r in res if r[2].startswith('identical-abort')]
print(f'(1) shared TLAs probed {len(res)}: identical {len(res) - len(bad) - len(mod)}, '
      f'identical aborts modulo source positions {len(mod)} {[r[0] for r in mod]}, DIFFER {len(bad)} {[r[0] for r in bad]}')

null_post = compile_norm(a.post, [])
rows = []
for name, default in gained:
    extra = tpg.probe_arg(name, default)
    r = compile_norm(a.post, extra)
    rows.append((name, extra, r[0], r != null_post))
    print(f'(2) gained {name}: probe {extra} rc={r[0]} changes-output={r != null_post}')
ok2 = all(r[3] for r in rows)
print('PASS' if not bad and ok2 else 'FAIL')
sys.exit(0 if not bad and ok2 else 1)

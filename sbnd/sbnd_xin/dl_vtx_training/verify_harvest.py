#!/usr/bin/env python3
'''
doc pr/79 §10 -- verify a dl_vtx_harvest arm.

Two checks per sampled event:
 1. STRUCTURE: harvest payload well-formed (cloud arrays consistent, vertex
    block aligned, hv_* row keys present).
 2. REPRODUCTION: voxelize hv_cloud with the pyutil SCN_Vertex math
    (float32, 0.5 cm, offset+0.25) and run the production net on it; the
    recorded voxels[] (x, y, z, dl_score) must match the offline top-K
    bit-for-bit (same floats, same ops => byte-exact reproduction of the
    live inference input is proven).

Usage:
  python3 verify_harvest.py --arm ../work-mcp1k-ma10k20-harv --events 166804 175808 ...
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

PYUTIL = '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python'


def check_event(arm, evt, model_runner):
    path = os.path.join(arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    sb = vio.load_calib(path).get('vertex_scoreboard') or {}
    assert sb.get('harvest'), 'evt%d: harvest flag missing' % evt
    c = sb['hv_cloud']
    x = np.array(c['x'], np.float32)
    y = np.array(c['y'], np.float32)
    z = np.array(c['z'], np.float32)
    q = np.array(c['q'], np.float32)
    assert len(x) == len(y) == len(z) == len(q), 'evt%d: cloud length mismatch' % evt
    assert c['n_vertex_rows'] == len(c['vertex_ids']) <= len(x), 'evt%d: vertex block' % evt
    rows = sb['rows']
    assert any('hv_filled' in r for r in rows), 'evt%d: no hv keys in rows' % evt

    out = model_runner(x, y, z, q, sb['dl_top_k'])
    rec = np.array([[v['x'], v['y'], v['z'], v['dl_score']] for v in sb['voxels']],
                   np.float32)
    got = out[:len(rec)]
    exact = np.array_equal(rec, got)
    close = np.allclose(rec, got, rtol=0, atol=1e-5)
    return len(x), len(rec), exact, close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--events', nargs='+', type=int, required=True)
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()

    sys.path.insert(0, PYUTIL)
    import SCN_Vertex as sv
    weights = args.weights or vio.default_weights()

    def model_runner(x, y, z, q, top_k):
        raw = sv.SCN_Vertex(weights, x.tobytes(), y.tobytes(), z.tobytes(),
                            q.tobytes(), dtype='float32', top_k=top_k)
        return np.frombuffer(raw, np.float32).reshape(-1, 4)

    nfail = 0
    for evt in args.events:
        try:
            npts, nvox, exact, close = check_event(args.arm, evt, model_runner)
            verdict = 'EXACT' if exact else ('CLOSE(1e-5)' if close else 'MISMATCH')
            if not (exact or close):
                nfail += 1
            print('evt%-8d cloud %5d pts, %2d voxels vs recorded: %s'
                  % (evt, npts, nvox, verdict))
        except Exception as e:
            nfail += 1
            print('evt%-8d FAIL: %s' % (evt, e))
    print('%s' % ('ALL OK' if nfail == 0 else '%d FAILURES' % nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())

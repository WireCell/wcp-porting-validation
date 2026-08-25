"""PYTHONPATH shim: wraps the real SCN_Vertex and dumps input/output byte hashes.

Set WCT_SCN_PROBE=<path> to append one line per inference:
  seq npts inhash outhash rawfloats...
Nothing here changes the numbers -- the real module is imported by file path
and its function is called unchanged.
"""
import os, sys, hashlib, importlib.util

_REAL = "/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python/SCN_Vertex.py"
_spec = importlib.util.spec_from_file_location("_scn_real", _REAL)
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)

_seq = [0]
_probe = os.environ.get("WCT_SCN_PROBE")


def SCN_Vertex(weights, x, y, z, q, dtype='float32', top_k=1, resolution=0.5, verbose=False):
    out = _real.SCN_Vertex(weights, x, y, z, q, dtype, top_k, resolution, verbose)
    if _probe:
        _seq[0] += 1
        h = hashlib.sha256()
        for b in (x, y, z, q):
            h.update(b)
        import numpy as np
        vals = np.frombuffer(out, dtype=dtype)
        with open(_probe, "a") as fp:
            fp.write("seq=%d pid=%d npts=%d in=%s out=%s vals=%s\n" % (
                _seq[0], os.getpid(), len(x) // 4, h.hexdigest()[:20],
                hashlib.sha256(out).hexdigest()[:20],
                ",".join("%.9e" % v for v in vals)))
    return out

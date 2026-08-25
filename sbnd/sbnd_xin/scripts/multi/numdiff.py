#!/usr/bin/env python3
"""Are the two outcome states an exact REORDERING of the same numbers, or do
the numbers themselves differ?  A pure permutation rules floating point out;
a few-ulp delta rules it back in."""
import io, sys, tarfile, zipfile, numpy as np

def load(p):
    out = {}
    if p.endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                out[n] = z.read(n)
    else:
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if m.isfile(): out[m.name] = t.extractfile(m).read()
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
nperm = nnum = nsame = nother = 0
for n in sorted(set(a) & set(b)):
    if a[n] == b[n]:
        nsame += 1; continue
    if not n.endswith(".npy"):
        nother += 1; print(f"  NON-NPY differs: {n}"); continue
    try:
        x = np.load(io.BytesIO(a[n]), allow_pickle=False)
        y = np.load(io.BytesIO(b[n]), allow_pickle=False)
    except Exception as e:
        nother += 1; print(f"  unreadable {n}: {e}"); continue
    if x.shape != y.shape:
        nother += 1
        print(f"  SHAPE {n}: {x.shape} vs {y.shape}"); continue
    if x.dtype.kind == 'f':
        xs, ys = np.sort(x.ravel()), np.sort(y.ravel())
        if np.array_equal(xs, ys):
            nperm += 1
        else:
            d = np.abs(xs - ys); m = d.max()
            scale = np.maximum(np.abs(xs), np.abs(ys)); scale[scale == 0] = 1
            rel = (d / scale).max()
            ndiff = int((xs != ys).sum())
            nnum += 1
            print(f"  VALUES {n}: {ndiff}/{xs.size} sorted entries differ, max|d|={m:.6g} max rel={rel:.3e}")
    else:
        xs, ys = np.sort(x.ravel()), np.sort(y.ravel())
        if np.array_equal(xs, ys): nperm += 1
        else:
            nnum += 1
            print(f"  INT VALUES {n}: multiset differs, {int((xs!=ys).sum())}/{xs.size}")
print(f"\nidentical={nsame}  pure-permutation={nperm}  value-differences={nnum}  other={nother}")

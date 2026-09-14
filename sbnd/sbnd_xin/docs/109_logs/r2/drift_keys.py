import json, sys
def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flatten(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flatten(v, f"{p}[{i}]")
    else: yield p, o
A, B = sys.argv[1], sys.argv[2]
for name in sys.argv[3:]:
    a = dict(flatten(json.load(open(f"{A}/{name}")))); b = dict(flatten(json.load(open(f"{B}/{name}"))))
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in b: out.append(f"  REMOVED {k} = {a[k]!r}")
        elif k not in a: out.append(f"  ADDED   {k} = {b[k]!r}")
        elif a[k] != b[k]: out.append(f"  CHANGED {k} : {a[k]!r} -> {b[k]!r}")
    print(f"== {name}: {len(out)} keys (eacacafe -> c203b400)")
    print("\n".join(out[:40]) + (f"\n  ... {len(out)-40} more" if len(out) > 40 else ""))

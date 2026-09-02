#!/usr/bin/env python3
"""doc 97 -- name, key by key, exactly what the sep_fv_point flip changes.

prod_cfg_gate.py reports WHICH artifacts drifted and can name the keys only for
prod_prjob.json (the SBND PR job).  This flip lands in the SBND clustering/Q/L
artifacts instead, so this walks every compiled node of every drifted artifact
and reports each added / removed / changed key with its (type, name) -- never
by array index, which reorders.

  Usage: d97_flip_drift.py <off-compiled-dir> <on-compiled-dir> [artifact ...]
Read-only.
"""
import json, os, sys

OFF, ON = sys.argv[1], sys.argv[2]
NAMES = sys.argv[3:] or ['sbnd_clus.json', 'sbnd_ql.json', 'prod.standalone', 'prod.wcls',
                         'sbnd_pr.json', 'sbnd_img.json', 'prod_prjob.json', 'bare_prjob.json',
                         'uboone.json']


def nodes(path):
    """{(type, name): data-dict} for one compiled config."""
    try:
        d = json.load(open(path))
    except Exception as e:
        return None
    out = {}
    for n in d if isinstance(d, list) else []:
        out[(n.get('type', ''), n.get('name', ''))] = n.get('data')
    return out


total = 0
for name in NAMES:
    a, b = nodes(os.path.join(OFF, name)), nodes(os.path.join(ON, name))
    if a is None or b is None:
        print(f'{name}: not comparable ({"off" if a is None else "on"} missing/unparsable)')
        continue
    lines = []
    for k in sorted(set(a) | set(b), key=lambda x: (x[0], x[1])):
        if k not in a:
            lines.append(f'    NODE ADDED   {k[0]}:{k[1]}'); continue
        if k not in b:
            lines.append(f'    NODE REMOVED {k[0]}:{k[1]}'); continue
        da, db = a[k] or {}, b[k] or {}
        if not isinstance(da, dict) or not isinstance(db, dict):
            if da != db:
                lines.append(f'    DATA CHANGED {k[0]}:{k[1]}')
            continue
        for key in sorted(set(da) | set(db)):
            if key not in da:
                lines.append(f'    ADDED   {k[0]}:{k[1]} .{key} = {db[key]!r}')
            elif key not in db:
                lines.append(f'    REMOVED {k[0]}:{k[1]} .{key} = {da[key]!r}')
            elif da[key] != db[key]:
                lines.append(f'    CHANGED {k[0]}:{k[1]} .{key} = {da[key]!r} -> {db[key]!r}')
    total += len(lines)
    print(f'{name}: {len(lines)} key difference(s)' if lines else f'{name}: identical')
    for l in lines:
        print(l)
print(f'\nTOTAL key differences across {len(NAMES)} artifacts: {total}')

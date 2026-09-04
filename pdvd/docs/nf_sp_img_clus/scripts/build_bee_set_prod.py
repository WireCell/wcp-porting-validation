#!/usr/bin/env python3
"""doc pdvd/31 round 7: build one Bee set from named arms, with the steiner layers.

Same construction as build_bee_sets.py's set A -- every layer of each arm's
mabc-pr.zip, plus two layers synthesised from the calib-pr dump's `steiner`
section -- but parameterised, so a set can be assembled for any event/arm list
without editing a hard-coded table.

  steiner-global      every point of every cluster's steiner_pc
  steinerterm-global  the subset with flag_steiner_terminal == 1

No `q` field is written: the calib section carries no charge, and sst.js only
applies its QTHRESH=500 cut to layers named `truth` or `L1` (wire-cell-bee3
sst.js:47-52), so a null q is rendered as 0 with no cut.  Layer names are
otherwise arbitrary -- recon_list (events/models.py:67-86) lists any name that
does not contain `-track`, start with `channel`, or contain `auto-sel`.

Usage:
  python3 build_bee_set_prod.py OUTDIR SETNAME \
      039252_2_d31r4on:calib-pr-evt298595.json \
      039252_2_d31r6e2e:calib-pr-evt298595.json
"""
import json
import os
import sys
import zipfile

W = '/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work'


def steiner_layers(calib_path, meta):
    d = json.load(open(calib_path))
    allp = {k: [] for k in ('x', 'y', 'z', 'cluster_id', 'real_cluster_id')}
    trmp = {k: [] for k in allp}
    for e in d.get('steiner', []):
        cid = int(e['cluster_id'])
        flag = e.get('flag_terminal', [])
        for i in range(len(e['x'])):
            for tgt in (allp,) + ((trmp,) if i < len(flag) and flag[i] else ()):
                tgt['x'].append(e['x'][i]); tgt['y'].append(e['y'][i]); tgt['z'].append(e['z'][i])
                tgt['cluster_id'].append(cid); tgt['real_cluster_id'].append(cid)
    return {name: dict(body, type=name, **meta)
            for name, body in (('steiner-global', allp), ('steinerterm-global', trmp))}


def build(out, setname, arms):
    stage = os.path.join(out, setname, 'data')
    os.makedirs(stage, exist_ok=True)
    for idx, spec in enumerate(arms):
        tag, calib = spec.split(':', 1)
        ev = os.path.join(stage, str(idx))
        os.makedirs(ev, exist_ok=True)
        meta = None
        with zipfile.ZipFile(f'{W}/{tag}/mabc-pr.zip') as src:
            for member in src.namelist():
                base = os.path.basename(member)              # "0-clustering-global.json"
                layer = base[base.find('-') + 1:-5]
                raw = src.read(member)
                if meta is None and layer == 'clustering-global':
                    j = json.loads(raw)
                    meta = {k: j[k] for k in ('runNo', 'subRunNo', 'eventNo', 'geom')}
                open(os.path.join(ev, f'{idx}-{layer}.json'), 'wb').write(raw)
        for name, body in steiner_layers(f'{W}/{tag}/{calib}', meta).items():
            json.dump(body, open(os.path.join(ev, f'{idx}-{name}.json'), 'w'))
            print(f'  event {idx} ({tag}): {name} {len(body["x"])} points')
    zpath = os.path.join(out, setname + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(out, setname)):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.join(out, setname)))
    print(f'{setname}: {os.path.getsize(zpath)/1e6:.2f} MB  {zpath}')
    return zpath


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], sys.argv[3:])

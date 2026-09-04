#!/usr/bin/env python3
"""doc pdvd/31 round 6: build the three hand-scan Bee sets.

Set A  density (039349/14)  : 3 events = production / retile-fix / both knobs,
                              each carrying two NEW layers built from the calib
                              dump's `steiner` section (x/y/z + flag_terminal):
                                steiner-global      all steiner points
                                steinerterm-global  flag_terminal == 1 only
Set B  cluster 47 (039349/14): round-3 knob OFF (event 0) vs ON (event 1)
Set C  cluster 109 (039252/2): round-3 knob OFF (event 0) vs ON (event 1)

Frame proof (run before this script, see frame_check.py): cluster 34's 2561
steiner points nearest-neighbour into that arm's own Bee clustering-global at
median 0.83 cm and match Bee cluster_id 34 on 2561 of 2561 -- same frame, same
numbering.  No `q` field is written for the steiner layers: the calib section
carries no charge and sst.js treats a null q as 0 with no threshold cut.
"""
import json, os, shutil, sys, zipfile

W = '/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work'
OUT = sys.argv[1]

SETS = {
    'A_density_039349_14': [
        ('039349_14_d31r5off',  'calib-pr-evt19689.json'),
        ('039349_14_d31r5on',   'calib-pr-evt19689.json'),
        ('039349_14_d31r5both', 'calib-pr-evt19689.json'),
    ],
    'B_cl47_039349_14': [
        ('039349_14_d31fix2off', None),
        ('039349_14_d31fix2on',  None),
    ],
    'C_cl109_039252_2': [
        ('039252_2_d31r4off', None),
        ('039252_2_d31r4on',  None),
    ],
}

def steiner_layers(calib_path, meta):
    """Two point-cloud layers from the calib dump's steiner section."""
    d = json.load(open(calib_path))
    allp = {k: [] for k in ('x', 'y', 'z', 'cluster_id', 'real_cluster_id')}
    trmp = {k: [] for k in allp}
    for e in d.get('steiner', []):
        cid = int(e['cluster_id'])
        flag = e['flag_terminal']
        for i in range(len(e['x'])):
            for tgt in (allp,) + ((trmp,) if flag[i] else ()):
                tgt['x'].append(e['x'][i]); tgt['y'].append(e['y'][i]); tgt['z'].append(e['z'][i])
                tgt['cluster_id'].append(cid); tgt['real_cluster_id'].append(cid)
    out = {}
    for name, body in (('steiner-global', allp), ('steinerterm-global', trmp)):
        out[name] = dict(body, type=name, **meta)
    return out

def build(setname, arms):
    stage = os.path.join(OUT, setname, 'data')
    os.makedirs(stage, exist_ok=True)
    for idx, (tag, calib) in enumerate(arms):
        ev = os.path.join(stage, str(idx))
        os.makedirs(ev, exist_ok=True)
        src = zipfile.ZipFile(f'{W}/{tag}/mabc-pr.zip')
        meta = None
        for member in src.namelist():
            base = os.path.basename(member)                 # "0-clustering-global.json"
            layer = base[base.find('-') + 1:-5]
            raw = src.read(member)
            if meta is None and layer == 'clustering-global':
                j = json.loads(raw)
                meta = {k: j[k] for k in ('runNo', 'subRunNo', 'eventNo', 'geom')}
            open(os.path.join(ev, f'{idx}-{layer}.json'), 'wb').write(raw)
        if calib:
            for name, body in steiner_layers(f'{W}/{tag}/{calib}', meta).items():
                json.dump(body, open(os.path.join(ev, f'{idx}-{name}.json'), 'w'))
                print(f'  {setname} event {idx} ({tag}): {name} {len(body["x"])} points')
    zpath = os.path.join(OUT, setname + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(OUT, setname)):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.join(OUT, setname)))
    print(f'{setname}: {os.path.getsize(zpath)/1e6:.2f} MB  {zpath}')
    return zpath

for k, v in SETS.items():
    build(k, v)

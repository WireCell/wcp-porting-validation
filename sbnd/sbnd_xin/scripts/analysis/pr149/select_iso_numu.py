#!/usr/bin/env python3
"""doc pr/149 Stage 2: the numuCC large-ISO manifest, exactly as 149_pred.txt sec 3 froze it.

Source: the epoch-reference metrics 149_figs/metrics/d102mpr-{mcp1k,mcp2k}.tsv
(pr149_metrics.py extract --arm d102mpr) -- never a measured arm.

  pool     events with a calib dump (has_calib == 1)
  iso      iso_len80 >= 30 cm, ranked by iso_len80 desc (ties: sample, event), top 150
  vtx_iso  vtx105-labelled events with iso_len75 >= 10 cm not already in iso
  owner    the 5 mcp1k owner cases (docs pr/24, 45, 67, 73)
  control  150 events, numpy default_rng(149).choice without replacement from the pool with
           iso_len_any75 == 0, pool sorted by (sample, event) first

Writes 149_figs/manifest_stage2/{mcp1k,mcp2k}.txt and 149_figs/149_stage2_selection.tsv.
"""
import csv
import math
import os
import sys

import numpy as np

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
FIGS = os.path.join(SX, 'docs/pr/149_figs')
sys.path.insert(0, SX)
from vtx_rules import vtx_io  # noqa: E402

OWNER = [('mcp1k', 57903), ('mcp1k', 56463), ('mcp1k', 284794), ('mcp1k', 59899), ('mcp1k', 58717)]


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except ValueError:
        return float('nan')


def main():
    rows = []
    for s in ('mcp1k', 'mcp2k'):
        with open(os.path.join(FIGS, 'metrics', f'd102mpr-{s}.tsv')) as fh:
            rows += [r for r in csv.DictReader(fh, delimiter='\t')]
    pool = [r for r in rows if r['has_calib'] == '1']
    key = lambda r: (r['sample'], int(r['event']))
    labelled = {d['eventNo'] for d in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if d['truth'] is not None}

    iso_c = [r for r in pool if f(r['iso_len80']) >= 30]
    iso_c.sort(key=lambda r: (-f(r['iso_len80']), r['sample'], int(r['event'])))
    iso = iso_c[:150]
    chosen = {key(r) for r in iso}
    vtx_iso = sorted([r for r in pool if int(r['event']) in labelled and f(r['iso_len75']) >= 10
                      and key(r) not in chosen], key=key)
    chosen |= {key(r) for r in vtx_iso}
    byk = {key(r): r for r in rows}
    owner = [byk[k] for k in OWNER if k in byk and k not in chosen]
    chosen |= {key(r) for r in owner}
    ctl_pool = sorted([r for r in pool if f(r['iso_len_any75']) == 0], key=key)
    rng = np.random.default_rng(149)
    idx = rng.choice(len(ctl_pool), size=min(150, len(ctl_pool)), replace=False)
    control = sorted([ctl_pool[i] for i in idx], key=key)

    out = os.path.join(FIGS, '149_stage2_selection.tsv')
    with open(out, 'w') as fo:
        fo.write('stratum\tsample\tevent\tiso_len80\tiso_len75\tiso_len_any75\tlabelled\n')
        for name, rs in (('iso', iso), ('vtx_iso', vtx_iso), ('owner', owner), ('control', control)):
            for r in rs:
                fo.write(f"{name}\t{r['sample']}\t{r['event']}\t{r['iso_len80']}\t{r['iso_len75']}\t"
                         f"{r['iso_len_any75']}\t{int(int(r['event']) in labelled)}\n")
    man = os.path.join(FIGS, 'manifest_stage2')
    os.makedirs(man, exist_ok=True)
    allk = sorted({key(r) for r in iso + vtx_iso + owner + control})
    for s in ('mcp1k', 'mcp2k'):
        with open(os.path.join(man, f'{s}.txt'), 'w') as fo:
            fo.write(f'# doc pr/149 Stage 2 manifest ({s}); rule 149_pred.txt sec 3\n')
            for ss, e in allk:
                if ss == s:
                    fo.write(f'{e}\n')
    print(f'pool {len(pool)} of {len(rows)}; iso candidates (iso_len80>=30) {len(iso_c)} -> iso {len(iso)} '
          f'(min iso_len80 {f(iso[-1]["iso_len80"]):.1f} cm); vtx_iso {len(vtx_iso)}; owner extra {len(owner)}; '
          f'control pool {len(ctl_pool)} -> {len(control)}; total {len(allk)} '
          f'(mcp1k {sum(1 for k in allk if k[0] == "mcp1k")}, mcp2k {sum(1 for k in allk if k[0] == "mcp2k")}); '
          f'labelled in iso {sum(1 for r in iso if int(r["event"]) in labelled)}, '
          f'in control {sum(1 for r in control if int(r["event"]) in labelled)}')


if __name__ == '__main__':
    main()

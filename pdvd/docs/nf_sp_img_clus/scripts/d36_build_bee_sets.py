#!/usr/bin/env python3
"""doc pdvd/36 follow-up: hand-scan Bee sets for the ctpc-metric comparison.

Each set is ONE physics event carried three times, so the Bee event index is the
arm:
    event 0 = d36p000   isotropic ctpc, good_point_pitch_frac 0  (legacy)
    event 1 = d36off    isotropic ctpc, frac 0.35                (PRODUCTION)
    event 2 = d36on     anisotropic two-level metric, frac 0     (candidate)

The three arms share the same imaging input, so the `clustering-global` charge
point set is identical in all three (verified: lexsorted arrays equal); only the
`stm_fit` / `track_fit` / `vertices` trajectory layers differ.
"""
import json, os, sys, zipfile
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
OUT=sys.argv[1]
ARMS=['d36p000','d36off','d36on']
SETS={
 'd36_A_cl109_039252_2':   '039252_2',
 'd36_B_cl53_039349_48':   '039349_48',
 'd36_C_cl103_039252_16':  '039252_16',
 'd36_D_cl68_039349_58':   '039349_58',
 'd36_E_cl71_039349_3':    '039349_3',
}
def build(setname, ev):
    stage=os.path.join(OUT,setname,'data')
    os.makedirs(stage,exist_ok=True)
    for idx,tag in enumerate(ARMS):
        d=os.path.join(stage,str(idx)); os.makedirs(d,exist_ok=True)
        src=zipfile.ZipFile(f'{W}/{ev}_{tag}/mabc-pr.zip')
        for member in src.namelist():
            base=os.path.basename(member)
            layer=base[base.find('-')+1:-5]
            open(os.path.join(d,f'{idx}-{layer}.json'),'wb').write(src.read(member))
    z=os.path.join(OUT,setname+'.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zf:
        for root,_,files in os.walk(os.path.join(OUT,setname)):
            for f in sorted(files):
                p=os.path.join(root,f)
                zf.write(p,os.path.relpath(p,os.path.join(OUT,setname)))
    print(f'{setname}: {os.path.getsize(z)/1e6:.2f} MB  {z}')
    return z
for k,v in SETS.items(): build(k,v)

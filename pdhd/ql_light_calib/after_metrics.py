#!/usr/bin/env python3
"""Real-C++ normalization-study metrics from QLMatching calib dumps (run 29107).

Reads the matched bundles' own ``pred_pe`` and ``ks_dis`` (so KS is the matcher's
metric by construction) for the clean two-boundary crosser anchors, and reports
the aggregate N90 / dark-fraction / direct-PMT-scale / integral-scale table used
in the study doc.  Point it at the BEFORE dumps (lambda=100, eff=0.023) or the
reprocessed AFTER dumps (lambda=300, eff=0.0145) via DUMPDIR.

Usage: ``python3 after_metrics.py [DUMPDIR]``  (default = production work dir)."""
import json, glob, sys, statistics

RUN=29107
DUMPDIR = sys.argv[1] if len(sys.argv)>1 else \
    '/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work'
STATIC=set([3,86,87,97,107,116,117])|set(range(120,160))

def n90(v):
    s=sum(v)
    if s<=0: return 0
    c=0;n=0
    for x in sorted(v,reverse=True):
        c+=x;n+=1
        if c>=0.9*s: break
    return n

rows=[]
for g0 in sorted(glob.glob('%s/0%d_*/calib-*-group*.json'%(DUMPDIR,RUN))):
    d=json.load(open(g0)); art=g0.split('/')[-1].split('-')[1]; grp=g0.split('-')[-1].split('.')[0]
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    flB={f['gid']:f for f in d['flashes']}
    byflash={}
    for b in d['bundles']:
        if not b.get('at_x_boundary') or not b.get('two_boundary') or b['ndf']<30: continue
        f=flB.get(b['flash_gid'])
        if f is None: continue
        meas=[f['pe'][i] for i in valid]
        if max(meas)<=300 or sum(meas)<=3000 or b['ks_dis']>0.40: continue
        k=b['flash_gid']
        if k not in byflash or b['ks_dis']<byflash[k]['ks_dis']: byflash[k]=b
    for b in byflash.values():
        f=flB[b['flash_gid']]
        meas=[f['pe'][i] for i in valid]; pred=[b['pred_pe'][i] for i in valid]
        if n90(meas)<2 or sum(pred)<=0: continue
        dark=sum(pred[k] for k in range(len(valid)) if meas[k]==0)/sum(pred)
        order=sorted(range(len(valid)),key=lambda k:-meas[k])[:3]
        ds=sum(meas[k] for k in order)/sum(pred[k] for k in order)
        rows.append(dict(art=art,grp=grp,ndf=b['ndf'],ks=b['ks_dis'],measTot=sum(meas),
                         N90m=n90(meas),N90p=n90(pred),dark=dark*100,ds=ds,isc=sum(meas)/sum(pred)))

clean=[r for r in rows if r['ks']<0.20]
print('dumpdir: %s'%DUMPDIR)
print('two-boundary anchors: %d   (ks<0.20: %d)'%(len(rows),len(clean)))
print('%-8s %-7s %4s %6s %5s %5s %6s %7s %7s'%('evt','grp','ndf','ks','N90m','N90p','dark%','direct','integ'))
for r in sorted(clean,key=lambda r:-r['measTot']):
    print('%-8s %-7s %4d %6.3f %5d %5d %6.0f %7.2f %7.2f'%(
        r['art'],r['grp'],r['ndf'],r['ks'],r['N90m'],r['N90p'],r['dark'],r['ds'],r['isc']))
def med(k): return statistics.median([r[k] for r in clean])
print('\nMEDIAN (ks<0.20, %d): KS=%.3f  N90rat=%.2f  dark%%=%.0f  direct=%.2f  integral=%.2f'%(
    len(clean), med('ks'), statistics.median([r['N90p']/r['N90m'] for r in clean]),
    med('dark'), med('ds'), med('isc')))

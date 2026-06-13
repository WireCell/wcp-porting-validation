#!/usr/bin/env python3
"""Normalization-study metrics on the REPROCESSED dumps (lambda=100, vuv_eff=0.023),
over ALL 27305 events on disk. Crosser anchors: at_x_boundary, ndf>=30, brightest
valid meas>300, dedup by main cluster keeping lowest ks. Uses pred_pe from the dumps."""
import json, glob, statistics
STATIC=set([3,86,87,97,107,116,117])|set(range(120,160))
def n90(v):
    s=sum(v)
    if s<=0: return 0
    c=0;n=0
    for x in sorted(v,reverse=True):
        c+=x;n+=1
        if c>=0.9*s: break
    return n
WORK='/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work'
dumps=sorted(glob.glob(WORK+'/027305_*/calib-*-group*.json'))
events=sorted(set(p.rsplit('/',2)[1] for p in dumps))
rows=[]
for g0 in dumps:
    d=json.load(open(g0)); art=g0.split('/')[-1].split('-')[1]; grp=g0.split('-')[-1].split('.')[0]
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    flB={f['gid']:f for f in d['flashes']}
    best={}
    for b in d['bundles']:
        if not b.get('at_x_boundary') or b['ndf']<30: continue
        f=flB.get(b['flash_gid'])
        if f is None or max(f['pe'][i] for i in valid)<=300: continue
        k=b['main_cluster']
        if k not in best or b['ks_dis']<best[k]['ks_dis']: best[k]=b
    for b in best.values():
        f=flB[b['flash_gid']]
        meas=[f['pe'][i] for i in valid]; pred=[b['pred_pe'][i] for i in valid]
        if n90(meas)<2 or sum(pred)<=0: continue
        dark=sum(pred[k] for k in range(len(valid)) if meas[k]==0)/sum(pred)
        order=sorted(range(len(valid)),key=lambda k:-meas[k])[:3]
        ds=sum(meas[k] for k in order)/sum(pred[k] for k in order)
        rows.append(dict(art=art,grp=grp,main=b['main_cluster'],ndf=b['ndf'],ks=b['ks_dis'],
                         measTot=sum(meas),N90m=n90(meas),N90p=n90(pred),dark=dark*100,
                         ds=ds,isc=sum(meas)/sum(pred)))
print('events on disk: %d   (%s)'%(len(events),', '.join(e.split('_')[1] for e in events)))
print('all at_x/ndf>=30/bright anchors: %d'%len(rows))
clean=[r for r in rows if r['measTot']>3000 and r['ks']<0.4]
print()
print('%-7s %-7s %5s %6s %5s %5s %6s %6s %6s'%('evt','grp','ndf','ks','N90m','N90p','dark%','dscale','iscale'))
for r in sorted(clean,key=lambda r:(r['art'],r['grp'])):
    print('%-7s %-7s %5d %6.3f %5d %5d %6.0f %6.2f %6.2f'%(
        r['art'],r['grp'],r['ndf'],r['ks'],r['N90m'],r['N90p'],r['dark'],r['ds'],r['isc']))
print()
def med(k): return statistics.median([r[k] for r in clean])
print('CLEAN anchors (measTot>3000, ks<0.4): %d'%len(clean))
print('MEDIAN  N90m=%.0f  N90p=%.0f  N90ratio=%.2f  dark%%=%.0f  direct-scale=%.2f  integral-scale=%.2f'%(
    med('N90m'),med('N90p'),statistics.median([r['N90p']/r['N90m'] for r in clean]),
    med('dark'),med('ds'),med('isc')))

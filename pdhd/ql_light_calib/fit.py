#!/usr/bin/env python3
"""Joint {lambda, normalization} fit on CLEAN crosser anchors.
Objective for lambda = KS shape distance between predicted and measured optical
patterns (the matcher's own shape metric), independent of the scalar norm.
Then the normalization = measured/predicted on the directly-lit PMTs at lambda*."""
import json, glob, sys, statistics
sys.path.insert(0,'/home/xqian/tmp/ql_lambda')
from repredict import predict

STATIC=set([3,86,87,97,107,116,117])|set(range(120,160))
LAMBDAS=[2000,400,300,250,200,175,150,125,110,100,90,80,70,60]

def n90(vals):
    s=sum(vals);
    if s<=0: return 0
    c=0;n=0
    for v in sorted(vals,reverse=True):
        c+=v;n+=1
        if c>=0.9*s: break
    return n

def ks_shape(meas,pred,valid):
    """KS distance between the two normalized PE patterns, channels ordered by
    measured PE (the std 1-D KS used for flash shape comparison)."""
    m=[meas[i] for i in valid]; p=[pred[i] for i in valid]
    sm=sum(m); sp=sum(p)
    if sm<=0 or sp<=0: return 1.0
    order=sorted(range(len(valid)),key=lambda k:m[k])
    cm=cp=0.0; d=0.0
    for k in order:
        cm+=m[k]/sm; cp+=p[k]/sp
        d=max(d,abs(cm-cp))
    return d

def load(p):
    d=json.load(open(p)); return d,{c['ident']:c for c in d['clusters']}

def pts_for(b,cl):
    out=[]
    for cid in [b['main_cluster']]+b.get('other_clusters',[]):
        c=cl.get(cid%1000000)
        if c: out+=list(zip(c['x'],c['y'],c['z'],c['q']))
    return out

# ---- collect clean anchors: at_x_boundary, ndf>=30, ks<0.20, brightest valid meas>300,
#      dedup by main cluster keeping lowest ks ----
anchors={}
for ev in range(10):
    for grp in ('group02','group13'):
        g=glob.glob('/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work/027305_%d/calib-*-%s.json'%(ev,grp))
        if not g: continue
        d,cl=load(g[0])
        art=g[0].split('/')[-1].split('-')[1]   # 'calib-evtNNN-groupXX.json' -> 'evtNNN'
        gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
        geom={k:gr[k] for k in ('anode_x','s','u_cathode','y_lo','y_hi','z_lo','z_hi')}
        trig=d['trigger_offset']; vd=d['drift_speed']; sign=gr['sign_offset']
        flB={f['gid']:f for f in d['flashes']}
        valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
               and (d['opdets'][i]['x']>0)==(geom['anode_x']>0)]
        for b in d['bundles']:
            if not b.get('at_x_boundary') or b['ndf']<30 or b['ks_dis']>0.40: continue
            f=flB.get(b['flash_gid'])
            if f is None or max(f['pe'][i] for i in valid)<=1000: continue
            key=(art,grp,b['main_cluster'])
            if key not in anchors or b['ks_dis']<anchors[key]['b']['ks_dis']:
                anchors[key]=dict(b=b,f=f,geom=geom,offset=sign*(f['time']+trig)*vd,
                                  valid=valid,pts=pts_for(b,cl),art=art,grp=grp)
A=list(anchors.values())
print('clean anchors: %d'%len(A))
for a in A: print('  %s %s main%d ndf=%d ks=%.3f measTot=%.0f npts=%d'%(
    a['art'],a['grp'],a['b']['main_cluster'],a['b']['ndf'],a['b']['ks_dis'],
    sum(a['f']['pe'][i] for i in a['valid']),len(a['pts'])))
print()

# ---- per-anchor lambda sweep: KS, N90, direct-scale ----
best=[]
for a in A:
    meas=a['f']['pe']
    rows=[]
    for lam in LAMBDAS:
        pred=predict(a['pts'],a['geom'],a['offset'],float(lam))
        m=[meas[i] for i in a['valid']]; p=[pred[i] for i in a['valid']]
        order=sorted(range(len(a['valid'])),key=lambda k:-m[k])[:3]
        ds=sum(m[k] for k in order)/sum(p[k] for k in order) if sum(p[k] for k in order)>0 else 0
        rows.append((lam,ks_shape(meas,pred,a['valid']),n90(m),n90(p),ds))
    bl=min(rows,key=lambda r:r[1])
    measN90=bl[2]
    # lambda where predicted N90 first matches measured N90 (concentration match)
    n90match=min(rows,key=lambda r:(abs(r[3]-measN90), -r[0]))
    a['n90match']=n90match[0]; a['dscale_n90']=n90match[4]
    best.append((a,bl,rows))
    print('=== %s %s main%d  (measN90=%d) ==='%(a['art'],a['grp'],a['b']['main_cluster'],bl[2]))
    print(' lam |  KS  N90p dscale')
    for lam,ks,N90m,N90p,ds in rows:
        star=' *' if lam==bl[0] else ''
        print(' %4d| %.3f %3d  %7.2f%s'%(lam,ks,N90p,ds,star))
    print()

print('==== SUMMARY ====')
n90lams=[a['n90match'] for a,_,_ in best]
print('per-anchor N90-match lambda (concentration):',sorted(n90lams))
print('median N90-match lambda = %.0f cm  (mean %.0f)'%(statistics.median(n90lams),statistics.mean(n90lams)))
lstar=statistics.median(n90lams)
dss=[]
for a,bl,rows in best:
    pred=predict(a['pts'],a['geom'],a['offset'],lstar)
    meas=a['f']['pe']; m=[meas[i] for i in a['valid']]; p=[pred[i] for i in a['valid']]
    order=sorted(range(len(a['valid'])),key=lambda k:-m[k])[:3]
    sp=sum(p[k] for k in order)
    if sp>0: dss.append(sum(m[k] for k in order)/sp)
print('at lambda*=%.0f cm: direct-PMT scale per anchor = %s'%(lstar,['%.2f'%x for x in dss]))
print('   median direct scale = %.3f'%statistics.median(dss))
print('   => QtoL*VUVEff: current 0.0300  ->  suggested %.4f  (vs %.4f if tuned at lambda=2000)'%(
   0.03*statistics.median(dss), 0.03*0.45))

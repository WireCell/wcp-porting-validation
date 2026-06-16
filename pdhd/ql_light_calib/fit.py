#!/usr/bin/env python3
"""Joint {lambda, normalization} fit on CLEAN two-boundary crosser anchors (run 29107).

Objective for lambda = the matcher's OWN KS shape distance (calc_ks_test: an
index-order CDF over all 160 channels, with the prediction zeroed on the
static|auto masked channels exactly as the C++ opdet_mask does).  This Python KS
reproduces the dumped C++ ``ks_dis`` to machine precision at lambda=100 (the
production tuning the dumps were generated at), so the lambda sweep below is the
matcher's metric, not a proxy.  The re-predictor (repredict.py) reproduces the
C++ ``pred_pe`` to ~1e-12, so the whole per-PMT pattern is recomputable at any
lambda without re-running the chain.

Normalization = measured/predicted on the directly-lit PMTs (top-3) at lambda*.

Parallelised across anchors (multiprocessing).  Run: ``python3 fit.py``."""
import json, glob, sys, os, statistics
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUN      = 29107
NEVENTS  = 30
WORK     = '/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work'
EFF_BASE = 0.023                  # eff the pred is computed at; new_eff = EFF_BASE * direct_scale
NPROC    = 32
LAMBDAS  = [2000,800,600,500,400,350,300,250,200,175,150,125,110,100,90,80,70,60]
STATIC   = set([3,86,87,97,107,116,117]) | set(range(120,160))

def n90(v):
    s=sum(v)
    if s<=0: return 0
    c=0;n=0
    for x in sorted(v,reverse=True):
        c+=x;n+=1
        if c>=0.9*s: break
    return n

def calc_ks(meas, pred):
    """Exact port of TimingTPCBundle::calc_ks_test -- index-order cumulative CDFs
    of the two normalised distributions, max |CDF_meas - CDF_pred|."""
    sm=sum(meas); sp=sum(pred)
    if sp<=0: return 0.0
    m=[x/sm for x in meas] if sm>0 else meas
    p=[x/sp for x in pred]
    cm=cp=0.0; d=0.0
    for i in range(len(meas)): cm+=m[i]; cp+=p[i]; d=max(d,abs(cm-cp))
    return d

def pts_for(b, cl):
    out=[]
    for cid in [b['main_cluster']]+b.get('other_clusters',[]):
        c=cl.get(cid%1000000)
        if c: out+=list(zip(c['x'],c['y'],c['z'],c['q']))
    return out

def collect(two_only=True):
    """Clean anchors: at_x_boundary (two_boundary if two_only), ndf>=30,
    brightest-valid meas>300, measTot>3000, ks<0.40, dedup per FLASH (lowest ks)."""
    A=[]
    for ev in range(NEVENTS):
        for grp in ('group02','group13'):
            g=glob.glob('%s/0%d_%d/calib-*-%s.json'%(WORK,RUN,ev,grp))
            if not g: continue
            d=json.load(open(g[0])); cl={c['ident']:c for c in d['clusters']}
            art=g[0].split('/')[-1].split('-')[1]
            gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
            geom={k:gr[k] for k in ('anode_x','s','u_cathode','y_lo','y_hi','z_lo','z_hi')}
            trig=d['trigger_offset']; vd=d['drift_speed']; sign=gr['sign_offset']
            amask=set(i for i in range(160) if d['opdets'][i]['auto_masked'])
            mask=sorted(STATIC|amask)
            valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
                   and (d['opdets'][i]['x']>0)==(geom['anode_x']>0)]
            flB={f['gid']:f for f in d['flashes']}
            byflash={}
            for b in d['bundles']:
                if not b.get('at_x_boundary') or b['ndf']<30: continue
                if two_only and not b.get('two_boundary'): continue
                f=flB.get(b['flash_gid'])
                if f is None: continue
                meas=[f['pe'][i] for i in valid]
                if max(meas)<=300 or sum(meas)<=3000 or b['ks_dis']>0.40: continue
                k=b['flash_gid']
                if k not in byflash or b['ks_dis']<byflash[k]['ks_dis']: byflash[k]=b
            for b in byflash.values():
                f=flB[b['flash_gid']]
                A.append(dict(art=art,grp=grp,two=bool(b.get('two_boundary')),
                    ndf=b['ndf'],ks=b['ks_dis'],main=b['main_cluster'],
                    pe160=list(f['pe']), valid=valid, mask=mask,
                    pts=pts_for(b,cl), geom=geom, offset=sign*(f['time']+trig)*vd))
    return A

def _init():
    import repredict; repredict.VUVEFF=EFF_BASE
    global predict
    from repredict import predict

def sweep_one(a):
    pe160=a['pe160']; valid=a['valid']; mask=set(a['mask'])
    meas=[pe160[i] for i in valid]; N90m=n90(meas); smeas=sum(meas)
    order3=sorted(range(len(meas)),key=lambda k:-meas[k])[:3]
    rows=[]
    for lam in LAMBDAS:
        pf=predict(a['pts'],a['geom'],a['offset'],float(lam))
        prm=[0.0 if i in mask else pf[i] for i in range(160)]    # C++ opdet_mask on pred
        kc=calc_ks(pe160, prm)
        p=[pf[i] for i in valid]
        sp3=sum(p[k] for k in order3); ds=sum(meas[k] for k in order3)/sp3 if sp3>0 else 0
        sp=sum(p)
        dark=(sum(p[k] for k in range(len(meas)) if meas[k]==0)/sp) if sp>0 else 0
        isc=(smeas/sp) if sp>0 else 0
        rows.append((lam,kc,N90m,n90(p),ds,sp,dark,isc))
    kmin=min(rows,key=lambda r:r[1]); n90m=min(rows,key=lambda r:(abs(r[3]-N90m),-r[0]))
    return dict(art=a['art'],grp=a['grp'],two=a['two'],ndf=a['ndf'],ks=a['ks'],
                main=a['main'],N90m=N90m,kslam=kmin[0],n90lam=n90m[0],rows=rows,measTot=smeas)

def med(xs): return statistics.median(xs) if xs else 0
def row_at(r,L): return [x for x in r['rows'] if x[0]==L][0]

if __name__=='__main__':
    A=collect(two_only=True)
    print('run %d: clean two-boundary anchors = %d'%(RUN,len(A)))
    with Pool(NPROC,initializer=_init) as pool:
        R=pool.map(sweep_one,A,chunksize=2)
    json.dump(R,open(os.path.join(HERE,'sweep_rows_%d.json'%RUN),'w'))
    for label,sub in (('two-boundary (all)',R),
                      ('two-boundary ks<0.20 [primary]',[r for r in R if r['ks']<0.20])):
        kl=[r['kslam'] for r in sub]
        print('\n==== %s : %d anchors ===='%(label,len(sub)))
        print('  per-anchor KS-min lambda: median=%.0f'%med(kl))
        print('  %5s %8s %8s %7s %8s %9s'%('lam','medKS','N90rat','dark%','iscale','newEff'))
        for L in [150,200,250,300,350,400,500]:
            ks =[row_at(r,L)[1] for r in sub]
            n90r=[row_at(r,L)[3]/row_at(r,L)[2] for r in sub if row_at(r,L)[2]]
            dk =[row_at(r,L)[6]*100 for r in sub]
            isc=[row_at(r,L)[7] for r in sub if row_at(r,L)[7]>0]
            ds =[row_at(r,L)[4] for r in sub if row_at(r,L)[4]>0]
            print('  %5d %8.4f %8.2f %7.0f %8.2f %9.4f'%(L,med(ks),med(n90r),med(dk),med(isc),EFF_BASE*med(ds)))
    # Chosen tuning: lambda* = centre of the flat KS valley on the ks<0.20 set.
    print('\nCHOSEN: lambda = 300 cm, vuv_eff = %.4f  (direct-scale %.3f at lambda=300)'%(
        EFF_BASE*med([row_at(r,300)[4] for r in R if r['ks']<0.20]),
        med([row_at(r,300)[4] for r in R if r['ks']<0.20])))

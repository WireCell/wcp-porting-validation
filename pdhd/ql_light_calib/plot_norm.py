#!/usr/bin/env python3
"""Normalization-study figures from REAL QLMatching calib-mode pred_pe (not the
offline re-predictor): measured vs predicted BEFORE (lambda=2000) vs AFTER
(lambda=100), for the 7 clean crosser anchors.

Inputs = two sets of -calib dumps produced by run_clus_evt.sh -calib at each model:
  BEFORE dir: dumps with vuv_absorption_length=2000, vuv_eff=0.03
  AFTER  dir: dumps with vuv_absorption_length=100,  vuv_eff=0.023
Outputs (../pics/): ql_norm_perpmt.png, ql_norm_concentration.png,
ql_norm_crosser_patterns.png."""
import json, glob, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BEFORE='/home/xqian/tmp/ql_plots/before'
AFTER ='/home/xqian/tmp/ql_plots/after'
PICS  ='/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/pics'
STATIC=set([3,86,87,97,107,116,117])|set(range(120,160))

def n90(v):
    s=sum(v)
    if s<=0: return 0
    c=0;n=0
    for x in sorted(v,reverse=True):
        c+=x;n+=1
        if c>=0.9*s: break
    return n

def find_clean(dump):
    """clean anchors in an AFTER dump: at_x_boundary, ndf>=30, bright>1000,
    measTot>3000, ks<0.4, dedup by main keeping min ks."""
    d=json.load(open(dump))
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    flB={f['gid']:f for f in d['flashes']}
    best={}
    for b in d['bundles']:
        if not b.get('at_x_boundary') or b['ndf']<30 or b['ks_dis']>0.40: continue
        f=flB.get(b['flash_gid'])
        if f is None or max(f["pe"][i] for i in valid)<=300: continue
        if sum(f['pe'][i] for i in valid)<3000: continue
        k=b['main_cluster']
        if k not in best or b['ks_dis']<best[k]['ks_dis']: best[k]=b
    out=[]
    for b in best.values():
        f=flB[b['flash_gid']]
        out.append(dict(main=b['main_cluster']%1000000, meas=f['pe'][:], valid=valid,
                        measTot=sum(f['pe'][i] for i in valid), opdets=d['opdets']))
    return out

def pred_before(bdump, main_ident, measTot):
    """pred_pe for the same crosser cluster + same physical flash in the BEFORE dump."""
    d=json.load(open(bdump))
    flB={f['gid']:f for f in d['flashes']}
    # the physical flash = same measured total (reco identical across lambda)
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    cand=[b for b in d['bundles'] if b['main_cluster']%1000000==main_ident]
    best=None
    for b in cand:
        f=flB.get(b['flash_gid'])
        if f is None: continue
        if abs(sum(f['pe'][i] for i in valid)-measTot)<1.0:
            if best is None or b['ks_dis']<best['ks_dis']: best=b
    if best is None and cand:  # fallback: closest flash total
        best=min(cand,key=lambda b: abs(sum(flB[b['flash_gid']]['pe'][i] for i in valid)-measTot)
                 if b['flash_gid'] in flB else 1e9)
    return best['pred_pe'] if best else [0.0]*160

# ---- assemble the 7 anchors with measured / before / after ----
EVMAP={'150':'0','158':'2','162':'3','198':'12','270':'30','282':'33'}
anchors=[]
for af in sorted(glob.glob(AFTER+'/*/calib-*-group*.json')):
    art=af.split('/')[-1].split('-')[1]; grp=af.split('-')[-1].split('.')[0]
    ev=af.split('/')[-2]
    d=json.load(open(af))
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    # re-find clean anchors and their after pred
    flB={f['gid']:f for f in d['flashes']}
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    best={}
    for b in d['bundles']:
        if not b.get('at_x_boundary') or b['ndf']<30 or b['ks_dis']>0.40: continue
        f=flB.get(b['flash_gid'])
        if f is None or max(f["pe"][i] for i in valid)<=300: continue
        if sum(f['pe'][i] for i in valid)<3000: continue
        k=b['main_cluster']
        if k not in best or b['ks_dis']<best[k]['ks_dis']: best[k]=b
    for b in best.values():
        f=flB[b['flash_gid']]
        mid=b['main_cluster']%1000000; mt=sum(f['pe'][i] for i in valid)
        bdump=BEFORE+'/%s/calib-%s-%s.json'%(ev,art,grp)
        anchors.append(dict(art=art,grp=grp,main=mid,valid=valid,opdets=d['opdets'],
                            meas=f['pe'][:],after=b['pred_pe'][:],
                            before=pred_before(bdump,mid,mt),measTot=mt))
anchors.sort(key=lambda a:-a['measTot'])
print('anchors: %d  ->'%len(anchors),[(a['art'],round(a['measTot'])) for a in anchors])
N=len(anchors)

def series(a):
    v=a['valid']
    return ([a['meas'][i] for i in v],[a['before'][i] for i in v],[a['after'][i] for i in v])

# ============ FIG 1: 1D per-PMT PE comparison (channels sorted by measured) ====
fig,axes=plt.subplots((N+1)//2,2,figsize=(13,2.6*((N+1)//2)),squeeze=False)
axf=axes.flatten()
for ax,a in zip(axf,anchors):
    m,bef,aft=series(a)
    order=sorted(range(len(m)),key=lambda k:-m[k])
    x=range(len(order))
    ax.plot(x,[m[k] for k in order],  '-o',ms=3,lw=1.6,color='k',     label='measured')
    ax.plot(x,[bef[k] for k in order],'-s',ms=3,lw=1.2,color='tab:red',  label='pred λ=2000 (before)')
    ax.plot(x,[aft[k] for k in order],'-^',ms=3,lw=1.2,color='tab:green',label='pred λ=100 (after)')
    ax.set_yscale('symlog',linthresh=1)
    ax.set_title('%s/%s  meas %.0f PE  (N90 m=%d b=%d a=%d)'%(
        a['art'],a['grp'],a['measTot'],n90(m),n90(bef),n90(aft)),fontsize=8)
    ax.set_xlabel('PMT (sorted by measured PE)',fontsize=8); ax.set_ylabel('PE',fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5)
    ax.set_xlim(0,min(45,len(order)))
for ax in axf[N:]: ax.axis('off')
fig.suptitle('Per-PMT PE: measured vs predicted before (λ=2000) vs after (λ=100) — 7 clean crossers\n'
             'after pulls the bright PMTs up to the measured level and the long over-predicted tail down',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(PICS+'/ql_norm_perpmt.png',dpi=110); print('wrote ql_norm_perpmt.png')

# ============ FIG 2: concentration (cumulative light vs PMT rank) =============
fig,axes=plt.subplots((N+1)//2,2,figsize=(13,2.6*((N+1)//2)),squeeze=False)
axf=axes.flatten()
for ax,a in zip(axf,anchors):
    m,bef,aft=series(a)
    for vals,lab,sty,col in [(m,'measured','-','k'),(bef,'before λ=2000','--','tab:red'),
                             (aft,'after λ=100',':','tab:green')]:
        s=sum(vals)
        if s<=0: continue
        cum=np.cumsum(sorted(vals,reverse=True))/s
        ax.plot(range(1,len(cum)+1),cum,sty,lw=2,color=col,label='%s (N90=%d)'%(lab,n90(vals)))
    ax.axhline(0.9,color='grey',ls=':',lw=0.8)
    ax.set_title('%s/%s  (meas %.0f PE)'%(a['art'],a['grp'],a['measTot']),fontsize=8)
    ax.set_xlabel('PMT rank',fontsize=8); ax.set_xlim(0,30); ax.set_ylim(0,1.02)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5,loc='lower right')
    ax.set_ylabel('cumulative fraction',fontsize=8)
for ax in axf[N:]: ax.axis('off')
fig.suptitle('Predicted light concentration: before vs after vs measured (steeper = more concentrated)',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(PICS+'/ql_norm_concentration.png',dpi=110); print('wrote ql_norm_concentration.png')

# ============ FIG 3: per-crosser 2D PMT maps (y vs z), shape only ============
fig,axes=plt.subplots(N,3,figsize=(11,2.5*N),squeeze=False)
cols=['measured','pred λ=2000 (before)','pred λ=100 (after)']
for r,a in enumerate(anchors):
    v=a['valid']; ys=[a['opdets'][i]['y'] for i in v]; zs=[a['opdets'][i]['z'] for i in v]
    m,bef,aft=series(a)
    for c,(vals,ct) in enumerate(zip([m,bef,aft],cols)):
        ax=axes[r][c]; s=sum(vals) or 1.0; frac=np.array(vals)/s
        ax.scatter(zs,ys,c='lightgrey',s=6,zorder=1)
        mk=frac>0; fmax=frac.max() or 1
        ax.scatter(np.array(zs)[mk],np.array(ys)[mk],c=frac[mk],s=90*np.sqrt(frac[mk]/fmax)+6,
                   cmap='viridis',vmin=0,vmax=fmax,zorder=2)
        if r==0: ax.set_title(ct,fontsize=9)
        if c==0: ax.set_ylabel('%s/%s\nN90=%d  y(cm)'%(a['art'],a['grp'],n90(vals)),fontsize=7)
        else: ax.set_ylabel('N90=%d'%n90(vals),fontsize=7)
        if r==N-1: ax.set_xlabel('z (cm)',fontsize=8)
        ax.tick_params(labelsize=6)
fig.suptitle('Per-crosser PMT pattern (y vs z), colour=normalized PE (shape). After (λ=100) localises toward measured.',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(PICS+'/ql_norm_crosser_patterns.png',dpi=110); print('wrote ql_norm_crosser_patterns.png')

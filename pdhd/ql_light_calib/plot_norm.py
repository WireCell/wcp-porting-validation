#!/usr/bin/env python3
"""Before/after normalization-study figures (run 29107) from REAL QLMatching
calib-mode pred_pe: measured vs predicted BEFORE (lambda=100, vuv_eff=0.023, the
old 27305 tuning) vs AFTER (lambda=300, vuv_eff=0.0145, the 29107 tuning).

BEFORE = backed-up production dumps; AFTER = anchor events reprocessed at the new
model.  Outputs (../pics/): ql_norm_perpmt.png, ql_norm_concentration.png,
ql_norm_crosser_patterns.png."""
import json, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BEFORE='/home/xqian/tmp/before29107'
AFTER ='/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work'
PICS  ='/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/pics'
STATIC=set([3,86,87,97,107,116,117])|set(range(120,160))
RUN=29107

def load(path):
    d=json.load(open(path))
    gk=list(d['geometry'].keys())[0]; gr=d['geometry'][gk]
    valid=[i for i in range(160) if i not in STATIC and not d['opdets'][i]['auto_masked']
           and (d['opdets'][i]['x']>0)==(gr['anode_x']>0)]
    flB={f['gid']:f for f in d['flashes']}
    return d,flB,valid

def clean_anchors(d,flB,valid):
    byflash={}
    for b in d['bundles']:
        if not b.get('at_x_boundary') or not b.get('two_boundary') or b['ndf']<30: continue
        f=flB.get(b['flash_gid'])
        if f is None: continue
        meas=[f['pe'][i] for i in valid]
        if max(meas)<=300 or sum(meas)<=3000 or b['ks_dis']>0.20: continue
        k=b['flash_gid']
        if k not in byflash or b['ks_dis']<byflash[k]['ks_dis']: byflash[k]=b
    return byflash

anchors=[]
for ev in range(30):
    for grp in ('group02','group13'):
        ga=glob.glob('%s/0%d_%d/calib-*-%s.json'%(AFTER,RUN,ev,grp))
        gb=glob.glob('%s/0%d_%d/calib-*-%s.json'%(BEFORE,RUN,ev,grp))
        if not ga or not gb: continue
        da,flA,valid=load(ga[0]); db,flB,_=load(gb[0])
        art=ga[0].split('/')[-1].split('-')[1]
        bb_by_flash={}
        for b in db['bundles']:
            bb_by_flash.setdefault(b['flash_gid'],[]).append(b)
        for fg,ba in clean_anchors(da,flA,valid).items():
            cand=bb_by_flash.get(fg,[])
            if not cand: continue
            same=[b for b in cand if b['main_cluster']==ba['main_cluster']]
            bb=(same or sorted(cand,key=lambda b:b['ks_dis']))[0]
            f=flA[fg]; meas=np.array([f['pe'][i] for i in valid])
            pa=np.array([ba['pred_pe'][i] for i in valid])
            pb=np.array([bb['pred_pe'][i] for i in valid])
            anchors.append(dict(art=art,grp=grp,meas=meas,pa=pa,pb=pb,
                ya=np.array([da['opdets'][i]['y'] for i in valid]),
                za=np.array([da['opdets'][i]['z'] for i in valid]),
                ksA=ba['ks_dis'],ksB=bb['ks_dis'],measTot=meas.sum()))
print('matched before/after anchors: %d'%len(anchors))
top=sorted(anchors,key=lambda a:-a['measTot'])[:9]

# FIG 1: per-PMT PE (sorted by measured, symlog)
fig,axs=plt.subplots(3,3,figsize=(15,11)); axs=axs.ravel()
for ax,a in zip(axs,top):
    order=np.argsort(-a['meas']); x=np.arange(len(order))
    ax.plot(x,a['meas'][order],'-o',ms=3,color='k',label='measured')
    ax.plot(x,a['pb'][order],'-s',ms=3,color='tab:red',alpha=.8,label='before λ=100')
    ax.plot(x,a['pa'][order],'-^',ms=3,color='tab:green',alpha=.8,label='after λ=300')
    ax.set_yscale('symlog',linthresh=1)
    ax.set_title('%s %s  ks %.2f→%.2f'%(a['art'],a['grp'],a['ksB'],a['ksA']),fontsize=9)
    ax.set_xlabel('PMT rank (by measured)',fontsize=8); ax.set_ylabel('PE',fontsize=8)
    ax.legend(fontsize=7)
for ax in axs[len(top):]: ax.axis('off')
fig.suptitle('PDHD run 29107 two-boundary crossers: per-PMT PE, measured vs predicted (before λ=100 / after λ=300)')
fig.tight_layout(); fig.savefig('%s/ql_norm_perpmt.png'%PICS,dpi=90); plt.close(fig)

# FIG 2: cumulative concentration
fig,ax=plt.subplots(figsize=(7,6))
for a in anchors:
    for arr,c in ((a['meas'],'k'),(a['pb'],'tab:red'),(a['pa'],'tab:green')):
        s=np.sort(arr)[::-1]; cu=np.cumsum(s)
        if cu[-1]<=0: continue
        ax.plot(np.arange(1,len(s)+1),cu/cu[-1],color=c,alpha=.22,lw=1)
for c,lab in (('k','measured'),('tab:red','before λ=100'),('tab:green','after λ=300')):
    ax.plot([],[],color=c,label=lab)
ax.axhline(0.9,ls=':',color='gray'); ax.set_xlim(0,60)
ax.set_xlabel('PMT rank'); ax.set_ylabel('cumulative fraction of light')
ax.set_title('Light concentration, %d two-boundary crossers (run 29107)'%len(anchors))
ax.legend(); fig.tight_layout(); fig.savefig('%s/ql_norm_concentration.png'%PICS,dpi=90); plt.close(fig)

# FIG 3: 2-D y-z PMT maps (measured | before | after)
nshow=min(6,len(top))
fig,axs=plt.subplots(nshow,3,figsize=(11,2.2*nshow))
if nshow==1: axs=axs[None,:]
for r in range(nshow):
    a=top[r]
    for c,(arr,title) in enumerate(((a['meas'],'measured'),(a['pb'],'before λ=100'),(a['pa'],'after λ=300'))):
        ax=axs[r,c]; norm=arr/arr.max() if arr.max()>0 else arr
        ax.scatter(a['za'],a['ya'],c=norm,s=60,cmap='viridis',vmin=0,vmax=1,marker='s')
        if r==0: ax.set_title(title)
        if c==0: ax.set_ylabel('%s %s'%(a['art'],a['grp']),fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Normalised PMT pattern (y vs z), run 29107 two-boundary crossers')
fig.tight_layout(); fig.savefig('%s/ql_norm_crosser_patterns.png'%PICS,dpi=90); plt.close(fig)
print('wrote 3 figures to %s'%PICS)

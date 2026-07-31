import uproot, numpy as np

# --- segment 5030 data from the ON-arm tracking root ---
f = uproot.open("/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/work-nuecc48-prsmoke2/nupr_evt172230_mipvote_on/tracking-pr.root")
a = f["T_rec_charge"].arrays(library="np")
m = a["real_cluster_id"] == 5030
x,y,z,q,nq = (a[k][m] for k in ("x","y","z","q","nq"))
dqdx = (q+1000.0)*10.0/nq          # e/cm  (nq holds dx in cm in this fork)
pts = np.stack([x,y,z],axis=1)
L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))])  # cm

# --- SBND templates (particle_dataset.jsonnet, start=0.5 step=1.0) ---
muon_v = [168151,118771,103292,94371,88307.2,83853,80495.8,77603.1,75317,73308.4,71636.5,70151.1,68864.5,67719.2,66707.8,65786.6,64970.8,64209.7,63535.9,62893.8,62330.4,61786.6,61311.3,60857.5,60442.2,60055,59685.5,59351.1,59023.4,58723.8,58432.7,58167.8,57927.9,57693.4,57478,57268.3,57115.7,57007.3,56899.1,56791,56683.1,56575.2,56467.6,56360.1,56252.6,56145.4,56038.3,55931.3,55824.4,55717.7,55611.1,55504.6,55398.3,55292.1,55186.1,55080.2,54974.3,54868.7,54763.1,54657.7]
proton_v = [261588,200147,178238,164708,155050,147607,141597,136584,132684,129187,125736,122447,119992,117710,115452,113226,111412,109817,108234,106663,105130,103879,102708,101544,100388,99240.5,98242.7,97352.9,96467.7,95587.1,94711,93850.8,93117.4,92420.7,91726.9,91036.2,90348.5,89667.3,89072.9,88515.5,87960.1,87406.8,86855.5,86306.3,85784.4,85326.7,84873,84420.9,83970.1,83520.8,83072.9,82665.6,82313.6,81962.9,81613,81264.1,80916.1,80569,80222.7,79877.4]
elec_v = [52137.8,52137.8,55095.7,57364.7,59109.2,60597.8,61904.9,63075,64104.6,65027.7,65851.8,66589.5,67250.1,67842.4,68370.4]+[68633.5]*45

def linterp(vals, xq, start=0.5, step=1.0):
    vals=np.asarray(vals); i=(xq-start)/step
    return np.interp(i, np.arange(len(vals)), vals)   # np.interp clamps at both ends

def kslike(test, ref):
    t=np.cumsum(test/np.sum(test)); r=np.cumsum(ref/np.sum(ref))
    return np.max(np.abs(t-r))

def eval_ks_ratio(ks1,ks2,r1,r2):
    if ks1-ks2 >= 0.0: return False
    if np.sqrt((ks2/0.06)**2+((r2-1)/0.06)**2)<1.4 and ks1-ks2+(abs(r1-1)-abs(r2-1))/1.5*0.3 > -0.02: return False
    if ks1-ks2 < -0.02 and ((ks2>0.09 and abs(r2-1)>0.1) or r2>1.5 or ks2>0.2): return True
    if ks1-ks2+(abs(r1-1)-abs(r2-1))/1.5*0.3 < 0: return True
    return False

def track_comp(L, dQdx, cr, off, mip):
    endL = L[-1] + 0.15 - off
    sel = (endL-L < cr) & (endL-L > 0)
    vx, vy = endL-L[sel], dQdx[sel]
    if len(vx)==0: return None
    mu, pr, el = linterp(muon_v,vx), linterp(proton_v,vx), linterp(elec_v,vx)
    fl = np.full(len(vx), mip)
    out={}
    for nm,ref in (("mu",mu),("flat",fl),("p",pr),("e",el)):
        ks=kslike(vy,ref); rat=np.sum(ref)/(np.sum(vy)+1e-9)
        out[nm]=(ks,rat)
    out["dirbit"]=eval_ks_ratio(out["mu"][0],out["flat"][0],out["mu"][1],out["flat"][1])
    out["score"]={nm: np.sqrt(out[nm][0]**2+(out[nm][1]-1)**2) for nm in ("mu","p","e")}
    out["n"]=len(vx)
    return out

def track_pid(L,dQdx,cr,off,mip,length_cm):
    fw = track_comp(L,dQdx,cr,off,mip)
    rL = L[-1]-L[::-1]; rd = dQdx[::-1]
    bw = track_comp(rL,rd,cr,off,mip)
    def best(r):
        t,v=13,r["score"]["mu"]
        if r["score"]["p"]<v: t,v=2212,r["score"]["p"]
        if r["score"]["e"]<v and length_cm<20: t,v=11,r["score"]["e"]
        return t,v
    ft,fv = best(fw); bt,bv = best(bw)
    dec = ("ABSTAIN",0,0,100.0)
    if fw["dirbit"] and not bw["dirbit"]: dec=("DIR",1,ft,fv)
    elif bw["dirbit"] and not fw["dirbit"]: dec=("DIR",-1,bt,bv)
    elif fw["dirbit"] and bw["dirbit"]: dec=("DIR",1,ft,fv) if fv<bv else ("DIR",-1,bt,bv)
    return fw,bw,dec,(ft,fv),(bt,bv)

length = L[-1]
print(f"npts={len(L)} length={length:.2f}cm  dQdx tip={dqdx[-1]:.0f} peak={dqdx[-3]:.0f}")
for mip in (56000,):
  for off in (0.0, 0.5, 1.0, 2.0, 3.0):
    for cr in (35.0, 15.0):
        fw,bw,dec,fbest,bbest = track_pid(L,dqdx,cr,off,mip,length)
        print(f"off={off:.1f} cr={cr:.0f} | fwd: n={fw['n']} ks_mu={fw['mu'][0]:.4f} ks_flat={fw['flat'][0]:.4f} ks_p={fw['p'][0]:.4f} r_p={fw['p'][1]:.3f} dir={int(fw['dirbit'])} s_mu={fw['score']['mu']:.3f} s_p={fw['score']['p']:.3f} s_e={fw['score']['e']:.3f}"
              f" | bwd: ks_mu={bw['mu'][0]:.4f} ks_flat={bw['flat'][0]:.4f} dir={int(bw['dirbit'])} s_p={bw['score']['p']:.3f}"
              f" | DECISION={dec[0]} dir={dec[1]} type={dec[2]} score={dec[3]:.3f}")

print("\n--- trim-only variants: drop last N points from the comparison, template zero stays at tip+0.15 ---")
def track_comp_trim(L, dQdx, cr, mip, ntrim):
    endL = L[-1] + 0.15
    sel = (endL-L < cr) & (endL-L > 0)
    vx, vy = endL-L[sel], dQdx[sel]
    if ntrim>0: vx, vy = vx[:-0], vy  # placeholder
    return None

def track_pid_trim(L,dQdx,cr,mip,length_cm,ntrim):
    # forward: drop the last ntrim samples (nearest the tip) from BOTH data and template eval
    def comp(Lv, dv):
        endL = Lv[-1] + 0.15
        sel = (endL-Lv < cr) & (endL-Lv > 0)
        vx, vy = endL-Lv[sel], dv[sel]
        # samples are ordered along Lv; the tip samples are the LAST ones (smallest residual)
        if ntrim>0:
            keep = vx > (0.15 + (Lv[-1]-Lv[-1-ntrim]) - 1e-6)
            vx, vy = vx[keep], vy[keep]
        mu, pr, el = linterp(muon_v,vx), linterp(proton_v,vx), linterp(elec_v,vx)
        fl = np.full(len(vx), mip)
        out={}
        for nm,ref in (("mu",mu),("flat",fl),("p",pr),("e",el)):
            ks=kslike(vy,ref); rat=np.sum(ref)/(np.sum(vy)+1e-9)
            out[nm]=(ks,rat)
        out["dirbit"]=eval_ks_ratio(out["mu"][0],out["flat"][0],out["mu"][1],out["flat"][1])
        out["score"]={nm: np.sqrt(out[nm][0]**2+(out[nm][1]-1)**2) for nm in ("mu","p","e")}
        out["n"]=len(vx)
        return out
    fw = comp(L,dqdx)
    rL = L[-1]-L[::-1]; rd = dQdx[::-1]
    bw = comp(rL,rd)
    def best(r):
        t,v=13,r["score"]["mu"]
        if r["score"]["p"]<v: t,v=2212,r["score"]["p"]
        if r["score"]["e"]<v and length_cm<20: t,v=11,r["score"]["e"]
        return t,v
    ft,fv=best(fw); bt,bv=best(bw)
    dec=("ABSTAIN",0,0,100.0)
    if fw["dirbit"] and not bw["dirbit"]: dec=("DIR",1,ft,fv)
    elif bw["dirbit"] and not fw["dirbit"]: dec=("DIR",-1,bt,bv)
    elif fw["dirbit"] and bw["dirbit"]: dec=("DIR",1,ft,fv) if fv<bv else ("DIR",-1,bt,bv)
    return fw,bw,dec

for ntrim in (1,2):
    fw,bw,dec = track_pid_trim(L,dqdx,35.0,56000,length,ntrim)
    print(f"trim={ntrim} | fwd: n={fw['n']} ks_mu={fw['mu'][0]:.4f} ks_flat={fw['flat'][0]:.4f} dir={int(fw['dirbit'])} s_mu={fw['score']['mu']:.3f} s_p={fw['score']['p']:.3f} s_e={fw['score']['e']:.3f}"
          f" | bwd: ks_mu={bw['mu'][0]:.4f} ks_flat={bw['flat'][0]:.4f} dir={int(bw['dirbit'])} s_p={bw['score']['p']:.3f}"
          f" | DECISION={dec[0]} dir={dec[1]} type={dec[2]} score={dec[3]:.3f}")

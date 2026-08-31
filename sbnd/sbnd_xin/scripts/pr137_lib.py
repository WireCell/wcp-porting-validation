#!/usr/bin/env python3
# doc pr/137 sec 10-14 -- shared loaders + feature kernel for the split-TRIGGER round.
# READ-ONLY.  Nothing in this module writes to disk.
"""Shared library for the pr/137 trigger bake-off.

WHY THIS FILE EXISTS.  The five pr137_split_*.py probes copy-pasted their
loaders, and the copies drifted: pr137_split_feasibility.py reads geometry from
the off2 arm while the other four read onV1c90, and four of the five truncate
their point lists ([:200]/[:400]) inside an O(N^2) min-distance loop so the
published gaps are approximations rather than minima (doc pr/137 sec 1d).  One
loader, one gap routine, one population definition.

THE ONE STRUCTURAL FACT EVERYTHING HERE OBEYS (doc pr/137 sec 1b):
geometry and charge are per POINT; shower membership is per SEGMENT.  The dump
gives segments[].points[] = {x,y,z,dQ,dx,rr}; the sidecar gives
showers{}.members[] = {seg, dQ, ...}.  There is no per-point shower assignment
anywhere in the chain, and C++ agrees -- Shower::detach_member_set takes a set
of SEGMENTS (PRShower.cxx:640-700).  So: seed at point level, assign at segment
level.  Any function here that returns a partition returns a segment partition.

Run from the sbnd_xin working root (paths are cwd-relative, as in pr/137).
"""
import json, glob, os, math, collections
import numpy as np

CM = 1.0                      # dump positions are already in cm
X0_LAR   = 14.0               # cm, radiation length in liquid argon (scale only)
RM_LAR   = 10.0               # cm, Moliere radius (scale only -- NOT used as a
                              # threshold anywhere; the width scale is fitted
                              # in-situ by pr137_null_model.py)
CONV_LAR = 9.0/7.0*X0_LAR     # cm, mean gamma conversion length ~18 cm

# ---------------------------------------------------------------- loaders

def prep_dir(tag):
    return os.path.join('em_display', tag)

def prep_events(tag):
    """event ids present in a sidecar prepdir"""
    out=[]
    for f in glob.glob(os.path.join(prep_dir(tag), 'emprep-evt*.json')):
        b=os.path.basename(f)
        try: out.append(int(b[len('emprep-evt'):-len('.json')]))
        except ValueError: pass
    return sorted(out)

def prep(tag, ev):
    """sidecar membership: {shower_node_id: {seg_id: dQ}}.

    Keyed on node_id (= cluster_id*1000 + segment index), NEVER on shower_id --
    shower_id is a per-process sequential counter and is not stable across runs
    (prep_em_scan.py:363-367)."""
    p=os.path.join(prep_dir(tag), 'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    j=json.load(open(p))
    return {int(n): {int(m['seg']): float(m.get('dQ') or 0.0)
                     for m in (e.get('members') or [])}
            for n,e in (j.get('showers') or {}).items()}

def prep_raw(tag, ev):
    """the whole sidecar record, for start/dir/kine fields"""
    p=os.path.join(prep_dir(tag), 'emprep-evt%d.json'%ev)
    if not os.path.exists(p): return None
    return json.load(open(p))

def dump(ev, arm):
    """the calib PR dump for one event of one arm"""
    for a in sorted(glob.glob('work-pr136-%s-*/pr_evt%d/calib-pr-evt%d.json'%(arm,ev,ev))):
        return json.load(open(a))
    return None

def seg_pts(d):
    """{seg_id: ndarray[N,5] of (x, y, z, dQ, dx)} -- per-point geometry+charge.

    dx is kept so dE/dx is computable; pr/137's probes dropped it."""
    out={}
    for s in d.get('segments') or ():
        pts=[(p['x'], p['y'], p['z'], p.get('dQ') or 0.0, p.get('dx') or 0.0)
             for p in (s.get('points') or ())]
        if pts: out[int(s['id'])]=np.asarray(pts, dtype=float)
    return out

def seg_meta(d):
    """{seg_id: {...}} -- flag_shower, length, cluster_id, particle_id, shower_id"""
    out={}
    for s in d.get('segments') or ():
        out[int(s['id'])]=dict(flag_shower=bool(s.get('flag_shower')),
                               length=float(s.get('length') or 0.0),
                               cluster_id=int(s.get('cluster_id') or -1),
                               particle_id=int(s.get('particle_id') or 0),
                               shower_id=int(s.get('shower_id') or -1))
    return out

def main_vertex(d):
    mv=d.get('main_vertex')
    if isinstance(mv, dict):
        for k in ('fit','point','pos'):
            if k in mv and mv[k]: return np.asarray(mv[k], dtype=float)
        if all(c in mv for c in 'xyz'): return np.asarray([mv['x'],mv['y'],mv['z']], float)
    if isinstance(mv,(list,tuple)) and len(mv)==3: return np.asarray(mv, float)
    for v in (d.get('vertices') or ()):
        if v.get('is_main') and v.get('fit'): return np.asarray(v['fit'], float)
    return None

def shower_recs(d):
    """{node_id: shower record} from the dump's showers[] block"""
    return {int(s['id']): s for s in (d.get('showers') or ()) if 'id' in s}

# ---------------------------------------------------------------- geometry

def pack(P, segs):
    """stack the per-point arrays of a segment list -> (pts[N,3], q[N], dx[N])"""
    arrs=[P[s] for s in segs if s in P and len(P[s])]
    if not arrs: return None, None, None
    A=np.concatenate(arrs, axis=0)
    return A[:,:3], A[:,3], A[:,4]

def min_gap(P, segsA, segsB):
    """EXACT minimum 3-D distance between two segment groups (cm).

    pr/137 sec 1d: the probes truncated to [:200]/[:400] inside an O(N^2) loop.
    cKDTree makes the exact answer cheaper than the approximation was."""
    from scipy.spatial import cKDTree
    A,_,_ = pack(P, segsA); B,_,_ = pack(P, segsB)
    if A is None or B is None: return float('nan')
    d,_ = cKDTree(A).query(B, k=1)
    return float(d.min())

def rays(pts, v):
    """unit rays from reference point v, and depths r"""
    u = pts - v[None,:]
    r = np.linalg.norm(u, axis=1)
    r = np.where(r<=0, 1e-9, r)
    return u/r[:,None], r

def qwt(q):
    """charge weights, clamped at zero.

    dQ in the dump can be NEGATIVE (noise-subtracted charge); an unclamped
    weighted variance then goes negative and sqrt() raises.  Every weighted
    statistic in this module goes through here."""
    w = np.clip(np.asarray(q, float), 0.0, None)
    return w if w.sum() > 0 else np.ones(len(w))

def qw_centroid(pts, q):
    q = qwt(q)
    w = q.sum()
    if w<=0: return pts.mean(axis=0)
    return (pts*q[:,None]).sum(axis=0)/w

def transverse_rms(pts, q, v, axis=None):
    """charge-weighted RMS distance of points from the ray v->centroid (cm).

    This is 'how wide is this object', the Family-S primitive.  The prototype's
    tuned analogue is ProtoSegment::is_shower_topology (ProtoSegment.cxx:319-450),
    which calls 0.4 cm 'large spread' and decides at 0.7/0.8/1.0 cm."""
    if len(pts)<2: return 0.0
    if axis is None:
        c = qw_centroid(pts,q); axis = c-v
    n = np.linalg.norm(axis)
    if n<=0: return 0.0
    a = axis/n
    d = pts - v[None,:]
    perp = d - np.outer(d@a, a)
    t = np.einsum('ij,ij->i', perp, perp)
    w = qwt(q)
    return float(math.sqrt(max(0.0,(t*w).sum()/w.sum())))

def width_profile(pts, q, v, nbin=6):
    """(r_mid[], w[]) -- transverse RMS in depth bins along the object's own ray.

    Owner factor 2: 'for a single shower we expect the growth of the shower to be
    bigger as the distance going further'.  A monotone-rising w(r) is a single
    shower; a step or a too-wide bin is the merged signature."""
    if len(pts)<8: return np.array([]), np.array([])
    c = qw_centroid(pts,q); axis = c-v
    n = np.linalg.norm(axis)
    if n<=0: return np.array([]), np.array([])
    a = axis/n
    d = pts - v[None,:]
    t = d@a                                  # depth along the axis
    perp = d - np.outer(t, a)
    rho = np.linalg.norm(perp, axis=1)       # transverse distance
    lo,hi = np.percentile(t,[2,98])
    if hi<=lo: return np.array([]), np.array([])
    edges = np.linspace(lo,hi,nbin+1)
    idx = np.clip(np.digitize(t,edges)-1, 0, nbin-1)
    rm=[]; wm=[]
    w = qwt(q)
    for b in range(nbin):
        m = idx==b
        if m.sum()<4: continue
        ww=w[m]
        if ww.sum()<=0: continue
        rm.append(float((t[m]*ww).sum()/ww.sum()))
        wm.append(float(math.sqrt(max(0.0,((rho[m]**2)*ww).sum()/ww.sum()))))
    return np.asarray(rm), np.asarray(wm)

def mip_dqdx(d):
    """the MIP dQ/dx reference for this event, from the dump's dqdx_ref table.

    dqdx_ref["electron"] is dQ/dx vs residual range; its plateau (the small-rr
    end, before the Bragg rise) is the MIP value.  Normalising by it is what
    makes the 2xMIP photon-conversion signature testable: MicroBooNE separates
    electrons (2.1 MeV/cm) from photons (4.2 MeV/cm) on exactly this ratio."""
    t=(d.get('dqdx_ref') or {}).get('electron')
    if not t: return None
    a=np.asarray(t, float)
    return float(np.median(a[:max(3,len(a)//10)]))

def start_dedx(pts, q, dx, v, span=3.0):
    """median dE/dx-proxy (dQ/dx) over the first `span` cm from the part's start.

    Owner factor 3/4 and the LArTPC handle no calorimeter algorithm has: a gamma
    converts to e+e- and deposits ~2x MIP at the shower start (MicroBooNE: MIP
    2.1 MeV/cm, photon conversion 4.2 MeV/cm).  Two distinct 2-MIP stubs = two
    conversions, and that survives where theta-phi bimodality dies."""
    if len(pts)==0: return float('nan')
    _, r = rays(pts, v)
    r0 = r.min()
    m = (r <= r0+span) & (dx>0) & (q>0)
    if m.sum()<3: return float('nan')
    return float(np.median(q[m]/dx[m]))

def vertex_gap(pts, v):
    if len(pts)==0: return float('nan')
    return float(np.linalg.norm(pts-v[None,:], axis=1).min())

def void_frac(pts, q, v, nstep=20):
    """fraction of the v -> part-start ray that carries no charge nearby.

    A real second gamma has an EMPTY conversion gap; an artificial cut through
    one shower does not."""
    if len(pts)<4: return float('nan')
    c = qw_centroid(pts,q)
    a = c-v; n=np.linalg.norm(a)
    if n<=0: return float('nan')
    a=a/n
    start = vertex_gap(pts, v)
    if start<=1.0: return 0.0
    from scipy.spatial import cKDTree
    T=cKDTree(pts)
    ts = np.linspace(0.5, start-0.5, nstep)
    probe = v[None,:] + np.outer(ts, a)
    d,_ = T.query(probe, k=1)
    return float((d>2.0).mean())

# ---------------------------------------------------------------- population

def build_population(ev, off_tag='emprep-136off2', on_tag='emprep-136onV1c90',
                     arm='onV1c90', q_floor=1e6, nseg_floor=3,
                     frac=0.05, qmin=1e5):
    """The doc pr/137 sec 2 population, rebuilt once so every script agrees.

    For each ON shower, count distinct OFF showers contributing > `frac` of its
    charge and > `qmin`.  >=2 ancestors => MERGED, ==1 => SINGLE, 0 => unlabelled.
    Returns list of dicts.  Geometry always comes from `arm` -- the arm whose
    showers are being split (pr/137 sec 1d: feasibility.py read off2 here)."""
    mo, mn = prep(off_tag, ev), prep(on_tag, ev)
    if mo is None or mn is None: return []
    d = dump(ev, arm)
    if d is None: return []
    v = main_vertex(d)
    if v is None: return []
    P = seg_pts(d); M = seg_meta(d); SR = shower_recs(d)
    owner={}
    for n,mem in mo.items():
        for s,qq in mem.items(): owner.setdefault(s,[]).append((n,qq))
    out=[]
    for n,mem in mn.items():
        segs=[s for s in mem if s in P]
        if len(segs)<nseg_floor: continue
        Q=sum(mem[s] for s in segs)
        if Q<q_floor: continue
        anc=collections.Counter()
        for s in segs:
            for (on_,qq) in owner.get(s,()):
                anc[on_]+=mem[s]
        big=[a for a,qq in anc.items() if qq>frac*Q and qq>qmin]
        cls = 'MERGED' if len(big)>=2 else ('SINGLE' if len(big)==1 else 'ORPHAN')
        out.append(dict(event=ev, node=n, segs=segs, mip=mip_dqdx(d), ms={s:mem[s] for s in segs},
                        Q=Q, cls=cls, nanc=len(big), anc=big,
                        truth={s: (max(owner.get(s,[(None,0)]), key=lambda t:t[1])[0]
                                   if owner.get(s) else None) for s in segs},
                        v=v, P=P, M=M, rec=SR.get(n)))
    return out

def dominant_other(pop_rows, row):
    """the 'nearby large EM shower' the owner's factors 2 and 3 normalise against.

    Rule (doc pr/137 sec 11): the largest OTHER EM object in the event above a
    charge floor; if none exists, the caller must fall back to the in-situ
    w_single(r) model and FLAG the row -- otherwise the owner's stated
    normalisation silently becomes the agent's in exactly the events where the
    candidate IS the biggest shower, which is most of them."""
    cands=[r for r in pop_rows if r['node']!=row['node'] and r['Q']>0.2*row['Q']]
    if not cands: return None
    return max(cands, key=lambda r: r['Q'])

# ---------------------------------------------------------------- seeding

def default_sigma_fn(w_scale=6.0, floor_deg=3.0, ceil_deg=45.0):
    """bandwidth(r) in RADIANS for the angular density.

    sigma(r) = w_scale / r  -- a transverse length divided by the depth.  This is
    the whole of owner factors 2+3 expressed as a kernel: a compact part that
    converted far away subtends a SMALL angle and must not be smoothed with the
    same kernel as a near, wide one.  w_scale is fitted in-situ by
    pr137_null_model.py; the LAr Moliere radius (10 cm) is quoted for scale only
    and is never used as a threshold."""
    lo=math.radians(floor_deg); hi=math.radians(ceil_deg)
    def f(r):
        s = w_scale/np.maximum(r, 1e-6)
        return np.clip(s, lo, hi)
    return f

def angular_seeds(pts, q, v, sigma_fn, f_prom=0.20, sep_scale=1.6,
                  max_seeds=4, max_pts=4000, rng=None):
    """Point-level charge-weighted angular density on the unit sphere -> seeds.

    This is the ATLAS/CMS/GARLIC shape (doc pr/137 sec 10): seed on local maxima
    and let the SEED COUNT be the multiplicity decision.  pr/137 sec 4 instead ran
    a 2-means that always fires and then hunted for an external veto, which is why
    its null distribution was broad.

    Acceptance follows the prototype's find_peak_point_indices
    (PR3DCluster_steiner.h:733): sort candidates by density descending and accept
    a peak only if no already-accepted peak lies within its neighbourhood -- here
    `sep_scale` bandwidths of angular separation.

    Returns (dirs[k,3], dens[k], valley) where valley is the density minimum along
    the arc between the two strongest seeds, relative to the weaker peak
    (ATLAS's local-maxima-with-a-valley); nan when k<2.
    """
    if len(pts)<8: return np.zeros((0,3)), np.zeros(0), float('nan')
    U, r = rays(pts, v)
    w = qwt(q)
    if len(U)>max_pts:                       # deterministic subsample
        rng = rng or np.random.default_rng(20260901)
        keep = rng.choice(len(U), max_pts, replace=False); keep.sort()
        U, r, w = U[keep], r[keep], w[keep]
    sig = sigma_fn(r)
    C = np.clip(U@U.T, -1.0, 1.0)
    ang = np.arccos(C)                        # [N,N] angular distances
    K = np.exp(-0.5*(ang/sig[None,:])**2)     # kernel width from the SOURCE point
    dens = K@w
    order = np.argsort(-dens)
    seeds=[]
    for i in order:
        if len(seeds)>=max_seeds: break
        if dens[i] < f_prom*dens[order[0]]: break
        if any(ang[i,j] < sep_scale*max(sig[i],sig[j]) for j in seeds): continue
        seeds.append(i)
    S = U[seeds]; D = dens[seeds]
    valley=float('nan')
    if len(seeds)>=2:
        a,b = U[seeds[0]], U[seeds[1]]
        ts = np.linspace(0,1,25)[:,None]
        arc = a[None,:]*(1-ts) + b[None,:]*ts
        arc /= np.linalg.norm(arc,axis=1)[:,None]
        Ca = np.clip(arc@U.T,-1,1)
        da = (np.exp(-0.5*(np.arccos(Ca)/sig[None,:])**2))@w
        valley = float(da.min()/max(D[1],1e-12))
    return S, D, valley

def assign_segments(P, segs, v, seed_dirs):
    """SEGMENT-level assignment to point-level seeds (doc pr/137 sec 1b).

    Each segment goes to the seed holding the plurality of its charge.  Returns
    (assignment {seg: k}, ambiguous_fraction) -- the ambiguous fraction is the
    charge share sitting in segments whose top-two seeds are within 10 %, i.e.
    the price of NOT having the literature's fractional assignment (we cannot
    share a segment without re-fitting a trajectory)."""
    k = len(seed_dirs)
    if k<2: return {s:0 for s in segs}, 0.0
    asg={}; amb=0.0; tot=0.0
    for s in segs:
        A=P.get(s)
        if A is None or not len(A): asg[s]=0; continue
        U,_ = rays(A[:,:3], v)
        w = qwt(A[:,3])
        sc = (U@seed_dirs.T)                  # [N,k] cosine to each seed
        best = np.argmax(sc, axis=1)
        share = np.array([w[best==j].sum() for j in range(k)])
        asg[s]=int(np.argmax(share))
        tot += share.sum()
        srt=np.sort(share)[::-1]
        if len(srt)>1 and srt[0]>0 and srt[1] > 0.9*srt[0]: amb += share.sum()
    return asg, (amb/tot if tot>0 else 0.0)

def profile_sigma_fn(w0=3.575, slope=0.0283, floor_deg=2.0, ceil_deg=60.0):
    """bandwidth(r) in RADIANS from the FITTED single-shower width profile.

    pr137_null_model.py measures w_single(r) = w0 + slope*r on the SINGLE
    population (owner factor 2, confirmed: the slope is positive).  The angular
    bandwidth a single shower subtends at depth r is then w_single(r)/r, which
    SHRINKS with depth -- so a compact, late-converting second gamma (owner
    factor 3) is not smoothed away by the kernel that fits the near, wide one."""
    lo=math.radians(floor_deg); hi=math.radians(ceil_deg)
    def f(r):
        r=np.maximum(np.asarray(r,float),1e-6)
        return np.clip((w0+slope*r)/r, lo, hi)
    return f

def angular_maxima(pts, q, v, sigma_fn, sep_scale=1.6, max_seeds=6,
                   max_pts=4000, rng=None):
    """All candidate angular maxima, with the pairwise VALLEY depth between them.

    Separates seed FINDING from seed ACCEPTANCE, which angular_seeds() conflated:
    the trigger is then a rule over (density ratio, valley) and the kernel can be
    scored with k forced to 2.

    Returns dict(dirs[k,3], dens[k], valley[k,k], frac[k]) where
      valley[i,j] = min density along the great-circle arc i->j, divided by the
                    weaker of the two peaks.  ATLAS's local-maxima-with-a-valley:
                    a genuine second core has a dip between it and the first; a
                    bright patch inside ONE shower does not.
      frac[i]     = charge share nearest to seed i.
    """
    if len(pts)<8:
        return dict(dirs=np.zeros((0,3)), dens=np.zeros(0),
                    valley=np.zeros((0,0)), frac=np.zeros(0))
    U, r = rays(pts, v)
    w = qwt(q)
    if len(U)>max_pts:
        rng = rng or np.random.default_rng(20260901)
        keep = rng.choice(len(U), max_pts, replace=False); keep.sort()
        U, r, w = U[keep], r[keep], w[keep]
    sig = sigma_fn(r)
    ang = np.arccos(np.clip(U@U.T, -1.0, 1.0))
    K = np.exp(-0.5*(ang/sig[None,:])**2)
    dens = K@w
    order = np.argsort(-dens)
    seeds=[]
    for i in order:
        if len(seeds)>=max_seeds: break
        if any(ang[i,j] < sep_scale*max(sig[i],sig[j]) for j in seeds): continue
        seeds.append(int(i))
    S=U[seeds]; D=dens[seeds]; k=len(seeds)
    val=np.ones((k,k))
    ts=np.linspace(0,1,25)[:,None]
    for i in range(k):
        for j in range(i+1,k):
            arc = S[i][None,:]*(1-ts) + S[j][None,:]*ts
            nn=np.linalg.norm(arc,axis=1); nn[nn<=0]=1
            arc/=nn[:,None]
            da = np.exp(-0.5*(np.arccos(np.clip(arc@U.T,-1,1))/sig[None,:])**2)@w
            v_=float(da.min()/max(min(D[i],D[j]),1e-12))
            val[i,j]=val[j,i]=v_
    frac=np.zeros(k)
    if k:
        best=np.argmax(U@S.T,axis=1)
        for i in range(k): frac[i]=w[best==i].sum()
        if frac.sum()>0: frac/=frac.sum()
    return dict(dirs=S, dens=D, valley=val, frac=frac)

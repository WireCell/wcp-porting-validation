#!/usr/bin/env python3
"""Re-predict PDHD Q/L light at an arbitrary VUV absorption length (lambda),
faithfully porting SemiAnalyticalModel::VUVVisibility, to (1) validate against
the C++ pred_pe dumped at lambda=2000 and (2) sweep lambda for the spread fit."""
import json, math, sys

WCD = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json'
M = json.load(open(WCD))
OPD = M['OpDets']                       # 160 dicts: x,y,z,h,w,type,orientation (cm)
GEO = M['Geometry']
ACY, ACZ = GEO['active_center_y'], GEO['active_center_z']
V = M['VUVHits']
GHP = V['GH_PARS_flat']                 # 4 x 9
BANG = V['GH_border_angulo_flat']       # 9
BCOR = V['GH_border_flat']              # 3 x 9
DELTA = V['delta_angulo_vuv']           # 10
NBINS = len(GHP[0])                      # 9
MAXPD = 1000.0
kPi = math.pi

def fast_acos(xin):
    a3,a2,a1,a0 = -2.08730442907856008e-02,7.68769404161671888e-02,-2.12871094165645952e-01,1.57075835365209659e+00
    x = abs(xin); ret = a3
    ret = ret*x + a2; ret = ret*x + a1; ret = ret*x + a0
    ret *= math.sqrt(1.0 - x)
    return ret if xin >= 0 else math.pi - ret

def interpolate(xD, yD, x, extrap, i=0):
    if i == 0:
        n = len(xD)
        if x >= xD[n-2]: i = n-2
        else:
            while x > xD[i+1]: i += 1
    xL,xR,yL,yR = xD[i],xD[i+1],yD[i],yD[i+1]
    if not extrap:
        if x < xL: return yL
        if x > xR: return yL
    return yL + (yR-yL)/(xR-xL)*(x-xL)

def angle_bin(theta):
    if not math.isfinite(theta) or theta <= 0.0: return 0
    j = int(theta/DELTA)
    return j if j < NBINS else NBINS-1

def rect_solid_1(a,b,d):
    aa,bb = a/(2*d), b/(2*d)
    aux = (1+aa*aa+bb*bb)/((1+aa*aa)*(1+bb*bb))
    return 4*fast_acos(math.sqrt(aux))

def approx_zero(a): return abs(a) <= sys.float_info.epsilon
def gt(a,b):
    diff=a-b
    return diff>sys.float_info.epsilon or diff>max(abs(a),abs(b))*sys.float_info.epsilon

def rect_solid(h,w,vx,vy,vz):
    # orientation 0 (anode/cathode): d1=|vy|, d2=|vx|
    d1, d2, az = abs(vy), abs(vx), abs(vz)
    if approx_zero(d1) and approx_zero(vz):
        return rect_solid_1(h,w,d2)
    if gt(d1,h*.5) and gt(az,w*.5):
        A,B = d1-h*.5, az-w*.5
        return (rect_solid_1(2*(A+h),2*(B+w),d2)-rect_solid_1(2*A,2*(B+w),d2)
                -rect_solid_1(2*(A+h),2*B,d2)+rect_solid_1(2*A,2*B,d2))*.25
    if d1<=h*.5 and az<=w*.5:
        A,B = -d1+h*.5, -az+w*.5
        return (rect_solid_1(2*(h-A),2*(w-B),d2)+rect_solid_1(2*A,2*(w-B),d2)
                +rect_solid_1(2*(h-A),2*B,d2)+rect_solid_1(2*A,2*B,d2))*.25
    if gt(d1,h*.5) and az<=w*.5:
        A,B = d1-h*.5, -az+w*.5
        return (rect_solid_1(2*(A+h),2*(w-B),d2)-rect_solid_1(2*A,2*(w-B),d2)
                +rect_solid_1(2*(A+h),2*B,d2)-rect_solid_1(2*A,2*B,d2))*.25
    if d1<=h*.5 and gt(az,w*.5):
        A,B = -d1+h*.5, az-w*.5
        return (rect_solid_1(2*(h-A),2*(B+w),d2)-rect_solid_1(2*(h-A),2*B,d2)
                +rect_solid_1(2*A,2*(B+w),d2)-rect_solid_1(2*A,2*B,d2))*.25
    return 0.0

def gaisser_hillas(x,par):
    X0,Norm = par[3],par[0]
    Diff = par[1]-X0
    Term = ((x-X0)/Diff)**(Diff/par[2])
    Exp = math.exp((par[1]-x)/par[2])
    return Norm*Term*Exp

def vuv_vis(sx,sy,sz, od, lam):
    rx,ry,rz = sx-od['x'], sy-od['y'], sz-od['z']
    dist = math.sqrt(rx*rx+ry*ry+rz*rz)
    if dist > MAXPD: return 0.0
    if (sx<0) != (od['x']<0): return 0.0
    cosine = abs(rx)/dist
    theta = fast_acos(cosine)*180.0/kPi
    sa = rect_solid(od['h'],od['w'], rx,ry,rz)
    vis_geo = math.exp(-dist/lam)*(sa/(4*kPi))
    r = math.hypot(sy-ACY, sz-ACZ)
    j = angle_bin(theta)
    par = [GHP[0][j],GHP[1][j],GHP[2][j],GHP[3][j]]
    s1 = interpolate(BANG,BCOR[0],theta,True)
    s2 = interpolate(BANG,BCOR[1],theta,True)
    s3 = interpolate(BANG,BCOR[2],theta,True)
    par[0]+=s1*r; par[1]+=s2*r; par[2]+=s3*r
    GH = gaisser_hillas(dist,par)
    if not math.isfinite(GH) or GH<0 or GH>10: GH=0.0
    try:
        return GH*vis_geo/cosine
    except ZeroDivisionError:
        return 0.0

# ---- gate constants ----
ANODE_EXT1, CATHODE_EXT1 = -2.0, 1.2
QTOL, VUVEFF = 1.0, 0.03

def predict(points, geom, offset, lam, nch=160):
    """points: list of (x,y,z,q) cm; geom: dict with anode_x,s,u_cathode,y_lo.. ;
    offset: flash_x_offset cm. Returns pred_pe[nch]."""
    ax,s,uc = geom['anode_x'],geom['s'],geom['u_cathode']
    ylo,yhi,zlo,zhi = geom['y_lo'],geom['y_hi'],geom['z_lo'],geom['z_hi']
    pred=[0.0]*nch
    for (x0,y,z,q) in points:
        x = x0+offset
        u = s*(x-ax)
        if u<ANODE_EXT1 or u>uc+CATHODE_EXT1: continue
        if y<ylo or y>yhi or z<zlo or z>zhi: continue
        for idet in range(nch):
            od=OPD[idet]
            if (x<0)!=(od['x']<0): continue
            v=vuv_vis(x,y,z,od,lam)
            if v: pred[idet]+= q*QTOL*VUVEFF*v
    return pred

#!/usr/bin/env python3
"""Read-only sizing check for the PDHD QLMatching bundle prefilters
(require_containment + reject_overpred), run on the run-29107 calib dumps.

Answers the three questions behind enabling the two cuts as PDHD defaults:
  1. Is the TWO-APA detector box in effect?  (geometry z-span ~4.6 m, not ~2.3 m)
  2. Would require_containment cull current winners / the two-boundary light anchors?
     (compares a recomputed u=s*((x+offset)-anode_x) min/max containment proxy against
     the dumped C++ `contained` flag, then reports how many auto_selected winners fail.)
  3. What loose overpred ceilings keep the current winners?  Reports the R_total / R_max
     distribution over non-boundary winners (boundary bundles are exempt in C++) and the
     cull fraction at candidate (overpred_total_ratio, overpred_maxch_ratio) pairs.

Usage: python3 containment_overpred_check.py [WORKDIR]
The C++ cut is in match/src/QLMatching.cxx (containment :1056, overpred :1140-1162)."""
import json, glob, statistics, sys

WORK = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/work'
RUN  = 29107
STATIC = set([3,86,87,97,107,116,117]) | set(range(120,160))
ANODE_EXT1, CATHODE_EXT1 = -2.0, 1.2   # cm, C++ m_anode_ext1 / m_cathode_ext1 defaults

def cluster_us(d, b, gr):
    cl = {c['ident']: c for c in d['clusters']}
    fl = {f['gid']: f for f in d['flashes']}
    f = fl.get(b['flash_gid'])
    if f is None: return None
    offset = gr['sign_offset'] * (f['time'] + d['trigger_offset']) * d['drift_speed']
    s, ax = gr['s'], gr['anode_x']
    us = [s*((x+offset)-ax) for cid in [b['main_cluster']]+b.get('other_clusters',[])
          for c in [cl.get(cid % 1000000)] if c for x in c['x']]
    return (min(us), max(us), gr['u_cathode']) if us else None

def contained_proxy(umin, umax, ucath, cext1=CATHODE_EXT1):
    return umin > ANODE_EXT1-1.0 and umax > 0.0 and umax < ucath+cext1 and umin < ucath

zspan=[]; nwin=0; agree=0; notc=0; tbwin=0; tbnotc=0
rt=[]; rm=[]; zero_meas=0
for ev in range(1300):
    for grp in ('group02','group13'):
        g = glob.glob('%s/0%d_%d/calib-*-%s.json' % (WORK,RUN,ev,grp))
        if not g: continue
        d = json.load(open(g[0]))
        gr = d['geometry'][list(d['geometry'].keys())[0]]
        zspan.append(gr['z_hi']-gr['z_lo'])
        fl = {f['gid']: f for f in d['flashes']}
        amask = set(i for i in range(160) if d['opdets'][i]['auto_masked'])
        mask = STATIC | amask
        for b in d['bundles']:
            if not b.get('auto_selected'): continue
            uu = cluster_us(d, b, gr)
            if uu:
                nwin += 1
                if contained_proxy(*uu) == bool(b.get('contained')): agree += 1
                if not b.get('contained'): notc += 1
                if b.get('two_boundary'):
                    tbwin += 1
                    if not b.get('contained'): tbnotc += 1
            if b.get('close_to_PMT') or b.get('window_truncated') or b.get('at_x_boundary'):
                continue
            f = fl.get(b['flash_gid'])
            if f is None: continue
            tp=tm=mp=mam=0.0
            for j in range(len(b['pred_pe'])):
                if j in mask: continue
                p=b['pred_pe'][j]; m=f['pe'][j]; tp+=p; tm+=m
                if p>mp: mp=p; mam=m
            if tm<=0: zero_meas += 1; continue
            rt.append(tp/tm); rm.append(mp/(mam if mam>1 else 1.0) if mp>0 else 0)

print("1. two-APA box: dumps=%d  z-span median=%.1f cm (single APA ~230)" % (len(zspan), statistics.median(zspan)))
print("2. containment winners=%d  proxy/dumped-agree=%.1f%%  not-contained=%d  two_boundary=%d (not-contained %d)"
      % (nwin, 100.*agree/nwin if nwin else 0, notc, tbwin, tbnotc))
def pct(xs,p): xs=sorted(xs); return xs[int(p*len(xs))] if xs else 0
n=len(rt)
print("3. overpred non-boundary winners=%d (degenerate tot_meas=0: %d)" % (n, zero_meas))
print("   R_total p95=%.2f p99=%.2f   R_max p95=%.2f p99=%.2f" % (pct(rt,.95),pct(rt,.99),pct(rm,.95),pct(rm,.99)))
for rtc,rmc in [(2.9,4.3),(5.0,12.0),(6.0,15.0)]:
    cut = sum(1 for a,b in zip(rt,rm) if a>rtc or b>rmc)
    print("   ceiling (%.1f, %.1f) culls %d/%d (%.1f%%) +%d degenerate" % (rtc,rmc,cut,n,100.*cut/n if n else 0,zero_meas))

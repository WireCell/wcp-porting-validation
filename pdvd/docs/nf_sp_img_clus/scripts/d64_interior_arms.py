#!/usr/bin/env python3
"""doc pdvd/64 (T6) -- the interior-arm table against the scan's tags.

Fork of d56_failure_mechanisms.py sec 8, re-keyed on the chain's own
classification now that role 2 (delta) reaches chain_role and role 7 (the
kOther arm, publish_other_arms) exists.  For every segment the scan tagged
on a stopper: the chain's role (1 muon / 2 delta / 3 michel / 5 gamma /
6 survey / 7 other / none), attachment to the muon chain, length, MIP, and
for role 7 whether it hangs at the stop or interior.  Then the residual
"no role at all" set, which by construction is what the PR never made a
chain arm of.

Usage: python3 d64_interior_arms.py <prep_dir> [<prep_dir_before>]
"""
import sys, collections
import numpy as np
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

prep = sys.argv[1]
rec = C.load_record()
P, missing = C.load_payloads(prep, rec)
keys = [k for k in rec if k in P]
stop = [k for k in keys if C.judged(rec[k]) and C.is_stopper(rec[k])]
print("arm prep %s: %d payloads, %d stoppers judged" % (prep, len(keys), len(stop)))

ROLE = {1: "role 1 muon", 2: "role 2 delta", 3: "role 3 michel", 5: "role 5 gamma", 6: "role 6 survey", 7: "role 7 other", None: "no role"}

def attach(pay, s):
    rr_, q_, xyz_ = C.profile(pay); pts = C.seg_points(s)
    d0 = np.linalg.norm(xyz_ - pts[0], axis=1); d1 = np.linalg.norm(xyz_ - pts[-1], axis=1)
    d = float(min(d0.min(), d1.min())); i = int(np.argmin(np.minimum(d0, d1)))
    where = "attached" if d <= 2 else "near 2-6 cm" if d <= 6 else "far >6 cm"
    return where, ("stop rr<3" if rr_[i] < 3 else "interior"), d

# ---- 1. every scan tag on a stopper, by the chain's role ----------------
tab = collections.defaultdict(collections.Counter)     # scan tag -> chain role -> n
for k in stop:
    p = P[k]; segs, cl = C.seg_index(p); role = p["pf"]["chain_role"]
    for t, v in rec[k]["tags"].items():
        if t.startswith("C"):
            tab[v]["(unfitted cluster)"] += 1; continue
        if t not in segs:
            tab[v]["(not in pf.seg)"] += 1; continue
        same = cl.get(t, {}).get("id") == p["cluster_id"]
        rl = role.get(t)
        tab[v][ROLE.get(rl, "role %s" % rl) + ("" if same else " [other cluster]")] += 1
print("\n=== 1. scan tag (rows) x chain role (cols), scan stoppers ===")
cols = sorted({c for v in tab.values() for c in v}, key=lambda c: (c.startswith("("), c))
print("%-20s" % "scan tag" + "".join("%22s" % c[:21] for c in cols))
for tag in sorted(tab):
    print("%-20s" % tag + "".join("%22d" % tab[tag][c] for c in cols))

# ---- 2. the scanner's 'delta / other' tags: chain role x attachment ----
print("\n=== 2. the scan's 'delta / other' segments: chain role, attachment, length, MIP ===")
cat = collections.Counter(); mips = collections.defaultdict(list); big = collections.Counter()
for k in stop:
    p = P[k]; segs, cl = C.seg_index(p); role = p["pf"]["chain_role"]
    for t, v in rec[k]["tags"].items():
        if v != "delta / other" or t.startswith("C") or t not in segs: continue
        if cl.get(t, {}).get("id") != p["cluster_id"]: cat[("other cluster", ROLE.get(role.get(t), "?"), "", "")] += 1; continue
        s = segs[t]; where, pos, d = attach(p, s)
        rl = ROLE.get(role.get(t), "role %s" % role.get(t))
        cat[(rl, where, "<=8 cm" if s["len_cm"] <= 8 else ">8 cm", pos)] += 1
        if s["dqdx_med"]: mips[rl].append(s["dqdx_med"] / C.MIP_MEDIAN)
        if role.get(t) is None and s["len_cm"] > 8: big[k] += 1
for kk, c in sorted(cat.items(), key=lambda x: -x[1]):
    print("  %4d  %s" % (c, ", ".join(x for x in kk if x)))
for rl, m in sorted(mips.items()):
    print("  %-16s n %3d  mip median %.2f  > 1.4 MIP: %d" % (rl, len(m), np.median(m), sum(x > 1.4 for x in m)))
print("items with most > 8 cm NO-ROLE same-cluster 'delta / other' segments:", big.most_common(6))

# ---- 3. role 7 on the arm: where, and against the tags ------------------
print("\n=== 3. role-7 segments (the chain's kOther arms), all scored items ===")
n7 = 0; where7 = collections.Counter(); tag7 = collections.Counter(); len7 = []; mip7 = []
for k in keys:
    p = P[k]; segs, cl = C.seg_index(p); role = p["pf"]["chain_role"]
    for t, rl in role.items():
        if rl != 7 or t not in segs: continue
        n7 += 1; s = segs[t]; where, pos, d = attach(p, s)
        where7[(where, pos)] += 1; len7.append(s["len_cm"])
        if s["dqdx_med"]: mip7.append(s["dqdx_med"] / C.MIP_MEDIAN)
        tag7[rec[k]["tags"].get(t, "(untagged)") if k in rec else "?"] += 1
print("role-7 segments: %d on %d items" % (n7, sum(1 for k in keys if 7 in P[k]["pf"]["chain_role"].values())))
if len7:
    print("  length median %.1f cm, > 8 cm: %d;  mip median %.2f, > 1.4: %d" % (np.median(len7), sum(l > 8 for l in len7), np.median(mip7), sum(m > 1.4 for m in mip7)))
for kk, c in sorted(where7.items(), key=lambda x: -x[1]): print("  %4d  %s" % (c, ", ".join(kk)))
print("  scan tags on the role-7 segments:", dict(tag7.most_common()))
# agreement: n_body_other + n_stop_other vs published
tot_other = sum((P[k]["verdict"].get("n_body_other") or 0) + (P[k]["verdict"].get("n_stop_other") or 0) for k in keys)
tot_pub = sum(P[k]["verdict"].get("n_other_published") or 0 for k in keys)
print("  sum n_body_other + n_stop_other = %d; sum n_other_published = %d; role-7 segments in chain_role = %d" % (tot_other, tot_pub, n7))

# ---- 4. C3 as census_score prints it, for the before/after quote --------
n_items = n_segs = n_long = 0
for k in stop:
    pay = P[k]; segs, cl = C.seg_index(pay); role = pay["pf"]["chain_role"]
    here = [s for t, s in segs.items() if t not in role and cl.get(t, {}).get("id") == pay["cluster_id"]]
    if here: n_items += 1; n_segs += len(here); n_long += sum(1 for s in here if s["len_cm"] > 5)
print("\n=== 4. C3 (same-cluster no-role segments on scan stoppers): items %d, segments %d, > 5 cm %d ===" % (n_items, n_segs, n_long))

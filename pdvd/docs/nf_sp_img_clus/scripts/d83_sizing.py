#!/usr/bin/env python3
"""doc pdvd/83 -- doc 78 action item 4 ("the attached arm near the stop") sized
offline on the segment-census arm against the smx1a+smx3+smx4 record, read-only.

The payloads are doc 80's census arm (p80bcen: production p79vprod + segment_census,
so every PR segment of the muon's cluster has a row and a `rej`).  The graph is
rebuilt from each payload's pf.seg / pf.vtx: segment endpoints are matched to the
nearest vertex (exact, 0.00 cm, on every item looked at); a segment is a CHAIN
segment when >= 80 % of its points lie within 0.5 cm of the muon's role-1 rows.

Every arm is measured the way stm_michel_classify_stop_arm measures one
(StmMichelFunctions.cxx measure_arm), with the compiled PDVD production values:
  mip       segment dQ/dx median / mip_dqdx_median = 47000 e/cm
  kink      the incoming muon direction over dir_window = 15 cm (read off the
            role-1 rows) against the arm's direction over 15 cm from the vertex
  far_len   every segment reachable from the arm's far vertex without stepping
            back into the vertex it leaves (segment_far_subtree_track_length)
and judged with the doc pdvd/48 Michel clause production runs (no P2 sub-knob on):
  len + far_len <= michel_max_len_cm 25,  0.3 < mip < 2.0 (michel_mip_lo/hi),
  kink >= michel_min_kink_deg 30  (a shower flag is not in the payload; see sec 0)

Sections:
  0  the twin's OWN validation against the C++'s `stop-arm:` DEBUG lines of the
     same arm: the numbers the classifier printed for every stop arm vs this
     reading of the same segment.  Sets the error bar on every margin below.
  1  where the four doc 78 sec 3.2 segments hang, and why nothing reads them.
  2  the population: non-chain arms at interior chain vertices within G cm of
     the stop, by record class, michel_found and role.
  3  the rule: an UNCLAIMED arm (role 7/8/none) within G, offered production's
     stop-arm gate, on items with no Michel today -- fires by name, per G.
  4  three variants, sized and not built: (a) arms production already claimed
     as deltas (role 2); (b) doc 78's "as energy" half -- unclaimed stop-vertex
     arms on items that HAVE a Michel, by the scanner's tag; (c) arms whose
     reach walk re-enters the muon chain.
  5  the verdict channel: topology_stop_evidence clears no_bragg / shape_flat
     for a conn-1/2 Michel >= 10 MeV, >= 3 cm -- the reject bits of every item
     sec 3 fires on.

Repro (doc 83 sec 0):
  python3 d83_sizing.py > /home/xqian/tmp/p83/sizing.txt
"""
import argparse, collections, glob, json, math, re
import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/p80/prep_p80bcen")
ap.add_argument("--arm", default="p80bcen", help="work/<evt>_<arm>/ whose wct_pr logs carry the stop-arm lines")
ap.add_argument("--record", default=IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
args = ap.parse_args()

MIPMED, DIRW = 47000.0, 15.0              # mip_dqdx_median (compiled), StmMichelArmThresholds::dir_window
MAXLEN, MIPLO, MIPHI, KMIN = 25.0, 0.3, 2.0, 30.0
TARGETS = ["039349_64/24", "039349_9/19", "039349_64/52", "039349_69/56"]

R = {r["key"]: r for r in json.load(open(args.record))}
P = {}
for fn in glob.glob(args.prep + "/smprep-*.json"):
    d = json.load(open(fn))
    P["%s/%d" % (d["event"], d["cluster_id"])] = d
print("prep %s: payloads %d, of them in the record %d" % (args.prep, len(P), sum(1 for k in P if k in R)))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cls(k):
    r = R.get(k)
    if not r:
        return "unjudged"
    return r["verdict"] if r["verdict"] in ("STM_MICHEL", "STM_ONLY", "THRU") else "other:" + r["verdict"]


def tag(k, sid):
    return ((R[k]["tags"].get(str(sid)) if k in R else None) or "-")


def arms_of(p, G=15.0):
    m = p["muon"]
    o = np.argsort(np.asarray(m["rr"], float))
    M = np.c_[np.asarray(m["x"])[o], np.asarray(m["y"])[o], np.asarray(m["z"])[o]].astype(float)
    rr = np.asarray(m["rr"], float)[o]
    pf = p["pf"]
    if not pf.get("vtx") or not pf["vtx"].get("x"):
        return []
    V = np.c_[pf["vtx"]["x"], pf["vtx"]["y"], pf["vtx"]["z"]].astype(float)
    segs = {}
    for s in pf["seg"]:
        X = np.c_[s["x"], s["y"], s["z"]].astype(float)
        if len(X) < 1:
            continue
        ia = int(np.argmin(np.linalg.norm(V - X[0], axis=1)))
        ib = int(np.argmin(np.linalg.norm(V - X[-1], axis=1)))
        dnn = np.array([np.linalg.norm(M - x, axis=1).min() for x in X])
        segs[s["id"]] = dict(X=X, a=ia, b=ib, chain=(np.mean(dnn < 0.5) >= 0.8), len=s["len_cm"], med=s.get("dqdx_med") or 0)
    if not segs:
        return []
    vs = int(np.argmin(np.linalg.norm(V - M[0], axis=1)))
    adj = collections.defaultdict(list)
    for sid, s in segs.items():
        adj[s["a"]].append(sid)
        adj[s["b"]].append(sid)
    cv = {v for s in segs.values() if s["chain"] for v in (s["a"], s["b"])}
    out = []
    for v in sorted(cv):
        rrv = rr[int(np.argmin(np.linalg.norm(M - V[v], axis=1)))]
        if rrv > G:
            continue
        din = unit(V[v] - M[min(int(np.searchsorted(rr, rrv + DIRW)), len(rr) - 1)])
        for sid in adj[v]:
            s = segs[sid]
            if s["chain"]:
                continue
            X = s["X"] if s["a"] == v else s["X"][::-1]
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
            dout = unit(X[min(int(np.searchsorted(L, DIRW)), len(X) - 1)] - V[v])
            kink = math.degrees(math.acos(max(-1.0, min(1.0, float(din @ dout)))))
            far = s["b"] if s["a"] == v else s["a"]
            seen, stack, vseen, farlen, into = {sid}, [far], {far, v}, 0.0, False
            while stack:
                u = stack.pop()
                into = into or (u in cv)
                for s2 in adj[u]:
                    if s2 in seen:
                        continue
                    seen.add(s2)
                    t = segs[s2]
                    ov = t["b"] if t["a"] == u else t["a"]
                    if ov == v:
                        continue
                    farlen += t["len"]
                    if ov not in vseen:
                        vseen.add(ov)
                        stack.append(ov)
            out.append(dict(sid=sid, is_stop=(v == vs), rrv=float(rrv), len=s["len"], mip=s["med"] / MIPMED, kink=kink,
                            far=farlen, into_chain=into, role=pf["chain_role"].get(str(sid)),
                            rej=(pf["seg_rej"].get(str(sid)) or {}).get("rej"), deg=len(adj[v])))
    return out


def gate(a):
    return a["len"] + a["far"] <= MAXLEN and MIPLO < a["mip"] < MIPHI and a["kink"] >= KMIN


def why(a):
    w = []
    if a["len"] + a["far"] > MAXLEN: w.append("reach")
    if not (MIPLO < a["mip"] < MIPHI): w.append("charge")
    if a["kink"] < KMIN: w.append("turn")
    return "+".join(w) or "PASS"


def row(k, a):
    v = P[k]["verdict"]
    return ("  %-14s %-11s is_stm %d mf %d | seg %6d role %-4s rej %-4s rr_v %4.1f len %5.1f far %5.1f%s mip %.2f kink %5.1f deg %d | tag %-14s %s"
            % (k, cls(k), v["is_stm"], v["michel_found"], a["sid"], a["role"], a["rej"], a["rrv"], a["len"], a["far"],
               "*" if a["into_chain"] else " ", a["mip"], a["kink"], a["deg"], tag(k, a["sid"]), why(a)))


A = {k: arms_of(p) for k, p in P.items()}

# ---------------------------------------------------------------- 0. validation
print("\n=== 0. the twin against the C++'s own stop-arm lines (%s logs) ===" % args.arm)
pat = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg shower (\d)")
logv = {}
for k in P:
    ev, cid = k.split("/")
    for f in glob.glob("%s/pdvd/work/%s_%s/wct_pr_*.log" % (IMG, ev, args.arm)):
        for line in open(f, errors="ignore"):
            m = pat.search(line)
            if m and m.group(1) == cid:
                logv[(k, int(m.group(2)))] = dict(len=float(m.group(4)), far=float(m.group(5)), mip=float(m.group(6)),
                                                 kink=float(m.group(7)), shower=int(m.group(8)))
dd = collections.defaultdict(list)
nsh = 0
for k in P:
    for a in A[k]:
        L = logv.get((k, a["sid"]))
        if not a["is_stop"] or not L:
            continue
        nsh += L["shower"]
        dd["mip"].append(a["mip"] - L["mip"]); dd["len cm"].append(a["len"] - L["len"])
        dd["far cm (cap 60)"].append(min(a["far"], 60) - min(L["far"], 60)); dd["kink deg"].append(a["kink"] - L["kink"])
print("stop arms matched twin <-> log: %d of %d log lines (%d shower-flagged)" % (len(dd["mip"]), len(logv), nsh))
for nm in ("mip", "len cm", "far cm (cap 60)", "kink deg"):
    x = np.abs(np.asarray(dd[nm]))
    print("  %-16s |twin - C++|  p50 %.3f  p90 %.3f  p99 %.3f  max %.3f" % (nm, np.median(x), np.percentile(x, 90), np.percentile(x, 99), x.max()))
print("  (the kink's tail is a SHORT reference segment: the C++ reads the stop arm against `last`, the twin against 15 cm of")
print("   muon rows; at an interior vertex the reference is the incoming chain segment, >= 35 cm on every target)")

# ---------------------------------------------------------------- 1. the four
print("\n=== 1. the four doc 78 sec 3.2 segments: where they hang ===")
for k in TARGETS:
    for a in A[k]:
        if tag(k, a["sid"]) == "michel" and a["rej"] == 13:
            print(row(k, a), "stop-vertex arm" if a["is_stop"] else "INTERIOR vertex, %s" % ("penultimate" if a["deg"] == 3 else "deg %d" % a["deg"]))
    print("     no stop-arm log line for these segments: %s" % (not any((k, a["sid"]) in logv for a in A[k] if tag(k, a["sid"]) == "michel")))

# ---------------------------------------------------------------- 2. population
print("\n=== 2. non-chain arms at interior chain vertices, rr_v <= G: (class, michel_found, role) -> n ===")
for G in (3, 5, 7, 10, 15):
    c = collections.Counter()
    for k, arms in A.items():
        for a in arms:
            if not a["is_stop"] and a["rrv"] <= G:
                c[(cls(k), P[k]["verdict"]["michel_found"], a["role"])] += 1
    print("G = %2d cm: %s" % (G, ", ".join("%s mf%d role %s: %d" % (x[0], x[1], x[2], n) for x, n in sorted(c.items(), key=lambda t: -t[1]))))

# ---------------------------------------------------------------- 3. the rule
print("\n=== 3. the rule: unclaimed (role 7 / 8 / none) arm within G, the stop-arm gate, items with no Michel ===")
for G in (3, 5, 7, 10, 15):
    fires = []
    for k in sorted(A):
        if P[k]["verdict"]["michel_conn_type"] != 0:
            continue
        hit = [a for a in A[k] if not a["is_stop"] and a["rrv"] <= G and a["role"] in (7, 8, None) and gate(a)]
        if hit:
            fires.append((k, cls(k), hit[0]["sid"], hit[0]["rrv"]))
    by = collections.Counter(f[1] for f in fires)
    print("G = %2d cm: %d fires  %s" % (G, len(fires), dict(by)))
    for f in fires:
        print("     %-14s %-11s seg %d at %.1f cm" % f)
# the shower flag is not in the payload: a shower-flagged arm's turn bar is
# michel_shower_min_kink_deg = 15 instead of 30 (sec 0 counts how common the flag is)
amb = [(k, a["sid"], round(a["kink"], 1)) for k in sorted(A) for a in A[k]
       if not a["is_stop"] and a["rrv"] <= 15 and a["role"] in (7, 8, None) and why(a) == "turn" and a["kink"] >= 15]
print("\nshower-flag ambiguity: unclaimed interior arms within 15 cm refused on the turn ALONE with kink in [15, 30): %d %s" % (len(amb), amb))
print("\nevery unclaimed interior arm within 10 cm, gate reading (items with a Michel today are unaffected by the rule):")
for k in sorted(A):
    for a in A[k]:
        if not a["is_stop"] and a["rrv"] <= 10 and a["role"] in (7, 8, None):
            print(row(k, a))

# ---------------------------------------------------------------- 4. variants
print("\n=== 4a. arms production already CLAIMED as deltas (role 2) within G, gate applied, items with no Michel ===")
for G in (5, 7, 10):
    fires = [(k, cls(k), a["sid"], a["rrv"]) for k in sorted(A) if P[k]["verdict"]["michel_conn_type"] == 0
             for a in A[k] if not a["is_stop"] and a["rrv"] <= G and a["role"] == 2 and gate(a)]
    print("G = %2d cm: %d fires %s" % (G, len(fires), dict(collections.Counter(f[1] for f in fires))))
    for f in fires:
        print("     %-14s %-11s seg %d at %.1f cm" % f)
print("\n=== 4b. doc 78's 'as energy' half: unclaimed STOP-vertex arms on items WITH a Michel, by the scanner's tag ===")
c = collections.Counter()
for k in sorted(A):
    if not P[k]["verdict"]["michel_found"]:
        continue
    for a in A[k]:
        if a["is_stop"] and a["role"] in (7, 8, None):
            c[(cls(k), tag(k, a["sid"]))] += 1
for x, n in sorted(c.items(), key=lambda t: (t[0][0], -t[1])):
    print("  %-11s %-14s %d" % (x[0], x[1], n))
t = collections.Counter()
for (cl_, tg), n in c.items():
    t["michel" if tg == "michel" else ("untagged" if tg == "-" else "not michel")] += n
print("  total: %s" % dict(t))
print("\n=== 4c. arms within 15 cm whose reach walk re-enters the muon chain ===")
for k in sorted(A):
    for a in A[k]:
        if a["into_chain"] and a["role"] not in (1, 3):
            print(row(k, a), "STOP" if a["is_stop"] else "")

# ---------------------------------------------------------------- 5. verdict
print("\n=== 5. the verdict channel: reject bits of every item sec 3 fires on at G <= 10 ===")
for k in sorted(A):
    if P[k]["verdict"]["michel_conn_type"] != 0:
        continue
    if any(not a["is_stop"] and a["rrv"] <= 10 and a["role"] in (7, 8, None) and gate(a) for a in A[k]):
        v = P[k]["verdict"]
        print("  %-14s %-11s is_stm %d bits %4d %s" % (k, cls(k), v["is_stm"], v["reject_bits"], v.get("reject_names")))
print("  (topology_stop_evidence clears only no_bragg | shape_flat | profile_sparse)")

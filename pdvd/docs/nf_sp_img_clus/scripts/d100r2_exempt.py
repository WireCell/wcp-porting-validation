#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- what the readout-edge exemption (readout_edge_defer + readout_edge_require_michel) does, against
the predictions and the owner's verdicts pre-registered in d100/prereg_round2.md sec 2.

    python3 d100r2_exempt.py --arm p100bx  --base p100c    --control p100bg --defer p100bd --side latest \
        --carried /home/xqian/tmp/p100/carry/latest_on_p99rwon.json --unjudged-out /home/xqian/tmp/p100/exempt_unjudged_p100bx.tsv
    python3 d100r2_exempt.py --arm p100bxp --base p99wflip --side production \
        --carried /home/xqian/tmp/p100/carry/latest_on_p99wflip.json --unjudged-out /home/xqian/tmp/p100/exempt_unjudged_p100bxp.tsv

1. gained / lost is_stm, arm against base, matched by geometry (d99_swap_scan_set.rows_of, both directions).
2. every deferral on the arm (its log line and stop tick) and what became of it: final is_stm, michel_found, R_READOUT_EDGE.
3. E1 (with --control/--defer): gained == {clusters the defer arm deferred that the guard-off control tags is_stm with
   michel_found 1 and the base does not tag}; exceptions both ways.
4. E2: own100's objects of this side (record keys on p98vonq for latest, p96vprod for production) located on the arm by
   geometry: restored or not, michel_found on the arm (and the control), the owner's verdict.
5. the gained set by verdict source (own100 by geometry > own100m by key > the carried record by key); the objects no owner
   verdict covers go to --unjudged-out (the own100x scan list).
"""
import argparse, collections, glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99_match as M
import d99_swap_scan_set as W

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WORK = IMG + "/pdvd/work"
SCAN = IMG + "/pdvd/docs/scan"
OWN100 = SCAN + "/pdvd_stm_michel_own100_verdicts.json"
OWN100M = SCAN + "/pdvd_stm_michel_own100m_verdicts.json"
R_READOUT_EDGE = 1 << 14
STOP = ("STM_MICHEL", "STM_ONLY")
BASE_OF_SIDE = {"latest": "p98vonq", "production": "p96vprod"}


def cls(v):
    b = (v or "").split(" ")[0].replace("FRAG_", "")
    if not b:
        return "unjudged"
    return "stopper" if b in STOP else ("excluded" if b in ("MESSY", "UNCLEAR") else "non-stopper")


def events():
    out = []
    for line in open(IMG + "/pdvd/stm/events.txt"):
        f = line.split()
        if line.startswith("#") or len(f) < 2:
            continue
        out.append(f"{int(f[0]):06d}_{f[1]}")
    return out


def tree(evt, arm):
    f = f"{WORK}/{evt}_{arm}/tracking-pr.root"
    if not os.path.exists(f):
        return None
    u = uproot.open(f)
    if "T_stm_michel" not in [k.split(";")[0] for k in u.keys()]:
        return {}
    t = u["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found", "reject_bits", "michel_ke_best", "stop_x"], library="np")
    return {int(c): {k: t[k][i] for k in t} for i, c in enumerate(t["cluster_id"])}


def deferrals(evt, arm):
    out, tick = {}, {}
    for lg in sorted(glob.glob(f"{WORK}/{evt}_{arm}/wct_pr_*.log")):
        for line in open(lg, errors="replace"):
            m = re.search(r"readout_edge_guard: cluster (\d+) rejected: stop at tick ([0-9.]+)", line)
            if m:
                tick[int(m.group(1))] = float(m.group(2))
            m = re.search(r"readout_edge_guard: cluster (\d+) deferred", line)
            if m:
                c = int(m.group(1))
                out[c] = tick.get(c)
    return out


def do_event(job):
    evt, a = job
    r = dict(evt=evt)
    r["gained"] = [x for x in W.rows_of((evt, a["arm"], a["base"])) if x["status"] != "is_stm"]
    r["lost"] = [x for x in W.rows_of((evt, a["base"], a["arm"])) if x["status"] != "is_stm"]
    r["arm"] = tree(evt, a["arm"]) or {}
    r["defer"] = deferrals(evt, a["arm"])
    if a.get("control"):
        r["control"] = tree(evt, a["control"]) or {}
        r["defer_arm"] = deferrals(evt, a["defer"])
        exp = []
        if r["control"]:
            ctl_to_base = {x["key"]: x for x in W.rows_of((evt, a["control"], a["base"]))}
            for c, d in r["control"].items():
                if int(d["is_stm"]) == 1 and int(d["michel_found"]) == 1 and c in r["defer_arm"]:
                    st = ctl_to_base.get(f"{evt}/{c}", {}).get("status")
                    if st != "is_stm":
                        exp.append((f"{evt}/{c}", st))
        r["expected"] = exp
    return r


def locate(key, src_arm, arm, cache):
    evt, cid = key.split("/")
    if (evt, src_arm) not in cache:
        cache[(evt, src_arm)] = M.load_bee(evt, src_arm)
    if (evt, arm) not in cache:
        cache[(evt, arm)] = M.load_bee(evt, arm)
    sb, ab = cache[(evt, src_arm)], cache[(evt, arm)]
    if sb is None or ab is None:
        return None, "no bee"
    pts = sb[0][sb[1] == int(cid)]
    if len(pts) == 0:
        return None, "no points"
    m = M.match_cluster(pts, ab[0], ab[1], cKDTree(ab[0]))
    st = M.status_of(m)
    return (f"{evt}/{m['best']}" if st == "ok" else None), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--defer", default=None)
    ap.add_argument("--side", required=True, choices=("latest", "production"))
    ap.add_argument("--carried", required=True)
    ap.add_argument("--unjudged-out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    if os.path.exists(a.unjudged_out):
        sys.exit(f"REFUSING: {a.unjudged_out} exists")
    cfg = dict(arm=a.arm, base=a.base, control=a.control, defer=a.defer)
    with ProcessPoolExecutor(a.jobs) as ex:
        R = list(ex.map(do_event, [(e, cfg) for e in events()]))
    print(f"# doc pdvd/100 round 2 -- readout-edge exemption: {a.arm} against {a.base} ({a.side} side); d100r2_exempt.py")
    print(f"# events {len(R)}; with PR output on {a.arm}: {sum(1 for r in R if r['arm'])}")

    gained = [(r, x) for r in R for x in r["gained"]]
    lost = [(r, x) for r in R for x in r["lost"]]
    print(f"\n== 1. is_stm {a.base} -> {a.arm}: gained {len(gained)}, lost {len(lost)}")
    for r, x in lost:
        print(f"  LOST   {x['key']:16s} {x['vol']:6s} on {a.arm}: {x['status']} {x['other']}")

    print(f"\n== 2. deferrals on {a.arm} (log line 'readout_edge_guard: cluster N deferred')")
    fate = collections.Counter()
    for r in R:
        for c, t in r["defer"].items():
            d = r["arm"].get(c)
            if d is None:
                fate["not a check_stm_michel candidate (tagger did not accept the deferred pass)"] += 1
            elif int(d["is_stm"]) == 1:
                fate["is_stm (Michel present, nothing else rejects)"] += 1
            elif int(d["reject_bits"]) & R_READOUT_EDGE:
                fate["R_READOUT_EDGE" + (" only" if int(d["reject_bits"]) == R_READOUT_EDGE else " + other bits")] += 1
            else:
                fate["rejected by other bits, has a Michel"] += 1
    by_run = collections.Counter(r["evt"][:6] for r in R for _ in r["defer"])
    print(f"  deferred clusters {sum(len(r['defer']) for r in R)} (by run {dict(by_run)})")
    for k, v in fate.most_common():
        print(f"    {v:4d}  {k}")

    if a.control:
        print(f"\n== 3. E1: gained on {a.arm} == deferred on {a.defer} & is_stm + michel_found on {a.control} & not is_stm on {a.base}")
        obs = {x["key"] for _, x in gained}
        exp = {k for r in R for k, _ in r["expected"]}
        print(f"  observed {len(obs)}, expected {len(exp)}, both {len(obs & exp)}")
        for k in sorted(obs - exp):
            print(f"  gained, not predicted: {k}")
        for k in sorted(exp - obs):
            print(f"  predicted, not gained: {k}")
        print(f"  E1 {'HOLDS' if obs == exp else 'EXCEPTIONS LISTED ABOVE'}")

    own = [it for it in json.load(open(OWN100)) if it.get("side") == a.side]
    cache, own_on_arm = {}, {}
    print(f"\n== 4. E2: own100's {len(own)} {a.side}-side objects located on {a.arm} (from {BASE_OF_SIDE[a.side]} by geometry)")
    gk = {x["key"]: (r, x) for r, x in gained}
    Rm = {r["evt"]: r for r in R}
    for it in sorted(own, key=lambda it: it["key"]):
        k, st = locate(it["key"], BASE_OF_SIDE[a.side], a.arm, cache)
        if k:
            own_on_arm[k] = it
        d = Rm[it["key"].split("/")[0]]["arm"].get(int(k.split("/")[1])) if k else None
        c = (Rm[it["key"].split("/")[0]].get("control") or {}).get(int(k.split("/")[1])) if k else None
        print(f"  {it['key']:14s} -> {k or st:14s} OWNER {it['verdict']:10s} ({cls(it['verdict']):11s}) "
              f"{'RESTORED' if k in gk else 'not restored':12s} arm michel_found {int(d['michel_found']) if d is not None else '-'}"
              f"{'' if c is None else '  control michel_found %d is_stm %d' % (int(c['michel_found']), int(c['is_stm']))}")
    rest = [it for it in own if any(k == x for x, v in own_on_arm.items() if v is it)]
    s = [own_on_arm[k] for k in gk if k in own_on_arm]
    print(f"  restored: {len(s)}: stoppers {sum(cls(it['verdict']) == 'stopper' for it in s)}, "
          f"non-stoppers {sum(cls(it['verdict']) == 'non-stopper' for it in s)}; located {len(own_on_arm)} of {len(own)}")

    carried = {it["key"]: it for it in json.load(open(a.carried))}
    ownm = {}
    if os.path.exists(OWN100M):
        ownm = {it["key"]: it for it in json.load(open(OWN100M))}
    print(f"\n== 5. the gained set by verdict ({len(gained)})")
    tally, unj = collections.Counter(), []
    for r, x in sorted(gained, key=lambda rx: rx[1]["key"]):
        c = int(x["key"].split("/")[1])
        d = r["arm"].get(c, {})
        if x["key"] in own_on_arm:
            v, src = own_on_arm[x["key"]]["verdict"], "own100"
        elif x["key"] in ownm:
            v, src = ownm[x["key"]]["verdict"], "own100m"
        elif x["key"] in carried:
            v, src = carried[x["key"]].get("verdict"), f"record ({carried[x['key']].get('confidence')})"
        else:
            v, src = None, "none"
        owner = src in ("own100", "own100m") or (src.startswith("record") and "owner" in src)
        tally[(cls(v) if owner else "not owner-judged")] += 1
        if not owner:
            unj.append((x, v, src, d, r["defer"].get(c)))
        print(f"  {x['key']:16s} {x['vol']:6s} tick {r['defer'].get(c) or float('nan'):7.1f} michel_ke {float(d.get('michel_ke_best', float('nan'))):5.1f} "
              f"| {src:18s} {v or '-':10s} {cls(v) if v else ''}")
    print(f"  owner-judged gained: {dict(tally)}")
    st, ns = tally["stopper"], tally["non-stopper"]
    print(f"  gained purity (owner-judged, excluded left out): {st}/{st + ns} = {st / max(1, st + ns):.3f}")
    with open(a.unjudged_out, "w") as fh:
        fh.write("key\tvol\tstop_tick\tmichel_ke\trecord_verdict\trecord_source\n")
        for x, v, src, d, t in unj:
            fh.write(f"{x['key']}\t{x['vol']}\t{t}\t{float(d.get('michel_ke_best', float('nan'))):.1f}\t{v or ''}\t{src}\n")
    print(f"  not owner-judged: {len(unj)} -> {a.unjudged_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc qlmatch/33 sec 4 -- the round-2 blind-scan TARGET: items over a FROZEN arm set, light-neutral sheets.

Population (pre-registered in d33/prereg.md): the union, over the frozen arms, of long clusters (the scorer's filter)
of the 18 run-039252 events whose auto-selected flash set differs from the control arm's after the arm's time map
(ToT-light arms: doc 32 time_map_ctl_to_q32ti.json; control-light arms: identity).  Every truth lives in the CONTROL
frame (control-arm flash times); an arm is scored by mapping its own picks back into that frame.

Per mover: LOOKS independent sheets (different scanners, independent letter shuffles) rendered from the CONTROL dump
with d33_blind_sheet.render_item (railed values hidden => identical for every light).  Candidates (<= MAX_CAND): the
flashes any frozen arm auto-selected for the cluster (mapped to the control frame), the owner's objective verdict
flashes, then the remaining bundles by |log(pred/meas)|.  Co-matched clusters per candidate flash: the clusters
EVERY frozen arm (and the control) auto-selects on that physical flash -- an arm-neutral intersection.
Calibration: CALIB_N owner-judged non-movers (one look).  Duplicates: DUP_FRAC of mover looks re-emitted to a third
scanner.

    python3 d33_scan_items.py --work-root /home/xqian/tmp/p31/wroot --ctl q31ctl \
        --arms q32ti:T,q33ts:T,q33tm:T,q33cs:C,q33cm:C --time-map ../d32/time_map_ctl_to_q32ti.json \
        --sheet-dir /home/xqian/tmp/p33/scan/r2 --key /home/xqian/tmp/p33/scan_key_r2/key.json [--dry-run]
"""
import argparse
import json
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../../ql_display"))
import ql_agree_score as S                                        # noqa: E402
from d32_scan_items import (RUN, EVT0, STEP, NEVT, calib_path, load, long_uids, auto_times,   # noqa: E402
                            railed_cath)
from d33_blind_sheet import render_item                           # noqa: E402

SEED = 33
CALIB_N = 40
GOLD_EVT = 298567
GOLD_PER_CLASS = 6
NWAVE = 9            # primary scanner waves (agents 1-9); agent 10 is reserved for adjudication
LOOKS = 2
DUP_FRAC = 0.05
MAX_CAND = 5
MAX_MOVERS = 360     # pre-registered sizing cap (d33/prereg.md)
TOL = 0.5


def mapper(pairs):
    fwd = lambda t: next((t1 for t0, t1 in pairs if abs(t - t0) <= 1e-3), t)
    inv = lambda t: next((t0 for t0, t1 in pairs if abs(t - t1) <= 1e-3), t)
    return fwd, inv


def inv_all(t, pairs):
    """ToT-frame time -> every control-frame time mapped onto it (itself if unmapped)."""
    xs = [t0 for t0, t1 in pairs if abs(t - t1) <= 1e-3]
    return xs if xs else [t]


def tset(ts):
    return sorted({round(t, 3) for t in ts})


def same(a, b):
    return len(a) == len(b) and all(abs(x - y) <= TOL for x, y in zip(a, b))


def build(a):
    """-> movers, calib_pool, co (per event: ctl-frame flash time -> set of co-matched uids)"""
    arms = [tuple(x.split(":")) for x in a.arms.split(",")]
    tmap = load(a.time_map)["events"]
    truth = S.load_truth(os.path.join(HERE, "../../../work/ql_labels/wfresc/labels-evt298567.json"),
                         os.path.join(HERE, "../../../ql_display/decisions-cathxa"))
    movers, calib_pool, co_all = [], {"railed": [], "unrailed": []}, {}
    for idx in range(NEVT):
        evt = EVT0 + STEP * idx
        pc = calib_path(a.work_root, idx, a.ctl)
        C = load(pc)
        fwd, inv = mapper(tmap.get(str(evt), []))
        ac = auto_times(C)
        longs = long_uids(C)
        pairs = tmap.get(str(evt), [])
        arm_auto, arm_own, fwd_of = {}, {}, {}
        for tag, light in arms:
            A = load(calib_path(a.work_root, idx, tag))
            longs &= long_uids(A)
            at = auto_times(A)
            arm_own[tag] = at                                     # the arm's picks in ITS OWN light frame
            fwd_of[tag] = fwd if light == "T" else (lambda t: t)
            # ... and in the CONTROL frame (a ToT flash that absorbed several control flashes maps to all of them)
            arm_auto[tag] = {u: [x for t in ts for x in (inv_all(t, pairs) if light == "T" else [t])]
                             for u, ts in at.items()}
        # arm-neutral co-matched sets: (flash time in ctl frame) -> uids auto in ctl AND every arm
        co = defaultdict(set)
        for u, ts in ac.items():
            for t in ts:
                if all(any(abs(fwd_of[tag](t) - x) <= TOL for x in arm_own[tag].get(u, [])) for tag, _ in arms):
                    co[round(t, 3)].add(u)
        co_all[evt] = co
        umap = S.build_uid_map(calib_path(a.work_root, idx, "keep"), pc)
        own = defaultdict(list)
        for e in truth[evt]:
            if e["uid"] in umap and e["conf"] in S.OBJECTIVE_TIERS:
                own[umap[e["uid"]]].append(e)
        for uid in sorted(longs):
            # mover test in the arm's own frame (control picks mapped forward; the doc 32 definition)
            moved = [tag for tag, _ in arms
                     if not same(tset(fwd_of[tag](t) for t in ac.get(uid, [])), tset(arm_own[tag].get(uid, [])))]
            ot = [e["time"] for e in own.get(uid, [])]
            owner = [dict(time=e["time"], positive=e["positive"], conf=e["conf"]) for e in own.get(uid, [])]
            if moved:
                keys = list(ac.get(uid, []))
                for tag, _ in arms:
                    keys += arm_auto[tag].get(uid, [])
                movers.append(dict(evt=evt, idx=idx, uid=uid, moved_in=moved, ctl_auto=ac.get(uid, []),
                                   arm_auto={tag: arm_auto[tag].get(uid, []) for tag, _ in arms},
                                   owner=owner, keys=keys + ot))
            elif own.get(uid):
                ftimes = {round(f["time"], 3): f for f in C["flashes"]}
                railed = any(railed_cath(f) for t in ot for tt, f in ftimes.items() if abs(tt - t) <= TOL)
                calib_pool["railed" if railed else "unrailed"].append(
                    dict(evt=evt, idx=idx, uid=uid, ctl_auto=ac.get(uid, []), owner=owner,
                         keys=list(ac.get(uid, [])) + ot))
    return arms, movers, calib_pool, co_all


def pick_candidates(C, uid, keys):
    fl = {f["gid"]: f for f in C["flashes"]}
    mine = [i for i, b in enumerate(C["bundles"]) if b["main_cluster"] == uid]
    t = lambda i: fl[C["bundles"][i]["flash_gid"]]["time"]
    keyed = []
    for k in keys:                                  # key order = ctl autos, arm autos, owner
        for i in mine:
            if abs(t(i) - k) <= TOL and i not in keyed:
                keyed.append(i)
    missing = [k for k in keys if not any(abs(t(i) - k) <= TOL for i in mine)]

    def closeness(i):
        b = C["bundles"][i]
        return abs(math.log(max(b["total_pred_light"], 1e-3) / max(fl[b["flash_gid"]]["total_PE"], 1e-3)))
    rest = sorted((i for i in mine if i not in keyed), key=closeness)
    return (keyed + rest)[:MAX_CAND], len(keyed) > MAX_CAND, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--arms", required=True, help="tag:T|C,...  (T = ToT light, mapped; C = control light)")
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--sheet-dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    for p in (a.sheet_dir, os.path.dirname(a.key)):
        if os.path.exists(p) and os.listdir(p) and not a.dry_run:
            sys.exit(f"refusing to write into non-empty {p} (M13: new round => new dir)")
    rng = random.Random(SEED)
    arms, movers, calib_pool, co_all = build(a)
    print(f"movers (union over {[t for t, _ in arms]}): {len(movers)}")
    print("  movers per arm:", dict(Counter(t for m in movers for t in m["moved_in"])))
    print("  movers per event parity: odd", sum(m["idx"] % 2 for m in movers), "even",
          sum(1 - m["idx"] % 2 for m in movers))
    print(f"calibration pool railed {len(calib_pool['railed'])} unrailed {len(calib_pool['unrailed'])}")
    if len(movers) > MAX_MOVERS:
        # pre-registered strata: arms in the order given (first = highest priority)
        order = [t for t, _ in arms]
        rank = lambda m: min(order.index(t) for t in m["moved_in"])
        keep = []
        for r in range(len(order)):
            stratum = [m for m in movers if rank(m) == r]
            room = MAX_MOVERS - len(keep)
            keep += stratum if len(stratum) <= room else rng.sample(stratum, room)
        print(f"  SIZING CAP: kept {len(keep)} of {len(movers)} by stratum")
        movers = sorted(keep, key=lambda m: (m["evt"], m["uid"]))
    cal = []
    for cls, n in (("railed", CALIB_N // 2), ("unrailed", CALIB_N - CALIB_N // 2)):
        pool = calib_pool[cls]
        gold = [m for m in pool if m["evt"] == GOLD_EVT]
        pick = rng.sample(gold, min(GOLD_PER_CLASS, len(gold)))
        rest = [m for m in pool if m not in pick]
        cal += pick + rng.sample(rest, min(n - len(pick), len(rest)))
    for m in cal:
        m["owner_source"] = "gold (owner)" if m["evt"] == GOLD_EVT else "cathxa (AI+owner)"
    sheets = []
    wload = [0] * NWAVE

    def least(excl):
        w = min((k for k in range(NWAVE) if k not in excl), key=lambda k: (wload[k], rng.random()))
        wload[w] += 1
        return w
    for m in movers:
        used = set()
        for look in range(LOOKS):
            w = least(used)
            used.add(w)
            sheets.append(dict(kind="mover", look=look, wave=w, **m))
    for m in cal:
        sheets.append(dict(kind="calib", look=0, wave=least(set()), **m))
    mover_sheets = [s for s in sheets if s["kind"] == "mover"]
    ndup = int(round(DUP_FRAC * len(mover_sheets)))
    for s in rng.sample(mover_sheets, ndup):
        used = {x["wave"] for x in mover_sheets if (x["evt"], x["uid"]) == (s["evt"], s["uid"])}
        sheets.append(dict(s, kind="dup", dup_of_look=s["look"], wave=least(used)))
    ids = rng.sample(range(10000, 99999), len(sheets))
    for s, i in zip(sheets, ids):
        s["id"] = f"s{i}"
        s["shuffle_seed"] = rng.randrange(1 << 30)
    rng.shuffle(sheets)
    print("wave loads", wload)
    print(f"sheets: {len(sheets)} (mover looks {len(mover_sheets)}, calib {len(cal)}, dup {ndup})")
    if a.dry_run:
        return
    os.makedirs(os.path.dirname(a.key), exist_ok=True)
    for w in range(NWAVE):
        os.makedirs(os.path.join(a.sheet_dir, f"wave{w}"), exist_ok=True)
    tmap_ev = load(a.time_map)["events"]
    jobs = [(s, calib_path(a.work_root, s["idx"], a.ctl), a.sheet_dir,
             {round(t, 3): sorted(u) for t, u in co_all[s["evt"]].items()}, arms, a.work_root,
             tmap_ev.get(str(s["evt"]), [])) for s in sheets]
    with multiprocessing.Pool(a.jobs) as pool:
        key = pool.map(render_one, jobs, chunksize=1)
    with open(a.key, "w") as fh:
        json.dump(dict(seed=SEED, ctl=a.ctl, arms=arms, time_map=a.time_map, sheets=key), fh, indent=1)
    for w in range(NWAVE):
        with open(os.path.join(a.sheet_dir, f"wave{w}", "INDEX.md"), "w") as fh:
            fh.write(f"# doc qlmatch/33 blind scan, round 2, wave {w} -- scanner index\n\nblind=true.  One sheet "
                     "per id; each sheet shows one cluster and up to 5 candidate flashes, lettered in RANDOM order.  "
                     "Railed channels are drawn hatched with no value on every sheet.\n\n")
            for s in sorted((s for s in key if s["wave"] == w and s["letters"]), key=lambda s: s["id"]):
                fh.write(f"- {s['id']}  candidates {''.join(sorted(s['letters']))}\n")
    nskip = sum(1 for s in key if not s["letters"])
    print(f"key -> {a.key}; skipped (no bundle) {nskip}; key overflow {sum(1 for s in key if s.get('key_overflow'))}; "
          f"keys drawn from another arm's dump {sum(len(s.get('fallback', [])) for s in key)}; keys not drawn "
          f"{sum(len(s.get('keys_missing', [])) for s in key)} (arm picks among them "
          f"{sum(len(s.get('arm_picks_not_drawn', [])) for s in key)})")


def render_one(job):
    s, p, sheet_dir, co, arms, work_root, pairs = job
    C = load(p)
    bidx, over, missing = pick_candidates(C, s["uid"], s["keys"])
    fwd, _ = mapper(pairs)
    # an arm pick whose flash has no bundle for this cluster in the control dump is drawn from the dump of an arm
    # that has it (control-light arms first: identical light); its letter keeps the CONTROL-frame time
    cands = [(p, i, None) for i in bidx]
    fb, unresolved = [], []
    for t in missing:
        tags = [tag for tag, lt in sorted(arms, key=lambda x: x[1] != "C")
                if any(abs(t - x) <= TOL for x in s.get("arm_auto", {}).get(tag, []))]
        got = None
        for tag in tags:
            lt = dict(arms)[tag]
            pa = calib_path(work_root, s["idx"], tag)
            A = load(pa)
            fla = {f["gid"]: f for f in A["flashes"]}
            ta = fwd(t) if lt == "T" else t
            js = [j for j, b in enumerate(A["bundles"])
                  if b["main_cluster"] == s["uid"] and abs(fla[b["flash_gid"]]["time"] - ta) <= TOL]
            if js:
                got = (pa, js[0], t, tag)
                break
        if got:
            if not any(g[2] is not None and abs(g[2] - t) <= TOL for g in fb):
                fb.append(got)
        elif tags:
            unresolved.append(t)
    keyed = [c for c in cands if any(abs(_t(C, c[1]) - k) <= TOL for k in s["keys"])]
    rest = [c for c in cands if c not in keyed]
    allc = (keyed + [(g[0], g[1], g[2]) for g in fb] + rest)[:MAX_CAND]
    s["key_overflow"] = over or len(keyed) + len(fb) > MAX_CAND
    s["keys_missing"] = [t for t in missing if t not in [g[2] for g in fb]]
    s["arm_picks_not_drawn"] = unresolved
    s["fallback"] = [dict(time=g[2], tag=g[3]) for g in fb]
    if not allc:
        s["letters"] = {}
        s["skipped"] = "no bundle for this cluster"
        return s
    random.Random(s["shuffle_seed"]).shuffle(allc)
    entries, cob, tctl = [], {}, {}
    for src, j, t in allc:
        D = C if src == p else load(src)
        fl = {f["gid"]: f for f in D["flashes"]}
        g = D["bundles"][j]["flash_gid"]
        tc = t if t is not None else fl[g]["time"]
        e = j if src == p else (src, j)
        entries.append(e)
        tctl[e] = tc
        uids = next((u for tt, u in co.items() if abs(tt - tc) <= TOL), [])
        on = {}
        for jj, b in enumerate(D["bundles"]):
            if b["flash_gid"] == g:
                on.setdefault(b["main_cluster"], jj)
        cob[e] = [on[u] for u in uids if u != s["uid"] and u in on]
    wd = os.path.join(sheet_dir, f"wave{s['wave']}")
    letters = render_item(p, s["uid"], entries, cob, s["id"], os.path.join(wd, f"{s['id']}.png"))
    for L, e in zip(sorted(letters), entries):
        letters[L]["time_own_dump"] = letters[L]["time"]
        letters[L]["time"] = tctl[e]
        letters[L]["co_matched_n"] = len(cob[e])
    s["letters"] = letters
    return s


def _t(C, j):
    g = C["bundles"][j]["flash_gid"]
    return next(f["time"] for f in C["flashes"] if f["gid"] == g)


if __name__ == "__main__":
    main()

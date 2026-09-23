#!/usr/bin/env python3
"""doc qlmatch/32 sec 4 -- build the blind-scan item set and render the sheets.

Items (pre-registered in d32/prereg.md):
  mover  -- a long cluster (>= 25 cm or >= 100 points, the scorer's filter) of the 18 run-039252 events whose set of
            auto-selected flashes differs between the control arm (twoside light) and the candidate arm (ToT light),
            after translating control flash times into the candidate light with the doc 31 time map (a retimed or
            absorbed flash is the same physical flash).  Only movers can change the ctl-vs-candidate comparison: a
            cluster both arms treat alike scores alike under any truth.  Rendered in BOTH lights.
  calib  -- CALIB_N owner-judged clusters that are NOT movers, drawn with a fixed seed, half with an owner verdict on a
            cathode-railed flash (hard) and half on unrailed flashes, rendered in the CONTROL light (the light the owner
            judged), to measure scanner-vs-owner agreement on the same evidence.
  dup    -- DUP_FRAC of the mover sheets rendered a second time under a new id, for scanner-to-scanner consistency.
Candidate flashes per sheet (<= 6, drawn in time order): the cluster's bundles on the flashes that either arm
auto-selected, the owner's verdict flashes, then the remaining bundles by |log(pred/meas)| (a charge-light amplitude
closeness that says nothing about which one the matcher chose).

Outputs: sheets (PNG) to --sheet-dir, named by random id only; the KEY (id -> event, uid, light, candidate letters ->
flash times, owner verdicts) to --key, which scanners must never read; the scanner-visible INDEX.md per wave dir.

    cd pdvd/docs/qlmatch/scripts && python3 d32_scan_items.py --work-root /home/xqian/tmp/p31/wroot \
        --ctl q31ctl --cand q32ti --time-map ../d32/time_map_ctl_to_q32ti.json \
        --sheet-dir /home/xqian/tmp/p32/scan/r1q --key /home/xqian/tmp/p32/scan_key_r1q/key.json \
"""
import argparse
import json
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../../ql_display"))
import ql_agree_score as S                       # noqa: E402
from d32_blind_sheet import render_item          # noqa: E402

RUN, EVT0, STEP, NEVT = "039252", 298567, 14, 18
CATH = range(4, 12)
SEED = 32
CALIB_N = 40
GOLD_EVT = 298567      # the owner's own scan (wfresc gold); other events are the cathxa AI scan
GOLD_PER_CLASS = 6
NWAVE = 5
DUP_FRAC = 0.10
MAX_CAND = 6
TOL = 0.5


def calib_path(root, idx, tag):
    return os.path.join(root, f"{RUN}_{idx}_{tag}", f"calib-evt{EVT0 + STEP * idx}.json")


def load(p):
    with open(p) as fh:
        return json.load(fh)


def long_uids(calib):
    return {c["uid"] for c in calib["clusters"] if S.cluster_len_cm(c) >= 25.0 or c["npoints"] >= 100}


def auto_times(calib):
    fl = {f["gid"]: f for f in calib["flashes"]}
    out = defaultdict(list)
    for b in calib["bundles"]:
        if b.get("auto_selected"):
            out[b["main_cluster"]].append(fl[b["flash_gid"]]["time"])
    return out


def railed_cath(f):
    s = f.get("sat", [])
    return any(s[k] > 0 for k in CATH if k < len(s))


def pick_candidates(calib, uid, key_times):
    """Bundle indices of `uid` (as main cluster) to draw: key-time flashes first, then amplitude-closest."""
    fl = {f["gid"]: f for f in calib["flashes"]}
    mine = [i for i, b in enumerate(calib["bundles"]) if b["main_cluster"] == uid]
    t = lambda i: fl[calib["bundles"][i]["flash_gid"]]["time"]
    keyed = [i for i in mine if any(abs(t(i) - k) <= TOL for k in key_times)]

    def closeness(i):
        b = calib["bundles"][i]
        m = max(fl[b["flash_gid"]]["total_PE"], 1e-3)
        return abs(math.log(max(b["total_pred_light"], 1e-3) / m))
    rest = sorted((i for i in mine if i not in keyed), key=closeness)
    return (keyed + rest)[:MAX_CAND], len(keyed) > MAX_CAND


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--cand", default="q32ti")
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--sheet-dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    for p in (a.sheet_dir, os.path.dirname(a.key)):
        if os.path.exists(p) and os.listdir(p) and not a.dry_run:
            sys.exit(f"refusing to write into non-empty {p} (M13: new round => new dir)")
    rng = random.Random(SEED)
    truth = S.load_truth(os.path.join(HERE, "../../../work/ql_labels/wfresc/labels-evt298567.json"),
                         os.path.join(HERE, "../../../ql_display/decisions-cathxa"))
    tmap = load(a.time_map)["events"]

    movers, calib_pool = [], {"railed": [], "unrailed": []}
    for idx in range(NEVT):
        evt = EVT0 + STEP * idx
        pc, pt = calib_path(a.work_root, idx, a.ctl), calib_path(a.work_root, idx, a.cand)
        C, T = load(pc), load(pt)
        umap = S.build_uid_map(calib_path(a.work_root, idx, "keep"), pc)
        own = defaultdict(list)
        for e in truth[evt]:
            if e["uid"] in umap and e["conf"] in S.OBJECTIVE_TIERS:
                own[umap[e["uid"]]].append(e)
        pairs = tmap.get(str(evt), [])

        def to_cand(t):
            for t0, t1 in pairs:
                if abs(t - t0) <= 1e-3:
                    return t1
            return t
        ac, at = auto_times(C), auto_times(T)
        flc = {f["gid"]: f for f in C["flashes"]}
        for uid in sorted(long_uids(C) & long_uids(T)):
            sc = sorted({round(to_cand(t), 3) for t in ac.get(uid, [])})
            st = sorted({round(t, 3) for t in at.get(uid, [])})
            same = len(sc) == len(st) and all(abs(x - y) <= TOL for x, y in zip(sc, st))
            ot = [e["time"] for e in own.get(uid, [])]
            if not same:
                movers.append(dict(evt=evt, idx=idx, uid=uid, ctl_auto=ac.get(uid, []), cand_auto=at.get(uid, []),
                                   owner=[dict(time=e["time"], positive=e["positive"], conf=e["conf"])
                                          for e in own.get(uid, [])],
                                   key_ctl=ac.get(uid, []) + [x for x in (to_inv(t, pairs) for t in at.get(uid, []))
                                                              if x is not None] + ot,
                                   key_cand=at.get(uid, []) + [to_cand(t) for t in ac.get(uid, []) + ot]))
            elif own.get(uid):
                # owner-judged non-mover: calibration pool, classed by the railed state of its verdict flashes
                ftimes = {round(f["time"], 3): f for f in C["flashes"]}
                railed = any(railed_cath(f) for t in ot for tt, f in ftimes.items() if abs(tt - t) <= TOL)
                calib_pool["railed" if railed else "unrailed"].append(
                    dict(evt=evt, idx=idx, uid=uid, ctl_auto=ac.get(uid, []),
                         owner=[dict(time=e["time"], positive=e["positive"], conf=e["conf"]) for e in own[uid]],
                         key_ctl=ac.get(uid, []) + ot))
    print(f"movers {len(movers)}; calibration pool railed {len(calib_pool['railed'])} unrailed "
          f"{len(calib_pool['unrailed'])}")
    cal = []
    for cls, n in (("railed", CALIB_N // 2), ("unrailed", CALIB_N - CALIB_N // 2)):
        pool = calib_pool[cls]
        gold = [m for m in pool if m["evt"] == GOLD_EVT]
        ng = min(GOLD_PER_CLASS, len(gold))
        pick = rng.sample(gold, ng)
        rest = [m for m in pool if m not in pick]
        cal += pick + rng.sample(rest, min(n - ng, len(rest)))
    for m in cal:
        m["owner_source"] = "gold (owner)" if m["evt"] == GOLD_EVT else "cathxa (AI+owner)"
    sheets = []
    for m in movers:
        sheets.append(dict(kind="mover", light="ctl", tag=a.ctl, keys=m["key_ctl"], **m))
        sheets.append(dict(kind="mover", light="cand", tag=a.cand, keys=m["key_cand"], **m))
    for m in cal:
        sheets.append(dict(kind="calib", light="ctl", tag=a.ctl, keys=m["key_ctl"], **m))
    # waves: the two lights of one mover go to DIFFERENT scanners; a duplicate goes to a third scanner
    wload = [0] * NWAVE
    def least(excl):
        w = min((k for k in range(NWAVE) if k not in excl), key=lambda k: (wload[k], rng.random()))
        wload[w] += 1
        return w
    by_mover = defaultdict(list)
    for s in sheets:
        if s["kind"] == "mover":
            by_mover[(s["evt"], s["uid"])].append(s)
    for pair in by_mover.values():
        rng.shuffle(pair)
        used = set()
        for s in pair:
            s["wave"] = least(used)
            used.add(s["wave"])
    for s in sheets:
        if s["kind"] == "calib":
            s["wave"] = least(set())
    ndup = int(round(DUP_FRAC * 2 * len(movers)))
    for s in rng.sample([s for s in sheets if s["kind"] == "mover"], ndup):
        pair_waves = {x["wave"] for x in by_mover[(s["evt"], s["uid"])]}
        sheets.append(dict(s, kind="dup", dup_of_wave=s["wave"], wave=least(pair_waves)))
    ids = rng.sample(range(10000, 99999), len(sheets))
    for s, i in zip(sheets, ids):
        s["id"] = f"s{i}"
    rng.shuffle(sheets)
    print("wave loads", wload)
    print(f"sheets: {len(sheets)} (mover {2 * len(movers)}, calib {len(cal)}, dup {ndup})")
    if a.dry_run:
        return
    os.makedirs(a.sheet_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.key), exist_ok=True)
    for w in range(NWAVE):
        os.makedirs(os.path.join(a.sheet_dir, f"wave{w}"), exist_ok=True)
    jobs = [(s, calib_path(a.work_root, s["idx"], s["tag"]), a.sheet_dir) for s in sheets]
    with multiprocessing.Pool(a.jobs) as pool:
        key = pool.map(render_one, jobs, chunksize=1)
    with open(a.key, "w") as fh:
        json.dump(dict(seed=SEED, ctl=a.ctl, cand=a.cand, sheets=key), fh, indent=1)
    for w in range(NWAVE):
        with open(os.path.join(a.sheet_dir, f"wave{w}", "INDEX.md"), "w") as fh:
            fh.write(f"# doc qlmatch/32 blind scan, round 1, wave {w} -- scanner index\n\nblind=true.  One sheet "
                     "per id; each sheet shows one cluster and up to 6 candidate flashes (A, B, ... by flash time).  "
                     "Known leak: two light reconstructions are mixed in this set, so railed (R) channels read "
                     "higher in some sheets than in others.\n\n")
            for s in sorted((s for s in key if s["wave"] == w and s["letters"]), key=lambda s: s["id"]):
                fh.write(f"- {s['id']}  candidates {''.join(sorted(s['letters']))}\n")
    print(f"key -> {a.key}; per-wave INDEX.md under {a.sheet_dir}")


def render_one(job):
    s, p, sheet_dir = job
    C = load(p)
    bidx, over = pick_candidates(C, s["uid"], s["keys"])
    if not bidx:
        s["letters"] = {}
        s["skipped"] = "no bundle for this cluster"
        return s
    wd = os.path.join(sheet_dir, f"wave{s['wave']}")
    s["letters"] = render_item(p, s["uid"], bidx, s["id"], os.path.join(wd, f"{s['id']}.png"))
    s["key_overflow"] = over
    return s


def to_inv(t, pairs):
    """candidate-light time -> control-light time (first pair whose B time matches), else t itself."""
    for t0, t1 in pairs:
        if abs(t - t1) <= 1e-3:
            return t0
    return t


if __name__ == "__main__":
    main()

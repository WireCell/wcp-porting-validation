#!/usr/bin/env python3
"""doc qlmatch/33 sec 5 -- adjudication wave: a third look (new id, fresh letter shuffle) at every mover whose two
committed looks contradict (d33_merge_record.py --adjudicate-out).  Written to <scan-root>/wave<W>/ and a SEPARATE key
file (the primary key is never rewritten).

    python3 d33_adjudicate_items.py --key /home/xqian/tmp/p33/scan_key_r2d/key.json \
        --contradicted ../d33/scan_r2/contradicted.json --work-root /home/xqian/tmp/p31/wroot \
        --time-map ../d32/time_map_ctl_to_q32ti.json --scan-root /home/xqian/tmp/p33/scan/r2d --wave 9 \
        --out-key /home/xqian/tmp/p33/scan_key_r2d/key_adj.json
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d33_scan_items as I                        # noqa: E402
from d32_scan_items import calib_path, load       # noqa: E402

SEED = 3309


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--contradicted", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--scan-root", required=True)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--out-key", required=True)
    a = ap.parse_args()
    K = json.load(open(a.key))
    wd = os.path.join(a.scan_root, f"wave{a.wave}")
    if os.path.exists(a.out_key) or (os.path.exists(wd) and os.listdir(wd)):
        sys.exit("refusing to overwrite an existing adjudication key/wave (M13)")
    os.makedirs(wd, exist_ok=True)
    rng = random.Random(SEED)
    arms = [tuple(x) for x in K["arms"]]
    tmap = load(a.time_map)["events"]
    used = {s["id"] for s in K["sheets"]}
    # the co-matched sets are rebuilt exactly as the primary render did
    _, _, _, co_all = I.build(argparse.Namespace(arms=",".join(f"{t}:{l}" for t, l in arms), time_map=a.time_map,
                                                 work_root=a.work_root, ctl=K["ctl"]))
    out = []
    for c in json.load(open(a.contradicted)):
        base = next(s for s in K["sheets"] if s["id"] in c["looks"])
        s = {k: v for k, v in base.items() if k not in ("letters", "id", "wave", "look", "shuffle_seed")}
        s.update(kind="adj", wave=a.wave, look=2, shuffle_seed=rng.randrange(1 << 30))
        while True:
            sid = f"s{rng.randrange(10000, 99999)}"
            if sid not in used:
                break
        used.add(sid)
        s["id"] = sid
        co = {round(t, 3): sorted(u) for t, u in co_all[s["evt"]].items()}
        out.append(I.render_one((s, calib_path(a.work_root, s["idx"], K["ctl"]), a.scan_root, co, arms, a.work_root,
                                 tmap.get(str(s["evt"]), []))))
    json.dump(dict(seed=SEED, of_key=a.key, sheets=out), open(a.out_key, "w"), indent=1)
    with open(os.path.join(wd, "INDEX.md"), "w") as fh:
        fh.write(f"# doc qlmatch/33 blind scan, round 2, wave {a.wave} (adjudication) -- scanner index\n\nblind=true.  "
                 "One sheet per id; each sheet shows one cluster and up to 5 candidate flashes, lettered in RANDOM "
                 "order.  Railed channels are drawn hatched with no value on every sheet.\n\n")
        for s in sorted(out, key=lambda s: s["id"]):
            fh.write(f"- {s['id']}  candidates {''.join(sorted(s['letters']))}\n")
    print(f"{len(out)} adjudication sheets -> {wd}; key -> {a.out_key}")


if __name__ == "__main__":
    main()

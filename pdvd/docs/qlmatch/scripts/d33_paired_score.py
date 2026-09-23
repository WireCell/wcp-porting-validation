#!/usr/bin/env python3
"""doc qlmatch/33 sec 6 -- score the frozen arms against the round-2 target (rules: d33/prereg.md).

PRIMARY (paired): per resolved non-tie mover, an arm is RIGHT if its auto-selected flash set, in its own light frame,
is exactly the truth -- one flash within 0.5 us of the mapped truth time for `pos`, empty for `none`.  Each arm is
compared with the control arm: wins (arm right, control wrong) vs losses (control right, arm wrong), two-sided sign
test.  Reported for odd events (tuning half), even events (confirmation half) and all.
SECONDARY (mover-level pairs): agree (auto on the truth flash), phantom (auto elsewhere), missed (truth flash not
auto); ties and unresolved dropped in every arm alike.  Also writes ql_agree_score.py --override-truth files per arm
(the target replaces the owner entries of every scanned mover; unresolved/tie movers are dropped).

    python3 d33_paired_score.py --work-root /home/xqian/tmp/p31/wroot --ctl q31ctl --arms q32ti:T,... \
        --time-map ../d32/time_map_ctl_to_q32ti.json --truth ../d33/scan_r2/truth.jsonl --key <key.json> \
        --out ../d33/scan_r2
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d32_scan_items import calib_path, load, auto_times, EVT0, STEP   # noqa: E402

TOL = 0.5


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-extra")
    a = ap.parse_args()
    arms = [("ctl:" + a.ctl, "C")] + [tuple(x.split(":")) for x in a.arms.split(",")]
    tmap = load(a.time_map)["events"]
    truth = [json.loads(ln) for ln in open(a.truth) if ln.strip()]
    key = json.load(open(a.key))["sheets"]
    if a.key_extra:
        key += json.load(open(a.key_extra))["sheets"]
    drawn = defaultdict(set)                         # (evt, uid) -> control-frame candidate times drawn
    movers = set()
    for s in key:
        if s["kind"] in ("mover", "adj"):
            movers.add((s["evt"], s["uid"]))
            for x in s["letters"].values():
                drawn[(s["evt"], s["uid"])].add(round(x["time"], 3))
    tr = {(e["evt"], e["uid"]): e for e in truth}
    auto = {}
    for tag, light in arms:
        t = tag.split(":")[-1]
        for idx in range(18):
            evt = EVT0 + STEP * idx
            auto[(tag, evt)] = auto_times(load(calib_path(a.work_root, idx, t)))
    right = defaultdict(dict)                        # tag -> (evt, uid) -> bool
    sec = {tag: defaultdict(lambda: [0, 0, 0]) for tag, _ in arms}   # tag -> parity -> [agree, phantom, missed]
    lines = []
    L = lines.append
    for tag, light in arms:
        ov = []
        for (evt, uid) in sorted(movers):
            pairs = tmap.get(str(evt), [])
            fwd = (lambda t: next((t1 for t0, t1 in pairs if abs(t - t0) <= 1e-3), t)) if light == "T" \
                else (lambda t: t)
            picks = sorted({round(x, 3) for x in auto[(tag, evt)].get(uid, [])})
            e = tr.get((evt, uid))
            if e is None or e["kind"] == "tie":
                ov.append(dict(event=evt, uid=uid, time=None, positive=False, conf="med"))
                continue
            par = "odd" if e["idx"] % 2 else "even"
            if e["kind"] == "none":
                ok = not picks
                tt = None
                ph, ag, mi = len(picks), 0, 0
            else:
                tt = fwd(e["times"][0])
                ok = len(picks) == 1 and abs(picks[0] - tt) <= TOL
                ag = int(any(abs(p - tt) <= TOL for p in picks))
                ph = sum(1 for p in picks if abs(p - tt) > TOL)
                mi = 1 - ag
            right[tag][(evt, uid)] = ok
            for k in (par, "all"):
                s = sec[tag][k]
                s[0] += ag
                s[1] += ph
                s[2] += mi
            for t0 in drawn[(evt, uid)]:
                t1 = fwd(t0)
                ov.append(dict(event=evt, uid=uid, time=t1, positive=tt is not None and abs(t1 - tt) <= TOL,
                               conf="med"))
        if not tag.startswith("ctl:"):
            with open(os.path.join(a.out, f"override_target_{tag}.jsonl"), "w") as fh:
                for x in ov:
                    fh.write(json.dumps(x) + "\n")
        else:
            with open(os.path.join(a.out, f"override_target_{a.ctl}.jsonl"), "w") as fh:
                for x in ov:
                    fh.write(json.dumps(x) + "\n")
    ctag = "ctl:" + a.ctl
    L(f"# PRIMARY: paired right/wrong vs {a.ctl} on resolved non-tie movers (truth {a.truth})")
    L(f"{'arm':>10} {'half':>5} {'n':>4} {'arm right':>9} {'ctl right':>9} {'wins':>5} {'losses':>6} {'net':>4} {'p':>6}")
    for tag, light in arms[1:]:
        for half in ("odd", "even", "all"):
            ks = [k for k in right[ctag] if half == "all" or (tr[k]["idx"] % 2 == (1 if half == "odd" else 0))]
            w = sum(1 for k in ks if right[tag][k] and not right[ctag][k])
            lo = sum(1 for k in ks if right[ctag][k] and not right[tag][k])
            L(f"{tag:>10} {half:>5} {len(ks):>4} {sum(right[tag][k] for k in ks):>9} "
              f"{sum(right[ctag][k] for k in ks):>9} {w:>5} {lo:>6} {w - lo:>+4} {sign_p(w, lo):>6.3f}")
    L("\n# SECONDARY: agree / phantom / missed on the resolved non-tie movers")
    for tag, _ in arms:
        L(f"{tag:>12}: " + "  ".join(f"{h} {sec[tag][h][0]}/{sec[tag][h][1]}/{sec[tag][h][2]}"
                                      for h in ("odd", "even", "all")))
    txt = "\n".join(lines)
    with open(os.path.join(a.out, "paired_score.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

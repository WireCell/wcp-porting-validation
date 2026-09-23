#!/usr/bin/env python3
"""doc qlmatch/32 sec 4-5 -- merge the blind-scan verdicts with the key into (1) the scanner calibration report,
(2) duplicate consistency, (3) the consensus truth of the movers, written as ql_agree_score.py --override-truth files
for the control and the candidate arm, and (4) the per-light (secondary) truth files.  Rules: d32/prereg.md.

    cd pdvd/docs/qlmatch/scripts && python3 d32_merge_record.py --key /home/xqian/tmp/p32/scan_key/r1p_key.json \
        --scan-root /home/xqian/tmp/p32/scan/r1p --time-map ../d32/time_map_ctl_to_q32ti.json --out ../d32/scan_r1
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

TOL = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--scan-root", required=True)
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = json.load(open(a.key))["sheets"]
    tmap = json.load(open(a.time_map))["events"]
    ver = {}
    for p in sorted(glob.glob(os.path.join(a.scan_root, "wave*", "verdicts.jsonl"))):
        for ln in open(p):
            if ln.strip():
                d = json.loads(ln)
                ver[d["id"]] = d
    os.makedirs(a.out, exist_ok=True)
    rep = []
    P = lambda *x: rep.append(" ".join(str(y) for y in x))
    shown = [s for s in key if s["letters"]]
    P(f"# sheets {len(key)} (rendered {len(shown)}); verdicts {len(ver)}; missing "
      f"{sum(1 for s in shown if s['id'] not in ver)}")
    P("# picks:", dict(Counter((ver[s['id']]['pick'] if ver[s['id']]['pick'] in ('none', 'unsure') else 'letter',
                                 ver[s['id']]['conf']) for s in shown if s['id'] in ver)))

    def decision(s):
        """-> ('pos', time) | ('none', None) | ('unsure', None)"""
        v = ver.get(s["id"])
        if v is None or v["pick"] == "unsure" or v["conf"] == "low":
            return ("unsure", None)
        if v["pick"] == "none":
            return ("none", None)
        return ("pos", s["letters"][v["pick"]]["time"])

    # ---- (1) calibration vs the owner record ----
    cal = defaultdict(Counter)
    cal_rows = []
    for s in shown:
        if s["kind"] != "calib":
            continue
        src = s["owner_source"]
        kind, t = decision(s)
        drawn = [x["time"] for x in s["letters"].values()]
        pos = [o["time"] for o in s["owner"] if o["positive"] and any(abs(o["time"] - d) <= TOL for d in drawn)]
        neg = [o["time"] for o in s["owner"] if not o["positive"]]
        if kind == "unsure":
            res = "unsure"
        elif kind == "none":
            res = "agree" if not pos else "disagree"
        elif pos:
            res = "agree" if any(abs(t - x) <= TOL for x in pos) else "disagree"
        elif any(abs(t - x) <= TOL for x in neg):
            res = "disagree"
        else:
            res = "not-comparable"
        cal[src][res] += 1
        cal["all"][res] += 1
        cal_rows.append((s["id"], s["evt"], s["uid"], src, kind, t, pos, neg, res))
    P("\n## scanner calibration vs the owner record (control light; agreement on judged candidates only)")
    for src in sorted(cal):
        c = cal[src]
        n = c["agree"] + c["disagree"]
        P(f"{src:>18}: agree {c['agree']} disagree {c['disagree']} -> {100 * c['agree'] / n if n else float('nan'):.1f} %"
          f"  (unsure {c['unsure']}, not-comparable {c['not-comparable']})")
    for r in cal_rows:
        if r[-1] == "disagree":
            P(f"   disagree: {r[0]} evt{r[1]} uid{r[2]} {r[3]}: scanner {r[4]} {r[5]} | owner pos {r[6]} neg {r[7]}")

    # ---- (2) duplicate consistency ----
    orig = {(s["evt"], s["uid"], s["light"], s["kind"]): s for s in shown if s["kind"] == "mover"}
    dc = Counter()
    for s in shown:
        if s["kind"] != "dup":
            continue
        o = orig.get((s["evt"], s["uid"], s["light"], "mover"))
        if not o:
            continue
        k1, k2 = decision(o), decision(s)
        if "unsure" in (k1[0], k2[0]):
            dc["one unsure"] += 1
        elif k1[0] == k2[0] and (k1[1] is None or abs(k1[1] - k2[1]) <= TOL):
            dc["same"] += 1
        else:
            dc["different"] += 1
    P("\n## duplicate-sheet consistency (different scanners, same sheet):", dict(dc))

    # ---- (3) consensus truth of movers ----
    movers = defaultdict(dict)
    for s in shown:
        if s["kind"] == "mover":
            movers[(s["evt"], s["uid"])][s["light"]] = s
    # movers with no rendered sheet in a light count as unresolved
    for s in key:
        if s["kind"] == "mover":
            movers.setdefault((s["evt"], s["uid"]), {})
    ov = {"ctl": [], "cand": []}
    per_light = {"ctl": [], "cand": []}
    outc = Counter()
    for (evt, uid), d in sorted(movers.items()):
        pairs = tmap.get(str(evt), [])
        fwd = lambda t: next((t1 for t0, t1 in pairs if abs(t - t0) <= 1e-3), t)
        dec = {lt: decision(d[lt]) if lt in d else ("unsure", None) for lt in ("ctl", "cand")}
        for lt in ("ctl", "cand"):                      # secondary: per-light truth
            if lt not in d:
                continue
            k, t = dec[lt]
            for x in d[lt]["letters"].values():
                if k == "pos":
                    per_light[lt].append(dict(event=evt, uid=uid, time=x["time"], positive=abs(x["time"] - t) <= 1e-6,
                                              conf="med"))
                elif k == "none":
                    per_light[lt].append(dict(event=evt, uid=uid, time=x["time"], positive=False, conf="med"))
            if k == "unsure":
                per_light[lt].append(dict(event=evt, uid=uid, time=None, positive=False, conf="med"))
        (kc, tc), (kt, tt) = dec["ctl"], dec["cand"]
        if kc == "pos" and kt == "pos" and abs(fwd(tc) - tt) <= TOL:
            res = "positive"
        elif kc == "none" and kt == "none":
            res = "none"
        else:
            res = "unresolved:" + ("unsure" if "unsure" in (kc, kt) else "disagree")
        outc[res] += 1
        for lt, t in (("ctl", tc), ("cand", tt)):
            if res == "positive":
                for x in d[lt]["letters"].values():
                    ov[lt].append(dict(event=evt, uid=uid, time=x["time"], positive=abs(x["time"] - t) <= 1e-6,
                                       conf="med"))
            elif res == "none":
                for x in d[lt]["letters"].values():
                    ov[lt].append(dict(event=evt, uid=uid, time=x["time"], positive=False, conf="med"))
            else:                                   # unresolved: drop the owner's entries, add none
                ov[lt].append(dict(event=evt, uid=uid, time=None, positive=False, conf="med"))
    nres = outc["positive"] + outc["none"]
    P(f"\n## consensus over {len(movers)} movers: {dict(outc)}; resolved {nres} = "
      f"{100 * nres / max(len(movers), 1):.1f} % (prereg minimum 60 %)")
    for lt in ("ctl", "cand"):
        with open(os.path.join(a.out, f"override_consensus_{lt}.jsonl"), "w") as fh:
            for e in ov[lt]:
                fh.write(json.dumps(e) + "\n")
        with open(os.path.join(a.out, f"override_perlight_{lt}.jsonl"), "w") as fh:
            for e in per_light[lt]:
                fh.write(json.dumps(e) + "\n")
    with open(os.path.join(a.out, "merge_report.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()

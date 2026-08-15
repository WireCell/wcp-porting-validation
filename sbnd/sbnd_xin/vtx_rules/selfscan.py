"""Blind self-scan harness: can an AI session scan events by EYE? (doc pr/80 sec 7)

`eval_rules.py` measures the *engine*.  This measures a different scanner: an
agent looking at rendered pictures and applying the §3 procedure with judgement,
the way the owner does.  There is no reason to assume the two score alike, so
the question is answered by measurement rather than by assertion.

Two rules make the number honest, and both are enforced here rather than trusted:

  * `prepare` NEVER prints or renders the truth, the reconstructed vertex, or
    the engine's pick.  A scanner that can see the reco star will anchor on it,
    and labels that only confirm the reconstruction are worse than no labels --
    they teach a future fit that the reconstruction is right on exactly the
    events where it is wrong.
  * `score` refuses to run until the picks file exists, so the picks are on disk
    before any truth is read.  The file is the record; self-grading after the
    fact is not a measurement.

  cd sbnd_xin
  python3 vtx_rules/selfscan.py prepare --n 20 --out /home/xqian/tmp/selfscan1
  # ... look at the PNGs, write picks.json ...
  python3 vtx_rules/selfscan.py score --dir /home/xqian/tmp/selfscan1
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines                                                 # noqa: E402
import make_split                                                # noqa: E402
import render_event                                              # noqa: E402
import vtx_io                                                    # noqa: E402


def _labels(half):
    keep = make_split.load_split()[half]
    labs = [L for L in vtx_io.load_labels() if L["key"] in keep]
    labs.sort(key=lambda L: (L["tag"], L.get("runNo") or -1,
                             L.get("subRunNo") or -1, L.get("eventNo") or -1))
    return labs


def prepare(half, n, out, seed):
    labs = _labels(half)
    rng = random.Random(seed)
    rng.shuffle(labs)
    labs = labs[:n]
    labs.sort(key=lambda L: L["event"])

    os.makedirs(out, exist_ok=True)
    manifest = []
    sheet = []
    for L in labs:
        path = baselines.deployed_dump_path(L) or L["source"]
        with open(path) as fh:
            dump = json.load(fh)
        png = os.path.join(out, "%s.png" % L["event"])
        render_event.render(dump, png, title=L["event"], blind=True)

        cid = vtx_io.main_cluster_id(dump)
        cands = []
        for v in dump.get("vertices", []):
            if v.get("cluster_id") != cid:
                continue
            p = vtx_io.vertex_xyz(v)
            if p is None:
                continue
            cands.append(dict(vertex_id=v["id"], degree=v.get("degree", 0),
                              x=round(p[0], 2), y=round(p[1], 2),
                              z=round(p[2], 2)))
        cands.sort(key=lambda c: c["vertex_id"])
        manifest.append(dict(event=L["event"], key=list(L["key"]),
                             dump=path, png=png, cluster_id=cid,
                             candidates=cands))
        sheet.append("%-12s cluster %-4s  %2d candidates: %s"
                     % (L["event"], cid, len(cands),
                        ", ".join("%d(z=%.0f)" % (c["vertex_id"], c["z"])
                                  for c in cands)))

    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(out, "worksheet.txt"), "w") as fh:
        fh.write("\n".join(sheet) + "\n")
    print("prepared %d blind events in %s" % (len(manifest), out))
    print("\n".join(sheet))
    print("\nNow write %s/picks.json as a list of:" % out)
    print('  {"event": "evtNNN", "vertex_id": 12345 | null,')
    print('   "confidence": "certain"|"likely"|"unclear", "why": "..."}')
    print("Use vertex_id null with confidence unclear to abstain.")
    return 0


def score(out):
    mpath = os.path.join(out, "manifest.json")
    ppath = os.path.join(out, "picks.json")
    if not os.path.exists(ppath):
        print("no picks.json in %s -- write the picks BEFORE scoring." % out)
        return 1
    with open(mpath) as fh:
        manifest = {m["event"]: m for m in json.load(fh)}
    with open(ppath) as fh:
        picks = {p["event"]: p for p in json.load(fh)}

    labs = {L["event"]: L for L in vtx_io.load_labels()}
    rows = []
    for ev, m in sorted(manifest.items()):
        L = labs[ev]
        p = picks.get(ev)
        pos = None
        if p and p.get("vertex_id") is not None:
            for c in m["candidates"]:
                if c["vertex_id"] == p["vertex_id"]:
                    pos = (c["x"], c["y"], c["z"])
        d = vtx_io.dist(L["truth"], pos)
        with open(m["dump"]) as fh:
            dump = json.load(fh)
        reco = vtx_io.dist(L["truth"], vtx_io.xyz(dump.get("main_vertex")))
        rows.append(dict(event=ev, conf=(p or {}).get("confidence", "-"),
                         vid=(p or {}).get("vertex_id"), dist=d,
                         ok=vtx_io.correct(d), ok3=vtx_io.correct(d, 3.0),
                         reco_ok=vtx_io.correct(reco), reco=reco,
                         why=(p or {}).get("why", "")))

    ansd = [r for r in rows if r["vid"] is not None]
    ok = sum(1 for r in ansd if r["ok"])
    print("%-12s %-8s %8s %7s %9s" % ("event", "conf", "dist_cm", "ok@1cm",
                                      "reco_ok"))
    for r in rows:
        print("%-12s %-8s %8s %7s %9s   %s"
              % (r["event"], r["conf"],
                 ("%.2f" % r["dist"]) if r["dist"] is not None else "-",
                 "yes" if r["ok"] else "no", "yes" if r["reco_ok"] else "no",
                 r["why"][:60]))
    n = len(rows)
    print("\nscanned %d, answered %d (%.0f%%), correct %d (precision %.1f%%)"
          % (n, len(ansd), 100.0 * len(ansd) / n, ok,
             100.0 * ok / max(len(ansd), 1)))
    print("reconstruction on the same %d events: %d correct (%.1f%%)"
          % (n, sum(1 for r in rows if r["reco_ok"]),
             100.0 * sum(1 for r in rows if r["reco_ok"]) / n))
    for tier in ("certain", "likely", "unclear"):
        t = [r for r in rows if r["conf"] == tier]
        ta = [r for r in t if r["vid"] is not None]
        if t:
            print("   %-8s n=%2d answered=%2d correct=%2d"
                  % (tier, len(t), len(ta), sum(1 for r in ta if r["ok"])))
    with open(os.path.join(out, "scored.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--half", default="test", choices=["dev", "test"])
    p1.add_argument("--n", type=int, default=20)
    p1.add_argument("--seed", type=int, default=20260815)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("score")
    p2.add_argument("--dir", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.half, a.n, a.out, a.seed)
    return score(a.dir)


if __name__ == "__main__":
    sys.exit(main())

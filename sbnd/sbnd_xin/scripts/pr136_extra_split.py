#!/usr/bin/env python3
"""doc pr/136 round 3 -- is `q_extra` a scanner verdict or an artefact of the metric?

THE PROBLEM.  em117_score's target is `(members | marked-in) - marked-out`,
where `members` is the SCAN-TIME membership.  So `q_extra` = "charge the shower
holds that the target excludes" mixes two very different things:

  VIOLATION   the scanner looked at this segment and said OUT.  A real cost.
  UNJUDGED    the segment was not in the shower when the scan was made and
              carries no mark either way, so the target excludes it BY
              CONSTRUCTION.  Any knob that grows a shower manufactures this.

This is the same defect class doc pr/136 sec 2 raised against the gamma ledger,
turned on the metric this campaign actually uses, and it matters because doc
sec 11.2's kill criterion 1 is stated in `q_extra`.  The pr/125 K5 flip already
accepted exactly this trade once ("the accepted cost is unlabeled crumb charge
booked as impurity", wct-pr-perevt.jsonnet:1951), so the split is precedented.

It does NOT rescue a knob on its own: unjudged charge can still be genuine
over-clustering the scanner never got to rule on.  What it does is say whether
the measured cost is something the owner has already refused, or something he
has never been asked about -- and those need different answers.

READ-ONLY.

    scripts/pr136_extra_split.py --arm off1 --arm onV1c90 [--arm onV1c90d25]
"""
import argparse, importlib.util, os, sys

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
ED = os.path.join(SX, "em_display")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S = _load("em117_score", os.path.join(ED, "em117_score.py"))


def one_arm(tag):
    tot = dict(target=0.0, extra=0.0, viol=0.0, unjudged=0.0, miss=0.0)
    worst = []
    for man, ltag in ((("em117-136%s98-manifest.tsv" % tag), "emscan-0827"),
                      (("em114c-136%s141-manifest.tsv" % tag), "emscan-0828-agent5")):
        mp = os.path.join(ED, man)
        if not os.path.exists(mp):
            print("[warn] missing %s" % mp); continue
        labels = S.load_labels(ltag)
        prep_dir = os.path.join(ED, "emprep-136%s" % tag)
        for ev, mrow in sorted(S.load_manifest(mp).items()):
            rec = labels.get(ev)
            md = ((rec or {}).get("em") or {}).get("marks_detail") or {}
            if not md:
                continue
            dump = S.load_dump(mrow["dump"])
            if not dump:
                continue
            prep = S.load_prep(ev, prep_dir)
            actual, seginfo, _ = S.digest_dump(dump, prep)
            _, rows = S.score_event(rec, dump, prep, cross_run=True)
            for r in rows or ():
                det = md.get(str(r["shower"])) or md.get(r["shower"]) or {}
                marked = det.get("marked") or {}
                outs = {int(s) for s, m in marked.items() if m.get("kind") == "out"}
                have = actual.get(r["matched"], set())
                members = {int(x) for x in (det.get("members") or ())}
                ins = {int(s) for s, m in marked.items() if m.get("kind") == "in"}
                target = (members | ins) - outs
                extra = have - target
                q = lambda ss: sum(seginfo.get(i, {}).get("charge", 0.0) for i in ss)
                v, u = q(extra & outs), q(extra - outs)
                tot["target"] += r["q_target"]; tot["extra"] += r["q_extra"]
                tot["miss"] += r["q_miss"]; tot["viol"] += v; tot["unjudged"] += u
                if r["q_extra"] > 0:
                    worst.append((ev, r["matched"], r["q_extra"], v, u))
    return tot, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    a = ap.parse_args()
    print("q_extra SPLIT: scanner VIOLATION vs UNJUDGED growth  (doc pr/136 r3)")
    print("\n%-12s %10s %10s %10s %10s %10s" %
          ("arm", "q_target", "q_miss", "q_extra", "VIOLATION", "UNJUDGED"))
    keep = {}
    for tag in a.arm:
        t, w = one_arm(tag)
        keep[tag] = (t, w)
        print("%-12s %10.3e %10.3e %10.3e %10.3e %10.3e" %
              (tag, t["target"], t["miss"], t["extra"], t["viol"], t["unjudged"]))
        print("%-12s %10s %9.1f%% %9.1f%% %9.1f%% %9.1f%%" %
              ("", "", 100 * t["miss"] / t["target"], 100 * t["extra"] / t["target"],
               100 * t["viol"] / t["target"], 100 * t["unjudged"] / t["target"]))
    if len(a.arm) >= 2:
        base, last = keep[a.arm[0]][0], keep[a.arm[-1]][0]
        print("\nDELTA %s -> %s (as a fraction of q_target)" % (a.arm[0], a.arm[-1]))
        f = lambda k, d: 100 * (d[k] / d["target"])
        print("  q_miss     %+.2f pt" % (f("miss", last) - f("miss", base)))
        print("  q_extra    %+.2f pt   of which" % (f("extra", last) - f("extra", base)))
        print("    VIOLATION %+.2f pt  (the scanner said OUT and the arm holds it anyway)"
              % (f("viol", last) - f("viol", base)))
        print("    UNJUDGED  %+.2f pt  (segments the scan never ruled on)"
              % (f("unjudged", last) - f("unjudged", base)))
        print("\n  kill criterion 1 restated on VIOLATION only: %s"
              % ("PASS -- recovered more than it violated"
                 if (f("miss", base) - f("miss", last)) >
                    (f("viol", last) - f("viol", base)) else
                 "FAIL -- violated more than it recovered"))
        print("  worst q_extra showers on %s:" % a.arm[-1])
        for ev, sh, qe, v, u in sorted(keep[a.arm[-1]][1], key=lambda x: -x[2])[:8]:
            print("    evt %-8d shw %-8d q_extra=%.2e  violation=%.2e  unjudged=%.2e"
                  % (ev, sh, qe, v, u))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

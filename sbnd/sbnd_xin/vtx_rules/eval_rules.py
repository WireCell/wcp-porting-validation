"""Score the hand-scan rule engine against the owner's labels (doc pr/80).

  cd sbnd_xin
  python3 vtx_rules/eval_rules.py --half dev --limit 20     # stage 1
  python3 vtx_rules/eval_rules.py --half dev --limit 60     # stage 2
  python3 vtx_rules/eval_rules.py --half dev                # stage 3 (150)
  python3 vtx_rules/eval_rules.py --half test --out runs/test-final   # ONCE

Scoring conventions, all fixed before the first measurement:

  correct        = Euclidean distance <= 1.0 cm to the label's rank-1 pick
                   (doc pr/78/79's definition; 3 cm reported alongside).
  answered       = the engine committed to a vertex.
  ANSWER-RATE    = answered / scored.
  PRECISION      = correct / answered.  This is the headline: the tool's job is
                   to pre-label events an AI is confident about and hand the
                   rest to a human, so being right when it speaks matters more
                   than speaking often.

Two exclusions, both decided before any number was seen and both reported so
they can be audited rather than taken on trust:

  * `not_a_candidate` labels (3 events).  The owner placed the vertex by hand
    because no PR-graph vertex was near it, so no selection rule of any kind can
    reach them -- doc pr/52's Tier D.
  * events the engine routes to R8 "just dots".  The owner states the answer
    does not matter there, yet they still had to click something, so scoring
    them normally charges the engine a miss for abstaining AND a miss for
    answering differently.  They are counted in the abstain rate and excluded
    from precision.  The headline is also printed WITHOUT this exclusion, as a
    floor, so the exclusion cannot flatter the result unnoticed.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines                                                 # noqa: E402
import make_split                                                # noqa: E402
import vtx_io                                                    # noqa: E402
import vtx_rules                                                 # noqa: E402

COLS = ["tag", "event", "runNo", "subRunNo", "eventNo", "half",
        "decision", "confidence", "branch", "rule",
        "truth_x", "truth_y", "truth_z", "ans_x", "ans_y", "ans_z",
        "dist_cm", "ok1", "ok3", "prod_dist_cm", "prod_ok1",
        "margin", "not_a_candidate", "notes"]


def run_one(label, crosscheck=True):
    path = baselines.deployed_dump_path(label)
    arm = "deployed"
    if not path:
        # The 8 vtxscan-prod0813-mc labels have no -ma10 arm.  Fall back to the
        # dump the scan was taken on and MARK it, rather than dropping them
        # silently or pretending they are on the deployed operating point.
        path = label["source"]
        arm = "scanned"
    with open(path) as fh:
        dump = json.load(fh)
    res = vtx_rules.decide(dump, crosscheck_reco=crosscheck)
    prod = vtx_io.dist(label["truth"], vtx_io.xyz(dump.get("main_vertex")))
    ans = None
    if res["decision"] == "answer":
        ans = (res["x"], res["y"], res["z"])
    return res, ans, prod, arm


def evaluate(half, limit=None, out=None, quiet=False, crosscheck=True):
    keep = make_split.load_split().get(half) if half else None
    labels = [L for L in vtx_io.load_labels()
              if keep is None or L["key"] in keep]
    labels.sort(key=lambda L: (L["tag"], L.get("runNo") or -1,
                               L.get("subRunNo") or -1, L.get("eventNo") or -1))
    if limit:
        labels = labels[:limit]

    rows = []
    c = collections.Counter()
    per_rule = collections.defaultdict(lambda: collections.Counter())
    abstain_reason = collections.Counter()
    arms = collections.Counter()

    for L in labels:
        res, ans, prod, arm = run_one(L, crosscheck)
        arms[arm] += 1
        d = vtx_io.dist(L["truth"], ans) if ans else None
        ok1 = vtx_io.correct(d)
        ok3 = vtx_io.correct(d, vtx_io.TOL_LOOSE)
        nac = bool(L.get("not_a_candidate"))
        rows.append(dict(
            tag=L["tag"], event=L["event"], runNo=L.get("runNo"),
            subRunNo=L.get("subRunNo"), eventNo=L.get("eventNo"), half=half or "",
            decision=res["decision"], confidence=res["confidence"],
            branch=res["branch"], rule=res["rule"],
            truth_x="%.3f" % L["truth"][0], truth_y="%.3f" % L["truth"][1],
            truth_z="%.3f" % L["truth"][2],
            ans_x=("%.3f" % ans[0]) if ans else "",
            ans_y=("%.3f" % ans[1]) if ans else "",
            ans_z=("%.3f" % ans[2]) if ans else "",
            dist_cm=("%.3f" % d) if d is not None else "",
            ok1=int(ok1), ok3=int(ok3),
            prod_dist_cm=("%.3f" % prod) if prod is not None else "",
            prod_ok1=int(vtx_io.correct(prod)),
            margin=("%.2f" % res["margin"]) if res.get("margin") is not None else "",
            not_a_candidate=int(nac),
            notes=" | ".join(res["notes"])))

        c["all"] += 1
        c["tierD"] += nac
        dots = res["branch"] == "dots"
        c["dots"] += dots
        if nac or dots:
            continue                       # excluded from precision, see docstring
        c["scored"] += 1
        c["prod_ok"] += vtx_io.correct(prod)
        if res["decision"] == "answer":
            c["answered"] += 1
            c["ok1"] += ok1
            c["ok3"] += ok3
            per_rule[res["rule"]]["n"] += 1
            per_rule[res["rule"]]["ok"] += ok1
            per_rule["conf:" + res["confidence"]]["n"] += 1
            per_rule["conf:" + res["confidence"]]["ok"] += ok1
        else:
            c["abstain"] += 1
            # Abstain reasons are counted in their own table.  Putting them in
            # the per-rule precision table would print "0.0%" beside a rule that
            # never claimed anything, which reads as a failure and is not one.
            abstain_reason[res["rule"]] += 1

    if out:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "events.tsv")
        with open(path, "w") as fh:
            fh.write("\t".join(COLS) + "\n")
            for r in rows:
                fh.write("\t".join("" if r[k] is None else str(r[k])
                                   for k in COLS) + "\n")
        if not quiet:
            print("wrote %s (%d rows)" % (path, len(rows)))

    if not quiet:
        report(c, per_rule, abstain_reason, rows, arms, half)
    return c, per_rule, rows


def report(c, per_rule, abstain_reason, rows, arms, half):
    n = c["scored"]
    ans = c["answered"]
    print("\n=== half=%s  labels=%d  arms=%s" % (half or "all", c["all"],
                                                 dict(arms)))
    print("excluded: %d Tier-D (not_a_candidate), %d R8 'just dots'"
          % (c["tierD"], c["dots"]))
    print("scored: %d" % n)
    print("  production baseline on the same events : %d  (%.1f%%)"
          % (c["prod_ok"], 100.0 * c["prod_ok"] / max(n, 1)))
    print("  engine answered                        : %d  (answer-rate %.1f%%)"
          % (ans, 100.0 * ans / max(n, 1)))
    print("  engine PRECISION @1cm                  : %d  (%.1f%%)"
          % (c["ok1"], 100.0 * c["ok1"] / max(ans, 1)))
    print("  engine precision @3cm                  : %d  (%.1f%%)"
          % (c["ok3"], 100.0 * c["ok3"] / max(ans, 1)))
    print("  engine overall (abstain counts wrong)  : %d  (%.1f%%)"
          % (c["ok1"], 100.0 * c["ok1"] / max(n, 1)))

    # The floor: nothing excluded at all.
    tot = len(rows)
    ok_all = sum(r["ok1"] for r in rows)
    ansd_all = sum(1 for r in rows if r["decision"] == "answer")
    print("  FLOOR, no exclusions: %d/%d answered, precision %.1f%%, "
          "overall %.1f%%"
          % (ansd_all, tot, 100.0 * ok_all / max(ansd_all, 1),
             100.0 * ok_all / max(tot, 1)))

    print("\nrules that ANSWERED (precision among answered):")
    for k in sorted(per_rule):
        v = per_rule[k]
        if k.startswith("conf:") or v["n"] == 0:
            continue
        print("   %-34s n=%4d  ok=%4d  %5.1f%%"
              % (k, v["n"], v["ok"], 100.0 * v["ok"] / max(v["n"], 1)))
    print("\n   by confidence tier:")
    for k in sorted(per_rule):
        if not k.startswith("conf:"):
            continue
        v = per_rule[k]
        print("   %-34s n=%4d  ok=%4d  %5.1f%%"
              % (k, v["n"], v["ok"], 100.0 * v["ok"] / max(v["n"], 1)))
    if abstain_reason:
        print("\nwhy the engine ABSTAINED (%d events, no claim made):"
              % sum(abstain_reason.values()))
        for k in sorted(abstain_reason, key=lambda k: -abstain_reason[k]):
            print("   %-34s n=%4d" % (k, abstain_reason[k]))

    # Precision vs answer-rate, sliced by the engine's own confidence: this is
    # the curve that says whether the abstain branch is buying anything.
    order = ["certain", "likely"]
    cum_n = cum_ok = 0
    print("\nprecision vs coverage, by confidence tier:")
    for tier in order:
        v = per_rule.get("conf:" + tier, {})
        cum_n += v.get("n", 0)
        cum_ok += v.get("ok", 0)
        if cum_n:
            print("   <= %-8s answered %4d (%5.1f%% of scored)  precision %5.1f%%"
                  % (tier, cum_n, 100.0 * cum_n / max(n, 1),
                     100.0 * cum_ok / cum_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["dev", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--independent", action="store_true",
                    help="skip the R9 reco cross-check: the rules alone")
    args = ap.parse_args()
    if args.half == "test" and not args.out:
        print("refusing to open the locked test half without --out: the run "
              "must leave a record.")
        return 1
    evaluate(args.half, args.limit, args.out,
             crosscheck=not args.independent)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Group two em117_score.py TSVs by the SCAN VERDICT and report the delta.

em117_score.py's own bucket column reads `pr115-handscan-buckets.tsv`, which
covers the 97-event pr/115 scan and not the 141-event pr/116 one, so every row
in a 141 table prints bucket `?`.  Rather than fork the scorer a second time
just to move a filename, this joins the verdict in from the labels.

Reports, per verdict bucket:  n showers, median charge-weighted F1, and
Sum q_miss / Sum q_extra AS SEPARATE COLUMNS -- a merge-widening change can
improve completeness and degrade purity at once, and a flat median F1 hides
exactly that.

    ./score_by_verdict.py BASE.tsv NEW.tsv [--tag emscan-0828-agent5]
"""
import argparse, csv, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ORDER = ["under-clustered", "over-clustered", "both", "correct",
         "vertex-bad (undecidable)", "not an EM shower", "(none)"]


def verdicts(tag):
    out = {}
    for p in glob.glob(os.path.join(SX, "em_labels", tag, "labels-*.json")):
        d = json.load(open(p))
        out[str(d["eventNo"])] = (d.get("em") or {}).get("verdict") or "(none)"
    return out


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def agg(rs):
    f1 = [float(r["q_f1"]) for r in rs]
    return (len(rs), statistics.median(f1) if f1 else float("nan"),
            sum(float(r["q_miss"]) for r in rs),
            sum(float(r["q_extra"]) for r in rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("new")
    ap.add_argument("--tag", default="emscan-0828-agent5")
    a = ap.parse_args()
    v = verdicts(a.tag)
    B, N = rows(a.base), rows(a.new)
    keyed = lambda rs: {(r["event"], r["shower"]): r for r in rs}
    kb, kn = keyed(B), keyed(N)
    common = sorted(set(kb) & set(kn))
    print("shower rows: base %d  new %d  common %d" % (len(B), len(N), len(common)))
    if len(common) != len(B) or len(common) != len(N):
        print("  NOTE: %d base-only, %d new-only row(s) -- reported but not delta'd"
              % (len(set(kb) - set(kn)), len(set(kn) - set(kb))))

    print("\n%-26s %4s  %-21s  %-21s  %-21s" % ("", "n", "median q-F1", "Sum q_miss (missing)", "Sum q_extra (wrongly held)"))
    print("%-26s %4s  %9s %9s  %10s %10s  %10s %10s"
          % ("verdict bucket", "", "base", "new", "base", "new", "base", "new"))
    tot = []
    for vb in ORDER:
        rb = [kb[k] for k in common if v.get(k[0]) == vb]
        rn = [kn[k] for k in common if v.get(k[0]) == vb]
        if not rb:
            continue
        nb, mb, qmb, qeb = agg(rb)
        _, mn, qmn, qen = agg(rn)
        print("%-26s %4d  %9.3f %9.3f  %10.3g %10.3g  %10.3g %10.3g"
              % (vb, nb, mb, mn, qmb, qmn, qeb, qen))
        tot.append((vb, nb, mb, mn, qmb, qmn, qeb, qen))
    rb = [kb[k] for k in common]; rn = [kn[k] for k in common]
    nb, mb, qmb, qeb = agg(rb); _, mn, qmn, qen = agg(rn)
    print("%-26s %4d  %9.3f %9.3f  %10.3g %10.3g  %10.3g %10.3g"
          % ("ALL", nb, mb, mn, qmb, qmn, qeb, qen))

    # movers: per shower, signed change in charge-weighted F1
    mv = sorted(((float(kn[k]["q_f1"]) - float(kb[k]["q_f1"]), k) for k in common),
                key=lambda t: -abs(t[0]))
    print("\nlargest movers (charge-weighted F1, new - base); matched = the reco")
    print("shower the label was joined to, which can RE-ROOT when membership changes:")
    print("%-10s %-9s %8s %8s %8s  %-9s %-9s %s" %
          ("event", "shower", "base", "new", "delta", "match b", "match n", "verdict"))
    for d, k in mv[:15]:
        if abs(d) < 1e-9:
            break
        print("%-10s %-9s %8.3f %8.3f %+8.3f  %-9s %-9s %s"
              % (k[0], k[1], float(kb[k]["q_f1"]), float(kn[k]["q_f1"]), d,
                 kb[k]["matched"], kn[k]["matched"], v.get(k[0])))
    same = sum(1 for d, _ in mv if abs(d) < 1e-9)
    print("\nunchanged shower rows: %d of %d" % (same, len(common)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pr/141 session 2, item 2 -- AUDIT of doc pr/139 sec 22.5's EM/non-EM table.

READ-ONLY.  Reads the SHOWER_SPLIT tape from an arm's stdout.log plus the two
split-label tags, and writes only its --tsv.

WHY.  sec 22.5 proposed restricting the splitter to EM-typed objects on

    | candidate class     | n  | confirmed cuts | fires | purity |
    | EM-typed (|pdg|=11) | 63 |             31 |    35 |  0.857 |
    | not EM-typed        |  8 |              1 |     1 |  0.000 |

doc pr/141 sec 14 measured the pi-typed half by hand and got 4 confirmed cuts
and 0 false fires, i.e. every number in the second row is wrong.  This script
asks the prior question -- **is the table reproducible at all?** -- because
nothing in scripts/ produces it and no committed TSV carries a pdg column.

Three checks, in order:

  1. JOIN.  Do all 71 labelled objects appear in the arm's tape?  A silent join
     miss is the campaign's known failure mode (node ids drift between arms).
  2. SOURCE.  The tape's `pdg=` and the dump's `particle_id` are two different
     reads of the type.  If they disagree the class assignment depends on which
     one a script happened to use -- that alone could produce sec 22.5.
  3. TABLE.  Rebuild the two rows from the tape, on every arm that has one, and
     say which (if any) reproduces 63/31/35 and 8/1/1.

    python3 scripts/pr141_em_audit.py --tsv docs/pr/pr141-em-audit.tsv
"""
import argparse, collections, glob, json, os, re

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["splitscan-0902-pi0", "splitscan-0903-wide"]
# The pi-typed sweep of doc pr/141 sec 14.  Kept SEPARATE from TAGS because the
# 71-object set is EM-enriched by construction and this one is not; the union is
# what sec 17.3 prices.
PI_TAG = "pisplit-0905-owner"
ARMS = ["work-pr140r2-off", "work-pr140r1-on", "work-pr140r2-on"]
CAND = re.compile(r"SHOWER_SPLIT cand shower=(\d+) pdg=(-?\d+) nseg=(\d+) npts=(\d+) "
                  r"Q=([\d.eE+-]+) n_seed=(\d+) valley_best=([\d.]+) angle_best=(-?[\d.]+) "
                  r"nacc=(\d+) nparts=(\d+) fired=(\d)")


def labels(tags=None):
    """(event, shower) -> dict(verdict, tag, nparts)."""
    out = {}
    for tag in (tags if tags is not None else TAGS):
        for f in sorted(glob.glob(os.path.join(SX, "em_labels", tag, "labels-evt*.json"))):
            d = json.load(open(f))
            ev = int(str(d.get("event", "")).replace("evt", "") or d.get("eventNo") or 0)
            for n, x in (d.get("split_labels") or {}).items():
                out[(ev, int(n))] = dict(verdict=x.get("verdict"), tag=tag,
                                         nparts=int(x.get("n_parts") or 0),
                                         conf=x.get("confidence"))
    return out


def tape(arm):
    """(event, shower) -> dict from the SHOWER_SPLIT cand line."""
    out = {}
    for lg in sorted(glob.glob(os.path.join(SX, arm) + "-*/pr_evt*/stdout.log")):
        ev = int(re.search(r"pr_evt(\d+)/", lg).group(1))
        for line in open(lg, errors="replace"):
            m = CAND.search(line)
            if not m:
                continue
            g = m.groups()
            out[(ev, int(g[0]))] = dict(pdg=int(g[1]), nseg=int(g[2]), nacc=int(g[8]),
                                        nparts=int(g[9]), fired=int(g[10]))
    return out


def dump_pdg(arm):
    out = {}
    for f in sorted(glob.glob(os.path.join(SX, arm) + "-*/pr_evt*/calib-pr-evt*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = int((d.get("meta") or {}).get("eventNo") or 0)
        for s in (d.get("showers") or ()):
            out[(ev, int(s["id"]))] = int(s.get("particle_id") or 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    lab = labels()
    print("labelled objects across %s: %d" % (" + ".join(TAGS), len(lab)))
    rows = []
    for arm in ARMS:
        tp = tape(arm)
        if not tp:
            print("\n%-20s NO TAPE" % arm)
            continue
        hit = {k: v for k, v in lab.items() if k in tp}
        miss = sorted(k for k in lab if k not in tp)
        dp = dump_pdg(arm)
        # source disagreement, over every taped object (not only the labelled)
        both = [(k, tp[k]["pdg"], dp[k]) for k in tp if k in dp]
        dis = [(k, a, b) for k, a, b in both if a != b]
        print("\n=== %s ===" % arm)
        print("  tape candidates            : %d" % len(tp))
        print("  labelled objects joined    : %d of %d   (join misses: %d)"
              % (len(hit), len(lab), len(miss)))
        if miss:
            print("    misses: %s" % ", ".join("%d/%d" % k for k in miss[:12]))
        print("  tape pdg vs dump particle_id: %d compared, %d DISAGREE" % (len(both), len(dis)))
        for k, a, b in dis[:8]:
            print("    %d/%-7d tape pdg=%-5d dump particle_id=%d" % (k[0], k[1], a, b))

        for src, get in (("tape", lambda k: tp[k]["pdg"]),
                         ("dump", lambda k: dp.get(k, 0))):
            tab = collections.defaultdict(lambda: dict(n=0, cuts=0, fires=0, tp=0))
            for k, L in sorted(hit.items()):
                cls = "EM" if abs(get(k)) == 11 else "nonEM"
                t = tab[cls]
                t["n"] += 1
                cut = (L["verdict"] or "").startswith("SPLIT")
                fire = tp[k]["fired"] == 1
                t["cuts"] += cut
                t["fires"] += fire
                t["tp"] += (cut and fire)
                rows.append(dict(arm=arm, src=src, event=k[0], shower=k[1],
                                 pdg=get(k), cls=cls, verdict=L["verdict"],
                                 conf=L["conf"], fired=int(fire),
                                 nacc=tp[k]["nacc"], nparts=tp[k]["nparts"], tag=L["tag"]))
            print("  --- class table, pdg from the %s ---" % src)
            print("      %-8s %4s %14s %6s %8s %8s" %
                  ("class", "n", "confirmed cuts", "fires", "purity", "eff"))
            for cls in ("EM", "nonEM"):
                t = tab[cls]
                pur = (t["tp"] / t["fires"]) if t["fires"] else float("nan")
                eff = (t["tp"] / t["cuts"]) if t["cuts"] else float("nan")
                print("      %-8s %4d %14d %6d %8.3f %8.3f"
                      % (cls, t["n"], t["cuts"], t["fires"], pur, eff))
        # --- sec 17.3: the union of every labelled NON-EM object, tape classes ---
        if arm == ARMS[0]:
            pi = labels([PI_TAG])
            uni = dict(hit)
            uni.update({k: v for k, v in pi.items() if k in tp})
            nem = {k: v for k, v in uni.items() if abs(tp[k]["pdg"]) != 11}
            cuts = sum(1 for v in nem.values() if (v["verdict"] or "").startswith("SPLIT"))
            fires = sum(1 for k in nem if tp[k]["fired"] == 1)
            tps = sum(1 for k, v in nem.items()
                      if tp[k]["fired"] == 1 and (v["verdict"] or "").startswith("SPLIT"))
            print("\n  --- sec 17.3: ALL labelled non-EM objects (tape pdg), %s + %s ---"
                  % (" + ".join(TAGS), PI_TAG))
            print("      labelled in the pi sweep      : %d (joined to the tape: %d)"
                  % (len(pi), sum(1 for k in pi if k in tp)))
            print("      overlap with the 71-object set: %d" % len(set(pi) & set(hit)))
            print("      non-EM, union                 : n=%d  confirmed cuts=%d  fires=%d  true=%d"
                  % (len(nem), cuts, fires, tps))
            print("      -> purity %.3f, efficiency %.3f"
                  % (tps / fires if fires else float("nan"),
                     tps / cuts if cuts else float("nan")))
            for k, v in sorted(nem.items()):
                print("        %-8d %-7d pdg=%-5d %-7s fired=%d"
                      % (k[0], k[1], tp[k]["pdg"], v["verdict"], tp[k]["fired"]))

    if args.tsv:
        cols = ["arm", "src", "event", "shower", "pdg", "cls", "verdict", "conf",
                "fired", "nacc", "nparts", "tag"]
        with open(os.path.join(SX, args.tsv), "w") as fh:
            fh.write("# doc pr/141 sec 17 -- audit of doc pr/139 sec 22.5\n")
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("\nwrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()

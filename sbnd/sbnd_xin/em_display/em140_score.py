#!/usr/bin/env python3
"""Score EM shower clustering against the hand scan -- PER-PART edition.

A FORK of em117_score.py, not a generalisation of it (CLAUDE.md sec 2 "fork by
duplication"; em117 is itself a fork of em115).  em117_score.py stays byte-
untouched and pinned to the single-target definition every doc from pr/117 to
pr/139 quotes.  THIS script adds exactly one thing, doc pr/139 item 2:

    --split-tag TAG   merge the owner's PER-PART boundaries (the split-scan
                      label schema: split_labels[node].parts = {part: [seg]})
                      into the completeness target.

WHY.  em117's target is one set per scanned shower, `(members | ins) - outs`.
When the reco correctly splits that shower into two, only ONE of the daughters
can match the single target; the other daughter's charge scores as q_miss.  A
CORRECT SPLIT IS THEREFORE PENALISED AS UNDER-CLUSTERING, which is why doc
pr/139 says q_miss/q_extra "cannot grade a splitter or a re-home" and why P1.4
(re-home) has been parked with no metric that can see it.

WHAT CHANGES.  For a scanned shower that also carries a split label with
n_parts >= 2, the single target becomes N targets, one per hand-labelled part.
Two details carry the whole result:

  1. MATCHING IS INJECTIVE.  Each reconstructed shower may be claimed by at
     most one part (greedy, descending charge overlap).  Without this, one
     un-split reco object wins part 0 AND part 1 and a failure to split scores
     high on both -- the metric would be inverted, which is the opposite of the
     bug it exists to fix.  An unmatched part scores q_comp = 0.
  2. THE DENOMINATOR IS PRESERVED.  Target segments the split label does not
     mention (measured: 50 of 318 over the 12 overlapping SPLIT showers, and
     they are mostly `in` marks -- the split display shows reco membership, so
     it never showed a segment the reco does not hold) become a RESIDUAL part
     "*" that competes for a match on the same injective rule.  So the parts
     plus the residual partition exactly the target em117 would have used, and
     the two metrics are comparable on total charge.

Scoring stays on SEGMENT SETS weighted by segment charge, as em117 does.  That
matters here: `kine_charge` credits a 2D charge cell to every shower within
0.6 cm with no cross-shower dedup, so a charge-level split metric would double
count the overlap.  Segments are exclusive, so a segment-set metric cannot.

READ THE BASELINE UNDER BOTH METRICS.  Only 12 of ~90 scored showers change
target.  Report the merged-target number for the baseline arm as well as the
new one, or a metric change and a reco change land in the same number -- the
doc pr/130 census bug that looked like a physics result.

    ./em140_score.py --split-tag splitscan-0902-pi0 \
        --tag emscan-0827 --manifest M --prepdir D --tsv OUT.tsv
    ./em140_score.py ...            # no --split-tag: byte-equal to em117_score

READ-ONLY over `em_labels/` (CLAUDE.md M13) and over the calib dumps.
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(SX, "em_display", "em114-manifest.tsv")
LABEL_ROOT = os.path.join(SX, "em_labels")
DEFAULT_PREP_DIR = os.path.join(SX, "em_display", "emprep")
BUCKETS_TSV = os.path.join(SX, "docs", "pr", "pr115-handscan-buckets.tsv")


# ------------------------------------------------------------------ loading
def load_manifest(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[int(r["event"])] = r
    return rows


def load_buckets():
    """pr/115's own classification, so this script can group by bucket."""
    if not os.path.exists(BUCKETS_TSV):
        return {}
    out = {}
    with open(BUCKETS_TSV) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[int(r["event"])] = r
    return out


def load_labels(tag):
    out = {}
    d = os.path.join(LABEL_ROOT, tag)
    if not os.path.isdir(d):
        sys.exit("no such label tag: %s" % d)
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("labels-evt") and fn.endswith(".json")):
            continue
        with open(os.path.join(d, fn)) as fh:
            rec = json.load(fh)
        out[int(rec["eventNo"])] = rec
    return out


def load_split_labels(tag):
    """-> {(event, node): {"verdict":…, "n_parts":…, "parts": {part: set(seg)}}}.

    The split-scan tag.  Its node key is the start-segment display id of the
    object on the PRE-SPLIT arm the split display reads, the same key space the
    completeness scan uses for `marks_by_shower`."""
    out = {}
    if not tag:
        return out
    d = os.path.join(LABEL_ROOT, tag)
    if not os.path.isdir(d):
        sys.exit("no such split-label tag: %s" % d)
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("labels-evt") and fn.endswith(".json")):
            continue
        with open(os.path.join(d, fn)) as fh:
            rec = json.load(fh)
        ev = int(str(rec.get("event", "")).replace("evt", "") or rec.get("eventNo"))
        for node, det in (rec.get("split_labels") or {}).items():
            parts = {str(p): set(int(x) for x in segs)
                     for p, segs in (det.get("parts") or {}).items()}
            out[(ev, int(node))] = {"verdict": det.get("verdict"),
                                    "n_parts": int(det.get("n_parts") or 1),
                                    "parts": parts}
    return out


def load_prep(ev, prep_dir):
    """The stage-2 probe sidecar.  This is what em_display SHOWED the scanner
    (`probe_members` in em_display_viewer.py), and it is the only faithful
    membership when two showers overlap: the dump's `segments[].shower_id` is
    single-valued, so a segment held by two showers is credited to one.
    Scoring against the lossy join would invent misses that are not there."""
    p = os.path.join(prep_dir, "emprep-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def load_dump(path):
    """Manifest `dump` paths are relative to sbnd_xin, not to the caller's cwd."""
    if not os.path.isabs(path):
        path = os.path.join(SX, path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------- dump digest
def seg_charge(seg):
    """Sum of fitted-point dQ.  The quantity cal_kine_charge integrates, so it
    is the right weight for 'how much of the shower is this member worth'."""
    tot = 0.0
    for p in seg.get("points") or ():
        q = p.get("dQ")
        if q and q > 0:
            tot += q
    return tot


def digest_dump(dump, prep=None):
    """-> (actual: shower id -> set(seg id), seginfo: seg id -> dict).

    Membership comes from the probe sidecar when there is one, exactly as the
    display does; the dump join is the fallback.  Charge comes from the
    sidecar's per-member dQ when available, else from summing points[].dQ."""
    actual = defaultdict(set)
    seginfo = {}
    for s in dump.get("segments") or ():
        sid = int(s["id"])
        seginfo[sid] = {
            "cluster": int(s["cluster_id"]),
            "length": float(s.get("length") or 0.0),
            "charge": seg_charge(s),
            "pdg": s.get("particle_id"),
            "owner": int(s.get("shower_id", -1)),
        }
        own = int(s.get("shower_id", -1))
        if own >= 0:
            actual[own].add(sid)
    join_actual = {k: set(v) for k, v in actual.items()}
    if prep:
        actual = defaultdict(set)
        for node, e in (prep.get("showers") or {}).items():
            for m in e.get("members") or ():
                sid = int(m["seg"])
                actual[int(node)].add(sid)
                info = seginfo.setdefault(sid, {"cluster": int(m.get("cluster", -1)),
                                                "length": 0.0, "charge": 0.0,
                                                "pdg": m.get("pdg"), "owner": int(node)})
                if m.get("dQ"):
                    info["charge"] = float(m["dQ"])
                if m.get("length"):
                    info["length"] = float(m["length"])
    return actual, seginfo, join_actual


# ------------------------------------------------------------------ scoring
def weigh(ids, seginfo, key):
    return sum(seginfo.get(i, {}).get(key, 0.0) for i in ids)


def match_shower(shw, target, actual, seginfo):
    """Cross-run matching: the reconstructed shower with the largest
    charge-weighted overlap with `target` (ties -> smaller node id; zero
    overlap everywhere -> the exact key if present, else no shower).  The
    label's key is the SCAN-TIME start-segment display id; a knob that
    re-roots or renames the shower breaks the exact join, and an exact-only
    score would read the rename as comp=0."""
    best, best_q, best_n = None, 0.0, 0
    for node in sorted(actual):
        inter = actual[node] & target
        if not inter:
            continue
        qi = weigh(inter, seginfo, "charge")
        if qi > best_q or (qi == best_q and best is not None and node < best):
            best, best_q, best_n = node, qi, len(inter)
    if best is None:
        return shw if shw in actual else None
    return best


def assign_injective(part_targets, actual, seginfo):
    """part -> reco node, each node claimed by AT MOST ONE part.

    Greedy on descending charge overlap; ties broken on (part, node) so the
    result is deterministic and independent of dict order.  A part with no
    unclaimed overlapping shower gets None and scores q_comp = 0 -- which is
    the point: when the reco fails to split, the single big object can win one
    part and the other part must score zero."""
    cand = []
    for p, tp in part_targets.items():
        if not tp:
            continue
        for node in sorted(actual):
            inter = actual[node] & tp
            if not inter:
                continue
            cand.append((weigh(inter, seginfo, "charge"), p, node))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    got, used = {}, set()
    for _q, p, node in cand:
        if p in got or node in used:
            continue
        got[p] = node
        used.add(node)
    return {p: got.get(p) for p in part_targets}


def score_shower_parts(shw, det, split, actual, seginfo):
    """One scanned shower with a hand per-part boundary -> N + 1 rows.

    The target em117 would have used is partitioned into the labelled parts
    plus a residual "*" holding whatever the split label does not mention (see
    the module docstring); every group is matched INJECTIVELY."""
    marked = det.get("marked") or {}
    if not marked:
        return None
    members = set(int(x) for x in (det.get("members") or ()))
    ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
    outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
    target_all = (members | ins) - outs
    if not target_all:
        return None

    part_targets = {}
    covered = set()
    for p, segs in sorted(split["parts"].items()):
        tp = target_all & segs
        covered |= segs
        part_targets[p] = tp
    residual = target_all - covered
    if residual:
        part_targets["*"] = residual

    got = assign_injective(part_targets, actual, seginfo)
    q = lambda st: weigh(st, seginfo, "charge")
    rows = []
    for p in sorted(part_targets, key=lambda x: (x == "*", x)):
        target = part_targets[p]
        node = got.get(p)
        have = actual.get(node, set()) if node is not None else set()
        inter = have & target
        miss, extra = target - have, have - target
        qt, qh, qi = q(target), q(have), q(inter)
        row = {
            "shower": shw, "part": p,
            "matched": node if node is not None else -1,
            "n_target": len(target), "n_actual": len(have), "n_inter": len(inter),
            "n_miss": len(miss), "n_extra": len(extra),
            "comp": (len(inter) / len(target)) if target else float("nan"),
            "pur": (len(inter) / len(have)) if have else float("nan"),
            "q_target": qt, "q_actual": qh,
            "q_comp": (qi / qt) if qt > 0 else float("nan"),
            "q_pur": (qi / qh) if qh > 0 else float("nan"),
            "q_miss": q(miss), "q_extra": q(extra),
            "miss": sorted(miss), "extra": sorted(extra),
            "drift": len(members ^ have),
        }
        c, pu = row["q_comp"], row["q_pur"]
        row["q_f1"] = (2 * c * pu / (c + pu)) if (c == c and pu == pu and c + pu > 0) else 0.0
        rows.append(row)
    return rows


def score_shower(shw, det, actual, seginfo, cross_run=False):
    """One scanned shower -> the comparison row, or None if it carries no mark."""
    marked = det.get("marked") or {}
    if not marked:
        return None
    members = set(int(x) for x in (det.get("members") or ()))
    ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
    outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
    target = (members | ins) - outs
    if cross_run:
        node = match_shower(shw, target, actual, seginfo)
        have = actual.get(node, set()) if node is not None else set()
    else:
        node = shw
        have = actual.get(shw, set())
    if not target and not have:
        return None
    inter = have & target
    miss = target - have          # under-clustering: target says in, reco left out
    extra = have - target         # over-clustering: reco holds what target excludes

    q = lambda s: weigh(s, seginfo, "charge")
    qt, qh, qi = q(target), q(have), q(inter)
    row = {
        "shower": shw, "part": "-",
        "matched": node if node is not None else -1,
        "n_target": len(target), "n_actual": len(have), "n_inter": len(inter),
        "n_miss": len(miss), "n_extra": len(extra),
        "comp": (len(inter) / len(target)) if target else float("nan"),
        "pur": (len(inter) / len(have)) if have else float("nan"),
        "q_target": qt, "q_actual": qh,
        "q_comp": (qi / qt) if qt > 0 else float("nan"),
        "q_pur": (qi / qh) if qh > 0 else float("nan"),
        "q_miss": q(miss), "q_extra": q(extra),
        "miss": sorted(miss), "extra": sorted(extra),
        # Same-run integrity: the sidecar membership should equal what the
        # display showed the scanner.  Cross-run, nonzero is the SIGNAL.
        "drift": len(members ^ have),
    }
    c, p = row["q_comp"], row["q_pur"]
    row["q_f1"] = (2 * c * p / (c + p)) if (c == c and p == p and c + p > 0) else float("nan")
    return row


def score_event(rec, dump, prep=None, cross_run=False, splits=None, ev=None):
    actual, seginfo, join_actual = digest_dump(dump, prep)
    md = (rec.get("em") or {}).get("marks_detail") or {}
    rows, n_split = [], 0
    for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
        sl = (splits or {}).get((ev, int(shw)))
        if sl and sl["n_parts"] >= 2 and sl["parts"]:
            rr = score_shower_parts(int(shw), det, sl, actual, seginfo)
            if rr:
                rows.extend(rr)
                n_split += 1
                continue
        r = score_shower(int(shw), det, actual, seginfo, cross_run)
        if r:
            rows.append(r)
    if not rows:
        return None, []
    # Event scalar: charge-weighted F1, itself weighted by each shower's target
    # charge -- a 300 MeV gamma scored badly must not be averaged away by a
    # 2 MeV stub scored well.
    wsum = sum(r["q_target"] for r in rows)
    if wsum > 0:
        f1 = sum(r["q_f1"] * r["q_target"] for r in rows if r["q_f1"] == r["q_f1"]) / wsum
    else:
        f1 = float("nan")
    ev = {
        "join_loss": sum(len(actual.get(r["matched"], set()) - join_actual.get(r["matched"], set()))
                         for r in rows),
        "n_showers": len(rows),
        "n_split": n_split,
        "q_f1": f1,
        "n_miss": sum(r["n_miss"] for r in rows),
        "n_extra": sum(r["n_extra"] for r in rows),
        "q_miss": sum(r["q_miss"] for r in rows),
        "q_extra": sum(r["q_extra"] for r in rows),
        "q_target": wsum,
        "drift": sum(r["drift"] for r in rows),
    }
    return ev, rows


# --------------------------------------------------------------------- main
def fmt(x, n=3):
    return "-" if x != x else ("%.*f" % (n, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="emscan-0827")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="manifest tsv (default: the em114 scan-time one)")
    ap.add_argument("--prepdir", default=DEFAULT_PREP_DIR,
                    help="probe sidecar dir (default: the em114 scan-time one)")
    ap.add_argument("--cross-run", action="store_true",
                    help="overlap-based shower matching + drift demoted to info "
                         "(implied by a non-default --manifest/--prepdir)")
    ap.add_argument("--split-tag",
                    help="split-scan label tag whose per-part boundaries are merged "
                         "into the target (doc pr/139 item 2).  Omitted => this "
                         "script reproduces em117_score.py.")
    ap.add_argument("--tsv", help="write the per-shower row table here")
    ap.add_argument("--baseline", help="a previous --tsv, for a delta table")
    ap.add_argument("--compare", help="a second --tsv, for a delta table")
    ap.add_argument("--diffstat", nargs=2, metavar=("BASE_PREPDIR", "NEW_PREPDIR"),
                    help="membership diff between two prepdirs over all shared events")
    args = ap.parse_args()

    if args.diffstat:
        return diffstat(args.diffstat[0], args.diffstat[1])
    if args.baseline and args.compare:
        return delta(args.baseline, args.compare)

    cross_run = (args.cross_run
                 or os.path.abspath(args.manifest) != os.path.abspath(DEFAULT_MANIFEST)
                 or os.path.abspath(args.prepdir) != os.path.abspath(DEFAULT_PREP_DIR))

    man, labs, buck = load_manifest(args.manifest), load_labels(args.tag), load_buckets()
    splits = load_split_labels(args.split_tag)
    if args.split_tag:
        n = sum(1 for k, v in splits.items() if v["n_parts"] >= 2 and k[0] in labs)
        print("split labels: %d object(s) from %s, %d of them SPLIT and in this scan\n"
              % (len(splits), args.split_tag, n))
    dumps, preps = {}, {}
    for ev in labs:
        m = man.get(ev)
        dumps[ev] = load_dump(m["dump"]) if m and m.get("dump") else None
        preps[ev] = load_prep(ev, args.prepdir)

    return report_score(labs, dumps, buck, args.tsv, preps, cross_run, splits)


def report_score(labs, dumps, buck, tsv_path, preps, cross_run=False, splits=None):
    rows, evs = [], {}
    for ev in sorted(labs):
        d = dumps.get(ev)
        if not d:
            continue
        e, rr = score_event(labs[ev], d, preps.get(ev), cross_run, splits, ev)
        if not e:
            continue
        evs[ev] = e
        for r in rr:
            r["event"] = ev
            r["bucket"] = (buck.get(ev) or {}).get("bucket", "?")
            rows.append(r)

    print("=" * 78)
    print("PER-EVENT SCORE  (charge-weighted F1 of hand-marked showers)")
    print("=" * 78)
    print("%-9s %-26s %5s %7s %6s %6s %10s %10s" %
          ("event", "bucket", "nshw", "qF1", "nmiss", "nextra", "q_miss", "q_extra"))
    for ev in sorted(evs, key=lambda e: evs[e]["q_f1"]):
        e = evs[ev]
        print("%-9d %-26s %5d %7s %6d %6d %10.3g %10.3g" %
              (ev, (buck.get(ev) or {}).get("bucket", "?")[:26], e["n_showers"],
               fmt(e["q_f1"]), e["n_miss"], e["n_extra"], e["q_miss"], e["q_extra"]))

    nsplit = sum(e.get("n_split", 0) for e in evs.values())
    if splits:
        srows = [r for r in rows if r.get("part", "-") != "-"]
        res = [r for r in srows if r["part"] == "*"]
        # A part whose intersection with the completeness target is EMPTY is not
        # a failed cut: it means the completeness scan had ALREADY excluded that
        # part from this shower (its segments are `out` marks, or were never
        # members).  Counting it as a miss would invent failures.
        empty = [r for r in srows if r["part"] != "*" and r["n_target"] == 0]
        zero = [r for r in srows if r["part"] != "*" and r["n_target"] > 0
                and r["q_comp"] == 0]
        qs = sorted(r["q_f1"] for r in srows if r["part"] != "*" and r["n_target"] > 0)
        print("\nPER-PART TARGET  (doc pr/139 item 2): %d shower(s) expanded into %d row(s)"
              % (nsplit, len(srows)))
        print("  labelled parts scored      : %d   median q_f1 %s   mean %s"
              % (len(qs), fmt(qs[len(qs) // 2]) if qs else "-",
                 fmt(sum(qs) / len(qs)) if qs else "-"))
        print("  parts with NO reco match   : %d  <- the reco failed to make this cut"
              % len(zero))
        for r in sorted(zero, key=lambda r: (r["event"], r["shower"], r["part"])):
            print("      evt%-8d shower %-7d part %-3s  q_target %.4g"
                  % (r["event"], r["shower"], r["part"], r["q_target"]))
        print("  parts already excluded     : %d  (empty target: the completeness scan"
              % len(empty))
        print("      had already marked this part out of this shower -- NOT a failed cut)")
        print("  residual '*' rows          : %d  (target segs the split label did not"
              % len(res))
        print("      mention -- mostly `in` marks; they carry %.4g of %.4g charge)"
              % (sum(r["q_target"] for r in res), sum(r["q_target"] for r in srows)))

    drift = sum(e["drift"] for e in evs.values())
    loss = sum(e["join_loss"] for e in evs.values())
    if cross_run:
        print("\ninfo: cross-run scoring -- membership moved on %d segment slot(s) over"
              % drift)
        print("  %d event(s) vs the scan-time labels; that movement is the SIGNAL here,"
              % len(evs))
        print("  not an integrity failure.  Shower matching is by charge-weighted overlap.")
    else:
        print("\nintegrity: sidecar-vs-label membership drift = %d segment(s) over %d event(s)"
              % (drift, len(evs)))
        if drift:
            print("  WARNING: nonzero drift means sidecar and label came from different runs.")
    print("lossiness: the dump's single-valued segments[].shower_id loses %d member(s)"
          % loss)
    print("  to shower overlap on these events -- the reason this script scores")
    print("  against the sidecar, as the display does.")

    by = defaultdict(list)
    for ev, e in evs.items():
        by[(buck.get(ev) or {}).get("bucket", "?")].append(e)
    print("\n%-30s %5s %8s %10s %10s" % ("bucket", "n", "med qF1", "sum q_miss", "sum q_extra"))
    for b in sorted(by):
        v = sorted(x["q_f1"] for x in by[b] if x["q_f1"] == x["q_f1"])
        med = v[len(v) // 2] if v else float("nan")
        print("%-30s %5d %8s %10.4g %10.4g" %
              (b[:30], len(by[b]), fmt(med), sum(x["q_miss"] for x in by[b]),
               sum(x["q_extra"] for x in by[b])))

    if tsv_path:
        cols = ["event", "bucket", "shower", "part", "matched", "n_target", "n_actual", "n_inter",
                "n_miss", "n_extra", "comp", "pur", "q_comp", "q_pur", "q_f1",
                "q_target", "q_actual", "q_miss", "q_extra", "drift", "miss", "extra"]
        with open(tsv_path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(cols)
            for r in sorted(rows, key=lambda r: (r["event"], r["shower"], str(r.get("part", "-")))):
                w.writerow([",".join(str(x) for x in r[c]) if isinstance(r[c], list)
                            else (fmt(r[c], 4) if isinstance(r[c], float) else r[c])
                            for c in cols])
        print("\nwrote %s (%d shower rows)" % (tsv_path, len(rows)))
    return 0


def delta(a_path, b_path):
    def rd(p):
        out = {}
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out[(int(r["event"]), int(r["shower"]))] = r
        return out
    A, B = rd(a_path), rd(b_path)
    keys = sorted(set(A) | set(B))
    print("%-9s %-9s %9s %9s %9s" % ("event", "shower", "qF1_base", "qF1_new", "delta"))
    tot = 0.0
    for k in keys:
        a, b = A.get(k), B.get(k)
        fa = float(a["q_f1"]) if a and a["q_f1"] not in ("-", "") else float("nan")
        fb = float(b["q_f1"]) if b and b["q_f1"] not in ("-", "") else float("nan")
        d = fb - fa
        if fa != fa and fb != fb:
            continue           # unscored in both runs (no charge on either side)
        if d == d and abs(d) < 1e-9:
            continue
        tot += 0 if d != d else d
        print("%-9d %-9d %9s %9s %+9s" % (k[0], k[1], fmt(fa), fmt(fb), fmt(d)))
    print("\nsum of deltas: %+.4f   (rows shown = changed only)" % tot)
    return 0


# ----------------------------------------------------------- hold-flat check
def prep_membership(prep_dir, ev):
    prep = load_prep(ev, prep_dir)
    if not prep:
        return None, None
    nodes = {}
    segq = {}
    for node, e in (prep.get("showers") or {}).items():
        mem = set()
        for m in e.get("members") or ():
            sid = int(m["seg"])
            mem.add(sid)
            if m.get("dQ"):
                segq[sid] = float(m["dQ"])
        nodes[int(node)] = mem
    return nodes, segq


def diffstat(base_dir, new_dir):
    """Membership diff between two prepdirs, all events present in both.

    Shower identity across runs is NOT comparable by node id alone (a re-rooted
    shower renames); each base shower is matched to the new shower with the
    largest member overlap.  A segment counts as MOVED when its owning matched
    pair disagrees about it; showers with no counterpart count whole.  This is
    the hold-flat check for the mark-free events (good / scanned-no-correction),
    which the score table cannot see."""
    def evs_of(d):
        out = set()
        for fn in os.listdir(d):
            if fn.startswith("emprep-evt") and fn.endswith(".json"):
                out.add(int(fn[len("emprep-evt"):-len(".json")]))
        return out
    if not (os.path.isdir(base_dir) and os.path.isdir(new_dir)):
        sys.exit("diffstat: both arguments must be prepdirs")
    shared = sorted(evs_of(base_dir) & evs_of(new_dir))
    buck = load_buckets()
    print("=" * 78)
    print("MEMBERSHIP DIFF  %s  ->  %s" % (base_dir, new_dir))
    print("=" * 78)
    print("%-9s %-26s %5s %5s %7s %7s %10s" %
          ("event", "bucket", "nshwA", "nshwB", "moved", "unmtch", "q_moved"))
    tot_moved = tot_ev_changed = 0
    changed_events = []
    for ev in shared:
        A, qA = prep_membership(base_dir, ev)
        B, qB = prep_membership(new_dir, ev)
        if A is None or B is None:
            continue
        segq = dict(qA or {})
        segq.update(qB or {})
        # One-to-one greedy matching by descending overlap (ties: smaller
        # ids).  One-to-one matters: showers can OVERLAP (share members), and
        # a many-to-one match would orphan the twin and invent churn -- the
        # identity check (diffstat of a prepdir against itself) must be zero.
        pairs = sorted(((len(A[a] & B[b]), a, b) for a in A for b in B
                        if A[a] & B[b]), key=lambda t: (-t[0], t[1], t[2]))
        match = {}
        used_b = set()
        for n, a, b in pairs:
            if a in match or b in used_b:
                continue
            match[a] = b
            used_b.add(b)
        moved = set()
        unmatched = 0
        for a in sorted(A):
            if a in match:
                moved |= A[a] ^ B[match[a]]
            else:
                moved |= A[a]      # dissolved shower: every member counts
                unmatched += 1
        for b in sorted(set(B) - used_b):
            moved |= B[b]          # newly created shower not matched to any A
            unmatched += 1
        q_moved = sum(segq.get(s, 0.0) for s in moved)
        if moved or len(A) != len(B):
            tot_ev_changed += 1
            changed_events.append(ev)
            tot_moved += len(moved)
            print("%-9d %-26s %5d %5d %7d %7d %10.3g" %
                  (ev, (buck.get(ev) or {}).get("bucket", "?")[:26],
                   len(A), len(B), len(moved), unmatched, q_moved))
    print("\nevents compared: %d   changed: %d   unchanged: %d   segments moved: %d"
          % (len(shared), tot_ev_changed, len(shared) - tot_ev_changed, tot_moved))
    if changed_events:
        print("changed events: %s" % " ".join(str(e) for e in changed_events))
    return 0


if __name__ == "__main__":
    sys.exit(main())

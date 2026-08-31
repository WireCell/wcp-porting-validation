#!/usr/bin/env python3
"""Score EM shower clustering against the hand scan (doc pr/115 sec 15.3).

The hand scan is the only ground truth we have -- the toolkit ships no
truth-matching machinery -- so "did this change help?" has to mean "did it move
toward what the scanner marked".  This script makes that a number.

    ./em115_score.py                       # per-shower / per-event score table
    ./em115_score.py --tsv OUT.tsv         # the row table
    ./em115_score.py --pi0                 # sec 15.6a: T_kine fill vs the identification
    ./em115_score.py --level               # sec 15.5: cluster-level vs shower-level
    ./em115_score.py --baseline A.tsv --compare B.tsv    # delta after a knob change

READ-ONLY over `em_labels/` (CLAUDE.md M13) and over the calib dumps.

HOW THE TWO SIDES MEET.  `PrDisplayDump::dump_showers()` writes
`segments[].shower_id` -- the display id (cluster_id*1000 + seg_id) of the
owning shower's start segment, or -1.  The labels record the same display ids in
`em.marks_detail[shw]{members, marked}`.  So per scanned shower:

    actual  = {seg : dump segments[].shower_id == shw}
    target  = (members at scan time  UNION  IN marks)  MINUS  OUT marks
    completeness = |actual & target| / |target|      (what the reco missed)
    purity       = |actual & target| / |actual|      (what it wrongly holds)

Both are reported unweighted and CHARGE-weighted (sum of `points[].dQ` over the
segment).  Charge weighting is the one that matters: the pi0 mass consumes
charge, so a missing 30 cm member and a missing 1 cm stub are not the same
miss, and counting segments says they are.

WHAT THIS DOES NOT MEASURE.  Agreement with one scanner's marks on 25 events
selected for being wrong.  A gain here is evidence a change does what was
asked, NOT evidence of a physics improvement.  The 37 "good" events are carried
as the regression set for exactly that reason.
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(SX, "em_display", "em114-manifest.tsv")
LABEL_ROOT = os.path.join(SX, "em_labels")
PREP_DIR = os.path.join(SX, "em_display", "emprep")
BUCKETS_TSV = os.path.join(SX, "docs", "pr", "pr115-handscan-buckets.tsv")


# ------------------------------------------------------------------ loading
def load_manifest():
    rows = {}
    with open(MANIFEST) as fh:
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


def load_prep(ev):
    """The stage-2 probe sidecar.  This is what em_display SHOWED the scanner
    (`probe_members` in em_display_viewer.py), and it is the only faithful
    membership when two showers overlap: the dump's `segments[].shower_id` is
    single-valued, so a segment held by two showers is credited to one.
    Scoring against the lossy join would invent misses that are not there."""
    p = os.path.join(PREP_DIR, "emprep-evt%d.json" % ev)
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


def score_shower(shw, det, actual, seginfo):
    """One scanned shower -> the comparison row, or None if it carries no mark."""
    marked = det.get("marked") or {}
    if not marked:
        return None
    members = set(int(x) for x in (det.get("members") or ()))
    ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
    outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
    target = (members | ins) - outs
    have = actual.get(shw, set())
    if not target and not have:
        return None
    inter = have & target
    miss = target - have          # under-clustering: target says in, reco left out
    extra = have - target         # over-clustering: reco holds what target excludes

    q = lambda s: weigh(s, seginfo, "charge")
    qt, qh, qi = q(target), q(have), q(inter)
    row = {
        "shower": shw,
        "n_target": len(target), "n_actual": len(have), "n_inter": len(inter),
        "n_miss": len(miss), "n_extra": len(extra),
        "comp": (len(inter) / len(target)) if target else float("nan"),
        "pur": (len(inter) / len(have)) if have else float("nan"),
        "q_target": qt, "q_actual": qh,
        "q_comp": (qi / qt) if qt > 0 else float("nan"),
        "q_pur": (qi / qh) if qh > 0 else float("nan"),
        "q_miss": q(miss), "q_extra": q(extra),
        "miss": sorted(miss), "extra": sorted(extra),
        # integrity: the dump's membership should equal the membership the
        # display showed the scanner.  A mismatch means the dump and the label
        # are from different runs and every number in the row is suspect.
        "drift": len(members ^ have),
    }
    for k in ("q_comp", "q_pur"):
        pass
    c, p = row["q_comp"], row["q_pur"]
    row["q_f1"] = (2 * c * p / (c + p)) if (c == c and p == p and c + p > 0) else float("nan")
    return row


def score_event(rec, dump, prep=None):
    actual, seginfo, join_actual = digest_dump(dump, prep)
    md = (rec.get("em") or {}).get("marks_detail") or {}
    rows = []
    for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
        r = score_shower(int(shw), det, actual, seginfo)
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
        "join_loss": sum(len(actual.get(r["shower"], set()) - join_actual.get(r["shower"], set()))
                         for r in rows),
        "n_showers": len(rows),
        "q_f1": f1,
        "n_miss": sum(r["n_miss"] for r in rows),
        "n_extra": sum(r["n_extra"] for r in rows),
        "q_miss": sum(r["q_miss"] for r in rows),
        "q_extra": sum(r["q_extra"] for r in rows),
        "q_target": wsum,
        "drift": sum(r["drift"] for r in rows),
    }
    return ev, rows


# ---------------------------------------------------- sec 15.6a: pi0 fill vs id
def pi0_compare(dump):
    """The T_kine fill takes the highest summed kine_charge with NO mass window;
    the identification uses a window and writes pio_id/pio_mass onto the
    showers.  They are different selections -- this reports whether they agreed.
    """
    kine = dump.get("kine") or {}
    fill_mass = kine.get("kine_pio_mass")
    fill_flag = kine.get("kine_pio_flag")
    fe1, fe2 = kine.get("kine_pio_energy_1"), kine.get("kine_pio_energy_2")
    groups = defaultdict(list)
    for s in dump.get("showers") or ():
        pid = int(s.get("pio_id", -1))
        if pid >= 0:
            groups[pid].append(s)
    ident = []
    for pid, shs in sorted(groups.items()):
        ident.append({
            "pio_id": pid,
            "mass": shs[0].get("pio_mass"),
            "showers": sorted(int(s["id"]) for s in shs),
            "kq": sorted((s.get("kine_charge") or 0.0) for s in shs),
        })
    # Did the fill land on an identified pair?  Match on mass (float32 in the
    # ROOT branch vs double in the shower row, so compare with a tolerance).
    matched = None
    if fill_mass:
        for g in ident:
            if g["mass"] and abs(g["mass"] - fill_mass) < 0.01 * max(1.0, abs(fill_mass)):
                matched = g["pio_id"]
                break
    return {
        "fill_mass": fill_mass, "fill_flag": fill_flag,
        "fill_e1": fe1, "fill_e2": fe2,
        "n_ident": len(ident), "ident": ident, "matched": matched,
    }


# ------------------------------------- sec 15.5: cluster level or shower level
def level_test(rec, dump, prep=None):
    """For an over-clustered event: are the two sides the scanner drew already
    in DIFFERENT clusters?

    NOTE THE DIRECTION, it is the opposite of the intuitive reading.  If the two
    sides are in DIFFERENT clusters, image-level separation already did its job
    and the SHOWER pass crossed a correct cluster boundary to merge them -- the
    lever is then the cross-cluster absorb cones, not a split.  Only when the two
    sides SHARE a cluster is this pr/53 / pr/57 sec 14 territory, or a case for a
    shower-level split.

    Two ways the scanner draws the line, and they need different readings:

      in-vs-out    both kinds present under one scanned shower.  The IN set and
                   the OUT set ARE the two objects -- this is the `split-by-proxy`
                   idiom, where a one-segment stub is used as a scratch pad to
                   partition somebody else's shower (evt142421: 33 IN + 10 OUT,
                   all 43 owned by shower 108104).  Reading it as
                   marked-vs-unmarked leaves nothing on the other side.
      marked-rest  one kind only.  The marked segments are one object and the
                   remainder of whichever shower owns them is the other.
    """
    actual, seginfo, _ = digest_dump(dump, prep)
    md = (rec.get("em") or {}).get("marks_detail") or {}
    out = []
    for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
        marked = det.get("marked") or {}
        if not marked:
            continue
        ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
        outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
        owners = Counter(int(m["owner"]) for m in marked.values()
                         if m.get("owner") is not None)
        owner = owners.most_common(1)[0][0] if owners else -1
        if ins and outs:
            rule, side_a, side_b = "in-vs-out", ins, outs
        else:
            side_a = ins | outs
            side_b = actual.get(owner, set()) - side_a
            rule = "marked-rest"
        if not side_a or not side_b:
            continue
        ca = set(seginfo.get(i, {}).get("cluster") for i in side_a) - {None, -1}
        cb = set(seginfo.get(i, {}).get("cluster") for i in side_b) - {None, -1}
        out.append({
            "shower": int(shw), "owner": owner, "rule": rule,
            "n_a": len(side_a), "n_b": len(side_b),
            "clusters_a": sorted(ca), "clusters_b": sorted(cb),
            "shared": sorted(ca & cb),
            # True  = already distinct clusters -> a shower-pass absorb crossed
            #         a correct boundary (cone knobs reach it)
            # False = one cluster holds both -> cluster split or shower split
            "cross_cluster": bool(ca) and bool(cb) and not (ca & cb),
        })
    return out


# --------------------------------------------------------------------- main
def fmt(x, n=3):
    return "-" if x != x else ("%.*f" % (n, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="emscan-0827")
    ap.add_argument("--tsv", help="write the per-shower row table here")
    ap.add_argument("--pi0", action="store_true", help="sec 15.6a measurement")
    ap.add_argument("--level", action="store_true", help="sec 15.5 measurement")
    ap.add_argument("--absorb", action="store_true",
                    help="which pass absorbed each wrongly-held segment")
    ap.add_argument("--baseline", help="a previous --tsv, for a delta table")
    ap.add_argument("--compare", help="a second --tsv, for a delta table")
    args = ap.parse_args()

    if args.baseline and args.compare:
        return delta(args.baseline, args.compare)

    man, labs, buck = load_manifest(), load_labels(args.tag), load_buckets()
    dumps, preps = {}, {}
    for ev in labs:
        m = man.get(ev)
        dumps[ev] = load_dump(m["dump"]) if m and m.get("dump") else None
        preps[ev] = load_prep(ev)

    if args.absorb:
        return report_absorb(labs, buck)
    if args.pi0:
        return report_pi0(labs, dumps, buck, man)
    if args.level:
        return report_level(labs, dumps, buck, preps)
    return report_score(labs, dumps, buck, args.tsv, preps)


def report_score(labs, dumps, buck, tsv_path, preps):
    rows, evs = [], {}
    for ev in sorted(labs):
        d = dumps.get(ev)
        if not d:
            continue
        e, rr = score_event(labs[ev], d, preps.get(ev))
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

    drift = sum(e["drift"] for e in evs.values())
    loss = sum(e["join_loss"] for e in evs.values())
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
        cols = ["event", "bucket", "shower", "n_target", "n_actual", "n_inter",
                "n_miss", "n_extra", "comp", "pur", "q_comp", "q_pur", "q_f1",
                "q_target", "q_actual", "q_miss", "q_extra", "drift", "miss", "extra"]
        with open(tsv_path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(cols)
            for r in sorted(rows, key=lambda r: (r["event"], r["shower"])):
                w.writerow([",".join(str(x) for x in r[c]) if isinstance(r[c], list)
                            else (fmt(r[c], 4) if isinstance(r[c], float) else r[c])
                            for c in cols])
        print("\nwrote %s (%d shower rows)" % (tsv_path, len(rows)))
    return 0


def report_pi0(labs, dumps, buck, man):
    print("=" * 78)
    print("sec 15.6a  --  T_kine fill (highest summed kine_charge, NO mass window)")
    print("                vs the identification (mass window, writes pio_id)")
    print("=" * 78)
    print("%-9s %-24s %9s %5s %6s %28s" %
          ("event", "pi0 group", "fill_mass", "flag", "n_id", "identified mass(es)"))
    agree = differ = fill_only = none_both = 0
    for ev in sorted(labs):
        d = dumps.get(ev)
        if not d:
            continue
        b = buck.get(ev) or {}
        if b.get("pi0", "-") == "-":
            continue
        c = pi0_compare(d)
        masses = ", ".join("%.1f" % (g["mass"] or 0) for g in c["ident"]) or "-"
        print("%-9d %-24s %9s %5s %6d %28s%s" %
              (ev, b.get("pi0", "?")[:24],
               fmt(c["fill_mass"], 1) if c["fill_mass"] else "-",
               c["fill_flag"], c["n_ident"], masses,
               "" if c["matched"] is not None else "   <- fill is NOT an identified pair"))
        if c["n_ident"] == 0 and not c["fill_mass"]:
            none_both += 1
        elif c["n_ident"] == 0:
            fill_only += 1
        elif c["matched"] is not None:
            agree += 1
        else:
            differ += 1
    print("\nfill agrees with an identified pair : %d" % agree)
    print("fill differs from every identified pair: %d" % differ)
    print("fill filled but NOTHING identified    : %d   <- kine_pio_mass with no pi0" % fill_only)
    print("neither                                : %d" % none_both)
    return 0


def report_level(labs, dumps, buck, preps):
    print("=" * 78)
    print("sec 15.5  --  are the two sides the scanner drew already in different CLUSTERS?")
    print("   CROSS-CLUSTER -> image separation already worked; a SHOWER pass crossed")
    print("                    a correct boundary.  Lever = the absorb cones.")
    print("   one-cluster   -> pr/53 / pr/57 sec 14 territory, or a shower-level split.")
    print("=" * 78)
    ev_cross, ev_one, seen = set(), set(), set()
    for ev in sorted(labs):
        b = (buck.get(ev) or {}).get("bucket", "")
        if not (b.startswith("2 ") or b == "1+2 both"):
            continue
        d = dumps.get(ev)
        if not d:
            continue
        seen.add(ev)
        res = level_test(labs[ev], d, preps.get(ev))
        if not res:
            print("%-9d %-14s (marks carry no two-sided partition)" % (ev, b[:14]))
            continue
        for r in res:
            print("%-9d %-14s shw=%-8d %-12s A=%-3d B=%-3d clA=%-16s clB=%-16s %s" %
                  (ev, b[:14], r["shower"], r["rule"], r["n_a"], r["n_b"],
                   str(r["clusters_a"])[:16], str(r["clusters_b"])[:16],
                   "CROSS-CLUSTER" if r["cross_cluster"]
                   else "one-cluster %s" % r["shared"]))
            (ev_cross if r["cross_cluster"] else ev_one).add(ev)
    # Count EVENTS: 47212 / 76346 / 269774 each emit two partition rows, so a
    # row count would read 9-of-10 where the event count reads 6-of-7.
    print("\nover-clustered events examined        : %d" % len(seen))
    print("  with a two-sided partition          : %d" % len(ev_cross | ev_one))
    print("  cross-cluster (shower overreached)  : %d  %s"
          % (len(ev_cross), sorted(ev_cross)))
    print("  one-cluster (pr/53 / pr/57 or split): %d  %s"
          % (len(ev_one), sorted(ev_one)))
    print("\nA cross-cluster row needs a cone guard in the shower passes, NOT a new")
    print("split pass and NOT pr/57 sec 14 -- the clusters were already right.")
    return 0


def report_absorb(labs, buck):
    """Which pass put each wrongly-held segment where it is.

    `marks_detail[shw].marked[sid].absorbed_by` is recorded by the display from
    the stage-2 probe, so every mark carries the name of the pass that absorbed
    that segment (or None -- never absorbed by anything, i.e. an orphan).  This
    turns "over-clustering" into a ranked list of call sites."""
    groups = [("over-clustered (+ both): segments that should NOT be there",
               lambda b: b.startswith("2 ") or b == "1+2 both"),
              ("under-clustered: segments the reco MISSED",
               lambda b: b.startswith("1 "))]
    for title, pick in groups:
        print("=" * 78)
        print(title)
        print("=" * 78)
        c, byev = Counter(), defaultdict(Counter)
        for ev in sorted(labs):
            b = (buck.get(ev) or {}).get("bucket", "")
            if not pick(b):
                continue
            md = (labs[ev].get("em") or {}).get("marks_detail") or {}
            for det in md.values():
                for m in (det.get("marked") or {}).values():
                    src = m.get("absorbed_by") or "(never absorbed - orphan)"
                    c[src] += 1
                    byev[src][ev] += 1
        tot = sum(c.values()) or 1
        print("%-38s %6s %6s   %s" % ("absorbed_by", "n", "share", "events"))
        for k, v in c.most_common():
            evs = ",".join(str(e) for e, _ in byev[k].most_common(4))
            print("%-38s %6d %5.0f%%   %s%s" %
                  (k, v, 100.0 * v / tot, evs, " ..." if len(byev[k]) > 4 else ""))
        print()
    return report_where(labs, buck)


def report_where(labs, buck):
    """Disambiguate what `absorbed_by` means on an IN mark.

    On an OUT mark it names the pass that wrongly PUT the segment in the shower.
    On an IN mark it names where the segment sits NOW -- which is a different
    statement, and the one that decides whether a pass is over-reaching or
    under-reaching.  If an IN-marked segment currently sits in a NEIGHBOUR
    shower, the named pass absorbed it into the wrong object: that is
    over-reach, seen from the other side, not failure to reach."""
    print("=" * 78)
    print("under-clustered: for each IN mark, where is that segment NOW?")
    print("  neighbour -> the named pass absorbed it into the WRONG shower (over-reach)")
    print("  orphan    -> nothing absorbed it at all (genuine non-reach: the stub class)")
    print("=" * 78)
    tab, tot = Counter(), Counter()
    for ev in sorted(labs):
        if not (buck.get(ev) or {}).get("bucket", "").startswith("1 "):
            continue
        for shw, det in ((buck and (labs[ev].get("em") or {}).get("marks_detail")) or {}).items():
            for sid, m in (det.get("marked") or {}).items():
                if m.get("kind") != "in":
                    continue
                src = m.get("absorbed_by") or "(never absorbed)"
                own = m.get("owner")
                if own is None:
                    where = "unowned"
                elif int(own) == int(shw):
                    where = "already in the scanned shower"
                elif int(own) == int(sid):
                    where = "orphan"
                else:
                    where = "neighbour shower"
                tab[(src, where)] += 1
                tot[where] += 1
    print("%-38s %-30s %s" % ("absorbed_by", "where it sits now", "n"))
    for (src, where), n in sorted(tab.items(), key=lambda kv: -kv[1]):
        print("%-38s %-30s %d" % (src, where, n))
    print("\ntotals: %s" % dict(tot))
    print("\nSo the two columns above are largely ONE defect seen twice: a cone that")
    print("absorbs a segment into the wrong shower is over-clustering for the shower")
    print("that gained it and under-clustering for the shower that lost it.")
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


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Score EM shower clustering against the hand scan -- CROSS-RUN edition.

A FORK of em115_score.py, not a generalisation of it (CLAUDE.md sec 2 "fork by
duplication"; the run_em114c_probe.sh precedent).  em115_score.py stays pinned
to the prod0825 dumps the scan was made on; THIS script adds what a knob round
needs (doc pr/117):

    --manifest M --prepdir D    score a NEW reconstruction (defaults = the
                                em114 paths, so a no-flag run reproduces
                                em115_score.py's stdout as the fork-fidelity
                                gate)
    cross-run shower matching   the label's shower key is the SCAN-TIME start
                                segment display id; after a knob changes
                                membership a shower can re-root and rename, and
                                the exact-key join would score the rename as a
                                catastrophe.  With --cross-run (implied by a
                                non-default --manifest/--prepdir) each labelled
                                shower is matched to the reconstructed shower
                                with the largest charge-weighted overlap with
                                its target set; the matched node id is reported.
    --diffstat BASE NEW         per-event membership diff between two prepdirs
                                over ALL events present in both -- the
                                hold-flat check for the 37 "good" + 13
                                scanned-no-correction events, which carry no
                                marks and hence no score rows.
    drift reinterpretation      sidecar-vs-label drift is an integrity check
                                ONLY when scoring the scan-time run; on a
                                changed reconstruction nonzero drift is the
                                expected signal, demoted to an info line.

Everything else -- target definition, charge weighting, the event scalar, the
--pi0/--level/--absorb reports, the delta mode -- is em115_score.py verbatim.

    ./em117_score.py                       # fork-fidelity: == em115_score.py
    ./em117_score.py --manifest M --prepdir D --tsv OUT.tsv
    ./em117_score.py --baseline A.tsv --compare B.tsv
    ./em117_score.py --diffstat emprep emprep-117on

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
        "shower": shw,
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


def score_event(rec, dump, prep=None, cross_run=False):
    actual, seginfo, join_actual = digest_dump(dump, prep)
    md = (rec.get("em") or {}).get("marks_detail") or {}
    rows = []
    for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
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
    dumps, preps = {}, {}
    for ev in labs:
        m = man.get(ev)
        dumps[ev] = load_dump(m["dump"]) if m and m.get("dump") else None
        preps[ev] = load_prep(ev, args.prepdir)

    return report_score(labs, dumps, buck, args.tsv, preps, cross_run)


def report_score(labs, dumps, buck, tsv_path, preps, cross_run=False):
    rows, evs = [], {}
    for ev in sorted(labs):
        d = dumps.get(ev)
        if not d:
            continue
        e, rr = score_event(labs[ev], d, preps.get(ev), cross_run)
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
        cols = ["event", "bucket", "shower", "matched", "n_target", "n_actual", "n_inter",
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

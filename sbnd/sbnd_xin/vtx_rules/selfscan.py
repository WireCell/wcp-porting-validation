"""Blind self-scan harness: can an AI session scan events by EYE? (doc pr/80 sec 7, 9)

`eval_rules.py` measures the *engine*.  This measures a different scanner: an
agent looking at rendered pictures and applying the sec 3 procedure with
judgement, the way the owner does.  There is no reason to assume the two score
alike, so the question is answered by measurement rather than by assertion.

Three rules make the number honest, and all three are enforced here rather than
trusted:

  * `prepare` NEVER renders or prints the truth, the reconstructed vertex, or the
    engine's pick.  For `--kit new` that guarantee is structural -- scankit.py
    strips the fields from the dump before any renderer sees it.  A scanner that
    can see the reco star will anchor on it, and labels that only confirm the
    reconstruction are worse than no labels: they teach a future fit that the
    reconstruction is right on exactly the events where it is wrong.
  * `score` refuses to run until every worker's picks file exists, and records
    each file's mtime beside the result, so "picks were written before truth was
    read" is auditable after the fact rather than promised.
  * The candidate pool the scorer resolves against is EVERY PR-graph vertex.
    The first round's `prepare` filtered to the main cluster, which capped the
    scanner at a 92.0% ceiling at 1 cm against 97.7% for the full pool -- events
    were scored wrong that could not have been answered right.  The kit=old arm
    still *shows* the main-cluster-only worksheet, because it exists to
    reproduce the first round's conditions faithfully; only the scoring lookup
    is repaired.

  cd sbnd_xin
  python3 vtx_rules/selfscan.py prepare --half dev --n 60 --kit new \\
      --workers 4 --out /home/xqian/tmp/scan2/dev-new
  # ... workers look at the panels and write picks-<w>.json ...
  python3 vtx_rules/selfscan.py score --dir /home/xqian/tmp/scan2/dev-new
  python3 vtx_rules/selfscan.py compare --a <dirA> --b <dirB>
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines                                                 # noqa: E402
import make_split                                                # noqa: E402
import render_event                                              # noqa: E402
import scankit                                                   # noqa: E402
import vtx_io                                                    # noqa: E402


def _labels(half, exclude=()):
    keep = make_split.load_split()[half]
    labs = [L for L in vtx_io.load_labels()
            if L["key"] in keep and L["event"] not in exclude
            and baselines.deployed_dump_path(L)]
    labs.sort(key=lambda L: (L["tag"], L.get("runNo") or -1,
                             L.get("subRunNo") or -1, L.get("eventNo") or -1))
    return labs


def _used(path):
    """Events already scanned by eye, read from a previous run's manifest."""
    if not path:
        return set()
    with open(path) as fh:
        return {m["event"] for m in json.load(fh)}


def prepare(half, n, out, seed, kit, workers, exclude_manifest,
            only_manifest=None):
    if only_manifest:
        # Re-scan an exact, previously-scanned event set (the §7 twenty) with a
        # different kit.  Diagnostic only: those events' answers are known to
        # whoever launches the run, so it must never be scored before the
        # held-out arm is committed.
        keep = _used(only_manifest)
        labs = [L for L in vtx_io.load_labels()
                if L["event"] in keep and baselines.deployed_dump_path(L)]
        labs.sort(key=lambda L: L["event"])
    else:
        labs = _labels(half, _used(exclude_manifest))
        rng = random.Random(seed)
        rng.shuffle(labs)
        labs = labs[:n]
        labs.sort(key=lambda L: L["event"])

    os.makedirs(out, exist_ok=True)
    manifest = []
    for L in labs:
        path = baselines.deployed_dump_path(L)
        with open(path) as fh:
            raw = json.load(fh)
        d = scankit.sanitize(raw)
        scankit.assert_blind(d, L["event"])

        # The scorer's lookup table: every vertex, every cluster, always.
        cands = [dict(vertex_id=v["id"], cluster_id=v["cluster_id"],
                      degree=v.get("degree", 0),
                      x=round(p[0], 2), y=round(p[1], 2), z=round(p[2], 2))
                 for v in scankit.candidates(d)
                 for p in [scankit.vertex_xyz(v)]]

        ed = os.path.join(out, L["event"])
        if kit == "new":
            made = scankit.prepare(path, ed, title=L["event"])
        else:
            # Faithful reproduction of the first round: one flat PNG, and a
            # worksheet listing only the main cluster's vertices.  Reproducing
            # the old arm honestly means reproducing its blind spot too.
            os.makedirs(ed, exist_ok=True)
            png = os.path.join(ed, "event.png")
            render_event.render(raw, png, title=L["event"], blind=True)
            cid = vtx_io.main_cluster_id(raw)
            shown = [c for c in cands if c["cluster_id"] == cid]
            with open(os.path.join(ed, "candidates.txt"), "w") as fh:
                fh.write("%s  cluster %s  %d candidates\n"
                         % (L["event"], cid, len(shown)))
                for c in shown:
                    fh.write("  vertex %-8s deg %-2s (%8.2f, %8.2f, %8.2f)\n"
                             % (c["vertex_id"], c["degree"], c["x"], c["y"],
                                c["z"]))
            made = ["event.png", "candidates.txt"]
        manifest.append(dict(event=L["event"], key=list(L["key"]), dump=path,
                             dir=ed, files=made, candidates=cands))

    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    # Split into worker shards by position, so each worker gets a contiguous,
    # reproducible slice and two runs of the same command shard identically.
    per = (len(manifest) + workers - 1) // workers
    for w in range(workers):
        chunk = manifest[w * per:(w + 1) * per]
        if not chunk:
            continue
        with open(os.path.join(out, "worklist-%d.txt" % w), "w") as fh:
            # event, panel dir, dump -- the dump so a worker can drive
            # `scankit.py zoom` without being handed the whole manifest (which
            # would cost it a large read it does not otherwise need).
            for m in chunk:
                fh.write("%s\t%s\t%s\n" % (m["event"], m["dir"], m["dump"]))
    with open(os.path.join(out, "KIT"), "w") as fh:
        fh.write("%s\n" % kit)
    print("prepared %d blind events (kit=%s) in %s, %d worklists"
          % (len(manifest), kit, out, min(workers, len(manifest))))
    return 0


def _load_picks(out, need):
    files = sorted(glob.glob(os.path.join(out, "picks-*.json")))
    if not files:
        alt = os.path.join(out, "picks.json")
        if os.path.exists(alt):
            files = [alt]
    if not files:
        print("no picks-*.json in %s -- write the picks BEFORE scoring." % out)
        return None, None
    picks, stamps = {}, {}
    for f in files:
        with open(f) as fh:
            for p in json.load(fh):
                picks[p["event"]] = p
        stamps[os.path.basename(f)] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))
    # A scanner that is told to write its output file early will write stubs
    # first and fill them in later.  A stub is an abstention with no reasoning,
    # and scoring one as a considered abstention would silently understate the
    # answer rate of a worker that died halfway.  Refuse instead.
    stub = [e for e, p in picks.items()
            if len((p.get("why") or "").strip()) < 25]
    if stub:
        print("%d picks have no substantive reasoning (e.g. %s) -- these look "
              "like placeholders from a worker that did not finish. Refusing "
              "to score; a stub abstention is not an abstention."
              % (len(stub), ", ".join(sorted(stub)[:5])))
        return None, None
    missing = sorted(need - set(picks))
    if missing:
        print("picks missing for %d events (%s%s) -- refusing to score a "
              "partial run, since dropping events silently moves every "
              "denominator." % (len(missing), ", ".join(missing[:6]),
                                " ..." if len(missing) > 6 else ""))
        return None, None
    return picks, stamps


def score(out):
    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = {m["event"]: m for m in json.load(fh)}
    picks, stamps = _load_picks(out, set(manifest))
    if picks is None:
        return 1

    labs = {L["event"]: L for L in vtx_io.load_labels()}
    rows = []
    for ev, m in sorted(manifest.items()):
        L = labs[ev]
        p = picks[ev]
        pos = None
        for c in m["candidates"]:
            if c["vertex_id"] == p.get("vertex_id"):
                pos = (c["x"], c["y"], c["z"])
        d = vtx_io.dist(L["truth"], pos)
        with open(m["dump"]) as fh:
            dump = json.load(fh)
        reco = vtx_io.dist(L["truth"], vtx_io.xyz(dump.get("main_vertex")))
        # The best any vertex-selection procedure could have done here.  The
        # owner's click is not snapped to a vertex, so this is not always 0.
        reach = [x for x in (vtx_io.dist(L["truth"], (c["x"], c["y"], c["z"]))
                             for c in m["candidates"]) if x is not None]
        best = min(reach) if reach else None
        # Distance is the pr/78-79 convention and is kept as the headline so the
        # numbers stay comparable.  But the owner's click is snapped to a vertex
        # (470/470 labels sit exactly on their recorded vertex id IN THE ARM THE
        # LABEL WAS TAKEN ON), and `improve_vertex` refits trajectories after the
        # vertex is chosen, so the SAME vertex id moves between arms -- 2.1% of
        # labels by more than 1 cm, up to 10.7 cm.  Scoring a scan taken on the
        # deployed arm against a click recorded on the prod0813 arm therefore
        # charges the scanner for a refit it can neither see nor control.
        # `ok_vid` is the same question asked without that artifact.
        ok_vid = (L["truth_vid"] is not None
                  and p.get("vertex_id") == L["truth_vid"])
        rows.append(dict(event=ev, conf=p.get("confidence", "-"),
                         vid=p.get("vertex_id"), dist=d,
                         ok=vtx_io.correct(d), ok3=vtx_io.correct(d, 3.0),
                         ok_vid=ok_vid, truth_vid=L["truth_vid"],
                         reco=reco, reco_ok=vtx_io.correct(reco),
                         reachable=vtx_io.correct(best), best=best,
                         why=p.get("why", "")))
    report(rows, out, stamps)
    with open(os.path.join(out, "scored.json"), "w") as fh:
        json.dump(dict(picks_written=stamps, scored_at=time.strftime(
            "%Y-%m-%d %H:%M:%S"), rows=rows), fh, indent=1, default=str)
    return 0


def report(rows, out, stamps):
    print("%-12s %-8s %8s %7s %7s %8s" % ("event", "conf", "dist_cm", "ok@1cm",
                                          "ok@3cm", "reco_ok"))
    for r in rows:
        print("%-12s %-8s %8s %7s %7s %8s   %s"
              % (r["event"], r["conf"],
                 ("%.2f" % r["dist"]) if r["dist"] is not None else "-",
                 "yes" if r["ok"] else "no", "yes" if r["ok3"] else "no",
                 "yes" if r["reco_ok"] else "no", r["why"][:52]))
    n = len(rows)
    ans = [r for r in rows if r["vid"] is not None]
    ok = sum(1 for r in ans if r["ok"])
    print("\npicks files written at: %s" % stamps)
    print("scanned %d, answered %d (%.0f%%), correct %d "
          "(precision %.1f%%, @3cm %.1f%%)"
          % (n, len(ans), 100.0 * len(ans) / n, ok,
             100.0 * ok / max(len(ans), 1),
             100.0 * sum(1 for r in ans if r["ok3"]) / max(len(ans), 1)))
    okv = sum(1 for r in ans if r["ok_vid"])
    havev = [r for r in ans if r["truth_vid"] is not None]
    print("by VERTEX ID (no cross-arm refit artifact): %d of %d answered with a "
          "recorded truth id (%.1f%%)"
          % (okv, len(havev), 100.0 * okv / max(len(havev), 1)))
    print("reconstruction on the same %d events: %d correct (%.1f%%)"
          % (n, sum(1 for r in rows if r["reco_ok"]),
             100.0 * sum(1 for r in rows if r["reco_ok"]) / n))
    unreach = [r for r in rows if not r["reachable"]]
    print("events where NO vertex in this arm is within 1 cm of the click: %d "
          "(%.1f%%) -- unanswerable by the distance metric; the click IS "
          "snapped, but to a vertex in the arm the label was taken on"
          % (len(unreach), 100.0 * len(unreach) / n))

    # The number the owner's workflow depends on.  If these tiers are flat,
    # "leave the unsure ones to me" does not work: the errors are not in the
    # pile that gets handed back.
    # COVERAGE is printed beside accuracy on purpose.  `certain` is not a
    # fixed-size stratum -- a scanner that spends the word freely can put half
    # the sample there.  A `certain` tier holding 20% of events at 93% and one
    # holding 65% at 85% are very different products, and an accuracy-only
    # table scores the first as the winner while the second may be the one the
    # owner actually wants.  Both numbers, always.
    print("\nconfidence calibration (this is what decides the hand-back "
          "workflow -- read coverage and accuracy TOGETHER):")
    for tier in ("certain", "likely", "unclear"):
        t = [r for r in rows if r["conf"] == tier]
        ta = [r for r in t if r["vid"] is not None]
        k = sum(1 for r in ta if r["ok"])
        kv = sum(1 for r in ta if r["ok_vid"])
        if t:
            print("   %-8s coverage %3d/%d (%4.1f%%)  answered %3d  "
                  "correct %3d (%.1f%%)  by-vid %3d (%.1f%%)"
                  % (tier, len(t), n, 100.0 * len(t) / n, len(ta),
                     k, 100.0 * k / max(len(ta), 1),
                     kv, 100.0 * kv / max(len(ta), 1)))

    # Triage value: is the reco wrong more often where the scanner declined or
    # disagreed?  Enrichment, on consistent denominators -- NOT the precision on
    # the agreeing subset, which is the reconstruction's own accuracy wearing
    # the scanner's name (doc pr/80 sec 4.1).
    agree = {r["event"] for r in rows
             if r["vid"] is not None and r["dist"] is not None
             and r["reco"] is not None and abs(r["dist"] - r["reco"]) <= 1.0}
    dis = [r for r in rows if r["event"] not in agree]
    br = sum(1 for r in rows if not r["reco_ok"]) / float(n)
    if dis:
        bd = sum(1 for r in dis if not r["reco_ok"]) / float(len(dis))
        print("\nreco wrong on %.1f%% of all events, %.1f%% of the %d the "
              "scanner declined or disagreed with -> enrichment x%.2f"
              % (100 * br, 100 * bd, len(dis), (bd / br) if br else 0))


def compare(a, b):
    """Paired old-vs-new (or old-vs-old) on the same events: discordant counts.

    McNemar's setup.  Concordant pairs carry no information about which kit is
    better and are reported only so the reader can see how much of the sample
    the comparison actually rests on.
    """
    def load(d):
        with open(os.path.join(d, "scored.json")) as fh:
            return {r["event"]: r for r in json.load(fh)["rows"]}
    A, B = load(a), load(b)
    ev = sorted(set(A) & set(B))
    print("paired on %d common events:  A = %s   B = %s"
          % (len(ev), os.path.basename(a.rstrip("/")),
             os.path.basename(b.rstrip("/"))))
    for key, name in (("ok", "correct @1cm"), ("ok_vid", "correct by vertex id")):
        n01 = [e for e in ev if not A[e].get(key) and B[e].get(key)]
        n10 = [e for e in ev if A[e].get(key) and not B[e].get(key)]
        both = sum(1 for e in ev if A[e].get(key) and B[e].get(key))
        neither = len(ev) - both - len(n01) - len(n10)
        print("\n  --- %s ---   A %d/%d, B %d/%d"
              % (name, both + len(n10), len(ev), both + len(n01), len(ev)))
        print("    both %3d    neither %3d" % (both, neither))
        print("    B only (B wins) %2d   %s" % (len(n01), " ".join(n01[:12])))
        print("    A only (A wins) %2d   %s" % (len(n10), " ".join(n10[:12])))
        d = len(n01) + len(n10)
        print("    discordant %d; the comparison rests on these alone" % d)
        if d:
            # Exact two-sided binomial under H0: a discordant pair is a coin
            # flip.  No pass/fail threshold is applied -- a pre-registered cut
            # would call a real but half-size effect negative, and the counts
            # plus the old-vs-old noise floor are the honest report.
            from math import comb
            k = min(len(n01), len(n10))
            p = min(2.0 * sum(comb(d, i) for i in range(k + 1)) / (2.0 ** d), 1.0)
            print("    exact binomial p = %.4f (two-sided, H0: kit irrelevant)"
                  % p)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--half", default="test", choices=["dev", "test"])
    p1.add_argument("--n", type=int, default=60)
    p1.add_argument("--seed", type=int, default=20260815)
    p1.add_argument("--kit", default="new", choices=["old", "new"])
    p1.add_argument("--workers", type=int, default=4)
    p1.add_argument("--exclude-manifest",
                    help="a previous manifest.json whose events to skip")
    p1.add_argument("--only-manifest",
                    help="re-scan exactly a previous manifest.json's events "
                         "(diagnostic; those answers are already known)")
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("score")
    p2.add_argument("--dir", required=True)
    p3 = sub.add_parser("compare")
    p3.add_argument("--a", required=True)
    p3.add_argument("--b", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.half, a.n, a.out, a.seed, a.kit, a.workers,
                       a.exclude_manifest, a.only_manifest)
    if a.cmd == "score":
        return score(a.dir)
    return compare(a.a, a.b)


if __name__ == "__main__":
    sys.exit(main())

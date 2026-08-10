#!/usr/bin/env python3
"""doc pr/57 round 4: pair-level connectivity for the S6 separation scan.

The scan dump records one line per S6-evaluated CANDIDATE edge with a `killed`
bit.  That bit does not answer the question the hand scan is actually asking --
"should these two components be separated" -- because

  * one component pair (j,k) gets up to three independent candidates
    (closest / dir1 / dir2), so killing one leaves the others free to connect
    the very same pair, and
  * even with every (j,k) candidate killed, j and k can stay in one piece
    through a third component m.

`connect_graph_relaxed_strict.cxx` therefore emits one "connectivity" record
per graph call, after every emit decision is final: the final connected-
component label of each starting component (`final`) plus the component edges
the call actually emitted (`edges`, with endpoints).  This module turns those
into a per-pair answer, and is imported by the viewer, oc56_autoscan.py and
oc56_dump_check.py so all three agree by construction.

`final` and `edges` are computed independently inside the C++, so
`separated == not reachable(edges)` is a real cross-check, not a tautology --
check_consistency() below is what oc56_dump_check.py runs on it.

Self-test (no dump needed):  ./oc56_conn.py --selftest
"""
import collections
import sys


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def index_conn(records):
    """{graph_call: connectivity record} from an iterable of dump records."""
    return {r["graph_call"]: r for r in records if r.get("type") == "connectivity"}


def _adjacency(rec):
    adj = collections.defaultdict(list)
    for e in rec.get("edges", []):
        adj[e["j"]].append((e["k"], e))
        adj[e["k"]].append((e["j"], e))
    return adj


# ---------------------------------------------------------------------------
# The per-pair answer
# ---------------------------------------------------------------------------
def pair_status(rec, j, k):
    """Connectivity of starting components j and k after this graph call.

    Returns a dict; `known` is False (and everything else None/empty) when the
    dump predates round 4 and carries no connectivity record -- callers must
    degrade gracefully rather than assume "connected".

      known      -- the connectivity record was present
      separated  -- j and k ended in different final components
      direct     -- emitted edges whose endpoints are exactly this pair
      path       -- [(a, b, edge), ...] shortest hop chain j -> ... -> k,
                    [] when separated, one hop when direct
      hops       -- len(path)
    """
    out = dict(known=False, separated=None, direct=[], path=[], hops=0)
    if rec is None:
        return out
    out["known"] = True

    final = rec.get("final", [])
    if j < len(final) and k < len(final):
        out["separated"] = final[j] != final[k]

    edges = rec.get("edges", [])
    out["direct"] = [e for e in edges
                     if (e["j"], e["k"]) in ((j, k), (k, j))]

    # Shortest chain of emitted component edges, BFS from j.  Ties are broken
    # by the shorter edge so the reported chain is reproducible run to run
    # (never by dict/set order -- these are integer-keyed, but the edge choice
    # would otherwise depend on emit order).
    adj = _adjacency(rec)
    prev = {j: None}
    queue = collections.deque([j])
    while queue and k not in prev:
        v = queue.popleft()
        for w, e in sorted(adj[v], key=lambda t: (t[0], t[1]["dis"])):
            if w in prev:
                continue
            prev[w] = (v, e)
            queue.append(w)
    if k in prev:
        chain = []
        cur = k
        while prev[cur] is not None:
            v, e = prev[cur]
            chain.append((v, cur, e))
            cur = v
        chain.reverse()
        out["path"] = chain
        out["hops"] = len(chain)
    return out


def siblings(edge_records, graph_call, j, k, exclude=None):
    """The other S6 candidates dumped for this same component pair.

    exclude -- an edge record to leave out (the one under review)."""
    out = []
    for e in edge_records:
        if e.get("graph_call") != graph_call:
            continue
        if (e["j"], e["k"]) not in ((j, k), (k, j)):
            continue
        if exclude is not None and e is exclude:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def short_token(st):
    """One compact cell for the scan table's `pair` column."""
    if not st["known"]:
        return "?"
    if st["separated"]:
        return "SEP"
    if st["direct"]:
        return "dir"
    if st["hops"]:
        return "via %d" % st["hops"]
    # final[] says connected but no emitted chain reaches it -- a real
    # inconsistency, surfaced rather than hidden (see check_consistency).
    return "??"


def describe(st, j, k):
    """One human line for the selected-edge panel."""
    if not st["known"]:
        return ("pair j=%d k=%d -> connectivity UNKNOWN "
                "(pre-round-4 dump: no connectivity record)" % (j, k))
    if st["separated"]:
        return "pair j=%d k=%d -> SEPARATED (no remaining path)" % (j, k)
    if st["direct"]:
        best = min(st["direct"], key=lambda e: e["dis"])
        others = ", ".join(sorted({e["src"] for e in st["direct"]}))
        return ("pair j=%d k=%d -> STILL CONNECTED, direct %s edge dis=%.2f cm"
                " [emitted: %s]" % (j, k, best["src"], best["dis"], others))
    if st["hops"]:
        route = " -> ".join([str(st["path"][0][0])] + [str(b) for _, b, _ in st["path"]])
        hops = ", ".join("%d-%d %s %.2fcm" % (a, b, e["src"], e["dis"])
                         for a, b, e in st["path"])
        # SHORTEST route -- other independent routes may also exist, so this
        # is "a thing holding them together", not necessarily "the" thing.
        # `separated` comes from final[], so the verdict is right either way.
        return ("pair j=%d k=%d -> STILL CONNECTED via %s   [%s]"
                "   (shortest route; others may exist)"
                % (j, k, route, hops))
    return ("pair j=%d k=%d -> INCONSISTENT: final[] says connected, "
            "no emitted chain reaches it" % (j, k))


# ---------------------------------------------------------------------------
# Cross-check: final[] (from the graph) vs edges[] (from the emit sites)
# ---------------------------------------------------------------------------
def check_consistency(rec):
    """[] when the record is self-consistent, else a list of complaints.

    For every pair of starting components, `final[j] == final[k]` must hold
    exactly when j reaches k over the emitted component edges.  The two sides
    come from two independent C++ computations, so a disagreement is a bug in
    the record, not a rounding difference."""
    bad = []
    final = rec.get("final", [])
    n = len(final)
    adj = _adjacency(rec)

    # One BFS per component is O(n * E); n is small (components of one
    # cluster), and this runs only in the checker.
    for j in range(n):
        seen = {j}
        queue = collections.deque([j])
        while queue:
            v = queue.popleft()
            for w, _ in adj[v]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        for k in range(j + 1, n):
            same_final = final[j] == final[k]
            reachable = k in seen
            if same_final != reachable:
                bad.append("graph_call=%s j=%d k=%d final_same=%s reachable=%s"
                           % (rec.get("graph_call"), j, k, same_final, reachable))
    return bad


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    def E(j, k, src, dis):
        return dict(j=j, k=k, src=src, dis=dis, p1=[0, 0, 0], p2=[1, 1, 1], dup=False)

    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    # 0-1 direct; 0-2 via 1; 3 isolated.  final labels match that shape.
    rec = dict(type="connectivity", graph_call=7, ncomp=4, nfinal=2,
               final=[0, 0, 0, 1],
               edges=[E(0, 1, "mst", 1.10), E(1, 2, "dir1", 0.44)])

    st = pair_status(rec, 0, 1)
    check("direct.known", st["known"], True)
    check("direct.separated", st["separated"], False)
    check("direct.hops", st["hops"], 1)
    check("direct.ndirect", len(st["direct"]), 1)
    check("direct.token", short_token(st), "dir")

    st = pair_status(rec, 0, 2)
    check("via.separated", st["separated"], False)
    check("via.ndirect", len(st["direct"]), 0)
    check("via.hops", st["hops"], 2)
    check("via.route", [(a, b) for a, b, _ in st["path"]], [(0, 1), (1, 2)])
    check("via.token", short_token(st), "via 2")

    st = pair_status(rec, 0, 3)
    check("sep.separated", st["separated"], True)
    check("sep.hops", st["hops"], 0)
    check("sep.token", short_token(st), "SEP")

    st = pair_status(None, 0, 1)
    check("missing.known", st["known"], False)
    check("missing.separated", st["separated"], None)
    check("missing.token", short_token(st), "?")

    check("consistent", check_consistency(rec), [])

    # Same pair emitted twice (closest killed, dir1 survived is the common
    # case; here both survive) -- both must be reported as direct.
    rec2 = dict(graph_call=1, final=[0, 0], type="connectivity",
                edges=[E(0, 1, "mst", 2.4), E(0, 1, "dir1", 0.82)])
    st = pair_status(rec2, 0, 1)
    check("twice.ndirect", len(st["direct"]), 2)
    check("twice.best", describe(st, 0, 1).count("dir1"), 2)  # best + emitted list

    # A record whose final[] disagrees with its edges[] must be caught.
    rec3 = dict(graph_call=2, type="connectivity", final=[0, 1], edges=[E(0, 1, "mst", 1.0)])
    if not check_consistency(rec3):
        fails.append("inconsistent record was not flagged")

    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("oc56_conn selftest: PASS (%d cases)" % 18)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)

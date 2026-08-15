"""Census behind doc pr/84 -- the "disconnected gamma" bug and near-vertex PR.

Every number in docs/pr/84_disconnected-gamma-and-near-vertex-pr.md comes from
one run of this script.  Read-only: it opens the deployed-arm calib dumps and
Bee archives and writes nothing.

The point of the script is a NEGATIVE result.  A gamma node in the Bee particle
tree is not a particle-ID statement at all -- it is a synthetic parent that
MultiAlgBlobClustering.cxx:1776 splices in above any shower whose
start_connection_type is not 1.  The obvious way to suppress the nonsensical
ones is to threshold on the RENDERED length of that node, and `discriminator()`
shows that would be wrong: the zero-length ones are the genuinely REMOTE
associations (median 38.8 cm of real charge to the neutrino vertex, none in the
main cluster) whose parent merely fails to draw its own gap.  What actually
selects the owner's class is the distance from the shower's nearest CHARGE to
the neutrino vertex, together with main-cluster membership.

Run:
  cd sbnd_xin && python3 pr84_pseudo_parent_census.py > /home/xqian/tmp/pr84.txt 2>&1; echo rc=$?
"""
import collections
import glob
import json
import math
import os
import re
import subprocess
import zipfile

# The deployed operating point (min_accept = 10, doc pr/79).  Named explicitly
# rather than globbed: there are five work-*-ma10* variants and only these
# three are the prod0813 deployed arms.
ARMS = ("work-mcp1k-ma10", "work-ncpi0-ma10", "work-nuecc48-ma10")

KE_MIN = 20.0        # MeV; below this a pseudo-parent is not worth arguing about
TOUCH = 3.0          # cm; "the shower's charge touches the neutrino vertex"
MICRO = 2.0          # cm; a PF node this short is a micro-parent candidate
MICRO_CHILD = 10.0   # cm; ... parenting a child at least this long
MICRO_RATIO = 5.0    # ... and at least this many times longer than itself


def dumps():
    """(path, parsed calib dump) for every deployed-arm event."""
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(arm, "pr_evt*", "calib-pr-evt*.json"))):
            try:
                with open(p) as fh:
                    yield p, json.load(fh)
            except (OSError, ValueError):
                continue


def bee_trees():
    """(eventNo, particle-flow tree) from every deployed-arm Bee archive.

    The tree is Bee's `0-mc.json`: a jsTree forest of nodes carrying
    {"text": "<name>  <KE> MeV", "data": {"start": [...], "end": [...]}}.
    """
    for arm in ARMS:
        for z in sorted(glob.glob(os.path.join(arm, "pr_evt*", "mabc-pr.zip"))):
            try:
                with zipfile.ZipFile(z) as zf:
                    names = [n for n in zf.namelist() if n.endswith("0-mc.json")]
                    if not names:
                        continue
                    tree = json.loads(zf.read(names[0]))
            except (OSError, ValueError, zipfile.BadZipFile):
                continue
            yield re.search(r"pr_evt(\d+)", z).group(1), tree


def node_len(n):
    return math.dist(n["data"]["start"], n["data"]["end"])


def node_name(n):
    return n["text"].split()[0]


def node_ke(n):
    try:
        return float(n["text"].split()[1])
    except (IndexError, ValueError):
        return 0.0


def walk(forest):
    """Yield (node, parent) over the whole forest, parent None at the roots."""
    stack = [(n, None) for n in reversed(forest)]
    while stack:
        node, parent = stack.pop()
        yield node, parent
        for child in reversed(node.get("children", [])):
            stack.append((child, node))


def vertex_xyz(v):
    f = v["fit"]
    return (f["x"], f["y"], f["z"])


def shower_rows(dump):
    """One row per conn-2/3 shower, with the two competing distance measures.

    `flight` is what the Bee tree draws: anchor vertex -> shower start point.
    For conn 2/3 the start point is DERIVED from the anchor (PRShower.cxx:1140),
    which is why it collapses to zero whenever the PF writer passes the shower's
    own start vertex as the connection vertex (MultiAlgBlobClustering.cxx:1562).

    `charge_to_nu` is the honest measure: the closest fitted point of the
    shower's start segment to the reconstructed neutrino vertex.
    """
    mv = dump.get("main_vertex")
    if not mv:
        return []
    nu = (mv["x"], mv["y"], mv["z"])
    verts = {v["id"]: v for v in dump["vertices"]}
    segs = {s["id"]: s for s in dump["segments"]}
    out = []
    for sh in dump.get("showers") or []:
        if sh.get("start_connection_type") not in (2, 3):
            continue
        seg = segs.get(sh["id"])        # shower id == its start segment's id
        if not seg:
            continue
        pts = seg.get("points") or []
        if not pts:
            continue
        anchor = verts.get(sh.get("start_vertex_id"))
        start = (sh["start"]["x"], sh["start"]["y"], sh["start"]["z"])
        out.append(dict(
            event=dump["meta"]["eventNo"],
            shower=sh["id"],
            ke=sh["kine_best"],
            conn=sh["start_connection_type"],
            length=sh["total_length"],
            in_main=(seg["cluster_id"] == mv["cluster_id"]),
            flight=math.dist(vertex_xyz(anchor), start) if anchor else None,
            charge_to_nu=min(math.dist(nu, (q["x"], q["y"], q["z"])) for q in pts),
        ))
    return out


def split_main_clusters():
    """How often is the main cluster's own PR graph disconnected?

    A cluster is one contiguous lump of charge, so its segment graph ought to be
    connected.  When it is not, everything on the component without the main
    vertex is unreachable by the shower BFS and can only be rendered through a
    synthetic neutral parent.
    """
    n = split = 0
    gaps = []
    for _, d in dumps():
        mv = d.get("main_vertex")
        if not mv:
            continue
        n += 1
        parent = {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for s in d["segments"]:
            ra, rb = find(s["start_vertex_id"]), find(s["end_vertex_id"])
            if ra != rb:
                parent[ra] = rb
        main_id = next((v["id"] for v in d["vertices"] if v.get("is_main")), None)
        if main_id is None:
            continue
        root = find(main_id)
        nu = (mv["x"], mv["y"], mv["z"])
        stranded = [v for v in d["vertices"]
                    if v["cluster_id"] == mv["cluster_id"] and find(v["id"]) != root]
        if stranded:
            split += 1
            gaps.append(min(math.dist(nu, vertex_xyz(v)) for v in stranded))
    print("\n=== main-cluster PR graph connectivity")
    print("events with a dump and a main vertex: %d" % n)
    print("main cluster split into >=2 components: %d (%.1f%%)"
          % (split, 100.0 * split / max(1, n)))
    gaps.sort()
    if gaps:
        print("  distance from the main vertex to the nearest stranded vertex, cm:")
        print("    median %.2f   <=1cm %d   <=3cm %d   <=5cm %d   of %d"
              % (gaps[len(gaps) // 2],
                 sum(1 for g in gaps if g <= 1), sum(1 for g in gaps if g <= 3),
                 sum(1 for g in gaps if g <= 5), len(gaps)))


def pseudo_parents():
    """Pseudo-parent nodes in the Bee trees, by rendered length and by context.

    Context matters: a pi0 renders as pi0 -> gamma -> e- BY DESIGN, and its
    gammas legitimately sit at the decay vertex with zero length.  Counting
    those as defects inflates the bug by a factor of two.
    """
    tab = collections.Counter()
    events = 0
    for _, tree in bee_trees():
        events += 1
        for node, parent in walk(tree):
            if node_name(node) not in ("gamma", "neutron"):
                continue
            if not node.get("children"):
                continue
            g = node_len(node)
            ctx = ("under-pi0" if parent is not None and node_name(parent) == "pi0"
                   else "root" if parent is None else "under-" + node_name(parent))
            b = ("0.00" if g < 0.005 else "<1" if g < 1
                 else "<3" if g < 3 else "<10" if g < 10 else ">=10")
            tab[(ctx, b)] += 1
    bins = ["0.00", "<1", "<3", "<10", ">=10"]
    print("\n=== pseudo-parent (gamma/neutron) nodes in %d Bee trees" % events)
    print("rendered length of the synthetic parent, cm")
    print("%-16s %6s %6s %6s %6s %6s" % ("context", *bins))
    for ctx in sorted({k[0] for k in tab}):
        print("%-16s %6d %6d %6d %6d %6d"
              % (ctx, *[tab[(ctx, b)] for b in bins]))
    zero_pi0 = tab[("under-pi0", "0.00")]
    zero_all = sum(tab[(c, "0.00")] for c in {k[0] for k in tab})
    print("  of %d zero-length parents, %d are pi0 daughters (correct by design)"
          % (zero_all, zero_pi0))


def discriminator():
    """The load-bearing measurement: rendered length is NOT the discriminator."""
    rows = []
    for _, d in dumps():
        rows += [r for r in shower_rows(d)
                 if r["flight"] is not None and r["ke"] >= KE_MIN]
    groups = [
        ("rendered length == 0.00", [r for r in rows if r["flight"] < 0.005]),
        ("rendered length 0-3 cm", [r for r in rows if 0.005 <= r["flight"] < 3.0]),
        ("rendered length >=3 cm (control)", [r for r in rows if r["flight"] >= 3.0]),
    ]
    print("\n=== does the rendered length select the right showers?  (KE >= %.0f MeV)"
          % KE_MIN)
    print("%-34s %5s %9s %9s %9s %14s"
          % ("population", "n", "min", "median", "max", "in main clus"))
    for name, g in groups:
        if not g:
            continue
        v = sorted(r["charge_to_nu"] for r in g)
        print("%-34s %5d %9.2f %9.2f %9.2f %10d /%3d"
              % (name, len(g), v[0], v[len(v) // 2], v[-1],
                 sum(1 for r in g if r["in_main"]), len(g)))
    print("  columns min/median/max are the shower's nearest CHARGE to the")
    print("  neutrino vertex, in cm.  The zero-length group is the REMOTE one.")


def target_population():
    """The owner's class: main-cluster showers whose charge touches the vertex."""
    hits = []
    events = 0
    for _, d in dumps():
        if not d.get("main_vertex"):
            continue
        events += 1
        hits += [r for r in shower_rows(d)
                 if r["in_main"] and r["charge_to_nu"] <= TOUCH]
    print("\n=== target population: conn-2/3 shower IN the main cluster whose")
    print("    charge comes within %.0f cm of the neutrino vertex" % TOUCH)
    print("%d showers in %d events, of %d events"
          % (len(hits), len({h["event"] for h in hits}), events))
    for h in sorted(hits, key=lambda h: -h["ke"]):
        print("  evt%-8d shower=%-7d %7.1f MeV  conn=%d  charge->nu %5.2f cm  len=%6.1f"
              % (h["event"], h["shower"], h["ke"], h["conn"],
                 h["charge_to_nu"], h["length"]))


def micro_parents():
    """Near-vertex over-segmentation, as it shows up in the particle tree.

    A sub-2 cm segment wedged between the true junction and the chosen main
    vertex does not merely add a node -- it INVERTS the hierarchy, because the
    long prong then hangs off the stub.  evt287517 is a 343 MeV muon parented by
    a 6 MeV proton.
    """
    hits = []
    events = 0
    for evid, tree in bee_trees():
        events += 1
        for node, _ in walk(tree):
            if node_name(node) in ("gamma", "neutron", "pi0"):
                continue
            ln = node_len(node)
            if ln >= MICRO:
                continue
            for child in node.get("children", []):
                if node_name(child) in ("gamma", "neutron", "pi0"):
                    continue
                cl = node_len(child)
                if cl > MICRO_CHILD and cl > MICRO_RATIO * max(ln, 0.2):
                    hits.append((evid, node["text"], ln, child["text"], cl))
    print("\n=== micro-parent inversions: a real-particle node < %.0f cm parenting a"
          % MICRO)
    print("    child > %.0f cm and > %.0fx longer than itself" % (MICRO_CHILD, MICRO_RATIO))
    print("%d inversions in %d of %d events"
          % (len(hits), len({h[0] for h in hits}), events))
    for h in sorted(hits, key=lambda h: h[2]):
        print("  evt%-8s %-22s %5.2f cm  ->  %-22s %7.1f cm" % h)


def keep_isolated_drops():
    """Residual segments discarded by find_other_segments' AND-floor.

    other_seg_keep_isolated_ok() requires points >= 25 AND length >= 3 cm, so a
    long sparse track dies on point count alone -- evt284794 loses a 71 cm
    segment whose near end is 7.8 cm from the neutrino vertex.
    """
    seen = set()
    for arm in ARMS:
        logs = sorted(glob.glob(os.path.join(arm, "pr_evt*", "wct_pr_evt*.log")))
        for i in range(0, len(logs), 200):
            # grep -a: WCT logs contain invalid UTF-8 and count as binary
            # without it, which makes a plain grep print nothing at all.
            proc = subprocess.run(["grep", "-ah", "pr54 isolated-residual drop"]
                                  + logs[i:i + 200],
                                  capture_output=True, text=True, errors="replace")
            for line in proc.stdout.splitlines():
                m = re.search(r"n_points=(\d+) length=([\d.]+) cm", line)
                if m:
                    seen.add((int(m.group(1)), float(m.group(2))))
    print("\n=== isolated residuals DROPPED by find_other_segments")
    print("%d distinct drop records" % len(seen))
    for cut in (15.0, 30.0):
        long_ = [(p, l) for p, l in seen if l >= cut]
        blocked = [(p, l) for p, l in long_ if p < 25]
        print("  length >= %4.0f cm: %3d   of which blocked ONLY by the 25-point "
              "floor: %d" % (cut, len(long_), len(blocked)))


def main():
    print(__doc__.split("Run:")[0].strip())
    split_main_clusters()
    pseudo_parents()
    discriminator()
    target_population()
    micro_parents()
    keep_isolated_drops()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

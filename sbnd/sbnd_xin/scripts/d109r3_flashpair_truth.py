#!/usr/bin/env python3
"""sbnd_xin/docs/109 rev 3 sec 8.7.2/8.7.3: should nu_dedup_flash_group be flipped?

Short answer measured here: no.  The rule keeps the LONGEST candidate of a flash
group, and on the one event of 22 where both rows are real reconstructions that
is the wrong half of a neutrino whose muon crossed the cathode.

The knob collapses neutrino candidates that share a flash_group, keeping the
LONGEST selected activity.  The 25-event local census (d109r3_dedup_census.py)
cannot answer whether that is the right survivor, because in all of its events
the other row is a vertex-less placeholder and the choice is free.

This script answers it on a sample big enough to contain the hard case: it
scans a tracking-pr/ directory for same-flash both-TPC candidate pairs, joins
each to the Bee truth in the matching bee/bee_<tag>.zip (data/0/0-mc.json), and
reports, for the events where BOTH rows are real reconstructions, whether
"keep the longest" keeps or drops the row nearest a true neutrino vertex.

Usage:
  scripts/d109r3_flashpair_truth.py <run_dir>      # e.g. sbnd_mc_data
     <run_dir>/tracking-pr/tracking-pr_<tag>.root  and  <run_dir>/bee/bee_<tag>.zip
"""
import argparse, collections, glob, json, math, os, zipfile
from concurrent.futures import ProcessPoolExecutor

import uproot

DT_US = 0.05   # TaggerCheckNeutrino flash_pair_dt_us default


def pair_rows(path):
    """The two candidate rows of a same-flash both-TPC pair event, or None."""
    try:
        f = uproot.open(path)
        if "T_tagger" not in {k.split(";")[0] for k in f.keys()}:
            return None
        if f["T_tagger"].num_entries != 2:
            return None
        tg = f["T_tagger"].arrays(["matched_flash_gid", "nu_x", "nu_y", "nu_z", "neutrino_type",
                                   "act_cluster_id", "act_is_selected", "act_length_cm",
                                   "numu_score"], library="np")
        kn = f["T_kine"].arrays(["kine_reco_Enu"], library="np")
        cl = f["T_cluster"].arrays(["flash_id", "flash_time_us"], library="np")
        gidt = {int(g): float(t) for g, t in zip(cl["flash_id"], cl["flash_time_us"]) if g >= 0}
        gid = [int(tg["matched_flash_gid"][i]) for i in range(2)]
        t = [gidt.get(g, float("nan")) for g in gid]
        # Same rule as PR::group_flashes: different TPC and |dt| < dt_us.
        if gid[0] // 1000000 == gid[1] // 1000000:
            return None
        if not abs(t[0] - t[1]) < DT_US:
            return None
        rows = []
        for i in range(2):
            sel = [(int(c), float(L)) for c, s, L in zip(tg["act_cluster_id"][i],
                                                         tg["act_is_selected"][i],
                                                         tg["act_length_cm"][i]) if s == 1][0]
            v = (float(tg["nu_x"][i]), float(tg["nu_y"][i]), float(tg["nu_z"][i]))
            enu = float(kn["kine_reco_Enu"][i])
            nt = int(tg["neutrino_type"][i])
            rows.append(dict(gid=gid[i], tpc=gid[i] // 1000000, cluster=sel[0], L=sel[1],
                             enu=enu, numu=float(tg["numu_score"][i]), v=v,
                             # pre-doc-109 files have no has_vertex; this is the
                             # same condition (final_main_vertex == nullptr).
                             placeholder=(nt == 0 and enu == 0.0 and v == (0., 0., 0.))))
        tag = os.path.basename(path).replace("tracking-pr_", "").replace(".root", "")
        return tag, rows
    except Exception:
        return None


def truth_vertices(run_dir, tag):
    """(label, vertex) per true interaction from the Bee mc.json, or []."""
    z = os.path.join(run_dir, "bee", "bee_%s.zip" % tag)
    if not os.path.isfile(z):
        return []
    try:
        zz = zipfile.ZipFile(z)
        name = [n for n in zz.namelist() if n.endswith("-mc.json")][0]
        entries = json.loads(zz.read(name))
    except Exception:
        return []
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("id", 0) >= 19999999:       # the reco tree, not truth
            continue
        v = e.get("data", {}).get("start")
        if v:
            # Primary daughters whose start and end straddle x = 0 (the SBND
            # cathode): a charged one means this interaction's charge lands in
            # BOTH drift volumes, which is the mechanism behind sec 8.7.3.
            cross = []
            for c in e.get("children", []):
                cs = c.get("data", {}).get("start")
                ce = c.get("data", {}).get("end")
                if cs and ce and cs[0] * ce[0] < 0:
                    cross.append(c.get("text", "").split()[0])
            out.append((e.get("text", ""), tuple(v), cross))
    return out


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run_dir, "tracking-pr", "tracking-pr_*.root")))
    pairs = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(pair_rows, files, chunksize=64):
            if r:
                pairs.append(r)
    print("scanned %d files; same-flash both-TPC pair events: %d" % (len(files), len(pairs)))

    c = collections.Counter()
    for tag, rows in pairs:
        both = not rows[0]["placeholder"] and not rows[1]["placeholder"]
        c["one row is a vertex-less placeholder" if not both else "both rows reconstructed"] += 1
        if not both:
            continue
        tr = truth_vertices(a.run_dir, tag)
        if not tr:
            c["both reconstructed but NO truth available"] += 1
            continue
        d = [min(dist(r["v"], tv) for _, tv, _ in tr) for r in rows]
        longest = 0 if rows[0]["L"] >= rows[1]["L"] else 1
        nearest = 0 if d[0] < d[1] else 1
        ok = longest == nearest
        c["dedup KEEPS the truth-nearest row" if ok else "dedup DROPS the truth-nearest row"] += 1
        print("\n  %s: %s" % (tag, "KEEPS truth" if ok else "!! DROPS truth"))
        for i, r in enumerate(rows):
            print("     row%d gid %8d tpc%d L=%7.1fcm Enu=%7.1f numu=%+.2f d_truth=%7.1fcm%s%s"
                  % (i, r["gid"], r["tpc"], r["L"], r["enu"], r["numu"], d[i],
                     "  <-longest" if i == longest else "", "  <-truth" if i == nearest else ""))
        for txt, tv, cross in tr[:3]:
            print("        truth: %s  |x_nu|=%.1f cm  cathode-crossing primaries: %s"
                  % (txt[:56], abs(tv[0]), ", ".join(cross) or "-"))
    print("\n== totals")
    for k in sorted(c):
        print("  %-44s %d" % (k, c[k]))


if __name__ == "__main__":
    main()

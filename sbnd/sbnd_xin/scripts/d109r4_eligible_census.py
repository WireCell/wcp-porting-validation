#!/usr/bin/env python3
"""sbnd_xin/docs/109 rev 4: which events CAN nu_bundle_flash_group touch?

The knob merges two in-window neutrino bundles that share a flash_group (one
physical flash seen by both drift volumes) and meet at the cathode.  So the only
events it can change are those with a candidate whose flash group holds a
SECOND gid that has clusters of its own.  This script sizes that population on
the local 3067-event sample from the files alone (T_cluster flash_id /
flash_time_us for the grouping, is_main / is_associated for the roles), before
any code runs -- CLAUDE.md "size a knob on its eligible population".

It writes docs/109_logs/r4/events_eligible_<sample>.txt (one event id per line,
the arm manifests) and prints the census.

Usage: scripts/d109r4_eligible_census.py [--label d102mpr] [--jobs 24]
"""
import argparse
import collections
import glob
import os
from multiprocessing import Pool

import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DT_US = 0.05   # TaggerCheckNeutrino flash_pair_dt_us default (same rule as PR::group_flashes)
SAMPLES = ["nuecc48", "ncpi0", "mcp1k", "mcp2k"]


def groups_of(gidt):
    """gid -> group id (smallest gid), union-find over different-TPC pairs within DT_US."""
    gids = sorted(gidt)
    par = {g: g for g in gids}

    def fd(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            if a // 1000000 == b // 1000000:
                continue
            if abs(gidt[a] - gidt[b]) < DT_US:
                ra, rb = fd(a), fd(b)
                if ra != rb:
                    par[max(ra, rb)] = min(ra, rb)
    return {g: fd(g) for g in gids}


def one(path):
    try:
        f = uproot.open(path)
        names = {k.split(";")[0] for k in f.keys()}
        if "T_cluster" not in names or "T_tagger" not in names or f["T_tagger"].num_entries == 0:
            return []
        cl = f["T_cluster"].arrays(["is_main", "is_associated", "flash_id", "length_cm"], library="np")
        tg = f["T_tagger"].arrays(["matched_flash_gid", "nu_index"], library="np")
        gidt = {}
        for g, t in zip(cl["flash_id"], f["T_cluster"]["flash_time_us"].array(library="np")):
            if g >= 0:
                gidt[int(g)] = float(t)
        grp = groups_of(gidt)
        members = collections.defaultdict(list)
        for g in gidt:
            members[grp[g]].append(g)
        per = collections.defaultdict(lambda: dict(main=0, assoc=0, main_len=0.0))
        for i in range(len(cl["flash_id"])):
            g = int(cl["flash_id"][i])
            if g < 0:
                continue
            if cl["is_main"][i]:
                per[g]["main"] += 1
                per[g]["main_len"] = max(per[g]["main_len"], float(cl["length_cm"][i]))
            if cl["is_associated"][i]:
                per[g]["assoc"] += 1
        cand_gids = [int(x) for x in tg["matched_flash_gid"]]
        out = []
        for r in range(len(cand_gids)):
            g = cand_gids[r]
            others = [x for x in members.get(grp.get(g, g), [g]) if x != g and (per[x]["main"] or per[x]["assoc"])]
            if not others:
                continue
            o = others[0]
            out.append(dict(evt=os.path.basename(os.path.dirname(path)), nu=int(tg["nu_index"][r]), gid=g,
                            other=o, other_is_cand=o in cand_gids, other_main=per[o]["main"],
                            other_main_len=per[o]["main_len"], other_assoc=per[o]["assoc"]))
        return out
    except Exception as e:  # noqa: BLE001
        return [dict(err="%s: %s" % (path, e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="d102mpr")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    outdir = os.path.join(SX, "docs", "109_logs", "r4")
    os.makedirs(outdir, exist_ok=True)
    total = collections.Counter()
    nfiles = 0
    for s in SAMPLES:
        paths = sorted(glob.glob("%s/work-%s-%s/pr_evt*/tracking-pr.root" % (SX, s, a.label)))
        nfiles += len(paths)
        ev = collections.defaultdict(list)
        with Pool(a.jobs) as p:
            for res in p.imap_unordered(one, paths, chunksize=8):
                for r in res:
                    if "err" in r:
                        print("ERR", r["err"])
                        continue
                    ev[r["evt"]].append(r)
        ids = sorted(int(e.replace("pr_evt", "")) for e in ev)
        with open(os.path.join(outdir, "events_eligible_%s.txt" % s), "w") as fh:
            for i in ids:
                fh.write("%d\n" % i)
        c = collections.Counter()
        for rows in ev.values():
            kinds = {("other gid is a candidate" if r["other_is_cand"] else "other gid is NOT a candidate") for r in rows}
            for k in kinds:
                c[k] += 1
        print("== %s: %d files, %d eligible events  %s" % (s, len(paths), len(ev), dict(c)))
        for e in sorted(ev):
            for r in ev[e]:
                print("   %s nu%d gid %d  other gid %d %s  mains %d (longest %.1f cm) assoc %d"
                      % (e, r["nu"], r["gid"], r["other"], "CAND" if r["other_is_cand"] else "----",
                         r["other_main"], r["other_main_len"], r["other_assoc"]))
        total["eligible events"] += len(ev)
        for k, v in c.items():
            total[k] += v
    print("== total: %d files; %s" % (nfiles, dict(total)))


if __name__ == "__main__":
    main()

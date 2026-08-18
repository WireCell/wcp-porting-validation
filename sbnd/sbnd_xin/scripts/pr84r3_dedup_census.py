#!/usr/bin/env python3
"""pr/84 round 3 -- duplicate-shower census over one or two PR arms.

Per event it reports the three observables of the round-3 defect:

  dup_ids    repeated jsTree node ids in mabc-pr.zip::data/0/0-mc.json
             (invalid input for the Bee tree: jsTree keys its model by id)
  twins      showers of calib-pr-evt<ID>.json sharing a start segment id,
             conn_type 4 excluded (never rendered, never in the kine tree)
  kine_dup   repeated (particle_type, energy) entries in kine_energy_particle,
             with kine_reco_Enu and the de-duplicated sum

With two arms it prints a before/after line per event and the totals, which is
the enumerated mover list the flip decision needs.

Repro:
  scripts/pr84r3_dedup_census.py work-pr84r3-off-mc [work-pr84r3-dedup-mc]
"""
import glob
import json
import os
import sys
import zipfile
from collections import Counter


def load_tree(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("0-mc.json")]
        if not names:
            return None
        return json.loads(z.read(names[0]))


def stats(evt_dir, evt):
    """(n_dup_ids, twins, kine) for one pr_evt dir; None when unreadable."""
    out = {"dup_ids": 0, "dup_id_list": [], "twins": [], "enu": None,
           "kine_dup": [], "kine_sum": None, "nodes": 0}
    z = os.path.join(evt_dir, "mabc-pr.zip")
    if os.path.exists(z):
        try:
            tree = load_tree(z)
        except Exception:
            tree = None
        if tree:
            ids = []

            def walk(n):
                ids.append(n.get("id"))
                for c in n.get("children", []):
                    walk(c)

            for n in tree:
                walk(n)
            cnt = Counter(ids)
            out["nodes"] = len(ids)
            out["dup_ids"] = sum(v - 1 for v in cnt.values() if v > 1)
            out["dup_id_list"] = sorted(k for k, v in cnt.items() if v > 1)
    c = os.path.join(evt_dir, f"calib-pr-evt{evt}.json")
    if os.path.exists(c):
        d = json.load(open(c))
        seg_cnt = Counter(s["id"] for s in d.get("showers", [])
                          if s.get("start_connection_type") != 4)
        out["twins"] = sorted(k for k, v in seg_cnt.items() if v > 1)
        k = d.get("kine") or {}
        parts = k.get("kine_energy_particle") or []
        types = k.get("kine_particle_type") or []
        out["enu"] = k.get("kine_reco_Enu")
        pcnt = Counter(zip(types, [round(x, 3) for x in parts]))
        out["kine_dup"] = sorted((t, e, v) for (t, e), v in pcnt.items() if v > 1)
        out["kine_sum"] = round(sum(parts), 2)
    return out


def fmt(s):
    return (f"nodes={s['nodes']} dup_ids={s['dup_ids']}{s['dup_id_list'] or ''} "
            f"twins={s['twins']} enu={s['enu']} kine_dup={s['kine_dup']}")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    arm_a = sys.argv[1]
    arm_b = sys.argv[2] if len(sys.argv) == 3 else None
    tot = Counter()
    for d in sorted(glob.glob(os.path.join(arm_a, "pr_evt*"))):
        evt = os.path.basename(d).replace("pr_evt", "")
        sa = stats(d, evt)
        tot["events"] += 1
        if sa["dup_ids"]:
            tot["a_dup_id_evts"] += 1
        if sa["twins"]:
            tot["a_twin_evts"] += 1
        if sa["kine_dup"]:
            tot["a_kine_dup_evts"] += 1
        if arm_b is None:
            if sa["dup_ids"] or sa["twins"] or sa["kine_dup"]:
                print(f"evt={evt} {fmt(sa)}")
            continue
        db = os.path.join(arm_b, f"pr_evt{evt}")
        if not os.path.isdir(db):
            print(f"evt={evt} MISSING in {arm_b}")
            tot["missing"] += 1
            continue
        sb = stats(db, evt)
        if sb["dup_ids"]:
            tot["b_dup_id_evts"] += 1
        if sb["twins"]:
            tot["b_twin_evts"] += 1
        if sb["kine_dup"]:
            tot["b_kine_dup_evts"] += 1
        changed = (sa["dup_ids"] != sb["dup_ids"] or sa["twins"] != sb["twins"]
                   or sa["enu"] != sb["enu"] or sa["nodes"] != sb["nodes"]
                   or sa["kine_dup"] != sb["kine_dup"])
        if changed:
            tot["changed"] += 1
            print(f"== evt={evt}")
            print(f"   A: {fmt(sa)}")
            print(f"   B: {fmt(sb)}")
    print("# census: " + " ".join(f"{k}={v}" for k, v in sorted(tot.items())))


if __name__ == "__main__":
    main()

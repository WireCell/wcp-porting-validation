#!/usr/bin/env python3
"""doc pr/40 round 4 G2/G4: the two owner-reported cases, off vs on, plus a
population census of every segment either fix moves.

F7 shower_proton_daughter_pion_dissolve -- evt 256587 seg 11079: a shower
relabelled pion (211) by F5 should stop being wrapped as a Shower, so its
proton daughter (seg 11080) gets its own particle-flow node in mc.json and
the pi+ node's own end point (not the shower's) is displayed.

F8 muon_multi_proton_pion -- evt 489330 seg 4019: a muon segment whose far
end is a two-proton hadronic vertex should relabel to pion (211); the
sibling muon segment 4043 across the degree-2 kink must stay mu- (owner
decision -- no propagation).

Usage: pr40r4_check.py <off_arm_dir> <on_arm_dir>
"""
import json
import os
import sys
import glob
import zipfile

MIP = 43000.0 / 1.0  # e/cm, m_mip_dqdx_median internal-unit scale (see doc)

F7_CASE = (256587, 11079, 11080)   # evt, pion-candidate seg, proton daughter seg
F8_CASE = (489330, 4019, 4043)     # evt, muon-to-relabel seg, sibling (must stay mu-)


def load_calib(arm, evt):
    f = os.path.join(arm, f"pr_evt{evt}", f"calib-pr-evt{evt}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def load_mc(arm, evt):
    f = os.path.join(arm, f"pr_evt{evt}", "mabc-pr.zip")
    if not os.path.exists(f):
        return None
    z = zipfile.ZipFile(f)
    names = [n for n in z.namelist() if n.endswith("-mc.json")]
    if not names:
        return None
    return json.loads(z.read(names[0]))


def find_node(tree, node_id):
    for n in tree:
        if n.get("id") == node_id:
            return n
        found = find_node(n.get("children", []), node_id)
        if found is not None:
            return found
    return None


def seg_by_id(calib, sid):
    for s in calib["segments"]:
        if s["id"] == sid:
            return s
    return None


def check_f7(off, on):
    evt, pion_seg, proton_seg = F7_CASE
    print(f"=== F7 (evt {evt}) ===")
    off_c, on_c = load_calib(off, evt), load_calib(on, evt)
    for label, c in (("off", off_c), ("on", on_c)):
        if c is None:
            print(f"  {label}: MISSING calib json")
            continue
        s = seg_by_id(c, pion_seg)
        print(f"  {label}: seg {pion_seg} pdg={s['particle_id']} flag_shower={s['flag_shower']} shower_id={s.get('shower_id')}")
        sp = seg_by_id(c, proton_seg)
        print(f"  {label}: seg {proton_seg} (proton) pdg={sp['particle_id']} flag_shower={sp['flag_shower']} shower_id={sp.get('shower_id')}")

    on_mc = load_mc(on, evt)
    if on_mc is None:
        print("  on: MISSING mc.json")
        return
    node = find_node(on_mc, pion_seg)
    if node is None:
        print(f"  on mc.json: node {pion_seg} NOT FOUND")
        return
    print(f"  on mc.json: node {pion_seg} text='{node['text']}' end={node['data']['end']}")
    child_ids = [c["id"] for c in node.get("children", [])]
    print(f"  on mc.json: children of {pion_seg}: {child_ids}")
    has_proton_child = proton_seg in child_ids
    print(f"  VERDICT: proton {proton_seg} is a direct PF child of pion {pion_seg}: {has_proton_child}")
    # end point should be segment 11079's own end (-90.2,-17.2,264.3), not the
    # shower's absorbed fragment end (-95.0,-10.1,266.6).
    end = node["data"]["end"]
    near_own_end = abs(end[0] - (-90.2)) < 1.0 and abs(end[1] - (-17.2)) < 1.0
    print(f"  VERDICT: node end is segment's OWN end (not shower fragment's): {near_own_end}")


def check_f8(off, on):
    evt, muon_seg, sibling_seg = F8_CASE
    print(f"=== F8 (evt {evt}) ===")
    off_c, on_c = load_calib(off, evt), load_calib(on, evt)
    for label, c in (("off", off_c), ("on", on_c)):
        if c is None:
            print(f"  {label}: MISSING calib json")
            continue
        s = seg_by_id(c, muon_seg)
        print(f"  {label}: seg {muon_seg} pdg={s['particle_id']}")
        sib = seg_by_id(c, sibling_seg)
        print(f"  {label}: seg {sibling_seg} (sibling, should stay mu-) pdg={sib['particle_id']}")

    on_mc = load_mc(on, evt)
    if on_mc is None:
        print("  on: MISSING mc.json")
        return
    node = find_node(on_mc, muon_seg)
    if node is None:
        print(f"  on mc.json: node {muon_seg} NOT FOUND")
        return
    print(f"  on mc.json: node {muon_seg} text='{node['text']}'")
    sib_node = find_node(on_mc, sibling_seg)
    if sib_node is not None:
        print(f"  on mc.json: node {sibling_seg} text='{sib_node['text']}'")


def census(off, on):
    print("=== census: pdg 211 && flag_shower segments (F7 population) ===")
    n = 0
    for f in sorted(glob.glob(os.path.join(off, "pr_evt*", "calib-pr-evt*.json"))):
        evt = int(os.path.basename(f).split("evt")[-1].split(".")[0])
        c = json.load(open(f))
        for s in c["segments"]:
            if s.get("particle_id") == 211 and s.get("flag_shower"):
                n += 1
                print(f"  off-arm ev {evt} seg {s['id']} len {s['length']:.1f} main_clus {s['is_main_cluster']}")
    print(f"  total (off-arm, should be 0 pre-fix baseline; this counts whatever slipped through): {n}")

    print("=== census: on-arm segments where pdg/flag_shower moved vs off ===")
    nmoved = 0
    for f in sorted(glob.glob(os.path.join(off, "pr_evt*", "calib-pr-evt*.json"))):
        evt = int(os.path.basename(f).split("evt")[-1].split(".")[0])
        off_c = json.load(open(f))
        on_c = load_calib(on, evt)
        if on_c is None:
            continue
        on_segs = {s["id"]: s for s in on_c["segments"]}
        for s in off_c["segments"]:
            sid = s["id"]
            sn = on_segs.get(sid)
            if sn is None:
                continue
            if sn["particle_id"] != s["particle_id"] or sn["flag_shower"] != s["flag_shower"]:
                nmoved += 1
                print(f"  ev {evt} seg {sid}: off pdg={s['particle_id']} fs={s['flag_shower']} "
                      f"-> on pdg={sn['particle_id']} fs={sn['flag_shower']}")
    print(f"  total moved: {nmoved}")


def main():
    off, on = sys.argv[1], sys.argv[2]
    check_f7(off, on)
    print()
    check_f8(off, on)
    print()
    census(off, on)


if __name__ == "__main__":
    main()

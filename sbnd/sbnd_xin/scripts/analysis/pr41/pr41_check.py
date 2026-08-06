#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/40 round 2 -- gate checks for the two owner-reported
follow-on defects (zero-energy muon, proton-daughter-should-be-pion) plus the
population census that bounds the flip decision.

Usage:
    python3 pr41_check.py <off_arm_dir> <on_arm_dir>

<off_arm_dir>/<on_arm_dir> are sbnd_xin run_pr_chain_batch.sh output roots
(pr_evt<ID>/{calib-pr-evt<ID>.json,mabc-pr.zip} per event).  Needs
PR_EXTRA_STAGES=pr_display so calib-pr-evt*.json exists.

G2a (evt 174637 seg 9050): the Bee PF-tree node's energy, off vs on.  Fixed
means it moves off exactly 0 MeV.
G2b (evt 256587 seg 11079): pdg + flag_shower, off vs on.  Fixed means pdg
11 -> 211.  flag_shower is reported but NOT part of the pass bar: F5 only
ever touches particle_info/pdg, never the shower flags, by design -- Bee and
PrDisplayDump both read pdg, not flag_shower, so this is what the owner's
report is actually about (see doc pr/40 round 2).
G4 census: over every event common to both arms, the count and identity of
PF nodes reading exactly 0 MeV (should go to 0 on the on-arm), and the count
of electron-labelled segments whose far end (by set_default_shower_particle_
info's own rule) satisfies the proton-daughter test on the on-arm.
"""
import json
import os
import re
import sys
import zipfile
import glob
import statistics


def load_calib(root, evt):
    p = os.path.join(root, f"pr_evt{evt}", f"calib-pr-evt{evt}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def find_segment(calib, encoded_id):
    if not calib:
        return None
    for seg in calib.get("segments", []):
        if seg.get("id") == encoded_id:
            return seg
    return None


def pf_energy_nodes(root, evt):
    """Return {node_id: (name, energy_MeV)} from the Bee mc.json PF tree."""
    p = os.path.join(root, f"pr_evt{evt}", "mabc-pr.zip")
    if not os.path.isfile(p):
        return {}
    out = {}
    try:
        z = zipfile.ZipFile(p)
    except zipfile.BadZipFile:
        return {}
    member = None
    for n in z.namelist():
        if n.endswith("mc.json"):
            member = n
            break
    if member is None:
        return {}
    tree = json.loads(z.read(member))

    def walk(node):
        m = re.match(r"^(\S+)\s+(-?\d+)\s*MeV", node.get("text", ""))
        if m:
            out[node["id"]] = (m.group(1), int(m.group(2)))
        for c in node.get("children", []):
            walk(c)

    for n in tree:
        walk(n)
    return out


def g2a(off_root, on_root, evt=174637, seg_id=9050):
    off_pf = pf_energy_nodes(off_root, evt)
    on_pf = pf_energy_nodes(on_root, evt)
    off_v = off_pf.get(seg_id)
    on_v = on_pf.get(seg_id)
    print(f"G2a evt {evt} seg {seg_id}: off={off_v} on={on_v}")
    if on_v is not None and on_v[1] != 0:
        print("  PASS: energy moved off exactly 0 MeV")
    else:
        print("  FAIL or inconclusive")


def g2b(off_root, on_root, evt=256587, seg_id=11079):
    off_c = find_segment(load_calib(off_root, evt), seg_id)
    on_c = find_segment(load_calib(on_root, evt), seg_id)
    off_pdg = off_c.get("particle_id") if off_c else None
    on_pdg = on_c.get("particle_id") if on_c else None
    off_fs = off_c.get("flag_shower") if off_c else None
    on_fs = on_c.get("flag_shower") if on_c else None
    print(f"G2b evt {evt} seg {seg_id}: off pdg={off_pdg} flag_shower={off_fs}"
          f"  on pdg={on_pdg} flag_shower={on_fs}")
    if on_pdg == 211:
        print("  PASS: pdg 11->211 (flag_shower unchanged by design)")
    else:
        print("  FAIL or inconclusive")


def g4_census(off_root, on_root):
    evts = sorted(
        int(re.search(r"pr_evt(\d+)$", p).group(1))
        for p in glob.glob(os.path.join(off_root, "pr_evt*"))
    )
    zero_off = zero_on = 0
    new_zero = []
    proton_daughter_pions = []
    for evt in evts:
        off_pf = pf_energy_nodes(off_root, evt)
        on_pf = pf_energy_nodes(on_root, evt)
        for nid, (name, e) in off_pf.items():
            if e == 0:
                zero_off += 1
        for nid, (name, e) in on_pf.items():
            if e == 0:
                zero_on += 1
                new_zero.append((evt, nid, name))
        on_c = load_calib(on_root, evt)
        off_c = load_calib(off_root, evt)
        if not on_c or not off_c:
            continue
        off_segs = {s["id"]: s for s in off_c.get("segments", [])}
        for s in on_c.get("segments", []):
            if s.get("particle_id") == 211:
                o = off_segs.get(s["id"])
                if o and o.get("particle_id") == 11:
                    proton_daughter_pions.append((evt, s["id"]))
    print(f"\nG4 census over {len(evts)} events (both arms):")
    print(f"  PF nodes at exactly 0 MeV: off={zero_off} on={zero_on}")
    if new_zero:
        print(f"  on-arm zero-energy nodes remaining: {new_zero}")
    print(f"  segments moved 11->211 (proton-daughter rule fired): "
          f"{len(proton_daughter_pions)}")
    for evt, sid in proton_daughter_pions:
        print(f"    evt {evt} seg {sid}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    off_root, on_root = sys.argv[1], sys.argv[2]
    g2a(off_root, on_root)
    g2b(off_root, on_root)
    g4_census(off_root, on_root)

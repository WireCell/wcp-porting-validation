#!/usr/bin/env python3
"""The pr/132 pi0 pairing pass over the pr/126 rescan candidates.

pr/126 sec 4h measured that of 238 scanned events only 50 carry a stored pi0
pairing, while 109 of the unpaired ones have >=2 EM showers above the code's
own 15 MeV / 3 cm cuts -- unasked data, not absent data.  This tool asks: for
one event it prints an adjudication PACKET (every EM shower, every candidate
pair under both mass conventions, the without-vertex back-projection verdict),
and a chosen pairing is then WRITTEN as a label under a NEW tag
(`em_labels/pi0scan-0829-agent/`), never into an existing scan tag (M13).

The label carries the same `pio` block shape the emscan labels use, so
`pr126_pi0_select.py` / `pr132_pi0_census.py --overlay-tag` can consume it.
Energies are the CURRENT arm's kine_charge (fudge 0.80 scale, the labels'
`now` hypothesis); geometry is the dump's own (shower_init_dir for the axis
convention, the chosen decay vertex for the vertex convention).

    ./pr132_pi0_pair.py --packet 415278
    ./pr132_pi0_pair.py --write 415278 --g1 23037 --g2 97153 \
        --vertex-how main_vertex --confidence medium --note "..."
    ./pr132_pi0_pair.py --nopair 415278 --note "no defensible pairing: ..."
"""
import argparse, csv, json, math, os, sys, datetime

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
import em_geom as G

TAG = "pi0scan-0829-agent"
MANIFESTS = [("98", "em117-132denom98-manifest.tsv"),
             ("141", "em114c-132denom141-manifest.tsv")]
RESCAN = os.path.join(SX, "docs", "pr", "pr126-pi0-rescan.tsv")


def find_event(ev):
    for setname, mname in MANIFESTS:
        p = os.path.join(SX, "em_display", mname)
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if int(r["event"]) == ev:
                    return setname, r
    raise SystemExit("event %d not in either 132denom manifest" % ev)


def rescan_row(ev):
    if not os.path.exists(RESCAN):
        return None
    with open(RESCAN) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if int(r["event"]) == ev:
                return r
    return None


def load(ev):
    setname, mrow = find_event(ev)
    with open(os.path.join(SX, mrow["dump"])) as fh:
        dump = json.load(fh)
    return setname, mrow, dump


def em_candidates(dump, min_mev=10.0, min_len=3.0):
    out = []
    for s in (dump.get("showers") or ()):
        e = s.get("kine_charge") or 0
        if e < min_mev or (s.get("total_length") or 0) < min_len:
            continue
        out.append(s)
    return sorted(out, key=lambda s: -(s.get("kine_charge") or 0))


def conventions(s1, s2, dump, vertex):
    """(mass_v, theta_v, mass_a, theta_a) at the given decay vertex."""
    segs = dump.get("segments") or ()
    vby = {v["id"]: v for v in (dump.get("vertices") or ())}
    e1, e2 = s1.get("kine_charge"), s2.get("kine_charge")
    p1, p2 = G.pt(s1.get("start")), G.pt(s2.get("start"))
    th_v = G.angle_deg(G.vsub(p1, vertex), G.vsub(p2, vertex))
    m_v = G.pi0_mass(e1, e2, th_v)
    d1, _ = G.shower_init_dir(s1, segs, vby)
    d2, _ = G.shower_init_dir(s2, segs, vby)
    th_a = G.angle_deg(d1, d2) if (G.vmag(d1) and G.vmag(d2)) else None
    m_a = G.pi0_mass(e1, e2, th_a) if th_a is not None else None
    return m_v, th_v, m_a, th_a, d1, d2


def packet(ev, min_mev):
    setname, mrow, dump = load(ev)
    mv = G.pt(dump.get("main_vertex"))
    segs = dump.get("segments") or ()
    cands = em_candidates(dump, min_mev=0.0, min_len=0.0)
    print("event %d  set %s  sample %s  main_vertex (%.1f,%.1f,%.1f)"
          % (ev, setname, mrow["sample"], *mv))
    rr = rescan_row(ev)
    if rr:
        print("rescan row: origin=%s n_em=%s e_max=%s e_2nd=%s"
              % (rr["origin"], rr["n_em"], rr["e_max"], rr["e_2nd"]))
    print("\n-- showers (all, sorted by kine_charge; * = EM >=%.0f MeV >=3cm) --" % min_mev)
    for s in cands:
        e = s.get("kine_charge") or 0
        em = (abs(int(s.get("particle_id") or 0)) == 11 and e >= min_mev
              and (s.get("total_length") or 0) >= 3.0)
        st = G.pt(s.get("start"))
        print("  %s id=%-7d pdg=%-5s E=%-7.1f len=%-6.1f conn=%s pio_id=%-3s start=(%.1f,%.1f,%.1f) dvtx=%.1f"
              % ("*" if em else " ", s["id"], s.get("particle_id"), e,
                 s.get("total_length") or 0, s.get("start_connection_type"),
                 s.get("pio_id"), st[0], st[1], st[2],
                 G.vmag(G.vsub(st, mv))))
    ems = [s for s in cands if abs(int(s.get("particle_id") or 0)) == 11
           and (s.get("kine_charge") or 0) >= min_mev
           and (s.get("total_length") or 0) >= 3.0]
    print("\n-- pairs (vertex convention at main vertex | axis convention | backproject) --")
    for i in range(len(ems)):
        for j in range(i + 1, len(ems)):
            s1, s2 = ems[i], ems[j]
            m_v, th_v, m_a, th_a, _, _ = conventions(s1, s2, dump, mv)
            bp = G.pi0_backproject(s1, s2, segs, mv)
            bpv = bp.get("vertex")
            print("  %d+%d E=(%.0f,%.0f): m_vtx=%s th=%s | m_axis=%s th=%s | bp=%s m=%s gap=%s vtx=%s"
                  % (s1["id"], s2["id"], s1.get("kine_charge") or 0, s2.get("kine_charge") or 0,
                     "%.1f" % m_v if m_v else "-", "%.1f" % th_v if th_v else "-",
                     "%.1f" % m_a if m_a else "-", "%.1f" % th_a if th_a else "-",
                     bp.get("verdict"), "%.1f" % bp["mass"] if bp.get("mass") else "-",
                     "%.1f" % bp["gap"] if bp.get("gap") else "-",
                     "(%.1f,%.1f,%.1f)" % tuple(bpv) if bpv else "-"))
    print("\n-- reco kine_pio: flag=%s mass=%s E1=%s E2=%s --" % tuple(
        (dump.get("kine") or {}).get(k) for k in
        ("kine_pio_flag", "kine_pio_mass", "kine_pio_energy_1", "kine_pio_energy_2")))


def write_label(ev, g1, g2, vertex_how, vertex_xyz, note, confidence, nopair=False):
    setname, mrow, dump = load(ev)
    mv = G.pt(dump.get("main_vertex"))
    rr = rescan_row(ev)
    origin = rr["origin"] if rr else None
    d = os.path.join(SX, "em_labels", TAG)
    os.makedirs(d, exist_ok=True)
    out = {
        "eventNo": ev, "runNo": int(mrow["run"]), "subRunNo": int(mrow["subrun"]),
        "sample": mrow["sample"], "origin": origin,
        "scan_tag": TAG, "source": "model pi0 pairing pass, doc pr/132 (pr/126 sec 4h item 0)",
        "arm": os.path.dirname(mrow["dump"]),
        "saved_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "confidence": confidence, "note": note,
        "main_vertex": {"x": mv[0], "y": mv[1], "z": mv[2]},
        "pio": None,
    }
    if not nopair:
        by = {int(s["id"]): s for s in (dump.get("showers") or ())}
        s1, s2 = by[g1], by[g2]
        segs = dump.get("segments") or ()
        if vertex_how == "backproject":
            bp = G.pi0_backproject(s1, s2, segs, mv)
            if not bp.get("vertex"):
                raise SystemExit("backproject failed: %s" % bp.get("verdict"))
            vtx = tuple(bp["vertex"])
        elif vertex_how == "manual":
            vtx = vertex_xyz
        else:
            vtx = mv
        m_v, th_v, m_a, th_a, d1, d2 = conventions(s1, s2, dump, vtx)
        vby = {v["id"]: v for v in (dump.get("vertices") or ())}
        gam = {}
        for k, s, dv in (("1", s1, d1), ("2", s2, d2)):
            gam[k] = {"shower": int(s["id"]), "particle_id": s.get("particle_id"),
                      "flag_shower": True,
                      "energy": s.get("kine_charge"),
                      "energy_as_reconstructed": s.get("kine_charge"),
                      "energy_hypothesis": "as_reconstructed",
                      "kine_hypothesis": "shower",
                      "axis": list(dv), "axis_source": "shower_init_dir",
                      "start": list(G.pt(s.get("start"))),
                      "reco_start": list(G.pt(s.get("start"))),
                      "start_source": "reco", "start_override": None}
        groups = {}
        for s in (dump.get("showers") or ()):
            pid = int(s.get("pio_id", -1))
            if pid >= 0:
                groups.setdefault(str(pid), {"mass": s.get("pio_mass"), "showers": []})[
                    "showers"].append(int(s["id"]))
        out["pio"] = {
            "gammas": gam,
            "vertex": list(vtx), "vertex_how": vertex_how,
            "mass_vertex_convention": m_v, "theta_vertex_convention": th_v,
            "mass_axis_convention": m_a, "theta_axis_convention": th_a,
            "backproject": None, "candidates": [],
            "reco_groups": groups,
            "reco_kine": {k: (dump.get("kine") or {}).get(k) for k in
                          ("kine_pio_flag", "kine_pio_mass", "kine_pio_energy_1",
                           "kine_pio_energy_2", "kine_pio_angle", "kine_pio_vtx_dis")},
        }
    p = os.path.join(d, "labels-evt%d.json" % ev)
    if os.path.exists(p):
        raise SystemExit("%s exists -- refusing to overwrite (M13); delete manually if truly re-adjudicating" % p)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    fmt = lambda x: ("%.1f" % x) if x is not None else "-"
    print("wrote %s%s" % (p, "" if nopair else "  m_vtx=%s m_axis=%s" % (fmt(m_v), fmt(m_a))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=int)
    ap.add_argument("--write", type=int)
    ap.add_argument("--nopair", type=int)
    ap.add_argument("--g1", type=int)
    ap.add_argument("--g2", type=int)
    ap.add_argument("--vertex-how", default="main_vertex",
                    choices=["main_vertex", "backproject", "manual"])
    ap.add_argument("--vertex", help="x,y,z for --vertex-how manual")
    ap.add_argument("--note", default="")
    ap.add_argument("--confidence", default="medium")
    ap.add_argument("--min-mev", type=float, default=10.0)
    a = ap.parse_args()
    if a.packet:
        packet(a.packet, a.min_mev)
    elif a.write:
        if not (a.g1 and a.g2):
            raise SystemExit("--write needs --g1 and --g2")
        vxyz = tuple(float(x) for x in a.vertex.split(",")) if a.vertex else None
        write_label(a.write, a.g1, a.g2, a.vertex_how, vxyz, a.note, a.confidence)
    elif a.nopair:
        write_label(a.nopair, None, None, None, None, a.note, a.confidence, nopair=True)
    else:
        raise SystemExit("one of --packet / --write / --nopair required")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc pr/125 fronts 2+3 -- shower-merge anatomy + manifest-wide collateral scans.

Owner (2026-08-29): 37112 gamma 549 + p 469 should be ONE EM shower ("they
are connected"); 69314 should be one EM shower instead of a cascade of
electrons (38 pdg-11 PF entries, 28 below 5 MeV).

Measured anatomy (this doc sec 3.2): in 37112 the proton-typed shower 9008
shares start vertex 84104 (non-main) with the gamma 67048 at 1.28 cm cloud
gap; the pr/124 tier-2 prune detaches 9008's 12-seg component because it is
28.8 cm from 9008's OWN kept stem -- but only 3.81 cm from the gamma.  In
69314 the satellites sit 17-35 cm from the main 68.9 MeV shower.

Three scans over the OFF dumps (dbgv2 + dbg141v2, no new arms):

  pairs      K3 candidates: (bigger EM shower, smaller shower) pairs that
             share a start vertex or sit within a small cloud gap; report
             type/size/gap/vertex-sharing/main-vertex/label status.
  satellites K5 candidates: small showers (E < e_max) within R of a big
             shower (E > e_big); count per R, label + pio status.
  prunecomp  K2 check: for every tier-2 fired component (OFF vs flipA
             membership diff), min cloud gap to every OTHER shower ->
             would a neighbor-contiguity exemption spare 37112's comp
             while leaving the shipped pr/124 wins pruned?

Repro:
  ./scripts/pr125_merge_anatomy.py pairs      --tsv docs/pr/pr125-pairs.tsv
  ./scripts/pr125_merge_anatomy.py satellites --tsv docs/pr/pr125-satellites.tsv
  ./scripts/pr125_merge_anatomy.py prunecomp  --tsv docs/pr/pr125-prunecomp.tsv
"""
import argparse
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
OFF_ROOTS = ("work-pr124r1-dbg141v2-*", "work-pr124r1-dbgv2-*")
ON_ROOTS = ("work-pr124r1-flipA141-*", "work-pr124r1-flipA98-*")
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]
TRK_PIDS = (13, 211, 2212)


def load_marks(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            em = json.load(open(p)).get("em") or {}
            marks = em.get("marks_by_shower") or {}
            ins, outs = set(), set()
            for mm in marks.values():
                for s, v in mm.items():
                    (ins if v == "in" else outs).add(int(s))
            return ins, outs
    return None


def iter_events(roots):
    seen = set()
    for g in roots:
        for r in sorted(glob.glob(os.path.join(SX, g))):
            for dj in sorted(glob.glob(os.path.join(r, "pr_evt*", "calib-pr-evt*.json"))):
                ev = int(os.path.basename(os.path.dirname(dj))[len("pr_evt"):])
                if ev in seen:
                    continue
                seen.add(ev)
                yield ev, dj, os.path.basename(r)


class Evt:
    def __init__(self, dj):
        j = json.load(open(dj))
        self.segs = {s["id"]: s for s in j.get("segments", [])}
        self.verts = {v["id"]: v for v in j.get("vertices", [])}
        self.shws = {s["id"]: s for s in j.get("showers", [])}
        self.mem = {}
        for sid, s in self.segs.items():
            shw = s.get("shower_id", -1)
            if shw is not None and shw >= 0:
                self.mem.setdefault(shw, []).append(sid)
        self.mvid = next((vid for vid, v in self.verts.items() if v.get("is_main")), None)
        self._clouds = {}

    def cloud(self, shw):
        if shw not in self._clouds:
            pts = [[p["x"], p["y"], p["z"]]
                   for m in self.mem.get(shw, []) for p in self.segs[m].get("points", [])]
            self._clouds[shw] = np.array(pts) if pts else None
        return self._clouds[shw]

    def gap(self, a, b):
        ca, cb = self.cloud(a), self.cloud(b)
        if ca is None or cb is None:
            return -1.0
        return math.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(2).min())


def scan_pairs(args):
    rows = []
    for ev, dj, root in iter_events(OFF_ROOTS):
        e = Evt(dj)
        lab = load_marks(ev)
        big = [i for i, s in e.shws.items()
               if abs(s.get("particle_id", 0)) == 11 and s.get("kine_charge", 0) > 20
               and e.cloud(i) is not None]
        for bi in big:
            bs = e.shws[bi]
            for fi, fs in e.shws.items():
                if fi == bi or e.cloud(fi) is None:
                    continue
                if not (fs.get("kine_charge", 0) < bs.get("kine_charge", 0)):
                    continue
                sharev = int(fs.get("start_vertex_id") == bs.get("start_vertex_id"))
                g = e.gap(bi, fi)
                if not sharev and (g < 0 or g > 6.0):
                    continue
                fseg = e.segs.get(fi, {})
                marked = "-"
                if lab:
                    ins, outs = lab
                    ms = set(e.mem.get(fi, []))
                    marked = ("in" if ms & ins else "out" if ms & outs else "unl")
                rows.append(dict(
                    ev=ev, big=bi, frag=fi,
                    big_pdg=bs.get("particle_id"), frag_pdg=fs.get("particle_id"),
                    big_E=round(bs.get("kine_charge", 0), 1),
                    frag_E=round(fs.get("kine_charge", 0), 1),
                    frag_nseg=fs.get("num_segments", -1),
                    frag_len=round(fs.get("total_length", 0), 1),
                    frag_conn=fs.get("start_connection_type", -1),
                    sharev=sharev,
                    at_main=int(bs.get("start_vertex_id") == e.mvid or
                                fs.get("start_vertex_id") == e.mvid),
                    gap=round(g, 2),
                    frag_track=int(abs(fs.get("particle_id", 0)) in TRK_PIDS),
                    frag_score=fseg.get("particle_score", -1.0),
                    pio=int(fs.get("pio_id", -1) >= 0),
                    marks=marked))
    emit(rows, args, ["ev", "big", "frag", "big_pdg", "frag_pdg", "big_E", "frag_E",
                      "frag_nseg", "frag_len", "frag_conn", "sharev", "at_main",
                      "gap", "frag_track", "frag_score", "pio", "marks"])
    trk = [r for r in rows if r["frag_track"]]
    print("\n%d close/vertex-sharing (big-EM, smaller) pairs; %d with track-typed frag"
          % (len(rows), len(trk)))
    print("track-typed frags (K3 exposure), by sharev/gap:")
    for r in sorted(trk, key=lambda r: (-r["sharev"], r["gap"])):
        print("  evt%-7d big=%-6d(%5d,%7.1f) frag=%-6d pdg=%-5d E=%-7.1f len=%-6.1f "
              "conn=%d sharev=%d at_main=%d gap=%-6.2f pio=%d marks=%s"
              % (r["ev"], r["big"], r["big_pdg"], r["big_E"], r["frag"],
                 r["frag_pdg"], r["frag_E"], r["frag_len"], r["frag_conn"],
                 r["sharev"], r["at_main"], r["gap"], r["pio"], r["marks"]))


def scan_satellites(args):
    RS = (10.0, 20.0, 30.0, 40.0)
    rows = []
    for ev, dj, root in iter_events(OFF_ROOTS):
        e = Evt(dj)
        lab = load_marks(ev)
        big = [i for i, s in e.shws.items()
               if abs(s.get("particle_id", 0)) == 11 and s.get("kine_charge", 0) > 20
               and e.cloud(i) is not None]
        small = [i for i, s in e.shws.items()
                 if s.get("kine_charge", 0) < 10 and i not in big
                 and e.cloud(i) is not None]
        for si in small:
            ss = e.shws[si]
            gaps = sorted((e.gap(bi, si), bi) for bi in big)
            if not gaps or gaps[0][0] < 0 or gaps[0][0] > max(RS):
                continue
            g, bi = gaps[0]
            marked = "-"
            if lab:
                ins, outs = lab
                ms = set(e.mem.get(si, []))
                marked = ("in" if ms & ins else "out" if ms & outs else "unl")
            rows.append(dict(
                ev=ev, sat=si, near=bi, gap=round(g, 2),
                sat_pdg=ss.get("particle_id"), sat_E=round(ss.get("kine_charge", 0), 2),
                sat_len=round(ss.get("total_length", 0), 1),
                sat_conn=ss.get("start_connection_type", -1),
                sat_track=int(abs(ss.get("particle_id", 0)) in TRK_PIDS),
                pio=int(ss.get("pio_id", -1) >= 0),
                near_E=round(e.shws[bi].get("kine_charge", 0), 1),
                marks=marked))
    emit(rows, args, ["ev", "sat", "near", "gap", "sat_pdg", "sat_E", "sat_len",
                      "sat_conn", "sat_track", "pio", "near_E", "marks"])
    print("\nsatellite candidates (E<10 within %g cm of an EM shower E>20): %d"
          % (max(RS), len(rows)))
    for R in RS:
        sel = [r for r in rows if r["gap"] <= R]
        n_pio = sum(1 for r in sel if r["pio"])
        n_trk = sum(1 for r in sel if r["sat_track"])
        n_out = sum(1 for r in sel if r["marks"] == "out")
        n_in = sum(1 for r in sel if r["marks"] == "in")
        ne = sum(1 for r in sel if r["ev"] == 69314)
        print("  R=%4.0f : %4d sats (%2d in evt 69314) | pio-paired %d | "
              "track-typed %d | marks in/out %d/%d"
              % (R, len(sel), ne, n_pio, n_trk, n_in, n_out))
    print("\npio-paired satellites (pi0 exposure — must be inspected before K5):")
    for r in rows:
        if r["pio"]:
            print("  evt%-7d sat=%-6d E=%-6.2f gap=%-6.2f near=%-6d(E=%.1f) marks=%s"
                  % (r["ev"], r["sat"], r["sat_E"], r["gap"], r["near"],
                     r["near_E"], r["marks"]))


def scan_prunecomp(args):
    on_dumps = {}
    for ev, dj, root in iter_events(ON_ROOTS):
        on_dumps[ev] = dj
    rows = []
    for ev, dj, root in iter_events(OFF_ROOTS):
        if ev not in on_dumps:
            continue
        off = Evt(dj)
        jn = json.load(open(on_dumps[ev]))
        on_own = {s["id"]: s.get("shower_id", -1) for s in jn.get("segments", [])}
        # tier-2 fired comps: segs owned by S in OFF that left S in ON,
        # grouped by (old owner, new owner)
        moved = {}
        for sid, s in off.segs.items():
            o = s.get("shower_id", -1)
            n = on_own.get(sid, -1)
            if o is None or o < 0 or n == o:
                continue
            moved.setdefault((o, n if n is not None else -1), []).append(sid)
        if not moved:
            continue
        lab = load_marks(ev)
        for (o, n), sids in sorted(moved.items()):
            pts = [[p["x"], p["y"], p["z"]]
                   for m in sids for p in off.segs[m].get("points", [])]
            if not pts:
                continue
            cp = np.array(pts)
            best_g, best_s = -1.0, -1
            for oi in off.shws:
                if oi == o or off.cloud(oi) is None:
                    continue
                other = off.cloud(oi)
                # exclude the comp's own points from the old owner comparison
                g = math.sqrt(((cp[:, None, :] - other[None, :, :]) ** 2).sum(2).min())
                if best_g < 0 or g < best_g:
                    best_g, best_s = g, oi
            marked = "-"
            if lab:
                ins, outs = lab
                marked = ("in" if set(sids) & ins else
                          "out" if set(sids) & outs else "unl")
            rows.append(dict(ev=ev, old=o, new=n, nseg=len(sids),
                             q=round(sum(abs(p.get("dQ", 0.0))
                                         for m in sids
                                         for p in off.segs[m].get("points", [])), 0),
                             neigh=best_s, neigh_gap=round(best_g, 2),
                             neigh_E=round(off.shws.get(best_s, {}).get("kine_charge", -1), 1)
                             if best_s in off.shws else -1,
                             marks=marked))
    emit(rows, args, ["ev", "old", "new", "nseg", "q", "neigh", "neigh_gap",
                      "neigh_E", "marks"])
    print("\ntier-2 fired comps (OFF->flipA membership diffs) sorted by neighbor gap:")
    for r in sorted(rows, key=lambda r: r["neigh_gap"]):
        print("  evt%-7d old=%-6d new=%-6d nseg=%-3d q=%-9.0f neigh=%-6d "
              "gap=%-7.2f E=%-8.1f marks=%s"
              % (r["ev"], r["old"], r["new"], r["nseg"], r["q"], r["neigh"],
                 r["neigh_gap"], r["neigh_E"], r["marks"]))


def emit(rows, args, cols):
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("wrote %d rows -> %s" % (len(rows), args.tsv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "satellites", "prunecomp"])
    ap.add_argument("--tsv")
    args = ap.parse_args()
    {"pairs": scan_pairs, "satellites": scan_satellites,
     "prunecomp": scan_prunecomp}[args.mode](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc 84 -- long-muon chain construction and MCS delta-ray audit.

Read-only census over EXISTING arms.  No WCT job is run, no arm is written.

Three measurements, all from `calib-pr-evt<ID>.json` (written only under
PR_EXTRA_STAGES=pr_display, which the prod0825 arms used) plus the MCS log
sentinel in an mcs80-on arm:

  Part 2  the long-muon energy consumer
          - delta-ray contamination of `calculate_kinematics_long_muon`'s
            dQ/dx integral (PRShower.cxx: `total_length` sums chain members
            only, `vec_dQ`/`vec_dx` sum EVERY pseudo-shower member)
          - chain completeness: L_chain recovered by inverting the shipped
            SBND muon range->KE table on `showers[].kine_range`, against the
            muon-typed member length actually present in the pseudo-shower
          - which estimator won `kine_best`, and why mode 2 fell back

  Part 3  gate replay: re-run `find_cont_muon_segment` + the accept gate
          offline on the FINAL graph, with and without the SBND-ON
          `long_muon_stub_bridge`, and record which cut stopped the walk.

          CAVEAT, stated in the doc too: calib JSON is the FINAL PR state;
          `examine_direction` ran earlier, on a graph that may since have
          changed.  The replay therefore measures "on the final graph, is
          there a collinear MIP continuation the shipped chain does not
          contain", not a bit-exact re-execution of the in-flight decision.

  Part 4  join the MCS sentinel (`mcs: source=... ke_MCS=...`) onto the above.

Usage:
  d84_longmu_census.py --calib-arms DIR:LABEL [DIR:LABEL ...]
                       [--mcs-arm DIR] --out OUTDIR [--out-exist-ok]
"""
import argparse
import bisect
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys

# The toolkit tree.  NOT derived from __file__: sbnd_xin is a symlink into the
# wcp-porting-img repo, so realpath(__file__) lands in the wrong tree.  Run
# this script from sbnd_xin (its parent is the toolkit), or pass --toolkit.
def find_toolkit(explicit=None):
    # $PWD keeps the LOGICAL path (sbnd_xin is a symlink; getcwd() resolves it
    # into the wcp-porting-img tree, which has no cfg/).
    logical = os.environ.get("PWD") or os.getcwd()
    for cand in (explicit, os.environ.get("WCT_TOOLKIT"),
                 os.path.abspath(os.path.join(logical, "..")),
                 os.path.abspath(logical),
                 os.path.abspath(os.path.join(os.getcwd(), "..")),
                 os.path.abspath(os.getcwd())):
        if cand and os.path.exists(os.path.join(
                cand, "cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet")):
            return cand
    raise SystemExit("cannot locate the toolkit tree; run from sbnd_xin or pass --toolkit")

# ---------------------------------------------------------------- constants
# NeutrinoVertexFinder.cxx examine_direction / find_cont_muon_segment.
MIP_RATIO_CUT = 1.3          # seed and candidate dQ/dx gate
ANGLE_CUT = 10.0             # deg, 15 cm lever arm
ANGLE_CUT_SHORT = 15.0       # deg, when the incoming segment is < 6 cm
SHORT_SEG = 6.0              # cm
LEVER_NEAR = 15.0            # cm
LEVER_FAR = 50.0             # cm, used when the candidate is > 50 cm
LONG_CAND = 50.0             # cm
ACCEPT_TOTAL = 45.0          # cm
ACCEPT_MAX = 35.0            # cm
# long_muon_stub_bridge (doc pr/46), SBND PRODUCTION ON
BRIDGE_ANGLE = 45.0          # deg
BRIDGE_MIN_LEN = 35.0        # cm
BRIDGE_VETO_LEN = 10.0       # cm

# kine_long_muon_mode 2 (doc pr/101 K4), SBND PRODUCTION ON
RATIO_LO, RATIO_HI = 0.3, 0.5

# PowerBoxRecombination, SBND operating point
# (cfg/pgrapher/experiment/sbnd/clus.jsonnet sbnd_power_recomb, use_power_recomb=true)
RC = dict(A=0.93, k=0.282371, p=1.362179, C=0.855175, pivot=2.1, Wi=23.6e-6, dedx_max=77.0)

SENT = re.compile(
    r"mcs: source=(?P<source>\S+) nseg=(?P<nseg>\d+) npoints=(?P<npoints>\d+) "
    r"len=(?P<len>[\d.]+)cm seg_id=(?P<segid>-?\d+) cluster=(?P<cid>-?\d+) -> "
    r"ke_MCS=(?P<ke>-?[\d.]+) MeV amb=(?P<amb>[\d.eE+-]+) tracklen=(?P<tracklen>-?[\d.]+)cm "
    r"ke_range=(?P<kerange>-?[\d.]+) MeV ke_range_toolkit=(?P<kert>-?[\d.]+) MeV "
    r"ke_dqdx_toolkit=(?P<kedq>-?[\d.]+) MeV")


# ------------------------------------------------------------- range tables
def load_muon_range(cache, toolkit):
    """coords = range [cm], values = KE [MeV] of the shipped SBND LinterpFunction."""
    if not os.path.exists(cache):
        env = dict(os.environ)
        env["WIRECELL_PATH"] = "%s/cfg:%s" % (
            toolkit, os.path.join(os.path.dirname(toolkit), "wire-cell-data"))
        cfg = os.path.join(toolkit, "cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet")
        exe = os.path.join(toolkit, "build/apps/wcsonnet")
        if not os.path.exists(exe):
            exe = "wcsonnet"
        with open(cache, "w") as fp:
            rc = subprocess.call([exe, cfg], stdout=fp, env=env)
        if rc != 0:
            raise SystemExit("wcsonnet failed on particle_dataset.jsonnet (rc=%d)" % rc)
    d = json.load(open(cache))["muon_range_function"]["data"]
    return d["coords"], d["values"]


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0


# ------------------------------------------------------- recombination (dE)
def _fwd_dqdx(dedx):
    if dedx <= 0:
        return 0.0
    u = RC["k"] * (dedx / RC["pivot"]) ** RC["p"]
    return RC["C"] * math.log(RC["A"] + u) / u * dedx / RC["Wi"]


_FWD_MAX = _fwd_dqdx(RC["dedx_max"])


def dE_MeV(dQ, dx_cm):
    """Gen::PowerBoxRecombination::dE, reproduced.  dx in cm, returns MeV."""
    if dx_cm <= 0:
        return 0.0
    dqdx = dQ / dx_cm
    if not dqdx > 0:
        return 0.0
    if dqdx >= _FWD_MAX:
        return RC["dedx_max"] * dx_cm
    lo, hi = 0.0, RC["dedx_max"]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _fwd_dqdx(mid) < dqdx:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * dx_cm


def kine_dQdx(points):
    """PRSegmentFunctions::cal_kine_dQdx over a list of {dQ,dx} points [MeV]."""
    tot = 0.0
    for p in points:
        dQ, dx = p.get("dQ", 0.0), p.get("dx", 0.0)
        if dx <= 0:
            continue
        if dQ / dx / 43e3 > 1000:
            dQ = 0.0
        dE = dE_MeV(dQ, dx)
        if dE < 0:
            dE = 0.0
        if dE > 50.0 * dx:
            dE = 50.0 * dx
        tot += dE
    return tot


# --------------------------------------------------------------- geometry
def seg_median_ratio(seg, mip):
    v = [p["dQ"] / (p["dx"] + 1e-9) for p in seg["points"] if p.get("dx", 0) > 0 and p.get("dQ", -1) >= 0]
    if not v:
        return 0.0
    v.sort()
    return v[len(v) // 2] / mip


def cal_dir(seg, pt, lever):
    """segment_cal_dir_3vector(seg, p, dis_cut): mean of fit points within
    dis_cut of p, minus p, normalised."""
    sx = sy = sz = 0.0
    n = 0
    px, py, pz = pt
    for p in seg["points"]:
        dx, dy, dz = p["x"] - px, p["y"] - py, p["z"] - pz
        if dx * dx + dy * dy + dz * dz < lever * lever:
            sx += p["x"]
            sy += p["y"]
            sz += p["z"]
            n += 1
    if n == 0:
        return None
    vx, vy, vz = sx / n - px, sy / n - py, sz / n - pz
    m = math.sqrt(vx * vx + vy * vy + vz * vz)
    if m <= 0:
        return None
    return (vx / m, vy / m, vz / m)


def cont_angle(d1, d2):
    """(180 deg - angle between) -- the toolkit's collinearity measure."""
    c = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]))
    return (math.pi - math.acos(c)) / math.pi * 180.0


def seg_is_showerish(seg):
    return bool(seg.get("flag_shower")) or abs(seg.get("particle_id", 0)) == 11


# ---------------------------------------------------------------- the event
class Evt:
    def __init__(self, path):
        self.path = path
        d = json.load(open(path))
        self.meta = d["meta"]
        self.mip = float(self.meta.get("mip_dqdx_median", 43000.0))
        self.segs = {s["id"]: s for s in d["segments"]}
        self.vtxs = {v["id"]: v for v in d["vertices"]}
        self.showers = d["showers"]
        self.main_vertex = d.get("main_vertex")
        self.adj = collections.defaultdict(list)
        for s in d["segments"]:
            for k in ("start_vertex_id", "end_vertex_id"):
                vid = s.get(k, -1)
                if vid in self.vtxs:
                    self.adj[vid].append(s["id"])
        for vid in self.adj:                       # deterministic order
            self.adj[vid].sort()
        self.evt = int(self.meta["eventNo"])
        self.run = int(self.meta["runNo"])
        self.subrun = int(self.meta["subRunNo"])
        # the main-vertex id, matched on the fitted point (main_vertex block
        # carries no id); fall back to vertices[].is_main
        self.mvid = None
        for v in d["vertices"]:
            if v.get("is_main"):
                self.mvid = v["id"]
                break
        if self.mvid is None and self.main_vertex:
            best, bd = None, 1e9
            for v in d["vertices"]:
                f = v["fit"]
                dd = ((f["x"] - self.main_vertex["x"]) ** 2 + (f["y"] - self.main_vertex["y"]) ** 2
                      + (f["z"] - self.main_vertex["z"]) ** 2)
                if dd < bd:
                    bd, best = dd, v["id"]
            if bd < 1e-4:
                self.mvid = best

    def vpt(self, vid):
        f = self.vtxs[vid]["fit"]
        return (f["x"], f["y"], f["z"])

    def other_vtx(self, sid, vid):
        s = self.segs[sid]
        a, b = s.get("start_vertex_id", -1), s.get("end_vertex_id", -1)
        if a == vid:
            return b
        if b == vid:
            return a
        return -1

    # ---- find_cont_muon_segment, replayed ----
    def cont(self, sid, vid, stub_bridge):
        """Returns (best_sid, best_vid, reject_summary).

        reject_summary describes the best NON-taken candidate when nothing
        qualifies: (reason, cand_len, angle, ratio)."""
        if vid not in self.adj:
            return None, None, ("no_vertex", 0, -1, -1)
        sg = self.segs[sid]
        sg_len = sg["length"]
        vpt = self.vpt(vid)
        d1 = cal_dir(sg, vpt, LEVER_NEAR)
        best = None
        best_proj = 0.0
        rej = None                                   # (reason, len, angle, ratio)
        others = [x for x in self.adj[vid] if x != sid]
        for cid in others:
            sg2 = self.segs[cid]
            v2 = self.other_vtx(cid, vid)
            if v2 not in self.vtxs:
                continue
            d2 = cal_dir(sg2, vpt, LEVER_NEAR)
            if d1 is None or d2 is None:
                continue
            length = sg2["length"]
            angle = cont_angle(d1, d2)
            ratio = seg_median_ratio(sg2, self.mip)
            angle1 = angle
            if length > LONG_CAND:
                d3 = cal_dir(sg, vpt, LEVER_FAR)
                d4 = cal_dir(sg2, vpt, LEVER_FAR)
                if d3 and d4:
                    angle1 = cont_angle(d3, d4)
            angle_ok = (angle < ANGLE_CUT or angle1 < ANGLE_CUT or
                        (sg_len < SHORT_SEG and (angle < ANGLE_CUT_SHORT or angle1 < ANGLE_CUT_SHORT)))
            ratio_ok = ratio < MIP_RATIO_CUT
            bridged = False
            if (stub_bridge and not angle_ok and ratio_ok and sg_len < SHORT_SEG and
                    length > BRIDGE_MIN_LEN and (angle < BRIDGE_ANGLE or angle1 < BRIDGE_ANGLE)):
                has_other = False
                for oid in others:
                    if oid == cid:
                        continue
                    o = self.segs[oid]
                    if seg_is_showerish(o):
                        continue
                    if o["length"] > BRIDGE_VETO_LEN:
                        has_other = True
                        break
                if not has_other:
                    angle_ok = True
                    bridged = True
            if angle_ok and ratio_ok:
                proj = length * math.cos(angle / 180.0 * math.pi)
                if proj > best_proj:
                    best_proj = proj
                    best = (cid, v2, bridged)
            else:
                reason = ("angle" if ratio_ok else ("dqdx" if angle_ok else "both"))
                cand = (reason, length, min(angle, angle1), ratio)
                if rej is None or length > rej[1]:
                    rej = cand
        if best:
            return best[0], best[1], ("taken", self.segs[best[0]]["length"], -1, -1)
        if rej is None:
            rej = ("dead_end", 0.0, -1, -1)
        return None, None, rej

    def walk(self, seed_sid, seed_vid, stub_bridge):
        segs = [seed_sid]
        vtxs = [seed_vid]
        seen = {seed_sid}
        sid, vid = seed_sid, seed_vid
        stop = ("dead_end", 0.0, -1, -1)
        stop_vid = seed_vid
        for _ in range(64):                          # walk guard (no visited set upstream)
            nsid, nvid, why = self.cont(sid, vid, stub_bridge)
            if nsid is None or nsid in seen:
                stop = why if nsid is None else ("loop", 0.0, -1, -1)
                stop_vid = vid
                break
            segs.append(nsid)
            vtxs.append(nvid)
            seen.add(nsid)
            sid, vid = nsid, nvid
            stop_vid = vid
        return segs, vtxs, stop, stop_vid

    def replay(self, stub_bridge):
        """examine_direction's long-muon block, replayed at the main vertex."""
        out = []
        if self.mvid is None or self.mvid not in self.adj:
            return out
        for sid in self.adj[self.mvid]:
            sg = self.segs[sid]
            if seg_median_ratio(sg, self.mip) > MIP_RATIO_CUT:
                out.append(dict(seed=sid, accept=False, gate="seed_dqdx", nseg=0,
                                total=0.0, maxlen=0.0, stop="seed_dqdx",
                                stop_len=0.0, stop_angle=-1, stop_ratio=-1,
                                stop_degree=-1))
                continue
            v = self.other_vtx(sid, self.mvid)
            if v not in self.vtxs:
                continue
            segs, _v, stop, stop_vid = self.walk(sid, v, stub_bridge)
            lens = [self.segs[x]["length"] for x in segs]
            total, mx = sum(lens), max(lens)
            ok = total > ACCEPT_TOTAL and mx > ACCEPT_MAX and len(segs) > 1
            gate = "accepted"
            if not ok:
                if len(segs) <= 1:
                    gate = "size1"
                elif mx <= ACCEPT_MAX:
                    gate = "maxlen"
                elif total <= ACCEPT_TOTAL:
                    gate = "total"
            out.append(dict(seed=sid, accept=ok, gate=gate, nseg=len(segs),
                            total=total, maxlen=mx, stop=stop[0],
                            stop_len=stop[1], stop_angle=stop[2], stop_ratio=stop[3],
                            stop_degree=len(self.adj.get(stop_vid, [])), segs=segs))
        return out


# ----------------------------------------------------------------- Part 2/3
def census_event(ev, xs, ys):
    """One row per muon-typed pseudo-shower; plus the replay summary."""
    by_sh = collections.defaultdict(list)
    for s in ev.segs.values():
        by_sh[s.get("shower_id", -1)].append(s)

    rows = []
    for sh in ev.showers:
        if abs(sh.get("particle_id", 0)) != 13:
            continue
        mem = by_sh.get(sh["id"], [])
        mu = [s for s in mem if abs(s.get("particle_id", 0)) == 13]
        oth = [s for s in mem if abs(s.get("particle_id", 0)) != 13]
        L_mu = sum(s["length"] for s in mu)
        L_oth = sum(s["length"] for s in oth)
        dQ_mu = sum(p["dQ"] for s in mu for p in s["points"])
        dQ_oth = sum(p["dQ"] for s in oth for p in s["points"])
        # the shipped dQ/dx integral (all members) vs chain-members-only
        pts_all = [p for s in mem for p in s["points"]]
        pts_mu = [p for s in mu for p in s["points"]]
        e_all = kine_dQdx(pts_all)
        e_mu = kine_dQdx(pts_mu)
        # chain length actually used by cal_kine_range, recovered by inversion
        kr = sh.get("kine_range", 0.0)
        L_chain = interp(ys, xs, kr) if kr > 0 else 0.0
        ratio = (sh["kine_dQdx"] / kr) if kr > 0 else -1.0
        best, tag = sh.get("kine_best", 0.0), "other"
        for k in ("kine_range", "kine_dQdx", "kine_charge"):
            if abs(sh.get(k, -9e9) - best) < 1e-6:
                tag = k[5:]
                break
        why = ""
        if tag != "range":
            if kr <= 0:
                why = "range_zero"
            elif not (1.0 - RATIO_LO <= ratio <= 1.0 + RATIO_HI):
                why = "ratio_outside"
            else:
                why = "end_degree_or_override"
        # The owner's signature: an internal junction of the muon where a
        # third arm (delta ray) hangs off.  Counted over the vertices shared
        # by >= 2 muon-typed members of this pseudo-shower.
        mu_ids = set(s["id"] for s in mu)
        touch = collections.Counter()
        for sid in mu_ids:
            for k in ("start_vertex_id", "end_vertex_id"):
                vid = ev.segs[sid].get(k, -1)
                if vid in ev.vtxs:
                    touch[vid] += 1
        n_deg3 = 0
        min_arm = -1.0
        for vid, nmu_here in touch.items():
            if nmu_here < 2:
                continue
            arms = [x for x in ev.adj.get(vid, []) if x not in mu_ids]
            if not arms:
                continue
            n_deg3 += 1
            L = min(ev.segs[x]["length"] for x in arms)
            if min_arm < 0 or L < min_arm:
                min_arm = L

        # PRECONDITION for the kine_range inversion: cal_kine_range is called
        # (PRShower.cxx:1781) with the START SEGMENT's pdg, not the shower type
        # filtered on here, so L_chain is only meaningful when that is 13.
        # showers[].id == pf_node_id(start_segment) (PrDisplayDump.cxx:577).
        start_seg = ev.segs.get(sh["id"])
        start_pdg = abs(start_seg["particle_id"]) if start_seg else 0

        rows.append(dict(
            run=ev.run, subrun=ev.subrun, evt=ev.evt, shower=sh["id"],
            n_deg3_internal=n_deg3, min_arm_cm=min_arm, start_pdg=start_pdg,
            n_mem=len(mem), n_mu=len(mu), n_oth=len(oth),
            L_mu_cm=L_mu, L_oth_cm=L_oth, L_chain_cm=L_chain,
            L_unchained_cm=max(0.0, L_mu - L_chain),
            dQ_frac_oth=(dQ_oth / (dQ_mu + dQ_oth) if (dQ_mu + dQ_oth) > 0 else 0.0),
            kine_range=kr, kine_dQdx=sh.get("kine_dQdx", 0.0),
            kine_charge=sh.get("kine_charge", 0.0), kine_best=best,
            best_is=tag, fallback_why=why,
            dqdx_all_MeV=e_all, dqdx_chain_MeV=e_mu,
            dqdx_shift_MeV=e_all - e_mu,
            total_length_cm=sh.get("total_length", 0.0),
            ratio_dqdx_range=ratio,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-arms", nargs="+", required=True, metavar="DIR:LABEL")
    ap.add_argument("--mcs-arm", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-exist-ok", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--toolkit", default=None, help="toolkit tree (default: cwd/..)")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.out_exist_ok:
        raise SystemExit("refusing to write into existing %s (M13); pass --out-exist-ok" % a.out)
    os.makedirs(a.out, exist_ok=True)

    toolkit = find_toolkit(a.toolkit)
    xs, ys = load_muon_range(os.path.join(a.out, "sbnd_particle_dataset.json"), toolkit)
    # sanity: the table must be monotone so the KE->range inversion is defined
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)), "muon range table not monotone"

    files = []
    for spec in a.calib_arms:
        d, _, lab = spec.partition(":")
        lab = lab or os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "pr_evt*", "calib-pr-evt*.json"))):
            files.append((lab, p))
    if a.limit:
        files = files[:a.limit]
    print("calib events: %d" % len(files), file=sys.stderr)

    rows = []
    rep_rows = []
    gate = collections.Counter()
    stopc = {True: collections.Counter(), False: collections.Counter()}
    n_evt = 0
    for i, (lab, p) in enumerate(files):
        if i % 200 == 0:
            print("  %d/%d" % (i, len(files)), file=sys.stderr)
        try:
            ev = Evt(p)
        except Exception as exc:                       # noqa: BLE001
            print("  SKIP %s: %s" % (p, exc), file=sys.stderr)
            continue
        n_evt += 1
        for r in census_event(ev, xs, ys):
            r["arm"] = lab
            rows.append(r)
        for sb in (True, False):
            for r in ev.replay(sb):
                stopc[sb][r["stop"]] += 1
                if sb:
                    gate[r["gate"]] += 1
                rep_rows.append(dict(arm=lab, run=ev.run, evt=ev.evt, stub_bridge=int(sb),
                                     seed=r["seed"], accept=int(r["accept"]), gate=r["gate"],
                                     nseg=r["nseg"], total_cm=r["total"], maxlen_cm=r["maxlen"],
                                     stop=r["stop"], stop_len_cm=r["stop_len"],
                                     stop_angle_deg=r["stop_angle"], stop_ratio=r["stop_ratio"],
                                     stop_degree=r["stop_degree"]))

    def write_tsv(path, recs, cols):
        with open(path, "w") as fp:
            fp.write("\t".join(cols) + "\n")
            for r in recs:
                fp.write("\t".join(
                    ("%.4g" % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    cols = ["arm", "run", "subrun", "evt", "shower", "n_mem", "n_mu", "n_oth",
            "L_mu_cm", "L_oth_cm", "L_chain_cm", "L_unchained_cm", "dQ_frac_oth",
            "kine_range", "kine_dQdx", "kine_charge", "kine_best", "best_is",
            "fallback_why", "dqdx_all_MeV", "dqdx_chain_MeV", "dqdx_shift_MeV",
            "total_length_cm", "ratio_dqdx_range", "n_deg3_internal", "min_arm_cm",
            "start_pdg"]
    write_tsv(os.path.join(a.out, "d84-longmu-showers.tsv"), rows, cols)

    rcols = ["arm", "run", "evt", "stub_bridge", "seed", "accept", "gate", "nseg",
             "total_cm", "maxlen_cm", "stop", "stop_len_cm", "stop_angle_deg",
             "stop_ratio", "stop_degree"]
    write_tsv(os.path.join(a.out, "d84-gate-replay.tsv"), rep_rows, rcols)

    # ------------------------------------------------------------ Part 4
    mcs = {}
    if a.mcs_arm:
        for lg in sorted(glob.glob(os.path.join(a.mcs_arm, "pr_evt*", "wct_pr_evt*.log"))):
            evt = int(re.search(r"evt(\d+)", os.path.basename(lg)).group(1))
            with open(lg, errors="ignore") as fp:
                for line in fp:
                    m = SENT.search(line)
                    if m:
                        mcs.setdefault(evt, []).append(
                            {k: m.group(k) for k in
                             ("source", "nseg", "npoints", "len", "segid", "cid",
                              "ke", "amb", "tracklen", "kerange", "kert", "kedq")})
        with open(os.path.join(a.out, "d84-mcs-sentinel.tsv"), "w") as fp:
            fp.write("evt\tsource\tnseg\tnpoints\tlen_cm\tseg_id\tcluster\tke_MCS\tamb\t"
                     "tracklen_cm\tke_range_mcs\tke_range_toolkit\tke_dqdx_toolkit\n")
            for evt in sorted(mcs):
                for r in mcs[evt]:
                    fp.write("\t".join([str(evt)] + [r[k] for k in
                             ("source", "nseg", "npoints", "len", "segid", "cid",
                              "ke", "amb", "tracklen", "kerange", "kert", "kedq")]) + "\n")

    # ---- Part 4b: WITHIN-ARM check that the MCS muon is the whole muon.
    # The sentinel's ke_range_toolkit is cal_kine_range over the SELECTED
    # segment only; T_kine's kine_energy_particle for the muon is what the PR
    # chain assigned the muon (the long-muon chain for a broken one).  Both
    # come from the same arm and the same job, so this needs no cross-arm join
    # (a cross-arm join by event number is NOT valid: the arms are different
    # binaries and segment the same track differently).
    mcs_pr = []
    if a.mcs_arm and mcs:
        try:
            import uproot
        except ImportError:
            uproot = None
        if uproot is not None:
            for evt in sorted(mcs):
                rp = os.path.join(a.mcs_arm, "pr_evt%d" % evt, "tracking-pr.root")
                if not os.path.exists(rp):
                    continue
                try:
                    t = uproot.open(rp)["T_kine"]
                    ep = t["kine_energy_particle"].array(library="np")[0]
                    pt = t["kine_particle_type"].array(library="np")[0]
                except Exception:                       # noqa: BLE001
                    continue
                mu = [float(ep[i]) for i in range(len(pt)) if abs(int(pt[i])) == 13]
                if not mu:
                    continue
                m = mcs[evt][0]
                mcs_pr.append(dict(evt=evt, mcs_len_cm=float(m["len"]),
                                   ke_MCS=float(m["ke"]),
                                   ke_range_toolkit=float(m["kert"]),
                                   ke_dqdx_toolkit=float(m["kedq"]),
                                   pr_muon_KE=max(mu),
                                   pr_over_fragment=(max(mu) / float(m["kert"])
                                                     if float(m["kert"]) > 20 else -1.0)))
            write_tsv(os.path.join(a.out, "d84-mcs-vs-pr-muon.tsv"), mcs_pr,
                      ["evt", "mcs_len_cm", "ke_MCS", "ke_range_toolkit",
                       "ke_dqdx_toolkit", "pr_muon_KE", "pr_over_fragment"])

    # ------------------------------------ the deliverable case list (Part 3)
    # One row per muon-typed pseudo-shower that shows EITHER of the two
    # defects this doc is about, joined to the MCS sentinel where one exists.
    cases = []
    for r in rows:
        broken = (r["L_unchained_cm"] > 20.0) or (r["kine_range"] <= 0 and r["L_mu_cm"] > ACCEPT_TOTAL)
        dirty = (r["dQ_frac_oth"] > 0.05)
        if not (broken or dirty):
            continue
        cls = "both" if (broken and dirty) else ("broken_chain" if broken else "deltaray_charge")
        m = (mcs.get(r["evt"]) or [None])[0]
        c = dict(r)
        c["defect"] = cls
        c["mcs_source"] = m["source"] if m else ""
        c["mcs_nseg"] = int(m["nseg"]) if m else -1
        c["mcs_len_cm"] = float(m["len"]) if m else -1.0
        c["mcs_ke"] = float(m["ke"]) if m else -1.0
        c["mcs_ke_range_toolkit"] = float(m["kert"]) if m else -1.0
        cases.append(c)
    cases.sort(key=lambda r: -max(r["L_unchained_cm"], 3.0 * r["dqdx_shift_MeV"] / 10.0))
    ccols = ["arm", "run", "subrun", "evt", "shower", "defect", "n_mu", "n_oth",
             "L_mu_cm", "L_chain_cm", "L_unchained_cm", "start_pdg",
             "n_deg3_internal", "min_arm_cm",
             "dQ_frac_oth", "dqdx_shift_MeV", "kine_range", "kine_dQdx", "kine_charge",
             "kine_best", "best_is", "fallback_why", "mcs_source", "mcs_nseg",
             "mcs_len_cm", "mcs_ke", "mcs_ke_range_toolkit"]
    write_tsv(os.path.join(a.out, "d84-broken-muons.tsv"), cases, ccols)

    # ---------------------------------------------------------- summary
    s = []
    s.append("events with calib JSON: %d" % n_evt)
    s.append("muon-typed pseudo-showers: %d" % len(rows))
    if rows:
        n = len(rows)
        c = collections.Counter()
        for r in rows:
            if r["n_oth"] > 0:
                c["has_nonmu_member"] += 1
            if r["L_oth_cm"] > 0.05 * max(r["L_mu_cm"], 1e-9):
                c["nonmu_len_gt_5pct"] += 1
            if r["dQ_frac_oth"] > 0.05:
                c["nonmu_charge_gt_5pct"] += 1
            if r["kine_range"] <= 0:
                c["range_zero"] += 1
            if r["L_unchained_cm"] > 20:
                c["unchained_gt_20cm"] += 1
            if r["L_unchained_cm"] > 50:
                c["unchained_gt_50cm"] += 1
            c["best=" + r["best_is"]] += 1
            if r["fallback_why"]:
                c["why=" + r["fallback_why"]] += 1
        for k, v in sorted(c.items()):
            s.append("  %-28s %5d / %d  (%.1f%%)" % (k, v, n, 100.0 * v / n))
        sh = sorted(r["dqdx_shift_MeV"] for r in rows)
        s.append("  dQ/dx delta-ray shift [MeV]: median %.1f  p90 %.1f  max %.1f"
                 % (sh[len(sh) // 2], sh[int(0.9 * (len(sh) - 1))], sh[-1]))
        frac = sorted(r["dqdx_shift_MeV"] / r["kine_dQdx"]
                      for r in rows if r["kine_dQdx"] > 0)
        if frac:
            s.append("  as a fraction of kine_dQdx: median %.3f  p90 %.3f  max %.3f"
                     % (frac[len(frac) // 2], frac[int(0.9 * (len(frac) - 1))], frac[-1]))
        # CLOSURE: the offline PowerBox dE/dx integral must reproduce the
        # shipped kine_dQdx, else the "exclude the delta rays" number below
        # is not measuring what it claims.
        cl = [abs(r["dqdx_all_MeV"] / r["kine_dQdx"] - 1.0)
              for r in rows if r["kine_dQdx"] > 0]
        if cl:
            s.append("  CLOSURE max |dqdx_all/kine_dQdx - 1| = %.2e over %d showers"
                     % (max(cl), len(cl)))
        # The range inversion re-forwards through the SAME table, so this only
        # shows the interpolation is invertible -- NOT independent evidence.
        rc = [abs(interp(xs, ys, r["L_chain_cm"]) - r["kine_range"])
              for r in rows if r["kine_range"] > 0]
        if rc:
            s.append("  (weak) max |range(L_chain) - kine_range| = %.2e MeV" % max(rc))
        # The real precondition: cal_kine_range used the MUON table only if the
        # shower's START SEGMENT is pdg 13 (PRShower.cxx:1745/:1781).
        nb = [r for r in rows if r["start_pdg"] != 13]
        s.append("  PRECONDITION start-segment pdg == 13: %d / %d"
                 % (len(rows) - len(nb), len(rows)))
        for r in nb:
            s.append("    L_chain MEANINGLESS (wrong range table): %s evt %d shower %d"
                     " start_pdg=%d kine_range=%.4g"
                     % (r["arm"], r["evt"], r["shower"], r["start_pdg"], r["kine_range"]))
    s.append("")
    s.append("gate outcome of the main-vertex seeds (stub_bridge ON):")
    for k, v in gate.most_common():
        s.append("  %-12s %6d" % (k, v))
    for sb in (True, False):
        s.append("walk stop reason (stub_bridge %s):" % ("ON" if sb else "OFF"))
        for k, v in stopc[sb].most_common():
            s.append("  %-12s %6d" % (k, v))
    if a.mcs_arm:
        s.append("")
        s.append("MCS sentinel events: %d (arm %s)" % (len(mcs), a.mcs_arm))
        src = collections.Counter(r["source"] for v in mcs.values() for r in v)
        nsg = collections.Counter(r["nseg"] for v in mcs.values() for r in v)
        s.append("  muon_source: %s" % dict(src))
        s.append("  nseg (segments handed to MCS): %s" % dict(nsg))
    s.append("")
    s.append("case list d84-broken-muons.tsv: %d rows" % len(cases))
    dc = collections.Counter(r["defect"] for r in cases)
    for k, v in dc.most_common():
        s.append("  %-16s %4d" % (k, v))
    s.append("  with an MCS sentinel: %d" % sum(1 for r in cases if r["mcs_nseg"] > 0))
    s.append("  internal junction with a non-muon arm (n_deg3_internal > 0): %d"
             % sum(1 for r in cases if r["n_deg3_internal"] > 0))
    if mcs_pr:
        v = sorted(r["pr_over_fragment"] for r in mcs_pr if r["pr_over_fragment"] > 0)
        s.append("")
        s.append("Part 4b, within the MCS arm (%d events):" % len(v))
        s.append("  PR muon KE / ke_range_toolkit(selected segment):"
                 " median %.3f  p90 %.3f" % (v[len(v) // 2], v[int(0.9 * (len(v) - 1))]))
        s.append("  the selected segment is short of the PR muon by >= 20%%: %d / %d"
                 % (sum(1 for x in v if x > 1.2), len(v)))
        km = sorted(r["ke_MCS"] / r["ke_range_toolkit"] for r in mcs_pr
                    if r["ke_range_toolkit"] > 20)
        s.append("  ke_MCS / ke_range_toolkit: median %.3f ; > 1.5 in %d / %d"
                 % (km[len(km) // 2], sum(1 for x in km if x > 1.5), len(km)))
    txt = "\n".join(s)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt + "\n")
    print(txt)

    if rows:
        try:
            make_plots(a.out, rows)
        except Exception as exc:                       # noqa: BLE001
            print("plots skipped: %s" % exc, file=sys.stderr)
    return 0


def make_plots(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    f = [r["dQ_frac_oth"] for r in rows]
    ax[0].hist(f, bins=40, range=(0, 0.5), color="#4477aa")
    ax[0].set_xlabel("non-muon member charge / total pseudo-shower charge")
    ax[0].set_ylabel("muon-typed pseudo-showers")
    ax[0].set_title("delta-ray charge entering kine_dQdx")

    u = [r["L_unchained_cm"] for r in rows]
    ax[1].hist(u, bins=40, range=(0, 200), color="#cc6677")
    ax[1].set_xlabel("L(muon-typed members) - L(chain)  [cm]")
    ax[1].set_title("muon length outside the chain")
    ax[1].set_yscale("log")

    xr = [r["kine_range"] for r in rows if r["kine_range"] > 0]
    yr = [r["kine_dQdx"] for r in rows if r["kine_range"] > 0]
    ax[2].scatter(xr, yr, s=8, alpha=0.6, color="#117733")
    lim = max(max(xr or [1]), max(yr or [1])) * 1.05
    ax[2].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[2].plot([0, lim], [0, lim * (1 - RATIO_LO)], color="grey", lw=0.6)
    ax[2].plot([0, lim], [0, lim * (1 + RATIO_HI)], color="grey", lw=0.6)
    ax[2].set_xlabel("kine_range [MeV]  (chain length)")
    ax[2].set_ylabel("kine_dQdx [MeV]  (all members)")
    ax[2].set_title("mode 2 fallback window [0.7, 1.5]")
    ax[2].set_xlim(0, lim)
    ax[2].set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "d84_longmu_overview.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

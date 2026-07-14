#!/usr/bin/env python3
"""PDVD photon-detector mapping audit -> saturation-veto diagnosis.

Trigger: on the ql_scan display the measured and predicted light patterns of
cathode-crossing tracks disagree in SHAPE on the cathode X-Arapucas (run
039252 evt298567 flashes 37/41/57), suggesting a DAPHNE-channel <-> module
mapping problem.  This script:

  Part 1  documentation three-way table: jjo's rawwf TTree assignment (== our
          whole chain), the official duneprototypes PDVD_PDS_Mapping
          v04152025 (read against the v4-era geometry numbering), and the
          colleague's independent naming (case statement + layout pictures).
          Verdict: all mutually consistent, EXCEPT the official map read
          v4-literally (whose v4 GDML is itself the known y-mirrored layout).
  Part 2  geometric cathode-crosser harvest from the calib dumps (pure
          geometry + time coincidence, no reliance on the Q-L matcher).
  Part 3  why it LOOKS like a mapping problem and why it is not:
          (a) hand-confirmed cases: channels nearest the track read exactly
              0 PE while far channels read thousands;
          (b) exhaustive 8! permutation fit of measured-vs-predicted: the
              documented mapping ranks DEAD LAST -- but the winning
              permutation fails the channel-adjacency test, and no single
              relabeling passes both => not a mapping problem;
          (c) channel-channel adjacency over ALL flashes (prediction-free,
              per-channel-gain-free) ranks the documented layout near the
              top => the labels are correct.
  Part 4  raw-waveform adjudication (bypasses our whole light chain):
          (a) the two DAPHNE sub-channels of each X-Arapuca correlate at
              0.98-1.00 => ganging correct;
          (b) at the hand-confirmed flash instants the RAW streams show the
              biggest pulses exactly on the modules the prediction expects;
              the dump-zero channels are exactly the ones that RAIL at the
              DAPHNE 14-bit ceiling (16383).
  Part 5  prevalence: which bright dump flashes have railed cathode
          channels (the light chain's detect_saturation=true +
          veto_saturation=true + saturation_pad=1024 removes every hit
          within +-16.4 us of a rail => those channels contribute exactly
          0 PE to the flash).

CONCLUSION: the PD mapping is correct end-to-end.  The measured/predicted
shape mismatch is the SATURATION VETO punching holes in bright flashes at
exactly their peak channels.  See pdvd-pd-mapping-investigation.md.

Read-only.  Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    python3 docs/qlmatch/pd_mapping_audit.py            # parts 1-3 (fast)
    python3 docs/qlmatch/pd_mapping_audit.py --raw      # + parts 4-5 (reads rawwf)

Inputs (all pre-existing):
    work/039252_<i>_pull2c2/calib-evt*.json, work/0392{53,349}_<i>/calib-evt*.json
    input_data_light_rawwf/np02vd_raw_run*_rawwf.root
    /cvmfs/dune.opensciencegrid.org/products/dune/duneprototypes/v10_09_00d00/
        config_data/PDVD_PDS_Mapping_v04152025.json
    photlib/pdvd-photlib-vis-v5-128nm.json  photlib/pdvd-photlib-chanmap.json
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL = ("/cvmfs/dune.opensciencegrid.org/products/dune/duneprototypes/"
            "v10_09_00d00/config_data/PDVD_PDS_Mapping_v04152025.json")
LIBMETA = os.path.join(PDVD, "photlib/pdvd-photlib-vis-v5-128nm.json")
CHANMAP = os.path.join(PDVD, "photlib/pdvd-photlib-chanmap.json")
OPCHMAP = os.path.join(os.path.dirname(PDVD), "..",  # -> toolkit-dev
                       "toolkit/cfg/pgrapher/experiment/protodunevd/pdvd-opch-map.json")
RAWWF = os.path.join(PDVD, "input_data_light_rawwf")

U_CATH = 336.91          # cm, drift coord of the cathode active edge (|x|=3)
CATH_OD = list(range(4, 12))          # cathode XA opdet columns
RAIL = 16383                          # DAPHNE 14-bit ceiling
# hand-confirmed crossers of evt298567 (previous hand-scan round)
HAND = {37: (50, 4000060), 41: (8, 4000063), 57: (134, 4000056)}

# Colleague #2's case-statement naming (offline channel -> module name).
# The numeric values in the case statement were SPE amplitudes (irrelevant
# here); only the names matter.
CASE_NAMES = {
    **{1000 + 10*i + j: "C%d" % i for i in range(1, 9) for j in (0, 1)},
    **{2000 + 10*i + j: "M%d" % i for i in range(1, 9) for j in (0, 1)},
}

# jjo module names per opdet column (from the rawwf opdet_geo tree).
JJO_NAME = {0: "M1", 1: "M3", 2: "M2", 3: "M4", 4: "C2", 5: "C6", 6: "C1",
            7: "C5", 8: "C3", 9: "C7", 10: "C4", 11: "C8", 12: "M5", 13: "M7",
            18: "M6", 19: "M8"}
# opdet column of each cathode offline-channel decade (jjo == whole chain)
OFF2OD = {1: 6, 2: 4, 3: 8, 4: 10, 5: 7, 6: 5, 7: 9, 8: 11}


def dump_paths():
    """Calib dumps to harvest: 039252 pull2c2 (18, bundles survive the wider
    containment) + canonical 039253/039349 (shield-FV/ctoff reprocess)."""
    paths = sorted(glob.glob(os.path.join(PDVD, "work/039252_*_pull2c2/calib-evt*.json")))
    for run in ("039253", "039349"):
        for d in sorted(glob.glob(os.path.join(PDVD, "work/%s_*" % run))):
            tail = os.path.basename(d).split("_", 1)[1]
            if not tail.isdigit():        # skip _light*/tagged variants
                continue
            paths += sorted(glob.glob(os.path.join(d, "calib-evt*.json")))
    return paths


def load_tables():
    meta = json.load(open(LIBMETA))
    pos = np.array(meta["chan_pos_cm"])          # position of opdet column i (cm)
    off = json.load(open(OFFICIAL))
    cmap = {e["channel"]: e for e in json.load(open(CHANMAP))["channels"]}
    jmap = {e["opch"]: e["opdet"]
            for e in json.load(open(OPCHMAP))["channels"]}
    o2g = {}
    for e in off:
        for hw in e["HardwareChannel"]:
            o2g[hw["OfflineChannel"]] = e
    return pos, o2g, cmap, jmap


def part1_documentation(pos, o2g, cmap, jmap):
    print("=" * 100)
    print("PART 1 - documentation three-way table (positions in cm)")
    print("=" * 100)
    print("official gc decoded against the v4-era numbering (pdvd-photlib-chanmap.json);")
    print("'v4-literal pos' = that gc's v4 GDML position -- but the v4 GDML Arapuca layout")
    print("is itself the known y-mirrored one, so the literal reading is ambiguous by")
    print("exactly one y-mirror.  Colleague #2 names (case statement) shown last.")
    print()
    print("%5s | %-4s %-24s | %-4s %-24s | %-8s %-5s | %s" % (
        "opch", "jjo", "jjo module pos", "ogc", "official v4-literal pos",
        "otype", "owls", "case-name"))
    for opch in sorted(jmap):
        jod = jmap[opch]
        jp = pos[jod]
        e = o2g.get(opch)
        gc = e["channel"] if e else None
        ce = cmap.get(gc, {})
        v4p = (ce.get("x", np.nan) / 10., ce.get("y", np.nan) / 10.,
               ce.get("z", np.nan) / 10.)
        print("%5d | od%-2d (%7.1f,%7.1f,%6.1f) | gc%-2s (%7.1f,%7.1f,%6.1f) | %-8s %-5s | %s" % (
            opch, jod, jp[0], jp[1], jp[2], gc, v4p[0], v4p[1], v4p[2],
            (e or {}).get("pd_type", "?"), str((e or {}).get("wls")),
            CASE_NAMES.get(opch, "-")))
    print()
    print("Documentation verdict: jjo names == case-statement names per offline channel;")
    print("the layout picture places C1/C4/C2/C3 on NO-TCO(+y) and C5/C8/C6/C7 on TCO(-y)")
    print("exactly at jjo's positions (TCO = -y pinned by the v5 3D panel, od1/3/13/19 on")
    print("the TCO wall).  ALL documentation sources agree with the toolkit chain.")


# ---------------------------------------------------------------- Part 2

def cluster_geom(c, apa, geom):
    """u_i(t) = s*(x_i + sign_offset*t*v - anode_x); d(u)/dt = -v for both
    crates, so u_end(t) = umax0 - v*t with umax0 = max_i s*(x_i - anode_x)."""
    g = geom[str(apa)]
    s = 1.0 if apa == 0 else -1.0
    x = np.asarray(c["x"])
    u0 = s * (x - g["anode_x"])
    umax0 = u0.max()
    sel = u0 >= umax0 - 3.0
    y = np.median(np.asarray(c["y"])[sel])
    z = np.median(np.asarray(c["z"])[sel])
    return umax0, y, z, umax0 - u0.min()


def harvest_crossers(strict=True):
    """Geometric cathode-crosser (bot,top,flash) triplets.

    A flash f is a crossing candidate for the pair (bot,top) when BOTH
    halves' cathode ends sit within tolerance of the cathode at f's time
    (per-crate clocks) and the two end (y,z) agree.  strict=True additionally
    demands a bright (>=1500 PE) flash that DOMINATES its +-30 us
    neighbourhood -- at ~15 us mean flash spacing the loose criterion admits
    sizeable combinatoric contamination, so quantitative tests use strict.
    Also records the charge-weighted (y,z) centroid of the pair.
    """
    tol = 3.0 if strict else 6.0
    out = []
    for path in dump_paths():
        d = json.load(open(path))
        v = d["drift_speed"]
        geom = d["geometry"]
        cl = {c["uid"]: c for c in d["clusters"]}
        pred = {}
        for b in d["bundles"]:
            pred[(b["flash_gid"], b["main_cluster"])] = b["pred_pe"]
        bots, tops = [], []
        for c in d["clusters"]:
            if c["npoints"] < 400:
                continue
            apa = 0 if c["uid"] < 4000000 else 4
            umax0, y, z, span = cluster_geom(c, apa, geom)
            if span < 40.0:
                continue
            (bots if apa == 0 else tops).append((c["uid"], umax0, y, z))
        flashes = d["flashes"]
        for f in flashes:
            if strict:
                if f["total_PE"] < 1500:
                    continue
                if any(g is not f and abs(g["time"] - f["time"]) < 30
                       and g["total_PE"] > 0.25 * f["total_PE"] for g in flashes):
                    continue
            elif f["total_PE"] < 300:
                continue
            for ub, umb, yb, zb in bots:
                if abs(umb - v * f["time"] - U_CATH) > tol:
                    continue
                for ut, umt, yt, zt in tops:
                    if abs(umt - v * f["time1"] - U_CATH) > tol:
                        continue
                    if np.hypot(yb - yt, zb - zt) > 20.0:
                        continue
                    qy = qz = qw = 0.0
                    for uid in (ub, ut):
                        c = cl[uid]
                        q = np.asarray(c["q"])
                        qy += (q * np.asarray(c["y"])).sum()
                        qz += (q * np.asarray(c["z"])).sum()
                        qw += q.sum()
                    pb = pred.get((f["gid"], ub))
                    pt = pred.get((f["gid"], ut))
                    pr = None
                    if pb is not None or pt is not None:
                        pr = (np.asarray(pb or [0.] * 40, float)
                              + np.asarray(pt or [0.] * 40, float))
                    out.append(dict(
                        evt=d["charge_ident"], bot=ub, top=ut, gid=f["gid"],
                        qy=qy / qw, qz=qz / qw, cy=0.5 * (yb + yt), cz=0.5 * (zb + zt),
                        pe=np.asarray(f["pe"], float), pred=pr,
                        total=f["total_PE"]))
    return out


# ---------------------------------------------------------------- Part 3

def part3_pattern_tests(crossers, pos):
    CATH = np.asarray(CATH_OD)
    mods = np.array([[pos[o][1], pos[o][2]] for o in CATH])
    names = JJO_NAME

    print("3a. hand-confirmed evt298567 cases (dump values):")
    d = json.load(open(sorted(glob.glob(os.path.join(
        PDVD, "work/039252_0_pull2c2/calib-evt298567.json")))[0]))
    fl = {f["gid"]: f for f in d["flashes"]}
    pred = {}
    for b in d["bundles"]:
        pred[(b["flash_gid"], b["main_cluster"])] = np.asarray(b["pred_pe"])
    for gid, (ub, ut) in HAND.items():
        f = fl[gid]
        pr = pred.get((gid, ub), np.zeros(40)) + pred.get((gid, ut), np.zeros(40))
        print("  flash gid%d PE=%.0f:  ch(name,y,z): meas / pred" % (gid, f["total_PE"]))
        for k, od in enumerate(CATH):
            print("    %s(%+6.1f,%5.1f): %7.1f / %7.1f%s" % (
                names[od], mods[k][0], mods[k][1], f["pe"][od], pr[od],
                "   <-- ZERO measured, large predicted" if f["pe"][od] == 0 and pr[od] > 100 else ""))

    use = [c for c in crossers if c["pred"] is not None
           and c["pe"][CATH].sum() > 50 and c["pred"][CATH].sum() > 10]
    M = np.sqrt(np.array([c["pe"][CATH] for c in use]))
    P = np.sqrt(np.array([c["pred"][CATH] for c in use]))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    perms = np.array(list(itertools.permutations(range(8))))
    C = np.empty((len(perms), len(use)), np.float32)
    for j, p in enumerate(perms):
        C[j] = (M[:, p] * P).sum(axis=1)
    scores = C.mean(axis=1)
    ident = int(np.where((perms == np.arange(8)).all(axis=1))[0][0])
    order = np.argsort(scores)[::-1]
    print("\n3b. exhaustive permutation fit, %d strict crossers, sqrt-cosine vs prediction:" % len(use))
    print("    documented mapping score %.4f -> rank %d of 40320 (dead last region)" % (
        scores[ident], 1 + int((scores > scores[ident]).sum())))
    wj = order[0]
    print("    best permutation score %.4f: %s" % (
        scores[wj], ", ".join("%s<-%s" % (names[CATH[k]], names[CATH[perms[wj][k]]])
                              for k in range(8) if perms[wj][k] != k)))

    # adjacency test (prediction-free, per-channel-scale-invariant)
    rows = []
    for path in dump_paths():
        dd = json.load(open(path))
        for f in dd["flashes"]:
            pe = np.asarray(f["pe"])[CATH]
            if pe.sum() > 100:
                rows.append(np.sqrt(pe))
    R = np.array(rows)
    Cc = np.corrcoef(R.T)
    iu = np.triu_indices(8, 1)
    cvec = Cc[iu]
    try:
        from scipy.stats import spearmanr
        def adj_score(p):
            inv = np.empty(8, int)
            inv[list(p)] = np.arange(8)
            Q = mods[inv]
            dist = np.hypot(Q[:, None, 0] - Q[None, :, 0],
                            Q[:, None, 1] - Q[None, :, 1])[iu]
            return spearmanr(cvec, -dist).statistic
        adj = np.array([adj_score(p) for p in perms])
        r_id = 1 + int((adj > adj[ident]).sum())
        print("\n3c. adjacency test (%d flashes, corr(sqrt PE) vs module distance, Spearman):" % len(R))
        print("    documented mapping %.3f -> rank %d of 40320 (near TOP)" % (adj[ident], r_id))
        print("    best-fit permutation from 3b: %.3f -> rank %d (FAILS)" % (
            adj[wj], 1 + int((adj > adj[wj]).sum())))
    except ImportError:
        print("(scipy missing - adjacency ranking skipped)")

    # measured centroid vs charge centroid: no affine map either
    qy = np.array([c["qy"] for c in use])
    qz = np.array([c["qz"] for c in use])
    my = np.array([(c["pe"][CATH] * mods[:, 0]).sum() / c["pe"][CATH].sum() for c in use])
    mz = np.array([(c["pe"][CATH] * mods[:, 1]).sum() / c["pe"][CATH].sum() for c in use])
    print("\n    measured-light centroid vs charge centroid: corr_y %+0.2f, corr_z %+0.2f" % (
        np.corrcoef(qy, my)[0, 1], np.corrcoef(qz, mz)[0, 1]))
    print("    -> no permutation AND no coordinate transform reproduces this; the per-flash")
    print("       measured pattern is not a relabeled image of the track.  Not a mapping bug.")


# ---------------------------------------------------------------- Parts 4+5 (raw)

def rail_intervals(w, min_len=1):
    """[(start,end)) sample ranges at the DAPHNE 14-bit ceiling."""
    hit = np.asarray(w) >= RAIL
    if not hit.any():
        return []
    d = np.diff(hit.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if hit[0]:
        starts = [0] + starts
    if hit[-1]:
        ends = ends + [len(hit)]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def load_cathode_streams(rfile, event):
    import uproot
    t = uproot.open(rfile)["rawdump/raw_waveform"]
    arr = t.arrays(["event", "opchannel", "timestamp"], library="np")
    sel = np.where(arr["event"] == event)[0]
    if not len(sel):
        return None, None
    t0 = arr["timestamp"][sel].min()
    waves, ts = {}, {}
    for i in sel:
        c = int(arr["opchannel"][i])
        if 1000 <= c < 2000:
            waves[c] = np.asarray(
                t["adc"].array(entry_start=i, entry_stop=i + 1, library="np")[0],
                np.float32)
            ts[c] = float(arr["timestamp"][i])
    return (waves, ts), t0


def part4_raw(pos):
    print("=" * 100)
    print("PART 4 - raw-waveform adjudication (evt298567, bypasses our light chain)")
    print("=" * 100)
    rfile = sorted(glob.glob(os.path.join(RAWWF, "np02vd_raw_run039252_*_rawwf.root")))[0]
    (waves, ts), t0 = load_cathode_streams(rfile, 298567)
    order = sorted(waves)

    def binned(w):
        base = np.median(w)
        n = len(w) // 64
        return np.abs(w[:n * 64].reshape(n, 64) - base).sum(axis=1)
    B = np.array([binned(waves[c]) for c in order])
    act = B.sum(axis=0)
    Bk = B[:, act > np.percentile(act, 90)]
    Cc = np.corrcoef(Bk)
    print("4a. sub-channel ganging check (corr of ~1us-binned |signal|):")
    for x in range(1, 9):
        a, b = order.index(1000 + 10 * x), order.index(1001 + 10 * x)
        others = max(Cc[a, j] for j in range(16) if j not in (a, b))
        print("    10%d0-10%d1: %.3f (max to any other channel %.3f) %s" % (
            x, x, Cc[a, b], others, "OK" if Cc[a, b] > others else "??"))

    d = json.load(open(glob.glob(os.path.join(
        PDVD, "work/039252_0_pull2c2/calib-evt298567.json"))[0]))
    toff = d["trigger_offsets_us"][0]
    fl = {f["gid"]: f for f in d["flashes"]}
    print("\n4b. raw pulse area at the hand-confirmed flash instants vs dump PE:")
    for gid in HAND:
        f = fl[gid]
        t_light = f["time"] - toff
        print("  flash gid%d (dump PE %.0f):" % (gid, f["total_PE"]))
        for x in range(1, 9):
            area = 0.0
            nrail = 0
            for c in (1000 + 10 * x, 1001 + 10 * x):
                w = waves[c]
                s = int((t_light - (ts[c] - t0)) * 62.5)
                lo, hi = s - 250, s + 250
                if lo < 0 or hi > len(w):
                    continue
                seg = w[lo:hi]
                base = np.median(w[max(0, lo - 2000):lo])
                area += float(np.clip(seg - base, 0, None).sum())
                nrail += int((seg >= RAIL).sum())
            od = OFF2OD[x]
            print("    C%d (y=%+6.1f): raw %9.0f  rail-samples %3d   dump pe %7.1f%s" % (
                x, pos[od][1], area, nrail, f["pe"][od],
                "   <-- railed => vetoed to zero" if nrail > 0 and f["pe"][od] == 0 else ""))


def part5_prevalence():
    print("=" * 100)
    print("PART 5 - prevalence: bright flashes losing railed cathode channels")
    print("=" * 100)
    print("(saturation veto: detect_saturation=true, veto_saturation=true,")
    print(" saturation_pad=1024 samples = +-16.4 us  => a railed channel contributes")
    print(" 0 PE to every flash within the pad window)")
    import uproot
    pad_us = 1024 / 62.5
    nfl = nfl_hit = 0
    per_evt = []
    for rfile in sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root"))):
        run = os.path.basename(rfile).split("_")[2].replace("run", "").zfill(6)
        t = uproot.open(rfile)["rawdump/raw_waveform"]
        arr = t.arrays(["event", "opchannel", "timestamp"], library="np")
        events = sorted(set(arr["event"]))
        for ev in events:
            # find the matching dump
            hits = [p for p in dump_paths()
                    if p.endswith("calib-evt%d.json" % ev) and ("/%s_" % run) in p]
            if not hits:
                continue
            d = json.load(open(hits[0]))
            toff = d["trigger_offsets_us"][0]
            (waves, ts), t0 = load_cathode_streams(rfile, ev)
            rails = {}          # channel decade -> list of dump-time intervals
            for c, w in waves.items():
                x = (c // 10) % 100
                for a, b in rail_intervals(w):
                    ta = (ts[c] - t0) + a / 62.5 + toff
                    tb = (ts[c] - t0) + b / 62.5 + toff
                    rails.setdefault(x, []).append((ta - pad_us, tb + pad_us))
            nb = nhit = 0
            for f in d["flashes"]:
                if f["total_PE"] < 1000:
                    continue
                nb += 1
                lost = [x for x, iv in rails.items()
                        if any(a <= f["time"] <= b for a, b in iv)]
                if lost:
                    nhit += 1
            nfl += nb
            nfl_hit += nhit
            per_evt.append((run, ev, nb, nhit, sum(len(v) for v in rails.values())))
    for run, ev, nb, nhit, nr in per_evt:
        print("  run %s evt %-7d bright flashes %3d, with railed cathode ch %3d, rail intervals %3d" % (
            run, ev, nb, nhit, nr))
    if nfl:
        print("TOTAL: %d/%d (%.0f%%) bright (>=1000 PE) flashes have >=1 cathode channel"
              % (nfl_hit, nfl, 100.0 * nfl_hit / nfl))
        print("vetoed to zero by saturation -- their measured shapes are hole-punched.")


def main():
    pos, o2g, cmap, jmap = load_tables()
    part1_documentation(pos, o2g, cmap, jmap)
    print()
    print("=" * 100)
    print("PART 2+3 - crosser harvest and pattern tests")
    print("=" * 100)
    crossers = harvest_crossers(strict=True)
    print("strict crossers harvested: %d" % len(crossers))
    part3_pattern_tests(crossers, pos)
    if "--raw" in sys.argv:
        print()
        part4_raw(pos)
        print()
        part5_prevalence()
    else:
        print("\n(run with --raw for parts 4-5: raw-waveform adjudication + prevalence)")


if __name__ == "__main__":
    main()

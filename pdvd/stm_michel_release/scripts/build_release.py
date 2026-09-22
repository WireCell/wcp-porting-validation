#!/usr/bin/env python3
"""build_release.py -- assemble the colleague-facing STM + Michel release from the
production outputs (owner tool; colleagues do not need to run it).

    python3 build_release.py --det pdhd [--arm d116hflip] [--out ../] [--events 029107_19,...]
    python3 build_release.py --det pdvd [--arm d116vflip] [--out ../]

Per event it reads
    <work>/<run6>_<idx>_<arm>/tracking-pr.root    T_stm_michel, T_stm_michel_pts, T_stm_michel_2d,
                                                  T_cluster, T_rec_charge, T_proj_data, Trun
    <work>/<run6>_<idx>_<arm>/mabc-pr.zip         the Bee event display
    <work>/<run6>_<idx>_<arm>/calib-pr-evt<id>.json   dqdx_ref (expectation tables), meta
    <work>/<run6>_<idx>_<arm>/pctree-evt<id>.tlas  trigger offsets, wires file
    PDHD <work>/<run6>_allpd<id>/opflash_pdhd-allpd-wct.tar.gz      the flashes Q/L matching used
         <work>/<run6>_allpd<id>_wf/light-frames-allpd-{snip,fs}-wct.tar.bz2   raw/decon frames
    PDVD <work>/<run6>_light<id>_keep/opflash_pdvd-wct.tar.gz
         <work>/<run6>_light<id>_wf/light-frames-{cath,mem,pmt}-wct.tar.bz2
and writes
    <out>/events/<run6>_<id>/stm_michel_<det>_<run6>_<id>.root   (TTrees, see ROOTFILE.md)
    <out>/events/<run6>_<id>/bee_<det>_<run6>_<id>.zip
    <out>/index.csv, <out>/dqdx_ref.json, <out>/build_log.txt

Every number is copied from the production products; nothing is recomputed except the
convenience columns documented in ROOTFILE.md (dqdx on T_stm_fit, plane / chan_rank on
T_stm_2d, the flash join columns on T_stm, the waveform windows).  Self-checks per event:
  * every candidate's t0_us equals the matched flash time (T_cluster.flash_id -> opflash row)
  * every ophit of an exported flash lies inside its waveform window
  * the decon peak of the brightest PD of each exported flash sits within 1 us of the flash time
  * the region Michel estimator re-derived from T_michel_2d reproduces michel_q2d_region_{u,v,w}
  * row counts equal the source trees
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

import awkward as ak
import numpy as np
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
TOOLKIT = "/nfs/data/1/xqian/toolkit-dev/toolkit"
WCDATA = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data"

DET = {
    "pdhd": dict(
        arm="d116hflip",
        light_dir=lambda run6, evid: f"{IMG}/pdhd/work/{run6}_allpd{evid}",
        frames_dir=lambda run6, evid, suf: f"{IMG}/pdhd/work/{run6}_allpd{evid}{suf}",
        opflash="opflash_pdhd-allpd-wct.tar.gz",
        frames=["light-frames-allpd-snip-wct.tar.bz2", "light-frames-allpd-fs-wct.tar.bz2"],
        nchan=160,
        opch_map=None,  # readout channel == OpDet
    ),
    "pdvd": dict(
        arm="d116vflip",
        light_dir=lambda run6, evid: f"{IMG}/pdvd/work/{run6}_light{evid}_keep",
        frames_dir=lambda run6, evid, suf: f"{IMG}/pdvd/work/{run6}_light{evid}{suf}",
        opflash="opflash_pdvd-wct.tar.gz",
        frames=["light-frames-cath-wct.tar.bz2", "light-frames-mem-wct.tar.bz2", "light-frames-pmt-wct.tar.bz2"],
        nchan=40,
        opch_map=f"{TOOLKIT}/cfg/pgrapher/experiment/protodunevd/pdvd-opch-map.json",
    ),
}
REGION_CM = 10.0   # michel_q2d_region_cm in both productions (doc pdvd/95, pdhd/26)
# a production file WITH candidates, used only for the branch schema of the empty trees
# written on events that have no candidate (so every file has the same branches)
SCHEMA_REF = {"pdhd": f"{IMG}/pdhd/work/029107_19_d116hflip/tracking-pr.root",
              "pdvd": f"{IMG}/pdvd/work/039252_0_d116vflip/tracking-pr.root"}


def ref_schema(det):
    """{tree: {branch: numpy dtype}} of the source trees in the reference file."""
    u = uproot.open(SCHEMA_REF[det]); out = {}
    for t in ["T_stm_michel", "T_stm_michel_pts", "T_stm_michel_2d", "T_rec_charge"]:
        a = u[t].arrays(library="np", entry_stop=1)
        out[t] = {k: v.dtype for k, v in a.items()}
    return out


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n"); fh.flush()


def load_tar(path):
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            b = t.extractfile(m).read()
            out[m.name] = json.loads(b) if m.name.endswith(".json") else np.load(io.BytesIO(b))
    return out


def read_tlas(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short=8", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- channel-rank scheme
class ChanScheme:
    """The per-plane channel-rank coordinate of T_proj_data / T_rec_charge pu,pv,pw
    (toolkit root/src/PdvdPrMagnifyTrackingVisitor.cxx chan_scheme): global = base[plane] +
    rank of the wire's channel id among all channel ids of that plane over the whole detector."""

    def __init__(self, wires_file):
        from wirecell.util.wires import persist
        st = persist.load(wires_file)
        chans = [set(), set(), set()]
        seg = {}  # (plane, channel) -> list of (apa, face, wire_index)
        for a in st.anodes:
            for fi, f in enumerate(a.faces):
                face = st.faces[f]
                for pi, p in enumerate(face.planes):
                    if pi > 2:
                        continue
                    plane = st.planes[p]
                    for wi, w in enumerate(plane.wires):
                        wire = st.wires[w]
                        chans[pi].add(wire.channel)
                        seg.setdefault((pi, wire.channel), []).append((a.ident, fi, wi))
        self.nch = [len(c) for c in chans]
        self.base = [0, self.nch[0], self.nch[0] + self.nch[1]]
        self.rank2ch = [np.array(sorted(c), int) for c in chans]
        self.seg = seg

    def decode(self, glob):
        """global rank -> (plane, chan_rank, channel, apa, face, wire) (first segment of a wrapped channel)."""
        glob = np.asarray(glob, int)
        plane = np.where(glob >= self.base[2], 2, np.where(glob >= self.base[1], 1, 0))
        rank = glob - np.asarray(self.base)[plane]
        channel = np.empty(len(glob), int); apa = np.empty(len(glob), int); face = np.empty(len(glob), int); wire = np.empty(len(glob), int)
        for i, (p, r) in enumerate(zip(plane, rank)):
            ch = int(self.rank2ch[p][r]); channel[i] = ch
            a, f, w = self.seg[(p, ch)][0]
            apa[i] = a; face[i] = f; wire[i] = w
        return plane, rank, channel, apa, face, wire


# ---------------------------------------------------------------- per-event build
def build_event(det, cfg, arm_dir, out_dir, opts, scheme_cache, logfh):
    t_start = time.time()
    run6, idx = os.path.basename(arm_dir).split("_")[:2]
    idx = int(idx)
    pct = glob.glob(f"{arm_dir}/pctree-evt*.tlas")
    if not pct:
        raise RuntimeError(f"no pctree .tlas in {arm_dir}")
    evid = int(re.search(r"pctree-evt(\d+)\.tlas", pct[0]).group(1))
    tlas = read_tlas(pct[0])
    tag = f"{run6}_{evid}"
    ev_out = f"{out_dir}/events/{tag}"
    os.makedirs(ev_out, exist_ok=True)
    root_out = f"{ev_out}/stm_michel_{det}_{tag}.root"
    bee_out = f"{ev_out}/bee_{det}_{tag}.zip"
    checks = {}

    # ---- charge side: tracking-pr.root
    tp = uproot.open(f"{arm_dir}/tracking-pr.root")
    trun = tp["Trun"].arrays(library="np")
    scale, offset = float(trun["dQdx_scale"][0]), float(trun["dQdx_offset"][0])
    run, subrun = int(trun["runNo"][0]), int(trun["subRunNo"][0])
    assert int(trun["eventNo"][0]) == evid, (trun["eventNo"], evid)
    have_cand = "T_stm_michel" in tp
    S = tp["T_stm_michel"].arrays(library="np") if have_cand else {}
    P = tp["T_stm_michel_pts"].arrays(library="np") if "T_stm_michel_pts" in tp else {}
    M2 = tp["T_stm_michel_2d"].arrays(library="np") if "T_stm_michel_2d" in tp else {}
    C = tp["T_cluster"].arrays(library="np")
    # an event with no candidate at all has neither T_rec_charge nor T_proj_data
    R = tp["T_rec_charge"].arrays(library="np") if "T_rec_charge" in tp else {"cluster_id": np.zeros(0, np.int32)}
    PD = tp["T_proj_data"].arrays(library="np") if "T_proj_data" in tp else None
    n_cand = len(S["cluster_id"]) if have_cand else 0
    cand_ids = [int(c) for c in S["cluster_id"]] if have_cand else []
    c_index = {int(c): i for i, c in enumerate(C["cluster_id"])}

    # calib-pr: dqdx_ref + meta (nticks_per_slice)
    cal_files = glob.glob(f"{arm_dir}/calib-pr-evt{evid}.json")
    if cal_files:
        calib = json.load(open(cal_files[0]))
        meta = calib["meta"]
        nt_slice = sorted(set(x["nticks_per_slice"] for x in meta["nticks_per_slice"]))
        assert len(nt_slice) == 1, nt_slice
        nt_slice = int(nt_slice[0])
        dqdx_ref = calib["dqdx_ref"]
    else:
        # an event with no candidate writes no calib-pr dump; the slice width is the
        # detector constant (4 ticks on PDHD and PDVD) and the expectation table comes
        # from the other events (identical in every event, asserted in main()).
        nt_slice = 4
        dqdx_ref = None

    # ---- light side: production opflash archive + the _wf frames
    ldir = cfg["light_dir"](run6, evid)
    op = load_tar(f"{ldir}/{cfg['opflash']}")
    opmeta = op[f"opflash_tensorset_{evid}_metadata.json"]
    assert int(opmeta["event"]) == evid
    tens = {}
    for k, v in op.items():
        m = re.match(rf"opflash_tensor_{evid}_(\d+)_metadata\.json", k)
        if m:
            tens[v["name"]] = op[f"opflash_tensor_{evid}_{m.group(1)}_array.npy"]
    FL, FS, OH = tens["opflash"], tens["flash_summary"], tens.get("ophits", np.zeros((0, 9)))
    nflash = FL.shape[0]
    assert FL.shape[1] == 1 + cfg["nchan"], FL.shape
    opch2det = None
    if cfg["opch_map"]:
        opch2det = {c["opch"]: c["opdet"] for c in json.load(open(cfg["opch_map"]))["channels"]}
    # readout channel -> OpDet; -1 for a channel the flash finder does not gang
    # (e.g. PDVD DAPHNE 2031 in run 039253: recorded, but not in pdvd-opch-map.json)
    od_of = (lambda ch: opch2det.get(int(ch), -1)) if opch2det else (lambda ch: int(ch))
    n_unmapped = 0

    # flash join: T_cluster.flash_id is the opflash row index
    fid_of = {int(c): int(f) for c, f in zip(C["cluster_id"], C["flash_id"])}
    ftime_of = {int(c): float(t) for c, t in zip(C["cluster_id"], C["flash_time_us"])}
    matched = {}  # flash_id -> [cluster ids]
    for c, f in fid_of.items():
        if f >= 0:
            matched.setdefault(f, []).append(c)
    stm_flash = {}  # flash_id -> [candidate cluster ids]
    dt_max = 0.0
    for c in cand_ids:
        f = fid_of.get(c, -1)
        if f < 0:
            continue
        stm_flash.setdefault(f, []).append(c)
        t0 = float(S["t0_us"][cand_ids.index(c)])
        dt_max = max(dt_max, abs(t0 - FL[f, 0] / 1000.0), abs(t0 - ftime_of[c]))
    checks["t0_vs_flash_us_max"] = dt_max
    stm_fids = sorted(stm_flash)

    # ---- frames (for the waveform windows) -- only needed when there is a flash to export
    frames = []
    frames_dir = cfg["frames_dir"](run6, evid, opts.wf_suffix)
    if stm_fids:
        for fn in cfg["frames"]:
            fr = load_tar(f"{frames_dir}/{fn}")
            tags = [m.group(1) for k in fr for m in [re.match(rf"frame_(\w+)_{evid}\.npy", k)] if m]
            tags = [t for t in ["raw", "decon", "decon_roi"] if t in tags]
            ti = fr[f"tickinfo_raw_{evid}.npy"]
            # each tag's dense array starts at ITS OWN first tick (tbin0): a channel recorded
            # but never deconvolved (PDVD DAPHNE 2031, run 039349) makes raw start earlier
            # than decon, so the axis is kept per tag (same frame time and tick).
            tinfo = {t: fr[f"tickinfo_{t}_{evid}.npy"] for t in tags}
            for t in tags:
                assert tinfo[t][0] == ti[0] and tinfo[t][1] == ti[1], (fn, t, tinfo[t], ti)
            frames.append(dict(name=fn, tags=tags, t0_ns=float(ti[0]), tick_ns=float(ti[1]),
                               tbin0={t: int(tinfo[t][2]) for t in tags},
                               chans={t: [int(c) for c in fr[f"channels_{t}_{evid}.npy"]] for t in tags},
                               data={t: fr[f"frame_{t}_{evid}.npy"] for t in tags}))

    # ================================================================ write
    f = uproot.recreate(root_out)

    # ---- T_event
    ev = dict(det=det, run=run, subrun=subrun, event=evid, event_index=idx, arm=os.path.basename(arm_dir).split("_", 2)[2],
              toolkit_commit=git_head(TOOLKIT), wcp_commit=git_head(IMG),
              pr_dir=arm_dir, light_dir=ldir, frames_dir=frames_dir if stm_fids else "",
              wires=tlas.get("wires", ""), dQdx_scale=scale, dQdx_offset=offset, nticks_per_slice=nt_slice,
              readout_window_ticks=int(tlas.get("readout_window_ticks", 0)),
              light_offset_us=float(opmeta.get("offset_us", 0.0)),
              trigger_offset_us=float(tlas.get("trigger_offset_us", tlas.get("trigger_offset_bot_us", 0.0))),
              trigger_offset_bot_us=float(tlas.get("trigger_offset_bot_us", tlas.get("trigger_offset_us", 0.0))),
              trigger_offset_top_us=float(tlas.get("trigger_offset_top_us", tlas.get("trigger_offset_us", 0.0))),
              opflash_offset_bot_us=float(opmeta.get("offset_bot_us", opmeta.get("offset_us", 0.0))),
              opflash_offset_top_us=float(opmeta.get("offset_top_us", opmeta.get("offset_us", 0.0))),
              nchan=cfg["nchan"], n_flash=nflash, n_candidates=n_cand,
              n_stm=int((S["is_stm"] == 1).sum()) if have_cand else 0,
              n_michel=int(((S["is_stm"] == 1) & (S["michel_found"] == 1)).sum()) if have_cand else 0,
              wf_pre_us=opts.pre, wf_post_us=opts.post, n_stm_flash=len(stm_fids))
    f.mktree("T_event", {k: ("string" if isinstance(v, str) else (np.int32 if isinstance(v, (int, np.integer)) else np.float64)) for k, v in ev.items()})
    f["T_event"].extend({k: (ak.Array([v]) if isinstance(v, str) else np.array([v])) for k, v in ev.items()})

    # ---- T_stm: every T_stm_michel branch + the join columns
    stm = {}
    if have_cand:
        stm.update({k: v for k, v in S.items()})
        stm["run"] = np.full(n_cand, run, np.int32); stm["event"] = np.full(n_cand, evid, np.int32)
        stm["flash_id"] = np.array([fid_of.get(c, -1) for c in cand_ids], np.int32)
        stm["flash_time_us"] = np.array([FL[fid_of[c], 0] / 1000.0 if fid_of.get(c, -1) >= 0 else -1e9 for c in cand_ids])
        stm["flash_total_pe"] = np.array([FS[fid_of[c], 1] if fid_of.get(c, -1) >= 0 else 0.0 for c in cand_ids])
        stm["cluster_length_cm"] = np.array([float(C["length_cm"][c_index[c]]) if c in c_index else -1.0 for c in cand_ids])
        stm["cluster_npoints"] = np.array([int(C["npoints"][c_index[c]]) if c in c_index else -1 for c in cand_ids], np.int32)
        f.mktree("T_stm", {k: v.dtype for k, v in stm.items()})
        f["T_stm"].extend(stm)
    else:
        sch = dict(scheme_cache.setdefault("schema", ref_schema(det))["T_stm_michel"])
        sch.update(run=np.int32, event=np.int32, flash_id=np.int32, flash_time_us=np.float64, flash_total_pe=np.float64,
                   cluster_length_cm=np.float64, cluster_npoints=np.int32)
        f.mktree("T_stm", sch)

    # ---- T_stm_pts
    if P:
        n = len(P["cluster_id"])
        is_stm = {c: int(S["is_stm"][i]) for i, c in enumerate(cand_ids)}
        mf = {c: int(S["michel_found"][i]) for i, c in enumerate(cand_ids)}
        pts = dict(P)
        pts["run"] = np.full(n, run, np.int32); pts["event"] = np.full(n, evid, np.int32)
        pts["is_stm"] = np.array([is_stm.get(int(c), 0) for c in P["cluster_id"]], np.int32)
        pts["michel_found"] = np.array([mf.get(int(c), 0) for c in P["cluster_id"]], np.int32)
        f.mktree("T_stm_pts", {k: v.dtype for k, v in pts.items()}); f["T_stm_pts"].extend(pts)
    else:
        sch = dict(scheme_cache.setdefault("schema", ref_schema(det))["T_stm_michel_pts"])
        sch.update(run=np.int32, event=np.int32, is_stm=np.int32, michel_found=np.int32)
        f.mktree("T_stm_pts", sch)

    # ---- T_stm_fit: T_rec_charge rows of the candidate clusters
    sel = np.isin(R["cluster_id"], cand_ids) if (cand_ids and len(R["cluster_id"])) else np.zeros(len(R["cluster_id"]), bool)
    if "q" in R:
        fit = {k: v[sel] for k, v in R.items()}
        nq = fit["nq"]
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdx = np.where(nq > 0, ((fit["q"] - offset) / scale) / nq, -1.0)
        fit["dQ"] = (fit["q"] - offset) / scale
        fit["dx"] = nq
        fit["dqdx"] = dqdx
        fit["run"] = np.full(int(sel.sum()), run, np.int32); fit["event"] = np.full(int(sel.sum()), evid, np.int32)
        f.mktree("T_stm_fit", {k: v.dtype for k, v in fit.items()})
        if sel.sum():
            f["T_stm_fit"].extend(fit)
    else:
        sch = dict(scheme_cache.setdefault("schema", ref_schema(det))["T_rec_charge"])
        sch.update(dQ=np.float64, dx=np.float64, dqdx=np.float64, run=np.int32, event=np.int32)
        f.mktree("T_stm_fit", sch)

    # ---- T_stm_2d: T_proj_data flattened for the candidate clusters
    wires = tlas.get("wires", "")
    if wires and wires not in scheme_cache:
        scheme_cache[wires] = ChanScheme(f"{WCDATA}/{wires}")
    cs = scheme_cache.get(wires)
    cols = dict(cluster_id=[], channel_rank_global=[], time_slice=[], charge=[], charge_err=[], charge_pred=[])
    pd_ids = [int(c) for c in PD["cluster_id"][0]] if PD is not None else []
    for j, c in enumerate(pd_ids):
        if c not in cand_ids:
            continue
        ch = np.asarray(PD["channel"][0][j], int); n = len(ch)
        cols["cluster_id"].append(np.full(n, c, np.int32)); cols["channel_rank_global"].append(ch)
        cols["time_slice"].append(np.asarray(PD["time_slice"][0][j], int))
        cols["charge"].append(np.asarray(PD["charge"][0][j], float)); cols["charge_err"].append(np.asarray(PD["charge_err"][0][j], float))
        cols["charge_pred"].append(np.asarray(PD["charge_pred"][0][j], float))
    if cols["cluster_id"]:
        s2 = {k: np.concatenate(v) for k, v in cols.items()}
        n = len(s2["cluster_id"])
        s2["tick"] = s2["time_slice"] * nt_slice
        if cs:
            plane, rank, channel, apa, face, wire = cs.decode(s2["channel_rank_global"])
            s2.update(plane=plane.astype(np.int32), chan_rank=rank.astype(np.int32), channel=channel.astype(np.int32),
                      apa=apa.astype(np.int32), face=face.astype(np.int32), wire=wire.astype(np.int32))
        s2["run"] = np.full(n, run, np.int32); s2["event"] = np.full(n, evid, np.int32)
        f.mktree("T_stm_2d", {k: v.dtype for k, v in s2.items()}); f["T_stm_2d"].extend(s2)
        checks["T_stm_2d_rows"] = n
    else:
        sch = dict(cluster_id=np.int32, channel_rank_global=np.int64, time_slice=np.int64, charge=np.float64, charge_err=np.float64,
                   charge_pred=np.float64, tick=np.int64, plane=np.int32, chan_rank=np.int32, channel=np.int32, apa=np.int32,
                   face=np.int32, wire=np.int32, run=np.int32, event=np.int32)
        f.mktree("T_stm_2d", sch)

    # ---- T_michel_2d
    if M2:
        n = len(M2["cluster_id"]); m2 = dict(M2)
        m2["run"] = np.full(n, run, np.int32); m2["event"] = np.full(n, evid, np.int32)
        f.mktree("T_michel_2d", {k: v.dtype for k, v in m2.items()}); f["T_michel_2d"].extend(m2)
    else:
        sch = dict(scheme_cache.setdefault("schema", ref_schema(det))["T_stm_michel_2d"])
        sch.update(run=np.int32, event=np.int32)
        f.mktree("T_michel_2d", sch)

    # ---- T_flash: every flash of the event
    fl = dict(
        run=np.full(nflash, run, np.int32), event=np.full(nflash, evid, np.int32),
        flash_id=np.arange(nflash, dtype=np.int32),
        time_us=FL[:, 0] / 1000.0,
        time_charge_bot_us=FL[:, 0] / 1000.0 + ev["trigger_offset_bot_us"],
        time_charge_top_us=FL[:, 0] / 1000.0 + ev["trigger_offset_top_us"],
        total_pe=FS[:, 1], y_center_mm=FS[:, 2], z_center_mm=FS[:, 3], y_width_mm=FS[:, 4], z_width_mm=FS[:, 5],
        nhits=FS[:, 7].astype(np.int32),
        n_matched_clusters=np.array([len(matched.get(i, [])) for i in range(nflash)], np.int32),
        is_stm_flash=np.array([1 if i in stm_flash else 0 for i in range(nflash)], np.int32),
    )
    assert np.array_equal(FS[:, 0].astype(int), np.arange(nflash)), "flash_summary flash_id != row"
    fl_j = dict(pe=ak.Array(FL[:, 1:].tolist()),
                matched_cluster_ids=ak.Array([sorted(matched.get(i, [])) for i in range(nflash)]),
                stm_cluster_ids=ak.Array([sorted(stm_flash.get(i, [])) for i in range(nflash)]))
    if "flash_sat" in tens:
        fl_j["sat"] = ak.Array(tens["flash_sat"].tolist())
    if "flash_cov" in tens:
        fl_j["cov"] = ak.Array(tens["flash_cov"].tolist())
    types = {k: v.dtype for k, v in fl.items()}
    types.update(pe="var * float64", matched_cluster_ids="var * int32", stm_cluster_ids="var * int32")
    if "sat" in fl_j: types["sat"] = "var * float64"
    if "cov" in fl_j: types["cov"] = "var * float64"
    f.mktree("T_flash", types)
    if nflash:
        d = dict(fl); d.update(fl_j)
        d["matched_cluster_ids"] = ak.values_astype(d["matched_cluster_ids"], np.int32)
        d["stm_cluster_ids"] = ak.values_astype(d["stm_cluster_ids"], np.int32)
        f["T_flash"].extend(d)

    # ---- T_ophit: hits of the STM-matched flashes
    hsel = np.isin(OH[:, 7].astype(int), stm_fids) if len(OH) else np.zeros(0, bool)
    H = OH[hsel]
    oh = dict(run=np.full(len(H), run, np.int32), event=np.full(len(H), evid, np.int32),
              flash_id=H[:, 7].astype(np.int32), channel=H[:, 0].astype(np.int32),
              opdet=np.array([od_of(c) for c in H[:, 0]], np.int32),
              peak_time_us=H[:, 1] / 1000.0, start_time_us=H[:, 6] / 1000.0, width_us=H[:, 2] / 1000.0,
              area=H[:, 3], amplitude=H[:, 4], pe=H[:, 5], fast_to_total=H[:, 8])
    f.mktree("T_ophit", {k: v.dtype for k, v in oh.items()})
    if len(H):
        f["T_ophit"].extend(oh)

    # ---- T_opwf: raw / decon windows of every readout channel for each STM-matched flash
    rows = dict(run=[], event=[], flash_id=[], cluster_id=[], channel=[], opdet=[], branch=[], t0_us=[], tick_ns=[],
                n=[], n_raw_nonzero=[], pe=[], raw=[], decon=[], decon_roi=[])
    hit_out = 0; n_hits_checked = 0; peak_bad = 0; peak_checked = 0; n_no_decon = 0
    for fid in stm_fids:
        t_fl = FL[fid, 0]
        for fr in frames:
            tick = fr["tick_ns"]
            # one common window on the time axis for every tag; sample i is at t_start + i * tick
            t_start = t_fl - opts.pre * 1000.0
            n = int(round((opts.pre + opts.post) * 1000.0 / tick))
            t0_us = t_start / 1000.0

            def cut(tag, ch):
                """the window of channel ch in tag's dense array, zero-padded where the array does not reach"""
                chs = fr["chans"][tag]
                if ch not in chs:
                    return None
                arr = fr["data"][tag][chs.index(ch)]
                i0 = int(round((t_start - fr["t0_ns"]) / tick)) - fr["tbin0"][tag]
                a, b = max(i0, 0), min(i0 + n, arr.shape[0])
                out = np.zeros(n, np.float32)
                if b > a:
                    out[a - i0:b - i0] = arr[a:b]
                return out

            for ch in fr["chans"]["raw"]:
                od = od_of(ch)
                if od < 0:
                    n_unmapped += 1
                raw = cut("raw", ch)
                dec = cut("decon", ch)
                if dec is None:   # recorded but never deconvolved (no template / not ganged): empty decon
                    dec = np.zeros(0, np.float32); n_no_decon += 1
                roi = cut("decon_roi", ch) if "decon_roi" in fr["tags"] else None
                if roi is None:
                    roi = np.zeros(0, np.float32)
                rows["run"].append(run); rows["event"].append(evid); rows["flash_id"].append(fid)
                rows["cluster_id"].append(stm_flash[fid][0]); rows["channel"].append(ch); rows["opdet"].append(od)
                rows["branch"].append(fr["name"].split("light-frames-")[1].split("-wct")[0])
                rows["t0_us"].append(t0_us); rows["tick_ns"].append(tick); rows["n"].append(len(raw))
                rows["n_raw_nonzero"].append(int(np.count_nonzero(raw))); rows["pe"].append(float(FL[fid, 1 + od]) if od >= 0 else 0.0)
                rows["raw"].append(np.asarray(raw, np.float32)); rows["decon"].append(np.asarray(dec, np.float32))
                rows["decon_roi"].append(np.asarray(roi, np.float32))
            # self-check: the brightest PD of this archive peaks within 1 us of the flash time
            pe_here = [(float(FL[fid, 1 + od_of(c)]) if od_of(c) >= 0 else 0.0, int(c)) for c in fr["chans"]["raw"]]
            pe_best, ch_best = max(pe_here)
            if pe_best > 0 and ch_best in fr["chans"]["decon"]:
                w = cut("decon", ch_best)
                t_peak = t_start + int(np.argmax(w)) * tick
                peak_checked += 1
                if abs(t_peak - t_fl) > 1000.0:
                    peak_bad += 1
                    log(f"    WARN {tag} flash {fid} {fr['name']}: brightest PD ch {ch_best} pe {pe_best:.0f} peaks {(t_peak - t_fl) / 1000:.2f} us from the flash time", logfh)
        # ophits inside the window?
        for h in OH[OH[:, 7].astype(int) == fid]:
            n_hits_checked += 1
            if not (t_fl - opts.pre * 1000.0 <= h[1] <= t_fl + opts.post * 1000.0):
                hit_out += 1
    checks["ophits_outside_window"] = (hit_out, n_hits_checked)
    checks["peak_check_bad"] = (peak_bad, peak_checked)
    checks["unmapped_channel_rows"] = n_unmapped
    checks["no_decon_rows"] = n_no_decon
    types = {k: np.int32 for k in ["run", "event", "flash_id", "cluster_id", "channel", "opdet", "n", "n_raw_nonzero"]}
    types.update(branch="string", t0_us=np.float64, tick_ns=np.float64, pe=np.float64, raw="var * float32", decon="var * float32", decon_roi="var * float32")
    f.mktree("T_opwf", types)
    if rows["run"]:
        d = {k: (np.array(v) if k not in ("branch", "raw", "decon", "decon_roi") else ak.Array(v)) for k, v in rows.items()}
        for k in ["run", "event", "flash_id", "cluster_id", "channel", "opdet", "n", "n_raw_nonzero"]:
            d[k] = d[k].astype(np.int32)
        f["T_opwf"].extend(d)
    f.close()

    # ---- Bee zip
    shutil.copyfile(f"{arm_dir}/mabc-pr.zip", bee_out)

    # ---- self-checks: region twin + row counts
    twin_ok = twin_bad = 0
    if have_cand and M2:
        for i, c in enumerate(cand_ids):
            if int(S["michel_found"][i]) != 1:
                continue
            s = (M2["cluster_id"] == c) & (M2["d_stop_cm"] >= 0) & (M2["d_stop_cm"] <= REGION_CM) & (M2["own_blob"] != 0)
            pmu = np.maximum(M2["pred_mu"][s], 0.0); xs = M2["xshared"][s] == 1
            contrib = np.where(xs, np.maximum(M2["pred_all"][s] - pmu, 0.0), M2["charge"][s] - pmu)
            pl = M2["plane"][s]
            off = [float(contrib[pl == p].sum()) for p in range(3)]
            cpp = [float(S[f"michel_q2d_region_{p}"][i]) for p in "uvw"]
            dev = max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(off, cpp))
            ncpp = [int(S[f"michel_q2d_region_n_{p}"][i]) for p in "uvw"]
            noff = [int((pl == p).sum()) for p in range(3)]
            if dev < 1e-6 and noff == ncpp:
                twin_ok += 1
            else:
                twin_bad += 1
                log(f"    WARN {tag} cluster {c}: region twin off {off} vs C++ {cpp}; n {noff} vs {ncpp}", logfh)
    checks["region_twin"] = (twin_ok, twin_bad)
    g = uproot.open(root_out)
    counts = {k.split(";")[0]: g[k].num_entries for k in g.keys()}
    assert counts["T_stm"] == n_cand
    assert counts["T_stm_pts"] == (len(P["cluster_id"]) if P else 0)
    assert counts["T_michel_2d"] == (len(M2["cluster_id"]) if M2 else 0)
    assert counts["T_flash"] == nflash
    assert counts["T_stm_fit"] == int(sel.sum())
    g.close()

    row = dict(run=run, idx=idx, evtid=evid, n_candidates=n_cand, n_stm=ev["n_stm"], n_michel=ev["n_michel"],
               n_flash=nflash, n_stm_flash=len(stm_fids), n_opwf=counts["T_opwf"],
               root=os.path.relpath(root_out, out_dir), bee=os.path.relpath(bee_out, out_dir))
    log(f"  {tag} idx {idx}: candidates {n_cand} stm {ev['n_stm']} michel {ev['n_michel']} | flashes {nflash}, stm flashes {len(stm_fids)}, "
        f"opwf rows {counts['T_opwf']} | checks {checks} | {os.path.getsize(root_out) / 1e6:.1f} MB, {time.time() - t_start:.1f} s", logfh)
    return row, dqdx_ref, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, choices=list(DET))
    ap.add_argument("--arm", default=None, help="PR arm tag (default: the production arm)")
    ap.add_argument("--out", default=None, help="release dir (default: the parent of this scripts dir)")
    ap.add_argument("--events", default="", help="comma list of <run6>_<idx> to build (default all)")
    ap.add_argument("--pre", type=float, default=2.0, help="waveform window: us BEFORE the flash time (the flash time is the 1-us bin the finder seeded on; the first OpHit can peak ~1 us earlier)")
    ap.add_argument("--post", type=float, default=10.0, help="waveform window: us AFTER the flash time")
    ap.add_argument("--wf-suffix", default="_wf", help="suffix of the light dir that carries the frames")
    a = ap.parse_args()
    cfg = DET[a.det]
    arm = a.arm or cfg["arm"]
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(out, exist_ok=True)
    arms = sorted(glob.glob(f"{IMG}/{a.det}/work/*_{arm}"))
    if a.events:
        want = set(a.events.split(","))
        arms = [d for d in arms if "_".join(os.path.basename(d).split("_")[:2]) in want]
    logfh = open(f"{out}/build_log.txt", "a")
    log(f"== build_release {a.det} arm {arm}: {len(arms)} events -> {out}  ({time.strftime('%Y-%m-%d %H:%M:%S')})", logfh)
    rows = []; ref0 = None; scheme_cache = {}; agg = {}
    for d in arms:
        row, ref, checks = build_event(a.det, cfg, d, out, a, scheme_cache, logfh)
        rows.append(row)
        if ref is None:
            pass
        elif ref0 is None:
            ref0 = ref
        else:
            assert ref["grid"] == ref0["grid"] and np.array_equal(ref["muon"], ref0["muon"]) and np.array_equal(ref["electron"], ref0["electron"]), d
        for k, v in checks.items():
            if isinstance(v, tuple):
                agg[k] = tuple(x + y for x, y in zip(agg.get(k, (0,) * len(v)), v))
            else:
                agg[k] = max(agg.get(k, 0.0), v)
    if ref0 is not None:
        with open(f"{out}/dqdx_ref.json", "w") as fh:
            json.dump(dict(ref0, description="dQ/dx expectation vs residual range from the reconstruction's own "
                           "ParticleDataSet table (calib-pr-evt*.json dqdx_ref): Modified-Box recombination at the detector "
                           "field, x 0.85; grid in cm, values in electrons/cm"), fh)
    # index.csv: merge with what is there (a partial rebuild keeps the other events)
    idx_path = f"{out}/index.csv"
    old = {}
    if os.path.exists(idx_path):
        for r in csv.DictReader(open(idx_path)):
            old[(int(r["run"]), int(r["idx"]))] = r
    for r in rows:
        old[(r["run"], r["idx"])] = {k: str(v) for k, v in r.items()}
    with open(idx_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for k in sorted(old):
            w.writerow(old[k])
    log(f"== done: {len(rows)} events; aggregate checks {agg}", logfh)
    logfh.close()


if __name__ == "__main__":
    main()

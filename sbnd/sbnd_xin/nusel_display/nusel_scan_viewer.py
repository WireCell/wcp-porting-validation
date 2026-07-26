#!/usr/bin/env python3
"""SBND neutrino-selection (nusel) hand-scan event display (Bokeh server).

Scans the per-bundle tagger verdicts of the PR chain (TGM / STM / FC) against
the same inputs Bee shows, so a Bee tab can be used side by side:

  * charge  : ql_evt<ID>/mabc-all-apa.zip -> data/0/0-clustering-global.json
              (the exact layer uploaded to Bee: T0-corrected x, per-point
              cluster_id / real_cluster_id)
  * light   : same zip -> data/0/0-op.json (op_pes measured, op_pes_pred
              predicted, per flash)
  * bundles : ql_evt<ID>/pctree-evt<ID>.tar.gz via nusel_extract.parse_pctree
              (cluster ident / matched_flash_gid / flag_main) + group_flashes
              (the 80 ns cross-APA physical-flash grouping the table uses)
  * rows    : nusel_evt<ID>/nusel-evt<ID>.tsv (nusel_extract output: one row
              per qualifying bundle + in-beam no-bundle flashes, with the
              tgm/stm/fc verdicts and the auto label)

Navigation: prev/next event buttons; within an event click a bundle row (or
use the < > bundle buttons) -- one row per physical flash group, like Bee's
> < flash stepper.  Two modes: all bundles, or in-beam flash bundles only.

Per bundle the display shows the three charge projections (X-Y, Y-Z, X-Z)
with the SBND detector box + the taggers' effective fiducial volume, the main
cluster in red and companion (same-flash) clusters in blue (or all bundle
points colored by charge), and the measured vs predicted light pattern of the
flash on both TPC sides.

Scan labels: TGM / STM / FC / LM (light mismatch), multi-select, plus a
free-text comment per bundle and per event.  Everything autosaves to
<work_root>/nusel_labels/<tag>/.scan_state-evt<ID>.json on every change and
"Save labels" writes the actionable per-event JSON
<work_root>/nusel_labels/<tag>/nusel-labels-evt<ID>.json.

Previous-scan baselines (--prev ROOT[:TAG], repeatable, priority = given
order): each names an OLDER work root and the label tag scanned there.  They
are READ-ONLY (M13: old records are never touched).  Bundles are matched
across runs by flash APA + time (cluster idents/gids may relabel), and:
  * carry-over: on first load of an event under the CURRENT tag, the previous
    scan labels / comments / event comment are copied in (per key, first prev
    that has one wins) so a scan never has to be redone from scratch.  Carried
    keys are remembered; clearing one does not resurrect it on reload.
  * change flag: the "prev" column shows the baseline verdicts (first prev
    with the event); a row whose tgm/stm/fc changed is tinted amber until it
    is re-scanned (any label/comment edit, or the "re-scan OK" button).

Launched by serve_nusel_scan.sh; mirrors ql_scan/ql_scan_viewer.py.
"""

import sys
import os
import glob
import json
import math
import zipfile
from collections import defaultdict

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, Select, Button,
                          Div, ColorBar, HoverTool, BasicTicker,
                          NumeralTickFormatter, HTMLTemplateFormatter,
                          CheckboxButtonGroup, Toggle, TextAreaInput)
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # sbnd_xin/ for nusel_extract
from nusel_extract import parse_pctree, group_flashes, read_table, COLUMNS  # noqa: E402

# uproot is optional: only needed for the STM dQ/dx panel (tracking-stm.root
# written by run_nusel_evt.sh -stm-fit, doc 40).  Without it the panel just
# stays empty.
try:
    import uproot  # noqa: E402
except Exception:
    uproot = None

# Reference dQ/dx-vs-residual-range curves (e/cm).  The json now carries TWO
# sets of five (doc 55 sec 10), and the distinction matters here:
#
#   *DeDxBox  -- the Modified-Box tables the compiled WCT config actually holds
#                (sbnd/particle_dataset.jsonnet), i.e. the EXACT curve
#                TaggerCheckSTM's eval_stm compares against.  Anything on this
#                panel that reports on a tagger DECISION must use these:  the
#                drawn reference, and MIP_DQDX below.
#   *DeDx     -- the free-power fit of doc 55 sec 7g, i.e. the best current
#                description of real SBND stopping tracks.  Drawn alongside,
#                dashed, so the gap between "what the tagger believes" and
#                "what the data say" is visible rather than assumed away.
#
# See docs/40_stm-trackfit-validation-plan.md, docs/48 and docs/55.
STM_REF = {}
try:
    with open(os.path.join(HERE, "stm_ref_dqdx.json")) as _f:
        STM_REF = json.load(_f)
except Exception:
    pass
# Key of the curve the tagger uses.  Falls back to the canonical name for a
# pre-doc-55 json, which held only the Box tables under those names.
REF_TAGGER = "MuonDeDxBox" if "MuonDeDxBox" in STM_REF else "MuonDeDx"
REF_FIT = "MuonDeDx" if "MuonDeDxBox" in STM_REF else None

# Flat MIP dQ/dx (e/cm) = TaggerCheckSTM's ref_flat and its threshold divisor,
# i.e. the jsonnet `mip_dqdx` knob.  MUST track the config or the drawn line and
# the "MIP" units in the summary lie: SBND runs 56000 (doc 48), the C++ default
# and MicroBooNE value is 50000.  Derived from the muon table's plateau, which
# is where 56000/50000 came from (50000/48879.4 = 56000/54657.7 = 1.023), so it
# follows a regenerated stm_ref_dqdx.json automatically instead of drifting.
MIP_DQDX = 50e3
if REF_TAGGER in STM_REF:
    _plateau = STM_REF[REF_TAGGER]["values"][-1]
    MIP_DQDX = round(_plateau * 1.02293, -3)   # 48879.4 -> 50000, 54657.7 -> 56000

# TaggerCheckSTM save_stm_fit pass-status codes (clus/src/TaggerCheckSTM.cxx).
STM_STATUS = {0: "accepted (STM)", 1: "TGM", 2: "rej: long leftover",
              3: "rej: dQ/dx eval", 4: "rej: extra tracks",
              5: "rej: proton end", 6: "fitted, no decision"}

# Max charge points drawn for the gray full-event context (stride-downsampled).
MAX_CTX_PTS = 8000

# Scan labels offered (multi-select). LM = light mismatch (pred vs meas pattern).
SCAN_LABELS = ["TGM", "STM", "FC", "LM"]

# SBND PR fiducial: BoxFiducial sbnd_pr_fv (cfg/pgrapher/experiment/sbnd/
# clus.jsonnet) = the overall sensitive bounds spanning BOTH TPCs, and the
# fv_tolerance margins [-2,-2,-2.5,-2.5,-3,-3] cm that inset it into the
# effective FV the taggers test.  Cathode at x=0.
DET_BOX = dict(x=(-201.05, 201.05), y=(-199.312, 199.312), z=(0.85, 500.15))
FV_BOX = dict(x=(-199.05, 199.05), y=(-196.812, 196.812), z=(3.85, 497.15))

# Cross-APA physical-flash coincidence (us) -- matches nusel_extract FLASH_GROUP_NS.
OP_MATCH_TOL_US = 0.02


# ---------------------------------------------------------------------------
# Charge source (doc 50 Q1 / doc 52).  "ql" (default, historical) reads the
# Q/L job's mabc-all-apa.zip, which is PRE-un-merge: the drawn charge is the
# merged bundle, while the STM trajectory over it came from the POST-un-merge
# nusel job -- the mismatch that made fits look like they crossed empty space.
# "pr" reads the nusel job's own mabc-pr.zip, so charge and fit are the same
# epoch.  Light (the op layer) only exists in the Q/L zip, so it is read from
# there either way.
# Inputs: work roots (dirs holding nusel_evt*/ + ql_evt*/) from --args.
# ---------------------------------------------------------------------------
def extract_args(argv):
    out, tag, prevs, csrc = [], "", [], "ql"
    it = iter(argv)
    for a in it:
        if a in ("--tag", "-t"):
            tag = next(it, "")
        elif a == "--prev":
            s = next(it, "")
            if s:
                prevs.append(s)
        elif a == "--charge-src":
            csrc = next(it, "ql")
        else:
            out.append(a)
    if csrc not in ("ql", "pr"):
        sys.exit(f"--charge-src must be 'ql' or 'pr', not {csrc!r}")
    return tag, prevs, out, csrc


def discover_events(argv):
    """[(label, tsv_path, work_root)] sorted by event id.  Args are work roots
    (dirs containing nusel_evt<ID>/nusel-evt<ID>.tsv) or explicit tsv paths."""
    found = []
    for a in argv:
        if os.path.isdir(a):
            found += glob.glob(os.path.join(a, "nusel_evt*", "nusel-evt*.tsv"))
        elif any(ch in a for ch in "*?["):
            found += glob.glob(a)
        elif os.path.isfile(a):
            found.append(a)
    out = []
    for f in sorted(set(found)):
        base = os.path.basename(f)                     # nusel-evt<ID>.tsv
        if not (base.startswith("nusel-") and base.endswith(".tsv")):
            continue
        label = base[len("nusel-"):-len(".tsv")]       # evt<ID>
        work_root = os.path.dirname(os.path.dirname(os.path.abspath(f)))
        out.append((label, f, work_root))
    def keyf(e):
        digits = "".join(ch for ch in e[0] if ch.isdigit())
        return (int(digits) if digits else 0, e[0])
    return sorted(out, key=keyf)


SCAN_TAG, _PREV_SPECS, _ARGS, CHARGE_SRC = extract_args(sys.argv[1:])
EVENTS = discover_events(_ARGS)
LABELS = [e[0] for e in EVENTS]
EVENT_OF = {e[0]: e for e in EVENTS}


# ---------------------------------------------------------------------------
# Previous-scan baselines (read-only).
# ---------------------------------------------------------------------------
class PrevScan:
    """One earlier run + hand scan: work root and (optional) label tag.
    Spec 'ROOT[:TAG]'.  Without TAG only the auto verdicts are compared."""

    def __init__(self, spec):
        root, _, tag = spec.partition(":")
        self.root = os.path.abspath(root)
        self.tag = tag
        self.name = tag if tag else os.path.basename(self.root)
        self._cache = {}

    def event(self, label):
        """{'rows', 'labels', 'comments', 'event_comment'} or None if this
        prev has no nusel table for the event."""
        if label in self._cache:
            return self._cache[label]
        evtid = label[len("evt"):]
        tsv = os.path.join(self.root, "nusel_evt" + evtid, f"nusel-{label}.tsv")
        res = None
        if os.path.isfile(tsv):
            try:
                header, raw = read_table(tsv, COLUMNS[0])
                i = {c: n for n, c in enumerate(header or COLUMNS)}
                rows = [dict(main_id=int(r[i["main_id"]]),
                             flash_gid=int(r[i["flash_gid"]]),
                             flash_apa=int(r[i["flash_apa"]]),
                             t_us=float(r[i["flash_time_us"]]),
                             npts=int(r[i["npts_bundle"]]),
                             len_cm=float(r[i["len_main_cm"]]),
                             tgm=int(r[i["tgm"]]), stm=int(r[i["stm"]]),
                             fc=int(r[i["fc"]]),
                             # lm column exists only on doc-34+ tables; -1 =
                             # no verdict, matching nusel_extract's convention.
                             lm=int(r[i["lm"]]) if "lm" in i else -1,
                             auto_label=r[i["label"]])
                        for r in raw]
                labels, comments, evtc = {}, {}, ""
                if self.tag:
                    p = os.path.join(self.root, "nusel_labels", self.tag,
                                     f".scan_state-{label}.json")
                    if os.path.isfile(p):
                        with open(p) as fh:
                            d = json.load(fh)
                        labels = d.get("labels", {})
                        comments = d.get("comments", {})
                        evtc = d.get("event_comment", "")
                res = dict(rows=rows, labels=labels, comments=comments,
                           event_comment=evtc)
            except (OSError, ValueError, KeyError, IndexError):
                res = None
        self._cache[label] = res
        return res


PREVS = [PrevScan(s) for s in _PREV_SPECS]

# Bundle identity ACROSS runs: flash APA + time (cluster idents and flash
# gids can relabel between runs; flash times survive any charge-side change).
PREV_MATCH_TOL_US = 0.5


def match_prev_rows(cur_rows, prev_rows):
    """{current row index -> prev row dict}, greedy 1-1 by |dt| within the
    same APA; equal main_id breaks ties (helps two flashes < tol apart)."""
    cands = []
    for ci, r in enumerate(cur_rows):
        for pj, p in enumerate(prev_rows):
            if p["flash_apa"] != r["flash_apa"]:
                continue
            dt = abs(p["t_us"] - r["t_us"])
            if dt > PREV_MATCH_TOL_US:
                continue
            cands.append((dt, 0 if p["main_id"] == r["main_id"] else 1, ci, pj))
    cands.sort()
    used_c, used_p, out = set(), set(), {}
    for _, _, ci, pj in cands:
        if ci in used_c or pj in used_p:
            continue
        used_c.add(ci)
        used_p.add(pj)
        out[ci] = prev_rows[pj]
    return out


def build_previnfo(evt):
    """Attach evt.previnfo / evt.prev_event_comment.

    previnfo: None when no --prev covers this event, else
    {row index: {'name','row','changed','labels','lab_from','comment',
                 'cmt_from'}} for every current row.  Verdict baseline = the
    FIRST prev that has the event; labels/comments each come from the first
    prev that annotated the matched bundle (priority = --prev order)."""
    if getattr(evt, "previnfo", "unset") != "unset":
        return
    evt.previnfo, evt.prev_event_comment = None, ""
    base_done = False
    for pv in PREVS:
        d = pv.event(evt.label)
        if d is None:
            continue
        if evt.previnfo is None:
            evt.previnfo = {i: dict(name=pv.name, row=None, changed=False,
                                    labels=[], lab_from="", comment="",
                                    cmt_from="")
                            for i in range(len(evt.rows))}
        matched = match_prev_rows(evt.rows, d["rows"])
        for ci, p in matched.items():
            pi = evt.previnfo[ci]
            if not base_done:
                pi["name"] = pv.name
                pi["row"] = p
                r = evt.rows[ci]
                # auto_label included so an LM demotion (nu-candidate -> LM,
                # same tgm/stm/fc) still tints amber; raw lm codes are NOT
                # compared (a pre-LM baseline reads -1 everywhere and would
                # flag every row).
                pi["changed"] = ((p["tgm"], p["stm"], p["fc"], p["auto_label"])
                                 != (r["tgm"], r["stm"], r["fc"], r["auto_label"]))
            pkey = f"{p['main_id']}:{p['flash_gid']}"
            if not pi["labels"] and d["labels"].get(pkey):
                pi["labels"] = list(d["labels"][pkey])
                pi["lab_from"] = pv.name
            if not pi["comment"] and d["comments"].get(pkey):
                pi["comment"] = d["comments"][pkey]
                pi["cmt_from"] = pv.name
        if not evt.prev_event_comment and d["event_comment"]:
            evt.prev_event_comment = d["event_comment"]
        base_done = True


def find_opdets():
    """Opdet positions from wire-cell-data sbnd/photodet/semi-analytical-sbnd.json
    (list index = op channel index).  Resolved via SBND_OPDET_JSON or WIRECELL_PATH."""
    cands = [os.environ.get("SBND_OPDET_JSON", "")]
    for d in os.environ.get("WIRECELL_PATH", "").split(":"):
        if d:
            cands.append(os.path.join(d, "sbnd", "photodet", "semi-analytical-sbnd.json"))
    cands.append("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet/"
                 "semi-analytical-sbnd.json")
    for c in cands:
        if c and os.path.isfile(c):
            with open(c) as fh:
                od = json.load(fh)["OpDets"]
            return (np.array([o["x"] for o in od]),
                    np.array([o["y"] for o in od]),
                    np.array([o["z"] for o in od]))
    return None, None, None


OD_X, OD_Y, OD_Z = find_opdets()
OD_APA = None if OD_X is None else (OD_X > 0).astype(int)   # TPC0 = x<0, TPC1 = x>0


# ---------------------------------------------------------------------------
# Per-event model.
# ---------------------------------------------------------------------------
class Event:
    def __init__(self, label, tsv, work_root):
        self.label = label                              # evt<ID>
        self.work_root = work_root
        self.tsv = tsv
        evtid = label[len("evt"):]
        qldir = os.path.join(work_root, "ql_evt" + evtid)
        self.problems = []

        # --- scan rows (nusel_extract output) --------------------------------
        header, rows = read_table(tsv, COLUMNS[0])
        i = {c: n for n, c in enumerate(header or COLUMNS)}
        self.rows = []
        for r in rows:
            self.rows.append(dict(
                run=int(r[i["run"]]), subrun=int(r[i["subrun"]]),
                event=int(r[i["event"]]),
                main_id=int(r[i["main_id"]]), flash_gid=int(r[i["flash_gid"]]),
                flash_apa=int(r[i["flash_apa"]]), flash_grp=int(r[i["flash_grp"]]),
                t_us=float(r[i["flash_time_us"]]), pe=float(r[i["flash_pe"]]),
                pe_grp=float(r[i["flash_pe_grp"]]), in_beam=int(r[i["in_beam"]]),
                n_bundle=int(r[i["n_bundle"]]), npts_main=int(r[i["npts_main"]]),
                npts_bundle=int(r[i["npts_bundle"]]),
                len_cm=float(r[i["len_main_cm"]]), n_frag=int(r[i["n_frag"]]),
                tgm=int(r[i["tgm"]]), stm=int(r[i["stm"]]), fc=int(r[i["fc"]]),
                lm=int(r[i["lm"]]) if "lm" in i else -1,
                # stmfit: 'fit' = the STM trajectory fit ran (the dQ/dx panel
                # will have curves); anything else names the pre-fit exit that
                # stopped it ('contained', 'midkink', 'nexits', 'nosteiner',
                # 'shortfit', 'nofidu', 'tgm', '-' = never evaluated).
                # '' on TSVs written before the column existed.
                stmfit=r[i["stmfit"]] if "stmfit" in i else "",
                auto_label=r[i["label"]]))
        self.rows.sort(key=lambda r: r["t_us"])

        # --- bundle structure from the pctree --------------------------------
        self.clusters = []          # [{ident,gid,main,npts,...}]
        self.flashes = {}           # gid -> {t_us, pe, apa}
        self.fgrp = {}              # gid -> physical-flash group id
        pctree = os.path.join(qldir, f"pctree-{label}.tar.gz")
        if os.path.isfile(pctree):
            try:
                _, self.clusters, self.flashes = parse_pctree(pctree)
                self.fgrp = group_flashes(self.flashes)
            except Exception as e:          # keep scanning even if the tree is bad
                self.problems.append(f"pctree: {e}")
        else:
            self.problems.append(f"missing {pctree}")

        # --- Bee layers (charge points + light) ------------------------------
        self.pts = None             # dict of numpy arrays x,y,z,q,cid,rid
        self.op = None              # dict of lists (op.json)
        qlzip = os.path.join(qldir, "mabc-all-apa.zip")
        # Charge from the requested epoch; light always from the Q/L zip.
        przip = os.path.join(work_root, "nusel_evt" + evtid, "mabc-pr.zip")
        chzip = przip if (CHARGE_SRC == "pr" and os.path.isfile(przip)) else qlzip
        if CHARGE_SRC == "pr" and not os.path.isfile(przip):
            self.problems.append("charge_src=pr but missing " + przip
                                 + " -- falling back to the PRE-un-merge Q/L dump")
        if os.path.isfile(chzip):
            try:
                with zipfile.ZipFile(chzip) as z:
                    names = z.namelist()
                    cg = [n for n in names if n.endswith("-clustering-global.json")]
                    opn = [n for n in names if n.endswith("-op.json")]
                    if cg:
                        d = json.loads(z.read(cg[0]))
                        self.pts = dict(
                            x=np.array(d["x"], float), y=np.array(d["y"], float),
                            z=np.array(d["z"], float), q=np.array(d["q"], float),
                            cid=np.array(d["cluster_id"], int),
                            rid=np.array(d.get("real_cluster_id",
                                               d["cluster_id"]), int))
                    else:
                        self.problems.append("no clustering-global layer in " + chzip)
                    if opn:
                        self.op = json.loads(z.read(opn[0]))
            except Exception as e:
                self.problems.append(f"{chzip}: {e}")
        else:
            self.problems.append(f"missing {chzip}")
        # mabc-pr.zip carries no op layer, so read the light from the Q/L zip.
        if self.op is None and os.path.isfile(qlzip):
            try:
                with zipfile.ZipFile(qlzip) as z:
                    opn = [n for n in z.namelist() if n.endswith("-op.json")]
                    if opn:
                        self.op = json.loads(z.read(opn[0]))
                    else:
                        self.problems.append("no op layer in " + qlzip)
            except Exception as e:
                self.problems.append(f"{qlzip}: {e}")

        # --- STM track-fit dump (run_nusel_evt.sh -stm-fit, doc 40) ----------
        # Absent on knob-off rounds -- the dQ/dx panel then stays empty.
        self.stm = None
        stmroot = os.path.join(work_root, "nusel_evt" + evtid, "tracking-stm.root")
        if uproot is not None and os.path.isfile(stmroot):
            try:
                f = uproot.open(stmroot)
                trun = f["Trun"].arrays(library="np")
                scale = float(trun["dQdx_scale"][0])
                offset = float(trun["dQdx_offset"][0])
                rec = f["T_rec_charge"].arrays(
                    ["x", "y", "z", "q", "nq", "rr", "cluster_id", "pass",
                     "status", "reduced_chi2"], library="np")
                dQ = (rec["q"] - offset) / scale            # raw electrons
                with np.errstate(divide="ignore", invalid="ignore"):
                    dqdx = np.where(rec["nq"] > 0, dQ / rec["nq"], 0.0)  # e/cm
                self.stm = dict(
                    cid=rec["cluster_id"].astype(int),
                    x=rec["x"], y=rec["y"], z=rec["z"],
                    rr=rec["rr"], dqdx=dqdx, dQ=dQ, dx=rec["nq"],
                    passno=rec["pass"].astype(int),
                    status=rec["status"].astype(int),
                    chi2=rec["reduced_chi2"],
                    passes=f["T_stm_pass"].arrays(library="np"),
                    evals=f["T_stm_eval"].arrays(library="np"))
            except Exception as e:
                self.problems.append(f"{stmroot}: {e}")

    # --- bundle helpers -----------------------------------------------------
    def bundle_cluster_ids(self, r):
        """(main_ident, [companion idents]) for a scan row (peers = clusters
        matched to the same flash gid, exactly nusel_extract's definition)."""
        if r["main_id"] < 0:
            return None, []
        comps = [c["ident"] for c in self.clusters
                 if c["gid"] == r["flash_gid"] and c["ident"] != r["main_id"]]
        return r["main_id"], sorted(comps)

    def components(self, ident):
        """Merge components of cluster `ident`, biggest first:
        [{rid, n, len_cm}].

        QLMatching's examine_bundles flash merge can fuse several physically
        separate clusters into ONE cluster object when they share a flash; the
        Bee clustering layer records each point's pre-merge identity in
        real_cluster_id (it is written only on merged clusters, and does not
        survive into the pctree).  The taggers run on the MERGED object, so all
        components are in their scope -- but the table's len_main_cm is the
        DOMINANT (most-points) component's length, matching
        nusel_extract.parse_qlbee.  Distinguishing them is what makes a
        multi-piece red blob readable during a scan.
        """
        out = []
        if self.pts is None or ident is None:
            return out
        m = self.pts["cid"] == ident
        for r in sorted(set(self.pts["rid"][m].tolist())):
            s = m & (self.pts["rid"] == r)
            P = np.c_[self.pts["x"][s], self.pts["y"][s], self.pts["z"][s]]
            if len(P) > 1:
                a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
                b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
                length = float(np.linalg.norm(P[a] - P[b]))
            else:
                length = 0.0
            out.append(dict(rid=int(r), n=int(s.sum()), len_cm=length))
        out.sort(key=lambda c: -c["n"])
        return out

    def partner_gid(self, gid):
        """The opposite-APA flash gid in the same physical-flash group, if any."""
        g = self.fgrp.get(gid)
        if g is None or gid not in self.flashes:
            return None
        apa = self.flashes[gid]["apa"]
        for o, og in self.fgrp.items():
            if og == g and o != gid and self.flashes[o]["apa"] != apa:
                return o
        return None

    def op_index(self, gid):
        """Index into the op.json arrays for pctree flash `gid` (matched by
        APA + time; op.json carries no gid)."""
        if self.op is None or gid not in self.flashes:
            return None
        f = self.flashes[gid]
        best, bestdt = None, OP_MATCH_TOL_US
        for k in range(len(self.op["op_t"])):
            if int(self.op["apa"][k]) != f["apa"]:
                continue
            dt = abs(self.op["op_t"][k] - f["t_us"])
            if dt <= bestdt:
                best, bestdt = k, dt
        return best

    def op_pattern(self, k):
        """(meas[312], pred[312]) for op.json entry k (pred zeros if empty)."""
        meas = np.array(self.op["op_pes"][k], float)
        pred = self.op["op_pes_pred"][k]
        pred = np.array(pred, float) if pred else np.zeros_like(meas)
        return meas, pred


_EVENT_CACHE = {}


def get_event(label):
    if label not in _EVENT_CACHE:
        _EVENT_CACHE[label] = Event(*EVENT_OF[label])
    return _EVENT_CACHE[label]


# ---------------------------------------------------------------------------
# Global UI state.
# ---------------------------------------------------------------------------
state = {
    "evt": None,          # Event
    "focus": None,        # focused row index (into evt.rows), or None
    "order": [],          # visible row indices in table display order
    "inbeam_only": False, # mode: in-beam flash bundles only
    "color_mode": "role", # 'role' (main red / companions blue) or 'charge'
    "labels": {},         # row key -> [scan labels]
    "comments": {},       # row key -> str
    "event_comment": "",
    "seeded": set(),      # keys whose annotation was carried from a --prev
                          # and not yet touched under the current tag
    "seen": set(),        # keys ever seeded (never re-seed: a cleared
                          # carry-over must not resurrect on reload)
    "confirmed": set(),   # keys re-scanned under the current tag (any
                          # label/comment edit, or the re-scan OK button)
    "suppress": False,    # guard programmatic widget writes
}


def row_key(r):
    return f"{r['main_id']}:{r['flash_gid']}"


# ---------------------------------------------------------------------------
# Widgets.
# ---------------------------------------------------------------------------
event_select = Select(title="Event", value=(LABELS[0] if LABELS else ""),
                      options=LABELS, width=180)
prev_evt_btn = Button(label="< prev evt", width=90)
next_evt_btn = Button(label="next evt >", width=90)
prev_bun_btn = Button(label="< prev bundle", width=110)
next_bun_btn = Button(label="next bundle >", width=110)
mode_btn = Toggle(label="Mode: ALL bundles", width=200, active=False)
color_select = Select(title="point color", value="role",
                      options=[("role", "main=red / companions=blue"),
                               ("charge", "charge (viridis)")], width=220)
save_btn = Button(label="Save labels", button_type="success", width=120)
status = Div(text="", width=1200)
metrics = Div(text="", width=430)
proj_info = Div(text="", width=1200)

label_title = Div(text="<b>scan labels</b> (click to tag the focused bundle; "
                       "multi-select; LM = light mismatch)", width=380)
label_group = CheckboxButtonGroup(labels=SCAN_LABELS, active=[], width=340)
clear_lbl_btn = Button(label="Clear labels (focused)", width=170)
confirm_btn = Button(label="✓ re-scan OK (focused)", width=170,
                     button_type="primary")
comment_in = TextAreaInput(title="bundle comment (focused)", value="", rows=3,
                           width=380, max_length=2000)
evt_comment_in = TextAreaInput(title="event comment", value="", rows=3,
                               width=380, max_length=4000)

# Row tint: green once the bundle carries any scan label or comment (progress
# tracking); the focused row uses the table's native blue selection highlight.
def cell_fmt(align):
    return HTMLTemplateFormatter(template=(
        '<div style="background-color:<%= sel_bg %>;text-align:' + align
        + ';margin:-2px -5px;padding:2px 5px"><%= value %></div>'))
fmt_l, fmt_r = cell_fmt("left"), cell_fmt("right")
SEL_BG = "#cde6cd"
CHG_BG = "#f7d8a8"   # amber: verdicts changed vs --prev, not yet re-scanned

table_src = ColumnDataSource(data=dict())
table_cols = [
    TableColumn(field="row", title="#", width=30, formatter=fmt_l, sortable=False),
    TableColumn(field="grp", title="grp", width=40, formatter=fmt_l, sortable=False),
    TableColumn(field="t_us", title="t(us)", width=75, formatter=fmt_r, sortable=False),
    TableColumn(field="pe", title="PE(grp)", width=75, formatter=fmt_r, sortable=False),
    TableColumn(field="beam", title="beam", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="main", title="main", width=45, formatter=fmt_l, sortable=False),
    TableColumn(field="clusters", title="clusters", width=95, formatter=fmt_l, sortable=False),
    TableColumn(field="npts", title="npts", width=55, formatter=fmt_r, sortable=False),
    TableColumn(field="len", title="len(cm)", width=65, formatter=fmt_r, sortable=False),
    TableColumn(field="verdicts", title="tgm/stm/fc[/lm]", width=95, formatter=fmt_l, sortable=False),
    # Did the STM trajectory fit run for this bundle's main cluster?  "fit" =>
    # the dQ/dx panel below has curves; any other value is the reason it does
    # not (see the stmfit note in Evt.rows).
    TableColumn(field="stmfit", title="stm fit", width=80, formatter=fmt_l, sortable=False),
    TableColumn(field="prev", title="prev", width=80, formatter=fmt_l, sortable=False),
    TableColumn(field="auto", title="auto label (+FC)", width=120, formatter=fmt_l, sortable=False),
    TableColumn(field="scan", title="scan", width=95, formatter=fmt_l, sortable=False),
    TableColumn(field="cmt", title="✎", width=25, formatter=fmt_l, sortable=False),
]
table = DataTable(source=table_src, columns=table_cols, width=980, height=290,
                  selectable=True, index_position=None)

# --- charge projections -----------------------------------------------------
proj_kw = dict(height=330, tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
f_xy = figure(title="X-Y", width=400, **proj_kw)
f_yz = figure(title="Y-Z", width=400, **proj_kw)
f_xz = figure(title="X-Z", width=400, **proj_kw)
f_xy.xaxis.axis_label, f_xy.yaxis.axis_label = "x (cm)", "y (cm)"
f_yz.xaxis.axis_label, f_yz.yaxis.axis_label = "z (cm)", "y (cm)"
f_xz.xaxis.axis_label, f_xz.yaxis.axis_label = "x (cm)", "z (cm)"

ctx_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))       # gray: whole event
EMPTY_PTS = dict(x=[], y=[], z=[], c=[], cid=[], rid=[], q=[])
comp_src = ColumnDataSource(data=dict(EMPTY_PTS))   # companion clusters (blue)
frag_src = ColumnDataSource(data=dict(EMPTY_PTS))   # main's other merge comps (orange)
main_src = ColumnDataSource(data=dict(EMPTY_PTS))   # main's dominant comp (red)
det_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                     xs_xz=[], ys_xz=[]))
fv_src = ColumnDataSource(data=dict(xs_xy=[], ys_xy=[], xs_yz=[], ys_yz=[],
                                    xs_xz=[], ys_xz=[]))
for f, sx, sy in ((f_xy, "xs_xy", "ys_xy"), (f_yz, "xs_yz", "ys_yz"),
                  (f_xz, "xs_xz", "ys_xz")):
    f.multi_line(xs=sx, ys=sy, source=det_src, line_color="#cc4444", line_width=1)
    f.multi_line(xs=sx, ys=sy, source=fv_src, line_color="#2ca02c",
                 line_width=1.5, line_dash="dashed")
for f, hx, hy in ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z")):
    f.scatter(hx, hy, source=ctx_src, marker="circle", size=2,
              fill_color="#c8c8c8", line_color=None, fill_alpha=0.45)
    rs = [f.scatter(hx, hy, source=comp_src, marker="circle", size=3,
                    fill_color="c", line_color=None, fill_alpha=0.8),
          # merge fragments of the MAIN cluster: same tagger scope as the
          # dominant piece, drawn distinctly (orange squares) so a multi-piece
          # main cluster is not misread as two separate clusters.
          f.scatter(hx, hy, source=frag_src, marker="square", size=3,
                    fill_color="c", line_color=None, fill_alpha=0.85),
          f.scatter(hx, hy, source=main_src, marker="circle", size=3,
                    fill_color="c", line_color=None, fill_alpha=0.9)]
    f.add_tools(HoverTool(renderers=rs, tooltips=[
        ("cluster", "@cid"), ("merge comp.", "@rid"), ("q", "@q{0,0}")]))

# STM fitted trajectory overlaid on the projections (black crosses; only
# populated when tracking-stm.root exists for the event -- doc 40).
stmtraj_src = ColumnDataSource(data=dict(x=[], y=[], z=[]))
for f, hx, hy in ((f_xy, "x", "y"), (f_yz, "z", "y"), (f_xz, "x", "z")):
    f.scatter(hx, hy, source=stmtraj_src, marker="cross", size=5,
              line_color="black", line_width=1.2, fill_color=None)

# --- STM dQ/dx panel (doc 40) ----------------------------------------------
# Measured fitted dQ/dx vs residual range for the focused bundle's main
# cluster, against the muon stopping expectation (the exact eval_stm reference
# table) and the flat MIP line.  Marker: TPC0 circle / TPC1 triangle
# (from the point's drift-x sign); color: forward pass blue / backward red.
f_dqdx = figure(title="STM fit: dQ/dx vs residual range (main cluster)",
                width=640, height=300,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom")
f_dqdx.xaxis.axis_label = "residual range (cm)"
f_dqdx.yaxis.axis_label = "dQ/dx (e/cm)"
f_dqdx.yaxis.formatter = NumeralTickFormatter(format="0a")
stmdqdx_src = ColumnDataSource(data=dict(rr=[], dqdx=[], color=[], marker=[],
                                         passno=[], apa=[], chi2=[]))
stmref_src = ColumnDataSource(data=dict(rr=[], mu=[], mu2=[]))
stmflat_src = ColumnDataSource(data=dict(rr=[], flat=[]))
f_dqdx.line("rr", "mu", source=stmref_src, line_color="#2ca02c", line_width=2,
            legend_label="muon expectation (tagger's table)")
f_dqdx.line("rr", "mu2", source=stmref_src, line_color="#2a78d6", line_width=2,
            line_dash="dotted", legend_label="muon, free-power fit (doc 55)")
f_dqdx.line("rr", "flat", source=stmflat_src, line_color="#888888",
            line_dash="dashed",
            # Label from MIP_DQDX, never a literal: the flat line IS the
            # mip_dqdx knob (SBND 56000 e/cm, doc 48), and a hardcoded "50"
            # mislabelled it as the inherited MicroBooNE value.
            legend_label=f"flat {MIP_DQDX / 1e3:.0f} ke/cm (mip_dqdx)")
r_dqdx = f_dqdx.scatter("rr", "dqdx", source=stmdqdx_src, marker="marker",
                        size=5, fill_color="color", line_color=None,
                        fill_alpha=0.85)
f_dqdx.add_tools(HoverTool(renderers=[r_dqdx], tooltips=[
    ("rr", "@rr{0.0} cm"), ("dQ/dx", "@dqdx{0,0} e/cm"),
    ("pass", "@passno"), ("TPC", "@apa"), ("chi2", "@chi2{0.00}")]))
f_dqdx.legend.location = "top_right"
f_dqdx.legend.label_text_font_size = "8pt"
stm_info = Div(text="", width=520, height=300)


def box_polys(box, cathode=False):
    """Rectangle polylines for a {x:(lo,hi),y:..,z:..} box in the 3 projections;
    with cathode=True adds the x=0 plane line to X-Y and X-Z."""
    (xl, xh), (yl, yh), (zl, zh) = box["x"], box["y"], box["z"]
    xy = [([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl])]
    yz = [([zl, zh, zh, zl, zl], [yl, yl, yh, yh, yl])]
    xz = [([xl, xh, xh, xl, xl], [zl, zl, zh, zh, zl])]
    if cathode:
        # every projection needs the same polyline count (one shared CDS)
        xy.append(([0, 0], [yl, yh]))
        yz.append(([], []))
        xz.append(([0, 0], [zl, zh]))
    return xy, yz, xz


def set_boxes():
    for src, box, cath in ((det_src, DET_BOX, True), (fv_src, FV_BOX, False)):
        xy, yz, xz = box_polys(box, cathode=cath)
        src.data = dict(xs_xy=[p[0] for p in xy], ys_xy=[p[1] for p in xy],
                        xs_yz=[p[0] for p in yz], ys_yz=[p[1] for p in yz],
                        xs_xz=[p[0] for p in xz], ys_xz=[p[1] for p in xz])


set_boxes()

# --- light-pattern figures --------------------------------------------------
def radii(vals, hi):
    return (2.5 + 9.0 * np.sqrt(np.clip(vals, 0, None) / hi)).tolist()


def make_light_fig(title):
    f = figure(title=title, height=270, width=430, tools="pan,reset,save")
    f.xaxis.axis_label = "z (cm)"
    f.yaxis.axis_label = "y (cm)"
    base = ColumnDataSource(data=dict(z=[], y=[]))
    src = ColumnDataSource(data=dict(z=[], y=[], pe=[], r=[], ch=[]))
    f.scatter("z", "y", source=base, marker="circle", size=5,
              fill_color=None, line_color="#cccccc")
    g = f.circle("z", "y", source=src, radius="r",
                 fill_color=linear_cmap("pe", "Viridis256", 0, 1),
                 line_color="#333333", fill_alpha=0.85)
    cbar = ColorBar(color_mapper=g.glyph.fill_color["transform"], title="PE",
                    ticker=BasicTicker(desired_num_ticks=6),
                    formatter=NumeralTickFormatter(format="0,0"),
                    width=14, padding=4, label_standoff=6,
                    title_text_font_size="11px", title_standoff=6,
                    major_label_text_font_size="10px")
    f.add_layout(cbar, "right")
    f.add_tools(HoverTool(renderers=[g], tooltips=[("ch", "@ch"), ("PE", "@pe{0,0.0}")]))
    return dict(fig=f, base=base, src=src, glyph=g)


LIGHT = {apa: {"meas": make_light_fig(f"TPC{apa} measured"),
               "pred": make_light_fig(f"TPC{apa} predicted")}
         for apa in (0, 1)}


def make_overlay_fig(title):
    f = figure(title=title, height=220, width=430,
               tools="pan,box_zoom,wheel_zoom,reset,save")
    f.xaxis.axis_label = "op channel"
    f.yaxis.axis_label = "PE"
    meas_src = ColumnDataSource(data=dict(x=[], pe=[]))
    pred_src = ColumnDataSource(data=dict(x=[], pe=[]))
    f.vbar(x="x", top="pe", width=0.9, source=meas_src, fill_color="#6baed6",
           line_color=None, fill_alpha=0.6, legend_label="measured")
    f.line("x", "pe", source=pred_src, line_color="#d62728", line_width=1.5,
           legend_label="predicted")
    f.scatter("x", "pe", source=pred_src, marker="circle", size=4,
              fill_color="#d62728", line_color=None, legend_label="predicted")
    f.legend.label_text_font_size = "9px"
    f.legend.padding = 2
    f.legend.location = "top_right"
    f.legend.background_fill_alpha = 0.6
    return dict(fig=f, meas=meas_src, pred=pred_src)


OVERLAY = {apa: make_overlay_fig(f"TPC{apa} meas vs pred") for apa in (0, 1)}


def set_light_ranges():
    for apa in (0, 1):
        for panel in (LIGHT[apa]["meas"], LIGHT[apa]["pred"]):
            f = panel["fig"]
            f.x_range.start, f.x_range.end = -25, 525
            f.y_range.start, f.y_range.end = -220, 220


set_light_ranges()


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def visible_rows(evt):
    idxs = list(range(len(evt.rows)))
    if state["inbeam_only"]:
        idxs = [i for i in idxs if evt.rows[i]["in_beam"] == 1]
    return idxs


def rebuild_table():
    evt = state["evt"]
    order = visible_rows(evt)
    state["order"] = order
    cols = defaultdict(list)
    for n, i in enumerate(order):
        r = evt.rows[i]
        main, comps = evt.bundle_cluster_ids(r)
        key = row_key(r)
        labs = state["labels"].get(key, [])
        cols["row"].append(str(n))
        cols["grp"].append(str(r["flash_grp"]))
        cols["t_us"].append(f"{r['t_us']:.3f}")
        cols["pe"].append(f"{r['pe_grp']:.0f}")
        cols["beam"].append("Y" if r["in_beam"] else "")
        cols["main"].append(str(main) if main is not None else "-")
        cols["clusters"].append("+".join(str(c) for c in ([main] + comps))
                                if main is not None else "-")
        cols["npts"].append(str(r["npts_bundle"]))
        cols["len"].append(f"{r['len_cm']:.1f}")
        cols["verdicts"].append(f"{r['tgm']}/{r['stm']}/{r['fc']}"
                                + (f"/{r['lm']}" if r["lm"] >= 0 else ""))
        # The TSV's 'eval' means "the tagger evaluated it with no logged exit";
        # only the dQ/dx dump proves a trajectory was persisted, and that is
        # exactly what decides whether the panel below has curves.  Upgrade to
        # 'fit' (green) or demote to 'postfit' when the dump is loaded.
        sf = r.get("stmfit", "")
        if evt.stm is not None and main is not None and sf in ("eval", "fit"):
            sf = "fit" if (evt.stm["cid"] == main).any() else "postfit"
        cols["stmfit"].append(
            "<b style='color:#1a7f37'>fit</b>" if sf == "fit"
            else (f"<span style='color:#888'>{sf}</span>" if sf else "?"))
        # FC is orthogonal to the TGM/STM/nu-candidate label (it never enters
        # nusel_extract's 'label'), so badge it explicitly onto the auto cell:
        # a fully-contained bundle -- especially an in-beam nu-candidate -- is
        # a neutrino-like signature worth seeing at a glance.  fmt_l is an
        # HTMLTemplateFormatter (<%= %>, unescaped), so inline HTML renders.
        auto = r["auto_label"]
        if r["fc"] == 1:
            auto += " <b style='color:#2ca02c'>FC</b>"
        # An out-of-beam light mismatch never enters 'label' (only in-beam LM
        # demotes nu-candidate), so badge it here like FC.
        if r["lm"] == 2 and r["auto_label"] != "LM":
            auto += " <b style='color:#b06000'>lm</b>"
        cols["auto"].append(auto)
        # prev column: baseline verdicts from the first --prev with this
        # event ('=' same, 'a/b/c→' changed, 'new' unmatched); ✓ = re-scanned.
        pi = evt.previnfo.get(i) if evt.previnfo else None
        pending = False
        if pi is None:
            ptxt = ""
        elif pi["row"] is None:
            ptxt = "new"
        elif pi["changed"]:
            p = pi["row"]
            ptxt = f"<b>{p['tgm']}/{p['stm']}/{p['fc']}→</b>"
            pending = key not in state["confirmed"]
        else:
            ptxt = "="
        if pi is not None and key in state["confirmed"]:
            ptxt += " ✓"
        cols["prev"].append(ptxt)
        cols["scan"].append("+".join(labs)
                            + (" ◦" if key in state["seeded"] else ""))
        cols["cmt"].append("✎" if state["comments"].get(key) else "")
        cols["sel_bg"].append(CHG_BG if pending
                              else SEL_BG if (labs or state["comments"].get(key))
                              else "transparent")
    new_data = dict(cols)
    # Bokeh 3.9 DataTable does not redraw when the replacement data has the
    # SAME row count (the client CDS updates but the slick-grid keeps the
    # stale cells; verified headless-chromium 2026-07-23, evt286065->286197).
    # Force a row-count change so the grid invalidates.
    if len(new_data.get("row", [])) == len(table_src.data.get("row", [])):
        table_src.data = {k: [] for k in new_data} if new_data else {}
    table_src.data = new_data
    if state["focus"] is not None and state["focus"] in order:
        table_src.selected.indices = [order.index(state["focus"])]
    else:
        table_src.selected.indices = []


def charge_colors(q):
    """Viridis colors for charge values (robust 99th-percentile scale)."""
    hi = max(1.0, float(np.percentile(q, 99))) if q.size else 1.0
    idx = np.clip(q / hi * 255, 0, 255).astype(int)
    return [Viridis256[k] for k in idx]


def render_projections():
    evt = state["evt"]
    # gray context: the whole event (downsampled), exactly the Bee clustering layer
    if evt.pts is not None:
        n = len(evt.pts["x"])
        s = max(1, int(math.ceil(n / MAX_CTX_PTS)))
        ctx_src.data = dict(x=evt.pts["x"][::s].tolist(),
                            y=evt.pts["y"][::s].tolist(),
                            z=evt.pts["z"][::s].tolist())
    else:
        ctx_src.data = dict(x=[], y=[], z=[])

    i = state["focus"]
    blank = lambda: (main_src.update(data=dict(EMPTY_PTS)),
                     frag_src.update(data=dict(EMPTY_PTS)),
                     comp_src.update(data=dict(EMPTY_PTS)))
    if i is None or evt.pts is None:
        blank()
        return
    r = evt.rows[i]
    main, comps = evt.bundle_cluster_ids(r)
    if main is None:
        blank()
        return
    cid, rid = evt.pts["cid"], evt.pts["rid"]
    mm = cid == main
    # Split the main cluster into its merge components: the dominant one (red)
    # vs the grafted fragments (orange).  Both are inside the SAME cluster
    # object the taggers evaluated -- this is a provenance distinction, not a
    # scope one (see Event.components).
    parts = evt.components(main)
    dom_rid = parts[0]["rid"] if parts else None
    dm = mm & (rid == dom_rid) if dom_rid is not None else mm
    fm = mm & ~dm
    cm = np.isin(cid, comps) if comps else np.zeros_like(mm)
    for src, mask, role_color in ((main_src, dm, "#d62728"),
                                  (frag_src, fm, "#ff7f0e"),
                                  (comp_src, cm, "#1f77b4")):
        x, y, z = evt.pts["x"][mask], evt.pts["y"][mask], evt.pts["z"][mask]
        q = evt.pts["q"][mask]
        if state["color_mode"] == "charge":
            c = charge_colors(q)
        else:
            c = [role_color] * len(x)
        src.data = dict(x=x.tolist(), y=y.tolist(), z=z.tolist(), c=c,
                        cid=cid[mask].tolist(), rid=rid[mask].tolist(),
                        q=q.tolist())


def render_light():
    """Per TPC side: measured + predicted pattern of the physical flash of the
    focused bundle (own-side flash and its 80 ns cross-APA partner)."""
    evt = state["evt"]
    i = state["focus"]
    side_gid = {0: None, 1: None}
    if i is not None and evt.flashes:
        gid = evt.rows[i]["flash_gid"]
        if gid in evt.flashes:
            side_gid[evt.flashes[gid]["apa"]] = gid
            p = evt.partner_gid(gid)
            if p is not None:
                side_gid[evt.flashes[p]["apa"]] = p
    for apa in (0, 1):
        mp, pp = LIGHT[apa]["meas"], LIGHT[apa]["pred"]
        ov = OVERLAY[apa]
        if OD_APA is not None:
            am = OD_APA == apa
            chans = np.nonzero(am)[0]
            mp["base"].data = dict(z=OD_Z[am].tolist(), y=OD_Y[am].tolist())
            pp["base"].data = dict(z=OD_Z[am].tolist(), y=OD_Y[am].tolist())
        else:
            chans = None
        gid = side_gid[apa]
        k = evt.op_index(gid) if gid is not None else None
        if k is None:
            mp["src"].data = dict(z=[], y=[], pe=[], r=[], ch=[])
            pp["src"].data = dict(z=[], y=[], pe=[], r=[], ch=[])
            ov["meas"].data = dict(x=[], pe=[])
            ov["pred"].data = dict(x=[], pe=[])
            what = "no flash this side" if gid is None else "flash not in op layer"
            mp["fig"].title.text = f"TPC{apa} measured  ({what})"
            pp["fig"].title.text = f"TPC{apa} predicted"
            ov["fig"].title.text = f"TPC{apa} meas vs pred"
            continue
        meas_full, pred_full = evt.op_pattern(k)
        f = evt.flashes[gid]
        if chans is not None:
            meas, pred = meas_full[chans], pred_full[chans]
            hi_m = max(1.0, float(meas.max()))
            hi_p = max(1.0, float(pred.max()))
            mp["glyph"].glyph.fill_color["transform"].high = hi_m
            pp["glyph"].glyph.fill_color["transform"].high = hi_p
            zz, yy = OD_Z[chans].tolist(), OD_Y[chans].tolist()
            ch = chans.tolist()
            mp["src"].data = dict(z=zz, y=yy, pe=meas.tolist(),
                                  r=radii(meas, hi_m), ch=ch)
            pp["src"].data = dict(z=zz, y=yy, pe=pred.tolist(),
                                  r=radii(pred, hi_p), ch=ch)
            xidx = chans
        else:
            meas, pred = meas_full, pred_full
            xidx = np.arange(meas.size)
        mp["fig"].title.text = (f"TPC{apa} measured  gid {gid}  t={f['t_us']:.3f} us"
                                f"  totPE={meas_full.sum():.0f}")
        pp["fig"].title.text = (f"TPC{apa} predicted  totPE={pred_full.sum():.1f}"
                                + ("" if pred_full.any() else "  (no prediction)"))
        ov["meas"].data = dict(x=xidx.tolist(), pe=meas.tolist())
        ov["pred"].data = dict(x=xidx.tolist(), pe=pred.tolist())
        ov["fig"].title.text = f"TPC{apa} meas vs pred  (gid {gid}, t={f['t_us']:.3f} us)"


def render_metrics():
    evt = state["evt"]
    i = state["focus"]
    if i is None:
        metrics.text = "<i>click a bundle row to inspect</i>"
        return
    r = evt.rows[i]
    main, comps = evt.bundle_cluster_ids(r)
    key = row_key(r)
    partner = evt.partner_gid(r["flash_gid"])
    ptxt = (f"gid {partner} (t={evt.flashes[partner]['t_us']:.3f} us)"
            if partner is not None else "-")
    verdict = lambda v: {1: "YES", 0: "no", -1: "n/a"}.get(v, str(v))
    # Merge-component breakdown of the main cluster: the taggers ran ONCE on
    # the whole merged object, so several disconnected pieces on screen can all
    # belong to the single main cluster.
    parts = evt.components(main)
    if len(parts) > 1:
        chips = []
        for n_, p in enumerate(parts):
            col = "#d62728" if n_ == 0 else "#ff7f0e"
            role = "dominant" if n_ == 0 else "grafted"
            chips.append(f"<span style='color:{col}'>#{p['rid']} {role}: "
                         f"{p['n']} pts, {p['len_cm']:.1f} cm</span>")
        comp_txt = ("<b>%d merge components</b> (one cluster, tagged once)<br>"
                    % len(parts)) + "<br>".join(chips)
    else:
        comp_txt = "single component (no flash merge)"
    rows_ = [
        ("flash", f"gid {r['flash_gid']} / TPC {r['flash_apa']} / grp {r['flash_grp']}"),
        ("flash time", f"{r['t_us']:.3f} us" + ("  (IN BEAM)" if r["in_beam"] else "")),
        ("flash PE", f"{r['pe']:.1f}  (group {r['pe_grp']:.1f})"),
        ("cross-APA partner", ptxt),
        ("main cluster", str(main) if main is not None else "(no bundle)"),
        ("companions", ", ".join(str(c) for c in comps) or "-"),
        ("npts main / bundle", f"{r['npts_main']} / {r['npts_bundle']}"),
        ("len (dominant comp.)", f"{r['len_cm']:.1f} cm"),
        ("main cluster makeup", comp_txt),
        ("TGM", verdict(r["tgm"])),
        ("STM", verdict(r["stm"])),
        ("FC (fully-contained)",
         "<span style='color:#2ca02c'>YES</span>" if r["fc"] == 1
         else verdict(r["fc"])),
        ("LM (light-mismatch)",
         {2: "<span style='color:#b06000'>LIGHT MISMATCH</span>",
          1: "low-energy (unjudgeable)", 0: "pass"}.get(r["lm"], "-")),
        ("auto label", f"<b>{r['auto_label']}</b>"
         + ("  <span style='color:#2ca02c'>+FC</span>" if r["fc"] == 1 else "")),
        ("scan labels", "+".join(state["labels"].get(key, [])) or "-"),
        ("comment", state["comments"].get(key, "") or "-"),
    ]
    pi = evt.previnfo.get(i) if evt.previnfo else None
    if pi is not None:
        if pi["row"] is None:
            rows_.append((f"prev run ({pi['name']})", "no matching bundle"))
        else:
            p = pi["row"]
            cmp_txt = ("<span style='color:#b06000'>CHANGED</span>"
                       if pi["changed"] else "same")
            if key in state["confirmed"]:
                cmp_txt += "  ✓ re-scanned"
            rows_.append((f"prev verdicts ({pi['name']})",
                          f"{p['tgm']}/{p['stm']}/{p['fc']}"
                          f" [{p['auto_label']}] — {cmp_txt}"))
        if pi["labels"]:
            rows_.append((f"prev scan labels ({pi['lab_from']})",
                          "+".join(pi["labels"])))
        if pi["comment"]:
            rows_.append((f"prev comment ({pi['cmt_from']})", pi["comment"]))
        if key in state["seeded"]:
            rows_.append(("carry-over",
                          "<i>labels/comment above copied from the previous "
                          "scan (edit or ✓ to adopt)</i>"))
    metrics.text = ("<table style='font-size:12px'>" + "".join(
        f"<tr><td style='color:#555;padding-right:8px'>{a}</td><td><b>{b}</b></td></tr>"
        for a, b in rows_) + "</table>")


def render_proj_info():
    evt = state["evt"]
    if evt is None:
        proj_info.text = ""
        return
    i = state["focus"]
    r0 = evt.rows[0] if evt.rows else None
    rse = f"run {r0['run']} subrun {r0['subrun']} event {r0['event']}" if r0 else ""
    foc = ""
    if i is not None:
        r = evt.rows[i]
        main, comps = evt.bundle_cluster_ids(r)
        parts = evt.components(main)
        foc = (f" &nbsp;&nbsp; <b>bundle</b> grp {r['flash_grp']} t={r['t_us']:.3f} us"
               f" main <span style='color:#d62728'><b>{main}</b></span>"
               + (f" (<span style='color:#ff7f0e'>+{len(parts) - 1} merge "
                  f"fragment(s), squares</span>)" if len(parts) > 1 else "")
               + (f" + companions <span style='color:#1f77b4'><b>"
                  f"{','.join(str(c) for c in comps)}</b></span>" if comps else "")
               + (" &mdash; <b style='color:#2ca02c'>FC (fully-contained)</b>"
                  if r["fc"] == 1 else "")
               + (" &mdash; <b style='color:#b06000'>LM (light mismatch)</b>"
                  if r["lm"] == 2 else ""))
    nlab = sum(1 for r in evt.rows if state["labels"].get(row_key(r)))
    ptxt = ""
    if evt.previnfo:
        chg = [i2 for i2, pi in evt.previnfo.items() if pi["changed"]]
        pend = [i2 for i2 in chg
                if row_key(evt.rows[i2]) not in state["confirmed"]]
        base = evt.previnfo[0]["name"]
        ptxt = (f", <span style='color:#b06000'><b>{len(chg)} changed</b> vs "
                f"{base}, {len(pend)} pending re-scan</span>" if chg
                else f", no verdict change vs {base}")
    proj_info.text = (f"<b>{evt.label}</b> ({rse}) &mdash; "
                      f"{len(state['order'])} bundle(s) shown "
                      f"[{'IN-BEAM ONLY' if state['inbeam_only'] else 'all'}], "
                      f"{nlab} scanned{ptxt}{foc}"
                      + (f"<br><span style='color:#b00'>{'; '.join(evt.problems)}"
                         "</span>" if evt.problems else ""))


def sync_label_widgets():
    """Push the focused row's labels/comment into the widgets (guarded)."""
    state["suppress"] = True
    i = state["focus"]
    if i is None:
        label_group.active = []
        comment_in.value = ""
    else:
        key = row_key(state["evt"].rows[i])
        labs = state["labels"].get(key, [])
        label_group.active = [SCAN_LABELS.index(l) for l in labs if l in SCAN_LABELS]
        comment_in.value = state["comments"].get(key, "")
    evt_comment_in.value = state["event_comment"]
    state["suppress"] = False


def render_dqdx():
    """STM dQ/dx-vs-residual-range panel for the focused bundle's main cluster
    (tracking-stm.root from run_nusel_evt.sh -stm-fit; empty otherwise)."""
    evt = state["evt"]
    i = state["focus"]

    # Name the bundle this panel is about, in both the title and the summary.
    # Without it a panel that failed to refresh is indistinguishable from one
    # reporting on the bundle you just clicked (which is exactly how the
    # missing render_dqdx call in on_row_select hid for a whole scan round).
    r = evt.rows[i] if (evt is not None and i is not None) else None
    tag = (f"grp {r['flash_grp']} t={r['t_us']:.3f} us, main {r['main_id']}"
           if r else "no bundle focused")
    f_dqdx.title.text = f"STM fit: dQ/dx vs residual range — {tag}"

    def blank(msg):
        stmdqdx_src.data = dict(rr=[], dqdx=[], color=[], marker=[],
                                passno=[], apa=[], chi2=[])
        stmref_src.data = dict(rr=[], mu=[], mu2=[])
        stmflat_src.data = dict(rr=[], flat=[])
        stmtraj_src.data = dict(x=[], y=[], z=[])
        stm_info.text = f"<b>{tag}</b><br><i>{msg}</i>"

    if evt.stm is None:
        blank("no STM fit dump (rerun with -stm-fit; needs uproot)" if uproot
              else "python env lacks uproot — STM panel disabled")
        return
    if i is None:
        blank("no bundle focused")
        return
    main, _ = evt.bundle_cluster_ids(evt.rows[i])
    if main is None:
        blank("row has no main cluster")
        return
    s = evt.stm
    m = s["cid"] == main
    if not m.any():
        # The 'stm fit' table column carries the reason, straight from
        # TaggerCheckSTM's "no STM fit:" DEBUG lines -- report it here instead
        # of guessing at a list of possibilities.
        why = {
            "contained": "cluster_fc_check says fully contained &rarr; no FV exit "
                         "point, so check_stm_conditions exits at Mid Point A "
                         "before any fit.  NB TaggerCheckSTM is the only tagger "
                         "given no fiducial/fv_tolerance, so this test uses the "
                         "un-inset per-face SENSITIVE volume (|x|&le;201.45 cm) "
                         "rather than TGM/FC's inset box (|x|&le;198.55) &mdash; "
                         "check the 'fc' column, and see docs/49",
            "torn": "the tagger DID log a reason but the log line was torn "
                    "mid-word by an interleaved write, so it could not be "
                    "identified (this is a logging artifact, not a verdict)",
            "midkink": "one exit point but a &gt;120&deg; mid-track kink with both "
                       "arms &gt;20 cm (Mid Point B)",
            "nexits": "not exactly one candidate exit point (Mid Point C)",
            "nosteiner": "no steiner_pc on this cluster",
            "nofidu": "no fiducialutils (MakeFiducialUtils missing from the pipeline)",
            "shortfit": "round-1 forward fit gave &le;3 points",
            "postfit": "the tagger evaluated this cluster but exited after the "
                       "round-1 fit through a path that records no pass, so no "
                       "trajectory was persisted",
            "eval": "the tagger evaluated this cluster but no trajectory was "
                    "persisted (exited after the round-1 fit)",
            "tgm": "already tagged TGM &rarr; the STM tagger skipped it",
            "-": "never evaluated by the STM tagger (out of scope?)",
        }.get(evt.rows[i].get("stmfit", ""), None)
        msg = f"cluster {main}: no STM fit"
        blank(msg + (f" &mdash; {why}" if why else
                     " recorded (rerun with -stm-fit for the reason)"))
        return

    # TPC from drift-x sign (SBND: x<0 TPC0, x>0 TPC1).
    apa = (s["x"][m] > 0).astype(int)
    passno = s["passno"][m]
    stmdqdx_src.data = dict(
        rr=s["rr"][m].tolist(), dqdx=s["dqdx"][m].tolist(),
        color=["#1f77b4" if p == 0 else "#d62728" for p in passno],
        marker=["circle" if a == 0 else "triangle" for a in apa],
        passno=passno.tolist(), apa=apa.tolist(), chi2=s["chi2"][m].tolist())
    stmtraj_src.data = dict(x=s["x"][m].tolist(), y=s["y"][m].tolist(),
                            z=s["z"][m].tolist())

    rr_max = float(max(1.0, s["rr"][m].max()))
    if REF_TAGGER in STM_REF:
        t = STM_REF[REF_TAGGER]
        t2 = STM_REF[REF_FIT] if REF_FIT else t
        rr = [t["start"] + k * t["step"] for k in range(len(t["values"]))]
        keep = [k for k, v in enumerate(rr) if v <= rr_max + 5]
        stmref_src.data = dict(rr=[rr[k] for k in keep],
                               mu=[t["values"][k] for k in keep],
                               mu2=[t2["values"][k] for k in keep])
    stmflat_src.data = dict(rr=[0, rr_max + 5], flat=[MIP_DQDX, MIP_DQDX])

    # Scalar summary: per-pass verdicts + best eval rows for this cluster.
    P, E = s["passes"], s["evals"]
    lines = []
    pm = P["cluster_id"] == main
    for k in np.nonzero(pm)[0]:
        lines.append(
            f"pass {int(P['pass'][k])} ({'fwd' if P['pass'][k]==0 else 'bwd'}): "
            f"<b>{STM_STATUS.get(int(P['status'][k]), P['status'][k])}</b>, "
            f"kink@{int(P['kink_num'][k])}, exit_L={P['exit_L'][k]:.1f} cm, "
            f"left_L={P['left_L'][k]:.1f} cm, "
            f"left dQ/dx={P['left_dqdx'][k]/MIP_DQDX:.2f} MIP")
    em = E["cluster_id"] == main
    for k in np.nonzero(em)[0]:
        lines.append(
            f"eval p{int(E['pass'][k])} "
            f"[peak {E['peak_range'][k]:.0f}/off {E['offset_length'][k]:.0f}"
            f"/com {E['com_range'][k]:.0f} cm{', strong' if E['strong'][k] else ''}]: "
            f"ks1={E['ks1'][k]:.3f} ks2={E['ks2'][k]:.3f} "
            f"r1={E['ratio1'][k]:.2f} r2={E['ratio2'][k]:.2f} "
            f"res_L={E['res_length'][k]:.1f} cm "
            f"&rarr; <b>{'PASS' if E['verdict'][k]==1 else 'fail'}</b>")
    npts = int(m.sum())
    n0, n1 = int((apa == 0).sum()), int((apa == 1).sum())
    stm_info.text = ("<b>STM fit, cluster %d</b> [%s] — %d pts "
                     "(TPC0 %d ○ / TPC1 %d △)<br>"
                     % (main, tag, npts, n0, n1) + "<br>".join(lines))


def render_focus():
    """Redraw everything that depends on state['focus'].

    Deliberately excludes rebuild_table(): that writes table_src, whose
    selected.on_change re-enters on_row_select, so folding it in here would
    make every click a callback loop.  Every focus change must go through
    THIS function -- on_row_select used to list the renderers by hand and
    silently omitted render_dqdx, which left the STM panel (and the black
    fit-trajectory crosses on the projections) showing whichever bundle was
    focused at load time.
    """
    render_projections()
    render_light()
    render_metrics()
    render_dqdx()
    render_proj_info()
    sync_label_widgets()


def refresh():
    rebuild_table()
    render_focus()


# ---------------------------------------------------------------------------
# Persistence (autosave state + final labels JSON).
# ---------------------------------------------------------------------------
def labels_dir(evt):
    d = os.path.join(evt.work_root, "nusel_labels", SCAN_TAG)
    os.makedirs(d, exist_ok=True)
    return d


def state_file(evt):
    return os.path.join(labels_dir(evt), f".scan_state-{evt.label}.json")


def save_state():
    evt = state["evt"]
    if evt is None:
        return
    try:
        with open(state_file(evt), "w") as fh:
            json.dump({"labels": state["labels"], "comments": state["comments"],
                       "event_comment": state["event_comment"],
                       "seeded": sorted(state["seeded"]),
                       "seen": sorted(state["seen"]),
                       "confirmed": sorted(state["confirmed"])}, fh)
    except OSError:
        pass


def load_state(evt):
    p = state_file(evt)
    if not os.path.isfile(p):
        return {}, {}, "", set(), set(), set()
    try:
        with open(p) as fh:
            d = json.load(fh)
        return (d.get("labels", {}), d.get("comments", {}),
                d.get("event_comment", ""),
                set(d.get("seeded", [])), set(d.get("seen", [])),
                set(d.get("confirmed", [])))
    except (OSError, ValueError):
        return {}, {}, "", set(), set(), set()


def seed_from_prev(evt):
    """Copy the previous scan's labels/comments onto matched bundles of the
    current tag -- once per key ('seen' persists, so a cleared carry-over is
    not resurrected on reload).  Old label dirs are never written (M13)."""
    if not evt.previnfo:
        return
    changed = False
    for i, r in enumerate(evt.rows):
        pi = evt.previnfo[i]
        if not (pi["labels"] or pi["comment"]):
            continue
        key = row_key(r)
        if key in state["seen"]:
            continue
        state["seen"].add(key)
        took = False
        if pi["labels"] and key not in state["labels"]:
            state["labels"][key] = list(pi["labels"])
            took = True
        if pi["comment"] and key not in state["comments"]:
            state["comments"][key] = pi["comment"]
            took = True
        if took:
            state["seeded"].add(key)
        changed = True
    if not state["event_comment"] and evt.prev_event_comment:
        state["event_comment"] = evt.prev_event_comment
        changed = True
    if changed:
        save_state()


def on_save():
    evt = state["evt"]
    if evt is None:
        return
    r0 = evt.rows[0] if evt.rows else {}
    bundles = []
    for n, r in enumerate(evt.rows):
        key = row_key(r)
        main, comps = evt.bundle_cluster_ids(r)
        pi = evt.previnfo.get(n) if evt.previnfo else None
        b = {
            "main_id": r["main_id"], "flash_gid": r["flash_gid"],
            "flash_grp": r["flash_grp"], "flash_apa": r["flash_apa"],
            "flash_time_us": r["t_us"], "flash_pe_grp": r["pe_grp"],
            "in_beam": r["in_beam"],
            "clusters": ([main] + comps) if main is not None else [],
            "auto": {"tgm": r["tgm"], "stm": r["stm"], "fc": r["fc"],
                     "lm": r["lm"], "label": r["auto_label"]},
            "scan_labels": state["labels"].get(key, []),
            "comment": state["comments"].get(key, ""),
        }
        if pi is not None:
            b["prev"] = ({"tag": pi["name"], "matched": False}
                         if pi["row"] is None else
                         {"tag": pi["name"], "matched": True,
                          "tgm": pi["row"]["tgm"], "stm": pi["row"]["stm"],
                          "fc": pi["row"]["fc"], "lm": pi["row"]["lm"],
                          "label": pi["row"]["auto_label"],
                          "changed": pi["changed"],
                          "scan_labels": pi["labels"],
                          "comment": pi["comment"]})
            b["carried_over"] = key in state["seeded"]
            b["rescan_confirmed"] = key in state["confirmed"]
        bundles.append(b)
    out = {"event": evt.label, "run": r0.get("run"), "subrun": r0.get("subrun"),
           "event_no": r0.get("event"), "source": os.path.basename(evt.tsv),
           "work_root": evt.work_root, "tag": SCAN_TAG,
           "event_comment": state["event_comment"], "bundles": bundles}
    dest = os.path.join(labels_dir(evt), f"nusel-labels-{evt.label}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    nlab = sum(1 for b in bundles if b["scan_labels"] or b["comment"])
    status.text = f"Saved {len(bundles)} bundle(s) ({nlab} annotated) -> {dest}"


# ---------------------------------------------------------------------------
# Callbacks.
# ---------------------------------------------------------------------------
def load_event(label):
    evt = get_event(label)
    build_previnfo(evt)
    state["evt"] = evt
    (state["labels"], state["comments"], state["event_comment"],
     state["seeded"], state["seen"], state["confirmed"]) = load_state(evt)
    seed_from_prev(evt)
    order = visible_rows(evt)
    state["focus"] = order[0] if order else None
    n_in = sum(1 for r in evt.rows if r["in_beam"])
    ptxt = (" Baselines: " + ", ".join(pv.name for pv in PREVS)
            + " (amber rows = verdict changed, re-scan them)." if PREVS else "")
    status.text = (f"Loaded <b>{label}</b>: {len(evt.rows)} bundle row(s), "
                   f"{n_in} in-beam; charge/light from the Bee mabc-all-apa.zip "
                   "layers. Click a row, tag with the label buttons, Save when "
                   "done." + ptxt)
    refresh()


def on_event_change(attr, old, new):
    if new:
        load_event(new)


def step_event(d):
    if not LABELS:
        return
    i = LABELS.index(event_select.value)
    event_select.value = LABELS[(i + d) % len(LABELS)]


def step_bundle(d):
    order = state["order"]
    if not order:
        return
    cur = state["focus"]
    pos = order.index(cur) if cur in order else -d
    state["focus"] = order[(pos + d) % len(order)]
    refresh()


def on_row_select(attr, old, new):
    if not new:
        return
    order = state["order"]
    pos = new[0]
    if 0 <= pos < len(order):
        state["focus"] = order[pos]
        render_focus()


def on_mode(attr, old, new):
    state["inbeam_only"] = bool(new)
    mode_btn.label = "Mode: IN-BEAM only" if new else "Mode: ALL bundles"
    mode_btn.button_type = "warning" if new else "default"
    order_before = state["focus"]
    if order_before is not None and state["evt"] is not None:
        if new and state["evt"].rows[order_before]["in_beam"] != 1:
            state["focus"] = None
    if state["focus"] is None:
        vis = visible_rows(state["evt"]) if state["evt"] else []
        state["focus"] = vis[0] if vis else None
    refresh()


def on_color(attr, old, new):
    state["color_mode"] = new
    render_projections()


def mark_touched(key):
    """Any hand edit under the current tag = this bundle was (re)scanned."""
    state["seeded"].discard(key)
    state["confirmed"].add(key)
    state["seen"].add(key)


def on_labels(attr, old, new):
    if state["suppress"] or state["focus"] is None:
        return
    r = state["evt"].rows[state["focus"]]
    key = row_key(r)
    labs = [SCAN_LABELS[k] for k in sorted(new)]
    if labs:
        state["labels"][key] = labs
    else:
        state["labels"].pop(key, None)
    mark_touched(key)
    save_state()
    rebuild_table()
    render_metrics()
    render_proj_info()


def on_clear_labels():
    if state["focus"] is None:
        return
    key = row_key(state["evt"].rows[state["focus"]])
    state["labels"].pop(key, None)
    state["comments"].pop(key, None)
    mark_touched(key)
    save_state()
    refresh()


def on_confirm():
    if state["focus"] is None:
        return
    key = row_key(state["evt"].rows[state["focus"]])
    mark_touched(key)
    save_state()
    rebuild_table()
    render_metrics()
    render_proj_info()


def on_comment(attr, old, new):
    if state["suppress"] or state["focus"] is None:
        return
    key = row_key(state["evt"].rows[state["focus"]])
    if new.strip():
        state["comments"][key] = new
    else:
        state["comments"].pop(key, None)
    mark_touched(key)
    save_state()
    rebuild_table()
    render_metrics()


def on_evt_comment(attr, old, new):
    if state["suppress"]:
        return
    state["event_comment"] = new
    save_state()


event_select.on_change("value", on_event_change)
prev_evt_btn.on_click(lambda: step_event(-1))
next_evt_btn.on_click(lambda: step_event(+1))
prev_bun_btn.on_click(lambda: step_bundle(-1))
next_bun_btn.on_click(lambda: step_bundle(+1))
table_src.selected.on_change("indices", on_row_select)
mode_btn.on_change("active", on_mode)
color_select.on_change("value", on_color)
label_group.on_change("active", on_labels)
clear_lbl_btn.on_click(on_clear_labels)
confirm_btn.on_click(on_confirm)
comment_in.on_change("value", on_comment)
evt_comment_in.on_change("value", on_evt_comment)
save_btn.on_click(on_save)


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
header = Div(text="<h2>SBND nusel (TGM/STM/FC) hand-scan</h2>", width=1200)
controls = column(
    row(event_select, prev_evt_btn, next_evt_btn, prev_bun_btn, next_bun_btn,
        mode_btn, save_btn),
    row(color_select))
layout = column(
    row(f_xy, f_yz, f_xz),
    proj_info,
    header,
    controls,
    status,
    row(table, column(label_title, label_group,
                      row(clear_lbl_btn, confirm_btn), comment_in,
                      evt_comment_in), metrics),
    row(f_dqdx, stm_info),
    row(LIGHT[0]["meas"]["fig"], LIGHT[0]["pred"]["fig"],
        LIGHT[1]["meas"]["fig"], LIGHT[1]["pred"]["fig"]),
    row(OVERLAY[0]["fig"], OVERLAY[1]["fig"]),
)

curdoc().add_root(layout)
curdoc().title = "nusel hand-scan"

if LABELS:
    load_event(LABELS[0])
else:
    status.text = ("<b>No nusel TSVs found.</b> Produce them with "
                   "run_nusel_evt.sh, then pass the work root (dir holding "
                   "nusel_evt*/ + ql_evt*/) to serve_nusel_scan.sh.")

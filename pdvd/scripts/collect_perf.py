#!/usr/bin/env python3
"""Collect per-event wall/RSS + QLtiming stats for a perf campaign tag.

Usage:
  ./scripts/collect_perf.py clus <tag>            # QL/clustering tags: work/<run6>_<idx>_<tag>/
  ./scripts/collect_perf.py light <suffix>        # light tags: work/<run6>_light<EVT><suffix>/

Emits a TSV on stdout, one row per event.
  clus columns: run idx event wall_s peak_rss_gb ql_took_ms prefit_ms
                xtpc_cull_ms fit_ms output_ms vis_loop_ms cand_pts nflash ngroups
  light columns: run event wall_s peak_rss_gb

QLtiming fields come from the wct_clus_*.log debug lines (QLMatching.cxx):
  "QLMatching timing: ident I took T ms, proc RSS ..."
  "QLtiming operator: ident I prefit P xtpc_cull X fit F output O ms"
  "QLtiming build_bundles: ident I nflash NF ngroups NG nbundles NB vis_loop V ms over C candidate-points"
build_bundles fires once per ApaRun (bottom/top): vis_loop/cand_pts/ngroups are
summed, nflash is the max (shared flash list).  The event number is taken from
the _keep calib-evt<ID>.json (perf tags run without -calib).
"""
import csv
import glob
import os
import re
import sys

PDVD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDVD_DIR, "work")

RE_RES = re.compile(r"run=(\S+) evt=(\S+) wall_s=([\d.]+) peak_rss_gb=([\d.]+)")
RE_TOOK = re.compile(r"QLMatching timing: ident \S+ took (\d+) ms")
RE_OP = re.compile(
    r"QLtiming operator: ident \S+ prefit ([\d.]+) xtpc_cull ([\d.]+) "
    r"fit ([\d.]+) output ([\d.]+) ms")
RE_BB = re.compile(
    r"QLtiming build_bundles: ident \S+ nflash (\d+) ngroups (\d+) nbundles \d+ "
    r"vis_loop ([\d.]+) ms over (\d+) candidate-points")
RE_PF = re.compile(
    r"QLtiming prefit: ident \S+ mask ([\d.]+) read_flashes ([\d.]+) "
    r"decompose ([\d.]+) geometry ([\d.]+) build_bundles ([\d.]+) "
    r"build_maps ([\d.]+) ms")


def keep_event(run6, idx):
    hits = glob.glob(os.path.join(WORK, f"{run6}_{idx}_keep", "calib-evt*.json"))
    if hits:
        m = re.search(r"calib-evt(\d+)", hits[0])
        if m:
            return m.group(1)
    return ""


def collect_clus(tag):
    rows = []
    for d in sorted(glob.glob(os.path.join(WORK, f"[0-9]*_[0-9]*_{tag}"))):
        base = os.path.basename(d)
        m = re.match(rf"(\d{{6}})_(\d+)_{re.escape(tag)}$", base)
        if not m:
            continue
        run6, idx = m.group(1), m.group(2)
        row = dict(run=run6, idx=idx, event=keep_event(run6, idx),
                   wall_s="", peak_rss_gb="", ql_took_ms="", prefit_ms="",
                   xtpc_cull_ms="", fit_ms="", output_ms="",
                   build_bundles_ms="", decompose_ms="",
                   vis_loop_ms="", cand_pts="", nflash="", ngroups="")
        res = glob.glob(os.path.join(d, "clus_resource_*.txt"))
        if res:
            rm = RE_RES.search(open(res[0]).read())
            if rm:
                row["wall_s"], row["peak_rss_gb"] = rm.group(3), rm.group(4)
        logs = glob.glob(os.path.join(d, "wct_clus_*.log"))
        if logs:
            vis = pts = ngrp = 0.0
            nfl = 0
            # wire-cell appends across reruns: keep only the LAST run's
            # markers (last "QLMatching timing" wins; build_bundles lines
            # are re-scanned from the final occurrence backwards by pairs).
            text = open(logs[0], errors="replace").read()
            took = RE_TOOK.findall(text)
            if took:
                row["ql_took_ms"] = took[-1]
            op = RE_OP.findall(text)
            if op:
                (row["prefit_ms"], row["xtpc_cull_ms"],
                 row["fit_ms"], row["output_ms"]) = op[-1]
            pf = RE_PF.findall(text)
            # last two prefit lines = bottom+top ApaRun of the last run
            if pf:
                row["build_bundles_ms"] = f"{sum(float(p[4]) for p in pf[-2:]):.1f}"
                row["decompose_ms"] = f"{sum(float(p[2]) for p in pf[-2:]):.1f}"
            bb = RE_BB.findall(text)
            # last two build_bundles lines = bottom+top ApaRun of the last run
            for nf, ng, v, c in bb[-2:]:
                vis += float(v)
                pts += int(c)
                ngrp += int(ng)
                nfl = max(nfl, int(nf))
            if bb:
                row["vis_loop_ms"] = f"{vis:.1f}"
                row["cand_pts"] = str(int(pts))
                row["nflash"] = str(nfl)
                row["ngroups"] = str(int(ngrp))
        rows.append(row)
    return rows, ["run", "idx", "event", "wall_s", "peak_rss_gb", "ql_took_ms",
                  "prefit_ms", "xtpc_cull_ms", "fit_ms", "output_ms",
                  "build_bundles_ms", "decompose_ms",
                  "vis_loop_ms", "cand_pts", "nflash", "ngroups"]


def collect_light(suffix):
    rows = []
    for d in sorted(glob.glob(os.path.join(WORK, f"[0-9]*_light*{suffix}"))):
        base = os.path.basename(d)
        m = re.match(rf"(\d{{6}})_light(\d+){re.escape(suffix)}$", base)
        if not m:
            continue
        run6, evt = m.group(1), m.group(2)
        row = dict(run=run6, event=evt, wall_s="", peak_rss_gb="")
        res = glob.glob(os.path.join(d, "light_resource_*.txt"))
        if res:
            rm = RE_RES.search(open(res[0]).read())
            if rm:
                row["wall_s"], row["peak_rss_gb"] = rm.group(3), rm.group(4)
        rows.append(row)
    return rows, ["run", "event", "wall_s", "peak_rss_gb"]


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("clus", "light"):
        sys.exit(__doc__)
    mode, tag = sys.argv[1], sys.argv[2]
    rows, cols = collect_clus(tag) if mode == "clus" else collect_light(tag)
    w = csv.DictWriter(sys.stdout, fieldnames=cols, delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    print(f"# {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()

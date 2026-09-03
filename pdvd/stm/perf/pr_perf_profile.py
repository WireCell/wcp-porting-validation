#!/usr/bin/env python3
"""doc pdvd/28: per-event running-time + memory profile of a PDVD PR arm, from
what the runner already leaves on disk (no rerun):

  work/<run6>_<idx>_<tag>/pr_resource_*.txt   wall_s (bash SECONDS, batch-contended)
                                              peak_rss_gb (VmHWM, exact)
  work/<run6>_<idx>_<tag>/wct_pr_*.log        'MABC timing: <stage> took <ms> ms'
                                              'Timer: <wall> wall-sec, <core> core-sec: (<Class>) "<name>"'
                                              'MEM: total: size=..K, res=..K increment: .. <stage>'
                                              tagger verdict lines, steiner keep line,
                                              strict-connector census lines

Writes <out>_events.tsv (one row per event) and <out>_stages.tsv (stage totals),
prints the percentile table, stage shares and per-unit costs.  Wall under a
batch is contention-inflated (doc pdvd/18 sec 4); the contention-free cost is
node_core_s (MultiAlgBlobClustering core-sec) -- quote that.

Usage: python3 stm/perf/pr_perf_profile.py --tag d27fresh [--out docs/perf/pr_d27fresh] [--png]
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict, defaultdict

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RE_TIMING = re.compile(r"MABC timing: (.+?) took ([0-9.]+) ms \(cumulative")
RE_TIMER = re.compile(r'Timer: ([0-9.]+) wall-sec, ([0-9.]+) core-sec:\s+\((\S+)\) "(\S+)"')
RE_TOTAL = re.compile(r"Timer: Total ([0-9.]+) wall-sec, ([0-9.]+) core-sec")
RE_MEM = re.compile(r"MEM: total: size=([0-9.e+]+)K, res=([0-9.e+]+)K increment: size=([-0-9.e+]+)K, res=([-0-9.e+]+)K (.*)$")
RE_STM = re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=(\d)")
RE_TGM = re.compile(r"TaggerCheckTGM: cluster (\d+) . TGM=(true|false)")
RE_CAND = re.compile(r"\[nu_per_bundle\] gid (-?\d+): candidate main cluster (\d+)")
RE_KEPT = re.compile(r"<CreateSteinerGraph:pr> CreateSteinerGraph: .*kept (\d+) of (\d+) cluster")
RE_CENSUS = re.compile(r"connect_graph_relaxed_strict: nblobs=(\d+) npoints=(\d+) ncomp=(\d+) threshold=(\d+) lazy=(true|false)")
RE_LAZY = re.compile(r"connect_graph_relaxed_strict lazy: num=(\d+) pairs=(\d+) walked=(\d+) accepted=(\d+)")
RE_RES = re.compile(r"run=(\d+) evt=(\d+) wall_s=(\d+) peak_rss_gb=([0-9.]+)")

# stage name -> short column
STAGE_COLS = OrderedDict([
    ("loaded live", "load"),
    ("ClusteringSwitchScope:pr", "scope"),
    ("ClusteringFlagMatchedMains:pr", "flag"),
    ("CreateSteinerGraph:pr", "steiner"),
    ("TaggerCheckTGM:pr", "tgm"),
    ("TaggerCheckSTM:pr", "stm"),
    ("TaggerCheckFC:pr", "fc"),
    ("ClusteringProtectBundle:pr", "protect"),
    ("CreateSteinerGraph:prrefresh", "refresh"),
    ("TaggerCheckNeutrino:pr", "nu"),
    ("PdvdPrMagnifyTrackingVisitor:pr", "w_trk"),
    ("UbooneTaggerOutputVisitor:pr", "w_tag"),
    ("PrDisplayDump:pr", "w_disp"),
    ("PdvdMagnifyTrackingVisitor:pr", "w_stm"),
    ("done", "done"),
])
MEM_COLS = OrderedDict([
    ("loaded live", "m_load"),
    ("CreateSteinerGraph:pr", "m_steiner"),
    ("TaggerCheckSTM:pr", "m_stm"),
    ("ClusteringProtectBundle:pr", "m_protect"),
    ("TaggerCheckNeutrino:pr", "m_nu"),
    ("PrDisplayDump:pr", "m_disp"),
    ("done", "m_done"),
])


def scan_log(path):
    r = dict(stages=OrderedDict(), mem=OrderedDict(), timers={}, total_wall=-1.0, total_core=-1.0,
             mains_tgm=0, tgm_true=0, mains_stm=0, stm1=0, nu_cands=0, kept=-1, ntot=-1,
             ncomp_max=0, lazy_calls=0, conn_calls=0, conn_max_pairs=0, conn_max_walked=0,
             mem_peak_res_gb=0.0)
    with open(path, errors="replace") as fp:
        for line in fp:
            m = RE_TIMING.search(line)
            if m:
                r["stages"][m.group(1)] = r["stages"].get(m.group(1), 0.0) + float(m.group(2))
                continue
            m = RE_MEM.search(line)
            if m:
                res_gb = float(m.group(2)) / 1048576.0
                r["mem"][m.group(5).strip()] = res_gb
                r["mem_peak_res_gb"] = max(r["mem_peak_res_gb"], res_gb)
                continue
            m = RE_TOTAL.search(line)
            if m:
                r["total_wall"], r["total_core"] = float(m.group(1)), float(m.group(2))
                continue
            m = RE_TIMER.search(line)
            if m:
                r["timers"][m.group(4) + ":" + m.group(3).split("::")[-1]] = (float(m.group(1)), float(m.group(2)))
                continue
            m = RE_TGM.search(line)
            if m:
                r["mains_tgm"] += 1
                r["tgm_true"] += m.group(2) == "true"
                continue
            m = RE_STM.search(line)
            if m:
                r["mains_stm"] += 1
                r["stm1"] += m.group(2) == "1"
                continue
            if RE_CAND.search(line):
                r["nu_cands"] += 1
                continue
            m = RE_KEPT.search(line)
            if m and r["kept"] < 0:
                r["kept"], r["ntot"] = int(m.group(1)), int(m.group(2))
                continue
            m = RE_CENSUS.search(line)
            if m:
                r["conn_calls"] += 1
                r["ncomp_max"] = max(r["ncomp_max"], int(m.group(3)))
                r["lazy_calls"] += m.group(5) == "true"
                continue
            m = RE_LAZY.search(line)
            if m:
                r["conn_max_pairs"] = max(r["conn_max_pairs"], int(m.group(2)))
                r["conn_max_walked"] = max(r["conn_max_walked"], int(m.group(3)))
    return r


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None, help="output prefix (default docs/perf/pr_<tag>)")
    ap.add_argument("--png", action="store_true", help="also write docs/pics/doc28_<tag>_*.png")
    a = ap.parse_args()
    out = a.out or os.path.join(PDVD, "docs", "perf", "pr_%s" % a.tag)

    dirs = sorted(glob.glob(os.path.join(PDVD, "work", "*_%s" % a.tag)))
    rows = []
    for d in dirs:
        logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
        res = glob.glob(os.path.join(d, "pr_resource_*.txt"))
        if not logs:
            continue
        base = os.path.basename(d)[: -len(a.tag) - 1]
        run, idx = base.split("_")
        r = scan_log(logs[0])
        wall_s, peak = -1, -1.0
        if res:
            m = RE_RES.search(open(res[0]).read())
            if m:
                wall_s, peak = int(m.group(3)), float(m.group(4))
        node = r["timers"].get("clus_pr:MultiAlgBlobClustering", (-1.0, -1.0))
        tfs = r["timers"].get("pr_pctree:TensorFileSource", (-1.0, -1.0))
        row = OrderedDict(run=run, idx=idx, wall_s=wall_s, peak_rss_gb=peak,
                          total_wall_s=r["total_wall"], total_core_s=r["total_core"],
                          node_wall_s=node[0], node_core_s=node[1], tfs_wall_s=tfs[0], tfs_core_s=tfs[1],
                          mem_peak_res_gb=round(r["mem_peak_res_gb"], 3),
                          mains=r["mains_tgm"], tgm=r["tgm_true"], stm_eval=r["mains_stm"], stm=r["stm1"],
                          nu_cands=r["nu_cands"], steiner_kept=r["kept"], clusters=r["ntot"],
                          conn_calls=r["conn_calls"], lazy_calls=r["lazy_calls"], ncomp_max=r["ncomp_max"],
                          conn_max_pairs=r["conn_max_pairs"], conn_max_walked=r["conn_max_walked"])
        for k, c in STAGE_COLS.items():
            row[c + "_ms"] = round(r["stages"].get(k, 0.0), 1)
        known = set(STAGE_COLS)
        row["other_ms"] = round(sum(v for k, v in r["stages"].items() if k not in known), 1)
        for k, c in MEM_COLS.items():
            row[c + "_gb"] = round(r["mem"].get(k, float("nan")), 3)
        row["_stages"] = r["stages"]
        rows.append(row)
    if not rows:
        sys.exit("no events under work/*_%s" % a.tag)

    cols = [k for k in rows[0] if not k.startswith("_")]
    with open(out + "_events.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    tot, mx, mxev = defaultdict(float), defaultdict(float), {}
    for row in rows:
        for k, v in row["_stages"].items():
            tot[k] += v
            if v > mx[k]:
                mx[k], mxev[k] = v, "%s/%s" % (row["run"], int(row["idx"]))
    total = sum(tot.values()) or 1.0
    with open(out + "_stages.tsv", "w") as fh:
        fh.write("stage\tsum_s\tshare\tmax_s\tmax_event\n")
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
            fh.write("%s\t%.1f\t%.4f\t%.1f\t%s\n" % (k, v / 1000, v / total, mx[k] / 1000, mxev[k]))

    n = len(rows)
    print("arm %s: %d events -> %s_{events,stages}.tsv" % (a.tag, n, out))
    print("%-16s %8s %8s %8s %8s %8s %8s" % ("quantity", "sum", "p25", "p50", "p90", "max", "max_ev"))
    for q in ("wall_s", "total_core_s", "node_core_s", "tfs_wall_s", "peak_rss_gb", "mem_peak_res_gb"):
        vals = [row[q] for row in rows if row[q] >= 0]
        mrow = max(rows, key=lambda r: r[q])
        print("%-16s %8.1f %8.2f %8.2f %8.2f %8.2f %s/%d" % (q, sum(vals), pct(vals, .25), pct(vals, .5), pct(vals, .9), max(vals), mrow["run"], int(mrow["idx"])))
    print("\n%-36s %9s %6s %9s %s" % ("MABC stage", "sum_s", "share", "max_s", "max_event"))
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:14]:
        print("%-36s %9.1f %5.1f%% %9.1f %s" % (k, v / 1000, 100 * v / total, mx[k] / 1000, mxev[k]))
    print("%-36s %9.1f" % ("TOTAL", total / 1000))

    kept = sum(r["steiner_kept"] for r in rows if r["steiner_kept"] > 0)
    stm_eval = sum(r["stm_eval"] for r in rows)
    cands = sum(r["nu_cands"] for r in rows)
    mains = sum(r["mains"] for r in rows)
    print("\nper-unit: steiner %.0f ms/kept cluster (%d) | STM %.0f ms/evaluated main (%d of %d mains) | "
          "TGM %.0f ms/main | neutrino %.0f ms/candidate (%d)" % (
              tot["CreateSteinerGraph:pr"] / max(kept, 1), kept,
              tot["TaggerCheckSTM:pr"] / max(stm_eval, 1), stm_eval, mains,
              tot["TaggerCheckTGM:pr"] / max(mains, 1),
              tot["TaggerCheckNeutrino:pr"] / max(cands, 1), cands))
    lazy = [r for r in rows if r["lazy_calls"]]
    print("strict connector: %d calls over the arm, %d events with a lazy (ncomp>200) call: %s" % (
        sum(r["conn_calls"] for r in rows), len(lazy),
        ", ".join("%s/%d ncomp=%d pairs=%d walked=%d protect=%.0fs" % (r["run"], int(r["idx"]), r["ncomp_max"], r["conn_max_pairs"], r["conn_max_walked"], r["protect_ms"] / 1000) for r in lazy)))

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        pics = os.path.join(PDVD, "docs", "pics")
        top = sorted(rows, key=lambda r: -r["node_core_s"])[:15]
        keys = ["steiner_ms", "tgm_ms", "stm_ms", "protect_ms", "nu_ms"]
        labels = ["CreateSteinerGraph", "TaggerCheckTGM", "TaggerCheckSTM", "ProtectBundle", "TaggerCheckNeutrino"]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(top) + 1)
        bottom = np.zeros(len(top) + 1)
        armtot = sum(r["node_wall_s"] for r in rows if r["node_wall_s"] > 0)
        for k, lab in zip(keys, labels):
            vals = np.array([r[k] / 1000 for r in top] + [tot[[s for s, c in STAGE_COLS.items() if c + "_ms" == k][0]] / 1000 / n])
            ax.bar(x, vals, bottom=bottom, label=lab)
            bottom += vals
        rest = np.array([max(r["node_wall_s"] - sum(r[k] for k in keys) / 1000, 0) for r in top] + [max(armtot / n - bottom[-1], 0)])
        ax.bar(x, rest, bottom=bottom, label="other stages", color="lightgrey")
        ax.set_xticks(x)
        ax.set_xticklabels(["%s/%d" % (r["run"][-3:], int(r["idx"])) for r in top] + ["arm mean"], rotation=60, fontsize=8)
        ax.set_ylabel("MultiAlgBlobClustering wall [s]")
        ax.set_title("PDVD PR job, arm %s: stage breakdown of the 15 costliest events (n=%d)" % (a.tag, n))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(pics, "doc28_%s_stage_shares.png" % a.tag), dpi=110)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter([r["node_core_s"] for r in rows], [r["peak_rss_gb"] for r in rows], s=14)
        ax.set_xscale("log")
        ax.set_xlabel("node core-sec (contention-free)")
        ax.set_ylabel("peak RSS [GB] (VmHWM)")
        ax.set_title("arm %s: peak RSS vs cost, n=%d" % (a.tag, n))
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pics, "doc28_%s_rss_vs_core.png" % a.tag), dpi=110)
        print("png -> %s/doc28_%s_{stage_shares,rss_vs_core}.png" % (pics, a.tag))


if __name__ == "__main__":
    main()

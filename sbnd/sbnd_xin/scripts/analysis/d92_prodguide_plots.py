#!/usr/bin/env python3
"""doc 92 -- every plot and every quoted number of the production running +
validation guide (docs/92_production-running-and-validation-guide.html).

Inputs (all committed, or copied into a committed TSV on first run):
  products/<tag>/<sample>-scores-<tag>.tsv      pr_scores_table.py output, one row per event
  work-<sample>-grp0825/g*/.{img,ql}.time.meta   stage-A per-group timecmd records (+ events.txt)
        -> copied to docs/92_prodguide/d92-stageA-timing.tsv so the plot survives arm retirement;
           when the arms are gone the TSV is read instead.
  docs/pr/pr142-perf-{mcp1k,nuecc48}.tsv         per-MABC-step medians of the PR job (pr142_perf.py)

Outputs:
  <out>/d92_*.png                                 the nine figures embedded in the HTML
  docs/92_prodguide/d92-summary.tsv               every number the guide quotes (sample, metric, value)
  docs/92_prodguide/d92-stageA-timing.tsv         the stage-A timing records

Conventions (doc 85 / doc pr/142):
  * degenerate rows (nu_evaluated==1, kine_reco_Enu==0.0 exactly, vertex (0,0,0) exactly) are cut
    before every physics distribution;
  * cosmict_flag is the cosmic verdict, cosmic_flag is a BDT input and is never used;
  * nue_score == -15 means "BDT not filled", not a score;
  * medians are the UPPER median sorted(v)[n//2], the convention of every sbnd_xin doc;
  * core_s is the comparable PR time; wall_s is PR_JOBS-way-concurrency contaminated and is
    only reported beside it.

Usage:
  scripts/analysis/d92_prodguide_plots.py --products products/prod0901b --tag prod0901b \
      --out <pngdir> [--tsvdir docs/92_prodguide] [--grp 'work-{s}-grp0825'] \
      [--perf 'docs/pr/pr142-perf-{s}.tsv' --perf-col 'work-{s}-prod0901'] [--pr-jobs 32]
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES = ["nuecc48", "ncpi0", "mcp1k", "mcp2k"]
LABEL = {"nuecc48": "nueCC-48", "ncpi0": "NCpi0-19", "mcp1k": "numu 1k", "mcp2k": "numu 2k"}
# validated categorical palette (dataviz skill, slots 1-4), fixed order, never cycled
HUE = {"nuecc48": "#2a78d6", "ncpi0": "#eb6834", "mcp1k": "#1baf7a", "mcp2k": "#eda100"}
# stages are a different entity set: their own fixed hues (slots 7, 5, 6)
STAGE_HUE = {"imaging": "#4a3aa7", "clus+QL": "#e87ba4", "PR": "#008300"}
GROUP_SIZE_DEFAULT = 16

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d9d9d4", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8a85", "axes.axisbelow": True, "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def fnum(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def umedian(v):
    """Upper median -- the convention of every sbnd_xin doc (sorted(v)[n//2])."""
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def load_scores(products, tag):
    rows = {}
    for s in SAMPLES:
        p = os.path.join(products, f"{s}-scores-{tag}.tsv")
        with open(p) as f:
            rows[s] = list(csv.DictReader(f, delimiter="\t"))
    return rows


def is_degenerate(r):
    if fnum(r["nu_evaluated"]) != 1:
        return False
    e = fnum(r["kine_reco_Enu_MeV"])
    x, y, z = (fnum(r["nu_x_cm"]), fnum(r["nu_y_cm"]), fnum(r["nu_z_cm"]))
    return e == 0.0 and x == 0.0 and y == 0.0 and z == 0.0


def clean_rows(rows):
    return [r for r in rows if fnum(r["nu_evaluated"]) == 1 and not is_degenerate(r)]


def step_hist(ax, data, bins, color, label, density=False, lw=1.8, overflow=True):
    d = np.asarray(data, dtype=float)
    if overflow and len(d):
        d = np.clip(d, bins[0], bins[-1] - 1e-9)
    ax.hist(d, bins=bins, histtype="step", color=color, lw=lw, label=label, density=density)


# ---------------------------------------------------------------- stage A timing
def read_meta(path):
    d = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            d[k] = v
    return d


def collect_stageA(grp_pattern, tsv_path):
    recs = []
    for s in SAMPLES:
        root = grp_pattern.format(s=s)
        for gdir in sorted(glob.glob(os.path.join(root, "g*"))):
            ev = os.path.join(gdir, "events.txt")
            if not os.path.exists(ev):
                continue
            with open(ev) as f:
                n = sum(1 for line in f if line.strip())
            for stage, fn in (("imaging", ".img.time.meta"), ("clus+QL", ".ql.time.meta")):
                mp = os.path.join(gdir, fn)
                if not os.path.exists(mp):
                    continue
                m = read_meta(mp)
                recs.append({"sample": s, "group": os.path.basename(gdir), "stage": stage,
                             "n_events": n, "rc": int(m.get("rc", -1)),
                             "wall_s": float(m["wall_s"]), "maxrss_kb": float(m["maxrss_kb"])})
    if recs:
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        with open(tsv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(recs)
        print(f"stage-A records: {len(recs)} from arms -> {tsv_path}")
    elif os.path.exists(tsv_path):
        with open(tsv_path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                r["n_events"] = int(r["n_events"]); r["rc"] = int(r["rc"])
                r["wall_s"] = float(r["wall_s"]); r["maxrss_kb"] = float(r["maxrss_kb"])
                recs.append(r)
        print(f"stage-A records: {len(recs)} from committed TSV {tsv_path} (arms not found)")
    else:
        print("WARNING: no stage-A timing found (no arms, no TSV); stage-A plots skipped", file=sys.stderr)
    return recs


def read_perf(perf_pattern, col_pattern):
    out = {}
    for s in ("mcp1k", "nuecc48"):
        p = perf_pattern.format(s=s)
        if not os.path.exists(p):
            continue
        col = col_pattern.format(s=s)
        steps = {}
        with open(p) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["kind"] == "step_ms" and col in r and r[col] not in ("", None):
                    steps[r["name"]] = float(r[col])
        out[s] = steps
    return out


# ------------------------------------------------- operating-point size (TLAs)
def _strip_jsonnet_comments(t):
    """Remove //, # and /* */ comments without touching string literals.

    Needed before any paren matching: the entry points' comments contain
    unbalanced parentheses, and matching on the raw text closes the signature
    hundreds of lines early (it reported 39 of 501 TLAs).
    """
    out, i, n, instr = [], 0, len(t), None
    while i < n:
        c = t[i]
        if instr:
            out.append(c)
            if c == "\\":
                out.append(t[i + 1]); i += 2; continue
            if c == instr:
                instr = None
            i += 1; continue
        if c in "\"'":
            instr = c; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n and t[i + 1] == "/":
            while i < n and t[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and t[i + 1] == "*":
            j = t.find("*/", i + 2); i = (j + 2) if j >= 0 else n; continue
        if c == "#":
            while i < n and t[i] != "\n":
                i += 1
            continue
        out.append(c); i += 1
    return "".join(out)


TRIVIAL_TLA = {"null", "''", '""', "{}", "[]", "false", "0"}


def count_tlas(path):
    """(total top-level args, args whose default is a non-trivial production value).

    'Non-trivial' = not null / '' / {} / [] / false / 0, i.e. the arguments that
    actually carry an operating point rather than a default-OFF switch.  The
    rule is stated here because the count is quoted in the guide.
    """
    t = _strip_jsonnet_comments(open(path).read())
    m = re.search(r"^function\(", t, re.M)
    if not m:
        return (0, 0)
    i, d = m.start() + len("function"), 0
    for k in range(i, len(t)):
        if t[k] == "(":
            d += 1
        elif t[k] == ")":
            d -= 1
            if d == 0:
                break
    args, depth, cur = [], 0, ""
    for ch in t[i + 1:k]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur); cur = ""
        else:
            cur += ch
    args.append(cur)
    args = [a.strip() for a in args if a.strip()]
    nont = [a for a in args if "=" in a and a.split("=", 1)[1].strip() not in TRIVIAL_TLA]
    return (len(args), len(nont))


# ------------------------------------------------- stage-A per-event Q/L wall
RE_STATUS = re.compile(r"rc=(\d+) evt=(\d+) wall=(\d+)s")


def collect_qlarm(pattern):
    """Per-event Q/L wall from a per-event stage-A arm's .status/<evt> records.

    The group-mode .{img,ql}.time.meta records are per GROUP of 16; a per-event
    stage-A arm (d97_ql_arm.sh) writes one .status line per event instead.  The
    two are not comparable -- group mode amortises the ~5 s configure cost over
    16 events -- so this is reported beside the group number, never instead of
    it.  Integer seconds, and contention-contaminated by the arm's own
    concurrency: it is a wall, not a core time, and there is no RSS record.
    """
    out = {}
    for s in SAMPLES:
        root = pattern.format(s=s)
        w, bad = [], 0
        for f in glob.glob(os.path.join(root, ".status", "*")):
            m = RE_STATUS.search(open(f).read())
            if not m:
                continue
            if m.group(1) != "0":
                bad += 1
            w.append(int(m.group(3)))
        if w:
            out[s] = {"n": len(w), "rc_nonzero": bad, "median_s": umedian(w),
                      "p90_s": sorted(w)[int(0.9 * len(w))], "max_s": max(w),
                      "sum_h": round(sum(w) / 3600.0, 2)}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="PNG output dir")
    ap.add_argument("--tsvdir", default="docs/92_prodguide")
    ap.add_argument("--grp", default="work-{s}-grp0825")
    ap.add_argument("--perf", default="docs/pr/pr142-perf-{s}.tsv")
    ap.add_argument("--perf-col", default="work-{s}-prod0901")
    ap.add_argument("--pr-jobs", default="32", help="PR_JOBS the arm ran at (labels the wall axis)")
    ap.add_argument("--cfg", default=None,
                    help="SBND cfg dir holding wct-{pr,clus-matching}-perevt.jsonnet; "
                         "counts the operating point's TLAs into the summary")
    ap.add_argument("--qlarm", default=None,
                    help="per-event stage-A arm pattern, e.g. 'work-{s}-d97fv'; "
                         "reads .status/<evt> for the per-event Q/L wall")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tsvdir, exist_ok=True)

    rows = load_scores(a.products, a.tag)
    summary = []  # (sample, metric, value)

    def put(s, k, v):
        summary.append((s, k, v))

    clean, evaluated = {}, {}
    for s in SAMPLES:
        R = rows[s]
        ev = [r for r in R if fnum(r["nu_evaluated"]) == 1]
        deg = [r for r in ev if is_degenerate(r)]
        C = clean_rows(R)
        evaluated[s], clean[s] = ev, C
        put(s, "n_events", len(R))
        put(s, "n_rc_nonzero", sum(1 for r in R if fnum(r["rc"]) != 0))
        put(s, "n_evaluated", len(ev))
        put(s, "n_degenerate", len(deg))
        put(s, "n_clean", len(C))
        labels = {}
        for r in R:
            labels[r["event_label"]] = labels.get(r["event_label"], 0) + 1
        for k in sorted(labels):
            put(s, f"label:{k}", labels[k])
        put(s, "n_cosmict_flag_0", sum(1 for r in C if fnum(r["cosmict_flag"]) == 0))
        put(s, "n_dl_warn", sum(1 for r in R if fnum(r["dl_warn"]) not in (None, 0)))
        enu = [fnum(r["kine_reco_Enu_MeV"]) for r in C]
        put(s, "enu_min_MeV", f"{min(enu):.1f}")
        put(s, "enu_median_MeV", f"{umedian(enu):.1f}")
        put(s, "enu_max_MeV", f"{max(enu):.1f}")
        numu = [fnum(r["numu_score"]) for r in C]
        nue_all = [fnum(r["nue_score"]) for r in C]
        nue = [v for v in nue_all if v > -15]
        put(s, "numu_median", f"{umedian(numu):.3f}")
        put(s, "n_numu_gt_0.9", sum(1 for v in numu if v > 0.9))
        put(s, "n_nue_filled", len(nue))
        put(s, "nue_median_filled", f"{umedian(nue):.2f}" if nue else "nan")
        put(s, "n_nue_gt_7.0", sum(1 for v in nue if v > 7.0))
        put(s, "n_nue_gt_0.7", sum(1 for v in nue if v > 0.7))
        exc = [fnum(r["kine_energy_excluded_MeV"]) / fnum(r["kine_reco_Enu_MeV"]) * 100
               for r in C if fnum(r["kine_energy_excluded_MeV"]) is not None and fnum(r["kine_reco_Enu_MeV"]) > 0]
        put(s, "excluded_frac_median_pct", f"{umedian(exc):.2f}" if exc else "nan")
        core = [fnum(r["core_s"]) for r in R if fnum(r["core_s"]) is not None]
        wall = [fnum(r["wall_s"]) for r in R if fnum(r["wall_s"]) is not None]
        twall = [fnum(r["timecmd_wall_s"]) for r in R if fnum(r["timecmd_wall_s"]) is not None]
        rss = [fnum(r["maxrss_kb"]) / 1048576.0 for r in R if fnum(r["maxrss_kb"]) is not None]
        put(s, "pr_core_s_median", f"{umedian(core):.2f}")
        put(s, "pr_core_s_p90", f"{pct(core, 90):.2f}")
        put(s, "pr_core_s_max", f"{max(core):.2f}")
        put(s, "pr_core_h_sum", f"{sum(core)/3600:.2f}")
        put(s, f"pr_wall_s_median@PR_JOBS={a.pr_jobs}", f"{umedian(wall):.2f}")
        put(s, f"pr_procwall_s_median@PR_JOBS={a.pr_jobs}", f"{umedian(twall):.0f}")
        put(s, "pr_maxrss_GiB_median", f"{umedian(rss):.2f}")
        put(s, "pr_maxrss_GiB_p90", f"{pct(rss, 90):.2f}")
        put(s, "pr_maxrss_GiB_max", f"{max(rss):.2f}")

    allR = [r for s in SAMPLES for r in rows[s]]
    allC = [r for s in SAMPLES for r in clean[s]]
    put("all", "n_events", len(allR))
    put("all", "n_evaluated", sum(len(evaluated[s]) for s in SAMPLES))
    put("all", "n_degenerate", sum(1 for s in SAMPLES for r in evaluated[s] if is_degenerate(r)))
    put("all", "n_clean", len(allC))
    for k in ("nu-candidate", "cosmic-tagged", "no-bundle", "no-beam-flash"):
        put("all", f"label:{k}", sum(1 for r in allR if r["event_label"] == k))
    core_all = [fnum(r["core_s"]) for r in allR]
    rss_all = [fnum(r["maxrss_kb"]) / 1048576.0 for r in allR]
    put("all", "pr_core_s_median", f"{umedian(core_all):.2f}")
    put("all", "pr_core_h_sum", f"{sum(core_all)/3600:.2f}")
    put("all", "pr_maxrss_GiB_median", f"{umedian(rss_all):.2f}")
    put("all", "pr_maxrss_GiB_p90", f"{pct(rss_all, 90):.2f}")
    put("all", "pr_maxrss_GiB_max", f"{max(rss_all):.2f}")

    # ------------------------------------------------------------ fig 1: Enu
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.arange(0, 4001, 100)
    for s in SAMPLES:
        enu = [fnum(r["kine_reco_Enu_MeV"]) for r in clean[s]]
        step_hist(axs[0], enu, bins, HUE[s], f"{LABEL[s]} (n={len(enu)})", density=True)
        step_hist(axs[1], enu, bins, HUE[s], None)
    axs[0].set_xlabel("kine_reco_Enu [MeV]  (last bin = overflow)"); axs[0].set_ylabel("area-normalised")
    axs[1].set_xlabel("kine_reco_Enu [MeV]  (last bin = overflow)"); axs[1].set_ylabel("events"); axs[1].set_yscale("log")
    axs[0].legend(loc="upper right")
    fig.suptitle(f"Reconstructed neutrino energy, {a.tag} (evaluated, degenerate rows cut)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_enu.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 2: BDT
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.arange(-6, 7.01, 0.4)
    for s in SAMPLES:
        numu = [fnum(r["numu_score"]) for r in clean[s]]
        step_hist(axs[0], numu, bins, HUE[s], f"{LABEL[s]} (n={len(numu)})", density=True)
    axs[0].axvline(0.9, color="#555", ls="--", lw=1); axs[0].text(0.95, axs[0].get_ylim()[1]*0.95, "0.9", color="#555", fontsize=8)
    axs[0].set_xlabel("numu_score"); axs[0].set_ylabel("area-normalised"); axs[0].legend(loc="upper left")
    bins = np.arange(-12, 16.01, 1.0)
    for s in SAMPLES:
        nue_all = [fnum(r["nue_score"]) for r in clean[s]]
        nue = [v for v in nue_all if v > -15]
        step_hist(axs[1], nue, bins, HUE[s], f"{LABEL[s]} filled {len(nue)}/{len(nue_all)}")
    axs[1].axvline(7.0, color="#555", ls="--", lw=1); axs[1].text(7.1, axs[1].get_ylim()[1]*0.9, "7.0", color="#555", fontsize=8)
    axs[1].set_xlabel("nue_score  (rows at -15 = BDT not filled, excluded)"); axs[1].set_ylabel("events"); axs[1].legend(loc="upper left")
    fig.suptitle(f"BDT scores, {a.tag} (evaluated, degenerate rows cut)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_bdt.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 3: labels + cosmict
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    cats = ["nu-candidate", "cosmic-tagged", "no-bundle", "no-beam-flash"]
    # ordinal-like shades of one neutral hue family for the label categories (identity is written on the bar)
    cat_col = {"nu-candidate": "#2a78d6", "cosmic-tagged": "#9b9b95", "no-bundle": "#c9c9c4", "no-beam-flash": "#e6e6e2"}
    y = np.arange(len(SAMPLES))[::-1]
    for i, s in enumerate(SAMPLES):
        n = len(rows[s]); left = 0.0
        for c in cats:
            k = sum(1 for r in rows[s] if r["event_label"] == c)
            frac = k / n
            axs[0].barh(y[i], frac, left=left, color=cat_col[c], edgecolor="white", lw=1.5, height=0.62)
            if frac > 0.06:
                axs[0].text(left + frac / 2, y[i], f"{k}", ha="center", va="center", fontsize=8,
                            color="white" if c == "nu-candidate" else "#333")
            left += frac
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{LABEL[s]} (n={len(rows[s])})" for s in SAMPLES])
    axs[0].set_xlim(0, 1); axs[0].set_xlabel("fraction of events, nusel event_label"); axs[0].grid(False)
    from matplotlib.patches import Patch
    axs[0].legend([Patch(color=cat_col[c]) for c in cats], cats, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    for i, s in enumerate(SAMPLES):
        C = clean[s]; k = sum(1 for r in C if fnum(r["cosmict_flag"]) == 0)
        axs[1].barh(y[i], k / len(C), color=HUE[s], height=0.62)
        axs[1].text(k / len(C) + 0.01, y[i], f"{k}/{len(C)}", va="center", fontsize=8)
    axs[1].set_yticks(y); axs[1].set_yticklabels([]); axs[1].set_xlim(0, 1.25); axs[1].grid(False)
    axs[1].set_xlabel("cosmict_flag == 0\n(clean evaluated events)")
    fig.suptitle(f"nusel labels and cosmic verdict, {a.tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_labels.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 4: vertex
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    spec = (("nu_x_cm", np.arange(-220, 221, 20), "nu_x [cm]"),
            ("nu_y_cm", np.arange(-220, 221, 20), "nu_y [cm]"),
            ("nu_z_cm", np.arange(-20, 541, 20), "nu_z [cm]"))
    for ax, (k, bins, xl) in zip(axs, spec):
        for s in SAMPLES:
            v = [fnum(r[k]) for r in clean[s]]
            step_hist(ax, v, bins, HUE[s], (LABEL[s] + f" (n={len(v)})") if k == "nu_x_cm" else None, density=True, overflow=False, lw=1.4 if s == "ncpi0" else 1.8)
        ax.set_xlabel(xl); ax.set_yticks([])
    axs[0].set_ylabel("area-normalised"); axs[0].legend(loc="upper center", fontsize=7.5)
    fig.suptitle(f"Reconstructed neutrino vertex (T_tagger nu_x/y/z), {a.tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_vertex.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 5: excluded energy
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.arange(0, 30.5, 1.0)
    for s in SAMPLES:
        exc = [fnum(r["kine_energy_excluded_MeV"]) / fnum(r["kine_reco_Enu_MeV"]) * 100
               for r in clean[s] if fnum(r["kine_energy_excluded_MeV"]) is not None and fnum(r["kine_reco_Enu_MeV"]) > 0]
        step_hist(ax, exc, bins, HUE[s], f"{LABEL[s]} median {umedian(exc):.1f}%", density=True)
    ax.set_xlabel("kine_energy_excluded / kine_reco_Enu [%]  (last bin = overflow)"); ax.set_ylabel("area-normalised")
    ax.set_yscale("log"); ax.legend()
    ax.set_title(f"Energy excluded from the Enu sum (doc 85 r2 quantity), {a.tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_excluded.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 6: PR cost
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    bins = np.logspace(np.log10(0.3), np.log10(200), 40)
    for s in SAMPLES:
        core = [fnum(r["core_s"]) for r in rows[s]]
        step_hist(axs[0], core, bins, HUE[s], f"{LABEL[s]} median {umedian(core):.1f} s", overflow=False)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("PR job core-seconds per event (Timer: Total)"); axs[0].set_ylabel("events"); axs[0].legend(fontsize=8)
    bins = np.arange(0.2, 1.71, 0.05)
    for s in SAMPLES:
        rss = [fnum(r["maxrss_kb"]) / 1048576.0 for r in rows[s]]
        step_hist(axs[1], rss, bins, HUE[s], f"{LABEL[s]} median {umedian(rss):.2f} GiB", overflow=False)
    axs[1].set_xlabel("PR job peak RSS per event [GiB]  (timecmd RUSAGE_CHILDREN)"); axs[1].set_ylabel("events"); axs[1].set_yscale("log"); axs[1].legend(fontsize=8)
    fig.suptitle(f"Stage B (PR job) cost per event, {a.tag}, per-event mode, PR_JOBS={a.pr_jobs}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_pr_cost.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 7: stage A
    recs = collect_stageA(a.grp, os.path.join(a.tsvdir, "d92-stageA-timing.tsv"))
    stageA = {}
    if recs:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        for stage in ("imaging", "clus+QL"):
            rr = [r for r in recs if r["stage"] == stage and r["n_events"] > 0]
            per_evt = [r["wall_s"] / r["n_events"] for r in rr]
            rss = [r["maxrss_kb"] / 1048576.0 for r in rr]
            stageA[stage] = {"n_groups": len(rr), "s_per_evt_median": umedian(per_evt), "s_per_evt_max": max(per_evt),
                             "rss_median": umedian(rss), "rss_max": max(rss), "sum_wall_s": sum(r["wall_s"] for r in rr),
                             "n_events": sum(r["n_events"] for r in rr)}
            step_hist(axs[0], per_evt, np.arange(0, 12.01, 0.4), STAGE_HUE[stage],
                      f"{stage}: median {umedian(per_evt):.1f} s/evt, {len(rr)} groups", overflow=True)
            step_hist(axs[1], rss, np.arange(0.3, 1.31, 0.03), STAGE_HUE[stage], f"{stage}: median {umedian(rss):.2f} GiB", overflow=False)
            put("all", f"stageA_{stage}_s_per_evt_median", f"{umedian(per_evt):.2f}")
            put("all", f"stageA_{stage}_s_per_evt_max", f"{max(per_evt):.2f}")
            put("all", f"stageA_{stage}_rss_GiB_median", f"{umedian(rss):.2f}")
            put("all", f"stageA_{stage}_rss_GiB_max", f"{max(rss):.2f}")
            put("all", f"stageA_{stage}_sum_wall_h", f"{sum(r['wall_s'] for r in rr)/3600:.2f}")
            put("all", f"stageA_{stage}_n_groups", len(rr))
        axs[0].set_xlabel("process wall per event [s]  (group wall / events in group)"); axs[0].set_ylabel("groups"); axs[0].legend(fontsize=8, loc="upper right")
        axs[1].set_xlabel("process peak RSS [GiB]  (one process = one group of 16)"); axs[1].set_ylabel("groups"); axs[1].legend(fontsize=8)
        fig.suptitle("Stage A cost (reco1 dump excluded), group mode --size 16, arms work-*-grp0825, SBND_MAX_JOBS-way concurrent", fontsize=9.5)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_stageA_cost.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 8: PR steps
    perf = read_perf(a.perf, a.perf_col)
    if perf:
        skip = {"done", "pre clustering", "start clustering"}
        names = sorted({n for s in perf for n in perf[s] if n not in skip},
                       key=lambda n: -max(perf[s].get(n, 0) for s in perf))[:12]
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        y = np.arange(len(names))[::-1]
        h = 0.38
        for j, s in enumerate(("mcp1k", "nuecc48")):
            if s not in perf:
                continue
            vals = [perf[s].get(n, 0) for n in names]
            ax.barh(y + (h / 2 if j == 0 else -h / 2), vals, height=h, color=HUE[s], label=f"{LABEL[s]} ({a.perf_col.format(s=s)})")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.set_xscale("log")
        ax.set_xlabel("median ms per event  (MABC timing: <step>, from pr142_perf.py)"); ax.legend(loc="lower right")
        ax.set_title("Inside the PR job: per-step median cost (data-loading and Bee dump steps included)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_pr_steps.png")); plt.close(fig)

    # ------------------------------------------------------------ fig 9: stage summary
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    stages = ["imaging", "clus+QL", "PR"]
    t_med = [stageA.get("imaging", {}).get("s_per_evt_median", np.nan), stageA.get("clus+QL", {}).get("s_per_evt_median", np.nan), umedian(core_all)]
    t_max = [stageA.get("imaging", {}).get("s_per_evt_max", np.nan), stageA.get("clus+QL", {}).get("s_per_evt_max", np.nan), pct(core_all, 99)]
    m_med = [stageA.get("imaging", {}).get("rss_median", np.nan), stageA.get("clus+QL", {}).get("rss_median", np.nan), umedian(rss_all)]
    m_max = [stageA.get("imaging", {}).get("rss_max", np.nan), stageA.get("clus+QL", {}).get("rss_max", np.nan), max(rss_all)]
    x = np.arange(3)
    for ax, med, mx, yl, note in ((axs[0], t_med, t_max, "seconds per event", "bar = median; tick = max (stage A) / p99 (PR)"),
                                  (axs[1], m_med, m_max, "peak RSS per process [GiB]", "bar = median; tick = max")):
        ax.bar(x, med, color=[STAGE_HUE[s] for s in stages], width=0.55)
        ax.scatter(x, mx, marker="_", s=400, color="#333", zorder=3)
        for i, (v, w) in enumerate(zip(med, mx)):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
            ax.text(i + 0.32, w, f"{w:.2f}", ha="left", va="center", fontsize=7.5, color="#333")
        ax.set_xticks(x); ax.set_xticklabels(["imaging\n(group/16)", "clustering+Q/L\n(group/16)", "PR\n(per event, core-s)"]) ; ax.set_ylabel(yl); ax.set_title(note, fontsize=8.5, color="#555")
    fig.suptitle(f"Per-event cost by stage, {a.tag} production (3067 events)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "d92_stage_summary.png")); plt.close(fig)

    # ------------------------------------------------------------ operating-point size
    if a.cfg:
        for fn, key in (("wct-pr-perevt.jsonnet", "pr"), ("wct-clus-matching-perevt.jsonnet", "ql")):
            fp = os.path.join(a.cfg, fn)
            if os.path.exists(fp):
                n_all, n_val = count_tlas(fp)
                put("cfg", f"n_tla_{key}", n_all)
                put("cfg", f"n_tla_{key}_with_value", n_val)

    # ------------------------------------------------------------ per-event stage-A Q/L
    if a.qlarm:
        for s, d in collect_qlarm(a.qlarm).items():
            for k, v in d.items():
                put(s, f"qlarm_{k}", v)

    # ------------------------------------------------------------ summary tsv
    sp = os.path.join(a.tsvdir, "d92-summary.tsv")
    with open(sp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["sample", "metric", "value"])
        for row in summary:
            w.writerow(row)
    print(f"summary: {len(summary)} rows -> {sp}")
    for s, k, v in summary:
        print(f"{s:8s} {k:36s} {v}")


if __name__ == "__main__":
    main()

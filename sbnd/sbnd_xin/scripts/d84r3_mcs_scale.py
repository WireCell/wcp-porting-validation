#!/usr/bin/env python3
"""doc 84 round 3 -- MCS absolute-scale study (read-only harvest).

Harvests the three MCS log sentinels from per-event PR-chain logs
(`pr_evt*/wct_pr_evt*.log`, present in both per-event and group-mode arms --
group mode slices its logs per event):

  main       `mcs: source=... ke_MCS=... (nsegs14=.. bad_path=.. cathode_drop=..)`
             (MuonMCSDriver.cxx main INFO sentinel)
  comparator `mcs: chain comparator nseg_chain=.. len_chain=..cm ke_range_chain=.. MeV seg_id=..`
             (doc 84 round 1 P5; restrict to 'chain comparator' -- the
             kine_long_muon: DEBUG line also carries a nseg_chain= token)
  bridged    `mcs: bridged members added nseg_bridged=.. len_bridged=..cm`
             (doc 84 round 3 mcs_bridged_members)

Joins them per event (comparator by seg_id), dedups events across arms by
(sample, event) with first-listed-arm priority, and writes:

  d84r3-mcs-scale.tsv       one row per main sentinel
  d84r3-mcs-scale.png       ke_MCS vs ke_range_toolkit scatter + ratio hists
  summary.txt               counts, null-chain rate, ratio medians/quantiles

Usage:
  scripts/d84r3_mcs_scale.py --arms DIR:SAMPLE [--arms DIR:SAMPLE ...] --out OUTDIR

Read-only over the arms; OUTDIR must be fresh (M13).
"""

import argparse
import glob
import os
import re
import sys

SENT = re.compile(
    r"mcs: source=(?P<source>\S+) nseg=(?P<nseg>\d+) npoints=(?P<npoints>\d+) "
    r"len=(?P<len>[\d.]+)cm seg_id=(?P<segid>-?\d+) cluster=(?P<cid>-?\d+) -> "
    r"ke_MCS=(?P<ke>-?[\d.]+) MeV amb=(?P<amb>[\d.eE+-]+) tracklen=(?P<tracklen>-?[\d.]+)cm "
    r"ke_range=(?P<kerange>-?[\d.]+) MeV ke_range_toolkit=(?P<kert>-?[\d.]+) MeV "
    r"ke_dqdx_toolkit=(?P<kedq>-?[\d.]+) MeV "
    r"\(nsegs14=(?P<nsegs14>-?\d+) bad_path=(?P<badpath>\w+) "
    r"cathode_drop=(?P<cdrop>\d+)/(?P<cmask>\d+)\)")
CHAIN = re.compile(
    r"mcs: chain comparator nseg_chain=(?P<nseg>\d+) len_chain=(?P<len>[\d.]+)cm "
    r"ke_range_chain=(?P<ke>-?[\d.]+) MeV seg_id=(?P<segid>-?\d+)")
BRIDGED = re.compile(
    r"mcs: bridged members added nseg_bridged=(?P<nseg>\d+) len_bridged=(?P<len>[\d.]+)cm")

COLS = ("sample event arm segid source nseg npoints len_cm ke_mcs amb tracklen_cm "
        "ke_range_mcs ke_range_toolkit ke_dqdx_toolkit nsegs14 bad_path cathode_drop "
        "cathode_mask nseg_chain len_chain ke_range_chain nseg_bridged len_bridged").split()


def harvest_event(logpath):
    mains, chains, bridged = [], {}, []
    with open(logpath, errors="replace") as fp:
        for line in fp:
            m = SENT.search(line)
            if m:
                mains.append(m.groupdict())
                continue
            m = CHAIN.search(line)
            if m:
                chains[m.group("segid")] = m.groupdict()
                continue
            m = BRIDGED.search(line)
            if m:
                bridged.append(m.groupdict())
    return mains, chains, bridged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", action="append", required=True,
                    metavar="DIR:SAMPLE", help="arm dir and its sample tag; first listed wins dedup")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=False)  # fresh tag only (M13)

    rows, seen = [], set()
    per_arm = {}
    for spec in a.arms:
        adir, sample = spec.rsplit(":", 1)
        arm = os.path.basename(os.path.normpath(adir))
        nlog = nsent = nnew = 0
        for logpath in sorted(glob.glob(os.path.join(adir, "pr_evt*", "wct_pr_evt*.log"))):
            evt = re.search(r"pr_evt(\d+)", logpath).group(1)
            nlog += 1
            key = (sample, evt)
            if key in seen:
                continue
            mains, chains, bridged = harvest_event(logpath)
            if not mains:
                continue
            seen.add(key)
            nnew += 1
            for i, m in enumerate(mains):
                ch = chains.get(m["segid"], {})
                br = bridged[i] if i < len(bridged) else {}
                rows.append([sample, evt, arm, m["segid"], m["source"], m["nseg"],
                             m["npoints"], m["len"], m["ke"], m["amb"], m["tracklen"],
                             m["kerange"], m["kert"], m["kedq"], m["nsegs14"],
                             m["badpath"], m["cdrop"], m["cmask"],
                             ch.get("nseg", "-1"), ch.get("len", "-1"), ch.get("ke", "-1"),
                             br.get("nseg", "0"), br.get("len", "0")])
                nsent += 1
        per_arm[arm] = (nlog, nnew, nsent)

    tsv = os.path.join(a.out, "d84r3-mcs-scale.tsv")
    with open(tsv, "w") as fp:
        fp.write("\t".join(COLS) + "\n")
        for r in rows:
            fp.write("\t".join(r) + "\n")

    # ---- analysis ----
    def f(r, k):
        return float(r[COLS.index(k)])

    good = [r for r in rows if f(r, "ke_mcs") > 0 and f(r, "ke_range_toolkit") > 0
            and r[COLS.index("bad_path")] == "false"]
    chain_sel = [r for r in good if int(r[COLS.index("nseg_chain")]) > 0]
    pf_fb = [r for r in good if int(r[COLS.index("nseg_chain")]) <= 0]
    with_bridge = [r for r in good if int(r[COLS.index("nseg_bridged")]) > 0]

    def ratios(rs):
        return sorted(f(r, "ke_mcs") / f(r, "ke_range_toolkit") for r in rs)

    def q(xs, p):
        return xs[int(p * (len(xs) - 1))] if xs else float("nan")

    lines = ["doc 84 round 3 MCS scale harvest", ""]
    for arm, (nlog, nnew, nsent) in per_arm.items():
        lines.append("arm %-40s logs=%d new_events=%d sentinels=%d" % (arm, nlog, nnew, nsent))
    nulls = sum(1 for r in rows if int(r[COLS.index("nseg_chain")]) <= 0)
    lines += ["",
              "total sentinel rows: %d (distinct events: %d)" % (len(rows), len(seen)),
              "null-chain comparator rate: %d/%d = %.1f%%" % (nulls, len(rows), 100.0 * nulls / len(rows) if rows else 0),
              "usable (ke>0, bad_path=false): %d   chain-selected: %d   pf-fallback: %d   bridged: %d"
              % (len(good), len(chain_sel), len(pf_fb), len(with_bridge)), ""]
    for name, rs in (("chain-selected", chain_sel), ("pf-fallback", pf_fb), ("all", good),
                     ("bridged", with_bridge)):
        xs = ratios(rs)
        lines.append("ke_MCS/ke_range %-14s n=%-4d median=%.3f q25=%.3f q75=%.3f"
                     % (name, len(xs), q(xs, 0.5), q(xs, 0.25), q(xs, 0.75)))
    summary = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "summary.txt"), "w").write(summary)
    print(summary)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for rs, color, label in ((chain_sel, "tab:blue", "chain-selected"),
                             (pf_fb, "tab:orange", "pf-fallback")):
        ax.scatter([f(r, "ke_range_toolkit") for r in rs], [f(r, "ke_mcs") for r in rs],
                   s=12, alpha=0.6, color=color, label="%s (n=%d)" % (label, len(rs)))
    if with_bridge:
        ax.scatter([f(r, "ke_range_toolkit") for r in with_bridge],
                   [f(r, "ke_mcs") for r in with_bridge],
                   s=60, facecolors="none", edgecolors="red", label="bridged (n=%d)" % len(with_bridge))
    lim = max([1] + [f(r, "ke_range_toolkit") for r in good] + [f(r, "ke_mcs") for r in good]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("ke_range_toolkit [MeV] (CSDA range, selected muon)")
    ax.set_ylabel("ke_MCS [MeV]")
    ax.set_title("MCS vs range KE")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = [0.05 * i for i in range(0, 61)]
    for rs, color, label in ((chain_sel, "tab:blue", "chain-selected"),
                             (pf_fb, "tab:orange", "pf-fallback")):
        ax.hist(ratios(rs), bins=bins, histtype="step", color=color,
                label="%s med=%.3f" % (label, q(ratios(rs), 0.5)))
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("ke_MCS / ke_range_toolkit")
    ax.set_ylabel("events")
    ax.set_title("MCS/range ratio")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter([f(r, "len_cm") for r in good],
               [f(r, "ke_mcs") / f(r, "ke_range_toolkit") for r in good],
               s=12, alpha=0.6, color="tab:green")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("selected muon length [cm]")
    ax.set_ylabel("ke_MCS / ke_range_toolkit")
    ax.set_title("ratio vs track length")

    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "d84r3-mcs-scale.png"), dpi=130)
    print("wrote %s (%d rows) + png" % (tsv, len(rows)))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/72 round 2 -- offline parser for examine_structure_3's
WCT_ES3_MERGE_CENSUS log lines (ES3CENSUS / ES3MERGE / ES3PB, emitted by
clus/src/NeutrinoStructureExaminer.cxx).

Walks one or more pr_evt<ID>/wct_pr_evt<ID>.log files under a set of sample
work dirs and writes three TSVs: one row per degree-2 junction evaluated
(census.tsv, merged and declined alike), one row per junction production
actually merged (merges.tsv, ground truth), and one row per protected-break
skip (pb.tsv). All downstream threshold-fitting/grid-scan work reads these
TSVs -- no reconstruction re-run needed to try a different predicate.

Usage:
    python3 es3_census.py --sample nuecc48=work-pr72-cen48 \\
                           --sample ncpi0=work-pr72-cen19 \\
                           --sample mcp1k50=work-pr72-cen50 \\
                           --out-prefix es3_census

Writes <out-prefix>_census.tsv, <out-prefix>_merges.tsv, <out-prefix>_pb.tsv
in the current directory (or --out-dir).
"""
import argparse
import glob
import os
import re
import sys

RADII_CM = [2, 3, 5, 10, 15, 20]

POINT_RE = re.compile(r"(\w+)=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
RSCAN_RE = re.compile(
    r"R(\d+): ang=(\S+) nA1=(\d+) dA1=(\S+) nA2=(\d+) dA2=(\S+)"
)
KV_RE = re.compile(r"(\w+)=(-?\d+\.?\d*)(?![\d(:])")

CENSUS_COLS = (
    ["sample", "event", "call", "sweep", "clus"]
    + ["vtx_x", "vtx_y", "vtx_z", "v1_x", "v1_y", "v1_z", "v2_x", "v2_y", "v2_z"]
    + ["vfit", "v1fit", "v2fit"]
    + ["deg1", "deg2", "pb1", "pb2", "ngv", "nge"]
    + ["len1", "len2", "dlen1", "dlen2", "dev1", "dev2"]
    + ["nfit1", "nfit2", "nwcp1", "nwcp2"]
    + ["q1", "q2", "nq1", "nq2"]
    + ["qn1", "qn2", "nqn1", "nqn2", "qf1", "qf2", "nqf1", "nqf2"]
)
for _r in RADII_CM:
    CENSUS_COLS += [f"ang_R{_r}", f"nA1_R{_r}", f"dA1_R{_r}", f"nA2_R{_r}", f"dA2_R{_r}"]
CENSUS_COLS += ["ang10", "ang3", "pass10", "pass3", "predmerge"]

MERGE_COLS = (
    ["sample", "event", "call", "sweep", "clus"]
    + ["vtx_x", "vtx_y", "vtx_z", "v1_x", "v1_y", "v1_z", "v2_x", "v2_y", "v2_z"]
    + ["ang10", "ang3", "merged_nwcp"]
)

PB_COLS = ["sample", "event", "call", "sweep", "clus", "vtx_x", "vtx_y", "vtx_z"]


def parse_points(line):
    """Return {key: (x,y,z)} for every `key=(x,y,z)` token, and the line
    with those tokens stripped (so the plain key=value scan below doesn't
    also try to match the numbers inside the parens)."""
    pts = {}
    def _sub(m):
        pts[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        return ""
    stripped = POINT_RE.sub(_sub, line)
    return pts, stripped


def parse_rscan(line):
    """Return {radius_cm: {ang, nA1, dA1, nA2, dA2}}, and the line with
    those "R<n>: ..." blocks stripped."""
    scans = {}
    def _sub(m):
        r = int(m.group(1))
        scans[r] = {
            "ang": float(m.group(2)),
            "nA1": int(m.group(3)),
            "dA1": float(m.group(4)),
            "nA2": int(m.group(5)),
            "dA2": float(m.group(6)),
        }
        return ""
    stripped = RSCAN_RE.sub(_sub, line)
    return scans, stripped


def parse_kv(line):
    """Plain key=value scan for whatever remains after points/R-scan blocks
    are stripped out. Values that look like ints stay ints (call/sweep/clus/
    deg*/pb*/ngv/nge/vfit*/nfit*/nwcp*/nq*/pass*/predmerge/merged_nwcp);
    everything else is float."""
    out = {}
    int_keys = {
        "call", "sweep", "clus", "vfit", "v1fit", "v2fit", "deg1", "deg2",
        "pb1", "pb2", "ngv", "nge", "nfit1", "nfit2", "nwcp1", "nwcp2",
        "nq1", "nq2", "nqn1", "nqn2", "nqf1", "nqf2", "pass10", "pass3",
        "predmerge", "merged_nwcp",
    }
    for m in KV_RE.finditer(line):
        k, v = m.group(1), m.group(2)
        out[k] = int(v) if k in int_keys else float(v)
    return out


def parse_es3_line(line):
    """Dispatch on the tag (ES3CENSUS / ES3MERGE / ES3PB) at the start of
    the message body (after spdlog's "[time] D [logger] " prefix). Returns
    (tag, row_dict) or (None, None) if this isn't a census line."""
    idx = line.find("ES3CENSUS ")
    tag = "ES3CENSUS"
    if idx < 0:
        idx = line.find("ES3MERGE ")
        tag = "ES3MERGE"
    if idx < 0:
        idx = line.find("ES3PB ")
        tag = "ES3PB"
    if idx < 0:
        return None, None
    body = line[idx + len(tag) + 1:].rstrip("\n")

    pts, body = parse_points(body)
    scans, body = parse_rscan(body)
    kv = parse_kv(body)

    row = dict(kv)
    for key, (x, y, z) in pts.items():
        row[f"{key}_x"], row[f"{key}_y"], row[f"{key}_z"] = x, y, z
    for r, vals in scans.items():
        row[f"ang_R{r}"] = vals["ang"]
        row[f"nA1_R{r}"] = vals["nA1"]
        row[f"dA1_R{r}"] = vals["dA1"]
        row[f"nA2_R{r}"] = vals["nA2"]
        row[f"dA2_R{r}"] = vals["dA2"]
    return tag, row


def event_id_from_path(log_path):
    # .../pr_evt<ID>/wct_pr_evt<ID>.log
    base = os.path.basename(log_path)
    m = re.search(r"wct_pr_evt(\d+)\.log$", base)
    if m:
        return m.group(1)
    m = re.search(r"pr_evt(\d+)", log_path)
    return m.group(1) if m else "?"


def write_tsv(path, cols, rows):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"wrote {path}: {len(rows)} rows", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="append", required=True,
                     help="name=work_dir, repeatable (e.g. nuecc48=work-pr72-cen48)")
    ap.add_argument("--out-prefix", default="es3_census")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    census_rows, merge_rows, pb_rows = [], [], []
    n_logs = 0
    for spec in args.sample:
        name, work_dir = spec.split("=", 1)
        pattern = os.path.join(work_dir, "pr_evt*", "wct_pr_evt*.log")
        logs = sorted(glob.glob(pattern))
        if not logs:
            print(f"WARNING: no logs matched {pattern}", file=sys.stderr)
        for log_path in logs:
            n_logs += 1
            evt = event_id_from_path(log_path)
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    if "ES3CENSUS" not in line and "ES3MERGE" not in line and "ES3PB" not in line:
                        continue
                    tag, row = parse_es3_line(line)
                    if tag is None:
                        continue
                    row["sample"] = name
                    row["event"] = evt
                    if tag == "ES3CENSUS":
                        census_rows.append(row)
                    elif tag == "ES3MERGE":
                        merge_rows.append(row)
                    elif tag == "ES3PB":
                        pb_rows.append(row)

    print(f"parsed {n_logs} logs", file=sys.stderr)

    # doc pr/72 round 2: a very long single-line ES3CENSUS message can be
    # split across a non-atomic write() when another thread's log line
    # flushes concurrently (confirmed empirically: mcp1k50 evt57903 clus14
    # vtx=(-35.31,-35.38,226.09) -- the tail of the line, from dev2= onward,
    # was replaced mid-write by an unrelated "TaggerCheckNeutrino: match_isFC"
    # line from a different logger). This is a log-writer race, not a bug in
    # the C++ census logic or in production's actual merge decision -- the
    # corresponding ES3MERGE line (a separate, shorter write) for the same
    # junction was NOT corrupted and independently confirms ang10=9.651,
    # ang3=4.641, both comfortably inside the merge thresholds. Detect and
    # drop truncated rows here rather than silently feeding blank fields
    # into the grid scan.
    complete_census = [r for r in census_rows if "predmerge" in r]
    truncated_census = [r for r in census_rows if "predmerge" not in r]
    if truncated_census:
        print(f"WARNING: dropped {len(truncated_census)}/{len(census_rows)} "
              f"ES3CENSUS row(s) truncated by concurrent log writes (missing "
              f"trailing fields) -- see comment above", file=sys.stderr)
    census_rows = complete_census
    truncated_vtx_keys = {
        (r["sample"], r["event"], r["clus"], round(r["vtx_x"], 2), round(r["vtx_y"], 2), round(r["vtx_z"], 2))
        for r in truncated_census
    }

    os.makedirs(args.out_dir, exist_ok=True)
    write_tsv(os.path.join(args.out_dir, f"{args.out_prefix}_census.tsv"), CENSUS_COLS, census_rows)
    write_tsv(os.path.join(args.out_dir, f"{args.out_prefix}_merges.tsv"), MERGE_COLS, merge_rows)
    write_tsv(os.path.join(args.out_dir, f"{args.out_prefix}_pb.tsv"), PB_COLS, pb_rows)

    # Self-check (V4): every ES3MERGE must coordinate-match (0.01 cm
    # quantization) an ES3CENSUS line with ang10<18 and ang3<27 and
    # predmerge=1. Report any violation loudly -- this would mean the
    # census's independent recomputation disagrees with what production
    # actually did, which should be impossible by construction.
    def qkey(row, prefix):
        return (row["sample"], row["event"], row["clus"],
                round(row[f"{prefix}_x"], 2), round(row[f"{prefix}_y"], 2), round(row[f"{prefix}_z"], 2))

    census_by_vtx = {}
    for row in census_rows:
        census_by_vtx.setdefault(qkey(row, "vtx"), []).append(row)

    n_bad, n_known_corrupt = 0, 0
    for mrow in merge_rows:
        key = qkey(mrow, "vtx")
        matches = census_by_vtx.get(key, [])
        ok = any(
            c.get("ang10", -1) < 18 and c.get("ang3", -1) < 27 and c.get("predmerge") == 1
            for c in matches
        )
        if not ok:
            if key in truncated_vtx_keys:
                n_known_corrupt += 1
                continue
            n_bad += 1
            print(f"V4 SELF-CHECK FAILED: {mrow['sample']} evt{mrow['event']} clus{mrow['clus']} "
                  f"vtx={key[3:]}: no matching ES3CENSUS predmerge=1 line "
                  f"({len(matches)} candidates)", file=sys.stderr)
    if n_bad == 0:
        print(f"V4 self-check PASSED: all {len(merge_rows) - n_known_corrupt} "
              f"ES3MERGE rows with an intact census counterpart coordinate-"
              f"match a predmerge=1 ES3CENSUS row "
              f"({n_known_corrupt} excluded as known log-write corruption)", file=sys.stderr)
    else:
        print(f"V4 self-check: {n_bad}/{len(merge_rows)} ES3MERGE rows FAILED "
              f"to match for an UNEXPLAINED reason (excluding "
              f"{n_known_corrupt} known-corrupted)", file=sys.stderr)


if __name__ == "__main__":
    main()

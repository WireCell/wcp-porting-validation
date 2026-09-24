#!/usr/bin/env python3
"""doc sbnd_xin/123 round 3 -- the evidence sheet of one event for adjudicating a mover.

Collects, for one event of a sample, everything the campaign has on it:
  * the PR candidate rows of both arms (products/d123/<sample>_<arm>/candidates.tsv): scores, Enu, vertex;
  * the Q/L-level cluster moves (r3_ql_compare.py --tsv) and the rescue lines of both arms;
  * the beam-window flashes of both sources (r1_census.py --tsv of the hits arm): reco1 vs hit flashes,
    with the class of every hit flash reco1 did not have;
  * the PR main-bundle rows (nusel-table.tsv) of both arms.

usage: r3_evidence.py <sample> <event> [--arms base,hits] [--r3 r3.tsv] [--r1 r1.tsv]
"""
import argparse, csv, glob, os, re, subprocess, sys

SX = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"


def rows_where(path, pred, delim="\t"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f, delimiter=delim) if pred(r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample"); ap.add_argument("event", type=int)
    ap.add_argument("--arms", default="base,hits")
    ap.add_argument("--r3"); ap.add_argument("--r1")
    a = ap.parse_args()
    ev = a.event; arms = a.arms.split(",")
    print("=" * 100); print("event %d  sample %s" % (ev, a.sample)); print("=" * 100)
    for arm in arms:
        p = os.path.join(SX, "products/d123/%s_%s/candidates.tsv" % (a.sample, arm))
        rs = rows_where(p, lambda r: int(r["event"]) == ev)
        print("\n[%s] PR candidates: %d" % (arm, len(rs)))
        for r in rs:
            print("   numu=%s nue=%s Enu=%s vtx=(%s,%s,%s) cluster_id=%s sel=%s len=%s cm fc=%s flav=%s ccnc=%s"
                  % (r["numu_score"], r["nue_score"], r["reco_Enu"], r["nu_x"], r["nu_y"], r["nu_z"],
                     r["cluster_id"], r["sel_cluster_id"], r.get("sel_length_cm", ""), r.get("sel_fc", ""),
                     r.get("t_flav", ""), r.get("t_ccnc", "")))
        nt = os.path.join(SX, "work-%s-d123%spr/nusel-table.tsv" % (a.sample, arm))
        if os.path.exists(nt):
            print("   PR bundles (main_id flash_time_us flash_pe in_beam len_cm label):")
            with open(nt) as f:
                hdr = None
                for ln in f:
                    parts = ln.split()
                    if hdr is None:
                        hdr = parts; continue
                    d = dict(zip(hdr, parts))
                    if d.get("event") == str(ev):
                        print("      %6s %10s %9s %s %8s %s" % (d["main_id"], d["flash_time_us"], d["flash_pe"], d["in_beam"], d["len_main_cm"], d["label"]))
    if a.r3:
        rs = rows_where(a.r3, lambda r: int(r["event"]) == ev and r["class"] != "same")
        print("\nQ/L cluster changes (A -> B): %d" % len(rs))
        for r in rs:
            print("   an%s id%-4s %-6s A t=%9s pe=%9s pred=%8s | B t=%10s pe=%9s pred=%8s beam %s->%s"
                  % (r["anode"], r["ident"], r["class"], r["A_t_us"], r["A_pe"], r["A_pred"], r["B_t_us"], r["B_pe"], r["B_pred"], r["A_beam"], r["B_beam"]))
    for arm in arms:
        for lg in glob.glob(os.path.join(SX, "work-%s-d123%s/g*/wct_ql.log" % (a.sample, arm))):
            out = subprocess.run(["grep", "-c", "clus_all_apa> loading tensor set ident=%d " % ev, lg], capture_output=True, text=True).stdout.strip()
            if out and out != "0":
                res = subprocess.run(["bash", "-c", "awk '/clus_all_apa> loading tensor set ident=/{on=($0 ~ /ident=%d /)} on && /rescue round/' %s" % (ev, lg)], capture_output=True, text=True).stdout
                print("\n[%s] rescue lines: %s" % (arm, res.strip() if res.strip() else "none"))
    if a.r1:
        rs = rows_where(a.r1, lambda r: int(r["event"]) == ev and (
            (r["reco1_t_us"] not in ("nan", "") and -2 <= float(r["reco1_t_us"]) <= 12) or
            (r["ours_t_us"] not in ("nan", "") and -2 <= float(r["ours_t_us"]) <= 12)))
        print("\nflashes -2..12 us (census of the hits arm; class of each hit flash):")
        for r in sorted(rs, key=lambda r: (int(r["tpc"]), float(r["ours_t_us"]) if r["ours_t_us"] != "nan" else float(r["reco1_t_us"]))):
            print("   tpc%s %-9s reco1 t=%9s pe=%9s | hit t=%9s pe=%9s" % (r["tpc"], r["class"], r["reco1_t_us"], r["reco1_pe"], r["ours_t_us"], r["ours_pe"]))


if __name__ == "__main__":
    main()

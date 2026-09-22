#!/usr/bin/env python3
"""list_candidates.py -- one line per STM candidate over the whole release.

    python3 scripts/list_candidates.py [release_dir] [--stm-only] [--michel-only] [--csv out.csv]

Columns: run, event, idx, cluster, is_stm, michel_found, reject bits (names), muon length [cm],
muon KE by range / dQdx / MCS [MeV], Michel KE (region estimator = production) / best [MeV],
matched flash id, flash time [us, light axis], flash total PE.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr

BR = ["cluster_id", "is_stm", "michel_found", "reject_bits", "muon_len", "muon_ke_range", "muon_ke_dqdx",
      "muon_ke_mcs", "michel_ke_q2d_region", "michel_ke_best", "michel_conn_type", "flash_id", "flash_time_us",
      "flash_total_pe", "run", "event"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--stm-only", action="store_true", help="only is_stm == 1")
    ap.add_argument("--michel-only", action="store_true", help="only is_stm == 1 and michel_found == 1")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    rows = []
    for path in sr.event_files(rel):
        ev = sr.open_event(path)
        if ev["T_stm"].num_entries == 0:
            continue
        info = sr.event_info(ev)
        c = sr.candidates(ev, BR)
        for i in range(len(c["cluster_id"])):
            if a.stm_only and c["is_stm"][i] != 1:
                continue
            if a.michel_only and not (c["is_stm"][i] == 1 and c["michel_found"][i] == 1):
                continue
            r = {k: c[k][i] for k in BR}
            r["idx"] = info["event_index"]
            r["reject"] = "+".join(sr.reject_names(r["reject_bits"])) or "-"
            rows.append(r)
    hdr = f"{'run':>6} {'event':>7} {'idx':>3} {'clus':>4} {'stm':>3} {'mic':>3} {'reject':22} {'len_cm':>7} {'KEr':>7} {'KEq':>7} {'KEmcs':>7} {'Michel':>7} {'Mbest':>7} {'flash':>5} {'t_us':>9} {'PE':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['run']:6d} {r['event']:7d} {r['idx']:3d} {r['cluster_id']:4d} {r['is_stm']:3d} {r['michel_found']:3d} {r['reject'][:22]:22} "
              f"{r['muon_len']:7.1f} {r['muon_ke_range']:7.1f} {r['muon_ke_dqdx']:7.1f} {r['muon_ke_mcs']:7.1f} "
              f"{r['michel_ke_q2d_region']:7.1f} {r['michel_ke_best']:7.1f} {r['flash_id']:5d} {r['flash_time_us']:9.1f} {r['flash_total_pe']:8.0f}")
    n_stm = sum(1 for r in rows if r["is_stm"] == 1)
    n_mic = sum(1 for r in rows if r["is_stm"] == 1 and r["michel_found"] == 1)
    print(f"# {len(rows)} candidates, {n_stm} stopping muons (is_stm), {n_mic} with a Michel")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()

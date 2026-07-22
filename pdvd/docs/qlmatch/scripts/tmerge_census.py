#!/usr/bin/env python3
"""Doc 26 step 3: census of every flash_tail_merge action across the 18-evt
scan set (run 039252, evts 298567..298805 step 14) -- compare the production
`_keep` light tags against the merge-ON `_tmerge` tags.

For each event reports:
  MERGE  - a `_keep` flash absorbed into an earlier seed: dt, PE ratio,
           wide(>=1us)+seed-lit PE fraction (the gate quantities), so a
           genuine-pile-up absorption (low wide-lit fraction) would stand out.
  RESCUE - a `_tmerge` flash whose hits are all flash_id=-1 in `_keep`:
           sub-quality-cut fragment pairs reunited past min_total_pe/пds.
Anything not matching either pattern is flagged ANOMALY.

Run from pdvd/docs/qlmatch:  python3 scripts/tmerge_census.py
"""
import io
import json
import os
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CFG = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")
EVENTS = list(range(298567, 298806, 14))
FIRED_PE = 0.5          # C++ refine_fired_pe: OpDet counts as lit in the seed
TAIL_MIN_WIDTH = 1000.0  # ns, C++ tail_min_width_us default


def opch_map():
    j = json.load(open(os.path.join(CFG, "pdvd-opch-map.json")))
    return {c["opch"]: c["opdet"] for c in j["channels"]}


def load(ev, suffix):
    path = os.path.join(PDVD, "work", f"039252_light{ev}_{suffix}",
                        "opflash_pdvd-wct.tar.gz")
    with tarfile.open(path) as tf:
        op = np.load(io.BytesIO(tf.extractfile(f"opflash_tensor_{ev}_0_array.npy").read()))
        oh = np.load(io.BytesIO(tf.extractfile(f"opflash_tensor_{ev}_2_array.npy").read()))
    return op, oh


def main():
    ch2od = opch_map()
    tot = {"merge": 0, "rescue": 0, "anomaly": 0}
    for ev in EVENTS:
        opk, ohk = load(ev, "keep")
        opm, ohm = load(ev, "tmerge")
        tk, tm = opk[:, 0], opm[:, 0]
        set_m = set(np.round(tm, 3).tolist())
        set_k = set(np.round(tk, 3).tolist())
        gone = [i for i, t in enumerate(np.round(tk, 3)) if t not in set_m]
        new = [i for i, t in enumerate(np.round(tm, 3)) if t not in set_k]
        print(f"== evt {ev}: keep {len(opk)} -> tmerge {len(opm)} flashes "
              f"({len(gone)} gone, {len(new)} new) ==")

        for gi in gone:  # expect: absorbed into a seed (MERGE)
            gh = ohk[ohk[:, 7] == gi]
            if not len(gh):
                print(f"  ANOMALY: gone flash t={tk[gi]/1e3:.3f} has no hits")
                tot["anomaly"] += 1
                continue
            # locate the tmerge flash holding this flash's first hit
            r0 = gh[0]
            mk = (ohm[:, 0] == r0[0]) & (np.abs(ohm[:, 1] - r0[1]) < 0.5)
            sid = int(ohm[mk][0, 7]) if mk.any() and ohm[mk][0, 7] >= 0 else None
            if sid is None:
                print(f"  ANOMALY: gone flash t={tk[gi]/1e3:.3f} "
                      f"pe={opk[gi,1:41].sum():.0f} hits unassigned in tmerge")
                tot["anomaly"] += 1
                continue
            seed_t = tm[sid]
            # seed-lit OpDets from the KEEP seed flash (pre-merge state)
            ksid = int(np.argmin(np.abs(tk - seed_t)))
            lit = set(np.nonzero(opk[ksid, 1:41] >= FIRED_PE)[0].tolist())
            pe_j = opk[gi, 1:41].sum()
            wide_lit = sum(r[5] for r in gh
                           if r[2] >= TAIL_MIN_WIDTH
                           and ch2od.get(int(r[0]), -1) in lit)
            frac = wide_lit / pe_j if pe_j > 0 else 0.0
            ratio = pe_j / max(opk[ksid, 1:41].sum(), 1e-9)
            dt = (tk[gi] - seed_t) / 1e3
            flag = "" if frac >= 0.7 and 0 < dt <= 3.0 and ratio <= 1.0 \
                else "  <-- CHECK"
            print(f"  MERGE  t={tk[gi]/1e3:9.3f} pe={pe_j:8.1f} -> seed "
                  f"t={seed_t/1e3:9.3f} pe={opk[ksid,1:41].sum():8.1f} | "
                  f"dt={dt:+.3f}us ratio={ratio:.2f} wide-lit={frac:.3f}{flag}")
            tot["merge"] += 1
            if flag:
                tot["anomaly"] += 1

        for ni in new:  # expect: reunited sub-cut fragments (RESCUE)
            nh = ohm[ohm[:, 7] == ni]
            offs = []
            for r in nh:
                mk = (ohk[:, 0] == r[0]) & (np.abs(ohk[:, 1] - r[1]) < 0.5)
                offs.append(int(ohk[mk][0, 7]) if mk.any() else None)
            all_unassigned = all(o is not None and o < 0 for o in offs)
            pe = opm[ni, 1:41].sum()
            span = (nh[:, 1].max() - nh[:, 1].min()) / 1e3
            tag = "RESCUE" if all_unassigned else "ANOMALY"
            print(f"  {tag} t={tm[ni]/1e3:9.3f} pe={pe:8.1f} "
                  f"nhit={len(nh)} span={span:.2f}us "
                  f"(keep flash_ids {sorted(set(offs), key=str)})")
            tot["rescue" if all_unassigned else "anomaly"] += 1
        print()
    print(f"TOTAL: {tot['merge']} merges, {tot['rescue']} rescues, "
          f"{tot['anomaly']} anomalies over {len(EVENTS)} events")


if __name__ == "__main__":
    main()

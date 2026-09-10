#!/usr/bin/env python3
"""doc pdvd/67.  Item-for-item check that an arm run with ks_margin=m / michel_range_energy_dis_cm=dis
lands exactly where the offline re-verdict of the baseline arm said it would, and that
NOTHING else in the verdict moved (the single-consumer claim).
Usage: confirm.py <baseline prep> <arm prep> <m> <dis>"""
import sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
rec = C.load_record(); SF = 1 << 3
base, arm, m, dis = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
PB, _ = C.load_payloads(base, rec); PA, _ = C.load_payloads(arm, rec)
keys = sorted(k for k in rec if k in PB and k in PA)
EXPECT_MOVE = {"is_stm", "reject_bits", "reject_names", "michel_conn_type", "michel_found", "n_michel_range_veto", "pass"}
bad_stm = bad_mf = 0; other = {}
for k in keys:
    b, a = PB[k]["verdict"], PA[k]["verdict"]
    bits = int(b["reject_bits"] or 0)
    p_stm = 0 if bits & ~SF else (1 if not (bits & SF) or (b["ks_flat"] - b["ks_mu"]) > m else 0)
    veto = b["michel_conn_type"] in (2, 3) and b["michel_dis_cm"] > dis and b["michel_ke_best"] < 10.0
    p_mf = 0 if veto else int(b["michel_found"])
    if p_stm != int(a["is_stm"]): bad_stm += 1; print("is_stm MISMATCH", k, "pred", p_stm, "arm", a["is_stm"], "bits", bits, "->", a["reject_bits"])
    if p_mf != int(a["michel_found"]): bad_mf += 1; print("michel_found MISMATCH", k, "pred", p_mf, "arm", a["michel_found"])
    for f in b:
        if f in EXPECT_MOVE: continue
        if b[f] != a.get(f): other.setdefault(f, []).append(k)
print("common items %d; is_stm mismatches %d; michel_found mismatches %d" % (len(keys), bad_stm, bad_mf))
print("other verdict fields that moved: %s" % ({f: len(v) for f, v in other.items()} or "none"))
for f, v in list(other.items())[:8]: print("   %s: %s" % (f, v[:6]))

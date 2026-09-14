#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- own100x: the objects the readout-edge exemption tags that no owner record judges (owner look,
not blind; d100/prereg_round2.md sec 2 adoption rule).

    python3 d100_exempt_scan_set.py --out /home/xqian/tmp/p100/xscan \
        --arm p100bx  /home/xqian/tmp/p100/exempt_unjudged_p100bx.tsv  /home/xqian/tmp/p100/prep_p100bx \
        --arm p100bxp /home/xqian/tmp/p100/exempt_unjudged_p100bxp.tsv /home/xqian/tmp/p100/prep_p100bxp

Input lists come from d100r2_exempt.py --unjudged-out; payloads from prep_stm_michel_scan.py --det pdvd --arm <arm>
(scratch).  An object on the production-side arm that matches a listed gain-side object by geometry is shown once, on the
gain side (the configuration going to production), with both keys named.  Writes <out>/{prep/, manifest.tsv,
questions.json, items.tsv}; refuses an existing <out>, an existing own100x label dir, or two items sharing a key.
"""
import argparse, csv, json, os, sys
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99_match as M

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
LABELS = IMG + "/pdvd/work/stm_michel_labels/own100x"
DESC = {"p100bx": "gain-flip candidate (top gain 0.889, v7 wires, C 0.8630) + the exemption",
        "p100bxp": "production (gain OFF, v5 wires, C 0.7941) + the exemption"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", nargs=3, action="append", required=True, metavar=("ARM", "TSV", "PREP"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    if os.path.exists(LABELS):
        sys.exit(f"REFUSING: {LABELS} exists (M13)")
    items = []
    for arm, tsv, prep in a.arm:
        for r in csv.DictReader(open(tsv), delimiter="\t"):
            items.append(dict(r, arm=arm, prep=prep, other=None))
    first = a.arm[0][0]
    kept = [it for it in items if it["arm"] == first]
    for it in items:
        if it["arm"] == first:
            continue
        evt, cid = it["key"].split("/")
        sb, ab = M.load_bee(evt, it["arm"]), M.load_bee(evt, first)
        hit = None
        if sb is not None and ab is not None:
            pts = sb[0][sb[1] == int(cid)]
            if len(pts):
                m = M.match_cluster(pts, ab[0], ab[1], cKDTree(ab[0]))
                if M.status_of(m) == "ok":
                    hit = next((k for k in kept if k["arm"] == first and k["key"] == f"{evt}/{m['best']}"), None)
        if hit:
            hit["other"] = f"{it['arm']} {it['key']}"
        else:
            kept.append(it)
    keys = [it["key"] for it in kept]
    if len(set(keys)) != len(keys):
        sys.exit(f"two items share a key across arms: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    kept.sort(key=lambda it: (int(it["key"].split("_")[0]), int(it["key"].split("_")[1].split("/")[0]), int(it["key"].split("/")[1])))

    os.makedirs(a.out + "/prep")
    os.symlink(os.path.join(a.arm[0][2], "dqdx_ref_pdvd.json"), a.out + "/prep/dqdx_ref_pdvd.json")
    man, q, rows = [], {}, []
    for i, it in enumerate(kept, 1):
        evt, cid = it["key"].split("/")
        src = f"{it['prep']}/smprep-{evt}-c{cid}.json"
        if not os.path.exists(src):
            sys.exit(f"missing payload {src}")
        os.symlink(src, f"{a.out}/prep/smprep-{evt}-c{cid}.json")
        p = json.load(open(src))
        man.append(f"{i}\t1\t{evt}\t{cid}\t{p.get('npts')}\t{p.get('muon_len_cm')}")
        v = p["verdict"]
        # which edge (advisor, 2026-09-13): own100 measured only run 039349's LATE edge of its 6400-tick frame
        run, t = int(evt.split("_")[0]), float(it["stop_tick"])
        nt = 6400 if run == 39349 else 10000
        it["edge"] = "early edge (frame start)" if t < 60 else f"late edge of a {nt}-tick frame"
        html = (f"<b>own100x &mdash; what the readout-edge exemption brings back</b> (doc pdvd/100 round 2; not blind)<br>"
                f"Shown on <b>{it['arm']}</b> ({DESC.get(it['arm'], it['arm'])}){' &mdash; also ' + it['other'] if it['other'] else ''}."
                f"<ul style='margin:2px 0 2px 0'><li>the readout-edge guard fired on this stop (tick {it['stop_tick']}, "
                f"<b>{it['edge']}</b>) and the tag stands only because the chain found a Michel "
                f"({float(v.get('michel_ke_best') or 0):.1f} MeV, {float(v.get('michel_len') or 0):.1f} cm)</li>"
                f"<li>existing record: {it['record_verdict'] or 'none'} ({it['record_source']})</li></ul>"
                f"<b>Please judge:</b> a muon that <b>stops inside the frame</b> (STM + MICHEL / STM), or a track <b>the readout "
                f"window cut</b> (THRU)? UNCLEAR if the frame edge makes it undecidable.")
        q[it["key"]] = dict(html=html)
        rows.append("\t".join([str(i), it["key"], it["arm"], it["vol"], it["stop_tick"], it["edge"], it["michel_ke"],
                               it["record_verdict"], it["record_source"], it["other"] or ""]))
    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write("# doc pdvd/100 round 2 -- own100x, readout-edge exemption gains, owner look (not blind)\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n" + "\n".join(man) + "\n")
    json.dump(dict(scan="pdvd own100x, what the readout-edge exemption brings back (doc pdvd/100 round 2)",
                   prep=a.out + "/prep", items=q), open(a.out + "/questions.json", "w"), indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\tarm\tvol\tstop_tick\tedge\tmichel_ke\trecord_verdict\trecord_source\talso\n" + "\n".join(rows) + "\n")
    print(f"items {len(kept)} (of {len(items)} listed; shown once when both arms name the object)")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

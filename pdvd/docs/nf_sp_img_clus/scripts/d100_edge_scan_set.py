#!/usr/bin/env python3
"""doc pdvd/100 step A -- the owner's look at run 039349's STM tags the real readout window removed (not blind; owner
2026-09-13: "we do not need blind scan").

    python3 d100_edge_scan_set.py --out /home/xqian/tmp/p100/scan

The real window (doc pdvd/99 sec 4.5, production since wcp f52a374e) untags, on run 039349:
  * production  p96vprod is_stm -> p99wflip not is_stm
  * latest      p98vonq  is_stm -> p99rwon  not is_stm
Both lists are re-derived here from the arms (d99_swap_scan_set.rows_of, the record carry's matcher).  A production object
that matches a latest one by geometry is shown ONCE, on the latest arm (the configuration going to production), with both
keys named.  Each item's question carries the guard's own log line (stop tick, ticks before the 6400-tick frame end) and
every existing record verdict for either key (smx record on production keys; sw99 + the latest carried record on latest
keys).

No prep is run: the payloads are the doc 99 sec 6.4 preps (prep_stm_michel_scan.py --ctx-cells, scratch) and are symlinked
into <out>/prep with their dqdx reference.  Writes <out>/{prep/, manifest.tsv, questions.json, items.tsv}; refuses an
existing <out>.
"""
import argparse, glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
import d99_match as M
import d99_swap_scan_set as W

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WORK = IMG + "/pdvd/work"
SCAN = IMG + "/pdvd/docs/scan"
P99 = "/home/xqian/tmp/p99scan"
NTICKS = 6400
GUARD_TICKS = 60
RECORDS = {
    "smx": glob.glob(SCAN + "/pdvd_stm_michel_smx1a_*smx9_verdicts.json"),
    "sw99": [SCAN + "/pdvd_stm_michel_sw99_verdicts.json"],
    "latest": [SCAN + "/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json"],
}
PAIRS = {"production": ("p96vprod", "p99wflip"), "latest": ("p98vonq", "p99rwon")}


def removed(A, B, jobs):
    ev = [f"039349_{i}" for i in range(84)]
    with ProcessPoolExecutor(jobs) as ex:
        rows = [r for rr in ex.map(W.rows_of, [(e, A, B) for e in ev]) for r in rr]
        back = [r for rr in ex.map(W.rows_of, [(e, B, A) for e in ev]) for r in rr]
    return [r for r in rows if r["status"] != "is_stm"], [r for r in back if r["status"] != "is_stm"]


def guard_line(evt, arm, cid):
    for lg in sorted(glob.glob(f"{WORK}/{evt}_{arm}/wct_pr_*.log")):
        for line in open(lg, errors="replace"):
            m = re.search(r"readout_edge_guard: cluster (\d+) rejected: (.*)", line)
            if m and int(m.group(1)) == cid:
                return m.group(2).strip()
    return None


def sheet_rows(arm):
    out = {}
    for line in open(f"{P99}/sheet_{arm}/pdvd_stm_michel_scan_sheet.tsv"):
        if line.startswith("#") or line.startswith("scan_id"):
            continue
        f = line.rstrip("\n").split("\t")
        out[f"{f[2]}/{f[3]}"] = dict(npts=f[4], muon_len_cm=f[5])
    return out


def verdicts(keys_by_rec):
    out = []
    for rec, key in keys_by_rec:
        it = REC[rec].get(key)
        if it:
            out.append(f"{rec} {key}: {it.get('verdict')} ({it.get('confidence')})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists (a scan set is never rebuilt in place)")
    for k, v in RECORDS.items():
        if len(v) != 1:
            sys.exit(f"record {k}: expected one file, found {v}")
    global REC
    REC = {k: {it["key"]: it for it in json.load(open(v[0]))} for k, v in RECORDS.items()}

    lists = {}
    for side, (A, B) in PAIRS.items():
        rows, back = removed(A, B, a.jobs)
        print(f"{side}: {A} is_stm not is_stm on {B} (run 039349): {len(rows)}; reverse: {len(back)}")
        lists[side] = rows
    latest_keys = {r["key"] for r in lists["latest"]}

    items = []
    for r in lists["latest"]:
        evt, cid = r["key"].split("/")
        items.append(dict(event=evt, cluster=int(cid), arm="p98vonq", vol=r["vol"], latest_key=r["key"],
                          prod_key=None, prod_status=None))
    for r in lists["production"]:
        evt, cid = r["key"].split("/")
        ba, bn = M.load_bee(evt, "p96vprod"), M.load_bee(evt, "p98vonq")
        pts = ba[0][ba[1] == int(cid)]
        m = M.match_cluster(pts, bn[0], bn[1], cKDTree(bn[0]))
        st = M.status_of(m)
        lk = f"{evt}/{m['best']}" if st == "ok" else None
        hit = [it for it in items if lk is not None and it["latest_key"] == lk]
        if hit:
            hit[0]["prod_key"] = r["key"]
            continue
        items.append(dict(event=evt, cluster=int(cid), arm="p96vprod", vol=r["vol"], latest_key=None,
                          prod_key=r["key"], prod_status=f"on p98vonq: {st} {lk or ''}".strip()))
    items.sort(key=lambda it: (int(it["event"].split("_")[1]), it["cluster"]))

    os.makedirs(a.out + "/prep")
    ref = {os.path.realpath(f"{P99}/prep_{arm}/dqdx_ref_pdvd.json") for arm in ("p96vprod", "p98vonq")}
    os.symlink(f"{P99}/prep_p98vonq/dqdx_ref_pdvd.json", a.out + "/prep/dqdx_ref_pdvd.json")
    sheets = {arm: sheet_rows(arm) for arm in ("p96vprod", "p98vonq")}
    q_items, man, key = {}, [], []
    for i, it in enumerate(items, 1):
        evt, cid, arm = it["event"], it["cluster"], it["arm"]
        k = f"{evt}/{cid}"
        src = f"{P99}/prep_{arm}/smprep-{evt}-c{cid}.json"
        if not os.path.exists(src):
            sys.exit(f"missing payload {src}")
        os.symlink(src, f"{a.out}/prep/smprep-{evt}-c{cid}.json")
        sr = sheets[arm][k]
        man.append(f"{i}\t1\t{evt}\t{cid}\t{sr['npts']}\t{sr['muon_len_cm']}")

        lines = []
        for side, kk in (("latest", it["latest_key"]), ("production", it["prod_key"])):
            if kk is None:
                continue
            new_arm = PAIRS[side][1]
            g = guard_line(evt, new_arm, int(kk.split("/")[1]))
            if g:
                t = float(re.search(r"tick ([0-9.]+)", g).group(1))
                lines.append(f"{side} ({PAIRS[side][0]} {kk} &rarr; {new_arm}): the edge guard rejects it, "
                             f"<b>stop at tick {t:.1f}, {NTICKS - t:.1f} ticks before the frame end</b>")
            else:
                lines.append(f"{side} ({PAIRS[side][0]} {kk} &rarr; {new_arm}): <b>no guard line</b> &mdash; it stopped "
                             f"being a candidate another way (its event's flash match moved)")
        vs = verdicts([("latest", it["latest_key"]), ("sw99", it["latest_key"]), ("smx", it["prod_key"])])
        other = "" if it["prod_status"] is None else f" (production object {it['prod_status']})"
        html = (
            f"<b>own100 &mdash; run 039349's readout-window edge</b> (doc pdvd/100; not blind)<br>"
            f"Shown on <b>{arm}</b>, where the chain still tagged it (the chain's panel shows that arm's is_stm 1){other}.<br>"
            f"Run 039349's frames are {NTICKS} ticks long; production now tells PR so, and the readout-edge guard untags any "
            f"stop in the last {GUARD_TICKS} ticks (30 &micro;s, about 4.4 cm of drift). In the measurement tab the frame "
            f"ends at slice 1600.<ul style='margin:2px 0 2px 0'>"
            + "".join(f"<li>{x}</li>" for x in lines)
            + f"<li>existing record: {'; '.join(vs) if vs else 'none'}</li></ul>"
            "<b>Please judge:</b> does the muon <b>stop inside the frame</b> (STM + MICHEL with the Michel radio, or STM, no "
            "Michel), or is it a longer track <b>the readout window cut</b> (THRU)? UNCLEAR if the frame edge makes it "
            "undecidable. Pin and notes as usual.")
        q_items[k] = dict(html=html)
        key.append("\t".join([str(i), k, arm, it["vol"], it["latest_key"] or "", it["prod_key"] or "",
                              " | ".join(re.sub("<[^>]+>", "", x).replace("&rarr;", "->").replace("&mdash;", "--")
                                         for x in lines), " | ".join(vs)]))

    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write("# doc pdvd/100 step A -- run 039349 readout-window edge, owner look (not blind)\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
        fh.write("\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan="pdvd own100, run 039349's STM tags the real readout window removed (doc pdvd/100)",
                       prep=a.out + "/prep", items=q_items), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\tshown_arm\tvol\tlatest_key\tprod_key\tguard\trecord_verdicts\n")
        fh.write("\n".join(key) + "\n")
    print(f"items {len(items)} (shown on p98vonq {sum(it['arm'] == 'p98vonq' for it in items)}, "
          f"p96vprod {sum(it['arm'] == 'p96vprod' for it in items)}); dqdx refs distinct files {len(ref)}")
    print("\n".join(key))


if __name__ == "__main__":
    main()

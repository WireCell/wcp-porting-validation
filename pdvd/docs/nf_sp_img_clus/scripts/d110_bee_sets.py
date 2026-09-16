#!/usr/bin/env python3
"""doc pdvd/110 -- rebuild the owner's 5-event PDVD Bee set on current PDVD production.

    python3 d110_bee_sets.py /home/xqian/tmp/d110/bee

Fork by duplication of d109_bee_sets.py (untouched).  The only difference is the SETS table: the PDVD set
of doc 109 (Bee 51c1e410) came from q29flip, which ran on 2026-09-13, before the PDVD trajectory flip
8fc6070e (2026-09-15).  This set takes the same five events from d103vflip, the flipped config's proof
arm (doc pdvd/103 sec 14), so it is comparable with the PDHD set c3165b38 (d109hstm, post-flip).
d103vflip carries the round-3 layer scope PDVD has had since 2026-09-05.  The PDHD set is not rebuilt.

In production 039349_20 cl80 is no longer tagged STM (status 3), so it is absent from event 1's
stm_fit layer.
"""
import json, os, sys, zipfile

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SETS = {
    "d110_pdvd_stm5": ("pdvd", "d103vflip",
                       ["039349_2", "039349_20", "039349_18", "039252_8", "039253_6"]),
}


def build(out, setname, det, arm, evs):
    stage = os.path.join(out, setname, "data")
    os.makedirs(stage, exist_ok=True)
    index = []
    for idx, ev in enumerate(evs):
        src_zip = f"{IMG}/{det}/work/{ev}_{arm}/mabc-pr.zip"
        if not os.path.exists(src_zip):
            raise SystemExit(f"missing {src_zip}")
        d = os.path.join(stage, str(idx))
        os.makedirs(d, exist_ok=True)
        n_stm = 0
        with zipfile.ZipFile(src_zip) as src:
            for member in src.namelist():
                base = os.path.basename(member)
                layer = base[base.find("-") + 1:-5]
                blob = src.read(member)
                open(os.path.join(d, f"{idx}-{layer}.json"), "wb").write(blob)
                if layer == "stm_fit-global":
                    j = json.loads(blob)
                    n_stm = len(set(j.get("cluster_id", [])))
        index.append(f"{idx}\t{det}\t{ev}\t{arm}\tstm_fit_clusters={n_stm}")
    open(os.path.join(out, setname + ".index.txt"), "w").write(
        "# bee event index -> detector / run_event / arm\n" + "\n".join(index) + "\n")
    z = os.path.join(out, setname + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        root_dir = os.path.join(out, setname)
        for root, _, files in os.walk(root_dir):
            for f in sorted(files):
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, root_dir))
    print(f"{setname}: {os.path.getsize(z)/1e6:.2f} MB  {z}")
    for line in index:
        print("   " + line)
    return z


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/xqian/tmp/d110/bee"
    os.makedirs(out, exist_ok=True)
    for k, (det, arm, evs) in SETS.items():
        build(out, k, det, arm, evs)

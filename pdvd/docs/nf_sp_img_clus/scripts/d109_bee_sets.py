#!/usr/bin/env python3
"""doc pdvd/109 -- build one 5-event Bee set per detector, for the owner to compare the
STM display now that both detectors carry the round-3 scope.

    python3 d109_bee_sets.py /home/xqian/tmp/d109/bee

Fork by duplication of d36_build_bee_sets.py (untouched).  Difference: there a set was ONE
event carried by three arms (the Bee event index selected the arm); here a set is FIVE
events of ONE arm, so the Bee event index selects the physics event.  `index.txt` records
which is which, since the Bee UI shows only the index.

PDHD comes from d109hstm -- the rescoped config (four layers on require_flag:'STM',
`stm_tagged` dropped).  PDVD comes from q29flip, which has carried that scope since doc
pdvd/39 sec 17.  Both sets therefore show the same thing: the trajectories of the clusters
the tagger TAGGED.

029107_16 (PDHD) and 039349_2 (PDVD) are the two events the owner viewed before the fix, so
the same links can be reopened for a before/after comparison.
"""
import json, os, sys, zipfile

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SETS = {
    "d109_pdhd_stm5": ("pdhd", "d109hstm",
                       ["029107_16", "029107_5", "029107_18", "028084_3", "028084_12"]),
    "d109_pdvd_stm5": ("pdvd", "q29flip",
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
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/xqian/tmp/d109/bee"
    os.makedirs(out, exist_ok=True)
    for k, (det, arm, evs) in SETS.items():
        build(out, k, det, arm, evs)

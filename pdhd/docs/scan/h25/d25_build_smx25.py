#!/usr/bin/env python3
"""doc pdhd/25 sec 6 -- build the `smx25` blind re-judge tranche: the arms' decision set plus the controls
frozen before any arm ran (controls_frozen.txt, preregistered.txt).

    python3 d25_build_smx25.py --arms h25k,h25r,h25kr --round /home/xqian/tmp/h25r \
        --prep .../prep-pdhd-h25base --outdir .../docs/scan/smx25 [--seed 25]

Fork by duplication (CLAUDE.md M10) of doc pdvd/92's d92_build_smx8.py, which stays as that doc's record.
Changed for PDHD and for this round:
  * the decision set comes from the ARMS (every is_stm mover, either direction, base h25base, on the
    committed 303-item population with APA0-majority candidates excluded), not from an offline twin;
  * TWO STRATA on production michel_found, because a found Michel is visible on the display even with
    --blind; each item carries its stratum, and a decision item that does not read like its stratum
    (production is_stm 0, reject bits a subset of the shape bits) is NAMED, not silently kept or dropped;
  * the controls are READ from controls_frozen.txt's FROZEN LISTS (first 4 THRU and first 2 stoppers per
    stratum, skipping any item that an arm moved -- the replacement rule written before the arms ran);
  * agents, not the owner, scan it: two blind scans per item.  The shuffled items are cut into G1..G3;
    wave 1 gives a1=G1 a2=G2 a3=G3, wave 2 (fresh agents) a4=G2 a5=G3 a6=G1, so every item is judged by two
    different scanners and no scanner sees an item twice.

Writes (refuses to overwrite anything):
  ROUND/items_all.txt, ROUND/items_a<1..6>.txt           what the harness and the agents read
  OUTDIR/key_smx25.tsv                                    the group key -- never shown to a scanner
  OUTDIR/items_smx25.txt, OUTDIR/items_a<1..6>.txt        committed copies of the assignment
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d25_bragg_michel as Q
import d25_misses as M

SHAPE = M.B["no_bragg"] | M.B["shape_flat"] | M.B["plateau_off_mip"] | M.B["profile_sparse"]


def frozen(path):
    L = {}
    for line in open(path):
        m = re.match(r"\s+(THRU|STOP)_([MN])\s+(.*)$", line)
        if m:
            L[(m.group(1), m.group(2))] = m.group(3).split()
    if len(L) != 4:
        sys.exit(f"{path}: expected 4 frozen lists, found {sorted(L)}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="h25k,h25r,h25kr")
    ap.add_argument("--base", default="h25base")
    ap.add_argument("--round", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--controls", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls_frozen.txt"))
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()
    outs = [os.path.join(a.round, "items_all.txt"), os.path.join(a.outdir, "key_smx25.tsv")]
    for p in outs:
        if os.path.exists(p):
            sys.exit(f"REFUSING: {p} exists (a scan set is never rebuilt in place, M13)")

    base = {x[0]: x for x in Q.items("pdhd", a.base, "majority")[0]}
    if Q.census(list(base.values()))[0] != (79, 1, 30, 70):
        sys.exit("GATE FAILED: h25base majority is not production's 79/1/30/70")
    # every scored AND unscored candidate of the population, so a mover whose truth is MESSY/UNCLEAR is not lost
    raw0 = Q.read_arm("pdhd", a.base)
    import csv
    POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
           csv.DictReader([l for l in open(Q.HD_KEY) if not l.startswith("#")], delimiter="\t")}
    rec = {r["key"]: r for r in json.load(open(Q.HD_REC))}
    moved_by = collections.defaultdict(list)
    for arm in a.arms.split(","):
        raw1 = Q.read_arm("pdhd", arm)
        for k, d in raw0.items():
            if k not in POP or d["apa_major"] == 0:
                continue
            if k not in raw1:
                moved_by[k].append(f"{arm}:left-pool")
            elif int(d["is_stm"]) != int(raw1[k]["is_stm"]):
                moved_by[k].append(f"{arm}:{int(d['is_stm'])}->{int(raw1[k]['is_stm'])}")
    decision = sorted(moved_by)

    def stratum(k):
        return "M" if int(raw0[k]["michel_found"]) == 1 else "N"

    def reads_alike(k):
        d = raw0[k]
        rb = int(d["reject_bits"])
        return int(d["is_stm"]) == 0 and rb != 0 and rb & ~SHAPE == 0

    F = frozen(a.controls)
    controls = {}
    for kind, n in (("THRU", 4), ("STOP", 2)):
        for s in ("M", "N"):
            take = [k for k in F[(kind, s)] if k not in moved_by][:n]
            if len(take) < n:
                sys.exit(f"frozen list {kind}_{s} has fewer than {n} unmoved items")
            skipped = [k for k in F[(kind, s)][:F[(kind, s)].index(take[-1]) + 1] if k in moved_by]
            for k in take:
                controls[k] = f"control_{kind.lower()}"
            print(f"controls {kind}_{s}: {' '.join(take)}" + (f"   (skipped, moved by an arm: {' '.join(skipped)})" if skipped else ""))

    items = {k: "decision" for k in decision}
    for k, g in controls.items():
        if k in items:
            sys.exit(f"{k} is both a decision item and a control")
        items[k] = g
    missing = [k for k in items if not os.path.exists(os.path.join(a.prep, "smprep-%s-c%s.json" % tuple(k.split("/"))))]
    if missing:
        sys.exit(f"no prep sidecar for {missing}")

    print(f"\ndecision set {len(decision)} (movers in {a.arms} on the APA0-majority-excluded 303 population):")
    notalike = []
    for k in decision:
        al = reads_alike(k)
        if not al:
            notalike.append(k)
        tv = Q.hd_truth(rec[k]) if k in rec else ("-", None, "-")
        print(f"  {k:15s} stratum {stratum(k)}  {' '.join(moved_by[k]):40s} truth {tv[0]}/{tv[1]}/{tv[2]}  "
              f"apa_any0 {bool(raw0[k]['apa_any0'])}  {'reads alike' if al else 'NOT blind-equivalent: ' + M.names(int(raw0[k]['reject_bits']))}")

    order = sorted(items)
    random.Random(a.seed).shuffle(order)
    G = [order[0::3], order[1::3], order[2::3]]
    assign = {1: G[0], 2: G[1], 3: G[2], 4: G[1], 5: G[2], 6: G[0]}
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.round, "items_all.txt"), "w") as fh:
        fh.write("".join(k + "\n" for k in order))
    with open(os.path.join(a.outdir, "items_smx25.txt"), "w") as fh:
        fh.write("".join(k + "\n" for k in order))
    for i, ks in assign.items():
        body = "".join(k + "\n" for k in ks)
        for d in (a.round, a.outdir):
            with open(os.path.join(d, f"items_a{i}.txt"), "w") as fh:
                fh.write(body)
    with open(os.path.join(a.outdir, "key_smx25.tsv"), "w") as fh:
        fh.write("# doc pdhd/25 sec 6 -- smx25 KEY (never shown to a scanner). group: decision = an arm moved is_stm; "
                 "control_thru / control_stop = frozen before the arms ran (controls_frozen.txt). stratum = production "
                 "michel_found (M 1 / N 0). truth = the smx23 record's (source). scanners = the two agents given it.\n")
        fh.write("order\tkey\tgroup\tstratum\treads_alike\tstrict\tmoved_by\ttruth_verdict\ttruth_kind\ttruth_source\t"
                 "record_confidence\tprod_reject\tprod_michel_found\tscanners\n")
        for i, k in enumerate(order, 1):
            d, r = raw0[k], rec.get(k, {})
            tv = Q.hd_truth(r) if r else ("-", None, "-")
            sc = ",".join(f"rv5_a{j}" for j, ks in assign.items() if k in ks)
            fh.write(f"{i}\t{k}\t{items[k]}\t{stratum(k)}\t{int(reads_alike(k))}\t{int(not d['apa_any0'])}\t"
                     f"{' '.join(moved_by.get(k, [])) or '-'}\t{tv[0]}\t{tv[1]}\t{tv[2]}\t{r.get('confidence')}\t"
                     f"{M.names(int(d['reject_bits']))}\t{int(d['michel_found'])}\t{sc}\n")
    cnt = collections.Counter((items[k], stratum(k)) for k in order)
    print(f"\nitems {len(order)}: " + ", ".join(f"{g}/{s} {n}" for (g, s), n in sorted(cnt.items())))
    print(f"not blind-equivalent decision items: {' '.join(notalike) or '-'}")
    print("groups: " + " | ".join(f"G{i+1} {len(g)}" for i, g in enumerate(G)) + f"; seed {a.seed}")


if __name__ == "__main__":
    main()

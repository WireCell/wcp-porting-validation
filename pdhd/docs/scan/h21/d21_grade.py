#!/usr/bin/env python3
"""doc pdhd/21 -- grade a knob arm against the smx22 hand-scan record, straight from
tracking-pr.root.

    python3 grade.py <arm> [<arm> ...]

Why not prep + d18_census.py: d18_census reads a key TSV that prep_stm_michel_scan.py
writes, and a prep costs ~200 MB and minutes per arm.  Everything d18_census needs from
the chain (is_stm, michel_found, reject_bits) is already in T_stm_michel, so this reads
it directly.  The instrument is validated by reproducing the COMMITTED census on the
production arm before any arm number is printed -- if p82bhoff does not come back
61/0/86/110, nothing below it is trustworthy and the script refuses.

Truth is d18_census.py:53-60's precedence: owner_review > owner_smx1 > agent, FRAG_
stripped, MESSY/UNCLEAR unscored.
"""
import glob, os, json, csv, sys, collections
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
X = IMG + "/pdhd/docs/scan"
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid",
        "continuation", "stop_near_boundary", "vertex_hadron", "short",
        "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
names = lambda rb: "|".join(n for i, n in enumerate(BITS) if int(rb) >> i & 1) or "STM"


def arm(tag):
    """-> {(event, cluster): {is_stm, michel_found, reject_bits}}"""
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(
            ["cluster_id", "is_stm", "michel_found", "reject_bits"], library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = {
                "is_stm": int(c["is_stm"][i]), "michel_found": int(c["michel_found"][i]),
                "reject_bits": int(c["reject_bits"][i])}
    return out


rec = json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))
def truth(r):
    if r.get("owner_review"):
        return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    return base(r["verdict"])

# the 303 items production reaches -- fixed by the committed key, so every arm is
# scored on the SAME population and a cap change cannot silently move the denominator
rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
       rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}


def census(A, label, quiet=False):
    c = collections.Counter(); rows = {}
    for r in rec:
        k = r["key"]
        if k not in POP:
            continue
        d = A.get(tuple(k.split("/")))
        if d is None:
            continue
        v = truth(r)
        if v in ("MESSY", "UNCLEAR"):
            continue
        hs = v in ("STM_MICHEL", "STM_ONLY"); cs = d["is_stm"] == 1
        cls = "TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"
        c[cls] += 1; rows[k] = (cls, hs, d)
    p = c["TP"] / max(1, c["TP"] + c["FP"]); e = c["TP"] / max(1, c["TP"] + c["FN"])
    if not quiet:
        print(f"  {label:10s} TP {c['TP']:3d} FP {c['FP']:3d} FN {c['FN']:3d} TN {c['TN']:3d}"
              f" | purity {p:.3f} efficiency {e:.3f}")
    return c, rows


BASE = arm("p82bhoff")
cb, rb_ = census(BASE, "p82bhoff", quiet=True)
if (cb["TP"], cb["FP"], cb["FN"], cb["TN"]) != (61, 0, 86, 110):
    sys.exit(f"INSTRUMENT FAILED: p82bhoff reads {dict(cb)}, committed census says "
             "61/0/86/110 -- refusing to grade any arm")
print("instrument validated: p82bhoff reproduces the committed census 61/0/86/110\n")
census(BASE, "p82bhoff")

for tag in sys.argv[1:]:
    A = arm(tag)
    if not A:
        print(f"  {tag:10s} NO OUTPUT"); continue
    ca, ra = census(A, tag)
    gained = sorted(k for k in ra if ra[k][0] == "TP" and rb_.get(k, ("",))[0] == "FN")
    newfp = sorted(k for k in ra if ra[k][0] == "FP" and rb_.get(k, ("",))[0] == "TN")
    lost = sorted(k for k in ra if ra[k][0] == "FN" and rb_.get(k, ("",))[0] == "TP")
    remfp = sorted(k for k in ra if ra[k][0] == "TN" and rb_.get(k, ("",))[0] == "FP")
    print(f"             gained TP ({len(gained)}): {' '.join(gained) or 'none'}")
    print(f"             NEW FP   ({len(newfp)}): {' '.join(newfp) or 'none'}")
    if lost:  print(f"             LOST TP  ({len(lost)}): {' '.join(lost)}")
    if remfp: print(f"             removed FP ({len(remfp)}): {' '.join(remfp)}")
    mm = [k for k in ra if k in rb_ and ra[k][2]["michel_found"] != rb_[k][2]["michel_found"]]
    print(f"             michel_found moved on {len(mm)} item(s)"
          + (f": {' '.join(sorted(mm))}" if mm else " (predicted: none)"))

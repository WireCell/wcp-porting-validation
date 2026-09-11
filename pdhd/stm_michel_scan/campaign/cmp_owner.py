#!/usr/bin/env python3
"""Grade agent scan records against the owner's labels (the blind calibration).

    cmp_owner.py OWNER_LABELS_JSON RECORD_DIR [RECORD_DIR ...] [--out TSV]

doc pdhd/18 sec 5.  OWNER_LABELS_JSON is a viewer labels.json (the owner's tag,
here PDHD `smx1`, read from its Phase-0 backup copy); each RECORD_DIR holds
mkv.py records (<event>_<cluster>.json).  Only keys present in both are graded.

Reported, each stratified by the AGENT's own confidence (doc pdvd/55 sec 16 /
findings sec 0: the field is calibrated, so the strata are the point):
  * verdict, exact (FRAG_X is compared as X plus the partial flag);
  * stopper-or-not (STM_MICHEL / STM_ONLY and their FRAG forms = stopper;
    MESSY and UNCLEAR are their own class);
  * michel_kind, where the owner set one;
  * per-object tags on the object keys both sides tagged (the owner's pf_segments
    were taken on arm d53h, the agents' on h18s; segment ids are stable where the
    fit is -- doc pdhd/18 sec 2 -- but role-7 rows exist only on h18s);
  * pins: the owner's placed pins, and the agent's pin_rr.
"""
import argparse, collections, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("owner"); ap.add_argument("recdirs", nargs="+")
ap.add_argument("--out", default=None)
a = ap.parse_args()

O = json.load(open(a.owner))["labels"]
R = {}
for d in a.recdirs:
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        R[r["key"]] = r


def base(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def cls(v):
    b = base(v)
    return "stopper" if b in ("STM_MICHEL", "STM_ONLY") else ("thru" if b == "THRU" else b)


keys = sorted(set(O) & set(R))
rows = []
agree = collections.defaultdict(lambda: [0, 0])
tag_ok = tag_n = 0
for k in keys:
    o, r = O[k], R[k]
    ov = o.get("choice") or o.get("label")
    ok_v = base(ov) == base(r["verdict"])
    ok_c = cls(ov) == cls(r["verdict"])
    ok_p = bool(o.get("partial")) == r["verdict"].startswith("FRAG_")
    ok_k = None
    if o.get("michel_kind") not in (None, "— not set —"):
        ok_k = o["michel_kind"] == r["michel_kind"]
    for s in ("all", r["confidence"]):
        agree[("verdict", s)][0] += ok_v; agree[("verdict", s)][1] += 1
        agree[("stopper", s)][0] += ok_c; agree[("stopper", s)][1] += 1
        if ok_k is not None:
            agree[("kind", s)][0] += ok_k; agree[("kind", s)][1] += 1
    ot = o.get("pf_segments") or {}
    sh = [x for x in ot if x in r["tags"]]
    nt = sum(ot[x] == r["tags"][x] for x in sh)
    tag_ok += nt; tag_n += len(sh)
    pin = o.get("pin") or {}
    rows.append([k, ov, "FRAG" if o.get("partial") else "", o.get("michel_kind", ""),
                 r["verdict"], r["michel_kind"], r["confidence"],
                 "Y" if ok_v else "n", "Y" if ok_c else "n",
                 "" if ok_k is None else ("Y" if ok_k else "n"),
                 "%d/%d" % (nt, len(sh)),
                 ("rr %.1f" % pin["rr"]) if pin.get("placed") and pin.get("rr") is not None else "",
                 "" if r.get("pin_rr") is None else "rr %.1f" % r["pin_rr"],
                 (o.get("notes") or "").replace("\t", " ")[:120]])

print("graded %d items (owner %d, records %d)" % (len(keys), len(O), len(R)))
for what in ("verdict", "stopper", "kind"):
    line = "  %-8s" % what
    for s in ("all", "high", "medium", "low"):
        g, n = agree[(what, s)]
        if n:
            line += "   %s %d/%d" % (s, g, n)
    print(line)
print("  tags on shared object keys: %d/%d" % (tag_ok, tag_n))
print("  owner stopper / thru / other: %s" % dict(collections.Counter(cls(O[k].get("choice") or O[k].get("label")) for k in keys)))
print("  agent stopper / thru / other: %s" % dict(collections.Counter(cls(R[k]["verdict"]) for k in keys)))
print("\ndisagreements (verdict):")
for row in rows:
    if row[7] == "n":
        print("  %-14s owner %s%s/%s  agent %s/%s (%s)  stopper-agree %s  tags %s" % (
            row[0], row[1], "(FRAG)" if row[2] else "", row[3], row[4], row[5], row[6], row[8], row[10]))
if a.out:
    with open(a.out, "w") as fh:
        fh.write("key\towner_verdict\towner_frag\towner_kind\tagent_verdict\tagent_kind\tagent_conf"
                 "\tverdict_agree\tstopper_agree\tkind_agree\ttags_agree\towner_pin\tagent_pin\towner_notes\n")
        fh.write("".join("\t".join(map(str, r)) + "\n" for r in rows))
    print("\nwrote %s" % a.out)

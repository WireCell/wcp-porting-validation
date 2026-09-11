#!/usr/bin/env python3
"""doc pdhd/18 sec 7 -- the chain against the hand scan, graded on BARE production.

    d18_census.py --record pdhd/docs/scan/pdhd_stm_michel_smx18_verdicts.json \
        --key pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv \
        --key-extras pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_h18b.tsv \
        --shots /home/xqian/tmp/h18/shots [--agent-only]

Truth per item = the owner's smx1 verdict where the owner labelled the item, else
the agent record (--agent-only: always the agent record, i.e. the blind calls).
The chain = the bare-production key: p82bhoff (PDHD production, max_candidates 8)
for the 303 items it holds, and h18b (production + max_candidates 64) for the 14
cap extras, which production does not reach at all.  NEVER the scan arm's key:
the survey bag moves 197 of 325 candidates' profile branches (doc pdhd/18 sec 2).

Classes (score_stm_michel_scan.py's TRUTH table): STM_MICHEL / STM_ONLY and their
FRAG forms are stoppers; THRU / FRAG_THRU are not; MESSY and UNCLEAR are
UNSCORED and counted separately.  Michel truth = a stopper whose michel_kind is
attached or both (a michel tag), graded against the chain's michel_found.
"""
import argparse, collections, csv, json, os


def read_tsv(p):
    return list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))


def base(v):
    return v[5:] if v and v.startswith("FRAG_") else v


ap = argparse.ArgumentParser()
ap.add_argument("--record", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--key-extras", default=None)
ap.add_argument("--shots", default=None, help="for the stop's APA (context.json ends.stop.xyz)")
ap.add_argument("--agent-only", action="store_true")
a = ap.parse_args()

rec = json.load(open(a.record))
K = {"%s/%s" % (r["event"], r["cluster"]): dict(r, src="p82bhoff") for r in read_tsv(a.key)}
if a.key_extras:
    for r in read_tsv(a.key_extras):
        k = "%s/%s" % (r["event"], r["cluster"])
        if k not in K:
            K[k] = dict(r, src="h18b")


def truth(r):
    if not a.agent_only and r.get("owner_smx1"):
        o = r["owner_smx1"]
        return base(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner"
    return base(r["verdict"]), r["michel_kind"], "agent"


def apa(k):
    if not a.shots:
        return "?"
    p = os.path.join(a.shots, k.replace("/", "_"), "context.json")
    try:
        x, y, z = json.load(open(p))["ends"]["stop"]["xyz"]
    except Exception:
        return "?"
    return "APA%d" % ((0 if x < 0 else 1) + (2 if z > 231.5 else 0))


def tally(items, label):
    c = collections.Counter(); m = collections.Counter(); uns = collections.Counter()
    for r, (v, kind, src), kk in items:
        if v in ("MESSY", "UNCLEAR"):
            uns[v] += 1
            continue
        hs = v in ("STM_MICHEL", "STM_ONLY")
        cs = int(kk["is_stm"]) == 1
        c[("TP" if hs and cs else "FN" if hs else "FP" if cs else "TN")] += 1
        if hs:
            hm = kind in ("attached", "both")
            cm = int(kk["michel_found"]) == 1
            m[("TP" if hm and cm else "FN" if hm else "FP" if cm else "TN")] += 1

    def pe(t):
        p = t["TP"] / max(1, t["TP"] + t["FP"]); e = t["TP"] / max(1, t["TP"] + t["FN"])
        f = 2 * p * e / max(1e-9, p + e)
        return "TP %3d FP %3d FN %3d TN %3d | purity %.3f efficiency %.3f F1 %.3f" % (
            t["TP"], t["FP"], t["FN"], t["TN"], p, e, f)
    print("  %-26s is_stm       %s   unscored %s" % (label, pe(c), dict(uns)))
    print("  %-26s michel_found %s   (on hand stoppers only)" % ("", pe(m)))
    return c, m


items = []
for r in rec:
    kk = K.get(r["key"])
    if kk is None:
        raise SystemExit("no key row for %s" % r["key"])
    items.append((r, truth(r), kk))

print("record %s (%d items)  truth: %s" % (a.record, len(rec), "AGENT ONLY" if a.agent_only else "owner where labelled, else agent"))
print("chain: bare production key %s; cap extras from %s" % (a.key, a.key_extras))
print("\n== verdicts: %s" % dict(collections.Counter(t[1][0] for t in items)))
print("== truth source: %s" % dict(collections.Counter(t[1][2] for t in items)))
print("\n== chain vs hand")
tally([t for t in items if t[2]["src"] == "p82bhoff"], "production (303)")
tally([t for t in items if t[2]["src"] == "h18b"], "cap-64 extras (h18b)")
tally(items, "all")
print("\n== by the scanner's confidence (agent truth only)")
for cf in ("high", "medium", "low"):
    tally([t for t in items if t[1][2] == "agent" and t[0]["confidence"] == cf and t[2]["src"] == "p82bhoff"], "production, %s" % cf)
print("\n== robustness of the efficiency (production 303, agent truth where no owner label)")
flat = lambda r: "FLAT_STOP:" in (r.get("notes") or "")
tally([t for t in items if t[2]["src"] == "p82bhoff" and not (t[1][2] == "agent" and flat(t[0]))],
      "without FLAT_STOP calls")
tally([t for t in items if t[2]["src"] == "p82bhoff" and (t[1][2] == "owner" or t[0]["confidence"] == "high")],
      "owner + agent-high only")


def upper_end(k):
    """True when the fit end is the higher end of the track (a stop there needs an upward-going particle)."""
    if not a.shots:
        return None
    try:
        e = json.load(open(os.path.join(a.shots, k.replace("/", "_"), "context.json")))["ends"]
        return e["stop"]["xyz"][1] > e["entry"]["xyz"][1]
    except Exception:
        return None


hs = [t for t in items if t[1][0] in ("STM_MICHEL", "STM_ONLY")]
up = [t for t in hs if upper_end(t[0]["key"])]
print("\n== hand stoppers whose fit end is the UPPER end of the track: %d of %d (FLAT_STOP among them: %d; chain is_stm=1 among them: %d)"
      % (len(up), len(hs), sum(1 for t in up if flat(t[0])), sum(1 for t in up if int(t[2]["is_stm"]) == 1)))
print("  " + " ".join(sorted(t[0]["key"] for t in up)))

print("\n== by the stop's APA (production 303)")
for ap_ in ("APA0", "APA1", "APA2", "APA3", "?"):
    sub = [t for t in items if t[2]["src"] == "p82bhoff" and apa(t[0]["key"]) == ap_]
    if sub:
        tally(sub, "%s (n=%d)" % (ap_, len(sub)))
print("\n== pins and notes prefixes (agent records)")
notes = collections.Counter()
for r in rec:
    n = (r.get("notes") or "")
    for pfx in ("OVERSHOOT:", "UNDERSHOOT:", "NO_MEASUREMENT:", "FLAT_STOP:", "CATHODE:", "SEAM:",
                "ANODE:", "FACE:", "DEAD:", "ISOCHRONOUS:", "REVERSED:"):
        if n.startswith(pfx) or (" " + pfx) in n:
            notes[pfx] += 1
print("  pins moved (pin_rr): %d   notes prefixes: %s" % (sum(1 for r in rec if r.get("pin_rr") is not None), dict(notes)))
print("  confidence: %s" % dict(collections.Counter(r["confidence"] for r in rec)))
print("  michel_kind: %s" % dict(collections.Counter(r["michel_kind"] for r in rec)))
print("  tags: %s" % dict(collections.Counter(t for r in rec for t in r["tags"].values())))
print("\n== disagreements, production (303), by name")
for r, (v, kind, src), kk in items:
    if kk["src"] != "p82bhoff" or v in ("MESSY", "UNCLEAR"):
        continue
    hs = v in ("STM_MICHEL", "STM_ONLY"); cs = int(kk["is_stm"]) == 1
    if hs != cs:
        print("  %-15s hand %-10s (%s, %s)  chain is_stm %s  reject %s" % (
            r["key"], v, src, r["confidence"], kk["is_stm"], kk.get("reject_names", "")))

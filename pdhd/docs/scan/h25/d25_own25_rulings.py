#!/usr/bin/env python3
"""doc pdhd/25 sec 7 -- turn the owner's own25 viewer labels into a rulings file for mkowner_record.py.

    python3 d25_own25_rulings.py --labels work/stm_michel_labels/own25/labels.json \
        --sheet /home/xqian/tmp/h25r/own25/own25_sheet.tsv --key smx25/key_smx25.tsv \
        --record pdhd_stm_michel_smx25_verdicts.json --shas-before /home/xqian/tmp/h25r/labels_sha_before_own25.txt \
        --out smx25/owner_rulings_own25.json

Refuses (exit 1) unless every sheet item carries a label, the labels name no key outside the sheet, and every
label tag listed in --shas-before is byte-unchanged (the owner's session could reach only own25).
The viewer's row field is `label` / `choice`, NOT `verdict` (doc pdhd/20: reading `verdict` gave two false
"unchanged" diffs).  A michel_kind of "-- not set --" becomes None and is never filled from the chain.
Forked by duplication of the hand-built owner_rulings_own23.json layout; no earlier rulings file is written.
"""
import argparse, csv, datetime, hashlib, json, os, sys

ap = argparse.ArgumentParser()
for f in ("--labels", "--sheet", "--key", "--record", "--shas-before", "--out"):
    ap.add_argument(f, required=True)
# doc pdhd/25 sec 8 reuses this for own26; the defaults reproduce the own25 rulings file (date aside)
ap.add_argument("--tag", default="own25")
ap.add_argument("--section", default="sec 7")
ap.add_argument("--question", default="is this a stopping muon, or through-going? -- smx25 items the two blind agents "
                                      "split on or called with low confidence, plus one anchor")
a = ap.parse_args()

L = json.load(open(a.labels))
rows = L.get("labels") or {}
sheet = [r for r in csv.DictReader((l for l in open(a.sheet) if not l.startswith("#")), delimiter="\t")]
skeys = {f"{r['event']}/{r['cluster']}": int(r["scan_id"]) for r in sheet}
key = {}
for r in csv.reader((l for l in open(a.key) if not l.startswith("#")), delimiter="\t"):
    if r and r[0] != "order":
        key[r[1]] = r
rec = {o["key"]: o for o in json.load(open(a.record))}

bad = []
judged = {k: v for k, v in rows.items() if isinstance(v, dict) and (v.get("label") or v.get("choice"))}
if set(judged) != set(skeys):
    bad.append(f"judged {sorted(judged)} != sheet {sorted(skeys)}")
for line in open(a.shas_before):
    sha, path = line.split()
    full = path if os.path.isabs(path) else os.path.join("/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd", path)
    now = hashlib.sha256(open(full, "rb").read()).hexdigest()
    if now != sha:
        bad.append(f"{path} changed during the session ({sha[:12]} -> {now[:12]})")
if bad:
    sys.exit("REFUSING:\n  " + "\n  ".join(bad))

items = {}
for k in sorted(skeys, key=lambda k: skeys[k]):
    v = judged[k]
    kind = v.get("michel_kind")
    kind = None if (not kind or "not set" in str(kind)) else kind
    kr = key[k]
    o = rec[k]
    scans = " / ".join(f"{s['verdict']} {s['confidence']}" for s in o["review_v5"]["scans"])
    prev = o.get("owner_review")
    pin = v.get("pin") or {}
    note = (f"smx25 group {kr[2]}, agents {scans} ({o['review_v5']['outcome']})"
            + (f"; replaces the owner's earlier ruling {prev['verdict']} ({prev.get('source')}, {prev.get('date')})" if prev else "")
            + (f"; pin placed rr {pin.get('rr')} moved {pin.get('moved_cm')} cm" if pin.get("placed") else "; no pin placed"))
    items[k] = dict(scan_id=skeys[k],
                    question=a.question,
                    owner_verdict=v.get("label") or v.get("choice"), michel_kind=kind,
                    words=(v.get("notes") or "").strip() or None, note=note)
    if v.get("label") and v.get("choice") and v["label"] != v["choice"]:
        sys.exit(f"{k}: label {v['label']} != choice {v['choice']}")

sha = hashlib.sha256(open(a.labels, "rb").read()).hexdigest()
mtime = datetime.datetime.fromtimestamp(os.path.getmtime(a.labels)).isoformat(timespec="seconds")
out = dict(doc=f"pdhd/docs/25_pdvd-config-on-pdhd-and-bragg-michel-fraction.md {a.section}",
           date=datetime.date.today().isoformat(),
           viewer=f"tag {a.tag} on :5017, its own empty label tag; prepdir prep-pdhd-h25base (today's production, so the "
                  f"chain's answer on screen was is_stm 0 on all {len(items)}); --dead-points ON",
           how=f"The owner judged the {len(items)} items of {a.tag}_sheet.tsv (queue rule in preregistered_{a.tag}.md). Labels from "
               f"{a.tag}/labels.json ({mtime}, sha256 {sha}). Every other label tag byte-unchanged.",
           caveat="The blind is gone (owner decision 2026-09-08): the chain's verdict was on screen, and the doc 25 "
                  "report had named three of the items. Owner rulings, not independent blind verdicts.",
           items=items)
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"wrote {a.out}: {len(items)} rulings")
for k, u in items.items():
    print(f"  {k:15s} {u['owner_verdict']:11s} {str(u['michel_kind']):9s} | {u['note']} | words: {u['words']}")

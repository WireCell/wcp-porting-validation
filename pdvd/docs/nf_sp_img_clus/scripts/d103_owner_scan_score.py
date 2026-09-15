#!/usr/bin/env python3
"""doc pdvd/103 -- fold the owner's own103h adjudication into a new record (figs/103_pred_amend2.txt sec 3), and print
the served set item by item plus the control check (sec 4).  Fork of d100_michel_scan_score.py (untouched).  The
re-grade itself is d103_union_grade.py --owner-record.

    python3 d103_owner_scan_score.py --set /home/xqian/tmp/d103/own/set_pdhd_smx27 \
        --labels $IMG/pdhd/work/stm_michel_labels/own103h/labels.json \
        --record-out $IMG/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json > figs/103_own103h_pdhd.txt

Descriptive only: the set is selected by the disputed metric, so no purity is computed on it.  PDHD Michel class: a
stopper is Michel-positive when michel_kind is attached / both.  An owner stopper with no michel_kind is excluded from
the Michel population by d103_union_grade.py.  Refuses an existing record (M13).
"""
import argparse, collections, csv, hashlib, json, os, sys

STOP = ("STM_MICHEL", "STM_ONLY")
UNSET = (None, "", "None", "— not set —")


def base(v):
    return (v or "").replace("FRAG_", "")


def cls(v):
    b = base(v)
    return "stopper" if b in STOP else ("unjudged" if b in ("MESSY", "UNCLEAR", "") else "non-stopper")


DET = "pdhd"


def mcls(v, mk):
    if cls(v) != "stopper":
        return cls(v)
    if DET == "pdvd":                                  # PDVD Michel truth = verdict STM_MICHEL (amendment 4 sec 5)
        return "michel" if base(v) == "STM_MICHEL" else "no michel"
    if mk in UNSET:
        return "stopper, kind unset"
    return "michel" if mk in ("attached", "both") else "no michel"


def tagged(ans, what):
    return f"{what} 1" in ans


def outcome(r, ov, omk):
    """what the owner's verdict does to the false positive this tier-1 item was charged with"""
    out = []
    for charge in r["fp_in"].split(","):
        cell, chain = charge.split(":")
        if chain == "is_stm":
            c = cls(ov)
            out.append(f"{charge}: " + {"stopper": "now a TP", "non-stopper": "FP confirmed",
                                          "unjudged": "leaves the population"}[c])
        else:
            m = mcls(ov, omk)
            s = {"michel": "now a TP", "no michel": "FP confirmed", "stopper, kind unset": "leaves the Michel population (kind unset)",
                 "non-stopper": ("FP confirmed (not a stopper)" if DET == "pdvd" else "leaves the Michel population (not a stopper)"),
                 "unjudged": "leaves the population"}[m]
            if m == "non-stopper" and tagged(r[cell], "is_stm"):
                s += f"; becomes an is_stm FP in {cell}"
            out.append(f"{charge}: {s}")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--record-out", required=True)
    ap.add_argument("--shown-arm", default="d102hcs")
    ap.add_argument("--det", default="pdhd", choices=["pdhd", "pdvd"])
    a = ap.parse_args()
    global DET
    DET = a.det
    if os.path.exists(a.record_out):
        sys.exit(f"REFUSING: {a.record_out} exists (M13)")
    raw = open(a.labels, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    J = json.loads(raw)
    L = J["labels"]
    rows = {r["key"]: r for r in csv.DictReader(open(a.set + "/items.tsv"), delimiter="\t")}
    extra = set(L) - set(rows)
    if extra:
        sys.exit(f"labels name keys outside the set: {sorted(extra)}")
    nrev = sum(1 for x in L.values() if x.get("revealed_before_label"))
    print(f"# doc pdvd/103 own103h (owner, 2026-09-15): {len(L)} of {len(rows)} items labelled; tag {J.get('tag')}; "
          f"labels sha256 {sha[:12]}; revealed the chain answer before labelling: {nrev} of {len(L)} (NOT blind)")
    print(f"# set {a.set}; shown on {a.shown_arm}\n")

    print("== 1. each item")
    for k, r in sorted(rows.items(), key=lambda kv: int(kv[1]["scan_id"])):
        x = L.get(k)
        if not x:
            print(f"  {r['scan_id']:>2s} {k:14s} {r['role']:7s} UNLABELLED")
            continue
        ov, omk = x["label"], x.get("michel_kind")
        omk = None if omk in UNSET else omk
        lv, lmk = r["label_verdict"], (None if r["label_michel_kind"] in UNSET else r["label_michel_kind"])
        chg = ("class " + cls(lv) + " -> " + cls(ov)) if cls(lv) != cls(ov) else (
            ("michel " + mcls(lv, lmk) + " -> " + mcls(ov, omk)) if mcls(lv, lmk) != mcls(ov, omk) else "same")
        pin = x.get("pin") or {}
        pmv = f" pin moved {pin.get('moved_cm', 0):.1f} cm" if pin.get("placed") else ""
        note = f" notes: {x['notes'].strip()!r}" if (x.get("notes") or "").strip() else ""
        print(f"  {r['scan_id']:>2s} {k:14s} {r['role']:7s} A0 [{r['A0']}] A1 [{r['A1']}] | label {lv}/{lmk} "
              f"({r['label_source']}, {r['label_confidence']}) -> OWNER {ov}/{omk} | {chg}{pmv}{note}")

    print("\n== 2. tier 1: what each charged false positive becomes (descriptive; NOT a purity)")
    by = collections.Counter()
    for k, r in rows.items():
        if r["role"] != "tier1" or k not in L:
            continue
        omk = None if L[k].get("michel_kind") in UNSET else L[k].get("michel_kind")
        o = outcome(r, L[k]["label"], omk)
        print(f"  {k:14s} {o}")
        for part in o.split("; "):
            if ":" in part:
                ch, res = part.split(": ", 1)
                by[(ch.split(":")[1], res.split(";")[0])] += 1
    for (ch, res), n in sorted(by.items()):
        print(f"    {ch:7s} {res}: {n}")

    print("\n== 3. controls (figs/103_pred_amend2.txt sec 4: >= 3 of 8 stopper-class changes -> not settled)")
    ctl = [k for k, r in rows.items() if r["role"] == "control" and k in L]
    nulls = [k for k in ctl if rows[k]["fp_in"] == "-"]
    shared = [k for k in ctl if rows[k]["fp_in"] != "-"]
    def ch(k):
        r, x = rows[k], L[k]
        lmk = None if r["label_michel_kind"] in UNSET else r["label_michel_kind"]
        omk = None if x.get("michel_kind") in UNSET else x.get("michel_kind")
        return cls(r["label_verdict"]) != cls(x["label"]), mcls(r["label_verdict"], lmk) != mcls(x["label"], omk)
    sc = sum(1 for k in ctl if ch(k)[0])
    print(f"  stopper class changed: {sc} of {len(ctl)} -> {'NOT SETTLED' if sc >= 3 else 'controls hold on the stopper call'}")
    print(f"  Michel class changed (incl. kind set/unset): {sum(1 for k in ctl if ch(k)[1])} of {len(ctl)}: "
          f"{[k for k in ctl if ch(k)[1]]}")
    print(f"  null controls {len(nulls)}; shared-FP control(s) reported apart: "
          f"{[(k, rows[k]['fp_in'], rows[k]['label_verdict'] + '/' + rows[k]['label_michel_kind'], L[k]['label'] + '/' + str(L[k].get('michel_kind'))) for k in shared]}")

    rec = []
    for k, x in L.items():
        r = rows[k]
        omk = None if x.get("michel_kind") in UNSET else x.get("michel_kind")
        rec.append(dict(
            key=k, verdict=x["label"], michel_kind=omk, confidence="owner", source="own103h (owner, 2026-09-15)",
            owner_review=dict(verdict=x["label"], michel_kind=omk, notes=x.get("notes") or "", date="2026-09-15",
                              tag="own103h", question="doc pdvd/103 figs/103_pred_amend2.txt: owner adjudication"),
            evidence=x.get("notes") or "", scan_id=int(r["scan_id"]), role=r["role"], fp_in=r["fp_in"],
            shown_arm=a.shown_arm, pin=x.get("pin") if (x.get("pin") or {}).get("placed") else None,
            revealed_before_label=bool(x.get("revealed_before_label")), pf_segments=x.get("pf_segments"),
            prior_label=f"{r['label_verdict']}/{r['label_michel_kind']} ({r['label_source']}, {r['label_confidence']})",
            labels_sha256=sha))
    rec.sort(key=lambda it: it["scan_id"])
    json.dump(rec, open(a.record_out, "w"), indent=1)
    print(f"\n== 4. wrote {a.record_out}: {len(rec)} items")


if __name__ == "__main__":
    main()

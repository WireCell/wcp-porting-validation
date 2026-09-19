#!/usr/bin/env python3
"""doc pdvd/116 sec 6.3 -- fold the owner's own116v / own116h look at R2's false positives on the new trajectory into a
new record, and print the served set item by item plus the control check.  Fork by duplication of
d103_owner_scan_score.py (untouched): the R2 column replaces A1, the date and the rule name change.  The re-grade
itself is d116_grade.py --owner-record.

    python3 d116_owner_scan_score.py --det pdvd --set /home/xqian/tmp/d116/own/set_pdvd \
        --labels $IMG/pdvd/work/stm_michel_labels/own116v/labels.json --shown-arm d116vr2 \
        --record-out $IMG/pdvd/docs/scan/pdvd_stm_michel_own116v_verdicts.json > figs/116_own116v_pdvd.txt

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
DATE = "2026-09-18"


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
    ap.add_argument("--shown-arm", required=True)
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
    tag = J.get("tag") or {"pdvd": "own116v", "pdhd": "own116h"}[DET]
    rule = "doc pdvd/116 sec 6.3"
    print(f"# doc pdvd/116 {tag} (owner, {DATE}): {len(L)} of {len(rows)} items labelled; tag {J.get('tag')}; "
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
        print(f"  {r['scan_id']:>2s} {k:14s} {r['role']:7s} A0 [{r['A0']}] R2 [{r['R2']}] | label {lv}/{lmk} "
              f"({r['label_source']}, {r['label_confidence']}) -> OWNER {ov}/{omk} | {chg}{pmv}{note}")

    print("\n== 2. tier 1: what each charged false positive becomes (descriptive; NOT a purity -- the re-grade is d116_grade.py --owner-record)")
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

    print("\n== 3. controls (the doc-103 reading: >= 3 stopper-class changes -> not settled)")
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
    if DET == "pdvd":                                  # PDVD Michel truth is the verdict: a class change moves both axes
        within = [k for k in ctl if not ch(k)[0] and ch(k)[1]]
        print(f"  Michel call changed with the stopper class held (STM_MICHEL <-> STM_ONLY): {len(within)} of {len(ctl)}: {within}")
    print(f"  null controls {len(nulls)}; shared-FP control(s) reported apart: "
          f"{[(k, rows[k]['fp_in'], rows[k]['label_verdict'] + '/' + rows[k]['label_michel_kind'], L[k]['label'] + '/' + str(L[k].get('michel_kind'))) for k in shared]}")

    rec = []
    for k, x in L.items():
        r = rows[k]
        omk = None if x.get("michel_kind") in UNSET else x.get("michel_kind")
        rec.append(dict(
            key=k, verdict=x["label"], michel_kind=omk, confidence="owner", source=f"{tag} (owner, {DATE})",
            owner_review=dict(verdict=x["label"], michel_kind=omk, notes=x.get("notes") or "", date=DATE,
                              tag=tag, question=f"{rule}: owner look at R2's false positives on the new trajectory"),
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

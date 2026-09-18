#!/usr/bin/env python3
"""doc pdvd/115 round 2 sec 7.4 -- where do the FIT and D-3live stretches an arm GAINS come from?  Read-only.

Joins the base and the arm stretch catalogues (d114_support_census.py <out>_stretches.tsv) per (event, cluster):
an arm stretch is MATCHED to a base stretch when their centroids are within max(MIN_R, half the two chords) of each
other (the fit's row indices are not stable across arms, so the join is spatial), and NEW otherwise; the same in the
other direction gives the base stretches the arm LOST (census both directions).  For the two classes that grew
under the pricing in round 1 (FIT: a supported seed the fit leaves; D-3live: a detour whose seed vertices all see three
planes) the table lists, per class, the arm's stretches split into matched (and the base class they matched, i.e.
the migration) and new, with their length and max ridge offset, and the base's stretches lost.  The new FIT / D-3live
rows are written in the catalogue's own format (<out>_new_stretches.tsv) so d114_case_figs.py can draw them.

Usage: d115r2_adjudicate.py --base figs/115_support_pdhd_off_stretches.tsv --arm figs/115r2_support_pdhd_p3bwp05_stretches.tsv \
           --out figs/115r2_newstretch_pdhd [--top 8]
"""
import argparse, collections, csv, math

MIN_R = 5.0
CLASSES = ("FIT", "D-3live")


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["event"], r["cluster"])].append(r)
    return rows, by


def match(r, cands):
    """the nearest candidate stretch by centroid within the overlap radius, else None."""
    best, bd = None, None
    for c in cands:
        d = math.dist((float(r["mx"]), float(r["my"]), float(r["mz"])), (float(c["mx"]), float(c["my"]), float(c["mz"])))
        rad = max(MIN_R, 0.5 * (float(r["chord_cm"]) + float(c["chord_cm"])))
        if d <= rad and (bd is None or d < bd):
            best, bd = c, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    brows, bby = load(a.base)
    arows, aby = load(a.arm)
    L = [f"# doc pdvd/115 round 2: FIT / D-3live provenance, base {a.base} vs arm {a.arm}; join radius max({MIN_R} cm, mean chord)"]
    new_rows = []
    for klass in CLASSES:
        arm_k = [r for r in arows if r["class"] == klass]
        base_k = [r for r in brows if r["class"] == klass]
        matched = collections.Counter(); m_len = collections.Counter()
        new = []
        for r in arm_k:
            m = match(r, bby.get((r["event"], r["cluster"]), []))
            if m is None:
                new.append(r)
            else:
                matched[m["class"]] += 1; m_len[m["class"]] += float(r["length_cm"])
        lost = [r for r in base_k if match(r, aby.get((r["event"], r["cluster"]), [])) is None]
        L.append(f"\n## {klass}: base {len(base_k)} stretches {sum(float(r['length_cm']) for r in base_k)/100:.2f} m; "
                 f"arm {len(arm_k)} stretches {sum(float(r['length_cm']) for r in arm_k)/100:.2f} m")
        L.append(f"  arm stretches matched to a base stretch: {sum(matched.values())} "
                 f"(by base class: {', '.join(f'{k} {v} ({m_len[k]/100:.2f} m)' for k, v in matched.most_common())})")
        L.append(f"  arm stretches NEW (no base stretch of any class at the place): {len(new)}, "
                 f"{sum(float(r['length_cm']) for r in new)/100:.2f} m, max ridge offset median "
                 f"{sorted(float(r['max_d_ridge']) for r in new)[len(new)//2] if new else float('nan'):.2f} cm, "
                 f"open-ended {sum(int(r['open_ended']) for r in new)}, >= 3 rows {sum(int(r['nrows']) >= 3 for r in new)}")
        L.append(f"  base stretches LOST (no arm stretch of any class at the place): {len(lost)}, "
                 f"{sum(float(r['length_cm']) for r in lost)/100:.2f} m")
        new.sort(key=lambda r: -float(r["length_cm"]) * float(r["max_d_ridge"]))
        L.append(f"  top {a.top} new by length x max ridge offset (event cluster pass rows length_cm max_d_ridge max_d_img max_noff seed_vertices):")
        for r in new[:a.top]:
            L.append(f"    {r['event']} {r['cluster']} {r['pass']} {r['row_a']}-{r['row_b']} {r['length_cm']} {r['max_d_ridge']} {r['max_d_img']} "
                     f"{r['max_noff']} {r['seed_vertices'][:80]}")
        new_rows.extend(new)
    hdr = list(arows[0].keys()) if arows else []
    with open(a.out + "_new_stretches.tsv", "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in new_rows:
            fh.write("\t".join(r[k] for k in hdr) + "\n")
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

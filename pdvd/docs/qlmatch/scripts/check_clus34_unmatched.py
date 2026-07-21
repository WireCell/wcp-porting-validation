#!/usr/bin/env python3
"""Why PDVD 039252 evt298567 apa-0 cluster ident=34 matches no flash.

Repro:
    cd pdvd
    python3 docs/qlmatch/scripts/check_clus34_unmatched.py

Reads the Q/L calib dumps (trim off = work/039252_0_keep, trim on =
work/039252_0_gaptrim) plus the Bee zip, prints the diagnosis, and writes the
three figures used by docs/qlmatch/15_pdvd-clus34-unmatched-evt298567.md.

The claim under test: 8 points carrying 0.08% of cluster 34's charge sit 28.4 cm
off its anode end and stretch its raw drift extent to 371.7 cm -- past the
336.9 cm bottom-volume drift depth -- so no flash T0 can place it inside the TPC
box, every candidate bundle fails containment, and require_containment drops them
all before the fit.
"""
import json
import os
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
KEEP = os.path.join(PDVD, "work", "039252_0_keep")
GAPTRIM = os.path.join(PDVD, "work", "039252_0_gaptrim")
WALKFLOOR = os.path.join(PDVD, "work", "039252_0_walkfloor")
PICS = os.path.join(PDVD, "docs", "qlmatch", "pics")
UID = 34            # ql-display "clus 34" == calib ident 34, apa 0 (uid 34)
GAP_CM = 3.0        # robust_endpoint_gap used by the demo (PDHD's value)


def load(path):
    with open(path) as f:
        return json.load(f)


def selections(dump):
    """uid -> set of auto-selected flash gids."""
    sel = {}
    for b in dump["bundles"]:
        if b.get("auto_selected"):
            sel.setdefault(b["main_cluster"], set()).add(b["flash_gid"])
    return sel


def pool_sizes(dump):
    """uid -> number of bundles surviving into the dump (contained ones only)."""
    pool = {}
    for b in dump["bundles"]:
        pool[b["main_cluster"]] = pool.get(b["main_cluster"], 0) + 1
    return pool


def main():
    os.makedirs(PICS, exist_ok=True)
    base = load(os.path.join(KEEP, "calib-evt298567.json"))
    clusters = {c["uid"]: c for c in base["clusters"]}
    geom = base["geometry"]["0"]
    box = abs(geom["anode_x"] - geom["cathode_x"])

    c34 = clusters[UID]
    x = np.array(c34["x"])
    y = np.array(c34["y"])
    z = np.array(c34["z"])
    q = np.array(c34["q"])
    order = np.argsort(x)
    xs, qs = x[order], q[order]

    # --- the detached fragment -------------------------------------------------
    gaps = np.diff(xs)
    big = np.where(gaps > GAP_CM)[0]
    print(f"apa-0 drift box: anode {geom['anode_x']:.2f} .. cathode "
          f"{geom['cathode_x']:.2f} cm  => depth {box:.2f} cm")
    print(f"clus {UID}: npoints={c34['npoints']} raw drift extent "
          f"{xs.max() - xs.min():.2f} cm")
    for i in big:
        print(f"  gap {gaps[i]:.2f} cm at x {xs[i]:.2f} -> {xs[i+1]:.2f}: "
              f"{i+1} pts outside, carrying "
              f"{qs[:i+1].sum() / q.sum() * 100:.3f}% of the charge")
    if len(big):
        i = big[0]
        outer = order[: i + 1]
        inner = order[i + 1:]
        print(f"  drop those {i+1} pts => extent {xs.max() - xs[i+1]:.2f} cm "
              f"(fits the ~342.9 cm ext-widened window)")
    else:
        outer, inner = np.array([], int), order

    # --- fig 1: projections ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (h, v, hl, vl) in zip(axes, [
            (x, y, "raw drift x [cm]", "y [cm]"),
            (x, z, "raw drift x [cm]", "z [cm]"),
            (z, y, "z [cm]", "y [cm]")]):
        ax.scatter(h[inner], v[inner], s=1.5, c="0.25", label="track body (2641 pts)")
        ax.scatter(h[outer], v[outer], s=42, c="crimson", marker="o",
                   edgecolors="k", linewidths=0.4, zorder=5,
                   label=f"detached fragment ({len(outer)} pts, 0.08% of q)")
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[:2]:
        ax.axvline(geom["anode_x"], color="tab:blue", ls=":", lw=1.4)
        ax.text(geom["anode_x"], ax.get_ylim()[1], " bottom anode", color="tab:blue",
                fontsize=7, va="top")
    axes[0].annotate("", xy=(xs[big[0]], y[order][big[0]]),
                     xytext=(xs[big[0] + 1], y[order][big[0] + 1]),
                     arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
    axes[0].text(0.5 * (xs[big[0]] + xs[big[0] + 1]), y[order][big[0]] - 9,
                 f"{gaps[big[0]]:.1f} cm gap", color="crimson", fontsize=9,
                 ha="center", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("PDVD 039252 evt298567 — ql clus 34 (apa 0) = Bee img-global 35: "
                 "one clean track + an 8-point detached fragment", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "clus34_projections.png"), dpi=130)
    plt.close(fig)

    # --- fig 2: extent census --------------------------------------------------
    spans, uids = [], []
    for uid, c in clusters.items():
        if c["apa"] != 0:
            continue
        cx = np.array(c["x"])
        spans.append(cx.max() - cx.min())
        uids.append(uid)
    spans = np.array(spans)
    order2 = np.argsort(spans)[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["crimson" if uids[i] == UID else "0.55" for i in order2]
    ax.bar(range(len(spans)), spans[order2], color=colors)
    ax.axhline(box, color="tab:blue", ls="--", lw=1.5,
               label=f"drift depth {box:.1f} cm — a single-T0 cluster cannot exceed this")
    # The ext-widened window is ~343 cm (u_cathode 336.91 + cathode_ext1 2.0 +
    # anode floor 4.0).  Drawn as approximate on purpose: ident 1 (342.94 cm) is
    # contained in practice, so the true edge is >= 342.94 -- the census only needs
    # to show that clus 34 at 371.66 clears it by ~29 cm and the trimmed 342.05 does
    # not.  Chasing the exact edge is the open cathode_ext1 / anode-pull item.
    ax.axhline(342.9, color="tab:green", ls=":", lw=1.5,
               label="~343 cm ext-widened containment window (approximate)")
    trimmed = xs.max() - xs[big[0] + 1]
    ax.plot([0], [trimmed], marker="v", ms=11, color="tab:orange", zorder=6,
            label=f"clus 34 after the gap trim: {trimmed:.1f} cm — now contained")
    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels([uids[i] for i in order2], rotation=90, fontsize=6)
    ax.set_xlabel("apa-0 cluster ident (as shown in the ql display)")
    ax.set_ylabel("raw drift extent [cm]")
    ax.set_title("PDVD 039252 evt298567 — clus 34 is the only apa-0 cluster longer "
                 "than the drift volume")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y", lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "clus34_extent_census.png"), dpi=130)
    plt.close(fig)

    # --- fig 3: the Bee id bridge ---------------------------------------------
    zf = zipfile.ZipFile(os.path.join(KEEP, "mabc-all-apa.zip"))
    bee = json.loads(zf.read("data/0/0-img-global.json"))
    bx = np.array(bee["x"])
    by = np.array(bee["y"])
    bz = np.array(bee["z"])
    bc = np.array(bee["cluster_id"])
    pts = np.vstack([x, y, z]).T
    B = np.vstack([bx, by, bz]).T
    rng = np.random.default_rng(0)
    probe = rng.choice(len(pts), size=min(400, len(pts)), replace=False)
    votes = {}
    for p in pts[probe]:
        j = np.abs(B - p).sum(1).argmin()
        votes[int(bc[j])] = votes.get(int(bc[j]), 0) + 1
    bee_id = max(votes, key=votes.get)
    print(f"\nql clus ident {UID} (apa 0)  ==  Bee img-global cluster_id {bee_id} "
          f"({votes[bee_id]}/{len(probe)} probes)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sel = bc == bee_id
    axes[0].scatter(bx[~sel], by[~sel], s=0.5, c="0.85")
    axes[0].scatter(bx[sel], by[sel], s=2.0, c="crimson")
    axes[0].set_title(f"Bee img-global — cluster_id {bee_id} highlighted\n"
                      f"(all {len(set(bc.tolist()))} clusters, both drift volumes)")
    axes[1].scatter(x, y, s=2.0, c="crimson")
    axes[1].set_title(f"ql calib dump — clus ident {UID}, apa 0 (uid {UID})\n"
                      f"same {len(x)} points, same raw coords")
    for ax in axes:
        ax.set_xlabel("raw drift x [cm]")
        ax.set_ylabel("y [cm]")
        ax.grid(alpha=0.25, lw=0.4)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    fig.suptitle(f"The id spaces differ: ql display says \"clus {UID}\", "
                 f"Bee says cluster {bee_id} — same object", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "clus34_bee_id_bridge.png"), dpi=130)
    plt.close(fig)

    # --- the demo run ----------------------------------------------------------
    demo = os.path.join(GAPTRIM, "calib-evt298567.json")
    if not os.path.exists(demo):
        print("\n(no gaptrim dump yet; run with PDVD_QL_ROBUST_TRIM=1 -s gaptrim)")
        return
    new = load(demo)
    sb, sn = selections(base), selections(new)
    pb, pn = pool_sizes(base), pool_sizes(new)
    fl = {f["gid"]: f for f in new["flashes"]}
    print(f"\ntrim ON: dumped bundles {len(base['bundles'])} -> {len(new['bundles'])}")
    print(f"         clus {UID} pool {pb.get(UID,0)} -> {pn.get(UID,0)}")
    for gid in sorted(sn.get(UID, [])):
        f = fl[gid]
        print(f"         clus {UID} MATCHED flash gid={gid} t={f['time']:.2f} us "
              f"PE={f['total_PE']:.1f}")
    changed = [u for u in sorted(set(sb) | set(sn)) if sb.get(u, set()) != sn.get(u, set())]
    print(f"\n         side effects: {len(changed)} of {len(set(sb)|set(sn))} "
          f"clusters change their auto-selection")
    for u in changed:
        print(f"           uid={u}: {sorted(sb.get(u,set())) or 'UNMATCHED'} -> "
              f"{sorted(sn.get(u,set())) or 'UNMATCHED'}")

    geometric_check(new, clusters, geom)
    sibling_check(base, new)
    walkfloor_check(base)


def geometric_check(new, clusters, geom):
    """Is gid 91 the right flash?  Answered WITHOUT the fit (chi2/KS would be
    circular -- the matcher picked gid 91 by them).  A full-drift track pins its own
    T0 to ~1 cm, so ask which flashes place it inside the box at all."""
    print("\n--- independent geometric check of the clus 34 match ---")
    c = clusters[UID]
    body = np.sort(np.array(c["x"]))[8:]          # the 8 trimmed points removed
    v = new["drift_speed"]
    lo_u, hi_u = -4.0, geom["u_cathode"] + 2.0    # anode floor, cathode_ext1 2.0
    off_min = lo_u + geom["anode_x"] - body.min()
    off_max = hi_u + geom["anode_x"] - body.max()
    print(f"  required flash_x_offset window: [{off_min:.2f}, {off_max:.2f}] cm "
          f"(width {off_max - off_min:.2f} cm, out of ~800 cm possible)")
    hits = []
    for f in new["flashes"]:
        off = -f["time"] * v                      # sign_offset -1 for apa 0
        if off_min <= off <= off_max:
            hits.append(f["gid"])
            print(f"  gid {f['gid']} t={f['time']:.2f} us PE={f['total_PE']:.1f} "
                  f"-> offset {off:.2f} cm  INSIDE")
    print(f"  flashes of {len(new['flashes'])} that geometrically contain the track: {hits}")


def sibling_check(base, new):
    """The trim does NOT fix apa-4 ident 1 vs flash gid 120.  Shows why: the walk's
    break threshold (anode_in = -2.0) is 2 cm stricter than the containment floor
    (-4.0), so a body starting in that dead band drags real track charge into the
    'outside' pile and every judge then correctly refuses."""
    print("\n--- sibling case: apa-4 ident 1 (uid 4000001) vs flash gid 120 ---")
    for tag, d in (("trim off", base), ("trim on", new)):
        pool = sorted(b["flash_gid"] for b in d["bundles"]
                      if b["main_cluster"] == 4000001)
        print(f"  {tag:9s}: pool {len(pool)} gids {pool[:3]}...{pool[-1] if pool else '-'}"
              f"  (stops short of 120 either way)")
    g = new["geometry"]["4"]
    v = new["drift_speed"]
    c = {cc["uid"]: cc for cc in new["clusters"]}[4000001]
    x = np.array(c["x"])
    q = np.array(c["q"])
    fl = {f["gid"]: f for f in new["flashes"]}
    anode_in, floor = -2.0, -4.0
    allow = max(0.01 * len(x), 15)
    for gid in (119, 120):
        f = fl[gid]
        off = g["sign_offset"] * f["time1"] * v   # top volume uses time1
        u = g["s"] * (x + off - g["anode_x"])
        o = np.argsort(u)
        us, qs = u[o], q[o]
        brk = int(np.searchsorted(us, anode_in, side="right"))
        q_out = qs[:brk].sum()
        fires = (brk <= allow) or (q_out <= 0.01 * q.sum())
        first_u = us[brk] if fires and brk < len(us) else us.min()
        print(f"  gid {gid} (t1={f['time1']:.1f}): first_u={us.min():.2f}, walk swallows "
              f"{brk} pts / {q_out/q.sum()*100:.2f}% of q "
              f"({q_out/max(brk,1):.0f} q/pt) -> trim fires={fires}, "
              f"first_u->{first_u:.2f}, contained={first_u > floor}")


def walkfloor_check(base):
    """robust_endpoint_walk_to_floor: break the anode-end walk at the floor the
    containment gate actually uses, so it stops counting material the gate would have
    accepted.  Fixes apa-4 ident 1 without disturbing clus 34."""
    path = os.path.join(WALKFLOOR, "calib-evt298567.json")
    if not os.path.exists(path):
        print("\n(no walkfloor dump; run with PDVD_QL_ROBUST_TRIM=1 "
              "PDVD_QL_ROBUST_WALK_FLOOR=1 -s walkfloor)")
        return
    w = load(path)
    sb, sw = selections(base), selections(w)
    pb, pw = pool_sizes(base), pool_sizes(w)
    fl = {f["gid"]: f for f in w["flashes"]}
    print("\n--- robust_endpoint_walk_to_floor ON ---")
    print(f"  dumped bundles {len(base['bundles'])} -> {len(w['bundles'])}; "
          f"auto-selected {sum(len(v) for v in sb.values())} -> "
          f"{sum(len(v) for v in sw.values())}")
    for uid, label in ((4000001, "apa-4 ident 1"), (UID, "apa-0 clus 34")):
        print(f"  {label}: pool {pb.get(uid,0)} -> {pw.get(uid,0)}, "
              f"selection {sorted(sb.get(uid,set())) or 'UNMATCHED'} -> "
              f"{sorted(sw.get(uid,set())) or 'UNMATCHED'}")
    # is ident 1's new pick better on non-fit grounds?
    for d, tag in ((base, "keep"), (w, "walkfloor")):
        for b in d["bundles"]:
            if b["main_cluster"] == 4000001 and b.get("auto_selected"):
                f = {x["gid"]: x for x in d["flashes"]}[b["flash_gid"]]
                print(f"    {tag:9s} ident 1 -> gid {b['flash_gid']}: PE={f['total_PE']:.1f} "
                      f"at_x_boundary={b['at_x_boundary']} strength={b['strength']:.3f} "
                      f"ks={b['ks_dis']:.3f}   (at_x_boundary is GEOMETRIC, not fitted)")
    changed = [u for u in sorted(set(sb) | set(sw)) if sb.get(u, set()) != sw.get(u, set())]
    print(f"  side effects: {len(changed)} of {len(set(sb)|set(sw))} clusters change")
    for u in changed:
        print(f"    uid={u}: {sorted(sb.get(u,set())) or 'UNMATCHED'} -> "
              f"{sorted(sw.get(u,set())) or 'UNMATCHED'}")


if __name__ == "__main__":
    main()

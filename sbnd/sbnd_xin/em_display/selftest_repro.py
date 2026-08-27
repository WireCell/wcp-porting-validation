"""doc pr/114: the reproduction check and the membership-repair check, over the
whole 94-event scan sample."""
import json, os, sys, glob, collections
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/em_display")
import em_geom as G

SX = "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"
PREP = os.path.join(SX, "em_display", "emprep")

man = {}
with open(os.path.join(SX, "em_display", "em114-manifest.tsv")) as fh:
    cols = fh.readline().rstrip("\n").split("\t")
    for ln in fh:
        f = ln.rstrip("\n").split("\t")
        man[f[cols.index("event")]] = dict(zip(cols, f))

nev = 0
diff_fields = collections.Counter()
bad_events = []
sh_tot = sh_lossy = sh_repaired = sh_exact = 0
axis_src = collections.Counter()

for evt, row in sorted(man.items(), key=lambda kv: int(kv[0])):
    arm = row["sample"]
    a_p = os.path.join(SX, "work-%s-prod0825" % arm, "pr_evt%s" % evt,
                       "calib-pr-evt%s.json" % evt)
    b_p = os.path.join(SX, "work-em114-%s" % arm, "pr_evt%s" % evt,
                       "calib-pr-evt%s.json" % evt)
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        continue
    nev += 1
    a = json.load(open(a_p))
    b = json.load(open(b_p))

    def strip(d):
        return dict(
            main_vertex=d.get("main_vertex"),
            kine=d.get("kine"), tagger=d.get("tagger"),
            showers={s["id"]: {k: v for k, v in s.items() if k != "shower_id"}
                     for s in (d.get("showers") or [])},
            segments={s["id"]: {k: v for k, v in s.items() if k != "points"}
                      for s in (d.get("segments") or [])})
    sa, sb = strip(a), strip(b)
    if sa != sb:
        for f in sa:
            if sa[f] != sb[f]:
                diff_fields[f] += 1
        bad_events.append(evt)

    # membership repair
    segs = a.get("segments") or []
    prep = json.load(open(os.path.join(PREP, "emprep-evt%s.json" % evt)))
    for sh in (a.get("showers") or []):
        sh_tot += 1
        j, n = G.join_completeness(sh, segs)
        pm = (prep.get("showers") or {}).get(str(sh["id"]))
        pn = len(pm["members"]) if pm else None
        if pn == n:
            sh_exact += 1
        if j != n:
            sh_lossy += 1
            if pn == n:
                sh_repaired += 1
        if pm and pm.get("dir15") and G.vmag(tuple(pm["dir15"])) > 0:
            axis_src["probe dir15"] += 1
        else:
            axis_src["python fallback"] += 1

print("events compared: %d" % nev)
print("REPRODUCTION (shower_id excluded -- it is a per-PROCESS counter):")
if not bad_events:
    print("   all %d events identical on main_vertex, kine, tagger, showers, segments" % nev)
else:
    print("   %d events differ; fields: %s" % (len(bad_events), dict(diff_fields)))
    print("   first few: %s" % bad_events[:8])
print()
print("MEMBERSHIP:")
print("   showers total                    : %d" % sh_tot)
print("   lossy segments[].shower_id join   : %d (%.1f%%)" % (sh_lossy, 100.0*sh_lossy/sh_tot))
print("   of those, repaired by the probe   : %d" % sh_repaired)
print("   probe member count == num_segments: %d/%d" % (sh_exact, sh_tot))
print()
print("AXIS SOURCE: %s" % dict(axis_src))

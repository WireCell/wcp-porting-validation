"""doc pr/114 -- headless self-test for the EM / pi0 hand-scan display.

Drives the viewer's own callbacks with no browser: the three regression cases
named in doc pr/114 sec 8 (the 0-of-5 lossy shower, the 43-of-50 one, and the
two-pi0 event whose kine block names a THIRD pairing), plus a mark/impact pass,
a hand-built pi0, the snap, a label round trip and the M13 tag guard.

Run:  python em_display/selftest_em_display.py        (expects 0 failures)
"""
import os, sys, json, shutil, collections

SX = "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"
TAG = "selftest114"
shutil.rmtree(os.path.join(SX, "em_labels", TAG), ignore_errors=True)

sys.argv = ["em_display_viewer.py", "--scan-tag", TAG]
sys.path.insert(0, os.path.join(SX, "em_display"))
os.chdir(SX)
import em_display_viewer as V  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print("%-58s %s %s" % (name, "PASS" if cond else "**FAIL**", detail))
    if not cond:
        fails.append(name)


check("events loaded from the manifest", len(V.LABELS) == 94, "%d" % len(V.LABELS))

# ---- regression case 1: the 0-of-5 lossy shower must not render as empty ----
V.event_select.value = "evt463565"
V.on_event(None, None, "evt463565")
row = None
for i, n in enumerate(V.shower_src.data["node"]):
    if n == 109073:
        row = i
check("evt463565 shower 109073 present in the table", row is not None)
if row is not None:
    joined = V.shower_src.data["joined"][row]
    nseg = V.shower_src.data["nseg"][row]
    note = V.shower_src.data["flag"][row]
    check("  ... membership repaired (not 0)", str(joined) == "5" and nseg == 5,
          "joined=%s nseg=%s" % (joined, nseg))
    check("  ... and the note says the join was lossy", "lossy" in note, note[:60])
check("evt463565 members_of returns 5", len(V.members_of(109073)) == 5,
      str(len(V.members_of(109073))))

# ---- regression case 2: the 43-of-50 shower still draws ----
V.on_event(None, None, "evt84229")
V.state["sel_shower"] = 69134
V.fill_cand_table()
V.draw_arrows()
ax, br, src = V.shower_axis(69134)
check("evt84229 shower 69134 has 50 members", len(V.members_of(69134)) == 50,
      str(len(V.members_of(69134))))
check("  ... axis is unit-length and from the probe",
      abs(V.G.vmag(ax) - 1.0) < 1e-6 and src == "probe", "%s %s" % (br, src))
check("  ... an axis arrow was drawn in >=2 panels",
      sum(1 for k in ("xy", "yz", "xz") if V.arrow_src[k].data["xs"]) >= 2)
check("  ... candidate table is populated", len(V.cand_src.data["sid"]) > 0,
      "%d rows" % len(V.cand_src.data["sid"]))
nsite = sum(1 for x in V.cand_src.data["site"] if x)
check("  ... absorb-site column is filled for the members that were absorbed",
      nsite > 0, "%d of %d rows carry a site" % (nsite, len(V.cand_src.data["site"])))
tiers = [t for t in V.cand_src.data["tier"] if t != "-"]
check("  ... some candidate falls inside a pass-1 tier", len(tiers) > 0,
      "%d in-tier" % len(tiers))

# ---- marking + impact ----
sid = V.cand_src.data["sid"][0]
V.cand_src.selected.indices = [0]
V.mark("out")()
check("marking a segment records it", V.state["marks"].get(sid) == "out")
check("  ... impact readout mentions the shower", "kine_charge" in V.impact.text)
check("  ... marked polyline pushed to the out layer",
      bool(V.out_src["xy"].data["xs"]))

# ---- regression case 3: the two-pi0 event, groups kept SEPARATE from kine ----
V.on_event(None, None, "evt21073")
V.mode_group.active = 1
V.on_mode(None, None, 1)
grp = V.G.pi0_groups(V.cur_showers())
check("evt21073 has exactly 2 accepted pi0 groups", len(grp) == 2, str(sorted(grp)))
masses = sorted(round(shl[0].get("pio_mass"), 1) for shl in grp.values())
check("  ... group masses are 111.2 and 127.2", masses == [111.2, 127.2], str(masses))
V.refresh_kine()
txt = V.kine_div.text
check("  ... kine block is labelled a BDT feature, not the pairing",
      "BDT feature" in txt and "NOT the pairing" in txt)
check("  ... the kine block's own 207.25 is shown separately", "207.2" in txt or "207.3" in txt)

# ---- build a pi0 by hand and get a mass ----
V.state["sel_shower"] = 60081
V.on_gamma(1)()
V.state["sel_shower"] = 31023
V.on_gamma(2)()
V.refresh_kine()
e1 = V.shower_energy(60081)
e2 = V.shower_energy(31023)
d1 = V.shower_axis(60081)[0]
d2 = V.shower_axis(31023)[0]
th = V.G.angle_deg(d1, d2)
m = V.G.pi0_mass(e1, e2, th)
check("hand-built gamma pair yields a mass", m is not None and m > 0,
      "E1=%.1f E2=%.1f theta=%.1f m=%.1f" % (e1, e2, th, m))
check("  ... both gamma markers drawn", len(V.gstart_src.data["x"]) == 2)

# back-projection path
V.vtx_mode_group.active = 1
v, how, detail = V.pio_vertex()
check("back-projection returns a verdict", detail.get("verdict") is not None,
      "%s gap=%s" % (detail.get("verdict"), detail.get("gap")))

# ---- back-projection: all three branches must be mirrored --------------
# 49 of the 78 accepted pairs on disk take the one-short branch, where the C++
# re-rays the short gamma and keeps the closest point on the LONG gamma's ray
# instead of the midpoint.  A midpoint-only mirror was wrong by a median 2.7 cm
# (p90 24.9, max 43.4) on exactly those pairs.
import glob as _glob
_br = collections.Counter()
for _p in sorted(_glob.glob(os.path.join(SX, "work-*-prod0825", "pr_evt*",
                                         "calib-pr-evt*.json"))):
    _d = json.load(open(_p))
    _g = V.G.pi0_groups(_d.get("showers") or [])
    if not _g:
        continue
    _anchor = V.G.pt(_d.get("main_vertex"))
    if _anchor is None:
        continue
    for _pid, _shl in _g.items():
        if len(_shl) == 2:
            _br[V.G.pi0_backproject(_shl[0], _shl[1], _d["segments"],
                                    _anchor)["branch"]] += 1
check("back-projection exercises both live branches",
      _br.get("both_long", 0) > 0 and _br.get("one_short", 0) > 0, str(dict(_br)))
check("  ... and the one-short branch is the majority (49/78)",
      _br.get("one_short", 0) == 49 and sum(_br.values()) == 78, str(dict(_br)))

# ---- snap ----
V.gstart_slot.active = 0
V.man_x.value, V.man_y.value, V.man_z.value = "-32.0", "26.0", "367.0"
V.on_snap()
check("snap moved gamma1's start onto a fitted point",
      V.state["gstart"][1] is not None, str(V.state["gstart"][1]))

# ---- save + reload round trip ----
V.em_verdict.active = 1
V.pio_verdict.active = 0
V.conf_group.active = 0
V.note_in.value = "selftest"
V.on_save()
p = V.label_path("evt21073")
check("label file written", os.path.exists(p), p)
rec = json.load(open(p))
check("  ... record carries the pi0 block", rec.get("pio") is not None)
check("  ... and the reco's own groups alongside the human's",
      len((rec["pio"] or {}).get("reco_groups") or {}) == 2)
check("  ... and the kine_pio_* block, separately",
      "kine_pio_mass" in ((rec["pio"] or {}).get("reco_kine") or {}))
check("  ... provenance recorded", bool(rec.get("source")) and bool(rec.get("arm"))
      and bool(rec.get("probe_sidecar")))
V.load("evt21073")
check("reload restores the pi0 verdict",
      V.pio_verdict.active == 0, str(V.pio_verdict.active))
check("reload restores both gamma slots",
      V.state["gamma"][1] == 60081 and V.state["gamma"][2] == 31023,
      str(V.state["gamma"]))

# ---- M13: an implicit tag must refuse to write into a populated dir ----
V.SCAN_TAG_EXPLICIT = False
check("implicit tag refuses to write into a populated tag", not V.write_allowed())
V.SCAN_TAG_EXPLICIT = True
check("explicit tag is consent", V.write_allowed())

print()
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)

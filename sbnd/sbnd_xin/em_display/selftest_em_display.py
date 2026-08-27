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


# ===========================================================================
# round 3 -- the 3-D view
# ===========================================================================
import math as _m                                                # noqa: E402
import zipfile as _zip                                           # noqa: E402
import em3d as D3                                                # noqa: E402

print()

# ---- camera geometry -------------------------------------------------------
_ortho_ok = _inv_ok = _fit_ok = True
for _i in range(13):
    for _j in range(9):
        _az = -_m.pi + 2 * _m.pi * _i / 12.0
        _el = -1.4 + 2.8 * _j / 8.0
        r, u, f = D3.camera_basis(_az, _el)
        for _a in (r, u, f):
            if abs(sum(v * v for v in _a) - 1.0) > 1e-12:
                _ortho_ok = False
        for _a, _b in ((r, u), (u, f), (f, r)):
            if abs(sum(x * y for x, y in zip(_a, _b))) > 1e-12:
                _ortho_ok = False
        for _p in ((17.0, -3.5, 220.0), (-201.0, 199.0, 0.9), (0.0, 0.0, 0.0)):
            _c = (5.0, -2.0, 250.0)
            (_u, _v, _d), = D3.project([_p], _az, _el, _c)
            _r2 = sum((a - b) ** 2 for a, b in zip(_p, _c))
            if abs(_u * _u + _v * _v + _d * _d - _r2) > 1e-8:
                _inv_ok = False
check("camera basis is orthonormal over a (az, el) grid", _ortho_ok)
check("  ... so u^2 + v^2 + d^2 == |p - centre|^2 exactly", _inv_ok)

# The framing guarantee the whole zoom story rests on: |(u,v)| <= R for EVERY
# camera, so rotating cannot push the event out of frame or shrink it to a dot.
_pts = [(x, y, z) for x in (-180.0, 12.0, 190.0) for y in (-150.0, 0.0, 170.0)
        for z in (5.0, 250.0, 495.0)]
_c, _R = D3.bounding_sphere(_pts)
for _i in range(24):
    for _j in range(7):
        for (_u, _v, _d) in D3.project(_pts, -_m.pi + _m.pi * _i / 12.0,
                                       -1.5 + 3.0 * _j / 6.0, _c):
            if _u * _u + _v * _v > _R * _R + 1e-6:
                _fit_ok = False
check("bounding sphere frames every camera (no rotation can escape it)", _fit_ok,
      "R=%.1f" % _R)
check("  ... and it has a floor so a tiny event is not magnified to noise",
      D3.bounding_sphere([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])[1] >= 30.0)

# Three presets must reproduce the 2-D panels exactly, or "step back to a view
# you trust" is a lie.
_r, _u, _f = D3.camera_basis(_m.radians(-90.0), 0.0)
check("preset x-z reproduces the X-Z panel (right=+x, up=+z)",
      max(abs(_r[0] - 1), abs(_r[1]), abs(_r[2])) < 1e-9
      and max(abs(_u[0]), abs(_u[1]), abs(_u[2] - 1)) < 1e-9)
_r, _u, _f = D3.camera_basis(0.0, 0.0)
check("preset z-y reproduces the Y-Z panel (right=+y, up=+z)",
      max(abs(_r[0]), abs(_r[1] - 1), abs(_r[2])) < 1e-9
      and max(abs(_u[0]), abs(_u[1]), abs(_u[2] - 1)) < 1e-9)

# ---- the frame the charge cloud is in -- doc pr/114 sec 11 -----------------
# THE blocker of this round, pinned so it cannot rot: the calib dump and the Bee
# zip's own PR layers are the same numbers, which is what licenses drawing
# clustering-global (the corrected frame, doc pr/13) under the skeleton.
_res = []
for _lbl in ("evt21073", "evt84229", "evt463565"):
    _row = V.MANIFEST[_lbl[3:]]
    _idx = D3.bee_event_index(SX, _row, _lbl[3:])
    _zp = D3.bee_zip_path(SX, _row)
    with _zip.ZipFile(_zp) as _z:
        _tf = json.loads(_z.read("data/%d/%d-track_fit-global.json" % (_idx, _idx)))
    _T = list(zip(_tf["x"], _tf["y"], _tf["z"]))
    _d = json.load(open(V.EVENTS[_lbl]))
    _P = [(p["x"], p["y"], p["z"]) for s in _d["segments"]
          for p in (s.get("points") or [])]
    _nn = []
    for _p in _P[::7]:
        _nn.append(min((_p[0] - t[0]) ** 2 + (_p[1] - t[1]) ** 2
                       + (_p[2] - t[2]) ** 2 for t in _T) ** 0.5)
    _nn.sort()
    _res.append((_lbl, _nn[len(_nn) // 2], _nn[-1]))
check("dump fit points ARE the Bee track_fit-global layer (same frame)",
      all(m < 0.001 for _, m, _x in _res),
      "; ".join("%s med %.5f max %.5f cm" % r for r in _res))

# ---- the cloud loader ------------------------------------------------------
_row = V.MANIFEST["21073"]
_cl = D3.load_bee_cloud(SX, _row, "21073", max_pts=5000)
check("cloud loader honours the point budget exactly",
      _cl is not None and _cl["kept"] == 5000 and _cl["total"] > 5000,
      "kept %s of %s" % (_cl["kept"], _cl["total"]))
check("  ... and returns equal-length columns",
      len({len(_cl[k]) for k in ("x", "y", "z", "q", "cid20")}) == 1)
check("  ... a budget above the total keeps everything",
      D3.load_bee_cloud(SX, _row, "21073", max_pts=10 ** 7)["kept"] == _cl["total"])
check("  ... every cluster survives decimation (proportional, not truncating)",
      len({int(v) for v in D3.load_bee_cloud(SX, _row, "21073",
                                             max_pts=2000)["cid20"]})
      == len({int(v) for v in D3.load_bee_cloud(SX, _row, "21073",
                                                max_pts=10 ** 7)["cid20"]}))
check("  ... a missing zip degrades to None, it does not raise",
      D3.load_bee_cloud(SX, {"bee_round": "nope/nope", "bee_url": "x/event/0/"},
                        "21073") is None)
check("  ... img-global is available but is the RAW frame",
      D3.CLOUD_LAYERS == ["clustering-global", "img-global"])

# ---- the layer contract ----------------------------------------------------
# A 3-D renderer not registered under a layer key would silently ignore the
# checkboxes; one registered under a key nobody lists would be unreachable.
check("every RENDER key is a listed layer",
      set(V.RENDER) == set(V.LAYER_KEYS),
      "render=%s layers=%s" % (sorted(V.RENDER), sorted(V.LAYER_KEYS)))
_f3d_r = set(V.f3d.renderers)
_reg = {r for rs in V.RENDER.values() for r in rs}
_unreg = [r for r in _f3d_r if r not in _reg]
check("every 3-D renderer is layer-controlled except the pick surface",
      _unreg == [V.r_pick], "%d unregistered" % len(_unreg))
V.layer_group.active = [i for i in range(len(V.LAYER_KEYS))]
V.apply_layers(None, None, None)
check("  ... turning every layer on shows exactly one cloud colour mode",
      V.r_cloud_c.visible != V.r_cloud_q.visible)
V.layer_group.active = [i for i, k in enumerate(V.LAYER_KEYS) if k != "cloud"]
V.apply_layers(None, None, None)
check("  ... and the cloud checkbox hides BOTH colour modes",
      not V.r_cloud_c.visible and not V.r_cloud_q.visible)
V.layer_group.active = [0, 1, 2, 3, 4, 6, 7, 8]
V.apply_layers(None, None, None)

# ---- the CustomJS, linted (there is no JS engine in this tree) --------------
_JS_KEYWORDS = set("""break case catch class const continue debugger default
delete do else export extends finally for function if import in instanceof let
new return super switch this throw try typeof var void while with yield of
true false null undefined Math Number Array Float64Array Float32Array Object
JSON String Boolean console window document isNaN parseFloat parseInt NaN
Infinity cb_obj cb_data""".split())


def js_free_names(code):
    """Identifiers a CustomJS body reads without declaring: they must all be
    supplied through `args` or the browser throws a ReferenceError that nothing
    here would ever see.

    Deliberately crude, and it has to handle multi-declarator statements --
    `const rx = -_sa, ry = _ca, rz = 0.0;` declares three names, and a regex that
    only takes the first reports the other two as missing args.  That false
    positive is what this comment exists to stop somebody re-introducing."""
    import re
    body = re.sub(r"//[^\n]*", "", code)
    body = re.sub(r'"[^"]*"|\'[^\']*\'', '""', body)
    decl = set()
    for m in re.finditer(r"\b(?:const|let|var)\b([^;]*)", body):
        seg = m.group(1)
        decl |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=(?!=)", seg))
        decl |= {n.strip() for n in seg.split(",")
                 if re.fullmatch(r"\s*[A-Za-z_$][\w$]*\s*", n)}
    body = re.sub(r"\.\s*[A-Za-z_$][\w$]*", "", body)
    names = set(re.findall(r"[A-Za-z_$][\w$]*", body))
    return {n for n in names if n not in decl and n not in _JS_KEYWORDS}


_js_all = [("panstart", V.js_panstart), ("rotate", V.js_rotate),
           ("panend", V.js_panend), ("apply", V.js_apply)]
_js_all += [("preset[%d]" % i, cb) for i, cb in enumerate(V.preset_js)]
_missing = []
for _nm, _cb in _js_all:
    for _free in sorted(js_free_names(_cb.code)):
        if _free not in _cb.args:
            _missing.append("%s:%s" % (_nm, _free))
check("every CustomJS free name is supplied through args",
      not _missing, ", ".join(_missing) or "%d handlers" % len(_js_all))
# Test the test: a linter that never fires proves nothing.  It must flag a name
# that is genuinely undefined and must NOT flag a second declarator or a
# property, which is exactly where the first version of it was wrong.
check("  ... and the linter itself catches a genuinely missing name",
      "nosuchthing" in js_free_names("const a = 1, b = 2;\n"
                                     "cam.data.az[0] = a + b + nosuchthing;")
      and "b" not in js_free_names("const a = 1, b = 2; a = b;")
      and "az" not in js_free_names("cam.data.az[0] = 1;"))
_bal_ok = True
for _nm, _cb in _js_all:
    _s = _cb.code
    for _o, _c in "{}", "()", "[]":
        if _s.count(_o) != _s.count(_c):
            _bal_ok = False
check("  ... and their brackets balance", _bal_ok)

# A shape trap worth pinning, stated as it actually is: Bokeh serialises a Python
# dict as {"type":"map", ...}, and bokehjs's `_decode_map` (bokeh.js) returns a
# plain JS object ONLY when every key is a string -- one non-string key and the
# handler receives a real Map instead, so `cfg.foo` is undefined, sizes go NaN,
# and the layer vanishes with no server-side error anywhere to notice.
_dicty = []
for _nm, _cb in _js_all:
    for _k, _val in _cb.args.items():
        for _cand in ([_val] + list(_val) if isinstance(_val, (list, tuple))
                      else [_val]):
            if isinstance(_cand, dict) and not all(isinstance(kk, str)
                                                   for kk in _cand):
                _dicty.append("%s:%s" % (_nm, _k))
check("no CustomJS arg is a non-string-keyed dict (that arrives as a JS Map)",
      not _dicty, ", ".join(sorted(set(_dicty))) or "checked %d handlers"
      % len(_js_all))
# ... and the one table that both mirrors read really is one table.
check("  ... base size/alpha have a single source of truth",
      [V._PT_CFG[s][0] for s in V._PT_SRC] == V._PT_SIZE
      and [V._PT_CFG[s][1] for s in V._PT_SRC] == V._PT_ALPHA
      and V.js_apply.args["ptsize"] is V._PT_SIZE)
# bokehjs's Toolbar._active_change writes the live gesture state to
# `gestures[et].active` and never to `active_drag`, which stays at whatever it
# was configured to.  A guard on active_drag would therefore never fire and
# rotation would fight box-select on every drag -- with nothing on the server to
# show for it.  There is no JS engine here to prove the fix, so pin the shape.
_rot_code = __import__("re").sub(r"//[^\n]*", "", V.js_rotate.code)
check("rotate guards on the LIVE gesture state, not the config property",
      "gestures" in _rot_code and "pan.active" in _rot_code
      and "active_drag" not in _rot_code,
      "(comments stripped; the code names active_drag only to warn against it)")
check("  ... and Python still starts with no drag tool auto-activated",
      V.f3d.toolbar.active_drag is None)
check("the viewer splices em3d's JS, it does not keep a second copy",
      D3.JS_PROJECT in V.js_rotate.code and D3.JS_PROJECT in V.js_apply.code
      and "const rx =" not in open(os.path.join(SX, "em_display",
                                                "em_display_viewer.py")).read())

# ---- 3-D selection resolves to SEGMENTS ------------------------------------
V.on_event(None, None, "evt84229")
V.tap_action.value = V.TAP_SELECT
_sids = V.pick_src.data["sid"]
_idx = [i for i, s in enumerate(_sids) if s == _sids[0]]
V.pick_src.selected.indices = _idx + [len(_sids) - 1]
check("a 3-D box over many points resolves to a handful of segments",
      len(V.selected_cand_ids()) <= 2 and len(_idx) > 2,
      "%d points -> %d segment(s)" % (len(_idx) + 1, len(V.selected_cand_ids())))
check("  ... and the cyan halo shows exactly what would be marked",
      len(V.sel3_src.data["xs3"]) == len(V.selected_cand_ids())
      and len(V.sel_src["xy"].data["xs"]) == len(V.selected_cand_ids()),
      "%d polylines" % len(V.sel3_src.data["xs3"]))
V.state["sel_shower"] = 69134
V.fill_cand_table()
V.mark("in")()
check("  ... and marking works off that 3-D selection",
      V.state["marks"].get(_sids[0]) == "in")
V.pick_src.selected.indices = []

# tap in "fill x/y/z" mode must land on a REAL fitted point (a ray needs an
# anchor; this is the 3-D answer to the two-panel tap).
V.tap_action.value = V.TAP_XYZ
V.pick_src.selected.indices = [5]
_want = (V.pick_src.data["x"][5], V.pick_src.data["y"][5], V.pick_src.data["z"][5])
check("tap in fill mode writes a real fitted point into x/y/z",
      abs(float(V.man_x.value) - _want[0]) < 0.06
      and abs(float(V.man_y.value) - _want[1]) < 0.06
      and abs(float(V.man_z.value) - _want[2]) < 0.06,
      "%s %s %s" % (V.man_x.value, V.man_y.value, V.man_z.value))

# ---- round 4: a tap that IS the mark ---------------------------------------
V.state["marks"] = {}
_target = _sids[7]
_hit = [i for i, s in enumerate(_sids) if s == _target][:3]
for want in ("in", "out", None):
    V.tap_action.value = V.TAP_TOGGLE
    V.pick_src.selected.indices = list(_hit)
    got = V.state["marks"].get(_target)
    check("toggle tap: %s -> %s" % (want and "next" or "third", want or "cleared"),
          got == want, "seg %s -> %r" % (_target, got))
    # THE re-arm check: Bokeh does not re-fire selected.indices for the same
    # index, so a toggle that did not clear its own selection would be dead
    # after one click.  The loop above only works if _clear_pick ran.
    check("  ... and the selection re-armed for the next tap on the SAME segment",
          list(V.pick_src.selected.indices) == [])
V.tap_action.value = V.TAP_IN
V.pick_src.selected.indices = list(_hit)
check("tap in 'mark IN' mode marks on the click itself",
      V.state["marks"].get(_target) == "in")
# Found by the REAL browser, not by this file: a selection left standing from
# the previous action makes the first gesture of the new one a no-op, because
# Bokeh fires selected.indices only on a change and the index list is identical.
V.pick_src.selected.indices = list(_hit)
V.tap_action.value = V.TAP_OUT
check("  ... and switching the action drops the standing selection, so the "
      "next gesture fires",
      list(V.pick_src.selected.indices) == []
      and not V.sel3_src.data["xs3"])
V.tap_action.value = V.TAP_OUT
V.pick_src.selected.indices = list(_hit)
check("  ... and 'mark OUT' overrides it", V.state["marks"].get(_target) == "out")

# ---- round 4: orbit centre --------------------------------------------------
V.tap_action.value = V.TAP_CENTRE
_span0 = V.f3d.x_range.end - V.f3d.x_range.start
_R0 = V.state["cam_R"]
V.pick_src.selected.indices = [11]
_p = (V.pick_src.data["x"][11], V.pick_src.data["y"][11], V.pick_src.data["z"][11])
check("tap in 'orbit around it' re-centres the camera on that point",
      max(abs(V.state["cam_c"][i] - _p[i]) for i in range(3)) < 0.06,
      "centre %s" % (tuple(round(v, 1) for v in V.state["cam_c"]),))
check("  ... keeping the zoom (span unchanged) and centred on zero",
      abs((V.f3d.x_range.end - V.f3d.x_range.start) - _span0) < 1e-6
      and abs(V.f3d.x_range.start + V.f3d.x_range.end) < 1e-6,
      "span %.1f" % (V.f3d.x_range.end - V.f3d.x_range.start))
check("  ... and NOT rewriting cam_R, which the depth cue normalises by",
      V.state["cam_R"] == _R0)
check("  ... the browser is told the new centre", V.cam_src.data["cx"][0] == _p[0])
# The ON-SCREEN reprojection is the browser's job: pushing cam_src.data fires
# em3d.JS_APPLY, which rewrites u/v in place for every source without shipping
# 25 000 points back.  So do NOT assert on the server's stale u/v columns here --
# assert the invariant the server actually owns, that its projection of the
# clicked point is now the origin, because that is what every later Python-side
# fill will use.
_pu, _pv, _ = V._proj([_p])[0]
check("  ... and the server's own projection now puts it at the origin",
      abs(_pu) < 1e-9 and abs(_pv) < 1e-9, "u=%.2e v=%.2e" % (_pu, _pv))
V.refit_camera()

# ---- round 4: a vertex is clickable, and can never be marked ----------------
V.tap_action.value = V.TAP_PIO
_vx = (V.vtx3_src.data["x"][2], V.vtx3_src.data["y"][2], V.vtx3_src.data["z"][2])
V.vtx3_src.selected.indices = [2]
check("tapping a reconstructed vertex sets the pi0 vertex",
      V.vtx_mode_group.active == 2
      and max(abs(V.state["vtx_manual"][i] - _vx[i]) for i in range(3)) < 0.06,
      "manual=%s" % (tuple(round(v, 1) for v in (V.state["vtx_manual"] or ())),))
check("  ... and the pi0 vertex marker moved there",
      V.piovtx_src.data["x"] and abs(V.piovtx_src.data["x"][0] - _vx[0]) < 0.06)
check("  ... x/y/z boxes agree with the state (no stale-read reentrancy)",
      abs(float(V.man_x.value) - _vx[0]) < 0.06)
_marks_before = dict(V.state["marks"])
V.tap_action.value = V.TAP_IN
V.vtx3_src.selected.indices = [3]
check("a vertex tap can NEVER reach state['marks']",
      V.state["marks"] == _marks_before, str(V.state["marks"]))
check("  ... and box-select cannot pick vertices up at all",
      [r.data_source for r in V._box3.renderers] == [V.pick_src]
      and V.vtx3_src in [r.data_source for r in V._tap3.renderers],
      "box sees %d source(s), tap sees %d"
      % (len(V._box3.renderers), len(V._tap3.renderers)))
V.tap_action.value = V.TAP_SELECT
V.vtx_mode_group.active = 0

# ---- round 4: the reco/hand distinction is DRAWN, not just stored -----------
V.on_event(None, None, "evt84229")
V.state["sel_shower"] = 69134
V.push_polys(V.mem_src, V.members_of(69134), V.mem3_src)
_mem = V.members_of(69134)
V.state["marks"] = {_mem[0]: "out", _sids[0] if _sids[0] not in _mem else _mem[1]: "in"}
V.refresh_marks()
check("reco membership and your marks are separate, simultaneous layers",
      len(V.mem3_src.data["xs3"]) == len(_mem)
      and len(V.out3_src.data["xs3"]) == 1 and len(V.in3_src.data["xs3"]) == 1,
      "member %d / in %d / out %d" % (len(V.mem3_src.data["xs3"]),
                                      len(V.in3_src.data["xs3"]),
                                      len(V.out3_src.data["xs3"])))
_order = [r.data_source for r in V.f3d.renderers if r in V.RENDER.get("mark", [])
          + V.RENDER.get("member", []) + V.RENDER.get("select", [])]
check("  ... and the widest halo is UNDERNEATH, so a mark cannot erase the "
      "reco band",
      _order.index(V.sel3_src) < _order.index(V.in3_src) < _order.index(V.mem3_src),
      " -> ".join({id(V.sel3_src): "select", id(V.in3_src): "in",
                   id(V.out3_src): "out", id(V.mem3_src): "member",
                   id(V.g1mem3_src): "g1", id(V.g2mem3_src): "g2"}[id(s)]
                  for s in _order))
_srcs = [r.data_source for r in V.f3d.renderers]
_iseg = _srcs.index(V.seg3_src)
_dash = [i for i, r in enumerate(V.f3d.renderers)
         if getattr(r.glyph, "line_dash", None) == "dashed"]
check("  ... the dashed repeat of your mark is drawn ON TOP of the segment",
      len(_dash) == 2 and min(_dash) > _iseg
      and {_srcs[i] for i in _dash} == {V.in3_src, V.out3_src},
      "segment at %d, dashes at %s" % (_iseg, _dash))
check("dim is OFF by default (hiding markable segments went backwards once)",
      V.dim_toggle.active is False and set(V.seg3_src.data["a"]) == {0.95})
# refresh_dim runs on EVERY mark, and since round 4 every tap is a mark.
# Assigning .data there re-serialises every polyline in all four segment sources
# (~7 400 coordinates) and ships them down the ssh tunnel to change one list of
# floats.  Static guard, in the spirit of the JS lint: it must patch, not assign.
import inspect as _insp                                             # noqa: E402
_dimsrc = _insp.getsource(V.refresh_dim)
_dimcode = __import__("re").sub(r"#[^\n]*", "", _dimsrc.split('"""')[-1])
check("  ... and marking patches the alpha column, it does not re-push geometry",
      ".patch(" in _dimcode and "m.data =" not in _dimcode
      and "continue" in _dimcode,
      "patch=%s assign=%s early-return=%s"
      % (".patch(" in _dimcode, "m.data =" in _dimcode, "continue" in _dimcode))
V.dim_toggle.active = True
V.refresh_dim()
_a = dict(zip(V.seg3_src.data["sid"], V.seg3_src.data["a"]))
check("  ... and with it on, members stay bright and the rest fade",
      _a[_mem[0]] > 0.9 and min(_a.values()) < 0.2,
      "members %.2f, faintest %.2f" % (_a[_mem[0]], min(_a.values())))
V.dim_toggle.active = False
V.refresh_dim()

# ---- round 4: a reopened label draws its own marks --------------------------
V.state["marks"] = {_mem[0]: "out"}
V.state["sel_shower"] = 69134
V.em_verdict.active = 1
V.on_save()
V.on_event(None, None, "evt21073")
V.on_event(None, None, "evt84229")
check("re-opening a labelled event DRAWS the marks it restores",
      V.state["marks"].get(_mem[0]) == "out"
      and len(V.out3_src.data["xs3"]) == 1
      and len(V.mem3_src.data["xs3"]) == len(_mem),
      "marks=%s out-halos=%d member-halos=%d"
      % (V.state["marks"], len(V.out3_src.data["xs3"]),
         len(V.mem3_src.data["xs3"])))
check("  ... and the shower table row is selected again",
      list(V.shower_src.selected.indices) != [])

# ---- round 4: the neutrino-candidate cloud filter --------------------------
V.on_event(None, None, "evt84229")
_cl_on = V.state["cloud"]
check("the cloud defaults to the neutrino candidate, not the whole readout",
      V.cloud_scope.active == 0 and _cl_on["filtered"]
      and _cl_on["candidate"] < _cl_on["total"],
      "%d of %d points, %d of %d clusters (ids %s)"
      % (_cl_on["candidate"], _cl_on["total"], _cl_on["ncluster_kept"],
         _cl_on["ncluster"], _cl_on["kept_ids"]))
check("  ... the readout names all three numbers, not two",
      "of %s clusters" % _cl_on["ncluster"] in V.cloud_div.text
      and "{:,}".format(_cl_on["total"]) in V.cloud_div.text)
V.cloud_scope.active = 1
V.on_cloud_opt(None, None, 1)
_cl_off = V.state["cloud"]
check("  ... 'all clusters' really does put the cosmics back",
      not _cl_off["filtered"] and _cl_off["candidate"] == _cl_off["total"]
      and _cl_off["total"] == _cl_on["total"],
      "%d points" % _cl_off["candidate"])
V.cloud_scope.active = 0
V.on_cloud_opt(None, None, 0)
# The filter must run BEFORE decimation, or the budget eats the candidate twice.
V.cloud_max.value = "10000"
V.on_cloud_opt(None, None, None)
_cl_small = V.state["cloud"]
check("  ... and the budget is spent on the CANDIDATE, not on the whole cloud",
      _cl_small["kept"] == min(10000, _cl_small["candidate"]),
      "kept %d of candidate %d (total %d)"
      % (_cl_small["kept"], _cl_small["candidate"], _cl_small["total"]))
V.cloud_max.value = "25000"
V.on_cloud_opt(None, None, None)
V.tap_action.value = V.TAP_SELECT

# ---- round 4: framing on the shower rather than on the whole TPC -----------
# EM mode explicitly: focus_points() follows the MODE (the selected shower in EM,
# the two assigned gammas in pi0), and an earlier check left the app in pi0.
V.mode_group.active = 0
V.on_event(None, None, "evt64591")
V.fit_mode.active = 0
V.refit_camera()
_R_all = V.state["cam_R"]
V.fit_mode.active = 2
V.on_shower_select(None, None, [0])          # the biggest shower, row 0
_R_shw = V.state["cam_R"]
check("'frame the shower' fills the panel with the thing being judged",
      _R_shw < 0.5 * _R_all,
      "R %.0f cm over all reco -> %.0f cm over the shower" % (_R_all, _R_shw))
check("  ... and it is NOT the default: a table click must not move the camera "
      "under the scanner",
      V.fit_mode.labels[0] == "frame the reco")
V.state["sel_shower"] = None
V.refit_camera()
check("  ... with nothing selected it falls back to the reco, not to nothing",
      abs(V.state["cam_R"] - _R_all) < 1e-9, "R=%.0f" % V.state["cam_R"])
V.fit_mode.active = 0

# ---- round 4: the filter, over the WHOLE sample ----------------------------
# Two things that a single-event check cannot say, and that the doc quotes:
#   (a) the numpy grid hash and scipy's cKDTree return the SAME kept set.  The
#       grid is the fallback when scipy is absent, and a fallback that quietly
#       disagreed would change what is on screen depending on the box.
#   (b) the filter does not eat the charge of the thing being scanned -- the
#       largest shower's fitted points stay covered by a KEPT cluster.
# Slow (~40 s: it parses all 94 clouds) but this is the claim the round rests on.
import time as _time                                                # noqa: E402
try:
    from scipy.spatial import cKDTree as _KD                        # noqa: E402
except ImportError:
    _KD = None
_zips, _same, _cov, _covall, _frac = {}, 0, [], [], []
_t0 = _time.time()
_rows = [l.rstrip("\n").split("\t")
         for l in open(os.path.join(SX, "em_display", "em114-manifest.tsv"))]
_MAN = [dict(zip(_rows[0], r)) for r in _rows[1:] if len(r) == len(_rows[0])]
for _r in _MAN:
    _d = json.load(open(os.path.join(SX, _r["dump"])))
    _zp = D3.bee_zip_path(SX, _r)
    _zi = D3.bee_event_index(SX, _r, _r["event"])
    import zipfile as _zf                                           # noqa: E402
    _z = _zips.setdefault(_zp, _zf.ZipFile(_zp))
    _c = json.loads(_z.read("data/%d/%d-clustering-global.json" % (_zi, _zi)))
    _cid = _c.get("real_cluster_id") or _c["cluster_id"]
    _pts = [(p["x"], p["y"], p["z"]) for s in (_d.get("segments") or [])
            for p in (s.get("points") or []) if p.get("x") is not None]
    _ts = _d.get("track_shower") or {}
    _pts += [(_ts["x"][i], _ts["y"][i], _ts["z"][i])
             for i in range(len(_ts.get("x") or []))]
    _keep, _ = D3.candidate_clusters(_c["x"], _c["y"], _c["z"], _cid, _pts)
    _kg, _ = D3.candidate_clusters(_c["x"], _c["y"], _c["z"], _cid, _pts,
                                   force_grid=True)
    _same += (_keep == _kg)
    _frac.append(sum(1 for q in _cid if q in _keep) / max(1, len(_cid)))
    # Coverage of the largest shower's own fitted points.  NOTE the join key:
    # segments carry the shower's `id`, not its `shower_id` field (`shower_id` on
    # a segment is 78025 while the shower's own `shower_id` is 4).  Joining on the
    # wrong one silently yields an empty member list and a vacuous check.
    _sh = max((_d.get("showers") or []), key=lambda s: s.get("kine_charge") or 0,
              default=None)
    if _sh is not None and _KD is not None:
        _seg = [s for s in (_d.get("segments") or [])
                if s.get("shower_id") == _sh.get("id")]
        _q = [(p["x"], p["y"], p["z"]) for s in _seg for p in (s.get("points") or [])
              if p.get("x") is not None]
        import numpy as _np                                         # noqa: E402
        _tree = _KD(_np.column_stack([_c["x"], _c["y"], _c["z"]]))
        _cida = _np.asarray(_cid)
        if _q:
            _dd, _ii = _tree.query(_np.asarray(_q), distance_upper_bound=2.0)
            _ok = _np.isfinite(_dd)
            if _ok.any():
                _cov.append(float(_np.isin(_cida[_ii[_ok]], list(_keep)).mean()))
        # and of EVERY reco point, which is the stronger statement
        _dd, _ii = _tree.query(_np.asarray(_pts), distance_upper_bound=2.0)
        _ok = _np.isfinite(_dd)
        _covall.append(float(_np.isin(_cida[_ii[_ok]], list(_keep)).mean())
                       if _ok.any() else 1.0)
check("scipy and the numpy fallback agree on the kept clusters, 94/94",
      _same == len(_MAN), "%d/%d agree (%.0f s)" % (_same, len(_MAN),
                                                    _time.time() - _t0))
_frac.sort()
check("  ... and the candidate is a minority of the cloud on every event",
      max(_frac) < 0.75,
      "kept median %.3f p90 %.3f max %.3f" % (_frac[len(_frac) // 2],
                                              _frac[int(0.9 * len(_frac))],
                                              max(_frac)))
check("  ... while the largest shower's own charge stays covered",
      bool(_cov) and min(_cov) > 0.95,
      ("largest shower: min %.4f median %.4f over %d events"
       % (min(_cov), sorted(_cov)[len(_cov) // 2], len(_cov))) if _cov
      else "NOT MEASURED (no scipy) -- the claim in the doc is unchecked here")
check("  ... and so does every reco point of the candidate, not just that shower",
      bool(_covall) and min(_covall) > 0.98,
      ("all reco: min %.4f median %.4f"
       % (min(_covall), sorted(_covall)[len(_covall) // 2])) if _covall
      else "NOT MEASURED (no scipy)")
for _z in _zips.values():
    _z.close()

# A stale selection would make the next "mark IN" hit a segment of the PREVIOUS
# event: replacing a CDS's .data does not clear its .selected.
V.pick_src.selected.indices = [3, 4, 5]
V.on_event(None, None, "evt21073")
check("switching events clears the 3-D selection",
      list(V.pick_src.selected.indices) == [] and V.selected_cand_ids() == [],
      str(list(V.pick_src.selected.indices)))
V.on_event(None, None, "evt84229")

# ---- camera round trip -----------------------------------------------------
V.on_camtxt(None, None, "1.2345,-0.5000")
check("the browser's panend report sets the server-side camera",
      abs(V.state["cam"][0] - 1.2345) < 1e-9
      and abs(V.state["cam"][1] + 0.5) < 1e-9)
check("  ... and the readout shows it", "camera az" in V.cam_div.text)
V.on_camtxt(None, None, "not-a-camera")
check("  ... a malformed report is ignored, not crashed on",
      abs(V.state["cam"][0] - 1.2345) < 1e-9)
V.on_save()
_rec = json.load(open(V.label_path("evt84229")))
check("the saved record carries the view the judgement was made from",
      (_rec.get("camera") or {}).get("cloud") == "clustering-global"
      and _rec["camera"]["az_deg"] is not None, str(_rec.get("camera")))

print()
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)

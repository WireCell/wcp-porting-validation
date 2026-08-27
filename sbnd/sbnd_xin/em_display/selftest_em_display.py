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
V.pick_mode.active = 0
_sids = V.pick_src.data["sid"]
_idx = [i for i, s in enumerate(_sids) if s == _sids[0]]
V.pick_src.selected.indices = _idx + [len(_sids) - 1]
check("a 3-D box over many points resolves to a handful of segments",
      len(V.selected_cand_ids()) <= 2 and len(_idx) > 2,
      "%d points -> %d segment(s)" % (len(_idx) + 1, len(V.selected_cand_ids())))
V.state["sel_shower"] = 69134
V.fill_cand_table()
V.mark("in")()
check("  ... and marking works off that 3-D selection",
      V.state["marks"].get(_sids[0]) == "in")
V.pick_src.selected.indices = []

# tap in "fill x/y/z" mode must land on a REAL fitted point (a ray needs an
# anchor; this is the 3-D answer to the two-panel tap).
V.pick_mode.active = 1
V.pick_src.selected.indices = [5]
_want = (V.pick_src.data["x"][5], V.pick_src.data["y"][5], V.pick_src.data["z"][5])
check("tap in fill mode writes a real fitted point into x/y/z",
      abs(float(V.man_x.value) - _want[0]) < 0.06
      and abs(float(V.man_y.value) - _want[1]) < 0.06
      and abs(float(V.man_z.value) - _want[2]) < 0.06,
      "%s %s %s" % (V.man_x.value, V.man_y.value, V.man_z.value))
V.pick_mode.active = 0

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
